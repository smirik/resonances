import fcntl
import numpy as np
import pandas as pd
from pathlib import Path

from .config import SimulationConfig, SavePlotMode, MMR_PLOT_KINDS
from resonances.body import Body
from resonances.logger import logger
from resonances.secular.secular_resonance import SecularResonance
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance, LidovKozaiParameters
from resonances.mmr.mmr import MMR
from resonances.plotting import Plotter, EccentricityVectorPlotter, CrossSpectrumPlotter, FreeOmegaPlotter, MMRPlotter
from resonances.resonance import coherence_analysis, omega_free_gate
from resonances.secular import free_elements
from resonances.resonance.classify.models import ResonanceStatus
from .serializer import SimulationSerializer

_S = ResonanceStatus


class DataManager:
    """Manages data saving and export functionality."""

    STATUS_FOLDERS = {
        _S.PROBABLY_NEAR_SEPARATRIX: 'probably_near_separatrix',
        _S.PROBABLY_SLOW_CIRCULATION: 'slow-circulation',
        _S.NEAR_SEPARATRIX: 'near_separatrix',
        _S.LIBRATION: 'resonant',
        _S.TRANSIENT: 'transient',
        _S.NON_RESONANT: 'non-resonant',
        _S.TRANSIENT_UNCERTAIN: 'controversial-transient',
        _S.LIBRATION_UNCERTAIN: 'controversial-libration',
        _S.UNCERTAIN: 'uncertain',
        _S.CHAOTIC: 'chaotic',
    }

    # Fields extracted per segment for the segments summary CSV
    SEGMENT_FIELDS = ('revolutions_true', 'trend_to_oscillation', 'amplitude', 'sign_dominance', 'mean_sigma_dot')

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.skip_simulation_json = False  # Set to True for batch workers

    def should_save_body(self, body: Body, resonance):
        """Check if body MMR data should be saved."""
        return self._process_status(body.statuses.get(resonance.to_s(), 0), self.config.save)

    def should_plot_body(self, body: Body, resonance):
        """Check if the figures of this body-resonance pair should be drawn.

        The `plot` mode decides by status; resonance types listed in `config.plot_always`
        are drawn whatever their status, as long as plotting is on at all.
        """
        if self.config.plot is not None and getattr(resonance, 'type', None) in getattr(self.config, 'plot_always', []):
            return True
        return self._process_status(body.statuses.get(resonance.to_s(), 0), self.config.plot)

    @staticmethod
    def _process_status(status: int, mode) -> bool:
        """Process status against mode to determine if action should be taken."""
        if mode is None:
            return False
        if mode == SavePlotMode.ALL:
            return True
        if mode == SavePlotMode.RESONANT and status > 0:
            return True
        if mode == SavePlotMode.CANDIDATES and status in (2, 1, -1, -2, -3, -4):
            return True
        if mode == SavePlotMode.EXTENDED and (status > 0 or status in (-3, -4, -5, -9)):
            return True
        if mode == SavePlotMode.NONZERO and status != 0:
            return True
        if mode == SavePlotMode.NEGATIVE and status < 0:
            return True
        return False

    def ensure_save_path_exists(self):
        """Ensure save and plot paths exist."""
        Path(self.config.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.plot_path).mkdir(parents=True, exist_ok=True)

    def save_data(self, bodies, times, simulation=None):
        """Save simulation data and plots."""
        self.ensure_save_path_exists()

        if self.config.save_summary:
            self.save_simulation_summary(bodies)
            if self.config.coherence_enabled:
                self.save_coherence_summary(bodies)

        if simulation and self.config.save_planets:
            self.save_planets(times, simulation.integration_engine.planets_data)

        for body in bodies:
            self.save_body(body, times)
        if simulation:
            simulation.running_time["bodies_saved"] = logger.get_current_time()
        for body in bodies:
            self.plot_body(body, simulation)
        if simulation:
            simulation.running_time["stop"] = logger.get_current_time()

        if not self.skip_simulation_json:
            self.save_configuration_details(bodies, simulation)

    def save_body(self, body: Body, times):
        """Save all resonance data for a body."""

        self.ensure_save_path_exists()
        if self.config.save is None:
            return

        if not any(self.should_save_body(body, r) for r in body.resonances()):
            return

        body_data = body.keplerian_elements_to_dict()
        body_data["times"] = times / (2 * np.pi)
        for resonance in body.resonances():
            if self.should_save_body(body, resonance):
                body_data.update(body.resonance_to_dict(resonance))

        df = pd.DataFrame(data=body_data)
        df.to_csv(f'{self.config.save_path}/data-{body.name}.csv')

        self._save_periodogram_data(body)
        if body.lidov_kozai_resonances:
            self.save_free_elements_series(body)

    def _save_periodogram_data(self, body: Body):
        """Save periodogram data for a resonance."""
        # Save resonant angle periodogram
        df_data = {}

        if body.axis_periodogram_frequency is not None:
            freq = body.axis_periodogram_frequency
            power = body.axis_periodogram_power
            df_data = {'a_frequency': freq, 'a_power': power, 'a_period': 1.0 / freq}

        for resonance in body.resonances():
            if self.should_save_body(body, resonance):
                resonance_key = resonance.to_s()
                if body.periodogram_frequency.get(resonance_key) is not None:
                    freq = body.periodogram_frequency[resonance_key]
                    power = body.periodogram_power[resonance_key]

                    df_data[resonance_key + '_frequency'] = freq
                    df_data[resonance_key + '_power'] = power
                    df_data[resonance_key + '_period'] = 1.0 / freq

        df = pd.DataFrame(df_data)
        df.to_csv(f'{self.config.save_path}/data-{body.name}-periodograms.csv', index=False)

    def plot_body(self, body: Body, simulation=None):
        """Plot data for a body based on configured plot types."""
        plots_to_generate = getattr(self.config, 'plots', ['evolution'])
        plotted_resonances = [r for r in body.resonances() if self.should_plot_body(body, r)]

        for resonance in plotted_resonances:
            plot_path = self._get_plot_path(body, resonance)

            # Evolution plot (original behavior)
            if 'evolution' in plots_to_generate:
                self._plot_evolution(body, resonance, simulation, plot_path)

            # MMR diagnostics: combined figure and its panels as standalone figures
            mmr_kinds = [kind for kind in MMR_PLOT_KINDS if kind in plots_to_generate]
            if mmr_kinds and isinstance(resonance, MMR):
                self._plot_mmr(body, resonance, simulation, plot_path, mmr_kinds)

            # Cross spectra of the configured pairs
            if 'cross_spectrum' in plots_to_generate:
                self._plot_cross_spectrum(body, resonance, plot_path)

            # Free argument of pericentre — only meaningful for the Lidov-Kozai angle
            if 'free_omega' in plots_to_generate and isinstance(resonance, LidovKozaiResonance):
                self._plot_free_omega(body, resonance, plot_path)

        # The eccentricity vector depends on the body alone, not on any resonance, so it
        # is drawn once rather than repeated identically for every resonance of the body.
        if plotted_resonances and 'ecc_vector' in plots_to_generate:
            self._plot_ecc_vector(body, simulation, self._get_plot_path(body, plotted_resonances[0]))

    def _show_or_save(self, plotter, filename):
        """Show and/or save a plot depending on config, then close."""
        if self.config.plot_type in ["show", "both"]:
            plotter.show()
        if self.config.plot_type in ["save", "both"]:
            plotter.save(filename)

    def _plot_evolution(self, body: Body, resonance, simulation, plot_path: str):
        """Plot evolution (resonant angle, semi-major axis, etc.)."""
        config = self.config.plot_config if self.config.plot_config is not None else 'full'
        plot_filename = f'{plot_path}/{body.name}-{resonance.to_s()}.{self.config.image_type}'
        plotter = Plotter.from_body(body, resonance, simulation).configure(config).plot()
        self._show_or_save(plotter, plot_filename)
        plotter.close()

    def _plot_mmr(self, body: Body, resonance, simulation, plot_path: str, kinds):
        """MMR diagnostic figures; the oscillation quantities are computed once for all kinds."""
        plotter = MMRPlotter.from_body(body, resonance, simulation, options=self.config.plot_options)
        stem = f'{plot_path}/{body.name}-{resonance.to_s()}'
        for kind in kinds:
            if plotter.plot(kind):
                self._show_or_save(plotter, f'{stem}-{kind}.{self.config.image_type}')
                plotter.close()

    def _plot_ecc_vector(self, body: Body, simulation, plot_path: str):
        """Plot the non-singular eccentricity vector k = e*cos(omega), h = e*sin(omega)."""
        plotter = EccentricityVectorPlotter.from_body(body, sim=simulation)
        plotter.plot()
        self._show_or_save(plotter, f'{plot_path}/{body.name}-ecc-vector.{self.config.image_type}')
        plotter.close()

    def _plot_free_omega(self, body: Body, resonance, plot_path: str):
        """Plot the free argument of pericentre: drift, Kozai portrait, and the free vector."""
        if body.free_elements is None:
            return
        plotter = FreeOmegaPlotter.from_body(body, resonance)
        res_key = resonance.to_s()
        img_type = self.config.image_type

        plotter.plot_drift()
        self._show_or_save(plotter, f'{plot_path}/{body.name}-{res_key}-free-omega-drift.{img_type}')

        plotter.plot_portrait()
        self._show_or_save(plotter, f'{plot_path}/{body.name}-{res_key}-free-omega-portrait.{img_type}')

        # Same window as the osculating portrait of this body, so the two rings can be
        # compared by eye — which is what tells a real free circle from a forced offset.
        osculating = EccentricityVectorPlotter.from_body(body, resonance)
        plotter.plot_vector(limit=osculating.extent() if body.has_eccentricity_vector() else None)
        self._show_or_save(plotter, f'{plot_path}/{body.name}-{res_key}-free-omega-vector.{img_type}')

        plotter.close()

    def _plot_cross_spectrum(self, body: Body, resonance, plot_path: str):
        """Plot one cross spectrum per configured pair of series."""
        analyses = body.coherence.get(resonance.to_s())
        if not analyses:
            return

        res_key = resonance.to_s()
        for pair in analyses:
            plotter = CrossSpectrumPlotter.from_body(body, resonance, pair)
            plotter.plot()
            self._show_or_save(plotter, f'{plot_path}/{body.name}-{res_key}-coherence-{pair}.{self.config.image_type}')
            plotter.close()

    def _get_plot_path(self, body: Body, resonance) -> str:
        """
        Get the plot path, optionally with subfolder based on strategy.

        If plot_subfolder_strategy is 'status', creates subfolders based on
        ResonanceStatus enum values (see STATUS_FOLDERS mapping).
        """
        base_path = self.config.plot_path

        if self.config.plot_subfolder_strategy == 'status':
            status = body.statuses.get(resonance.to_s(), 0)
            subfolder = self.STATUS_FOLDERS.get(status, 'non-resonant')
            plot_path = f'{base_path}/{subfolder}'
            Path(plot_path).mkdir(parents=True, exist_ok=True)
            return plot_path

        return base_path

    def save_planets(self, times, planets_data):
        """Save planetary data."""
        self.ensure_save_path_exists()

        for planet, data in planets_data.items():
            df = pd.DataFrame(data=data)
            df.to_csv(f'{self.config.save_path}/data-planet-{planet}.csv', index=False)

    def save_simulation_summary(self, bodies):
        """Save simulation summary."""
        self.ensure_save_path_exists()

        df, df_segments = self.get_simulation_summary(bodies)
        summary_filename = f'{self.config.save_path}/summary.csv'
        segments_filename = f'{self.config.save_path}/segments.csv'

        self._append_csv(df, summary_filename)
        self._append_csv(df_segments, segments_filename)

        return df, df_segments

    def get_coherence_summary(self, bodies) -> pd.DataFrame:
        """Build the cross-spectral line table, one row per detected coherent line."""
        rows = []
        for body in bodies:
            for resonance in body.resonances():
                analyses = body.coherence.get(resonance.to_s())
                if analyses:
                    rows.extend(coherence_analysis.coherence_rows(body.name, resonance.to_s(), analyses))
        return pd.DataFrame(rows)

    def save_coherence_summary(self, bodies):
        """Append the cross-spectral line table to coherence.csv."""
        self.ensure_save_path_exists()
        df = self.get_coherence_summary(bodies)
        if not df.empty:
            self._append_csv(df, f'{self.config.save_path}/coherence.csv')
        return df

    def save_free_elements_series(self, body: Body):
        """Write one body's forced/free series to `{body}-omega-free.csv`.

        Its own file rather than extra columns on data-{body}.csv, because the split runs
        on a decimated grid and would not line up with the integration one.
        """
        columns = free_elements.series_rows(body.free_elements)
        if columns is None:
            return None
        self.ensure_save_path_exists()
        df = pd.DataFrame(columns)
        df.to_csv(f'{self.config.save_path}/{body.name}-omega-free.csv', index=False)
        return df

    @staticmethod
    def _append_csv(df, filename):
        """Append a DataFrame atomically across batch worker processes.

        Workers share the summary paths.  The separate lock file keeps the header check and
        the complete pandas write in one critical section, preventing duplicate headers and
        interleaved CSV records when two batches finish together.
        """
        lock_filename = f'{filename}.lock'
        with open(lock_filename, 'a') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                write_header = not Path(filename).exists()
                df.to_csv(filename, mode='a', header=write_header, index=False)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def get_simulation_summary(self, bodies):
        """Generate simulation summary dataframe."""
        rows = []
        segments = []
        for body in bodies:
            for resonance in body.resonances():
                try:
                    rows.append(self._build_summary_row(body, resonance))
                    seg_entry = self._build_segments_entry(body, resonance)
                    if seg_entry is not None:
                        segments.append(seg_entry)
                except Exception as e:
                    logger.error(f"Error getting resonance summary for {body.name}: {e}")

        return pd.DataFrame(rows), pd.DataFrame(segments)

    def _build_summary_row(self, body: Body, resonance) -> dict:
        """Build a single summary row for one body+resonance pair."""
        # Resonance type
        if isinstance(resonance, SecularResonance):
            res_type = 'Secular'
        elif isinstance(resonance, LidovKozaiResonance):
            res_type = 'Lidov-Kozai'
        else:
            res_type = 'MMR'

        # Lidov-Kozai params
        if isinstance(resonance, LidovKozaiResonance):
            c1, c2, c = LidovKozaiParameters.evaluate(
                body.initial_data['e'],
                body.initial_data['inc'],
                body.initial_data['omega'],
            )
        else:
            c1, c2, c = None, None, None

        flat = body.librations[resonance.to_s()].to_flat_dict()
        # Extract fields to control column ordering
        comments = flat.pop('comments', None)
        if comments is not None:
            comments = str(comments).replace('\n', ' ').replace('\r', ' ')
        chaos_flag = flat.pop('chaos_flag', 0)
        chaos_comment = flat.pop('chaos_comment', '')

        # Extract segment counts and rename to user-friendly column names
        segment_count_columns = {
            'n_good_total': flat.pop('segment_counts_n_good_total', 0),
            'n_reasonable_total': flat.pop('segment_counts_n_reasonable_total', 0),
            'n_good_0.1': flat.pop('segment_counts_n_good_0_1', 0),
            'n_reasonable_0.1': flat.pop('segment_counts_n_reasonable_0_1', 0),
            'n_good_0.2': flat.pop('segment_counts_n_good_0_2', 0),
            'n_reasonable_0.2': flat.pop('segment_counts_n_reasonable_0_2', 0),
            'n_good_0.3': flat.pop('segment_counts_n_good_0_3', 0),
            'n_reasonable_0.3': flat.pop('segment_counts_n_reasonable_0_3', 0),
        }

        # Build row with segment counts after metrics_trend_to_oscillation,
        # and comments as the last column
        row = {
            'name': body.name,
            'resonance': resonance.to_s(),
            'type': res_type,
            'status': body.statuses.get(resonance.to_s(), 0),
            'chaos_flag': chaos_flag,
            'chaos_comment': chaos_comment,
        }
        # 'status' set above is overwritten here by flat['status'] from the classification
        # result. The two agree only because every post-classification status change (the
        # Lidov-Kozai coherence gate) updates both the result object and body.statuses.
        for k, v in flat.items():
            row[k] = v
            if k == 'metrics_trend_to_oscillation':
                row.update(segment_count_columns)
        e_forced, e_free = body.forced_free_eccentricity()
        row.update(
            {
                'a': body.initial_data['a'],
                'e': body.initial_data['e'],
                'inc': body.initial_data['inc'],
                'Omega': body.initial_data['Omega'],
                'omega': body.initial_data['omega'],
                'M': body.initial_data['M'],
                'c1': c1,
                'c2': c2,
                'c': c,
                'e_forced': e_forced,
                'e_free': e_free,
            }
        )
        row.update(coherence_analysis.summary_fields(body.coherence.get(resonance.to_s()), body.zlk_gates.get(resonance.to_s())))
        # Free-element columns are written for Lidov-Kozai rows only: omega = varpi - Omega
        # is the angle they describe, and an MMR row would carry them without using them.
        if isinstance(resonance, LidovKozaiResonance):
            row.update(free_elements.summary_fields(body.free_elements))
            row.update(omega_free_gate.summary_fields(body.free_gates.get(resonance.to_s())))
        row['comments'] = comments
        return row

    def _build_segments_entry(self, body: Body, resonance) -> dict | None:
        """Build a segments entry for one body+resonance pair, or None if no segments."""
        res_key = resonance.to_s()
        if body.libration_segments.get(res_key) is None:
            return None

        metrics = {}
        for window_key, window_segments in body.libration_segments[res_key].items():
            for segment_key, segment in window_segments.items():
                compound_key = f"{window_key}_{segment_key}"
                for fld in self.SEGMENT_FIELDS:
                    metrics[f"{compound_key}_{fld}"] = getattr(segment, fld)

        return {
            'body': body.name,
            'resonance': res_key,
            **metrics,
        }

    def save_configuration_details(self, bodies, simulation):
        """Save configuration details to file."""
        SimulationSerializer.save_simulation_json(self.config, bodies, simulation)
