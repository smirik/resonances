import numpy as np
import pandas as pd
from pathlib import Path

from .config import SimulationConfig
from resonances.body import Body
from resonances.logger import logger
from resonances.secular.secular_resonance import SecularResonance
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance, LidovKozaiParameters
from resonances.plotting import Plotter, PhasePlotter
from .serializer import SimulationSerializer


class DataManager:
    """Manages data saving and export functionality."""

    STATUS_FOLDERS = {
        -5: 'probably_near_separatrix',
        -4: 'slow-circulation',
        -3: 'near_separatrix',
        2: 'resonant',
        1: 'transient',
        0: 'non-resonant',
        -1: 'controversial-transient',
        -2: 'controversial-libration',
        -9: 'uncertain',
        -99: 'chaotic',
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
        """Check if body MMR should be plotted."""
        return self._process_status(body.statuses.get(resonance.to_s(), 0), self.config.plot)

    def _process_status(self, status: int, mode) -> bool:
        """Process status against mode to determine if action should be taken."""
        if (mode is None) or (mode is False) or (isinstance(mode, str) and mode.lower() == 'none'):
            return False
        if mode == 'all':
            return True
        if mode == 'resonant' and status > 0:
            return True
        if mode == 'nonzero' and status != 0:
            return True
        if mode == 'candidates' and status < 0:
            return True
        return False

    def ensure_save_path_exists(self):
        """Ensure save and plot paths exist."""
        Path(self.config.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.plot_path).mkdir(parents=True, exist_ok=True)

    def save_data(self, bodies, times, simulation=None):
        """Save simulation data and plots."""
        if self.config.save_summary:
            self.save_simulation_summary(bodies)

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
        save_mode = self.config.save
        if (save_mode is None) or (save_mode is False) or (isinstance(save_mode, str) and save_mode.lower() == 'none'):
            return

        body_data = body.keplerian_elements_to_dict()
        body_data["times"] = times / (2 * np.pi)
        for resonance in body.resonances():
            if self.should_save_body(body, resonance):
                body_data.update(body.resonance_to_dict(resonance))

        if body_data is not None:
            df = pd.DataFrame(data=body_data)
            df.to_csv(f'{self.config.save_path}/data-{body.name}.csv')

        self._save_periodogram_data(body)

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

        for resonance in body.resonances():
            if self.should_plot_body(body, resonance):
                plot_path = self._get_plot_path(body, resonance)

                # Evolution plot (original behavior)
                if 'evolution' in plots_to_generate:
                    self._plot_evolution(body, resonance, simulation, plot_path)

                # Phase portrait plot
                if 'phase_portrait' in plots_to_generate:
                    self._plot_phase_portrait(body, resonance, simulation, plot_path)

    def _plot_evolution(self, body: Body, resonance, simulation, plot_path: str):
        """Plot evolution (resonant angle, semi-major axis, etc.)."""
        config = self.config.plot_config if self.config.plot_config is not None else 'full'
        plot_filename = f'{plot_path}/{body.name}-{resonance.to_s()}.{self.config.image_type}'
        plotter = Plotter.from_body(body, resonance, simulation).configure(config).plot()
        if self.config.plot_type in ["show", "both"]:
            plotter.show()
        if self.config.plot_type in ["save", "both"]:
            plotter.save(plot_filename)
        plotter.close()

    def _plot_phase_portrait(self, body: Body, resonance, simulation, plot_path: str):
        """Plot all phase portrait variants (filtered, unfiltered, slow points)."""
        plotter = PhasePlotter.from_body(body, resonance, simulation)
        res_key = resonance.to_s()
        img_type = self.config.image_type

        # Filtered phase portrait
        plotter.plot_phase_portrait_filtered()
        if self.config.plot_type in ["show", "both"]:
            plotter.show()
        if self.config.plot_type in ["save", "both"]:
            plotter.save(f'{plot_path}/{body.name}-{res_key}-filtered.{img_type}')

        # Unfiltered phase portrait
        plotter.plot_phase_portrait_unfiltered()
        if self.config.plot_type in ["show", "both"]:
            plotter.show()
        if self.config.plot_type in ["save", "both"]:
            plotter.save(f'{plot_path}/{body.name}-{res_key}-unfiltered.{img_type}')

        # Slow points phase portrait
        percentile = getattr(self.config, 'phase_portrait_slow_percentile', 95)
        plotter.plot_phase_portrait_slow(percentile=percentile)
        if self.config.plot_type in ["show", "both"]:
            plotter.show()
        if self.config.plot_type in ["save", "both"]:
            plotter.save(f'{plot_path}/{body.name}-{res_key}-percentile{int(percentile)}.{img_type}')

        plotter.close()

    def _get_plot_path(self, body: Body, resonance) -> str:
        """
        Get the plot path, optionally with subfolder based on strategy.

        If plot_subfolder_strategy is 'status', creates subfolders:
        - 'resonant' for status=2
        - 'transient' for status=1
        - 'non-resonant' for status=0
        - 'controversial-transient' for status=-1
        - 'controversial-libration' for status=-2
        - 'chaotic' for status=-3 (integration failure, e > 1.1)
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

        summary_file = Path(summary_filename)
        segments_file = Path(segments_filename)
        if summary_file.exists():
            df.to_csv(summary_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(summary_filename, mode='a', header=True, index=False)

        if segments_file.exists():
            df_segments.to_csv(segments_filename, mode='a', header=False, index=False)
        else:
            df_segments.to_csv(segments_filename, mode='a', header=True, index=False)

        return df, df_segments

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
        # Extract comments to place it last; strip line breaks
        comments = flat.pop('comments', None)
        if comments is not None:
            comments = str(comments).replace('\n', ' ').replace('\r', ' ')

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
        }
        for k, v in flat.items():
            row[k] = v
            if k == 'metrics_trend_to_oscillation':
                row.update(segment_count_columns)
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
                'comments': comments,
            }
        )
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
