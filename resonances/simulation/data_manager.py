import fcntl
import functools
import math
import os
import subprocess
from contextlib import contextmanager
import numpy as np
import pandas as pd
from pathlib import Path

from . import layout
from .config import SimulationConfig, status_selected
from resonances.body import Body
from resonances.logger import logger
from resonances.secular.secular_resonance import SecularResonance
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance, LidovKozaiParameters
from resonances.mmr.mmr import MMR
from resonances.plotting import Plotter, EccentricityVectorPlotter, CrossSpectrumPlotter, FreeOmegaPlotter, MMRPlotter, SecularPlotter
from resonances.plotting.base import BasePlotter
from resonances.resonance import coherence_analysis, omega_free_gate
from resonances.resonance.analysis.verdict import OLD_ANGLE_STATUS, Status, Verdict, reason_from_text
from resonances.secular import free_elements
from .mmr_adapter import series_from_body
from .serializer import SimulationSerializer


@functools.lru_cache(maxsize=1)
def package_commit() -> str:
    """The git commit of the package source (`-dirty` when resonances/ differs from it), else the installed version.

    Written to every summary.csv row (`package_commit`) so that a result can be traced to the code that made it."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10, check=True)
        dirty = subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', 'resonances'], capture_output=True, timeout=30)
        return commit.stdout.strip() + ('-dirty' if dirty.returncode == 1 else '')
    except (OSError, subprocess.SubprocessError):
        try:
            from importlib.metadata import version

            return f'resonances {version("resonances")}'
        except Exception:  # noqa: BLE001 - neither a repository nor an installed distribution
            return ''


def _same_number(left, right, rtol: float = 1e-6) -> bool:
    """Two CSV numbers agree: both empty/NaN, or equal to `rtol` (a float written and read back)."""

    def number(value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        value = float(value)
        return None if math.isnan(value) else value

    a, b = number(left), number(right)
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rtol, abs_tol=0.0)


def _refuse_integer_status(status) -> None:
    """Integer statuses are gone: one raises TypeError rather than being misread."""
    if isinstance(status, (int, np.integer)) and not isinstance(status, bool):
        raise TypeError(f'Statuses are strings ({", ".join(Status)}), got the integer {status}')


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    return str(value).strip()


class DataManager:
    """Writes the results of a run: summary.csv and its companion tables, the data files and the figures.

    Tables (SPEC §11): summary.csv (one row per body x resonance, string statuses), coherence.csv, episodes.csv
    (every run of turning points of every analysed angle, MMR, secular and ZLK), and for the MMR analysis
    mmr_numbers.csv (every number, one row per body x MMR), candidates.csv (subresonances and owners) and
    constants.csv (one per run), all at the top of save_path. The series CSVs go to `config.data_path`
    (save_path/data_folder), the figures to plot_path/plot_figures_folder and the combined figures to
    plot_path/plot_combined_folder, each split by its layout (`figure_folder`, simulation/layout.py).
    """

    # Diagnostic plotters by resonance type; each declares the plot kinds it draws. The combined MMR figure is the
    # verdict figure (plotting/mmr_combined.py), drawn from the analysis by `_plot_mmr_combined`; the secular and
    # ZLK `combined` kind is their one-page verdict figure (`secular_plots.draw_long_period_combined`).
    DIAGNOSTIC_PLOTTERS = ((MMR, MMRPlotter), (SecularResonance, SecularPlotter), (LidovKozaiResonance, SecularPlotter))

    # The tables written per body, and the column naming the body in each (rows are replaced by it)
    SUMMARY_KEYS = {
        'summary': 'name',
        'coherence': 'body',
        'mmr_numbers': 'name',
        'episodes': 'name',
        'candidates': 'name',
    }
    MMR_TABLES = ('mmr_numbers', 'episodes', 'candidates')

    # Initial elements of the body in summary.csv (column, Body.initial_data key)
    INITIAL_ELEMENTS = (('a0', 'a'), ('e0', 'e'), ('inc0', 'inc'), ('Omega0', 'Omega'), ('omega0', 'omega'), ('M0', 'M'))

    # What a figure re-drawn from a re-run analysis is checked against in summary.csv (SPEC §2.6)
    FIGURE_CHECK_COLUMNS = ('status', 'angle_status', 'confidence', 'flt_cutoff_yr')
    FIGURE_CHECK_NUMBERS = ('flt_cutoff_yr',)

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.skip_simulation_json = False  # Set to True for batch workers
        self.replace_rows = True  # False in batch workers: see save_simulation_summary

    # ------------------------------------------------------------------ what is saved and plotted

    def should_save_body(self, body: Body, resonance):
        """Check if body data of this resonance should be saved (by status string and libration origin)."""
        status, origin = self._status_and_origin(body, resonance.to_s())
        return self._process_status(status, self.config.save, origin)

    def should_plot_body(self, body: Body, resonance):
        """Check if the figures of this body-resonance pair should be drawn.

        The `plot` mode decides by status; resonance types listed in `config.plot_always`
        are drawn whatever their status, as long as plotting is on at all.
        """
        if self.config.plot is not None and getattr(resonance, 'type', None) in getattr(self.config, 'plot_always', []):
            return True
        status, origin = self._status_and_origin(body, resonance.to_s())
        return self._process_status(status, self.config.plot, origin)

    @staticmethod
    def _status_and_origin(body: Body, key: str):
        """The status string of a row and the libration origin of its verdict ('' without a verdict)."""
        verdict = getattr(body, 'librations', {}).get(key)
        origin = str(verdict.libration_origin) if isinstance(verdict, Verdict) else ''
        return body.statuses.get(key), origin

    @staticmethod
    def _process_status(status, mode, origin: str = '') -> bool:
        """Whether a row with this status string and libration origin is saved/plotted under `mode` (SPEC §11.5,
        `config.status_selected`)."""
        _refuse_integer_status(status)
        return status_selected(mode, status, origin)

    def ensure_save_path_exists(self):
        """Ensure save and plot paths exist."""
        Path(self.config.save_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.plot_path).mkdir(parents=True, exist_ok=True)

    def data_file(self, filename: str) -> str:
        """The path of a series CSV in the data folder (`config.data_path`), created when missing."""
        folder = Path(self.config.data_path)
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder / filename)

    def save_data(self, bodies, times, simulation=None, analyse=None):
        """Write the run's files and figures one body at a time: `analyse(body)` first when given (a run: the body's
        verdicts and periodograms), then its data files and figures, then its analysis arrays are released
        (`release_arrays`), so only one body's analyses are in memory at a time. summary.csv and the tables come
        last, from the rows the release froze (the same rows)."""
        self.ensure_save_path_exists()
        if simulation and self.config.save_planets:
            self.save_planets(times, simulation.integration_engine.planets_data)

        for body in bodies:
            if analyse is not None:
                analyse(body)
            self.save_body(body, times)
            self.plot_body(body, simulation)
            self.release_arrays(body)
        if simulation:
            simulation.running_time["bodies_saved"] = logger.get_current_time()

        if self.config.save_summary:
            self.save_simulation_summary(bodies)
            if self.config.coherence_enabled:
                self.save_coherence_summary(bodies)
        if simulation:
            simulation.running_time["stop"] = logger.get_current_time()

        if not self.skip_simulation_json:
            self.save_configuration_details(bodies, simulation)

    @staticmethod
    def release_arrays(body: Body):
        """Drop the arrays of the body's analyses once its files and figures are written (SPEC §2.4: 40 angles of
        2e5 rows hold ~1 GB). The scalars stay, so summary.csv can be written again; a later figure re-runs the
        analysis (`_figure_analysis`, `Simulation.long_period_analysis`)."""
        for analysis in getattr(body, 'mmr_analyses', {}).values():
            if not DataManager._arrays_dropped(analysis):
                analysis.drop_arrays()
        for analysis in getattr(body, 'secular_analyses', {}).values():
            if analysis.angle is not None:
                analysis.drop_arrays()

    @staticmethod
    def _arrays_dropped(analysis) -> bool:
        """`drop_arrays` ran: the preprocessed series is gone while the geometry (scalars) stayed.
        An analysis stopped at G0 has neither and draws as it is."""
        return analysis.prep is None and analysis.geometry is not None

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

        pd.DataFrame(body_data, copy=False).to_csv(self.data_file(f'data-{body.name}.csv'))  # no copy of the columns

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

        # No periodogram at all (e.g. the end cut ate a short series): no file, rather than an unreadable empty one
        if not df_data:
            return
        # The angle periodograms of an MMR are computed on the record's own cutoff and may differ in length
        df = pd.DataFrame({name: pd.Series(values) for name, values in df_data.items()})
        df.to_csv(self.data_file(f'data-{body.name}-periodograms.csv'), index=False)

    # ------------------------------------------------------------------ figures

    def plot_body(self, body: Body, simulation=None):
        """Plot data for a body based on configured plot types (nothing for a body that was never integrated)."""
        if body.times is None:
            return
        plots_to_generate = getattr(self.config, 'plots', ['evolution'])
        plotted_resonances = [r for r in body.resonances() if self.should_plot_body(body, r)]

        for resonance in plotted_resonances:
            plot_path = self.figure_folder(body, resonance)

            # Evolution plot (original behavior)
            if 'evolution' in plots_to_generate:
                self._plot_evolution(body, resonance, simulation, plot_path)

            # The verdict figure of an MMR row (catalogue or derived subresonance), one file per body x resonance
            if 'combined' in plots_to_generate and isinstance(resonance, MMR):
                self._plot_mmr_combined(body, resonance, simulation)

            # Diagnostic figures of the resonance type (MMR panels, secular): their kinds
            for resonance_type, plotter_class in self.DIAGNOSTIC_PLOTTERS:
                kinds = [kind for kind in plotter_class.KINDS if kind in plots_to_generate]
                if kinds and isinstance(resonance, resonance_type):
                    self._plot_diagnostics(plotter_class, body, resonance, simulation, plot_path, kinds)

            # Cross spectra of the configured pairs
            if 'cross_spectrum' in plots_to_generate:
                self._plot_cross_spectrum(body, resonance, plot_path)

            # Free argument of pericentre — only meaningful for the Lidov-Kozai angle
            if 'free_omega' in plots_to_generate and isinstance(resonance, LidovKozaiResonance):
                self._plot_free_omega(body, resonance, plot_path)

        # The eccentricity vector depends on the body alone, not on any resonance, so it
        # is drawn once rather than repeated identically for every resonance of the body.
        if plotted_resonances and 'ecc_vector' in plots_to_generate:
            self._plot_ecc_vector(body, simulation, self.figure_folder(body, plotted_resonances[0]))

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

    def _plot_diagnostics(self, plotter_class, body: Body, resonance, simulation, plot_path: str, kinds):
        """Diagnostic figures of one plotter class; its quantities are computed once for all kinds."""
        try:
            plotter = plotter_class.from_body(body, resonance, simulation, options=self.config.plot_options)
        except ValueError as exc:  # an MMR without sigma_f: the analysis stopped at G0 (unphysical orbit, failed unwrap)
            if not isinstance(resonance, MMR):
                raise
            logger.warning(f'{body.name} {resonance.to_s()}: panels {kinds} skipped ({exc})')
            return
        for kind in kinds:
            if plotter.plot(kind):
                # Combined figures get a folder of their own, to be browsed one after another.
                folder = self.figure_folder(body, resonance, combined=True) if kind == 'combined' else plot_path
                self._show_or_save(plotter, f'{folder}/{body.name}-{resonance.to_s()}-{kind}.{self.config.image_type}')
                plotter.close()

    def _plot_mmr_combined(self, body: Body, resonance, simulation):
        """The one-page verdict figure of an MMR row (plotting/mmr_combined.py), in the combined folder.

        Drawn from the analysis in memory; after a restore without recomputation (or once its arrays were
        released) the analysis is re-run on the saved series and compared with the summary row (SPEC §2.6)."""
        from resonances.plotting.mmr_combined import draw_combined

        key = resonance.to_s()
        found = self._figure_analysis(body, resonance, simulation)
        if found is None:
            return
        analysis, series, stamp = found
        summary = self._mmr_summary_row(body, resonance, analysis)
        constants = analysis.context.constants_rows() if analysis.context is not None else []
        rows = (summary, analysis.numbers_row(), analysis.episode_rows(), constants)
        plotter = BasePlotter()
        plotter._figure = draw_combined(analysis, series, stamp=stamp, rows=rows, options=self.config.plot_options)
        folder = self.figure_folder(body, resonance, combined=True)
        self._show_or_save(plotter, f'{folder}/{body.name}-{key}-combined.{self.config.image_type}')
        plotter.close()

    def _figure_analysis(self, body: Body, resonance, simulation):
        """(analysis, series, stamp) for the verdict figure of a row, or None when it cannot be drawn.

        `stamp` is set when the analysis had to be re-run and disagrees with the row it is checked against:
        the summary row in memory, else the row of summary.csv."""
        key = resonance.to_s()
        analysis = getattr(body, 'mmr_analyses', {}).get(key)
        if analysis is not None and not self._arrays_dropped(analysis):
            series = analysis.derived_series or self._mmr_series(body, resonance, simulation)
            return None if series is None else (analysis, series, None)

        reference = self._mmr_summary_row(body, resonance, analysis) if analysis is not None else self._saved_summary_row(body, key)
        series = self._mmr_series(body, resonance, simulation)
        if series is None:
            return None
        from resonances.resonance.analysis.mmr.analyse import analyse_mmr

        parent = getattr(body, 'derived_resonances', {}).get(key, '')
        derived = bool(parent)
        settings = simulation.config.analysis_settings if simulation is not None else self.config.analysis_settings
        again = analyse_mmr(series, settings, scan_subresonances=not derived, scan_owner=not derived, parent=parent)
        if derived and analysis is not None and analysis.multiplet_status is not None:
            again.multiplet_status = analysis.multiplet_status  # a derived row shares its parent's (depth 1)
        stamp = None
        if reference is None:
            logger.warning(f'{body.name} {key}: the figure is drawn from a re-run analysis; no summary row to check it against')
        else:
            differences = self.figure_differences(reference, self._mmr_summary_row(body, resonance, again))
            if differences:
                stamp = 'recomputed != summary: ' + ', '.join(differences)
                logger.warning(f'{body.name} {key}: the re-run analysis of the figure differs from summary.csv ({", ".join(differences)})')
        return again, series, stamp

    @classmethod
    def figure_differences(cls, reference: dict, recomputed: dict) -> list:
        """The checked columns (FIGURE_CHECK_COLUMNS) on which a re-run analysis differs from the summary row,
        as 'column summary -> recomputed'."""
        differences = []
        for column in cls.FIGURE_CHECK_COLUMNS:
            left, right = reference.get(column), recomputed.get(column)
            if column in cls.FIGURE_CHECK_NUMBERS:
                same = _same_number(left, right)
            else:
                same = _text(left) == _text(right)
            if not same:
                differences.append(f'{column} {_text(left) or "-"} -> {_text(right) or "-"}')
        return differences

    def _saved_summary_row(self, body: Body, key: str):
        """The row of (body, resonance) in summary.csv, as strings (angle_status and reason in today's spelling), or
        None."""
        path = Path(self.config.save_path) / 'summary.csv'
        if not path.exists() or path.stat().st_size == 0:
            return None
        with self._locked(str(path)):
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if 'name' not in frame.columns or 'resonance' not in frame.columns:
            return None
        rows = frame[(frame['name'] == str(body.name)) & (frame['resonance'] == key)]
        if rows.empty:
            return None
        row = rows.iloc[-1].to_dict()
        if row.get('angle_status') in OLD_ANGLE_STATUS:
            row['angle_status'] = str(OLD_ANGLE_STATUS[row['angle_status']])
        if 'reason' in row:  # dynamic_* of the geometry alone before 2026-10-26 (origin unknown) -> geometric_*
            row['reason'] = str(reason_from_text(row['reason'], row.get('libration_origin', '')))
        return row

    @staticmethod
    def _mmr_series(body: Body, resonance, simulation):
        """The `MMRSeries` the analysis of this row reads (`mmr_adapter`), or None with a warning."""
        try:
            return series_from_body(body, resonance, simulation)
        except Exception as exc:  # noqa: BLE001 - one row without saved series must not stop the others
            logger.warning(f'{body.name} {resonance.to_s()}: cannot build the MMR series for the figure ({exc}); figure skipped')
            return None

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

    def figure_folder(self, body: Body, resonance, combined: bool = False) -> str:
        """The folder of a figure of this body x resonance: plot_path/plot_figures_folder (`combined`:
        plot_path/plot_combined_folder) split by plot_layout (plot_combined_layout), e.g. `combined/high/resonant`
        for 'confidence/status' (simulation/layout.py). Saving a figure creates it (`BasePlotter.save`)."""
        folder = self.config.plot_combined_folder if combined else self.config.plot_figures_folder
        levels = layout.parse_layout(self.config.plot_combined_layout if combined else self.config.plot_layout)
        return str(layout.layout_path(self.config.plot_path, folder, levels, self.layout_row(body, resonance)))

    @staticmethod
    def layout_row(body: Body, resonance) -> dict:
        """The columns the layout reads (`layout.LEVEL_COLUMNS`), as summary.csv has them: the status string, the
        confidence of the verdict ('' without one), the resonance type and key, the body's name."""
        key = resonance.to_s()
        status = body.statuses.get(key)
        _refuse_integer_status(status)
        verdict = getattr(body, 'librations', {}).get(key)
        confidence = str(verdict.confidence) if isinstance(verdict, Verdict) else ''
        return {
            'status': status,
            'confidence': confidence,
            'resonance_type': getattr(resonance, 'type', ''),
            'resonance': key,
            'name': body.name,
        }

    @classmethod
    def status_folder(cls, status) -> str:
        """The folder of a status string under the 'status' level (a row without one: unclassified)."""
        _refuse_integer_status(status)
        if status is not None:
            status = Status(str(status))
        return layout.layout_parts(('status',), {'status': status})[0]

    def save_planets(self, times, planets_data):
        """Save planetary data.

        Every batch worker integrates the same planets and writes the same files: each is written to a
        temporary name and moved into place, so two workers never interleave in one file."""
        self.ensure_save_path_exists()

        for planet, data in planets_data.items():
            path = self.data_file(f'data-planet-{planet}.csv')
            pd.DataFrame(data, copy=False).to_csv(f'{path}.{os.getpid()}.tmp', index=False)
            os.replace(f'{path}.{os.getpid()}.tmp', path)

    # ------------------------------------------------------------------ summary.csv and the tables

    @staticmethod
    def _in_memory(body: Body, key: str) -> bool:
        """Whether this row's full result is in memory, so its rows may be (re)written: an MMR row with its
        analysis, a secular/ZLK row with its verdict and classifier numbers. A verdict rebuilt from summary.csv
        (restore without recompute) is not: rewriting its row would lose the numbers it does not carry."""
        verdict = getattr(body, 'librations', {}).get(key)
        if verdict is None:
            return False
        if key in getattr(body, 'mmr_analyses', {}):
            return True
        if isinstance(verdict, Verdict) and str(verdict.resonance_type) == 'mmr':
            return False
        numbers = getattr(body, 'classifier_numbers', None)
        return numbers is None or key in numbers

    @classmethod
    def _classified(cls, bodies):
        """What the bodies in memory can write again: the names of the bodies classified in every resonance,
        and the (body, resonance) pairs classified in the others. A pair not in memory (`_in_memory`: a restore
        without recompute_librations, or a resonance whose series was not saved) keeps its rows in the files."""
        names, pairs = [], set()
        for body in bodies:
            keys = [r.to_s() for r in body.resonances()]
            done = [k for k in keys if cls._in_memory(body, k)]
            if len(done) == len(keys):
                names.append(body.name)
            else:
                pairs.update((str(body.name), k) for k in done)
        return names, pairs

    def _replace_rows_of(self, bodies, files):
        names, pairs = self._classified(bodies)
        self.drop_rows(names, files=files)
        self.drop_rows([], files=files, pairs=pairs)

    def save_simulation_summary(self, bodies):
        """Write these bodies' rows to summary.csv and the analysis tables (mmr_numbers.csv, episodes.csv,
        candidates.csv, constants.csv), replacing any they already have.

        Only what is in memory is written and replaced (`_classified`); the other rows stay.
        With `replace_rows` off (batch workers, whose old rows the parent drops before dispatching:
        `BatchManager`) the rows are only appended, which spares a scan of the whole file per batch.
        Returns the summary frame.
        """
        self.ensure_save_path_exists()

        df = self.get_simulation_summary(bodies)
        tables = self.get_mmr_tables(bodies)
        if self.replace_rows:
            self._replace_rows_of(bodies, files=('summary',) + self.MMR_TABLES)
        self._append_csv(df, f'{self.config.save_path}/summary.csv')
        for name in self.MMR_TABLES:
            if not tables[name].empty:
                self._append_csv(tables[name], f'{self.config.save_path}/{name}.csv')
        if self.config.save_constants and not tables['constants'].empty:
            self.save_constants(tables['constants'])

        return df

    def drop_rows(self, names, files=tuple(SUMMARY_KEYS), pairs=None) -> int:
        """Remove the rows of the bodies `names`, and of the (body, resonance) `pairs`, from summary.csv,
        coherence.csv and the analysis tables.

        Called before these bodies are written again (a retried or resumed batch, a run repeated in
        the same folder), so each body and resonance keeps one row. A summary row whose `parent` is a dropped
        pair (a derived subresonance row of a reclassified angle) goes with it. Returns the number of rows removed.
        """
        keys = {str(name) for name in names}
        pairs = set(pairs or ())
        removed = 0
        for name in files:
            filename = f'{self.config.save_path}/{name}.csv'
            path = Path(filename)
            if not (keys or pairs) or not path.exists() or path.stat().st_size == 0:
                continue
            with self._locked(filename):
                existing = pd.read_csv(filename, dtype=str, keep_default_na=False)
                key_column = self.SUMMARY_KEYS[name]
                if key_column not in existing.columns:
                    continue
                stale = existing[key_column].isin(keys)
                for column in ('resonance', 'parent'):
                    if pairs and column in existing.columns:
                        rows = zip(existing[key_column], existing[column])
                        stale |= np.fromiter((row in pairs for row in rows), dtype=bool, count=len(existing))
                if stale.any():
                    existing[~stale].to_csv(f'{filename}.tmp', index=False)
                    os.replace(f'{filename}.tmp', filename)
                    removed += int(stale.sum())
        return removed

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
        """Write these bodies' cross-spectral lines to coherence.csv, replacing any they already have."""
        self.ensure_save_path_exists()
        df = self.get_coherence_summary(bodies)
        if self.replace_rows:
            self._replace_rows_of(bodies, files=('coherence',))
        if not df.empty:
            self._append_csv(df, f'{self.config.save_path}/coherence.csv')
        return df

    def save_free_elements_series(self, body: Body):
        """Write one body's forced/free series to `{body}-omega-free.csv` in the data folder.

        Its own file rather than extra columns on data-{body}.csv, because the split runs
        on a decimated grid and would not line up with the integration one.
        """
        columns = free_elements.series_rows(body.free_elements)
        if columns is None:
            return None
        self.ensure_save_path_exists()
        df = pd.DataFrame(columns)
        df.to_csv(self.data_file(f'{body.name}-omega-free.csv'), index=False)
        return df

    def save_constants(self, constants: pd.DataFrame):
        """constants.csv, one per run: the rows of planets (and planetary terms) not yet in the file are added,
        those already there are kept (every body of a run sees the same planets)."""
        filename = f'{self.config.save_path}/constants.csv'
        with self._locked(filename):
            path = Path(filename)
            if path.exists() and path.stat().st_size > 0:
                existing = pd.read_csv(filename, dtype=str, keep_default_na=False)
                new = constants[~constants['planet'].astype(str).isin(set(existing['planet']))]
                if new.empty:
                    return
                merged = pd.concat([existing, new.astype(object)], ignore_index=True)
                merged = merged.reindex(columns=list(existing.columns) + [c for c in new.columns if c not in existing.columns])
            else:
                merged = constants
            merged.to_csv(f'{filename}.tmp', index=False)
            os.replace(f'{filename}.tmp', filename)

    @staticmethod
    @contextmanager
    def _locked(filename):
        """Hold an exclusive lock on `filename` across batch worker processes (a separate `<filename>.lock`).

        The holder removes the lock file before it lets go, so a finished run leaves none. A process that was
        waiting on that file now holds a lock nobody else sees: it enters only when the path still names the file it
        locked, otherwise it opens the path again. The file is removed only by its holder, so two processes never
        both see their own file at the path.
        """
        lock_path = f'{filename}.lock'
        while True:
            with open(lock_path, 'a') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    current = os.path.samestat(os.fstat(lock_file.fileno()), os.stat(lock_path))
                except FileNotFoundError:
                    current = False
                if not current:
                    continue  # closing the file releases the stale lock
                try:
                    yield
                finally:
                    try:
                        os.unlink(lock_path)
                    except FileNotFoundError:
                        pass
                return

    @classmethod
    def _append_csv(cls, df, filename):
        """Append a DataFrame atomically across batch worker processes.

        Workers share the summary paths.  The lock keeps the header check and the complete
        pandas write in one critical section, preventing duplicate headers and interleaved
        CSV records when two batches finish together.

        Batches need not share columns (an MMR row and a secular row bring different numbers).
        Rows are aligned to the existing header (missing values left empty); new columns rewrite
        the file with the union.
        """
        with cls._locked(filename):
            path = Path(filename)
            if not path.exists() or path.stat().st_size == 0:
                if not df.empty:
                    df.to_csv(filename, mode='w', header=True, index=False)
                return
            header = list(pd.read_csv(filename, nrows=0).columns)
            new_columns = [c for c in df.columns if c not in header]
            if not new_columns:
                if not df.empty:
                    df.reindex(columns=header).to_csv(filename, mode='a', header=False, index=False)
                return
            existing = pd.read_csv(filename, dtype=str, keep_default_na=False)
            merged = pd.concat([existing, df.astype(object)], ignore_index=True)
            merged = merged.reindex(columns=header + new_columns)
            merged.to_csv(f'{filename}.tmp', index=False)
            os.replace(f'{filename}.tmp', filename)

    def get_simulation_summary(self, bodies) -> pd.DataFrame:
        """The summary rows of every (body, resonance) in memory (`_in_memory`), derived subresonance rows
        included (they are resonances of the body).

        After a default restore (`recompute_librations=False`) or a batched run there is none: the
        frame is empty and summary.csv holds the results.
        """
        rows = []
        unclassified = 0
        for body in bodies:
            for resonance in body.resonances():
                if not self._in_memory(body, resonance.to_s()):
                    unclassified += 1
                    continue
                try:
                    rows.append(self._build_summary_row(body, resonance))
                except Exception as e:
                    logger.error(f"Error getting resonance summary for {body.name}: {e}")
        if unclassified:
            logger.warning(
                f"{unclassified} body-resonance pairs are not classified in memory (restore without "
                "recompute_librations, or a batched run): they are not in this summary, summary.csv has them"
            )

        summary = pd.DataFrame(rows)
        if 'comments' in summary.columns:  # rows of different types bring their own columns: comments stays last
            summary = summary[[c for c in summary.columns if c != 'comments'] + ['comments']]
        return summary

    def get_mmr_tables(self, bodies) -> dict:
        """mmr_numbers, episodes, candidates and constants frames of the rows in memory (SPEC §11.1–11.4).

        episodes (MMR, secular and ZLK rows) only with `save_episodes`; candidates top-N and the accepted ones
        (`save_candidates='top'`) or all; constants from the first analysis that saw planets (one table per run)."""
        tables = {name: [] for name in self.MMR_TABLES}
        constants = []
        top_only = getattr(self.config, 'save_candidates', 'top') == 'top'
        for body in bodies:
            analyses = getattr(body, 'mmr_analyses', {})
            for resonance in body.resonances():
                analysis = analyses.get(resonance.to_s())
                if analysis is None:
                    continue
                name = {'name': body.name}
                tables['mmr_numbers'].append({**analysis.numbers_row(), **name})
                if getattr(self.config, 'save_episodes', True):
                    tables['episodes'].extend({**row, **name} for row in analysis.episode_rows())
                tables['candidates'].extend({**row, **name} for row in analysis.candidate_rows(top_only=top_only))
                if not constants and analysis.context is not None:
                    constants = analysis.context.constants_rows()
            if getattr(self.config, 'save_episodes', True):
                for key, analysis in getattr(body, 'secular_analyses', {}).items():
                    tables['episodes'].extend({**row, 'name': body.name, 'resonance': key} for row in analysis.episode_rows())
        frames = {name: pd.DataFrame(rows) for name, rows in tables.items()}
        frames['constants'] = pd.DataFrame(constants)
        return frames

    def _build_summary_row(self, body: Body, resonance) -> dict:
        """One summary.csv row (SPEC §11.1): identity, verdict, checks, margins and decision path, key numbers,
        reproducibility, then the initial elements and the secular/ZLK columns, `comments` last.

        The type of the resonance is `resonance_type` (mmr, secular, lidov_kozai); `status` is the status string.
        """
        key = resonance.to_s()
        analysis = getattr(body, 'mmr_analyses', {}).get(key)
        if analysis is not None:
            return self._mmr_summary_row(body, resonance, analysis)
        return self._verdict_summary_row(body, resonance, body.librations[key])

    def _mmr_summary_row(self, body: Body, resonance, analysis) -> dict:
        """The summary row of an MMR analysis (`analysis.summary_row()`) with the package's columns."""
        row = analysis.summary_row()
        row['name'] = body.name
        row['package_commit'] = package_commit()
        row.update(self._body_columns(body, resonance))
        row['comments'] = ''
        return row

    def _verdict_summary_row(self, body: Body, resonance, verdict: Verdict) -> dict:
        """The summary row of a secular/ZLK row, as `LongPeriodAnalysis.summary_row` builds it: identity, the verdict
        (with the checks added after the analysis, e.g. check_zlk_ei), every measured number
        (`body.classifier_numbers[key]` = `numbers_row()`), `settings_hash` and `confidence_cells_sha`; then the
        package's columns, `comments` last."""
        key = resonance.to_s()
        added_by, parent = body.resonance_origin(key) if hasattr(body, 'resonance_origin') else ('', '')
        row = {'name': body.name, 'resonance': key, 'resonance_type': str(verdict.resonance_type), 'added_by': added_by, 'parent': parent}
        row.update(verdict.to_flat_dict())
        row.update({column: value for column, value in (body.classifier_numbers.get(key) or {}).items() if column not in row})
        analysis = body.secular_analyses.get(key)
        row['settings_hash'] = analysis.settings_hash if analysis is not None else ''
        row['confidence_cells_sha'] = analysis.confidence_cells_sha if analysis is not None else ''
        row['package_commit'] = package_commit()
        row.update(self._body_columns(body, resonance))
        row['comments'] = ''
        return row

    def _body_columns(self, body: Body, resonance) -> dict:
        """Initial elements (a0, e0, ...), the forced/free e and coherence columns of any row; the Lidov-Kozai
        parameters and free elements of a ZLK row; the free divisor of a secular row."""
        key = resonance.to_s()
        row = {column: body.initial_data.get(element) for column, element in self.INITIAL_ELEMENTS}
        if isinstance(resonance, LidovKozaiResonance):
            initial = body.initial_data  # osculating, at the epoch, invariable plane
            c1, c2, h_quad = (
                float(value)
                for value in LidovKozaiParameters.evaluate_invariable(initial['e'], initial['inc'], initial['Omega'], initial['omega'])
            )
        else:
            c1, c2, h_quad = None, None, None
        e_forced, e_free = body.forced_free_eccentricity()
        row.update({'c1': c1, 'c2': c2, 'h_quad': h_quad, 'e_forced': e_forced, 'e_free': e_free})
        row.update(coherence_analysis.summary_fields(body.coherence.get(key), body.zlk_gates.get(key)))
        # Free-element columns are written for Lidov-Kozai rows only: omega = varpi - Omega
        # is the angle they describe, and an MMR row would carry them without using them.
        if isinstance(resonance, LidovKozaiResonance):
            row.update(free_elements.summary_fields(body.free_elements))
            row.update(omega_free_gate.summary_fields(body.free_gates.get(key)))
        if isinstance(resonance, SecularResonance):
            divisor = body.secular_free_divisor(resonance)
            row['free_divisor'] = None if divisor is None else round(divisor, 4)
        return row

    def save_configuration_details(self, bodies, simulation):
        """Save configuration details to file."""
        SimulationSerializer.save_simulation_json(self.config, bodies, simulation)
