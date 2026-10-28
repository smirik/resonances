import datetime
import math
import json
from enum import StrEnum
import numpy as np

import astdys
from resonances import mpcorb
from resonances.simulation.mpc_source import EPOCH_POLICIES as MPC_EPOCH_POLICIES, mpc_settings
from resonances.simulation.mpc_getorb import LOOKUP_MODES as MPC_LOOKUP_MODES
from resonances.data.util import datetime_from_string
from resonances.config import config as c
from resonances.logger import logger
from resonances.plotting.style import MMR_PLOT_KINDS, SECULAR_PLOT_KINDS, resolve_plot_options
from resonances.simulation import layout
from resonances.resonance.analysis.settings import (
    AnalysisSettings,
    DryRunSettings,
    dry_run_settings_from_config,
    parse_value,
    setting_fields,
    settings_from_config,
)
import os

# Plot kinds accepted in `plots`. The MMR kinds are drawn for MMRs only.
# 'clones' is the figure of an ensemble of clones (also drawn with 'combined'; simulation/ensemble_run.py).
PLOT_KINDS = ('evolution', 'ecc_vector', 'cross_spectrum', 'free_omega', 'clones') + tuple(
    dict.fromkeys(MMR_PLOT_KINDS + SECULAR_PLOT_KINDS)
)
# CLONES_SOURCE values: auto = the provider of the run's elements (astdys, mpc; nasa -> sbdb)
CLONE_SOURCES = ('auto', 'astdys', 'neodys', 'sbdb', 'mpc')


class SavePlotMode(StrEnum):
    """Mode for saving/plotting body data, by the verdict's status string (SPEC §11.5, `status_selected`):

    none       — save/plot nothing
    all        — save/plot everything
    resonant   — resonant, transient
    candidates — resonant, transient, near_separatrix, uncertain
    extended   — candidates and non_resonant with a libration origin (kinematic, induced, subresonance, ...)
    nonzero    — everything except non_resonant without a libration origin
    negative   — near_separatrix, uncertain, invalid
    """

    NONE = 'none'
    ALL = 'all'
    RESONANT = 'resonant'
    CANDIDATES = 'candidates'
    EXTENDED = 'extended'
    NONZERO = 'nonzero'
    NEGATIVE = 'negative'


RESONANT_STATUSES = frozenset({'resonant', 'transient'})
CANDIDATE_STATUSES = RESONANT_STATUSES | {'near_separatrix', 'uncertain'}
NEGATIVE_STATUSES = frozenset({'near_separatrix', 'uncertain', 'invalid'})
NON_RESONANT = 'non_resonant'


def status_selected(mode, status, libration_origin: str = '') -> bool:
    """Whether a row with this status string (and libration origin) is saved/plotted under `mode`.

    `status` None (no verdict) is selected by `all` only. `libration_origin` is the verdict's
    (kinematic, induced, subresonance, dynamic, unknown, or '' when nothing is bounded).
    """
    mode = _normalize_mode(mode)
    if mode is None or mode == SavePlotMode.NONE:
        return False
    if mode == SavePlotMode.ALL:
        return True
    if status is None:
        return False
    status, origin = str(status), str(libration_origin or '')
    if mode == SavePlotMode.RESONANT:
        return status in RESONANT_STATUSES
    if mode == SavePlotMode.CANDIDATES:
        return status in CANDIDATE_STATUSES
    if mode == SavePlotMode.EXTENDED:
        return status in CANDIDATE_STATUSES or (status == NON_RESONANT and origin != '')
    if mode == SavePlotMode.NONZERO:
        return status != NON_RESONANT or origin != ''
    if mode == SavePlotMode.NEGATIVE:
        return status in NEGATIVE_STATUSES
    return False


def _normalize_mode(value) -> SavePlotMode | None:
    """Normalize a save/plot mode value to SavePlotMode enum or None."""
    if value is None or value is False:
        return None
    if isinstance(value, SavePlotMode):
        return value
    if isinstance(value, str):
        try:
            return SavePlotMode(value.lower())
        except ValueError:
            return None
    return None


class SavePlanets(StrEnum):
    """Which planets are kept in memory and written to data-planet-<planet>.csv.

      all      — every planet of the solar system (needed by secular_planet_phases='measured')
      analysis — the planets the analysis reads: those of the bodies' resonances, plus MMR_OWNER_PLANETS when the run
                 has an MMR and the owner search is on (`Simulation.planets_needed`)
      none     — no planet (the MMR owner search and the ripple removal are then not measured)

    `bool(SavePlanets.NONE)` is False, so `if config.save_planets:` still reads "some planets are kept".
    """

    ALL = 'all'
    ANALYSIS = 'analysis'
    NONE = 'none'

    def __bool__(self) -> bool:
        return self != SavePlanets.NONE


def parse_save_planets(value) -> SavePlanets:
    """SAVE_PLANETS: all | analysis | none; a bool (or 'True'/'False') is accepted: True -> all, False -> none."""
    if isinstance(value, SavePlanets):
        return value
    if isinstance(value, bool) or value is None:
        return SavePlanets.ALL if value else SavePlanets.NONE
    text = str(value).strip().lower()
    if text in ('true', '1', 'yes'):
        return SavePlanets.ALL
    if text in ('false', '0', 'no', ''):
        return SavePlanets.NONE
    try:
        return SavePlanets(text)
    except ValueError:
        raise ValueError(f"save_planets is one of all, analysis, none (or a bool), got {value!r}") from None


def parse_output_step(value) -> float | str | None:
    """OUTPUT_STEP: None/'' (the default Nout), 'auto' (chosen by the static dry-run), or a positive number of years."""
    if value is None or str(value).strip() == '':
        return None
    if str(value).strip().lower() == 'auto':
        return 'auto'
    step = float(value)
    if not step > 0:
        raise ValueError(f"output_step is 'auto' or a positive number of years, got {value!r}")
    return step


def nout_for_step(span_yr: float, step_yr: float) -> int:
    """Nout = ceil(|span| / step) + 1 (at least 2): `np.linspace(0, tmax, Nout)` then steps by at most `step_yr`, the
    bound the 'auto' step is (5.0012 yr came from Nout = 4000 over 20 kyr; 4001 gives 4.9999 yr; 1001 yr at 2.5 yr
    gives 402 rows of 2.49 yr, 1 yr gives 2 rows). The ratio is rounded to 1e-9 first: tmax in whole REBOUND units
    leaves 20000.000000001 yr."""
    return max(1, math.ceil(round(abs(span_yr) / step_yr, 9))) + 1


DATA_SOURCES = ('nasa', 'astdys', 'mpc')

# Integrators that can carry the Moon: only the adaptive IAS15 follows its 27-day orbit (see docs/config.md).
INTEGRATORS_WITH_MOON = ('ias15',)


def check_integrator_with_moon(integrator: str):
    """Raise ValueError unless `integrator` can integrate the separate Moon."""
    if str(integrator).lower() not in INTEGRATORS_WITH_MOON:
        raise ValueError(
            f"solar_system_moon needs an adaptive integrator ({', '.join(INTEGRATORS_WITH_MOON)}), got {integrator!r}: "
            "a symplectic step of days to months cannot follow the Moon around the Earth."
        )


def _config_or(config, key: str, default: str) -> str:
    """A key of .env/.env.dist, or `default` when it is not there (older .env.dist, keys added later)."""
    return config.get(key) if config.has(key) else default


# The figures' subfolders when neither PLOT_LAYOUT nor the old PLOT_SUBFOLDER_STRATEGY is set (the benchmark's tree)
DEFAULT_PLOT_LAYOUT = 'confidence/status'


def _legacy_layout(strategy) -> str:
    """The layout of the old PLOT_SUBFOLDER_STRATEGY ('' / None: flat, 'status': by status)."""
    if strategy in (None, '', 'flat'):
        return 'flat'
    if strategy != 'status':
        raise ValueError(f"Invalid plot_subfolder_strategy: {strategy}. Valid values are None or 'status' (or use plot_layout).")
    return 'status'


class SimulationConfig:
    """Handles simulation configuration and setup parameters."""

    def __init__(self, **kwargs):
        """Initialize simulation configuration."""
        self.name = kwargs.get('name', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        # Internal flag for batch workers to skip path verification
        self._skip_path_verification = kwargs.get('_skip_path_verification', False)
        # Internal: the planets a batch worker keeps (its parent's `Simulation.planets_needed`), None elsewhere
        self._planets_needed = kwargs.get('_planets_needed')
        self._setup_date(kwargs.get('date'), kwargs.get('source'), kwargs)
        self._setup_integration_params(kwargs)
        self._setup_save_params(kwargs)
        self._setup_plot_params(kwargs)
        self._setup_libration_params(kwargs)
        self._setup_filtering_params(kwargs)
        self._setup_batch_params(kwargs)
        self._setup_coherence_params(kwargs)
        self._setup_analysis_params(kwargs)

        self.secular_angle_mode = kwargs.get('secular_angle_mode', c.get('SECULAR_ANGLE_MODE'))
        if self.secular_angle_mode not in ['osculating', 'proper']:
            raise ValueError(f"Invalid secular angle mode: {self.secular_angle_mode}. Valid modes are 'osculating' and 'proper'.")
        self._setup_secular_params(kwargs)

    @property
    def secular_reduced(self) -> bool:
        """Whether the classified secular angle is the reduced proper one (see `Body.build_proper_angle`)."""
        return self.secular_angle_mode == 'proper' and self.secular_remove_forced

    def _setup_secular_params(self, kwargs):
        """The reduced secular angle and the planets' mode phases (see docs/secular.md)."""
        self.secular_remove_forced = kwargs.get('secular_remove_forced', c.get('SECULAR_REMOVE_FORCED', 'True') == 'True')
        # The planets' mode phases: the linear law of const.py, or measured in the integration (planet_modes)
        self.secular_planet_phases = kwargs.get('secular_planet_phases', c.get('SECULAR_PLANET_PHASES', 'linear'))
        self.secular_phase_window = float(kwargs.get('secular_phase_window', c.get('SECULAR_PHASE_WINDOW', 5.0e6)))
        if self.secular_planet_phases not in ('linear', 'measured'):
            raise ValueError(f"secular_planet_phases is 'linear' or 'measured', got {self.secular_planet_phases!r}")
        if self.secular_planet_phases == 'measured' and self.save_planets != SavePlanets.ALL:
            raise ValueError("secular_planet_phases='measured' needs save_planets='all' (or True): the modes are fitted on every planet")

    def _setup_analysis_params(self, kwargs):
        """Every setting of the libration analysis and the dry-run (see resonance/analysis/settings.py).

        Each field of `AnalysisSettings`/`DryRunSettings` becomes an attribute of the same name, read
        from the kwarg or from its .env key. The settings objects are built (and validated) here once,
        and again whenever `analysis_settings` is read, so later attribute changes are honoured.
        """
        for cls in (AnalysisSettings, DryRunSettings):
            for f in setting_fields(cls):
                env = f.metadata['env']
                raw = kwargs[f.name] if f.name in kwargs else (c.get(env) if c.has(env) else None)
                setattr(self, f.name, parse_value(f, raw))
        settings_from_config(self)  # validates (ValueError with the reason)
        dry_run_settings_from_config(self)

        # Output of the MMR analysis (docs/config.md)
        self.save_episodes = self._read_bool(kwargs, 'save_episodes', 'SAVE_EPISODES', True)
        self.save_candidates = kwargs.get('save_candidates', c.get('SAVE_CANDIDATES', 'top'))
        if self.save_candidates not in ('top', 'all'):
            raise ValueError(f"save_candidates is 'top' or 'all', got {self.save_candidates!r}")
        self.save_constants = self._read_bool(kwargs, 'save_constants', 'SAVE_CONSTANTS', True)
        self.output_step = parse_output_step(kwargs.get('output_step', c.get('OUTPUT_STEP', '')))
        if isinstance(self.output_step, float):
            nout = nout_for_step(self.tmax_yrs, self.output_step)
            if kwargs.get('Nout') is None:
                self.Nout = nout
            elif int(kwargs['Nout']) != nout:
                logger.warning(f"Nout={kwargs['Nout']} overrides output_step={self.output_step} yr (that step would need Nout={nout})")

    @property
    def analysis_settings(self) -> AnalysisSettings:
        """The validated settings object passed to the analysis modules."""
        return settings_from_config(self)

    @property
    def dry_run_settings(self) -> DryRunSettings:
        """The validated settings of the dry-run."""
        return dry_run_settings_from_config(self)

    @staticmethod
    def _read_bool(kwargs, key: str, env_key: str, default: bool) -> bool:
        """A boolean setting from a kwarg or its .env key ('True'/'False')."""
        if key in kwargs:
            return bool(kwargs[key])
        return str(c.get(env_key, str(default))).strip().lower() == 'true'

    def _setup_date(self, date, source, kwargs=None):
        """The source of the asteroids' elements and the start date of the integration.

        Without an explicit date: the AstDyS catalog epoch for source 'astdys' (set by argument,
        .env or DATA_SOURCE), the standard epoch of the MPC catalogue for 'mpc', today's midnight
        for 'nasa'. The bodies' elements must be at this date (`BodyManager.check_epoch`). Dates are
        naive calendar values in TT (AstDyS, MPC), which Horizons reads as TDB for the planets.
        """
        self.source = source or c.get('DATA_SOURCE')
        if self.source not in DATA_SOURCES:
            raise ValueError(f"Unknown data source {self.source!r}. Valid: {', '.join(DATA_SOURCES)}")
        if self.source == 'mpc':
            self._setup_mpc(kwargs or {})

        if date is not None:
            self.date = datetime_from_string(date)
        elif self.source == 'astdys':
            astdys.set_type("osculating")
            self.date = astdys.get_catalog_datetime()
        elif self.source == 'mpc':
            self.date = mpcorb.datetime_from_mjd(self.mpc_snapshot['standard_epoch_mjd'])
        else:
            self.date = datetime.datetime.combine(datetime.datetime.today(), datetime.time.min)

    def _setup_mpc(self, kwargs):
        """The MPC catalogue (MPC_CATALOG), its epoch policy and the snapshot's provenance (resonances/mpcorb,
        settings from simulation/mpc_source.py).

        Only for source 'mpc'. A batch worker or a restored run gets `mpc_snapshot` from its parent's config and
        does not open the catalogue; elsewhere the cached snapshot is read (downloaded on first use).
        """
        self.mpc_catalog = kwargs.get('mpc_catalog', _config_or(c, 'MPC_CATALOG', 'mpcorb')) or 'mpcorb'
        self.mpc_epoch_policy = kwargs.get('mpc_epoch_policy', _config_or(c, 'MPC_EPOCH_POLICY', 'propagate')) or 'propagate'
        if self.mpc_epoch_policy not in MPC_EPOCH_POLICIES:
            raise ValueError(f"mpc_epoch_policy is one of {', '.join(MPC_EPOCH_POLICIES)}, got {self.mpc_epoch_policy!r}")
        # None (empty key): no cap on the propagated span
        max_days = kwargs.get('mpc_propagate_max_days', _config_or(c, 'MPC_PROPAGATE_MAX_DAYS', ''))
        self.mpc_propagate_max_days = None if max_days in (None, '') else float(max_days)
        # file: the catalogue only; file_then_api: an identifier it lacks is asked of MPC get-orb (simulation/mpc_getorb.py)
        self.mpc_lookup = str(kwargs.get('mpc_lookup', _config_or(c, 'MPC_LOOKUP', 'file')) or 'file').lower()
        if self.mpc_lookup not in MPC_LOOKUP_MODES:
            raise ValueError(f"mpc_lookup is one of {', '.join(MPC_LOOKUP_MODES)}, got {self.mpc_lookup!r}")
        snapshot = kwargs.get('mpc_snapshot')
        self.mpc_snapshot = dict(snapshot) if snapshot else mpcorb.provenance(mpcorb.snapshot(settings=mpc_settings(self.mpc_catalog)))

    def _setup_integration_params(self, kwargs):
        """Setup integration parameters."""
        integration_years = kwargs.get('integration_years', None)
        if integration_years is not None:
            self.tmax = int(integration_years * 2 * np.pi)
        else:
            self.tmax = kwargs.get('tmax', int(c.get('INTEGRATION_TMAX')))
        if 'Nout' in kwargs and kwargs['Nout'] is not None:
            self.Nout = int(kwargs['Nout'])
        self.integrator = kwargs.get('integrator', c.get('INTEGRATION_INTEGRATOR'))
        self.dt = kwargs.get('dt', float(c.get('INTEGRATION_DT')))
        self.integration_corrector = kwargs.get('integration_corrector', int(c.get('INTEGRATION_CORRECTOR')))
        # None = the integrator's own default in setup_integrator (WHFast 0, SABA 1)
        safe_mode = kwargs.get('integration_safe_mode', c.get('INTEGRATION_SAFE_MODE', ''))
        self.integration_safe_mode = None if safe_mode in (None, '') else int(safe_mode)
        # The Moon as a separate body, Earth as the geocentre (IntegrationEngine.create_solar_system_with_moon)
        self.solar_system_moon = bool(kwargs.get('solar_system_moon', c.get('SOLAR_SYSTEM_MOON', 'False') == 'True'))
        if self.solar_system_moon:
            check_integrator_with_moon(self.integrator)

    def _setup_save_params(self, kwargs):
        """Setup save and output parameters."""
        self.save = _normalize_mode(kwargs.get('save', c.get('SAVE')))
        self.save_summary = kwargs.get('save_summary', bool(c.get('SAVE_SUMMARY') == 'True'))
        self.save_planets = parse_save_planets(kwargs.get('save_planets', c.get('SAVE_PLANETS', 'analysis')))

        self.save_path = kwargs.get('save_path', None)
        if self.save_path is None:
            self.save_path = self._verify_existing_path(f"{c.get('SAVE_PATH')}/{self.name}")
        else:
            self.save_path = self._verify_existing_path(self.save_path)
        # The series CSVs (data-<body>.csv, periodograms, omega-free, planets) under save_path; '' = save_path itself
        self.data_folder = layout.folder_setting(kwargs.get('data_folder', _config_or(c, 'DATA_FOLDER', 'data')))

    @property
    def data_path(self) -> str:
        """The folder of the series CSVs: save_path/data_folder (save_path itself when data_folder is empty)."""
        return str(layout.layout_path(self.save_path, self.data_folder, (), {}))

    def _setup_plot_params(self, kwargs):
        """Setup plotting parameters."""
        self.plot = _normalize_mode(kwargs.get('plot', c.get('PLOT')))
        self.plot_type = kwargs.get('plot_type', c.get('PLOT_TYPE'))
        self.image_type = kwargs.get('image_type', c.get('PLOT_IMAGE_TYPE'))
        self.plot_config = kwargs.get('plot_config', c.get('PLOT_CONFIG'))  # Optional plot configuration

        # Where the figures go under plot_path: the individual figures in plot_figures_folder, the combined (verdict)
        # figures in plot_combined_folder ('' = plot_path itself), each split by a layout (simulation/layout.py)
        self.plot_figures_folder = layout.folder_setting(kwargs.get('plot_figures_folder', _config_or(c, 'PLOT_FIGURES_FOLDER', 'figures')))
        self.plot_combined_folder = layout.folder_setting(
            kwargs.get('plot_combined_folder', _config_or(c, 'PLOT_COMBINED_FOLDER', 'combined'))
        )
        # plot_layout, else the old name plot_subfolder_strategy ('' or 'status'), else PLOT_LAYOUT, else the old key
        # when it is set, else DEFAULT_PLOT_LAYOUT (author 2026-10-27, W24 Q8)
        plot_layout = kwargs.get('plot_layout')
        if plot_layout is None and 'plot_subfolder_strategy' in kwargs:
            plot_layout = _legacy_layout(kwargs['plot_subfolder_strategy'])
        if plot_layout is None:
            legacy = _config_or(c, 'PLOT_SUBFOLDER_STRATEGY', '')
            plot_layout = _config_or(c, 'PLOT_LAYOUT', '') or (_legacy_layout(legacy) if legacy else DEFAULT_PLOT_LAYOUT)
        self.plot_layout = layout.layout_name(layout.parse_layout(plot_layout))
        combined_layout = kwargs.get('plot_combined_layout', _config_or(c, 'PLOT_COMBINED_LAYOUT', ''))
        self.plot_combined_layout = layout.layout_name(layout.parse_layout(combined_layout or self.plot_layout))

        self.plot_path = kwargs.get('plot_path', None)
        if self.plot_path is None:
            self.plot_path = self._verify_existing_path(f"{c.get('PLOT_PATH')}/{self.name}")
        else:
            self.plot_path = self._verify_existing_path(self.plot_path)

        # Plot kinds to generate (see PLOT_KINDS); an unknown name is an error, not a silent no-op
        plots_param = kwargs.get('plots', None)
        if plots_param is not None:
            self.plots = list(plots_param) if isinstance(plots_param, (list, tuple)) else [plots_param]
        else:
            plots_str = c.get('PLOTS', 'combined')
            self.plots = [p.strip() for p in plots_str.split(',') if p.strip()]
        unknown = [p for p in self.plots if p not in PLOT_KINDS]
        if unknown:
            raise ValueError(f"Unknown plot kind(s): {', '.join(unknown)}. Valid: {', '.join(PLOT_KINDS)}")

        # Resonance types whose figures are drawn whatever their status (e.g. ['lidov_kozai'] to
        # always see the ZLK angle); the `plot` mode still decides for every other resonance.
        plot_always = kwargs.get('plot_always', None)
        if plot_always is None:
            plot_always = c.get('PLOT_ALWAYS', '') if c.has('PLOT_ALWAYS') else ''
        if isinstance(plot_always, str):
            plot_always = [x.strip() for x in plot_always.split(',') if x.strip()]
        self.plot_always = list(plot_always)

        # Options of the MMR diagnostic plots (style, raw series, per-kind settings), merged over
        # DEFAULT_PLOT_OPTIONS and validated. PLOT_OPTIONS in .env holds the same dict as JSON.
        plot_options = kwargs.get('plot_options', None)
        if plot_options is None and c.has('PLOT_OPTIONS') and c.get('PLOT_OPTIONS', '').strip():
            plot_options = json.loads(c.get('PLOT_OPTIONS'))
        self.plot_options = resolve_plot_options(plot_options)

    @property
    def plot_subfolder_strategy(self):
        """The old name of the layout: 'status' when the figures are split by status alone, else None."""
        return 'status' if self.plot_layout == 'status' else None

    @plot_subfolder_strategy.setter
    def plot_subfolder_strategy(self, value):
        """Setting the old name sets both layouts ('status' or None/'')."""
        self.plot_layout = self.plot_combined_layout = layout.layout_name(layout.parse_layout(_legacy_layout(value)))

    def _verify_existing_path(self, path):
        """
        Verify if the given path exists and modify it to avoid overwriting.

        Returns original path if:
        - _skip_path_verification is True (batch workers)
        - State file exists (resuming simulation): simulation_state.json, or the clone checkpoint
          clones_state/clones_state.json (simulation/clone_store.py)
        - Directory doesn't exist or is empty

        Returns timestamped path if directory exists and is not empty.
        """
        if self._skip_path_verification:
            return path

        state_files = (os.path.join(path, "simulation_state.json"), os.path.join(path, "clones_state", "clones_state.json"))
        if any(os.path.exists(state_file) for state_file in state_files):
            return path

        is_non_empty_dir = os.path.isdir(path) and os.listdir(path)
        if is_non_empty_dir:
            now = datetime.datetime.now()
            new_path = f"{path}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
            logger.warning(f"Path {path} already exists and is not empty. Using new path: {new_path}")
            return new_path

        return path

    def _setup_libration_params(self, kwargs):
        """Setup libration analysis parameters.

        The span-derived defaults use the absolute length of the integration: for a backward
        run (negative tmax) they must stay positive, otherwise the periodogram trims a negative
        number of points and keeps only the tail of every series.
        """
        span_yrs = abs(self.tmax_yrs)
        self.oscillations_cutoff = kwargs.get('oscillations_cutoff', float(c.get('LIBRATION_FILTER_CUTOFF')))
        self.oscillations_filter_order = kwargs.get('oscillations_filter_order', int(c.get('LIBRATION_FILTER_ORDER')))
        self.periodogram_frequency_min = kwargs.get('periodogram_frequency_min', None)
        if self.periodogram_frequency_min is None:  # to show all possible frequencies based on integration time
            self.periodogram_frequency_min = 1.0 / span_yrs
        self.periodogram_frequency_max = kwargs.get('periodogram_frequency_max', float(c.get('LIBRATION_FREQ_MAX')))
        self.periodogram_soft = kwargs.get('periodogram_soft', float(c.get('LIBRATION_SOFT')))

        # Years cut from each end of a series before its periodogram (5% of the span if unset)
        self.libration_period_min = kwargs.get('libration_period_min', None)
        if self.libration_period_min is None:
            self.libration_period_min = span_yrs * 0.05

    def _setup_filtering_params(self, kwargs):
        self.filter = kwargs.get('filter', c.get('FILTER'))

    def _setup_coherence_params(self, kwargs):
        """Setup cross-spectral coherence parameters, the free elements and the free-omega gate.

        The e-i coherence changes no status (the columns zlk_coherent, zlk_check, zlk_line_*); the gate's outcome and
        the period-specific e-i coupling (`confirmed_ei`, zlk_ei_*) are steps of `analyse_zlk`.
        `coherence_pairs` and `coherence_phase_targets` are structured per-resonance-type
        overrides and are kwarg-only (a nested mapping has no sensible .env encoding).
        Both are merged per resonance type over the presets in `cross_spectrum.py`, so
        overriding 'lidov_kozai' leaves 'mmr' at its default.
        """
        self.coherence_enabled = kwargs.get('coherence_enabled', c.get('COHERENCE_ENABLED', 'True') == 'True')
        self.coherence_skip_non_resonant = kwargs.get('coherence_skip_non_resonant', c.get('COHERENCE_SKIP_NON_RESONANT', 'True') == 'True')
        self.coherence_alpha = kwargs.get('coherence_alpha', float(c.get('COHERENCE_ALPHA', 0.01)))
        self.coherence_window = kwargs.get('coherence_window', c.get('COHERENCE_WINDOW', 'hann'))
        self.coherence_max_lines = kwargs.get('coherence_max_lines', int(c.get('COHERENCE_MAX_LINES', 10)))

        n_segments = kwargs.get('coherence_n_segments', None)
        if n_segments is None:
            n_segments = [int(x.strip()) for x in c.get('COHERENCE_N_SEGMENTS', '4,8,16').split(',') if x.strip()]
        self.coherence_n_segments = list(n_segments)

        self.coherence_period_min = self._optional_float(kwargs, 'coherence_period_min', 'COHERENCE_PERIOD_MIN')
        self.coherence_period_max = self._optional_float(kwargs, 'coherence_period_max', 'COHERENCE_PERIOD_MAX')

        # Structured, kwarg-only. Keys are resonance types ('mmr', 'secular', 'lidov_kozai').
        self.coherence_pairs = kwargs.get('coherence_pairs', None)
        self.coherence_phase_targets = kwargs.get('coherence_phase_targets', None)

        # Forced/free split of the element vectors, and the free-omega gate built on it: a step of
        # analyse_zlk whose outcome analysis.secular.decide.decide_status turns into the Lidov-Kozai
        # status (confirmed: resonant, kinematic: non_resonant, unavailable: uncertain) — see
        # docs/free-elements.md.
        self.free_elements_enabled = kwargs.get('free_elements_enabled', c.get('FREE_ELEMENTS_ENABLED', 'True') == 'True')
        self.free_elements_sampling_years = self._optional_float(kwargs, 'free_elements_sampling_years', 'FREE_ELEMENTS_SAMPLING_YEARS')
        self.free_gate_enabled = kwargs.get('free_gate_enabled', c.get('FREE_GATE_ENABLED', 'True') == 'True')
        self.free_gate_min_cycles = kwargs.get('free_gate_min_cycles', float(c.get('FREE_GATE_MIN_CYCLES', 3)))
        self.free_gate_mask_quantile = kwargs.get('free_gate_mask_quantile', float(c.get('FREE_GATE_MASK_QUANTILE', 0.15)))

        # e-i coherence. Diagnostic since the free-omega gate took over the status decision:
        # this now only produces the strongest-line columns (zlk_coherent, zlk_check, zlk_line_*).
        self.zlk_coherence_check = kwargs.get('zlk_coherence_check', c.get('ZLK_COHERENCE_CHECK', 'True') == 'True')
        # PROVISIONAL default, calibrated on two objects only — see docs/coherence.md.
        self.zlk_min_anticorrelation = kwargs.get('zlk_min_anticorrelation', float(c.get('ZLK_MIN_ANTICORRELATION', 0.5)))

    @staticmethod
    def _optional_float(kwargs, key: str, env_key: str):
        """Read a float setting that may legitimately be unset (kwarg None or empty env value)."""
        if key in kwargs:
            value = kwargs[key]
            return None if value is None else float(value)
        raw = c.get(env_key, '')
        return float(raw) if str(raw).strip() else None

    @property
    def tmax(self):
        """Get integration time maximum."""
        return self.__tmax

    @tmax.setter
    def tmax(self, value):
        """Set integration time maximum and calculate related values: Nout from a numeric `output_step`, else
        the default |tmax| / 100 (an output every 100 REBOUND units = 15.9 yr)."""
        self.__tmax = value
        step = getattr(self, 'output_step', None)
        if isinstance(step, float):
            self.Nout = nout_for_step(self.tmax_yrs, step)
        else:
            self.Nout = abs(int(self.__tmax / 100))

    @property
    def output_step_yr(self) -> float:
        """The output step this config gives [yr]: |tmax| / (Nout - 1) of `np.linspace(0, tmax, Nout)`."""
        return abs(self.tmax_yrs) / (self.Nout - 1) if self.Nout > 1 else float('nan')

    @tmax.deleter
    def tmax(self):
        """Delete tmax property."""
        del self.__tmax

    @property
    def tmax_yrs(self):
        """Get integration time in years."""
        return self.__tmax / (2 * np.pi)

    @tmax_yrs.setter
    def tmax_yrs(self, value):
        """Set integration time in years."""
        self.__tmax = value * (2 * np.pi)

    def _setup_batch_params(self, kwargs):
        """Setup batch processing parameters."""
        self.batch_enabled = kwargs.get('batch_enabled', c.get('BATCH_ENABLED', 'True') == 'True')
        self.batch_threshold = kwargs.get('batch_threshold', int(c.get('BATCH_THRESHOLD', 100)))

        # Handle batch_size (can be None or empty string from config)
        batch_size_val = kwargs.get('batch_size', c.get('BATCH_SIZE', None))
        if batch_size_val is not None and batch_size_val != '':
            self.batch_size = int(batch_size_val)
        else:
            self.batch_size = None

        self.n_cores = kwargs.get('n_cores', int(c.get('BATCH_N_CORES', 1)))
        self.resume_enabled = kwargs.get('resume_enabled', True)
        self._setup_clone_params(kwargs)

    def _setup_clone_params(self, kwargs):
        """Clones (virtual asteroids, docs/clones.md): CLONES per body added by number (0 = off), their source, the
        covariance's epoch for AstDyS/NEODyS ('' = the source's default), the seed, clones per REBOUND simulation, and
        whether each clone gets its own figures."""
        self.clones = int(kwargs.get('clones', _config_or(c, 'CLONES', '0')) or 0)
        if self.clones < 0:
            raise ValueError(f'clones is 0 (off) or a number of clones, got {self.clones}')
        self.clones_source = str(kwargs.get('clones_source', _config_or(c, 'CLONES_SOURCE', 'auto')) or 'auto').lower()
        if self.clones_source not in CLONE_SOURCES:
            raise ValueError(f"clones_source is one of {', '.join(CLONE_SOURCES)}, got {self.clones_source!r}")
        self.clones_epoch = str(kwargs.get('clones_epoch', _config_or(c, 'CLONES_EPOCH', '')) or '').lower()
        if self.clones_epoch not in ('', 'fit', 'standard'):
            raise ValueError(f"clones_epoch is 'fit', 'standard' or empty, got {self.clones_epoch!r}")
        self.clones_seed = int(kwargs.get('clones_seed', _config_or(c, 'CLONES_SEED', '0')) or 0)
        self.clones_batch = int(kwargs.get('clones_batch', _config_or(c, 'CLONES_BATCH', '20')) or 20)
        if self.clones_batch < 1:
            raise ValueError(f'clones_batch is at least 1, got {self.clones_batch}')
        self.clones_plot_each = self._read_bool(kwargs, 'clones_plot_each', 'CLONES_PLOT_EACH', False)
