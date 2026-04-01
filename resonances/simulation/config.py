import datetime
from enum import StrEnum
import numpy as np

import astdys
from resonances.data.util import datetime_from_string
from resonances.config import config as c
from resonances.logger import logger
import os


class SavePlotMode(StrEnum):
    """Mode for saving/plotting body data.

    Modes filter by ResonanceStatus value:
      none       — save/plot nothing
      all        — save/plot everything
      resonant   — status > 0 (TRANSIENT, LIBRATION)
      candidates — status > 0 or -3 (resonant + NEAR_SEPARATRIX)
      extended   — status > 0 or in {-3, -4, -5, -9} (candidates + uncertain/slow)
      nonzero    — status != 0
      negative   — status < 0
    """

    NONE = 'none'
    ALL = 'all'
    RESONANT = 'resonant'
    CANDIDATES = 'candidates'
    EXTENDED = 'extended'
    NONZERO = 'nonzero'
    NEGATIVE = 'negative'


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


class SimulationConfig:
    """Handles simulation configuration and setup parameters."""

    def __init__(self, **kwargs):
        """Initialize simulation configuration."""
        self.name = kwargs.get('name', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        # Internal flag for batch workers to skip path verification
        self._skip_path_verification = kwargs.get('_skip_path_verification', False)
        self._setup_date(kwargs.get('date'), kwargs.get('source'))
        self._setup_integration_params(kwargs)
        self._setup_save_params(kwargs)
        self._setup_plot_params(kwargs)
        self._setup_libration_params(kwargs)
        self._setup_filtering_params(kwargs)
        self._setup_batch_params(kwargs)
        self._setup_classify_params(kwargs)

        self.secular_angle_mode = kwargs.get('secular_angle_mode', c.get('SECULAR_ANGLE_MODE'))
        if self.secular_angle_mode not in ['osculating', 'proper']:
            raise ValueError(f"Invalid secular angle mode: {self.secular_angle_mode}. Valid modes are 'osculating' and 'proper'.")

    def _setup_date(self, date, source):
        """Setup date and source configuration."""
        self.source = source or c.get('DATA_SOURCE')

        if date is not None:
            self.date = datetime_from_string(date)
        elif source == 'astdys':
            astdys.set_type("osculating")
            self.date = astdys.get_catalog_datetime()
        else:
            self.date = datetime.datetime.combine(datetime.datetime.today(), datetime.time.min)

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
        self.integration_safe_mode = kwargs.get('integration_safe_mode', 1)

    def _setup_save_params(self, kwargs):
        """Setup save and output parameters."""
        self.save = _normalize_mode(kwargs.get('save', c.get('SAVE')))
        self.save_summary = kwargs.get('save_summary', bool(c.get('SAVE_SUMMARY') == 'True'))
        self.save_planets = kwargs.get('save_planets', bool(c.get('SAVE_PLANETS') == 'True'))

        self.save_path = kwargs.get('save_path', None)
        if self.save_path is None:
            self.save_path = self._verify_existing_path(f"{c.get('SAVE_PATH')}/{self.name}")
        else:
            self.save_path = self._verify_existing_path(self.save_path)

    def _setup_plot_params(self, kwargs):
        """Setup plotting parameters."""
        self.plot = _normalize_mode(kwargs.get('plot', c.get('PLOT')))
        self.plot_type = kwargs.get('plot_type', c.get('PLOT_TYPE'))
        self.image_type = kwargs.get('image_type', c.get('PLOT_IMAGE_TYPE'))
        self.plot_config = kwargs.get('plot_config', c.get('PLOT_CONFIG'))  # Optional plot configuration

        plot_subfolder_strategy = kwargs.get('plot_subfolder_strategy', c.get('PLOT_SUBFOLDER_STRATEGY', None))
        self.plot_subfolder_strategy = plot_subfolder_strategy if plot_subfolder_strategy else None
        if self.plot_subfolder_strategy is not None and self.plot_subfolder_strategy not in ['status']:
            raise ValueError(f"Invalid plot_subfolder_strategy: {self.plot_subfolder_strategy}. Valid values are None or 'status'.")

        self.plot_path = kwargs.get('plot_path', None)
        if self.plot_path is None:
            self.plot_path = self._verify_existing_path(f"{c.get('PLOT_PATH')}/{self.name}")
        else:
            self.plot_path = self._verify_existing_path(self.plot_path)

        # Plot types to generate (list of: evolution, phase_portrait)
        plots_param = kwargs.get('plots', None)
        if plots_param is not None:
            self.plots = plots_param if isinstance(plots_param, list) else [plots_param]
        else:
            plots_str = c.get('PLOTS', 'evolution')
            self.plots = [p.strip() for p in plots_str.split(',') if p.strip()]

        # Phase portrait slow points percentile threshold (0-100)
        self.phase_portrait_slow_percentile = kwargs.get(
            'phase_portrait_slow_percentile', float(c.get('PHASE_PORTRAIT_SLOW_PERCENTILE', 95))
        )

    def _verify_existing_path(self, path):
        """
        Verify if the given path exists and modify it to avoid overwriting.

        Returns original path if:
        - _skip_path_verification is True (batch workers)
        - State file exists (resuming simulation)
        - Directory doesn't exist or is empty

        Returns timestamped path if directory exists and is not empty.
        """
        if self._skip_path_verification:
            return path

        state_file = os.path.join(path, "simulation_state.json")
        if os.path.exists(state_file):
            return path

        is_non_empty_dir = os.path.isdir(path) and os.listdir(path)
        if is_non_empty_dir:
            now = datetime.datetime.now()
            new_path = f"{path}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
            logger.warning(f"Path {path} already exists and is not empty. Using new path: {new_path}")
            return new_path

        return path

    def _setup_libration_params(self, kwargs):
        """Setup libration analysis parameters."""
        self.oscillations_cutoff = kwargs.get('oscillations_cutoff', float(c.get('LIBRATION_FILTER_CUTOFF')))
        self.oscillations_filter_order = kwargs.get('oscillations_filter_order', int(c.get('LIBRATION_FILTER_ORDER')))
        self.periodogram_frequency_min = kwargs.get('periodogram_frequency_min', None)
        if self.periodogram_frequency_min is None:  # to show all possible frequencies based on integration time
            self.periodogram_frequency_min = 1.0 / self.tmax_yrs
        self.periodogram_frequency_max = kwargs.get('periodogram_frequency_max', float(c.get('LIBRATION_FREQ_MAX')))
        self.periodogram_critical = kwargs.get('periodogram_critical', None)
        if self.periodogram_critical is None:
            self.periodogram_critical = self.tmax_yrs * 0.2  # if not set, 20% of integration time should be in libration
        self.periodogram_soft = kwargs.get('periodogram_soft', float(c.get('LIBRATION_SOFT')))

        self.libration_period_critical = kwargs.get('libration_period_critical', None)
        if self.libration_period_critical is None:
            self.libration_period_critical = round(self.tmax_yrs * 0.1)  # if not set, 10% of integration time should be in libration

        # Handle libration_monotony_critical specially since it's a list
        if 'libration_monotony_critical' in kwargs:
            self.libration_monotony_critical = kwargs['libration_monotony_critical']
        else:
            self.libration_monotony_critical = [float(x.strip()) for x in c.get('LIBRATION_MONOTONY_CRITICAL').split(",")]

        self.libration_period_min = kwargs.get('libration_period_min', None)
        if self.libration_period_min is None:
            self.libration_period_min = self.tmax_yrs * 0.05  # if not set, let's librate at least 5%

    def _setup_filtering_params(self, kwargs):
        self.filter = kwargs.get('filter', c.get('FILTER'))

    @property
    def tmax(self):
        """Get integration time maximum."""
        return self.__tmax

    @tmax.setter
    def tmax(self, value):
        """Set integration time maximum and calculate related values."""
        self.__tmax = value
        self.Nout = abs(int(self.__tmax / 100))

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

    def get_bodies_date(self):
        """Get the date to use for body elements."""
        return self.date

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
        self.stop_on_failure = kwargs.get('stop_on_failure', True)
        self.resume_enabled = kwargs.get('resume_enabled', True)

    def _setup_classify_params(self, kwargs):
        """Setup classification parameters matching ClassifyParams fields."""
        # Window settings for segment analysis
        self.classify_window_step = kwargs.get(
            'classify_window_step',
            float(c.get('CLASSIFY_WINDOW_STEP', 0.05)),
        )
        window_steps_raw = kwargs.get('classify_window_steps', None)
        if window_steps_raw is not None:
            self.classify_window_steps = window_steps_raw
        else:
            self.classify_window_steps = [float(x.strip()) for x in c.get('CLASSIFY_WINDOW_STEPS', '0.1,0.2,0.3').split(',')]

        # Global thresholds
        self.classify_rev_libration = kwargs.get(
            'classify_rev_libration',
            float(c.get('CLASSIFY_REV_LIBRATION', 1.0)),
        )
        self.classify_tto_pure_libration = kwargs.get(
            'classify_tto_pure_libration',
            float(c.get('CLASSIFY_TTO_PURE_LIBRATION', 0.5)),
        )
        self.classify_tto_partial_libration = kwargs.get(
            'classify_tto_partial_libration',
            float(c.get('CLASSIFY_TTO_PARTIAL_LIBRATION', 2.5)),
        )
        self.classify_tto_non_resonant = kwargs.get(
            'classify_tto_non_resonant',
            float(c.get('CLASSIFY_TTO_NON_RESONANT', 6.0)),
        )
        self.classify_tto_transient_global = kwargs.get(
            'classify_tto_transient_global',
            float(c.get('CLASSIFY_TTO_TRANSIENT_GLOBAL', 3.0)),
        )
        self.classify_tto_near_separatrix = kwargs.get(
            'classify_tto_near_separatrix',
            float(c.get('CLASSIFY_TTO_NEAR_SEPARATRIX', 6.0)),
        )

        # Staircase detection thresholds
        self.classify_staircase_min_tto = kwargs.get(
            'classify_staircase_min_tto',
            float(c.get('CLASSIFY_STAIRCASE_MIN_TTO', 2.0)),
        )
        self.classify_staircase_max_resid_acf_zero = kwargs.get(
            'classify_staircase_max_resid_acf_zero',
            float(c.get('CLASSIFY_STAIRCASE_MAX_RESID_ACF_ZERO', 0.15)),
        )

        # Segment quality thresholds
        self.classify_good_seg_max_rev = kwargs.get(
            'classify_good_seg_max_rev',
            float(c.get('CLASSIFY_GOOD_SEG_MAX_REV', 1.0)),
        )
        self.classify_good_seg_max_tto = kwargs.get(
            'classify_good_seg_max_tto',
            float(c.get('CLASSIFY_GOOD_SEG_MAX_TTO', 0.5)),
        )
        self.classify_reasonable_seg_max_rev_1 = kwargs.get(
            'classify_reasonable_seg_max_rev_1',
            float(c.get('CLASSIFY_REASONABLE_SEG_MAX_REV_1', 2.0)),
        )
        self.classify_reasonable_seg_max_tto_1 = kwargs.get(
            'classify_reasonable_seg_max_tto_1',
            float(c.get('CLASSIFY_REASONABLE_SEG_MAX_TTO_1', 1.0)),
        )
        self.classify_reasonable_seg_max_rev_2 = kwargs.get(
            'classify_reasonable_seg_max_rev_2',
            float(c.get('CLASSIFY_REASONABLE_SEG_MAX_REV_2', 1.0)),
        )
        self.classify_reasonable_seg_max_tto_2 = kwargs.get(
            'classify_reasonable_seg_max_tto_2',
            float(c.get('CLASSIFY_REASONABLE_SEG_MAX_TTO_2', 1.5)),
        )
