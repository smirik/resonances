import datetime
import numpy as np

import astdys
from resonances.data.util import datetime_from_string
from resonances.config import config as c
from resonances.logger import logger
import os


class SimulationConfig:
    """Handles simulation configuration and setup parameters."""

    def __init__(self, **kwargs):
        """Initialize simulation configuration."""
        self.name = kwargs.get('name', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self._setup_date(kwargs.get('date'), kwargs.get('source'))
        self._setup_integration_params(kwargs)
        self._setup_save_params(kwargs)
        self._setup_plot_params(kwargs)
        self._setup_libration_params(kwargs)
        self._setup_batch_params(kwargs)

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
        self.save = kwargs.get('save', c.get('SAVE'))
        self.save_summary = kwargs.get('save_summary', bool(c.get('SAVE_SUMMARY') == 'True'))
        self.save_planets = kwargs.get('save_planets', bool(c.get('SAVE_PLANETS') == 'True'))

        now = datetime.datetime.now()
        self.save_path = kwargs.get('save_path', None)
        if self.save_path is None:
            save_path = f"{c.get('SAVE_PATH')}/{self.name}"
            if os.path.exists(save_path):
                save_path += f"_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
                logger.info(f"Save path already exists. Using new path: {save_path}")
            self.save_path = save_path

    def _setup_plot_params(self, kwargs):
        """Setup plotting parameters."""
        self.plot = kwargs.get('plot', c.get('PLOT'))
        self.plot_type = kwargs.get('plot_type', c.get('PLOT_TYPE'))
        self.image_type = kwargs.get('image_type', c.get('PLOT_IMAGE_TYPE'))
        self.plot_config = kwargs.get('plot_config', None)  # Optional plot configuration

        now = datetime.datetime.now()
        self.plot_path = kwargs.get('plot_path', None)
        if self.plot_path is None:
            plot_path = f"{c.get('PLOT_PATH')}/{self.name}"
            if os.path.exists(plot_path):
                plot_path += f"_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
                logger.info(f"Plot path already exists. Using new path: {plot_path}")
            self.plot_path = plot_path

    def _setup_libration_params(self, kwargs):
        """Setup libration analysis parameters."""
        self.oscillations_cutoff = kwargs.get('oscillations_cutoff', float(c.get('LIBRATION_FILTER_CUTOFF')))
        self.oscillations_filter_order = kwargs.get('oscillations_filter_order', int(c.get('LIBRATION_FILTER_ORDER')))
        self.periodogram_frequency_min = kwargs.get('periodogram_frequency_min', float(c.get('LIBRATION_FREQ_MIN')))
        self.periodogram_frequency_max = kwargs.get('periodogram_frequency_max', float(c.get('LIBRATION_FREQ_MAX')))
        self.periodogram_critical = kwargs.get('periodogram_critical', float(c.get('LIBRATION_CRITICAL')))
        self.periodogram_soft = kwargs.get('periodogram_soft', float(c.get('LIBRATION_SOFT')))
        self.libration_period_critical = kwargs.get('libration_period_critical', int(c.get('LIBRATION_PERIOD_CRITICAL')))

        # Handle libration_monotony_critical specially since it's a list
        if 'libration_monotony_critical' in kwargs:
            self.libration_monotony_critical = kwargs['libration_monotony_critical']
        else:
            self.libration_monotony_critical = [float(x.strip()) for x in c.get('LIBRATION_MONOTONY_CRITICAL').split(",")]

        self.libration_period_min = kwargs.get('libration_period_min', int(c.get('LIBRATION_PERIOD_MIN')))

    @property
    def tmax(self):
        """Get integration time maximum."""
        return self.__tmax

    @tmax.setter
    def tmax(self, value):
        """Set integration time maximum and calculate related values."""
        self.__tmax = value
        self.tmax_yrs = self.__tmax / (2 * np.pi)
        self.Nout = abs(int(self.tmax / 100))

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
        return astdys.get_catalog_datetime() if self.source == 'astdys' else self.date

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
