import datetime
import numpy as np

import resonances
import astdys
import resonances.data.util
from resonances.config import config as c


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

    def _setup_date(self, date, source):
        """Setup date and source configuration."""
        self.source = source or c.get('DATA_SOURCE')

        if date is not None:
            self.date = resonances.data.util.datetime_from_string(date)
        elif source == 'astdys':
            self.date = astdys.datetime()
        else:
            self.date = datetime.datetime.combine(datetime.datetime.today(), datetime.time.min)

    def _setup_integration_params(self, kwargs):
        """Setup integration parameters."""
        self.tmax = kwargs.get('tmax', int(c.get('INTEGRATION_TMAX')))
        self.integrator = kwargs.get('integrator', c.get('INTEGRATION_INTEGRATOR'))
        self.dt = kwargs.get('dt', float(c.get('INTEGRATION_DT')))
        self.integrator_corrector = kwargs.get('integrator_corrector', int(c.get('INTEGRATION_CORRECTOR')))
        self.integrator_safe_mode = kwargs.get('integrator_safe_mode', 1)

    def _setup_save_params(self, kwargs):
        """Setup save and output parameters."""
        self.save = kwargs.get('save', c.get('SAVE_MODE'))
        self.save_summary = kwargs.get('save_summary', bool(c.get('SAVE_SUMMARY')))

        now = datetime.datetime.now()
        self.save_path = kwargs.get('save_path', f"{c.get('SAVE_PATH')}/{now.strftime('%Y-%m-%d_%H:%M:%S')}")

    def _setup_plot_params(self, kwargs):
        """Setup plotting parameters."""
        self.plot = kwargs.get('plot', c.get('PLOT_MODE'))
        self.plot_type = kwargs.get('plot_type', c.get('PLOT_TYPE'))
        self.image_type = kwargs.get('image_type', c.get('PLOT_IMAGE_TYPE'))

        now = datetime.datetime.now()
        self.plot_path = kwargs.get('plot_path', f"{c.get('PLOT_PATH')}/{now.strftime('%Y-%m-%d_%H:%M:%S')}")

    def _setup_libration_params(self, kwargs):
        """Setup libration analysis parameters."""
        self.oscillations_cutoff = kwargs.get('oscillations_cutoff', float(resonances.config.get('LIBRATION_FILTER_CUTOFF')))
        self.oscillations_filter_order = kwargs.get('oscillations_filter_order', int(resonances.config.get('LIBRATION_FILTER_ORDER')))
        self.periodogram_frequency_min = kwargs.get('periodogram_frequency_min', float(resonances.config.get('LIBRATION_FREQ_MIN')))
        self.periodogram_frequency_max = kwargs.get('periodogram_frequency_max', float(resonances.config.get('LIBRATION_FREQ_MAX')))
        self.periodogram_critical = kwargs.get('periodogram_critical', float(resonances.config.get('LIBRATION_CRITICAL')))
        self.periodogram_soft = kwargs.get('periodogram_soft', float(resonances.config.get('LIBRATION_SOFT')))
        self.libration_period_critical = kwargs.get('libration_period_critical', int(resonances.config.get('LIBRATION_PERIOD_CRITICAL')))

        # Handle libration_monotony_critical specially since it's a list
        if 'libration_monotony_critical' in kwargs:
            self.libration_monotony_critical = kwargs['libration_monotony_critical']
        else:
            self.libration_monotony_critical = [float(x.strip()) for x in resonances.config.get('LIBRATION_MONOTONY_CRITICAL').split(",")]

        self.libration_period_min = kwargs.get('libration_period_min', int(resonances.config.get('LIBRATION_PERIOD_MIN')))

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
        return astdys.datetime() if self.source == 'astdys' else self.date
