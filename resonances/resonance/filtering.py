import functools
import numpy as np
from scipy import signal

from resonances.logger import logger
from resonances.simulation.config import SimulationConfig


@functools.lru_cache(maxsize=4)
def _butter_coefficients(cutoff, nyq, order):
    """Compute and cache Butterworth filter coefficients."""
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
        logger.warning(
            f"Cutoff frequency ({cutoff}) >= Nyquist frequency ({nyq}). "
            f"Adjusting normalized cutoff to {normal_cutoff} for filter stability."
        )
    elif normal_cutoff <= 0.0:
        normal_cutoff = 0.01
        logger.warning(f"Cutoff frequency ({cutoff}) <= 0. " f"Adjusting normalized cutoff to {normal_cutoff} for filter stability.")

    return signal.butter(order, normal_cutoff, btype='low', analog=False)


def filter_angle(angle, sim_config: SimulationConfig):
    integration_time = abs(sim_config.tmax / (2 * np.pi))
    fs = sim_config.Nout / integration_time
    cutoff = sim_config.oscillations_cutoff
    order = sim_config.oscillations_filter_order
    nyq = 0.5 * fs

    filter_name = str(sim_config.filter)
    if filter_name.find("butter") != -1:
        b, a = _butter_coefficients(cutoff, nyq, order)
        return signal.filtfilt(b, a, angle, method="gust")
    raise ValueError(f"Unknown angle filter '{filter_name}'.")


def butter_lowpass_filter(data, cutoff, fs, order, nyq):
    b, a = _butter_coefficients(cutoff, nyq, order)
    return signal.filtfilt(b, a, data, method="gust")


# ============================================================================
# Optional helpers (NOT used internally)
# ============================================================================


def wrap(angle):
    """
    Wrap angle(s) to [0, 2π).
    Use ONLY for plotting or export.
    """
    return np.mod(angle, 2 * np.pi)
