import numpy as np
from scipy import signal

from resonances.logger import logger
from resonances.simulation.config import SimulationConfig


def filter_angle(angle, sim_config: SimulationConfig):
    integration_time = abs(sim_config.tmax / (2 * np.pi))
    fs = sim_config.Nout / integration_time
    cutoff = sim_config.oscillations_cutoff
    order = sim_config.oscillations_filter_order
    nyq = 0.5 * fs

    filter_name = str(sim_config.filter)  # e.g. "butter", "savgol"
    if filter_name.find("butter") != -1:
        return butter_lowpass_filter(angle, cutoff, fs, order, nyq)
    raise ValueError(f"Unknown angle filter '{filter_name}'.")


def butter_lowpass_filter(data, cutoff, fs, order, nyq):
    normal_cutoff = cutoff / nyq
    # Ensure normalized cutoff frequency is valid for Butterworth filter (0 < Wn < 1)
    if normal_cutoff >= 1.0:
        # For secular resonances with very long integration times, adjust cutoff
        normal_cutoff = 0.99  # Use maximum allowable value
        logger.warning(
            f"Cutoff frequency ({cutoff}) >= Nyquist frequency ({nyq}). "
            f"Adjusting normalized cutoff to {normal_cutoff} for filter stability."
        )
    elif normal_cutoff <= 0.0:
        normal_cutoff = 0.01  # Use minimum allowable value
        logger.warning(f"Cutoff frequency ({cutoff}) <= 0. " f"Adjusting normalized cutoff to {normal_cutoff} for filter stability.")

    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data, method="gust")
    return y


# ============================================================================
# Optional helpers (NOT used internally)
# ============================================================================


def wrap(angle):
    """
    Wrap angle(s) to [0, 2π).
    Use ONLY for plotting or export.
    """
    return np.mod(angle, 2 * np.pi)
