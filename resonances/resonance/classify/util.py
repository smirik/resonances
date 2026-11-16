"""The unphysical-orbit rule shared by every analysis (`unphysical_arrays`) and the rate of an angle by central
differences (`calc_sigma_derivative`, the MMR diagnostic plots)."""

from typing import Tuple

import numpy as np


def unphysical_arrays(axis: np.ndarray, ecc: np.ndarray) -> Tuple[bool, str]:
    """(bad, reason) of the unphysical-orbit rule on the arrays of a and e: non_finite, hyperbolic, negative_sma or ok."""

    # NaN/inf elements: the integration lost the orbit (comparisons with NaN below would all pass)
    if not (np.isfinite(ecc).all() and np.isfinite(axis).all()):
        return True, "non_finite"

    # hyperbolic orbit (e > 1 = unbound)
    if np.any(ecc > 1.3):
        return True, "hyperbolic"

    # negative semimajor axis = negative energy,
    # but it's derived from e > 1, so it's redundant,
    # but it's safer to check
    if np.any(axis < 0):
        return True, "negative_sma"

    return False, "ok"


def calc_sigma_derivative(times: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """d sigma / dt by central differences, one-sided at the two ends (same length as sigma)."""
    sigma_dot = np.zeros_like(sigma)
    sigma_dot[1:-1] = (sigma[2:] - sigma[:-2]) / (times[2:] - times[:-2])
    dt = np.diff(times)
    sigma_dot[0] = (sigma[1] - sigma[0]) / dt[0]
    sigma_dot[-1] = (sigma[-1] - sigma[-2]) / dt[-1]
    return sigma_dot
