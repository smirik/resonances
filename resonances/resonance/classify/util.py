from typing import Tuple
import numpy as np


def merge_intervals(intervals: np.ndarray, *, join_touching: bool = True) -> np.ndarray:
    """
    intervals: shape (M, 2), each [start, end], start <= end
    join_touching=True => [1,2] and [2,3] will be merged into [1,3]
    """
    intervals = np.asarray(intervals)
    if intervals.size == 0:
        return intervals.reshape(0, 2)

    order = np.argsort(intervals[:, 0], kind="mergesort")
    s = intervals[order, 0]
    e = intervals[order, 1]

    emax = np.maximum.accumulate(e)

    if join_touching:
        new_block = np.r_[True, s[1:] > emax[:-1]]
    else:
        new_block = np.r_[True, s[1:] >= emax[:-1]]

    block_starts = np.flatnonzero(new_block)
    merged_s = s[block_starts]

    # end of the block = emax on the last element of the block
    last_idx = np.r_[block_starts[1:] - 1, len(s) - 1]
    merged_e = emax[last_idx]

    return np.column_stack([merged_s, merged_e])


def is_unphysical_orbit(body) -> Tuple[bool, str]:
    """Minimum check for unphysical orbit (`unphysical_arrays` on body.axis and body.ecc)."""
    return unphysical_arrays(body.axis, body.ecc)


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


def check_chaos(body) -> Tuple[int, str]:
    """Check for chaotic or suspicious orbital behaviour.

    Returns (flag, comment):
      1  — unphysical orbit (ecc > 1.3, a < 0, or non-finite a or ecc)
     -1  — semi-major axis changed > 100% (|a_final-a0|/a0 or |a_max-a0|/a0)
      0  — normal
    """
    if body.axis is None or body.ecc is None:
        return 0, ""

    is_unstable, reason = is_unphysical_orbit(body)
    if is_unstable:
        return 1, f"unphysical: {reason}"

    a0 = body.axis[0]
    if a0 == 0:
        return 1, "unphysical: a0=0"

    a_final = body.axis[-1]
    a_max = np.max(body.axis)
    abs_a0 = abs(a0)

    rel_change_final = abs(a_final - a0) / abs_a0
    rel_change_max = abs(a_max - a0) / abs_a0

    if rel_change_final > 1.0 or rel_change_max > 1.0:
        parts = []
        if rel_change_final > 1.0:
            parts.append(f"|da_final|/a0={rel_change_final:.1%}")
        if rel_change_max > 1.0:
            parts.append(f"|da_max|/a0={rel_change_max:.1%}")
        return -1, f"large |da|>100%: {', '.join(parts)}"

    return 0, ""


def calc_sigma_derivative(times: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """d sigma / dt by central differences, one-sided at the two ends (same length as sigma)."""
    sigma_dot = np.zeros_like(sigma)
    sigma_dot[1:-1] = (sigma[2:] - sigma[:-2]) / (times[2:] - times[:-2])
    dt = np.diff(times)
    sigma_dot[0] = (sigma[1] - sigma[0]) / dt[0]
    sigma_dot[-1] = (sigma[-1] - sigma[-2]) / dt[-1]
    return sigma_dot
