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
    """Minimum check for unphysical orbit."""

    # hyperbolic orbit (e > 1 = unbound)
    if np.any(body.ecc > 1.3):
        return True, "hyperbolic"

    # negative semimajor axis = negative energy,
    # but it's derived from e > 1, so it's redundant,
    # but it's safer to check
    if np.any(body.axis < 0):
        return True, "negative_sma"

    return False, "ok"


def check_chaos(body) -> Tuple[int, str]:
    """Check for chaotic or suspicious orbital behaviour.

    Returns (flag, comment):
      1  — unphysical orbit (ecc > 1.3 or a < 0)
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
