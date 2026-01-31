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


def is_unphysical_orbit(body) -> bool:
    """Minimum check for unphysical orbit."""

    # hyperbolic orbit (e > 1 = unbound)
    if np.any(body.eccentricity > 1.0):
        return True

    # negative semimajor axis = negative energy,
    # but it's derived from e > 1, so it's redundant,
    # but it's safer to check
    if np.any(body.semimajor_axis < 0):
        return True

    return False, "ok"
