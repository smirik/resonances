"""
Resonance Status Resolver
=========================

This module resolves the final resonance status for MMRs (Mean-Motion Resonances)
by checking periodogram overlap between resonant angle and semi-major axis.

STATUS CODES
------------
    2: Resonant - angle librates for the entire duration AND periodogram peaks match
   -2: Uncertain - angle librates for the entire duration but periodogram peaks don't match
    1: Transient - angle librates for a significant portion AND periodogram peaks match
   -1: Uncertain - angle librates for a significant portion but periodogram peaks don't match
    0: Non-resonant - angle does not librate sufficiently

The key insight is that for MMRs, the libration period of the resonant angle should match
the libration period of the semi-major axis. When both show peaks at the same frequencies,
this confirms the resonance is genuine.
"""

from typing import Optional

from resonances.resonance.periodogram import Periodogram
from resonances.resonance.classify.models import ResonanceStatus


def resolve_mmr_status(
    classification_status: int,
    angle_periodogram_peaks: Optional[dict],
    axis_periodogram_peaks: Optional[dict],
    overlap_delta: float = 0,
) -> dict:
    """Resolve the final MMR status based on classification and periodogram overlap.

    This function takes the basic classification status and periodogram peaks data,
    computes the overlap between angle and axis peaks, and returns the final status
    along with all diagnostic information.

    Parameters
    ----------
    classification_status : int
        The basic classification status (0, 1, or 2)
    angle_periodogram_peaks : dict or None
        Periodogram peaks data for the resonant angle (with 'position' key)
    axis_periodogram_peaks : dict or None
        Periodogram peaks data for the semi-major axis (with 'position' key)
    overlap_delta : float
        Tolerance for periodogram peak overlap (default: 0)

    Returns
    -------
    dict with keys:
        status : int
            Final status code
        overlapping_peaks : list
            List of overlapping peak intervals
        n_angle_peaks : int
            Number of peaks in angle periodogram
        n_axis_peaks : int
            Number of peaks in axis periodogram
        has_overlap : bool
            Whether any peaks overlap
    """
    # Extract peak positions
    angle_positions = angle_periodogram_peaks.get('position', []) if angle_periodogram_peaks else []
    axis_positions = axis_periodogram_peaks.get('position', []) if axis_periodogram_peaks else []

    # Compute overlapping peaks
    overlapping = Periodogram.overlap_list(angle_positions, axis_positions, delta=overlap_delta)
    has_overlap = len(overlapping) > 0

    # Determine final status based on classification and periodogram overlap
    S = ResonanceStatus
    if classification_status == S.NON_RESONANT:
        final_status = S.NON_RESONANT
    elif classification_status == S.LIBRATION:
        final_status = S.LIBRATION if has_overlap else S.LIBRATION_UNCERTAIN
    elif classification_status == S.TRANSIENT:
        final_status = S.TRANSIENT if has_overlap else S.TRANSIENT_UNCERTAIN
    else:
        final_status = S.NON_RESONANT

    return {
        'status': int(final_status),
        'overlapping_peaks': overlapping,
        'n_angle_peaks': len(angle_positions),
        'n_axis_peaks': len(axis_positions),
        'has_overlap': has_overlap,
    }
