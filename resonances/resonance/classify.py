"""
Resonance Angle Classification Algorithm
=========================================

PURPOSE
-------
This module classifies resonant angle time series from celestial mechanics simulations
into categories based on the dynamical behavior of the resonant argument:

    Status 2: Libration for the entire duration AND (for MMRs) periodogram peaks match
    Status -2: Libration for the entire duration but periodogram peaks don't match
    Status 1: Libration for a significant portion AND (for MMRs) periodogram peaks match
    Status -1: Libration for a significant portion but periodogram peaks don't match
    Status 0: Other cases (circulation or chaotic behavior)


CORE CONCEPT: CUMULATIVE DRIFT ANALYSIS
---------------------------------------
The fundamental distinction between libration and circulation lies in how the angle
evolves over time:

    LIBRATION: The resonant angle oscillates around a fixed center (either 0, π, or
    another equilibrium point). Even with short-period perturbations superimposed,
    the angle keeps returning to similar values. If you "unwrap" the angle (remove
    the 2π modulo), the resulting curve oscillates around a constant level.

    CIRCULATION: The resonant angle continuously increases (or decreases) over time.
    The asteroid is not captured in resonance - the resonant argument sweeps through
    all values from 0 to 2π repeatedly. When unwrapped, this produces a line with
    non-zero slope.

The key insight is that by computing the CUMULATIVE DRIFT (the unwrapped angle
progression), we transform the problem:
    - Libration → cumulative drift oscillates around zero (bounded)
    - Circulation → cumulative drift grows linearly with time


WHY THIS APPROACH WORKS FOR NOISY DATA
--------------------------------------
Your data contains strong short-period oscillations (likely from planetary
perturbations at different frequencies). These oscillations can make the raw
angle plot look chaotic, but they don't affect the cumulative drift's long-term
behavior:

    - For libration: short-period oscillations add "noise" to the cumulative drift,
      but it remains bounded because positive and negative excursions cancel out
      over time.

    - For circulation: short-period oscillations add noise, but the underlying
      linear trend persists. The oscillations cannot reverse the systematic drift.

By fitting a linear regression to the cumulative drift:
    - High R² (close to 1) → the drift is well-explained by a line → CIRCULATION
    - Low R² (close to 0) → the drift doesn't follow a line → LIBRATION


ALGORITHM STEPS
---------------
1. UNWRAP THE ANGLE (Compute Cumulative Drift)
   - For each consecutive pair of angle values, compute the "shortest path"
     difference on the circle (handling the 2π wrap-around correctly)
   - A jump from 6.2 to 0.1 is interpreted as +0.18 radians, not -6.1 radians
   - Accumulate these differences to get the cumulative drift

2. DETECT OVERALL CIRCULATION
   - Fit a linear model to cumulative_drift vs time
   - Compute R² (coefficient of determination) to measure fit quality
   - Compute total drift in units of complete cycles (2π)
   - Flag as circulation if:
     * Total drift exceeds threshold (default: 2 cycles), OR
     * Strong linear trend exists (slope > 1.5 cycles AND R² > 0.3)

3. FIND LOCAL LIBRATION SEGMENTS (for partial libration detection)
   - Use sliding windows to identify periods where behavior is libration-like
   - Within each window, compare net drift to oscillation amplitude
   - If |net drift| < amplitude/2, the window shows libration behavior

4. FINAL CLASSIFICATION
   - If circulation detected AND R² > threshold (default 0.85):
     → Status 0 (definite circulation, no false positive)
   - If circulation detected but R² lower, check libration fraction:
     → Status 1 if significant local libration exists
     → Status 0 otherwise
   - If no circulation detected:
     → Status 2 if ≥90% of time shows libration
     → Status 1 if ≥20% of time shows libration
     → Status 0 otherwise

5. FOR MMRs: CHECK PERIODOGRAM OVERLAP
   - Compare periodogram peaks of resonant angle and semi-major axis
   - Modify status based on whether frequencies match


Authors: Evgeny Smirnov + Claude (Anthropic)
"""

import numpy as np
from typing import Tuple, List

from resonances.body import Body
from resonances.resonance.resolver import resolve_mmr_status
from resonances.resonance.resonance import Resonance
from resonances.mmr.mmr import MMR


def compute_angle_difference(angle1: float, angle2: float) -> float:
    """
    Compute the signed difference between two angles, handling 2π wrapping.

    Returns the shortest path on the circle from angle1 to angle2.
    Result is in the range [-π, π].

    Examples:
        compute_angle_difference(0.1, 0.3) → +0.2  (small forward step)
        compute_angle_difference(6.1, 0.2) → +0.38 (crossing 2π boundary forward)
        compute_angle_difference(0.2, 6.1) → -0.38 (crossing 2π boundary backward)
    """
    diff = angle2 - angle1
    # Wrap to [-π, π] using modular arithmetic
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff


def compute_cumulative_drift(angles: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative drift of the angle over time.

    This "unwraps" the angle by accumulating the shortest-path differences
    between consecutive points. The result shows how far the angle has
    drifted from its starting position (in radians, unbounded).

    For libration: cumulative drift oscillates around zero
    For circulation: cumulative drift grows/decreases monotonically
    """
    n = len(angles)
    cumulative = np.zeros(n)

    for i in range(1, n):
        diff = compute_angle_difference(angles[i - 1], angles[i])
        cumulative[i] = cumulative[i - 1] + diff

    return cumulative


def analyze_cumulative_drift(cumulative: np.ndarray, times: np.ndarray, circulation_threshold_cycles: float) -> Tuple[bool, float, float]:
    """
    Analyze cumulative drift to determine if it shows circulation.

    Fits a linear model to the cumulative drift and computes R² to
    measure how well the drift follows a straight line.

    Returns:
        is_circulation: True if circulation detected
        drift_rate: Net drift rate (radians per unit time)
        r_squared: Coefficient of determination for linear fit
    """
    # Calculate total drift
    total_drift = cumulative[-1] - cumulative[0]
    total_time = times[-1] - times[0]

    # Net drift rate (radians per unit time)
    drift_rate = total_drift / total_time if total_time > 0 else 0

    # Total drift in cycles
    total_cycles = abs(total_drift) / (2 * np.pi)

    # Fit a linear trend to the cumulative drift
    # Normalize time to [0, 1] for numerical stability
    times_normalized = (times - times[0]) / (times[-1] - times[0])

    # Simple linear regression: cumulative = intercept + slope * time
    mean_t = np.mean(times_normalized)
    mean_c = np.mean(cumulative)

    numerator = np.sum((times_normalized - mean_t) * (cumulative - mean_c))
    denominator = np.sum((times_normalized - mean_t) ** 2)

    if denominator > 0:
        slope = numerator / denominator
        intercept = mean_c - slope * mean_t

        # Predicted values from linear model
        predicted = intercept + slope * times_normalized

        # R-squared: fraction of variance explained by linear model
        ss_res = np.sum((cumulative - predicted) ** 2)  # Residual sum of squares
        ss_tot = np.sum((cumulative - mean_c) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        slope = 0
        r_squared = 0

    # Slope in cycles per normalized time unit
    slope_cycles = abs(slope) / (2 * np.pi)

    # Circulation detection criteria:
    # 1. Total drift exceeds threshold cycles, OR
    # 2. Significant slope with moderate linear fit quality
    is_circulation = total_cycles > circulation_threshold_cycles or (slope_cycles > 1.5 and r_squared > 0.3)

    return is_circulation, drift_rate, r_squared


def find_libration_segments(
    cumulative: np.ndarray, times: np.ndarray, window_fraction: float, min_window_points: int, max_libration_drift: float
) -> List[Tuple[int, int]]:
    """
    Find segments where the behavior looks like libration (bounded oscillation).

    Uses a sliding window approach: within each window, compares the net drift
    to the oscillation amplitude. If net drift is small compared to amplitude,
    the window shows libration-like behavior.
    """
    n = len(cumulative)

    # Determine window size
    window_size = max(min_window_points, int(n * window_fraction))

    # Track which points are in libration-like regions
    libration_mask = np.zeros(n, dtype=bool)

    # Slide through the data
    step = max(1, window_size // 10)  # Step size for efficiency

    for start in range(0, n - window_size, step):
        end = start + window_size
        window_cumulative = cumulative[start:end]

        # Net drift within window
        window_drift = window_cumulative[-1] - window_cumulative[0]

        # Oscillation amplitude within window
        window_range = np.max(window_cumulative) - np.min(window_cumulative)

        # Libration criterion:
        # Net drift should be small compared to oscillation range
        # This catches libration even with some secular drift
        is_libration_window = abs(window_drift) < max(window_range * 0.5, np.pi) and abs(window_drift) < max_libration_drift

        if is_libration_window:
            libration_mask[start:end] = True

    # Convert boolean mask to list of (start, end) segments
    segments = []
    in_segment = False
    segment_start = 0

    for i in range(n):
        if libration_mask[i] and not in_segment:
            segment_start = i
            in_segment = True
        elif not libration_mask[i] and in_segment:
            segments.append((segment_start, i))
            in_segment = False

    if in_segment:
        segments.append((segment_start, n))

    return segments


def compute_libration_fraction(segments: List[Tuple[int, int]], n_total: int) -> float:
    """Compute the fraction of time spent in libration-like segments."""
    if not segments:
        return 0.0

    libration_points = sum(end - start for start, end in segments)
    return libration_points / n_total


def classify_resonance(
    body: Body,
    times: np.ndarray,
    resonance: Resonance,
    min_libration_fraction: float = 0.20,
    circulation_threshold_cycles: float = 2.0,
    r_squared_circulation_threshold: float = 0.85,
    window_fraction: float = 0.1,
    min_window_points: int = 100,
    max_libration_drift: float = 2.0 * np.pi,
    overlap_delta: float = 0,
) -> dict:
    """
    Classify with detailed diagnostics.

    Returns a dictionary with classification result and diagnostic information
    useful for debugging and understanding the classification decision.

    Parameters
    ----------
    body : Body
        Body object with the resonant angle and periodogram data
    times : np.ndarray
        Array of time values (x-axis of the plot)
    resonance : Resonance
        The resonance object (used to determine if it's an MMR)
    min_libration_fraction : float
        Minimum fraction of time in libration to qualify for status 1
    circulation_threshold_cycles : float
        Number of drift cycles to trigger circulation detection
    r_squared_circulation_threshold : float
        R² above which circulation is definite (no false positive)
    window_fraction : float
        Sliding window size as fraction of total data
    min_window_points : int
        Minimum points per analysis window
    max_libration_drift : float
        Maximum drift within a window for libration classification
    overlap_delta : float
        Tolerance for periodogram peak overlap (default: 0)

    Returns
    -------
    dict with keys:
        status : int
            Final status code (including periodogram check for MMRs)
        classification_status : int
            Basic classification (0, 1, or 2) before periodogram check
        libration_fraction : float
            Fraction of time in libration-like behavior
        is_circulation : bool
            Whether circulation was detected
        drift_rate : float
            Net drift rate (radians per time unit)
        r_squared : float
            R² of linear fit to cumulative drift
        total_drift_cycles : float
            Total drift in units of 2π
        n_libration_segments : int
            Number of distinct libration segments found
        cumulative_drift : np.ndarray
            The computed cumulative drift array
        overlapping_peaks : list
            List of overlapping peak intervals (MMR only)
        n_angle_peaks : int
            Number of peaks in angle periodogram (MMR only)
        n_axis_peaks : int
            Number of peaks in axis periodogram (MMR only)
        has_overlap : bool
            Whether any peaks overlap (MMR only)
    """
    cumulative = compute_cumulative_drift(
        body.angles[resonance.to_s()]
    )  # using non-filtered angle to avoid false positives with circulation
    is_circulation, drift_rate, r_squared = analyze_cumulative_drift(cumulative, times, circulation_threshold_cycles)
    segments = find_libration_segments(cumulative, times, window_fraction, min_window_points, max_libration_drift)
    libration_fraction = compute_libration_fraction(segments, len(times))

    if is_circulation:
        # Strong linear drift (high R²) is definitely circulation
        if r_squared > r_squared_circulation_threshold:
            classification_status = 0  # Clear circulation, no false positive allowed
        # Weaker circulation signal - check for partial libration
        elif libration_fraction >= min_libration_fraction:
            classification_status = 1  # Significant libration despite overall circulation
        else:
            classification_status = 0  # Circulation dominates
    else:
        # No strong overall circulation - classify by libration fraction
        if libration_fraction >= 0.9:  # 90% or more in libration
            classification_status = 2  # Full libration
        elif libration_fraction >= min_libration_fraction:
            classification_status = 1  # Partial libration
        else:
            classification_status = 0  # Chaotic or insufficient libration

    # Step 6: For MMRs, resolve final status with periodogram overlap check
    is_mmr = isinstance(resonance, MMR)

    if is_mmr:
        resolver_result = resolve_mmr_status(
            classification_status=classification_status,
            angle_periodogram_peaks=body.periodogram_peaks.get(resonance.to_s()),
            axis_periodogram_peaks=body.axis_periodogram_peaks,
            overlap_delta=overlap_delta,
        )
        final_status = resolver_result['status']
        overlapping_peaks = resolver_result['overlapping_peaks']
        n_angle_peaks = resolver_result['n_angle_peaks']
        n_axis_peaks = resolver_result['n_axis_peaks']
        has_overlap = resolver_result['has_overlap']
    else:
        # For non-MMRs (secular resonances), no periodogram check
        final_status = classification_status
        overlapping_peaks = []
        n_angle_peaks = 0
        n_axis_peaks = 0
        has_overlap = False

    return {
        'status': final_status,
        'classification_status': classification_status,
        'libration_fraction': libration_fraction,
        'is_circulation': is_circulation,
        'drift_rate': drift_rate,
        'r_squared': r_squared,
        'total_drift_cycles': abs(cumulative[-1] - cumulative[0]) / (2 * np.pi),
        'n_libration_segments': len(segments),
        'cumulative_drift': cumulative,
        'overlapping_peaks': overlapping_peaks,
        'n_angle_peaks': n_angle_peaks,
        'n_axis_peaks': n_axis_peaks,
        'has_overlap': has_overlap,
    }
