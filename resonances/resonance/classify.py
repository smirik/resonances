"""
Resonance Angle Classification Algorithm
=========================================

PURPOSE
-------
This module classifies resonant angle time series from celestial mechanics simulations
into categories based on the dynamical behavior of the resonant argument:

    Status 2: Pure libration (entire duration) AND (for MMRs) periodogram peaks match
    Status -2: Pure libration but periodogram peaks don't match
    Status 1: Transient libration (≥20% of time) AND (for MMRs) periodogram peaks match
    Status -1: Transient libration but periodogram peaks don't match
    Status 0: Non-resonant (circulation or chaotic behavior)


CORE CONCEPT: CUMULATIVE DRIFT ANALYSIS
---------------------------------------
The fundamental distinction between libration and circulation lies in how the angle
evolves over time:

    LIBRATION: The resonant angle oscillates around a fixed center. When "unwrapped"
    (cumulative drift computed), it oscillates around a constant level.

    CIRCULATION: The resonant angle continuously increases/decreases. When unwrapped,
    it produces a line with non-zero slope.

By fitting a linear regression to the cumulative drift:
    - High R² (close to 1) → linear drift → CIRCULATION
    - Low R² (close to 0) → bounded oscillation → LIBRATION


ALGORITHM (SLIDING WINDOW APPROACH)
-----------------------------------
1. CHECK FOR PURE LIBRATION (Status 2)
   Apply check_libration() to the full time series. Criteria:
   - No circulation detected (total drift < 2 cycles)
   - Low total drift cycles (< 1.2)
   - Moderate uniformity (< 0.5)
   - One of:
     * Very low R² (< 0.2) with very high libration fraction (≥ 0.95)
     * Low R² (< 0.8) with high lib_frac (≥ 0.9) and low uniformity (< 0.14)
     * Very low R² (< 0.1) with low uniformity (< 0.15) - for apocentric libration

2. CHECK FOR CLEAR NON-RESONANT (Status 0)
   Filter out cases where sliding windows would incorrectly detect libration:
   - Slow circulation: R² > 0.97 and cycles > 1.5
   - Chaotic: uniformity > 0.7 and libration fraction < 0.5
   - High-cycle chaotic: circulation with > 25 cycles and uniformity > 0.63

3. SLIDING WINDOW DETECTION (for Transient)
   - Window width: 10% of total time
   - Window shift: 2% of total time
   - Apply check_libration() with is_window=True to each window
   - Window libration criteria:
     * Base criteria (same as status 2) AND
     * Normalized drift (drift_norm) below adaptive threshold:
       - uniformity < 0.1:  drift_norm < 0.30 (relaxed for concentrated angles)
       - uniformity < 0.45: drift_norm < 0.25 (moderate)
       - uniformity >= 0.45: drift_norm < 0.15 (strict for spread angles)
   - Track which time points are covered by ANY libration window
   - Merge overlapping windows to get total libration time

4. FINAL CLASSIFICATION
   - If ≥ 20% of time is in libration windows → Status 1 (transient)
   - Otherwise → Status 0 (non-resonant)

5. FOR MMRs: PERIODOGRAM CHECK
   Compare periodogram peaks of resonant angle and semi-major axis.
   If no overlap, negate the status (2 → -2, 1 → -1).


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


def compute_drift_norm(cumulative: np.ndarray) -> float:
    """
    Compute the normalized cumulative drift metric.

    This metric divides the standard deviation of cumulative drift by sqrt(N),
    which accounts for the random walk behavior. For a random walk, std grows
    as sqrt(N), so drift_norm stays roughly constant regardless of series length.

    For libration: drift_norm is small (bounded oscillation)
    For circulation/chaos: drift_norm is larger (unbounded drift)

    The threshold drift_norm < 0.086 provides perfect separation for status 2 (libration)
    when applied to the full time series.
    """
    n = len(cumulative)
    if n == 0:
        return 0.0
    return np.std(cumulative) / np.sqrt(n)


def compute_circulation_segment_fraction(
    angles: np.ndarray,
    n_segments: int = 10,
    full_range_threshold: float = 5.5,
    high_drift_threshold: float = 0.7,
) -> float:
    """
    Compute the fraction of segments showing circulation behavior.

    A segment shows circulation if it has BOTH:
    1. Full angle range (spans nearly 0-2π)
    2. High drift (net change > 0.5 cycles in one direction)

    This distinguishes between:
    - Apocentric libration: full range but drift ~0 (oscillates back and forth)
    - Circulation/transient: full range AND high drift (moves in one direction)

    Args:
        angles: Array of angle values
        n_segments: Number of equal segments to divide the data into (default: 10)
        full_range_threshold: Angle range above which a segment spans full range
                             (default: 5.5 radians, ~87% of 2π)
        high_drift_threshold: Drift in cycles above which segment shows circulation
                             (default: 0.7 cycles - allows for noisy libration)

    Returns:
        Fraction of segments showing circulation (0.0 to 1.0)
    """
    n = len(angles)
    if n < n_segments:
        return 0.0

    # Compute cumulative drift for the full series
    cumulative = np.zeros(n)
    for i in range(1, n):
        diff = angles[i] - angles[i - 1]
        # Wrap to [-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        cumulative[i] = cumulative[i - 1] + diff

    circulation_count = 0
    for i in range(n_segments):
        start = int(n * i / n_segments)
        end = int(n * (i + 1) / n_segments)
        seg_angles = angles[start:end]
        angle_range = np.max(seg_angles) - np.min(seg_angles)

        # Calculate drift in this segment (in cycles)
        drift_in_seg = abs(cumulative[end - 1] - cumulative[start]) / (2 * np.pi)

        # Circulation: full range AND high drift
        if angle_range > full_range_threshold and drift_in_seg > high_drift_threshold:
            circulation_count += 1

    return circulation_count / n_segments


def compute_angle_uniformity(angles: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute how uniformly the angles are distributed across 0 to 2π.

    Returns the ratio of minimum to maximum bin counts. High uniformity (close to 1)
    indicates chaotic or circulation behavior where the angle visits all values equally.
    Low uniformity (close to 0) indicates libration where the angle concentrates
    around certain values.

    This metric is used in classification:
    - uniformity > 0.7: chaotic detection threshold (combined with lib_frac/r_sq checks)
    - uniformity > 0.5: slow circulation edge case detection (combined with high R²)
    - uniformity < 0.14: required for status 2 with moderate R² (concentrated angles)
    - uniformity can be high for large-amplitude librations if R² is very low (<0.2)
    """
    hist, _ = np.histogram(angles, bins=n_bins, range=(0, 2 * np.pi))
    if hist.max() == 0:
        return 0.0
    return hist.min() / hist.max()


def check_libration(
    times: np.ndarray,
    angles: np.ndarray,
    circulation_threshold_cycles: float = 2.0,
    max_drift_cycles: float = 1.2,
    window_fraction: float = 0.1,
    min_window_points: int = 100,
    max_libration_drift: float = 2.0 * np.pi,
    window_drift_norm_threshold: float = 0.15,
    is_window: bool = False,
) -> Tuple[bool, dict]:
    """
    Check if the angle time series shows libration (status 2) characteristics.

    This is the core libration detection function used for both:
    - Full time series check (status 2)
    - Sliding window checks (status 1 detection)

    Uses the status 2 criteria:
    - No circulation detected
    - Low total drift cycles
    - Moderate uniformity (< 0.5) - angles must be somewhat concentrated, not randomly scattered
    - Either: (R² < 0.2 and lib_frac >= 0.95) or (R² < 0.8 and lib_frac >= 0.9 and uniformity < 0.14)
             or (R² < 0.1 and uniformity < 0.15) for apocentric libration

    For sliding windows (is_window=True), also requires:
    - Normalized drift (drift_norm) below adaptive threshold based on uniformity:
      * uniformity < 0.1:  drift_norm < 0.30 (relaxed)
      * uniformity < 0.45: drift_norm < 0.25 (moderate)
      * uniformity >= 0.45: drift_norm < 0.15 (strict)
    - This helps reject chaotic windows that accidentally pass base criteria

    Returns
    -------
    Tuple of (is_libration, diagnostics)
    """
    cumulative = compute_cumulative_drift(angles)
    is_circulation, drift_rate, r_squared = analyze_cumulative_drift(cumulative, times, circulation_threshold_cycles)
    segments = find_libration_segments(cumulative, times, window_fraction, min_window_points, max_libration_drift)
    libration_fraction = compute_libration_fraction(segments, len(times))
    angle_uniformity = compute_angle_uniformity(angles)
    total_drift_cycles = abs(cumulative[-1] - cumulative[0]) / (2 * np.pi)
    drift_norm = compute_drift_norm(cumulative)
    circulation_segment_fraction = compute_circulation_segment_fraction(angles) if not is_window else 0.0

    diagnostics = {
        'is_circulation': is_circulation,
        'drift_rate': drift_rate,
        'r_squared': r_squared,
        'libration_fraction': libration_fraction,
        'angle_uniformity': angle_uniformity,
        'total_drift_cycles': total_drift_cycles,
        'n_libration_segments': len(segments),
        'cumulative_drift': cumulative,
        'drift_norm': drift_norm,
        'circulation_segment_fraction': circulation_segment_fraction,
    }

    # Core libration criteria (used for both full series and windows)
    # Uniformity < 0.5 ensures angles are somewhat concentrated (not chaotic)
    meets_base_criteria = (
        not is_circulation
        and total_drift_cycles < max_drift_cycles
        and angle_uniformity < 0.5
        and (
            (r_squared < 0.2 and libration_fraction >= 0.95)
            or (r_squared < 0.8 and libration_fraction >= 0.9 and angle_uniformity < 0.14)
            or (r_squared < 0.1 and angle_uniformity < 0.15)
        )
    )

    # For sliding windows, add normalized drift constraint
    # This filters out chaotic windows that pass base criteria by chance
    # Use adaptive threshold: very concentrated angles (low uniformity) allow higher drift_norm
    if is_window:
        # Stricter drift_norm threshold for windows with moderate uniformity
        # Very low uniformity (< 0.1) indicates strong libration, allow higher drift_norm
        if angle_uniformity < 0.1:
            drift_threshold = 0.30  # Relaxed for very concentrated angles
        elif angle_uniformity < 0.45:
            drift_threshold = 0.25  # Moderate threshold
        else:
            drift_threshold = window_drift_norm_threshold  # 0.15 - strict for less concentrated
        is_libration = meets_base_criteria and drift_norm < drift_threshold
        return is_libration, diagnostics

    # For full time series: use base criteria + check for hidden circulation episodes
    # If > 30% of segments show circulation (full range + high drift), reject pure libration
    # This catches transient cases where circulation episodes cancel out in total drift
    has_hidden_circulation = circulation_segment_fraction >= 0.3
    is_libration = meets_base_criteria and not has_hidden_circulation
    return is_libration, diagnostics


def check_clear_circulation(
    is_circulation: bool,
    r_squared: float,
    total_drift_cycles: float,
    angle_uniformity: float,
    libration_fraction: float,
    r_squared_definite_threshold: float = 0.95,
    chaotic_uniformity_threshold: float = 0.7,
) -> bool:
    """
    Check if the global metrics indicate clear circulation/non-resonant behavior.

    This filters out cases where sliding window would incorrectly detect libration
    in slow circulation or chaotic data.
    """
    # Slow steady circulation: VERY high R² (>0.97) + significant cycles
    is_slow_circulation = r_squared > 0.97 and total_drift_cycles > 1.5

    # Chaotic detection: high uniformity means angles spread uniformly across 0-2π
    # Three tiers based on uniformity level:
    # - Extremely high uniformity (>0.84): likely transient, not chaotic (protect these)
    # - Very high uniformity (0.74-0.84): chaotic if circulation detected
    # - High uniformity (0.7-0.74): need circulation + cycles > 12
    is_very_high_uniformity_chaotic = angle_uniformity > 0.74 and angle_uniformity <= 0.84 and is_circulation
    is_high_uniformity_chaotic = (
        angle_uniformity > chaotic_uniformity_threshold
        and angle_uniformity <= 0.74
        and (libration_fraction < 0.5 or (is_circulation and total_drift_cycles > 12))
    )
    is_chaotic = is_very_high_uniformity_chaotic or is_high_uniformity_chaotic

    # High-cycle chaotic: many cycles with moderate uniformity
    # Threshold of 25 cycles protects real transients (max ~22 cycles) while catching chaotic cases
    is_high_cycle_chaotic = is_circulation and total_drift_cycles > 25 and angle_uniformity > 0.63

    # Slow circulation without is_circ flag: high R² + high uniformity
    is_slow_no_circ = not is_circulation and r_squared > 0.95 and angle_uniformity > 0.7

    return is_slow_circulation or is_chaotic or is_high_cycle_chaotic or is_slow_no_circ


def classify_angle(  # noqa: C901
    times: np.ndarray,
    angles: np.ndarray,
    circulation_threshold_cycles: float = 2.0,
    r_squared_definite_threshold: float = 0.95,
    window_fraction: float = 0.1,
    min_window_points: int = 100,
    max_libration_drift: float = 2.0 * np.pi,
    chaotic_uniformity_threshold: float = 0.7,
    pure_libration_max_cycles: float = 1.2,
    sliding_window_width: float = 0.10,
    sliding_window_shift: float = 0.02,
) -> dict:
    """
    Classify the resonant angle time series using a sliding window approach.

    Algorithm:
    1. Check if the entire time series shows pure libration (status 2)
    2. Check for clear circulation/chaotic behavior (status 0)
    3. Use sliding windows to detect transient libration periods
    4. If >= 20% of time is in libration windows → status 1 (transient)
    5. Otherwise → status 0 (non-resonant)

    Parameters
    ----------
    times : np.ndarray
        Array of time values
    angles : np.ndarray
        Array of resonant angle values (in radians, 0 to 2π)
    sliding_window_width : float
        Width of sliding window as fraction of total time (default: 0.10 = 10%)
    sliding_window_shift : float
        Shift of sliding window as fraction of total time (default: 0.02 = 2%)

    Returns
    -------
    dict with classification result and diagnostic information
    """
    # Step 1: Check for pure libration (status 2) on full time series
    is_pure_libration, diagnostics = check_libration(
        times,
        angles,
        circulation_threshold_cycles=circulation_threshold_cycles,
        max_drift_cycles=pure_libration_max_cycles,
        window_fraction=window_fraction,
        min_window_points=min_window_points,
        max_libration_drift=max_libration_drift,
    )

    if is_pure_libration:
        return {
            'status': 2,
            'classification_status': 2,
            'n_libration_windows': 0,
            'n_total_windows': 0,
            'libration_window_fraction': 0.0,
            **diagnostics,
        }

    # Step 2: Check for clear circulation (status 0) based on global metrics
    is_clear_circulation = check_clear_circulation(
        is_circulation=diagnostics['is_circulation'],
        r_squared=diagnostics['r_squared'],
        total_drift_cycles=diagnostics['total_drift_cycles'],
        angle_uniformity=diagnostics['angle_uniformity'],
        libration_fraction=diagnostics['libration_fraction'],
        r_squared_definite_threshold=r_squared_definite_threshold,
        chaotic_uniformity_threshold=chaotic_uniformity_threshold,
    )

    if is_clear_circulation:
        return {
            'status': 0,
            'classification_status': 0,
            'n_libration_windows': 0,
            'n_total_windows': 0,
            'libration_window_fraction': 0.0,
            **diagnostics,
        }

    # Step 3: Sliding window approach for transient detection
    n_points = len(times)
    window_size = max(min_window_points, int(n_points * sliding_window_width))
    shift_size = max(1, int(n_points * sliding_window_shift))

    # Track which points are covered by at least one libration window
    libration_coverage = np.zeros(n_points, dtype=bool)
    n_libration_windows = 0
    n_total_windows = 0

    start = 0
    while start + window_size <= n_points:
        end = start + window_size

        window_times = times[start:end]
        window_angles = angles[start:end]

        n_total_windows += 1

        has_libration, _ = check_libration(
            window_times,
            window_angles,
            circulation_threshold_cycles=circulation_threshold_cycles,
            max_drift_cycles=pure_libration_max_cycles,
            window_fraction=window_fraction,
            min_window_points=min_window_points,
            max_libration_drift=max_libration_drift,
            is_window=True,
        )

        if has_libration:
            libration_coverage[start:end] = True
            n_libration_windows += 1

        start += shift_size

    # Calculate total libration time fraction (non-overlapping)
    total_libration_time_frac = np.sum(libration_coverage) / n_points

    # Find contiguous libration periods (merged from overlapping windows)
    libration_periods = []
    in_libration = False
    period_start_idx = 0
    for i in range(n_points):
        if libration_coverage[i] and not in_libration:
            period_start_idx = i
            in_libration = True
        elif not libration_coverage[i] and in_libration:
            libration_periods.append(
                {
                    'start_idx': period_start_idx,
                    'end_idx': i,
                    'start_time': times[period_start_idx],
                    'end_time': times[i - 1],
                    'start_frac': period_start_idx / n_points,
                    'end_frac': i / n_points,
                    'duration_frac': (i - period_start_idx) / n_points,
                }
            )
            in_libration = False
    if in_libration:
        libration_periods.append(
            {
                'start_idx': period_start_idx,
                'end_idx': n_points,
                'start_time': times[period_start_idx],
                'end_time': times[-1],
                'start_frac': period_start_idx / n_points,
                'end_frac': 1.0,
                'duration_frac': (n_points - period_start_idx) / n_points,
            }
        )

    # Step 4: Final classification
    diagnostics['n_libration_windows'] = n_libration_windows
    diagnostics['n_total_windows'] = n_total_windows
    diagnostics['libration_window_fraction'] = n_libration_windows / n_total_windows if n_total_windows > 0 else 0.0
    diagnostics['total_libration_time_frac'] = total_libration_time_frac
    diagnostics['libration_periods'] = libration_periods

    # Require >= 20% of time in libration for transient status
    min_libration_time_frac = 0.20
    if total_libration_time_frac >= min_libration_time_frac:
        classification_status = 1  # Transient resonance
    else:
        classification_status = 0  # Non-resonant

    return {
        'status': classification_status,
        'classification_status': classification_status,
        **diagnostics,
    }


def classify_resonance(
    body: Body,
    times: np.ndarray,
    resonance: Resonance,
    circulation_threshold_cycles: float = 2.0,
    r_squared_definite_threshold: float = 0.95,
    window_fraction: float = 0.1,
    min_window_points: int = 100,
    max_libration_drift: float = 2.0 * np.pi,
    overlap_delta: float = 0,
    chaotic_uniformity_threshold: float = 0.7,
    pure_libration_max_cycles: float = 1.2,
) -> dict:
    """
    Classify resonance with detailed diagnostics.

    Parameters
    ----------
    body : Body
        Body object with the resonant angle and periodogram data
    times : np.ndarray
        Array of time values (x-axis of the plot)
    resonance : Resonance
        The resonance object (used to determine if it's an MMR)
    circulation_threshold_cycles : float
        Number of drift cycles to trigger circulation detection (default: 2.0)
    r_squared_definite_threshold : float
        R² above which slow circulation is detected (default: 0.95)
    window_fraction : float
        Sliding window size as fraction of total data (default: 0.1)
    min_window_points : int
        Minimum points per analysis window (default: 100)
    max_libration_drift : float
        Maximum drift within a window for libration classification (default: 2π)
    overlap_delta : float
        Tolerance for periodogram peak overlap (default: 0)
    chaotic_uniformity_threshold : float
        Uniformity threshold for chaotic detection (default: 0.7)
    pure_libration_max_cycles : float
        Maximum drift cycles allowed for status 2 (default: 1.2)

    Returns
    -------
    dict with classification result and diagnostic information
    """
    angles = body.angles[resonance.to_s()]
    classification = classify_angle(
        times,
        angles,
        circulation_threshold_cycles,
        r_squared_definite_threshold,
        window_fraction,
        min_window_points,
        max_libration_drift,
        chaotic_uniformity_threshold,
        pure_libration_max_cycles,
    )

    # For MMRs, resolve final status with periodogram overlap check
    is_mmr = isinstance(resonance, MMR)

    if is_mmr:
        resolver_result = resolve_mmr_status(
            classification_status=classification['status'],
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
        final_status = classification['status']
        overlapping_peaks = []
        n_angle_peaks = 0
        n_axis_peaks = 0
        has_overlap = False

    return {
        **classification,
        'status': final_status,
        'overlapping_peaks': overlapping_peaks,
        'n_angle_peaks': n_angle_peaks,
        'n_axis_peaks': n_axis_peaks,
        'has_overlap': has_overlap,
    }
