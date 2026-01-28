import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from enum import IntEnum
from astropy.timeseries import LombScargle


class ResonanceStatus(IntEnum):
    """Статусы классификации резонанса."""

    STICKINESS = 3
    LIBRATION = 2
    TRANSIENT = 1
    CIRCULATION = 0
    TRANSIENT_UNCERTAIN = -1
    LIBRATION_UNCERTAIN = -2
    UNCERTAIN = -4
    CHAOTIC = -99


def _calc_sigma_dot(times: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Calculate the derivative of the resonant angle using central differences."""
    sigma_dot = np.zeros_like(sigma)
    sigma_dot[1:-1] = (sigma[2:] - sigma[:-2]) / (times[2:] - times[:-2])
    dt = np.diff(times)
    sigma_dot[0] = (sigma[1] - sigma[0]) / dt[0]
    sigma_dot[-1] = (sigma[-1] - sigma[-2]) / dt[-1]
    return sigma_dot


def _calc_zero_crossing_stats(sigma_dot: np.ndarray) -> Tuple[int, float]:
    """Computes the number of zero crossings and the CV of the inter-zero-crossing intervals."""
    # sign_changes = np.where(np.diff(np.sign(sigma_dot)) != 0)[0]
    sign_changes = np.where(sigma_dot[:-1] * sigma_dot[1:] < 0)[0]
    n_crossings = len(sign_changes)

    if n_crossings < 2:
        return n_crossings, np.inf

    intervals = np.diff(sign_changes)
    if len(intervals) == 0 or np.mean(intervals) == 0:
        return n_crossings, np.inf

    return n_crossings, np.std(intervals) / np.mean(intervals)


def _calc_periodogram(
    times: np.ndarray,
    sigma_wrapped: np.ndarray,
    min_period: Optional[float] = None,
    max_period: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculates Lomb-Scargle for cos(σ) and sin(σ)."""
    if len(times) < 10:
        return None, None, None

    dt_total = times[-1] - times[0]
    dt_median = np.median(np.diff(times))

    if min_period is None:
        min_period = 2 * dt_median
    if max_period is None:
        max_period = dt_total / 2

    if min_period >= max_period:
        return None, None, None

    cos_sigma = np.cos(sigma_wrapped)
    sin_sigma = np.sin(sigma_wrapped)

    ls_cos = LombScargle(times, cos_sigma)
    ls_sin = LombScargle(times, sin_sigma)

    freq, power_cos = ls_cos.autopower(minimum_frequency=1 / max_period, maximum_frequency=1 / min_period, samples_per_peak=5)
    _, power_sin = ls_sin.autopower(minimum_frequency=1 / max_period, maximum_frequency=1 / min_period, samples_per_peak=5)

    power = power_cos + power_sin
    idx = np.argmax(power)

    period = 1 / freq[idx]
    fap = ls_cos.false_alarm_probability(power_cos[idx])
    snr = power[idx] / np.mean(power) if np.mean(power) > 0 else 0

    return period, fap, snr


def _calc_metrics(
    times: np.ndarray, sigma_wrapped: np.ndarray, sigma_unwrapped: np.ndarray, compute_periodogram: bool = True
) -> Dict[str, Any]:
    """Calculates all metrics."""

    N = len(times)
    dt_total = times[-1] - times[0]

    cos_avg = np.mean(np.cos(sigma_wrapped))
    sin_avg = np.mean(np.sin(sigma_wrapped))
    R = np.sqrt(cos_avg**2 + sin_avg**2)
    phi_rad = np.arctan2(sin_avg, cos_avg)
    phi_deg = np.rad2deg(phi_rad) % 360

    amplitude = np.max(sigma_unwrapped) - np.min(sigma_unwrapped)

    displacement = sigma_unwrapped[-1] - sigma_unwrapped[0]
    revolutions = displacement / (2 * np.pi)

    sigma_dot = _calc_sigma_dot(times, sigma_unwrapped)
    frac_positive = np.mean(sigma_dot > 0)
    sign_dominance = max(frac_positive, 1 - frac_positive)

    n_zero_crossings, cv_intervals = _calc_zero_crossing_stats(sigma_dot)

    ls_period, ls_fap, ls_snr = None, None, None
    if compute_periodogram:
        ls_period, ls_fap, ls_snr = _calc_periodogram(times, sigma_wrapped)

    return {
        'R': R,
        'phi_rad': phi_rad,
        'phi_deg': phi_deg,
        'circular_variance': 1 - R,
        'amplitude': amplitude,
        'revolutions': revolutions,
        'displacement': displacement,
        'sign_dominance': sign_dominance,
        'frac_positive': frac_positive,
        'mean_sigma_dot': np.mean(sigma_dot),
        'std_sigma_dot': np.std(sigma_dot),
        'n_zero_crossings': n_zero_crossings,
        'cv_intervals': cv_intervals,
        'ls_period': ls_period,
        'ls_fap': ls_fap,
        'ls_snr': ls_snr,
        'N': N,
        'dt_total': dt_total,
    }


def _classify_segment(
    times: np.ndarray,
    sigma_wrapped: np.ndarray,
    sigma_unwrapped: np.ndarray,
    total_duration: float,
    circulation_threshold_cycles: float,
    sign_dominance_threshold: float,
    ls_fap_threshold: float,
    ls_snr_threshold: float,
    is_window: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Classify segment (time series or window) as LIBRATION/CIRCULATION/UNCERTAIN.
    """
    segment_duration = times[-1] - times[0]
    duration_ratio = segment_duration / total_duration

    # Metrics
    amplitude = np.max(sigma_unwrapped) - np.min(sigma_unwrapped)
    displacement = sigma_unwrapped[-1] - sigma_unwrapped[0]
    revolutions = displacement / (2 * np.pi)

    sigma_dot = _calc_sigma_dot(times, sigma_unwrapped)
    frac_positive = np.mean(sigma_dot > 0)
    sign_dominance = max(frac_positive, 1 - frac_positive)

    # Threshold are proportional
    rev_threshold_lib = 0.5 * duration_ratio
    rev_threshold_circ = circulation_threshold_cycles * duration_ratio

    # Resolvers
    votes_libration = 0
    votes_circulation = 0

    # A1: amplitude < 2π -> only globally, not for windows
    if not is_window and amplitude < 2 * np.pi:
        votes_libration += 1

    # A2: |rev| < threshold (proportional) -> LIBRATION
    if abs(revolutions) < rev_threshold_lib:
        votes_libration += 1

    # A3: |rev| > threshold (for circulation) -> CIRCULATION
    if abs(revolutions) > rev_threshold_circ:
        votes_circulation += 1

    # A4: sign_dominance > 0.95 -> CIRCULATION
    if sign_dominance > sign_dominance_threshold:
        votes_circulation += 1

    # C1: Lomb-Scargle — only if |rev| is moderate (otherwise does not make sense)
    if len(times) >= 50 and abs(revolutions) < rev_threshold_lib * 2:
        period, fap, snr = _calc_periodogram(times, sigma_wrapped)
        if fap is not None and fap < ls_fap_threshold and snr is not None and snr > ls_snr_threshold:
            votes_libration += 1

    info = {
        't_start': times[0],
        't_end': times[-1],
        'amplitude': amplitude,
        'revolutions': revolutions,
        'sign_dominance': sign_dominance,
        'votes_libration': votes_libration,
        'votes_circulation': votes_circulation,
    }

    # Resolving
    if votes_libration >= 2 and votes_circulation == 0:
        return 'LIBRATION', info
    if votes_circulation >= 1:
        return 'CIRCULATION', info
    if votes_libration >= 1 and votes_circulation == 0:
        return 'LIBRATION', info

    return 'UNCERTAIN', info


def _analyze_windows(
    times: np.ndarray,
    sigma_wrapped: np.ndarray,
    sigma_unwrapped: np.ndarray,
    window_fraction: float,
    circulation_threshold_cycles: float,
    sign_dominance_threshold: float,
    ls_fap_threshold: float,
    ls_snr_threshold: float,
) -> List[Dict[str, Any]]:
    """Analyzing time series by sliding window's method."""

    N = len(times)
    total_duration = times[-1] - times[0]
    window_size = int(N * window_fraction)

    if window_size >= N:
        return []

    step = window_size // 2  # 50% overlap
    results = []

    start = 0
    while start + window_size <= N:
        end = start + window_size

        verdict, info = _classify_segment(
            times[start:end],
            sigma_wrapped[start:end],
            sigma_unwrapped[start:end],
            total_duration,
            circulation_threshold_cycles,
            sign_dominance_threshold,
            ls_fap_threshold * 5,  # softer for windows
            ls_snr_threshold / 2,  # softer for windows
            is_window=True,
        )

        info['verdict'] = verdict
        results.append(info)
        start += step

    return results


def _is_stickiness(window_results: List[Dict], stickiness_ratio_threshold: float = 0.2) -> bool:
    """
    Checks stickiness by the contrast |rev| between windows.

    Stickiness = there are windows with very small |rev| (sticking)
        and windows with very large |rev| (breakout).

    Parameters
    ----------
    window_results : list
        Window analysis results
    stickiness_ratio_threshold : float
        Threshold of min/max ratio to determine stickiness

    Returns
    -------
    bool
        True if stickiness
    """
    if len(window_results) < 3:
        return False

    revs = [abs(w['revolutions']) for w in window_results]
    min_rev = min(revs)
    max_rev = max(revs)

    if max_rev == 0:
        return False

    ratio = min_rev / max_rev
    return ratio < stickiness_ratio_threshold


def _determine_subtype(status: int, phi_deg: float, revolutions: float) -> str:
    """Find subtype (apocentric / pericentric for libration; slow/fast for circulation)"""
    if status in [ResonanceStatus.LIBRATION, ResonanceStatus.LIBRATION_UNCERTAIN]:
        return 'apocentric' if (phi_deg < 90 or phi_deg > 270) else 'pericentric'
    elif status == ResonanceStatus.CIRCULATION:
        return 'fast' if abs(revolutions) > 10 else 'slow'
    return ''


def classify_resonance(
    body,
    resonance,
    circulation_threshold_cycles: float = 2.0,
    sign_dominance_threshold: float = 0.95,
    ls_fap_threshold: float = 0.01,
    ls_snr_threshold: float = 5.0,
    regularity_cv_threshold: float = 0.5,
    stickiness_ratio_threshold: float = 0.2,
    window_fraction: float = 0.1,
    compute_periodogram: bool = True,
) -> Dict[str, Any]:
    """
    Classifies the resonance status of a body.

    Parameters
    ----------
    body : Body
    resonance : Resonance
    circulation_threshold_cycles : float
        Threhold of |rev| for circulation (in revolutions) (2.0 by default)
    sign_dominance_threshold : float
        Threhold sign_dominance for circulation (0.95 by default)
    ls_fap_threshold : float
        Threshold FAP for Lomb-Scargle (0.01 by default)
    ls_snr_threshold : float
        Threshold SNR for Lomb-Scargle (5.0 by default)
    regularity_cv_threshold : float
        Threhold CV for regularity of the derivative of the resonant angle (0.5 by default)
    stickiness_ratio_threshold : float
        Threshold min/max the ratio |rev| for stickiness (0.2 by default)
    window_fraction : float
        Width of the sliding window as a fraction of the total time series (0.1 by default)
    compute_periodogram : bool
        Calculate or not Lomb-Scargle periodogram (True by default)

    Returns
    -------
    dict with status, type, subtype, metrics, resolvers, window_analysis
    """

    times = body.times / (2 * np.pi)
    sigma_wrapped = body.angle(resonance)
    sigma_unwrapped = body.angle_unwrapped(resonance)
    total_duration = times[-1] - times[0]

    metrics = _calc_metrics(times, sigma_wrapped, sigma_unwrapped, compute_periodogram)

    # Classify full time series
    global_verdict, global_info = _classify_segment(
        times,
        sigma_wrapped,
        sigma_unwrapped,
        total_duration,
        circulation_threshold_cycles,
        sign_dominance_threshold,
        ls_fap_threshold,
        ls_snr_threshold,
        is_window=False,
    )

    # Perform window analysis
    window_results = _analyze_windows(
        times,
        sigma_wrapped,
        sigma_unwrapped,
        window_fraction,
        circulation_threshold_cycles,
        sign_dominance_threshold,
        ls_fap_threshold,
        ls_snr_threshold,
    )

    window_verdicts = [w['verdict'] for w in window_results]
    has_libration_window = 'LIBRATION' in window_verdicts
    has_circulation_window = 'CIRCULATION' in window_verdicts

    # Checking stickiness by the contrast |rev|
    is_stickiness = _is_stickiness(window_results, stickiness_ratio_threshold)

    # Additional resolver: regularity of the derivative of the resonant angle
    has_regular_oscillations = metrics['n_zero_crossings'] >= 4 and metrics['cv_intervals'] < regularity_cv_threshold

    # Final status
    # Chain: LIBRATION > TRANSIENT > STICKINESS > CIRCULATION > UNCERTAIN

    # Simple case: if librates globally
    if global_verdict == 'LIBRATION':
        status = ResonanceStatus.LIBRATION
        type_str = 'libration'

    # If there is at least one window with libration (and not full but elif here)
    elif has_libration_window and has_circulation_window:
        status = ResonanceStatus.TRANSIENT
        type_str = 'transient'

    # Stickiness case if there is a big contrast between windows
    elif is_stickiness:
        status = ResonanceStatus.STICKINESS
        type_str = 'stickiness'

    # Circulation case detected globally
    elif global_verdict == 'CIRCULATION':
        status = ResonanceStatus.CIRCULATION
        type_str = 'circulation'

    # Derivative crosses zero often and the intervals are regular (amplitude small or moderate)
    elif has_regular_oscillations and metrics['amplitude'] < 3 * np.pi:
        status = ResonanceStatus.LIBRATION
        type_str = 'libration'

    # Dunno else
    else:
        status = ResonanceStatus.UNCERTAIN
        type_str = 'uncertain'

    # Subtype
    subtype = _determine_subtype(status, metrics['phi_deg'], metrics['revolutions'])

    resolvers = {
        'A1_amplitude': metrics['amplitude'] < 2 * np.pi,
        'A2_revolutions_low': abs(metrics['revolutions']) < 0.5,
        'A3_revolutions_high': abs(metrics['revolutions']) > circulation_threshold_cycles,
        'A4_sign_dominance': metrics['sign_dominance'] > sign_dominance_threshold,
        'B1_regularity': has_regular_oscillations,
        'C1_periodogram': (
            metrics['ls_fap'] is not None
            and metrics['ls_fap'] < ls_fap_threshold
            and metrics['ls_snr'] is not None
            and metrics['ls_snr'] > ls_snr_threshold
        ),
        'global_verdict': global_verdict,
        'has_libration_window': has_libration_window,
        'has_circulation_window': has_circulation_window,
        'is_stickiness': is_stickiness,
    }

    return {
        'status': int(status),
        'type': type_str,
        'subtype': subtype,
        'metrics': metrics,
        'resolvers': resolvers,
        'window_analysis': window_results,
    }
