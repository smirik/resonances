import numpy as np
from typing import Tuple, Union
from scipy.stats import linregress

from resonances.logger import logger
from resonances.resonance.classify.models import ResonanceClassifyResult, ResonanceStatus, SegmentMetrics
from resonances.resonance.classify.util import is_unphysical_orbit, merge_intervals


def _calc_sigma_derivative(times: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Calculate the derivative of the resonant angle using central differences."""
    sigma_dot = np.zeros_like(sigma)
    sigma_dot[1:-1] = (sigma[2:] - sigma[:-2]) / (times[2:] - times[:-2])
    dt = np.diff(times)
    # Add two "lost" points to preserve the length of the array
    sigma_dot[0] = (sigma[1] - sigma[0]) / dt[0]
    sigma_dot[-1] = (sigma[-1] - sigma[-2]) / dt[-1]
    return sigma_dot


def _calc_zero_crossing_by_sigma_derivative(sigma_dot: np.ndarray) -> Tuple[int, float]:
    """Computes the number of zero crossings and the CV of the inter-zero-crossing intervals."""
    sign_changes = np.where(sigma_dot[:-1] * sigma_dot[1:] < 0)[0]
    n_crossings = len(sign_changes)

    if n_crossings < 2:
        return n_crossings, np.inf

    intervals = np.diff(sign_changes)
    if len(intervals) == 0 or np.mean(intervals) == 0:
        return n_crossings, np.inf

    return n_crossings, np.std(intervals) / np.mean(intervals)


def _calc_metrics(times: np.ndarray, sigma_wrapped: np.ndarray, sigma_unwrapped: np.ndarray) -> SegmentMetrics:
    """Calculates all metrics."""

    cos_avg = np.mean(np.cos(sigma_wrapped))
    sin_avg = np.mean(np.sin(sigma_wrapped))
    R = np.sqrt(cos_avg**2 + sin_avg**2)
    phi_rad = np.arctan2(sin_avg, cos_avg)
    phi_deg = np.rad2deg(phi_rad) % 360

    amplitude = np.max(sigma_unwrapped) - np.min(sigma_unwrapped)

    displacement = sigma_unwrapped[-1] - sigma_unwrapped[0]
    revolutions = displacement / (2 * np.pi)
    revolutions_true = (np.max(sigma_unwrapped) - np.min(sigma_unwrapped)) / (2 * np.pi)

    sigma_dot = _calc_sigma_derivative(times, sigma_unwrapped)
    frac_positive = np.mean(sigma_dot > 0)
    sign_dominance = max(frac_positive, 1 - frac_positive)

    n_zero_crossings, cv_intervals = _calc_zero_crossing_by_sigma_derivative(sigma_dot)

    ls_period, ls_fap, ls_snr = None, None, None
    # if compute_periodogram:
    #     ls_period, ls_fap, ls_snr = _calc_periodogram(times, sigma_wrapped)

    mean_sigma_dot = np.mean(sigma_dot)
    std_sigma_dot = np.std(sigma_dot)
    sigma_dot_ratio = np.abs(mean_sigma_dot) / std_sigma_dot if std_sigma_dot > 0 else 0

    slope, intercept, r_value, _, _ = linregress(times, sigma_unwrapped)

    # trend
    trend = intercept + slope * times
    trend_displacement = abs(slope * (times[-1] - times[0]))
    # residual is the oscillation
    residual = sigma_unwrapped - trend
    oscillation_amplitude = np.max(residual) - np.min(residual)
    # ratio of trend to oscillation
    trend_to_oscillation = trend_displacement / (oscillation_amplitude + 1e-10)

    metrics = SegmentMetrics(
        phi_rad=phi_rad,
        phi_deg=phi_deg,
        R=R,
        revolutions=revolutions,
        revolutions_true=revolutions_true,
        amplitude=amplitude,
        sign_dominance=sign_dominance,
        sigma_dot_ratio=sigma_dot_ratio,
        n_zero_crossings=n_zero_crossings,
        cv_intervals=cv_intervals,
        mean_sigma_dot=mean_sigma_dot,
        std_sigma_dot=std_sigma_dot,
        ls_period=ls_period,
        ls_fap=ls_fap,
        ls_snr=ls_snr,
        trend_to_oscillation=trend_to_oscillation,
    )

    return metrics


def _classify_segments(
    times: np.ndarray,
    sigma_wrapped: np.ndarray,
    sigma_unwrapped: np.ndarray,
    window_length_percentage: float,
    window_step_percentage: float,
    config=None,
):
    revolutions_segment_hard = getattr(config, 'classify_revolutions_segment_hard', 2.0) if config else 2.0
    tto_transient = getattr(config, 'classify_tto_transient', 1.5) if config else 1.5
    mean_derivative_threshold = getattr(config, 'classify_mean_derivative_threshold', 0.00005) if config else 0.00005
    revolutions_libration_soft = getattr(config, 'classify_revolutions_libration_soft', 1.5) if config else 1.5
    sign_dominance_segment = getattr(config, 'classify_sign_dominance_segment', 0.7) if config else 0.7

    N = len(times)
    window_length = int(N * window_length_percentage)
    window_step = int(N * window_step_percentage)

    start = 0
    libration_segments = []
    libration_segments_metrics = {}
    segments_data = {}

    while start + window_length <= N:
        end = start + window_length

        segment_start = times[start]
        segment_end = times[end - 1]

        segment_metrics = _calc_metrics(times[start:end], sigma_wrapped[start:end], sigma_unwrapped[start:end])

        has_small_rev_and_trend = (segment_metrics.revolutions_true <= revolutions_segment_hard) and (
            segment_metrics.trend_to_oscillation < tto_transient
        )
        has_small_mean_sigma_dot_and_rev = (abs(segment_metrics.mean_sigma_dot) <= mean_derivative_threshold) and (
            segment_metrics.revolutions_true <= revolutions_libration_soft
        )
        has_reasonable_sign_dominance = abs(segment_metrics.sign_dominance) < sign_dominance_segment

        has_libration = has_reasonable_sign_dominance and (has_small_rev_and_trend or has_small_mean_sigma_dot_and_rev)

        if has_libration:
            libration_segments.append((segment_start, segment_end))
            libration_segments_metrics[f"{segment_start:.2f}-{segment_end:.2f}"] = segment_metrics
        segments_data[f"{segment_start:.2f}-{segment_end:.2f}"] = segment_metrics
        start += window_step

    libration_segments = np.array(libration_segments, dtype=times.dtype).reshape(-1, 2)
    merged = merge_intervals(libration_segments, join_touching=True)

    lengths = (merged[:, 1] - merged[:, 0]).astype(float) if merged.size else np.array([], float)
    total_true_length = float(lengths.sum())

    total_length = float(times[-1] - times[0]) if N >= 2 else 0.0
    ratio = (total_true_length / total_length) if total_length > 0 else 0.0

    return {
        'merged': merged,
        'good_segments_metrics': libration_segments_metrics,
        'ratio': ratio,
        'total_libration_length': total_true_length,
        'segments_data': segments_data,
    }


def classify_resonance(  # noqa: C901
    body,
    resonance,
    config=None,
    window_length_percentage: float = None,
    window_step_percentage: float = None,
) -> dict[str, Union[ResonanceClassifyResult, dict]]:
    # Get window parameters from config or use defaults
    if window_length_percentage is None:
        window_length_percentage = getattr(config, 'classify_window_length', 0.1) if config else 0.1
    if window_step_percentage is None:
        window_step_percentage = getattr(config, 'classify_window_step', 0.05) if config else 0.05

    # Get thresholds from config or use defaults
    mean_derivative_threshold = getattr(config, 'classify_mean_derivative_threshold', 0.00005) if config else 0.00005
    sign_dominance_libration = getattr(config, 'classify_sign_dominance_libration', 0.6) if config else 0.6
    revolutions_libration_soft = getattr(config, 'classify_revolutions_libration_soft', 1.5) if config else 1.5
    revolutions_uncertain_segment = getattr(config, 'classify_revolutions_uncertain_segment', 1.0) if config else 1.0
    revolutions_transient_segment = getattr(config, 'classify_revolutions_transient_segment', 1.0) if config else 1.0
    tto_high_amp_libration = getattr(config, 'classify_tto_high_amp_libration', 0.01) if config else 0.01
    tto_transient = getattr(config, 'classify_tto_transient', 1.5) if config else 1.5
    tto_uncertain_segment = getattr(config, 'classify_tto_uncertain_segment', 2.0) if config else 2.0
    tto_non_resonant_above = getattr(config, 'classify_tto_non_resonant_above', 5.0) if config else 5.0
    tto_transient_segment = getattr(config, 'classify_tto_transient_segment', 0.4) if config else 0.4
    tto_transient_segment_min = getattr(config, 'classify_tto_transient_segment_min', 0.15) if config else 0.15
    transient_segments_min = getattr(config, 'classify_transient_segments_min', 1) if config else 1

    is_unstable, reason = is_unphysical_orbit(body)
    if is_unstable:
        logger.warning(f"Unphysical orbit for {body.name} at {resonance.to_s()}: {reason}")
        return {
            "result": ResonanceClassifyResult(
                status=ResonanceStatus.CHAOTIC,
                type='chaotic',
                metrics=SegmentMetrics(),
            ),
            "extra": {},
        }

    times = body.times / (2 * np.pi)
    sigma_wrapped = body.angles_filtered[resonance.to_s()]
    sigma_unwrapped = body.angles_filtered_unwrapped[resonance.to_s()]

    metrics = _calc_metrics(times, sigma_wrapped, sigma_unwrapped)

    # if globally librates, no need to classify further
    if metrics.revolutions_true <= 1:
        result = ResonanceClassifyResult(
            status=ResonanceStatus.LIBRATION,
            type='libration_by_revolutions',
            subtype='test',
            metrics=metrics,
        )
        return {"result": result, "extra": {}}

    if (
        (metrics.trend_to_oscillation < tto_high_amp_libration)
        and (metrics.sign_dominance < sign_dominance_libration)
        and (metrics.mean_sigma_dot < mean_derivative_threshold)
    ):
        result = ResonanceClassifyResult(
            status=ResonanceStatus.LIBRATION,
            type='libration_very_high_amplitude',
            subtype='test',
            metrics=metrics,
        )
        return {"result": result, "extra": {}}

    if (abs(metrics.mean_sigma_dot) <= mean_derivative_threshold) and (metrics.revolutions_true <= revolutions_libration_soft):
        result = ResonanceClassifyResult(
            status=ResonanceStatus.LIBRATION,
            type='libration_by_mean_derivative',
            subtype='test',
            metrics=metrics,
        )
        return {"result": result, "extra": {}}

    results = _classify_segments(times, sigma_wrapped, sigma_unwrapped, window_length_percentage, window_step_percentage, config)

    # 0 good segments
    if results['ratio'] == 0:
        # not very good segments, but still (no sign dominance condition + relaxed to trend)
        uncertain_segments = [
            key
            for key, value in results["segments_data"].items()
            if (value.trend_to_oscillation < tto_uncertain_segment) and (value.revolutions_true < revolutions_uncertain_segment)
        ]
        if len(uncertain_segments) > 0:
            result = ResonanceClassifyResult(
                status=ResonanceStatus.UNCERTAIN,
                type='uncertain_by_segments',
                subtype='test',
                metrics=metrics,
            )
            return {"result": result, "extra": results}

        result = ResonanceClassifyResult(
            status=ResonanceStatus.NON_RESONANT,
            type='no_libration_segments',
            subtype='test',
            metrics=metrics,
        )
        return {"result": result, "extra": results}

    # If there is a strong trend with small or no oscillation, it is non-resonant
    if metrics.trend_to_oscillation > tto_non_resonant_above:
        result = ResonanceClassifyResult(
            status=ResonanceStatus.NON_RESONANT,
            type='trend_with_no_or_weak_oscillation',
            subtype='test',
            metrics=metrics,
        )
        return {"result": result, "extra": results}

    if metrics.trend_to_oscillation < tto_transient:
        result = ResonanceClassifyResult(
            status=ResonanceStatus.TRANSIENT,
            type='transient_by_trend_to_oscillation',
            subtype='test',
            metrics=metrics,
        )
        return {"result": result, "extra": results}

    segment_ttos = np.array([value.trend_to_oscillation for value in results["segments_data"].values()])
    segment_revs = np.array([value.revolutions_true for value in results["segments_data"].values()])
    min_segment_tto = np.min(segment_ttos)
    n_transient_segments = np.sum((segment_ttos < tto_transient_segment) & (segment_revs < revolutions_transient_segment))

    if (n_transient_segments >= transient_segments_min) and (min_segment_tto < tto_transient_segment_min):
        result = ResonanceClassifyResult(
            status=ResonanceStatus.TRANSIENT,
            type='transient_by_segments',
            subtype='test',
            metrics=metrics,
        )
        return {"result": result, "extra": results}

    result = ResonanceClassifyResult(
        status=ResonanceStatus.STICKINESS,
        type='trend_with_libration',
        subtype='test',
        metrics=metrics,
    )
    return {"result": result, "extra": results}
