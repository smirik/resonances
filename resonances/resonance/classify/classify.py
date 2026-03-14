import numpy as np
from typing import Tuple, Union
from scipy.stats import linregress

from resonances.logger import logger
from resonances.resonance.classify.models import ResonanceClassifyResult, ResonanceStatus, SegmentMetrics
from resonances.resonance.classify.util import is_unphysical_orbit, merge_intervals


def calc_sigma_derivative(times: np.ndarray, sigma: np.ndarray) -> np.ndarray:
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

    sigma_dot = calc_sigma_derivative(times, sigma_unwrapped)
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


def _calc_segments_metrics(
    times: np.ndarray,
    sigma_wrapped: np.ndarray,
    sigma_unwrapped: np.ndarray,
    window_length_percentage: float,
    window_step_percentage: float,
) -> dict[str, SegmentMetrics]:
    N = len(times)
    window_length = int(N * window_length_percentage)
    window_step = int(N * window_step_percentage)

    start = 0
    segments_metrics = {}

    while start + window_length <= N:
        end = start + window_length

        segment_start = times[start]
        segment_end = times[end - 1]

        segment_metrics = _calc_metrics(times[start:end], sigma_wrapped[start:end], sigma_unwrapped[start:end])
        segments_metrics[f"{segment_start:.2f}-{segment_end:.2f}"] = segment_metrics

        start += window_step

    return segments_metrics


def classify_resonance(  # noqa: C901
    body,
    resonance,
    config=None,
    window_length_percentage: float = None,
    window_step_percentage: float = None,
) -> dict[str, Union[ResonanceClassifyResult, dict]]:
    # Get window parameters from config or use defaults
    window_steps = [0.1, 0.2, 0.3]
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
    segments_metrics = {}
    exported_metrics = {}
    for wlp in window_steps:
        segments_metrics[str(wlp)] = _calc_segments_metrics(times, sigma_wrapped, sigma_unwrapped, wlp, window_step_percentage)
        for skey, smetrics in segments_metrics[str(wlp)].items():
            exported_metrics[f"w_{wlp}_segment_{skey}"] = smetrics

    # if globally librates, no need to classify further
    if metrics.revolutions_true <= 1.0:
        if metrics.trend_to_oscillation < 0.5:
            result = ResonanceClassifyResult(
                status=ResonanceStatus.LIBRATION,
                type='resonance',
                subtype='pure libration',
                confidence="high",
                metrics=metrics,
            )
        elif metrics.trend_to_oscillation < 2.5:
            result = ResonanceClassifyResult(
                status=ResonanceStatus.LIBRATION,
                type='resonance',
                subtype='partial libration',
                confidence="medium",
                metrics=metrics,
                comments="High trend to oscillation ratio. Maybe, not enough libration cycles",
            )
        else:
            result = ResonanceClassifyResult(
                status=ResonanceStatus.PROBABLY_SLOW_CIRCULATION,
                type='non-resonant',
                subtype='probably, slow libration',
                confidence="low",
                metrics=metrics,
                comments="Very high trend to oscillation ratio. Looks like a very slow libration or circulation",
            )
        return {"result": result, "segments": segments_metrics}

    if metrics.trend_to_oscillation > 6.0:
        result = ResonanceClassifyResult(
            status=ResonanceStatus.NON_RESONANT,
            type='circulation',
            subtype='high trend to oscillation',
            confidence="high",
            metrics=metrics,
        )
        return {"result": result, "segments": segments_metrics}

    good_segments = {}
    reasonable_segments = {}
    has_good_segment = False
    good_segments_s = ""
    has_reasonable_segment = False
    reasonable_segments_s = ""

    for wlp, segments_metrics in segments_metrics.items():
        for segment_key, smetrics in segments_metrics.items():
            if (smetrics.revolutions_true <= 1.0) and (smetrics.trend_to_oscillation < 0.5):
                good_segments[segment_key] = smetrics
                has_good_segment = True
                good_segments_s += f"{segment_key}: rev {smetrics.revolutions_true:.2f}, tto {smetrics.trend_to_oscillation:.2f}\n"
            if (
                (smetrics.revolutions_true <= 2.0)
                and (smetrics.trend_to_oscillation < 1.0)
                or (smetrics.revolutions_true <= 1.0)
                and (smetrics.trend_to_oscillation < 1.5)
            ):
                has_reasonable_segment = True
                reasonable_segments[segment_key] = smetrics
                reasonable_segments_s += f"{segment_key}: rev {smetrics.revolutions_true:.2f}, tto {smetrics.trend_to_oscillation:.2f}\n"

    if has_good_segment and (metrics.trend_to_oscillation < 3.0):
        result = ResonanceClassifyResult(
            status=ResonanceStatus.TRANSIENT,
            type='transient',
            subtype='transient by segments',
            confidence="high",
            metrics=metrics,
            comments=f"Good segments: {good_segments_s}\nReasonable segments:\n{reasonable_segments_s}\nOverall trend to oscillation ratio is not very high.",
        )
        return {"result": result, "segments": segments_metrics}
    elif has_good_segment > 0:
        result = ResonanceClassifyResult(
            status=ResonanceStatus.NEAR_SEPARATRIX,
            type='near separatrix',
            subtype='by segments',
            confidence="medium",
            metrics=metrics,
            comments=f"Good segments: {good_segments_s}, but trend to oscillation ratio is high.",
        )
        return {"result": result, "segments": segments_metrics}

    if (metrics.trend_to_oscillation < 6.0) and has_reasonable_segment:
        result = ResonanceClassifyResult(
            status=ResonanceStatus.PROBABLY_NEAR_SEPARATRIX,
            type='probably near separatrix',
            subtype='by limited trend to oscillation',
            confidence="medium",
            metrics=metrics,
            comments=f"Has no good segments, has reasonable: {reasonable_segments_s}\n but trend to oscillation ratio is not very high.",
        )
        return {"result": result, "segments": segments_metrics}

    result = ResonanceClassifyResult(
        status=ResonanceStatus.NON_RESONANT,
        type='circulation',
        subtype='remaining',
        confidence="medium",
        metrics=metrics,
    )
    return {"result": result, "segments": segments_metrics}
    # if (
    #     (metrics.trend_to_oscillation < tto_high_amp_libration)
    #     and (metrics.sign_dominance < sign_dominance_libration)
    #     and (metrics.mean_sigma_dot < mean_derivative_threshold)
    # ):
    #     result = ResonanceClassifyResult(
    #         status=ResonanceStatus.LIBRATION,
    #         type='resonance',
    #         subtype='libration_very_high_amplitude',
    #         metrics=metrics,
    #     )
    #     return {"result": result, "extra": {}}

    # if (abs(metrics.mean_sigma_dot) <= mean_derivative_threshold) and (metrics.revolutions_true <= revolutions_libration_soft):
    #     result = ResonanceClassifyResult(
    #         status=ResonanceStatus.LIBRATION,
    #         type='resonance',
    #         subtype='libration_by_mean_derivative',
    #         metrics=metrics,
    #     )
    #     return {"result": result, "extra": {}}

    # results = _classify_segments(times, sigma_wrapped, sigma_unwrapped, window_length_percentage, window_step_percentage, config)

    # # 0 good segments
    # if results['ratio'] == 0:
    #     # not very good segments, but still (no sign dominance condition + relaxed to trend)
    #     uncertain_segments = [
    #         key
    #         for key, value in results["segments_data"].items()
    #         if (value.trend_to_oscillation < tto_uncertain_segment) and (value.revolutions_true < revolutions_uncertain_segment)
    #     ]
    #     if len(uncertain_segments) > 0:
    #         result = ResonanceClassifyResult(
    #             status=ResonanceStatus.UNCERTAIN,
    #             type='uncertain',
    #             subtype='uncertain_by_segments',
    #             metrics=metrics,
    #         )
    #         return {"result": result, "extra": results}

    #     result = ResonanceClassifyResult(
    #         status=ResonanceStatus.NON_RESONANT,
    #         type='non_resonant',
    #         subtype='no_libration_segments',
    #         metrics=metrics,
    #     )
    #     return {"result": result, "extra": results}

    # if (metrics.trend_to_oscillation < tto_transient) and (metrics.sign_dominance < 0.7):
    #     result = ResonanceClassifyResult(
    #         status=ResonanceStatus.TRANSIENT,
    #         type='transient',
    #         subtype='transient_by_trend_to_oscillation',
    #         metrics=metrics,
    #     )
    #     return {"result": result, "extra": results}

    # segment_ttos = np.array([value.trend_to_oscillation for value in results["segments_data"].values()])
    # segment_revs = np.array([value.revolutions_true for value in results["segments_data"].values()])
    # min_segment_tto = np.min(segment_ttos)
    # n_transient_segments = np.sum((segment_ttos < tto_transient_segment) & (segment_revs < revolutions_transient_segment))

    # # If there is a strong trend with small or no oscillation, it is non-resonant
    # if metrics.trend_to_oscillation > tto_non_resonant_above:
    #     # backup for weird cases that have an interesting interval
    #     if n_transient_segments >= transient_segments_min:
    #         result = ResonanceClassifyResult(
    #             status=ResonanceStatus.UNCERTAIN,
    #             type='uncertain',
    #             subtype='uncertain_high_tto_but_transient_segments',
    #             metrics=metrics,
    #         )
    #         return {"result": result, "extra": results}

    #     result = ResonanceClassifyResult(
    #         status=ResonanceStatus.NON_RESONANT,
    #         type='non_resonant',
    #         subtype='trend_with_no_or_weak_oscillation',
    #         metrics=metrics,
    #     )
    #     return {"result": result, "extra": results}

    # if (
    #     (n_transient_segments >= transient_segments_min)
    #     and (min_segment_tto < tto_transient_segment_min)
    #     and (metrics.trend_to_oscillation < 4)
    # ):
    #     result = ResonanceClassifyResult(
    #         status=ResonanceStatus.TRANSIENT,
    #         type='transient',
    #         subtype='transient_by_segments',
    #         metrics=metrics,
    #     )
    #     return {"result": result, "extra": results}

    # result = ResonanceClassifyResult(
    #     status=ResonanceStatus.STICKINESS,
    #     type='stickiness',
    #     subtype='trend_with_libration',
    #     metrics=metrics,
    # )
    # return {"result": result, "extra": results}
