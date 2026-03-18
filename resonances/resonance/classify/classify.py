import numpy as np
from typing import Optional, Tuple, Union
from scipy.stats import linregress

from resonances.logger import logger
from resonances.resonance.classify.models import (
    ClassifyParams,
    ResonanceClassifyResult,
    ResonanceStatus,
    SegmentCounts,
    SegmentMetrics,
)
from resonances.resonance.classify.util import is_unphysical_orbit


def calc_sigma_derivative(times: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Calculate the derivative of the resonant angle using central differences."""
    sigma_dot = np.zeros_like(sigma)
    sigma_dot[1:-1] = (sigma[2:] - sigma[:-2]) / (times[2:] - times[:-2])
    dt = np.diff(times)
    # Add two "lost" points to preserve the length of the array
    sigma_dot[0] = (sigma[1] - sigma[0]) / dt[0]
    sigma_dot[-1] = (sigma[-1] - sigma[-2]) / dt[-1]
    return sigma_dot


def _calc_zero_crossing_by_sigma_derivative(
    sigma_dot: np.ndarray,
) -> Tuple[int, float]:
    """Computes the number of zero crossings and the CV of the inter-zero-crossing intervals."""
    sign_changes = np.where(sigma_dot[:-1] * sigma_dot[1:] < 0)[0]
    n_crossings = len(sign_changes)

    if n_crossings < 2:
        return n_crossings, np.inf

    intervals = np.diff(sign_changes)
    if len(intervals) == 0 or np.mean(intervals) == 0:
        return n_crossings, np.inf

    return n_crossings, np.std(intervals) / np.mean(intervals)


def _calc_metrics(
    times: np.ndarray,
    sigma_wrapped: np.ndarray,
    sigma_unwrapped: np.ndarray,
) -> SegmentMetrics:
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

        segment_metrics = _calc_metrics(
            times[start:end],
            sigma_wrapped[start:end],
            sigma_unwrapped[start:end],
        )
        segments_metrics[f"{segment_start:.2f}-{segment_end:.2f}"] = segment_metrics

        start += window_step

    return segments_metrics


def _is_good_segment(smetrics: SegmentMetrics, params: ClassifyParams) -> bool:
    return (smetrics.revolutions_true <= params.good_seg_max_rev) and (smetrics.trend_to_oscillation < params.good_seg_max_tto)


def _is_reasonable_segment(smetrics: SegmentMetrics, params: ClassifyParams) -> bool:
    return (
        smetrics.revolutions_true <= params.reasonable_seg_max_rev_1 and smetrics.trend_to_oscillation < params.reasonable_seg_max_tto_1
    ) or (smetrics.revolutions_true <= params.reasonable_seg_max_rev_2 and smetrics.trend_to_oscillation < params.reasonable_seg_max_tto_2)


def _count_segments(
    segments_metrics: dict,
    params: ClassifyParams,
) -> SegmentCounts:
    """Count good and reasonable segments per window step and totals."""
    counts = {}
    total_good = 0
    total_reasonable = 0

    for wlp in params.window_steps:
        wlp_key = str(wlp)
        n_good = 0
        n_reasonable = 0
        if wlp_key in segments_metrics:
            for smetrics in segments_metrics[wlp_key].values():
                if _is_good_segment(smetrics, params):
                    n_good += 1
                if _is_reasonable_segment(smetrics, params):
                    n_reasonable += 1
        counts[wlp] = (n_good, n_reasonable)
        total_good += n_good
        total_reasonable += n_reasonable

    return SegmentCounts(
        n_good_total=total_good,
        n_reasonable_total=total_reasonable,
        n_good_0_1=counts.get(0.1, (0, 0))[0],
        n_reasonable_0_1=counts.get(0.1, (0, 0))[1],
        n_good_0_2=counts.get(0.2, (0, 0))[0],
        n_reasonable_0_2=counts.get(0.2, (0, 0))[1],
        n_good_0_3=counts.get(0.3, (0, 0))[0],
        n_reasonable_0_3=counts.get(0.3, (0, 0))[1],
    )


def _segment_comment_strings(segments_metrics: dict, params: ClassifyParams) -> Tuple[str, str]:
    """Build human-readable comment strings for good and reasonable segments."""
    good_s = ""
    reasonable_s = ""
    for _wlp, wlp_segments in segments_metrics.items():
        for segment_key, smetrics in wlp_segments.items():
            if _is_good_segment(smetrics, params):
                good_s += f"{segment_key}: rev {smetrics.revolutions_true:.2f}, " f"tto {smetrics.trend_to_oscillation:.2f}; "
            if _is_reasonable_segment(smetrics, params):
                reasonable_s += f"{segment_key}: rev {smetrics.revolutions_true:.2f}, " f"tto {smetrics.trend_to_oscillation:.2f}; "
    return good_s, reasonable_s


def classify_from_metrics(  # noqa: C901
    metrics: SegmentMetrics,
    segment_counts: SegmentCounts,
    segments_metrics: dict,
    params: Optional[ClassifyParams] = None,
) -> dict[str, Union[ResonanceClassifyResult, dict]]:
    """Pure decision logic: classify from pre-computed metrics.

    This is the lowest layer. It applies the classification decision tree
    using only pre-computed metrics and threshold parameters. No raw time
    series data is needed, making it suitable for reclassification from
    summary.csv / segments.csv without re-running the simulation.

    Args:
        metrics: Global SegmentMetrics for the full time series.
        segment_counts: SegmentCounts aggregated across window steps.
        segments_metrics: Dict keyed by window step (str) → dict of
            segment_key → SegmentMetrics.
        params: Classification thresholds. Defaults to ClassifyParams().

    Returns:
        Dict with "result" (ResonanceClassifyResult) and "segments".
    """
    if params is None:
        params = ClassifyParams()

    # Global libration branch
    if metrics.revolutions_true <= params.rev_libration:
        if metrics.trend_to_oscillation < params.tto_pure_libration:
            result = ResonanceClassifyResult(
                status=ResonanceStatus.LIBRATION,
                type="resonance",
                subtype="pure libration",
                confidence="high",
                metrics=metrics,
                segment_counts=segment_counts,
            )
        elif metrics.trend_to_oscillation < params.tto_partial_libration:
            result = ResonanceClassifyResult(
                status=ResonanceStatus.LIBRATION,
                type="resonance",
                subtype="partial libration",
                confidence="medium",
                metrics=metrics,
                segment_counts=segment_counts,
                comments=("High trend to oscillation ratio. " "Maybe, not enough libration cycles"),
            )
        else:
            result = ResonanceClassifyResult(
                status=ResonanceStatus.PROBABLY_SLOW_CIRCULATION,
                type="non-resonant",
                subtype="probably, slow libration",
                confidence="low",
                metrics=metrics,
                segment_counts=segment_counts,
                comments=("Very high trend to oscillation ratio. " "Looks like a very slow libration or circulation"),
            )
        return {"result": result, "segments": segments_metrics}

    # Strong circulation
    if metrics.trend_to_oscillation > params.tto_non_resonant:
        result = ResonanceClassifyResult(
            status=ResonanceStatus.NON_RESONANT,
            type="circulation",
            subtype="high trend to oscillation",
            confidence="high",
            metrics=metrics,
            segment_counts=segment_counts,
        )
        return {"result": result, "segments": segments_metrics}

    # Segment-based classification using pre-computed counts
    n_good = segment_counts.n_good_total
    has_reasonable_segment = segment_counts.n_reasonable_total > 0

    # >=2 good segments: transient regardless of global tto
    if n_good >= 2:
        good_s, reasonable_s = _segment_comment_strings(segments_metrics, params)
        result = ResonanceClassifyResult(
            status=ResonanceStatus.TRANSIENT,
            type="transient",
            subtype="transient by segments",
            confidence="high",
            metrics=metrics,
            segment_counts=segment_counts,
            comments=(f"Good segments ({n_good}): {good_s} " f"Reasonable segments: {reasonable_s} " "Multiple good segments found."),
        )
        return {"result": result, "segments": segments_metrics}

    # 1 good segment + moderate global tto: transient
    if n_good == 1 and (metrics.trend_to_oscillation < params.tto_transient_global):
        good_s, reasonable_s = _segment_comment_strings(segments_metrics, params)
        result = ResonanceClassifyResult(
            status=ResonanceStatus.TRANSIENT,
            type="transient",
            subtype="transient by segments",
            confidence="medium",
            metrics=metrics,
            segment_counts=segment_counts,
            comments=(
                f"Good segments: {good_s} " f"Reasonable segments: {reasonable_s} " "Overall trend to oscillation ratio is not very high."
            ),
        )
        return {"result": result, "segments": segments_metrics}

    # 1 good segment but high global tto: near separatrix
    if n_good == 1:
        good_s, _ = _segment_comment_strings(segments_metrics, params)
        result = ResonanceClassifyResult(
            status=ResonanceStatus.NEAR_SEPARATRIX,
            type="near separatrix",
            subtype="by segments",
            confidence="medium",
            metrics=metrics,
            segment_counts=segment_counts,
            comments=(f"Good segments: {good_s}, " "but trend to oscillation ratio is high."),
        )
        return {"result": result, "segments": segments_metrics}

    if (metrics.trend_to_oscillation < params.tto_near_separatrix) and has_reasonable_segment:
        _, reasonable_s = _segment_comment_strings(segments_metrics, params)
        result = ResonanceClassifyResult(
            status=ResonanceStatus.PROBABLY_NEAR_SEPARATRIX,
            type="probably near separatrix",
            subtype="by limited trend to oscillation",
            confidence="medium",
            metrics=metrics,
            segment_counts=segment_counts,
            comments=("Has no good segments, has reasonable: " f"{reasonable_s} " "but trend to oscillation ratio is not very high."),
        )
        return {"result": result, "segments": segments_metrics}

    result = ResonanceClassifyResult(
        status=ResonanceStatus.NON_RESONANT,
        type="circulation",
        subtype="remaining",
        confidence="medium",
        metrics=metrics,
        segment_counts=segment_counts,
    )
    return {"result": result, "segments": segments_metrics}


def classify_from_data(
    times: np.ndarray,
    sigma_wrapped: np.ndarray,
    sigma_unwrapped: np.ndarray,
    params: Optional[ClassifyParams] = None,
) -> dict[str, Union[ResonanceClassifyResult, dict]]:
    """Classify from raw time series arrays (no Body object needed).

    Middle layer: computes metrics from arrays, then delegates to
    classify_from_metrics. Use this when you have the raw angle arrays
    but no Body object (e.g., loaded from CSV).

    Args:
        times: Time array in years (not radians).
        sigma_wrapped: Wrapped resonant angle array (0..2pi).
        sigma_unwrapped: Unwrapped (continuous) resonant angle array.
        params: Classification thresholds. Defaults to ClassifyParams().

    Returns:
        Dict with "result" (ResonanceClassifyResult) and "segments".
    """
    if params is None:
        params = ClassifyParams()

    metrics = _calc_metrics(times, sigma_wrapped, sigma_unwrapped)

    # Fast paths: skip expensive segment computation for early-exit branches
    if metrics.revolutions_true <= params.rev_libration:
        return classify_from_metrics(metrics, SegmentCounts(), {}, params)
    if metrics.trend_to_oscillation > params.tto_non_resonant:
        return classify_from_metrics(metrics, SegmentCounts(), {}, params)

    # Segment-dependent branches
    segments_metrics = {}
    for wlp in params.window_steps:
        segments_metrics[str(wlp)] = _calc_segments_metrics(
            times,
            sigma_wrapped,
            sigma_unwrapped,
            wlp,
            params.window_step_percentage,
        )

    segment_counts = _count_segments(segments_metrics, params)

    return classify_from_metrics(metrics, segment_counts, segments_metrics, params)


def _params_from_config(config) -> ClassifyParams:
    """Bridge SimulationConfig attributes to ClassifyParams.

    Uses ClassifyParams defaults as fallback so that defaults are
    defined in exactly one place.
    """
    d = ClassifyParams()
    return ClassifyParams(
        window_steps=getattr(config, "classify_window_steps", d.window_steps),
        window_step_percentage=getattr(config, "classify_window_step", d.window_step_percentage),
        rev_libration=getattr(config, "classify_rev_libration", d.rev_libration),
        tto_pure_libration=getattr(config, "classify_tto_pure_libration", d.tto_pure_libration),
        tto_partial_libration=getattr(config, "classify_tto_partial_libration", d.tto_partial_libration),
        tto_non_resonant=getattr(config, "classify_tto_non_resonant", d.tto_non_resonant),
        tto_transient_global=getattr(config, "classify_tto_transient_global", d.tto_transient_global),
        tto_near_separatrix=getattr(config, "classify_tto_near_separatrix", d.tto_near_separatrix),
        good_seg_max_rev=getattr(config, "classify_good_seg_max_rev", d.good_seg_max_rev),
        good_seg_max_tto=getattr(config, "classify_good_seg_max_tto", d.good_seg_max_tto),
        reasonable_seg_max_rev_1=getattr(config, "classify_reasonable_seg_max_rev_1", d.reasonable_seg_max_rev_1),
        reasonable_seg_max_tto_1=getattr(config, "classify_reasonable_seg_max_tto_1", d.reasonable_seg_max_tto_1),
        reasonable_seg_max_rev_2=getattr(config, "classify_reasonable_seg_max_rev_2", d.reasonable_seg_max_rev_2),
        reasonable_seg_max_tto_2=getattr(config, "classify_reasonable_seg_max_tto_2", d.reasonable_seg_max_tto_2),
    )


def classify_resonance(
    body,
    resonance,
    config=None,
    params: Optional[ClassifyParams] = None,
) -> dict[str, Union[ResonanceClassifyResult, dict]]:
    """Classify a resonant angle time series from a Body object.

    Top layer: extracts arrays from the Body, checks for unphysical orbits,
    then delegates to classify_from_data. This preserves backward
    compatibility with existing callers.

    Args:
        body: Body object with times, angles_filtered, etc.
        resonance: Resonance object (TwoBody, ThreeBody, SecularResonance).
        config: Optional SimulationConfig. Used to build ClassifyParams
            if params is not provided.
        params: Optional ClassifyParams overriding config and defaults.

    Returns:
        Dict with "result" (ResonanceClassifyResult) and "segments".
    """
    if params is None:
        if config is not None:
            params = _params_from_config(config)
        else:
            params = ClassifyParams()

    is_unstable, reason = is_unphysical_orbit(body)
    if is_unstable:
        logger.warning(f"Unphysical orbit for {body.name} " f"at {resonance.to_s()}: {reason}")
        return {
            "result": ResonanceClassifyResult(
                status=ResonanceStatus.CHAOTIC,
                type="chaotic",
                metrics=SegmentMetrics(),
            ),
            "segments": {},
        }

    times = body.times / (2 * np.pi)
    sigma_wrapped = body.angles_filtered[resonance.to_s()]
    sigma_unwrapped = body.angles_filtered_unwrapped[resonance.to_s()]

    return classify_from_data(times, sigma_wrapped, sigma_unwrapped, params)
