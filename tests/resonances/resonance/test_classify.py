import numpy as np

from resonances.resonance.classify import (
    ClassifyParams,
    check_chaos,
    classify_from_data,
    classify_from_metrics,
    classify_resonance,
    ResonanceStatus,
    SegmentCounts,
    SegmentMetrics,
)
from resonances.body import Body
from resonances.mmr.two_body import TwoBody
from resonances.secular.secular_resonance import SecularResonance


def _create_mmr():
    """Create a simple MMR for testing."""
    return TwoBody("2J-1")


def _create_secular():
    """Create a simple secular resonance for testing."""
    return SecularResonance("g-g5")


def _create_body_with_angles(times, angles, resonance):
    """Create a Body object with the given angles."""
    body = Body()
    body.name = "test_body"
    body.times = times

    # Set initial data (needed for is_unphysical_orbit check)
    body.initial_data = {'a': 2.5, 'e': 0.1, 'inc': 0.1, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}

    # Set orbital elements (stable values)
    body.axis = np.full_like(times, 2.5)
    body.ecc = np.full_like(times, 0.1)
    body.inc = np.full_like(times, 0.1)

    # Set filtered angles (this is what classify_resonance uses)
    body.angles_filtered[resonance.to_s()] = angles
    body.angles_filtered_unwrapped[resonance.to_s()] = np.unwrap(angles)

    # Register the resonance with the body
    if isinstance(resonance, TwoBody):
        body.mmrs.append(resonance)
    elif isinstance(resonance, SecularResonance):
        body.secular_resonances.append(resonance)

    return body


def test_classify_libration():
    """Test that pure libration is classified as LIBRATION (status 2)."""
    times = np.linspace(0.0, 100.0, 2000) * 2 * np.pi  # times are in radians
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / (10.0 * 2 * np.pi))) % (2 * np.pi)

    resonance = _create_mmr()
    body = _create_body_with_angles(times, angles, resonance)

    result = classify_resonance(body, resonance)

    assert result['result'].status == ResonanceStatus.LIBRATION


def test_classify_circulation():
    """Test that circulation is classified as NON_RESONANT (status 0)."""
    times = np.linspace(0.0, 100.0, 2000) * 2 * np.pi
    angles = (0.05 * times) % (2 * np.pi)  # steady circulation

    resonance = _create_mmr()
    body = _create_body_with_angles(times, angles, resonance)

    result = classify_resonance(body, resonance)

    assert result['result'].status == ResonanceStatus.NON_RESONANT


def test_classify_secular_resonance():
    """Test that libration in secular resonance is detected."""
    times = np.linspace(0.0, 100.0, 2000) * 2 * np.pi
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / (10.0 * 2 * np.pi))) % (2 * np.pi)

    resonance = _create_secular()
    body = _create_body_with_angles(times, angles, resonance)

    result = classify_resonance(body, resonance)

    assert result['result'].status == ResonanceStatus.LIBRATION


def test_classify_returns_result_object():
    """Test that classify_resonance returns proper structure."""
    times = np.linspace(0.0, 100.0, 2000) * 2 * np.pi
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / (10.0 * 2 * np.pi))) % (2 * np.pi)

    resonance = _create_mmr()
    body = _create_body_with_angles(times, angles, resonance)

    result = classify_resonance(body, resonance)

    assert 'result' in result
    assert 'segments' in result
    assert hasattr(result['result'], 'status')
    assert hasattr(result['result'], 'type')
    assert hasattr(result['result'], 'metrics')
    assert hasattr(result['result'], 'segment_counts')


def test_classify_chaotic_orbit():
    """Test that body with unphysical orbit (e > 1.3) is classified as CHAOTIC."""
    times = np.linspace(0.0, 100.0, 2000) * 2 * np.pi
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / (10.0 * 2 * np.pi))) % (2 * np.pi)

    resonance = _create_mmr()
    body = _create_body_with_angles(times, angles, resonance)
    # Set eccentricity > 1.3 to trigger hyperbolic/chaotic classification
    body.ecc = np.full_like(times, 1.5)

    result = classify_resonance(body, resonance)

    assert result['result'].status == ResonanceStatus.CHAOTIC


def test_result_to_flat_dict():
    """Test that ResonanceClassifyResult can be converted to flat dict."""
    times = np.linspace(0.0, 100.0, 2000) * 2 * np.pi
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / (10.0 * 2 * np.pi))) % (2 * np.pi)

    resonance = _create_mmr()
    body = _create_body_with_angles(times, angles, resonance)

    result = classify_resonance(body, resonance)
    flat = result['result'].to_flat_dict()

    assert isinstance(flat, dict)
    assert 'status' in flat
    assert 'type' in flat


# ── classify_from_data tests ──


def test_classify_from_data_libration():
    """classify_from_data on pure libration arrays returns LIBRATION."""
    times_yrs = np.linspace(0.0, 100.0, 2000)
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times_yrs / 10.0)) % (2 * np.pi)
    sigma_unwrapped = np.unwrap(angles)

    result = classify_from_data(times_yrs, angles, sigma_unwrapped)

    assert result['result'].status == ResonanceStatus.LIBRATION
    assert result['result'].subtype == 'pure libration'


def test_classify_from_data_circulation():
    """classify_from_data on steady circulation returns NON_RESONANT."""
    times_yrs = np.linspace(0.0, 100.0, 2000)
    sigma_unwrapped = 0.3 * times_yrs  # monotonic increase
    angles = sigma_unwrapped % (2 * np.pi)

    result = classify_from_data(times_yrs, angles, sigma_unwrapped)

    assert result['result'].status == ResonanceStatus.NON_RESONANT


def test_classify_from_data_matches_classify_resonance():
    """classify_from_data and classify_resonance produce the same status."""
    times = np.linspace(0.0, 100.0, 2000) * 2 * np.pi
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / (10.0 * 2 * np.pi))) % (2 * np.pi)

    resonance = _create_mmr()
    body = _create_body_with_angles(times, angles, resonance)

    body_result = classify_resonance(body, resonance)

    times_yrs = times / (2 * np.pi)
    sigma_w = body.angles_filtered[resonance.to_s()]
    sigma_u = body.angles_filtered_unwrapped[resonance.to_s()]
    data_result = classify_from_data(times_yrs, sigma_w, sigma_u)

    assert body_result['result'].status == data_result['result'].status
    assert body_result['result'].subtype == data_result['result'].subtype


# ── classify_from_metrics tests ──


def test_classify_from_metrics_pure_libration():
    """Pre-computed metrics with low rev and low TTO → pure libration."""
    metrics = SegmentMetrics(revolutions_true=0.3, trend_to_oscillation=0.1)
    counts = SegmentCounts()
    result = classify_from_metrics(metrics, counts, {})

    assert result['result'].status == ResonanceStatus.LIBRATION
    assert result['result'].subtype == 'pure libration'
    assert result['result'].confidence == 'high'


def test_classify_from_metrics_partial_libration():
    """Low rev but moderate TTO → partial libration."""
    metrics = SegmentMetrics(revolutions_true=0.5, trend_to_oscillation=1.5)
    counts = SegmentCounts()
    result = classify_from_metrics(metrics, counts, {})

    assert result['result'].status == ResonanceStatus.LIBRATION
    assert result['result'].subtype == 'partial libration'


def test_classify_from_metrics_slow_circulation():
    """Low rev but very high TTO → probably slow circulation."""
    metrics = SegmentMetrics(revolutions_true=0.8, trend_to_oscillation=3.0)
    counts = SegmentCounts()
    result = classify_from_metrics(metrics, counts, {})

    assert result['result'].status == ResonanceStatus.PROBABLY_SLOW_CIRCULATION


def test_classify_from_metrics_non_resonant_high_tto():
    """High rev + very high TTO → non-resonant circulation."""
    metrics = SegmentMetrics(revolutions_true=5.0, trend_to_oscillation=7.0)
    counts = SegmentCounts()
    result = classify_from_metrics(metrics, counts, {})

    assert result['result'].status == ResonanceStatus.NON_RESONANT
    assert result['result'].subtype == 'high trend to oscillation'


def test_classify_from_metrics_non_resonant_remaining():
    """Moderate rev + moderate TTO, no good/reasonable segments → remaining."""
    metrics = SegmentMetrics(revolutions_true=3.0, trend_to_oscillation=4.0)
    counts = SegmentCounts()
    result = classify_from_metrics(metrics, counts, {})

    assert result['result'].status == ResonanceStatus.NON_RESONANT
    assert result['result'].subtype == 'remaining'


def test_classify_from_metrics_transient_with_good_segments():
    """Good segments + moderate global TTO → transient."""
    metrics = SegmentMetrics(revolutions_true=3.0, trend_to_oscillation=2.0)
    # Counts must reflect actual segment quality (1 good, 1 reasonable)
    counts = SegmentCounts(n_good_total=1, n_reasonable_total=1, n_good_0_1=1, n_reasonable_0_1=1)
    good_seg = SegmentMetrics(revolutions_true=0.5, trend_to_oscillation=0.2)
    segments = {"0.1": {"0.00-10.00": good_seg}}

    result = classify_from_metrics(metrics, counts, segments)

    assert result['result'].status == ResonanceStatus.TRANSIENT


def test_classify_from_metrics_near_separatrix_with_good_segments():
    """Good segments but high global TTO → near separatrix."""
    metrics = SegmentMetrics(revolutions_true=3.0, trend_to_oscillation=4.0)
    counts = SegmentCounts(n_good_total=1, n_reasonable_total=1, n_good_0_1=1, n_reasonable_0_1=1)
    good_seg = SegmentMetrics(revolutions_true=0.5, trend_to_oscillation=0.2)
    segments = {"0.1": {"0.00-10.00": good_seg}}

    result = classify_from_metrics(metrics, counts, segments)

    assert result['result'].status == ResonanceStatus.NEAR_SEPARATRIX


def test_classify_from_metrics_probably_near_separatrix():
    """No good segments but reasonable + moderate TTO → probably near separatrix."""
    metrics = SegmentMetrics(revolutions_true=3.0, trend_to_oscillation=4.0)
    # Only reasonable, not good
    counts = SegmentCounts(n_good_total=0, n_reasonable_total=1, n_reasonable_0_1=1)
    reasonable_seg = SegmentMetrics(revolutions_true=1.5, trend_to_oscillation=0.8)
    segments = {"0.1": {"0.00-10.00": reasonable_seg}}

    result = classify_from_metrics(metrics, counts, segments)

    assert result['result'].status == ResonanceStatus.PROBABLY_NEAR_SEPARATRIX


# ── ClassifyParams override tests ──


def test_classify_params_stricter_tto_changes_result():
    """Lowering tto_pure_libration turns a pure libration into partial."""
    metrics = SegmentMetrics(revolutions_true=0.3, trend_to_oscillation=0.3)
    counts = SegmentCounts()

    # Default: TTO 0.3 < 0.5 → pure libration
    result_default = classify_from_metrics(metrics, counts, {})
    assert result_default['result'].subtype == 'pure libration'

    # Stricter: TTO 0.3 >= 0.2 → partial libration
    strict = ClassifyParams(tto_pure_libration=0.2)
    result_strict = classify_from_metrics(metrics, counts, {}, params=strict)
    assert result_strict['result'].subtype == 'partial libration'


def test_classify_params_relaxed_non_resonant_changes_result():
    """Raising tto_non_resonant can turn circulation into remaining."""
    metrics = SegmentMetrics(revolutions_true=5.0, trend_to_oscillation=7.0)
    counts = SegmentCounts()

    # Default tto_non_resonant=6.0 → "high trend to oscillation"
    result_default = classify_from_metrics(metrics, counts, {})
    assert result_default['result'].subtype == 'high trend to oscillation'

    # Relaxed tto_non_resonant=8.0 → falls through to "remaining"
    relaxed = ClassifyParams(tto_non_resonant=8.0)
    result_relaxed = classify_from_metrics(metrics, counts, {}, params=relaxed)
    assert result_relaxed['result'].subtype == 'remaining'


def test_classify_params_segment_thresholds_affect_good():
    """Tightening good_seg thresholds can remove segments from good count."""
    metrics = SegmentMetrics(revolutions_true=3.0, trend_to_oscillation=2.0)
    # Segment with rev=0.8, tto=0.4 — good with defaults
    seg = SegmentMetrics(revolutions_true=0.8, trend_to_oscillation=0.4)
    segments = {"0.1": {"0.00-10.00": seg}}

    # With defaults: segment is good → transient
    counts_good = SegmentCounts(n_good_total=1, n_reasonable_total=1, n_good_0_1=1, n_reasonable_0_1=1)
    result_default = classify_from_metrics(metrics, counts_good, segments)
    assert result_default['result'].status == ResonanceStatus.TRANSIENT

    # Tighten good_seg_max_tto to 0.3 → segment no longer good, but still reasonable
    strict = ClassifyParams(good_seg_max_tto=0.3)
    counts_no_good = SegmentCounts(n_good_total=0, n_reasonable_total=1, n_reasonable_0_1=1)
    result_strict = classify_from_metrics(metrics, counts_no_good, segments, params=strict)
    assert result_strict['result'].status != ResonanceStatus.TRANSIENT


def test_classify_from_data_with_custom_params():
    """classify_from_data respects custom ClassifyParams."""
    times_yrs = np.linspace(0.0, 100.0, 2000)
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times_yrs / 10.0)) % (2 * np.pi)
    sigma_unwrapped = np.unwrap(angles)

    # Default → pure libration
    result_default = classify_from_data(times_yrs, angles, sigma_unwrapped)
    assert result_default['result'].subtype == 'pure libration'

    # Stricter threshold → partial libration
    strict = ClassifyParams(tto_pure_libration=0.01)
    result_strict = classify_from_data(times_yrs, angles, sigma_unwrapped, params=strict)
    assert result_strict['result'].subtype == 'partial libration'


# ── check_chaos tests ──


class _FakeBody:
    """Minimal body-like object for check_chaos tests."""

    def __init__(self, axis, ecc):
        self.axis = np.asarray(axis, dtype=float)
        self.ecc = np.asarray(ecc, dtype=float)


def test_check_chaos_normal():
    """Normal orbit → flag 0."""
    body = _FakeBody(axis=[2.5, 2.6, 2.55], ecc=[0.1, 0.12, 0.11])
    flag, comment = check_chaos(body)
    assert flag == 0
    assert comment == ""


def test_check_chaos_hyperbolic():
    """Eccentricity > 1.3 → flag 1 (unphysical)."""
    body = _FakeBody(axis=[2.5, 2.6, 2.55], ecc=[0.1, 1.5, 0.3])
    flag, comment = check_chaos(body)
    assert flag == 1
    assert "hyperbolic" in comment


def test_check_chaos_negative_sma():
    """Negative semi-major axis → flag 1 (unphysical)."""
    body = _FakeBody(axis=[2.5, -1.0, 2.55], ecc=[0.1, 0.2, 0.1])
    flag, comment = check_chaos(body)
    assert flag == 1
    assert "negative_sma" in comment


def test_check_chaos_large_sma_change_final():
    """Final a differs > 100% from initial → flag -1."""
    # a0=2.5, a_final=0.5 → |da|/a0 = 0.8 (80%) — not enough
    # a0=2.5, a_final=6.0 → |da|/a0 = 1.4 (140%) — triggers
    body = _FakeBody(axis=[2.5, 2.6, 6.0], ecc=[0.1, 0.1, 0.1])
    flag, comment = check_chaos(body)
    assert flag == -1
    assert "|da|>100%" in comment
    assert "da_final" in comment


def test_check_chaos_large_sma_change_max():
    """Max a differs > 100% from initial but final is close → flag -1."""
    # a0=2.5, max=5.5 → |da|/a0 = 1.2 (120%), final=2.6 → only 4%
    body = _FakeBody(axis=[2.5, 5.5, 2.6], ecc=[0.1, 0.1, 0.1])
    flag, comment = check_chaos(body)
    assert flag == -1
    assert "da_max" in comment


def test_check_chaos_zero_a0():
    """a0 = 0 → flag 1 (unphysical, division guard)."""
    body = _FakeBody(axis=[0.0, 2.5, 2.5], ecc=[0.1, 0.1, 0.1])
    flag, comment = check_chaos(body)
    assert flag == 1
    assert "a0=0" in comment


def test_check_chaos_borderline_99_percent():
    """99% change should NOT trigger flag -1."""
    # a0=2.5, a_final=4.975 → |da|/a0 = 0.99
    body = _FakeBody(axis=[2.5, 2.5, 4.975], ecc=[0.1, 0.1, 0.1])
    flag, comment = check_chaos(body)
    assert flag == 0
