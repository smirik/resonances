import numpy as np

from resonances.resonance.classify import (
    classify_resonance,
    compute_angle_difference,
    compute_angle_uniformity,
)
from resonances.body import Body
from resonances.mmr.two_body import TwoBody
from resonances.secular.secular_resonance import SecularResonance


def _create_mmr():
    """Create a simple MMR for testing."""
    return TwoBody("2J-1")  # 2:1 Jupiter MMR


def _create_secular():
    """Create a simple secular resonance for testing."""
    return SecularResonance("g-g5")  # nu5 secular resonance


def _create_body_with_angles(angles, resonance, angle_peaks=None, axis_peaks=None):
    """Create a Body object with the given angles and periodogram peaks."""
    body = Body()
    body.name = "test_body"
    body.angles[resonance.to_s()] = angles
    body.periodogram_peaks[resonance.to_s()] = angle_peaks
    body.axis_periodogram_peaks = axis_peaks

    # Register the resonance with the body
    if isinstance(resonance, TwoBody):
        body.mmrs.append(resonance)
    elif isinstance(resonance, SecularResonance):
        body.secular_resonances.append(resonance)

    return body


def test_compute_angle_difference_wraps_correctly():
    forward = compute_angle_difference(6.1, 0.2)
    backward = compute_angle_difference(0.2, 6.1)

    assert forward > 0
    assert backward < 0
    assert np.isclose(abs(forward), abs(backward), atol=1e-6)


def test_classify_libration_with_matching_peaks():
    times = np.linspace(0.0, 100.0, 2000)
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / 10.0)) % (2 * np.pi)

    # Create mock periodogram peaks that overlap
    angle_peaks = {'position': [(9.0, 11.0)], 'peaks': [0]}
    axis_peaks = {'position': [(9.5, 10.5)], 'peaks': [0]}

    resonance = _create_mmr()
    body = _create_body_with_angles(angles, resonance, angle_peaks, axis_peaks)

    result = classify_resonance(body, times, resonance=resonance)

    assert result['status'] == 2


def test_classify_libration_no_matching_peaks():
    times = np.linspace(0.0, 100.0, 2000)
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / 10.0)) % (2 * np.pi)

    # Create mock periodogram peaks that don't overlap
    angle_peaks = {'position': [(9.0, 11.0)], 'peaks': [0]}
    axis_peaks = {'position': [(20.0, 22.0)], 'peaks': [0]}

    resonance = _create_mmr()
    body = _create_body_with_angles(angles, resonance, angle_peaks, axis_peaks)

    result = classify_resonance(body, times, resonance=resonance)

    assert result['status'] == -2


def test_classify_libration_secular_resonance():
    times = np.linspace(0.0, 100.0, 2000)
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / 10.0)) % (2 * np.pi)

    # For secular resonances, no periodogram check
    resonance = _create_secular()
    body = _create_body_with_angles(angles, resonance)

    result = classify_resonance(body, times, resonance=resonance)

    assert result['status'] == 2


def test_classify_circulation():
    times = np.linspace(0.0, 100.0, 2000)
    angles = (0.5 * times) % (2 * np.pi)

    # Create mock periodogram peaks (won't matter for circulation)
    angle_peaks = {'position': [(9.0, 11.0)], 'peaks': [0]}
    axis_peaks = {'position': [(9.5, 10.5)], 'peaks': [0]}

    resonance = _create_mmr()
    body = _create_body_with_angles(angles, resonance, angle_peaks, axis_peaks)

    result = classify_resonance(body, times, resonance=resonance)

    assert result['status'] == 0


def test_classify_none_periodograms():
    times = np.linspace(0.0, 100.0, 2000)
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / 10.0)) % (2 * np.pi)

    # With None periodograms for MMR, should return uncertain status
    resonance = _create_mmr()
    body = _create_body_with_angles(angles, resonance, None, None)

    result = classify_resonance(body, times, resonance=resonance)

    assert result['status'] == -2


def test_classify_returns_all_fields():
    times = np.linspace(0.0, 100.0, 2000)
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / 10.0)) % (2 * np.pi)

    angle_peaks = {'position': [(9.0, 11.0)], 'peaks': [0]}
    axis_peaks = {'position': [(9.5, 10.5)], 'peaks': [0]}

    resonance = _create_mmr()
    body = _create_body_with_angles(angles, resonance, angle_peaks, axis_peaks)

    result = classify_resonance(body, times, resonance=resonance)

    # Check all required fields are present
    assert 'status' in result
    assert 'classification_status' in result
    assert 'libration_fraction' in result
    assert 'is_circulation' in result
    assert 'drift_rate' in result
    assert 'r_squared' in result
    assert 'total_drift_cycles' in result
    assert 'n_libration_segments' in result
    assert 'cumulative_drift' in result
    assert 'overlapping_peaks' in result
    assert 'n_angle_peaks' in result
    assert 'n_axis_peaks' in result
    assert 'has_overlap' in result

    # Check values
    assert result['status'] == 2
    assert result['classification_status'] == 2
    assert result['has_overlap'] is True


def test_classify_secular_resonance_fields():
    times = np.linspace(0.0, 100.0, 2000)
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / 10.0)) % (2 * np.pi)

    resonance = _create_secular()
    body = _create_body_with_angles(angles, resonance)

    result = classify_resonance(body, times, resonance=resonance)

    assert result['status'] == 2
    assert result['classification_status'] == 2
    assert result['has_overlap'] is False
    assert result['overlapping_peaks'] == []


def test_compute_angle_uniformity_libration():
    """Test that libration has low uniformity (angles concentrated)."""
    # Libration: angles clustered around π
    angles = np.pi + 0.3 * np.sin(np.linspace(0, 20 * np.pi, 1000))
    uniformity = compute_angle_uniformity(angles)
    assert uniformity < 0.1  # Should be very low


def test_compute_angle_uniformity_chaotic():
    """Test that chaotic behavior has high uniformity (angles spread evenly)."""
    # Random angles covering full range
    np.random.seed(42)
    angles = np.random.uniform(0, 2 * np.pi, 1000)
    uniformity = compute_angle_uniformity(angles)
    assert uniformity > 0.5  # Should be high


def test_classify_chaotic_as_nonresonant():
    """Test that chaotic behavior (uniform angle distribution) is classified as 0."""
    times = np.linspace(0.0, 100.0, 2000)
    # Random angles filling the entire 0-2π range uniformly
    np.random.seed(42)
    angles = np.random.uniform(0, 2 * np.pi, 2000)

    angle_peaks = {'position': [(9.0, 11.0)], 'peaks': [0]}
    axis_peaks = {'position': [(9.5, 10.5)], 'peaks': [0]}

    resonance = _create_mmr()
    body = _create_body_with_angles(angles, resonance, angle_peaks, axis_peaks)

    result = classify_resonance(body, times, resonance=resonance)

    # Should be classified as 0 (non-resonant) due to high uniformity
    assert result['status'] == 0
    assert result['angle_uniformity'] > 0.5
