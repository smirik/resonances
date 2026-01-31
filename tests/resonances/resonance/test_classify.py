import numpy as np

from resonances.resonance.classify import classify_resonance, ResonanceStatus
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
    assert 'extra' in result
    assert hasattr(result['result'], 'status')
    assert hasattr(result['result'], 'type')
    assert hasattr(result['result'], 'metrics')


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
