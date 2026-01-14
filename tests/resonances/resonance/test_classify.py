import numpy as np

from resonances.resonance.classify import classify_resonance, compute_angle_difference


def test_compute_angle_difference_wraps_correctly():
    forward = compute_angle_difference(6.1, 0.2)
    backward = compute_angle_difference(0.2, 6.1)

    assert forward > 0
    assert backward < 0
    assert np.isclose(abs(forward), abs(backward), atol=1e-6)


def test_classify_libration():
    times = np.linspace(0.0, 100.0, 2000)
    angles = (np.pi + 0.5 * np.sin(2 * np.pi * times / 10.0)) % (2 * np.pi)

    status = classify_resonance(times, angles)

    assert status == 2


def test_classify_circulation():
    times = np.linspace(0.0, 100.0, 2000)
    angles = (0.5 * times) % (2 * np.pi)

    status = classify_resonance(times, angles)

    assert status == 0
