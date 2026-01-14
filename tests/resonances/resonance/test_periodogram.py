import numpy as np

from resonances.resonance.periodogram import Periodogram


def test_lomb_scargle_detects_frequency():
    t = np.linspace(0.0, 10.0, 1000)
    freq = 0.5
    y = np.sin(2 * np.pi * freq * t)

    frequency, power = Periodogram.lomb_scargle(t, y, minimum_frequency=0.1, maximum_frequency=1.0)
    peak = frequency[np.argmax(power)]

    assert np.isclose(peak, freq, atol=0.02)


def test_periodogram_returns_peaks():
    t_years = np.linspace(0.0, 10.0, 1000)
    t_sim = t_years * (2 * np.pi)
    y = np.sin(2 * np.pi * 0.5 * t_years)

    frequency, power, peaks = Periodogram.periodogram(
        t_sim,
        y,
        integration_time_yrs=10.0,
        Nout=len(t_sim),
        libration_period_min=0,
        minimum_frequency=0.0,
        maximum_frequency=1.0,
        threshold=0.1,
    )

    assert frequency is not None
    assert power is not None
    assert peaks is not None
    assert "position" in peaks
    assert "peaks" in peaks
