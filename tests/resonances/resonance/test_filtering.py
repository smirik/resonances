import resonances
import numpy as np
import pytest


def test_periodogram():
    t = np.linspace(0, 10, 101)
    y = np.sin(np.pi * t)
    (frequency, power) = resonances.libration.periodogram(t, y, minimum_frequency=0.0, maximum_frequency=2.0)

    key = 0
    for i, elem in enumerate(frequency):
        if elem == 0.5:  # the frequency (1 period in 2s => 1/2 = 0.5 Hz)
            key = i

    assert 1.0 == pytest.approx(power[key], 0.1)


def test_find_peaks_with_position():
    t = np.linspace(0, 10, 101)
    y = np.sin(np.pi * t) + np.sin(2 * np.pi * t)  # 0.5 Hz and 1 Hz
    (frequency, power) = resonances.libration.periodogram(t, y, minimum_frequency=0.0, maximum_frequency=2.0)

    peaks = resonances.libration.find_peaks_with_position(frequency, power)

    keys = []
    for i, elem in enumerate(frequency):
        if (elem == 0.5) or (elem == 1.0):  # 0.5 and 1 Hz
            keys.append(i)

    # find_peaks returns times, not frequencies
    assert True == (peaks['position'][0][0] <= 1.0 / 1.0 <= peaks['position'][0][1])
    assert True == (peaks['position'][1][0] <= 1.0 / 0.5 <= peaks['position'][1][1])

    for key in keys:
        assert key in peaks['peaks']


def test_filter():
    t = np.linspace(0, 10, 101)
    y = np.sin(np.pi * t) + np.sin(2 * np.pi * t)  # 0.5 Hz and 1 Hz
    (frequency, power) = resonances.libration.periodogram(t, y, minimum_frequency=0.0, maximum_frequency=2.0)
    y_filtered = resonances.libration.butter_lowpass_filter(y, 0.6, 1, 2, 5)
    frequency, power = resonances.libration.periodogram(t, y_filtered, minimum_frequency=0.0, maximum_frequency=2.0)
    peaks = resonances.libration.find_peaks_with_position(frequency, power, height=0.2)

    assert True == (peaks['position'][0][0] <= 1.0 / 0.5 <= peaks['position'][0][1])
    assert 1 == len(peaks['position'])
