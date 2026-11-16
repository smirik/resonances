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


def test_overlap_no_overlap():
    a = (1.0, 2.0)
    b = (3.0, 4.0)
    assert Periodogram.overlap(a, b) == 0


def test_overlap_partial():
    a = (1.0, 3.0)
    b = (2.0, 4.0)
    overlap = Periodogram.overlap(a, b)
    assert overlap == 1.0


def test_overlap_contained():
    a = (1.0, 5.0)
    b = (2.0, 3.0)
    overlap = Periodogram.overlap(a, b)
    assert overlap == 1.0


def test_overlap_with_delta():
    a = (1.0, 2.0)
    b = (2.5, 3.5)
    # Without delta, no overlap
    assert Periodogram.overlap(a, b) == 0
    # With delta=0.5, they should overlap
    assert Periodogram.overlap(a, b, delta=0.5) == 0.5


def test_overlap_list_no_matches():
    a_list = [(1.0, 2.0), (3.0, 4.0)]
    b_list = [(5.0, 6.0), (7.0, 8.0)]
    result = Periodogram.overlap_list(a_list, b_list)
    assert result == []


def test_overlap_list_some_matches():
    a_list = [(1.0, 2.0), (3.0, 5.0), (10.0, 11.0)]
    b_list = [(4.0, 6.0), (20.0, 21.0)]
    result = Periodogram.overlap_list(a_list, b_list)
    assert len(result) == 1
    assert result[0] == (3.0, 5.0)


def test_overlap_list_all_matches():
    a_list = [(1.0, 3.0), (5.0, 7.0)]
    b_list = [(2.0, 4.0), (6.0, 8.0)]
    result = Periodogram.overlap_list(a_list, b_list)
    assert len(result) == 2


def test_overlap_list_with_delta():
    a_list = [(1.0, 2.0)]
    b_list = [(2.5, 3.5)]
    # Without delta, no match
    assert Periodogram.overlap_list(a_list, b_list) == []
    # With delta=0.5, should match
    result = Periodogram.overlap_list(a_list, b_list, delta=0.5)
    assert len(result) == 1
