import numpy as np
import pytest

from resonances.resonance.analysis.filtering import butterworth_gain, edge_window, lowpass, time_derivative, window_mask


def test_gain_analytic_values_order_two():
    # 1 / (1 + (P_c/P)^4): 1/(1 + 1/16) = 16/17, 1/2, 1/(1 + 1.5^4) = 1/6.0625
    assert butterworth_gain(200.0, 100.0, 2) == pytest.approx(16 / 17)
    assert butterworth_gain(100.0, 100.0, 2) == pytest.approx(0.5)
    assert butterworth_gain(100.0 / 1.5, 100.0, 2) == pytest.approx(1 / 6.0625)
    gains = butterworth_gain(np.array([1000.0, 100.0]), 100.0, 2)
    assert gains == pytest.approx([1 / (1 + 1e-4), 0.5])


@pytest.mark.parametrize('ratio', [10.0, 2.0, 1.0, 1 / 1.5])
def test_lowpass_amplitude_matches_gain(ratio):
    t = np.arange(0, 40000, 1.0)
    cutoff = 100.0
    period = ratio * cutoff
    y = lowpass(t, np.sin(2 * np.pi * t / period), cutoff, 2)
    middle = (t > 5000) & (t < 35000)
    assert np.max(np.abs(y[middle])) == pytest.approx(butterworth_gain(period, cutoff, 2), rel=1e-2)


def test_lowpass_keeps_the_trend_exactly():
    t = np.arange(0, 5000, 0.5)
    line = 3.0 + 0.01 * t
    y = lowpass(t, line + 0.2 * np.sin(2 * np.pi * t / 7.0), 100.0, 2)
    inside = window_mask(t, edge_window(t, 100.0))  # filtfilt edges spread over ~P_c
    assert np.max(np.abs(y - line)[inside]) < 1e-3


def test_lowpass_rejects_bad_grids_and_cutoffs():
    t = np.arange(0, 100, 1.0)
    with pytest.raises(ValueError):
        lowpass(t, np.sin(t), 2.0, 2)
    t_bad = t.copy()
    t_bad[50] += 0.4
    with pytest.raises(ValueError):
        lowpass(t_bad, np.sin(t_bad), 20.0, 2)


def test_derivative_and_windows():
    t = np.arange(0, 100, 0.01)
    assert time_derivative(t, np.sin(t))[100:-100] == pytest.approx(np.cos(t)[100:-100], abs=1e-4)
    assert edge_window(t, 10.0) == pytest.approx((10.0, t[-1] - 10.0))
    assert all(np.isnan(edge_window(t, 60.0)))
    mask = window_mask(t, (10.0, 20.0))
    assert t[mask].min() >= 10.0 and t[mask].max() <= 20.0 and mask.sum() == 1001
    assert not window_mask(t, (np.nan, np.nan)).any()
