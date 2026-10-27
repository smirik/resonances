"""W3 G5: the exact gain of the forward-backward digital Butterworth (analysis.filtering).

Truths (Codex astra §3): dt = 1, P_c = 5, N = 2. tan(pi/10)/tan(pi/5) = 1/sqrt(5), so G(10) = 1/(1 + 1/25) = 25/26;
tan(3 pi/10)/tan(pi/5) = 1/tan^2(pi/5) = 1/(5 - 2 sqrt 5), so G(10/3) = 1/(1 + (5 - 2 sqrt 5)^-4) = 0.0720468; the
analog form gives 16/17 = 0.941 and 0.165. Checked against scipy's own frequency response and against the amplitude
`lowpass` actually leaves on a sinusoid.
"""

import numpy as np
import pytest
from scipy import signal

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.settings import settings_from_config


def test_digital_gain_closed_forms():
    assert filtering.butterworth_gain(10.0, 5.0, 2, step_yr=1.0) == pytest.approx(25 / 26, rel=1e-12)
    assert filtering.butterworth_gain(10 / 3, 5.0, 2, step_yr=1.0) == pytest.approx(1 / (1 + (5 - 2 * np.sqrt(5)) ** -4), rel=1e-12)
    assert filtering.butterworth_gain(10 / 3, 5.0, 2, step_yr=1.0) == pytest.approx(0.0720467543, abs=1e-10)
    # the dt -> 0 limit (no step) is the analog form
    assert filtering.butterworth_gain(10.0, 5.0, 2) == pytest.approx(16 / 17, rel=1e-12)
    # at the cutoff the gain is 1/2 in both; a period at or above Nyquist (P <= 2 dt) has no gain; P = inf passes
    assert filtering.butterworth_gain(5.0, 5.0, 2, step_yr=1.0) == pytest.approx(0.5, rel=1e-12)
    assert np.isnan(filtering.butterworth_gain(2.0, 5.0, 2, step_yr=1.0))
    assert np.isnan(filtering.butterworth_gain(1.4, 5.0, 2, step_yr=1.0))
    assert filtering.butterworth_gain(np.inf, 5.0, 2, step_yr=1.0) == 1.0


@pytest.mark.parametrize('step, cutoff, order', [(1.0, 5.0, 2), (0.5, 3.0, 3), (2.0, 100.0, 2)])
def test_gain_equals_the_squared_response_of_the_sos_filter(step, cutoff, order):
    periods = np.array([2.2 * step, 3 * step, cutoff / 2, cutoff, 2 * cutoff, 10 * cutoff])
    sos = signal.butter(order, 1 / cutoff, fs=1 / step, output='sos')
    _, response = signal.sosfreqz(sos, worN=1 / periods, fs=1 / step)
    np.testing.assert_allclose(filtering.butterworth_gain(periods, cutoff, order, step_yr=step), np.abs(response) ** 2, rtol=1e-9)


def test_gain_is_what_lowpass_leaves_on_a_sinusoid():
    t = np.arange(0.0, 4000.0, 1.0)
    for period, expected in ((10.0, 25 / 26), (10 / 3, 0.0720467543)):
        out = filtering.lowpass(t, np.sin(2 * np.pi * t / period), 5.0, 2, detrend=False)
        middle = out[500:-500]
        amplitude = np.sqrt(2 * np.mean(middle**2))
        assert amplitude == pytest.approx(expected, rel=1e-3)
        assert abs(amplitude - filtering.butterworth_gain(period, 5.0, 2)) > 0.019  # the analog form misses it


def test_uniform_grid_tolerance_is_a_setting():
    assert settings_from_config().filter_grid_rtol == 1e-3 == filtering.DEFAULT_GRID_RTOL
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 500.0, 1.0) + rng.uniform(-0.004, 0.004, 500)  # steps off by up to 0.8 %
    with pytest.raises(ValueError):
        filtering.lowpass(t, np.sin(t / 30), 20.0, 2)
    out = filtering.lowpass(t, np.sin(t / 30), 20.0, 2, grid_rtol=0.01)
    assert out.shape == t.shape
