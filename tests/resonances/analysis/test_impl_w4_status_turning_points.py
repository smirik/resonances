"""W4 status: `geometry.prominent_peaks` selects the same maxima as `scipy.signal.find_peaks(x, prominence=p)`.

The fast path measures prominences on the skeleton of local extrema with a monotone stack (W3 left
`find_turning_points` at ~19 s of 27 on a 10^6-point record). The reference is scipy's own sample-by-sample
search, on series built to stress it: ties (integer random walks), plateaus, a ripple on a trend (every ripple
maximum's left base is the start of the series), a slow libration and the ends of the series.
"""

import time

import numpy as np
import pytest
from scipy import signal

from resonances.resonance.analysis.geometry import find_turning_points, prominent_peaks


def scipy_peaks(x, prominence):
    return signal.find_peaks(x, prominence=prominence)[0]


def series_cases():
    rng = np.random.default_rng(20261016)
    t = np.linspace(0.0, 1000.0, 20001)
    yield 'integer random walk (ties)', np.cumsum(rng.integers(-2, 3, 5000)).astype(float), [0.5, 1.0, 3.0, 10.0]
    yield 'plateaus', np.repeat(rng.normal(size=400), rng.integers(1, 5, 400)), [0.1, 0.5, 1.5]
    trend = 0.01 * t + 0.02 * np.sin(2 * np.pi * t / 3.0)
    yield 'ripple on a trend', trend, [0.001, 0.05, 0.2]
    libration = 1.2 * np.sin(2 * np.pi * t / 137.0) * np.exp(-t / 2000.0) + 0.01 * np.sin(2 * np.pi * t / 2.3)
    yield 'slow libration with ripple', libration, [0.005, 0.05, 1.0]
    yield 'rising then falling ends', np.concatenate([np.linspace(0, 5, 50), np.linspace(5, -1, 80), [3.0, 2.0]]), [0.1, 2.0]
    yield 'constant', np.zeros(100), [0.1]
    yield 'random normal', rng.normal(size=3000), [0.0, 0.3, 1.0, 2.5]


@pytest.mark.parametrize('name,x,prominences', list(series_cases()), ids=[case[0] for case in series_cases()])
def test_same_maxima_and_minima_as_scipy(name, x, prominences):
    for prominence in prominences:
        assert np.array_equal(prominent_peaks(x, prominence), scipy_peaks(x, prominence)), (name, prominence)
        assert np.array_equal(prominent_peaks(-x, prominence), scipy_peaks(-x, prominence)), (name, prominence)


def test_hand_computed_prominences():
    """x = [0, 3, 1, 5, 2, 4, 0]: maxima 3 (prominence 3 - max(0, 1) = 2), 5 (5 - max(0, 0) = 5), 4 (4 - max(2, 0) = 2)."""
    x = np.array([0.0, 3.0, 1.0, 5.0, 2.0, 4.0, 0.0])
    assert list(prominent_peaks(x, 2.0)) == [1, 3, 5]
    assert list(prominent_peaks(x, 2.5)) == [3]
    assert list(prominent_peaks(x, 5.0)) == [3]
    assert list(prominent_peaks(x, 5.1)) == []


def test_turning_points_of_a_ripple_on_a_long_trend_are_fast():
    """10^6 samples of a circulating angle with 4 x 10^4 ripple maxima and two genuine turns: scipy's search walks to
    the start of the series from every ripple maximum (seconds); the skeleton takes a fraction of a second."""
    t = np.linspace(0.0, 2.0e6, 1_000_001)
    angle = 1e-4 * t + 0.002 * np.sin(2 * np.pi * t / 50.0) + 20.0 * np.exp(-(((t - 1.0e6) / 2.0e4) ** 2))
    start = time.perf_counter()
    turns = find_turning_points(t, angle, 0.5, (t[0], t[-1]))
    assert time.perf_counter() - start < 2.0
    # the bump (slope up to 20 sqrt(2/e) / 2e4 = 8.6e-4 against the trend's 1e-4) makes one maximum and the dip after
    # it one minimum; the ripple (prominence 0.004) makes none
    assert list(turns.kinds) == [1, -1]
    assert 1.0e6 < turns.times_yr[0] < 1.02e6 < turns.times_yr[1] < 1.1e6
