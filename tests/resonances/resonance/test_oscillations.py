"""Oscillation diagnostics on analytic signals whose answers are known in closed form."""

import numpy as np
import pytest

from resonances.resonance import oscillations as osc
from resonances.resonance.classify.classify import calc_sigma_derivative

T = np.linspace(0, 10, 10001)  # 10 periods of 1 yr, 1000 samples per period


class TestProminence:
    @pytest.mark.parametrize(
        'span, expected',
        [
            (4.0, 0.1),  # 5% = 0.2, capped
            (1.0, 0.05),  # 5% inside the bounds
            (0.02, 0.005),  # 5% = 0.001, floored
        ],
    )
    def test_five_percent_within_bounds(self, span, expected):
        assert osc.default_prominence(np.array([0.0, span])) == pytest.approx(expected)


class TestFindCycles:
    def test_cosine(self):
        # 2 cos(2 pi t): maxima at t = 0..10; the two end ones are not peaks, which leaves
        # 9 interior maxima and 8 full cycles of period 1, diameter 4, centre 0.
        cycles = osc.find_cycles(T, 2 * np.cos(2 * np.pi * T))
        assert len(cycles) == 8
        assert [c.start for c in cycles] == pytest.approx(list(range(1, 9)))
        for c in cycles:
            assert c.period == pytest.approx(1.0)
            assert c.diameter == pytest.approx(4.0)
            assert c.centre == pytest.approx(0.0, abs=1e-9)

    def test_monotone_angle_has_no_cycles(self):
        assert osc.find_cycles(T, 3.0 * T) == []

    def test_drifting_centre(self):
        # 0.5 t + 2 cos(2 pi t): every maximum sits at the same phase, so consecutive
        # centres move by exactly 0.5 rad per period (0.5 rad/yr times 1 yr).
        cycles = osc.find_cycles(T, 0.5 * T + 2 * np.cos(2 * np.pi * T))
        centres = np.array([c.centre for c in cycles])
        assert np.diff(centres) == pytest.approx(np.full(len(centres) - 1, 0.5), abs=1e-3)

    def test_prominence_filters_ripples(self):
        # A 0.01 rad ripple on a 0.2 rad cosine: explicit prominence 0.05 keeps only the
        # large oscillation, a tiny one also counts the ripple maxima.
        sigma = 0.1 * np.cos(2 * np.pi * T) + 0.005 * np.cos(2 * np.pi * 37 * T)
        assert len(osc.find_cycles(T, sigma, prominence=0.05)) == 8
        assert len(osc.find_cycles(T, sigma, prominence=1e-4)) > 8


class TestFoldAndCentres:
    def test_fold_cycle_maps_one_period_onto_unit_interval(self):
        sigma = 2 * np.cos(2 * np.pi * T)
        cycle = osc.find_cycles(T, sigma)[3]
        phase, values = osc.fold_cycle(cycle, T, sigma)
        assert phase[0] == pytest.approx(0) and phase[-1] == pytest.approx(1)
        assert values == pytest.approx(2 * np.cos(2 * np.pi * phase), abs=1e-9)

    def test_fold_subtracts_the_cycle_median(self):
        axis = 2.5 + 0.01 * np.sin(2 * np.pi * T)
        cycle = osc.find_cycles(T, 2 * np.cos(2 * np.pi * T))[0]
        _, values = osc.fold_cycle(cycle, T, axis, subtract_median=True)
        assert np.median(values) == pytest.approx(0, abs=1e-12)

    def test_axis_centre_is_midpoint_of_range(self):
        # a = 2.5 + 0.01 sin: over any full period its range is [2.49, 2.51].
        cycles = osc.find_cycles(T, 2 * np.cos(2 * np.pi * T))
        mid, centre, axis_mid = osc.cycle_centres(cycles, T, 2.5 + 0.01 * np.sin(2 * np.pi * T))
        assert mid == pytest.approx(np.arange(1.5, 9.5))
        assert centre == pytest.approx(np.zeros(8), abs=1e-9)
        assert axis_mid == pytest.approx(np.full(8, 2.5), abs=1e-6)


class TestRecurrence:
    def test_scales_of_a_cosine(self):
        # sigma = 2 cos(2 pi t): diameter 4 -> s_sigma = 2. rate = -4 pi sin(2 pi t); the
        # 5th and 95th percentiles of sin over whole periods are -+sin(0.45 pi).
        sigma = 2 * np.cos(2 * np.pi * T)
        rate = calc_sigma_derivative(T, sigma)
        s_sigma, s_rate = osc.recurrence_scales(sigma, rate, osc.find_cycles(T, sigma))
        assert s_sigma == pytest.approx(2.0)
        assert s_rate == pytest.approx(4 * np.pi * np.sin(0.45 * np.pi), rel=1e-3)

    def test_periodic_signal_returns_after_one_period(self):
        # 1001 samples over 10 periods: decimated index step 1 in time 0.01 yr; the state
        # one period (100 samples) later is the same state.
        t = np.linspace(0, 10, 1001)
        sigma = 2 * np.cos(2 * np.pi * t)
        rate = calc_sigma_derivative(t, sigma)
        rec = osc.recurrence(t, sigma, rate, 2.0, 4 * np.pi, max_points=2000, exclude_samples=2)
        assert rec.distance.shape == (1001, 1001)
        interior = np.arange(10, 890)
        assert np.nanmax(rec.distance[interior, interior + 100]) < 1e-3
        # Half a period later the angle is opposite: |d sigma| = 4 = 2 s_sigma at the maxima.
        assert rec.distance[100, 150] == pytest.approx(2.0, abs=1e-2)

    def test_excluded_band_and_symmetry(self):
        t = np.linspace(0, 10, 300)
        sigma = np.sin(t)
        rec = osc.recurrence(t, sigma, np.cos(t), 1.0, 1.0, max_points=300, exclude_samples=2)
        k = np.arange(300)
        band = np.abs(k[:, None] - k) <= 2
        assert np.isnan(rec.distance[band]).all()
        assert np.isfinite(rec.distance[~band]).all()
        assert np.array_equal(np.isnan(rec.distance), np.isnan(rec.distance.T))
        assert np.nanmax(np.abs(rec.distance - rec.distance.T)) == 0

    def test_decimation(self):
        t = np.linspace(0, 1, 4000)
        rec = osc.recurrence(t, t, np.ones_like(t), 1.0, 1.0, max_points=700)
        assert len(rec.times) == 700
        assert rec.times[0] == 0 and rec.times[-1] == 1

    def test_circulation_never_returns(self):
        # On the real line a monotone angle moves away for ever: the distance only grows
        # with the time separation (same rate, so it is |d sigma| / s_sigma exactly).
        t = np.linspace(0, 10, 200)
        sigma = 2 * np.pi * t
        rec = osc.recurrence(t, sigma, np.full_like(t, 2 * np.pi), 1.0, 1.0, max_points=200, exclude_samples=0)
        expected = np.abs(sigma[:, None] - sigma)
        off_diagonal = ~np.eye(200, dtype=bool)
        assert rec.distance[off_diagonal] == pytest.approx(expected[off_diagonal])
        assert np.isnan(np.diag(rec.distance)).all()  # 0 still excludes the diagonal


class TestFair:
    def test_exact_inner_resonance_lies_on_two_strips(self):
        # Inner 2:1 with the planet (n_p = n / 2) and sigma = 2 lambda_p - lambda = 0,
        # varpi = 0 so M = lambda. Then lambda_p - lambda = -lambda / 2: modulo 360 deg the
        # points lie on y = -x/2 or y = -x/2 + 180 (lambda / 2 runs over two turns).
        longitude = np.linspace(0, 40 * np.pi, 5000)
        planet_longitude = longitude / 2
        x, y = osc.fair_coordinates(longitude, longitude, planet_longitude, inner=True)
        line1 = np.mod(-x / 2, 360)
        line2 = np.mod(-x / 2 + 180, 360)
        on_line = np.minimum(_circular(y - line1), _circular(y - line2))
        assert np.max(on_line) < 1e-9
        assert x.min() >= 0 and x.max() < 360

    def test_outer_flips_the_difference(self):
        rng = np.random.default_rng(1)
        m, lam, lam_p = rng.uniform(-50, 50, (3, 100))
        _, inner = osc.fair_coordinates(m, lam, lam_p, inner=True)
        _, outer = osc.fair_coordinates(m, lam, lam_p, inner=False)
        assert _circular(inner + outer).max() < 1e-9

    def test_is_inner(self):
        assert osc.is_inner(np.full(5, 2.5), np.full(5, 5.2)) is True
        assert osc.is_inner(np.full(5, 1.5), np.full(5, 1.0)) is False

    def test_step_fraction(self):
        # a = 4 au -> P = 8 yr; a 2-yr output step is 1/4 of the orbit.
        assert osc.fair_step_fraction(np.arange(0, 100, 2.0), np.full(50, 4.0)) == pytest.approx(0.25)


def _circular(degrees):
    return np.abs((np.asarray(degrees) + 180) % 360 - 180)
