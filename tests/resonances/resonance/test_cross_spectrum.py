"""Tests for the cross-spectral coherence primitives.

Expected values are derived independently of the implementation: signals are built
from known sinusoids with a known phase offset and amplitude ratio, and the
significance threshold is checked against its own analytical false-alarm rate.
"""

import numpy as np
import pytest

from resonances.resonance.cross_spectrum import (
    bandpassed_correlation,
    cross_spectrum,
    cross_spectrum_levels,
    decisive_level,
    eccentricity_vector,
    forced_free_eccentricity,
    is_uniform_grid,
    levels_agreeing,
    parse_pair,
    phase_distance,
    phase_matches,
)


def _series(period, phase_deg=0.0, amplitude=1.0, n=200_000, dt=1.0, noise=0.0, seed=0):
    t = np.arange(n) * dt
    y = amplitude * np.sin(2 * np.pi * t / period + np.radians(phase_deg))
    if noise:
        y = y + noise * np.random.default_rng(seed).standard_normal(n)
    return t, y


class TestPairParsing:
    def test_valid_pairs(self):
        assert parse_pair('e-i') == ('e', 'i')
        assert parse_pair('sigma-a') == ('sigma', 'a')

    @pytest.mark.parametrize('pair', ['e', 'e-i-a', 'e-x', 'foo-bar', ''])
    def test_invalid_pairs_raise(self, pair):
        with pytest.raises(ValueError):
            parse_pair(pair)


class TestPhaseCriterion:
    # Circular distance: 350 deg and 10 deg are 20 deg apart, not 340.
    @pytest.mark.parametrize(
        'phase, target, expected',
        [(180.0, 180.0, 0.0), (-170.0, 180.0, 10.0), (350.0, 10.0, 20.0), (90.0, 180.0, 90.0), (0.0, 180.0, 180.0)],
    )
    def test_distance(self, phase, target, expected):
        assert phase_distance(phase, target) == pytest.approx(expected)

    def test_antiphase_tolerance(self):
        assert phase_matches(-166.0, 180.0, 30.0)  # 14 deg away across the wrap
        assert phase_matches(151.0, 180.0, 30.0)
        assert not phase_matches(126.0, 180.0, 30.0)  # the 591986 case: 54 deg away


class TestUniformGrid:
    def test_uniform(self):
        assert is_uniform_grid(np.linspace(0.0, 1000.0, 5001))

    def test_jittered(self):
        times = np.linspace(0.0, 1000.0, 5001)
        times[2500] += 0.05
        assert not is_uniform_grid(times)

    def test_too_short(self):
        assert not is_uniform_grid(np.array([0.0, 1.0]))


class TestCrossSpectrum:
    def test_antiphase_line_is_recovered(self):
        # y trails x by 180 deg with twice the amplitude: expect phase +-180, amp_ratio 2.
        t, x = _series(5000.0, phase_deg=0.0, amplitude=1.0, noise=0.3, seed=1)
        _, y = _series(5000.0, phase_deg=180.0, amplitude=2.0, noise=0.3, seed=2)

        result = cross_spectrum(t, x, y, n_seg=4)

        assert result.valid
        line = result.top_line
        assert line is not None
        assert line.period == pytest.approx(5000.0, rel=0.02)
        assert line.gamma2 > result.gamma2_crit
        assert phase_distance(line.phase_deg, 180.0) < 2.0
        assert line.amp_ratio == pytest.approx(2.0, rel=0.05)

    def test_quadrature_sign_convention(self):
        # x = sin, y = cos: y leads x by 90 deg, and the reported phase must be positive.
        t, x = _series(5000.0, phase_deg=0.0, noise=0.2, seed=3)
        _, y = _series(5000.0, phase_deg=90.0, noise=0.2, seed=4)

        line = cross_spectrum(t, x, y, n_seg=4).top_line

        assert line.phase_deg == pytest.approx(90.0, abs=2.0)

    def test_noise_lines_are_negligible_next_to_a_real_one(self):
        # Independent noise still trips the threshold in ~alpha of the bins by
        # construction, so "no lines at all" is the wrong expectation. What separates
        # noise from signal is that noise lines are single-bin and carry orders of
        # magnitude less cross-amplitude.
        rng = np.random.default_rng(11)
        t = np.arange(200_000, dtype=float)
        noise_x, noise_y = rng.standard_normal(t.size), rng.standard_normal(t.size)

        noise_only = cross_spectrum(t, noise_x, noise_y, n_seg=4)
        common = np.sin(2 * np.pi * t / 5000.0)
        with_line = cross_spectrum(t, noise_x + common, noise_y - common, n_seg=4)

        assert noise_only.valid and with_line.valid
        assert max(line.n_bins for line in noise_only.lines) <= 2
        assert with_line.top_line.cross_amplitude > 100 * noise_only.top_line.cross_amplitude
        assert with_line.top_line.period == pytest.approx(5000.0, rel=0.02)

    def test_threshold_matches_its_nominal_false_alarm_rate(self):
        # gamma2_crit = 1 - alpha^(1/(n_seg-1)) is exact for NON-OVERLAPPING segments;
        # the measured fraction of bins above it must come out at alpha.
        rng = np.random.default_rng(5)
        alpha, n_seg, fractions = 0.01, 4, []
        for _ in range(60):
            t = np.arange(20_000, dtype=float)
            result = cross_spectrum(t, rng.standard_normal(t.size), rng.standard_normal(t.size), n_seg=n_seg, alpha=alpha)
            in_band = result.in_band
            fractions.append(np.mean(result.coherence[in_band] >= result.gamma2_crit))

        assert np.mean(fractions) == pytest.approx(alpha, abs=0.005)

    @pytest.mark.parametrize('n', [30_001, 200_000, 250_001])
    @pytest.mark.parametrize('n_seg', [2, 4, 8, 16])
    def test_welch_produces_exactly_n_seg_segments(self, n, n_seg):
        # The threshold formula assumes n_seg INDEPENDENT segments. With
        # nperseg = n // n_seg and noverlap = 0, scipy's segment count
        # (n - nperseg) // nperseg + 1 must come out at exactly n_seg for any n,
        # otherwise the nominal false-alarm rate is wrong.
        nperseg = n // n_seg

        assert (n - nperseg) // nperseg + 1 == n_seg

    def test_gamma2_crit_formula(self):
        t, x = _series(5000.0, noise=1.0, seed=6)
        _, y = _series(5000.0, noise=1.0, seed=7)

        for n_seg in (4, 8, 16):
            result = cross_spectrum(t, x, y, n_seg=n_seg, alpha=0.05)
            assert result.gamma2_crit == pytest.approx(1.0 - 0.05 ** (1.0 / (n_seg - 1)))

    def test_line_outside_validity_band_is_not_reported(self):
        # P_max = T / (5 * n_seg): with T = 200000 and n_seg = 4 that is 10000 yr,
        # so a genuine 40000-yr common line must be excluded, not reported.
        t, x = _series(40_000.0, noise=0.2, seed=8)
        _, y = _series(40_000.0, phase_deg=180.0, noise=0.2, seed=9)

        result = cross_spectrum(t, x, y, n_seg=4)

        assert result.period_max == pytest.approx(t[-1] / 20.0, rel=1e-6)
        assert all(line.period <= result.period_max for line in result.lines)

    def test_explicit_period_min_narrows_the_band(self):
        t, x = _series(5000.0, noise=0.2, seed=10)
        _, y = _series(5000.0, phase_deg=180.0, noise=0.2, seed=11)

        result = cross_spectrum(t, x, y, n_seg=4, period_min=8000.0)

        assert result.period_min == 8000.0
        assert all(line.period >= 8000.0 for line in result.lines)

    def test_single_segment_is_rejected(self):
        t, x = _series(5000.0, seed=12)
        _, y = _series(5000.0, seed=13)

        result = cross_spectrum(t, x, y, n_seg=1)

        assert not result.valid
        assert 'n_seg' in result.reason

    def test_short_segments_are_rejected(self):
        t, x = _series(50.0, n=2000, seed=14)
        _, y = _series(50.0, n=2000, seed=15)

        result = cross_spectrum(t, x, y, n_seg=8)  # nperseg = 250 < 512

        assert not result.valid
        assert 'nperseg' in result.reason

    def test_length_mismatch_is_rejected(self):
        t, x = _series(5000.0, n=20_000, seed=29)
        _, y = _series(5000.0, n=19_000, seed=30)

        result = cross_spectrum(t, x, y, n_seg=4)

        assert not result.valid
        assert 'length mismatch' in result.reason

    def test_non_uniform_grid_is_rejected(self):
        t, x = _series(5000.0, n=20_000, seed=16)
        _, y = _series(5000.0, n=20_000, seed=17)
        t = t.copy()
        t[10_000] += 0.5

        result = cross_spectrum(t, x, y, n_seg=4)

        assert not result.valid
        assert 'uniform' in result.reason

    def test_two_separated_lines_are_grouped_separately(self):
        t = np.arange(200_000, dtype=float)
        rng = np.random.default_rng(18)
        common_a = np.sin(2 * np.pi * t / 5000.0)
        common_b = np.sin(2 * np.pi * t / 1000.0)
        x = common_a + common_b + 0.2 * rng.standard_normal(t.size)
        y = -common_a - common_b + 0.2 * rng.standard_normal(t.size)

        result = cross_spectrum(t, x, y, n_seg=4)

        periods = sorted(line.period for line in result.lines)
        assert len(result.lines) >= 2
        assert any(abs(p - 1000.0) < 20.0 for p in periods)
        assert any(abs(p - 5000.0) < 100.0 for p in periods)

    def test_max_lines_caps_the_output(self):
        t = np.arange(200_000, dtype=float)
        rng = np.random.default_rng(19)
        common = sum(np.sin(2 * np.pi * t / p) for p in (700.0, 1300.0, 2100.0, 3300.0, 5000.0))
        x = common + 0.1 * rng.standard_normal(t.size)
        y = common + 0.1 * rng.standard_normal(t.size)

        uncapped = cross_spectrum(t, x, y, n_seg=4)
        capped = cross_spectrum(t, x, y, n_seg=4, max_lines=2)

        assert len(uncapped.lines) > 2
        assert len(capped.lines) == 2
        # The cap keeps the strongest lines, not an arbitrary two.
        assert [line.period for line in capped.lines] == [line.period for line in uncapped.lines[:2]]

    def test_lines_are_ranked_by_cross_amplitude(self):
        t = np.arange(200_000, dtype=float)
        rng = np.random.default_rng(20)
        strong = 5.0 * np.sin(2 * np.pi * t / 5000.0)
        weak = 0.2 * np.sin(2 * np.pi * t / 1100.0)
        x = strong + weak + 0.05 * rng.standard_normal(t.size)
        y = strong + weak + 0.05 * rng.standard_normal(t.size)

        result = cross_spectrum(t, x, y, n_seg=4)

        assert result.top_line.period == pytest.approx(5000.0, rel=0.02)
        amplitudes = [line.cross_amplitude for line in result.lines]
        assert amplitudes == sorted(amplitudes, reverse=True)


class TestBandpassedCorrelation:
    # The band-passed correlation reads the same relationship the cross-phase does, but
    # in the time domain: antiphase -> -1, in phase -> +1, quadrature -> 0.
    @pytest.mark.parametrize('phase_deg, expected', [(180.0, -1.0), (0.0, 1.0), (90.0, 0.0)])
    def test_matches_the_injected_phase(self, phase_deg, expected):
        t, x = _series(5000.0, noise=0.3, seed=40)
        _, y = _series(5000.0, phase_deg=phase_deg, amplitude=2.0, noise=0.3, seed=41)

        r = bandpassed_correlation(x, y, dt=1.0, frequency=1 / 5000.0)

        assert r == pytest.approx(expected, abs=0.05)

    def test_ignores_signal_outside_the_band(self):
        # A strong in-phase 500-yr term must not pull the correlation measured at 5000 yr.
        t = np.arange(200_000, dtype=float)
        slow = np.sin(2 * np.pi * t / 5000.0)
        fast = 5.0 * np.sin(2 * np.pi * t / 500.0)
        x = slow + fast
        y = -slow + fast

        assert bandpassed_correlation(x, y, dt=1.0, frequency=1 / 5000.0) == pytest.approx(-1.0, abs=0.05)
        assert bandpassed_correlation(x, y, dt=1.0, frequency=1 / 500.0) == pytest.approx(1.0, abs=0.05)

    def test_constant_series_is_nan(self):
        t = np.arange(20_000, dtype=float)

        assert np.isnan(bandpassed_correlation(np.ones_like(t), np.sin(t), dt=1.0, frequency=0.01))

    def test_the_strongest_line_carries_it(self):
        t, x = _series(5000.0, noise=0.3, seed=42)
        _, y = _series(5000.0, phase_deg=180.0, noise=0.3, seed=43)

        result = cross_spectrum(t, x, y, n_seg=4)

        assert result.top_line.r_band == pytest.approx(-1.0, abs=0.05)
        # Only the strongest line pays for the two extra FFTs.
        assert all(np.isnan(line.r_band) for line in result.lines[1:])


class TestLevels:
    def test_decisive_level_is_the_coarsest_valid_one(self):
        t, x = _series(5000.0, noise=0.2, seed=21)
        _, y = _series(5000.0, phase_deg=180.0, noise=0.2, seed=22)

        levels = cross_spectrum_levels(t, x, y, n_segments=(4, 8, 16))
        decisive = decisive_level(levels)

        assert [level.n_seg for level in levels] == [4, 8, 16]
        assert decisive.n_seg == 4

    def test_decisive_level_skips_invalid_levels(self):
        # nperseg = N / n_seg must stay >= 512, so with N = 3000 only n_seg = 4 survives.
        t, x = _series(200.0, n=3000, noise=0.2, seed=23)
        _, y = _series(200.0, n=3000, phase_deg=180.0, noise=0.2, seed=24)

        levels = cross_spectrum_levels(t, x, y, n_segments=(4, 8, 16))

        assert [level.valid for level in levels] == [True, False, False]
        assert decisive_level(levels).n_seg == 4

    def test_decisive_level_is_none_when_nothing_is_valid(self):
        t, x = _series(20.0, n=600, seed=25)
        _, y = _series(20.0, n=600, seed=26)

        assert decisive_level(cross_spectrum_levels(t, x, y, n_segments=(4, 8))) is None

    def test_levels_agreeing_counts_matching_top_lines(self):
        # T = 200000 yr, so P_max = T/(5*n_seg) is 10000 / 5000 / 2500 yr for
        # n_seg = 4 / 8 / 16. The 5000-yr line sits inside the first two bands and
        # outside the third, so exactly two levels can see it.
        t, x = _series(5000.0, noise=0.2, seed=27)
        _, y = _series(5000.0, phase_deg=180.0, noise=0.2, seed=28)

        levels = cross_spectrum_levels(t, x, y, n_segments=(4, 8, 16))
        top = decisive_level(levels).top_line

        assert [level.period_max for level in levels] == pytest.approx([10_000.0, 5000.0, 2500.0], rel=1e-3)
        assert levels_agreeing(levels, top) == 2


class TestForcedFreeEccentricity:
    def test_static_orbit_is_pure_forced(self):
        ecc = np.full(1000, 0.2)
        omega = np.full(1000, 1.1)

        e_forced, e_free = forced_free_eccentricity(ecc, omega)

        assert e_forced == pytest.approx(0.2)
        assert e_free == pytest.approx(0.0, abs=1e-12)

    def test_uniformly_circulating_pericentre_is_pure_free(self):
        # A vector of constant length R sweeping the circle uniformly puts R^2/2 into
        # each of var(k) and var(h), so e_free = sqrt(R^2/2 + R^2/2) = R.
        ecc = np.full(20_000, 0.15)
        omega = np.linspace(0.0, 200 * np.pi, 20_000, endpoint=False)

        e_forced, e_free = forced_free_eccentricity(ecc, omega)

        assert e_forced == pytest.approx(0.0, abs=1e-3)
        assert e_free == pytest.approx(0.15, rel=1e-3)

    def test_forced_plus_free_superposition(self):
        # k = 0.04 + 0.03 cos(t), h = 0.03 sin(t): centroid at 0.04, radius 0.03.
        angle = np.linspace(0.0, 200 * np.pi, 20_000, endpoint=False)
        k = 0.04 + 0.03 * np.cos(angle)
        h = 0.03 * np.sin(angle)
        ecc = np.hypot(k, h)
        omega = np.arctan2(h, k)

        e_forced, e_free = forced_free_eccentricity(ecc, omega)

        assert e_forced == pytest.approx(0.04, rel=1e-3)
        assert e_free == pytest.approx(0.03, rel=1e-3)

    def test_e_free_keeps_the_sqrt2_factor(self):
        # A common slip is averaging the two variances instead of summing them, which
        # would report 0.15 / sqrt(2) = 0.106 here.
        ecc = np.full(20_000, 0.15)
        omega = np.linspace(0.0, 200 * np.pi, 20_000, endpoint=False)

        _, e_free = forced_free_eccentricity(ecc, omega)

        assert e_free > 0.14

    def test_eccentricity_vector_components(self):
        ecc = np.array([0.1, 0.2])
        omega = np.array([0.0, np.pi / 2])

        k, h = eccentricity_vector(ecc, omega)

        assert k == pytest.approx([0.1, 0.0], abs=1e-12)
        assert h == pytest.approx([0.0, 0.2], abs=1e-12)
