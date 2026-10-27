"""Backward integrations (tmax < 0, decreasing times) must give the numbers of a forward run.

Each case builds a signal whose answer is known by construction (a libration period, a free rate,
a mode phase, a cross-phase), samples it once on a forward grid t in [0, T] and once on the backward
grid of `Simulation.run` (np.linspace(0, -T, N), so decreasing), and checks that both directions give
that answer. Before the fix the backward runs lost the period, collapsed the free-element basis,
invalidated every cross spectrum and drew empty evolution figures.
"""

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from resonances.body import Body  # noqa: E402
from resonances.data.const import PLANETARY_FREQUENCIES as F  # noqa: E402
from resonances.data.util import time_ordered  # noqa: E402
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance  # noqa: E402
from resonances.plotting import Plotter  # noqa: E402
from resonances.plotting.base import time_unit  # noqa: E402
from resonances.resonance import cross_spectrum as cs  # noqa: E402
from resonances.resonance.analysis.secular import LongPeriodSeries  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import AngleStatus, Reason, ResonanceType, Status  # noqa: E402
from resonances.resonance.analysis.zlk import analyse_zlk  # noqa: E402
from resonances.resonance.classify import check_chaos  # noqa: E402
from resonances.resonance.omega_free_gate import GateOutcome, evaluate_gate  # noqa: E402
from resonances.simulation.simulation import long_period_series  # noqa: E402
from resonances.secular import planet_modes as pm  # noqa: E402
from resonances.secular.free_elements import free_elements  # noqa: E402
from resonances.secular.proper_angle import removed_modes  # noqa: E402
from resonances.secular.secular_resonance import SecularResonance  # noqa: E402

RAD_PER_ARCSEC = np.pi / 180.0 / 3600.0


def grids(span, n):
    """The forward grid and the backward one `Simulation.run` builds, in years."""
    return np.linspace(0.0, span, n), np.linspace(0.0, -span, n)


def body_with_angle(times_years, angle, resonance):
    body = Body()
    body.name = 'synthetic'
    body.times = times_years * 2 * np.pi
    body.axis = np.full_like(times_years, 2.5)
    body.ecc = np.full_like(times_years, 0.1)
    body.inc = np.full_like(times_years, 0.1)
    body.angles_filtered[resonance.to_s()] = np.mod(angle, 2 * np.pi)
    body.angles_filtered_unwrapped[resonance.to_s()] = angle
    if resonance.type == 'mmr':
        body.mmrs.append(resonance)
    else:
        body.lidov_kozai_resonances.append(resonance)
    return body


class TestTimeOrdered:
    def test_a_backward_grid_is_reversed_with_its_series(self):
        times, x, nothing = time_ordered(np.array([0.0, -1.0, -2.0]), np.array([10.0, 11.0, 12.0]), None)
        assert list(times) == [-2.0, -1.0, 0.0] and list(x) == [12.0, 11.0, 10.0] and nothing is None

    def test_a_forward_grid_comes_back_as_it_is(self):
        times, x = np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0])
        ordered_times, ordered_x = time_ordered(times, x)
        assert ordered_times is times and ordered_x is x


NO_GATE = SimpleNamespace(free_gate_enabled=False, free_gate_min_cycles=3.0)  # the geometry alone: no free elements here


def zlk_verdict(times, angle):
    """The ZLK analysis of a synthetic omega, the series built by the Simulation adapter (time order, years)."""
    resonance = LidovKozaiResonance(identifier='LK')
    body = body_with_angle(times, angle, resonance)
    body.angles_unwrapped[resonance.to_s()] = angle
    body.varpi, body.Omega = np.zeros_like(times), np.zeros_like(times)
    return analyse_zlk(long_period_series(body, resonance), settings_from_config(), gate_config=NO_GATE)


def bare_series(times, angle):
    n = len(times)
    return LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', times, angle, np.full(n, 2.5), np.full(n, 0.1), np.full(n, 0.1))


class TestAnalysis:
    def test_a_ten_kyr_libration_has_the_same_period_both_ways(self):
        # sigma = pi + 0.5 sin(2 pi t / 10 kyr) over 100 kyr at the default 6283 outputs: P = 10 000 yr
        for times in grids(1e5, 6283):
            analysis = zlk_verdict(times, np.pi + 0.5 * np.sin(2 * np.pi * times / 1e4))
            assert analysis.verdict.status == Status.RESONANT
            assert analysis.verdict.angle_status == AngleStatus.LIBRATION
            assert analysis.libration_period_yr == pytest.approx(1e4, rel=0.01)

    def test_a_capture_is_analysed_alike_in_both_directions(self):
        # Circulation at 3 turns per 10 kyr until 40 kyr, then a 5-kyr libration: libration episodes that do not cover
        # the record. The backward series, read in time order, is the forward one shifted by -T.
        T, n = 1e5, 6283
        forward, backward = grids(T, n)

        def angle(elapsed):
            return 2 * np.pi * 3e-4 * np.minimum(elapsed, 4e4) + 0.5 * np.sin(2 * np.pi * elapsed / 5e3)

        f = zlk_verdict(forward, angle(forward))
        b = zlk_verdict(backward, angle(backward + T))
        assert f.verdict.angle_status == b.verdict.angle_status == AngleStatus.EPISODIC
        assert f.verdict.status == b.verdict.status == Status.TRANSIENT
        assert len(f.geometry.turning_points) == len(b.geometry.turning_points)
        assert f.libration_period_yr == pytest.approx(b.libration_period_yr, rel=1e-9)
        assert f.geometry.episodes[0].t_start_yr == pytest.approx(b.geometry.episodes[0].t_start_yr + T, abs=1e-6)


class TestShortAndNonFiniteSeries:
    @pytest.mark.parametrize('n', [12, 15, 19])
    def test_short_series_get_a_verdict_without_hanging(self, n):
        # A V: three turns up and back down over 1000 yr, shorter than the filter can take
        times = np.linspace(0.0, 1000.0, n)
        angle = 6 * np.pi * (1 - np.abs(2 * times / 1000.0 - 1))
        verdict = analyse_zlk(bare_series(times, angle), settings_from_config(), gate_config=NO_GATE).verdict
        assert verdict.status in set(Status)

    def test_too_few_samples_are_uncertain(self):
        verdict = analyse_zlk(bare_series(np.array([0.0, 1.0]), np.array([0.0, 0.1])), settings_from_config()).verdict
        assert verdict.status == Status.UNCERTAIN and verdict.reason == Reason.UNUSABLE_SERIES

    def test_nan_in_the_angle_is_uncertain(self):
        times = np.linspace(0.0, 1e5, 6283)
        angle = np.pi + 0.5 * np.sin(2 * np.pi * times / 5000)
        angle[3000:] = np.nan
        verdict = analyse_zlk(bare_series(times, angle), settings_from_config()).verdict
        assert verdict.status == Status.UNCERTAIN and verdict.reason == Reason.UNUSABLE_SERIES

    def test_a_non_finite_semi_major_axis_is_unphysical(self):
        axis = np.full(100, 2.5)
        axis[60:] = np.nan
        assert check_chaos(SimpleNamespace(axis=axis, ecc=np.full(100, 0.1))) == (1, 'unphysical: non_finite')


class TestSecular:
    @pytest.mark.parametrize(
        'baseline, modes_e, modes_i',
        [
            # 10 Myr: 2/T = 0.26"/yr separates every fundamental; g6 is the angle's own and stays
            (1e7, {'const', 'g5', 'g7', 'g8'}, {'const', 's6', 's7', 's8'}),
            # 2 Myr: 2/T = 1.30"/yr merges const-g8 (0.67 apart, g8 fitted) and g7-g5 (1.17, g5 fitted),
            # and s8-const (0.69, const fitted); s6 and s7 stand alone
            (2e6, {'g5', 'g8'}, {'const', 's6', 's7'}),
        ],
    )
    def test_removed_modes_ignore_the_direction(self, baseline, modes_e, modes_i):
        terms = SecularResonance('g-g6').formula.terms
        for signed in (baseline, -baseline):
            removed_e, removed_i = removed_modes(terms, signed)
            assert (set(removed_e), set(removed_i)) == (modes_e, modes_i)

    def test_free_elements_and_the_gate_agree_both_ways(self):
        # Forced terms at g5, g6, g7, g8 and s6, s7, s8 plus the invariable-plane offset; free e and node both
        # turn at -12"/yr, so omega_free stands still: a genuine Lidov-Kozai lock over 10 Myr at 100 yr.
        def vectors(t):
            def line(amplitude, rate, phase=0.0):
                return amplitude * np.exp(1j * (rate * RAD_PER_ARCSEC * t + phase))

            z_e = line(0.03, F['g5'], 0.4) + line(0.02, F['g6'], 1.1) + line(0.005, F['g7']) + line(0.004, F['g8']) + line(0.2, -12.0, 2.0)
            z_i = 0.0138 * np.exp(1.9j) + line(0.01, F['s6']) + line(0.004, F['s7']) + line(0.003, F['s8']) + line(0.3, -12.0, 0.5)
            ecc, varpi = np.abs(z_e), np.angle(z_e)
            inc, Omega = 2 * np.arcsin(np.abs(z_i)), np.angle(z_i)
            return ecc, inc, Omega, varpi - Omega

        config = SimpleNamespace(free_gate_enabled=True, free_gate_min_cycles=3.0)
        for times in grids(1e7, 100001):
            elements = free_elements(times, *vectors(times))
            assert set(elements.basis_e) == {'const', 'g5', 'g6', 'g7', 'g8'}
            assert set(elements.basis_i) == {'const', 's6', 's7', 's8'}
            assert elements.baseline == pytest.approx(1e7, rel=1e-3)
            assert elements.b_varpi == pytest.approx(-12.0, abs=0.05)
            assert elements.b_Omega == pytest.approx(-12.0, abs=0.05)
            assert elements.librates
            gate = evaluate_gate(elements, 4e4, config)
            assert gate.outcome == GateOutcome.CONFIRMED
            assert gate.n_cycles == pytest.approx(1e7 / 4e4, rel=1e-3)  # 250 cycles of 40 kyr

    def test_measured_mode_longitudes_follow_the_backward_grid(self):
        # Saturn's vectors from g6 at 2.2 rad (next to g5) and s6 at -0.7 rad, sampled over 10 Myr backwards:
        # the measured longitude is g6*t + 2.2 at every (negative) t
        times = np.linspace(0.0, -1e7, 20001)
        phases = {'g6': 2.2, 's6': -0.7}
        z = 0.048 * np.exp(1j * (F['g6'] * RAD_PER_ARCSEC * times + 2.2)) + 0.033 * np.exp(1j * (F['g5'] * RAD_PER_ARCSEC * times + 0.5))
        zeta = 0.008 * np.exp(1j * (F['s6'] * RAD_PER_ARCSEC * times - 0.7)) + 0.0138 * np.exp(1.9j)
        saturn = {'e': np.abs(z), 'varpi': np.angle(z), 'inc': 2 * np.arcsin(np.abs(zeta)), 'Omega': np.angle(zeta)}
        longitudes = pm.measured_longitudes(times, {'Saturn': saturn}, ['g6', 's6'])
        assert set(longitudes) == {'g6', 's6'}
        for label, phase in phases.items():
            error = np.angle(np.exp(1j * (longitudes[label] - F[label] * RAD_PER_ARCSEC * times - phase)))
            assert np.max(np.abs(error)) < np.radians(0.5), label


class TestCrossSpectrum:
    def test_a_backward_pair_keeps_its_period_and_phase(self):
        # y leads x by 60 deg at 2 kyr; on the backward grid the pair is sampled at the same physical times
        rng = np.random.default_rng(0)
        for times in grids(1e5, 6283):
            x = np.sin(2 * np.pi * times / 2000.0) + 0.3 * rng.standard_normal(len(times))
            y = np.sin(2 * np.pi * times / 2000.0 + np.radians(60.0)) + 0.3 * rng.standard_normal(len(times))
            assert cs.is_uniform_grid(times)
            result = cs.cross_spectrum(times, x, y, 4, pair='x-y', psd_cache={})
            assert result.valid
            assert result.top_line.period == pytest.approx(2000.0, rel=0.05)
            assert result.top_line.phase_deg == pytest.approx(60.0, abs=3.0)


class TestPlots:
    def test_time_unit_reads_the_length_of_a_negative_span(self):
        assert time_unit(-1e7) == time_unit(1e7) == (1e6, 'Myr')
        assert time_unit(-2e4) == (1e3, 'kyr')

    def test_the_evolution_figure_of_a_backward_run_shows_its_data(self, tmp_path):
        T, n = 2e4, 1001
        times = np.linspace(0.0, -T, n)
        key = '4J-2S-1+0+0-1'
        angle = np.pi + 0.5 * np.sin(2 * np.pi * times / 4000.0)
        frame = pd.DataFrame({'a': 2.4 + 1e-4 * np.sin(times), 'e': np.full(n, 0.1), 'inc': np.full(n, 0.05), 'times': times})
        frame[f'{key}_angle'] = np.mod(angle, 2 * np.pi)
        frame[f'{key}_angle_filtered_unwrapped'] = angle
        path = tmp_path / 'data-backward.csv'
        frame.to_csv(path)

        plotter = Plotter.from_csv(path, key).configure('full').plot()
        panels = plotter._config.get_enabled_panels()
        drawn = 0
        for ax, panel in zip(plotter._axes, panels):
            if panel.x_column == 'times' and ax.lines:
                assert ax.get_xlim() == pytest.approx((-T, 0.0))
                drawn += 1
        assert drawn >= 3  # a, e and the angle at least
        plotter.close()
