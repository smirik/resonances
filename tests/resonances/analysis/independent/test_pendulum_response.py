"""Independent functional tests of the pendulum, the a-response, kinematics and strength (SPEC §4.5-4.7, §4.9).

Written from the API and the SPEC only. Every expected value comes from a closed form, a synthetic series whose
truth is constructed here (a pendulum integrated with scipy, vectors rotating at known rates), or a hand
calculation shown in a comment.
"""

import math

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, quad, solve_ivp

from resonances.resonance.analysis import pendulum
from resonances.resonance.analysis.mmr import kinematics, response, strength
from resonances.resonance.analysis.mmr.models import ForceWindow, MultipletFit, SlowComponents
from resonances.resonance.analysis.pendulum import CycleClass, CycleEnergy, Potential, Zone, ZoneMethod
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue

SETTINGS = settings_from_config()
OMEGA = 2 * np.pi / 300.0  # rad/yr: a 300-yr small-amplitude libration


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def pendulum_rhs(omega, well):
    return lambda t, y: [y[1], -(omega**2) * np.sin(y[0] - well)]


def libration_track(omega, well, amplitude, t_end, dt):
    """Pendulum released at rest at well + amplitude; returns t, sigma, rate and the times of the minima of sigma."""

    def minimum(t, y):
        return y[1]

    minimum.direction = 1
    t = np.arange(0.0, t_end + dt / 2, dt)
    sol = solve_ivp(
        pendulum_rhs(omega, well),
        (0.0, t[-1]),
        [well + amplitude, 0.0],
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        t_eval=t,
        events=minimum,
    )
    return t, sol.y[0], sol.y[1], sol.t_events[0]


def revolution_track(omega, well, h, t_end, dt):
    """Revolution starting at the bottom with energy h (V = -omega^2 cos(sigma - well), V_max - V_min = 2 omega^2)."""
    rate0 = 2 * omega * np.sqrt(h)  # rate^2 / 2 = 2 omega^2 h at the bottom
    t = np.arange(0.0, t_end + dt / 2, dt)
    events = []
    for m in range(1, 200):

        def turn(tt, y, m=m):
            return y[0] - well - 2 * np.pi * m

        events.append(turn)
    sol = solve_ivp(
        pendulum_rhs(omega, well), (0.0, t[-1]), [well, rate0], method='DOP853', rtol=1e-11, atol=1e-13, t_eval=t, events=events
    )
    turns = np.sort(np.concatenate([e for e in sol.t_events if len(e)]))
    return t, sol.y[0], sol.y[1], np.concatenate([[0.0], turns])


def cosine_potential(omega, well):
    """V = -omega^2 cos(sigma - well) written as a_1 cos sigma + b_1 sin sigma."""
    return Potential(
        n_harmonics=1,
        cos_coeffs=np.array([-(omega**2) * np.cos(well)]),
        sin_coeffs=np.array([-(omega**2) * np.sin(well)]),
        r2=1.0,
        well_rad=well,
        top_rad=float(wrap(well + np.pi)),
        omega_rad_yr=omega,
        v_min=-(omega**2),
        v_max=omega**2,
    )


def action_ratio_by_quadrature(h):
    """J / J_s for V = 1 - cos sigma (Omega = 1): J = 2 int sqrt(2 (E - V)) dsigma, E = 2h, J_s = 16."""
    amplitude = 2 * np.arcsin(np.sqrt(h))
    integral, _ = quad(lambda s: np.sqrt(max(0.0, 2 * (2 * h - (1 - np.cos(s))))), -amplitude, amplitude, limit=200)
    return 2 * integral / 16.0


# ---------------------------------------------------------------- pendulum: closed forms


@pytest.mark.parametrize('amplitude_deg, expected', [(90, 1.180), (120, 1.373), (150, 1.762)])
def test_amplitude_factor_md_table(amplitude_deg, expected):
    # Murray & Dermott 1999, eqs. 8.50-8.52
    assert pendulum.amplitude_factor(np.radians(amplitude_deg)) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize('amplitude_deg', [5, 30, 90, 150, 170])
def test_amplitude_factor_against_integrated_pendulum(amplitude_deg):
    # period of sigma_ddot = -sin sigma (small period 2 pi), measured between successive minima
    amplitude = np.radians(amplitude_deg)
    _, _, _, minima = libration_track(1.0, 0.0, amplitude, 200.0, 0.5)
    period = np.median(np.diff(minima))
    assert pendulum.amplitude_factor(amplitude) == pytest.approx(period / (2 * np.pi), rel=1e-7)


def test_amplitude_factor_small_amplitude_limit():
    # F(A) = 1 + A^2/16 + O(A^4); at A = 0.01 the remainder is ~ 1e-9
    assert pendulum.amplitude_factor(0.01) == pytest.approx(1 + 0.01**2 / 16, abs=1e-9)
    assert pendulum.amplitude_factor(1e-8) == pytest.approx(1.0, abs=1e-12)


def test_small_period_from_integrated_pendulum():
    # truth: Omega = 2 pi / 300 yr, so T0 = 300 yr whatever the amplitude
    amplitude = np.radians(120)
    _, _, _, minima = libration_track(OMEGA, 0.4, amplitude, 3000.0, 0.05)
    period = np.median(np.diff(minima))
    assert pendulum.small_period(period, amplitude) == pytest.approx(300.0, rel=1e-6)


def test_half_width_and_excursion_by_hand_with_negative_coefficient():
    # 2 * 0.01 / 0.5 = 0.04 au; 2 * 0.01 * sin(30 deg) / 0.5 = 0.02 au
    assert pendulum.half_width(0.01, -0.5) == pytest.approx(0.04)
    assert pendulum.half_width(0.01, 0.5) == pytest.approx(0.04)
    assert pendulum.excursion(0.01, np.pi / 3, -0.5) == pytest.approx(0.02)


def test_excursion_matches_the_slow_variable_of_a_simulated_pendulum():
    # sigma_dot = c (x - x*): the half-range of x is max|sigma_dot| / |c|
    amplitude, c = np.radians(100), -0.6
    _, _, rate, _ = libration_track(OMEGA, 0.0, amplitude, 900.0, 0.05)
    x = rate / c
    assert pendulum.excursion(OMEGA, amplitude, c) == pytest.approx(np.ptp(x) / 2, rel=1e-5)


@pytest.mark.parametrize('h', [0.05, 0.5, 0.9, 0.99])
def test_libration_period_against_solve_ivp(h):
    amplitude = 2 * np.arcsin(np.sqrt(h))  # h = sin^2(A/2) at the turning point
    _, _, _, minima = libration_track(OMEGA, 0.0, amplitude, 20 * 300.0 * 4, 0.5)
    period = np.median(np.diff(minima))
    assert pendulum.libration_period(h, OMEGA) == pytest.approx(period, rel=1e-7)


def test_libration_period_small_energy_limit():
    assert pendulum.libration_period(1e-10, OMEGA) == pytest.approx(300.0, rel=1e-8)


@pytest.mark.parametrize('h', [1.05, 1.5, 3.0])
def test_revolution_period_against_solve_ivp(h):
    _, _, _, turns = revolution_track(OMEGA, 0.3, h, 4000.0, 1.0)
    period = np.median(np.diff(turns))
    assert pendulum.revolution_period(h, OMEGA) == pytest.approx(period, rel=1e-7)


@pytest.mark.parametrize('h', [0.1, 0.3, 0.6, 0.9])
def test_action_ratio_against_quadrature(h):
    assert pendulum.action_ratio(h) == pytest.approx(action_ratio_by_quadrature(h), rel=1e-6)


def test_action_ratio_limits():
    # J/J_s -> pi h / 4 as h -> 0 and -> 1 at h = 1
    assert pendulum.action_ratio(1e-6) == pytest.approx(np.pi * 1e-6 / 4, rel=1e-5)
    assert pendulum.action_ratio(1.0) == pytest.approx(1.0, abs=1e-9)
    assert pendulum.action_ratio(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize('h', [0.01, 0.2, 0.5, 0.8, 0.999])
def test_h_from_action_ratio_inverts_the_quadrature(h):
    assert pendulum.h_from_action_ratio(action_ratio_by_quadrature(h)) == pytest.approx(h, rel=1e-5)


def test_h_from_action_ratio_edges():
    assert pendulum.h_from_action_ratio(0.0) == pytest.approx(0.0, abs=1e-8)
    assert pendulum.h_from_action_ratio(1.0) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize('h, expected', [(1.012, 0.25), (1.14, 0.554)])
def test_rate_min_over_mean_reference_values(h, expected):
    assert pendulum.rate_min_over_mean(h) == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize('h', [1.14, 1.59, 2.5])
def test_rate_min_over_mean_and_mean_rate_against_a_simulated_revolution(h):
    # min rate at the top: rate^2/2 = 2 Omega^2 (h - 1); mean rate = 2 pi / T_rev, T_rev from events
    _, _, _, turns = revolution_track(OMEGA, 0.0, h, 4000.0, 1.0)
    period = np.median(np.diff(turns))
    mean_rate = 2 * np.pi / period
    min_rate = 2 * OMEGA * np.sqrt(h - 1)
    assert pendulum.rate_min_over_mean(h) == pytest.approx(min_rate / mean_rate, rel=1e-6)


# ---------------------------------------------------------------- pendulum: Potential


def test_potential_value_two_harmonics_by_hand():
    # V(pi/3) = 1 cos(pi/3) + 2 cos(2pi/3) + 0.5 sin(pi/3) = 0.5 - 1 + 0.4330127 = -0.0669873
    pot = Potential(2, np.array([1.0, 2.0]), np.array([0.5, 0.0]), 1.0, 0.0, np.pi, 1.0, -3.0, 3.0)
    assert pot.value(np.array([np.pi / 3]))[0] == pytest.approx(-0.0669873, abs=1e-7)


def test_energy_h_is_sin2_half_amplitude_and_constant_along_a_libration():
    well, amplitude = 0.7, 1.2
    _, sigma, rate, _ = libration_track(OMEGA, well, amplitude, 1500.0, 0.5)
    h = cosine_potential(OMEGA, well).energy_h(sigma, rate)
    assert np.allclose(h, np.sin(amplitude / 2) ** 2, rtol=1e-7)


def test_energy_h_of_a_wrapped_revolution():
    # wrap-around at +-pi must not change the energy of a revolution
    _, sigma, rate, _ = revolution_track(OMEGA, 0.3, 1.5, 1500.0, 0.5)
    h = cosine_potential(OMEGA, 0.3).energy_h(wrap(sigma), rate)
    assert np.allclose(h, 1.5, rtol=1e-7)


def test_fit_potential_from_rate_recovers_a_shifted_pendulum():
    well, amplitude = 0.7, 1.2
    t, sigma, rate, minima = libration_track(OMEGA, well, amplitude, 3000.0, 0.5)
    windows = list(zip(minima[:-1], minima[1:]))
    total, per_window = pendulum.fit_potential_from_rate(t, sigma, rate, windows, SETTINGS)
    assert total is not None
    assert len(per_window) == len(windows)
    assert total.n_harmonics == 1
    assert total.r2 > 0.999
    assert total.omega_rad_yr == pytest.approx(OMEGA, rel=1e-3)
    assert float(wrap(total.well_rad - well)) == pytest.approx(0.0, abs=1e-3)
    assert float(wrap(total.top_rad - well - np.pi)) == pytest.approx(0.0, abs=1e-3)
    assert total.v_max - total.v_min == pytest.approx(2 * OMEGA**2, rel=1e-3)
    assert pendulum.well_coherence(per_window) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('c', [0.6, -0.6])
def test_fit_potential_from_force_recovers_the_pendulum(c):
    # sigma_ddot = c x_dot = -Omega^2 sin(sigma - s)  ->  V = -Omega^2 cos(sigma - s)
    well, amplitude = -2.5, 1.0
    _, sigma, _, _ = libration_track(OMEGA, well, amplitude, 3000.0, 0.5)
    x_dot = -(OMEGA**2) * np.sin(sigma - well) / c
    pot = pendulum.fit_potential_from_force(sigma, x_dot, c, 1)
    assert pot is not None
    assert pot.omega_rad_yr == pytest.approx(OMEGA, rel=1e-6)
    assert float(wrap(pot.well_rad - well)) == pytest.approx(0.0, abs=1e-6)
    assert pot.r2 > 0.999


def test_well_coherence_opposite_wells_is_zero():
    pots = [cosine_potential(OMEGA, 0.0), cosine_potential(OMEGA, np.pi)]
    assert pendulum.well_coherence(pots) == pytest.approx(0.0, abs=1e-9)
    # wells at +-(pi - 0.05) straddle the wrap and are nearly the same point: |<e^{i s}>| = cos(0.05)
    pots = [cosine_potential(OMEGA, np.pi - 0.05), cosine_potential(OMEGA, -np.pi + 0.05)]
    assert pendulum.well_coherence(pots) == pytest.approx(np.cos(0.05), abs=1e-9)


# ---------------------------------------------------------------- pendulum: cycles


def test_cycle_energies_libration_matches_constructed_truth():
    well, amplitude = 0.7, 1.2
    t, sigma, rate, minima = libration_track(OMEGA, well, amplitude, 3000.0, 0.5)
    bounds = [(a, b, False) for a, b in zip(minima[:-1], minima[1:])]
    cycles = pendulum.cycle_energies(t, sigma, rate, bounds, cosine_potential(OMEGA, well))
    h_true = np.sin(amplitude / 2) ** 2
    assert len(cycles) == len(bounds)
    for cyc in cycles:
        assert not cyc.rotation
        assert cyc.h == pytest.approx(h_true, rel=1e-6)
        assert cyc.action_ratio == pytest.approx(action_ratio_by_quadrature(h_true), rel=1e-5)
        assert cyc.period_law_ratio == pytest.approx(1.0, abs=1e-6)


def test_cycle_energies_exactly_three_turning_points_give_one_cycle():
    # max (t = 0) - min - max: one full cycle
    well, amplitude = 0.0, 0.8
    t, sigma, rate, minima = libration_track(OMEGA, well, amplitude, 400.0, 0.5)
    t_max2 = 2 * minima[0]  # the second maximum: symmetric about the first minimum
    cycles = pendulum.cycle_energies(t, sigma, rate, [(0.0, t_max2, False)], cosine_potential(OMEGA, well))
    assert len(cycles) == 1
    assert cycles[0].h == pytest.approx(np.sin(0.4) ** 2, rel=1e-6)
    assert cycles[0].period_yr == pytest.approx(t_max2, rel=1e-9)


def test_cycle_energies_revolution_on_a_wrapped_angle():
    t, sigma, rate, turns = revolution_track(OMEGA, 0.3, 1.5, 2000.0, 0.5)
    bounds = [(a, b, True) for a, b in zip(turns[:-1], turns[1:])]
    cycles = pendulum.cycle_energies(t, wrap(sigma), rate, bounds, cosine_potential(OMEGA, 0.3))
    assert len(cycles) == len(bounds) >= 3
    for cyc in cycles:
        assert cyc.rotation
        assert cyc.h == pytest.approx(1.5, rel=1e-6)
        assert math.isnan(cyc.action_ratio)
        assert cyc.period_law_ratio == pytest.approx(1.0, abs=1e-6)


def test_energy_scatter_by_hand_and_floor():
    # differences +-0.1, median 0, MAD 0.1: 1.4826 * 0.1 / sqrt(2) = 0.104837
    assert pendulum.energy_scatter([0.5, 0.6, 0.5, 0.6, 0.5], 0.02) == pytest.approx(0.104837, rel=1e-4)
    # constant input: the scatter is zero, the floor applies
    assert pendulum.energy_scatter([0.4] * 6, 0.02) == pytest.approx(0.02)


def test_sigma_only_check():
    # pendula give 0.9-1.1 and coherence 0.95-0.99; kinematics 0.61 and 0.71 (SPEC §4.6)
    assert pendulum.sigma_only_check(1.05, 0.97, SETTINGS) == CheckValue.PASS
    assert pendulum.sigma_only_check(0.61, 0.71, SETTINGS) == CheckValue.FAIL
    assert pendulum.sigma_only_check(1.0, 0.5, SETTINGS) == CheckValue.FAIL
    assert pendulum.sigma_only_check(1.5, 0.99, SETTINGS) == CheckValue.FAIL
    assert pendulum.sigma_only_check(float('nan'), 0.97, SETTINGS) == CheckValue.NOT_MEASURED


def test_with_slow_check():
    assert pendulum.with_slow_check(CheckValue.PASS, [0.95, 0.97, 0.99, 0.5], SETTINGS) == CheckValue.PASS  # 3/4
    assert pendulum.with_slow_check(CheckValue.PASS, [0.95, 0.5, 0.2], SETTINGS) == CheckValue.FAIL  # 1/3
    assert pendulum.with_slow_check(CheckValue.FAIL, [0.99, 0.99, 0.99], SETTINGS) != CheckValue.PASS


def make_cycle(t0, t1, h, j, rotation=False):
    return CycleEnergy(t0, t1, rotation, h, j, t1 - t0, 1.0)


def test_classify_cycles_far_from_the_edges():
    zone = Zone(ZoneMethod.NUMERIC, j_lo=0.5, h_lo=0.45, h_hi=1.2, adiabatic_lo=0.8, adiabatic_hi=1.1, mismatch=False)
    cycles = [
        make_cycle(0, 1, 0.12, 0.1),  # libration far below J_lo
        make_cycle(1, 2, 0.75, 0.7),  # libration above J_lo: inside the zone
        make_cycle(2, 3, 1.1, float('nan'), rotation=True),  # revolution below h_hi
        make_cycle(3, 4, 1.9, float('nan'), rotation=True),  # revolution far above h_hi
    ]
    classes = pendulum.classify_cycles(cycles, zone, 0.02, SETTINGS)
    assert list(classes) == [CycleClass.DEEP, CycleClass.ZONE, CycleClass.ZONE, CycleClass.ABOVE]


def test_crossings_between_cycles_and_slips():
    # h: 0.8, 0.9 | 1.1, 1.2 | 0.9: two sign changes of h - 1, plus one slip
    hs = [0.8, 0.9, 1.1, 1.2, 0.9]
    cycles = [make_cycle(100.0 * i, 100.0 * (i + 1), h, 0.5, rotation=h > 1) for i, h in enumerate(hs)]
    times = sorted(pendulum.crossings(cycles, [450.0]))
    assert len(times) == 3
    assert 100.0 <= times[0] <= 300.0  # between cycle 2 and cycle 3
    assert 300.0 <= times[1] <= 500.0  # between cycle 4 and cycle 5 (or the slip)
    assert 300.0 <= times[2] <= 500.0
    assert any(abs(x - 450.0) < 1e-9 for x in times)


def test_action_trend_exact_and_noisy():
    mids = np.arange(10) * 300.0 + 150.0
    exact = [make_cycle(m - 150, m + 150, 0.5, 0.3 + 2e-5 * m) for m in mids]
    slope, err = pendulum.action_trend(exact)
    assert slope == pytest.approx(2e-5, rel=1e-9)
    assert err == pytest.approx(0.0, abs=1e-10)
    # OLS standard error by hand: sqrt(sum r^2 / (n - 2) / sum (x - xbar)^2)
    noise = np.array([0.01, -0.02, 0.015, 0.0, -0.01, 0.02, -0.015, 0.005, 0.0, -0.005])
    j = 0.3 + 2e-5 * mids + noise
    slope, err = pendulum.action_trend([make_cycle(m - 150, m + 150, 0.5, v) for m, v in zip(mids, j)])
    b, a0 = np.polyfit(mids, j, 1)
    resid = j - (a0 + b * mids)
    se = np.sqrt(np.sum(resid**2) / (len(mids) - 2) / np.sum((mids - mids.mean()) ** 2))
    assert slope == pytest.approx(b, rel=1e-9)
    assert err == pytest.approx(se, rel=1e-6)


# ---------------------------------------------------------------- response: identity and force law

HILDA_A = 3.97
HILDA_N = 2 * np.pi / 7.9  # rad/yr
HILDA_K = -2  # coefficient of the body lambda in 3:2
HILDA_C = -1.5 * HILDA_K * HILDA_N / HILDA_A  # +0.6 rad/yr/au


@pytest.mark.parametrize('c', [0.6, -0.8])
def test_identity_passes_for_the_identity(c):
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 5000.0, 1.0)
    a = 3.97 + 0.01 * np.sin(2 * np.pi * t / 250.0)
    rate = c * (a - 3.97) + 1e-6 * rng.standard_normal(t.size)
    ratio, corr, check = response.identity(rate, a, c, SETTINGS)
    assert ratio == pytest.approx(1.0, abs=1e-3)
    assert corr > 0.999
    assert check == CheckValue.PASS


def test_identity_fails_when_slope_is_half_or_rate_does_not_follow_a():
    t = np.arange(0.0, 5000.0, 1.0)
    a = 3.97 + 0.01 * np.sin(2 * np.pi * t / 250.0)
    ratio, _, check = response.identity(0.3 * (a - 3.97), a, 0.6, SETTINGS)
    assert ratio == pytest.approx(0.5, abs=1e-6)
    assert check == CheckValue.FAIL
    # the rate carries a wave a does not have (varpi part): corr = 1/sqrt(1 + 4^2) ~ 0.24
    rate = 0.6 * (a - 3.97) + 4 * 0.006 * np.sin(2 * np.pi * t / 97.0)
    _, corr, check = response.identity(rate, a, 0.6, SETTINGS)
    assert corr == pytest.approx(1 / np.sqrt(17), abs=0.03)
    assert check == CheckValue.FAIL


def test_identity_constant_a_is_not_a_pass():
    t = np.arange(0.0, 500.0, 1.0)
    a = np.full(t.size, 3.97)
    rate = 0.01 * np.sin(t / 30.0)
    _, _, check = response.identity(rate, a, 0.6, SETTINGS)
    assert check != CheckValue.PASS


@pytest.mark.parametrize('k', [-2, 3])
def test_identity_wave_of_a_constructed_angle(k):
    # angle = centre + k int (n(a) - <n>) dt: the predicted wave is the angle itself
    t = np.arange(0.0, 6000.0, 1.0)
    a = HILDA_A + 0.01 * np.sin(2 * np.pi * t / 250.0 + 0.4)
    n = 2 * np.pi * a**-1.5
    wave = k * cumulative_trapezoid(n - n.mean(), t, initial=0.0)
    centre = 0.5 + 1e-4 * t
    corr, amp = response.identity_wave(t, a, k, centre + wave, centre)
    assert corr > 0.98
    assert amp == pytest.approx(1.0, abs=0.05)
    # an angle that moves against the a wave: correlation ~ -1
    corr, _ = response.identity_wave(t, a, k, centre - wave, centre)
    assert corr < -0.98


def libration_and_force(phi0, amplitude_au_yr=2e-5, offset=3e-7, t_end=5000.0):
    t = np.arange(0.0, t_end, 0.5)
    sigma = 0.3 + 1.5 * np.sin(2 * np.pi * t / 300.0)
    adot = amplitude_au_yr * np.sin(sigma - phi0) + offset
    return t, sigma, adot


@pytest.mark.parametrize('phi0', [0.0, 3.0, -3.0, np.pi])
def test_force_windows_recover_amplitude_phase_offset(phi0):
    t, sigma, adot = libration_and_force(phi0)
    windows = response.force_windows(t, sigma, adot, 1050.0, 0.5)
    assert len(windows) >= 2
    for w in windows:
        assert w.t_end_yr - w.t_start_yr == pytest.approx(1050.0, abs=1.0)
        assert w.r2 > 0.999
        assert w.amplitude_au_yr == pytest.approx(2e-5, rel=1e-6)
        assert float(wrap(w.phi0_rad - phi0)) == pytest.approx(0.0, abs=1e-6)
        assert w.offset_au_yr == pytest.approx(3e-7, rel=1e-4)


def test_force_windows_negative_force_is_phase_pi():
    # a_dot = -K sin sigma = K sin(sigma - pi)
    t = np.arange(0.0, 3000.0, 0.5)
    sigma = 1.5 * np.sin(2 * np.pi * t / 300.0)
    windows = response.force_windows(t, sigma, -1e-5 * np.sin(sigma), 1050.0, 0.5)
    for w in windows:
        assert w.amplitude_au_yr == pytest.approx(1e-5, rel=1e-6)
        assert abs(float(wrap(w.phi0_rad - np.pi))) < 1e-6


def test_force_windows_series_shorter_than_one_window():
    t, sigma, adot = libration_and_force(0.0, t_end=500.0)
    windows = response.force_windows(t, sigma, adot, 1050.0, 0.5)
    assert len(windows) == 0
    check, _, _ = response.a_response_check(windows, None, np.radians(90), SETTINGS)
    assert check == CheckValue.NOT_MEASURED


def test_multiplet_fit_recovers_constructed_coefficients():
    t = np.arange(0.0, 10000.0, 1.0)
    sigma = 0.2 + 1.4 * np.sin(2 * np.pi * t / 300.0)
    offset = 2 * np.pi * t / 2000.0 + 0.5  # varpi - varpi_P circulating in 2 kyr
    truth = {0: 2e-5 + 0.5e-5j, 1: 0.6e-5 - 0.2e-5j, -1: -0.3e-5 + 0.4e-5j}
    adot = 1e-7 + sum(c.real * np.sin(sigma + j * offset) + c.imag * np.cos(sigma + j * offset) for j, c in truth.items())
    windows = [(0.0, 2500.0), (2500.0, 5000.0), (5000.0, 7500.0), (7500.0, 9999.0)]
    harmonics = (0, 1, -1)
    fit = response.multiplet_fit(t, sigma, adot, offset, harmonics, windows, 0.8)
    assert fit.r2 > 0.999
    assert fit.coefficient_coherence == pytest.approx(1.0, abs=1e-6)
    coeffs = np.asarray(fit.coefficients)
    assert coeffs.shape == (len(windows), len(harmonics))
    for col, j in enumerate(harmonics):
        assert np.allclose(coeffs[:, col], truth[j], rtol=0, atol=1e-10)
    # eps_j = |C_j| / |C_0| for the side terms
    expected = sorted([abs(truth[1]) / abs(truth[0]), abs(truth[-1]) / abs(truth[0])])
    got = sorted(e for e in fit.eps if e < 0.999)
    assert np.allclose(got, expected, rtol=1e-6)


def test_phase_tolerance_by_hand():
    floor = np.radians(30)
    # arcsin(0.3) + 2 * 0.05 = 0.30469 + 0.1 = 0.40469 < 0.5236: the floor
    assert response.phase_tolerance((0.1, 0.2), 0.05, floor) == pytest.approx(floor)
    # arcsin(0.7) + 0.1 = 0.77540 + 0.1 = 0.87540
    assert response.phase_tolerance((0.4, 0.3), 0.05, floor) == pytest.approx(0.875397, abs=1e-6)
    # sum eps > 1: the phase can turn by pi, no bound
    assert np.isnan(response.phase_tolerance((0.8, 0.7), 0.05, floor))
    # no modulating terms: max(floor, 2 sigma)
    assert response.phase_tolerance((), 0.4, floor) == pytest.approx(0.8)


def test_circular_coherence_by_hand():
    assert response.circular_coherence([1.3]) == pytest.approx(1.0)
    assert response.circular_coherence([0.0, np.pi / 2]) == pytest.approx(np.sqrt(2) / 2)
    assert response.circular_coherence([0.0, np.pi]) == pytest.approx(0.0, abs=1e-12)
    assert response.circular_coherence([0.0, np.pi / 2, np.pi, -np.pi / 2]) == pytest.approx(0.0, abs=1e-12)
    # wrap-around: pi - 0.1 and -pi + 0.1 are 0.2 rad apart, |<e^{i phi}>| = cos(0.1)
    assert response.circular_coherence([np.pi - 0.1, -np.pi + 0.1]) == pytest.approx(np.cos(0.1))


def force_windows_by_hand(r2s, phases):
    return [ForceWindow(500.0 * i, 500.0 * i + 1000.0, r2, 2e-5, ph, 0.0) for i, (r2, ph) in enumerate(zip(r2s, phases))]


def test_a_response_check_single_law():
    big = np.radians(90)
    good = force_windows_by_hand([0.95] * 4, [0.3, 0.32, 0.28, 0.31])
    check, thin, law = response.a_response_check(good, None, big, SETTINGS)
    assert (check, thin, law) == (CheckValue.PASS, False, 'single')
    # small amplitude (30 deg < 45 deg): sine and line differ < 10 %, not measured
    check, _, _ = response.a_response_check(good, None, np.radians(30), SETTINGS)
    assert check == CheckValue.NOT_MEASURED
    # one window only
    check, _, _ = response.a_response_check(good[:1], None, big, SETTINGS)
    assert check == CheckValue.NOT_MEASURED
    # no force
    check, _, _ = response.a_response_check(force_windows_by_hand([0.1] * 4, [0.3] * 4), None, big, SETTINGS)
    assert check == CheckValue.FAIL
    # strong fit while the phase turns round: R^2 decides, the spread of phi0 is a column only (W7 'phi0', variant D:
    # with a free offset C the phase is degenerate with C; a modulated pendulum's phase moves)
    spun = force_windows_by_hand([0.95] * 4, [0.0, np.pi / 2, np.pi, -np.pi / 2])
    check, _, _ = response.a_response_check(spun, None, big, SETTINGS)
    assert check == CheckValue.PASS
    # R^2 between 0.4 and 0.7: thin margin
    _, thin, _ = response.a_response_check(force_windows_by_hand([0.55] * 4, [0.3] * 4), None, big, SETTINGS)
    assert thin


def test_a_response_check_phase_across_the_wrap_is_stable():
    # phases pi - 0.05 and -pi + 0.05 are 0.1 rad apart
    windows = force_windows_by_hand([0.95] * 4, [np.pi - 0.05, -np.pi + 0.05, np.pi - 0.04, -np.pi + 0.04])
    check, _, _ = response.a_response_check(windows, None, np.radians(90), SETTINGS)
    assert check == CheckValue.PASS


def test_a_response_check_takes_the_multiplet_law():
    windows = force_windows_by_hand([0.5] * 4, [0.0, 1.2, -1.0, 2.0])
    multiplet = MultipletFit((0, 1, -1), np.ones((4, 3), dtype=complex), 0.95, 0.95, (0.4, 0.2))
    check, _, law = response.a_response_check(windows, multiplet, np.radians(90), SETTINGS)
    assert check == CheckValue.PASS
    assert law == 'multiplet'


def test_omega0_from_force_by_hand():
    # sqrt(0.04 * 1e-4) = 2e-3
    assert response.omega0_from_force(-0.04, 1e-4) == pytest.approx(2e-3)
    assert response.omega0_from_force(0.04, 1e-4) == pytest.approx(2e-3)


def test_b1_amplitude_resonant_and_kinematic():
    # required = (2 pi / 250) * 0.5 / 0.6 = 0.0209440 au
    t = np.arange(0.0, 5000.0, 1.0)
    required = 2 * np.pi / 250.0 * 0.5 / 0.6
    a = 3.97 + required * np.sin(2 * np.pi * t / 250.0 + 0.3)
    req, meas, ratio = response.b1_amplitude(t, a, 250.0, 0.5, -0.6, (0.0, 4999.0))
    assert req == pytest.approx(0.0209440, rel=1e-5)
    assert meas == pytest.approx(required, rel=1e-3)
    assert ratio == pytest.approx(1.0, rel=1e-3)
    # the angle oscillates but a barely moves (285637: 1/5600)
    a = 3.97 + required / 5600 * np.sin(2 * np.pi * t / 250.0)
    _, _, ratio = response.b1_amplitude(t, a, 250.0, 0.5, 0.6, (0.0, 4999.0))
    assert ratio == pytest.approx(1 / 5600, rel=1e-2)


def test_circulation_strength_negative_rate_by_hand():
    # <|rate|> = 0.1; 0.1 / (2 * 0.02) = 2.5; wave K / <|rate|> = 1e-4 / 0.1 = 1e-3 au
    ratio, wave = response.circulation_strength(np.full(100, -0.1), 0.02, 1e-4)
    assert ratio == pytest.approx(2.5)
    assert wave == pytest.approx(1e-3)


def test_circulating_angle_forced_wave_matches_the_simulation():
    # uniform circulation sigma = nu t, a_dot = -K sin sigma: delta a = K / nu
    nu, force = 0.05, 2e-5
    t = np.arange(0.0, 2000.0, 0.1)
    a = 3.0 + force / nu * np.cos(nu * t)
    _, wave = response.circulation_strength(np.full(t.size, nu), 0.01, force)
    assert wave == pytest.approx(np.ptp(a) / 2, rel=1e-4)


def slow_components(t, angle, adot, varpi=None, e=None):
    zeros = np.zeros_like(t)
    return SlowComponents(
        window_yr=100.0,
        t_yr=t,
        angle_slow=angle,
        a_slow=3.0 + zeros,
        adot_slow=adot,
        varpi_slow=zeros if varpi is None else varpi,
        e_slow=0.1 + zeros if e is None else e,
        window=(float(t[0]), float(t[-1])),
    )


def test_slow_adot_r2_answering_and_silent():
    t = np.arange(0.0, 50000.0, 10.0)
    angle = 0.4 + 2.0 * np.sin(2 * np.pi * t / 9000.0)
    rng = np.random.default_rng(3)
    assert response.slow_adot_r2(slow_components(t, angle, 1e-6 * np.sin(angle - 0.2))) > 0.999
    assert response.slow_adot_r2(slow_components(t, angle, 1e-6 * rng.standard_normal(t.size))) < 0.05


def test_slow_libration_bound_by_hand_and_true_force_recovered():
    # truth: slow libration with force K, arc S = 2.36 centred on the stable point
    t = np.linspace(0.0, 20000.0, 4001)
    nu, force, c = -2.36 / 20000.0, 3e-9, 0.6
    sigma = -nu * 10000.0 + nu * t  # from +1.18 to -1.18 rad
    a = 3.0 + force / nu * (np.cos(sigma) - np.cos(sigma[0]))
    out = response.slow_libration_bound(t, sigma, a, c, (0.0, 20000.0))
    assert out['centre_rate'] == pytest.approx(nu, rel=1e-9) or out['centre_rate'] == pytest.approx(abs(nu), rel=1e-9)
    assert out['centre_rate_err'] == pytest.approx(0.0, abs=1e-12)
    assert out['arc'] == pytest.approx(2.36, rel=1e-6)
    assert out['force_max'] == pytest.approx(force, rel=1e-6)
    omega0_max = np.sqrt(c * force)
    assert out['omega0_max'] == pytest.approx(omega0_max, rel=1e-6)
    assert out['ratio'] == pytest.approx(abs(nu) / (2 * omega0_max), rel=1e-6)
    # K_max / (da / T_w) = S / (1 - cos(S/2)) = 3.81 at S = 2.36
    assert out['force_max'] / (np.ptp(a) / 20000.0) == pytest.approx(3.81, abs=0.01)


def test_slow_libration_bound_large_arc_and_error():
    # S = 3 pi > 2 pi: K_max = |nu| da / 2; centre with a wiggle b sin -> err = 2 std sqrt 3 / T_w
    t = np.linspace(0.0, 30000.0, 30001)
    nu, b = 3 * np.pi / 30000.0, 0.05
    centre = 1.0 + nu * t + b * np.sin(2 * np.pi * t / 1000.0)
    a = 3.0 + 1e-3 * np.sin(2 * np.pi * t / 7000.0)
    out = response.slow_libration_bound(t, centre, a, -0.6, (0.0, 30000.0))
    assert out['arc'] == pytest.approx(3 * np.pi, rel=1e-3)
    assert out['force_max'] == pytest.approx(abs(nu) * np.ptp(a) / 2, rel=1e-3)
    assert out['centre_rate_err'] == pytest.approx(2 * (b / np.sqrt(2)) * np.sqrt(3) / 30000.0, rel=0.03)


def bound(rate, err, ratio):
    return {
        'centre_rate': rate,
        'centre_rate_err': err,
        'arc': 1.0,
        'force_max': 1e-9,
        'omega0_max': 1e-4,
        'period_min': 6e4,
        'ratio': ratio,
    }


def test_slow_libration_check_routes():
    nan = float('nan')
    check, route, exact = response.slow_libration_check(bound(1e-3, 1e-5, 2.5), nan, nan, 0.01, SETTINGS)
    assert (check, route, exact) == (CheckValue.PASS, 'data', False)
    # data fail: the mu e^q scale excludes nothing (W3-F3)
    check, route, _ = response.slow_libration_check(bound(-1e-3, 1e-5, 0.3), 1e-4, nan, 0.01, SETTINGS)
    assert (check, route) == (CheckValue.FAIL, 'none')
    # data fail, no theory, the owner's half-width 0.001 < 0.01 / 2
    check, route, _ = response.slow_libration_check(bound(1e-3, 1e-5, 0.3), nan, 0.001, 0.01, SETTINGS)
    assert (check, route) == (CheckValue.PASS, 'owner')
    # nothing excludes it
    check, route, exact = response.slow_libration_check(bound(1e-3, 1e-5, 0.3), nan, nan, 0.01, SETTINGS)
    assert (check, route, exact) == (CheckValue.FAIL, 'none', False)
    # centre rate within 2 sigma of zero: exact commensurability
    _, _, exact = response.slow_libration_check(bound(-1e-6, 1e-5, 0.01), nan, nan, 0.01, SETTINGS)
    assert exact


def test_onset_bands_before_inside_after():
    t = np.arange(0.0, 12000.0, 1.0)
    period, delta = 250.0, 0.01
    inside = (t >= 4000.0) & (t <= 8000.0)
    rng = np.random.default_rng(5)
    a = 3.97 + 1e-6 * rng.standard_normal(t.size) + np.where(inside, delta * np.sin(2 * np.pi * t / period), 0.0)
    before, during, after = response.onset_bands(t, a, period, 4000.0, 8000.0)
    assert during == pytest.approx(delta / np.sqrt(2), rel=0.15)
    assert before < 0.5 * during
    assert after < 0.5 * during
    # an episode at the start of the record has no "before" stretch
    before, _, _ = response.onset_bands(t, a, period, 0.0, 8000.0)
    assert math.isnan(before)


# ---------------------------------------------------------------- kinematics

G_BODY = np.radians(40.0 / 3600.0)  # 40 arcsec/yr
G5 = np.radians(4.257 / 3600.0)


@pytest.mark.parametrize('g, g5', [(G_BODY, G5), (-G_BODY, G5), (G5, G_BODY)])
def test_varpi_rate_modulation_against_rotating_vectors(g, g5):
    # e exp(i varpi) = e_f exp(i g t) + e_F exp(i g5 t); truth from the numerical derivative of the argument
    e_free, e_forced = 0.15, 0.05
    t = np.linspace(0.0, 4 * 2 * np.pi / abs(g - g5), 400001)
    vec = e_free * np.exp(1j * g * t) + e_forced * np.exp(1j * g5 * t)
    varpi_dot = np.gradient(np.unwrap(np.angle(vec)), t)
    out = kinematics.varpi_rate_modulation(np.abs(vec), g, g5)
    assert out[0] == pytest.approx(e_free, rel=1e-6)
    assert out[1] == pytest.approx(e_forced, rel=1e-6)
    assert out[2] == pytest.approx(varpi_dot.min(), rel=1e-4)
    assert out[3] == pytest.approx(varpi_dot.max(), rel=1e-4)
    assert out[4] == pytest.approx(2 * np.pi / abs(g - g5), rel=1e-12)


@pytest.mark.parametrize('omega_rate', [np.pi / 17954.0, -np.pi / 17954.0])
def test_two_omega_cycle_285637(omega_rate):
    # 285637: P = pi / |omega_dot| = 17954 yr; e follows cos 2 omega exactly; omega given wrapped to [0, 2 pi)
    t = np.arange(0.0, 100000.0, 10.0)
    omega = 1.0 + omega_rate * t
    e = 0.2 + 0.01 * np.cos(2 * omega)
    for given in (omega, omega % (2 * np.pi)):
        period, corr = kinematics.two_omega_cycle(t, given, e)
        assert period == pytest.approx(17954.0, rel=1e-3)
        assert corr > 0.999


@pytest.mark.parametrize('p', [2, -1])
def test_cancellation_by_hand(p):
    # varpi = g t + beta sin(nu t): p varpi_dot in p (g +- beta nu); psi = -p g t + b sin(mu t): rms b / sqrt 2
    t = np.arange(0.0, 200000.0, 5.0)
    g, beta, nu = 2e-4, 0.3, 2 * np.pi / 40000.0
    varpi = g * t + beta * np.sin(nu * t)
    psi = -p * g * t + 0.2 * np.sin(2 * np.pi * t / 3000.0)
    rate, rms, mean, vmin, vmax = kinematics.cancellation(t, psi, p, varpi)
    lo, hi = sorted([p * (g - beta * nu), p * (g + beta * nu)])
    assert rate == pytest.approx(-p * g, rel=1e-3)
    assert rms == pytest.approx(0.2 / np.sqrt(2), rel=0.02)
    assert mean == pytest.approx(p * g, rel=1e-3)
    assert vmin == pytest.approx(lo, rel=1e-3)
    assert vmax == pytest.approx(hi, rel=1e-3)
    assert rate / mean == pytest.approx(-1.0, rel=1e-3)  # cancel ratio


@pytest.mark.parametrize('p', [1, -1, 2])
def test_kinematic_correlations_angle_follows_p_varpi(p):
    t = np.arange(0.0, 200000.0, 50.0)
    varpi = 2e-4 * t + 0.3 * np.sin(2 * np.pi * t / 40000.0)
    slow = slow_components(t, 0.7 + p * varpi, np.zeros_like(t), varpi=varpi)
    corr_rate, corr_detr = kinematics.kinematic_correlations(slow, p)
    assert corr_rate > 0.99
    assert corr_detr > 0.99


def test_kinematic_correlations_unrelated_angle():
    t = np.arange(0.0, 200000.0, 50.0)
    varpi = 2e-4 * t + 0.3 * np.sin(2 * np.pi * t / 40000.0)
    angle = 0.5 * np.sin(2 * np.pi * t / 13377.0 + 0.2)
    corr_rate, corr_detr = kinematics.kinematic_correlations(slow_components(t, angle, np.zeros_like(t), varpi=varpi), 1)
    assert abs(corr_rate) < 0.2
    assert abs(corr_detr) < 0.2


def test_kinematic_check():
    assert kinematics.kinematic_check(0.97, 0.1, 10.0, 1, SETTINGS) == CheckValue.FAIL  # mechanism found
    assert kinematics.kinematic_check(0.97, 0.6, 10.0, 1, SETTINGS) == CheckValue.PASS  # a answers
    assert kinematics.kinematic_check(0.5, 0.1, 10.0, 1, SETTINGS) == CheckValue.PASS
    assert kinematics.kinematic_check(-0.97, 0.1, 10.0, -1, SETTINGS) == CheckValue.PASS  # anticorrelated is no follow
    assert kinematics.kinematic_check(0.97, 0.1, 2.0, 1, SETTINGS) == CheckValue.NOT_MEASURED
    assert kinematics.kinematic_check(0.97, 0.1, 10.0, 0, SETTINGS) == CheckValue.NOT_APPLICABLE
    assert kinematics.kinematic_check(float('nan'), 0.1, 10.0, 1, SETTINGS) == CheckValue.NOT_MEASURED


# ---------------------------------------------------------------- strength


def test_mu_scale_by_hand():
    assert strength.mu_scale(1e-3, 0.1, 2) == pytest.approx(1e-5)
    assert strength.mu_scale(1e-3, 0.1, 0) == pytest.approx(1e-3)
    assert strength.mu_scale(1e-3, 0.2, 3) == pytest.approx(8e-6)


def test_eq_legal():
    assert strength.eq_legal(0.1, [], SETTINGS)
    assert not strength.eq_legal(0.35, [], SETTINGS)
    assert not strength.eq_legal(0.3, [], SETTINGS)  # "mean e < 0.3"
    assert not strength.eq_legal(0.1, ['Jupiter'], SETTINGS)
    assert not strength.eq_legal(float('nan'), [], SETTINGS)


@pytest.mark.parametrize(
    'small_period, k, orbital_period, expected',
    [
        # (7.9 / 300)^2 / (3 * 4) = 6.93444e-4 / 12 = 5.77870e-5
        (300.0, -2, 7.9, 5.77870e-5),
        # (5 / 1000)^2 / (3 * 9) = 2.5e-5 / 27 = 9.25926e-7
        (1000.0, 3, 5.0, 9.25926e-7),
    ],
)
def test_mu_eff_required_by_hand(small_period, k, orbital_period, expected):
    n = 2 * np.pi / orbital_period
    assert strength.mu_eff_required(small_period, k, n) == pytest.approx(expected, rel=1e-5)


def test_omega0_theory_by_hand_and_inverse_of_mu_eff():
    # 2 * 0.795 * sqrt(3e-5) = 1.59 * 0.00547723 = 0.00870879
    assert strength.omega0_theory(-2, 0.795, 1e-5) == pytest.approx(0.00870879, rel=1e-5)
    omega0 = strength.omega0_theory(3, HILDA_N, 2.4e-6)
    assert strength.mu_eff_required(2 * np.pi / omega0, 3, HILDA_N) == pytest.approx(2.4e-6, rel=1e-10)


def test_mu_eff_uses_small_period_not_libration_period():
    # at A = 150 deg the libration period is F(A) = 1.762 times T0; mu_eff from T0 is F^2 = 3.10 times larger
    period = 450.0
    t0 = pendulum.small_period(period, np.radians(150))
    assert strength.mu_eff_required(t0, -2, HILDA_N) / strength.mu_eff_required(period, -2, HILDA_N) == pytest.approx(1.762**2, rel=2e-3)


def test_strength_check():
    assert strength.strength_check(True, 1.0, SETTINGS) == CheckValue.PASS
    assert strength.strength_check(True, 10.0, SETTINGS) == CheckValue.PASS  # [0.1, 10] inclusive
    assert strength.strength_check(True, 0.1, SETTINGS) == CheckValue.PASS
    assert strength.strength_check(True, 20.0, SETTINGS) == CheckValue.FAIL
    assert strength.strength_check(True, 0.05, SETTINGS) == CheckValue.FAIL
    assert strength.strength_check(False, 1.0, SETTINGS) == CheckValue.NOT_MEASURED
    assert strength.strength_check(True, float('nan'), SETTINGS) == CheckValue.NOT_MEASURED
