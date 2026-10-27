"""Shared pendulum (analysis.pendulum): closed forms against M&D 1999, numerical integration and hand values.

Independent truths: F(90/120/150 deg) = 1.180/1.373/1.762 (M&D 1999, eqs. 8.50-8.52); periods and the rate
ratio of a revolution from `solve_ivp` of sigma_ddot = -omega0^2 sin sigma; J/J_s from the phase-space area;
hand-computed scatter, crossings and cyclic-shift p-values.
"""

import numpy as np
import pytest
from scipy import integrate

from resonances.resonance.analysis import pendulum as pend
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue

SETTINGS = settings_from_config()


def integrate_pendulum(omega0, h, t_end, step, well=0.0):
    """sigma_ddot = -omega0^2 sin(sigma - well) from the stable point with sigma_dot = 2 omega0 sqrt(h)."""

    def rhs(_, y):
        return [y[1], -(omega0**2) * np.sin(y[0] - well)]

    t = np.arange(0.0, t_end, step)
    sol = integrate.solve_ivp(rhs, (0.0, t[-1]), [well, 2 * omega0 * np.sqrt(h)], t_eval=t, rtol=1e-11, atol=1e-12)
    return t, sol.y[0], sol.y[1]


def numeric_period(omega0, h):
    """Libration (h < 1) or revolution period from the event times of solve_ivp."""

    def rhs(_, y):
        return [y[1], -(omega0**2) * np.sin(y[0])]

    if h < 1:

        def event(_, y):
            return y[1]

        event.terminal, event.direction = True, -1
        sol = integrate.solve_ivp(rhs, (0, 1e6), [0.0, 2 * omega0 * np.sqrt(h)], events=event, rtol=1e-12, atol=1e-14)
        return 4 * sol.t_events[0][0]

    def event(_, y):
        return y[0] - 2 * np.pi

    event.terminal, event.direction = True, 1
    sol = integrate.solve_ivp(rhs, (0, 1e6), [0.0, 2 * omega0 * np.sqrt(h)], events=event, rtol=1e-12, atol=1e-14)
    return sol.t_events[0][0]


def test_amplitude_factor_against_murray_dermott():
    values = pend.amplitude_factor(np.radians([90.0, 120.0, 150.0]))
    assert values == pytest.approx([1.180, 1.373, 1.762], abs=1e-3)
    assert pend.amplitude_factor(0.0) == pytest.approx(1.0)
    assert np.isinf(pend.amplitude_factor(np.pi))


def test_small_period():
    # a pendulum of T0 = 1000 yr at A = 120 deg has P = 1373 yr
    assert pend.small_period(1372.88, np.radians(120)) == pytest.approx(1000.0, rel=1e-4)


@pytest.mark.parametrize('h', [0.1, 0.5, 0.9, 0.99])
def test_libration_period_against_integration(h):
    omega0 = 0.37
    assert pend.libration_period(h, omega0) == pytest.approx(numeric_period(omega0, h), rel=1e-8)


@pytest.mark.parametrize('h', [1.01, 1.2, 2.0, 5.0])
def test_revolution_period_against_integration(h):
    omega0 = 0.37
    assert pend.revolution_period(h, omega0) == pytest.approx(numeric_period(omega0, h), rel=1e-8)
    assert np.isnan(pend.revolution_period(0.5, omega0))
    assert np.isnan(pend.libration_period(1.5, omega0))


def test_rate_min_over_mean_values_and_integration():
    assert pend.rate_min_over_mean(1.012) == pytest.approx(0.25, abs=1e-3)
    assert pend.rate_min_over_mean(1.14) == pytest.approx(0.554, abs=1e-3)
    # min/mean of |sigma_dot| over one integrated turn
    omega0, h = 1.0, 1.3
    period = numeric_period(omega0, h)
    t, _, rate = integrate_pendulum(omega0, h, period, period / 20000)
    assert pend.rate_min_over_mean(h) == pytest.approx(rate.min() / (2 * np.pi / period), rel=1e-4)
    assert np.isnan(pend.rate_min_over_mean(0.9))


def test_action_ratio_limits_area_and_inverse():
    assert pend.action_ratio(1e-4) == pytest.approx(np.pi * 1e-4 / 4, rel=1e-3)
    assert pend.action_ratio(1.0) == pytest.approx(1.0)
    # area inside the libration curve (omega0 = 1): 2 int 2 sqrt(h - sin^2(phi/2)) dphi over the separatrix area 16
    h = 0.5
    amplitude = 2 * np.arcsin(np.sqrt(h))
    area, _ = integrate.quad(lambda p: 4 * np.sqrt(max(h - np.sin(p / 2) ** 2, 0.0)), -amplitude, amplitude, limit=200)
    assert pend.action_ratio(h) == pytest.approx(area / 16, rel=1e-6)
    assert pend.h_from_action_ratio(pend.action_ratio(0.73)) == pytest.approx(0.73, abs=1e-10)
    assert pend.h_from_action_ratio(1.5) == 1.0


def test_half_width_and_excursion():
    assert pend.half_width(0.002, -0.5) == pytest.approx(0.008)
    assert pend.excursion(0.002, np.pi, 0.5) == pytest.approx(0.008)
    assert pend.excursion(0.002, np.radians(60), 0.5) == pytest.approx(0.004)


def test_potential_value_extrema_and_energy():
    # V = -2 cos(sigma - 1) = -2 cos 1 cos sigma - 2 sin 1 sin sigma: well at 1 rad, Omega = sqrt(2)
    pot = pend.potential_from_coefficients([-2 * np.cos(1.0)], [-2 * np.sin(1.0)])
    assert pot.well_rad == pytest.approx(1.0, abs=1e-6)
    assert pot.top_rad == pytest.approx(1.0 + np.pi, abs=1e-6)
    assert pot.omega_rad_yr == pytest.approx(np.sqrt(2))
    assert (pot.v_min, pot.v_max) == pytest.approx((-2.0, 2.0))
    assert pot.energy_h(np.array([1.0 + np.pi]), np.array([0.0]))[0] == pytest.approx(1.0)
    assert pot.energy_h(np.array([1.0]), np.array([2.0]))[0] == pytest.approx(0.5)  # (2 + (-2) + 2) / 4
    assert pend.potential_from_coefficients([0.0], [0.0]) is None


def test_period_in_two_harmonic_potential_matches_closed_form_when_second_is_negligible():
    pot = pend.potential_from_coefficients([-2.0, 1e-12], [0.0, 0.0])
    omega = np.sqrt(2.0)
    for h in (0.3, 0.95):
        assert pend.period_in_potential(pot, h) == pytest.approx(pend.libration_period(h, omega), rel=1e-6)
    assert pend.period_in_potential(pot, 1.4) == pytest.approx(pend.revolution_period(1.4, omega), rel=1e-6)


def cycle_bounds(t, angle, rotation=False):
    """Full cycles from maximum to maximum (libration) or turns of 2 pi (revolution)."""
    if rotation:
        turns = np.floor((angle - angle[0]) / (2 * np.pi))
        edges = t[np.flatnonzero(np.diff(turns)) + 1]
    else:
        peaks = np.flatnonzero((angle[1:-1] > angle[:-2]) & (angle[1:-1] >= angle[2:])) + 1
        edges = t[peaks]
    return [(a, b, rotation) for a, b in zip(edges[:-1], edges[1:])]


def test_potential_from_rate_recovers_pendulum_and_energy():
    omega0, h, well = 2 * np.pi / 300.0, 0.5, 2.0
    t, angle, rate = integrate_pendulum(omega0, h, 3000.0, 0.5, well)
    bounds = cycle_bounds(t, angle)
    potential, per_window = pend.fit_potential_from_rate(t, angle, rate, [(a, b) for a, b, _ in bounds], SETTINGS)
    assert potential.n_harmonics == 1
    assert potential.well_rad == pytest.approx(well, abs=1e-3)
    assert potential.omega_rad_yr == pytest.approx(omega0, rel=1e-3)
    assert potential.r2 == pytest.approx(1.0, abs=1e-6)
    assert pend.well_coherence(per_window) == pytest.approx(1.0, abs=1e-6)
    cycles = pend.cycle_energies(t, angle, rate, bounds, potential)
    assert [c.h for c in cycles] == pytest.approx([h] * len(cycles), abs=1e-3)
    assert [c.period_law_ratio for c in cycles] == pytest.approx([1.0] * len(cycles), abs=2e-3)
    assert [c.action_ratio for c in cycles] == pytest.approx([pend.action_ratio(h)] * len(cycles), abs=1e-3)
    law = float(np.median([c.period_law_ratio for c in cycles]))
    assert pend.sigma_only_check(law, pend.well_coherence(per_window), SETTINGS) == CheckValue.PASS


def test_potential_from_rate_on_revolutions():
    omega0, h = 2 * np.pi / 300.0, 1.4
    t, angle, rate = integrate_pendulum(omega0, h, 4000.0, 0.5)
    bounds = cycle_bounds(t, angle, rotation=True)
    potential, _ = pend.fit_potential_from_rate(t, angle, rate, [(a, b) for a, b, _ in bounds], SETTINGS)
    cycles = pend.cycle_energies(t, angle, rate, bounds, potential)
    assert all(c.rotation and np.isnan(c.action_ratio) for c in cycles)
    assert np.median([c.h for c in cycles]) == pytest.approx(h, abs=2e-3)
    assert np.median([c.period_law_ratio for c in cycles]) == pytest.approx(1.0, abs=2e-3)


def test_second_harmonic_is_kept_only_when_it_pays():
    # V = -cos sigma - 0.6 cos 2 sigma, a revolution (E = 1.78 > V_top = 0.81): sigma_dot^2 follows cos 2 sigma,
    # one harmonic explains little of it, so N = 2 (as for 145480, R^2 0.30 -> 0.87)
    def rhs(_, y):
        return [y[1], -(np.sin(y[0]) + 1.2 * np.sin(2 * y[0]))]

    t = np.arange(0.0, 200.0, 0.01)
    sol = integrate.solve_ivp(rhs, (0, t[-1]), [0.0, 2.6], t_eval=t, rtol=1e-11, atol=1e-12)
    angle, rate = sol.y
    bounds = cycle_bounds(t, angle, rotation=True)
    windows = [(a, b) for a, b, _ in bounds]
    potential, _ = pend.fit_potential_from_rate(t, angle, rate, windows, SETTINGS)
    assert potential.n_harmonics == 2
    assert potential.cos_coeffs == pytest.approx([-1.0, -0.6], abs=1e-3)
    assert potential.sin_coeffs == pytest.approx([0.0, 0.0], abs=1e-3)
    cycles = pend.cycle_energies(t, angle, rate, bounds, potential)
    assert np.median([c.period_law_ratio for c in cycles]) == pytest.approx(1.0, abs=2e-3)
    one = pend.fit_potential_from_rate(t, angle, rate, windows, settings_from_config(separatrix_harmonics_max=1))[0]
    assert one.n_harmonics == 1 and one.r2 < potential.r2 - SETTINGS.separatrix_harmonic_gain


def test_potential_from_force_equals_potential_from_rate():
    # identity: a = a* + sigma_dot / c, so c a_dot = sigma_ddot = -V'(sigma)
    omega0, h, well, c = 2 * np.pi / 300.0, 0.6, 0.5, -0.7
    t, angle, rate = integrate_pendulum(omega0, h, 3000.0, 0.5, well)
    a_dot = np.gradient(rate / c, t)
    potential = pend.fit_potential_from_force(angle, a_dot, c, 1)
    assert potential.well_rad == pytest.approx(well, abs=1e-3)
    assert potential.omega_rad_yr == pytest.approx(omega0, rel=1e-3)
    assert potential.r2 > 0.999


def test_kinematic_swing_fails_the_period_law():
    # forced swing (synthetic case D of SEPARATRIX §4.1): sigma = c(t) + 2.5 sin(w t) with a drifting centre
    w = 2 * np.pi / 300.0
    t = np.arange(0.0, 6000.0, 0.5)
    angle = 0.0004 * t + 2.5 * np.sin(w * t)
    rate = 0.0004 + 2.5 * w * np.cos(w * t)
    bounds = cycle_bounds(t, angle)
    potential, per_window = pend.fit_potential_from_rate(t, angle, rate, [(a, b) for a, b, _ in bounds], SETTINGS)
    cycles = pend.cycle_energies(t, angle, rate, bounds, potential)
    law = float(np.nanmedian([c.period_law_ratio for c in cycles]))
    check = pend.sigma_only_check(law, pend.well_coherence(per_window), SETTINGS)
    assert check == CheckValue.FAIL


def test_energy_scatter_by_hand():
    # differences 1, 2, 3: median 2, MAD 1 -> 1.4826 / sqrt 2
    assert pend.energy_scatter([0.0, 1.0, 3.0, 6.0], 0.02) == pytest.approx(1.4826 / np.sqrt(2))
    assert pend.energy_scatter([1.0, 1.1, 1.0, 1.1], 0.02) == pytest.approx(0.02)  # MAD 0: the floor
    assert np.isnan(pend.energy_scatter([1.0], 0.02))


def cycle(t0, t1, h, rotation=False):
    ratio = pend.action_ratio(h) if not rotation else float('nan')
    return pend.CycleEnergy(t0, t1, rotation, h, ratio, t1 - t0, 1.0)


def test_crossings_between_cycles_and_slips():
    cycles = [cycle(0, 10, 0.9), cycle(10, 20, 1.1, True), cycle(20, 30, 1.2, True), cycle(30, 40, 0.95)]
    assert pend.crossings(cycles, [35.0]) == [10.0, 30.0, 35.0]


def test_classify_cycles_by_zone():
    zone = pend.Zone(
        pend.ZoneMethod.NUMERIC, j_lo=0.6, h_lo=pend.h_from_action_ratio(0.6), h_hi=1.2, adiabatic_lo=0.9, adiabatic_hi=1.1, mismatch=False
    )
    cycles = [cycle(0, 1, 0.2), cycle(1, 2, 0.9), cycle(2, 3, 1.5, True), cycle(3, 4, 1.1, True)]
    classes = pend.classify_cycles(cycles, zone, 0.02, SETTINGS)
    assert classes == [pend.CycleClass.DEEP, pend.CycleClass.ZONE, pend.CycleClass.ABOVE, pend.CycleClass.ZONE]
    # J(0.2) = 0.166 + 2 * (K(0.2)/2) * 0.3 = 0.166 + 0.495 > 0.6: a large scatter moves it into the zone
    assert pend.classify_cycles(cycles[:1], zone, 0.3, SETTINGS) == [pend.CycleClass.ZONE]


def test_cyclic_shift_p_value_by_hand():
    # weak phase: the first 2 of every 10 samples (20 %); three crossings at 0, 10, 20 are all weak.
    # Shifting the pattern keeps all three weak only for shifts = 0, 1 (mod 10): p = 20/100.
    t = np.arange(100.0)
    weak = (t % 10) < 2
    assert pend.cyclic_shift_p_value([0.0, 10.0, 20.0], weak, t) == pytest.approx(0.2)
    assert np.isnan(pend.cyclic_shift_p_value([], weak, t))


def test_action_trend():
    cycles = [cycle(10 * i, 10 * i + 10, pend.h_from_action_ratio(0.3 + 0.001 * (10 * i + 5))) for i in range(6)]
    slope, error = pend.action_trend(cycles)
    assert slope == pytest.approx(0.001, rel=1e-6)
    assert error == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(pend.action_trend(cycles[:2])[0])


def test_r0_checks():
    assert pend.sigma_only_check(1.1, 0.95, SETTINGS) == CheckValue.PASS
    assert pend.sigma_only_check(0.61, 0.95, SETTINGS) == CheckValue.FAIL
    assert pend.sigma_only_check(1.0, 0.71, SETTINGS) == CheckValue.FAIL
    assert pend.sigma_only_check(float('nan'), 0.95, SETTINGS) == CheckValue.NOT_MEASURED
    assert pend.with_slow_check(CheckValue.PASS, [0.95, 0.92, 0.91, 0.5], SETTINGS) == CheckValue.PASS  # 3/4 >= 0.667
    assert pend.with_slow_check(CheckValue.PASS, [0.95, 0.5, 0.5], SETTINGS) == CheckValue.FAIL
    assert pend.with_slow_check(CheckValue.FAIL, [0.95], SETTINGS) == CheckValue.FAIL
    assert pend.with_slow_check(CheckValue.PASS, [], SETTINGS) == CheckValue.NOT_MEASURED


def test_energy_analysis_columns():
    pot = pend.potential_from_coefficients([0.0], [-1.0], r2=0.9)  # V = -sin sigma: well at 90 deg
    zone = pend.Zone(pend.ZoneMethod.EQ_ONLY, 0.8, 0.7, 1.1, 0.9, 1.1, False)
    energy = pend.EnergyAnalysis(
        potential_e=pot,
        potential_f=None,
        ve_vf_agree_fraction=0.5,
        well_coherence=0.97,
        zone=zone,
        cycle_classes=[pend.CycleClass.DEEP, pend.CycleClass.ZONE, pend.CycleClass.ZONE],
        crossing_times_yr=[100.0, 250.0],
        eps=(0.3, 0.1),
    )
    cols = energy.to_columns()
    assert cols['sep_well_deg'] == pytest.approx(90.0, abs=1e-4)
    assert cols['sep_top_deg'] == pytest.approx(270.0, abs=1e-4)
    assert cols['sep_N'] == 1 and cols['sep_VE_R2'] == 0.9 and np.isnan(cols['sep_VF_R2'])
    assert (cols['sep_n_deep'], cols['sep_n_zone'], cols['sep_n_above'], cols['sep_n_crossings']) == (1, 2, 0, 2)
    assert cols['sep_crossing_times_yr'] == '100;250'
    assert cols['sep_eps'] == '0.3;0.1'
    assert cols['sep_zone_method'] == 'eq_only'
    assert cols['sep_permutation_method'] == ''
