"""W2 zone-rules: energies in a modulated frame, regime crossings and the zone classes (pendulum.py).

Truths come from closed forms (h = sin^2(A/2) at the turning points, the small-oscillation period 2 pi / Omega) and
from trajectories integrated here independently of the code under test.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import pendulum
from resonances.resonance.analysis.pendulum import CycleClass, CycleEnergy, EnergyAnalysis, Zone, ZoneMethod
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
OMEGA = 2 * np.pi / 300.0


def moving_well_record(amplitude_deg=20.0, wobble_deg=30.0, wobble_period=6000.0, span=24000.0, step=1.0):
    """sigma_ddot = -Omega^2 sin(sigma - s(t)), s(t) = wobble sin(2 pi t / P_w): a libration whose stable point follows a
    slow forced term (a Hilda's centre follows the forced varpi while it librates)."""
    wobble = np.radians(wobble_deg)

    def well(t):
        return wobble * np.sin(2 * np.pi * t / wobble_period)

    def rhs(t, y):
        return [y[1], -(OMEGA**2) * np.sin(y[0] - well(t))]

    t = np.arange(0.0, span, step)
    h = np.sin(np.radians(amplitude_deg) / 2) ** 2
    sol = solve_ivp(rhs, (0, span), [0.0, 2 * OMEGA * np.sqrt(h)], t_eval=t, rtol=1e-11, atol=1e-13)
    return t, sol.y[0], sol.y[1], well(t)


def cycles_of(t, sigma):
    """Bounds (TP_i, TP_i+2) of the maxima/minima of sigma."""
    rate = np.gradient(sigma, t)
    turns = t[1:][np.diff(np.sign(rate)) != 0]
    return [(turns[i], turns[i + 2], False) for i in range(0, len(turns) - 2, 2)]


def test_fixed_frame_is_the_unmodulated_pendulum():
    t, sigma, rate, _ = moving_well_record(wobble_deg=0.0)
    bounds = cycles_of(t, sigma)
    windows = [(a, b) for a, b, _ in bounds]
    plain, _ = pendulum.fit_potential_from_rate(t, sigma, rate, windows, S)
    framed, _ = pendulum.fit_potential_from_rate(t, sigma, rate, windows, S, frame=pendulum.fixed_frame(t))
    assert framed.cos_coeffs == pytest.approx(plain.cos_coeffs) and framed.sin_coeffs == pytest.approx(plain.sin_coeffs)
    first = pendulum.cycle_energies(t, sigma, rate, bounds, plain)
    second = pendulum.cycle_energies(t, sigma, rate, bounds, plain, frame=pendulum.fixed_frame(t))
    assert [c.h for c in first] == pytest.approx([c.h for c in second])
    assert all(c.h_mean == pytest.approx(c.h) and c.strength_rel_sq == pytest.approx(1.0) for c in second)


def test_a_moving_well_is_read_in_its_frame():
    t, sigma, rate, well = moving_well_record()
    bounds = cycles_of(t, sigma)
    windows = [(a, b) for a, b, _ in bounds]
    fixed, _ = pendulum.fit_potential_from_rate(t, sigma, rate, windows, S)
    shift_rate = np.gradient(well, t)
    frame = pendulum.ModulatedFrame(t, np.ones_like(t), well - well.mean(), shift_rate)
    moving, per_window = pendulum.fit_potential_from_rate(t, sigma, rate, windows, S, frame=frame)
    # the wobble (30 deg) is larger than the libration (20 deg): a fixed well cannot explain sigma_dot^2
    assert fixed.r2 < 0.5 < 0.95 < moving.r2
    assert moving.omega_rad_yr == pytest.approx(OMEGA, rel=0.03)
    cycles = pendulum.cycle_energies(t, sigma, rate, bounds, moving, frame=frame)
    assert np.median([c.h for c in cycles]) == pytest.approx(np.sin(np.radians(20) / 2) ** 2, rel=0.15)
    assert np.median([c.period_law_ratio for c in cycles]) == pytest.approx(1.0, abs=0.05)
    assert pendulum.well_coherence(per_window) > 0.99


def test_one_harmonic_potential_in_closed_form():
    # V = R cos(sigma - theta): top at theta, well at theta + pi, Omega^2 = R, V from -R to R
    radius, theta = 2.0, 0.7
    potential = pendulum.potential_from_coefficients([radius * np.cos(theta)], [radius * np.sin(theta)])
    assert potential.top_rad == pytest.approx(theta)
    assert potential.well_rad == pytest.approx(theta + np.pi)
    assert potential.omega_rad_yr == pytest.approx(np.sqrt(radius))
    assert (potential.v_min, potential.v_max) == pytest.approx((-radius, radius))
    grid = np.linspace(0, 2 * np.pi, 7)
    assert potential.value(grid) == pytest.approx(radius * np.cos(grid - theta))


def _moments(*args):
    """The distinct times of `regime_transitions`, sorted (a slip between two librations is one time)."""
    return sorted(set(moment for moment, _ in pendulum.regime_transitions(*args)))


def test_regime_moments_from_the_angle_itself():
    t = np.arange(0.0, 100.0, 0.1)
    # libration about 0 (TPs at +-1), a slip to the next turn at t = 50, libration about 2 pi
    angle = np.where(t < 50, np.sin(t), 2 * np.pi + np.sin(t))
    times = [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0]
    # the half-cycle 45 -> 55 swings by 2 pi + 2 > 2 pi: a slip, through the middle of the swing (pi) at t = 50
    values = [1.0, -1.0, 1.0, -1.0, -1.0, 2 * np.pi + 1.0, 2 * np.pi - 1.0]
    crossings = _moments(t, angle, times, values, [])
    assert crossings == pytest.approx([50.0], abs=0.2)
    # turns only: no crossing; a libration that ends in turns: one, between the last half-cycle (middle 4) and turn (20)
    assert _moments(t, angle, [], [], [10.0, 20.0, 30.0]) == []
    assert _moments(t, angle, [1.0, 3.0, 5.0], [1.0, -1.0, 1.0], [20.0, 30.0]) == pytest.approx([12.0])
    # a large libration about a moving well (swing <= 2 pi) is not a crossing
    assert _moments(t, angle, [1.0, 3.0, 5.0], [3.0, -3.0, 3.1], []) == []


def _cycle(h, j=np.nan, rotation=False, h_mean=np.nan, m=1.0):
    return CycleEnergy(0.0, 1.0, rotation, h, j, 1.0, 1.0, h_mean, m)


def test_zone_classes_use_the_mean_normalisation():
    zone = Zone(ZoneMethod.NUMERIC, 0.5, pendulum.h_from_action_ratio(0.5), 1.3, 0.7, 1.25, False)
    deep = _cycle(0.2, j=0.2)
    inside = _cycle(0.9, j=0.8)
    # a turn with h = 1.2 at a weak moment (m = 0.8) has 0.96 of the mean separatrix energy: inside the zone
    weak_turn = _cycle(1.2, rotation=True, h_mean=0.96, m=0.8)
    high_turn = _cycle(1.5, rotation=True, h_mean=1.5)
    classes = pendulum.classify_cycles([deep, inside, weak_turn, high_turn], zone, 0.02, S)
    assert classes == [CycleClass.DEEP, CycleClass.ZONE, CycleClass.ZONE, CycleClass.ABOVE]
    edges = pendulum.edge_cycles([_cycle(0.4, j=0.501), deep, _cycle(1.3, rotation=True, h_mean=1.31)], zone, 0.02, S)
    assert edges == [True, False, True]


def test_energy_analysis_counts_and_turn_distance():
    zone = Zone(ZoneMethod.NUMERIC, 0.5, 0.6, 1.2, 0.7, 1.2, False)
    energy = EnergyAnalysis(
        None,
        None,
        np.nan,
        np.nan,
        cycles=[_cycle(1.5, rotation=True, h_mean=1.5), _cycle(1.4, rotation=True, h_mean=1.4)],
        zone=zone,
        cycle_classes=[CycleClass.ABOVE, CycleClass.ZONE],
    )
    assert energy.n_outside == 1
    assert energy.turn_zone_widths == pytest.approx((1.45 - 1.2) / (1.2 - 0.6))
    columns = energy.to_columns()
    assert columns['sep_turn_zone_widths'] == pytest.approx(energy.turn_zone_widths)
    assert columns['sep_cycle_classes'] == 'above;zone' and columns['sep_n_outside'] == 1
