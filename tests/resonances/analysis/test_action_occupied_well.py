"""W25: the action J/J_s of a libration cycle is that of the well it occupies.

The action of a libration is the phase-space area of its own loop over 2 pi, and the separatrix action J_s that bounds
it is that of the separatrix of the well holding the loop. Before W25 `cycle_energies`, `classify_cycles` and
`edge_cycles` integrated both in the deepest well of a fitted V for every cycle, while the period law already read the
cycle's own well (`cycle_period`). The reference values here come from mpmath (its own quadrature and root finding, at
30 digits), not from the package's scipy quadratures.

V = -cos s + 0.6 cos 2s - 0.25 sin s (two harmonics): wells at 68.0 deg (h = 0) and 298.2 deg (h = 0.1714), maxima at
349.6 deg (h = 0.2495, the separatrix of both wells) and 184.2 deg (h = 1). With two harmonics the two wells have one
period at one energy and J_deep(E) - J_shallow(E) is a constant (z = e^{i s} makes E - V a quartic over z^2), so the
deepest well's ratio of a shallow-well cycle, (J + C)/(J_s + C), is always too large: at h = 0.18 it is 0.664 where the
cycle's own is 0.096.
"""

import math

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import pendulum as p
from resonances.resonance.analysis.pendulum import CycleClass, Zone, ZoneMethod
from resonances.resonance.analysis.settings import settings_from_config

mp.mp.dps = 30
S = settings_from_config()
COS, SIN = [-1.0, 0.6], [-0.25, 0.0]
DOUBLE = p.potential_from_coefficients(COS, SIN)
SINGLE = p.potential_from_coefficients([-1.0, -0.2], [0.0, 0.0])  # one well: V = -cos s - 0.2 cos 2s


def _v(x):
    return COS[0] * mp.cos(x) + COS[1] * mp.cos(2 * x) + SIN[0] * mp.sin(x)


def _dv(x):
    return -COS[0] * mp.sin(x) - 2 * COS[1] * mp.sin(2 * x) + SIN[0] * mp.cos(x)


# the extrema of V by mpmath's root finder from rough guesses (deg), independent of `pendulum.extrema`
DEEP, TOP, SHALLOW, BARRIER = (mp.findroot(_dv, mp.radians(g)) for g in (68.0, 184.0, 298.0, 350.0))
V_MIN, V_MAX = _v(DEEP), _v(TOP)


def _energy(h):
    return V_MIN + mp.mpf(h) * (V_MAX - V_MIN)


def _action(energy, well):
    """J(E) = (1/pi) int sqrt(2 (E - V)) ds between the turning points of the loop of `energy` around `well`."""
    # the deep well (68 deg) lies between the barrier at -10.4 deg and the top at 184.2 deg, the shallow one (298.2 deg)
    # between the top and the barrier at 349.6 deg; the turning points are bracketed there (anderson's method)
    low, high = (BARRIER - 2 * mp.pi, TOP) if well == DEEP else (TOP, BARRIER)
    left = mp.findroot(lambda x: _v(x) - energy, (low, well), solver='anderson')
    right = mp.findroot(lambda x: _v(x) - energy, (well, high), solver='anderson')
    return mp.quad(lambda x: mp.sqrt(max(2 * (energy - _v(x)), 0)), [left, well, right]) / mp.pi


def _ratio(h, well):
    j_s = _action(_v(BARRIER) - mp.mpf('1e-25'), well)
    return float(_action(_energy(h), well) / j_s)


def test_the_double_well_found_by_mpmath_is_the_packages():
    found = p.extrema(DOUBLE)
    assert np.degrees(found.sigma_rad) == pytest.approx([float(mp.degrees(x)) for x in (DEEP, TOP, SHALLOW, BARRIER)], abs=1e-8)
    assert float((_v(SHALLOW) - V_MIN) / (V_MAX - V_MIN)) == pytest.approx(0.171407, abs=1e-6)
    assert float((_v(BARRIER) - V_MIN) / (V_MAX - V_MIN)) == pytest.approx(0.249494, abs=1e-6)


@pytest.mark.parametrize('m', [0.1, 0.5, 0.9])
def test_cosine_action_matches_the_closed_form(m):
    """V = -cos s: J = (8/pi)[E(k) - (1 - k^2) K(k)] with k^2 = m = h, J_s = 8/pi (mpmath quadrature and mpmath
    ellipe/ellipk, which take the parameter m as scipy does)."""
    amplitude = 2 * mp.asin(mp.sqrt(m))  # E = -cos A = 2m - 1
    j = mp.quad(lambda x: mp.sqrt(2 * (mp.cos(x) - mp.cos(amplitude))), [-amplitude, amplitude]) / mp.pi
    closed = 8 / mp.pi * (mp.ellipe(m) - (1 - m) * mp.ellipk(m))
    assert float(j) == pytest.approx(float(closed), rel=1e-20)
    assert p.action_ratio(m) == pytest.approx(float(closed / (8 / mp.pi)), abs=1e-13)
    cosine = p.potential_from_coefficients([-1.0], [0.0])
    assert p.separatrix_action(cosine) == pytest.approx(float(8 / mp.pi), rel=1e-15)


def test_two_harmonic_wells_differ_in_action_by_a_constant():
    """J_deep(E) - J_shallow(E) does not depend on E (mpmath), and equals J_deep at the shallow well's bottom."""
    offsets = [float(_action(_energy(h), DEEP) - _action(_energy(h), SHALLOW)) for h in (0.175, 0.2, 0.23, 0.249)]
    assert np.ptp(offsets) < 1e-12
    assert offsets[0] == pytest.approx(float(_action(_v(SHALLOW), DEEP)), rel=1e-10)


@pytest.mark.parametrize('h', [0.18, 0.2, 0.22, 0.245])
def test_action_ratio_in_the_shallow_well(h):
    shallow = p.occupied_well(DOUBLE, float(SHALLOW) + 0.05)
    assert shallow == pytest.approx(float(SHALLOW), abs=1e-10)
    assert p.action_ratio_in_potential(DOUBLE, h, well_rad=shallow) == pytest.approx(_ratio(h, SHALLOW), abs=1e-8)
    assert p.action_ratio_in_potential(DOUBLE, h) == pytest.approx(_ratio(h, DEEP), abs=1e-8)  # the default: the deepest
    step = 1e-5  # the slope in the shallow well against a finite difference of its own ratio
    numeric = (
        p.action_ratio_in_potential(DOUBLE, h + step, well_rad=shallow) - p.action_ratio_in_potential(DOUBLE, h - step, well_rad=shallow)
    ) / (2 * step)
    assert p.action_slope_in_potential(DOUBLE, h, well_rad=shallow) == pytest.approx(numeric, rel=1e-4)


def test_the_deepest_well_and_one_well_keep_their_numbers():
    assert p.occupied_well(DOUBLE, float(DEEP) - 0.3) is None
    assert p.occupied_well(SINGLE, 2.0) is None and p.occupied_well(p.potential_from_coefficients([-1.0], [0.0]), 2.0) is None


def _cycle_in(well_deg, h, potential=DOUBLE, cos_c=COS, sin_c=SIN):
    """One cycle integrated at energy h from the bottom of the well at `well_deg` (DOP853, rtol 1e-12)."""
    found = p.extrema(potential)
    minima = found.sigma_rad[found.is_min]
    well = float(minima[np.argmin(np.abs(np.degrees(minima) - well_deg))])
    energy = potential.v_min + h * potential.depth
    start = [well, math.sqrt(2 * (energy - float(potential.value(np.array([well]))[0])))]

    def rhs(_, y):
        n = len(cos_c)
        return [y[1], sum(k * (cos_c[k - 1] * math.sin(k * y[0]) - sin_c[k - 1] * math.cos(k * y[0])) for k in range(1, n + 1))]

    t = np.arange(0.0, 40.0, 0.001)
    angle, speed = solve_ivp(rhs, (0.0, t[-1]), start, t_eval=t, rtol=1e-12, atol=1e-14, method='DOP853').y
    tops = np.flatnonzero(np.diff(np.sign(speed)) < 0)
    (cycle,) = p.cycle_energies(t, angle, speed, [(float(t[tops[0]]), float(t[tops[1]]), False)], potential)
    return cycle


ZONE = Zone(ZoneMethod.NUMERIC, 0.5, np.nan, 1.0, 1.0, 1.0, False)  # J_lo = 0.5


def test_a_cycle_deep_in_the_shallow_well_is_deep():
    """At h = 0.18 the cycle sits low in its own well (J/J_s 0.096): deep. The deepest well's ratio (0.664, the code
    before W25) is above J_lo = 0.5 and called it zone."""
    cycle = _cycle_in(298.2, 0.18)
    assert cycle.well_rad == pytest.approx(float(SHALLOW), abs=1e-10)
    assert cycle.action_ratio == pytest.approx(_ratio(0.18, SHALLOW), abs=1e-5)
    assert cycle.period_law_ratio == pytest.approx(1.0, abs=5e-4)
    assert p.classify_cycles([cycle], ZONE, 0.001, S, DOUBLE) == [CycleClass.DEEP]
    assert _ratio(0.18, DEEP) == pytest.approx(0.6638, abs=1e-4)  # what the deepest well gave


def test_a_cycle_near_the_shallow_wells_separatrix_is_in_the_zone():
    cycle = _cycle_in(298.2, 0.245)
    assert cycle.well_rad == pytest.approx(float(SHALLOW), abs=1e-10)
    assert cycle.action_ratio == pytest.approx(_ratio(0.245, SHALLOW), abs=1e-4)
    assert cycle.action_ratio > 0.9
    assert p.classify_cycles([cycle], ZONE, 0.001, S, DOUBLE) == [CycleClass.ZONE]
    assert p.edge_cycles([cycle], ZONE, 0.001, S, DOUBLE) == [False]


def test_deep_well_and_one_well_cycles_are_unchanged():
    """A cycle of the deepest well and every cycle of a one-well V keep the deepest well's numbers exactly (no
    `well_rad`, the same call as before W25)."""
    deep = _cycle_in(68.0, 0.2)
    assert np.isnan(deep.well_rad)
    assert deep.action_ratio == p.action_ratio_in_potential(DOUBLE, deep.h)
    assert deep.action_ratio == pytest.approx(_ratio(0.2, DEEP), abs=1e-5)
    single = _cycle_in(0.0, 0.5, SINGLE, [-1.0, -0.2], [0.0, 0.0])
    assert np.isnan(single.well_rad)
    assert single.action_ratio == p.action_ratio_in_potential(SINGLE, single.h)
    assert single.action_ratio == pytest.approx(0.3841667556, abs=1e-6)
