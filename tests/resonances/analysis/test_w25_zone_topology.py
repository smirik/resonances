"""W25: two fixes of the MMR separatrix zone (docs/physics/pendulum.md, docs/physics/geometry.md "The separatrix zone").

1. Above the lower saddle of its well a libration loop encloses several wells: it has no single-well J/J_s. Before W25
   `action_ratio_in_potential` returned 1 there (and the slope inf), so such a cycle was called "in the zone". Now the
   ratio is nan, the cycle's topology is merged_wells, its class not_measured (it cannot make "every cycle in the
   zone"), and the connected loop's own action J_loop and dJ_loop/dE = T_loop/(2 pi) are reported.
2. The averaged route compared turning-point energies h with an h_lo mapped from J_lo through V_E's deepest well. Now it
   compares actions: J/J_s of each turning point's loop in the averaged profile itself against J_lo.

Independent numbers: mpmath quadratures written here (30 digits, mpmath's own root finder), the closed form
J/J_s = E(m) - (1 - m) K(m) of the cosine (M&D 1999 eq. 8.58), and the reviewers' own quadratures of 2026-10-28:
- V = -cos s + cos 2s: wells at +-acos(1/4) (V = -9/8), an internal saddle V(0) = 0, the outer top V(pi) = 2. At E = 1
  the loop through a well encloses both: turning points +-acos((1 - sqrt 17)/4) = +-2.466703808004,
  J_loop = 2.605103537913, T_loop = 6.890547807067 (Codex);
- V = -cos s + 0.5 cos 2s at E = E_b + 0.6 (E_top - E_b): J_loop / J_loop(E_top) = 0.670 (Claude);
- J/J_s = 0.5 at h = 0.580817655060 for V = -cos s and at h = 0.627780678948 for V = -cos s - 0.2 cos 2s (Codex): a
  turning point at h = 0.60 is in the zone of J_lo = 0.5 in the first profile and deep in the second.
"""

import math

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ellipe, ellipk

from resonances.resonance.analysis import pendulum as p
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import averaged_potential, decide, response, zone
from resonances.resonance.analysis.pendulum import CycleClass, CycleEnergy, Zone, ZoneMethod
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.builders import make_energy, make_episode

mp.mp.dps = 30
S = settings_from_config()
MERGED = p.potential_from_coefficients([-1.0, 1.0], [0.0, 0.0])  # V = -cos s + cos 2s
ZONE = Zone(ZoneMethod.NUMERIC, 0.5, np.nan, 1.0, 1.0, 1.0, False)  # J_lo = 0.5


def _h(potential, energy):
    return (energy - potential.v_min) / potential.depth


def _action(v, energy, lower, upper):
    """(1/pi) int sqrt(2 (E - V)) ds between two turning points (mpmath)."""
    return mp.quad(lambda x: mp.sqrt(max(2 * (energy - v(x)), 0)), [lower, upper]) / mp.pi


def _v_merged(x):
    return -mp.cos(x) + mp.cos(2 * x)


# --------------------------------------------------------------------------------------------- 1, topology


def test_topology_and_ratio_of_the_double_well():
    well = p.occupied_well(MERGED, 1.0)  # the well at +acos(1/4), the other one is the deepest by rounding
    for energy, topology in ((-0.5, p.TOPOLOGY_SINGLE_WELL), (-1e-9, p.TOPOLOGY_SINGLE_WELL), (1e-9, p.TOPOLOGY_MERGED_WELLS)):
        assert p.libration_topology(MERGED, _h(MERGED, energy), well_rad=well) == topology
    for energy in (1e-9, 1.0, 1.99):  # above the internal saddle, below the outer top
        h = _h(MERGED, energy)
        assert p.libration_topology(MERGED, h, well_rad=well) == p.TOPOLOGY_MERGED_WELLS
        assert np.isnan(p.action_ratio_in_potential(MERGED, h, well_rad=well))  # 1 before W25
        assert np.isnan(p.action_slope_in_potential(MERGED, h, well_rad=well))  # inf before W25
    assert p.libration_topology(MERGED, _h(MERGED, 2.5), well_rad=well) == p.TOPOLOGY_ROTATION
    # a V of one well keeps the separatrix convention above its top: J/J_s = 1, slope inf
    one = p.potential_from_coefficients([-1.0, -0.2], [0.0, 0.0])
    assert p.libration_topology(one, 1.05) == p.TOPOLOGY_SEPARATRIX
    assert p.action_ratio_in_potential(one, 1.05) == 1.0 and p.action_slope_in_potential(one, 1.05) == math.inf
    # below the internal saddle the ratio is that of the lobe: J(-0.5) / J(0) by mpmath, one lobe from 0 to 2 pi / 3
    lobe = mp.acos(mp.mpf(1) / 4)
    turning = [mp.findroot(lambda x: _v_merged(x) + mp.mpf('0.5'), guess) for guess in (0.6, 2.0)]
    expected = _action(_v_merged, mp.mpf('-0.5'), *turning) / _action(_v_merged, 0, 0, 2 * mp.pi / 3)
    assert lobe > turning[0] and p.action_ratio_in_potential(MERGED, _h(MERGED, -0.5), well_rad=well) == pytest.approx(
        float(expected), abs=1e-8
    )


def test_the_connected_loop_action_and_period():
    edge = mp.acos((1 - mp.sqrt(17)) / 4)
    assert float(edge) == pytest.approx(2.466703808004, abs=1e-12)
    expected = _action(_v_merged, 1, -edge, edge)
    assert float(expected) == pytest.approx(2.605103537913, abs=1e-11)  # the reviewer's quadrature
    loop = p.loop_through(MERGED, 1.0, math.acos(0.25))
    assert loop.lower_rad == pytest.approx(-float(edge), abs=1e-10) and loop.upper_rad == pytest.approx(float(edge), abs=1e-10)
    assert loop.period_yr == pytest.approx(6.890547807067, abs=1e-8)
    assert p.loop_action_through(MERGED, 1.0, math.acos(0.25)) == pytest.approx(float(expected), abs=1e-9)
    # the same loop from the other well, and V x 4: J x 2 (J ~ sqrt of the scale at a scaled energy)
    assert p.loop_action_through(MERGED, 1.0, -math.acos(0.25)) == pytest.approx(float(expected), abs=1e-9)
    scaled = p.potential_from_coefficients([-4.0, 4.0], [0.0, 0.0])
    assert p.loop_action_through(scaled, 4.0, math.acos(0.25)) == pytest.approx(2 * float(expected), abs=1e-8)
    assert np.isnan(p.loop_action_through(MERGED, 2.5, 0.0))  # above the top: no loop


def test_the_loop_action_slope_is_its_period_over_two_pi():
    """dJ_loop/dE by a centred difference of the mpmath action (its own turning points) = T_loop / (2 pi)."""

    def j(energy):
        edge = mp.acos((1 - mp.sqrt(1 + 8 * (1 + energy))) / 4)  # 2c^2 - c - 1 - E = 0 for V = E
        return _action(_v_merged, energy, -edge, edge)

    step = mp.mpf('1e-6')
    slope = float((j(1 + step) - j(1 - step)) / (2 * step))
    assert slope == pytest.approx(6.890547807067 / (2 * math.pi), rel=1e-8)
    assert p.loop_through(MERGED, 1.0, 1.0).period_yr / (2 * math.pi) == pytest.approx(slope, rel=1e-8)


def test_the_merged_loop_tends_to_the_sum_of_the_lobes():
    """Just above the internal saddle the connected loop is the two lobes of E = 0 together (each from 0 to 2 pi / 3)."""
    lobes = 2 * _action(_v_merged, 0, 0, 2 * mp.pi / 3)
    assert p.loop_action_through(MERGED, 1e-10, 1.0) == pytest.approx(float(lobes), abs=1e-4)


def test_the_loop_under_the_outer_separatrix_of_the_second_profile():
    """V = -cos s + 0.5 cos 2s: E_b = V(0) = -1/2, E_top = V(pi) = 3/2; at E = E_b + 0.6 (E_top - E_b) = 0.7 the loop's
    action over that of the outer separatrix is 0.670 (the reviewer's quadrature; mpmath here)."""
    second = p.potential_from_coefficients([-1.0, 0.5], [0.0, 0.0])

    def v(x):
        return -mp.cos(x) + mp.cos(2 * x) / 2

    def j(energy):
        edge = mp.acos((1 - mp.sqrt(1 + 4 * (0.5 + energy))) / 2)  # c^2 - c - 1/2 - E = 0 for V = E
        return _action(v, energy, -edge, edge)

    expected = j(mp.mpf('0.7')) / j(mp.mpf('1.5'))
    assert float(expected) == pytest.approx(0.670, abs=5e-4)
    ratio = p.loop_action_through(second, 0.7, 1.0) / p.loop_action_through(second, 1.5 - 1e-12, 1.0)
    assert ratio == pytest.approx(float(expected), abs=1e-5)


def _integrated_cycle(potential, cos_c, energy, start):
    """One cycle of sigma_ddot = -V'(sigma) integrated at `energy` from `start` (DOP853, rtol 1e-12)."""
    speed0 = math.sqrt(2 * (energy - float(potential.value(np.array([start]))[0])))

    def rhs(_, y):
        return [y[1], sum(k * cos_c[k - 1] * math.sin(k * y[0]) for k in range(1, len(cos_c) + 1))]

    t = np.arange(0.0, 40.0, 0.001)
    angle, speed = solve_ivp(rhs, (0.0, t[-1]), [start, speed0], t_eval=t, rtol=1e-12, atol=1e-14, method='DOP853').y
    tops = np.flatnonzero(np.diff(np.sign(speed)) < 0)
    (cycle,) = p.cycle_energies(t, angle, speed, [(float(t[tops[0]]), float(t[tops[1]]), False)], potential)
    return cycle


def test_a_merged_cycle_has_no_single_well_ratio_and_is_not_measured():
    cycle = _integrated_cycle(MERGED, [-1.0, 1.0], 1.0, math.acos(0.25))
    assert cycle.topology == p.TOPOLOGY_MERGED_WELLS
    assert np.isnan(cycle.action_ratio)
    assert cycle.loop_action == pytest.approx(2.605103537913, rel=1e-4)
    assert cycle.loop_action_slope == pytest.approx(6.890547807067 / (2 * math.pi), rel=1e-3)
    assert cycle.period_law_ratio == pytest.approx(1.0, abs=1e-3)
    assert p.classify_cycles([cycle], ZONE, 0.001, S, MERGED) == [CycleClass.NOT_MEASURED]
    assert p.edge_cycles([cycle], ZONE, 0.001, S, MERGED) == [False]
    # a lobe below the saddle: its own ratio, J_loop = J (mpmath above: E = -0.5, from 0.6 to 2.0 rad)
    lobe = _integrated_cycle(MERGED, [-1.0, 1.0], -0.5, math.acos(0.25))
    turning = [mp.findroot(lambda x: _v_merged(x) + mp.mpf('0.5'), guess) for guess in (0.6, 2.0)]
    assert lobe.topology == p.TOPOLOGY_SINGLE_WELL
    assert lobe.loop_action == pytest.approx(float(_action(_v_merged, mp.mpf('-0.5'), *turning)), rel=1e-4)


def test_the_fallback_never_regenerates_a_clipped_ratio():
    """A cycle without a stored ratio is recomputed topology-aware (nan above the lower saddle); a stored topology
    merged_wells wins over a stale ratio. A V of one well keeps J/J_s = 1 above its top (zone)."""
    h = _h(MERGED, 1.0)
    unknown = CycleEnergy(0.0, 1.0, False, h, np.nan, 1.0, 1.0)
    stale = CycleEnergy(0.0, 1.0, False, h, 1.0, 1.0, 1.0, topology=p.TOPOLOGY_MERGED_WELLS)
    assert p.classify_cycles([unknown, stale], ZONE, 0.001, S, MERGED) == [CycleClass.NOT_MEASURED] * 2
    one = p.potential_from_coefficients([-1.0, -0.2], [0.0, 0.0])
    assert p.classify_cycles([CycleEnergy(0.0, 1.0, False, 1.05, np.nan, 1.0, 1.0)], ZONE, 0.001, S, one) == [CycleClass.ZONE]


def test_an_unmeasured_cycle_is_neither_in_the_zone_nor_outside():
    classes = [CycleClass.ZONE, CycleClass.NOT_MEASURED]
    assert not zone.episodes_in_zone([100.0], classes)
    assert zone.episodes_in_zone([100.0], [CycleClass.ZONE, CycleClass.ZONE])
    assert not zone.episodes_in_zone([], [CycleClass.ZONE])
    energy = make_energy(cycle_classes=[CycleClass.NOT_MEASURED, CycleClass.DEEP, CycleClass.ABOVE])
    assert energy.n_outside == 2 and energy.n_not_measured == 1 and not p.has_zone_cycle(energy.cycle_classes)
    assert response._episode_zone_class([CycleClass.DEEP, CycleClass.NOT_MEASURED]) == 'not_measured'
    assert response._episode_zone_class([CycleClass.DEEP, CycleClass.NOT_MEASURED, CycleClass.ZONE]) == 'zone'


def test_columns_of_a_merged_cycle():
    cycle = CycleEnergy(
        0.0, 1.0, False, 0.68, np.nan, 1.0, 1.0, topology=p.TOPOLOGY_MERGED_WELLS, loop_action=2.6051, loop_action_slope=1.0967
    )
    columns = make_energy(cycles=[cycle], cycle_classes=[CycleClass.NOT_MEASURED]).to_columns()
    assert columns['sep_cycle_topology'] == 'merged_wells' and columns['sep_n_not_measured'] == 1
    assert columns['sep_cycle_J'] == 'nan' and columns['sep_cycle_J_loop_rad2_yr'] == '2.605'
    assert columns['sep_cycle_dJdE_loop_yr'] == '1.097' and columns['sep_n_outside'] == 0


@pytest.mark.parametrize(
    'classes, expected',
    [
        ([CycleClass.NOT_MEASURED, CycleClass.DEEP], Margin.NOT_MEASURED),
        ([CycleClass.NOT_MEASURED, CycleClass.ZONE], Margin.THIN),
        ([CycleClass.DEEP, CycleClass.DEEP], Margin.FULL),
    ],
)
def test_l2_reads_an_unmeasured_cycle_as_not_measured(classes, expected):
    """L2 with the zone known: thin with a cycle in the zone, full with every cycle deep (before W25 a merged loop was
    'zone', so thin), not_measured with an unmeasured cycle and none in the zone."""
    episode = make_episode(half_amp_max_rad=math.radians(60.0))
    energy = make_energy(zone=ZONE, cycle_classes=classes)
    margin = decide._l2(episode, energy, Ledger(thin_band=0.1), S)
    assert margin.margin == expected and margin.value == pytest.approx(60.0)


# --------------------------------------------------------------------------------------------- 2, the averaged route

GRID = 2 * np.pi * np.arange(360) / 360


def _point(v, energy, lo, hi):
    """sigma in (lo, hi) where v = energy (mpmath bisection-free root by its own solver)."""
    return float(mp.findroot(lambda x: v(x) - energy, (lo, hi), solver='anderson'))


def _cosine(x):
    return -mp.cos(x)


def _sharp(x):
    return -mp.cos(x) - mp.cos(2 * x) / 5


def _ratio_at_h(v, h):
    """J/J_s at h of a one-well profile with its bottom at 0 and its top at pi (mpmath)."""
    low, top = v(0), v(mp.pi)
    energy = low + mp.mpf(h) * (top - low)
    edge = mp.findroot(lambda x: v(x) - energy, (mp.mpf('0.01'), mp.pi - mp.mpf('0.01')), solver='anderson')
    return float(_action(v, energy, -edge, edge) / _action(v, top, -mp.pi, mp.pi))


def test_the_cosine_ratio_is_the_closed_form():
    assert _ratio_at_h(_cosine, 0.6) == pytest.approx(float(ellipe(0.6) - 0.4 * ellipk(0.6)), abs=1e-12)


@pytest.mark.parametrize('v, h_at_half', [(_cosine, 0.580817655060), (_sharp, 0.627780678948)])
def test_profile_action_compares_actions_not_energies(v, h_at_half):
    """The same h = 0.60 straddles J_lo = 0.5: zone in the cosine (J/J_s 0.5186), deep in the sharper profile (0.4739).
    The h of J/J_s = 0.5 differs between the two by 0.047, which no barrier normalisation of h can give."""
    values = np.array([float(v(x)) for x in GRID])
    low, top = float(v(0)), float(v(mp.pi))
    for h in (0.6, h_at_half):
        point = _point(v, low + h * (top - low), 0.01, math.pi - 0.01)
        assert averaged_potential.energy_in_well(GRID, values, point, 0.0) == pytest.approx(h, abs=1e-7)
        assert averaged_potential.action_in_well(GRID, values, point, 0.0) == pytest.approx(_ratio_at_h(v, h), abs=2e-7)
        # scale-free, and the other side of the swing gives the same loop
        assert averaged_potential.action_in_well(GRID, 7 * values + 3, -point, 0.0) == pytest.approx(_ratio_at_h(v, h), abs=2e-7)
    assert _ratio_at_h(v, h_at_half) == pytest.approx(0.5, abs=1e-10)
    assert (_ratio_at_h(v, 0.6) >= 0.5) == (v is _cosine)


# V = -cos s + 0.6 cos 2s - 0.25 sin s (tests/resonances/analysis/test_action_occupied_well.py): wells at 68.0 deg
# (deepest) and 298.2 deg (shallow), maxima at 349.6 deg (the barrier of both) and 184.2 deg (the top)


def _double(x):
    return -mp.cos(x) + mp.mpf('0.6') * mp.cos(2 * x) - mp.mpf('0.25') * mp.sin(x)


def _d_double(x):
    return mp.sin(x) - mp.mpf('1.2') * mp.sin(2 * x) - mp.mpf('0.25') * mp.cos(x)


def test_profile_action_in_the_shallow_well_and_above_its_barrier():
    deep, top, shallow, barrier = (mp.findroot(_d_double, mp.radians(g)) for g in (68.0, 184.0, 298.0, 350.0))
    values = np.array([float(_double(x)) for x in GRID])
    energy = _double(shallow) + (_double(barrier) - _double(shallow)) * mp.mpf('0.4')
    left = mp.findroot(lambda x: _double(x) - energy, (top, shallow), solver='anderson')
    right = mp.findroot(lambda x: _double(x) - energy, (shallow, barrier), solver='anderson')
    # the shallow well's separatrix loop: from the root of V = V(barrier) on the top's side to the barrier itself
    left_s = mp.findroot(lambda x: _double(x) - _double(barrier), (top, shallow), solver='anderson')
    j_s = _action(_double, _double(barrier), left_s, barrier)
    expected = float(_action(_double, energy, left, right) / j_s)
    got = averaged_potential.action_in_well(GRID, values, float(right), float(shallow))
    assert got == pytest.approx(expected, abs=1e-6)
    assert averaged_potential.energy_in_well(GRID, values, float(right), float(shallow)) == pytest.approx(0.4, abs=1e-6)
    # above the barrier (0.249 of the global depth) a turning point's loop encloses both wells: no single-well ratio
    above = _double(deep) + (_double(top) - _double(deep)) * mp.mpf('0.4')
    beyond = mp.findroot(lambda x: _double(x) - above, (deep + 2 * mp.pi, top + 2 * mp.pi), solver='anderson')  # past the barrier
    assert np.isnan(averaged_potential.action_in_well(GRID, values, float(beyond), float(shallow)))
    # in a profile of one well the top itself is its separatrix: 1
    cosine = -np.cos(GRID)
    assert averaged_potential.action_in_well(GRID, cosine, math.pi, 0.0) == 1.0
    assert averaged_potential.action_in_well(GRID, cosine, 0.0, 0.0) == 0.0


def test_averaged_classes_read_actions():
    """The two profiles at h = 0.60 (J/J_s 0.5186 and 0.4739, mpmath above): zone and deep against J_lo = 0.5, whatever
    h says (both turning points are at h = 0.60 here; before W25 h decided)."""
    j_cosine, j_sharp = _ratio_at_h(_cosine, 0.6), _ratio_at_h(_sharp, 0.6)
    cycles = [CycleEnergy(0.0, 2.0, False, 0.6, 0.3, 2.0, 1.0), CycleEnergy(2.0, 4.0, False, 0.6, 0.3, 2.0, 1.0)]
    kinds = [zone.CYCLE_EPISODE] * 2
    energies = zone.TurningPointEnergies(np.array([0.0, 1.0, 2.5, 3.5]), np.full(4, 0.6), np.array([j_cosine, j_cosine, j_sharp, j_sharp]))
    classes, edges, source = zone.averaged_classes(cycles, kinds, [CycleClass.DEEP, CycleClass.ZONE], [True, True], energies, ZONE)
    assert classes == [CycleClass.ZONE, CycleClass.DEEP] and edges == [False, False] and source == zone.SOURCE_AVERAGED_POTENTIAL
    # a turning point whose loop encloses another well (nan) and none at J_lo: the pendulum's class stays
    unmatched = zone.TurningPointEnergies(np.array([0.0, 1.0]), np.full(2, 1.2), np.array([np.nan, j_sharp]))
    kept = zone.averaged_classes(cycles[:1], kinds[:1], [CycleClass.NOT_MEASURED], [False], unmatched, ZONE)
    assert kept == ([CycleClass.NOT_MEASURED], [False], zone.SOURCE_PENDULUM)
