"""W25 (Codex final review, finding 3): the averaged route reads J/<J_s>, not J/J_s(t).

The zone's lower edge J_lo is J0/<J_s>: the action over the separatrix action at the record's mean strength
(`zone.numeric_zone`: J/J_s(t0) = J0 / sqrt(m0)). The averaged route measured each turning point's J/J_s(t) in the
averaged profile at its own time, the units of that moment. The action of a libration at a fixed h scales as the square
root of the well's strength, so where the well is at a quarter of its mean strength J/J_s(t) = 0.6 is J/<J_s> = 0.3:
deep below J_lo = 0.5, while the route called it zone (and the pendulum route, J/J_s(h) sqrt(m), called it deep).

Independent numbers (not the package's quadratures): the cosine well V = -m cos s has J/J_s = E(k^2) - (1 - k^2) K(k^2)
with k^2 = sin^2(s_TP / 2) (Murray & Dermott 1999, eq. 8.58), solved here with scipy's ellipe/ellipk and brentq, and
J_s = 8 sqrt(m) / pi, checked by scipy.quad; a strength of 0.25 on [0, 2] yr and 2.25 on (2, 4] yr has the mean
sqrt(m) = (2 x 0.5 + 2 x 1.5) / 4 = 1.
"""

import dataclasses

import numpy as np
import pytest
from scipy import integrate, optimize
from scipy.special import ellipe, ellipk

from resonances.resonance.analysis import pendulum as p
from resonances.resonance.analysis.mmr import strength, zone
from resonances.resonance.analysis.pendulum import CycleClass, CycleEnergy, Zone, ZoneMethod
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.builders import make_episode, make_geometry, make_turning_points

S = dataclasses.replace(settings_from_config(), mmr_separatrix_avgpot_tp_max=40)
ZONE = Zone(ZoneMethod.NUMERIC, 0.5, np.nan, 1.0, 1.0, 1.0, False)  # J_lo = 0.5


def _cosine_ratio(k2):
    return ellipe(k2) - (1 - k2) * ellipk(k2)


def _tp_angle(ratio):
    """s_TP of the cosine loop with J/J_s = ratio (any strength)."""
    k2 = optimize.brentq(lambda x: _cosine_ratio(x) - ratio, 1e-12, 1 - 1e-12, xtol=1e-15)
    return 2 * np.arcsin(np.sqrt(k2))


class _Line:
    """<D> = -cos s on the grid s_m = 2 pi m / L (both quadratures the same: a converged profile)."""

    def __init__(self, nodes, fails):
        self.nodes_per_turn, self.fails = nodes, fails

    def sigma(self, m):
        return 2 * np.pi * np.asarray(m, dtype=float) / self.nodes_per_turn

    def evaluate(self, m):
        values = np.full(len(m), np.nan) if self.fails else -np.cos(self.sigma(m))
        return values, values.copy(), np.full(len(m), np.inf)


class _CosineWell:
    """The well along a record of strength m(t): V = m(t) x (-cos s), i.e. `scale` = m(t) on the profile -cos s. It
    stands for `centre_track.WellAlongRecord` (whose <D> comes from the planet's elements); everything the route does
    with it (the profile quadratures, J_s(t), the record mean, the conversion, the classes) is the package's own code."""

    available = True
    rule = strength.quadrature(S)

    def __init__(self, strength_of_t, fails_at=None):
        self.strength_of_t, self.fails_at = strength_of_t, fails_at

    def elements(self, time_yr):
        return float(time_yr), None, 0.0

    def scale(self, body):
        return self.strength_of_t(body)

    def line(self, time_yr, nodes_per_turn):
        return _Line(nodes_per_turn, self.fails_at is not None and self.fails_at(time_yr))


def _two_episodes(ratio_early, ratio_late):
    """Two accepted episodes, one cycle each: TPs at 0.5, 1.0, 1.5 yr at +-s_TP(ratio_early), at 2.5, 3.0, 3.5 yr at
    +-s_TP(ratio_late) (local J/J_s of the cosine), and their cycles."""
    early, late = _tp_angle(ratio_early), _tp_angle(ratio_late)
    times = np.array([0.5, 1.0, 1.5, 2.5, 3.0, 3.5])
    values = np.array([early, -early, early, late, -late, late])
    runs = [make_episode(tp_times_yr=times[:3], tp_values_rad=values[:3]), make_episode(tp_times_yr=times[3:], tp_values_rad=values[3:])]
    geometry = make_geometry(window=(0.0, 4.0), turning_points=make_turning_points(times, values), runs=runs)
    cycles = [CycleEnergy(0.5, 1.5, False, np.nan, np.nan, 1.0, 1.0), CycleEnergy(2.5, 3.5, False, np.nan, np.nan, 1.0, 1.0)]
    return geometry, cycles


def _step_strength(t):
    return 0.25 if t <= 2.0 else 2.25


def test_independent_numbers():
    """The separatrix action of -m cos s is 8 sqrt(m) / pi, so J_s(0.25) / J_s(1) = 0.5 (scipy.quad)."""

    def separatrix(m):
        return 2 * integrate.quad(lambda s: np.sqrt(2 * m * (1 + np.cos(s))), 0, np.pi)[0] / np.pi

    assert separatrix(1.0) == pytest.approx(8 / np.pi, rel=1e-10)
    assert separatrix(0.25) / separatrix(1.0) == pytest.approx(0.5, rel=1e-10)
    assert _cosine_ratio(np.sin(_tp_angle(0.6) / 2) ** 2) == pytest.approx(0.6, abs=1e-12)


def test_a_cycle_in_a_weak_phase_of_the_well_is_deep():
    """Reviewer's case: J/J_s(t) = 0.6 where m = 0.25 is J/<J_s> = 0.3 < J_lo 0.5, deep (it was zone); and the other
    way: J/J_s(t) = 0.4 where m = 2.25 is J/<J_s> = 0.6 >= 0.5, zone (it was deep)."""
    geometry, cycles = _two_episodes(0.6, 0.4)
    energies = zone.tp_energies(_CosineWell(_step_strength), geometry, cycles, S, (0.0, 4.0))
    assert energies.j == pytest.approx([0.3, 0.3, 0.3, 0.6, 0.6, 0.6], abs=1e-6)
    kinds = zone.cycle_kinds(geometry, cycles)
    assert kinds == [zone.CYCLE_EPISODE] * 2
    classes, edges, source = zone.averaged_classes(cycles, kinds, [CycleClass.ZONE, CycleClass.DEEP], [True, True], energies, ZONE)
    assert classes == [CycleClass.DEEP, CycleClass.ZONE] and edges == [False, False] and source == zone.SOURCE_AVERAGED_POTENTIAL


def test_the_pendulum_route_agrees():
    """The direct consumer reads the same J/<J_s> = J/J_s(h) sqrt(m): the first cycle is deep there too."""
    h = float(np.sin(_tp_angle(0.6) / 2) ** 2)
    cycle = CycleEnergy(0.5, 1.5, False, h, np.nan, 1.0, 1.0, strength_rel_sq=0.25)
    cosine = p.potential_from_coefficients([-1.0], [0.0])
    assert p.classify_cycles([cycle], ZONE, 0.001, S, cosine) == [CycleClass.DEEP]


def test_a_constant_depth_keeps_the_local_ratio():
    """The absolute scale cancels: in a well of constant strength 4, J/<J_s> = J/J_s(t)."""
    geometry, cycles = _two_episodes(0.6, 0.4)
    energies = zone.tp_energies(_CosineWell(lambda t: 4.0), geometry, cycles, S, (0.0, 4.0))
    assert energies.j == pytest.approx([0.6, 0.6, 0.6, 0.4, 0.4, 0.4], abs=1e-6)


def test_no_record_mean_keeps_the_pendulum_classes():
    """A profile that fails between the episodes (2.0-2.4 yr, no turning point there) leaves <J_s> unknown: no J/<J_s>,
    the pendulum's classes stay; h is still measured. Without a zone the mean is not computed at all."""
    geometry, cycles = _two_episodes(0.6, 0.4)
    well = _CosineWell(_step_strength, fails_at=lambda t: 2.0 < t < 2.4)
    energies = zone.tp_energies(well, geometry, cycles, S, (0.0, 4.0))
    assert np.all(np.isnan(energies.j)) and np.all(np.isfinite(energies.h))
    kinds = zone.cycle_kinds(geometry, cycles)
    pendulum_classes = [CycleClass.ZONE, CycleClass.DEEP]
    assert zone.averaged_classes(cycles, kinds, pendulum_classes, [True, False], energies, ZONE) == (
        pendulum_classes,
        [True, False],
        zone.SOURCE_PENDULUM,
    )
    without_zone = zone.tp_energies(_CosineWell(_step_strength), geometry, cycles, S)
    assert np.all(np.isnan(without_zone.j)) and np.all(np.isfinite(without_zone.h))
