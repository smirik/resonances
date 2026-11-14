"""Co-orbital (1:1) geometry read from the averaged potential: where a turn happens, how far the potential reaches,
how long the loop through observed turning points takes, which well holds a cycle, and the local law on a window.

The potential is the one of the 1:1 force law (`coorbital.py`): the direct + indirect disturbing function averaged over
the planet's longitude at fixed theta = lambda - lambda_P, R~ = <R> a_P / (G m_P), tabulated by
`coorbital.table_for` (the record's cache and its `record_tables` budget; no retabulation of its own). The planet's
saved elements are used when the record has them, else its mean J2000 orbit (`coorbital.planet_orbits`): no planet
longitude is needed, theta comes from the angle itself (`theta_of`).

Theory
------
The energy integral of `coorbital.py` (eq. 2), (1/2) theta_dot^2 + 3 mu n^2 R~(theta) = E, with
theta_dot = -(3/2) n (a - a_P) / a_P, reads

    (da / a_P)^2 + (8/3) mu R~(theta) = const,        da = a - a_P,                                         (1)

M&D eqs. 3.178-3.179 (p. 105) for e = i = 0, where R~ = 1 / (2 |sin(theta/2)|) - cos theta = H / 2 + 1. A turning
point (da = 0) at theta_t fixes the constant: the loop through it has |da(theta)| = a_P sqrt((8/3) mu (R~(theta_t) -
R~(theta))), at L3 the closed form sqrt((4/3) mu (H(theta_t) - H(pi))) of a horseshoe. The potential
reverses a body that moves towards theta_t only while its level stays below the highest R~ ahead, before the body
reaches the planet (theta = 0) or L3 (pi): `da_max` is the largest |da| at theta that it still reverses there (the
W28 C1 study measured |da| / da_max <= 1.04 on 95 % of the turns of confirmed cycles). A reversal by the potential
mirrors |da| (time reversal of (1)); `turn_conservation` measures out/in = |da| after / |da| before (1 for a
reflection, the C1 q90 of confirmed reflections is |ln out/in| = 0.33).

The period of the loop through observed turning points is the period integral of (1),
T = 2 int dtheta / sqrt(2 (E - V)) with V = 3 mu n^2 R~ (`averaged_potential.orbit_in_profile`): the horseshoe period
T_hs(theta_t), a tadpole or a quasi-satellite at the amplitude of its turning points, the linear tadpole period
P / sqrt(27 mu / 4) for a small swing about L4/L5 (M&D eq. 3.151 at small mu; Earth 221 yr, Venus 151 yr); inf at a
separatrix.

Sites and wells. The extremes of R~ are found on the table grid and refined on its periodic cubic spline. A minimum
is the quasi-satellite well when its basin holds theta = 0 nearer its bottom than its wall (`coorbital.
quasi_satellite_well`: 617 Patroclus' L4 basin reaches past 0 and is no such well); the other minima are L4 (theta in
(0, 180) deg) and L5. The planet's tops are the two that bound the quasi-satellite basin, or, without one, the top
nearest theta = 0; every other top is an L3 top. A turning point below the barrier of its basin (the lower of its two
tops) is a turn inside that well (tadpole_L4, tadpole_L5, qs); above it, the turn is a reflection on the top uphill
of it (planet: a horseshoe or compound turn; l3). A cycle's well (`well_of`) is named by the minima its swing holds:
one well, both tadpole wells (horseshoe), the quasi-satellite well with a tadpole well (compound).

Pure: arrays and dataclasses in, no status; the models the geometry calls (`CoorbitalRunFloor`, W28 C3;
`CoorbitalEdgeTurn`, W28 C4) read their settings. The circular closed forms are exact; at high e the sites and periods
are those of the averaged problem (smoke-tested only, the risk named in W28 C2).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize
from scipy.interpolate import CubicSpline

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.mmr import coorbital as coorbital_module
from resonances.resonance.analysis.mmr.averaged_potential import orbit_in_profile
from resonances.resonance.analysis.mmr.coorbital import GM_SUN, CoorbitalInput, CoorbitalLaw, CoorbitalTable, Orbit
from resonances.resonance.analysis.settings import AnalysisSettings

NAN = float('nan')
TWO_PI = 2 * np.pi
ENERGY_FACTOR = 8.0 / 3.0  # (da / a_P)^2 + (8/3) mu R~ = const (eq. 1 of the module docstring)
LEVEL_RTOL = 1e-9  # numerical: a level within this fraction of the profile's range of a top is at that top
ARC_SAMPLES_PER_NODE = 4  # numerical: the arc of `da_max` is sampled at a quarter of the table step
# numerical: `loop_period` integrates V sampled at a quarter of the table step from `CoorbitalProfile.potential`: the
# spline of orbit_in_profile through the table nodes themselves rings from a one-node peak (0.9 % on T_hs at 2.9 deg)
PERIOD_SAMPLES_PER_NODE = 4
MIN_SIDE_SAMPLES = 3  # a median of |da| on a side of a turn needs a few samples (numerical, not a threshold)
NO_LOOP_PENALTY = 1e30  # numerical: the minimiser's stand-in for 'no horseshoe loop there' (inf breaks Brent)
HORSESHOE_SCAN_TURNS = 16  # numerical: turns per side `fastest_horseshoe_period` scans before refining its best

SITE_PLANET, SITE_L3 = 'planet', 'l3'  # a reflection on the planet's top (horseshoe/compound) or on an L3 top
SITE_QS, SITE_L4, SITE_L5 = 'qs', 'tadpole_L4', 'tadpole_L5'  # a turn inside that well (also the wells' names)
WELL_HORSESHOE, WELL_COMPOUND, SITE_NONE = 'horseshoe', 'compound', ''

LEG_COLUMNS = (
    'cpl_turn_site_entry',
    'cpl_turn_site_exit',
    'cpl_turn_da_ratio_entry',
    'cpl_turn_da_ratio_exit',
    'cpl_turn_out_in_entry',
    'cpl_turn_out_in_exit',
    'cpl_loop_period_yr',
)
LEG_TEXT_COLUMNS = ('cpl_turn_site_entry', 'cpl_turn_site_exit')


def theta_of(angle: np.ndarray, body_coefficient: int) -> np.ndarray:
    """theta = lambda - lambda_P from a 1:1 angle sigma = k (lambda - lambda_P), k = +-1 (`coorbital.CoorbitalLaw.
    angle_sign`; the benchmark's saved `angle_raw` is lambda_P - lambda, k = -1)."""
    return (1 if body_coefficient > 0 else -1) * np.asarray(angle, dtype=float)


@dataclass(frozen=True)
class Extreme:
    """A minimum or a top of R~(theta): its place [rad, in [0, 2 pi)], value and name (SITE_*)."""

    theta_rad: float
    value: float
    is_top: bool
    name: str


@dataclass(frozen=True)
class CoorbitalProfile:
    """R~(theta) of one set of mean elements (`profile_at`) with the planet's mass mu and mean motion n [rad/yr]."""

    table: CoorbitalTable
    planet_mass: float
    planet_mean_motion_rad_yr: float

    @property
    def a_planet_au(self) -> float:
        """The planet's semi-major axis from its mean motion (n^2 a^3 = G M_sun), the exact 1:1 axis [au]."""
        return float((GM_SUN / self.planet_mean_motion_rad_yr**2) ** (1.0 / 3.0))

    @property
    def scale(self) -> float:
        """V = scale x R~: 3 mu n^2 [rad^2/yr^2] (`coorbital.py` eq. 2)."""
        return 3.0 * self.planet_mass * self.planet_mean_motion_rad_yr**2

    @cached_property
    def _reciprocal(self) -> Tuple[CubicSpline, float]:
        """(the periodic cubic spline of w = 1 / (R~ - R~_min + 1) through the table, R~_min - 1). The planet's softened
        peak of a near-circular coplanar orbit is one grid node wide; a spline of R~ itself rings from it (1.2 % at 2.9
        deg in the circular problem, 0.9 % on the horseshoe period), while w ~ |theta| there is smooth (exact at the
        nodes, 1e-5 between them at 2.9 deg) and as smooth as R~ elsewhere."""
        values = np.append(self.table.potential, self.table.potential[0])
        offset = float(np.min(values)) - 1.0
        theta = np.append(self.table.theta_rad, TWO_PI)
        return CubicSpline(theta, 1.0 / (values - offset), bc_type='periodic'), offset

    def potential(self, theta_rad) -> np.ndarray:
        """R~ at any theta [rad], from the spline of its reciprocal through the table (`_reciprocal`)."""
        spline, offset = self._reciprocal
        return 1.0 / spline(np.mod(np.asarray(theta_rad, dtype=float), TWO_PI)) + offset

    @cached_property
    def extremes(self) -> Tuple[Extreme, ...]:
        """The minima and tops of R~ in increasing theta, alternating, named (module docstring); () for a flat table."""
        return _extremes(self)

    @cached_property
    def value_range(self) -> float:
        return float(np.ptp(self.table.potential))


def _refine(profile: CoorbitalProfile, index: int, top: bool) -> Tuple[float, float]:
    """(theta in [0, 2 pi), R~) of the extreme of the spline within one table step of the grid node `index`."""
    grid = profile.table.theta_rad
    step = TWO_PI / len(grid)
    sign = -1.0 if top else 1.0
    centre = float(grid[index])
    found = optimize.minimize_scalar(
        lambda x: sign * float(profile.potential(x)), bounds=(centre - step, centre + step), method='bounded', options={'xatol': 1e-12}
    )
    theta = float(found.x) if found.success else centre
    return float(np.mod(theta, TWO_PI)), float(profile.potential(theta))


def _extremes(profile: CoorbitalProfile) -> Tuple[Extreme, ...]:
    values = np.asarray(profile.table.potential, dtype=float)
    if len(values) < 4 or not np.all(np.isfinite(values)):
        return ()
    following, preceding = np.roll(values, -1), np.roll(values, 1)
    minima = np.flatnonzero((values < following) & (values <= preceding))
    tops = np.flatnonzero((values > following) & (values >= preceding))
    if not len(minima) or not len(tops):
        return ()
    raw = sorted([(*_refine(profile, int(i), False), False) for i in minima] + [(*_refine(profile, int(i), True), True) for i in tops])
    n = len(raw)
    qs = -1
    for k, (theta, _, is_top) in enumerate(raw):
        if is_top:
            continue
        left, right = raw[(k - 1) % n][0], raw[(k + 1) % n][0]
        ahead = float(np.mod(-theta, TWO_PI))  # 0 is this far ahead of the bottom, and TWO_PI - ahead behind it
        behind = TWO_PI - ahead
        wall_ahead, wall_behind = float(np.mod(right - theta, TWO_PI)), float(np.mod(theta - left, TWO_PI))
        if ahead < wall_ahead - ahead or behind < wall_behind - behind:  # nearer the bottom than the wall
            qs = k
            break
    if qs >= 0:
        planet_tops = {(qs - 1) % n, (qs + 1) % n}
    else:
        top_indices = [k for k in range(n) if raw[k][2]]
        planet_tops = {min(top_indices, key=lambda k: abs(np.angle(np.exp(1j * raw[k][0]))))}
    out = []
    for k, (theta, value, is_top) in enumerate(raw):
        if is_top:
            name = SITE_PLANET if k in planet_tops else SITE_L3
        else:
            name = SITE_QS if k == qs else (SITE_L4 if theta < np.pi else SITE_L5)
        out.append(Extreme(theta, value, is_top, name))
    return tuple(out)


def _bracket(profile: CoorbitalProfile, theta_rad: float) -> Optional[Tuple[int, int]]:
    """(index of the extreme at or before theta, index of the one after) in `profile.extremes` (cyclic)."""
    extremes = profile.extremes
    if not extremes:
        return None
    places = np.array([x.theta_rad for x in extremes])
    after = int(np.searchsorted(places, float(np.mod(theta_rad, TWO_PI)), side='right'))
    return (after - 1) % len(extremes), after % len(extremes)


def _barrier(profile: CoorbitalProfile, bottom: int) -> float:
    """R~ of the lower of the two tops that bound the basin of the minimum `bottom` (an index of `extremes`)."""
    extremes = profile.extremes
    n = len(extremes)
    return min(extremes[(bottom - 1) % n].value, extremes[(bottom + 1) % n].value)


def turn_site(profile: CoorbitalProfile, theta_turn_rad: float) -> str:
    """Where a turning point at theta_turn lies (module docstring): inside a well below its barrier (SITE_QS,
    SITE_L4, SITE_L5), or a reflection on the top uphill of it (SITE_PLANET, SITE_L3); SITE_NONE for a flat profile.
    In the circular problem 60 / 300 deg are tadpole turns, 180 deg is L3, 3 deg is a horseshoe turn at the planet."""
    pair = _bracket(profile, theta_turn_rad)
    if pair is None:
        return SITE_NONE
    extremes = profile.extremes
    first, second = pair
    bottom, top = (second, first) if extremes[first].is_top else (first, second)
    level = float(profile.potential(theta_turn_rad))
    if level < _barrier(profile, bottom) - LEVEL_RTOL * profile.value_range:
        return extremes[bottom].name
    return extremes[top].name


@dataclass(frozen=True)
class Well:
    """The well of a cycle (`well_of`): its name (SITE_QS, SITE_L4, SITE_L5, WELL_HORSESHOE, WELL_COMPOUND or
    SITE_NONE) and the minima of R~ its swing holds [rad, in [0, 2 pi)]."""

    name: str
    minima_rad: Tuple[float, ...]


def well_of(profile: CoorbitalProfile, centre_rad: float, half_amplitude_rad: float = 0.0) -> Well:
    """The well that holds a cycle of centre `centre_rad` and half-amplitude `half_amplitude_rad` (theta, rad): named
    by the minima inside the swing [centre - A, centre + A] (module docstring). With A = 0 the basin whose slope holds
    the centre (steepest descent). SITE_NONE when the swing holds no minimum or covers a whole turn (no libration of
    this potential has it) or the profile is flat."""
    extremes = profile.extremes
    minima = [x for x in extremes if not x.is_top]
    if not minima or not half_amplitude_rad < np.pi:
        return Well(SITE_NONE, ())
    if not half_amplitude_rad > 0:
        first, second = _bracket(profile, centre_rad)
        bottom = extremes[second] if extremes[first].is_top else extremes[first]
        return Well(bottom.name, (bottom.theta_rad,))
    lo, hi = centre_rad - half_amplitude_rad, centre_rad + half_amplitude_rad
    inside = [x for x in minima if np.floor((hi - x.theta_rad) / TWO_PI) >= np.ceil((lo - x.theta_rad) / TWO_PI)]
    names = {x.name for x in inside}
    if not names:
        return Well(SITE_NONE, ())
    if len(names) == 1:
        name = names.pop()
    elif SITE_QS in names:
        name = WELL_COMPOUND
    else:
        name = WELL_HORSESHOE
    return Well(name, tuple(x.theta_rad for x in inside))


def da_max(profile: CoorbitalProfile, theta_turn_rad: float, theta_rad) -> np.ndarray:
    """The largest |a - a_P| [au] at `theta_rad` (unwrapped, on the same branch as theta_turn) that the averaged
    potential still reverses while the body moves towards the turn: a_P sqrt((8/3) mu (R~_top - R~(theta))), R~_top
    the highest R~ on the arc from theta through theta_turn to the first of {planet (0), L3 (pi)} beyond it (the
    definition of the W28 C1 study, whose ratio |da| / da_max the C6 zone tolerance is calibrated on). nan at
    theta = theta_turn (no direction)."""
    theta = np.atleast_1d(np.asarray(theta_rad, dtype=float))
    out = np.full(theta.shape, NAN)
    turn = float(theta_turn_rad)
    for direction in (1.0, -1.0):
        side = direction * (turn - theta) > 0
        if not np.any(side):
            continue
        beyond = np.pi * (np.floor(turn / np.pi) + 1) if direction > 0 else np.pi * (np.ceil(turn / np.pi) - 1)
        start = float(np.min(theta[side])) if direction > 0 else float(np.max(theta[side]))
        step = TWO_PI / len(profile.table.theta_rad) / ARC_SAMPLES_PER_NODE
        arc = np.linspace(start, beyond, max(2, int(np.ceil(abs(beyond - start) / step)) + 1))  # from the far side to beyond
        tops = [
            x.theta_rad + TWO_PI * k
            for x in profile.extremes
            if x.is_top
            for k in range(int(np.floor(min(start, beyond) / TWO_PI)), int(np.ceil(max(start, beyond) / TWO_PI)) + 1)
        ]
        tops = [x for x in tops if min(start, beyond) < x < max(start, beyond)]  # a sharp top between the samples
        arc = np.sort(np.concatenate([arc, tops]))[:: int(direction)]
        values = profile.potential(arc)
        ahead_max = np.maximum.accumulate(values[::-1])[::-1]  # the highest R~ from each arc point to `beyond`
        top = np.interp(direction * theta[side], direction * arc, ahead_max)
        top = np.maximum(top, profile.potential(theta[side]))  # an arc point between samples never lowers it
        out[side] = profile.a_planet_au * np.sqrt(ENERGY_FACTOR * profile.planet_mass * (top - profile.potential(theta[side])))
    return out if np.ndim(theta_rad) else out[0]


def loop_period(profile: CoorbitalProfile, turns_rad: Sequence[float]) -> float:
    """The period [yr] of the loop of the averaged potential through the observed turning points `turns_rad` (theta,
    unwrapped: the swing runs from the lower to the upper value): one turn, the orbit at rest there; two, the orbit of
    their mean level started at the lower one (`averaged_potential.orbit_in_profile`). inf at or above a separatrix
    (no turn ahead); nan when no loop of this potential joins the two turns (a top higher than both between them:
    2022 AP1's ripple at 174/212 deg), or at an extremum."""
    turns = [float(x) for x in turns_rad]
    if not 1 <= len(turns) <= 2 or not np.all(np.isfinite(turns)):
        return NAN
    levels = [float(profile.potential(x)) for x in turns]
    if len(turns) == 2:
        lo, hi = sorted(turns)
        inside = profile.potential(np.linspace(lo, hi, max(3, int(np.ceil((hi - lo) / (TWO_PI / len(profile.table.theta_rad)))) + 1)))
        if float(np.max(inside)) > max(levels) + LEVEL_RTOL * profile.value_range:
            return NAN
    energy = float(np.mean(levels))
    tolerance = LEVEL_RTOL * profile.value_range
    if any(x.is_top and abs(x.value - energy) <= tolerance for x in profile.extremes):
        return np.inf  # the loop of a top's level: the separatrix
    start = turns[int(np.argmin(levels))]
    nodes = len(profile.table.theta_rad) * PERIOD_SAMPLES_PER_NODE
    grid = TWO_PI * (np.floor(start / TWO_PI) - 1 + np.arange(3 * nodes) / nodes)  # three turns, the start in the middle
    kinetic = profile.scale * max(energy - min(levels), 0.0)
    period, _, _ = orbit_in_profile(grid, profile.scale * profile.potential(grid), start, kinetic)
    return float(period)


def fastest_horseshoe_period(profile: CoorbitalProfile, theta_min_rad: float) -> float:
    """The shortest period [yr] of a loop that reflects on the planet's top (`turn_site` SITE_PLANET: a horseshoe or
    compound turn) at a turn at least `theta_min_rad` from the planet (theta = 0), on either side: the minimum of the
    single-turn `loop_period` over turns from theta_min to L3 (pi). In the circular problem T_hs falls as the turn nears
    the planet, so the minimum is at theta_min (Earth at 3 r_H = 1.72 deg: 113 yr); a top of finite height (an
    eccentric orbit) makes T_hs grow again to the separatrix at its level, the minimum lies between. inf without such a
    turn (a profile whose planet top is lower than L3's). The grid of HORSESHOE_SCAN_TURNS turns per side, refined
    between the neighbours of its best point, is numerical."""
    best = np.inf

    def period_at(theta: float) -> float:
        if turn_site(profile, theta) != SITE_PLANET:
            return np.inf
        value = loop_period(profile, [theta])
        return float(value) if np.isfinite(value) and value > 0 else np.inf

    lo = float(theta_min_rad)
    if not (np.isfinite(lo) and 0 <= lo < np.pi):
        return np.inf
    grid = np.linspace(lo, np.pi, HORSESHOE_SCAN_TURNS + 1)[:-1]
    for side in (1.0, -1.0):
        values = [period_at(side * x) for x in grid]
        k = int(np.argmin(values))
        if not np.isfinite(values[k]):
            continue
        left, right = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        found = optimize.minimize_scalar(
            lambda x, sign=side: min(period_at(sign * x), NO_LOOP_PENALTY), bounds=(left, right), method='bounded', options={'xatol': 1e-4}
        )
        best = min(best, values[k], float(found.fun) if found.success and found.fun < NO_LOOP_PENALTY else np.inf)
    return float(best)


def averaged_turns(
    t_yr: np.ndarray, averaged: np.ndarray, tp_times_yr: np.ndarray, tp_values_rad: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """(times [yr], values [rad]) of a run's turning points read on the orbit-averaged angle (`geometry.orbit_averaged`
    of the raw sigma): for each TP the extremum of its kind (a maximum where the TP is above its neighbour) of
    `averaged` within half the shorter adjacent half-cycle of its time, on the branch of the filtered TP (the nearest
    multiple of 2 pi). The filter rounds a sharp turn at the planet off: 2003 YN107's horseshoe turns read 39 / 319 deg
    on sigma_f (a tadpole turn to `turn_site`) and 2.0 / 358.3 deg averaged. nan for a run of fewer than two TPs or a
    TP with no averaged sample near it."""
    t, x = np.asarray(t_yr, dtype=float), np.asarray(averaged, dtype=float)
    times, values = np.asarray(tp_times_yr, dtype=float), np.asarray(tp_values_rad, dtype=float)
    out_t, out_v = np.full(len(times), NAN), np.full(len(times), NAN)
    if len(times) < 2:
        return out_t, out_v
    for i, ti in enumerate(times):
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(times)]
        half = 0.5 * min(abs(times[j] - ti) for j in neighbours)
        kind = 1.0 if values[i] > values[neighbours[0]] else -1.0
        lo, hi = np.searchsorted(t, ti - half, 'left'), np.searchsorted(t, ti + half, 'right')
        part = kind * x[lo:hi]
        if not np.any(np.isfinite(part)):
            continue
        k = int(np.nanargmax(part))
        value = float(x[lo + k])
        out_t[i] = float(t[lo + k])
        out_v[i] = value + TWO_PI * np.round((values[i] - value) / TWO_PI)
    return out_t, out_v


def turn_conservation(
    t_yr: np.ndarray, a: np.ndarray, t_turn_yr: float, side_window_yr: Tuple[float, float], a_ref_au: float
) -> Tuple[float, float, float]:
    """(|da| before, |da| after [au], out/in) across a turn at t_turn: the medians of |a - a_ref| over
    [t_turn - far, t_turn - near] and [t_turn + near, t_turn + far], side_window_yr = (near, far). out/in = 1 for a
    reflection by the potential (|da| mirrored), not 1 for a kick. Pass the orbit-averaged a
    (`geometry.orbit_averaged`) and a_ref = the exact 1:1 axis (`CoorbitalProfile.a_planet_au`). nan with fewer than
    MIN_SIDE_SAMPLES samples on a side."""
    t, x = np.asarray(t_yr, dtype=float), np.asarray(a, dtype=float)
    near, far = float(side_window_yr[0]), float(side_window_yr[1])
    before = (t >= t_turn_yr - far) & (t <= t_turn_yr - near) & np.isfinite(x)
    after = (t >= t_turn_yr + near) & (t <= t_turn_yr + far) & np.isfinite(x)
    if np.count_nonzero(before) < MIN_SIDE_SAMPLES or np.count_nonzero(after) < MIN_SIDE_SAMPLES:
        return NAN, NAN, NAN
    da_in = float(np.median(np.abs(x[before] - a_ref_au)))
    da_out = float(np.median(np.abs(x[after] - a_ref_au)))
    return da_in, da_out, (da_out / da_in if da_in > 0 else NAN)


@dataclass(frozen=True)
class TurnPhysics:
    """One turning point of a 1:1 angle read against the averaged potential (`turn_physics`). `da_ratio_in/out` are
    the medians of |a - a_P| / da_max on each side (<= 1: the potential can reverse the body there)."""

    t_yr: float
    theta_deg: float  # wrapped to [0, 360)
    site: str  # SITE_*
    da_in_au: float
    da_out_au: float
    out_in: float
    da_ratio_in: float
    da_ratio_out: float


def turn_physics(
    profile: CoorbitalProfile, t_yr: np.ndarray, theta_rad: np.ndarray, a: np.ndarray, t_turn_yr: float, side_window_yr: Tuple[float, float]
) -> TurnPhysics:
    """`TurnPhysics` of a turn at t_turn from the (orbit-averaged) theta = lambda - lambda_P, unwrapped, and a on
    the record's grid; the turn's theta is the sample nearest t_turn, the sides are those of `turn_conservation`."""
    t, theta, x = (np.asarray(v, dtype=float) for v in (t_yr, theta_rad, a))
    i = int(np.argmin(np.abs(t - t_turn_yr)))
    turn = float(theta[i])
    a_p = profile.a_planet_au
    da_in, da_out, out_in = turn_conservation(t, x, t_turn_yr, side_window_yr, a_p)
    near, far = float(side_window_yr[0]), float(side_window_yr[1])
    ratios = []
    for side in ((t >= t_turn_yr - far) & (t <= t_turn_yr - near), (t >= t_turn_yr + near) & (t <= t_turn_yr + far)):
        side &= np.isfinite(theta) & np.isfinite(x)
        if np.count_nonzero(side) < MIN_SIDE_SAMPLES:
            ratios.append(NAN)
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios.append(float(np.median(np.abs(x[side] - a_p) / da_max(profile, turn, theta[side]))))
    return TurnPhysics(
        t_yr=float(t[i]),
        theta_deg=float(np.degrees(np.mod(turn, TWO_PI))),
        site=turn_site(profile, turn),
        da_in_au=da_in,
        da_out_au=da_out,
        out_in=out_in,
        da_ratio_in=ratios[0],
        da_ratio_out=ratios[1],
    )


def leg_columns(entry: Optional[TurnPhysics], exit_: Optional[TurnPhysics], loop_period_yr: float = NAN) -> Dict[str, object]:
    """CSV columns `cpl_*` of one leg (no status): the site, |da| / da_max on the leg's side (after the entry turn,
    before the exit turn) and out/in of its entry and exit turns, and the loop period; empty for a missing turn."""
    columns: Dict[str, object] = {name: ('' if name in LEG_TEXT_COLUMNS else NAN) for name in LEG_COLUMNS}
    if entry is not None:
        columns.update(cpl_turn_site_entry=entry.site, cpl_turn_da_ratio_entry=entry.da_ratio_out, cpl_turn_out_in_entry=entry.out_in)
    if exit_ is not None:
        columns.update(cpl_turn_site_exit=exit_.site, cpl_turn_da_ratio_exit=exit_.da_ratio_in, cpl_turn_out_in_exit=exit_.out_in)
    columns['cpl_loop_period_yr'] = float(loop_period_yr)
    return columns


# --- the profile of a record's window ------------------------------------------------------------------------------


def profile_from_orbits(
    body: Orbit, planet: Orbit, planet_mass: float, planet_mean_motion_rad_yr: float, settings: AnalysisSettings
) -> Optional[CoorbitalProfile]:
    """The profile of fixed orbits (the table through `coorbital.table_for`, softened as the force law); None without a
    positive mass and mean motion."""
    if not (np.isfinite(planet_mass) and planet_mass > 0 and np.isfinite(planet_mean_motion_rad_yr) and planet_mean_motion_rad_yr > 0):
        return None
    softening = coorbital_module.hill_softening(planet_mass, settings.mmr_coorbital_softening_hill)
    table = coorbital_module.table_for(body, planet, settings.mmr_coorbital_grid_points, softening)
    return CoorbitalProfile(table, float(planet_mass), float(planet_mean_motion_rad_yr))


def profile_at(
    t_yr: np.ndarray,
    e,
    inc,
    omega,
    Omega,
    t0_yr: float,
    t1_yr: float,
    planet: str,
    planet_mass: float,
    planet_mean_motion_rad_yr: float,
    settings: AnalysisSettings,
    planet_elements: Optional[Mapping[str, np.ndarray]] = None,
) -> Optional[CoorbitalProfile]:
    """The profile of a prograde 1:1 record over [t0, t1]: the body's median e and i and circular means of omega and
    Omega there (`coorbital.block_orbits` with one block), the planet's saved elements {'e', 'inc', 'Omega', 'varpi'}
    averaged the same way, else its mean J2000 orbit (`coorbital.planet_orbits`). None for an empty window or without a
    positive mass and mean motion."""
    t = np.asarray(t_yr, dtype=float)
    window = (t >= t0_yr) & (t <= t1_yr)
    if not np.any(window):
        return None

    def part(x) -> np.ndarray:
        return np.broadcast_to(np.asarray(x, dtype=float), t.shape)[window]

    _, (body,) = coorbital_module.block_orbits(t[window], part(e), part(inc), part(omega), part(Omega), np.inf)
    saved = None if planet_elements is None else {name: part(values) for name, values in planet_elements.items()}
    planet_orbit = coorbital_module.planet_orbits(t[window], planet, saved, np.array([0]), np.inf)[0]
    return profile_from_orbits(body, planet_orbit, planet_mass, planet_mean_motion_rad_yr, settings)


def series_profile(series, context, t0_yr: float, t1_yr: float, settings: AnalysisSettings) -> Optional[CoorbitalProfile]:
    """`profile_at` of an `MMRSeries` with its `Context` (the planet's mass and mean motion, its saved elements);
    None for an angle that is not a prograde 1:1 (`preprocess.coorbital_planet`)."""
    # imported here: preprocess may come to import this module (W28 C3)
    from resonances.resonance.analysis.mmr.preprocess import coorbital_inputs, coorbital_planet

    planet = coorbital_planet(series.resonance)
    inputs = coorbital_inputs(series, context, planet) if planet is not None else None
    if inputs is None:
        return None
    mu, n_planet, elements = inputs
    return profile_at(series.t_yr, series.e, series.inc, series.omega, series.Omega, t0_yr, t1_yr, planet, mu, n_planet, settings, elements)


# --- the local law on a window -------------------------------------------------------------------------------------


def local_law(
    record: CoorbitalInput, window_yr: Tuple[float, float], cutoff_yr: float, settings: AnalysisSettings
) -> Optional[CoorbitalLaw]:
    """The 1:1 force law and energy integral (`coorbital.coorbital_law`, its own force_check and energy_check) on one
    window: a of the record low-passed at `cutoff_yr` and differentiated, the model through the same low-pass, the
    elements in blocks of the cutoff, one energy over the window (a leg conserves it). `record` holds the record's
    unfiltered angle, a and elements (its mask, pieces, a_dot, cutoffs and cycle edges are replaced). None when the
    cutoff is not longer than two steps, the window (clipped to the record) is shorter than the cutoff (less than one
    filtered period: no fit) or the filter fails."""
    t = np.asarray(record.t_yr, dtype=float)
    if len(t) < 2 or not (np.isfinite(cutoff_yr) and cutoff_yr > 2 * float(np.median(np.diff(t)))):
        return None
    lo, hi = max(float(window_yr[0]), float(t[0])), min(float(window_yr[1]), float(t[-1]))
    if not hi - lo >= cutoff_yr:
        return None
    a_f = filtering.lowpass_or_none(t, np.asarray(record.a, dtype=float), cutoff_yr, settings)
    if a_f is None:
        return None
    data = replace(
        record,
        adot=filtering.time_derivative(t, a_f),
        mask=(t >= lo) & (t <= hi),
        pieces=((lo, hi),),
        cutoffs_yr=(float(cutoff_yr),),
        cycle_edges_yr=(),
        block_yr=float(cutoff_yr),
    )
    return coorbital_module.coorbital_law(data, settings)


def series_input(series, context, angle: np.ndarray) -> Optional[CoorbitalInput]:
    """The `CoorbitalInput` of a whole 1:1 record for `local_law` (prograde or retrograde 1/-1, `preprocess.
    law_planet`): `angle` is the unfiltered unwrapped sigma (`Preprocessed.angle_clean`); None for another angle or
    without the planet's mass and mean motion."""
    from resonances.resonance.analysis.mmr.preprocess import coorbital_inputs, law_planet  # see `series_profile`

    spec = series.resonance
    planet = law_planet(spec)
    inputs = coorbital_inputs(series, context, planet) if planet is not None else None
    if inputs is None:
        return None
    mu, n_planet, elements = inputs
    t = np.asarray(series.t_yr, dtype=float)
    return CoorbitalInput(
        t_yr=t,
        angle=np.asarray(angle, dtype=float),
        a=np.asarray(series.a, dtype=float),
        adot=np.zeros_like(t),
        mask=np.ones(len(t), dtype=bool),
        pieces=((float(t[0]), float(t[-1])),),
        e=series.e,
        inc=series.inc,
        omega=series.omega,
        Omega=series.Omega,
        planet=planet,
        planet_mass=mu,
        planet_mean_motion_rad_yr=n_planet,
        body_coefficient=spec.body_coefficient,
        cutoffs_yr=(),
        planet_elements=elements,
        retrograde=spec.retrograde,
        omega_coefficient=spec.body_node_coefficient - 1 if spec.retrograde else 1,  # 2k - 1 of 1J-1r<k>, as response
    )


# --- the floor of one run from the loop through its own turning points (W28 C3) ------------------------------------

FLOOR_SOURCE_LOOP = 'profile_loop'  # factor x the loop period through the run's own reflection turns
FLOOR_SOURCE_NOT_REFLECTION = 'not_reflection'  # a turn inside a well (tadpole, QS) or no site: the angle's floor
FLOOR_SOURCE_NO_LOOP = 'no_loop'  # no loop of the potential joins its turns (a top between them, a separatrix)
FLOOR_SOURCE_LAW = 'law_not_positive'  # the local 1:1 law has K / K_theory <= 0 or cannot be fitted on the run
FLOOR_SOURCE_NO_PROFILE = 'no_profile'
REFLECTION_SITES = (SITE_PLANET, SITE_L3)
LOOP_PAIRS_MAX = 9  # numerical (cost): the loop period of a run is the median over this many of its turn pairs


@dataclass
class CoorbitalRunFloor:
    """`geometry.RunFloorModel` of a prograde 1:1 angle (rule `coorbital_floor_profile`, W28 C3; cards g02-1 A, g01-6
    A, q-3 A): the floor of a run whose turning points are all reflections on a top (`turn_site` planet or l3, read on
    the orbit-averaged angle, `averaged_turns`) is mmr_coorbital_period_floor_factor x the period of the loop of the
    averaged potential through those turns (`loop_period`, the median over its consecutive pairs, at most
    LOOP_PAIRS_MAX of them), provided the local
    1:1 law on the run (`local_law` on the run +- one median half-cycle, at the angle's cutoff) has K / K_theory > 0:
    the force that reverses it is the planet's. Any other run (a turn inside a tadpole or quasi-satellite well, no loop
    between its turns such as 2022 AP1's 148 / 186 deg about L3, or no positive law) has no floor of its own and keeps
    the angle's (the tadpole floor; the QS well of G1q keeps its own, `mmr.decide.period_floor`). The reading's columns
    (the C2 leg columns of its first and last turn, `cpl_turn_sites`, `cpl_law_K_over_K_theory`, `cpl_floor_yr`, `cpl_floor_source`) go to
    episodes.csv for every run of >= 2 TPs.

    Why: the tadpole floor P / sqrt(27 mu / 4) / 2 is the shortest tadpole libration, but a horseshoe that turns close to
    the planet is faster (2003 YN107: a closed 103-yr cycle turning 2 deg from the Earth, loop 119 yr, under the 111-yr
    tadpole floor). The a<->sigma identity makes F2 of card g01-6 (a crosses a_P at each turn) automatic, so the test is
    the potential's own period at the observed turns (q-3 A). Pure apart from the profile and law fits it caches."""

    series: object
    context: object
    t_yr: np.ndarray
    averaged: np.ndarray  # the orbit-averaged unwrapped raw sigma (`geometry.orbit_averaged`)
    a_averaged: np.ndarray  # the orbit-averaged a [au]
    body_coefficient: int
    record: Optional[CoorbitalInput]
    cutoff_yr: float
    settings: AnalysisSettings
    cache: Dict[tuple, object] = field(default_factory=dict)  # run_key -> RunFloor

    def __call__(self, run):
        from resonances.resonance.analysis.geometry import RunFloor, run_key  # geometry does not import this module

        key = run_key(run)
        if key not in self.cache:
            floor, setting, source, columns = self.read(run.tp_times_yr, run.tp_values_rad)
            self.cache[key] = RunFloor(floor, setting, source, tuple(columns.items()))
        return self.cache[key]

    def read(self, tp_times_yr, tp_values_rad) -> Tuple[float, str, str, Dict[str, object]]:
        """(floor [yr] or nan, its setting, source FLOOR_SOURCE_*, columns) of one run's turning points."""
        columns: Dict[str, object] = {name: ('' if name in LEG_TEXT_COLUMNS else NAN) for name in LEG_COLUMNS}
        columns.update(cpl_turn_sites='', cpl_law_K_over_K_theory=NAN, cpl_floor_yr=NAN, cpl_floor_source=FLOOR_SOURCE_NO_PROFILE)
        times, values = np.asarray(tp_times_yr, dtype=float), np.asarray(tp_values_rad, dtype=float)
        turn_t, turn_sigma = averaged_turns(self.t_yr, self.averaged, times, values)
        if len(times) < 2 or not np.all(np.isfinite(turn_sigma)):
            return NAN, '', FLOOR_SOURCE_NO_PROFILE, columns
        profile = series_profile(self.series, self.context, float(times[0]), float(times[-1]), self.settings)
        if profile is None:
            return NAN, '', FLOOR_SOURCE_NO_PROFILE, columns
        theta = theta_of(turn_sigma, self.body_coefficient)
        sites = [turn_site(profile, x) for x in theta]
        reflections = all(site in REFLECTION_SITES for site in sites)
        # the loops of its consecutive pairs: at most LOOP_PAIRS_MAX of them, evenly spread (a 20-kyr QS run has
        # 5000 turns); the first pair alone where the run is no reflection (a column only)
        pairs = np.unique(np.linspace(0, len(theta) - 2, min(len(theta) - 1, LOOP_PAIRS_MAX if reflections else 1)).round().astype(int))
        loops = [loop_period(profile, theta[i : i + 2]) for i in pairs]
        loop = float(np.median(loops)) if np.all(np.isfinite(loops)) else NAN
        half = float(np.median(np.diff(times)))
        near, far = (float(f) * half for f in self.settings.mmr_coorbital_turn_side_window)
        theta_all = theta_of(self.averaged, self.body_coefficient)
        entry, exit_ = (turn_physics(profile, self.t_yr, theta_all, self.a_averaged, float(turn_t[i]), (near, far)) for i in (0, -1))
        columns.update(leg_columns(entry, exit_, loop))
        columns['cpl_turn_sites'] = ';'.join(sites)
        if not reflections:
            columns['cpl_floor_source'] = FLOOR_SOURCE_NOT_REFLECTION
            return NAN, '', FLOOR_SOURCE_NOT_REFLECTION, columns
        if not (np.isfinite(loop) and loop > 0):
            columns['cpl_floor_source'] = FLOOR_SOURCE_NO_LOOP
            return NAN, '', FLOOR_SOURCE_NO_LOOP, columns
        law = (
            local_law(self.record, (float(times[0]) - half, float(times[-1]) + half), self.cutoff_yr, self.settings)
            if self.record
            else None
        )
        scale = float(law.scale) if law is not None else NAN
        columns['cpl_law_K_over_K_theory'] = scale
        if not (np.isfinite(scale) and scale > 0):
            columns['cpl_floor_source'] = FLOOR_SOURCE_LAW
            return NAN, '', FLOOR_SOURCE_LAW, columns
        floor = float(self.settings.mmr_coorbital_period_floor_factor * loop)
        columns.update(cpl_floor_yr=floor, cpl_floor_source=FLOOR_SOURCE_LOOP)
        return floor, 'mmr_coorbital_period_floor_factor', FLOOR_SOURCE_LOOP, columns


def coorbital_run_floor(
    series, context, t_yr, raw_sigma, orbit_period_yr: float, cutoff_yr: float, settings
) -> Optional[CoorbitalRunFloor]:
    """The `CoorbitalRunFloor` of a prograde 1:1 record (None for another angle, without the planet's mass and mean
    motion, or with MMR_COORBITAL_PROFILE_FLOOR off): the averaged angle and a over one orbital period (the edge rule's
    `geometry.orbit_averaged`), the record input of the local law."""
    from resonances.resonance.analysis.geometry import orbit_averaged
    from resonances.resonance.analysis.mmr.preprocess import coorbital_inputs, coorbital_planet  # see `series_profile`

    planet = coorbital_planet(series.resonance)
    if not settings.mmr_coorbital_profile_floor or planet is None or coorbital_inputs(series, context, planet) is None:
        return None
    t = np.asarray(t_yr, dtype=float)
    return CoorbitalRunFloor(
        series=series,
        context=context,
        t_yr=t,
        averaged=orbit_averaged(t, np.asarray(raw_sigma, dtype=float), orbit_period_yr),
        a_averaged=orbit_averaged(t, np.asarray(series.a, dtype=float), orbit_period_yr),
        body_coefficient=int(series.resonance.body_coefficient),
        record=series_input(series, context, raw_sigma),
        cutoff_yr=float(cutoff_yr),
        settings=settings,
    )


FASTEST_CACHE_SIZE = 64  # numerical: profiles whose fastest horseshoe is kept (`fast_reference_yr`)
_FASTEST_CACHE: Dict[tuple, float] = {}


def fast_reference_yr(series, context, settings, floor_yr: float) -> float:
    """The period below which a 1:1 angle holds no libration at all, the reference of `analyse.fast_part` (W28 C3):
    min(the angle's tadpole floor `floor_yr`, mmr_coorbital_period_floor_factor x the fastest horseshoe loop that turns
    at least mmr_coorbital_min_turn_hill Hill radii a_P (mu / 3)^(1/3) from the planet, `fastest_horseshoe_period`, on
    the record's mean profile): the shortest floor any run of the angle can get (`CoorbitalRunFloor`), so a horseshoe
    the profile floor accepts is no fast residual that hides its own turning points. `floor_yr` for another angle,
    without a profile or with MMR_COORBITAL_PROFILE_FLOOR off."""
    from resonances.resonance.analysis.mmr.preprocess import coorbital_inputs, coorbital_planet  # see `series_profile`

    planet = coorbital_planet(series.resonance)
    inputs = coorbital_inputs(series, context, planet) if planet is not None and settings.mmr_coorbital_profile_floor else None
    if inputs is None:
        return float(floor_yr)
    t = np.asarray(series.t_yr, dtype=float)
    profile = series_profile(series, context, float(t[0]), float(t[-1]), settings)
    if profile is None:
        return float(floor_yr)
    hill = (inputs[0] / 3.0) ** (1.0 / 3.0)
    theta_min = float(settings.mmr_coorbital_min_turn_hill * hill)
    table = profile.table  # one record is preprocessed more than once (G1q, the owner): the scan is kept per profile
    key = (
        table.body,
        table.planet,
        len(table.theta_rad),
        float(table.potential[0]),
        profile.planet_mass,
        profile.planet_mean_motion_rad_yr,
        theta_min,
    )
    if key not in _FASTEST_CACHE:
        if len(_FASTEST_CACHE) >= FASTEST_CACHE_SIZE:
            _FASTEST_CACHE.pop(next(iter(_FASTEST_CACHE)))
        _FASTEST_CACHE[key] = fastest_horseshoe_period(profile, theta_min)
    fastest = _FASTEST_CACHE[key]
    reference = settings.mmr_coorbital_period_floor_factor * fastest
    return float(min(floor_yr, reference)) if np.isfinite(reference) else float(floor_yr)


# --- the turning point the filter's edge trim hides (W28 C4) -------------------------------------------------------

EDGE_NO_PROFILE = 'no_profile'  # no averaged potential for the window of the turn
EDGE_SITE = 'site'  # the turn lies inside a well (tadpole, quasi-satellite), no reflection on a top
EDGE_A_CROSS = 'a_cross'  # a does not cross a_P across the turn
EDGE_ENERGY = 'energy'  # |da| not mirrored: |ln out/in| above mmr_coorbital_turn_energy_tol, or not measurable


@dataclass
class CoorbitalEdgeTurn:
    """`geometry.EdgeTurnModel` of a prograde 1:1 angle (rule coorbital_edge_turn, W28 C4, cards g02-2 A, g01-3 A):
    the physics of a reversal of the raw angle averaged over mmr_coorbital_edge_turn_orbits orbits inside the filter's
    edge trim (`geometry._edge_raw_turns` finds it and adds it only when it closes a cycle). It holds when

    - it is a reflection on a top of the averaged potential (`turn_site` planet or l3, on the averaged potential of
      the window [t - h, t + h], h the run's median half-cycle): the turns of a horseshoe; inside a well the angle
      turns with no top to reflect it, and a ripple there is not separable from a turn;
    - a crosses a_P across it: theta_dot = -(3/2) n (a - a_P) / a_P, so a reversal of the averaged angle that a does
      not answer is no turn of the 1:1 dynamics (a consistency check of the averaged angle against a);
    - |da| is mirrored, |ln(out/in)| <= mmr_coorbital_turn_energy_tol (`turn_conservation` on the averaged a): the
      averaged problem is time-reversible about a turn, a kick (a close approach) is not. Its sides are those of
      mmr_coorbital_turn_side_window x h, shrunk by one factor to the stretch left to the record's edge (both sides
      alike, so a reflection still reads 1): 2007 VU6 turns 6 yr before the end, the 30-yr side of its 75-yr
      half-cycle does not exist.

    The swing it needs on both sides (`min_prominence_rad`, mmr_coorbital_edge_turn_min_deg) is tested by the
    geometry. 2006 BZ147: 960 yr, swing 180 deg, planet, out/in 0.90: it closes the horseshoe cycle 817-960 yr."""

    series: object
    context: object
    t_yr: np.ndarray
    averaged: np.ndarray  # the raw sigma averaged over mmr_coorbital_edge_turn_orbits orbits (`geometry.orbit_averaged`)
    a_averaged: np.ndarray  # a averaged the same way [au]
    body_coefficient: int
    settings: AnalysisSettings
    profile: Optional[CoorbitalProfile] = None  # one averaged potential for every turn (else the series' at the turn)

    @property
    def min_prominence_rad(self) -> float:
        return float(np.radians(self.settings.mmr_coorbital_edge_turn_min_deg))

    def read(self, t_turn_yr: float, value_rad: float, half_period_yr: float, record: Tuple[float, float]):
        """`geometry.EdgeTurnReading` of the reversal at t_turn (sigma = value_rad, unwrapped) of a run with the median
        half-cycle `half_period_yr`, inside `record` [yr]."""
        from resonances.resonance.analysis.geometry import EdgeTurnReading  # geometry does not import this module

        t, a = np.asarray(self.t_yr, dtype=float), np.asarray(self.a_averaged, dtype=float)
        half = float(half_period_yr)
        profile = self.profile
        if profile is None and self.series is not None:
            profile = series_profile(
                self.series, self.context, max(record[0], t_turn_yr - half), min(record[1], t_turn_yr + half), self.settings
            )
        if profile is None or not (np.isfinite(half) and half > 0):
            return EdgeTurnReading(SITE_NONE, NAN, False, EDGE_NO_PROFILE)
        site = turn_site(profile, float(theta_of(np.array([value_rad]), self.body_coefficient)[0]))
        near, far = (float(f) * half for f in self.settings.mmr_coorbital_turn_side_window)
        left = min(t_turn_yr - record[0], record[1] - t_turn_yr)
        shrink = min(1.0, left / far) if far > 0 else NAN
        side = (near * shrink, far * shrink)
        _, _, out_in = turn_conservation(t, a, t_turn_yr, side, profile.a_planet_au)
        before = (t >= t_turn_yr - side[1]) & (t <= t_turn_yr - side[0]) & np.isfinite(a)
        after = (t >= t_turn_yr + side[0]) & (t <= t_turn_yr + side[1]) & np.isfinite(a)
        crosses = bool(
            np.count_nonzero(before) >= MIN_SIDE_SAMPLES
            and np.count_nonzero(after) >= MIN_SIDE_SAMPLES
            and np.median(a[before] - profile.a_planet_au) * np.median(a[after] - profile.a_planet_au) < 0
        )
        if site not in REFLECTION_SITES:
            outcome = EDGE_SITE
        elif not crosses:
            outcome = EDGE_A_CROSS
        elif not (np.isfinite(out_in) and out_in > 0 and abs(np.log(out_in)) <= self.settings.mmr_coorbital_turn_energy_tol):
            outcome = EDGE_ENERGY
        else:
            outcome = ''
        return EdgeTurnReading(site, float(out_in), crosses, outcome)


def coorbital_edge_turn(series, context, t_yr, raw_sigma, orbit_period_yr: float, settings) -> Optional[CoorbitalEdgeTurn]:
    """The `CoorbitalEdgeTurn` of a prograde 1:1 record: its raw sigma and a averaged over
    mmr_coorbital_edge_turn_orbits orbital periods (`geometry.orbit_averaged`). None for another angle, without the
    planet's mass and mean motion, or with MMR_COORBITAL_EDGE_TURN off."""
    from resonances.resonance.analysis.geometry import orbit_averaged
    from resonances.resonance.analysis.mmr.preprocess import coorbital_inputs, coorbital_planet  # see `series_profile`

    planet = coorbital_planet(series.resonance)
    if not settings.mmr_coorbital_edge_turn or planet is None or coorbital_inputs(series, context, planet) is None:
        return None
    t = np.asarray(t_yr, dtype=float)
    period = float(settings.mmr_coorbital_edge_turn_orbits * orbit_period_yr)
    return CoorbitalEdgeTurn(
        series=series,
        context=context,
        t_yr=t,
        averaged=orbit_averaged(t, np.asarray(raw_sigma, dtype=float), period),
        a_averaged=orbit_averaged(t, np.asarray(series.a, dtype=float), period),
        body_coefficient=int(series.resonance.body_coefficient),
        settings=settings,
    )


def reversal_indices(values: np.ndarray, min_reversal: float) -> np.ndarray:
    """The sample indices of the reversals of a series that each go back by at least `min_reversal` from the extreme
    reached since the last one (a zigzag; the first also by min_reversal from the start, so a wiggle at the start is no
    reversal): the legs of the W28 C1 study (`CoorbitalWells.legs`)."""
    x = np.asarray(values, dtype=float)
    out: list = []
    if len(x) < 3:
        return np.zeros(0, dtype=int)
    direction, extreme = 0, 0
    high = low = 0
    for i in range(1, len(x)):
        v = x[i]
        if direction == 0:
            if v > x[high]:
                high = i
            if v < x[low]:
                low = i
            if x[high] - v >= min_reversal and x[high] - x[0] >= min_reversal:
                out.append(high)
                direction, extreme = -1, i
            elif v - x[low] >= min_reversal and x[0] - x[low] >= min_reversal:
                out.append(low)
                direction, extreme = 1, i
        elif (v - x[extreme]) * direction > 0:
            extreme = i
        elif (x[extreme] - v) * direction >= min_reversal:
            out.append(extreme)
            direction, extreme = -direction, i
    return np.asarray(out, dtype=int)


# --- the wells a run's swings hold (W28 C7) ---------------------------------------------------------------------------


@dataclass
class CoorbitalWells:
    """`geometry.WellsModel` of a prograde 1:1 angle (rule coorbital_well_cut, W28 C7; cards g03-1 A, g03-9 A): the
    wells of the averaged potential that swings of sigma hold, on the profile of the window they come from (one
    profile per window, through `series_profile` and the record's table cache). An episode of a 1:1 angle ends only
    where sigma crosses a separatrix; a change of well (quasi-satellite, tadpole, horseshoe, compound) or of amplitude
    inside the co-orbital wells is a sub-regime of one libration (`geometry._accept_in_wells`, `geometry.coorbital_marks`).
    `profile` fixes one potential for every window (tests)."""

    series: object
    context: object
    body_coefficient: int
    settings: AnalysisSettings
    profile: Optional[CoorbitalProfile] = None
    cache: Dict[Tuple[float, float], Optional[CoorbitalProfile]] = field(default_factory=dict)

    def profile_of(self, t0_yr: float, t1_yr: float) -> Optional[CoorbitalProfile]:
        """The averaged potential of the window [t0, t1] (`series_profile`; the fixed `profile` when given)."""
        if self.profile is not None:
            return self.profile
        key = (float(t0_yr), float(t1_yr))
        if key not in self.cache:
            self.cache[key] = series_profile(self.series, self.context, key[0], key[1], self.settings)
        return self.cache[key]

    def wells(self, t0_yr: float, t1_yr: float, lo_rad, hi_rad) -> Tuple[str, ...]:
        """The name of the well (`well_of`) each swing [lo, hi] of sigma [rad, unwrapped] holds in the potential of
        [t0, t1]: SITE_QS, SITE_L4, SITE_L5, WELL_HORSESHOE, WELL_COMPOUND, or SITE_NONE for a swing that holds no
        minimum (or covers a whole turn). () without a profile."""
        profile = self.profile_of(t0_yr, t1_yr)
        if profile is None:
            return ()
        ends = np.vstack(
            [
                theta_of(np.asarray(lo_rad, dtype=float), self.body_coefficient),
                theta_of(np.asarray(hi_rad, dtype=float), self.body_coefficient),
            ]
        )
        low, high = ends.min(axis=0), ends.max(axis=0)
        return tuple(well_of(profile, float((a + b) / 2), float((b - a) / 2)).name for a, b in zip(low, high))

    def levels(self, t0_yr: float, t1_yr: float, values_rad) -> np.ndarray:
        """R~ at theta of each value of sigma [rad] in the potential of [t0, t1]: at a turning point (da = 0) the level
        of its loop in the energy integral (eq. 1 of the module docstring). nan without a profile."""
        values = np.asarray(values_rad, dtype=float)
        profile = self.profile_of(t0_yr, t1_yr)
        if profile is None:
            return np.full(values.shape, NAN)
        return np.asarray(profile.potential(theta_of(values, self.body_coefficient)), dtype=float)

    # --- the residence and the sub-regimes' laws (W28 C8, rule coorbital_residence; data only) -----------------------

    def _averaged(self, t_yr: np.ndarray, raw_rad: np.ndarray, orbit_period_yr: float) -> Tuple[np.ndarray, np.ndarray]:
        """(t, theta averaged over mmr_coorbital_edge_turn_orbits orbits): the W28 C1 leg's angle (`geometry.
        orbit_averaged`, the edge turn's), theta = lambda - lambda_P; cached per raw array (one per geometry pass)."""
        from resonances.resonance.analysis.geometry import orbit_averaged  # geometry does not import this module

        key = ('averaged', id(raw_rad), len(raw_rad), float(orbit_period_yr))
        if key not in self.cache:
            t = np.asarray(t_yr, dtype=float)
            period = float(self.settings.mmr_coorbital_edge_turn_orbits * orbit_period_yr)
            raw = theta_of(np.asarray(raw_rad, dtype=float), self.body_coefficient)
            self.cache[key] = (t, orbit_averaged(t, raw, period), period)
        t, theta, _ = self.cache[key]
        return t, theta

    def legs(self, t_yr: np.ndarray, raw_rad: np.ndarray, orbit_period_yr: float):
        """`geometry.AveragedLegs` of the raw angle: theta averaged over mmr_coorbital_edge_turn_orbits orbits, cut at
        its reversals of at least mmr_coorbital_edge_turn_min_deg (from the extreme reached since the last one: the W28
        C1 study's legs, 20 deg), the sweep max - min of each leg in turns. By sigma_dot = -(3/2) n (a - a_P) / a_P a
        reversal of the averaged angle is a crossing of a_P; a leg that sweeps a whole turn has passed the planet or
        L3 without turning back. None for fewer than three finite samples."""
        from resonances.resonance.analysis.geometry import AveragedLegs  # geometry does not import this module

        t, theta = self._averaged(t_yr, raw_rad, orbit_period_yr)
        finite = np.isfinite(theta)
        if np.count_nonzero(finite) < 3:
            return None
        t, theta = t[finite], theta[finite]
        turns = reversal_indices(theta, float(np.radians(self.settings.mmr_coorbital_edge_turn_min_deg)))
        edges = np.concatenate([[0], turns, [len(t) - 1]]).astype(int)
        sweeps = np.array([np.ptp(theta[a : b + 1]) / TWO_PI for a, b in zip(edges[:-1], edges[1:])])
        return AveragedLegs(t_yr=t[turns], sweep_rev=sweeps, record=(float(t[0]), float(t[-1])))

    def zone_max(self, legs, turns_yr: Sequence[float], raw_rad: np.ndarray) -> float:
        """The largest |a - a_P| / da_max on either side of the reversals `turns_yr` of `legs` (`turn_physics`, its
        sides mmr_coorbital_turn_side_window x the shorter adjacent leg, on the averaged potential of the two legs; a
        averaged as theta), over at most LOOP_PAIRS_MAX of them evenly spread (a numerical cost cap). Report only: at
        20 kyr the accepted core of 2024 TY11 1M-1 starts at a turn of zone 2.2. nan without the series' a or a
        reading."""
        if self.series is None or not len(turns_yr):
            return NAN
        from resonances.resonance.analysis.geometry import orbit_averaged  # geometry does not import this module

        cached = [value for key, value in self.cache.items() if key[:3] == ('averaged', id(raw_rad), len(raw_rad))]
        if not cached:
            return NAN
        t, theta, period = cached[-1]
        key = ('a_averaged', len(t), period)
        if key not in self.cache:
            self.cache[key] = orbit_averaged(t, np.asarray(self.series.a, dtype=float), period)
        a = self.cache[key]
        bounds = legs.bounds_yr
        chosen = np.asarray(turns_yr, dtype=float)
        if len(chosen) > LOOP_PAIRS_MAX:
            chosen = chosen[np.unique(np.linspace(0, len(chosen) - 1, LOOP_PAIRS_MAX).round().astype(int))]
        ratios = []
        for turn in chosen:
            k = int(np.searchsorted(bounds, turn))
            lo, hi = float(bounds[max(k - 1, 0)]), float(bounds[min(k + 1, len(bounds) - 1)])
            half = min(turn - lo, hi - turn)
            profile = self.profile_of(max(lo, turn - half), min(hi, turn + half)) if half > 0 else None
            if profile is None:
                continue
            near, far = (float(f) * half for f in self.settings.mmr_coorbital_turn_side_window)
            physics = turn_physics(profile, t, theta, a, float(turn), (near, far))
            ratios += [x for x in (physics.da_ratio_in, physics.da_ratio_out) if np.isfinite(x)]
        return float(max(ratios)) if ratios else NAN

    def law(self, t0_yr: float, t1_yr: float, raw_rad: np.ndarray, cutoff_yr: float) -> Tuple[str, str]:
        """(force check, energy check) of the local 1:1 law (`local_law`) on [t0, t1] at `cutoff_yr`, the raw angle
        `raw_rad` on the series' grid; 'not_measured' each without the series, or when the window is too short."""
        from resonances.resonance.analysis.verdict import CheckValue  # geometry's enum, imported lazily as the others

        missing = (str(CheckValue.NOT_MEASURED), str(CheckValue.NOT_MEASURED))
        if self.series is None:
            return missing
        record = series_input(self.series, self.context, np.asarray(raw_rad, dtype=float))
        found = local_law(record, (t0_yr, t1_yr), cutoff_yr, self.settings) if record is not None else None
        if found is None:
            return missing
        return str(found.force_check), str(found.energy_check)

    # --- the single legs outside the accepted runs (W28 C6, rule coorbital_leg; the decision is mmr.decide leaf G4c) ----

    def single_legs(
        self,
        t_yr: np.ndarray,
        raw_rad: np.ndarray,
        orbit_period_yr: float,
        floor_yr: float,
        min_floor_ratio: float,
        outside: Sequence[Tuple[float, float]],
    ):
        """The `geometry.SingleLeg` readings of the averaged angle (the legs of `legs`): every leg that sweeps less
        than a turn, lies outside the spans `outside` (the accepted runs) and whose period, twice its length L, is at
        least min_floor_ratio x floor_yr. Each gets (the W28 C1 study's measurement, calibrated there on the turns of
        confirmed cycles):

        - its topology from the crossings of the planet (theta = 0) and L3 (pi) of the averaged theta;
        - at each of its turns (an edge leg has one) `turn_physics` on the averaged theta and a: out/in and the zone
          |a - a_P| / da_max on the leg's side, the sides mmr_coorbital_turn_side_window x the shorter of the two legs
          that meet there (the turn reads the same from either leg), on the averaged potential of the turn +- a
          quarter of that length;
        - the local 1:1 law (`local_law`) on the leg widened by mmr_coorbital_leg_window_factor x L on each side (C1:
          the leg alone is too short for the energy integral of a horseshoe, W2 = +-0.25 L keeps 85 % of real legs and
          separates them 2.0x from single legs), at the leg's own cutoff max(mmr_coorbital_leg_cutoff_orbits x P_orb,
          mmr_coorbital_leg_cutoff_fraction x 2 L) (the angle's P_c is as long as the shortest legs).

        () without the series' a or fewer than three finite samples."""
        from resonances.resonance.analysis.geometry import LEG_EDGE_END, LEG_EDGE_START, SingleLeg

        if self.series is None or not (np.isfinite(floor_yr) and floor_yr > 0):
            return ()
        legs = self.legs(t_yr, raw_rad, orbit_period_yr)
        if legs is None or len(legs.sweep_rev) < 2:  # one stretch without a reversal: no turn to read
            return ()
        t, theta = self._averaged(t_yr, raw_rad, orbit_period_yr)
        a = self._a_averaged(t, orbit_period_yr)
        bounds, n = legs.bounds_yr, len(legs.sweep_rev)
        out = []
        for j in range(n):
            t0, t1 = float(bounds[j]), float(bounds[j + 1])
            length = t1 - t0
            if legs.sweep_rev[j] >= 1.0 or not length > 0 or 2 * length < min_floor_ratio * floor_yr:
                continue  # a sweep of a whole turn is no leg (the angle went past the planet and L3 without turning)
            if any(t0 < hi and t1 > lo for lo, hi in outside):
                continue  # a leg of an accepted run: a half-cycle of its closed cycle, never read again
            part = theta[(t >= t0) & (t <= t1)]
            if len(part) < 2:
                continue
            topology = leg_topology(float(np.min(part)), float(np.max(part)))
            turns = [self._leg_turn(bounds, k, t, theta, a, entry=k == j) if 0 < k < n else None for k in (j, j + 1)]
            force, scale, energy, cutoff = self._leg_law(topology, t0, t1, raw_rad, orbit_period_yr)
            edge = LEG_EDGE_START if j == 0 else (LEG_EDGE_END if j == n - 1 else '')
            out.append(SingleLeg(t0, t1, edge, topology, 2 * length / floor_yr, *turns, force, scale, energy, cutoff))
        return tuple(out)

    def _a_averaged(self, t: np.ndarray, orbit_period_yr: float) -> np.ndarray:
        """a averaged as `_averaged` averages theta (cached)."""
        from resonances.resonance.analysis.geometry import orbit_averaged  # geometry does not import this module

        period = float(self.settings.mmr_coorbital_edge_turn_orbits * orbit_period_yr)
        key = ('a_averaged', len(t), period)
        if key not in self.cache:
            self.cache[key] = orbit_averaged(t, np.asarray(self.series.a, dtype=float), period)
        return self.cache[key]

    def _leg_turn(self, bounds: np.ndarray, k: int, t: np.ndarray, theta: np.ndarray, a: np.ndarray, entry: bool):
        """The `geometry.LegTurn` at the reversal bounds[k]: `turn_physics` with sides
        mmr_coorbital_turn_side_window x the shorter of the two legs that meet there, on the averaged potential of the
        turn +- a quarter of that length; the zone on the leg's side (after an entry turn, before an exit turn)."""
        from resonances.resonance.analysis.geometry import LegTurn  # geometry does not import this module

        at = float(bounds[k])
        shorter = min(at - float(bounds[k - 1]), float(bounds[k + 1]) - at)
        profile = self.profile_of(at - 0.25 * shorter, at + 0.25 * shorter) if shorter > 0 else None
        if profile is None:
            return LegTurn(at)
        near, far = (float(f) * shorter for f in self.settings.mmr_coorbital_turn_side_window)
        physics = turn_physics(profile, t, theta, a, at, (near, far))
        zone = physics.da_ratio_out if entry else physics.da_ratio_in
        return LegTurn(at, physics.site, float(physics.out_in), float(zone))

    def _leg_law(self, topology: str, t0: float, t1: float, raw_rad: np.ndarray, orbit_period_yr: float):
        """(force check, K / K_theory, energy check, cutoff [yr]) of the local 1:1 law on the leg widened by
        mmr_coorbital_leg_window_factor x its length L on each side, at max(mmr_coorbital_leg_cutoff_orbits x P_orb,
        mmr_coorbital_leg_cutoff_fraction x 2 L); read only for a clean leg (every leg with the force gate: no other
        outcome reads it), else not_measured."""
        from resonances.resonance.analysis.geometry import CLEAN_LEGS  # geometry does not import this module
        from resonances.resonance.analysis.verdict import CheckValue

        missing = (str(CheckValue.NOT_MEASURED), NAN, str(CheckValue.NOT_MEASURED), NAN)
        if topology not in CLEAN_LEGS and not self.settings.mmr_coorbital_leg_force_gate:
            return missing
        key = ('record', id(raw_rad), len(raw_rad))
        if key not in self.cache:
            self.cache[key] = series_input(self.series, self.context, np.asarray(raw_rad, dtype=float))
        record = self.cache[key]
        length = t1 - t0
        margin = self.settings.mmr_coorbital_leg_window_factor * length
        cutoff = max(
            self.settings.mmr_coorbital_leg_cutoff_orbits * float(orbit_period_yr),
            self.settings.mmr_coorbital_leg_cutoff_fraction * 2 * length,
        )
        law = local_law(record, (t0 - margin, t1 + margin), cutoff, self.settings) if record is not None else None
        if law is None:
            return missing[:3] + (float(cutoff),)
        return str(law.force_check), float(law.scale), str(law.energy_check), float(cutoff)


def leg_topology(lo_rad: float, hi_rad: float) -> str:
    """The topology of a leg of the averaged theta = lambda - lambda_P spanning [lo, hi] (unwrapped): it crosses the
    planet (a multiple of 2 pi) and L3 (pi + a multiple of 2 pi): passing; L3 only: horseshoe; the planet only:
    quasi-satellite; neither: tadpole."""
    from resonances.resonance.analysis.geometry import LEG_HORSESHOE, LEG_PASSING, LEG_QS, LEG_TADPOLE

    planet = np.floor(hi_rad / TWO_PI) > np.floor(lo_rad / TWO_PI)
    l3 = np.floor((hi_rad - np.pi) / TWO_PI) > np.floor((lo_rad - np.pi) / TWO_PI)
    return LEG_PASSING if planet and l3 else (LEG_HORSESHOE if l3 else (LEG_QS if planet else LEG_TADPOLE))


def coorbital_wells(series, context, settings: AnalysisSettings) -> Optional[CoorbitalWells]:
    """The `CoorbitalWells` of a prograde 1:1 record; None for another angle, without the planet's mass and mean
    motion, or with MMR_COORBITAL_WELL_EPISODES off (the geometry then keeps the band test and the regime cuts)."""
    from resonances.resonance.analysis.mmr.preprocess import coorbital_inputs, coorbital_planet  # see `series_profile`

    planet = coorbital_planet(series.resonance)
    if not settings.mmr_coorbital_well_episodes or planet is None or coorbital_inputs(series, context, planet) is None:
        return None
    return CoorbitalWells(series=series, context=context, body_coefficient=int(series.resonance.body_coefficient), settings=settings)
