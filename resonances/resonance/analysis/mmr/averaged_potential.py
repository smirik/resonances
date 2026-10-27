"""The averaged disturbing function of a two-body resonance at fixed sigma, and its curvature at the occupied centre.

The theoretical strength of a two-body MMR (`strength.py`, check_mmr_strength) is read here from the disturbing
function itself, averaged over the fast angle at fixed sigma (Gallardo 2006, Icarus 184, 29, sect. 2), with both
orbits three-dimensional and the planet on its actual eccentric, inclined orbit. No expansion in e, i or alpha:
the coefficient of the term (f(alpha), the inclination members of the multiplet, the higher harmonics of sigma) is in
the number, which the scale mu e^q leaves out by 10^3-10^4 at high order (W6 C8).

Theory (sign and scale)
-----------------------
The disturbing function of the body at r perturbed by the planet at r_P (Murray & Dermott 1999, eqs. 6.44-6.45):

    R = G m_P D,   D = 1 / |r - r_P| - r . r_P / r_P^3          (direct + indirect part).

The angle sigma = k_P lambda_P + k lambda + phi (k the body's coefficient, phi the slow part: the varpi and node
terms of the body and the planet) is written with the body's mean anomaly, sigma = k M + phi_B + k_P lambda_P + phi_P
(`phase`). Averaging R over lambda_P at fixed sigma and fixed slow elements (M = (sigma - k_P lambda_P - phase) / k:
|k| turns of lambda_P close the line on the torus) gives <R>(sigma) = G m_P <D>(sigma). Lagrange's equation for a
(M&D eq. 6.145) at first order in m_P: da/dt = (2 / (n a)) dR/d lambda = (2 k / (n a)) d<R>/d sigma; with
n_dot = -(3/2)(n / a) a_dot and sigma_dot = k n + k_P n_P + slow terms,

    sigma_ddot = k n_dot = -(3 k^2 / a^2) d<R>/d sigma.

So sigma is a pendulum in the potential (3 k^2 / a^2) <R>: it librates about a MINIMUM sigma_c of <R>, with the
small-amplitude frequency

    omega0^2 = 3 k^2 <R>''(sigma_c) / a^2 = 3 k^2 n^2 mu_eff,   mu_eff = <R>'' a / (G M_sun) = mu_P a <D>''(sigma_c),

mu_P = m_P / M_sun, a and the lengths of <D> in au (n^2 a^3 = G M_sun). This is the convention of
`strength.mu_eff_required` (omega0 = |k| n sqrt(3 mu_eff), M&D §8.7 eqs. 8.53-8.58); for a single term
<R> = 2 C cos sigma it is mu_eff = 2 |C| a, the convention of `multiplet.mu_eff`. Not the first harmonic of <R>: the
small-amplitude period is set by the curvature of the well the body occupies, which the higher harmonics change
(15875 2N-3: required/theory 1.00 by the curvature, 0.18 by the first harmonic; independent Codex check 1.0015).

Quadrature
----------
lambda_P runs over L nodes per turn, N = |k| L nodes in all (the trapezoidal rule, exponentially convergent for a
smooth periodic integrand). Along this line the body's mean anomaly takes N distinct values that are a permutation of
a uniform N-point circle, and a step of sigma by 2 pi / L is an index shift of that circle: the body and the planet
are placed once (one Kepler solve each) and every sigma_m = 2 pi m / L costs N kernel evaluations (the trick of
`coorbital.averaged_table`). The even nodes are the same rule with L / 2 nodes per turn, so every value comes with
its half-resolution twin at no cost: the curvature is converged when the two agree within the tolerance, else L is
doubled up to its maximum. Where the orbits nearly meet, the peak of 1/|r - r_P| at the conjunction is narrow and
needs many nodes; that is what the doubling buys, and why an unconverged curvature is reported, not used.

The occupied centre: the minimum of <D> whose basin holds the measured centre of the libration (steepest descent on a
coarse sigma grid from the grid point nearest to it: an asymmetric or secondary well is a well of its own, and a
minimum across a maximum is never taken), then the curvature from a quadratic fit over +-half_width about the discrete minimum of a fine grid
(2 pi / L steps). A centre where the body passes within a few Hill radii of the planet is outside the averaged
problem (an encounter is not a secular effect): the caller compares `min_distance_au` with its own limit.

The well, not only its minimum. The descent from the measured centre always ends in some minimum, so a minimum alone
is no evidence that the body occupies it. A libration in this potential contains its stable centre and never passes
the top of <R>: `swing_in_well` tests an observed half-cycle against the occupied centre and the global maximum of <D>
on the coarse grid (`CentreCurvature.global_max_rad`). A horseshoe of an exterior 1:n around both asymmetric wells
crosses the inner maximum near 180 deg, not the global one (Gallardo 2006, sect. 3.1), and passes. 482824 19N-32 does
not: the measured centre is 178 deg from the minimum and every half-cycle crosses the global maximum near 350 deg.

Pure: arrays and dataclasses in, no Body/Simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence, Tuple

import mpmath as mp
import numpy as np
from mpmath import libmp
from scipy.interpolate import CubicSpline

from resonances.resonance.analysis.mmr.coorbital import Orbit, kepler_position

NAN = float('nan')
TWO_PI = 2 * np.pi
CHUNK_ELEMENTS = 1 << 20  # kernel values per vectorised block (memory: ~10 arrays of this many doubles)
MIN_FIT_POINTS = 3  # a quadratic needs three points
EXTREME_REFINE_POINTS = 41  # spline samples over +-one coarse step about a discrete extreme (numerical)
# Gauss-Legendre nodes of the period integral in theta (sigma = middle + half sin theta: smooth, no end singularity);
# numerical resolution: 32 nodes already agree with the analytic pendulum to the cubic spline's 1e-6 up to 3.1 rad
GAUSS_X, GAUSS_W = np.polynomial.legendre.leggauss(64)
GAUSS_THETA, GAUSS_WEIGHTS = GAUSS_X * np.pi / 2, GAUSS_W * np.pi / 2

REASON_NONE = ''
REASON_NO_CENTRE = 'no_stable_centre'  # <D> has no local minimum, or a non-positive curvature at it
REASON_NOT_CONVERGED = 'quadrature_not_converged'  # L and L/2 disagree even at the largest L
REASON_NON_FINITE = 'non_finite'  # a node fell on the planet (an exact collision of the two ellipses)


@dataclass(frozen=True)
class Ellipse:
    """A Keplerian orbit in the reference frame: a [au] and the shape and orientation (`coorbital.Orbit`, angles in rad).
    For the body the mean anomaly is the argument; for the planet the mean longitude lambda_P = M_P + `Orbit.phase`."""

    a: float
    orbit: Orbit


@dataclass(frozen=True)
class Quadrature:
    """Resolution of the averaging (numerical, not thresholds)."""

    nodes_per_turn: int  # L: nodes of lambda_P per turn at the first attempt
    max_nodes_per_turn: int  # L is doubled up to this while the curvature is not converged
    sigma_grid: int  # coarse sigma grid that finds the wells
    half_width_rad: float  # the quadratic fit spans +- this about the centre
    tolerance: float  # |curvature(L) - curvature(L/2)| <= tolerance x |curvature(L)|


@dataclass(frozen=True)
class CentreCurvature:
    """<D>'' at the occupied centre [au^-1 rad^-2] (D per unit G m_P; mu_eff = mu_P a curvature)."""

    curvature: float
    curvature_half: float  # the same with L / 2 nodes per turn
    centre_rad: float  # the vertex of the fit, in [0, 2 pi)
    offset_rad: float  # circular distance from the measured centre
    min_distance_au: float  # closest approach of the body to the planet along the average at the centre
    nodes_per_turn: int  # L of the result
    reason: str  # '' when measured, else REASON_*
    global_max_rad: float = NAN  # the top of <D> on the coarse sigma grid, in [0, 2 pi) (`swing_in_well`)
    level: float = NAN  # D0, the mean of <D> over the coarse sigma grid (the scale double precision resolves against)

    @property
    def measured(self) -> bool:
        return self.reason == REASON_NONE and bool(np.isfinite(self.curvature)) and self.curvature > 0


def wrap_two_pi(angle: float) -> float:
    return float(np.mod(angle, TWO_PI))


def circular_distance(x, y):
    """|x - y| on the circle, in [0, pi]."""
    return np.abs(np.angle(np.exp(1j * (np.asarray(x, dtype=float) - y))))


class LineAverage:
    """<D>(sigma_m), sigma_m = 2 pi m / L, along the line of one (body, planet, k, k_P, phase): the nodes are placed once.

    `evaluate(m)` returns (<D> with L nodes per turn, <D> with L / 2, the closest approach [au]) for integer m."""

    def __init__(
        self, body: Ellipse, planet: Ellipse, body_coefficient: int, planet_coefficient: int, phase_rad: float, nodes_per_turn: int
    ):
        k, k_p = int(body_coefficient), int(planet_coefficient)
        if k == 0:
            raise ValueError('the angle holds no mean longitude of the body (k = 0)')
        self.nodes_per_turn = int(nodes_per_turn)
        turns = abs(k)
        self.sign = 1 if k > 0 else -1
        self.n_nodes = turns * self.nodes_per_turn
        j = np.arange(self.n_nodes)
        self.body_index = np.mod(-self.sign * k_p * j, self.n_nodes)  # node j: body at (index + sign m) mod N
        lambda_p = TWO_PI * np.arange(self.nodes_per_turn) / self.nodes_per_turn
        r_planet = planet.a * kepler_position(planet.orbit, lambda_p - planet.orbit.phase)[0]
        r_planet = r_planet[np.mod(j, self.nodes_per_turn)]  # (N, 3): the planet repeats every L nodes
        self.planet_xyz = r_planet.T.copy()
        self.planet_gradient = (r_planet / (np.sum(r_planet**2, axis=1) ** 1.5)[:, None]).T.copy()  # r_P / r_P^3
        mean_anomaly0 = -float(phase_rad) / k  # sigma = 0
        anomalies = mean_anomaly0 + TWO_PI * np.arange(self.n_nodes) / self.n_nodes
        self.body_xyz = (body.a * kepler_position(body.orbit, anomalies)[0]).T.copy()

    def sigma(self, m) -> np.ndarray:
        return TWO_PI * np.asarray(m, dtype=float) / self.nodes_per_turn

    def evaluate(self, m) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = np.atleast_1d(np.asarray(m, dtype=np.int64))
        full, half, closest = np.empty(len(m)), np.empty(len(m)), np.empty(len(m))
        xp, yp, zp = self.planet_xyz
        gx, gy, gz = self.planet_gradient
        chunk = max(1, CHUNK_ELEMENTS // self.n_nodes)
        with np.errstate(divide='ignore', invalid='ignore'):
            for first in range(0, len(m), chunk):
                block = m[first : first + chunk]
                index = np.mod(self.body_index[None, :] + self.sign * block[:, None], self.n_nodes)
                xb, yb, zb = (coordinate[index] for coordinate in self.body_xyz)
                distance = np.sqrt((xb - xp) ** 2 + (yb - yp) ** 2 + (zb - zp) ** 2)
                kernel = 1.0 / distance - (xb * gx + yb * gy + zb * gz)
                rows = slice(first, first + len(block))
                full[rows] = np.mean(kernel, axis=1)
                half[rows] = np.mean(kernel[:, ::2], axis=1)
                closest[rows] = np.min(distance, axis=1)
        return full, half, closest


def swing_in_well(centre_rad: float, global_max_rad: float, lo_rad: float, hi_rad: float) -> bool:
    """The observed half-cycle [lo, hi] (the unwrapped sigma of its two turning points, lo <= hi) is a libration in
    the well of `centre_rad`: it contains that stable centre and does not pass the global maximum of <D> (module
    docstring; a horseshoe crosses only an inner maximum). Both angles are placed on the unwrapped axis nearest the
    middle of the swing. False when the centre or the maximum is not finite."""
    if not (np.isfinite(centre_rad) and np.isfinite(global_max_rad)):
        return False
    middle = (lo_rad + hi_rad) / 2

    def near_middle(angle: float) -> float:
        return angle + TWO_PI * np.round((middle - angle) / TWO_PI)

    return bool(lo_rad <= near_middle(centre_rad) <= hi_rad and not lo_rad < near_middle(global_max_rad) < hi_rad)


def basin_minimum(values: np.ndarray, start: int) -> int:
    """The local minimum of a periodic sequence reached from index `start` by steepest descent (the well whose basin
    holds `start`); -1 when the descent finds no finite minimum."""
    n = len(values)
    i = int(start) % n
    for _ in range(n):
        left, right = values[(i - 1) % n], values[(i + 1) % n]
        lower = (i - 1) % n if left < right else (i + 1) % n
        if not values[lower] < values[i]:
            return i if np.isfinite(values[i]) else -1
        i = lower
    return -1


def _quadratic_curvature(sigma: np.ndarray, values: np.ndarray, centre: float) -> Tuple[float, float]:
    """(second derivative, vertex) of the least-squares quadratic through (sigma, values)."""
    if len(sigma) < MIN_FIT_POINTS or not np.all(np.isfinite(values)):
        return NAN, NAN
    c2, c1, _ = np.polyfit(sigma - centre, values, 2)
    vertex = centre - c1 / (2 * c2) if c2 != 0 else centre
    return float(2 * c2), float(vertex)


def _vertex_fit(
    line: LineAverage, node: int, step: int, quadrature: Quadrature, reference_rad: float, top: float = NAN, level: float = NAN
) -> CentreCurvature:
    """The vertex and curvature of the quadratic fits over +-half_width about the discrete minimum of the fine grid
    within +-`step` nodes of `node` (L and L / 2: `curvature_half`); `offset_rad` is the vertex's distance from
    `reference_rad`, `top` and `level` are passed through. non_finite when a fine node falls on the planet."""
    nodes = line.nodes_per_turn
    width = max(1, int(round(quadrature.half_width_rad * nodes / TWO_PI)))
    offsets = np.arange(-(step + width), step + width + 1)
    fine = node + offsets
    full, half, closest = line.evaluate(fine)
    if not (np.all(np.isfinite(full)) and np.all(np.isfinite(half))):
        return CentreCurvature(NAN, NAN, NAN, NAN, NAN, nodes, REASON_NON_FINITE, top, level)
    inner = np.abs(offsets) <= step
    best = int(np.flatnonzero(inner)[np.argmin(full[inner])])
    span = slice(best - width, best + width + 1)
    sigma = line.sigma(fine[span])
    curvature, vertex = _quadratic_curvature(sigma, full[span], float(line.sigma(fine[best])))
    curvature_half, _ = _quadratic_curvature(sigma, half[span], float(line.sigma(fine[best])))
    centre = wrap_two_pi(vertex) if np.isfinite(vertex) else NAN
    offset = float(circular_distance(centre, reference_rad)) if np.isfinite(centre) else NAN
    reason = REASON_NONE if np.isfinite(curvature) and curvature > 0 else REASON_NO_CENTRE
    return CentreCurvature(curvature, curvature_half, centre, offset, float(closest[best]), nodes, reason, top, level)


def _curvature_at(line: LineAverage, sigma_observed_rad: float, quadrature: Quadrature) -> CentreCurvature:
    """One attempt at L = line.nodes_per_turn (module docstring)."""
    nodes = line.nodes_per_turn
    coarse = np.unique(np.round(np.arange(quadrature.sigma_grid) * nodes / quadrature.sigma_grid).astype(np.int64))
    values, _, _ = line.evaluate(coarse)
    values = np.where(np.isfinite(values), values, np.inf)  # a node on the planet is a peak, never a well
    top = wrap_two_pi(float(line.sigma(coarse[int(np.argmax(values))])))  # a node on the planet (+inf) is the top
    well = basin_minimum(values, int(np.argmin(circular_distance(line.sigma(coarse), sigma_observed_rad))))
    level = float(np.mean(values)) if np.all(np.isfinite(values)) else NAN
    if well < 0:
        return CentreCurvature(NAN, NAN, NAN, NAN, NAN, nodes, REASON_NO_CENTRE, top, level)
    step = int(np.ceil(nodes / quadrature.sigma_grid))  # the coarse spacing in fine steps
    return _vertex_fit(line, int(coarse[well]), step, quadrature, sigma_observed_rad, top, level)


def local_minimum(line: LineAverage, start_rad: float, quadrature: Quadrature) -> CentreCurvature:
    """The minimum of <D> reached from `start_rad` by steepest descent in steps of the coarse grid
    (2 pi / quadrature.sigma_grid), then the vertex and the curvature of the quadratic fit over +-half_width about the
    discrete minimum of the fine grid, as `_curvature_at` does: the same well, found without evaluating the whole coarse
    circle (the moving centre follows a well along a record from one node to the next, `centre_track`). The descent
    stops after one turn without a minimum (no_stable_centre). `global_max_rad` is not measured (nan); `offset_rad` is
    the distance of the vertex from `start_rad`. One attempt at L = line.nodes_per_turn: the track needs the place of
    the minimum, not a converged curvature."""
    nodes = line.nodes_per_turn
    step = max(1, int(round(nodes / quadrature.sigma_grid)))
    here = int(round(float(start_rad) * nodes / TWO_PI))

    def value(m: int) -> float:
        v = float(line.evaluate([m])[0][0])
        return v if np.isfinite(v) else np.inf

    current = value(here)
    left, right = value(here - step), value(here + step)
    direction = -1 if left < right else 1
    walked = 0
    while walked * step < nodes:
        candidate = value(here + direction * step)
        if not candidate < current:
            break
        here, current, walked = here + direction * step, candidate, walked + 1
    else:
        return CentreCurvature(NAN, NAN, NAN, NAN, NAN, nodes, REASON_NO_CENTRE)
    return _vertex_fit(line, here, step, quadrature, start_rad)


def converged(result: CentreCurvature, tolerance: float) -> bool:
    """L and L / 2 nodes per turn give the same curvature within `tolerance` (relative)."""
    return bool(
        np.isfinite(result.curvature)
        and np.isfinite(result.curvature_half)
        and abs(result.curvature - result.curvature_half) <= tolerance * abs(result.curvature)
    )


def centre_curvature(
    body: Ellipse,
    planet: Ellipse,
    body_coefficient: int,
    planet_coefficient: int,
    phase_rad: float,
    sigma_observed_rad: float,
    quadrature: Quadrature,
) -> CentreCurvature:
    """<D>'' at the minimum of <D>(sigma) whose basin holds `sigma_observed_rad` (module docstring), L doubled from
    quadrature.nodes_per_turn up to quadrature.max_nodes_per_turn until converged; reason 'quadrature_not_converged'
    when even the largest L is not."""
    nodes = int(quadrature.nodes_per_turn)
    while True:
        line = LineAverage(body, planet, body_coefficient, planet_coefficient, phase_rad, nodes)
        result = _curvature_at(line, sigma_observed_rad, quadrature)
        if result.reason == REASON_NON_FINITE or converged(result, quadrature.tolerance):
            return result
        if 2 * nodes > quadrature.max_nodes_per_turn:
            break
        nodes *= 2
    return replace(result, reason=REASON_NOT_CONVERGED) if result.reason == REASON_NONE else result


def period_in_profile(sigma: np.ndarray, values: np.ndarray, start_rad: float) -> Tuple[float, float]:
    """(period [yr], far turning point [rad]) of sigma_ddot = -V'(sigma) started at rest at `start_rad`
    (`orbit_in_profile` with no kinetic energy; the far turning point is the bound that is not the start)."""
    period, lower, upper = orbit_in_profile(sigma, values, start_rad)
    if not np.isfinite(lower):
        return period, NAN
    return period, upper if abs(upper - start_rad) >= abs(lower - start_rad) else lower


def orbit_in_profile(sigma: np.ndarray, values: np.ndarray, start_rad: float, kinetic: float = 0.0) -> Tuple[float, float, float]:
    """(period [yr], lower and upper turning point [rad]) of the orbit of sigma_ddot = -V'(sigma) that passes
    `start_rad` with the kinetic energy `kinetic` = sigma_dot^2 / 2 [rad^2/yr^2], V [rad^2/yr^2] given at increasing
    `sigma` (a cubic spline through them): the energy E = V(start) + kinetic, the turning points the nearest roots of
    V = E on either side where V rises through E along the motion (the spline's own roots: a swing narrower than the
    grid step is found too), and T = 2 int dsigma / sqrt(2 (E - V)) between them, with sigma = middle + half sin(theta)
    (no end singularity). Started at rest (kinetic 0) the start is one turning point and the motion goes downhill. On
    either side the orbit passes lower tops (a well of several minima: Codex W18, V = -cos x + 0.5 cos 2x). (inf, nan,
    nan) when a side has no such root to the end of the segment (a start at rest at the very top of a periodic V, or
    an energy above every top: the separatrix or a turn); (nan, nan, nan) at rest at an extremum of V (no
    downhill side) or on too short a segment. A start on a whole turn of a periodic V always has its turning points
    within one turn: the orbit of that energy, over lower tops if any."""
    sigma, values = np.asarray(sigma, dtype=float), np.asarray(values, dtype=float)
    blank = (NAN, NAN, NAN)
    if len(sigma) < MIN_FIT_POINTS + 1 or not np.all(np.isfinite(values)) or not sigma[0] < start_rad < sigma[-1]:
        return blank
    if not kinetic >= 0:
        return blank
    spline = CubicSpline(sigma, values)
    energy = float(spline(start_rad)) + float(kinetic)
    roots = np.asarray(spline.solve(energy, extrapolate=False), dtype=float)
    slopes = np.asarray(spline(roots, 1), dtype=float)
    slope = float(spline(start_rad, 1))
    if not slope != 0:
        if not kinetic > 0:
            return blank
        # through the bottom: V rises through E going left at the lower bound, going right at the upper one
        below, above = roots[(roots < start_rad) & (slopes < 0)], roots[(roots > start_rad) & (slopes > 0)]
        if not len(below) or not len(above):
            return np.inf, NAN, NAN
        lower, upper = float(below.max()), float(above.min())
    else:
        direction = -1.0 if slope > 0 else 1.0  # downhill
        # the far turning point: V rises through E along the motion (the start itself, a root where V falls, never counts)
        ahead = roots[(direction * (roots - start_rad) > 0) & (direction * slopes > 0)]
        if not len(ahead):
            return np.inf, NAN, NAN
        far = float(ahead[np.argmin(np.abs(ahead - start_rad))])
        near = _near_turning_point(spline, sigma, roots, slopes, start_rad, direction, energy) if kinetic > 0 else float(start_rad)
        if not np.isfinite(near):
            return np.inf, NAN, NAN
        lower, upper = sorted((near, far))
    middle, half = (upper + lower) / 2, (upper - lower) / 2
    theta = GAUSS_THETA
    speed_sq = 2 * (energy - spline(middle + half * np.sin(theta)))
    if not np.all(speed_sq > 0):  # the spline wiggles above E inside the swing: no orbit of this energy here
        return blank
    return float(2 * np.sum(GAUSS_WEIGHTS * half * np.cos(theta) / np.sqrt(speed_sq))), lower, upper


def _near_turning_point(
    spline: CubicSpline, sigma: np.ndarray, roots: np.ndarray, slopes: np.ndarray, start_rad: float, direction: float, energy: float
) -> float:
    """The turning point behind a start that moves downhill (`direction`) with energy E above V(start)
    (`orbit_in_profile`): uphill behind the start V rises monotonically to the nearest top; below that top the orbit
    turns at the one root between them (the start itself when E - V(start) is below the rounding of V), above it at the
    first root beyond where V rises through E (over a lower top); inf when there is none to the end of the segment."""
    critical = np.asarray(spline.derivative().roots(extrapolate=False), dtype=float)
    behind = critical[direction * (critical - start_rad) < 0]
    top = float(behind[np.argmin(np.abs(behind - start_rad))]) if len(behind) else float(sigma[0] if direction > 0 else sigma[-1])
    if energy > float(spline(top)):
        beyond = roots[(direction * (roots - top) < 0) & (direction * slopes < 0)]
        return float(beyond[np.argmin(np.abs(beyond - start_rad))]) if len(beyond) else np.inf
    between = roots[(direction * (roots - start_rad) <= 0) & (direction * (roots - top) >= 0)]
    return float(between[np.argmin(np.abs(between - start_rad))]) if len(between) else float(start_rad)


@dataclass(frozen=True)
class WellPeriods:
    """`periods_from_turning_points`: the periods [yr] of the orbits through the starts, why they are
    missing (REASON_*), the closest approach along the swings [au] and whether the swing [min start, max start] librates
    in the well whose basin holds its middle (`swing_in_well`, the test of the strength check)."""

    periods: np.ndarray
    reason: str
    closest_au: float
    in_well: bool


def periods_from_turning_points(
    make_line: Callable[[int], LineAverage],
    starts_rad: Sequence[float],
    scale: float,
    quadrature: Quadrature,
    kinetic: Optional[Sequence[float]] = None,
) -> WellPeriods:
    """The orbits through each of `starts_rad` (unwrapped, one swing) in the full well V = scale x <D>(sigma)
    (`scale` = 3 k^2 mu_P n^2 a [rad^2/yr^2 per au^-1]: sigma_ddot = -(3 k^2 / a^2) d<R>/dsigma, module docstring),
    at rest there or with the kinetic energies `kinetic` [rad^2/yr^2] (`orbit_in_profile`), `make_line(L)` the line at
    L nodes per turn (`WellPeriods`).

    V is evaluated on the coarse grid (2 pi / sigma_grid) over the whole turn and `orbit_in_profile` integrates its
    cubic spline (three turns, so that any start lies inside): a smooth well needs no finer grid (the spline error of a
    cosine at 5 deg is 1e-7 of its depth), and the sharp peak where the orbits nearly meet lies at the conjunction, not
    in the well. The same with the L / 2 twin, and L doubled up to max_nodes_per_turn until the two give every period
    within the tolerance; a doubled L is evaluated only on the coarse nodes of the swings found (one more node at each
    side): the cost of L grows as |k| L, and nearly crossing orbits need it (2013 AS27 31E-36). A period is inf for a
    start at the very top of <D> (the separatrix), nan where V does not fall from its start; the reason is non_finite
    for a node on the planet, quadrature_not_converged, no_stable_centre when no period is finite."""
    nodes = int(quadrature.nodes_per_turn)
    kinetic = np.zeros(len(starts_rad)) if kinetic is None else np.asarray(kinetic, dtype=float)
    result, bounds = _periods_on_line(make_line(nodes), starts_rad, scale, quadrature, kinetic)
    span = _swings_span(starts_rad, bounds)
    while result.reason == REASON_NOT_CONVERGED and 2 * nodes <= quadrature.max_nodes_per_turn and span is not None:
        nodes *= 2
        refined, _ = _periods_on_line(make_line(nodes), starts_rad, scale, quadrature, kinetic, span)
        result = replace(refined, in_well=result.in_well)  # the well's shape is read on the whole turn, at the first L
    return result


def _swings_span(starts_rad: Sequence[float], bounds: np.ndarray) -> Optional[Tuple[float, float]]:
    """The unwrapped interval of every start and its turning points (None when some turning point is missing)."""
    bounds = np.asarray(bounds, dtype=float)
    if not bounds.size or not np.all(np.isfinite(bounds)):
        return None
    points = np.concatenate([np.asarray(starts_rad, dtype=float), bounds.ravel()])
    return float(points.min()), float(points.max())


def _periods_on_line(
    line: LineAverage,
    starts_rad: Sequence[float],
    scale: float,
    quadrature: Quadrature,
    kinetic: np.ndarray,
    span: Optional[Tuple[float, float]] = None,
) -> Tuple[WellPeriods, np.ndarray]:
    """(`WellPeriods`, the (lower, upper) turning points of each orbit on the starts' branch) on one line: the whole
    turn (three turns tiled), or only the coarse nodes of `span` and one beyond each end (in_well False there: the
    caller keeps the first one's)."""
    nodes = line.nodes_per_turn
    blank = np.full(len(starts_rad), NAN)
    blank_bounds = np.full((len(starts_rad), 2), NAN)
    if span is None:
        coarse = np.unique(np.round(np.arange(quadrature.sigma_grid) * nodes / quadrature.sigma_grid).astype(np.int64))
    else:
        step = TWO_PI / quadrature.sigma_grid
        cells = np.arange(int(np.floor(span[0] / step)) - 1, int(np.ceil(span[1] / step)) + 2)
        coarse = np.unique(np.round(cells * nodes / quadrature.sigma_grid).astype(np.int64))
    full, half, closest = line.evaluate(coarse)
    if not (np.all(np.isfinite(full)) and np.all(np.isfinite(half))):
        return WellPeriods(blank, REASON_NON_FINITE, float(np.min(closest)), False), blank_bounds
    sigma = line.sigma(coarse)
    in_well = False
    if span is None:
        lo, hi = float(np.min(starts_rad)), float(np.max(starts_rad))
        well = basin_minimum(full, int(np.argmin(circular_distance(sigma, (lo + hi) / 2))))
        in_well = well >= 0 and swing_in_well(float(sigma[well]), float(sigma[int(np.argmax(full))]), lo, hi)
        offset = [float(sigma[0] + np.mod(start - sigma[0], TWO_PI)) - float(start) for start in starts_rad]
        grid, values, twin_values = np.concatenate([sigma - TWO_PI, sigma, sigma + TWO_PI]), np.tile(full, 3), np.tile(half, 3)
    else:
        offset = [0.0] * len(starts_rad)  # the span's nodes are on the starts' own branch
        grid, values, twin_values = sigma, full, half
    periods, twins, bounds, nearest = blank.copy(), blank.copy(), blank_bounds.copy(), np.inf
    for i, start in enumerate(starts_rad):
        inside = float(start) + offset[i]  # on the middle turn of the tiled profile
        periods[i], lower, upper = orbit_in_profile(grid, scale * values, inside, kinetic[i])
        twins[i], _, _ = orbit_in_profile(grid, scale * twin_values, inside, kinetic[i])
        if np.isfinite(lower) and np.isfinite(upper):
            bounds[i] = (lower - offset[i], upper - offset[i])
            swing = (np.mod(sigma - lower, TWO_PI) <= upper - lower) | (circular_distance(sigma, inside) <= TWO_PI / quadrature.sigma_grid)
            nearest = min(nearest, float(np.min(closest[swing]))) if np.any(swing) else nearest
    measured = np.isfinite(periods)
    if not np.any(measured):
        return WellPeriods(periods, REASON_NO_CENTRE, nearest, in_well), bounds
    agree = np.abs(periods[measured] - twins[measured]) <= quadrature.tolerance * periods[measured]
    if not np.all(agree):  # a non-finite twin is no agreement either
        return WellPeriods(periods, REASON_NOT_CONVERGED, nearest, in_well), bounds
    return WellPeriods(periods, REASON_NONE, nearest, in_well), bounds


def energy_in_well(sigma: np.ndarray, values: np.ndarray, point_rad: float, middle_rad: float) -> float:
    """h = (<D>(point) - <D>_min) / (<D>_barrier - <D>_min) of a turning point at `point_rad` in the well whose basin
    holds `middle_rad` (the middle of its swing), from <D> on a coarse periodic grid `sigma` (increasing, one turn):
    0 at the bottom, 1 at the barrier, the lower of the two tops that bound the basin (a turning point of a libration
    lies below it). The bottom and the barrier are refined on the periodic cubic spline through the grid, within one
    grid step of the discrete extremes. nan without a finite profile, a basin or a positive depth."""
    sigma, values = np.asarray(sigma, dtype=float), np.asarray(values, dtype=float)
    n = len(values)
    if n < MIN_FIT_POINTS + 1 or not np.all(np.isfinite(values)):
        return NAN
    bottom = basin_minimum(values, int(np.argmin(circular_distance(sigma, middle_rad))))
    if bottom < 0:
        return NAN

    def uphill(direction: int) -> int:
        i = bottom
        for _ in range(n):
            following = (i + direction) % n
            if not values[following] > values[i]:
                return i
            i = following
        return i

    step = TWO_PI / n
    spline = CubicSpline(np.append(sigma, sigma[0] + TWO_PI), np.append(values, values[0]), bc_type='periodic')

    def refined(index: int, sign: float) -> float:
        near = np.mod(sigma[index] + np.linspace(-step, step, EXTREME_REFINE_POINTS), TWO_PI)
        return float(sign * np.max(sign * spline(near)))

    low = refined(bottom, -1.0)
    barrier = min(refined(uphill(-1), 1.0), refined(uphill(1), 1.0))
    depth = barrier - low
    if not depth > 0:
        return NAN
    return float((spline(np.mod(point_rad, TWO_PI)) - low) / depth)


def averaged_profile(
    body: Ellipse, planet: Ellipse, body_coefficient: int, planet_coefficient: int, phase_rad: float, nodes_per_turn: int
) -> Tuple[np.ndarray, np.ndarray]:
    """(sigma [rad], <D>(sigma) [au^-1]) on the full grid sigma_m = 2 pi m / L (diagnostics and tests)."""
    line = LineAverage(body, planet, body_coefficient, planet_coefficient, phase_rad, nodes_per_turn)
    m = np.arange(line.nodes_per_turn)
    return line.sigma(m), line.evaluate(m)[0]


# --- the resonant harmonic in extended precision -------------------------------------------------------------------
#
# A high-order term is far below the rounding of <D>: its harmonic is ~ e^q of D0 (2897 35S-4: 2|C| / |D0| ~ 1e-39), so
# the double-precision profile has no resolvable curvature (no_stable_centre, or a curvature that is rounding noise).
# Its strength is then the resonant harmonic itself, C = < D exp(-i (k_P lambda_P + k M)) > over the torus of the two
# mean anomalies: <D>(sigma) = D0 + 2 |C| cos(sigma - phase + arg C) + higher harmonics (which are of order e^(2q)),
# so the curvature at the minimum is 2 |C| and mu_eff = mu_P a 2 |C| (the convention of `centre_curvature`). C is a
# 2-D trapezoidal DFT over N_M body anomalies and N_L planet longitudes in mpmath at `dps` digits: the sum cancels to
# |C| / |D0|, so the digits must exceed log10(|D0| / |C|) by a guard. The trapezoidal error is the aliased harmonics
# (k_P + r N_L, k + s N_M): a grid shifted by half a node in one angle turns an alias by (-1)^r (planet) or (-1)^s
# (body). Two such twins, one per angle, each compared with the base grid, see every alias with r or s odd, the mixed
# ones (+-1, -+1) included, which a single twin shifted in both angles turns by (-1)^(r+s) and so misses (Codex W20:
# with N_M = N_L = 8 the 2:1 is 2.1 % off while that twin agreed within 2 %). The mean of the two twins cancels the
# single-axis aliases, leaving the mixed ones (r, s both odd) with their sign flipped. The half-node twin of
# `LineAverage` (N / 2 nodes) would not do here: N_L / 2 can alias k_P - N_L / 2 onto an order close to q.

KEPLER_MAX_ITERATIONS = 200  # Newton converges quadratically; this only bounds a pathological start
REASON_PRECISION = 'precision_exhausted'  # |C| / |D0| below 10^(guard - dps) even at the largest dps


@dataclass(frozen=True)
class HarmonicRule:
    """Precision and nodes of `harmonic_amplitude_mp` (numerical, not thresholds)."""

    dps: int  # decimal digits of the first attempt
    dps_max: int  # doubled up to this while |C| / |D0| < 10^(guard_digits - dps)
    guard_digits: int
    nodes_body: int  # N_M, body mean anomalies
    nodes_planet: int  # N_L, planet mean longitudes
    tolerance: float  # |C(grid) - C(twin)| <= tolerance x |C| for both one-axis twins


@dataclass(frozen=True)
class HarmonicAmplitude:
    """|C| of the resonant harmonic [au^-1] (D per unit G m_P), D0, the minimum of the first harmonic and how they were
    obtained; `curvature` = 2 |C| is <D>'' at that minimum."""

    value: float
    mean: float  # D0
    centre_rad: float  # sigma_c = phase - arg C + pi, in [0, 2 pi)
    nodes_body: int
    nodes_planet: int
    dps: int
    converged: bool  # both twins agree with the grid within the tolerance
    reason: str  # '' when measured, else REASON_*

    @property
    def curvature(self) -> float:
        return 2 * self.value

    @property
    def measured(self) -> bool:
        return self.reason == REASON_NONE and bool(np.isfinite(self.value)) and self.value > 0


def _mp_positions(ellipse: Ellipse, anomalies) -> list:
    """Heliocentric positions [au] of `ellipse` at the mean anomalies (mpf), at the working precision."""
    orbit = ellipse.orbit
    a, e = mp.mpf(ellipse.a), mp.mpf(orbit.e)
    ci, si = mp.cos(mp.mpf(orbit.inc)), mp.sin(mp.mpf(orbit.inc))
    cn, sn = mp.cos(mp.mpf(orbit.Omega)), mp.sin(mp.mpf(orbit.Omega))
    cw, sw = mp.cos(mp.mpf(orbit.omega)), mp.sin(mp.mpf(orbit.omega))
    p = (cn * cw - sn * sw * ci, sn * cw + cn * sw * ci, sw * si)
    q = (-cn * sw - sn * cw * ci, -sn * sw + cn * cw * ci, cw * si)
    root = mp.sqrt(1 - e * e)
    tiny = mp.mpf(2) ** (-mp.mp.prec + 8)
    out = []
    for mean_anomaly in anomalies:
        big_e = mean_anomaly + e * mp.sin(mean_anomaly)
        for _ in range(KEPLER_MAX_ITERATIONS):  # Newton on Kepler's equation from M + e sin M (e < 1)
            step = (big_e - e * mp.sin(big_e) - mean_anomaly) / (1 - e * mp.cos(big_e))
            big_e -= step
            if abs(step) < tiny:
                break
        x, y = a * (mp.cos(big_e) - e), a * root * mp.sin(big_e)
        out.append((x * p[0] + y * q[0], x * p[1] + y * q[1], x * p[2] + y * q[2]))
    return out


def _harmonic_sum(
    body: Ellipse,
    planet: Ellipse,
    body_coefficient: int,
    planet_coefficient: int,
    rule: HarmonicRule,
    shift_body: float,
    shift_planet: float,
):
    """(C, D0) on the grid M_j = 2 pi (j + shift_body) / N_M, lambda_P,b = 2 pi (b + shift_planet) / N_L at the working
    precision.

    The indirect part r . sum_b (r_P / r_P^3) e^{-i k_P lambda_b} is summed once per body node; the direct part is the
    inner loop, in mpmath's low-level arithmetic (raw mpf tuples) to keep N_M N_L square roots fast."""
    n_body, n_planet = int(rule.nodes_body), int(rule.nodes_planet)
    two_pi = 2 * mp.pi
    anomalies = [two_pi * (j + mp.mpf(shift_body)) / n_body for j in range(n_body)]
    longitudes = [two_pi * (b + mp.mpf(shift_planet)) / n_planet for b in range(n_planet)]
    r_body = _mp_positions(body, anomalies)
    r_planet = _mp_positions(planet, [lam - mp.mpf(planet.orbit.phase) for lam in longitudes])
    cos_l = [mp.cos(planet_coefficient * lam) for lam in longitudes]
    sin_l = [-mp.sin(planet_coefficient * lam) for lam in longitudes]  # e^{-i k_P lambda}
    grad, grad_mean = [mp.mpc(0)] * 3, [mp.mpf(0)] * 3
    for b, xyz in enumerate(r_planet):
        r3 = (xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2) ** mp.mpf(1.5)
        turn = mp.mpc(cos_l[b], sin_l[b])
        grad = [grad[i] + xyz[i] / r3 * turn for i in range(3)]
        grad_mean = [grad_mean[i] + xyz[i] / r3 for i in range(3)]
    prec, rnd = mp.mp.prec, libmp.round_nearest
    sub, mul, add, sqrt, div = libmp.mpf_sub, libmp.mpf_mul, libmp.mpf_add, libmp.mpf_sqrt, libmp.mpf_div
    planet_raw = [(x._mpf_, y._mpf_, z._mpf_) for x, y, z in r_planet]
    cos_raw, sin_raw = [c._mpf_ for c in cos_l], [s._mpf_ for s in sin_l]
    total, level = mp.mpc(0), mp.mpf(0)
    for j, (x, y, z) in enumerate(r_body):
        bx, by, bz = x._mpf_, y._mpf_, z._mpf_
        re = im = mean = libmp.fzero
        for b in range(n_planet):
            px, py, pz = planet_raw[b]
            dx, dy, dz = sub(bx, px, prec), sub(by, py, prec), sub(bz, pz, prec)
            squared = add(add(mul(dx, dx, prec), mul(dy, dy, prec), prec), mul(dz, dz, prec), prec)
            inverse = div(libmp.fone, sqrt(squared, prec, rnd), prec, rnd)  # a node on the planet raises
            re, im = add(re, mul(inverse, cos_raw[b], prec), prec), add(im, mul(inverse, sin_raw[b], prec), prec)
            mean = add(mean, inverse, prec)
        inner = mp.mpc(mp.mpf(re), mp.mpf(im)) - (x * grad[0] + y * grad[1] + z * grad[2])
        total += inner * mp.expj(-body_coefficient * anomalies[j])
        level += mp.mpf(mean) - (x * grad_mean[0] + y * grad_mean[1] + z * grad_mean[2])
    return total / (n_body * n_planet), level / (n_body * n_planet)


def harmonic_amplitude_mp(
    body: Ellipse, planet: Ellipse, body_coefficient: int, planet_coefficient: int, phase_rad: float, rule: HarmonicRule
) -> HarmonicAmplitude:
    """|C| of the resonant harmonic exp(i (k_P lambda_P + k M)) of D in extended precision (block comment above);
    the digits doubled from rule.dps up to rule.dps_max while |C| / |D0| < 10^(guard - dps). Reasons:
    quadrature_not_converged (a twin and the base grid disagree), precision_exhausted, non_finite (a node on the planet).
    The value is the mean of the two one-axis twins (block comment above)."""
    k, k_p = int(body_coefficient), int(planet_coefficient)
    if k == 0:
        raise ValueError('the angle holds no mean longitude of the body (k = 0)')
    dps = int(rule.dps)
    while True:
        with mp.workdps(dps):
            try:
                grid, _ = _harmonic_sum(body, planet, k, k_p, rule, 0.0, 0.0)
                twin_body, level_body = _harmonic_sum(body, planet, k, k_p, rule, 0.5, 0.0)
                twin_planet, level_planet = _harmonic_sum(body, planet, k, k_p, rule, 0.0, 0.5)
            except ZeroDivisionError:
                return HarmonicAmplitude(NAN, NAN, NAN, rule.nodes_body, rule.nodes_planet, dps, False, REASON_NON_FINITE)
            coefficient, level = (twin_body + twin_planet) / 2, (level_body + level_planet) / 2
            resolved = abs(coefficient) >= abs(level) * mp.mpf(10) ** (int(rule.guard_digits) - dps)
            error = max(abs(grid - twin_body), abs(grid - twin_planet))
            converged = bool(error <= rule.tolerance * abs(coefficient))
            value, mean, argument = float(abs(coefficient)), float(level), float(mp.arg(coefficient))
        if resolved or dps >= rule.dps_max:
            break
        dps = min(2 * dps, int(rule.dps_max))
    centre = wrap_two_pi(float(phase_rad) - argument + np.pi)
    reason = REASON_NONE if resolved and converged else (REASON_PRECISION if not resolved else REASON_NOT_CONVERGED)
    return HarmonicAmplitude(value, mean, centre, rule.nodes_body, rule.nodes_planet, dps, converged, reason)
