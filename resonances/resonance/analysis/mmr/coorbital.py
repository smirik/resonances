"""The co-orbital (1:1) force law and energy integral from the averaged disturbing function.

A harmonic law a_dot = sum_k A_k sin k sigma (k <= 2) cannot describe a 1:1 angle: the averaged co-orbital
potential is singular at sigma = 0 for e = i = 0 and far from harmonic for horseshoes, tadpoles and
quasi-satellites alike. Here the potential is the averaged direct + indirect disturbing function itself,
computed by quadrature at fixed theta = lambda - lambda_P, and the a-response of the record is compared with it.

Theory (sign and scale)
-----------------------
The disturbing function of the body perturbed by the planet P (Murray & Dermott 1999, eq. 6.13, p. 227):

    R = G m_P [ 1 / |r_P - r| - r . r_P / r_P^3 ]          (direct + indirect part)

Lagrange's equation for a (M&D eq. 6.145, p. 251): da/dt = (2 / (n a)) dR/d(epsilon); at first order in m_P the
derivative with respect to the mean longitude at epoch is the derivative with respect to lambda at fixed lambda_P.
Averaging over the fast angle lambda_P at fixed theta = lambda - lambda_P (both mean motions equal at 1:1),

    <R>(theta) = (1 / 2 pi) int R(lambda = theta + lambda_P, lambda_P) d lambda_P,
    da/dt = (2 / (n a)) d<R>/d theta = (2 / (n a)) <dR/d lambda>,                                   (1)

and dR/d lambda = grad_r R . dr/dM with dr/dM = v / n (Keplerian velocity over the mean motion; the elements e, i,
omega, Omega are held fixed). With theta_dot = n - n_P = -(3/2)(n / a) delta a, theta_ddot = -(3 / a^2) d<R>/d theta, so

    (1/2) theta_dot^2 + V(theta) = E,   V = (3 / a^2) <R>.                                            (2)

In units G m_P = mu n^2 a^3 (mu = m_P / M_sun, a = a_P: the guiding centre sits on the planet's orbit) and the
dimensionless averages R~ = <R> a / (G m_P), F~ = <dR/d lambda> a / (G m_P):

    da/dt = 2 mu n a F~(theta)  [au/yr],   V = 3 mu n^2 R~(theta)  [rad^2/yr^2].

For e = i = 0 the average is R itself: R~ = 1 / (2 |sin(theta/2)|) - cos theta, and (2) is M&D eq. (3.178)-(3.179)
(p. 105): (3/4) delta r^2 + mu H(theta) = const with H = 2 R~ - 2. Minimum of V at theta = 60 deg, 300 deg (L4, L5,
tadpoles), maximum at 180 deg (L3: horseshoes pass over it), V -> infinity at theta -> 0. An eccentric or inclined
orbit removes the singularity at 0 (the body's epicycle keeps it away from the planet: quasi-satellites librate about
theta = 0) and adds new equilibria (Namouni 1999, Icarus 137, 293; Namouni, Christou & Murray 1999, PRL 83, 2506;
Christou 2000, Icarus 144, 1; Brasser et al. 2004, Icarus 171, 102). The averaging fails inside the planet's Hill
sphere (an encounter is not a secular effect) and the average diverges (logarithmically) at an exact orbit crossing:
the direct term is softened, |d|^2 -> |d|^2 + eps^2 with eps = mmr_coorbital_softening_hill x r_H,
r_H = a_P (mu / 3)^(1/3) (0.1 r_H changes K by < 2 % against no softening; 1 r_H raises it by 10-20 % for the
horseshoes that approach the Earth closest).

Test
----
The body's and the planet's slowly varying elements are tabulated block by block (blocks of P_c): a table of R~ and
F~ on a theta grid is computed only when the eccentricity vector e exp(i omega) or the inclination vector
sin i exp(i Omega) of the body or the planet moved more than the retabulation tolerances since the table was made;
each sample interpolates its block's table (no n_samples x n_quadrature arrays; tables are cached for the second pass
of the analysis). The model (1) is evaluated along the UNFILTERED angle and then passed through the data's own
low-passes (P_c, then the response cutoff): F~ is sharply peaked where the body passes the planet, and smoothing theta
before the nonlinear evaluation flattens the peak (2002 AA29: R^2 0.25 instead of 0.985, the same finding as the
independent Codex check). a_dot_data = K a_dot_model + C by least squares over the response pieces (the accepted
episodes, else the window), in both time halves, gives R^2 and the scale K / K_theory (K_theory = 1: the model holds
the planet's mass; with the Moon integrated separately the Earth's pull is that of Earth + Moon, 1.012).

The energy integral (2) is fitted with a instead of theta_dot (`energy_fit`: squares formed before the filter), one
energy per libration cycle: it is a global statement and fails when E is not conserved within a cycle (encounters,
the drift of e, i, omega over a transient episode) even where the local law (1) holds sample by sample; R0 of a 1:1
angle (`potential_check`) therefore passes on the integral, is not measured when only the integral fails, and fails
when both fail. A circulating co-orbital follows (1) and (2) as well: the law tells the origin of an oscillation of
a (the planet's 1:1 force), not that the angle librates (the geometry does).

Retrograde 1/-1 (i > 90 deg)
----------------------------
The angle of `RetrogradeTwoBody` 1J-1r<k> is sigma = lambda* - lambda' - (2 - 2k) varpi* + 2k Omega with the retrograde
lambda* = M + omega - Omega, varpi* = omega - Omega (Morais & Namouni 2013, Celest. Mech. Dyn. Astr. 117, 405, sect. 2.1;
the 1/-1 angle lambda* - lambda' - 2 varpi* (k = 0) is of second order in e, sect. 2.2), i.e.

    sigma = M + (2k - 1) omega + Omega - lambda',

built from M, omega, Omega only (never from REBOUND's l or pomega, which are -lambda*, -varpi* there). The same
quadrature then holds with M = sigma + lambda' - ((2k - 1) omega + Omega) in place of M = theta + lambda_P - varpi
(`Orbit.omega_coefficient` = 2k - 1, 1 for the prograde lambda): <R>(sigma) is the direct + indirect R averaged over
lambda' at fixed sigma, the average of Huang et al. 2018 (MNRAS 481, 5401, eq. 11, p. 5404) for k = k' = 1. Lagrange's
da/dt = (2 / (n a)) dR/dM does not depend on the inclination, sigma_dot = n - n' + slow terms, so (1) and (2) keep
their form and scale. The tadpole/horseshoe/quasi-satellite names belong to the prograde problem: a retrograde angle
gets the regime 'retrograde' (Huang et al. 2018, sect. 3: the retrograde 1:1 librates about 0 or 180 deg only).

Literature check: M&D's horseshoes and tadpoles are sections 3.9-3.10 (pp. 97-107; section 3.13 is Hill's equations).

Pure: arrays and dataclasses in, no Body/Simulation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.geometry import wrap_pi
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

NAN = float('nan')
TWO_PI = 2 * np.pi
GM_SUN = 4 * np.pi**2  # au^3 / yr^2 in the package's (Gaussian) years, 2 pi REBOUND time units; the MMR analysis' one
KEPLER_ITERATIONS = 30  # Newton steps for E - e sin E = M; converged to machine precision for e < 0.99 long before
MIN_FIT_SAMPLES = 4
TABLE_CACHE_SIZE = 512  # tables kept for reuse (720 points: 12 kB each)
LAW_COORBITAL = 'coorbital'
# Mean J2000 ecliptic elements (e, i [deg], Omega [deg], varpi [deg]) of the planets, used only when the planet's
# elements are not saved: Standish (1992), JPL "Keplerian Elements for Approximate Positions of the Major Planets",
# Table 1 (1800-2050 AD); Earth = the Earth-Moon barycentre.
PLANET_MEAN_ORBITS_J2000 = {
    'Mercury': (0.20563593, 7.00497902, 48.33076593, 77.45779628),
    'Venus': (0.00677672, 3.39467605, 76.67984255, 131.60246718),
    'Earth': (0.01671123, 0.0, 0.0, 102.93768193),
    'Mars': (0.09339410, 1.84969142, 49.55953891, 336.05637041),
    'Jupiter': (0.04838624, 1.30439695, 100.47390909, 14.72847983),
    'Saturn': (0.05386179, 2.48599187, 113.66242448, 92.59887831),
    'Uranus': (0.04725744, 0.77263783, 74.01692503, 170.95427630),
    'Neptune': (0.00859048, 1.77004347, 131.78422574, 44.96476227),
}
# regimes of the angle theta = lambda - lambda_P over the response pieces (a column, no status)
REGIME_TADPOLE_L4, REGIME_TADPOLE_L5 = 'tadpole_L4', 'tadpole_L5'
REGIME_HORSESHOE, REGIME_QUASI_SATELLITE, REGIME_COMPOUND, REGIME_NONE = 'horseshoe', 'quasi_satellite', 'compound', ''
REGIME_RETROGRADE = 'retrograde'  # a 1/-1 angle: the prograde names do not apply (module doc)


@dataclass(frozen=True)
class Orbit:
    """Shape and orientation of a Keplerian orbit (a is the planet's: exact 1:1); angles in rad."""

    e: float
    inc: float
    omega: float  # argument of perihelion
    Omega: float  # longitude of the node
    omega_coefficient: int = 1  # c of the angle's longitude M + c omega + Omega: 1 = lambda, 2k - 1 for 1J-1r<k>

    @property
    def phase(self) -> float:
        """The longitude of the angle minus M: varpi (prograde), (2k - 1) omega + Omega (retrograde 1/-1)."""
        return self.omega_coefficient * self.omega + self.Omega

    def eccentricity_vector(self) -> np.ndarray:
        """e times the unit vector to the perihelion, in the reference frame (defined for e -> 0, any i)."""
        cO, sO, co, so, ci = np.cos(self.Omega), np.sin(self.Omega), np.cos(self.omega), np.sin(self.omega), np.cos(self.inc)
        return self.e * np.array([cO * co - sO * so * ci, sO * co + cO * so * ci, so * np.sin(self.inc)])

    def inclination_vector(self) -> np.ndarray:
        """The unit normal of the orbit plane (defined for i -> 0; tells i from 180 deg - i)."""
        si = np.sin(self.inc)
        return np.array([si * np.sin(self.Omega), -si * np.cos(self.Omega), np.cos(self.inc)])


@dataclass(frozen=True)
class CoorbitalTable:
    """R~(theta) and F~(theta) = dR~/dtheta on a periodic theta grid (theta = lambda - lambda_P, rad)."""

    theta_rad: np.ndarray
    potential: np.ndarray  # R~ = <R> a_P / (G m_P)
    force: np.ndarray  # F~ = <dR/dlambda> a_P / (G m_P)
    body: Orbit
    planet: Orbit

    def interpolate(self, theta_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(R~, F~) at `theta_rad` by periodic linear interpolation."""
        grid = np.append(self.theta_rad, TWO_PI)
        x = np.mod(np.asarray(theta_rad, dtype=float), TWO_PI)
        return np.interp(x, grid, np.append(self.potential, self.potential[0])), np.interp(x, grid, np.append(self.force, self.force[0]))


@dataclass
class CoorbitalLaw:
    """The 1:1 force law and energy integral of one record (columns `rsp_coorb_*`)."""

    planet: str
    angle_sign: int  # theta = angle_sign x sigma (sigma = k (lambda - lambda_P), k = +-1)
    n_tables: int
    n_blocks: int
    r2: float  # a_dot_data on K a_dot_model + C over all pieces
    r2_h1: float  # the same in the first / second half of the piece samples (time order)
    r2_h2: float
    scale: float  # K / K_theory (K_theory = 1)
    scale_h1: float
    scale_h2: float
    energy_r2: float  # (1/2) theta_dot^2 on -K_E V + E_piece
    energy_scale: float  # K_E / K_theory
    harmonic_r2: float  # the same data on sin theta, cos theta, sin 2 theta, cos 2 theta, 1 (the law this one replaces)
    regime: str
    theta_min_deg: float  # range of theta over the pieces, in (-180, 180]
    theta_max_deg: float
    force_check: CheckValue
    force_thin: bool
    energy_check: CheckValue
    potential_check: CheckValue  # R0 with a of a 1:1 angle (`potential_check`)
    regime_sequence: Tuple[str, ...] = ()  # the pieces' regimes in time order (`piece_regimes`), for the orbit type

    def to_columns(self) -> dict:
        """CSV columns `rsp_coorb_*`."""
        return coorbital_columns(self)


COLUMN_FIELDS = {
    'rsp_coorb_planet': 'planet',
    'rsp_coorb_regime': 'regime',
    'rsp_coorb_n_tables': 'n_tables',
    'rsp_coorb_n_blocks': 'n_blocks',
    'rsp_coorb_R2': 'r2',
    'rsp_coorb_R2_h1': 'r2_h1',
    'rsp_coorb_R2_h2': 'r2_h2',
    'rsp_coorb_K_over_K_theory': 'scale',
    'rsp_coorb_K_over_K_theory_h1': 'scale_h1',
    'rsp_coorb_K_over_K_theory_h2': 'scale_h2',
    'rsp_coorb_energy_R2': 'energy_r2',
    'rsp_coorb_energy_K_over_K_theory': 'energy_scale',
    'rsp_coorb_harmonic_R2': 'harmonic_r2',
    'rsp_coorb_theta_min_deg': 'theta_min_deg',
    'rsp_coorb_theta_max_deg': 'theta_max_deg',
    'rsp_coorb_force_check': 'force_check',
    'rsp_coorb_energy_check': 'energy_check',
    'rsp_coorb_potential_check': 'potential_check',
}
TEXT_COLUMNS = ('rsp_coorb_planet', 'rsp_coorb_regime', 'rsp_coorb_force_check', 'rsp_coorb_energy_check', 'rsp_coorb_potential_check')


def coorbital_columns(law: Optional[CoorbitalLaw]) -> dict:
    """CSV columns `rsp_coorb_*`; the same keys, empty, for an angle that is not 1:1 (no law)."""
    if law is None:
        return {column: ('' if column in TEXT_COLUMNS else NAN) for column in COLUMN_FIELDS}
    return {column: (str(getattr(law, name)) if column in TEXT_COLUMNS else getattr(law, name)) for column, name in COLUMN_FIELDS.items()}


# --- Kepler orbits --------------------------------------------------------------------------------


def eccentric_anomaly(mean_anomaly: np.ndarray, e: float) -> np.ndarray:
    """E of Kepler's equation E - e sin E = M by Newton's method (start at pi for e > 0.8)."""
    m = np.mod(np.asarray(mean_anomaly, dtype=float), TWO_PI)
    big_e = np.full_like(m, np.pi) if e > 0.8 else m + e * np.sin(m)
    for _ in range(KEPLER_ITERATIONS):
        step = (big_e - e * np.sin(big_e) - m) / (1 - e * np.cos(big_e))
        big_e = big_e - step
        if np.all(np.abs(step) < 1e-14):
            break
    return big_e


def _rotation(orbit: Orbit) -> Tuple[np.ndarray, np.ndarray]:
    """The unit vectors P (to the pericentre) and Q (90 deg ahead in the orbit plane) in the reference frame."""
    co, so = np.cos(orbit.omega), np.sin(orbit.omega)
    cn, sn = np.cos(orbit.Omega), np.sin(orbit.Omega)
    ci, si = np.cos(orbit.inc), np.sin(orbit.inc)
    p = np.array([cn * co - sn * so * ci, sn * co + cn * so * ci, so * si])
    q = np.array([-cn * so - sn * co * ci, -sn * so + cn * co * ci, co * si])
    return p, q


def kepler_position(orbit: Orbit, mean_anomaly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(r, dr/dM) for a = 1, shape (n, 3): position and its derivative with respect to the mean anomaly (= v / n)."""
    e = float(orbit.e)
    big_e = eccentric_anomaly(mean_anomaly, e)
    cos_e, sin_e = np.cos(big_e), np.sin(big_e)
    root = np.sqrt(1 - e * e)
    x, y = cos_e - e, root * sin_e
    denominator = 1 - e * cos_e
    dx, dy = -sin_e / denominator, root * cos_e / denominator
    p, q = _rotation(orbit)
    return np.outer(x, p) + np.outer(y, q), np.outer(dx, p) + np.outer(dy, q)


# --- the averaged potential -----------------------------------------------------------------------


def hill_softening(mu: float, hill_factor: float) -> float:
    """eps / a_P = hill_factor x (mu / 3)^(1/3): the averaging does not describe the motion inside the Hill sphere."""
    return float(hill_factor * (mu / 3.0) ** (1.0 / 3.0)) if mu > 0 else 0.0


def averaged_table(body: Orbit, planet: Orbit, grid_points: int, softening: float, chunk: int = 64) -> CoorbitalTable:
    """R~ and F~ on theta_s = 2 pi s / N (s = 0 .. N - 1) by the trapezoidal rule over the same N values of lambda_P.

    The body's mean anomaly M = theta + lambda_P - phase (`Orbit.phase`) then runs over the same N-point circle as lambda_P, so the body
    is placed once on N points and each theta is an index shift (cost N^2, memory `chunk` x N). The periodic trapezoidal
    rule converges exponentially for the softened integrand while the orbits stay apart (inclined orbits: N = 360
    agrees with N = 5760 to 1e-13, e = 0.02-0.5, Earth's softening), so N is set by the theta grid that interpolates the
    sharp near-encounter peaks of F~ (linear interpolation, rms error 4-8 % at 0.5 deg, 1-2 % at 0.25 deg). Where the
    orbits nearly cross, the nodes undersample the peak: e = 0.2 coplanar, Earth's softening, theta = 337 deg gives
    167.3 at N = 720 against 144.1 converged. On the benchmark's 257 co-orbital records 8x the nodes changes no check
    and no verdict (R^2 by <= 0.07, K/K_theory by <= 0.004) at 29x the cost, so N stays (W5, codex astra finding 3).
    Lengths in a_P, both orbits with a = a_P."""
    n = int(grid_points)
    if n < MIN_FIT_SAMPLES:
        raise ValueError(f'grid_points must be at least {MIN_FIT_SAMPLES}, got {n}')
    nodes = TWO_PI * np.arange(n) / n
    r_body, dr_body = kepler_position(body, nodes - body.phase)  # index m <-> M = 2 pi m / N - phase
    r_planet, _ = kepler_position(planet, nodes - planet.phase)  # index j <-> lambda_P = 2 pi j / N
    gx, gy, gz = (r_planet / (np.sum(r_planet**2, axis=1) ** 1.5)[:, None]).T  # grad of r . r_P / r_P^3 in r
    xp, yp, zp = r_planet.T
    # the body on the doubled circle: row s of a sliding window is the body at m = s + j (no copy, no modulo)
    windows = [np.lib.stride_tricks.sliding_window_view(np.concatenate([x, x]), n) for x in (*r_body.T, *dr_body.T)]
    potential, force = np.empty(n), np.empty(n)
    eps2 = softening**2
    # The chunk's temporaries live in preallocated buffers (`out=`), in the order of the expressions in the comments,
    # so the bytes are those of the expressions: no allocation per operation (without the allocator's large cache,
    # MallocLargeCache=0, each temporary was an mmap/munmap; W23 P4)
    dx, dy, dz, first_sum, second_sum, inverse, along = (np.empty((chunk, n)) for _ in range(7))
    with np.errstate(divide='ignore', invalid='ignore'):  # e = i = 0 without softening: the planet itself at theta = 0
        for first in range(0, n, chunk):
            rows = slice(first, min(first + chunk, n))
            m = rows.stop - rows.start
            xb, yb, zb, tx, ty, tz = (w[rows] for w in windows)
            ddx, ddy, ddz, s1, s2, inv, acc = (b[:m] for b in (dx, dy, dz, first_sum, second_sum, inverse, along))
            # dx, dy, dz = xp - xb, yp - yb, zp - zb
            np.subtract(xp, xb, out=ddx)
            np.subtract(yp, yb, out=ddy)
            np.subtract(zp, zb, out=ddz)
            # inverse = 1 / sqrt(dx * dx + dy * dy + dz * dz + eps2)
            np.multiply(ddx, ddx, out=s1)
            np.multiply(ddy, ddy, out=s2)
            np.add(s1, s2, out=s1)
            np.multiply(ddz, ddz, out=s2)
            np.add(s1, s2, out=s1)
            np.add(s1, eps2, out=s1)
            np.sqrt(s1, out=s1)
            np.divide(1.0, s1, out=inv)
            # potential = mean(inverse - (xb * gx + yb * gy + zb * gz))
            np.multiply(xb, gx, out=s1)
            np.multiply(yb, gy, out=s2)
            np.add(s1, s2, out=s1)
            np.multiply(zb, gz, out=s2)
            np.add(s1, s2, out=s1)
            np.subtract(inv, s1, out=s1)
            potential[rows] = np.mean(s1, axis=1)
            # along = (dx * tx + dy * ty + dz * tz) * inverse**3 - (gx * tx + gy * ty + gz * tz)
            np.multiply(ddx, tx, out=s1)
            np.multiply(ddy, ty, out=s2)
            np.add(s1, s2, out=s1)
            np.multiply(ddz, tz, out=s2)
            np.add(s1, s2, out=s1)
            np.power(inv, 3, out=s2)
            np.multiply(s1, s2, out=acc)
            np.multiply(gx, tx, out=s1)
            np.multiply(gy, ty, out=s2)
            np.add(s1, s2, out=s1)
            np.multiply(gz, tz, out=s2)
            np.add(s1, s2, out=s1)
            np.subtract(acc, s1, out=acc)
            force[rows] = np.mean(acc, axis=1)
    return CoorbitalTable(nodes, potential, force, body, planet)


@lru_cache(maxsize=TABLE_CACHE_SIZE)
def cached_table(body: Orbit, planet: Orbit, grid_points: int, softening: float) -> CoorbitalTable:
    """`averaged_table`, remembered: the analysis measures a record twice when it re-measures the last regime, with the
    same blocks (cut at P_c), so the second pass reuses every table. Read-only arrays."""
    table = averaged_table(body, planet, grid_points, softening)
    for array in (table.theta_rad, table.potential, table.force):
        array.flags.writeable = False
    return table


_RECORD_TABLES: Optional[Dict[tuple, CoorbitalTable]] = None  # the tables of the record analysed now (`record_tables`)
_RECORD_BYTES = [0, 0]  # [bytes the scope keeps, at most]


def table_bytes(table: CoorbitalTable) -> int:
    """Bytes of a table's arrays (theta, potential, force)."""
    return int(table.theta_rad.nbytes + table.potential.nbytes + table.force.nbytes)


@contextmanager
def record_tables(max_bytes: float):
    """Keep the tables until the analysis of one record ends, at most `max_bytes` of their arrays
    (mmr_coorbital_record_tables_mb x 1e6; re-entrant: an inner scope uses the outer one). The force law tabulates the
    whole record at each call (pooled, per episode, the G1q pass), and the bounded `cached_table` evicts the tables
    between calls once a record has more of them than it holds: 2010 CG236 over 20 kyr computed 7909 tables for its
    3774 blocks (Codex W20 A #13). When the store is full, a new table is computed and not kept: the passes read the
    blocks in the same time order, so an evicting store (FIFO, LRU) smaller than a pass would miss every table, while
    keeping the first ones hits them (W20 round C, review B #4). The bound is in bytes, so it holds at any grid (W20
    round D, review C #6). The tables are dropped when the scope ends."""
    global _RECORD_TABLES
    if _RECORD_TABLES is not None:
        yield
        return
    _RECORD_TABLES = {}
    _RECORD_BYTES[:] = [0, int(max_bytes)]
    try:
        yield
    finally:
        _RECORD_TABLES = None
        _RECORD_BYTES[:] = [0, 0]


def table_for(body: Orbit, planet: Orbit, grid_points: int, softening: float) -> CoorbitalTable:
    """`cached_table`, also kept for the rest of the record inside `record_tables` while its store has room."""
    if _RECORD_TABLES is None:
        return cached_table(body, planet, grid_points, softening)
    key = (body, planet, grid_points, softening)
    found = _RECORD_TABLES.get(key)
    if found is None:
        found = cached_table(body, planet, grid_points, softening)
        size = table_bytes(found)
        if _RECORD_BYTES[0] + size <= _RECORD_BYTES[1]:
            _RECORD_TABLES[key] = found
            _RECORD_BYTES[0] += size
    return found


def record_tables_kept() -> int:
    """How many tables the open `record_tables` scope holds (0 outside one)."""
    return len(_RECORD_TABLES) if _RECORD_TABLES is not None else 0


def quasi_satellite_well(table: CoorbitalTable) -> Optional[Tuple[float, float, float]]:
    """(theta_c, R~''(theta_c), half-width) [rad, -, rad] of the quasi-satellite well of R~(theta): the local minimum
    nearest theta = 0 whose basin, between its two neighbouring maxima, holds theta = 0 nearer its bottom than its wall;
    the half-width is the distance to the nearer maximum. R~ is interpolated by a periodic cubic spline. None when no such minimum exists: for
    e, i -> 0 the potential peaks at theta = 0 (the planet itself) and the minimum nearest 0 is L4 or L5, whose basin
    ends there; a well narrower than the grid is not resolved either."""
    theta = np.append(table.theta_rad, TWO_PI)
    spline = CubicSpline(theta, np.append(table.potential, table.potential[0]), bc_type='periodic')
    roots = np.unique(np.mod(spline.derivative(1).roots(extrapolate=False), TWO_PI))
    if not len(roots):
        return None
    curvature = spline.derivative(2)(roots)
    minima, maxima = roots[curvature > 0], roots[curvature < 0]
    if not len(minima) or not len(maxima):
        return None
    centre = float(minima[np.argmin(np.abs(wrap_pi(minima)))])
    right = float(np.min(np.mod(maxima - centre, TWO_PI)))  # the nearest maximum ahead and behind (one maximum: both)
    left = float(np.min(np.mod(centre - maxima, TWO_PI)))
    # theta = 0 must lie nearer the bottom of the well than its wall on that side: a tadpole minimum whose basin reaches
    # past 0 (the planet's softened peak just beyond it: 617 Patroclus, L4 at 60.5 deg, a wall at -0.5 deg) is no
    # quasi-satellite well
    ahead = float(np.mod(-centre, TWO_PI))
    behind = TWO_PI - ahead
    if not (ahead < right - ahead or behind < left - behind):
        return None
    return float(wrap_pi(centre)), float(spline.derivative(2)(centre)), min(left, right)


@dataclass(frozen=True)
class QuasiSatelliteWell:
    """The quasi-satellite well of the averaged potential along a record, per sample (nan where there is none):
    centre theta_c and half-width [rad] (theta = lambda - lambda_P), and the small-amplitude period
    P_QS = 2 pi / omega_QS, omega_QS^2 = 3 mu n^2 R~''(theta_c) (eq. 2 of the module docstring) [yr]."""

    t_yr: np.ndarray
    centre_rad: np.ndarray
    halfwidth_rad: np.ndarray
    period_yr: np.ndarray

    @property
    def period_min_yr(self) -> float:
        """The shortest P_QS of the record: the shortest libration the averaged potential allows there [yr]."""
        finite = self.period_yr[np.isfinite(self.period_yr)]
        return float(finite.min()) if len(finite) else NAN

    def inside(self, theta_rad: np.ndarray) -> np.ndarray:
        """Samples whose theta lies inside the well (False where there is none)."""
        with np.errstate(invalid='ignore'):
            return np.abs(wrap_pi(np.asarray(theta_rad, dtype=float) - self.centre_rad)) < self.halfwidth_rad


def record_quasi_satellite_well(
    t_yr: np.ndarray,
    body_blocks: Sequence[Orbit],
    planet_blocks: Sequence[Orbit],
    starts: np.ndarray,
    planet_mass: float,
    planet_mean_motion_rad_yr: float,
    settings: AnalysisSettings,
) -> Optional[QuasiSatelliteWell]:
    """`QuasiSatelliteWell` along the record from the tables of `model_along` (the same blocks and retabulation, so the
    tables are the cached ones of the force law); None when no block has a well."""
    n = len(t_yr)
    centre, halfwidth, period = np.full(n, NAN), np.full(n, NAN), np.full(n, NAN)
    softening = hill_softening(planet_mass, settings.mmr_coorbital_softening_hill)
    tolerance_inc = np.radians(settings.mmr_coorbital_retabulate_inc_deg)
    key, well = None, None
    ends = list(starts[1:]) + [n]
    for start, end, body, planet in zip(starts, ends, body_blocks, planet_blocks):
        if key is None or needs_new_table(key, (body, planet), settings.mmr_coorbital_retabulate_e, tolerance_inc):
            key = (body, planet)
            well = quasi_satellite_well(table_for(body, planet, settings.mmr_coorbital_grid_points, softening))
        if well is not None and well[1] > 0:
            omega = np.sqrt(3 * planet_mass * planet_mean_motion_rad_yr**2 * well[1])
            centre[start:end], halfwidth[start:end], period[start:end] = well[0], well[2], TWO_PI / omega
    if not np.any(np.isfinite(period)):
        return None
    return QuasiSatelliteWell(np.asarray(t_yr, dtype=float), centre, halfwidth, period)


# --- slowly varying elements and the tables along a record ----------------------------------------


def block_orbits(t_yr: np.ndarray, e, inc, omega, Omega, block_yr: float, omega_coefficient: int = 1) -> Tuple[np.ndarray, List[Orbit]]:
    """Consecutive blocks of about `block_yr` (at least one sample) and the orbit of each: median e and i, circular
    means of omega and Omega. Returns (block start indices, orbits); scalars are broadcast to every block."""
    t_yr = np.asarray(t_yr, dtype=float)
    n = len(t_yr)
    step = float(np.median(np.diff(t_yr))) if n > 1 else 1.0
    length = max(1, int(round(block_yr / step))) if np.isfinite(block_yr) and block_yr > 0 else n
    starts = np.arange(0, n, length)
    arrays = [np.broadcast_to(np.asarray(x, dtype=float), (n,)) for x in (e, inc, omega, Omega)]
    orbits = []
    for start in starts:
        part = slice(start, min(start + length, n))
        e_b, i_b, w_b, o_b = (x[part] for x in arrays)
        orbits.append(
            Orbit(
                float(np.median(e_b)), float(np.median(i_b)), filtering.circular_mean(w_b), filtering.circular_mean(o_b), omega_coefficient
            )
        )
    return starts, orbits


def needs_new_table(current: Tuple[Orbit, Orbit], candidate: Tuple[Orbit, Orbit], tolerance_e: float, tolerance_inc_rad: float) -> bool:
    """True when the eccentricity vector or the orbit normal of the body or the planet moved more than the tolerance.
    Both are 3D vectors in the reference frame: well defined for e -> 0 and i -> 0, and a node turning at i = 0 turns
    the perihelion with it (a planar e vector e exp(i omega) would miss that)."""
    for old, new in zip(current, candidate):
        if np.linalg.norm(new.eccentricity_vector() - old.eccentricity_vector()) > tolerance_e:
            return True
        if np.linalg.norm(new.inclination_vector() - old.inclination_vector()) > np.sin(tolerance_inc_rad):
            return True
    return False


def model_along(
    t_yr: np.ndarray,
    theta_rad: np.ndarray,
    body_blocks: Sequence[Orbit],
    planet_blocks: Sequence[Orbit],
    starts: np.ndarray,
    softening: float,
    settings: AnalysisSettings,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """(R~(t), F~(t), number of tables) along the record: one table per run of blocks whose orbits stay within the
    retabulation tolerances of the block that opened the table."""
    n = len(t_yr)
    potential, force = np.empty(n), np.empty(n)
    tolerance_inc = np.radians(settings.mmr_coorbital_retabulate_inc_deg)
    key, table, count = None, None, 0
    ends = list(starts[1:]) + [n]
    for start, end, body, planet in zip(starts, ends, body_blocks, planet_blocks):
        if key is None or needs_new_table(key, (body, planet), settings.mmr_coorbital_retabulate_e, tolerance_inc):
            key = (body, planet)
            table = table_for(body, planet, settings.mmr_coorbital_grid_points, softening)
            count += 1
        potential[start:end], force[start:end] = table.interpolate(theta_rad[start:end])
    return potential, force, count


# --- fits ---------------------------------------------------------------------------------------


def scale_fit(target: np.ndarray, model: np.ndarray, groups: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """target = K model + C_g (one constant per group, one in all without groups) by least squares; (K, R^2).
    nan for fewer than MIN_FIT_SAMPLES finite samples or a constant model."""
    target, model = np.asarray(target, dtype=float), np.asarray(model, dtype=float)
    mask = np.isfinite(target) & np.isfinite(model)
    if np.count_nonzero(mask) < MIN_FIT_SAMPLES:
        return NAN, NAN
    target, model = target[mask], model[mask]
    labels = np.zeros(len(target), dtype=int) if groups is None else np.unique(np.asarray(groups)[mask], return_inverse=True)[1]
    counts = np.bincount(labels).astype(float)
    centred_t = target - (np.bincount(labels, target) / counts)[labels]
    centred_m = model - (np.bincount(labels, model) / counts)[labels]
    variance = float(np.sum(centred_m**2))
    if variance <= 0:
        return NAN, NAN
    scale = float(np.sum(centred_m * centred_t) / variance)
    total = float(np.sum((target - target.mean()) ** 2))
    if total <= 0:
        return scale, NAN
    return scale, 1 - float(np.sum((centred_t - scale * centred_m) ** 2)) / total


def harmonic_r2(theta_rad: np.ndarray, target: np.ndarray) -> float:
    """R^2 of target on sin theta, cos theta, sin 2 theta, cos 2 theta and 1 (the harmonic law of the other MMRs)."""
    theta, target = np.asarray(theta_rad, dtype=float), np.asarray(target, dtype=float)
    mask = np.isfinite(theta) & np.isfinite(target)
    if np.count_nonzero(mask) <= 5:
        return NAN
    theta, target = theta[mask], target[mask]
    design = np.column_stack([np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta), np.ones_like(theta)])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return filtering.r2(target, target - design @ coef)


def _piece_regime(theta: np.ndarray) -> str:
    """Regime of one piece of the unwrapped theta: the range holds 0 (mod 2 pi), pi (mod 2 pi), both or neither."""
    low, high = float(np.min(theta)), float(np.max(theta))
    passes_zero = np.floor(high / TWO_PI) >= np.ceil(low / TWO_PI)
    passes_pi = np.floor((high - np.pi) / TWO_PI) >= np.ceil((low - np.pi) / TWO_PI)
    if passes_zero and passes_pi:
        return REGIME_COMPOUND
    if passes_pi:
        return REGIME_HORSESHOE
    if passes_zero:
        return REGIME_QUASI_SATELLITE
    return REGIME_TADPOLE_L4 if np.mod(np.median(theta), TWO_PI) < np.pi else REGIME_TADPOLE_L5


def piece_regimes(theta_rad: np.ndarray, labels: np.ndarray) -> Tuple[str, ...]:
    """The regime of each response piece (`labels`: piece index of each sample, the pieces in time order), repeats of
    neighbours merged: ('horseshoe', 'quasi_satellite') for a horseshoe that becomes a quasi-satellite. A piece that
    passes both 0 and 180 deg is 'compound' itself."""
    theta, labels = np.asarray(theta_rad, dtype=float), np.asarray(labels)
    finite = np.isfinite(theta)
    names: List[str] = []
    for label in np.unique(labels[finite]):
        name = _piece_regime(theta[finite & (labels == label)])
        if not names or names[-1] != name:
            names.append(name)
    return tuple(names)


def regime(theta_rad: np.ndarray, labels: np.ndarray) -> Tuple[str, float, float]:
    """Regime of the unwrapped theta = lambda - lambda_P over the response pieces (`labels`: piece of each sample):
    tadpole about L4 (theta in (0, 180) deg) or L5, horseshoe (passes 180 deg, never 0), quasi-satellite (passes 0,
    never 180), compound (both, or pieces of different regimes; which ones: `piece_regimes`). A column, no status.
    Returns (regime, min, max) with min and max of the wrapped theta in (-180, 180] deg."""
    theta, labels = np.asarray(theta_rad, dtype=float), np.asarray(labels)
    finite = np.isfinite(theta)
    if not np.any(finite):
        return REGIME_NONE, NAN, NAN
    names = set(piece_regimes(theta, labels))
    name = names.pop() if len(names) == 1 else REGIME_COMPOUND
    wrapped = np.degrees(np.angle(np.exp(1j * theta[finite])))
    return name, float(wrapped.min()), float(wrapped.max())


def force_check(law_r2_halves: Sequence[float], scales: Sequence[float], settings: AnalysisSettings) -> Tuple[CheckValue, bool]:
    """check_mmr_a_response of a co-orbital: pass when R^2 >= mmr_force_r2_full in both halves and K / K_theory lies in
    [1 / mmr_coorbital_scale_ratio_max, mmr_coorbital_scale_ratio_max] overall and in both halves (the force has the
    sign and the size of the planet's); fail when R^2 < mmr_force_r2_fail in a half or K / K_theory is out of the band;
    thin pass with R^2 in [fail, full). not_measured when a half or the scale is missing."""
    halves = [float(x) for x in law_r2_halves]
    ratios = [float(x) for x in scales]
    if not halves or not all(np.isfinite(halves)) or not ratios or not all(np.isfinite(ratios)):
        return CheckValue.NOT_MEASURED, False
    limit = settings.mmr_coorbital_scale_ratio_max
    in_band = all(1 / limit <= ratio <= limit for ratio in ratios)
    if min(halves) < settings.mmr_force_r2_fail or not in_band:
        return CheckValue.FAIL, False
    return CheckValue.PASS, min(halves) < settings.mmr_force_r2_full


def energy_check(r2: float, scale: float, settings: AnalysisSettings) -> CheckValue:
    """The co-orbital potential with a (in place of R0's harmonic V_E ~ V_F): pass when (1/2) theta_dot^2 = -K_E V + E
    explains >= mmr_force_r2_full of the variance with K_E / K_theory in the scale band; fail below mmr_force_r2_fail or
    out of the band; not_measured without numbers. A value in [fail, full) is not a failure: pass."""
    if not (np.isfinite(r2) and np.isfinite(scale)):
        return CheckValue.NOT_MEASURED
    limit = settings.mmr_coorbital_scale_ratio_max
    if r2 < settings.mmr_force_r2_fail or not (1 / limit <= scale <= limit):
        return CheckValue.FAIL
    return CheckValue.PASS


def potential_check(force: CheckValue, energy: CheckValue) -> CheckValue:
    """R0 with a of a 1:1 angle from the two forms of the same averaged dynamics: the energy integral (2) holds -> pass;
    it fails while the local force law (1) passes -> not_measured: the potential is right (the force follows its
    derivative sample by sample) but E is not conserved within the cycles (encounters, the slow drift of e, i, omega
    over a transient episode), so the integral cannot measure it; it fails and the force law does not pass -> fail."""
    if energy == CheckValue.PASS:
        return CheckValue.PASS
    if energy == CheckValue.FAIL:
        return CheckValue.NOT_MEASURED if force == CheckValue.PASS else CheckValue.FAIL
    return CheckValue.NOT_MEASURED


# --- the record ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class CoorbitalInput:
    """What the law reads, as arrays on the record's grid (angles in rad)."""

    t_yr: np.ndarray
    angle: np.ndarray  # sigma, unwrapped and unfiltered (the model is filtered after the evaluation, like a)
    a: np.ndarray  # the body's osculating a, unfiltered [au]
    adot: np.ndarray  # the data's a_dot after the response filter [au/yr]
    mask: np.ndarray  # the response pieces
    pieces: Sequence[Tuple[float, float]]
    e: np.ndarray
    inc: np.ndarray
    omega: np.ndarray
    Omega: np.ndarray
    planet: str
    planet_mass: float  # m_P / M_sun
    planet_mean_motion_rad_yr: float
    body_coefficient: int  # k = +-1 of sigma = k (lambda - lambda_P)
    cutoffs_yr: Tuple[float, ...]  # the low-passes the data went through, in order (P_c, then the response cutoff)
    planet_elements: Optional[Dict[str, np.ndarray]] = None  # e, inc, Omega, varpi of the planet when saved
    cycle_edges_yr: Sequence[float] = ()  # boundaries of the libration cycles (every second turning point): one energy each
    retrograde: bool = False  # a 1/-1 angle of an orbit with i > 90 deg (`RetrogradeTwoBody`)
    omega_coefficient: int = 1  # c of the body's longitude M + c omega + Omega in sigma (`Orbit`): 2k - 1 for 1J-1r<k>
    block_yr: float = NAN  # blocks of the slowly varying elements; nan = the first cutoff (P_c)


def planet_orbits(
    t_yr: np.ndarray, planet: str, planet_elements: Optional[Dict[str, np.ndarray]], starts: np.ndarray, block_length_yr: float
) -> List[Orbit]:
    """The planet's orbit in each block: its saved elements (e, inc, Omega, varpi), else its mean J2000 orbit."""
    if planet_elements is not None:
        p = planet_elements
        omega = np.asarray(p['varpi'], dtype=float) - np.asarray(p['Omega'], dtype=float)
        _, orbits = block_orbits(t_yr, p['e'], p['inc'], omega, p['Omega'], block_length_yr)
        return orbits
    e, inc_deg, node_deg, varpi_deg = PLANET_MEAN_ORBITS_J2000.get(planet, (0.0, 0.0, 0.0, 0.0))
    orbit = Orbit(e, np.radians(inc_deg), np.radians(varpi_deg - node_deg), np.radians(node_deg))
    return [orbit] * len(starts)


def _planet_blocks(data: CoorbitalInput, starts: np.ndarray, block_length_yr: float) -> List[Orbit]:
    return planet_orbits(data.t_yr, data.planet, data.planet_elements, starts, block_length_yr)


def well_along(
    t_yr: np.ndarray,
    e,
    inc,
    omega,
    Omega,
    planet: str,
    planet_elements: Optional[Dict[str, np.ndarray]],
    planet_mass: float,
    planet_mean_motion_rad_yr: float,
    block_yr: float,
    settings: AnalysisSettings,
) -> Optional[QuasiSatelliteWell]:
    """`record_quasi_satellite_well` of a prograde 1:1 record from the body's and the planet's elements in blocks of
    `block_yr` (the force law's blocks: P_c)."""
    t = np.asarray(t_yr, dtype=float)
    starts, body_blocks = block_orbits(t, e, inc, omega, Omega, block_yr)
    planet_blocks = planet_orbits(t, planet, planet_elements, starts, block_yr)
    return record_quasi_satellite_well(t, body_blocks, planet_blocks, starts, planet_mass, planet_mean_motion_rad_yr, settings)


def _filtered(t_yr: np.ndarray, values: np.ndarray, cutoffs_yr: Sequence[float], settings: AnalysisSettings) -> np.ndarray:
    """`values` through the same low-passes as the data (zero-phase, so the order does not matter)."""
    out = np.asarray(values, dtype=float)
    step = float(np.median(np.diff(t_yr)))
    for cutoff in cutoffs_yr:
        if np.isfinite(cutoff) and cutoff > 2 * step:
            out = filtering.lowpass_with(t_yr, out, cutoff, settings)
    return out


def _piece_labels(t_yr: np.ndarray, pieces: Sequence[Tuple[float, float]]) -> np.ndarray:
    labels = np.full(len(t_yr), -1)
    for i, (start, end) in enumerate(pieces):
        labels[(t_yr >= start) & (t_yr <= end)] = i
    return labels


def energy_fit(
    t_yr: np.ndarray,
    a: np.ndarray,
    model_v: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    identity_rate: float,
    cutoffs_yr: Sequence[float],
    settings: AnalysisSettings,
) -> Tuple[float, float]:
    """(K_E / K_theory, R^2) of the energy integral (2) written with a: theta_dot = -|c| (a - a*), |c| = 3 n / (2 a_P),

        (1/2) c^2 (a - a*)^2 = E_piece - K_E V(theta)          (both sides through the data's filters).

    Squares are formed before the filter (filtering is linear, squaring is not: the filtered square of a sharp
    encounter is not the square of the filtered a). a* (the exact 1:1 axis, unknown to ~1e-4 au from the osculating
    elements) comes from the regression of (1/2) c^2 a^2 on c^2 a, V and the piece constants; then (1/2) c^2 (a - a*)^2
    on V and the piece constants gives K_E and R^2 (R^2 of the quadratic part only: the linear part is a* itself)."""
    c2 = identity_rate**2
    square = _filtered(t_yr, 0.5 * c2 * a**2, cutoffs_yr, settings)[mask]
    linear = _filtered(t_yr, c2 * a, cutoffs_yr, settings)[mask]
    v = model_v[mask]
    finite = np.isfinite(square) & np.isfinite(linear) & np.isfinite(v)
    if np.count_nonzero(finite) < MIN_FIT_SAMPLES:
        return NAN, NAN
    groups = np.unique(labels[finite], return_inverse=True)[1]
    counts = np.bincount(groups).astype(float)

    def within(x: np.ndarray) -> np.ndarray:  # minus the group mean: the piece/cycle constants without a dummy matrix
        return x - (np.bincount(groups, x) / counts)[groups]

    design = np.column_stack([within(linear[finite]), within(v[finite])])
    coef, *_ = np.linalg.lstsq(design, within(square[finite]), rcond=None)
    a_star = float(coef[0])  # (1/2) c^2 a^2 = a* c^2 a - K_E V + const
    kinetic = _filtered(t_yr, 0.5 * c2 * (a - a_star) ** 2, cutoffs_yr, settings)[mask]
    slope, r2 = scale_fit(kinetic, v, labels)
    return -slope, r2


def coorbital_law(data: CoorbitalInput, settings: AnalysisSettings) -> CoorbitalLaw:
    """The force law (1) and the energy integral (2) of a 1:1 angle on the response pieces (module docstring)."""
    t = np.asarray(data.t_yr, dtype=float)
    sign = 1 if data.body_coefficient > 0 else -1
    theta = sign * np.asarray(data.angle, dtype=float)  # sigma = k (lambda - lambda_P), k = +-1
    mu, n = float(data.planet_mass), float(data.planet_mean_motion_rad_yr)
    a_planet = (GM_SUN / n**2) ** (1.0 / 3.0) if n > 0 else NAN
    block_yr = data.block_yr  # P_c (`Preprocessed.elements_block_yr`): the same blocks whatever the cycle period
    if not np.isfinite(block_yr):
        block_yr = data.cutoffs_yr[0] if data.cutoffs_yr else NAN
    starts, body_blocks = block_orbits(t, data.e, data.inc, data.omega, data.Omega, block_yr, data.omega_coefficient)
    planet_blocks = _planet_blocks(data, starts, block_yr)
    softening = hill_softening(mu, settings.mmr_coorbital_softening_hill)
    potential, force, n_tables = model_along(t, theta, body_blocks, planet_blocks, starts, softening, settings)

    model_adot = _filtered(t, 2 * mu * n * a_planet * force, data.cutoffs_yr, settings)  # (1)
    model_v = _filtered(t, 3 * mu * n**2 * potential, data.cutoffs_yr, settings)  # (2)
    mask = np.asarray(data.mask, dtype=bool)
    adot = np.asarray(data.adot, dtype=float)
    scale, r2 = scale_fit(adot[mask], model_adot[mask])
    index = np.flatnonzero(mask)
    first, second = np.array_split(index, 2) if len(index) else (index, index)
    scale_h1, r2_h1 = scale_fit(adot[first], model_adot[first])
    scale_h2, r2_h2 = scale_fit(adot[second], model_adot[second])
    labels = _piece_labels(t, data.pieces)[mask]
    cycles = np.searchsorted(np.sort(np.asarray(data.cycle_edges_yr, dtype=float)), t[mask])
    energy_groups = labels * (len(data.cycle_edges_yr) + 1) + cycles
    energy_scale, energy_r2 = energy_fit(
        t, np.asarray(data.a, dtype=float), model_v, mask, energy_groups, 1.5 * n / a_planet, data.cutoffs_yr, settings
    )
    check, thin = force_check((r2_h1, r2_h2), (scale, scale_h1, scale_h2), settings)
    energy = energy_check(energy_r2, energy_scale, settings)
    name, theta_min, theta_max = regime(theta[mask], labels)
    sequence = piece_regimes(theta[mask], labels)
    if data.retrograde:
        name, sequence = REGIME_RETROGRADE, (REGIME_RETROGRADE,)
    return CoorbitalLaw(
        planet=data.planet,
        angle_sign=sign,
        n_tables=n_tables,
        n_blocks=len(starts),
        r2=r2,
        r2_h1=r2_h1,
        r2_h2=r2_h2,
        scale=scale,
        scale_h1=scale_h1,
        scale_h2=scale_h2,
        energy_r2=energy_r2,
        energy_scale=energy_scale,
        harmonic_r2=harmonic_r2(theta[mask], adot[mask]),
        regime=name,
        theta_min_deg=theta_min,
        theta_max_deg=theta_max,
        force_check=check,
        force_thin=thin,
        energy_check=energy,
        potential_check=potential_check(check, energy),
        regime_sequence=sequence,
    )
