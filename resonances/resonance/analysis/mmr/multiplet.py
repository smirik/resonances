"""Coefficients of the multiplet by direct averaging of the disturbing function (SPEC §6.4 D; SISTERS §1.3).

The planar members sigma_j = k_P lambda_P + k lambda + (q - j) varpi + j varpi_P of one two-body
commensurability (q = -(k_P + k), signed) share the fast part and differ by j beta, beta = varpi_P - varpi.
The resonant part of the averaged disturbing function is

    R_res = sum_j 2 C_j cos(sigma_0 + j beta) = 2 |P(beta)| cos(sigma_0 + arg P(beta)),  P(beta) = sum_j C_j e^{i j beta},

(Morbidelli 2002, eq. 9.17; Tsiganis 2010, eqs. 24-27), so the composite pendulum has its stable point at
c_0(beta) = pi - arg P(beta) for sigma_0 (the pure-asteroid member, j = 0) and at c_0 + j beta for member j.

C_j is computed without any series expansion: R = mu (1/|r - r_P| - r.r_P / r_P^3) (G M_sun = 1, au) on a
uniform grid of (lambda, lambda_P) with varpi = 0 and varpi_P = beta; its Fourier coefficient at
k lambda + k_P lambda_P is the average over the fast angle at fixed sigma_0 (`resonant_coefficients`), and the
Fourier coefficients of that in beta are the C_j. Valid for any e, e_P while the orbits do not cross. Planar:
node members are not included. Ported from research/scripts/multiplet_theory.py (a double loop over sigma_0,
beta and the fast angle) to a 2-D Fourier sum, which is independent of k_P and ~30x faster; the two agree to
the round-off of the grids (tests), and the 3:1 with Jupiter gives f = 0.5987 against 0.5988 of M&D.

A term 2 C cos(sigma) has mu_eff = 2 |C| a, the small-amplitude frequency omega_0 = |k| n sqrt(3 mu_eff) and
the half-width a sqrt(16 mu_eff / 3) (M&D 1999, §8.7; SISTERS §1.2). Coefficient tables are cached in memory
by `cache_key` (the averaging is deterministic: without the cache the result is the same).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

MULTIPLET_VERSION = 2  # part of the cache key; bump when the averaging changes (2: the 2-D Fourier sum)

TWO_PI = 2 * np.pi
# Numerical resolution of the averaging (not thresholds). The grid in each longitude must resolve the
# harmonic of the resonance and the peak of 1/|r - r_P| at the closest approach; the error is the aliased
# coefficient at (k + m N_body, k_P + m' N_planet), which decays exponentially with N. 256 samples and 8 per
# unit of the coefficient keep the error below 1e-8 of max |C| for 9M-16, 13M-24 (e 0.23, close to Mars) and
# 54M-89 against grids four times finer; beta needs twice the largest |j| (no aliasing of the kept members).
GRID_MIN_SAMPLES = 256
GRID_SAMPLES_PER_HARMONIC = 8
BETA_MIN_SAMPLES = 32
BETA_SAMPLES_PER_HARMONIC = 2
# The averaging resolves coefficients down to ~1e-13 of the mean |R| ~ mu / a; a multiplet whose largest coefficient
# is below this (orders of ~40 and more: 85J-27 has e^58) has no usable theory (`resolved`).
ROUND_OFF_FLOOR = 1e-11
KEPLER_TOLERANCE = 1e-13
KEPLER_MAX_ITERATIONS = 50
KEPLER_HIGH_E = 0.8  # above it Newton starts from E = pi (always converges)
BETA_CHUNK = 8  # beta values per vectorised block (memory: BETA_CHUNK x N_body x N_planet doubles)
CACHE_DIGITS = 4  # significant digits of alpha, e, e_P in the cache key
WINDING_SAMPLES = 4096

_CACHE: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}


def signed_order(planet_coefficient: int, body_coefficient: int) -> int:
    """q = -(k_P + k): the d'Alembert sum of the slow coefficients (negative for some exterior angles)."""
    return -(int(planet_coefficient) + int(body_coefficient))


def planar_members(planet_coefficient: int, body_coefficient: int) -> np.ndarray:
    """The j of the lowest-order planar members: 0..q (q > 0) or q..0 (q < 0)."""
    q = signed_order(planet_coefficient, body_coefficient)
    return np.arange(min(0, q), max(0, q) + 1)


def kepler_eccentric_anomaly(mean_anomaly: np.ndarray, e: float) -> np.ndarray:
    """E - e sin E = M by Newton's method (vectorised; 0 <= e < 1)."""
    mean_anomaly = np.mod(np.asarray(mean_anomaly, dtype=float), TWO_PI)
    anomaly = mean_anomaly + e * np.sin(mean_anomaly) if e < KEPLER_HIGH_E else np.full_like(mean_anomaly, np.pi)
    for _ in range(KEPLER_MAX_ITERATIONS):
        step = (anomaly - e * np.sin(anomaly) - mean_anomaly) / (1 - e * np.cos(anomaly))
        anomaly = anomaly - step
        if np.max(np.abs(step)) < KEPLER_TOLERANCE:
            break
    return anomaly


def planar_position(a_au: float, e: float, varpi_rad, mean_longitude_rad) -> Tuple[np.ndarray, np.ndarray]:
    """Heliocentric (x, y) [au] of a planar Keplerian orbit; broadcasts varpi against the mean longitude."""
    varpi = np.asarray(varpi_rad, dtype=float)
    anomaly = kepler_eccentric_anomaly(np.asarray(mean_longitude_rad, dtype=float) - varpi, e)
    x = a_au * (np.cos(anomaly) - e)
    y = a_au * math.sqrt(1 - e * e) * np.sin(anomaly)
    c, s = np.cos(varpi), np.sin(varpi)
    return x * c - y * s, x * s + y * c


def _grid_size(coefficient: int) -> int:
    return int(max(GRID_MIN_SAMPLES, GRID_SAMPLES_PER_HARMONIC * abs(int(coefficient))))


def _first_harmonic(planet_coefficient, body_coefficient, a_au, planet_a_au, e, planet_e, mu, beta) -> np.ndarray:
    """R1(beta): the coefficient of e^{i sigma_0} of R averaged over the fast angle, for each beta (varpi = 0)."""
    n_body, n_planet = _grid_size(body_coefficient), _grid_size(planet_coefficient)
    lam = np.arange(n_body) * TWO_PI / n_body
    lam_p = np.arange(n_planet) * TWO_PI / n_planet
    x, y = planar_position(a_au, e, 0.0, lam)
    body_phase = np.exp(-1j * body_coefficient * lam)  # e^{-i k lambda}
    planet_phase = np.exp(-1j * planet_coefficient * lam_p)  # e^{-i k_P lambda_P}
    result = np.empty(len(beta), dtype=complex)
    for start in range(0, len(beta), BETA_CHUNK):
        block = np.asarray(beta[start : start + BETA_CHUNK], dtype=float)
        xp, yp = planar_position(planet_a_au, planet_e, block[:, None], lam_p[None, :])  # (n_beta, n_planet)
        dx = x[None, :, None] - xp[:, None, :]
        dy = y[None, :, None] - yp[:, None, :]
        rp3 = ((xp * xp + yp * yp) ** 1.5)[:, None, :]
        disturbing = mu * (1 / np.sqrt(dx * dx + dy * dy) - (x[None, :, None] * xp[:, None, :] + y[None, :, None] * yp[:, None, :]) / rp3)
        # mean over the torus of R e^{-i (k lambda + k_P lambda_P)}
        result[start : start + len(block)] = np.einsum('bij,i,j->b', disturbing, body_phase, planet_phase) / (n_body * n_planet)
    return result


def resonant_coefficients(
    planet_coefficient: int,
    body_coefficient: int,
    a_au: float,
    planet_a_au: float,
    e: float,
    planet_e: float,
    mu: float,
    j_min: int,
    j_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """C_j (complex; G M_sun = 1, au; proportional to `mu`, so mu = 1 gives them per unit mass ratio) of the members
    j = j_min..j_max by numerical averaging over the fast angle at fixed sigma_0 and beta. Returns (j values, C_j).

    `j_max` defaults to max(0, q) + 3. The C_j are real up to round-off (a cosine-only expansion); the imaginary part
    is kept as a check. Raises ValueError for e or e_P outside [0, 1) or a non-positive axis."""
    q = signed_order(planet_coefficient, body_coefficient)
    j_max = max(0, q) + 3 if j_max is None else int(j_max)
    if j_max < j_min:
        raise ValueError(f'j_max {j_max} < j_min {j_min}')
    if not (0 <= e < 1 and 0 <= planet_e < 1) or a_au <= 0 or planet_a_au <= 0:
        raise ValueError(f'bad orbit: a={a_au}, e={e}, a_P={planet_a_au}, e_P={planet_e}')
    j_values = np.arange(int(j_min), j_max + 1)
    n_beta = int(max(BETA_MIN_SAMPLES, BETA_SAMPLES_PER_HARMONIC * (int(np.max(np.abs(j_values))) + 1)))
    beta = np.arange(n_beta) * TWO_PI / n_beta
    first = _first_harmonic(planet_coefficient, body_coefficient, a_au, planet_a_au, e, planet_e, mu, beta)
    coefficients = np.array([np.mean(first * np.exp(-1j * j * beta)) for j in j_values])
    return j_values, coefficients


def round_off_floor(mu: float, a_au: float, planet_a_au: float) -> float:
    """|C| below which the averaging returns round-off: ROUND_OFF_FLOOR x mu / max(a, a_P)."""
    return ROUND_OFF_FLOOR * mu / max(a_au, planet_a_au)


def resolved(coefficients: np.ndarray, mu: float, a_au: float, planet_a_au: float) -> bool:
    """The largest |C_j| stands above the round-off of the averaging (`round_off_floor`)."""
    return bool(np.max(np.abs(np.asarray(coefficients))) > round_off_floor(mu, a_au, planet_a_au))


def crossing(a_au: float, e: float, planet_a_au: float, planet_e: float) -> bool:
    """The planar orbits can intersect: perihelion of the outer below aphelion of the inner (the averaging of
    1/|r - r_P| is then singular and not a resonant coefficient)."""
    inner, outer = sorted([(a_au, e), (planet_a_au, planet_e)])
    return bool(outer[0] * (1 - outer[1]) <= inner[0] * (1 + inner[1]))


def cache_key(resonance_key: str, alpha: float, e: float, planet_e: float) -> str:
    """File name of the cached coefficient table: <resonance>-<version>-<rounded alpha, e, e_P>."""

    def rounded(value: float) -> str:
        return f'{float(value):.{CACHE_DIGITS}g}'

    return f'{resonance_key}-v{MULTIPLET_VERSION}-a{rounded(alpha)}-e{rounded(e)}-ep{rounded(planet_e)}'


def cached_coefficients(
    resonance_key: str,
    planet_coefficient: int,
    body_coefficient: int,
    a_au: float,
    planet_a_au: float,
    e: float,
    planet_e: float,
    mu: float,
    j_values: Sequence[int],
) -> Tuple[np.ndarray, str]:
    """C_j for the given j (in their order) at (alpha, e, e_P) rounded to the cache key, from memory or computed."""
    alpha = a_au / planet_a_au
    key = cache_key(resonance_key, alpha, e, planet_e)
    j_values = np.asarray(j_values, dtype=int)
    stored = _CACHE.get((key, float(mu)))
    if stored is None or not set(j_values) <= set(stored[0]):
        alpha_r, e_r, ep_r = (float(f'{float(v):.{CACHE_DIGITS}g}') for v in (alpha, e, planet_e))
        js, cs = resonant_coefficients(
            planet_coefficient,
            body_coefficient,
            alpha_r * planet_a_au,
            planet_a_au,
            e_r,
            ep_r,
            mu,
            int(j_values.min()),
            int(j_values.max()),
        )
        stored = (js, cs)
        _CACHE[(key, float(mu))] = stored
    lookup = dict(zip(stored[0].tolist(), stored[1]))
    return np.array([lookup[int(j)] for j in j_values]), key


def composite(coefficients: np.ndarray, j_values: np.ndarray, beta_rad: np.ndarray) -> np.ndarray:
    """P(beta) = sum_j C_j e^{i j beta} (complex). `coefficients` is (n_j,) or (n_t, n_j) with `beta_rad` (n_t,)."""
    coefficients = np.asarray(coefficients)
    beta = np.asarray(beta_rad, dtype=float)
    phases = np.exp(1j * np.multiply.outer(beta, np.asarray(j_values, dtype=float)))
    if coefficients.ndim == 1:
        return phases @ coefficients
    return np.sum(coefficients * phases, axis=-1)


def composite_centre(coefficients: np.ndarray, j_values: np.ndarray, beta_rad: np.ndarray) -> np.ndarray:
    """c0(beta) = pi - arg P(beta) [rad] (unwrapped along beta): the stable point of sigma_0 (j = 0)."""
    return np.unwrap(np.pi - np.angle(composite(coefficients, j_values, beta_rad)))


def winding_number(coefficients: np.ndarray, j_values: np.ndarray) -> int:
    """Turns of P(beta) around 0 over one turn of beta: the member whose centre stays fixed over a beta cycle
    (argument principle; 0 = the pure-asteroid angle, as in the composite regime of both prototypes)."""
    beta = np.linspace(0, TWO_PI, WINDING_SAMPLES + 1)
    phase = np.unwrap(np.angle(composite(coefficients, j_values, beta)))
    return int(round((phase[-1] - phase[0]) / TWO_PI))


def rouche_holds(coefficients: np.ndarray, j_values: np.ndarray, j: int) -> bool:
    """|C_j| > sum_{i != j} |C_i|: member j dominates (its centre stays within arcsin(sum/|C_j|) of 0 or pi)."""
    magnitudes = np.abs(np.asarray(coefficients))
    j_values = np.asarray(j_values)
    if j not in j_values:
        return False
    own = float(magnitudes[j_values == j][0])
    return own > float(np.sum(magnitudes)) - own


def binomial_mode(order: int, e: float, planet_e: float, ratio: float) -> int:
    """j* = floor((|q| + 1) x / (1 + x)), x = ratio e_P / e: the leading member if amplitudes are binomial,
    C_j ~ C(q, j) e^{q-j} (ratio e_P)^j (the mode of a binomial distribution; trigger T6 only). The sign of q
    is kept (j* <= 0 for q < 0); 0 when a value is not finite, q when e = 0 (only the planet term is left)."""
    q = int(order)
    if not (np.isfinite(e) and np.isfinite(planet_e) and np.isfinite(ratio)):
        return 0
    if e <= 0:
        return q
    x = abs(ratio) * planet_e / e
    mode = int(math.floor((abs(q) + 1) * x / (1 + x)))
    return int(math.copysign(min(mode, abs(q)), q)) if q else 0


def binomial_ratio(coefficients: np.ndarray, j_values: np.ndarray, order: int, e: float, planet_e: float) -> float:
    """The relative-eccentricity weight c of the binomial form from the two lowest members:
    C_1 / C_0 = -q c e_P / e (SISTERS §1.3, `binomial_weight.py`); nan when it cannot be formed."""
    q = int(order)
    j_values = np.asarray(j_values)
    step = int(np.sign(q))
    if not q or 0 not in j_values or step not in j_values or planet_e <= 0:
        return float('nan')
    c0 = complex(np.asarray(coefficients)[j_values == 0][0])
    c1 = complex(np.asarray(coefficients)[j_values == step][0])
    if c0 == 0:
        return float('nan')
    return float(-(c1 / c0).real * e / (abs(q) * planet_e))


def mu_eff(coefficient_magnitude, a_au: float):
    """mu_eff = 2 |C| a of a term 2 C cos(sigma) (G M_sun = 1)."""
    return 2 * np.abs(coefficient_magnitude) * a_au


def small_period_yr(coefficient_magnitude, a_au: float, body_coefficient: int):
    """T0 = 2 pi / (|k| n sqrt(3 mu_eff)) [yr] of the pendulum with force 2 |C| (n = 2 pi a^-3/2 rad/yr)."""
    n = TWO_PI * a_au**-1.5
    with np.errstate(divide='ignore'):
        return TWO_PI / (abs(body_coefficient) * n * np.sqrt(3 * mu_eff(coefficient_magnitude, a_au)))


def member_spacing_au(a_au: float, varpi_rate_rad_yr: float, planet_varpi_rate_rad_yr: float, body_coefficient: int) -> float:
    """delta a = (2/3) (a/n) |varpi_dot - varpi_dot_P| / |k| [au]: exact locations of neighbouring members
    (a linear estimate at fixed rates, SISTERS §1.1)."""
    n = TWO_PI * a_au**-1.5
    return float(2 / 3 * a_au / n * abs(varpi_rate_rad_yr - planet_varpi_rate_rad_yr) / abs(body_coefficient))
