"""Kinematic origin: an oscillation of the angle carried by its varpi part, not by a (SPEC §4.7).

check_mmr_kinematic fails (the mechanism is found) when the slow angle follows p varpi
(corr >= mmr_kin_corr_min) and the slow a does not answer (slow a_dot R^2 <= mmr_kin_slow_r2_max). For an episodic
or librating angle the tree judges the episode's cycle instead, or the slow centre (`slow_centre_kinematics`, K2:
the centre drifts with the secular part while the episodes are a ripple on it; `decide.kinematic_check`).
The concrete cause goes to `Kinematics.detail` (OriginDetail): varpi_term, secular_cycle
(the e-cycle 2 pi/(g - g5) or the double omega cycle pi/|omega_dot|) or planet_ripple.
K1 on the decisive episode and the swing shares (`episode_kinematics`, A5 W28) feed the margin K1e
(`decide.kinematic_episode_margin`): with a flat a and a secular period it is the measured physics of a kinematic
libration.

The varpi-rate modulation follows from the linear secular solution, a free and a forced vector rotating at their own
frequencies (Murray & Dermott 1999, eqs. 7.18-7.19, p. 277): for z = e exp(i varpi) = e_f exp(i g t) + e_F exp(i g5 t),
varpi_dot = Im(conj(z) z_dot) / |z|^2 = g5 + (g - g5)(e_f^2 + e_f e_F cos theta) / (e_f^2 + e_F^2 + 2 e_f e_F cos theta),
monotonic in cos theta, so its extremes are g5 + (g - g5) e_f / (e_f + e_F) and g5 + (g - g5) e_f / (e_f - e_F).
|z| is symmetric in e_f <-> e_F, so e alone gives only the unordered pair (e_max +- e_min)/2: which radius is the
free one (turning at g) is read from the vector z itself (`free_forced_split`). When e_f = e_F the vector passes
through 0: varpi is undefined there and its rate unbounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.filtering import unwrapped
from resonances.resonance.analysis.geometry import Geometry
from resonances.resonance.analysis.mmr.context import ARCSEC_TO_RAD, linear_slope, planet_frequency
from resonances.resonance.analysis.mmr.models import (
    PLANET_NAMES_ORDER,
    CauseLevel,
    Context,
    Kinematics,
    MMRSeries,
    Preprocessed,
    Response,
    SlowComponents,
    SlowOrigin,
    SlowScale,
    SlowSource,
    SlowX,
)
from resonances.resonance.analysis.mmr.response import b1_amplitude, correlation, identity
from resonances.resonance.analysis.mmr.sampling import body_varpi_slow_unwrap
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue, OriginDetail

NAN = float('nan')
TWO_PI = 2 * np.pi
G5_RAD_YR = PLANETARY_FREQUENCIES['g5'] * ARCSEC_TO_RAD
MIN_POINTS = 3  # a slope or a correlation needs at least three samples
# Numerical resolution of the free/forced split (not thresholds): the vector is smooth on P_c, a few thousand
# samples fix its two amplitudes and g (what P_c leaks is noise to the least squares, not a bias); the Brent bracket
# starts a tenth of |g - g5| from the first guess.
SPLIT_MAX_POINTS = 4096
BRACKET_FRACTION = 0.1
ROUND_OFF = float(np.sqrt(np.finfo(float).eps))  # a remainder below this fraction of |z| is round-off


def _on_grid(t_yr: np.ndarray, t_source: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values if len(values) == len(t_yr) else np.interp(t_yr, np.asarray(t_source, float), values)


def lambda_part(series: MMRSeries, angle_unwrapped: np.ndarray, settings: Optional[AnalysisSettings] = None) -> np.ndarray:
    """psi_lambda = sigma - p varpi - sum_j p_j varpi_j - nodal terms [rad] (unwrapped).

    With `settings` the body's varpi has the winding the unwrap of sigma gave it (`sampling.body_varpi_slow_unwrap`),
    so psi has no 2pi p steps where e passes near 0; without, np.unwrap. Planets absent from `series.planets` are
    skipped (their term is then a straight line in psi; the caller may remove its rate)."""
    spec = series.resonance
    t = np.asarray(series.t_yr, float)
    varpi = unwrapped(series.varpi) if settings is None else body_varpi_slow_unwrap(series, settings)[0]
    psi = np.asarray(angle_unwrapped, dtype=float) - spec.body_varpi_coefficient * varpi
    if spec.body_node_coefficient:
        psi = psi - spec.body_node_coefficient * unwrapped(series.Omega)
    node_coefficients = spec.planet_node_coefficients or (0,) * len(spec.planets)
    for name, p_planet, v_planet in zip(spec.planets, spec.planet_varpi_terms, node_coefficients):
        planet = series.planets.get(name)
        if planet is None:
            continue
        if p_planet:
            psi = psi - p_planet * _on_grid(t, planet.t_yr, unwrapped(planet.varpi))
        if v_planet:
            psi = psi - v_planet * _on_grid(t, planet.t_yr, unwrapped(planet.Omega))
    return psi


def cancellation(
    t_yr: np.ndarray, psi_lambda: np.ndarray, varpi_coefficient: int, varpi_slow: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """(psi_lambda rate [rad/yr], its rms from a line [rad], p varpi_dot mean, min, max [rad/yr]); cancel ratio = first / third."""
    t_yr, psi_lambda, varpi_slow = (np.asarray(x, dtype=float) for x in (t_yr, psi_lambda, varpi_slow))
    if len(t_yr) < MIN_POINTS:
        return NAN, NAN, NAN, NAN, NAN
    slope, intercept = np.polyfit(t_yr, psi_lambda, 1)
    rms = float(np.sqrt(np.mean((psi_lambda - (slope * t_yr + intercept)) ** 2)))
    p_rate = varpi_coefficient * np.gradient(unwrapped(varpi_slow), t_yr)
    return float(slope), rms, float(np.mean(p_rate)), float(np.min(p_rate)), float(np.max(p_rate))


def _slow_window(slow: SlowComponents) -> np.ndarray:
    return filtering.window_mask(slow.t_yr, slow.window)


def _detrend(t_yr: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x - np.polyval(np.polyfit(t_yr, x, 1), t_yr)


def kinematic_correlations(slow: SlowComponents, varpi_coefficient: int) -> Tuple[float, float]:
    """(corr(sigma_dot_slow, p varpi_dot_slow), corr of the detrended sigma_slow and p varpi_slow), inside the slow window."""
    if varpi_coefficient == 0:
        return NAN, NAN
    full_t = np.asarray(slow.t_yr, float)
    sigma_rate = np.gradient(np.asarray(slow.angle_slow, float), full_t)
    p_varpi = varpi_coefficient * unwrapped(slow.varpi_slow)
    p_rate = np.gradient(p_varpi, full_t)
    mask = _slow_window(slow)
    if np.count_nonzero(mask) < MIN_POINTS:
        return NAN, NAN
    t = full_t[mask]
    rate_corr = correlation(sigma_rate[mask], p_rate[mask])
    detrended = correlation(_detrend(t, np.asarray(slow.angle_slow, float)[mask]), _detrend(t, p_varpi[mask]))
    return rate_corr, detrended


def varpi_rate_range(e_free: float, e_forced: float, g_free_rad_yr: float, g5_rad_yr: float) -> Tuple[float, float]:
    """(min, max) of varpi_dot [rad/yr] for z = e_free exp(i g t) + e_forced exp(i g5 t) (module docstring), any branch.

    e_free = e_forced: the vector passes through the origin, varpi jumps by pi and the rate is unbounded: (-inf, inf).
    e_free = 0: the forced vector alone, (g5, g5) whatever g. Both radii zero or a value missing: (nan, nan)."""
    radii = np.array([e_free, e_forced], dtype=float)
    if not (np.all(np.isfinite(radii)) and np.isfinite(g5_rad_yr)) or np.any(radii < 0) or e_free + e_forced == 0:
        return NAN, NAN
    if e_free == 0:
        return float(g5_rad_yr), float(g5_rad_yr)
    if not np.isfinite(g_free_rad_yr):
        return NAN, NAN
    if e_free == e_forced:
        return -np.inf, np.inf
    ends = [g5_rad_yr + (g_free_rad_yr - g5_rad_yr) * e_free / (e_free + sign * e_forced) for sign in (1.0, -1.0)]
    return float(min(ends)), float(max(ends))


def _modulation_period(g_free_rad_yr: float, g5_rad_yr: float) -> float:
    """2 pi / |g - g5| [yr]; inf for g = g5, nan for a missing rate."""
    difference = abs(np.float64(g_free_rad_yr) - g5_rad_yr)
    if not np.isfinite(difference):
        return NAN
    return float(TWO_PI / difference) if difference > 0 else np.inf


def _free_rate(t_yr: np.ndarray, free: np.ndarray) -> float:
    """Slope of the unwrapped argument of the free vector [rad/yr]; nan when it passes too close to 0 to have one."""
    if len(t_yr) < MIN_POINTS or not np.all(np.abs(free) > 0):
        return NAN
    return linear_slope(t_yr, np.unwrap(np.angle(free)))


def _two_vector_fit(t_yr: np.ndarray, z: np.ndarray, g_free_rad_yr: float, g5_rad_yr: float) -> Tuple[complex, complex, float]:
    """Least squares z ~ A exp(i g t) + B exp(i g5 t): (A, B, residual sum of squares)."""
    design = np.column_stack([np.exp(1j * g_free_rad_yr * t_yr), np.exp(1j * g5_rad_yr * t_yr)])
    (free, forced), *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - design @ np.array([free, forced])
    return complex(free), complex(forced), float(np.vdot(residual, residual).real)


def free_forced_split(t_yr: np.ndarray, e: np.ndarray, varpi: np.ndarray, g5_rad_yr: float) -> Tuple[float, float, float]:
    """(e_free, e_forced, g_free [rad/yr]) of z = e exp(i varpi) = e_free exp(i g t) + e_forced exp(i g5 t).

    The first guess of g is the rate of the argument of z minus its projection on exp(i g5 t); then g is the minimum of
    the two-vector least-squares residual (Brent from that guess), which a record shorter than 2 pi/|g - g5| needs
    (the projection alone is biased there: 0.6 of a cycle gives g 45 % off). The branch follows from the vector: with
    e_free < e_forced varpi turns at g5 on average and the free radius is the smaller one. nan without three finite
    samples; when the remainder has no rate (e_free ~ 0) g is nan and e_free its mean modulus."""
    t_yr, e, varpi = (np.asarray(x, dtype=float) for x in (t_yr, e, varpi))
    finite = np.isfinite(t_yr) & np.isfinite(e) & np.isfinite(varpi)
    if np.count_nonzero(finite) < MIN_POINTS or not np.isfinite(g5_rad_yr):
        return NAN, NAN, NAN
    stride = max(1, int(np.ceil(np.count_nonzero(finite) / SPLIT_MAX_POINTS)))
    t, z = t_yr[finite][::stride], (e[finite] * np.exp(1j * varpi[finite]))[::stride]
    t = t - t[0]
    forced_basis = np.exp(1j * g5_rad_yr * t)
    forced = np.mean(z * np.conj(forced_basis))
    remainder = z - forced * forced_basis
    if np.mean(np.abs(remainder)) <= ROUND_OFF * np.mean(np.abs(z)):  # a pure forced vector: no free term to time
        return 0.0, float(np.abs(forced)), NAN
    guess = _free_rate(t, remainder)
    if not np.isfinite(guess) or guess == g5_rad_yr:
        return float(np.mean(np.abs(remainder))), float(np.abs(forced)), guess
    span = abs(guess - g5_rad_yr)
    best = optimize.minimize_scalar(lambda g: _two_vector_fit(t, z, g, g5_rad_yr)[2], bracket=(guess, guess + BRACKET_FRACTION * span))
    g_free = float(best.x) if best.success and np.isfinite(best.x) else guess
    free, forced, _ = _two_vector_fit(t, z, g_free, g5_rad_yr)
    return float(abs(free)), float(abs(forced)), g_free


def short_period_vector(series: MMRSeries, cutoff_yr: float, settings: AnalysisSettings) -> Tuple[np.ndarray, np.ndarray]:
    """(e, varpi) of the body's vector e exp(i varpi) low-passed at the short-period cutoff P_c: the components are
    filtered (the filter is linear in the vector, not in its modulus and angle); the raw vector when the low-pass
    cannot run (a grid that is not uniform, P_c not longer than two steps).

    The free/forced split reads this vector, not the slow one smoothed at W_c = 3 P_cyc: when P_cyc is the e-cycle
    2 pi/(g - g5) itself, W_c removes the very cycle the split measures (37087 3J+8S-3: e_forced 0.0003 at W_c,
    0.0286 from the cutoff series, g 231.28"/yr)."""
    t = np.asarray(series.t_yr, float)
    e, varpi = np.asarray(series.e, float), np.asarray(series.varpi, float)
    x = filtering.lowpass_or_none(t, e * np.cos(varpi), cutoff_yr, settings)
    y = filtering.lowpass_or_none(t, e * np.sin(varpi), cutoff_yr, settings)
    if x is None or y is None:
        return e, varpi
    return np.hypot(x, y), np.arctan2(y, x)


def two_omega_cycle(t_yr: np.ndarray, omega: np.ndarray, e_slow: np.ndarray) -> Tuple[float, float]:
    """(pi / |omega_dot| [yr], corr(e_slow, cos 2 omega)): the kinematic mechanism of 285637."""
    t_yr, e_slow = np.asarray(t_yr, float), np.asarray(e_slow, float)
    omega = unwrapped(omega)
    if len(t_yr) < MIN_POINTS:
        return NAN, NAN
    rate = float(np.polyfit(t_yr, omega, 1)[0])
    period = np.pi / abs(rate) if rate != 0 else np.inf
    mask = np.isfinite(e_slow)
    return float(period), correlation(e_slow[mask], np.cos(2 * omega[mask]))


# Secular phases of `secular_forcing_fit` (A2 W28): the double perihelion argument 2 omega and the body's varpi
# against the giant planets' modes, varpi - g_j t (j = 5..8, the literature frequencies). The planet's osculating
# varpi is not used (it does not turn at its own mode for Uranus and Neptune, nor for the inner planets), and no phase
# is needed: the fit's cosine and sine absorb a constant phase, so neither the J2000 phases nor the epoch enter. The
# inner modes g1-g4 are left out: chaotic, g3 and g4 nearly equal, and each extra term is one more period a slow x
# could match by chance.
SECULAR_2OMEGA = '2omega'
SECULAR_MODE_PLANETS = ('Jupiter', 'Saturn', 'Uranus', 'Neptune')
CAUSE_2OMEGA, CAUSE_G_GJ = 'secular_2omega', 'secular_g_gj'  # own_x_cause


def secular_mode_term(planet: str) -> str:
    """The key of the term varpi - varpi_j of a planet's mode, e.g. 'g-g5' for Jupiter."""
    return f'g-g{PLANET_NAMES_ORDER.index(planet) + 1}'


def secular_varpi_terms(t_yr: np.ndarray, varpi: np.ndarray) -> Dict[str, np.ndarray]:
    """{'g-g5': varpi - g5 t, ...} [rad] of the body's unwrapped varpi against each giant planet's mode (the constant
    phase left out, block comment above)."""
    t, varpi = np.asarray(t_yr, float), unwrapped(varpi)
    return {secular_mode_term(name): varpi - planet_frequency(name, 'g') * t for name in SECULAR_MODE_PLANETS}


def secular_cause(term: str) -> str:
    """own_x_cause of a secular term: 'secular_2omega' or 'secular_g_gj' ('' for none)."""
    if not term:
        return ''
    return CAUSE_2OMEGA if term == SECULAR_2OMEGA else CAUSE_G_GJ


def _harmonic_r2(theta: np.ndarray, y: np.ndarray, harmonics: int, samples_per_term: int) -> float:
    columns = [np.ones_like(theta)] + [f(k * theta) for k in range(1, harmonics + 1) for f in (np.cos, np.sin)]
    if len(y) < samples_per_term * len(columns) or not np.var(y) > 0:
        return NAN
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(1.0 - np.var(y - design @ coefficients) / np.var(y))


def secular_forcing_terms(
    t_yr: np.ndarray,
    y: np.ndarray,
    omega: np.ndarray,
    varpi_terms: Mapping[str, np.ndarray],
    harmonics: int,
    samples_per_term: int = MIN_POINTS,
    period_max_yr: float = np.inf,
) -> Dict[str, Tuple[float, float]]:
    """{term: (R^2, period [yr])} of y fitted alone on each secular phase theta: [1, cos k theta, sin k theta],
    k = 1..`harmonics`, for theta = 2 omega ('2omega', period pi / |omega_dot|) and each varpi term (period 2 pi /
    |theta_dot|, e.g. 2 pi / |g - g5|). R^2 is nan below `samples_per_term` samples per fitted term, for a constant y,
    and for a term whose period exceeds `period_max_yr`: over a fraction of its cycle cos and sin of a nearly uniform
    phase are a low polynomial in t and fit any slow trend. The rate is the least-squares slope of theta over the
    samples given."""
    t, y = np.asarray(t_yr, float), np.asarray(y, float)
    phases = {SECULAR_2OMEGA: 2 * unwrapped(omega)}
    phases.update({name: np.asarray(theta, float) for name, theta in varpi_terms.items()})
    result = {}
    for name, theta in phases.items():
        if len(t) < MIN_POINTS:
            result[name] = (NAN, NAN)
            continue
        rate = float(np.polyfit(t - t[0], theta, 1)[0])
        period = TWO_PI / abs(rate) if rate != 0 else np.inf
        r2 = _harmonic_r2(theta, y, harmonics, samples_per_term) if period <= period_max_yr else NAN
        result[name] = (r2, float(period))
    return result


def secular_forcing_fit(
    t_yr: np.ndarray,
    y: np.ndarray,
    omega: np.ndarray,
    varpi_terms: Mapping[str, np.ndarray],
    harmonics: int,
    samples_per_term: int = MIN_POINTS,
    period_max_yr: float = np.inf,
) -> Tuple[float, float, str]:
    """(R^2, period [yr], term) of the secular phase that explains y best (`secular_forcing_terms`): is y driven by the
    double omega cycle or by the e-cycle varpi - varpi_j? (nan, nan, '') when no term is measured. Used by the second
    capture's route F3 (y = x_ddot) and the slow x (y = x_dot, `owner.slow_x`)."""
    terms = secular_forcing_terms(t_yr, y, omega, varpi_terms, harmonics, samples_per_term, period_max_yr)
    measured = [(r2, period, name) for name, (r2, period) in terms.items() if np.isfinite(r2)]
    if not measured:
        return NAN, NAN, ''
    return max(measured, key=lambda item: item[0])


def kinematic_check(
    corr_rate: float, slow_r2: float, n_independent: float, varpi_coefficient_sum: int, settings: AnalysisSettings
) -> CheckValue:
    """fail: corr >= mmr_kin_corr_min and slow R^2 <= mmr_kin_slow_r2_max; not_applicable for a zero-order angle (p = sum p_j = 0);
    not_measured with fewer than mmr_kin_min_points independent points (length / W_c).

    `varpi_coefficient_sum` is |p| + sum |p_j| (a sum of signed coefficients could cancel)."""
    if varpi_coefficient_sum == 0:
        return CheckValue.NOT_APPLICABLE
    if not np.isfinite(n_independent) or n_independent < settings.mmr_kin_min_points:
        return CheckValue.NOT_MEASURED
    if not (np.isfinite(corr_rate) and np.isfinite(slow_r2)):
        return CheckValue.NOT_MEASURED
    follows = corr_rate >= settings.mmr_kin_corr_min and slow_r2 <= settings.mmr_kin_slow_r2_max
    return CheckValue.FAIL if follows else CheckValue.PASS


def _periods_match(period_yr: float, reference_yr: float, tolerance: float) -> bool:
    return bool(
        np.isfinite(period_yr) and np.isfinite(reference_yr) and reference_yr > 0 and abs(period_yr / reference_yr - 1) <= tolerance
    )


def origin_detail(
    kinematics: Kinematics,
    response: Response,
    settings: AnalysisSettings,
    period_cycle_yr: float = NAN,
    ripple_period_yr: float = NAN,
    check: Optional[CheckValue] = None,
) -> str:
    """The cause for a human: 'varpi_term' (B1 fail or cancellation), 'secular_cycle' (the angle's period matches
    the e-cycle or the double omega cycle), 'planet_ripple' (matches a planetary inequality), '' otherwise.

    A cause is named only when the kinematic origin is found (check_mmr_kinematic or B1 fails; `check`: the
    check as the decision tree reads it, `decide.kinematic_check`, default `kinematics.check`). The angle's
    period `period_cycle_yr` matches a reference period within mmr_kin_period_tol; without a match the cause
    is the varpi term.
    """
    check = kinematics.check if check is None else check
    found = check == CheckValue.FAIL or response.a_amplitude == CheckValue.FAIL
    if not found:
        return str(OriginDetail.NONE)
    tolerance = settings.mmr_kin_period_tol
    if _periods_match(period_cycle_yr, kinematics.modulation_period_yr, tolerance) or _periods_match(
        period_cycle_yr, kinematics.two_omega_period_yr, tolerance
    ):
        return str(OriginDetail.SECULAR_CYCLE)
    if _periods_match(period_cycle_yr, ripple_period_yr, tolerance):
        return str(OriginDetail.PLANET_RIPPLE)
    return str(OriginDetail.VARPI_TERM)


def slow_centre_kinematics(
    series: MMRSeries, prep: Preprocessed, geometry: Geometry, settings: AnalysisSettings
) -> Tuple[float, float, float]:
    """(corr, slope, intercept [rad/yr]) of the rate of the slow centre c(t) (`Geometry.centre`, low-passed at W_c)
    regressed on the rate of the secular part s = sigma - psi_lambda of the angle (p varpi, the planets' varpi and the
    nodal terms) low-passed at the same W_c, farther than W_c from the window's edges (K2, the kinematic drift of the
    slow centre). The model of the centre is the uniform lambda part plus s: its rate is a constant plus s_dot, so the
    correlation and the slope read the secular part alone, and the intercept should be the lambda part's own rate
    (21028 7E-10M-6+9: corr 0.998, slope 1.015, intercept -0.0300 deg/yr against -0.0292 deg/yr). nan without a
    centre, a low-pass at W_c or three samples, and nan with fewer than mmr_kin_min_points independent points
    ((window - 2 W_c) / W_c, K1's guard): over a fraction of W_c two smooth curves correlate whatever they are (485 Genua
    3J-1S-1: 15 kyr of a 42-kyr W_c right after a capture, corr 0.98 at slope 35)."""
    centre = geometry.centre
    t_c = np.asarray(centre.t_yr, float)
    if len(t_c) < MIN_POINTS or not (np.isfinite(centre.window_yr) and centre.window_yr > 0):
        return NAN, NAN, NAN
    t = np.asarray(prep.t_yr, float)
    secular = np.asarray(prep.angle_unwrapped, float) - lambda_part(series, prep.angle_unwrapped, settings)
    secular_f = filtering.lowpass_or_none(t, secular, centre.window_yr, settings)
    if secular_f is None:
        return NAN, NAN, NAN
    start, end = geometry.window
    if (end - start - 2 * centre.window_yr) / centre.window_yr < settings.mmr_kin_min_points:
        return NAN, NAN, NAN
    inside = (t_c - start > centre.window_yr) & (end - t_c > centre.window_yr)
    if np.count_nonzero(inside) < MIN_POINTS:
        return NAN, NAN, NAN
    centre_rate = np.gradient(np.asarray(centre.centre_rad, float), t_c)[inside]
    secular_rate = np.gradient(np.interp(t_c, t, secular_f), t_c)[inside]
    spread = secular_rate - secular_rate.mean()
    variance = float(np.dot(spread, spread))
    if not variance > 0:
        return NAN, NAN, NAN
    slope = float(np.dot(spread, centre_rate - centre_rate.mean())) / variance  # least squares centre_rate ~ slope s_dot + b
    intercept = float(centre_rate.mean() - slope * secular_rate.mean())
    return correlation(centre_rate, secular_rate), slope, intercept


# Why a kinematic correlation has no number ('' when it has one): kin_check_reason for the slow-angle one (K1,
# `check_reason`), kin_k1_episode_reason for K1 on the decisive episode (`episode_kinematics`)
K1E_RETROGRADE = 'retrograde'  # no varpi convention for i > 90 deg
K1E_NO_VARPI = 'no_body_varpi_term'  # p = 0: sigma has no part of the body's varpi to follow
K1E_NO_EPISODE = 'no_episode'  # the angle does not librate: there is no episode to read
K1E_FEW_POINTS = 'too_few_points'  # episode length / response cutoff < mmr_kin_min_points
K1E_NO_FILTER = 'filter_not_run'  # the low-pass at the response cutoff could not run (grid, cutoff)
K1E_NO_SWING = 'no_swing'  # sigma or p varpi does not move over the episode


def _half_range(x: np.ndarray) -> float:
    return float((np.max(x) - np.min(x)) / 2) if len(x) else NAN


def episode_kinematics(
    series: MMRSeries,
    prep: Preprocessed,
    window: Tuple[float, float],
    cutoff_yr: float,
    settings: AnalysisSettings,
) -> Tuple[float, float, float, str]:
    """K1 on the decisive episode (A5, W28, cards g05-3, g10-2): (corr, swing share, lambda swing share, reason).

    sigma (prep's unwrapped angle) and p varpi (the body's varpi with the winding of the unwrap of sigma,
    `sampling.body_varpi_slow_unwrap`) are low-passed at `cutoff_yr` (the episode's response cutoff max(P_c,
    mmr_response_cutoff_cycles x P_e), `response.response_cutoff_yr`), cut to the episode `window` [t0, t1] and
    detrended there; the lambda part is their difference (sigma - p varpi; the planets' varpi and nodal terms, nearly
    uniform, go with it and leave with the trend).
    - corr: correlation of the detrended sigma with the detrended p varpi (`kin_k1_episode`): ~1 when the episode's
      swing is the oscillation of p varpi (285637 -4 varpi, beating at the 2 omega cycle).
    - swing share: half-range of the detrended p varpi / half-range of the detrended sigma (`kin_swing_share`);
      lambda swing share: the same for the lambda part (`kin_lambda_swing_share`). A kinematic libration has ~1 and ~0;
      a pendulum whose own a swings sigma has a lambda share ~1.
    nan with the reason (K1E_*) for a retrograde angle, p = 0, an episode shorter than mmr_kin_min_points response
    cutoffs, a low-pass that cannot run, or no swing. The caller passes no window without a librating episode."""
    spec = series.resonance
    if spec.retrograde:
        return NAN, NAN, NAN, K1E_RETROGRADE
    p = spec.body_varpi_coefficient
    if p == 0:
        return NAN, NAN, NAN, K1E_NO_VARPI
    t0, t1 = (float(x) for x in window)
    if not (np.isfinite(t0) and np.isfinite(t1) and np.isfinite(cutoff_yr) and cutoff_yr > 0 and t1 > t0):
        return NAN, NAN, NAN, K1E_NO_EPISODE
    if (t1 - t0) / cutoff_yr < settings.mmr_kin_min_points:
        return NAN, NAN, NAN, K1E_FEW_POINTS
    t = np.asarray(prep.t_yr, float)
    sigma = np.asarray(prep.angle_unwrapped, float)
    p_varpi = p * body_varpi_slow_unwrap(series, settings)[0]
    sigma_f = filtering.lowpass_or_none(t, sigma, cutoff_yr, settings)
    p_varpi_f = filtering.lowpass_or_none(t, p_varpi, cutoff_yr, settings)
    if sigma_f is None or p_varpi_f is None:
        return NAN, NAN, NAN, K1E_NO_FILTER
    inside = (t >= t0) & (t <= t1)
    if np.count_nonzero(inside) < MIN_POINTS:
        return NAN, NAN, NAN, K1E_FEW_POINTS
    t_in = t[inside]
    sigma_d = _detrend(t_in, sigma_f[inside])
    p_varpi_d = _detrend(t_in, p_varpi_f[inside])
    lambda_d = sigma_d - p_varpi_d  # detrending is linear: the detrended lambda part
    sigma_half = _half_range(sigma_d)
    if not (sigma_half > 0 and _half_range(p_varpi_d) > 0):
        return NAN, NAN, NAN, K1E_NO_SWING
    corr = correlation(sigma_d, p_varpi_d)
    return float(corr), _half_range(p_varpi_d) / sigma_half, _half_range(lambda_d) / sigma_half, ''


def _empty(check: CheckValue) -> Kinematics:
    reason = K1E_RETROGRADE if check == CheckValue.NOT_APPLICABLE else ''
    return Kinematics(*([NAN] * 15), check=check, detail=str(OriginDetail.NONE), check_reason=reason, k1_episode_reason=reason)


def check_reason(corr_rate: float, n_independent: float, p: int, settings: AnalysisSettings) -> str:
    """Why the slow-angle correlation (K1's number, `kinematic_correlations`) does not judge the record ('' when it
    does): p = 0 (K1E_NO_VARPI), fewer than mmr_kin_min_points independent points (length / W_c: a record shorter
    than a few centre windows, 285637 50 kyr against W_c = 54 kyr; K1E_FEW_POINTS), or no variation (K1E_NO_SWING)."""
    if p == 0:
        return K1E_NO_VARPI
    if not np.isfinite(n_independent) or n_independent < settings.mmr_kin_min_points:
        return K1E_FEW_POINTS
    return '' if np.isfinite(corr_rate) else K1E_NO_SWING


def _missing_planet_rate(series: MMRSeries, planet_rates: Mapping[str, float]) -> float:
    """sum p_j varpi_dot_j over the planets of the term that are not saved (their line is left in psi_lambda)."""
    spec = series.resonance
    total = 0.0
    for name, p_planet in zip(spec.planets, spec.planet_varpi_terms):
        if p_planet and name not in series.planets:
            total += p_planet * planet_rates.get(name, NAN)
    return total


def analyse_kinematics(
    series: MMRSeries,
    prep: Preprocessed,
    slow: SlowComponents,
    geometry: Geometry,
    context: Context,
    response: Response,
    settings: AnalysisSettings,
) -> Kinematics:
    """Every number of SPEC §4.7, the check and the detail.

    Retrograde angles: not_applicable (no varpi convention). e_free, e_forced and the free frequency g come from the
    vector e exp(i varpi) low-passed at the cutoff P_c over the analysis window (`short_period_vector`,
    `free_forced_split`), not from the slope of varpi (which is g5 when the forced term dominates). g5 is the mode
    frequency (G5_RAD_YR, the package's g5), never the measured varpi rate of Jupiter: Jupiter's own vector is
    0.0447 at g5 + 0.0156 at g6, so its instantaneous rate swings between -8.6 and +10.5"/yr with period
    2 pi/(g6 - g5) = 54 kyr. In the 37087 record its varpi slope over the last mmr_varpi_ref_window_yr (5 kyr) is
    -5.67"/yr (e_forced 0.0077 at that rate), over the whole 100 kyr +4.47"/yr; at g5 = 4.257"/yr e_forced is 0.0286.
    The independent points are the window length over W_c. When the slow
    window is empty (a record shorter than 2 W_c), the line fits use the analysis window.
    """
    spec = series.resonance
    if spec.retrograde:
        return _empty(CheckValue.NOT_APPLICABLE)
    p = spec.body_varpi_coefficient
    coefficient_sum = abs(p) + sum(abs(x) for x in spec.planet_varpi_coefficients)
    t = np.asarray(prep.t_yr, float)
    start, end = slow.window
    slow_t = np.asarray(slow.t_yr, float)
    in_slow = _slow_window(slow)
    if np.count_nonzero(in_slow) < MIN_POINTS:
        # a record shorter than 2 W_c: the line fits (cancellation, e range, omega cycle) use the analysis window;
        # the correlations stay not measured (fewer than mmr_kin_min_points independent points anyway)
        in_slow = (slow_t >= prep.window[0]) & (slow_t <= prep.window[1])
    mask = (t >= slow_t[in_slow][0]) & (t <= slow_t[in_slow][-1]) if np.any(in_slow) else np.zeros(len(t), dtype=bool)
    psi = lambda_part(series, prep.angle_unwrapped, settings)
    varpi_slow_on_t = np.interp(t, slow_t, unwrapped(slow.varpi_slow))
    psi_rate, psi_rms, p_rate, p_rate_min, p_rate_max = cancellation(t[mask], psi[mask], p, varpi_slow_on_t[mask])
    psi_rate -= _missing_planet_rate(series, context.planet_varpi_rate_rad_yr)
    with np.errstate(divide='ignore', invalid='ignore'):
        cancel = psi_rate / p_rate if np.isfinite(p_rate) and p_rate != 0 else NAN
    corr_rate, corr_detrended = kinematic_correlations(slow, p)

    g5 = G5_RAD_YR
    e_vector, varpi_vector = short_period_vector(series, prep.cutoff.cutoff_yr, settings)
    split = (t >= prep.window[0]) & (t <= prep.window[1])
    e_free, e_forced, g_free = free_forced_split(t[split], e_vector[split], varpi_vector[split], g5)
    varpi_min, varpi_max = varpi_rate_range(e_free, e_forced, g_free, g5)
    modulation = _modulation_period(g_free, g5)
    omega_on_slow = np.interp(slow_t, np.asarray(series.t_yr, float), unwrapped(series.omega))
    two_omega, e_cos = two_omega_cycle(slow_t[in_slow], omega_on_slow[in_slow], np.asarray(slow.e_slow, float)[in_slow])

    n_independent = max(end - start, 0.0) / slow.window_yr if slow.window_yr > 0 else NAN
    check = kinematic_check(corr_rate, response.slow_adot_r2, n_independent, coefficient_sum, settings)
    kinematics = Kinematics(
        lambda_part_rate_rad_yr=psi_rate,
        lambda_part_resid_rad=psi_rms,
        p_varpi_rate_rad_yr=p_rate,
        p_varpi_rate_min_rad_yr=p_rate_min,
        p_varpi_rate_max_rad_yr=p_rate_max,
        cancel_ratio=float(cancel),
        corr_rate=corr_rate,
        corr_detrended=corr_detrended,
        e_free=e_free,
        e_forced=e_forced,
        varpi_rate_pred_min_rad_yr=varpi_min,
        varpi_rate_pred_max_rad_yr=varpi_max,
        modulation_period_yr=modulation,
        two_omega_period_yr=two_omega,
        e_cos2omega_corr=e_cos,
        check=check,
        detail='',
        check_reason=check_reason(corr_rate, n_independent, p, settings),
        k1_episode_reason=K1E_NO_EPISODE,  # set on the decisive episode by analyse_mmr (`episode_kinematics`)
    )
    kinematics.centre_corr_rate, kinematics.centre_slope, kinematics.centre_intercept_rad_yr = slow_centre_kinematics(
        series, prep, geometry, settings
    )
    kinematics.pred_swing_rad = predicted_swing_rad(p, e_free, e_forced)
    ripple = context.great_inequality.period_yr if context.great_inequality is not None else NAN
    kinematics.detail = origin_detail(kinematics, response, settings, geometry.period_cycle_yr, ripple)
    return kinematics


def kinematics_columns(kinematics: Kinematics) -> dict:
    """CSV columns `kin_*` (the body of `Kinematics.to_columns`)."""
    per_kyr = 1e3
    return {
        'kin_lambda_part_rate_rad_kyr': kinematics.lambda_part_rate_rad_yr * per_kyr,
        'kin_lambda_part_resid_deg': float(np.degrees(kinematics.lambda_part_resid_rad)),
        'kin_p_varpi_rate_rad_kyr': kinematics.p_varpi_rate_rad_yr * per_kyr,
        'kin_p_varpi_rate_min_rad_kyr': kinematics.p_varpi_rate_min_rad_yr * per_kyr,
        'kin_p_varpi_rate_max_rad_kyr': kinematics.p_varpi_rate_max_rad_yr * per_kyr,
        'kin_cancel_ratio': kinematics.cancel_ratio,
        'kin_corr_rate': kinematics.corr_rate,
        'kin_corr_detrended': kinematics.corr_detrended,
        'kin_e_free': kinematics.e_free,
        'kin_e_forced': kinematics.e_forced,
        'kin_varpi_dot_pred_min_arcsec_yr': kinematics.varpi_rate_pred_min_rad_yr / ARCSEC_TO_RAD,
        'kin_varpi_dot_pred_max_arcsec_yr': kinematics.varpi_rate_pred_max_rad_yr / ARCSEC_TO_RAD,
        'kin_modulation_period_kyr': kinematics.modulation_period_yr / per_kyr,
        'kin_two_omega_period_yr': kinematics.two_omega_period_yr,
        'kin_e_cos2omega_corr': kinematics.e_cos2omega_corr,
        'kin_origin_detail': kinematics.detail,
        'kin_centre_corr_rate': kinematics.centre_corr_rate,
        'kin_centre_slope': kinematics.centre_slope,
        'kin_centre_intercept_rad_kyr': kinematics.centre_intercept_rad_yr * per_kyr,
        'kin_check_reason': kinematics.check_reason,
        'kin_k1_episode': kinematics.k1_episode,
        'kin_swing_share': kinematics.swing_share,
        'kin_lambda_swing_share': kinematics.lambda_swing_share,
        'kin_k1_episode_reason': kinematics.k1_episode_reason,
        'kin_pred_swing_deg': float(np.degrees(kinematics.pred_swing_rad)),
        'kin_meas_swing_deg': float(np.degrees(kinematics.meas_swing_rad)),
    }


# --------------------------------------------------------------------------------------------- B10a slow scale
# (W28, cards g08-7, g08-8, g05-2, g05-1): the slower libration of the centre stored at its own scale, its origin and
# its probable cause named with a level. A description; only K3s reads it (`decide.secular_cycle_kinematic`).

CAUSE_PVARPI = 'pvarpi_term'  # slow_cause: the slow centre follows the secular part of sigma at no secular period


def predicted_swing_rad(varpi_coefficient: int, e_free: float, e_forced: float) -> float:
    """The half-swing of p varpi [rad] that the two secular vectors make (kin_pred_swing_deg): z = e_f exp(i g t) +
    e_F exp(i g5 t) turns about the larger radius, so varpi wobbles about that vector's phase by asin(min / max)
    (the angle of the smaller one seen from the tip of the larger, at most pi / 2), with the period 2 pi / |g - g5|;
    sigma carries it times |p|. nan for p = 0 or a radius missing."""
    radii = np.array([e_free, e_forced], dtype=float)
    if varpi_coefficient == 0 or not np.all(np.isfinite(radii)) or np.any(radii < 0) or not radii.max() > 0:
        return NAN
    return float(abs(varpi_coefficient) * np.arcsin(min(radii.min() / radii.max(), 1.0)))


def turn_cycles(times_yr: Sequence[np.ndarray], values_rad: Sequence[np.ndarray], min_cycles: int) -> Tuple[int, float, float]:
    """(independent closed cycles, median full period [yr], median half-amplitude [rad]) of alternating turning points,
    one array pair per run: a run of n turns closes (n - 1) // 2 cycles without a shared half-cycle; the full periods
    t_(i+2) - t_i and the half-amplitudes |v_i - v_(i+1)| / 2 are pooled over the runs. The period is nan below
    `min_cycles` cycles (one cycle is one estimate: nothing to compare), the half-amplitude nan without two turns."""
    cycles, periods, halves = 0, [], []
    for times, values in zip(times_yr, values_rad):
        times, values = np.asarray(times, float), np.asarray(values, float)
        cycles += max(len(times) - 1, 0) // 2
        periods.extend(times[2:] - times[:-2])
        halves.extend(np.abs(np.diff(values)) / 2)
    period = float(np.median(periods)) if periods and cycles >= min_cycles else NAN
    return cycles, period, float(np.median(halves)) if halves else NAN


def _in_spans(t_yr: np.ndarray, spans: Sequence[Tuple[float, float]]) -> np.ndarray:
    inside = np.zeros(len(t_yr), dtype=bool)
    for start, end in spans:
        inside |= (t_yr >= start) & (t_yr <= end)
    return inside


def secular_phases_on(series: MMRSeries, t_yr: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """(omega unwrapped, {'g-g5': varpi - g5 t, ...}) of the body on the grid `t_yr` (as `owner.secular_phases`)."""
    t, source = np.asarray(t_yr, float), np.asarray(series.t_yr, float)
    omega, varpi = unwrapped(series.omega), unwrapped(series.varpi)
    if len(source) != len(t) or not np.array_equal(source, t):
        omega, varpi = np.interp(t, source, omega), np.interp(t, source, varpi)
    return omega, secular_varpi_terms(t, varpi)


def slow_secular_match(
    t_yr: np.ndarray, centre_rate: np.ndarray, omega: np.ndarray, varpi_terms: Mapping[str, np.ndarray], period_yr: float, settings
) -> Tuple[float, float, str, float, float, str]:
    """(offset, its secular period [yr], its term, that term's R^2, the best R^2, the best term) of the slow centre on
    its span (the arrays given): the secular periods pi / |omega_dot| ('2omega') and 2 pi / |g - g_j| (the varpi
    terms) from the body's own phases there; offset = min |P / P_term - 1| over them (the fundamental only: sigma
    carries p varpi, whose wobble has the period of the term itself; harmonics would add chance matches). The R^2 is
    that of c_dot on [1, cos k theta, sin k theta], k <= mmr_owner_capture_harmonics (`secular_forcing_terms`; a term
    is fitted only when mmr_owner_x_min_cycles of its cycles fit in the span): c_dot, not c, so a drift of the centre
    is a constant and not a trend the harmonics must absorb. Several terms within mmr_kin_period_tol (for a varpi that
    turns much faster than the planets' modes 2 pi/|g - g_j| differ by |g_j - g_k| / g: 0.5 % for g5 and g7 at
    231"/yr) are told apart by that R^2: the term is the matching one c_dot follows best. nan / '' where not
    measured."""
    t = np.asarray(t_yr, float)
    if len(t) < MIN_POINTS:
        return NAN, NAN, '', NAN, NAN, ''
    longest = (t[-1] - t[0]) / settings.mmr_owner_x_min_cycles
    terms = secular_forcing_terms(
        t, centre_rate, omega, varpi_terms, settings.mmr_owner_capture_harmonics, settings.mmr_owner_capture_samples_per_term, longest
    )
    offsets = {
        name: abs(period_yr / reference - 1)
        for name, (_, reference) in terms.items()
        if np.isfinite(period_yr) and np.isfinite(reference) and reference > 0
    }
    measured = {name: r2 for name, (r2, _) in terms.items() if np.isfinite(r2)}
    best = max(measured, key=measured.get) if measured else ''
    best_r2 = measured.get(best, NAN)
    if not offsets:
        return NAN, NAN, '', NAN, best_r2, best
    term = min(offsets, key=offsets.get)
    matching = [name for name in offsets if offsets[name] <= settings.mmr_kin_period_tol and name in measured]
    if matching:  # periods the tolerance cannot tell apart (g - g5 and g - g7 of a fast varpi): the one c_dot follows
        term = max(matching, key=measured.get)
    return float(offsets[term]), float(terms[term][1]), term, float(terms[term][0]), best_r2, best


def slow_origin(identity_check: CheckValue, offset: float, settings: AnalysisSettings) -> str:
    """dynamic_candidate when the identity holds on the slow scale (a slow libration a may make: no status follows);
    kinematic when it fails and the period is a secular one (offset <= mmr_kin_period_tol); unresolved otherwise."""
    if identity_check == CheckValue.PASS:
        return str(SlowOrigin.DYNAMIC_CANDIDATE)
    if identity_check == CheckValue.FAIL and np.isfinite(offset) and offset <= settings.mmr_kin_period_tol:
        return str(SlowOrigin.KINEMATIC)
    return str(SlowOrigin.UNRESOLVED)


def slow_cause(
    offset: float, term: str, term_r2: float, best_r2: float, best_term: str, centre_corr: float, p_term: str, settings
) -> Tuple[str, str, str]:
    """(cause, cause term, level) of the slow scale (card g05-2): the period matches a secular period (offset <=
    mmr_kin_period_tol) and the centre follows that cycle (that term's R^2 >= mmr_owner_r2_min, or the slow centre's
    rate follows the secular part of sigma, centre_corr >= mmr_kin_corr_min) -> identified; one of the two ->
    possible (the matching term; else the best-fitting term; else `p_term`, cause pvarpi_term); neither -> unknown."""
    follows = np.isfinite(centre_corr) and centre_corr >= settings.mmr_kin_corr_min
    if np.isfinite(offset) and offset <= settings.mmr_kin_period_tol:
        fits = np.isfinite(term_r2) and term_r2 >= settings.mmr_owner_r2_min
        return secular_cause(term), term, str(CauseLevel.IDENTIFIED if fits or follows else CauseLevel.POSSIBLE)
    if np.isfinite(best_r2) and best_r2 >= settings.mmr_owner_r2_min:
        return secular_cause(best_term), best_term, str(CauseLevel.POSSIBLE)
    if follows:
        return CAUSE_PVARPI, p_term, str(CauseLevel.POSSIBLE)
    return '', '', str(CauseLevel.UNKNOWN)


def _p_term(series: MMRSeries) -> str:
    p = series.resonance.body_varpi_coefficient
    return f'{p}varpi' if p else ''


@dataclass(frozen=True)
class SlowTrack:
    """What a source of the slow scale measured: its scale W_c, its runs (G2c only), sigma_s's rate and a_s on its own
    grid, the spans the slow cycles occupy and their turning points (one array pair per run)."""

    source: str
    cutoff_yr: float
    t_yr: np.ndarray
    rate_rad_yr: np.ndarray
    a_au: np.ndarray
    spans: Tuple[Tuple[float, float], ...]
    turn_times: Tuple[np.ndarray, ...]
    turn_values: Tuple[np.ndarray, ...]
    identity_coefficient: float
    episodes: Tuple = ()


def measure_slow_scale(
    series: MMRSeries, prep: Preprocessed, kinematics: Optional[Kinematics], track: SlowTrack, settings, owner_x: Optional[SlowX] = None
) -> SlowScale:
    """`SlowScale` of a slow track: the closed cycles of its turns (`turn_cycles`); the identity on the slow scale
    (sigma_s_dot against a_s over the spans, `response.identity`); B1 in the band of the slow period (`response.b1_amplitude`
    on the record's a and sigma at P_c); the secular match of c_dot over the whole slow span (`slow_secular_match`, the
    body's own omega and varpi there); the centre correlation (`Kinematics.centre_corr_rate`); the origin and the
    cause with its level (`slow_origin`, `slow_cause`); an owner row (`owner_x`: A2's slow x of the owner with a level)
    takes that cause first (cause_source 'owner_x')."""
    t = np.asarray(track.t_yr, float)
    rate, a_s = np.asarray(track.rate_rad_yr, float), np.asarray(track.a_au, float)
    cycles, period, halfamp = turn_cycles(track.turn_times, track.turn_values, settings.libration_centre_min_estimates)
    t0, t1 = min(start for start, _ in track.spans), max(end for _, end in track.spans)
    inside = _in_spans(t, track.spans) & np.isfinite(rate) & np.isfinite(a_s)
    ratio, corr, check = identity(rate[inside], a_s[inside], track.identity_coefficient, settings)
    span = (t >= t0) & (t <= t1) & np.isfinite(rate)
    offset, period_sec, term, term_r2, best_r2, best = NAN, NAN, '', NAN, NAN, ''
    if not series.resonance.retrograde:  # no varpi convention for i > 90 deg: no secular period, the cause stays unknown
        omega, terms = secular_phases_on(series, t[span])
        offset, period_sec, term, term_r2, best_r2, best = slow_secular_match(t[span], rate[span], omega, terms, period, settings)
    b1 = NAN
    if np.isfinite(period) and np.isfinite(halfamp):
        b1 = b1_amplitude(prep.t_yr, prep.a_f, period, halfamp, track.identity_coefficient, (t0, t1), None, prep.angle_f, settings)[2]
    centre_corr = kinematics.centre_corr_rate if kinematics is not None else NAN
    cause, cause_term, level = slow_cause(offset, term, term_r2, best_r2, best, centre_corr, _p_term(series), settings)
    source = 'centre'
    if owner_x is not None and owner_x.level:  # an owner row: A2's slow x of the owner names the cause first
        cause, cause_term, level, source = owner_x.cause, owner_x.term, owner_x.level, 'owner_x'
    return SlowScale(
        source=track.source,
        cutoff_yr=float(track.cutoff_yr),
        episodes=tuple(track.episodes),
        t_start_yr=float(t0),
        t_end_yr=float(t1),
        n_cycles=cycles,
        period_yr=period,
        halfamp_rad=halfamp,
        identity_slope_ratio=ratio,
        identity_corr=corr,
        identity=check,
        b1_ratio=float(b1),
        secular_offset=offset,
        secular_period_yr=period_sec,
        secular_term=term,
        secular_r2=term_r2,
        centre_corr=float(centre_corr),
        origin=slow_origin(check, offset, settings),
        cause=cause,
        cause_term=cause_term,
        cause_level=level,
        cause_source=source,
    )


def centre_scale_track(coarse, slow_geometry: Geometry) -> Optional[SlowTrack]:
    """The slow track of G2c's angle at the centre scale `coarse` (`geometry.FilteredAngle` at W_c) and its geometry
    (`analyse._decide_at_scale` measured it): its accepted runs, covering the record or not, are the slow episodes
    (37087 3J+8S-3: 17.3-62.4 and 65.8-96.6 kyr at W_c 506 yr); sigma_s = the angle at W_c, a_s = a at W_c
    (`FilteredAngle.slow_f`). None without an accepted run."""
    runs = tuple(run for run in slow_geometry.runs if run.accepted)
    if not runs:
        return None
    return SlowTrack(
        source=str(SlowSource.CENTRE_SCALE),
        cutoff_yr=float(coarse.cutoff_yr),
        t_yr=coarse.t_yr,
        rate_rad_yr=coarse.rate_f,
        a_au=coarse.slow_f,
        spans=tuple((run.t_start_yr, run.t_end_yr) for run in runs),
        turn_times=tuple(run.tp_times_yr for run in runs),
        turn_values=tuple(run.tp_values_rad for run in runs),
        identity_coefficient=float(coarse.identity_coefficient),
        episodes=runs,
    )


def slower_than(track: Optional[SlowTrack], main_period_yr: float, settings: AnalysisSettings) -> bool:
    """A slow track is slower than the main scale: its pooled full period exceeds libration_centre_cycles x the period
    of the main libration (the centre is the angle smoothed over that many cycles, so a swing of the centre is slower
    by definition). True for any track when the angle has no librating episode (`main_period_yr` nan: 37087's fine
    runs are forced waves); False for None, or a track without a full period next to a main libration."""
    if track is None:
        return False
    if not np.isfinite(main_period_yr):
        return True
    periods = [p for times in track.turn_times for p in np.asarray(times, float)[2:] - np.asarray(times, float)[:-2]]
    return bool(periods) and float(np.median(periods)) > settings.libration_centre_cycles * main_period_yr


def interior_centre_turns(geometry: Geometry) -> Tuple[np.ndarray, np.ndarray]:
    """(times [yr], values [rad]) of the slow centre's turns farther than its own window W_c from both window edges
    (nearer, the low-pass of the centre bends at the edge; as `decide.interior_turn_times`)."""
    centre = geometry.centre
    times = np.asarray(centre.turns.times_yr, dtype=float)
    values = np.asarray(centre.turns.values_rad, dtype=float)
    order = np.argsort(times)
    times, values = times[order], values[order]
    start, end = geometry.window
    keep = (times - start > centre.window_yr) & (end - times > centre.window_yr)
    return times[keep], values[keep]


def centre_track_slow(geometry: Geometry, prep: Preprocessed, identity_coefficient: float, settings) -> Optional[SlowTrack]:
    """The slow track of a librating angle without a G2c pass: the slow centre c(t) (`Geometry.centre`, sigma smoothed at
    W_c = libration_centre_cycles x P_cyc, undetrended) and its turns away from the edges (`interior_centre_turns`): at
    least three (one closed slow cycle), else None; a_s = a at P_c low-passed at W_c. No episodes: c(t) is no angle
    the episode rules ran on."""
    centre = geometry.centre
    times, values = interior_centre_turns(geometry)
    t = np.asarray(centre.t_yr, dtype=float)
    if len(times) < 3 or len(t) < MIN_POINTS or not (np.isfinite(centre.window_yr) and centre.window_yr > 0):
        return None
    a_all, t_all = np.asarray(prep.a_f, float), np.asarray(prep.t_yr, float)
    a_on = a_all if len(a_all) == len(t) and np.array_equal(t_all, t) else np.interp(t, t_all, a_all)
    a_slow = filtering.lowpass_or_none(t, a_on, centre.window_yr, settings)
    if a_slow is None:
        return None
    return SlowTrack(
        source=str(SlowSource.CENTRE_TRACK),
        cutoff_yr=float(centre.window_yr),
        t_yr=t,
        rate_rad_yr=np.gradient(np.asarray(centre.centre_rad, dtype=float), t),
        a_au=a_slow,
        spans=((float(times[0]), float(times[-1])),),
        turn_times=(times,),
        turn_values=(values,),
        identity_coefficient=float(identity_coefficient),
    )
