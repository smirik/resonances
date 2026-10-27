"""Kinematic origin: an oscillation of the angle carried by its varpi part, not by a (SPEC §4.7).

check_mmr_kinematic fails (the mechanism is found) when the slow angle follows p varpi
(corr >= mmr_kin_corr_min) and the slow a does not answer (slow a_dot R^2 <= mmr_kin_slow_r2_max). For an episodic
or librating angle the tree judges the episode's cycle instead, or the slow centre (`slow_centre_kinematics`, K2:
the centre drifts with the secular part while the episodes are a ripple on it; `decide.kinematic_check`).
The concrete cause goes to `Kinematics.detail` (OriginDetail): varpi_term, secular_cycle
(the e-cycle 2 pi/(g - g5) or the double omega cycle pi/|omega_dot|) or planet_ripple.

The varpi-rate modulation follows from the linear secular solution, a free and a forced vector rotating at their own
frequencies (Murray & Dermott 1999, eqs. 7.18-7.19, p. 277): for z = e exp(i varpi) = e_f exp(i g t) + e_F exp(i g5 t),
varpi_dot = Im(conj(z) z_dot) / |z|^2 = g5 + (g - g5)(e_f^2 + e_f e_F cos theta) / (e_f^2 + e_F^2 + 2 e_f e_F cos theta),
monotonic in cos theta, so its extremes are g5 + (g - g5) e_f / (e_f + e_F) and g5 + (g - g5) e_f / (e_f - e_F).
|z| is symmetric in e_f <-> e_F, so e alone gives only the unordered pair (e_max +- e_min)/2: which radius is the
free one (turning at g) is read from the vector z itself (`free_forced_split`). When e_f = e_F the vector passes
through 0: varpi is undefined there and its rate unbounded.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
from scipy import optimize

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.geometry import Geometry
from resonances.resonance.analysis.mmr.context import ARCSEC_TO_RAD, linear_slope
from resonances.resonance.analysis.mmr.models import Context, Kinematics, MMRSeries, Preprocessed, Response, SlowComponents
from resonances.resonance.analysis.mmr.response import correlation
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


def _unwrap(x: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(x, dtype=float))


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
    varpi = _unwrap(series.varpi) if settings is None else body_varpi_slow_unwrap(series, settings)[0]
    psi = np.asarray(angle_unwrapped, dtype=float) - spec.body_varpi_coefficient * varpi
    if spec.body_node_coefficient:
        psi = psi - spec.body_node_coefficient * _unwrap(series.Omega)
    node_coefficients = spec.planet_node_coefficients or (0,) * len(spec.planets)
    for name, p_planet, v_planet in zip(spec.planets, spec.planet_varpi_terms, node_coefficients):
        planet = series.planets.get(name)
        if planet is None:
            continue
        if p_planet:
            psi = psi - p_planet * _on_grid(t, planet.t_yr, _unwrap(planet.varpi))
        if v_planet:
            psi = psi - v_planet * _on_grid(t, planet.t_yr, _unwrap(planet.Omega))
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
    p_rate = varpi_coefficient * np.gradient(_unwrap(varpi_slow), t_yr)
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
    p_varpi = varpi_coefficient * _unwrap(slow.varpi_slow)
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


def varpi_rate_modulation(e: np.ndarray, g_body_rad_yr: float, g5_rad_yr: float) -> Tuple[float, float, float, float, float]:
    """From e alone: e_free = (e_max + e_min)/2, e_forced = (e_max - e_min)/2, predicted varpi_dot min/max [rad/yr],
    period 2pi/|g - g5| [yr].

    e cannot tell which radius is free (|z| is symmetric in the swap): this form assumes the free one is the larger,
    i.e. that `g_body_rad_yr` is the free frequency and varpi turns at it on average. When the forced term dominates
    (varpi turns at g5) it is wrong; `free_forced_split` reads the branch from the vector (what analyse_kinematics uses).
    e_min = 0 gives the unbounded range of `varpi_rate_range`, never a division by zero."""
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    if not len(e):
        return NAN, NAN, NAN, NAN, NAN
    e_free = float((e.max() + e.min()) / 2)
    e_forced = float((e.max() - e.min()) / 2)
    low, high = varpi_rate_range(e_free, e_forced, g_body_rad_yr, g5_rad_yr)
    return e_free, e_forced, low, high, _modulation_period(g_body_rad_yr, g5_rad_yr)


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
    omega = _unwrap(omega)
    if len(t_yr) < MIN_POINTS:
        return NAN, NAN
    rate = float(np.polyfit(t_yr, omega, 1)[0])
    period = np.pi / abs(rate) if rate != 0 else np.inf
    mask = np.isfinite(e_slow)
    return float(period), correlation(e_slow[mask], np.cos(2 * omega[mask]))


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
    centre, a low-pass at W_c or three samples."""
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


def _empty(check: CheckValue) -> Kinematics:
    return Kinematics(*([NAN] * 15), check=check, detail=str(OriginDetail.NONE))


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
    varpi_slow_on_t = np.interp(t, slow_t, _unwrap(slow.varpi_slow))
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
    omega_on_slow = np.interp(slow_t, np.asarray(series.t_yr, float), _unwrap(series.omega))
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
    )
    kinematics.centre_corr_rate, kinematics.centre_slope, kinematics.centre_intercept_rad_yr = slow_centre_kinematics(
        series, prep, geometry, settings
    )
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
    }
