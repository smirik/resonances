"""Planet ripple, the filter cutoff, the filter and the derivatives (SPEC §3 with DELTA item 1).

The cutoff comes from the physics of the angle only: P_c = max(mmr_filter_synodic_factor x the longest synodic
period of the body with the planets of the angle, P_floor / filter_min_libration_cutoffs), or the explicit
mmr_filter_cutoff_yr; P_floor is the shortest libration the angle can have (`libration_period_floor`). The output
step does not enter: the step only decides whether the record can be filtered at all (P_c > 2 dt, else invalid /
undersampled in `analyse`) and whether short-period terms fold into slow ones (`sampling.short_period_step`). A
cutoff that grew with the step (the old 20 dt) removed a 148-yr Trojan libration at 15.9 yr (G = 0.05) and set the
TP prominence, the response scales and the x2 guard by the step instead of by the angle (inv-09, inv-15).

There is no search for forced lines and no separate rate cutoff; sigma_dot is the derivative of the same filtered
angle. Two guards: a measured libration period whose digital filter gain is below mmr_l8_gain_full sets
filter_too_slow, and the shape-dependent checks become not_measured; and the decisive numbers (turning points,
episode boundaries, stalls, energy per cycle, amplitude and centre shift) are measured again at the cutoff x2 on the
interval both cutoffs see (`filter_sensitivity`, DELTA 13a): on the episode the status reads and its counterpart at
2 P_c, matched by time overlap (W21 Q9), else on the whole interval. A number that changes beyond its tolerance (or a
measurement made at one cutoff only) makes the checks that read it not_measured and sets filter_sensitive; what
cannot be compared is 'not checked', never 'unchanged'. Known planetary inequalities enter
sigma linearly through the planets' longitudes and are subtracted exactly, sum_j k_j dlambda_j^slow, when the
planets are saved; otherwise the flag ripple_not_removed and the predicted half-amplitude raise the TP prominence.
A co-orbital angle keeps its planet's slow longitude (no ripple, `preprocess`).

The shortest libration period of the angle (`libration_period_floor`) separates librations from forced waves:
a co-orbital angle has its own (the planet's tadpole period), every other angle the convention
mmr_libration_period_min_yr.

A co-orbital planet (prograde angle whose lambda-part is k (lambda - lambda_j)) has no synodic wave of its
own: its synodic angle is the resonant angle, so it does not bound the cutoff (`coorbital_planet`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis.filtering import butterworth_gain, edge_window, lowpass, time_derivative
from resonances.resonance.analysis import geometry as geometry_module
from resonances.resonance.analysis.geometry import FilteredAngle, Geometry
from resonances.resonance.analysis.mmr import sampling as sampling_module
from resonances.resonance.analysis.mmr.context import trend_line
from resonances.resonance.analysis.mmr.models import (
    Context,
    Cutoff,
    GreatInequality,
    MMRSeries,
    Preprocessed,
    ResonanceSpec,
    Sampling,
    SlowComponents,
    column_number,
)
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Flag

NAN = float('nan')
# omega = n sqrt(27 mu / 4): the linear (small-amplitude) tadpole frequency about L4/L5 of the restricted three-body problem
TADPOLE_MASS_FACTOR = 27.0 / 4.0


def libration_period_floor(spec: ResonanceSpec, context: Context, settings: AnalysisSettings) -> Tuple[float, str]:
    """The shortest libration period this angle can have [yr] and the setting it comes from (0 = off).

    A co-orbital angle (`coorbital_planet`): mmr_coorbital_period_floor_factor x P_orb / sqrt(27 mu / 4), the linear
    tadpole period of its planet (P_orb = 2 pi / n_j, mu = m_j / M_sun; Jupiter 147.8 yr, so a small Trojan
    libration of 147.8 yr is a libration, not a forced wave). Every other angle: mmr_libration_period_min_yr, a
    convention - no physical minimum holds for every planet and order (Codex W1, G3). Without n_j or m_j the
    convention applies to a co-orbital too."""
    planet = coorbital_planet(spec)
    if planet is not None:
        n = getattr(context, 'planet_mean_motion_rad_yr', {}).get(planet, NAN)
        mu = getattr(context, 'planet_mass', {}).get(planet, NAN)
        if np.isfinite(n) and n > 0 and np.isfinite(mu) and mu > 0:
            tadpole = (2 * math.pi / n) / math.sqrt(TADPOLE_MASS_FACTOR * mu)
            return float(settings.mmr_coorbital_period_floor_factor * tadpole), 'mmr_coorbital_period_floor_factor'
    return float(settings.mmr_libration_period_min_yr), 'mmr_libration_period_min_yr'


def choose_cutoff(
    synodic_periods_yr: Mapping[str, float],
    step_yr: float,
    settings: AnalysisSettings,
    period_floor_yr: Optional[float] = None,
) -> Cutoff:
    """P_c = max(mmr_filter_synodic_factor x max P_syn, P_floor / filter_min_libration_cutoffs) [yr]; binding
    'synodic' or 'floor'; the explicit mmr_filter_cutoff_yr wins ('user'). Without planets of the angle the synodic
    bound is 0; `step_yr` only gives samples_per_cutoff.

    The floor bound P_floor / filter_min_libration_cutoffs (`period_floor_yr` from `libration_period_floor`, default
    mmr_libration_period_min_yr) keeps the shortest libration of the angle at G = 0.94 where it sets P_c (a longer
    synodic or user cutoff lowers that gain: the filter guard and the dry-run say so) and removes what is faster: the synodic and orbital terms of a
    co-orbital angle (its planet left out of `synodic_periods_yr`, P_syn -> infinity), forced waves faster than the
    floor. The synodic bound damps the synodic lines when they are slower than that. What a forced wave of a nearby
    commensurability the cutoff lets through does is handled downstream, by the TP prominence
    (`analyse.fast_residual`) and the episode period floor (`decide.librating_episodes`), both at P_floor. Nothing
    here depends on the output step (inv-09: the classical filters, Nesvorny & Morbidelli 1998 and Knezevic & Milani
    2000, also set their bands by the physics; the step is a validity condition, `analyse`)."""
    floor = settings.mmr_libration_period_min_yr if period_floor_yr is None else period_floor_yr
    periods = [float(p) for p in synodic_periods_yr.values() if p is not None and np.isfinite(p) and p > 0]
    bounds = {
        'synodic': settings.mmr_filter_synodic_factor * max(periods) if periods else 0.0,
        'floor': floor / settings.filter_min_libration_cutoffs if floor > 0 else 0.0,
    }
    if settings.mmr_filter_cutoff_yr is not None:
        cutoff, binding = float(settings.mmr_filter_cutoff_yr), 'user'
    else:
        binding = max(bounds, key=lambda name: bounds[name])  # ties: synodic
        cutoff = bounds[binding]
    return Cutoff(cutoff, binding, float(bounds['synodic']), float(bounds['floor']), float(cutoff / step_yr))


def filter_realizable(cutoff_yr: float, step_yr: float) -> bool:
    """The Butterworth low-pass of `filtering.lowpass` exists at this step: P_c > 2 dt. False: the record is
    undersampled for its own filter (invalid / undersampled; `analyse`, dry-run)."""
    return bool(np.isfinite(cutoff_yr) and cutoff_yr > 2 * step_yr)


def _planet_coefficients(spec: ResonanceSpec) -> dict:
    return {name: k for name, k in zip(spec.planets, spec.planet_coefficients) if k != 0}


def planet_ripple(
    t_yr: np.ndarray,
    spec: ResonanceSpec,
    planet_longitudes: Mapping[str, np.ndarray],
    planet_n: Mapping[str, float],
    cutoff_yr: float,
    filter_order: int,
    grid_rtol: Optional[float] = None,
) -> np.ndarray:
    """sum_j k_j dlambda_j^slow [rad], dlambda_j^slow = lowpass_{P_c}(lambda_j - (lambda_j0 + n_j t)) of the unwrapped lambda_j.

    The line lambda_j0 + n_j t is the least-squares line of the unwrapped lambda_j (n_j only guides
    the unwrap): the subtracted term then has no linear part and leaves the rate of sigma unchanged.
    Planets of the angle missing from `planet_longitudes` contribute nothing (the caller checks).
    """
    t = np.asarray(t_yr, dtype=float)
    ripple = np.zeros_like(t)
    for name, k in _planet_coefficients(spec).items():
        if name not in planet_longitudes or planet_longitudes[name] is None:
            continue
        longitude = sampling_module.unwrap_planet_longitude(t, np.asarray(planet_longitudes[name], dtype=float), planet_n[name])
        line = trend_line(t, longitude)
        ripple += k * lowpass(t, longitude - line, cutoff_yr, filter_order, detrend=True, grid_rtol=grid_rtol)
    return ripple


def _band(
    t: np.ndarray, x: np.ndarray, band_yr: Tuple[float, float], filter_order: int, grid_rtol: Optional[float] = None
) -> Optional[np.ndarray]:
    """Band-pass [P_lo, P_hi] as the difference of two low-passes; None when the band cannot be formed."""
    low, high = band_yr
    try:
        return lowpass(t, x, low, filter_order, grid_rtol=grid_rtol) - lowpass(t, x, high, filter_order, grid_rtol=grid_rtol)
    except ValueError:
        return None


def remove_planet_ripple(
    t_yr: np.ndarray,
    angle: np.ndarray,
    ripple: np.ndarray,
    band_yr: Tuple[float, float],
    filter_order: int,
    grid_rtol: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """angle - ripple. Returns (clean angle, ripple half-amplitude [rad], band correlation).

    The subtraction is exact (lambda_j enters sigma linearly); whether it applies is decided by the
    caller from the angle reconstruction. The band correlation is a diagnostic only: a librating angle
    has its own power in the band and decorrelates from the ripple (0.65 / 0.12 / 0.006 at 1 / 5 / 30 deg).
    Half-amplitude = ptp / 2 of the ripple; a correlation that cannot be measured is nan.
    """
    t = np.asarray(t_yr, dtype=float)
    angle = np.asarray(angle, dtype=float)
    ripple = np.asarray(ripple, dtype=float)
    half_amplitude = float(np.ptp(ripple) / 2) if len(ripple) else NAN
    angle_band = _band(t, angle, band_yr, filter_order, grid_rtol)
    ripple_band = _band(t, ripple, band_yr, filter_order, grid_rtol)
    corr = NAN
    if angle_band is not None and ripple_band is not None and np.std(angle_band) > 0 and np.std(ripple_band) > 0:
        corr = float(np.corrcoef(angle_band, ripple_band)[0, 1])
    return angle - ripple, half_amplitude, corr


def predicted_ripple(
    spec: ResonanceSpec, great_inequality: Optional[GreatInequality], cutoff_yr: float, filter_order: int, step_yr: Optional[float] = None
) -> float:
    """r_GI = |k_J dlambda_J - k_S dlambda_S| G(P_GI) [rad]: what the filter leaves when the planets are not saved
    (G the digital gain at the output step `step_yr`; None = the analog limit).

    The great inequality moves Jupiter and Saturn in antiphase (the 5:2 terms exchange their angular momentum:
    dlambda_J = +A_J sin theta, dlambda_S = -A_S sin theta with the half-amplitudes A > 0 of `GreatInequality`), so
    the wave of k_J lambda_J + k_S lambda_S is (k_J A_J - k_S A_S) sin theta: 5J+6S keeps |5 x 0.34 - 6 x 0.83| = 3.3
    deg, not the 6.7 deg of the unsigned sum."""
    if great_inequality is None:
        return 0.0
    coefficients = _planet_coefficients(spec)
    amplitude = abs(
        coefficients.get('Jupiter', 0) * great_inequality.dlambda_jupiter_rad
        - coefficients.get('Saturn', 0) * great_inequality.dlambda_saturn_rad
    )
    if amplitude == 0:
        return 0.0
    return float(amplitude * butterworth_gain(great_inequality.period_yr, cutoff_yr, filter_order, step_yr))


def filter_guard(
    period_cycle_yr: float, cutoff_yr: float, settings: AnalysisSettings, step_yr: Optional[float] = None
) -> Tuple[bool, float]:
    """(filter_too_slow = G(P_cyc) < mmr_l8_gain_full, gain G(P_cyc)); (False, nan) without a period. G is the digital
    gain at the output step `step_yr` (None or not finite = the analog limit dt -> 0; nan at or above Nyquist: too slow).

    The gain, not the ratio P_cyc / P_c (inv-15, Codex): at 15.9 yr a 148-yr Trojan cycle is 9 or 10 steps, 143 or
    159 yr, and 143 < 2 x 73.9 while its digital gain at P_c = 73.9 yr is 0.96. In the analog limit both agree at
    P_cyc = 2 P_c (G = 0.941 with N = 2). The period is measured after the filter: a cutoff that erased the libration
    left a slow wave whose gain is ~1 (588 at P_c = 318 yr: 1035 yr, G = 0.99, the libration's own G = 0.04). The
    physical cutoff (`choose_cutoff`) closes that circle: where P_c = P_floor / filter_min_libration_cutoffs every
    libration of the angle keeps G >= 0.94; a longer synodic bound is the case the dry-run warns of."""
    if period_cycle_yr is None or not np.isfinite(period_cycle_yr) or period_cycle_yr <= 0:
        return False, NAN
    step = step_yr if step_yr is not None and np.isfinite(step_yr) and step_yr > 0 else None  # unknown step: analog
    gain = float(butterworth_gain(period_cycle_yr, cutoff_yr, settings.filter_order, step))
    return bool(not gain >= settings.mmr_l8_gain_full), gain


def coorbital_planet(spec: ResonanceSpec) -> Optional[str]:
    """The planet a prograde angle is co-orbital with: the only planet with k_j != 0, and k_j = -k (the lambda-part
    of sigma is k (lambda - lambda_j)); None otherwise. A three-body angle such as 3J+8S-3 has k_J = -k too, but
    its body is far from Jupiter (n = n_J + 8 n_S / 3), and Jupiter's synodic period bounds its cutoff."""
    nonzero = [(name, k) for name, k in zip(spec.planets, spec.planet_coefficients) if k != 0]
    if spec.retrograde or len(nonzero) != 1 or nonzero[0][1] != -spec.body_coefficient:
        return None
    return nonzero[0][0]


def law_planet(spec: Optional[ResonanceSpec]) -> Optional[str]:
    """The planet of a 1:1 angle judged by the co-orbital law: a prograde co-orbital (`coorbital_planet`) or a
    retrograde 1/-1 (`1J-1r<k>`, p = q = 1: the same quadrature, coorbital.py); None for any other angle. The one
    predicate of "a 1:1 angle" (the force law, the strength check, the caps; W20 round D, review C #8)."""
    if spec is None:
        return None
    if spec.retrograde:
        return spec.planets[0] if spec.body_coefficient == 1 and spec.planet_coefficients == (-1,) else None
    return coorbital_planet(spec)


def coorbital_inputs(series: MMRSeries, context: Context, planet: str) -> Optional[Tuple[float, float, Optional[dict]]]:
    """(mu, n_P [rad/yr], the planet's saved elements {'e', 'inc', 'Omega', 'varpi'} or None) of a 1:1 angle's planet;
    None without a positive mass and mean motion (the 1:1 law and the quasi-satellite well need both)."""
    mu = context.planet_mass.get(planet, NAN)
    n_planet = context.planet_mean_motion_rad_yr.get(planet, NAN)
    if not (np.isfinite(mu) and mu > 0 and np.isfinite(n_planet) and n_planet > 0):
        return None
    saved = series.planets.get(planet)
    elements = None if saved is None else {'e': saved.e, 'inc': saved.inc, 'Omega': saved.Omega, 'varpi': saved.varpi}
    return mu, n_planet, elements


def _synodic_bounds(spec: ResonanceSpec, synodic_periods_yr: Mapping[str, float]) -> dict:
    """The synodic periods that bound the cutoff: all but the co-orbital planet's (`coorbital_planet`)."""
    planet = coorbital_planet(spec)
    return {name: period for name, period in synodic_periods_yr.items() if name != planet}


def _nan_like(x: np.ndarray) -> np.ndarray:
    return np.full(np.shape(x), NAN)


def angle_cutoff(
    spec: ResonanceSpec, context: Context, step_yr: float, settings: AnalysisSettings, period_floor: Optional[Tuple[float, str]] = None
) -> Tuple[Cutoff, float, str]:
    """(the cutoff of this angle at this step, its period floor [yr], the floor's setting): `choose_cutoff` with the
    synodic periods that bound it (`_synodic_bounds`) and `libration_period_floor`, or the given `period_floor`
    (floor [yr], setting): the quasi-satellite floor of a 1:1 angle (rule G1q, `analyse`)."""
    floor, floor_setting = period_floor if period_floor is not None else libration_period_floor(spec, context, settings)
    return choose_cutoff(_synodic_bounds(spec, context.synodic_periods_yr), step_yr, settings, floor), floor, floor_setting


def preprocess(
    series: MMRSeries,
    context: Context,
    sampling: Sampling,
    settings: AnalysisSettings,
    period_floor: Optional[Tuple[float, str]] = None,
    elements_block_yr: Optional[float] = None,
) -> Preprocessed:
    """Ripple, cutoff, sos filter of the unwrapped angle (detrended) and of a, e, inc at x0.5/x1/x2 of P_c
    (angle) and x1 (elements), derivatives, edge-trimmed window [t0 + trim P_c, t1 - trim P_c]. `period_floor`
    replaces the angle's own floor (`angle_cutoff`), `elements_block_yr` the blocks of the 1:1 law (default P_c);
    both are set by the quasi-satellite pass (G1q).

    The ripple is subtracted only when every planet of the angle has a saved mean longitude, the angle is not
    co-orbital and `mmr_ripple_remove` is on; `ripple_not_removed` is raised when a known inequality (r_GI > 0)
    stays in the angle. An angle filter at x0.5 that would be shorter than two steps (a coarse step) gives a nan
    series. The caller checks P_c > 2 dt first (`filter_realizable`).
    """
    t = np.asarray(series.t_yr, dtype=float)
    spec = series.resonance
    step = float(sampling.step_yr)
    order, rtol = settings.filter_order, settings.filter_grid_rtol
    cutoff, floor, floor_setting = angle_cutoff(spec, context, step, settings, period_floor)
    coorbital = coorbital_planet(spec) is not None
    pc = cutoff.cutoff_yr
    angle = np.asarray(sampling.angle_unwrapped, dtype=float)

    names = list(_planet_coefficients(spec))
    longitudes = {
        name: series.planets[name].mean_longitude
        for name in names
        if name in series.planets and series.planets[name].mean_longitude is not None
    }
    clean, ripple, removed, half_amplitude, corr = angle, None, False, NAN, NAN
    # the saved angle must be the combination of the saved longitudes (nan = not checkable: the identity is trusted)
    reconstruction = getattr(sampling, 'angle_reconstruction_rad', NAN)
    consistent = not (np.isfinite(reconstruction) and reconstruction > np.radians(settings.mmr_ripple_reconstruction_max_deg))
    # A co-orbital body follows its planet's slow longitude (the libration, ~150 yr for Jupiter, is far faster than the
    # 900-yr great inequality), so sigma = lambda_j - lambda holds none of the planet's ripple: corr(slow sigma, ripple)
    # = 0.01 in 588/617/624, and sigma - ripple doubles the slow wander of the centre (rms 0.51 -> 1.03 deg in 588)
    # and drops the force-law R^2 of 588 from 0.996 to 0.954 at 1 yr (inv-15). No subtraction, nothing unremoved.
    if settings.mmr_ripple_remove and names and len(longitudes) == len(names) and consistent and not coorbital:
        ripple = planet_ripple(t, spec, longitudes, context.planet_mean_motion_rad_yr, pc, order, rtol)
        clean, half_amplitude, corr = remove_planet_ripple(t, angle, ripple, tuple(settings.mmr_ripple_band_yr), order, rtol)
        removed = True
    unremoved = 0.0 if (removed or coorbital) else predicted_ripple(spec, context.great_inequality, pc, order, step)
    flags = [str(Flag.RIPPLE_NOT_REMOVED)] if unremoved > 0 else []

    angle_f = lowpass(t, clean, pc, order, grid_rtol=rtol)
    try:
        angle_f_half = lowpass(t, clean, pc / 2, order, grid_rtol=rtol)
    except ValueError:
        angle_f_half = _nan_like(clean)
    a_f = lowpass(t, series.a, pc, order, grid_rtol=rtol)
    return Preprocessed(
        cutoff=cutoff,
        t_yr=t,
        angle_unwrapped=angle,
        angle_clean=clean,
        angle_f=angle_f,
        rate_f=time_derivative(t, angle_f),
        angle_f_half=angle_f_half,
        angle_f_double=lowpass(t, clean, 2 * pc, order, grid_rtol=rtol),
        a_f=a_f,
        adot_f=time_derivative(t, a_f),
        e_f=lowpass(t, series.e, pc, order, grid_rtol=rtol),
        inc_f=lowpass(t, series.inc, pc, order, grid_rtol=rtol),
        window=edge_window(t, settings.filter_edge_trim_cutoffs * pc),
        ripple_rad=ripple,
        ripple_removed=removed,
        ripple_half_amp_rad=half_amplitude,
        ripple_corr=corr,
        unremoved_ripple_rad=float(unremoved),
        period_floor_yr=floor,
        period_floor_basis=floor_setting,
        elements_block_yr=float(elements_block_yr) if elements_block_yr is not None else cutoff.cutoff_yr,
        flags=flags,
    )


def filtered_angle(
    prep: Preprocessed,
    context: Context,
    step_yr: float,
    coorbital: bool = False,
    regime_cuts: bool = True,
    floor_well: Optional[geometry_module.FloorWell] = None,
) -> FilteredAngle:
    """The input of the shared geometry (slow variable a_f, c = context.identity_coefficient) with the MMR geometry
    rules on: the edge rule on the raw angle averaged over the body's orbital period, closure by phase return, the
    centre travel of an episode read on its own cycles, runs cut at regime changes (`regime_cuts`,
    LIBRATION_REGIME_CUTS), and for a 1:1 angle (`coorbital`) coverage by the union of its regimes
    (`geometry.FilteredAngle`); `floor_well`: the well inside which the floor of `prep` holds (G1q)."""
    return FilteredAngle(
        t_yr=prep.t_yr,
        angle_raw=prep.angle_unwrapped,
        angle_f=prep.angle_f,
        rate_f=prep.rate_f,
        angle_f_half=prep.angle_f_half,
        angle_f_double=prep.angle_f_double,
        slow_f=prep.a_f,
        identity_coefficient=context.identity_coefficient,
        cutoff_yr=prep.cutoff.cutoff_yr,
        step_yr=float(step_yr),
        window=prep.window,
        period_floor_yr=prep.period_floor_yr,
        period_floor_setting=prep.period_floor_basis,
        orbit_period_yr=float(getattr(context, 'orbital_period_yr', NAN)),
        edge_arc_rule=True,
        return_closure=True,
        split_regimes=bool(regime_cuts),
        coorbital=bool(coorbital),
        cycle_travel=True,
        floor_well=floor_well,
    )


def slow_components(series: MMRSeries, prep: Preprocessed, period_cycle_yr: float, settings: AnalysisSettings) -> SlowComponents:
    """Slow parts with W_c = libration_centre_cycles x P_cyc (3 P_c without cycles), trimmed by W_c at each edge.

    varpi is unwrapped first, with the winding of its slow vector (`sampling.body_varpi_slow_unwrap`: a short-period
    loop of e exp(i varpi) around the origin adds no turn); for a retrograde angle the slow varpi is varpi* = omega - Omega.
    """
    t = np.asarray(prep.t_yr, dtype=float)
    order = settings.filter_order
    base = (
        period_cycle_yr if period_cycle_yr is not None and np.isfinite(period_cycle_yr) and period_cycle_yr > 0 else prep.cutoff.cutoff_yr
    )
    window_yr = float(settings.libration_centre_cycles * base)
    varpi, _ = sampling_module.body_varpi_slow_unwrap(series, settings)
    rtol = settings.filter_grid_rtol
    a_slow = lowpass(t, series.a, window_yr, order, grid_rtol=rtol)
    return SlowComponents(
        window_yr=window_yr,
        t_yr=t,
        angle_slow=lowpass(t, prep.angle_clean, window_yr, order, grid_rtol=rtol),
        a_slow=a_slow,
        adot_slow=time_derivative(t, a_slow),
        varpi_slow=lowpass(t, varpi, window_yr, order, grid_rtol=rtol),
        e_slow=lowpass(t, series.e, window_yr, order, grid_rtol=rtol),
        window=edge_window(t, window_yr),
    )


def preprocessed_columns(prep: Preprocessed) -> dict:
    """CSV columns `flt_*` of `Preprocessed` (what `Preprocessed.to_columns` returns)."""
    cutoff = prep.cutoff
    deg = 180.0 / np.pi
    return {
        'flt_cutoff_yr': cutoff.cutoff_yr,
        'flt_cutoff_binding': cutoff.binding,
        'flt_cutoff_synodic_bound_yr': column_number(cutoff.synodic_bound_yr),
        'flt_cutoff_floor_bound_yr': column_number(cutoff.floor_bound_yr),
        'flt_samples_per_cutoff': column_number(cutoff.samples_per_cutoff),
        'flt_window_t0_yr': column_number(prep.window[0]),
        'flt_window_t1_yr': column_number(prep.window[1]),
        'flt_ripple_removed': prep.ripple_removed,
        'flt_ripple_removed_deg': column_number(prep.ripple_half_amp_rad, deg),
        'flt_ripple_corr': column_number(prep.ripple_corr),
        'flt_ripple_unremoved_deg': column_number(prep.unremoved_ripple_rad, deg),
        'flt_gain_at_pcyc': column_number(prep.gain_at_period_cycle),
        'flt_filter_too_slow': prep.filter_too_slow,
        'flt_period_floor_yr': column_number(prep.period_floor_yr),
        'flt_period_floor_basis': prep.period_floor_basis,
    }


# --------------------------------------------------------------------------------------------- the x2 guard (DELTA 13a)

# the quantities the guard compares, and the checks / margins that read each of them ('shape': the main episode's
# amplitude and centre shift, read by L2, L5, L6 and one-pendulum; also emptied under filter_too_slow)
GUARD_QUANTITIES = ('turning_points', 'episodes', 'stall', 'energy', 'shape')
SENSITIVE_CHECKS = {
    'turning_points': ('check_mmr_one_pendulum', 'check_mmr_onset', 'check_mmr_exit'),
    'episodes': ('check_mmr_one_pendulum', 'check_mmr_onset', 'check_mmr_exit'),
    'stall': (),
    'energy': ('check_mmr_potential', 'check_mmr_separatrix_zone'),
    'shape': ('check_mmr_one_pendulum',),
}
SENSITIVE_MARGINS = {
    'turning_points': ('L5', 'L6', 'L7', 'O1'),
    'episodes': ('L5', 'L6', 'L7', 'O1'),
    'stall': (),
    'energy': ('R0', 'Z', 'L2'),
    'shape': ('L2', 'L5', 'L6'),
}


class GuardOutcome(StrEnum):
    SAME = 'same'  # within the tolerance of its setting
    CHANGED = 'changed'  # beyond it, or measured at one cutoff only
    NOT_CHECKED = 'not_checked'  # nothing comparable (no interval, nothing measured, or 2 P_c too slow for the shape)


@dataclass(frozen=True)
class DecisiveQuantities:
    """What the decision reads from the filtered angle at one cutoff (DELTA 13a), on `interval` (the part of the
    record both cutoffs see: the x2 filter trims twice as much at each edge)."""

    cutoff_yr: float
    n_tp: int  # turning points in the interval
    n_episodes: int  # accepted episodes reaching into the interval
    stall: bool  # pattern B: the turns stall at the top of the potential
    stall_fraction: float
    h_med: float  # median energy of the cycles over the separatrix energy (nan without cycles)
    interval: Tuple[float, float] = (NAN, NAN)
    period_yr: float = NAN  # P_cyc of this measurement (on the decisive episode: its own median period)
    episode_bounds: Tuple[Tuple[float, float], ...] = ()  # accepted episodes clipped to the interval
    half_amp_max_rad: float = NAN  # the episode measured (see `on_decisive_episode`): largest half-amplitude (L2)
    centre_shift_over_amp: float = NAN  # and |centre shift| / mean amplitude (L6)
    # W21 Q9: True when n_tp, h_med, period and shape are those of the episode the status reads (at 2 P_c: its
    # counterpart by time overlap; `decisive_bounds` nan = not reproduced); False: the record's on the interval, and
    # the shape that of the episode with the most TPs
    on_decisive_episode: bool = False
    decisive_bounds: Tuple[float, float] = (NAN, NAN)  # that episode (or counterpart) clipped to the interval

    def conclusions(self, settings: AnalysisSettings) -> Dict[str, Optional[bool]]:
        """The yes/no facts the tree acts on: a full cycle exists (>= libration_min_turning_points TPs), an episode
        exists, the turns stall, the cycles lie below the separatrix (None when not measured)."""
        return {
            'turning_points': self.n_tp >= settings.libration_min_turning_points,
            'episodes': self.n_episodes >= 1,
            'stall': self.stall,
            'energy': bool(self.h_med < 1) if np.isfinite(self.h_med) else None,
        }


@dataclass(frozen=True)
class FilterSensitivity:
    """The decisive quantities at P_c and 2 P_c and the outcome of each comparison (all of it in columns).

    `sensitive`: True when a quantity changed, False when the compared ones agree, None when nothing could be
    compared (no x2 series, or no interval both cutoffs see) - 'not checked' is not 'unchanged'."""

    at_cutoff: DecisiveQuantities
    at_double: Optional[DecisiveQuantities]
    changed: Tuple[str, ...]
    not_checked: Tuple[str, ...] = ()
    decisive_episode_id: Optional[int] = None  # the episode compared (W21 Q9); None = the record's quantities

    @property
    def checked(self) -> bool:
        return self.at_double is not None and len(self.not_checked) < len(GUARD_QUANTITIES)

    @property
    def sensitive(self) -> Optional[bool]:
        if self.changed:
            return True
        return False if self.checked else None

    @property
    def decisive_overlap(self) -> float:
        """The decisive episode's time overlap with its counterpart at 2 P_c over their union (0 = not reproduced,
        nan = no decisive episode compared)."""
        if self.decisive_episode_id is None or self.at_double is None:
            return NAN
        first, second = self.at_cutoff.decisive_bounds, self.at_double.decisive_bounds
        if not np.all(np.isfinite(second)):
            return 0.0
        union = max(first[1], second[1]) - min(first[0], second[0])
        return _overlap(first, second) / union if union > 0 else NAN

    def to_columns(self) -> dict:
        """`flt_x1_*`, `flt_x2_*`, the interval, the changed and the unchecked quantities, the decisive episode
        compared and its overlap with its counterpart."""
        interval = self.at_cutoff.interval
        row = {
            'flt_filter_sensitive': self.sensitive,
            'flt_sensitive_quantities': ';'.join(self.changed),
            'flt_unchecked_quantities': ';'.join(self.not_checked),
            'flt_guard_t0_yr': column_number(interval[0]),
            'flt_guard_t1_yr': column_number(interval[1]),
            'flt_decisive_episode_id': self.decisive_episode_id,
            'flt_decisive_overlap': column_number(self.decisive_overlap),
        }
        deg = 180.0 / np.pi
        for tag, values in (('x1', self.at_cutoff), ('x2', self.at_double)):
            present = values is not None
            row[f'flt_{tag}_cutoff_yr'] = column_number(values.cutoff_yr) if present else None
            row[f'flt_{tag}_n_tp'] = values.n_tp if present else None
            row[f'flt_{tag}_n_episodes'] = values.n_episodes if present else None
            row[f'flt_{tag}_episode_bounds_yr'] = (
                ';'.join(f'{start:.6g}-{end:.6g}' for start, end in values.episode_bounds) if present else None
            )
            row[f'flt_{tag}_period_yr'] = column_number(values.period_yr) if present else None
            row[f'flt_{tag}_stall'] = values.stall if present else None
            row[f'flt_{tag}_stall_fraction'] = column_number(values.stall_fraction) if present else None
            row[f'flt_{tag}_h_med'] = column_number(values.h_med) if present else None
            row[f'flt_{tag}_half_amp_max_deg'] = column_number(values.half_amp_max_rad, deg) if present else None
            row[f'flt_{tag}_centre_shift_over_amp'] = column_number(values.centre_shift_over_amp) if present else None
            row[f'flt_{tag}_decisive_bounds_yr'] = (
                '-'.join(f'{t:.6g}' for t in values.decisive_bounds) if present and np.all(np.isfinite(values.decisive_bounds)) else None
            )
        return row


def _inside(times: np.ndarray, interval: Tuple[float, float]) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    return (times >= interval[0]) & (times <= interval[1])


def decisive_quantities(
    geometry: Geometry,
    cycle_h: np.ndarray,
    cutoff_yr: float,
    interval: Optional[Tuple[float, float]] = None,
    cycle_times_yr: Optional[np.ndarray] = None,
    episode=None,
    span: Optional[Tuple[float, float]] = None,
) -> DecisiveQuantities:
    """The quantities of one measurement on `interval` (default: its own window): TPs and cycles (by their mid-times
    `cycle_times_yr`; all cycles when not given) inside it, accepted episodes clipped to it, the stall from
    geometry.pattern_b, the amplitude and centre shift of the episode with the most TPs.

    With `span` (the decisive episode's (t_start, t_end) at P_c, W21 Q9) the TPs, cycles, period and shape are those
    of `episode` (the decisive episode, or its counterpart at 2 P_c) on its own stretch of the interval; `episode` None
    = not reproduced: the TPs and cycles of `span`, no period, no shape, `decisive_bounds` nan. The episode count and
    bounds stay the record's (columns)."""
    interval = tuple(geometry.window) if interval is None else tuple(interval)
    pattern = geometry.pattern_b
    h = np.asarray(cycle_h, dtype=float)
    bounds, main, most = [], None, 0
    for run in geometry.episodes:
        start, end = max(run.t_start_yr, interval[0]), min(run.t_end_yr, interval[1])
        if not end > start:
            continue
        bounds.append((float(start), float(end)))
        n_inside = int(np.sum(_inside(run.tp_times_yr, interval)))
        if n_inside > most:
            main, most = run, n_inside
    region, period, tp_times = interval, float(geometry.period_cycle_yr), geometry.turning_points.times_yr
    decisive_bounds = (NAN, NAN)
    if span is not None:
        own = (episode.t_start_yr, episode.t_end_yr) if episode is not None else span
        region = (max(own[0], interval[0]), min(own[1], interval[1]))
        main, period = episode, float(episode.period_med_yr) if episode is not None else NAN
        tp_times = episode.tp_times_yr if episode is not None else tp_times
        decisive_bounds = (float(region[0]), float(region[1])) if episode is not None else decisive_bounds
    if cycle_times_yr is not None and len(h):
        h = h[_inside(cycle_times_yr, region)]
    h = h[np.isfinite(h)]
    return DecisiveQuantities(
        cutoff_yr=float(cutoff_yr),
        n_tp=int(np.sum(_inside(tp_times, region))),
        n_episodes=len(bounds),
        stall=bool(pattern is not None and pattern.detected),
        stall_fraction=float(pattern.stall_fraction) if pattern is not None else NAN,
        h_med=float(np.median(h)) if len(h) else NAN,
        interval=(float(interval[0]), float(interval[1])),
        period_yr=period,
        episode_bounds=tuple(bounds),
        half_amp_max_rad=float(main.half_amp_max_rad) if main is not None else NAN,
        centre_shift_over_amp=float(main.centre_shift_over_amp) if main is not None else NAN,
        on_decisive_episode=span is not None,
        decisive_bounds=decisive_bounds,
    )


def _overlap(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def counterpart(episode, candidates: Sequence) -> Optional[object]:
    """The run among `candidates` (the accepted episodes at the other cutoff) that overlaps `episode` in time the
    most (W21 Q9: the guard compares an episode with itself, not with the first run with as many TPs); None when
    none overlaps (the episode is not reproduced)."""
    own = (episode.t_start_yr, episode.t_end_yr)
    best, longest = None, 0.0
    for run in candidates:
        shared = _overlap(own, (run.t_start_yr, run.t_end_yr))
        if shared > longest:
            best, longest = run, shared
    return best


def _measured_on_one_side(first: float, second: float) -> Optional[GuardOutcome]:
    """CHANGED when exactly one value is measured (a vanished measurement is a change), NOT_CHECKED when neither is,
    None when both are (compare them)."""
    finite = (np.isfinite(first), np.isfinite(second))
    if all(finite):
        return None
    return GuardOutcome.CHANGED if any(finite) else GuardOutcome.NOT_CHECKED


def _compare_turning_points(first: DecisiveQuantities, second: DecisiveQuantities, settings: AnalysisSettings) -> GuardOutcome:
    """Changed when the counts differ by more than max(mmr_filter_guard_tp_abs, mmr_filter_guard_tp_rel x the larger),
    or when 'a full cycle exists' (>= libration_min_turning_points) flips."""
    tolerance = max(settings.mmr_filter_guard_tp_abs, settings.mmr_filter_guard_tp_rel * max(first.n_tp, second.n_tp))
    flips = (first.n_tp >= settings.libration_min_turning_points) != (second.n_tp >= settings.libration_min_turning_points)
    return GuardOutcome.CHANGED if flips or abs(first.n_tp - second.n_tp) > tolerance else GuardOutcome.SAME


def _compare_episodes(first: DecisiveQuantities, second: DecisiveQuantities, settings: AnalysisSettings) -> GuardOutcome:
    """Changed when an episode appears or vanishes, or a start or end moves by more than
    mmr_filter_guard_boundary_periods libration periods (the period at P_c; boundaries are not compared without one).
    On the decisive episode (W21 Q9) only that episode counts: not reproduced at 2 P_c, or its own start or end moved;
    an episode elsewhere that appears or vanishes is a column (`flt_x1/x2_n_episodes`), not a change."""
    if first.on_decisive_episode:
        if not np.all(np.isfinite(second.decisive_bounds)):
            return GuardOutcome.CHANGED
        pairs = [(first.decisive_bounds, second.decisive_bounds)]
    elif first.n_episodes != second.n_episodes or len(first.episode_bounds) != len(second.episode_bounds):
        return GuardOutcome.CHANGED
    else:
        pairs = list(zip(first.episode_bounds, second.episode_bounds))
    period = first.period_yr if np.isfinite(first.period_yr) else second.period_yr
    if not np.isfinite(period):
        return GuardOutcome.SAME
    tolerance = settings.mmr_filter_guard_boundary_periods * period
    for (start1, end1), (start2, end2) in pairs:
        if abs(start1 - start2) > tolerance or abs(end1 - end2) > tolerance:
            return GuardOutcome.CHANGED
    return GuardOutcome.SAME


def _shape_comparable(
    first: DecisiveQuantities, second: DecisiveQuantities, settings: AnalysisSettings, step_yr: Optional[float] = None
) -> bool:
    """Amplitudes and energies are compared only when 2 P_c itself lets the libration through: the filter_too_slow rule
    (`filter_guard`) at the doubled cutoff and the output step `step_yr` (None = the analog limit)."""
    return not filter_guard(first.period_yr, second.cutoff_yr, settings, step_yr)[0]


def _compare_energy(
    first: DecisiveQuantities, second: DecisiveQuantities, settings: AnalysisSettings, step_yr: Optional[float] = None
) -> GuardOutcome:
    """Changed when the median cycle energy crosses h = 1 or moves by more than mmr_filter_guard_h_tol x max(1, the
    smaller |h|): absolute near the separatrix, relative far above it (W20 CS10: a far circulation's h ~ 1e4-1e7
    moves by < 1 % between the cutoffs, 30963 14M-25: 6879852 against 6879123, and is not filter-sensitive)."""
    if not _shape_comparable(first, second, settings, step_yr):
        return GuardOutcome.NOT_CHECKED
    one_side = _measured_on_one_side(first.h_med, second.h_med)
    if one_side is not None:
        return one_side
    crosses = (first.h_med < 1) != (second.h_med < 1)
    tolerance = settings.mmr_filter_guard_h_tol * max(1.0, min(abs(first.h_med), abs(second.h_med)))
    return GuardOutcome.CHANGED if crosses or abs(first.h_med - second.h_med) > tolerance else GuardOutcome.SAME


def _compare_shape(
    first: DecisiveQuantities, second: DecisiveQuantities, settings: AnalysisSettings, step_yr: Optional[float] = None
) -> GuardOutcome:
    """Changed when the measured episode's largest half-amplitude (the decisive one and its counterpart, else the one
    with the most TPs) moves by more than mmr_filter_guard_amp_rel of itself,
    or the full/not-full outcome of L2 (amplitude <= mmr_l2_amp_full_deg) or L6 (shift <= mmr_l6_shift_full) flips."""
    if not _shape_comparable(first, second, settings, step_yr):
        return GuardOutcome.NOT_CHECKED
    one_side = _measured_on_one_side(first.half_amp_max_rad, second.half_amp_max_rad)
    if one_side is not None:
        return one_side
    amp1, amp2 = first.half_amp_max_rad, second.half_amp_max_rad
    full_l2 = math.radians(settings.mmr_l2_amp_full_deg)
    if (amp1 <= full_l2) != (amp2 <= full_l2) or abs(amp2 - amp1) > settings.mmr_filter_guard_amp_rel * amp1:
        return GuardOutcome.CHANGED
    shift1, shift2 = first.centre_shift_over_amp, second.centre_shift_over_amp
    if np.isfinite(shift1) != np.isfinite(shift2):
        return GuardOutcome.CHANGED
    if np.isfinite(shift1) and (shift1 <= settings.mmr_l6_shift_full) != (shift2 <= settings.mmr_l6_shift_full):
        return GuardOutcome.CHANGED
    return GuardOutcome.SAME


def compare_quantities(
    at_cutoff: DecisiveQuantities,
    at_double: Optional[DecisiveQuantities],
    settings: AnalysisSettings,
    step_yr: Optional[float] = None,
) -> FilterSensitivity:
    """Each decisive quantity at P_c against 2 P_c with the tolerance of its setting (`GuardOutcome`); without the
    x2 measurement nothing is checked. `step_yr`: the output step of the filter gain (None = the analog limit)."""
    if at_double is None:
        return FilterSensitivity(at_cutoff, None, (), GUARD_QUANTITIES)
    outcomes = {
        'turning_points': _compare_turning_points(at_cutoff, at_double, settings),
        'episodes': _compare_episodes(at_cutoff, at_double, settings),
        'stall': GuardOutcome.SAME if at_cutoff.stall == at_double.stall else GuardOutcome.CHANGED,
        'energy': _compare_energy(at_cutoff, at_double, settings, step_yr),
        'shape': _compare_shape(at_cutoff, at_double, settings, step_yr),
    }
    changed = tuple(name for name in GUARD_QUANTITIES if outcomes[name] == GuardOutcome.CHANGED)
    unchecked = tuple(name for name in GUARD_QUANTITIES if outcomes[name] == GuardOutcome.NOT_CHECKED)
    return FilterSensitivity(at_cutoff, at_double, changed, unchecked)


def decisive_run(geometry: Geometry, episode_id: Optional[int], interval: Tuple[float, float]):
    """The accepted episode `episode_id` (the one the status reads) when it reaches into `interval`, else None (no
    episode, or one before the last jump of a that the x2 window does not see: the record's quantities are compared)."""
    if episode_id is None:
        return None
    for run in geometry.episodes:
        if run.episode_id == episode_id and _overlap((run.t_start_yr, run.t_end_yr), interval) > 0:
            return run
    return None


def doubled_angle(angle: FilteredAngle, prep: Preprocessed, regime_start_yr: float, settings: AnalysisSettings) -> Optional[FilteredAngle]:
    """The angle `angle` (at P_c) at the cutoff x2 for the geometry: its own edge trim, the window of the last regime;
    built from `angle` itself, so its MMR rules, period floor and well model carry over and the x2 geometry judges the
    runs as the x1 one does. None when the x2 series is missing or its window is empty."""
    cutoff = 2 * prep.cutoff.cutoff_yr
    t = np.asarray(prep.t_yr, float)
    angle_f = np.asarray(prep.angle_f_double, float)
    if not np.all(np.isfinite(angle_f)):
        return None
    window = edge_window(t, settings.filter_edge_trim_cutoffs * cutoff)
    if not np.isfinite(window[0]) or window[1] <= window[0]:
        return None
    window = (max(window[0], regime_start_yr), window[1])
    if window[1] <= window[0]:
        return None
    return replace(
        angle,
        angle_f=angle_f,
        rate_f=time_derivative(t, angle_f),
        angle_f_half=prep.angle_f,
        angle_f_double=np.full_like(angle_f, NAN),
        cutoff_yr=cutoff,
        window=window,
    )


def _cycle_mid_times(cycles: Sequence) -> np.ndarray:
    return np.array([(c.t_start_yr + c.t_end_yr) / 2 for c in cycles], dtype=float)


def filter_sensitivity(
    prep: Preprocessed,
    angle: FilteredAngle,
    geometry: Geometry,
    energy,
    prominence_rad: float,
    settings: AnalysisSettings,
    decisive_id: Optional[int] = None,
) -> FilterSensitivity:
    """DELTA 13a: measure turning points, episodes, stalls, the energy per cycle and the shape again at 2 P_c and
    compare the numbers with those at P_c (`geometry`, `energy` of the analysis) on the interval both see (the x1
    window of the last regime intersected with the x2 window). With `decisive_id` (the episode the status reads,
    `decide.status_basis`) inside that interval the TPs, cycles, boundaries, period and shape are those of that episode
    and of its counterpart at 2 P_c, the accepted episode that overlaps it most in time (`counterpart`; none = not
    reproduced): a ripple elsewhere in the record no longer stands for it (W21 Q9, 117306 11M-20: the 161.9 deg cycle
    was compared with a 0.8 deg ripple episode, the period was a median over both). A forced wave that P_c lets through makes
    turning points and stalls of its own, which 2 P_c damps; a real libration (G(P_cyc) >= mmr_l8_gain_full by the
    other guard) keeps them. The stall at x2 uses the potential top and zone of `energy` with the x2 turns' energies. Without the
    x2 series or a common interval nothing is checked (`FilterSensitivity.sensitive` None). The x2 turning points are
    selected at the x1 thresholds: `prominence_rad` outside the stretches, each stretch's own inside
    (`geometry.turning_points.local`); the record's alone loses every swing of a quiet libration before an exit
    (W16 independent test: 31 TPs at x1, none at x2 with the record's 35 deg). `angle`: the x1 angle of `geometry`;
    the x2 one is built from it (`doubled_angle`), so the runs at x2 are judged in the same frames (its well model) and
    by the same MMR rules (edges, returns, cuts)."""
    from resonances.resonance.analysis.mmr import zone as zone_module

    cycles = list(energy.cycles) if energy is not None else []
    h_cutoff, t_cutoff = np.array([c.h for c in cycles], dtype=float), _cycle_mid_times(cycles)
    step_yr = angle.step_yr
    angle = doubled_angle(angle, prep, geometry.regime_start_yr, settings)
    interval = (NAN, NAN)
    if angle is not None:
        interval = (max(geometry.window[0], angle.window[0]), min(geometry.window[1], angle.window[1]))
    if angle is None or not interval[1] > interval[0]:
        at_cutoff = decisive_quantities(geometry, h_cutoff, prep.cutoff.cutoff_yr, cycle_times_yr=t_cutoff)
        return FilterSensitivity(at_cutoff, None, (), GUARD_QUANTITIES)
    # the x1 thresholds of every stretch of turning points: the comparison differs only by the smoothing (as L1's counts)
    doubled = geometry_module.measure_geometry(angle, prominence_rad, settings, thresholds=geometry.turning_points.local)
    doubled_cycles = (
        zone_module.energies_at_cutoff(prep, doubled, energy, 2 * prep.cutoff.cutoff_yr, settings) if energy is not None else []
    )
    h = np.array([c.h for c in doubled_cycles], dtype=float)
    mid = _cycle_mid_times(doubled_cycles)
    if energy is not None and energy.potential_e is not None:
        turns = []
        for turn in doubled.revolutions:
            inside = np.flatnonzero((mid >= turn.t_start_yr) & (mid <= turn.t_end_yr) & np.isfinite(h))
            turns.append(replace(turn, energy_h=float(h[inside[0]])) if len(inside) else turn)
        h_hi = energy.zone.h_hi if energy.zone is not None else NAN
        doubled.pattern_b = geometry_module.pattern_b(turns, energy.potential_e.top_rad, h_hi, energy.sigma_h, settings)
    episode = decisive_run(geometry, decisive_id, interval)
    span = (episode.t_start_yr, episode.t_end_yr) if episode is not None else None
    other = counterpart(episode, doubled.episodes) if episode is not None else None
    at_cutoff = decisive_quantities(geometry, h_cutoff, prep.cutoff.cutoff_yr, interval, t_cutoff, episode, span)
    at_double = decisive_quantities(doubled, h, 2 * prep.cutoff.cutoff_yr, interval, mid, other, span)
    result = compare_quantities(at_cutoff, at_double, settings, step_yr)
    return replace(result, decisive_episode_id=episode.episode_id if episode is not None else None)
