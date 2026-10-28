"""The filter of a secular or Lidov-Kozai angle: its cutoff, the filtered angle and the turning-point prominence.

What a slow angle carries that is not its libration (measured on the benchmark, see the W4 report):
- short-period terms of the osculating elements; the longest of them is the Jupiter-Saturn great inequality
  (937 yr, `const.GREAT_INEQUALITY`), so P_c >= F_GI x 937 yr;
- in a secular angle with two or more planetary modes, the forced wave of each mode j of the angle: the forced term
  e_j exp(i g_j t) of the body's eccentricity vector (or s_j of its inclination vector) turns relative to the free
  one at f - f_j, f the body's own g or s, and shakes varpi (Omega) with that period. 4305 g-3g6+2g5: g = 76.2"/yr,
  waves at g - g6 (27.0 kyr) and g - g5 (18.0 kyr); 8089 z2: 136.9 and 68.4 kyr; 1020 z1: 37.1 kyr — the spectral
  peaks of the angles at exactly these periods. They are forced, not librations (an isolated resonance needs its
  angle slower than the other angles, the condition of averaging), so P_c >= SECULAR_FILTER_FORCED_FACTOR x the
  longest of them. A single-mode angle (nu6 = g - g6) has none: the forced term at g6 there is the resonance itself
  (its period 2 pi/|g - g6| is Tina's libration period, 594 kyr).
- anything faster than the type's shortest libration P_min (SECULAR_/ZLK_LIBRATION_PERIOD_MIN_YR) is no libration:
  P_c >= P_min / FILTER_MIN_LIBRATION_CUTOFFS keeps every libration on the right side of the shared filter guard
  (gain >= 0.94 at P_min) and removes the faster forced terms nobody listed (80218 nu6: waves of 50-200 kyr over a
  316 kyr oscillation).

The prominence of a turning point covers what sigma_f still holds faster than the type's shortest libration
(`_set_prominence`): terms the cutoff rule does not know (Tina's 8 kyr wave) cannot make turns.

The filter acts on the unwrapped angle with its trend removed and put back (`analysis.filtering.lowpass`), so it
keeps the number of turns: it cannot turn a circulation into a libration (the trap of filtering a wrapped angle or
e cos sigma; Carruba et al. 2025, Icarus 425, 116346, p. 7).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np

from resonances.data.const import GREAT_INEQUALITY, PLANETARY_FREQUENCIES
from resonances.resonance.analysis import geometry
from resonances.resonance.analysis.filtering import edge_window, lowpass_or_none, time_derivative, window_mask
from resonances.resonance.analysis.geometry import FilteredAngle
from resonances.resonance.analysis.secular.models import ARCSEC_PER_TURN, AngleTerm, Cutoff, LongPeriodSeries
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import ResonanceType

NAN = float('nan')
FAST_PERCENTILE = 95.0  # the half-amplitude of the fast part: the 95th percentile of |sigma - sigma_f|


def body_rates(series: LongPeriodSeries) -> Tuple[float, float]:
    """The body's own frequencies g and s [arcsec/yr]: the mean rates of its free vectors (`proper_angle.free_rates`);
    (nan, nan) when the elements are missing or the decomposition fails."""
    from resonances.secular.proper_angle import free_rates

    if series.varpi is None or series.Omega is None:
        return NAN, NAN
    try:
        g, s = free_rates(series.t_yr, series.e, series.varpi, series.inc, series.Omega)
    except (ValueError, np.linalg.LinAlgError):
        return NAN, NAN
    return float(g), float(s)


def forced_waves(terms, g: float, s: float, frequencies=None) -> Dict[str, float]:
    """{mode label: period [yr]} of the forced waves 2 pi/|f - f_j| of the angle's planetary modes (f = g for a g-mode,
    s for an s-mode); empty for an angle with fewer than two planetary modes, or when f is not measured."""
    frequencies = PLANETARY_FREQUENCIES if frequencies is None else frequencies
    planetary = [term for term in terms if term.index is not None]
    if len({(term.mode, term.index) for term in planetary}) < 2:
        return {}
    waves = {}
    for term in planetary:
        label = f'{term.mode}{term.index}'
        own = g if term.mode == 'g' else s
        difference = own - frequencies[label]
        if np.isfinite(difference) and difference != 0:
            waves[label] = ARCSEC_PER_TURN / abs(difference)
    return waves


def _type_settings(series: LongPeriodSeries, settings: AnalysisSettings) -> Tuple[float, float, float, float, str]:
    """(F_GI, prominence floor [deg], shortest libration period [yr], ripple factor, the name of that period's setting)
    of the angle's type."""
    if series.resonance_type == ResonanceType.LIDOV_KOZAI:
        return (
            settings.zlk_filter_gi_factor,
            settings.zlk_tp_prominence_deg,
            settings.zlk_libration_period_min_yr,
            settings.zlk_tp_ripple_factor,
            'zlk_libration_period_min_yr',
        )
    return (
        settings.secular_filter_gi_factor,
        settings.secular_tp_prominence_deg,
        settings.secular_libration_period_min_yr,
        settings.secular_tp_ripple_factor,
        'secular_libration_period_min_yr',
    )


def choose_cutoff(series: LongPeriodSeries, settings: AnalysisSettings, rates: Optional[Tuple[float, float]] = None) -> Cutoff:
    """P_c = max(F_GI x P_GI, SECULAR_FILTER_FORCED_FACTOR x P_w, P_min / FILTER_MIN_LIBRATION_CUTOFFS) [yr] (module
    docstring). The output step does not enter (the MMR rule, `mmr.preprocess.choose_cutoff`): P_min / 2 is kyr to Myr,
    far above any step that resolves a secular angle. `rates` = (g, s) [arcsec/yr] replaces the rates measured on the
    series (the dry-run's static estimate; the series then needs no elements)."""
    gi_factor, _, period_min, _, _ = _type_settings(series, settings)
    gi_bound = gi_factor * GREAT_INEQUALITY['period_yr']
    g = s = NAN
    waves: Dict[str, float] = {}
    if series.resonance_type == ResonanceType.SECULAR and series.terms:
        g, s = rates if rates is not None else body_rates(series)
        waves = forced_waves(series.terms, g, s, series.planet_frequencies or None)
    forced_bound = settings.secular_filter_forced_factor * max(waves.values()) if waves else NAN
    libration_bound = period_min / settings.filter_min_libration_cutoffs
    bounds = {'great_inequality': gi_bound, 'forced_wave': forced_bound, 'shortest_libration': libration_bound}
    binding = max((name for name in bounds if np.isfinite(bounds[name])), key=lambda name: bounds[name])
    return Cutoff(
        cutoff_yr=float(bounds[binding]),
        binding=binding,
        gi_bound_yr=float(gi_bound),
        forced_bound_yr=float(forced_bound),
        libration_bound_yr=float(libration_bound),
        forced_waves_yr=waves,
        body_g_arcsec_yr=g,
        body_s_arcsec_yr=s,
    )


def period_floor(series: LongPeriodSeries, cutoff_yr: float, settings: AnalysisSettings) -> Tuple[float, str]:
    """The shortest period an episode may have to be a libration of this angle [yr], and its setting: the type's
    shortest libration P_min, or FILTER_MIN_LIBRATION_CUTOFFS x P_c when that is longer.

    Below 2 P_c the filter itself shapes the oscillation (the shared guard), and for a secular angle that is where its
    unlisted forced terms live: P_c = 2 P_w puts 2 P_c at four times the longest known forced wave. 2008 UZ220
    g-g1+s3-s8: 2.6-deg waves of 270 kyr at P_c = 143 kyr made two 3-TP 'episodes' near the top of a 10-Myr arc."""
    _, _, period_min, _, setting = _type_settings(series, settings)
    guard = settings.filter_min_libration_cutoffs * cutoff_yr
    return (float(guard), 'filter_min_libration_cutoffs') if guard > period_min else (float(period_min), setting)


def filtered_angle(series: LongPeriodSeries, cutoff: Cutoff, settings: AnalysisSettings) -> Optional[FilteredAngle]:
    """sigma_f at P_c (and at P_c x 0.5 and x 2 for the robustness of the turning-point count), its rate, e_f as the
    slow variable, the edge-trimmed window and the shortest libration (`period_floor`); sets the prominence of `cutoff`.
    None when the record cannot be filtered (non-uniform grid, or shorter than the cutoff needs)."""
    t = np.asarray(series.t_yr, dtype=float)
    x = np.asarray(series.angle, dtype=float)
    angle_f = lowpass_or_none(t, x, cutoff.cutoff_yr, settings)
    if angle_f is None:
        return None
    half = lowpass_or_none(t, x, cutoff.cutoff_yr / 2, settings)
    double = lowpass_or_none(t, x, 2 * cutoff.cutoff_yr, settings)
    e_f = lowpass_or_none(t, np.asarray(series.e, dtype=float), cutoff.cutoff_yr, settings)
    window = edge_window(t, settings.filter_edge_trim_cutoffs * cutoff.cutoff_yr)
    floor = period_floor(series, cutoff.cutoff_yr, settings)
    _set_prominence(series, cutoff, x, angle_f, window, floor[0], settings)
    return FilteredAngle(
        t_yr=t,
        angle_raw=x,
        angle_f=angle_f,
        rate_f=time_derivative(t, angle_f),
        angle_f_half=half if half is not None else np.full_like(x, NAN),
        angle_f_double=double if double is not None else np.full_like(x, NAN),
        slow_f=e_f if e_f is not None else np.full_like(x, NAN),
        identity_coefficient=NAN,  # no identity of an MMR: the arc/stall force bounds are not defined here
        cutoff_yr=cutoff.cutoff_yr,
        step_yr=series.step_yr,
        window=window,
        period_floor_yr=floor[0],
        period_floor_setting=floor[1],
    )


def centre_scale_angle(
    series: LongPeriodSeries, cutoff: Cutoff, angle: FilteredAngle, period_cycle_yr: float, settings: AnalysisSettings
) -> Optional[Tuple[FilteredAngle, float]]:
    """(the angle filtered at the scale of its centre, its prominence), or None when there is no such scale
    (`geometry.centre_scale_angle`; its shortest libration is `period_floor` at that cutoff, the prominence that of a
    centre turn, `geometry.centre_prominence`). A slow secular libration under a faster oscillation (518 z1: 3.3 Myr
    under 400 kyr waves that fragment the fine turning points) is the libration of this centre."""
    coarse = geometry.centre_scale_angle(angle, period_cycle_yr, settings)
    if coarse is None:
        return None
    floor, setting = period_floor(series, coarse.cutoff_yr, settings)
    coarse = replace(coarse, period_floor_yr=floor, period_floor_setting=setting)
    return coarse, float(geometry.centre_prominence(cutoff.prominence_rad, settings))


def _set_prominence(series: LongPeriodSeries, cutoff: Cutoff, x: np.ndarray, angle_f: np.ndarray, window, period_min, settings) -> None:
    """p_TP = max(floor, K r): r is the 95th percentile of |sigma_f - lowpass(sigma, P_min)| inside the window, what
    sigma_f still holds faster than the shortest libration P_min (`period_floor`), K the type's ripple factor (a wave of
    half-amplitude r turns by up to 2 r; the MMR rule, `mmr.analyse.fast_residual`).

    Terms the cutoff rule does not know make such waves: Tina (1222 nu6, P_c = 10 kyr) keeps 2-4 deg of an 8 kyr
    term (about 2(g - s)), 363 Padua 4 deg of a 170 kyr one; their turns near an extremum of the slow libration,
    where the centre stands still for a few cycles, would otherwise pass for 3-TP 'episodes'."""
    _, floor_deg, _, factor, _ = _type_settings(series, settings)
    t = np.asarray(series.t_yr, dtype=float)
    inside = window_mask(t, window)
    cutoff.fast_amplitude_rad = float(np.percentile(np.abs(x - angle_f)[inside], FAST_PERCENTILE)) if inside.any() else NAN
    slow = lowpass_or_none(t, x, period_min, settings) if period_min > cutoff.cutoff_yr else None
    ripple = float(np.percentile(np.abs(angle_f - slow)[inside], FAST_PERCENTILE)) if slow is not None and inside.any() else 0.0
    floor = np.radians(floor_deg)
    cutoff.ripple_rad = ripple
    cutoff.prominence_rad = float(max(floor, factor * ripple))
    cutoff.prominence_binding = 'floor' if floor >= factor * ripple else 'ripple'


def terms_from_formula(terms) -> Tuple[AngleTerm, ...]:
    """`AngleTerm`s of a `SecularResonanceFormula`'s terms (objects with mode, index, coefficient)."""
    return tuple(AngleTerm(str(term.mode), term.index, int(term.coefficient)) for term in terms)
