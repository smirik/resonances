"""The owner route 'identity' (W22 R-OWNER-ID, author 2026-10-24; DECISIONS 2026-10-17 on 433 1R-5V+7).

A three-body angle sigma whose period needs >= mmr_owner_identity_ratio_min x mu_i mu_j e^q (str_ratio, before the
small-divisor envelope) and whose owner search accepted nothing may still follow a two-body resonance phi of the same
body: sigma = (k_sigma / k_phi) phi + Psi / k_phi with Psi free of the body's lambda (`owner.psi_combination`). The
route accepts phi as the owner (route 'identity', origin owner_libration) when on sigma's decisive episode

  1. sigma_f - (k_sigma / k_phi) phi_f is slow: the fit sigma_f = c phi_f + a + b t gives |c / (k_sigma / k_phi) - 1|
     <= mmr_owner_identity_coef_tol and the rms of sigma_f - (k_sigma / k_phi) phi_f about its line is at most
     mmr_owner_identity_resid_max of sigma's median half-amplitude (both low-passed at sigma's P_c; phi unwrapped by
     its model rate, `owner.unwrap_ok`);
  2. phi librates there: its own analysis (`analyse_mmr` on phi's angle, no scans) has librating episodes covering at
     least mmr_owner_cover_min of the window (a libration over its record covers it all);
  3. phi's strength by direct averaging passes (`strength.DIRECT_METHODS`, check pass).

The identity is exact, so step 1 passes for almost any candidate whose angle unwraps (433: the circulating 25E-44
fits with 4e-5 and 5e-4): steps 2 and 3, with sigma's required strength, decide. Nothing is measured on a, so a folded
short period does not bias the route. Cost: only candidates whose phi_f spans less than mmr_owner_identity_span_max_deg
on the window, at most mmr_owner_identity_top_n of them (smallest span first), get the full analysis of step 2. The
second capture is not tested (cap second_capture_unresolved stays). Candidates: the two-body angles of the owner
enumeration on the episode's period and the a range of the window, and those of the owner search. Pure: arrays in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.geometry import Episode, Geometry
from resonances.resonance.analysis.ledger import Ledger, Op
from resonances.resonance.analysis.mmr import owner as owner_module
from resonances.resonance.analysis.mmr.context import mean_motion
from resonances.resonance.analysis.mmr.models import (
    Context,
    IdentityRoute,
    MMRSeries,
    OwnerCandidate,
    OwnerResult,
    Preprocessed,
    ResonanceSpec,
    Strength,
)
from resonances.resonance.analysis.mmr.strength import DIRECT_METHODS
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue, OriginDetail

NAN = float('nan')
ROUTE = owner_module.GAIN_IDENTITY
FIT_TERMS = 3  # sigma_f = c phi_f + a + b t
SAMPLES_PER_TERM = 3  # a least-squares fit wants a few samples per term
INTERVAL_LIBRATION = owner_module.INTERVAL_LIBRATION


@dataclass(frozen=True)
class IdentityFit:
    """Step 1 for one candidate: c / (k_sigma / k_phi) - 1, the residual over sigma's half-amplitude, phi_f's span."""

    spec: ResonanceSpec
    coef_dev: float
    resid: float
    span_rad: float
    unwrap_ok: bool


@dataclass(frozen=True)
class PhiOwnership:
    """Steps 2 and 3 for one candidate, from its own analysis."""

    angle_status: str
    intervals: List[Tuple[float, float, str]]
    cover: float
    strength_method: str
    strength_check: CheckValue
    strength_ratio: float


def route_applies(strength: Optional[Strength], owner: Optional[OwnerResult], run: Optional[Episode], settings: AnalysisSettings) -> bool:
    """A three-body angle whose period needs >= mmr_owner_identity_ratio_min x mu_i mu_j e^q, without an accepted owner,
    with a decisive episode of at least two turning points."""
    if strength is None or not strength.three_body or not np.isfinite(strength.ratio):
        return False
    if strength.ratio < settings.mmr_owner_identity_ratio_min:
        return False
    if owner is not None and owner.accepted is not None:
        return False
    return run is not None and len(run.tp_times_yr) >= 2


def longitudes_saved(series: MMRSeries, settings: AnalysisSettings) -> bool:
    """The route reads the body's lambda and at least one planet's: without them it does not run (the owner search
    reports owner_no_longitudes then)."""
    return series.mean_longitude is not None and bool(owner_module.available_planets(series, settings))


def episode_window(run: Episode) -> Tuple[float, float]:
    """sigma's decisive episode: its first and last turning point [yr]."""
    return float(run.tp_times_yr[0]), float(run.tp_times_yr[-1])


def candidates(
    series: MMRSeries, context: Context, geometry: Geometry, run: Episode, owner: Optional[OwnerResult], settings: AnalysisSettings
) -> List[ResonanceSpec]:
    """Two-body candidates (a body coefficient): the owner enumeration on the episode's period and the a range of its
    window, then those of the owner search, each once."""
    window = episode_window(run)
    t = np.asarray(series.t_yr, float)
    a = np.asarray(series.a, float)[filtering.window_mask(t, window)]
    if not len(a):
        return []
    planets = owner_module.available_planets(series, settings)
    n_range = (float(mean_motion(np.max(a))), float(mean_motion(np.min(a))))
    specs = owner_module.enumerate_owners(
        series.resonance, context, geometry.period_cycle_yr, planets, settings, period_yr=float(run.period_med_yr), body_n_range=n_range
    )
    if owner is not None:
        specs += [c.spec for c in owner.candidates]
    seen, kept = set(), []
    for spec in specs:
        if spec.key in seen or len(spec.planets) != 1 or spec.body_coefficient == 0:
            continue
        seen.add(spec.key)
        kept.append(spec)
    return kept


def identity_fit(
    t: np.ndarray, sigma_f: np.ndarray, phi_f: np.ndarray, ratio: float, half_amplitude_rad: float
) -> Tuple[float, float, float]:
    """(c / ratio - 1, rms of sigma_f - ratio phi_f about its line / half_amplitude, ptp of phi_f) on the given samples,
    c from the least-squares fit sigma_f = c phi_f + a + b t."""
    tw = t - t[0]
    design = np.column_stack([phi_f, np.ones_like(phi_f), tw])
    coefficients, *_ = np.linalg.lstsq(design, sigma_f, rcond=None)
    fixed = sigma_f - ratio * phi_f
    line = np.polyfit(tw, fixed, 1)
    rms = float(np.sqrt(np.mean((fixed - np.polyval(line, tw)) ** 2)))
    resid = rms / half_amplitude_rad if half_amplitude_rad > 0 else NAN
    return float(coefficients[0] / ratio - 1.0), resid, float(np.ptp(phi_f))


def identity_fits(
    series: MMRSeries, prep: Preprocessed, context: Context, run: Episode, specs: Sequence[ResonanceSpec], settings: AnalysisSettings
) -> List[IdentityFit]:
    """Step 1 for every candidate on sigma's decisive window (both angles low-passed at sigma's P_c)."""
    t = np.asarray(prep.t_yr, float)
    window = episode_window(run)
    mask = filtering.window_mask(t, window)
    if np.count_nonzero(mask) < SAMPLES_PER_TERM * FIT_TERMS:
        return []
    cutoff, order, rtol = prep.cutoff.cutoff_yr, settings.filter_order, settings.filter_grid_rtol
    sigma_f = filtering.lowpass(t, np.asarray(prep.angle_unwrapped, float), cutoff, order, grid_rtol=rtol)
    planet_n = {name: owner_module.planet_mean_motion(context, name) for name in owner_module.available_planets(series, settings)}
    k_sigma = series.resonance.body_coefficient
    fits = []
    for spec in specs:
        if any(name not in planet_n for name in spec.planets):
            continue
        phi, residual = owner_module.owner_angle_with_residual(spec, series, planet_n)
        phi_f = filtering.lowpass(t, phi, cutoff, order, grid_rtol=rtol)
        if not (np.all(np.isfinite(phi_f[mask])) and np.all(np.isfinite(sigma_f[mask]))):
            continue  # a gap in the saved longitudes: no fit
        ratio = k_sigma / spec.body_coefficient
        coef_dev, resid, span = identity_fit(t[mask], sigma_f[mask], phi_f[mask], ratio, float(run.half_amp_med_rad))
        fits.append(IdentityFit(spec, coef_dev, resid, span, owner_module.unwrap_ok(residual, settings)))
    return fits


def passes_identity(fit: IdentityFit, settings: AnalysisSettings) -> bool:
    """Step 1 and the cost pre-filter: the coefficient and the residual within their limits, phi unwrapped, phi_f
    spanning less than mmr_owner_identity_span_max_deg."""
    return bool(
        fit.unwrap_ok
        and abs(fit.coef_dev) <= settings.mmr_owner_identity_coef_tol
        and fit.resid <= settings.mmr_owner_identity_resid_max
        and fit.span_rad < np.radians(settings.mmr_owner_identity_span_max_deg)
    )


def shortlist(fits: Sequence[IdentityFit], settings: AnalysisSettings) -> List[IdentityFit]:
    """The candidates passing step 1, smallest phi_f span first, at most mmr_owner_identity_top_n."""
    passing = sorted((f for f in fits if passes_identity(f, settings)), key=lambda f: f.span_rad)
    return passing[: settings.mmr_owner_identity_top_n]


def phi_series(series: MMRSeries, spec: ResonanceSpec) -> MMRSeries:
    """The candidate's own record: its angle from the saved longitudes (`owner.wrapped_angle`)."""
    return replace(series, resonance=spec, angle_wrapped=owner_module.wrapped_angle(spec, series), angle_unwrapped_numpy=None)


def phi_ownership(phi_analysis, window: Tuple[float, float]) -> PhiOwnership:
    """Steps 2 and 3 from phi's own analysis: its librating intervals (the analysis window for a libration over the
    record, else its accepted episodes from first to last turning point), their cover of sigma's window, its strength."""
    verdict, geometry, strength = phi_analysis.verdict, phi_analysis.geometry, phi_analysis.strength
    angle_status = str(verdict.angle_status)
    intervals: List[Tuple[float, float, str]] = []
    if angle_status == AngleStatus.LIBRATION and geometry is not None:
        intervals = [(float(geometry.window[0]), float(geometry.window[1]), INTERVAL_LIBRATION)]
    elif geometry is not None:
        intervals = [episode_window(e) + (INTERVAL_LIBRATION,) for e in geometry.episodes if len(e.tp_times_yr) >= 2]
    cover = owner_module.covered_fraction(intervals, window) if intervals else 0.0
    return PhiOwnership(
        angle_status,
        intervals,
        float(cover),
        strength.method if strength is not None else '',
        strength.check if strength is not None else CheckValue.NOT_MEASURED,
        float(strength.ratio) if strength is not None else NAN,
    )


def owns(phi: PhiOwnership, settings: AnalysisSettings) -> bool:
    """Steps 2 and 3: phi's libration covers at least mmr_owner_cover_min of sigma's window and its directly averaged
    strength passes."""
    return bool(
        phi.cover >= settings.mmr_owner_cover_min and phi.strength_method in DIRECT_METHODS and phi.strength_check == CheckValue.PASS
    )


def search(
    series: MMRSeries,
    prep: Preprocessed,
    context: Context,
    geometry: Geometry,
    run: Episode,
    owner: Optional[OwnerResult],
    settings: AnalysisSettings,
    analyse: Callable[[MMRSeries], object],
) -> Optional[Tuple[IdentityFit, PhiOwnership, IdentityRoute]]:
    """The route on one record: the first shortlisted candidate that phi's own analysis (`analyse`, called with phi's
    series) accepts, with its numbers; None when none does."""
    fits = identity_fits(series, prep, context, run, candidates(series, context, geometry, run, owner, settings), settings)
    listed = shortlist(fits, settings)
    window = episode_window(run)
    n_pass = sum(passes_identity(f, settings) for f in fits)
    for count, fit in enumerate(listed, start=1):
        phi = phi_ownership(analyse(phi_series(series, fit.spec)), window)
        if owns(phi, settings):
            route = IdentityRoute(
                fit.spec.key, fit.coef_dev, fit.resid, fit.span_rad, phi.angle_status, phi.cover, phi.strength_ratio, n_pass, count
            )
            return fit, phi, route
    return None


def record(ledger: Ledger, strength: Strength, fit: IdentityFit, phi: PhiOwnership, settings: AnalysisSettings) -> None:
    """The route's decisive comparisons of an accepted owner (nothing is recorded for a route that accepts none)."""
    ledger.compare_setting(
        '2 identity: required / mu_i mu_j e^q', strength.ratio, Op.GE, settings, 'mmr_owner_identity_ratio_min', column='str_ratio'
    )
    ledger.compare_setting(
        '2 identity: |c / (k_sigma / k_phi) - 1|',
        abs(fit.coef_dev),
        Op.LE,
        settings,
        'mmr_owner_identity_coef_tol',
        column='own_id_coef_dev',
    )
    ledger.compare_setting(
        '2 identity: residual / half-amplitude', fit.resid, Op.LE, settings, 'mmr_owner_identity_resid_max', column='own_id_resid'
    )
    ledger.compare_setting(
        "2 identity: phi's libration covers sigma's episode", phi.cover, Op.GE, settings, 'mmr_owner_cover_min', column='own_id_phi_cover'
    )
    passed = 1.0 if phi.strength_check == CheckValue.PASS else 0.0
    ledger.compare("2 identity: phi's directly averaged strength passes", passed, Op.EQ, 1, '', 'own_id_phi_str_ratio')


def identity_owner(
    original: Optional[OwnerResult], series: MMRSeries, fit: IdentityFit, phi: PhiOwnership, route: IdentityRoute, run: Episode
) -> OwnerResult:
    """The owner result with phi accepted by the route 'identity' at rank 1 (a candidate of the search keeps its own
    numbers): check fail, origin detail owner_libration, phi's libration intervals and their cover of sigma's episode,
    no second-capture test."""
    sigma_spec, spec = series.resonance, fit.spec
    base = original if original is not None else owner_module._empty(CheckValue.FAIL, [])
    previous = [c for c in base.candidates if c.spec.key != spec.key]
    prior = next((c for c in base.candidates if c.spec.key == spec.key), None)
    reason = f'accepted:{ROUTE}'
    if prior is not None:
        accepted = replace(prior, rank=1, accepted=True, reason=reason, route=ROUTE, regime=INTERVAL_LIBRATION)
    else:
        accepted = OwnerCandidate(spec, NAN, NAN, (), NAN, NAN, NAN, NAN, NAN, 1, True, reason, regime=INTERVAL_LIBRATION, route=ROUTE)
    ranked = [accepted] + [replace(c, rank=rank, accepted=False) for rank, c in enumerate(previous, start=2)]
    t = np.asarray(series.t_yr, float)
    fields = dict(
        candidates=ranked,
        accepted=accepted,
        check=CheckValue.FAIL,
        detail=str(OriginDetail.OWNER_LIBRATION),
        intervals=list(phi.intervals),
        decisive_window=episode_window(run),
        cover=phi.cover,
        step_yr=float(np.median(np.diff(t))) if len(t) > 1 else NAN,
        psi_def=owner_module.psi_definition(sigma_spec, spec),
        identity_text=owner_module.identity_text(sigma_spec, spec),
        capture=None,
        sigma_force_bound=NAN,
        sigma_period_min_yr=NAN,
        identity=route,
        # an owner is accepted now: no 'probable' or 'unresolved' owner of the search stays
        flags=[x for x in base.flags if x not in (owner_module.FLAG_PROBABLE, owner_module.FLAG_UNRESOLVED)],
    )
    return replace(base, n_candidates=max(base.n_candidates, len(ranked)), **fields)
