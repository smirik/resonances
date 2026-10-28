"""`analyse_secular`: the steps of the analysis of a secular angle, and the shared core of the Lidov-Kozai one.

    0 input        an unphysical orbit (e > 1.3, a < 0, non-finite a or e) -> invalid/undefined (unphysical);
                   too few samples, NaN/inf in the angle -> uncertain/undefined (unusable_series)
    1 cutoff       preprocess.choose_cutoff: P_c and the turning-point prominence
    2 filter       preprocess.filtered_angle: sigma_f at P_c, x0.5, x2; the window [t0 + P_c, t1 - P_c]
    3 geometry     geometry.measure_geometry (turning points, runs, episodes, centre and passes, exit and entry)
    4 angle        decide.decide_angle (geometry only); without a libration at P_c, the same at the scale of the
                   centre W_c = LIBRATION_CENTRE_CYCLES x P_cyc (`preprocess.centre_scale_angle`), which replaces the
                   fine scale when it shows a libration (or libration episodes the fine scale lacks): a slow secular
                   libration carries faster waves (518 z1, 52773 z2), and its centre is what librates. This is the
                   former secular centre check with the scale of the centre measured instead of 10% of the record.
                   A `centre_guard` (ZLK) can refuse the centre scale: a period that is a beat of the forced vector
                   while the free circle stays off the origin is no libration of omega (`sec_centre_rejected`).
    5 physics      secular: response.analyse_response (a column); ZLK: the free-omega gate (analysis.zlk)
    6 status       decide.decide_status
    7 confidence   decide.margins_and_caps, confidence.assess
    8 action       an uncertain verdict: extend (with extend_to_years) or review, `extend.recommend`; no status changes
    9 columns      a modulated secular angle: the slow line of its centre (`slow.secular_centre_line`); ZLK: the
                   slow line, e-i coupling and attribution (`analysis.zlk`). No status reads them.

Pure: arrays in (`LongPeriodSeries`), `LongPeriodAnalysis` out.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from resonances.resonance.analysis import confidence as confidence_module, orbit_type as orbit_type_module
from resonances.resonance.analysis.confidence import ConfidenceCells
from resonances.resonance.analysis.geometry import measure_geometry
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.secular import decide, extend as extend_module, preprocess, slow as slow_module
from resonances.resonance.analysis.secular.models import GateResult, LongPeriodAnalysis, LongPeriodSeries
from resonances.resonance.analysis.secular.response import analyse_response
from resonances.resonance.analysis.settings import AnalysisSettings, settings_hash
from resonances.resonance.analysis.verdict import (
    Action,
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    Reason,
    ResonanceType,
    Status,
    Verdict,
    default_action,
    unmeasured_checks,
)
from resonances.resonance.classify.util import unphysical_arrays

MIN_SAMPLES = 3  # a derivative and a turning point need three points
SCALE_FINE = 'fine'  # the verdict rests on the angle filtered at P_c
SCALE_CENTRE = 'centre'  # ... on the angle filtered at the scale of its centre (`preprocess.centre_scale_angle`)

# gate(angle decision, geometry) -> GateResult: the free-omega gate of a Lidov-Kozai angle (analysis.zlk)
GateFunction = Callable[[decide.AngleDecision, object], Optional[GateResult]]
# centre_guard(centre geometry, fine geometry, ledger) -> '' or why a libration at the centre scale is not the angle's
# (ZLK: a beat of the forced vector, `zlk.diagnostics.forced_beat_centre`); its comparisons join the ledger
CentreGuard = Callable[[object, object, Ledger], str]


def unphysical_reason(series: LongPeriodSeries) -> str:
    """'' or why the orbit is unphysical (`classify.util.unphysical_arrays`: e > 1.3, a < 0, non-finite a or e)."""
    bad, reason = unphysical_arrays(np.asarray(series.a, dtype=float), np.asarray(series.e, dtype=float))
    return reason if bad else ''


def _unusable(series: LongPeriodSeries) -> bool:
    t, x = np.asarray(series.t_yr, dtype=float), np.asarray(series.angle, dtype=float)
    return len(t) < MIN_SAMPLES or not (np.all(np.isfinite(t)) and np.all(np.isfinite(x)))


def _empty_verdict(series: LongPeriodSeries, status: Status, subtype: AngleSubtype, reason: Reason, ledger: Ledger) -> Verdict:
    checks = unmeasured_checks(series.resonance_type)
    # unusable_series: review, no length would repair it (`extend.recommend`); invalid: review by default
    action = Action.REVIEW if reason == Reason.UNUSABLE_SERIES else default_action(status, reason, Confidence.NONE)
    return Verdict(
        resonance_type=series.resonance_type,
        status=status,
        angle_status=AngleStatus.UNDEFINED,
        angle_subtype=subtype,
        reason=reason,
        action=action,
        checks=checks,
        decision_path=list(ledger),
    )


def analyse_long_period(
    series: LongPeriodSeries,
    settings: AnalysisSettings,
    cells: Optional[ConfidenceCells] = None,
    gate: Optional[GateFunction] = None,
    centre_guard: Optional[CentreGuard] = None,
) -> LongPeriodAnalysis:
    """The analysis of a secular (no `gate`) or Lidov-Kozai (`gate`) angle; the steps of the module docstring."""
    cells = cells if cells is not None else confidence_module.cells_from_settings(settings)
    ledger = Ledger.from_settings(settings)
    common = dict(settings_hash=settings_hash(settings), confidence_cells_sha=cells.sha, filter_order=settings.filter_order)
    reason = unphysical_reason(series) if len(series.t_yr) else ''
    if reason:  # before the series check: a lost orbit leaves NaN in the angle too, and the orbit is the cause
        verdict = _empty_verdict(series, Status.INVALID, AngleSubtype.UNPHYSICAL, Reason.UNPHYSICAL, ledger)
        return LongPeriodAnalysis(series, verdict, ledger, unphysical=True, unphysical_reason=reason, **common)
    if _unusable(series):
        verdict = _empty_verdict(series, Status.UNCERTAIN, AngleSubtype.NONE, Reason.UNUSABLE_SERIES, ledger)
        return LongPeriodAnalysis(series, verdict, ledger, **common)

    cutoff = preprocess.choose_cutoff(series, settings)
    angle = preprocess.filtered_angle(series, cutoff, settings)
    if angle is None:
        verdict = _empty_verdict(series, Status.UNCERTAIN, AngleSubtype.NONE, Reason.UNUSABLE_SERIES, ledger)
        return LongPeriodAnalysis(series, verdict, ledger, cutoff=cutoff, **common)
    geometry = measure_geometry(angle, cutoff.prominence_rad, settings)
    angle_decision = decide.decide_angle(geometry, settings, ledger)
    fine_geometry, scale, rejected = geometry, SCALE_FINE, ''
    if angle_decision.angle_status != AngleStatus.LIBRATION:
        slow = _centre_scale(series, cutoff, angle, geometry, angle_decision, settings, ledger)
        rejected = centre_guard(slow[1], geometry, ledger) if slow is not None and centre_guard is not None else ''
        if slow is not None and not rejected:
            angle, geometry, angle_decision = slow
            scale = SCALE_CENTRE

    response = analyse_response(series, angle, settings) if series.resonance_type == ResonanceType.SECULAR else None
    gate_result = gate(angle_decision, geometry) if gate is not None else None
    status = decide.decide_status(series.resonance_type, angle_decision, gate_result, ledger)
    # F is read at the cutoff of the decisive angle (the centre scale's when it decided)
    margins, caps, flags = decide.margins_and_caps(angle_decision, geometry, angle.cutoff_yr, gate_result, settings, ledger)
    assessment = confidence_module.assess(
        status.status,
        str(angle_decision.angle_status),
        str(status.reason),
        margins,
        caps,
        ledger.thin_margins(),
        decide.PHYSICS_MARGINS[series.resonance_type],
        cells,
    )
    extension = extend_module.recommend(
        status.status,
        status.reason,
        gate_result,
        angle.period_floor_yr,
        geometry.period_cycle_yr,
        angle.cutoff_yr,
        series.span_yr,
        settings,
    )
    checks = unmeasured_checks(series.resonance_type)
    if response is not None:
        checks['check_secular_e_response'] = CheckValue(response.check)
    checks.update(status.checks)
    all_flags = list(dict.fromkeys(angle_decision.flags + flags))
    if status.status in (Status.UNCERTAIN, Status.INVALID):
        all_flags = [flag for flag in all_flags if flag != str(Flag.THREE_TP_EPISODE)]
    verdict = Verdict(
        resonance_type=series.resonance_type,
        status=status.status,
        angle_status=angle_decision.angle_status,
        angle_subtype=angle_decision.subtype,
        orbit_type=orbit_type_module.long_period_orbit_type(series.resonance_type, geometry, angle_decision.decisive_episode_id),
        confidence=assessment.confidence,
        confidence_raw=assessment.confidence_raw,
        **assessment.cell_fields(),
        reason=status.reason,
        flags=all_flags,
        thin_margins=assessment.thin_margins,
        failed_support=assessment.failed_support,
        caps=assessment.caps,
        action=extension.action if extension.action is not None else default_action(status.status, status.reason, assessment.confidence),
        extend_to_years=extension.extend_to_years,
        libration_origin=status.origin,
        origin_detail=status.origin_detail,
        checks=checks,
        margins=margins,
        decision_path=list(ledger),
    )
    centre_line = None
    if series.resonance_type == ResonanceType.SECULAR and angle_decision.subtype == AngleSubtype.MODULATED:
        centre_line = slow_module.secular_centre_line(fine_geometry.centre, cutoff.forced_waves_yr, series.span_yr, settings)
    return LongPeriodAnalysis(
        series,
        verdict,
        ledger,
        cutoff=cutoff,
        angle=angle,
        geometry=geometry,
        response=response,
        gate=gate_result,
        extension=extension,
        decisive_episode_id=angle_decision.decisive_episode_id,
        scale=scale,
        fine_geometry=fine_geometry if scale == SCALE_CENTRE else None,
        centre_rejected=rejected,
        centre_line=centre_line,
        n_librating_episodes=len(decide.librating_episodes(geometry)),
        **common,
    )


def _centre_scale(series, cutoff, angle, geometry, angle_decision, settings, ledger):
    """(angle, geometry, decision) at the scale of the centre, when that scale shows more libration than the fine one:
    a libration there (the centre librates: the slow libration under faster waves), or libration episodes where the
    fine scale has none. None otherwise; the comparisons of the centre scale join the ledger either way."""
    found = preprocess.centre_scale_angle(series, cutoff, angle, geometry.period_cycle_yr, settings)
    if found is None:
        return None
    angle, prominence = found
    slow_geometry = measure_geometry(angle, prominence, settings)
    slow_ledger = Ledger.from_settings(settings)
    slow_decision = decide.decide_angle(slow_geometry, settings, slow_ledger)
    ledger.extend(Ledger(slow_ledger.thin_band, slow_ledger.conventions, [_centre_entry(entry) for entry in slow_ledger]))
    bounded = (AngleStatus.LIBRATION, AngleStatus.EPISODIC)
    better = slow_decision.angle_status == AngleStatus.LIBRATION or (
        slow_decision.angle_status == AngleStatus.EPISODIC and angle_decision.angle_status not in bounded
    )
    return (angle, slow_geometry, slow_decision) if better else None


def _centre_entry(entry):
    """A ledger entry of the centre scale, named so (`centre_scale:<name>`)."""
    return replace(entry, name=f'centre_scale:{entry.name}')


def analyse_secular(series: LongPeriodSeries, settings: AnalysisSettings, cells: Optional[ConfidenceCells] = None) -> LongPeriodAnalysis:
    """The verdict of a secular angle (module docstring): geometry decides, the e/i response is a column."""
    return analyse_long_period(series, settings, cells)
