"""The decision of a secular or Lidov-Kozai angle: angle_status from the geometry, status from the physics that is
measured for the type, the margins and the caps of the confidence. Every comparison goes through the ledger.

angle_status (the shared geometry of the filtered angle, `analysis.geometry`):
    an accepted episode (>= LIBRATION_MIN_TURNING_POINTS turning points, a stationary or modulated centre, a period
    not below the type's shortest libration, a centre that does not run a turn inside it) covering the record (both
    ends within LIBRATION_COVERAGE_PERIODS periods)                             -> libration (stationary | modulated)
    accepted episodes, none covering                                            -> episodic
    no episode; the centre's largest monotone pass >= 1 turn                    -> circulation
    no turning point; the filtered angle turns >= 1 time                         -> circulation
    no turning point, less than a turn                                          -> insufficient_time (arc | stall)
    turning points; runs rejected for a moving centre cover
        >= LIBRATION_DRIFT_MIN_WINDOW_FRACTION of the window                    -> drift
    otherwise (fewer turning points than one cycle)                             -> insufficient_time (few_cycles)

This keeps the meaning of the former secular centre check: a bounded angle is a libration only when it turns back
LIBRATION_MIN_TURNING_POINTS times (a cycle: 1020 Arcadia, turning points at 1.3, 4.6 and 8.0 Myr); a bounded angle
that turns back fewer times is uncertain, a longer integration would tell (2005 GQ107, 363 Padua).

status:
    libration -> resonant (dynamic_libration), episodic -> transient (dynamic_episode); circulation -> non_resonant;
    drift and insufficient_time -> uncertain, baseline_needed (action extend).
    Lidov-Kozai: the free-omega gate judges every omega that stays within a turn (`GATED`, `analysis.zlk`): on a
    libration, confirmed keeps the status; kinematic (the free omega winds) -> non_resonant, kinematic_libration,
    origin kinematic; no verdict (short window, phase undefined, a residual at a planetary rate, no free elements)
    -> uncertain, gate_unavailable. On an episodic angle, a drift or too few cycles only a kinematic verdict with
    rho_max < 1 (the free circle never around the origin) changes the status, to non_resonant, kinematic_libration;
    an episodic one also when inside every librating episode the median local rho < 1 and the free omega advances
    >= ZLK_EPISODE_FREE_TURNS_PER_CYCLE turns per cycle (the episodes are the forced vector's, 624 Hektor).
    The two lower outcomes are different statuses and stay apart (CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.geometry import EpisodeShape, Geometry, covering_episode, librating_episodes_with, run_of
from resonances.resonance.analysis.ledger import Ledger, Margin, Op
from resonances.resonance.analysis.secular.models import GateResult
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Flag,
    LibrationOrigin,
    MarginClass,
    MarginValue,
    OriginDetail,
    Reason,
    ResonanceType,
    Status,
    margin_of_check,
)

NAN = float('nan')

# The physics margins of the confidence rule (`confidence.assess`): the free-omega gate for ZLK; none for a secular
# angle, whose physics check (the e/i response) is a column only for now (author, Q11).
PHYSICS_MARGINS = {ResonanceType.SECULAR: (), ResonanceType.LIDOV_KOZAI: ('G',)}
# GateOutcome values (omega_free_gate) the decision reads
GATE_CONFIRMED = 'confirmed'
GATE_KINEMATIC = 'kinematic'
GATE_DISABLED = 'disabled'
GATE_NOT_APPLICABLE = 'not_applicable'
GATE_NO_VERDICT = ('short_window', 'undefined_phase', 'planetary_residual', 'unavailable')
# ZLK angle statuses the gate judges: every omega that stays within a turn (the TTO classifier's LIBRATION, which the
# gate judged, is any of them in the shared geometry; `decide_status` says how much each takes from it)
GATED = (AngleStatus.LIBRATION, AngleStatus.EPISODIC, AngleStatus.DRIFT, AngleStatus.INSUFFICIENT_TIME)


@dataclass
class AngleDecision:
    angle_status: AngleStatus
    subtype: AngleSubtype = AngleSubtype.NONE
    decisive_episode_id: Optional[int] = None
    flags: List[str] = field(default_factory=list)


@dataclass
class StatusDecision:
    status: Status
    reason: Reason | str
    origin: LibrationOrigin = LibrationOrigin.NONE
    origin_detail: str = ''
    checks: Dict[str, CheckValue] = field(default_factory=dict)


def librating_episodes(geometry: Geometry) -> List:
    """The accepted episodes that are librations of this angle: not faster than the type's shortest libration
    (geometry.period_floor_yr: a faster 'episode' is a forced wave the cutoff let through), and the centre does not run
    a full turn inside the episode (turns around a centre that circulates are no libration of this angle)."""
    floor = geometry.period_floor_yr if np.isfinite(geometry.period_floor_yr) else NAN
    return librating_episodes_with(geometry, lambda episode: floor)


def decide_angle(geometry: Geometry, settings: AnalysisSettings, ledger: Ledger) -> AngleDecision:
    """angle_status and subtype from the geometry (module docstring)."""
    accepted = librating_episodes(geometry)
    if ledger.compare('librating_episodes', len(accepted), Op.GE, 1, '', 'sec_n_librating_episodes'):
        covering = covering_episode(accepted, settings)
        decisive = covering if covering is not None else max(accepted, key=lambda episode: episode.n_tp)
        coverage = 'libration_coverage_periods'
        start = ledger.compare_setting(
            'coverage_start', decisive.coverage_start_gap_p, Op.LE, settings, coverage, 'P', 'ep_coverage_start_gap_P'
        )
        end = ledger.compare_setting('coverage_end', decisive.coverage_end_gap_p, Op.LE, settings, coverage, 'P', 'ep_coverage_end_gap_P')
        flags = list(geometry.flags)
        if covering is not None and start and end:
            subtype = AngleSubtype.MODULATED if covering.shape == EpisodeShape.MODULATED else AngleSubtype.STATIONARY
            return AngleDecision(AngleStatus.LIBRATION, subtype, covering.episode_id, flags)
        return AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, decisive.episode_id, flags)
    if ledger.compare('centre_pass', geometry.centre.max_pass_rev, Op.GE, 1.0, 'rev', 'geo_centre_max_pass_rev'):
        return AngleDecision(AngleStatus.CIRCULATION)
    if not ledger.compare('turning_points', len(geometry.turning_points), Op.GE, 1, '', 'geo_n_tp'):
        if ledger.compare('net_revolutions', abs(geometry.net_rev_filtered), Op.GE, 1.0, 'rev', 'geo_net_rev_filtered'):
            return AngleDecision(AngleStatus.CIRCULATION)
        arc_range = geometry.arc_stall.range_rad if geometry.arc_stall is not None else float('nan')
        arc = ledger.compare('arc_range', arc_range, Op.GE, np.pi, 'rad', 'geo_arc_range_rad')
        return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC if arc else AngleSubtype.STALL)
    drift = ledger.compare_setting(
        'drift_fraction', geometry.drift_fraction, Op.GE, settings, 'libration_drift_min_window_fraction', '', 'geo_drift_fraction'
    )
    if drift:
        return AngleDecision(AngleStatus.DRIFT)
    return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES)


def decide_status(resonance_type: ResonanceType, angle: AngleDecision, gate: Optional[GateResult], ledger: Ledger) -> StatusDecision:
    """The status of the angle decision (module docstring); for ZLK with the gate's outcome."""
    status = _status_by_geometry(angle.angle_status)
    if resonance_type != ResonanceType.LIDOV_KOZAI:
        return status
    bounded = angle.angle_status in GATED
    if not bounded or gate is None or gate.outcome in ('', GATE_NOT_APPLICABLE):
        status.checks['check_zlk_free_omega'] = CheckValue.NOT_MEASURED if bounded else CheckValue.NOT_APPLICABLE
        return status
    # the gate's own comparisons are in omega_free_gate; its cycle count is recorded here (FREE_GATE_MIN_CYCLES)
    ledger.compare('gate_cycles', gate.n_cycles, Op.GE, gate.min_cycles, 'cycles', 'zlk_gate_n_cycles')
    if angle.angle_status != AngleStatus.LIBRATION:
        # an episodic angle, a drift, too few cycles: the free elements of the whole record judge them only when they
        # find the forced vector holding omega throughout (kinematic, free circle never around the origin), or, for an
        # episodic one, inside every librating episode (free circle off the origin there, rho < 1 the geometric bound;
        # the free vector turning about the forced one with each cycle: `zlk.analyse._episode_pinning`)
        pinned = gate.outcome == GATE_KINEMATIC and (
            ledger.compare('free_circle_rho_max', gate.rho_max, Op.LT, 1.0, '', 'zlk_gate_rho_max')
            or (
                angle.angle_status == AngleStatus.EPISODIC
                and ledger.compare('free_circle_rho_episodes', gate.rho_episodes, Op.LT, 1.0, '', 'zlk_gate_rho_episodes')
                and ledger.compare(
                    'free_omega_turns_episodes',
                    gate.free_turns_episodes,
                    Op.GE,
                    gate.free_turns_min,
                    'turns/cycle',
                    'zlk_gate_free_turns_episodes',
                    setting='zlk_episode_free_turns_per_cycle',
                )
            )
        )
        if not pinned:
            status.checks['check_zlk_free_omega'] = CheckValue.NOT_MEASURED
            return status
    if gate.outcome in (GATE_CONFIRMED, GATE_DISABLED):
        status.checks['check_zlk_free_omega'] = CheckValue.PASS if gate.outcome == GATE_CONFIRMED else CheckValue.NOT_MEASURED
        return status
    if gate.outcome == GATE_KINEMATIC:
        return StatusDecision(
            Status.NON_RESONANT,
            Reason.KINEMATIC_LIBRATION,
            LibrationOrigin.KINEMATIC,
            OriginDetail.FREE_OMEGA_WINDS,  # the free omega winds, the forced e holds the osculating one (591986)
            {'check_zlk_free_omega': CheckValue.FAIL},
        )
    return StatusDecision(
        Status.UNCERTAIN, Reason.GATE_UNAVAILABLE, LibrationOrigin.UNKNOWN, '', {'check_zlk_free_omega': CheckValue.NOT_MEASURED}
    )


def _status_by_geometry(angle_status: AngleStatus) -> StatusDecision:
    if angle_status == AngleStatus.LIBRATION:
        return StatusDecision(Status.RESONANT, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC)
    if angle_status == AngleStatus.EPISODIC:
        return StatusDecision(Status.TRANSIENT, Reason.DYNAMIC_EPISODE, LibrationOrigin.DYNAMIC)
    if angle_status == AngleStatus.CIRCULATION:
        return StatusDecision(Status.NON_RESONANT, Reason.CIRCULATION)
    origin = LibrationOrigin.UNKNOWN if angle_status == AngleStatus.DRIFT else LibrationOrigin.NONE
    return StatusDecision(Status.UNCERTAIN, Reason.BASELINE_NEEDED, origin)


def margins_and_caps(
    angle: AngleDecision,
    geometry: Geometry,
    cutoff_yr: float,
    gate: Optional[GateResult],
    settings: AnalysisSettings,
    ledger: Ledger,
) -> Tuple[Dict[str, MarginValue], List[str], List[str]]:
    """(margins, caps, flags) of the confidence rule.

    N  turning points of the decisive episode: thin at the minimum (with the cap three_tp_episode, ceiling high);
    S  its centre: full when every cycle's band overlaps with enough independent cycles, not measured with fewer
       (one cycle, one centre), not applicable to a modulated centre;
    F  the filter: the libration period >= FILTER_MIN_LIBRATION_CUTOFFS x P_c, else not measured (filter_too_slow);
    P  a circulation: its largest centre pass >= 1 turn at every smoothing (x0.5, x1, x2), else thin;
    G  ZLK: the gate confirmed with n_cycles clear of its minimum (thin within CONFIDENCE_THIN_BAND).
    A visible exit or entry caps the level at medium (exit_suspected, entry_suspected).
    """
    margins: Dict[str, MarginValue] = {}
    caps: List[str] = []
    flags: List[str] = []
    episode = run_of(geometry, angle.decisive_episode_id)
    if episode is not None:
        _episode_margins(episode, geometry, cutoff_yr, settings, ledger, margins, caps, flags)
    if angle.angle_status == AngleStatus.CIRCULATION:
        low = geometry.centre.max_pass_rev_lo
        if np.isfinite(low):
            margins['P'] = MarginValue(Margin.FULL if low >= 1.0 else Margin.THIN, float(low), MarginClass.DECISIVE)
    if gate is not None and gate.outcome == GATE_CONFIRMED:
        cycles = gate.n_cycles if gate.n_cycles is not None else float('nan')
        clear = np.isfinite(cycles) and cycles >= gate.min_cycles * (1 + settings.confidence_thin_band)
        margins['G'] = MarginValue(Margin.FULL if clear else Margin.THIN, float(cycles), MarginClass.DECISIVE)
    elif gate is not None and gate.outcome == GATE_KINEMATIC and (angle.angle_status == AngleStatus.LIBRATION or gate.rho_max < 1.0):
        margins['G'] = MarginValue(Margin.FULL, gate.n_cycles, MarginClass.DECISIVE)
    return margins, caps, flags


def _episode_margins(episode, geometry: Geometry, cutoff_yr: float, settings, ledger: Ledger, margins, caps, flags) -> None:
    """N, S and F of the decisive episode, and the caps of a visible exit or entry (`margins_and_caps`)."""
    n_tp = episode.n_tp
    at_minimum = n_tp <= settings.libration_min_turning_points
    margins['N'] = MarginValue(Margin.THIN if at_minimum else Margin.FULL, float(n_tp), MarginClass.SUPPORT)
    if at_minimum:
        caps.append(str(Cap.THREE_TP_EPISODE))
        flags.append(str(Flag.THREE_TP_EPISODE))
    if episode.shape == EpisodeShape.MODULATED:
        margins['S'] = MarginValue(Margin.NOT_APPLICABLE, None, MarginClass.SUPPORT)
    else:
        margins['S'] = MarginValue(margin_of_check(episode.centre_stationarity), float(episode.centre_estimates), MarginClass.SUPPORT)
    period = geometry.period_cycle_yr
    ratio = period / cutoff_yr if np.isfinite(period) and cutoff_yr > 0 else None
    guard = ledger.compare_setting('filter_guard', ratio, Op.GE, settings, 'filter_min_libration_cutoffs', 'P_c', 'geo_period_cycle_yr')
    margins['F'] = MarginValue(Margin.FULL if guard else Margin.NOT_MEASURED, guard.value, MarginClass.SUPPORT)
    if guard.result is False:
        flags.append(str(Flag.FILTER_TOO_SLOW))
    for flag, cap in ((Flag.EXIT_SUSPECTED, Cap.EXIT_SUSPECTED), (Flag.ENTRY_SUSPECTED, Cap.ENTRY_SUSPECTED)):
        if str(flag) in geometry.flags:
            caps.append(str(cap))
