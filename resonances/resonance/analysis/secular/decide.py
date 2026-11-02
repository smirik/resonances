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
        >= LIBRATION_DRIFT_MIN_WINDOW_FRACTION of the window, and the centre
        monotone (no turn of the slow centre away from the edges)               -> drift
    the same, but the centre turned back (largest pass < 1 turn)                -> insufficient_time (few_cycles)
    otherwise (fewer turning points than one cycle)                             -> insufficient_time (few_cycles)

A secular libration or episode under the linear mode phases must swing more than the phases can stray (`phase_check`,
W28 D5, card g09-3): 2 A_max <= SECULAR_PHASE_UNCERTAINTY_FACTOR x dphi(T), dphi(T) = sum_j |k_j| range_j(T) over the
modes of the angle that follow the linear law (`secular.phase_uncertainty`)  -> insufficient_time (phase_unresolved),
action review: rerun with SECULAR_PLANET_PHASES=measured (5026 g+g2-g4-g6: 2 A_max = 22 deg, dphi(10 Myr) = 34 deg).

This keeps the meaning of the former secular centre check: a bounded angle is a libration only when it turns back
LIBRATION_MIN_TURNING_POINTS times (a cycle: 1020 Arcadia, turning points at 1.3, 4.6 and 8.0 Myr); a bounded angle
that turns back fewer times is uncertain, a longer integration would tell (2005 GQ107, 363 Padua).

status:
    libration -> resonant (dynamic_libration), episodic -> transient (dynamic_episode); circulation -> non_resonant;
    drift and insufficient_time -> uncertain, baseline_needed (action extend; phase_unresolved: review).
    Lidov-Kozai: the free-omega gate judges every omega that stays within a turn (`GATED`, `analysis.zlk`): on a
    libration, confirmed keeps the status; kinematic (the free omega winds) -> non_resonant, kinematic_libration,
    origin kinematic; no verdict (short window, phase undefined, a residual at a planetary rate, no free elements)
    -> uncertain, gate_unavailable. On an episodic angle, a drift or too few cycles only a kinematic verdict with
    rho_max < 1 (the free circle never around the origin) changes the status, to non_resonant, kinematic_libration;
    an episodic one also when inside every librating episode the median local rho < 1 and the free omega advances
    >= ZLK_EPISODE_FREE_TURNS_PER_CYCLE turns per cycle (the episodes are the forced vector's, 624 Hektor). A
    co-orbital of a planet whose modes the split does not fit (outcome unavailable, detail coorbital_unfitted_forcing:
    the free-element model does not apply) is uncertain, gate_unavailable whatever bounded angle it has, as a
    libration is (2020 PP1, W28 g09-6 B; rule zlk_coorbital_episodic). Any other unverdicted outcome of an episodic
    angle goes by ZLK_EPISODIC_NO_VERDICT (`episodic_without_verdict`, W28 D10, the author's q-1 E): transient only
    when e and i trade at constant H inside the deciding episode (`zlk.episode_ei`, coupling zlk), else uncertain,
    gate_unavailable (88899 over 500 kyr: none; 797204 over 100 kyr and 2023 RS2: episodes too short to measure); rule
    zlk_gate_unconfirmed. A drift or too few cycles keeps baseline_needed. The two lower outcomes are different
    statuses and stay apart (CLAUDE.md).

The period the physics reads (the gate's period and cycle count, F, the e-i coupling, the rho window, the extension)
is that of the deciding librating episode (`decisive_period_yr`), not the median over every episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from resonances.resonance.analysis import rules as rules_module
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.geometry import (
    Episode,
    EpisodeShape,
    Geometry,
    covering_episode,
    librating_episodes_with,
    run_of,
)
from resonances.resonance.analysis.ledger import Ledger, Margin, Op
from resonances.resonance.analysis.mmr.decide import interior_turns
from resonances.resonance.analysis.secular.models import EpisodeCoupling, GateResult, LongPeriodSeries
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.secular import phase_uncertainty
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
GATE_UNAVAILABLE = 'unavailable'
GATE_NO_VERDICT = ('short_window', 'undefined_phase', 'planetary_residual', GATE_UNAVAILABLE)
# GateResult.detail of an unavailable outcome: a co-orbital of a planet whose own modes the split does not fit
# (`omega_free_gate.COORBITAL_UNFITTED`, kept as a string here: the decision does not import the gate)
DETAIL_COORBITAL_UNFITTED = 'coorbital_unfitted_forcing'
# ZLK_EPISODIC_NO_VERDICT values (`episodic_without_verdict`) and the coupling class it reads (`EpisodeCoupling`)
EPISODIC_EI = 'ei'
EPISODIC_UNCERTAIN = 'uncertain'
EPISODIC_KEEP = 'keep'
COUPLING_ZLK = 'zlk'
# ZLK angle statuses the gate judges: every omega that stays within a turn (the TTO classifier's LIBRATION, which the
# gate judged, is any of them in the shared geometry; `decide_status` says how much each takes from it)
GATED = (AngleStatus.LIBRATION, AngleStatus.EPISODIC, AngleStatus.DRIFT, AngleStatus.INSUFFICIENT_TIME)


@dataclass
class AngleDecision:
    angle_status: AngleStatus
    subtype: AngleSubtype = AngleSubtype.NONE
    decisive_episode_id: Optional[int] = None
    flags: List[str] = field(default_factory=list)
    rules_fired: frozenset = frozenset()  # rules (`analysis.rules`) that acted on the angle word, merged by `rules_fired`


@dataclass
class StatusDecision:
    status: Status
    reason: Reason | str
    origin: LibrationOrigin = LibrationOrigin.NONE
    origin_detail: str = ''
    checks: Dict[str, CheckValue] = field(default_factory=dict)
    rules_fired: frozenset = frozenset()  # rules (`analysis.rules`) that acted on the status, merged by `rules_fired`


def librating_episodes(geometry: Geometry) -> List:
    """The accepted episodes that are librations of this angle: not faster than the type's shortest libration
    (geometry.period_floor_yr: a faster 'episode' is a forced wave the cutoff let through), and the centre does not run
    a full turn inside the episode (turns around a centre that circulates are no libration of this angle)."""
    floor = geometry.period_floor_yr if np.isfinite(geometry.period_floor_yr) else NAN
    return librating_episodes_with(geometry, lambda episode: floor)


def decisive_librating_episode(geometry: Optional[Geometry], episode_id: Optional[int]) -> Optional[Episode]:
    """The episode the angle decision rests on (`AngleDecision.decisive_episode_id`) when it is a librating one
    (`librating_episodes`): the covering episode of a libration, the one with the most turning points of an episodic
    angle; None otherwise (drift, too few cycles, circulation, no geometry)."""
    if geometry is None or episode_id is None:
        return None
    return next((episode for episode in librating_episodes(geometry) if episode.episode_id == episode_id), None)


def decisive_period_yr(geometry: Optional[Geometry], episode_id: Optional[int]) -> float:
    """The libration period the physics of the row reads [yr]: the gate's period and cycle count, F, the e-i coupling's
    period, the rho window and the extension. The median full period of the deciding librating episode
    (`decisive_librating_episode`); without one, the median full period pooled over the runs not faster than the type's
    shortest libration (accepted runs first, as P_cyc), nan when there is none.

    Never the median pooled over every episode (`Geometry.period_cycle_yr`, a column): a run faster than the floor (a
    forced wave the cutoff let through) or a second episode at another period would set the scale of a verdict that
    does not rest on it (2020 PP1, W28 g09-6)."""
    if geometry is None:
        return NAN
    episode = decisive_librating_episode(geometry, episode_id)
    if episode is not None:
        return float(episode.period_med_yr)
    floor = geometry.period_floor_yr if np.isfinite(geometry.period_floor_yr) else -np.inf
    runs = [run for run in geometry.runs if run.n_tp >= 3 and not run.period_med_yr < floor]
    pool = [run for run in runs if run.accepted] or runs
    if not pool:
        return NAN
    times = [np.asarray(run.tp_times_yr, dtype=float) for run in pool]
    return float(np.median(np.concatenate([t[2:] - t[:-2] for t in times])))


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
        # a drift is a centre that keeps going one way; one that turned back inside the record (its largest pass is
        # below a turn here) holds part of a slow swing, too little to tell a drift from a slow libration (2306 z1:
        # turn at 2.04 Myr, largest pass 0.92 rev in 10 Myr)
        if ledger.compare('centre_monotone', interior_turns(geometry), Op.EQ, 0, '', 'geo_centre_n_tp_interior'):
            return AngleDecision(AngleStatus.DRIFT)
        fired = rules_module.fire(frozenset(), rules_module.Rule.CENTRE_TURNED_INSUFFICIENT)
        return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, rules_fired=fired)
    return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES)


def phase_uncertainty_rad(series: LongPeriodSeries, table: Optional[phase_uncertainty.PhaseUncertaintyTable] = None) -> float:
    """dphi(T) of a secular angle [rad]: sum_j |k_j| range_j(T) over its planetary modes that follow the linear law (not
    in `series.measured_modes`), T the record's span (`secular.phase_uncertainty`); nan for a ZLK angle or a secular
    angle without planetary modes (nothing to stray)."""
    if series.resonance_type != ResonanceType.SECULAR:
        return NAN
    coefficients: Dict[str, int] = {}
    for term in series.terms:
        if term.index is not None:
            label = f'{term.mode}{term.index}'
            coefficients[label] = coefficients.get(label, 0) + term.coefficient
    if not any(coefficients.values()):
        return NAN
    table = table if table is not None else phase_uncertainty.load_table()
    return table.delta_phi_rad(coefficients, series.span_yr, series.measured_modes)


def phase_swing_rad(angle: AngleDecision, geometry: Geometry) -> float:
    """2 x the largest half-amplitude of the deciding episode [rad]; nan without one."""
    episode = run_of(geometry, angle.decisive_episode_id)
    return 2.0 * float(episode.half_amp_max_rad) if episode is not None else NAN


def phase_check(
    angle: AngleDecision, geometry: Geometry, delta_phi_rad: float, settings: AnalysisSettings, ledger: Ledger
) -> AngleDecision:
    """A secular libration or episode whose swing is within the uncertainty of its linear mode phases becomes
    insufficient_time / phase_unresolved (module docstring): its turning points may be those of the phases. Ledger
    entry phase_uncertainty: swing / dphi > SECULAR_PHASE_UNCERTAINTY_FACTOR (a convention; 0 switches it off). Any
    other angle status, or dphi not positive (every mode measured, none in the angle), passes unchanged; so does an
    unmeasured swing (no verdict changes on a number that is not there)."""
    if angle.angle_status not in (AngleStatus.LIBRATION, AngleStatus.EPISODIC):
        return angle
    if not (np.isfinite(delta_phi_rad) and delta_phi_rad > 0):
        return angle
    swing = phase_swing_rad(angle, geometry)
    entry = ledger.compare_setting(
        'phase_uncertainty', swing / delta_phi_rad, Op.GT, settings, 'secular_phase_uncertainty_factor', '', 'sec_phase_swing_ratio'
    )
    if entry.result is not False:
        return angle
    fired = rules_module.fire(angle.rules_fired, rules_module.Rule.PHASE_UNRESOLVED)
    return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.PHASE_UNRESOLVED, angle.decisive_episode_id, list(angle.flags), fired)


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
    if angle.angle_status != AngleStatus.LIBRATION and gate.outcome == GATE_UNAVAILABLE and gate.detail == DETAIL_COORBITAL_UNFITTED:
        # a co-orbital of a planet whose own modes the split does not fit: the free-element model does not apply, so an
        # episodic omega, a drift or too few cycles has no gate verdict, as a libration has none (W28 g09-6 B: 2020 PP1)
        return StatusDecision(
            Status.UNCERTAIN,
            Reason.GATE_UNAVAILABLE,
            LibrationOrigin.UNKNOWN,
            '',
            {'check_zlk_free_omega': CheckValue.NOT_MEASURED},
            rules_fired=rules_module.fire(frozenset(), rules_module.Rule.ZLK_COORBITAL_EPISODIC),
        )
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
            if angle.angle_status == AngleStatus.EPISODIC and gate.outcome in GATE_NO_VERDICT:
                return episodic_without_verdict(status, gate, ledger)
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


def episodic_without_verdict(status: StatusDecision, gate: GateResult, ledger: Ledger) -> StatusDecision:
    """The status of an episodic ZLK omega whose gate gives no verdict (not an unfitted co-orbital), by
    ZLK_EPISODIC_NO_VERDICT (`gate.episodic_policy`; the author's answer q-1 E, 2026-10-31):

        ei         transient (dynamic_episode) when e and i trade at constant H inside the deciding episode
                   (`gate.episode_ei.coupling` zlk, `zlk.episode_ei`); otherwise (none, forced_beat, not measured)
                   uncertain / gate_unavailable (-9): no physical measurement stands behind the episode;
        uncertain  always uncertain / gate_unavailable;
        keep       the geometric status (transient), as before W28.

    Under ei and uncertain the rule zlk_gate_unconfirmed fires (it decided the status, also when it kept it). The -9
    is the absence of a measurement; a kinematic -2 never comes from here."""
    policy = gate.episodic_policy
    if policy == EPISODIC_KEEP:
        return status
    fired = rules_module.fire(status.rules_fired, rules_module.Rule.ZLK_GATE_UNCONFIRMED)
    if policy == EPISODIC_EI:
        _record_episode_coupling(gate.episode_ei, ledger)
        if gate.episode_ei.coupling == COUPLING_ZLK:
            status.rules_fired = fired
            return status
    return StatusDecision(Status.UNCERTAIN, Reason.GATE_UNAVAILABLE, LibrationOrigin.UNKNOWN, '', dict(status.checks), rules_fired=fired)


def _record_episode_coupling(ei: EpisodeCoupling, ledger: Ledger) -> None:
    """The comparisons behind the class zlk of the deciding episode's coupling (`zlk.episode_ei`), in the ledger: its
    length in periods, the phase of i against e off antiphase, the signed coherence and the residual of H. The class
    itself is the measurement's; a number that is not there is recorded as not measured."""
    if not ledger.compare(
        'ei_episode_cycles', ei.n_periods, Op.GE, ei.min_cycles, 'P', 'zlk_ei_episode_n_periods', setting='zlk_ei_min_cycles'
    ):
        return
    off = abs(ei.phase_deg % 360.0 - 180.0) if np.isfinite(ei.phase_deg) else NAN  # distance from antiphase
    ledger.compare('ei_episode_antiphase', off, Op.LE, ei.phase_tol_deg, 'deg', '', setting='zlk_ei_phase_tol_deg')
    ledger.compare(
        'ei_episode_coherence',
        ei.coherence,
        Op.LE,
        -ei.min_coherence,
        '',
        'zlk_ei_episode_coherence',
        setting='zlk_ei_episode_min_coherence',
    )
    ledger.compare('ei_episode_h_residual', ei.eta_h, Op.LE, ei.h_residual_max, '', 'zlk_ei_episode_eta_h', setting='zlk_ei_h_residual_max')


def rules_fired(geometry: Optional[Geometry], angle: AngleDecision, status: StatusDecision, *fired) -> List[str]:
    """`Verdict.rules_fired` of a secular or ZLK row: the rules of the decisive geometry, of the angle and status
    decisions and of every other frozenset given (e.g. the gate's), deduplicated, in registry order (`rules.ordered`)."""
    geometric = geometry.rules_fired if geometry is not None else ()
    return rules_module.ordered(geometric, angle.rules_fired, status.rules_fired, *fired)


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
    F  the filter: the decisive episode's period (`decisive_period_yr`) >= FILTER_MIN_LIBRATION_CUTOFFS x P_c, else
       not measured (filter_too_slow);
    P  a circulation: its largest centre pass at every smoothing (x0.5, x1, x2) clear of one turn by
       CONFIDENCE_THIN_BAND (>= 1.1 rev), else thin (a pass of 1.04 rev is a circulation by 4 %: 88899 500 kyr);
    G  ZLK: the gate confirmed with n_cycles clear of its minimum (thin within CONFIDENCE_THIN_BAND).
    A visible exit or entry caps the level at medium (exit_suspected, entry_suspected).
    """
    margins: Dict[str, MarginValue] = {}
    caps: List[str] = []
    flags: List[str] = []
    episode = run_of(geometry, angle.decisive_episode_id)
    if episode is not None:
        period = decisive_period_yr(geometry, angle.decisive_episode_id)
        _episode_margins(episode, geometry, period, cutoff_yr, settings, ledger, margins, caps, flags)
    if angle.angle_status == AngleStatus.CIRCULATION:
        low = geometry.centre.max_pass_rev_lo
        if np.isfinite(low):
            clear = low >= 1.0 + settings.confidence_thin_band
            margins['P'] = MarginValue(Margin.FULL if clear else Margin.THIN, float(low), MarginClass.DECISIVE)
    if gate is not None and gate.outcome == GATE_CONFIRMED:
        cycles = gate.n_cycles if gate.n_cycles is not None else float('nan')
        clear = np.isfinite(cycles) and cycles >= gate.min_cycles * (1 + settings.confidence_thin_band)
        margins['G'] = MarginValue(Margin.FULL if clear else Margin.THIN, float(cycles), MarginClass.DECISIVE)
    elif gate is not None and gate.outcome == GATE_KINEMATIC and (angle.angle_status == AngleStatus.LIBRATION or gate.rho_max < 1.0):
        margins['G'] = MarginValue(Margin.FULL, gate.n_cycles, MarginClass.DECISIVE)
    return margins, caps, flags


def _episode_margins(episode, geometry: Geometry, period: float, cutoff_yr: float, settings, ledger: Ledger, margins, caps, flags) -> None:
    """N, S and F of the decisive episode (F at `period`, `decisive_period_yr`), and the caps of a visible exit or
    entry (`margins_and_caps`)."""
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
    ratio = period / cutoff_yr if np.isfinite(period) and cutoff_yr > 0 else None
    guard = ledger.compare_setting('filter_guard', ratio, Op.GE, settings, 'filter_min_libration_cutoffs', 'P_c', 'sec_decisive_period_yr')
    margins['F'] = MarginValue(Margin.FULL if guard else Margin.NOT_MEASURED, guard.value, MarginClass.SUPPORT)
    if guard.result is False:
        flags.append(str(Flag.FILTER_TOO_SLOW))
    for flag, cap in ((Flag.EXIT_SUSPECTED, Cap.EXIT_SUSPECTED), (Flag.ENTRY_SUSPECTED, Cap.ENTRY_SUSPECTED)):
        if str(flag) in geometry.flags:
            caps.append(str(cap))
