"""The W29 minimal policy layer: the status, the level and the action of a finished v2 analysis, re-read by a few
physical principles (tmp W29 PRINCIPLES, agreed with the author 2026-11-02; docs/physics/status-tree.md).

The v2 tree (`mmr.decide`, `secular.decide`) runs first and measures everything; this layer then reads its finished
analysis and decides again:

    P1 status      from the geometry of sigma_f: libration -> resonant, episodic -> transient, near_separatrix ->
                   near_separatrix, circulation -> non_resonant; a drift whose centre passes >= one own revolution ->
                   non_resonant / modulated_circulation, less -> uncertain; insufficient_time -> uncertain, or
                   non_resonant / circulation when the record ends in circulation (sigma_f travels >= one own revolution
                   after the last turning point of an accepted or candidate run). One own revolution is 2 pi, a ZLK
                   omega's too (W30 D3: the quadrupole's cos 2 omega symmetry is not general; its runs still break at
                   half a turn).
    P2 vetoes      a bounded angle becomes non_resonant only on a measured contradiction (`VETO_*`): an owner that
                   induces the episode, a kinematic origin, the strength computed from the disturbing function far
                   below the need (two-body), the ZLK gate measured kinematic or the free e below the forced e over the
                   record, the ZLK e-i integral broken where the forced terms are small (W30 D1).
                   Every other leaf of the tree that moved a status gives the geometry's status back with a flag.
    P3 level       `level`: from the complete cycles N, the amplitude margin M, the veto state V and the filter
                   stability S; no caps (every v2 cap becomes a flag).
    P5 extend      `extend_clause`: only an unfinished libration at the record's end, a deciding period longer
                   than 2T / LIBRATION_MIN_TURNING_POINTS, or a needed veto a longer record measures within the
                   next-check budget (clause C, flag veto_needs_length); a record that ends in circulation is decided
                   now. A needed veto beyond the budget: review.
    P6 episodes    the counted episodes and the candidates (drawn faint, not counted) go to `PolicyResult`, the
                   verdict's columns and episodes.csv (`counted`, `candidate`).

It runs once per record, after every v2 pass (`mmr.analyse.analyse_mmr`, `secular.analyse.analyse_secular`,
`zlk.analyse.analyse_zlk`); every comparison it makes goes to the decision path as `policy:*`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis.geometry import (
    LEG_EDGE_END,
    LEG_EDGE_START,
    Episode,
    Geometry,
    covers_edges,
    edge_travel_rev,
    run_of,
    short_runs,
)
from resonances.resonance.analysis.ledger import Ledger, LedgerEntry, Margin, Op
from resonances.resonance.analysis import orbit_type as orbit_type_module
from resonances.resonance.analysis.mmr import decide as mmr_decide
from resonances.resonance.analysis.mmr.strength import DIRECT_METHODS
from resonances.resonance.analysis.pendulum import has_zone_cycle
from resonances.resonance.analysis.secular import decide as secular_decide
from resonances.resonance.analysis.secular import extend as secular_extend
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    CONFIDENCE_ORDER,
    POLICY_MINIMAL,
    Action,
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    OriginDetail,
    Reason,
    ResonanceType,
    Status,
    Verdict,
    best_status,
    join_list,
)

logger = logging.getLogger(__name__)

NAN = float('nan')
TWO_PI = 2.0 * math.pi

# the veto that moved a bounded angle to non_resonant (column `veto`); F6/B6b forced waves act at geometry level (they
# remove their runs before the tree); VETO_FORCED_WAVE is the W31 background test (`_without_background_waves`)
VETO_OWNER = 'owner'
VETO_KINEMATIC = 'kinematic'
VETO_FORCED_WAVE = 'forced_wave'
FORCED_WAVE_TERM = 'background_wave'  # `cause_term` of a forced-wave veto: the steady wave of the background, unnamed
VETO_STRENGTH = 'strength'
VETO_ZLK_KINEMATIC = 'zlk_kinematic'
VETO_ZLK_FREE_E = 'zlk_free_e'
VETO_ZLK_INTEGRAL = 'zlk_integral'  # W30 D1: the e-i exchange breaks sqrt(1-e^2) cos i where the forced terms are small
ZLK_VETOES = (VETO_ZLK_KINEMATIC, VETO_ZLK_FREE_E, VETO_ZLK_INTEGRAL)
# the e-i integral in the deciding episode (`EpisodeCoupling.coupling`): held (zlk), failed (measured, not zlk), or ''
EI_HELD, EI_FAILED = 'held', 'failed'

# V, the state of the vetoes the row needed (`level`): precedence ambiguous > not_measured > partial > clean
V_CLEAN, V_PARTIAL, V_NOT_MEASURED, V_AMBIGUOUS = 'clean', 'partial', 'not_measured', 'ambiguous'
V_ORDER = (V_CLEAN, V_PARTIAL, V_NOT_MEASURED, V_AMBIGUOUS)
# M, the amplitude margin, as `confidence_basis` writes it
M_FULL, M_THIN, M_NOT_MEASURED = 'full', 'thin', 'not_measured'
# F, whether an MMR row's own force is shown (W33): a pass of one of FORCE_CHECKS, or none
F_SHOWN, F_NONE = 'shown', 'none'
FORCE_CHECKS = ('check_mmr_a_response', 'check_mmr_potential', 'check_mmr_strength')
# how N was counted (`confidence_basis`); 'member': a subresonance parent's N is its member's (no episode of its own)
N_EPISODE, N_UNION, N_COVER, N_LEG, N_MEMBER = 'episode', 'union', 'cover', 'leg', 'member'
# `pol_extend_clause`: an end-edge 1:1 leg (A1), an end-edge half-swing (A2), an unfinished libration at the end (A:
# A3 insufficient_time, A4 the last counted episode reaches the end with fewer than CONFIDENCE_HIGH_CYCLES), the
# deciding period longer than 2T / LIBRATION_MIN_TURNING_POINTS (B), a needed veto not measured for lack of length that
# the next-check budget measures (C: the author, W29 Q5)
CLAUSE_LEG, CLAUSE_HALF, CLAUSE_END, CLAUSE_PERIOD, CLAUSE_VETO = 'A1', 'A2', 'A', 'B', 'C'
# the ids a 1:1 leg and a half-swing have in counted/candidate_episode_ids (they are no episode rows)
LEG_ID, HALF_ID = 'leg', 'half'

# The MMR leaves of `mmr.decide` (`rec_rule`), by what the policy does with them (spec §3.1). Every leaf decide.py can
# emit is a key of LEAVES (test_policy_mapping); one that is not maps to the geometry's status with policy_unknown_leaf.
KEEP, MEMBER, VETO, OBJECTION, SCALE, GEOMETRIC, DRIFT, LEG, HALF, INSUFFICIENT = (
    'keep',
    'member',
    'veto',
    'objection',
    'scale',
    'geometric',
    'drift',
    'leg',
    'half',
    'insufficient',
)
LEAVES = {
    '': KEEP,
    'G0': KEEP,
    'C5': KEEP,
    'C': KEEP,
    'R0': KEEP,
    '5c/6': KEEP,
    '5d/6': KEEP,
    '5c/6u': KEEP,
    '5d/6u': KEEP,
    '1': MEMBER,
    '2': VETO,
    '3': VETO,
    '3k2': VETO,
    'K3s': VETO,
    '5s': VETO,  # two-body by direct averaging or the harmonic; the three-body e^q route is SCALE (`_strength_veto`)
    '5s*': SCALE,
    '5a': OBJECTION,
    '5a*': OBJECTION,
    '5b': OBJECTION,
    '5e': OBJECTION,
    '5f': OBJECTION,
    '5g': OBJECTION,
    '5o': OBJECTION,
    '2j': GEOMETRIC,
    '5x': GEOMETRIC,
    '5n': GEOMETRIC,
    '5u/6': GEOMETRIC,
    '5u/6u': GEOMETRIC,
    '4': DRIFT,
    'G4c': LEG,
    'G4h': HALF,
    'G6': INSUFFICIENT,
    '3b5': INSUFFICIENT,
}
# the leaves on which the single 1:1 legs are read (`_legs`): G4c decides by them, rule C keeps a scattering one as a
# candidate
LEG_RULES = ('G4c', 'C')
# objection leaves that leave a near_separatrix angle as the tree read it (non_resonant there, level medium; spec §3.1)
NEAR_SEPARATRIX_KEPT = ('5a', '5a*', '5e', '5g')
LIBRATING_STATUSES = (Status.RESONANT, Status.TRANSIENT, Status.NEAR_SEPARATRIX)


@dataclass(frozen=True)
class PolicyResult:
    """What the minimal policy counted and drew for one row (`MMRAnalysis.policy`, `LongPeriodAnalysis.policy`; the
    figure and episodes.csv read it). Spans are [t0, t1] yr of a counted 1:1 leg or of candidate legs and half-swings
    (they are no episode rows)."""

    policy: str
    counted_episode_ids: frozenset = frozenset()
    candidate_episode_ids: frozenset = frozenset()
    counted_spans_yr: Tuple[Tuple[float, float], ...] = ()
    candidate_spans_yr: Tuple[Tuple[float, float], ...] = ()
    n_cycles: float = NAN
    margin: bool = False
    veto_state: str = ''
    stable: bool = False
    extend: bool = False
    extend_clause: str = ''


@dataclass(frozen=True)
class Evidence:
    """What `level` reads: N complete cycles (and how they were counted), M the amplitude margin, V the veto state, S
    the filter stability, the veto and whether it is thin, an unresolved second capture, and the candidates and a
    clear circulation of a non_resonant row."""

    n_cycles: float = NAN
    n_basis: str = ''
    margin: str = M_NOT_MEASURED
    veto_state: str = V_NOT_MEASURED
    stable: bool = False
    veto: str = ''
    veto_thin: bool = False
    second_capture: bool = False
    candidate: bool = False
    circulation_clear: bool = False
    core_cycles: float = NAN  # (n_TP - 1) / 2 observed inside the decisive episode (`core_cycles`): certain reads it
    force: str = ''  # MMR: F_SHOWN when L4, R0 or F passes, F_NONE when none does (`mmr_force_shown`); '' not read

    def basis(self) -> str:
        """The `confidence_basis` column: N=<float>(<basis>);M=..;V=..;S=..;veto=<name|->[:thin][;F=shown|none]."""
        n = f'{self.n_cycles:.3g}' if math.isfinite(self.n_cycles) else 'nan'
        veto = (self.veto or '-') + (':thin' if self.veto_thin else '')
        stable = M_FULL if self.stable else M_THIN
        force = f';F={self.force}' if self.force else ''
        return f'N={n}({self.n_basis or "-"});M={self.margin};V={self.veto_state};S={stable};veto={veto}{force}'


def level(status: Status | str, ev: Evidence, settings: AnalysisSettings) -> Confidence:
    """The level of a row from its evidence (PRINCIPLES §3; pure). uncertain and invalid have no level.

    resonant, transient:  V ambiguous or N < CONFIDENCE_MEDIUM_CYCLES -> low; N >= CONFIDENCE_CERTAIN_CYCLES and the
                          decisive episode's own cycles >= CONFIDENCE_CERTAIN_CORE_CYCLES, with M full, S and V clean ->
                          certain; N >= CONFIDENCE_HIGH_CYCLES with V clean or partial -> high; otherwise medium
                          (too few cycles, or a needed veto not measured). The same counts for every resonance type.
                          An MMR row whose own force is not shown (F none) is at most CONFIDENCE_GEOMETRY_ONLY_MAX.
    near_separatrix:      medium, low when V is ambiguous
    non_resonant by veto: medium when the veto is thin or a second capture is unresolved; else high (a measured cause:
                          the author, W30 285637; a stable vetoed episode is still drawn as a candidate)
    non_resonant:         a candidate -> medium; a clear circulation (MMR: C1 and C2 full, an alias C5 and C2 full: C5
                          measures a far from the resonance; long-period: P full and no turning point at any cutoff)
                          -> certain; anything else -> high
    For fixed (M, S, V) the level never falls as N rises."""
    status = Status(status)
    if status in (Status.UNCERTAIN, Status.INVALID):
        return Confidence.NONE
    if status == Status.NEAR_SEPARATRIX:
        return Confidence.LOW if ev.veto_state == V_AMBIGUOUS else Confidence.MEDIUM
    if status == Status.NON_RESONANT:
        if ev.veto:
            return Confidence.MEDIUM if (ev.veto_thin or ev.second_capture) else Confidence.HIGH
        if ev.candidate:
            return Confidence.MEDIUM
        return Confidence.CERTAIN if ev.circulation_clear else Confidence.HIGH
    n = ev.n_cycles if math.isfinite(ev.n_cycles) else 0.0
    if ev.veto_state == V_AMBIGUOUS or n < settings.confidence_medium_cycles:
        return Confidence.LOW
    core = ev.core_cycles if math.isfinite(ev.core_cycles) else 0.0
    full = ev.margin == M_FULL and ev.stable and ev.veto_state == V_CLEAN
    if n >= settings.confidence_certain_cycles and core >= settings.confidence_certain_core_cycles and full:
        found = Confidence.CERTAIN
    elif n >= settings.confidence_high_cycles and ev.veto_state in (V_CLEAN, V_PARTIAL):
        found = Confidence.HIGH
    else:
        found = Confidence.MEDIUM
    if ev.force == F_NONE:
        return max(found, Confidence(settings.confidence_geometry_only_max), key=CONFIDENCE_ORDER.index)  # the lower
    return found


def worst_veto_state(states: Iterable[str]) -> str:
    """The state by precedence ambiguous > not_measured > partial > clean (clean for none)."""
    return max(list(states) or [V_CLEAN], key=V_ORDER.index)


# --------------------------------------------------------------------------------------------- extend (P5)


@dataclass(frozen=True)
class EndReading:
    """The §5 quantities of one row (columns pol_*): the last turning point of an accepted or candidate run in the
    window, sigma_f's travel from it to the window's end in own revolutions, the deciding period, the cycles of the last
    counted episode and whether it reaches the record's end."""

    t_last_yr: float = NAN
    end_travel_rev: float = NAN
    period_dec_yr: float = NAN
    n_last: float = NAN
    last_reaches_end: Optional[bool] = None

    @property
    def ends_in_circulation(self) -> bool:
        """sigma_f travels at least one own revolution after the last turning point (nan: not measured, bounded)."""
        return bool(math.isfinite(self.end_travel_rev) and self.end_travel_rev >= 1.0)


def extend_clause(
    status: Status | str,
    end: EndReading,
    span_yr: float,
    settings: AnalysisSettings,
    *,
    leg_end: bool = False,
    half_end: bool = False,
    insufficient_time: bool = False,
    has_turning_points: bool = True,
    blocked: bool = False,
    veto_need_yr: float = NAN,
    ledger: Optional[Ledger] = None,
) -> str:
    """The clause of the extend predicate that holds ('' none; spec §5, PRINCIPLES §5):

        status != invalid and the record's end is bounded (no own revolution after the last turning point) and
        A1 an end-edge 1:1 leg with outcome extend (`leg_end`), or A2 an end-edge half-swing holding its law
        (`half_end`), or A3 the angle is insufficient_time and no counted episode reaches the end, or A4 the last
        counted episode reaches the end with N_last < CONFIDENCE_HIGH_CYCLES, or B the deciding period exceeds
        2 T / LIBRATION_MIN_TURNING_POINTS (no turning point at all counts as B on an uncertain row), or C a needed
        veto was not measured for lack of length (flag veto_needs_length) and the record it needs (`veto_need_yr`)
        is within the next-check budget MMR_NEXT_CHECK_SPAN_FACTOR x T (`veto_within_budget`).

    `blocked` (secular phase_unresolved: a longer record does not resolve the phases) gives ''."""
    if Status(status) == Status.INVALID or blocked or end.ends_in_circulation:
        return ''
    if leg_end:
        return CLAUSE_LEG
    if half_end:
        return CLAUSE_HALF
    if insufficient_time and not end.last_reaches_end:
        return CLAUSE_END
    if end.last_reaches_end and math.isfinite(end.n_last):
        high = end.n_last < settings.confidence_high_cycles
        if ledger is not None:
            ledger.compare_setting(
                'policy:A4 N of the last counted episode at the end',
                end.n_last,
                Op.LT,
                settings,
                'confidence_high_cycles',
                '',
                'pol_n_last',
            )
        if high:
            return CLAUSE_END
    if _period_clause(status, end, span_yr, settings, has_turning_points, ledger):
        return CLAUSE_PERIOD
    return CLAUSE_VETO if veto_within_budget(veto_need_yr, span_yr, settings, ledger) else ''


def _period_clause(
    status: Status | str, end: EndReading, span_yr: float, settings: AnalysisSettings, has_turning_points: bool, ledger: Optional[Ledger]
) -> bool:
    """Clause B: P_dec > 2 T / LIBRATION_MIN_TURNING_POINTS; no turning point at all (an arc, a stall) counts on an
    uncertain row (the period exceeds the record; only for a row that is not decided)."""
    bound = 2.0 * span_yr / settings.libration_min_turning_points
    if ledger is not None:
        ledger.compare('policy:B P_dec > 2 T / libration_min_turning_points', end.period_dec_yr, Op.GT, bound, 'yr', 'pol_period_dec_yr')
    if math.isfinite(end.period_dec_yr):
        return bool(end.period_dec_yr > bound)
    return bool(not has_turning_points and Status(status) == Status.UNCERTAIN)


def veto_budget_yr(span_yr: float, settings: AnalysisSettings) -> float:
    """The longest record a next check asks for, MMR_NEXT_CHECK_SPAN_FACTOR x T [yr] (`mmr.decide.next_check`)."""
    return settings.mmr_next_check_span_factor * span_yr


def veto_within_budget(veto_need_yr: float, span_yr: float, settings: AnalysisSettings, ledger: Optional[Ledger] = None) -> bool:
    """Clause C: the record a needed but unmeasured veto needs (`pol_veto_need_yr`; nan: none needed or no length
    named, inf: no length within ZLK_EXTEND_MAX_YR) is <= the next-check budget."""
    if not math.isfinite(veto_need_yr):
        return False
    if ledger is not None:
        entry = ledger.compare(
            'policy:C veto need <= mmr_next_check_span_factor x T',
            veto_need_yr,
            Op.LE,
            veto_budget_yr(span_yr, settings),
            'yr',
            'pol_veto_need_yr',
        )
        return bool(entry)
    return bool(veto_need_yr <= veto_budget_yr(span_yr, settings))


def veto_beyond_budget(veto_need_yr: float, span_yr: float, settings: AnalysisSettings, end: EndReading) -> bool:
    """A needed veto no next check measures: a length was named (finite or inf) above the budget and the record's
    end is bounded (a record that ends in circulation needs no veto). Such a row goes to review."""
    named = not math.isnan(veto_need_yr)
    return bool(named and not end.ends_in_circulation and not veto_within_budget(veto_need_yr, span_yr, settings))


# --------------------------------------------------------------------------------------------- counting (P3, P6)


def episode_cycles(episode: Episode, settings: AnalysisSettings, entry_seen: bool = False, exit_seen: bool = False) -> float:
    """N of one episode: (n_TP - 1) / 2 complete cycles (a 2-TP run its phase return closed: 1), plus the gap in
    periods of each covered edge (gap <= LIBRATION_COVERAGE_PERIODS and no observed entry/exit on that side: the
    libration runs on beyond the turning points the filter's trim hides)."""
    n = 1.0 if episode.n_tp == 2 and episode.closed_by_return else (episode.n_tp - 1) / 2.0
    cover = settings.libration_coverage_periods
    for gap, seen in (
        (episode.coverage_start_gap_p, entry_seen or episode.entry_observed),
        (episode.coverage_end_gap_p, exit_seen or episode.exit_observed),
    ):
        if math.isfinite(gap) and 0 <= gap <= cover and not seen:
            n += float(gap)
    return float(max(n, 0.0))


def _union_cycles(episodes: Sequence[Episode], settings: AnalysisSettings, **seen) -> float:
    """N of a union of counted episodes: each episode's own cycles, an overlap counted once (the later episode starts
    where the earlier ends)."""
    total, end = 0.0, -math.inf
    for episode in sorted(episodes, key=lambda e: e.t_start_yr):
        cycles = episode_cycles(episode, settings, **seen)
        if episode.t_start_yr < end and np.isfinite(episode.period_med_yr) and episode.period_med_yr > 0:
            cycles = max(cycles - (end - episode.t_start_yr) / episode.period_med_yr, 0.0)
        total += cycles
        end = max(end, episode.t_end_yr)
    return float(total)


def _spans_text(ids: Iterable, spans: Sequence, tag: str) -> str:
    return join_list([str(i) for i in sorted(ids)] + [tag] * len(spans))


# --------------------------------------------------------------------------------------------- the row's outcome


@dataclass
class _Outcome:
    """The status the policy gives one row before its level and action."""

    status: Status
    reason: Reason | str
    origin: LibrationOrigin | str
    origin_detail: str = ''
    angle_status: Optional[AngleStatus] = None  # None: the v2 word stays
    flags: List[str] = field(default_factory=list)
    veto: str = ''
    veto_thin: bool = False
    second_capture: bool = False
    veto_state: Optional[str] = None  # forced V (ambiguous for an objection, not_measured for a gate without verdict)
    counted: frozenset = frozenset()
    candidates: frozenset = frozenset()
    counted_spans: Tuple[Tuple[float, float], ...] = ()
    candidate_spans: Tuple[Tuple[float, float], ...] = ()
    n_cycles: Optional[float] = None  # None: count the counted episodes
    n_basis: str = ''
    level: Optional[Confidence] = None  # a level the policy table fixes (subresonance member, 5a on near_separatrix)
    period_dec_yr: Optional[float] = None  # P_dec override (a drift < 1 rev: span / pass)
    check_zlk_free_omega: Optional[CheckValue] = None


def _keep(v: Verdict, **fields) -> _Outcome:
    return _Outcome(v.status, v.reason, v.libration_origin, str(v.origin_detail), **fields)


def _geometric_status(word: AngleStatus | str, v: Verdict, keep_reason: bool) -> Tuple[Status, Reason | str, LibrationOrigin | str]:
    """(status, reason, origin) of a bounded word read by its geometry alone: libration -> resonant, episodic ->
    transient, near_separatrix -> near_separatrix; the v2 reason when it already claims that status (`keep_reason`),
    else the geometric reason (origin unknown, unless the tree named another origin: joint)."""
    status = {AngleStatus.LIBRATION: Status.RESONANT, AngleStatus.EPISODIC: Status.TRANSIENT}.get(word, Status.NEAR_SEPARATRIX)
    if keep_reason and v.status == status:
        return status, v.reason, v.libration_origin
    reason = {Status.RESONANT: Reason.GEOMETRIC_LIBRATION, Status.TRANSIENT: Reason.GEOMETRIC_EPISODE}.get(status, Reason.SEPARATRIX_ZONE)
    origin = v.libration_origin if v.libration_origin in (LibrationOrigin.JOINT, LibrationOrigin.DYNAMIC) else LibrationOrigin.UNKNOWN
    return status, reason, origin


def _write(
    v: Verdict,
    outcome: _Outcome,
    ev: Evidence,
    confidence: Confidence,
    action: Action,
    end: EndReading,
    clause: str,
    extend_to: Optional[float],
    resample: Optional[float],
    veto_need: float,
    ledger: Ledger,
) -> None:
    """Write the policy's verdict over the tree's (the tree's own in `v2_verdict`), every cap into the flags."""
    v.v2_verdict = f'{v.status}/{v.confidence}/{v.reason}/{v.action}/{v.angle_status}'
    flags = list(v.flags) + list(v.caps) + list(outcome.flags)
    if outcome.veto_thin:
        flags.append(str(Flag.VETO_THIN))
    v.flags = list(dict.fromkeys(str(flag) for flag in flags))
    v.caps, v.failed_support = [], []
    v.cell_n_examples = v.cell_n_errors = v.cell_accuracy_low_95 = v.cell_accuracy_high_95 = None
    v.status, v.reason, v.libration_origin = Status(outcome.status), outcome.reason, outcome.origin
    v.origin_detail = outcome.origin_detail
    if outcome.angle_status is not None:
        v.angle_status = outcome.angle_status
    v.confidence = v.confidence_raw = confidence
    v.action, v.resample_step_yr = action, resample
    if action == Action.EXTEND:
        v.extend_to_years = extend_to
    elif str(Flag.EXTEND_TIME) not in v.flags:
        v.extend_to_years = None
    if outcome.check_zlk_free_omega is not None:
        v.checks['check_zlk_free_omega'] = outcome.check_zlk_free_omega
    v.policy = POLICY_MINIMAL
    v.n_cycles = ev.n_cycles
    v.n_counted_episodes = len(outcome.counted) + len(outcome.counted_spans)
    v.counted_episode_ids = _spans_text(outcome.counted, outcome.counted_spans, LEG_ID)
    v.candidate_episode_ids = _spans_text(outcome.candidates, outcome.candidate_spans, _span_tag(outcome))
    v.veto = outcome.veto
    v.confidence_basis = ev.basis()
    v.pol_end_travel_rev, v.pol_t_last_tp_yr, v.pol_period_dec_yr = end.end_travel_rev, end.t_last_yr, end.period_dec_yr
    v.pol_n_last, v.pol_last_reaches_end, v.pol_extend_clause = end.n_last, end.last_reaches_end, clause
    v.pol_veto_need_yr = veto_need
    v.pol_core_cycles = ev.core_cycles
    v.decision_path = list(v.decision_path) + list(ledger)


def _span_tag(outcome: _Outcome) -> str:
    return HALF_ID if str(Flag.ONE_HALF_LIBRATION) in outcome.flags or outcome.reason == Reason.ONE_HALF_LIBRATION else LEG_ID


def core_cycles(geometry: Optional[Geometry], decisive_id) -> float:
    """(n_TP - 1) / 2 cycles observed inside the decisive episode (a 2-TP run closed by its phase return: 1), no edge
    coverage added; nan without one."""
    run = run_of(geometry, decisive_id) if geometry is not None else None
    if run is None:
        return NAN
    return 1.0 if run.n_tp == 2 and run.closed_by_return else max((run.n_tp - 1) / 2.0, 0.0)


def _record_level(ev: Evidence, settings: AnalysisSettings, ledger: Ledger) -> None:
    """The level's comparisons on the decision path (N against the cycle counts)."""
    n = ev.n_cycles if math.isfinite(ev.n_cycles) else None
    ledger.compare_setting('policy:N >= confidence_medium_cycles', n, Op.GE, settings, 'confidence_medium_cycles', 'cycles', 'n_cycles')
    ledger.compare_setting('policy:N >= confidence_high_cycles', n, Op.GE, settings, 'confidence_high_cycles', 'cycles', 'n_cycles')
    ledger.compare_setting('policy:N >= confidence_certain_cycles', n, Op.GE, settings, 'confidence_certain_cycles', 'cycles', 'n_cycles')
    core = ev.core_cycles if math.isfinite(ev.core_cycles) else None
    ledger.compare_setting(
        'policy:decisive episode cycles >= confidence_certain_core_cycles',
        core,
        Op.GE,
        settings,
        'confidence_certain_core_cycles',
        'cycles',
        'pol_core_cycles',
    )


def _action(
    status: Status, confidence: Confidence, clause: str, resample: Optional[float], v2_action: Action, veto_beyond: bool = False
) -> Action:
    """Action order (spec §5): invalid keeps the tree's; extend; a needed check R-ALIAS-A withdrew with a fold-free step
    -> resample; uncertain, or a needed veto no next check measures (`veto_beyond_budget`) -> review; else by level
    (certain/high none, medium spot_check, low review)."""
    if status == Status.INVALID:
        return v2_action
    if clause:
        return Action.EXTEND
    if resample is not None:
        return Action.RESAMPLE
    if status == Status.UNCERTAIN or veto_beyond:
        return Action.REVIEW
    return {Confidence.MEDIUM: Action.SPOT_CHECK, Confidence.LOW: Action.REVIEW}.get(confidence, Action.NONE)


def _thin_entries(entries: Iterable[LedgerEntry], prefixes: Tuple[str, ...]) -> bool:
    """A comparison of the tree whose name starts with one of `prefixes` lies within the thin band of its threshold."""
    return any(entry.margin == Margin.THIN and entry.result is not None and entry.name.startswith(prefixes) for entry in entries)


# --------------------------------------------------------------------------------------------- MMR


def apply_mmr(analysis, settings: AnalysisSettings):
    """The policy on a finished MMR analysis (`mmr.analyse.analyse_mmr`, outermost call only): every derived row first,
    then the catalogue row (a subresonance parent takes its member's minimal status, level and N), then the multiplet
    status again. Returns the analysis (changed in place)."""
    for row in analysis.derived:
        _apply_mmr_row(row, settings, None)
    member = analysis.derived[0].verdict if analysis.derived else None
    _apply_mmr_row(analysis, settings, member)
    if analysis.derived:
        status = best_status([analysis.verdict.status] + [d.verdict.status for d in analysis.derived])
        for row in [analysis] + analysis.derived:
            row.multiplet_status = status
            row.verdict.multiplet_status = status
    return analysis


def mmr_librating(geometry: Optional[Geometry], settings: AnalysisSettings) -> List[Episode]:
    """The episodes the status counts: the librating episodes of the last regime and of the regimes before the last
    jump of a (G1e/G1l read those); forced waves and induced episodes excluded (`mmr.decide.librating_episodes`)."""
    if geometry is None:
        return []
    return mmr_decide.librating_episodes(geometry, settings) + mmr_decide.librating_episodes(geometry, settings, before_jump=True)


def _mmr_t_last(geometry: Geometry, candidate_spans: Sequence[Tuple[float, float]], near_runs: Sequence[Episode]) -> float:
    """The last turning point of an accepted or candidate run inside the window (the window's start without one)."""
    lo, hi = geometry.window
    times = [run.t_end_yr for run in geometry.runs if run.accepted and run.n_tp]
    times += [t1 for _, t1 in candidate_spans] + [run.t_end_yr for run in near_runs]
    inside = [t for t in times if lo <= t <= hi]
    return float(max(inside)) if inside else float(lo)


def _near_runs(geometry: Geometry, analysis, settings: AnalysisSettings) -> List[Episode]:
    """The near-episode run(s) (`mmr.decide.near_episode_tp` > 0): the short runs with that many turning points."""
    count = analysis.extra_columns.get('geo_near_episode_n_tp')
    count = int(count) if count is not None and np.isfinite(count) else 0
    if count <= 0:
        return []
    return [run for run in short_runs(geometry.runs, coorbital=geometry.coorbital) if run.n_tp == count]


def _legs(geometry: Geometry, settings: AnalysisSettings):
    """(legs with their outcome, the deciding interior leg and its outcome or None, an end-edge extend leg exists, a
    start-edge extend leg exists) of a 1:1 angle (`mmr.decide.leg_outcome`)."""
    if geometry is None or not geometry.coorbital or not settings.mmr_coorbital_single_leg:
        return [], None, False, False
    read = [(leg, mmr_decide.leg_outcome(leg, settings)[0]) for leg in geometry.single_legs]
    interior = [(leg, outcome) for leg, outcome in read if not leg.edge and outcome]
    best = max(interior, key=lambda item: (mmr_decide.LEG_RANK[item[1]], item[0].t1_yr - item[0].t0_yr), default=None)
    end = any(leg.edge == LEG_EDGE_END and outcome == mmr_decide.LEG_EXTEND for leg, outcome in read)
    start = any(leg.edge == LEG_EDGE_START and outcome == mmr_decide.LEG_EXTEND for leg, outcome in read)
    return read, best, end, start


def _held_halves(geometry: Optional[Geometry], settings: AnalysisSettings) -> list:
    if geometry is None or geometry.coorbital:
        return []
    return [half for half in geometry.half_librations if mmr_decide.half_libration_holds(half, settings)]


def _mmr_leg_outcome(read, best, end_leg: bool, start_leg: bool) -> Optional[_Outcome]:
    """§3.2 G4c: the status from the strongest interior leg; an end-edge extend leg alone -> uncertain (extend A1); a
    start-edge one alone -> non_resonant / circulation (no extend). The other legs are candidates."""
    spans = [(leg.t0_yr, leg.t1_yr) for leg, outcome in read if outcome]
    if best is not None:
        leg, outcome = best
        span = (leg.t0_yr, leg.t1_yr)
        others = tuple(s for s in spans if s != span)
        if outcome == mmr_decide.LEG_TRANSIENT:
            return _Outcome(
                Status.TRANSIENT,
                Reason.DYNAMIC_EPISODE,
                LibrationOrigin.DYNAMIC,
                angle_status=AngleStatus.EPISODIC,
                flags=[str(Flag.ONE_COORBITAL_LEG)],
                counted_spans=(span,),
                candidate_spans=others,
                n_cycles=0.5,
                n_basis=N_LEG,
            )
        if outcome == mmr_decide.LEG_UNCERTAIN:
            return _Outcome(
                Status.UNCERTAIN,
                Reason.ONE_COORBITAL_LEG,
                LibrationOrigin.UNKNOWN,
                angle_status=AngleStatus.EPISODIC,
                candidate_spans=tuple(spans),
            )
        return _Outcome(
            Status.NON_RESONANT,
            Reason.CIRCULATION,
            LibrationOrigin.NONE,
            angle_status=AngleStatus.CIRCULATION,
            flags=[str(Flag.NEAR_EPISODE)],
            candidate_spans=tuple(spans),
        )
    if end_leg:
        return _Outcome(
            Status.UNCERTAIN,
            Reason.BASELINE_NEEDED,
            LibrationOrigin.NONE,
            angle_status=AngleStatus.INSUFFICIENT_TIME,
            candidate_spans=tuple(spans),
        )
    if start_leg:
        flags = [str(Flag.ONE_COORBITAL_LEG), str(Flag.EXTEND_TIME), str(Flag.REJECTED_CANDIDATE)]
        return _Outcome(
            Status.NON_RESONANT,
            Reason.CIRCULATION,
            LibrationOrigin.NONE,
            angle_status=AngleStatus.CIRCULATION,
            flags=flags,
            candidate_spans=tuple(spans),
        )
    return None


def _mmr_half_outcome(geometry: Geometry, held: list, settings: AnalysisSettings) -> Optional[_Outcome]:
    """§3.2 G4h: an end-edge half-swing -> uncertain / baseline_needed (extend A2); between passages or at the start ->
    non_resonant / circulation, its half-swing a candidate (its law is only 'not failed', its turns no measured
    reflections)."""
    half = mmr_decide.deciding_half_libration(geometry, settings)
    if half is None:
        return None
    spans = tuple((h.t0_yr, h.t1_yr) for h in held)
    if half.edge == LEG_EDGE_END:
        return _Outcome(
            Status.UNCERTAIN,
            Reason.BASELINE_NEEDED,
            LibrationOrigin.NONE,
            angle_status=AngleStatus.INSUFFICIENT_TIME,
            candidate_spans=spans,
        )
    flags = [str(Flag.ONE_HALF_LIBRATION), str(Flag.REJECTED_CANDIDATE)]
    return _Outcome(
        Status.NON_RESONANT,
        Reason.CIRCULATION,
        LibrationOrigin.NONE,
        angle_status=AngleStatus.CIRCULATION,
        flags=flags,
        candidate_spans=spans,
    )


def _mmr_veto(analysis, rule: str, settings: AnalysisSettings, ledger: Ledger) -> Optional[Tuple[str, bool]]:
    """(veto, thin) when the measured predicate of the leaf's veto holds (spec §4; the tree's own comparisons, read
    from the stored analysis and its ledger), None otherwise. A leaf whose predicate does not hold is no veto."""
    v = analysis.verdict
    path = v.decision_path
    if rule == '2':
        owner = analysis.owner
        held = owner is not None and owner.accepted is not None and v.libration_origin == LibrationOrigin.INDUCED
        ledger.compare('policy:veto owner accepted and inducing the episode', 1.0 if held else 0.0, Op.EQ, 1, '', 'own_name')
        w = v.margins.get('W')
        thin = _thin_entries(path, ('2 ',)) or (w is not None and w.margin == Margin.THIN)
        return (VETO_OWNER, thin) if held else None
    if rule in ('3', '3k2', 'K3s'):
        measured = (
            rule != '3'
            or v.checks.get('check_mmr_a_amplitude') == CheckValue.FAIL
            or v.checks.get('check_mmr_kinematic') == CheckValue.FAIL
            or any(entry.name.startswith('3 ') and entry.result is not None for entry in path)
        )
        held = measured and v.libration_origin == LibrationOrigin.KINEMATIC
        ledger.compare('policy:veto kinematic origin measured', 1.0 if held else 0.0, Op.EQ, 1, '', 'kin_corr_rate')
        return (VETO_KINEMATIC, _thin_entries(path, ('3 ', 'K3s'))) if held else None
    if rule == '5s':
        return _strength_veto(analysis, settings, ledger)
    return None


def _strength_veto(analysis, settings: AnalysisSettings, ledger: Ledger) -> Optional[Tuple[str, bool]]:
    """5s two-body (direct averaging or the harmonic): the period needs >= mmr_strength_implausible_ratio x the force
    computed from the disturbing function: veto strength. The three-body e^q route is no computed force (None)."""
    strength = analysis.strength
    if strength is None or strength.method not in DIRECT_METHODS:
        return None
    entry = ledger.compare_setting(
        'policy:veto strength: required / directly averaged',
        strength.ratio,
        Op.GE,
        settings,
        'mmr_strength_implausible_ratio',
        '',
        'str_ratio',
    )
    return (VETO_STRENGTH, entry.margin == Margin.THIN) if entry else None


def _stable_candidate(analysis, settings: AnalysisSettings, ledger: Ledger) -> bool:
    """The 138537 test: a vetoed MMR episode is drawn as a candidate only when the decisive episode keeps
    >= LIBRATION_MIN_TURNING_POINTS turning points at the cutoffs x0.5, x1 and x2 (per-cutoff counts required; without
    them the run is not stable: L1 falls back to the episode's own count)."""
    run = run_of(analysis.geometry, analysis.decisive_episode_id)
    half, double = analysis.extra_columns.get('ep_n_tp_cut_half'), analysis.extra_columns.get('ep_n_tp_cut_double')
    if run is None or half is None or double is None or not (np.isfinite(half) and np.isfinite(double)):
        ledger.compare('policy:candidate L1 counts at x0.5, x1, x2', None, Op.GE, settings.libration_min_turning_points, '', 'mrg_L1_value')
        return False
    least = min(float(half), float(run.n_tp), float(double))
    return bool(
        ledger.compare_setting(
            'policy:candidate L1 counts at x0.5, x1, x2', least, Op.GE, settings, 'libration_min_turning_points', '', 'mrg_L1_value'
        )
    )


@dataclass
class _Row:
    """One MMR row as the leaf handlers read it."""

    analysis: object
    rule: str
    settings: AnalysisSettings
    ledger: Ledger
    member: Optional[Verdict]
    end: EndReading
    legs: tuple
    held: list
    ids: frozenset
    word: AngleStatus

    @property
    def v(self) -> Verdict:
        return self.analysis.verdict


def _on_member(row: _Row) -> _Outcome:
    """Rule 1: the catalogue row takes its member's minimal status, level and N (a nodal member has no row of its own:
    the tree's verdict stays)."""
    member = row.member
    if member is None:
        return _keep(row.v, candidates=row.ids)
    status = Status.UNCERTAIN if member.status == Status.INVALID else Status(member.status)
    level_ = Confidence(member.confidence) if status not in (Status.UNCERTAIN, Status.INVALID) else Confidence.NONE
    outcome = _keep(row.v, candidates=row.ids, n_cycles=member.n_cycles, n_basis=N_MEMBER, level=level_)
    outcome.status = status
    return outcome


def _on_veto(row: _Row) -> _Outcome:
    """A leaf that moved the angle to non_resonant: a veto when its measured predicate holds; the three-body 5s and a
    leaf whose predicate does not hold are objections."""
    v = row.v
    veto = _mmr_veto(row.analysis, row.rule, row.settings, row.ledger)
    if veto is None:
        flag = Flag.STRENGTH_SCALE_IMPLAUSIBLE if row.rule == '5s' else Flag.MEASURED_OBJECTION
        return _bounded(v, row.word, row.end, row.ids, [flag], V_AMBIGUOUS)
    name, thin = veto
    ids = row.ids - _strength_marked(row.analysis.geometry)  # an upheld strength veto keeps its marks
    stable = _stable_candidate(row.analysis, row.settings, row.ledger) if ids else False
    second = str(Flag.SECOND_CAPTURE_UNRESOLVED) in v.flags or str(Flag.SECOND_CAPTURE_UNRESOLVED) in v.caps
    outcome = _keep(v, veto=name, veto_thin=thin, second_capture=second, candidates=ids if stable else frozenset())
    if name == VETO_STRENGTH and not stable:
        # the strength veto says no run is this angle's libration and none is drawn: the angle's word is what is left,
        # a circulation or a drift (the author, W30: 138537, 182259, 2897 'тут нет эпизодов')
        outcome.angle_status = _moving_word(row.analysis.geometry)
    return outcome


def _moving_word(geometry: Optional[Geometry]) -> AngleStatus:
    """circulation when the slow centre passes at least one revolution, else drift."""
    passed = geometry.centre.max_pass_rad / TWO_PI if geometry is not None and geometry.centre is not None else NAN
    return AngleStatus.CIRCULATION if math.isfinite(passed) and passed >= 1.0 else AngleStatus.DRIFT


def _on_objection(row: _Row) -> _Outcome:
    if row.word == AngleStatus.NEAR_SEPARATRIX and row.rule in NEAR_SEPARATRIX_KEPT:
        return _keep(row.v, level=Confidence.MEDIUM)
    return _bounded(row.v, row.word, row.end, row.ids, [Flag.MEASURED_OBJECTION], V_AMBIGUOUS)


def _on_leg(row: _Row) -> Optional[_Outcome]:
    # the legs decide only where v2 let them: leaf G4c, or rule C (a 1:1 circulation whose scattering leg is a
    # candidate). On another kept leaf (5c/6, 5d/6, ...) an accepted episode decides and a leg beside it changes
    # nothing (W29 integration: 59 co-orbital librators were read as their legs).
    if row.rule != 'G4c' and (row.rule != 'C' or not row.legs[0]):
        return None
    return _mmr_leg_outcome(*row.legs)


HANDLERS = {
    MEMBER: _on_member,
    VETO: _on_veto,
    SCALE: lambda row: _bounded(row.v, row.word, row.end, row.ids, [Flag.STRENGTH_SCALE_IMPLAUSIBLE], V_AMBIGUOUS),
    OBJECTION: _on_objection,
    GEOMETRIC: lambda row: _bounded(row.v, row.word, row.end, row.ids, [], None),
    DRIFT: lambda row: _drift(row.v, row.analysis.geometry, TWO_PI, _span(row.analysis)),
    LEG: _on_leg,
    KEEP: _on_leg,  # rule C of a 1:1 angle with legs: a scattering leg is a candidate (G4c)
    HALF: lambda row: _mmr_half_outcome(row.analysis.geometry, row.held, row.settings),
    INSUFFICIENT: lambda row: _insufficient(row.v, row.end, row.analysis.geometry),
}


def _span(analysis) -> float:
    return float(analysis.sampling.span_yr) if analysis.sampling is not None else NAN


def _mmr_outcome(
    analysis, rule: str, settings: AnalysisSettings, ledger: Ledger, member: Optional[Verdict], end: EndReading, legs, held
) -> _Outcome:
    """The minimal status of one MMR row from its leaf (`LEAVES`, `HANDLERS`) and its geometry (spec §3.1)."""
    v = analysis.verdict
    if v.status == Status.INVALID or analysis.geometry is None or rule in ('', 'G0', 'C5'):
        return _keep(v)
    ids = frozenset(e.episode_id for e in mmr_librating(analysis.geometry, settings))
    if rule in ('5s', '5s*'):
        # rule 5s marks the runs it judged as forced waves by strength after the verdict (`strength_forced_waves`);
        # where the policy does not uphold the strength veto (three-body scale, a predicate that does not hold) they
        # are the angle's episodes again (W29 integration: transient rows with no counted episode)
        ids |= _strength_marked(analysis.geometry)
    word = AngleStatus(v.angle_status) if v.angle_status in tuple(AngleStatus) else AngleStatus.UNDEFINED
    kind = LEAVES.get(rule)
    if kind is None:
        key = analysis.resonance.key if analysis.resonance is not None else ''
        logger.warning('policy: no mapping for the MMR leaf %r (%s %s): the geometry decides', rule, analysis.series_name, key)
        return _bounded(v, word, end, ids, [Flag.POLICY_UNKNOWN_LEAF], None)
    found = HANDLERS[kind](_Row(analysis, rule, settings, ledger, member, end, legs, held, ids, word))
    if found is None:
        # a leaf the table keeps (or a leg/half-swing leaf without a deciding one): the tree's status stays
        found = _keep(v, counted=ids or _decisive(analysis)) if v.status in LIBRATING_STATUSES else _keep(v, candidates=ids)
    return _without_background_waves(analysis, found, settings, ledger)


def _without_background_waves(analysis, outcome: _Outcome, settings: AnalysisSettings, ledger: Ledger) -> _Outcome:
    """W31 (the author, three B): the counted episodes that are the forced wave their background carries
    (`background_wave`, below 45 deg) leave the count as candidates; when they are all of them and the slow centre
    passes a revolution, veto forced_wave: non_resonant / kinematic_libration, detail forced_wave_background, word
    circulation; a centre that passes less is read by the drift rule (`_drift`: uncertain, W32 redundancy review)."""
    if Status(outcome.status) not in LIBRATING_STATUSES or outcome.veto or not outcome.counted or outcome.counted_spans:
        return outcome
    waves = {wave.episode_id: wave for wave in analysis.background_waves if wave.forced}
    forced = frozenset(outcome.counted) & frozenset(waves)
    ledger.compare(
        'policy:veto forced_wave: counted episodes the background carries',
        float(len(forced)),
        Op.GE,
        float(len(outcome.counted)),
        '',
        'bw_outcome',
    )
    if not forced:
        return outcome
    if forced == frozenset(outcome.counted):
        if _moving_word(analysis.geometry) == AngleStatus.DRIFT:
            # the slow centre passes less than a revolution: without the waves the drift rule decides (P1), not the veto
            drift = _drift(analysis.verdict, analysis.geometry, TWO_PI, _span(analysis))
            drift.angle_status, drift.candidates, drift.flags = AngleStatus.DRIFT, forced, list(outcome.flags)
            return drift
        thin = any(waves[eid].thin() for eid in forced)
        return _Outcome(
            Status.NON_RESONANT,
            Reason.KINEMATIC_LIBRATION,
            LibrationOrigin.KINEMATIC,
            str(OriginDetail.FORCED_WAVE_BACKGROUND),
            angle_status=_moving_word(analysis.geometry),
            veto=VETO_FORCED_WAVE,
            veto_thin=thin,
            candidates=forced,
            flags=list(outcome.flags),
        )
    outcome.counted = frozenset(outcome.counted) - forced
    outcome.candidates = frozenset(outcome.candidates) | forced
    return outcome


def _countable(geometry: Optional[Geometry], settings: AnalysisSettings) -> List[Episode]:
    """The Episode rows a counted id may name: `mmr_librating` and the runs rule 5s marked as forced waves by strength
    (counted again where the strength veto is not upheld, `_mmr_outcome`; without them N was nan on 2012 KL9)."""
    if geometry is None:
        return []
    marked = _strength_marked(geometry)
    return mmr_librating(geometry, settings) + [run for run in geometry.runs if run.episode_id in marked]


def _strength_marked(geometry: Geometry) -> frozenset:
    """The accepted runs rule 5s marked as forced waves by strength (`ForcedWave.by_strength`)."""
    marked = {wave.episode_id for wave in geometry.forced_waves if wave.by_strength}
    return frozenset(run.episode_id for run in geometry.runs if run.accepted and run.episode_id in marked)


def _decisive(analysis) -> frozenset:
    return frozenset([analysis.decisive_episode_id]) if analysis.decisive_episode_id is not None else frozenset()


def _bounded(v: Verdict, word: AngleStatus, end: EndReading, ids: frozenset, flags: Sequence, veto_state: Optional[str]) -> _Outcome:
    """The geometry's status of a leaf that may not move it (P2): a bounded word keeps its status, the others take
    their geometry row (circulation, drift, insufficient_time), with `flags` and the forced V."""
    flags = [str(flag) for flag in flags]
    if word in (AngleStatus.LIBRATION, AngleStatus.EPISODIC, AngleStatus.NEAR_SEPARATRIX):
        status, reason, origin = _geometric_status(word, v, keep_reason=True)
        return _Outcome(status, reason, origin, str(v.origin_detail), flags=flags, veto_state=veto_state, counted=ids)
    if word == AngleStatus.INSUFFICIENT_TIME:
        outcome = _insufficient(v, end, None)
    elif word == AngleStatus.DRIFT:
        outcome = _Outcome(Status.UNCERTAIN, Reason.DRIFT_MECHANISM_UNKNOWN, LibrationOrigin.UNKNOWN)
    elif word == AngleStatus.CIRCULATION:
        outcome = _Outcome(Status.NON_RESONANT, Reason.CIRCULATION, LibrationOrigin.NONE)
    else:
        outcome = _keep(v)
    outcome.flags += flags
    outcome.veto_state = veto_state
    outcome.candidates = ids
    return outcome


def _drift(v: Verdict, geometry: Geometry, revolution: float, span_yr: float) -> _Outcome:
    """P1 drift: the centre's largest monotone pass >= one own revolution -> non_resonant / modulated_circulation;
    less -> uncertain / drift_mechanism_unknown (MMR; a long-period row keeps baseline_needed), P_dec = span / pass."""
    passed = geometry.centre.max_pass_rad / revolution if geometry is not None else NAN
    if math.isfinite(passed) and passed >= 1.0:
        return _Outcome(Status.NON_RESONANT, Reason.MODULATED_CIRCULATION, LibrationOrigin.NONE)
    reason = Reason.DRIFT_MECHANISM_UNKNOWN if v.resonance_type == ResonanceType.MMR else v.reason
    period = span_yr / passed if math.isfinite(passed) and passed > 0 else NAN
    return _Outcome(Status.UNCERTAIN, reason, LibrationOrigin.UNKNOWN, period_dec_yr=period)


def _insufficient(v: Verdict, end: EndReading, geometry: Optional[Geometry]) -> _Outcome:
    """P1/P5 insufficient_time: non_resonant / circulation when the record ends in circulation (flag
    rejected_candidate when it had turning points), else uncertain / baseline_needed; a secular phase_unresolved angle
    is never non_resonant (P7)."""
    if end.ends_in_circulation and v.angle_subtype != AngleSubtype.PHASE_UNRESOLVED:
        had = geometry is not None and len(geometry.turning_points) > 0
        return _Outcome(Status.NON_RESONANT, Reason.CIRCULATION, LibrationOrigin.NONE, flags=[str(Flag.REJECTED_CANDIDATE)] if had else [])
    origin = v.libration_origin if v.status == Status.UNCERTAIN else LibrationOrigin.NONE
    return _Outcome(Status.UNCERTAIN, Reason.BASELINE_NEEDED, origin, angle_status=AngleStatus.INSUFFICIENT_TIME)


def _mmr_veto_state(analysis, outcome: _Outcome) -> str:
    """V of an MMR row (spec §6): ambiguous for a measured objection, an onset that failed or an identity the tree read
    as failed without a status move; not_measured when the identity was not measured or R-ALIAS-A withdrew it;
    partial when the identity passes but the owner search was not measured, a candidate owner is probable or W is
    thin; clean otherwise (the owner search measured with none accepted, or not applicable)."""
    v = analysis.verdict
    if outcome.veto_state is not None:
        return outcome.veto_state
    withdrawn = str(analysis.extra_columns.get('rsp_short_period_unmeasured') or '')
    identity = v.checks.get('check_mmr_identity', CheckValue.NOT_MEASURED)
    states = []
    if v.checks.get('check_mmr_onset') == CheckValue.FAIL or 'onset_failed' in v.caps:
        states.append(V_AMBIGUOUS)
    if 'identity' in withdrawn or identity not in (CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_APPLICABLE):
        states.append(V_NOT_MEASURED)
    elif identity == CheckValue.FAIL and not outcome.veto:
        # at e near 0 the step turns only varpi fast (flag varpi_fast_low_e): the identity's sigma_dot reads that
        # varpi, so its failure is expected and no objection (W31, 17346 2J-1: e 0.005-0.08)
        states.append(V_NOT_MEASURED if str(Flag.VARPI_FAST_LOW_E) in v.flags else V_AMBIGUOUS)
    owner = v.checks.get('check_mmr_owner', CheckValue.NOT_MEASURED)
    w = v.margins.get('W')
    if owner not in (CheckValue.PASS, CheckValue.NOT_APPLICABLE) or str(Flag.OWNER_PROBABLE) in v.flags:
        states.append(V_PARTIAL)
    if w is not None and w.margin == Margin.THIN:
        states.append(V_PARTIAL)
    return worst_veto_state(states)


def _mmr_margin(analysis) -> str:
    """M: L2 full; a 1:1 angle: its zone known and no cycle of the counted episodes in the separatrix zone (a
    horseshoe's half-amplitude near 180 deg is its topology, not a thin margin)."""
    geometry, energy = analysis.geometry, analysis.energy
    if geometry is not None and geometry.coorbital:
        if energy is None or energy.zone is None:
            return M_NOT_MEASURED
        return M_THIN if has_zone_cycle(energy.cycle_classes) else M_FULL
    l2 = analysis.verdict.margins.get('L2')
    if l2 is None:
        return M_NOT_MEASURED
    return {Margin.FULL: M_FULL, Margin.THIN: M_THIN}.get(l2.margin, M_NOT_MEASURED)


def _counted_cycles(episodes: List[Episode], decisive_id, status: Status, settings: AnalysisSettings) -> Tuple[float, str]:
    """(N, basis): a covering libration -> its episode ('cover'); a resonant row over several episodes (regime change,
    union) -> their sum ('union'); a transient -> the decisive episode ('episode', else the longest)."""
    if not episodes:
        return NAN, ''
    decisive = next((e for e in episodes if e.episode_id == decisive_id), None) or max(episodes, key=lambda e: e.n_tp)
    if status == Status.RESONANT:
        if all(covers_edges(decisive, settings)) or len(episodes) == 1:
            return episode_cycles(decisive, settings), N_COVER
        return _union_cycles(episodes, settings), N_UNION
    return episode_cycles(decisive, settings), N_EPISODE


def _apply_mmr_row(analysis, settings: AnalysisSettings, member: Optional[Verdict]) -> None:
    geometry, prep = analysis.geometry, analysis.prep
    rule = str(analysis.extra_columns.get('rec_rule', '') or '')
    ledger = Ledger.from_settings(settings)
    # the 1:1 legs are read where they decide (G4c; rule C: a scattering leg is a candidate). Beside an accepted
    # episode a leg is no evidence: an end-edge leg in the filter's trim of a libration reaching the end set clause A1
    # on resonant/certain rows (W29 integration: 624, 121514, 2016 NZ55)
    legs = _legs(geometry, settings) if rule in LEG_RULES else ([], None, False, False)
    held = _held_halves(geometry, settings)
    near = _near_runs(geometry, analysis, settings) if geometry is not None else []
    # the end of the record: the last turning point of an accepted or candidate run, sigma_f's travel after it
    end = EndReading()
    if geometry is not None:
        spans = [(leg.t0_yr, leg.t1_yr) for leg, outcome in legs[0] if outcome] + [(h.t0_yr, h.t1_yr) for h in held]
        t_last = _mmr_t_last(geometry, spans, near)
        travel = edge_travel_rev(prep.t_yr, prep.angle_f, t_last, geometry.window[1], geometry.window) if prep is not None else NAN
        ledger.compare('policy:end travel after the last turning point < 1 rev', travel, Op.LT, 1.0, 'rev', 'pol_end_travel_rev')
        end = EndReading(t_last, travel)
    outcome = _mmr_outcome(analysis, rule, settings, ledger, member, end, legs, held)
    _finish_mmr(analysis, outcome, settings, ledger, end, legs, held, near)


def _settle_mmr_episodes(analysis, outcome: _Outcome, settings: AnalysisSettings, near) -> List[Episode]:
    """The counted episodes (a librating status without a veto) and the candidates (the vetoed stable episodes, the
    near-episode runs of a row without libration); returns the counted Episode rows."""
    status = Status(outcome.status)
    episodes = [e for e in _countable(analysis.geometry, settings) if e.episode_id in outcome.counted]
    if status in LIBRATING_STATUSES and not outcome.veto:
        if not outcome.counted_spans:
            outcome.counted = frozenset(e.episode_id for e in episodes) or outcome.counted
    else:
        outcome.counted = frozenset()
        episodes = []
    if status not in LIBRATING_STATUSES and not outcome.veto:  # a vetoed row draws only its stable episodes (138537)
        outcome.candidates = outcome.candidates | frozenset(run.episode_id for run in near)
    outcome.candidates = outcome.candidates - outcome.counted
    return episodes


def _mmr_end(analysis, outcome: _Outcome, episodes: List[Episode], end: EndReading, settings, ledger: Ledger) -> EndReading:
    """The end reading completed: P_dec (the decisive run's median period, else P_cyc; a drift's span / pass) and the
    last counted episode's cycles and whether it reaches the record's end."""
    geometry = analysis.geometry
    decisive = run_of(geometry, analysis.decisive_episode_id)
    period = outcome.period_dec_yr
    if period is None and decisive is not None:
        period = float(decisive.period_med_yr)
    elif period is None:
        period = float(geometry.period_cycle_yr) if geometry is not None else NAN
    last = max(episodes, key=lambda e: e.t_end_yr, default=None)
    if last is None:
        return EndReading(end.t_last_yr, end.end_travel_rev, period, NAN, False)
    ledger.compare_setting(
        'policy:last counted episode end gap <= libration_coverage_periods',
        last.coverage_end_gap_p,
        Op.LE,
        settings,
        'libration_coverage_periods',
        'P',
        'ep_coverage_end_gap_P',
    )
    return EndReading(end.t_last_yr, end.end_travel_rev, period, episode_cycles(last, settings), bool(covers_edges(last, settings)[1]))


def _mmr_veto_need(analysis, outcome: _Outcome, settings: AnalysisSettings) -> float:
    """A needed check not measured for lack of length (B9: L4 with fewer than MMR_FORCE_MIN_WINDOWS windows): the
    flag veto_needs_length and the length the last episode at the end needs (nan otherwise)."""
    geometry = analysis.geometry
    if outcome.status not in LIBRATING_STATUSES or geometry is None or not mmr_decide.force_windows_short(analysis.response, settings):
        return NAN
    outcome.flags.append(str(Flag.VETO_NEEDS_LENGTH))
    at_end = mmr_decide.last_episode_at_end(geometry, settings)
    return mmr_decide.force_window_extension(at_end, settings) if at_end is not None else NAN


def _final_level(status: Status, outcome: _Outcome, ev: Evidence, settings: AnalysisSettings) -> Confidence:
    """`level`, or the level the table fixes (low instead of medium with V ambiguous); a thin veto at most medium."""
    confidence = outcome.level if outcome.level is not None else level(status, ev, settings)
    if outcome.level == Confidence.MEDIUM and ev.veto_state == V_AMBIGUOUS:
        confidence = Confidence.LOW
    if outcome.veto_thin and confidence in (Confidence.CERTAIN, Confidence.HIGH):
        confidence = Confidence.MEDIUM
    return confidence


def _finish_mmr(analysis, outcome: _Outcome, settings, ledger, end: EndReading, legs, held, near) -> None:
    v = analysis.verdict
    geometry = analysis.geometry
    status = Status(outcome.status)
    span = _span(analysis)
    episodes = _settle_mmr_episodes(analysis, outcome, settings, near)
    if outcome.n_cycles is not None:
        n, basis = outcome.n_cycles, outcome.n_basis
    elif status in LIBRATING_STATUSES and not outcome.veto:
        n, basis = _counted_cycles(episodes, analysis.decisive_episode_id, status, settings)
    else:
        n, basis = NAN, ''
    end = _mmr_end(analysis, outcome, episodes, end, settings, ledger)
    veto_need = _mmr_veto_need(analysis, outcome, settings)
    clause = extend_clause(
        status,
        end,
        span,
        settings,
        leg_end=legs[2],
        half_end=any(h.edge == LEG_EDGE_END for h in held),
        insufficient_time=(outcome.angle_status or v.angle_status) == AngleStatus.INSUFFICIENT_TIME and status == Status.UNCERTAIN,
        has_turning_points=geometry is not None and len(geometry.turning_points) > 0,
        veto_need_yr=veto_need,
        ledger=ledger,
    )
    ev = Evidence(
        n_cycles=float(n) if n is not None else NAN,
        n_basis=basis,
        margin=_mmr_margin(analysis),
        veto_state=_mmr_veto_state(analysis, outcome),
        stable=_mmr_stable(v),
        core_cycles=core_cycles(geometry, analysis.decisive_episode_id),
        veto=outcome.veto,
        veto_thin=outcome.veto_thin,
        second_capture=outcome.second_capture,
        candidate=bool(outcome.candidates or outcome.candidate_spans),
        circulation_clear=_mmr_circulation_clear(v),
        force=mmr_force_shown(v) if status in (Status.RESONANT, Status.TRANSIENT) else '',
    )
    _record_level(ev, settings, ledger)
    confidence = _final_level(status, outcome, ev, settings)
    resample = _alias_resample_step(analysis)
    action = _action(status, confidence, clause, resample, v.action, veto_beyond_budget(veto_need, span, settings, end))
    extend_to = _mmr_extend_to(analysis, end, span, settings, veto_need, clause) if action == Action.EXTEND else None
    if action == Action.RESAMPLE:
        resample = v.resample_step_yr if v.resample_step_yr is not None else resample
    else:
        resample = v.resample_step_yr if status == Status.INVALID else None
    moved = Status(outcome.status) != v.status
    _write(v, outcome, ev, confidence, action, end, clause, extend_to, resample, veto_need, ledger)
    if outcome.veto == VETO_FORCED_WAVE:  # the roles (A3a): a kinematic cause, the steady wave of the background
        v.cause_role, v.cause_term, v.owner = 'kinematic', FORCED_WAVE_TERM, ''
    if moved:
        v.orbit_type = _mmr_orbit_type(analysis, settings, outcome.counted)
    analysis.policy = _result(outcome, ev, action, clause)


def _mmr_orbit_type(analysis, settings: AnalysisSettings, counted: frozenset = frozenset()) -> str:
    """The header's orbit type read again for a status the policy moved (2012 KL9: 'episodes about 322° (not this
    resonance)' on a transient row)."""
    v = analysis.verdict
    coorbital_law = analysis.response.coorbital if analysis.response is not None else None
    not_this = orbit_type_module.not_this_resonance(v.status, v.libration_origin, v.reason, ())
    return orbit_type_module.mmr_orbit_type(
        analysis.resonance, analysis.geometry, analysis.decisive_episode_id, coorbital_law, settings, str(v.angle_status), not_this, counted
    )


def _result(outcome: _Outcome, ev: Evidence, action: Action, clause: str) -> PolicyResult:
    return PolicyResult(
        POLICY_MINIMAL,
        outcome.counted,
        outcome.candidates,
        tuple(outcome.counted_spans),
        tuple(outcome.candidate_spans),
        ev.n_cycles,
        ev.margin == M_FULL,
        ev.veto_state,
        ev.stable,
        action == Action.EXTEND,
        clause,
    )


def _alias_resample_step(analysis) -> Optional[float]:
    """The step to resample at when R-ALIAS-A withdrew a needed check (`rsp_short_period_unmeasured`): the sampling's
    resample step, else its fold-free step (where the short-period terms fold into nothing); None otherwise."""
    withdrawn = str(analysis.extra_columns.get('rsp_short_period_unmeasured') or '')
    sampling = analysis.sampling
    if not withdrawn or sampling is None:
        return None
    for step in (sampling.resample_step_yr, sampling.fold_free_step_yr):
        if step is not None and np.isfinite(step) and step > 0:
            return float(step)
    return None


def _margin_full(margin) -> bool:
    return margin is not None and margin.margin == Margin.FULL


def _mmr_stable(v: Verdict) -> bool:
    """S of an MMR row: the decisive episode keeps its turning points at the cutoffs x0.5, x1, x2 (L1 full) and no
    decisive number changes at the cutoff x2 (no filter_sensitive: 2002 XA 5J-2 was certain with it)."""
    return _margin_full(v.margins.get('L1')) and str(Flag.FILTER_SENSITIVE) not in v.flags


def mmr_force_shown(v: Verdict) -> str:
    """F of an MMR row: F_SHOWN when its own force is measured passing (L4 the force law on a, the 1:1 law included;
    R0 the energy and period law; F the strength), else F_NONE. The a<->sigma identity is no force: it holds for any
    orbit (W33, the author: no resonant/transient level above low by the geometry alone)."""
    return F_SHOWN if any(v.checks.get(name) == CheckValue.PASS for name in FORCE_CHECKS) else F_NONE


def _mmr_circulation_clear(v: Verdict) -> bool:
    """A clear MMR circulation: C2 full (no turning point at any cutoff) and the circulation measured, C1 full (the
    centre passes whole turns), or for an alias, where C1 is not defined, C5 full (a far from the resonance and the
    step's branch unique: the alias test measures the non-resonance; the author, W29 Q18)."""
    measured = v.margins.get('C5') if v.angle_subtype == AngleSubtype.ALIAS else v.margins.get('C1')
    return _margin_full(measured) and _margin_full(v.margins.get('C2'))


def _mmr_extend_to(
    analysis, end: EndReading, span: float, settings: AnalysisSettings, veto_need: float = NAN, clause: str = ''
) -> Optional[float]:
    """extend_to_years of an MMR row (spec §5): clause C the next check of the veto's need; otherwise the next check of
    span + max(HIGH - N_last, 1) x P_dec, at least the veto's need when one is named, and at least the next check of the
    baseline's estimate when that is a slow swing seen in part (`mmr.decide.SLOW_SWING_SOURCES`: its slow cycles, the
    author W30 epext B; 523683 2U-9 400 kyr); without a deciding period (an arc, a stall) the tree's baseline estimate
    (`mmr.decide.extend_years`)."""
    if clause == CLAUSE_VETO:
        return mmr_decide.next_check(veto_need, span, settings)
    n_last = end.n_last if math.isfinite(end.n_last) else 0.0
    if math.isfinite(end.period_dec_yr) and end.period_dec_yr > 0 and math.isfinite(span):
        estimate = span + max(settings.confidence_high_cycles - n_last, 1.0) * end.period_dec_yr
        if math.isfinite(veto_need):
            estimate = max(estimate, veto_need)
        base = analysis.baseline
        if base is not None and base.period_source in mmr_decide.SLOW_SWING_SOURCES and base.extend_to_years is not None:
            estimate = max(estimate, base.extend_to_years)
        return mmr_decide.next_check(estimate, span, settings)
    if analysis.baseline is not None and math.isfinite(span):
        return mmr_decide.extend_years(Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, analysis.baseline, span, settings)
    return mmr_decide.next_check(NAN, span, settings) if math.isfinite(span) else None


# --------------------------------------------------------------------------------------------- secular and ZLK


def apply_long_period(analysis, settings: AnalysisSettings):
    """The policy on a finished secular or ZLK analysis (`analyse_secular`, `analyse_zlk` after its columns).
    Returns the analysis (changed in place)."""
    v = analysis.verdict
    zlk = analysis.series.resonance_type == ResonanceType.LIDOV_KOZAI
    revolution = TWO_PI  # omega too (W30 D3): the half-turn of the quadrupole's cos 2 omega is no general symmetry
    geometry, angle = analysis.geometry, analysis.angle
    ledger = Ledger.from_settings(settings)
    span = float(analysis.series.span_yr)
    if v.status == Status.INVALID or geometry is None:
        outcome = _keep(v)
        end = EndReading()
    else:
        lo, hi = geometry.window
        times = [run.t_end_yr for run in geometry.runs if run.accepted and run.n_tp and lo <= run.t_end_yr <= hi]
        t_last = float(max(times)) if times else float(lo)
        travel = edge_travel_rev(angle.t_yr, angle.angle_f, t_last, hi, geometry.window, revolution) if angle is not None else NAN
        ledger.compare('policy:end travel after the last turning point < 1 own rev', travel, Op.LT, 1.0, 'rev', 'pol_end_travel_rev')
        end = EndReading(t_last, travel)
        outcome = _long_outcome(analysis, end, revolution, span, settings, ledger, zlk)
    _finish_long(analysis, outcome, settings, ledger, end, span, zlk)
    return analysis


def _long_geometric(analysis, end: EndReading, revolution: float, span: float) -> _Outcome:
    """P1 on a long-period angle (`secular.decide.decide_angle` after `phase_check`)."""
    v = analysis.verdict
    word = v.angle_status
    ids = frozenset(e.episode_id for e in secular_decide.librating_episodes(analysis.geometry))
    # the geometry alone names no mechanism (W30 D6): geometric, origin unknown; a confirmed ZLK gate makes it dynamic
    if word == AngleStatus.LIBRATION:
        return _Outcome(Status.RESONANT, Reason.GEOMETRIC_LIBRATION, LibrationOrigin.UNKNOWN, counted=ids)
    if word == AngleStatus.EPISODIC:
        return _Outcome(Status.TRANSIENT, Reason.GEOMETRIC_EPISODE, LibrationOrigin.UNKNOWN, counted=ids)
    if word == AngleStatus.CIRCULATION:
        return _Outcome(Status.NON_RESONANT, Reason.CIRCULATION, LibrationOrigin.NONE)
    if word == AngleStatus.DRIFT:
        outcome = _drift(v, analysis.geometry, revolution, span)
        outcome.reason = Reason.BASELINE_NEEDED if outcome.status == Status.UNCERTAIN else outcome.reason
        return outcome
    if word == AngleStatus.INSUFFICIENT_TIME:
        outcome = _insufficient(v, end, analysis.geometry)
        outcome.candidates = ids
        return outcome
    return _keep(v)


def _long_outcome(
    analysis, end: EndReading, revolution: float, span: float, settings: AnalysisSettings, ledger: Ledger, zlk: bool
) -> _Outcome:
    """The minimal status of a secular or ZLK row (spec §8, W30 D1-D6): the geometry's; for ZLK with the gate's
    outcome and the e-i integral of the deciding episode (`_ei_state`):

        confirmed gate        the geometry's status, dynamic; V clean (partial when G is thin), ambiguous when the
                              integral failed
        a measured kinematic  `_zlk_measured` (vetoes and objections)
        integral failed       veto zlk_integral where the forced terms are small (`_forced_small`), else an objection
        no verdict            the geometry's status; with the integral held and the forced terms small V clean and
                              flag kinematic_excluded (D2), else V not measured (<= medium)"""
    geometric = _long_geometric(analysis, end, revolution, span)
    if not zlk:
        return geometric
    if analysis.verdict.angle_status not in secular_decide.GATED:
        geometric.check_zlk_free_omega = CheckValue.NOT_APPLICABLE
        return geometric
    gate = analysis.gate
    outcome = gate.outcome if gate is not None else ''
    ei = _ei_state(gate, ledger)
    if outcome == secular_decide.GATE_DISABLED:
        geometric.flags.append(str(Flag.GATE_DISABLED))
        geometric.veto_state = V_NOT_MEASURED
        geometric.check_zlk_free_omega = CheckValue.NOT_MEASURED
        return geometric
    if outcome == secular_decide.GATE_CONFIRMED:
        return _zlk_confirmed(analysis, geometric, ei)
    measured = _zlk_measured(analysis, geometric, gate, settings, ledger, ei == EI_FAILED)
    if measured is not None:
        return measured
    small = _forced_small(gate, settings, ledger)
    if ei == EI_FAILED:
        if small:
            ledger.compare(
                'policy:veto zlk_integral: e-i integral broken where the forced terms are small',
                1.0,
                Op.EQ,
                1,
                '',
                'zlk_ei_episode_coupling',
            )
            ids = frozenset(e.episode_id for e in secular_decide.librating_episodes(analysis.geometry))
            veto = _zlk_veto(VETO_ZLK_INTEGRAL, _near(gate.episode_ei.eta_h, settings.zlk_ei_h_residual_max, settings), ids)
            veto.origin_detail = str(OriginDetail.EI_INTEGRAL_NOT_HELD)
            return veto
        outcome_ = _zlk_no_verdict(geometric)
        outcome_.flags.append(str(Flag.EI_INTEGRAL_NOT_HELD))
        outcome_.veto_state = V_AMBIGUOUS
        return outcome_
    outcome_ = _zlk_no_verdict(geometric)
    if ei == EI_HELD and small:
        outcome_.flags.append(str(Flag.KINEMATIC_EXCLUDED))
        outcome_.veto_state = V_CLEAN
    return outcome_


def _zlk_confirmed(analysis, geometric: _Outcome, ei: str) -> _Outcome:
    """A confirmed gate: the geometry's status, dynamic; V clean (partial when G is thin), ambiguous when the e-i
    integral failed (an objection, not a veto: the gate measured the free omega librating)."""
    g = analysis.verdict.margins.get('G')
    geometric.veto_state = V_PARTIAL if g is not None and g.margin == Margin.THIN else V_CLEAN
    geometric.check_zlk_free_omega = CheckValue.PASS
    if geometric.status in (Status.RESONANT, Status.TRANSIENT):
        geometric.reason = Reason.DYNAMIC_LIBRATION if geometric.status == Status.RESONANT else Reason.DYNAMIC_EPISODE
        geometric.origin = LibrationOrigin.DYNAMIC
    if ei == EI_FAILED:
        geometric.flags.append(str(Flag.EI_INTEGRAL_NOT_HELD))
        geometric.veto_state = V_AMBIGUOUS
    return geometric


def _ei_state(gate, ledger: Ledger) -> str:
    """The e-i integral in the deciding librating episode (W30 D1): held (`zlk`: antiphase at constant sqrt(1-e^2)
    cos i), failed (measured and not zlk), '' (not measured: too few periods, no episode, an orbit flip)."""
    ei = gate.episode_ei if gate is not None else None
    if ei is None or ei.coupling == 'not_measured':
        ledger.compare('policy:zlk e-i integral eta_H', None, Op.LE, NAN, '', 'zlk_ei_episode_eta_h', setting='zlk_ei_h_residual_max')
        return ''
    ledger.compare(
        'policy:zlk e-i integral eta_H', ei.eta_h, Op.LE, ei.h_residual_max, '', 'zlk_ei_episode_eta_h', setting='zlk_ei_h_residual_max'
    )
    return EI_HELD if ei.coupling == 'zlk' else EI_FAILED


def _forced_small(gate, settings: AnalysisSettings, ledger: Ledger) -> bool:
    """rho_max >= ZLK_RHO_KINEMATIC_EXCLUDED on a split that fits the forcing (not an unfitted co-orbital): the forced
    vector is too small to hold the perihelion (W30 D1/D2)."""
    if gate is None or gate.detail == secular_decide.DETAIL_COORBITAL_UNFITTED:
        return False
    entry = ledger.compare_setting(
        'policy:zlk forced terms small: rho_max', gate.rho_max, Op.GE, settings, 'zlk_rho_kinematic_excluded', '', 'zlk_gate_rho_max'
    )
    return bool(entry)


def _near(value: float, threshold: float, settings: AnalysisSettings) -> bool:
    return bool(
        np.isfinite(value) and np.isfinite(threshold) and threshold > 0 and abs(value / threshold - 1.0) < settings.confidence_thin_band
    )


def _zlk_measured(analysis, geometric: _Outcome, gate, settings: AnalysisSettings, ledger: Ledger, ei_failed: bool) -> Optional[_Outcome]:
    """A kinematic gate (a veto on a libration or where the free e stays below the forced e over the record, else an
    objection) or an undefined phase with the free circle off the origin; None for a gate without a verdict. The
    episode-median pin and the undefined phase veto only with the e-i integral failed too (W30 D4): a median and a
    missing phase are no pointwise measurement, otherwise they are objections."""
    word = analysis.verdict.angle_status
    ids = frozenset(e.episode_id for e in secular_decide.librating_episodes(analysis.geometry))
    if gate.outcome == secular_decide.GATE_KINEMATIC:
        if word == AngleStatus.LIBRATION:
            ledger.compare('policy:veto zlk_kinematic: gate cycles', gate.n_cycles, Op.GE, gate.min_cycles, 'cycles', 'zlk_gate_n_cycles')
            return _zlk_veto(VETO_ZLK_KINEMATIC, _gate_cycles_thin(gate, settings), ids)
        pinned = _free_e_pins(gate, word, settings, ledger)
        if pinned is not None and (pinned[0] or ei_failed):
            return _zlk_veto(VETO_ZLK_FREE_E, pinned[1], ids)
        return _zlk_objection(geometric)
    if gate.outcome == 'undefined_phase' and np.isfinite(gate.rho_max):
        if ledger.compare('policy:veto zlk_free_e: rho_max < 1', gate.rho_max, Op.LT, 1.0, '', 'zlk_gate_rho_max'):
            if ei_failed:
                return _zlk_veto(VETO_ZLK_FREE_E, _near_one(gate.rho_max, settings), ids)
            return _zlk_objection(geometric)
    return None


def _zlk_objection(geometric: _Outcome) -> _Outcome:
    geometric.flags.append(str(Flag.MEASURED_OBJECTION))
    geometric.veto_state = V_AMBIGUOUS
    geometric.check_zlk_free_omega = CheckValue.FAIL
    return geometric


def _zlk_no_verdict(geometric: _Outcome) -> _Outcome:
    """No gate verdict (unavailable, coorbital_unfitted_forcing, planetary_residual, short_window, undefined_phase,
    none): the geometry's status with the geometric reason, flag gate_unavailable, V not measured."""
    if geometric.status in (Status.RESONANT, Status.TRANSIENT):
        geometric.reason = Reason.GEOMETRIC_LIBRATION if geometric.status == Status.RESONANT else Reason.GEOMETRIC_EPISODE
        geometric.origin = LibrationOrigin.UNKNOWN
    geometric.flags.append(str(Flag.GATE_UNAVAILABLE))
    geometric.veto_state = V_NOT_MEASURED
    geometric.check_zlk_free_omega = CheckValue.NOT_MEASURED
    return geometric


def _near_one(value: float, settings: AnalysisSettings) -> bool:
    return bool(np.isfinite(value) and abs(value - 1.0) < settings.confidence_thin_band)


def _gate_cycles_thin(gate, settings: AnalysisSettings) -> bool:
    cycles = gate.n_cycles if gate.n_cycles is not None else NAN
    return bool(np.isfinite(cycles) and np.isfinite(gate.min_cycles) and cycles < gate.min_cycles * (1 + settings.confidence_thin_band))


def _free_e_pins(gate, word, settings: AnalysisSettings, ledger: Ledger) -> Optional[Tuple[bool, bool]]:
    """The zlk_free_e predicate of a kinematic gate on an episodic angle, a drift or too few cycles (spec §4): the free
    circle never around the origin (rho_max < 1, over the record), or inside every librating episode of an episodic
    angle (rho_episodes < 1 with the free omega turning >= ZLK_EPISODE_FREE_TURNS_PER_CYCLE per cycle). Returns
    (over the record, the deciding comparison is thin); None when the predicate does not hold."""
    if ledger.compare('policy:veto zlk_free_e: rho_max < 1', gate.rho_max, Op.LT, 1.0, '', 'zlk_gate_rho_max'):
        return True, _near_one(gate.rho_max, settings)
    if word != AngleStatus.EPISODIC:
        return None
    inside = ledger.compare('policy:veto zlk_free_e: rho_episodes < 1', gate.rho_episodes, Op.LT, 1.0, '', 'zlk_gate_rho_episodes')
    turns = ledger.compare(
        'policy:veto zlk_free_e: free omega turns per cycle',
        gate.free_turns_episodes,
        Op.GE,
        gate.free_turns_min,
        'turns/cycle',
        'zlk_gate_free_turns_episodes',
        setting='zlk_episode_free_turns_per_cycle',
    )
    if inside and turns:
        return False, _near_one(gate.rho_episodes, settings) or turns.margin == Margin.THIN
    return None


def _zlk_veto(name: str, thin: bool, ids: frozenset) -> _Outcome:
    """A ZLK veto: non_resonant / kinematic_libration (code -2); its episodes are candidates. check_zlk_free_omega
    fails only on the free omega's own vetoes (zlk_kinematic, zlk_free_e); the e-i integral (zlk_integral) is no
    free-omega measurement, so the gate's check stays not_measured there (W34: a short_window gate is no kinematic)."""
    free_omega = name != VETO_ZLK_INTEGRAL
    return _Outcome(
        Status.NON_RESONANT,
        Reason.KINEMATIC_LIBRATION,
        LibrationOrigin.KINEMATIC,
        str(OriginDetail.FREE_OMEGA_WINDS),
        veto=name,
        veto_thin=thin,
        candidates=ids,
        check_zlk_free_omega=CheckValue.FAIL if free_omega else CheckValue.NOT_MEASURED,
    )


def _long_veto_state(analysis, outcome: _Outcome) -> str:
    """V of a long-period row: the ZLK gate's (set by `_long_outcome`); secular: clean when the phase check was
    measured and passed or every mode was measured, not_measured otherwise."""
    if outcome.veto_state is not None:
        return outcome.veto_state
    if analysis.series.resonance_type == ResonanceType.LIDOV_KOZAI:
        # a measured veto, or no gate needed (ω circulates: 555527 read 'a needed objection not measured')
        needed = outcome.check_zlk_free_omega != CheckValue.NOT_APPLICABLE
        return V_CLEAN if outcome.veto or not needed else V_NOT_MEASURED
    delta = analysis.phase_uncertainty_rad
    if np.isnan(delta) and any(term.index is not None and term.coefficient for term in analysis.series.terms):
        return V_NOT_MEASURED  # a planetary mode the table lacks: the phase error is unknown, not zero (Codex W30)
    if not (np.isfinite(delta) and delta > 0):
        return V_CLEAN  # every mode of the angle measured (or none in it): nothing to stray
    return V_CLEAN if np.isfinite(analysis.phase_swing_rad) else V_NOT_MEASURED


def _long_margin(analysis, settings: AnalysisSettings) -> str:
    """M: the decisive episode's largest half-amplitude <= (1 - CONFIDENCE_THIN_BAND) x the run break."""
    run = run_of(analysis.geometry, analysis.decisive_episode_id)
    angle = analysis.angle
    if run is None or angle is None or not np.isfinite(run.half_amp_max_rad):
        return M_NOT_MEASURED
    limit = (1.0 - settings.confidence_thin_band) * angle.run_break_half_amplitude_rad
    return M_FULL if run.half_amp_max_rad <= limit else M_THIN


def _long_circulation_clear(analysis) -> bool:
    """A clear long-period circulation: margin P full and no turning point at any cutoff (x0.5, x1, x2)."""
    geometry = analysis.geometry
    p = analysis.verdict.margins.get('P')
    if geometry is None or not _margin_full(p):
        return False
    return len(geometry.turning_points) == 0 and geometry.n_tp_cut_half == 0 and geometry.n_tp_cut_double == 0


def _finish_long(analysis, outcome: _Outcome, settings: AnalysisSettings, ledger: Ledger, end: EndReading, span: float, zlk: bool) -> None:
    v = analysis.verdict
    geometry = analysis.geometry
    status = Status(outcome.status)
    episodes = []
    if geometry is not None:
        bounded = secular_decide.librating_episodes(geometry)
        if status in LIBRATING_STATUSES and not outcome.veto:
            episodes = [e for e in bounded if e.episode_id in outcome.counted] or bounded
            outcome.counted = frozenset(e.episode_id for e in episodes)
        else:
            outcome.counted = frozenset()
            outcome.candidates = outcome.candidates | frozenset(e.episode_id for e in bounded)
    outcome.candidates = outcome.candidates - outcome.counted
    exit_seen = geometry is not None and str(Flag.EXIT_SUSPECTED) in geometry.flags
    entry_seen = geometry is not None and str(Flag.ENTRY_SUSPECTED) in geometry.flags
    seen = {'entry_seen': entry_seen, 'exit_seen': exit_seen}
    n, basis = NAN, ''
    if episodes:
        decisive = next((e for e in episodes if e.episode_id == analysis.decisive_episode_id), None) or max(episodes, key=lambda e: e.n_tp)
        n = episode_cycles(decisive, settings, **seen)
        basis = N_COVER if v.angle_status == AngleStatus.LIBRATION else N_EPISODE
    period = outcome.period_dec_yr if outcome.period_dec_yr is not None else float(analysis.decisive_period_yr)
    last = max(episodes, key=lambda e: e.t_end_yr, default=None)
    reaches = bool(
        last is not None
        and np.isfinite(last.coverage_end_gap_p)
        and last.coverage_end_gap_p <= settings.libration_coverage_periods
        and not exit_seen
    )
    end = EndReading(
        end.t_last_yr, end.end_travel_rev, period, episode_cycles(last, settings, **seen) if last is not None else NAN, reaches
    )
    if last is not None:
        ledger.compare_setting(
            'policy:last counted episode end gap <= libration_coverage_periods',
            last.coverage_end_gap_p,
            Op.LE,
            settings,
            'libration_coverage_periods',
            'P',
            'ep_coverage_end_gap_P',
        )
    phase_unresolved = v.angle_subtype == AngleSubtype.PHASE_UNRESOLVED
    insufficient = (outcome.angle_status or v.angle_status) == AngleStatus.INSUFFICIENT_TIME and status == Status.UNCERTAIN
    has_tp = geometry is not None and len(geometry.turning_points) > 0
    gate = analysis.gate
    veto_need = NAN
    excluded = str(Flag.KINEMATIC_EXCLUDED) in outcome.flags  # W30 D2: no length needed to separate a node from s_j
    if zlk and gate is not None and gate.outcome in ('short_window', 'planetary_residual') and not outcome.veto and not excluded:
        outcome.flags.append(str(Flag.VETO_NEEDS_LENGTH))
        if gate.outcome == 'short_window':
            veto_need = secular_extend.gate_cycles_need(gate, span)
        else:
            veto_need = secular_extend.gate_resolution_need(gate, span, settings)
    clause = ''
    if geometry is not None:
        clause = extend_clause(
            status,
            end,
            span,
            settings,
            insufficient_time=insufficient,
            has_turning_points=has_tp,
            blocked=phase_unresolved,
            veto_need_yr=veto_need,
            ledger=ledger,
        )
    ev = Evidence(
        n_cycles=n,
        n_basis=basis,
        margin=_long_margin(analysis, settings),
        veto_state=_long_veto_state(analysis, outcome),
        stable=_margin_full(v.margins.get('F')) and str(Flag.FILTER_TOO_SLOW) not in v.flags,
        core_cycles=core_cycles(geometry, analysis.decisive_episode_id),
        veto=outcome.veto,
        veto_thin=outcome.veto_thin,
        # a vetoed long-period row draws its episodes without a stability test (spec §4), so they do not lower the
        # level the way a stable MMR candidate does (591986: the gate measured kinematic over 4.6 cycles, high); an
        # unvetoed non_resonant row with a candidate is medium as for MMR (88899 at 100 kyr was high)
        candidate=bool(outcome.candidates) and not outcome.veto,
        circulation_clear=_long_circulation_clear(analysis),
    )
    _record_level(ev, settings, ledger)
    confidence = level(status, ev, settings)
    if outcome.veto_thin and confidence in (Confidence.CERTAIN, Confidence.HIGH):
        confidence = Confidence.MEDIUM
    beyond = geometry is not None and not phase_unresolved and veto_beyond_budget(veto_need, span, settings, end)
    action = _action(status, confidence, clause, None, v.action, beyond)
    extend_to = None
    if action == Action.EXTEND:
        extend_to = _long_extend_to(analysis, end, span, settings, veto_need, clause)
    _write(v, outcome, ev, confidence, action, end, clause, extend_to, None, veto_need, ledger)
    analysis.policy = PolicyResult(
        POLICY_MINIMAL,
        outcome.counted,
        outcome.candidates,
        (),
        (),
        ev.n_cycles,
        ev.margin == M_FULL,
        ev.veto_state,
        ev.stable,
        action == Action.EXTEND,
        clause,
    )


def _long_extend_to(
    analysis, end: EndReading, span: float, settings: AnalysisSettings, gate_need: float, clause: str = ''
) -> Optional[float]:
    """extend_to_years of a long-period row: clause C the next check of the gate's need; otherwise the next check of
    span + max(HIGH - N_last, 1) x P_dec, at least the gate's need when the gate needs more length; without a deciding
    period the tree's estimate."""
    if clause == CLAUSE_VETO:
        return mmr_decide.next_check(gate_need, span, settings)
    n_last = end.n_last if math.isfinite(end.n_last) else 0.0
    if math.isfinite(end.period_dec_yr) and end.period_dec_yr > 0:
        estimate = span + max(settings.confidence_high_cycles - n_last, 1.0) * end.period_dec_yr
    elif analysis.extension is not None and analysis.extension.extend_to_years is not None:
        estimate = float(analysis.extension.extend_to_years)
    else:
        estimate = NAN
    if math.isfinite(gate_need):
        estimate = max(estimate, gate_need) if math.isfinite(estimate) else gate_need
    return mmr_decide.next_check(estimate, span, settings)
