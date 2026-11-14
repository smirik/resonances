"""The verdict of one angle of one body: the output model shared by every resonance type.

`status` is the physics, `angle_status` (+ `angle_subtype`) only the geometry of the angle over the record
(librating episodes followed by another regime are `episodic`; `final_regime` says how the record ends),
`reason` the one main explanation, `libration_origin` who makes the bounded oscillation
(with `origin_detail`, the concrete cause for a human, and the roles of the other resonance: `owner` an owner or driver
that covers the episode, `neighbours` one that sits before, during or after it without making it, `resonant_argument`
the member of the multiplet that librates, `cause_role`/`cause_term` the term behind the verdict, `commensurability`
the key a statistic counts once; `analysis.mmr.roles`),
`checks` the values of the named checks, `margins` and `decision_path` how sure the decision is
and which comparisons made it, `rules_fired` which named rules acted on the row (`analysis.rules`).
`status_code` is the old integer, kept for comparison with old runs.

MMR verdicts come from `analysis.mmr`, secular and ZLK verdicts from `analysis.secular` and `analysis.zlk`;
every one ends with the W29 minimal policy (`analysis.policy`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Optional

from resonances.resonance.analysis.ledger import LedgerEntry, Margin
from resonances.resonance.analysis.rules import ordered as ordered_rules


class ResonanceType(StrEnum):
    MMR = 'mmr'
    SECULAR = 'secular'
    LIDOV_KOZAI = 'lidov_kozai'


class Status(StrEnum):
    RESONANT = 'resonant'
    TRANSIENT = 'transient'
    NEAR_SEPARATRIX = 'near_separatrix'
    NON_RESONANT = 'non_resonant'
    UNCERTAIN = 'uncertain'
    INVALID = 'invalid'


class AngleStatus(StrEnum):
    LIBRATION = 'libration'
    EPISODIC = 'episodic'  # 'transient' before 2026-10-17
    NEAR_SEPARATRIX = 'near_separatrix'
    CIRCULATION = 'circulation'
    DRIFT = 'drift'
    INSUFFICIENT_TIME = 'insufficient_time'
    UNDEFINED = 'undefined'


class AngleSubtype(StrEnum):
    STATIONARY = 'stationary'
    MODULATED = 'modulated'
    MOVING_CENTRE = 'moving_centre'  # libration about the slowly moving minimum of the averaged potential (psi frame)
    REGIME_CHANGE = 'regime_change'  # episodes without a circulation between them cover the record (1:1: its regimes)
    RECURRENT = 'recurrent'
    ALIAS = 'alias'
    ARC = 'arc'
    STALL = 'stall'
    FEW_CYCLES = 'few_cycles'
    UNPHYSICAL = 'unphysical'
    UNDEFINED_PHASE = 'undefined_phase'
    # a secular swing within how far the linear mode phases stray over the record (W28 D5, `secular.decide.phase_check`)
    PHASE_UNRESOLVED = 'phase_unresolved'
    PATTERN_A = 'pattern_a'
    PATTERN_B = 'pattern_b'
    NONE = ''


class Confidence(StrEnum):
    CERTAIN = 'certain'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'
    NONE = ''


# Best first; NONE (no level: uncertain, invalid) is not a level and never compared
CONFIDENCE_ORDER = (Confidence.CERTAIN, Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW)


class Action(StrEnum):
    NONE = 'none'
    SPOT_CHECK = 'spot_check'
    REVIEW = 'review'
    EXTEND = 'extend'
    RESAMPLE = 'resample'


class CheckValue(StrEnum):
    """pass supports the resonant reading, fail refutes it, not_measured: needed but impossible here,
    not_applicable: does not concern this shape of the angle, NONE: a column of another resonance type."""

    PASS = 'pass'
    FAIL = 'fail'
    NOT_MEASURED = 'not_measured'
    NOT_APPLICABLE = 'not_applicable'
    NONE = ''


class MarginClass(StrEnum):
    DECISIVE = 'D'  # in the decision tree; cannot be `fail` in the final verdict
    SUPPORT = 'S'  # may fail without changing the verdict; each failure lowers the level
    CAP = 'cap'  # a failure limits the level instead of being counted


class LibrationOrigin(StrEnum):
    DYNAMIC = 'dynamic'  # the angle's own force
    INDUCED = 'induced'  # another resonance (the owner) drives a; the angle follows by the identity
    SUBRESONANCE = 'subresonance'  # another member of the same multiplet has the fixed centre
    KINEMATIC = 'kinematic'  # bounded by geometry (varpi part, secular cycle), a does not respond
    # the owner holds its own angle and this angle's own term holds Psi = u sigma - v phi_o as well: both resonances at
    # once (`owner.second_capture` confirmed, W28 A2b); the verdict is the angle's own reading
    JOINT = 'joint'
    UNKNOWN = 'unknown'  # bounded, mechanism not established
    NONE = ''  # no bounded oscillation


class OriginDetail(StrEnum):
    """The concrete cause behind `libration_origin`, for a human checking the verdict."""

    VARPI_TERM = 'varpi_term'  # kinematic: the lambda part cancels p varpi_dot; a carries < B1 of the amplitude
    SECULAR_CYCLE = 'secular_cycle'  # kinematic: forced oscillation with a secular period (e-cycle, 2 omega)
    PLANET_RIPPLE = 'planet_ripple'  # kinematic: ripple of a planetary term survived the filter
    OWNER_LIBRATION = 'owner_libration'  # induced: the owner librates and drives a
    OWNER_FORCED_WAVE = 'owner_forced_wave'  # induced: the owner circulates at the oscillation period
    FIXED_SUBRESONANCE = 'fixed_subresonance'  # subresonance: one member has a fixed centre (Rouche holds)
    COMPOSITE_MULTIPLET = 'composite_multiplet'  # subresonance: composite regime, no single member dominates
    FREE_OMEGA_WINDS = 'free_omega_winds'  # kinematic (ZLK): the free omega winds, the forced e holds the osculating one
    # kinematic (MMR, W31): every counted episode is the forced wave its background carries (`mmr.background_wave`)
    FORCED_WAVE_BACKGROUND = 'forced_wave_background'
    # forced (ZLK, W30 D1): the bounded omega's e-i exchange breaks sqrt(1-e^2) cos i where the forced terms are small
    EI_INTEGRAL_NOT_HELD = 'ei_integral_not_held'
    # joint: x = Psi / u closes cycles of its own well (energy integral, x_ddot on sin x, a period no secular one)
    JOINT_CAPTURE = 'joint_capture'
    NONE = ''


class CauseRole(StrEnum):
    """The role of the other term named next to an MMR verdict (`analysis.mmr.roles`, W28 A3a, card g04-6)."""

    OWNER = 'owner'  # another commensurability librates and makes the episode (origin_detail owner_libration)
    DRIVER = 'driver'  # another commensurability circulates; its forced wave makes the cycles (owner_forced_wave)
    NEIGHBOUR = 'neighbour'  # an accepted owner that does not make the row's own episodes (F7/P2): before/during/after
    KINEMATIC = 'kinematic'  # the secular part of sigma (p varpi, ...) makes the cycles, a does not respond
    JOINT = 'joint'  # another commensurability librates at the same time; both resonances hold (origin joint, A2b)
    NONE = ''


class Reason(StrEnum):
    DYNAMIC_LIBRATION = 'dynamic_libration'  # a measured force (L4, R0 or the directly averaged F) passes: origin dynamic
    DYNAMIC_EPISODE = 'dynamic_episode'
    # nothing measured the force and the origin is unknown: only the geometry decided (MMR leaf 5u/6; author
    # 2026-10-26, Q3: was dynamic_libration/dynamic_episode with origin unknown)
    GEOMETRIC_LIBRATION = 'geometric_libration'
    GEOMETRIC_EPISODE = 'geometric_episode'
    SEPARATRIX_ZONE = 'separatrix_zone'
    CIRCULATION = 'circulation'
    ALIAS = 'alias'
    NO_RESONANT_POTENTIAL = 'no_resonant_potential'
    MODULATED_CIRCULATION = 'modulated_circulation'
    KINEMATIC_LIBRATION = 'kinematic_libration'
    KINEMATIC_DRIFT = 'kinematic_drift'
    INDUCED_LIBRATION = 'induced_libration'
    SUBRESONANCE_LIBRATES = 'subresonance_librates'
    BASELINE_NEEDED = 'baseline_needed'
    DRIFT_MECHANISM_UNKNOWN = 'drift_mechanism_unknown'
    ORIGIN_UNKNOWN = 'origin_unknown'
    GATE_UNAVAILABLE = 'gate_unavailable'
    UNUSABLE_SERIES = 'unusable_series'
    UNPHYSICAL = 'unphysical'
    UNDEFINED_PHASE = 'undefined_phase'
    # the output step is too coarse for the angle's filter: P_c <= 2 dt, so no filter keeps the shortest libration
    UNDERSAMPLED = 'undersampled'
    INVALID_INPUT = 'invalid_input'  # the angle is no MMR of the body (k = 0): refused before the analysis
    # the force law and the potential with a are not measured and the episode's period needs a force >=
    # mmr_strength_implausible_ratio x the directly averaged one (two-body) or x mu_i mu_j e^q even after the generous
    # small-divisor envelope (three-body): non_resonant, origin unknown (W18, author 2026-10-17)
    IMPLAUSIBLE_STRENGTH = 'implausible_strength'
    # a 1:1 angle without a librating episode whose best single leg between two passages (leaf G4c, W28 C6, q-4 A) is
    # a bounded swing of a co-orbital well with one discrepancy (a lap past the planet, the local law or energy, a turn
    # that does not reflect): no capture shown, none excluded (uncertain; the figure decides, action review)
    ONE_COORBITAL_LEG = 'one_coorbital_leg'
    # an MMR angle that is not 1:1 circulating but for one bounded half-swing between passages that follows the
    # resonance's own one-harmonic law (leaf G4h, W28 C6, q-5 i): a half cycle, no capture shown, none excluded
    ONE_HALF_LIBRATION = 'one_half_libration'


class Flag(StrEnum):
    """Every flag a verdict may carry (column `flags`, ';'-separated)."""

    EXIT_SUSPECTED = 'exit_suspected'
    ENTRY_SUSPECTED = 'entry_suspected'
    KICK = 'kick'
    LAST_REGIME_AFTER_JUMP = 'last_regime_after_jump'
    # G2a: an accepted run cut at a step of its libration action (`geometry.ActionStep`), as at a confirmed jump of a
    ACTION_STEP = 'action_step'
    # G2a cut with fewer than LIBRATION_JUMP_CYCLES energy cycles on a side: a provisional split (cap action_step_provisional)
    ACTION_STEP_PROVISIONAL = 'action_step_provisional'
    NEAR_EPISODE = 'near_episode'  # a verdict without libration next to a run a few TPs short of an episode
    LIBRATION_IN_SEPARATRIX_ZONE = 'libration_in_separatrix_zone'
    NEAR_SEPARATRIX_ENERGY = 'near_separatrix_energy'
    IMPLAUSIBLE_STRENGTH = 'implausible_strength'
    RIPPLE_NOT_REMOVED = 'ripple_not_removed'
    FILTER_TOO_SLOW = 'filter_too_slow'
    # dt >= min(P_orb, P_syn of the angle's planets) / 2, or a harmonic up to MMR_SHORT_PERIOD_HARMONICS folded slower
    # than the period floor: short-period terms of the elements fold into slow ones
    SHORT_PERIOD_ALIASED = 'short_period_aliased'
    # with short_period_aliased: the short-period scatter of a exceeds MMR_SHORT_PERIOD_A_RATIO x the decisive
    # episode's a half-excursion (the folded lines can reach the a-based checks; a warning)
    A_SHORT_PERIOD_SCATTER = 'a_short_period_scatter'
    PLANETS_MISSING = 'planets_missing'
    JITTER_LOW_E = 'jitter_low_e'
    ALIAS_NP_UNWRAP_DIFFERS = 'alias_np_unwrap_differs'
    FAR_FROM_RESONANCE = 'far_from_resonance'
    SUBRESONANCE_COMPOSITE = 'subresonance_composite'
    SUBRESONANCE_UNDETERMINED = 'subresonance_undetermined'
    NODAL_SUBRESONANCE_CANDIDATE = 'nodal_subresonance_candidate'
    SUBRESONANCE_TIE = 'subresonance_tie'  # test C decided by the order only
    SUBRESONANCE_THEORY_DISAGREES = 'subresonance_theory_disagrees'  # test D measured and failed (confidence only)
    SUBRESONANCE_THEORY_CROSSING = 'subresonance_theory_crossing'  # no theory: the orbits can intersect
    SUBRESONANCE_THEORY_ROUND_OFF = 'subresonance_theory_round_off'  # no theory: coefficients below round-off
    SUBRESONANCE_THEORY_RANGE = 'subresonance_theory_range'  # no theory: e or e_P outside (0, 1) or an empty window
    OWNER_PROBABLE = 'owner_probable'  # the best owner candidate fits a with a reachable force, but its phase drifts
    # the owner search took some planet's lambda from the catalogue line lambda_j0 + n_j t (it was not saved)
    CATALOGUE_LONGITUDES = 'catalogue_longitudes'
    # a candidate on catalogue longitudes whose coherence lies within the line's error bound of a threshold
    OWNER_CATALOGUE_AMBIGUOUS = 'owner_catalogue_ambiguous'
    # an accepted owner whose intervals (its measured regimes only) cover less than mmr_owner_cover_min of an accepted
    # episode of sigma: that episode is the row's own; kept in `owner`, origin none when no episode is induced (F7)
    OWNER_OUTSIDE_EPISODE = 'owner_outside_episode'
    # P2: an owner in its libration regime whose Psi/u turns at least mmr_owner_psi_turns_per_cycle_max times per cycle
    # of an episode of sigma: it cannot make that episode's cycles, the episode is the row's own (A1, W28)
    OWNER_PSI_TOO_FAST = 'owner_psi_too_fast'
    # no owner accepted, the best candidate's phases cannot separate it from sigma (phase_unresolved) and its potential
    # route is not applicable: check_mmr_owner not_measured
    OWNER_UNRESOLVED = 'owner_unresolved'
    # the accepted owner leaves open whether sigma's own term also holds Psi (`owner.second_capture`): confidence <= medium
    SECOND_CAPTURE_UNRESOLVED = 'second_capture_unresolved'
    # crossings with every cycle inside the model's separatrix zone (a column, no status; 'separatrix_chaos' before
    # 2026-10-17)
    EPISODES_IN_SEPARATRIX_ZONE = 'episodes_in_separatrix_zone'
    # an episodic angle that left a libration at least once and came back at least once (sep_n_exits, sep_n_captures),
    # with its own force measured: a mark that names no mechanism and changes no status (W17 Q12)
    EXIT_AND_CAPTURE = 'exit_and_capture'
    ZONE_EQ_ONLY = 'zone_eq_only'
    ZONE_NUMERIC_MISMATCH = 'zone_numeric_mismatch'
    ZONE_EDGE_CYCLE = 'zone_edge_cycle'
    THREE_TP_EPISODE = 'three_tp_episode'
    FILTER_SENSITIVE = 'filter_sensitive'  # a decisive number changes at cutoff x2: check_mmr_filter not_measured
    UNWRAP_STEP_SUSPECT = 'unwrap_step_suspect'
    VARPI_FAST_LOW_E = 'varpi_fast_low_e'  # a step where only the slow elements turned fast (usually e near 0): no alias
    PHASE_UNDEFINED_LOCAL = 'phase_undefined_local'  # somewhere |p| s/|z_f| > mmr_jitter_max_rad (a column, no status)
    STALL_RATE_INCONSISTENT = 'stall_rate_inconsistent'
    FORCE_LAW_DISAGREES = 'force_law_disagrees'
    FORCE_LAW_CONTAMINATED = 'force_law_contaminated'  # a forced line of the angle's lattice at the libration period
    # rule R: the status reads a witness episode whose force law holds while another episode's is measured and fails
    EPISODE_FORCE_LAW_MIXED = 'episode_force_law_mixed'
    JUMP_BY_A_ONLY = 'jump_by_a_only'
    JUMP_AT_EDGE = 'jump_at_edge'  # a step of a without data on both sides to confirm it: kept, the record not cut
    JUMP_LOCATOR_ALIASED = 'jump_locator_aliased'
    # W28 C14 (card g01-7, rule end_arc_extend): a longer integration would resolve how a record edge ends, in
    # `Verdict.extend_direction` to `extend_to_years` (an end arc after circulation, one reflection or a clean 1:1 leg
    # or half-libration cut by the edge); changes no status and no level, whatever the action
    EXTEND_TIME = 'extend_time'
    # W29 minimal policy (`analysis.policy`): marks that change no status and no level by themselves
    GATE_UNAVAILABLE = 'gate_unavailable'  # ZLK: the free-omega gate gave no verdict; the geometry's status, <= medium
    # ZLK (W30 D2): no gate verdict, but rho_max >= ZLK_RHO_KINEMATIC_EXCLUDED and the e-i integral held in the deciding
    # episode: the forced vector cannot hold the perihelion, so the gate's abstention is no cap and asks no extension
    KINEMATIC_EXCLUDED = 'kinematic_excluded'
    # ZLK (W30 D1): inside the deciding episode e and i do not trade at constant sqrt(1-e^2) cos i (eta_H above
    # ZLK_EI_H_RESIDUAL_MAX, or not in antiphase): a measured objection (low), a veto where the forced terms are small
    EI_INTEGRAL_NOT_HELD = 'ei_integral_not_held'
    GATE_DISABLED = 'gate_disabled'  # ZLK: the gate is switched off; the geometry's status, <= medium
    ONE_COORBITAL_LEG = 'one_coorbital_leg'  # the status rests on one 1:1 leg (G4c), or a start-edge leg was set aside
    ONE_HALF_LIBRATION = 'one_half_libration'  # a bounded half-swing between passages (G4h), not counted
    REJECTED_CANDIDATE = 'rejected_candidate'  # a bounded stretch was read and not counted (drawn faint)
    MEASURED_OBJECTION = 'measured_objection'  # a measurement objects to the angle's own dynamics: level low
    STRENGTH_SCALE_IMPLAUSIBLE = 'strength_scale_implausible'  # three-body e^q scale far below the need: level low
    VETO_THIN = 'veto_thin'  # the veto's deciding comparison lies within CONFIDENCE_THIN_BAND of its threshold
    VETO_NEEDS_LENGTH = 'veto_needs_length'  # a needed veto was not measured for lack of length (pol_veto_need_yr)
    POLICY_UNKNOWN_LEAF = 'policy_unknown_leaf'  # a v2 leaf the policy has no mapping for: the geometry's status


# the `policy` column of a verdict the W29 minimal policy layer wrote (`analysis.policy`); empty in older files
POLICY_MINIMAL = 'minimal'


MMR_CHECKS = (
    'check_mmr_alias',
    'check_mmr_filter',
    'check_mmr_baseline',
    'check_mmr_identity',
    'check_mmr_a_response',
    'check_mmr_potential',
    'check_mmr_a_amplitude',
    'check_mmr_kinematic',
    'check_mmr_subresonances',
    'check_mmr_owner',
    'check_mmr_slow_libration',
    'check_mmr_one_pendulum',
    'check_mmr_onset',
    'check_mmr_exit',
    'check_mmr_separatrix_zone',
    'check_mmr_strength',
)
SECULAR_CHECKS = ('check_secular_e_response',)
ZLK_CHECKS = ('check_zlk_free_omega', 'check_zlk_ei')
ALL_CHECKS = MMR_CHECKS + SECULAR_CHECKS + ZLK_CHECKS
CHECKS_OF_TYPE = {ResonanceType.MMR: MMR_CHECKS, ResonanceType.SECULAR: SECULAR_CHECKS, ResonanceType.LIDOV_KOZAI: ZLK_CHECKS}


def unmeasured_checks(resonance_type: ResonanceType) -> dict:
    """Every check column of the type, not measured (the baseline of a verdict reached before any check)."""
    return dict.fromkeys(CHECKS_OF_TYPE[resonance_type], CheckValue.NOT_MEASURED)


# multiplet_status: the best status among the checked members of a multiplet, best first
STATUS_RANK = (Status.RESONANT, Status.TRANSIENT, Status.NEAR_SEPARATRIX, Status.UNCERTAIN, Status.NON_RESONANT, Status.INVALID)

LIST_SEPARATOR = ';'


def margin_of_check(check, thin: bool = False) -> Margin:
    """The margin of a check value: pass full (thin when `thin`), fail fail, not_applicable not_applicable, anything
    else not_measured."""
    return {
        CheckValue.PASS: Margin.THIN if thin else Margin.FULL,
        CheckValue.FAIL: Margin.FAIL,
        CheckValue.NOT_APPLICABLE: Margin.NOT_APPLICABLE,
    }.get(check, Margin.NOT_MEASURED)


@dataclass(frozen=True)
class MarginValue:
    """One margin (SPEC §7.1): how far the decision is from changing, as a class and a number."""

    margin: Margin
    value: Optional[float] = None
    margin_class: MarginClass = MarginClass.DECISIVE


@dataclass
class Verdict:
    resonance_type: ResonanceType
    status: Status
    angle_status: AngleStatus
    angle_subtype: AngleSubtype = AngleSubtype.NONE
    confidence: Confidence = Confidence.NONE
    confidence_raw: Confidence = Confidence.NONE
    reason: Reason | str = ''
    flags: list[str] = field(default_factory=list)
    # how the record ends, when it ends in a regime of its own (MMR): the angle_status of the last regime after librating
    # episodes (circulation after G1c; that of the regime after the last jump of a, G1e), or libration when the last
    # episode reaches the record's end; '' otherwise. `final_regime_start_yr`: where that regime starts
    final_regime: AngleStatus | str = ''
    final_regime_start_yr: Optional[float] = None
    # the orbit's type in the librating episodes, for the figure's header ('tadpole L4', 'horseshoe',
    # 'interior, asymmetric libration about 72°', 'ω libration about 90°'; `orbit_type.py`); '' without libration
    orbit_type: str = ''
    # the calibration cell's columns (`confidence.CellRecord`: author-labelled verdicts at high or certain, their errors
    # and the Wilson 95 % interval of the accuracy): empty under the W29 minimal policy, which reads no cell
    cell_n_examples: Optional[int] = None
    cell_n_errors: Optional[int] = None
    cell_accuracy_low_95: Optional[float] = None
    cell_accuracy_high_95: Optional[float] = None
    thin_margins: list[str] = field(default_factory=list)
    failed_support: list[str] = field(default_factory=list)
    caps: list[str] = field(default_factory=list)
    action: Action = Action.NONE
    # with action extend: the total length to integrate; with the flag extend_time (any action) the length that flag
    # asks for, `extend_direction` forward (past the end), backward (before the start) or both; '' without the flag
    extend_to_years: Optional[float] = None
    extend_direction: str = ''
    resample_step_yr: Optional[float] = None
    libration_origin: LibrationOrigin = LibrationOrigin.NONE
    origin_detail: OriginDetail | str = OriginDetail.NONE
    # the roles of the other resonance (MMR, `analysis.mmr.roles`; '' for another type and in older files): `owner` keeps
    # only an owner or driver that covers the deciding episode; `neighbours` 'KEY t0–t1 before|during|after;...' [yr]
    # for an accepted owner F7/P2 refused; `resonant_argument` the multiplet member that librates (rule C) or 'composite';
    # `cause_role` (CauseRole) and `cause_term` the full key of that resonance or the secular term ('-4varpi') of a
    # kinematic libration; `commensurability` the key without the varpi and node coefficients ('9M-16')
    owner: str = ''
    neighbours: str = ''
    resonant_argument: str = ''
    cause_role: CauseRole | str = CauseRole.NONE
    cause_term: str = ''
    commensurability: str = ''
    multiplet_status: Optional[Status] = None
    checks: dict[str, CheckValue] = field(default_factory=dict)
    margins: dict[str, MarginValue] = field(default_factory=dict)
    decision_path: list[LedgerEntry] = field(default_factory=list)
    # the named rules (`analysis.rules.Rule`) that acted on this row, also when the verdict stayed the same: merged by
    # the decide steps from the geometry and the result objects, deduplicated, in registry order (column `rules_fired`)
    rules_fired: list[str] = field(default_factory=list)
    # W29 policy layer (`analysis.policy`, docs/physics/status-tree.md): which policy set the row ('' in older files),
    # the complete cycles N its level read, the episodes it counts and the candidates it drew faint (';'-joined ids, a
    # 1:1 leg 'leg', a half-swing 'half'), the measured veto that moved a bounded angle to non_resonant, the evidence
    # behind the level (`confidence_basis`), the W28 tree's verdict under minimal (status/confidence/reason/action/angle)
    # and the quantities of the extend predicate (pol_*)
    policy: str = ''
    n_cycles: float = math.nan
    n_counted_episodes: Optional[int] = None
    counted_episode_ids: str = ''
    candidate_episode_ids: str = ''
    veto: str = ''
    confidence_basis: str = ''
    v2_verdict: str = ''
    pol_end_travel_rev: float = math.nan
    pol_t_last_tp_yr: float = math.nan
    pol_period_dec_yr: float = math.nan
    pol_n_last: float = math.nan
    pol_last_reaches_end: Optional[bool] = None
    pol_extend_clause: str = ''
    pol_veto_need_yr: float = math.nan
    pol_core_cycles: float = math.nan  # (n_TP - 1) / 2 of the decisive episode: the observed cycles certain needs
    # each input the level read with the highest level it allows (`policy.level_inputs`, 'N:4.12>=3:certain;M:thin:high'),
    # the policy step that decided the status ('tree' when it kept the tree's; `policy.DECIDED_*`) and the checks whose
    # measured objection made V ambiguous (`policy.OBJECTION_*`, ';'-joined)
    pol_level_inputs: str = ''
    pol_decided_by: str = ''
    pol_objection_check: str = ''

    @property
    def status_code(self) -> int:
        """The old integer status (temporary, see `status_code`)."""
        return status_code(self.resonance_type, self.status, self.angle_status, self.reason)

    @property
    def review_required(self) -> bool:
        """Whether the row is in the review queue (`needs_review`; column `review_required`). Changes no status."""
        return needs_review(self.confidence, self.action, self.libration_origin, self.reason)

    @property
    def review_reasons(self) -> list[str]:
        """Why the row is in the review queue (column `review_reasons`; empty outside it): `confidence:<level>` for
        medium/low, `action:<action>` for a spot_check/review action at no such level, `origin_unknown` for a bounded
        angle without an established mechanism, then `veto_thin` when the veto's comparison was thin. The minimal policy
        leaves no caps and no failed supports (the level is the evidence's, `policy.level`), so neither is a reason."""
        if not self.review_required:
            return []
        reasons = []
        if str(self.confidence) in REVIEW_LEVELS:
            reasons.append(f'confidence:{self.confidence}')
        elif str(self.action) in REVIEW_ACTIONS:
            reasons.append(f'action:{self.action}')
        if _origin_unknown(self.libration_origin, self.reason):
            reasons.append(str(Reason.ORIGIN_UNKNOWN))
        if str(Flag.VETO_THIN) in self.flags:
            reasons.append(str(Flag.VETO_THIN))
        return reasons

    def to_flat_dict(self) -> dict[str, Any]:
        """The verdict columns of summary.csv; unmeasured numbers are None, empty texts ''."""
        row: dict[str, Any] = {
            'resonance_type': str(self.resonance_type),
            'status': str(self.status),
            'angle_status': str(self.angle_status),
            'angle_subtype': str(self.angle_subtype),
            'final_regime': str(self.final_regime),
            'final_regime_start_yr': self.final_regime_start_yr,
            'orbit_type': self.orbit_type,
            'confidence': str(self.confidence),
            'confidence_raw': str(self.confidence_raw),
            'cell_n_examples': self.cell_n_examples,
            'cell_n_errors': self.cell_n_errors,
            'cell_accuracy_low_95': self.cell_accuracy_low_95,
            'cell_accuracy_high_95': self.cell_accuracy_high_95,
            'reason': str(self.reason),
            'flags': join_list(self.flags),
            'action': str(self.action),
            'review_required': self.review_required,
            'review_reasons': join_list(self.review_reasons),
            'extend_to_years': self.extend_to_years,
            'extend_direction': self.extend_direction,
            'resample_step_yr': self.resample_step_yr,
            'libration_origin': str(self.libration_origin),
            'origin_detail': str(self.origin_detail),
            'owner': self.owner,
            'neighbours': self.neighbours,
            'resonant_argument': self.resonant_argument,
            'cause_role': str(self.cause_role),
            'cause_term': self.cause_term,
            'commensurability': self.commensurability,
            'multiplet_status': str(self.multiplet_status) if self.multiplet_status is not None else '',
            'status_code': self.status_code,
        }
        for name in ALL_CHECKS:
            row[name] = str(self.checks.get(name, CheckValue.NONE))
        row['thin_margins'] = join_list(self.thin_margins)
        row['failed_support'] = join_list(self.failed_support)
        row['caps'] = join_list(self.caps)
        for margin_id, margin in self.margins.items():
            row[f'mrg_{margin_id}'] = str(margin.margin)
            row[f'mrg_{margin_id}_value'] = margin.value
        row['decision_path'] = join_list(entry.name for entry in self.decision_path)
        row['rules_fired'] = join_list(ordered_rules(self.rules_fired))
        row.update(
            {
                'policy': self.policy,
                'n_cycles': finite_or_none(self.n_cycles),
                'n_counted_episodes': self.n_counted_episodes,
                'counted_episode_ids': self.counted_episode_ids,
                'candidate_episode_ids': self.candidate_episode_ids,
                'veto': self.veto,
                'confidence_basis': self.confidence_basis,
                'v2_verdict': self.v2_verdict,
                'pol_end_travel_rev': finite_or_none(self.pol_end_travel_rev),
                'pol_t_last_tp_yr': finite_or_none(self.pol_t_last_tp_yr),
                'pol_period_dec_yr': finite_or_none(self.pol_period_dec_yr),
                'pol_n_last': finite_or_none(self.pol_n_last),
                'pol_last_reaches_end': self.pol_last_reaches_end,
                'pol_extend_clause': self.pol_extend_clause,
                # inf: no record up to ZLK_EXTEND_MAX_YR resolves the veto (written, not blank: blank is "no length named")
                'pol_veto_need_yr': math.inf if self.pol_veto_need_yr == math.inf else finite_or_none(self.pol_veto_need_yr),
                'pol_core_cycles': finite_or_none(self.pol_core_cycles),
                'pol_level_inputs': self.pol_level_inputs,
                'pol_decided_by': self.pol_decided_by,
                'pol_objection_check': self.pol_objection_check,
            }
        )
        return row

    @classmethod
    def from_flat_dict(cls, row: Mapping[str, Any]) -> 'Verdict':
        """Rebuild a verdict from a summary.csv row (restore without recomputation).

        The decision path keeps only the comparison names (the numbers are in the row's columns);
        margins keep their class as DECISIVE (the class is a property of the rule, not stored).
        `review_required`/`review_reasons` are not read: they follow from the other columns (absent in older files).
        """
        margins = {}
        for key, value in row.items():
            if key.startswith('mrg_') and not key.endswith('_value') and cell_text(value):
                margin_id = key[len('mrg_') :]
                margins[margin_id] = MarginValue(Margin(cell_text(value)), _number(row.get(f'{key}_value')))
        multiplet = cell_text(row.get('multiplet_status'))
        return cls(
            resonance_type=ResonanceType(cell_text(row.get('resonance_type'))),
            status=Status(cell_text(row.get('status'))),
            angle_status=AngleStatus(cell_text(row.get('angle_status'))),
            angle_subtype=AngleSubtype(cell_text(row.get('angle_subtype'))),
            final_regime=_enum_or_text(AngleStatus, cell_text(row.get('final_regime'))),
            final_regime_start_yr=_number(row.get('final_regime_start_yr')),
            orbit_type=cell_text(row.get('orbit_type')),
            confidence=Confidence(cell_text(row.get('confidence'))),
            confidence_raw=Confidence(cell_text(row.get('confidence_raw'))),
            cell_n_examples=_integer(row.get('cell_n_examples')),
            cell_n_errors=_integer(row.get('cell_n_errors')),
            cell_accuracy_low_95=_number(row.get('cell_accuracy_low_95')),
            cell_accuracy_high_95=_number(row.get('cell_accuracy_high_95')),
            reason=_enum_or_text(Reason, cell_text(row.get('reason'))),
            flags=split_list(row.get('flags')),
            thin_margins=split_list(row.get('thin_margins')),
            failed_support=split_list(row.get('failed_support')),
            caps=split_list(row.get('caps')),
            action=Action(cell_text(row.get('action')) or Action.NONE),
            extend_to_years=_number(row.get('extend_to_years')),
            extend_direction=cell_text(row.get('extend_direction')),
            resample_step_yr=_number(row.get('resample_step_yr')),
            libration_origin=LibrationOrigin(cell_text(row.get('libration_origin'))),
            origin_detail=_enum_or_text(OriginDetail, cell_text(row.get('origin_detail'))),
            owner=cell_text(row.get('owner')),
            neighbours=cell_text(row.get('neighbours')),
            resonant_argument=cell_text(row.get('resonant_argument')),
            cause_role=_enum_or_text(CauseRole, cell_text(row.get('cause_role'))),
            cause_term=cell_text(row.get('cause_term')),
            commensurability=cell_text(row.get('commensurability')),
            multiplet_status=Status(multiplet) if multiplet else None,
            checks={name: CheckValue(cell_text(row.get(name))) for name in ALL_CHECKS if cell_text(row.get(name))},
            margins=margins,
            decision_path=[LedgerEntry.name_only(name) for name in split_list(row.get('decision_path'))],
            rules_fired=ordered_rules(split_list(row.get('rules_fired'))),
            policy=cell_text(row.get('policy')),
            n_cycles=_nan_number(row.get('n_cycles')),
            n_counted_episodes=_integer(row.get('n_counted_episodes')),
            counted_episode_ids=cell_text(row.get('counted_episode_ids')),
            candidate_episode_ids=cell_text(row.get('candidate_episode_ids')),
            veto=cell_text(row.get('veto')),
            confidence_basis=cell_text(row.get('confidence_basis')),
            v2_verdict=cell_text(row.get('v2_verdict')),
            pol_end_travel_rev=_nan_number(row.get('pol_end_travel_rev')),
            pol_t_last_tp_yr=_nan_number(row.get('pol_t_last_tp_yr')),
            pol_period_dec_yr=_nan_number(row.get('pol_period_dec_yr')),
            pol_n_last=_nan_number(row.get('pol_n_last')),
            pol_last_reaches_end=_boolean(row.get('pol_last_reaches_end')),
            pol_extend_clause=cell_text(row.get('pol_extend_clause')),
            pol_veto_need_yr=_nan_number(row.get('pol_veto_need_yr')),
            pol_core_cycles=_nan_number(row.get('pol_core_cycles')),
            pol_level_inputs=cell_text(row.get('pol_level_inputs')),
            pol_decided_by=cell_text(row.get('pol_decided_by')),
            pol_objection_check=cell_text(row.get('pol_objection_check')),
        )


def join_list(items: Iterable[Any]) -> str:
    """A list cell of the CSV: ';'-separated, never ','."""
    return LIST_SEPARATOR.join(str(item) for item in items)


def split_list(cell: Any) -> list[str]:
    """Inverse of `join_list`; an empty or missing (NaN) cell is an empty list."""
    text = cell_text(cell)
    return [item for item in text.split(LIST_SEPARATOR) if item] if text else []


def cell_text(value: Any) -> str:
    """A CSV cell as stripped text; '' for None or NaN (an empty cell read back)."""
    if value is None or (isinstance(value, float) and value != value):
        return ''
    return str(value).strip()


def finite_or_none(value: Any, scale: float = 1.0) -> Optional[float]:
    """A number for a CSV cell, times `scale`: None (an empty cell) when missing, NaN or infinite."""
    return None if value is None or not math.isfinite(value) else float(value) * scale


def _number(value: Any) -> Optional[float]:
    text = cell_text(value)
    return float(text) if text else None


def _integer(value: Any) -> Optional[int]:
    number = _number(value)
    return int(number) if number is not None else None


def _nan_number(value: Any) -> float:
    number = _number(value)
    return number if number is not None else math.nan


def _boolean(value: Any) -> Optional[bool]:
    """A saved True/False cell (pandas reads it as bool or text); None for an empty one."""
    if isinstance(value, bool):
        return value
    text = cell_text(value).lower()
    if text in ('true', '1', '1.0'):
        return True
    if text in ('false', '0', '0.0'):
        return False
    return None


def _enum_or_text(enum_cls, text: str):
    try:
        return enum_cls(text)
    except ValueError:
        return text


def lower_confidence(*levels: Confidence) -> Confidence:
    """The lowest of the given levels (NONE ignored; all NONE gives NONE)."""
    ranked = [level for level in levels if level != Confidence.NONE]
    if not ranked:
        return Confidence.NONE
    return max(ranked, key=CONFIDENCE_ORDER.index)


def best_status(statuses: Iterable[Status]) -> Optional[Status]:
    """The best status by `STATUS_RANK` (multiplet_status); None for an empty input."""
    present = list(statuses)
    if not present:
        return None
    return min(present, key=STATUS_RANK.index)


# uncertain reasons a longer record does not settle: the figure decides (action review)
REVIEW_UNCERTAIN = (Reason.ORIGIN_UNKNOWN, Reason.ONE_COORBITAL_LEG, Reason.ONE_HALF_LIBRATION)


def default_action(status: Status, reason: Reason | str, confidence: Confidence) -> Action:
    """The action of SPEC §5.6: invalid alias or undersampled -> resample, invalid -> review, uncertain -> extend
    (origin_unknown, one_coorbital_leg, one_half_libration -> review), otherwise by confidence (certain/high none,
    medium spot_check, low review)."""
    if status == Status.INVALID:
        return Action.RESAMPLE if reason in (Reason.ALIAS, Reason.UNDERSAMPLED) else Action.REVIEW
    if status == Status.UNCERTAIN:  # one co-orbital leg: the body has left, a longer record need not help (g02-4 A)
        return Action.REVIEW if reason in REVIEW_UNCERTAIN else Action.EXTEND
    if confidence == Confidence.MEDIUM:
        return Action.SPOT_CHECK
    if confidence == Confidence.LOW:
        return Action.REVIEW
    return Action.NONE


# The review queue (W28 g10-1b; the author's q1 of 19.10: medium = spot check, low = review)
REVIEW_LEVELS = frozenset({str(Confidence.MEDIUM), str(Confidence.LOW)})
REVIEW_ACTIONS = frozenset({str(Action.SPOT_CHECK), str(Action.REVIEW)})


def _origin_unknown(libration_origin: Any, reason: Any) -> bool:
    return str(libration_origin or '') == LibrationOrigin.UNKNOWN or str(reason or '') == Reason.ORIGIN_UNKNOWN


def needs_review(confidence: Any, action: Any = '', libration_origin: Any = '', reason: Any = '') -> bool:
    """Whether a human should look at a row (the review queue, column `review_required`, `SavePlotMode.REVIEW`): its
    level is medium or low, its action is spot_check or review, or it has a bounded angle whose origin is not
    established. The level decides on its own (a medium row whose action became resample stays in the queue); the
    action adds the rows without a level (invalid, uncertain origin_unknown). Plain strings are accepted (a clone's
    row in a results archive has no action). Changes no status."""
    if str(confidence or '') in REVIEW_LEVELS or str(action or '') in REVIEW_ACTIONS:
        return True
    return _origin_unknown(libration_origin, reason)


_MMR_CODES = {
    Status.RESONANT: 2,
    Status.TRANSIENT: 1,
    Status.NEAR_SEPARATRIX: -3,
    Status.NON_RESONANT: 0,
    Status.INVALID: -99,
}


def status_code(
    resonance_type: ResonanceType | str,
    status: Status,
    angle_status: AngleStatus,
    reason: Reason | str,
) -> int:
    """The old integer status of a verdict (DECISIONS: temporary, for consumers of the old codes).

    MMR (SPEC §2.2): resonant 2, transient 1, near_separatrix -3, non_resonant 0 (kinematic,
    induced, subresonance and alias included), uncertain baseline_needed -6, other uncertain -9,
    invalid -99. Secular and ZLK (SPEC §5.7): kinematic libration -2, near_separatrix -3,
    uncertain insufficient_time -4, uncertain drift -6, other uncertain -9 (the old -5, a near_separatrix
    read from the old classifier's files, is produced by no current analysis).
    """
    resonance_type = ResonanceType(resonance_type)
    if resonance_type == ResonanceType.MMR:
        if status == Status.UNCERTAIN:
            return -6 if reason == Reason.BASELINE_NEEDED else -9
        return _MMR_CODES[status]
    if status == Status.NON_RESONANT:
        return -2 if reason == Reason.KINEMATIC_LIBRATION else 0
    if status == Status.NEAR_SEPARATRIX:
        return -3
    if status == Status.UNCERTAIN:
        if reason == Reason.BASELINE_NEEDED:
            return -4 if angle_status == AngleStatus.INSUFFICIENT_TIME else -6
        return -9
    return _MMR_CODES[status]
