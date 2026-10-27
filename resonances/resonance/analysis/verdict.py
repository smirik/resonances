"""The verdict of one angle of one body: the output model shared by every resonance type.

`status` is the physics, `angle_status` (+ `angle_subtype`) only the geometry of the angle over the record
(librating episodes followed by another regime are `episodic`; `final_regime` says how the record ends),
`reason` the one main explanation, `libration_origin` who makes the bounded oscillation
(with `owner`, the angle that carries it, and `origin_detail`, the concrete cause for a human),
`checks` the values of the named checks, `margins` and `decision_path` how sure the decision is
and which comparisons made it. `status_code` is the old integer, kept for a transition period.

MMR verdicts come from `analysis.mmr.decide`; secular and ZLK verdicts are, until they move to
the same scheme, translated from their integer status by `verdict_from_status_code` (SPEC §5.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Optional

from resonances.resonance.analysis.ledger import LedgerEntry, Margin


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
    EPISODIC = 'episodic'  # 'transient' before 2026-10-17 (`angle_status_from_text` reads old files)
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
    NONE = ''


class Reason(StrEnum):
    DYNAMIC_LIBRATION = 'dynamic_libration'  # a measured force (L4, R0 or the directly averaged F) passes: origin dynamic
    DYNAMIC_EPISODE = 'dynamic_episode'
    # nothing measured the force and the origin is unknown: only the geometry decided (MMR leaf 5u/6; author
    # 2026-10-26, Q3: was dynamic_libration/dynamic_episode with origin unknown, `reason_from_text` reads old files)
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
    # an accepted owner whose intervals cover less than mmr_owner_cover_min of sigma's decisive episode: kept in `owner`,
    # origin none (F7)
    OWNER_OUTSIDE_EPISODE = 'owner_outside_episode'
    # no owner accepted, the best candidate's phases cannot separate it from sigma (phase_unresolved) and its potential
    # route is not applicable: check_mmr_owner not_measured
    OWNER_UNRESOLVED = 'owner_unresolved'
    # the accepted owner leaves open whether sigma's own term also holds Psi (`owner.second_capture`): confidence <= medium
    SECOND_CAPTURE_UNRESOLVED = 'second_capture_unresolved'
    # crossings with every cycle inside the model's separatrix zone (a column, no status; 'separatrix_chaos' before
    # 2026-10-17, `OLD_FLAGS`)
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
    OWNER_SEARCH_SKIPPED = 'owner_search_skipped'
    CONFIDENCE_UNCALIBRATED = 'confidence_uncalibrated'
    PROBABLE_NEAR_SEPARATRIX = 'probable_near_separatrix'  # secular/ZLK only: the old -5


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
    # the calibration cell (status, angle_status, reason): author-labelled verdicts at high or certain, their errors and
    # the Wilson 95 % interval of the accuracy (`confidence.CellRecord`); reported only, they change no level
    cell_n_examples: Optional[int] = None
    cell_n_errors: Optional[int] = None
    cell_accuracy_low_95: Optional[float] = None
    cell_accuracy_high_95: Optional[float] = None
    thin_margins: list[str] = field(default_factory=list)
    failed_support: list[str] = field(default_factory=list)
    caps: list[str] = field(default_factory=list)
    action: Action = Action.NONE
    extend_to_years: Optional[float] = None
    resample_step_yr: Optional[float] = None
    libration_origin: LibrationOrigin = LibrationOrigin.NONE
    origin_detail: OriginDetail | str = OriginDetail.NONE
    owner: str = ''
    multiplet_status: Optional[Status] = None
    checks: dict[str, CheckValue] = field(default_factory=dict)
    margins: dict[str, MarginValue] = field(default_factory=dict)
    decision_path: list[LedgerEntry] = field(default_factory=list)

    @property
    def status_code(self) -> int:
        """The old integer status (temporary, see `status_code`)."""
        return status_code(self.resonance_type, self.status, self.angle_status, self.reason, self.flags)

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
            'extend_to_years': self.extend_to_years,
            'resample_step_yr': self.resample_step_yr,
            'libration_origin': str(self.libration_origin),
            'origin_detail': str(self.origin_detail),
            'owner': self.owner,
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
        return row

    @classmethod
    def from_flat_dict(cls, row: Mapping[str, Any]) -> 'Verdict':
        """Rebuild a verdict from a summary.csv row (restore without recomputation).

        The decision path keeps only the comparison names (the numbers are in the row's columns);
        margins keep their class as DECISIVE (the class is a property of the rule, not stored).
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
            angle_status=angle_status_from_text(cell_text(row.get('angle_status'))),
            angle_subtype=AngleSubtype(cell_text(row.get('angle_subtype'))),
            final_regime=_angle_or_text(cell_text(row.get('final_regime'))),
            final_regime_start_yr=_number(row.get('final_regime_start_yr')),
            orbit_type=cell_text(row.get('orbit_type')),
            confidence=Confidence(cell_text(row.get('confidence'))),
            confidence_raw=Confidence(cell_text(row.get('confidence_raw'))),
            cell_n_examples=_integer(row.get('cell_n_examples')),
            cell_n_errors=_integer(row.get('cell_n_errors')),
            cell_accuracy_low_95=_number(row.get('cell_accuracy_low_95')),
            cell_accuracy_high_95=_number(row.get('cell_accuracy_high_95')),
            reason=reason_from_text(cell_text(row.get('reason')), cell_text(row.get('libration_origin'))),
            flags=flags_from_cell(row.get('flags')),
            thin_margins=split_list(row.get('thin_margins')),
            failed_support=split_list(row.get('failed_support')),
            caps=flags_from_cell(row.get('caps')),
            action=Action(cell_text(row.get('action')) or Action.NONE),
            extend_to_years=_number(row.get('extend_to_years')),
            resample_step_yr=_number(row.get('resample_step_yr')),
            libration_origin=LibrationOrigin(cell_text(row.get('libration_origin'))),
            origin_detail=_enum_or_text(OriginDetail, cell_text(row.get('origin_detail'))),
            owner=cell_text(row.get('owner')),
            multiplet_status=Status(multiplet) if multiplet else None,
            checks={name: CheckValue(cell_text(row.get(name))) for name in ALL_CHECKS if cell_text(row.get(name))},
            margins=margins,
            decision_path=[LedgerEntry.name_only(name) for name in split_list(row.get('decision_path'))],
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


# angle_status spellings of older files -> the current value (author 2026-10-17: 'transient' -> 'episodic', so that
# the angle's word is not the status's)
OLD_ANGLE_STATUS = {'transient': AngleStatus.EPISODIC}


def angle_status_from_text(text: str) -> AngleStatus:
    """The `AngleStatus` of a saved cell, the old spelling ('transient') included."""
    text = str(text).strip()
    return OLD_ANGLE_STATUS.get(text) or AngleStatus(text)


# reasons of older files -> the current value. Before 2026-10-26 (author, Q3) the geometry alone (origin unknown, MMR
# leaf 5u/6) wrote dynamic_libration/dynamic_episode; only the origin column tells those rows from the measured ones.
OLD_GEOMETRIC_REASONS = {'dynamic_libration': Reason.GEOMETRIC_LIBRATION, 'dynamic_episode': Reason.GEOMETRIC_EPISODE}


def reason_from_text(text: str, origin: str = ''):
    """The `Reason` of a saved cell (the text itself when it is no Reason), an old row's dynamic_* with origin unknown
    read as geometric_*."""
    text = str(text).strip()
    if str(origin).strip() == LibrationOrigin.UNKNOWN and text in OLD_GEOMETRIC_REASONS:
        return OLD_GEOMETRIC_REASONS[text]
    return _enum_or_text(Reason, text)


# flag and cap spellings of older files -> the current ones (2026-10-17: separatrix_chaos is a model-zone test, not a
# measurement of chaos)
OLD_FLAGS = {'separatrix_chaos': 'episodes_in_separatrix_zone'}


def flags_from_cell(cell: Any) -> list[str]:
    """A saved flags or caps cell (`split_list`) with the old spellings replaced (`OLD_FLAGS`)."""
    return [OLD_FLAGS.get(item, item) for item in split_list(cell)]


def _angle_or_text(text: str):
    try:
        return angle_status_from_text(text)
    except ValueError:
        return text


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


def default_action(status: Status, reason: Reason | str, confidence: Confidence) -> Action:
    """The action of SPEC §5.6: invalid alias or undersampled -> resample, invalid -> review, uncertain -> extend
    (origin_unknown -> review), otherwise by confidence (certain/high none, medium spot_check, low review)."""
    if status == Status.INVALID:
        return Action.RESAMPLE if reason in (Reason.ALIAS, Reason.UNDERSAMPLED) else Action.REVIEW
    if status == Status.UNCERTAIN:
        return Action.REVIEW if reason == Reason.ORIGIN_UNKNOWN else Action.EXTEND
    if confidence == Confidence.MEDIUM:
        return Action.SPOT_CHECK
    if confidence == Confidence.LOW:
        return Action.REVIEW
    return Action.NONE


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
    flags: Iterable[str] = (),
) -> int:
    """The old integer status of a verdict (DECISIONS: temporary, for consumers of the old codes).

    MMR (SPEC §2.2): resonant 2, transient 1, near_separatrix -3, non_resonant 0 (kinematic,
    induced, subresonance and alias included), uncertain baseline_needed -6, other uncertain -9,
    invalid -99. Secular and ZLK (SPEC §5.7, exact inverse of `verdict_from_status_code`):
    kinematic libration -2, near_separatrix -3 or -5 (flag probable_near_separatrix),
    uncertain insufficient_time -4, uncertain drift -6, other uncertain -9.
    """
    resonance_type = ResonanceType(resonance_type)
    if resonance_type == ResonanceType.MMR:
        if status == Status.UNCERTAIN:
            return -6 if reason == Reason.BASELINE_NEEDED else -9
        return _MMR_CODES[status]
    if status == Status.NON_RESONANT:
        return -2 if reason == Reason.KINEMATIC_LIBRATION else 0
    if status == Status.NEAR_SEPARATRIX:
        return -5 if Flag.PROBABLE_NEAR_SEPARATRIX in set(flags) else -3
    if status == Status.UNCERTAIN:
        if reason == Reason.BASELINE_NEEDED:
            return -4 if angle_status == AngleStatus.INSUFFICIENT_TIME else -6
        return -9
    return _MMR_CODES[status]


# old integer -> (status, angle_status, subtype, reason, origin, action or None = by confidence, keeps confidence)
_LEGACY_TABLE = {
    2: (Status.RESONANT, AngleStatus.LIBRATION, AngleSubtype.NONE, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC, None, True),
    1: (Status.TRANSIENT, AngleStatus.EPISODIC, AngleSubtype.NONE, Reason.DYNAMIC_EPISODE, LibrationOrigin.DYNAMIC, None, True),
    0: (Status.NON_RESONANT, AngleStatus.CIRCULATION, AngleSubtype.NONE, Reason.CIRCULATION, LibrationOrigin.NONE, Action.NONE, True),
    -2: (
        Status.NON_RESONANT,
        AngleStatus.LIBRATION,
        AngleSubtype.NONE,
        Reason.KINEMATIC_LIBRATION,
        LibrationOrigin.KINEMATIC,
        Action.NONE,
        True,
    ),
    -3: (
        Status.NEAR_SEPARATRIX,
        AngleStatus.NEAR_SEPARATRIX,
        AngleSubtype.NONE,
        Reason.SEPARATRIX_ZONE,
        LibrationOrigin.DYNAMIC,
        Action.SPOT_CHECK,
        True,
    ),
    -4: (
        Status.UNCERTAIN,
        AngleStatus.INSUFFICIENT_TIME,
        AngleSubtype.ARC,
        Reason.BASELINE_NEEDED,
        LibrationOrigin.NONE,
        Action.EXTEND,
        False,
    ),
    -5: (
        Status.NEAR_SEPARATRIX,
        AngleStatus.NEAR_SEPARATRIX,
        AngleSubtype.NONE,
        Reason.SEPARATRIX_ZONE,
        LibrationOrigin.DYNAMIC,
        Action.REVIEW,
        False,
    ),
    -6: (Status.UNCERTAIN, AngleStatus.DRIFT, AngleSubtype.NONE, Reason.BASELINE_NEEDED, LibrationOrigin.UNKNOWN, Action.EXTEND, False),
    -9: (Status.UNCERTAIN, AngleStatus.UNDEFINED, AngleSubtype.NONE, None, LibrationOrigin.NONE, Action.EXTEND, False),
    -99: (Status.INVALID, AngleStatus.UNDEFINED, AngleSubtype.UNPHYSICAL, Reason.UNPHYSICAL, LibrationOrigin.NONE, Action.REVIEW, False),
}


def verdict_from_status_code(
    code: int,
    resonance_type: ResonanceType | str,
    confidence: Confidence | str | None = None,
    checks: Optional[Mapping[str, CheckValue]] = None,
    flags: Iterable[str] = (),
) -> Verdict:
    """The verdict of a secular or ZLK angle from its integer status (SPEC §5.7).

    The level is the old classifier's confidence with the flag `confidence_uncalibrated`
    (never `certain`); -5 gets level low and the flag `probable_near_separatrix`; uncertain and
    invalid have no level. -9 is `gate_unavailable` for ZLK and `unusable_series` for secular.
    -1 is produced by no classifier and is refused.
    """
    resonance_type = ResonanceType(resonance_type)
    if resonance_type == ResonanceType.MMR:
        raise ValueError('MMR verdicts come from the MMR analysis, not from an integer status')
    code = int(code)
    if code not in _LEGACY_TABLE:
        raise ValueError(f'No verdict for status code {code}: known codes are {sorted(_LEGACY_TABLE)}')
    status, angle_status, subtype, reason, origin, action, keeps_confidence = _LEGACY_TABLE[code]
    if reason is None:
        reason = Reason.GATE_UNAVAILABLE if resonance_type == ResonanceType.LIDOV_KOZAI else Reason.UNUSABLE_SERIES
    level = Confidence(confidence or Confidence.NONE)
    if level == Confidence.CERTAIN:
        level = Confidence.HIGH
    if code == -5:
        level = Confidence.LOW
    elif not keeps_confidence:
        level = Confidence.NONE
    verdict_flags = list(flags)
    if level != Confidence.NONE and Flag.CONFIDENCE_UNCALIBRATED not in verdict_flags:
        verdict_flags.append(str(Flag.CONFIDENCE_UNCALIBRATED))
    if code == -5:
        verdict_flags.append(str(Flag.PROBABLE_NEAR_SEPARATRIX))
    if action is None:
        action = default_action(status, reason, level)
    return Verdict(
        resonance_type=resonance_type,
        status=status,
        angle_status=angle_status,
        angle_subtype=subtype,
        confidence=level,
        confidence_raw=level,
        reason=reason,
        flags=verdict_flags,
        action=action,
        libration_origin=origin,
        origin_detail=OriginDetail.NONE,
        checks=dict(checks or {}),
    )
