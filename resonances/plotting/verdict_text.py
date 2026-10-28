"""The explanation of a verdict in words, for the right column of the one-page figures (layout A′).

Pure functions: an analysis (MMR, secular or Lidov-Kozai), its CSV rows and the settings in, a `VerdictText` out;
nothing is drawn here and nothing feeds a status. Every number in the text is read from the analysis' own objects or
CSV columns and every threshold from `AnalysisSettings`, so a sentence can be checked against the record
(`tests/resonances/plotting/test_verdict_text.py`).

What the text holds (the author's N1–N4 of 2026-10-16 and the fixes of 16b):

- `badge`: STATUS · angle_status (subtype) · CONFIDENCE, all three always (author, 2026-10-17) — the only place the
  status is printed; `ends`: 'ends: <final regime>' when the record ends in another regime than angle_status, drawn
  as a tag beside the plaque, not inside it;
- `sentences`: what the angle does, why this status (the rule that decided it, in plain words, with its numbers),
  the confidence (what kept it from the next level, with the numbers), the action. 'Over the whole record' is said
  when the libration covers the record; 'episode' only when there are several;
- `steps`: the steps of the status tree in their fixed order, each with what it found (leaf ●, passed ○, not run ◌,
  not reached ·);
- `checks`: every margin and cap the confidence read, with a state: pass ✓, near its threshold ⚠, fail ✗, skipped by
  design – (grey, with the reason), should have been measured but was not ?, not applicable ·; caps ▼;
- `arithmetic`: how the level came out; `info`: tests measured but not read by this verdict;
- `bold_columns` / `grey_columns`: numbers the decision compared / numbers of a test skipped by design;
- `physics` (MMR, W20 CS11): A 'Is σ the own pendulum of <key>?' and B 'What makes the cycles?' (`physics_sections`),
  every number a column (`PhysicsRow.numbers`); the columns it prints leave the numbers below it.

A cap that brings the level down to medium or low names the other reading and what on the figure decides it
(`CAP_ALTERNATIVES`, the author's q1 of 2026-10-19); the W20 ledger entries are printed in words (`LEDGER_WORDS`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from resonances.resonance.analysis.confidence import CAP_CEILING, Cap
from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.mmr.analyse import CENTRE_SCALE_UNION, QS_STALL_ENTRY, SHORT_PERIOD_CAP, SHORT_PERIOD_HARMONIC
from resonances.resonance.analysis.mmr.decide import (
    ALIAS_A,
    B1_AMPLITUDE,
    C5_BRANCH,
    C5_FAR,
    DIRECT_BRANCH,
    F_IN_WELL,
    G4B_NET,
    G5A_ONE_REVOLUTION,
    G6_RANGE,
    INDEPENDENT_CYCLES,
    K2_CORR,
    K2_NET,
    K2_SHARE,
    K2B_SHIFT,
    KIN_VARPI,
    NULL_TEST,
    O1_ONSET,
    OWNER_COVER,
    SHORT_PERIOD_A,
    STRENGTH_DIRECT,
    STRENGTH_DIVISORS,
    STRENGTH_FIRST,
    STRENGTH_THREE_BODY,
    UNION_PROMOTED,
    edge_arc_name,
    edge_monotone_name,
    force_laws,
    kinematic_f_applies,
    strength_upper_name,
    taken_harmonic_law,
)
from resonances.resonance.analysis.mmr.quasi_satellite import QS_FLOOR, QS_VISIT
from resonances.resonance.analysis.orbit_type import centre_moves
from resonances.resonance.analysis.verdict import CONFIDENCE_ORDER, Flag, MarginClass


class CheckState(StrEnum):
    PASS = 'pass'  # full
    THIN = 'thin'  # passed, but near its threshold
    FAIL = 'fail'
    SKIPPED = 'skipped'  # not measurable here by design (the reason is printed)
    MISSING = 'missing'  # should have been measured, was not
    NOT_APPLICABLE = 'not_applicable'


class StepState(StrEnum):
    LEAF = 'leaf'  # the step that set the status
    PASSED = 'passed'  # ran and sent the verdict on
    NOT_RUN = 'not_run'  # not triggered, not possible or not applicable
    NOT_REACHED = 'not_reached'  # after the leaf


ICONS = {
    CheckState.PASS: '✓',
    CheckState.THIN: '⚠',
    CheckState.FAIL: '✗',
    CheckState.SKIPPED: '–',
    CheckState.MISSING: '?',
    CheckState.NOT_APPLICABLE: '·',
}
CAP_ICON = '▼'
# Steps: decided here, passed on to the next step, not run; a step not reached has no icon (its name is pale)
STEP_ICONS = {StepState.LEAF: '●', StepState.PASSED: '↓', StepState.NOT_RUN: '◌', StepState.NOT_REACHED: ''}
# The confidence block, worst first; the quiet rows (skipped by design, n/a) after the passed ones; caps come last
STATE_ORDER = (CheckState.FAIL, CheckState.MISSING, CheckState.THIN, CheckState.PASS, CheckState.SKIPPED, CheckState.NOT_APPLICABLE)
COUNTS_FULL, COUNTS_THIN, COUNTS_NOT_MEASURED, COUNTS_FAILED, COUNTS_NA = 'full', 'thin', 'not measured', 'failed', 'n/a'
LIST_MAX = 3  # reasons named in a sentence; the rest is 'N more'
YEAR, KYR, MYR = 1.0, 1e3, 1e6
DIGITS = 3
DIGITS_MAX = 6  # a comparison is printed with up to this many digits so that it holds as printed (3.49997 < 3.5)
OPS = {'<': '<', '<=': '≤', '>': '>', '>=': '≥', '==': '=', '!=': '≠'}
NEGATED = {'<': '≥', '<=': '>', '>': '≤', '>=': '<', '==': '≠', '!=': '='}


@dataclass
class Step:
    """One step of the status tree as this verdict went through it."""

    key: str
    title: str
    state: StepState
    outcome: str = ''


def _shape(text: str, state: StepState = StepState.PASSED) -> Step:
    """The step 'angle shape' with its outcome."""
    return Step('shape', 'angle shape', state, text)


@dataclass
class CheckRow:
    """One row of the confidence block: a margin (or one part of it), a near-threshold ledger entry, or a cap."""

    key: str
    name: str
    state: CheckState
    detail: str = ''  # the comparison as it came out, or the reason it was not measured
    need: str = ''  # what a full margin needs
    counts: str = ''  # how the confidence counted it: full | thin | not measured | failed | n/a | 'cap ≤ medium'
    note: str = ''  # why it is thin or not measured (printed in the column)
    short: str = ''  # the reason in a few words, for the confidence sentence (default: note, else detail)
    alternative: str = ''  # a cap: the other verdict the figure may show instead (`CAP_ALTERNATIVES`)
    look: str = ''  # a cap: what on the figure decides between the two
    said: bool = False  # the confidence sentence already names this cap's alternative

    @property
    def is_cap(self) -> bool:
        return self.key.startswith('cap:')

    @property
    def icon(self) -> str:
        return CAP_ICON if self.is_cap else ICONS[self.state]


@dataclass
class VerdictText:
    status: str
    angle: str  # angle_status with its subtype, as the badge prints it
    confidence: str
    subtitle: str
    sentences: List[str]
    steps: List[Step]
    checks: List[CheckRow]
    arithmetic: str
    info: List[str] = field(default_factory=list)
    bold_columns: List[str] = field(default_factory=list)
    grey_columns: Dict[str, str] = field(default_factory=dict)
    final_regime: str = ''  # how the record ends, when that differs from angle_status (`final_regime_text`)
    physics: List['PhysicsSection'] = field(default_factory=list)  # MMR: own pendulum (A) and what makes the cycles (B)

    @property
    def badge(self) -> str:
        """'STATUS · angle_status (subtype) · CONFIDENCE' ('TRANSIENT · episodic (recurrent) · MEDIUM'): the three
        always, the status upper-case and hyphenated as the sentences write it; no confidence for uncertain/invalid."""
        status = status_head(self.status).upper()  # NON-RESONANT, as the sentences below write it
        return ' · '.join(part for part in (status, self.angle, self.confidence.upper()) if part)

    @property
    def ends(self) -> str:
        """'ends: <final regime>' beside the plaque, or '' when the record ends in the regime of angle_status."""
        return f'ends: {self.final_regime}' if self.final_regime else ''


# --------------------------------------------------------------------------------------------- numbers


def finite(value: Any) -> bool:
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt(value: Any, digits: int = DIGITS) -> str:
    """A number for a sentence: no exponents, thousands grouped by a thin space, '≈0' below 1e-5, '—' when missing."""
    if not finite(value):
        return '—'
    value = float(value)
    if value == 0:
        return '0'
    size = abs(value)
    if size >= 1000:
        return f'{value:,.0f}'.replace(',', ' ')
    if size >= 100:
        return f'{value:.0f}'
    if size < 1e-5:
        return '≈0'
    if size < 1e-3:
        return f'{value:.2g}'
    return f'{value:.{digits}g}'


def years(value: Any) -> str:
    """A time in yr, kyr or Myr, 3 significant digits."""
    if not finite(value):
        return '—'
    value = float(value)
    for scale, unit in ((MYR, 'Myr'), (KYR, 'kyr')):
        if abs(value) >= scale:
            return f'{fmt(value / scale)} {unit}'
    return f'{value:.3g} yr'


def percent(value: Any) -> str:
    return '—' if not finite(value) else f'{100 * float(value):.0f}%'


def holds(value: Any, op: str, threshold: Any) -> Optional[bool]:
    if not (finite(value) and finite(threshold)):
        return None
    value, threshold = float(value), float(threshold)
    return {
        '<': value < threshold,
        '<=': value <= threshold,
        '>': value > threshold,
        '>=': value >= threshold,
        '==': value == threshold,
        '!=': value != threshold,
    }[op]


def distinct(value: float, threshold: float, digits: int = DIGITS) -> Tuple[str, str]:
    """`value` and `threshold` printed with the fewest digits (>= `digits`, <= DIGITS_MAX) that tell unequal numbers
    apart, so that a strict comparison holds as printed ('3.49997 < 3.5', not '3.5 < 3.5')."""
    shown = fmt(value, digits), fmt(threshold, digits)
    for more in range(digits + 1, DIGITS_MAX + 1):
        if shown[0] != shown[1] or float(value) == float(threshold):
            break
        shown = fmt(value, more), fmt(threshold, more)
    return shown


def compare(value: Any, op: str, threshold: Any, unit: str = '', digits: int = DIGITS) -> str:
    """'value op threshold' with the operator as the comparison came out ('0.669 < 0.7' when '≥ 0.7' failed), printed
    so that it holds as printed (`distinct`)."""
    return _comparison(value, op, threshold, unit, digits)[0]


def _comparison(value: Any, op: str, threshold: Any, unit: str, digits: int) -> Tuple[str, Optional[str]]:
    """`compare`'s text and the value as it printed it (None when the comparison could not be made)."""
    result = holds(value, op, threshold)
    if result is None:
        return f'— (need {OPS[op]} {fmt(threshold, digits)}{unit})', None
    shown = OPS[op] if result else NEGATED[op]
    left, right = distinct(float(value), float(threshold), digits)
    return f'{left}{unit} {shown} {right}{unit}', left


def entry_compare(entry, unit: str = '') -> str:
    """`compare` of a ledger entry with its own operator ('' without an entry)."""
    return '' if entry is None else compare(entry.value, str(entry.op), entry.threshold, unit)


def compare_percent(value: Any, op: str, threshold: Any) -> str:
    """`compare` of two fractions written in percent ('100% ≥ 67%')."""
    result = holds(value, op, threshold)
    if result is None:
        return f'— (need {OPS[op]} {percent(threshold)})'
    return f'{percent(value)} {OPS[op] if result else NEGATED[op]} {percent(threshold)}'


def degrees(rad: Any) -> Optional[float]:
    return math.degrees(float(rad)) if finite(rad) else None


def _text(value: Any) -> str:
    return '' if value is None else str(value)


# what sigma does in a regime (final_regime, the last regime after a jump) for a sentence: no raw enum value there
REGIME_PHRASES = {
    'libration': 'σ librates',
    'episodic': 'σ librates in episodes',
    'circulation': 'σ circulates',
    'drift': 'σ drifts',
    'near_separatrix': 'σ moves near the separatrix',
    'insufficient_time': 'the record is too short to tell what σ does',
    'undefined': 'σ is not defined',
}


def regime_phrase(regime: Any) -> str:
    """'σ circulates', 'the record is too short to tell what σ does': a regime (`AngleStatus` value) in plain words."""
    text = _text(regime)
    return REGIME_PHRASES.get(text, f'σ is in the regime {text.replace("_", " ")}')


def status_head(status: Any) -> str:
    """The status opening its sentence: 'Non-resonant', 'Near-separatrix', 'Resonant'."""
    return str(status).replace('_', '-').capitalize()


def _plural(n: int, word: str, plural: str = '') -> str:
    return f'{n} {word if n == 1 else (plural or word + "s")}'


def episode_accepted(row: Mapping[str, Any]) -> bool:
    """An accepted run of episodes.csv (its `accepted` cell as the CSV or the analysis spells it)."""
    return str(row.get('accepted')).lower() in ('true', '1', 'yes')


def _same_id(a: Any, b: Any) -> bool:
    try:
        return a is not None and b is not None and int(float(a)) == int(float(b))
    except (TypeError, ValueError):
        return False


def _float(value: Any) -> float:
    return float(value) if finite(value) else float('nan')


def _duration(row: Mapping[str, Any]) -> float:
    start, end = row.get('t_start_yr'), row.get('t_end_yr')
    return float(end) - float(start) if finite(start) and finite(end) else float('nan')


# --------------------------------------------------------------------------------------------- shared wording


@dataclass(frozen=True)
class CapText:
    """How the explained column speaks of a cap: what it means (`words`), the other verdict the record may hold and
    what on the figure decides between the two (`alternative`, `decides`; author 2026-10-19 q1: medium is a spot check,
    so a cap that lowers the level names the alternative and where to look; the panel letters are those of the
    combined figure, `mmr_combined.PANELS`), and the margin whose number explains it (`margin`, printed with the cap
    in the confidence sentence)."""

    words: str
    alternative: str = ''
    decides: str = ''
    margin: str = ''


# One row per cap (every cap of `confidence.Cap` has an alternative: tested); CAP_WORDS, CAP_ALTERNATIVES and
# CAP_MARGINS are views of it
CAPS = {
    'three_tp_episode': CapText(
        'the libration rests on three turning points only',
        'no libration yet (insufficient_time): three turning points are one cycle',
        '(a): whether σ_f closes a second cycle about the same centre',
        margin='N',
    ),
    'exit_suspected': CapText(
        'the angle seems to leave the resonance at the end',
        'the libration goes on past the end of the record (or σ leaves for good)',
        '(a) and (i) at the right edge: whether σ_f turns back once more or runs a full revolution after the last turn',
    ),
    'entry_suspected': CapText(
        'the angle seems to enter the resonance at the start',
        'the libration started before the record (or σ is captured only here)',
        '(a) and (i) at the left edge: whether σ_f runs a full revolution before the first turn',
    ),
    'transient_unexplained': CapText(
        'no mechanism explains the transitions',
        'near_separatrix: σ wanders across the separatrix without a capture or an escape',
        '(h): h of the cycles next to each transition against the zone and h = 1; (c): the slips',
    ),
    'one_return_cycle': CapText(
        'the libration is one cycle closed by its phase return (two turning points)',
        'no libration yet (insufficient_time): a half-swing that happened to come back',
        '(a): the one cycle between its two turning points; σ_f must cross the return level back at the rate it left',
    ),
    'force_law_disagrees': CapText(
        'the force law fails while the potential holds',
        'uncertain (origin unknown): the potential holds while ȧ does not follow sin σ',
        '(e): ȧ against R sin(σ_f − φ0) per window and the R² of the halves',
    ),
    'slow_libration_not_excluded': CapText(
        'a slower libration is not excluded',
        'a slower libration of σ about a moving centre (the verdict of the slow scale)',
        '(i): whether the slow centre of σ_f turns back inside the record',
        margin='B5',
    ),
    'planets_missing': CapText(
        'the planets were not saved: no subresonance scan',
        'non_resonant: a member of the multiplet holds the centre (subresonance)',
        '(j) has no members to draw: rerun with the planets saved',
    ),
    'owner_search_skipped': CapText(
        'the owner search was switched off',
        'non_resonant: another resonance drives a (induced)',
        '(b) and (j) draw no owner: rerun with the owner search on',
    ),
    'catalogue_longitudes': CapText(
        'the owner search used catalogue planet longitudes (approximate)',
        'another owner, or none, with the true longitudes of the planets',
        "(b): the owner's angle against σ; rerun with the planets saved",
    ),
    'nodal_subresonance_candidate': CapText(
        'a nodal member may carry the libration',
        'non_resonant: a nodal member of the multiplet holds the centre',
        "(j): the nodal member's centre against σ's",
    ),
    'subresonance_tie': CapText(
        'the member was chosen by its order only',
        'another member of the same order holds the centre',
        '(j): the centres of the tied members, the flatter one owns the libration',
    ),
    'subresonance_t6_only': CapText(
        'the scan ran on the theory trigger alone',
        'no member holds the centre: σ itself librates',
        "(j): whether the member's centre is flatter than σ's",
    ),
    'subresonance_theory_unconfirmed': CapText(
        'the multiplet theory does not confirm the member',
        'σ itself, or another member, holds the centre',
        "(j): the member's centre against the multiplet theory's line",
    ),
    'subresonance_undetermined': CapText(
        'the members cannot be told apart (ϖ_P − ϖ librates)',
        'another member of the multiplet holds the centre (they move together while ϖ_P − ϖ librates)',
        '(j): the members whose centres stay flat with σ',
    ),
    'jitter_low_e': CapText(
        'the p·ϖ term wobbles by more than 45° at short periods (osculating)',
        'invalid (undefined phase): ϖ itself is undefined at small e',
        '(b): the spread of the osculating σ about σ_f',
    ),
    'all_physics_not_measured': CapText(
        'none of force law, potential, zone, owner and strength could be measured',
        'uncertain: nothing measured shows this angle’s own force',
        '(e) and (h) have no force law or energy to read: only the shape of σ in (a) decides',
    ),
    'zone_not_measured': CapText(
        'the separatrix zone was not measured',
        'near_separatrix: the cycles may sit in the separatrix zone',
        '(h): h of each cycle against h = 1',
    ),
    'l2_without_zone': CapText(
        'the amplitude exceeds the thin band without a measured zone',
        'near_separatrix: the large swings may cross the separatrix',
        '(h): h of the widest cycles against h = 1',
        margin='L2',
    ),
    'onset_failed': CapText(
        'a strong a-wave precedes the libration',
        'non_resonant: the cycles continue a forced a-wave that began before the libration',
        '(d): the a-wave before the episode against the one inside it',
        margin='O1',
    ),
    'episodes_in_separatrix_zone': CapText(
        (
            "every cycle lies in the model's separatrix zone and the angle crosses it: the edges of the librating "
            "stretches may be the zone's own crossings, not a capture or an escape"
        ),
        'near_separatrix: the edges of the librating stretches are crossings of the zone, not a capture or an escape',
        '(h): h of the cycles at each edge inside the blue band',
    ),
    'near_separatrix_without_zone': CapText(
        'the pattern alone places σ at the separatrix: the zone was not measured',
        'non_resonant circulation, or a libration close to the separatrix',
        '(c): the slips; (h): h near 1',
    ),
    'geometry_only': CapText(
        "neither the force law, the potential with a nor the strength measures this angle's own force",
        'uncertain: only the shape of σ says libration',
        '(e) and (h): no force law and no energy were measured',
    ),
    'strength_scale_only': CapText(
        "neither the force law nor the potential with a measures this angle's own force; only the order-of-magnitude strength scale (μ_i μ_j e^q) agrees",
        'uncertain: the scale agrees by an order of magnitude only, the angle’s own force is not computed',
        '(e) and (h): no force law and no energy were measured; the physics block: μ_req against μ_i μ_j e^q',
        margin='F',
    ),
    'implausible_strength': CapText(
        "the verdict rests on the strength's order of magnitude, not on the angle's dynamics",
        'a libration whose force the first-order average misses',
        'the physics block: μ_req against μ_th, and whether a forced wave explains the cycles',
    ),
    'episode_before_jump': CapText(
        'the librations precede a jump of a; their potential and zone were not measured',
        'the libration ended with the jump of a: the episodes belong to the earlier regime',
        '(a) and (d): the episodes before the hatched jump (their energy is not measured)',
    ),
    'episode_force_law_mixed': CapText(
        'the force law holds on one librating episode and fails on another',
        'only the witness episode librates in this resonance, the others carry another force',
        '(e): R² of the halves per episode',
    ),
    'near_episode': CapText(
        'one more turning point would make a librating episode',
        'transient: one more turning point would make a librating episode',
        '(a): the grey run with most turning points',
    ),
    'short_period_aliased': CapText(
        'the output step folds short-period terms into slow ones (Δt ≥ P_orb/2 or P_syn/2, or a harmonic folded slower than the period floor)',
        'the turning points are partly the step’s: short-period terms folded into σ_f and a_f',
        '(a): the fast spread about σ_f; a rerun at a step that folds no short period decides',
    ),
    'one_cycle_energy_not_conserved': CapText(
        'the 1:1 libration is one cycle and its energy integral fails',
        'no co-orbital libration yet: the one cycle may be a passage, not motion in the averaged well',
        '(h): the energy of the cycle against the averaged potential; a longer record with a second cycle decides',
    ),
    'action_step_provisional': CapText(
        'the libration was cut at a step of its action (G2a) with one energy cycle on a side, or with one cycle of the larger '
        'swings, whose period law is then read in sample',
        'one libration over the whole window: the step is a large modulation, not two energy levels',
        '(a): the half-amplitudes on each side of the dotted cut; (h): h of the cycles before and after it',
    ),
    'second_capture_unresolved': CapText(
        '',
        'two resonances at once: σ is held by its own term as well (resonant, not only induced)',
        '(j): Ψ/u = σ − (v/u) φ — a sweep of 180° or more that follows no potential excludes it',
    ),
}
CAP_WORDS = {cap: text.words for cap, text in CAPS.items() if text.words}
CAP_ALTERNATIVES = {cap: (text.alternative, text.decides) for cap, text in CAPS.items() if text.alternative}
CAP_MARGINS = {cap: text.margin for cap, text in CAPS.items() if text.margin}
# the cap short_period_aliased named by the condition that fired (`_cap_detail`; the analysis' `short_period_trigger`)
SHORT_PERIOD_TITLES = {
    'nyquist': 'the output step folds short periods into slow ones (Δt {comparison} = min(P_orb, P_syn)/2)',
    'harmonic': 'the output step folds a harmonic of the short periods slower than the period floor',
}


LOWERING_CEILINGS = ('medium', 'low')  # caps at these ceilings name their alternative in the right column


def cap_ceiling(cap: str) -> str:
    try:
        return str(CAP_CEILING[Cap(cap)])
    except ValueError:
        return ''


def cap_rows(caps: Sequence[str], level: str, whole_record: bool = False) -> List[CheckRow]:
    """The caps as rows, the one that sets the level first."""
    order = [str(c) for c in CONFIDENCE_ORDER]

    def rank(cap: str) -> Tuple[bool, int]:
        ceiling = cap_ceiling(cap)
        return ceiling != level, order.index(ceiling) if ceiling in order else len(order)

    rows = []
    for cap in sorted(caps, key=rank):
        words = CAP_WORDS.get(cap, cap.replace('_', ' '))
        if cap == 'three_tp_episode' and whole_record:
            words = 'the libration over the whole record has three turning points only'
        # q1: a cap that brings the level down to medium (or low) names the other reading; a cap at high only reports
        alternative, look = CAP_ALTERNATIVES.get(cap, ('', '')) if cap_ceiling(cap) in LOWERING_CEILINGS else ('', '')
        rows.append(CheckRow(f'cap:{cap}', words, CheckState.FAIL, counts=f'cap ≤ {cap_ceiling(cap)}', alternative=alternative, look=look))
    return rows


def alternative_text(row: CheckRow) -> str:
    """'if not: <the other verdict>; look at <what decides it>' of a cap row ('' without one); 'the other reading: in
    the sentence above' when the confidence sentence already names it."""
    if not row.alternative:
        return ''
    if row.said:
        return 'if not: the other reading in the confidence sentence above'
    return f'if not: {row.alternative}; look at {row.look}' if row.look else f'if not: {row.alternative}'


def margin_counts(margin: Margin, margin_class: Any = None) -> str:
    if margin == Margin.FULL:
        return COUNTS_FULL
    if margin == Margin.THIN:
        return COUNTS_THIN
    if margin == Margin.NOT_MEASURED:
        return COUNTS_NOT_MEASURED
    if margin == Margin.FAIL:
        return 'cap' if margin_class == MarginClass.CAP else COUNTS_FAILED
    return COUNTS_NA


def margin_state(margin: Margin) -> CheckState:
    return {
        Margin.FULL: CheckState.PASS,
        Margin.THIN: CheckState.THIN,
        Margin.FAIL: CheckState.FAIL,
        Margin.NOT_APPLICABLE: CheckState.NOT_APPLICABLE,
    }.get(margin, CheckState.SKIPPED)


def sort_checks(rows: List[CheckRow]) -> List[CheckRow]:
    return sorted(rows, key=lambda row: (row.is_cap, STATE_ORDER.index(row.state)))


def _names(rows: Sequence[CheckRow], limit: int = LIST_MAX) -> str:
    """'name (why)' for the first `limit` reasons, rows with the same reason named together ('a, b (not measured:
    withdrawn by the filter guard)'), then 'N more'."""
    groups: Dict[str, List[str]] = {}
    for row in rows:
        why = row.short or row.note or row.detail
        if row.counts == COUNTS_NOT_MEASURED:
            why = f'not measured: {why}' if why else 'not measured'
        groups.setdefault(why, []).append(row.name)
    shown = [f'{", ".join(names)} ({why})' if why else ', '.join(names) for why, names in list(groups.items())[:limit]]
    hidden = sum(len(names) for names in list(groups.values())[limit:])
    return '; '.join(shown) + (f'; {hidden} more' if hidden else '')


def _bare_names(rows: Sequence[CheckRow], limit: int = LIST_MAX) -> str:
    """The names alone (their numbers are in the check rows below the sentence), then 'N more'."""
    names = [row.name for row in rows]
    hidden = len(names) - limit
    return ', '.join(names[:limit]) + (f' and {hidden} more' if hidden > 0 else '')


def calibration_note(verdict) -> str:
    """The verdict's calibration cell (status, angle_status, reason), reported only (it changes no level): its author
    labels, errors and the Wilson 95 % interval of the accuracy."""
    n, errors = getattr(verdict, 'cell_n_examples', None), getattr(verdict, 'cell_n_errors', None)
    if not finite(n):
        return ''
    if not n:
        return 'Its cell (status, angle, reason) has no author-labelled verdict yet.'
    text = f'Its cell (status, angle, reason) holds {fmt(n)} author-labelled verdicts, {fmt(errors)} wrong'
    low, high = getattr(verdict, 'cell_accuracy_low_95', None), getattr(verdict, 'cell_accuracy_high_95', None)
    if finite(low) and finite(high):
        text += f' (accuracy {percent(low)}–{percent(high)}, 95%)'
    return text + '.'


def confidence_sentence(verdict, checks: Sequence[CheckRow]) -> str:
    """Status → confidence: the level, what capped it, and what kept it from the next level, with the numbers."""
    level, raw = str(verdict.confidence), str(verdict.confidence_raw)
    if not level:
        return ''
    margins = [row for row in checks if not row.is_cap]
    failed = [row for row in margins if row.counts == COUNTS_FAILED]
    soft = [row for row in margins if row.counts in (COUNTS_THIN, COUNTS_NOT_MEASURED)]
    caps = [row for row in checks if row.is_cap]
    if failed:
        raw_text = f'{_plural(len(failed), "support test")} failed: {_names(failed)}'
    elif raw == 'high':
        verb = 'is' if len(soft) == 1 else 'are'
        raw_text = (
            f'nothing failed, but certain needs every margin full and {len(soft)} {verb} not: {_bare_names(soft)}'
            if soft
            else ('nothing failed, but a comparison of the decision lies near its threshold')
        )
    else:
        raw_text = 'every margin is full'
    setting_caps = [row for row in caps if row.counts.split()[-1] == level]
    if setting_caps and level != raw:
        first = setting_caps[0]
        source = next((row for row in margins if row.key == CAP_MARGINS.get(first.key[4:]) and row.detail), None)
        rest = len(setting_caps) - 1
        others = f' (and {rest} more cap{"s" if rest > 1 else ""} at the same level)' if rest else ''
        number = f' ({source.detail})' if source is not None else (f' ({first.short})' if first.short else '')
        text = f'{level.capitalize()}, capped: {first.name}{number}{others}. Without the cap it would be {raw}: {raw_text}.'
        if first.alternative:
            where = f'; the figure decides it in {first.look}' if first.look else ''
            text += f' The other reading: {first.alternative}{where}.'
            first.said = True  # its row does not repeat it
    else:
        text = f'{level.capitalize()}: {raw_text}.'
    note = calibration_note(verdict)
    return f'{text} {note}' if note else text


def arithmetic_line(verdict, checks: Sequence[CheckRow]) -> str:
    if not str(verdict.confidence):
        return 'No confidence level: uncertain and invalid verdicts carry none.'
    margins = [row for row in checks if not row.is_cap]
    failed = sum(1 for row in margins if row.counts == COUNTS_FAILED)
    thin = sum(1 for row in margins if row.counts == COUNTS_THIN)
    unmeasured = sum(1 for row in margins if row.counts == COUNTS_NOT_MEASURED)
    if failed >= 2:
        rule = '≥ 2 failed → low'
    elif failed == 1:
        rule = '1 failed → medium'
    elif thin + unmeasured:
        rule = 'thin or not measured → high'
    else:
        rule = 'all full → certain'
    caps = ', '.join(f'{row.key[4:]} {row.counts[4:]}' for row in checks if row.is_cap) or 'none'
    return (
        f'raw: {failed} failed, {thin} thin, {unmeasured} not measured ({rule}) = {verdict.confidence_raw}  ·  '
        f'caps: {caps}  →  {verdict.confidence}'
    )


def action_sentence(verdict) -> str:
    action = str(verdict.action)
    if action == 'extend':
        if str(verdict.angle_status) == 'insufficient_time':  # N8: a checkpoint, the record is analysed again there
            return f'Action: extend the integration; next check: {years(verdict.extend_to_years)}.'
        return f'Action: extend the integration to {years(verdict.extend_to_years)}.'
    if action == 'resample':
        # a level held down by short_period_aliased resamples at its fold-free step (W21 A; `analyse.short_period_resample`)
        lifts = ' (it folds no short period: lifts the cap short_period_aliased)' if _short_period_resample(verdict) else ''
        return f'Action: rerun with an output step of {fmt(verdict.resample_step_yr)} yr{lifts}.'
    return {'spot_check': 'Action: glance at the figure.', 'review': 'Action: review the figure and the numbers.'}.get(action, '')


def _short_period_resample(verdict) -> bool:
    """A resample of a verdict with a level whose cap short_period_aliased holds it down (alias and undersampled are
    invalid and carry no level)."""
    return bool(str(verdict.confidence)) and 'short_period_aliased' in verdict.caps


def angle_text(verdict) -> str:
    """angle_status as the badge prints it: 'angle_status (subtype)'."""
    subtype = str(verdict.angle_subtype)
    return str(verdict.angle_status) + (f' ({subtype})' if subtype else '')


def final_regime_text(verdict) -> str:
    """The final regime when it differs from angle_status (the tag 'ends: …' beside the plaque), else ''."""
    final = _text(getattr(verdict, 'final_regime', ''))
    return final if final and final != str(verdict.angle_status) else ''


# --------------------------------------------------------------------------------------------- episodes


@dataclass
class Episodes:
    """The runs of turning points of one record (episodes.csv rows) and the record's span."""

    rows: Sequence[Mapping[str, Any]]
    span_yr: float
    decisive_id: Any = None

    @property
    def accepted(self) -> List[Mapping[str, Any]]:
        return [row for row in self.rows if episode_accepted(row)]

    @property
    def rejected(self) -> List[Mapping[str, Any]]:
        return [row for row in self.rows if not episode_accepted(row)]

    @property
    def decisive(self) -> Optional[Mapping[str, Any]]:
        found = next((row for row in self.rows if _same_id(row.get('episode_id'), self.decisive_id)), None)
        if found is None and self.accepted:
            found = max(self.accepted, key=_duration)
        return found

    @property
    def longest(self) -> Optional[Mapping[str, Any]]:
        return max(self.accepted, key=_duration) if self.accepted else None

    def covered_fraction(self) -> float:
        if not finite(self.span_yr) or self.span_yr <= 0:
            return float('nan')
        return sum(_duration(row) for row in self.accepted) / self.span_yr

    def rejected_text(self) -> str:
        runs = self.rejected
        if not runs:
            return ''
        reasons = sorted({_text(row.get('reject_reason')).replace('_', ' ') for row in runs} - {''})
        return f'{_plural(len(runs), "run")} of turning points not accepted' + (f' ({", ".join(reasons)})' if reasons else '')


def libration_numbers(row: Mapping[str, Any], centre: bool = True) -> str:
    """'about a centre at 183°: 9 turning points, period 193 yr, half-amplitude 161° (max 175°)'."""
    parts = []
    if centre and finite(row.get('centre_mod_2pi_deg')):
        value = float(row['centre_mod_2pi_deg']) % 360
        parts.append(f'about {0 if value > 359.5 else round(value):.0f}°')
    numbers = [f'{fmt(row.get("n_tp"))} turning points', f'period {years(row.get("period_med_yr"))}']
    if finite(row.get('half_amp_med_deg')):
        amp = f'half-amplitude {fmt(row.get("half_amp_med_deg"))}°'
        if finite(row.get('half_amp_max_deg')):
            amp += f' (max {fmt(row.get("half_amp_max_deg"))}°)'
        numbers.append(amp)
    return (' '.join(parts) + ': ' if parts else '') + ', '.join(numbers)


def coverage_text(row: Mapping[str, Any], settings) -> str:
    """'it misses the start by 8.6 and the end by 4.9 libration periods (at most 1 each covers the record)'."""
    start, end = row.get('coverage_start_gap_P'), row.get('coverage_end_gap_P')
    if not (finite(start) and finite(end)):
        return ''
    limit = settings.libration_coverage_periods
    return (
        f'it misses the start by {fmt(start, 2)} and the end by {fmt(end, 2)} libration periods '
        f'(a libration that covers the record misses each by ≤ {fmt(limit)})'
    )


# --------------------------------------------------------------------------------------------- MMR


MMR_MARGIN_NAMES = {
    'L1': 'enough turning points at P_c ×½, 1, 2',
    'L2': 'cycles away from the separatrix',
    'L3': 'dσ/dt = c (a − a*): the rate follows a',
    'L4': 'ȧ = K sin(σ − φ0): force law',
    'L5': 'one pendulum: T0 stable',
    'L6': 'centre stationary',
    'L7': 'covers the record',
    'L8a': 'the output step resolves σ',
    'L8b': 'the filter passes the cycle',
    'O1': 'no a-wave before the libration',
    'R0': 'energy fits the potential from a',
    'Z': 'separatrix zone measured',
    'S': 'no multiplet member holds the centre',
    'W': 'no other resonance drives a',
    'F': 'strength plausible (×10 of theory)',
    'C1': 'centre passes ≥ 10 rev',
    'C2': 'no turning points at any cutoff',
    'C3': 'dσ/dt smooth (one sign, no stalls)',
    'C4': 'a-wave as a circulation predicts',
    'C5': 'far from a* and one alias branch',
    'K1': 'slow σ follows p ϖ',
    'B5': 'a slower libration excluded',
    'N': 'not one turning point short of an episode',
}
# check_mmr_<name> -> the margin that grades it (a check with a margin is read through the margin)
CHECK_MARGIN = {
    'a_response': 'L4',
    'potential': 'R0',
    'subresonances': 'S',
    'owner': 'W',
    'strength': 'F',
    'one_pendulum': 'L5',
    'onset': 'O1',
    'separatrix_zone': 'Z',
    'identity': 'L3',
    'filter': 'L8',
    'slow_libration': 'B5',
    'kinematic': 'K1',
}
TREE_CHECKS = ('alias', 'baseline', 'a_amplitude', 'subresonances', 'owner')
# check -> the ledger columns through which the status tree reads it (a check read there is not 'info')
CHECK_COLUMNS = {
    'a_response': ('rsp_force_R2_h1', 'rsp_force_R2_h2', 'rsp_phi0_max_dev_deg', 'rsp_force_multi_coherence', 'rsp_coorb_K_over_K_theory'),
    'potential': ('sep_VE_VF_agree_fraction', 'rsp_coorb_energy_R2'),
    'kinematic': ('kin_corr_rate', 'rsp_slow_adot_R2'),
}
CHECK_WORDS = {
    'slow_libration': 'B5 slower libration',
    'kinematic': 'ϖ-kinematic test',
    'exit': 'exit/entry',
    'one_pendulum': 'one pendulum',
    'onset': 'onset',
    'separatrix_zone': 'zone',
    'strength': 'strength',
    'identity': 'identity',
    'a_response': 'force law',
    'potential': 'potential',
    'filter': 'filter guard',
    'baseline': 'baseline',
    'alias': 'alias',
    'a_amplitude': 'B1',
}
REJECT_WORDS = {
    'range': 'its centre swings as far as σ’s',
    'rate': 'its centre drifts',
    'pass': 'its centre makes a full turn',
    'A_not_measured': 'its force phase was not measured',
    'A_coherence': 'its force phase is not fixed',
    'A_gain': 'it explains ȧ no better than σ',
    'B_centre': 'its centre is not at 0 or π',
    'C_tie': 'another member fits as well',
    'D_theory': 'the multiplet theory disagrees',
    'E_t0': 'its T0 disagrees',
    'E_strength': "its strength factor is below the strongest term's share",
    'planets_missing': 'planets not saved',
    'undetermined': 'members indistinguishable',
    'nodal': 'a nodal member',
}
THEORY_WORDS = {
    'subresonance_theory_crossing': 'the orbits can cross',
    'subresonance_theory_round_off': 'coefficients below round-off',
    'subresonance_theory_range': 'e or e_P outside (0, 1)',
    'subresonance_theory_disagrees': 'measured and disagrees',
}
# Columns a skipped test leaves on the figure (printed grey, 'not used')
L4_COLUMNS = ('rsp_force_R2_h1', 'rsp_force_R2_h2', 'rsp_force_phi0_deg', 'rsp_phi0_coherence', 'rsp_phi0_max_dev_deg', 'rsp_phi0_tol_deg')


class MMRFacts:
    """Shortcuts over one MMR analysis and its rows (summary ⋈ numbers, episodes)."""

    def __init__(self, analysis, settings, row: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]):
        self.a = analysis
        self.v = analysis.verdict
        self.s = settings
        self.row = row
        checks = {key[len('check_mmr_') :]: str(value) for key, value in self.v.checks.items() if key.startswith('check_mmr_')}
        # R-ALIAS-A (W22): a failed L4/L3 the tree and the level read as not_measured is shown so (its measured fail
        # printed as not used); the verdict's margins and checks keep the measured fail (`decide.margins_as_tree_read`)
        self.alias_unmeasured = alias_unmeasured(analysis, row, checks)
        self.margins = alias_margins(self.v.margins, self.alias_unmeasured)
        self.checks = {name: 'not_measured' if name in self.alias_unmeasured else value for name, value in checks.items()}
        self.rule = _text(row.get('rec_rule'))
        response = analysis.response
        self.coorbital = response is not None and response.force_law == 'coorbital'
        self.derived = bool(analysis.parent)
        span = row.get('smp_span_yr')
        self.episodes = Episodes(episodes, float(span) if finite(span) else float('nan'), analysis.decisive_episode_id)
        self.path = {entry.name: entry for entry in self.v.decision_path}
        self.withdrawn = {q for q in _text(row.get('flt_sensitive_quantities')).split(';') if q}
        self.v_columns = read_columns(self.v)
        # F of a kinematic non_resonant verdict reads in the inverted sense (W22 R-FK): the analysis' own predicate
        self.f_kinematic = kinematic_f_applies(self.v.status, self.v.libration_origin, analysis.strength)
        # margin id -> why its measured number was not used: withdrawn by the filter guard, or read as not measured by
        # R-ALIAS-A (`unused_margins`)
        self.unused = unused_margins(self)

    @property
    def whole_record(self) -> bool:
        return str(self.v.angle_status) == 'libration'

    def entry(self, name: str):
        return self.path.get(name)

    def amplitude_max_deg(self) -> Optional[float]:
        episode = self.episodes.decisive
        return float(episode['half_amp_max_deg']) if episode is not None and finite(episode.get('half_amp_max_deg')) else None


def read_columns(verdict) -> List[str]:
    """The columns the decision compared (with a value), in the order of its path: bold on the figure."""
    return list(dict.fromkeys(entry.column for entry in verdict.decision_path if entry.column and entry.value is not None))


def _held(f: MMRFacts, name: str):
    """The decision-path entry `name` when its comparison held, else None."""
    entry = f.entry(name)
    return entry if entry is not None and entry.result else None


def alias_unmeasured(analysis, row: Mapping[str, Any], checks: Mapping[str, str]) -> Tuple[str, ...]:
    """The checks (a_response, identity) that R-ALIAS-A read as not_measured (`decide.short_period_unmeasured`,
    column rsp_short_period_unmeasured), each only where its measured check failed."""
    extra = getattr(analysis, 'extra_columns', None) or {}
    text = _text(extra.get('rsp_short_period_unmeasured', row.get('rsp_short_period_unmeasured')))
    return tuple(name for name in text.split(';') if name and checks.get(name) == 'fail')


def alias_margins(margins: Mapping[str, Any], unmeasured: Sequence[str]) -> Mapping[str, Any]:
    """The margins as the confidence read them: a failed margin of an R-ALIAS-A check counts as not_measured."""
    if not unmeasured:
        return margins
    read = dict(margins)
    for name in unmeasured:
        margin_id = CHECK_MARGIN.get(name)
        value = read.get(margin_id)
        if value is not None and value.margin == Margin.FAIL:
            read[margin_id] = replace(value, margin=Margin.NOT_MEASURED)
    return read


def alias_reason(f: MMRFacts) -> str:
    """Why R-ALIAS-A read a failed a-based check as not measured, with the scatter of a that decided it."""
    entry = f.entry(ALIAS_A)
    scatter = f', its short-period scatter {entry_compare(entry)} × the a swing' if entry else ''
    return f'failed on a, read as not measured by R-ALIAS-A: the step folds short periods into a{scatter}'


def unused_margins(f: MMRFacts) -> Dict[str, str]:
    """margin id -> why its measured number was not used: the filter guard withdrew it (DELTA 13a: a quantity it reads
    changes at 2 P_c, `flt_sensitive_quantities`), else R-ALIAS-A read its failed check as not measured
    (`alias_unmeasured`)."""
    from resonances.resonance.analysis.mmr.preprocess import SENSITIVE_MARGINS

    unused = {}
    for margin_id in dict.fromkeys(m for q in sorted(f.withdrawn) for m in SENSITIVE_MARGINS.get(q, ())):
        changed = [q for q in sorted(f.withdrawn) if margin_id in SENSITIVE_MARGINS.get(q, ())]
        unused[margin_id] = f'withdrawn by the filter guard: the {", ".join(changed)} change at 2 P_c'
    for name in f.alias_unmeasured:
        unused.setdefault(CHECK_MARGIN.get(name), alias_reason(f))
    return unused


def _filter_shapes(f: MMRFacts, margin_id: str) -> bool:
    """A shape margin of a libration faster than 2 P_c (filter_too_slow): the filter shapes it, nothing was measured."""
    from resonances.resonance.analysis.mmr.preprocess import SENSITIVE_MARGINS

    return f.a.prep is not None and f.a.prep.filter_too_slow and margin_id in SENSITIVE_MARGINS.get('shape', ())


def force_windows_text(f: MMRFacts) -> str:
    """Why no force window fits (52356): the longest librating stretch in cycles of the record's P_cyc (the period the
    windows are cut with) against one window of MMR_FORCE_WINDOW_CYCLES cycles, or, when it holds one, against the span
    of MMR_FORCE_MIN_WINDOWS windows sliding by MMR_FORCE_WINDOW_STEP (`owner.windows_needed_span`)."""
    from resonances.resonance.analysis.mmr.owner import windows_needed_span

    s, period = f.s, _float(f.row.get('geo_period_cycle_yr'))
    longest = f.episodes.longest
    window = s.mmr_force_window_cycles
    if longest is None or not (finite(period) and period > 0):
        return f'fewer than {s.mmr_force_min_windows} force windows of {fmt(window)} cycles'
    length = _duration(longest)
    stretch = 'the libration' if len(f.episodes.accepted) == 1 else 'the longest librating stretch'
    own = _float(longest.get('period_med_yr'))
    own_text = f', {fmt(length / own, 2)} of its own {years(own)}' if finite(own) and own > 0 and abs(own - period) > 0.01 * period else ''
    cycles = length / period
    if cycles < window:
        shown, needed = distinct(cycles, window, 2)
        against = f'{needed} of one force window'
    else:
        shown, needed = distinct(cycles, windows_needed_span(period, s) / period)
        step = fmt(s.mmr_force_window_step)
        against = f'{needed} spanned by {s.mmr_force_min_windows} force windows of {fmt(window)} cycles (step {step} window)'
    return (
        f'{stretch} lasts {years(length)} = {shown} cycles < {against} '
        f'(cycles of P_cyc {years(period)}{own_text}); the test needs {s.mmr_force_min_windows} windows'
    )


def earlier_regime_text(f: MMRFacts) -> str:
    """G1e (`decide._g1e`): the status rests on episodes before the last jump of a while the response and the energy
    were measured on the last regime after it, which holds no episode: 'the force law is measured only in the last
    regime after the jump (t0–t1), where σ circulates; 4 episodes before the jump are not measured'. '' otherwise."""
    g = f.a.geometry
    if g is None or g.episodes or not g.episodes_before_jump or str(Flag.LAST_REGIME_AFTER_JUMP) not in g.flags:
        return ''
    if f.a.response is not None and str(f.a.response.force_source) != 'record':  # rule R measured them
        return ''
    final = _text(f.v.final_regime)
    where = regime_phrase(final) if final else 'σ has no episode'
    n = len(g.episodes_before_jump)
    return (
        f'measured only in the last regime after the jump ({years(g.regime_start_yr)}–{years(g.window[1])}), where {where}; '
        f'{_plural(n, "episode")} before the jump {"is" if n == 1 else "are"} not measured'
    )


def measured_on(f: MMRFacts, source_field: str = 'force_source') -> str:
    """Where rule R took a check from (`response.deciding_response`, the response's `*_source`): 'measured on
    witness #4 (14.2 kyr–15.2 kyr, the one of 2 witnesses with most turning points)', 'pooled over the 3 episodes
    before the jump (1.2 kyr–9.8 kyr)', or '' for the record's pooled measurement."""
    from resonances.resonance.analysis.mmr.response import witnesses

    response = f.a.response
    source = str(getattr(response, source_field, '')) if response is not None else ''
    if source == 'witness':
        found = witnesses(response)
        witness = next((e for e in found if e.episode_id == response.witness_episode_id), None)
        if witness is None:
            return ''
        others = f', the one of {len(found)} witnesses with most turning points' if len(found) > 1 else ''
        return f'measured on witness #{witness.episode_id} ({years(witness.t_start_yr)}–{years(witness.t_end_yr)}{others})'
    pooled = getattr(response, 'before_jump_response', None)
    if source == 'before_jump' and pooled is not None:
        n = sum(1 for e in response.episode_responses if e.before_jump)
        return f'pooled over the {_plural(n, "episode")} before the jump ({years(pooled.t_start_yr)}–{years(pooled.t_end_yr)})'
    return ''


def _with_source(text: str, f: MMRFacts, source_field: str = 'force_source') -> str:
    where = measured_on(f, source_field)
    return f'{text}; {where}' if text and where else text


def _skip_force_law(f: MMRFacts) -> Tuple[CheckState, str]:
    if f.a.response is None:
        return CheckState.MISSING, 'no response of a was measured'
    earlier = earlier_regime_text(f)
    if earlier:
        return CheckState.SKIPPED, f'it is {earlier}'
    where = measured_on(f)
    if where:  # rule R, G1e: pooled over the episodes before the jump, and not measured there either
        return CheckState.SKIPPED, f'{where}: {_force_not_measured(f)[1]}'
    return _force_not_measured(f)


def _force_not_measured(f: MMRFacts) -> Tuple[CheckState, str]:
    """Why the force law of the response was not measured: the amplitude, the windows, the phase bound."""
    s, response = f.s, f.a.response
    amp = f.amplitude_max_deg()
    if amp is not None and amp < s.mmr_a_response_min_amp_deg and not f.coorbital:
        return CheckState.SKIPPED, (
            f'half-amplitude {fmt(amp)}° < {fmt(s.mmr_a_response_min_amp_deg)}°: sin σ ≈ σ, the law cannot be told from the identity'
        )
    if len(response.windows) < s.mmr_force_min_windows:
        pooled = response.before_jump_response if str(response.force_source) == 'before_jump' else None
        if pooled is not None:
            return CheckState.SKIPPED, (
                f'{_plural(len(pooled.windows), "force window")} of {fmt(s.mmr_force_window_cycles)} cycles '
                f'(P {years(pooled.period_yr)}) < {s.mmr_force_min_windows}'
            )
        return CheckState.SKIPPED, force_windows_text(f)
    if not finite(response.phi0_tol_rad):
        return CheckState.SKIPPED, 'the modulation can turn the phase of the force by π (Σε ≥ 1), so its stability has no bound'
    return CheckState.MISSING, 'no force-law number'


def _skip_potential(f: MMRFacts) -> Tuple[CheckState, str]:
    s, energy, geometry = f.s, f.a.energy, f.a.geometry
    amp = f.amplitude_max_deg()
    if energy is None:
        return CheckState.MISSING, 'no energy analysis'
    law = getattr(energy, 'potential_law', None)
    if law is not None and str(law.potential_check) == 'not_measured':  # the 1:1 law decided R0 (W20 round D)
        # '?', not the pale '–' of a test skipped by design: the confidence counts it as not measured (W20 round E)
        if str(law.energy_check) == 'fail' and str(law.force_check) == 'pass':
            return CheckState.MISSING, (
                f'the 1:1 energy integral fails (R² {fmt(law.energy_r2, 2)}) while the force law follows the same '
                'potential: E is not conserved within the cycles (encounters, a drifting e, i, ω)'
            )
        return CheckState.MISSING, 'the 1:1 energy integral was not measured'
    ve = energy.potential_e.r2 if energy.potential_e is not None else float('nan')
    if not (finite(ve) and ve >= s.separatrix_ve_r2_min):
        return CheckState.SKIPPED, f'V_E explains {fmt(ve, 2)} < {fmt(s.separatrix_ve_r2_min)} of dσ/dt²: no potential to test'
    if amp is not None and amp < s.mmr_a_response_min_amp_deg and not (geometry is not None and geometry.revolutions):
        return CheckState.SKIPPED, (
            f'half-amplitude {fmt(amp)}° < {fmt(s.mmr_a_response_min_amp_deg)}°: V_E ≈ V_F holds for any swing a carries'
        )
    if str(energy.sigma_only) == 'not_measured':
        return CheckState.SKIPPED, 'fewer than two cycles: no period law'
    return CheckState.MISSING, 'no window correlation of V_E and V_F'


def _not_triggered(f: MMRFacts) -> str:
    """Why no subresonance scan ran (no trigger T1, T2, T5, T6 fired): 'the centre is stationary', or, when the centre
    moves by the header's test (`decisive_centre_moves`), what the trigger T1 read instead: the bands overlap."""
    if decisive_centre_moves(f):
        return 'not triggered: the cycles’ bands overlap (no trigger fired)'
    return 'not triggered: the centre is stationary'


def _skip_subresonances(f: MMRFacts) -> Tuple[CheckState, str]:
    sub = f.a.subresonances
    if f.derived:
        return CheckState.NOT_APPLICABLE, 'a derived row is not scanned again'
    if sub is None:
        return CheckState.NOT_APPLICABLE, _not_triggered(f)
    if str(Flag.PLANETS_MISSING) in sub.flags:
        return CheckState.MISSING, 'the planets were not saved'
    if str(Flag.NODAL_SUBRESONANCE_CANDIDATE) in sub.flags:
        return CheckState.SKIPPED, 'a nodal member came first: nodal members are not measured'
    if sub.undetermined and getattr(sub, 'beta_librates', True):
        return CheckState.SKIPPED, 'ϖ_P − ϖ librates: the members cannot be told apart'
    if sub.undetermined:
        return CheckState.SKIPPED, 'σ swings below the force-law floor: the members cannot be told apart'
    return CheckState.MISSING, 'the scan gave no verdict'


# own_flags -> why the owner search has no verdict (by design, or data missing)
OWNER_FLAG_REASONS = {
    'owner_no_longitudes': (CheckState.MISSING, "the planets' longitudes were not saved"),
    'owner_too_few_windows': (CheckState.SKIPPED, 'the libration holds too few force windows for the owner fit'),
    'owner_no_period': (CheckState.SKIPPED, 'no cycle period to cut the windows with'),
    'owner_unwrap_failed': (CheckState.MISSING, "a candidate's angle could not be unwrapped"),
    'owner_search_off': (CheckState.SKIPPED, 'the owner search is switched off'),
    'owner_retrograde': (CheckState.SKIPPED, 'no owner search for a retrograde angle'),
    'owner_catalogue_ambiguous': (CheckState.MISSING, 'a candidate on catalogue λ (approx.) is too close to call'),
}


def _skip_owner(f: MMRFacts) -> Tuple[CheckState, str]:
    owner = f.a.owner
    if owner is None:
        return CheckState.NOT_APPLICABLE, 'not run on a derived row' if f.derived else 'not run'
    for flag, (state, why) in OWNER_FLAG_REASONS.items():
        if flag in owner.flags:
            return state, why
    return CheckState.MISSING, 'the owner search gave no number'


def _skip_strength(f: MMRFacts) -> Tuple[CheckState, str]:
    strength = f.a.strength
    if strength is None:
        return CheckState.MISSING, 'no strength'
    if _text(strength.not_measured_reason):
        return CheckState.SKIPPED, strength.not_measured_reason.replace('_', ' ')
    if f.f_kinematic and str(strength.check) == 'pass':
        return CheckState.SKIPPED, 'the term could make this period: the strength does not exclude a resonance'
    if not strength.eq_legal:
        crosses = list(getattr(f.a.context, 'crosses', ()) or ())
        why = f'the orbit crosses {", ".join(crosses)}' if crosses else f'mean e ≥ {fmt(f.s.mmr_strength_e_max)}'
        return CheckState.SKIPPED, f'e^q is not a valid scale here ({why})'
    return CheckState.SKIPPED, 'no required strength (no T0)'


# margin -> why it was not measured: a function of the facts, or a fixed reason (by design)
def _skip_c4(f: MMRFacts) -> Tuple[CheckState, str]:
    """C4 not_measured: no force fit, or (W22 R-C4) a failed C4 under a force law that explains nothing anywhere."""
    response = f.a.response
    if response is None:
        return CheckState.MISSING, 'no force fit'
    # the laws of decide.force_law_rejected: the halves and the multiplet (`force_laws`), then the whole window
    laws = [value for _, value, _ in force_laws(response)] + [getattr(response, 'force_r2_whole', None)]
    laws = [x for x in laws if finite(x)]
    if not laws:
        return CheckState.MISSING, 'no force fit'
    best = compare(max(laws), '<', f.s.mmr_force_r2_fail)
    return CheckState.SKIPPED, f'the force law explains nothing (best R² {best}): K is no measured force'


SKIP_REASONS = {
    'C4': _skip_c4,
    'L4': _skip_force_law,
    'R0': _skip_potential,
    'S': _skip_subresonances,
    'W': _skip_owner,
    'F': _skip_strength,
    'L6': 'one cycle gives one centre: nothing to compare',
    'L5': 'one amplitude regime only',
    'O1': 'no a-band before the libration (it starts the record)',
    'Z': 'the zone was not integrated',
    'L3': 'the libration is shorter than 2 periods',
    'B5': 'no slow-libration test for this shape',
}


def why_not_measured(f: MMRFacts, margin_id: str) -> Tuple[CheckState, str]:
    """(SKIPPED | MISSING | NOT_APPLICABLE, reason) of a not_measured margin: skipped by the method's design, missing
    data (or measured and not used: `MMRFacts.unused`), or a test that does not apply."""
    if _filter_shapes(f, margin_id):
        return CheckState.MISSING, 'the libration is faster than 2 P_c: the filter shapes it'
    if margin_id in f.unused:
        return CheckState.MISSING, f.unused[margin_id]
    reason = SKIP_REASONS.get(margin_id)
    if callable(reason):
        return reason(f)
    if reason:
        return CheckState.SKIPPED, reason
    return CheckState.MISSING, 'not measured'


def _force_detail(f: MMRFacts) -> Tuple[str, str]:
    response, s = f.a.response, f.s
    if response is None:
        return '', ''
    halves = f'R² {fmt(response.force_r2_h1)}/{fmt(response.force_r2_h2)}'
    if f.coorbital and response.coorbital is not None:
        law, limit = response.coorbital, s.mmr_coorbital_scale_ratio_max
        inside = finite(law.scale) and 1 / limit <= law.scale <= limit
        return (
            _with_source(
                f'{halves}, K/K_theory {fmt(law.scale)} {"within" if inside else "outside"} [{fmt(1 / limit, 2)}, {fmt(limit)}]', f
            ),
            f'R² ≥ {fmt(s.mmr_force_r2_full)} in both halves; K/K_theory in [{fmt(1 / limit, 2)}, {fmt(limit)}]',
        )
    harmonic = taken_harmonic_law(response)
    if harmonic is not None:  # CS12: the well is no sinusoid, the force law holds with N harmonics of σ
        n = harmonic.n_harmonics
        text = (
            f'{n}-harmonic law R² {compare(harmonic.r2, ">=", s.mmr_force_r2_fail)}, '
            f'{"full" if harmonic.r2 >= s.mmr_force_r2_full else "thin"}: {compare(harmonic.r2, ">=", s.mmr_force_r2_full)} '
            f'(one harmonic: {halves}); '
            f'its coefficients coherent {compare(harmonic.coherence, ">=", s.mmr_force_coef_coh_min)}; '
            f'R² above ȧ shifted in time by {compare(harmonic.null_gain, ">=", s.mmr_force_null_gain)}; '
            f'one half predicts the other with R² {compare(harmonic.oos_r2, ">=", s.mmr_force_r2_fail)}'
        )
        return _with_source(text, f), (
            f'R² ≥ {fmt(s.mmr_force_r2_fail)} (full from {fmt(s.mmr_force_r2_full)}) with up to {s.mmr_force_harmonics_max} '
            'harmonics of σ (V(σ) need not be a sinusoid), coherent, above the null, on independent halves'
        )
    low = min(response.force_r2_h1, response.force_r2_h2) if finite(response.force_r2_h1) and finite(response.force_r2_h2) else None
    text = f'{halves}, lower {compare(low, ">=", s.mmr_force_r2_full)}' if low is not None else halves
    if finite(response.phi0_tol_rad) and finite(response.phi0_max_dev_rad):
        text += f'; φ0 spread {compare(degrees(response.phi0_max_dev_rad), "<=", degrees(response.phi0_tol_rad), "°")}'
    null = f.entry(NULL_TEST)
    if null is not None and null.result is not None:
        text += f'; {ledger_words(NULL_TEST)}: R² gain {_entry_text(f, NULL_TEST)}'
    return _with_source(text, f), f'R² ≥ {fmt(s.mmr_force_r2_full)} in both halves, φ0 fixed'


def _potential_detail(f: MMRFacts) -> Tuple[str, str]:
    energy, s = f.a.energy, f.s
    law = getattr(energy, 'potential_law', None)  # R0 printed from the law it was read from (W20 round D)
    if law is not None:
        r2 = law.energy_r2
        return f'1:1 energy integral R² {compare(r2, ">=", s.mmr_force_r2_fail)}', f'R² ≥ {fmt(s.mmr_force_r2_fail)}'
    if energy is None:
        return '', ''
    return (
        f'V_E ≈ V_F in {compare_percent(energy.ve_vf_agree_fraction, ">=", s.separatrix_vevf_fraction)} of the windows',
        f'≥ {percent(s.separatrix_vevf_fraction)} of the windows with r ≥ {fmt(s.separatrix_vevf_r_min)}',
    )


def _best_candidate(candidates):
    return min(candidates, key=lambda c: c.rank, default=None)


OWNER_REASON_WORDS = {
    'phase_incoherent': 'its phase φ0 does not hold, coherence {coherence}',
    'unwrap_failed': 'its angle could not be unwrapped',
    'too_few_windows': 'too few force windows',
    'psi_share_high': 'σ − its angle keeps too much of the swing',
    'ripple_only': 'it explains only a ripple on σ',
    'not_better_than_sigma': 'it fits a no better than σ itself',
    'catalogue_ambiguous': 'on catalogue longitudes its coherence sits at a threshold',
    'rate_outside_band': 'its forced wave would lie outside the a band',
    'phase_unresolved': 'the phases cannot tell it from σ',
    'potential_scale_out': 'its averaged potential needs a scale outside the band',
    'potential_unconfirmed': 'the averaged force disagrees with the fit',
}


def _owner_rejection(f: MMRFacts, candidate) -> str:
    """Why the best owner candidate was not accepted: its R² against the bar when that was the reason, else the
    test it failed with its number where the candidate keeps one (433 1R-5V+7: R² 0.73, phase incoherent)."""
    r2 = f'R² {fmt(candidate.r2_median, 2)}'
    reason = str(candidate.reason or '')
    if reason in ('r2_low', ''):  # owner.py rejects on mmr_owner_r2_min (W20 round D, review C #4)
        return f'R² {compare(candidate.r2_median, "<", f.s.mmr_owner_r2_min, digits=2)}'
    words = OWNER_REASON_WORDS.get(reason, reason.replace('_', ' '))
    return f'{r2}, but ' + words.format(coherence=fmt(getattr(candidate, 'phi0_coherence', None), 2))


def _member_rejection(f: MMRFacts, candidate) -> str:
    """Why the best member was not accepted, with the number of its criterion where the scan keeps one."""
    sub, s = f.a.subresonances, f.s
    own = getattr(sub, 'centre0_range_rad', float('nan'))
    if candidate.reason == 'range' and finite(candidate.centre_range_rad) and finite(own) and own > 0:
        return f'its centre swings {fmt(candidate.centre_range_rad / own)}× as far as σ’s centre'
    if candidate.reason == 'rate' and finite(candidate.centre_rate_rad_yr):
        return f'its centre drifts at {fmt(abs(candidate.centre_rate_rad_yr) * KYR)} rad/kyr'
    if candidate.reason == 'pass' and finite(candidate.centre_max_pass_rad):
        return f'its centre makes a pass of {fmt(degrees(candidate.centre_max_pass_rad))}°'
    if candidate.reason in ('A_coherence', 'A_gain') and finite(candidate.phi0_coherence):
        return f'its force phase coherence {compare(candidate.phi0_coherence, ">=", s.mmr_subresonance_phi0_coh_min)}'
    return REJECT_WORDS.get(candidate.reason, candidate.reason.replace('_', ' '))


def _s_detail(f: MMRFacts, margin) -> Tuple[str, str, str]:
    sub, s = f.a.subresonances, f.s
    if sub is None:
        return '', '', ''
    note = ''
    if margin.margin == Margin.THIN:
        if sub.tie_by_order:
            note = 'the member was chosen by its order only'
        elif sub.accepted is not None and not sub.theory_measured:
            why = [THEORY_WORDS.get(flag, flag) for flag in sub.flags if flag.startswith('subresonance_theory_')]
            note = 'the multiplet theory was not computed' + (f': {", ".join(why)}' if why else '')
        elif finite(margin.value):
            note = f'a member’s centre is nearly as still as σ’s ({compare(margin.value, ">", s.mmr_s_close_ratio)})'
    if sub.accepted is not None:
        return (
            f'{sub.accepted.spec.key}: φ0 coherence {compare(sub.accepted.phi0_coherence, ">=", s.mmr_subresonance_phi0_coh_min)}',
            'the theory confirms the member',
            note,
        )
    best = _best_candidate(sub.candidates)
    value = f'{_plural(len(sub.candidates), "member")} scanned, none holds the centre'
    if best is not None:
        value += f'; best {best.spec.key}: {_member_rejection(f, best)}'
    return value, f'none accepted; closest centre range > {fmt(s.mmr_s_close_ratio)}× σ’s', note


def identity_route(owner) -> bool:
    """The owner was accepted by the route 'identity' (W22 R-OWNER-ID): phi librates, its strength passes."""
    accepted = getattr(owner, 'accepted', None)
    return getattr(accepted, 'route', '') == 'identity' and getattr(owner, 'identity', None) is not None


def _identity_claim(owner) -> str:
    """'47M-44+0-3 librates over σ’s episode (cover 1) and its directly averaged strength passes (ratio 1.35):
    identity route'."""
    route = owner.identity
    return (
        f'{owner.accepted.spec.key} librates over σ’s episode (cover {fmt(route.phi_cover, 2)}) and its directly '
        f'averaged strength passes (ratio {fmt(route.phi_strength_ratio)}): identity route'
    )


def _w_detail(f: MMRFacts) -> Tuple[str, str]:
    owner, s = f.a.owner, f.s
    if owner is None:
        return '', ''
    if identity_route(owner):
        return _identity_claim(owner), f'R² ≥ {fmt(s.mmr_w_r2_full)} and reachable (not measured by this route)'
    if owner.accepted is not None:
        return f'{owner.accepted.spec.key} drives a: R² {fmt(owner.accepted.r2_median, 2)}', f'R² ≥ {fmt(s.mmr_w_r2_full)} and reachable'
    best = _best_candidate(owner.candidates)
    if best is None:
        return f'{_plural(owner.n_candidates, "candidate")}, none fits', ''
    return (
        f'{_plural(owner.n_candidates, "candidate")}; best {best.spec.key} R² {compare(best.r2_median, "<", s.mmr_w_r2_none, digits=2)}',
        f'best R² < {fmt(s.mmr_w_r2_none)}',
    )


def _l8_rows(f: MMRFacts, margin) -> List[CheckRow]:
    s, sampling, prep = f.s, f.a.sampling, f.a.prep
    rows = []
    counts = margin_counts(margin.margin)
    if sampling is not None and _far_circulation(f):
        rotation = degrees(sampling.step_rotation_max_rad)
        rows.append(
            CheckRow(
                'L8a',
                MMR_MARGIN_NAMES['L8a'],
                CheckState.NOT_APPLICABLE,
                f'turn per step {fmt(rotation)}°, not graded: ⟨a⟩ {_far_text(sampling.far_ratio, s)} a-scales from a*',
                f'≥ {fmt(s.mmr_far_from_res_factor)} a-scales',
                COUNTS_NA,
                'a far circulation has no turning point for the step to resolve',
            )
        )
    elif sampling is not None:
        rotation = degrees(sampling.step_rotation_max_rad)
        ok = holds(rotation, '<', s.mmr_l8_step_full_deg)
        note = '' if ok else 'σ turns more than the limit per output step'
        if str(f.v.angle_subtype) == 'alias':
            note = 'the step folds σ: the alias rule (C5) decides instead'
        rows.append(
            CheckRow(
                'L8a',
                MMR_MARGIN_NAMES['L8a'],
                CheckState.PASS if ok else CheckState.THIN,
                f'turn per step {compare(rotation, "<", s.mmr_l8_step_full_deg, "°")}',
                f'< {fmt(s.mmr_l8_step_full_deg)}°',
                COUNTS_FULL if ok else counts,
                note,
            )
        )
    if sampling is not None and sampling.aliased_lines_yr:
        rows.append(CheckRow('L8a', 'no aliased line in the band', CheckState.THIN, 'aliased line(s) in the band', '', COUNTS_THIN))
    gain = prep.gain_at_period_cycle if prep is not None else float('nan')
    if finite(gain):
        state = CheckState.PASS if gain >= s.mmr_l8_gain_full else (CheckState.THIN if gain >= s.mmr_l8_gain_fail else CheckState.FAIL)
        rows.append(
            CheckRow(
                'L8b',
                MMR_MARGIN_NAMES['L8b'],
                state,
                f'gain at the cycle period {compare(gain, ">=", s.mmr_l8_gain_full)}',
                f'≥ {fmt(s.mmr_l8_gain_full)} (fail < {fmt(s.mmr_l8_gain_fail)})',
                COUNTS_FULL if state == CheckState.PASS else counts,
            )
        )
    if margin.margin == Margin.THIN and all(row.state in (CheckState.PASS, CheckState.NOT_APPLICABLE) for row in rows):
        rows.append(CheckRow('L8c', 'filter guard (cutoff ×2) compared', CheckState.THIN, 'nothing could be compared', '', COUNTS_THIN))
    return rows or [CheckRow('L8', 'step and filter', margin_state(margin.margin), fmt(margin.value), '', counts)]


def _far_circulation(f: MMRFacts) -> bool:
    """L8's step part is not graded (`decide.far_circulation`): a non_resonant circulation whose <a> stays
    MMR_FAR_FROM_RES_FACTOR a-scales from a* (far_ratio inf: no commensurability at any a)."""
    far = f.a.sampling.far_ratio if f.a.sampling is not None else float('nan')
    circulation = str(f.v.status) == 'non_resonant' and str(f.v.angle_status) == 'circulation'
    return circulation and not math.isnan(far) and far >= f.s.mmr_far_from_res_factor


def _far_text(far: float, s) -> str:
    if math.isinf(far):
        return '∞ (no commensurability)'
    return compare(far, '>=', s.mmr_far_from_res_factor)


def _circulation_detail(f: MMRFacts, key: str, margin) -> Tuple[str, str]:
    s, g, sampling, context = f.s, f.a.geometry, f.a.sampling, f.a.context
    if key == 'C1':
        return f'largest pass {compare(margin.value, ">=", s.mmr_c1_rev_full)} rev', f'≥ {fmt(s.mmr_c1_rev_full)} rev'
    if key == 'C2':
        return f'{fmt(margin.value)} turning points over the three cutoffs', '0'
    if key == 'C3':
        parts = []
        if g is not None and finite(g.sigmadot_negative_fraction):
            parts.append(f'dσ/dt of the other sign in {percent(g.sigmadot_negative_fraction)} of the record')
        if g is not None and finite(g.rate_modulation):
            parts.append(f'modulation {compare(g.rate_modulation, "<", s.mmr_c3_modulation_max)}')
        return ', '.join(parts), f'0%; < {fmt(s.mmr_c3_modulation_max)}'
    if key == 'C4':
        return f'ratio {compare(margin.value, ">=", s.mmr_c4_ratio_full)}', f'≥ {fmt(s.mmr_c4_ratio_full)}'
    if str(f.v.angle_subtype) == 'alias' and sampling is not None:
        clearance = 360.0 * sampling.branch_margin_turns
        resid = degrees(sampling.unwrap_resid_max_rad)
        bound = s.mmr_c5_branch_resid_factor * resid if resid is not None else float('nan')
        return (
            f'⟨a⟩ {_far_text(sampling.far_ratio, s)} a-scales from a*; branch clearance per step '
            f'{compare(clearance, ">", bound, "°")} ({fmt(s.mmr_c5_branch_resid_factor)} × the largest unwrap residual)',
            f'≥ {fmt(s.mmr_far_from_res_factor)}; > {fmt(s.mmr_c5_branch_resid_factor)} × residual',
        )
    if context is not None and finite(margin.value):
        return f'|ν_meas/ν_exp − 1| {compare(abs(margin.value - 1), "<=", s.mmr_c5_rate_tol)}', f'≤ {fmt(s.mmr_c5_rate_tol)}'
    return fmt(margin.value), ''


def _detail_l1(f: MMRFacts, margin) -> Tuple[str, str, str]:
    s = f.s
    note = ''
    if margin.margin == Margin.THIN and finite(margin.value) and margin.value < s.mmr_l1_tp_full:
        note = f'{fmt(margin.value)} < {s.mmr_l1_tp_full}: the libration is short'
    count = int(margin.value) if finite(margin.value) else 0
    return f'{_plural(count, "turning point")} at the coarsest cutoff', f'≥ {s.mmr_l1_tp_full}', note


def _detail_l2(f: MMRFacts, margin) -> Tuple[str, str, str]:
    energy, s = f.a.energy, f.s
    if energy is not None and energy.zone is not None and energy.cycle_classes:
        n_zone = sum(1 for c in energy.cycle_classes if str(c) == 'zone')
        n_unmeasured = sum(1 for c in energy.cycle_classes if str(c) == 'not_measured')
        unmeasured = f', {n_unmeasured} not measured (above a lower saddle)' if n_unmeasured else ''
        detail = f'{n_zone} of {len(energy.cycle_classes)} cycles in the zone{unmeasured} ' f'(max half-amplitude {fmt(margin.value)}°)'
        note = f'{n_zone} of {len(energy.cycle_classes)} cycles in the separatrix zone' if n_zone else ''
        return detail, 'no cycle in the zone', note
    return f'max half-amplitude {compare(margin.value, "<=", s.mmr_l2_amp_full_deg, "°")}', f'≤ {fmt(s.mmr_l2_amp_full_deg)}°', ''


def _detail_l3(f: MMRFacts, margin) -> Tuple[str, str, str]:
    response, s = f.a.response, f.s
    detail = f'slope/c {fmt(response.identity_slope_ratio)}, corr {fmt(response.identity_corr)}' if response is not None else ''
    return (
        _with_source(detail, f, 'identity_source'),
        f'|slope/c − 1| ≤ {fmt(s.mmr_identity_slope_tol)}, corr ≥ {fmt(s.mmr_identity_corr_min)}',
        '',
    )


def _detail_l4(f: MMRFacts, margin) -> Tuple[str, str, str]:
    return _force_detail(f) + ('',)


def _detail_l5(f: MMRFacts, margin) -> Tuple[str, str, str]:
    energy, s = f.a.energy, f.s
    modulation = 1.0
    if energy is not None and finite(energy.omega_max_rel) and finite(energy.omega_min_rel) and energy.omega_min_rel > 0:
        modulation = energy.omega_max_rel / energy.omega_min_rel
    return (
        f'T0 max/min {compare(margin.value, "<=", s.mmr_l5_ratio_full * modulation)}',
        f'≤ {fmt(s.mmr_l5_ratio_full)} × Ω-modulation {fmt(modulation)}',
        '',
    )


def _detail_l6(f: MMRFacts, margin) -> Tuple[str, str, str]:
    return f'shift/amplitude {compare(margin.value, "<=", f.s.mmr_l6_shift_full)}', f'≤ {fmt(f.s.mmr_l6_shift_full)}', ''


def _detail_l7(f: MMRFacts, margin) -> Tuple[str, str, str]:
    return f'larger edge gap {compare(margin.value, "<=", f.s.mmr_l7_gap_full)} P', f'≤ {fmt(f.s.mmr_l7_gap_full)} P', ''


def _detail_o1(f: MMRFacts, margin) -> Tuple[str, str, str]:
    s, episode = f.s, f.episodes.decisive or {}
    if margin.margin == Margin.NOT_APPLICABLE and f.a.energy is not None and f.a.energy.episodes_in_separatrix_zone:
        return "n/a: inside the separatrix zone the wave before is the same force's", '', ''
    if margin.margin == Margin.NOT_APPLICABLE and str(f.v.origin_detail) == 'owner_libration':
        from resonances.resonance.analysis.mmr.decide import owner_background_cover

        window = (episode.get('onset_before_t0_yr'), episode.get('onset_before_t1_yr'))
        cover = owner_background_cover(f.a.owner, tuple(float(t) if finite(t) else float('nan') for t in window))
        if finite(cover):  # W22 R-O1-OWNER: the a-wave before is the owner's libration
            text = f"n/a: the owner's libration covers {compare_percent(cover, '>=', s.mmr_o1_owner_cover_min)} of the band before"
            return text, '', ''
    before, inside = episode.get('a_band_std_before_au'), episode.get('a_band_std_in_au')
    detail = ''
    if finite(before) and finite(inside) and float(inside) > 0:
        ratio = float(before) / float(inside)
        limit = s.mmr_o1_ratio_fail if ratio >= s.mmr_o1_ratio_thin else s.mmr_o1_ratio_thin
        detail = f'a-wave before/inside {compare(ratio, "<", limit)}'
    return detail, f'< {fmt(s.mmr_o1_ratio_thin)} (fail ≥ {fmt(s.mmr_o1_ratio_fail)})', ''


def _detail_r0(f: MMRFacts, margin) -> Tuple[str, str, str]:
    return _potential_detail(f) + ('',)


def _detail_z(f: MMRFacts, margin) -> Tuple[str, str, str]:
    energy = f.a.energy
    detail = note = ''
    if energy is not None and energy.zone is not None:
        method = str(getattr(energy.zone, 'method', '')).split('.')[-1].lower()
        detail = f'h {fmt(energy.zone.h_lo)}…{fmt(energy.zone.h_hi)} ({method})'
        if margin.margin == Margin.THIN:
            note = 'only estimated: ' + ', '.join(flag.replace('zone_', '') for flag in energy.flags if flag.startswith('zone_'))
    return detail, 'a numeric zone, no edge cycle', note


def _detail_w(f: MMRFacts, margin) -> Tuple[str, str, str]:
    if margin.margin == Margin.NOT_APPLICABLE:
        return 'a 1:1 has no owner by construction', '', ''
    return _w_detail(f) + ('',)


def _detail_f(f: MMRFacts, margin) -> Tuple[str, str, str]:
    s, strength = f.s, f.a.strength
    ratio = strength.ratio if strength is not None else None
    limit = s.mmr_strength_check_ratio
    detail = ''
    if finite(ratio):
        inside = 1 / limit <= ratio <= limit
        detail = f'required/theory {fmt(ratio)} {"within" if inside else "outside"} [{fmt(1 / limit)}, {fmt(limit)}]'
    if f.f_kinematic:  # W22 R-FK: F supports the verdict when the strength fails
        return detail, f'outside [{fmt(1 / limit)}, {fmt(limit)}]', ''
    note = 'the owner search did not run, so a foreign force is not excluded' if margin.margin == Margin.THIN else ''
    return detail, f'in [{fmt(1 / limit)}, {fmt(limit)}]', note


def _detail_k1(f: MMRFacts, margin) -> Tuple[str, str, str]:
    return compare(margin.value, '>=', f.s.mmr_k1_full), f'≥ {fmt(f.s.mmr_k1_full)}', ''


def _detail_b5(f: MMRFacts, margin) -> Tuple[str, str, str]:
    return f'ratio {compare(margin.value, ">=", 1)}', f'≥ 1 (full ≥ {fmt(f.s.mmr_b5_ratio_full)})', ''


def _detail_n(f: MMRFacts, margin) -> Tuple[str, str, str]:
    return f'longest short run: {fmt(margin.value)} turning points', '', ''


def _detail_circulation(key: str):
    return lambda f, margin: _circulation_detail(f, key, margin) + ('',)


# margin id -> (detail, need, note) of its row
MARGIN_DETAILS = {
    'L1': _detail_l1,
    'L2': _detail_l2,
    'L3': _detail_l3,
    'L4': _detail_l4,
    'L5': _detail_l5,
    'L6': _detail_l6,
    'L7': _detail_l7,
    'O1': _detail_o1,
    'R0': _detail_r0,
    'Z': _detail_z,
    'S': _s_detail,
    'W': _detail_w,
    'F': _detail_f,
    'K1': _detail_k1,
    'B5': _detail_b5,
    'N': _detail_n,
    **{key: _detail_circulation(key) for key in ('C1', 'C2', 'C3', 'C4', 'C5')},
}


def _margin_name(f: MMRFacts, key: str) -> str:
    """The row's name: the accepted member or owner by its key, the 1:1 force law by its own name."""
    sub, owner = f.a.subresonances, f.a.owner
    if key == 'S' and sub is not None and sub.accepted is not None:
        return f'member {sub.accepted.spec.key} holds the centre'
    if key == 'W' and owner is not None and owner.accepted is not None:
        approximate = ' (catalogue λ, approx.)' if 'catalogue_longitudes' in owner.flags else ''
        if identity_route(owner):
            return f'owner {owner.accepted.spec.key} librates (identity route){approximate}'
        return f'owner {owner.accepted.spec.key} drives a{approximate}'
    if key == 'L4' and f.coorbital:
        return 'ȧ follows the 1:1 force law'
    if key == 'F' and f.f_kinematic:
        return 'the term is too weak for this period'
    return MMR_MARGIN_NAMES.get(key, key)


NOTE_IN_COLUMN = ('S', 'F', 'Z')  # margins whose thin note says more than their detail (the others: sentence only)


def short_reason(text: str) -> str:
    """The first clause of a reason, up to ': ' or '; ' ('half-amplitude 30.1° < 45°: sin σ ≈ σ, ...' -> 'half-amplitude
    30.1° < 45°')."""
    return text.split(': ')[0].split('; ')[0]


def _margin_row(f: MMRFacts, key: str, margin) -> List[CheckRow]:
    if key == 'L8':
        return _l8_rows(f, margin)
    detail, need, note = MARGIN_DETAILS[key](f, margin) if key in MARGIN_DETAILS else (fmt(margin.value), '', '')
    counts = margin_counts(margin.margin, margin.margin_class)
    shown = note if key in NOTE_IN_COLUMN else ''
    row = CheckRow(key, _margin_name(f, key), margin_state(margin.margin), detail, need, counts, shown, note)
    return [_finish(f, key, margin, row)]


def _finish(f: MMRFacts, key: str, margin, row: CheckRow) -> CheckRow:
    """A not_measured margin: its state (skipped by design or missing) and reason; its number printed as not used."""
    if margin.margin != Margin.NOT_MEASURED:
        return row
    state, why = why_not_measured(f, key)
    row.state = CheckState.MISSING if state == CheckState.NOT_APPLICABLE else state
    if state == CheckState.NOT_APPLICABLE:
        why = f'{why}, yet counted as not measured'
    row.note, row.short = why, short_reason(why)
    measured = key in f.unused and why == f.unused[key]  # a number that was measured, then not used
    # a number of a missing measurement says nothing
    row.detail = _unused_detail(row.detail, state != CheckState.MISSING or measured)
    return row


def _unused_detail(detail: str, keep: bool = True) -> str:
    """The number of a not_measured margin printed as not used ('(…: not used)'); '' for a dash or when not `keep`."""
    return f'({detail}: not used)' if keep and detail and '—' not in detail else ''


# Ledger entries of W20 (parts A and B) in words, by the start of their name (the longest match wins): the names stay
# the code's (columns, tests and the 'all' dump read them), the explained column prints these
LEDGER_WORDS = {
    '5 L4 null test': 'the force law beats a shifted in time (null test)',
    '5 L4 N-harmonic law: R2 gain': 'a second harmonic of σ improves the force law',
    '5 L4 N-harmonic law R2': 'the force law with harmonics of σ explains ȧ',
    '5 L4 N-harmonic law coefficients': 'the harmonics of the force law keep their phase from window to window',
    '5 L4 N-harmonic law null test': 'the harmonic force law beats ȧ shifted in time (full windows)',
    '5 L4 N-harmonic law on independent halves': 'the force law of one half of the windows predicts the other half',
    'cap short_period_aliased': 'the fast spread of σ_f against the swing (the step folds short periods)',
    'short_period_aliased: a harmonic folded': 'a harmonic of the short periods folded slower than the period floor',
    'short-period scatter of a': 'the short-period scatter of a against the libration’s a swing',
    '5 R-ALIAS-A folded step': 'the short-period scatter of a that reads a failed L4/L3 as not measured (R-ALIAS-A)',
    'G2 exit observed: the raw angle runs on': 'exit seen: the orbit-averaged σ runs on beyond the episode',
    'G2 exit observed: the raw angle moves monotonically': 'exit seen: the orbit-averaged σ moves one way',
    'G2 entry observed: the raw angle runs on': 'entry seen: the orbit-averaged σ ran in before the episode',
    'G2 entry observed: the raw angle moves monotonically': 'entry seen: the orbit-averaged σ moved one way',
    'G2c not when the exit is observed': 'no slower libration sought: the exit is seen at the fine scale',
    'G2c not when the entry is observed': 'no slower libration sought: the entry is seen at the fine scale',
    'G2 1:1 regimes without a circulation': 'the 1:1 regimes join without a circulation between them',
    'G2 1:1 range of the turning points': 'the turning points between the 1:1 regimes stay within a revolution',
    'G2 1:1 range of the raw angle': 'the raw angle between the 1:1 regimes stays within a revolution',
    'G2 accepted runs that are forced waves': 'the accepted runs are forced waves, not librations',
    'G2 forced wave': 'a run on a forced line of another argument',
    'G5a slow centre': 'the slow centre turns back once: an arc, not a drift',
    'G1q the orbit-averaged angle stays': 'θ stays in the quasi-satellite well long enough',
    'G1q quasi-satellite floor': 'the quasi-satellite floor lies below the tadpole floor',
    'G1q the pass keeps': 'the quasi-satellite pass keeps the episodes outside the well',
    'G1q a stall with P_QS below the cutoff': 'a stall the cutoff hides: the quasi-satellite libration is faster',
    '2 owner route': 'the owner is accepted by its route',
    '2 owner cover of the decisive episode': 'the owner’s intervals cover σ’s decisive episode',
    '2 phase lever': 'the owner’s phase is steadier than σ’s own term could make it',
    '2 potential scale': 'the owner’s averaged force matches ȧ in scale',
    '2 second capture: x sweep': 'Ψ/u sweeps far enough to test a second capture',
    '2 second capture: energy R2': 'Ψ/u follows no potential (no second capture)',
    '3 K2 angle word': 'the angle word of the slow centre',
    '3 K2b kinematic libration': 'the kinematic libration’s centre moves more than its amplitude (a drift)',
    'G4b net shift': 'the slow centre advances by whole revolutions net (circulation)',
    '3 kinematic (slow centre)': 'the slow centre follows the perihelion term (K2)',
    '5s (before 5a/5e/5g/5u)': 'the strength, read before the force law',
    'G0 angle jitter': 'the p·ϖ term wobbles at short periods',
    'G0 full-angle residual': 'the full angle unwraps',
    'G0 varpi phase defined': 'ϖ itself is defined',
}


def ledger_words(name: str) -> str:
    """A ledger entry's name in words (`LEDGER_WORDS`, longest matching start), else the name itself."""
    found = max((key for key in LEDGER_WORDS if name.startswith(key)), key=len, default='')
    return LEDGER_WORDS[found] if found else name


def near_threshold_rows(verdict, settings) -> List[CheckRow]:
    """Ledger comparisons within the thin band of a convention threshold (the U items of thin_margins)."""
    rows = []
    thin = {item[2:] for item in verdict.thin_margins if str(item).startswith('U:')}
    for entry in verdict.decision_path:
        if entry.name in thin and entry.result is not None:
            rows.append(
                CheckRow(
                    f'U:{entry.name}',
                    f'{ledger_words(entry.name)} near its convention',
                    CheckState.THIN,
                    entry_compare(entry),
                    f'outside ±{fmt(100 * settings.confidence_thin_band)}% of {fmt(entry.threshold)}',
                    COUNTS_THIN,
                    'near its threshold',
                )
            )
    return rows


def return_clause(f: MMRFacts) -> str:
    """The decisive episode closed by its phase return (CS4 E2, FIX2 level): 'one cycle closed by its phase return:
    σ_f leaves the 0.5 level at 120 yr and crosses it back at 780 yr', '' otherwise."""
    episode = f.episodes.decisive or {}
    if str(episode.get('closed_by_return')).lower() not in ('true', '1'):
        return ''
    level = episode.get('return_level_fraction')
    at = f'the {fmt(level, 2)} level of the swing' if finite(level) else 'its return level'
    return (
        f'one cycle closed by its phase return: σ_f leaves {at} at {years(episode.get("return_t_a_yr"))} and crosses it back '
        f'at {years(episode.get("return_t_b_yr"))}'
    )


def centre_scale_clause(f: MMRFacts) -> str:
    """A verdict decided at the scale of the centre (G2c, `analyse._at_centre_scale`): 'at the scale of the centre:
    σ filtered at W_c 1108 yr > 75 yr, the fine cutoff without a libration; its turning points are turns of the centre
    (prominence ≥ 10°)' (W23 rule a, libration_centre_scale_tp_at_centre: the floor raised to the centre-turn
    prominence p_c, a smaller swing is residue of the fast component W_c attenuates); '' otherwise."""
    window, fine = f.row.get('geo_centre_scale_cutoff_yr'), f.row.get('geo_fine_cutoff_yr')
    if not finite(window):
        return ''
    text = f'at the scale of the centre: σ filtered at W_c {compare(window, ">", fine, " yr")}, the fine cutoff without a libration'
    first = f.row.get('geo_centre_scale_first_cutoff_yr')
    if finite(first):
        text += f' (W_c {years(first)} filtered the libration found there: analysed again at P / {fmt(f.s.filter_min_libration_cutoffs)})'
    prominence = f.row.get('geo_tp_prominence_deg')
    if f.s.libration_centre_scale_tp_at_centre and finite(prominence):
        text += f'; its turning points are turns of the centre (prominence ≥ {fmt(prominence)}°)'
    return text


def edge_clauses(f: MMRFacts) -> List[str]:
    """The record edges seen at the fine scale (CS4 E1, rule (a)): 'exit seen: the orbit-averaged σ runs on 1.97 rev
    ≥ 1 rev beyond the episode, one way'."""
    out = []
    for edge in ('exit', 'entry'):
        arc, monotone = _held(f, edge_arc_name(edge)), _held(f, edge_monotone_name(edge))
        if not (arc and monotone):
            continue
        where = 'beyond the episode' if edge == 'exit' else 'before the episode'
        out.append(f'{edge} seen: the orbit-averaged σ runs {entry_compare(arc, " rev")} {where}, one way')
    return out


def quasi_satellite_clause(f: MMRFacts) -> str:
    """Rule G1q (a 1:1 analysed again at the quasi-satellite floor), '' when it did not fire."""
    visit, floor = _held(f, QS_VISIT), _held(f, QS_FLOOR)
    if not (visit and floor):
        return ''
    return (
        f'θ stays in the quasi-satellite well for {entry_compare(visit, " yr")}; its floor from '
        f'the averaged potential {entry_compare(floor, " yr")} (the tadpole floor), so σ is '
        'read again at that floor'
    )


def rule_clauses(f: MMRFacts) -> str:
    """'; '-joined clauses of the geometry rules on this record (the scale of the centre, quasi-satellite floor, phase
    return, edges)."""
    parts = [centre_scale_clause(f), quasi_satellite_clause(f), return_clause(f)] + edge_clauses(f)
    return '; '.join(part for part in parts if part)


def _cap_detail(f: MMRFacts, row: CheckRow) -> CheckRow:
    """The numbers behind a cap, where the record keeps them (the return, the fast spread of σ_f)."""
    cap = row.key[4:]
    if cap == 'one_return_cycle':  # the times are in the angle-shape step (no number twice)
        episode = f.episodes.decisive or {}
        start, end = episode.get('return_t_a_yr'), episode.get('return_t_b_yr')
        period = float(end) - float(start) if finite(start) and finite(end) else None
        row.detail = f'one cycle of {years(period)} (its return: the angle-shape step)' if return_clause(f) else ''
    elif cap == 'short_period_aliased':
        row.detail = _short_period_detail(f)
        row.name, row.short = _short_period_title(f)
        row.look = short_period_look(f.row.get('smp_fold_free_step_yr'), row.look)
    return row


def short_period_look(fold_free_step_yr: Any, look: str) -> str:
    """Where the cap short_period_aliased is decided, naming the output step that lifts it (smp_fold_free_step_yr, the
    largest grid step below the record's that folds no short period; W21 A); the general words without one."""
    if not finite(fold_free_step_yr):
        return look
    return f'(a): the fast spread about σ_f; a rerun at ≤ {fmt(fold_free_step_yr)} yr output step lifts this cap'


def _short_period_title(f: MMRFacts) -> Tuple[str, str]:
    """(title, short reason) of the cap short_period_aliased: the title names the condition that fired (the harmonic
    entry, else dt >= min(P)/2), the short reason what holds the cap, for the compact cap line and the confidence
    sentence ('a scatter 1.21 × half-swing', 'harmonic folded to 36 kyr'; none when the title says Δt ≥ min(P)/2)."""
    harmonic, scatter = _held(f, SHORT_PERIOD_HARMONIC), _held(f, SHORT_PERIOD_A)
    if harmonic:
        title = SHORT_PERIOD_TITLES['harmonic']
        short = f'harmonic folded to {years(harmonic.value)}'
    else:
        step, limit = f.row.get('smp_step_yr'), f.row.get('smp_short_period_step_yr')
        if holds(step, '>=', limit):
            title = SHORT_PERIOD_TITLES['nyquist'].format(comparison=compare(step, '>=', limit, ' yr'))
        else:  # neither part on record (an old row): the cap's general words
            title = CAP_WORDS['short_period_aliased']
        short = ''  # the title names it (Δt ≥ min(P)/2)
    if scatter:
        short = f'a scatter {fmt(scatter.value, 3)} × half-swing'
    return title, short


def _short_period_detail(f: MMRFacts) -> str:
    """Why the step's folded lines cap the level (`analyse.short_period_caps`): the harmonic folded below the
    fundamental step, the scatter of a against the libration's a swing, the fast spread of σ_f (the lift by the record,
    MMR_SHORT_PERIOD_CAP_RATIO > 0), else that the record cannot bound them."""
    parts = []
    if _held(f, SHORT_PERIOD_HARMONIC):
        parts.append(f'a harmonic folded to {_entry_text(f, SHORT_PERIOD_HARMONIC, " yr")} (the period floor)')
    if _held(f, SHORT_PERIOD_A):
        parts.append(f'short-period scatter of a {_entry_text(f, SHORT_PERIOD_A)} × the a swing (folded lines reach a)')
    if _held(f, SHORT_PERIOD_CAP):
        parts.append(f'{ledger_words(SHORT_PERIOD_CAP)}: {_entry_text(f, SHORT_PERIOD_CAP)}')
    return '; '.join(parts) if parts else 'how much of the folded lines reached σ_f and a_f only a finer step measures'


def mmr_checks(f: MMRFacts) -> List[CheckRow]:
    if not str(f.v.confidence):
        return []
    rows = [row for key, margin in f.margins.items() for row in _margin_row(f, key, margin)]
    return confidence_rows(f.v, f.s, rows, f.whole_record, lambda row: _cap_detail(f, row))


def confidence_rows(verdict, settings, margin_rows: Sequence[CheckRow], whole_record: bool, finish=None) -> List[CheckRow]:
    """The confidence block of a verdict with a level: its margin rows, the comparisons near their thresholds and the
    caps (each through `finish` when given), in reading order."""
    caps = cap_rows(verdict.caps, str(verdict.confidence), whole_record)
    rows = list(margin_rows) + near_threshold_rows(verdict, settings) + [finish(row) if finish else row for row in caps]
    return sort_checks(rows)


# --- the tree


def _undefined_phase_text(f: MMRFacts) -> str:
    """undefined_phase: the share of the record where ϖ itself is undefined, or the old wording when not measured."""
    share, e_min = f.row.get('smp_varpi_undefined_fraction'), fmt(f.row.get('ctx_e_min'))
    if not finite(share):
        return f'ϖ undefined at small e (e down to {e_min}; its own scatter not measured)'
    shares = compare_percent(share, '>', f.s.mmr_varpi_undefined_max_share)
    return f'ϖ itself undefined on {shares} of the record (e down to {e_min})'


def _undefined_step(f: MMRFacts) -> Step:
    """The leaf of an angle that is not measured at all (G0)."""
    if str(f.v.reason) == 'undersampled':
        text = f'the cutoff {fmt(f.row.get("flt_cutoff_yr"))} yr is not longer than two output steps → resample'
        return _shape(text, StepState.LEAF)
    words = {
        'alias': 'the step aliases the angle beyond reconstruction',
        'unphysical': 'the orbit is unphysical',
        'undefined_phase': _undefined_phase_text(f),
    }
    return _shape(words.get(str(f.v.angle_subtype), 'undefined'), StepState.LEAF)


def _shape_step(f: MMRFacts) -> Step:
    v, s = f.v, f.s
    angle, subtype = str(v.angle_status), str(v.angle_subtype)
    accepted = f.episodes.accepted
    if angle == 'undefined':
        return _undefined_step(f)
    rules = rule_clauses(f)
    if angle == 'libration':
        what = (_union_clause(f) or 'σ librates over the whole record') + (f'; {rules}' if rules else '')
        return _shape(f'{what} → libration')
    if angle == 'episodic' and _forced_wave_word(f):
        text = (
            f'σ librates only in {_forced_wave_clause(f)} (the angle word); the status reads the angle without it: the '
            f'centre passes {fmt(f.row.get("geo_centre_max_pass_rev"))} rev in one direction → episodic, read as {_tree_word(f)}'
        )
        return _shape(text)
    if angle == 'episodic':
        episode = f.episodes.decisive or {}
        gaps = coverage_text(episode, s)
        what = 'σ librates in part of the record' if len(accepted) == 1 else f'{len(accepted)} separate librating stretches'
        if len(accepted) > 1:
            gaps = gaps.replace('it misses', 'the deciding one misses', 1)
        what = '; '.join(part for part in (what, gaps, rules) if part)
        return _shape(f'{what} → episodic')
    if angle == 'circulation':
        return _circulation_shape_step(f, subtype)
    if angle == 'drift':
        return _drift_shape_step(f)
    if angle == 'near_separatrix':
        what = 'stalls at the top in each turn (pattern B)' if subtype == 'pattern_b' else 'slips over the top (pattern A)'
        return _shape(f'no libration; σ {what} → near_separatrix')
    if angle == 'insufficient_time':
        return _shape(_insufficient_text(f, subtype), StepState.LEAF)
    return _shape(angle)


def union_kind(f: MMRFacts) -> str:
    """How a union of episodes (W22) made the word libration: 'centre' (found only at the scale of the centre,
    `analyse._centre_scale_union_word`), 'promoted' (the union rule G2u, read as episodic and promoted by rule 6u,
    `decide.promote_union`), 'kept' (G2u, the status kept its episodic reading), '' (no union of that rule)."""
    if str(f.v.angle_subtype) != 'regime_change' or f.coorbital:
        return ''
    if f.entry(CENTRE_SCALE_UNION) is not None:
        return 'centre'
    entry = f.entry(UNION_PROMOTED)
    if entry is None:
        return ''
    return 'promoted' if entry.result else 'kept'


def _union_clause(f: MMRFacts) -> str:
    """The shape of a libration word set by a union of episodes ('' for any other libration)."""
    kind = union_kind(f)
    if kind == 'centre':
        return (
            'episodes without a circulation between them cover the record at the scale of the centre (the word only: '
            'the fine scale reads them as episodic)'
        )
    if kind:
        many = _union_count(f)
        return f'{many} without a circulation between them cover the record (union rule; the status tree reads them as episodic)'
    return ''


def _union_count(f: MMRFacts) -> str:
    """The episodes a union covers the record with: '3 episodes', or 'episodes' when the count is not known."""
    n = f.row.get('geo_union_n_episodes')
    return f'{fmt(n)} episodes' if finite(n) else 'episodes'


def _insufficient_text(f: MMRFacts, subtype: str) -> str:
    """The shape of insufficient_time by its subtype: an arc of the slow centre (G5a, the centre turns back once within
    one revolution: part of one slow swing, 2003 KQ18), an arc of sigma_f without turning points (G6), a stall, or too
    few turning points (few_cycles)."""
    tp = fmt(f.row.get('geo_n_tp'))
    turn = _slow_arc(f)
    if subtype == 'arc' and turn is not None:
        return (
            f'{tp} turning points ride on a slow centre that turns back once, within {fmt(turn.value / (2 * math.pi), 2)} rev: '
            'part of one slow swing, where a drift and a slow libration look alike → insufficient_time (arc)'
        )
    if subtype == 'arc':
        arc = f.entry(G6_RANGE)
        degrees = math.degrees(arc.value) if arc is not None and finite(arc.value) else None
        return f'no turning point; σ_f runs an arc of {fmt(degrees)}° → insufficient_time (arc)'
    if subtype == 'stall':
        return 'no turning point and no arc: σ_f stalls → insufficient_time (stall)'
    return f'{tp} turning points, too few for a libration → insufficient_time'


def _slow_arc(f: MMRFacts):
    """G5a's last entry when it held (the slow centre turns back once within a revolution), else None."""
    entry = _held(f, G5A_ONE_REVOLUTION)
    return entry if entry and finite(entry.value) else None


FORCED_WAVE_WORD = 'sigma librates in the forced-wave runs'


def _forced_wave_word(f: MMRFacts) -> bool:
    """The reported word is the geometry's with the forced-wave runs (`decide.forced_wave_word`, its ledger line): the
    status read a circulation or a drift without them."""
    rewritten = any(FORCED_WAVE_WORD in entry.name for entry in f.v.decision_path)
    return bool(_forced_wave_clause(f)) and rewritten


def _tree_word(f: MMRFacts) -> str:
    """'a circulation' or 'a drift': the word the status tree read under a forced-wave rewrite (`_forced_wave_word`)."""
    return 'a drift' if 'drift' in str(f.v.reason) else 'a circulation'


def _circulation_shape_step(f: MMRFacts, subtype: str) -> Step:
    """The angle shape of a circulation (an alias: the turn per step; accepted runs that are forced waves named)."""
    if subtype == 'alias':
        text = f'σ turns {fmt(f.row.get("smp_step_rotation_deg_max"))}° per output step → circulation (alias)'
        return _shape(text)
    if f.rule == '3k2':
        net = _net_shift(f, K2_NET)
        moves = f'advances {net} rev net' if net else f'passes {fmt(f.row.get("geo_centre_max_pass_rev"))} rev'
        text = f'librating stretches, a ripple on a centre that {moves} (step 3) → circulation'
        return _shape(text)
    forced = _forced_wave_clause(f)
    why = f'no libration ({forced})' if forced else 'no libration'
    net = _net_shift(f, G4B_NET)
    if net:
        passed = fmt(f.row.get('geo_centre_max_pass_rev'))
        text = f'{why}; the centre advances {net} rev net, at most {passed} rev in one pass (G4b) → circulation'
        return _shape(text)
    text = f'{why}; the centre passes {fmt(f.row.get("geo_centre_max_pass_rev"))} rev in one direction → circulation'
    return _shape(text)


def _net_shift(f: MMRFacts, name: str) -> str:
    """'|net| >= x' when the net-shift rule `name` (G4b or its use in K2's word) held, else ''."""
    entry = _held(f, name)
    return compare(entry.value, '>=', entry.threshold) if entry else ''


def _drift_shape_step(f: MMRFacts) -> Step:
    """The angle shape of a drift: the geometry's (G5), or K2's word (rule 3k2: librating stretches that are a ripple
    on a kinematic drift of the centre)."""
    if f.rule == '3k2':
        shift = _held(f, K2B_SHIFT)
        if shift:
            text = f'σ librates about a centre that moves {compare(shift.value, ">", shift.threshold)} × its amplitude (step 3) → drift'
            return _shape(text)
        accepted = f.episodes.accepted
        what = 'σ librates in part of the record' if len(accepted) == 1 else f'{len(accepted)} separate librating stretches'
        return _shape(f'{what}, a ripple on a kinematic drift of the centre (step 3) → drift')
    text = f'oscillates, but the centre drifts over {percent(f.row.get("geo_drift_fraction"))} of the record → drift'
    return _shape(text)


def _forced_wave_clause(f: MMRFacts) -> str:
    """G2: accepted runs judged forced waves of another argument (`decide.forced_wave_episodes`), '' without one."""
    count = f.row.get('geo_n_forced_wave_episodes')
    if not finite(count) or float(count) < 1:
        return ''
    line = f.row.get('geo_forced_wave_line')
    why = f'on the forced line {line} with a failed identity' if isinstance(line, str) and line else 'σ_dot against c (a − a*)'
    return f'{_plural(int(count), "run")} of P {fmt(f.row.get("geo_forced_wave_period_yr"))} yr {why}: a forced wave'


def _circulation_steps(f: MMRFacts) -> List[Step]:
    v, s = f.v, f.s
    if f.rule == 'C5':
        far = f.entry(C5_FAR)
        branch = _held(f, C5_BRANCH)
        text = (
            f'{"one branch" if branch else "several branches"} of ν·Δt over the a range; '
            f'⟨a⟩ {compare(far.value if far else None, ">=", s.mmr_c5_far_ratio_min)} a-ranges from a* → {v.status}'
        )
        return [Step('alias', 'alias (C5)', StepState.LEAF, text)]
    if f.rule == 'R0':
        text = f'σ turns back at times, but the energy does not fit a potential from a → {v.status}'
        return [Step('potential', 'potential (R0)', StepState.LEAF, text)]
    owner = f' (its wave comes from {v.owner})' if v.owner else ''
    return [Step('circ', 'circulation', StepState.LEAF, f'circulation → {v.status}{owner}')]


def _subresonance_step(f: MMRFacts) -> Step:
    sub, s, v = f.a.subresonances, f.s, f.v
    title = '1. subresonances'
    if sub is None:
        return Step('sub', title, StepState.NOT_RUN, 'not run: a derived row' if f.derived else _not_triggered(f))
    if sub.accepted is not None:
        coherence = compare(sub.accepted.phi0_coherence, '>=', s.mmr_subresonance_phi0_coh_min)
        text = f'{sub.accepted.spec.key} keeps a fixed centre (φ0 coherence {coherence}) → {v.status}, origin subresonance'
        return Step('sub', title, StepState.LEAF, text)
    if str(sub.check) in ('pass', 'fail'):
        best = _best_candidate(sub.candidates)
        text = f'{_plural(len(sub.candidates), "member")} scanned, none holds the centre'
        return Step('sub', title, StepState.PASSED, text + (f' (best {best.spec.key}: {_member_rejection(f, best)})' if best else ''))
    return Step('sub', title, StepState.NOT_RUN, f'could not decide: {why_not_measured(f, "S")[1]}')


def _owner_step(f: MMRFacts) -> Step:
    owner, v = f.a.owner, f.v
    title = '2. owner'
    if owner is None:
        return Step('owner', title, StepState.NOT_RUN, 'not run on a derived row' if f.derived else 'not run')
    if 'owner_coorbital' in owner.flags:
        return Step('owner', title, StepState.NOT_RUN, 'a 1:1 has no owner by construction')
    if owner.accepted is not None:
        drives = f'{owner.accepted.spec.key} drives a (R² {fmt(owner.accepted.r2_median, 2)})'
        if identity_route(owner):
            drives = _identity_claim(owner)
        cover = f.entry(OWNER_COVER)
        if 'owner_outside_episode' in v.flags and cover is not None:  # F7: the text follows the decision
            where = f'covers {compare(cover.value, ">=", cover.threshold, digits=2)} of σ’s decisive episode'
            return Step('owner', title, StepState.PASSED, f'{drives}, but {where}: it acts elsewhere, the tree goes on')
        return Step('owner', title, StepState.LEAF, f'{drives} → {v.status}, induced')
    if str(owner.check) in ('pass', 'fail'):
        best = _best_candidate(owner.candidates)
        text = f'{_plural(owner.n_candidates, "candidate")}, none drives a'
        if best is not None:
            text += f' (best {best.spec.key}: {_owner_rejection(f, best)})'
        return Step('owner', title, StepState.PASSED, text)
    return Step('owner', title, StepState.NOT_RUN, f'could not run: {why_not_measured(f, "W")[1]}')


def _kinematic_step(f: MMRFacts) -> Step:
    b1 = f.entry(B1_AMPLITUDE)
    kin = f.entry(KIN_VARPI)
    title = '3. kinematic'
    if f.rule == '3k2':
        return Step('kin', title, StepState.LEAF, f'{_k2_clause(f)} → {f.v.status}')
    if f.rule in ('3', '3b5'):
        if b1 is not None and b1.result is False:
            text = f'a does not carry the swing: B1 {compare(b1.value, ">=", b1.threshold)}'
        else:
            text = f'slow σ follows p ϖ (corr {fmt(kin.value if kin else None)}) with a flat a'
        return Step('kin', title, StepState.LEAF, f'{text} → {f.v.status}')
    text = f'a carries the swing: B1 {compare(b1.value, ">=", b1.threshold)}' if b1 is not None and finite(b1.value) else 'B1 not measured'
    if kin is None and str(f.v.angle_status) in ('libration', 'episodic'):
        text += '; the ϖ-test does not judge a libration'
    return Step('kin', title, StepState.PASSED, text)


def _drift_step(f: MMRFacts) -> Step:
    angle = str(f.v.angle_status)
    if angle != 'drift':
        return Step('drift', '4. drift', StepState.NOT_RUN, f'n/a: the centre does not drift ({angle})')
    if f.rule == '4':
        return Step('drift', '4. drift', StepState.LEAF, f'no mechanism explains the drift → {f.v.status}')
    return Step('drift', '4. drift', StepState.PASSED)


def _force_step(f: MMRFacts) -> Step:
    rule, title = f.rule, '5. force law + potential'
    if rule.startswith('5a') or rule in ('5b', '5e', '5f', '5g'):
        return Step('force', title, StepState.LEAF, origin_unknown_clauses(f) + f' → {f.v.status}')
    if strength_first(f):
        return Step('force', title, StepState.LEAF, f'read first, whatever L4 and R0 say: {branch_clause(f)} → {f.v.status}')
    if rule in BRANCH_LEAVES:
        return Step('force', title, StepState.LEAF, f'both not measured; {branch_clause(f)} → {f.v.status}')
    if not rule.startswith('5'):
        return Step('force', title, StepState.NOT_REACHED)
    if rule == '5u/6':
        return Step('force', title, StepState.PASSED, f'both not measured; {branch_clause(f)} → the geometry decides')
    l4, r0 = f.checks.get('a_response', ''), f.checks.get('potential', '')
    force = {'pass': 'the force law holds', 'fail': 'the force law fails (capped)'}
    potential = {'pass': 'the potential holds', 'fail': 'the potential fails'}
    parts = [
        force.get(l4, f'the force law is not measured ({short_reason(why_not_measured(f, "L4")[1])})'),
        potential.get(r0, f'the potential is not measured ({short_reason(why_not_measured(f, "R0")[1])})'),
    ]
    # the filter guard runs before the tree (W21 Q9): a check it withdrew was not measured for step 5 either
    conclusion = _force_conclusion(l4, r0, _direct_strength_clause(f))
    return Step('force', title, StepState.PASSED, '; '.join(parts) + f': {conclusion} → the angle’s own dynamics')


DIRECT_STRENGTH_ENTRY = strength_upper_name(DIRECT_BRANCH)  # decide.unmeasured_branch (i)


def _direct_strength_clause(f: MMRFacts) -> str:
    """Rule 5d/6 (i) with L4 and R0 both not measured: 'the strength by direct averaging decides (required / theory
    1.58 within [0.1, 10])'; '' when that rule did not pass the step."""
    entry = _held(f, DIRECT_STRENGTH_ENTRY)
    if not entry:
        return ''
    x = f.s.mmr_strength_check_ratio
    return f'the strength by direct averaging decides (required / theory {fmt(entry.value)} within [{fmt(1 / x)}, {fmt(x)}])'


def _force_conclusion(l4: str, r0: str, unmeasured: str = '') -> str:
    """Which of L4 (the force law) and R0 (the potential) passes step 5: both ('neither fails'), the one measured when
    the other is not ('the force law decides': 2024 WJ, whose 1:1 energy integral fails so R0 is not measured), or the
    one that holds when the other fails ('the potential carries it'); with neither measured, what passed it instead
    (`unmeasured`, rule 5d/6 (i): 1999 TJ10 19M-22)."""
    held = [name for name, check in (('the force law', l4), ('the potential', r0)) if check == 'pass']
    failed = [check for check in (l4, r0) if check == 'fail']
    if len(held) == 2:
        return 'neither fails'
    if len(held) == 1:
        return f'{held[0]} carries it' if failed else f'{held[0]} decides'
    if len(failed) == 2:
        return 'both fail'
    if failed:
        return (
            'the force law fails, the potential is not measured' if l4 == 'fail' else 'the potential fails, the force law is not measured'
        )
    return f'neither is measured; {unmeasured}' if unmeasured else 'neither is measured'


def _zone_step(f: MMRFacts) -> Step:
    if not f.rule.endswith(('/6', '/6u')):  # 6u: a union promoted after leaf 6 (`decide.promote_union`)
        return Step('zone', '6. dynamic leaf', StepState.NOT_REACHED)
    energy, note = f.a.energy, ''
    if energy is not None and energy.cycle_classes:
        n_zone = sum(1 for c in energy.cycle_classes if str(c) == 'zone')
        note = f'; {n_zone} of {len(energy.cycle_classes)} cycles in the separatrix zone'
        if energy.episodes_in_separatrix_zone:
            note += ' and the angle crosses it (episodes_in_separatrix_zone)'
    kind = union_kind(f)
    if kind == 'promoted':  # rule 6u: the episodic reading reached a dynamic transient
        outcome = 'episodic → transient; the union covers the record → resonant (6u)'
    else:  # a union the status kept reads as episodic there
        outcome = f'{"episodic" if kind == "kept" else f.v.angle_status} → {f.v.status}'
    return Step('zone', '6. dynamic leaf', StepState.LEAF, f'{outcome}{note}')


TREE_STEPS = (_subresonance_step, _owner_step, _kinematic_step, _drift_step, _force_step, _zone_step)


def mmr_steps(f: MMRFacts) -> List[Step]:
    """The steps of the status tree in their fixed order, with what each found; the steps after the leaf are
    'not reached'."""
    steps = [_shape_step(f)]
    if steps[0].state == StepState.LEAF:
        return steps
    if str(f.v.angle_status) == 'circulation':
        return steps + _circulation_steps(f)
    reached = False
    for build in TREE_STEPS:
        step = build(f)
        if reached:
            step = Step(step.key, step.title, StepState.NOT_REACHED)
        steps.append(step)
        reached = reached or step.state == StepState.LEAF
    return steps


def _episodes_clause(noun: str, eps: Episodes, settings) -> str:
    """Where an episodic angle librates, without the final period: its one episode ('<noun> librates from … to … only
    (x% of the record), numbers; gaps') or its several ('… in N separate episodes covering … (longest …)')."""
    longest = eps.longest
    if longest is None:
        return f'{noun} librates in part of the record only'
    span = f'{years(longest.get("t_start_yr"))}–{years(longest.get("t_end_yr"))}'
    if len(eps.accepted) == 1:
        text = (
            f'{noun} librates from {years(longest.get("t_start_yr"))} to {years(longest.get("t_end_yr"))} only '
            f'({percent(eps.covered_fraction())} of the record), {libration_numbers(longest, centre=False)}'
        )
        gaps = coverage_text(longest, settings)
        return text + (f'; {gaps}' if gaps else '')
    return (
        f'{noun} librates in {len(eps.accepted)} separate episodes covering {percent(eps.covered_fraction())} of the record '
        f'(longest {span}: {libration_numbers(longest, centre=False)})'
    )


def _episodic_sentence(f: MMRFacts) -> str:
    v, eps = f.v, f.episodes
    text = _episodes_clause('σ', eps, f.s)
    if not eps.accepted:
        return text + '.'
    rejected = eps.rejected_text()
    text += f'; {rejected}' if rejected else ''
    if _forced_wave_word(f):
        text += f'; those loops are a forced wave of another argument, so the status reads σ as {_tree_word(f)}'
    final = _text(v.final_regime)
    if final and final != str(v.angle_status) and finite(v.final_regime_start_yr):
        text += f'; from {years(v.final_regime_start_yr)} to the end {regime_phrase(final)}'
    return text + '.'


def decisive_centre_moves(f: MMRFacts) -> bool:
    """`orbit_type.centre_moves` of the deciding episode: the one test of the header's ', moving centre' and of the
    words 'fixed centre' / 'the centre is stationary' here (W23 F5)."""
    shift = (f.episodes.decisive or {}).get('centre_shift_over_amp')
    return finite(shift) and centre_moves(float(shift), f.s.mmr_l6_shift_full)


def _stationary_clause(f: MMRFacts) -> str:
    """The subtype stationary in words: the cycles' bands overlap (`geometry.centre_bands_overlap`); a fixed centre only
    when the header's test agrees (`decisive_centre_moves`), else the shift that makes the header say 'moving centre'."""
    if not decisive_centre_moves(f):
        return ' with a fixed centre'
    shift = f.episodes.decisive.get('centre_shift_over_amp')
    return (
        f' with a moving centre (shift / mean amplitude {compare(shift, ">", f.s.mmr_l6_shift_full)}; every cycle’s band '
        'still overlaps the others, so the subtype is stationary)'
    )


def _libration_sentence(f: MMRFacts) -> str:
    kind, decisive = union_kind(f), f.episodes.decisive or {}
    if kind == 'centre':  # the fine episode does not cover the record: its numbers are not the whole record's
        start, end = years(decisive.get('t_start_yr')), years(decisive.get('t_end_yr'))
        return (
            'σ librates over the whole record at the scale of the centre: episodes without a circulation between them '
            f'cover it there; at the fine scale the deciding episode runs {start}–{end}, {libration_numbers(decisive)}.'
        )
    if kind:
        many = _union_count(f)
        return (
            f'σ librates over the whole record through {many} without a circulation between them (union rule); the '
            f'deciding one {years(decisive.get("t_start_yr"))}–{years(decisive.get("t_end_yr"))}, {libration_numbers(decisive)}.'
        )
    centre = {
        'modulated': ' with a modulated centre',
        'stationary': _stationary_clause(f),
        'moving_centre': ' about the moving centre of its well',
        'regime_change': (
            ' through a change of its 1:1 regime, without a circulation between them'
            if f.coorbital
            else ' through episodes without a circulation between them'  # the union rule (W22)
        ),
    }.get(str(f.v.angle_subtype), '')
    regimes = _regimes_clause(f)
    if regimes:
        return f'σ librates {regimes}{centre} {libration_numbers(f.episodes.decisive or {})}.'
    return f'σ librates over the whole record{centre} {libration_numbers(f.episodes.decisive or {})}.'


def _regimes_clause(f: MMRFacts) -> str:
    """'in 2 episodes, 2.21–5.92 kyr and 7.99–17.49 kyr, cut by a step of its action (G2a); the deciding one' when a
    confirmed jump (of a, or G2a's) cut the libration into regimes: then it does not librate 'over the whole record' as
    one episode. '' otherwise."""
    g = f.a.geometry
    if g is None or not g.episodes_before_jump or not g.episodes:
        return ''
    runs = sorted(list(g.episodes_before_jump) + list(g.episodes), key=lambda run: run.t_start_yr)
    spans = ' and '.join(f'{years(run.t_start_yr)}–{years(run.t_end_yr)}' for run in runs)
    cuts = [jump for jump in g.jumps if not jump.kick]
    how = 'a step of its action (G2a)' if cuts and all(jump.action_step for jump in cuts) else 'jumps of a'
    return f'in {len(runs)} episodes, {spans}, cut by {how}; the deciding one'


def _circulation_sentence(f: MMRFacts) -> str:
    if str(f.v.angle_subtype) == 'alias':
        return (
            f'The output step folds σ: it turns {fmt(f.row.get("smp_step_rotation_deg_max"))}° per step, so the wrapped picture is '
            'the step’s, not the body’s; the unwrapped angle circulates.'
        )
    net = _net_shift(f, G4B_NET)
    if net:
        passed = fmt(f.row.get('geo_centre_max_pass_rev'))
        return f'σ circulates: its centre advances {net} revolutions net, in steps of at most {passed} revolution with small step-backs.'
    return f'σ circulates: its centre passes {fmt(f.row.get("geo_centre_max_pass_rev"))} revolutions in one direction.'


ANGLE_SENTENCES = {
    'libration': _libration_sentence,
    'episodic': _episodic_sentence,
    'circulation': _circulation_sentence,
    'drift': lambda f: (
        f'σ makes {_plural(len(f.episodes.accepted), "short librating stretch", "short librating stretches")} on a centre that '
        f'drifts {fmt(f.row.get("geo_centre_net_rev"))} rev over the record (the drift decides the word, step 3).'
        if f.rule == '3k2'
        else f'σ oscillates, but its centre drifts over {percent(f.row.get("geo_drift_fraction"))} of the record.'
    ),
    'near_separatrix': lambda f: 'σ has no libration: it '
    + ('stalls at the top of its potential' if str(f.v.angle_subtype) == 'pattern_b' else 'slips over the top')
    + ' in each revolution.',
    'insufficient_time': lambda f: (
        'σ rides on a slow centre that turns back once: the record holds part of one slow swing, too short to tell a '
        'drift from a slow libration.'
        if str(f.v.angle_subtype) == 'arc' and _slow_arc(f) is not None
        else f'σ has {fmt(f.row.get("geo_n_tp"))} turning points only: the record is too short to tell a libration.'
    ),
}


def _a_swing_change(row) -> str:
    """'grew 7.16×' / 'shrank 7.16×': geo_action_step_a_ratio is the larger side's swing of a over the smaller's, the
    larger side being the one of the larger half-amplitude (`geometry.action_step_of`), so it shrank on a step down."""
    down = finite(row.get('geo_action_step_amp_before_deg')) and finite(row.get('geo_action_step_amp_after_deg'))
    down = down and float(row['geo_action_step_amp_after_deg']) < float(row['geo_action_step_amp_before_deg'])
    return f'{"shrank" if down else "grew"} {fmt(row.get("geo_action_step_a_ratio"))}×'


def _one_potential(row) -> str:
    """' in one potential (that of the larger swings, 2 wells: T_obs/T_pred 0.956 before, 0.959 after)': G2a test 7
    (`geometry.regime_periods`); '' without its columns."""
    before, after, wells = (row.get(f'geo_action_step_{key}') for key in ('law_before', 'law_after', 'shape_wells'))
    if not (finite(before) and finite(after)):
        return ''
    shape = f', {_plural(int(wells), "well")}' if finite(wells) and int(wells) > 0 else ''
    return f' in one potential (that of the larger swings{shape}: worst-cycle T_obs/T_pred {fmt(before)} before, {fmt(after)} after)'


# G2a test 7's sub-tests (`geometry.T7_*`) in words: why one potential does not hold both regimes
T7_WORDS = {
    'isochronous': 'both regimes keep one period, as a forced wave would (the period law cannot tell)',
    'shape': 'V of the larger swings explains too little of their σ̇²',
    'covered': 'the smaller swings leave the range the larger ones sampled',
    'no_loop': 'a cycle has no loop of its energy in V',
    'period_law': 'a cycle\'s period is not the loop\'s',
    'turning_points': 'a cycle swings off the turning points of its loop',
    'energy_drift': 'a side\'s energy does not stay put',
    'wells_unstable': 'V\'s wells change with the shape cutoff',
}


def _regime_periods_failure(row) -> str:
    """'and did not cut: one potential does not hold both regimes (a cycle's period is not the loop's at the cutoff
    0.075 x the shorter period; worst-cycle T_obs/T_pred 0.98 before, 1.61 after)': G2a test 7 failed."""
    name, _, factor = str(row.get('geo_action_step_shape_failed') or '').partition('@')
    where = f' at the cutoff {factor} × the shorter period' if factor else ''
    laws = ''
    before, after = row.get('geo_action_step_law_before'), row.get('geo_action_step_law_after')
    if finite(before) and finite(after):
        laws = f'; worst-cycle T_obs/T_pred {fmt(before)} before, {fmt(after)} after'
    return f'and did not cut: one potential does not hold both regimes ({T7_WORDS.get(name, name)}{where}{laws})'


def action_step_sentence(f: MMRFacts) -> str:
    """G2a (`geometry.ActionStep`): 'Rule G2a cut the libration at 6.96 kyr: its half-amplitude stepped from 146° to
    12° within one half-cycle (5.92–7.99 kyr), the swing of a shrank 7.16×, the cycle energy h by −0.541 (|Δh| > 0.1):
    two energy levels in one potential (that of the larger swings, 2 wells: T_obs/T_pred 0.956 before, 0.959 after),
    analysed as regimes like a jump of a (provisional: 1 energy cycle before the step).' A step the guard withheld
    (`geometry.STEP_WITHHELD`, `analyse._action_step_guard`): 'Rule G2a found a step of the libration at 6.93 kyr (146°
    to 12.2°) and did not cut: the cut leaves check_mmr_strength, failed on the whole record, unmeasured, and would have
    made it resonant/low.' '' otherwise. Every number a geo_action_step_* column."""
    row = f.row
    if row.get('geo_action_step_failed') == 'withheld':
        checks = str(row.get('geo_action_step_withheld_checks') or '').replace(';', ', ')
        return (
            f'Rule G2a found a step of the libration at {years(row.get("geo_action_step_time_yr"))} '
            f'({fmt(row.get("geo_action_step_amp_before_deg"))}° to {fmt(row.get("geo_action_step_amp_after_deg"))}°) and '
            f'did not cut: the cut leaves {checks}, failed on the whole record, unmeasured, and would have made it '
            f'{row.get("geo_action_step_withheld_verdict")}.'
        )
    if row.get('geo_action_step_failed') == 'regime_periods':
        return (
            f'Rule G2a found a step of the libration at {years(row.get("geo_action_step_time_yr"))} '
            f'({fmt(row.get("geo_action_step_amp_before_deg"))}° to {fmt(row.get("geo_action_step_amp_after_deg"))}°) '
            f'{_regime_periods_failure(row)}.'
        )
    if not (finite(row.get('geo_action_step_n_cuts')) and float(row['geo_action_step_n_cuts']) > 0):
        return ''
    before, after = row.get('geo_action_step_cycles_before'), row.get('geo_action_step_cycles_after')
    provisional = ''
    if finite(row.get('geo_action_step_provisional')) and float(row['geo_action_step_provisional']) > 0:
        fewest, side = (before, 'before') if finite(before) and finite(after) and float(before) <= float(after) else (after, 'after')
        provisional = f' (provisional: {_plural(int(fewest), "energy cycle")} {side} the step, {f.s.libration_jump_cycles} confirm it)'
        if finite(row.get('geo_action_step_held_out')) and not float(row['geo_action_step_held_out']):
            provisional = ' (provisional: one cycle of the larger swings, its period law read in sample)'
    return (
        f'Rule G2a cut the libration at {years(row.get("geo_action_step_time_yr"))}: its half-amplitude stepped from '
        f'{fmt(row.get("geo_action_step_amp_before_deg"))}° to {fmt(row.get("geo_action_step_amp_after_deg"))}° within one '
        f'half-cycle ({years(row.get("geo_action_step_t_from_yr"))}–{years(row.get("geo_action_step_t_to_yr"))}), the swing '
        f'of a {_a_swing_change(row)}, the cycle energy h by {fmt(row.get("geo_action_step_dh"))} '
        f'(|Δh| > {fmt(row.get("geo_action_step_dh_threshold"))}): two energy levels{_one_potential(row)}, analysed as '
        f'regimes like a jump of a{provisional}.'
    )


def mmr_angle_sentence(f: MMRFacts) -> str:
    """What σ does over the record (the author's wording: 'over the whole record' when it covers it)."""
    build = ANGLE_SENTENCES.get(str(f.v.angle_status))
    return build(f) if build else ''


# the leaves of the branch where the force law and the potential with a are both not measured (decide.unmeasured_branch)
BRANCH_LEAVES = ('5s', '5s*', '5o', '5n')


def _entry_text(f: MMRFacts, name: str, unit: str = '') -> str:
    """'value op threshold' of a decision-path entry as it came out ('' when the entry is not on the path)."""
    return entry_compare(f.entry(name), unit)


def strength_first(f: MMRFacts) -> bool:
    """Rule 5s decided before the force law and the potential were read (decide._implausible_strength_first) while
    one of them was measured (failed): with both not measured the branch's wording holds as before."""
    measured = any(f.checks.get(key) in ('pass', 'fail') for key in ('a_response', 'potential'))
    return f.rule == '5s' and measured and any(name.startswith(STRENGTH_FIRST) for name in f.path)


def _strength_entry(f: MMRFacts, name: str) -> str:
    """The 5s entry `name` as recorded by either place of the rule (before step 5 or in the branch)."""
    return _entry_text(f, f'{STRENGTH_FIRST} {name}') or _entry_text(f, f'5s {name}')


def _implausible_clause(f: MMRFacts) -> str:
    """5s: the ratio (two-body) or the ratio and the residual after the small-divisor envelope (three-body)."""
    strength = f.a.strength
    direct = _strength_entry(f, STRENGTH_DIRECT)
    if direct:
        if strength is not None and str(strength.method) == 'direct_harmonic':
            return f'the period needs {direct} × the strength of the resonant harmonic (extended precision)'
        return f'the period needs {direct} × the directly averaged strength'
    ratio = _strength_entry(f, STRENGTH_THREE_BODY)
    residual = _strength_entry(f, STRENGTH_DIVISORS)
    b = f'B = n/|Δ| {fmt(strength.divisor_b)}' if strength is not None else 'B'
    if strength is not None and strength.divisor_capped:  # W22 R-DIV: the candidate turns slower than σ swings
        b += f', capped at n·min(P, P_cyc)/2π: {strength.divisor_candidate} turns slower than σ swings'
    return f'the period needs {ratio} × μ_i μ_j e^q, and even with small divisors up to B² ({b}) still {residual} × μ_i μ_j e^q B²'


def _divisor_open_clause(f: MMRFacts) -> str:
    """5s*: why the small divisors may close the gap."""
    strength = f.a.strength
    ratio = _entry_text(f, f'5s {STRENGTH_THREE_BODY}')
    residual = _entry_text(f, f'5s {STRENGTH_DIVISORS}')
    kind = _text(strength.divisor_kind) if strength is not None else ''
    if kind == 'crosses_zero':  # B at the swing cap n min(P, P_cyc) / 2 pi (W22 R-DIV)
        why = (
            f'the two-body angle {strength.divisor_candidate} passes its exact commensurability, and with B capped at '
            f'n·min(P, P_cyc)/2π it needs only {residual} × μ_i μ_j e^q B²'
        )
    elif kind == 'no_period':
        why = 'no period bounds the two-body rates'
    elif kind == 'no_candidate':  # B < n / bound: the residual is a lower bound only
        why = f'no two-body angle comes within the rate bound, so the residual is only bounded below ({residual} ×)'
        return f'the period needs {ratio} × μ_i μ_j e^q, but {why}: the envelope cannot rule out the gap closing'
    else:
        why = f'with small divisors up to B² it needs only {residual} × μ_i μ_j e^q B²'
    return f'the period needs {ratio} × μ_i μ_j e^q, but {why}: the gap may close'


def _objection_clause(f: MMRFacts) -> str:
    """5o: the measured objections (O1, F) that failed."""
    parts = []
    onset = f.entry(O1_ONSET)
    if onset is not None and onset.result is False:
        # O1's background skips the earlier episodes of its group (`response.onset_group`: ids, latest first)
        group = [float(i) for i in _text((f.episodes.decisive or {}).get('onset_group')).split(';') if i.strip()]
        group = [int(i) for i in group if finite(i)]
        where = f', the wave before episode {group[-1]}, which no circulation separates from it' if group else ''
        parts.append(f'a strong a-wave precedes the episode (before/inside {_entry_text(f, onset.name)}{where})')
    well = f.entry(F_IN_WELL)
    if well is not None and well.result is False:
        parts.append('no observed swing librates in the directly averaged well')
    for name in ('5o F: required / theory <= x', '5o F: required / theory >= 1/x'):
        entry = f.entry(name)
        if entry is not None and entry.result is False:
            parts.append(f'the strength disagrees (required/theory {_entry_text(f, name)})')
    return 'a measured test argues against the libration: ' + '; '.join(parts) if parts else 'a measured test argues against the libration'


def _geometry_only_clause(f: MMRFacts) -> str:
    """5u/6: nothing measures the angle's own force."""
    strength = f.a.strength
    if strength is not None and str(strength.check) == 'pass':  # the e^q scale: an order of magnitude only
        return 'the strength agrees only with an order-of-magnitude scale (μ e^q), not with a computed force'
    return f'nothing else measures this angle’s own force (strength: {why_not_measured(f, "F")[1]})'


def _one_cycle_clause(f: MMRFacts) -> str:
    cycles = _entry_text(f, INDEPENDENT_CYCLES)
    return f'the episode holds one centre only (independent cycles {cycles}): nothing to compare'


BRANCH_CLAUSES = {
    '5s': _implausible_clause,
    '5s*': _divisor_open_clause,
    '5o': _objection_clause,
    '5n': _one_cycle_clause,
    '5u/6': _geometry_only_clause,
}


def branch_clause(f: MMRFacts) -> str:
    """What decided the branch where neither the force law nor the potential with a is measured (rules 5s, 5s*, 5o,
    5n, 5u/6 of `decide.unmeasured_branch`), with the numbers of its decision-path entries."""
    build = BRANCH_CLAUSES.get(f.rule)
    return build(f) if build else ''


def _branch_sentence(f: MMRFacts) -> str:
    """The status sentence of the branch's leaves 5s, 5s*, 5o, 5n (5s also before step 5: `strength_first`)."""
    if strength_first(f):
        head = f'{status_head(f.v.status)}: {branch_clause(f)}'
        return head + ': no libration of this angle’s own force, whatever the force law and the potential with a say.'
    text = f'{status_head(f.v.status)}: the force law and the potential with a are not measured; {branch_clause(f)}'
    if f.rule == '5s':
        return text + ': no libration of this angle’s own force.'
    return text + '.'


def origin_unknown_clauses(f: MMRFacts) -> str:
    """5a/5a*/5b/5e/5f/5g: what each of the force law and the potential said, one clause each (5g: and the failed
    identity that decides)."""
    s, energy = f.s, f.a.energy
    l4, r0 = f.checks.get('a_response'), f.checks.get('potential')
    clauses = []
    if r0 == 'fail' and energy is not None and str(energy.sigma_only) == 'fail':
        what = []
        if finite(energy.period_law_med) and abs(energy.period_law_med - 1) > s.separatrix_period_tol:
            what.append(
                f'the cycle periods do not follow the pendulum law (T_obs/T_pend {fmt(energy.period_law_med)}, '
                f'need 1 ± {fmt(s.separatrix_period_tol)})'
            )
        if finite(energy.well_coherence) and energy.well_coherence < s.separatrix_phase_coherence:
            what.append(f'the well moves (coherence {compare(energy.well_coherence, ">=", s.separatrix_phase_coherence)})')
        clauses.append('the energy does not fit a pendulum potential' + (': ' + ' and '.join(what) if what else ''))
    elif r0 == 'fail':
        clauses.append(f'the energy disagrees with the potential from a ({_potential_detail(f)[0]})')
    elif r0 == 'pass':
        clauses.append(f'the energy fits the potential from a ({_potential_detail(f)[0]})')
    else:
        clauses.append(f'the potential was not measured ({why_not_measured(f, "R0")[1]})')
    if l4 == 'fail':
        clauses.append(f'ȧ does not follow the force law ({_force_detail(f)[0]})')
    elif l4 == 'pass':
        clauses.append(f'ȧ follows the force law ({_force_detail(f)[0]})')
    else:
        clauses.append(f'the force law cannot be judged ({why_not_measured(f, "L4")[1]})')
    response = f.a.response
    if f.rule == '5g' and response is not None:
        clauses.append(
            f'σ̇ ≠ c (a − a*): identity slope/c {fmt(response.identity_slope_ratio)}, corr {fmt(response.identity_corr)} '
            f'(need |slope/c − 1| ≤ {fmt(s.mmr_identity_slope_tol)}, corr ≥ {fmt(s.mmr_identity_corr_min)})'
        )
    return '; '.join(clauses)


def _dynamic_claims(f: MMRFacts) -> str:
    """What the force law and the potential said, as the reason for an own-dynamics verdict."""
    s, a = f.s, f.a
    l4, r0 = f.checks.get('a_response'), f.checks.get('potential')
    claims = []
    if l4 == 'pass':
        law = 'the 1:1 force law' if f.coorbital else 'ȧ = K sin(σ − φ0)'
        if taken_harmonic_law(a.response) is not None:
            law = 'ȧ = Σ_k [A_k sin kσ + B_k cos kσ] + C, the force of a well that is no sinusoid'
        detail, need = _force_detail(f)
        claims.append(f'a responds to σ as a pendulum ({law}: {detail}; need {need})')
    elif l4 == 'fail' and r0 == 'pass':
        claims.append(f'the force law fails ({_force_detail(f)[0]}), but the potential with a carries the libration')
    elif l4 == 'fail':
        claims.append(f'the force law fails ({_force_detail(f)[0]})')
    if r0 == 'pass':
        claims.append(f'the energy fits the potential built from a ({_potential_detail(f)[0]})')
    if not claims:
        b1 = a.response.b1_ratio if a.response is not None else None
        direct = _direct_strength_clause(f)
        decided = f'; {direct}' if direct else ''
        rest = 'nothing measured contradicts a libration of this angle'
        if f.alias_unmeasured:  # R-ALIAS-A: what failed was read as not measured, it still was measured
            failed = ' and '.join(CHECK_WORDS[name] for name in f.alias_unmeasured)
            rest = f'nothing else measured contradicts a libration of this angle ({failed} failed on a folded step)'
        return (
            f'the force law is not measured ({why_not_measured(f, "L4")[1]}){decided}; a carries the swing '
            f'(B1 {_with_source(compare(b1, ">=", s.mmr_b1_ratio_min), f, "b1_source")}) and {rest}'
        )
    text = ' and '.join(claims)
    if l4 not in ('pass', 'fail'):
        text += f'; the force law is not measured: {why_not_measured(f, "L4")[1]}'
    return text


def _foreign_clauses(f: MMRFacts) -> str:
    """'no member holds the centre and no other resonance drives a (…)', '(… not triggered)'."""
    s, sub, owner = f.s, f.a.subresonances, f.a.owner
    foreign, not_run = [], []
    if sub is not None and str(sub.check) == 'pass':
        foreign.append('no member of the multiplet holds the centre')
    if owner is not None and str(owner.check) == 'pass':
        best = _best_candidate(owner.candidates)
        detail = f' (best of {owner.n_candidates}: R² {compare(best.r2_median, "<", s.mmr_w_r2_none, digits=2)})' if best else ''
        foreign.append(f'no other resonance drives a{detail}')
    if sub is None and not f.derived:
        not_run.append('the subresonance scan was not triggered')
    if owner is not None and 'owner_coorbital' in owner.flags:
        not_run.append('a 1:1 has no owner')
    text = ('; ' + ' and '.join(foreign)) if foreign else ''
    return text + (' (' + '; '.join(not_run) + ')' if not_run else '')


def _dynamic_sentence(f: MMRFacts) -> str:
    v = f.v
    if f.rule == '5u/6':  # the geometry alone: origin unknown
        text = (
            f'{status_head(v.status)}: by the geometry alone, the force law and the potential with a are not measured and '
            f'{branch_clause(f)}; the origin stays unknown{_foreign_clauses(f)}'
        )
    else:
        text = f'{status_head(v.status)}: {_dynamic_claims(f)}{_foreign_clauses(f)}'
    if str(v.status) == 'transient':
        text += '. Transient, not resonant, because the libration does not cover the record'
    elif str(v.reason) == 'separatrix_zone' and 'near_separatrix_without_zone' in v.caps:  # G3 without a measured zone
        text += '. Near the separatrix by the pattern of σ; the separatrix zone is not measured'
    elif str(v.reason) == 'separatrix_zone':
        text += '. Near the separatrix: the cycles lie in the separatrix zone'
    return text + '.'


def _subresonance_sentence(f: MMRFacts) -> str:
    """Rule C (`decide.member_status`): the members of one commensurability are one resonance, so the catalogue row
    takes the status of the member that librates (its own row); a nodal member without a row of its own leaves the
    row non_resonant."""
    head = status_head(f.v.status)
    accepted = f.a.subresonances.accepted if f.a.subresonances is not None else None
    if accepted is None:
        return f'{head}: the subresonance {f.v.owner} was taken as the origin of the libration.'
    coherence = compare(accepted.phi0_coherence, '>=', f.s.mmr_subresonance_phi0_coh_min)
    found = (
        f'the subresonance scan found the member {accepted.spec.key} of the same commensurability with a fixed centre '
        f'(φ0 coherence {coherence}), while the centre of this angle moves with it'
    )
    own_row = any(d.resonance.key == accepted.spec.key for d in f.a.derived) or str(f.v.status) != 'non_resonant'
    if not own_row:
        return f'{head} for this angle: {found}; the member has no row of its own, so this angle is reported non_resonant.'
    return f'{head}: {found}. The members of one commensurability are one resonance, so this row takes the status of {accepted.spec.key} (its own row).'


def _induced_sentence(f: MMRFacts) -> str:
    owner = f.a.owner
    if identity_route(owner):
        return (
            f'Non-resonant for this angle: {_identity_claim(owner)}, while σ’s own three-body term would need '
            f'{sci(f.a.strength.ratio)} × μ_i μ_j e^q; this angle only follows {f.v.owner} through the identity.'
        )
    r2 = owner.accepted.r2_median if owner is not None and owner.accepted is not None else None
    return (
        f'Non-resonant for this angle: {f.v.owner} drives a (R² {compare(r2, ">=", f.s.mmr_owner_r2_min, digits=2)}); '
        'this angle only follows it through the identity.'
    )


def _k2_clause(f: MMRFacts) -> str:
    """K2 (rule 3k2) in words: the slow centre follows the secular part, the episodes are a ripple on it; or K2b, a
    kinematic libration whose centre moves more than its amplitude."""
    shift = _held(f, K2B_SHIFT)
    if shift:
        return (
            f'the oscillation sits in the perihelion term and its centre moves {compare(shift.value, ">", shift.threshold)} × '
            'its amplitude: a kinematic drift of the centre, not a libration'
        )
    corr, share = f.entry(K2_CORR), f.entry(K2_SHARE)
    return (
        f'the slow centre drifts with the perihelion term (corr {compare(corr.value if corr else None, ">=", f.s.mmr_kin_corr_min)}) '
        f'while a does not answer σ, and the episodes are a ripple on that drift '
        f'(swing / centre range {compare(share.value if share else None, "<", f.s.mmr_kin_ripple_share_max)})'
    )


def _kinematic_sentence(f: MMRFacts) -> str:
    if f.rule == '3k2':
        return f'Non-resonant: {_k2_clause(f)}.'
    b1 = f.entry(B1_AMPLITUDE)
    if b1 is not None and b1.result is False:
        return f'Non-resonant: a does not carry the swing (B1 {compare(b1.value, ">=", b1.threshold)}): it sits in the perihelion term.'
    return 'Non-resonant: the slow angle follows p ϖ while a stays flat: the swing sits in the perihelion term.'


def _alias_sentence(f: MMRFacts) -> str:
    far = f.entry(C5_FAR)
    distance = compare(far.value if far is not None else None, '>=', f.s.mmr_c5_far_ratio_min)
    if str(f.v.status) == 'invalid':
        return f'Invalid: several alias branches, or ⟨a⟩ near a* ({distance} a-ranges): a libration hidden by the step cannot be excluded.'
    return (
        f'Non-resonant: one alias branch only, and ⟨a⟩ lies {distance} a-ranges from a*: the body is far from this resonance, '
        'so the folded picture is no libration.'
    )


# base_period_source -> where the period P of the baseline comes from (`decide.baseline_period`)
BASE_SOURCE_WORDS = {
    'tp_spacing': 'twice the median spacing of the turning points',
    'arc_slow': 'twice the longest stretch of the slow centre without a turn',
    'two_omega': 'the 2ω cycle the angle follows',
    'modulation': 'the e-cycle 2π/|g − g5| the angle follows',
    'stall_rate': 'one revolution at the rate of the stall',
    'stall_force': 'the shortest libration period the force bound of the stall allows',
    'cycle': 'the measured cycle',
    'b5': 'the shortest libration period B5 allows',
}
# base_cause -> what is missing when the record already holds the time P needs
BASE_CAUSE_WORDS = {
    'too_few_tp': 'the run of turning points stops before a cycle closes',
    'centre_moves': 'the centre moves along the run of turning points',
    'edge': 'the centre moves along the run of turning points',
    'over_the_top': 'σ goes over the top next to the run of turning points',
    'no_closed_cycle': 'no cycle of the run of turning points returns about its centre',
    'period_fits_record': 'the angle does not come back within it',
}


def _baseline_sentence(f: MMRFacts) -> str:
    """baseline_needed: the next check (N8: the total length of the next record, at most MMR_NEXT_CHECK_SPAN_FACTOR x
    the record) and the time the period estimate asks for, from which period (F8); or, when time is not what is
    missing, why (the cutoff hides a quasi-satellite libration; P already fits the record)."""
    qs = _held(f, QS_STALL_ENTRY)
    if qs:
        return (
            f'Uncertain: the cutoff hides the libration (P_QS against P_c: {compare(qs.value, "<", qs.threshold, " yr")}), '
            'so a longer record would not decide; the step must hold the quasi-satellite cutoff.'
        )
    required, period = f.row.get('base_required_kyr'), f.row.get('base_period_yr')
    source = BASE_SOURCE_WORDS.get(str(f.row.get('base_period_source') or ''), '')
    if not finite(required):
        return 'Uncertain: the record is too short to decide.'
    need = years(float(required) * KYR)
    basis = f'P = {years(period)} from {source}' if source else f'P = {years(period)}'
    cause = str(f.row.get('base_cause') or '')
    if cause:
        return (
            f'Uncertain, but not for lack of time: the record ({years(f.row.get("smp_span_yr"))}) already holds the {need} '
            f'that {basis} needs; {BASE_CAUSE_WORDS.get(cause, cause.replace("_", " "))}.'
        )
    check, estimate = f.row.get('extend_to_years'), f.row.get('base_extend_to_years')
    if not finite(check):
        return f'Uncertain: the record is too short to decide: about {need} needed ({basis}).'
    budget = f'at most {fmt(f.s.mmr_next_check_span_factor)} × the record; ' if finite(estimate) and check < estimate else ''
    return f'Uncertain: the record is too short to decide; next check: {years(check)} ({budget}about {need} estimated from {basis}).'


STATUS_SENTENCES = {
    'dynamic_libration': _dynamic_sentence,
    'dynamic_episode': _dynamic_sentence,
    'geometric_libration': _dynamic_sentence,  # rule 5u/6: `_dynamic_sentence` words the geometry alone
    'geometric_episode': _dynamic_sentence,
    'separatrix_zone': _dynamic_sentence,
    'subresonance_librates': _subresonance_sentence,
    'induced_libration': _induced_sentence,
    'kinematic_libration': _kinematic_sentence,
    'kinematic_drift': _kinematic_sentence,
    'alias': _alias_sentence,
    'origin_unknown': lambda f: (
        _branch_sentence(f)
        if f.rule in BRANCH_LEAVES
        else f'Uncertain: {origin_unknown_clauses(f)}. σ librates, but no measurement establishes that this resonance makes it.'
    ),
    'implausible_strength': _branch_sentence,
    'circulation': lambda f: 'Non-resonant: σ circulates.',
    'no_resonant_potential': lambda f: (
        f'Non-resonant: σ turns back at times, but the energy does not fit a potential from a ({_potential_detail(f)[0]}).'
    ),
    'modulated_circulation': lambda f: 'Non-resonant: σ stalls at the top of its potential in each turn (pattern B) without a libration.',
    'drift_mechanism_unknown': lambda f: 'Uncertain: σ oscillates about a drifting centre and no tested mechanism explains the drift.',
    'baseline_needed': _baseline_sentence,
}


def mmr_status_sentence(f: MMRFacts) -> str:
    """Why this status: the rule that decided it, in plain words, with its numbers."""
    build = STATUS_SENTENCES.get(str(f.v.reason))
    if build:
        return build(f)
    return f'{status_head(f.v.status)}: {str(f.v.reason).replace("_", " ")}.'


def _unexplained_lifted(f: MMRFacts) -> bool:
    """Whether the mark exit_and_capture took the cap transient_unexplained off this verdict (`sep_unexplained_lifted`,
    recorded by the analysis; rows written before 2026-10-18 have none)."""
    value = f.row.get('sep_unexplained_lifted', f.a.extra_columns.get('sep_unexplained_lifted'))
    return finite(value) and float(value) == 1.0


def _edges_seen(f: MMRFacts) -> str:
    """The exit and entry of a check_mmr_exit pass (none suspected): each one the record shows (an observed one points
    to the angle-shape step, which prints its arc); 'no exit or entry seen' only when there is none."""
    geometry = f.a.geometry
    seen = []
    for kind in ('entry', 'exit'):
        event = getattr(geometry, kind, None) if geometry is not None else None
        if event is None:
            continue
        # an observed edge is said with its arc in the angle-shape step (`edge_clauses`): no number twice
        seen.append(f'{kind} observed (the angle-shape step)' if event.observed else f'{kind} at {years(event.time_yr)}')
    return '; '.join(seen) + ', not suspected' if seen else 'no exit or entry seen'


def mmr_info(f: MMRFacts, physics: Sequence[PhysicsSection]) -> List[str]:
    """Checks measured (pass/fail) that neither a margin nor a step of this verdict read: names only. `physics` =
    `physics_sections(f)`: the info line 'strength pass/fail' is left out when section A prints the strength."""
    read = set(f.v_columns)
    out = []
    if 'exit_and_capture' in f.v.flags and f.a.energy is not None:
        lifted = _unexplained_lifted(f)
        out.append(
            f'exit and capture: σ left a libration {_plural(f.a.energy.n_exits, "time")} and came back '
            f'{_plural(f.a.energy.n_captures, "time")} (a mark; no status, no mechanism'
            + ('; lifts the cap "' + CAP_WORDS[str(Cap.TRANSIENT_UNEXPLAINED)] + '")' if lifted else ')')
        )
    spread = f.entry(SHORT_PERIOD_CAP)
    if spread is not None and spread.result is False and 'short_period_aliased' not in f.v.caps:
        out.append(
            f'the step folds short periods, but no cap: the fast spread of σ_f {_entry_text(f, SHORT_PERIOD_CAP)} and the '
            f'short-period scatter of a {_entry_text(f, SHORT_PERIOD_A)} × the a swing'
        )
    shown_physics = any(row.key in ('strength', 'scale') for section in physics for row in section.rows)  # section A only
    for name, value in f.checks.items():
        if value not in ('pass', 'fail') or CHECK_MARGIN.get(name) in f.margins or name in TREE_CHECKS:
            continue
        if name == 'strength' and shown_physics:
            continue
        if read & set(CHECK_COLUMNS.get(name, ())):
            continue
        if name == 'exit' and value == 'pass':
            out.append(_edges_seen(f))
        elif name == 'slow_libration':
            ratio = f.a.response.b5_ratio if f.a.response is not None else None
            out.append(f'B5 ratio {fmt(ratio, 2)} (read only for drift and kinematic verdicts)')
        elif name == 'kinematic':
            out.append('ϖ-kinematic test (not read for a libration)')
        else:
            out.append(f'{CHECK_WORDS.get(name, name)} {value}')
    return out


# --- the physics block (R1 §3: own pendulum and what makes the cycles; R5 §5: the owner lines; CS11)


PHYSICS_ICONS = {True: '✓', False: '✗', None: '?'}
PHYSICS_NA = '–'


@dataclass
class PhysicsRow:
    """One line of a physics section: icon, label, text. `numbers` = (column, printed, scale): every number the text
    prints from a column, as printed and the factor from the column's unit to the printed one (tested against the
    record); `columns` = every column the row shows (the numbers below the explanation leave them out)."""

    key: str
    icon: str
    label: str
    text: str
    read_by_verdict: bool = False  # a number the decision compared: bold
    grey: bool = False  # context, not a test (legality, the σ wave): grey
    numbers: Tuple[Tuple[str, str, float], ...] = ()
    columns: Tuple[str, ...] = ()


@dataclass
class PhysicsSection:
    title: str
    subtitle: str
    rows: List[PhysicsRow]


def sci(value: Any) -> str:
    """A number of any size in 3 significant digits ('5.8e-10', '2.74e-5', '0.0197', '1.39'), '—' when missing."""
    if not finite(value):
        return '—'
    value = float(value)
    if value == 0 or 1e-3 <= abs(value) < 1e4:
        return fmt(value)
    mantissa, exponent = f'{value:.3g}'.split('e') if 'e' in f'{value:.3g}' else (f'{value:.3g}', '')
    return f'{mantissa}e{int(exponent)}' if exponent else mantissa


class _Printer:
    """Prints numbers of the row `row` and records what it printed (`PhysicsRow.numbers`)."""

    def __init__(self, row: Mapping[str, Any]):
        self.row = row
        self.numbers: List[Tuple[str, str, float]] = []
        self.columns: List[str] = []

    def value(self, column: str) -> float:
        return _float(self.row.get(column))

    def has(self, *columns: str) -> bool:
        return all(finite(self.row.get(column)) for column in columns)

    def show(self, *columns: str) -> None:
        self.columns += [column for column in columns if column not in self.columns]

    def num(self, column: str, scale: float = 1.0, form=fmt) -> str:
        text = form(self.value(column) * scale)
        self.numbers.append((column, text, scale))
        self.show(column)
        return text

    def cmp(self, column: str, op: str, threshold: float, unit: str = '', digits: int = DIGITS) -> str:
        """`compare` of a column against a threshold, the column's number recorded as printed."""
        self.show(column)
        text, left = _comparison(self.value(column), op, threshold, unit, digits)
        if left is not None:
            self.numbers.append((column, left, 1.0))
        return text

    def span(self, low: str, high: str, scale: float = 1.0, form=fmt, unit: str = '') -> str:
        """'a–b unit', or 'a unit' when both print the same."""
        left, right = self.num(low, scale, form), self.num(high, scale, form)
        return (left if left == right else f'{left}–{right}') + unit

    def time_span(self, low: str, high: str, unit_yr: float = YEAR) -> str:
        """Two time columns in one unit (kyr when the larger reaches 1 kyr): 'a–b kyr'."""
        larger = max(abs(self.value(low)), abs(self.value(high))) * unit_yr
        scale, unit = (1 / KYR, 'kyr') if larger >= KYR else (1.0, 'yr')
        return self.span(low, high, unit_yr * scale, unit=f' {unit}')

    def time(self, column: str, unit_yr: float = YEAR) -> str:
        """A time column (in yr, or kyr with `unit_yr` = KYR) in yr or kyr, as `years` would print it."""
        value_yr = self.value(column) * unit_yr
        scale, unit = (1 / KYR, 'kyr') if abs(value_yr) >= KYR else (1.0, 'yr')
        return f'{self.num(column, unit_yr * scale)} {unit}'

    def take(self, key: str, icon: str, label: str, text: str, **kwargs) -> PhysicsRow:
        row = PhysicsRow(key, icon, label, text, numbers=tuple(self.numbers), columns=tuple(self.columns), **kwargs)
        self.numbers, self.columns = [], []
        return row


def _check_icon(value: Any) -> str:
    """✓ pass, ✗ fail, ? anything else (a check value or a bool)."""
    if isinstance(value, bool):
        return PHYSICS_ICONS[value]
    return {'pass': PHYSICS_ICONS[True], 'fail': PHYSICS_ICONS[False]}.get(str(value), PHYSICS_ICONS[None])


STRENGTH_REASON_WORDS = {
    'no_required_strength': 'no cycle to read a required strength from',
    'no_planet_orbit': "the planet's orbit is not known",
    'close_approach_at_centre': 'the body comes too close to the planet at the centre: outside the averaged problem',
    'no_elements_in_window': 'no elements in the window of the snapshots',
    'no_stable_centre': 'the averaged potential has no stable centre',
    'quadrature_not_converged': 'the averaged potential did not converge',
    'non_finite': 'a quadrature node fell on the planet',
    'precision_exhausted': 'the resonant harmonic is below the extended precision',
}
POTENTIAL_REASON_WORDS = {
    'off': 'the potential route is switched off',
    'three_body_owner': 'no direct theory for a three-body owner (second order)',
    'coorbital_owner': 'a 1:1 owner has its own force law',
    'retrograde': 'no averaged potential for a retrograde owner',
    'catalogue': "the planet's elements were not saved",
    'no_planet_mass': "the planet's mass is not known",
    'potential_cost': 'skipped for its cost (MMR_OWNER_POTENTIAL_MAX_KERNELS)',
    'not_converged': 'the averaged potential did not converge',
    'close_approach': 'the body comes too close to the planet',
    'no_windows': 'no window to evaluate it in',
}
SOURCE_RUN = 'rejected_run:'


def _t_obs_source(text: str) -> str:
    """'episode #3' or 'a rejected run' from str_T_obs_source (its window is the snapshots' window, printed before)."""
    if text.startswith('episode:'):
        return f'episode #{text.split(":", 1)[1]}'
    if text.startswith(SOURCE_RUN):
        return 'a rejected run of turning points, not an episode'
    return text or 'the decisive run'


def _direct_rows(f: MMRFacts) -> List[PhysicsRow]:
    """Section A of a two-body angle (direct averaging or the resonant harmonic): period, strength, width, slow part,
    legality."""
    s, strength, p = f.s, f.a.strength, _Printer(f.row)
    rows = []
    icon = _check_icon(strength.check)
    read = 'str_ratio' in f.v_columns
    harmonic = str(strength.method) == 'direct_harmonic'
    if p.has('str_P_lib_theory_min_kyr', 'str_P_lib_theory_max_kyr'):
        window = f'{p.time("str_snap_t0_yr")}–{p.time("str_snap_t1_yr")}' if p.has('str_snap_t0_yr', 'str_snap_t1_yr') else ''
        n = int(strength.n_evaluations)
        well = f', {strength.n_consistent} of {n} in the well' if not harmonic else ''
        e = f', ⟨e⟩ {p.span("str_e_snap_min", "str_e_snap_max")}' if p.has('str_e_snap_min', 'str_e_snap_max') else ''
        theory = f'T0_th {p.time_span("str_P_lib_theory_min_kyr", "str_P_lib_theory_max_kyr", KYR)}'
        where = f' ({window}: {_plural(n, "snapshot")}{well}{e})' if window else ''
        observed = ''
        if p.has('str_T_obs_yr'):
            observed = f' against the observed T0 {p.time("str_T_obs_yr")} ({_t_obs_source(_text(f.row.get("str_T_obs_source")))})'
        p.show('str_T_obs_source', 'str_n_evaluations', 'str_n_consistent')
        rows.append(p.take('period', icon, 'own period', theory + where + observed))
    if p.has('str_mu_eff_req', 'str_ratio_all_min', 'str_ratio_all_max'):
        limit, implausible = s.mmr_strength_check_ratio, s.mmr_strength_implausible_ratio
        low, high = p.value('str_ratio_all_min'), p.value('str_ratio_all_max')
        inside = high >= 1 / limit and low <= limit
        theory = 'μ_P a 2|C| of the resonant harmonic' if harmonic else 'μ_th'
        text = (
            f'μ_req {p.num("str_mu_eff_req", form=sci)} against {theory} '
            f'{p.span("str_mu_eff_theory_all_min", "str_mu_eff_theory_all_max", form=sci)}: '
            f'μ_req/μ_th = (T0_th/T_obs)² = {p.span("str_ratio_all_min", "str_ratio_all_max", form=sci)} '
            f'{"overlaps" if inside else "lies outside"} [{fmt(1 / limit)}, {fmt(limit)}]'
        )
        if low >= implausible:
            text += f'; ≥ {fmt(implausible)} × is implausible'
        p.show('str_ratio', 'str_ratio_min', 'str_ratio_max', 'str_implausible', 'str_mu_eff_theory_min', 'str_mu_eff_theory_max')
        rows.append(p.take('strength', icon, 'strength', text, read_by_verdict=read))
    if p.has('str_sep_halfwidth_min_au', 'str_sep_halfwidth_max_au', 'rsp_b1_meas_au'):
        # a pendulum's a swings at most the zone's half-width; the width goes as √μ, so the strength check's band [1/x, x]
        # on μ is √x on the width: the same tolerance, no new number
        wave, width = p.value('rsp_b1_meas_au'), p.value('str_sep_halfwidth_max_au')
        ratio, tolerance = wave / width if width > 0 else float('nan'), math.sqrt(s.mmr_strength_check_ratio)
        fits = holds(ratio, '<=', tolerance)
        text = (
            f'zone half-width 2ω0/|c| {p.span("str_sep_halfwidth_min_au", "str_sep_halfwidth_max_au", form=sci, unit=" au")} '
            f'against the a-wave ±{p.num("rsp_b1_meas_au", form=sci)} au (B1): ratio {compare(ratio, "<=", tolerance)} '
            f'(√ of the strength band)' + ('' if fits else ': a libration of this force cannot move a so far')
        )
        rows.append(p.take('width', _check_icon(fits) if fits is not None else PHYSICS_ICONS[None], 'width in a', text))
    if p.has('rsp_slow_adot_R2') or p.has('rsp_b5_ratio'):
        parts = []
        if p.has('rsp_slow_adot_R2'):
            parts.append(f'slow ȧ on sin σ_slow R² {p.num("rsp_slow_adot_R2")}')
        if p.has('rsp_b5_ratio'):
            parts.append(f'B5 ratio {p.num("rsp_b5_ratio")} (≥ 1: a slower libration excluded)')
        rows.append(p.take('slow', '', 'slow part', '; '.join(parts), grey=True))
    elements = 'the planet’s saved orbit' if str(strength.planet_elements) == 'saved' else 'the planet’s mean orbit, its phase scanned'
    p.show('str_planet_elements', 'str_method')
    rows.append(
        p.take(
            'legality',
            '',
            'legality',
            f'μ_req is what σ would need if these cycles were its own libration; the range is over snapshots, not a '
            f'confidence interval; frozen 3-D first-order average with {elements}',
            grey=True,
        )
    )
    return rows


def _scale_rows(f: MMRFacts) -> List[PhysicsRow]:
    """Section A of a three-body (or retrograde) angle: the e^q scale only, and the small divisors when evaluated."""
    strength, p = f.a.strength, _Printer(f.row)
    scale = 'μ_i μ_j e^q' if strength.three_body else 'μ e^q'
    if not p.has('str_mu_eff_req'):  # no cycle to read a required strength from: nothing to compare
        return []
    if not strength.eq_legal:  # the strength check row says why e^q is no scale here
        return []
    text = f'{scale} {p.num("str_mu_scale", form=sci)} is a scale only (the coefficient is unknown)'
    if p.has('str_mu_eff_req', 'str_ratio'):
        text += f': μ_req {p.num("str_mu_eff_req", form=sci)}, ratio {p.num("str_ratio", form=sci)}'
    if _text(f.row.get('str_divisor_kind')) and p.has('str_divisor_B', 'str_divisor_residual'):
        bound = 'at least ' if _text(f.row.get('str_divisor_kind')) == 'no_candidate' else ''
        text += (
            f'; with small divisors up to B² (B = n/|Δ| {p.num("str_divisor_B", form=sci)}) μ_req is still {bound}'
            f'{p.num("str_divisor_residual", form=sci)} × {scale} B²'
        )
        p.show('str_divisor_kind', 'str_divisor_candidate')
    p.show('str_method', 'str_eq_legal')
    read = 'str_ratio' in f.v_columns
    return [p.take('scale', _check_icon(strength.check), 'strength', text, read_by_verdict=read)]


def own_pendulum_section(f: MMRFacts) -> Optional[PhysicsSection]:
    """Section A, 'Is σ the own pendulum of <key>?': can these cycles be σ's own libration (the strength its period
    needs against the averaged potential's, its width in a); None when nothing was measured for it."""
    key, strength = f.a.resonance.key, f.a.strength
    title = f'Is σ the own pendulum of {key}?'
    if f.coorbital:
        text = 'judged by its own 1:1 force law (averaged direct + indirect potential; K/K_theory = 1 is the planet’s mass, check L4)'
        return PhysicsSection(title, '1:1', [PhysicsRow('law', PHYSICS_NA, 'force law', text)])
    if strength is None:
        return None
    reason = _text(strength.not_measured_reason)
    if str(strength.method) == 'eq_scale':
        rows = _scale_rows(f)
        return PhysicsSection(title, 'order-of-magnitude scale', rows) if rows else None
    if reason and reason != 'centre_outside_well':
        if reason == 'no_required_strength' and str(f.v.angle_status) not in ('libration', 'episodic'):
            return None
        words = STRENGTH_REASON_WORDS.get(reason, reason.replace('_', ' '))
        return PhysicsSection(
            title, 'not measured', [PhysicsRow('strength', PHYSICS_ICONS[None], 'strength', words, columns=('str_not_measured_reason',))]
        )
    subtitle = 'resonant harmonic in extended precision' if str(strength.method) == 'direct_harmonic' else 'directly averaged potential'
    if reason == 'centre_outside_well':
        subtitle += '; no observed swing librates in its well'
    return PhysicsSection(title, subtitle, _direct_rows(f))


def _intervals(text: str) -> str:
    """own_intervals 't0-t1:regime;…' [yr] -> '9.06–88.1 kyr (forced wave), …' (LIST_MAX, then 'N more')."""
    items = []
    for item in [part for part in text.split(';') if part]:
        span, _, kind = item.partition(':')
        start, _, end = span.partition('-')
        if finite(start) and finite(end):
            items.append(f'{years(float(start))}–{years(float(end))} ({kind.replace("_", " ")})')
    hidden = len(items) - LIST_MAX
    return ', '.join(items[:LIST_MAX]) + (f' and {hidden} more' if hidden > 0 else '')


def _owner_rows(f: MMRFacts) -> List[PhysicsRow]:
    """Section B for an accepted owner: where it acts, the identity, its period, its force on a, its theory, the
    amplitude, the σ wave (measured only) and the second capture (R1 §3, R5 §5)."""
    s, owner, p = f.s, f.a.owner, _Printer(f.row)
    key = owner.accepted.spec.key
    forced = _text(f.row.get('own_detail')) == 'owner_forced_wave'
    rows = []
    cover = ''
    if p.has('own_cover'):
        cover = (
            f' and covers {p.cmp("own_cover", ">=", s.mmr_owner_cover_min)} of σ’s decisive episode'
            f' ({p.time("own_decisive_t0_yr")}–{p.time("own_decisive_t1_yr")})'
        )
    route = _text(f.row.get('own_route'))
    where = _intervals(_text(f.row.get('own_intervals'))) or 'the record'
    p.show('own_intervals', 'own_route', 'own_name', 'own_best', 'own_detail', 'own_regime')
    inside = not finite(f.row.get('own_cover')) or p.value('own_cover') >= s.mmr_owner_cover_min
    rows.append(p.take('owner', _check_icon(inside), 'owner', f'{key} on {where}' + (f', by its {route} route' if route else '') + cover))
    if _text(f.row.get('own_identity_text')):
        p.show('own_identity_text', 'own_psi_def')
        text = f'σ = {f.row["own_identity_text"]}: both angles hold the same λ, so a wave of λ shows in both (one fact, not two)'
        rows.append(p.take('identity', '', 'identity', text))
    rows += _identity_route_rows(f, p)  # empty unless the route 'identity' accepted the owner
    if forced and p.has('own_rate_rad_kyr', 'own_owner_period_yr'):
        text = f'φ turns at {p.num("own_rate_rad_kyr")} rad/kyr → {p.time("own_owner_period_yr")}'
        if p.has('own_freq_period_yr'):
            text += f'; the dominant a-wave {p.time("own_freq_period_yr")}'
        rows.append(p.take('period', PHYSICS_ICONS[True], 'period', text))
    if p.has('own_R2'):
        rows.append(_owner_force_row(f, p, route))
    if p.has('own_lever_owner', 'own_phi0_coherence', 'own_lever_gain'):
        gain = p.value('own_lever_gain')
        text = (
            f'C_B {p.num("own_phi0_coherence")} against L_B {p.num("own_lever_owner")} if σ drove a: '
            f'gain {p.cmp("own_lever_gain", ">=", s.mmr_owner_phi0_coh_gain)}'
            + (f'; L_σ {p.num("own_lever_sigma")}' if p.has('own_lever_sigma') else '')
        )
        if route == 'phase':
            rows.append(p.take('phases', _check_icon(gain >= s.mmr_owner_phi0_coh_gain), 'phases', text, read_by_verdict=True))
        else:  # accepted by another route: the phases are context
            rows.append(p.take('phases', '', 'phases', text, grey=True))
    rows += _owner_theory_rows(f, p, forced)
    if p.has('own_sigma_wave_deg'):
        text = f'σ on sin/cos φ: {p.num("own_sigma_wave_deg")}°'
        if p.has('own_sigma_wave_R2'):
            text += f' (R² {p.num("own_sigma_wave_R2")})'
        rows.append(p.take('sigma_wave', '', 'σ wave', text + ', ripple-corrected; measured only, nothing predicts it here', grey=True))
    rows += _capture_rows(f, p)
    return rows


def _owner_force_row(f: MMRFacts, p: _Printer, route: str) -> PhysicsRow:
    """The owner's force law on a: R², φ0 coherence and σ's own law; context only for the route 'identity'."""
    s = f.s
    if route == 'identity':  # the route measures nothing on a
        text = f'ȧ = K sin(φ − φ0): R² {p.num("own_R2", form=lambda v: fmt(v, 2))} (not read by the identity route)'
        return p.take('force', '', 'force on a', text, grey=True)
    text = (
        f'ȧ = K sin(φ − φ0): R² {p.cmp("own_R2", ">=", s.mmr_owner_r2_min, digits=2)}'
        + (f', φ0 coherence {p.num("own_phi0_coherence")}' if p.has('own_phi0_coherence') else '')
        + (f'; σ’s own law on the same windows: R² {p.num("own_sigma_R2", form=lambda v: fmt(v, 2))}' if p.has('own_sigma_R2') else '')
    )
    p.show('own_R2_windows')
    read = 'own_R2' in f.v_columns
    return p.take('force', _check_icon(p.value('own_R2') >= s.mmr_owner_r2_min), 'force on a', text, read_by_verdict=read)


def _identity_route_rows(f: MMRFacts, p: _Printer) -> List[PhysicsRow]:
    """The owner route 'identity' (W22 R-OWNER-ID): σ − (k_σ/k_φ)φ slow on σ's episode (exact, so nearly any φ passes),
    φ's own libration over that episode and its directly averaged strength, which decide, and σ's required strength."""
    s, rows = f.s, []
    if p.has('own_id_coef_dev', 'own_id_resid'):
        text = (
            f'identity route: on σ’s episode σ − (k_σ/k_φ)φ is slow, c/(k_σ/k_φ) − 1 = {p.num("own_id_coef_dev", form=sci)} '
            f'(limit ±{fmt(s.mmr_owner_identity_coef_tol)}), residual {p.cmp("own_id_resid", "<=", s.mmr_owner_identity_resid_max)} '
            'of σ’s half-amplitude; the identity is exact, so this holds for nearly any φ: the rows below decide'
        )
        ok = abs(p.value('own_id_coef_dev')) <= s.mmr_owner_identity_coef_tol and p.value('own_id_resid') <= s.mmr_owner_identity_resid_max
        rows.append(p.take('identity_fit', _check_icon(ok), 'identity', text, read_by_verdict=True))
    if p.has('own_id_phi_cover'):
        status = _text(f.row.get('own_id_phi_angle_status')) or 'its angle'
        text = f'φ’s own analysis: {status}, its librating episodes cover {p.cmp("own_id_phi_cover", ">=", s.mmr_owner_cover_min)} of σ’s episode'
        p.show('own_id_phi_angle_status', 'own_id_phi_span_deg', 'own_id_n_pass', 'own_id_n_analysed')
        rows.append(
            p.take(
                'identity_phi', _check_icon(p.value('own_id_phi_cover') >= s.mmr_owner_cover_min), 'φ librates', text, read_by_verdict=True
            )
        )
    if p.has('own_id_phi_str_ratio'):
        text = f'φ’s directly averaged strength passes: required / theory {p.num("own_id_phi_str_ratio")}'
        if p.has('str_ratio'):
            ratio_min = sci(s.mmr_owner_identity_ratio_min)
            text += f'; σ needs {p.num("str_ratio", form=sci)} × μ_i μ_j e^q (the route opens at {ratio_min})'
        rows.append(p.take('identity_strength', PHYSICS_ICONS[True], 'φ strength', text, read_by_verdict=True))
    return rows


def _owner_theory_rows(f: MMRFacts, p: _Printer, forced: bool) -> List[PhysicsRow]:
    """The owner's force from its averaged potential (two-body), the bound on σ's own term, the forced amplitude."""
    reason = _text(f.row.get('own_pot_reason'))
    p.show('own_pot_reason')
    if reason == 'three_body_owner':
        text = POTENTIAL_REASON_WORDS[reason]
        if p.has('own_strength_ratio'):
            text += f'; force scale ratio {p.num("own_strength_ratio", form=sci)} (no coefficient)'
            p.show('own_K_theory_au_yr', 'own_K_au_yr')
        return [p.take('theory', PHYSICS_NA, 'theory', text)]
    if reason:
        words = POTENTIAL_REASON_WORDS.get(reason, reason.replace('_', ' '))
        return [p.take('theory', PHYSICS_ICONS[None], 'theory', f'not evaluated: {words}')]
    if not p.has('own_pot_R2', 'own_pot_scale_lo', 'own_pot_scale_hi'):
        return []
    rows = [_owner_potential_row(f, p)]
    if p.has('own_sigma_force_bound', 'own_sigma_period_min_yr'):
        rows.append(_own_term_row(f, p))
    if forced and p.has('own_wave_da_pred_au', 'own_wave_da_meas_au'):
        rows.append(_amplitude_row(f, p))
    return rows


def _owner_potential_row(f: MMRFacts, p: _Printer) -> PhysicsRow:
    """The owner's averaged potential against ȧ: R², the scale s, K_th and φ0_th against the fit."""
    s, limit = f.s, f.s.mmr_owner_potential_scale_max
    r2, low, high = p.value('own_pot_R2'), p.value('own_pot_scale_lo'), p.value('own_pot_scale_hi')
    inside = low >= 1 / limit and high <= limit
    text = (
        f'its averaged potential predicts ȧ: R² {p.cmp("own_pot_R2", ">=", s.mmr_owner_r2_min)}, scale '
        f'{p.span("own_pot_scale_lo", "own_pot_scale_hi")} {"within" if inside else "outside"} [{fmt(1 / limit)}, {fmt(limit)}]'
    )
    if p.has('own_K_direct_min', 'own_K_direct_max', 'own_K_over_K_direct_min', 'own_K_over_K_direct_max'):
        text += (
            f'; K_th {p.span("own_K_direct_min", "own_K_direct_max", form=sci, unit=" au/yr")}'
            + (f', φ0_th {p.num("own_phi0_direct_deg")}°' if p.has('own_phi0_direct_deg') else '')
            + f'; fitted/predicted {p.span("own_K_over_K_direct_min", "own_K_over_K_direct_max")}'
            + (f', |Δφ0| ≤ {p.num("own_phi0_diff_max_deg")}°' if p.has('own_phi0_diff_max_deg') else '')
        )
    p.show('own_pot_scale', 'own_pot_scale_worst', 'own_pot_n_windows', 'own_K_direct_au_yr', 'own_pot_min_distance_hill')
    read = any(column in f.v_columns for column in ('own_pot_scale_worst', 'own_pot_R2'))
    return p.take('theory', _check_icon(r2 >= s.mmr_owner_r2_min and inside), 'theory', text, read_by_verdict=read)


def _own_term_row(f: MMRFacts, p: _Printer) -> PhysicsRow:
    """The bound on σ's own term, max |s − 1|, and its own period T/√(|v/u|·bound) (R5 §1.8); below the model's error
    the bound is that error and the period is taken at it."""
    bound, error = p.value('own_sigma_force_bound'), f.s.mmr_owner_potential_model_error
    if bound < error:
        # own_sigma_period_min_yr = T / sqrt(|v/u| bound): the same T at the model's error
        period = p.value('own_sigma_period_min_yr') * math.sqrt(bound / error) if bound > 0 else float('nan')
        text = f'σ’s own force ≤ {fmt(error)} of it (model limit; measured {p.num("own_sigma_force_bound")}) → own period ≥ {years(period)}'
        p.show('own_sigma_period_min_yr')
    else:
        text = f'σ’s own force ≤ {p.num("own_sigma_force_bound")} of it → own period ≥ {p.time("own_sigma_period_min_yr")}'
    cycles = _decisive_period(f, bound)
    if finite(cycles):
        text += f' (σ’s decisive period: {years(cycles)})'
    return p.take('own_term', '', 'σ’s own term', text)


def _amplitude_row(f: MMRFacts, p: _Printer) -> PhysicsRow:
    """The forced wave of a, δa = K_th/ν predicted against measured; the linear formula holds for ν ≫ ω0."""
    limit, linear = f.s.mmr_owner_potential_scale_max, f.s.mmr_forced_wave_linear_ratio
    pred, meas = p.value('own_wave_da_pred_au'), p.value('own_wave_da_meas_au')
    ratio = pred / meas if meas > 0 else float('nan')
    text = f'δa = K_th/ν {p.num("own_wave_da_pred_au", form=sci)} au predicted, {p.num("own_wave_da_meas_au", form=sci)} au measured'
    ok = finite(ratio) and 1 / limit <= ratio <= limit
    if p.has('own_nu_over_omega0'):
        nu = p.value('own_nu_over_omega0')
        text += f'; ν/ω0 {p.cmp("own_nu_over_omega0", ">=", linear)}: ' + (
            'a linear forced wave' if nu >= linear else 'not a linear forced wave'
        )
        p.show('own_omega0_direct_rad_yr')
        ok = ok and nu >= linear
    return p.take('amplitude', _check_icon(ok), 'amplitude', text)


PSI_DEF = re.compile(r'^\s*(-?\d+)\*sigma\s*([+-])\s*(-?\d+)\*phi\s*$')


def _decisive_period(f: MMRFacts, bound: float) -> float:
    """The period of σ's decisive episode the owner's bound was read with: own_sigma_period_min_yr = T / √(|v/u|·bound),
    u and v from own_psi_def ('5*sigma-27*phi'); nan when it cannot be read back."""
    match = PSI_DEF.match(_text(f.row.get('own_psi_def')))
    period = _float(f.row.get('own_sigma_period_min_yr'))
    if match is None or not (finite(period) and bound > 0):
        return float('nan')
    u, v = int(match.group(1)), int(match.group(3))
    return period * math.sqrt(abs(v / u) * bound) if u else float('nan')


def _capture_rows(f: MMRFacts, p: _Printer) -> List[PhysicsRow]:
    """R5 §1.7: x = Ψ/u of the owner's block over σ's episode, its sweep and energy fit -> second capture."""
    s = f.s
    verdict = _text(f.row.get('own_second_capture'))
    if not verdict:
        return []
    psi = _text(f.row.get('own_psi_def')) or 'uσ − vφ'
    parts = [f'Ψ = {psi}']
    if p.has('own_psi_turns'):
        parts.append(f'{p.num("own_psi_turns")} turns')
    if p.has('own_psi_sweep_deg'):
        parts.append(f'Ψ/u sweeps {p.cmp("own_psi_sweep_deg", ">=", s.mmr_owner_capture_sweep_deg, "°")}')
    if p.has('own_psi_energy_R2'):
        parts.append(f'energy fit R² {p.cmp("own_psi_energy_R2", "<", s.mmr_owner_r2_min)}')
    p.show('own_psi_sweep_deg', 'own_psi_energy_R2', 'own_second_capture', 'own_adiabatic', 'own_psi_def')
    excluded = verdict == 'excluded'
    if len(parts) == 1:
        parts.append('not measurable on the owner’s blocks over σ’s episode')
    text = ', '.join(parts) + f' → second capture {verdict}'
    read = any(column in f.v_columns for column in ('own_psi_sweep_deg', 'own_psi_energy_R2'))
    return [p.take('capture', PHYSICS_ICONS[True] if excluded else PHYSICS_ICONS[None], 'second capture', text, read_by_verdict=read)]


def cycles_section(f: MMRFacts) -> Optional[PhysicsSection]:
    """Section B, 'What makes the cycles?': the accepted owner and its evidence, or why there is none."""
    owner, title = f.a.owner, 'What makes the cycles?'
    if owner is None:
        if f.derived or str(f.v.angle_status) not in ('libration', 'episodic'):
            return None
        return PhysicsSection(
            title, 'owner search not run', [PhysicsRow('owner', PHYSICS_ICONS[None], 'owner', 'the owner search did not run')]
        )
    if 'owner_coorbital' in owner.flags:  # the step '2. owner' says it; section A holds the 1:1 law
        return None
    if owner.accepted is not None:
        forced = _text(f.row.get('own_detail')) == 'owner_forced_wave'
        what = 'the forced wave' if forced else 'the libration'
        subtitle = f'{what} of {owner.accepted.spec.key} (owner search: {_plural(owner.n_candidates, "candidate")})'
        if str(Flag.OWNER_OUTSIDE_EPISODE) in f.v.flags:
            subtitle += '; it acts outside σ’s episode, so it is not the origin'
        return PhysicsSection(title, subtitle, _owner_rows(f))
    state, why = _skip_owner(f)
    if str(owner.check) not in ('pass', 'fail'):
        icon = PHYSICS_NA if state == CheckState.SKIPPED else PHYSICS_ICONS[None]
        return PhysicsSection(title, 'owner search without a verdict', [PhysicsRow('owner', icon, 'owner', why, columns=('own_flags',))])
    best = _best_candidate(owner.candidates)
    p = _Printer(f.row)
    text = f'{_plural(owner.n_candidates, "candidate")}, none accepted'
    if best is not None:
        text += (
            f'; best {best.spec.key} R² {compare(best.r2_median, "<", f.s.mmr_w_r2_none, digits=2)}'
            if best.r2_median < f.s.mmr_w_r2_none
            else (f'; best {best.spec.key} R² {fmt(best.r2_median, 2)}, rejected: {best.reason.replace("_", " ")}')
        )
    p.show('own_best', 'own_R2', 'own_n_candidates')
    return PhysicsSection(title, 'no other resonance drives a', [p.take('owner', PHYSICS_ICONS[False], 'owner', text)])


def physics_sections(f: MMRFacts) -> List[PhysicsSection]:
    """The physics block of the right column (R1 §3–4, R5 §5): A 'Is σ the own pendulum of <key>?' and B 'What makes
    the cycles?'. Every number comes from a column of `f.row`, every threshold from the settings; display only."""
    return [section for section in (own_pendulum_section(f), cycles_section(f)) if section is not None and section.rows]


def physics_columns(sections: Sequence[PhysicsSection]) -> List[str]:
    """Every column the physics block shows (the numbers below the explanation leave them out)."""
    return list(dict.fromkeys(column for section in sections for row in section.rows for column in row.columns))


def explain_mmr(analysis, settings, row: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]) -> VerdictText:
    """The explanation of one MMR verdict. `row` = summary ⋈ numbers (summary.csv over mmr_numbers.csv),
    `episodes` = episodes.csv rows of the angle."""
    f = MMRFacts(analysis, settings, row, episodes)
    v = f.v
    checks = mmr_checks(f)
    subtitle = ' · '.join(
        part
        for part in (
            f'reason {v.reason}',
            f'origin {v.libration_origin}' if str(v.libration_origin) else '',
            f'owner {v.owner}' if v.owner else '',
            f'rule {f.rule}' if f.rule else '',
        )
        if part
    )
    grey = {}
    if f.checks.get('a_response') == 'not_measured' or (f.margins.get('L4') is not None and f.margins['L4'].margin == Margin.NOT_MEASURED):
        why = why_not_measured(f, 'L4')[1].split(':')[0]
        grey.update({column: why for column in L4_COLUMNS})
    if f.checks.get('potential') == 'not_measured':
        grey['sep_VE_VF_agree_fraction'] = why_not_measured(f, 'R0')[1].split(':')[0]
    head = [' '.join(part for part in (mmr_angle_sentence(f), action_step_sentence(f)) if part), mmr_status_sentence(f)]
    physics = physics_sections(f)
    return _verdict_text(v, subtitle, head, mmr_steps(f), checks, mmr_info(f, physics), grey, physics, mmr_angle_badge(f))


def mmr_angle_badge(f: MMRFacts) -> str:
    """The badge's angle word: 'libration (stationary)' says the cycles' bands overlap; when the deciding episode's
    centre moves (`decisive_centre_moves`, the header's ', moving centre') it reads 'libration (stationary bands, moving
    centre)', so the badge and the header agree (2003 TX9 3E+2J-10)."""
    if str(f.v.angle_subtype) == 'stationary' and decisive_centre_moves(f):
        return f'{f.v.angle_status} (stationary bands, moving centre)'
    return angle_text(f.v)


def _verdict_text(verdict, subtitle: str, head: Sequence[str], steps, checks, info, grey=None, physics=(), angle='') -> VerdictText:
    """A `VerdictText` (MMR, secular, ZLK): the head sentences, then the confidence and the action, and the columns the
    decision read in bold; `angle` the badge's angle word when not `angle_text`."""
    sentences = list(head) + [confidence_sentence(verdict, checks), action_sentence(verdict)]
    return VerdictText(
        str(verdict.status),
        angle or angle_text(verdict),
        str(verdict.confidence),
        subtitle,
        [sentence for sentence in sentences if sentence],
        list(steps),
        checks,
        arithmetic_line(verdict, checks),
        list(info),
        read_columns(verdict),
        dict(grey or {}),
        final_regime_text(verdict),
        list(physics),
    )


# --------------------------------------------------------------------------------------------- secular and ZLK


LONG_PERIOD_MARGIN_NAMES = {
    'N': 'more than three turning points',
    'S': 'centre stationary (bands overlap)',
    'F': 'libration slower than 2 P_c (filter guard)',
    'P': 'centre passes ≥ 1 rev at every smoothing',
    'G': 'free-ω gate: cycles to spare',
}


def _long_n(margin, settings, where: str) -> Tuple[str, str, str]:
    note = f'{fmt(margin.value)} turning points: the fewest that make a libration' if margin.margin == Margin.THIN else ''
    return f'{fmt(margin.value)} turning points {where}', f'> {settings.libration_min_turning_points}', note


def _long_s(margin, settings, where: str) -> Tuple[str, str, str]:
    note = 'one cycle gives one centre: nothing to compare' if margin.margin == Margin.NOT_MEASURED else ''
    return f'{fmt(margin.value)} independent centre estimate(s)', '≥ 2, bands overlap', note


def _long_f(margin, settings, where: str) -> Tuple[str, str, str]:
    note = 'the libration is faster than 2 P_c: the filter shapes it' if margin.margin == Margin.NOT_MEASURED else ''
    detail = f'P_cyc / P_c {compare(margin.value, ">=", settings.filter_min_libration_cutoffs)}'
    return detail, f'≥ {fmt(settings.filter_min_libration_cutoffs)}', note


LONG_PERIOD_DETAILS = {
    'N': _long_n,
    'S': _long_s,
    'F': _long_f,
    'P': lambda margin, settings, where: (f'smallest pass {fmt(margin.value)} rev', '≥ 1', ''),
    # n_cycles = baseline / P_cyc of the decisive osculating scale (not turns of the free omega)
    'G': lambda margin, settings, where: (f'{fmt(margin.value)} cycles of the osculating ω (T / P_cyc)', 'the minimum + 10%', ''),
}


def _long_period_margin_row(key: str, margin, settings, whole_record: bool) -> CheckRow:
    where = 'over the whole record' if whole_record else 'in the deciding libration'
    build = LONG_PERIOD_DETAILS.get(key)
    detail, need, note = build(margin, settings, where) if build else (fmt(margin.value), '', '')
    state, counts = margin_state(margin.margin), margin_counts(margin.margin, margin.margin_class)
    if margin.margin == Margin.NOT_MEASURED:
        note = note or 'not measured'
        detail = _unused_detail(detail)
    return CheckRow(key, LONG_PERIOD_MARGIN_NAMES.get(key, key), state, detail, need, counts, note)


def _response_words(analysis) -> str:
    """'e responds to the angle', 'sin(i/2) responds ...' or 'e and sin(i/2) respond ...': the elements the secular angle
    contains (`Response.variables`), which check_secular_e_response measured."""
    response = analysis.response
    names = {'e': 'e', 'i': 'sin(i/2)'}
    variables = [names[name] for name in (response.variables if response is not None else ()) if name in names] or ['e']
    return f'{" and ".join(variables)} {"responds" if len(variables) == 1 else "respond"} to the angle'


def _long_period_angle_sentence(analysis, episodes: Episodes, settings) -> str:
    v = analysis.verdict
    angle = str(v.angle_status)
    noun = 'The proper angle' if str(v.resonance_type) == 'secular' else 'The angle'
    if angle == 'libration':
        return f'{noun} librates over the whole record {libration_numbers(episodes.decisive or {})}.'
    if angle == 'episodic':
        return _episodes_clause(noun, episodes, settings) + '.'
    if angle == 'circulation':
        return f'{noun} circulates.'
    if angle == 'drift':
        return f'{noun} oscillates about a drifting centre.'
    if angle == 'near_separatrix':
        return f'{noun} has no libration: it slips over the top.'
    if angle == 'insufficient_time':
        return f'{noun} has too few turning points for the record to tell a libration.'
    return ''


GATE_WORDS = {
    'confirmed': 'the free ω librates too: a confirmed Lidov-Kozai libration',
    'kinematic': 'the free ω winds, so the osculating libration is that of the forced eccentricity: kinematic',
    'no_verdict': 'the gate gave no verdict',
}


def _long_period_status_sentence(analysis, settings, episodes: Sequence[Mapping[str, Any]] = ()) -> str:
    v = analysis.verdict
    status, angle, reason = str(v.status), str(v.angle_status), str(v.reason)
    head = status_head(status)
    gate = getattr(analysis, 'gate', None)
    if gate is not None and _text(gate.outcome):
        words = GATE_WORDS.get(gate.outcome, gate.outcome.replace('_', ' '))
        # n_cycles is the record over P_cyc of the decisive osculating scale, the gate's cycle count; not free-ω turns
        cycles = (
            f' ({fmt(gate.n_cycles)} cycles of the osculating ω, T / P_cyc, need ≥ {fmt(gate.min_cycles)})' if finite(gate.n_cycles) else ''
        )
        return f'{head}: {words}{cycles}.'
    if reason == 'gate_unavailable':
        return f'{head}: the free-ω gate could not run, so the Lidov-Kozai libration has no verdict.'
    if angle == 'libration':
        episode = Episodes(episodes, float('nan'), analysis.decisive_episode_id).decisive or {}
        gaps = coverage_text(episode, settings).replace('it misses', 'the libration misses', 1)
        return (
            f'{head}: {gaps or "the libration covers the record"}; a secular status rests on the shape of the angle '
            '(no force-law or owner test exists for it).'
        )
    if angle == 'episodic':
        return f'{head}: the libration does not cover the record.'
    return f'{head}: {reason.replace("_", " ")}.'


EI_WORDS = {'zlk': 'ZLK (antiphase, constant H)', 'forced_beat': 'forced beat (in phase)', 'none': 'neither law'}


def _ei_line(ei) -> str:
    """'e-i at 6 kyr: forced beat (in phase): i against e 328°, r +0.78, δi/δe 1.2 × the beat law (0.36 × H)'."""
    if ei.coupling == 'not_measured':
        return f'e-i coupling not measured ({ei.reason.replace("_", " ") or "no period"})'
    where = f'at {years(ei.period_yr)}' + (' (ω harmonic)' if ei.period_source == 'circulation_harmonic' else '')
    return (
        f'e-i {where}: {EI_WORDS.get(ei.coupling, ei.coupling)}: i against e {ei.phase_deg:.0f}°, r {ei.band_r:+.2f}, '
        f'δi/δe {fmt(ei.ratio_h)} × the H law, {fmt(ei.ratio_beat)} × the beat law (a column only; coupling is not libration)'
    )


def _attribution_line(attribution, gate) -> str:
    """The forced mode that holds a kinematic libration, or why none is named; '' when the gate is not kinematic."""
    if gate is None or gate.outcome != 'kinematic':
        return ''
    if not attribution.attributed:
        return f'forcing not attributed ({attribution.reason.replace("_", " ")})'
    return (
        f'held by the forced mode {attribution.mode}: |c| {fmt(attribution.forcing_e)}, e_free/|c| '
        f'{fmt(attribution.free_over_forcing)}, beat 2π/|b_ϖ − g| {years(attribution.beat_period_yr)} = P_cyc, node − mode '
        f'{attribution.node_minus_mode_arcsec_yr:+.2f}″/yr ({fmt(attribution.node_minus_mode_turns)} turn)'
    )


def _centre_line(line) -> str:
    """'slow line of the centre: 53.8 kyr, ±21.6°: forced beat g6-g5 (r 0.99, model ±22.8°)'; '' when not measured."""
    if line is None or not finite(line.period_yr):
        return ''
    origin = line.origin.replace('forced_beat:', 'forced beat ').replace('forced_wave:', 'forced wave ')
    model = ''
    if finite(line.model_r):
        model = (
            f' (r {line.model_r:.2f}'
            + (f', model ±{math.degrees(line.model_half_amp_rad):.1f}°' if finite(line.model_half_amp_rad) else '')
            + ')'
        )
    return f'slow line of the centre: {years(line.period_yr)}, ±{math.degrees(line.half_amp_rad):.3g}°: {origin}{model}'


def zlk_column_lines(analysis) -> List[str]:
    """The measured facts of a ZLK row that change no status (e-i coupling, attribution, the centre's slow line), and
    the slow line of a modulated secular angle; one line each, '' lines left out."""
    lines = []
    zlk = getattr(analysis, 'zlk', None)
    if zlk is not None:
        lines += [_ei_line(zlk.ei), _attribution_line(zlk.attribution, getattr(analysis, 'gate', None))]
    lines.append(_centre_line(getattr(analysis, 'centre_line', None)))
    return [line for line in lines if line]


def explain_long_period(analysis, settings, episodes: Sequence[Mapping[str, Any]]) -> VerdictText:
    """The explanation of a secular or Lidov-Kozai verdict (`LongPeriodAnalysis`); `episodes` = its episodes.csv rows."""
    v = analysis.verdict
    span = (
        float(analysis.series.t_yr[-1] - analysis.series.t_yr[0])
        if analysis.series is not None and len(analysis.series.t_yr) > 1
        else float('nan')
    )
    eps = Episodes(episodes, span, analysis.decisive_episode_id)
    whole = str(v.angle_status) == 'libration'
    checks = []
    if str(v.confidence):
        margins = [_long_period_margin_row(key, margin, settings, whole) for key, margin in v.margins.items()]
        checks = confidence_rows(v, settings, margins, whole)
    path = {entry.name: entry for entry in v.decision_path}
    steps = []
    count = path.get('librating_episodes')
    shape = (
        f'{_plural(int(count.value), "librating stretch", "librating stretches")}'
        if count is not None and finite(count.value)
        else 'the angle'
    )
    if whole:
        shape = 'the angle librates over the whole record'
    steps.append(_shape(f'{shape} → {v.angle_status}'))
    gate = getattr(analysis, 'gate', None)
    if gate is not None and _text(gate.outcome):
        steps.append(Step('gate', 'free-ω gate', StepState.LEAF, f'{gate.outcome} → {v.status}'))
    else:
        steps.append(Step('leaf', 'secular leaf', StepState.LEAF, f'{v.angle_status} → {v.status} (the shape of the angle decides)'))
    info = []
    value = str(v.checks.get('check_secular_e_response', ''))
    if value in ('pass', 'fail'):
        info.append(f'{_response_words(analysis)}: {value} (a column only)')
    info += zlk_column_lines(analysis)
    head = [_long_period_angle_sentence(analysis, eps, settings), _long_period_status_sentence(analysis, settings, episodes)]
    subtitle = f'reason {v.reason}' + (f' · origin {v.libration_origin}' if str(v.libration_origin) else '')
    return _verdict_text(v, subtitle, head, steps, checks, info)
