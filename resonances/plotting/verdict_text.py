"""The explanation of a verdict in words: the predicates and clause functions that the right column of the one-page
figures reads (`verdict_column_*` builders), and the `VerdictText` of an MMR verdict (`explain_mmr`), of which the
MMR builder reads the steps, checks, physics and grey columns.

Pure functions: an analysis (MMR, secular or Lidov-Kozai), its CSV rows and the settings in, words out; nothing is
drawn here and nothing feeds a status. Every number in the text is read from the analysis' own objects or CSV columns
and every threshold from `AnalysisSettings`, so a clause can be checked against the record
(`tests/resonances/plotting/test_verdict_text.py`).

What a `VerdictText` holds:

- `steps` (cards g10-4/g10-4b, W28 E3): the steps of the status tree in their fixed order, each a question
  '<number> · <question> [ids]' with 'Why:' and what it found: step 0 the angle shape with its G rules, the branch
  letter on the leaf ('5d'), decided here, passed on, not run (a gap, with its reason), not needed (the leaf decides
  without it: the owner at a kinematic leaf, 'n/a: no force on a'), n/a (the owner search is not applied to a 1:1);
- `checks` (cards g10-4/g10-4b, g03-5, W28 E4): every margin and cap the confidence read, each 'name [id]' with what
  was measured; a kind (`KIND_*`): the limits first (a cap, with the failed cap-class margin it explains printed
  inside it once, `merge_limits`), then the support margins: fail, should have been measured but was not, near its
  threshold, skipped by design (with the reason), full, n/a; K1 without a value says why (kin_check_reason);
- `grey_columns`: numbers of a test skipped by design, with why;
- `physics` (W20 CS11): A 'Is σ the own pendulum of <key>?' and B 'What makes the cycles?' (`physics_sections`),
  every number a column (`PhysicsRow.numbers`).

The W20 ledger entries are printed in words (`LEDGER_WORDS`). The words (caps, reasons, flags, ledger entries, the
short names of the rule ids) live in `verdict_vocab.py`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from resonances.plotting.verdict_vocab import (
    snake_words,
    CAP_MARGINS,
    CAP_PREFIX,
    CAP_WORDS,
    CHECK_WORDS,
    EI_WORDS,
    KIN_REASON_WORDS,
    LEDGER_WORDS,
    LONG_PERIOD_MARGIN_NAMES,
    MMR_MARGIN_NAMES,
    OWNER_FLAG_WORDS,
    OWNER_REASON_WORDS,
    POTENTIAL_REASON_WORDS,
    REGIME_PHRASES,
    REJECT_WORDS,
    SHORT_PERIOD_TITLES,
    STRENGTH_REASON_WORDS,
    THEORY_WORDS,
    long_period_id,
    rule_name,
)
from resonances.resonance.analysis.confidence import CAP_CEILING, Cap
from resonances.resonance.analysis.geometry import LEG_EDGE_START, RejectReason
from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.mmr.analyse import CENTRE_SCALE_UNION, SHORT_PERIOD_CAP, SHORT_PERIOD_HARMONIC
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
    LEG_EXTEND,
    LEG_SCATTER,
    LEG_TRANSIENT,
    LEG_UNCERTAIN,
    NULL_TEST,
    O1_ONSET,
    SCALE_OUT_OF_RANGE,
    SHORT_PERIOD_A,
    STRENGTH_DIRECT,
    STRENGTH_DIVISORS,
    STRENGTH_FIRST,
    STRENGTH_THREE_BODY,
    UNION_PROMOTED,
    deciding_half_libration,
    deciding_leg,
    edge_arc_name,
    edge_monotone_name,
    force_laws,
    kinematic_f_applies,
    period_floor,
    strength_upper_name,
    taken_direct_law,
    taken_harmonic_law,
)
from resonances.resonance.analysis.mmr.models import PLANET_NAMES_ORDER
from resonances.resonance.analysis.mmr.quasi_satellite import QS_FLOOR, QS_VISIT
from resonances.resonance.analysis.orbit_type import centre_moves
from resonances.resonance.analysis.pendulum import CycleClass, has_zone_cycle
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.secular.decide import librating_episodes as secular_librating_episodes
from resonances.resonance.analysis.secular.models import ARCSEC_PER_TURN
from resonances.resonance.analysis.zlk.episode_ei import phase_off_deg
from resonances.secular import phase_uncertainty
from resonances.resonance.analysis.verdict import CONFIDENCE_ORDER, POLICY_MINIMAL, Flag, MarginClass


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
    NOT_RUN = 'not_run'  # not triggered or not possible: a gap the reader should know of (the reason is printed)
    NOT_NEEDED = 'not_needed'  # did not run, and the leaf decides without it (`step_not_needed`)
    NOT_APPLICABLE = 'not_applicable'  # does not apply to this kind of record (a 1:1 owner, a drift step without drift)
    NOT_REACHED = 'not_reached'  # after the leaf (collapsed into one line, `collapse_after_leaf`)


ICONS = {
    CheckState.PASS: '✓',
    CheckState.THIN: '⚠',
    CheckState.FAIL: '✗',
    CheckState.SKIPPED: '–',
    CheckState.MISSING: '?',
    CheckState.NOT_APPLICABLE: '·',
}
CAP_ICON = '▼'
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
    """One step of the status tree as this verdict went through it (cards g10-4/g10-4b, extra_ru §2/§5): printed as
    '<number> · <question> [ids]', then 'Why: …' and the answer (`outcome`). `title` is the step's short label ('2.
    owner'); `number` is the step's number, with the branch letter on the leaf ('5d', '3k2'); `ids` are the rule and
    check ids the answer reads (`verdict_vocab.RULE_NAMES`)."""

    key: str
    title: str
    state: StepState
    outcome: str = ''
    number: str = ''
    question: str = ''
    why: str = ''
    ids: Tuple[str, ...] = ()

    @property
    def heading(self) -> str:
        """'3 · Does a swing with σ? [B1]', or the short label without a question; '' parts left out."""
        text = self.question or self.title
        head = f'{self.number} · {text}' if self.number else text
        return f'{head} [{", ".join(self.ids)}]' if self.ids else head


# The emphasis of a number (card g10-5, W28 E10): the block 'Numbers behind this verdict' over the numbers table prints
# these lines framed and tagged; each emphasised column leaves the table for a pointer '↑ decisive [B1]' (W20 CS11: no
# number twice). decisive: the comparison of the leaf that set the status; limit: the cap that sets the shown level;
# conflict: a measured check that fails and that the verdict did not read; unmeasured: a decisive test without a value.
EMPHASIS_DECISIVE, EMPHASIS_LIMIT, EMPHASIS_CONFLICT, EMPHASIS_UNMEASURED = 'decisive', 'limit', 'conflict', 'unmeasured'
EMPHASES = (EMPHASIS_DECISIVE, EMPHASIS_LIMIT, EMPHASIS_CONFLICT, EMPHASIS_UNMEASURED)  # also the block's order


@dataclass(frozen=True)
class EmphasisPart:
    """One number of an emphasised line: its CSV column and the comparison as it came out ('0.0954 < 0.1');
    `labeled`: the figure prints the column's label before it (False when `text` names the quantity itself)."""

    column: str
    text: str
    labelled: bool = True


@dataclass(frozen=True)
class Emphasis:
    """One line of 'Numbers behind this verdict' (`number_emphasis`): its kind (`EMPHASES`), the id and short name of
    the rule or cap ('B1', 'a carries the swing'), its numbers, and a closing note ('→ MEDIUM', 'not measured')."""

    kind: str
    rule_id: str
    name: str
    parts: Tuple[EmphasisPart, ...] = ()
    note: str = ''

    @property
    def tag(self) -> str:
        return self.kind.upper()

    @property
    def columns(self) -> Tuple[str, ...]:
        return tuple(part.column for part in self.parts if part.column)


def _shape(text: str, state: StepState = StepState.PASSED) -> Step:
    """The step 'angle shape' with its outcome (its number and G rules are set by `mmr_steps`)."""
    return Step('shape', 'angle shape', state, text)


# What a row of the confidence block is (cards g10-4/g10-4b, g03-5; from `MarginClass`): a support margin, counted in
# the raw level (full, thin, not measured, failed); a limit, which caps the level instead of being counted (a cap, or a
# failed cap-class margin, printed once, inside its cap's LIMIT row: `merge_limits`); n/a, not applicable here
KIND_SUPPORT, KIND_LIMIT, KIND_NA = 'support', 'limit', 'na'


@dataclass
class CheckRow:
    """One row of the confidence block: a margin (or one part of it), a near-threshold ledger entry, or a cap.
    Printed as 'name [id]' (`label`) and the measurement."""

    key: str
    name: str
    state: CheckState
    detail: str = ''  # the comparison as it came out, or the reason it was not measured
    need: str = ''  # what a full margin needs
    counts: str = ''  # how the confidence counted it: full | thin | not measured | failed | n/a | 'cap ≤ medium'
    note: str = ''  # why it is thin or not measured (printed in the column)
    short: str = ''  # the reason in a few words, for the confidence sentence (default: note, else detail)
    kind: str = KIND_SUPPORT  # support | limit | na (`KIND_*`)
    rule_id: str = ''  # the printed id when it is not the key ('SZ.N'; a cap with its margin merged: the margin's id)
    short_name: str = ''  # a cap: its short name (`verdict_vocab.CAP_NAMES`); `name` then holds what the cap means
    merged: str = ''  # a cap: the id of the failed cap-class margin printed inside it (`merge_limits`)

    @property
    def is_cap(self) -> bool:
        return self.key.startswith('cap:')

    @property
    def is_limit(self) -> bool:
        return self.kind == KIND_LIMIT

    @property
    def printed_id(self) -> str:
        """The id in brackets: `rule_id`, else the key ('U' for a ledger entry 'U:<name>')."""
        if self.rule_id:
            return self.rule_id
        return 'U' if self.key.startswith('U:') else self.key

    @property
    def label(self) -> str:
        """'name [id]' (card g10-3): 'away from the separatrix [L2]', 'earlier a-wave [O1]'."""
        return f'{self.short_name or self.name} [{self.printed_id}]'

    @property
    def icon(self) -> str:
        return CAP_ICON if self.is_limit or self.is_cap else ICONS[self.state]


@dataclass
class VerdictText:
    """What the MMR right column (`verdict_column_mmr.mmr_column`) reads of one verdict: the decision steps, the
    confidence rows, the pendulum sections (A, B) and the columns printed grey with why."""

    steps: List[Step]
    checks: List[CheckRow]
    grey_columns: Dict[str, str] = field(default_factory=dict)
    physics: List['PhysicsSection'] = field(default_factory=list)  # MMR: own pendulum (A) and what makes the cycles (B)


# --------------------------------------------------------------------------------------------- numbers


def finite(value: Any) -> bool:
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


SCI_FROM = 1e6  # a number of this size or larger is printed as '9.63e35' (2897: a 36-digit ratio in full, W34)


def fmt(value: Any, digits: int = DIGITS) -> str:
    """A number for a sentence: thousands grouped by a thin space, '9.63e35' from `SCI_FROM` on (`digits` significant),
    '≈0' below 1e-5, '—' when missing."""
    if not finite(value):
        return '—'
    value = float(value)
    if value == 0:
        return '0'
    size = abs(value)
    if abs(round(value)) >= SCI_FROM:  # as the grouped form would round it: 999 999.7 is '1e6', not '1 000 000'
        return _exp_form(value, digits)
    if size >= 1000:
        return f'{value:,.0f}'.replace(',', ' ')
    if size >= 100:
        return f'{value:.0f}'
    if size < 1e-5:
        return '≈0'
    if size < 1e-3:
        return f'{value:.2g}'
    return f'{value:.{digits}g}'


def _exp_form(value: float, digits: int) -> str:
    """'9.63e35', '2.56e-4': `digits` significant digits and a bare exponent."""
    mantissa, exponent = f'{value:.{digits - 1}e}'.split('e')
    return f'{float(mantissa):.{digits}g}e{int(exponent)}'


def _time_scale(value: Any) -> Tuple[float, str]:
    """(scale, unit) of a time [yr], chosen on its value rounded to 3 significant digits (999.7 yr is in kyr)."""
    rounded = abs(float(f'{float(value):.3g}'))
    return next(((s, u) for s, u in ((MYR, 'Myr'), (KYR, 'kyr')) if rounded >= s), (YEAR, 'yr'))


def years(value: Any) -> str:
    """A time in yr, kyr or Myr, 3 significant digits; the unit is chosen on the rounded value (999.95 yr -> '1 kyr',
    999 950 yr -> '1 Myr', never '1e+03 yr' or '1000 kyr')."""
    if not finite(value):
        return '—'
    value = float(value)
    scale, unit = _time_scale(value)
    return f'{value:.3g} yr' if scale == YEAR else f'{fmt(value / scale)} {unit}'


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


def regime_phrase(regime: Any) -> str:
    """'σ circulates', 'the record is too short to tell what σ does': a regime (`AngleStatus` value) in plain words."""
    text = _text(regime)
    return REGIME_PHRASES.get(text, f'σ is in the regime {snake_words(text)}')


def status_head(status: Any) -> str:
    """The status opening its sentence: 'Non-resonant', 'Near-separatrix', 'Resonant'."""
    return str(status).replace('_', '-').capitalize()


def plural(n: int, word: str, form: str = '') -> str:
    """'1 episode', '3 episodes'; `form` the plural when it is not `word` + 's'."""
    return f'{n} {word if n == 1 else (form or word + "s")}'


# ------------------------------------------------------------------------------- the minimal policy (W29 SPEC §2.3, §7)


def policy_minimal(verdict) -> bool:
    """The verdict was set by the minimal policy layer (its `policy` field; '' on a verdict the policy did not write: an
    inner pass, a verdict built by hand)."""
    return str(getattr(verdict, 'policy', '') or '') == POLICY_MINIMAL


def tree_status(verdict) -> str:
    """The status the status tree's leaf gave: under the minimal policy the v2 verdict's (`v2_verdict`, first field),
    which the policy may move; the verdict's own status otherwise. The steps describe the tree, so their leaf outcomes
    print this one, and `decided_at` adds the policy's move (2024 JR16: 'force law fails → resonant' read as if a
    failed law made it resonant)."""
    first = str(getattr(verdict, 'v2_verdict', '') or '').split('/')[0] if policy_minimal(verdict) else ''
    return first or str(verdict.status)


def policy_ids(text: Any) -> frozenset:
    """The episode ids of a `;`-joined `counted_episode_ids`/`candidate_episode_ids` cell; 'leg' and 'half' (a 1:1 leg,
    a half-swing: no episode row) are left out."""
    out = set()
    for part in str(text or '').split(';'):
        try:
            value = float(part)
        except ValueError:
            continue
        if math.isfinite(value):
            out.add(int(value))
    return frozenset(out)


def has_mark(verdict, name: str) -> bool:
    """`name` among the verdict's caps or flags: under the minimal policy every v2 cap is a flag (SPEC §2.3), so the
    prose that a cap used to trigger (short_period_aliased, the separatrix zone) reads either."""
    return name in (getattr(verdict, 'caps', ()) or ()) or name in (getattr(verdict, 'flags', ()) or ())


def and_list(items: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = list(items)
    return ', '.join(items[:-1]) + f' and {items[-1]}' if len(items) > 1 else ''.join(items)


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
        words = CAP_WORDS.get(cap, snake_words(cap))
        if cap == 'three_tp_episode' and whole_record:
            words = 'the libration over the whole record has three turning points only'
        short_name = rule_name(f'{CAP_PREFIX}{cap}')
        rows.append(
            CheckRow(
                f'cap:{cap}',
                words,
                CheckState.FAIL,
                counts=f'cap ≤ {cap_ceiling(cap)}',
                kind=KIND_LIMIT,
                short_name=short_name,
            )
        )
    return rows


def margin_kind(margin) -> str:
    """support | limit | na of a margin (`KIND_*`): a failed cap-class margin limits, a not-applicable one is n/a."""
    if margin.margin == Margin.NOT_APPLICABLE:
        return KIND_NA
    if margin.margin == Margin.FAIL and margin.margin_class == MarginClass.CAP:
        return KIND_LIMIT
    return KIND_SUPPORT


def merge_limits(rows: Sequence[CheckRow]) -> List[CheckRow]:
    """A failed cap-class margin is printed once, inside the LIMIT row of the cap it explains (`CAP_MARGINS`, card
    g03-5): its measurement and need move to the cap row, which then carries the margin's id ('earlier a-wave [O1]'); a
    failed cap-class margin without its cap stays a LIMIT row of its own. Every other row is kept as it is."""
    caps = {row.key[len(CAP_PREFIX) :]: row for row in rows if row.is_cap}
    by_margin = {margin: caps[cap] for cap, margin in CAP_MARGINS.items() if cap in caps}
    kept = []
    for row in rows:
        target = by_margin.get(row.key) if row.is_limit and not row.is_cap else None
        if target is None:
            kept.append(row)
            continue
        target.merged = target.rule_id = row.key
        target.detail = '; '.join(part for part in (row.detail, target.detail) if part)
        target.need = row.need
        target.short = target.short or short_reason(row.detail)
    return kept


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
    """Reading order (card g10-4b): the limits first (the cap that sets the level first, `cap_rows`), then the support
    margins worst first (failed, missing, thin, full, skipped by design), the n/a ones last."""
    return sorted(rows, key=lambda row: (not row.is_limit, row.kind == KIND_NA, STATE_ORDER.index(row.state)))


def _listed(items: Sequence[str], limit: int = LIST_MAX) -> str:
    """'a, b, c and 2 more': the first `limit` items, then how many are left out."""
    hidden = len(items) - limit
    return ', '.join(items[:limit]) + (f' and {hidden} more' if hidden > 0 else '')


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
    def counted(self) -> List[Mapping[str, Any]]:
        """The accepted runs without those the status does not count and no other mark names (`uncounted_rows`: below
        the period floor, the center turns; W28 C11, card g02-9 A)."""
        uncounted = [id(row) for row in uncounted_rows(self.rows)]
        return [row for row in self.accepted if id(row) not in uncounted]

    @property
    def longest(self) -> Optional[Mapping[str, Any]]:
        return max(self.counted, key=_duration) if self.counted else None

    def covered_fraction(self) -> float:
        if not finite(self.span_yr) or self.span_yr <= 0:
            return float('nan')
        return sum(_duration(row) for row in self.counted) / self.span_yr


def coverage_text(row: Mapping[str, Any], settings) -> str:
    """'it misses the start by 8.6 and the end by 4.9 libration periods (a libration that covers the record may miss
    each end by up to 1 period)'; when both gaps are within that bar (`coverage_within`) it says so, and the episodic word
    then comes from something else (an exit seen: 24820)."""
    start, end = row.get('coverage_start_gap_P'), row.get('coverage_end_gap_P')
    if not (finite(start) and finite(end)):
        return ''
    limit = settings.libration_coverage_periods
    head = f'it misses the start by {fmt(start, 2)} and the end by {fmt(end, 2)} libration periods'
    period = 'period' if fmt(limit) == '1' else 'periods'
    if coverage_within(row, settings):
        return f'{head}, both within the {fmt(limit)} {period} a libration that covers the record may miss at each end'
    return f'{head} (a libration that covers the record may miss each end by up to {fmt(limit)} {period})'


def coverage_within(row: Mapping[str, Any], settings) -> bool:
    """Both edge gaps of an episode within LIBRATION_COVERAGE_PERIODS (`geometry.covers_edges`' first reading)."""
    start, end = row.get('coverage_start_gap_P'), row.get('coverage_end_gap_P')
    limit = settings.libration_coverage_periods
    return finite(start) and finite(end) and float(start) <= limit and float(end) <= limit


# --------------------------------------------------------------------------------------------- MMR


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
        # the angle decision's own rule (G5a, G5w, G6, ...: `AngleDecision.rule`; column geo_angle_rule)
        self.angle_rule = '' if _text(row.get('geo_angle_rule')) in ('', 'nan') else _text(row.get('geo_angle_rule'))
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
        self.brief = False  # the slotted text (`mmr_bullets`): a reason step 5 prints in full is a pointer there

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
        listed = and_list(changed)
        verb = 'changes' if len(changed) == 1 else 'change'
        unused[margin_id] = f'withdrawn by the filter guard: at twice the filter cutoff the {listed} {verb}'
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
    noun = 'cycle' if shown == '1' else 'cycles'
    return (
        f'{stretch} lasts {years(length)} = {shown} {noun} < {against} '
        f'(in cycles of the mean cycle period P_cyc {years(period)}{own_text}); the test needs {s.mmr_force_min_windows} windows'
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
        f'{plural(n, "episode")} before the jump {"is" if n == 1 else "are"} not measured'
    )


def measured_on(f: MMRFacts, source_field: str = 'force_source') -> str:
    """Where rule R took a check from (`response.deciding_response`, the response's `*_source`): 'measured on
    episode #4 (14.2–15.2 kyr, of 2 eligible episodes the one with most turning points)', 'measured on the libration
    (0.08–19.9 kyr)' when it is the only one, 'pooled over the 3 episodes
    before the jump (1.2 kyr–9.8 kyr)', or '' for the record's pooled measurement."""
    from resonances.resonance.analysis.mmr.response import witnesses

    response = f.a.response
    source = str(getattr(response, source_field, '')) if response is not None else ''
    if source == 'witness':
        found = witnesses(response)
        witness = next((e for e in found if e.episode_id == response.witness_episode_id), None)
        if witness is None:
            return ''
        # rule R's witness is an episode (review 2026-11-05 #109: no 'witness'); 'episode' only when there are several
        span = interval(witness.t_start_yr, witness.t_end_yr)
        if len(found) == 1 and len(f.episodes.accepted) <= 1:
            return f'measured on the libration ({span})'
        others = f', of {len(found)} eligible episodes the one with most turning points' if len(found) > 1 else ''
        return f'measured on episode #{witness.episode_id} ({span}{others})'
    pooled = getattr(response, 'before_jump_response', None)
    if source == 'before_jump' and pooled is not None:
        n = sum(1 for e in response.episode_responses if e.before_jump)
        return f'pooled over the {plural(n, "episode")} before the jump ({years(pooled.t_start_yr)}–{years(pooled.t_end_yr)})'
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
                f'{plural(len(pooled.windows), "force window")} of {fmt(s.mmr_force_window_cycles)} cycles '
                f'(P {years(pooled.period_yr)}) < {s.mmr_force_min_windows}'
            )
        return CheckState.SKIPPED, force_windows_text(f)
    if not finite(response.phi0_tol_rad):
        return CheckState.SKIPPED, 'the modulation can turn the phase of the force by π (Σε ≥ 1), so its stability has no bound'
    return CheckState.MISSING, 'no force-law number'


def filter_too_slow_text(f: MMRFacts) -> str:
    """'the filter is too slow for the period (P_cyc 94.2 yr at P_c 55.5 yr, gain 0.892 < 0.911)' when the libration
    is filter_too_slow (2026 AC4, W28 Codex S9: its 1:1 energy test was printed 'not measured'), else ''."""
    from resonances.resonance.analysis.mmr.preprocess import gain_too_slow

    prep = getattr(f.a, 'prep', None)
    if prep is None or not prep.filter_too_slow:
        return ''
    gain = compare(prep.gain_at_period_cycle, '>=', gain_too_slow(f.s))
    period, cutoff = f.row.get('geo_period_cycle_yr'), f.row.get('flt_cutoff_yr')
    at = f'P_cyc {years(period)} at P_c {years(cutoff)}, ' if finite(period) and finite(cutoff) else ''
    return f'the filter is too slow for the period ({at}gain {gain})'


def _skip_potential(f: MMRFacts) -> Tuple[CheckState, str]:
    s, energy, geometry = f.s, f.a.energy, f.a.geometry
    amp = f.amplitude_max_deg()
    if energy is None:
        return CheckState.MISSING, 'no energy analysis'
    slow = filter_too_slow_text(f)
    if slow and getattr(geometry, 'coorbital', False):  # `response.with_coorbital_potential`: R0 reads the shape
        return CheckState.MISSING, f'{slow}; the 1:1 energy integral reads the shape the filter damps, so it is not used'
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
    """Why no subresonance scan ran (no trigger T1, T2, T5, T6 fired): 'the center is stationary', or, when the center
    moves by the header's test (`decisive_centre_moves`), what the trigger T1 read instead: the bands overlap."""
    if decisive_centre_moves(f):
        return 'not triggered: the cycles’ bands overlap (no trigger fired)'
    return 'not triggered: the center is stationary'


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
_OWNER_FLAG_STATES = {
    'owner_no_longitudes': CheckState.MISSING,
    'owner_too_few_windows': CheckState.SKIPPED,
    'owner_no_period': CheckState.SKIPPED,
    'owner_unwrap_failed': CheckState.MISSING,
    'owner_search_off': CheckState.SKIPPED,
    'owner_retrograde': CheckState.SKIPPED,
    'owner_catalogue_ambiguous': CheckState.MISSING,
}
OWNER_FLAG_REASONS = {flag: (state, OWNER_FLAG_WORDS[flag]) for flag, state in _OWNER_FLAG_STATES.items()}


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
        return CheckState.SKIPPED, snake_words(strength.not_measured_reason)
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


def _kinematic_reason(f: MMRFacts, field_name: str) -> Tuple[CheckState, str]:
    """K1 or K1e without a value: why the correlation has none (`kinematics.check_reason`, kin_check_reason, for K1;
    kin_k1_episode_reason for K1e, card g10-4b) and where the figure shows σ against p·ϖ (panel (j)); 'not measured'
    when the record keeps no reason."""
    kinematics = f.a.kinematics
    code = _text(getattr(kinematics, field_name) if kinematics is not None else f.row.get(f'kin_{field_name}'))
    if not code:
        return CheckState.MISSING, 'not measured'
    words = KIN_REASON_WORDS.get(code, snake_words(code)).format(min_points=f.s.mmr_kin_min_points)
    return CheckState.MISSING, f'no value — {words}; compare σ and p·ϖ in (j)'


def _skip_l2(f: MMRFacts) -> Tuple[CheckState, str]:
    """A not_measured L2. On a 1:1 angle under the minimal policy the level's M does not read L2 but the 1:1 zone
    (`policy._mmr_margin`: the zone known and no cycle in it → clear; 2002 AA29, 164207): the row says so."""
    energy = f.a.energy
    if getattr(f.a.geometry, 'coorbital', False) and policy_minimal(f.v):
        if energy is None or energy.zone is None:
            return CheckState.MISSING, 'not measured; the level’s amplitude margin (M) reads the 1:1 zone, which is not measured either'
        n_zone, n_above, n_all = _zone_counts(energy)
        above = f'{plural(n_above, "cycle")} above a lower saddle, which this test cannot grade' if n_above else 'not graded on a 1:1'
        clear = 'near' if n_zone else 'clear of'
        return (
            CheckState.SKIPPED,
            f'{above}; the level reads the 1:1 zone instead: {n_zone} of {n_all} cycles in it → {clear} the separatrix',
        )
    return CheckState.MISSING, 'not measured'


SKIP_REASONS = {
    'L2': _skip_l2,
    'C4': _skip_c4,
    'K1': lambda f: _kinematic_reason(f, 'check_reason'),
    'K1e': lambda f: _kinematic_reason(f, 'k1_episode_reason'),
    'L4': _skip_force_law,
    'R0': _skip_potential,
    'S': _skip_subresonances,
    'W': _skip_owner,
    'F': _skip_strength,
    'L6': 'one cycle gives one center: nothing to compare',
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
            _with_source(f'{halves}, K/K_theory {fmt(law.scale)} {"within" if inside else "outside"} [{fmt(1 / limit)}, {fmt(limit)}]', f),
            f'R² ≥ {fmt(s.mmr_force_r2_full)} in both halves; K/K_theory in [{fmt(1 / limit)}, {fmt(limit)}]',
        )
    direct = taken_direct_law(response)
    if direct is not None:  # B8 (card g07-2 A): the term's averaged force decided L4 where the harmonic law did not
        return _with_source(direct_force_text(f, direct, halves), f), direct_force_need(f)
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


DIRECT_FORCE_NAME = 'ȧ follows the averaged force of the term'  # L4 decided by the direct law (B8, `taken_direct_law`)


def _strength_band(f: MMRFacts) -> Tuple[float, float]:
    """[1/√x, √x], x = MMR_STRENGTH_CHECK_RATIO: the band of the direct law's scale s (`response.judge_direct`)."""
    band = math.sqrt(f.s.mmr_strength_check_ratio)
    return 1 / band, band


def direct_force_text(f: MMRFacts, law, halves: str) -> str:
    """The direct averaged-force law that decided L4 (W28 B8, card g07-2 A; `models.DirectForceLaw`): the term's force
    with the planet's real mass and ȧ, both low-passed at f P_large, one scale s per window: 'averaged force of
    8N-13+0+5 (Neptune's real mass) at 3.07 kyr = 0.35 P 8.78 kyr (the large turns): R²
    0.755 ≥ 0.4 (full), s 1.14–1.29 within [0.32, 3.16], above the shifted null by 0.737 ≥ 0.2, 4 windows; harmonic
    law R² 0.163/0.168'."""
    s = f.s
    spec = f.a.resonance
    planet = spec.planets[0] if spec is not None and spec.planets else 'the planet'
    lo, hi = _strength_band(f)
    scales = (law.scale_min, law.scale_max)
    inside = all(finite(x) for x in scales) and lo <= law.scale_min and law.scale_max <= hi
    factor = s.mmr_direct_force_cutoff_cycles
    turns = ' (the large turns)' if law.survived else ''
    grade = 'full' if finite(law.r2) and law.r2 >= s.mmr_force_r2_full else 'thin'
    return (
        f'averaged force of {spec.key} ({planet}’s real mass) at {years(law.cutoff_yr)} = {fmt(factor)} P '
        f'{years(law.cutoff_yr / factor)}{turns}: R² {compare(law.r2, ">=", s.mmr_force_r2_fail)} ({grade}), '
        f's {fmt(law.scale_min)}–{fmt(law.scale_max)} {"within" if inside else "outside"} [{fmt(lo)}, {fmt(hi)}], '
        f'above the shifted null by {compare(law.null_gain, ">=", s.mmr_force_null_gain)}, {plural(int(law.n_windows), "window")}; '
        f'harmonic law {halves}'
    )


def direct_force_need(f: MMRFacts) -> str:
    """What the direct law needs (`response.judge_direct`'s guards)."""
    s = f.s
    lo, hi = _strength_band(f)
    return (
        f'R² ≥ {fmt(s.mmr_force_r2_fail)} (full from {fmt(s.mmr_force_r2_full)}) with only s free, s in [{fmt(lo, 2)}, '
        f'{fmt(hi)}], above the shifted null, on windows that share no sample'
    )


def coorbital_energy_scale_clause(f: MMRFacts) -> str:
    """Why the 1:1 energy integral failed when its R² passes (`coorbital.energy_check`: the scale K_E/K_theory out of
    the band [1/x, x], x = MMR_COORBITAL_SCALE_RATIO_MAX): 'its scale K_E/K_th −0.0457 lies outside [0.333, 3]'
    (2016 JA 5 kyr); '' otherwise (column rsp_coorb_energy_K_over_K_theory)."""
    law = getattr(f.a.energy, 'potential_law', None)
    if law is None or str(law.energy_check) != 'fail' or not finite(law.energy_scale):
        return ''
    limit = f.s.mmr_coorbital_scale_ratio_max
    if 1 / limit <= law.energy_scale <= limit:
        return ''
    return f'its scale K_E/K_th {fmt(law.energy_scale)} lies outside [{fmt(1 / limit)}, {fmt(limit)}]'


def tree_read_potential_fail(f: MMRFacts) -> bool:
    """R0 failed for the tree although the published check is not a failure: the filter guard empties a sensitive
    check after the tree, while a failed one stays an objection for it (`decide.energy_for_tree`, principle b; 2016 JA
    5 kyr: leaf 5a, check_mmr_potential not_measured)."""
    energy = f.a.energy
    guard = f.unused.get('R0', '').startswith('withdrawn by the filter guard')
    return guard and energy is not None and str(energy.with_slow) == 'fail' and f.checks.get('potential') != 'fail'


def _potential_detail(f: MMRFacts) -> Tuple[str, str]:
    energy, s = f.a.energy, f.s
    law = getattr(energy, 'potential_law', None)  # R0 printed from the law it was read from (W20 round D)
    if law is not None:
        r2, scale = law.energy_r2, coorbital_energy_scale_clause(f)
        text = f'1:1 energy integral R² {compare(r2, ">=", s.mmr_force_r2_fail)}' + (f', but {scale}' if scale else '')
        return text, f'R² ≥ {fmt(s.mmr_force_r2_fail)}'
    if energy is None:
        return '', ''
    need = f'≥ {percent(s.separatrix_vevf_fraction)} of the windows with r ≥ {fmt(s.separatrix_vevf_r_min)}'
    if energy.r0_episode_id is not None:  # W34 `zone.episode_r0`: the pooled potential measured nothing
        row = next((r for r in f.episodes.rows if _same_id(r.get('episode_id'), energy.r0_episode_id)), {})
        span = f' {interval(row.get("t_start_yr"), row.get("t_end_yr"))}' if row else ''
        return (
            f'the period law and V_E ≈ V_F hold inside episode #{energy.r0_episode_id}{span} (the record\'s one potential '
            'mixes regimes and measures nothing)',
            need,
        )
    eligible = window_count_text(f)
    if eligible:  # B3 (cards g07-1 A, g08-1 A): counted over the eligible windows, compared on the visited arc
        need += f' (a window counts with a half-swing ≥ {fmt(s.mmr_a_response_min_amp_deg)}° or a revolution)'
        return f'V_E ≈ V_F in {eligible}', need
    return f'V_E ≈ V_F in {compare_percent(energy.ve_vf_agree_fraction, ">=", s.separatrix_vevf_fraction)} of the windows', need


def window_count_text(f: MMRFacts) -> str:
    """R0 with a over its eligible windows (W28 B3, `zone.window_eligible`; columns sep_VE_VF_n_eligible,
    sep_VE_VF_n_windows, sep_VE_VF_arc_deg): '18 of 22 eligible windows (82% ≥ 67%); 38 of 60 windows swing less and
    are not counted; compared on the visited arc of 165°'. '' when the counts are not in the record."""
    s, energy = f.s, f.a.energy
    n_eligible, n_windows = f.row.get('sep_VE_VF_n_eligible'), f.row.get('sep_VE_VF_n_windows')
    fraction = energy.ve_vf_agree_fraction if energy is not None else None
    if not (finite(n_eligible) and finite(n_windows) and n_eligible > 0 and finite(fraction)):
        return ''
    eligible, total = int(n_eligible), int(n_windows)
    agree = int(round(float(fraction) * eligible))
    text = f'{agree} of {eligible} eligible windows ({compare_percent(fraction, ">=", s.separatrix_vevf_fraction)})'
    if total > eligible:
        text += f'; {total - eligible} of {total} windows swing less and are not counted'
    arc = f.row.get('sep_VE_VF_arc_deg')
    if s.separatrix_vevf_arc and finite(arc):
        text += f'; compared on the visited arc of {fmt(arc)}°'
    return text


def _best_candidate(candidates):
    return min(candidates, key=lambda c: c.rank, default=None)


def _owner_rejection(f: MMRFacts, candidate) -> str:
    """Why the best owner candidate was not accepted: its R² against the bar when that was the reason, else the
    test it failed with its number where the candidate keeps one (433 1R-5V+7: R² 0.73, phase incoherent)."""
    r2 = f'R² {fmt(candidate.r2_median, 2)}'
    reason = str(candidate.reason or '')
    if reason in ('r2_low', ''):  # owner.py rejects on mmr_owner_r2_min (W20 round D, review C #4)
        return f'R² {compare(candidate.r2_median, "<", f.s.mmr_owner_r2_min, digits=2)}'
    words = OWNER_REASON_WORDS.get(reason, snake_words(reason))
    return f'{r2}, but ' + words.format(coherence=fmt(getattr(candidate, 'phi0_coherence', None), 2))


def _member_rejection(f: MMRFacts, candidate) -> str:
    """Why the best member was not accepted, with the number of its criterion where the scan keeps one."""
    sub, s = f.a.subresonances, f.s
    own = getattr(sub, 'centre0_range_rad', float('nan'))
    if candidate.reason == 'range' and finite(candidate.centre_range_rad) and finite(own) and own > 0:
        return f'its center swings {fmt(candidate.centre_range_rad / own)}× as far as σ’s center'
    if candidate.reason == 'rate' and finite(candidate.centre_rate_rad_yr):
        return f'its center drifts at {fmt(abs(candidate.centre_rate_rad_yr) * KYR)} rad/kyr'
    if candidate.reason == 'pass' and finite(candidate.centre_max_pass_rad):
        return f'its center makes a pass of {fmt(degrees(candidate.centre_max_pass_rad))}°'
    if candidate.reason in ('A_coherence', 'A_gain') and finite(candidate.phi0_coherence):
        return f'its force phase coherence {compare(candidate.phi0_coherence, ">=", s.mmr_subresonance_phi0_coh_min)}'
    return REJECT_WORDS.get(candidate.reason, snake_words(candidate.reason))


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
            # the ratio is the member's centre range over σ's: at or below the bar the member is as still as σ or stiller
            note = f'a member’s center is as still as σ’s or stiller (range ratio {compare(margin.value, ">", s.mmr_s_close_ratio)})'
    if sub.accepted is not None:
        return (
            f'{sub.accepted.spec.key}: φ0 coherence {compare(sub.accepted.phi0_coherence, ">=", s.mmr_subresonance_phi0_coh_min)}',
            'the theory confirms the member',
            note,
        )
    best = _best_candidate(sub.candidates)
    value = f'{plural(len(sub.candidates), "member")} scanned, none holds the center'
    if best is not None:
        value += f'; best {best.spec.key}: {_member_rejection(f, best)}'
    return value, f'none accepted; closest center range > {fmt(s.mmr_s_close_ratio)}× σ’s', note


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


def refused_owner(f: MMRFacts) -> bool:
    """An accepted owner the decision did not take (F7 outside σ's episode, or P2): a neighbor, a candidate driver."""
    return bool({str(Flag.OWNER_OUTSIDE_EPISODE), 'owner_psi_too_fast'} & set(map(str, f.v.flags)))


def _w_detail(f: MMRFacts) -> Tuple[str, str]:
    owner, s = f.a.owner, f.s
    if owner is None:
        return '', ''
    if identity_route(owner):
        return _identity_claim(owner), f'R² ≥ {fmt(s.mmr_w_r2_full)} and reachable (not measured by this route)'
    if owner.accepted is not None:
        verb = 'candidate driver of a' if refused_owner(f) else 'drives a'
        return f'{owner.accepted.spec.key} {verb}: R² {fmt(owner.accepted.r2_median, 2)}', f'R² ≥ {fmt(s.mmr_w_r2_full)} and reachable'
    best = _best_candidate(owner.candidates)
    if best is None:
        return ('no candidate', '') if not owner.n_candidates else (f'{plural(owner.n_candidates, "candidate")}, none fits', '')
    need = f'best R² < {fmt(s.mmr_w_r2_none)} (thin from {fmt(s.mmr_w_r2_none)}; an owner needs ≥ {fmt(s.mmr_owner_r2_min)})'
    head = f'{plural(owner.n_candidates, "candidate")}; best {best.spec.key}'
    if not finite(best.r2_median) or best.r2_median < s.mmr_w_r2_none:
        return f'{head} R² {compare(best.r2_median, "<", s.mmr_w_r2_none, digits=2)}', need
    # the thin band (2013 AS27, 2897): it fits a, but no owner was accepted; why, as the owner step says it
    if best.r2_median < s.mmr_owner_r2_min:
        band = f'R² {fmt(best.r2_median, 2)} in the thin band {fmt(s.mmr_w_r2_none)}–{fmt(s.mmr_owner_r2_min)}'
    else:
        band = f'R² {compare(best.r2_median, ">=", s.mmr_owner_r2_min, digits=2)}'
    reason = str(best.reason or '')
    words = OWNER_REASON_WORDS.get(reason, snake_words(reason)).format(coherence=fmt(getattr(best, 'phi0_coherence', None), 2))
    why = '' if reason in ('r2_low', '') else f', but {words}'
    return f'{head} {band}{why}: it fits a, not accepted as an owner (thin)', need


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
    return f'{plural(count, "turning point")} at the coarsest cutoff', f'≥ {s.mmr_l1_tp_full}', note


def _zone_counts(energy) -> Tuple[int, int, int]:
    """(cycles in the separatrix zone, cycles not measured (above a lower saddle), all cycles) of the energy's classes."""
    classes = [CycleClass(c) for c in energy.cycle_classes]
    return classes.count(CycleClass.ZONE), energy.n_not_measured, len(classes)


def _zone_grade(energy) -> Optional[Margin]:
    """L2 as `decide._l2` grades it from the cycle classes, with its predicates (`has_zone_cycle`, `n_not_measured`):
    thin with a cycle in the zone, not_measured with one above a lower saddle, else full; None without a zone."""
    if energy is None or energy.zone is None or not energy.cycle_classes:
        return None
    if has_zone_cycle(energy.cycle_classes):
        return Margin.THIN
    return Margin.NOT_MEASURED if energy.n_not_measured else Margin.FULL


def _detail_l2(f: MMRFacts, margin) -> Tuple[str, str, str]:
    """The cycles in the zone when the tree graded L2 by them; else the amplitude against MMR_L2_AMP_FULL_DEG, which
    is what graded it (2003 UG22 40 kyr: the tree read the episode before the jump, where no zone is measured, so
    138° > 120° made L2 thin while the record's zone holds none of its 181 cycles)."""
    energy, s = f.a.energy, f.s
    if margin.margin == Margin.NOT_MEASURED and _skip_l2(f)[0] == CheckState.SKIPPED:
        return '', 'no cycle in the zone', ''  # a 1:1 under the minimal policy: the reason carries the counts (_skip_l2)
    grade = _zone_grade(energy)
    amplitude = f'max half-amplitude {compare(margin.value, "<=", s.mmr_l2_amp_full_deg, "°")}'
    if grade is not None:
        n_zone, n_unmeasured, n_all = _zone_counts(energy)
        unmeasured = f', {n_unmeasured} not measured (above a lower saddle)' if n_unmeasured else ''
        cycles = f'{n_zone} of {n_all} cycles in the zone{unmeasured}'
        if grade != margin.margin and margin.value is not None and margin.margin in (Margin.FULL, Margin.THIN):
            detail = f'{amplitude} (graded without a zone: the record’s zone, {cycles}, is not this regime’s)'
            return detail, f'≤ {fmt(s.mmr_l2_amp_full_deg)}°', ''
        detail = f'{cycles} (max half-amplitude {fmt(margin.value)}°)'
        note = f'{n_zone} of {n_all} cycles in the separatrix zone' if n_zone else ''
        return detail, 'no cycle in the zone', note
    return amplitude, f'≤ {fmt(s.mmr_l2_amp_full_deg)}°', ''


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
    """L7 grades a libration that covers the record: full when the larger edge gap is <= MMR_L7_GAP_FULL periods; the
    record counts as covered up to LIBRATION_COVERAGE_PERIODS, so a gap between the two bars is a thin margin, not a
    miss (2001 HW7: 0.705 > 0.7, covered)."""
    s = f.s
    detail = f'larger edge gap {compare(margin.value, "<=", s.mmr_l7_gap_full)} P'
    if margin.margin != Margin.FULL and finite(margin.value):
        detail += f' for a full margin; the record counts as covered up to {fmt(s.libration_coverage_periods)} P'
    return detail, f'≤ {fmt(s.mmr_l7_gap_full)} P', ''


def _detail_o1(f: MMRFacts, margin) -> Tuple[str, str, str]:
    s, episode = f.s, f.episodes.decisive or {}
    if margin.margin == Margin.NOT_APPLICABLE and f.a.energy is not None and f.a.energy.episodes_in_separatrix_zone:
        if zone_reading_withdrawn(f):  # O1 was set n/a by the zone's reading, which the guard withdrew afterwards
            later = 'a reading the filter guard later withdrew' if _zone_withdrawn_why(f) else 'a reading later withdrawn'
            return f'n/a: set by the zone’s crossings, {later}', '', ''
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
        if ratio >= s.mmr_o1_ratio_fail:  # B4: the ratio alone no longer fails O1
            detail += f'; {foreign_wave_text(f, episode)}'
    need = f'< {fmt(s.mmr_o1_ratio_thin)}; fails at ≥ {fmt(s.mmr_o1_ratio_fail)} only for a foreign wave (coherent at P, in phase, not the own law)'
    return detail, need, ''


def foreign_wave_text(f: MMRFacts, episode: Mapping[str, Any]) -> str:
    """O1's three parts (W28 B4, cards g03-4 A, g08-3 A; `decide.foreign_wave`, episodes.csv onset_*): the background
    a-wave coherent at P, continuing into the episode in phase, and not explained by the episode's own law. 'coherent
    at P: R² 0.98 ≥ 0.5; phase gap 4° ≤ 90°; own law explains 0.87 ≥ 0.5 of the episode's R² → not a foreign wave'."""
    s = f.s
    coherence, gap = episode.get('onset_coherence_r2'), episode.get('onset_phase_gap_deg')
    own, own_episode = episode.get('onset_own_law_r2'), episode.get('onset_own_law_r2_episode')
    share = float(own) / float(own_episode) if finite(own) and finite(own_episode) and float(own_episode) > 0 else None
    coherent = holds(coherence, '>=', s.mmr_o1_coherence_r2_min)
    in_phase = holds(gap, '<=', s.mmr_o1_phase_gap_max_deg)
    explained = holds(share, '>=', s.mmr_o1_own_law_r2_fraction)
    parts = [
        f'coherent at P: R² {compare(coherence, ">=", s.mmr_o1_coherence_r2_min)}' if finite(coherence) else 'coherence at P not measured',
        f'phase gap {compare(gap, "<=", s.mmr_o1_phase_gap_max_deg, "°")}' if finite(gap) else 'phase gap not measured',
        (
            f'own law explains {compare(share, ">=", s.mmr_o1_own_law_r2_fraction)} of the episode’s R²'
            if share is not None
            else 'own law not measured'
        ),
    ]
    foreign = bool(coherent) and bool(in_phase) and not bool(explained)
    return '; '.join(parts) + (' → a foreign wave' if foreign else ' → not a foreign wave')


def _detail_r0(f: MMRFacts, margin) -> Tuple[str, str, str]:
    return _potential_detail(f) + ('',)


# pendulum.ZoneMethod and the zone_* flags (mmr/zone.py) in words
ZONE_METHOD_WORDS = {
    'numeric': 'integrated in the modulated pendulum',
    'eq_only': 'from the modulation of e alone (no multiplet member resolved)',
    'adiabatic': 'adiabatic estimate',
}
ZONE_FLAG_WORDS = {
    'zone_eq_only': 'the modulation from e alone',
    'zone_external_unresolved': 'the record resolves no motion slower than the angle’s own period (a standing separatrix)',
    'zone_members_unresolved': 'the multiplet members fit in sample only (their modulation is not identified)',
    'zone_numeric_mismatch': 'the integrated and adiabatic estimates disagree',
    'zone_edge_cycle': 'a cycle lies at the zone’s edge',
}


def _detail_z(f: MMRFacts, margin) -> Tuple[str, str, str]:
    """The separatrix zone in cycle energy h (1 = the top of the fitted well) and how it was found; a thin Z says
    why it is only estimated (the zone_* flags)."""
    energy = f.a.energy
    detail = note = ''
    if energy is not None and energy.zone is not None:
        method = str(getattr(energy.zone, 'method', '')).split('.')[-1].lower()
        lo, hi = fmt(energy.zone.h_lo), fmt(energy.zone.h_hi)
        detail = f'h {lo if lo == hi else f"{lo}–{hi}"}, {ZONE_METHOD_WORDS.get(method, snake_words(method))}'
        if margin.margin == Margin.THIN:
            shown = [flag for flag in energy.flags if flag.startswith('zone_') and not (flag == 'zone_eq_only' and method == 'eq_only')]
            note = '; '.join(ZONE_FLAG_WORDS.get(flag, snake_words(flag.replace('zone_', ''))) for flag in shown)
    return detail, 'a numeric zone, no edge cycle', note


def _detail_w(f: MMRFacts, margin) -> Tuple[str, str, str]:
    if margin.margin == Margin.NOT_APPLICABLE:
        return COORBITAL_OWNER, '', ''
    return _w_detail(f) + ('',)


def _detail_f(f: MMRFacts, margin) -> Tuple[str, str, str]:
    s, strength = f.s, f.a.strength
    ratio = strength.ratio if strength is not None else None
    limit = s.mmr_strength_check_ratio
    detail = ''
    if finite(ratio):
        inside = 1 / limit <= ratio <= limit
        # direct averaging: the ratio is the end of the snapshots' range nearest 1 (Strength.ratio), not the range itself
        nearest = (
            ' (the snapshots’ value nearest 1)'
            if strength is not None and str(strength.method) in ('direct_averaging', 'direct_harmonic')
            else ''
        )
        detail = f'required/theory {fmt(ratio)}{nearest} {"within" if inside else "outside"} [{fmt(1 / limit)}, {fmt(limit)}]'
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
        return f'member {sub.accepted.spec.key} holds the center'
    if key == 'W' and owner is not None and owner.accepted is not None:
        approximate = ' (catalog λ, approx.)' if 'catalogue_longitudes' in owner.flags else ''
        if identity_route(owner):
            return f'owner {owner.accepted.spec.key} librates (identity route){approximate}'
        if refused_owner(f):  # g10-8: an owner the decision did not take is a candidate, never 'drives a'
            return f'candidate driver {owner.accepted.spec.key}{approximate}'
        role = ROLE_DRIVER if str(f.v.cause_role) == ROLE_DRIVER else ROLE_OWNER  # g04-5: a circulating driver
        return f'{role} {owner.accepted.spec.key} drives a{approximate}'
    if key == 'L4' and f.coorbital:
        return 'ȧ follows the 1:1 force law'
    if key == 'L4' and taken_direct_law(f.a.response) is not None:
        return DIRECT_FORCE_NAME
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
    row = CheckRow(key, _margin_name(f, key), margin_state(margin.margin), detail, need, counts, shown, note, kind=margin_kind(margin))
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
    if key == 'W':  # the search did not answer: no claim 'no other resonance drives a' (card g10-8)
        row.name = OTHER_NOT_TESTED
        if no_force_on_a(f):  # g04-7: not a gap at a kinematic leaf
            row.note = f'not needed: {NO_FORCE_ON_A} (step 3); not run: {why}'
    measured = key in f.unused and why == f.unused[key]  # a number that was measured, then not used
    # a number of a missing measurement says nothing
    row.detail = _unused_detail(row.detail, state != CheckState.MISSING or measured)
    return row


def _unused_detail(detail: str, keep: bool = True) -> str:
    """The number of a not_measured margin printed as not used ('(…: not used)'); '' for a dash or when not `keep`."""
    return f'({detail}: not used)' if keep and detail and '—' not in detail else ''


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
            name, detail = f'{ledger_words(entry.name)} near its convention', entry_compare(entry)
            same = next((row for row in rows if row.name == name), None)
            if same is not None:  # two entries of one family (79653: K2 on both halves of the force law, 0.382 each)
                if detail not in same.detail.split('; '):
                    same.detail = f'{same.detail}; {detail}'
                continue
            rows.append(
                CheckRow(
                    f'U:{entry.name}',
                    name,
                    CheckState.THIN,
                    detail,
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
    """A verdict decided at the scale of the center (G2c, `analyse._at_centre_scale`): 'at the scale of the center:
    σ filtered at W_c 1108 yr > 75 yr, the fine cutoff without a libration; its turning points are turns of the center
    (prominence ≥ 10°)' (W23 rule a, libration_centre_scale_tp_at_centre: the floor raised to the center-turn
    prominence p_c, a smaller swing is residue of the fast component W_c attenuates); '' otherwise."""
    window, fine = f.row.get('geo_centre_scale_cutoff_yr'), f.row.get('geo_fine_cutoff_yr')
    if not finite(window):
        return ''
    text = f'at the scale of the center: σ smoothed at W_c {years(window)}, longer than the fine cutoff {years(fine)}, at which σ shows no libration'
    first = f.row.get('geo_centre_scale_first_cutoff_yr')
    if finite(first):
        text += f' (W_c {years(first)} filtered the libration found there: analyzed again at P / {fmt(f.s.filter_min_libration_cutoffs)})'
    prominence = f.row.get('geo_tp_prominence_deg')
    if f.s.libration_centre_scale_tp_at_centre and finite(prominence):
        text += f'; its turning points are turns of the center that stand out by ≥ {fmt(prominence)}°'
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
    """Rule G1q (a 1:1 analyzed again at the quasi-satellite floor), '' when it did not fire."""
    visit, floor = _held(f, QS_VISIT), _held(f, QS_FLOOR)
    if not (visit and floor):
        return ''
    return (
        f'the orbit-averaged angle stays in the quasi-satellite well for {entry_compare(visit, " yr")}; there the '
        f'shortest libration period the averaged potential allows (its floor) is {entry_compare(floor, " yr")}, the '
        'tadpole floor, so σ is read again with that floor'
    )


def rule_clauses(f: MMRFacts) -> str:
    """'; '-joined clauses of the geometry rules on this record (the scale of the center, quasi-satellite floor, phase
    return, edges, the 1:1 profile floor and a turn read in the edge trim)."""
    parts = [centre_scale_clause(f), quasi_satellite_clause(f), subfloor_clause(f), return_clause(f)] + edge_clauses(f)
    parts += [profile_floor_clause(f), qs_run_floor_clause(f), edge_turn_clause(f)]  # W28 C3, C4, W38 (1:1 only)
    return '; '.join(part for part in parts if part)


def subfloor_clause(f: MMRFacts) -> str:
    """Rule G1f (W28 B5, card g08-2 B; `analyse._subfloor_pass`, columns geo_subfloor_*), '' when its pass was not
    kept: 'a two-body cycle of 105 yr, below the 150-yr floor, read again at the floor 35.3 yr of its own averaged
    potential (T0_th 112 yr / √10) and kept: F by direct averaging, the center and the force law (L4) confirm it'."""
    if not (finite(f.row.get('geo_subfloor_kept')) and float(f.row['geo_subfloor_kept']) == 1):
        return ''
    period, t0, floor = (f.row.get(f'geo_subfloor_{name}') for name in ('period_yr', 't0_theory_yr', 'floor_yr'))
    entry = next((e for name, e in f.path.items() if name.startswith('G1f floor')), None)  # its threshold: the own floor
    own_text = f'the {years(entry.threshold)} floor' if entry is not None and finite(entry.threshold) else 'the floor'
    return (
        f'a two-body cycle of {years(period)}, below {own_text}, read again at the floor {years(floor)} of its own averaged '
        f'potential (T0_th {years(t0)} / √{fmt(f.s.mmr_strength_check_ratio)}) and kept: F by direct averaging, the '
        'center and the force law (L4) confirm it'
    )


# --------------------------------------------------------------------------------------------- W28 E19b: co-orbitals
# The words of the co-orbital and record-edge rules (W28 C3, C4, C6, C11, C14; cards g02-5 B, g02-6 B, g02-9 A,
# g01-7 A, q-4 A, q-5 i): a single 1:1 leg (leaf G4c) or a half-swing between passages (leaf G4h), the flag
# extend_time, the 1:1 profile floor, a turn read in the edge trim, accepted runs the status does not count.

SINGLE_LEAVES = ('G4c', 'G4h')  # leaves that judge a circulation by one swing: a 1:1 leg (C6), a half-swing (q-5 i)
LEG_WORDS = {'tadpole': 'tadpole', 'horseshoe': 'horseshoe', 'qs': 'quasi-satellite', 'passing': 'passing'}
# the first discrepancy of a leg that is not transient (`decide.leg_outcome`, column geo_leg_discrepancy) -> words
LEG_DISCREPANCY_WORDS = {
    'passing': 'it laps past the planet (it crosses both the planet and L3), not one swing of a co-orbital well',
    'force_law': 'its local 1:1 law fails',
    'energy': 'its local energy integral fails',
    'turn': 'a turn is a kick, not a reflection',
}
# accepted runs the status does not count and that no other mark names (C11; a forced wave, an induced episode and a
# run before the last jump of a have their own marks): their tag after '#id' on (a) and their words
UNCOUNTED_TAGS = {RejectReason.BELOW_PERIOD_FLOOR: ' < floor', RejectReason.CENTRE_TURNS: ' center turns'}
TURN_SITE_WORDS = {'planet': 'the planet’s', 'l3': 'L3’s'}


def interval(t0: Any, t1: Any) -> str:
    """'586–665 yr', '28.5–32.3 kyr': a span in the unit of its end (chosen as `years` does: 500–999.7 yr is
    '0.5–1 kyr'); two unequal ends that print alike get the digits that tell them apart ('139.8–140.2 kyr')."""
    if not (finite(t0) and finite(t1)):
        return '—'
    scale, unit = _time_scale(t1)
    a, b = float(t0) / scale, float(t1) / scale
    return f'{_apart(a, b)} {unit}' if fmt(a) == fmt(b) and a != b else f'{fmt(a)}–{fmt(b)} {unit}'


def _apart(a: float, b: float) -> str:
    """'139.8–140.2': two numbers with the fewest decimals (1 to DIGITS_MAX) that tell them apart."""
    decimals = next((d for d in range(1, DIGITS_MAX) if f'{a:.{d}f}' != f'{b:.{d}f}'), DIGITS_MAX)
    return f'{a:.{decimals}f}–{b:.{decimals}f}'


def _out_in(turns) -> str:
    """'out/in 0.836 and 1.2': |a − a_P| after / before each turn (1 for a reflection)."""
    values = [fmt(turn.out_in) for turn in turns if turn is not None and finite(turn.out_in)]
    return f'out/in {" and ".join(values)}' if values else 'out/in not measured'


def single_leg(f: MMRFacts):
    """(leg, outcome, discrepancy) of the 1:1 leg that leaf G4c reads (`decide.deciding_leg`), None without one."""
    g = f.a.geometry
    return deciding_leg(g, f.s) if g is not None and g.coorbital else None


def _edge_words(edge: str) -> Tuple[str, str]:
    """('start', 'before the start') or ('end', 'beyond the end') of a leg or half-swing the record edge cuts."""
    return ('start', 'before the start') if edge == LEG_EDGE_START else ('end', 'beyond the end')


def leg_clause(f: MMRFacts) -> str:
    """Leaf G4c's leg in words (C6, q-4 A), by its outcome: transient 'one horseshoe leg 586–665 yr, force and energy
    conserved: …'; uncertain 'one passing leg 779–934 yr between passages: it laps past the planet …, so no capture is
    shown and none excluded'; extend 'one clean … leg cut by the record’s end: …'; scatter 'its best 1:1 leg 552–613
    yr turns by kicks (out/in 0.605 and 1.59), not reflections'. '' without a deciding leg."""
    found = single_leg(f)
    if found is None:
        return ''
    leg, outcome, why = found
    topology, span = LEG_WORDS.get(leg.topology, leg.topology), interval(leg.t0_yr, leg.t1_yr)
    tol = f'|ln out/in| ≤ {fmt(f.s.mmr_coorbital_turn_energy_tol)}'
    if outcome == LEG_TRANSIENT:
        return (
            f'one {topology} leg {span}, force and energy conserved: its local 1:1 law holds (K/K_theory '
            f'{fmt(leg.force_scale)}), its energy integral passes and both turns reflect ({_out_in(leg.turns)}; {tol}); '
            'half a cycle, no closed one'
        )
    if outcome == LEG_UNCERTAIN:
        words = LEG_DISCREPANCY_WORDS.get(why, snake_words(why))
        if why == 'force_law' and str(leg.force) == 'pass' and finite(leg.force_scale):
            words = f'its local 1:1 law pushes a the wrong way (K/K_theory {fmt(leg.force_scale)} ≤ 0)'
        if why == 'turn':
            words += f' ({_out_in(leg.turns)}; a reflection keeps {tol})'
        return f'one {topology} leg {span} between passages: {words}, so no capture is shown and none excluded'
    if outcome == LEG_EXTEND:
        where, rest = _edge_words(leg.edge)
        turn = leg.exit if leg.edge == LEG_EDGE_START else leg.entry
        at = f' at {years(turn.t_yr)}' if turn is not None and finite(turn.t_yr) else ''
        return (
            f'one clean {topology} leg {span} cut by the record’s {where}: its turn{at} reflects ({_out_in(leg.turns)}); '
            f'the rest of the cycle lies {rest}'
        )
    if outcome == LEG_SCATTER:
        return f'its best 1:1 leg {span} turns by kicks ({_out_in(leg.turns)}), not reflections'
    return ''


def half_libration_clause(f: MMRFacts) -> str:
    """Leaf G4h's half-swing in words (q-5 i, `decide.deciding_half_libration`): 'one half-swing 28.5–32.3 kyr cut by
    the record’s end follows the resonance’s one-harmonic law (R² 0.716/0.621 ≥ 0.4) and its force turns σ back at both
    turns; the rest of the cycle lies beyond the end'. '' without one."""
    g = f.a.geometry
    half = deciding_half_libration(g, f.s) if g is not None and not g.coorbital else None
    if half is None:
        return ''
    law = f'R² {fmt(half.r2_h1)}/{fmt(half.r2_h2)} ≥ {fmt(f.s.mmr_force_r2_fail)}'
    head = f'one half-swing {interval(half.t0_yr, half.t1_yr)}'
    follows = f'follows the resonance’s one-harmonic law ({law}) and its force turns σ back at both turns'
    if half.edge:
        where, rest = _edge_words(half.edge)
        return f'{head} cut by the record’s {where} {follows}; the rest of the cycle lies {rest}'
    return f'{head} between passages over the top {follows}; half a cycle, no closed one, so no libration is shown and none excluded'


def single_swing_clause(f: MMRFacts) -> str:
    """The swing a single-swing leaf decided on (G4c: the leg, G4h: the half-swing), '' at any other leaf."""
    return {'G4c': leg_clause, 'G4h': half_libration_clause}.get(f.rule, lambda _: '')(f)


# (rec_rule, reason) of a single-swing leaf -> the verdict in plain words ({comm}: the commensurability)
SINGLE_SWING_LINES = {
    ('G4c', 'dynamic_episode'): 'TRANSIENT: one leg in a co-orbital well of {comm}, held by the planet’s force; no closed cycle.',
    ('G4c', 'one_coorbital_leg'): 'UNCERTAIN: one leg near the planet does not show a capture, nor exclude one.',
    ('G4c', 'baseline_needed'): 'UNCERTAIN: a clean leg near the planet runs into the record’s edge.',
    ('G4h', 'one_half_libration'): 'UNCERTAIN: one half-swing between passages does not show a libration, nor exclude one.',
    ('G4h', 'baseline_needed'): 'UNCERTAIN: one half-swing of the resonance runs into the record’s edge.',
}


def _single_swing_steps(f: MMRFacts) -> List[Step]:
    """Step 0 (the circulation the leaf judged, the reported word after it) and the leaf G4c/G4h with its swing."""
    circulation = _circulation_shape_step(f, '')
    judged = 'its best single leg' if f.rule == 'G4c' else 'its one half-swing between passages'
    text = circulation.outcome.replace('→ circulation', f'→ a circulation judged on {judged} → {f.v.angle_status}')
    leaf = Step('single', rule_name(f.rule), StepState.LEAF, f'{single_swing_clause(f)} → {tree_status(f.v)}')
    return [replace(circulation, outcome=text), leaf]


def profile_floor_clause(f: MMRFacts) -> str:
    """Rule coorbital_floor_profile (C3, cards g02-1 A, q-3 A; episodes.csv cpl_*): '1:1 floor of #0: its turns
    reflect on the planet’s top of the averaged potential; the loop through them takes 119 yr, so its own floor is
    59.4 yr (the tadpole floor 111 yr)'. The deciding episode's, else the first run with its own floor; '' when the
    rule did not act."""
    if str(Rule.COORBITAL_FLOOR_PROFILE) not in f.v.rules_fired:
        return ''
    rows = [row for row in f.episodes.rows if _text(row.get('cpl_floor_source')) == 'profile_loop' and finite(row.get('cpl_floor_yr'))]
    if not rows:
        return ''
    decisive = f.episodes.decisive or {}
    row = next((r for r in rows if _same_id(r.get('episode_id'), decisive.get('episode_id'))), rows[0])
    sites = [TURN_SITE_WORDS.get(site, site) for site in dict.fromkeys(_text(row.get('cpl_turn_sites')).split(';')) if site]
    tops = f'{" and ".join(sites)} top{"s" if len(sites) > 1 else ""}' if sites else 'the tops'
    tadpole = f.row.get('flt_period_floor_yr')
    return (
        f'1:1 floor of #{fmt(row.get("episode_id"))}: its turns reflect on {tops} of the averaged potential; the loop '
        f'through them takes {years(row.get("cpl_loop_period_yr"))}, so its own floor is {years(row.get("cpl_floor_yr"))}'
        + (f' (the tadpole floor {years(tadpole)})' if finite(tadpole) else '')
    )


def qs_run_floor_clause(f: MMRFacts) -> str:
    """Rule coorbital_qs_run_floor (G1q pass; episodes.csv cpl_*): '1:1 floor of #1: its turns all lie in the
    quasi-satellite well; the loop through them takes 50.5 yr, so its own floor is 35.4 yr (the QS floor 55.7 yr)'.
    The deciding episode's, else the first qs_loop run; '' when the rule did not act."""
    if str(Rule.COORBITAL_QS_RUN_FLOOR) not in f.v.rules_fired:
        return ''
    rows = [row for row in f.episodes.rows if _text(row.get('cpl_floor_source')) == 'qs_loop' and finite(row.get('cpl_floor_yr'))]
    if not rows:
        return ''
    decisive = f.episodes.decisive or {}
    row = next((r for r in rows if _same_id(r.get('episode_id'), decisive.get('episode_id'))), rows[0])
    qs_floor = f.row.get('flt_qs_floor_yr')
    return (
        f'1:1 floor of #{fmt(row.get("episode_id"))}: its turns all lie in the quasi-satellite well; the loop through '
        f'them takes {years(row.get("cpl_loop_period_yr"))}, so its own floor is {years(row.get("cpl_floor_yr"))}'
        + (f' (the QS floor {years(qs_floor)})' if finite(qs_floor) else '')
    )


def edge_turn_clause(f: MMRFacts) -> str:
    """Rule coorbital_edge_turn (C4, card g02-2 A): 'the turn at 960 yr lies in the filter’s edge trim: read on σ
    averaged over 3 orbits, a reflection on the planet’s top (out/in 0.9, a crosses a_P); it only closes the cycle'.
    '' when the rule did not act."""
    if str(Rule.COORBITAL_EDGE_TURN) not in f.v.rules_fired:
        return ''
    g = f.a.geometry
    turns = [turn for turn in (g.edge_turns if g is not None else ()) if turn.added]
    orbits = fmt(f.s.mmr_coorbital_edge_turn_orbits)
    parts = []
    for turn in turns:
        site = TURN_SITE_WORDS.get(turn.site, turn.site)
        out_in = f'out/in {fmt(turn.out_in)}' if finite(turn.out_in) else 'out/in not measured'
        crosses = ', a crosses a_P' if turn.a_crosses else ''
        parts.append(
            f'the turn at {years(turn.t_yr)} lies in the filter’s edge trim: read on σ averaged over {orbits} orbits, a '
            f'reflection on {site} top ({out_in}{crosses})'
        )
    if not parts:
        times = [t for t in _text(f.row.get('geo_edge_raw_tp_t')).split(';') if t]
        parts = [f'the turn at {years(float(t))} lies in the filter’s edge trim, read on σ averaged over {orbits} orbits' for t in times]
    return '; '.join(parts) + '; it only closes the cycle' if parts else ''


def _run_floor(f: MMRFacts, row: Mapping[str, Any]) -> float:
    """The period floor the decision compared this run with (`decide.period_floor`: its own 1:1 floor, the quasi-
    satellite or tadpole floor), read from the geometry's run; the row's own floor or the angle's when the run is gone."""
    g = f.a.geometry
    run = next((r for r in (g.runs if g is not None else ()) if _same_id(r.episode_id, row.get('episode_id'))), None)
    if run is not None:
        return period_floor(g, f.s, run)[0]
    if _text(row.get('cpl_floor_source')) == 'profile_loop' and finite(row.get('cpl_floor_yr')):
        return float(row['cpl_floor_yr'])
    floor = _float(f.row.get('flt_period_floor_yr'))
    # coorbital_qs_run_floor: a qs_loop reading exists only for a run inside the G1q well (`CoorbitalRunFloor.read`)
    if _text(row.get('cpl_floor_source')) == 'qs_loop' and finite(row.get('cpl_floor_yr')):
        return min(float(row['cpl_floor_yr']), floor) if finite(floor) else float(row['cpl_floor_yr'])
    return floor


def uncounted_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Accepted runs the status does not count and no other mark names (episodes.csv ineligible_reason
    below_period_floor or centre_turns, C11)."""
    return [row for row in rows if episode_accepted(row) and _text(row.get('ineligible_reason')) in UNCOUNTED_TAGS]


def not_counted_clause(f: MMRFacts, short: bool = False) -> str:
    """Card g02-9 A, the 'not counted' phrase: 'Not counted: #0 263–299 yr, P 74 yr < its floor 75.7 yr (faster than
    the angle’s shortest libration).' (`short`: '#0 P 74 yr < floor 75.7 yr: not counted', for step 0); '' when every
    accepted run counts."""
    parts, shorts = [], []
    for row in uncounted_rows(f.episodes.rows):
        name = f'#{fmt(row.get("episode_id"))}'
        if _text(row.get('ineligible_reason')) == RejectReason.BELOW_PERIOD_FLOOR:
            floor = _run_floor(f, row)
            period = row.get('period_med_yr')
            parts.append(
                f'{name} {interval(row.get("t_start_yr"), row.get("t_end_yr"))}, P {years(period)} < its floor {years(floor)} '
                '(faster than the angle’s shortest libration)'
            )
            shorts.append(f'{name} P {years(period)} < floor {years(floor)}')
        else:
            parts.append(f'{name} {interval(row.get("t_start_yr"), row.get("t_end_yr"))} (its center runs a whole revolution inside it)')
            shorts.append(f'{name} center turns')
    if not parts:
        return ''
    return f'{", ".join(shorts)}: not counted' if short else f'Not counted: {"; ".join(parts)}.'


def _cap_detail(f: MMRFacts, row: CheckRow) -> CheckRow:
    """The numbers behind a cap, where the record keeps them (the return, the fast spread of σ_f, the 1:1 leg whose
    turns scatter)."""
    cap = row.key[4:]
    if cap == 'one_return_cycle':  # the times are in the angle-shape step (no number twice)
        episode = f.episodes.decisive or {}
        start, end = episode.get('return_t_a_yr'), episode.get('return_t_b_yr')
        period = float(end) - float(start) if finite(start) and finite(end) else None
        row.detail = f'one cycle of {years(period)} (its return: the angle-shape step)' if return_clause(f) else ''
    elif cap == 'near_episode' and leg_clause(f).startswith('its best 1:1 leg'):  # W28 C6: a 1:1 leg that scatters
        leg = single_leg(f)[0]
        row.name = 'a 1:1 leg almost made a capture, but both its turns are kicks, not reflections'
        row.short = f'leg {interval(leg.t0_yr, leg.t1_yr)}, {_out_in(leg.turns)}'
    elif cap == 'short_period_aliased':
        row.detail = _short_period_detail(f)
        row.name, row.short = _short_period_title(f)
    return row


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
        else:  # neither part on record: the cap's general words
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


CIRCULATION_MARGINS = ('C1', 'C2', 'C3', 'C4')  # they measure how σ circulates (C5 and L8: the sampling, kept)
AROUND_LEG = 'the circulation around the leg, not the leg'  # their note on a single-swing capture (W28 Codex S8)


def _around_leg(f: MMRFacts, rows: List[CheckRow]) -> List[CheckRow]:
    """A capture decided on one swing (G4c, G4h: transient) keeps C1–C4, which measure the circulation around the
    leg: they neither support nor object to the capture, so they go to the n/a line with that note (2024 XP16, 2023
    OE5, 2025 YU11: C4 failed and 'lowered' a transient there). Under the minimal policy only (the v2 counts kept)."""
    if not (policy_minimal(f.v) and f.rule in SINGLE_LEAVES and str(f.v.status) in ('transient', 'resonant')):
        return rows
    for row in rows:
        if row.key in CIRCULATION_MARGINS:
            row.state, row.kind, row.note, row.short = CheckState.NOT_APPLICABLE, KIND_NA, AROUND_LEG, AROUND_LEG
    return rows


def mmr_checks(f: MMRFacts) -> List[CheckRow]:
    if not str(f.v.confidence):
        return []
    rows = _around_leg(f, [row for key, margin in f.margins.items() for row in _margin_row(f, key, margin)])
    rows += _ungraded_step_rows(f)
    return confidence_rows(f.v, f.s, rows, f.whole_record, lambda row: _cap_detail(f, row))


def _ungraded_step_rows(f: MMRFacts) -> List[CheckRow]:
    """The output step's turn per step when no margin was graded (the W28 tree left the row uncertain: no margins,
    `decide` margins) and σ turns at least the L8a bar per step (1580: 44.9°, 2003 TG2): printed, not graded, so Q's
    'Yes' does not stand beside an unexplained number in Numbers."""
    sampling = f.a.sampling
    if 'L8' in f.margins or f.v.margins or sampling is None:
        return []
    rotation = degrees(sampling.step_rotation_max_rad)
    if not holds(rotation, '>=', f.s.mmr_l8_step_full_deg):
        return []
    return [
        CheckRow(
            'L8a',
            MMR_MARGIN_NAMES['L8a'],
            CheckState.NOT_APPLICABLE,
            f'turn per step {compare(rotation, "<", f.s.mmr_l8_step_full_deg, "°")}',
            f'< {fmt(f.s.mmr_l8_step_full_deg)}°',
            COUNTS_NA,
            'not graded: the status tree left the row uncertain, so it graded no margin',
        )
    ]


def confidence_rows(verdict, settings, margin_rows: Sequence[CheckRow], whole_record: bool, finish=None) -> List[CheckRow]:
    """The confidence block of a verdict with a level: its margin rows, the comparisons near their thresholds and the
    caps (each through `finish` when given), a failed cap-class margin merged into its cap (`merge_limits`), in reading
    order (`sort_checks`)."""
    caps = cap_rows(verdict.caps, str(verdict.confidence), whole_record)
    rows = list(margin_rows) + near_threshold_rows(verdict, settings) + [finish(row) if finish else row for row in caps]
    for row in rows:  # a margin part without its own kind (L8a, L8b) that does not apply here
        if row.state == CheckState.NOT_APPLICABLE and not row.is_limit:
            row.kind = KIND_NA
    return sort_checks(merge_limits(rows))


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
    v = f.v
    angle, subtype = str(v.angle_status), str(v.angle_subtype)
    accepted = f.episodes.counted  # W28 C11: a run below the floor is no librating stretch
    if angle == 'undefined':
        return _undefined_step(f)
    rules = rule_clauses(f)
    if angle == 'libration':
        what = (_union_clause(f) or 'σ librates over the whole record') + (f'; {rules}' if rules else '')
        return _shape(f'{what} → libration')
    if angle == 'episodic' and _forced_wave_word(f):
        text = (
            f'σ librates only in {_forced_wave_clause(f)} (the angle word); the status reads the angle without it: the '
            f'center passes {fmt(f.row.get("geo_centre_max_pass_rev"))} rev in one direction → episodic, read as {_tree_word(f)}'
        )
        return _shape(text)
    if angle == 'episodic':
        return _episodic_shape_step(f, len(accepted), rules)
    if angle == 'circulation':
        return _circulation_shape_step(f, subtype)
    if angle == 'drift':
        return _drift_shape_step(f)
    if angle == 'near_separatrix':
        pattern_b = 'slows at the top below its rotating branch in most revolutions (pattern B)'
        what = pattern_b if subtype == 'pattern_b' else 'slips over the top (pattern A)'
        return _shape(f'no libration; σ {what} → near_separatrix')
    if angle == 'insufficient_time':
        return _shape(_insufficient_text(f, subtype), StepState.LEAF)
    return _shape(angle)


def _episodic_shape_step(f: MMRFacts, n_accepted: int, rules: str) -> Step:
    """Step 0 of an episodic σ: the deciding episode's edge gaps against LIBRATION_COVERAGE_PERIODS; when both are
    within it, what made the angle episodic (an exit seen, 24820) follows with 'but'."""
    episode = f.episodes.decisive or {}
    gaps = coverage_text(episode, f.s)
    what = 'σ librates in part of the record' if n_accepted == 1 else f'{n_accepted} separate librating stretches'
    if n_accepted > 1:
        gaps = gaps.replace('it misses', 'the deciding one misses', 1)
    if rules and gaps and coverage_within(episode, f.s):  # the gaps alone would cover: what made it episodic follows
        rules = f'but {rules}'
    what = '; '.join(part for part in (what, gaps, rules) if part)
    return _shape(f'{what} → episodic')


def union_kind(f: MMRFacts) -> str:
    """How a union of episodes (W22) made the word libration: 'center' (found only at the scale of the center,
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
            'episodes without a circulation between them cover the record at the scale of the center (the word only: '
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
    """The shape of insufficient_time by its subtype: an arc of the slow center (G5a, the center turns back once within
    one revolution: part of one slow swing, 2003 KQ18), an arc of sigma_f without turning points (G6), a stall, or too
    few turning points (few_cycles)."""
    tp = fmt(f.row.get('geo_n_tp'))
    turn = _slow_arc(f)
    ripple = ripple_clause(f)
    if slow_swing_decided(f):
        return '; '.join(part for part in (ripple, slow_swing_text(f), orbit_period_clause(f)) if part) + ' → insufficient_time (arc)'
    if subtype == 'arc' and turn is not None and ripple:
        return (
            f'{ripple}; outside it the slow center turns back once, within {fmt(turn.value / (2 * math.pi), 2)} rev: part '
            'of one slow swing, where a drift and a slow libration look alike → insufficient_time (arc)'
        )
    if subtype == 'arc' and turn is not None:
        return (
            f'{tp} turning points ride on a slow center that turns back once, within {fmt(turn.value / (2 * math.pi), 2)} rev: '
            'part of one slow swing, where a drift and a slow libration look alike → insufficient_time (arc)'
        )
    if ripple:
        return f'{ripple}; {_insufficient_rest(f, subtype, tp)}'
    return _insufficient_rest(f, subtype, tp)


def _insufficient_rest(f: MMRFacts, subtype: str, tp: str) -> str:
    """The arc of σ_f (G6), a stall, or too few turning points."""
    if subtype == 'arc':
        arc = f.entry(G6_RANGE)
        degrees = math.degrees(arc.value) if arc is not None and finite(arc.value) else None
        return f'no turning point; σ_f runs an arc of {fmt(degrees)}° → insufficient_time (arc)'
    if subtype == 'stall':
        return 'no turning point and no arc: σ_f stalls → insufficient_time (stall)'
    return f'{tp} turning points, too few for a libration → insufficient_time'


def _slow_arc(f: MMRFacts):
    """G5a's last entry when G5a set the word (the slow center turns back once within a revolution), else None. The
    entry alone does not say so (B6b's ripple test writes it too: 1992 SZ 1M-10J+1, 2002 BG 1E-10S-1 are few_cycles
    G6 rows with it held), so the angle rule (geo_angle_rule) decides, and where that rule set no insufficient_time word
    (`insufficient_angle_rule` '') only an arc can be G5a's."""
    rule = insufficient_angle_rule(f)
    if rule and rule != 'G5a':
        return None
    if not rule and getattr(f, 'v', None) is not None and str(f.v.angle_subtype) != 'arc':
        return None
    entry = _held(f, G5A_ONE_REVOLUTION)
    return entry if entry and finite(entry.value) else None


def ripple_waves_of(f: MMRFacts) -> list:
    """The unconfirmed ripples set aside before G2 (W28 B6b, `ForcedWave.ripple`, rule ripple_on_arc); none for facts
    without an analysis (a step text read from the ledger alone)."""
    g = getattr(getattr(f, 'a', None), 'geometry', None)
    return [wave for wave in g.forced_waves if wave.ripple] if g is not None else []


def ripple_clause(f: MMRFacts) -> str:
    """B6b (card g08-4 A): 'the ripple #0 (P 680 yr, half-amplitude 3.11° < 45°: no force law or potential measurable;
    identity |slope/c − 1| 0.35 > 0.2) is set aside as an unconfirmed ripple, an auxiliary interval'; '' without one.
    The numbers are the ripple's, those its G2 ledger entries compared."""
    waves = ripple_waves_of(f)
    if not waves:
        return ''
    s = f.s
    parts = []
    for wave in waves:
        amp = degrees(wave.half_amp_max_rad)
        slope = abs(wave.identity_slope_ratio - 1) if finite(wave.identity_slope_ratio) else None
        if holds(slope, '<=', s.mmr_identity_slope_tol) is False:
            identity = f'identity |slope/c − 1| {compare(slope, "<=", s.mmr_identity_slope_tol)}'
        else:
            identity = f'identity corr {compare(wave.identity_corr, ">=", s.mmr_identity_corr_min)}'
        parts.append(
            f'#{wave.episode_id} (P {years(wave.period_yr)}, half-amplitude {compare(amp, "<", s.mmr_a_response_min_amp_deg, "°")}: '
            f'no force law or potential measurable; {identity})'
        )
    if len(waves) == 1:
        return f'the ripple {parts[0]} is set aside as an {RIPPLE_MARK}, an auxiliary interval'
    return f'the ripples {", ".join(parts)} are set aside as {RIPPLE_MARK}s, auxiliary intervals'


def slow_swing_decided(f: MMRFacts) -> bool:
    """G5w decided the angle: the angle rule (geo_angle_rule), else (`insufficient_angle_rule` '') the rule
    slow_swing_in_well (W28 B6c: `analyse._decide_angle` fires it exactly then)."""
    if getattr(f, 'v', None) is None or str(f.v.angle_status) != 'insufficient_time':
        return False
    rule = insufficient_angle_rule(f)
    return rule == 'G5w' if rule else str(Rule.SLOW_SWING_IN_WELL) in f.v.rules_fired


# the angle rules that set an insufficient_time word (`decide._g4_to_g6`, `_g6_ripple`)
INSUFFICIENT_ANGLE_RULES = ('G5a', 'G5w', 'G6')


def insufficient_angle_rule(f: MMRFacts) -> str:
    """The angle rule of an insufficient_time row as saved (geo_angle_rule: G5a, G5w or G6); '' when it is another
    rule: the status tree set the word (G4c's edge leg, G4h's edge half-libration keep the circulation's angle rule),
    and the readers then rebuild it from the ledger."""
    rule = getattr(f, 'angle_rule', '')
    return rule if rule in INSUFFICIENT_ANGLE_RULES else ''


def slow_swing_text(f: MMRFacts) -> str:
    """G5w (W28 B6c, card g07-5 A; columns geo_slow_swing_*): the slow center turns back twice and the half-swing
    between the turns is one of this resonance's own well: the slow identity, the a swing and the period 2(t2 − t1)
    against the directly averaged pendulum at its amplitude (T0_th F(A)), the two ratios within [1/√x, √x]."""
    row, s = f.row, f.s
    lo, hi = _strength_band(f)

    def value(name: str):
        return row.get(f'geo_slow_swing_{name}')

    slope = value('identity_slope_ratio')
    offset = abs(float(slope) - 1) if finite(slope) else None
    rng = row.get('geo_centre_range_rad')
    within = f'; the center stays within {fmt(float(rng) / (2 * math.pi), 2)} rev' if finite(rng) else ''
    return (
        f'the slow center turns back twice (at {years(value("t0_yr"))} and {years(value("t1_yr"))}{within}): the '
        f'{fmt(value("amp_deg"))}° half-swing between them is this resonance’s well: slow identity |slope/c − 1| '
        f'{compare(offset, "<=", s.mmr_identity_slope_tol)}, corr {compare(value("identity_corr"), ">=", s.mmr_identity_corr_min)}; '
        f'a swing {fmt(value("a_ratio"))} × the pendulum’s, period {years(value("period_yr"))} = {fmt(value("period_ratio"))} × '
        f'T0_th F(A) (the pendulum’s period at this amplitude; T0_th {years(value("t0_theory_yr"))}), both in [{fmt(lo)}, {fmt(hi)}]'
    )


def orbit_period_clause(f: MMRFacts) -> str:
    """The cutoff bound by the body's orbital period (W28 B6a, flt_cutoff_binding 'orbit', ctx_P_orb_yr): 'the 379-yr
    cycles are the orbital period (cutoff 569 yr = 1.5 P_orb)'; '' otherwise."""
    if _text(f.row.get('flt_cutoff_binding')) != 'orbit':
        return ''
    period, cutoff = f.row.get('ctx_P_orb_yr'), f.row.get('flt_cutoff_yr')
    if not finite(period):
        return ''
    return f'the {years(period)} cycles are the orbital period (cutoff {years(cutoff)} = {fmt(f.s.mmr_filter_orbit_factor)} P_orb)'


FORCED_WAVE_WORD = 'sigma librates in the forced-wave runs'
RIPPLE_MARK = 'unconfirmed ripple'


def _forced_wave_word(f: MMRFacts) -> bool:
    """The reported word is the geometry's with the forced-wave runs (`decide.forced_wave_word`, its ledger line): the
    status read a circulation or a drift without them."""
    rewritten = any(FORCED_WAVE_WORD in entry.name for entry in f.v.decision_path)
    return bool(_forced_wave_clause(f)) and rewritten


def _tree_word(f: MMRFacts) -> str:
    """'a circulation' or 'a drift': the word the status tree read under a forced-wave rewrite (`_forced_wave_word`)."""
    return 'a drift' if 'drift' in str(f.v.reason) else 'a circulation'


def _circulation_shape_step(f: MMRFacts, subtype: str) -> Step:
    """The angle shape of a circulation (an alias: the turn per step; accepted runs that are forced waves or that the
    status does not count named, W28 C11)."""
    if subtype == 'alias':
        text = f'σ turns {fmt(f.row.get("smp_step_rotation_deg_max"))}° per output step → circulation (alias)'
        return _shape(text)
    if f.rule == '3k2':
        net = _net_shift(f, K2_NET)
        moves = f'advances {net} rev net' if net else f'passes {fmt(f.row.get("geo_centre_max_pass_rev"))} rev'
        text = f'librating stretches, a ripple on a center that {moves} (step 3) → circulation'
        return _shape(text)
    forced = '; '.join(part for part in (_forced_wave_clause(f), not_counted_clause(f, short=True)) if part)
    why = f'no libration ({forced})' if forced else 'no libration'
    net = _net_shift(f, G4B_NET)
    if net:
        passed = fmt(f.row.get('geo_centre_max_pass_rev'))
        text = f'{why}; the center advances {net} rev net, at most {passed} rev in one pass (G4b) → circulation'
        return _shape(text)
    text = f'{why}; the center passes {fmt(f.row.get("geo_centre_max_pass_rev"))} rev in one direction → circulation'
    return _shape(text)


def _net_shift(f: MMRFacts, name: str) -> str:
    """'|net| >= x' when the net-shift rule `name` (G4b or its use in K2's word) held, else ''."""
    entry = _held(f, name)
    return compare(entry.value, '>=', entry.threshold) if entry else ''


def _drift_shape_step(f: MMRFacts) -> Step:
    """The angle shape of a drift: the geometry's (G5), or K2's word (rule 3k2: librating stretches that are a ripple
    on a kinematic drift of the center)."""
    if f.rule == '3k2':
        shift = _held(f, K2B_SHIFT)
        if shift:
            text = f'σ librates about a center that moves {compare(shift.value, ">", shift.threshold)} × its amplitude (step 3) → drift'
            return _shape(text)
        accepted = f.episodes.accepted
        what = 'σ librates in part of the record' if len(accepted) == 1 else f'{len(accepted)} separate librating stretches'
        return _shape(f'{what}, a ripple on a kinematic drift of the center (step 3) → drift')
    text = f'oscillates, but the center drifts over {percent(f.row.get("geo_drift_fraction"))} of the record → drift'
    return _shape(text)


def _forced_wave_clause(f: MMRFacts) -> str:
    """G2: accepted runs judged forced waves of another argument (`decide.forced_wave_episodes`), '' without one."""
    count = f.row.get('geo_n_forced_wave_episodes')
    if not finite(count) or float(count) < 1:
        return ''
    line = f.row.get('geo_forced_wave_line')
    why = f'on the forced line {line} with a failed identity' if isinstance(line, str) and line else 'σ_dot against c (a − a*)'
    return f'{plural(int(count), "run")} of P {fmt(f.row.get("geo_forced_wave_period_yr"))} yr {why}: a forced wave'


def _circulation_steps(f: MMRFacts) -> List[Step]:
    v, s = f.v, f.s
    if f.rule == 'C5':
        far = f.entry(C5_FAR)
        branch = _held(f, C5_BRANCH)
        text = (
            f'{"one branch" if branch else "several branches (ambiguous)"} of ν·Δt over the a range; '
            f'⟨a⟩ {compare(far.value if far else None, ">=", s.mmr_c5_far_ratio_min)} a-scales from a* → {tree_status(v)}'
        )
        return [Step('alias', 'alias (C5)', StepState.LEAF, text)]
    if f.rule == 'R0':
        text = f'σ turns back at times, but the energy does not fit a potential from a → {tree_status(v)}'
        return [Step('potential', 'potential (R0)', StepState.LEAF, text)]
    if f.rule in STRENGTH_LEAVES:  # 2897: the strength (step 5s) decided; under the policy the angle's word is circulation
        return [_force_step(f)]
    source = v.owner or v.resonant_argument  # A3a: a neighbour makes no wave of this angle
    owner = f' (its wave comes from {source})' if source else ''
    leg = leg_clause(f)  # W28 C6: a 1:1 leg whose turns both scatter is a near miss of a capture (near-episode)
    scatter = f'; {leg}' if leg.startswith('its best 1:1 leg') else ''
    return [Step('circ', 'circulation', StepState.LEAF, f'circulation → {tree_status(v)}{owner}{scatter}')]


def _subresonance_step(f: MMRFacts) -> Step:
    sub, s, v = f.a.subresonances, f.s, f.v
    title = '1. subresonances'
    if sub is None:
        return Step('sub', title, StepState.NOT_RUN, 'not run: a derived row' if f.derived else _not_triggered(f))
    if sub.accepted is not None:
        coherence = compare(sub.accepted.phi0_coherence, '>=', s.mmr_subresonance_phi0_coh_min)
        text = f'{sub.accepted.spec.key} keeps a fixed center (φ0 coherence {coherence}) → {tree_status(v)}, origin subresonance'
        return Step('sub', title, StepState.LEAF, text)
    if str(sub.check) in ('pass', 'fail'):
        best = _best_candidate(sub.candidates)
        text = f'{plural(len(sub.candidates), "member")} scanned, none holds the center'
        return Step('sub', title, StepState.PASSED, text + (f' (best {best.spec.key}: {_member_rejection(f, best)})' if best else ''))
    why = why_not_measured(f, 'S')[1]
    return Step('sub', title, StepState.NOT_RUN, why if 'no verdict' in why else f'no verdict: {why}')


def _owner_step(f: MMRFacts) -> Step:
    owner, v = f.a.owner, f.v
    title = '2. owner'
    if owner is None:
        return Step('owner', title, StepState.NOT_RUN, 'not run on a derived row' if f.derived else 'not run')
    if 'owner_coorbital' in owner.flags:
        return Step('owner', title, StepState.NOT_APPLICABLE, f'n/a: {COORBITAL_OWNER}')
    if owner.accepted is not None:
        key, r2 = owner.accepted.spec.key, f'R² {fmt(owner.accepted.r2_median, 2)}'
        if refused_owner(f):  # F7/P2 refused it (A1): a neighbour, not the cause (g04-4, g04-6; g10-8 never 'drives a')
            claim = _identity_claim(owner) if identity_route(owner) else f'neighbor {key} ({r2})'
            where = neighbour_text(f)
            where = where[len(f.v.cause_term or key) :].strip() if where.startswith(f.v.cause_term or key) else ''
            refusal = _refusal(f)
            detail = '; '.join(part for part in (where, refusal) if part)
            return Step('owner', title, StepState.PASSED, f'{claim}{" " + detail if detail else ""}: not the cause, the tree goes on')
        if identity_route(owner):
            drives = _identity_claim(owner)
        elif not owner_confirmed(f):
            drives = f'candidate driver {key} ({r2})'
        elif str(f.v.cause_role) == ROLE_DRIVER:  # g04-5/g04-6: the circulating driver and its forced wave
            period = f.row.get('own_owner_period_yr')
            circulates = f'circulates (P {years(float(period))})' if finite(period) else 'circulates'
            drives = f'driver {key} {circulates} and its forced wave drives a ({r2})'
        else:
            drives = f'owner {key} librates in σ’s episode and drives a ({r2})'
        return Step('owner', title, StepState.LEAF, f'{drives} → {tree_status(v)}, induced')
    if str(owner.check) in ('pass', 'fail'):
        best = _best_candidate(owner.candidates)
        text = f'{plural(owner.n_candidates, "candidate")}, none drives a'
        if best is not None:
            text += f' (best {best.spec.key}: {_owner_rejection(f, best)})'
        return Step('owner', title, StepState.PASSED, text)
    return Step('owner', title, StepState.NOT_RUN, f'not run: {why_not_measured(f, "W")[1]}')


def _kinematic_step(f: MMRFacts) -> Step:
    b1 = f.entry(B1_AMPLITUDE)
    kin = f.entry(KIN_VARPI)
    title = '3. kinematic'
    if f.rule == '3k2':
        return Step('kin', title, StepState.LEAF, f'{_k2_clause(f)} → {tree_status(f.v)}')
    if f.rule == 'K3s':  # B10a, after the leaf: an episodic σ read as a circulation has no leaf of its own (37087)
        _, reference, name, _ = secular_reference(f)
        cycle = f'{name} = {years(reference)}' if name else 'a secular cycle'
        text = f'on the slow scale a does not follow σ: its slow swing is {cycle} seen through ϖ'
        return Step('kin', title, StepState.LEAF, f'{text} → {tree_status(f.v)}')
    if f.rule in ('3', '3b5'):
        if b1 is not None and b1.result is False:
            text = f'a does not carry the swing σ needs (a’s swing / the swing σ needs {compare(b1.value, ">=", b1.threshold)})'
        else:
            text = f'slow σ follows p ϖ (corr {fmt(kin.value if kin else None)}) with a flat a'
        return Step('kin', title, StepState.LEAF, f'{text} → {tree_status(f.v)}')
    if b1 is not None and finite(b1.value):
        text = f'a carries the swing σ needs (a’s swing / the swing σ needs {compare(b1.value, ">=", b1.threshold)})'
    else:
        text = 'the swing of a against the one σ needs is not measured'
    if kin is None and str(f.v.angle_status) in ('libration', 'episodic'):
        # decide._kinematic: on a librating angle the slow σ against p ϖ is not read (the episode's own cycle is)
        text += '; whether a slow σ follows the perihelion term p ϖ is asked of circulations and drifts, not of a libration'
    return Step('kin', title, StepState.PASSED, text)  # a failed L3 below is named by the builder (`_unread_failure`)


def _drift_step(f: MMRFacts) -> Step:
    angle = str(f.v.angle_status)
    if angle != 'drift':
        if decisive_centre_moves(f):  # 2003 TX9: the header's 'moving centre' beside 'does not drift'
            return Step('drift', '4. drift', StepState.NOT_APPLICABLE, f'n/a: the center moves, but σ reads as {angle}, not a drift')
        return Step('drift', '4. drift', StepState.NOT_APPLICABLE, f'n/a: the center does not drift ({angle})')
    if f.rule == '4':
        return Step('drift', '4. drift', StepState.LEAF, f'no mechanism explains the drift → {tree_status(f.v)}')
    return Step('drift', '4. drift', StepState.PASSED)


def _force_step(f: MMRFacts) -> Step:
    rule, title = f.rule, '5. force law + potential'
    if rule.startswith('5a') or rule in ('5b', '5e', '5f', '5g'):
        return Step('force', title, StepState.LEAF, origin_unknown_clauses(f) + f' → {tree_status(f.v)}')
    if strength_first(f):
        return Step('force', title, StepState.LEAF, f'read first, whatever L4 and R0 say: {branch_clause(f)} → {tree_status(f.v)}')
    if rule in BRANCH_LEAVES:
        slow = filter_too_slow_text(f) if getattr(f.a.geometry, 'coorbital', False) else ''
        slow = f' (R0: {slow})' if slow else ''  # 2026 AC4 (S9): why the 1:1 energy integral is not read
        return Step('force', title, StepState.LEAF, f'{NEITHER_MEASURED}{slow}; {branch_clause(f)} → {tree_status(f.v)}')
    if not rule.startswith('5'):
        return Step('force', title, StepState.NOT_REACHED)
    if rule == '5u/6':
        return Step('force', title, StepState.PASSED, f'{NEITHER_MEASURED}; {branch_clause(f)} → the geometry decides')
    l4, r0 = f.checks.get('a_response', ''), f.checks.get('potential', '')
    force = {'pass': 'the force law holds', 'fail': 'the force law fails (capped)'}
    potential = {'pass': 'the potential holds', 'fail': 'the potential fails'}
    parts = [
        force.get(l4, f'the force law is not measured ({short_reason(why_not_measured(f, "L4")[1])})'),
        potential.get(r0, f'the potential is not measured ({short_reason(why_not_measured(f, "R0")[1])})'),
    ]
    # the filter guard runs before the tree (W21 Q9): a check it withdrew was not measured for step 5 either
    if l4 == r0 == 'pass':
        return Step('force', title, StepState.PASSED, 'the force law and the potential both hold → the angle’s own dynamics')
    conclusion = _force_conclusion(l4, r0, _direct_strength_clause(f))
    return Step('force', title, StepState.PASSED, '; '.join(parts) + f': {conclusion} → the angle’s own dynamics')


NEITHER_MEASURED = 'neither the force law nor the potential is measured'


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
    if energy is not None and energy.cycle_classes and zone_reading_withdrawn(f):
        # the zone's reading did not survive (`decide.zone_withdrawn`: Z or L2 emptied by a guard): its counts are no
        # statement about the row, which shows zone_not_measured instead (79653 3M+3J-7)
        why = _zone_withdrawn_why(f)
        note = f'; zone not measured ({why})' if why else '; zone not measured'
    elif energy is not None and energy.cycle_classes:
        n_zone, _, n_all = _zone_counts(energy)
        note = f'; {n_zone} of {n_all} cycles in the separatrix zone'
        if has_mark(f.v, 'episodes_in_separatrix_zone'):  # the surviving flag, not the energy's raw reading
            note += ' and the angle crosses it (episodes_in_separatrix_zone)'
    kind = union_kind(f)
    if kind == 'promoted':  # rule 6u: the episodic reading reached a dynamic transient
        outcome = 'episodic → transient; the union covers the record → resonant (6u)'
    else:  # a union the status kept reads as episodic there
        outcome = f'{"episodic" if kind == "kept" else f.v.angle_status} → {tree_status(f.v)}'
    return Step('zone', '6. dynamic leaf', StepState.LEAF, f'{outcome}{note}')


def zone_reading_withdrawn(f: MMRFacts) -> bool:
    """The zone's reading was withdrawn for the row (`decide.zone_withdrawn` on the verdict's margins): its cycle
    counts and the crossing flag say nothing about the row (it shows zone_not_measured)."""
    from resonances.resonance.analysis.mmr.decide import zone_withdrawn

    return f.a.energy is not None and zone_withdrawn(f.a.energy, f.v.margins)


def _zone_withdrawn_why(f: MMRFacts) -> str:
    """'withdrawn by the filter guard' when the guard emptied Z or L2 (`unused_margins`), else ''."""
    why = f.unused.get('Z') or f.unused.get('L2') or ''
    return 'withdrawn by the filter guard' if why.startswith('withdrawn by the filter guard') else ''


TREE_STEPS = (_subresonance_step, _owner_step, _kinematic_step, _drift_step, _force_step, _zone_step)
COORBITAL_OWNER = 'the owner search is not applied to a 1:1'  # the project's rule (CLAUDE.md), not 'a 1:1 has no owner'
NO_FORCE_ON_A = 'no force on a'
OTHER_NOT_TESTED = 'other resonances not tested'  # the name of W when the owner search gave no answer
# a kinematic leaf: a does not carry the swing, so no resonance (own or other) acts on it
KINEMATIC_LEAVES = ('3', '3b5', '3k2')
# key -> (number, question, why): the steps of the status tree as questions (docs/physics/status-tree.md#steps-in-order;
# card g10-4b, extra_ru §2). 'Why' is said for a step that ran (passed or decided)
STEP_QUESTIONS = {
    'shape': ('0', 'Does σ librate?', ''),
    'sub': ('1', 'Does a sibling angle hold the center?', 'a multiplet member may be the real librator.'),
    'owner': ('2', 'Does another resonance drive a?', 'a neighbor’s force on a makes σ swing too (an induced libration).'),
    'kin': ('3', 'Does a (semi-major axis) swing with σ?', 'a resonance turns σ back through a, so a must carry the swing σ needs.'),
    'drift': ('4', 'Does the center drift without a cause?', 'a drifting center is no libration unless a mechanism explains it.'),
    'force': ('5', 'Is it σ’s own force?', 'its own pendulum makes a obey its force law and the energy fit its potential.'),
    'zone': ('6', 'How does σ behave?', 'with the own force measured (or nothing objecting), the behavior gives the status.'),
}
# the short names of the steps, for the one line of the steps after the leaf ('4–6 · drift, own force, behaviour')
STEP_SHORT = {
    'sub': 'siblings',
    'owner': 'another resonance',
    'kin': 'a swings',
    'drift': 'drift',
    'force': 'own force',
    'zone': 'behavior',
}
# a leaf of the circulation branch (rec_rule) -> its question
CIRCULATION_QUESTIONS = {
    'C5': 'Can the alias exclude a libration?',
    'R0': 'Do the turning points come from a potential?',
    'C': 'Circulation',
    'G4c': 'Is its one leg near the planet a capture?',  # W28 C6, q-4 A
    'G4h': 'Is its one half-swing the resonance’s own?',  # W28 C6, q-5 i
}


def shape_rule_ids(f: MMRFacts) -> Tuple[str, ...]:
    """The G rules behind the angle word of step 0 (`decide.decide_angle`), read from the angle word and the decision
    path: G0 unreadable; G1c/G1e/G1l (a jump of a), G1q (the quasi-satellite floor); G2 episodes (G2u joined, G2c at
    the scale of the center, G2a cut by a step of the action); G3 the separatrix patterns; G4/G4b a center that turns;
    G5 drift, G5a one slow arc; G6 too few cycles; K2 when step 3 set the word (rule 3k2)."""
    angle = str(f.v.angle_status)
    if angle == 'undefined':
        return ('G0',)
    ids = _base_shape_ids(f, angle)
    if quasi_satellite_clause(f):
        ids.insert(0, 'G1q')
    if subfloor_clause(f):
        ids.insert(0, 'G1f')
    if any(name.startswith('G1l ') for name in f.path) and not _held(f, G1L_LAST):
        ids.insert(0, 'G1l')
    if _held(f, G1E_EARLIER) and angle not in ('libration', 'episodic'):
        ids.append('G1e')
    return tuple(dict.fromkeys(ids))


def _base_shape_ids(f: MMRFacts, angle: str) -> List[str]:
    """The G rule of the angle word itself (`shape_rule_ids` adds the jump and quasi-satellite rules)."""
    if f.rule == '3k2':
        return ['K2']
    if f.rule in SINGLE_LEAVES:  # the circulation a single-swing leaf judged (W28 C6)
        return ['G4b' if _net_shift(f, G4B_NET) else 'G4']
    if angle in ('libration', 'episodic'):
        kind = union_kind(f)
        ids = ['G2'] + (['G2u'] if kind in ('promoted', 'kept') else [])
        ids += ['G2c'] if kind == 'centre' or centre_scale_clause(f) else []
        return ids + (['G2a'] if _action_step_cut(f) else [])
    if angle == 'circulation':
        return ['G1c' if _held(f, G1C_FINAL) else ('G4b' if _net_shift(f, G4B_NET) else 'G4')]
    if angle == 'insufficient_time':
        ripple = ['F6r'] if ripple_waves_of(f) else []
        if insufficient_angle_rule(f):  # the angle rule as saved; below, its reconstruction for a word the tree set
            return ripple + [insufficient_angle_rule(f)]
        if slow_swing_decided(f):
            return ripple + ['G5w']
        return ripple + ['G5a' if _slow_arc(f) else 'G6']
    return [{'near_separatrix': 'G3', 'drift': 'G5'}[angle]] if angle in ('near_separatrix', 'drift') else []


G1C_FINAL = 'G1c circulation lasts beyond the last period'  # decide._g1c: the rule fired when this held
G1E_EARLIER = 'G1e librating episodes before the last jump of a'  # decide._g1e
G1L_LAST = 'G1l it lasts to the jump of a'  # decide._g1l: the libration on both sides of a jump stays one (no G1l)


def _action_step_cut(f: MMRFacts) -> bool:
    """A step of the libration action (G2a) cut the record into regimes: the cutting jumps are all action steps (the
    predicate of `_regimes_clause`)."""
    g = f.a.geometry
    if g is None or not g.episodes_before_jump:
        return False
    cuts = [jump for jump in g.jumps if not jump.kick]
    return bool(cuts) and all(jump.action_step for jump in cuts)


def _step_ids(f: MMRFacts, step: Step) -> Tuple[str, ...]:
    """The rule and check ids the answer of a step reads (each in `verdict_vocab.RULE_NAMES`)."""
    rule = f.rule
    if step.key == 'shape':
        return shape_rule_ids(f)
    if step.key == 'kin':
        if rule == '3k2':
            return ('K2',)
        if rule in ('3', '3b5'):
            b1 = f.entry(B1_AMPLITUDE)
            first = ('B1',) if b1 is not None and b1.result is False else ('K1',)
            return first + (('B5',) if rule == '3b5' else ())
        return ('B1',)
    if step.key == 'force':
        extra = {'5s': ('F',), '5s*': ('F',), '5o': ('O1',), '5x': ('F',)}.get(rule.split('/')[0], ())
        return ('L4', 'R0') + extra
    return {'sub': ('S',), 'owner': ('W',), 'zone': ('Z',)}.get(step.key, ())


def _step_number(f: MMRFacts, step: Step) -> str:
    """The step's number; on the branch the leaf took, with its letter: '5d' (rule 5d/6), '6u', '3k2', '5a*'."""
    number = STEP_QUESTIONS[step.key][0]
    parts = f.rule.split('/') if f.rule else []
    for part in parts:
        if part.startswith(number) and step.state in (StepState.LEAF, StepState.PASSED):
            return part
    return number


def no_force_on_a(f: MMRFacts) -> bool:
    """The leaf is kinematic (step 3) and a does not carry the swing (B1 not held): the decision's own predicate, so no
    resonance, σ's or another, acts on a (card g04-7)."""
    return f.rule in KINEMATIC_LEAVES and not _held(f, B1_AMPLITUDE)


def step_not_needed(f: MMRFacts, step: Step) -> str:
    """Why a step that did not run is not needed ('' when it is a gap the reader should know of): the owner search at a
    kinematic leaf (`no_force_on_a`) fits a force on a that is not there (card g04-7: 41588 3J+9S-2, too few force
    windows is no gap then)."""
    if step.key != 'owner' or step.state != StepState.NOT_RUN or not no_force_on_a(f):
        return ''
    return f'n/a: {NO_FORCE_ON_A}; not needed: step 3 decides without it'


def collapse_after_leaf(steps: Sequence[Step]) -> List[Step]:
    """The steps after the leaf as one pale line ('4–6 · drift, own force, behavior: not reached'), no outcome text."""
    leaf = next((i for i, step in enumerate(steps) if step.state == StepState.LEAF), None)
    if leaf is None:
        return list(steps)
    out, rest = list(steps[: leaf + 1]), [step for step in steps[leaf + 1 :] if step.state == StepState.NOT_REACHED]
    if not rest:
        return out
    first, last = STEP_QUESTIONS[rest[0].key][0], STEP_QUESTIONS[rest[-1].key][0]
    number = first if first == last else f'{first}–{last}'
    names = ', '.join(STEP_SHORT[step.key] for step in rest)
    return out + [Step('rest', f'{names}: not reached', StepState.NOT_REACHED, number=number)]


def _ask(f: MMRFacts, step: Step) -> Step:
    """The step with its number, question, why (only when it ran) and ids; a step that did not run and is not needed
    says so (`step_not_needed`)."""
    not_needed = step_not_needed(f, step)
    if not_needed:
        step = replace(step, state=StepState.NOT_NEEDED, outcome=not_needed)
    if step.key not in STEP_QUESTIONS:  # a leaf of the circulation branch
        question = CIRCULATION_QUESTIONS.get(f.rule, step.title)
        return replace(step, number=f.rule, question=question, ids=(f.rule,) if f.rule in ('C5', 'R0') + SINGLE_LEAVES else ())
    _, question, why = STEP_QUESTIONS[step.key]
    ran = step.state in (StepState.LEAF, StepState.PASSED)
    return replace(step, number=_step_number(f, step), question=question, why=why if ran else '', ids=_step_ids(f, step))


def mmr_steps(f: MMRFacts) -> List[Step]:
    """The steps of the status tree in their fixed order, each a question with what it found; the steps after the leaf
    are 'not reached', collapsed into one line."""
    if f.rule in SINGLE_LEAVES and single_swing_clause(f):  # W28 C6: a circulation decided on one swing
        return collapse_after_leaf([_ask(f, step) for step in _single_swing_steps(f)])
    steps = [_shape_step(f)]
    if steps[0].state != StepState.LEAF:
        if str(f.v.angle_status) == 'circulation':
            steps += _circulation_steps(f)
        else:
            reached = False
            for build in TREE_STEPS:
                step = build(f)
                if reached:
                    step = Step(step.key, step.title, StepState.NOT_REACHED)
                steps.append(step)
                reached = reached or step.state == StepState.LEAF
    return collapse_after_leaf([_ask(f, step) for step in steps])


# --- the parts of an accepted 1:1 episode in words (W28 C13, cards g03-1 A, g03-3 C, g03-8 A, g03-9 A), read from its
# episodes.csv row (C7 `marks`, C8 `subregimes` and `residence_*`): the measured core (its first to last turning point),
# the sub-regimes between the changes that are no boundary, the amplitude changes and kicks of a near them, and the
# inferred residence around the core. Words only: nothing here feeds a status.

# a well of `geometry.Episode.wells` / `SubRegime.well` (coorbital_geometry's names) in words
WELL_WORDS = {'qs': 'quasi-satellite', 'tadpole_L4': 'tadpole L4', 'tadpole_L5': 'tadpole L5', 'horseshoe': 'horseshoe'}


def well_words(well: Any) -> str:
    """'quasi-satellite', 'tadpole L4', 'compound', ...; '' for an unnamed well ('-', '')."""
    name = _text(well).strip()
    if name in ('', '-', 'nan'):
        return ''
    return WELL_WORDS.get(name, snake_words(name))


def _timed_items(value: Any) -> List[Tuple[str, float, float]]:
    """'name@t' / 'name@t0:t1' items, ';'-separated (episodes.csv `marks`, `subregimes`) as (name, t0, t1) [yr]."""
    text = _text(value)
    out = []
    for item in text.split(';') if text and text != 'nan' else ():
        name, _, when = item.partition('@')
        start, _, end = when.partition(':')
        try:
            t0 = float(start)
            out.append((name.strip(), t0, float(end) if end else t0))
        except ValueError:
            continue
    return out


def episode_marks(row: Mapping[str, Any]) -> List[Tuple[str, float, float]]:
    """The changes inside an accepted 1:1 run that are no boundary (episodes.csv `marks`, rule coorbital_well_cut):
    ('well_change', t, t), ('amplitude_change', t0, t1), ('kick', t, t) [yr]; ('unheld', t0, t1), the stretch of an
    episode beyond its well-holding core (rule coorbital_well_core)."""
    return _timed_items(row.get('marks'))


def residence_sides(row: Mapping[str, Any]) -> Tuple[bool, bool]:
    """Whether the inferred residence (W28 C8) reaches before and after the measured core (its first and last turning
    points)."""
    r0, r1 = row.get('residence_t0_yr'), row.get('residence_t1_yr')
    c0 = row.get('core_t0_yr') if finite(row.get('core_t0_yr')) else row.get('t_start_yr')
    c1 = row.get('core_t1_yr') if finite(row.get('core_t1_yr')) else row.get('t_end_yr')
    if not all(finite(v) for v in (r0, r1, c0, c1)):
        return False, False
    return float(r0) < float(c0), float(r1) > float(c1)


def forced_wave_source(f: MMRFacts) -> str:
    """'a forced wave from driver 3N-5+0+2' when the verdict names the circulating argument that drives a (A3a's role
    driver; card g04-5 A: 385266, never 'capture'), 'a forced wave of another argument' otherwise."""
    if str(f.v.cause_role) == 'driver' and f.v.owner:
        return f'a forced wave from driver {f.v.owner}'
    return 'a forced wave of another argument'


def decisive_centre_moves(f: MMRFacts) -> bool:
    """`orbit_type.centre_moves` of the deciding episode: the one test of the header's ', moving center' and of the
    words 'fixed center' / 'the center is stationary' here (W23 F5)."""
    shift = (f.episodes.decisive or {}).get('centre_shift_over_amp')
    return finite(shift) and centre_moves(float(shift), f.s.mmr_l6_shift_full)


# the leaves of the branch where the force law and the potential with a are both not measured (decide.unmeasured_branch)
BRANCH_LEAVES = ('5s', '5s*', '5o', '5x', '5n')
STRENGTH_LEAVES = ('5s', '5s*')  # leaves of the strength that a circulation word can carry (the 5s veto, W29 §2)


def _entry_text(f: MMRFacts, name: str, unit: str = '') -> str:
    """'value op threshold' of a decision-path entry as it came out ('' when the entry is not on the path)."""
    return entry_compare(f.entry(name), unit)


def strength_first(f: MMRFacts) -> bool:
    """Rule 5s decided before the force law and the potential were read (decide._implausible_strength_first) while
    one of them was measured (failed): with both not measured the branch's wording holds as before."""
    measured = any(f.checks.get(key) in ('pass', 'fail') for key in ('a_response', 'potential'))
    return f.rule == '5s' and measured and any(name.startswith(STRENGTH_FIRST) for name in f.path)


def _strength_entry_object(f: MMRFacts, name: str):
    """The 5s ledger entry `name` from either place of the rule (before step 5 or in the branch), None without one."""
    return f.entry(f'{STRENGTH_FIRST} {name}') or f.entry(f'5s {name}')


def times_scale(entry, scale: str) -> str:
    """A required/theory ratio of the ledger as a multiple of its scale with the bar named apart: '9.63e35 × the
    directly averaged strength (×1 000 or more is implausible)', never '9.63e35 ≥ 1 000 × the strength', which reads
    as the factor (review 2026-11-05 #71); '' without the entry."""
    if entry is None or not finite(entry.value):
        return ''
    op = str(entry.op)
    bar = f'×{fmt(entry.threshold)} or more is implausible' if op in ('>=', '>') else f'need {OPS.get(op, op)} {fmt(entry.threshold)}'
    return f'{fmt(entry.value)} × {scale} ({bar})'


def _implausible_clause(f: MMRFacts) -> str:
    """5s: the ratio (two-body) or the ratio and the residual after the small-divisor envelope (three-body)."""
    strength = f.a.strength
    direct = _strength_entry_object(f, STRENGTH_DIRECT)
    if direct is not None:
        if strength is not None and str(strength.method) == 'direct_harmonic':
            return f'cycles of this period would need {times_scale(direct, "the strength of the resonant harmonic (extended precision)")}'
        return f'cycles of this period would need {times_scale(direct, "the directly averaged strength")}'
    ratio = times_scale(_strength_entry_object(f, STRENGTH_THREE_BODY), 'the scale μ_i μ_j e^q') or '— × the scale μ_i μ_j e^q'
    residual = _strength_entry_object(f, STRENGTH_DIVISORS)
    b = f'B = n/|Δ| {fmt(strength.divisor_b)}' if strength is not None else 'B'
    if strength is not None and strength.divisor_capped:  # W22 R-DIV: the candidate turns slower than σ swings
        b += f', capped at n·min(P, P_cyc)/2π because the nearest two-body angle {strength.divisor_candidate} turns slower than σ swings'
    still = fmt(residual.value) if residual is not None else '—'
    return f'cycles of this period would need {ratio}; even with small divisors up to B² ({b}) still {still} × μ_i μ_j e^q B²'


def _divisor_open_clause(f: MMRFacts) -> str:
    """5s*: why the small divisors may close the gap."""
    strength = f.a.strength
    ratio = times_scale(f.entry(f'5s {STRENGTH_THREE_BODY}'), 'μ_i μ_j e^q') or '— × μ_i μ_j e^q'
    entry = f.entry(f'5s {STRENGTH_DIVISORS}')
    residual = fmt(entry.value) if entry is not None else '—'
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
        return f'cycles of this period would need {ratio}, but {why}: the envelope cannot rule out the gap closing'
    else:
        why = f'with small divisors up to B² it needs only {residual} × μ_i μ_j e^q B²'
    return f'cycles of this period would need {ratio}, but {why}: the gap may close'


def _objection_clause(f: MMRFacts) -> str:
    """5o: the measured objections (O1, F) that failed."""
    parts = []
    onset = f.entry(O1_ONSET)
    if onset is not None and onset.result is False:
        # O1's background skips the earlier episodes of its group (`response.onset_group`: ids, latest first)
        group = [float(i) for i in _text((f.episodes.decisive or {}).get('onset_group')).split(';') if i.strip()]
        group = [int(i) for i in group if finite(i)]
        where = f', the wave before episode {group[-1]}, which no circulation separates from it' if group else ''
        parts.append(
            f'before the episode a already oscillated with comparable strength (before/inside {_entry_text(f, onset.name)}{where})'
        )
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
    """5n: one independent cycle; nothing measured contradicts the libration, nothing confirms it (W28 B9)."""
    cycles = _entry_text(f, INDEPENDENT_CYCLES)
    return f'the episode holds one center only (independent cycles {cycles}): nothing to compare, so nothing is measured, not contradicted'


def _scale_clause(f: MMRFacts) -> str:
    """5x (W28 B9, card g08-6 A; `decide._scale_out_of_range`): the strength is known only as an order-of-magnitude
    scale (μ e^q, μ_i μ_j e^q for three bodies) and lies out of its range; a diagnostic, not a measured objection."""
    entries = [entry for name, entry in f.path.items() if name.startswith(SCALE_OUT_OF_RANGE) and entry.value is not None]
    failed = next((entry for entry in entries if entry.result is False), entries[-1] if entries else None)
    x = f.s.mmr_strength_check_ratio
    scale = 'μ_i μ_j e^q' if f.a.resonance is not None and len(f.a.resonance.planets) > 1 else 'μ e^q'
    ratio = (
        f'{fmt(failed.value)} × {scale}, outside [{fmt(1 / x)}, {fmt(x)}]'
        if failed is not None
        else f'a ratio outside [{fmt(1 / x)}, {fmt(x)}]'
    )
    return (
        f'the strength is known only as the order-of-magnitude scale {scale}, and the period needs {ratio}: '
        'a diagnostic, not a measured objection, so the origin is not established'
    )


BRANCH_CLAUSES = {
    '5s': _implausible_clause,
    '5s*': _divisor_open_clause,
    '5o': _objection_clause,
    '5n': _one_cycle_clause,
    '5x': _scale_clause,
    '5u/6': _geometry_only_clause,
}


def branch_clause(f: MMRFacts) -> str:
    """What decided the branch where neither the force law nor the potential with a is measured (rules 5s, 5s*, 5o,
    5n, 5u/6 of `decide.unmeasured_branch`), with the numbers of its decision-path entries."""
    build = BRANCH_CLAUSES.get(f.rule)
    return build(f) if build else ''


FORCE_WINDOWS_POINTER = 'too few force windows: step 5'


def _why_l4(f: MMRFacts) -> str:
    """Why the force law was not measured; in the slotted text (`MMRFacts.brief`) the window arithmetic, which step 5
    prints, is a pointer there (no number twice)."""
    why = why_not_measured(f, 'L4')[1]
    if getattr(f, 'brief', False):
        windows = force_windows_text(f)
        return why.replace(windows, FORCE_WINDOWS_POINTER) if windows in why else why
    return why


def _potential_clause(f: MMRFacts) -> str:
    """What the potential (R0) said, for `origin_unknown_clauses`."""
    s, energy, r0 = f.s, f.a.energy, f.checks.get('potential')
    if r0 == 'fail' and energy is not None and str(energy.sigma_only) == 'fail':
        what = []
        if finite(energy.period_law_med) and abs(energy.period_law_med - 1) > s.separatrix_period_tol:
            what.append(
                f'the cycle periods do not follow the pendulum law (T_obs/T_pend {fmt(energy.period_law_med)}, '
                f'need 1 ± {fmt(s.separatrix_period_tol)})'
            )
        if finite(energy.well_coherence) and energy.well_coherence < s.separatrix_phase_coherence:
            what.append(f'the well moves (coherence {compare(energy.well_coherence, ">=", s.separatrix_phase_coherence)})')
        return 'the energy does not fit a pendulum potential' + (': ' + ' and '.join(what) if what else '')
    if r0 == 'fail' and energy is not None and finite(energy.ve_vf_agree_fraction) and energy.ve_vf_agree_fraction == 0:
        return f'the energy does not fit the potential from a in any window ({_potential_detail(f)[0]})'
    if r0 == 'fail':
        return f'the energy disagrees with the potential from a ({_potential_detail(f)[0]})'
    if r0 == 'pass':
        return f'the energy fits the potential from a ({_potential_detail(f)[0]})'
    if tree_read_potential_fail(f):
        return (
            f'the energy does not fit the potential from a ({_potential_detail(f)[0]}); the filter guard emptied the '
            'published check afterwards, a failed check stays an objection'
        )
    return f'the potential was not measured ({why_not_measured(f, "R0")[1]})'


def _force_clause(f: MMRFacts) -> str:
    """What the force law (L4) said, for `origin_unknown_clauses`."""
    l4 = f.checks.get('a_response')
    if l4 == 'fail':
        return f'ȧ does not follow the force law ({_force_detail(f)[0]})'
    if l4 == 'pass':
        return f'ȧ follows the force law ({_force_detail(f)[0]})'
    if one_force_window(f):  # g10-8 (137295): one window has no independent half to check it with
        return f'the force law has only one window, so nothing independent checks it ({_why_l4(f)})'
    return f'the force law cannot be judged ({_why_l4(f)})'


def origin_unknown_clauses(f: MMRFacts) -> str:
    """5a/5a*/5b/5e/5f/5g: what each of the force law and the potential said, one clause each (5g: and the failed
    identity that decides)."""
    s, response = f.s, f.a.response
    clauses = [_potential_clause(f), _force_clause(f)]
    if f.rule == '5g' and response is not None:
        clauses.append(
            f'σ̇ ≠ c (a − a*): identity slope/c {fmt(response.identity_slope_ratio)}, corr {fmt(response.identity_corr)} '
            f'(need |slope/c − 1| ≤ {fmt(s.mmr_identity_slope_tol)}, corr ≥ {fmt(s.mmr_identity_corr_min)})'
        )
    return '; '.join(clauses)


def _k2_clause(f: MMRFacts) -> str:
    """K2 (rule 3k2) in words: the slow center follows the secular part, the episodes are a ripple on it; or K2b, a
    kinematic libration whose center moves more than its amplitude."""
    shift = _held(f, K2B_SHIFT)
    if shift:
        return (
            f'the oscillation sits in the perihelion term and its center moves {compare(shift.value, ">", shift.threshold)} × '
            'its amplitude: a kinematic drift of the center, not a libration'
        )
    corr, share = f.entry(K2_CORR), f.entry(K2_SHARE)
    return (
        f'the slow center drifts with the perihelion term (corr {compare(corr.value if corr else None, ">=", f.s.mmr_kin_corr_min)}) '
        f'while a does not answer σ, and the episodes are a ripple on that drift '
        f'(swing / center range {compare(share.value if share else None, "<", f.s.mmr_kin_ripple_share_max)})'
    )


def _unexplained_lifted(f: MMRFacts) -> bool:
    """Whether the mark exit_and_capture took the cap transient_unexplained off this verdict (`sep_unexplained_lifted`,
    recorded by the analysis)."""
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
            f'exit and capture: σ left a libration {plural(f.a.energy.n_exits, "time")} and came back '
            f'{plural(f.a.energy.n_captures, "time")} (a mark; no status, no mechanism'
            + ('; lifts the cap "' + CAP_WORDS[str(Cap.TRANSIENT_UNEXPLAINED)] + '")' if lifted else ')')
        )
    spread = f.entry(SHORT_PERIOD_CAP)
    if spread is not None and spread.result is False and not has_mark(f.v, 'short_period_aliased'):
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
    # below 1e-3 always with an exponent: '2.56e-4', never '0.000256' beside '7.6e-5' in one range (review #105)
    return _exp_form(value, DIGITS)


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
    harmonic = str(strength.method) == 'direct_harmonic'
    if p.has('str_P_lib_theory_min_kyr', 'str_P_lib_theory_max_kyr'):
        window = f'{p.time("str_snap_t0_yr")}–{p.time("str_snap_t1_yr")}' if p.has('str_snap_t0_yr', 'str_snap_t1_yr') else ''
        n = int(strength.n_evaluations)
        well = f', {strength.n_consistent} of {n} in the well' if not harmonic else ''
        e = f', ⟨e⟩ {p.span("str_e_snap_min", "str_e_snap_max")}' if p.has('str_e_snap_min', 'str_e_snap_max') else ''
        # one unit for the pair compared (review #105: '1.39–1.4 kyr against 960 yr'): yr when any of them is below 1 kyr
        values = [abs(p.value('str_P_lib_theory_min_kyr')) * KYR] + ([abs(p.value('str_T_obs_yr'))] if p.has('str_T_obs_yr') else [])
        scale, unit = (1 / KYR, 'kyr') if min(values) >= KYR else (1.0, 'yr')
        theory = f'T0_th {p.span("str_P_lib_theory_min_kyr", "str_P_lib_theory_max_kyr", KYR * scale, unit=f" {unit}")}'
        where = f' ({window}: {plural(n, "snapshot")}{well}{e})' if window else ''
        observed = ''
        if p.has('str_T_obs_yr'):
            # W28 E11 (g10-7): str_T_obs is T0 = P/F(A) inferred from the observed cycles, not an observed period
            source = _t_obs_source(_text(f.row.get('str_T_obs_source')))
            observed = f' against T0 {p.num("str_T_obs_yr", scale)} {unit} from the observed cycles ({source})'
        p.show('str_T_obs_source', 'str_n_evaluations', 'str_n_consistent')
        rows.append(p.take('period', icon, 'own period', theory + where + observed))
    if p.has('str_mu_eff_req', 'str_ratio_all_min', 'str_ratio_all_max'):
        limit, implausible = s.mmr_strength_check_ratio, s.mmr_strength_implausible_ratio
        low, high = p.value('str_ratio_all_min'), p.value('str_ratio_all_max')
        inside = high >= 1 / limit and low <= limit
        theory = 'μ_th = μ_P a 2|C| (planet mass × a × twice the term’s coefficient)' if harmonic else 'μ_th'
        text = (
            f'μ_req {p.num("str_mu_eff_req", form=sci)} against {theory} '
            f'{p.span("str_mu_eff_theory_all_min", "str_mu_eff_theory_all_max", form=sci)}: '
            f'μ_req/μ_th = (T0_th/T_obs)² = {p.span("str_ratio_all_min", "str_ratio_all_max", form=sci)} '
            f'{"overlaps" if inside else "lies outside"} [{fmt(1 / limit)}, {fmt(limit)}]'
        )
        if low >= implausible:
            text += f'; ≥ {fmt(implausible)} × is implausible'
        p.show('str_ratio', 'str_ratio_min', 'str_ratio_max', 'str_implausible', 'str_mu_eff_theory_min', 'str_mu_eff_theory_max')
        rows.append(p.take('strength', icon, 'strength', text))
    if p.has('str_sep_halfwidth_min_au', 'str_sep_halfwidth_max_au', 'rsp_b1_meas_au'):
        # a pendulum's a swings at most the zone's half-width; the width goes as √μ, so the strength check's band [1/x, x]
        # on μ is √x on the width: the same tolerance, no new number
        wave, width = p.value('rsp_b1_meas_au'), p.value('str_sep_halfwidth_max_au')
        ratio, tolerance = wave / width if width > 0 else float('nan'), math.sqrt(s.mmr_strength_check_ratio)
        fits = holds(ratio, '<=', tolerance)
        limit = s.mmr_strength_check_ratio
        text = (
            f'a swings ±{p.num("rsp_b1_meas_au", form=sci)} au against the theoretical half-width 2ω0/|c| '
            f'{p.span("str_sep_halfwidth_min_au", "str_sep_halfwidth_max_au", form=sci, unit=" au")}: ratio '
            f'{compare(ratio, "<=", tolerance)} (the width goes as √μ, so the strength band [{fmt(1 / limit)}, {fmt(limit)}] '
            'allows its square root)' + ('' if fits else ': a libration of this force cannot move a so far')
        )
        rows.append(p.take('width', _check_icon(fits) if fits is not None else PHYSICS_ICONS[None], 'width in a', text))
    # the slow part is read only where B5 grades (a non_resonant drift or kinematic verdict, decide's margins) or the
    # kinematic test read the slow ȧ: on a libration its numbers judge nothing (review #102: 1921's −0.00112)
    graded = 'B5' in f.margins
    slow_read = f.entry('3 kinematic: slow a_dot answers sin sigma') is not None
    if (graded or slow_read) and (p.has('rsp_slow_adot_R2') or p.has('rsp_b5_ratio')):
        parts = []
        if p.has('rsp_slow_adot_R2'):
            parts.append(f'slow ȧ on sin σ_slow R² {p.num("rsp_slow_adot_R2")}')
        if graded and p.has('rsp_b5_ratio'):
            excluded = p.value('rsp_b5_ratio') >= 1
            ratio = p.cmp('rsp_b5_ratio', '>=', 1)
            parts.append(f'slower libration: ratio {ratio}, {"excluded" if excluded else "not excluded"}')
        rows.append(p.take('slow', '', 'slow part', '; '.join(parts)))
    elements = 'the planet’s saved orbit' if str(strength.planet_elements) == 'saved' else 'the planet’s mean orbit, its phase scanned'
    p.show('str_planet_elements', 'str_method')
    rows.append(
        p.take(
            'legality',
            '',
            'assumptions',
            f'μ_req is what σ would need if these cycles were its own libration; the range spans the snapshots, not an '
            f'error bar; first-order average over a frozen 3-D orbit with {elements}',
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
    return [p.take('scale', _check_icon(strength.check), 'strength', text)]


def own_pendulum_section(f: MMRFacts) -> Optional[PhysicsSection]:
    """Section A, 'Is σ the own pendulum of <key>?': can these cycles be σ's own libration (the strength its period
    needs against the averaged potential's, its width in a); None when nothing was measured for it."""
    key, strength = f.a.resonance.key, f.a.strength
    title = f'Is σ the own pendulum of {key}?'
    if f.coorbital:
        return PhysicsSection(title, '1:1', [coorbital_law_row(f)])
    if strength is None:
        return None
    reason = _text(strength.not_measured_reason)
    if str(strength.method) == 'eq_scale':
        rows = _scale_rows(f)
        return PhysicsSection(title, 'order-of-magnitude scale', rows) if rows else None
    if reason and reason != 'centre_outside_well':
        if reason == 'no_required_strength' and str(f.v.angle_status) not in ('libration', 'episodic'):
            return None
        words = STRENGTH_REASON_WORDS.get(reason, snake_words(reason))
        return PhysicsSection(
            title, 'not measured', [PhysicsRow('strength', PHYSICS_ICONS[None], 'strength', words, columns=('str_not_measured_reason',))]
        )
    subtitle = (
        'resonant harmonic computed in extended precision (double precision resolves no well)'
        if str(strength.method) == 'direct_harmonic'
        else 'directly averaged potential'
    )
    if reason == 'centre_outside_well':
        subtitle += '; no observed swing librates in its well'
    return PhysicsSection(title, subtitle, _direct_rows(f))


def coorbital_law_row(f: MMRFacts) -> PhysicsRow:
    """The 1:1 force law as Evidence (review #104): ȧ against the averaged direct + indirect potential of the planet,
    its scale K/K_th (1 = the planet's own mass) and the R² of both halves, the icon from check_mmr_a_response (as the
    tree read it); the numbers are rule L4's (`_force_detail`)."""
    response = f.a.response
    law = response.coorbital if response is not None else None
    check = f.checks.get('a_response', '')
    planet = f.a.resonance.planets[0] if f.a.resonance is not None and f.a.resonance.planets else 'the planet'
    if law is None:
        return PhysicsRow('law', PHYSICS_ICONS[None], 'force law', 'the 1:1 force law is not measured')
    verb = {'pass': 'follows', 'fail': 'does not follow'}.get(check, 'is not judged against')
    text = (
        f'ȧ {verb} the 1:1 force law: K/K_th {fmt(law.scale)} (1 = {planet}’s mass; averaged direct + indirect '
        f'potential), R² {fmt(response.force_r2_h1)}/{fmt(response.force_r2_h2)} in the two halves'
    )
    return PhysicsRow('law', _check_icon(check), 'force law', text)


# own_intervals' regime -> what the owner does there (owner.py REGIME_*)
OWNER_REGIME_WORDS = {'libration': 'librating', 'forced_wave': 'a forced wave', 'circulation': 'circulating'}


def _intervals(text: str) -> str:
    """own_intervals 't0-t1:regime;…' [yr] -> 'librating during 0.455–11.6 kyr, 11.6–12.4 kyr' when every block has one
    regime, else 'during 9.06–88.1 kyr as a forced wave, …' (LIST_MAX, then 'N more'); '' without a block."""
    blocks = []
    for item in [part for part in text.split(';') if part]:
        span, _, kind = item.partition(':')
        start, _, end = span.partition('-')
        if finite(start) and finite(end):
            blocks.append((interval(float(start), float(end)), OWNER_REGIME_WORDS.get(kind, snake_words(kind))))
    if not blocks:
        return ''
    if len({kind for _, kind in blocks}) == 1:
        return f'{blocks[0][1]} during {_listed([span for span, _ in blocks])}'
    return 'during ' + _listed([f'{span} as {kind}' for span, kind in blocks])


# own_route (owner.py GAIN_*): how the owner was accepted over σ's own term
OWNER_ROUTE_WORDS = {
    'r2': 'accepted because its force law fits ȧ better than σ’s own',
    'phase': 'accepted by its phases, which σ’s own term could not hold',
    'potential': 'accepted because its averaged potential predicts ȧ',
    'identity': 'accepted because φ librates and its strength passes (the identity route)',
}


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
            f'; it covers {p.cmp("own_cover", ">=", s.mmr_owner_cover_min)} of σ’s deciding episode, '
            f'{p.time("own_decisive_t0_yr")}–{p.time("own_decisive_t1_yr")}'
        )
    route = _text(f.row.get('own_route'))
    where = _intervals(_text(f.row.get('own_intervals'))) or 'over the record'
    p.show('own_intervals', 'own_route', 'own_name', 'own_best', 'own_detail', 'own_regime')
    inside = not finite(f.row.get('own_cover')) or p.value('own_cover') >= s.mmr_owner_cover_min
    accepted = f', {OWNER_ROUTE_WORDS.get(route, f"accepted by its {route} route")}' if route else ''
    rows.append(p.take('owner', _check_icon(inside), 'owner', f'{key}, {where}{accepted}{cover}'))
    if _text(f.row.get('own_identity_text')):
        p.show('own_identity_text', 'own_psi_def')
        text = f'σ = {f.row["own_identity_text"]}: both angles hold the same λ, so a wave of λ shows in both (one fact, not two)'
        rows.append(p.take('identity', '', 'identity', text))
    rows += _identity_route_rows(f, p)  # empty unless the route 'identity' accepted the owner
    if forced and p.has('own_rate_rad_kyr', 'own_owner_period_yr'):
        # two measurements, not one derivation (review #64: 2π/1.55 is not 2 kyr): the rate is the slope over the search
        # window (OwnerCandidate.rate_rad_yr), the period the regime's own rate (owner_period_yr)
        text = (
            f'φ turns at {p.num("own_rate_rad_kyr")} rad/kyr over the search window; its period in the stretch that '
            f'drives a {p.time("own_owner_period_yr")}'
        )
        if p.has('own_freq_period_yr'):
            text += f'; the dominant a-wave {p.time("own_freq_period_yr")}'
        rows.append(p.take('period', PHYSICS_ICONS[True], 'period', text))
    if p.has('own_R2'):
        rows.append(_owner_force_row(f, p, route))
    if p.has('own_lever_owner', 'own_phi0_coherence', 'own_lever_gain'):
        gain = p.value('own_lever_gain')
        text = (
            f'the owner’s φ0 stays fixed: coherence C_B {p.num("own_phi0_coherence")} against the L_B '
            f'{p.num("own_lever_owner")} it could keep if σ drove a, gain {p.cmp("own_lever_gain", ">=", s.mmr_owner_phi0_coh_gain)}'
            + (f'; σ’s own bound L_σ {p.num("own_lever_sigma")}' if p.has('own_lever_sigma') else '')
        )
        if route == 'phase':
            rows.append(p.take('phases', _check_icon(gain >= s.mmr_owner_phi0_coh_gain), 'phases', text))
        else:  # accepted by another route: the phases are context
            rows.append(p.take('phases', '', 'phases', text))
    rows += _owner_theory_rows(f, p, forced)
    if p.has('own_sigma_wave_deg'):
        text = f'σ on sin/cos φ: {p.num("own_sigma_wave_deg")}°'
        if p.has('own_sigma_wave_R2'):
            text += f' (R² {p.num("own_sigma_wave_R2")})'
        rows.append(p.take('sigma_wave', '', 'σ wave', text + ', ripple-corrected; measured only, nothing predicts it here'))
    rows += _capture_rows(f, p)
    return rows


def _owner_force_row(f: MMRFacts, p: _Printer, route: str) -> PhysicsRow:
    """The owner's force law on a: R², φ0 coherence and σ's own law; context only for the route 'identity'."""
    s = f.s
    if route == 'identity':  # the route measures nothing on a
        text = f'ȧ = K sin(φ − φ0): R² {p.num("own_R2", form=lambda v: fmt(v, 2))} (not read by the identity route)'
        return p.take('force', '', 'force on a', text)
    text = (
        f'ȧ = K sin(φ − φ0): R² {p.cmp("own_R2", ">=", s.mmr_owner_r2_min, digits=2)}'
        + (f', φ0 coherence {p.num("own_phi0_coherence")}' if p.has('own_phi0_coherence') else '')
        + (f'; σ’s own law on the same windows: R² {p.num("own_sigma_R2", form=lambda v: fmt(v, 2))}' if p.has('own_sigma_R2') else '')
        + (_sigma_law_note(f, p) if p.has('own_sigma_R2') else '')
    )
    p.show('own_R2_windows')
    return p.take('force', _check_icon(p.value('own_R2') >= s.mmr_owner_r2_min), 'force on a', text)


def _sigma_law_note(f: MMRFacts, p: _Printer) -> str:
    """Why σ's own law does not win when its R² is the higher one (review #65, 1997 GK3: 0.99 against the owner's
    0.84): the R² route needs the owner ahead by MMR_OWNER_R2_GAIN, so the phase route decided (`owner.better_than_sigma`);
    below MMR_A_RESPONSE_MIN_AMP_DEG of half-amplitude sin σ ≈ σ, so σ's law fits any swing a carries. '' otherwise."""
    if _text(f.row.get('own_route')) != 'phase' or p.value('own_sigma_R2') < p.value('own_R2'):
        return ''
    note = f', higher: R² does not separate them, since the owner would need σ’s R² + {fmt(f.s.mmr_owner_r2_gain)}; the phases decide'
    amp = (f.episodes.decisive or {}).get('half_amp_med_deg')
    if finite(amp) and float(amp) < f.s.mmr_a_response_min_amp_deg:
        note += (
            f'; at σ’s half-amplitude {fmt(amp)}° < {fmt(f.s.mmr_a_response_min_amp_deg)}° sin σ ≈ σ, so σ’s law fits any '
            'swing a carries'
        )
    return note


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
        rows.append(p.take('identity_fit', _check_icon(ok), 'identity', text))
    if p.has('own_id_phi_cover'):
        status = _text(f.row.get('own_id_phi_angle_status')) or 'its angle'
        text = f'φ’s own analysis: {status}, its librating episodes cover {p.cmp("own_id_phi_cover", ">=", s.mmr_owner_cover_min)} of σ’s episode'
        p.show('own_id_phi_angle_status', 'own_id_phi_span_deg', 'own_id_n_pass', 'own_id_n_analysed')
        rows.append(p.take('identity_phi', _check_icon(p.value('own_id_phi_cover') >= s.mmr_owner_cover_min), 'φ librates', text))
    if p.has('own_id_phi_str_ratio'):
        text = f'φ’s directly averaged strength passes: required / theory {p.num("own_id_phi_str_ratio")}'
        if p.has('str_ratio'):
            ratio_min = sci(s.mmr_owner_identity_ratio_min)
            text += f'; σ needs {p.num("str_ratio", form=sci)} × μ_i μ_j e^q (the route opens at {ratio_min})'
        rows.append(p.take('identity_strength', PHYSICS_ICONS[True], 'φ strength', text))
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
        words = POTENTIAL_REASON_WORDS.get(reason, snake_words(reason))
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
    return p.take('theory', _check_icon(r2 >= s.mmr_owner_r2_min and inside), 'theory', text)


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


CAPTURE_LABEL = 'both resonances at once?'  # card g05-5 A: never 'second capture' / '2 captures' as a count
# own_second_capture -> words; 'confirmed' and 'one_cycle' are A2b's third outcome (card g05-8 B: x = Ψ/u closes
# cycles of its own well, off every secular period)
CAPTURE_WORDS = {
    'excluded': 'excluded',
    'unresolved': 'not ruled out',
    'confirmed': 'both resonances at once',
    'one_cycle': 'one cycle of x: not decided yet (extend)',
}


def capture_window(f: MMRFacts) -> str:
    """'3.44 kyr–81.6 kyr': the window x = Ψ/u was tested on (`SecondCapture.t_start_yr/t_end_yr`, the hull of the
    owner's blocks that meet σ's episodes), '' when unknown."""
    capture = getattr(f.a.owner, 'capture', None) if f.a.owner is not None else None
    t0, t1 = (getattr(capture, 't_start_yr', None), getattr(capture, 't_end_yr', None)) if capture is not None else (None, None)
    return _span(float(t0), float(t1)) if finite(t0) and finite(t1) else ''


def capture_hull(f: MMRFacts) -> str:
    """', over both stretches of the owner, including the 93.5 kyr between them' when the tested window of x spans several of the
    owner's measured blocks and the stretch between them (2001 HW7 150 kyr, W28 Codex S5: x winds over the gap too, so
    its sweep is not one inside either block); '' for one block or an unknown window."""
    owner = f.a.owner
    capture = getattr(owner, 'capture', None) if owner is not None else None
    t0, t1 = (getattr(capture, 't_start_yr', None), getattr(capture, 't_end_yr', None)) if capture is not None else (None, None)
    if not (finite(t0) and finite(t1)) or t1 <= t0:
        return ''
    blocks = sorted((max(a, t0), min(b, t1)) for a, b, _ in owner.intervals if b > t0 and a < t1)
    if len(blocks) < 2:
        return ''
    covered, end = 0.0, t0
    for a, b in blocks:
        a = max(a, end)
        if b > a:
            covered, end = covered + (b - a), b
    gap = (t1 - t0) - covered
    stretches = 'both stretches' if len(blocks) == 2 else f'the {len(blocks)} stretches'
    return f', over {stretches} of the owner, including the {years(gap)} between them' if gap > 0 else ''


def _capture_secular(f: MMRFacts, p: _Printer) -> str:
    """Route F3 (A2) in words, each part the way it went (the text never asserts what its numbers refute: HW7's ẍ
    follows the 2ω cycle, R² 0.88, but also sin x, R² 0.63 ≥ 0.2): 'ẍ follows the secular harmonics (R² 0.98 ≥ 0.7),
    not a well of its own (R² on sin x 0.001 < 0.2)' excludes; '' without the two R²."""
    s = f.s
    if not p.has('own_psi_secular_R2', 'own_psi_self_R2'):
        return ''
    secular = p.value('own_psi_secular_R2') >= s.mmr_owner_r2_min
    own = p.value('own_psi_self_R2') >= s.mmr_owner_capture_self_r2_max
    r2 = p.cmp('own_psi_secular_R2', '>=' if secular else '<', s.mmr_owner_r2_min)
    r2_own = p.cmp('own_psi_self_R2', '>=' if own else '<', s.mmr_owner_capture_self_r2_max)
    if secular and not own:
        return f'ẍ follows the secular harmonics (R² {r2}), not a well of its own (R² on sin x {r2_own})'
    if secular:
        return f'ẍ follows the secular harmonics (R² {r2}), but also a well of its own (R² on sin x {r2_own})'
    if not own:
        return f'ẍ follows neither the secular harmonics (R² {r2}) nor a well of its own (R² on sin x {r2_own})'
    return f'ẍ does not follow the secular harmonics (R² {r2}), but a well of its own (R² on sin x {r2_own})'


def _capture_rows(f: MMRFacts, p: _Printer) -> List[PhysicsRow]:
    """'Both resonances at once?' (card g05-5 A; R5 §1.7 and A2's route F3): is σ held by its own term too while the
    owner pins φ? x = Ψ/u over the tested window: excluded by the sweep and energy route (x sweeps ≥ 180° and follows
    no potential) or by F3 (ẍ follows the secular harmonics, not sin x), else not ruled out; then the slow swing of x
    at its level (`slow_x_line`)."""
    s = f.s
    verdict = _text(f.row.get('own_second_capture'))
    if not verdict:
        return []
    route = _text(f.row.get('own_second_capture_route'))
    psi = _text(f.row.get('own_psi_def')) or 'uσ − vφ'
    window = capture_window(f)
    hull = capture_hull(f)
    match = PSI_DEF.match(psi)
    u = match.group(1) if match is not None and int(match.group(1)) else 'u'  # Ψ = uσ ∓ vφ: x = Ψ/u (review #66)
    parts = [f'x = Ψ/{u} with Ψ = {psi}' + (f', tested over {window}{hull}' if window else '')]
    if p.has('own_psi_turns'):
        parts.append(f'{p.num("own_psi_turns")} revolutions of Ψ')
    if p.has('own_psi_sweep_deg'):
        over = ' over the whole window, not inside one stretch' if hull else ''
        parts.append(f'x moved {p.cmp("own_psi_sweep_deg", ">=", s.mmr_owner_capture_sweep_deg, "°")}{over}')
    if p.has('own_psi_energy_R2'):
        parts.append(f'energy fit R² {p.cmp("own_psi_energy_R2", "<", s.mmr_owner_r2_min)}')
    secular = _capture_secular(f, p)
    if secular and route != 'sweep_energy':
        parts.append(secular)
    cycles = _x_cycles_text(f, p)  # A2b: confirmed or one_cycle
    if cycles and verdict in ('confirmed', 'one_cycle'):
        parts.append(cycles)
    p.show('own_psi_sweep_deg', 'own_psi_energy_R2', 'own_second_capture', 'own_second_capture_route', 'own_adiabatic', 'own_psi_def')
    excluded = verdict == 'excluded'
    if len(parts) == 1:
        parts.append('not measurable on the owner’s stretches over σ’s episode')
    text = '; '.join(parts) + f' → {CAPTURE_WORDS.get(verdict, verdict)}'
    decided_icon = excluded or verdict == 'confirmed'
    rows = [p.take('capture', PHYSICS_ICONS[True] if decided_icon else PHYSICS_ICONS[None], CAPTURE_LABEL, text)]
    slow = slow_x_line(f, p)
    if slow:
        rows.append(p.take('slow_x', '', 'slow x', slow))
    return rows


def _x_cycles_text(f: MMRFacts, p: Optional[_Printer] = None) -> str:
    """'x closes 18 cycles of its own well (period 1.2 kyr, off every secular period by 0.12)' (A2b, columns
    own_x_cycles, own_x_cycle_period_yr, own_x_secular_offset), '' without the count."""
    p = p if p is not None else _Printer(f.row)
    if not p.has('own_x_cycles'):
        return ''
    n = int(p.value('own_x_cycles'))
    p.num('own_x_cycles')
    head = f'x closes {plural(n, "cycle")} of its own well'
    numbers = []
    if p.has('own_x_cycle_period_yr'):
        numbers.append(f'period {p.time("own_x_cycle_period_yr")}')
    if p.has('own_x_secular_offset'):
        numbers.append(f'off every secular period by {p.num("own_x_secular_offset")}')
    return head + (f' ({", ".join(numbers)})' if numbers else '')


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
        subtitle = f'{what} of {owner.accepted.spec.key} (owner search: {plural(owner.n_candidates, "candidate")})'
        if str(Flag.OWNER_OUTSIDE_EPISODE) in f.v.flags:
            subtitle += '; it acts outside σ’s episode, so it is not the origin'
        return PhysicsSection(title, subtitle, _owner_rows(f))
    state, why = _skip_owner(f)
    if str(owner.check) not in ('pass', 'fail') and no_force_on_a(f):  # g04-7: no gap at a kinematic leaf
        text = f'n/a: {NO_FORCE_ON_A} (step 3), so no resonance drives it; the search itself: {why}'
        return PhysicsSection(title, 'owner search not needed', [PhysicsRow('owner', PHYSICS_NA, 'owner', text, columns=('own_flags',))])
    if str(owner.check) not in ('pass', 'fail'):
        icon = PHYSICS_NA if state == CheckState.SKIPPED else PHYSICS_ICONS[None]
        return PhysicsSection(title, 'owner search without a verdict', [PhysicsRow('owner', icon, 'owner', why, columns=('own_flags',))])
    best = _best_candidate(owner.candidates)
    p = _Printer(f.row)
    text = f'{plural(owner.n_candidates, "candidate")}, none accepted'
    if best is not None:
        text += (
            f'; best {best.spec.key} R² {compare(best.r2_median, "<", f.s.mmr_w_r2_none, digits=2)}'
            if best.r2_median < f.s.mmr_w_r2_none
            else f'; best {best.spec.key} {_owner_rejection(f, best)}'  # its reason in words, not 'r2 low'
        )
    p.show('own_best', 'own_R2', 'own_n_candidates')
    return PhysicsSection(title, 'no other resonance drives a', [p.take('owner', PHYSICS_ICONS[False], 'owner', text)])


def physics_sections(f: MMRFacts) -> List[PhysicsSection]:
    """The physics block of the right column (R1 §3–4, R5 §5): A 'Is σ the own pendulum of <key>?' and B 'What makes
    the cycles?'. Every number comes from a column of `f.row`, every threshold from the settings; display only."""
    return [section for section in (own_pendulum_section(f), cycles_section(f)) if section is not None and section.rows]


# --- the slotted explanation (cards g10-2 style (i), g10-8, g05-1, g04-7; W28 E8)


def one_force_window(f: MMRFacts) -> bool:
    """The force law was cut into exactly one window while the test needs more: no independent half checks it."""
    response = f.a.response
    return response is not None and len(response.windows) == 1 and f.s.mmr_force_min_windows > 1


def _decisive_cover(f: MMRFacts):
    """The owner's F7 entry on σ's decisive episode (`OwnerResult.episode_covers`: that episode, else the analysis
    window), or None: the measured cover and P2 turns the decision read (A1)."""
    owner = f.a.owner
    covers = list(getattr(owner, 'episode_covers', ()) or ())
    decisive = f.a.decisive_episode_id
    found = next((c for c in covers if decisive is not None and _same_id(c.episode_id, decisive)), None)
    return found or next((c for c in covers if c.episode_id is None), None)


def owner_confirmed(f: MMRFacts) -> bool:
    """The consistency guard of an affirmative cause (card g10-8): the verdict names an owner or driver that the
    decision kept, no F7/P2 refusal is on record, and its measured regimes (A1: no slip bridge) cover σ's decisive
    episode by at least MMR_OWNER_COVER_MIN with P2 holding (`EpisodeCover.induced`, the decision's own predicate).
    Otherwise the text says 'candidate'."""
    owner, s = f.a.owner, f.s
    if str(f.v.cause_role) not in ('owner', 'driver') or owner is None or owner.accepted is None:
        return False
    if refused_owner(f):
        return False
    cover = _decisive_cover(f)
    if cover is not None:
        return cover.induced(s.mmr_owner_cover_min, s.mmr_owner_psi_turns_per_cycle_max)
    return finite(owner.cover) and owner.cover >= s.mmr_owner_cover_min


def neighbour_items(neighbours: str) -> List[Tuple[str, float, float, str]]:
    """(key, t0, t1, where) [yr] of the verdict's neighbors 'KEY t0–t1 before|during|after;…' (A3a); an item without
    an interval gives (key, nan, nan, '')."""
    items = []
    for item in [part for part in _text(neighbours).split(';') if part.strip()]:
        name, _, rest = item.strip().partition(' ')
        span, _, where = rest.partition(' ')
        start, _, end = span.partition('–')
        if finite(start) and finite(end):
            items.append((name, float(start), float(end), where.strip()))
        else:
            items.append((name, float('nan'), float('nan'), ''))
    return items


def _refusal(f: MMRFacts) -> str:
    """Why the decision did not take the owner: its cover of σ's decisive episode (F7) and the P2 turns."""
    s, cover, parts = f.s, _decisive_cover(f), []
    value = cover.cover if cover is not None else getattr(f.a.owner, 'cover', None)
    if finite(value):
        parts.append(f'it covers {compare(value, ">=", s.mmr_owner_cover_min, digits=2)} of σ’s decisive episode')
    turns = cover.psi_turns if cover is not None else getattr(f.a.owner, 'psi_turns_per_cycle', None)
    if finite(turns) and turns >= s.mmr_owner_psi_turns_per_cycle_max:
        parts.append(f'Ψ/u makes {compare(turns, "<", s.mmr_owner_psi_turns_per_cycle_max)} revolutions per σ cycle')
    return '; '.join(parts)


# --- the roles of the other resonance in the header, the steps and the text (W28 A3b; cards g04-4, g04-5, g04-6,
# g04-7, g05-3, g06-1, g06-2): read from the verdict's role columns (A3a: cause_role, cause_term, owner, neighbours,
# resonant_argument, commensurability) and the owner's measured regimes; nothing is measured here

ROLE_OWNER, ROLE_DRIVER, ROLE_CANDIDATE = 'owner', 'driver', 'candidate driver'
ROLE_NEIGHBOUR, ROLE_ARGUMENT, ROLE_KINEMATIC = 'neighbour', 'resonant argument', 'kinematic'
ROLE_JOINT = 'joint'  # W28 A2b: origin joint, the other commensurability that librates at the same time
TERM_ITEM = re.compile(r'([+-])(\d+)(varpi\*?|Omega)(_[A-Za-z]+)?')


def term_text(term: str) -> str:
    """A secular term of `cause_term` (`roles.secular_term`) in symbols: '-4varpi' → '−4ϖ', '+2varpi+1varpi_M' →
    '2ϖ + ϖ_M', '-4varpi*' → '−4ϖ*', '+1Omega-1Omega_J' → 'Ω − Ω_J'; anything else (a resonance key, '2J-5S') as it is."""
    items = TERM_ITEM.findall(_text(term))
    if not items or ''.join(''.join(item) for item in items) != _text(term):
        return _text(term)
    out = []
    for i, (sign, n, name, sub) in enumerate(items):
        symbol = f'{"" if n == "1" else n}{name.replace("varpi", "ϖ").replace("Omega", "Ω")}{sub}'
        minus = sign == '-'
        out.append((f'{"−" if minus else ""}{symbol}') if i == 0 else f'{"−" if minus else "+"} {symbol}')
    return ' '.join(out)


def _span(t0: float, t1: float) -> str:
    """'494 yr–46.3 kyr'; two ends that print alike get the digits that tell them apart ('139.8–140.2 kyr', not
    '140 kyr–140 kyr': 2001 HW7 150 kyr)."""
    first, last = years(t0), years(t1)
    if first != last or not (finite(t0) and finite(t1)) or float(t0) == float(t1):
        return f'{first}–{last}'
    scale, unit = _time_scale(t1)
    return f'{_apart(float(t0) / scale, float(t1) / scale)} {unit}'


def _neighbour_verb(f: MMRFacts, t0: float, t1: float) -> str:
    """What the neighbor does in its block, from its measured regimes there (`OwnerResult.intervals`): 'librates',
    'drives a forced wave', else 'acts'."""
    owner = f.a.owner
    kinds = {str(kind) for a, b, kind in (owner.intervals if owner is not None else ()) if b >= t0 and a <= t1}
    return {frozenset({'libration'}): 'librates', frozenset({'forced_wave'}): 'drives a forced wave'}.get(frozenset(kinds), 'acts')


def neighbour_text(f: MMRFacts) -> str:
    """The neighbor in words (card g04-6: the owner outside the episode 'librates later/earlier, outside it'; A3a):
    '6E-11+0+5 librates after the episode (13.7 kyr–17.9 kyr), outside it'; a block during the row's own episodes
    'librates during 3 kyr–5 kyr but does not make σ’s cycles'; '' without neighbors."""
    items = neighbour_items(f.v.neighbours)
    if not items:
        return ''
    key = f.v.cause_term or items[0][0]
    timed = [item for item in items if finite(item[1])]
    if not timed:
        return f'{key} acts outside σ’s episodes'
    outside = [item for item in timed if item[3] in ('before', 'after')]
    during = [item for item in timed if item[3] not in ('before', 'after')]
    parts = []
    if outside:
        verb = _neighbour_verb(f, min(i[1] for i in outside), max(i[2] for i in outside))
        where = ' and '.join(dict.fromkeys(i[3] for i in outside))
        spans = ', '.join(_span(i[1], i[2]) for i in outside[:LIST_MAX])
        parts.append(f'{verb} {where} the episode ({spans}), outside it')
    if during:
        verb = _neighbour_verb(f, min(i[1] for i in during), max(i[2] for i in during))
        spans = ', '.join(_span(i[1], i[2]) for i in during[:LIST_MAX])
        parts.append(f'{verb} during {spans} but does not make σ’s cycles')
    return f'{key} ' + '; '.join(parts)


def _driver_period(f: MMRFacts) -> str:
    """', P 4.51 kyr' from own_owner_period_yr (the driver's own period, the one the forced wave has), else ''."""
    period = f.row.get('own_owner_period_yr')
    return f', P {years(float(period))}' if finite(period) else ''


def _kinematic_role(f: MMRFacts) -> str:
    """'kinematic: −4ϖ term (2ω cycle, 18.0 kyr); a does not follow' (card g05-3, the author: the term in the owner's
    place); the period is the secular one σ's period matched (`secular_reference`), 'a does not follow' when B1 failed
    (the leaf's own reading)."""
    v = f.v
    if str(v.origin_detail) == 'planet_ripple':
        return f'{ROLE_KINEMATIC}: a ripple of {v.cause_term or "the great inequality"}'
    _, ref, _, short = secular_reference(f)
    period = f' ({short}, {years(ref)})' if short else ''
    b1 = f.entry(B1_AMPLITUDE)
    follow = '; a does not follow' if b1 is not None and b1.result is False else ''
    return f'{ROLE_KINEMATIC}: {term_text(v.cause_term)} term{period}{follow}'


def cause_line(verdict, f: MMRFacts) -> Tuple[str, str]:
    """(role, text) of what makes the cycles, for the header's second line and the subtitle (cards g04-6 A: two words
    owner and driver, the reason unchanged; g04-4 the neighbor; g04-7/g05-3 the kinematic term; g06-1/g06-2 the
    resonant argument):

      owner             'owner: 5E-16+0+11 (librates in the episode)'
      driver            'driver: 3N-5+0+2 (circulates, P 4.51 kyr): forced wave'
      candidate driver  an owner or driver the text cannot confirm on σ's episode (`owner_confirmed`)
      neighbor         '6E-11+0+5 librates after the episode (13.7 kyr–17.9 kyr), outside it'
      resonant argument 'resonant argument: 9M-16+2+5 (one resonance 9M-16)' (rule C)
      kinematic         'kinematic: −4ϖ term (2ω cycle, 18.0 kyr); a does not follow'
      joint             'also in: 5J+6S-2+0+0-9 (both resonances at once)' (W28 A2b)

    ('', '') when the verdict names none. `verdict` is `f.v` (kept apart so the caller can pass the verdict it prints)."""
    v = verdict if verdict is not None else f.v
    role = str(v.cause_role)
    key = v.cause_term or v.owner
    if role in (ROLE_OWNER, ROLE_DRIVER) and key:
        if not owner_confirmed(f):
            return ROLE_CANDIDATE, f'{ROLE_CANDIDATE}: {key}: not shown to make these cycles'
        if role == ROLE_DRIVER:
            return ROLE_DRIVER, f'{ROLE_DRIVER}: {key} (circulates{_driver_period(f)}): forced wave'
        return ROLE_OWNER, f'{ROLE_OWNER}: {key} (librates in the episode)'
    argument = (
        f'{ROLE_ARGUMENT}: {v.resonant_argument} (one resonance {v.commensurability})'
        if v.resonant_argument and v.commensurability
        else (f'{ROLE_ARGUMENT}: {v.resonant_argument}' if v.resonant_argument else '')
    )
    if role == ROLE_NEIGHBOUR:
        text = neighbour_text(f)
        return ROLE_NEIGHBOUR, '; '.join(part for part in (argument, text) if part)
    if role == ROLE_KINEMATIC:
        return ROLE_KINEMATIC, _kinematic_role(f)
    if role == ROLE_JOINT and key:  # A2b: both resonances at once; the other one keeps its own row
        return ROLE_JOINT, f'also in: {key} (both resonances at once)'
    if argument:
        return ROLE_ARGUMENT, argument
    return '', ''


def slow_x_line(f: MMRFacts, p: Optional[_Printer] = None) -> str:
    """The slow swing of σ's center against the owner (A2 `SlowX`, columns own_x_*; card g05-1, the author's question
    about a ~10 kyr libration) at its level: identified 'σ’s center swings with period 6.6 kyr (±11°): the 2ω cycle
    π/|ω̇| = 7.11 kyr'; possible '…: probably the 2ω cycle (…); a slow libration of another resonance is not excluded';
    no cause '…: cause not identified; a slow libration of another resonance is not excluded'. '' without a period.
    `p`: the printer of the physics row that prints it (its numbers are recorded there)."""
    p = p if p is not None else _Printer(f.row)
    if not p.has('own_x_period_yr'):
        return ''
    amp = f' (±{p.num("own_x_halfamp_deg")}°)' if p.has('own_x_halfamp_deg') else ''
    head = f'σ’s center swings against the owner with period {p.time("own_x_period_yr")}{amp}'
    p.show('own_x_cause', 'own_x_cause_level', 'own_x_secular_term')
    level, term = _text(f.row.get('own_x_cause_level')), _text(f.row.get('own_x_secular_term'))
    name = SLOW_X_TERMS.get(term, f'the secular cycle {term}' if term else 'a secular cycle')
    # measured on the window x was tested on (`owner.slow_x`), not the record's KINEMATICS number (review #67)
    reference = f' {p.time("own_x_secular_period_yr")} on the tested window' if p.has('own_x_secular_period_yr') else ''
    not_excluded = 'a slow libration of another resonance is not excluded'
    if level == 'identified':
        return f'{head}: {name} ={reference}.' if reference else f'{head}: {name}.'
    if level == 'possible':
        return f'{head}: probably {name}{" (" + reference.strip() + ")" if reference else ""}; {not_excluded}.'
    return f'{head}: cause not identified; {not_excluded}.'


# own_x_secular_term (A2, `kinematics.secular_varpi_terms`) -> words
SLOW_X_TERMS = {'2omega': 'the 2ω cycle π/|ω̇|', **{f'g-g{j}': f'the cycle 2π/|g − g{j}|' for j in range(5, 9)}}


def _fraction(value: Fraction, symbol: str) -> str:
    """A coefficient times a symbol: 'φ/8', '(13/4)φ', '2φ', '−φ/8'."""
    sign, value = ('−' if value < 0 else ''), abs(value)
    if value.denominator == 1:
        return f'{sign}{"" if value.numerator == 1 else value.numerator}{symbol}'
    if value.numerator == 1:
        return f'{sign}{symbol}/{value.denominator}'
    return f'{sign}({value.numerator}/{value.denominator}){symbol}'


def identity_line(f: MMRFacts) -> str:
    """Card g05-1: why 'both angles librate' is one fact. From own_psi_def 'u*sigma∓v*phi' (Ψ = uσ ∓ vφ holds no λ):
    'σ = φ/8 + Ψ/8: what φ (5E-16+0+11) does shows in σ scaled by 1/8; σ is in both resonances only if Ψ/8, which holds
    no λ, stays bounded too (both resonances at once?: unresolved)'. '' without the definition."""
    match = PSI_DEF.match(_text(f.row.get('own_psi_def')))
    if match is None or not int(match.group(1)):
        return ''
    u, sign, v = int(match.group(1)), (1 if match.group(2) == '+' else -1), int(match.group(3))
    coefficient = Fraction(-sign * v, u)
    if not coefficient:
        return ''
    psi = f'Ψ/{u}' if u > 0 else f'Ψ/({u})'
    owner = f.v.owner or f.v.cause_term
    scale = f'{coefficient.numerator}/{coefficient.denominator}' if coefficient.denominator != 1 else str(coefficient.numerator)
    capture = _text(f.row.get('own_second_capture'))
    hull = ' over the owner’s stretches and the time between them' if capture and capture_hull(f) else ''
    test = f' ({CAPTURE_LABEL}: {CAPTURE_WORDS.get(capture, capture)}{hull})' if capture else ''
    return (
        f'σ = {_fraction(coefficient, "φ")} + {psi}: what φ ({owner}) does shows in σ scaled by {scale}; σ is in both '
        f'resonances only if {psi}, which holds no λ, stays bounded too{test}.'
    )


def varpi_term(p: int, mark: str = '') -> str:
    """'−10ϖ', '+9ϖ' → '9ϖ', '−ϖ': the body's perihelion term of σ."""
    coefficient = '' if abs(p) == 1 else str(abs(p))
    return f'{"−" if p < 0 else ""}{coefficient}ϖ{mark}'


# the secular periods a kinematic libration's period is compared with (`decide.secular_period_offset`): column, unit
# factor to yr, the long name (text) and the short one (the header's cause line)
SECULAR_REFERENCES = (
    ('kin_two_omega_period_yr', 1.0, 'the 2ω cycle π/|ω̇|', '2ω cycle'),
    ('kin_modulation_period_kyr', KYR, 'the e cycle 2π/|g − g5|', 'e cycle g − g5'),
)


# --- the secular cause of a slow or kinematic swing (W28 B10b, cards g08-8 'here and everywhere', g05-2, g05-3; the
# columns of B10a): what panel (j) labels and the text names. Nothing is measured here.

# a secular term (slow_secular_term, slow_cause_term, own_x_secular_term) -> (long name, short name, symbol)
SECULAR_TERM_NAMES = {
    '2omega': ('the 2ω cycle π/|ω̇|', '2ω cycle', 'π/|ω̇|'),
    'g-g5': ('the e cycle 2π/|g − g5|', 'e cycle g − g5', '2π/|g−g5|'),
    **{f'g-g{j}': (f'the cycle 2π/|g − g{j}|', f'cycle g − g{j}', f'2π/|g−g{j}|') for j in range(6, 9)},
}
SECULAR_CAUSES = ('secular_2omega', 'secular_g_gj')  # slow_cause / own_x_cause of a secular cycle
SLOW_KINEMATIC = 'kinematic'  # slow_origin of a secular cycle whose identity fails (B10a `SlowOrigin.KINEMATIC`)


def secular_term_period(row: Mapping[str, Any], term: str) -> float:
    """The period [yr] of a secular term from the column that measured it: the slow scale's matched term
    (slow_secular_period_yr), A2's slow x (own_x_secular_period_yr), the 2ω cycle π/|ω̇| (kin_two_omega_period_yr),
    the e cycle 2π/|g − g5| (kin_modulation_period_kyr); nan for none."""
    term = _text(term)
    if not term:
        return float('nan')
    for term_column, period_column in (('slow_secular_term', 'slow_secular_period_yr'), ('own_x_secular_term', 'own_x_secular_period_yr')):
        if _text(row.get(term_column)) == term and finite(row.get(period_column)):
            return float(row.get(period_column))
    if term == '2omega' and finite(row.get('kin_two_omega_period_yr')):
        return float(row.get('kin_two_omega_period_yr'))
    if term == 'g-g5' and finite(row.get('kin_modulation_period_kyr')):
        return float(row.get('kin_modulation_period_kyr')) * KYR
    return float('nan')


def slow_secular_cause(row: Mapping[str, Any]) -> Tuple[str, float, str]:
    """(term, its period [yr], level) of the slow scale's cause when it is a secular cycle (slow_cause secular_2omega
    or secular_g_gj, at any level: card g05-2, 'not sure is OK, just say it'); ('', nan, '') otherwise. What panel (j)
    overlays as p·ϖ (`mmr_combined.draw_varpi_overlay`)."""
    if _text(row.get('slow_cause')) not in SECULAR_CAUSES:
        return '', float('nan'), ''
    term = _text(row.get('slow_cause_term'))
    return term, secular_term_period(row, term), _text(row.get('slow_cause_level'))


def slow_kinematic(row: Mapping[str, Any], tolerance: float) -> bool:
    """The swing a kinematic verdict names is the slow scale's (rule K3s, B10a; 37087 3J+8S-3): slow_origin kinematic
    and σ's own decisive period matches no secular period (kin_cycle_secular_offset > `tolerance` =
    mmr_kin_period_tol, or not measured). Otherwise a kinematic libration is σ's main-scale swing (285637)."""
    if _text(row.get('slow_origin')) != SLOW_KINEMATIC:
        return False
    offset = row.get('kin_cycle_secular_offset')
    return not (finite(offset) and float(offset) <= float(tolerance))


def secular_reference(f: MMRFacts) -> Tuple[float, float, str, str]:
    """(σ's decisive period, the matched secular period [yr], long name, short name) for origin_detail secular_cycle:
    the closer of the two secular periods the decision compared (`decide.secular_period_offset`); on a row whose
    kinematic swing is the slow scale's (`slow_kinematic`, K3s) the slow period and its matched term
    (slow_period_yr, slow_secular_period_yr, slow_secular_term); (nan, nan, '', '') otherwise."""
    none = (float('nan'), float('nan'), '', '')
    if str(f.v.origin_detail) != 'secular_cycle':
        return none
    if slow_kinematic(f.row, f.s.mmr_kin_period_tol):
        term = _text(f.row.get('slow_secular_term'))
        names = SECULAR_TERM_NAMES.get(term)
        period, ref = _float(f.row.get('slow_period_yr')), _float(f.row.get('slow_secular_period_yr'))
        return (period, ref, names[0], names[1]) if names and finite(period) and finite(ref) else none
    period = _float((f.episodes.decisive or {}).get('period_med_yr'))
    ref, long, short = closest_secular_period(f.row, period)
    return (period, ref, long, short) if long else none


def closest_secular_period(row: Mapping[str, Any], period: Any) -> Tuple[float, str, str]:
    """(secular period [yr], long name, short name) of SECULAR_REFERENCES closest to `period` [yr] (the comparison
    `decide.secular_period_offset` made); (nan, '', '') without a period or a reference."""
    references = [(_float(row.get(column)) * scale, long, short) for column, scale, long, short in SECULAR_REFERENCES]
    references = [item for item in references if finite(item[0]) and item[0] > 0]
    if not (references and finite(period)):
        return float('nan'), '', ''
    period = float(period)
    return min(references, key=lambda item: abs(period / item[0] - 1))


def _signed(term: Any) -> str:
    """'-8varpi' as `term_text` reads it ('+8varpi' for a positive coefficient without a sign)."""
    term = _text(term)
    return term if term[:1] in '+-' else f'+{term}'


def info_items(f: MMRFacts, physics: Sequence[PhysicsSection]) -> List[Tuple[str, str]]:
    """(check value, words) of `mmr_info`; a 'fail' among them is an objection that did not decide (card g10-8)."""
    items = []
    for text in mmr_info(f, physics):
        name = next((key for key, value in f.checks.items() if value == 'fail' and text == f'{CHECK_WORDS.get(key, key)} fail'), '')
        kinematic = text.startswith('ϖ-kinematic test') and f.checks.get('kinematic') == 'fail'
        items.append(('fail' if name or kinematic else '', text))
    return items


# where the planets' q–Q bands come from (ctx_orbit_overlap_source): a planet not saved is taken at its J2000 orbit
ORBIT_SOURCE_WORDS = {'saved': '', 'J2000': 'planets taken at their J2000 orbits'}


def orbit_line(row: Mapping[str, Any]) -> str:
    """The radial overlap of the body's band q…Q with the eight planets' bands (ctx_orbit_overlap, card g10-6, E6):
    'Radial overlap with Venus, Earth', 'Radial overlap: none', 'Radial overlap: no data'; the source of the planet
    bands when not all saved. Necessary for an encounter, not sufficient; the q and Q are in the numbers below."""
    overlap, source = _text(row.get('ctx_orbit_overlap')), _text(row.get('ctx_orbit_overlap_source'))
    if overlap in ('', 'nan'):
        return 'Radial overlap: no data.'
    words = ORBIT_SOURCE_WORDS.get(source)
    if words is None and source.startswith('saved:'):  # 'saved:Earth+Mars': the others at J2000
        saved = set(source[len('saved:') :].split('+'))
        words = f'{", ".join(name for name in PLANET_NAMES_ORDER if name not in saved)} taken at their J2000 orbits'
    note = f' ({words})' if words else ''
    if overlap == 'none':
        return f'Radial overlap: none{note}.'
    return f'Radial overlap with {", ".join(overlap.split(";"))}{note}.'


def _comm(f: MMRFacts) -> str:
    return f.v.commensurability or f.a.resonance.key


VERDICT_LINES = {
    'kinematic_libration': lambda f: f'NOT THIS RESONANCE: σ librates, but {_comm(f)} does not make the swing.',
    'kinematic_drift': lambda f: f'NOT THIS RESONANCE: σ’s center follows its perihelion term, not {_comm(f)}.',
    'induced_libration': lambda f: (
        f'NOT THIS RESONANCE: σ follows {f.v.owner} through the identity.'
        if owner_confirmed(f) or not f.v.owner
        else f'NOT THIS RESONANCE: σ follows a candidate driver, {f.v.owner}, through the identity.'
    ),
    'subresonance_librates': lambda f: f'{status_head(f.v.status).upper()}: a member of the multiplet {_comm(f)} holds the center.',
    'dynamic_libration': lambda f: f'RESONANT: σ is the own pendulum of {_comm(f)} over the whole record.',
    'dynamic_episode': lambda f: f'TRANSIENT: σ is the own pendulum of {_comm(f)} in part of the record only.',
    'geometric_libration': lambda f: 'RESONANT by the shape alone: σ librates over the whole record; its own force is not measured.',
    'geometric_episode': lambda f: 'TRANSIENT by the shape alone: σ librates in part of the record; its own force is not measured.',
    'separatrix_zone': lambda f: f'{status_head(f.v.status).upper()}: σ moves near the separatrix of {_comm(f)}.',
    'origin_unknown': lambda f: f'UNCERTAIN: σ librates, but nothing measured shows that {_comm(f)} makes it.',
    'implausible_strength': lambda f: f'{status_head(f.v.status).upper()}: {_comm(f)} looks too weak for the period of these cycles.',
    'circulation': lambda f: 'NOT IN THIS RESONANCE: σ circulates.',
    'alias': lambda f: f'{status_head(f.v.status).upper()}: the output step folds σ.',
    'no_resonant_potential': lambda f: 'NOT THIS RESONANCE: σ reverses at times, but no potential from a holds it.',
    'modulated_circulation': lambda f: 'NOT IN THIS RESONANCE: σ stalls at the top of its potential without a libration.',
    'drift_mechanism_unknown': lambda f: 'UNCERTAIN: σ’s center drifts and no tested mechanism explains it.',
    'baseline_needed': lambda f: 'UNCERTAIN: the record does not decide.',
}


def verdict_line(f: MMRFacts) -> str:
    """The verdict in plain words (the slot 'verdict'), by the verdict's reason; '' for a reason without one."""
    single = SINGLE_SWING_LINES.get((f.rule, str(f.v.reason)))  # W28 C6: a leaf that decided on one swing
    if single:
        return single.format(comm=_comm(f))
    build = VERDICT_LINES.get(str(f.v.reason))
    return build(f) if build else ''


# --------------------------------------------------------------------------------------------- numbers behind the verdict

# the second word of a decision-path entry that names its check ('3 B1 a carries the amplitude', '5 L4 force law R2')
ENTRY_IDS = ('B1', 'B5', 'K2', 'K2b', 'L3', 'L4', 'R0', 'O1', 'F')
# an entry of a step that names no check: the step's own check ('1 subresonance …' -> S, '2 owner R2' -> W)
STEP_ENTRY_IDS = {'1': 'S', '2': 'W', '3': 'K1'}
# entries of a step that belong to another question than the step's check (`CAPTURE_LABEL`: both resonances at once?)
ENTRY_PREFIX_IDS = {'2 second capture': 'capture', '3 kinematic (slow centre)': 'K2'}
# a cap-class margin -> the column of the number its LIMIT row compares (`_detail_o1`, `_detail_b5`, `_detail_f`)
CAP_MARGIN_COLUMNS = {'O1': 'ep_onset_ratio', 'B5': 'rsp_b5_ratio', 'F': 'str_ratio'}
# a check that can object (`objection_line`) -> the columns of its measurement, the first ones present are printed
OBJECTION_COLUMNS = {
    'identity': ('rsp_identity_corr', 'rsp_identity_slope_ratio'),
    'potential': ('sep_VE_VF_agree_fraction', 'rsp_coorb_energy_R2'),
    'a_response': ('rsp_force_R2_h1', 'rsp_force_R2_h2', 'rsp_coorb_K_over_K_theory'),
    'kinematic': ('kin_corr_rate',),
}
OBJECTION_NOTE = 'fails, not read by this verdict'


def entry_rule_id(name: str) -> str:
    """The id a decision-path entry belongs to, read from its name: its first word for a G rule or a leaf ('G4 largest
    center pass' -> G4, '5n independent cycles' -> 5n, 'C5 alias: …' -> C5); a step's entry by the check it names ('3 B1
    a carries the amplitude' -> B1, '5o F: required / theory' -> F, any strength entry of 5s -> F), else by its step
    (`STEP_ENTRY_IDS`: '2 owner R2' -> W)."""
    words = name.split()
    if not words:
        return ''
    head = words[0]
    if not head[0].isdigit():
        return head
    prefix = next((key for key in ENTRY_PREFIX_IDS if name.startswith(key)), '')
    if prefix:
        return ENTRY_PREFIX_IDS[prefix]
    named_ids = [word.rstrip(':') for word in words[1:4]]
    if named_ids and named_ids[0] in ENTRY_IDS:
        return named_ids[0]
    if 'F' in named_ids or head.startswith('5s'):
        return 'F'
    return STEP_ENTRY_IDS.get(head, head)


def decisive_ids(f: MMRFacts, steps: Sequence[Step]) -> Tuple[str, ...]:
    """The ids whose comparisons set the status: the leaf step's ids, those of a step the rule passed through on its
    branch ('5d' of 5d/6), and the rule's parts themselves ('5n', 'C5'); a leaf without ids (circulation) reads the
    angle shape's G rules."""
    parts = f.rule.split('/') if f.rule else []
    chosen = [step for step in steps if step.state == StepState.LEAF or (step.state == StepState.PASSED and step.number in parts)]
    ids = [rule_id for step in chosen for rule_id in step.ids]
    if not ids:
        ids = [rule_id for step in steps if step.key == 'shape' for rule_id in step.ids]
    return tuple(dict.fromkeys(ids + parts))


def _ledger_unit(unit: str) -> str:
    return '' if not unit else ('°' if unit in ('deg', '°') else f' {unit}')


def _column_comparison(entries) -> str:
    """The comparisons of one column as they came out: '0.0954 < 0.1'; two bounds that both held as one interval
    '1.04 within [0.333, 3]'."""
    bounds = {entry.op: entry for entry in entries}
    unit = _ledger_unit(entries[0].unit)
    if len(entries) == 2 and all(entry.result for entry in entries) and {'<=', '>='} <= set(bounds):
        lo, hi = bounds['>='].threshold, bounds['<='].threshold
        return f'{fmt(entries[0].value)}{unit} within [{fmt(lo)}, {fmt(hi)}]'
    entry = next((entry for entry in entries if entry.result is False), entries[-1])
    return compare(entry.value, str(entry.op), entry.threshold, unit)


STRENGTH_FAILED_NAME = 'strength implausible for this period'  # a DECISIVE F whose check failed
PASSING_CHECKS = {'L4': 'a_response', 'R0': 'potential'}  # a decisive id whose check passed without a ledger value


def _decisive_lines(f: MMRFacts, steps: Sequence[Step]) -> List[Emphasis]:
    """DECISIVE: per decisive id, its decision-path comparisons with a value (one part per column; a flag compared
    with '==', such as the owner's route, says nothing a reader can weigh and is left out); UNMEASURED: a decisive id
    whose every comparison lacks its value."""
    ids = decisive_ids(f, steps)
    by_id: Dict[str, Dict[str, list]] = {}
    for entry in f.v.decision_path:
        rule_id = entry_rule_id(entry.name)
        if entry.column and rule_id in ids and str(entry.op) != '==':
            by_id.setdefault(rule_id, {}).setdefault(entry.column, []).append(entry)
    out = []
    for rule_id in ids:
        direct = _direct_law_parts(f) if rule_id == 'L4' else ()
        if direct:  # B8: the direct law decided L4; the harmonic entries on the path are what it overruled
            out.append(Emphasis(EMPHASIS_DECISIVE, rule_id, DIRECT_FORCE_NAME, direct))
            continue
        columns = by_id.get(rule_id)
        if not columns:
            continue
        measured = {column: entries for column, entries in columns.items() if any(entry.value is not None for entry in entries)}
        name = decisive_name(f, rule_id)
        if measured:
            parts = tuple(
                EmphasisPart(column, _column_comparison([entry for entry in entries if entry.value is not None]))
                for column, entries in measured.items()
            )
            out.append(Emphasis(EMPHASIS_DECISIVE, rule_id, name, parts))
        elif f.checks.get(PASSING_CHECKS.get(rule_id, '')) == 'pass':  # 79653: R0 passed inside an episode, the
            detail = (_potential_detail if rule_id == 'R0' else _force_detail)(f)[0]  # record's entry has no value
            out.append(Emphasis(EMPHASIS_DECISIVE, rule_id, name, (EmphasisPart('', detail, labelled=False),) if detail else ()))
        else:  # L4 and R0 say why even where the tree left no margin (2026 AC4: an uncertain leaf, no level)
            why = short_reason(why_not_measured(f, rule_id)[1]) if rule_id in f.margins or rule_id in ('L4', 'R0') else ''
            note = f'not measured ({why})' if why and why != 'not measured' else 'not measured'
            out.append(Emphasis(EMPHASIS_UNMEASURED, rule_id, name, (), note))
    return out


def decisive_name(f: MMRFacts, rule_id: str) -> str:
    """The name of a DECISIVE line: what the comparison found. S and W are named by their check ('no member holds the
    center') only when nothing was accepted; an accepted member or owner is named as the margin row names it ('member
    13M-24+2+9 holds the center', 'owner 11M-20+0+9 drives a', 'candidate driver …': 82831, 117306, 1997 GK3)."""
    if rule_id in ('S', 'W'):
        named_row = _margin_name(f, rule_id)
        if named_row != MMR_MARGIN_NAMES.get(rule_id, rule_id):
            return named_row
    if rule_id == 'F' and f.checks.get('strength') == 'fail':  # 2897 (5s): not 'strength plausible'
        return STRENGTH_FAILED_NAME
    return rule_name(rule_id, subtype=str(f.v.angle_subtype)) or rule_id  # G6 by the word it set (arc, stall)


def _direct_law_parts(f: MMRFacts) -> Tuple[EmphasisPart, ...]:
    """The comparisons of the direct averaged-force law that decided L4 (W28 B8; columns rsp_dforce_*), () when it did
    not: its R², its scale band, its gain over the shifted null."""
    law = taken_direct_law(f.a.response)
    if law is None:
        return ()
    s = f.s
    lo, hi = _strength_band(f)
    return (
        EmphasisPart('rsp_dforce_R2', compare(law.r2, '>=', s.mmr_force_r2_fail)),
        EmphasisPart('rsp_dforce_scale_min', compare(law.scale_min, '>=', lo)),
        EmphasisPart('rsp_dforce_scale_max', compare(law.scale_max, '<=', hi)),
        EmphasisPart('rsp_dforce_null_gain', compare(law.null_gain, '>=', s.mmr_force_null_gain)),
    )


def _limit_lines(f: MMRFacts, checks: Sequence[CheckRow]) -> List[Emphasis]:
    """LIMIT: the caps that set the shown level (their ceiling is the confidence), with the comparison of the failed
    cap-class margin merged into them (`merge_limits`)."""
    level = str(f.v.confidence)
    out = []
    for row in checks:
        ceiling = row.counts.split('≤')[-1].strip() if '≤' in row.counts else ''
        if not row.is_limit or ceiling != level:
            continue
        margin = row.merged or CAP_MARGINS.get(row.key[len(CAP_PREFIX) :], '')
        column = CAP_MARGIN_COLUMNS.get(margin, '') if row.merged else ''
        parts = (EmphasisPart(column, row.detail, labelled=False),) if column and row.detail else ()
        out.append(Emphasis(EMPHASIS_LIMIT, row.printed_id, row.short_name or row.name, parts, f'→ {level.upper()}'))
    return out


def _conflict_lines(f: MMRFacts, items: Sequence[Tuple[str, str]]) -> List[Emphasis]:
    """CONFLICT: a measured check that fails and that the verdict did not read (the objection, `objection_line`)."""
    failed = {text for value, text in items if value == 'fail'}
    out = []
    for check, value in f.checks.items():
        if value != 'fail' or f'{CHECK_WORDS.get(check, check)} fail' not in failed:
            continue
        columns = [column for column in OBJECTION_COLUMNS.get(check, ()) if finite(f.row.get(column))]
        parts = tuple(EmphasisPart(column, fmt(f.row[column])) for column in columns)
        rule_id = CHECK_MARGIN.get(check, check)
        out.append(Emphasis(EMPHASIS_CONFLICT, rule_id, rule_name(rule_id) or CHECK_WORDS.get(check, check), parts, OBJECTION_NOTE))
    return out


def number_emphasis(
    f: MMRFacts,
    steps: Sequence[Step],
    checks: Sequence[CheckRow],
    items: Sequence[Tuple[str, str]],
) -> List[Emphasis]:
    """The lines of 'Numbers behind this verdict' (card g10-5), read from the ledger and the confidence rows the figure
    already prints: decisive comparisons, the binding caps, the objections, the decisive tests without a value; in
    that order (`EMPHASES`). A number is printed wherever it is needed (D1), the physics block's ones too."""
    lines = _decisive_lines(f, steps) + _limit_lines(f, checks) + _conflict_lines(f, items)
    return sorted(lines, key=lambda line: EMPHASES.index(line.kind))


def explain_mmr(analysis, settings, row: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]) -> VerdictText:
    """The steps, confidence rows, physics sections and grey columns of one MMR verdict. `row` = summary ⋈ numbers
    (summary.csv over mmr_numbers.csv), `episodes` = episodes.csv rows of the angle."""
    f = MMRFacts(analysis, settings, row, episodes)
    checks = mmr_checks(f)
    grey = {}
    if f.checks.get('a_response') == 'not_measured' or (f.margins.get('L4') is not None and f.margins['L4'].margin == Margin.NOT_MEASURED):
        why = why_not_measured(f, 'L4')[1].split(':')[0]
        grey.update({column: why for column in L4_COLUMNS})
    if f.checks.get('potential') == 'not_measured':
        grey['sep_VE_VF_agree_fraction'] = why_not_measured(f, 'R0')[1].split(':')[0]
    physics = physics_sections(f)
    return VerdictText(mmr_steps(f), checks, grey, physics)


def shown_rule(f: MMRFacts) -> str:
    """The rule the subtitle and the steps' header name: rec_rule, except that an insufficient_time leaf (rec_rule G6
    for every subtype, `decide.decide_status`) names the angle rule that set the word (column geo_angle_rule: G5a one
    slow arc, G5w a slow swing in the well, W28 B6c, or G6 itself), so 'decided: slow swing in the well (G5w)' agrees
    with step 0; an angle rule that set no such word is rebuilt from the ledger (`_base_shape_ids`)."""
    if f.rule == 'G6' and str(f.v.angle_status) == 'insufficient_time':
        if insufficient_angle_rule(f):
            return insufficient_angle_rule(f)
        shape = [rule_id for rule_id in _base_shape_ids(f, 'insufficient_time') if rule_id.startswith('G')]
        return shape[-1] if shape else f.rule
    return f.rule


# --------------------------------------------------------------------------------------------- secular and ZLK


def _long_n(margin, settings, where: str) -> Tuple[str, str, str]:
    """SZ.N: thin at the minimum (`decide`: n_tp <= LIBRATION_MIN_TURNING_POINTS), which is enough for a libration
    but not for a full margin; said once, without a 'need' that reads as a failure (review #20)."""
    minimum = settings.libration_min_turning_points
    if margin.margin == Margin.THIN:
        note = f'{minimum}, the fewest that make a libration; a full margin needs more than {minimum}'
        return f'{fmt(margin.value)} turning points {where}', '', note
    return f'{fmt(margin.value)} turning points {where}', f'> {minimum}', ''


def _long_s(margin, settings, where: str) -> Tuple[str, str, str]:
    """SZ.S: not measured on one cycle (one center estimate, nothing to compare): only that sentence."""
    if margin.margin == Margin.NOT_MEASURED:
        return '', '', 'not measured: one cycle gives one center estimate, two are needed to compare'
    return f'{fmt(margin.value)} independent center estimate(s)', '≥ 2, bands overlap', ''


def _long_f(margin, settings, where: str) -> Tuple[str, str, str]:
    note = 'the libration is faster than 2 P_c: the filter shapes it' if margin.margin == Margin.NOT_MEASURED else ''
    detail = f'P_cyc / P_c {compare(margin.value, ">=", settings.filter_min_libration_cutoffs)}'
    return detail, f'≥ {fmt(settings.filter_min_libration_cutoffs)}', note


LONG_PERIOD_DETAILS = {
    'N': _long_n,
    'S': _long_s,
    'F': _long_f,
    'P': lambda margin, settings, where: (f'smallest pass {fmt(margin.value)} turns', '≥ 1', ''),
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
    name = LONG_PERIOD_MARGIN_NAMES.get(key, key)
    return CheckRow(key, name, state, detail, need, counts, note, kind=margin_kind(margin), rule_id=long_period_id(key))


EI_TAIL = ' (says how e and i trade, not whether ω librates)'  # the record's coupling changes no status


def _ei_line(ei) -> str:
    """'e-i at 6 kyr, whole record: forced beat (in phase): i against e 328°, r +0.78, δi/δe 0.358 × the constant-H
    slope (H = √(1−e²) cos i), 1.2 × the forced-beat slope (says how e and i trade, not whether ω librates)'; the two
    slopes are `zlk.diagnostics.measure_ei_coupling`'s laws (review #16)."""
    if ei.coupling == 'not_measured':
        return f'e-i coupling not measured ({snake_words(ei.reason) or "no period"})'
    where = f'at {years(ei.period_yr)}, whole record' + (
        ' (half the circulation period of ω)' if ei.period_source == 'circulation_harmonic' else ''
    )
    r = f'{ei.band_r:+.2f}'.replace('-', '−')
    return (
        f'e-i {where}: {EI_WORDS.get(ei.coupling, ei.coupling)}: i against e {ei.phase_deg:.0f}°, r {r}, '
        f'δi/δe {fmt(ei.ratio_h)} × the constant-H slope (H = √(1−e²) cos i), {fmt(ei.ratio_beat)} × the forced-beat '
        f'slope{EI_TAIL}'
    )


def _centre_line(line) -> str:
    """'slow line of the center: 53.8 kyr, ±21.6°: forced beat g6-g5 (r 0.99, model ±22.8°)'; '' when not measured."""
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
    return f'slow line of the center: {years(line.period_yr)}, ±{math.degrees(line.half_amp_rad):.3g}°: {origin}{model}'


def _mode_longitude(mode: str) -> str:
    """'g5' -> 'ϖ5': the longitude of a forced eccentricity mode."""
    return f'ϖ{mode[1:]}' if mode.startswith('g') and len(mode) > 1 else f'ϖ({mode})'


CENTRE_CAUSE_ORIGINS = ('forced_beat:', 'forced_direction', 'node')


def _node_lock_words(lock) -> str:
    """'the node is locked to g5: Ω − ϖ5 within 50–130° (1.5 cycles of 1.46 Myr)', or, for a rate within the
    resolution whose difference closes no cycle, that a lock is not resolved; '' otherwise."""
    if lock is None or not lock.candidate:
        return ''
    mode, sym = lock.candidate, _mode_longitude(lock.candidate)
    span = f'Ω − {sym} within {lock.min_deg:.0f}–{lock.max_deg:.0f}°'
    if lock.locked:
        return f'the node is locked to {mode}: {span} ({fmt(lock.cycles)} cycles of {years(lock.period_yr)})'
    if lock.reason == 'unresolved':
        return (
            f'the node follows {mode} within the resolution (Ω̇ − {mode} {lock.rate_diff_arcsec_yr:+.2f}″/yr, '
            f'{span}); a lock is not resolved: no full cycle of Ω − {sym} in the record'
        )
    return ''


# --------------------------------------------------------------------------------------------- secular and ZLK: slots
#
# The slotted explanation of a secular or Lidov-Kozai verdict (W28 E9; cards g10-2 style (i), g10-8, g09-5 C, g09-3,
# g09-6, g09-7, g10-4b): the same slots as the MMR one (`mmr_bullets`) plus the time scales (`SLOT_SCALE`). Every
# sentence reads the verdict, the ledger or a column of the row.

GATE_DECISIVE = {
    'confirmed': 'The free ω librates too (the gate confirms it): a libration of the free elements, not only of the osculating ω.',
    'kinematic': (
        'The free ω winds: the osculating ω librates only because the forced eccentricity holds ϖ; a kinematic libration, ' 'not a ZLK one.'
    ),
    'short_window': 'The record holds too few libration periods for the free-element split: the gate needs more of them.',
    'undefined_phase': 'The free vector passes too near the origin for its phase to be unwrapped: the free ω is not measured.',
    'unavailable': 'The free elements could not be built: nothing measured the free ω.',
    'disabled': 'The free-ω gate is switched off: the shape of ω decides.',
}
SECULAR_DECISIVE = {
    'libration': 'The libration covers the record; a secular status rests on the shape of the angle (no force-law or owner test exists for it).',
    'episodic': 'The libration does not cover the record.',
    'circulation': 'A circulating angle is not in the resonance: no other test is needed.',
    'drift': 'The center moves one way only: there is no fixed center to librate about in this record.',
    'near_separatrix': 'No libration: the angle slips over the top.',
    'insufficient_time': 'Fewer turning points than one full cycle needs: the shape cannot tell a libration from a slow circulation.',
}


def _row_float(row: Mapping[str, Any], column: str) -> float:
    value = row.get(column)
    return float(value) if finite(value) else float('nan')


def record_span_yr(analysis) -> float:
    """T: the span of the analyzed record [yr] (nan without a series)."""
    series = analysis.series
    return float(series.t_yr[-1] - series.t_yr[0]) if series is not None and len(series.t_yr) > 1 else float('nan')


def librating_episode_rows(analysis, episodes: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """The episodes.csv rows of the librating episodes: the predicate of sec_n_librating_episodes
    (`secular.decide.librating_episodes`: accepted, not faster than the type's shortest libration, the center not
    running a turn); every accepted row when the analysis has no geometry (a row restored without arrays)."""
    accepted = [row for row in episodes if episode_accepted(row)]
    if analysis.geometry is None:
        return accepted
    ids = [episode.episode_id for episode in secular_librating_episodes(analysis.geometry)]
    return [row for row in accepted if any(_same_id(row.get('episode_id'), i) for i in ids)]


def _coorbital_planet(row: Mapping[str, Any]) -> str:
    """'Earth' from zlk_gate_matches 'coorbital_unfitted_forcing:Earth' ('a planet' when not named)."""
    for item in _text(row.get('zlk_gate_matches')).split(';'):
        if item.startswith('coorbital_unfitted_forcing:'):
            return item.split(':', 1)[1] or 'a planet'
    return 'a planet'


def _resolution_said(row: Mapping[str, Any], span_yr: float) -> bool:
    """True when the decisive line prints the free node's resolution and it is 1/T of the record (one Rayleigh width
    over the whole record): the time-scale line then refers to it instead of printing it twice."""
    resolution = _row_float(row, 'zlk_gate_node_near_resolution_arcsec_yr')
    if not (_text(row.get('zlk_gate_node_near')) and finite(resolution) and finite(span_yr) and span_yr > 0):
        return False
    return abs(resolution - ARCSEC_PER_TURN / span_yr) <= 1e-3 * resolution


def _node_near_sentence(row: Mapping[str, Any], span_yr: float) -> Tuple[str, Tuple[str, ...]]:
    """The free node not yet separated from s_j (D8, card g09-7): its distance against the resolution, when they
    separate; not the number of cycles. '' when no node is flagged near a planetary rate."""
    mode = _text(row.get('zlk_gate_node_near'))
    distance = _row_float(row, 'zlk_gate_node_near_d_arcsec_yr')
    resolution = _row_float(row, 'zlk_gate_node_near_resolution_arcsec_yr')
    if not mode or not finite(distance):
        return '', ()
    name = '1/T, the resolution of the record' if _resolution_said(row, span_yr) else 'the resolution of the free elements'
    separation = _row_float(row, 'zlk_gate_node_separation_yr')
    after = f'; they separate after {years(separation)}' if finite(separation) else ''
    text = (
        f'The free node is not yet separated from {mode}: |b_Ω − {mode}| {compare(distance, "<", resolution, "″/yr")} '
        f'({name}){after}. ω has the cycles the gate needs; the node’s frequency is what is not resolved.'
    )
    return text, ('zlk_gate_node_near_d_arcsec_yr', 'zlk_gate_node_near_resolution_arcsec_yr', 'zlk_gate_node_separation_yr')


def _no_verdict_ei_sentence(analysis, row: Mapping[str, Any]) -> str:
    """D10 (q-1 E): an episodic ω without a gate verdict takes its status from the e-i exchange at the deciding period
    (rule zlk_gate_unconfirmed): the class found and the status it gave."""
    coupling = _text(row.get('zlk_ei_coupling'))
    found = EI_WORDS.get(coupling, snake_words(coupling) or 'not measured')
    if policy_minimal(analysis.verdict):  # W29 P7: no e-i requirement, the shape keeps its status (flag gate_unavailable)
        return (
            f'The free-ω gate gives no verdict, so the angle’s shape keeps its status '
            f'({status_head(analysis.verdict.status).lower()}, flag gate_unavailable); the e-i exchange at the deciding '
            f'period: {found} (a column only).'
        )
    return (
        f'The free-ω gate gives no verdict, so an episodic ω needs the ZLK exchange of e and i at the deciding period: '
        f'{found} → {status_head(analysis.verdict.status).lower()}.'
    )


def _phase_modes(analysis) -> Tuple[List[str], str]:
    """(the planetary modes of a secular angle that follow the linear law, the one whose phase strays most over the
    record by the same table `decide.phase_check` reads); ([], '') without modes."""
    series = analysis.series
    if series is None:
        return [], ''
    coefficients: Dict[str, int] = {}
    for term in series.terms:
        if term.index is not None:
            label = f'{term.mode}{term.index}'
            coefficients[label] = coefficients.get(label, 0) + term.coefficient
    modes = [label for label, k in coefficients.items() if k and label not in series.measured_modes]
    if not modes:
        return [], ''
    table = phase_uncertainty.load_table()
    spread = {label: table.delta_phi_rad({label: coefficients[label]}, series.span_yr, series.measured_modes) for label in modes}
    largest = max(modes, key=lambda label: spread[label] if finite(spread[label]) else -1.0)
    return modes, largest


def _phase_sentence(analysis, row: Mapping[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    """D5 (card g09-3 B): the swing is within the uncertainty of the linear mode phases."""
    swing, spread = _row_float(row, 'sec_phase_swing_deg'), _row_float(row, 'sec_phase_uncertainty_deg')
    modes, largest = _phase_modes(analysis)
    names = ', '.join(modes) + (f' (mostly {largest})' if len(modes) > 1 and largest else '')
    text = (
        f'The swing {compare(swing, "<=", spread, "°")}, the phase uncertainty of {names or "the angle’s modes"} over the record '
        f'under linear phases: these turning points may be those of the phases. Rerun with measured phases '
        f'(SECULAR_PLANET_PHASES=measured, SAVE_PLANETS=all).'
    )
    return text, ('sec_phase_swing_deg', 'sec_phase_uncertainty_deg')


def long_period_decisive_line(analysis, row: Mapping[str, Any]) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """(the argument that set the status in plain physics, its rule ids, the columns it prints): the free-ω gate's
    outcome for a ZLK row (the co-orbital model not applying, the node not separated from s_j, the e-i exchange of an
    episodic ω without a gate verdict), else the shape of the angle (the swing within the phase uncertainty, a center
    that turned back)."""
    v = analysis.verdict
    angle, rules = str(v.angle_status), set(v.rules_fired)
    gate = getattr(analysis, 'gate', None)
    outcome = _text(gate.outcome) if gate is not None else ''
    span = record_span_yr(analysis)
    if outcome and outcome != 'not_applicable':
        if 'zlk_gate_unconfirmed' in rules:
            return _no_verdict_ei_sentence(analysis, row), ('SZ.G',), ()
        if 'zlk_coorbital_episodic' in rules or (outcome == 'unavailable' and gate.detail == 'coorbital_unfitted_forcing'):
            planet = _coorbital_planet(row)
            return (
                f'No gate verdict: the free-element model does not apply to a co-orbital of {planet} (its forcing is the '
                'planet’s own vector, whose modes the split does not fit).',
                ('SZ.G',),
                (),
            )
        if outcome == 'planetary_residual':
            text, columns = _node_near_sentence(row, span)
            if text:
                return text, ('SZ.G',), columns
            matches = _text(row.get('zlk_gate_matches')).replace(';', ', ')
            return (
                f'A residual rate of the free elements matches a planetary one{f" ({matches})" if matches else ""}: the free ω '
                'is not told apart from a planetary rate in this record.',
                ('SZ.G',),
                (),
            )
        return GATE_DECISIVE.get(outcome, f'free-ω gate: {snake_words(outcome)}.'), ('SZ.G',), ()
    if str(v.angle_subtype) == 'phase_unresolved':
        text, columns = _phase_sentence(analysis, row)
        return text, ('G5p',), columns
    if 'centre_turned_insufficient' in rules:
        return (
            'The center turned back within less than one revolution: an unfinished slow oscillation, not a drift; the record '
            'is too short to close one cycle.',
            ('G5s',),
            (),
        )
    if outcome == 'not_applicable' and angle == 'circulation':
        return 'ω circulates: there is no libration for the free-ω gate to check.', (), ()
    return SECULAR_DECISIVE.get(angle, f'{snake_words(str(v.reason))}.'), (), ()


def episode_ei_parts(ei) -> List[str]:
    """The three tests of the e-i exchange inside the deciding episode (`zlk.episode_ei`), each the way it came out:
    'i against e 262° (82° from antiphase > 45°)', 'coherence −0.777 ≤ −0.5', 'η_H 1.24 > 0.5'."""
    off = phase_off_deg(ei.phase_deg, 180.0)
    return [
        f'i against e {ei.phase_deg:.0f}° ({compare(off, "<=", ei.phase_tol_deg, "°")} from antiphase)',
        f'coherence {compare(ei.coherence, "<=", -ei.min_coherence)}',
        f'η_H {compare(ei.eta_h, "<=", ei.h_residual_max)} (√(1−e²) cos i {"kept" if ei.eta_h <= ei.h_residual_max else "not kept"})',
    ]


def kinematic_excluded_line(analysis, settings) -> str:
    """Flag kinematic_excluded (W30 D2): no gate verdict, but the forced terms are too small to hold the perihelion and
    the e-i integral holds in the deciding episode, so the gate's abstention lowers nothing (1981: certain). The two
    comparisons as they came out; '' without the flag."""
    v, gate = analysis.verdict, getattr(analysis, 'gate', None)
    if not has_mark(v, str(Flag.KINEMATIC_EXCLUDED)) or gate is None:
        return ''
    rho = f'ρ_max {compare(gate.rho_max, ">=", settings.zlk_rho_kinematic_excluded)}'
    ei = getattr(gate, 'episode_ei', None)
    held = ''
    if ei is not None and finite(ei.eta_h):
        where = f' #{ei.episode_id}' if ei.episode_id is not None else ''
        held = f' and the e-i integral holds in the deciding episode{where} ({", ".join(episode_ei_parts(ei))})'
    level = f': the level follows the evidence ({str(v.confidence).upper()})' if str(v.confidence) else ''
    return (
        f'Why no cap without a gate verdict: the forced terms are too small to hold the perihelion ({rho}){held}, so a '
        f'kinematic libration is excluded (flag kinematic_excluded){level}.'
    )


def zlk_physics_line(analysis, row: Mapping[str, Any], turns: bool = True) -> Tuple[str, Tuple[str, ...]]:
    """The physics of a Lidov-Kozai row (card g10-8): the ranges of i and e and how many times the free ω (g − s,
    free_b_omega) turns over the record, |g − s| / (1/T). '' for a secular row. Kozai's 39.2° is not printed: it
    belongs to the hierarchical quadrupole problem, outside its domain for main-belt and planet-crossing asteroids
    (W30, the author; Gronchi & Milani 1999)."""
    if str(analysis.verdict.resonance_type) != 'lidov_kozai':
        return '', ()
    lo, hi = _row_float(row, 'ctx_inc_min_deg'), _row_float(row, 'ctx_inc_max_deg')
    e_lo, e_hi = _row_float(row, 'ctx_e_min'), _row_float(row, 'ctx_e_max')
    parts, columns = [], []
    if finite(lo) and finite(hi):
        ranges = f'i {fmt(lo)}–{fmt(hi)}°' + (f', e {fmt(e_lo)}–{fmt(e_hi)}' if finite(e_lo) and finite(e_hi) else '')
        parts.append(f'Physics: {ranges} in the record.')
        columns += ['ctx_inc_min_deg', 'ctx_inc_max_deg'] + (['ctx_e_min', 'ctx_e_max'] if finite(e_lo) and finite(e_hi) else [])
    rate = _row_float(row, 'free_b_omega')
    span = record_span_yr(analysis)
    if not turns:  # an unphysical orbit: its free elements mean nothing
        pass
    elif finite(rate) and finite(span) and span > 0:
        parts.append(f'The free ω (g − s) turns {fmt(abs(rate) * span / ARCSEC_PER_TURN)} times in the record (|g − s| / (1/T)).')
    else:
        parts.append('The free ω is not measured (no free elements).')
    return ' '.join(parts), tuple(columns)


IDENTIFY_MAX = 3  # rows named for one period seen on the angle


def _identify(period: float, low: float, high: float, resolution: float, frequencies) -> Tuple[str, str]:
    """(the FREQUENCIES rows a period seen on the angle is, nearest first and joined by ' or ', '' for none; the
    nearest row): a row matches when its |rate| lies in the frequency band of the period's range [low, high] widened
    by 1/T (frequencies closer than 1/T are one in the record, so all such rows are named). Display only."""
    f = ARCSEC_PER_TURN / period
    f_lo = ARCSEC_PER_TURN / high if finite(high) and high > 0 else f
    f_hi = ARCSEC_PER_TURN / low if finite(low) and low > 0 else f
    candidates = [item for item in frequencies if finite(item.rate) and item.rate]
    if not candidates:
        return '', ''
    candidates.sort(key=lambda item: abs(abs(item.rate) - f))
    band = (min(f_lo, f) - resolution, max(f_hi, f) + resolution)
    inside = [item.name for item in candidates if band[0] <= abs(item.rate) <= band[1]]
    return ' or '.join(inside[:IDENTIFY_MAX]), candidates[0].name


def _seen_periods(analysis, row: Mapping[str, Any], said: bool) -> List[Tuple[str, float, float, float]]:
    """(label, period, low, high) of the periods seen on the angle: the deciding period (sec_decisive_period_yr, the
    deciding librating episode's or the runs' not faster than the type's floor) with its range, and the fine scale's
    when the center scale decided. `said`: the angle's line already printed the deciding episode's period."""
    out = []
    period = _row_float(row, 'sec_decisive_period_yr')
    if finite(period):
        if finite(_row_float(row, 'ep_period_med_yr')) and abs(_row_float(row, 'ep_period_med_yr') - period) <= 1e-6 * period:
            low, high, label = _row_float(row, 'ep_period_min_yr'), _row_float(row, 'ep_period_max_yr'), 'the deciding episode'
            label += ' (above)' if said else ''
        else:
            low, high, label = _row_float(row, 'geo_period_min_yr'), _row_float(row, 'geo_period_max_yr'), 'its turning points'
        out.append((label, period, low, high))
    if _text(row.get('sec_scale')) == 'centre' and finite(_row_float(row, 'sec_fine_period_cycle_yr')):
        out.append(('the fine scale', _row_float(row, 'sec_fine_period_cycle_yr'), float('nan'), float('nan')))
    return out


UNPHYSICAL_DECISIVE = 'The elements of an unbound orbit are not those of an ellipse: neither the angle nor the free elements are read.'
UNPHYSICAL_WORDS = {'hyperbolic': 'hyperbolic', 'negative_sma': 'unbound', 'non_finite': 'lost by the integration'}


def unphysical_text(analysis, row: Optional[Mapping[str, Any]] = None) -> str:
    """Why an orbit is unphysical (`classify.util.unphysical_arrays`), with the value: 'e reaches 2.11 > 1
    (hyperbolic)' (2010 HY22), 'a falls to -506 au < 0 au (unbound)', 'a or e is not finite (…)'; '' for a physical
    row."""
    reason = str(getattr(analysis, 'unphysical_reason', '') or (row or {}).get('ctx_unphysical') or '')
    if not getattr(analysis, 'unphysical', False) and not reason:
        return ''
    row = row or {}
    series = getattr(analysis, 'series', None)
    e_max, a_min = _row_float(row, 'ctx_e_max'), _row_float(row, 'ctx_a_min_au')
    if not finite(e_max) and series is not None:
        e_max = max((float(x) for x in series.e if finite(x)), default=float('nan'))
    if not finite(a_min) and series is not None:
        a_min = min((float(x) for x in series.a if finite(x)), default=float('nan'))
    if reason == 'hyperbolic' and finite(e_max):
        return f'e reaches {compare(e_max, "<=", 1.0)} (hyperbolic)'
    if reason == 'negative_sma' and finite(a_min):
        return f'a falls to {compare(a_min, ">=", 0.0, " au")} (unbound)'
    if reason == 'non_finite':
        return 'a or e is not finite (the integration lost the orbit)'
    return f'the orbit is {UNPHYSICAL_WORDS.get(reason, snake_words(reason) or "unphysical")}'


# (question, why, ids) of the leaf of a secular or Lidov-Kozai row: the free-ω gate (ZLK), else the angle's shape
# the gate outcomes that decide a ZLK row's status (`secular.decide`); any other one leaves it to the angle's shape (555527:
# not_applicable was printed as 'Decided at 1: free-ω gate')
GATE_VERDICTS = ('confirmed', 'kinematic')
