"""The explanation of a verdict in words, for the right column of the one-page figures (layout A′).

Pure functions: an analysis (MMR, secular or Lidov-Kozai), its CSV rows and the settings in, a `VerdictText` out;
nothing is drawn here and nothing feeds a status. Every number in the text is read from the analysis' own objects or
CSV columns and every threshold from `AnalysisSettings`, so a sentence can be checked against the record
(`tests/resonances/plotting/test_verdict_text.py`).

What the text holds (the author's N1–N4 of 2026-10-16 and the fixes of 16b):

- `badge`: STATUS · angle_status (subtype) · CONFIDENCE, all three always (author, 2026-10-17) — the only place the
  status is printed; `ends`: 'ends: <final regime>' when the record ends in another regime than angle_status, drawn
  as a tag beside the plaque, not inside it; `action` (W29, the author on q05): the row's action as an amber tag between
  them ('→ extend to 40 kyr', `action_tag`), read from the action alone;
- `sentences`: what the angle does, why this status (the rule that decided it, in plain words, with its numbers),
  the confidence (what kept it from the next level, with the numbers), the action. 'Over the whole record' is said
  when the libration covers the record; 'episode' only when there are several;
- `steps` (cards g10-4/g10-4b, W28 E3): under the header 'Decided at <leaf>: <name> → STATUS' (`decided_at`), the
  steps of the status tree in their fixed order, each a question '<number> · <question> [ids]' with 'Why:' and what
  it found: step 0 the angle shape with its G rules, the branch letter on the leaf ('5d'), decided here ●, passed on ↓,
  not run ◌ (a gap, with its reason), not needed ○ (the leaf decides without it: the owner at a kinematic leaf, 'n/a:
  no force on a'), n/a · (the owner search is not applied to a 1:1); the steps after the leaf in one pale line;
  `docs` names the docs pages in one footer line;
- `checks` (cards g10-4/g10-4b, g03-5, W28 E4): every margin and cap the confidence read, each 'name [id]' with what
  was measured; a kind (`KIND_*`): the limits first (▼ LIMIT: a cap, with the failed cap-class margin it explains
  printed inside it once, `merge_limits`; never drawn ✗), then the support margins: fail ✗, should have been measured
  but was not ?, near its threshold ⚠, skipped by design – (grey, with the reason), the full ones ✓ in one line
  (`full_line`), the n/a ones · in one grey line without numbers (`na_line`); K1 without a value says why
  (kin_check_reason);
- `arithmetic`: how the level came out, two lines: 'Support: … → HIGH' (the raw level) and 'Limits: [O1] … → LOW'
  (`arithmetic_line`); `info`: tests measured but not read by this verdict;
- `bold_columns` / `grey_columns`: numbers the decision compared / numbers of a test skipped by design;
- `emphasis` (MMR, card g10-5, W28 E10): the lines of 'Numbers behind this verdict' (`number_emphasis`): DECISIVE
  comparisons of the leaf, the LIMIT caps that set the level, a CONFLICT (an objection), an UNMEASURED decisive test;
- `physics` (MMR, W20 CS11): A 'Is σ the own pendulum of <key>?' and B 'What makes the cycles?' (`physics_sections`),
  every number a column (`PhysicsRow.numbers`); the columns it prints leave the numbers below it.

A cap that brings the level down to medium or low names the other reading and what on the figure decides it
(`CAP_ALTERNATIVES`, the author's q1 of 2026-10-19); the W20 ledger entries are printed in words (`LEDGER_WORDS`).
The words (caps, reasons, flags, ledger entries, the short names of the rule ids) live in `verdict_vocab.py`; the
subtitle names the deciding rule as 'decided: circulation (C)' (`verdict_vocab.decided`, card g10-3).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from resonances.plotting.verdict_vocab import (  # noqa: F401  (the registry, also read as verdict_text.<NAME>)
    BASE_CAUSE_WORDS,
    BASE_SOURCE_WORDS,
    CAP_ALTERNATIVES,
    CAP_MARGINS,
    CAP_PREFIX,
    CAP_WORDS,
    CAPS,
    CHECK_WORDS,
    EI_WORDS,
    GATE_WORDS,
    KIN_REASON_WORDS,
    LEDGER_WORDS,
    LONG_PERIOD_MARGIN_NAMES,
    MMR_MARGIN_NAMES,
    OWNER_FLAG_WORDS,
    OWNER_REASON_WORDS,
    POTENTIAL_REASON_WORDS,
    REGIME_MARK_WORDS,
    REGIME_PHRASES,
    REJECT_WORDS,
    RULE_NAMES,
    SHORT_PERIOD_TITLES,
    STRENGTH_REASON_WORDS,
    T7_WORDS,
    THEORY_WORDS,
    CapText,
    decided,
    long_period_id,
    named,
    rule_name,
)
from resonances.resonance.analysis.confidence import CAP_CEILING, Cap
from resonances.resonance.analysis.geometry import LEG_EDGE_START, RejectReason
from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.mmr.analyse import CENTRE_SCALE_UNION, QS_STALL_ENTRY, SHORT_PERIOD_CAP, SHORT_PERIOD_HARMONIC
from resonances.resonance.analysis.mmr.decide import (
    ALIAS_A,
    B1_AMPLITUDE,
    C5_BRANCH,
    C5_FAR,
    DIRECT_BRANCH,
    EXTEND_LEAVES,
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
    LENGTH_FORCE_WINDOWS,
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
    force_windows_short,
    kinematic_f_applies,
    last_episode_at_end,
    period_floor,
    strength_upper_name,
    taken_direct_law,
    taken_harmonic_law,
)
from resonances.resonance.analysis.mmr.models import planet_letter
from resonances.resonance.analysis.mmr.quasi_satellite import QS_FLOOR, QS_VISIT
from resonances.resonance.analysis.orbit_type import centre_moves
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.secular.decide import librating_episodes as secular_librating_episodes
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
# Steps: decided here, passed on to the next step, not run, not needed, n/a (the glyph of the checks' n/a: one glyph,
# one meaning); a step not reached has no icon (its name is pale)
STEP_ICONS = {
    StepState.LEAF: '●',
    StepState.PASSED: '↓',
    StepState.NOT_RUN: '◌',
    StepState.NOT_NEEDED: '○',
    StepState.NOT_APPLICABLE: '·',
    StepState.NOT_REACHED: '',
}
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


# The slots of the MMR explanation (cards g10-2 style (i), g10-8; W28 E8), in their drawing order: the verdict in plain
# words, what σ does, the argument that decided, an objection that did not decide, the cause (who makes the cycles),
# what would change the verdict (the level, its limit, the other reading, the action), the radial overlap
SLOT_VERDICT, SLOT_SIGMA, SLOT_DECISIVE, SLOT_OBJECTION = 'verdict', 'sigma', 'decisive', 'objection'
SLOT_CAUSE, SLOT_CHANGE, SLOT_ORBIT = 'cause', 'change', 'orbit'
# secular and ZLK (W28 E9, cards g09-5 C, g10-8): the time scales of the record (1/T, the search window, the periods
# seen on the angle and what they are); the slot sits after the cause
SLOT_SCALE = 'scale'
# the minimal policy (W29 SPEC §7): the explanation opens with four lines (`policy_block`): status · confidence · reason
# (origin / owner); the basis of the level in words; the flags; the v2 tree's verdict
SLOT_POLICY = 'policy'
SLOTS = (SLOT_POLICY, SLOT_VERDICT, SLOT_SIGMA, SLOT_DECISIVE, SLOT_OBJECTION, SLOT_CAUSE, SLOT_SCALE, SLOT_CHANGE, SLOT_ORBIT)


@dataclass
class Bullet:
    """One line of the slotted explanation (`mmr_bullets`, `long_period_bullets`): its slot (`SLOTS`), the text, the
    rule/check ids it reads, and the CSV columns whose numbers it prints (secular/ZLK: those leave the numbers table for
    a pointer, W20 CS11: no number twice). `short`: the form the compact explanation draws ('' = `text`), where a fact
    the steps print word for word is pointed to instead of repeated (W28 FIX-plot: `candidate_driver_short`)."""

    slot: str
    text: str
    ids: Tuple[str, ...] = ()
    columns: Tuple[str, ...] = ()
    short: str = ''


# The emphasis of a number (card g10-5, W28 E10): the block 'Numbers behind this verdict' over the numbers table prints
# these lines framed and tagged; each emphasised column leaves the table for a pointer '↑ decisive [B1]' (W20 CS11: no
# number twice). decisive: the comparison of the leaf that set the status; limit: the cap that sets the shown level;
# conflict: a measured check that fails and that the verdict did not read; unmeasured: a decisive test without a value.
EMPHASIS_DECISIVE, EMPHASIS_LIMIT, EMPHASIS_CONFLICT, EMPHASIS_UNMEASURED = 'decisive', 'limit', 'conflict', 'unmeasured'
EMPHASES = (EMPHASIS_DECISIVE, EMPHASIS_LIMIT, EMPHASIS_CONFLICT, EMPHASIS_UNMEASURED)  # also the block's order


@dataclass(frozen=True)
class EmphasisPart:
    """One number of an emphasised line: its CSV column and the comparison as it came out ('0.0954 < 0.1');
    `labelled`: the figure prints the column's label before it (False when `text` names the quantity itself)."""

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

    @property
    def pointer(self) -> str:
        """What the table prints instead of the number: '↑ decisive [B1]'."""
        return f'↑ {self.kind} [{self.rule_id}]'


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
    Printed as 'name [id]' (`label`) and the measurement; a limit as 'LIMIT: short name [id] → LOW' (`limit_heading`)."""

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
    decided_at: str = ''  # the steps' header: 'Decided at 5d → 6: force measured, then judged by behaviour → TRANSIENT'
    docs: str = ''  # one footer line: the docs pages that hold the steps, the rules and the margins (`DOCS_MMR`)
    # MMR (cards g10-2, g10-8, W28 E8): the explanation as slotted bullets, drawn instead of `sentences`; the sentences
    # stay the plain-paragraph form (secular and ZLK draw them until their slots come)
    bullets: List[Bullet] = field(default_factory=list)
    # MMR (card g10-5, W28 E10): the lines of 'Numbers behind this verdict', in the block's order (`number_emphasis`)
    emphasis: List[Emphasis] = field(default_factory=list)
    # W29 (the author on q05, 2026-11-02: the action belongs in the column's header): the row's action as the plaque's
    # tag, '→ extend to 40 kyr' (`action_tag`); '' for action none
    action: str = ''

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


def regime_phrase(regime: Any) -> str:
    """'σ circulates', 'the record is too short to tell what σ does': a regime (`AngleStatus` value) in plain words."""
    text = _text(regime)
    return REGIME_PHRASES.get(text, f'σ is in the regime {text.replace("_", " ")}')


def status_head(status: Any) -> str:
    """The status opening its sentence: 'Non-resonant', 'Near-separatrix', 'Resonant'."""
    return str(status).replace('_', '-').capitalize()


def _plural(n: int, word: str, plural: str = '') -> str:
    return f'{n} {word if n == 1 else (plural or word + "s")}'


# ------------------------------------------------------------------------------- the minimal policy (W29 SPEC §2.3, §7)

# the words of the `confidence_basis` grammar 'N=<n>(<source>);M=<m>;V=<v>;S=<s>;veto=<name|->[:thin]'
BASIS_N_SOURCE = {'episode': 'the episode', 'union': 'the union of episodes', 'cover': 'the whole record', 'leg': 'one 1:1 leg'}
BASIS_M_WORDS = {
    'full': 'amplitude clear of the separatrix',
    'thin': 'amplitude near the separatrix',
    'not_measured': 'amplitude margin not measured',
}
BASIS_V_WORDS = {
    'clean': 'objections measured, none',
    'partial': 'objections partly measured',
    'not_measured': 'a needed objection not measured',
    'ambiguous': 'a measured objection',
}
BASIS_S_WORDS = {'full': 'filter-stable', 'thin': 'filter-sensitive'}


def policy_minimal(verdict) -> bool:
    """The verdict was set by the minimal policy layer (its `policy` field; '' in older files and under v2)."""
    return str(getattr(verdict, 'policy', '') or '') == POLICY_MINIMAL


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


def basis_words(basis: Any) -> str:
    """The `confidence_basis` cell in words: 'N 3.5 cycles (the episode) · amplitude clear of the separatrix · objections
    measured, none · filter-stable · no veto'; an unknown part is printed as written; '' for an empty cell."""
    parts = dict(item.split('=', 1) for item in str(basis or '').split(';') if '=' in item)
    if not parts:
        return ''
    out = []
    match = re.fullmatch(r'([^()]*)\(([^()]*)\)', parts.get('N', ''))
    if match and finite(_float(match.group(1))):
        source = BASIS_N_SOURCE.get(match.group(2), match.group(2))
        out.append(f'N {fmt(_float(match.group(1)))} cycles' + (f' ({source})' if source not in ('', '-') else ''))
    elif match:  # 'nan(-)': no cycles were counted (a vetoed or circulating row)
        out.append('no cycles counted')
    elif 'N' in parts:
        out.append(f'N {parts["N"]}')
    for key, words in (('M', BASIS_M_WORDS), ('V', BASIS_V_WORDS), ('S', BASIS_S_WORDS)):
        if key in parts:
            out.append(words.get(parts[key], f'{key} {parts[key]}'))
    veto = parts.get('veto', '')
    if veto:
        name, _, thin = veto.partition(':')
        out.append('no veto' if name in ('', '-') else f'veto {name}' + (' (thin)' if thin else ''))
    return ' · '.join(out)


def policy_reason_words(verdict) -> str:
    """The reason first (SPEC §7: GK3 reads 'induced by 5M-2' before the angle word), with the origin and the owner in
    parentheses: 'induced by 5M-2 (origin induced)', 'geometric episode (origin unknown)'."""
    reason, owner, origin = str(verdict.reason or ''), str(getattr(verdict, 'owner', '') or ''), str(verdict.libration_origin or '')
    words = reason.replace('_', ' ')
    if reason == 'induced_libration' and owner:
        words, owner = f'induced by {owner}', ''
    extra = ' / '.join(part for part in (f'origin {origin}' if origin else '', f'owner {owner}' if owner else '') if part)
    return words + (f' ({extra})' if extra else '')


def policy_block(verdict) -> List['Bullet']:
    """The four lines that open the explanation under the minimal policy (W29 SPEC §7): STATUS · CONFIDENCE · the
    reason (origin / owner); 'basis:' the `confidence_basis` in words (with the counted episodes); 'flags:'; 'v2:' what
    the v2 tree said (`v2_verdict`). [] for a row not under the policy."""
    if not policy_minimal(verdict):
        return []
    head = ' · '.join(
        part for part in (status_head(verdict.status).upper(), str(verdict.confidence).upper(), policy_reason_words(verdict)) if part
    )
    basis = basis_words(getattr(verdict, 'confidence_basis', ''))
    counted = getattr(verdict, 'n_counted_episodes', None)
    if counted is not None and finite(counted):
        basis = '; '.join(part for part in (basis, _plural(int(counted), 'counted episode')) if part)
    flags = ', '.join(str(flag) for flag in (verdict.flags or ())) or 'none'
    v2 = ' · '.join(part for part in str(getattr(verdict, 'v2_verdict', '') or '').split('/') if part) or 'not recorded'
    return [
        Bullet(SLOT_POLICY, head),
        Bullet(SLOT_POLICY, f'basis: {basis or "not recorded"}'),
        Bullet(SLOT_POLICY, f'flags: {flags}'),
        Bullet(SLOT_POLICY, f'v2: {v2}'),
    ]


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
        short_name = rule_name(f'{CAP_PREFIX}{cap}')
        rows.append(
            CheckRow(
                f'cap:{cap}',
                words,
                CheckState.FAIL,
                counts=f'cap ≤ {cap_ceiling(cap)}',
                alternative=alternative,
                look=look,
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


def limit_heading(row: CheckRow, level: str) -> str:
    """'LIMIT: earlier a-wave [O1] → LOW' for a cap that sets the shown level, '… → at most MEDIUM' for one above it;
    'LIMIT: name [id]' for a failed cap-class margin whose cap is not on record."""
    ceiling = row.counts.split('≤')[-1].strip() if '≤' in row.counts else ''
    if not ceiling:
        return f'LIMIT: {row.label}'
    to = ceiling.upper() if ceiling == str(level) else f'at most {ceiling.upper()}'
    return f'LIMIT: {row.label} → {to}'


def full_line(rows: Sequence[CheckRow]) -> str:
    """The full margins in one line (card g10-4b): 'full: one pendulum [L5]; fixed centre [L6]; …' ('' for none)."""
    return f'full: {"; ".join(row.label for row in rows)}' if rows else ''


def na_line(rows: Sequence[CheckRow]) -> str:
    """The n/a margins in one grey line without numbers: 'not for this record: σ̇ follows a − a* [L3], …'."""
    return f'not for this record: {", ".join(row.label for row in rows)}' if rows else ''


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
    """Reading order (card g10-4b): the limits first (the cap that sets the level first, `cap_rows`), then the support
    margins worst first (failed, missing, thin, full, skipped by design), the n/a ones last."""
    return sorted(rows, key=lambda row: (not row.is_limit, row.kind == KIND_NA, STATE_ORDER.index(row.state)))


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


def confidence_parts(verdict, checks: Sequence[CheckRow]) -> Tuple[str, str, str, str]:
    """The confidence sentence in its four parts: the level and what capped it ('Medium, capped: … (number)'), what the
    level would be without the cap ('Without the cap it would be high: …'), the other reading of a lowering cap (CS11,
    `CAP_ALTERNATIVES`) and the calibration cell; '' for a part that does not apply. Without a cap the first part says
    what kept the level from the next ('High: nothing failed, but …'). Marks the cap whose other reading is said."""
    level, raw = str(verdict.confidence), str(verdict.confidence_raw)
    if not level:
        return '', '', '', ''
    margins = [row for row in checks if not row.is_limit]
    failed = [row for row in margins if row.counts == COUNTS_FAILED]
    soft = [row for row in margins if row.counts in (COUNTS_THIN, COUNTS_NOT_MEASURED)]
    caps = [row for row in checks if row.is_cap]
    none_failed = 'no support test failed' if any(row.is_limit for row in checks) else 'nothing failed'
    if failed:
        raw_text = f'{_plural(len(failed), "support test")} failed: {_names(failed)}'
    elif raw == 'high':
        verb = 'is' if len(soft) == 1 else 'are'
        raw_text = (
            f'{none_failed}, but certain needs every margin full and {len(soft)} {verb} not: {_bare_names(soft)}'
            if soft
            else f'{none_failed}, but a comparison of the decision lies near its threshold'
        )
    else:
        raw_text = 'every margin is full'
    setting_caps = [row for row in caps if row.counts.split()[-1] == level]
    head, without, other = f'{level.capitalize()}: {raw_text}.', '', ''
    if setting_caps and level != raw:
        first = setting_caps[0]
        source = next((row for row in margins if row.key == CAP_MARGINS.get(first.key[4:]) and row.detail), None)
        if first.merged and first.detail:  # the failed margin is printed inside the cap row (`merge_limits`)
            source = first
        rest = len(setting_caps) - 1
        others = f' (and {rest} more cap{"s" if rest > 1 else ""} at the same level)' if rest else ''
        number = f' ({source.detail})' if source is not None else (f' ({first.short})' if first.short else '')
        head = f'{level.capitalize()}, capped: {first.name}{number}{others}.'
        without = f'Without the cap it would be {raw}: {raw_text}.'
        if first.alternative:
            where = f'; the figure decides it in {first.look}' if first.look else ''
            other = f'The other reading: {first.alternative}{where}.'
            first.said = True  # its row does not repeat it
    return head, without, other, calibration_note(verdict)


def confidence_sentence(verdict, checks: Sequence[CheckRow]) -> str:
    """Status → confidence: the level, what capped it, and what kept it from the next level, with the numbers. 'Nothing
    failed' only when no limit failed either (card g03-5); with a limit on record the support margins are said apart
    ('no support test failed')."""
    if policy_minimal(verdict) and str(verdict.confidence):  # W29: the level is the evidence's, said in `policy_block`
        return f'{str(verdict.confidence).capitalize()} from the evidence: {basis_words(getattr(verdict, "confidence_basis", ""))}.'
    return ' '.join(part for part in confidence_parts(verdict, checks) if part)


ARITHMETIC_SUPPORT, ARITHMETIC_LIMITS = 'Support: ', 'Limits: '


def arithmetic_line(verdict, checks: Sequence[CheckRow]) -> str:
    """How the level came out, in two lines (cards g10-4/g10-4b, g03-5): the support margins and the raw level they
    give ('Support: no counted failures, 4 thin, 5 not measured → HIGH'; ≥ 2 failed → low, 1 → medium, thin or not
    measured → high, all full → certain: `confidence.raw_level`), then the limits and the shown level ('Limits: earlier
    a-wave [O1] (a-wave before/inside 0.865 ≥ 0.5) → LOW'; a limit above the shown level with its ceiling; 'none')."""
    if not str(verdict.confidence):
        return 'No confidence level: uncertain and invalid verdicts carry none.'
    if policy_minimal(verdict):  # the level is the evidence's (W29 `policy.level`): no support count, no caps
        return f'Level from the evidence: {basis_words(getattr(verdict, "confidence_basis", "")) or "not recorded"} → {str(verdict.confidence).upper()}'
    margins = [row for row in checks if not row.is_limit]
    failed = sum(1 for row in margins if row.counts == COUNTS_FAILED)
    thin = sum(1 for row in margins if row.counts == COUNTS_THIN)
    unmeasured = sum(1 for row in margins if row.counts == COUNTS_NOT_MEASURED)
    parts = [f'{failed} failed' if failed else 'no counted failures']
    parts += [f'{thin} thin'] if thin else []
    parts += [f'{unmeasured} {COUNTS_NOT_MEASURED}'] if unmeasured else []
    support = ', '.join(parts) if failed or thin or unmeasured else 'every margin full'
    level = str(verdict.confidence)
    limits = []
    for row in checks:
        if not row.is_limit:
            continue
        item = row.label + (f' ({row.short})' if row.short else '')
        ceiling = row.counts.split('≤')[-1].strip() if '≤' in row.counts else ''
        limits.append(item + (f' (≤ {ceiling})' if ceiling and ceiling != level else ''))
    return (
        f'{ARITHMETIC_SUPPORT}{support} → {str(verdict.confidence_raw).upper()}\n'
        f'{ARITHMETIC_LIMITS}{"; ".join(limits) or "none"} → {level.upper()}'
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


ACTION_TAGS = {'spot_check': '→ spot check', 'review': '→ review'}


def action_tag(verdict) -> str:
    """The action as the plaque's tag (W29, the author on q05: TX1's 'extend' was missed in the sentences): '→ extend to
    40 kyr', '→ resample to 0.5 yr', '→ review', '→ spot check'; '' for action none. Read from the action alone: the
    flag extend_time and extend_to_years of a row whose action is not extend print nothing here."""
    action = str(verdict.action)
    if action == 'extend':
        to = getattr(verdict, 'extend_to_years', None)
        return f'→ extend to {years(to)}' if finite(to) else '→ extend'
    if action == 'resample':
        step = getattr(verdict, 'resample_step_yr', None)
        return f'→ resample to {fmt(step)} yr' if finite(step) else '→ resample'
    return ACTION_TAGS.get(action, '')


def _short_period_resample(verdict) -> bool:
    """A resample of a verdict with a level whose cap short_period_aliased holds it down (alias and undersampled are
    invalid and carry no level)."""
    return bool(str(verdict.confidence)) and has_mark(verdict, 'short_period_aliased')


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
    def counted(self) -> List[Mapping[str, Any]]:
        """The accepted runs without those the status does not count and no other mark names (`uncounted_rows`: below
        the period floor, the centre turns; W28 C11, card g02-9 A)."""
        uncounted = [id(row) for row in uncounted_rows(self.rows)]
        return [row for row in self.accepted if id(row) not in uncounted]

    @property
    def longest(self) -> Optional[Mapping[str, Any]]:
        return max(self.counted, key=_duration) if self.counted else None

    def covered_fraction(self) -> float:
        if not finite(self.span_yr) or self.span_yr <= 0:
            return float('nan')
        return sum(_duration(row) for row in self.counted) / self.span_yr

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


def _kinematic_reason(f: MMRFacts, field_name: str) -> Tuple[CheckState, str]:
    """K1 or K1e without a value: why the correlation has none (`kinematics.check_reason`, kin_check_reason, for K1;
    kin_k1_episode_reason for K1e, card g10-4b) and where the figure shows σ against p·ϖ (panel (j)); 'not measured'
    when the record keeps no reason."""
    kinematics = f.a.kinematics
    code = _text(getattr(kinematics, field_name) if kinematics is not None else f.row.get(f'kin_{field_name}'))
    if not code:
        return CheckState.MISSING, 'not measured'
    words = KIN_REASON_WORDS.get(code, code.replace('_', ' ')).format(min_points=f.s.mmr_kin_min_points)
    return CheckState.MISSING, f'no value — {words}; compare σ and p·ϖ in (j)'


SKIP_REASONS = {
    'C4': _skip_c4,
    'K1': lambda f: _kinematic_reason(f, 'check_reason'),
    'K1e': lambda f: _kinematic_reason(f, 'k1_episode_reason'),
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
        f's {fmt(law.scale_min)}–{fmt(law.scale_max)} {"within" if inside else "outside"} [{fmt(lo, 2)}, {fmt(hi)}], '
        f'above the shifted null by {compare(law.null_gain, ">=", s.mmr_force_null_gain)}, {_plural(int(law.n_windows), "window")}; '
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


def _potential_detail(f: MMRFacts) -> Tuple[str, str]:
    energy, s = f.a.energy, f.s
    law = getattr(energy, 'potential_law', None)  # R0 printed from the law it was read from (W20 round D)
    if law is not None:
        r2 = law.energy_r2
        return f'1:1 energy integral R² {compare(r2, ">=", s.mmr_force_r2_fail)}', f'R² ≥ {fmt(s.mmr_force_r2_fail)}'
    if energy is None:
        return '', ''
    need = f'≥ {percent(s.separatrix_vevf_fraction)} of the windows with r ≥ {fmt(s.separatrix_vevf_r_min)}'
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


def refused_owner(f: MMRFacts) -> bool:
    """An accepted owner the decision did not take (F7 outside σ's episode, or P2): a neighbour, a candidate driver."""
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
        return ('no candidate', '') if not owner.n_candidates else (f'{_plural(owner.n_candidates, "candidate")}, none fits', '')
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
        return COORBITAL_OWNER, '', ''
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
        if refused_owner(f):  # g10-8: an owner the decision did not take is a candidate, never 'drives a'
            return f'candidate driver {owner.accepted.spec.key}{approximate}'
        return f'owner {owner.accepted.spec.key} drives a{approximate}'
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
    return, edges, the 1:1 profile floor and a turn read in the edge trim)."""
    parts = [centre_scale_clause(f), quasi_satellite_clause(f), subfloor_clause(f), return_clause(f)] + edge_clauses(f)
    parts += [profile_floor_clause(f), edge_turn_clause(f)]  # W28 C3, C4 (1:1 only)
    return '; '.join(part for part in parts if part)


def subfloor_clause(f: MMRFacts) -> str:
    """Rule G1f (W28 B5, card g08-2 B; `analyse._subfloor_pass`, columns geo_subfloor_*), '' when its pass was not
    kept: 'a two-body cycle of 105 yr, below the 150-yr floor, read again at the floor 35.3 yr of its own averaged
    potential (T0_th 112 yr / √10) and kept: F by direct averaging, the centre and the force law (L4) confirm it'."""
    if not (finite(f.row.get('geo_subfloor_kept')) and float(f.row['geo_subfloor_kept']) == 1):
        return ''
    period, t0, floor = (f.row.get(f'geo_subfloor_{name}') for name in ('period_yr', 't0_theory_yr', 'floor_yr'))
    entry = next((e for name, e in f.path.items() if name.startswith('G1f floor')), None)  # its threshold: the own floor
    own_text = f'the {years(entry.threshold)} floor' if entry is not None and finite(entry.threshold) else 'the floor'
    return (
        f'a two-body cycle of {years(period)}, below {own_text}, read again at the floor {years(floor)} of its own averaged '
        f'potential (T0_th {years(t0)} / √{fmt(f.s.mmr_strength_check_ratio)}) and kept: F by direct averaging, the '
        'centre and the force law (L4) confirm it'
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
UNCOUNTED_TAGS = {RejectReason.BELOW_PERIOD_FLOOR: ' < floor', RejectReason.CENTRE_TURNS: ' centre turns'}
# geo_extend_time_from (C14, `decide.RecordEdges.sources`): what set the flag extend_time -> words ({edge}: start/end)
EXTEND_SOURCE_WORDS = {
    'slowdown': 'σ_f slows to {final} after circulating',
    'reflection': 'σ turns back once at the {edge}',
    'leg': 'a clean 1:1 leg is cut by the {edge}',
    'half': 'a half-swing that follows the law is cut by the {edge}',
}
EXTEND_DIRECTIONS = {'forward': ('forward', 'ends'), 'backward': ('backward', 'starts'), 'both': ('both ways', 'starts and ends')}
TURN_SITE_WORDS = {'planet': 'the planet’s', 'l3': 'L3’s'}


def interval(t0: Any, t1: Any) -> str:
    """'586–665 yr', '28.5–32.3 kyr': a span in the unit of its end."""
    if not (finite(t0) and finite(t1)):
        return '—'
    scale, unit = next(((s, u) for s, u in ((MYR, 'Myr'), (KYR, 'kyr')) if abs(float(t1)) >= s), (YEAR, 'yr'))
    return f'{fmt(float(t0) / scale)}–{fmt(float(t1) / scale)} {unit}'


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
        words = LEG_DISCREPANCY_WORDS.get(why, why.replace('_', ' '))
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


def _single_swing_sentence(f: MMRFacts) -> str:
    """Why a single-swing leaf gave this status: transient (one clean leg), uncertain (one leg or half-swing with a
    discrepancy, or between passages) or uncertain/baseline_needed (cut by the record's edge: with the next check)."""
    clause, reason = single_swing_clause(f), str(f.v.reason)
    if reason == 'baseline_needed':
        return f'Uncertain: {clause}; {_plain(_baseline_sentence(f), f.v.status)}'
    if reason == 'dynamic_episode':
        return f'Transient: {clause}. Transient, not resonant, because one leg covers part of the record only.'
    return f'{status_head(f.v.status)}: {clause}.'


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
    leaf = Step('single', rule_name(f.rule), StepState.LEAF, f'{single_swing_clause(f)} → {f.v.status}')
    return [replace(circulation, outcome=text), leaf]


def extend_flag_sentence(f: MMRFacts) -> str:
    """The flag extend_time (C14, card g01-7 A: 'а нет ли возможности дать дополнительный флаг extend_time?'): 'Extend:
    a longer integration forward (next check: 2 kyr) would show how the record ends (σ_f slows to an arc after
    circulating; a clean 1:1 leg is cut by the end).' The next check (extend_to_years, round_up(2T)) is named only when
    the action is not extend (the action sentence names it then); '' without the flag.
    Under the minimal policy the sentence reads the action, never the flag alone (P5; W29 S1, 469333 7N-17 advised
    'backward' at action spot_check): an extension is advised only when the action is extend, and only forward, for the
    end-edge causes; a cause the action does not carry is named as a flag without advice (`MINIMAL_FLAG_ONLY`)."""
    v = f.v
    direction = _text(getattr(v, 'extend_direction', ''))
    if direction not in EXTEND_DIRECTIONS:
        return ''
    final = {'arc': 'an arc', 'stall': 'a stall'}.get(_text(v.final_regime), 'an arc or a stall')
    causes = []  # (edge, words)
    for token in _text(f.row.get('geo_extend_time_from')).split(';'):
        first, _, second = token.partition(':')
        edge, kind = (first, second) if first in ('start', 'end') else (second, first)
        edge = edge or {'forward': 'end', 'backward': 'start'}.get(direction, '')
        words = EXTEND_SOURCE_WORDS.get(kind)
        if words:
            causes.append((edge, words.format(final=final, edge=edge or 'edge')))
    if policy_minimal(v):
        return _minimal_flag_sentence(str(v.action) == 'extend', causes)
    causes = [words for _, words in causes]
    way, ends = EXTEND_DIRECTIONS[direction]
    check = f' (next check: {years(v.extend_to_years)})' if str(v.action) != 'extend' and finite(v.extend_to_years) else ''
    because = f' ({"; ".join(dict.fromkeys(causes))})' if causes else ''
    return f'Extend: a longer integration {way}{check} would show how the record {ends}{because}.'


# the flag extend_time under the minimal policy, for causes the action does not carry (a start-edge feature, or any
# feature of a row whose action is not extend): named, not advised (P5)
MINIMAL_FLAG_ONLY = (
    'Flag extend_time ({causes}): no extension; the action extends only an unfinished libration at the record’s end or '
    'a deciding period too long for the record.'
)


def _minimal_flag_sentence(extend: bool, causes: Sequence[Tuple[str, str]]) -> str:
    """extend_flag_sentence under the minimal policy: 'Extend: a longer integration forward would show how the record
    ends (…).' for the end-edge causes of an extend row, then `MINIMAL_FLAG_ONLY` for the others; '' without causes."""
    advised = list(dict.fromkeys(words for edge, words in causes if extend and edge == 'end'))
    flagged = list(dict.fromkeys(words for _, words in causes if words not in advised))
    parts = []
    if advised:
        parts.append(f'Extend: a longer integration forward would show how the record ends ({"; ".join(advised)}).')
    if flagged:
        parts.append(MINIMAL_FLAG_ONLY.format(causes='; '.join(flagged)))
    return ' '.join(parts)


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
    return _float(f.row.get('flt_period_floor_yr'))


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
            parts.append(f'{name} {interval(row.get("t_start_yr"), row.get("t_end_yr"))} (its centre runs a whole revolution inside it)')
            shorts.append(f'{name} centre turns')
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
    v, s = f.v, f.s
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
        what = 'stalls at the top in each revolution (pattern B)' if subtype == 'pattern_b' else 'slips over the top (pattern A)'
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
    ripple = ripple_clause(f)
    if slow_swing_decided(f):
        return '; '.join(part for part in (ripple, slow_swing_text(f), orbit_period_clause(f)) if part) + ' → insufficient_time (arc)'
    if subtype == 'arc' and turn is not None and ripple:
        return (
            f'{ripple}; outside it the slow centre turns back once, within {fmt(turn.value / (2 * math.pi), 2)} rev: part '
            'of one slow swing, where a drift and a slow libration look alike → insufficient_time (arc)'
        )
    if subtype == 'arc' and turn is not None:
        return (
            f'{tp} turning points ride on a slow centre that turns back once, within {fmt(turn.value / (2 * math.pi), 2)} rev: '
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
    """G5a's last entry when it held (the slow centre turns back once within a revolution), else None."""
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
    """G5w decided the angle (W28 B6c: `analyse._decide_angle` fires slow_swing_in_well exactly then)."""
    if getattr(f, 'v', None) is None:
        return False
    return str(f.v.angle_status) == 'insufficient_time' and str(Rule.SLOW_SWING_IN_WELL) in f.v.rules_fired


def slow_swing_text(f: MMRFacts) -> str:
    """G5w (W28 B6c, card g07-5 A; columns geo_slow_swing_*): the slow centre turns back twice and the half-swing
    between the turns is one of this resonance's own well: the slow identity, the a swing and the period 2(t2 − t1)
    against the directly averaged pendulum at its amplitude (T0_th F(A)), the two ratios within [1/√x, √x]."""
    row, s = f.row, f.s
    lo, hi = _strength_band(f)

    def value(name: str):
        return row.get(f'geo_slow_swing_{name}')

    slope = value('identity_slope_ratio')
    offset = abs(float(slope) - 1) if finite(slope) else None
    rng = row.get('geo_centre_range_rad')
    within = f', within {fmt(float(rng) / (2 * math.pi), 2)} rev' if finite(rng) else ''
    return (
        f'the slow centre turns back twice ({years(value("t0_yr"))}, {years(value("t1_yr"))}{within}): the '
        f'{fmt(value("amp_deg"))}° half-swing between them is this resonance’s well: slow identity |slope/c − 1| '
        f'{compare(offset, "<=", s.mmr_identity_slope_tol)}, corr {compare(value("identity_corr"), ">=", s.mmr_identity_corr_min)}; '
        f'a swing {fmt(value("a_ratio"))} × the pendulum’s, period {years(value("period_yr"))} = {fmt(value("period_ratio"))} × '
        f'T0_th F(A) (T0_th {years(value("t0_theory_yr"))}), both in [{fmt(lo, 2)}, {fmt(hi)}]'
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
# the marks of accepted runs that are no episodes of the angle and no forced wave found (W28 B2, B6b): the name of the
# ground (the angle's strength cannot make the period: card g11-1 A) and an auxiliary interval (card g08-4 A)
STRENGTH_MARK = 'forced (strength)'
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
        text = f'librating stretches, a ripple on a centre that {moves} (step 3) → circulation'
        return _shape(text)
    forced = '; '.join(part for part in (_forced_wave_clause(f), not_counted_clause(f, short=True)) if part)
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
    source = v.owner or v.resonant_argument  # A3a: a neighbour makes no wave of this angle
    owner = f' (its wave comes from {source})' if source else ''
    leg = leg_clause(f)  # W28 C6: a 1:1 leg whose turns both scatter is a near miss of a capture (near-episode)
    scatter = f'; {leg}' if leg.startswith('its best 1:1 leg') else ''
    return [Step('circ', 'circulation', StepState.LEAF, f'circulation → {v.status}{owner}{scatter}')]


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
            claim = _identity_claim(owner) if identity_route(owner) else f'neighbour {key} ({r2})'
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
        return Step('owner', title, StepState.LEAF, f'{drives} → {v.status}, induced')
    if str(owner.check) in ('pass', 'fail'):
        best = _best_candidate(owner.candidates)
        text = f'{_plural(owner.n_candidates, "candidate")}, none drives a'
        if best is not None:
            text += f' (best {best.spec.key}: {_owner_rejection(f, best)})'
        return Step('owner', title, StepState.PASSED, text)
    return Step('owner', title, StepState.NOT_RUN, f'not run: {why_not_measured(f, "W")[1]}')


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
        return Step('drift', '4. drift', StepState.NOT_APPLICABLE, f'n/a: the centre does not drift ({angle})')
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
COORBITAL_OWNER = 'the owner search is not applied to a 1:1'  # the project's rule (CLAUDE.md), not 'a 1:1 has no owner'
NO_FORCE_ON_A = 'no force on a'
OTHER_NOT_TESTED = 'other resonances not tested'  # the name of W when the owner search gave no answer
# a kinematic leaf: a does not carry the swing, so no resonance (own or other) acts on it
KINEMATIC_LEAVES = ('3', '3b5', '3k2')
# key -> (number, question, why): the steps of the status tree as questions (docs/physics/status-tree.md#steps-in-order;
# card g10-4b, extra_ru §2). 'Why' is said for a step that ran (passed or decided)
STEP_QUESTIONS = {
    'shape': ('0', 'Angle shape', ''),
    'sub': ('1', 'Does a sibling angle hold the centre?', 'a multiplet member may be the real librator.'),
    'owner': ('2', 'Does another resonance drive a?', 'a neighbour’s force on a makes σ swing too (an induced libration).'),
    'kin': ('3', 'Does a swing with σ?', 'a resonance turns σ back through a, so a must carry the swing σ needs.'),
    'drift': ('4', 'Does the centre drift without a cause?', 'a drifting centre is no libration unless a mechanism explains it.'),
    'force': ('5', 'Is it σ’s own force?', 'its own pendulum makes a obey its force law and the energy fit its potential.'),
    'zone': ('6', 'How does σ behave?', 'with the own force measured (or nothing objecting), the behaviour gives the status.'),
}
# the short names of the steps, for the one line of the steps after the leaf ('4–6 · drift, own force, behaviour')
STEP_SHORT = {
    'sub': 'siblings',
    'owner': 'another resonance',
    'kin': 'a swings',
    'drift': 'drift',
    'force': 'own force',
    'zone': 'behaviour',
}
# a leaf of the circulation branch (rec_rule) -> its question
CIRCULATION_QUESTIONS = {
    'C5': 'Can the alias exclude a libration?',
    'R0': 'Do the turning points come from a potential?',
    'C': 'Circulation',
    'G4c': 'Is its one leg near the planet a capture?',  # W28 C6, q-4 A
    'G4h': 'Is its one half-swing the resonance’s own?',  # W28 C6, q-5 i
}
# the footer line (a PNG has no links): the pages and sections that hold the steps, the G rules, the margins and caps
DOCS_MMR = (
    'Docs: steps — physics/status-tree.md#steps-in-order; G rules — physics/geometry.md; margins and caps — '
    'physics/confidence.md; O1 — physics/response.md; names [id] — libration-analysis.md#names'
)
DOCS_LONG_PERIOD = (
    'Docs: steps — physics/secular-zlk.md; margins and caps — physics/confidence.md; names [id] — libration-analysis.md#names'
)


def shape_rule_ids(f: MMRFacts) -> Tuple[str, ...]:
    """The G rules behind the angle word of step 0 (`decide.decide_angle`), read from the angle word and the decision
    path: G0 unreadable; G1c/G1e/G1l (a jump of a), G1q (the quasi-satellite floor); G2 episodes (G2u joined, G2c at
    the scale of the centre, G2a cut by a step of the action); G3 the separatrix patterns; G4/G4b a centre that turns;
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
    """The steps after the leaf as one pale line ('4–6 · drift, own force, behaviour: not reached'), no outcome text."""
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


def decided_at(rule: str, status: Any, steps: Sequence[Step]) -> str:
    """The header of the steps: 'Decided at 5d → 6: force measured, then judged by behaviour → TRANSIENT' (rec_rule and
    its short names, `verdict_vocab.rule_name`); without a rule (secular, ZLK), the leaf step's number and label:
    'Decided at 1: free-ω gate → NON-RESONANT'."""
    head = status_head(status).upper()
    rule = str(rule or '')
    if rule:
        names = [rule_name(part, leaf=True) or 'unmapped rule' for part in rule.split('/')]
        return f'Decided at {rule.replace("/", " → ")}: {", then ".join(names)} → {head}'
    leaf = next((step for step in steps if step.state == StepState.LEAF), None)
    if leaf is None:
        return ''
    where = f'{leaf.number}: ' if leaf.number else ''
    return f'Decided at {where}{leaf.title} → {head}'


# --- the parts of an accepted 1:1 episode in words (W28 C13, cards g03-1 A, g03-3 C, g03-8 A, g03-9 A), read from its
# episodes.csv row (C7 `marks`, C8 `subregimes` and `residence_*`): the measured core (its first to last turning point),
# the sub-regimes between the changes that are no boundary, the amplitude changes and kicks of a near them, and the
# inferred residence around the core. Words only: nothing here feeds a status.

# a well of `geometry.Episode.wells` / `SubRegime.well` (coorbital_geometry's names) in words
WELL_WORDS = {'qs': 'quasi-satellite', 'tadpole_L4': 'tadpole L4', 'tadpole_L5': 'tadpole L5', 'horseshoe': 'horseshoe'}
# where an inferred residence stays, by the well of the sub-regime next to it ('' or 'compound': co-orbital)
RESIDENCE_WORDS = {
    'qs': 'in the quasi-satellite well',
    'tadpole_L4': 'in the L4 tadpole well',
    'tadpole_L5': 'in the L5 tadpole well',
    'horseshoe': 'on horseshoe orbits',
}
SUBREGIMES_SHOWN = 4  # the sentence names at most this many stretches: the first ones, '…' and the last with the count


def well_words(well: Any) -> str:
    """'quasi-satellite', 'tadpole L4', 'compound', ...; '' for an unnamed well ('-', '')."""
    name = _text(well).strip()
    if name in ('', '-', 'nan'):
        return ''
    return WELL_WORDS.get(name, name.replace('_', ' '))


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
    ('well_change', t, t), ('amplitude_change', t0, t1), ('kick', t, t) [yr]."""
    return _timed_items(row.get('marks'))


def episode_subregimes(row: Mapping[str, Any]) -> List[Tuple[str, float, float]]:
    """The sub-regimes of an accepted 1:1 run (episodes.csv `subregimes`, W28 C8): (well, t0, t1) [yr]."""
    return _timed_items(row.get('subregimes'))


def well_stretches(row: Mapping[str, Any]) -> List[Tuple[str, float, float]]:
    """The sub-regimes with neighbours of one well joined (an amplitude change inside a well is no change of well):
    (well, t0, t1) [yr]; the first starts and the last ends at the core's turning points (the column keeps one
    decimal), so a span reads as the episode's ('129 yr', not '128')."""
    out: List[Tuple[str, float, float]] = []
    for well, t0, t1 in episode_subregimes(row):
        if out and out[-1][0] == well:
            out[-1] = (well, out[-1][1], t1)
        else:
            out.append((well, t0, t1))
    c0 = row.get('core_t0_yr') if finite(row.get('core_t0_yr')) else row.get('t_start_yr')
    c1 = row.get('core_t1_yr') if finite(row.get('core_t1_yr')) else row.get('t_end_yr')
    if out and finite(c0) and finite(c1):
        out[0] = (out[0][0], float(c0), out[0][2])
        out[-1] = (out[-1][0], out[-1][1], float(c1))
    return out


def residence_sides(row: Mapping[str, Any]) -> Tuple[bool, bool]:
    """Whether the inferred residence (W28 C8) reaches before and after the measured core (its first and last turning
    points)."""
    r0, r1 = row.get('residence_t0_yr'), row.get('residence_t1_yr')
    c0 = row.get('core_t0_yr') if finite(row.get('core_t0_yr')) else row.get('t_start_yr')
    c1 = row.get('core_t1_yr') if finite(row.get('core_t1_yr')) else row.get('t_end_yr')
    if not all(finite(v) for v in (r0, r1, c0, c1)):
        return False, False
    return float(r0) < float(c0), float(r1) > float(c1)


def residence_clause(row: Mapping[str, Any]) -> str:
    """'stays in the quasi-satellite well from 601 yr to the end (inferred, not measured)' (2020 CB 1E-1, card g03-3 C):
    the inferred residence beyond the measured core, by side, in the well of the sub-regime next to it; a side bounded
    by the record's edge reads 'the start' / 'the end', one bounded by a circulation its time. '' without one."""
    before, after = residence_sides(row)
    if not (before or after):
        return ''
    stretches = well_stretches(row)
    first, last = (stretches[0][0], stretches[-1][0]) if stretches else ('', '')
    c0 = row.get('core_t0_yr') if finite(row.get('core_t0_yr')) else row.get('t_start_yr')
    c1 = row.get('core_t1_yr') if finite(row.get('core_t1_yr')) else row.get('t_end_yr')
    parts = []
    if before:
        edge = _text(row.get('residence_bounded_by_start')) == 'edge'
        parts.append(
            (RESIDENCE_WORDS.get(first, 'co-orbital'), f'from {"the start" if edge else years(row.get("residence_t0_yr"))} to {years(c0)}')
        )
    if after:
        edge = _text(row.get('residence_bounded_by_end')) == 'edge'
        parts.append(
            (RESIDENCE_WORDS.get(last, 'co-orbital'), f'from {years(c1)} to {"the end" if edge else years(row.get("residence_t1_yr"))}')
        )
    if len(parts) == 2 and parts[0][0] == parts[1][0]:
        text = f'{parts[0][0]} {parts[0][1]} and {parts[1][1]}'
    else:
        text = ' and '.join(f'{where} {when}' for where, when in parts)
    return f'stays {text} (inferred, not measured)'


def subregime_clause(row: Mapping[str, Any]) -> str:
    """'horseshoe 129–339 yr → tadpole L4 339–537 yr' (2016 JA 1E-1, card g03-2 A): the wells of an accepted 1:1 run in
    time order with their stretches, above SUBREGIMES_SHOWN the first ones, '…' and the last with the count; '' for a
    run of one well."""
    stretches = well_stretches(row)
    if len(stretches) < 2:
        return ''
    names = [f'{well_words(well) or "unnamed"} {interval(t0, t1)}' for well, t0, t1 in stretches]
    if len(names) > SUBREGIMES_SHOWN:
        names = names[: SUBREGIMES_SHOWN - 1] + [f'… {names[-1]} ({len(names)} sub-regimes)']
    return ' → '.join(names)


def amplitude_clause(row: Mapping[str, Any]) -> str:
    """'its amplitude changes at ~326–446 yr and ~708–809 yr (kicks of a at ~348 yr, ~783 yr)' (2021 XO3 1V-1, card
    g03-1 A): the transitional half-cycles of the amplitude changes and the kicks of a met there; 'kick of a at
    ~4.69 kyr' alone when the kicks meet a change of well (2024 TY11 1M-1 20 kyr). '' without either."""
    marks = episode_marks(row)
    changes = [interval(t0, t1) for name, t0, t1 in marks if name == 'amplitude_change']
    times = [f'~{years(t0)}' for name, t0, _ in marks if name == 'kick']
    kicks = f'{"kicks" if len(times) > 1 else "kick"} of a at {", ".join(times)}' if times else ''
    if changes:
        text = 'its amplitude changes at ' + ' and '.join(f'~{span}' for span in changes)
        return text + (f' ({kicks})' if kicks else '')
    return kicks


def coorbital_parts_clause(row: Optional[Mapping[str, Any]], residence: bool = True) -> str:
    """The sub-regimes, the amplitude changes and kicks and (`residence`) the inferred residence of one accepted 1:1
    run, '; '-joined; '' when it has none of them."""
    if not row:
        return ''
    parts = [subregime_clause(row), amplitude_clause(row), residence_clause(row) if residence else '']
    return '; '.join(part for part in parts if part)


def _episodes_clause(noun: str, eps: Episodes, settings, gaps: bool = True, measured: bool = False) -> str:
    """Where an episodic angle librates, without the final period: its one episode ('<noun> librates from … to … only
    (x% of the record), numbers; gaps'; `measured`: a 1:1 core with an inferred residence beyond it, '<noun> librates,
    measured, from … to … (x% of the record)', W28 C13) or its several ('… in N separate episodes covering …
    (longest …)')."""
    longest = eps.longest
    if longest is None:
        return f'{noun} librates in part of the record only'
    span = f'{years(longest.get("t_start_yr"))}–{years(longest.get("t_end_yr"))}'
    if len(eps.counted) == 1:
        start, end = years(longest.get('t_start_yr')), years(longest.get('t_end_yr'))
        text = (
            f'{noun} librates, measured, from {start} to {end} ' if measured else f'{noun} librates from {start} to {end} only '
        ) + f'({percent(eps.covered_fraction())} of the record), {libration_numbers(longest, centre=False)}'
        missed = coverage_text(longest, settings) if gaps else ''  # the slotted text: step 0 prints the gaps
        return text + (f'; {missed}' if missed else '')
    return (
        f'{noun} librates in {len(eps.counted)} separate episodes covering {percent(eps.covered_fraction())} of the record '
        f'(longest {span}: {libration_numbers(longest, centre=False)})'
    )


def _episodic_sentence(f: MMRFacts, gaps: bool = True) -> str:
    """The episodes of an episodic angle; of one counted 1:1 episode also its sub-regimes, amplitude changes and kicks
    and its inferred residence (`coorbital_parts_clause`, W28 C13: 2020 CB 1E-1 '…, measured, from 443 yr to 601 yr …;
    stays in the quasi-satellite well from 601 yr to the end (inferred, not measured)')."""
    v, eps = f.v, f.episodes
    single = eps.longest if len(eps.counted) == 1 else None
    parts = coorbital_parts_clause(single) if f.coorbital else ''
    text = _episodes_clause('σ', eps, f.s, gaps, measured=f.coorbital and single is not None and any(residence_sides(single)))
    text += f'; {parts}' if parts else ''
    if not eps.accepted:
        return text + '.'
    rejected = eps.rejected_text()
    text += f'; {rejected}' if rejected else ''
    if _forced_wave_word(f):
        text += f'; those loops are {forced_wave_source(f)}, so the status reads σ as {_tree_word(f)}'
    final = _text(v.final_regime)
    if final and final != str(v.angle_status) and finite(v.final_regime_start_yr):
        text += f'; from {years(v.final_regime_start_yr)} to the end {regime_phrase(final)}'
    return text + '.'


def forced_wave_source(f: MMRFacts) -> str:
    """'a forced wave from driver 3N-5+0+2' when the verdict names the circulating argument that drives a (A3a's role
    driver; card g04-5 A: 385266, never 'capture'), 'a forced wave of another argument' otherwise."""
    if str(f.v.cause_role) == 'driver' and f.v.owner:
        return f'a forced wave from driver {f.v.owner}'
    return 'a forced wave of another argument'


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
            ' through a change of its 1:1 regime, without a circulation between them' if f.coorbital else regime_change_clause(f)
        ),
    }.get(str(f.v.angle_subtype), '')
    regimes = _regimes_clause(f)
    if regimes:  # the regimes clause names the episodes and ends with 'the deciding one'
        centre = ',' if str(f.v.angle_subtype) == 'regime_change' and not f.coorbital else centre
        return f'σ librates {regimes}{centre} {libration_numbers(f.episodes.decisive or {})}.'
    parts = coorbital_parts_clause(f.episodes.decisive, residence=False) if f.coorbital else ''
    if parts and str(f.v.angle_subtype) == 'regime_change':
        # W28 C13 (card g03-1 A): one 1:1 episode whose changes are no boundary; a change of well is a change of regime,
        # an amplitude change inside one well is not (2021 XO3 1V-1: quasi-satellite throughout)
        several = len(well_stretches(f.episodes.decisive or {})) > 1
        centre = ' through a change of its 1:1 regime, without a circulation,' if several else ', without a circulation,'
    tail = f'; {parts}' if parts else ''
    return f'σ librates over the whole record{centre} {libration_numbers(f.episodes.decisive or {})}{tail}.'


def regime_change_clause(f: MMRFacts) -> str:
    """The subtype regime_change of an angle that is not 1:1 (and no union rule, `union_kind`) by what made it, ending
    with the comma before the deciding episode's numbers: several counted episodes without a circulation between them
    (W22) ' through 3 episodes without a circulation between them; the deciding one 2.21 kyr–7.17 kyr,'; one episode
    joined across a step or a stall of its centre (W28 C16, rule regime_cut_mark, episodes.csv `marks`) ' in one
    episode through a stall of its centre (8 kyr–14.6 kyr), without a circulation,'; else ', without a circulation,'."""
    eps, decisive = f.episodes, f.episodes.decisive or {}
    if len(eps.counted) > 1:
        span = f'{years(decisive.get("t_start_yr"))}–{years(decisive.get("t_end_yr"))}'
        return f' through {len(eps.counted)} episodes without a circulation between them; the deciding one {span},'
    marks = [
        f'{REGIME_MARK_WORDS[name]} ({_span(t0, t1)})' for name, t0, t1 in _timed_items(decisive.get('marks')) if name in REGIME_MARK_WORDS
    ]
    if marks:
        return f' in one episode through {" and ".join(marks)}, without a circulation,'
    return ', without a circulation,'


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
    ends = _end_clause(f)
    if net:
        passed = fmt(f.row.get('geo_centre_max_pass_rev'))
        return (
            f'σ circulates: its centre advances {net} revolutions net, in steps of at most {passed} revolution with small step-backs{ends}.'
        )
    return f'σ circulates: its centre passes {fmt(f.row.get("geo_centre_max_pass_rev"))} revolutions in one direction{ends}.'


def _end_clause(f: MMRFacts) -> str:
    """'; from 361 yr to the end σ_f slows to an arc (less than one revolution)': a circulation whose record ends in
    an arc or a stall (final_regime, W28 C14 rule end_arc_extend), '' otherwise."""
    final, start = _text(f.v.final_regime), f.v.final_regime_start_yr
    if final not in ('arc', 'stall') or not finite(start):
        return ''
    return f'; from {years(start)} to the end {regime_phrase(final)}'


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
    'insufficient_time': lambda f: _insufficient_sentence(f),
}


def _insufficient_sentence(f: MMRFacts) -> str:
    """What σ does in an insufficient_time record: half of one slow swing in the well (G5w, B6c), part of one slow
    swing (G5a), or too few turning points."""
    if slow_swing_decided(f):
        return (
            'σ rides on a slow centre that turns back twice: the record holds half of one slow swing of this resonance’s '
            'well, too short to show the libration whole.'
        )
    if str(f.v.angle_subtype) == 'arc' and _slow_arc(f) is not None:
        return (
            'σ rides on a slow centre that turns back once: the record holds part of one slow swing, too short to tell a '
            'drift from a slow libration.'
        )
    return f'σ has {fmt(f.row.get("geo_n_tp"))} turning points only: the record is too short to tell a libration.'


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
    """What σ does over the record (the author's wording: 'over the whole record' when it covers it); at a leaf that
    judged a circulation on one swing (G4c, G4h) the circulation it judged."""
    if f.rule in SINGLE_LEAVES and single_swing_clause(f):
        return _circulation_sentence(f)
    build = ANGLE_SENTENCES.get(str(f.v.angle_status))
    return build(f) if build else ''


# the leaves of the branch where the force law and the potential with a are both not measured (decide.unmeasured_branch)
BRANCH_LEAVES = ('5s', '5s*', '5o', '5x', '5n')


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
    return f'the episode holds one centre only (independent cycles {cycles}): nothing to compare, so nothing is measured, not contradicted'


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


def _branch_sentence(f: MMRFacts) -> str:
    """The status sentence of the branch's leaves 5s, 5s*, 5o, 5n (5s also before step 5: `strength_first`)."""
    if strength_first(f):
        head = f'{status_head(f.v.status)}: {branch_clause(f)}'
        return head + ': no libration of this angle’s own force, whatever the force law and the potential with a say.'
    text = f'{status_head(f.v.status)}: the force law and the potential with a are not measured; {branch_clause(f)}'
    if f.rule == '5s':
        return text + ': no libration of this angle’s own force.' + strength_mark_sentence(f)
    return text + '.'


def strength_marked(f: MMRFacts) -> list:
    """The accepted runs a 5s row marks forced (strength) (W28 B2, `ForcedWave.by_strength`)."""
    g = f.a.geometry
    return [wave for wave in g.forced_waves if wave.by_strength] if g is not None else []


def strength_mark_sentence(f: MMRFacts) -> str:
    """' Its 2 accepted episodes are drawn as forced (strength): a period this angle’s force cannot make; the driver is
    not identified.' (W28 B2, card g11-1 A: the name of the ground, not a driver found); '' without marks."""
    marked = strength_marked(f)
    if not marked:
        return ''
    n = len(marked)
    which = 'Its accepted episode is' if n == 1 else f'Its {n} accepted episodes are'
    return f' {which} drawn as {STRENGTH_MARK}: a period this angle’s force cannot make; the driver is not identified.'


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


def _dynamic_claims(f: MMRFacts) -> str:
    """What the force law and the potential said, as the reason for an own-dynamics verdict."""
    s, a = f.s, f.a
    l4, r0 = f.checks.get('a_response'), f.checks.get('potential')
    claims = []
    if l4 == 'pass':
        law = 'the 1:1 force law' if f.coorbital else 'ȧ = K sin(σ − φ0)'
        if taken_harmonic_law(a.response) is not None:
            law = 'ȧ = Σ_k [A_k sin kσ + B_k cos kσ] + C, the force of a well that is no sinusoid'
        if taken_direct_law(a.response) is not None:  # B8: the term's own averaged force, one scale s free
            law = 'ȧ = s × the averaged force of the term, at the libration scale'
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
            f'the force law is not measured ({_why_l4(f)}){decided}; a carries the swing '
            f'(B1 {_with_source(compare(b1, ">=", s.mmr_b1_ratio_min), f, "b1_source")}) and {rest}'
        )
    text = ' and '.join(claims)
    if l4 not in ('pass', 'fail'):
        text += f'; the force law is not measured: {_why_l4(f)}'
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
        not_run.append(COORBITAL_OWNER)
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
    elif str(v.reason) == 'separatrix_zone' and has_mark(v, 'near_separatrix_without_zone'):  # G3 without a measured zone
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
    fit = f'R² {compare(r2, ">=", f.s.mmr_owner_r2_min, digits=2)}'
    if not owner_confirmed(f):  # g10-8: never 'drives a' where its regimes do not cover σ's episode
        return f'Non-resonant for this angle: {f.v.owner} is a candidate driver of a ({fit}); this angle follows it through the identity.'
    return f'Non-resonant for this angle: {f.v.owner} drives a ({fit}); this angle only follows it through the identity.'


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
    if b1 is not None and b1.result is False:  # card g10-2: the pendulum argument in words
        return (
            'Non-resonant: a resonance reverses σ through a, so a must oscillate with σ; here a carries too little of the '
            f'wave this libration needs (B1 {compare(b1.value, ">=", b1.threshold)}): the swing sits in the perihelion term.'
        )
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
    if not finite(check) or (policy_minimal(f.v) and str(f.v.action) != 'extend'):  # P5: no next check the action lacks
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
    'modulated_circulation': lambda f: 'Non-resonant: σ stalls at the top of its potential in each revolution (pattern B) without a libration.',
    'drift_mechanism_unknown': lambda f: 'Uncertain: σ oscillates about a drifting centre and no tested mechanism explains the drift.',
    'baseline_needed': _baseline_sentence,
}


def mmr_status_sentence(f: MMRFacts) -> str:
    """Why this status: the rule that decided it, in plain words, with its numbers."""
    if f.rule in SINGLE_LEAVES and single_swing_clause(f):  # W28 C6: the leg or half-swing decided
        return _single_swing_sentence(f)
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
            # W28 E11 (g10-7): str_T_obs is T0 = P/F(A) inferred from the observed cycles, not an observed period
            source = _t_obs_source(_text(f.row.get('str_T_obs_source')))
            observed = f' against T0 {p.time("str_T_obs_yr")} from the observed cycles ({source})'
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
    parts = [f'x = Ψ/u, Ψ = {psi}' + (f', on {window}' if window else '')]
    if p.has('own_psi_turns'):
        parts.append(f'{p.num("own_psi_turns")} revolutions of Ψ')
    if p.has('own_psi_sweep_deg'):
        parts.append(f'x moved {p.cmp("own_psi_sweep_deg", ">=", s.mmr_owner_capture_sweep_deg, "°")}')
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
        parts.append('not measurable on the owner’s blocks over σ’s episode')
    text = '; '.join(parts) + f' → {CAPTURE_WORDS.get(verdict, verdict)}'
    columns = ('own_psi_sweep_deg', 'own_psi_energy_R2', 'own_psi_secular_R2', 'own_psi_self_R2')
    read = any(column in f.v_columns for column in columns)
    decided_icon = excluded or verdict == 'confirmed'
    rows = [p.take('capture', PHYSICS_ICONS[True] if decided_icon else PHYSICS_ICONS[None], CAPTURE_LABEL, text, read_by_verdict=read)]
    slow = slow_x_line(f, p)
    if slow:
        rows.append(p.take('slow_x', '', 'slow x', slow, grey=True))
    return rows


def _x_cycles_text(f: MMRFacts, p: Optional[_Printer] = None) -> str:
    """'x closes 18 cycles of its own well (period 1.2 kyr, off every secular period by 0.12)' (A2b, columns
    own_x_cycles, own_x_cycle_period_yr, own_x_secular_offset), '' without the count."""
    p = p if p is not None else _Printer(f.row)
    if not p.has('own_x_cycles'):
        return ''
    n = int(p.value('own_x_cycles'))
    p.num('own_x_cycles')
    head = f'x closes {_plural(n, "cycle")} of its own well'
    numbers = []
    if p.has('own_x_cycle_period_yr'):
        numbers.append(f'period {p.time("own_x_cycle_period_yr")}')
    if p.has('own_x_secular_offset'):
        numbers.append(f'off every secular period by {p.num("own_x_secular_offset")}')
    return head + (f' ({", ".join(numbers)})' if numbers else '')


def joint_line(f: MMRFacts) -> str:
    """Card g05-8 B (W28 A2b, origin joint): 'Both resonances at once: x = Ψ/u closes 18 cycles of its own well
    (period 1.2 kyr, off every secular period by 0.12) while 5J+6S-2+0+0-9 holds φ; σ librates in this resonance too.'
    One cycle (rule 2j): '… closes one cycle of its own well: not decided yet; a longer record decides.' ''
    otherwise."""
    verdict = _text(f.row.get('own_second_capture'))
    if str(f.v.libration_origin) != 'joint' or verdict not in ('confirmed', 'one_cycle'):
        return ''
    cycles = _x_cycles_text(f) or 'x closes cycles of its own well'
    owner = f' while {f.v.owner} holds φ' if f.v.owner else ''
    if verdict == 'one_cycle':
        return f'Both resonances at once? x = Ψ/u: {cycles}{owner}: not decided yet; a longer record decides.'
    return f'Both resonances at once: x = Ψ/u: {cycles}{owner}; σ librates in this resonance too.'


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
    if str(owner.check) not in ('pass', 'fail') and no_force_on_a(f):  # g04-7: no gap at a kinematic leaf
        text = f'n/a: {NO_FORCE_ON_A} (step 3), so no resonance drives it; the search itself: {why}'
        return PhysicsSection(title, 'owner search not needed', [PhysicsRow('owner', PHYSICS_NA, 'owner', text, columns=('own_flags',))])
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


def _owner_spans(f: MMRFacts) -> str:
    """The owner's measured regimes that meet σ's decisive episode: '630 yr–19 kyr (libration)' (LIST_MAX)."""
    owner, cover = f.a.owner, _decisive_cover(f)
    t0, t1 = (cover.t0_yr, cover.t1_yr) if cover is not None else (-math.inf, math.inf)
    spans = [(a, b, kind) for a, b, kind in owner.intervals if b >= t0 and a <= t1] or list(owner.intervals)
    items = [f'{years(a)}–{years(b)} ({str(kind).replace("_", " ")})' for a, b, kind in spans]
    hidden = len(items) - LIST_MAX
    return ', '.join(items[:LIST_MAX]) + (f' and {hidden} more' if hidden > 0 else '')


def neighbour_items(neighbours: str) -> List[Tuple[str, float, float, str]]:
    """(key, t0, t1, where) [yr] of the verdict's neighbours 'KEY t0–t1 before|during|after;…' (A3a); an item without
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


def _neighbour_spans(neighbours: str) -> Tuple[str, str]:
    """(key, '1.91–5.24 kyr before, 13.6–16.9 kyr after') from the verdict's neighbours 'KEY t0–t1 before;…' [yr]."""
    parsed = neighbour_items(neighbours)
    key = parsed[0][0] if parsed else ''
    items = [f'{years(t0)}–{years(t1)} {where}'.strip() for _, t0, t1, where in parsed if finite(t0)]
    hidden = len(items) - LIST_MAX
    return key, ', '.join(items[:LIST_MAX]) + (f' and {hidden} more' if hidden > 0 else '')


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


CANDIDATE_TAIL = 'not shown to make these cycles.'


def driver_line(f: MMRFacts) -> Tuple[str, bool]:
    """The cause line of an owner, driver or neighbour (card g10-8), and whether it is affirmative: 'Driven by X
    [owner: <its regimes on the episode>]' only when `owner_confirmed`; else 'Candidate driver: X (…): not shown to make
    these cycles' (`candidate_driver`). ('', False) when the verdict names none."""
    role, v = str(f.v.cause_role), f.v
    if role in ('owner', 'driver') and f.a.owner is not None and f.a.owner.accepted is not None and owner_confirmed(f):
        return f'Driven by {v.cause_term or v.owner} [{role}: {_owner_spans(f)}].', True
    key, detail = candidate_driver(f)
    if not key:
        return '', False
    return f'Candidate driver: {key} ({detail}): {CANDIDATE_TAIL}', False


def candidate_driver(f: MMRFacts) -> Tuple[str, str]:
    """(key, what it does where and why the decision did not take it) of a refused owner or driver ('…; it covers 0 <
    0.5 of σ’s decisive episode') or of a neighbour ('librates before the episode (…), outside it; …'); ('', '') for
    none or a confirmed owner."""
    role, v = str(f.v.cause_role), f.v
    if role in ('owner', 'driver') and f.a.owner is not None and f.a.owner.accepted is not None:
        if owner_confirmed(f):
            return '', ''
        refusal = _refusal(f)
        return v.cause_term or v.owner, f'{_owner_spans(f)}{"; " + refusal if refusal else ""}'
    if role == 'neighbour' and v.neighbours:
        key = v.cause_term or _neighbour_spans(v.neighbours)[0]
        where = neighbour_text(f)  # A3b: what it does where ('librates after the episode (…), outside it')
        where = where[len(key) :].strip() if where.startswith(key) else where
        refusal = _refusal(f) if f.a.owner is not None else ''
        return key, '; '.join(part for part in (where, refusal) if part)
    return '', ''


def candidate_driver_short(f: MMRFacts, steps: Sequence[Step]) -> str:
    """The compact form of the candidate-driver line (W28 FIX-plot: 2000 SF8 8M-9 and 1627 13E-33 overflowed): when
    the owner step prints its regimes and the refusal word for word, the line points there, 'Candidate driver: X (its
    regimes and why it was not taken: step 2): not shown to make these cycles.'; '' otherwise (the full line stays)."""
    key, detail = candidate_driver(f)
    owner = next((step for step in steps if step.key == 'owner'), None)
    if not (key and detail and owner is not None and detail in owner.outcome):
        return ''
    return f'Candidate driver: {key} (its regimes and why it was not taken: step 2): {CANDIDATE_TAIL}'


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
    return f'{years(t0)}–{years(t1)}'


def _neighbour_verb(f: MMRFacts, t0: float, t1: float) -> str:
    """What the neighbour does in its block, from its measured regimes there (`OwnerResult.intervals`): 'librates',
    'drives a forced wave', else 'acts'."""
    owner = f.a.owner
    kinds = {str(kind) for a, b, kind in (owner.intervals if owner is not None else ()) if b >= t0 and a <= t1}
    return {frozenset({'libration'}): 'librates', frozenset({'forced_wave'}): 'drives a forced wave'}.get(frozenset(kinds), 'acts')


def neighbour_text(f: MMRFacts) -> str:
    """The neighbour in words (card g04-6: the owner outside the episode 'librates later/earlier, outside it'; A3a):
    '6E-11+0+5 librates after the episode (13.7 kyr–17.9 kyr), outside it'; a block during the row's own episodes
    'librates during 3 kyr–5 kyr but does not make σ’s cycles'; '' without neighbours."""
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


def argument_offset(spec, member) -> str:
    """The secular term by which a member of the multiplet differs from the row's angle, member − row
    ('9M-16+2+5' − '9M-16+0+7' → '−2ϖ + 2ϖ_M'), '' when they differ by none or in their λ part."""
    if spec.planets != member.planets or spec.planet_coefficients != member.planet_coefficients:
        return ''
    terms = [(member.body_varpi_coefficient - spec.body_varpi_coefficient, 'varpi')]
    for name, mine, theirs in zip(spec.planets, spec.planet_varpi_terms, member.planet_varpi_terms):
        terms.append((theirs - mine, f'varpi_{planet_letter(name)}'))
    return term_text(''.join(f'{k:+d}{name}' for k, name in terms if k))


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
    owner and driver, the reason unchanged; g04-4 the neighbour; g04-7/g05-3 the kinematic term; g06-1/g06-2 the
    resonant argument):

      owner             'owner: 5E-16+0+11 (librates in the episode)'
      driver            'driver: 3N-5+0+2 (circulates, P 4.51 kyr): forced wave'
      candidate driver  an owner or driver the text cannot confirm on σ's episode (`owner_confirmed`)
      neighbour         '6E-11+0+5 librates after the episode (13.7 kyr–17.9 kyr), outside it'
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


def subtitle_role(f: MMRFacts) -> str:
    """The role in the subtitle: 'driver 3N-5+0+2', 'owner …', 'neighbour …', 'resonant argument …', 'kinematic −4ϖ'."""
    v, role = f.v, str(f.v.cause_role)
    if role == ROLE_KINEMATIC:
        return f'{ROLE_KINEMATIC} {term_text(v.cause_term)}'.strip()
    if role == ROLE_JOINT and (v.cause_term or v.owner):
        return f'also in {v.cause_term or v.owner}'
    parts = [f'{ROLE_ARGUMENT} {v.resonant_argument}' if v.resonant_argument else '']
    if role in (ROLE_OWNER, ROLE_DRIVER, ROLE_NEIGHBOUR) and (v.cause_term or v.owner):
        word = role if role == ROLE_NEIGHBOUR or owner_confirmed(f) else ROLE_CANDIDATE
        parts.insert(0, f'{word} {v.cause_term or v.owner}')
    return ' · '.join(part for part in parts if part)


# sub_choice (A6b, `SubresonanceResult.choice`) -> how the member was chosen, in words
CHOICE_WORDS = {
    'theory': 'chosen by the multiplet theory: its centre π − arg P + jβ is the flattest over the episode',
    'measured': 'the flattest measured centre: the multiplet theory does not apply here or does not follow',
}
COMPOSITE = 'composite'  # resonant_argument when no single member stands (A6b, origin_detail composite_multiplet)


def argument_line(f: MMRFacts) -> str:
    """Rule C in the cause slot (cards g06-1/g06-2 A: one resonance; the argument instead of an owner): 'Resonant
    argument: 9M-16+2+5 (chosen by the multiplet theory: …), one resonance 9M-16 with this angle; the two differ by
    −2ϖ + 2ϖ_M, so this angle’s centre turns with that term.' With 'composite' (W28 A6b): no single member, σ follows
    the moving centre of the composite term. Then the other episodes' own members (episodes.csv resonant_argument, A6b
    `episode_arguments`): '#1 76.2–99.8 kyr chose no member in its own window'. '' without a resonant argument."""
    v = f.v
    if not v.resonant_argument:
        return ''
    one = f', one resonance {v.commensurability} with this angle' if v.commensurability else ''
    if v.resonant_argument == COMPOSITE:
        text = (
            f'Resonant argument: composite{one}: no single member of the multiplet stands; σ follows the moving centre of '
            'the composite term P(β).'
        )
        return text + _episode_arguments_text(f)
    sub = f.a.subresonances
    member = sub.accepted.spec if sub is not None and sub.accepted is not None else None
    offset = argument_offset(f.a.resonance, member) if member is not None and member.key == v.resonant_argument else ''
    turn = f'; the two differ by {offset}, so this angle’s centre turns with that term' if offset else ''
    choice = CHOICE_WORDS.get(_text(f.row.get('sub_choice')), '')
    chosen = f' ({choice})' if choice else ''
    return f'Resonant argument: {v.resonant_argument}{chosen}{one}{turn}.' + _episode_arguments_text(f)


def _episode_arguments_text(f: MMRFacts) -> str:
    """' Other episodes: #1 76.2–99.8 kyr chose no member in its own window.' (A6b: each accepted episode is scanned in
    its own window); '' with one counted episode or rows without the column."""
    rows = sorted(f.episodes.counted, key=lambda row: _float(row.get('t_start_yr')))
    decisive = f.episodes.decisive or {}
    others = [row for row in rows if 'resonant_argument' in row and not _same_id(row.get('episode_id'), decisive.get('episode_id'))]
    if len(rows) < 2 or not others:
        return ''
    parts = []
    for row in others:
        span = f'#{fmt(row.get("episode_id"))} {interval(row.get("t_start_yr"), row.get("t_end_yr"))}'
        key = _text(row.get('resonant_argument'))
        parts.append(f'{span}: {key}' if key else f'{span} chose no member in its own window')
    return f' Other episodes: {"; ".join(parts)}.'


def gap_cause_line(f: MMRFacts) -> str:
    """Card g06-2 A (12218 4M-7, W28 A6a; columns sub_th_P_min_gap, sub_th_P_edges): 'The gap 48.7–76.2 kyr between
    the episodes is where the composite term cancels: |P|/max |P| falls to 8.8e-05 there (0.142 and 0.195 at its edges)
    as ϖ_M − ϖ passes its cancellation phase.' Said for a row with a resonant argument whose counted episodes leave a
    gap, and only when |P| in the gap falls below its value at both edges of the gap; '' otherwise."""
    p_min = f.row.get('sub_th_P_min_gap')
    if not f.v.resonant_argument or not finite(p_min) or f.a.resonance is None or len(f.a.resonance.planets) != 1:
        return ''
    rows = sorted(f.episodes.counted, key=lambda row: _float(row.get('t_start_yr')))
    gaps = [
        (_float(a.get('t_end_yr')), _float(b.get('t_start_yr')))
        for a, b in zip(rows, rows[1:])
        if _float(b.get('t_start_yr')) > _float(a.get('t_end_yr'))
    ]
    if not gaps:
        return ''
    edges = []
    for item in _text(f.row.get('sub_th_P_edges')).split(';'):
        t, _, value = item.partition(':')
        if finite(t) and finite(value):
            edges.append((float(t) * KYR, float(value)))
    beta = f'ϖ_{planet_letter(f.a.resonance.planets[0])} − ϖ'
    if len(gaps) > 1:
        return f'The gaps between the episodes fall where |P|/max |P| of the composite term drops to {sci(p_min)} as {beta} turns.'
    g0, g1 = gaps[0]
    near = [min(edges, key=lambda e: abs(e[0] - t))[1] for t in (g0, g1)] if edges else []
    if len(near) != 2 or not float(p_min) < min(near):
        return ''
    return (
        f'The gap {interval(g0, g1)} between the episodes is where the composite term cancels: |P|/max |P| falls to '
        f'{sci(p_min)} there ({fmt(near[0])} and {fmt(near[1])} at its edges) as {beta} passes its cancellation phase.'
    )


def diagnostic_scan_line(f: MMRFacts) -> str:
    """Trigger T8 alone (W28 A6a; columns sub_diag_*): 'A diagnostic subresonance scan (T8) ran: its flattest member is
    1M-3J-1-1+0+4, check not measured; without a measured force it changes nothing.' '' when T8 did not run it."""
    if 'T8' not in _text(f.row.get('sub_diag_triggers')).split(';'):
        return ''
    accept, best = _text(f.row.get('sub_diag_would_accept')), _text(f.row.get('sub_diag_best'))
    check = _text(f.row.get('sub_diag_would_check')).replace('_', ' ')
    found = f'it would accept {accept}' if accept else (f'its flattest member is {best}' if best else 'no member stands out')
    return f'A diagnostic subresonance scan (T8) ran: {found}{f", check {check}" if check else ""}; without a measured force it changes nothing.'


def frame_line(f: MMRFacts) -> str:
    """Card g06-4 (2003 UG22): a member of the multiplet that test E′ rejected as too weak (reason E_strength, its term
    |C_j|/max|C| = coefficient_rel) is a frame, not a force: a steadier centre of that member is the reference frame
    turning with the body's own e, i, ω, not a resonance. Names the weakest such member when the scan ran and accepted
    none; without a scan the cheap member list (A6a, sub_cheap_list 'key:W/W0:strength:reason') names the flattest
    such members whose centre is flatter than this angle's (W/W0 < 1): 'Members 10E-17+5+2 and 10E-17+6+1 give a
    flatter centre (W/W0 0.42, 0.45), but their terms are 7.6e-09 and 1.8e-10 of the strongest (test E′): a frame, not a
    force; …'. ''  otherwise."""
    sub = f.a.subresonances
    if sub is None:
        return _cheap_frame_line(f)
    if sub.accepted is not None:
        return ''
    weak = [c for c in sub.candidates if str(c.reason) == 'E_strength' and finite(c.coefficient_rel) and not c.nodal]
    if not weak:
        return ''
    member = min(weak, key=lambda c: c.coefficient_rel)
    return (
        f'A steadier centre of a member such as {member.spec.key} would be a frame, not a force: its term is '
        f'{sci(member.coefficient_rel)} of the strongest (test E′); the centre moves with the body’s own e, i, ω.'
    )


FRAME_MEMBERS_MAX = 2  # members the cheap frame line names (the flattest first)


def _cheap_frame_line(f: MMRFacts) -> str:
    """`frame_line` from the cheap member list (no scan ran), '' without a weak member flatter than this angle."""
    if f.v.resonant_argument:
        return ''
    weak = []
    for item in _text(f.row.get('sub_cheap_list')).split(';'):
        key, width, strength, reason = (item.split(':') + ['', '', '', ''])[:4]
        if reason == 'E_strength' and finite(width) and finite(strength) and float(width) < 1:
            weak.append((float(width), float(strength), key))
    if not weak:
        return ''
    chosen = sorted(weak)[:FRAME_MEMBERS_MAX]
    keys = ' and '.join(key for _, _, key in chosen)
    widths = ', '.join(fmt(width, 2) for width, _, _ in chosen)
    terms = ' and '.join(sci(strength) for _, strength, _ in chosen)
    many = len(chosen) > 1
    return (
        f'{"Members" if many else "Member"} {keys} {"give" if many else "gives"} a flatter centre (W/W0 {widths}), but '
        f'{"their terms are" if many else "its term is"} {terms} of the strongest (test E′): a frame, not a force; the '
        'centre moves with the body’s own e, i, ω.'
    )


def slow_x_line(f: MMRFacts, p: Optional[_Printer] = None) -> str:
    """The slow swing of σ's centre against the owner (A2 `SlowX`, columns own_x_*; card g05-1, the author's question
    about a ~10 kyr libration) at its level: identified 'σ’s centre swings with period 6.6 kyr (±11°): the 2ω cycle
    π/|ω̇| = 7.11 kyr'; possible '…: probably the 2ω cycle (…); a slow libration of another resonance is not excluded';
    no cause '…: cause not identified; a slow libration of another resonance is not excluded'. '' without a period.
    `p`: the printer of the physics row that prints it (its numbers are recorded there)."""
    p = p if p is not None else _Printer(f.row)
    if not p.has('own_x_period_yr'):
        return ''
    amp = f' (±{p.num("own_x_halfamp_deg")}°)' if p.has('own_x_halfamp_deg') else ''
    head = f'σ’s centre swings against the owner with period {p.time("own_x_period_yr")}{amp}'
    p.show('own_x_cause', 'own_x_cause_level', 'own_x_secular_term')
    level, term = _text(f.row.get('own_x_cause_level')), _text(f.row.get('own_x_secular_term'))
    name = SLOW_X_TERMS.get(term, f'the secular cycle {term}' if term else 'a secular cycle')
    reference = f' {p.time("own_x_secular_period_yr")}' if p.has('own_x_secular_period_yr') else ''
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
    test = f' ({CAPTURE_LABEL}: {CAPTURE_WORDS.get(capture, capture)})' if capture else ''
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
DRIVER_MEASURED, DRIVER_CANDIDATE = 'measured', 'candidate'


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


def _secular_period_words(f: MMRFacts) -> str:
    """'its period 10.5 kyr matches the 2ω cycle π/|ω̇| = 11.3 kyr' for origin_detail secular_cycle
    (`secular_reference`), '' otherwise."""
    period, ref, name, _ = secular_reference(f)
    return f'its period {years(period)} matches {name} = {years(ref)}' if name else ''


def kinematic_cause(f: MMRFacts) -> str:
    """The cause of a kinematic libration (card g10-2): the body's p·ϖ term and how much of σ's swing it spans on the
    decisive episode against the λ part (kin_swing_share, kin_lambda_swing_share, A5), the secular period it matches;
    'the swing is σ's pϖ term' only when that term spans more of the swing than the λ part (the text never asserts what
    its numbers refute); the modulating planet is not identified."""
    p = int(f.a.resonance.body_varpi_coefficient)
    if not p:
        return ''
    term = varpi_term(p, '*' if f.a.resonance.retrograde else '')
    if slow_kinematic(f.row, f.s.mmr_kin_period_tol):
        return slow_kinematic_cause(f, term)
    share, part = f.row.get('kin_swing_share'), f.row.get('kin_lambda_swing_share')
    period = _secular_period_words(f)
    swings = swing_words(f, term)
    tail = ''.join(f'; {part_}' for part_ in (period, swings) if part_)
    if not (finite(share) and finite(part)):
        reason = _text(f.row.get('kin_k1_episode_reason'))
        why = KIN_REASON_WORDS.get(reason, reason.replace('_', ' ')) if reason else 'not measured'
        return f'σ’s {term} term: its share of the swing is not measured ({why}){tail}.'
    shares = f'over the episode {term} spans {fmt(share, 2)} of σ’s swing, the λ part {fmt(part, 2)}'
    if float(share) <= float(part):
        return f'σ’s {term} term does not carry the swing: {shares}{tail}; what moves σ is not identified.'
    return f'The swing is σ’s {term} term: {shares}{tail}. Which planet modulates ϖ is not identified.'


def swing_words(f: MMRFacts, term: str) -> str:
    """The predicted and the measured half-swing of a kinematic libration (W28 B10b, card g08-8: 'predicted 105°,
    measured 114°'): kin_pred_swing_deg (|p| asin of the smaller over the larger of e_free, e_forced: the wobble of ϖ
    the two secular vectors make) against the slow swing (kin_meas_swing_deg) on a K3s row, else the decisive
    episode's median half-amplitude. '' without the prediction."""
    predicted = f.row.get('kin_pred_swing_deg')
    if not finite(predicted):
        return ''
    if slow_kinematic(f.row, f.s.mmr_kin_period_tol):
        measured, what = f.row.get('kin_meas_swing_deg'), 'the slow swing'
    else:
        measured, what = (f.episodes.decisive or {}).get('half_amp_med_deg'), 'the episode’s half-amplitude'
    head = f'the free and forced e predict ±{fmt(predicted)}° for {term}'
    return f'{head}, {what} is ±{fmt(measured)}°' if finite(measured) else head


def slow_kinematic_cause(f: MMRFacts, term: str) -> str:
    """The cause of a kinematic libration on the slow scale (rule K3s, card g08-8; 37087 3J+8S-3): 'The slow swing is
    σ’s −8ϖ term: on the slow scale (17.3 kyr–96.6 kyr, cutoff 506 yr) its period 5.69 kyr matches the e cycle
    2π/|g − g5| = 5.70 kyr; the free and forced e predict ±96.7° for −8ϖ, the slow swing is ±124°; σ̇ = c(a − a*)
    fails there (slope ratio 2.78, corr 0.51).' From the slow_* columns of B10a."""
    row = f.row
    period, ref, name, _ = secular_reference(f)
    span = f'{years(row.get("slow_t0_yr"))}–{years(row.get("slow_t1_yr"))}'
    head = f'The slow swing is σ’s {term} term: on the slow scale ({span}, cutoff {years(row.get("slow_cutoff_yr"))})'
    parts = [f'{head} its period {years(period)} matches {name} = {years(ref)}' if name else head]
    swings = swing_words(f, term)
    if swings:
        parts.append(swings)
    if _text(row.get('slow_identity')) == 'fail':
        parts.append(
            f'σ̇ = c(a − a*) fails there (slope ratio {fmt(row.get("slow_identity_slope_ratio"))}, '
            f'corr {fmt(row.get("slow_identity_corr"), 2)})'
        )
    return '; '.join(parts) + '.'


def slow_scale_line(f: MMRFacts) -> str:
    """The slow scale of σ's centre at its cause level (W28 B10b, card g05-2: 'compute such a libration if it is there,
    so that it shows in the right column; name the potential cause with its confidence'; columns slow_* of B10a):
    'Slow scale: σ’s centre swings by ±124° with period 5.69 kyr (13 slow cycles, 17.3 kyr–96.6 kyr): the e cycle
    2π/|g − g5| = 5.70 kyr (identified).'; possible '…: probably …; a slow libration of another resonance is not
    excluded.'; unknown '…: cause not identified; …'. A slow scale whose identity holds says so (a dynamic candidate;
    no status follows). '' without a slow swing, on a K3s row (`slow_kinematic_cause` says it) and where the owner's
    slow x already does (`slow_x_line`)."""
    row = f.row
    if not _text(row.get('slow_source')) or not finite(row.get('slow_halfamp_deg')):
        return ''
    if str(f.v.reason) == 'kinematic_libration' and slow_kinematic(row, f.s.mmr_kin_period_tol):
        return ''
    if _text(row.get('slow_cause_source')) == 'owner_x' and str(f.v.reason) == 'induced_libration':
        if _text(row.get('own_x_cause_level')) in ('identified', 'possible'):
            return ''
    span = f'{years(row.get("slow_t0_yr"))}–{years(row.get("slow_t1_yr"))}'
    swing = f'±{fmt(row.get("slow_halfamp_deg"))}°'
    n = int(_float(row.get('slow_n_cycles'))) if finite(row.get('slow_n_cycles')) else 0
    if finite(row.get('slow_period_yr')):
        head = (
            f'Slow scale: σ’s centre swings by {swing} with period {years(row.get("slow_period_yr"))} ({_plural(n, "slow cycle")}, {span})'
        )
    else:
        head = f'Slow scale: σ’s centre swings by {swing} over {span} ({_plural(n, "slow cycle")}: no period)'
    not_excluded = 'a slow libration of another resonance is not excluded'
    holds = _text(row.get('slow_origin')) == 'dynamic_candidate'
    if holds:
        not_excluded = 'σ̇ = c(a − a*) holds on that scale, so a slow libration is not excluded'
    cause, level = _text(row.get('slow_cause')), _text(row.get('slow_cause_level'))
    if cause in SECULAR_CAUSES:
        term, ref, _ = slow_secular_cause(row)
        name = SECULAR_TERM_NAMES.get(term, (f'the secular cycle {term}',))[0]
        reference = f' = {years(ref)}' if finite(ref) else ''
        if level == 'identified':
            return f'{head}: {name}{reference} (identified){"; " + not_excluded if holds else ""}.'
        return f'{head}: probably {name}{reference}; {not_excluded}.'
    if cause == 'pvarpi_term':
        term = term_text(_signed(row.get('slow_cause_term')))
        return f'{head}: probably σ’s {term} term (the centre follows it at no secular period); {not_excluded}.'
    return f'{head}: cause not identified; {not_excluded}.'


def _signed(term: Any) -> str:
    """'-8varpi' as `term_text` reads it ('+8varpi' for a positive coefficient without a sign)."""
    term = _text(term)
    return term if term[:1] in '+-' else f'+{term}'


def secular_driver_line(f: MMRFacts) -> Tuple[str, Tuple[str, ...]]:
    """The Driver line of a kinematic libration at a secular period (W28 B10b; card g05-3, the author: show the term
    where an owner would be): 'Driven by: secular cycle −4ϖ (the 2ω cycle π/|ω̇|, P 18.0 kyr) [measured]', first
    under the plaque. measured: the period matches within mmr_kin_period_tol and the swing is shown to be that term's
    (on σ's episode the margin K1e passes; on the slow scale, K3s, the cause level is identified); candidate
    otherwise. ('', ()) for any other row."""
    v = f.v
    if str(v.cause_role) != ROLE_KINEMATIC or str(v.origin_detail) != 'secular_cycle':
        return '', ()
    period, ref, name, _ = secular_reference(f)
    if not name:
        return '', ()
    p = int(f.a.resonance.body_varpi_coefficient)
    term = term_text(v.cause_term) if v.cause_term else varpi_term(p)
    tol = f.s.mmr_kin_period_tol
    matches = finite(period) and finite(ref) and ref > 0 and abs(period / ref - 1) <= tol
    if slow_kinematic(f.row, tol):
        ids, shown = ('K3s',), _text(f.row.get('slow_cause_level')) == 'identified'
    else:
        k1e = f.margins.get('K1e')
        ids, shown = ('K1e',), k1e is not None and k1e.margin in (Margin.FULL, Margin.THIN)
    word = DRIVER_MEASURED if matches and shown else DRIVER_CANDIDATE
    return f'Driven by: secular cycle {term} ({name}, P {years(ref)}) [{word}].', ids


def other_resonances_line(f: MMRFacts, decisive: str) -> str:
    """The third kind of cause line (card g10-8): 'Other resonances: not tested (<reason>)', with '; not needed: a is
    not driven' at a kinematic leaf (`no_force_on_a`, g04-7); 'tested, none drives a (best …)' at a kinematic leaf whose
    owner search answered. '' where the verdict names a driver, for a 1:1 or a derived row, an angle that does not
    librate, or when the decisive sentence already says it."""
    owner = f.a.owner
    if str(f.v.cause_role) in ('owner', 'driver', 'neighbour') or f.derived or f.coorbital:
        return ''
    if owner is not None and 'owner_coorbital' in owner.flags:
        return ''
    if str(f.v.angle_status) not in ('libration', 'episodic') or 'no other resonance drives a' in decisive:
        return ''
    needed = '; not needed: a is not driven' if no_force_on_a(f) else ''
    if owner is None or str(owner.check) not in ('pass', 'fail'):
        return f'Other resonances: not tested ({_skip_owner(f)[1]}){needed}.'
    if needed and owner.accepted is None:
        best = _best_candidate(owner.candidates)
        detail = f' (best {best.spec.key}: {_owner_rejection(f, best)})' if best is not None else ''
        return f'Other resonances: tested, none drives a{detail}{needed}.'
    return ''


def info_items(f: MMRFacts, physics: Sequence[PhysicsSection]) -> List[Tuple[str, str]]:
    """(check value, words) of `mmr_info`; a 'fail' among them is an objection that did not decide (card g10-8)."""
    items = []
    for text in mmr_info(f, physics):
        name = next((key for key, value in f.checks.items() if value == 'fail' and text == f'{CHECK_WORDS.get(key, key)} fail'), '')
        kinematic = text.startswith('ϖ-kinematic test') and f.checks.get('kinematic') == 'fail'
        items.append(('fail' if name or kinematic else '', text))
    return items


def objection_line(items: Sequence[Tuple[str, str]]) -> str:
    """'Objection, not read by this verdict: identity fail; …' from the failed checks of the grey line ('' for none)."""
    failed = [text for value, text in items if value == 'fail']
    return f'Objection, not read by this verdict: {"; ".join(failed)}.' if failed else ''


ORBIT_SOURCE_WORDS = {'saved': '', 'J2000': 'planet bands at J2000'}
PLANET_ORDER = ('Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune')


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
        words = f'bands of {", ".join(name for name in PLANET_ORDER if name not in saved)} at J2000'
    note = f' ({words})' if words else ''
    if overlap == 'none':
        return f'Radial overlap: none{note}.'
    return f'Radial overlap with {", ".join(overlap.split(";"))}{note}.'


def _plain(sentence: str, status: Any) -> str:
    """A status sentence without its head 'Non-resonant: ' (the verdict line says the status)."""
    head = f'{status_head(status)}: '
    return sentence[len(head) :] if sentence.startswith(head) else sentence


def _comm(f: MMRFacts) -> str:
    return f.v.commensurability or f.a.resonance.key


VERDICT_LINES = {
    'kinematic_libration': lambda f: f'NOT THIS RESONANCE: σ librates, but {_comm(f)} does not make the swing.',
    'kinematic_drift': lambda f: f'NOT THIS RESONANCE: σ’s centre follows its perihelion term, not {_comm(f)}.',
    'induced_libration': lambda f: (
        f'NOT THIS RESONANCE: σ follows {f.v.owner} through the identity.'
        if owner_confirmed(f) or not f.v.owner
        else f'NOT THIS RESONANCE: σ follows a candidate driver, {f.v.owner}, through the identity.'
    ),
    'subresonance_librates': lambda f: f'{status_head(f.v.status).upper()}: a member of the multiplet {_comm(f)} holds the centre.',
    'dynamic_libration': lambda f: f'RESONANT: σ is the own pendulum of {_comm(f)} over the whole record.',
    'dynamic_episode': lambda f: f'TRANSIENT: σ is the own pendulum of {_comm(f)} in part of the record only.',
    'geometric_libration': lambda f: f'RESONANT by the shape alone: σ librates over the whole record; its own force is not measured.',
    'geometric_episode': lambda f: f'TRANSIENT by the shape alone: σ librates in part of the record; its own force is not measured.',
    'separatrix_zone': lambda f: f'{status_head(f.v.status).upper()}: σ moves near the separatrix of {_comm(f)}.',
    'origin_unknown': lambda f: f'UNCERTAIN: σ librates, but nothing measured shows that {_comm(f)} makes it.',
    'implausible_strength': lambda f: f'{status_head(f.v.status).upper()}: {_comm(f)} looks too weak for the period of these cycles.',
    'circulation': lambda f: 'NOT IN THIS RESONANCE: σ circulates.',
    'alias': lambda f: f'{status_head(f.v.status).upper()}: the output step folds σ.',
    'no_resonant_potential': lambda f: 'NOT THIS RESONANCE: σ reverses at times, but no potential from a holds it.',
    'modulated_circulation': lambda f: 'NOT IN THIS RESONANCE: σ stalls at the top of its potential without a libration.',
    'drift_mechanism_unknown': lambda f: 'UNCERTAIN: σ’s centre drifts and no tested mechanism explains it.',
    'baseline_needed': lambda f: 'UNCERTAIN: the record does not decide.',
}


def verdict_line(f: MMRFacts) -> str:
    """The verdict in plain words (the slot 'verdict'), by the verdict's reason; '' for a reason without one."""
    single = SINGLE_SWING_LINES.get((f.rule, str(f.v.reason)))  # W28 C6: a leaf that decided on one swing
    if single:
        return single.format(comm=_comm(f))
    build = VERDICT_LINES.get(str(f.v.reason))
    return build(f) if build else ''


def cell_words(verdict) -> str:
    """The calibration cell in a few words: 'cell: 35 author labels, 0 wrong (accuracy 90%–100%)' ('' without one)."""
    n, errors = getattr(verdict, 'cell_n_examples', None), getattr(verdict, 'cell_n_errors', None)
    if not finite(n):
        return ''
    if not n:
        return 'Cell: no author label yet.'
    low, high = getattr(verdict, 'cell_accuracy_low_95', None), getattr(verdict, 'cell_accuracy_high_95', None)
    accuracy = f' (accuracy {percent(low)}–{percent(high)})' if finite(low) and finite(high) else ''
    return f'Cell: {_plural(int(n), "author label")}, {fmt(errors)} wrong{accuracy}.'


def change_line(verdict, checks: Sequence[CheckRow]) -> str:
    """What would change the verdict (the slot 'change'): the level and what limits it (`confidence_parts`; 'Without
    the cap it would be …' is the arithmetic 'Support: …' below, the other reading of the cap is its LIMIT row), the
    calibration cell in a few words and the action."""
    head = '' if policy_minimal(verdict) else confidence_parts(verdict, checks)[0]  # W29: the basis line says it
    return ' '.join(part for part in (head, cell_words(verdict), action_sentence(verdict)) if part)


def mmr_bullets(f: MMRFacts, checks: Sequence[CheckRow], items: Sequence[Tuple[str, str]], steps: Sequence[Step] = ()) -> List[Bullet]:
    """The slotted explanation of an MMR verdict (card g10-2 style (i), g10-8): a confirmed driver first, directly
    under the plaque ('Driven by 11J-4+0-7 [driver: …]'); then the verdict in plain words, what σ does, the decisive
    argument (the status sentence without its head), an objection that did not decide, the cause (the kinematic term,
    the identity σ = φ/8 + Ψ/8 of an induced row, a candidate driver, other resonances not tested), what would change
    the verdict, and the radial overlap. Every sentence is built from the same facts as the status sentences. `steps`
    (`mmr_steps`): a candidate driver whose regimes and refusal the owner step prints gets a short form for the
    compact explanation (`candidate_driver_short`)."""
    v = f.v
    f.brief = True  # the decisive sentence points to step 5 for the force windows
    decisive = mmr_status_sentence(f)
    f.brief = False
    driver, confirmed = driver_line(f)
    out = [Bullet(SLOT_CAUSE, driver, ('W',))] if confirmed else []
    secular, secular_ids = secular_driver_line(f)  # B10b: a kinematic libration's secular cycle where an owner would be
    if secular:
        out.append(Bullet(SLOT_CAUSE, secular, secular_ids))
    out.append(Bullet(SLOT_VERDICT, verdict_line(f)))
    episodic = str(v.angle_status) == 'episodic' and not (f.rule in SINGLE_LEAVES and single_swing_clause(f))
    angle = _episodic_sentence(f, gaps=False) if episodic else mmr_angle_sentence(f)
    sigma = (angle, action_step_sentence(f), not_counted_clause(f))  # W28 C11: accepted runs the status does not count
    out.append(Bullet(SLOT_SIGMA, ' '.join(part for part in sigma if part)))
    out.append(Bullet(SLOT_DECISIVE, _plain(decisive, v.status), (f.rule,) if f.rule else ()))
    out.append(Bullet(SLOT_OBJECTION, objection_line(items)))
    if str(v.reason) == 'kinematic_libration':
        out.append(Bullet(SLOT_CAUSE, kinematic_cause(f), ('K1e',)))
    if not confirmed:
        out.append(Bullet(SLOT_CAUSE, driver, ('W',), short=candidate_driver_short(f, steps) if driver else ''))
    if str(v.reason) == 'induced_libration':
        out.append(Bullet(SLOT_CAUSE, identity_line(f)))
        if _text(f.row.get('own_x_cause_level')) in ('identified', 'possible'):  # g05-1: the slow swing has a cause
            out.append(Bullet(SLOT_CAUSE, slow_x_line(f)))
    out.append(Bullet(SLOT_CAUSE, slow_scale_line(f)))  # B10b, g05-2: the slow scale at its cause level
    out.append(Bullet(SLOT_CAUSE, joint_line(f), ('W',)))  # W28 A2b
    out.append(Bullet(SLOT_CAUSE, argument_line(f), ('S',)))
    out.append(Bullet(SLOT_CAUSE, gap_cause_line(f), ('S',)))  # W28 A6a: the 12218 gap
    out.append(Bullet(SLOT_CAUSE, frame_line(f), ('S',)))
    out.append(Bullet(SLOT_CAUSE, diagnostic_scan_line(f), ('S',)))  # W28 A6a: T8
    out.append(Bullet(SLOT_CAUSE, line_share_line(f), ('L4',)))
    out.append(Bullet(SLOT_CAUSE, kinematic_suspicion_line(f), ('K1',)))
    out.append(Bullet(SLOT_CAUSE, other_resonances_line(f, decisive), ('W',)))
    change = change_line(v, checks)
    extension = ' '.join(part for part in (extension_line(f), extend_flag_sentence(f)) if part)  # W28 C14: extend_time
    out.append(Bullet(SLOT_CHANGE, f'{change} {extension}' if change and extension else change or extension))
    out.append(Bullet(SLOT_ORBIT, orbit_line(f.row)))
    return [bullet for bullet in out if bullet.text]


def line_share_line(f: MMRFacts) -> str:
    """Which forced lines carry ȧ when the angle's own force law does not (W28 B8b, cards g07-3 A, g07-2 A; columns
    rsp_line_ids, rsp_line_period_yr, rsp_line_share, rsp_line_share_all): 'ȧ’s oscillation is mostly the 107-yr line
    57 P_M = P_conj (7M-10): 93% of its variance'. Only where the harmonic L4 fails or is thin (the direct law may
    have lifted it), not at a kinematic leaf (a carries no swing there); never the libration's own period (P_lib) and
    never the planets' radial overlap (E6, the orbit line)."""
    ids, share = _text(f.row.get('rsp_line_ids')), f.row.get('rsp_line_share')
    response, l4 = f.a.response, f.margins.get('L4')
    weak = f.checks.get('a_response') == 'fail' or (l4 is not None and l4.margin in (Margin.THIN, Margin.FAIL))
    if response is not None and taken_direct_law(response) is not None:
        weak = True  # the harmonic law failed; the term's averaged force decided
    names = [name.strip() for name in ids.split(';') if name.strip()]
    if not (weak and names and finite(share)) or names[0] == 'P_lib' or f.rule in KINEMATIC_LEAVES:
        return ''  # at a kinematic leaf a does not carry the swing: its lines say nothing about this angle
    period = f.row.get('rsp_line_period_yr')
    if names[0] == 'unidentified':
        return f'ȧ’s oscillation is mostly an unidentified {years(period)} line: {percent(share)} of its variance.'
    head, rest = names[0], [name for name in names[1:] if name not in ('P_lib',)]
    also = f' ({"; ".join(rest)})' if rest else ''
    text = f'ȧ’s oscillation is mostly the {years(period)} line {head}{also}: {percent(share)} of its variance'
    together = f.row.get('rsp_line_share_all')
    if finite(together) and float(together) > float(share) + 0.005:
        text += f', the lattice lines together {percent(together)}'
    return text + '.'


def kinematic_suspicion_line(f: MMRFacts) -> str:
    """An uncertain row whose swing the body's p·ϖ term spans more than the λ part does (kin_swing_share >
    kin_lambda_swing_share, the predicate `kinematic_cause` asserts with), while the force law that would tell a
    resonance from a kinematic swing is not measured (W28 B9, card g06-6 A: 2003 EH16): 'Suspected kinematic: over the
    episode 3ϖ spans 0.77 of σ’s swing, the λ part 0.46; the force law is not measured (half-amplitude 28.6° < 45°)'.
    A suspicion, not a status: '' for any other row."""
    if str(f.v.status) != 'uncertain' or f.checks.get('a_response') in ('pass', 'fail') or f.a.resonance is None:
        return ''
    p = int(f.a.resonance.body_varpi_coefficient)
    share, part = f.row.get('kin_swing_share'), f.row.get('kin_lambda_swing_share')
    if not (p and finite(share) and finite(part) and float(share) > float(part)):
        return ''
    term = varpi_term(p, '*' if f.a.resonance.retrograde else '')
    why = short_reason(_why_l4(f))
    return (
        f'Suspected kinematic: over the episode {term} spans {fmt(share, 2)} of σ’s swing, the λ part {fmt(part, 2)}; '
        f'the force law that would tell it from a resonance is not measured ({why}).'
    )


def extension_line(f: MMRFacts) -> str:
    """What a length-limited extension serves (W28 B9, cards g08-5 B, g08-6 A; rule extend_length_limited, A-FW of
    `decide.force_window_extension`): 'The extension serves the late regime: 2 force windows of 3.5 P for #0 (from 2.21
    kyr, P 3.14 kyr); the failed potential stays measured, a longer record does not remove it.' Only when that rule fired on a force-window
    leaf (`EXTEND_LEAVES`) and L4 is short of windows (`force_windows_short`)."""
    s, v, g = f.s, f.v, f.a.geometry
    if str(Rule.EXTEND_LENGTH_LIMITED) not in v.rules_fired or EXTEND_LEAVES.get(f.rule) != LENGTH_FORCE_WINDOWS or g is None:
        return ''
    if not force_windows_short(f.a.response, s):
        return ''
    late = last_episode_at_end(g, s)
    if late is None or not finite(late.period_med_yr):
        return ''
    windows = s.mmr_extend_force_windows
    text = (
        f'The extension serves the late regime: {fmt(windows)} force windows of {fmt(s.mmr_force_window_cycles)} P for '
        f'#{late.episode_id} (from {years(late.t_start_yr)}, P {years(late.period_med_yr)})'
    )
    failed = [CHECK_WORDS.get(name, name) for name, value in f.checks.items() if value == 'fail' and name in ('potential', 'identity')]
    if failed:
        verb, pronoun = ('stays', 'it') if len(failed) == 1 else ('stay', 'them')
        text += f'; the failed {" and ".join(failed)} {verb} measured, a longer record does not remove {pronoun}'
    return text + '.'


# --------------------------------------------------------------------------------------------- numbers behind the verdict

# the second word of a decision-path entry that names its check ('3 B1 a carries the amplitude', '5 L4 force law R2')
ENTRY_IDS = ('B1', 'B5', 'K2', 'K2b', 'L3', 'L4', 'R0', 'O1', 'F')
# an entry of a step that names no check: the step's own check ('1 subresonance …' -> S, '2 owner R2' -> W)
STEP_ENTRY_IDS = {'1': 'S', '2': 'W', '3': 'K1'}
# entries of a step that belong to another question than the step's check (`CAPTURE_LABEL`: both resonances at once?)
ENTRY_PREFIX_IDS = {'2 second capture': 'capture'}
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
    centre pass' -> G4, '5n independent cycles' -> 5n, 'C5 alias: …' -> C5); a step's entry by the check it names ('3 B1
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
        name = rule_name(rule_id) or rule_id
        if measured:
            parts = tuple(
                EmphasisPart(column, _column_comparison([entry for entry in entries if entry.value is not None]))
                for column, entries in measured.items()
            )
            out.append(Emphasis(EMPHASIS_DECISIVE, rule_id, name, parts))
        else:
            why = short_reason(why_not_measured(f, rule_id)[1]) if rule_id in f.margins else ''
            note = f'not measured ({why})' if why and why != 'not measured' else 'not measured'
            out.append(Emphasis(EMPHASIS_UNMEASURED, rule_id, name, (), note))
    return out


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


def _outside_physics(line: Emphasis, physics: Sequence[PhysicsSection]) -> Emphasis:
    """`line` without the numbers the physics block prints (W20 CS11: no number twice): they are named by the block's
    title instead ('strength plausible [F]: numbers in “Is σ the own pendulum of …?” above')."""
    titles = [
        section.title
        for section in physics
        if any(part.column in row.columns for part in line.parts for row in section.rows if part.column)
    ]
    if not titles:
        return line
    kept = tuple(part for part in line.parts if not any(part.column in row.columns for section in physics for row in section.rows))
    where = ' and '.join(f'“{title}”' for title in dict.fromkeys(titles))
    return replace(line, parts=kept, note=' '.join(part for part in (line.note, f'(numbers in {where} above)') if part))


def number_emphasis(
    f: MMRFacts,
    steps: Sequence[Step],
    checks: Sequence[CheckRow],
    items: Sequence[Tuple[str, str]],
    physics: Sequence[PhysicsSection] = (),
) -> List[Emphasis]:
    """The lines of 'Numbers behind this verdict' (card g10-5), read from the ledger and the confidence rows the figure
    already prints: decisive comparisons, the binding caps, the objections, the decisive tests without a value; in
    that order (`EMPHASES`). A number the physics block prints stays there (`_outside_physics`)."""
    lines = _decisive_lines(f, steps) + _limit_lines(f, checks) + _conflict_lines(f, items)
    lines = [_outside_physics(line, physics) for line in lines]
    return sorted(lines, key=lambda line: EMPHASES.index(line.kind))


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
            subtitle_role(f),  # A3b: the role words (driver, owner, neighbour, resonant argument, kinematic term)
            decided(shown_rule(f)),
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
    steps = mmr_steps(f)
    items = info_items(f, physics)
    # the figure draws `bullets`; a failed check of the grey line is an objection there and leaves the grey line
    info = [text for value, text in items if value != 'fail']
    text = _verdict_text(v, subtitle, head, steps, checks, info, grey, physics, mmr_angle_badge(f))
    bullets = mmr_bullets(f, checks, items, steps)
    for row in checks:  # the drawn text has no confidence sentence: each LIMIT row names its other reading (CS11)
        row.said = False
    emphasis = number_emphasis(f, steps, checks, items, physics)
    bullets = policy_block(v) + bullets  # W29 §7: the minimal policy's four lines first
    return replace(text, decided_at=decided_at(shown_rule(f), v.status, steps), docs=DOCS_MMR, bullets=bullets, emphasis=emphasis)


def shown_rule(f: MMRFacts) -> str:
    """The rule the subtitle and the steps' header name: rec_rule, except that an insufficient_time leaf (rec_rule G6
    for every subtype, `decide.decide_status`) names the angle-shape rule that set the word (G5a one slow arc, G5w a
    slow swing in the well: W28 B6c), so 'decided: slow swing in the well (G5w)' agrees with step 0."""
    if f.rule == 'G6' and str(f.v.angle_status) == 'insufficient_time':
        shape = [rule_id for rule_id in _base_shape_ids(f, 'insufficient_time') if rule_id.startswith('G')]
        return shape[-1] if shape else f.rule
    return f.rule


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
        action=action_tag(verdict),
    )


# --------------------------------------------------------------------------------------------- secular and ZLK


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
    name = LONG_PERIOD_MARGIN_NAMES.get(key, key)
    return CheckRow(key, name, state, detail, need, counts, note, kind=margin_kind(margin), rule_id=long_period_id(key))


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
    noun = _long_noun(analysis)
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


def _mode_longitude(mode: str) -> str:
    """'g5' -> 'ϖ5': the longitude of a forced eccentricity mode."""
    return f'ϖ{mode[1:]}' if mode.startswith('g') and len(mode) > 1 else f'ϖ({mode})'


CENTRE_CAUSE_ORIGINS = ('forced_beat:', 'forced_direction', 'node')


def _centre_cause(line) -> str:
    """The centre's slow line as a cause: what its origin (`slow.zlk_centre_line`) names; '' for unexplained or none."""
    if line is None or not finite(line.period_yr) or not str(line.origin).startswith(CENTRE_CAUSE_ORIGINS):
        return ''
    wave = f'The centre\'s {years(line.period_yr)} wave (±{math.degrees(line.half_amp_rad):.3g}°)'
    if line.origin.startswith('forced_beat:'):
        small, big = line.origin.split(':', 1)[1].split('-')
        return f'{wave} is the {small}−{big} beat of the forced vector (r {line.model_r:.2f}).'
    if line.origin == 'forced_direction':
        return f'{wave} follows the forced direction against the node, arg z_forced − Ω (r {line.model_r:.2f}).'
    return f'{wave} is the node\'s own (r {line.model_r:.2f}).'


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


def zlk_cause(analysis) -> Tuple[str, Tuple[str, ...]]:
    """(the sentence naming what induces a kinematic ZLK libration, the facts it used: 'attribution', 'centre',
    'node_lock'). Only for a row whose status took the kinematic outcome (origin kinematic); from the measured columns
    only: the attribution (`diagnostics.attribute_forcing`), the centre's slow line and the node lock. Unattributed: the
    reason, briefly. ('', ()) otherwise."""
    zlk, gate, verdict = getattr(analysis, 'zlk', None), getattr(analysis, 'gate', None), analysis.verdict
    if zlk is None or gate is None or gate.outcome != 'kinematic' or str(verdict.libration_origin) != 'kinematic':
        return '', ()
    attribution = zlk.attribution
    if not attribution.attributed:
        return f'Cause not attributed: {attribution.reason.replace("_", " ")}.', ('attribution',)
    mode, _, group = attribution.mode.partition(' ')
    sym = _mode_longitude(mode)
    amplitude = f'|c| {fmt(attribution.forcing_e)}' + (f', fitted for the cluster {group.strip("()")}' if group else '')
    beat = f'the beat 2π/|b_ϖ − {mode}| {years(attribution.beat_period_yr)} = P_cyc'
    if attribution.scope == 'episodes':
        parts = [
            f'Induced, inside the librating episodes, by the forced eccentricity of mode {mode} ({amplitude}), which holds ϖ '
            f'there (median |e_free|/|e_forced| ≤ {fmt(gate.rho_episodes)} in every episode): the free vector circles it with {beat}.'
        ]
    else:
        parts = [
            f'Induced by the forced eccentricity of mode {mode} ({amplitude}), which holds ϖ: the free vector (e_free/|c| '
            f'{fmt(attribution.free_over_forcing)}) circles it with {beat}.'
        ]
    used = ['attribution']
    centre = _centre_cause(getattr(analysis, 'centre_line', None))
    if centre:
        parts.append(centre)
        used.append('centre')
    node = _node_lock_words(zlk.node_lock) if zlk.node_lock.candidate == mode else ''
    if node:
        parts.append(node[0].upper() + node[1:] + '.')
        used.append('node_lock')
    elif attribution.scope == 'episodes' and finite(attribution.node_minus_mode_arcsec_yr):
        parts.append(
            f'The centre {sym} − Ω turns with the node: Ω̇ − {mode} {attribution.node_minus_mode_arcsec_yr:+.3g}″/yr, '
            f'{fmt(attribution.node_minus_mode_turns)} turns over the record.'
        )
    return ' '.join(parts), tuple(used)


def zlk_column_lines(analysis) -> List[str]:
    """The measured facts of a ZLK row that change no status (e-i coupling, attribution, the centre's slow line, the
    node lock), and the slow line of a modulated secular angle; one line each, '' lines left out, and none the cause
    sentence (`zlk_cause`) already states."""
    lines = []
    used = zlk_cause(analysis)[1]
    zlk = getattr(analysis, 'zlk', None)
    if zlk is not None:
        lines.append(_ei_line(zlk.ei))
        if 'attribution' not in used:
            lines.append(_attribution_line(zlk.attribution, getattr(analysis, 'gate', None)))
        if 'node_lock' not in used:
            node = _node_lock_words(zlk.node_lock)
            lines.append(f'{node} (a flag)' if node else '')
    if 'centre' not in used:
        lines.append(_centre_line(getattr(analysis, 'centre_line', None)))
    return [line for line in lines if line]


# --------------------------------------------------------------------------------------------- secular and ZLK: slots
#
# The slotted explanation of a secular or Lidov-Kozai verdict (W28 E9; cards g10-2 style (i), g10-8, g09-5 C, g09-3,
# g09-6, g09-7, g10-4b): the same slots as the MMR one (`mmr_bullets`) plus the time scales (`SLOT_SCALE`). Every
# sentence reads the verdict, the ledger or a column of the row; a number a bullet prints leaves the numbers table for
# a pointer (`Bullet.columns`, `secular_plots.draw_long_period_explained`).

TURN_ARCSEC = 1296000.0  # arcsec in one turn: a rate of f arcsec/yr turns once in 1296000 / |f| yr
# Kozai's critical inclination arccos sqrt(3/5) = 39.23 deg: the quadrupole ZLK of a body under an outer perturber on a
# circular orbit makes omega librate only above it (Kozai 1962). A literature reference printed in the text, never a
# test: no status reads it (inner perturbers and crossing orbits move the domain, Michel & Thomas 1996).
ZLK_CRITICAL_INC_DEG = math.degrees(math.acos(math.sqrt(0.6)))
PRECESSION_SENTENCE = 'Planetary periods are precession times; they are not the period of a libration.'  # card g09-5 C
TEXT_POINTER = '↑ in the text'  # what the numbers table prints for a number a bullet prints (W20 CS11)
# the deciding episode's numbers as `libration_numbers` prints them
EPISODE_NUMBER_COLUMNS = ('ep_centre_mod_2pi_deg', 'ep_n_tp', 'ep_period_med_yr', 'ep_half_amp_med_deg', 'ep_half_amp_max_deg')
OBJECTION_HEAD = 'Objection, not deciding the status: '
RESPONSE_NAMES = {'e': 'e', 'i': 'sin(i/2)'}  # the elements of `Response.variables` in words
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
    'drift': 'The centre moves one way only: there is no fixed centre to librate about in this record.',
    'near_separatrix': 'No libration: the angle slips over the top.',
    'insufficient_time': 'Fewer turning points than one full cycle needs: the shape cannot tell a libration from a slow circulation.',
}


def _row_float(row: Mapping[str, Any], column: str) -> float:
    value = row.get(column)
    return float(value) if finite(value) else float('nan')


def record_span_yr(analysis) -> float:
    """T: the span of the analysed record [yr] (nan without a series)."""
    series = analysis.series
    return float(series.t_yr[-1] - series.t_yr[0]) if series is not None and len(series.t_yr) > 1 else float('nan')


def librating_episode_rows(analysis, episodes: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """The episodes.csv rows of the librating episodes: the predicate of sec_n_librating_episodes
    (`secular.decide.librating_episodes`: accepted, not faster than the type's shortest libration, the centre not
    running a turn); every accepted row when the analysis has no geometry (a row restored without arrays)."""
    accepted = [row for row in episodes if episode_accepted(row)]
    if analysis.geometry is None:
        return accepted
    ids = [episode.episode_id for episode in secular_librating_episodes(analysis.geometry)]
    return [row for row in accepted if any(_same_id(row.get('episode_id'), i) for i in ids)]


def _long_noun(analysis) -> str:
    return 'The proper angle' if str(analysis.verdict.resonance_type) == 'secular' else 'ω'


def _decisive_row(analysis, eps: Episodes) -> Optional[Mapping[str, Any]]:
    """The deciding episode's row when it is a librating one (its numbers are the row's ep_* columns), else None."""
    row = eps.decisive
    return row if row is not None and _same_id(row.get('episode_id'), analysis.decisive_episode_id) else None


def long_period_sigma_line(analysis, eps: Episodes, others: int) -> Tuple[str, Tuple[str, ...]]:
    """(what the angle does, the columns it prints): a libration over the record with the deciding episode's numbers;
    an episodic angle by its librating episodes only (card g09-6: a faster run is not counted) and the deciding one;
    a stretch too short or too small to count with its numbers; else the shape in words."""
    v, noun = analysis.verdict, _long_noun(analysis)
    angle = str(v.angle_status)
    decisive = _decisive_row(analysis, eps)
    columns = EPISODE_NUMBER_COLUMNS if decisive is not None else ()
    if angle == 'libration':
        return f'{noun} librates over the whole record {libration_numbers(decisive or eps.decisive or {})}.', columns
    if angle == 'episodic':
        shown = decisive or eps.decisive
        if shown is None:
            return f'{noun} librates in part of the record only.', ()
        span = f'{years(shown.get("t_start_yr"))}–{years(shown.get("t_end_yr"))}'
        n = len(eps.rows)
        if n <= 1:
            text = f'{noun} librates from {span} only ({percent(eps.covered_fraction())} of the record), {libration_numbers(shown)}'
        else:
            text = (
                f'{noun} librates in {n} separate episodes covering {percent(eps.covered_fraction())} of the record; '
                f'the deciding one {span}, {libration_numbers(shown)}'
            )
        if others:
            text += f'; {_plural(others, "more accepted run")} not counted (faster than the shortest libration, or the centre turns)'
        return text + '.', columns
    if angle == 'insufficient_time' and decisive is not None:
        span = f'{years(decisive.get("t_start_yr"))}–{years(decisive.get("t_end_yr"))}'
        return f'{noun} swings once, {span}: {libration_numbers(decisive)}.', columns
    return _long_period_angle_sentence(analysis, eps, None), ()


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
    return abs(resolution - TURN_ARCSEC / span_yr) <= 1e-3 * resolution


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
    found = EI_WORDS.get(coupling, coupling.replace('_', ' ') or 'not measured')
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
    episodic ω without a gate verdict), else the shape of the angle (the swing within the phase uncertainty, a centre
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
        return GATE_DECISIVE.get(outcome, f'free-ω gate: {outcome.replace("_", " ")}.'), ('SZ.G',), ()
    if str(v.angle_subtype) == 'phase_unresolved':
        text, columns = _phase_sentence(analysis, row)
        return text, ('G5p',), columns
    if 'centre_turned_insufficient' in rules:
        return (
            'The centre turned back within less than one revolution: an unfinished slow oscillation, not a drift; the record '
            'is too short to close one cycle.',
            ('G5s',),
            (),
        )
    if outcome == 'not_applicable' and angle == 'circulation':
        return 'ω circulates: there is no libration for the free-ω gate to check.', (), ()
    return SECULAR_DECISIVE.get(angle, f'{str(v.reason).replace("_", " ")}.'), (), ()


def long_period_objections(analysis, row: Mapping[str, Any]) -> str:
    """'Objection, not deciding the status: …' (card g10-8: what did not decide moves from the grey line into the text):
    e (or sin(i/2)) does not follow the angle; only one independent centre estimate; a ZLK libration without the ZLK
    exchange of e and i at its deciding period (unless that exchange decided, D10). '' for none."""
    v = analysis.verdict
    items = []
    if str(v.checks.get('check_secular_e_response', '')) == 'fail':
        response = analysis.response
        names = [RESPONSE_NAMES[name] for name in (response.variables if response is not None else ()) if name in RESPONSE_NAMES]
        names = names or ['e']
        items.append(f'{" and ".join(names)} {"does" if len(names) == 1 else "do"} not follow the angle (the response test fails)')
    estimates = row.get('ep_centre_estimates')
    if str(v.angle_status) in ('libration', 'episodic', 'insufficient_time') and finite(estimates) and int(float(estimates)) == 1:
        items.append('only one independent centre estimate: one cycle cannot show that the centre stands still')
    coupling = _text(row.get('zlk_ei_coupling'))
    if (
        str(v.resonance_type) == 'lidov_kozai'
        and str(v.status) in ('resonant', 'transient')
        and coupling not in ('', 'zlk', 'not_measured')
        and 'zlk_gate_unconfirmed' not in set(v.rules_fired)
    ):
        items.append('no ZLK exchange of e and i at the deciding period (the e-i line below)')
    return f'{OBJECTION_HEAD}{"; ".join(items)}.' if items else ''


def zlk_physics_line(analysis, row: Mapping[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    """The physics of a Lidov-Kozai row (card g10-8): the ranges of i and e, where i stands against Kozai's 39.2°
    (`ZLK_CRITICAL_INC_DEG`, a reference, not a test) and how many times the free ω (g − s, free_b_omega) turns over
    the record, |g − s| / (1/T). '' for a secular row."""
    if str(analysis.verdict.resonance_type) != 'lidov_kozai':
        return '', ()
    lo, hi = _row_float(row, 'ctx_inc_min_deg'), _row_float(row, 'ctx_inc_max_deg')
    e_lo, e_hi = _row_float(row, 'ctx_e_min'), _row_float(row, 'ctx_e_max')
    parts, columns = [], []
    if finite(lo) and finite(hi):
        side = 'stays below it' if hi < ZLK_CRITICAL_INC_DEG else ('stays above it' if lo > ZLK_CRITICAL_INC_DEG else 'crosses it')
        ranges = f'i {fmt(lo)}–{fmt(hi)}°' + (f', e {fmt(e_lo)}–{fmt(e_hi)}' if finite(e_lo) and finite(e_hi) else '')
        parts.append(
            f'Physics: {ranges} in the record; the classical ZLK (the quadrupole of an outer planet) makes ω librate only above '
            f'i = {ZLK_CRITICAL_INC_DEG:.1f}° (Kozai 1962; a reference, not a test): i {side}.'
        )
        columns += ['ctx_inc_min_deg', 'ctx_inc_max_deg'] + (['ctx_e_min', 'ctx_e_max'] if finite(e_lo) and finite(e_hi) else [])
    rate = _row_float(row, 'free_b_omega')
    span = record_span_yr(analysis)
    if finite(rate) and finite(span) and span > 0:
        turns = abs(rate) * span / TURN_ARCSEC
        parts.append(f'The free ω (g − s) turns {fmt(turns)} times in the record (|g − s| / (1/T)).')
    else:
        parts.append('The free ω is not measured (no free elements).')
    return ' '.join(parts), tuple(columns)


IDENTIFY_MAX = 3  # rows named for one period seen on the angle


def _identify(period: float, low: float, high: float, resolution: float, frequencies) -> Tuple[str, str]:
    """(the FREQUENCIES rows a period seen on the angle is, nearest first and joined by ' or ', '' for none; the
    nearest row): a row matches when its |rate| lies in the frequency band of the period's range [low, high] widened
    by 1/T (frequencies closer than 1/T are one in the record, so all such rows are named). Display only."""
    f = TURN_ARCSEC / period
    f_lo = TURN_ARCSEC / high if finite(high) and high > 0 else f
    f_hi = TURN_ARCSEC / low if finite(low) and low > 0 else f
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
    when the centre scale decided. `said`: the angle's line already printed the deciding episode's period."""
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


def time_scale_line(analysis, row: Mapping[str, Any], settings, frequencies, said: bool = False) -> Tuple[str, Tuple[str, ...]]:
    """The time scales of the record (card g09-5 C): 1/T; for a secular angle |σ̇|/(1/T), σ̇ its mean rate from the
    free vectors and the planetary constants (the σ row of FREQUENCIES); the search window, from F × P_c (the filter
    guard, or the type's shortest libration when longer) to one full cycle inside the record trimmed by the filter's
    edges; the periods seen on the angle with the FREQUENCIES row they are, or 'unidentified' (`_identify`); and the
    sentence that planetary periods are precession times. `said`: the angle's line printed the deciding period."""
    span = record_span_yr(analysis)
    if not (finite(span) and span > 0):
        return '', ()
    resolution = TURN_ARCSEC / span
    columns = ['smp_span_yr']
    if _resolution_said(row, span):
        parts = [f'Time scales: T = {years(span)}, so 1/T is the resolution above']
    else:
        parts = [f'Time scales: the record resolves 1/T = {fmt(resolution)}″/yr (T = {years(span)})']
    if str(analysis.verdict.resonance_type) == 'secular':
        sigma = next((item for item in frequencies if item.name == 'σ' and finite(item.rate)), None)
        if sigma is not None:
            ratio = abs(sigma.rate) / resolution
            turns = 'turns less than once' if ratio < 1 else 'turns that many times'
            parts.append(f'|σ̇|/(1/T) = {fmt(ratio)}: the mean rate of σ {turns} in the record')
    cutoff = analysis.angle.cutoff_yr if analysis.angle is not None else _row_float(row, 'flt_cutoff_yr')
    if finite(cutoff):
        floor = analysis.geometry.period_floor_yr if analysis.geometry is not None else float('nan')
        guard = settings.filter_min_libration_cutoffs * cutoff
        low = max(guard, floor) if finite(floor) else guard
        high = span - 2 * settings.filter_edge_trim_cutoffs * cutoff
        start = f'{fmt(settings.filter_min_libration_cutoffs)} P_c' if low == guard else 'the shortest libration of the type'
        if high > low:
            parts.append(f'search window {years(low)}–{years(high)} (from {start} to one full cycle inside the trimmed record)')
    seen = []
    for label, period, low, high in _seen_periods(analysis, row, said):
        match, nearest = _identify(period, low, high, resolution, frequencies)
        shown = f'the period of {label}' if label.endswith('(above)') else f'{years(period)} ({label})'
        if match:
            seen.append(f'{shown} ≈ {match}')
        else:
            seen.append(f'{shown}: unidentified' + (f', the nearest row is {nearest}' if nearest else ''))
    parts.append('seen on the angle: ' + '; '.join(seen) if seen else 'no period seen on the angle')
    return '; '.join(parts) + '. ' + PRECESSION_SENTENCE, tuple(columns)


LONG_VERDICT_LINES = {
    ('secular', 'resonant'): 'RESONANT: the proper angle librates over the whole record.',
    ('secular', 'transient'): 'TRANSIENT: the proper angle librates in part of the record only.',
    ('secular', 'non_resonant'): 'NOT IN THIS RESONANCE: the proper angle circulates.',
    ('lidov_kozai', 'resonant'): 'RESONANT: ω and the free ω librate: a Lidov-Kozai libration over the whole record.',
    ('lidov_kozai', 'transient'): 'TRANSIENT: ω librates as a Lidov-Kozai libration in part of the record only.',
}


def long_period_verdict_line(analysis) -> str:
    """The verdict in plain words (the slot 'verdict') of a secular or Lidov-Kozai row."""
    v = analysis.verdict
    kind, status, reason = str(v.resonance_type), str(v.status), str(v.reason)
    if kind == 'lidov_kozai' and status == 'non_resonant':
        if reason == 'kinematic_libration':
            return 'NOT A LIDOV-KOZAI LIBRATION: ω librates, but the free ω winds.'
        return 'NOT IN A LIDOV-KOZAI LIBRATION: ω circulates.'
    if status == 'uncertain':
        if str(v.angle_subtype) == 'phase_unresolved':
            return 'UNCERTAIN: the swing is within the error of the linear mode phases.'
        if reason == 'gate_unavailable':
            verb = 'librates' if str(v.angle_status) == 'libration' else 'stays bounded'
            return f'UNCERTAIN: ω {verb}, but the free-ω gate gives no verdict.'
        return 'UNCERTAIN: the record does not decide.'
    if kind == 'lidov_kozai' and status in ('resonant', 'transient') and (has_mark(v, 'gate_unavailable') or has_mark(v, 'gate_disabled')):
        where = 'over the whole record' if status == 'resonant' else 'in part of the record'
        return f'{status.upper()}: ω librates {where}; the free-ω gate gives no verdict, so it is not confirmed as Lidov-Kozai.'
    return LONG_VERDICT_LINES.get((kind, status), f'{status_head(status).upper()}: {reason.replace("_", " ")}.')


def long_period_bullets(analysis, settings, episodes, row, checks, frequencies) -> List[Bullet]:
    """The slotted explanation of a secular or Lidov-Kozai verdict (W28 E9): the verdict in plain words, what the angle
    does (librating episodes only), the decisive argument, the objections that did not decide, the cause (ZLK: the
    physics line and, for a kinematic libration, `zlk_cause`), the time scales, what would change the verdict and the
    radial overlap."""
    v = analysis.verdict
    span = record_span_yr(analysis)
    rows = librating_episode_rows(analysis, episodes)
    eps = Episodes(rows, span, analysis.decisive_episode_id)
    others = sum(1 for item in episodes if episode_accepted(item)) - len(rows)
    sigma, sigma_columns = long_period_sigma_line(analysis, eps, others)
    decisive, ids, decisive_columns = long_period_decisive_line(analysis, row)
    physics, physics_columns = zlk_physics_line(analysis, row)
    said = 'ep_period_med_yr' in sigma_columns
    scale, scale_columns = time_scale_line(analysis, row, settings, frequencies, said)
    out = [
        Bullet(SLOT_VERDICT, long_period_verdict_line(analysis)),
        Bullet(SLOT_SIGMA, sigma, columns=sigma_columns),
        Bullet(SLOT_DECISIVE, decisive, ids, columns=decisive_columns),
        Bullet(SLOT_OBJECTION, long_period_objections(analysis, row)),
        Bullet(SLOT_CAUSE, zlk_cause(analysis)[0]),
        Bullet(SLOT_CAUSE, physics, columns=physics_columns),
        Bullet(SLOT_SCALE, scale, columns=scale_columns),
        Bullet(SLOT_CHANGE, change_line(v, checks)),
        Bullet(SLOT_ORBIT, orbit_line(row)),
    ]
    return [bullet for bullet in out if bullet.text]


def bullet_columns(text: VerdictText) -> Tuple[str, ...]:
    """The columns whose numbers the bullets print (they leave the numbers table for `TEXT_POINTER`)."""
    return tuple(dict.fromkeys(column for bullet in text.bullets for column in bullet.columns))


def explain_long_period(
    analysis, settings, episodes: Sequence[Mapping[str, Any]], row: Optional[Mapping[str, Any]] = None, frequencies=None
) -> VerdictText:
    """The explanation of a secular or Lidov-Kozai verdict (`LongPeriodAnalysis`); `episodes` = its episodes.csv rows,
    `row` its summary row with the caller's extra columns (default: the analysis' own), `frequencies` the rows of the
    figure's FREQUENCIES table (`secular_plots.long_period_frequencies`; default: built from `row`). The figure draws
    the slotted `bullets` (W28 E9); `sentences` stay the paragraph form."""
    v = analysis.verdict
    row = dict(row) if row is not None else analysis.summary_row()
    if frequencies is None:
        from resonances.plotting.secular_plots import long_period_frequencies

        frequencies = long_period_frequencies(analysis, row)
    span = record_span_yr(analysis)
    eps = Episodes(librating_episode_rows(analysis, episodes), span, analysis.decisive_episode_id)
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
    steps.append(replace(_shape(f'{shape} → {v.angle_status}'), number='0', question='Angle shape'))
    gate = getattr(analysis, 'gate', None)
    if gate is not None and _text(gate.outcome):
        steps.append(Step('gate', 'free-ω gate', StepState.LEAF, f'{gate.outcome} → {v.status}', '1', *GATE_STEP))
    else:
        outcome = f'{v.angle_status} → {v.status} (the shape of the angle decides)'
        steps.append(Step('leaf', 'the angle’s shape', StepState.LEAF, outcome, '1', *SHAPE_LEAF_STEP))
    bullets = long_period_bullets(analysis, settings, episodes, row, checks, frequencies)
    info = []
    value = str(v.checks.get('check_secular_e_response', ''))
    if value == 'pass':  # a failed response is an objection in the bullets (card g10-8) and leaves the grey line
        info.append(f'{_response_words(analysis)}: {value} (a column only)')
    elif value == 'fail' and not bullets:
        info.append(f'{_response_words(analysis)}: {value} (a column only)')
    info += zlk_column_lines(analysis)
    head = [
        long_period_sigma_line(analysis, eps, 0)[0],
        _long_period_status_sentence(analysis, settings, episodes),
        zlk_cause(analysis)[0],
    ]
    subtitle = f'reason {v.reason}' + (f' · origin {v.libration_origin}' if str(v.libration_origin) else '')
    text = _verdict_text(v, subtitle, head, steps, checks, info)
    for check in checks:  # the drawn text has no confidence sentence: each LIMIT row names its other reading (CS11)
        check.said = False
    bullets = policy_block(v) + bullets  # W29 §7: the minimal policy's four lines first
    return replace(text, decided_at=decided_at('', v.status, steps), docs=DOCS_LONG_PERIOD, bullets=bullets)


# (question, why, ids) of the leaf of a secular or Lidov-Kozai row: the free-ω gate (ZLK), else the angle's shape
GATE_STEP = (
    'Does the free ω librate too?',
    'an osculating ω also librates with a forced eccentricity; only a librating free ω is the ZLK libration.',
    ('SZ.G',),
)
SHAPE_LEAF_STEP = ('The angle’s shape decides', 'the status of a secular angle is that of its shape.', ())
