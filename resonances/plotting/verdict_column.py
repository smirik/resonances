"""The right column of the verdict figures as a model of ten blocks (the author, 2026-11-04: the spec of Claude Desktop,
its visual part adopted, its science checked first; decisions D1-D10 in dev/decisions.md, Figures).

Every family (two-body, three-body, co-orbital, secular, ZLK) fills the same blocks in the same order:

    header      status plaque, the level as four dots, one action chip
    subtitle    the resonance and the subtype in words; the type and where it was decided
    key_facts   'label : value' lines (σ, period, episodes, resonance, orbit, overlaps, run)
    why         three to five plain sentences with panel letters, then 'Would change the verdict: …'
    confidence  'Confidence: high — not certain because …' and one pill per input that set the level
    checks      the steps of the tree in order, the margins nested under their step, one ◆ on the deciding step
    evidence    what makes the cycles / is σ the own pendulum: icon, label, value
    caveats     sentences a reader must know; 'none' when empty
    numbers     DECISIVE first, then fixed groups of labeled numbers
    footer      the codes: status code, rules, reason/origin/veto, raw flags, settings hash, docs

The builders (`verdict_column_mmr`, `verdict_column_long`) fill a `Column`; `verdict_column_draw` draws it;
`column_text` writes it as text (the same blocks, nothing shortened: the figure may trim, the text never does).

Inline markup in every text field (the drawing and the text read it; `plain` strips it):
    **x**   bold                          !!x!!   bold in the status color
    [[x]]   numbers, gray mono '(x)'      <<x>>   panel letters, gray '(x)'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# -- vocabulary (one table of colours: the spec's plaque colours, white text >= 4.5:1, decision D4)

STATUS_WORDS = {
    'resonant': 'RESONANT',
    'transient': 'TRANSIENT',
    'near_separatrix': 'NEAR-SEPARATRIX',
    'non_resonant': 'NON-RESONANT',
    'uncertain': 'UNCERTAIN',
    'invalid': 'INVALID',
}
LEVELS = ('certain', 'high', 'medium', 'low')  # best first; '' = no level (uncertain, invalid)
LEVEL_DOTS = {'certain': 4, 'high': 3, 'medium': 2, 'low': 1, '': 0}
NO_LEVEL = 'no level'

# icons of checks, pills and evidence rows: the colour says how much to look, not for or against the resonance
OK, THIN, FAIL, MISSING, QUIET, DECIDES = '✓', '⚠', '✗', '?', '–', '◆'
ICONS = (OK, THIN, FAIL, MISSING, QUIET, DECIDES)
ICON_MEANING = {
    OK: 'clean',
    THIN: 'marginal',
    FAIL: 'failed',
    MISSING: 'not measured',
    QUIET: 'not needed',
    DECIDES: 'decides',
}

# action chips (exactly one per column; the text never repeats it)
CHIP_NONE, CHIP_GLANCE, CHIP_REVIEW, CHIP_EXTEND, CHIP_RESAMPLE = 'none', 'glance', 'review', 'extend', 'resample'
CHIP_KINDS = (CHIP_NONE, CHIP_GLANCE, CHIP_REVIEW, CHIP_EXTEND, CHIP_RESAMPLE)

BLOCKS = ('header', 'subtitle', 'key_facts', 'why', 'confidence', 'checks', 'evidence', 'caveats', 'numbers', 'footer')
BLOCK_TITLES = {
    'key_facts': 'Key facts',
    'why': 'Why',
    'confidence': 'Confidence',
    'checks': 'Checks',
    'evidence': 'Evidence',
    'caveats': 'Caveats',
    'numbers': 'Numbers',
    'footer': 'Record',
}
NONE_CAVEAT = 'none'

_MARKUP = re.compile(r'\*\*(.+?)\*\*|!!(.+?)!!|\[\[(.+?)\]\]|<<(.+?)>>')
STYLE_PLAIN, STYLE_BOLD, STYLE_STATUS, STYLE_NUMBERS, STYLE_PANELS = 'plain', 'bold', 'status', 'numbers', 'panels'


def spans(text: str) -> List[Tuple[str, str]]:
    """The text cut into (piece, style) by the inline markup; numbers and panel letters get their parentheses here."""
    out: List[Tuple[str, str]] = []
    position = 0
    for match in _MARKUP.finditer(text):
        if match.start() > position:
            out.append((text[position : match.start()], STYLE_PLAIN))
        bold, status, numbers, panels = match.groups()
        if bold is not None:
            out.append((bold, STYLE_BOLD))
        elif status is not None:
            out.append((status, STYLE_STATUS))
        elif numbers is not None:
            out.append((f'({numbers})', STYLE_NUMBERS))
        else:
            out.append((f'({panels})', STYLE_PANELS))
        position = match.end()
    if position < len(text):
        out.append((text[position:], STYLE_PLAIN))
    return out


def plain(text: str) -> str:
    """The text without markup, as the text export prints it."""
    return ''.join(piece for piece, _ in spans(text))


def numbers(text: str) -> str:
    """'[[text]]', or '' for an empty text: the gray numbers after a statement."""
    return f'[[{text}]]' if text else ''


def panels(letters: Sequence[str]) -> str:
    """'<<b, k>>' for the panel letters, '' for none."""
    letters = [letter for letter in letters if letter]
    return f'<<{", ".join(letters)}>>' if letters else ''


# -- the blocks


@dataclass(frozen=True)
class Chip:
    """The action: its kind (`CHIP_KINDS`) and its words ('extend to 200 kyr', 'glance at (b), (i)')."""

    kind: str
    text: str


@dataclass(frozen=True)
class Header:
    status: str  # the Status value ('non_resonant')
    level: str  # a LEVELS word, or '' (no level)
    chip: Chip

    @property
    def word(self) -> str:
        return STATUS_WORDS.get(self.status, self.status.upper())

    @property
    def level_word(self) -> str:
        return self.level or NO_LEVEL


@dataclass(frozen=True)
class Subtitle:
    title: str  # '3M+3J−7+0+0+1 · its own libration in 2 episodes; still librating at the end'
    detail: str  # 'three-body Mars–Jupiter · decided at step 6 (behaviour)'


@dataclass(frozen=True)
class Fact:
    label: str
    value: str


@dataclass(frozen=True)
class Why:
    sentences: Tuple[str, ...]
    would_change: str = ''  # 'Would change the verdict: …' / 'Needed: …' (the label included)


@dataclass(frozen=True)
class Pill:
    """One input of the level: ✓ it allows more than the shown level, ⚠ it holds the level down (`ceiling` says to
    what), ? not measured, ✗ it failed and the level reads it; `key` = the policy's input name (N, M, S, V, F, …)."""

    icon: str
    text: str
    key: str = ''
    ceiling: str = ''


@dataclass(frozen=True)
class Confidence:
    level: str  # LEVELS word or '' (no level)
    because: str  # 'not certain because …' / 'nothing holds it below certain' / for no level: what is missing
    pills: Tuple[Pill, ...] = ()


@dataclass(frozen=True)
class SubCheck:
    icon: str
    text: str  # 'centre fixed' (markup allowed)
    numbers: str = ''  # 'shift/A 0.07 ≤ 0.5' (drawn grey mono in parentheses)
    rule_id: str = ''  # the small grey tag (decision D3)


@dataclass(frozen=True)
class CheckLine:
    """One step: icon, its number ('0'…'6', 'Q'), the question, the short answer, its numbers and the sub-checks.
    `decides`: the deciding step (icon ◆, filled in the status color); `quiet`: drawn gray (not reached, not
    needed); a run of not-reached steps is one line '3–6 · a swings · drift · … — not reached'."""

    icon: str
    number: str
    question: str
    answer: str = ''
    numbers: str = ''
    rule_id: str = ''
    children: Tuple[SubCheck, ...] = ()
    decides: bool = False
    quiet: bool = False


@dataclass(frozen=True)
class EvidenceRow:
    icon: str
    label: str
    value: str


@dataclass(frozen=True)
class Evidence:
    title: str  # 'Evidence: is this 2J−1’s own pendulum?'
    rows: Tuple[EvidenceRow, ...] = ()


@dataclass(frozen=True)
class NumberGroup:
    """A group of labeled numbers ('LIBRATION', 'ORBIT (min / med / max)'); `table`: a header row of a wide table
    (FREQUENCIES) whose rows are then tuples of cells joined by the drawing."""

    title: str
    rows: Tuple[Tuple[str, str], ...] = ()
    table: Tuple[str, ...] = ()
    cells: Tuple[Tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class Numbers:
    decisive: Tuple[str, ...] = ()  # 'energy in own well, episode #0: period law and V_E ≈ V_F hold [R0]'
    groups: Tuple[NumberGroup, ...] = ()


@dataclass(frozen=True)
class Column:
    family: str  # 'mmr2', 'mmr3', 'coorbital', 'secular', 'zlk'
    header: Header
    subtitle: Subtitle
    key_facts: Tuple[Fact, ...] = ()
    why: Why = field(default_factory=lambda: Why(()))
    confidence: Confidence = field(default_factory=lambda: Confidence('', ''))
    checks: Tuple[CheckLine, ...] = ()
    evidence: Tuple[Evidence, ...] = ()
    caveats: Tuple[str, ...] = ()
    numbers: Numbers = field(default_factory=Numbers)
    footer: Tuple[str, ...] = ()
    decided_rule: str = ''  # the rule of the ◆ step, as the footer names it (invariant 2)


# -- the text export


def _check_lines(line: CheckLine) -> List[str]:
    head = f'{line.icon} {line.number:>2} · {plain(line.question)}'
    rest = ' '.join(part for part in (plain(line.answer), f'({plain(line.numbers)})' if line.numbers else '') if part)
    tag = f' [{line.rule_id}]' if line.rule_id else ''
    out = [f'{head}{" " + rest if rest else ""}{tag}']
    for child in line.children:
        tag = f' [{child.rule_id}]' if child.rule_id else ''
        detail = f' ({plain(child.numbers)})' if child.numbers else ''
        out.append(f'        {child.icon} {plain(child.text)}{detail}{tag}')
    return out


def _group_lines(group: NumberGroup) -> List[str]:
    out = [group.title]
    if group.table:
        widths = [max(len(cell) for cell in column) for column in zip(group.table, *group.cells)]
        for cells in (group.table, *group.cells):
            out.append('  ' + '  '.join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip())
    width = max((len(label) for label, _ in group.rows), default=0)
    out += [f'  {label.ljust(width)}  {value}' for label, value in group.rows]
    return out


def _header_text(column: Column) -> List[str]:
    h = column.header
    dots = '●' * LEVEL_DOTS[h.level] + '○' * (4 - LEVEL_DOTS[h.level]) if h.level else '- - - -'
    return [f'[{h.word}]  {dots} {h.level_word}  → {h.chip.text}']


def _facts_text(column: Column) -> List[str]:
    width = max((len(fact.label) for fact in column.key_facts), default=0)
    return [f'{fact.label.ljust(width)}  {plain(fact.value)}' for fact in column.key_facts]


def _why_text(column: Column) -> List[str]:
    out = [' '.join(plain(sentence) for sentence in column.why.sentences)]
    return out + ([plain(column.why.would_change)] if column.why.would_change else [])


def _confidence_text(column: Column) -> List[str]:
    c = column.confidence
    out = [f'Confidence: {c.level or NO_LEVEL}' + (f' — {plain(c.because)}' if c.because else '')]
    return out + [f'  {pill.icon} {plain(pill.text)}' for pill in c.pills]


def _evidence_text(column: Column) -> List[str]:
    out: List[str] = []
    for evidence in column.evidence:
        out.append(plain(evidence.title))
        width = max((len(row.label) for row in evidence.rows), default=0)
        out += [f'  {row.icon} {row.label.ljust(width)}  {plain(row.value)}' for row in evidence.rows]
    return out


def _numbers_text(column: Column) -> List[str]:
    out = [f'DECISIVE {plain(line)}' for line in column.numbers.decisive]
    for group in column.numbers.groups:
        out += _group_lines(group)
    return out


_BLOCK_TEXT = {
    'header': _header_text,
    'subtitle': lambda column: [plain(column.subtitle.title), plain(column.subtitle.detail)],
    'key_facts': _facts_text,
    'why': _why_text,
    'confidence': _confidence_text,
    'checks': lambda column: [text for line in column.checks for text in _check_lines(line)],
    'evidence': _evidence_text,
    'caveats': lambda column: [f'• {plain(text)}' for text in column.caveats] or [NONE_CAVEAT],
    'numbers': _numbers_text,
    'footer': lambda column: [' · '.join(plain(item) for item in column.footer)] if column.footer else [],
}


def block_text(column: Column, name: str) -> List[str]:
    """The lines of one block (`BLOCKS`) as the export prints them, without the block's title."""
    if name not in _BLOCK_TEXT:
        raise ValueError(f'unknown block {name!r}')
    return _BLOCK_TEXT[name](column)


def shown(column: Column, name: str) -> bool:
    """Whether a block is drawn: a block without content is not, except Caveats ('none')."""
    if name in ('header', 'subtitle', 'caveats'):
        return True
    if name == 'numbers':
        return bool(column.numbers.decisive or column.numbers.groups)
    if name == 'why':
        return bool(column.why.sentences)
    if name == 'confidence':
        return True
    return bool(getattr(column, name))


def column_text(column: Column, title: str = '') -> str:
    """The whole column as text, block by block in `BLOCKS` order (the figure's order); nothing is shortened."""
    out = [title] if title else []
    for name in BLOCKS:
        if not shown(column, name):
            continue
        heading = BLOCK_TITLES.get(name)
        if heading and name not in ('confidence', 'evidence'):  # those print their own head line
            out.append(f'— {heading}')
        out += block_text(column, name)
    return '\n'.join(out) + '\n'


# -- the invariants (the spec's 1-8 as the author kept them; tests run them on every built column)

CODE_NAME = re.compile(r'\b(?:ctx|rsp|ep|geo|sep|kin|smp|flt|str|sub|pol|zlk|sec)_[a-z0-9_]+\b')
W_CODE = re.compile(r'\bW\d{2}\b')
POINTER = '↑'


def _header_problems(column: Column) -> List[str]:
    out: List[str] = []
    h = column.header
    if h.status not in STATUS_WORDS:
        out.append(f'unknown status {h.status!r}')
    if h.level not in LEVELS + ('',):
        out.append(f'unknown level {h.level!r}')
    if (h.level == '') != (h.status in ('uncertain', 'invalid')):
        out.append(f'level {h.level!r} on status {h.status}')
    if column.confidence.level != h.level:  # 1: one verdict object
        out.append(f'confidence {column.confidence.level!r} differs from the header {h.level!r}')
    if h.chip.kind not in CHIP_KINDS:
        out.append(f'unknown chip {h.chip.kind!r}')
    if h.level and h.level != 'certain' and not column.confidence.because:
        out.append('a level below certain without its reason')
    return out


def _check_problems(column: Column) -> List[str]:
    out: List[str] = []
    decides = [line for line in column.checks if line.decides]
    if column.checks and len(decides) != 1:  # 7: exactly one ◆
        out.append(f'{len(decides)} deciding steps')
    for line in column.checks:  # 7: one icon set
        if line.icon not in ICONS:
            out.append(f'step {line.number}: icon {line.icon!r}')
        if line.decides != (line.icon == DECIDES):
            out.append(f'step {line.number}: ◆ and decides disagree')
        out += [f'step {line.number}: sub-check icon {c.icon!r}' for c in line.children if c.icon not in ICONS[:-1]]
    if decides and column.decided_rule and decides[0].rule_id and decides[0].rule_id != column.decided_rule:  # 2
        out.append(f'◆ names {decides[0].rule_id}, the footer {column.decided_rule}')
    return out


def _pill_problems(column: Column) -> List[str]:
    out: List[str] = []
    level = column.header.level
    for pill in column.confidence.pills:  # 4: a ⚠ pill is an input of the level and says to what it holds it
        if pill.icon not in (OK, THIN, FAIL, MISSING, QUIET):
            out.append(f'pill icon {pill.icon!r}')
        if pill.icon == THIN and not (pill.key and pill.ceiling):
            out.append(f'⚠ pill without its input: {plain(pill.text)!r}')
        # ✓ only for an input in its best state (it allows certain; a full veto: high), – never for one holding the level
        if pill.icon == OK and pill.ceiling not in ('', 'certain') and (pill.key, pill.ceiling) != ('veto', 'high'):
            out.append(f'✓ pill of a limited input: {plain(pill.text)!r} (at most {pill.ceiling})')
        if pill.icon == QUIET and level in LEVELS and pill.ceiling in LEVELS:
            if LEVELS.index(pill.ceiling) >= LEVELS.index(level):
                out.append(f'– pill that holds the level: {plain(pill.text)!r}')
    return out


def _code_problems(column: Column) -> List[str]:
    out: List[str] = []
    for name in (name for name in BLOCKS if name not in ('numbers', 'footer')):  # 5: codes only in footer, DECISIVE
        for text in block_text(column, name):
            if POINTER in text:
                out.append(f'{name}: pointer {text!r}')
            out += [f'{name}: code {match!r}' for match in CODE_NAME.findall(text) + W_CODE.findall(text)]
    for group in column.numbers.groups:
        for label, value in group.rows:
            if POINTER in value or CODE_NAME.search(label):
                out.append(f'numbers {group.title}: {label!r} {value!r}')
    return out


def _chip_problems(column: Column) -> List[str]:
    chip = column.header.chip  # 6: the action once
    words = plain(chip.text)
    if chip.kind == CHIP_NONE or not words:
        return []
    body = '\n'.join(text for name in BLOCKS if name not in ('header', 'footer') for text in block_text(column, name))
    return [f'the action {words!r} repeated'] if words in body else []


def problems(column: Column) -> List[str]:
    """What breaks the column's contract; [] when it holds. Read by the tests and by the figure builder's debug log."""
    checks = (_header_problems, _check_problems, _pill_problems, _code_problems, _chip_problems)
    return [problem for check in checks for problem in check(column)]


def find(column: Column, number: str) -> Optional[CheckLine]:
    """The step with this number ('0', '5', 'Q'), or None."""
    return next((line for line in column.checks if line.number == number), None)
