"""The right-hand column of the one-page combined figures (V1), shared by the MMR figure (`mmr_verdict_panel`,
`mmr_combined`) and the secular/Lidov-Kozai one (`secular_plots`). Two forms (plot option `combined.numbers`):

- explained (default, layout A′ chosen by the author on 2026-10-16; `draw_explained`): the verdict plaque (the only
  place the status is printed), the explanation in words (`verdict_text`), two columns 'How it was decided' (the steps
  of the status tree) and 'Confidence' (every margin and cap with ✓ ⚠ ✗ – ? · ▼), the level arithmetic, then the
  numbers the reader needs in balanced monospaced columns (`group_lines`: episodes, the libration, the response, ...,
  characteristic periods, orbit, record). Bold numbers were compared by the decision, grey ones belong to a test
  skipped by design;
- all (`draw_text`): every number of the decision, as described below.

The 'all' form:
The text is built from CSV rows only (summary.csv and the type's numbers) and the decision ledger, so the figure and
the CSV always print the same number: each `NumberLine` keeps the column its value came from. A block (`BlockSpec`)
lists its known columns in a fixed order and then every other column of its prefixes, grouped under bold headers.
Numbers are written in CSV units; the unit is read from the column suffix (`_deg`, `_au_yr`, ...) or from a unit
token before a qualifier (`_deg_min`, `_rev_raw`).

- a pure duplicate (the same quantity under another prefix with the same printed value) is printed once, at its
  first place; the kept line lists the other columns in `NumberLine.also`;
- a long list cell is printed as its count, first and last value;
- floats: 4 significant digits in fixed notation from 1e-3 to 1e4, whole numbers above, `1.234e-05` below;
  semi-major axes in au with 7 digits.

Placement: one stream in block order flowing through the text columns of the layout (a block cut by a column end
continues under '<TITLE> (cont.)'); only when every column is full is the font reduced (down to
`VerdictFigureLayout.text_min_font_pt`). Nothing is cut.
"""

from __future__ import annotations

import fnmatch
import math
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from resonances.resonance.analysis.ledger import LedgerEntry

from .style import (
    MARK_COLOURS,
    MONO_CHAR_WIDTH,
    MONO_FONT,
    TEXT_GROUP,
    TEXT_HEADER_FILL,
    TEXT_KEY,
    TEXT_MISSING,
    VerdictFigureLayout,
)

MISSING = '—'
NOT_COMPUTED = 'not computed'
MARK_PASS, MARK_THIN, MARK_FAIL, MARK_NONE = '✓', '~', '✗', '–'

KEY_WIDTH = 15  # characters reserved for a label before its value (a longer label pushes its value right)
PAIR_KEY_WIDTH = 15  # the same in a cell of a two-per-row line: values of both kinds of rows start in one column
BLOCK_GAP = 0.4  # empty space after a block, in lines
FONT_STEP_PT = 0.25  # the text font is reduced in these steps until the columns hold everything
LIST_MAX_ITEMS = 6  # a longer ';' list prints as 'n values: first … last'
CONTINUED = ' (cont.)'

# The verdict block of every resonance type (Verdict.to_flat_dict)
VERDICT_COLUMNS = (
    'status',
    'angle_status',
    'angle_subtype',
    'final_regime',
    'final_regime_start_yr',
    'confidence',
    'confidence_raw',
    'reason',
    'flags',
    'action',
    'extend_to_years',
    'resample_step_yr',
    'libration_origin',
    'origin_detail',
    'owner',
    'multiplet_status',
    'status_code',
)
# Columns printed elsewhere (path, margins) or not at all
SPECIAL_COLUMNS = ('decision_path', 'thin_margins', 'failed_support', 'caps')
# Blocks whose lines are never dropped as duplicates (they are the verdict and its comparisons)
NO_DEDUPE = ('verdict', 'path', 'checks', 'margins')


@dataclass(frozen=True)
class NumberLine:
    """One printed line of the text columns: label, value (formatted), unit, its CSV column and a mark.

    A line with an empty label is free text (a ledger entry, a table row, a reason). `group` is the
    bold group header the line is printed under; `also` lists columns of the same value not printed again."""

    block: str
    label: str
    value: str
    unit: str = ''
    column: str = ''
    mark: str = ''
    group: str = ''
    also: Tuple[str, ...] = ()
    style: str = ''  # the explained column: STYLE_BOLD (the decision compared it) | STYLE_GREY (its test was skipped)
    keep: bool = False  # the explained column: a table row kept in one column with the row before it in its group


@dataclass
class TextBlock:
    name: str
    title: str
    lines: List[NumberLine] = field(default_factory=list)


@dataclass(frozen=True)
class BlockSpec:
    title: str
    columns: Tuple[str, ...] = ()  # known columns, in print order (printed when present in the rows)
    prefixes: Tuple[str, ...] = ()  # every other column with these prefixes follows
    always: Tuple[str, ...] = ()  # printed even when absent from the rows (as '—')
    groups: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()  # (header, fnmatch patterns of columns), first match wins


# Unit of a column from its suffix, longest suffix first
_UNITS = (
    ('_rad_yr_au', 'rad/yr/au'),
    ('_rad_kyr2', 'rad/kyr²'),
    ('_arcsec_yr', '″/yr'),
    ('_deg_yr', '°/yr'),
    ('_rad_kyr', 'rad/kyr'),
    ('_rad_yr', 'rad/yr'),
    ('_au_yr', 'au/yr'),
    ('_per_kyr', '/kyr'),
    ('_deg', '°'),
    ('_rad', 'rad'),
    ('_rev', 'rev'),
    ('_au', 'au'),
    ('_kyr', 'kyr'),
    ('_Myr', 'Myr'),
    ('_yr', 'yr'),
    ('_P', 'P'),
)
# A unit token before a trailing qualifier (`smp_step_rotation_deg_min`, `geo_net_rev_raw`)
_QUALIFIERS = ('min', 'max', 'lo', 'hi', 'med', 'raw', 'filtered', 'predictive', 'numpy', 'pred', 'meas')
_INNER_UNITS = (('deg', '°'), ('rev', 'rev'), ('revs', 'rev'), ('yr', 'yr'), ('kyr', 'kyr'), ('au', 'au'))
_PRECISE_SUFFIX = '_au'  # semi-major axes need 7 digits: a* and a_res differ in the 5th
_PRECISE_DIGITS = 7
_DIGITS = 4
_SMALL = 1e-3  # below: 1.234e-05
_LARGE = 1e4  # at and above: a whole number


# --------------------------------------------------------------------------------------------- cells


def missing(value: Any) -> bool:
    """None, NaN and empty strings are 'not measured' (an empty CSV cell)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _inner_unit(column: str) -> Tuple[str, str]:
    """(unit, token) of a unit written before a trailing qualifier, else ('', '')."""
    parts = column.split('_')
    if len(parts) >= 3 and parts[-1] in _QUALIFIERS:
        for token, unit in _INNER_UNITS:
            if parts[-2] == token:
                return unit, token
    return '', ''


def unit_of(column: str) -> str:
    for suffix, unit in _UNITS:
        if column.endswith(suffix):
            return unit
    return _inner_unit(column)[0]


def label_of(column: str, prefixes: Sequence[str] = ()) -> str:
    """The column without its block prefix (one of `prefixes`) and unit (`smp_unwrap_resid_p99_deg` ->
    `unwrap_resid_p99`, `smp_step_rotation_deg_min` -> `step_rotation_min`)."""
    label = column
    for prefix in prefixes:
        if label.startswith(prefix) and len(label) > len(prefix):
            label = label[len(prefix) :]
            break
    for suffix, _ in _UNITS:
        if label.endswith(suffix) and len(label) > len(suffix):
            return label[: -len(suffix)]
    unit, token = _inner_unit(column)
    if unit == '°':  # 'deg' is the unit only; 'rev' stays a word of the label (net_rev_raw)
        parts = label.split('_')
        if len(parts) >= 3 and parts[-2] == token:
            return '_'.join(parts[:-2] + parts[-1:])
    return label


def _list_items(text: str) -> List[str]:
    return [item.strip() for item in text.split(';') if item.strip()]


def format_cell(column: str, value: Any) -> str:
    """A CSV cell as printed: '—' when empty, integers as integers, floats with 4 significant digits
    (7 for semi-major axes, whole numbers from 1e4, `1.234e-05` below 1e-3), lists with '; ' and a
    list longer than LIST_MAX_ITEMS as 'n values: first … last'."""
    if missing(value):
        return MISSING
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if hasattr(value, 'item') and not isinstance(value, str):
        value = value.item()
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(column, value)
    text = str(value).strip()
    if text in ('True', 'False'):
        return 'yes' if text == 'True' else 'no'
    items = _list_items(text)
    if len(items) > LIST_MAX_ITEMS:
        return f'{len(items)} values: {items[0]} … {items[-1]}'
    return '; '.join(items) if ';' in text else text


def _format_float(column: str, value: float) -> str:
    if math.isinf(value):
        return '∞' if value > 0 else '−∞'
    if value == 0:
        return '0'
    if value.is_integer() and abs(value) < 1e7:
        return str(int(value))
    size = abs(value)
    if column.endswith(_PRECISE_SUFFIX) and size >= 0.1:
        return f'{value:.{_PRECISE_DIGITS}g}'
    if size >= _LARGE:
        return f'{value:.0f}'
    if size < _SMALL:
        return f'{value:.{_DIGITS - 1}e}'
    return f'{value:.{_DIGITS}g}'


def unit_space(column: str) -> str:
    unit = unit_of(column)
    return f' {unit}' if unit else ''


# --------------------------------------------------------------------------------------------- blocks


def ledger_marks(decision_path: Sequence[LedgerEntry]) -> Dict[str, str]:
    """Column -> mark of the last ledger entry that compared it."""
    return {entry.column: entry.mark for entry in decision_path if entry.column}


def block_prefixes(specs: Mapping[str, BlockSpec], extra: Sequence[str] = ()) -> Tuple[str, ...]:
    """The prefixes `label_of` strips: those of the blocks, then `extra`."""
    return tuple(prefix for spec in specs.values() for prefix in spec.prefixes) + tuple(extra)


def claimed_columns(specs: Mapping[str, BlockSpec]) -> set:
    return {column for spec in specs.values() for column in spec.columns}


def owner_block(column: str, specs: Mapping[str, BlockSpec], claimed: set) -> Optional[str]:
    """The block a column belongs to: by explicit listing first, then by prefix."""
    for name, spec in specs.items():
        if column in spec.columns:
            return name
    if column in claimed:
        return None
    for name, spec in specs.items():
        if spec.prefixes and column.startswith(spec.prefixes):
            return name
    return None


def group_of(spec: BlockSpec, column: str) -> str:
    for title, patterns in spec.groups:
        if any(fnmatch.fnmatchcase(column, pattern) for pattern in patterns):
            return title
    return 'other' if spec.groups else ''


def number_line(
    block: str, column: str, row: Mapping[str, Any], marks: Mapping[str, str], group: str = '', prefixes: Sequence[str] = ()
) -> NumberLine:
    value = row.get(column)
    return NumberLine(
        block,
        label_of(column, prefixes),
        format_cell(column, value),
        '' if missing(value) else unit_of(column),
        column,
        marks.get(column, ''),
        group,
    )


def column_lines(
    name: str,
    spec: BlockSpec,
    row: Mapping[str, Any],
    marks,
    claimed,
    specs: Mapping[str, BlockSpec],
    prefixes: Sequence[str] = (),
    block_check: Optional[Mapping[str, str]] = None,
) -> List[NumberLine]:
    """The value lines of one block: its known columns, then every other column of its prefixes, grouped. A block
    named in `block_check` (block -> check column) prints nothing when all its columns are empty."""
    if name in ('checks', 'margins', 'path'):
        return []
    columns = [column for column in spec.columns if column in row or column in spec.always]
    columns += [column for column in row if column not in spec.columns and spec.prefixes and owner_block(column, specs, claimed) == name]
    if name in (block_check or {}) and all(missing(row.get(column)) for column in columns):
        return []
    lines = [number_line(name, column, row, marks, group_of(spec, column), prefixes) for column in columns]
    if not spec.groups:
        return lines
    order = [title for title, _ in spec.groups] + ['other']  # lines grouped, the column order kept inside a group
    return sorted(lines, key=lambda line: order.index(line.group))


def other_lines(
    row: Mapping[str, Any], marks, claimed, specs: Mapping[str, BlockSpec], prefixes: Sequence[str] = (), block: str = 'record'
) -> List[NumberLine]:
    """Columns of no block (identity extras, comments): printed in `block`, never dropped."""
    lines = []
    for column in row:
        if column in SPECIAL_COLUMNS or column.startswith(('mrg_', 'check_')) or owner_block(column, specs, claimed):
            continue
        lines.append(number_line(block, column, row, marks, 'other', prefixes))
    return lines


def _stem(column: str, alias_prefixes: Sequence[str]) -> str:
    for prefix in alias_prefixes:
        if column.startswith(prefix):
            return column[len(prefix) :]
    return column


def drop_duplicates(
    blocks: Dict[str, TextBlock], order: Sequence[str], alias_prefixes: Sequence[str], no_dedupe: Sequence[str] = NO_DEDUPE
) -> None:
    """Print a quantity once: lines with the same stem (column without its namespace in `alias_prefixes`), printed
    value and unit are one quantity. The record-level line is kept (an `ep_*` copy of the decisive episode only when
    nothing else prints it), else the first in print order; the kept line records the others in `also`. Empty values
    and the blocks of `no_dedupe` are never merged."""
    seen: Dict[Tuple[str, str, str], List[Tuple[str, int]]] = {}
    for name in [name for name in order if name in blocks]:
        for index, line in enumerate(blocks[name].lines):
            if line.column and line.label and line.value != MISSING:
                seen.setdefault((_stem(line.column, alias_prefixes), line.value, line.unit), []).append((name, index))
    dropped: Dict[str, set] = {name: set() for name in blocks}
    for places in seen.values():
        if len(places) < 2:
            continue
        protected = [place for place in places if place[0] in no_dedupe]
        record = [place for place in places if not blocks[place[0]].lines[place[1]].column.startswith('ep_')]
        keep = (protected or record or places)[0]
        others = [place for place in places if place != keep and place[0] not in no_dedupe]
        if not others:
            continue
        name, index = keep
        kept = blocks[name].lines[index]
        also = tuple(blocks[other[0]].lines[other[1]].column for other in others)
        blocks[name].lines[index] = NumberLine(**{**kept.__dict__, 'also': kept.also + also})
        for other in others:
            dropped[other[0]].add(other[1])
    for name, indices in dropped.items():
        if indices:
            blocks[name].lines = [line for i, line in enumerate(blocks[name].lines) if i not in indices]


def path_lines(decision_path: Sequence[LedgerEntry]) -> List[NumberLine]:
    return [NumberLine('path', '', entry.describe(), column=entry.column, mark=entry.mark) for entry in decision_path]


CHECK_MARKS = {'pass': MARK_PASS, 'fail': MARK_FAIL}
MARGIN_MARKS = {'full': MARK_PASS, 'thin': MARK_THIN, 'fail': MARK_FAIL}


def check_lines(row: Mapping[str, Any], checks: Sequence[str], prefix: str) -> List[NumberLine]:
    """One line per check column of the type (`checks`), labelled without `prefix`."""
    lines = []
    for column in checks:
        value = row.get(column)
        text = MISSING if missing(value) else str(value)
        label = column[len(prefix) :] if column.startswith(prefix) else column
        lines.append(NumberLine('checks', label, text, column=column, mark=CHECK_MARKS.get(text, MARK_NONE)))
    return lines


def margin_ids(row: Mapping[str, Any], order: Sequence[str] = ()) -> List[str]:
    ids = [key[len('mrg_') :] for key in row if key.startswith('mrg_') and not key.endswith('_value')]
    known = [margin for margin in order if margin in ids]
    return known + sorted(margin for margin in ids if margin not in order)


def margin_lines(row: Mapping[str, Any], order: Sequence[str] = ()) -> List[NumberLine]:
    lines = []
    for margin in margin_ids(row, order):
        state = row.get(f'mrg_{margin}')
        state = '' if missing(state) else str(state)
        column = f'mrg_{margin}_value'
        label = f'{margin} {state}' if state else margin
        lines.append(
            NumberLine('margins', label, format_cell(column, row.get(column)), column=column, mark=MARGIN_MARKS.get(state, MARK_NONE))
        )
    for column in ('thin_margins', 'failed_support', 'caps'):
        lines.append(NumberLine('margins', column, format_cell(column, row.get(column)), column=column))
    return lines


def build_blocks(
    row: Mapping[str, Any],
    specs: Mapping[str, BlockSpec],
    decision_path: Sequence[LedgerEntry],
    checks: Sequence[str],
    check_prefix: str,
    alias_prefixes: Sequence[str],
    margin_order: Sequence[str] = (),
    label_prefixes: Sequence[str] = (),
    empty_reason: Optional[Callable[[str, Mapping[str, Any]], str]] = None,
    extra: Optional[Mapping[str, List[NumberLine]]] = None,
) -> Dict[str, TextBlock]:
    """Every block of `specs` (in their order) from one CSV row and the ledger: the verdict, the decision path, the
    checks, the margins, then the value blocks; `extra` lines are appended to their block (tables). Columns of no
    block go to 'record' when there is one. Pure duplicates are printed once; an empty block says why."""
    prefixes = tuple(label_prefixes) or block_prefixes(specs)
    marks = ledger_marks(decision_path)
    claimed = claimed_columns(specs)
    blocks = {name: TextBlock(name, spec.title) for name, spec in specs.items()}
    for name, spec in specs.items():
        blocks[name].lines.extend(column_lines(name, spec, row, marks, claimed, specs, prefixes))
    if 'path' in blocks:
        blocks['path'].lines = path_lines(decision_path)
    if 'checks' in blocks:
        blocks['checks'].lines = check_lines(row, checks, check_prefix)
    if 'margins' in blocks:
        blocks['margins'].lines = margin_lines(row, margin_order)
    for name, lines in (extra or {}).items():
        blocks[name].lines += list(lines)
    if 'record' in blocks:
        blocks['record'].lines += other_lines(row, marks, claimed, specs, prefixes)
    drop_duplicates(blocks, list(specs), alias_prefixes)
    for name, block in blocks.items():
        if not block.lines:
            block.lines.append(NumberLine(name, '', empty_reason(name, row) if empty_reason else NOT_COMPUTED, mark=MARK_NONE))
    return blocks


# --------------------------------------------------------------------------------------------- layout


@dataclass(frozen=True)
class Cell:
    mark: str
    label: str
    value: str  # value and unit
    missing: bool


@dataclass(frozen=True)
class Row:
    """A printed row: 'title', 'group' (a bold group header), 'kv' (one cell), 'pair' (two cells),
    'text' (free text at `indent`) or 'gap'."""

    kind: str
    cells: Tuple[Cell, ...] = ()
    text: str = ''
    indent: int = 0
    mark: str = ''
    missing: bool = False
    height: float = 1.0  # in lines
    block: str = ''  # title of the block the row belongs to (for '(cont.)' headers)
    group: str = ''  # the group header the row is printed under (repeated at the top of a new column)


@dataclass
class Placement:
    font_pt: float
    chars: int  # characters per column line
    capacity: float  # lines per column
    columns: List[List[Row]]
    overflow: float  # lines that did not fit (only at the smallest font)


def _wrap(text: str, width: int) -> List[str]:
    return textwrap.wrap(text, width=max(width, 8), break_long_words=True, break_on_hyphens=False) or ['']


def _cell(line: NumberLine) -> Cell:
    value = f'{line.value} {line.unit}' if line.unit else line.value
    return Cell(line.mark, line.label, value, line.value == MISSING)


def _pair_value_offset(cell: Cell) -> int:
    return 2 + max(len(cell.label) + 1, PAIR_KEY_WIDTH)


def _short(cell: Cell, half: int) -> bool:
    """A cell goes two per row when its label fits the key width and its value the half line."""
    return len(cell.label) <= PAIR_KEY_WIDTH - 1 and _pair_value_offset(cell) + len(cell.value) <= half - 1


def _kv_rows(cell: Cell, chars: int) -> List[Row]:
    if len(cell.label) <= KEY_WIDTH - 1:
        start = 2 + KEY_WIDTH
    else:
        start = 2 + len(cell.label) + 1
    if chars - start < 10:
        rows = [Row('kv', (Cell(cell.mark, cell.label, '', cell.missing),))]
        start = 2 + KEY_WIDTH
        return rows + [Row('text', text=piece, indent=start, missing=cell.missing) for piece in _wrap(cell.value, chars - start)]
    pieces = _wrap(cell.value, chars - start)
    rows = [Row('kv', (Cell(cell.mark, cell.label, pieces[0], cell.missing),))]
    return rows + [Row('text', text=piece, indent=start, missing=cell.missing) for piece in pieces[1:]]


def _text_rows(line: NumberLine, chars: int) -> List[Row]:
    if len(line.value) <= chars - 2:  # a table row keeps its spaces
        return [Row('text', text=line.value, indent=2, mark=line.mark, missing=line.mark == MARK_NONE and not line.column)]
    pieces = _wrap(line.value, chars - 2)
    rows = [Row('text', text=pieces[0], indent=2, mark=line.mark, missing=line.mark == MARK_NONE and not line.column)]
    rest = ' '.join(pieces[1:])
    return rows + [Row('text', text=piece, indent=4) for piece in (_wrap(rest, chars - 4) if rest else [])]


def block_rows(block: TextBlock, chars: int) -> List[Row]:
    """The rows of one block at `chars` characters per line: title, group headers, values (short ones two
    per row), gap."""
    rows = [Row('title', text=block.title)]
    half = chars // 2
    pending: Optional[Cell] = None
    group = ''

    def flush():
        nonlocal pending
        if pending:
            rows.extend(_kv_rows(pending, chars))
            pending = None

    for line in block.lines:
        if line.group and line.group != group:
            flush()
            group = line.group
            rows.append(Row('group', text=group))
        if not line.label:
            flush()
            rows.extend(_text_rows(line, chars))
            continue
        cell = _cell(line)
        if not _short(cell, half):
            flush()
            rows.extend(_kv_rows(cell, chars))
        elif pending:
            rows.append(Row('pair', (pending, cell)))
            pending = None
        else:
            pending = cell
    flush()
    rows.append(Row('gap', height=BLOCK_GAP))
    tagged, group = [], ''
    for row in rows:
        group = row.text if row.kind == 'group' else ('' if row.kind == 'title' else group)
        tagged.append(Row(**{**row.__dict__, 'block': block.title, 'group': group}))
    return tagged


def _geometry(layout: VerdictFigureLayout, font_pt: float) -> Tuple[int, float]:
    width = layout.text_columns[0][2]
    height = layout.text_columns[0][3]
    chars = int(width * 72 / (MONO_CHAR_WIDTH * font_pt))
    capacity = height * 72 / (font_pt * layout.text_line_height)
    return chars, capacity


def _fits_header(queue: List[Row], used: float, capacity: float) -> bool:
    """A title or a group header never ends a column: it moves on with its first row."""
    if queue[0].kind not in ('title', 'group') or len(queue) < 2:
        return True
    need = queue[0].height + queue[1].height
    if queue[1].kind == 'group' and len(queue) > 2:
        need += queue[2].height
    return used + need <= capacity + 1e-9


def _fill(rows: List[Row], n_columns: int, capacity: float) -> Tuple[List[List[Row]], List[Row]]:
    """Fill the columns from one stream; a column that starts inside a block repeats its title + ' (cont.)'."""
    queue = list(rows)
    columns = []
    for _ in range(n_columns):
        while queue and queue[0].kind == 'gap':
            queue.pop(0)
        column, used = [], 0.0
        if queue and queue[0].kind != 'title':
            first = queue[0]
            heads = [Row('title', text=first.block + CONTINUED, block=first.block)]
            if first.group and first.kind != 'group':
                heads.append(Row('group', text=first.group, block=first.block, group=first.group))
            column += heads
            used += sum(head.height for head in heads)
        while queue and used + queue[0].height <= capacity + 1e-9 and _fits_header(queue, used, capacity):
            row = queue.pop(0)
            column.append(row)
            used += row.height
        columns.append(column)
    return columns, queue


def plan_columns(blocks: Dict[str, TextBlock], layout: VerdictFigureLayout, order: Optional[Sequence[str]] = None) -> Placement:
    """The rows of every column at the largest font (from `text_font_pt` down) that holds everything; blocks in
    `order` (default: the order of `blocks`)."""
    order = list(blocks) if order is None else order
    font = layout.text_font_pt
    n_columns = len(layout.text_columns)
    while True:
        chars, capacity = _geometry(layout, font)
        stream = [row for name in order if name in blocks for row in block_rows(blocks[name], chars)]
        columns, rest = _fill(stream, n_columns, capacity)
        overflow = sum(row.height for row in rest if row.kind != 'gap')
        if not overflow or font - FONT_STEP_PT < layout.text_min_font_pt - 1e-9:
            if overflow:  # nothing is cut: the remainder goes below the last column
                columns[-1] += rest
            return Placement(font, chars, capacity, columns, overflow)
        font -= FONT_STEP_PT


def place_blocks(blocks: Dict[str, TextBlock], layout: VerdictFigureLayout, order: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """Put the blocks into the columns in order; overflow moves to the next column, never truncates.
    Returns the number of lines per column (`column_1`...), the `capacity` of a column in lines and the
    `overflow` (lines below the last column, 0 unless even the smallest font cannot hold the text)."""
    return _stats(plan_columns(blocks, layout, order))


def _stats(placement: Placement) -> Dict[str, int]:
    stats = {f'column_{i + 1}': int(sum(row.height for row in column) + 1e-9) for i, column in enumerate(placement.columns)}
    stats['capacity'] = int(placement.capacity)
    stats['overflow'] = int(math.ceil(placement.overflow))
    stats['font_pt'] = placement.font_pt
    return stats


# --------------------------------------------------------------------------------------------- drawing


class _Writer:
    """Writes monospaced rows into one column rectangle of the figure (inches -> figure fraction)."""

    def __init__(self, fig, rect, placement: Placement, layout: VerdictFigureLayout):
        self.fig = fig
        self.x, y, self.width, height = rect
        self.top = y + height
        self.font = placement.font_pt
        self.char = MONO_CHAR_WIDTH * self.font / 72
        self.line = self.font * layout.text_line_height / 72
        self.size = fig.get_size_inches()

    def text(self, col: float, y: float, text: str, **kwargs):
        kwargs.setdefault('color', '#111111')
        self.fig.text(
            (self.x + col * self.char) / self.size[0],
            y / self.size[1],
            text,
            family=MONO_FONT,
            fontsize=kwargs.pop('fontsize', self.font),
            va='top',
            ha='left',
            **kwargs,
        )

    def mark(self, col: float, y: float, mark: str):
        if mark:
            self.text(col, y, mark, color=MARK_COLOURS.get(mark, TEXT_MISSING), weight='bold')

    def title(self, y: float, text: str):
        from matplotlib.patches import Rectangle

        box = Rectangle(
            (self.x / self.size[0], (y - self.line * 1.1) / self.size[1]),
            self.width / self.size[0],
            self.line * 1.15 / self.size[1],
            transform=self.fig.transFigure,
            color=TEXT_HEADER_FILL,
            lw=0,
            zorder=0,
        )
        self.fig.add_artist(box)
        self.text(0.3, y, text, weight='bold', color='#1a1a1a')

    def cell(self, col: float, y: float, cell: Cell, value_col: Optional[float] = None):
        self.mark(col, y, cell.mark)
        self.text(col + 2, y, cell.label, color=TEXT_KEY)
        if cell.value:
            start = value_col if value_col is not None else col + _pair_value_offset(cell)
            self.text(start, y, cell.value, color=TEXT_MISSING if cell.missing else '#111111')

    def row(self, y: float, row: Row, chars: int):
        if row.kind == 'title':
            self.title(y, row.text)
        elif row.kind == 'group':
            self.text(1, y, row.text, weight='bold', color=TEXT_GROUP)
        elif row.kind == 'kv':
            cell = row.cells[0]
            value_col = 2 + KEY_WIDTH if len(cell.label) <= KEY_WIDTH - 1 else None
            self.cell(0, y, cell, value_col)
        elif row.kind == 'pair':
            self.cell(0, y, row.cells[0])
            self.cell(chars // 2, y, row.cells[1])
        elif row.kind == 'text':
            self.mark(0, y, row.mark)
            self.text(row.indent, y, row.text, color=TEXT_MISSING if row.missing else '#111111')


def draw_text(fig, blocks: Dict[str, TextBlock], layout: VerdictFigureLayout, order: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """The text columns; returns `place_blocks` statistics."""
    placement = plan_columns(blocks, layout, order)
    for rect, rows in zip(layout.text_columns, placement.columns):
        writer = _Writer(fig, rect, placement, layout)
        y = writer.top
        for row in rows:
            writer.row(y, row, placement.chars)
            y -= row.height * writer.line
    return _stats(placement)


def printed_lines(blocks: Dict[str, TextBlock], order: Optional[Sequence[str]] = None) -> Iterable[NumberLine]:
    """Every line of every block, in print order."""
    for name in list(blocks) if order is None else order:
        if name in blocks:
            yield from blocks[name].lines


def printed_columns(blocks: Dict[str, TextBlock], order: Optional[Sequence[str]] = None) -> Dict[str, NumberLine]:
    """Column -> the line that prints its value (a dropped duplicate maps to the line printed for it)."""
    printed = {}
    for line in printed_lines(blocks, order):
        if line.column and line.label:
            for column in (line.column,) + line.also:
                printed.setdefault(column, line)
    return printed


# --------------------------------------------------------------------------------------------- explained column (A′)

STYLE_BOLD, STYLE_GREY = 'bold', 'grey'
SANS_FONT = 'DejaVu Sans'
# Sizes (pt): the key information (plaque, explanation, steps and checks) larger than the numbers (author's N4)
BADGE_PT = 16.0
BADGE_PAD = 0.28  # the plaque's box padding, in units of its font size
ENDS_PT = 12.0  # the tag 'ends: <final regime>' beside the plaque
ENDS_GAP = 0.12  # inches between the plaque's box and the tag's box
SUBTITLE_PT = 9.0
SENTENCE_PT = 10.5
SECTION_PT = 10.0
ITEM_PT = 9.2
DETAIL_PT = 8.0
ARITHMETIC_PT = 9.0
NOTE_PT = 7.6
NUMBERS_MAX_PT = 8.0  # a short list of numbers grows up to this (still below the checks' ITEM_PT)
NUMBERS_MIN_PT = 6.2  # ~10 px at the layout's 120 dpi: the smallest legible at 100 % zoom; the rest is counted
LINE_SPACING = 1.28  # in units of the font size
COLUMN_GAP = 0.3  # inches between the two check columns and between number columns
NUMBER_COLUMNS = 4
KEY_CHARS_MAX = 18  # a longer label pushes its value right
GROUP_HEAD_LINES = 2  # lines kept in one column with their group header
GROUP_EXTRA_PT = 0.3  # a group header is drawn this much larger than the numbers (wrapped to its own width)
STEPS_SHARE = 0.38  # of the width: 'How it was decided'; the rest is 'Confidence' (the longer list)
ICON_INDENT = 0.24  # inches from an icon to its text
INK, KEY_INK, GREY_INK, LIGHT_INK = '#111111', '#3a3a3a', '#8c8c8c', '#c4c4c4'
STATE_COLOURS = {
    'pass': '#1b7837',
    'thin': '#b8860b',  # amber warning: passed, near its threshold (dark enough for 3:1 on white)
    'fail': '#b2182b',
    'skipped': '#8c8c8c',
    'missing': '#7b3294',  # should have been measured, was not: the purple of 'uncertain' (a missing measurement)
    'not_applicable': '#bbbbbb',
}
# One line, one glyph per meaning (the step icons differ from the check icons)
LEGEND_TEXT = (
    '✓ full   ⚠ thin   ✗ failed   – skipped by design   ? not measured   · n/a   ▼ cap        '
    '● decided here   ↓ passed on   ◌ not run   pale: not reached'
)


class _Pen:
    """Proportional text in inches on a figure: measured widths, greedy word wrap, top-down drawing."""

    def __init__(self, fig):
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextToPath

        self.fig = fig
        self.width, self.height = fig.get_size_inches()
        self._measure = TextToPath()
        self._props = {}
        self._font = FontProperties

    def text_width(self, text: str, size: float, family: str = SANS_FONT, weight: str = 'normal') -> float:
        key = (family, weight)
        if key not in self._props:
            self._props[key] = self._font(family=family, weight=weight, size=10)
        width = self._measure.get_text_width_height_descent(text, self._props[key], ismath=False)[0]
        return width / 10 * size / 72

    def wrap(self, text: str, width: float, size: float, family: str = SANS_FONT, weight: str = 'normal') -> List[str]:
        lines, current = [], ''
        for word in text.split(' '):
            candidate = f'{current} {word}' if current else word
            if current and self.text_width(candidate, size, family, weight) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        return lines + [current] if current or not lines else lines

    def text(self, x: float, y: float, text: str, size: float, **kwargs):
        kwargs.setdefault('color', INK)
        kwargs.setdefault('family', SANS_FONT)
        return self.fig.text(x / self.width, y / self.height, text, fontsize=size, va='top', ha='left', **kwargs)

    def paragraph(self, x: float, y: float, text: str, size: float, width: float, draw: bool = True, **kwargs) -> float:
        """Wrapped text from y down; returns the y below it."""
        for piece in self.wrap(text, width, size, kwargs.get('family', SANS_FONT), kwargs.get('weight', 'normal')):
            if draw:
                self.text(x, y, piece, size, **kwargs)
            y -= size * LINE_SPACING / 72
        return y

    def rule(self, y: float, x0: float, x1: float):
        import matplotlib.pyplot as plt

        self.fig.add_artist(plt.Line2D([x0 / self.width, x1 / self.width], [y / self.height] * 2, color='#dddddd', lw=0.8))


def _box(layout: VerdictFigureLayout) -> Tuple[float, float, float, float]:
    """(x0, x1, top, bottom) of the right column, in inches: the span of the layout's text columns."""
    first, last = layout.text_columns[0], layout.text_columns[-1]
    return first[0], last[0] + last[2], first[1] + first[3], first[1]


def _steps_column(pen: _Pen, text, x: float, y: float, width: float, status_colour: str, draw: bool) -> float:
    if draw:
        pen.text(x, y, 'How it was decided', SECTION_PT + 0.5, weight='bold', color=KEY_INK)
    y -= SECTION_PT * 1.6 / 72
    for step in text.steps:
        state = str(step.state)
        colour = {'leaf': status_colour, 'not_reached': LIGHT_INK, 'not_run': GREY_INK}.get(state, INK)
        if draw:
            from .verdict_text import STEP_ICONS

            if STEP_ICONS[step.state]:
                pen.text(x, y + 0.01, STEP_ICONS[step.state], ITEM_PT + 2, color=colour, weight='bold')
            pen.text(x + ICON_INDENT, y, step.title, ITEM_PT, weight='bold', color=colour)
        y -= ITEM_PT * LINE_SPACING / 72
        if step.outcome:
            ink = GREY_INK if state in ('not_run', 'not_reached') else KEY_INK
            y = pen.paragraph(x + ICON_INDENT, y, step.outcome, DETAIL_PT, width - ICON_INDENT, draw, color=ink)
        y -= 0.05
    return y


def _check_detail(row, compact: bool = False) -> str:
    """'detail · need … — note' (compact: without the need); a cap: 'cap ≤ medium · its numbers', and below it the
    other verdict and what on the figure decides it (`verdict_text.alternative_text`)."""
    from .verdict_text import alternative_text

    if row.is_cap:
        head = f'{row.counts}  ·  {row.detail}' if row.detail and not compact else row.counts
        alternative = alternative_text(row)
        return f'{head}  —  {alternative}' if alternative else head
    detail = row.detail
    if row.need and str(row.state) != 'skipped' and not compact:
        detail = f'{detail}  ·  need {row.need}' if detail else f'need {row.need}'
    if row.note and row.note not in detail:
        detail = f'{detail}  —  {row.note}' if detail else row.note
    return detail


def _passed_line(pen: _Pen, passed, x: float, y: float, width: float, draw: bool) -> float:
    """The full margins of a compact column as one line of names."""
    if not passed:
        return y
    if draw:
        pen.text(x, y + 0.01, '✓', ITEM_PT + 1.5, color=STATE_COLOURS['pass'], weight='bold')
    names = '; '.join(row.name for row in passed)
    return pen.paragraph(x + ICON_INDENT, y, f'full: {names}', DETAIL_PT, width - ICON_INDENT, draw, color=KEY_INK) - 0.04


def _checks_column(pen: _Pen, text, x: float, y: float, width: float, draw: bool, compact: bool = False) -> float:
    """One row per margin (icon, name, detail) and cap. `compact` (when the numbers would not fit otherwise): the full
    margins in one line of names, the caps without their detail line but with their short reason (`CheckRow.short`)."""
    if draw:
        pen.text(x, y, 'Confidence: every margin and cap the level read', SECTION_PT + 0.5, weight='bold', color=KEY_INK)
    y -= SECTION_PT * 1.6 / 72
    if not text.checks:
        return pen.paragraph(x, y, 'uncertain and invalid verdicts carry no level', ITEM_PT, width, draw, color=GREY_INK)
    rows = text.checks
    if compact:
        passed = [row for row in rows if str(row.state) == 'pass' and not row.is_cap]
        rows = [row for row in rows if row not in passed]
        y = _passed_line(pen, passed, x, y, width, draw)
    for row in rows:
        if compact and row.is_cap:
            if draw:
                pen.text(x, y + 0.01, row.icon, ITEM_PT + 1.5, color=STATE_COLOURS['fail'], weight='bold')
            reason = f': {row.short}' if row.short else ''  # what holds the cap (W20 round E)
            name = f'{row.name}{reason} ({row.counts})'
            y = pen.paragraph(x + ICON_INDENT, y, name, ITEM_PT, width - ICON_INDENT, draw, weight='bold')
            from .verdict_text import alternative_text

            if alternative_text(row):  # the author's q1: a cap always names the other verdict, compact or not
                y = pen.paragraph(x + ICON_INDENT, y, alternative_text(row), DETAIL_PT, width - ICON_INDENT, draw, color=KEY_INK)
            y -= 0.03
            continue
        state = str(row.state)
        colour = STATE_COLOURS[state]
        quiet = state in ('skipped', 'not_applicable')
        weight = 'bold' if state in ('fail', 'thin', 'missing') else 'normal'
        if draw:
            pen.text(x, y + 0.01, row.icon, ITEM_PT + 1.5, color=colour, weight='bold')
        y = pen.paragraph(x + ICON_INDENT, y, row.name, ITEM_PT, width - ICON_INDENT, draw, color=GREY_INK if quiet else INK, weight=weight)
        detail = _check_detail(row, compact)
        if detail:
            y = pen.paragraph(x + ICON_INDENT, y, detail, DETAIL_PT, width - ICON_INDENT, draw, color=GREY_INK if quiet else KEY_INK)
        y -= 0.04
    return y


def _plaque(pen: _Pen, text, x: float, y: float, colour: str) -> None:
    """The plaque (status · angle_status · confidence, white on the status colour) and, when the record ends in another
    regime, the tag 'ends: …' right of it: outlined in the status colour, lighter, so it reads as a note on the verdict,
    not as a fourth part of it."""
    plaque = f' {text.badge} '
    pen.text(x, y, plaque, BADGE_PT, weight='bold', color='white', bbox={'boxstyle': f'round,pad={BADGE_PAD}', 'fc': colour, 'ec': 'none'})
    if not text.ends:
        return
    pad_in = BADGE_PAD * BADGE_PT / 72
    right = x + pen.text_width(plaque, BADGE_PT, weight='bold') + pad_in
    tag_x = right + ENDS_GAP + BADGE_PAD * ENDS_PT / 72
    tag_y = y - (BADGE_PT - ENDS_PT) / 2 / 72  # the two boxes share their middle line
    box = {'boxstyle': f'round,pad={BADGE_PAD}', 'fc': 'white', 'ec': colour, 'lw': 1.2}
    pen.text(tag_x, tag_y, f' {text.ends} ', ENDS_PT, weight='bold', color=colour, bbox=box)


PHYSICS_ICON_COLOURS = {'✓': 'pass', '✗': 'fail', '?': 'missing', '–': 'skipped'}
PHYSICS_LABEL_GAP = 0.08  # inches between the widest label of a section and the texts
PHYSICS_GAP = 0.06  # inches above the physics block
LEGEND_GAP = 0.06  # inches above the legend when it goes under the steps


def _physics_section(pen: _Pen, section, x: float, y: float, width: float, draw: bool, compact: bool = False) -> float:
    """One physics section (R1 §4): its title (bold) and subtitle, then per row the icon, the label in a column as wide
    as the section's widest and the text wrapped right of it; grey rows (context) grey, rows whose number the verdict
    compared bold. `compact` (the numbers would not fit otherwise): the grey context rows are left out (they are in the
    CSV files). Returns the y below it."""
    if draw:
        pen.text(x, y, section.title, SECTION_PT, weight='bold', color=KEY_INK)
    below = y - SECTION_PT * LINE_SPACING / 72
    if section.subtitle:
        offset = pen.text_width(f'{section.title}   ', SECTION_PT, weight='bold')
        if offset < width / 2:  # beside the title, else under it
            below = min(below, pen.paragraph(x + offset, y, section.subtitle, DETAIL_PT, width - offset, draw, color=KEY_INK))
        else:
            below = pen.paragraph(x, below, section.subtitle, DETAIL_PT, width, draw, color=KEY_INK)
    y = below
    rows = [row for row in section.rows if not (compact and row.grey)]
    labels = [row.label for row in rows] or ['']
    label_width = max(pen.text_width(label, DETAIL_PT, weight='bold') for label in labels) + PHYSICS_LABEL_GAP
    text_x = x + ICON_INDENT + label_width
    for row in rows:
        ink = GREY_INK if row.grey else INK
        weight = 'bold' if row.read_by_verdict else 'normal'
        if draw:
            if row.icon:
                pen.text(
                    x,
                    y + 0.01,
                    row.icon,
                    DETAIL_PT + 1.5,
                    color=STATE_COLOURS[PHYSICS_ICON_COLOURS.get(row.icon, 'skipped')],
                    weight='bold',
                )
            pen.text(x + ICON_INDENT, y, row.label, DETAIL_PT, color=GREY_INK if row.grey else KEY_INK, weight='bold')
        y = pen.paragraph(text_x, y, row.text, DETAIL_PT, x + width - text_x, draw, color=ink, weight=weight)
    return y - 0.04


def _legend_under_steps(pen: _Pen, x: float, left: float, right: float, width: float, draw: bool) -> Optional[float]:
    """The legend under the steps column (from `left`, `width` wide) when it ends above the checks' end `right`: the y
    below it; None (nothing drawn) when it would reach below the checks."""
    if pen.paragraph(x, left - LEGEND_GAP, LEGEND_TEXT, NOTE_PT, width, False) < right:
        return None
    return pen.paragraph(x, left - LEGEND_GAP, LEGEND_TEXT, NOTE_PT, width, draw, color=GREY_INK)


def _explanation(pen: _Pen, text, layout: VerdictFigureLayout, draw: bool, compact: bool = False, legend_beside: bool = False) -> float:
    """Plaque, sentences, the two check columns, arithmetic, info and legend; returns the y below them. `legend_beside`
    (when the numbers would not fit otherwise): the legend goes under the steps, as wide as they are, where it ends above
    the checks (the room is there anyway), instead of a line over the whole width."""
    from .style import STATUS_COLOURS

    x0, x1, top, _ = _box(layout)
    width = x1 - x0
    colour = STATUS_COLOURS.get(text.status, '#4d4d4d')
    y = top
    if draw:
        _plaque(pen, text, x0 + 0.05, y - 0.02, colour)
    y -= BADGE_PT * 2.0 / 72
    y = pen.paragraph(x0, y, text.subtitle, SUBTITLE_PT, width, draw, color=KEY_INK) - 0.04
    for sentence in text.sentences:
        y = pen.paragraph(x0, y, sentence, SENTENCE_PT, width, draw) - 0.05
    if draw:
        pen.rule(y + 0.02, x0, x1)
    y -= 0.08
    left_width = width * STEPS_SHARE - COLUMN_GAP / 2
    left = _steps_column(pen, text, x0, y, left_width, colour, draw)
    right = _checks_column(pen, text, x0 + left_width + COLUMN_GAP, y, width - left_width - COLUMN_GAP, draw, compact)
    # a physics section goes under the steps while it fits beside the checks (the room is there anyway), the rest under all
    below = []
    for section in getattr(text, 'physics', ()):
        if not below and _physics_section(pen, section, x0, left - PHYSICS_GAP, left_width, False, compact) >= right:
            left = _physics_section(pen, section, x0, left - PHYSICS_GAP, left_width, draw, compact)
        else:
            below.append(section)
    beside = _legend_under_steps(pen, x0, left, right, left_width, draw) if legend_beside else None
    y = min(left if beside is None else beside, right) - 0.04
    y = pen.paragraph(x0, y, text.arithmetic, ARITHMETIC_PT, width, draw, weight='bold') - 0.02
    if text.info:
        y = pen.paragraph(x0, y, 'Also measured, not read by this verdict: ' + '; '.join(text.info), NOTE_PT, width, draw, color=GREY_INK)
    for section in below:
        y = _physics_section(pen, section, x0, y - PHYSICS_GAP, width, draw, compact)
    if beside is None:
        y = pen.paragraph(x0, y, LEGEND_TEXT, NOTE_PT, width, draw, color=GREY_INK)
    if draw:
        pen.rule(y - 0.02, x0, x1)
    return y - 0.1


def group_lines(
    row: Mapping[str, Any],
    groups: Sequence[Tuple[str, Sequence[str]]],
    prefixes: Sequence[str] = (),
    bold: Sequence[str] = (),
    grey: Optional[Mapping[str, str]] = None,
) -> List[NumberLine]:
    """The numbers of the explained column: per group (title, columns) the columns present in `row` with a value,
    labelled without their prefix and unit; a column the decision compared is bold, one of a test skipped by design
    grey ('not used'). A group with no value is left out."""
    grey = grey or {}
    lines = []
    for title, columns in groups:
        present = [column for column in columns if not missing(row.get(column))]
        labels = [label_of(column, prefixes) for column in present]
        for column, label in zip(present, labels):
            if labels.count(label) > 1:  # the same quantity of two prefixes (sub_best, own_best): keep the prefix
                label = label_of(column)
            style = STYLE_GREY if column in grey else (STYLE_BOLD if column in bold else '')
            value = format_cell(column, row.get(column)) + unit_space(column) + (' (not used)' if column in grey else '')
            lines.append(NumberLine('numbers', label, value, '', column, group=title, style=style))
    return lines


def _number_rows(lines: Sequence[NumberLine], width: float, size: float) -> List[Tuple[str, str, str]]:
    """(kind, text, style) rows at `size`: a group header before each group, labels padded to one column."""
    key = max([len(line.label) for line in lines if line.label] + [8])
    key = min(key, KEY_CHARS_MAX)
    chars = max(int(width * 72 / (MONO_CHAR_WIDTH * size)), 12)
    group_chars = max(int(width * 72 / (MONO_CHAR_WIDTH * (size + GROUP_EXTRA_PT))), 12)  # the header is drawn larger
    rows, group = [], None
    for line in lines:
        kept = line.keep and line.group == group  # a table row: one unbreakable item with the rows before it
        if line.group != group:
            if rows:
                rows.append(('gap', '', ''))
            group = line.group
            pieces = textwrap.wrap(group, group_chars, subsequent_indent='  ', break_long_words=True) or ['']
            rows += [('group' if index == 0 else 'group_cont', piece, '') for index, piece in enumerate(pieces)]
        if line.label:
            value = f'{line.value} {line.unit}' if line.unit else line.value
            text = f'{line.label:<{key}} {value}'
        else:
            text = line.value
        pieces = textwrap.wrap(text, chars, subsequent_indent='  ', break_long_words=True) or ['']
        rows += [('line' if index == 0 and not kept else 'cont', piece, line.style) for index, piece in enumerate(pieces)]
    return rows


def _items(rows) -> List[List[Tuple[str, str, str]]]:
    """The rows as unbreakable items: a line with its continuations; a group header with the first GROUP_HEAD_LINES
    lines under it (a header never ends a column, nor is it left with a single line)."""
    items: List[List[Tuple[str, str, str]]] = []
    for row in rows:
        head = items[-1] if items else []
        under_head = head and head[0][0] == 'group' and sum(1 for r in head if r[0] == 'line') < GROUP_HEAD_LINES
        if row[0] in ('cont', 'group_cont') or (under_head and row[0] == 'line'):
            head.append(row)
        else:
            items.append([row])
    return items


def _place_numbers(rows, n_columns: int, capacity: int) -> Tuple[List[List[Tuple[str, str, str]]], int]:
    """Fill the columns top-down with whole items (`_items`): a group header never ends a column and a wrapped line is
    never split between two. Returns the columns and the number of lines left over."""
    columns: List[List[Tuple[str, str, str]]] = [[] for _ in range(n_columns)]
    index = 0
    queue = _items(rows)
    while queue and index < n_columns:
        column = columns[index]
        while queue:
            item = queue[0]
            if item[0][0] == 'gap' and not column:
                queue.pop(0)
                continue
            if len(column) + len(item) > capacity and column:
                break
            column.extend(queue.pop(0))
        index += 1
    return columns, sum(1 for item in queue for row in item if row[0] in ('line', 'cont'))


def _fit_numbers(pen: _Pen, lines, y: float, bottom: float, width: float, n_columns: int, layout: VerdictFigureLayout):
    """(size, line step, rows, columns, lines left over) at the largest font from NUMBERS_MAX_PT down to NUMBERS_MIN_PT
    that holds the numbers between y and bottom, the columns balanced (the shortest column height that holds them).
    At the smallest font one row is kept free for the count of the lines left over."""
    size = NUMBERS_MAX_PT
    while True:
        step = size * layout.text_line_height / 72
        available = max(int((y - bottom) / step), 1)
        rows = _number_rows(lines, width, size)
        for capacity in range(max(-(-len(rows) // n_columns), 1), available + 1):
            columns, left = _place_numbers(rows, n_columns, capacity)
            if not left:
                return size, step, rows, columns, 0
        if size - FONT_STEP_PT < NUMBERS_MIN_PT - 1e-9:
            columns, left = _place_numbers(rows, n_columns, max(available - 1, 1))
            return size, step, rows, columns, left
        size -= FONT_STEP_PT


@dataclass(frozen=True)
class _Fit:
    """One way to lay out the explained column (`FIT_LEVELS`): the grey numbers kept, the legend under the steps, the
    number of number columns, the compact explanation."""

    grey: bool
    legend_beside: bool
    extra_column: bool
    compact: bool


# The layouts tried in order until the numbers fit (W23 F1: by layout, not by dropping verdict lines): everything; the
# grey numbers (of tests skipped by design) go; the legend under the steps; a narrow fifth number column; and only then
# the compact explanation (the full margins in one line, the caps without their detail, no grey physics rows).
FIT_LEVELS = (
    _Fit(grey=True, legend_beside=False, extra_column=False, compact=False),
    _Fit(grey=False, legend_beside=False, extra_column=False, compact=False),
    _Fit(grey=False, legend_beside=True, extra_column=False, compact=False),
    _Fit(grey=False, legend_beside=True, extra_column=True, compact=False),
    _Fit(grey=False, legend_beside=True, extra_column=True, compact=True),
)


def draw_explained(fig, text, lines: Sequence[NumberLine], layout: VerdictFigureLayout, n_columns: int = NUMBER_COLUMNS) -> Dict[str, Any]:
    """The explained right column (layout A′): `text` is a `verdict_text.VerdictText`, `lines` the numbers
    (`group_lines` plus free-text lines). The numbers fill `n_columns` monospaced columns below the explanation at the
    largest font from NUMBERS_MAX_PT down to NUMBERS_MIN_PT that holds them in balanced columns; when even the smallest
    does not, the layouts of `FIT_LEVELS` are tried in order: the grey numbers (of tests skipped by design) are left out,
    the legend goes under the steps, a narrow extra number column is added, and last the explanation turns compact (the
    full margins in one line). What still does not fit is counted in a last line (every number is in the CSV files).
    Returns the statistics (`compact`, `font_pt`, `lines`, `overflow`, `wide`: the rows wider than their column, which
    would run into the next one, `numbers_top_in`, `n_columns`, `grey_dropped`, `legend_beside`)."""
    pen = _Pen(fig)
    x0, x1, _, bottom = _box(layout)
    for level in FIT_LEVELS:
        columns_n = n_columns + (1 if level.extra_column else 0)
        width = (x1 - x0 - COLUMN_GAP * (columns_n - 1)) / columns_n
        shown = list(lines) if level.grey else [line for line in lines if line.style != STYLE_GREY]
        y = _explanation(pen, text, layout, draw=False, compact=level.compact, legend_beside=level.legend_beside)
        if not _fit_numbers(pen, shown, y, bottom, width, columns_n, layout)[-1]:
            break
    y = _explanation(pen, text, layout, draw=True, compact=level.compact, legend_beside=level.legend_beside)
    size, step, rows, columns, left = _fit_numbers(pen, shown, y, bottom, width, columns_n, layout)
    wide = []  # rows wider than their column: they would run into the next one
    for index, column in enumerate(columns):
        x = x0 + index * (width + COLUMN_GAP)
        for position, (kind, piece, style) in enumerate(column):
            yy = y - position * step
            if kind in ('group', 'group_cont'):
                font, weight, ink = size + GROUP_EXTRA_PT, 'bold', TEXT_GROUP
            elif kind in ('line', 'cont'):
                font, weight, ink = size, 'bold' if style == STYLE_BOLD else 'normal', GREY_INK if style == STYLE_GREY else INK
            else:
                continue
            pen.text(x, yy, piece, font, family=MONO_FONT, weight=weight, color=ink)
            if index < columns_n - 1 and pen.text_width(piece, font, MONO_FONT, weight) > width + COLUMN_GAP:
                wide.append(piece)
    if left:
        pen.text(
            x0,
            bottom + step,
            f'… {left} more lines: every number is in summary.csv and the numbers CSV',
            size,
            family=MONO_FONT,
            color='#b2182b',
        )
    n_lines = sum(1 for row in rows if row[0] in ('line', 'cont'))
    return {
        'compact': level.compact,
        'font_pt': size,
        'lines': n_lines,
        'overflow': left,
        'wide': wide,
        'numbers_top_in': y,
        'n_columns': columns_n,
        'grey_dropped': not level.grey,
        'legend_beside': level.legend_beside,
    }
