"""Drawing of the right column of the verdict figures (`verdict_column.Column`), matplotlib only.

The column's blocks (`verdict_column.BLOCKS`) are laid out once for a width into units of drawing primitives (texts at
baselines, rounded boxes, rules) with heights measured by the Agg renderer at the figure's dpi; the same units are
measured and drawn, so what is measured is what is drawn. The right area of the figure (`VerdictFigureLayout`, from the
first text column to the last one) is cut into two sub-columns with one place for every block on every figure (MMR,
1:1, secular, ZLK): header, subtitle, key facts, Why, Confidence and Checks down the first, Evidence, Caveats, Numbers
and the footer down the second (`SECOND_SUBCOLUMN`). Neither part flows into the other's sub-column.

When the column does not fit (decision D8), the figure grows downward (`fit_layout` → `VerdictFigureLayout.grown`,
at most `COLUMN_MAX_GROWTH_IN`); beyond the cap it trims, in this order: Numbers groups from the last (DECISIVE and
ORBIT stay), the ✓ sub-checks collapse into one line per step, Evidence turns one line per row. Header, Subtitle, Key
facts, Why, Confidence, the step lines and the ⚠/✗/? sub-checks are never dropped; the text export
(`verdict_column.column_text`) is always complete.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from resonances.plotting import verdict_column as vc
from resonances.plotting.style import (
    CHIP_COLOURS,
    COLUMN_ICON_COLOURS,
    COLUMN_MAX_GROWTH_IN,
    LAYOUT_V1,
    STATUS_COLOURS,
    VerdictFigureLayout,
)

SANS = 'DejaVu Sans'
MONO = 'DejaVu Sans Mono'
GID = 'verdict_column'  # every artist drawn here carries this gid (tests and the figure builders find them by it)

# -- sizes (pt)
PLAQUE_PT = 16.0
LEVEL_PT = 11.0
CHIP_PT = 9.0
SUBTITLE_PT = 11.0
SUBTITLE_DETAIL_PT = 8.5
TITLE_PT = 10.5
BODY_PT = 9.5
BODY_NUMBERS_PT = 8.0
FACT_PT = 8.5
FACT_NUMBERS_PT = 7.5
PILL_PT = 8.0
CHECK_PT = 8.5
CHECK_NUMBERS_PT = 7.5
TAG_PT = 6.5
EVIDENCE_PT = 8.5
CAVEAT_PT = 8.5
NUMBERS_PT = 7.0
GROUP_TITLE_PT = 7.5
TABLE_MIN_PT = 5.5
FOOTER_PT = 6.5
LEGEND_PT = 7.5

# -- colours (the status, icon and chip colours: `style`, decision D4)
INK = '#1D1F23'
MUTED = '#5A6170'  # grey numbers, labels, the subtitle's second line
FAINT = '#8A9099'  # quiet steps, the footer
RULE = '#E1E4E8'
FACTS_FILL = '#F6F7F9'
CAVEAT_FILL = '#FFF4DE'
PILL_EDGE = '#CFD3D9'
PILL_STYLE = {  # icon -> (fill, edge)
    vc.OK: ('#FFFFFF', PILL_EDGE),
    vc.THIN: ('#FFF4DE', '#E2C27A'),
    vc.FAIL: ('#FCEBEA', '#E3A9A4'),
    vc.MISSING: ('#FFFFFF', PILL_EDGE),
    vc.QUIET: ('#FFFFFF', RULE),  # a limited input that does not hold the level: grey icon, muted words
}
DOT_ON, DOT_OFF = '#2A2E35', '#D5D8DD'
DECIDES_TINT = 0.1  # the ◆ row: the status colour at this share over white
FALLBACK_STATUS = '#4A4F57'

# -- geometry (inches)
SUBCOLUMN_GAP = 0.3  # between the two sub-columns of the right area
N_SUBCOLUMNS = 2
LEADING = 1.3  # line pitch in units of the font size
ASCENT = 0.95  # baseline below the line top, in units of the font size
BLOCK_GAP = 0.2  # between blocks, a thin rule in its middle
SMALL_GAP = 0.08  # between the header, subtitle and key facts; between number groups
STEP_GAP = 0.05
TINT_OVERLAP = 0.015  # a split ◆ step: each later piece's tint reaches this far up over the one before
GRID_GAP = 0.25  # between number groups side by side
GRID_MIN_WIDTH = 2.4  # a number group is at least this wide (2 groups side by side in a 5.5-in sub-column)
GROWTH_STEP_IN = 0.1  # the growth is rounded up to this
CHECK_ICON_X, CHECK_NUMBER_X, CHECK_TEXT_X = 0.06, 0.24, 0.42
SUB_ICON_X, SUB_TEXT_X = 0.46, 0.64
RULED = ('why', 'confidence', 'checks', 'evidence', 'numbers', 'footer')  # blocks with a rule above
SPLITTABLE = ('checks', 'evidence', 'numbers')
# the second sub-column always starts with the first of these that is shown (the author, 2026-11-05: Evidence at the top
# right on every figure, so that figures read in batches): header to Checks fill the first, Evidence to the footer the second
SECOND_SUBCOLUMN = ('evidence', 'caveats', 'numbers', 'footer')
KEEP_GROUPS = ('ORBIT',)  # Numbers groups never trimmed (with DECISIVE)
NBSP = '\u00a0'
COLLAPSED = 'all {n} sub-checks clean'
COLLAPSED_SOME = '{n} other sub-checks clean'
FOOTER_GLUE_CHARS = 40  # a footer item up to this long is kept on one line


def tint(colour: str, share: float = DECIDES_TINT) -> str:
    """`color` at `share` over white."""
    from matplotlib.colors import to_rgb

    rgb = [share * c + (1 - share) for c in to_rgb(colour)]
    return '#' + ''.join(f'{round(255 * c):02X}' for c in rgb)


def status_colour(status: str) -> str:
    return STATUS_COLOURS.get(status, FALLBACK_STATUS)


def icon_colour(icon: str, status: str) -> str:
    return status_colour(status) if icon == vc.DECIDES else COLUMN_ICON_COLOURS.get(icon, FAINT)


# -- measuring


@dataclass(frozen=True)
class Font:
    family: str = SANS
    weight: str = 'normal'
    size: float = BODY_PT
    colour: str = INK


class Measure:
    """Text widths in inches from the Agg renderer at `dpi` (the dpi the figure is drawn at, so the hinting matches)."""

    def __init__(self, dpi: float = LAYOUT_V1.dpi):
        self.dpi = dpi

    def width(self, text: str, font: Font) -> float:
        return _text_width(self.dpi, text, font.family, font.weight, font.size) if text else 0.0


@lru_cache(maxsize=65536)
def _text_width(dpi: float, text: str, family: str, weight: str, size: float) -> float:
    """One width for the process: the plan and the drawing of a figure, and every figure of a batch, share it."""
    renderer, prop = _renderer(dpi), _prop(family, weight, size)
    return renderer.get_text_width_height_descent(text, prop, ismath=False)[0] / dpi


@lru_cache(maxsize=8)
def _renderer(dpi: float):
    from matplotlib.backends.backend_agg import RendererAgg

    return RendererAgg(10, 10, dpi)


@lru_cache(maxsize=64)
def _prop(family: str, weight: str, size: float):
    from matplotlib.font_manager import FontProperties

    return FontProperties(family=family, weight=weight, size=size)


@lru_cache(maxsize=8)
def _charmap(family: str, weight: str) -> frozenset:
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    path = font_manager.findfont(font_manager.FontProperties(family=family, weight=weight), fallback_to_default=True)
    return frozenset(FT2Font(path).get_charmap())


def font_runs(text: str, font: Font) -> List[Tuple[str, Font]]:
    """`text` cut where a character is missing from `font` (it goes to DejaVu Sans; '?' when that lacks it too). A
    combining mark (σ̇) goes to DejaVu Sans with its base: the mono font gives the mark a cell of its own ('σ˙')."""
    own, sans = _charmap(font.family, font.weight), _charmap(SANS, font.weight)
    out: List[Tuple[str, Font]] = []
    for char in text:
        if font.family != SANS and out and unicodedata.combining(char) and ord(char) in sans:
            last, last_font = out.pop()
            if last_font.family == SANS:
                out.append((last + char, last_font))
                continue
            if last[:-1]:
                out.append((last[:-1], last_font))
            target, piece = replace(last_font, family=SANS), last[-1] + char
            if out and out[-1][1] == target:
                piece = out.pop()[0] + piece
            out.append((piece, target))
            continue
        if ord(char) in own or char.isspace():
            target, piece = font, char
        elif ord(char) in sans:
            target, piece = replace(font, family=SANS), char
        else:
            target, piece = font, '?'
        if out and out[-1][1] == target:
            out[-1] = (out[-1][0] + piece, target)
        else:
            out.append((piece, target))
    return out


# -- drawing primitives (inches; x from the sub-column's left edge, y downward from the unit's top)


@dataclass(frozen=True)
class Text:
    x: float
    y: float  # baseline
    text: str
    font: Font


@dataclass(frozen=True)
class Box:
    x: float
    y: float  # top
    w: float
    h: float
    fill: str = 'none'
    edge: str = 'none'
    lw: float = 0.0
    radius: float = 0.05
    dashed: bool = False


@dataclass
class Unit:
    """An unsplittable piece of a block: its primitives and height; `lead` = the gap above it when it does not start a
    sub-column (a rule in the gap's middle when `ruled`)."""

    block: str
    height: float
    items: List[Any]
    lead: float = 0.0
    ruled: bool = False


@dataclass(frozen=True)
class Line:
    segments: Tuple[Tuple[float, str, Font], ...]  # (x, text, font) from the line's left edge
    width: float


# -- wrapping text with inline markup


def _resolve(style: str, base: Font, numbers_pt: Optional[float], status: str) -> Font:
    if style == vc.STYLE_BOLD:
        return replace(base, weight='bold')
    if style == vc.STYLE_STATUS:
        return replace(base, weight='bold', colour=status_colour(status))
    if style == vc.STYLE_NUMBERS:
        return Font(MONO, 'normal', numbers_pt or base.size, MUTED)
    if style == vc.STYLE_PANELS:
        return replace(base, weight='normal', colour=MUTED)
    return base


Run = Tuple[str, Font]
_SPACES = re.compile(r'( +)')  # only ASCII spaces break a line (U+00A0 glues)


def runs(text: str, base: Font, status: str = '', numbers_pt: Optional[float] = None, plain_as: str = '') -> List[Run]:
    """The markup of `text` as runs of (text, font); `plain_as`='bold' draws the unmarked pieces bold."""
    out: List[Run] = []
    for piece, style in vc.spans(text):
        if style == vc.STYLE_PLAIN and plain_as:
            style = plain_as
        out.extend(font_runs(piece, _resolve(style, base, numbers_pt, status)))
    return out


def _words(items: Sequence[Run]) -> List[Tuple[List[Run], List[Run]]]:
    """The runs as words: (the spaces before it, its pieces); a word may span fonts (no break between '**65%**' and
    the text glued to it), and only ASCII spaces break (U+00A0 glues)."""
    out: List[Tuple[List[Run], List[Run]]] = []
    spaces: List[Run] = []
    word: List[Run] = []
    for text, font in items:
        for token in _SPACES.split(text.replace('\n', ' ')):
            if not token:
                continue
            if token.startswith(' '):
                if word:
                    out.append((spaces, word))
                    spaces, word = [], []
                spaces.append((token, font))
            else:
                word.append((token, font))
    if word:
        out.append((spaces, word))
    return out


def _glued(segments: Sequence[Sequence[Any]], pieces: Iterable[Run]) -> List[List[Any]]:
    """`segments` ([text, font]) with `pieces` appended: a piece in the last segment's font joins it."""
    out = [list(segment) for segment in segments]
    for text, font in pieces:
        if out and out[-1][1] == font:
            out[-1][0] += text
        else:
            out.append([text, font])
    return out


def _line(measure: Measure, segments: Sequence[Sequence[Any]]) -> Line:
    x, out = 0.0, []
    for text, font in segments:
        out.append((x, text, font))
        x += measure.width(text, font)
    return Line(tuple(out), x)


def wrap(measure: Measure, items: Sequence[Run], width: float, first_width: Optional[float] = None) -> List[Line]:
    """Greedy word wrap of runs over `width` (the first line `first_width`); spaces break, a word wider than a line
    breaks between characters. Neighboring pieces in one font are one segment, measured as drawn."""
    lines: List[Line] = []
    segments: List[List[Any]] = []  # [text, font]

    def limit() -> float:
        return first_width if (first_width is not None and not lines) else width

    for spaces, word in _words(items):
        candidate = _glued(segments, (spaces if segments else []) + word)
        if _line(measure, candidate).width <= limit():
            segments = candidate
            continue
        if segments:
            lines.append(_line(measure, segments))
        chunks = [word] if _line(measure, _glued([], word)).width <= limit() else _chop(measure, word, limit(), width)
        for index, chunk in enumerate(chunks):  # more than one: a word wider than a line
            if index:
                lines.append(_line(measure, segments))
            segments = _glued([], chunk)
    if segments or not lines:
        lines.append(_line(measure, segments))
    return lines


def _chop(measure: Measure, word: Sequence[Run], first: float, width: float) -> List[List[Run]]:
    """A word wider than a line cut between characters into pieces of at most `first`, then `width`."""
    out: List[List[Run]] = []
    current: List[Run] = []
    limit = first
    for text, font in word:
        for char in text:
            trial = current[:-1] + [(current[-1][0] + char, font)] if current and current[-1][1] == font else current + [(char, font)]
            if current and sum(measure.width(t, f) for t, f in trial) > limit:
                out.append(current)
                current, limit = [(char, font)], width
            else:
                current = trial
    return out + [current] if current else out


def ellipsised(measure: Measure, items: Sequence[Run], width: float) -> Line:
    """The first line of the runs, cut with '…' when they need more than one line."""
    lines = wrap(measure, items, width)
    if len(lines) == 1:
        return lines[0]
    segments = [[text, font] for _, text, font in lines[0].segments]
    while segments:
        segments[-1][0] = segments[-1][0].rstrip()
        trial = [(t, f) for t, f in segments[:-1]] + [(segments[-1][0] + '…', segments[-1][1])]
        if sum(measure.width(t, f) for t, f in trial) <= width:
            return wrap(measure, trial, width)[0]
        if len(segments[-1][0]) <= 1:
            segments.pop()
        else:
            segments[-1][0] = segments[-1][0][:-1]
    return wrap(measure, [('…', items[0][1] if items else Font())], width)[0]


class Canvas:
    """A unit under construction: primitives and a cursor `y` (inches down from the unit's top)."""

    def __init__(self, measure: Measure, width: float, status: str):
        self.m, self.width, self.status = measure, width, status
        self.items: List[Any] = []
        self.y = 0.0

    def lines(self, lines: Sequence[Line], x: float, size: float, ha: str = 'left', right: float = 0.0) -> float:
        """Draw wrapped lines from the cursor down (pitch from `size`); `ha='right'`: flush with `right`."""
        for line in lines:
            base = self.y + ASCENT * size / 72
            shift = right - line.width if ha == 'right' else x
            for dx, text, font in line.segments:
                if text.strip():
                    self.items.append(Text(shift + dx, base, text.rstrip() if text.endswith(' ') else text, font))
            self.y += LEADING * size / 72
        return self.y

    def paragraph(self, items: Sequence[Run], x: float, width: float, size: float) -> float:
        return self.lines(wrap(self.m, items, width), x, size)

    def hanging(self, items: Sequence[Run], x: float, width: float, size: float, hang: float) -> float:
        """A paragraph whose lines after the first are indented by `hang` (an episode line, a DECISIVE line)."""
        for index, line in enumerate(wrap(self.m, items, width - hang, first_width=width)):
            self.lines([line], x + (hang if index else 0.0), size)
        return self.y

    def unit(self, block: str, lead: float = 0.0, ruled: bool = False) -> Unit:
        return Unit(block, self.y, self.items, lead, ruled)


# -- the fit options (decision D8: what the column trims when even the grown figure does not hold it)


@dataclass(frozen=True)
class Fit:
    dropped_groups: Tuple[Tuple[int, str], ...] = ()  # (index, title) of the Numbers groups left out
    collapse_checks: bool = False  # the ✓ sub-checks of a step as one line
    compact_evidence: bool = False  # one line per evidence row

    @property
    def trimmed(self) -> List[str]:
        out = [f'numbers: {title}' for _, title in self.dropped_groups]
        return (
            out
            + (['checks: ✓ sub-checks collapsed'] if self.collapse_checks else [])
            + (['evidence: one line per row'] if self.compact_evidence else [])
        )


def trim_levels(column: vc.Column) -> List[Fit]:
    """The trimming steps in order: no trim, then one more Numbers group left out at a time (from the last; DECISIVE and
    ORBIT stay), then the ✓ sub-checks collapsed, then Evidence compact."""
    out = [Fit()]
    groups = column.numbers.groups
    droppable = [(index, g.title) for index, g in enumerate(groups) if not g.title.upper().startswith(KEEP_GROUPS)]
    dropped: Tuple[Tuple[int, str], ...] = ()
    for item in reversed(droppable):
        dropped += (item,)
        out.append(Fit(dropped_groups=dropped))
    out.append(Fit(dropped, collapse_checks=True))
    out.append(Fit(dropped, collapse_checks=True, compact_evidence=True))
    return out


# -- the blocks


def _title(canvas: Canvas, text: str, colour: str = INK) -> None:
    """A block title; its gray [[numbers]] (an Evidence title's gloss) at the body's numbers size, not the title's, and
    kept on one line (it moves whole under the title) when it fits one."""
    items = runs(text, Font(SANS, 'bold', TITLE_PT, colour), canvas.status, BODY_NUMBERS_PT)
    gloss = [(t, f) for t, f in items if f.family == MONO]
    if gloss and sum(canvas.m.width(t, f) for t, f in gloss) <= canvas.width:
        items = [(t.replace(' ', NBSP), f) if f.family == MONO else (t, f) for t, f in items]
    canvas.paragraph(items, 0.0, canvas.width, TITLE_PT)
    canvas.y += 0.02


def _header(m: Measure, column: vc.Column, width: float) -> List[Unit]:
    h = column.header
    c = Canvas(m, width, h.status)
    plaque_font = Font(SANS, 'bold', PLAQUE_PT, '#FFFFFF')
    pad_x, height = 0.13, 0.4
    word_w = m.width(h.word, plaque_font)
    c.items.append(Box(0.0, 0.0, word_w + 2 * pad_x, height, fill=status_colour(h.status), radius=0.06))
    mid = height / 2
    c.items.append(Text(pad_x, mid + 0.36 * PLAQUE_PT / 72, h.word, plaque_font))
    x = word_w + 2 * pad_x + 0.16
    dots = vc.LEVEL_DOTS.get(h.level, 0)
    dot_w, dot_h, dot_gap = 0.17, 0.095, 0.035
    for index in range(4):
        if h.level:
            c.items.append(Box(x, mid - dot_h / 2, dot_w, dot_h, fill=DOT_ON if index < dots else DOT_OFF, radius=0.02))
        else:
            c.items.append(Box(x, mid - dot_h / 2, dot_w, dot_h, edge=FAINT, lw=0.8, radius=0.02, dashed=True))
        x += dot_w + dot_gap
    level_font = Font(SANS, 'bold' if h.level else 'normal', LEVEL_PT, INK if h.level else MUTED)
    c.items.append(Text(x + 0.06, mid + 0.36 * LEVEL_PT / 72, h.level_word, level_font))
    used = x + 0.06 + m.width(h.level_word, level_font)
    edge, fill, ink = CHIP_COLOURS.get(h.chip.kind, CHIP_COLOURS[vc.CHIP_NONE])
    chip_font = Font(SANS, 'bold', CHIP_PT, ink)
    chip_runs = runs(h.chip.text, chip_font, h.status)
    chip_pad = 0.09
    lines = wrap(m, chip_runs, width - 2 * chip_pad)
    chip_w = max(line.width for line in lines) + 2 * chip_pad
    chip_h = len(lines) * LEADING * CHIP_PT / 72 + 0.07
    top = mid - chip_h / 2
    if used + 0.25 + chip_w > width:  # no room beside the level: the chip goes under the plaque, right-aligned
        top = height + 0.06
    c.items.append(Box(width - chip_w, top, chip_w, chip_h, fill=fill, edge=edge, lw=1.2, radius=0.05))
    c.y = top + 0.035
    c.lines(lines, 0.0, CHIP_PT, ha='right', right=width - chip_pad)
    c.y = max(height, top + chip_h)
    return [c.unit('header')]


def _subtitle(m: Measure, column: vc.Column, width: float) -> List[Unit]:
    s, status = column.subtitle, column.header.status
    c = Canvas(m, width, status)
    c.paragraph(runs(s.title, Font(SANS, 'bold', SUBTITLE_PT), status, SUBTITLE_PT - 1.5), 0.0, width, SUBTITLE_PT)
    if s.detail:
        c.paragraph(
            runs(s.detail, Font(SANS, 'normal', SUBTITLE_DETAIL_PT, MUTED), status, SUBTITLE_DETAIL_PT - 1), 0.0, width, SUBTITLE_DETAIL_PT
        )
    return [c.unit('subtitle', lead=SMALL_GAP)]


def _label_width(m: Measure, labels: Iterable[str], font: Font, cap: float) -> float:
    return min(cap, max((m.width(vc.plain(label), font) for label in labels), default=0.0))


def _key_facts(m: Measure, column: vc.Column, width: float) -> List[Unit]:
    status = column.header.status
    c = Canvas(m, width, status)
    pad = 0.1
    label_font, value_font = Font(SANS, 'normal', FACT_PT, MUTED), Font(SANS, 'normal', FACT_PT)
    label_w = _label_width(m, (f.label for f in column.key_facts), label_font, 0.3 * width)
    value_x = pad + label_w + 0.14
    c.y = pad - 0.02
    for fact in column.key_facts:
        top = c.y
        bottom_label = c.paragraph(runs(fact.label, label_font, status), pad, label_w, FACT_PT)
        c.y = top
        bottom_value = c.paragraph(runs(fact.value, value_font, status, FACT_NUMBERS_PT), value_x, width - value_x - pad, FACT_PT)
        c.y = max(bottom_label, bottom_value) + 0.015
    c.y += pad - 0.04
    c.items.insert(0, Box(0.0, 0.0, width, c.y, fill=FACTS_FILL, radius=0.06))
    return [c.unit('key_facts', lead=SMALL_GAP)]


def _why(m: Measure, column: vc.Column, width: float) -> List[Unit]:
    status = column.header.status
    c = Canvas(m, width, status)
    _title(c, vc.BLOCK_TITLES['why'])
    body = ' '.join(column.why.sentences)
    c.paragraph(runs(body, Font(SANS, 'normal', BODY_PT), status, BODY_NUMBERS_PT), 0.0, width, BODY_PT)
    if column.why.would_change:
        c.y += 0.04
        c.paragraph(runs(column.why.would_change, Font(SANS, 'normal', BODY_PT, MUTED), status, BODY_NUMBERS_PT), 0.0, width, BODY_PT)
    return [c.unit('why', lead=BLOCK_GAP, ruled=True)]


def _pill(m: Measure, pill: vc.Pill, status: str, max_width: float) -> Tuple[float, float, List[Any]]:
    """One pill as (width, height, items at its own top-left)."""
    pad_x, pad_y = 0.07, 0.035
    icon_font = Font(SANS, 'bold', PILL_PT + 0.5, icon_colour(pill.icon, status))
    icon_w = m.width(pill.icon, icon_font) + 0.05
    sub = Canvas(m, max_width, status)
    sub.y = pad_y
    ink = MUTED if pill.icon == vc.QUIET else INK
    lines = wrap(m, runs(pill.text, Font(SANS, 'normal', PILL_PT, ink), status, PILL_PT - 0.5), max_width - 2 * pad_x - icon_w)
    sub.items.append(Text(pad_x, pad_y + ASCENT * PILL_PT / 72, pill.icon, icon_font))
    sub.lines(lines, pad_x + icon_w, PILL_PT)
    w = 2 * pad_x + icon_w + max(line.width for line in lines)
    h = sub.y + pad_y - (LEADING - 1.15) * PILL_PT / 72
    fill, edge = PILL_STYLE.get(pill.icon, PILL_STYLE[vc.OK])
    radius = min(0.1, h / 2)
    return w, h, [Box(0.0, 0.0, w, h, fill=fill, edge=edge, lw=0.8, radius=radius)] + sub.items


def _shifted(items: Iterable[Any], dx: float, dy: float) -> List[Any]:
    return [replace(item, x=item.x + dx, y=item.y + dy) for item in items]


def _confidence(m: Measure, column: vc.Column, width: float) -> List[Unit]:
    status, conf = column.header.status, column.confidence
    c = Canvas(m, width, status)
    head = runs(f'Confidence: {conf.level or vc.NO_LEVEL}', Font(SANS, 'bold', TITLE_PT), status)
    if conf.because:
        head += [('  ', Font(SANS, 'normal', TITLE_PT))] + runs(conf.because, Font(SANS, 'normal', FACT_PT, MUTED), status, FACT_PT - 1)
    c.paragraph(head, 0.0, width, TITLE_PT)
    c.y += 0.04
    x, row_h, gap = 0.0, 0.0, 0.06
    for pill in conf.pills:
        w, h, items = _pill(m, pill, status, width)
        if x > 0 and x + w > width:
            c.y += row_h + gap
            x, row_h = 0.0, 0.0
        c.items += _shifted(items, x, c.y)
        x += w + gap
        row_h = max(row_h, h)
    c.y += row_h
    return [c.unit('confidence', lead=BLOCK_GAP, ruled=True)]


def _check_runs(
    question: str, answer: str, numbers: str, rule_id: str, size: float, status: str, quiet: bool, bold_question: bool
) -> List[Run]:
    ink = FAINT if quiet else INK
    out: List[Run] = []
    if question:
        out += runs(question, Font(SANS, 'normal', size, ink), status, CHECK_NUMBERS_PT, plain_as='bold' if bold_question else '')
    if answer:
        out += [(' ', Font(SANS, 'normal', size, ink))] + runs(answer, Font(SANS, 'normal', size, ink), status, CHECK_NUMBERS_PT)
    if numbers:
        out += [(' ', Font(SANS, 'normal', size, ink))] + font_runs(f'({vc.plain(numbers)})', Font(MONO, 'normal', CHECK_NUMBERS_PT, MUTED))
    if rule_id:  # glued to what precedes it (U+00A0): never alone on a line
        out += [(NBSP, Font(SANS, 'normal', size, ink))] + font_runs(f'[{rule_id}]', Font(MONO, 'normal', TAG_PT, FAINT))
    if quiet:
        out = [(text, replace(font, colour=FAINT)) for text, font in out]
    return out


def collapsed_children(children: Sequence[vc.SubCheck]) -> Tuple[vc.SubCheck, ...]:
    """The ✓ sub-checks as one line ('all N sub-checks clean'); the others kept as they are, first."""
    clean = [child for child in children if child.icon == vc.OK]
    if len(clean) < 2:
        return tuple(children)
    rest = tuple(child for child in children if child.icon != vc.OK)
    words = (COLLAPSED if not rest else COLLAPSED_SOME).format(n=len(clean))
    return rest + (vc.SubCheck(vc.OK, words),)


def _step(m: Measure, line: vc.CheckLine, width: float, status: str, fit: Fit) -> List[Tuple[List[Any], float]]:
    """One step as pieces (items, height): the step line with its first sub-check, then one piece per further
    sub-check, so a long step splits between its sub-checks (a heading never ends a sub-column alone). The ◆ step's
    tint is one box per piece, square when the step is in more than one piece."""
    pad = 0.035 if line.decides else 0.0
    c = Canvas(m, width, status)
    c.y = pad
    icon = vc.DECIDES if line.decides else line.icon
    icon_font = Font(SANS, 'bold', CHECK_PT + 1.5, icon_colour(icon, status))
    c.items.append(Text(CHECK_ICON_X, c.y + ASCENT * CHECK_PT / 72, icon, icon_font))
    number_font = Font(MONO, 'normal', CHECK_PT - 0.5, FAINT)
    c.items.append(Text(CHECK_NUMBER_X, c.y + ASCENT * CHECK_PT / 72, line.number, number_font))
    text_x = max(CHECK_TEXT_X, CHECK_NUMBER_X + m.width(line.number, number_font) + 0.06)  # '3–6' is wider than '3'
    items = _check_runs(line.question, line.answer, line.numbers, line.rule_id, CHECK_PT, status, line.quiet, True)
    c.paragraph(items, text_x, width - text_x - 0.04, CHECK_PT)
    pieces: List[Canvas] = [c]
    children = collapsed_children(line.children) if fit.collapse_checks else line.children
    for index, child in enumerate(children):
        if index:
            c = Canvas(m, width, status)
            pieces.append(c)
        c.y += 0.01
        c.items.append(
            Text(SUB_ICON_X, c.y + ASCENT * CHECK_PT / 72, child.icon, Font(SANS, 'bold', CHECK_PT, icon_colour(child.icon, status)))
        )
        child_runs = _check_runs('', '', child.numbers, child.rule_id, CHECK_PT, status, line.quiet, False)
        text_runs = runs(child.text, Font(SANS, 'normal', CHECK_PT, FAINT if line.quiet else INK), status, CHECK_NUMBERS_PT)
        c.paragraph(text_runs + child_runs, SUB_TEXT_X, width - SUB_TEXT_X - 0.04, CHECK_PT)
    c.y += pad
    if line.decides:
        fill = tint(status_colour(status))
        radius = 0.05 if len(pieces) == 1 else 0.0
        for index, piece in enumerate(pieces):  # a later piece's box reaches TINT_OVERLAP up: no anti-aliased seam
            up = TINT_OVERLAP if index else 0.0
            piece.items.insert(0, Box(0.0, -up, width, piece.y + up, fill=fill, radius=radius))
    return [(piece.items, piece.y) for piece in pieces]


def _legend(status: str) -> List[Run]:
    out: List[Run] = []
    for icon in vc.ICONS:
        word = vc.ICON_MEANING[icon]
        out += [(icon, Font(SANS, 'bold', LEGEND_PT, icon_colour(icon, status))), (f' {word}  ', Font(SANS, 'normal', LEGEND_PT, MUTED))]
    return out


def _checks(m: Measure, column: vc.Column, width: float, fit: Fit) -> List[Unit]:
    status = column.header.status
    units: List[Unit] = []
    c = Canvas(m, width, status)
    title_font = Font(SANS, 'bold', TITLE_PT)
    title = vc.BLOCK_TITLES['checks']
    legend = _legend(status)
    legend_lines = wrap(m, legend, width)
    legend_line = legend_lines[0]
    if len(legend_lines) == 1 and m.width(title, title_font) + 0.3 + legend_line.width <= width:
        c.items.append(Text(0.0, ASCENT * TITLE_PT / 72, title, title_font))
        c.y = (ASCENT * TITLE_PT - ASCENT * LEGEND_PT) / 72
        c.lines([legend_line], 0.0, LEGEND_PT, ha='right', right=width)
        c.y = LEADING * TITLE_PT / 72
    else:
        _title(c, title)
        c.paragraph(legend, 0.0, width, LEGEND_PT)
    c.y += 0.04
    for index, line in enumerate(column.checks):
        for position, (items, height) in enumerate(_step(m, line, width, status, fit)):
            if index == 0 and position == 0:  # the title stays with the first step
                c.items += _shifted(items, 0.0, c.y)
                c.y += height
                units.append(c.unit('checks', lead=BLOCK_GAP, ruled=True))
            else:  # a sub-check continues its step without a gap
                units.append(Unit('checks', height, items, 0.0 if position else STEP_GAP))
    if not column.checks:
        units.append(c.unit('checks', lead=BLOCK_GAP, ruled=True))
    return units


def _evidence(m: Measure, column: vc.Column, width: float, fit: Fit) -> List[Unit]:
    """One unit per row (a section's title stays with its first row), so the block splits between rows like Checks: a
    long Evidence no longer leaves the second sub-column half empty."""
    status = column.header.status
    label_font = Font(SANS, 'bold', EVIDENCE_PT)
    units: List[Unit] = []
    for evidence in column.evidence:
        c = Canvas(m, width, status)
        _title(c, evidence.title)
        label_w = _label_width(m, (row.label for row in evidence.rows), label_font, 0.32 * width)
        value_x = 0.24 + label_w + 0.12
        for position, row in enumerate(evidence.rows):
            if position:
                c = Canvas(m, width, status)
            top = c.y
            c.items.append(
                Text(0.04, top + ASCENT * EVIDENCE_PT / 72, row.icon, Font(SANS, 'bold', EVIDENCE_PT, icon_colour(row.icon, status)))
            )
            bottom_label = c.paragraph(runs(row.label, label_font, status), 0.24, label_w, EVIDENCE_PT)
            c.y = top
            value = runs(row.value, Font(SANS, 'normal', EVIDENCE_PT), status, EVIDENCE_PT - 1)
            if fit.compact_evidence:
                bottom_value = c.lines([ellipsised(m, value, width - value_x)], value_x, EVIDENCE_PT)
            else:
                bottom_value = c.paragraph(value, value_x, width - value_x, EVIDENCE_PT)
            c.y = max(bottom_label, bottom_value)
            if not units:
                units.append(c.unit('evidence', lead=BLOCK_GAP, ruled=True))
            else:
                units.append(c.unit('evidence', lead=SMALL_GAP if position == 0 else 0.01))
        if not evidence.rows:
            units.append(c.unit('evidence', lead=BLOCK_GAP if not units else SMALL_GAP, ruled=not units))
    return units


def _caveats(m: Measure, column: vc.Column, width: float) -> List[Unit]:
    status = column.header.status
    c = Canvas(m, width, status)
    title = vc.BLOCK_TITLES['caveats']
    if not column.caveats:
        c.paragraph(
            [(f'{title}: ', Font(SANS, 'bold', BODY_PT)), (vc.NONE_CAVEAT, Font(SANS, 'normal', BODY_PT, MUTED))], 0.0, width, BODY_PT
        )
        return [c.unit('caveats', lead=SMALL_GAP + 0.04)]
    pad = 0.1
    c.y = pad - 0.03
    c.paragraph([(title, Font(SANS, 'bold', BODY_PT))], pad, width - 2 * pad, BODY_PT)
    for text in column.caveats:
        c.items.append(Text(pad, c.y + ASCENT * CAVEAT_PT / 72, '•', Font(SANS, 'normal', CAVEAT_PT)))
        c.paragraph(runs(text, Font(SANS, 'normal', CAVEAT_PT), status, CAVEAT_PT - 1), pad + 0.14, width - 2 * pad - 0.14, CAVEAT_PT)
        c.y += 0.01
    c.y += pad - 0.05
    c.items.insert(0, Box(0.0, 0.0, width, c.y, fill=CAVEAT_FILL, radius=0.06))
    return [c.unit('caveats', lead=SMALL_GAP + 0.04)]


def _group(m: Measure, group: vc.NumberGroup, width: float, status: str) -> Tuple[List[Any], float]:
    """One number group (title, then label ··· value rows, the value flush right) in a cell `width` wide."""
    c = Canvas(m, width, status)
    c.paragraph([(group.title, Font(SANS, 'bold', GROUP_TITLE_PT))], 0.0, width, GROUP_TITLE_PT)
    label_font, value_font = Font(MONO, 'normal', NUMBERS_PT, INK), Font(MONO, 'normal', NUMBERS_PT, INK)
    if group.table:
        _table(c, group, width)
    for label, value in group.rows:
        if not vc.plain(label).strip():  # a row without a label (an episode line): left, wrapped under itself
            text = vc.plain(value)
            indent = m.width(text[: len(text) - len(text.lstrip(' '))], value_font)
            c.hanging(font_runs(text.lstrip(' '), value_font), indent, width - indent, NUMBERS_PT, 0.15)
            continue
        label_runs, value_runs = font_runs(vc.plain(label), label_font), font_runs(vc.plain(value), value_font)
        label_w = sum(m.width(t, f) for t, f in label_runs)
        value_w = sum(m.width(t, f) for t, f in value_runs)
        if label_w + 0.1 + value_w <= width:
            top = c.y
            c.lines([wrap(m, label_runs, width)[0]], 0.0, NUMBERS_PT)
            c.y = top
            c.lines([wrap(m, value_runs, width)[0]], 0.0, NUMBERS_PT, ha='right', right=width)
        else:  # the value under its label, flush right
            c.paragraph(label_runs, 0.0, width, NUMBERS_PT)
            c.lines(wrap(m, value_runs, width - 0.15), 0.0, NUMBERS_PT, ha='right', right=width)
    return c.items, c.y


def _table(c: Canvas, group: vc.NumberGroup, width: float) -> None:
    """A wide table (FREQUENCIES): each column starts at its own measured x (the widest cell plus two spaces, so a
    glyph from the fallback font or an empty header cell never shifts a column), the font shrunk (to TABLE_MIN_PT) to
    fit, a cell still too wide cut with '…'; the header row bold."""
    rows = [tuple(vc.plain(cell) for cell in row) for row in (group.table, *group.cells)]
    n = max(len(row) for row in rows)
    rows = [row + ('',) * (n - len(row)) for row in rows]
    size = NUMBERS_PT
    while True:
        fonts = [Font(MONO, 'bold' if index == 0 else 'normal', size, INK) for index in range(len(rows))]
        widths = [max(sum(c.m.width(t, f) for t, f in font_runs(row[j], font)) for row, font in zip(rows, fonts)) for j in range(n)]
        gap = 2 * c.m.width(' ', fonts[-1])
        if sum(widths) + gap * (n - 1) <= width or size <= TABLE_MIN_PT:
            break
        size -= 0.25
    xs = [sum(widths[:j]) + gap * j for j in range(n)]
    for row, font in zip(rows, fonts):
        top = c.y
        for x, cell in zip(xs, row):
            if cell.strip() and x < width:
                c.y = top
                c.lines([ellipsised(c.m, font_runs(cell, font), width - x)], x, size)
        c.y = top + LEADING * size / 72


def _numbers(m: Measure, column: vc.Column, width: float, fit: Fit) -> List[Unit]:
    status, numbers = column.header.status, column.numbers
    c = Canvas(m, width, status)
    _title(c, vc.BLOCK_TITLES['numbers'])
    if numbers.decisive:
        pad = 0.07
        top = c.y
        c.y += pad
        word = Font(MONO, 'bold', NUMBERS_PT, INK)
        for line in numbers.decisive:
            items = [('DECISIVE ', word)] + runs(line, Font(MONO, 'normal', NUMBERS_PT), status, NUMBERS_PT)
            c.hanging(items, pad, width - 2 * pad, NUMBERS_PT, m.width('DECISIVE ', word))
        c.y += pad - 0.02
        c.items.append(Box(0.0, top, width, c.y - top, edge=INK, lw=1.1, radius=0.04))
    units = [c.unit('numbers', lead=BLOCK_GAP, ruled=True)]
    gone = {index for index, _ in fit.dropped_groups}
    kept = [g for index, g in enumerate(numbers.groups) if index not in gone]
    n_grid = max(1, int((width + GRID_GAP) // (GRID_MIN_WIDTH + GRID_GAP)))
    cell = (width - GRID_GAP * (n_grid - 1)) / n_grid
    row: List[Tuple[List[Any], float]] = []

    def close_row():
        if row:
            items = [item for index, (cell_items, _) in enumerate(row) for item in _shifted(cell_items, index * (cell + GRID_GAP), 0.0)]
            units.append(Unit('numbers', max(h for _, h in row), items, SMALL_GAP))
            row.clear()

    for group in kept:
        if group.table or not any(vc.plain(label).strip() for label, _ in group.rows):  # a table, the episode lines:
            # the whole width
            close_row()
            items, height = _group(m, group, width, status)
            units.append(Unit('numbers', height, items, SMALL_GAP))
            continue
        row.append(_group(m, group, cell, status))
        if len(row) == n_grid:
            close_row()
    close_row()
    if fit.dropped_groups:
        note = Canvas(m, width, status)
        titles = ', '.join(title for _, title in fit.dropped_groups)
        words = f'{len(fit.dropped_groups)} more number groups ({titles}): the text export and the CSV files'
        note.paragraph([(words, Font(SANS, 'normal', FOOTER_PT + 0.5, MUTED))], 0.0, width, FOOTER_PT + 0.5)
        units.append(note.unit('numbers', lead=SMALL_GAP / 2))
    return units


def _footer(m: Measure, column: vc.Column, width: float) -> List[Unit]:
    c = Canvas(m, width, column.header.status)
    # a short item ('W28 tree: resonant/high', 'leaf 5d/6') never breaks inside; 'docs:' never ends a line
    items = [vc.plain(item) for item in column.footer]
    items = [item.replace(' ', NBSP) if len(item) <= FOOTER_GLUE_CHARS else item.replace('docs: ', 'docs:' + NBSP, 1) for item in items]
    c.paragraph(font_runs(' · '.join(items), Font(SANS, 'normal', FOOTER_PT, FAINT)), 0.0, width, FOOTER_PT)
    return [c.unit('footer', lead=BLOCK_GAP, ruled=True)]


@dataclass
class Block:
    name: str
    units: List[Unit]
    continuation: Optional[Unit] = None  # the head of a split block in the next sub-column


def build_blocks(column: vc.Column, width: float, fit: Optional[Fit] = None, measure: Optional[Measure] = None) -> List[Block]:
    """The shown blocks of `column` (`verdict_column.shown`) laid out for a sub-column `width` inches wide."""
    m, fit = measure or Measure(), fit or Fit()
    simple = {
        'header': _header,
        'subtitle': _subtitle,
        'key_facts': _key_facts,
        'why': _why,
        'confidence': _confidence,
        'caveats': _caveats,
        'footer': _footer,
    }
    fitted = {'checks': _checks, 'evidence': _evidence, 'numbers': _numbers}
    out: List[Block] = []
    for name in vc.BLOCKS:
        if not vc.shown(column, name):
            continue
        units = simple[name](m, column, width) if name in simple else fitted[name](m, column, width, fit)
        continuation = None
        if name in SPLITTABLE:
            c = Canvas(m, width, column.header.status)
            c.paragraph([(f'{vc.BLOCK_TITLES[name]} (continued)', Font(SANS, 'bold', BODY_PT, MUTED))], 0.0, width, BODY_PT)
            continuation = c.unit(name)
        out.append(Block(name, units, continuation))
    return out


# -- the flow over the sub-columns


@dataclass(frozen=True)
class Placement:
    unit: Unit
    column: int
    y: float  # top, inches below the area's top
    rule_y: Optional[float] = None  # the rule above it


def flow(blocks: Sequence[Block], height: float, n_columns: int) -> Tuple[List[Placement], bool]:
    """Place the units down the sub-columns. With two or more, the first `SECOND_SUBCOLUMN` block shown starts the
    second sub-column and the blocks before it stay in the first; after it a unit that does not fit goes to the next
    sub-column (a block whole, except the splittable ones, whose later units take the block's continuation head there).
    Returns the placements and whether they all end within `height`."""
    placements: List[Placement] = []
    col, y = 0, 0.0
    eps = 1e-6
    last = 0 if n_columns > 1 else n_columns - 1  # the last sub-column the current part may move into

    def put(unit: Unit, lead: float, ruled: bool):
        nonlocal y
        rule_y = y + lead / 2 if (ruled and y > 0) else None
        top = y + (lead if y > 0 else 0.0)
        placements.append(Placement(unit, col, top, rule_y))
        y = top + unit.height

    for block in blocks:
        if n_columns > 1 and last == 0 and block.name in SECOND_SUBCOLUMN:
            col, y, last = 1, 0.0, n_columns - 1
        whole = block.name not in SPLITTABLE
        pieces = [block.units] if whole else [[unit] for unit in block.units]
        for index, piece in enumerate(pieces):
            first = piece[0]
            need = (first.lead if y > 0 else 0.0) + sum(u.height for u in piece) + sum(u.lead for u in piece[1:])
            if y > 0 and y + need > height + eps and col < last:
                col, y = col + 1, 0.0
                if index > 0 and block.continuation is not None:
                    put(block.continuation, 0.0, False)
            for position, unit in enumerate(piece):
                put(unit, unit.lead, unit.ruled and (position == 0))
    ok = all(p.y + p.unit.height <= height + eps for p in placements)
    return placements, ok


def needed_height(blocks: Sequence[Block], n_columns: int) -> float:
    """The smallest sub-column height (inches) at which `flow` fits the blocks in `n_columns` (bisection)."""
    total = sum(u.height + u.lead for b in blocks for u in b.units) + sum(b.continuation.height for b in blocks if b.continuation)
    lo = max((u.height for b in blocks for u in b.units), default=0.0)
    hi = max(lo, total)
    if flow(blocks, lo, n_columns)[1]:
        return lo
    for _ in range(40):
        mid = (lo + hi) / 2
        if flow(blocks, mid, n_columns)[1]:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-3:
            break
    return hi


# -- the right area of the verdict figure


def area(layout: VerdictFigureLayout) -> Tuple[float, float, float, float]:
    """(x0, x1, top, bottom) of the right area, inches: from the first text column to the end of the last one."""
    first, last = layout.text_columns[0], layout.text_columns[-1]
    return first[0], last[0] + last[2], first[1] + first[3], first[1]


def subcolumn_width(layout: VerdictFigureLayout, n_columns: int = N_SUBCOLUMNS) -> float:
    x0, x1, _, _ = area(layout)
    return (x1 - x0 - SUBCOLUMN_GAP * (n_columns - 1)) / n_columns


def measure_column(column: vc.Column, layout: VerdictFigureLayout = LAYOUT_V1, fit: Optional[Fit] = None) -> float:
    """The height (inches) the right area needs for the whole column at the layout's width (in two sub-columns)."""
    blocks = build_blocks(column, subcolumn_width(layout), fit, Measure(layout.dpi))
    return needed_height(blocks, N_SUBCOLUMNS)


def _round_up(value: float, step: float = GROWTH_STEP_IN) -> float:
    return step * -(-round(value / step, 6) // 1)


@dataclass
class ColumnFit:
    """What `fit_layout` decided: the growth, the trimming level, and the heights (inches)."""

    fit: Fit
    grown_in: float
    height_needed_in: float  # the untrimmed column's need
    overflow: int  # 1 when even the grown, fully trimmed column does not fit


def plan(column: vc.Column, layout: VerdictFigureLayout, max_growth_in: float = COLUMN_MAX_GROWTH_IN) -> ColumnFit:
    """Decision D8 for `column` on `layout`: no growth when it fits; growth up to `max_growth_in` when that fits;
    else the cap and the first trimming level (`trim_levels`) that fits, or the last one with overflow 1."""
    m = Measure(layout.dpi)
    width = subcolumn_width(layout)
    available = area(layout)[2] - area(layout)[3]
    needed = needed_height(build_blocks(column, width, Fit(), m), N_SUBCOLUMNS)
    if needed <= available + 1e-6:
        return ColumnFit(Fit(), 0.0, needed, 0)
    if needed <= available + max_growth_in + 1e-6:
        grown = min(max_growth_in, _round_up(needed - available))
        return ColumnFit(Fit(), grown, needed, 0)
    levels = trim_levels(column)[1:]  # each trims more than the one before: bisect for the first that fits
    heights: Dict[int, float] = {}

    def height(index: int) -> float:
        if index not in heights:
            heights[index] = needed_height(build_blocks(column, width, levels[index], m), N_SUBCOLUMNS)
        return heights[index]

    room = available + max_growth_in + 1e-6
    lo, hi = 0, len(levels) - 1
    if height(hi) > room:
        return ColumnFit(levels[hi], max_growth_in, needed, 1)
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if height(mid) <= room else (mid + 1, hi)
    return ColumnFit(levels[lo], max_growth_in, needed, 0)


def fit_layout(
    column: vc.Column, layout: VerdictFigureLayout = LAYOUT_V1, max_growth_in: float = COLUMN_MAX_GROWTH_IN
) -> Tuple[VerdictFigureLayout, ColumnFit]:
    """The layout to build the figure on (grown when the column needs it, decision D8) and the plan to pass to
    `draw_column`. The figure builder calls this before it creates the figure."""
    decided = plan(column, layout, max_growth_in)
    return layout.grown(decided.grown_in), decided


@dataclass(frozen=True)
class Prepared:
    """A column measured before its figure exists (`prepare`): the column, the text it was built from (VerdictText)
    and the plan to draw it with."""

    column: vc.Column
    text: Any
    fit: ColumnFit


def prepare(column: vc.Column, text: Any, layout: VerdictFigureLayout = LAYOUT_V1) -> Tuple[VerdictFigureLayout, Prepared]:
    """The layout to build the figure on (`fit_layout`, decision D8) and the column ready for `draw_prepared`; both
    figures (`mmr_combined.draw_combined`, `secular_plots.draw_long_period_combined`) call it before the figure exists."""
    grown, fit = fit_layout(column, layout)
    return grown, Prepared(column, text, fit)


def draw_prepared(fig, prepared: Prepared, layout: VerdictFigureLayout) -> Dict[str, Any]:
    """Draw a `prepare`d column on the figure built on the grown `layout`: `draw_column`'s statistics with the
    `column` and the `text` it was built from."""
    stats = draw_column(fig, prepared.column, layout, prepared.fit)
    stats['column'], stats['text'] = prepared.column, prepared.text
    return stats


def draw_column(fig, column: vc.Column, layout: VerdictFigureLayout, fit: Optional[ColumnFit] = None) -> Dict[str, Any]:
    """Draw `column` into the right area of `fig` (built on `layout`, already grown by `fit_layout` when needed).
    Without `fit`, the column is trimmed to the layout as it is (no growth). Returns the statistics: `columns` (sub-columns
    used), `height_needed_in` (the untrimmed column), `grown_in` (the layout's growth), `trimmed` (what was left out),
    `overflow` (0/1) and `not_drawn` (units below the area, left out)."""
    m = Measure(layout.dpi)
    x0, _, top, bottom = area(layout)
    width = subcolumn_width(layout)
    if fit is None:
        fit = plan(column, layout, max_growth_in=0.0)
    blocks = build_blocks(column, width, fit.fit, m)
    placements, ok = flow(blocks, top - bottom, N_SUBCOLUMNS)
    drawn = [p for p in placements if p.y + p.unit.height <= top - bottom + 1e-6]
    for p in drawn:
        x = x0 + p.column * (width + SUBCOLUMN_GAP)
        _emit(fig, p, x, top, width)
    return {
        'columns': 1 + max((p.column for p in placements), default=0),
        'height_needed_in': round(fit.height_needed_in, 3),
        'grown_in': round(layout.growth_in, 3),
        'trimmed': fit.fit.trimmed,
        'overflow': int(fit.overflow or not ok),
        'not_drawn': len(placements) - len(drawn),
        'subcolumn_width_in': round(width, 3),
    }


def _emit(fig, placement: Placement, x: float, top: float, width: float) -> None:
    """Draw one placed unit; (x, top) = the sub-column's left edge and the area's top in figure inches."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    transform = fig.dpi_scale_trans
    if placement.rule_y is not None:
        y = top - placement.rule_y
        fig.add_artist(Line2D([x, x + width], [y, y], color=RULE, lw=0.8, transform=transform, gid=GID))
    y0 = top - placement.y
    for item in placement.unit.items:
        if isinstance(item, Box):
            radius = max(0.0, min(item.radius, item.h / 2, item.w / 2))
            patch = FancyBboxPatch(
                (x + item.x, y0 - item.y - item.h),
                item.w,
                item.h,
                boxstyle=f'round,pad=0,rounding_size={radius}',
                facecolor=item.fill,
                edgecolor=item.edge,
                linewidth=item.lw,
                linestyle='--' if item.dashed else '-',
                transform=transform,
                gid=GID,
            )
            fig.add_artist(patch)
        else:
            fig.text(
                x + item.x,
                y0 - item.y,
                item.text,
                transform=transform,
                fontsize=item.font.size,
                family=item.font.family,
                weight=item.font.weight,
                color=item.font.colour,
                ha='left',
                va='baseline',
                parse_math=False,
                gid=GID,
            )


def draw_column_figure(column: vc.Column, path, width_in: float = 6.0, dpi: int = LAYOUT_V1.dpi, margin_in: float = 0.25) -> Dict[str, Any]:
    """A standalone figure of the right column only, one column `width_in` wide and as tall as it needs (nothing
    trimmed): for the 'right column only' images and the docs. Saves to `path`; returns the statistics."""
    import matplotlib.pyplot as plt

    m = Measure(dpi)
    width = width_in - 2 * margin_in
    blocks = build_blocks(column, width, Fit(), m)
    height = needed_height(blocks, 1)
    fig = plt.figure(figsize=(width_in, height + 2 * margin_in), dpi=dpi)
    fig.patch.set_facecolor('white')
    try:
        placements, ok = flow(blocks, height, 1)
        for p in placements:
            _emit(fig, p, margin_in, height + margin_in, width)
        fig.savefig(path, dpi=dpi, facecolor='white')
    finally:
        plt.close(fig)
    return {'columns': 1, 'height_needed_in': round(height, 3), 'grown_in': 0.0, 'trimmed': [], 'overflow': int(not ok), 'not_drawn': 0}
