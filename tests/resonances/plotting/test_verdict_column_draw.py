"""The drawing of the right column (`plotting.verdict_column_draw`): synthetic columns (short, long, very long) on the
verdict figure's right area (two sub-columns), the growth of the figure (decision D8) and its cap, the trimming order,
the glyphs, the colors, and `VerdictFigureLayout.grown`."""

from functools import lru_cache

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from resonances.plotting import verdict_column as vc  # noqa: E402
from resonances.plotting import verdict_column_draw as d  # noqa: E402
from resonances.plotting.style import (  # noqa: E402
    CHIP_COLOURS,
    COLUMN_ICON_COLOURS,
    COLUMN_MAX_GROWTH_IN,
    LAYOUT_V1,
    STATUS_COLOURS,
    VerdictFigureLayout,
)

# -- synthetic columns (the 79653 3M+3J-7 mockup's structure; `scale` repeats the long blocks)

STEPS = (
    vc.CheckLine(
        vc.OK,
        '0',
        'Does σ librate?',
        'In part of the record: 2 episodes.',
        'misses the start by 7.1 P, the end by 0.12 P; need ≤ 1 P for “whole”',
        'C1',
        children=(
            vc.SubCheck(vc.OK, 'enough turning points', '7 ≥ 7', 'N'),
            vc.SubCheck(vc.OK, 'one pendulum', 'T₀ max/min 1.61 ≤ 2.77', 'L6'),
            vc.SubCheck(vc.THIN, 'center not fixed', 'shift/A 1.30 > 0.5', 'C2'),
            vc.SubCheck(vc.MISSING, 'distance to the separatrix', '7 of 7 cycles in the zone', 'L1'),
        ),
    ),
    vc.CheckLine(vc.THIN, '1', 'Does a sibling angle hold the center?', 'No, but one comes close.', '26 members; center 0.19 ≤ 0.8', 'S'),
    vc.CheckLine(vc.OK, '2', 'Does another resonance drive a?', 'No.', '2 candidates; best R² ≈ 0', 'W'),
    vc.CheckLine(vc.OK, '3', 'Does a swing with σ?', 'Yes.', 'B1 1.00 ≥ 0.1; slope/c 1.00, corr 0.999', 'B1'),
    vc.CheckLine(vc.QUIET, '4', 'Does the center drift without a cause?', 'Not needed: episodic.', quiet=True),
    vc.CheckLine(
        vc.OK,
        '5',
        'Is it σ’s own force?',
        'Yes, by the potential.',
        '',
        'R0',
        children=(
            vc.SubCheck(vc.OK, 'energy fits the potential inside episode #0', 'period law and V_E ≈ V_F hold', 'R0'),
            vc.SubCheck(vc.MISSING, 'force law', 'longest stretch 3.9 cycles < 5.25', 'L4'),
            vc.SubCheck(vc.QUIET, 'strength, order of magnitude', 'needed/scale 39.5; scale only', 'F'),
        ),
    ),
    vc.CheckLine(
        vc.DECIDES, '6', 'How does σ behave?', 'Episodic → !!transient!!; every cycle in the separatrix zone.', '', 'L2', decides=True
    ),
    vc.CheckLine(vc.THIN, 'Q', 'Can the data show it?', 'Yes, but filter-sensitive.', 'step 1.0° < 10°; filter gain 1.00', 'L8'),
)
GROUPS = (
    vc.NumberGroup(
        'LIBRATION (#1)', (('centre', '7.9°'), ('period P', '8.62 kyr'), ('half-amp med/max', '92.4/139°'), ('turning points', '9'))
    ),
    vc.NumberGroup('RESPONSE OF a', (('B1', '1.002'), ('slope/c · corr', '1.003 · 0.999'), ('force windows', '1 (need 2)'))),
    vc.NumberGroup('PENDULUM ENERGY', (('T₀ (observed)', '6.1–9.8 kyr'), ('well half-width', '2.5e-4 au'), ('μ needed', '2.05e-9'))),
    vc.NumberGroup('RESONANCE', (('order', '1'), ('a_res nominal', '2.429978 au'), ('a* / ⟨a⟩', '2.429948/2.429776'))),
    vc.NumberGroup('ORBIT (min / med / max)', (('e', '0.116/0.169/0.218'), ('i', '5.7/7.9/10.5°'), ('q_min / Q_max', '1.898/2.960 au'))),
    vc.NumberGroup('REFERENCE PERIODS', (('synodic M · J', '3.74 · 5.56 yr'), ('ϖ−ϖ₄ · ϖ−ϖ₅', '48.3 · 32.1 kyr'))),
    vc.NumberGroup(
        'FREQUENCIES', table=('mode', 'g ("/yr)', 'period (yr)'), cells=(('g5', '4.257', '304 000'), ('g6', '28.245', '45 880'))
    ),
)
SENTENCES = (
    'σ librates in two stretches, 4.8–32.5 kyr and 61.3 kyr to the end, and circulates in between <<b, l>>.',
    'Inside the first stretch the energy fits this resonance’s own potential, so the libration is its own and not induced.',
    'The force law could not be tested, because each stretch holds fewer than five cycles [[3.9 < 5.25]].',
    'The record ends while σ librates, so it cannot tell a short capture from a long stay.',
)


def make_column(scale: int = 1, status: str = 'transient', level: str = 'high', chip=(vc.CHIP_EXTEND, 'extend to 200 kyr')) -> vc.Column:
    groups = tuple(vc.NumberGroup(f'{g.title} {k}' if k else g.title, g.rows, g.table, g.cells) for k in range(scale) for g in GROUPS)
    extra = tuple(
        vc.CheckLine(
            vc.OK, f'{k}', 'Is this a repeated step of the long column?', 'Yes.', 'value 1.00 ≥ 0.5', 'X', children=STEPS[0].children
        )
        for k in range(3 * (scale - 1))
    )
    return vc.Column(
        family='mmr3',
        header=vc.Header(status, level, vc.Chip(*chip)),
        subtitle=vc.Subtitle('3M+3J−7+0+0+1 · its own libration in 2 episodes', 'three-body Mars–Jupiter · decided at step 6 (behavior)'),
        key_facts=(
            vc.Fact('σ', 'librates in **65%** of 100 kyr · 2 episodes · ends in **libration**'),
            vc.Fact('episodes', '#0 4.8–32.5 kyr P 10.4 kyr A 100° (111°) · #1 61.3–99 kyr P 8.6 kyr A 92° (139°) ◆'),
            vc.Fact('orbit', 'e 0.12–0.22 · i 5.7–10.5° · q_min 1.90 au · Q_max 2.96 au'),
            vc.Fact('run', '100 kyr forward · output 1 yr · cutoff 75 yr'),
        ),
        why=vc.Why(SENTENCES * scale, 'Would change the verdict: σ still librating at 200 kyr → resonant.'),
        confidence=vc.Confidence(
            level,
            'not certain because few cycles' if level not in ('', 'certain') else 'nothing holds it below certain',
            (
                vc.Pill(vc.THIN, '4.1 cycles in the deciding episode', 'core', 'high'),
                vc.Pill(vc.MISSING, 'separatrix margin not measured'),
                vc.Pill(vc.OK, 'alternatives ruled out'),
            ),
        ),
        checks=STEPS + extra,
        evidence=(
            vc.Evidence(
                'Evidence: is this 3M+3J−7’s own pendulum?',
                (
                    vc.EvidenceRow(
                        vc.OK, 'potential', 'energy fits the well built from a, inside episode #0, and the period law holds in it'
                    ),
                    vc.EvidenceRow(
                        vc.QUIET,
                        'strength (scale)',
                        'μ needed 2.05e−9 vs the scale 5.2e−11: ×39.5, scale only; an objection only from ×10³ after small divisors',
                    ),
                ),
            ),
        )
        * scale,
        caveats=('Every cycle lies in the model’s separatrix zone, and σ crosses it: expect exits and re-entries.',) * scale,
        numbers=vc.Numbers(('energy in own well, episode #0: period law and V_E ≈ V_F hold [R0]',), groups),
        footer=(
            'status 1',
            'leaf 5d/6 · angle rule G5w',
            'reason dynamic_episode',
            'flags filter_sensitive',
            'settings 1a2b3c4d5e',
            'docs: physics/status-tree.md',
        ),
        decided_rule='L2',
    )


def short_column() -> vc.Column:
    return vc.Column(
        family='mmr2',
        header=vc.Header('resonant', 'certain', vc.Chip(vc.CHIP_NONE, 'no action needed')),
        subtitle=vc.Subtitle('2J−1+0−1 · its own libration over the whole record', 'two-body, Jupiter 2:1, first order'),
        confidence=vc.Confidence('certain', 'nothing holds it below certain', (vc.Pill(vc.OK, '323 cycles counted'),)),
    )


AVAILABLE = d.area(LAYOUT_V1)[2] - d.area(LAYOUT_V1)[3]


@lru_cache(maxsize=None)
def needed(scale: int) -> float:
    return d.measure_column(make_column(scale), LAYOUT_V1)


def scale_needing(low: float, high: float) -> int:
    """The smallest scale whose need lies in [low, high) inches over the available height."""
    for scale in range(1, 12):
        if low <= needed(scale) - AVAILABLE < high:
            return scale
    raise AssertionError(f'no synthetic column needs {low}–{high} in more')


def drawn(column: vc.Column, layout: VerdictFigureLayout = LAYOUT_V1):
    grown, fit = d.fit_layout(column, layout)
    fig = plt.figure(figsize=grown.size, dpi=grown.dpi)
    stats = d.draw_column(fig, column, grown, fit)
    fig.canvas.draw()
    return fig, grown, stats


def ours(fig):
    return [a for a in fig.texts + fig.artists + fig.lines + fig.patches if a.get_gid() == d.GID]


def assert_inside(fig, layout: VerdictFigureLayout, tolerance_px: float = 1.5):
    """Every drawn text and box lies in the right area (pixels of the drawn figure; boxes may stick out by their edge)."""
    renderer = fig.canvas.get_renderer()
    x0, x1, top, bottom = d.area(layout)
    dpi = fig.dpi
    for artist in ours(fig):
        box = artist.get_window_extent(renderer)
        slack = tolerance_px + (artist.get_linewidth() * dpi / 72 / 2 if isinstance(artist, FancyBboxPatch) else 0.0)
        label = artist.get_text() if hasattr(artist, 'get_text') else type(artist).__name__
        assert box.x0 >= x0 * dpi - slack and box.x1 <= x1 * dpi + slack, (label, box, x0, x1)
        assert box.y0 >= bottom * dpi - slack and box.y1 <= top * dpi + slack, (label, box, bottom, top)


# -- columns on the figure


def test_short_column_keeps_both_parts_without_growth():
    """Even a short column uses the two places: header to Checks left, Evidence to the footer right (2026-11-05)."""
    fig, layout, stats = drawn(short_column())
    try:
        assert stats['overflow'] == 0 and stats['grown_in'] == 0 and stats['trimmed'] == []
        assert stats['columns'] == 2 and layout.size == LAYOUT_V1.size
        texts = {t.get_text() for t in fig.texts}
        assert {'RESONANT', 'certain', 'no action needed', 'Caveats:'} <= texts
        second = d.area(layout)[0] + d.subcolumn_width(layout) + d.SUBCOLUMN_GAP
        (caveats,) = [t for t in fig.texts if t.get_text() == 'Caveats:']
        assert caveats.get_position()[0] == pytest.approx(second)
        (badge,) = [t for t in fig.texts if t.get_text() == 'RESONANT']
        assert badge.get_position()[0] < second
        assert_inside(fig, layout)
    finally:
        plt.close(fig)


def test_evidence_starts_the_second_subcolumn():
    """The first block of `SECOND_SUBCOLUMN` that is shown is at the top of the second sub-column, every block before
    it in the first, every block after it in the second (the place of each block is the same on every figure)."""
    for column in (make_column(1), short_column()):
        blocks = d.build_blocks(column, d.subcolumn_width(LAYOUT_V1))
        placements, ok = d.flow(blocks, AVAILABLE, 2)
        assert ok
        names = [b.name for b in blocks]
        first = next(name for name in names if name in d.SECOND_SUBCOLUMN)
        by_block = {}
        for block in blocks:
            for unit in block.units:
                by_block.setdefault(block.name, []).extend(p for p in placements if p.unit is unit)
        assert by_block[first][0].column == 1 and by_block[first][0].y == 0.0
        for name in names:
            expected = 1 if names.index(name) >= names.index(first) else 0
            assert {p.column for p in by_block[name]} == {expected}, name


def test_long_column_uses_both_subcolumns_without_growth():
    assert needed(1) <= AVAILABLE
    fig, layout, stats = drawn(make_column(1))
    try:
        assert stats == {**stats, 'overflow': 0, 'grown_in': 0.0, 'trimmed': [], 'not_drawn': 0}
        assert stats['columns'] == 2
        assert_inside(fig, layout)
        # the second sub-column starts at x0 + width + gap
        x0 = d.area(layout)[0]
        second = x0 + d.subcolumn_width(layout) + d.SUBCOLUMN_GAP
        starts = {round(t.get_position()[0], 3) for t in fig.texts}
        assert any(abs(x - second) < 0.5 for x in starts)
    finally:
        plt.close(fig)


def test_growth_only_when_needed_and_as_much_as_needed():
    scale = scale_needing(0.05, COLUMN_MAX_GROWTH_IN - 0.2)
    need = needed(scale) - AVAILABLE
    fig, layout, stats = drawn(make_column(scale))
    try:
        assert stats['overflow'] == 0 and stats['trimmed'] == []
        assert need <= stats['grown_in'] < need + d.GROWTH_STEP_IN + 1e-9
        assert layout.size == (LAYOUT_V1.size[0], pytest.approx(LAYOUT_V1.size[1] + stats['grown_in']))
        assert tuple(fig.get_size_inches()) == pytest.approx(layout.size)
        assert_inside(fig, layout)
    finally:
        plt.close(fig)


def numbers_heavy(scale: int) -> vc.Column:
    """`make_column(1)` with the Numbers of `make_column(scale)`: the second sub-column is the long one."""
    return vc.Column(**{**make_column(1).__dict__, 'numbers': make_column(scale).numbers})


def test_very_long_column_grows_to_the_cap_then_trims_numbers_first():
    width = d.subcolumn_width(LAYOUT_V1)
    scale = next(
        k for k in range(1, 40) if d.needed_height(d.build_blocks(numbers_heavy(k), width), 2) - AVAILABLE > COLUMN_MAX_GROWTH_IN + 0.5
    )
    column = numbers_heavy(scale)
    fig, layout, stats = drawn(column)
    try:
        assert stats['grown_in'] == pytest.approx(COLUMN_MAX_GROWTH_IN)
        assert stats['trimmed'] and all(item.startswith('numbers: ') for item in stats['trimmed'])
        titles = [g.title for g in column.numbers.groups]
        dropped = [item[len('numbers: ') :] for item in stats['trimmed'] if item.startswith('numbers: ')]
        assert dropped == [t for t in reversed(titles) if not t.startswith('ORBIT')][: len(dropped)]  # from the last
        texts = {t.get_text() for t in fig.texts}
        assert 'DECISIVE ' in texts or 'DECISIVE' in texts
        assert any(t.startswith('ORBIT') for t in texts)
        if stats['overflow'] == 0:
            assert_inside(fig, layout)
    finally:
        plt.close(fig)


def test_trimming_beyond_numbers_collapses_checks_then_compacts_evidence():
    scale = next(k for k in range(7, 14) if d.plan(make_column(k), LAYOUT_V1).fit.compact_evidence)
    column = make_column(scale)
    fig, layout, stats = drawn(column)
    try:
        trimmed = stats['trimmed']
        assert trimmed[-2:] == ['checks: ✓ sub-checks collapsed', 'evidence: one line per row']
        assert all(item.startswith('numbers: ') and 'ORBIT' not in item for item in trimmed[:-2])
        assert len(trimmed) - 2 == sum(1 for g in column.numbers.groups if not g.title.startswith('ORBIT'))
        assert stats['grown_in'] == pytest.approx(COLUMN_MAX_GROWTH_IN)
        if stats['overflow'] == 0:
            assert_inside(fig, layout)
    finally:
        plt.close(fig)


def test_trim_levels_order():
    """Numbers groups from the last (ORBIT and DECISIVE stay), then the ✓ sub-checks, then Evidence compact."""
    column = make_column(1)
    levels = d.trim_levels(column)
    titles = [g.title for g in column.numbers.groups]
    droppable = [t for t in titles if not t.startswith('ORBIT')]
    assert levels[0] == d.Fit()
    for k, level in enumerate(levels[1 : 1 + len(droppable)], start=1):
        assert [t for _, t in level.dropped_groups] == list(reversed(droppable))[:k]
        assert not level.collapse_checks and not level.compact_evidence
    assert levels[-2].collapse_checks and not levels[-2].compact_evidence
    assert levels[-1].collapse_checks and levels[-1].compact_evidence
    assert all('ORBIT (min / med / max)' not in [t for _, t in level.dropped_groups] for level in levels)


def test_trimming_never_drops_what_must_stay():
    """At the last trimming level the header, subtitle, key facts, why, confidence, every step line and every ⚠/✗/?
    sub-check are still drawn; Evidence rows are one line each."""
    column = make_column(2)
    last = d.trim_levels(column)[-1]
    blocks = d.build_blocks(column, 5.5, last)
    names = [b.name for b in blocks]
    assert names[:5] == ['header', 'subtitle', 'key_facts', 'why', 'confidence']
    assert 'evidence' in names and 'caveats' in names
    checks = next(b for b in blocks if b.name == 'checks')
    words = ' '.join(item.text for unit in checks.units for item in unit.items if isinstance(item, d.Text))
    assert all(line.question in words for line in column.checks)
    assert 'center not fixed' in words and 'distance to the separatrix' in words and 'force law' in words
    assert 'enough turning points' not in words and '2 other sub-checks clean' in words
    evidence = next(b for b in blocks if b.name == 'evidence')
    full = next(b for b in d.build_blocks(column, 5.5, d.Fit()) if b.name == 'evidence')
    assert evidence.units[0].height < full.units[0].height


def test_collapsed_children():
    ok = vc.SubCheck(vc.OK, 'clean')
    thin = vc.SubCheck(vc.THIN, 'thin')
    assert d.collapsed_children((ok, ok, ok)) == (vc.SubCheck(vc.OK, 'all 3 sub-checks clean'),)
    assert d.collapsed_children((ok, thin, ok)) == (thin, vc.SubCheck(vc.OK, '2 other sub-checks clean'))
    assert d.collapsed_children((ok, thin)) == (ok, thin)  # one ✓ alone stays as it is


def test_overflow_reported_when_nothing_fits():
    tiny = VerdictFigureLayout(text_columns=((12.45, 0.25, 3.72, 1.0), (16.29, 0.25, 3.72, 1.0), (20.13, 0.25, 3.72, 1.0)))
    plan = d.plan(make_column(1), tiny, max_growth_in=0.0)
    assert plan.overflow == 1 and plan.fit == d.trim_levels(make_column(1))[-1]
    fig = plt.figure(figsize=tiny.size, dpi=tiny.dpi)
    try:
        stats = d.draw_column(fig, make_column(1), tiny, plan)
        assert stats['overflow'] == 1 and stats['not_drawn'] > 0
        fig.canvas.draw()
        assert_inside(fig, tiny)
    finally:
        plt.close(fig)


# -- measuring and flow


def test_needed_height_is_the_smallest_that_fits():
    blocks = d.build_blocks(make_column(1), d.subcolumn_width(LAYOUT_V1))
    height = d.needed_height(blocks, 2)
    assert d.flow(blocks, height, 2)[1]
    assert not d.flow(blocks, height - 0.02, 2)[1]
    assert d.needed_height(blocks, 1) > height  # one column needs more
    assert needed(2) > needed(1)


def test_wrap_keeps_every_line_within_the_width():
    measure = d.Measure(120)
    text = ' '.join(SENTENCES) + ' averyveryverylongwordwithoutanyspacesthatmustbebrokenbetweencharacters' * 2
    items = d.runs(text, d.Font(size=9.5))
    for width in (1.0, 2.5, 5.5):
        lines = d.wrap(measure, items, width)
        assert all(line.width <= width + 1e-9 for line in lines)
        joined = ''.join(t for line in lines for _, t, _ in line.segments).replace(' ', '')
        assert joined == vc.plain(text).replace(' ', '')


def test_block_split_only_for_checks_evidence_and_numbers():
    """A whole block moves to the next sub-column; Checks (between sub-check lines), Evidence (between rows) and
    Numbers (between groups) split, and the continuation head starts the second part."""
    blocks = d.build_blocks(make_column(1), d.subcolumn_width(LAYOUT_V1))
    height = d.needed_height(blocks, 2)
    placements, ok = d.flow(blocks, height, 2)
    assert ok
    by_block = {}
    for p in placements:
        by_block.setdefault(p.unit.block, set()).add(p.column)
    assert all(len(columns) == 1 for name, columns in by_block.items() if name not in d.SPLITTABLE)
    for name in d.SPLITTABLE:
        if len(by_block.get(name, ())) == 2:
            heads = [p for p in placements if p.column == 1 and p.unit is next(b for b in blocks if b.name == name).continuation]
            assert len(heads) == 1


def test_checks_split_between_sub_checks_and_a_step_line_keeps_its_first():
    """A step is one unit per sub-check line: the step line travels with its first sub-check, each further sub-check
    is a unit of its own (no gap before it), so one long step no longer moves whole to the next sub-column."""
    column = make_column(1)
    checks = next(b for b in d.build_blocks(column, d.subcolumn_width(LAYOUT_V1)) if b.name == 'checks')
    expected = sum(max(1, len(line.children)) for line in column.checks)
    assert len(checks.units) == expected
    texts = [[item.text for item in unit.items if isinstance(item, d.Text)] for unit in checks.units]
    step0 = STEPS[0]
    assert step0.number in texts[0] and step0.children[0].icon in texts[0]  # with the title, the step and sub-check 1
    assert any(t.startswith('one pendulum') for t in texts[1]) and step0.number not in texts[1]
    assert checks.units[1].lead == 0.0 and checks.units[len(step0.children)].lead == d.STEP_GAP


def test_a_long_step_stays_in_the_first_subcolumn():
    """A step of many sub-checks (79653's step 5) stays with Checks in the first sub-column: the figure grows for it
    (D8) instead of pushing Evidence down the second."""
    children = tuple(vc.SubCheck(vc.OK, f'sub-check number {k} of a long step', 'value 1.00 ≥ 0.5', 'X') for k in range(40))
    long_step = vc.CheckLine(vc.OK, '5', 'Is it σ’s own force?', 'Yes.', '', 'R0', children=children)
    column = make_column(1)
    column = vc.Column(**{**column.__dict__, 'checks': column.checks[:5] + (long_step,) + column.checks[5:]})
    blocks = d.build_blocks(column, d.subcolumn_width(LAYOUT_V1))
    height = d.needed_height(blocks, 2)
    placements, ok = d.flow(blocks, height, 2)
    assert ok
    long_units = [p for p in placements if any(isinstance(i, d.Text) and i.text.startswith('sub-check number') for i in p.unit.items)]
    assert {p.column for p in long_units} == {0}
    (evidence,) = [b for b in blocks if b.name == 'evidence']
    assert [p for p in placements if p.unit is evidence.units[0]][0].y == 0.0


def test_a_split_decides_step_tints_every_piece():
    """The ◆ step split into pieces: each piece carries its own tint box (square, the later ones reaching up over the
    seam), so the deciding step reads as one band in either sub-column."""
    line = vc.CheckLine(vc.DECIDES, '6', 'How does σ behave?', 'Episodic.', '', 'L2', children=STEPS[0].children, decides=True)
    pieces = d._step(d.Measure(), line, 5.0, 'transient', d.Fit())
    assert len(pieces) == len(line.children)
    for index, (items, height) in enumerate(pieces):
        box = items[0]
        assert isinstance(box, d.Box) and box.fill == d.tint(STATUS_COLOURS['transient']) and box.radius == 0.0
        assert box.y == pytest.approx(-d.TINT_OVERLAP if index else 0.0) and box.y + box.h == pytest.approx(height)


def test_pills_quiet_for_a_limited_input_that_does_not_hold_the_level():
    """79653-like inputs on a low row (N holds it): S filter-sensitive (at most high) is not the limit: '–', gray,
    with where it alone would cap the level; ✓ only for inputs in their best state; a full veto ✓ below its ceiling."""
    from resonances.plotting import verdict_column_common as common

    row = {'pol_level_inputs': 'N:2<3:low;core:2<3:high;M:full:certain;S:thin:high;V:clean:certain'}
    pills = {pill.key: pill for pill in common.pills(row, 'low')}
    assert pills['N'].icon == vc.THIN
    assert pills['S'].icon == vc.QUIET and pills['S'].text == 'filter-sensitive — alone it would cap the level at high'
    assert pills['core'].icon == vc.QUIET and pills['core'].text.endswith('— alone it would cap the level at high')
    # the bars say which level they open: N 2 < 3 holds the row at low, so 'medium needs ≥ 3'; core 2 < 3 at high
    assert pills['N'].text == '2 cycles counted [[medium needs ≥ 3]]'
    assert pills['core'].text.startswith('2 cycles between the deciding episode’s turning points [[edges not added; certain needs ≥ 3]]')
    assert pills['M'].icon == vc.OK and pills['V'].icon == vc.OK
    assert [p.icon for p in common.pills(row, 'low')] == [vc.THIN, vc.QUIET, vc.QUIET, vc.OK, vc.OK]
    vetoed = {'pol_level_inputs': 'veto:full(owner):high;second_capture:unresolved:medium'}
    pills = {pill.key: pill for pill in common.pills(vetoed, 'medium')}
    assert pills['veto'].icon == vc.OK and pills['veto'].text == 'overruled by another resonance that owns the libration'
    assert pills['second_capture'].icon == vc.THIN
    # drawn: the '–' icon grey, its words muted
    w, h, items = d._pill(d.Measure(), common.pills(row, 'low')[1], 'resonant', 5.0)
    texts = [item for item in items if isinstance(item, d.Text)]
    assert texts[0].text == vc.QUIET and texts[0].font.colour == COLUMN_ICON_COLOURS[vc.QUIET]
    assert all(t.font.colour == d.MUTED for t in texts[1:] if t.font.family == d.SANS)


def test_problems_read_the_pill_icons():
    """✓ on a limited input and – on an input that holds the level break the contract; – on a limited input above the
    level does not."""
    column = short_column()
    level = 'medium'
    base = dict(column.__dict__, header=vc.Header('resonant', level, column.header.chip))

    def with_pills(*pills):
        return vc.Column(**{**base, 'confidence': vc.Confidence(level, 'not certain because few cycles', pills)})

    quiet = vc.Pill(vc.QUIET, 'filter-sensitive — would allow high', 'S', 'high')
    thin = vc.Pill(vc.THIN, '3 cycles counted', 'N', 'medium')
    assert vc.problems(with_pills(thin, quiet)) == []
    assert any('✓ pill of a limited input' in p for p in vc.problems(with_pills(thin, vc.Pill(vc.OK, 'filter-sensitive', 'S', 'high'))))
    assert vc.problems(with_pills(thin, vc.Pill(vc.OK, 'overruled by the owner found', 'veto', 'high'))) == []
    assert any('– pill that holds' in p for p in vc.problems(with_pills(vc.Pill(vc.QUIET, '3 cycles counted', 'N', 'medium'))))


# -- look: glyphs, colours


def _font_cmap(family: str, weight: str) -> set:
    path = font_manager.findfont(font_manager.FontProperties(family=family, weight=weight))
    return set(TTFont(path)['cmap'].getBestCmap())


def test_every_drawn_character_is_in_its_dejavu_font():
    fig, _, _ = drawn(make_column(1))
    try:
        cmaps = {}
        for text in (t for t in fig.texts if t.get_gid() == d.GID):
            family = text.get_fontfamily()[0]
            assert family in ('DejaVu Sans', 'DejaVu Sans Mono')
            key = (family, text.get_fontweight())
            cmaps.setdefault(key, _font_cmap(*key))
            missing = [ch for ch in text.get_text() if ord(ch) not in cmaps[key] and not ch.isspace()]
            assert not missing, (text.get_text(), key, missing)
    finally:
        plt.close(fig)


def test_font_runs_route_a_missing_character_to_dejavu_sans():
    mono = _font_cmap('DejaVu Sans Mono', 'normal')
    absent = chr(min(_font_cmap('DejaVu Sans', 'normal') - mono))  # in DejaVu Sans, not in DejaVu Sans Mono
    pieces = d.font_runs(f'a{absent}b', d.Font(d.MONO, 'normal', 7.0))
    assert [(t, f.family) for t, f in pieces] == [('a', d.MONO), (absent, d.SANS), ('b', d.MONO)]
    nowhere = '\U000f0000'  # a private-use character: no DejaVu font has it
    assert d.font_runs(nowhere, d.Font())[0][0] == '?'


def test_status_colours_are_the_spec_palette():
    """Decision D4 (the spec's palette; invalid keeps #b2182b)."""
    assert STATUS_COLOURS == {
        'resonant': '#1b7a3a',
        'transient': '#b35400',
        'near_separatrix': '#6b3fa0',
        'non_resonant': '#4a4f57',
        'uncertain': '#2f5d9e',
        'invalid': '#b2182b',
    }
    # 0.1 of #1B7A3A over white, by hand: 0.1*27+229.5=232.2 (E8), 0.1*122+229.5=241.7 (F2), 0.1*58+229.5=235.3 (EB)
    assert d.tint('#1B7A3A') == '#E8F2EB'


def test_plaque_chip_and_decides_row_colours():
    fig, _, _ = drawn(make_column(1, status='transient', chip=(vc.CHIP_GLANCE, 'glance at (b), (i)')))
    try:
        patches = [p for p in ours(fig) if isinstance(p, FancyBboxPatch)]
        fills = {to_hex(p.get_facecolor()).upper() for p in patches}
        assert '#B35400' in fills  # the plaque
        assert d.tint('#B35400').upper() in fills  # the ◆ row
        edge, fill, ink = CHIP_COLOURS[vc.CHIP_GLANCE]
        chip = [p for p in patches if to_hex(p.get_edgecolor()).upper() == edge.upper()]
        assert chip and to_hex(chip[0].get_facecolor()).upper() == fill.upper()
        chip_text = next(t for t in fig.texts if t.get_text() == 'glance at (b), (i)')
        assert to_hex(chip_text.get_color()).upper() == ink.upper()
        diamond = next(t for t in fig.texts if t.get_text() == vc.DECIDES)
        assert to_hex(diamond.get_color()).upper() == '#B35400'
        plaque = next(t for t in fig.texts if t.get_text() == 'TRANSIENT')
        assert to_hex(plaque.get_color()).upper() == '#FFFFFF' and plaque.get_fontsize() == d.PLAQUE_PT
    finally:
        plt.close(fig)


def test_no_level_draws_dashed_dots():
    column = make_column(1, status='uncertain', level='', chip=(vc.CHIP_EXTEND, 'extend to 200 kyr'))
    header = d.build_blocks(column, 5.5)[0]
    dots = [item for item in header.units[0].items if isinstance(item, d.Box) and item.w == pytest.approx(0.17)]
    assert len(dots) == 4 and all(dot.dashed for dot in dots)
    assert any(isinstance(item, d.Text) and item.text == vc.NO_LEVEL for item in header.units[0].items)


# -- the grown layout


RECTS = ('header', 'panels', 'colour_bar', 'legend')


def _from_top(layout: VerdictFigureLayout, rect):
    x, y, w, h = rect
    return (x, layout.size[1] - (y + h), w, h)


def test_grown_layout_keeps_the_left_panels_at_the_top():
    delta = 2.35
    grown = LAYOUT_V1.grown(delta)
    assert grown.size == (24.0, 13.5 + delta) and grown.growth_in == delta
    for name in RECTS:
        assert _from_top(grown, getattr(grown, name)) == pytest.approx(_from_top(LAYOUT_V1, getattr(LAYOUT_V1, name)))
    for new, old in zip(grown.bottom, LAYOUT_V1.bottom):
        assert _from_top(grown, new) == pytest.approx(_from_top(LAYOUT_V1, old))
    for new, old in zip(grown.text_columns, LAYOUT_V1.text_columns):
        assert (new[0], new[1], new[2]) == old[:3] and new[3] == pytest.approx(old[3] + delta)
    assert LAYOUT_V1.grown(0.0) is LAYOUT_V1
    assert LAYOUT_V1.grown(1.0).grown(1.5).size[1] == pytest.approx(16.0)


def test_grown_figure_left_panels_unchanged_in_pixels():
    """Axes placed at the left rects of the grown figure sit at the same pixels from the top as on the base figure."""

    def extents(layout):
        width, height = layout.size
        fig = plt.figure(figsize=layout.size, dpi=layout.dpi)
        rects = [layout.header, layout.panels, layout.colour_bar, layout.legend, *layout.bottom]
        axes = [fig.add_axes((x / width, y / height, w / width, h / height)) for x, y, w, h in rects]
        fig.canvas.draw()
        top = fig.bbox.height
        out = [(b.x0, top - b.y1, b.width, b.height) for b in (ax.get_window_extent() for ax in axes)]
        plt.close(fig)
        return out

    base, grown = extents(LAYOUT_V1), extents(LAYOUT_V1.grown(3.0))
    for a, b in zip(base, grown):
        assert a == pytest.approx(b, abs=1.0)


# -- the standalone column


def test_column_figure(tmp_path):
    path = tmp_path / 'column.png'
    stats = d.draw_column_figure(make_column(1), path, width_in=6.0)
    assert path.stat().st_size > 0
    assert stats['overflow'] == 0 and stats['columns'] == 1
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    assert width == 6.0 * 120
    assert height == pytest.approx((stats['height_needed_in'] + 0.5) * 120, abs=2)


def test_wrap_never_breaks_inside_a_word_across_fonts():
    """'Δa**+0.0019**au' is one word in two fonts, and a rule tag is glued to its numbers: neither starts a line."""
    measure = d.Measure(120)
    glued = d.runs('word ' * 6 + 'Δa**+0.0019**au', d.Font(size=9.5))
    for width in (1.0, 1.3, 2.0):  # the word is about 0.9 in wide
        texts = [''.join(t for _, t, _ in line.segments).strip() for line in d.wrap(measure, glued, width)]
        assert any('Δa+0.0019au' in text for text in texts), (width, texts)
    tagged = d._check_runs('Does another resonance drive a?', 'No.', '2 candidates; best R² ≈ 0', 'W', 8.5, 'resonant', False, True)
    for width in (1.5, 2.0, 3.0, 4.5):
        lines = [''.join(t for _, t, _ in line.segments) for line in d.wrap(measure, tagged, width)]
        assert not any(line.strip().startswith('[W]') for line in lines), lines


def test_widths_are_measured_once_for_every_measure():
    """The plan and the drawing of a figure each make a `Measure`; a width measured by one is not measured again by
    the other (the cache is the module's, keyed by dpi), and it equals the renderer's own width."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    font = d.Font(size=9.5)
    text = 'center passes whole turns (cache probe)'
    first = d.Measure(120).width(text, font)
    hits = d._text_width.cache_info().hits
    assert d.Measure(120).width(text, font) == first and d._text_width.cache_info().hits == hits + 1
    prop = FontProperties(family=font.family, weight=font.weight, size=font.size)
    assert first == pytest.approx(RendererAgg(10, 10, 120).get_text_width_height_descent(text, prop, ismath=False)[0] / 120)
    assert d.Measure(240).width(text, font) == pytest.approx(first, rel=0.05)  # another dpi: its own entry, same inches


def test_a_prepared_column_draws_with_its_column():
    """`prepare` grows the layout as `fit_layout` does; `draw_prepared` draws with that plan and hands back the
    column."""
    column = make_column(2)
    grown, prepared = d.prepare(column, LAYOUT_V1)
    expected_layout, expected_fit = d.fit_layout(column, LAYOUT_V1)
    assert grown == expected_layout and prepared.fit == expected_fit
    fig = plt.figure(figsize=grown.size, dpi=grown.dpi)
    try:
        stats = d.draw_prepared(fig, prepared, grown)
        assert stats['column'] is column
        assert stats['grown_in'] == pytest.approx(expected_fit.grown_in) and stats['overflow'] == expected_fit.overflow
    finally:
        plt.close(fig)
