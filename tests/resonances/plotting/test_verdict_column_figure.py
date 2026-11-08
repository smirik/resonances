"""The verdict figures draw the ten-block right column (`verdict_column`, the redesign of 2026-11-04) that the package
exports (`verdict_column_mmr.figure_column`, `verdict_column_long.figure_column`), and grow down when it does not fit
(decision D8): the left panels keep their geometry at the top, the growth is capped at `style.COLUMN_MAX_GROWTH_IN`.

Synthetic analyses only (the libration of `test_impl_figure_v1`, the ZLK libration of `test_w29_figures`)."""

from dataclasses import replace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting import secular_plots as sp  # noqa: E402
from resonances.plotting import verdict_column as vc  # noqa: E402
from resonances.plotting import verdict_column_draw as draw  # noqa: E402
from resonances.plotting import verdict_column_long as vl  # noqa: E402
from resonances.plotting import verdict_column_mmr as vm  # noqa: E402
from resonances.plotting.style import COLUMN_MAX_GROWTH_IN, LAYOUT_V1  # noqa: E402
from tests.resonances.analysis.test_impl_figure_v1 import libration_case, rows_of  # noqa: E402
from tests.resonances.plotting.test_w29_figures import SETTINGS, _zlk  # noqa: E402

# the text area cut to 2 in under the header's level (13.0 in): a column that needs more grows the figure
SHORT = replace(LAYOUT_V1, text_columns=tuple((x, 11.0, w, 2.0) for x, _, w, _ in LAYOUT_V1.text_columns))


def _drawn(fig) -> str:
    return ' '.join(text.get_text() for text in fig.texts if text.get_gid() == draw.GID)


def _from_top(fig):
    """Each axes' (left, distance of its top from the figure's top, width, height) in inches."""
    width, height = fig.get_size_inches()
    out = []
    for ax in fig.axes:
        x0, y0, w, h = ax.get_position().bounds
        out.append(tuple(np.round((x0 * width, height * (1 - y0 - h), w * width, h * height), 6)))
    return out


def test_the_mmr_figure_draws_the_exported_column():
    analysis, series = libration_case()
    rows = rows_of(analysis)
    fig = mc.draw_combined(analysis, series, rows=rows, settings=SETTINGS)
    try:
        column = fig.text_stats['column']
        assert column == vm.figure_column(analysis, rows, SETTINGS) and vc.problems(column) == []
        assert vc.column_text(column) == vc.column_text(vm.figure_column(analysis, rows, SETTINGS))
        # the hand-built analysis has no leaf in its decision path: the one ◆ says so instead of naming a rule
        decides = vc.find(column, vm.UNRECORDED_NUMBER)
        assert decides.decides and column.decided_rule == '' and 'no deciding rule recorded' in column.subtitle.detail
        assert fig.text_stats['overflow'] == 0 and fig.text_stats['grown_in'] == 0
        assert tuple(fig.get_size_inches()) == LAYOUT_V1.size  # a column that fits leaves the figure as it was
        drawn = _drawn(fig)
        assert column.header.word in drawn and '↑' not in drawn
        assert fig.text_stats['text'].status == column.header.status  # the VerdictText the column was built from
    finally:
        plt.close(fig)


def test_the_zlk_figure_draws_the_exported_column():
    analysis = _zlk(2.5)
    fig = sp.draw_long_period_combined(analysis, settings=SETTINGS)
    try:
        column = fig.text_stats['column']
        assert column == vl.figure_column(analysis, None, None, SETTINGS) and vc.problems(column) == []
        assert column.family == 'zlk' and fig.text_stats['overflow'] == 0
        assert column.header.word in _drawn(fig)
    finally:
        plt.close(fig)


def test_a_column_that_does_not_fit_grows_the_figure_down():
    """D8: on a 2-in text area the column needs more; the figure grows by the need rounded up to 0.1 in (never past
    the cap), every left axes keeps its distance from the top, and the whole column is drawn."""
    analysis, series = libration_case()
    rows = rows_of(analysis)
    column = vm.figure_column(analysis, rows, SETTINGS)
    needed = draw.measure_column(column, SHORT)
    assert needed > 2.0
    expected = min(COLUMN_MAX_GROWTH_IN, np.ceil(round((needed - 2.0) / 0.1, 6)) / 10)
    base = mc.draw_combined(analysis, series, layout=SHORT, rows=rows, options={'combined': {'numbers': 'all'}}, settings=SETTINGS)
    fig = mc.draw_combined(analysis, series, layout=SHORT, rows=rows, settings=SETTINGS)
    try:
        assert fig.text_stats['grown_in'] == pytest.approx(expected, abs=1e-6)
        assert fig.get_size_inches()[1] == pytest.approx(SHORT.size[1] + expected)
        assert fig.text_stats['overflow'] == 0 and fig.text_stats['not_drawn'] == 0 and fig.text_stats['trimmed'] == []
        assert _from_top(fig) == _from_top(base)  # the left panels, portraits, colour bar: same place from the top
    finally:
        plt.close(fig)
        plt.close(base)


def test_the_growth_is_capped():
    """Beyond `COLUMN_MAX_GROWTH_IN` the figure grows by the cap and the column trims (`verdict_column_draw.plan`)."""
    analysis, series = libration_case()
    tiny = replace(LAYOUT_V1, text_columns=tuple((x, 12.9, w, 0.1) for x, _, w, _ in LAYOUT_V1.text_columns))
    column = vm.figure_column(analysis, rows_of(analysis), SETTINGS)
    _, fit = draw.fit_layout(column, tiny, max_growth_in=1.0)
    assert fit.grown_in == pytest.approx(1.0) and fit.height_needed_in > 1.1
    assert fit.fit.trimmed or fit.overflow == 1


def test_the_zlk_panels_figure_draws_the_panels_only():
    """combined.figure='panels' on the secular/ZLK figure: no header, legend (nor its band) or column; the four squares
    (they hold data), the page cropped as the MMR one (12.45 x 12.8 in, `mc.panels_layout`)."""
    analysis = _zlk(2.5)
    full = sp.draw_long_period_combined(analysis, settings=SETTINGS)
    paper = sp.draw_long_period_combined(analysis, settings=SETTINGS, options={'combined': {'figure': 'panels'}})
    try:
        assert tuple(paper.get_size_inches()) == pytest.approx((12.45, 12.8)) and paper.text_stats == {}
        assert not [a for a in paper.findobj() if a.get_gid() == draw.GID] and _drawn(full)
        full_texts, paper_texts = [t.get_text() for t in full.texts], [t.get_text() for t in paper.texts]
        assert 'Legend' in full_texts and 'Legend' not in paper_texts
        assert not [t for t in paper_texts if t.startswith('synthetic')] and [t for t in full_texts if t.startswith('synthetic')]
        titles = [ax.get_title(loc='left') for ax in paper.axes if ax.get_title(loc='left').startswith('(')]
        assert (
            titles
            == [ax.get_title(loc='left') for ax in full.axes if ax.get_title(loc='left').startswith('(')]
            == ['(i)', '(j)', '(k)', '(l)']
        )
        box = paper.get_tightbbox(paper.canvas.get_renderer())
        assert box.x1 <= 12.45 and 12.8 - box.y1 < 0.25
    finally:
        plt.close(full)
        plt.close(paper)


def test_the_panels_figure_leaves_out_a_second_portrait_without_data():
    """No filtered angle and no free elements: the full figure keeps empty (j), (k), (l) frames, the paper figure only
    (i); a stamp still prints, in the same style as the MMR one (`mc.panels_stamp`)."""
    bare = replace(_zlk(2.5), angle=None)
    full = sp.draw_long_period_combined(bare, settings=SETTINGS)
    paper = sp.draw_long_period_combined(bare, settings=SETTINGS, options={'combined': {'figure': 'panels'}}, stamp='mismatch')
    try:
        titles = [[ax.get_title(loc='left') for ax in fig.axes if ax.get_title(loc='left').startswith('(')] for fig in (full, paper)]
        assert titles == [['(i)', '(j)', '(k)', '(l)'], ['(i)']]
        (stamp,) = [t for t in paper.texts if t.get_text() == 'mismatch']
        assert stamp.get_color() == '#b2182b' and stamp.get_fontweight() == 'bold'
    finally:
        plt.close(full)
        plt.close(paper)
