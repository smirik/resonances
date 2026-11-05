"""W29: the action in the right column's plaque (the author on q05, 2026-11-02: TX1's 'extend' was missed in the
sentences, so it goes into the header) and the minimal policy's extension words (S1: 469333 7N-17 advised 'extend
backward' at action spot_check; the text must read the action, not the edge flags).

Expected strings are worked out by hand: a time in its unit with 3 significant digits (40000 yr = 40 kyr, 2e5 yr =
200 kyr), the causes' words from `EXTEND_SOURCE_WORDS`. The rows copy the benchmark's: 469333 7N-17 (minimal run
2026-11-02: action spot_check, extend_to_years 200000, direction backward, geo_extend_time_from start:half) and 2018 TX1
3M-5S-2 (action extend, extend_to_years 40000)."""

from __future__ import annotations

import math
from dataclasses import replace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from resonances.plotting import combined_numbers as cn  # noqa: E402
from resonances.plotting import secular_plots as sp  # noqa: E402
from resonances.plotting import verdict_column as vc  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.plotting.style import LAYOUT_V1  # noqa: E402
from resonances.resonance.analysis.verdict import POLICY_MINIMAL, Action  # noqa: E402
from tests.resonances.plotting.test_verdict_text_roles import _kinematic  # noqa: E402
from tests.resonances.plotting.test_verdict_text_w28c import S, _facts  # noqa: E402
from tests.resonances.plotting.test_w29_figures import _zlk  # noqa: E402

NAN = float('nan')


# --------------------------------------------------------------------------------------------- the action tag


@pytest.mark.parametrize(
    'action, to, step, expected',
    [
        ('extend', 40000.0, NAN, '→ extend to 40 kyr'),  # 2018 TX1 3M-5S-2
        ('extend', 2.0e5, NAN, '→ extend to 200 kyr'),
        ('extend', 2500.0, NAN, '→ extend to 2.5 kyr'),
        ('extend', NAN, NAN, '→ extend'),
        ('resample', NAN, 0.5, '→ resample to 0.5 yr'),
        ('resample', NAN, NAN, '→ resample'),
        ('review', NAN, NAN, '→ review'),
        ('spot_check', 2.0e5, NAN, '→ spot check'),  # 469333: extend_to_years set, the action is not extend
        ('none', 2.0e5, NAN, ''),
    ],
)
def test_the_action_tag_reads_the_action_only(action, to, step, expected):
    v = _facts(verdict={'action': Action(action), 'extend_to_years': to, 'resample_step_yr': step}).v
    assert vt.action_tag(v) == expected


def test_the_explanation_carries_the_action_tag():
    base, columns = _kinematic()
    for action, to, expected in (('extend', 40000.0, '→ extend to 40 kyr'), ('none', NAN, '')):
        a = replace(base, verdict=replace(base.verdict, action=Action(action), extend_to_years=to))
        text = vt.explain_mmr(a, S, columns, a.episode_rows())
        assert text.action == expected
        assert text.badge == vt.VerdictText(text.status, text.angle, text.confidence, '', [], [], [], '').badge  # unchanged


# --------------------------------------------------------------------------------------------- the plaque


def _pen():
    fig = plt.figure(figsize=(24, 13.5))
    return fig, cn._Pen(fig)


def _text(action='', final=''):
    return vt.VerdictText('resonant', 'libration (regime_change)', 'medium', '', [], [], [], '', final_regime=final, action=action)


def test_the_action_tag_stands_right_of_the_plaque_before_ends():
    fig, pen = _pen()
    try:
        x0, x1, _, _ = cn._box(LAYOUT_V1)
        text = _text('→ extend to 40 kyr', 'circulation')
        tags = cn.plaque_tags(pen, text, x0, x1)
        assert [(kind, label, row) for kind, label, _, row in tags] == [
            ('action', ' → extend to 40 kyr ', 0),
            ('ends', ' ends: circulation ', 0),
        ]
        plaque_end = x0 + pen.text_width(f' {text.badge} ', cn.BADGE_PT, weight='bold')
        assert plaque_end < tags[0][2] < tags[1][2] < x1
        assert cn.plaque_rows(pen, text, x0, x1) == 1
        # without an action the tag 'ends' stands where it stood before W29
        ends = cn.plaque_tags(pen, _text('', 'circulation'), x0, x1)
        pad_badge, pad_tag = cn.BADGE_PAD * cn.BADGE_PT / 72, cn.BADGE_PAD * cn.ENDS_PT / 72
        assert [kind for kind, *_ in ends] == ['ends']
        assert ends[0][2] == pytest.approx(plaque_end + pad_badge + cn.ENDS_GAP + pad_tag)
        assert cn.plaque_tags(pen, _text(), x0, x1) == []
    finally:
        plt.close(fig)


def test_a_tag_past_the_right_edge_goes_one_row_down():
    fig, pen = _pen()
    try:
        x0 = 12.45
        text = _text('→ extend to 40 kyr', 'circulation')
        plaque_end = x0 + pen.text_width(f' {text.badge} ', cn.BADGE_PT, weight='bold')
        right = plaque_end + 0.5  # room for neither tag beside the plaque
        tags = cn.plaque_tags(pen, text, x0, right)
        # the action starts the second row at the plaque's left; 'ends' follows it there (the row has room)
        assert [(kind, row) for kind, _, _, row in tags] == [('action', 1), ('ends', 1)]
        assert tags[0][2] == x0 < tags[1][2] < right
        assert cn.plaque_rows(pen, text, x0, right) == 2
        # a column narrower than the action and 'ends' together: one row each
        action_width = pen.text_width(' → extend to 40 kyr ', cn.ENDS_PT, weight='bold')
        tags = cn.plaque_tags(pen, text, x0, x0 + action_width + 0.3)
        assert [(kind, row) for kind, _, _, row in tags] == [('action', 1), ('ends', 2)]
        assert tags[0][2] == x0 and tags[1][2] == x0
        assert cn.plaque_rows(pen, text, x0, x0 + action_width + 0.3) == 3
        tags = cn.plaque_tags(pen, text, x0, x0 + 30.0)
        assert [row for *_, row in tags] == [0, 0]
    finally:
        plt.close(fig)


def test_a_wrapped_tag_moves_the_explanation_down_by_one_row():
    fig, pen = _pen()
    try:
        short = _text('→ extend to 40 kyr')
        long = _text('→ ' + 'extend ' * 40)  # wider than the column: under the plaque
        y_short = cn._explanation(pen, short, LAYOUT_V1, draw=False)
        y_long = cn._explanation(pen, long, LAYOUT_V1, draw=False)
        assert y_short - y_long == pytest.approx(cn.TAG_ROW)
        assert y_short == pytest.approx(cn._explanation(pen, _text(), LAYOUT_V1, draw=False))
    finally:
        plt.close(fig)


def test_the_long_period_figure_draws_the_action_in_its_plaque():
    """The secular/ZLK figure draws the action as the header's one chip (`verdict_column`, 2026-11-04): 'extend to
    200 kyr' (2.0e5 yr), right of the status word on the plaque's line, and the column fits without growing."""
    analysis = _zlk(2.5)
    analysis = replace(analysis, verdict=replace(analysis.verdict, action=Action.EXTEND, extend_to_years=2.0e5))
    fig = sp.draw_long_period_combined(analysis)
    try:
        texts = {t.get_text(): t for t in fig.texts if t.get_gid() == 'verdict_column'}
        column = fig.text_stats['column']
        assert column.header.chip.kind == 'extend' and column.header.chip.text == 'extend to 200 kyr'
        assert vc.problems(column) == []  # the action once: the chip, never repeated in the text
        tag, badge = texts['extend to 200 kyr'], texts[column.header.word]
        assert tag.get_position()[0] > badge.get_position()[0]
        assert tag.get_position()[1] == pytest.approx(badge.get_position()[1], abs=0.1)  # one line (inches)
        assert fig.text_stats['overflow'] == 0 and fig.text_stats['grown_in'] == 0
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------------------------- 469333: the words


def _minimal(row, **verdict):
    return _facts(row=row, verdict={'policy': POLICY_MINIMAL, **verdict})


def test_469333_minimal_advises_no_extension_its_action_lacks():
    f = _minimal({'geo_extend_time_from': 'start:half'}, action=Action.SPOT_CHECK, extend_to_years=2.0e5, extend_direction='backward')
    sentence = vt.extend_flag_sentence(f)
    assert sentence == (
        'Flag extend_time (a half-swing that follows the law is cut by the start): no extension; the action extends only '
        'an unfinished libration at the record’s end or a deciding period too long for the record.'
    )
    assert 'backward' not in sentence and 'Extend:' not in sentence and 'next check' not in sentence
    assert vt.action_sentence(f.v) == 'Action: glance at the figure.'


def test_minimal_extend_advises_forward_for_the_end_causes_only():
    f = _minimal(
        {'geo_extend_time_from': 'end:slowdown;start:reflection'},
        action=Action.EXTEND,
        extend_to_years=2000.0,
        extend_direction='both',
        final_regime='arc',
    )
    assert vt.extend_flag_sentence(f) == (
        'Extend: a longer integration forward would show how the record ends (σ_f slows to an arc after circulating). '
        'Flag extend_time (σ turns back once at the start): no extension; the action extends only an unfinished '
        'libration at the record’s end or a deciding period too long for the record.'
    )
    # an extend row whose flag has only a start cause: nothing is advised for the start (the action sentence speaks)
    f = _minimal({'geo_extend_time_from': 'start:reflection'}, action=Action.EXTEND, extend_to_years=2000.0, extend_direction='backward')
    assert not vt.extend_flag_sentence(f).startswith('Extend:')
    # an end cause at an action that is not extend: named, not advised
    f = _minimal({'geo_extend_time_from': 'half:end'}, action=Action.REVIEW, extend_to_years=1e5, extend_direction='forward')
    assert vt.extend_flag_sentence(f).startswith('Flag extend_time (a half-swing that follows the law is cut by the end): no extension')
    assert vt.extend_flag_sentence(_minimal({}, action=Action.SPOT_CHECK)) == ''


def test_minimal_baseline_names_a_next_check_only_with_action_extend():
    """baseline_needed: P = 12.5 kyr asks for about 50 kyr (base_required_kyr); the next check 40 kyr is the row's
    extend_to_years and is named only when the action is extend."""
    row = {'base_required_kyr': 50.0, 'base_period_yr': 12500.0, 'base_period_source': '', 'base_cause': '', 'extend_to_years': 40000.0}
    review = _minimal(row, status='uncertain', reason='baseline_needed', action=Action.REVIEW, extend_to_years=40000.0)
    assert vt._baseline_sentence(review) == 'Uncertain: the record is too short to decide: about 50 kyr needed (P = 12.5 kyr).'
    extend = _minimal(row, status='uncertain', reason='baseline_needed', action=Action.EXTEND, extend_to_years=40000.0)
    assert vt._baseline_sentence(extend) == (
        'Uncertain: the record is too short to decide; next check: 40 kyr (about 50 kyr estimated from P = 12.5 kyr).'
    )
    # a file written before W29 (no policy) names the next check whatever the action
    old = _facts(row=row, verdict={'status': 'uncertain', 'reason': 'baseline_needed', 'action': Action.REVIEW, 'policy': ''})
    assert 'next check: 40 kyr' in vt._baseline_sentence(old)
    assert math.isfinite(row['extend_to_years'])
