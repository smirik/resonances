"""W28 C13: what C7 and C8 store about a 1:1 episode, drawn on the combined figure (cards g03-1 A, g03-3 C, g03-8 A,
g03-9 A): the inferred residence hatched and lighter beyond the episode's fill, labeled 'inferred'; one bracket per
sub-regime over (a), named by its well; the kicks of a as ▼ on (d); the header 'orbit: A → B'; labels that would
overlap merged; nothing of it on (l).

The truths are worked out by hand:
- the fill of a run is its first to last turning point plus P/4 on each uncovered side, halfway to a neighbor at most
  (`episode_spans`): run A 1000-2000 yr, P 800 → 800-2200 yr; run B 3000-3500 yr, P 400 → 2900-3600 yr, its start
  clipped at max(2900, (2000 + 3000) / 2 = 2500); so A's residence 500-4000 yr leaves 500-800 and 2200-2900 yr
  (not into B's fill), B's residence 3000-4800 yr leaves 3600-4800 yr;
- labels 10 units per character: '#0 horseshoe' (12 characters, 120) centered at 50 reaches 110, 'tadpole L4' (10, 100)
  centered at 120 starts at 70: one label '#0 horseshoe → tadpole L4' over 0-140 (25 characters, reaching 195), clear
  of 'inferred' (80) centered at 400 (from 360);
- the synthetic figure: the 2:1 kinematic record of test_verdict_text_roles (σ librates through ϖ, P 400 yr, TPs every
  200 yr from 100.5 to 2900.5 yr over 0-3000 yr), its run cut to its first 9 TPs (100.5-1700.5 yr) with its exit seen:
  the fill 0-1800.5 yr (start covered, the end 1700.5 + 400/4), the residence to 3000 yr hatched over 1800.5-3000 yr;
  sub-regimes horseshoe 100.5-700.5, tadpole L4 700.5-1100.5 (well change), tadpole L4 1300.5-1700.5 after the
  amplitude change 1100.5-1300.5 yr: three brackets, named '#0 horseshoe', 'tadpole L4', 'new amplitude'; one kick at
  1150 yr.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.resonance.analysis.geometry import (  # noqa: E402
    MARK_AMPLITUDE_CHANGE,
    MARK_WELL_CHANGE,
    RESIDENCE_EDGE,
    Residence,
    SubRegime,
)
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402

S = settings_from_config()
SETTINGS = SimpleNamespace(libration_coverage_periods=0.7)
_CACHE = {}


def teardown_function():
    plt.close('all')


def _run(episode_id, t0, t1, period, residence=None, subregimes=(), marks=''):
    big = 1e9
    return SimpleNamespace(
        episode_id=episode_id, accepted=True, t_start_yr=t0, t_end_yr=t1, period_med_yr=period,
        coverage_start_gap_p=5.0, coverage_end_gap_p=5.0, window_start_gap_yr=big, window_end_gap_yr=big,
        tp_step_max_yr=period / 2, window_start_hidden_p=big, window_end_hidden_p=big, before_last_jump=False,
        residence=residence, subregimes=tuple(subregimes), marks=marks,
    )  # fmt: skip


def _residence(t0, t1):
    return Residence(t0_yr=t0, t1_yr=t1, bounded_by_start=RESIDENCE_EDGE, bounded_by_end=RESIDENCE_EDGE)


# --------------------------------------------------------------------------------------------- pure helpers


def test_residence_pieces_lie_beyond_the_fill_and_stop_at_a_neighbour():
    a = _run(0, 1000.0, 2000.0, 800.0, residence=_residence(500.0, 4000.0))
    b = _run(1, 3000.0, 3500.0, 400.0, residence=_residence(3000.0, 4800.0))
    spans = mc.episode_spans([a, b], (0.0, 5000.0), SETTINGS)
    assert spans == {0: (pytest.approx(800.0), pytest.approx(2200.0)), 1: (pytest.approx(2900.0), pytest.approx(3600.0))}
    pieces = mc.residence_pieces([a, b], spans, (0.0, 5000.0))
    assert pieces[0] == [(500.0, pytest.approx(800.0)), (pytest.approx(2200.0), pytest.approx(2900.0))]
    assert pieces[1] == [(pytest.approx(3600.0), 4800.0)]
    assert mc.residence_pieces([a, b], spans, (0.0, 5000.0), skip={0, 1}) == {}  # not counted: no residence drawn
    # a residence inside the fill and none at all draw nothing; the fill itself never moves (nor does (l), which reads it)
    inside = _run(0, 1000.0, 2000.0, 800.0, residence=_residence(1000.0, 2000.0))
    assert mc.residence_pieces([inside], mc.episode_spans([inside], (0.0, 5000.0), SETTINGS), (0.0, 5000.0)) == {}
    assert mc.residence_pieces([_run(0, 1000.0, 2000.0, 800.0)], spans, (0.0, 5000.0)) == {}
    assert mc.episode_spans([inside], (0.0, 5000.0), SETTINGS) == mc.episode_spans([a], (0.0, 5000.0), SETTINGS)


def test_overlapping_labels_merge_with_an_arrow_between_wells():
    def width(text):
        return 10.0 * len(text)

    items = [(0.0, 100.0, '#0 horseshoe', True), (100.0, 140.0, 'tadpole L4', True), (300.0, 500.0, 'inferred', False)]
    assert mc.merge_label_items(items, width) == [(0.0, 140.0, '#0 horseshoe → tadpole L4', True), items[2]]
    # a well's name that meets 'inferred' joins with a dot, not an arrow
    close = [(0.0, 100.0, 'tadpole L4', True), (100.0, 160.0, 'inferred', False)]
    assert mc.merge_label_items(close, width) == [(0.0, 160.0, 'tadpole L4 · inferred', False)]


def test_label_items_name_each_well_and_a_new_amplitude():
    subs = (
        SubRegime('horseshoe', 100.5, 700.5),
        SubRegime('tadpole_L4', 700.5, 1100.5, MARK_WELL_CHANGE),
        SubRegime('tadpole_L4', 1300.5, 1700.5, MARK_AMPLITUDE_CHANGE),
    )
    run = _run(0, 100.5, 1700.5, 400.0, subregimes=subs)
    items = mc.run_label_items(run, (0.0, 1800.5), [(1800.5, 3000.0)], suffix='')
    assert [text for *_, text, _ in items] == ['#0 horseshoe', 'tadpole L4', 'new amplitude', 'inferred']
    assert items[0][:2] == (100.5, 700.5) and items[-1][:2] == (1800.5, 3000.0)
    single = _run(3, 100.5, 1700.5, 400.0)
    assert [text for *_, text, _ in mc.run_label_items(single, (0.0, 1800.5), [(1800.5, 3000.0)], suffix=' wave')] == [
        '#3 wave',
        'inferred',
    ]


def test_kick_times_come_from_the_marks_of_accepted_runs():
    runs = [
        _run(0, 0.0, 1.0, 1.0, marks='well_change@4654.2;kick@4687.9;well_change@5118.5'),
        SimpleNamespace(accepted=False, marks='kick@10.0'),  # a rejected run's marks are not read
        _run(2, 0.0, 1.0, 1.0, marks='amplitude_change@326.5:446.0;kick@348.0'),
    ]
    assert mc.kick_times(runs) == [348.0, 4687.9]


def test_header_names_the_sequence_of_wells_as_the_orbit():
    analysis = SimpleNamespace(verdict=SimpleNamespace(orbit_type='compound: horseshoe → tadpole L4'), response=None)
    assert mc.orbit_type_text(analysis) == 'orbit: horseshoe → tadpole L4'
    one = SimpleNamespace(verdict=SimpleNamespace(orbit_type='compound (horseshoe ↔ quasi-satellite)'), response=None)
    assert mc.orbit_type_text(one) == 'compound (horseshoe ↔ quasi-satellite)'


# --------------------------------------------------------------------------------------------- the whole figure


def _synthetic():
    """(analysis, series) of the 2:1 kinematic synthetic (test_verdict_text_roles), built once."""
    if 'built' not in _CACHE:
        from scipy.optimize import brentq

        from resonances.data import const
        from resonances.resonance.analysis.mmr import context as ctx
        from resonances.resonance.analysis.mmr.analyse import analyse_mmr
        from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec

        spec = ResonanceSpec.from_key('2J-1+0-1')
        t = np.arange(0.0, 3000.0 + 0.25, 0.5)
        n_jupiter = const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'] * math.pi / 180
        a_au = brentq(lambda a: float(ctx.mean_motion(a)) - 2 * n_jupiter, 2.0, 5.0)
        lam = (2 * n_jupiter + 2 * math.pi / 60000.0) * t + 0.3
        varpi = 0.4 + 0.9 * np.sin(2 * math.pi * t / 400.0)
        sigma = 2 * (n_jupiter * t + 1.1) - lam - varpi
        ones = np.ones_like(t)
        a = a_au + 1e-6 * np.sin(2 * math.pi * t / 7.3)
        series = MMRSeries(
            t, a, 0.1 * ones, 0.05 * ones, 0.7 * ones, np.mod(varpi - 0.7, 2 * math.pi), np.mod(varpi, 2 * math.pi),
            np.mod(sigma, 2 * math.pi), spec, Regime(0.5, float(t[-1]), len(t)), {}, np.mod(lam, 2 * math.pi), name='c13',
        )  # fmt: skip
        _CACHE['built'] = (analyse_mmr(series, S, scan_owner=False, scan_subresonances=False), series)
    return _CACHE['built']


def _counted(analysis):
    """The synthetic with its run 0 counted by the minimal policy: the parts of (a) are drawn for a counted run only
    (`_shade_policy_episodes`, W28 C13); the synthetic itself is vetoed kinematic, its run 0 a candidate."""
    return replace(analysis, policy=replace(analysis.policy, counted_episode_ids=frozenset({0}), candidate_episode_ids=frozenset()))


def _with_parts():
    """The synthetic with its accepted run cut to 9 TPs, its exit seen, a residence to the end, three sub-regimes and a
    kick (module doc), its orbit type a sequence of wells."""
    analysis, series = _synthetic()
    (run,) = [run for run in analysis.geometry.runs if run.accepted]
    assert run.n_tp == 15 and run.t_start_yr == pytest.approx(100.5) and run.period_med_yr == pytest.approx(400.0)
    subs = (
        SubRegime('horseshoe', 100.5, 700.5),
        SubRegime('tadpole_L4', 700.5, 1100.5, MARK_WELL_CHANGE),
        SubRegime('tadpole_L4', 1300.5, 1700.5, MARK_AMPLITUDE_CHANGE),
    )
    cut = replace(
        run,
        tp_times_yr=run.tp_times_yr[:9],
        tp_values_rad=run.tp_values_rad[:9],
        exit_observed=True,
        residence=_residence(100.5, 3000.0),
        subregimes=subs,
        marks='well_change@700.5;amplitude_change@1100.5:1300.5;kick@1150.0',
    )
    runs = [cut if r is run else r for r in analysis.geometry.runs]
    geometry = replace(analysis.geometry, runs=runs)
    verdict = replace(analysis.verdict, orbit_type='compound: horseshoe → tadpole L4')
    return _counted(replace(analysis, geometry=geometry, verdict=verdict)), series, cut


def test_the_figure_draws_residence_brackets_kicks_and_the_orbit_sequence():
    analysis, series, run = _with_parts()
    assert run.t_end_yr == pytest.approx(1700.5)
    fig = mc.draw_combined(analysis, series, settings=S)
    axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    angle, axis_panel = axes[0], axes[3]
    hatched = [p for p in angle.patches if p.get_gid() == mc.RESIDENCE_GID]
    assert len(hatched) == 2 and {p.get_hatch() for p in hatched} == {None, mc.RESIDENCE_HATCH}
    for patch in hatched:  # beyond the fill only: 1700.5 + 400/4 to the record's end
        assert patch.get_x() == pytest.approx(1800.5) and patch.get_x() + patch.get_width() == pytest.approx(3000.0)
    brackets = [line for line in angle.lines if line.get_gid() == mc.BRACKET_GID]
    assert len(brackets) == len(run.subregimes) == 3
    names = [text.get_text() for text in angle.texts]
    assert {'#0 horseshoe', 'tadpole L4', 'new amplitude', 'inferred'} <= set(names) and '#0' not in names
    (kick,) = [line for line in axis_panel.lines if line.get_gid() == mc.KICK_GID]
    assert list(kick.get_xdata()) == [pytest.approx(1150.0)] and ' kick' in [text.get_text() for text in axis_panel.texts]
    title = [text.get_text() for text in fig.texts if 'orbit:' in text.get_text()]
    assert title and title[0].endswith('orbit: horseshoe → tadpole L4')
    assert fig.text_stats['overflow'] == 0


def test_a_run_without_parts_keeps_its_plain_label():
    """Without a residence beyond the fill or sub-regimes: '#0' as before, nothing hatched, no bracket, no kick."""
    analysis, series = _synthetic()
    fig = mc.draw_combined(_counted(analysis), series, settings=S)
    angle, axis_panel = fig.axes[0], fig.axes[3]
    assert '#0' in [text.get_text() for text in angle.texts]
    assert not [p for p in angle.patches if p.get_gid() == mc.RESIDENCE_GID]
    assert not [line for line in angle.lines if line.get_gid() == mc.BRACKET_GID]
    assert not [line for line in axis_panel.lines if line.get_gid() == mc.KICK_GID]
    # W34: as analysed, the minimal policy vetoes the kinematic synthetic (veto kinematic, 0 counted): (a) draws its run
    # faint and dashed as a candidate, 'cand' and not numbered (status-tree.md#minimal-policy, step 6)
    assert analysis.verdict.veto == 'kinematic' and analysis.policy.candidate_episode_ids == {0}
    names = [text.get_text() for text in mc.draw_combined(analysis, series, settings=S).axes[0].texts]
    assert mc.CANDIDATE_LABEL in names and '#0' not in names


def test_names_wider_than_the_fill_leave_the_brackets_unnamed():
    """524522 1V-1 20 kyr in miniature: 16 sub-regimes of 100 yr alternating quasi-satellite and tadpole L5 over
    100.5-1700.5 yr; each bracket is narrower than its name, so all merge, and the merged sequence of 16 names is wider
    than the fill (0-1800.5 yr): '#0' stands alone over the fill, the 16 brackets stay."""
    analysis, series, run = _with_parts()
    wells = ('qs', 'tadpole_L5')
    subs = tuple(SubRegime(wells[k % 2], 100.5 + 100 * k, 200.5 + 100 * k, MARK_WELL_CHANGE if k else 'start') for k in range(16))
    many = replace(run, subregimes=subs)
    runs = [many if r is run else r for r in analysis.geometry.runs]
    fig = mc.draw_combined(replace(analysis, geometry=replace(analysis.geometry, runs=runs)), series, settings=S)
    angle = fig.axes[0]
    assert len([line for line in angle.lines if line.get_gid() == mc.BRACKET_GID]) == 16
    names = [text.get_text() for text in angle.texts]
    assert '#0' in names and not [name for name in names if 'quasi-satellite' in name]
    assert fig.text_stats['overflow'] == 0


# --------------------------------------------------------------------------------------------- the 'panels' figure
# combined.figure='panels' (for a paper): the panels with their margin legends only. Its size, by hand from LAYOUT_V1:
# width = separator_x 12.3 + 0.15 = 12.45 in; height = the top of the time panels 3.6 + 8.95 = 12.55, + 0.25 = 12.8 in.

PANELS = {'combined': {'figure': 'panels'}}


def test_the_figure_option_defaults_to_the_full_page():
    from resonances.plotting import style

    assert style.resolve_plot_options()['combined']['figure'] == style.FigureMode.FULL
    assert style.resolve_plot_options(PANELS)['combined'] == {'osculating_a': 'full', 'numbers': 'explained', 'figure': 'panels'}
    with pytest.raises(ValueError, match='combined.figure'):
        style.resolve_plot_options({'combined': {'figure': 'paper'}})


def _margin_texts(fig):
    """The figure texts between the margin legends' left edge and the separator (the right column starts beyond)."""
    width = fig.get_size_inches()[0]
    return sorted(t.get_text() for t in fig.texts if mc.LAYOUT_V1.margin_legend_x <= t.get_position()[0] * width < mc.LAYOUT_V1.separator_x)


def _bottom_titles(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts if t.get_text().startswith('(') and ') ' in t.get_text()]


def test_the_panels_figure_draws_no_header_legend_or_column():
    analysis, series = _synthetic()
    full = mc.draw_combined(analysis, series, settings=S)
    paper = mc.draw_combined(analysis, series, settings=S, options=PANELS)
    assert tuple(full.get_size_inches()) == mc.LAYOUT_V1.size and tuple(paper.get_size_inches()) == pytest.approx((12.45, 12.8))
    full_texts, paper_texts = [t.get_text() for t in full.texts], [t.get_text() for t in paper.texts]
    assert [a for a in full.findobj() if a.get_gid() == 'verdict_column']
    assert not [a for a in paper.findobj() if a.get_gid() == 'verdict_column'] and paper.text_stats == {}
    assert 'Legend' in full_texts and 'Legend' not in paper_texts
    header = [text for text in full_texts if text.startswith('c13   2J-1+0-1')]
    record = mc.record_line(analysis, series)
    assert header and record in full_texts and not set(header + [record]) & set(paper_texts)
    assert [t for t in full_texts if t.startswith(mc.FAIR_WARNING_PREFIX)]  # no FAIR here: no warning in the paper figure
    assert not [t for t in paper_texts if t.startswith(mc.FAIR_WARNING_PREFIX)]
    margin = _margin_texts(paper)  # the margin legends of (c), (d), (h) stay, the same as in the full figure
    assert margin == _margin_texts(full) and {'h = 1 separatrix', 'σ̇ = 0'} <= set(margin)
    assert _bottom_titles(paper) == _bottom_titles(full) == ['(k) phase portrait, whole record', '(l) episodes, counted solid']
    # cropped to what is drawn: nothing beyond the page, no empty band over (a) (the labels over it reach 0.11 in)
    box = paper.get_tightbbox(paper.canvas.get_renderer())
    assert box.x1 <= 12.45 and 12.45 - box.x1 < 0.3 and 12.8 - box.y1 < 0.25


def test_the_panels_figure_leaves_out_an_empty_episode_portrait_and_reletters_fair():
    """No accepted episode: the full figure prints '(l) … no accepted episode'; the paper figure has no such axes, and
    FAIR (drawn once the series has the planet's longitudes) moves into the second slot as (l)."""
    from resonances.data import const
    from resonances.resonance.analysis.mmr.models import PlanetSeries

    analysis, series = _synthetic()
    geometry = replace(analysis.geometry, runs=[replace(run, accepted=False) for run in analysis.geometry.runs])
    bare = replace(analysis, geometry=geometry)
    full = mc.draw_combined(bare, series, settings=S)
    assert 'no accepted episode' in [t.get_text() for ax in full.axes for t in ax.texts]
    paper = mc.draw_combined(bare, series, settings=S, options=PANELS)
    assert 'no accepted episode' not in [t.get_text() for ax in paper.axes for t in ax.texts]
    assert _bottom_titles(paper) == ['(k) phase portrait, whole record']
    t = series.t_yr
    n_jupiter = const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'] * math.pi / 180
    jupiter = PlanetSeries('Jupiter', t, np.full(len(t), 5.2), *(np.zeros(len(t)),) * 4, np.mod(n_jupiter * t + 1.1, 2 * math.pi))
    with_fair = replace(series, planets={'Jupiter': jupiter}, mean_anomaly=np.mod(series.mean_longitude - series.varpi, 2 * math.pi))
    assert mc.fair_warning(bare, with_fair) is None
    paper = mc.draw_combined(bare, with_fair, settings=S, options=PANELS)
    assert _bottom_titles(paper) == ['(k) phase portrait, whole record', '(l) FAIR']
    (fair,) = [ax for ax in paper.axes if '(l) FAIR' in [t.get_text() for t in ax.texts]]
    assert fair.get_position().x0 * paper.get_size_inches()[0] == pytest.approx(mc.LAYOUT_V1.bottom[1][0])
    full = mc.draw_combined(bare, with_fair, settings=S)
    assert _bottom_titles(full)[-1] == '(m) FAIR'  # the full figure keeps its letters and slots
