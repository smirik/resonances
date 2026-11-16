"""W29 figures (SPEC §7, §9): the a panel (d) of the secular/ZLK figure with the planet's a and the co-orbital zone, the
episode rule of panel (a) under the minimal policy (counted episodes filled, candidates faint and dashed, nothing else),
and the right column of a minimal-policy verdict."""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting import secular_plots as sp  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.plotting.style import EPISODE_ALPHA  # noqa: E402
from resonances.resonance.analysis.secular import LongPeriodSeries  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import ResonanceType  # noqa: E402
from resonances.resonance.analysis.zlk import analyse_zlk  # noqa: E402

SETTINGS = settings_from_config()
GATE_OFF = SimpleNamespace(free_gate_enabled=False, free_gate_min_cycles=3.0)
N = 5001
T_END = 1e5

from resonances.resonance.analysis.policy import PolicyResult  # noqa: E402


def _policy(counted=(), candidates=(), counted_spans=(), candidate_spans=(), policy='minimal'):
    return PolicyResult(policy, frozenset(counted), frozenset(candidates), tuple(counted_spans), tuple(candidate_spans))


def _zlk(a_au: float):
    """A ZLK ω librating over the whole record (10 cycles of 10 kyr, 0.5 rad about 10°) at a constant a with a small
    wiggle; the gate disabled (the figure does not need it)."""
    t = np.linspace(0.0, T_END, N)
    angle = np.radians(10.0) + 0.5 * np.sin(2 * np.pi * t / 1e4)
    a = a_au * (1.0 + 1e-4 * np.sin(2 * np.pi * t / 3e3))
    series = LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, angle, a, np.full(N, 0.2), np.full(N, 0.6))
    return analyse_zlk(series, SETTINGS, gate_config=GATE_OFF)


def _time_panels(fig):
    first = fig.axes[0]
    return [ax for ax in fig.axes if ax is first or first.get_shared_x_axes().joined(first, ax)]


def _panel(fig, ylabel):
    return next(ax for ax in _time_panels(fig) if ax.get_ylabel() == ylabel)


def _texts(ax):
    return [text.get_text() for text in ax.texts]


# ------------------------------------------------------------------------------------------------- the a panel (d)


@pytest.mark.parametrize(
    'a_au, planet',
    [
        (1.02, 'Earth'),  # |1.02/1.00000018 - 1| = 0.020 < 3 (3.0035e-6 / 3)^(1/3) = 0.0300
        (1.04, ''),  # 0.040 > 0.0300
        (5.20, 'Jupiter'),  # a Trojan: 0.0005 < 3 (9.5479e-4 / 3)^(1/3) = 0.2048
        (4.00, ''),  # |4.0 / 5.2025 - 1| = 0.231 > 0.2048; Mars 1.524 and Saturn 9.54 far
        (2.50, ''),
        (float('nan'), ''),
    ],
)
def test_the_geometric_coorbital_planet(a_au, planet):
    assert sp.coorbital_planet_geometric(a_au, SETTINGS) == planet


def test_hill_half_width_by_hand():
    # Earth: 3 x (3.003489e-6 / 3)^(1/3) = 0.030012 (the 0.030 of omega_free_gate.coorbital_planet's docstring)
    assert sp.hill_half_width('Earth', 3.0) == pytest.approx(0.03001, abs=2e-5)
    assert sp.hill_half_width('Jupiter', 3.0) == pytest.approx(0.2048, abs=5e-4)


def test_the_long_period_figure_has_the_a_panel_before_e():
    fig = sp.draw_long_period_combined(_zlk(2.5))
    try:
        panels = _time_panels(fig)
        assert len(panels) == len(sp.LONG_PERIOD_PANELS) == 8
        letters = [next(t for t in _texts(ax) if t.startswith('(')) for ax in panels]
        assert letters == ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        assert [ax.get_ylabel() for ax in panels[3:6]] == ['a, au', 'e', 'i, °']
        assert [panel[0] for panel in sp.LONG_PERIOD_PANELS] == list('abcdefgh')
        # a far from every planet: no planet line
        assert not [line for line in panels[3].lines if line.get_gid() in ('planet_axis', 'hill_edge')]
        portraits = [ax.get_title(loc='left') for ax in fig.axes if ax.get_title(loc='left') in ('(i)', '(j)', '(k)', '(l)')]
        assert portraits == ['(i)', '(j)', '(k)', '(l)']
    finally:
        plt.close(fig)


@pytest.mark.parametrize('a_au, planet, a_p, width', [(1.0, 'Earth', 1.00000018, 0.030012), (5.2, 'Jupiter', 5.20248019, 1.0655)])
def test_the_planet_line_and_the_zone_edges(a_au, planet, a_p, width):
    """An Earth co-orbital and a Jupiter Trojan (whose modes are fitted: the gate's guard never names Jupiter) both get
    a_P dashed and a_P ± k r_H dotted on (d); r_H = a_P (m_P/3)^(1/3), k = 3 (Jupiter: 5.2025 x 0.2048 = 1.0655 au)."""
    fig = sp.draw_long_period_combined(_zlk(a_au))
    try:
        ax = _panel(fig, 'a, au')
        planet_lines = [line.get_ydata()[0] for line in ax.lines if line.get_gid() == 'planet_axis']
        edges = sorted(line.get_ydata()[0] for line in ax.lines if line.get_gid() == 'hill_edge')
        assert planet_lines == [pytest.approx(a_p)]
        assert edges == [pytest.approx(a_p - width, rel=1e-3), pytest.approx(a_p + width, rel=1e-3)]
        assert [line.get_linestyle() for line in ax.lines if line.get_gid() == 'planet_axis'] == ['--']
        assert any(text.startswith(f'{planet}: a_P') for text in _texts(ax))
    finally:
        plt.close(fig)


# ------------------------------------------------------------------------------- panel (a) under the minimal policy


def _extent(patch):
    """(x0, x1) of an axvspan (a Rectangle in data x)."""
    return patch.get_x(), patch.get_x() + patch.get_width()


def _fills(ax):
    """The axvspans of (a): (gid, facecolour alpha, 'dashed' | 'solid') per patch."""
    out = []
    for patch in ax.patches:
        face = patch.get_facecolor()
        out.append((patch.get_gid(), round(float(face[3]), 4), 'dashed' if patch.get_linestyle() in ('--', 'dashed') else 'solid'))
    return out


def test_under_v2_the_librating_episode_is_filled():
    analysis = _zlk(2.5)
    fig = sp.draw_long_period_combined(analysis)
    try:
        assert sp._librating(analysis)  # the episode the v2 drawing fills
        assert _fills(_panel(fig, 'σ, °'))
    finally:
        plt.close(fig)


def test_minimal_draws_no_run_that_is_neither_counted_nor_a_candidate():
    analysis = _zlk(2.5)
    analysis.policy = _policy()
    fig = sp.draw_long_period_combined(analysis)
    try:
        assert _fills(_panel(fig, 'σ, °')) == []
    finally:
        plt.close(fig)


def test_minimal_counted_episode_and_candidate():
    analysis = _zlk(2.5)
    run = sp._librating(analysis)[0]
    analysis.policy = _policy(counted=[run.episode_id])
    fig = sp.draw_long_period_combined(analysis)
    try:
        ax = _panel(fig, 'σ, °')
        fills = _fills(ax)
        assert fills and {gid for gid, _, _ in fills} <= {mc.COUNTED_GID, mc.TRIM_GID}
        assert (mc.COUNTED_GID, EPISODE_ALPHA, 'solid') in fills
        assert f'#{run.episode_id}' in _texts(ax)
    finally:
        plt.close(fig)
    analysis.policy = _policy(candidates=[run.episode_id])
    fig = sp.draw_long_period_combined(analysis)
    try:
        ax = _panel(fig, 'σ, °')
        fills = _fills(ax)
        assert {gid for gid, _, _ in fills} == {mc.CANDIDATE_GID}
        # faint: a quarter of an episode's alpha, and a dashed outline without a face
        assert (mc.CANDIDATE_GID, round(EPISODE_ALPHA * 0.25, 4), 'solid') in fills
        assert any(style == 'dashed' and alpha == 0 for gid, alpha, style in fills)
        assert mc.CANDIDATE_LABEL in _texts(ax) and f'#{run.episode_id}' not in _texts(ax)
    finally:
        plt.close(fig)


def test_minimal_leg_from_the_counted_spans_and_a_candidate_span():
    analysis = _zlk(2.5)
    analysis.policy = _policy(counted_spans=[(20e3, 40e3)], candidate_spans=[(60e3, 70e3)])
    fig = sp.draw_long_period_combined(analysis)
    try:
        ax = _panel(fig, 'σ, °')
        assert mc.LEG_LABEL in _texts(ax) and mc.CANDIDATE_LABEL in _texts(ax)
        counted = [p for p in ax.patches if p.get_gid() == mc.COUNTED_GID]
        assert len(counted) == 1
        divisor = sp._LongFrame(analysis).divisor
        assert _extent(counted[0]) == (pytest.approx(20e3 / divisor), pytest.approx(40e3 / divisor))
    finally:
        plt.close(fig)


def test_a_counted_episode_to_a_covered_edge_is_pale_in_the_filter_trim():
    """fill_counted: solid inside the analysis window, pale over the trimmed stretch to the record's edge."""
    fig, ax = plt.subplots()
    try:
        mc.fill_counted(ax, 0.0, 10.0, (1.0, 9.0), '#1b9e77')
        pieces = sorted((*(round(x, 6) for x in _extent(p)), p.get_gid(), p.get_alpha()) for p in ax.patches)
        assert pieces == [
            (0.0, 1.0, mc.TRIM_GID, EPISODE_ALPHA * mc.TRIM_ALPHA_SCALE),
            (1.0, 9.0, mc.COUNTED_GID, EPISODE_ALPHA),
            (9.0, 10.0, mc.TRIM_GID, EPISODE_ALPHA * mc.TRIM_ALPHA_SCALE),
        ]
    finally:
        plt.close(fig)


def test_policy_marks_from_a_stored_verdict_and_without_a_policy():
    """The figure re-run path (§7): without a PolicyResult the marks come from the verdict's id cells; 'leg' and 'half'
    have no episode row. An analysis the policy did not finish (`policy` '': an inner pass) cannot be drawn."""
    analysis = _zlk(2.5)
    analysis.policy = None
    analysis.verdict.policy = ''
    with pytest.raises(ValueError, match='minimal policy'):
        mc.policy_marks(analysis)
    analysis.verdict.policy = 'minimal'
    analysis.verdict.counted_episode_ids = '3;leg'
    analysis.verdict.candidate_episode_ids = '1;half;7'
    marks = mc.policy_marks(analysis)
    assert marks == mc.PolicyMarks(frozenset({3}), frozenset({1, 7}))
    assert vt.policy_ids('') == frozenset() and vt.policy_ids('2.0;nan;x') == frozenset({2})


def test_the_legend_of_a_follows_the_policy():
    assert sp.long_period_legend() == sp.LONG_PERIOD_LEGEND
    legend = sp.long_period_legend(forcing=True)
    assert legend[0] == sp.LONG_PERIOD_LEGEND[0] and legend[1] == sp.FORCING_LEGEND and 'cand' in legend[0]
    assert legend[2:] == sp.LONG_PERIOD_LEGEND[1:]
    assert 'cand' in mc.LEGEND[1] and 'other runs not drawn' in mc.LEGEND[1]


# ------------------------------------------------------------------------------------------------- the right column


def _minimal(verdict, **values):
    """The verdict as the policy layer leaves it (its fields set on the object: lane A's fields, before and after)."""
    for name, value in {'policy': 'minimal', 'caps': [], **values}.items():
        setattr(verdict, name, value)
    return verdict


def test_the_short_period_prose_survives_as_a_flag():
    """Under the minimal policy every cap is a flag (SPEC §2.3): the text that read the cap reads the flag (`has_mark`)."""
    analysis = _zlk(2.5)
    verdict = _minimal(analysis.verdict, flags=['short_period_aliased'])
    assert verdict.caps == [] and vt.has_mark(verdict, 'short_period_aliased')
    assert not vt.has_mark(verdict, 'near_separatrix_without_zone')


def test_the_figure_draws_under_minimal_with_the_block():
    analysis = _zlk(5.2)
    run = sp._librating(analysis)[0]
    analysis.policy = _policy(counted=[run.episode_id])
    _minimal(
        analysis.verdict, confidence_basis='N=10(cover);M=full;V=clean;S=full;veto=-', v2_verdict='resonant/high', n_counted_episodes=1
    )
    fig = sp.draw_long_period_combined(analysis)
    try:
        # the right column (`verdict_column`, 2026-11-04): the counted cycles are a Confidence pill; the W28 tree's
        # verdict is not printed (the author 2026-11-14); the column fits
        drawn = ' '.join(text.get_text() for text in fig.texts if text.get_gid() == 'verdict_column')
        drawn = drawn.replace('\xa0', ' ')  # a short footer item is glued with no-break spaces
        assert '10 cycles counted' in drawn and 'W28 tree' not in drawn
        from resonances.plotting.verdict_column import problems

        assert problems(fig.text_stats['column']) == []
        assert fig.text_stats['overflow'] == 0
    finally:
        plt.close(fig)


def test_mmr_panel_a_draws_only_counted_and_candidate_runs():
    """The runs of the v2 strip test (`test_rejected_runs_carry_the_letter_of_their_reason`: '#0', 'T', 'F', 'C'
    there): under the minimal policy, with #0 counted and #2 a candidate, (a) shows '#0' and 'cand', no gray strip."""
    from resonances.resonance.analysis.geometry import RejectReason

    from resonances.plotting.style import LAYOUT_V1

    runs = [
        SimpleNamespace(episode_id=0, t_start_yr=0.0, t_end_yr=4000.0, accepted=True, before_last_jump=False),
        SimpleNamespace(episode_id=1, t_start_yr=4000.0, t_end_yr=6000.0, accepted=False, reject_reason=RejectReason.OVER_THE_TOP),
        SimpleNamespace(episode_id=2, t_start_yr=6000.0, t_end_yr=8000.0, accepted=False, reject_reason=RejectReason.TOO_FEW_TP),
        SimpleNamespace(episode_id=3, t_start_yr=8000.0, t_end_yr=9990.0, accepted=False, reject_reason=RejectReason.CENTRE_MOVES),
    ]
    fig, ax = plt.subplots()
    try:
        px_per_kyr = LAYOUT_V1.panels[2] * LAYOUT_V1.dpi / 10.0
        marks = mc.PolicyMarks(frozenset({0}), frozenset({2}))
        spans = mc.shade_policy_runs(
            ax, runs, marks, {0: '#1b9e77'}, lambda v: np.asarray(v) / 1e3, px_per_kyr, LAYOUT_V1, (0.0, 1e4), (0.0, 1e4)
        )
        assert _texts(ax) == ['#0', mc.CANDIDATE_LABEL]
        assert spans == {0: (0.0, 4000.0)}  # no period known: to its last turn (P/4 = 0), as episode_spans fills
        assert {patch.get_gid() for patch in ax.patches} == {mc.COUNTED_GID, mc.CANDIDATE_GID}
        candidate = [patch for patch in ax.patches if patch.get_gid() == mc.CANDIDATE_GID]
        assert [_extent(patch) for patch in candidate] == [(6.0, 8.0)] * 2  # its turning points: fill and dashed outline
    finally:
        plt.close(fig)
