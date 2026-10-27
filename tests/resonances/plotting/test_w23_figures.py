"""W23 S7: the figure defects F2 (merged labels), F3 (the response variable of a secular angle), F4 (the colour bar's
label of the secular/ZLK figure) and F5 (one test for 'moving centre'), each on a case computed by hand or from the
record's own numbers."""

from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting.style import LAYOUT_V1  # noqa: E402
from resonances.resonance.analysis.geometry import Jump  # noqa: E402
from resonances.resonance.analysis.orbit_type import centre_moves  # noqa: E402
from resonances.resonance.analysis.secular import AngleTerm, LongPeriodSeries, analyse_secular  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import ResonanceType  # noqa: E402

S = settings_from_config()


# --------------------------------------------------------------------------------------------- F2: merged labels


@pytest.mark.parametrize(
    'spans, width, gap, expected',
    [
        # boxes [-0.75, 0.75] and [0.25, 1.75] overlap; the merged one (centre 0.5, right edge 1.25) is far from 10
        ([(0.0, 0.0), (1.0, 1.0), (10.0, 10.0)], lambda s: 1.5, 0.0, [(0.0, 1.0), (10.0, 10.0)]),
        # width 1 + span: 0.5 < 0.7 apart; 1.2 and 2.0 overlap (1.7 > 1.5); merged (1.2, 2.0): width 1.8, left edge 0.7
        ([(2.0, 2.0), (0.0, 0.0), (1.2, 1.2)], lambda s: 1.0 + s[1] - s[0], 0.0, [(0.0, 0.0), (1.2, 2.0)]),
        # the same with 0.3 of free space required: 0.5 + 0.3 > 0.7, so the cascade takes all three
        ([(0.0, 0.0), (1.2, 1.2), (2.0, 2.0)], lambda s: 1.0 + s[1] - s[0], 0.3, [(0.0, 2.0)]),
        ([(0.0, 1.0), (5.0, 6.0)], lambda s: 1.0, 0.0, [(0.0, 1.0), (5.0, 6.0)]),  # apart: unchanged
    ],
)
def test_overlapping_labels_merge_over_their_span(spans, width, gap, expected):
    assert mc.merge_overlapping_labels(spans, width, gap) == expected


def _frame(t_end: float, jumps=()):
    t = np.linspace(0.0, t_end, 1001)
    geometry = SimpleNamespace(jumps=[Jump(time, 0.0, 0.0, float('nan'), False, False) for time in jumps])  # jumps of a
    return SimpleNamespace(
        analysis=SimpleNamespace(geometry=geometry),
        t=t,
        px_per_unit=LAYOUT_V1.panels[2] * LAYOUT_V1.dpi / (t_end / 1000.0),
        times=lambda value: np.asarray(value, dtype=float) / 1000.0,
    )


def _boxes(fig, texts):
    renderer = fig.canvas.get_renderer()
    return [text.get_window_extent(renderer) for text in texts]


def _overlap(a, b) -> bool:
    return min(a.x1, b.x1) > max(a.x0, b.x0) and min(a.y1, b.y1) > max(a.y0, b.y0)


def test_jump_labels_that_would_collide_become_one_label():
    """Bands of 3, 2 and 2 jumps 1.2-1.4 % of a 100 kyr record apart (each band: jumps within 1 %), a single jump
    between the last two and one band far away: the three labels, ~6 % of the panel wide each, would overlap (TG2 4E-1M-3 100
    kyr: '3 jum2 ju2 jumps of a'); they read '8 jumps of a' (3 + 2 + 2 and the single), the far band '2 jumps of a'."""
    jumps = [60000, 60300, 60600, 62000, 62500, 63800, 65000, 65400, 90000, 90500]
    frame = _frame(100000.0, jumps)
    fig = plt.figure(figsize=LAYOUT_V1.size, dpi=LAYOUT_V1.dpi)
    ax = fig.add_axes((0.05, 0.5, LAYOUT_V1.panels[2] / LAYOUT_V1.size[0], 0.1))
    ax.set_xlim(0, 100)
    assert [count for _, _, count in mc.jump_groups(jumps, 100000.0)] == [3, 2, 1, 2, 2]
    mc._draw_jumps(ax, frame, LAYOUT_V1)
    labels = [text for text in ax.texts if text.get_text().endswith('jumps of a')]
    fig.canvas.draw()
    assert sorted(text.get_text() for text in labels) == ['2 jumps of a', '8 jumps of a']
    boxes = _boxes(fig, labels)
    assert not _overlap(*boxes)
    plt.close(fig)


def test_window_labels_that_would_collide_show_their_range():
    """(e): R² of 12 windows over 2 % of the panel each cannot be written side by side: the colliding ones become one
    'lowest–highest' label; labels far apart stay single."""
    frame = _frame(100000.0)
    fig = plt.figure(figsize=LAYOUT_V1.size, dpi=LAYOUT_V1.dpi)
    ax = fig.add_axes((0.05, 0.5, LAYOUT_V1.panels[2] / LAYOUT_V1.size[0], 0.1))
    ax.set_xlim(0, 100)
    labels = [(10.0 + i, 11.0 + i, 0.74 + 0.01 * i) for i in range(4)] + [(80.0, 90.0, 0.5)]
    mc._window_labels(ax, frame, labels, LAYOUT_V1)
    fig.canvas.draw()
    texts = sorted(text.get_text() for text in ax.texts)
    assert texts == ['0.50', '0.74–0.77']
    assert not _overlap(*_boxes(fig, ax.texts))
    plt.close(fig)


# --------------------------------------------------------------------------------------------- F5: one centre test


@pytest.mark.parametrize('shift, limit, moves', [(1.45, 0.5, True), (0.5, 0.5, False), (0.3, 0.5, False), (float('nan'), 0.5, False)])
def test_centre_moves_above_the_l6_threshold_only(shift, limit, moves):
    assert centre_moves(shift, limit) is moves


# --------------------------------------------------------------------------------------------- F3, F4: secular figure

T = np.linspace(0.0, 1e7, 20001)  # 10 Myr at 500 yr


def _s_term_series():
    """σ = s − s6 librating (period 1 Myr, amplitude 0.8 rad); d sin(i/2)/dt = K sin σ (Lagrange's equations for an
    s-term, Murray & Dermott §6.8), integrated by hand; e constant (the angle has no ϖ)."""
    sigma = 0.8 * np.sin(2 * np.pi * T / 1e6)
    dt = T[1] - T[0]
    x = 0.15 + 1e-8 * np.concatenate([[0.0], np.cumsum(np.sin(sigma[:-1]) * dt)])
    n = len(T)
    return LongPeriodSeries(
        ResonanceType.SECULAR,
        's-s6',
        'synthetic',
        T,
        sigma,
        np.full(n, 2.5),
        np.full(n, 0.1),
        2 * np.arcsin(x),
        terms=(AngleTerm('s', None, 1), AngleTerm('s', 6, -1)),
    )


def test_an_s_term_figure_draws_the_inclination_response():
    """127504 s-s6 drew de/dt with 'R² nan': panel (f) now draws d sin(i/2)/dt with the R² the analysis measured for i,
    and (h) is (sin(i/2) cos σ, sin(i/2) sin σ); the colour bar's label sits above the bar."""
    from resonances.plotting.secular_plots import draw_long_period_combined

    analysis = analyse_secular(_s_term_series(), S)
    response = analysis.response
    assert response.variables == ('i',) and response.r2_i > 0.9 and np.isnan(response.r2_e)
    fig = draw_long_period_combined(analysis)
    panel_f, portrait = fig.axes[5], fig.axes[7]
    assert 'sin(i/2)' in panel_f.get_ylabel()
    assert f'R² {response.r2_i:.3g}' in [text.get_text() for text in panel_f.texts]
    assert portrait.get_xlabel() == 'sin(i/2) cos σ' and portrait.get_ylabel() == 'sin(i/2) sin σ'
    bar = next(ax for ax in fig.axes if ax.get_title(loc='left').startswith('t, '))
    assert bar.get_ylabel() == ''
    plt.close(fig)


def test_the_secular_info_line_names_the_measured_element():
    from resonances.plotting import verdict_text as vt

    analysis = analyse_secular(_s_term_series(), S)
    assert vt._response_words(analysis) == 'sin(i/2) responds to the angle'
    analysis.response.variables = ('e', 'i')
    assert vt._response_words(analysis) == 'e and sin(i/2) respond to the angle'


def test_591986_zlk_labels_of_the_bottom_row_do_not_collide():
    """591986 ZLK: the y label of (i) ('e_free sin ω_free') was printed over the colour bar's 't, kyr'."""
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
    sys.path.insert(0, str(tools))
    import bench_io as bio

    if not (bio.EXAMPLES / 'zlk__591986__LK').exists():
        pytest.skip('benchmark data not available')
    import reclassify_secular_zlk as rsz
    from resonances.plotting.secular_plots import draw_long_period_combined

    sim, body, resonance = rsz.build_simulation(bio.load_example('zlk__591986__LK', with_planets=False))
    sim.identify_librations()
    row = rsz._extra_columns(rsz.summary_row('zlk__591986__LK', body, resonance))
    fig = draw_long_period_combined(sim.long_period_analysis(body, resonance), row=row, free_elements=body.free_elements)
    fig.canvas.draw()
    bar = next(ax for ax in fig.axes if ax.get_title(loc='left').startswith('t, '))
    vector = next(ax for ax in fig.axes if ax.get_ylabel() == 'e_free sin ω_free')
    assert not _overlap(*_boxes(fig, [bar._left_title, vector.yaxis.label]))
    plt.close(fig)
