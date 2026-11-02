"""W28 B10b (cards g08-8 'if the cause is a secular cycle, show it as a line, here and everywhere', g08-7, g05-2,
g05-3): panel (j) draws σ = λ part + pϖ on the scale of a kinematic swing and the p·ϖ of a secular slow cause in every
other mode, labelled with p and the secular period of its column; panel (i) hatches the slow scale's intervals.

The synthetic record is σ = 5λ_J − λ − 4ϖ (5:1 with Jupiter, p = −4) with ϖ = 0.4 + 0.25 sin(2π t / 400 yr) and a flat
a: the analysis reads it as a kinematic libration (non_resonant / kinematic_libration, the term −4ϖ). The truths are
computed here from that closed form, not by the package: −4ϖ swings by ±4 × 0.25 rad = ±57.3°; the λ part
5λ_J − λ is linear in t, so detrended it is 0. The slow scales below are stated by the test (B10a's numbers of 37087
3J+8S-3: g − g5, the identified level), the labels are their column values: 410 yr → '410 yr', the 2ω cycle
7156 yr → '7.16 kyr'. Nothing drawn feeds a status."""

import copy
import dataclasses
import math

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting.style import CENTRE, SUBRESONANCE  # noqa: E402
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec, SlowScale  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import LibrationOrigin  # noqa: E402
from tests.resonances.analysis.builders import make_episode  # noqa: E402

S = settings_from_config()
LAYOUT = mc.LAYOUT_V1
P_VARPI_SWING_DEG = math.degrees(4 * 0.25)  # 57.3°
_CACHE = {}


def teardown_function():
    plt.close('all')


def _record():
    """(analysis, series) of the synthetic −4ϖ kinematic libration, built once (0.2 s)."""
    if 'record' not in _CACHE:
        from scipy.optimize import brentq

        from resonances.data import const
        from resonances.resonance.analysis.mmr import context as ctx
        from resonances.resonance.analysis.mmr.analyse import analyse_mmr

        spec = ResonanceSpec.from_key('5J-1+0-4')
        t = np.arange(0.0, 3000.0 + 0.25, 0.5)
        n_jupiter = const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'] * math.pi / 180
        a_au = brentq(lambda a: float(ctx.mean_motion(a)) - 5 * n_jupiter, 1.0, 5.0)
        lam = (5 * n_jupiter + 2 * math.pi / 60000.0) * t + 0.3
        varpi = 0.4 + 0.25 * np.sin(2 * math.pi * t / 400.0)
        sigma = 5 * (n_jupiter * t + 1.1) - lam - 4 * varpi
        ones = np.ones_like(t)
        a = a_au + 1e-6 * np.sin(2 * math.pi * t / 7.3)
        series = MMRSeries(
            t, a, 0.1 * ones, 0.05 * ones, 0.7 * ones, np.mod(varpi - 0.7, 2 * math.pi), np.mod(varpi, 2 * math.pi),
            np.mod(sigma, 2 * math.pi), spec, Regime(0.5, float(t[-1]), len(t)), {}, np.mod(lam, 2 * math.pi), name='b10b',
        )  # fmt: skip
        _CACHE['record'] = (analyse_mmr(series, S, scan_owner=False, scan_subresonances=False), series)
    return _CACHE['record']


def _with(slow=None, origin=None, **columns):
    """A copy of the synthetic analysis with a slow scale (and the verdict's origin) set, and its row with `columns`."""
    analysis, series = _record()
    analysis = copy.copy(analysis)
    analysis.slow_scale = slow
    if origin is not None:
        analysis.verdict = dataclasses.replace(analysis.verdict, libration_origin=origin)
    row = {**analysis.numbers_row(), **analysis.summary_row(), **columns}
    return analysis, series, row


def _slow(**fields):
    base = dict(source='centre_scale', cutoff_yr=40.0, t_start_yr=500.0, t_end_yr=2500.0, n_cycles=4, period_yr=400.0)
    base.update(halfamp_rad=1.0, secular_term='g-g5', secular_period_yr=410.0, origin='kinematic')
    base.update(cause='secular_g_gj', cause_term='g-g5', cause_level='identified')
    base.update(fields)
    return SlowScale(**base)


SLOW_COLUMNS = dict(
    slow_source='centre_scale',
    slow_origin='kinematic',
    slow_secular_term='g-g5',
    slow_secular_period_yr=410.0,
    slow_cause='secular_g_gj',
    slow_cause_term='g-g5',
    slow_cause_level='identified',
)


def _panel(analysis, series, row, drawer):
    fig = plt.figure(figsize=LAYOUT.size, dpi=LAYOUT.dpi)
    ax = fig.add_axes(tuple(v / s for v, s in zip(mc.panel_rects(LAYOUT)[0], LAYOUT.size * 2)))
    frame = mc._Frame(analysis, series, LAYOUT, None, S, row)
    drawer(ax, frame, LAYOUT)
    return ax, frame


def _half_range(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float((values.max() - values.min()) / 2)


def _line(ax, colour, style):
    lines = [line for line in ax.get_lines() if line.get_color() == colour and line.get_linestyle() == style]
    assert len(lines) == 1, [(line.get_color(), line.get_linestyle()) for line in ax.get_lines()]
    return lines[0]


# --------------------------------------------------------------------------------------------- (j) kinematic


def test_kinematic_mode_draws_sigma_p_varpi_and_the_lambda_part():
    analysis, series = _record()
    assert str(analysis.verdict.reason) == 'kinematic_libration' and mc.centre_mode(analysis) == 'kinematic'
    ax, _ = _panel(analysis, series, None, mc.draw_centres_panel)
    sigma, p_varpi, part = _line(ax, CENTRE, '-'), _line(ax, SUBRESONANCE, '--'), _line(ax, mc.LAMBDA_PART, ':')
    # σ and −4ϖ swing by ±57.3° together (at the episode's response cutoff, ~40 yr, the 400-yr wobble keeps its size)
    assert _half_range(p_varpi.get_ydata()) == pytest.approx(P_VARPI_SWING_DEG, rel=0.03)
    assert _half_range(sigma.get_ydata()) == pytest.approx(P_VARPI_SWING_DEG, rel=0.03)
    assert _half_range(part.get_ydata()) < 1.0  # 5λ_J − λ is linear: flat once detrended
    # no secular period matched (origin_detail varpi_term): the term alone
    assert {text.get_text() for text in ax.texts} >= {'σ', '−4ϖ', 'λ part'}


def test_kinematic_mode_on_the_slow_scale_labels_p_varpi_with_the_column_period():
    analysis, series, row = _with(_slow(), **SLOW_COLUMNS)  # kin_cycle_secular_offset 0.99: σ's own period matches none
    ax, frame = _panel(analysis, series, row, mc.draw_centres_panel)
    assert mc.kinematic_scale(analysis, row, S)[0] == 'slow'
    p_varpi = _line(ax, SUBRESONANCE, '--')
    x = np.asarray(p_varpi.get_xdata(), dtype=float)
    assert x.min() >= frame.times(500.0) - 1e-9 and x.max() <= frame.times(2500.0) + 1e-9  # the slow span only
    assert '−4ϖ 410 yr' in {text.get_text() for text in ax.texts}


def test_a_non_resonant_row_with_a_kinematic_slow_scale_is_drawn_in_the_kinematic_mode():
    analysis, _, _ = _with(_slow(), origin=LibrationOrigin.NONE)
    assert str(analysis.verdict.status) == 'non_resonant' and mc.centre_mode(analysis) == 'kinematic'
    analysis, _, _ = _with(_slow(origin='unresolved'), origin=LibrationOrigin.NONE)
    assert mc.centre_mode(analysis) == 'centre'


# --------------------------------------------------------------------------------------------- (j) overlay


def test_secular_slow_cause_overlays_p_varpi_in_the_centre_mode():
    slow = _slow(origin='unresolved', cause='secular_2omega', cause_term='2omega', secular_term='', secular_period_yr=math.nan)
    columns = {**SLOW_COLUMNS, 'slow_origin': 'unresolved', 'slow_cause': 'secular_2omega', 'slow_cause_term': '2omega'}
    columns.update(slow_secular_term='', slow_secular_period_yr=math.nan, kin_two_omega_period_yr=7156.0)
    analysis, series, row = _with(slow, origin=LibrationOrigin.NONE, **columns)
    assert mc.centre_mode(analysis) == 'centre'
    ax, _ = _panel(analysis, series, row, mc.draw_centres_panel)
    overlay = _line(ax, SUBRESONANCE, '--')
    assert _half_range(overlay.get_ydata()) == pytest.approx(P_VARPI_SWING_DEG, rel=0.03)
    assert '−4ϖ 7.16 kyr' in {text.get_text() for text in ax.texts}


@pytest.mark.parametrize('slow', [None, 'no_swing', 'not_secular'])
def test_nothing_is_drawn_without_a_secular_slow_swing(slow):
    scale = {None: None, 'no_swing': _slow(origin='unresolved', halfamp_rad=math.nan), 'not_secular': _slow(origin='unresolved', cause='')}
    analysis, series, row = _with(scale[slow], origin=LibrationOrigin.NONE)
    ax, frame = _panel(analysis, series, row, mc.draw_centres_panel)
    assert mc.varpi_overlay(frame) is None
    assert not [line for line in ax.get_lines() if line.get_color() == SUBRESONANCE and line.get_linestyle() == '--']
    ax, _ = _panel(analysis, series, row, mc.draw_centre_panel)
    hatched = [patch for patch in ax.patches if patch.get_hatch()]
    assert hatched == [] if slow != 'not_secular' else len(hatched) == 1  # a slow swing without a secular cause: hatched


# --------------------------------------------------------------------------------------------- (i) slow intervals


def test_slow_episodes_are_hatched_and_labelled_on_panel_i():
    # an episode spans its first to last turning point
    runs = tuple(
        make_episode(episode_id=i, tp_times_yr=np.array(span)) for i, span in ((3, (300.0, 850.0, 1400.0)), (4, (1600.0, 2150.0, 2700.0)))
    )
    analysis, series, row = _with(_slow(episodes=runs))
    ax, frame = _panel(analysis, series, row, mc.draw_centre_panel)
    hatched = [patch for patch in ax.patches if patch.get_hatch() == mc.SLOW_HATCH]
    assert len(hatched) == 2
    spans = sorted((round(p.get_x(), 6), round(p.get_x() + p.get_width(), 6)) for p in hatched)
    assert spans == [
        (round(frame.times(300.0), 6), round(frame.times(1400.0), 6)),
        (round(frame.times(1600.0), 6), round(frame.times(2700.0), 6)),
    ]
    assert {'slow #3', 'slow #4'} <= {text.get_text() for text in ax.texts}


def test_a_slow_centre_track_is_one_hatched_span():
    analysis, _, _ = _with(_slow(source='centre_track'))
    assert mc.slow_intervals(analysis) == [(500.0, 2500.0, 'slow')]
