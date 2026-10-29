"""The left panels of the V1 verdict figures after the author's remarks P1-P15, C7, C9, C10, C12 (2026-10-16).

The truths are computed by hand, independently of the code:
- (d) draws a* and a_res moved by <a> - a(lambda_dot): for a plutino-like record with a* = 39.400, a_res = 39.450,
  <a> = 39.470, a(lambda_dot) = 39.401 the shift is 0.069 au, a* at 39.469 au (inside the a_f oscillation
  39.470 +- 0.005; a* itself would be 14 ranges off) and a_res at 39.519 au, 4.4 ranges of a_f above it: beyond the
  3 ranges that are brought into view, so a cyan triangle at the top edge;
- per regime: a = 1.002 au at lambda_dot = n(1.000 au), then a = 1.105 au at lambda_dot = n(1.100 au) after a jump of
  a: shifts 0.002 and 0.005 au;
- (c) without omega0 is not stretched to 0 when sigma_dot would fill less than a quarter of the height (1257 +- 5
  rad/kyr); +-2 omega0 is drawn over each episode with its own omega0 (30 and 20e-3 rad/yr: +-1 and +-40/60);
- jumps of a closer than 1 % of the record are one band: 10, 10.5, 11 | 50 | 90, 90.2 (span 1000 yr);
- (l) takes at most 12 cycles evenly: of 10 cycles 0-100, ..., 900-1000 yr the 4 at indices 0, 3, 6, 9; with no
  more than 12 (or the given number) the whole episode, 0-1050 yr, its partial cycle 1000-1050 included;
- (k) is never clipped tighter than every sample of the accepted episodes (23240 1M+3J-3+0+0-1, W16: (k) clipped to
  -0.09..0.45 rad/kyr, the drawn cycle of (l), while the episode's sigma_dot spans -0.38..0.47): episodes +-1 rad/kyr
  (two values, so the 0.2-99.8 percentiles are -1 and +1), padded by 0.08 of the span 2: -1.16..1.16; the run
  rejected (+-9) and the samples outside the window of (k) (+-3) do not count;
- a dense 1:1 record (about 680 cycles on a 1248 px panel, under 3 px a cycle) has no sigma_f line in (b) but the strip
  (b′) under it, about 5 cycles on its own time axis; the first 15 kyr of it (about 68 cycles, 18 px each) has no strip;
- (c) is sigma_dot / 2 omega0 on a symlog axis: 1257 rad/kyr against 2 omega0 = 0.17 rad/kyr is 7394; +-50 against 60
  is +-0.83; (h) is on a log axis: h from 0.5 to 9 framed by a factor 3, 1/6 to 27;
- the panels carry no explanatory text (author, 2026-10-16): only the letter, '#id', reason letters and data numbers.
"""

import math
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting import style  # noqa: E402
from resonances.resonance.analysis.geometry import RejectReason  # noqa: E402
from resonances.resonance.analysis.mmr.models import ResonanceSpec  # noqa: E402

LAYOUT = mc.LAYOUT_V1


def teardown_function():
    plt.close('all')


def _axis():
    fig = plt.figure(figsize=LAYOUT.size, dpi=LAYOUT.dpi)
    return fig.add_axes(tuple(v / s for v, s in zip(mc.panel_rects(LAYOUT)[0], LAYOUT.size * 2)))


def _texts(ax):
    return [text.get_text() for text in ax.texts]


# --------------------------------------------------------------------------------------------- pure helpers


@pytest.mark.parametrize(
    'value, label', [(0, '0'), (90, '90'), (180, '180'), (-180, '180'), (270, '270'), (450, '90'), (-90, '270'), (360, '0')]
)
def test_wrapped_angle_labels_are_in_0_360(value, label):
    assert mc.angle_label(value) == label


def test_jumps_closer_than_one_percent_are_one_band():
    groups = mc.jump_groups([90.2, 10.0, 50.0, 10.5, 11.0, 90.0], span_yr=1000.0)
    assert groups == [(10.0, 11.0, 3), (50.0, 50.0, 1), (90.0, 90.2, 2)]


def test_portrait_takes_at_most_twelve_cycles_evenly():
    run = SimpleNamespace(t_start_yr=0.0, t_end_yr=1050.0)
    maxima = np.arange(0.0, 1001.0, 100.0)
    times = np.sort(np.concatenate([maxima, maxima[:-1] + 50]))
    kinds = np.where(np.isin(times, maxima), 1, -1)
    assert mc.portrait_cycles(run, times, kinds, max_cycles=4) == [(0, 100), (300, 400), (600, 700), (900, 1000)]
    assert mc.portrait_cycles(run, times, kinds) == [(0.0, 1050.0)]  # 10 cycles, fewer than 12: every sample
    assert mc.portrait_cycles(run, times, kinds, max_cycles=10) == [(0.0, 1050.0)]
    assert mc.portrait_cycles(run, times[:1], kinds[:1]) == [(0.0, 1050.0)]  # one maximum: no whole cycle


def test_episode_rate_limits_hold_every_accepted_sample_inside_the_window():
    t = np.arange(0.0, 1001.0)
    rate = np.where(np.arange(len(t)) % 2, 5.0, -5.0)  # circulation, rad/kyr
    for t0, t1, value in ((100, 200, 1.0), (400, 500, 0.5), (700, 800, 9.0), (900, 1000, 1.0), (951, 1000, 3.0)):
        inside = (t >= t0) & (t <= t1)
        rate[inside] = np.where(np.arange(len(t))[inside] % 2, value, -value)
    runs = [
        SimpleNamespace(t_start_yr=100.0, t_end_yr=200.0, accepted=True),
        SimpleNamespace(t_start_yr=400.0, t_end_yr=500.0, accepted=True),
        SimpleNamespace(t_start_yr=700.0, t_end_yr=800.0, accepted=False),
        SimpleNamespace(t_start_yr=900.0, t_end_yr=1000.0, accepted=True),
    ]
    prep = SimpleNamespace(t_yr=t, window=(0.0, 950.0), rate_response=rate / mc.PER_KYR)
    analysis = SimpleNamespace(prep=prep, geometry=SimpleNamespace(runs=runs))
    assert mc.episode_rate_limits(analysis) == pytest.approx((-1.16, 1.16))
    for run in runs:
        run.accepted = False
    assert mc.episode_rate_limits(analysis) is None
    assert mc.episode_rate_limits(SimpleNamespace(prep=None, geometry=None)) is None


def test_line_breaks_where_the_unwrapped_angle_turned_by_more_than_pi():
    ax = _axis()
    unwrapped = np.array([0.0, 0.5, 1.0, 5.0, 5.5])
    mc._line(ax, np.arange(5.0), np.degrees(unwrapped % (2 * np.pi)), unwrapped=unwrapped)
    y = ax.lines[0].get_ydata()
    assert np.isnan(y[3]) and np.isfinite(np.delete(y, 3)).all()


# --------------------------------------------------------------------------------------------- palette and options


def test_one_palette_distinct_hues_and_no_grey_episode():
    named = [style.MODEL, style.REFERENCE, style.ZONE_FILL, style.SUBRESONANCE, style.OWNER, style.CENTRE, style.AXIS_NOMINAL]
    named += list(style.EPISODE_COLOURS)
    assert len({to_hex(colour) for colour in named}) == len(named)
    assert to_hex(style.MODEL) != to_hex(style.REFERENCE)  # 2 omega0 and a* (author's P14)
    assert to_hex(style.AXIS_EXACT) != to_hex(style.AXIS_NOMINAL)  # a* and a_res apart even when they overlap (P1)
    for colour in style.EPISODE_COLOURS:  # saturated: never confused with RAW, REJECTED, REFERENCE
        rgb = to_rgb(colour)
        assert max(rgb) - min(rgb) > 0.3
    assert to_hex(style.RAW) == '#7f8790' and style.RAW_ALPHA == pytest.approx(0.55)


def test_osculating_a_option():
    assert style.resolve_plot_options()['combined']['osculating_a'] == style.OsculatingAxis.FULL  # author, 2026-10-17
    assert style.resolve_plot_options({'combined': {'osculating_a': 'hidden'}})['combined']['osculating_a'] == 'hidden'
    with pytest.raises(ValueError):
        style.resolve_plot_options({'combined': {'osculating_a': 'sometimes'}})


def test_episode_colours_follow_time_and_cycle():
    runs = [SimpleNamespace(episode_id=i, accepted=i != 2, t_start_yr=100.0 * (6 - i)) for i in range(7)]
    colours = mc.episode_colours(runs)
    assert 2 not in colours
    ordered = [colours[i] for i in (6, 5, 4, 3, 1, 0)]  # time order
    assert ordered == list(style.EPISODE_COLOURS) + [style.EPISODE_COLOURS[0]]


# --------------------------------------------------------------------------------------------- (c), (d), (h)


def _rate_frame(rate_rad_yr, omega0_theory_rad_yr, episodes=()):
    """`episodes`: (id, t_start, t_end, omega0 [rad/yr]) of accepted runs."""
    t = np.arange(0.0, 20000.0, 5.0)
    context = SimpleNamespace(rate_from_mean_longitude_rad_yr=float(np.mean(rate_rad_yr)), rate_expected_rad_yr=float('nan'))
    runs = [SimpleNamespace(episode_id=i, accepted=True, t_start_yr=t0, t_end_yr=t1) for i, t0, t1, _ in episodes]
    physics = [SimpleNamespace(episode_id=i, omega0_rad_yr=omega0) for i, _, _, omega0 in episodes]
    analysis = SimpleNamespace(
        context=context,
        geometry=SimpleNamespace(runs=runs, slips=[], jumps=[]) if episodes else None,
        episode_physics=physics,
        strength=SimpleNamespace(omega0_theory_rad_yr=omega0_theory_rad_yr),
        decisive_episode_id=episodes[0][0] if episodes else None,
        resonance=ResonanceSpec.from_key('7E+2M-9+0+0+0'),
    )
    return SimpleNamespace(
        analysis=analysis,
        prep=SimpleNamespace(rate_response=rate_rad_yr, rate_f=rate_rad_yr),
        inside=np.ones(len(t), dtype=bool),
        t=t,
        x=t / 1e3,
        series=SimpleNamespace(mean_longitude=None),
        times=lambda values: np.asarray(values, dtype=float) / 1e3,
    )


def _dashed(ax):
    """(y, x0, x1) of the blue (model) levels: axhline (x over the axes) and hlines (x in data)."""
    levels = [
        (line.get_ydata()[0], None, None) for line in ax.lines if line.get_linestyle() == '--' and to_hex(line.get_color()) == style.MODEL
    ]
    for collection in ax.collections:
        if len(collection.get_edgecolor()) and to_hex(collection.get_edgecolor()[0]) == style.MODEL:
            for segment in collection.get_segments():
                levels.append((segment[0][1], segment[0][0], segment[1][0]))
    return sorted(levels, key=lambda level: (level[1] if level[1] is not None else -1, level[0]))


def _edge_markers(ax):
    return [line.get_marker() for line in ax.lines if line.get_marker() in ('^', 'v')]


def test_rate_panel_without_omega0_is_not_stretched_to_zero():
    t = np.arange(0.0, 20000.0, 5.0)
    rate = (1257.0 + 5.0 * np.sin(2 * np.pi * t / 400.0)) / 1e3  # rad/yr, a circulation (1999 UQ)
    ax = _axis()
    mc.draw_rate_panel(ax, _rate_frame(rate, omega0_theory_rad_yr=float('nan')), LAYOUT)
    low, high = ax.get_ylim()
    assert ax.get_yscale() == 'linear' and ax.get_ylabel() == 'dσ/dt\nrad/kyr'
    assert low > 1240 and high < 1275  # rad/kyr; 0 not in view: sigma_dot would fill under a quarter of the height
    assert _dashed(ax) == [] and _texts(ax) == ['(c)']


def test_rate_panel_symlog_holds_a_fast_circulation_and_the_separatrix():
    # 1999 UQ: sigma_dot 1257 +- 5 rad/kyr, 2 omega0 (theory) = 0.17 rad/kyr: sigma_dot / 2 omega0 = 7394 +- 29
    t = np.arange(0.0, 20000.0, 5.0)
    rate = (1257.0 + 5.0 * np.sin(2 * np.pi * t / 400.0)) / 1e3
    ax = _axis()
    mc.draw_rate_panel(ax, _rate_frame(rate, omega0_theory_rad_yr=0.085e-3), LAYOUT)
    assert ax.get_yscale() == 'symlog'
    low, high = ax.get_ylim()
    assert low == pytest.approx(-mc.RATE_SYMLOG_VIEW)  # nothing below -1: the separatrix side is still in view
    assert 7394 * 10**0.5 < high < 7424 * 10**0.5 + 1  # the data top widened by half a decade
    assert [level for level, _, _ in _dashed(ax)] == pytest.approx([-1.0, 1.0])  # +-2 omega0 of the theory = +-1
    ticks = list(ax.get_yticks())
    assert 0.0 in ticks and 1.0 in ticks and set(ticks) <= {0.0, -1.0, 1.0, 10.0, 100.0, 1e3, 1e4}
    assert _texts(ax) == ['(c)']


def test_rate_panel_symlog_libration_inside_one():
    # +-50 rad/kyr against 2 omega0 = 60 rad/kyr (theory): +-0.83, the axis +-2.2 with +-1 dashed
    t = np.arange(0.0, 20000.0, 5.0)
    rate = 50.0 * np.sin(2 * np.pi * t / 400.0) / 1e3
    ax = _axis()
    mc.draw_rate_panel(ax, _rate_frame(rate, omega0_theory_rad_yr=30e-3), LAYOUT)
    assert ax.get_ylim() == pytest.approx((-2.2, 2.2))
    assert np.nanmax(np.abs(ax.lines[0].get_ydata())) == pytest.approx(50.0 / 60.0, rel=1e-3)
    assert [level for level, _, _ in _dashed(ax)] == pytest.approx([-1.0, 1.0])
    assert sorted(ax.get_yticks()) == [-1.0, 0.0, 1.0]  # +-1 clear of the tick margin
    assert _texts(ax) == ['(c)']


def test_rate_panel_draws_two_omega0_over_each_episode():
    t = np.arange(0.0, 20000.0, 5.0)
    rate = 50.0 * np.sin(2 * np.pi * t / 400.0) / 1e3
    episodes = [(0, 1000.0, 6000.0, 30e-3), (3, 9000.0, 15000.0, 20e-3), (5, 16000.0, 19000.0, 0.5)]
    ax = _axis()
    mc.draw_rate_panel(ax, _rate_frame(rate, omega0_theory_rad_yr=1e-3, episodes=episodes), LAYOUT)
    assert ax.get_ylim() == pytest.approx((-2.2, 2.2))  # in units of 2 omega0 of the deciding episode #0 (60 rad/kyr)
    levels = _dashed(ax)
    expected = [(-1.0, 1.0, 6.0), (1.0, 1.0, 6.0), (-2 / 3, 9.0, 15.0), (2 / 3, 9.0, 15.0)]  # 40 / 60
    assert np.allclose(levels, expected)
    assert sorted(_edge_markers(ax)) == ['^', 'v']  # episode #5: +-1000 / 60 = +-16.7 beyond the axis, no theory line
    assert _texts(ax) == ['(c)']


def test_separatrix_omega0_from_the_deciding_episode_else_theory():
    runs = [
        SimpleNamespace(episode_id=0, accepted=True, t_start_yr=0.0, t_end_yr=100.0),
        SimpleNamespace(episode_id=1, accepted=True, t_start_yr=200.0, t_end_yr=900.0),
    ]
    physics = [SimpleNamespace(episode_id=0, omega0_rad_yr=0.02), SimpleNamespace(episode_id=1, omega0_rad_yr=0.03)]
    analysis = SimpleNamespace(
        geometry=SimpleNamespace(runs=runs),
        episode_physics=physics,
        decisive_episode_id=0,
        strength=SimpleNamespace(omega0_theory_rad_yr=0.01),
    )
    assert mc.separatrix_omega0(analysis) == (0.02, '#0')
    analysis.decisive_episode_id = None  # the longest librating episode
    assert mc.separatrix_omega0(analysis) == (0.03, '#1')
    analysis.geometry, analysis.episode_physics = SimpleNamespace(runs=[]), []
    assert mc.separatrix_omega0(analysis) == (0.01, 'theory')
    analysis.strength = None
    value, source = mc.separatrix_omega0(analysis)
    assert math.isnan(value) and source == ''


def _axis_frame(mode='clipped'):
    t = np.arange(0.0, 10000.0, 1.0)
    a_f = 39.470 + 0.005 * np.sin(2 * np.pi * t / 2000.0)
    a_osc = a_f + 0.05 * np.sin(2 * np.pi * t / 7.0)
    context = SimpleNamespace(exact_axis_au=39.400, a_mean_au=39.470, axis_from_mean_longitude_rate_au=39.401, nominal_axis_au=39.45)
    analysis = SimpleNamespace(context=context, geometry=None, episode_physics=[])
    return SimpleNamespace(
        analysis=analysis,
        prep=SimpleNamespace(a_f=a_f),
        series=SimpleNamespace(a=a_osc, mean_longitude=None),
        inside=np.ones(len(t), dtype=bool),
        t=t,
        x=t / 1e3,
        times=lambda values: np.asarray(values, dtype=float) / 1e3,
        options=style.resolve_plot_options({'combined': {'osculating_a': mode}}),
    )


def _levels(ax):
    """(colour, y) of the levels drawn over a time span (hlines)."""
    return sorted((to_hex(c.get_edgecolor()[0]), c.get_segments()[0][0][1]) for c in ax.collections if isinstance(c, LineCollection))


def test_axis_panel_moves_a_star_and_a_res_into_the_osculating_frame():
    frame = _axis_frame()
    assert mc.osculating_shift(frame.analysis.context) == pytest.approx(0.069)
    assert mc.axis_shifts(frame) == [(0.0, 9999.0, pytest.approx(0.069))]  # one regime: the context's value
    ax = _axis()
    mc.draw_axis_panel(ax, frame, LAYOUT)
    low, high = ax.get_ylim()
    # a* in the osculating frame in view; a_res (4.4 a_f ranges away) is held by the final view too (`hold_levels`,
    # author 2026-10-23: a_res is a line, not an edge triangle)
    assert low < 39.469 < high and high > 39.519
    levels = dict(_levels(ax))
    assert levels[style.AXIS_EXACT] == pytest.approx(39.469) and levels[style.AXIS_NOMINAL] == pytest.approx(39.519, abs=1e-3)
    markers = [(line.get_marker(), to_hex(line.get_color())) for line in ax.lines if line.get_marker() in ('^', 'v')]
    # the osculating a (+-0.05) beyond both edges
    assert sorted(markers) == sorted([('^', style.RAW), ('v', style.RAW)])
    assert _texts(ax) == ['(d)'] and ax.get_ylabel() == 'a, au'
    assert len([c for c in ax.collections if not isinstance(c, LineCollection)]) == 1  # the osculating band, clipped


def test_axis_panel_draws_a_res_thinner_and_translucent():
    """Variant B (author 2026-10-24): a_res 1.3 pt at alpha 0.8 with a 0.8 pt white halo (also at 0.8); a* 1.2 pt with
    its +1.6 pt halo, also at alpha 0.8 (author 2026-10-25)."""
    ax = _axis()
    mc.draw_axis_panel(ax, _axis_frame(), LAYOUT)
    lines = {to_hex(c.get_edgecolor()[0]): c for c in ax.collections if isinstance(c, LineCollection)}
    nominal, exact = lines[style.AXIS_NOMINAL], lines[style.AXIS_EXACT]
    assert nominal.get_linewidth()[0] == pytest.approx(1.3) and nominal.get_alpha() == pytest.approx(0.8)
    halo = nominal.get_path_effects()[0]
    assert halo._gc['linewidth'] == pytest.approx(2.1) and halo._gc['alpha'] == pytest.approx(0.8)
    assert exact.get_linewidth()[0] == pytest.approx(1.2) and exact.get_alpha() == pytest.approx(0.8)
    halo = exact.get_path_effects()[0]
    assert halo._gc['linewidth'] == pytest.approx(2.8) and halo._gc['alpha'] == pytest.approx(0.8)


def test_axis_panel_hides_a_wide_osculating_a_on_request():
    ax = _axis()
    mc.draw_axis_panel(ax, _axis_frame('hidden'), LAYOUT)
    assert not [c for c in ax.collections if not isinstance(c, LineCollection)] and _texts(ax) == ['(d)']


def test_axis_shift_per_regime_between_jumps():
    from resonances.resonance.analysis.mmr.context import GM_SUN

    t = np.arange(0.0, 2000.0, 0.05)
    jump = 1000.0
    first = t < jump
    a = np.where(first, 1.002, 1.105)
    rate = np.where(first, np.sqrt(GM_SUN / 1.000**3), np.sqrt(GM_SUN / 1.100**3))  # lambda_dot = n(1.000), n(1.100)
    longitude = np.concatenate([[0.0], np.cumsum(rate[1:] * np.diff(t))])
    frame = _axis_frame()
    frame.t, frame.x = t, t / 1e3
    frame.series = SimpleNamespace(a=a, mean_longitude=np.mod(longitude, 2 * np.pi))
    frame.analysis.geometry = SimpleNamespace(jumps=[SimpleNamespace(time_yr=jump, kick=False)])
    shifts = mc.axis_shifts(frame)
    assert [(t0, t1) for t0, t1, _ in shifts] == [(0.0, jump), (jump, t[-1])]
    assert [shift for _, _, shift in shifts] == pytest.approx([0.002, 0.005], abs=2e-5)


def test_energy_panel_log_axis_holds_every_h():
    cycles = [
        SimpleNamespace(t_start_yr=0.0, t_end_yr=100.0, h=0.5, rotation=False),
        SimpleNamespace(t_start_yr=100.0, t_end_yr=200.0, h=1.4, rotation=True),
        SimpleNamespace(t_start_yr=200.0, t_end_yr=300.0, h=9.0, rotation=True),
        SimpleNamespace(t_start_yr=300.0, t_end_yr=400.0, h=math.inf, rotation=True),
        SimpleNamespace(t_start_yr=400.0, t_end_yr=500.0, h=1e-4, rotation=False),
    ]
    zone = SimpleNamespace(h_lo=0.8, h_hi=1.2)
    frame = SimpleNamespace(analysis=SimpleNamespace(energy=SimpleNamespace(cycles=cycles, zone=zone)), times=lambda v: np.asarray(v) / 1e3)
    ax = _axis()
    mc.draw_energy_panel(ax, frame, LAYOUT)
    assert ax.get_yscale() == 'log'
    assert ax.get_ylim() == pytest.approx((0.5 / 3, 9.0 * 3))  # h from 0.5 to 9 (1e-4 is below the floor) x/ 3
    open_points = [line for line in ax.lines if line.get_marker() == 'o' and line.get_markerfacecolor() == 'none']
    assert len(open_points) == 1 and list(open_points[0].get_ydata()) == [1.4, 9.0]  # no cap: h = 9 is a point
    assert to_hex(open_points[0].get_markeredgecolor()) == '#000000'
    assert [len(line.get_xdata()) for line in ax.lines if line.get_marker() == '^'] == [1]  # h = inf at the top
    assert [len(line.get_xdata()) for line in ax.lines if line.get_marker() == 'v'] == [1]  # h = 1e-4 at the bottom
    assert 1.0 in ax.get_yticks()
    assert _texts(ax) == ['(h)']  # no labels: h = 1 dashed, the zone a light band (the legend says which)
    assert to_hex(ax.patches[0].get_facecolor()) == style.ZONE_FILL


def test_h_view_spans_a_circulation_and_a_deep_libration():
    # 5653-like: circulation cycles at h = 40 and 2000 with a libration at 0.01; zone 0.9 - 1.1
    assert mc.h_view([0.01, 40.0, 2000.0], (0.9, 1.1)) == pytest.approx((0.01 / 3, 6000.0))
    assert mc.h_view([1e-6], None) == pytest.approx((1 / 3, 3.0))  # below the floor: only h = 1 frames the axis
    assert mc.h_view([0.002], None)[0] == mc.H_FLOOR  # never below the floor


# --------------------------------------------------------------------------------------------- (a), (j) texts


def test_rejected_runs_carry_the_letter_of_their_reason():
    runs = [
        SimpleNamespace(episode_id=0, t_start_yr=0.0, t_end_yr=4000.0, accepted=True, before_last_jump=False),
        SimpleNamespace(episode_id=1, t_start_yr=4000.0, t_end_yr=6000.0, accepted=False, reject_reason=RejectReason.OVER_THE_TOP),
        SimpleNamespace(episode_id=2, t_start_yr=6000.0, t_end_yr=8000.0, accepted=False, reject_reason=RejectReason.TOO_FEW_TP),
        SimpleNamespace(episode_id=3, t_start_yr=8000.0, t_end_yr=9990.0, accepted=False, reject_reason=RejectReason.CENTRE_MOVES),
        SimpleNamespace(episode_id=4, t_start_yr=9990.0, t_end_yr=10000.0, accepted=False, reject_reason=RejectReason.CENTRE_MOVES),
    ]
    ax = _axis()
    px_per_kyr = LAYOUT.panels[2] * LAYOUT.dpi / 10.0
    mc.shade_runs(ax, runs, {0}, {0: style.EPISODE_COLOURS[0]}, lambda v: np.asarray(v) / 1e3, px_per_kyr, LAYOUT)
    assert _texts(ax) == ['#0', 'T', 'F', 'C']  # the last run is 1.2 px wide: no letter
    label = ax.texts[0]
    assert label.get_position()[1] > 1 and label.get_va() == 'bottom'  # '#id' above the top edge, clear of the data


def test_edge_marker_sits_just_inside_the_edge():
    ax = _axis()
    mc.edge_marker(ax, 0.5, True, style.MODEL)
    mc.edge_marker(ax, 0.5, False, style.MODEL)
    (up, down) = ax.lines
    assert (up.get_marker(), down.get_marker()) == ('^', 'v')
    assert 0.9 < up.get_ydata()[0] < 1.0 and 0.0 < down.get_ydata()[0] < 0.1  # axes fraction: inside the frame


def test_alias_plaque_is_one_short_line():
    sampling = SimpleNamespace(step_yr=5.0012, revs_predictive=3998.84, revs_numpy=-0.16)
    context = SimpleNamespace(rate_measured_rad_yr=2 * np.pi / 5.0015, rate_expected_rad_yr=2 * np.pi / 5.0001)
    assert mc.alias_text(SimpleNamespace(sampling=sampling, context=context)) == 'ALIAS  step 5.0012 yr ≈ P_σ 5.0015 yr · 3999 rev'
    context.rate_measured_rad_yr = float('nan')
    assert mc.alias_text(SimpleNamespace(sampling=sampling, context=context)) == 'ALIAS  step 5.0012 yr · 3999 rev'


# --------------------------------------------------------------------------------------------- a dense record end to end


MU_EARTH = 1.0 / 332946.05
N_EARTH = math.radians(360.000505)


def _dense_tadpole(span_yr: float = 150000.0):
    """A small Earth tadpole (theta = 60 deg +- 2 deg, P about 220 yr) over 150 kyr at 1 yr: about 680 cycles."""
    from resonances.resonance.analysis.mmr.models import MMRSeries, Regime

    t = np.arange(0.0, span_yr, 1.0)
    omega = N_EARTH * math.sqrt(27 / 4 * MU_EARTH)  # small tadpole oscillations about L4 (Murray & Dermott 3.191)
    theta = math.radians(60.0) + math.radians(2.0) * np.sin(omega * t)
    theta_dot = math.radians(2.0) * omega * np.cos(omega * t)
    a = (2 * np.pi / (N_EARTH + theta_dot)) ** (2.0 / 3.0)
    lam_p = N_EARTH * t
    lam = lam_p + theta
    n = len(t)
    varpi = np.full(n, 1.0)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full(n, 1e-3),
        inc=np.full(n, 1e-3),
        Omega=np.zeros(n),
        omega=varpi.copy(),
        varpi=varpi,
        angle_wrapped=np.mod(lam_p - lam, 2 * np.pi),
        resonance=ResonanceSpec.from_key('1E-1'),
        regime=Regime(step_yr=1.0, span_yr=float(t[-1]), n_rows=n),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name='dense tadpole',
    )


@pytest.mark.slow
def test_dense_libration_hides_the_wrapped_line_and_draws_the_strip():
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.resonance.analysis.settings import settings_from_config

    series = _dense_tadpole()
    analysis = analyse_mmr(series, settings_from_config(), scan_subresonances=False, scan_owner=False)
    fig = plt.figure(figsize=LAYOUT.size, dpi=LAYOUT.dpi)
    axes = mc.draw_time_panels(fig, analysis, series, LAYOUT)
    assert len(axes) == len(mc.PANELS) and axes[1] is fig.axes[1]
    wrapped = axes[1]
    assert _texts(wrapped) == ['(b)']  # no explanation on the panel (the rule is in the docs)
    assert not [line for line in wrapped.lines if to_hex(line.get_color()) == '#000000']
    assert not wrapped.child_axes  # no inset: the strip (b′) is a panel of its own
    assert [line.get_marker() for line in wrapped.lines] == ['v']  # where the strip's window is
    rects = mc.panel_rects(LAYOUT, len(mc.PANELS) + 1)
    width, height = LAYOUT.size
    strips = fig.axes[len(mc.PANELS) :]
    assert len(strips) == 1  # one regime: one window
    assert strips[0].get_position().bounds == pytest.approx(tuple(v / s for v, s in zip(rects[2], (width, height) * 2)))
    expected = [tuple(v / s for v, s in zip(rect, (width, height) * 2)) for k, rect in enumerate(rects) if k != 2]
    assert np.allclose([ax.get_position().bounds for ax in axes], expected)
    zoom_t = strips[0].lines[-1].get_xdata()
    period = 2 * np.pi / (N_EARTH * math.sqrt(27 / 4 * MU_EARTH))
    assert 4 * period <= zoom_t[-1] - zoom_t[0] <= 6 * period  # about 5 cycles (t in yr)
    assert _texts(strips[0]) == ['(b′) t, yr']
    points = wrapped.collections[0].get_offsets()
    assert len(points) == len(series.t_yr)  # every sample, no decimation


def test_resolved_cycles_get_no_strip():
    # the same tadpole over 15 kyr: about 68 cycles, 18 px each (>= STRIP_CYCLE_PX): ten panels, no strip
    series = _dense_tadpole(15000.0)
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.resonance.analysis.settings import settings_from_config

    analysis = analyse_mmr(series, settings_from_config(), scan_subresonances=False, scan_owner=False)
    fig = plt.figure(figsize=LAYOUT.size, dpi=LAYOUT.dpi)
    mc.draw_time_panels(fig, analysis, series, LAYOUT)
    assert len(fig.axes) == len(mc.PANELS) and not mc.needs_strip(mc._Frame(analysis, series, LAYOUT))


# --------------------------------------------------------------------------------------------- round 2 (2026-10-16 16:42)
# Truths by hand:
# - 2002 AA29: an episode from TP 70.7 to 883 yr (P 193.3 yr, gaps 0.37 P and 0.61 P <= 0.7 P) covers both edges of the
#   1000-yr record: filled 0-1000 yr;
# - 117306: TPs 37580-40940 yr (P 3359 yr, gaps 11.2 P and 2.7 P): filled +-P/4 = 839.75 yr beyond, 36740.25-41779.75
#   yr; a rejected run ending at 30800 yr is further than that (the middle of the gap is 34190 yr);
# - a run 1000-2000 yr with P = 800 yr after a run ending at 900 yr: P/4 = 200 yr reaches 800, the middle 950 binds;
# - the centre's largest pass is shaded only from one revolution: 0.68 rad (0.108 rev, AA29) no, 4.97 rev (117306) yes;
# - the black a of (d): 52356 response cutoff 306.9 yr >= 1.5 x 20 yr; 153 response cutoff = P_c = 35.43 yr with dense
#   cycles: 2 x P_cyc = 2 x 268 = 536 yr; the same with cycles 140 px wide: none; no episode: W_c = 18414 yr.


def _coverage_run(episode_id, t0, t1, period, start_gap_p, end_gap_p, accepted=True):
    big = 1e9
    return SimpleNamespace(
        episode_id=episode_id,
        accepted=accepted,
        t_start_yr=t0,
        t_end_yr=t1,
        period_med_yr=period,
        coverage_start_gap_p=start_gap_p,
        coverage_end_gap_p=end_gap_p,
        window_start_gap_yr=big,
        window_end_gap_yr=big,
        tp_step_max_yr=period / 2,
        window_start_hidden_p=big,
        window_end_hidden_p=big,
        before_last_jump=False,
    )


SETTINGS = SimpleNamespace(libration_coverage_periods=0.7)


def test_an_episode_covering_both_edges_is_filled_to_the_record_edges():
    run = _coverage_run(0, 70.7, 883.0, 193.3, 0.3656, 0.607)
    assert mc.episode_spans([run], (0.0, 1000.0), SETTINGS) == {0: (0.0, 1000.0)}


def test_an_episode_inside_the_record_is_filled_a_quarter_period_beyond_its_turns():
    rejected = _coverage_run(1, 27800.0, 30800.0, 3000.0, 9.0, 6.0, accepted=False)
    run = _coverage_run(2, 37580.0, 40940.0, 3359.0, 11.19, 2.697)
    spans = mc.episode_spans([rejected, run], (0.0, 50000.0), SETTINGS)
    assert spans == {2: (pytest.approx(36740.25), pytest.approx(41779.75))}
    assert mc.episode_spans([run], (0.0, 50000.0), None)[2] == (pytest.approx(36740.25), pytest.approx(41779.75))


def test_the_fill_stops_halfway_to_a_neighbouring_run():
    before = _coverage_run(0, 100.0, 900.0, 800.0, 0.1, 5.0)
    run = _coverage_run(1, 1000.0, 2000.0, 800.0, 5.0, 5.0)
    spans = mc.episode_spans([run, before], (0.0, 5000.0), SETTINGS)
    assert spans[1] == (pytest.approx(950.0), pytest.approx(2200.0))
    assert spans[0] == (pytest.approx(0.0), pytest.approx(950.0))  # covers the start; ends halfway to the next run


def test_shade_runs_fills_the_given_span():
    run = _coverage_run(0, 70.7, 883.0, 193.3, 0.3656, 0.607)
    ax = _axis()
    mc.shade_runs(ax, [run], {0}, {0: style.EPISODE_COLOURS[0]}, lambda v: np.asarray(v), 1.0, LAYOUT, {0: (0.0, 1000.0)})
    patch = ax.patches[0]
    assert patch.get_x() == pytest.approx(0.0) and patch.get_x() + patch.get_width() == pytest.approx(1000.0)


@pytest.mark.parametrize('max_pass_rad, shaded', [(0.68, False), (4.974 * 2 * math.pi, True), (2 * math.pi, True), (float('nan'), False)])
def test_largest_pass_is_shaded_only_from_one_revolution(max_pass_rad, shaded):
    centre = SimpleNamespace(max_pass_rad=max_pass_rad, max_pass_t0_yr=0.0, max_pass_t1_yr=30000.0)
    assert mc.centre_pass_matters(centre) is shaded


def _strong_frame(response, period, cycle_px, episode=True, cutoff=20.0, window=18414.0):
    runs = [SimpleNamespace(episode_id=0, accepted=True, t_start_yr=0.0, t_end_yr=1.0, period_med_yr=period)] if episode else []
    geometry = SimpleNamespace(runs=runs, centre=SimpleNamespace(window_yr=window))
    analysis = SimpleNamespace(
        prep=SimpleNamespace(cutoff=SimpleNamespace(cutoff_yr=cutoff), response_cutoff_yr=response),
        geometry=geometry,
        decisive_episode_id=0,
    )
    return SimpleNamespace(analysis=analysis, cycle_px=lambda: cycle_px)


def test_black_a_cutoff_from_the_resonance_scale():
    assert mc.strong_axis_cutoff(_strong_frame(306.9, 5930.0, 80.0)) == pytest.approx(306.9)
    assert mc.strong_axis_cutoff(_strong_frame(35.43, 268.0, 3.3, cutoff=35.43)) == pytest.approx(536.0)
    assert math.isnan(mc.strong_axis_cutoff(_strong_frame(1171.0, 23000.0, 140.0, cutoff=1013.0)))
    assert mc.strong_axis_cutoff(_strong_frame(float('nan'), float('nan'), 3.0, episode=False)) == pytest.approx(18414.0)


def test_axis_panel_three_levels_and_the_margin_legend():
    frame = _axis_frame()
    t = frame.t
    frame.analysis.prep = SimpleNamespace(cutoff=SimpleNamespace(cutoff_yr=20.0), response_cutoff_yr=300.0)
    frame.analysis.geometry = SimpleNamespace(
        runs=[SimpleNamespace(episode_id=0, accepted=True, t_start_yr=0.0, t_end_yr=9999.0, period_med_yr=2000.0)],
        centre=None,
        jumps=[],
        turning_points=SimpleNamespace(indices=[], kinds=[]),
    )
    frame.analysis.decisive_episode_id = 0
    frame.prep = SimpleNamespace(a_f=frame.prep.a_f + 0.002 * np.sin(2 * np.pi * t / 50.0), cutoff=frame.analysis.prep.cutoff)
    frame.settings, frame.filter_order, frame.cycle_px = None, 2, lambda: 50.0
    ax = _axis()
    mc.draw_axis_panel(ax, frame, LAYOUT)
    black = [line for line in ax.lines if to_hex(line.get_color()) == '#000000' and line.get_marker() in ('None', '')]
    solid = [line for line in black if line.get_linestyle() == '-']
    assert len(solid) == 1 and len(black) == 2  # the slow a at 300 yr (a_f is a grey band under it) and its edge spread
    shown = np.asarray(solid[0].get_ydata(), dtype=float)
    assert np.nanmax(np.abs(shown - (39.470 + 0.005 * np.sin(2 * np.pi * solid[0].get_xdata() * 1e3 / 2000.0)))) < 1e-3
    assert np.isnan(shown[0]) and np.isnan(shown[-1])  # one cutoff trimmed at each record end ...
    edge = np.asarray(next(line for line in black if line is not solid[0]).get_ydata(), dtype=float)
    assert np.isfinite(edge[0]) and np.isfinite(edge[-1])  # ... drawn there dashed, so the panel covers the record
    texts = [text.get_text() for text in ax.figure.texts]
    assert any(text.startswith('a* 39.400') for text in texts) and any(text.startswith('a_res 39.450') for text in texts)
    assert any(text.endswith('+0.069') for text in texts)  # <a> - a(lambda_dot)
    assert 'a_f 20 yr' in texts and 'a 300 yr' in texts
    assert _texts(ax) == ['(d)']  # the panel itself carries no text


def test_portraits_share_their_scales():
    fig = plt.figure(figsize=LAYOUT.size, dpi=LAYOUT.dpi)
    whole, episodes = (mc._rect_axes(fig, rect) for rect in LAYOUT.bottom[:2])
    for ax, limits in ((whole, (-4.0, 3.5)), (episodes, (-3.0, 4.5))):
        ax.set_xlabel('σ_f, °')
        ax.set_ylim(*limits)
    mc.share_portrait_limits((whole, episodes))
    assert whole.get_ylim() == episodes.get_ylim() == (-4.0, 4.5)
    whole.set_ylim(-2500.0, 2500.0)  # a circulation 30 times faster than the libration of (l): (l)'s view, (k) clipped
    episodes.set_ylim(-80.0, 80.0)
    mc.share_portrait_limits((whole, episodes))
    assert whole.get_ylim() == episodes.get_ylim() == (-80.0, 80.0)
    assert sorted(line.get_marker() for line in whole.lines) == ['^', 'v']


def _portraits(k_limits, l_limits):
    fig = plt.figure(figsize=LAYOUT.size, dpi=LAYOUT.dpi)
    whole, episodes = (mc._rect_axes(fig, rect) for rect in LAYOUT.bottom[:2])
    for ax, limits in ((whole, k_limits), (episodes, l_limits)):
        ax.set_xlabel('σ_f, °')
        ax.set_ylim(*limits)
    return whole, episodes


def test_whole_record_portrait_is_never_clipped_tighter_than_its_episodes():
    # 23240 1M+3J-3+0+0-1 (W16): (l) drew one cycle with a stop of sigma, -0.09..0.45 rad/kyr; the record -0.87..0.86.
    # (k) span 1.73 > 3 x 0.54: (k) was clipped to (l)'s view. The episode's samples, -0.45..0.54 padded, give the
    # episodes' view a span 0.99, 3 x 0.99 = 2.97 >= 1.73: the union, (k) whole.
    whole, episodes = _portraits((-0.87, 0.86), (-0.09, 0.45))
    mc.share_portrait_limits((whole, episodes))
    assert whole.get_ylim() == pytest.approx((-0.09, 0.45))  # without the episodes' samples: the old clipping
    whole, episodes = _portraits((-0.87, 0.86), (-0.09, 0.45))
    mc.share_portrait_limits((whole, episodes), (-0.45, 0.54))
    assert whole.get_ylim() == episodes.get_ylim() == pytest.approx((-0.87, 0.86))
    assert not whole.lines  # nothing beyond the view
    # a circulation 25 times faster than the episodes: still the episodes' view, widened to all their samples
    whole, episodes = _portraits((-2500.0, 2500.0), (-80.0, 80.0))
    mc.share_portrait_limits((whole, episodes), (-100.0, 90.0))
    assert whole.get_ylim() == episodes.get_ylim() == (-100.0, 90.0)
    assert sorted(line.get_marker() for line in whole.lines) == ['^', 'v']
    # (l) wider than the episodes' samples (its legend room): (l)'s view is kept
    whole, episodes = _portraits((-2500.0, 2500.0), (-80.0, 120.0))
    mc.share_portrait_limits((whole, episodes), (-70.0, 90.0))
    assert whole.get_ylim() == episodes.get_ylim() == (-80.0, 120.0)


def test_bottom_slots_are_squares():
    for _, _, w, h in LAYOUT.bottom:
        assert w == pytest.approx(h)
    assert LAYOUT.panels[0] + LAYOUT.panels[2] < LAYOUT.margin_legend_x < LAYOUT.separator_x


# --------------------------------------------------------------------------------------------- author, 2026-10-17
# Truths by hand: numpy's linear percentile of 0, 1, ..., 999 at 0.5 % is at index 0.005 x 999 = 4.995 (value 4.995),
# at 99.5 % at 994.005; spikes +5 and +3 at 10.0 and 10.1 yr (one group: 0.1 < 2 % of 100 yr), +2 at 50 yr and -4 at
# 80 yr beyond [-1, 1] are two triangles at the top (10.0 and 50) and one at the bottom (80); the d'Alembert sum of
# 10E-17 + p_E + p + 2 (Omega) = 0 gives p_E + p = 5 for the nodal members [N+2,NE+0] of 2003 UG22.


def test_envelope_range_is_the_half_percent_band():
    assert mc.envelope_range(np.arange(1000.0)) == pytest.approx((4.995, 994.005))
    assert mc.envelope_range([np.nan]) is None


def test_outlier_markers_one_triangle_per_group_at_its_time():
    x = np.arange(0.0, 100.0, 0.1)
    y = np.zeros_like(x)
    for at, value in ((10.0, 5.0), (10.1, 3.0), (50.0, 2.0), (80.0, -4.0)):
        y[int(round(at * 10))] = value
    ax = _axis()
    ax.set_xlim(0, 100)
    assert mc.outlier_markers(ax, x, y, -1.0, 1.0) == 3
    triangles = sorted((line.get_marker(), float(line.get_xdata()[0])) for line in ax.lines)
    assert triangles == [('^', 10.0), ('^', 50.0), ('v', 80.0)]
    assert mc.outlier_markers(_axis(), x, y, -1.0, 1.0, limit=1) == 2  # the largest excess per edge


def test_angle_panel_view_holds_the_osculating_band():
    t = np.arange(0.0, 10000.0, 1.0)
    filtered = np.sin(2 * np.pi * t / 2000.0)
    raw = filtered + 2.0 * np.sin(2 * np.pi * t / 7.3)  # the grey band reaches +-3, the filtered line +-1
    low, high = mc.angle_view(filtered, raw)
    band = np.percentile(raw, [0.5, 99.5])
    assert low < band[0] < -2.5 and 2.5 < band[1] < high
    assert (high - low) == pytest.approx((band[1] - band[0]) * (1 + 2 * mc.ANGLE_VIEW_PAD))
    assert mc.angle_view(filtered, None) == pytest.approx((-1.2, 1.2), abs=1e-3)  # sigma_f alone, as before


def test_axis_panel_full_holds_the_osculating_a():
    frame = _axis_frame('full')
    ax = _axis()
    mc.draw_axis_panel(ax, frame, LAYOUT)
    low, high = ax.get_ylim()
    band = np.percentile(frame.series.a, [0.5, 99.5])  # a_f +- 0.05: about 39.42 .. 39.52
    assert low < band[0] and band[1] < high and high - low < 1.2 * (band[1] - band[0])
    raw_triangles = [line for line in ax.lines if line.get_marker() in ('^', 'v') and to_hex(line.get_color()) == style.RAW]
    assert raw_triangles == []  # a sine of 7 yr sampled every year has 7 values: the band is its whole range
    assert len([c for c in ax.collections if not isinstance(c, LineCollection)]) == 1  # the osculating band


def test_edge_spread_is_drawn_pale_and_dashed():
    t = np.arange(0.0, 1001.0, 1.0)
    frame = SimpleNamespace(x=t / 1e3, inside=(t >= 100.0) & (t <= 900.0))
    ax = _axis()
    shown = mc._edge_line(ax, frame, np.sin(t / 50.0), color='black', lw=0.8)
    assert np.isnan(shown[:100]).all() and np.isfinite(shown[100:901]).all() and np.isnan(shown[901:]).all()
    solid, dashed = ax.lines
    assert solid.get_linestyle() == '-' and dashed.get_linestyle() == '--'
    assert dashed.get_alpha() == pytest.approx(style.EDGE_SPREAD_STYLE['alpha'])
    edge = np.asarray(dashed.get_ydata(), dtype=float)
    reach = 2 * mc._stride(len(t))  # two decimation strides into the window: the lines join after decimation too
    assert np.isfinite(edge[: 101 + reach]).all() and np.isfinite(edge[900 - reach :]).all()
    assert np.isnan(edge[101 + reach : 900 - reach]).all()


def test_edge_spread_beyond_the_view_is_a_triangle():
    # the view is fitted to the analysed window (|values| <= 1); an end transient of 50 must not vanish silently
    t = np.arange(0.0, 1001.0, 1.0)
    values = np.sin(t / 50.0)
    values[:20], values[-5:] = 50.0, -50.0
    frame = SimpleNamespace(x=t / 1e3, inside=(t >= 100.0) & (t <= 900.0))
    ax = _axis()
    mc._edge_line(ax, frame, values, color='black')
    ax.set_ylim(-1.2, 1.2)
    assert mc.edge_overflow_markers(ax, frame, values) == 2
    assert sorted(line.get_marker() for line in ax.lines if line.get_marker() in ('^', 'v')) == ['^', 'v']


def test_clipped_axis_panel_marks_an_osculating_a_beyond_it_at_the_record_ends():
    frame = _axis_frame('clipped')
    frame.inside = (frame.t >= 1000.0) & (frame.t <= 8999.0)
    frame.series.a = frame.prep.a_f.copy()
    frame.series.a[:500] = 100.0  # beyond the view, outside the analysis window only
    ax = _axis()
    mc.draw_axis_panel(ax, frame, LAYOUT)
    assert ('^', style.RAW) in [(line.get_marker(), to_hex(line.get_color())) for line in ax.lines]


def test_axis_view_without_a_filtered_a_uses_the_band_not_a_spike():
    raw = np.ones(10000)
    raw[0] = 1000.0  # one sample: outside the 0.5-99.5 % band
    low, high = mc._axis_view(np.full(10000, np.nan), raw, [])
    assert low < 1.0 < high and high < 2.0


def test_planar_members_of_a_nodal_angle_keep_its_nodes():
    from resonances.resonance.analysis.mmr.subresonances import member_spec

    nodal = member_spec(ResonanceSpec.from_key('10E-17+0+7'), (5, 0, 2, 0))  # 10E-17+0+5[N+2,NE+0]
    members = mc.planar_members(nodal)
    expected = {f'10E-17{p_e:+d}{5 - p_e:+d}[N+2,NE+0]' for p_e in range(1, 6)}
    assert {member.key for member in members} == expected
    for member in members:
        assert member.body_node_coefficient == 2 and member.planet_node_coefficient(0) == 0
        p_e = member.planet_varpi_coefficients[0]
        assert mc.varpi_offset(nodal, member) == (-p_e, (p_e,))  # sigma_s - sigma0 = p_E (varpi_E - varpi)


# --------------------------------------------------------------------------------------------- (e) the force law


def _adot_frame(response):
    """A record of 2000 yr (1-yr step): sigma at the response cutoff a 400-yr libration of 1 rad, sigma_f the same plus
    a 40-yr wave of 0.5 rad that the response cutoff removed; a_dot at the response cutoff of order 1e-6 au/yr."""
    t = np.arange(0.0, 2000.0, 1.0)
    angle_response = np.sin(2 * np.pi * t / 400.0)
    adot = 2e-6 * np.sin(angle_response - 0.3) + 1e-7
    prep = SimpleNamespace(
        angle_f=angle_response + 0.5 * np.sin(2 * np.pi * t / 40.0), angle_response=angle_response, adot_response=adot, adot_f=adot
    )
    analysis = SimpleNamespace(response=response, geometry=None, episode_physics=[])
    return SimpleNamespace(
        analysis=analysis,
        prep=prep,
        inside=np.ones(len(t), dtype=bool),
        t=t,
        x=t / 1e3,
        px_per_unit=LAYOUT.panels[2] * LAYOUT.dpi / 2.0,
        times=lambda values: np.asarray(values, dtype=float) / 1e3,
    )


def _model_lines(ax):
    return [line for line in ax.lines if to_hex(line.get_color()) == style.MODEL]


def test_force_window_is_drawn_on_the_angle_the_fit_read():
    """One window over the record, R = 2e-6 au/yr, phi0 = 0.3, C = 1e-7: the drawn law is R sin(sigma - phi0) + C on
    sigma at the response cutoff (what the fit read), by hand 2e-6 sin(sin(2 pi t / 400) - 0.3) + 1e-7, not on sigma_f
    with its 40-yr wave (which differs from it by up to ~1e-6)."""
    from resonances.resonance.analysis.mmr.models import ForceWindow
    from tests.resonances.analysis.builders import make_response

    window = ForceWindow(0.0, 2000.0, 0.81, 2e-6, 0.3, 1e-7)
    frame = _adot_frame(make_response(windows=[window], force_law='single'))
    ax = _axis()
    mc.draw_adot_panel(ax, frame, LAYOUT)
    (line,) = _model_lines(ax)
    t = np.arange(0.0, 2000.0, 1.0)
    expected = 2e-6 * np.sin(np.sin(2 * np.pi * t / 400.0) - 0.3) + 1e-7  # in units of 10^-6 au/yr: x 1e6
    np.testing.assert_allclose(line.get_ydata(), expected * 1e6, rtol=1e-12, atol=1e-12)
    on_sigma_f = 2e-6 * np.sin(frame.prep.angle_f - 0.3) + 1e-7
    assert np.max(np.abs(line.get_ydata() - on_sigma_f * 1e6)) > 0.5
    assert '0.81' in _texts(ax)


def test_harmonic_law_draws_no_sine_and_prints_its_r2():
    """2001 ND13 1V-3 (W23 CS12): the two-harmonic law decided L4 with R² 0.646 where one harmonic gave 0.18; the
    single-sine windows (R² 0.18/0.19) are not the law that was tested, so none is drawn and (e) reads
    'harmonic law N=2: R² 0.65 (single 0.18)'. A law not taken (or no force law 'harmonic') keeps the windows."""
    from resonances.resonance.analysis.mmr.models import ForceWindow, HarmonicLaw
    from tests.resonances.analysis.builders import make_response

    windows = [ForceWindow(0.0, 1000.0, 0.18, 2e-6, 0.3, 1e-7), ForceWindow(1000.0, 2000.0, 0.19, 2e-6, 0.3, 1e-7)]
    taken = HarmonicLaw(2, 0.646, 0.18, 2, taken=True)
    ax = _axis()
    mc.draw_adot_panel(ax, _adot_frame(make_response(windows=windows, force_law='harmonic', force_harmonic=taken)), LAYOUT)
    assert not _model_lines(ax)
    assert 'harmonic law N=2: R² 0.65 (single 0.18)' in _texts(ax)
    refused = HarmonicLaw(2, 0.646, 0.18, 2, taken=False, reason='null')
    ax = _axis()
    mc.draw_adot_panel(ax, _adot_frame(make_response(windows=windows, force_law='single', force_harmonic=refused)), LAYOUT)
    assert len(_model_lines(ax)) == 2 and not any(text.startswith('harmonic law') for text in _texts(ax))
