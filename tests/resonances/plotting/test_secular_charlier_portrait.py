"""W36 (the author, 2026-11-07, item 4): the bottom row of the secular/ZLK figure holds four squares, (i) the osculating
portrait, (j) sigma_f against its rate (or the free eccentricity vector), (k) the Charlier portrait of sigma and e
filtered at the decisive cutoff (Carruba, Aljbaae & Winter 2016, Fig. 3B), (l) the same per episode; the legend moved
to a band under the row. The paper figure (combined.figure='panels') leaves an empty square out and re-letters.

Charlier's radius by hand (sqrt(2P/L), P = L (1 - sqrt(1 - e^2)); for the node sqrt(2Q/L), Q = L sqrt(1-e^2)(1-cos i)):
    e = 0.1:              sqrt(1 - 0.01) = 0.9949874, R = sqrt(2 x 0.0050126) = 0.1001256
    e = 0.5:              sqrt(0.75) = 0.8660254,      R = sqrt(2 x 0.1339746) = 0.5176381
    e = 0,   i = 10 deg:  R = sqrt(2 (1 - cos 10 deg)) = 2 sin 5 deg = 0.1743115
    e = 0.6, i = 30 deg:  R = sqrt(2 x 0.8 x (1 - 0.8660254)) = sqrt(0.2143594) = 0.4629896

The synthetic angle (built here): a libration 1 + A sin(2 pi t / P), A = 60 deg, P = 2 Myr over 9.25 P, then 1.5 turns
down (as test_secular_edge_arc: one episode, then circulation); e = 0.1 + 0.02 sin(2 pi t / 20 kyr), a wave far below
the cutoff (>= the 200-kyr floor), so e_f = 0.1 and R = 0.1001256 inside the filter's window.
"""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from resonances.plotting import secular_plots as sp  # noqa: E402
from resonances.resonance.analysis.secular import LongPeriodSeries  # noqa: E402
from resonances.resonance.analysis.secular.analyse import analyse_secular  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import ResonanceType  # noqa: E402

SETTINGS = settings_from_config()
PERIOD, STEP = 2.0e6, 2000.0
PAPER = {'combined': {'figure': 'panels'}}


@pytest.mark.parametrize(
    'e, inc_deg, expected',
    [(0.1, None, 0.1001256), (0.5, None, 0.5176381), (0.0, 10.0, 0.1743115), (0.6, 30.0, 0.4629896), (0.0, None, 0.0)],
)
def test_charlier_radius_by_hand(e, inc_deg, expected):
    inc = None if inc_deg is None else np.radians(inc_deg)
    assert float(sp.charlier_radius(e, inc)) == pytest.approx(expected, abs=1e-6)


def _series(circulating: bool = False) -> LongPeriodSeries:
    t = np.arange(0.0, 10 * PERIOD + STEP / 2, STEP)
    if circulating:  # three turns at a steady rate: no turning point, no episode
        sigma = -2 * np.pi * 3 * t / t[-1]
    else:
        t_last = 9.25 * PERIOD
        sigma = 1.0 + np.radians(60.0) * np.sin(2 * np.pi * t / PERIOD)
        after = t > t_last
        s = (t[after] - t_last) / (t[-1] - t_last)
        sigma[after] = 1.0 + np.radians(60.0) - 2 * np.pi * 1.5 * s**2
    one = np.ones_like(t)
    e = 0.1 + 0.02 * np.sin(2 * np.pi * t / 2.0e4)
    return LongPeriodSeries(ResonanceType.SECULAR, 'g-g6', 'synthetic', t, sigma, 2.4 * one, e, 0.3 * one)


@pytest.fixture(scope='module')
def librating():
    return analyse_secular(_series(), SETTINGS)


def _titles(fig):
    return [ax.get_title(loc='left') for ax in fig.axes if ax.get_title(loc='left').startswith('(')]


def _square(fig, letter):
    return next(ax for ax in fig.axes if ax.get_title(loc='left') == f'({letter})')


def test_the_filtered_charlier_portrait_reads_e_and_sigma_at_the_decisive_cutoff(librating):
    charlier = sp.filtered_charlier(librating, SETTINGS)
    inside = charlier.inside
    # the 20-kyr wave of e is filtered out: R is that of e = 0.1 inside the window, the window is the angle's
    assert np.max(np.abs(charlier.radius[inside] - 0.1001256)) < 2e-3
    window = librating.angle.window
    assert charlier.t_yr[inside][0] >= window[0] and charlier.t_yr[inside][-1] <= window[1]
    # the polar angle of each point is sigma_f itself
    np.testing.assert_allclose(charlier.sigma[inside], librating.angle.angle_f[inside])
    assert charlier.name == 'R(e)' and charlier.limit == pytest.approx(1.05 * np.max(charlier.radius[inside]))


def test_the_four_squares_and_the_episode_colours_of_a(librating):
    fig = sp.draw_long_period_combined(librating, settings=SETTINGS)
    try:
        assert _titles(fig) == ['(i)', '(j)', '(k)', '(l)']
        k, episodes = _square(fig, 'k'), _square(fig, 'l')
        assert k.get_xlabel() == 'R(e) cos σ_f' and k.get_xlim() == pytest.approx(episodes.get_xlim())
        # (l): one curve per episode of (a), in the colour (a) fills it with
        colours = sp.episode_colour_map(librating)
        assert colours
        labelled = {line.get_label(): to_hex(line.get_color()) for line in episodes.lines if line.get_label().startswith('#')}
        for episode_id, colour in colours.items():
            (label,) = [key for key in labelled if key.startswith(f'#{episode_id} ')]
            assert labelled[label] == colour
        # the four squares lie left of the separator and in one row, (i) to (l)
        boxes = [ax.get_position().bounds for ax in (_square(fig, letter) for letter in 'ijkl')]
        width = fig.get_size_inches()[0]
        assert [round(b[0] * width, 6) for b in boxes] == [rect[0] for rect in sp.LONG_PERIOD_BOTTOM]
        assert (boxes[-1][0] + boxes[-1][2]) * width < sp.long_period_layout().separator_x
    finally:
        plt.close(fig)


def test_the_legend_has_its_own_band_under_the_squares(librating):
    """The legend no longer takes a square: it fits the band under the bottom row (its columns above the page's edge,
    below the squares' axis labels), the page LONG_PERIOD_LEGEND_BAND_IN taller than the MMR one."""
    fig = sp.draw_long_period_combined(librating, settings=SETTINGS, options={'combined': {'numbers': 'all'}})
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert fig.get_size_inches()[1] == pytest.approx(13.5 + sp.LONG_PERIOD_LEGEND_BAND_IN)
        heading = next(text for text in fig.texts if text.get_text() == 'Legend')
        columns = [text for text in fig.texts if text.get_text().startswith(('(a)', '(h)', '(g)', '(i)'))]
        assert len(columns) == sp.LONG_PERIOD_LEGEND_COLUMNS
        legend_top = heading.get_window_extent(renderer).y1
        squares = min(_square(fig, letter).get_tightbbox(renderer).y0 for letter in 'ijkl')
        assert legend_top < squares
        assert min(text.get_window_extent(renderer).y0 for text in columns) >= 0
        assert '(k) Charlier portrait' in ' '.join(text.get_text().replace('\n', ' ').replace('    ', ' ') for text in columns)
    finally:
        plt.close(fig)


def test_a_record_without_an_episode_leaves_l_out_of_the_paper_figure():
    analysis = analyse_secular(_series(circulating=True), SETTINGS)
    assert not sp.portrait_episodes(analysis, SETTINGS)[0]
    full = sp.draw_long_period_combined(analysis, settings=SETTINGS)
    paper = sp.draw_long_period_combined(analysis, settings=SETTINGS, options=PAPER)
    try:
        assert _titles(full) == ['(i)', '(j)', '(k)', '(l)'] and _titles(paper) == ['(i)', '(j)', '(k)']
        assert 'no librating episode' in [text.get_text() for text in _square(full, 'l').texts]
        assert tuple(paper.get_size_inches()) == pytest.approx((12.45, 12.8))
    finally:
        plt.close(full)
        plt.close(paper)


def test_legend_columns_balance_the_wrapped_rows():
    """At 20 characters 'word ' x 12 wraps to 4 rows ('word word word word' = 19, then 3 more with the 3-space
    indent), a 10-character line to 1: the split that keeps the taller column shortest, in order."""
    long, short = 'word ' * 12, 'a' * 10
    assert sp.legend_columns([short] * 4, 2, 20) == [[short] * 2, [short] * 2]
    assert sp.legend_columns([long, short, short, short], 2, 20) == [[long], [short] * 3]
    assert sp.legend_columns([short, short, long, short], 2, 20) == [[short, short], [long, short]]
    assert sp.legend_columns([short], 2, 20) == [[short]]


def test_the_longest_legend_fits_its_band_above_the_floor_font():
    """With the lines of (a′) and of the minimal policy (591986: the left column ran 0.07 in under the page at the
    floor font before the columns were balanced by wrapped rows) every column stays inside the band."""
    from resonances.plotting.mmr_combined import LEGEND_MIN_FONT_PT

    layout = sp.long_period_layout()
    fig = plt.figure(figsize=layout.size, dpi=layout.dpi)
    try:
        font = sp._draw_legend(fig, layout, forcing=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        columns = [text for text in fig.texts if text.get_text() != 'Legend']
        assert font > LEGEND_MIN_FONT_PT and len(columns) == sp.LONG_PERIOD_LEGEND_COLUMNS
        x, y, w, h = layout.legend
        for text in columns:
            box = text.get_window_extent(renderer)
            assert box.y0 >= y * layout.dpi - 0.5 and box.x1 <= (x + w) * layout.dpi + 0.5
    finally:
        plt.close(fig)


@pytest.mark.parametrize('options', [{'combined': {'numbers': 'all'}}, {'combined': {'numbers': 'all', 'figure': 'panels'}}])
def test_an_explicit_mmr_layout_gets_the_four_squares(librating, options):
    """LAYOUT_V1 (the MMR figure's three 2.4-in squares, the legend in a slot) passed explicitly used to run out of
    slots at (l) (Codex W36 review, IndexError): its page gets the secular/ZLK row, LONG_PERIOD_BOTTOM's squares at x =
    0.95, 4.35, 7.2, 10.0 in, and in the full figure the legend band (13.5 + 1.45 in tall), as without a layout."""
    from resonances.plotting.style import LAYOUT_V1

    assert len(LAYOUT_V1.bottom) == 3
    fig = sp.draw_long_period_combined(librating, settings=SETTINGS, layout=LAYOUT_V1, options=options)
    try:
        assert _titles(fig) == ['(i)', '(j)', '(k)', '(l)']
        width = fig.get_size_inches()[0]
        assert [round(_square(fig, letter).get_position().bounds[0] * width, 6) for letter in 'ijkl'] == [0.95, 4.35, 7.2, 10.0]
        paper = options['combined'].get('figure') == 'panels'
        assert fig.get_size_inches()[1] == pytest.approx(12.8 if paper else 13.5 + 1.45)
    finally:
        plt.close(fig)
