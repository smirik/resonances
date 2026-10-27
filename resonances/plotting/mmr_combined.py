"""The combined MMR figure of the v2 analysis: one page, layout V1 approved by the author ('A').

24 x 13.5 in (16:9), 120 dpi. Left: ten long thin time panels (a)–(j) sharing t, each in its own frame (eleven
when the cycles are too dense for (b): the strip (b′) under it shows a few of them on their own time axis); below
them the portrait of the whole record (k) with the time colour bar, the portrait of the accepted episodes (l), FAIR
(m, two-body angles) and the legend (in the third slot too when there is no FAIR). Right (`draw_right_column`): the
plaque with the status (printed nowhere else), the verdict explained in words, the steps of the decision, the margins
and caps of the confidence and the numbers a reader needs (`mmr_verdict_panel.draw_explained`, `verdict_text`), or
every number of the decision (plot option `combined.numbers='all'`). Height and layout are the same for every object
(figures of one status folder are leafed through).

The panels carry no explanatory text (the author, 2026-10-16: what a line means is in the legend and docs/plots.md):
lines are not labelled, a level outside a panel's axis is a small triangle at the edge (▲ above, ▼ below), and the
only numbers written are data (min – max of e and i, R² of the force law, keys of the multiplet's members). The
numbers of the reference lines of (c), (d), (h) stand right of them, outside the axes (`margin_legend`). An accepted
episode is filled to a record edge it covers, else P/4 beyond its turns (`episode_spans`).

(a) holds the whole grey band of the osculating sigma and (d) that of the osculating a (`style.ENVELOPE_PERCENTILES`,
the rare samples beyond are triangles at the edge, `outlier_markers`). The filtered lines of (c), (d), (e) cover the
whole record: within FILTER_EDGE_TRIM_CUTOFFS cutoffs of its ends (outside the analysis window, the filter's edge
spread) they are pale and dashed (`style.EDGE_SPREAD_STYLE`, `_edge_line`); the view is fitted to the analysed part,
an edge transient beyond it is a triangle at the edge (`edge_overflow_markers`).

(d) draws a on an absolute axis (au): the osculating a (light grey), a_f (mid grey), the slow a at the resonance's own
scale (black, `strong_axis_cutoff`), and a* and the catalogue a_res moved into the
osculating frame of a_f by <a> - a(lambda_dot). a* and a_res belong to the mean motion, a_f to the osculating a, and
the constant offset between them (up to 0.08 au for the plutinos) would put a libration outside its own resonance.
The size of the shift is a number of the right column (ctx_a_mean_au - ctx_a_from_mean_longitude_rate_au).

(i) is the centre of sigma: sigma_f low-passed over W_c = libration_centre_cycles x P_cyc (the analysis'
`geometry.centre`, column `geo_centre_window_yr`) drawn over sigma_f itself, with the centre's turns and its
largest monotone pass when it is a revolution or more. A centre that still oscillates there oscillates with a period longer than W_c: for
the Hildas (153) that is the forced wobble at the period of varpi - varpi_J (2.6 kyr), not a smoothing error.
(j) shows where that wobble comes from: the centres of the other members of the multiplet,
c_s = c0 + smooth_Wc(Delta_s) with Delta_s = sigma_s - sigma0 from the saved varpi of the body and the
planets (planar members of the same order), the accepted subresonance's own measured centre and the
owner's centre; each minus its circular mean. A member whose line is flat is the one that librates.
When a subresonance or an owner is accepted, its angle (filtered at the cutoff) is also drawn on (b).
(c) draws sigma_dot / 2 omega0 on a symlog axis (linear inside +-1, the speed at the separatrix) and +-2 omega0 over
each accepted episode with that episode's omega0; without an episode the theoretical omega0, thin. sigma_dot = 0 is a
solid line in the colour of the centre (`style.RATE_ZERO`): a libration oscillates about it, sigma_dot changing sign at
each turning point; in the portraits (k), (l) it is the level of the centres (sigma_c, 0), marked + per accepted
episode. (h) is on a log axis, so that h of a deep libration and of a fast circulation are both in view.

The members' and the owner's centres, the strip (b′) and the per-regime nu of (c) are drawing aids; every
number printed in the text column is taken from `analysis.summary_row()` / `numbers_row()` by its CSV column,
so the CSV and the figure agree. Nothing drawn here feeds a status.
Colours: one palette (`style`): a quantity has the same colour in every panel. The text side lives in
`mmr_verdict_panel`, the geometry in `style`.
"""

from __future__ import annotations

import itertools
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from matplotlib import patheffects
from matplotlib.ticker import FuncFormatter, NullLocator, ScalarFormatter

from resonances.plotting.base import BasePlotter, time_unit
from resonances.plotting.mmr_verdict_panel import (  # noqa: F401  (re-exported: the API of this module)
    TEXT_BLOCKS,
    NumberLine,
    TextBlock,
    draw_explained,
    draw_text,
    format_cell,
    text_blocks,
)
from resonances.plotting.style import (  # noqa: F401
    AXIS_EXACT,
    AXIS_NOMINAL,
    CENTRE,
    EDGE_SPREAD_STYLE,
    ENVELOPE_PERCENTILES,
    EPISODE_ALPHA,
    EPISODE_BEFORE_JUMP_HATCH,
    EPISODE_COLOURS,
    FRAME,
    INK,
    LAYOUT_V1,
    MARK_COLOURS,
    MEMBER_COLOURS,
    MODEL,
    NOTE,
    OWNER,
    RAW,
    RAW_ALPHA,
    RAW_LIGHT_ALPHA,
    RATE_ZERO,
    REFERENCE,
    REJECTED,
    REJECTED_ALPHA,
    STATUS_COLOURS,
    SUBRESONANCE,
    ZONE_FILL,
    NumbersMode,
    OsculatingAxis,
    Rect,
    VerdictFigureLayout,
    resolve_plot_options,
    time_cmap,
)
from resonances.resonance import oscillations as osc
from resonances.data.const import GAUSSIAN_YEAR_DAYS
from resonances.resonance.analysis.filtering import circular_mean, is_finite, lowpass
from resonances.resonance.analysis.geometry import RejectReason, wrap_pi
from resonances.resonance.analysis.mmr.models import MMRAnalysis, MMRSeries, ResonanceKind, ResonanceSpec
from resonances.resonance.analysis.mmr.owner import FLAG_CATALOGUE, wrapped_angle as owner_wrapped_angle
from resonances.resonance.analysis.orbit_type import accepted_runs
from resonances.resonance.analysis.verdict import AngleSubtype, LibrationOrigin

TWO_PI = 2 * np.pi
PER_KYR = 1e3  # rad/yr -> rad/kyr
LINE_POINTS = 20000  # samples per drawn line; longer series are decimated (never a wrapped angle: see CYCLE_MIN_PX)
ENVELOPE_BINS = 1600  # raw series denser than this become a min-max band
MAX_TP_MARKERS = 80  # more turning points than this are not marked one by one
MAX_CYCLE_MEANS = 200
MAX_MEMBER_LINES = 6  # members of the multiplet drawn in (j) besides sigma0 (the flattest, or the best ranked)
CENTRES_AXIS_DEG = 230  # (j): the axis, so that the ticks +-180 stay clear of the tick margin
VIEW_PAD = 0.12  # fraction of the data range added above and below
FAIR_STEP_FRACTION = 1 / 4.2  # FAIR resolves its strips only with a step <= P_orb / 4.2 (= DRYRUN_FAIR_FRACTION)
DEFAULT_FILTER_ORDER = 2  # FILTER_ORDER, when the configuration cannot be read
LEGEND_CHAR_WIDTH = 0.56  # mean advance of a DejaVu Sans character, in units of the font size (wrapping estimate)
LEGEND_MIN_FONT_PT = 6.0  # ~10 px at 120 dpi, the smallest legible size: keep LEGEND short enough to fit at it
# Drawing rules (pixels of a time panel at the layout's dpi)
CYCLE_MIN_PX = 3.0  # sigma_f is not drawn in (b) when a cycle is narrower: the line would be a solid block
STRIP_CYCLE_PX = CYCLE_MIN_PX  # below this the strip (b′) under (b) shows ZOOM_CYCLES cycles on their own time axis
ZOOM_CYCLES = 5
ZOOM_MIN_STEPS = 8  # a window of the strip spans at least this many output steps
ZOOM_SUBSTEPS = 24  # points per output step of the predicted path drawn in the strip
STRIP_GAP = 0.5  # inches between the two windows of the strip (the y tick labels of the second)
H_FLOOR = 1e-3  # (h): the log axis stops here; h below it (or h <= 0) is a triangle at the bottom edge
H_VIEW_FACTOR = 3.0  # (h): the axis spans the values / this factor .. the values x this factor (log)
RATE_SYMLOG_VIEW = 2.2  # (c) on the symlog axis holds at least +-this (2 omega0): +-1 then clears the tick margin
RATE_SYMLOG_PAD = 10**0.5  # (c): the symlog axis extends this factor beyond the data outside +-1
SYMLOG_LABEL_GAP = 1.2  # (c): tick labels of the symlog axis at least this many label heights apart
EPISODE_LABEL_MIN_PX = 30  # '#id' above an episode only when its fill is at least this wide
EPISODE_LABEL_PAD = 0.03  # '#id' sits this fraction of the panel's height above its top edge (clear of the data)
REJECT_LETTER_MIN_PX = 8
REJECTED_STRIP = 0.14  # height of the rejected-runs strip, fraction of the panel
REJECT_LETTERS = {
    RejectReason.OVER_THE_TOP: 'T',
    RejectReason.TOO_FEW_TP: 'F',
    RejectReason.CENTRE_MOVES: 'C',
    RejectReason.NO_CLOSED_CYCLE: 'O',
}
JUMP_MERGE_FRACTION = 0.01  # jumps of a closer than this share of the record are one band
JUMP_LABEL_PAD_PT = 0.5  # white box around an 'N jumps of a' label [pt]
PANEL_LETTER_X = 0.004  # the panel letter's left edge, axes fraction
WINDOW_LABELS_MAX = 12  # (e): the R² of each force window is written when there are at most this many
WINDOW_LABEL_GAP_PT = 2.0  # (e): least free space between two window labels; closer ones merge [pt]
TP_SIZE = 18  # marker area (pt^2) of the turning points of the angle
CENTRE_TURN_SIZE = 36  # of the turns of the centre
RATE_DATA_MIN_SHARE = 0.25  # (c) without omega0 is not stretched to 0 if sigma_dot would fill less of the height
AXIS_VIEW_PAD = 0.25  # (d): margin around the a_f range
AXIS_REFERENCE_SPANS = 3.0  # (d): a* and a_res are brought into view when this close to a_f (in a_f ranges)
AXIS_LINE_WIDTH = 1.2  # (d): a* (dotted)
AXIS_NOMINAL_WIDTH = 1.3  # (d): a_res (dashed), the level the author reads first (23.10; thinner, variant B 24.10)
AXIS_LEVEL_ALPHA = 0.8  # (d): a* and a_res with their halos, slightly transparent over a_f (variant B, author 24.10/25.10)
AXIS_NOMINAL_HALO_WIDTH = 0.8  # (d): a_res's white halo (variant B, author 24.10)
AXIS_LEVEL_ZORDER = 8  # (d): a* and a_res over every line and dot of the panel (author, 23.10)
AXIS_HALO_WIDTH = 1.6  # (d): a*'s white halo, so it stays visible on a dense black a_f
# (d): (line width, halo beyond it) of each level of `axis_levels`
AXIS_LEVEL_STYLE = {'a*': (AXIS_LINE_WIDTH, AXIS_HALO_WIDTH), 'a_res': (AXIS_NOMINAL_WIDTH, AXIS_NOMINAL_HALO_WIDTH)}
EDGE_MARKER_PT = 4.5  # a level outside the axis: a triangle of this size just inside the edge
AXIS_EDGE_AT = (0.48, 0.52)  # (d): where in a regime the edge triangles of a* and a_res sit (apart when both are off)
EDGE_X_RAW = 0.993  # (d): x (axes fraction) of the edge triangles of an osculating a beyond the axis
ANGLE_VIEW_PAD = 0.1  # (a): margin around sigma_f and the osculating band, fraction of their span
ENVELOPE_VIEW_PAD = 0.04  # (d): margin around the osculating band (ENVELOPE_PERCENTILES), fraction of its span
OUTLIER_MARKERS_MAX = 8  # per edge: the samples beyond the band are marked by groups, the largest excess first
OUTLIER_MERGE_FRACTION = 0.02  # clipped samples closer than this share of the record are one group (one triangle)
FORCED_WAVE_LABEL = ' wave'  # (a): after '#id' of an accepted run that is a forced wave (F6), no episode of the angle
PORTRAIT_MAX_CYCLES = 12  # (l): cycles per episode, evenly spaced; an episode with no more is drawn whole
PORTRAIT_MAX_EPISODES = 5  # (l): the longest episodes only
PORTRAIT_SHARE_MAX = 3.0  # (k) and (l) share the union of their y views unless (k)'s is wider than this times the episodes'
PORTRAIT_VIEW_PAD = 0.08  # (k), (l): margin around the samples' sigma_dot, fraction of their span
MARGIN_FONT_DELTA = 0.5  # the compact legend right of the panels: font_note_pt minus this
MARGIN_LINE_SPACING = 1.2  # its line height, in units of its font size
MARGIN_KEY_IN = 0.16  # width of its colour keys (inches)
MARGIN_KEY_GAP_IN = 0.04  # between a key and its text (inches)
AXIS_STRONG_MIN_RATIO = 1.5  # (d): the black a is drawn only at a cutoff at least this many times P_c (else a_f is black)
AXIS_DENSE_CYCLE_PX = 12.0  # (d): a_f with cycles narrower than this is a band: the slow a is drawn black over it
AXIS_STRONG_CYCLES = 2.0  # (d): the black a at this many periods of the deciding episode when the response cutoff is P_c

TYPE_LABELS = {ResonanceKind.TWO_BODY: 'mmr2', ResonanceKind.THREE_BODY: 'mmr3', ResonanceKind.RETROGRADE: 'mmr2 retrograde'}

# Time panels, top to bottom: (letter, what it shows) — SPEC §10.2, the approved mock-up V1 and the author's P1-P15
PANELS = (
    ('a', 'sigma unwrapped: raw, sigma_f, centre, accepted episodes (#id), rejected runs (reason), TPs, jumps, exit'),
    ('b', 'sigma in a 360-degree window around the centre: every osculating sample, sigma_f, centre; owner/subresonance'),
    ('c', 'sigma_dot / 2 omega0 (symlog, linear inside +-1) with +-1 (separatrix) of each episode and nu(lambda_dot); slips'),
    ('d', 'a (au): raw, a_f, a* and a_res moved by <a> - a(lambda_dot), band +- half-width, cycle means'),
    ('e', 'a_dot at the response cutoff with R sin(sigma_f - phi0) + C per window (R^2) and per episode half'),
    ('f', 'e: raw band, filtered line, min - max'),
    ('g', 'i: raw band, filtered line, min - max'),
    ('h', 'h per cycle on a log axis (libration dots, revolutions open) with the separatrix zone [h_lo, h_hi] and h = 1'),
    ('i', 'the centre of sigma (sigma_f smoothed over W_c) over sigma_f, its turns and largest monotone pass'),
    ('j', 'centres of sigma0, the members of the multiplet and the owner (minus their means), or the kinematic terms'),
)
# (b′) under (b) only when the cycles are narrower than STRIP_CYCLE_PX: a few cycles on their own time axis (yr)
# Below the panels (`style.VerdictFigureLayout.bottom`): (k) the whole record's portrait and the time colour bar,
# (l) the accepted episodes' portrait, (m) FAIR, the legend (in the slot of (m) too when there is no FAIR)

LEGEND = (
    'Grey osculating; black filtered (pale dashed: filter edge, unused); ▲▼ at an edge: beyond the axis; values of ' '(c), (d), (h) right.',
    '(a) red: centre; ▼▲ turning points; fill #id: accepted episode (to a record edge it covers, else ±P/4 beyond its '
    'turns; hatched: before the last jump of a); grey strip: rejected run (T over the top, F few turning points, '
    'C centre moves, O no closed cycle); dotted / hatched: jumps of a; ×: exit.',
    '(b) σ mod 360°; purple dashed: owner, orange dash-dot: subresonance; ▼: window of (b′) (● samples, dashed: path).',
    '(c) dσ/dt / 2ω₀, symlog; red σ̇ = 0; blue dashed: ±2ω₀ per episode (thin: theory); dotted: ν(λ̇); ○ slips.',
    '(d) a: light grey osculating, mid grey a_f, black slow a; dotted a*, cyan dashed a_res (osculating frame); '
    'blue band ±Δa; red dots: mean a per cycle.',
    '(e) da/dt; blue: force law per window; #id: R² of the halves.  (f, g) e, i.',
    '(h) h per cycle: ● libration, ○ revolution; dashed h = 1; blue band: separatrix zone.',
    '(i) red: centre of σ_f, ▼▲ its turns; shaded: largest monotone pass ≥ 1 rev.  (j) centres minus means: red σ, pale '
    'members, orange subresonance, purple owner (kinematic: red slow σ, orange pϖ).',
    '(k) whole record, colour: time; (l) accepted episodes, ≤ 12 cycles; same scales; + centre.',
)


# --------------------------------------------------------------------------------------------- helpers


def _stride(n: int, target: int = LINE_POINTS) -> int:
    return max(1, n // target)


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _circular_mean(angles) -> float:
    """`filtering.circular_mean`, 0 for no finite angle (a window to draw in)."""
    mean = circular_mean(angles)
    return mean if np.isfinite(mean) else 0.0


def _window_low(centre) -> float:
    """Lower edge of the 360-degree window centred on the circular mean of the centre."""
    return _circular_mean(centre) - np.pi


def _right_angles(base_deg: float, inner: bool = False) -> np.ndarray:
    """Multiples of 90 deg inside the window [base, base + 360] (without its edges when `inner`)."""
    ticks = 90 * np.arange(np.ceil(base_deg / 90), np.floor((base_deg + 360) / 90) + 1)
    return ticks[(ticks > base_deg) & (ticks < base_deg + 360)] if inner else ticks


def angle_label(value_deg: float) -> str:
    """A wrapped angle's tick label in [0, 360) (the author's convention: 270, not -90)."""
    value = round(value_deg) % 360
    return f'{value:.0f}'


def _angle_formatter():
    return FuncFormatter(lambda value, _: angle_label(value))


def _wrap_window(values, low: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) - low) % TWO_PI + low


def _break_jumps(y: np.ndarray, limit: float) -> np.ndarray:
    """NaN where a wrapped line would jump across the window, so no vertical line is drawn."""
    y = np.array(y, dtype=float)
    if len(y) > 1:
        y[1:][np.abs(np.diff(y)) > limit] = np.nan
    return y


def _envelope(t: np.ndarray, y: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(t[0], t[-1], bins + 1)
    index = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, bins - 1)
    low = np.full(bins, np.nan)
    high = np.full(bins, np.nan)
    np.fmin.at(low, index, y)
    np.fmax.at(high, index, y)
    return 0.5 * (edges[:-1] + edges[1:]), low, high


def _raw(ax, t, y, color=RAW, alpha=RAW_ALPHA):
    """A raw series: a thin line when sparse, a min-max band when denser than the panel's pixels."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if len(t) <= ENVELOPE_BINS:
        ax.plot(t, y, color=color, lw=0.5, alpha=alpha, zorder=1)
        return
    centres, low, high = _envelope(t, y, ENVELOPE_BINS)
    ax.fill_between(centres, low, high, color=color, alpha=alpha, lw=0, zorder=1)


def _line(ax, t, y, unwrapped=None, max_step: float = np.pi, **kwargs):
    """A decimated line; a gap (NaN) anywhere in a decimation block stays a gap (no wrap-around strokes).

    With `unwrapped` (the same angle unwrapped, rad) the line also breaks where the angle turned by more than
    `max_step` between two drawn points: a wrapped line there joins samples a revolution apart."""
    t, y = np.asarray(t), np.asarray(y, dtype=float)
    k = _stride(len(t))
    shown = y[::k].copy()
    if k > 1:
        gaps = np.add.reduceat(np.isnan(y).astype(int), np.arange(0, len(y), k)) > 0
        shown[gaps] = np.nan
    if unwrapped is not None and len(shown) > 1:
        steps = np.abs(np.diff(np.asarray(unwrapped, dtype=float)[::k]))
        shown[1:][~(steps <= max_step)] = np.nan
    ax.plot(t[::k], shown, **kwargs)


def _view(ax, y, pad: float = VIEW_PAD, extra_top: float = 0.0):
    """y limits from the finite data (0.2-99.8 percentiles), padded."""
    limits = _view_limits(y, pad, extra_top)
    if limits is not None:
        ax.set_ylim(*limits)


def _view_limits(y, pad: float = VIEW_PAD, extra_top: float = 0.0) -> Optional[Tuple[float, float]]:
    y = _finite(y)
    if not len(y):
        return None
    low, high = np.percentile(y, [0.2, 99.8])
    span = high - low or max(abs(high), 1.0) * 1e-6
    return low - pad * span, high + (pad + extra_top) * span


def envelope_range(values) -> Optional[Tuple[float, float]]:
    """The ENVELOPE_PERCENTILES band of the finite `values`: what a panel shows of an osculating series; None when
    nothing is finite."""
    values = _finite(values)
    if not len(values):
        return None
    low, high = np.percentile(values, ENVELOPE_PERCENTILES)
    return float(low), float(high)


def outlier_markers(ax, x, values, low: float, high: float, colour: str = RAW, limit: int = OUTLIER_MARKERS_MAX) -> int:
    """Samples of `values` beyond the view [low, high]: one triangle at that edge per group of clipped samples (closer
    than OUTLIER_MERGE_FRACTION of the record are one group), at the time of its largest excess (`edge_marker`, data
    x); at most `limit` per edge, the largest excess first. Returns the number of triangles."""
    x, values = np.asarray(x, dtype=float), np.asarray(values, dtype=float)
    gap = OUTLIER_MERGE_FRACTION * (x[-1] - x[0]) if len(x) > 1 else 0.0
    drawn = 0
    for above in (True, False):
        excess = values - high if above else low - values
        index = np.flatnonzero(np.isfinite(excess) & (excess > 0))
        if not len(index):
            continue
        groups = np.split(index, np.flatnonzero(np.diff(x[index]) > gap) + 1)
        peaks = sorted((group[np.argmax(excess[group])] for group in groups), key=lambda i: -excess[i])[:limit]
        for i in peaks:
            edge_marker(ax, float(x[i]), above, colour, data_x=True)
        drawn += len(peaks)
    return drawn


def _edge_split(values, inside) -> Tuple[np.ndarray, np.ndarray]:
    """`values` where `inside` (the analysis window; the rest NaN) and outside it, in the filter's edge spread (the
    rest NaN). The edge part reaches two decimation strides (`_line`) into the window, so that the two drawn lines
    share a point after decimation and join."""
    values = np.asarray(values, dtype=float)
    inside = np.asarray(inside, dtype=bool)
    edge = ~inside
    if inside.any():
        first, last = np.flatnonzero(inside)[[0, -1]]
        overlap = 2 * _stride(len(values))
        if first > 0:
            edge[first : first + overlap + 1] = True
        if last < len(values) - 1:
            edge[max(last - overlap, 0) : last + 1] = True
    return np.where(inside, values, np.nan), np.where(edge, values, np.nan)


def _edge_line(ax, frame, values, inside=None, **kwargs) -> np.ndarray:
    """A filtered line over the whole record: solid inside the analysis window (`inside`, default the frame's), in
    EDGE_SPREAD_STYLE within FILTER_EDGE_TRIM_CUTOFFS cutoffs of the record's ends (the filter's edge spread, which
    the analysis does not read). Returns the values inside the window (what the panel's view is fitted to)."""
    shown, edge = _edge_split(values, frame.inside if inside is None else inside)
    _line(ax, frame.x, shown, **kwargs)
    if np.isfinite(edge).any():
        style = {**kwargs, **EDGE_SPREAD_STYLE, 'alpha': EDGE_SPREAD_STYLE['alpha'] * kwargs.get('alpha', 1.0)}
        _line(ax, frame.x, edge, **style)
    return shown


def edge_overflow_markers(ax, frame, values, inside=None, colour: str = INK) -> int:
    """Triangles at the edge where the edge spread of a filtered line (`_edge_line`) leaves the view: the view is
    fitted to the analysed window only, so a large end transient would otherwise vanish silently. Call after the
    y limits are set. Returns the number of triangles."""
    _, edge = _edge_split(values, frame.inside if inside is None else inside)
    low, high = sorted(ax.get_ylim())
    return outlier_markers(ax, frame.x, edge, low, high, colour)


def _note(ax, text: str, fontsize: float, y: float = 0.5):
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='center', fontsize=fontsize, color=NOTE)


def edge_marker(ax, x: float, above: bool, colour: str, data_x: bool = False):
    """A level beyond the axis: ▲ just inside the top edge (above) or ▼ inside the bottom edge, at x (axes fraction,
    or data when `data_x`). No words: what the colour stands for is in the legend."""
    height_in = ax.get_position().height * ax.figure.get_size_inches()[1]
    inset = 0.6 * EDGE_MARKER_PT / 72 / max(height_in, 1e-6)
    y = 1 - inset if above else inset
    transform = ax.get_xaxis_transform() if data_x else ax.transAxes
    ax.plot([x], [y], marker='^' if above else 'v', ms=EDGE_MARKER_PT, color=colour, transform=transform, clip_on=False, zorder=9)


def margin_legend(ax, layout: VerdictFigureLayout, items) -> list:
    """The compact legend right of a time panel, outside its axes (from `layout.margin_legend_x`), top down from the
    panel's top: per item a colour key matching the panel's line — ('line', colour, line style), ('band', colour,
    alpha) or None — and a short text with the number. Figure artists, so the panel's own texts stay data only.
    Returns the texts."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    fig = ax.figure
    width, height = fig.get_size_inches()
    font = layout.font_note_pt - MARGIN_FONT_DELTA
    step = MARGIN_LINE_SPACING * font / 72 / height
    x0, key = layout.margin_legend_x / width, MARGIN_KEY_IN / width
    y = ax.get_position().y1 - 0.5 * step
    texts = []
    for mark, text in items:
        if mark is not None and mark[0] == 'line':
            fig.add_artist(Line2D([x0, x0 + key], [y, y], color=mark[1], ls=mark[2], lw=1.1, transform=fig.transFigure))
        elif mark is not None:
            fig.add_artist(
                Rectangle((x0, y - 0.3 * step), key, 0.6 * step, facecolor=mark[1], alpha=mark[2], lw=0, transform=fig.transFigure)
            )
        texts.append(fig.text(x0 + key + MARGIN_KEY_GAP_IN / width, y, text, fontsize=font, va='center', ha='left', color='#333333'))
        y -= step
    return texts


def _value_text(ax, text: str, layout: VerdictFigureLayout):
    """A small number of the data (no words) at the top left, after the panel letter."""
    ax.text(0.04, 0.94, text, transform=ax.transAxes, ha='left', va='top', fontsize=layout.font_note_pt, color=NOTE, zorder=9)


def _right_labels(ax, items, fontsize: float):
    """Labels at the right end of horizontal lines (value in data units), pushed apart by one text line and
    kept inside the panel's height, so neither they nor the labels of the neighbouring panels overlap."""
    if not items:
        return
    low, high = ax.get_ylim()
    height_in = ax.get_position().height * ax.figure.get_size_inches()[1]
    sep = min(1.2 * fontsize / 72 / max(height_in, 1e-6), 1.0 / len(items))
    positions = sorted(((value - low) / (high - low), text, color) for value, text, color in items)
    lowest, highest = 0.5 * sep, 1 - 0.5 * sep
    ys = [min(max(p, lowest), highest) for p, _, _ in positions]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + sep)
    excess = ys[-1] - highest
    if excess > 0:
        ys = [y - excess for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - sep)
        if ys[0] < lowest:  # more labels than fit one line apart: spread evenly
            ys = list(np.linspace(lowest, highest, len(ys)))
    for y, (_, text, color) in zip(ys, positions):
        ax.text(1.004, y, text, transform=ax.transAxes, ha='left', va='center', fontsize=fontsize, color=color, clip_on=False)


def _inside_locator(margin: float, nbins: int = 4):
    """Ticks only inside [low + margin, high - margin] of the view (as fractions of its span): the tick labels of
    stacked panels never meet at a shared edge."""
    from matplotlib.ticker import MaxNLocator

    class InsideLocator(MaxNLocator):
        def tick_values(self, vmin, vmax):
            ticks = super().tick_values(vmin, vmax)
            lo, hi = sorted((vmin, vmax))
            pad = margin * (hi - lo)
            return ticks[(ticks >= lo + pad) & (ticks <= hi - pad)]

    return InsideLocator(nbins=nbins)


def _power_of_ten(values) -> int:
    values = _finite(values)
    peak = np.percentile(np.abs(values), 99.8) if len(values) else 0.0
    return int(np.floor(np.log10(peak))) if peak > 0 else 0


def _multiplier(power: int, unit: str) -> str:
    """'10^{-8} au/yr' in mathtext ('au/yr' alone for 10^0)."""
    return unit if power == 0 else rf'$10^{{{power}}}$ {unit}'


def turn_markers(ax, x, values, kinds, size: float, colour: str, white_edge: bool = False):
    """Turning points: ▼ on a maximum (kind > 0), ▲ on a minimum; one convention for the MMR and secular figures."""
    x, values, kinds = np.asarray(x, dtype=float), np.asarray(values, dtype=float), np.asarray(kinds)
    edge = {'edgecolors': 'white', 'linewidths': 0.5} if white_edge else {'linewidths': 0}
    for kind, marker in ((1, 'v'), (-1, '^')):
        chosen = kinds == kind
        ax.scatter(x[chosen], values[chosen], marker=marker, s=size, color=colour, zorder=6, **edge)


def episode_colours(runs) -> Dict[int, str]:
    """Colour of each accepted run (episodes of the last regime and before the last jump), in time order, cycled."""
    accepted = sorted((run for run in runs if run.accepted), key=lambda run: run.t_start_yr)
    return {run.episode_id: EPISODE_COLOURS[i % len(EPISODE_COLOURS)] for i, run in enumerate(accepted)}


def episode_spans(runs, record: Tuple[float, float], settings=None) -> Dict[int, Tuple[float, float]]:
    """(start, end) [yr] of the fill of each accepted run. An edge the run covers by the coverage rule
    (`geometry.covers_edges`, LIBRATION_COVERAGE_PERIODS; read only with `settings`) is filled up to the record's edge;
    otherwise the fill runs from the first to the last turning point extended by half the half-period (P/4) on each
    side, since the libration exists before its first and after its last turn. Never past the record's edges nor past
    the middle of the gap to a neighbouring run (accepted or not)."""
    from resonances.resonance.analysis.geometry import covers_edges

    ordered = sorted(runs, key=lambda run: run.t_start_yr)
    spans = {}
    for i, run in enumerate(ordered):
        if not run.accepted:
            continue
        try:
            start_edge, end_edge = covers_edges(run, settings) if settings is not None else (False, False)
        except (AttributeError, TypeError):  # a run without the coverage numbers: turning points only
            start_edge, end_edge = False, False
        reach = 0.25 * run.period_med_yr if is_finite(getattr(run, 'period_med_yr', None)) else 0.0
        low = 0.5 * (ordered[i - 1].t_end_yr + run.t_start_yr) if i > 0 else record[0]
        high = 0.5 * (run.t_end_yr + ordered[i + 1].t_start_yr) if i + 1 < len(ordered) else record[1]
        start = record[0] if start_edge else run.t_start_yr - reach
        end = record[1] if end_edge else run.t_end_yr + reach
        spans[run.episode_id] = (max(start, low, record[0]), min(end, high, record[1]))
    return spans


def shade_runs(
    ax,
    runs,
    accepted_ids,
    colours: Mapping[int, str],
    times,
    px_per_unit: float,
    layout: VerdictFigureLayout,
    spans: Optional[Mapping[int, Tuple[float, float]]] = None,
    forced_ids: Collection[int] = (),
):
    """Accepted episodes as a coloured fill (hatched when before the last jump of a) over their `spans` (yr,
    `episode_spans`; first to last turning point without) with '#id' just above the panel's top edge when wide
    enough (`ax` is the top panel: the label never covers the data or the panel letter; '#id wave' for a forced wave,
    `forced_ids`, which is no episode of the angle), the other runs as a grey strip at the bottom with the letter of
    their rejection reason (`REJECT_LETTERS`)."""
    spans = spans or {}
    for run in runs:
        own = (run.t_start_yr, run.t_end_yr)
        start, end = (float(times(v)) for v in (spans.get(run.episode_id, own) if run.episode_id in accepted_ids else own))
        width_px = (end - start) * px_per_unit
        middle = 0.5 * (start + end)
        if run.episode_id in accepted_ids:
            colour = colours.get(run.episode_id, EPISODE_COLOURS[0])
            hatch = EPISODE_BEFORE_JUMP_HATCH if getattr(run, 'before_last_jump', False) else None
            ax.axvspan(start, end, facecolor=colour, alpha=EPISODE_ALPHA, lw=0, zorder=0)
            if hatch:
                ax.axvspan(start, end, facecolor='none', edgecolor=colour, hatch=hatch, alpha=0.5, lw=0, zorder=0)
            if width_px >= EPISODE_LABEL_MIN_PX:
                ax.text(
                    middle,
                    1 + EPISODE_LABEL_PAD,
                    f'#{run.episode_id}' + (FORCED_WAVE_LABEL if run.episode_id in forced_ids else ''),
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='bottom',
                    fontsize=layout.font_note_pt,
                    color=colour,
                    fontweight='bold',
                    clip_on=False,
                    zorder=8,
                )
        else:
            ax.axvspan(start, end, ymin=0, ymax=REJECTED_STRIP, facecolor=REJECTED, alpha=REJECTED_ALPHA, lw=0, zorder=0)
            letter = REJECT_LETTERS.get(getattr(run, 'reject_reason', RejectReason.NONE), '')
            if letter and width_px >= REJECT_LETTER_MIN_PX:
                ax.text(
                    middle,
                    0.5 * REJECTED_STRIP,
                    letter,
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='center',
                    fontsize=layout.font_note_pt - 0.5,
                    color=INK,
                    fontweight='bold',
                    zorder=8,
                )


def jump_groups(times_yr: Sequence[float], span_yr: float, fraction: float = JUMP_MERGE_FRACTION) -> List[Tuple[float, float, int]]:
    """Jumps closer than `fraction` of the record merged: (first time, last time, count) per group."""
    times = sorted(float(t) for t in times_yr if np.isfinite(t))
    groups: List[List[float]] = []
    for t in times:
        if groups and t - groups[-1][-1] <= fraction * span_yr:
            groups[-1].append(t)
        else:
            groups.append([t])
    return [(group[0], group[-1], len(group)) for group in groups]


class _Frame:
    """Times and the time unit of one figure, the arrays every panel reads and the drawing options."""

    def __init__(self, analysis: MMRAnalysis, series: MMRSeries, layout: VerdictFigureLayout = LAYOUT_V1, options=None, settings=None):
        self.analysis = analysis
        self.series = series
        self.options = resolve_plot_options(options)
        self.settings = settings if settings is not None else _settings()  # the coverage rule of the episodes' fill
        self.t = np.asarray(series.t_yr, dtype=float)
        self.divisor, self.unit = time_unit(self.t[-1] - self.t[0])
        self.x = self.t / self.divisor
        prep = analysis.prep
        self.prep = prep
        self.window = prep.window if prep is not None else (self.t[0], self.t[-1])
        self.inside = (self.t >= self.window[0]) & (self.t <= self.window[1])
        self.raw_angle = analysis.sampling.angle_unwrapped if analysis.sampling is not None else series.angle_unwrapped_numpy
        geometry = analysis.geometry
        self.centre_t = geometry.centre.t_yr if geometry is not None else None
        self.centre = geometry.centre.centre_rad if geometry is not None else None
        reference = self.centre if self.centre is not None else (prep.angle_f[self.inside] if prep is not None else series.angle_wrapped)
        self.low = _window_low(reference)
        self.filter_order = analysis.filter_order or _filter_order()
        self.centre_lines = centre_lines(analysis, series, self.filter_order) if geometry is not None else []
        self.colours = episode_colours(geometry.runs) if geometry is not None else {}
        span = self.x[-1] - self.x[0]
        self.px_per_unit = layout.panels[2] * layout.dpi / span if span > 0 else 0.0

    def times(self, t_yr) -> np.ndarray:
        return np.asarray(t_yr, dtype=float) / self.divisor

    def cycle_px(self) -> float:
        """Mean width of one cycle of sigma_f (libration or revolution) in pixels of a time panel; inf when unknown."""
        prep, geometry = self.prep, self.analysis.geometry
        if prep is None:
            return float('inf')
        angle = np.asarray(prep.angle_f, dtype=float)[self.inside]
        angle = angle[np.isfinite(angle)]
        if len(angle) < 2:
            return float('inf')
        librations = 0.5 * len(geometry.turning_points.times_yr) if geometry is not None else 0.0
        cycles = librations + abs(angle[-1] - angle[0]) / TWO_PI
        width = (self.window[1] - self.window[0]) / self.divisor * self.px_per_unit
        return width / cycles if cycles > 0 else float('inf')


def _rate(prep) -> np.ndarray:
    """sigma_dot as the response read it (smoothed at the response cutoff), else sigma_dot_f."""
    rate = getattr(prep, 'rate_response', None)
    return np.asarray(rate if rate is not None else prep.rate_f, dtype=float)


def _adot(prep) -> np.ndarray:
    """a_dot as the force law read it (smoothed at the response cutoff), else a_dot_f."""
    adot = getattr(prep, 'adot_response', None)
    return np.asarray(adot if adot is not None else prep.adot_f, dtype=float)


def _deciding_episode(analysis: MMRAnalysis):
    """The run the verdict rests on: the decisive episode, else the longest accepted run (any regime); None without."""
    geometry = analysis.geometry
    if geometry is None:
        return None
    accepted = accepted_runs(geometry)
    chosen = [run for run in accepted if run.episode_id == analysis.decisive_episode_id]
    if chosen:
        return chosen[0]
    return max(accepted, key=lambda run: run.t_end_yr - run.t_start_yr) if accepted else None


def separatrix_omega0(analysis: MMRAnalysis) -> Tuple[float, str]:
    """(omega0 [rad/yr], its source) that scales the axis of (c): the deciding (or longest librating) episode's
    omega0 ('#id'), else the theoretical omega0 = |k| n sqrt(3 mu e^q) ('theory'), else (nan, ''). Never the
    potential fitted over the whole record: with a low R^2 it is noise (1999 UQ: 2.5 against 0.04 rad/kyr)."""
    physics = {ep.episode_id: ep for ep in analysis.episode_physics}
    episode = _deciding_episode(analysis)
    ordered = [episode] if episode is not None else []
    if analysis.geometry is not None:
        others = sorted(
            (run for run in analysis.geometry.runs if run.accepted and run is not episode), key=lambda run: run.t_start_yr - run.t_end_yr
        )
        ordered += others
    for run in ordered:
        ep = physics.get(run.episode_id)
        if ep is not None and is_finite(ep.omega0_rad_yr) and ep.omega0_rad_yr > 0:
            return float(ep.omega0_rad_yr), f'#{run.episode_id}'
    strength = analysis.strength
    if strength is not None and is_finite(strength.omega0_theory_rad_yr) and strength.omega0_theory_rad_yr > 0:
        return float(strength.omega0_theory_rad_yr), 'theory'
    return float('nan'), ''


def _half_width(analysis: MMRAnalysis) -> float:
    accepted = {run.episode_id for run in analysis.geometry.runs if run.accepted} if analysis.geometry is not None else set()
    values = _finite([ep.half_width_au for ep in analysis.episode_physics if ep.episode_id in accepted])
    return float(np.median(values)) if len(values) else float('nan')


def _aliased(analysis: MMRAnalysis) -> bool:
    """The sampling verdict is aliased_*: the samples are stroboscopic, a portrait of them is an illusion."""
    return analysis.sampling is not None and str(analysis.sampling.verdict).startswith('aliased')


def _is_alias(analysis: MMRAnalysis) -> bool:
    return str(analysis.verdict.angle_subtype) == str(AngleSubtype.ALIAS)


# --------------------------------------------------------------------------------------------- header


def draw_header(
    fig, analysis: MMRAnalysis, layout: VerdictFigureLayout, series: Optional[MMRSeries] = None, stamp: Optional[str] = None
) -> None:
    """Name, resonance, type and the orbit's type (`verdict.orbit_type`: tadpole L4, horseshoe, interior asymmetric
    libration about 72°, ...); the record line. The status is printed once, in the plaque of the right column."""
    width, height = fig.get_size_inches()
    x, y, w, h = layout.header
    top = y + h
    title = (
        f'{analysis.series_name}   {analysis.resonance.key}   [{TYPE_LABELS.get(analysis.resonance.kind, str(analysis.resonance.kind))}]'
    )
    orbit = orbit_type_text(analysis)
    if orbit:
        title += f'   {orbit}'
    if analysis.parent:
        title += f'   subresonance of {analysis.parent}'
    fig.text(x / width, top / height, title, fontsize=layout.font_title_pt, fontweight='bold', va='top', ha='left')
    if series is not None:
        fig.text(x / width, (top - 0.36) / height, record_line(analysis, series), fontsize=layout.text_font_pt, color='#333333', va='top')
    if stamp:
        fig.text(
            (x + w) / width,
            top / height,
            stamp,
            fontsize=layout.font_letter_pt,
            va='top',
            ha='right',
            color='#b2182b',
            fontweight='bold',
        )


def orbit_type_text(analysis: MMRAnalysis) -> str:
    """The orbit's type for the header: `verdict.orbit_type`, else the 1:1 regime of a co-orbital (rsp_coorb_regime)."""
    orbit = str(getattr(analysis.verdict, 'orbit_type', '') or '')
    if orbit:
        return orbit
    law = getattr(analysis.response, 'coorbital', None) if analysis.response is not None else None
    return f'1:1 {law.regime}' if law is not None and law.regime else ''


def record_line(analysis: MMRAnalysis, series: MMRSeries) -> str:
    """How the record was made, one line (the old title): span, integrator, step, rows, cutoff, epoch, FAIR."""
    regime = series.regime
    span = series.t_yr[-1] - series.t_yr[0]
    divisor, unit = time_unit(span)
    parts = [f'span {span / divisor:.4g} {unit} {regime.direction}']
    if regime.integrator:
        days = f' ({regime.dt / TWO_PI * GAUSSIAN_YEAR_DAYS:.1f} d)' if is_finite(regime.dt) else ''
        parts.append(f'{regime.integrator} dt {regime.dt:g}{days}' if is_finite(regime.dt) else regime.integrator)
    parts.append(f'step {float(np.median(np.diff(series.t_yr))):.4g} yr')
    parts.append(f'{len(series.t_yr)} rows')
    if analysis.prep is not None:
        parts.append(f'cutoff {analysis.prep.cutoff.cutoff_yr:.4g} yr ({analysis.prep.cutoff.binding})')
        if is_finite(analysis.prep.response_cutoff_yr):
            parts.append(f'response cutoff {analysis.prep.response_cutoff_yr:.4g} yr')
    if is_finite(regime.epoch_jd):
        parts.append(f'epoch JD {regime.epoch_jd:.1f}')
    reason = fair_unavailable_reason(analysis, series)
    parts.append('FAIR yes' if reason is None else f'FAIR no ({reason})')
    return '  |  '.join(parts)


# --------------------------------------------------------------------------------------------- time panels


def _style_axis(ax, letter: str, ylabel: str, layout: VerdictFigureLayout):
    """Ticks, the y label, the panel letter, and a full frame (panels stand apart, `panel_gap`)."""
    ax.tick_params(labelsize=layout.font_ticks_pt, length=2.5, pad=1.5)
    ax.set_ylabel(ylabel, fontsize=layout.font_axis_pt, labelpad=2)
    box = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 0.6}
    ax.text(
        PANEL_LETTER_X,
        0.94,
        f'({letter})',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=layout.font_letter_pt,
        fontweight='bold',
        zorder=9,
        bbox=box,
    )
    ax.grid(alpha=0.15, lw=0.5)
    ax.yaxis.set_major_locator(_inside_locator(layout.tick_margin))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME)
        spine.set_linewidth(0.6)


def _shade_episodes(ax, frame: _Frame, layout: VerdictFigureLayout):
    geometry = frame.analysis.geometry
    if geometry is None:
        return
    accepted = {run.episode_id for run in geometry.runs if run.accepted}
    spans = episode_spans(geometry.runs, (float(frame.t[0]), float(frame.t[-1])), frame.settings)
    forced = getattr(geometry, 'forced_wave_ids', frozenset())
    shade_runs(ax, geometry.runs, accepted, frame.colours, frame.times, frame.px_per_unit, layout, spans, forced)


def merge_overlapping_labels(
    spans: Sequence[Tuple[float, float]], width_of: Callable[[Tuple[float, float]], float], gap: float = 0.0
) -> List[Tuple[float, float]]:
    """Labels centred over `spans` [(start, end)] whose text boxes overlap merged into one label over their joint span,
    in time order: a merged label is measured again (`width_of(span)`, its own text) and merged on with its neighbours
    until no two boxes overlap. `gap`: the least free space between two boxes, in the units of the spans."""
    merged: List[Tuple[float, float]] = []
    for span in sorted(spans):
        merged.append(span)
        while len(merged) > 1:
            (s0, e0), (s1, e1) = merged[-2], merged[-1]
            right = 0.5 * (s0 + e0) + 0.5 * width_of((s0, e0))
            left = 0.5 * (s1 + e1) - 0.5 * width_of((s1, e1))
            if right + gap <= left:
                break
            merged[-2:] = [(min(s0, s1), max(e0, e1))]
    return merged


def _text_width_pt(text: str, size: float) -> float:
    """Width of `text` in the default sans font at `size` [pt]."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    return TextToPath().get_text_width_height_descent(text, FontProperties(size=size), ismath=False)[0]


def _jumps_text(count: int) -> str:
    return f'{count} jumps of a'


def _draw_jumps(ax, frame: _Frame, layout: VerdictFigureLayout):
    """Jumps of a: one dotted line each, or one hatched band 'N jumps of a' for jumps closer than 1 % of the record.
    Band labels whose text boxes would overlap become one label over their joint span, counting every jump in it
    (`merge_overlapping_labels`; TG2 4E-1M-3 100 kyr read '3 jum2 ju2 jumps of a')."""
    geometry = frame.analysis.geometry
    if geometry is None or not geometry.jumps:
        return
    _draw_action_steps(ax, frame, layout, [jump for jump in geometry.jumps if jump.action_step])
    jumps = [jump for jump in geometry.jumps if not jump.action_step]
    if not jumps:
        return
    span = frame.t[-1] - frame.t[0]
    min_width = 2.0 / frame.px_per_unit if frame.px_per_unit > 0 else 0.0  # a band is at least 2 px wide
    bands = []
    for first, last, count in jump_groups([jump.time_yr for jump in jumps], span):
        start, end = float(frame.times(first)), float(frame.times(last))
        if count == 1:
            ax.axvline(start, color=REFERENCE, ls=':', lw=0.8, zorder=2)
            continue
        middle, half = 0.5 * (start + end), max(0.5 * (end - start), 0.5 * min_width)
        ax.axvspan(middle - half, middle + half, facecolor='none', edgecolor=REFERENCE, hatch='xxx', lw=0.4, alpha=0.6, zorder=2)
        bands.append((start, end))
    times = frame.times([jump.time_yr for jump in jumps])

    def count_in(piece: Tuple[float, float]) -> int:
        return int(np.count_nonzero((times >= piece[0]) & (times <= piece[1])))

    units_per_pt = layout.dpi / 72 / frame.px_per_unit if frame.px_per_unit > 0 else 0.0
    pad = 2 * JUMP_LABEL_PAD_PT * units_per_pt  # the white box around each label

    def width(piece: Tuple[float, float]) -> float:
        return _text_width_pt(_jumps_text(count_in(piece)), layout.font_note_pt) * units_per_pt + pad

    for start, end in merge_overlapping_labels(bands, width):
        ax.text(
            0.5 * (start + end),
            REJECTED_STRIP + 0.02,
            _jumps_text(count_in((start, end))),
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=layout.font_note_pt,
            color=REFERENCE,
            zorder=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': JUMP_LABEL_PAD_PT},
        )


def _draw_action_steps(ax, frame: _Frame, layout: VerdictFigureLayout, steps) -> None:
    """G2a cuts (`geometry.ActionStep`): a dotted line at the middle of the transitional half-cycle, like a jump of a,
    its half-cycle shaded lightly and labelled 'action step (G2a)'."""
    for jump in steps:
        start, end = (float(x) for x in frame.times([jump.zone_start_yr, jump.zone_end_yr]))
        middle = float(frame.times(jump.time_yr))
        ax.axvspan(start, end, facecolor=REFERENCE, edgecolor='none', alpha=0.08, zorder=1)
        ax.axvline(middle, color=REFERENCE, ls=':', lw=0.8, zorder=2)
        ax.text(
            middle,
            REJECTED_STRIP + 0.02,
            'action step (G2a)',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=layout.font_note_pt,
            color=REFERENCE,
            zorder=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': JUMP_LABEL_PAD_PT},
        )


def regimes(frame: _Frame) -> List[Tuple[float, float]]:
    """The analysis window split at the jumps of a that cut the record (merged as in (a)) [yr]."""
    geometry = frame.analysis.geometry
    start, end = float(frame.t[0]), float(frame.t[-1])
    if geometry is None:
        return [(start, end)]
    cuts = [jump.time_yr for jump in geometry.jumps if not jump.kick]
    pieces, edge = [], start
    for first, last, _ in jump_groups(cuts, end - start):
        if first > edge:
            pieces.append((edge, first))
        edge = max(edge, last)
    pieces.append((edge, end))
    return [piece for piece in pieces if piece[1] > piece[0]]


def draw_angle_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(a) sigma unwrapped with the centre, episodes, turning points, jumps and the exit."""
    analysis, prep, geometry = frame.analysis, frame.prep, frame.analysis.geometry
    _shade_episodes(ax, frame, layout)
    if frame.raw_angle is not None and prep is not None:
        _raw(ax, frame.x, frame.raw_angle)
    elif frame.raw_angle is not None:  # nothing filtered: the raw angle is the line to read
        _line(ax, frame.x, frame.raw_angle, color=INK, lw=0.8)
    if prep is not None:
        _line(ax, frame.x, prep.angle_f, color=INK, lw=0.9, zorder=3)
    if frame.centre is not None:
        _line(ax, frame.times(frame.centre_t), frame.centre, color=CENTRE, lw=1.0, zorder=4)
    if geometry is not None:
        _turning_points(ax, frame, geometry.turning_points)
        _draw_jumps(ax, frame, layout)
        if geometry.exit is not None and prep is not None:
            j = min(np.searchsorted(frame.t, geometry.exit.time_yr), len(frame.t) - 1)
            ax.scatter([frame.x[j]], [prep.angle_f[j]], marker='x', s=30, color=REFERENCE, zorder=6, linewidths=1.3)
    low, high = angle_view(prep.angle_f if prep is not None else None, frame.raw_angle)
    ax.set_ylim(low, high)
    if frame.raw_angle is not None:
        outlier_markers(ax, frame.x, frame.raw_angle, low, high)
    _style_axis(ax, 'a', 'σ, rad', layout)
    if analysis.sampling is None:
        _note(ax, 'unwrap not computed', layout.font_note_pt)


def angle_view(filtered, raw, pad: float = ANGLE_VIEW_PAD) -> Tuple[float, float]:
    """y limits of (a): sigma_f (0.2-99.8 percentiles) and the osculating sigma's band (ENVELOPE_PERCENTILES), so the
    whole grey band is in view, padded by `pad` of the span; the samples beyond are `outlier_markers`."""
    ranges = [
        r
        for r in (_view_limits(filtered, pad=0.0) if filtered is not None else None, envelope_range(raw) if raw is not None else None)
        if r
    ]
    if not ranges:
        return -1.0, 1.0
    low, high = min(r[0] for r in ranges), max(r[1] for r in ranges)
    span = high - low or max(abs(high), 1.0) * 1e-6
    return low - pad * span, high + pad * span


def alias_text(analysis: MMRAnalysis) -> str:
    """The one-line ALIAS plaque of (b): the step, the measured period of sigma (slope of the predictive unwrap) and
    the revolutions of the predictive unwrap, e.g. 'ALIAS  step 5.0012 yr ≈ P_σ 5.0015 yr · 3999 rev' (from
    `analysis.sampling` / `context`; the rest is in the right column)."""
    sampling, context = analysis.sampling, analysis.context
    step = f'step {sampling.step_yr:.5g} yr' if sampling is not None else ''
    rate = context.rate_measured_rad_yr if context is not None else float('nan')
    period = f'P_σ {TWO_PI / abs(rate):.5g} yr' if is_finite(rate) and rate != 0 else ''
    revs = f'{sampling.revs_predictive:.0f} rev' if sampling is not None and is_finite(sampling.revs_predictive) else ''
    return 'ALIAS  ' + ' · '.join(part for part in (' ≈ '.join(p for p in (step, period) if p), revs) if part)


def _turning_points(ax, frame: _Frame, tp):
    if tp is None or not len(tp.times_yr) or len(tp.times_yr) > MAX_TP_MARKERS:
        return
    turn_markers(ax, frame.times(tp.times_yr), tp.values_rad, tp.kinds, TP_SIZE, REFERENCE)


def _filtered_angle(frame: _Frame, angle) -> np.ndarray:
    """An angle (mod 2 pi) low-passed at the record's cutoff as a unit vector, unwrapped [rad]."""
    cutoff = frame.prep.cutoff.cutoff_yr if frame.prep is not None else None
    angle = np.asarray(angle, dtype=float)
    if not is_finite(cutoff):
        return np.unwrap(angle)
    return _circular_smooth(frame.t, angle, cutoff, frame.filter_order)


def draw_wrapped_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(b) sigma in a 360-degree window around the centre: every osculating sample, sigma_f (when its cycles are
    resolved), the centre, the owner / subresonance angle filtered at the cutoff; the ALIAS plaque. Cycles too dense
    for (b) are shown by the strip (b′) under it (`draw_strip`)."""
    low = frame.low
    points = np.degrees(_wrap_window(frame.series.angle_wrapped, low))
    ax.scatter(frame.x, points, s=0.5, color=RAW, alpha=0.5, linewidths=0, rasterized=True, zorder=3)
    cycle_px = frame.cycle_px()
    if frame.prep is not None and cycle_px >= CYCLE_MIN_PX:
        wrapped = _break_jumps(np.degrees(_wrap_window(frame.prep.angle_f, low)), 180)
        _line(ax, frame.x, wrapped, unwrapped=frame.prep.angle_f, color=INK, lw=0.5, zorder=5)
    if frame.centre is not None and cycle_px >= CYCLE_MIN_PX:
        centre = _break_jumps(np.degrees(_wrap_window(frame.centre, low)), 180)
        _line(ax, frame.times(frame.centre_t), centre, unwrapped=frame.centre, color=CENTRE, lw=0.9, zorder=4)
    for line in frame.centre_lines:  # the accepted subresonance or owner: its own angle at the cutoff, under sigma_f
        if line.angle_wrapped is None or len(line.angle_wrapped) != len(frame.t):
            continue
        colour, style = (SUBRESONANCE, '-.') if line.role == 'subresonance' else (OWNER, '--')
        smooth = _filtered_angle(frame, line.angle_wrapped)
        overlay = _break_jumps(np.degrees(_wrap_window(smooth, low)), 180)
        _line(ax, frame.x, overlay, unwrapped=smooth, color=colour, ls=style, lw=0.8, alpha=0.5, zorder=2)  # key: (j), legend
    base = np.degrees(low)
    ax.set_ylim(base, base + 360)
    _style_axis(ax, 'b', 'σ, °', layout)
    ticks = _right_angles(base, inner=True)
    edge = 360 * layout.tick_margin
    ax.set_yticks(ticks[(ticks > base + edge) & (ticks < base + 360 - edge)])
    ax.yaxis.set_major_formatter(_angle_formatter())
    if _is_alias(frame.analysis):  # one short plaque where the illusion is (the numbers are in the right column)
        ax.text(
            0.995,
            0.94,
            alias_text(frame.analysis),
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=layout.font_note_pt + 0.5,
            color=CENTRE,
            fontweight='bold',
            zorder=11,
            bbox={'facecolor': 'white', 'edgecolor': CENTRE, 'lw': 0.6, 'pad': 1.2},
        )


def needs_strip(frame: _Frame) -> bool:
    """The strip (b′) is drawn when the cycles of sigma_f are narrower than STRIP_CYCLE_PX (then (b) cannot show them)."""
    return frame.prep is not None and frame.cycle_px() < STRIP_CYCLE_PX


def zoom_window(frame: _Frame) -> Optional[Tuple[float, float]]:
    """[t0, t1] (yr) of the first window of the strip: ZOOM_CYCLES cycles (at least ZOOM_MIN_STEPS steps) in the middle
    of the deciding episode, else in the middle of the window with the circulation period 2 pi / median |sigma_dot_f|."""
    episode = _deciding_episode(frame.analysis)
    if episode is not None and is_finite(episode.period_med_yr):
        return _centred_window(frame, 0.5 * (episode.t_start_yr + episode.t_end_yr), episode.period_med_yr)
    return _circulation_window(frame, frame.window[0], frame.window[1])


def _centred_window(frame: _Frame, middle: float, period: float) -> Tuple[float, float]:
    step = float(np.median(np.diff(frame.t))) if len(frame.t) > 1 else 0.0
    half = 0.5 * max(ZOOM_CYCLES * period, ZOOM_MIN_STEPS * step)
    return middle - half, middle + half


def _circulation_window(frame: _Frame, t0: float, t1: float) -> Optional[Tuple[float, float]]:
    """ZOOM_CYCLES cycles in the middle of [t0, t1] with the period 2 pi / median |sigma_dot_f| there."""
    mask = (frame.t >= t0) & (frame.t <= t1) & frame.inside
    rate = _finite(np.abs(np.asarray(frame.prep.rate_f, dtype=float)[mask]))
    rate_med = float(np.median(rate)) if len(rate) else float('nan')
    if not is_finite(rate_med) or rate_med <= 0:
        return None
    return _centred_window(frame, 0.5 * (t0 + t1), TWO_PI / rate_med)


def strip_windows(frame: _Frame) -> List[Tuple[float, float]]:
    """[(t0, t1)] (yr) of the strip: `zoom_window` and, when the record ends in another regime after a jump of a and
    the deciding episode lies before it, ZOOM_CYCLES cycles in the middle of that last regime."""
    windows = [window for window in (zoom_window(frame),) if window is not None]
    pieces = regimes(frame)
    episode = _deciding_episode(frame.analysis)
    if len(pieces) > 1 and episode is not None and episode.t_end_yr < pieces[-1][0]:
        last = _circulation_window(frame, *pieces[-1])
        if last is not None:
            windows.append(last)
    return windows


def draw_strip(fig, rect: Rect, frame: _Frame, layout: VerdictFigureLayout, wrapped_ax) -> list:
    """The strip (b′) in `rect` (inches): one axes per window of `strip_windows`, each on its own time axis (yr) with
    the samples, the path between them by the predictive unwrap (dashed: a whole revolution may pass between two
    samples) and sigma_f (broken where it turns by more than pi between samples). The windows are marked ▼ on (b)."""
    windows = strip_windows(frame)
    width, height = fig.get_size_inches()
    x, y, w, h = rect
    sub_w = (w - STRIP_GAP * (len(windows) - 1)) / max(len(windows), 1)
    axes = []
    for i, (t0, t1) in enumerate(windows):
        mask = (frame.t >= t0) & (frame.t <= t1)
        if np.count_nonzero(mask) < 2:
            continue
        ax = fig.add_axes(((x + i * (sub_w + STRIP_GAP)) / width, y / height, sub_w / width, h / height))
        letter = 'b′' if len(windows) == 1 else f'b′{i + 1}'
        _draw_strip_window(ax, frame, layout, mask, letter)
        axes.append(ax)
        middle = frame.times(0.5 * (t0 + t1))
        wrapped_ax.plot(
            [middle], [1.0], marker='v', ms=6, color=REFERENCE, transform=wrapped_ax.get_xaxis_transform(), clip_on=False, zorder=9
        )
        if len(windows) > 1:
            wrapped_ax.text(
                middle + 6 / frame.px_per_unit,
                1.0,
                str(i + 1),
                transform=wrapped_ax.get_xaxis_transform(),
                ha='left',
                va='top',
                fontsize=layout.font_note_pt,
                color=REFERENCE,
                zorder=9,
            )
    return axes


def _draw_strip_window(ax, frame: _Frame, layout: VerdictFigureLayout, mask, letter: str):
    t = frame.t[mask]
    angle_f = np.asarray(frame.prep.angle_f, dtype=float)[mask]
    low = _circular_mean(angle_f) - np.pi
    spread = np.ptp(_finite(np.degrees(_wrap_window(angle_f, low)))) if len(_finite(angle_f)) else 360.0
    if frame.raw_angle is not None:
        raw = np.asarray(frame.raw_angle, dtype=float)[mask]
        fine = np.linspace(t[0], t[-1], ZOOM_SUBSTEPS * (len(t) - 1) + 1)
        path = np.degrees(_wrap_window(np.interp(fine, t, raw), low))
        ax.plot(fine, _break_jumps(path, 180), color=RAW, lw=0.6, ls='--', zorder=1)
        steps = _finite(np.abs(np.diff(raw)))
        spread = max(spread, np.degrees(steps.max())) if len(steps) else spread  # a revolution between samples
    ax.plot(t, np.degrees(_wrap_window(np.asarray(frame.series.angle_wrapped, dtype=float)[mask], low)), 'o', ms=1.8, color=RAW, zorder=2)
    shown = _break_jumps(np.degrees(_wrap_window(angle_f, low)), 180)
    shown[1:][~(np.abs(np.diff(angle_f)) <= np.pi)] = np.nan  # no line across a revolution between two samples
    ax.plot(t, shown, color=INK, lw=0.8, zorder=3)
    base = np.degrees(low)
    ax.set_xlim(t[0], t[-1])
    _style_axis(ax, letter, 'σ, °' if letter in ('b′', 'b′1') else '', layout)
    ax.texts[-1].set_text(f'({letter}) t, yr')
    if spread < 180:  # a libration: its own range, not the whole circle
        middle_deg, half = base + 180.0, max(0.75 * spread, 30.0)
        ax.set_ylim(middle_deg - half, middle_deg + half)
        ax.yaxis.set_major_locator(_inside_locator(layout.tick_margin + 0.05, nbins=3))
    else:
        ax.set_ylim(base, base + 360)
        ticks = _right_angles(base, inner=True)
        edge = 360 * layout.tick_margin
        ax.set_yticks(ticks[(ticks > base + edge) & (ticks < base + 360 - edge)])
    ax.yaxis.set_major_formatter(_angle_formatter())
    ax.xaxis.set_major_locator(_inside_locator(0.06, nbins=6))
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.tick_params(axis='x', direction='in', pad=-9, labelsize=layout.font_ticks_pt - 1, length=2)
    for label in ax.get_xticklabels():
        label.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 0.2})


def _nu_by_regime(frame: _Frame) -> List[Tuple[float, float, float]]:
    """(t0, t1, nu [rad/yr]) of each regime: nu(<a>) + k (lambda_dot_regime - n(<a>)), the context's
    rate_from_mean_longitude evaluated with the regime's own mean lambda_dot; one regime: the context's value."""
    from resonances.resonance.analysis.mmr.context import mean_longitude_rate

    context, series = frame.analysis.context, frame.series
    if context is None:
        return []
    pieces = regimes(frame)
    full = context.rate_from_mean_longitude_rad_yr if is_finite(context.rate_from_mean_longitude_rad_yr) else context.rate_expected_rad_yr
    if len(pieces) == 1 or series.mean_longitude is None:
        return [(frame.t[0], frame.t[-1], full)] if is_finite(full) else []
    result = []
    k = frame.analysis.resonance.body_coefficient
    for t0, t1 in pieces:
        mask = (frame.t >= t0) & (frame.t <= t1)
        rate = mean_longitude_rate(frame.t[mask], np.asarray(series.mean_longitude)[mask], np.asarray(series.a)[mask])
        if is_finite(rate):
            result.append((t0, t1, context.rate_expected_rad_yr + k * (rate - context.mean_motion_rad_yr)))
    return result


def episode_omega0(analysis: MMRAnalysis) -> List[Tuple[float, float, float]]:
    """(t_start, t_end [yr], omega0 [rad/yr]) of every accepted episode with a measured omega0, in time order."""
    if analysis.geometry is None:
        return []
    physics = {ep.episode_id: ep for ep in analysis.episode_physics}
    result = []
    for run in sorted(accepted_runs(analysis.geometry), key=lambda run: run.t_start_yr):
        ep = physics.get(run.episode_id)
        if ep is not None and is_finite(ep.omega0_rad_yr) and ep.omega0_rad_yr > 0:
            result.append((float(run.t_start_yr), float(run.t_end_yr), float(ep.omega0_rad_yr)))
    return result


def _level(ax, value: float, low: float, high: float, x0: float, x1: float, at: float = 0.5, **style):
    """A horizontal level over [x0, x1] (data), or a triangle at the edge when it is beyond the axis, at the fraction
    `at` of the span (levels that would share the edge are told apart by it)."""
    if low <= value <= high:
        ax.hlines(value, x0, x1, **style)
    else:
        edge_marker(ax, x0 + at * (x1 - x0), value > high, style['color'], data_x=True)


def draw_rate_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(c) sigma_dot at the response cutoff. With an omega0 (`separatrix_omega0`: the deciding episode's, else the
    theory's) as sigma_dot / 2 omega0 on a symlog axis, linear inside +-1 (the speed at the separatrix), logarithmic
    outside: a libration stays inside +-1 and a circulation of any speed is in view on every figure. Over each accepted
    episode +-2 omega0 of that episode (dashed; without an episode +-1 of the theory, thin), nu(lambda_dot) per regime
    and the slips. Without an omega0, sigma_dot in rad/kyr, stretched to 0 only when sigma_dot would still fill
    RATE_DATA_MIN_SHARE of the height. A level beyond the axis is a triangle at the edge."""
    analysis = frame.analysis
    if frame.prep is None:
        _style_axis(ax, 'c', 'dσ/dt\nrad/kyr', layout)
        _note(ax, 'sigma_dot not computed', layout.font_note_pt)
        return
    omega0, source = separatrix_omega0(analysis)
    if is_finite(omega0):
        _draw_rate_symlog(ax, frame, layout, 2 * omega0, source)
        return
    _style_axis(ax, 'c', 'dσ/dt\nrad/kyr', layout)
    rate = _edge_line(ax, frame, _rate(frame.prep) * PER_KYR, color=INK, lw=0.8)
    limits = _view_limits(rate)
    low, high = limits if limits is not None else (-1.0, 1.0)
    if high - low >= RATE_DATA_MIN_SHARE * (max(high, 0.0) - min(low, 0.0)):
        low, high = min(low, 0.0), max(high, 0.0)
        rate_zero(ax)
    ax.set_ylim(low, high)
    edge_overflow_markers(ax, frame, _rate(frame.prep) * PER_KYR)
    _rate_marks(ax, frame, low, high, PER_KYR)


def _draw_rate_symlog(ax, frame: _Frame, layout: VerdictFigureLayout, level: float, source: str):
    """(c) as sigma_dot / `level` (= 2 omega0, rad/yr) on the symlog axis (`draw_rate_panel`)."""
    ratio = _edge_line(ax, frame, _rate(frame.prep) / level, color=INK, lw=0.8)
    ax.set_yscale('symlog', linthresh=1.0, linscale=1.0)
    finite = _finite(ratio)
    lo, hi = np.percentile(finite, [0.2, 99.8]) if len(finite) else (0.0, 0.0)
    low = -RATE_SYMLOG_VIEW if lo >= -RATE_SYMLOG_VIEW else lo * RATE_SYMLOG_PAD
    high = RATE_SYMLOG_VIEW if hi <= RATE_SYMLOG_VIEW else hi * RATE_SYMLOG_PAD
    ax.set_ylim(low, high)
    edge_overflow_markers(ax, frame, _rate(frame.prep) / level)
    rate_zero(ax)
    _separatrix_lines(ax, frame, low, high, level / 2 if source == 'theory' else float('nan'), scale=1 / level)
    _rate_marks(ax, frame, low, high, 1 / level)
    _style_axis(ax, 'c', 'dσ/dt\n/ 2ω₀', layout)
    margin_legend(
        ax,
        layout,
        [(('line', MODEL, '--'), f'2ω₀ {source}'), (None, f'{level * PER_KYR:.3g} rad/kyr'), (('line', RATE_ZERO['color'], '-'), 'σ̇ = 0')],
    )
    ax.set_yticks(symlog_ticks(ax, low, high, layout.tick_margin, layout.font_ticks_pt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _symlog_label(value)))


def rate_zero(ax, vertical: bool = False):
    """dsigma/dt = 0 (`style.RATE_ZERO`): horizontal in the rate panels and portraits, vertical when sigma_dot is on x."""
    (ax.axvline if vertical else ax.axhline)(0, **RATE_ZERO)


def centre_marks(ax, runs, low: float):
    """+ at (sigma_c, 0) of each accepted episode: its centre (`centre_mod_2pi_rad`) in the 360-degree window from
    `low`, where a libration's portrait closes around."""
    centres = [run.centre_mod_2pi_rad for run in runs if is_finite(run.centre_mod_2pi_rad)]
    if centres:
        x = np.degrees(_wrap_window(centres, low))
        ax.plot(x, np.zeros(len(x)), '+', color=RATE_ZERO['color'], ms=9, mew=1.6, zorder=6)


def symlog_ticks(ax, low: float, high: float, margin: float, label_pt: float = 0.0) -> List[float]:
    """0, +-1 and the decades +-10^k of the symlog axis of (c) inside the tick margin (axes fraction); with more than
    three decades on a side, every other one from the outermost. With `label_pt` (the tick font), a decade closer to
    the tick kept before it (outward from 0) than SYMLOG_LABEL_GAP labels is dropped, so the labels never overlap
    (W20 round D: 142694 41J-14 at +-10^8 in a 0.6-inch panel)."""
    to_axes = ax.transData + ax.transAxes.inverted()
    height_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72
    gap = SYMLOG_LABEL_GAP * label_pt / height_pt if label_pt > 0 and height_pt > 0 else 0.0
    ticks = [0.0]
    for sign in (-1.0, 1.0):
        decades = [sign * 10.0**k for k in range(0, 40) if low <= sign * 10.0**k <= high]
        decades = decades[:1] + (decades[1:][::-1][::2][::-1] if len(decades) > 4 else decades[1:])
        decades = [v for v in decades if margin <= to_axes.transform((0.0, v))[1] <= 1 - margin]
        last = to_axes.transform((0.0, 0.0))[1]
        for value in decades:  # outward from 0
            position = to_axes.transform((0.0, value))[1]
            if abs(position - last) >= gap:
                ticks.append(value)
                last = position
    return sorted(v for v in ticks if margin <= to_axes.transform((0.0, v))[1] <= 1 - margin)


def _symlog_label(value: float) -> str:
    if value == 0 or abs(value) < 1000:
        return f'{value:g}'
    return rf"${'-' if value < 0 else ''}10^{{{int(round(np.log10(abs(value))))}}}$"


def _rate_marks(ax, frame: _Frame, low: float, high: float, scale: float):
    """nu(lambda_dot) of each regime (dotted) and the slips (○ at 0), sigma_dot in rad/yr times `scale`."""
    for t0, t1, nu in _nu_by_regime(frame):
        _level(ax, nu * scale, low, high, frame.times(t0), frame.times(t1), color=REFERENCE, ls=':', lw=0.9)
    if frame.analysis.geometry is not None:
        slips = [slip.time_yr for slip in frame.analysis.geometry.slips if not slip.suspect]
        ax.plot(frame.times(slips), np.zeros(len(slips)), 'o', ms=4, mfc='none', mec=REFERENCE, zorder=5)


def _separatrix_lines(ax, frame: _Frame, low: float, high: float, theory_omega0: float, scale: float = PER_KYR):
    """+-2 omega0 dashed over each accepted episode with its own omega0; without an episode +-2 omega0 of the theory
    (finite `theory_omega0`) thin across the panel. Levels in rad/yr times `scale`."""
    episodes = episode_omega0(frame.analysis)
    for t0, t1, omega0 in episodes:
        for sign in (-1, 1):
            _level(ax, sign * 2 * omega0 * scale, low, high, frame.times(t0), frame.times(t1), color=MODEL, ls='--', lw=0.8)
    if not episodes and is_finite(theory_omega0):
        for sign in (-1, 1):
            _level(ax, sign * 2 * theory_omega0 * scale, low, high, frame.x[0], frame.x[-1], color=MODEL, ls='--', lw=0.5)


def osculating_shift(context) -> float:
    """<a> - a(lambda_dot) of the whole record [au]: what moves a level of the mean motion (a*, a_res) into the
    osculating frame of a_f; 0 when a(lambda_dot) or <a> is not measured (the levels are then drawn as they are)."""
    if context is None or not (is_finite(context.axis_from_mean_longitude_rate_au) and is_finite(context.a_mean_au)):
        return 0.0
    return float(context.a_mean_au - context.axis_from_mean_longitude_rate_au)


def axis_shifts(frame: _Frame) -> List[Tuple[float, float, float]]:
    """(t0, t1 [yr], <a> - a(lambda_dot) [au]) of each regime between the jumps of a (`regimes`), as nu of (c): the
    osculating offset follows e and i, which a jump changes. One regime, or no saved lambda: the context's value."""
    from resonances.resonance.analysis.mmr.context import axis_from_mean_longitude_rate

    pieces, series = regimes(frame), frame.series
    whole = osculating_shift(frame.analysis.context)
    if len(pieces) == 1 or series.mean_longitude is None:
        return [(float(frame.t[0]), float(frame.t[-1]), whole)]
    result = []
    for t0, t1 in pieces:
        mask = (frame.t >= t0) & (frame.t <= t1)
        a = np.asarray(series.a, dtype=float)[mask]
        axis = axis_from_mean_longitude_rate(frame.t[mask], np.asarray(series.mean_longitude)[mask], a)
        result.append((t0, t1, float(np.mean(a)) - axis if is_finite(axis) and len(a) else whole))
    return result


def axis_levels(analysis: MMRAnalysis, shift: float) -> List[Tuple[str, float, str, str]]:
    """The levels of (d) in the osculating frame: (name, a [au], colour, line style) of a* (dotted) and the catalogue
    a_res (dashed), each + `shift` (`axis_shifts`); a level that is not known is left out."""
    context = analysis.context
    if context is None:
        return []
    levels = (('a*', context.exact_axis_au, AXIS_EXACT, ':'), ('a_res', context.nominal_axis_au, AXIS_NOMINAL, '--'))
    return [(name, float(value) + shift, colour, style) for name, value, colour, style in levels if is_finite(value)]


def draw_axis_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(d) a on an absolute axis (au) in three levels: the osculating a light grey, a_f at the cutoff P_c mid grey, and
    in black a filtered at the resonance's own scale (`strong_axis_cutoff`; without one, a_f is the black line); a* and
    a_res in the osculating frame of each regime (`axis_shifts`, `axis_levels`) with the band +- half-width around a*,
    the mean a of every cycle. The filtered lines cover the whole record: in EDGE_SPREAD_STYLE outside the analysis
    window (`_edge_line`). The view (option `combined.osculating_a`): `full` (default) holds the osculating band
    (ENVELOPE_PERCENTILES) and a_f, the samples beyond are triangles at the edge at their time (`outlier_markers`);
    `clipped` holds a_f and the black a only (`_axis_view`), an osculating a beyond is one triangle at the edge;
    `hidden` as clipped, without the osculating a when it exceeds the view. a* and a_res are brought into view when
    within AXIS_REFERENCE_SPANS ranges of a_f; a level beyond the axis is a triangle at the edge. Their values are in
    the compact legend right of the panel (`margin_legend`)."""
    analysis = frame.analysis
    filtered = np.asarray(frame.prep.a_f if frame.prep is not None else frame.series.a, dtype=float)
    filtered_inside = np.where(frame.inside, filtered, np.nan)  # outside the window: the filter's edge spread
    raw = np.asarray(frame.series.a, dtype=float)
    strong_yr = strong_axis_cutoff(frame)
    strong = _strong_axis(frame, raw, strong_yr)
    pieces = [(t0, t1, axis_levels(analysis, shift)) for t0, t1, shift in axis_shifts(frame)]
    viewed = filtered_inside[frame.inside]
    if strong is not None:
        viewed = np.concatenate([viewed, strong[0][strong[1]]])
    values = [value for _, _, levels in pieces for _, value, _, _ in levels]
    low, high = _axis_view(viewed, raw, values)
    mode = frame.options['combined']['osculating_a']
    band = envelope_range(raw)
    if mode == OsculatingAxis.FULL and band is not None:
        span = max(band[1], high) - min(band[0], low)
        low, high = min(low, band[0] - ENVELOPE_VIEW_PAD * span), max(high, band[1] + ENVELOPE_VIEW_PAD * span)
    low, high = hold_levels(low, high, values)
    raw_finite = _finite(raw)  # the whole record: the osculating a is drawn over it, edges included
    raw_range = np.percentile(raw_finite, [0.2, 99.8]) if len(raw_finite) else None
    beyond = (raw_range is not None and raw_range[1] > high, raw_range is not None and raw_range[0] < low)
    if not (mode == OsculatingAxis.HIDDEN and any(beyond)):
        _raw(ax, frame.x, raw, alpha=RAW_LIGHT_ALPHA)
    if strong is not None:  # a_f as a band, lighter in the edge spread
        shown, edge = _edge_split(filtered, frame.inside)
        _raw(ax, frame.x, shown)
        _raw(ax, frame.x, edge, alpha=RAW_ALPHA * EDGE_SPREAD_STYLE['alpha'])
        _edge_line(ax, frame, strong[0], inside=strong[1], color=INK, lw=1.0, zorder=3)
    else:
        _edge_line(ax, frame, filtered, color=INK, lw=0.9, zorder=3)
    _axis_levels(ax, frame, pieces, low, high)
    _cycle_means(ax, frame, filtered_inside)
    ax.set_ylim(low, high)
    if strong is not None:
        edge_overflow_markers(ax, frame, strong[0], inside=strong[1])
    else:
        edge_overflow_markers(ax, frame, filtered)
    if mode == OsculatingAxis.FULL:
        outlier_markers(ax, frame.x, raw, low, high)
    else:
        for above, exceeded in zip((True, False), beyond):
            if exceeded:
                edge_marker(ax, EDGE_X_RAW, above, RAW)
    _style_axis(ax, 'd', 'a, au', layout)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    margin_legend(ax, layout, axis_legend(frame, pieces[-1], strong_yr if strong is not None else float('nan'), high - low))


def _axis_levels(ax, frame: _Frame, pieces, low: float, high: float):
    """a* and a_res of each regime (a triangle at the edge when beyond the axis) and the band +- half-width round a*."""
    half_width = _half_width(frame.analysis)
    for t0, t1, levels in pieces:
        x0, x1 = frame.times(t0), frame.times(t1)
        exact = next((value for name, value, _, _ in levels if name == 'a*'), None)
        if exact is not None and is_finite(half_width):
            ax.fill_between([x0, x1], exact - half_width, exact + half_width, color=MODEL, alpha=0.10, lw=0, zorder=0)
        for (name, value, colour, style), at in zip(levels, AXIS_EDGE_AT):
            width, halo_extra = AXIS_LEVEL_STYLE[name]
            halo = [patheffects.withStroke(linewidth=width + halo_extra, foreground='white', alpha=AXIS_LEVEL_ALPHA)]
            line = dict(color=colour, ls=style, lw=width, alpha=AXIS_LEVEL_ALPHA, zorder=AXIS_LEVEL_ZORDER, path_effects=halo)
            _level(ax, value, low, high, x0, x1, at, **line)


def strong_axis_cutoff(frame: _Frame) -> float:
    """Cutoff [yr] of the black a of (d), from the resonance's own scale. With an accepted episode: the response cutoff
    (max(P_c, MMR_RESPONSE_CUTOFF_CYCLES x P_cyc): the a whose derivative the force law of (e) reads) when it is at least
    AXIS_STRONG_MIN_RATIO x P_c; else, when the cycles are too dense for a_f to be a line (narrower than
    AXIS_DENSE_CYCLE_PX), AXIS_STRONG_CYCLES periods of the deciding episode (the slow a under the libration).
    Without an episode, the centre-scale window W_c. nan when the choice is not AXIS_STRONG_MIN_RATIO x P_c or more,
    or not needed (then a_f is the black line)."""
    analysis = frame.analysis
    prep, geometry = getattr(analysis, 'prep', None), analysis.geometry
    cutoff = getattr(getattr(prep, 'cutoff', None), 'cutoff_yr', float('nan'))
    if not is_finite(cutoff):
        return float('nan')
    episode = _deciding_episode(analysis)
    if episode is None:
        value = geometry.centre.window_yr if geometry is not None and geometry.centre is not None else float('nan')
    else:
        value = getattr(prep, 'response_cutoff_yr', float('nan'))
        if not (is_finite(value) and value >= AXIS_STRONG_MIN_RATIO * cutoff):
            value = AXIS_STRONG_CYCLES * episode.period_med_yr if frame.cycle_px() < AXIS_DENSE_CYCLE_PX else float('nan')
    return float(value) if is_finite(value) and value >= AXIS_STRONG_MIN_RATIO * cutoff else float('nan')


def _strong_axis(frame: _Frame, raw: np.ndarray, cutoff_yr: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The osculating a low-passed at `cutoff_yr` over the whole record and the mask of its own analysis window: the
    frame's window less FILTER_EDGE_TRIM_CUTOFFS x `cutoff_yr` at each record end (outside it, the filter's edge
    spread); None without a cutoff, when the window is empty, or when the grid cannot be filtered."""
    if not is_finite(cutoff_yr):
        return None
    trim = getattr(frame.settings, 'filter_edge_trim_cutoffs', 1.0) * cutoff_yr
    keep = frame.inside & (frame.t >= frame.t[0] + trim) & (frame.t <= frame.t[-1] - trim)
    if np.count_nonzero(keep) < 2:
        return None
    try:
        smooth = lowpass(frame.t, raw, cutoff_yr, frame.filter_order)
    except ValueError:
        return None
    return smooth, keep


def axis_legend(frame: _Frame, piece, strong_yr: float, span: float) -> list:
    """Items of the compact legend of (d): a_res, a* (their mean-motion values; the lines are moved by the shift),
    +- half-width, the mean osculating a, the shift <a> - a(lambda_dot) of the last regime, and the cutoffs of the
    grey and the black a. Decimals from the axis span."""
    context, (_, _, levels) = frame.analysis.context, piece
    digits = int(np.clip(np.ceil(-np.log10(span)) + 2, 3, 7)) if is_finite(span) and span > 0 else 4
    shift = next((value - context.exact_axis_au for name, value, _, _ in levels if name == 'a*'), float('nan')) if context else float('nan')
    items = []
    names = {'a_res': ('a_res', getattr(context, 'nominal_axis_au', None)), 'a*': ('a*', getattr(context, 'exact_axis_au', None))}
    for name, _, colour, style in levels:
        label, value = names[name]
        items.append((('line', colour, style), f'{label} {value:.{digits}f}'))
    half_width = _half_width(frame.analysis)
    if is_finite(half_width):
        items.append((('band', MODEL, 0.25), f'±Δa {half_width:.2g}'))
    if context is not None and is_finite(context.a_mean_au):
        items.append((None, rf'$\langle a\rangle$ {context.a_mean_au:.{digits}f}'))
    if is_finite(shift):
        items.append((None, rf'$\langle a\rangle - a(\dot\lambda)$ {shift:+.2g}'))
    cutoff = getattr(getattr(frame.prep, 'cutoff', None), 'cutoff_yr', float('nan'))
    if is_finite(strong_yr):
        items.append((('band', RAW, RAW_ALPHA), f'a_f {cutoff:.3g} yr'))
        items.append((('line', INK, '-'), f'a {strong_yr:.3g} yr'))
    elif is_finite(cutoff):
        items.append((('line', INK, '-'), f'a_f {cutoff:.3g} yr'))
    return items


def _axis_view(filtered, raw, levels: Sequence[float]) -> Tuple[float, float]:
    """View of (d): the range of a_f (of the raw a when a_f is all NaN) padded by AXIS_VIEW_PAD, widened to hold
    the levels within AXIS_REFERENCE_SPANS ranges of it."""
    inside = _finite(filtered)
    band = envelope_range(raw)  # without a_f: the band of the raw a, never its extremes (a spike would set the view)
    lo, hi = (float(inside.min()), float(inside.max())) if len(inside) else (band if band is not None else (0.0, 1.0))
    spread = max(hi - lo, 1e-12 * max(abs(hi), 1.0))
    low, high = lo - AXIS_VIEW_PAD * spread, hi + AXIS_VIEW_PAD * spread
    for value in levels:
        if lo - AXIS_REFERENCE_SPANS * spread <= value <= hi + AXIS_REFERENCE_SPANS * spread:
            low, high = min(low, value - 0.1 * spread), max(high, value + 0.1 * spread)
    return low, high


def hold_levels(low: float, high: float, levels: Sequence[float]) -> Tuple[float, float]:
    """The final view of (d) widened once more to hold a* and a_res within AXIS_REFERENCE_SPANS spans of the view
    itself (the osculating band included), so a_res is a line, not an edge triangle, whenever the view is wider than a_f
    alone (author 2026-10-23: 2897 35S-4, a_res 6 a_f ranges below)"""
    span = high - low
    if not (np.isfinite(span) and span > 0):
        return low, high
    for value in levels:
        if np.isfinite(value) and low - AXIS_REFERENCE_SPANS * span <= value <= high + AXIS_REFERENCE_SPANS * span:
            low, high = min(low, value - 0.05 * span), max(high, value + 0.05 * span)
    return low, high


def _cycle_means(ax, frame: _Frame, values):
    """Mean of `values` over each cycle of sigma_f (maximum to maximum), red dots at the cycle's middle."""
    geometry = frame.analysis.geometry
    if geometry is None or frame.prep is None:
        return
    tp = geometry.turning_points
    maxima = np.asarray(tp.indices)[np.asarray(tp.kinds) > 0]
    if 2 <= len(maxima) <= MAX_CYCLE_MEANS:
        mids = [(frame.x[i] + frame.x[j]) / 2 for i, j in zip(maxima[:-1], maxima[1:])]
        means = [np.mean(_finite(values[i:j])) if len(_finite(values[i:j])) else np.nan for i, j in zip(maxima[:-1], maxima[1:])]
        ax.plot(mids, means, 'o', ms=2.8, color=CENTRE, zorder=4)


def draw_adot_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(e) a_dot_f with the force law R sin(sigma_f - phi0) + C fitted per window, and the R² of the episode halves; for
    a 1:1 angle a_dot_f with the numbers of its own law (`coorbital.CoorbitalLaw`) instead."""
    analysis, prep = frame.analysis, frame.prep
    if prep is None:
        _style_axis(ax, 'e', r'$\dot a$', layout)
        _note(ax, 'da/dt not computed', layout.font_note_pt)
        return
    adot = np.where(frame.inside, _adot(prep), np.nan)
    power = _power_of_ten(adot)
    scale = 10.0**-power
    label = r'$\dot a$' + '\n' + _multiplier(power, 'au/yr')
    _edge_line(ax, frame, _adot(prep) * scale, color=INK, lw=0.8, zorder=2)  # the edge spread dashed
    _view(ax, adot * scale, extra_top=0.35)
    edge_overflow_markers(ax, frame, _adot(prep) * scale)
    response = analysis.response
    law = getattr(response, 'coorbital', None) if response is not None else None
    if law is not None and response.force_law == 'coorbital':
        # a 1:1 angle is judged by its own law, not by the harmonic windows (which stay as columns): no sine is drawn
        _style_axis(ax, 'e', label, layout)
        numbers = [f'{value:.2f}' if is_finite(value) else '–' for value in (law.r2, law.r2_h1, law.r2_h2, law.scale)]
        _value_text(ax, f'R² {numbers[0]} ({numbers[1]} / {numbers[2]})   K/K_th {numbers[3]}', layout)
        return
    windows = response.windows if response is not None else []
    labels = [_draw_window(ax, frame, window, i, len(windows), scale) for i, window in enumerate(windows)]
    if len(windows) <= WINDOW_LABELS_MAX:
        _window_labels(ax, frame, [label for label in labels if label is not None], layout)
    _episode_halves(ax, frame, layout)
    _style_axis(ax, 'e', label, layout)
    if response is None:
        _note(ax, 'force law not computed', layout.font_note_pt, y=0.9)


def _draw_window(ax, frame: _Frame, window, i: int, n: int, scale: float) -> Optional[Tuple[float, float, float]]:
    """The force law of one window on its central half (windows overlap by half; the first and last to the edges);
    returns (start, end [display unit], R²) of its label, None without a finite R²."""
    quarter = (window.t_end_yr - window.t_start_yr) / 4
    start = window.t_start_yr if i == 0 else window.t_start_yr + quarter
    end = window.t_end_yr if i == n - 1 else window.t_end_yr - quarter
    mask = (frame.t >= start) & (frame.t < end) & frame.inside
    model = window.amplitude_au_yr * np.sin(frame.prep.angle_f[mask] - window.phi0_rad) + window.offset_au_yr
    _line(ax, frame.x[mask], model * scale, color=MODEL, lw=1.0, alpha=0.9, zorder=3)
    if not is_finite(window.r2):
        return None
    return float(frame.times(start)), float(frame.times(end)), float(window.r2)


def _r2_text(values: Sequence[float]) -> str:
    """'0.74' for one window, '0.74–0.81' (lowest–highest) for windows whose labels were merged."""
    low, high = f'{min(values):.2f}', f'{max(values):.2f}'
    return low if low == high else f'{low}–{high}'


def _window_labels(ax, frame: _Frame, labels: Sequence[Tuple[float, float, float]], layout: VerdictFigureLayout):
    """The R² of each window under its law; labels whose boxes would overlap become one 'lowest–highest' label over
    their joint span (`merge_overlapping_labels`, the rule of the jump labels in (a))."""
    units_per_pt = layout.dpi / 72 / frame.px_per_unit if frame.px_per_unit > 0 else 0.0
    centres = np.array([0.5 * (start + end) for start, end, _ in labels])

    def values(piece: Tuple[float, float]) -> List[float]:
        inside = (centres >= piece[0]) & (centres <= piece[1])
        return [label[2] for label, keep in zip(labels, inside) if keep]

    def width(piece: Tuple[float, float]) -> float:
        return _text_width_pt(_r2_text(values(piece)), layout.font_note_pt) * units_per_pt

    spans = [(float(c), float(c)) for c in centres]  # a label sits over the middle of its window's drawn half
    for start, end in merge_overlapping_labels(spans, width, gap=WINDOW_LABEL_GAP_PT * units_per_pt):
        ax.text(
            0.5 * (start + end),
            0.03,
            _r2_text(values((start, end))),
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=layout.font_note_pt,
            color=MODEL,
        )


def _episode_halves(ax, frame: _Frame, layout: VerdictFigureLayout):
    """'#id ½R² h1 / h2' over the middle of each accepted episode, in its colour; labels that would overlap the panel
    letter or each other are moved right, then to a second row, then left out."""
    geometry = frame.analysis.geometry
    if geometry is None:
        return
    physics = {ep.episode_id: ep for ep in frame.analysis.episode_physics}
    width_px = layout.panels[2] * layout.dpi
    char_px = 0.6 * layout.font_note_pt / 72 * layout.dpi
    x0, span = frame.x[0], frame.x[-1] - frame.x[0]
    letter_px = PANEL_LETTER_X * width_px + _text_width_pt('(e)', layout.font_letter_pt) / 72 * layout.dpi
    rows_end = [letter_px, letter_px]  # right edge (px) of the last label on each row: the first starts after '(e)'
    for episode in sorted(accepted_runs(geometry), key=lambda run: run.t_start_yr):
        ep = physics.get(episode.episode_id)
        if ep is None:
            continue
        colour = frame.colours.get(episode.episode_id, EPISODE_COLOURS[0])
        middle = frame.times(0.5 * (episode.t_start_yr + episode.t_end_yr))
        ax.axvline(middle, color=colour, lw=0.5, ls=':')
        r2 = (ep.response.force_r2_h1, ep.response.force_r2_h2) if ep.response is not None else (float('nan'),) * 2
        halves = [f'{value:.2f}' if is_finite(value) else '–' for value in r2]
        measured = any(is_finite(v) for v in r2)
        text = f'#{episode.episode_id} ½R² {halves[0]} / {halves[1]}' if measured else f'#{episode.episode_id} R² –'
        size = len(text) * char_px
        centre_px = (middle - x0) / span * width_px if span > 0 else 0.0
        for row, y in enumerate((0.97, 0.84)):
            left = max(centre_px - size / 2, rows_end[row] + char_px, 0.0)
            if left + size <= width_px:
                rows_end[row] = left + size
                ax.text(
                    x0 + (left + size / 2) / width_px * span,
                    y,
                    text,
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='top',
                    fontsize=layout.font_note_pt,
                    color=colour,
                )
                break


def draw_element_panel(ax, frame: _Frame, layout: VerdictFigureLayout, raw, filtered, letter: str, label: str, fmt: str):
    """(f) e, (g) i: the osculating band, the filtered line, 'min – max' of the filtered series in the window."""
    raw = np.asarray(raw, dtype=float)
    _raw(ax, frame.x, raw, alpha=RAW_ALPHA if filtered is not None else 0.9)
    shown = raw[frame.inside]
    if filtered is not None:
        filtered = np.asarray(filtered, dtype=float)
        _line(ax, frame.x, filtered, color=INK, lw=1.0, zorder=3)
        values = _finite(filtered[frame.inside])
        shown = np.concatenate([shown, values])
        if len(values):
            _value_text(ax, f'{fmt.format(values.min())} – {fmt.format(values.max())}', layout)
    _view(ax, shown, extra_top=0.25)
    _style_axis(ax, letter, label, layout)
    ax.yaxis.set_major_locator(_inside_locator(layout.tick_margin, nbins=3))


def draw_eccentricity_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(f) e."""
    prep = frame.prep
    draw_element_panel(ax, frame, layout, frame.series.e, prep.e_f if prep is not None else None, 'f', 'e', '{:.4g}')


def draw_inclination_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(g) i in degrees."""
    prep = frame.prep
    filtered = np.degrees(prep.inc_f) if prep is not None else None
    draw_element_panel(ax, frame, layout, np.degrees(frame.series.inc), filtered, 'g', 'i, °', '{:.4g}°')


def draw_energy_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(h) h per cycle relative to the separatrix on a log axis, so that deep librations (h << 1) and fast
    circulations (h >> 1) are all in view with the separatrix zone [h_lo, h_hi] (a light band) and h = 1 (dashed):
    libration dots, revolutions open, no labels (`h_view`). h below H_FLOOR (or h <= 0) is a triangle at the bottom
    edge; h = inf (a cycle without a potential top) a triangle at the top edge."""
    energy = frame.analysis.energy
    cycles = energy.cycles if energy is not None else []
    if not cycles:
        _style_axis(ax, 'h', 'h', layout)
        _note(ax, 'h not computed: ' + ('no energy analysis' if energy is None else 'no cycles'), layout.font_note_pt)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        return
    zone = getattr(energy, 'zone', None)
    zone = (zone.h_lo, zone.h_hi) if zone is not None and is_finite(zone.h_lo) and is_finite(zone.h_hi) else None
    x = frame.times([0.5 * (cycle.t_start_yr + cycle.t_end_yr) for cycle in cycles])
    h = np.array([cycle.h for cycle in cycles], dtype=float)
    rotation = np.array([cycle.rotation for cycle in cycles], dtype=bool)
    low, high = h_view(h, zone)
    ax.set_yscale('log')
    if zone is not None:
        ax.axhspan(max(zone[0], low), zone[1], color=ZONE_FILL, lw=0, zorder=0)
    ax.axhline(1, color=REFERENCE, lw=0.9, ls='--', zorder=2)
    shown = (h >= low) & (h <= high)  # NaN, inf, and h below the floor are not points
    ax.plot(x[~rotation & shown], h[~rotation & shown], 'o', ms=3.4, color=INK, zorder=3)
    ax.plot(x[rotation & shown], h[rotation & shown], 'o', ms=3.8, mfc='none', mec=INK, mew=1.0, zorder=4)
    for mask, above in ((h > high, True), (h < low, False)):
        if np.any(mask):
            inset = 0.6 * EDGE_MARKER_PT / 72 / max(ax.get_position().height * ax.figure.get_size_inches()[1], 1e-6)
            y = np.full(int(mask.sum()), 1 - inset if above else inset)
            ax.plot(x[mask], y, '^' if above else 'v', ms=4, color=REFERENCE, transform=ax.get_xaxis_transform(), clip_on=False, zorder=4)
    ax.set_ylim(low, high)
    _style_axis(ax, 'h', 'h', layout)
    keys = [(('line', REFERENCE, '--'), 'h = 1 separatrix')]
    if zone is not None:
        keys.append((('band', ZONE_FILL, 1.0), f'zone {zone[0]:.3g}–{zone[1]:.3g}'))
    margin_legend(ax, layout, keys)
    to_axes = ax.transData + ax.transAxes.inverted()
    decades = [10.0**k for k in range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)]
    inside = [v for v in decades if layout.tick_margin <= to_axes.transform((0.0, v))[1] <= 1 - layout.tick_margin]
    ax.set_yticks(_thin_around_one(inside))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))


def _thin_around_one(decades: Sequence[float], keep: int = 4) -> List[float]:
    """At most about `keep` decades of (h): those whose exponent is a multiple of a common step (h = 1 always)."""
    step = max(1, int(np.ceil((len(decades) - 1) / max(keep - 1, 1))))
    return [v for v in decades if round(np.log10(v)) % step == 0]


def h_view(h, zone: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    """The log axis of (h): every finite h above H_FLOOR, the zone and h = 1, widened by H_VIEW_FACTOR on both sides."""
    values = [float(v) for v in np.asarray(h, dtype=float) if np.isfinite(v) and v >= H_FLOOR]
    values += [1.0] + ([float(v) for v in zone if v > 0] if zone is not None else [])
    return max(H_FLOOR, min(values) / H_VIEW_FACTOR), max(values) * H_VIEW_FACTOR


def _relative_degrees(centre, low_reference) -> np.ndarray:
    """A centre track minus its circular mean, in degrees wrapped to +-180: a fixed centre is flat."""
    values = np.degrees(wrap_pi(np.asarray(centre, dtype=float) - _circular_mean(low_reference)))
    return _break_jumps(values, 180)


def centre_mode(analysis: MMRAnalysis) -> str:
    """What panel (j) shows: 'subresonance', 'owner', 'kinematic' or 'centre'."""
    subresonances, owner = analysis.subresonances, analysis.owner
    if subresonances is not None and subresonances.accepted is not None:
        return 'subresonance'
    if owner is not None and owner.accepted is not None:
        return 'owner'
    kinematic = analysis.verdict.libration_origin == LibrationOrigin.KINEMATIC
    if kinematic and analysis.slow is not None and analysis.resonance.body_varpi_coefficient:
        return 'kinematic'
    return 'centre'


# --------------------------------------------------------------------------------------------- members


@dataclass
class CentreLine:
    """One line of panel (j): the centre of an angle [rad] on the record's time grid.

    `role`: 'sigma0' (the catalogue angle), 'member' (another member of the multiplet), 'subresonance'
    (the accepted one) or 'owner'. `angle_wrapped` (accepted subresonance or owner only) is the raw angle
    mod 2 pi for the overlay on (b)."""

    key: str
    centre_rad: np.ndarray
    role: str
    angle_wrapped: Optional[np.ndarray] = None


def planar_members(spec: ResonanceSpec) -> List[ResonanceSpec]:
    """The other members of the catalogue angle's multiplet that differ from it in the varpi coefficients only: the
    same lambda part and the same node coefficients, the eccentricity coefficients (body and the angle's planets) of
    one sign summing to the same total (d'Alembert holds since the nodes are kept). For a nodal angle
    (`10E-17+0+5[N+2,NE+0]`) the members keep its nodes. None for a retrograde angle or one without eccentricity
    terms."""
    from resonances.resonance.analysis.mmr.subresonances import member_spec

    if spec.retrograde:
        return []
    total = spec.body_varpi_coefficient + sum(spec.planet_varpi_coefficients)
    if total == 0:
        return []
    sign, order, n = (1 if total > 0 else -1), abs(total), len(spec.planets)
    nodes = (spec.body_node_coefficient,) + tuple(spec.planet_node_coefficient(i) for i in range(n))
    members = []
    for parts in itertools.product(range(order + 1), repeat=1 + n):
        if sum(parts) != order:
            continue
        body, planets = sign * parts[0], tuple(sign * u for u in parts[1:])
        if body == spec.body_varpi_coefficient and planets == tuple(spec.planet_varpi_coefficients):
            continue
        members.append(member_spec(spec, (body,) + planets + nodes))
    return members


def varpi_offset(spec: ResonanceSpec, member: ResonanceSpec) -> Optional[Tuple[int, Tuple[int, ...]]]:
    """Delta_s = sigma_member - sigma0 as (coefficient of the body's varpi, of each planet's varpi); None when the
    lambda or node parts differ (a nodal member lives in another frame and is not drawn)."""
    same_lambda = member.planets == spec.planets and member.planet_coefficients == spec.planet_coefficients
    same_nodes = member.body_node_coefficient == spec.body_node_coefficient and all(
        member.planet_node_coefficient(i) == spec.planet_node_coefficient(i) for i in range(len(spec.planets))
    )
    if not (same_lambda and same_nodes and member.body_coefficient == spec.body_coefficient):
        return None
    planets = tuple(m - s for m, s in zip(member.planet_varpi_coefficients, spec.planet_varpi_coefficients))
    return member.body_varpi_coefficient - spec.body_varpi_coefficient, planets


def spec_angle(spec: ResonanceSpec, series: MMRSeries) -> Optional[np.ndarray]:
    """The angle of `spec` mod 2 pi from the saved longitudes of the body and the planets; None if any is missing
    (or for a retrograde angle, whose body longitudes are in another convention)."""
    planets = [series.planets.get(name) for name in spec.planets]
    if spec.retrograde or series.mean_longitude is None or any(p is None or p.mean_longitude is None for p in planets):
        return None
    angle = spec.body_coefficient * series.mean_longitude + spec.body_varpi_coefficient * series.varpi
    angle = angle + spec.body_node_coefficient * series.Omega
    for index, planet in enumerate(planets):
        angle = angle + spec.planet_coefficients[index] * planet.mean_longitude + spec.planet_varpi_coefficients[index] * planet.varpi
        angle = angle + spec.planet_node_coefficient(index) * planet.Omega
    return np.mod(angle, TWO_PI)


def _settings():
    """The AnalysisSettings of the configuration, None when it cannot be read (a figure is drawn without it too)."""
    try:
        from resonances.resonance.analysis.settings import settings_from_config

        return settings_from_config()
    except Exception:  # noqa: BLE001 - a figure is drawn without a readable configuration too
        return None


def _filter_order() -> int:
    """FILTER_ORDER of the configuration (the analysis' centre is smoothed with it)."""
    settings = _settings()
    return int(settings.filter_order) if settings is not None else DEFAULT_FILTER_ORDER


def _smooth(t_yr: np.ndarray, values: np.ndarray, window_yr: float, order: int) -> np.ndarray:
    try:
        return lowpass(t_yr, values, window_yr, order)
    except ValueError:  # a non-uniform grid or a window shorter than two steps: drawn unsmoothed
        return np.asarray(values, dtype=float)


def _circular_smooth(t_yr: np.ndarray, angle: np.ndarray, window_yr: float, order: int) -> np.ndarray:
    """Centre of an angle that may circulate: the argument of the smoothed unit vector."""
    cos = _smooth(t_yr, np.cos(angle), window_yr, order)
    sin = _smooth(t_yr, np.sin(angle), window_yr, order)
    return np.unwrap(np.arctan2(sin, cos))


def _flatness(centre: np.ndarray, inside: np.ndarray) -> float:
    values = np.unwrap(np.asarray(centre, dtype=float)[inside])
    values = values[np.isfinite(values)]
    return float(np.ptp(values)) if len(values) else float('inf')


def centre_lines(analysis: MMRAnalysis, series: MMRSeries, filter_order: Optional[int] = None) -> List[CentreLine]:
    """The lines of panel (j): sigma0's centre, the members of the multiplet, the accepted subresonance, the owner.

    Members are the scanned candidates when the subresonance step ran (best ranked first), else the planar
    members of the same order (flattest first), at most MAX_MEMBER_LINES; c_s = c0 + smooth_Wc(Delta_s). The
    accepted subresonance is drawn with its own measured centre when its derived row exists, and its raw angle
    is kept for (b). The accepted owner's centre is its angle smoothed over W_c. Empty without a centre."""
    geometry = analysis.geometry
    if geometry is None or geometry.centre is None:
        return []
    t = np.asarray(series.t_yr, dtype=float)
    c0 = np.asarray(geometry.centre.centre_rad, dtype=float)
    if len(c0) != len(t):
        return []
    order = DEFAULT_FILTER_ORDER if filter_order is None else filter_order
    window_yr = float(geometry.centre.window_yr)
    inside = (t >= geometry.window[0]) & (t <= geometry.window[1]) & np.isfinite(c0)
    spec = analysis.resonance
    subresonances = analysis.subresonances
    accepted = subresonances.accepted if subresonances is not None else None
    varpis = _smoothed_varpis(series, spec, t, window_yr, order)
    lines = [CentreLine(spec.key, c0, 'sigma0')] + _member_lines(analysis, c0, varpis, inside)
    if accepted is not None:
        line = _accepted_line(analysis, series, accepted.spec, c0, varpis, t)
        if line is not None:
            lines.append(line)
    owner = analysis.owner.accepted if analysis.owner is not None else None
    angle = spec_angle(owner.spec, series) if owner is not None else None
    key = owner.spec.key if owner is not None else ''
    if owner is not None and angle is None and FLAG_CATALOGUE in analysis.owner.flags:  # found on catalogue planet lines
        angle, key = owner_wrapped_angle(owner.spec, series), f'{key} catalogue λ (approx.)'
    if angle is not None:
        lines.append(CentreLine(key, _circular_smooth(t, angle, window_yr, order), 'owner', angle))
    return lines


def _offset_track(offset: Tuple[int, Tuple[int, ...]], varpis: Sequence[np.ndarray]) -> np.ndarray:
    """Delta_s(t) from its varpi coefficients and the (smoothed) varpi of the body and the planets [rad]."""
    return offset[0] * varpis[0] + sum(coefficient * varpi for coefficient, varpi in zip(offset[1], varpis[1:]))


def _member_lines(analysis: MMRAnalysis, c0: np.ndarray, varpis, inside: np.ndarray) -> List[CentreLine]:
    """c0 + smooth(Delta_s) of the scanned candidates (best ranked first) or, without a scan, of the planar
    members (flattest first); the accepted subresonance is drawn apart."""
    if varpis is None:
        return []
    spec, subresonances = analysis.resonance, analysis.subresonances
    scanned = subresonances is not None and bool(subresonances.candidates)
    accepted = subresonances.accepted.spec.key if subresonances is not None and subresonances.accepted is not None else None
    if scanned:
        candidates = [c.spec for c in sorted(subresonances.candidates, key=lambda c: c.rank) if not c.nodal]
    else:
        candidates = planar_members(spec)
    lines = []
    for member in candidates:
        offset = varpi_offset(spec, member)
        if offset is not None and member.key != accepted:
            lines.append(CentreLine(member.key, c0 + _offset_track(offset, varpis), 'member'))
    if not scanned:
        lines.sort(key=lambda line: _flatness(line.centre_rad, inside))
    return lines[:MAX_MEMBER_LINES]


def _smoothed_varpis(series: MMRSeries, spec: ResonanceSpec, t, window_yr: float, order: int) -> Optional[List[np.ndarray]]:
    """[smooth(varpi), smooth(varpi_P1), ...] unwrapped [rad]; None without the planets' varpi."""
    planets = [series.planets.get(name) for name in spec.planets]
    if spec.retrograde or any(p is None or p.varpi is None for p in planets):
        return None
    arrays = [series.varpi] + [p.varpi for p in planets]
    return [_smooth(t, np.unwrap(np.asarray(values, dtype=float)), window_yr, order) for values in arrays]


def _accepted_line(analysis: MMRAnalysis, series: MMRSeries, member: ResonanceSpec, c0, varpis, t) -> Optional[CentreLine]:
    derived = next((item for item in analysis.derived if item.resonance.key == member.key and item.geometry is not None), None)
    if derived is None and len(analysis.derived) == 1 and analysis.derived[0].geometry is not None:
        derived = analysis.derived[0]
    offset = varpi_offset(analysis.resonance, member)
    angle = None
    if derived is not None and derived.derived_series is not None:
        angle = np.mod(np.asarray(derived.derived_series.angle_wrapped, dtype=float), TWO_PI)
    elif offset is not None and series.planets:
        raw = [series.varpi] + [series.planets[name].varpi for name in analysis.resonance.planets if name in series.planets]
        if len(raw) == 1 + len(offset[1]):
            delta = offset[0] * raw[0] + sum(c * np.asarray(v, dtype=float) for c, v in zip(offset[1], raw[1:]))
            angle = np.mod(np.asarray(series.angle_wrapped, dtype=float) + delta, TWO_PI)
    if derived is not None:
        centre = np.asarray(derived.geometry.centre.centre_rad, dtype=float)
        if len(centre) == len(t):
            return CentreLine(member.key, centre, 'subresonance', angle)
    if offset is None or varpis is None:
        return None
    return CentreLine(member.key, c0 + _offset_track(offset, varpis), 'subresonance', angle)


# --------------------------------------------------------------------------------------------- (i), (j)


def centre_pass_matters(centre) -> bool:
    """The largest monotone pass of the centre is shaded only when it is a whole revolution or more (the centre
    circulates: what G4, G1c, B5 and the secular centre_pass read); a shorter pass decides nothing."""
    times = is_finite(centre.max_pass_t0_yr) and is_finite(centre.max_pass_t1_yr)
    return bool(times and is_finite(centre.max_pass_rad) and centre.max_pass_rad >= TWO_PI)


def draw_centre_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(i) the centre of sigma: sigma_f smoothed over W_c drawn over sigma_f, the centre's turns and its largest
    monotone pass when it is a revolution or more (`centre_pass_matters`; the numbers geo_centre_window_yr,
    geo_centre_n_tp, geo_centre_max_pass_rev)."""
    geometry, prep = frame.analysis.geometry, frame.prep
    _style_axis(ax, 'i', 'centre\nrad', layout)
    if frame.centre is None or geometry is None:
        _note(ax, 'centre not computed', layout.font_note_pt)
        return
    centre = geometry.centre
    if prep is not None:
        _line(ax, frame.x, np.where(frame.inside, prep.angle_f, np.nan), color=RAW, alpha=RAW_ALPHA, lw=0.5, zorder=1)
    centre_in = np.where(frame.inside, frame.centre, np.nan) if len(frame.centre) == len(frame.t) else frame.centre
    _line(ax, frame.times(frame.centre_t), centre_in, color=CENTRE, lw=1.3, zorder=3)
    turns = centre.turns
    if turns is not None and 0 < len(turns.times_yr) <= MAX_TP_MARKERS:
        turn_markers(ax, frame.times(turns.times_yr), turns.values_rad, turns.kinds, CENTRE_TURN_SIZE, CENTRE, white_edge=True)
    if centre_pass_matters(centre):
        ax.axvspan(frame.times(centre.max_pass_t0_yr), frame.times(centre.max_pass_t1_yr), color=CENTRE, alpha=0.1, lw=0, zorder=0)
    _view(ax, prep.angle_f[frame.inside] if prep is not None else centre_in, pad=0.08)


def draw_centres_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(j) the centres of sigma0, the members of the multiplet and the owner, each minus its circular mean, with
    their keys at the right (no caption: why a check did not run is in the right column); the kinematic terms
    instead when the libration is kinematic."""
    analysis = frame.analysis
    mode = centre_mode(analysis)
    _style_axis(ax, 'j', 'centre − mean\n°', layout)
    if frame.centre is None:
        _note(ax, 'centres not computed: no centre of σ', layout.font_note_pt)
        return
    if mode == 'kinematic':
        _draw_kinematic(ax, frame, layout)
        return
    items = []
    for i, line in enumerate(frame.centre_lines):
        colour, width, style, z = _line_style(line, i)
        values = np.where(frame.inside, _relative_degrees(line.centre_rad, line.centre_rad[frame.inside]), np.nan)
        _line(ax, frame.x, values, color=colour, lw=width, ls=style, zorder=z)
        finite = values[np.isfinite(values)]
        if len(finite):
            items.append((finite[-1], line.key, colour))
    ax.axhline(0, color=REFERENCE, lw=0.5, alpha=0.5, zorder=0)
    ax.set_ylim(-CENTRES_AXIS_DEG, CENTRES_AXIS_DEG)
    ax.set_yticks([-180, -90, 0, 90, 180])
    _right_labels(ax, [(float(np.clip(value, -180.0, 180.0)), text, colour) for value, text, colour in items], layout.font_note_pt)


def _line_style(line: CentreLine, index: int) -> Tuple[str, float, str, int]:
    if line.role == 'sigma0':
        return CENTRE, 1.3, '-', 5
    if line.role == 'subresonance':
        return SUBRESONANCE, 1.5, '-', 6
    if line.role == 'owner':
        return OWNER, 1.3, '--', 6
    return MEMBER_COLOURS[(index - 1) % len(MEMBER_COLOURS)], 1.1, '-', 3


def _detrended(t, y) -> np.ndarray:
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    return y - np.polyval(np.polyfit(t - t[0], y, 1), t - t[0]) if len(t) > 2 else y - np.mean(y)


def _draw_kinematic(ax, frame: _Frame, layout: VerdictFigureLayout):
    """Kinematic mode: the slow angle against p varpi (both detrended, labelled 'σ' and 'pϖ' at the right); a
    kinematic libration follows p varpi (kin_corr_rate in the right column)."""
    analysis, slow = frame.analysis, frame.analysis.slow
    p = analysis.resonance.body_varpi_coefficient
    mask = (slow.t_yr >= slow.window[0]) & (slow.t_yr <= slow.window[1])
    if np.count_nonzero(mask) < 3:  # a record shorter than two centre windows: the analysis window
        mask = (slow.t_yr >= frame.window[0]) & (slow.t_yr <= frame.window[1])
    x = frame.times(slow.t_yr[mask])
    # both detrended (kin_corr_detrended): the mean rate of p varpi is cancelled by the lambda part
    angle = _detrended(slow.t_yr[mask], slow.angle_slow[mask])
    varpi = _detrended(slow.t_yr[mask], p * slow.varpi_slow[mask])
    _line(ax, x, angle, color=CENTRE, lw=1.1)
    _line(ax, x, varpi, color=SUBRESONANCE, lw=1.0, ls='--')
    ax.set_ylabel('rad', fontsize=layout.font_axis_pt, labelpad=2)
    labels = [(values[-1], text, colour) for values, text, colour in ((angle, 'σ', CENTRE), (varpi, f'{p}ϖ', SUBRESONANCE)) if len(values)]
    _right_labels(ax, labels, layout.font_note_pt)


def panel_rects(layout: VerdictFigureLayout, n: Optional[int] = None) -> List[Rect]:
    """Rectangles (inches) of the time panels, top to bottom: `layout.panels` split evenly, `panel_gap` between."""
    n = n or layout.n_panels
    x, y, w, h = layout.panels
    height = (h - (n - 1) * layout.panel_gap) / n
    return [(x, y + h - (i + 1) * height - i * layout.panel_gap, w, height) for i in range(n)]


def stacked_axes(fig, layout: VerdictFigureLayout, n: int, rects: Optional[Sequence[Rect]] = None) -> list:
    """The time panels as axes sharing t (`panel_rects`, or the given `rects`)."""
    width, height = fig.get_size_inches()
    axes = []
    for x, y, w, h in rects if rects is not None else panel_rects(layout, n):
        axes.append(fig.add_axes((x / width, y / height, w / width, h / height), sharex=axes[0] if axes else None))
    return axes


def draw_time_panels(fig, analysis: MMRAnalysis, series: MMRSeries, layout: VerdictFigureLayout, options=None, settings=None) -> list:
    """The time panels of PANELS sharing the time axis (kyr or Myr by `base.time_unit`); returns their axes. When the
    cycles are too dense for (b) (`needs_strip`), the column is split in eleven and the third slot holds the strip
    (b′) (`draw_strip`, axes created after the ten panels, not returned); otherwise ten equal panels.

    `options`: plot options (`style.DEFAULT_PLOT_OPTIONS`; the figure reads `combined`); `settings`: the
    AnalysisSettings of the episodes' coverage rule (default: from the configuration)."""
    frame = _Frame(analysis, series, layout, options, settings)
    strip = needs_strip(frame)
    rects = panel_rects(layout, len(PANELS) + 1) if strip else panel_rects(layout, len(PANELS))
    strip_rect = rects.pop(2) if strip else None
    axes = stacked_axes(fig, layout, len(PANELS), rects)
    drawers = (
        draw_angle_panel,
        draw_wrapped_panel,
        draw_rate_panel,
        draw_axis_panel,
        draw_adot_panel,
        draw_eccentricity_panel,
        draw_inclination_panel,
        draw_energy_panel,
        draw_centre_panel,
        draw_centres_panel,
    )
    for ax, drawer in zip(axes, drawers):
        drawer(ax, frame, layout)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(f't, {frame.unit}', fontsize=layout.font_axis_pt, labelpad=1)
    axes[-1].set_xlim(frame.x[0], frame.x[-1])
    if strip_rect is not None:
        draw_strip(fig, strip_rect, frame, layout, axes[1])
    return axes


# --------------------------------------------------------------------------------------------- bottom row


def _bottom_title(ax, text: str, layout: VerdictFigureLayout):
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha='left', va='bottom', fontsize=layout.font_letter_pt, fontweight='bold')
    ax.tick_params(labelsize=layout.font_ticks_pt, length=2.5, pad=1.5)


def _empty(ax, text: str, layout: VerdictFigureLayout, crossed: bool = False):
    """An empty frame with the reason; `crossed` draws the cross of a portrait that would mislead."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#dddddd')
    if crossed:
        for ys in ((0, 1), (1, 0)):
            ax.plot((0, 1), ys, transform=ax.transAxes, color='#dddddd', lw=1.0, zorder=0)
    ax.text(
        0.5,
        0.5,
        textwrap.fill(text, 34),
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=layout.font_note_pt + 1,
        color=NOTE,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2},
    )


def _alias_reason(analysis: MMRAnalysis) -> str:
    return f'not drawn: the samples are aliased ({analysis.sampling.verdict}); a portrait of them shows an island that is not there'


def _angle_axis(ax, low: float, layout: VerdictFigureLayout, label: str):
    base = np.degrees(low)
    ax.set_xlim(base, base + 360)
    ax.set_xticks(_right_angles(base))
    ax.xaxis.set_major_formatter(_angle_formatter())
    rate_zero(ax)
    ax.set_xlabel(label, fontsize=layout.font_axis_pt, labelpad=1)
    ax.set_ylabel('dσ/dt, rad/kyr', fontsize=layout.font_axis_pt, labelpad=1)
    ax.grid(alpha=0.15, lw=0.5)


def portrait_cycles(run, tp_times, tp_kinds, max_cycles: int = PORTRAIT_MAX_CYCLES) -> List[Tuple[float, float]]:
    """The spans of an episode drawn in (l): the whole episode when it holds at most `max_cycles` cycles (maximum to
    maximum of sigma_f), so no sample of it is left out (its partial cycles at either end included); else
    `max_cycles` of its cycles, evenly spaced over it."""
    maxima = np.asarray(tp_times, dtype=float)[np.asarray(tp_kinds) > 0]
    maxima = maxima[(maxima >= run.t_start_yr) & (maxima <= run.t_end_yr)]
    bounds = list(zip(maxima[:-1], maxima[1:]))
    if len(bounds) <= max_cycles:
        return [(run.t_start_yr, run.t_end_yr)]
    chosen = np.unique(np.linspace(0, len(bounds) - 1, min(max_cycles, len(bounds))).astype(int))
    return [bounds[i] for i in chosen]


def _episodes_window_low(geometry, runs, t, angle) -> float:
    """Lower edge of the 360-degree window of (l): around the centre track over the episodes (a horseshoe's samples
    crowd its two ends, so their own mean would split it), else around sigma_f there."""
    centres = []
    for run in runs:
        track = geometry.centre_covering(run.t_start_yr)
        track_t = np.asarray(track.t_yr, dtype=float)
        centres.append(np.asarray(track.centre_rad, dtype=float)[(track_t >= run.t_start_yr) & (track_t <= run.t_end_yr)])
    values = np.concatenate(centres) if centres else np.array([])
    if not len(_finite(values)):
        covered = np.zeros(len(t), dtype=bool)
        for run in runs:
            covered |= (t >= run.t_start_yr) & (t <= run.t_end_yr)
        values = angle[covered]
    return _circular_mean(values) - np.pi


def _portrait_episodes(accepted) -> list:
    """The PORTRAIT_MAX_EPISODES longest accepted runs, in time order (the episodes of (l))."""
    longest = sorted(accepted, key=lambda run: run.t_start_yr - run.t_end_yr)[:PORTRAIT_MAX_EPISODES]
    return sorted(longest, key=lambda run: run.t_start_yr)


def portrait_window_low(analysis: MMRAnalysis) -> Optional[float]:
    """Lower edge [rad] of the 360-degree window shared by (k) and (l), so the two portraits compare: around the centre
    over the episodes of (l) (`_episodes_window_low`) when there are accepted episodes, else around the centre of the
    whole record; None without a filtered angle."""
    prep, geometry = analysis.prep, analysis.geometry
    if prep is None:
        return None
    accepted = accepted_runs(geometry)
    if accepted:
        t, angle = np.asarray(prep.t_yr, dtype=float), np.asarray(prep.angle_f, dtype=float)
        return _episodes_window_low(geometry, _portrait_episodes(accepted), t, angle)
    t = np.asarray(prep.t_yr, dtype=float)
    mask = (t >= prep.window[0]) & (t <= prep.window[1])
    return _window_low(geometry.centre.centre_rad if geometry is not None else np.asarray(prep.angle_f)[mask])


def episode_rate_limits(analysis: MMRAnalysis) -> Optional[Tuple[float, float]]:
    """The y view [rad/kyr] of every sample of every accepted episode inside the window of (k) (sigma_dot at the
    response cutoff, padded as the portraits are): the least (k) may show, whatever cycles (l) draws. None without
    a filtered angle or an accepted episode."""
    prep, geometry = analysis.prep, analysis.geometry
    if prep is None or geometry is None:
        return None
    t = np.asarray(prep.t_yr, dtype=float)
    covered = np.zeros(len(t), dtype=bool)
    for run in geometry.runs:
        if run.accepted:
            covered |= (t >= run.t_start_yr) & (t <= run.t_end_yr)
    covered &= (t >= prep.window[0]) & (t <= prep.window[1])
    if not covered.any():
        return None
    return _view_limits(_rate(prep)[covered] * PER_KYR, pad=PORTRAIT_VIEW_PAD)


def share_portrait_limits(axes, episodes: Optional[Tuple[float, float]] = None) -> None:
    """One y range for the portraits that were drawn, (k) then (l) (their x window is already shared), so they read on
    the same scales. The episodes' view is (l)'s, widened to `episodes` (`episode_rate_limits`: every sample of every
    accepted episode), so (k) is never clipped tighter than its own episodes. Both get the union of (k)'s and the
    episodes' views; when the whole record spans more than PORTRAIT_SHARE_MAX times the episodes' view (a circulation
    far faster than the libration would squash the island to a line), the episodes' view, the whole record's samples
    beyond it marked ▲▼ at the edge of (k)."""
    drawn = [ax for ax in axes if ax.get_xlabel()]
    if len(drawn) < 2:
        return
    (k_low, k_high), (l_low, l_high) = drawn[0].get_ylim(), drawn[1].get_ylim()
    if episodes is not None:
        l_low, l_high = min(l_low, episodes[0]), max(l_high, episodes[1])
    if k_high - k_low <= PORTRAIT_SHARE_MAX * (l_high - l_low):
        limits = (min(k_low, l_low), max(k_high, l_high))
    else:
        limits = (l_low, l_high)
        for above, beyond in ((True, k_high > l_high), (False, k_low < l_low)):
            if beyond:
                edge_marker(drawn[0], 0.5, above, REFERENCE)
    for ax in drawn:
        ax.set_ylim(*limits)


def draw_episode_portrait(
    ax, analysis: MMRAnalysis, layout: VerdictFigureLayout = LAYOUT_V1, divisor: float = 1.0, unit: str = 'yr', low: Optional[float] = None
):
    """(l) the accepted episodes only: sigma_f against sigma_dot at the response cutoff, the PORTRAIT_MAX_EPISODES
    longest, each whole or, when it holds more, PORTRAIT_MAX_CYCLES of its cycles (`portrait_cycles`), one colour per
    episode (the colours of (a)); a forced wave is labelled '#id wave', as on (a). `low`: the lower edge of the
    360-degree window [rad] (`portrait_window_low`, shared with (k)); default: around the episodes."""
    _bottom_title(ax, '(l) accepted episodes', layout)
    prep, geometry = analysis.prep, analysis.geometry
    if _aliased(analysis):
        _empty(ax, _alias_reason(analysis), layout, crossed=True)
        return
    if prep is None or geometry is None:
        _empty(ax, 'not computed: no filtered angle', layout)
        return
    accepted = accepted_runs(geometry)
    if not accepted:
        _empty(ax, 'no accepted episode', layout)
        return
    colours = episode_colours(geometry.runs)
    chosen = _portrait_episodes(accepted)
    waves = getattr(geometry, 'forced_wave_ids', frozenset())  # '#id wave', as on (a): no episode of the angle (F6)
    t = np.asarray(prep.t_yr, dtype=float)
    rate = _rate(prep) * PER_KYR
    angle = np.asarray(prep.angle_f, dtype=float)
    low = _episodes_window_low(geometry, chosen, t, angle) if low is None else low
    tp = geometry.turning_points
    shown = []
    for run in chosen:
        colour = colours.get(run.episode_id, EPISODE_COLOURS[0])
        label = f'#{run.episode_id}' + (FORCED_WAVE_LABEL if run.episode_id in waves else '')
        label += f' {run.t_start_yr / divisor:.4g}–{run.t_end_yr / divisor:.4g} {unit}'
        label += ' (before jump)' if run.before_last_jump else ''
        for n, (t0, t1) in enumerate(portrait_cycles(run, tp.times_yr, tp.kinds)):
            mask = (t >= t0) & (t <= t1)
            x = np.degrees(_wrap_window(angle[mask], low))
            y = rate[mask].copy()
            y[1:][np.abs(np.diff(x)) > 180] = np.nan
            ax.plot(x, y, color=colour, lw=0.7, alpha=0.85, label=label if n == 0 else None, rasterized=True)
            if n == 0 and len(x):
                ax.plot(x[:1], y[:1], 'o', ms=3, color=colour)
            shown.append(y)
    _angle_axis(ax, low, layout, 'σ_f, °')
    centre_marks(ax, chosen, low)
    _view(ax, np.concatenate(shown) if shown else [0.0], pad=PORTRAIT_VIEW_PAD, extra_top=0.25)
    ax.legend(fontsize=layout.font_note_pt, loc='upper right', frameon=False, handlelength=1.2)
    if len(accepted) > len(chosen):
        ax.text(
            0.01,
            0.01,
            f'{len(chosen)} longest of {len(accepted)} episodes',
            transform=ax.transAxes,
            fontsize=layout.font_note_pt,
            color=NOTE,
            va='bottom',
        )


def draw_portrait(
    ax, analysis: MMRAnalysis, norm, layout: VerdictFigureLayout = LAYOUT_V1, divisor: float = 1.0, low: Optional[float] = None
) -> None:
    """(k) sigma_f mod 360 around the centre against sigma_dot over the whole record, coloured by time, every sample
    (rasterized); crossed out for aliased samples.

    `divisor` converts years into the unit of `norm` (the figure's time unit); `low`: the lower edge of the 360-degree
    window [rad] (`portrait_window_low`, shared with (l)); default: around the centre."""
    from matplotlib.collections import LineCollection

    _bottom_title(ax, '(k) phase portrait, whole record', layout)
    prep = analysis.prep
    if _aliased(analysis):
        _empty(ax, _alias_reason(analysis), layout, crossed=True)
        return
    if prep is None:
        _empty(ax, 'not computed: no filtered angle', layout)
        return
    t = np.asarray(prep.t_yr, dtype=float)
    mask = (t >= prep.window[0]) & (t <= prep.window[1])
    if low is None:
        low = _window_low(analysis.geometry.centre.centre_rad if analysis.geometry is not None else prep.angle_f[mask])
    x = np.degrees(_wrap_window(prep.angle_f[mask], low))
    y = _rate(prep)[mask] * PER_KYR
    times = t[mask] / divisor
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    keep = np.abs(np.diff(x)) <= 180
    collection = LineCollection(segments[keep], cmap=time_cmap(), norm=norm, linewidths=0.7, rasterized=True)
    collection.set_array(times[:-1][keep])
    ax.add_collection(collection)
    _angle_axis(ax, low, layout, 'σ_f, °')
    if analysis.geometry is not None:
        centre_marks(ax, accepted_runs(analysis.geometry), low)
    _view(ax, y, pad=PORTRAIT_VIEW_PAD)


def fair_unavailable_reason(analysis: MMRAnalysis, series: MMRSeries, max_step_fraction: float = FAIR_STEP_FRACTION) -> Optional[str]:
    """Why FAIR is not drawn, or None: prograde two-body angles with the planet's longitudes and dt <= P/4.2 only.

    FAIR is not generalised to three-body angles: in (M, k1(lambda1 - lambda) + k2(lambda2 - lambda)) a libration
    gives |q| strips, but that plane needs k1, k2, so it is sigma shifted by qM and cannot identify p:q."""
    spec = analysis.resonance
    if spec.kind == ResonanceKind.RETROGRADE:
        return 'retrograde resonance'
    if spec.kind != ResonanceKind.TWO_BODY:
        return 'three-body angle'
    if series.mean_anomaly is None or series.mean_longitude is None:
        return 'no mean anomaly or mean longitude'
    planet = series.planets.get(spec.planets[0]) if spec.planets else None
    if planet is None or planet.mean_longitude is None:
        return 'no planet longitudes'
    fraction = analysis.context.fair_step_fraction if analysis.context is not None else float('nan')
    if not is_finite(fraction) or fraction > max_step_fraction:
        return f'step {fraction:.2g} of the orbital period > 1/{1 / max_step_fraction:.2g}'
    return None


def draw_fair(ax, analysis: MMRAnalysis, series: MMRSeries, norm, layout: VerdictFigureLayout = LAYOUT_V1, divisor: float = 1.0) -> None:
    """(m) FAIR plane of a prograde two-body angle with dt <= P/4.2 (`fair_unavailable_reason` is None)."""
    _bottom_title(ax, '(m) FAIR', layout)
    planet = series.planets[analysis.resonance.planets[0]]
    inner = osc.is_inner(series.a, planet.a)
    x, y = osc.fair_coordinates(series.mean_anomaly, series.mean_longitude, planet.mean_longitude, inner)
    ax.scatter(x, y, c=series.t_yr / divisor, cmap=time_cmap(), norm=norm, s=1.2, linewidths=0, rasterized=True)
    ticks = [0, 90, 180, 270, 360]
    ax.set(xlim=(0, 360), ylim=(0, 360), xticks=ticks, yticks=ticks)
    name = analysis.resonance.planets[0]
    ax.set_xlabel('M, °', fontsize=layout.font_axis_pt, labelpad=1)
    ax.set_ylabel(f'λ_{name} − λ, °' if inner else f'λ − λ_{name}, °', fontsize=layout.font_axis_pt, labelpad=1)


def legend_rect(layout: VerdictFigureLayout, third_slot_free: bool) -> Rect:
    """The legend's box: `layout.legend`, widened over the third slot of the bottom row when nothing is drawn there
    (no FAIR; the secular and ZLK figures)."""
    x, y, w, h = layout.legend
    if not third_slot_free:
        return layout.legend
    left = layout.bottom[2][0]
    return (left, y, x + w - left, h)


def draw_legend(fig, layout: VerdictFigureLayout, lines: Sequence[str] = LEGEND, rect: Optional[Rect] = None) -> None:
    """What every line and mark of the panels means, wrapped to the legend box (`rect`, default `layout.legend`);
    the font shrinks (to LEGEND_MIN_FONT_PT) until the text fits the box."""
    width, height = fig.get_size_inches()
    x, y, w, h = rect or layout.legend
    top = y + h
    fig.text(x / width, top / height, 'Legend', fontsize=layout.font_legend_pt + 1.5, fontweight='bold', va='top')
    renderer = fig.canvas.get_renderer()
    font = layout.font_legend_pt
    while True:
        chars = max(int(w * 72 / (LEGEND_CHAR_WIDTH * font)), 20)
        text = '\n'.join(textwrap.fill(line, chars, subsequent_indent='   ') for line in lines)
        artist = fig.text(x / width, (top - 0.22) / height, text, fontsize=font, va='top', color='#333333', linespacing=1.2)
        box = artist.get_window_extent(renderer)
        fits = box.width <= w * fig.dpi + 0.5 and box.height <= (h - 0.22) * fig.dpi + 0.5
        if fits or font <= LEGEND_MIN_FONT_PT:
            return
        artist.remove()
        font -= 0.25


def _colour_bar(fig, norm, layout: VerdictFigureLayout, unit: str, label_on_top: bool = False):
    """The time colour bar right of the first portrait; `label_on_top`: its label 't, kyr' above the bar instead of
    beside it, where the next portrait's y label and wide tick labels reach (the ZLK free eccentricity vector, 591986)."""
    from matplotlib.cm import ScalarMappable

    width, height = fig.get_size_inches()
    x, y, w, h = layout.colour_bar
    cax = fig.add_axes((x / width, y / height, w / width, h / height))
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=time_cmap()), cax=cax)
    if label_on_top:
        cax.set_title(f't, {unit}', fontsize=layout.font_ticks_pt, pad=3, loc='left')
    else:
        bar.set_label(f't, {unit}', fontsize=layout.font_ticks_pt, labelpad=2)
    bar.ax.tick_params(labelsize=layout.font_ticks_pt - 0.5)
    bar.outline.set_linewidth(0.5)


# --------------------------------------------------------------------------------------------- the figure


def _rect_axes(fig, rect: Rect):
    width, height = fig.get_size_inches()
    x, y, w, h = rect
    return fig.add_axes((x / width, y / height, w / width, h / height))


def figure_rows(analysis: MMRAnalysis) -> Tuple[Mapping[str, Any], Mapping[str, Any], List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """The CSV rows the text is printed from: summary, numbers, episodes and constants of the analysis."""
    constants = analysis.context.constants_rows() if analysis.context is not None else []
    return analysis.summary_row(), analysis.numbers_row(), analysis.episode_rows(), constants


def draw_combined(
    analysis: MMRAnalysis,
    series: MMRSeries,
    layout: VerdictFigureLayout = LAYOUT_V1,
    stamp: Optional[str] = None,
    rows: Optional[Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = None,
    options: Optional[Mapping[str, Any]] = None,
    settings=None,
):
    """The whole V1 figure; `stamp` (e.g. 'recomputed != summary') is printed in red in the header. Returns the Figure.
    `settings`: the AnalysisSettings the explanation compares with (default: from the configuration).

    `rows` = (summary, numbers, episodes, constants) as read from the CSV files (a restore without
    recomputation); by default they come from the analysis itself (`figure_rows`). `options`: plot options
    (`style.DEFAULT_PLOT_OPTIONS`, e.g. {'combined': {'osculating_a': 'hidden'}}).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    summary, numbers, episodes, constants = rows if rows is not None else figure_rows(analysis)
    settings = settings if settings is not None else _settings()  # read once for the panels and the right column
    fig = plt.figure(figsize=layout.size, dpi=layout.dpi)
    draw_header(fig, analysis, layout, series, stamp)
    draw_time_panels(fig, analysis, series, layout, options, settings)
    divisor, unit = time_unit(series.t_yr[-1] - series.t_yr[0])
    norm = Normalize(series.t_yr[0] / divisor, series.t_yr[-1] / divisor)
    whole_rect, episodes_rect, third_rect = layout.bottom
    low = portrait_window_low(analysis)
    portraits = (_rect_axes(fig, whole_rect), _rect_axes(fig, episodes_rect))
    draw_portrait(portraits[0], analysis, norm, layout, divisor, low)
    _colour_bar(fig, norm, layout, unit)
    draw_episode_portrait(portraits[1], analysis, layout, divisor, unit, low)
    share_portrait_limits(portraits, episode_rate_limits(analysis))
    fair = fair_unavailable_reason(analysis, series) is None
    if fair:
        draw_fair(_rect_axes(fig, third_rect), analysis, series, norm, layout, divisor)
    draw_legend(fig, layout, rect=legend_rect(layout, third_slot_free=not fair))
    width, height = layout.size
    fig.add_artist(plt.Line2D([layout.separator_x / width] * 2, [0.3 / height, (height - 0.2) / height], color='#dddddd', lw=0.8))
    fig.text_stats = draw_right_column(fig, analysis, (summary, numbers, episodes, constants), layout, options, settings)
    return fig


def draw_right_column(fig, analysis: MMRAnalysis, rows, layout: VerdictFigureLayout, options=None, settings=None) -> Dict[str, Any]:
    """The right column by the plot option `combined.numbers`: 'explained' (the verdict in words, the checks and the
    numbers a reader needs, `mmr_verdict_panel.draw_explained`) or 'all' (every number, `draw_text`). Returns the
    statistics of the column (for the caller and the tests)."""
    if resolve_plot_options(options)['combined']['numbers'] == NumbersMode.ALL:
        summary, numbers, episodes, constants = rows
        return draw_text(fig, text_blocks(summary, numbers, analysis.verdict.decision_path, episodes, constants), layout)
    return draw_explained(fig, analysis, rows, layout, settings)


class VerdictFigurePlotter(BasePlotter):
    """Draws, saves and closes the combined v2 figure of one angle (and of a derived subresonance row)."""

    def __init__(self, analysis: MMRAnalysis, series: MMRSeries, layout: VerdictFigureLayout = LAYOUT_V1, options=None):
        self.analysis = analysis
        self.series = series
        self.layout = layout
        self.options = options

    def plot(self, stamp: Optional[str] = None):
        """Draw the figure (kept in `self.figure`)."""
        self._figure = draw_combined(self.analysis, self.series, self.layout, stamp, options=self.options)
        return self._figure
