"""Figure styles and plot options of the MMR diagnostic plots.

Two styles share one layout: `screen` (default) for reviewing many PNGs, `paper` for a
two-column journal figure (180 mm wide, 8 pt serif, no title — the caption carries it).
Styles are applied through `matplotlib.rc_context`, so global settings stay untouched.
"""

import copy
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np


# MMR plot kinds, as named in `plots`; each is drawn by MMRPlotter.plot_<kind>.
MMR_PLOT_KINDS = ('combined', 'recurrence', 'fair', 'portrait', 'cycles')
# Secular plot kinds; `combined` and `portrait` mean the same kind of figure as for an MMR.
SECULAR_PLOT_KINDS = ('combined', 'portrait', 'angle', 'rate')


class PlotStyle(StrEnum):
    SCREEN = 'screen'
    PAPER = 'paper'


class RawMode(StrEnum):
    """How raw series accompany the filtered ones."""

    LINE = 'line'
    MARKERS = 'markers'  # the line plus hollow markers on actual samples
    NONE = 'none'


class SeriesMode(StrEnum):
    """How the time series are drawn: lines, min-max bands, or bands only when too dense for lines."""

    AUTO = 'auto'
    LINES = 'lines'
    ENVELOPE = 'envelope'


class PortraitY(StrEnum):
    RATE = 'rate'  # sigma_dot
    AXIS = 'axis'  # a - a0


class OsculatingAxis(StrEnum):
    """How panel (d) of the MMR verdict figure draws the osculating a around a_f."""

    CLIPPED = 'clipped'  # drawn, clipped to the view of a_f, a triangle at the edge it exceeds
    HIDDEN = 'hidden'  # not drawn when it exceeds the view of a_f
    FULL = 'full'  # (default) the view holds the osculating band (ENVELOPE_PERCENTILES); samples beyond: triangles


class NumbersMode(StrEnum):
    """The right column of the verdict figures (MMR, secular, ZLK; `combined_numbers`)."""

    EXPLAINED = 'explained'  # the verdict in words, the checks and the numbers a reader needs (layout A′)
    ALL = 'all'  # every number of the decision (the full dump, for debugging)


@dataclass(frozen=True)
class CombinedLayout:
    """Geometry of the combined figure, in inches.

    Top row: the three stacked time series together are exactly as tall as the square
    recurrence and FAIR panels (`top`). Bottom row: three equal squares, whose side
    follows from the figure width. The figure height follows from both rows.
    """

    left: float  # room for y tick labels and label of the first column
    right: float
    bottom: float  # room for x tick labels and labels of the bottom row
    title: float  # room above the top row (title, panel letters)
    gap: float  # between neighbouring panels: y ticks and label of the right one
    row_gap: float  # between the rows: x labels of the top row, letters of the bottom
    top: float  # height of the top row
    stack_gap: float  # between the three time-series panels
    bar: float  # colour bar width
    bar_pad: float  # between a panel and its colour bar
    bar_room: float  # tick labels (and label) of a colour bar


@dataclass(frozen=True)
class FigureStyle:
    width: float  # inches (combined figure)
    layout: CombinedLayout
    panel_size: float  # inches, one standalone panel
    dpi: int
    font_size: float
    font_family: str
    title: bool  # figure title (body and resonance)
    line: float  # filtered series
    raw_line: float  # raw series
    marker: float  # scatter point area
    series_marker: float  # point area of the wrapped angle against time

    def rc(self) -> Dict[str, Any]:
        serif = self.font_family == 'serif'
        return {
            'font.size': self.font_size,
            'font.family': self.font_family,
            'font.serif': ['STIX Two Text', 'STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'stix' if serif else 'dejavusans',
            'axes.titlesize': self.font_size,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'axes.linewidth': 0.6,
            'xtick.major.width': 0.6,
            'ytick.major.width': 0.6,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.formatter.useoffset': False,
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.dpi': self.dpi,
            'pdf.fonttype': 42,
        }


STYLES = {
    PlotStyle.SCREEN: FigureStyle(
        width=16.0,
        layout=CombinedLayout(
            left=1.15,
            right=0.2,
            bottom=0.65,
            title=0.75,
            gap=1.0,
            row_gap=1.15,
            top=3.3,
            stack_gap=0.08,
            bar=0.15,
            bar_pad=0.12,
            bar_room=0.75,
        ),
        panel_size=6.0,
        dpi=150,
        font_size=11,
        font_family='sans-serif',
        title=True,
        line=1.2,
        raw_line=0.8,
        marker=4.0,
        series_marker=2.5,
    ),
    PlotStyle.PAPER: FigureStyle(
        width=7.09,  # 180 mm
        layout=CombinedLayout(
            left=0.8,
            right=0.12,
            bottom=0.36,
            title=0.18,
            gap=0.52,
            row_gap=0.66,
            top=1.45,
            stack_gap=0.04,
            bar=0.08,
            bar_pad=0.06,
            bar_room=0.42,
        ),
        panel_size=3.46,  # 88 mm, one column
        dpi=300,
        font_size=8,
        font_family='serif',
        title=False,
        line=0.8,
        raw_line=0.5,
        marker=1.2,
        series_marker=0.8,
    ),
}

# Colours. Time is the only quantity in colour; everything else is ink or grey.
INK = 'black'
RAW = '#7f8790'  # osculating (unfiltered) series; drawn with RAW_ALPHA over white in the verdict figures
RAW_ALPHA = 0.55
RAW_LIGHT_ALPHA = 0.22  # (d): the osculating a, lighter than a filtered at P_c (drawn in RAW at RAW_ALPHA)
# The y view of a panel holds this percentile band of an osculating series ((a) sigma, (d) a): the rare samples beyond
# it are triangles at the edge, at their time (author 2026-10-17: the whole grey band must be in view)
ENVELOPE_PERCENTILES = (0.5, 99.5)
# A filtered line within FILTER_EDGE_TRIM_CUTOFFS x P_c of a record end (the filter's edge spread, outside the
# analysis window): drawn in this style, lighter and dashed, so the panel still covers the whole record
EDGE_SPREAD_STYLE = {'ls': (0, (3, 2)), 'alpha': 0.45}
CENTRE = '#b2182b'
OUT_OF_FOCUS = '#cfd4da'
ENVELOPE = '#3a3a3a'  # filtered series drawn as a min-max band


@lru_cache(maxsize=1)
def time_cmap():
    """Viridis without its palest yellow end, which vanishes on a white background."""
    import matplotlib
    from matplotlib.colors import ListedColormap

    return ListedColormap(matplotlib.colormaps['viridis'](np.linspace(0.0, 0.9, 256)), name='viridis_trunc')


def time_label(unit: str) -> str:
    """Label of every time colour bar, so all figures read the same."""
    return f'Time ({unit})'


def time_colorbar(fig, norm, label: str, **where):
    """Colour bar of the time scale; `where` is `ax=...` or `cax=...` plus layout keywords."""
    from matplotlib.cm import ScalarMappable

    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=time_cmap()), **where)
    bar.set_label(label)
    bar.outline.set_linewidth(0.5)
    return bar


def panel_letter(ax, letter: str, inside: bool = False):
    """Panel letter above the top-left corner, or inside it for tightly stacked panels."""
    if inside:
        box = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1}
        ax.text(0.006, 0.95, f'({letter})', transform=ax.transAxes, ha='left', va='top', fontweight='bold', zorder=8, bbox=box)
    else:
        ax.text(-0.02, 1.02, f'({letter})', transform=ax.transAxes, ha='right', va='bottom', fontweight='bold')


# Defaults of `plot_options` (SimulationConfig / PLOT_OPTIONS). Every nested key is
# validated against this tree, so a typo fails loudly instead of being ignored.
DEFAULT_PLOT_OPTIONS: Dict[str, Any] = {
    'style': PlotStyle.SCREEN,
    'raw': RawMode.LINE,
    'series': SeriesMode.AUTO,
    'recurrence': {'max_points': None, 'exclude_samples': 2},  # None: ~8 samples per cycle, 700-2500
    'portrait': {'y': PortraitY.RATE},
    'cycles': {'prominence': None},  # rad; None = 5% of the angle range within [0.005, 0.1]
    'fair': {'max_step_fraction': 0.25},  # warn when the output step exceeds this share of the orbit
    # panel (d) of the MMR verdict figure; the right column of every verdict figure
    'combined': {'osculating_a': OsculatingAxis.FULL, 'numbers': NumbersMode.EXPLAINED},
}

# Options whose value must be a member of an enum, by dotted path.
_CHOICES = {
    'style': PlotStyle,
    'raw': RawMode,
    'series': SeriesMode,
    'portrait.y': PortraitY,
    'combined.osculating_a': OsculatingAxis,
    'combined.numbers': NumbersMode,
}


def resolve_plot_options(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge user options over the defaults, rejecting unknown keys and invalid choices."""
    resolved = copy.deepcopy(DEFAULT_PLOT_OPTIONS)
    for key, value in (options or {}).items():
        if key not in resolved:
            raise ValueError(f"Unknown plot option '{key}'. Known: {', '.join(resolved)}")
        if not isinstance(resolved[key], dict):
            resolved[key] = _checked(key, value)
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Plot option '{key}' must be a dict, got {value!r}")
        for sub, sub_value in value.items():
            if sub not in resolved[key]:
                raise ValueError(f"Unknown plot option '{key}.{sub}'. Known: {', '.join(resolved[key])}")
            resolved[key][sub] = _checked(f'{key}.{sub}', sub_value)
    return resolved


def _checked(path: str, value):
    choices = _CHOICES.get(path)
    if choices is None:
        return value
    if value not in [member.value for member in choices]:
        raise ValueError(f"Plot option '{path}' must be one of {', '.join(choices)}; got {value!r}")
    return choices(value)


# -- The combined figure of the v2 MMR analysis (layout V1, `plotting.mmr_combined`)

Rect = Tuple[float, float, float, float]  # x, y, width, height in inches from the lower-left corner


@dataclass(frozen=True)
class VerdictFigureLayout:
    """Geometry of the V1 figure (inches): the same for every object, so that the figures of one
    status folder can be leafed through. The tests pin these rectangles.

    Left: ten time panels (a)-(j) in `panels`, split evenly with `panel_gap` between them, the compact legend of
    their values right of them (from `margin_legend_x`); below them three squares (`bottom`): the whole record's
    portrait (k) with the time colour bar right of it, the accepted episodes' portrait (l), FAIR (m); then the legend
    (over the third slot too when it is empty: `mmr_combined.legend_rect`). The secular and ZLK figures put their two
    portraits in the first two slots. Right: three text columns
    (`text_columns`)."""

    size: Tuple[float, float] = (24.0, 13.5)
    dpi: int = 120
    header: Rect = (0.35, 12.55, 11.95, 0.85)
    panels: Rect = (0.95, 3.6, 10.25, 8.95)  # the column of time panels, split evenly
    n_panels: int = 10
    panel_gap: float = 0.07  # between neighbouring time panels (each has its own frame)
    # three squares (the portraits keep the aspect of the phase plane; the author, 2026-10-16)
    bottom: Tuple[Rect, Rect, Rect] = ((0.95, 0.5, 2.4, 2.4), (4.45, 0.5, 2.4, 2.4), (7.45, 0.5, 2.4, 2.4))
    colour_bar: Rect = (3.42, 0.5, 0.1, 2.4)  # right of the first slot (the portraits coloured by time)
    legend: Rect = (10.0, 0.08, 2.2, 3.22)
    margin_legend_x: float = 11.27  # left edge of the compact legend right of the time panels (values + colour keys)
    separator_x: float = 12.3
    text_columns: Tuple[Rect, ...] = ((12.45, 0.25, 3.72, 13.0), (16.29, 0.25, 3.72, 13.0), (20.13, 0.25, 3.72, 13.0))
    text_font_pt: float = 7.5
    text_min_font_pt: float = 5.5  # the text shrinks down to this before it would leave its columns
    text_line_height: float = 1.1  # in units of the font size
    font_axis_pt: float = 8.0
    font_ticks_pt: float = 7.0
    font_letter_pt: float = 9.0
    font_note_pt: float = 6.5
    font_legend_pt: float = 6.6
    font_badge_pt: float = 12.0
    font_title_pt: float = 14.0
    tick_margin: float = 0.1  # no y tick label within this fraction of a time panel's edges (labels of neighbours)


LAYOUT_V1 = VerdictFigureLayout()

MONO_FONT = 'DejaVu Sans Mono'
MONO_CHAR_WIDTH = 0.602  # advance of one DejaVu Sans Mono character, in units of the font size

# Marks of checks, margins and ledger entries; never red and green alone (colour-blind readers)
MARK_COLOURS = {'✓': '#1b7837', '~': '#e08214', '✗': '#b2182b', '–': '#888888'}
STATUS_COLOURS = {
    'resonant': '#1b7837',
    'transient': '#e08214',
    'near_separatrix': '#c51b7d',
    'non_resonant': '#4d4d4d',
    'uncertain': '#7b3294',
    'invalid': '#b2182b',
}
# The palette of the verdict figures' left panels (MMR and secular/ZLK): one quantity, one colour in every panel.
# The time panels draw filtered series in INK and osculating ones in RAW (RAW_ALPHA); the colours below are the only
# others. MARK_COLOURS and STATUS_COLOURS belong to the text column and the badge, never to a panel.
MODEL = '#2c7bb6'  # a fitted or modelled quantity: force law per window and its R^2, +-2 omega0, the band +-half-width
REFERENCE = '#4d4d4d'  # reference levels and marks: zero, h = 1, nu, turning points of the angle, jumps, exit
# dsigma/dt = 0 in the rate panel and the portraits: the line a libration oscillates about (sigma_dot changes sign at
# each turning point) and the level of its centre (sigma_c, 0) in a portrait; the colour of the centre of sigma
RATE_ZERO = {'color': CENTRE, 'lw': 1.0, 'ls': '-', 'zorder': 1.5}
AXIS_EXACT = REFERENCE  # a* in (d), dotted: the exact commensurability is a reference level
AXIS_NOMINAL = '#00acc1'  # the catalogue a_res in (d), dashed (cyan: far from the blue +-2 omega0, author's P14)
ZONE_FILL = '#deebf7'  # the separatrix zone [h_lo, h_hi] (a modelled band, light)
SUBRESONANCE = '#e66101'  # the accepted subresonance
OWNER = '#5e3c99'  # the owner's angle and centre
# Accepted episodes (fill, labels '#id', R^2 of the halves, the episode portrait), cycled; no grey among them
EPISODE_COLOURS = ('#1b9e77', '#e6ab02', '#e7298a', '#66a61e', '#a6761d')
EPISODE_ALPHA = 0.15
EPISODE_BEFORE_JUMP_HATCH = '////'  # accepted episodes before the last jump of a (outside the last regime)
REJECTED = '#9a9a9a'  # runs of turning points that were not accepted (a strip with the reason's letter)
REJECTED_ALPHA = 0.55
# Centres of the other members of the multiplet in (j): pale, distinct from everything above
MEMBER_COLOURS = ('#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#bdbdbd')
FRAME = '#999999'  # frames of the time panels
TEXT_KEY = '#4a4a4a'
TEXT_MISSING = '#8c8c8c'
TEXT_HEADER_FILL = '#ebebeb'
TEXT_GROUP = '#2b2b2b'  # bold group headers inside a block
NOTE = '#666666'
