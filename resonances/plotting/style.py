"""Figure styles and plot options of the MMR diagnostic plots.

Two styles share one layout: `screen` (default) for reviewing many PNGs, `paper` for a
two-column journal figure (180 mm wide, 8 pt serif, no title — the caption carries it).
Styles are applied through `matplotlib.rc_context`, so global settings stay untouched.
"""

import copy
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np


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
RAW = '#a3a9b1'
CENTRE = '#b2182b'
OUT_OF_FOCUS = '#cfd4da'
ENVELOPE = '#3a3a3a'  # filtered series drawn as a min-max band


@lru_cache(maxsize=1)
def time_cmap():
    """Viridis without its palest yellow end, which vanishes on a white background."""
    import matplotlib
    from matplotlib.colors import ListedColormap

    return ListedColormap(matplotlib.colormaps['viridis'](np.linspace(0.0, 0.9, 256)), name='viridis_trunc')


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
}

# Options whose value must be a member of an enum, by dotted path.
_CHOICES = {'style': PlotStyle, 'raw': RawMode, 'series': SeriesMode, 'portrait.y': PortraitY}


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
