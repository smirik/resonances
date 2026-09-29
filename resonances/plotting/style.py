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


class PortraitY(StrEnum):
    RATE = 'rate'  # sigma_dot
    AXIS = 'axis'  # a - a0


@dataclass(frozen=True)
class FigureStyle:
    width: float  # inches (combined figure)
    height: float  # inches (combined figure)
    panel_size: float  # inches, one standalone panel
    dpi: int
    font_size: float
    font_family: str
    title: bool  # figure title (body and resonance)
    line: float  # filtered series
    raw_line: float  # raw series
    marker: float  # scatter point area

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
        height=9.0,
        panel_size=6.0,
        dpi=150,
        font_size=11,
        font_family='sans-serif',
        title=True,
        line=1.2,
        raw_line=0.8,
        marker=4.0,
    ),
    PlotStyle.PAPER: FigureStyle(
        width=7.09,  # 180 mm
        height=5.4,
        panel_size=3.46,  # 88 mm, one column
        dpi=300,
        font_size=8,
        font_family='serif',
        title=False,
        line=0.8,
        raw_line=0.5,
        marker=1.2,
    ),
}

# Colours. Time is the only quantity in colour; everything else is ink or grey.
INK = 'black'
RAW = '#a3a9b1'
CENTRE = '#b2182b'
OUT_OF_FOCUS = '#cfd4da'


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
    'recurrence': {'max_points': 700, 'exclude_samples': 2},
    'portrait': {'y': PortraitY.RATE},
    'cycles': {'prominence': None},  # rad; None = 5% of the angle range within [0.005, 0.1]
    'fair': {'max_step_fraction': 0.25},  # warn when the output step exceeds this share of the orbit
}

# Options whose value must be a member of an enum, by dotted path.
_CHOICES = {'style': PlotStyle, 'raw': RawMode, 'portrait.y': PortraitY}


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
