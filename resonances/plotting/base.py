"""Base plotter with shared save/show/close methods."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from resonances.logger import logger


def time_unit(span_years: float) -> Tuple[float, str]:
    """Divisor and name for displaying times of order `span_years`.

    One unit is chosen for a whole axis, and each switch happens as soon as the unit
    would otherwise run into four digits: a 10 Myr baseline reads as 10 Myr, not as
    10000 kyr. The sign of the span (a backward run) does not matter.
    """
    span_years = abs(span_years)
    if not np.isfinite(span_years) or span_years < 1e4:
        return 1.0, 'yr'
    if span_years < 1e6:
        return 1e3, 'kyr'
    return 1e6, 'Myr'


def format_period(period_years: float) -> str:
    """A single period written in whichever unit suits it, e.g. '52.1 kyr', '1.4 Myr'."""
    divisor, name = (1e6, 'Myr') if period_years >= 1e6 else time_unit(period_years)
    value = period_years / divisor
    digits = 0 if value >= 100 else 1
    return f'{value:.{digits}f} {name}'


def title_prefix(metadata: dict) -> str:
    """'<body>: <resonance>' from a plotter's metadata, skipping whatever is missing."""
    metadata = metadata or {}
    parts = (metadata.get('body_name', ''), metadata.get('resonance') or metadata.get('resonance_key', ''))
    return ': '.join(part for part in parts if part)


class BasePlotter:
    """Mixin providing save/show/close for any plotter with a _figure attribute."""

    _figure = None

    @property
    def figure(self):
        """The current matplotlib Figure, or None when nothing has been drawn."""
        return self._figure

    def save(self, path: Union[str, Path], **kwargs):
        """Save the figure to file."""
        if self._figure is None:
            raise RuntimeError("Must call a plot method before save()")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(path, **kwargs)
        logger.info(f"Plot saved to {path}")

        return self

    def show(self):
        """Display the figure."""
        if self._figure is None:
            raise RuntimeError("Must call a plot method before show()")

        import matplotlib.pyplot as plt

        plt.show()
        return self

    def close(self):
        """Close the figure to free memory."""
        import matplotlib.pyplot as plt

        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None


class DiagnosticPlotter(BasePlotter):
    """The contract shared by the per-resonance diagnostic plotters (MMR, secular).

    Subclasses set `KINDS`, implement `plot_<kind>` for each and `title()`; their diagnostics
    object `d` provides `unit` and `time_norm()`. Figures are drawn and saved in the style of
    `options['style']` (`plotting.style.STYLES`), titled when the style asks for it.
    """

    KINDS: Tuple[str, ...] = ()

    def __init__(self, diagnostics, options: Optional[Dict[str, Any]] = None):
        from .style import STYLES, resolve_plot_options

        self.d = diagnostics
        self.options = resolve_plot_options(options)
        self.style = STYLES[self.options['style']]
        self._figure = None

    def plot(self, kind: str) -> bool:
        """Draw one plot kind; False when there is nothing to draw."""
        if kind not in self.KINDS:
            raise ValueError(f"Unknown {self.KIND_NAME} plot kind '{kind}'. Known: {', '.join(self.KINDS)}")
        if self._figure is not None:
            self.close()
        getattr(self, f'plot_{kind}')()
        return self._figure is not None

    KIND_NAME = 'diagnostic'

    def title(self) -> str:
        return ''

    def save(self, path, **kwargs):
        import matplotlib as mpl

        kwargs.setdefault('dpi', self.style.dpi)
        with mpl.rc_context(self.style.rc()):
            return super().save(path, **kwargs)

    @contextmanager
    def _figure_of(self, width: float, height: float, tight: bool = True):
        """New figure in the style's rc context; titled (and tightened) on exit."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if self._figure is not None:
            plt.close(self._figure)
        with mpl.rc_context(self.style.rc()):
            self._figure = plt.figure(figsize=(width, height), dpi=self.style.dpi)
            yield self._figure
            title = self.title()
            if self.style.title and title:
                self._figure.suptitle(title, y=0.985, fontsize=self.style.font_size + 3)
            if tight:
                self._figure.tight_layout()

    def _time_panel(self, aspect: float, draw):
        """One panel coloured by time, with its time colour bar; `draw(ax, norm)` fills it."""
        from .style import time_colorbar, time_label

        norm = self.d.time_norm()
        with self._figure_of(self.style.panel_size * aspect, self.style.panel_size) as fig:
            ax = fig.add_subplot()
            draw(ax, norm)
            time_colorbar(fig, norm, time_label(self.d.unit), ax=ax, fraction=0.046, pad=0.03)
        return self
