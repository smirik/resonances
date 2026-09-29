"""Base plotter with shared save/show/close methods."""

from pathlib import Path
from typing import Tuple, Union

import numpy as np

from resonances.logger import logger


def time_unit(span_years: float) -> Tuple[float, str]:
    """Divisor and name for displaying times of order `span_years`.

    One unit is chosen for a whole axis, and each switch happens as soon as the unit
    would otherwise run into four digits: a 10 Myr baseline reads as 10 Myr, not as
    10000 kyr.
    """
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
