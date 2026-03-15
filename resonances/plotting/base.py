"""Base plotter with shared save/show/close methods."""

from pathlib import Path
from typing import Union

from resonances.logger import logger


class BasePlotter:
    """Mixin providing save/show/close for any plotter with a _figure attribute."""

    _figure = None

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
