"""
The eccentricity-vector plane (k, h) = (e*cos(omega), e*sin(omega)) coloured by time.

The resonant-angle portraits of MMRs live in `mmr_plots`.
"""

import numpy as np
from typing import Optional

from resonances.body import Body
from resonances.logger import logger
from .base import BasePlotter, time_unit, title_prefix
from .style import time_cmap, time_label


def _scatter_colored_by_time(ax, x, y, times, marker_size: float = 7.5):
    """Scatter `y` against `x`, coloured by time, with a colorbar labelled in its own unit."""
    import matplotlib.pyplot as plt

    divisor, unit = time_unit(times[-1] - times[0]) if len(times) > 1 else (1.0, 'yr')
    scatter = ax.scatter(x, y, c=np.asarray(times) / divisor, cmap=time_cmap(), s=marker_size, marker='.', linewidths=0, alpha=0.7)
    plt.colorbar(scatter, ax=ax).set_label(time_label(unit), fontsize=10)
    return scatter


class EccentricityVectorPlotter(BasePlotter):
    """
    Plotter for the non-singular eccentricity plane k = e*cos(omega), h = e*sin(omega).

    The plane separates the forced eccentricity (offset of the cloud from the origin)
    from the free one (its radius). A cloud that does not enclose the origin keeps omega
    bounded whatever the dynamics — which is why this figure explains *why* omega
    librates, but does not on its own say whether the libration is a Lidov-Kozai
    resonance. Genuine librators (3040, 4257, 162474) and the forced-geometry case
    (591986) all show an offset cloud; only the e-i exchange tells them apart.

    Examples
    --------
        >>> plotter = EccentricityVectorPlotter.from_body(body, resonance, sim)
        >>> plotter.plot().save('ecc-vector.png')
        >>> plotter.close()
    """

    def __init__(self):
        self._times: Optional[np.ndarray] = None
        self._k: Optional[np.ndarray] = None
        self._h: Optional[np.ndarray] = None
        self._metadata: Optional[dict] = None
        self._figure = None
        self._axes = None

    @classmethod
    def from_body(cls, body: Body, resonance=None, sim=None) -> 'EccentricityVectorPlotter':
        """Create a plotter from a Body. `resonance` and `sim` only enrich the title."""
        plotter = cls()
        plotter._k, plotter._h = body.eccentricity_vector()
        times = sim.times if sim is not None else body.times
        plotter._times = None if times is None else np.asarray(times) / (2 * np.pi)
        plotter._metadata = {
            'body_name': body.name,
            'resonance': resonance.to_short() if resonance is not None else '',
            'resonance_key': resonance.to_s() if resonance is not None else '',
            'status': body.statuses.get(resonance.to_s(), 0) if resonance is not None else None,
        }
        return plotter

    @classmethod
    def from_data(cls, times: np.ndarray, ecc: np.ndarray, omega: np.ndarray, body_name: str = '', resonance_key: str = ''):
        """Create a plotter from raw arrays (times in years, omega in radians)."""
        from resonances.resonance.cross_spectrum import eccentricity_vector

        plotter = cls()
        plotter._times = times
        plotter._k, plotter._h = eccentricity_vector(ecc, omega)
        plotter._metadata = {'body_name': body_name, 'resonance': resonance_key, 'resonance_key': resonance_key}
        return plotter

    def plot(
        self, figsize: tuple = (7, 6), dpi: int = 150, marker_size: float = 7.5, limit: Optional[float] = None
    ) -> 'EccentricityVectorPlotter':
        """Draw the (k, h) plane coloured by time.

        `limit` overrides the half-width of the window, so that a free-vector portrait can
        be drawn on exactly the same scale as the osculating one — comparing the two by eye
        is the whole point of having both.
        """
        import matplotlib.pyplot as plt

        if self._k is None or self._h is None or self._times is None:
            logger.warning("No eccentricity vector data available")
            return self

        if self._figure is not None:
            plt.close(self._figure)

        self._figure, ax_plane = plt.subplots(figsize=figsize, dpi=dpi)
        self._axes = [ax_plane]

        _scatter_colored_by_time(ax_plane, self._k, self._h, self._times, marker_size=marker_size)
        ax_plane.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_plane.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_plane.set_xlabel(r'$e\cos\omega$', fontsize=12)
        ax_plane.set_ylabel(r'$e\sin\omega$', fontsize=12)
        # A square window centred on the origin, sized by the data. Two things follow that
        # matter when the figures are compared side by side: the drawn box is the same in
        # every figure (an equal-aspect box fitted to the data would shrink to the shape of
        # the cloud), and the origin is always in view — which is the whole reading of this
        # plane, since a cloud that does not enclose it keeps omega bounded.
        limit = self.extent() if limit is None else limit
        ax_plane.set_xlim(-limit, limit)
        ax_plane.set_ylim(-limit, limit)
        ax_plane.set_aspect('equal', adjustable='box')
        ax_plane.grid(alpha=0.3)
        ax_plane.set_title(self._title(), fontsize=12)

        plt.tight_layout()
        return self

    def extent(self) -> float:
        """Half-width of a square window holding both the cloud and the origin."""
        largest = np.nanmax(np.abs(np.concatenate([self._k, self._h])))
        return 1.05 * float(largest) if np.isfinite(largest) and largest > 0 else 1.0

    def _title(self) -> str:
        prefix = title_prefix(self._metadata)
        return f'{prefix} - eccentricity vector' if prefix else 'Eccentricity vector'

    def close(self):
        """Close the figure to free memory."""
        super().close()
        self._axes = None
