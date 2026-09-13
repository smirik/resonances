"""
Free argument of pericentre, in three figures.

    drift    — ω accumulated over the baseline, free against osculating. A confined band
               is libration, a ramp is circulation at g - s.
    portrait — the Kozai diagram in free variables: free e against free ω, coloured by
               time. A librator draws a closed loop; a circulator draws a wave that runs
               through every ω. The centre is not always 90/270 degrees — inside a
               mean-motion resonance it shifts (Vinogradova 2024).
    vector   — the free eccentricity vector, drawn on exactly the same axes as the
               osculating `EccentricityVectorPlotter` figure of the same body. Side by
               side the two answer the question the osculating plane alone cannot: is the
               ring that bounds ω the object's own, or the forced offset?

No figure carries numbers. The verdict and every rate live in summary.csv — `free_gate`,
`free_b_varpi`, `free_b_Omega`, `free_b_omega`, `free_rho` — see `secular.free_elements`
and `resonance.omega_free_gate`.
"""

from typing import Optional

import numpy as np

from resonances.logger import logger
from .base import BasePlotter, time_unit
from .phase_plots import _scatter_colored_by_time

_FREE_COLOR = '#1f6fb4'
_OSC_COLOR = '#9a9a9a'
_MARK_COLOR = '#c1272d'

# Above this the drift no longer reads as an angle and is shown in turns instead.
DEGREES_SHOWN = 1080.0


class FreeOmegaPlotter(BasePlotter):
    """
    Plotter for the free (proper) argument of pericentre of one body.

    Examples
    --------
        >>> plotter = FreeOmegaPlotter.from_body(body)
        >>> plotter.plot_drift().save('free-omega-drift.png')
        >>> plotter.plot_portrait().save('free-omega-portrait.png')
        >>> plotter.close()
    """

    def __init__(self):
        self._elements = None
        self._omega_osculating: Optional[np.ndarray] = None
        self._metadata: Optional[dict] = None
        self._figure = None
        self._axes = None

    @classmethod
    def from_elements(
        cls,
        elements,
        omega_osculating: Optional[np.ndarray] = None,
        body_name: str = '',
        resonance_key: str = '',
        status=None,
    ) -> 'FreeOmegaPlotter':
        """Create a plotter from a `secular.free_elements.FreeElements`.

        `omega_osculating` is the raw unwrapped argument of pericentre on the *same*
        grid as `elements.times`; it only adds the grey comparison curve.
        """
        plotter = cls()
        plotter._elements = elements
        plotter._omega_osculating = omega_osculating
        plotter._metadata = {'body_name': body_name, 'resonance': resonance_key, 'resonance_key': resonance_key, 'status': status}
        return plotter

    @classmethod
    def from_body(cls, body, resonance=None) -> 'FreeOmegaPlotter':
        """Create a plotter for a body whose free elements have already been computed."""
        elements = body.free_elements
        osculating = None
        if elements is not None and body.omega is not None and body.times is not None:
            # body.omega lives on the integration grid; the free omega lives on a resampled
            # grid thinned further by the origin masks, so interpolate onto that one.
            years = np.asarray(body.times) / (2 * np.pi)
            osculating = np.unwrap(np.interp(elements.omega_times, years, np.unwrap(body.omega)))
        return cls.from_elements(
            elements,
            omega_osculating=osculating,
            body_name=body.name,
            resonance_key=resonance.to_s() if resonance is not None else '',
            status=body.statuses.get(resonance.to_s()) if resonance is not None else None,
        )

    def plot_drift(self, figsize: tuple = (7.5, 2.25), dpi: int = 150) -> 'FreeOmegaPlotter':
        """Accumulated drift of the free omega over the baseline, against the osculating one."""
        ax = self._new_figure(figsize, dpi)
        if ax is None:
            return self

        elements = self._elements
        grid = elements.omega_times
        divisor, unit = time_unit(grid[-1] - grid[0])
        times = (grid - grid[0]) / divisor
        free = elements.omega_free - elements.omega_free[0]
        curves = [free] if self._omega_osculating is None else [free, self._omega_osculating - self._omega_osculating[0]]

        turns = max(np.ptp(curve) for curve in curves) > np.radians(DEGREES_SHOWN)
        scale, label = (1.0 / (2 * np.pi), r'$\omega$ drift, turns') if turns else (180.0 / np.pi, r'$\omega$ drift, deg')

        # The osculating curve goes underneath and much wider, so that where the two
        # coincide — which is most of the time, and exactly where it matters least to hide
        # it — the grey still reads as a halo around the blue instead of vanishing under
        # it or under the x axis.
        if self._omega_osculating is not None:
            ax.plot(times, curves[1] * scale, color=_OSC_COLOR, lw=3.0, solid_capstyle='round', label=r'osculating $\omega$')
        ax.plot(times, free * scale, color=_FREE_COLOR, lw=1.0, label=r'free $\omega$')

        # Both curves start at zero by construction, so the origin is a real data point:
        # anchoring the axes to it makes the total accumulated drift readable as the
        # distance from the corner, and puts the end of the integration at the right edge.
        ax.set_xlim(0, times[-1])
        lowest = min(0.0, min(curve.min() for curve in curves) * scale)
        highest = max(0.0, max(curve.max() for curve in curves) * scale)
        ax.set_ylim(lowest, highest if highest > lowest else lowest + 1.0)

        ax.set_xlabel(f'time, {unit}')
        ax.set_ylabel(label)
        ax.grid(alpha=0.22, lw=0.5)
        ax.legend(frameon=False, fontsize=9, loc='best')
        ax.set_title(self._title(), fontsize=12, pad=8)
        self._figure.tight_layout()
        return self

    def plot_portrait(self, figsize: tuple = (7.5, 5.5), dpi: int = 150) -> 'FreeOmegaPlotter':
        """Free e against free omega — the Kozai diagram, in free variables."""
        ax = self._new_figure(figsize, dpi)
        if ax is None:
            return self

        elements = self._elements
        times = elements.omega_times
        amplitude = elements.e_free_series[elements.keep_e & elements.keep_i]
        _scatter_colored_by_time(ax, np.degrees(elements.omega_free_wrapped), amplitude, times, marker_size=3.5)
        for centre in (90, 270):
            ax.axvline(centre, color=_MARK_COLOR, ls=':', lw=0.9)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'free $\omega$, deg', fontsize=12)
        ax.set_ylabel('free $e$', fontsize=12)
        ax.grid(alpha=0.22, lw=0.5)
        ax.set_title(self._title(), fontsize=12, pad=8)
        self._figure.tight_layout()
        return self

    def plot_vector(self, figsize: tuple = (7, 6), dpi: int = 150, limit: Optional[float] = None) -> 'FreeOmegaPlotter':
        """The free eccentricity vector, in the same plane as the osculating portrait.

        `limit` is the half-width of the square window; pass the osculating figure's
        `EccentricityVectorPlotter.extent()` so the two can be read against each other.
        Overlaying them instead would only add noise, hence a separate image.
        """
        ax = self._new_figure(figsize, dpi)
        if ax is None:
            return self

        elements = self._elements
        k, h = elements.free_eccentricity_vector()
        _scatter_colored_by_time(ax, k, h, elements.omega_times, marker_size=7.5)
        ax.axhline(0, color=_OSC_COLOR, ls='--', lw=0.5, alpha=0.7)
        ax.axvline(0, color=_OSC_COLOR, ls='--', lw=0.5, alpha=0.7)
        if limit is None:
            largest = np.nanmax(np.abs(np.concatenate([k, h]))) if len(k) else 0.0
            limit = 1.05 * float(largest) if np.isfinite(largest) and largest > 0 else 1.0
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'free $e\cos\omega$', fontsize=12)
        ax.set_ylabel(r'free $e\sin\omega$', fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_title(self._title(), fontsize=12, pad=8)
        self._figure.tight_layout()
        return self

    def _new_figure(self, figsize: tuple, dpi: int):
        """Open a fresh single-axes figure, or None when there is nothing to draw."""
        import matplotlib.pyplot as plt

        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
        if self._elements is None:
            logger.warning("No free elements available for plotting")
            return None

        self._figure, ax = plt.subplots(figsize=figsize, dpi=dpi)
        self._axes = [ax]
        return ax

    def _title(self) -> str:
        """Body name only: the decomposition is a property of the orbit, not of a resonance."""
        name = (self._metadata or {}).get('body_name', '')
        return f'{name} - free $\\omega$' if name else r'Free $\omega$'

    def close(self):
        """Close the figure to free memory."""
        super().close()
        self._axes = None
