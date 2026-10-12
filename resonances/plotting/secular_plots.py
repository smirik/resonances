"""Diagnostic figures of a secular angle — what the phase portrait is for an MMR.

    angle    — sigma(t), degrees: the osculating angle in grey (as for every resonance type), the
               classified angle smoothed in black, its turning points as red circles;
    rate     — the rate of that smoothed angle, arcsec/yr, with zero and the mean rate (the divisor);
    portrait — (r cos sigma, r sin sigma), r = e when sigma holds varpi, else i from the invariable
               plane in degrees: osculating in grey, classified coloured by time;
    combined — the three on one figure.

The black curve is the centre of the status check (`resonance/secular_centre.analyse`), except when
that centre stands still: the libration is then faster than the window (nu6 of (1222) Tina, 600 kyr)
and the curve follows it on a quarter of the period the classifier found (`libration_period_1`).
Nothing here changes a status. Reading guide: docs/plots.md, "Secular diagnostics".
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np
from matplotlib.colors import Normalize

from resonances.resonance import oscillations as osc
from resonances.resonance import secular_centre
from resonances.secular.free_elements import RAD_PER_ARCSEC, rate
from resonances.secular.proper_angle import inclination_from_invariable_plane, inclination_from_vector, reduced_vectors
from .base import DiagnosticPlotter, time_unit
from .style import CENTRE, INK, RAW, SECULAR_PLOT_KINDS, panel_letter, time_cmap, time_colorbar, time_label

FAST_WINDOW_MIN = 50_000.0  # years: the shortest window the curve of a fast libration gets


@dataclass
class SecularDiagnostics:
    """What the secular figures show for one body and one angle (times in years, in time order)."""

    times: np.ndarray
    sigma: np.ndarray  # the classified angle, unwrapped, rad
    osculating: np.ndarray  # the angle from the osculating elements, unwrapped, rad
    radius: np.ndarray  # e, or i (deg) from the invariable plane, of the classified angle's vectors
    osculating_radius: np.ndarray
    p: int  # coefficient of varpi: the radius is e when it is non-zero
    config: object  # SimulationConfig: the centre-check settings
    libration_period: Optional[float] = None  # years, from the classifier
    body_name: str = ''
    resonance: str = ''
    status: Optional[int] = None
    divisor: float = field(init=False)  # years per displayed time unit
    unit: str = field(init=False)
    centre: secular_centre.CentreCheck = field(init=False)
    window: float = field(init=False)  # years, smoothing of `curve`
    curve: np.ndarray = field(init=False)  # deg, NaN at the ends
    turning: np.ndarray = field(init=False)  # indices into `curve`
    mean_rate: float = field(init=False)  # arcsec/yr, the divisor of the angle

    def __post_init__(self):
        if len(self.times) > 1 and self.times[-1] < self.times[0]:  # a backward integration, put in time order
            for name in ('times', 'sigma', 'osculating', 'radius', 'osculating_radius'):
                setattr(self, name, np.asarray(getattr(self, name))[::-1])
        self.divisor, self.unit = time_unit(abs(self.times[-1] - self.times[0]))
        self.centre = secular_centre.analyse(self.times, self.sigma, self.config)
        self.window = self.config.secular_centre_window * abs(self.times[-1] - self.times[0])
        self.curve, self.turning = self.centre.line, self.centre.turning
        if self.centre.drift_deg < self.config.secular_centre_drift and self.libration_period:
            # the centre stands still: follow the faster libration on a quarter of its period
            self.window = float(np.clip(self.libration_period / 4.0, FAST_WINDOW_MIN, self.window))
            self.curve = np.degrees(osc.smooth(self.times, self.sigma, self.window))
            self.turning = osc.turning_points(self.curve, self.config.secular_turning_prominence)
        self.mean_rate = rate(self.times, self.sigma)

    @property
    def t(self) -> np.ndarray:
        return self.times / self.divisor

    @cached_property
    def rate(self) -> np.ndarray:
        """d(curve)/dt in arcsec/yr, smoothed once more: a derivative amplifies the ripple left by one pass."""
        derivative = osc.smooth(self.times, self.sigma, self.window, deriv=1) / RAD_PER_ARCSEC
        inside = np.isfinite(derivative)
        out = np.full(len(derivative), np.nan)
        out[inside] = osc.smooth(self.times[inside], derivative[inside], self.window)
        return out

    def time_norm(self) -> Normalize:
        return Normalize(self.t[0], self.t[-1])

    @classmethod
    def from_body(cls, body, resonance, config) -> 'SecularDiagnostics':
        key = resonance.to_s()
        times = np.asarray(body.times) / (2 * np.pi)
        p = sum(t.coefficient for t in resonance.formula.terms if t.index is None and t.mode == 'g')
        osculating_radius = np.asarray(body.ecc) if p else np.degrees(inclination_from_invariable_plane(body.inc, body.Omega))
        radius = osculating_radius
        if config.secular_reduced:
            z_e, z_i = reduced_vectors(times, body.ecc, body.varpi, body.inc, body.Omega, resonance.formula.terms, ('e',) if p else ('i',))
            radius = np.abs(z_e) if p else np.degrees(inclination_from_vector(z_i))
        libration = body.librations.get(key)
        return cls(
            times=times,
            sigma=np.asarray(body.angles_unwrapped[key], dtype=float),
            osculating=np.unwrap(np.asarray(body.secular_angles_osculating.get(key, body.angles.get(key)), dtype=float)),
            radius=radius,
            osculating_radius=osculating_radius,
            p=p,
            config=config,
            libration_period=getattr(getattr(libration, 'metrics', None), 'libration_period_1', None),
            body_name=body.name,
            resonance=key,
            status=body.statuses.get(key),
        )


def draw_angle(ax, d: SecularDiagnostics, style):
    """sigma(t), degrees. The osculating angle is folded mod 360 into the band of a bounded curve when it
    turns, or its ramp would flatten everything else to a line."""
    s = np.degrees(d.osculating)
    span = np.nanmax(d.curve) - np.nanmin(d.curve)
    if span < 360.0 and np.ptp(s) > span + 360.0:
        low = float(np.nanmean(d.curve)) - 180.0
        ax.scatter(d.t, (s - low) % 360.0 + low, color=RAW, s=style.series_marker / 4, linewidths=0, rasterized=True)
        ax.set_ylim(low, low + 360.0)
    else:
        ax.plot(d.t, s, color=RAW, lw=style.raw_line * 0.6, rasterized=True)
    ax.plot(d.t, d.curve, color=INK, lw=style.line, zorder=4)
    if len(d.turning):
        ax.plot(d.t[d.turning], d.curve[d.turning], 'o', mfc='none', mec=CENTRE, ms=5, mew=1.2, zorder=5)
    ax.set_xlim(d.t[0], d.t[-1])
    ax.set_xlabel(time_label(d.unit))
    ax.set_ylabel(r'$\sigma$ (deg)')
    ax.grid(alpha=0.15, lw=0.5)


def draw_rate(ax, d: SecularDiagnostics, style):
    """Rate of the smoothed angle, arcsec/yr; zero in grey, the mean rate dashed."""
    ax.axhline(0.0, color=RAW, lw=0.8)
    ax.axhline(d.mean_rate, color=INK, ls='--', lw=0.9)
    ax.plot(d.t, d.rate, color=INK, lw=style.line)
    ax.set_xlim(d.t[0], d.t[-1])
    ax.set_xlabel(time_label(d.unit))
    ax.set_ylabel(r'$\dot\sigma$ (arcsec/yr)')
    ax.grid(alpha=0.15, lw=0.5)


def draw_portrait(ax, d: SecularDiagnostics, style, norm: Normalize):
    """(r cos sigma, r sin sigma): osculating in grey, classified coloured by time, origin marked."""
    r, s = d.osculating_radius, d.osculating
    ax.scatter(r * np.cos(s), r * np.sin(s), color=RAW, s=style.marker / 2, linewidths=0, alpha=0.6, rasterized=True)
    r2, s2 = d.radius, d.sigma
    ax.scatter(r2 * np.cos(s2), r2 * np.sin(s2), c=d.t, norm=norm, cmap=time_cmap(), s=style.marker / 2, linewidths=0, rasterized=True)
    limit = 1.05 * float(np.nanmax(np.concatenate([np.abs(r), np.abs(r2)]))) or 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color=RAW, lw=0.5, ls='--')
    ax.axvline(0, color=RAW, lw=0.5, ls='--')
    ax.plot(0, 0, '+', color=CENTRE, ms=8, mew=1.3)
    name, unit = ('e', '') if d.p else ('i', ' (deg)')
    ax.set_xlabel(rf'${name}\cos\sigma${unit}')
    ax.set_ylabel(rf'${name}\sin\sigma${unit}')
    ax.grid(alpha=0.15, lw=0.5)


class SecularPlotter(DiagnosticPlotter):
    """Standalone and combined diagnostic figures of one body in one secular resonance.

    Examples
    --------
        >>> plotter = SecularPlotter.from_body(body, resonance, sim, options={'style': 'paper'})
        >>> plotter.plot('portrait')  # by kind name, as `plots` in the config
        >>> plotter.save('portrait.pdf').close()
    """

    KINDS = SECULAR_PLOT_KINDS
    KIND_NAME = 'secular'

    @classmethod
    def from_body(cls, body, resonance, sim=None, options=None) -> 'SecularPlotter':
        """The same signature as `MMRPlotter.from_body`; the centre-check settings come from `sim.config`."""
        if sim is None:
            from resonances.simulation.config import SimulationConfig

            config = SimulationConfig(save=None, plot=None, save_summary=False)
        else:
            config = sim.config
        return cls(SecularDiagnostics.from_body(body, resonance, config), options)

    def title(self) -> str:
        d = self.d
        status = '' if d.status is None else f', status {d.status}'
        return f'Asteroid {d.body_name}, {d.resonance}{status}; divisor {d.mean_rate:+.3f}"/yr'

    def plot_combined(self) -> 'SecularPlotter':
        s, norm = self.style, self.d.time_norm()
        with self._figure_of(s.panel_size * 2.4, s.panel_size * 1.1) as fig:
            grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0])
            ax_angle, ax_rate, ax_portrait = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
            draw_angle(ax_angle, self.d, s)
            draw_rate(ax_rate, self.d, s)
            draw_portrait(ax_portrait, self.d, s, norm)
            time_colorbar(fig, norm, time_label(self.d.unit), ax=ax_portrait, fraction=0.046, pad=0.03)
            for ax, letter in zip((ax_angle, ax_rate, ax_portrait), 'abc'):
                panel_letter(ax, letter)
        return self

    def _series_panel(self, draw) -> 'SecularPlotter':
        with self._figure_of(self.style.panel_size * 1.25, self.style.panel_size * 0.5) as fig:
            draw(fig.add_subplot(), self.d, self.style)
        return self

    def plot_angle(self) -> 'SecularPlotter':
        return self._series_panel(draw_angle)

    def plot_rate(self) -> 'SecularPlotter':
        return self._series_panel(draw_rate)

    def plot_portrait(self) -> 'SecularPlotter':
        return self._time_panel(1.2, lambda ax, norm: draw_portrait(ax, self.d, self.style, norm))
