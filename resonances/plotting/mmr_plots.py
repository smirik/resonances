"""Diagnostic plots of a mean-motion resonance.

`MMRDiagnostics` gathers the arrays of one body-resonance pair and derives the
oscillation quantities once (see `resonances.resonance.oscillations`). Every panel is
a `draw_*` function drawing onto given axes (the standalone figures of `MMRPlotter`;
the combined MMR figure is the verdict figure of `plotting/mmr_combined.py`):

    recurrence  distance between states (sigma, sigma_dot) at two times
    fair        mean anomaly against the mean-longitude difference (prograde two-body only)
    portrait    sigma against sigma_dot (or a), coloured by time
    cycles      the sigma cycles, and a over the same cycles, stacked on one period

Times are in years, angles in radians internally; the portrait shows degrees. Every a
drawn here (portrait, cycles, the reference a0) is the analysis' own a_f, else the
osculating a, never the body-level `axis_filtered` (its 2000-yr low-pass erases a faster
MMR libration). Nothing here changes a status.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from resonances.logger import logger
from resonances.resonance import oscillations as osc
from resonances.resonance.classify.util import calc_sigma_derivative
from .base import DiagnosticPlotter, time_unit
from .style import (
    INK,
    MMR_PLOT_KINDS,
    OUT_OF_FOCUS,
    RATE_ZERO,
    RAW,
    FigureStyle,
    PortraitY,
    RawMode,
    resolve_plot_options,
    time_cmap,
    time_colorbar,
)

TWO_PI = 2 * np.pi
MILLI = 1e3  # a - a0 is shown in 10^-3 au


@dataclass
class PlanetSeries:
    name: str
    longitude: np.ndarray  # mean longitude, rad
    axis: np.ndarray  # au


@dataclass
class MMRDiagnostics:
    """Arrays of one body in one MMR plus the derived oscillation quantities.

    A backward integration (decreasing times) is put in time order here, so everything
    downstream sees increasing time; the axes still show the actual, negative, times.
    `focus` = (start, end) in years restricts cycles, portraits and the recurrence
    scales to an interval; outside it the time series are dimmed. None = whole record.
    """

    times: np.ndarray  # years
    sigma: np.ndarray  # raw angle wrapped to [0, 2pi)
    sigma_raw: np.ndarray  # raw angle, unwrapped
    sigma_filtered: np.ndarray  # low-pass filtered angle, unwrapped
    axis: np.ndarray  # osculating a, au
    body_name: str = ''
    resonance: str = ''  # full key, e.g. 2J-1+0-1
    label: str = ''  # short notation for titles, e.g. 2J-1
    n_planets: int = 1
    retrograde: bool = False  # a retrograde (i > 90 deg) resonance: no FAIR
    mean_anomaly: Optional[np.ndarray] = None
    longitude: Optional[np.ndarray] = None
    planet: Optional[PlanetSeries] = None
    focus: Optional[Tuple[float, float]] = None
    prominence: Optional[float] = None
    edge: float = 0.0  # years dropped at both ends of the portrait: the low-pass filter's edge transient
    # a of the cycles, the portrait (y='axis') and a0: the analysis' own a_f (at the angle's cutoff, as the combined
    # figure's (d)), else the osculating a; never `body.axis_filtered`, whose low-pass at the run's cutoff (2000 yr by
    # default) erases a faster libration
    axis_analysis: Optional[np.ndarray] = None

    mask: np.ndarray = field(init=False)
    rate: np.ndarray = field(init=False)  # d sigma_filtered / dt, rad/yr
    raw_rate: np.ndarray = field(init=False)
    cycles: List[osc.Cycle] = field(init=False)
    axis_reference: float = field(init=False)  # a0: median of `axis_analysis` over the focus
    divisor: float = field(init=False)  # years per display time unit
    unit: str = field(init=False)  # 'yr', 'kyr' or 'Myr'
    _recurrences: Dict[Tuple[int, int], osc.Recurrence] = field(init=False, default_factory=dict, repr=False)

    _SERIES = ('times', 'sigma', 'sigma_raw', 'sigma_filtered', 'axis', 'mean_anomaly', 'longitude', 'axis_analysis')

    def __post_init__(self):
        if self.axis_analysis is None:
            self.axis_analysis = self.axis
        for name in self._SERIES:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=float))
        if len(self.times) > 1 and self.times[-1] < self.times[0]:
            self._reverse()
        t = self.times
        self.mask = np.ones(len(t), dtype=bool) if self.focus is None else (t >= min(self.focus)) & (t <= max(self.focus))
        self.rate = calc_sigma_derivative(t, self.sigma_filtered)
        self.raw_rate = calc_sigma_derivative(t, self.sigma_raw)
        self.cycles = osc.find_cycles(t[self.mask], self.sigma_filtered[self.mask], self.prominence)
        self.axis_reference = float(np.median(self.axis_analysis[self.mask]))
        self.divisor, self.unit = time_unit(abs(t[-1] - t[0]))

    def _reverse(self):
        for name in self._SERIES:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[::-1].copy())
        if self.planet is not None:
            self.planet = PlanetSeries(self.planet.name, self.planet.longitude[::-1].copy(), self.planet.axis[::-1].copy())

    @property
    def t(self) -> np.ndarray:
        """Times in the display unit."""
        return self.times / self.divisor

    @property
    def portrait_mask(self) -> np.ndarray:
        """The focus without `edge` years at both ends of the record, where the zero-phase
        filter is distorted (kept whole if trimming would leave under half of it)."""
        inner = self.mask & (self.times >= self.times[0] + self.edge) & (self.times <= self.times[-1] - self.edge)
        return inner if inner.sum() >= self.mask.sum() / 2 else self.mask

    @property
    def has_focus(self) -> bool:
        """Whether the focus is a proper part of the record (otherwise nothing is dimmed)."""
        return not bool(self.mask.all())

    def milli_au(self, axis: np.ndarray) -> np.ndarray:
        """a - a0 in 10^-3 au."""
        return (np.asarray(axis) - self.axis_reference) * MILLI

    def recurrence(self, points: int, exclude_samples: int) -> osc.Recurrence:
        """The recurrence matrix, built once per size: the combined and standalone figures share it.

        Scales come from the focus, like the cycles; the matrix covers the whole record.
        """
        key = (points, exclude_samples)
        if key not in self._recurrences:
            scales = osc.recurrence_scales(self.sigma_filtered[self.mask], self.rate[self.mask], self.cycles)
            self._recurrences[key] = osc.recurrence(self.times, self.sigma_filtered, self.rate, *scales, points, exclude_samples)
        return self._recurrences[key]

    def time_norm(self) -> Normalize:
        return Normalize(self.t[0], self.t[-1])

    def fair_unavailable_reason(self) -> Optional[str]:
        """Why the FAIR plane cannot be drawn, or None when it can."""
        if self.n_planets != 1:
            return 'FAIR is a two-body diagnostic; a three-body argument is not tested by it'
        if self.retrograde:
            # FAIR is derived for prograde orbits; for i > 90 deg the saved `longitude` (REBOUND's l)
            # and `varpi` (Omega + omega) are not even in one convention.
            return 'FAIR is defined for prograde resonances only'
        if self.mean_anomaly is None or self.longitude is None:
            return 'no mean anomaly / mean longitude of the body'
        if self.planet is None:
            return 'no planet data (run the simulation with save_planets=True)'
        if len(self.planet.longitude) != len(self.times):
            return 'planet data do not match the body output times'
        return None

    @classmethod
    def from_body(cls, body, resonance, sim=None, focus=None, prominence=None) -> 'MMRDiagnostics':
        """From a simulated body; the portrait edge is one period of the filter cutoff.

        sigma_f is the analysis' filtered angle (`body.angles_filtered_unwrapped`); after a restore it is not in the
        data file and is reproduced by re-running the analysis on the saved series (`refiltered_angle`). The a of the
        cycles, the portrait and a0 is the analysis' a_f when its arrays are held (`body.mmr_analyses`, or that
        re-run), else the osculating a."""
        key = resonance.to_s()
        times = sim.times if sim is not None else body.times
        names = list(resonance.planets_names)
        planet = _planet_series(sim, names[0]) if len(names) == 1 and sim is not None else None
        sigma_filtered = body.angles_filtered_unwrapped.get(key)
        analysis = getattr(body, 'mmr_analyses', {}).get(key)
        if sigma_filtered is None:
            analysis = _rerun_analysis(body, resonance, sim)
            sigma_filtered = _in_body_order(body, analysis.prep.angle_f)
        axis_analysis = _in_body_order(body, analysis.prep.a_f) if analysis is not None and analysis.prep is not None else None
        return cls(
            times=np.asarray(times) / (2 * np.pi),
            sigma=body.angles[key],
            sigma_raw=body.angles_unwrapped[key],
            sigma_filtered=sigma_filtered,
            axis=body.axis,
            body_name=str(body.name),
            resonance=key,
            label=resonance.to_short(),
            n_planets=max(len(names), 1),
            retrograde=bool(getattr(resonance, 'retrograde', False)),
            mean_anomaly=body.M,
            longitude=body.longitude,
            planet=planet,
            focus=focus,
            prominence=prominence,
            edge=1.0 / sim.config.oscillations_cutoff if sim is not None else 0.0,
            axis_analysis=axis_analysis,
        )


def _rerun_analysis(body, resonance, sim):
    """The MMR analysis re-run on the saved series with the run's settings (same input, same cutoff as the verdict).

    ValueError when it cannot be: no simulation (planets, settings), or an analysis stopped at G0 (no sigma_f)."""
    key = resonance.to_s()
    if sim is None:
        raise ValueError(f'{body.name} {key}: no filtered angle in memory, and no simulation to re-run the analysis')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.simulation.mmr_adapter import series_from_body

    analysis = analyse_mmr(series_from_body(body, resonance, sim), sim.config.analysis_settings, scan_subresonances=False, scan_owner=False)
    if analysis.prep is None:
        raise ValueError(f'{body.name} {key}: the analysis is undefined ({analysis.verdict.reason}), there is no filtered angle')
    return analysis


def _in_body_order(body, values) -> Optional[np.ndarray]:
    """A time-ordered series of the analysis in the body's row order (None unless it has one value per row)."""
    from resonances.simulation.mmr_adapter import in_body_order, time_order

    return in_body_order(values, time_order(body), len(body.times))


def refiltered_angle(body, resonance, sim) -> np.ndarray:
    """sigma_f of an MMR angle reproduced from the saved series (`_rerun_analysis`), in the body's row order."""
    return _in_body_order(body, _rerun_analysis(body, resonance, sim).prep.angle_f)


def _planet_series(sim, name: str) -> Optional[PlanetSeries]:
    """Mean longitude and semi-major axis of a planet recorded by the simulation."""
    data = sim.integration_engine.planets_data.get(name)
    if not data or 'l' not in data or 'a' not in data:
        return None
    return PlanetSeries(name=name, longitude=np.asarray(data['l'], dtype=float), axis=np.asarray(data['a'], dtype=float))


# ------------------------------------------------------------------ pure geometry


def portrait_coordinates(d: MMRDiagnostics, y: PortraitY = PortraitY.RATE):
    """(x, y, x_raw, y_raw) of the portrait over the focus: sigma in degrees against
    sigma_dot in deg per display time unit, or against a - a0 in 10^-3 au (the analysis' a_f,
    `MMRDiagnostics.axis_analysis`; the grey dots are the osculating a).

    The unwrapped angle is shifted by a whole number of turns only, so a libration about
    180 deg is drawn about 180 deg while a circulation still runs off to the side. The raw
    angle is brought onto the same branch as the filtered one, sample by sample.
    """
    m = d.portrait_mask
    shift = TWO_PI * np.round(np.median(d.sigma_filtered[m]) / TWO_PI)
    x = np.degrees(d.sigma_filtered[m] - shift)
    raw_branch = TWO_PI * np.round((d.sigma_raw[m] - d.sigma_filtered[m]) / TWO_PI)
    x_raw = np.degrees(d.sigma_raw[m] - raw_branch - shift)
    if y == PortraitY.RATE:
        return x, np.degrees(d.rate[m]) * d.divisor, x_raw, np.degrees(d.raw_rate[m]) * d.divisor
    return x, d.milli_au(d.axis_analysis[m]), x_raw, d.milli_au(d.axis[m])


def recurrence_points(d: MMRDiagnostics, requested: Optional[int] = None) -> int:
    """Decimated size of the recurrence matrix: `requested`, or enough for ~8 samples per
    median cycle (fewer alias into a moire pattern), kept within [700, 2500]."""
    if requested is not None:
        return int(requested)
    if not d.cycles:
        return min(700, len(d.times))
    period = np.median([c.period for c in d.cycles])
    wanted = 8 * abs(d.times[-1] - d.times[0]) / period
    return int(min(len(d.times), np.clip(wanted, 700, 2500)))


# --------------------------------------------------------------------------- panels


def _colored_path(ax, x, y, t, norm, width):
    points = np.column_stack((x, y))
    if len(points) < 2:
        return
    lc = LineCollection(np.stack((points[:-1], points[1:]), axis=1), cmap=time_cmap(), norm=norm, linewidth=width)
    lc.set_array((t[:-1] + t[1:]) / 2)
    ax.add_collection(lc)
    ax.update_datalim(points)
    ax.autoscale_view()


def _dim_outside_focus(ax, d: MMRDiagnostics):
    """White veil over the part of the record outside the focus; nothing if no focus."""
    if not d.has_focus:
        return
    inside = d.t[d.mask]
    ax.axvspan(d.t[0], inside[0], color='white', alpha=0.65, zorder=6, lw=0)
    ax.axvspan(inside[-1], d.t[-1], color='white', alpha=0.65, zorder=6, lw=0)


def draw_recurrence(ax, d: MMRDiagnostics, options: Dict[str, Any], cax=None):
    """Recurrence image (dark = the state returns close to itself, in its own scale) with a D colour bar.

    The bar goes into `cax` when given (combined figure), otherwise above the panel.
    """
    m = d.mask
    rec = d.recurrence(recurrence_points(d, options['recurrence']['max_points']), options['recurrence']['exclude_samples'])
    cmap = mpl.colormaps['gray'].copy()
    cmap.set_bad('white')
    extent = [rec.times[0] / d.divisor, rec.times[-1] / d.divisor] * 2
    image = ax.imshow(rec.distance, origin='lower', extent=extent, vmin=0, vmax=2, cmap=cmap, interpolation='antialiased', rasterized=True)
    if d.has_focus:
        for bound in (d.t[m][0], d.t[m][-1]):
            ax.axvline(bound, color=INK, lw=0.6, ls='--')
            ax.axhline(bound, color=INK, lw=0.6, ls='--')
    ax.set_aspect('equal')
    ax.set_anchor('W')
    ax.set_xlabel(f'$t_1$ ({d.unit})')
    ax.set_ylabel(f'$t_2$ ({d.unit})')
    if cax is not None:
        bar = ax.figure.colorbar(image, cax=cax, extend='max')
        bar.set_label(r'$D$', labelpad=6)
    else:
        bar = ax.figure.colorbar(image, ax=ax, location='top', fraction=0.05, pad=0.04, extend='max', aspect=30)
        bar.ax.text(1.06, 0.5, r'$D$', transform=bar.ax.transAxes, ha='left', va='center')
    bar.set_ticks([0, 1, 2])
    bar.outline.set_linewidth(0.5)


def draw_fair(ax, d: MMRDiagnostics, style: FigureStyle, norm: Normalize):
    """FAIR plane: mean anomaly against the mean-longitude difference to the planet."""
    m = d.mask
    inner = osc.is_inner(d.axis[m], d.planet.axis[m])
    x, y = osc.fair_coordinates(d.mean_anomaly, d.longitude, d.planet.longitude, inner)
    if d.has_focus:
        ax.scatter(x[~m], y[~m], s=style.marker, color=OUT_OF_FOCUS, linewidths=0, rasterized=True)
    ax.scatter(x[m], y[m], s=style.marker * 1.5, c=d.t[m], cmap=time_cmap(), norm=norm, linewidths=0, rasterized=True)
    ticks = [0, 90, 180, 270, 360]
    ax.set(xlim=(0, 360), ylim=(0, 360), xticks=ticks, yticks=ticks, aspect='equal')
    ax.set_xlabel(r'$M$ (deg)')
    ax.set_ylabel(r'$\lambda_{\mathrm{p}} - \lambda$ (deg)' if inner else r'$\lambda - \lambda_{\mathrm{p}}$ (deg)')


def draw_portrait(ax, d: MMRDiagnostics, style: FigureStyle, options: Dict[str, Any], norm: Normalize):
    """sigma against sigma_dot (or a - a0) over the focus, coloured by time (see `portrait_coordinates`)."""
    y_kind = options['portrait']['y']
    x, y, x_raw, y_raw = portrait_coordinates(d, y_kind)
    if options['raw'] != RawMode.NONE:
        ax.scatter(x_raw, y_raw, s=style.marker * 0.6, color=RAW, alpha=0.5, linewidths=0, rasterized=True, zorder=1)
    _colored_path(ax, x, y, d.t[d.portrait_mask], norm, style.line)
    # Raw rates are far noisier than the filtered path; keep the view on the latter.
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = 0.08 * (hi - lo) or 1.0
    ax.set_ylim(lo - pad, hi + pad)
    if y_kind == PortraitY.RATE:
        ax.set_ylabel(rf'$\dot\sigma$ (deg/{d.unit})')
        ax.axhline(0, **RATE_ZERO)
    else:
        ax.set_ylabel(r'$\Delta a$ ($10^{-3}$ au)')
    ax.set_xlabel(r'$\sigma$ (deg)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # A circulating angle runs to 10^5 deg; a power-of-ten factor keeps the ticks readable.
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), useMathText=True)
    ax.grid(alpha=0.15, lw=0.5)


def draw_cycles(ax_sigma, ax_axis, d: MMRDiagnostics, style: FigureStyle, norm: Normalize):
    """sigma_f over each cycle, and a - median(a) over the same cycle, on one period; a is the analysis' a_f
    (`MMRDiagnostics.axis_analysis`), else the osculating a."""
    if d.cycles:
        mids = np.array([(c.start + c.end) / 2 for c in d.cycles]) / d.divisor
        for ax, series, scale, centred in ((ax_sigma, d.sigma_filtered, 1.0, False), (ax_axis, d.axis_analysis, MILLI, True)):
            folded = [np.column_stack(osc.fold_cycle(c, d.times, series, subtract_median=centred)) * (1.0, scale) for c in d.cycles]
            # One collection for all cycles: thousands of separate lines would dominate drawing.
            lines = LineCollection(folded, cmap=time_cmap(), norm=norm, linewidth=style.line, alpha=0.85)
            lines.set_array(mids)
            ax.add_collection(lines)
            ax.autoscale_view()
    ax_sigma.set_ylabel(r'$\sigma$ (rad)')
    ax_axis.set_ylabel(r'$a - \tilde a_{\mathrm{cycle}}$ ($10^{-3}$ au)')
    for ax in (ax_sigma, ax_axis):
        ax.set_xlim(0, 1)
        ax.set_xlabel('Cycle fraction (max to max)')
        ax.grid(alpha=0.15, lw=0.5)
        if not d.cycles:
            ax.set_yticks([])
            ax.text(0.5, 0.5, 'No complete cycles', transform=ax.transAxes, ha='center', va='center', color='#555555')
    if d.cycles:
        ax_sigma.text(0.99, 0.98, f'$n = {len(d.cycles)}$', transform=ax_sigma.transAxes, ha='right', va='top', fontsize='small')


# ------------------------------------------------------------------------- figures


def _warn(message: str):
    """Tell the user (console, notebook) and the log file."""
    logger.warning(message)
    warnings.warn(message, stacklevel=2)


class MMRPlotter(DiagnosticPlotter):
    """Standalone diagnostic figures of one body in one MMR (recurrence, FAIR, portrait, cycles).

    The combined MMR figure is the verdict figure of the analysis (`plotting/mmr_combined.py`, drawn by
    `DataManager` from `body.mmr_analyses`), not a page of these panels.

    Examples
    --------
        >>> plotter = MMRPlotter.from_body(body, mmr, sim, options={'style': 'paper'})
        >>> plotter.plot('recurrence')  # by kind name, as `plots` in the config
        >>> plotter.save('recurrence.png').close()
    """

    KINDS = tuple(kind for kind in MMR_PLOT_KINDS if kind != 'combined')
    KIND_NAME = 'MMR'

    @classmethod
    def from_body(cls, body, resonance, sim=None, options: Optional[Dict[str, Any]] = None, focus=None) -> 'MMRPlotter':
        options = resolve_plot_options(options)
        return cls(MMRDiagnostics.from_body(body, resonance, sim, focus=focus, prominence=options['cycles']['prominence']), options)

    def title(self) -> str:
        if not (self.d.body_name or self.d.resonance):
            return ''
        body = f'Asteroid {self.d.body_name}' if self.d.body_name else ''
        return ', '.join(part for part in (body, self.d.label or self.d.resonance) if part)

    def _fair_drawable(self) -> bool:
        """Whether FAIR can be drawn, warning when it cannot (three-body, retrograde, no planet data) or when
        it would be aliased."""
        d = self.d
        reason = d.fair_unavailable_reason()
        if reason is not None:
            _warn(f'FAIR plot skipped for {d.body_name} {d.resonance}: {reason}')
            return False
        fraction = osc.fair_step_fraction(d.times, d.axis)
        if fraction > self.options['fair']['max_step_fraction']:
            _warn(
                f'FAIR for {d.body_name} {d.resonance}: the output step is {fraction:.2f} of the orbital period; '
                'the plane is aliased and its strips are not resolved. Save output more often (larger Nout).'
            )
        return True

    # -- figures

    def plot_recurrence(self) -> 'MMRPlotter':
        with self._figure_of(self.style.panel_size * 1.15, self.style.panel_size) as fig:
            draw_recurrence(fig.add_subplot(), self.d, self.options)
        return self

    def plot_fair(self) -> 'MMRPlotter':
        """FAIR plane; warns and draws nothing when it is not available."""
        if not self._fair_drawable():
            return self
        return self._time_panel(1.2, lambda ax, norm: draw_fair(ax, self.d, self.style, norm))

    def plot_portrait(self) -> 'MMRPlotter':
        return self._time_panel(1.25, lambda ax, norm: draw_portrait(ax, self.d, self.style, self.options, norm))

    def plot_cycles(self) -> 'MMRPlotter':
        norm = self.d.time_norm()
        with self._figure_of(self.style.panel_size * 2.1, self.style.panel_size, tight=False) as fig:
            axes = fig.subplots(1, 2)
            draw_cycles(axes[0], axes[1], self.d, self.style, norm)
            time_colorbar(fig, norm, f'Cycle midpoint ({self.d.unit})', ax=list(axes), fraction=0.03, pad=0.02)
        return self
