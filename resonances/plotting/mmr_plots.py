"""Diagnostic plots of a mean-motion resonance.

`MMRDiagnostics` gathers the arrays of one body-resonance pair and derives the
oscillation quantities once (see `resonances.resonance.oscillations`). Every panel is
a `draw_*` function drawing onto given axes, so the standalone figures and the combined
one share a single implementation:

    series      unwrapped sigma, wrapped sigma and semi-major axis against time
    recurrence  distance between states (sigma, sigma_dot) at two times
    fair        mean anomaly against the mean-longitude difference (two-body only)
    portrait    sigma against sigma_dot (or a), coloured by time
    cycles      the sigma cycles, and a over the same cycles, stacked on one period

Times are in years, angles in radians internally; the portrait shows degrees. Nothing
here changes a status.
"""

import string
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from resonances.logger import logger
from resonances.resonance import oscillations as osc
from resonances.resonance.classify.classify import calc_sigma_derivative
from .base import BasePlotter, time_unit
from .style import (
    CENTRE,
    ENVELOPE,
    INK,
    MMR_PLOT_KINDS,
    OUT_OF_FOCUS,
    RAW,
    STYLES,
    FigureStyle,
    PortraitY,
    RawMode,
    SeriesMode,
    resolve_plot_options,
    time_cmap,
    time_label,
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
    axis: np.ndarray
    axis_filtered: np.ndarray
    body_name: str = ''
    resonance: str = ''  # full key, e.g. 2J-1+0-1
    label: str = ''  # short notation for titles, e.g. 2J-1
    n_planets: int = 1
    mean_anomaly: Optional[np.ndarray] = None
    longitude: Optional[np.ndarray] = None
    planet: Optional[PlanetSeries] = None
    focus: Optional[Tuple[float, float]] = None
    prominence: Optional[float] = None
    edge: float = 0.0  # years dropped at both ends of the portrait: the low-pass filter's edge transient

    mask: np.ndarray = field(init=False)
    rate: np.ndarray = field(init=False)  # d sigma_filtered / dt, rad/yr
    raw_rate: np.ndarray = field(init=False)
    cycles: List[osc.Cycle] = field(init=False)
    axis_reference: float = field(init=False)  # a0: median filtered a over the focus
    divisor: float = field(init=False)  # years per display time unit
    unit: str = field(init=False)  # 'yr', 'kyr' or 'Myr'
    _recurrences: Dict[Tuple[int, int], osc.Recurrence] = field(init=False, default_factory=dict, repr=False)

    _SERIES = ('times', 'sigma', 'sigma_raw', 'sigma_filtered', 'axis', 'axis_filtered', 'mean_anomaly', 'longitude')

    def __post_init__(self):
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
        self.axis_reference = float(np.median(self.axis_filtered[self.mask]))
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
        if self.mean_anomaly is None or self.longitude is None:
            return 'no mean anomaly / mean longitude of the body'
        if self.planet is None:
            return 'no planet data (run the simulation with save_planets=True)'
        if len(self.planet.longitude) != len(self.times):
            return 'planet data do not match the body output times'
        return None

    @classmethod
    def from_body(cls, body, resonance, sim=None, focus=None, prominence=None) -> 'MMRDiagnostics':
        """From a simulated body; the portrait edge is one period of the filter cutoff."""
        key = resonance.to_s()
        times = sim.times if sim is not None else body.times
        names = list(resonance.planets_names)
        planet = _planet_series(sim, names[0]) if len(names) == 1 and sim is not None else None
        return cls(
            times=np.asarray(times) / (2 * np.pi),
            sigma=body.angles[key],
            sigma_raw=body.angles_unwrapped[key],
            sigma_filtered=body.angles_filtered_unwrapped[key],
            axis=body.axis,
            axis_filtered=body.axis_filtered if body.axis_filtered is not None else body.axis,
            body_name=str(body.name),
            resonance=key,
            label=resonance.to_short(),
            n_planets=max(len(names), 1),
            mean_anomaly=body.M,
            longitude=body.longitude,
            planet=planet,
            focus=focus,
            prominence=prominence,
            edge=1.0 / sim.config.oscillations_cutoff if sim is not None else 0.0,
        )


def _planet_series(sim, name: str) -> Optional[PlanetSeries]:
    """Mean longitude and semi-major axis of a planet recorded by the simulation."""
    data = sim.integration_engine.planets_data.get(name)
    if not data or 'l' not in data or 'a' not in data:
        return None
    return PlanetSeries(name=name, longitude=np.asarray(data['l'], dtype=float), axis=np.asarray(data['a'], dtype=float))


# ------------------------------------------------------------------ pure geometry


def portrait_coordinates(d: MMRDiagnostics, y: PortraitY = PortraitY.RATE):
    """(x, y, x_raw, y_raw) of the portrait over the focus: sigma in degrees against
    sigma_dot in deg per display time unit, or against a - a0 in 10^-3 au.

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
    return x, d.milli_au(d.axis_filtered[m]), x_raw, d.milli_au(d.axis[m])


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


def centre_line(cycles: List[osc.Cycle], *series) -> Tuple[np.ndarray, ...]:
    """Per-cycle `series` with NaN where the line must not join two cycles.

    That is a gap between them, or a cycle more than three times the median period (two
    maxima bridging a stretch of circulation). Such a cycle's centre stays as a lone
    point; only the straight line across it is not drawn. The cycles are unchanged.
    """
    median = np.median([c.period for c in cycles])
    outlier = [c.period > 3 * median for c in cycles]
    breaks = [
        i + 1
        for i in range(len(cycles) - 1)
        if cycles[i + 1].start - cycles[i].end > 0.5 * cycles[i].period or outlier[i] or outlier[i + 1]
    ]
    return tuple(np.insert(np.asarray(values, dtype=float), breaks, np.nan) for values in series)


def lone_points(line: np.ndarray) -> np.ndarray:
    """Finite entries of a NaN-broken line with no finite neighbour on either side."""
    finite = np.pad(np.isfinite(line), 1)
    return finite[1:-1] & ~(finite[:-2] | finite[2:])


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


def _raw_line(ax, x, y, style: FigureStyle, raw: RawMode):
    if raw == RawMode.NONE:
        return
    ax.plot(x, y, color=RAW, lw=style.raw_line, zorder=2)
    if raw == RawMode.MARKERS:
        ax.plot(x, y, ls='none', marker='o', markevery=max(len(x) // 100, 1), ms=2.5, mfc='none', mec=RAW, mew=0.6, zorder=2)


def _envelope(ax, t, y, bins: int, **fill):
    """Min-max band of `y` in `bins` equal time bins: every sample is inside, at the display resolution."""
    edges = np.unique(np.linspace(0, len(t), bins + 1).astype(int))[:-1]
    lo, hi = np.minimum.reduceat(y, edges), np.maximum.reduceat(y, edges)
    centre = np.add.reduceat(t, edges) / np.diff(np.append(edges, len(t)))
    ax.fill_between(centre, lo, hi, linewidth=0, **fill)


def envelope_bins(d: MMRDiagnostics, mode: SeriesMode, pixels: float, line_pixels: float) -> int:
    """Number of min-max bins for the time series, or 0 to draw lines.

    `envelope` always bins, `lines` never; `auto` bins when a median cycle is narrower
    than four line widths, where single oscillations merge into a solid block. A bin spans
    about two cycles, so the band traces the amplitude (and its modulation), not the
    oscillation itself.
    """
    if mode == SeriesMode.LINES or not d.cycles:
        return 0
    cycles = abs(d.times[-1] - d.times[0]) / np.median([c.period for c in d.cycles])
    if mode == SeriesMode.AUTO and pixels / cycles >= 4 * line_pixels:
        return 0
    return int(round(max(2, min(pixels, cycles / 2))))


def draw_series(axes, d: MMRDiagnostics, style: FigureStyle, options: Dict[str, Any]):
    """Unwrapped sigma, wrapped sigma and a - a0 on three axes sharing the time axis.

    Raw series in grey, filtered in black. When the oscillations are too dense for the
    panel width (see `envelope_bins`), both are drawn as min-max bands instead and
    the wrapped points become translucent, so their density shows.
    """
    ax_u, ax_w, ax_a = axes
    t, raw = d.t, options['raw']
    pixels = ax_u.get_position().width * ax_u.figure.get_figwidth() * style.dpi
    bins = envelope_bins(d, options['series'], pixels, style.line / 72 * style.dpi)
    envelope = bins > 0

    for ax, raw_values, filtered in ((ax_u, d.sigma_raw, d.sigma_filtered), (ax_a, d.milli_au(d.axis), d.milli_au(d.axis_filtered))):
        if envelope:
            if raw != RawMode.NONE:
                _envelope(ax, t, raw_values, bins, color=RAW, zorder=2)
            _envelope(ax, t, filtered, bins, color=ENVELOPE, zorder=3)
        else:
            _raw_line(ax, t, raw_values, style, raw)
            ax.plot(t, filtered, color=INK, lw=style.line, zorder=3)
    ax_u.set_ylabel(r'$\sigma$ (rad)')
    ax_a.set_ylabel('$\\Delta a$\n($10^{-3}$ au)')

    ax_w.scatter(t, d.sigma, s=style.series_marker, color=INK, alpha=0.25 if envelope else 1.0, linewidths=0, rasterized=True)
    ax_w.set_ylim(0, TWO_PI)
    ax_w.set_yticks([0, np.pi, TWO_PI], ['0', r'$\pi$', r'$2\pi$'])
    # The extreme labels sit on the panel edges; keep them inside, clear of the neighbours.
    ax_w.get_yticklabels()[0].set_va('bottom')
    ax_w.get_yticklabels()[-1].set_va('top')
    ax_w.set_ylabel(r'$\sigma$ (rad)')

    if d.cycles:
        mid, centre, axis_mid = centre_line(d.cycles, *osc.cycle_centres(d.cycles, d.times, d.axis_filtered))
        # A centre cut off from both neighbours would be invisible as a line: mark it as a point.
        lone = lone_points(mid)
        for ax, values in ((ax_u, centre), (ax_a, d.milli_au(axis_mid))):
            ax.plot(mid / d.divisor, values, color=CENTRE, lw=style.line, zorder=4)
            ax.plot(mid[lone] / d.divisor, values[lone], ls='none', marker='o', ms=2.5 * style.line, color=CENTRE, zorder=4)

    for ax in axes:
        _dim_outside_focus(ax, d)
        ax.set_xlim(t[0], t[-1])
        ax.grid(alpha=0.15, lw=0.5)
    # Stacked without gaps: drop the ticks that would touch the neighbour's.
    ax_u.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='lower'))
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='upper'))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    ax_a.set_xlabel(f'Time ({d.unit})')
    # One x for the three y labels, whatever the tick label widths (a fixed distance in inches).
    width = ax_a.get_position().width * ax_a.figure.get_figwidth()
    for ax in axes:
        ax.yaxis.set_label_coords(-0.06 * style.font_size / width, 0.5)


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
        ax.axhline(0, color=RAW, lw=0.5, zorder=1)
    else:
        ax.set_ylabel(r'$\Delta a$ ($10^{-3}$ au)')
    ax.set_xlabel(r'$\sigma$ (deg)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    # A circulating angle runs to 10^5 deg; a power-of-ten factor keeps the ticks readable.
    ax.ticklabel_format(style='sci', scilimits=(-3, 4), useMathText=True)
    ax.grid(alpha=0.15, lw=0.5)


def draw_cycles(ax_sigma, ax_axis, d: MMRDiagnostics, style: FigureStyle, norm: Normalize):
    """sigma over each cycle, and a - median(a) over the same cycle, on one period."""
    if d.cycles:
        mids = np.array([(c.start + c.end) / 2 for c in d.cycles]) / d.divisor
        for ax, series, scale, centred in ((ax_sigma, d.sigma_filtered, 1.0, False), (ax_axis, d.axis_filtered, MILLI, True)):
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


def _letter(ax, letter: str, inside: bool = False):
    """Panel letter above the top-left corner, or inside it for tightly stacked panels."""
    if inside:
        box = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1}
        ax.text(0.006, 0.95, f'({letter})', transform=ax.transAxes, ha='left', va='top', fontweight='bold', zorder=8, bbox=box)
    else:
        ax.text(-0.02, 1.02, f'({letter})', transform=ax.transAxes, ha='right', va='bottom', fontweight='bold')


def _time_colorbar(fig, norm: Normalize, label: str, **where):
    """Colour bar of the time scale; `where` is `ax=...` or `cax=...` plus layout keywords."""
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=time_cmap()), **where)
    bar.set_label(label)
    bar.outline.set_linewidth(0.5)
    return bar


def combined_rects(style: FigureStyle, fair: bool) -> Tuple[Dict[str, Tuple[float, float, float, float]], float]:
    """Panel rectangles (x, y, width, height) of the combined figure, in inches, and its height.

    Top row: three stacked time series spanning exactly the height of the square
    recurrence (and FAIR) panels. Bottom row: three equal squares and the time colour bar.
    """
    lay = style.layout
    bar_block = lay.bar_pad + lay.bar + lay.bar_room
    side = (style.width - lay.left - lay.right - 2 * lay.gap - bar_block) / 3
    height = lay.bottom + side + lay.row_gap + lay.top + lay.title

    rects = {}
    x = lay.left
    for name in ('portrait', 'cycles_sigma', 'cycles_axis'):
        rects[name] = (x, lay.bottom, side, side)
        x += side + lay.gap
    rects['time_bar'] = (x - lay.gap + lay.bar_pad, lay.bottom, lay.bar, side)

    y = lay.bottom + side + lay.row_gap
    squares = 2 if fair else 1
    series_width = style.width - lay.left - lay.right - squares * (lay.gap + lay.top) - bar_block
    panel = (lay.top - 2 * lay.stack_gap) / 3
    for i, name in enumerate(('series_u', 'series_w', 'series_a')):
        rects[name] = (lay.left, y + (2 - i) * (panel + lay.stack_gap), series_width, panel)
    x = lay.left + series_width + lay.gap
    rects['recurrence'] = (x, y, lay.top, lay.top)
    rects['recurrence_bar'] = (x + lay.top + lay.bar_pad, y, lay.bar, lay.top)
    if fair:
        rects['fair'] = (x + lay.top + bar_block + lay.gap, y, lay.top, lay.top)
    return rects, height


def _warn(message: str):
    """Tell the user (console, notebook) and the log file."""
    logger.warning(message)
    warnings.warn(message, stacklevel=2)


class MMRPlotter(BasePlotter):
    """Standalone and combined diagnostic figures of one body in one MMR.

    Examples
    --------
        >>> plotter = MMRPlotter.from_body(body, mmr, sim, options={'style': 'paper'})
        >>> plotter.plot_combined().save('combined.pdf')
        >>> plotter.plot('recurrence')  # by kind name, as `plots` in the config
        >>> plotter.save('recurrence.png').close()
    """

    def __init__(self, diagnostics: MMRDiagnostics, options: Optional[Dict[str, Any]] = None):
        self.d = diagnostics
        self.options = resolve_plot_options(options)
        self.style = STYLES[self.options['style']]
        self._figure = None

    @classmethod
    def from_body(cls, body, resonance, sim=None, options: Optional[Dict[str, Any]] = None, focus=None) -> 'MMRPlotter':
        options = resolve_plot_options(options)
        return cls(MMRDiagnostics.from_body(body, resonance, sim, focus=focus, prominence=options['cycles']['prominence']), options)

    def plot(self, kind: str) -> bool:
        """Draw one plot kind; False when there is nothing to draw (FAIR unavailable)."""
        if kind not in MMR_PLOT_KINDS:
            raise ValueError(f"Unknown MMR plot kind '{kind}'. Known: {', '.join(MMR_PLOT_KINDS)}")
        if self._figure is not None:
            self.close()
        getattr(self, f'plot_{kind}')()
        return self._figure is not None

    def save(self, path, **kwargs):
        kwargs.setdefault('dpi', self.style.dpi)
        with mpl.rc_context(self.style.rc()):
            return super().save(path, **kwargs)

    # -- helpers

    @contextmanager
    def _figure_of(self, width: float, height: float, tight: bool = True):
        """New figure in the style's rc context; titled (and tightened) on exit."""
        if self._figure is not None:
            plt.close(self._figure)
        with mpl.rc_context(self.style.rc()):
            self._figure = plt.figure(figsize=(width, height), dpi=self.style.dpi)
            yield self._figure
            if self.style.title and (self.d.body_name or self.d.resonance):
                body = f'Asteroid {self.d.body_name}' if self.d.body_name else ''
                title = ', '.join(part for part in (body, self.d.label or self.d.resonance) if part)
                self._figure.suptitle(title, y=0.985, fontsize=self.style.font_size + 3)
            if tight:
                self._figure.tight_layout()

    def _fair_drawable(self, warn_three_body: bool) -> bool:
        """Whether FAIR can be drawn, warning when it cannot or when it would be aliased.

        A three-body argument is outside FAIR by construction: the combined figure omits
        the panel silently, a standalone FAIR request is told why.
        """
        d = self.d
        reason = d.fair_unavailable_reason()
        if reason is not None:
            if warn_three_body or d.n_planets == 1:
                _warn(f'FAIR plot skipped for {d.body_name} {d.resonance}: {reason}')
            return False
        fraction = osc.fair_step_fraction(d.times, d.axis)
        if fraction > self.options['fair']['max_step_fraction']:
            _warn(
                f'FAIR for {d.body_name} {d.resonance}: the output step is {fraction:.2f} of the orbital period; '
                'the plane is aliased and its strips are not resolved. Save output more often (larger Nout).'
            )
        return True

    def _time_panel(self, aspect: float, draw) -> 'MMRPlotter':
        """One panel coloured by time, with its time colour bar; `draw(ax, norm)` fills it."""
        norm = self.d.time_norm()
        with self._figure_of(self.style.panel_size * aspect, self.style.panel_size) as fig:
            ax = fig.add_subplot()
            draw(ax, norm)
            _time_colorbar(fig, norm, time_label(self.d.unit), ax=ax, fraction=0.046, pad=0.03)
        return self

    # -- figures

    def plot_combined(self) -> 'MMRPlotter':
        """All panels on one figure. FAIR is included only for a two-body argument with planet data."""
        d, s, o = self.d, self.style, self.options
        fair = self._fair_drawable(warn_three_body=False)
        norm = d.time_norm()
        rects, height = combined_rects(s, fair)
        with self._figure_of(s.width, height, tight=False) as fig:

            def axes(name):
                x, y, w, h = rects[name]
                return fig.add_axes((x / s.width, y / height, w / s.width, h / height))

            series = [axes('series_u'), axes('series_w'), axes('series_a')]
            for ax in series[1:]:
                ax.sharex(series[0])
            draw_series(series, d, s, o)
            upper = [axes('recurrence')]
            draw_recurrence(upper[0], d, o, cax=axes('recurrence_bar'))
            if fair:
                upper.append(axes('fair'))
                draw_fair(upper[1], d, s, norm)
            lower = [axes('portrait'), axes('cycles_sigma'), axes('cycles_axis')]
            draw_portrait(lower[0], d, s, o, norm)
            draw_cycles(lower[1], lower[2], d, s, norm)
            _time_colorbar(fig, norm, f'{time_label(d.unit)}; cycle midpoint for stacks', cax=axes('time_bar'))

            letters = iter(string.ascii_lowercase)
            for ax in series:
                _letter(ax, next(letters), inside=True)
            for ax in upper + lower:
                _letter(ax, next(letters))
        return self

    def plot_recurrence(self) -> 'MMRPlotter':
        with self._figure_of(self.style.panel_size * 1.15, self.style.panel_size) as fig:
            draw_recurrence(fig.add_subplot(), self.d, self.options)
        return self

    def plot_fair(self) -> 'MMRPlotter':
        """FAIR plane; warns and draws nothing when it is not available."""
        if not self._fair_drawable(warn_three_body=True):
            return self
        return self._time_panel(1.2, lambda ax, norm: draw_fair(ax, self.d, self.style, norm))

    def plot_portrait(self) -> 'MMRPlotter':
        return self._time_panel(1.25, lambda ax, norm: draw_portrait(ax, self.d, self.style, self.options, norm))

    def plot_cycles(self) -> 'MMRPlotter':
        norm = self.d.time_norm()
        with self._figure_of(self.style.panel_size * 2.1, self.style.panel_size, tight=False) as fig:
            axes = fig.subplots(1, 2)
            draw_cycles(axes[0], axes[1], self.d, self.style, norm)
            _time_colorbar(fig, norm, f'Cycle midpoint ({self.d.unit})', ax=list(axes), fraction=0.03, pad=0.02)
        return self
