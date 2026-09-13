"""
Cross-spectrum figure: three panels that are meant to be read together.

    [1] coherence gamma^2 vs period — one vertical line per detected line per
        segmentation level over a faint full spectrum, each level with its own
        significance threshold, the strongest circled
    [2] cross-phase vs period — one marker per detected line, filled when every level
        entitled to vote confirmed it, sized by cross-amplitude

Shading marks where to look: an interval every entitled level agrees on (all faint
curves above their dotted thresholds across it) that also carries one of the strongest
lines by cross-amplitude. Agreement alone is not selective — for e and i it mostly
measures conservation of the Kozai integral, which holds across the whole band.
    [3] the two series band-passed around the decisive line — what actually settles it

Panels 1 and 2 share the period axis and sit flush against each other. Panel 3 has its
own time axis and is set apart.

Everything is cropped to the search band. Bins outside it are not drawn at all: they
cannot be claimed as significant, and the short-period forest only distracts from the
band where the question is decided.

Period and time axes pick their own units (yr / kyr / Myr) so that the same code reads
correctly from a thousand-year synthetic to a 50 Myr integration.
"""

from typing import List, Optional, Tuple

import numpy as np

from resonances.logger import logger
from resonances.resonance.cross_spectrum import bandpass
from .base import BasePlotter, format_period, time_unit, title_prefix

# How each series is labelled and scaled for display. Angles are stored in radians but
# a Lidov-Kozai inclination amplitude is ~0.001 rad, which reads as 0.0010 / 0.0005 /
# 0.0000 on an axis — degrees keep the panel legible.
SERIES_DISPLAY = {
    'sigma': (r'$\delta\sigma$, deg', np.degrees),
    'a': (r'$\delta a$, au', None),
    'e': (r'$\delta e$', None),
    'i': (r'$\delta i$, deg', np.degrees),
}

MAX_CIRCLED_PEAKS = 5  # per level, otherwise the coherence panel fills up with rings
CYCLES_SHOWN = 16  # of the decisive line, in the time-domain panel
# Only the strongest lines get a highlight. Coherence between e and i is broadband
# whenever the Kozai integral is conserved — i is then a smooth function of e, so the
# two agree at every frequency carrying power and "all levels agree" on its own selects
# most of the band. What is worth looking at is where the cross-power actually is.
HIGHLIGHTED_LINES = 3

_LINE_COLOR = '#1f6fb4'
_SECOND_COLOR = '#c1272d'
_CONSENSUS_COLOR = '#f7e3ad'
_TARGET_COLOR = '#cfe6d2'


class CrossSpectrumPlotter(BasePlotter):
    """
    Plotter for one pair's cross spectrum.

    Examples
    --------
        >>> plotter = CrossSpectrumPlotter.from_body(body, resonance, 'e-i')
        >>> plotter.plot().save('coherence-e-i.png')
        >>> plotter.close()
    """

    def __init__(self):
        self._analysis = None
        self._times: Optional[np.ndarray] = None
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._metadata: Optional[dict] = None
        self._figure = None
        self._axes = None

    @classmethod
    def from_analysis(
        cls,
        analysis,
        times: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        body_name: str = '',
        resonance_key: str = '',
    ) -> 'CrossSpectrumPlotter':
        """Create a plotter from a `coherence_analysis.PairAnalysis`.

        `times` (in years) and the two series are only needed for the time-domain panel;
        without them the figure is drawn with the two spectral panels alone.
        """
        plotter = cls()
        plotter._analysis = analysis
        plotter._times, plotter._x, plotter._y = times, x, y
        plotter._metadata = {'body_name': body_name, 'resonance_key': resonance_key}
        return plotter

    @classmethod
    def from_body(cls, body, resonance, pair: str) -> 'CrossSpectrumPlotter':
        """Create a plotter for one already-analysed pair of a body."""
        from resonances.resonance.coherence_analysis import resolve_series
        from resonances.resonance.cross_spectrum import parse_pair

        analysis = body.coherence.get(resonance.to_s(), {}).get(pair)
        x_key, y_key = parse_pair(pair)
        return cls.from_analysis(
            analysis,
            times=body.times / (2 * np.pi),
            x=resolve_series(body, resonance, x_key),
            y=resolve_series(body, resonance, y_key),
            body_name=body.name,
            resonance_key=resonance.to_s(),
        )

    def plot(self, figsize: tuple = (7.5, 7.0), dpi: int = 150) -> 'CrossSpectrumPlotter':
        """Draw the three panels."""
        import matplotlib.pyplot as plt

        if self._figure is not None:
            plt.close(self._figure)

        self._figure = plt.figure(figsize=figsize, dpi=dpi)
        levels = [level for level in self._analysis.levels if level.valid] if self._analysis else []
        decisive = self._analysis.decisive if self._analysis else None

        if decisive is None:
            self._axes = [self._figure.add_subplot(1, 1, 1)]
            self._render_unavailable()
            return self

        # Panels 1 and 2 share the period axis, so they sit flush; panel 3 measures time
        # and carries its own axis, hence the nested grid with a wide gap above it.
        outer = self._figure.add_gridspec(2, 1, height_ratios=[2.75, 1.3], hspace=0.42)
        spectral = outer[0].subgridspec(2, 1, height_ratios=[1.75, 1.0], hspace=0.08)
        ax_coherence = self._figure.add_subplot(spectral[0])
        ax_phase = self._figure.add_subplot(spectral[1], sharex=ax_coherence)
        ax_time = self._figure.add_subplot(outer[1])
        self._figure.subplots_adjust(left=0.12, right=0.87, top=0.92, bottom=0.09)
        self._axes = [ax_coherence, ax_phase, ax_time]

        lo, hi = decisive.period_min, decisive.period_max
        divisor, unit = time_unit(hi)
        agreed = self._agreement_runs(levels, decisive)
        points = self._collect_points(levels, lo, hi, agreed)

        self._setup_period_axes(ax_coherence, ax_phase, lo, hi, divisor, unit)
        self._plot_coherence(ax_coherence, levels, lo, hi, divisor)
        self._shade([ax_coherence, ax_phase], self._highlighted(agreed, decisive), divisor)
        self._plot_phase(ax_phase, points, divisor)
        self._mark_decisive([ax_coherence, ax_phase], divisor)
        self._plot_bandpassed(ax_time)

        ax_coherence.set_title(self._title(), fontsize=11, pad=8)
        return self

    def _level_colors(self, levels) -> list:
        import matplotlib.pyplot as plt

        return plt.cm.Blues(np.linspace(0.45, 0.95, max(len(levels), 1)))

    def _collect_points(self, levels, lo: float, hi: float, agreed: List[Tuple[float, float]]) -> List[dict]:
        """Every detected line inside the band, with its consensus verdict."""
        colors = self._level_colors(levels)
        points = []
        for level, color in zip(levels, colors):
            for line in level.lines:
                if not (lo <= line.period <= hi):
                    continue
                points.append(
                    dict(
                        period=line.period,
                        phase=line.phase_deg,
                        gamma2=line.gamma2,
                        amplitude=line.cross_amplitude,
                        color=color,
                        unanimous=_within(line.period, agreed),
                    )
                )
        return points

    @staticmethod
    def _agreement_runs(levels, decisive) -> List[Tuple[float, float]]:
        """Period intervals where every level entitled to vote is above its own threshold.

        Read straight off the spectra rather than off the line list, so that the result is
        exactly what the faint background curves show: an interval is highlighted when all
        of them sit above their dotted threshold across it. A level whose validity band
        does not reach that period abstains rather than objects — it cannot see the line,
        and must not block a consensus.

        Each level votes on the decisive level's frequency grid, and `np.interp` spreads
        its verdict over its own bin width. That width *is* the matching tolerance: the
        finely segmented levels resolve a line into fewer, wider bins, and demanding an
        exact frequency match between grids would reject agreements that are real.
        """
        if decisive.frequency is None or decisive.in_band is None:
            return []
        grid = decisive.frequency[decisive.in_band]
        if grid.size < 2:
            return []

        period = 1.0 / grid
        agree = np.ones(grid.size, dtype=bool)
        entitled = np.zeros(grid.size, dtype=int)
        for level in levels:
            if level.frequency is None or level.coherence is None:
                continue
            significant = (level.in_band & (level.coherence >= level.gamma2_crit)).astype(float)
            on_grid = np.interp(grid, level.frequency, significant, left=0.0, right=0.0) > 0
            votes = (period >= level.period_min) & (period <= level.period_max)
            entitled += votes
            agree &= ~votes | on_grid

        # A single level agreeing with itself is not a consensus.
        indices = np.where(agree & (entitled >= 2))[0]
        if len(indices) == 0:
            return []
        half_bin = 0.5 * float(grid[1] - grid[0])
        runs = []
        for group in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1):
            runs.append((1.0 / (grid[group[-1]] + half_bin), 1.0 / (grid[group[0]] - half_bin)))
        return runs

    @staticmethod
    def _setup_period_axes(ax_coherence, ax_phase, lo: float, hi: float, divisor: float, unit: str):
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        for ax in (ax_coherence, ax_phase):
            ax.set_xscale('log')
            ax.set_xlim(lo / divisor * 0.92, hi / divisor * 1.08)
            ax.grid(alpha=0.22, lw=0.5)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            # ScalarFormatter renders sub-1 ticks as "0"; %g keeps 0.2 readable as 0.2.
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax_coherence.tick_params(labelbottom=False)
        ax_phase.set_xlabel(f'period, {unit}')

    def _plot_coherence(self, ax, levels, lo: float, hi: float, divisor: float):
        """One vertical line per detected line, per segmentation level.

        Stems rather than a continuous spectrum: what the analysis actually produces is a
        list of lines, each represented by the bin carrying the most cross-power within
        its run of significant bins. Drawing the full curve invites reading a circled
        line as the maximum of that curve, which it is not — the coherence maximum and
        the cross-power maximum of a line generally sit in different bins.
        """
        for level, color in zip(levels, self._level_colors(levels)):
            # Faint full spectrum behind the stems: the sub-threshold structure is what
            # tells a lone significant bin from the tip of a broad coherent feature.
            keep = (level.frequency > 0) & level.in_band
            ax.plot(1.0 / level.frequency[keep] / divisor, level.coherence[keep], color=color, lw=0.6, alpha=0.28, zorder=1)
            ax.axhline(level.gamma2_crit, color=color, ls=':', lw=0.8, zorder=2)
            in_band = [line for line in level.lines if lo <= line.period <= hi]
            if not in_band:
                # Keep the level in the legend even when it found nothing.
                ax.plot([], [], color=color, lw=1.4, label=rf'$n={level.n_seg}$')
                continue
            periods = [line.period / divisor for line in in_band]
            gammas = [line.gamma2 for line in in_band]
            ax.vlines(periods, 0, gammas, color=color, lw=1.4, zorder=4, label=rf'$n={level.n_seg}$')
            ax.plot(periods, gammas, '.', color=color, ms=4, zorder=5)
            for line in in_band[:MAX_CIRCLED_PEAKS]:
                ax.plot(line.period / divisor, line.gamma2, 'o', mfc='none', mec=_SECOND_COLOR, mew=1.1, ms=8, zorder=6)

        ax.set_ylabel(r'coherence $\gamma^2$')
        # Headroom above 1 so the legend never sits on top of a stem.
        ax.set_ylim(0, 1.32)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if levels:
            ax.legend(frameon=False, fontsize=9, loc='upper left', ncol=len(levels), handlelength=1.1, columnspacing=1.0)

    def _mark_decisive(self, axes, divisor: float):
        """Show which line the verdict came from.

        Worth marking explicitly because it need not fall in a consensus band: the
        decisive line is the one with the most cross-power at the coarsest valid level,
        while the shading marks periods where every level entitled to vote agrees. The
        two answer different questions and routinely point at different periods.
        """
        line = self._analysis.top_line if self._analysis else None
        if line is None:
            return
        for ax in axes:
            ax.axvline(line.period / divisor, color=_SECOND_COLOR, ls='--', lw=1.0, alpha=0.8, zorder=3)

    @staticmethod
    def _highlighted(agreed: List[Tuple[float, float]], decisive) -> List[Tuple[float, float]]:
        """Of the agreed intervals, the ones carrying a strongest line — where to look.

        Agreement alone is not selective enough to be worth a highlight: `decisive.lines`
        is ranked by integrated cross-amplitude, so requiring one of the top few inside
        the interval keeps the bands that carry the power and drops the rest.
        """
        strongest = [line.period for line in decisive.lines[:HIGHLIGHTED_LINES]]
        return [interval for interval in agreed if any(_within(period, [interval]) for period in strongest)]

    @staticmethod
    def _shade(axes, intervals: List[Tuple[float, float]], divisor: float):
        for low, high in intervals:
            for ax in axes:
                ax.axvspan(low / divisor, high / divisor, color=_CONSENSUS_COLOR, alpha=0.6, lw=0, zorder=0)

    def _plot_phase(self, ax, points: List[dict], divisor: float):
        target = self._analysis.phase_target
        if target is not None:
            centre, tolerance = target
            for sign in (1, -1):
                ax.axhspan(sign * centre - tolerance, sign * centre + tolerance, color=_TARGET_COLOR, lw=0, zorder=1)

        largest = max((p['amplitude'] for p in points), default=1.0) or 1.0
        for point in points:
            size = 14 + 300 * (point['amplitude'] / largest) ** 0.45
            style = (
                dict(color=point['color'], lw=0.6, edgecolor='w')
                if point['unanimous']
                else dict(facecolor='none', edgecolor=point['color'], lw=1.1)
            )
            ax.scatter(point['period'] / divisor, point['phase'], s=size, zorder=5 if point['unanimous'] else 4, **style)

        ax.set_ylabel('cross-phase, deg')
        ax.set_ylim(-190, 190)
        ax.set_yticks([-180, -90, 0, 90, 180])

    def _plot_bandpassed(self, ax):
        """The two series filtered around the decisive line, over a few of its cycles."""
        line = self._analysis.top_line
        if line is None or self._times is None or self._x is None or self._y is None:
            ax.text(0.5, 0.5, 'no line to band-pass', ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        dt = float(np.median(np.diff(self._times)))
        band_x, band_y = bandpass(self._x, dt, line.frequency), bandpass(self._y, dt, line.frequency)
        window = slice(*_centred_window(len(self._times), int(CYCLES_SHOWN * line.period / dt)))

        times = self._times[window]
        divisor, unit = time_unit(times[-1] - times[0])
        x_label, x_convert = self._display('x')
        y_label, y_convert = self._display('y')
        twin = ax.twinx()

        ax.plot(times / divisor, band_x[window] if x_convert is None else x_convert(band_x[window]), color=_LINE_COLOR, lw=1.15)
        twin.plot(times / divisor, band_y[window] if y_convert is None else y_convert(band_y[window]), color=_SECOND_COLOR, lw=1.15)

        ax.set_ylabel(x_label, color=_LINE_COLOR, fontsize=9)
        twin.set_ylabel(y_label, color=_SECOND_COLOR, fontsize=9)
        ax.tick_params(axis='y', colors=_LINE_COLOR)
        twin.tick_params(axis='y', colors=_SECOND_COLOR)
        ax.set_xlabel(f'time, {unit}')
        ax.grid(alpha=0.22, lw=0.5)
        # Placed as text rather than a title: a title would collide with the period label
        # of the panel above.
        ax.text(
            0.5,
            1.10,
            f'band-passed at {format_period(line.period)}',
            transform=ax.transAxes,
            ha='center',
            va='bottom',
            fontsize=9.5,
        )

    def _display(self, which: str) -> Tuple[str, Optional[callable]]:
        """Axis label and unit conversion for the first or second series of the pair."""
        keys = self._analysis.pair.split('-') if self._analysis else ['', '']
        key = keys[0] if which == 'x' else keys[-1]
        return SERIES_DISPLAY.get(key, (key, None))

    def _render_unavailable(self):
        reason = 'no data'
        if self._analysis is not None:
            invalid = [level for level in self._analysis.levels if not level.valid]
            if invalid:
                reason = invalid[0].reason
        logger.warning(f"Cross spectrum not available for plotting: {reason}")
        ax = self._axes[0]
        ax.text(0.5, 0.5, f'Cross spectrum not available\n{reason}', ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(self._title(), fontsize=11, pad=8)

    def _title(self) -> str:
        prefix = title_prefix(self._metadata)
        pair = self._analysis.pair if self._analysis else ''
        suffix = f'cross spectrum {pair}'
        return f'{prefix} - {suffix}' if prefix else suffix

    def close(self):
        """Close the figure to free memory."""
        super().close()
        self._axes = None


def _within(period: float, intervals: List[Tuple[float, float]]) -> bool:
    """Whether a period falls inside any of the given [low, high] period intervals."""
    return any(low <= period <= high for low, high in intervals)


def _centred_window(total: int, length: int) -> Tuple[int, int]:
    """A window of `length` samples near the middle of the series, clamped to it."""
    length = max(2, min(total, length))
    start = max(0, min(total - length, total // 2 - length // 2))
    return start, start + length
