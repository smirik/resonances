"""Figures of a secular or Lidov-Kozai angle — what the phase portrait and the combined figure are for an MMR.

    angle    — sigma(t), degrees: the osculating angle in grey, the analysed angle filtered at the scale the verdict
               rests on in black (`analysis.secular`: P_c, or the scale of the centre), its turning points as red
               circles;
    rate     — the rate of that filtered angle, arcsec/yr, with zero and the mean rate (the divisor);
    portrait — (r cos sigma, r sin sigma), r = e when sigma holds varpi, else i from the invariable plane in
               degrees: osculating in grey, classified coloured by time;
    combined — the one-page V1 figure of the verdict (`draw_long_period_combined`, below): time panels, portraits
               and every number of the decision in the shared right column (`combined_numbers`).

Nothing here changes a status. Reading guide: docs/plots.md, "Secular diagnostics".
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional

import numpy as np
from matplotlib.colors import Normalize

from resonances.data.util import time_ordered
from resonances.secular.free_elements import RAD_PER_ARCSEC, rate
from resonances.secular.proper_angle import inclination_from_invariable_plane, inclination_from_vector, reduced_vectors
from .base import DiagnosticPlotter, time_unit
from .style import CENTRE, EPISODE_COLOURS, INK, MODEL, RATE_ZERO, RAW, RAW_ALPHA, REFERENCE
from .style import SECULAR_PLOT_KINDS, NumbersMode, resolve_plot_options, time_cmap, time_label


@dataclass
class SecularDiagnostics:
    """What the secular figures show for one body and one angle (times in years, in time order)."""

    times: np.ndarray
    sigma: np.ndarray  # the classified angle, unwrapped, rad
    osculating: np.ndarray  # the angle from the osculating elements, unwrapped, rad
    radius: np.ndarray  # e, or i (deg) from the invariable plane, of the classified angle's vectors
    osculating_radius: np.ndarray
    p: int  # coefficient of varpi: the radius is e when it is non-zero
    analysis: Any = None  # the LongPeriodAnalysis with its arrays (`Simulation.long_period_analysis`)
    body_name: str = ''
    resonance: str = ''
    status: Optional[str] = None  # the verdict status string
    free_elements: Any = None  # a ZLK row: the body's FreeElements (the combined figure draws the free omega)
    extra_row: dict = field(default_factory=dict)  # columns of the body for the combined figure (free elements, e-i)
    divisor: float = field(init=False)  # years per displayed time unit
    unit: str = field(init=False)
    curve: np.ndarray = field(init=False)  # deg: the analysed angle filtered at the decisive scale (NaN outside it)
    turning: np.ndarray = field(init=False)  # indices into `curve`
    mean_rate: float = field(init=False)  # arcsec/yr, the divisor of the angle

    def __post_init__(self):
        names = ('times', 'sigma', 'osculating', 'radius', 'osculating_radius')
        for name, value in zip(names, time_ordered(*(getattr(self, name) for name in names))):  # a backward run
            setattr(self, name, value)
        self.divisor, self.unit = time_unit(abs(self.times[-1] - self.times[0]))
        angle = getattr(self.analysis, 'angle', None)
        if angle is not None:
            self.curve = np.degrees(np.asarray(angle.angle_f, dtype=float))
            self.turning = np.asarray(self.analysis.geometry.turning_points.indices, dtype=int)
        else:
            self.curve, self.turning = np.degrees(self.sigma), np.zeros(0, dtype=int)
        self.mean_rate = rate(self.times, self.sigma)

    @property
    def t(self) -> np.ndarray:
        return self.times / self.divisor

    @cached_property
    def rate(self) -> np.ndarray:
        """d(curve)/dt in arcsec/yr."""
        return np.gradient(np.radians(self.curve), self.times) / RAD_PER_ARCSEC

    def time_norm(self) -> Normalize:
        return Normalize(self.t[0], self.t[-1])

    @classmethod
    def from_body(cls, body, resonance, sim) -> 'SecularDiagnostics':
        """The diagnostics of a body's secular or Lidov-Kozai angle; `sim` gives the analysis
        (`Simulation.long_period_analysis`) and the settings."""
        from resonances.secular import free_elements as free_module

        key = resonance.to_s()
        times = np.asarray(body.times) / (2 * np.pi)
        secular = resonance.type == 'secular'
        terms = resonance.formula.terms if secular else ()
        p = sum(t.coefficient for t in terms if t.index is None and t.mode == 'g') if secular else 1
        osculating_radius = np.asarray(body.ecc) if p else np.degrees(inclination_from_invariable_plane(body.inc, body.Omega))
        radius = osculating_radius
        if secular and sim.config.secular_reduced:
            z_e, z_i = reduced_vectors(times, body.ecc, body.varpi, body.inc, body.Omega, terms, ('e',) if p else ('i',))
            radius = np.abs(z_e) if p else np.degrees(inclination_from_vector(z_i))
        osculating = body.secular_angles_osculating.get(key) if secular else None
        extra = {} if secular else free_module.summary_fields(body.free_elements)
        gate = getattr(body, 'zlk_gates', {}).get(key)
        if gate is not None:
            extra['confirmed_ei'] = gate.confirmed_ei
        return cls(
            times=times,
            sigma=np.asarray(body.angles_unwrapped[key], dtype=float),
            osculating=np.unwrap(np.asarray(osculating if osculating is not None else body.angles.get(key), dtype=float)),
            radius=radius,
            osculating_radius=osculating_radius,
            p=p,
            analysis=sim.long_period_analysis(body, resonance),
            body_name=body.name,
            resonance=key,
            status=body.statuses.get(key),
            free_elements=None if secular else body.free_elements,
            extra_row=extra,
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
    """Rate of the smoothed angle, arcsec/yr; zero (`style.RATE_ZERO`), the mean rate dashed."""
    ax.axhline(0.0, **RATE_ZERO)
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
    def from_body(cls, body, resonance, sim, options=None) -> 'SecularPlotter':
        """The same signature as `MMRPlotter.from_body`; `sim` (a Simulation) gives the analysis and its settings."""
        return cls(SecularDiagnostics.from_body(body, resonance, sim), options)

    def title(self) -> str:
        d = self.d
        status = '' if d.status is None else f', status {d.status}'
        return f'Asteroid {d.body_name}, {d.resonance}{status}; divisor {d.mean_rate:+.3f}"/yr'

    def plot_combined(self) -> 'SecularPlotter':
        """The one-page V1 figure of the verdict (`draw_long_period_combined`)."""
        import matplotlib.pyplot as plt

        if self._figure is not None:
            plt.close(self._figure)
        self._figure = draw_long_period_combined(
            self.d.analysis, row=self.d.extra_row, free_elements=self.d.free_elements, options=self.options
        )
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


# ============================================================================== the one-page V1 figure
#
# The combined figure of a secular or Lidov-Kozai verdict, layout V1 of the MMR figure (24 x 13.5 in): long thin time
# panels on the left, two portraits below them (the first two slots of `layout.bottom`), every number of the decision on the right (`combined_numbers`).
# Everything printed comes from the analysis' CSV row (`LongPeriodAnalysis.summary_row()` plus the caller's extra
# columns), so the figure and summary.csv agree.

LONG_PERIOD_PANELS = (
    ('a', 'sigma unwrapped: raw band, sigma_f at the decisive scale, its centre, librating episodes, turning points'),
    ('b', 'sigma mod 360 around the centre: osculating points and sigma_f'),
    ('c', 'sigma_dot_f'),
    ('d', 'e: raw band and filtered, min-max written'),
    ('e', 'i: raw band and filtered, min-max written'),
    ('f', 'secular: e_dot_f against A sin(sigma_f) + B cos(sigma_f) + C (R^2); ZLK: the free omega against the osculating'),
    ('g', 'the centre of sigma_f (smoothed over W_c), its turns and largest monotone pass'),
)
LONG_PERIOD_LEGEND = (
    '(a) grey band: raw σ; black: σ_f at the scale the verdict rests on (P_c, or the centre scale W_c); thin grey: σ_f at '
    'P_c when the centre scale decided; red: centre; ▼▲ turning points; coloured fill #id: librating episodes (to the record edge they '
    'cover, else ± P/4 beyond the turns); grey strip: other runs, letter = reason (T over the top, F too few turning points, C centre moves, '
    'O no closed cycle).',
    '(b) grey dots: osculating σ; black: σ_f.  (c) dσ_f/dt, red: σ̇ = 0; pale dashed ends of (c), (f): the filter\'s edge '
    'spread, not analysed.  (d, e) grey band: osculating e, i; black: filtered at P_c, min – max.',
    '(f) secular: black the rate of the element σ contains (de_f/dt for ϖ, d sin(i_f/2)/dt for Ω; both, i dashed on the '
    'right axis), blue: A sin σ_f + B cos σ_f + C (R² written; a column, no status). ZLK: grey osculating ω, blue free ω '
    '(forced terms removed): a free ω that winds makes the libration kinematic.',
    '(g) grey: σ_f; red: its centre; ▼▲ turns of the centre; shaded: its largest monotone pass, when ≥ 1 rev.',
    '(h) (e cos σ, e sin σ), (sin(i/2) cos σ, sin(i/2) sin σ) for an angle of Ω alone; colour: time. (i) secular: σ_f '
    'against dσ_f/dt; ZLK: the free eccentricity vector.',
)
LONG_PERIOD_BLOCK_ORDER = (
    'verdict',
    'path',
    'checks',
    'margins',
    'record',
    'filter',
    'constants',
    'geometry',
    'episodes',
    'physics',
    'orbit',
)
ARCSEC_PER_RAD = 206264.80624709636
KYR = 1e3
ANGLE_EXTRA_TOP = 0.1  # (a): room above the view of sigma_f and the raw band (the turning points, the episode labels)


def long_period_specs(resonance_type: str):
    """The blocks of the right-hand side of a secular or ZLK figure (`combined_numbers.BlockSpec`)."""
    from resonances.plotting.combined_numbers import VERDICT_COLUMNS, BlockSpec
    from resonances.resonance.analysis.verdict import CHECKS_OF_TYPE, ResonanceType

    checks = CHECKS_OF_TYPE[ResonanceType(resonance_type)]
    physics = (
        BlockSpec(
            'GATE: FREE OMEGA, e-i',
            ('zlk_gate', 'zlk_gate_n_cycles', 'zlk_gate_min_cycles', 'zlk_gate_period_yr'),
            ('zlk_', 'free_', 'ei_', 'confirmed_ei'),
        )
        if resonance_type == 'lidov_kozai'
        else BlockSpec('RESPONSE OF e AND i (a column)', ('rsp_variables', 'rsp_e_R2', 'rsp_i_R2'), ('rsp_',))
    )
    return {
        'verdict': BlockSpec('VERDICT', VERDICT_COLUMNS, always=VERDICT_COLUMNS),
        'path': BlockSpec('DECISION PATH (ledger)'),
        'checks': BlockSpec('CHECKS', checks, always=checks),
        'margins': BlockSpec('MARGINS'),
        'record': BlockSpec(
            'RECORD',
            ('name', 'resonance', 'resonance_type', 'smp_span_yr', 'smp_step_yr', 'smp_n_rows', 'settings_hash', 'confidence_cells_sha'),
            ('smp_',),
        ),
        'filter': BlockSpec(
            'FILTER AND SCALE',
            ('flt_cutoff_yr', 'flt_cutoff_binding', 'sec_scale', 'sec_scale_cutoff_yr'),
            ('flt_',),
            always=('flt_cutoff_yr',),
        ),
        'constants': BlockSpec(
            'BODY AND PLANETARY FREQUENCIES', ('sec_g_arcsec_yr', 'sec_s_arcsec_yr', 'flt_forced_waves_yr'), ('sec_g', 'sec_s_')
        ),
        'geometry': BlockSpec(
            'GEOMETRY',
            ('geo_n_tp', 'geo_tp_prominence_deg', 'geo_period_cycle_yr', 'sec_n_librating_episodes', 'geo_centre_max_pass_rev'),
            ('geo_', 'sec_', 'exit_', 'entry_'),
            groups=(
                ('turning points', ('geo_n_tp*', 'geo_tp_*', 'geo_period_*', 'geo_window_*', 'sec_fine*')),
                ('runs', ('geo_n_runs', 'geo_n_episodes', 'sec_n_librating*', 'geo_drift*')),
                ('centre', ('geo_centre_*', 'geo_net_rev*', 'geo_band*', 'geo_wiggle*')),
                ('exit and entry', ('exit_*', 'entry_*')),
            ),
        ),
        'episodes': BlockSpec('EPISODES', (), ('ep_',)),
        'physics': physics,
        'orbit': BlockSpec('ORBIT: e, i', ('ctx_e_min', 'ctx_e_max', 'ctx_inc_min_deg', 'ctx_inc_max_deg', 'ctx_unphysical'), ('ctx_',)),
    }


def long_period_blocks(row, decision_path, episodes=(), constants=()):
    """The text blocks of a secular/ZLK figure from its CSV row (`combined_numbers.build_blocks`); `episodes` are
    episodes.csv rows (a table as in the MMR figure), `constants` the planetary frequencies of the angle's modes."""
    from resonances.plotting import combined_numbers as numbers
    from resonances.plotting.mmr_verdict_panel import _episode_lines

    resonance_type = str(row.get('resonance_type') or 'secular')
    specs = long_period_specs(resonance_type)
    table, shown = _episode_lines(list(episodes), row.get('ep_episode_id'))
    prefix = 'check_zlk_' if resonance_type == 'lidov_kozai' else 'check_secular_'
    blocks = numbers.build_blocks(
        row,
        specs,
        decision_path,
        specs['checks'].columns,
        prefix,
        alias_prefixes=('ep_', 'geo_', 'flt_', 'sec_', 'rsp_', 'zlk_', 'ctx_', 'smp_'),
        margin_order=('N', 'S', 'F', 'P', 'G'),
        extra={'constants': list(constants)},
    )
    blocks['episodes'].lines = table + [line for line in blocks['episodes'].lines if line.column not in shown and line.label]
    if not blocks['episodes'].lines:
        blocks['episodes'].lines.append(numbers.NumberLine('episodes', '', 'no turning-point runs', mark=numbers.MARK_NONE))
    return blocks


def _constant_lines(analysis):
    """'g6  28.2455"/yr  wave 37.1 kyr' for each planetary mode of a secular angle."""
    from resonances.plotting.combined_numbers import NumberLine

    lines = []
    waves = analysis.cutoff.forced_waves_yr if analysis.cutoff is not None else {}
    for term in analysis.series.terms:
        if term.index is None:
            continue
        label = f'{term.mode}{term.index}'
        frequency = analysis.series.planet_frequencies.get(label)
        wave = waves.get(label)
        text = f'{term.coefficient:+d} {label}: {frequency:.4f}″/yr' if frequency is not None else f'{term.coefficient:+d} {label}'
        if wave is not None:
            text += f', forced wave {wave / KYR:.4g} kyr'
        lines.append(NumberLine('constants', '', text, group='modes of the angle'))
    return lines


class _LongFrame:
    """Times in the display unit and the arrays of one figure."""

    def __init__(self, analysis, settings=None):
        from resonances.plotting.mmr_combined import _settings

        self.analysis = analysis
        self.settings = settings if settings is not None else _settings()  # the coverage rule of the episodes' fill
        series = analysis.series
        self.t_yr = np.asarray(series.t_yr, dtype=float)
        self.divisor, self.unit = time_unit(self.t_yr[-1] - self.t_yr[0])
        self.x = self.t_yr / self.divisor

    def times(self, t_yr):
        return np.asarray(t_yr, dtype=float) / self.divisor

    def px_per_unit(self, layout) -> float:
        """Pixels of a time panel per display time unit."""
        span = self.x[-1] - self.x[0]
        return layout.panels[2] * layout.dpi / span if span > 0 else 0.0


def _filtered(analysis, values):
    from resonances.resonance.analysis.filtering import lowpass

    cutoff = analysis.cutoff.cutoff_yr if analysis.cutoff is not None else None
    try:
        return lowpass(analysis.series.t_yr, np.asarray(values, dtype=float), cutoff, analysis.filter_order) if cutoff else None
    except ValueError:
        return None


def _mark_turns(ax, frame, tp, scale=np.degrees, centre: bool = False):
    """Turning points of the angle (grey) or turns of the centre (red, larger, white edge): ▼ maximum, ▲ minimum, as in
    the MMR figure (`mmr_combined.turn_markers`)."""
    from resonances.plotting.mmr_combined import CENTRE_TURN_SIZE, MAX_TP_MARKERS, TP_SIZE, turn_markers

    if tp is None or not len(tp) or len(tp) > MAX_TP_MARKERS:
        return
    size, colour = (CENTRE_TURN_SIZE, CENTRE) if centre else (TP_SIZE, REFERENCE)
    turn_markers(ax, frame.times(np.asarray(tp.times_yr)), scale(np.asarray(tp.values_rad)), tp.kinds, size, colour, white_edge=centre)


def _draw_angle(ax, frame, layout):
    from resonances.plotting.mmr_combined import _raw, _style_axis, angle_view, episode_spans, outlier_markers, shade_runs

    analysis = frame.analysis
    _raw(ax, frame.x, np.degrees(analysis.series.angle))
    geometry, angle = analysis.geometry, analysis.angle
    if geometry is not None:
        librating = sorted(_librating(analysis), key=lambda run: run.t_start_yr)
        colours = {run.episode_id: EPISODE_COLOURS[i % len(EPISODE_COLOURS)] for i, run in enumerate(librating)}
        spans = episode_spans(geometry.runs, (float(frame.t_yr[0]), float(frame.t_yr[-1])), frame.settings)
        shade_runs(ax, geometry.runs, set(colours), colours, frame.times, frame.px_per_unit(layout), layout, spans)
    if analysis.fine_geometry is not None and analysis.cutoff is not None:
        fine = _filtered(analysis, analysis.series.angle)
        if fine is not None:
            ax.plot(frame.x, np.degrees(fine), color=REFERENCE, alpha=0.6, lw=0.6, zorder=2)
    raw = np.degrees(analysis.series.angle)
    if angle is not None:
        ax.plot(frame.x, np.degrees(angle.angle_f), color=INK, lw=0.9, zorder=3)
    # the view holds sigma_f and the grey band of the raw sigma (as (a) of the MMR figure); samples beyond: triangles
    low, high = angle_view(np.degrees(angle.angle_f) if angle is not None else None, raw)
    ax.set_ylim(low, high + ANGLE_EXTRA_TOP * (high - low))
    outlier_markers(ax, frame.x, raw, *ax.get_ylim())
    if geometry is not None and len(geometry.centre.centre_rad):
        ax.plot(frame.times(geometry.centre.t_yr), np.degrees(geometry.centre.centre_rad), color=CENTRE, lw=0.9, zorder=4)
        _mark_turns(ax, frame, geometry.turning_points)
    _style_axis(ax, 'a', 'σ, °', layout)


def _librating(analysis):
    from resonances.resonance.analysis.secular.decide import librating_episodes

    return librating_episodes(analysis.geometry) if analysis.geometry is not None else []


def _draw_wrapped(ax, frame, layout):
    from resonances.plotting.mmr_combined import (
        _angle_formatter,
        _break_jumps,
        _line,
        _right_angles,
        _style_axis,
        _window_low,
        _wrap_window,
    )

    analysis = frame.analysis
    centre = analysis.geometry.centre.centre_rad if analysis.geometry is not None else analysis.series.angle
    low = _window_low(centre)
    ax.scatter(frame.x, np.degrees(_wrap_window(analysis.series.angle, low)), s=0.4, color=RAW, alpha=0.5, linewidths=0, rasterized=True)
    if analysis.angle is not None:
        wrapped = np.degrees(_break_jumps(_wrap_window(analysis.angle.angle_f, low), np.pi))
        _line(ax, frame.x, wrapped, unwrapped=analysis.angle.angle_f, color=INK, lw=0.8)
    base = np.degrees(low)
    ax.set_ylim(base, base + 360)
    _style_axis(ax, 'b', 'σ mod 360, °', layout)
    # the window starts wherever the centre puts it: right angles, labelled in [0, 360), as the MMR panel does
    ticks = _right_angles(base, inner=True)
    edge = 360 * layout.tick_margin
    ax.set_yticks(ticks[(ticks > base + edge) & (ticks < base + 360 - edge)])
    ax.yaxis.set_major_formatter(_angle_formatter())


def _draw_rate(ax, frame, layout):
    from resonances.plotting.mmr_combined import _edge_line, _style_axis, _view, edge_overflow_markers, rate_zero

    analysis = frame.analysis
    secular = str(analysis.verdict.resonance_type) == 'secular'
    if analysis.angle is not None:
        rate = np.asarray(analysis.angle.rate_f, dtype=float) * (ARCSEC_PER_RAD if secular else np.degrees(1.0) * KYR)
        inside = (frame.t_yr >= analysis.angle.window[0]) & (frame.t_yr <= analysis.angle.window[1])
        _edge_line(ax, frame, rate, inside=inside, color=INK, lw=0.8)  # the filter's edge spread pale and dashed
        rate_zero(ax)
        _view(ax, rate[inside])
        edge_overflow_markers(ax, frame, rate, inside=inside)
    _style_axis(ax, 'c', 'dσ/dt, ″/yr' if secular else 'dσ/dt, °/kyr', layout)


def _draw_element(ax, frame, layout, values, letter, label, fmt):
    """e or i: the osculating band, the series filtered at P_c, 'min – max' of it (as the MMR panels (f), (g))."""
    from resonances.plotting.mmr_combined import _raw, _style_axis, _value_text, _view

    values = np.asarray(values, dtype=float)
    _raw(ax, frame.x, values)
    filtered = _filtered(frame.analysis, values)
    if filtered is not None:
        ax.plot(frame.x, filtered, color=INK, lw=0.8, zorder=3)
        finite = filtered[np.isfinite(filtered)]
        if len(finite):
            _value_text(ax, f'{fmt.format(finite.min())} – {fmt.format(finite.max())}', layout)
    _view(ax, values, extra_top=0.25)
    _style_axis(ax, letter, label, layout)


def _draw_physics(ax, frame, layout, free_elements):
    from resonances.plotting.mmr_combined import _edge_line, _multiplier, _note, _power_of_ten, _style_axis, _view, edge_overflow_markers
    from resonances.resonance.analysis.filtering import time_derivative

    analysis = frame.analysis
    if str(analysis.verdict.resonance_type) == 'lidov_kozai':
        if free_elements is not None and len(free_elements.omega_free):
            ax.plot(frame.x, np.degrees(analysis.series.angle), color=RAW, alpha=RAW_ALPHA, lw=1.2)
            ax.plot(frame.times(free_elements.omega_times), np.degrees(free_elements.omega_free), color=MODEL, lw=0.8)
        else:
            _note(ax, 'free elements not built', layout.font_note_pt)
        _style_axis(ax, 'f', 'ω, °', layout)
        return
    variables = response_variables(analysis)
    names = {name: RESPONSE_VARIABLES[name] for name in variables}
    if analysis.angle is None or any(_filtered(analysis, values(analysis.series)) is None for _, values, _ in names.values()):
        _note(ax, 'not measured', layout.font_note_pt)
        _style_axis(ax, 'f', ' and '.join(label for label, _, _ in names.values()) + ', /kyr', layout)
        return
    inside = (frame.t_yr >= analysis.angle.window[0]) & (frame.t_yr <= analysis.angle.window[1])
    sigma = np.asarray(analysis.angle.angle_f, dtype=float)
    design = np.column_stack([np.sin(sigma), np.cos(sigma), np.ones_like(sigma)])
    texts = []
    for index, (name, (label, values, r2_of)) in enumerate(names.items()):
        target = ax if index == 0 else ax.twinx()  # the second variable (e and i): its own scale, on the right
        style = {} if index == 0 else {'ls': '--'}
        rate = time_derivative(frame.t_yr, _filtered(analysis, values(analysis.series))) * KYR
        power = _power_of_ten(rate[inside])
        rate = rate * 10.0**-power
        _edge_line(target, frame, rate, inside=inside, color=INK, lw=0.8, **style)
        if inside.sum() > 3:
            coef, *_ = np.linalg.lstsq(design[inside], rate[inside], rcond=None)
            target.plot(frame.x, np.where(inside, design @ coef, np.nan), color=MODEL, lw=0.8, **style)
            _view(target, rate[inside])
            edge_overflow_markers(target, frame, rate, inside=inside)
        ylabel = label + '\n' + _multiplier(power, '/kyr')
        if index == 0:
            _style_axis(ax, 'f', ylabel, layout)
        else:
            target.set_ylabel(ylabel + ' (dashed)', fontsize=layout.font_axis_pt, labelpad=2)
            target.tick_params(labelsize=layout.font_ticks_pt, length=2.5, pad=1.5)
        r2 = r2_of(analysis.response) if analysis.response is not None else float('nan')
        texts.append(f'R² {r2:.3g}' if len(names) == 1 else f'R² {name} {r2:.3g}')
    ax.text(0.99, 0.92, '   '.join(texts), transform=ax.transAxes, ha='right', va='top', fontsize=layout.font_note_pt, color=MODEL)


# The response variables of a secular angle (`secular.response`): label of panel (f), the series, its R² on the Response
RESPONSE_VARIABLES = {
    'e': (r'$\dot e$', lambda series: np.asarray(series.e, dtype=float), lambda response: response.r2_e),
    'i': (r'$d\,\sin(i/2)/dt$', lambda series: np.sin(np.asarray(series.inc, dtype=float) / 2), lambda response: response.r2_i),
}


def response_variables(analysis) -> tuple:
    """The elements the secular angle contains (`Response.variables`: 'e' for a body g-term, 'i' for an s-term), which
    panel (f) draws; 'e' when the response was not built."""
    response = analysis.response
    variables = tuple(name for name in (response.variables if response is not None else ()) if name in RESPONSE_VARIABLES)
    return variables or ('e',)


def portrait_radius(analysis):
    """(r, its label) of the portrait (h) of a secular angle: e when the angle contains ϖ (or for ZLK), sin(i/2) for an
    angle of the node alone (an s-term: ν16 = s − s6), the element whose momentum is conjugate to σ."""
    if str(analysis.verdict.resonance_type) == 'secular' and response_variables(analysis) == ('i',):
        return np.sin(np.asarray(analysis.series.inc, dtype=float) / 2), 'sin(i/2)'
    return np.asarray(analysis.series.e, dtype=float), 'e'


def _draw_centre(ax, frame, layout):
    from resonances.plotting.mmr_combined import _style_axis, _view, centre_pass_matters

    analysis = frame.analysis
    geometry = analysis.geometry
    if analysis.angle is not None:
        ax.plot(frame.x, np.degrees(analysis.angle.angle_f), color=RAW, alpha=RAW_ALPHA, lw=0.6)
    if geometry is not None and len(geometry.centre.centre_rad):
        centre = geometry.centre
        ax.plot(frame.times(centre.t_yr), np.degrees(centre.centre_rad), color=CENTRE, lw=1.0)
        _mark_turns(ax, frame, centre.turns, centre=True)
        if centre_pass_matters(centre):  # a pass of a revolution or more: what centre_pass reads
            ax.axvspan(frame.times(centre.max_pass_t0_yr), frame.times(centre.max_pass_t1_yr), color=CENTRE, alpha=0.08, lw=0)
        _view(ax, np.degrees(centre.centre_rad))
    _style_axis(ax, 'g', 'centre, °', layout)


def _draw_portraits(fig, frame, layout, norm, free_elements):
    from resonances.plotting.mmr_combined import _rect_axes, rate_zero

    analysis = frame.analysis
    stride = max(1, len(frame.x) // 25000)
    ax = _rect_axes(fig, layout.bottom[0])
    radius, name = portrait_radius(analysis)
    sigma = np.asarray(analysis.series.angle, dtype=float)
    ax.scatter(
        (radius * np.cos(sigma))[::stride],
        (radius * np.sin(sigma))[::stride],
        c=frame.x[::stride],
        norm=norm,
        cmap=time_cmap(),
        s=0.8,
        linewidths=0,
        rasterized=True,
    )
    ax.plot(0, 0, '+', color=CENTRE, ms=7)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel(f'{name} cos σ', fontsize=layout.font_axis_pt)
    ax.set_ylabel(f'{name} sin σ', fontsize=layout.font_axis_pt)
    ax.tick_params(labelsize=layout.font_ticks_pt)
    ax.set_title('(h)', fontsize=layout.font_letter_pt, loc='left', fontweight='bold')
    ax2 = _rect_axes(fig, layout.bottom[1])
    if str(analysis.verdict.resonance_type) == 'lidov_kozai' and free_elements is not None and len(free_elements.omega_free):
        x, y = free_elements.free_eccentricity_vector()
        t = frame.times(free_elements.omega_times)
        ax2.scatter(x, y, c=t, norm=norm, cmap=time_cmap(), s=0.8, linewidths=0, rasterized=True)
        ax2.plot(0, 0, '+', color=CENTRE, ms=7)
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.set_xlabel('e_free cos ω_free', fontsize=layout.font_axis_pt)
        ax2.set_ylabel('e_free sin ω_free', fontsize=layout.font_axis_pt)
    elif analysis.angle is not None:
        inside = (frame.t_yr >= analysis.angle.window[0]) & (frame.t_yr <= analysis.angle.window[1])
        scale = ARCSEC_PER_RAD if str(analysis.verdict.resonance_type) == 'secular' else np.degrees(1.0) * KYR
        ax2.scatter(
            np.degrees(np.asarray(analysis.angle.angle_f)[inside][::stride]),
            (np.asarray(analysis.angle.rate_f) * scale)[inside][::stride],
            c=frame.x[inside][::stride],
            norm=norm,
            cmap=time_cmap(),
            s=0.8,
            linewidths=0,
            rasterized=True,
        )
        rate_zero(ax2)
        ax2.set_xlabel('σ_f, °', fontsize=layout.font_axis_pt)
        ax2.set_ylabel('dσ_f/dt', fontsize=layout.font_axis_pt)
    ax2.tick_params(labelsize=layout.font_ticks_pt)
    ax2.set_title('(i)', fontsize=layout.font_letter_pt, loc='left', fontweight='bold')


def _draw_header(fig, analysis, layout, stamp):
    """Name, angle, type and the orbit's type (`verdict.orbit_type`); the record line. The status is printed once, in
    the plaque of the right column."""
    width, height = fig.get_size_inches()
    x, y, w, h = layout.header
    top = y + h
    verdict = analysis.verdict
    kind = 'ZLK' if str(verdict.resonance_type) == 'lidov_kozai' else 'secular'
    fig.text(
        x / width,
        top / height,
        f'{analysis.series.name}   {analysis.series.key}   [{kind}]'
        + (f'   {verdict.orbit_type}' if getattr(verdict, 'orbit_type', '') else ''),
        fontsize=layout.font_title_pt,
        fontweight='bold',
        va='top',
    )
    series = analysis.series
    cutoff = analysis.cutoff.cutoff_yr if analysis.cutoff is not None else float('nan')
    line = (
        f'span {series.span_yr:.6g} yr  |  step {series.step_yr:.4g} yr  |  {len(series.t_yr)} rows  |  '
        f'P_c {cutoff:.4g} yr ({analysis.cutoff.binding if analysis.cutoff is not None else "—"})  |  scale {analysis.scale or "—"}'
    )
    fig.text(x / width, (top - 0.36) / height, line, fontsize=layout.text_font_pt, color='#333333', va='top')
    if stamp:
        fig.text(
            (x + w) / width,
            top / height,
            stamp,
            fontsize=layout.font_letter_pt,
            va='top',
            ha='right',
            color='#b2182b',
            fontweight='bold',
        )


def _draw_legend(fig, layout):
    """The legend over the third slot of the bottom row and the legend's own box (`mmr_combined.legend_rect`)."""
    import textwrap

    from resonances.plotting.mmr_combined import legend_rect

    width, height = fig.get_size_inches()
    x, y, w, h = legend_rect(layout, third_slot_free=True)
    top = y + h
    fig.text(x / width, top / height, 'Legend', fontsize=layout.font_legend_pt + 1.5, fontweight='bold', va='top')
    chars = max(int(w * 72 / (0.56 * layout.font_legend_pt)), 20)
    text = '\n'.join(textwrap.fill(line, chars, subsequent_indent='   ') for line in LONG_PERIOD_LEGEND)
    fig.text(x / width, (top - 0.22) / height, text, fontsize=layout.font_legend_pt, va='top', color='#333333', linespacing=1.2)


def draw_long_period_combined(
    analysis, row=None, free_elements=None, stamp: Optional[str] = None, layout=None, options=None, settings=None
):
    """The V1 figure of a secular or Lidov-Kozai verdict (`LongPeriodAnalysis` with its arrays); `row` adds columns to
    the analysis' summary row (the body's free elements, e-i coherence); `free_elements` (ZLK) draws (f) and (i).
    The right column follows the plot option `combined.numbers` (explained | all); `settings`: the AnalysisSettings the
    explanation compares with (default: from the configuration). Returns the Figure (with `text_stats`)."""
    import dataclasses

    import matplotlib.pyplot as plt

    from resonances.plotting import combined_numbers as numbers
    from resonances.plotting.mmr_combined import _colour_bar, stacked_axes
    from resonances.plotting.style import LAYOUT_V1

    layout = layout or dataclasses.replace(LAYOUT_V1, n_panels=len(LONG_PERIOD_PANELS))
    summary = {**analysis.summary_row(), **(row or {})}
    fig = plt.figure(figsize=layout.size, dpi=layout.dpi)
    _draw_header(fig, analysis, layout, stamp)
    frame = _LongFrame(analysis, settings)
    width, height = fig.get_size_inches()
    axes = stacked_axes(fig, layout, len(LONG_PERIOD_PANELS))
    if analysis.series is not None and len(analysis.series.t_yr) > 1:
        _draw_angle(axes[0], frame, layout)
        _draw_wrapped(axes[1], frame, layout)
        _draw_rate(axes[2], frame, layout)
        _draw_element(axes[3], frame, layout, analysis.series.e, 'd', 'e', '{:.4g}')
        _draw_element(axes[4], frame, layout, np.degrees(analysis.series.inc), 'e', 'i, °', '{:.4g}°')
        _draw_physics(axes[5], frame, layout, free_elements)
        _draw_centre(axes[6], frame, layout)
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        axes[-1].set_xlabel(f't, {frame.unit}', fontsize=layout.font_axis_pt, labelpad=1)
        axes[-1].set_xlim(frame.x[0], frame.x[-1])
        norm = Normalize(frame.x[0], frame.x[-1])
        _draw_portraits(fig, frame, layout, norm, free_elements)
        _colour_bar(fig, norm, layout, frame.unit, label_on_top=True)  # (i)'s y label sits right of it (W23 F4)
    _draw_legend(fig, layout)
    fig.add_artist(plt.Line2D([layout.separator_x / width] * 2, [0.3 / height, (height - 0.2) / height], color='#dddddd', lw=0.8))
    if resolve_plot_options(options)['combined']['numbers'] == NumbersMode.ALL:
        constants = _constant_lines(analysis) if analysis.series.terms else []
        blocks = long_period_blocks(summary, analysis.verdict.decision_path, analysis.episode_rows(), constants)
        fig.text_stats = numbers.draw_text(fig, blocks, layout, LONG_PERIOD_BLOCK_ORDER)
        return fig
    fig.text_stats = draw_long_period_explained(fig, analysis, summary, layout, settings)
    return fig


# The numbers of the explained column of a secular/ZLK figure, most important first (a group with no value is left out)
LONG_PERIOD_EXPLAINED_GROUPS = (
    (
        'LIBRATION (deciding episode)',
        (
            'ep_centre_mod_2pi_deg',
            'ep_n_tp',
            'ep_period_med_yr',
            'ep_half_amp_med_deg',
            'ep_half_amp_max_deg',
            'ep_centre_estimates',
            'ep_coverage_start_gap_P',
            'ep_coverage_end_gap_P',
            'geo_n_tp',
            'geo_period_cycle_yr',
            'geo_centre_max_pass_rev',
            'geo_drift_fraction',
        ),
    ),
    (
        'SCALE, FILTER',
        ('sec_scale', 'sec_scale_cutoff_yr', 'flt_cutoff_yr', 'flt_cutoff_binding', 'sec_fine_n_tp', 'sec_fine_period_cycle_yr'),
    ),
    ('FREE-ω GATE', ('zlk_gate', 'zlk_gate_n_cycles', 'zlk_gate_min_cycles', 'zlk_gate_period_yr', 'confirmed_ei')),
    ('RESPONSE OF e AND i (a column)', ('rsp_variables', 'rsp_e_R2', 'rsp_i_R2')),
    ('ORBIT', ('ctx_e_min', 'ctx_e_max', 'ctx_inc_min_deg', 'ctx_inc_max_deg')),
    ('RECORD', ('smp_span_yr', 'smp_step_yr', 'flags', 'status_code', 'settings_hash', 'confidence_cells_sha')),
)
LONG_PERIOD_EXPLAINED_PREFIXES = ('ep_', 'geo_', 'sec_', 'flt_', 'zlk_', 'rsp_', 'ctx_', 'smp_')
SECULAR_MODES_SHOWN = ('g5', 'g6', 's6')
TURN_ARCSEC = 1296000.0


def long_period_period_lines(analysis):
    """CHARACTERISTIC PERIODS of a secular/ZLK figure: the body's g and s, the angle's own rate from them and the
    modes, the forced waves of the modes, and the modes g5, g6, s6 (and those of the angle)."""
    from resonances.plotting.combined_numbers import NumberLine
    from resonances.plotting.verdict_text import years

    cutoff, series = analysis.cutoff, analysis.series
    frequencies = dict(series.planet_frequencies)
    periods = []
    if cutoff is not None:
        body = {'g': cutoff.body_g_arcsec_yr, 's': cutoff.body_s_arcsec_yr}
        periods += [(f'{mode} of the body', TURN_ARCSEC / abs(rate)) for mode, rate in body.items() if np.isfinite(rate) and rate]
        own = sum(
            term.coefficient * (frequencies.get(f'{term.mode}{term.index}', np.nan) if term.index else body[term.mode])
            for term in series.terms
        )
        if series.terms and np.isfinite(own) and own:
            periods.append(('σ from mean rates', TURN_ARCSEC / abs(own)))
        periods += [(f'forced wave {label}', wave) for label, wave in cutoff.forced_waves_yr.items()]
    modes = list(SECULAR_MODES_SHOWN) + [f'{term.mode}{term.index}' for term in series.terms if term.index]
    for label in dict.fromkeys(modes):
        frequency = frequencies.get(label)
        if frequency:
            periods.append((label, TURN_ARCSEC / abs(frequency)))
    group = 'CHARACTERISTIC PERIODS'
    return [NumberLine('numbers', label, years(period), group=group) for label, period in periods if np.isfinite(period)]


def draw_long_period_explained(fig, analysis, summary, layout, settings=None):
    """The explained right column of a secular/ZLK figure (`verdict_text.explain_long_period`, layout A′)."""
    from resonances.plotting import combined_numbers as numbers
    from resonances.plotting.mmr_verdict_panel import config_settings, explained_episode_lines, short_hashes
    from resonances.plotting.verdict_text import explain_long_period

    episodes = analysis.episode_rows()
    text = explain_long_period(analysis, config_settings(settings), episodes)
    row = short_hashes(dict(summary))
    lines = explained_episode_lines(episodes, row.get('ep_episode_id'))
    for title, columns in LONG_PERIOD_EXPLAINED_GROUPS:
        lines += numbers.group_lines(row, ((title, columns),), LONG_PERIOD_EXPLAINED_PREFIXES, text.bold_columns, text.grey_columns)
        if title == 'SCALE, FILTER':
            lines += long_period_period_lines(analysis)
    stats = numbers.draw_explained(fig, text, lines, layout)
    stats['text'] = text
    return stats
