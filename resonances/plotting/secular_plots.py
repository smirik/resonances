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

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Optional

import numpy as np
from matplotlib.colors import Normalize

from resonances.data.util import time_ordered
from resonances.secular.free_elements import RAD_PER_ARCSEC, rate
from resonances.secular.proper_angle import inclination_from_invariable_plane, inclination_from_vector, reduced_vectors
from .base import DiagnosticPlotter, time_unit
from .style import CENTRE, EPISODE_COLOURS, INK, MODEL, OWNER, RATE_ZERO, RAW, RAW_ALPHA, REFERENCE, SUBRESONANCE
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
    ('f', 'secular: e_dot_f against A sin(sigma_f) + B cos(sigma_f) + C (R^2); ZLK: the drift of the free and the osculating omega, turns'),
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
    'right axis), blue: A sin σ_f + B cos σ_f + C (R² written; a column, no status). ZLK: the drift of ω from its first '
    'value in turns, grey osculating, blue free (forced terms removed; net turns written): a free ω that winds makes the '
    'libration kinematic.',
    '(g) grey: σ_f; red: its centre; ▼▲ turns of the centre; shaded: its largest monotone pass, when ≥ 1 rev.',
    '(h) (e cos σ, e sin σ), (sin(i/2) cos σ, sin(i/2) sin σ) for an angle of Ω alone; colour: time. (i) secular: σ_f '
    'against dσ_f/dt; ZLK: the free eccentricity vector.',
)
# (a′) of a ZLK row whose libration is induced (`forcing_panel_drawn`), drawn under (a); other rows keep seven panels
FORCING_PANEL = ('a′', 'omega_f with the forced direction against the node and the dominant mode\'s longitude against it')
FORCING_LEGEND = (
    '(a′) ZLK, an induced libration only: grey ω_f; orange arg z_forced − Ω, the direction of the fitted forced '
    'eccentricity vector against the node (the centre the free circle turns about; it carries the beat of the modes); '
    'purple dashed ϖ_j − Ω, the longitude of the dominant mode against the node (bounded: a node lock).'
)
FORCED_DIRECTION = SUBRESONANCE  # arg z_forced - Omega in (a′)
MODE_LINE = OWNER  # varpi_j - Omega in (a′): the mode that holds omega
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
        'orbit': BlockSpec(
            'ORBIT: a, e, i',
            (
                'ctx_a_med_au',
                'ctx_a_min_au',
                'ctx_a_max_au',
                'ctx_e_min',
                'ctx_e_max',
                'ctx_inc_min_deg',
                'ctx_inc_max_deg',
                'ctx_unphysical',
            ),
            ('ctx_',),
        ),
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


def forcing_panel_drawn(analysis, free_elements) -> bool:
    """Whether a ZLK figure gets (a′): a kinematic libration attributed to a forced mode (`zlk_forcing_*`) or a centre
    whose slow line is a beat of the forced vector (`zlk_centre_origin` forced_beat), with the per-mode amplitudes and
    the node to draw them from."""
    if str(analysis.verdict.resonance_type) != 'lidov_kozai' or analysis.series is None or analysis.series.Omega is None:
        return False
    if free_elements is None or not free_elements.amplitudes_e:
        return False
    zlk, line = getattr(analysis, 'zlk', None), getattr(analysis, 'centre_line', None)
    attributed = zlk is not None and zlk.attribution.attributed
    return bool(attributed or (line is not None and str(line.origin).startswith('forced_beat:')))


def _on_branch(curve, reference):
    """`curve` (unwrapped, rad) moved by whole turns to the branch of `reference` (median difference within half a turn)."""
    offset = np.nanmedian(np.asarray(reference, dtype=float) - curve)
    return curve + 2 * np.pi * np.round(offset / (2 * np.pi))


def _dominant_mode(analysis, free_elements) -> str:
    """The mode of (a′): the attributed one, else the fitted eccentricity mode of the largest amplitude (not const)."""
    zlk = getattr(analysis, 'zlk', None)
    if zlk is not None and zlk.attribution.attributed:
        return zlk.attribution.mode.split(' ')[0]
    modes = {
        label: abs(value) for label, value in free_elements.amplitudes_e.items() if label != 'const' and label in free_elements.basis_e
    }
    return max(modes, key=modes.get) if modes else ''


def _forcing_labels(analysis, free_elements, mode: str):
    """(legend of arg z_forced − Ω, legend of ϖ_j − Ω, the note on the right): modes and periods from the fitted
    amplitudes (`FreeElements.forced_beats`), the attribution and the node lock, as measured."""
    from resonances.plotting.verdict_text import _mode_longitude, finite, fmt, years

    amplitudes = sorted(
        ((abs(value), label) for label, value in free_elements.amplitudes_e.items() if label != 'const' and abs(value) > 0), reverse=True
    )
    top = [label for _, label in amplitudes[:2]]
    direction = 'arg z_forced − Ω: the forced direction'
    if len(top) == 2:
        beat = free_elements.forced_beats().get(f'{top[1]}-{top[0]}')
        direction += f' (|c| {top[0]} {fmt(amplitudes[0][0])} + {top[1]} {fmt(amplitudes[1][0])}, beat {years(beat)})'
    elif top:
        direction += f' (|c| {top[0]} {fmt(amplitudes[0][0])})'
    sym = _mode_longitude(mode)
    line = f'{sym} − Ω: the mode {mode} against the node'
    zlk = getattr(analysis, 'zlk', None)
    lock = zlk.node_lock if zlk is not None else None
    if lock is not None and lock.locked and lock.mode == mode:
        line += f' (locked: {lock.min_deg:.0f}–{lock.max_deg:.0f}°, {fmt(lock.cycles)} cycles of {years(lock.period_yr)})'
    elif lock is not None and lock.candidate == mode and finite(lock.rate_diff_arcsec_yr):
        line += f' (Ω̇ − {mode} {lock.rate_diff_arcsec_yr:+.3g}″/yr)'
    note = ''
    if zlk is not None and zlk.attribution.attributed:
        note = f'free vector about the forced one: beat {years(zlk.attribution.beat_period_yr)} = P_cyc'
    return direction, line, note


def _draw_forcing(ax, frame, layout, free_elements):
    """(a′) of an induced ZLK libration (`forcing_panel_drawn`): ω_f thin, the forced direction against the node
    arg z_forced(t) − Ω(t) (the centre the free circle turns about: the beat of the modes and the node's motion), and
    the dominant mode's longitude against the node ϖ_j(t) − Ω(t) (bounded when the node is locked to it), both from
    the fitted forced vector (`FreeElements.forced_vector`, amplitudes c_j at the rates g_j) and the osculating node.
    The curves are moved by whole turns onto ω_f's branch and drawn unwrapped when together they stay within a turn
    (591986: the node-lock wave at full scale); otherwise in the 360° window about the forced direction (624: ω
    circulates between its episodes, ω_f is drawn inside them only, where it follows the curves)."""
    from resonances.plotting.mmr_combined import _angle_formatter, _break_jumps, _line, _right_angles, _style_axis, _view, _wrap_window

    analysis = frame.analysis
    series = analysis.series
    omega = np.asarray(analysis.angle.angle_f if analysis.angle is not None else series.angle, dtype=float)
    node = np.asarray(series.Omega, dtype=float)
    mode = _dominant_mode(analysis, free_elements)
    curves = [_on_branch(np.unwrap(np.angle(free_elements.forced_vector(frame.t_yr)) - node), omega)]
    if mode:
        longitude = free_elements.basis_e[mode] * RAD_PER_ARCSEC * frame.t_yr + np.angle(free_elements.amplitudes_e[mode])
        curves.append(_on_branch(np.unwrap(longitude - node), omega))
    labels = _forcing_labels(analysis, free_elements, mode)
    styles = (
        {'color': FORCED_DIRECTION, 'lw': 1.0, 'zorder': 3, 'label': labels[0]},
        {'color': MODE_LINE, 'lw': 1.3, 'ls': '--', 'zorder': 4, 'label': labels[1]},
    )
    every = np.degrees(np.concatenate([omega] + curves))
    every = every[np.isfinite(every)]
    if len(every) and every.max() - every.min() < 360.0:
        ax.plot(frame.x, np.degrees(omega), color=RAW, lw=0.5, alpha=0.8, zorder=2, label='ω_f')
        for curve, style in zip(curves, styles):
            ax.plot(frame.x, np.degrees(curve), **style)
        _view(ax, every, extra_top=0.45)
    else:
        # ω circulates between its episodes: ω_f is drawn inside the librating episodes only, where it follows the curves
        low = float(np.nanmedian(curves[0])) - np.pi
        inside = np.zeros(len(frame.t_yr), dtype=bool)
        for run in _librating(analysis):
            inside |= (frame.t_yr >= run.t_start_yr) & (frame.t_yr <= run.t_end_yr)
        shown = np.where(inside, _wrap_window(omega, low), np.nan)
        _line(
            ax,
            frame.x,
            np.degrees(_break_jumps(shown, np.pi)),
            unwrapped=omega,
            color=INK,
            lw=0.6,
            zorder=2,
            label='ω_f in the librating episodes',
        )
        for curve, style in zip(curves, styles):
            ax.plot(frame.x, np.degrees(_break_jumps(_wrap_window(curve, low), np.pi)), **style)
        base = np.degrees(low)
        ax.set_ylim(base, base + 360 * 1.35)  # the top: the legend
        ticks = _right_angles(base, inner=True)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(_angle_formatter())
    ax.legend(
        loc='upper left',
        fontsize=layout.font_note_pt,
        frameon=False,
        ncol=3,
        handlelength=1.8,
        borderaxespad=0.2,
        bbox_to_anchor=(0.03, 1.0),
    )
    if labels[2]:
        box = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 0.6}
        ax.text(
            0.99, 0.06, labels[2], transform=ax.transAxes, ha='right', va='bottom', fontsize=layout.font_note_pt, color=REFERENCE, bbox=box
        )
    _style_axis(ax, FORCING_PANEL[0], 'ω, °', layout)


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

    analysis = frame.analysis
    if str(analysis.verdict.resonance_type) == 'lidov_kozai':
        _draw_free_omega_drift(ax, frame, layout, free_elements)
        return
    variables = response_variables(analysis)
    names = {name: RESPONSE_VARIABLES[name] for name in variables}
    fits = {name: _response_fit(analysis, frame, name) for name in names}
    if analysis.angle is None or any(fit is None for fit in fits.values()):
        _note(ax, 'not measured', layout.font_note_pt)
        _style_axis(ax, 'f', ' and '.join(label for label, _, _ in names.values()) + ', /kyr', layout)
        return
    texts = []
    for index, (name, (label, _, r2_of)) in enumerate(names.items()):
        target = ax if index == 0 else ax.twinx()  # the second variable (e and i): its own scale, on the right
        style = {} if index == 0 else {'ls': '--'}
        fit = fits[name]  # the analysis' own fit: x and sigma at the decisive cutoff, on its window
        inside = fit.mask
        power = _power_of_ten(fit.rate[inside] * KYR)
        scale = KYR * 10.0**-power
        _edge_line(target, frame, fit.rate * scale, inside=inside, color=INK, lw=0.8, **style)
        target.plot(frame.x, np.where(inside, fit.model * scale, np.nan), color=MODEL, lw=0.8, **style)
        _view(target, fit.rate[inside] * scale)
        edge_overflow_markers(target, frame, fit.rate * scale, inside=inside)
        ylabel = label + '\n' + _multiplier(power, '/kyr')
        if index == 0:
            _style_axis(ax, 'f', ylabel, layout)
        else:
            target.set_ylabel(ylabel + ' (dashed)', fontsize=layout.font_axis_pt, labelpad=2)
            target.tick_params(labelsize=layout.font_ticks_pt, length=2.5, pad=1.5)
        r2 = r2_of(analysis.response) if analysis.response is not None else float('nan')
        texts.append(f'R² {r2:.3g}' if len(names) == 1 else f'R² {name} {r2:.3g}')
    if analysis.scale == 'centre':  # the fit of the decisive scale: the centre's cutoff, not P_c of panels (d), (e)
        texts.append(f'at the centre scale {analysis.angle.cutoff_yr / KYR:.4g} kyr')
    ax.text(0.99, 0.92, '   '.join(texts), transform=ax.transAxes, ha='right', va='top', fontsize=layout.font_note_pt, color=MODEL)


def _draw_free_omega_drift(ax, frame, layout, free_elements):
    """ZLK (f): the drift of omega from its first value, in turns (as the standalone free-omega drift figure): grey the
    osculating omega, blue the free one (forced terms removed); a free omega that winds while the osculating one stays
    within a turn is the kinematic case. The free omega's net turns are written."""
    from resonances.plotting.mmr_combined import _note, _style_axis, _view

    if free_elements is None or not len(free_elements.omega_free):
        _note(ax, 'free elements not built', layout.font_note_pt)
        _style_axis(ax, 'f', 'ω drift, turns', layout)
        return
    turn = 2 * np.pi
    osculating = np.asarray(frame.analysis.series.angle, dtype=float)
    free = np.asarray(free_elements.omega_free, dtype=float)
    ax.plot(frame.x, (osculating - osculating[0]) / turn, color=RAW, alpha=RAW_ALPHA, lw=2.0, label='osculating ω')
    ax.plot(frame.times(free_elements.omega_times), (free - free[0]) / turn, color=MODEL, lw=0.9, label='free ω')
    ax.axhline(0.0, color=REFERENCE, lw=0.5, alpha=0.6)
    _view(ax, np.concatenate([(osculating - osculating[0]) / turn, (free - free[0]) / turn, [0.0]]), extra_top=0.3)
    ax.legend(loc='upper left', fontsize=layout.font_note_pt, frameon=False, ncol=2, handlelength=1.5, borderaxespad=0.2)
    ax.text(
        0.99,
        0.92,
        f'free ω: {free_elements.revolutions:+.3g} turns',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=layout.font_note_pt,
        color=MODEL,
    )
    _style_axis(ax, 'f', 'ω drift, turns', layout)


def _response_fit(analysis, frame, name: str):
    """The response fit of `name` ('e' or 'i') exactly as the analysis made it (`secular.response.fit_response`: the
    decisive angle's cutoff and window, the run's filter order), so the curve of panel (f) is the fit of its R²."""
    from resonances.resonance.analysis.secular.response import fit_response, response_series
    from resonances.resonance.analysis.settings import AnalysisSettings

    if analysis.angle is None:
        return None
    settings = replace(frame.settings if frame.settings is not None else AnalysisSettings(), filter_order=analysis.filter_order)
    return fit_response(frame.t_yr, response_series(analysis.series, name), analysis.angle, settings)


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


def _draw_legend(fig, layout, forcing: bool = False):
    """The legend over the third slot of the bottom row and the legend's own box (`mmr_combined.legend_rect`); with
    `forcing` the line of (a′)."""
    import textwrap

    from resonances.plotting.mmr_combined import legend_rect

    width, height = fig.get_size_inches()
    x, y, w, h = legend_rect(layout, third_slot_free=True)
    top = y + h
    fig.text(x / width, top / height, 'Legend', fontsize=layout.font_legend_pt + 1.5, fontweight='bold', va='top')
    chars = max(int(w * 72 / (0.56 * layout.font_legend_pt)), 20)
    legend = LONG_PERIOD_LEGEND[:1] + (FORCING_LEGEND,) + LONG_PERIOD_LEGEND[1:] if forcing else LONG_PERIOD_LEGEND
    text = '\n'.join(textwrap.fill(line, chars, subsequent_indent='   ') for line in legend)
    fig.text(x / width, (top - 0.22) / height, text, fontsize=layout.font_legend_pt, va='top', color='#333333', linespacing=1.2)


def draw_long_period_combined(
    analysis, row=None, free_elements=None, stamp: Optional[str] = None, layout=None, options=None, settings=None
):
    """The V1 figure of a secular or Lidov-Kozai verdict (`LongPeriodAnalysis` with its arrays); `row` adds columns to
    the analysis' summary row (the body's free elements, e-i coherence); `free_elements` (ZLK) draws (f) and (i), and
    (a′) under (a) for an induced libration (`forcing_panel_drawn`; eight time panels then, seven otherwise).
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
    forcing = analysis.series is not None and len(analysis.series.t_yr) > 1 and forcing_panel_drawn(analysis, free_elements)
    axes = stacked_axes(fig, layout, len(LONG_PERIOD_PANELS) + int(forcing))
    if analysis.series is not None and len(analysis.series.t_yr) > 1:
        _draw_angle(axes[0], frame, layout)
        if forcing:
            _draw_forcing(axes[1], frame, layout, free_elements)
        rest = axes[1 + int(forcing) :]
        _draw_wrapped(rest[0], frame, layout)
        _draw_rate(rest[1], frame, layout)
        _draw_element(rest[2], frame, layout, analysis.series.e, 'd', 'e', '{:.4g}')
        _draw_element(rest[3], frame, layout, np.degrees(analysis.series.inc), 'e', 'i, °', '{:.4g}°')
        _draw_physics(rest[4], frame, layout, free_elements)
        _draw_centre(rest[5], frame, layout)
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        axes[-1].set_xlabel(f't, {frame.unit}', fontsize=layout.font_axis_pt, labelpad=1)
        axes[-1].set_xlim(frame.x[0], frame.x[-1])
        norm = Normalize(frame.x[0], frame.x[-1])
        _draw_portraits(fig, frame, layout, norm, free_elements)
        _colour_bar(fig, norm, layout, frame.unit, label_on_top=True)  # (i)'s y label sits right of it (W23 F4)
    _draw_legend(fig, layout, forcing)
    fig.add_artist(plt.Line2D([layout.separator_x / width] * 2, [0.3 / height, (height - 0.2) / height], color='#dddddd', lw=0.8))
    if resolve_plot_options(options)['combined']['numbers'] == NumbersMode.ALL:
        constants = [replace(line, block='constants') for line in long_period_frequency_lines(analysis, summary, free_elements)]
        blocks = long_period_blocks(summary, analysis.verdict.decision_path, analysis.episode_rows(), constants)
        fig.text_stats = numbers.draw_text(fig, blocks, layout, LONG_PERIOD_BLOCK_ORDER)
        return fig
    fig.text_stats = draw_long_period_explained(fig, analysis, summary, layout, settings, free_elements)
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
    ('FREE-ω GATE', ('zlk_gate', 'zlk_gate_n_cycles', 'zlk_gate_min_cycles', 'zlk_gate_period_yr', 'zlk_gate_rho_max', 'zlk_ei_coupling')),
    ('RESPONSE OF e AND i (a column)', ('rsp_variables', 'rsp_e_R2', 'rsp_i_R2')),
    ('ORBIT', ('ctx_a_med_au', 'ctx_a_min_au', 'ctx_a_max_au', 'ctx_e_min', 'ctx_e_max', 'ctx_inc_min_deg', 'ctx_inc_max_deg')),
    ('RECORD', ('smp_span_yr', 'smp_step_yr', 'flags', 'status_code', 'settings_hash', 'confidence_cells_sha')),
)
LONG_PERIOD_EXPLAINED_PREFIXES = ('ep_', 'geo_', 'sec_', 'flt_', 'zlk_', 'rsp_', 'ctx_', 'smp_')
TURN_ARCSEC = 1296000.0  # a rate of f arcsec/yr turns once in 1296000 / |f| yr
FREQUENCY_GROUP = 'FREQUENCIES (″/yr; period = 360°/|f|)'
FREQUENCY_MODES_SHOWN = ('g5', 'g6', 's6')  # Jupiter's and Saturn's modes: always listed (as in the MMR figure)
FREQUENCY_BEAT_MODES = 2  # ZLK: g - g_j of the fitted forced modes with the largest |c_j|, and their mutual beat
PLANET_SHORT = ('Mer', 'Ven', 'Ear', 'Mar', 'Jup', 'Sat', 'Ura', 'Nep')


def _rate_text(value: float, constant: bool = False) -> str:
    """A rate in arcsec/yr: a planetary constant with 4 decimals (as in const.py), a measured rate to 4 significant
    digits; a minus sign."""
    if not np.isfinite(value):
        return '—'
    if constant:
        text = f'{value:+.4f}'
    else:
        decimals = max(0, 3 - int(np.floor(np.log10(abs(value))))) if value else 0
        text = f'{value:+.{decimals}f}'
    return text.replace('-', '−')


def _frequency_line(name: str, rate: float, note: str = '', period: Optional[float] = None, constant=False, rate_text=None):
    """One row 'name  rate  period  note' (monospaced, aligned); the period is 360°/|rate| unless given."""
    from resonances.plotting.combined_numbers import NumberLine
    from resonances.plotting.verdict_text import years

    if period is None:
        period = TURN_ARCSEC / abs(rate) if np.isfinite(rate) and rate else float('nan')
    rate_text = rate_text if rate_text is not None else _rate_text(rate, constant)
    text = f'{name:<8}{rate_text:>9} {years(period):>9}' + (f'  {note}' if note else '')
    return NumberLine('numbers', '', text, group=FREQUENCY_GROUP, keep=True)


def _mode_name(label: str) -> str:
    """'g5' -> 'g5 Jup'."""
    index = int(label[1:]) if label[1:].isdigit() else 0
    return f'{label} {PLANET_SHORT[index - 1]}' if 1 <= index <= len(PLANET_SHORT) else label


def _mode_order(label: str):
    return (label[0] != 'g', int(label[1:]) if label[1:].isdigit() else 0)


def _row_number(row, column: str, fallback: float = float('nan')) -> float:
    """A finite number of the row's column, else `fallback`."""
    try:
        value = float(row.get(column))
    except (TypeError, ValueError):
        return fallback
    return value if np.isfinite(value) else fallback


def _secular_frequency_lines(analysis, row) -> list:
    """The body's g and s (mean rates of its free vectors), sigma's rate from them, the modes g5, g6, s6 and those of
    the angle, and the forced waves g - g_j, s - s_k of an angle with two or more modes (`Cutoff.forced_waves_yr`)."""
    from resonances.data.const import PLANETARY_FREQUENCIES

    cutoff, series = analysis.cutoff, analysis.series
    frequencies = {**PLANETARY_FREQUENCIES, **dict(series.planet_frequencies)}
    body = {
        'g': _row_number(row, 'sec_g_arcsec_yr', cutoff.body_g_arcsec_yr if cutoff is not None else float('nan')),
        's': _row_number(row, 'sec_s_arcsec_yr', cutoff.body_s_arcsec_yr if cutoff is not None else float('nan')),
    }
    lines = [_frequency_line(mode, body[mode], f'free {name}') for mode, name in (('g', 'ϖ'), ('s', 'Ω')) if np.isfinite(body[mode])]
    planetary = [term for term in series.terms if term.index is not None]
    own = sum(
        term.coefficient * (frequencies.get(f'{term.mode}{term.index}', np.nan) if term.index else body[term.mode]) for term in series.terms
    )
    if series.terms and np.isfinite(own):
        lines.append(_frequency_line('σ', own, 'mean rates'))
    in_angle = {f'{term.mode}{term.index}': term.coefficient for term in planetary}
    for label in sorted(dict.fromkeys([*FREQUENCY_MODES_SHOWN, *in_angle]), key=_mode_order):
        if label in frequencies:
            note = f'in σ {in_angle[label]:+d}'.replace('-', '−') if label in in_angle else ''
            lines.append(_frequency_line(_mode_name(label), frequencies[label], note, constant=True))
    waves = cutoff.forced_waves_yr if cutoff is not None else {}
    for label, wave in waves.items():
        rate = body[label[0]] - frequencies.get(label, np.nan)
        lines.append(_frequency_line(f'{label[0]} − {label}', rate, 'forced', period=wave))
    return lines


def _zlk_frequency_lines(analysis, row, free_elements) -> list:
    """The body's free rates g, s and g - s (the free omega), the fitted forced modes (and g5, g6, s6) with |c_j|, the
    mode that holds varpi marked, the beats g - g_j of the FREQUENCY_BEAT_MODES largest forced modes (that of the
    holding mode is the kinematic omega cycle), their mutual beat (the slow wave of the forced vector) and the node
    lock's period when it is flagged."""
    from resonances.data.const import PLANETARY_FREQUENCIES

    elements = free_elements
    frequencies = {**PLANETARY_FREQUENCIES, **(dict(elements.frequencies) if elements is not None else {})}
    rates = {
        name: _row_number(row, f'free_b_{name}', float(getattr(elements, f'b_{name}')) if elements is not None else float('nan'))
        for name in ('varpi', 'Omega', 'omega')
    }
    lines = [
        _frequency_line(mode, rates[name], f'free {symbol}')
        for mode, name, symbol in (('g', 'varpi', 'ϖ'), ('s', 'Omega', 'Ω'), ('g − s', 'omega', 'ω'))
        if np.isfinite(rates[name])
    ]
    basis = {**(elements.basis_e if elements is not None else {}), **(elements.basis_i if elements is not None else {})}
    amplitudes = {**(elements.amplitudes_e if elements is not None else {}), **(elements.amplitudes_i if elements is not None else {})}
    holding = str(row.get('zlk_forcing_mode') or '').split(' ')[0] if row.get('zlk_attribution') == 'attributed' else ''
    labels = [label for label in basis if label != 'const']
    for label in sorted(dict.fromkeys([*FREQUENCY_MODES_SHOWN, *labels]), key=_mode_order):
        rate = basis.get(label, frequencies.get(label, np.nan))
        amplitude = abs(amplitudes[label]) if label in amplitudes else float('nan')
        note = (f'{amplitude:.2g}' if np.isfinite(amplitude) else ('' if label in basis else 'not fitted')) + (
            ' ◀' if label == holding else ''
        )
        lines.append(_frequency_line(_mode_name(label), rate, note.strip(), constant=True))
    forced = {label: abs(amplitudes.get(label, 0.0)) for label in labels if label.startswith('g')}
    largest = sorted((label for label in forced if forced[label] > 0), key=lambda label: -forced[label])[:FREQUENCY_BEAT_MODES]
    if not largest:  # no per-mode amplitudes (an old file): the beats of Jupiter's and Saturn's modes when fitted
        largest = [label for label in ('g5', 'g6') if label in basis]
    for label in sorted(largest, key=_mode_order):
        lines.append(_frequency_line(f'g − {label}', rates['varpi'] - basis[label], 'kinematic' if label == holding else ''))
    beats = elements.forced_beats() if elements is not None else {}
    if len(largest) == 2:
        key = next((key for key in (f'{largest[1]}-{largest[0]}', f'{largest[0]}-{largest[1]}') if key in beats), None)
        if key is not None:
            first, second = sorted(largest, key=_mode_order)
            lines.append(_frequency_line(f'{second} − {first}', basis[second] - basis[first], 'beat', period=beats[key], constant=True))
    lock = str(row.get('zlk_node_lock') or '')
    if lock:
        low, high = _row_number(row, 'zlk_node_lock_min_deg'), _row_number(row, 'zlk_node_lock_max_deg')
        span = f'{low:.0f}–{high:.0f}°' if np.isfinite(low) and np.isfinite(high) else ''
        period = _row_number(row, 'zlk_node_lock_period_yr')
        lines.append(_frequency_line(f'Ω − ϖ{lock[1:]}', float('nan'), span, period=period, rate_text='lock'))
    return lines


def long_period_frequency_lines(analysis, row, free_elements=None) -> list:
    """FREQUENCIES of a secular/ZLK figure: one row 'name  rate ″/yr  period  note' per frequency a period seen on
    sigma(t) may come from (period = 1296000/|rate| yr); `row` is the figure's summary row, `free_elements` (ZLK) the
    body's FreeElements (the fitted modes and their |c_j|)."""
    if str(analysis.series.resonance_type) == 'lidov_kozai':
        lines = _zlk_frequency_lines(analysis, row, free_elements)
    else:
        lines = _secular_frequency_lines(analysis, row)
    if not lines:
        return []
    from resonances.plotting.combined_numbers import NumberLine

    amplitude = '  |c|' if str(analysis.series.resonance_type) == 'lidov_kozai' else ''
    header = NumberLine('numbers', '', f'{"":<8}{"arcsec/yr":>9} {"period":>9}{amplitude}', group=FREQUENCY_GROUP, keep=True)
    return [header, *lines]


def draw_long_period_explained(fig, analysis, summary, layout, settings=None, free_elements=None):
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
            lines += long_period_frequency_lines(analysis, summary, free_elements)
    stats = numbers.draw_explained(fig, text, lines, layout)
    stats['text'] = text
    return stats
