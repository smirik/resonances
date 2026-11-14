"""Figures of a secular or Lidov-Kozai angle — what the phase portrait and the combined figure are for an MMR.

    angle    — sigma(t), degrees: the osculating angle in gray, the analyzed angle filtered at the scale the verdict
               rests on in black (`analysis.secular`: P_c, or the scale of the center), its turning points as red
               circles;
    rate     — the rate of that filtered angle, arcsec/yr, with zero and the mean rate (the divisor);
    portrait — (r cos sigma, r sin sigma), r = e when sigma holds varpi, else i from the invariable plane in
               degrees: osculating in gray, classified colored by time;
    combined — the one-page V1 figure of the verdict (`draw_long_period_combined`, below): time panels, four squares
               under them ((i) the osculating portrait, (j) sigma_f against its rate or the free eccentricity vector,
               (k) the Charlier portrait of sigma and e filtered at the decisive cutoff, (l) the same per episode;
               `charlier_radius`, `filtered_charlier`), the legend under those, and every number of the decision in
               the shared right column (`combined_numbers`). A ZLK row's ω, i, Ω and free elements are those of its
               analysis, measured from the plane ZLK_ANGLE_FRAME names (the header's record line says which; (a), (a′),
               (f), (i)-(l) share it).

Nothing here changes a status. Reading guide: docs/plots.md, "Secular diagnostics".
"""

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Optional

import numpy as np
from matplotlib.colors import Normalize

from resonances.data.util import time_ordered
from resonances.resonance.analysis.mmr.models import PLANET_NAMES_ORDER
from resonances.resonance.analysis.secular.models import ARCSEC_PER_TURN
from resonances.secular.free_elements import RAD_PER_ARCSEC, rate
from resonances.secular.proper_angle import inclination_from_invariable_plane, inclination_from_vector, reduced_vectors
from .base import DiagnosticPlotter, time_unit
from .style import CENTRE, EPISODE_COLOURS, INK, MODEL, OWNER, RATE_ZERO, RAW, RAW_ALPHA, REFERENCE, SUBRESONANCE
from .style import SECULAR_PLOT_KINDS, FigureMode, NumbersMode, resolve_plot_options, time_cmap, time_label
from .verdict_vocab import snake_words


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
    """(r cos sigma, r sin sigma): osculating in gray, classified colored by time, origin marked."""
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
# panels on the left, four squares below them (`long_period_layout`: (i)-(l)) and the legend under those, every number
# of the decision on the right (`combined_numbers`).
# Everything printed comes from the analysis' CSV row (`LongPeriodAnalysis.summary_row()` plus the caller's extra
# columns), so the figure and summary.csv agree.

LONG_PERIOD_PANELS = (
    ('a', 'sigma unwrapped: raw band, sigma_f at the decisive scale, its center, librating episodes, turning points'),
    ('b', 'sigma mod 360 around the center: osculating points and sigma_f'),
    ('c', 'sigma_dot_f'),
    ('d', "a: osculating band and a filtered at P_c, min-max; co-orbital: the planet's a and the Hill-zone edges"),
    ('e', 'e: raw band and filtered, min-max written'),
    ('f', 'i: raw band and filtered, min-max written'),
    ('g', 'secular: e_dot_f against A sin(sigma_f) + B cos(sigma_f) + C (R^2); ZLK: the drift of the free and the osculating omega, turns'),
    ('h', 'the center of sigma_f (smoothed over W_c), its turns and largest monotone pass'),
)
LONG_PERIOD_LEGEND = (
    '(a) gray band: raw σ; black: σ_f at the scale the verdict rests on (P_c, or the center scale W_c); thin gray: σ_f at '
    'P_c when the center scale decided; red: center; ▼▲ turning points; colored fill #id: librating episodes (to the record edge they '
    'cover, else ± P/4 beyond the turns); gray strip: other runs, letter = reason (T over the top, F too few turning points, C center moves, '
    'O no closed cycle).',
    '(b) gray dots: osculating σ; black: σ_f.  (c) dσ_f/dt, red: σ̇ = 0; pale dashed ends of (c), (g): the filter\'s edge '
    'spread, not analyzed.  (d, e, f) gray band: osculating a, e, i; black: filtered at P_c, min – max; (d) a co-orbital: '
    "dashed the planet's a, dotted the Hill-zone edges (± ZLK_COORBITAL_HILL_FACTOR r_H).",
    '(g) secular: black the rate of the element σ contains (de_f/dt for ϖ, d sin(i_f/2)/dt for Ω; both, i dashed on the '
    'right axis), blue: A sin σ_f + B cos σ_f + C (R² written; a column, no status). ZLK: the drift of ω from its first '
    'value in turns, gray osculating, blue free (forced terms removed; net turns written): a free ω that winds makes the '
    'libration kinematic.',
    '(h) gray: σ_f; red: its center; ▼▲ turns of the center; shaded: its largest monotone pass, when ≥ 1 rev.',
    '(i) (e cos σ, e sin σ) of the unfiltered σ and e, (sin(i/2) cos σ, sin(i/2) sin σ) for an angle of Ω alone; color: '
    'time. (j) secular: σ_f against dσ_f/dt; ZLK: the free eccentricity vector.',
    '(k) Charlier portrait (R cos σ_f, R sin σ_f): σ and e filtered at the cutoff the verdict rests on, inside the '
    'filter\'s window, R = √(2(1 − √(1 − e_f²))) (an angle of Ω alone: R = √(2√(1 − e_f²)(1 − cos i_f))); color: time; '
    'red +: the origin, which a libration does not go round and a circulation does. (l) the same over each episode of '
    '(a), in its color: ○ its start, dashed beyond its first and last turning point and for a candidate.',
)
# the line of (a) under the minimal policy (W29 SPEC §7): only the counted episodes are filled, the candidates faint
LONG_PERIOD_LEGEND_POLICY_A = (
    '(a) gray band: raw σ; black: σ_f at the scale the verdict rests on (P_c, or the center scale W_c); thin gray: σ_f at '
    'P_c when the center scale decided; red: center; ▼▲ turning points; colored fill #id: counted episodes (to the record '
    'edge they cover, pale in the filter trim, else ± P/4 beyond the turns); faint dashed cand: a candidate, not counted; '
    'other runs are not drawn (episodes.csv lists them).'
)
# (a′) of a ZLK row whose libration is induced (`forcing_panel_drawn`), drawn under (a); other rows keep eight panels
FORCING_PANEL = ('a′', 'omega_f with the forced direction against the node and the dominant mode\'s longitude against it')
FORCING_LEGEND = (
    '(a′) ZLK, an induced libration only: gray ω_f; orange arg z_forced − Ω, the direction of the fitted forced '
    'eccentricity vector against the node (the center the free circle turns about; it carries the beat of the modes); '
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
# the letters of the panels after (a)-(c) (W29: (d) is a, author d11): e, i, the physics panel, the centre, the portraits
AXIS_LETTER, ECC_LETTER, INC_LETTER, PHYSICS_LETTER, CENTRE_LETTER = 'd', 'e', 'f', 'g', 'h'
PORTRAIT_LETTERS = ('i', 'j', 'k', 'l')  # the squares of the bottom row; the 'panels' figure re-letters the drawn ones
PLANET_AXIS_STYLE = {'color': OWNER, 'lw': 1.0, 'ls': '--', 'alpha': 0.85, 'zorder': 5}  # (d): a_P of a co-orbital's planet
HILL_EDGE_STYLE = {'color': OWNER, 'lw': 0.8, 'ls': ':', 'alpha': 0.85, 'zorder': 5}  # (d): a_P (1 ± k (m_P/3)^(1/3))


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
                ('center', ('geo_centre_*', 'geo_net_rev*', 'geo_band*', 'geo_wiggle*')),
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
    """Turning points of the angle (gray) or turns of the center (red, larger, white edge): ▼ maximum, ▲ minimum, as in
    the MMR figure (`mmr_combined.turn_markers`)."""
    from resonances.plotting.mmr_combined import CENTRE_TURN_SIZE, MAX_TP_MARKERS, TP_SIZE, turn_markers

    if tp is None or not len(tp) or len(tp) > MAX_TP_MARKERS:
        return
    size, colour = (CENTRE_TURN_SIZE, CENTRE) if centre else (TP_SIZE, REFERENCE)
    turn_markers(ax, frame.times(np.asarray(tp.times_yr)), scale(np.asarray(tp.values_rad)), tp.kinds, size, colour, white_edge=centre)


def _draw_angle(ax, frame, layout):
    from resonances.plotting.mmr_combined import _raw, _style_axis, angle_view, episode_spans, outlier_markers, policy_marks
    from resonances.plotting.mmr_combined import shade_policy_runs, shade_runs

    analysis = frame.analysis
    _raw(ax, frame.x, np.degrees(analysis.series.angle))
    geometry, angle = analysis.geometry, analysis.angle
    if geometry is not None:
        colours = episode_colour_map(analysis)
        record = (float(frame.t_yr[0]), float(frame.t_yr[-1]))
        marks = policy_marks(analysis)
        if marks is not None:  # the minimal policy (W29 SPEC §7): counted episodes, candidates faint, nothing else
            window = tuple(angle.window) if angle is not None else record
            shade_policy_runs(
                ax, geometry.runs, marks, colours, frame.times, frame.px_per_unit(layout), layout, record, window, frame.settings
            )
        else:
            spans = episode_spans(geometry.runs, record, frame.settings)
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
    """Whether a ZLK figure gets (a′): a kinematic libration attributed to a forced mode (`zlk_forcing_*`) or a center
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
    arg z_forced(t) − Ω(t) (the center the free circle turns about: the beat of the modes and the node's motion), and
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


def episode_colour_map(analysis) -> dict:
    """The color of each librating episode (`decide.librating_episodes`), in time order, cycled: the fills of (a) and
    the curves of (l) share it."""
    librating = sorted(_librating(analysis), key=lambda run: run.t_start_yr)
    return {run.episode_id: EPISODE_COLOURS[i % len(EPISODE_COLOURS)] for i, run in enumerate(librating)}


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
    """a, e or i: the osculating band, the series filtered at P_c, 'min – max' of it (as the MMR panels (f), (g))."""
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


def hill_half_width(planet: str, hill_factor: float) -> float:
    """The co-orbital zone's half-width in units of a_P: hill_factor x (m_P / 3 M_sun)^(1/3) (ZLK_COORBITAL_HILL_FACTOR;
    0.030 for the Earth at the default 3)."""
    from resonances.data.const import PLANETS_MASS

    return hill_factor * (PLANETS_MASS[planet] / 3.0) ** (1.0 / 3.0)


def coorbital_planet_geometric(a_med: float, settings) -> str:
    """The planet whose co-orbital zone holds the body's median a, |<a>/a_P − 1| < `hill_half_width` (the planets' a and
    masses of const.py), else ''. For drawing (d) only (W29 SPEC §9): unlike `omega_free_gate.coorbital_planet` it does
    not ask whether the planet's modes are fitted, so a Jupiter Trojan and a secular row get the line too; it decides
    nothing."""
    from resonances.data.const import PLANETS_AXIS

    if a_med is None or not np.isfinite(a_med) or a_med <= 0:
        return ''
    factor = settings.zlk_coorbital_hill_factor
    for planet in PLANET_NAMES_ORDER:
        if abs(a_med / PLANETS_AXIS[planet] - 1.0) < hill_half_width(planet, factor):
            return planet
    return ''


def _draw_axis(ax, frame, layout):
    """(d): a as (e) and (f) draw e and i (the osculating band, a filtered at P_c, its min – max); for a co-orbital
    (`coorbital_planet_geometric`) the planet's a dashed and the edges of its co-orbital zone dotted, named at the right."""
    from resonances.data.const import PLANETS_AXIS

    a = np.asarray(frame.analysis.series.a, dtype=float)
    _draw_element(ax, frame, layout, a, AXIS_LETTER, 'a, au', '{:.6g}')
    finite = a[np.isfinite(a)]
    planet = coorbital_planet_geometric(float(np.median(finite)) if len(finite) else float('nan'), frame.settings)
    if not planet:
        return
    a_p = PLANETS_AXIS[planet]
    width = hill_half_width(planet, frame.settings.zlk_coorbital_hill_factor) * a_p
    ax.axhline(a_p, **PLANET_AXIS_STYLE, gid='planet_axis')
    for level in (a_p - width, a_p + width):
        ax.axhline(level, **HILL_EDGE_STYLE, gid='hill_edge')
    ax.text(
        0.99,
        0.92,
        f'{planet}: a_P {a_p:.6g} au (dashed); co-orbital zone ± {width:.3g} au (dotted)',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=layout.font_note_pt,
        color=OWNER,
    )


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
        _style_axis(ax, PHYSICS_LETTER, ' and '.join(label for label, _, _ in names.values()) + ', /kyr', layout)
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
            _style_axis(ax, PHYSICS_LETTER, ylabel, layout)
        else:
            target.set_ylabel(ylabel + ' (dashed)', fontsize=layout.font_axis_pt, labelpad=2)
            target.tick_params(labelsize=layout.font_ticks_pt, length=2.5, pad=1.5)
        r2 = r2_of(analysis.response) if analysis.response is not None else float('nan')
        texts.append(f'R² {r2:.3g}' if len(names) == 1 else f'R² {name} {r2:.3g}')
    if analysis.scale == 'centre':  # the fit of the decisive scale: the centre's cutoff, not P_c of panels (e), (f)
        texts.append(f'at the center scale {analysis.angle.cutoff_yr / KYR:.4g} kyr')
    ax.text(0.99, 0.92, '   '.join(texts), transform=ax.transAxes, ha='right', va='top', fontsize=layout.font_note_pt, color=MODEL)


def _draw_free_omega_drift(ax, frame, layout, free_elements):
    """ZLK (g): the drift of omega from its first value, in turns (as the standalone free-omega drift figure): gray the
    osculating omega, blue the free one (forced terms removed); a free omega that winds while the osculating one stays
    within a turn is the kinematic case. The free omega's net turns are written."""
    from resonances.plotting.mmr_combined import _note, _style_axis, _view

    if free_elements is None or not len(free_elements.omega_free):
        _note(ax, 'free elements not built', layout.font_note_pt)
        _style_axis(ax, PHYSICS_LETTER, 'ω drift, turns', layout)
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
    _style_axis(ax, PHYSICS_LETTER, 'ω drift, turns', layout)


def _response_fit(analysis, frame, name: str):
    """The response fit of `name` ('e' or 'i') exactly as the analysis made it (`secular.response.fit_response`: the
    decisive angle's cutoff and window, the run's filter order), so the curve of panel (g) is the fit of its R²."""
    from resonances.resonance.analysis.secular.response import fit_response, response_series
    from resonances.resonance.analysis.settings import AnalysisSettings

    if analysis.angle is None:
        return None
    settings = replace(frame.settings if frame.settings is not None else AnalysisSettings(), filter_order=analysis.filter_order)
    return fit_response(frame.t_yr, response_series(analysis.series, name), analysis.angle, settings)


# The response variables of a secular angle (`secular.response`): label of panel (g), the series, its R² on the Response
RESPONSE_VARIABLES = {
    'e': (r'$\dot e$', lambda series: np.asarray(series.e, dtype=float), lambda response: response.r2_e),
    'i': (r'$d\,\sin(i/2)/dt$', lambda series: np.sin(np.asarray(series.inc, dtype=float) / 2), lambda response: response.r2_i),
}


def response_variables(analysis) -> tuple:
    """The elements the secular angle contains (`Response.variables`: 'e' for a body g-term, 'i' for an s-term), which
    panel (g) draws; 'e' when the response was not built."""
    response = analysis.response
    variables = tuple(name for name in (response.variables if response is not None else ()) if name in RESPONSE_VARIABLES)
    return variables or ('e',)


def portrait_radius(analysis):
    """(r, its label) of the portrait (i) of a secular angle: e when the angle contains ϖ (or for ZLK), sin(i/2) for an
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
    _style_axis(ax, CENTRE_LETTER, 'center, °', layout)


def charlier_radius(e, inc=None) -> np.ndarray:
    """Charlier's radius R of the momentum conjugate to the angle, in units of sqrt(L) (Carruba, Aljbaae & Winter 2016,
    Fig. 3B): R = sqrt(2 (1 - sqrt(1 - e^2))) for an angle of the perihelion (sqrt(2P/L), P = L (1 - sqrt(1 - e^2)));
    with `inc` [rad], for an angle of the node alone, R = sqrt(2 sqrt(1 - e^2) (1 - cos i)) (sqrt(2Q/L),
    Q = L sqrt(1 - e^2) (1 - cos i)). R ~ e, resp. ~ 2 sin(i/2), for small e and i."""
    e = np.asarray(e, dtype=float)
    root = np.sqrt(np.clip(1.0 - e**2, 0.0, None))
    if inc is None:
        return np.sqrt(2.0 * (1.0 - root))
    return np.sqrt(2.0 * root * (1.0 - np.cos(np.asarray(inc, dtype=float))))


CHARLIER_TICKS = 4  # at most this many intervals on each axis of (k) and (l)


@dataclass(frozen=True)
class Charlier:
    """The filtered Charlier portrait of (k) and (l): sigma_f and R of the elements filtered at the same cutoff
    (`filtered_charlier`), on the analysis' time grid; `inside` is the filter's window (finite samples only)."""

    t_yr: np.ndarray
    sigma: np.ndarray  # rad, sigma_f at the decisive scale
    radius: np.ndarray  # R of e_f (and i_f)
    inside: np.ndarray  # bool
    name: str  # 'R(e)' or 'R(e, i)': the axis label

    @property
    def limit(self) -> float:
        """Half the side of the square view: 1.05 x the largest R inside, so the origin is always in the middle."""
        values = self.radius[self.inside]
        return 1.05 * float(np.max(values)) if len(values) and np.max(values) > 0 else 1.0


def filtered_charlier(analysis, settings=None) -> Optional[Charlier]:
    """sigma_f and Charlier's R (`charlier_radius`) of e (and i for an angle of the node alone, `portrait_radius`)
    filtered at sigma_f's own cutoff (`angle.cutoff_yr`: P_c, or W_c when the center scale decided) with the analysis'
    filter order, inside the filter's window; None without a filtered angle, when the elements cannot be filtered or
    no sample is left."""
    from resonances.resonance.analysis.filtering import lowpass_or_none
    from resonances.resonance.analysis.settings import AnalysisSettings

    angle, series = analysis.angle, analysis.series
    if angle is None or series is None or len(series.t_yr) < 2:
        return None
    settings = replace(settings if settings is not None else AnalysisSettings(), filter_order=analysis.filter_order)
    t = np.asarray(series.t_yr, dtype=float)
    e_f = lowpass_or_none(t, np.asarray(series.e, dtype=float), angle.cutoff_yr, settings)
    nodal = portrait_radius(analysis)[1] == 'sin(i/2)'
    inc_f = lowpass_or_none(t, np.asarray(series.inc, dtype=float), angle.cutoff_yr, settings) if nodal else None
    if e_f is None or (nodal and inc_f is None):
        return None
    sigma = np.asarray(angle.angle_f, dtype=float)
    radius = charlier_radius(e_f, inc_f)
    inside = (t >= angle.window[0]) & (t <= angle.window[1]) & np.isfinite(sigma) & np.isfinite(radius)
    if not inside.any():
        return None
    return Charlier(t, sigma, radius, inside, 'R(e, i)' if nodal else 'R(e)')


def _charlier_axes(ax, charlier: Charlier, layout, letter: str):
    """The square view about the origin (+), the labels and the panel's letter, shared by (k) and (l)."""
    from matplotlib.ticker import MaxNLocator

    limit = charlier.limit
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    for axis in (ax.xaxis, ax.yaxis):  # a 2.25-in square: few ticks, or the labels of R ~ 0.0x run into each other
        axis.set_major_locator(MaxNLocator(CHARLIER_TICKS, symmetric=True))
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color=RAW, lw=0.5, ls='--', zorder=0)
    ax.axvline(0, color=RAW, lw=0.5, ls='--', zorder=0)
    ax.plot(0, 0, '+', color=CENTRE, ms=7, zorder=6)
    ax.set_xlabel(f'{charlier.name} cos σ_f', fontsize=layout.font_axis_pt, labelpad=1)
    ax.set_ylabel(f'{charlier.name} sin σ_f', fontsize=layout.font_axis_pt, labelpad=1)
    ax.tick_params(labelsize=layout.font_ticks_pt)
    ax.set_title(f'({letter})', fontsize=layout.font_letter_pt, loc='left', fontweight='bold')


def _draw_charlier(ax, frame, layout, norm, charlier: Charlier, letter: str):
    """(k): the filtered Charlier portrait over the filter's window, colored by time (a line: the series is smooth)."""
    from matplotlib.collections import LineCollection

    inside = charlier.inside
    stride = max(1, int(inside.sum()) // 25000)
    x = (charlier.radius * np.cos(charlier.sigma))[inside][::stride]
    y = (charlier.radius * np.sin(charlier.sigma))[inside][::stride]
    times = frame.x[inside][::stride]
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = LineCollection(segments, cmap=time_cmap(), norm=norm, linewidths=0.7, rasterized=True)
    collection.set_array(times[:-1])
    ax.add_collection(collection)
    _charlier_axes(ax, charlier, layout, letter)


def portrait_episodes(analysis, settings=None) -> tuple:
    """The episodes of (l) and how to draw them: (runs, their fill spans [yr] as in (a), colors, policy marks or None).
    Under the minimal policy the counted episodes and the candidates (as (a) draws them), else the librating ones
    (`decide.librating_episodes`); the PORTRAIT_MAX_EPISODES longest (counted first), in time order. `settings`: the
    coverage rule of the fill (`mc.episode_spans`), as in (a)."""
    from resonances.plotting.mmr_combined import _portrait_episodes, episode_spans, policy_marks

    geometry = analysis.geometry
    if geometry is None or analysis.series is None:
        return [], {}, {}, None
    marks = policy_marks(analysis)
    record = (float(analysis.series.t_yr[0]), float(analysis.series.t_yr[-1]))
    if marks is not None:
        runs = [run for run in geometry.runs if run.episode_id in marks.counted or run.episode_id in marks.candidates]
        spans = episode_spans(runs, record, settings)
    else:
        runs = list(_librating(analysis))
        spans = episode_spans(geometry.runs, record, settings)
    chosen = _portrait_episodes(runs, marks.counted if marks is not None else None)
    return chosen, spans, episode_colour_map(analysis), marks


def _draw_charlier_episodes(ax, frame, layout, charlier: Charlier, letter: str, episodes) -> None:
    """(l): the filtered Charlier portrait over each episode of `portrait_episodes`, in its color of (a), over its fill
    span inside the filter's window (`mc.portrait_span`): solid between its first and last turning point, dashed beyond
    them (`mc.portrait_pieces`, at most PORTRAIT_MAX_CYCLES cycles of a long one) and for a candidate the policy does not
    count; ○ at its start; labeled '#id [cand] t0–t1 unit' (with more than two episodes '#id [cand]', in three
    columns) in a legend above the square."""
    from resonances.plotting.mmr_combined import PORTRAIT_CORE_STYLE, PORTRAIT_EXTENSION_STYLE, _policy_tag, portrait_pieces
    from resonances.plotting.mmr_combined import portrait_span
    from resonances.plotting.style import REJECTED

    runs, spans, colours, marks = episodes
    timed = len(runs) <= 2  # more episodes: '#id' only (their times are on (a)), so the legend fits over the square
    tp = frame.analysis.geometry.turning_points
    t = charlier.t_yr
    window = tuple(frame.analysis.angle.window)
    for run in runs:
        colour = colours.get(run.episode_id, REJECTED if marks is not None else EPISODE_COLOURS[0])
        counted = marks is None or run.episode_id in marks.counted
        tag = _policy_tag(run.episode_id, marks) if marks is not None else ''
        label = f'#{run.episode_id}{tag}'
        if timed:
            label += f' {frame.times(run.t_start_yr):.3g}–{frame.times(run.t_end_yr):.3g} {frame.unit}'
        pieces = portrait_pieces(run, portrait_span(run, spans, window), tp.times_yr, tp.kinds)
        labelled = next((n for n, piece in enumerate(pieces) if not piece[2]), 0)
        for n, (t0, t1, extended) in enumerate(pieces):
            mask = (t >= t0) & (t <= t1) & charlier.inside
            x = charlier.radius[mask] * np.cos(charlier.sigma[mask])
            y = charlier.radius[mask] * np.sin(charlier.sigma[mask])
            look = PORTRAIT_EXTENSION_STYLE if extended or not counted else PORTRAIT_CORE_STYLE
            ax.plot(x, y, color=colour, label=label if n == labelled else None, rasterized=True, **look)
            if n == 0 and len(x):
                ax.plot(x[:1], y[:1], 'o', ms=3, color=colour)
    _charlier_axes(ax, charlier, layout, letter)
    handles = ax.get_legend_handles_labels()[0]
    if handles:  # above the square, right of its letter: inside, it covered the origin the panel is read against
        ax.legend(
            fontsize=layout.font_note_pt,
            loc='lower right',
            bbox_to_anchor=(1.0, 1.0),
            ncol=1 if len(handles) <= 2 else 3,
            frameon=False,
            handlelength=1.2,
            borderaxespad=0.1,
            columnspacing=0.8,
        )


def _free_vector_drawn(analysis, free_elements) -> bool:
    """Whether the second portrait is the free eccentricity vector: a ZLK row with its free elements."""
    return str(analysis.verdict.resonance_type) == 'lidov_kozai' and free_elements is not None and bool(len(free_elements.omega_free))


def _second_portrait_drawn(analysis, free_elements) -> bool:
    """Whether the second portrait has data: the free eccentricity vector of a ZLK row, else the filtered angle."""
    return _free_vector_drawn(analysis, free_elements) or analysis.angle is not None


def _draw_raw_portrait(ax, frame, layout, norm, letter: str):
    """(i): the osculating portrait (r cos σ, r sin σ), r = e or sin(i/2) (`portrait_radius`), colored by time."""
    analysis = frame.analysis
    stride = max(1, len(frame.x) // 25000)
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
    ax.set_title(f'({letter})', fontsize=layout.font_letter_pt, loc='left', fontweight='bold')


def _draw_second_portrait(ax, frame, layout, norm, free_elements, letter: str):
    """(j): the free eccentricity vector of a ZLK row, else σ_f against dσ_f/dt inside the filter's window."""
    from resonances.plotting.mmr_combined import rate_zero

    analysis = frame.analysis
    if _free_vector_drawn(analysis, free_elements):
        x, y = free_elements.free_eccentricity_vector()
        t = frame.times(free_elements.omega_times)
        ax.scatter(x, y, c=t, norm=norm, cmap=time_cmap(), s=0.8, linewidths=0, rasterized=True)
        ax.plot(0, 0, '+', color=CENTRE, ms=7)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('e_free cos ω_free', fontsize=layout.font_axis_pt)
        ax.set_ylabel('e_free sin ω_free', fontsize=layout.font_axis_pt)
    elif analysis.angle is not None:
        stride = max(1, len(frame.x) // 25000)
        inside = (frame.t_yr >= analysis.angle.window[0]) & (frame.t_yr <= analysis.angle.window[1])
        scale = ARCSEC_PER_RAD if str(analysis.verdict.resonance_type) == 'secular' else np.degrees(1.0) * KYR
        ax.scatter(
            np.degrees(np.asarray(analysis.angle.angle_f))[inside][::stride],
            (np.asarray(analysis.angle.rate_f) * scale)[inside][::stride],
            c=frame.x[inside][::stride],
            norm=norm,
            cmap=time_cmap(),
            s=0.8,
            linewidths=0,
            rasterized=True,
        )
        rate_zero(ax)
        ax.set_xlabel('σ_f, °', fontsize=layout.font_axis_pt)
        ax.set_ylabel('dσ_f/dt', fontsize=layout.font_axis_pt)
    ax.tick_params(labelsize=layout.font_ticks_pt)
    ax.set_title(f'({letter})', fontsize=layout.font_letter_pt, loc='left', fontweight='bold')


def _draw_portraits(fig, frame, layout, norm, free_elements, skip_empty: bool = False) -> list:
    """The four squares of the bottom row (`layout.bottom`): (i) the osculating portrait, (j) the free eccentricity
    vector (ZLK) or σ_f against its rate, (k) the filtered Charlier portrait (`filtered_charlier`), (l) the same per
    episode (`portrait_episodes`). In the full figure an empty one keeps its frame with the reason; with `skip_empty`
    (the 'panels' figure) it is left out and the next ones move left and take the next letter. Returns the letters drawn."""
    from resonances.plotting.mmr_combined import _empty, _rect_axes

    analysis = frame.analysis
    slots, letters, drawn = list(layout.bottom), iter(PORTRAIT_LETTERS), []

    def take():
        letter = next(letters)
        drawn.append(letter)
        return _rect_axes(fig, slots.pop(0)), letter

    ax, letter = take()
    _draw_raw_portrait(ax, frame, layout, norm, letter)
    if not skip_empty or _second_portrait_drawn(analysis, free_elements):
        ax, letter = take()
        _draw_second_portrait(ax, frame, layout, norm, free_elements, letter)
    charlier = filtered_charlier(analysis, frame.settings)
    if charlier is not None or not skip_empty:
        ax, letter = take()
        if charlier is None:
            _empty(ax, 'not computed: no filtered angle or elements', layout)
            ax.set_title(f'({letter})', fontsize=layout.font_letter_pt, loc='left', fontweight='bold')
        else:
            _draw_charlier(ax, frame, layout, norm, charlier, letter)
    episodes = portrait_episodes(analysis, frame.settings) if charlier is not None else ([], {}, {}, None)
    if episodes[0] or not skip_empty:
        ax, letter = take()
        if not episodes[0]:
            _empty(ax, 'no librating episode' if charlier is not None else 'not computed: no filtered angle or elements', layout)
            ax.set_title(f'({letter})', fontsize=layout.font_letter_pt, loc='left', fontweight='bold')
        else:
            _draw_charlier_episodes(ax, frame, layout, charlier, letter, episodes)
    return drawn


def _inclination_label(series) -> str:
    """The label of (f): the inclination of a ZLK series measured from the invariable plane (ZLK_ANGLE_FRAME) says so."""
    return 'i (invariable), °' if getattr(series, 'frame', '') == 'invariable' else 'i, °'


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
    binding = analysis.cutoff.binding if analysis.cutoff is not None else '—'
    scale = snake_words(analysis.scale) if analysis.scale else '—'
    line = (
        f'span {series.span_yr:.6g} yr  |  step {series.step_yr:.4g} yr  |  {len(series.t_yr)} rows  |  '
        f'P_c {cutoff:.4g} yr ({binding})  |  scale {scale}'
    )
    if kind == 'ZLK':
        line += f'  |  ω, i, Ω from the {getattr(series, "frame", "ecliptic")} plane'
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


def long_period_legend(forcing: bool = False, policy: bool = False) -> tuple:
    """LONG_PERIOD_LEGEND, with the line of (a′) after (a) when `forcing`, and (a)'s line of the minimal policy when
    `policy` (`mmr_combined.policy_marks`)."""
    legend = ((LONG_PERIOD_LEGEND_POLICY_A,) if policy else LONG_PERIOD_LEGEND[:1]) + LONG_PERIOD_LEGEND[1:]
    return legend[:1] + (FORCING_LEGEND,) + legend[1:] if forcing else legend


# The bottom row of the secular/ZLK figure (W36, the author 2026-11-07): four squares (i)-(l), a little smaller than the
# MMR figure's three (2.4 in) so that they fit left of the separator and inside the 12.45-in paper crop; the time color
# bar right of (i) (its label on top); room for the y labels of the next square between them
# (0.7 in after the bar's tick labels: 591986's 'e_free sin ω_free' beside '−0.015' ran into them at 0.55 in)
LONG_PERIOD_SQUARE_IN = 2.2
LONG_PERIOD_BOTTOM = tuple((x, 0.5, LONG_PERIOD_SQUARE_IN, LONG_PERIOD_SQUARE_IN) for x in (0.95, 4.35, 7.2, 10.0))
LONG_PERIOD_COLOUR_BAR = (3.22, 0.5, 0.1, LONG_PERIOD_SQUARE_IN)
# The legend is no longer in a slot of the bottom row: it takes a band under the row, across the left half, in
# LONG_PERIOD_LEGEND_COLUMNS columns; the full page is that much taller (the 'panels' figure has no legend and no band)
LONG_PERIOD_LEGEND_BAND_IN = 1.45
LONG_PERIOD_LEGEND_COLUMNS = 2
LONG_PERIOD_LEGEND_GAP_IN = 0.25  # between the legend's columns
LEGEND_HEADING_IN = 0.22  # the 'Legend' heading over the columns


def long_period_layout(n_panels: Optional[int] = None, legend: bool = True, base=None):
    """The layout of the secular/ZLK figure: `base` (default LAYOUT_V1) with `n_panels` time panels (default: the eight
    of LONG_PERIOD_PANELS) and the four squares of
    LONG_PERIOD_BOTTOM; with `legend` (the full figure) everything moved up by LONG_PERIOD_LEGEND_BAND_IN and the legend
    in the band under the bottom row (the D8 growth for the right column, `growth_in`, is not touched: it starts from
    this page)."""
    from resonances.plotting.style import LAYOUT_V1

    n_panels = len(LONG_PERIOD_PANELS) if n_panels is None else n_panels
    base = LAYOUT_V1 if base is None else base
    layout = replace(base, n_panels=n_panels, bottom=LONG_PERIOD_BOTTOM, colour_bar=LONG_PERIOD_COLOUR_BAR)
    if not legend:
        return layout
    band = LONG_PERIOD_LEGEND_BAND_IN
    left = layout.header[0]
    rect = (left, 0.08, layout.separator_x - 0.1 - left, band - 0.08)
    return replace(layout.grown(band), growth_in=layout.growth_in, legend=rect)


def legend_columns(lines, n: int = LONG_PERIOD_LEGEND_COLUMNS, chars: int = 100) -> list:
    """`lines` split in order into at most `n` columns whose tallest is as short as it can be, each line counted by the
    rows it wraps to at `chars` characters (`textwrap.wrap`, as `_draw_legend` wraps it)."""
    import textwrap
    from itertools import combinations

    lines = list(lines)
    rows = [max(1, len(textwrap.wrap(line, chars, subsequent_indent='   '))) for line in lines]
    best = [lines]
    best_height = sum(rows)
    for cuts in combinations(range(1, len(lines)), min(n, len(lines)) - 1):
        bounds = (0, *cuts, len(lines))
        height = max(sum(rows[i:j]) for i, j in zip(bounds[:-1], bounds[1:]))
        if height < best_height:
            best, best_height = [lines[i:j] for i, j in zip(bounds[:-1], bounds[1:])], height
    return best


def _draw_legend(fig, layout, forcing: bool = False, policy: bool = False) -> float:
    """The legend in `layout.legend` (the band under the bottom row, `long_period_layout`), in LONG_PERIOD_LEGEND_COLUMNS
    columns (`legend_columns`, balanced at the font tried); with `forcing` the line of (a′), with `policy` (a)'s line of the minimal policy. The font
    shrinks (to `mc.LEGEND_MIN_FONT_PT`) until every column fits the band. Returns the font size used."""
    import textwrap

    from resonances.plotting.mmr_combined import LEGEND_CHAR_WIDTH, LEGEND_MIN_FONT_PT

    width, height = fig.get_size_inches()
    x, y, w, h = layout.legend
    top = y + h
    renderer = fig.canvas.get_renderer()
    fig.text(x / width, top / height, 'Legend', fontsize=layout.font_legend_pt + 1.5, fontweight='bold', va='top')
    lines = long_period_legend(forcing, policy)
    gap = LONG_PERIOD_LEGEND_GAP_IN
    column_w = (w - gap * (LONG_PERIOD_LEGEND_COLUMNS - 1)) / LONG_PERIOD_LEGEND_COLUMNS
    font = layout.font_legend_pt
    while True:
        chars = max(int(column_w * 72 / (LEGEND_CHAR_WIDTH * font)), 20)
        columns = legend_columns(lines, LONG_PERIOD_LEGEND_COLUMNS, chars)
        artists = [
            fig.text(
                (x + i * (column_w + gap)) / width,
                (top - LEGEND_HEADING_IN) / height,
                '\n'.join(textwrap.fill(line, chars, subsequent_indent='   ') for line in column),
                fontsize=font,
                va='top',
                color='#333333',
                linespacing=1.2,
            )
            for i, column in enumerate(columns)
        ]
        boxes = [artist.get_window_extent(renderer) for artist in artists]
        room = (column_w * fig.dpi + 0.5, (h - LEGEND_HEADING_IN) * fig.dpi + 0.5)
        if all(box.width <= room[0] and box.height <= room[1] for box in boxes) or font <= LEGEND_MIN_FONT_PT:
            return font
        for artist in artists:
            artist.remove()
        font -= 0.25


def draw_long_period_combined(
    analysis, row=None, free_elements=None, stamp: Optional[str] = None, layout=None, options=None, settings=None
):
    """The V1 figure of a secular or Lidov-Kozai verdict (`LongPeriodAnalysis` with its arrays); `row` adds columns to
    the analysis' summary row (the body's free elements, e-i coherence); `free_elements` (ZLK) draws (g) and (j), and
    (a′) under (a) for an induced libration (`forcing_panel_drawn`; nine time panels then, eight otherwise). The bottom
    row holds four squares, (i)-(l) (`_draw_portraits`), the legend the band under it (`long_period_layout`); a `layout`
    with fewer than four bottom squares (LAYOUT_V1, the MMR figure's) gets them and the band (`long_period_layout` on it).
    The right column follows the plot option `combined.numbers` (explained | all); `settings`: the AnalysisSettings the
    explanation compares with (default: from the configuration). Returns the Figure (with `text_stats`).
    `combined.figure='panels'` (a paper figure, as `mmr_combined.draw_combined`): the time panels and the portraits
    only, no header, legend (nor its band) or right column (`text_stats` empty), a square without data left out and
    the next ones moved left and re-lettered, the page cropped (`mmr_combined.panels_layout`); a `stamp` is still
    printed."""
    import matplotlib.pyplot as plt

    from resonances.plotting import combined_numbers as numbers
    from resonances.plotting.mmr_combined import _colour_bar, label_time_axes, panels_layout, panels_stamp, stacked_axes

    summary = {**analysis.summary_row(), **(row or {})}
    resolved = resolve_plot_options(options)
    panels = resolved['combined']['figure'] == FigureMode.PANELS
    if layout is None or len(layout.bottom) < len(LONG_PERIOD_BOTTOM):
        # an explicit layout of the MMR figure (three squares, the legend in a slot: LAYOUT_V1) has no room for (i)-(l):
        # the secular/ZLK row and legend band are put on its page
        layout = long_period_layout(len(LONG_PERIOD_PANELS), legend=not panels, base=layout)
    explained = resolved['combined']['numbers'] != NumbersMode.ALL
    if panels:
        layout = panels_layout(layout)
    elif explained:  # the column is measured first: the figure grows down when it needs (decision D8)
        from resonances.plotting.verdict_column_draw import prepare
        from resonances.plotting.verdict_column_long import figure_column

        layout, prepared = prepare(figure_column(analysis, row, free_elements, settings), layout)
    fig = plt.figure(figsize=layout.size, dpi=layout.dpi)
    if not panels:
        _draw_header(fig, analysis, layout, stamp)
    else:
        panels_stamp(fig, layout, stamp)
    frame = _LongFrame(analysis, settings)
    width, height = fig.get_size_inches()
    forcing = analysis.series is not None and len(analysis.series.t_yr) > 1 and forcing_panel_drawn(analysis, free_elements)
    wrapped = 1 + int(forcing)  # (b) the wrapped angle, after (a′) when drawn
    axes = stacked_axes(fig, layout, len(LONG_PERIOD_PANELS) + int(forcing), labelled=wrapped)
    if analysis.series is not None and len(analysis.series.t_yr) > 1:
        _draw_angle(axes[0], frame, layout)
        if forcing:
            _draw_forcing(axes[1], frame, layout, free_elements)
        rest = axes[1 + int(forcing) :]
        _draw_wrapped(rest[0], frame, layout)
        _draw_rate(rest[1], frame, layout)
        _draw_axis(rest[2], frame, layout)
        _draw_element(rest[3], frame, layout, analysis.series.e, ECC_LETTER, 'e', '{:.4g}')
        _draw_element(rest[4], frame, layout, np.degrees(analysis.series.inc), INC_LETTER, _inclination_label(analysis.series), '{:.4g}°')
        _draw_physics(rest[5], frame, layout, free_elements)
        _draw_centre(rest[6], frame, layout)
        label_time_axes(axes, wrapped, layout, frame.unit)
        axes[-1].set_xlim(frame.x[0], frame.x[-1])
        norm = Normalize(frame.x[0], frame.x[-1])
        _draw_portraits(fig, frame, layout, norm, free_elements, skip_empty=panels)
        _colour_bar(fig, norm, layout, frame.unit, label_on_top=True)  # (j)'s y label sits right of it (W23 F4)
    if panels:
        fig.text_stats = {}
        return fig
    from resonances.plotting.mmr_combined import policy_marks

    _draw_legend(fig, layout, forcing, policy_marks(analysis) is not None)
    fig.add_artist(plt.Line2D([layout.separator_x / width] * 2, [0.3 / height, (height - 0.2) / height], color='#dddddd', lw=0.8))
    if not explained:
        constants = [replace(line, block='constants') for line in long_period_frequency_lines(analysis, summary, free_elements)]
        blocks = long_period_blocks(summary, analysis.verdict.decision_path, analysis.episode_rows(), constants)
        fig.text_stats = numbers.draw_text(fig, blocks, layout, LONG_PERIOD_BLOCK_ORDER)
        return fig
    from resonances.plotting.verdict_column_draw import draw_prepared

    fig.text_stats = draw_prepared(fig, prepared, layout)
    return fig


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


@dataclass(frozen=True)
class Frequency:
    """One row of FREQUENCIES: its name ('g5 Jup', 'g − g6', 'σ'), the rate [arcsec/yr], the period [yr] (360°/|rate|
    unless given: a beat's own), a note, and how the rate prints (`constant`: 4 decimals; `rate_text`: instead)."""

    name: str
    rate: float
    period: float
    note: str = ''
    constant: bool = False
    rate_text: Optional[str] = None


def _frequency_line(name: str, rate: float, note: str = '', period: Optional[float] = None, constant=False, rate_text=None):
    """One row of FREQUENCIES (`Frequency`); the period is 360°/|rate| unless given."""
    if period is None:
        period = ARCSEC_PER_TURN / abs(rate) if np.isfinite(rate) and rate else float('nan')
    return Frequency(name, rate, period, note, constant, rate_text)


def frequency_number_line(item: Frequency):
    """The printed row 'name  rate  period  note' (monospaced, aligned, kept in one column)."""
    from resonances.plotting.combined_numbers import NumberLine
    from resonances.plotting.verdict_text import years

    rate_text = item.rate_text if item.rate_text is not None else _rate_text(item.rate, item.constant)
    text = f'{item.name:<8}{rate_text:>9} {years(item.period):>9}' + (f'  {item.note}' if item.note else '')
    return NumberLine('numbers', '', text, group=FREQUENCY_GROUP)


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
        note = (f'{amplitude:.3g}' if np.isfinite(amplitude) else ('' if label in basis else 'not fitted')) + (
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


def long_period_frequencies(analysis, row, free_elements=None) -> list:
    """The rows of FREQUENCIES (`Frequency`): every frequency a period seen on sigma(t) may come from; `row` is the
    figure's summary row, `free_elements` (ZLK) the body's FreeElements (the fitted modes and their |c_j|)."""
    if analysis.series is None:
        return []
    if str(analysis.series.resonance_type) == 'lidov_kozai':
        return _zlk_frequency_lines(analysis, row, free_elements)
    return _secular_frequency_lines(analysis, row)


def long_period_frequency_lines(analysis, row, free_elements=None, frequencies=None) -> list:
    """FREQUENCIES of a secular/ZLK figure: one row 'name  rate ″/yr  period  note' per frequency a period seen on
    sigma(t) may come from (period = 1296000/|rate| yr; `long_period_frequencies`, or the given `frequencies`)."""
    items = long_period_frequencies(analysis, row, free_elements) if frequencies is None else list(frequencies)
    if not items:
        return []
    from resonances.plotting.combined_numbers import NumberLine

    amplitude = '  |c|' if str(analysis.series.resonance_type) == 'lidov_kozai' else ''
    header = NumberLine('numbers', '', f'{"":<8}{"arcsec/yr":>9} {"period":>9}{amplitude}', group=FREQUENCY_GROUP)
    return [header, *(frequency_number_line(item) for item in items)]
