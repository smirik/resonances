"""The combined MMR figure of the v2 analysis: one page, layout V1 approved by the author ('A').

24 x 13.5 in (16:9), 120 dpi. Left: ten long thin time panels (a)–(j) sharing t, each in its own frame (eleven
when the cycles are too dense for (b): the strip (b′) under it shows a few of them on their own time axis); below
them the portrait of the whole record (k) with the time color bar, the portrait of the accepted episodes (l) over
their fill spans of (a) (`portrait_span`), FAIR (m, two-body angles) and the legend (in the third slot too when there
is no FAIR, then under one warning line with the reason, `fair_warning`). Right (`draw_right_column`): the ten
blocks of `verdict_column` (header with the status, printed nowhere else, subtitle, key facts, why, confidence,
checks, evidence, caveats, numbers, footer; `verdict_column_draw.draw_prepared`), or every number of the decision
(plot option `combined.numbers='all'`; `combined.figure='panels'`: the left panels only, for a paper). The layout is
the same for every object (figures of one status folder are leafed through); a column that does not fit grows the
figure down, at most `style.COLUMN_MAX_GROWTH_IN`, the left panels keeping their place at the top (`prepare_column`,
decision D8).

The panels carry no explanatory text (the author, 2026-10-16: what a line means is in the legend and docs/plots.md):
lines are not labeled, a level outside a panel's axis is a small triangle at the edge (▲ above, ▼ below), and the
only numbers written are data (min – max of e and i, R² of the force law, keys of the multiplet's members). The
numbers of the reference lines of (c), (d), (h) stand right of them, outside the axes (`margin_legend`). An accepted
episode is filled to a record edge it covers, else P/4 beyond its turns (`episode_spans`).

(a) holds the whole gray band of the osculating sigma and (d) that of the osculating a (`style.ENVELOPE_PERCENTILES`,
the rare samples beyond are triangles at the edge, `outlier_markers`). The filtered lines of (c), (d), (e) cover the
whole record: within FILTER_EDGE_TRIM_CUTOFFS cutoffs of its ends (outside the analysis window, the filter's edge
spread) they are pale and dashed (`style.EDGE_SPREAD_STYLE`, `_edge_line`); the view is fitted to the analyzed part,
an edge transient beyond it is a triangle at the edge (`edge_overflow_markers`).

(d) draws a on an absolute axis (au): the osculating a (light gray), a_f (mid gray), the slow a at the resonance's own
scale (black, `strong_axis_cutoff`), and a* and the catalog a_res moved into the
osculating frame of a_f by <a> - a(lambda_dot). a* and a_res belong to the mean motion, a_f to the osculating a, and
the constant offset between them (up to 0.08 au for the plutinos) would put a libration outside its own resonance.
The size of the shift is a number of the right column (ctx_a_mean_au - ctx_a_from_mean_longitude_rate_au).

(i) is the center of sigma: sigma_f low-passed over W_c = libration_centre_cycles x P_cyc (the analysis'
`geometry.center`, column `geo_centre_window_yr`) drawn over sigma_f itself, with the center's turns and its
largest monotone pass when it is a revolution or more. A center that still oscillates there oscillates with a period longer than W_c: for
the Hildas (153) that is the forced wobble at the period of varpi - varpi_J (2.6 kyr), not a smoothing error.
(j) shows where that wobble comes from: the centers of the other members of the multiplet,
c_s = c0 + smooth_Wc(Delta_s) with Delta_s = sigma_s - sigma0 from the saved varpi of the body and the
planets (planar members of the same order), the accepted subresonance's own measured center and the
owner's center; each minus its circular mean. A member whose line is flat is the one that librates.
When a subresonance or an owner is accepted, its angle (filtered at the cutoff) is also drawn on (b).
W28 B10b (cards g08-8 'if the cause is a secular cycle, show it as a line, here and everywhere', g08-7, g05-2, g05-3):
(i) hatches the slow scale's intervals ('slow #id', `slow_intervals`); (j) adds the detrended p varpi of the slow
scale, labeled with p and the secular period from the columns, whenever the slow cause is secular
(`varpi_overlay`); for a kinematic libration (or a non_resonant row whose slow scale is a secular cycle) (j) is
sigma = lambda part + p varpi on the scale of that swing (`kinematic_scale`: the decisive episode at its response
cutoff, or the slow scale), in degrees.
(c) draws sigma_dot / 2 omega0 on a symlog axis (linear inside +-1, the speed at the separatrix) and +-2 omega0 over
each accepted episode with that episode's omega0; without an episode the theoretical omega0, thin. sigma_dot = 0 is a
solid line in the color of the center (`style.RATE_ZERO`): a libration oscillates about it, sigma_dot changing sign at
each turning point; in the portraits (k), (l) it is the level of the centers (sigma_c, 0), marked + per accepted
episode. (h) is on a log axis, so that h of a deep libration and of a fast circulation are both in view.

The members' and the owner's centers, the strip (b′) and the per-regime nu of (c) are drawing aids; every
number printed in the text column is taken from `analysis.summary_row()` / `numbers_row()` by its CSV column,
so the CSV and the figure agree. Nothing drawn here feeds a status.
Colors: one palette (`style`): a quantity has the same color in every panel. The text side lives in
`mmr_verdict_panel`, the geometry in `style`.
"""

from __future__ import annotations

import itertools
import textwrap
from dataclasses import dataclass, replace
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from matplotlib import patheffects
from matplotlib.ticker import FuncFormatter, NullLocator, ScalarFormatter

from resonances.plotting import verdict_text
from resonances.plotting.base import time_unit
from resonances.plotting.mmr_verdict_panel import (  # noqa: F401  (re-exported: the API of this module)
    TEXT_BLOCKS,
    NumberLine,
    TextBlock,
    draw_text,
    format_cell,
    text_blocks,
)
from resonances.plotting.style import (  # noqa: F401
    AXIS_EXACT,
    AXIS_NOMINAL,
    CENTRE,
    EDGE_SPREAD_STYLE,
    ENVELOPE_PERCENTILES,
    EPISODE_ALPHA,
    EPISODE_BEFORE_JUMP_HATCH,
    EPISODE_COLOURS,
    FRAME,
    INK,
    LAYOUT_V1,
    MARK_COLOURS,
    MEMBER_COLOURS,
    MODEL,
    NOTE,
    OWNER,
    RAW,
    RAW_ALPHA,
    RAW_LIGHT_ALPHA,
    RATE_ZERO,
    REFERENCE,
    REJECTED,
    REJECTED_ALPHA,
    STATUS_COLOURS,
    SUBRESONANCE,
    ZONE_FILL,
    FigureMode,
    NumbersMode,
    OsculatingAxis,
    Rect,
    VerdictFigureLayout,
    resolve_plot_options,
    time_cmap,
)
from resonances.resonance import oscillations as osc
from resonances.data.const import GAUSSIAN_YEAR_DAYS
from resonances.resonance.analysis.filtering import circular_mean, is_finite, lowpass, lowpass_or_none
from resonances.resonance.analysis.geometry import MARK_AMPLITUDE_CHANGE, RejectReason, wrap_pi
from resonances.resonance.analysis.geometry import well_shift
from resonances.resonance.analysis.mmr import response as response_module
from resonances.resonance.analysis.mmr.decide import taken_direct_law, taken_harmonic_law
from resonances.resonance.analysis.mmr.models import MMRAnalysis, MMRSeries, ResonanceKind, ResonanceSpec, SlowOrigin
from resonances.resonance.analysis.mmr.sampling import body_varpi_slow_unwrap
from resonances.resonance.analysis.mmr.preprocess import coorbital_planet
from resonances.resonance.analysis.mmr.owner import FLAG_CATALOGUE, wrapped_angle as owner_wrapped_angle
from resonances.resonance.analysis.orbit_type import accepted_runs
from resonances.resonance.analysis.verdict import AngleSubtype, LibrationOrigin

TWO_PI = 2 * np.pi
PER_KYR = 1e3  # rad/yr -> rad/kyr
LINE_POINTS = 20000  # samples per drawn line; longer series are decimated (never a wrapped angle: see CYCLE_MIN_PX)
ENVELOPE_BINS = 1600  # raw series denser than this become a min-max band
MAX_TP_MARKERS = 80  # more turning points than this are not marked one by one
MAX_CYCLE_MEANS = 200
MAX_MEMBER_LINES = 6  # members of the multiplet drawn in (j) besides sigma0 (the flattest, or the best ranked)
CENTRES_AXIS_DEG = 230  # (j): the axis, so that the ticks +-180 stay clear of the tick margin
VIEW_PAD = 0.12  # fraction of the data range added above and below
FAIR_STEP_FRACTION = 1 / 4.2  # FAIR resolves its strips only with a step <= P_orb / 4.2 (= DRYRUN_FAIR_FRACTION)
DEFAULT_FILTER_ORDER = 2  # FILTER_ORDER, when the configuration cannot be read
LEGEND_CHAR_WIDTH = 0.56  # mean advance of a DejaVu Sans character, in units of the font size (wrapping estimate)
LEGEND_MIN_FONT_PT = 6.0  # ~10 px at 120 dpi, the smallest legible size: keep LEGEND short enough to fit at it
LEGEND_WARNING_GAP = 0.08  # inches between the FAIR warning line and the legend's heading
FAIR_WARNING_COLOUR = '#b35900'  # the warning line of a figure without FAIR (dark orange, readable on white)
# Drawing rules (pixels of a time panel at the layout's dpi)
CYCLE_MIN_PX = 3.0  # sigma_f is not drawn in (b) when a cycle is narrower: the line would be a solid block
STRIP_CYCLE_PX = CYCLE_MIN_PX  # below this the strip (b′) under (b) shows ZOOM_CYCLES cycles on their own time axis
ZOOM_CYCLES = 5
ZOOM_MIN_STEPS = 8  # a window of the strip spans at least this many output steps
ZOOM_SUBSTEPS = 24  # points per output step of the predicted path drawn in the strip
STRIP_GAP = 0.5  # inches between the two windows of the strip (the y tick labels of the second)
STRIP_MARK_PT = 6.0  # the ▼ over (b) at the middle of a window of the strip [pt]
STRIP_MARK_GAP_PT = 3.6  # the window's number starts this far from its ▼'s middle (6 px at 120 dpi) [pt]
H_FLOOR = 1e-3  # (h): the log axis stops here; h below it (or h <= 0) is a triangle at the bottom edge
H_VIEW_FACTOR = 3.0  # (h): the axis spans the values / this factor .. the values x this factor (log)
RATE_SYMLOG_VIEW = 2.2  # (c) on the symlog axis holds at least +-this (2 omega0): +-1 then clears the tick margin
RATE_SYMLOG_PAD = 10**0.5  # (c): the symlog axis extends this factor beyond the data outside +-1
SYMLOG_LABEL_GAP = 1.2  # (c): tick labels of the symlog axis at least this many label heights apart
EPISODE_LABEL_MIN_PX = 30  # '#id' above an episode only when its fill is at least this wide
EPISODE_LABEL_PAD = 0.03  # '#id' sits this fraction of the panel's height above its top edge (clear of the data)
REJECT_LETTER_MIN_PX = 8
REJECTED_STRIP = 0.14  # height of the rejected-runs strip, fraction of the panel
REJECT_LETTERS = {
    RejectReason.OVER_THE_TOP: 'T',
    RejectReason.TOO_FEW_TP: 'F',
    RejectReason.CENTRE_MOVES: 'C',
    RejectReason.NO_CLOSED_CYCLE: 'O',
}
JUMP_MERGE_FRACTION = 0.01  # jumps of a closer than this share of the record are one band
JUMP_LABEL_PAD_PT = 0.5  # white box around an 'N jumps of a' label [pt]
PANEL_LETTER_X = 0.004  # the panel letter's left edge, axes fraction
VALUE_TEXT_X = 0.04  # axes fraction where `_value_text` starts
WINDOW_LABELS_MAX = 12  # (e): the R² of each force window is written when there are at most this many
WINDOW_LABEL_GAP_PT = 2.0  # (e): least free space between two window labels; closer ones merge [pt]
TP_SIZE = 18  # marker area (pt^2) of the turning points of the angle
CENTRE_TURN_SIZE = 36  # of the turns of the centre
RATE_DATA_MIN_SHARE = 0.25  # (c) without omega0 is not stretched to 0 if sigma_dot would fill less of the height
AXIS_VIEW_PAD = 0.25  # (d): margin around the a_f range
AXIS_REFERENCE_SPANS = 3.0  # (d): a* and a_res are brought into view when this close to a_f (in a_f ranges)
AXIS_LINE_WIDTH = 1.2  # (d): a* (dotted)
AXIS_NOMINAL_WIDTH = 1.3  # (d): a_res (dashed), the level the author reads first (23.10; thinner, variant B 24.10)
AXIS_LEVEL_ALPHA = 0.8  # (d): a* and a_res with their halos, slightly transparent over a_f (variant B, author 24.10/25.10)
AXIS_NOMINAL_HALO_WIDTH = 0.8  # (d): a_res's white halo (variant B, author 24.10)
AXIS_LEVEL_ZORDER = 8  # (d): a* and a_res over every line and dot of the panel (author, 23.10)
AXIS_HALO_WIDTH = 1.6  # (d): a*'s white halo, so it stays visible on a dense black a_f
# (d): (line width, halo beyond it) of each level of `axis_levels`
AXIS_LEVEL_STYLE = {'a*': (AXIS_LINE_WIDTH, AXIS_HALO_WIDTH), 'a_res': (AXIS_NOMINAL_WIDTH, AXIS_NOMINAL_HALO_WIDTH)}
EDGE_MARKER_PT = 4.5  # a level outside the axis: a triangle of this size just inside the edge
AXIS_EDGE_AT = (0.48, 0.52)  # (d): where in a regime the edge triangles of a* and a_res sit (apart when both are off)
EDGE_X_RAW = 0.993  # (d): x (axes fraction) of the edge triangles of an osculating a beyond the axis
ANGLE_VIEW_PAD = 0.1  # (a): margin around sigma_f and the osculating band, fraction of their span
ENVELOPE_VIEW_PAD = 0.04  # (d): margin around the osculating band (ENVELOPE_PERCENTILES), fraction of its span
OUTLIER_MARKERS_MAX = 8  # per edge: the samples beyond the band are marked by groups, the largest excess first
OUTLIER_MERGE_FRACTION = 0.02  # clipped samples closer than this share of the record are one group (one triangle)
FORCED_WAVE_LABEL = ' wave'  # (a): after '#id' of an accepted run that is a forced wave (F6), no episode of the angle
# (a), (l): the other marks of accepted runs that are no episodes of the angle (W28 B2, B6b; `ForcedWave`): a 5s row's
# runs, named by the ground (its own force cannot make that period, not a driver found: card g11-1 A), and an
# unconfirmed ripple, an auxiliary interval on an unfinished slow swing (card g08-4 A), its fill dotted
STRENGTH_WAVE_LABEL = ' forced (strength)'
RIPPLE_LABEL = ' ripple'
RIPPLE_HATCH = '...'
# (a): the fill of an accepted episode another resonance induces (A1, episodes.csv libration_origin induced; W28 A3b)
# is hatched the other way from EPISODE_BEFORE_JUMP_HATCH ('////'), so both read when they meet ('xx'); (l) labels it
INDUCED_HATCH = '\\\\\\\\'
INDUCED_LABEL = ' induced'
# (a), (d), 1:1 (W28 C13, cards g03-1 A, g03-3 C, g03-8 A, g03-9 A; what C7/C8 store): the inferred residence around an
# accepted run's core (`geometry.Residence`) beyond its fill, lighter and hatched '--', labelled 'inferred' (a stay
# without measured turning points: it decides nothing, and (l) never draws it); the sub-regimes of a run
# (`Episode.subregimes`) as brackets just over (a), named by their well (a repeated well, after an amplitude change,
# unnamed: the gap between two brackets is the transitional half-cycle); the kicks of a at such a change
# (`Episode.marks`, kick@t) as ▼ at the top of (d)
RESIDENCE_HATCH = '--'
RESIDENCE_ALPHA_SCALE = 0.5  # its fill: EPISODE_ALPHA x this
RESIDENCE_LABEL = 'inferred'
BRACKET_RISE_PT = 3.0  # (a): the bracket's line this far over the top edge [pt]
BRACKET_TICK_PT = 1.0  # its end ticks reach down to this height over the edge [pt]
BRACKET_LABEL_PT = 4.0  # the names (and '#id' of a run with brackets) start this far over the edge [pt]
BRACKET_GAP_PX = 2.0  # left free between the ticks of two neighbouring brackets [px]
BRACKET_GID, RESIDENCE_GID, KICK_GID = 'subregime_bracket', 'residence', 'kick'
# (a) under the minimal policy (W29 SPEC §7, PRINCIPLES 6): the counted episodes are filled and numbered as before (a
# covered edge filled to the record's edge, the stretch the filter's trim hides pale), a counted 1:1 leg from the
# policy's spans labelled 'leg'; a candidate (a vetoed episode that survives the filter, a leg or half-swing that is not
# counted, a near-episode run) faint with a dashed outline, labelled 'cand', not numbered; every other run is not drawn
# (it stays in episodes.csv). The gids name the artists for the tests.
COUNTED_GID, TRIM_GID, CANDIDATE_GID = 'counted_episode', 'counted_trim', 'candidate'
CANDIDATE_ALPHA_SCALE = 0.25  # a candidate's fill: EPISODE_ALPHA x this (faint: PRINCIPLES 6)
CANDIDATE_EDGE_ALPHA = 0.7  # its dashed outline
TRIM_ALPHA_SCALE = 0.4  # the filter-trim stretch of a counted episode filled to a covered record edge: EPISODE_ALPHA x this
CANDIDATE_LABEL = 'cand'
LEG_LABEL = 'leg'
NEW_AMPLITUDE_LABEL = 'new amplitude'  # (a): the bracket of a sub-regime in the same well after an amplitude change
KICK_LABEL = 'kick'
KICK_MARK_PT = 5.0  # (d): the ▼ of a kick [pt]
PORTRAIT_MAX_CYCLES = 12  # (l): cycles per episode, evenly spaced; an episode with no more is drawn whole
PORTRAIT_MAX_EPISODES = 5  # (l): the longest episodes only
PORTRAIT_CORE_STYLE = {'lw': 0.7, 'alpha': 0.85}  # (l): an episode between its first and last turning point
PORTRAIT_EXTENSION_STYLE = {'ls': (0, (2.5, 1.5)), 'lw': 0.6, 'alpha': 0.6}  # (l): its fill beyond them (W28 g03-6)
PORTRAIT_SHARE_MAX = 3.0  # (k) and (l) share the union of their y views unless (k)'s is wider than this times the episodes'
PORTRAIT_VIEW_PAD = 0.08  # (k), (l): margin around the samples' sigma_dot, fraction of their span
MARGIN_FONT_DELTA = 0.5  # the compact legend right of the panels: font_note_pt minus this
MARGIN_LINE_SPACING = 1.2  # its line height, in units of its font size
MARGIN_KEY_IN = 0.16  # width of its colour keys (inches)
MARGIN_KEY_GAP_IN = 0.04  # between a key and its text (inches)
AXIS_STRONG_MIN_RATIO = 1.5  # (d): the black a is drawn only at a cutoff at least this many times P_c (else a_f is black)
AXIS_DENSE_CYCLE_PX = 12.0  # (d): a_f with cycles narrower than this is a band: the slow a is drawn black over it
AXIS_STRONG_CYCLES = 2.0  # (d): the black a at this many periods of the deciding episode when the response cutoff is P_c

# (i): the slow scale's intervals (W28 B10b, card g08-7), hatched in the centre's colour, labelled 'slow #id'
SLOW_LABEL = 'slow'
SLOW_HATCH = '|'
SLOW_HATCH_ALPHA = 0.35
# (j) kinematic mode: the lambda part sigma - p varpi (W28 B10b, card g05-3), grey dotted beside red sigma, orange pϖ
LAMBDA_PART = REFERENCE
LAMBDA_PART_LABEL = 'λ part'
NAN = float('nan')

TYPE_LABELS = {ResonanceKind.TWO_BODY: 'mmr2', ResonanceKind.THREE_BODY: 'mmr3', ResonanceKind.RETROGRADE: 'mmr2 retrograde'}

# Time panels, top to bottom: (letter, what it shows) — SPEC §10.2, the approved mock-up V1 and the author's P1-P15
PANELS = (
    ('a', 'sigma unwrapped: raw, sigma_f, center, accepted episodes (#id), rejected runs (reason), TPs, jumps, exit'),
    ('b', 'sigma in a 360-degree window around the center: every osculating sample, sigma_f, center; owner/subresonance'),
    ('c', 'sigma_dot / 2 omega0 (symlog, linear inside +-1) with +-1 (separatrix) of each episode and nu(lambda_dot); slips'),
    ('d', 'a (au): raw, a_f, a* and a_res moved by <a> - a(lambda_dot), band +- half-width, cycle means'),
    (
        'e',
        'a_dot as the force law read it (the raised cutoff of the line rule too) with the law that decided on sigma - '
        'c(t) at that cutoff: R sin(psi - phi0) + C per window (R^2) or the multiplet law; per episode half R^2; under '
        'the harmonic or 1:1 law its numbers, no curve; under the direct law (B8) a_dot and s x the averaged force of '
        'the term, both low-passed at f P_large, per window',
    ),
    ('f', 'e: raw band, filtered line, min - max'),
    ('g', 'i: raw band, filtered line, min - max'),
    ('h', 'h per cycle on a log axis (libration dots, revolutions open) with the separatrix zone [h_lo, h_hi] and h = 1'),
    ('i', 'the center of sigma (sigma_f smoothed over W_c) over sigma_f, its turns, largest monotone pass, slow intervals'),
    (
        'j',
        'centers of sigma0, the members of the multiplet and the owner (minus their means), p varpi of a secular slow '
        'cause; or the kinematic terms sigma = lambda part + p varpi',
    ),
)
# (b′) under (b) only when the cycles are narrower than STRIP_CYCLE_PX: a few cycles on their own time axis (yr)
# Below the panels (`style.VerdictFigureLayout.bottom`): (k) the whole record's portrait and the time colour bar,
# (l) the accepted episodes' portrait, (m) FAIR, the legend (in the slot of (m) too when there is no FAIR, under the
# warning line of `fair_warning`)

LEGEND = (
    'Gray osculating, black filtered (pale: edge); ▲▼: off axis; (c), (d), (h) values right.',
    '(a) red: center; ▼▲ turns; fill #id: episode (to an edge it covers, else ±P/4 past its turns; / before the last '
    'jump of a, \\ induced; #id wave / forced (strength) / ripple (dotted): not σ’s); gray strip: rejected run (T over '
    'the top, F few TPs, C center moves, O no closed cycle; gray #id < floor: not counted); dotted/hatched: '
    'a jumps; × exit; 1:1: ⊓ sub-regime, hatched – inferred stay.',
    '(b) σ mod 360°; purple dashed: owner, orange dash-dot: subresonance; ▼: window of (b′).',
    '(c) dσ/dt / 2ω₀, symlog; red σ̇ = 0; blue dashed ±2ω₀ (thin: theory); dotted ν(λ̇); ○ slips.',
    '(d) a: light gray osculating, mid gray a_f, black slow a; dotted a*, cyan dashed a_res; blue band ±Δa; red dots: '
    'cycle means, ▼ kick.',
    '(e) da/dt; blue: deciding force law (direct: black da/dt, blue s × mean force); #id R² halves.',
    '(f, g) e, i.  (h) h per cycle: ● libration, ○ revolution; dashed h = 1; blue: separatrix.',
    '(i) red: center of σ_f, ▼▲ turns; shaded: monotone pass ≥ 1 rev; | slow #id: slow scale.  (j) centers − means: '
    'red σ, pale members, orange subresonance, purple owner; dashed orange pϖ (kinematic: dotted λ part).',
    # (k) and (l) share their scales (docs/reading-figures.md); the narrow box (FAIR drawn) has no room left to say so
    '(k) whole record, color: time; (l) episodes, ≤ 12 cycles, dashed past turns; +: center.',
)
# the line of (a) under the minimal policy (W29 SPEC §7): only counted episodes are filled, candidates faint
LEGEND_POLICY_A = (
    '(a) red: center; ▼▲ turns; fill #id: counted episode (to an edge it covers, pale in the filter trim, else ±P/4 past '
    'its turns; / before the last jump of a, \\ induced; #id wave / ripple: not σ’s); leg: a counted 1:1 leg; faint '
    'dashed cand: candidate, not counted; other runs not drawn; dotted/hatched: a jumps; × exit; 1:1: ⊓ sub-regime, '
    'hatched – inferred stay.'
)


def legend_lines(analysis) -> Tuple[str, ...]:
    """LEGEND, with the line of (a) of the minimal policy when the row is under it (`policy_marks`)."""
    if policy_marks(analysis) is None:
        return LEGEND
    return LEGEND[:1] + (LEGEND_POLICY_A,) + LEGEND[2:]


# --------------------------------------------------------------------------------------------- helpers


def _stride(n: int, target: int = LINE_POINTS) -> int:
    return max(1, n // target)


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _circular_mean(angles) -> float:
    """`filtering.circular_mean`, 0 for no finite angle (a window to draw in)."""
    mean = circular_mean(angles)
    return mean if np.isfinite(mean) else 0.0


def _window_low(centre) -> float:
    """Lower edge of the 360-degree window centered on the circular mean of the center."""
    return _circular_mean(centre) - np.pi


def _right_angles(base_deg: float, inner: bool = False) -> np.ndarray:
    """Multiples of 90 deg inside the window [base, base + 360] (without its edges when `inner`)."""
    ticks = 90 * np.arange(np.ceil(base_deg / 90), np.floor((base_deg + 360) / 90) + 1)
    return ticks[(ticks > base_deg) & (ticks < base_deg + 360)] if inner else ticks


def angle_label(value_deg: float) -> str:
    """A wrapped angle's tick label in [0, 360) (the author's convention: 270, not -90)."""
    value = round(value_deg) % 360
    return f'{value:.0f}'


def _angle_formatter():
    return FuncFormatter(lambda value, _: angle_label(value))


def _wrap_window(values, low: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) - low) % TWO_PI + low


def _break_jumps(y: np.ndarray, limit: float) -> np.ndarray:
    """NaN where a wrapped line would jump across the window, so no vertical line is drawn."""
    y = np.array(y, dtype=float)
    if len(y) > 1:
        y[1:][np.abs(np.diff(y)) > limit] = np.nan
    return y


def _envelope(t: np.ndarray, y: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(t[0], t[-1], bins + 1)
    index = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, bins - 1)
    low = np.full(bins, np.nan)
    high = np.full(bins, np.nan)
    np.fmin.at(low, index, y)
    np.fmax.at(high, index, y)
    return 0.5 * (edges[:-1] + edges[1:]), low, high


def _raw(ax, t, y, color=RAW, alpha=RAW_ALPHA):
    """A raw series: a thin line when sparse, a min-max band when denser than the panel's pixels."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if len(t) <= ENVELOPE_BINS:
        ax.plot(t, y, color=color, lw=0.5, alpha=alpha, zorder=1)
        return
    centres, low, high = _envelope(t, y, ENVELOPE_BINS)
    ax.fill_between(centres, low, high, color=color, alpha=alpha, lw=0, zorder=1)


def _line(ax, t, y, unwrapped=None, max_step: float = np.pi, **kwargs):
    """A decimated line; a gap (NaN) anywhere in a decimation block stays a gap (no wrap-around strokes).

    With `unwrapped` (the same angle unwrapped, rad) the line also breaks where the angle turned by more than
    `max_step` between two drawn points: a wrapped line there joins samples a revolution apart."""
    t, y = np.asarray(t), np.asarray(y, dtype=float)
    k = _stride(len(t))
    shown = y[::k].copy()
    if k > 1:
        gaps = np.add.reduceat(np.isnan(y).astype(int), np.arange(0, len(y), k)) > 0
        shown[gaps] = np.nan
    if unwrapped is not None and len(shown) > 1:
        steps = np.abs(np.diff(np.asarray(unwrapped, dtype=float)[::k]))
        shown[1:][~(steps <= max_step)] = np.nan
    ax.plot(t[::k], shown, **kwargs)


def _view(ax, y, pad: float = VIEW_PAD, extra_top: float = 0.0):
    """y limits from the finite data (0.2-99.8 percentiles), padded."""
    limits = _view_limits(y, pad, extra_top)
    if limits is not None:
        ax.set_ylim(*limits)


def _view_limits(y, pad: float = VIEW_PAD, extra_top: float = 0.0) -> Optional[Tuple[float, float]]:
    y = _finite(y)
    if not len(y):
        return None
    low, high = np.percentile(y, [0.2, 99.8])
    span = high - low or max(abs(high), 1.0) * 1e-6
    return low - pad * span, high + (pad + extra_top) * span


def envelope_range(values) -> Optional[Tuple[float, float]]:
    """The ENVELOPE_PERCENTILES band of the finite `values`: what a panel shows of an osculating series; None when
    nothing is finite."""
    values = _finite(values)
    if not len(values):
        return None
    low, high = np.percentile(values, ENVELOPE_PERCENTILES)
    return float(low), float(high)


def outlier_markers(ax, x, values, low: float, high: float, colour: str = RAW, limit: int = OUTLIER_MARKERS_MAX) -> int:
    """Samples of `values` beyond the view [low, high]: one triangle at that edge per group of clipped samples (closer
    than OUTLIER_MERGE_FRACTION of the record are one group), at the time of its largest excess (`edge_marker`, data
    x); at most `limit` per edge, the largest excess first. Returns the number of triangles."""
    x, values = np.asarray(x, dtype=float), np.asarray(values, dtype=float)
    gap = OUTLIER_MERGE_FRACTION * (x[-1] - x[0]) if len(x) > 1 else 0.0
    drawn = 0
    for above in (True, False):
        excess = values - high if above else low - values
        index = np.flatnonzero(np.isfinite(excess) & (excess > 0))
        if not len(index):
            continue
        groups = np.split(index, np.flatnonzero(np.diff(x[index]) > gap) + 1)
        peaks = sorted((group[np.argmax(excess[group])] for group in groups), key=lambda i: -excess[i])[:limit]
        for i in peaks:
            edge_marker(ax, float(x[i]), above, colour, data_x=True)
        drawn += len(peaks)
    return drawn


def _edge_split(values, inside) -> Tuple[np.ndarray, np.ndarray]:
    """`values` where `inside` (the analysis window; the rest NaN) and outside it, in the filter's edge spread (the
    rest NaN). The edge part reaches two decimation strides (`_line`) into the window, so that the two drawn lines
    share a point after decimation and join."""
    values = np.asarray(values, dtype=float)
    inside = np.asarray(inside, dtype=bool)
    edge = ~inside
    if inside.any():
        first, last = np.flatnonzero(inside)[[0, -1]]
        overlap = 2 * _stride(len(values))
        if first > 0:
            edge[first : first + overlap + 1] = True
        if last < len(values) - 1:
            edge[max(last - overlap, 0) : last + 1] = True
    return np.where(inside, values, np.nan), np.where(edge, values, np.nan)


def _edge_line(ax, frame, values, inside=None, **kwargs) -> np.ndarray:
    """A filtered line over the whole record: solid inside the analysis window (`inside`, default the frame's), in
    EDGE_SPREAD_STYLE within FILTER_EDGE_TRIM_CUTOFFS cutoffs of the record's ends (the filter's edge spread, which
    the analysis does not read). Returns the values inside the window (what the panel's view is fitted to)."""
    shown, edge = _edge_split(values, frame.inside if inside is None else inside)
    _line(ax, frame.x, shown, **kwargs)
    if np.isfinite(edge).any():
        style = {**kwargs, **EDGE_SPREAD_STYLE, 'alpha': EDGE_SPREAD_STYLE['alpha'] * kwargs.get('alpha', 1.0)}
        _line(ax, frame.x, edge, **style)
    return shown


def edge_overflow_markers(ax, frame, values, inside=None, colour: str = INK) -> int:
    """Triangles at the edge where the edge spread of a filtered line (`_edge_line`) leaves the view: the view is
    fitted to the analyzed window only, so a large end transient would otherwise vanish silently. Call after the
    y limits are set. Returns the number of triangles."""
    _, edge = _edge_split(values, frame.inside if inside is None else inside)
    low, high = sorted(ax.get_ylim())
    return outlier_markers(ax, frame.x, edge, low, high, colour)


def _note(ax, text: str, fontsize: float, y: float = 0.5):
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='center', fontsize=fontsize, color=NOTE)


def edge_marker(ax, x: float, above: bool, colour: str, data_x: bool = False):
    """A level beyond the axis: ▲ just inside the top edge (above) or ▼ inside the bottom edge, at x (axes fraction,
    or data when `data_x`). No words: what the color stands for is in the legend."""
    height_in = ax.get_position().height * ax.figure.get_size_inches()[1]
    inset = 0.6 * EDGE_MARKER_PT / 72 / max(height_in, 1e-6)
    y = 1 - inset if above else inset
    transform = ax.get_xaxis_transform() if data_x else ax.transAxes
    ax.plot([x], [y], marker='^' if above else 'v', ms=EDGE_MARKER_PT, color=colour, transform=transform, clip_on=False, zorder=9)


def margin_legend(ax, layout: VerdictFigureLayout, items) -> list:
    """The compact legend right of a time panel, outside its axes (from `layout.margin_legend_x`), top down from the
    panel's top: per item a color key matching the panel's line — ('line', color, line style), ('band', color,
    alpha) or None — and a short text with the number. Figure artists, so the panel's own texts stay data only.
    Returns the texts."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    fig = ax.figure
    width, height = fig.get_size_inches()
    font = layout.font_note_pt - MARGIN_FONT_DELTA
    step = MARGIN_LINE_SPACING * font / 72 / height
    x0, key = layout.margin_legend_x / width, MARGIN_KEY_IN / width
    y = ax.get_position().y1 - 0.5 * step
    texts = []
    for mark, text in items:
        if mark is not None and mark[0] == 'line':
            fig.add_artist(Line2D([x0, x0 + key], [y, y], color=mark[1], ls=mark[2], lw=1.1, transform=fig.transFigure))
        elif mark is not None:
            fig.add_artist(
                Rectangle((x0, y - 0.3 * step), key, 0.6 * step, facecolor=mark[1], alpha=mark[2], lw=0, transform=fig.transFigure)
            )
        texts.append(fig.text(x0 + key + MARGIN_KEY_GAP_IN / width, y, text, fontsize=font, va='center', ha='left', color='#333333'))
        y -= step
    return texts


def _value_text(ax, text: str, layout: VerdictFigureLayout):
    """A small number of the data (no words) at the top left, after the panel letter."""
    ax.text(VALUE_TEXT_X, 0.94, text, transform=ax.transAxes, ha='left', va='top', fontsize=layout.font_note_pt, color=NOTE, zorder=9)


def _right_labels(ax, items, fontsize: float):
    """Labels at the right end of horizontal lines (value in data units), pushed apart by one text line and
    kept inside the panel's height, so neither they nor the labels of the neighboring panels overlap."""
    if not items:
        return
    low, high = ax.get_ylim()
    height_in = ax.get_position().height * ax.figure.get_size_inches()[1]
    sep = min(1.2 * fontsize / 72 / max(height_in, 1e-6), 1.0 / len(items))
    positions = sorted(((value - low) / (high - low), text, color) for value, text, color in items)
    lowest, highest = 0.5 * sep, 1 - 0.5 * sep
    ys = [min(max(p, lowest), highest) for p, _, _ in positions]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + sep)
    excess = ys[-1] - highest
    if excess > 0:
        ys = [y - excess for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - sep)
        if ys[0] < lowest:  # more labels than fit one line apart: spread evenly
            ys = list(np.linspace(lowest, highest, len(ys)))
    for y, (_, text, color) in zip(ys, positions):
        ax.text(1.004, y, text, transform=ax.transAxes, ha='left', va='center', fontsize=fontsize, color=color, clip_on=False)


def _inside_locator(margin: float, nbins: int = 4):
    """Ticks only inside [low + margin, high - margin] of the view (as fractions of its span): the tick labels of
    stacked panels never meet at a shared edge."""
    from matplotlib.ticker import MaxNLocator

    class InsideLocator(MaxNLocator):
        def tick_values(self, vmin, vmax):
            ticks = super().tick_values(vmin, vmax)
            lo, hi = sorted((vmin, vmax))
            pad = margin * (hi - lo)
            return ticks[(ticks >= lo + pad) & (ticks <= hi - pad)]

    return InsideLocator(nbins=nbins)


def _power_of_ten(values) -> int:
    values = _finite(values)
    peak = np.percentile(np.abs(values), 99.8) if len(values) else 0.0
    return int(np.floor(np.log10(peak))) if peak > 0 else 0


def _multiplier(power: int, unit: str) -> str:
    """'10^{-8} au/yr' in mathtext ('au/yr' alone for 10^0)."""
    return unit if power == 0 else rf'$10^{{{power}}}$ {unit}'


def turn_markers(ax, x, values, kinds, size: float, colour: str, white_edge: bool = False):
    """Turning points: ▼ on a maximum (kind > 0), ▲ on a minimum; one convention for the MMR and secular figures."""
    x, values, kinds = np.asarray(x, dtype=float), np.asarray(values, dtype=float), np.asarray(kinds)
    edge = {'edgecolors': 'white', 'linewidths': 0.5} if white_edge else {'linewidths': 0}
    for kind, marker in ((1, 'v'), (-1, '^')):
        chosen = kinds == kind
        ax.scatter(x[chosen], values[chosen], marker=marker, s=size, color=colour, zorder=6, **edge)


def episode_colours(runs) -> Dict[int, str]:
    """Color of each accepted run (episodes of the last regime and before the last jump), in time order, cycled."""
    accepted = sorted((run for run in runs if run.accepted), key=lambda run: run.t_start_yr)
    return {run.episode_id: EPISODE_COLOURS[i % len(EPISODE_COLOURS)] for i, run in enumerate(accepted)}


def episode_spans(runs, record: Tuple[float, float], settings=None) -> Dict[int, Tuple[float, float]]:
    """(start, end) [yr] of the fill of each accepted run. An edge the run covers by the coverage rule
    (`geometry.covers_edges`, LIBRATION_COVERAGE_PERIODS; read only with `settings`) is filled up to the record's edge;
    otherwise the fill runs from the first to the last turning point extended by half the half-period (P/4) on each
    side, since the libration exists before its first and after its last turn. Never past the record's edges nor past
    the middle of the gap to a neighboring run (accepted or not)."""
    from resonances.resonance.analysis.geometry import covers_edges

    ordered = sorted(runs, key=lambda run: run.t_start_yr)
    spans = {}
    for i, run in enumerate(ordered):
        if not run.accepted:
            continue
        try:
            start_edge, end_edge = covers_edges(run, settings) if settings is not None else (False, False)
        except (AttributeError, TypeError):  # a run without the coverage numbers: turning points only
            start_edge, end_edge = False, False
        reach = 0.25 * run.period_med_yr if is_finite(getattr(run, 'period_med_yr', None)) else 0.0
        low = 0.5 * (ordered[i - 1].t_end_yr + run.t_start_yr) if i > 0 else record[0]
        high = 0.5 * (run.t_end_yr + ordered[i + 1].t_start_yr) if i + 1 < len(ordered) else record[1]
        start = _covered_to(run, record, -1) if start_edge else run.t_start_yr - reach
        end = _covered_to(run, record, +1) if end_edge else run.t_end_yr + reach
        spans[run.episode_id] = (max(start, low, record[0]), min(end, high, record[1]))
    return spans


def _covered_to(run, record: Tuple[float, float], side: int) -> float:
    """How far a covered edge is filled: to the record's edge, unless the covered gap (in the run's periods) ends more
    than a period before it, at the cut that opened the run's window (a confirmed jump of a: 439908 1M-1 20 kyr was
    filled across 13.6 kyr of circulation before its capture)."""
    period = getattr(run, 'period_med_yr', NAN)
    gap = getattr(run, 'coverage_start_gap_p' if side < 0 else 'coverage_end_gap_p', NAN)
    edge = record[0] if side < 0 else record[1]
    if not (is_finite(period) and is_finite(gap)):
        return edge
    reached = run.t_start_yr - gap * period if side < 0 else run.t_end_yr + gap * period
    return edge if abs(reached - edge) <= period else reached


def induced_episode_ids(analysis: MMRAnalysis) -> frozenset:
    """The accepted episodes another resonance induces, as the analysis wrote them (episodes.csv libration_origin
    induced: every episode of an induced row, or those a second pass took out, A1 `analyse._induced_origins`)."""
    if analysis.frozen_episode_rows is not None:
        rows = [(row.get('episode_id'), row.get('libration_origin')) for row in analysis.frozen_episode_rows]
    else:
        rows = [(p.episode_id, p.libration_origin) for p in (analysis.episode_physics or ())]
    ids = set()
    for episode_id, origin in rows:
        try:
            value = float(episode_id)
        except (TypeError, ValueError):
            continue
        if str(origin) == LibrationOrigin.INDUCED and np.isfinite(value):
            ids.add(int(value))
    return frozenset(ids)


def shade_runs(
    ax,
    runs,
    accepted_ids,
    colours: Mapping[int, str],
    times,
    px_per_unit: float,
    layout: VerdictFigureLayout,
    spans: Optional[Mapping[int, Tuple[float, float]]] = None,
    forced_ids: Collection[int] = (),
    induced_ids: Collection[int] = (),
    uncounted: Optional[Mapping[int, str]] = None,
    labelled: Collection[int] = (),
):
    """Accepted episodes as a colored fill (hatched '////' when before the last jump of a, '\\\\' when another
    resonance induces it, `induced_ids`, `INDUCED_HATCH`; dotted for an unconfirmed ripple, `RIPPLE_HATCH`) over their
    `spans` (yr, `episode_spans`; first to last turning point without) with '#id' just above the panel's top edge when
    wide enough (`ax` is the top panel: the label never covers the data or the panel letter; '#id wave' for a forced
    wave, '#id forced (strength)', '#id ripple': `forced_ids`, a collection (' wave') or id -> label (`wave_labels`),
    none an episode of the angle), the other runs as a gray strip at the bottom with the letter of their rejection
    reason (`REJECT_LETTERS`). An accepted run the status does not count (`uncounted`: id -> tag, `uncounted_tags`;
    W28 C11, card g02-9 A: below the period floor, the center turns) is drawn like a rejected run, over its turning
    points, with its gray tag '#0 < floor' above the panel instead of an episode's fill. The runs in `labeled` get no
    '#id' here: `draw_run_labels` writes it with their sub-regimes and residence (W28 C13)."""
    spans = spans or {}
    uncounted = uncounted or {}
    labels = dict(forced_ids) if isinstance(forced_ids, Mapping) else {i: FORCED_WAVE_LABEL for i in forced_ids}
    for run in runs:
        own = (run.t_start_yr, run.t_end_yr)
        start, end = (float(times(v)) for v in (spans.get(run.episode_id, own) if run.episode_id in accepted_ids else own))
        width_px = (end - start) * px_per_unit
        middle = 0.5 * (start + end)
        if run.episode_id in accepted_ids and run.episode_id in uncounted:
            start, end = (float(times(v)) for v in own)
            ax.axvspan(start, end, ymin=0, ymax=REJECTED_STRIP, facecolor=REJECTED, alpha=REJECTED_ALPHA, lw=0, zorder=0)
            if (end - start) * px_per_unit >= REJECT_LETTER_MIN_PX:
                ax.text(
                    0.5 * (start + end),
                    1 + EPISODE_LABEL_PAD,
                    f'#{run.episode_id}{uncounted[run.episode_id]}',
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='bottom',
                    fontsize=layout.font_note_pt,
                    color=REJECTED,
                    clip_on=False,
                    zorder=8,
                )
        elif run.episode_id in accepted_ids:
            colour = colours.get(run.episode_id, EPISODE_COLOURS[0])
            hatches = [EPISODE_BEFORE_JUMP_HATCH] if getattr(run, 'before_last_jump', False) else []
            hatches += [INDUCED_HATCH] if run.episode_id in induced_ids else []
            hatches += [RIPPLE_HATCH] if labels.get(run.episode_id) == RIPPLE_LABEL else []
            ax.axvspan(start, end, facecolor=colour, alpha=EPISODE_ALPHA, lw=0, zorder=0)
            for hatch in hatches:
                ax.axvspan(start, end, facecolor='none', edgecolor=colour, hatch=hatch, alpha=0.5, lw=0, zorder=0)
            if width_px >= EPISODE_LABEL_MIN_PX and run.episode_id not in labelled:
                ax.text(
                    middle,
                    1 + EPISODE_LABEL_PAD,
                    f'#{run.episode_id}' + labels.get(run.episode_id, ''),
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='bottom',
                    fontsize=layout.font_note_pt,
                    color=colour,
                    fontweight='bold',
                    clip_on=False,
                    zorder=8,
                )
        else:
            ax.axvspan(start, end, ymin=0, ymax=REJECTED_STRIP, facecolor=REJECTED, alpha=REJECTED_ALPHA, lw=0, zorder=0)
            letter = REJECT_LETTERS.get(getattr(run, 'reject_reason', RejectReason.NONE), '')
            if letter and width_px >= REJECT_LETTER_MIN_PX:
                ax.text(
                    middle,
                    0.5 * REJECTED_STRIP,
                    letter,
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='center',
                    fontsize=layout.font_note_pt - 0.5,
                    color=INK,
                    fontweight='bold',
                    zorder=8,
                )


@dataclass(frozen=True)
class PolicyMarks:
    """What (a) draws under the minimal policy (`policy_marks`): the counted and candidate episode ids, and the spans
    [yr] of a counted 1:1 leg and of candidate legs / half-swings, which are no episode row."""

    counted: frozenset = frozenset()
    candidates: frozenset = frozenset()
    counted_spans: Tuple[Tuple[float, float], ...] = ()
    candidate_spans: Tuple[Tuple[float, float], ...] = ()


def policy_marks(analysis) -> Optional[PolicyMarks]:
    """The episodes the minimal policy counts and its candidates (W29 SPEC §3.3), or None when the row is not under it
    (`policy` '': an inner pass or a verdict built by hand, (a) is drawn without marks). Read from the analysis'
    `PolicyResult` (`analysis.policy`), else from its verdict's `counted_episode_ids`/`candidate_episode_ids` (a stored
    verdict: the figure re-run path, §7), where a leg or a half-swing has no span and is not drawn."""
    result = getattr(analysis, 'policy', None)
    if result is not None and str(getattr(result, 'policy', '')) == verdict_text.POLICY_MINIMAL:
        return PolicyMarks(
            frozenset(result.counted_episode_ids),
            frozenset(result.candidate_episode_ids),
            tuple(tuple(span) for span in result.counted_spans_yr),
            tuple(tuple(span) for span in result.candidate_spans_yr),
        )
    verdict = getattr(analysis, 'verdict', None)
    if verdict is None or not verdict_text.policy_minimal(verdict):
        return None
    return PolicyMarks(
        verdict_text.policy_ids(getattr(verdict, 'counted_episode_ids', '')),
        verdict_text.policy_ids(getattr(verdict, 'candidate_episode_ids', '')),
    )


def _span_label(ax, middle: float, text: str, colour: str, layout: VerdictFigureLayout, bold: bool = True):
    return ax.text(
        middle,
        1 + EPISODE_LABEL_PAD,
        text,
        transform=ax.get_xaxis_transform(),
        ha='center',
        va='bottom',
        fontsize=layout.font_note_pt,
        color=colour,
        fontweight='bold' if bold else 'normal',
        clip_on=False,
        zorder=8,
    )


def fill_counted(ax, start: float, end: float, window: Tuple[float, float], colour: str, hatches: Sequence[str] = ()) -> None:
    """A counted episode's fill over [start, end] (display units): solid inside the analysis `window`, pale
    (TRIM_ALPHA_SCALE) over the stretch the filter's trim hides between the window and a covered record edge."""
    pieces = [(max(start, window[0]), min(end, window[1]), EPISODE_ALPHA, COUNTED_GID)]
    pieces += [(start, min(end, window[0]), EPISODE_ALPHA * TRIM_ALPHA_SCALE, TRIM_GID)]
    pieces += [(max(start, window[1]), end, EPISODE_ALPHA * TRIM_ALPHA_SCALE, TRIM_GID)]
    for x0, x1, alpha, gid in pieces:
        if x1 <= x0:
            continue
        ax.axvspan(x0, x1, facecolor=colour, alpha=alpha, lw=0, zorder=0, gid=gid)
        for hatch in hatches:
            ax.axvspan(x0, x1, facecolor='none', edgecolor=colour, hatch=hatch, alpha=0.5, lw=0, zorder=0, gid=gid)


def fill_candidate(ax, start: float, end: float, colour: str) -> None:
    """A candidate's fill (display units): faint (CANDIDATE_ALPHA_SCALE) with a dashed outline."""
    ax.axvspan(start, end, facecolor=colour, alpha=EPISODE_ALPHA * CANDIDATE_ALPHA_SCALE, lw=0, zorder=0, gid=CANDIDATE_GID)
    ax.axvspan(start, end, facecolor='none', edgecolor=colour, alpha=CANDIDATE_EDGE_ALPHA, lw=0.8, ls='--', zorder=0, gid=CANDIDATE_GID)


def shade_policy_runs(
    ax,
    runs,
    marks: PolicyMarks,
    colours: Mapping[int, str],
    times,
    px_per_unit: float,
    layout: VerdictFigureLayout,
    record: Tuple[float, float],
    window: Tuple[float, float],
    settings=None,
    labels: Optional[Mapping[int, str]] = None,
    induced_ids: Collection[int] = (),
    labelled: Collection[int] = (),
) -> Dict[int, Tuple[float, float]]:
    """(a) under the minimal policy (W29 SPEC §7): the counted runs filled and labeled '#id' (with `labels`, e.g.
    ' wave'; '/' before the last jump of a, '\\' induced) over their `episode_spans` among the drawn runs (`record`,
    `window` in yr: the fill reaches a covered record edge, `fill_counted`), a counted leg (`marks.counted_spans`)
    labeled 'leg'; candidates (`marks.candidates` over their
    turning points, `marks.candidate_spans`) faint and dashed, labeled 'cand' and not numbered; nothing for any other
    run. A counted run in `labeled` gets no '#id' here (`draw_run_labels` writes it). Returns the fill spans [yr] of the counted runs."""
    labels = labels or {}
    drawn = [run for run in runs if run.episode_id in marks.counted or run.episode_id in marks.candidates]
    spans = episode_spans(drawn, record, settings)
    shown_window = (float(times(window[0])), float(times(window[1])))
    counted_spans: Dict[int, Tuple[float, float]] = {}
    for run in sorted(drawn, key=lambda run: run.t_start_yr):
        eid = run.episode_id
        if eid in marks.counted:
            span = spans.get(eid, (run.t_start_yr, run.t_end_yr))
            counted_spans[eid] = span
            start, end = (float(times(v)) for v in span)
            colour = colours.get(eid, EPISODE_COLOURS[0])
            hatches = [EPISODE_BEFORE_JUMP_HATCH] if getattr(run, 'before_last_jump', False) else []
            hatches += [INDUCED_HATCH] if eid in induced_ids else []
            hatches += [RIPPLE_HATCH] if labels.get(eid) == RIPPLE_LABEL else []
            fill_counted(ax, start, end, shown_window, colour, hatches)
            if (end - start) * px_per_unit >= EPISODE_LABEL_MIN_PX and eid not in labelled:
                _span_label(ax, 0.5 * (start + end), f'#{eid}' + labels.get(eid, ''), colour, layout)
        else:
            start, end = (float(times(v)) for v in (run.t_start_yr, run.t_end_yr))
            colour = colours.get(eid, REJECTED)
            fill_candidate(ax, start, end, colour)
            if (end - start) * px_per_unit >= REJECT_LETTER_MIN_PX:
                _span_label(ax, 0.5 * (start + end), CANDIDATE_LABEL, colour, layout, bold=False)
    for t0, t1 in marks.counted_spans:
        start, end = (float(times(v)) for v in (t0, t1))
        fill_counted(ax, start, end, shown_window, EPISODE_COLOURS[0])
        if (end - start) * px_per_unit >= REJECT_LETTER_MIN_PX:
            _span_label(ax, 0.5 * (start + end), LEG_LABEL, EPISODE_COLOURS[0], layout)
    for t0, t1 in marks.candidate_spans:
        start, end = (float(times(v)) for v in (t0, t1))
        fill_candidate(ax, start, end, REJECTED)
        if (end - start) * px_per_unit >= REJECT_LETTER_MIN_PX:
            _span_label(ax, 0.5 * (start + end), CANDIDATE_LABEL, REJECTED, layout, bold=False)
    return counted_spans


def jump_groups(times_yr: Sequence[float], span_yr: float, fraction: float = JUMP_MERGE_FRACTION) -> List[Tuple[float, float, int]]:
    """Jumps closer than `fraction` of the record merged: (first time, last time, count) per group."""
    times = sorted(float(t) for t in times_yr if np.isfinite(t))
    groups: List[List[float]] = []
    for t in times:
        if groups and t - groups[-1][-1] <= fraction * span_yr:
            groups[-1].append(t)
        else:
            groups.append([t])
    return [(group[0], group[-1], len(group)) for group in groups]


class _Frame:
    """Times and the time unit of one figure, the arrays every panel reads and the drawing options."""

    def __init__(
        self, analysis: MMRAnalysis, series: MMRSeries, layout: VerdictFigureLayout = LAYOUT_V1, options=None, settings=None, row=None
    ):
        self.analysis = analysis
        self.series = series
        self._row = row  # summary ⋈ numbers row (the labels of (j) read its columns); built on first use when not given
        self.options = resolve_plot_options(options)
        self.settings = settings if settings is not None else _settings()  # the coverage rule of the episodes' fill
        self.t = np.asarray(series.t_yr, dtype=float)
        self.divisor, self.unit = time_unit(self.t[-1] - self.t[0])
        self.x = self.t / self.divisor
        prep = analysis.prep
        self.prep = prep
        self.window = prep.window if prep is not None else (self.t[0], self.t[-1])
        self.inside = (self.t >= self.window[0]) & (self.t <= self.window[1])
        self.raw_angle = analysis.sampling.angle_unwrapped if analysis.sampling is not None else series.angle_unwrapped_numpy
        geometry = analysis.geometry
        self.centre_t = geometry.centre.t_yr if geometry is not None else None
        self.centre = geometry.centre.centre_rad if geometry is not None else None
        reference = self.centre if self.centre is not None else (prep.angle_f[self.inside] if prep is not None else series.angle_wrapped)
        self.low = _window_low(reference)
        self.filter_order = analysis.filter_order or _filter_order()
        self.centre_lines = centre_lines(analysis, series, self.filter_order) if geometry is not None else []
        self.colours = episode_colours(geometry.runs) if geometry is not None else {}
        span = self.x[-1] - self.x[0]
        self.px_per_unit = layout.panels[2] * layout.dpi / span if span > 0 else 0.0

    def times(self, t_yr) -> np.ndarray:
        return np.asarray(t_yr, dtype=float) / self.divisor

    @property
    def row(self) -> Mapping[str, Any]:
        """The analysis' numbers ⋈ summary row (the columns the labels of (i) and (j) print), built once."""
        if self._row is None:
            self._row = {**self.analysis.numbers_row(), **self.analysis.summary_row()}
        return self._row

    def cycle_px(self) -> float:
        """Mean width of one cycle of sigma_f (libration or revolution) in pixels of a time panel; inf when unknown."""
        prep, geometry = self.prep, self.analysis.geometry
        if prep is None:
            return float('inf')
        angle = np.asarray(prep.angle_f, dtype=float)[self.inside]
        angle = angle[np.isfinite(angle)]
        if len(angle) < 2:
            return float('inf')
        librations = 0.5 * len(geometry.turning_points.times_yr) if geometry is not None else 0.0
        cycles = librations + abs(angle[-1] - angle[0]) / TWO_PI
        width = (self.window[1] - self.window[0]) / self.divisor * self.px_per_unit
        return width / cycles if cycles > 0 else float('inf')


def _rate(prep) -> np.ndarray:
    """sigma_dot as the response read it (smoothed at the response cutoff), else sigma_dot_f."""
    rate = getattr(prep, 'rate_response', None)
    return np.asarray(rate if rate is not None else prep.rate_f, dtype=float)


def _angle_response(prep) -> np.ndarray:
    """sigma as the force law read it (smoothed at the response cutoff, `response.response_series`), else sigma_f."""
    angle = getattr(prep, 'angle_response', None)
    return np.asarray(angle if angle is not None else prep.angle_f, dtype=float)


def _adot(prep) -> np.ndarray:
    """a_dot as the force law read it (smoothed at the response cutoff), else a_dot_f."""
    adot = getattr(prep, 'adot_response', None)
    return np.asarray(adot if adot is not None else prep.adot_f, dtype=float)


def _deciding_episode(analysis: MMRAnalysis):
    """The run the verdict rests on: the decisive episode, else the longest accepted run (any regime); None without."""
    geometry = analysis.geometry
    if geometry is None:
        return None
    accepted = accepted_runs(geometry)
    chosen = [run for run in accepted if run.episode_id == analysis.decisive_episode_id]
    if chosen:
        return chosen[0]
    return max(accepted, key=lambda run: run.t_end_yr - run.t_start_yr) if accepted else None


def separatrix_omega0(analysis: MMRAnalysis) -> Tuple[float, str]:
    """(omega0 [rad/yr], its source) that scales the axis of (c): the deciding (or longest librating) episode's
    omega0 ('#id'), else the theoretical omega0 = |k| n sqrt(3 mu e^q) ('theory'), else (nan, ''). Never the
    potential fitted over the whole record: with a low R^2 it is noise (1999 UQ: 2.5 against 0.04 rad/kyr)."""
    physics = {ep.episode_id: ep for ep in analysis.episode_physics}
    episode = _deciding_episode(analysis)
    ordered = [episode] if episode is not None else []
    if analysis.geometry is not None:
        others = sorted(
            (run for run in analysis.geometry.runs if run.accepted and run is not episode), key=lambda run: run.t_start_yr - run.t_end_yr
        )
        ordered += others
    for run in ordered:
        ep = physics.get(run.episode_id)
        if ep is not None and is_finite(ep.omega0_rad_yr) and ep.omega0_rad_yr > 0:
            return float(ep.omega0_rad_yr), f'#{run.episode_id}'
    strength = analysis.strength
    if strength is not None and is_finite(strength.omega0_theory_rad_yr) and strength.omega0_theory_rad_yr > 0:
        return float(strength.omega0_theory_rad_yr), 'theory'
    return float('nan'), ''


def _half_width(analysis: MMRAnalysis) -> float:
    accepted = {run.episode_id for run in analysis.geometry.runs if run.accepted} if analysis.geometry is not None else set()
    values = _finite([ep.half_width_au for ep in analysis.episode_physics if ep.episode_id in accepted])
    return float(np.median(values)) if len(values) else float('nan')


def _aliased(analysis: MMRAnalysis) -> bool:
    """The sampling verdict is aliased_*: the samples are stroboscopic, a portrait of them is an illusion."""
    return analysis.sampling is not None and str(analysis.sampling.verdict).startswith('aliased')


def _is_alias(analysis: MMRAnalysis) -> bool:
    return str(analysis.verdict.angle_subtype) == str(AngleSubtype.ALIAS)


# --------------------------------------------------------------------------------------------- header


def draw_header(
    fig,
    analysis: MMRAnalysis,
    layout: VerdictFigureLayout,
    series: Optional[MMRSeries] = None,
    stamp: Optional[str] = None,
    role: str = '',
) -> None:
    """Name, resonance, type and the orbit's type (`verdict.orbit_type`: tadpole L4, horseshoe, interior asymmetric
    libration about 72°, ...); under it the role of the other resonance, who makes the cycles (`role`, from
    `verdict_text.cause_line`: 'driver: 3N-5+0+2 (circulates, P 4.51 kyr): forced wave', W28 A3b, cards g04-6/g04-7);
    then the record line. The status is printed once, in the plaque of the right column."""
    width, height = fig.get_size_inches()
    x, y, w, h = layout.header
    top = y + h
    title = (
        f'{analysis.series_name}   {analysis.resonance.key}   [{TYPE_LABELS.get(analysis.resonance.kind, str(analysis.resonance.kind))}]'
    )
    orbit = orbit_type_text(analysis)
    if orbit:
        title += f'   {orbit}'
    if analysis.parent:
        title += f'   subresonance of {analysis.parent}'
    title_text = fig.text(x / width, top / height, title, fontsize=layout.font_title_pt, fontweight='bold', va='top', ha='left')
    fit_width(title_text, w * fig.dpi, layout.text_font_pt)  # a long sequence of 1:1 wells (524522 1V-1 20 kyr)
    below = HEADER_LINE_IN
    if role:
        line = fig.text(
            x / width, (top - below) / height, role, fontsize=layout.font_title_pt * HEADER_ROLE_SCALE, color=INK, va='top', ha='left'
        )
        fit_width(line, w * fig.dpi, layout.text_font_pt)
        below += HEADER_ROLE_LINE_IN
    if series is not None:
        fig.text(x / width, (top - below) / height, record_line(analysis, series), fontsize=layout.text_font_pt, color='#333333', va='top')
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


HEADER_LINE_IN = 0.36  # from the top of the header to the line under the title [in]
HEADER_ROLE_SCALE = 0.75  # the role line's font, a fraction of the title's
HEADER_ROLE_LINE_IN = 0.24  # the role line's height: the record line moves down by it


def fit_width(text, width_px: float, min_pt: float) -> None:
    """Shrink `text` until it is at most `width_px` wide, down to `min_pt` (a long three-body neighbor line)."""
    renderer = text.figure.canvas.get_renderer()
    extent = text.get_window_extent(renderer)
    if extent.width > width_px > 0:
        text.set_fontsize(max(min_pt, text.get_fontsize() * width_px / extent.width * 0.98))


def header_role(analysis: MMRAnalysis, rows, settings) -> str:
    """The role line of the header (`verdict_text.cause_line` on the figure's rows: summary ⋈ numbers, episodes); ''
    without a role or for a derived row's missing rows."""
    summary, numbers, episodes = rows[0], rows[1], rows[2]
    facts = verdict_text.MMRFacts(analysis, settings, {**numbers, **summary}, episodes)
    return verdict_text.cause_line(analysis.verdict, facts)[1]


ORBIT_SEQUENCE_PREFIX, ORBIT_HEADER_PREFIX = 'compound: ', 'orbit: '


def orbit_type_text(analysis: MMRAnalysis) -> str:
    """The orbit's type for the header: `verdict.orbit_type`, else the 1:1 regime of a co-orbital (rsp_coorb_regime).
    A sequence of 1:1 regimes (`orbit_type.coorbital_text`: 'compound: horseshoe → tadpole L4') reads 'orbit:
    horseshoe → tadpole L4' (W28 C13, card g03-9 A: one episode, its sub-regimes in time order; 'compound' stays the
    name of the regime whose swing holds the quasi-satellite and a tadpole well)."""
    orbit = str(getattr(analysis.verdict, 'orbit_type', '') or '')
    if orbit.startswith(ORBIT_SEQUENCE_PREFIX):
        return ORBIT_HEADER_PREFIX + orbit[len(ORBIT_SEQUENCE_PREFIX) :]
    if orbit:
        return orbit
    law = getattr(analysis.response, 'coorbital', None) if analysis.response is not None else None
    return f'1:1 {law.regime}' if law is not None and law.regime else ''


def record_line(analysis: MMRAnalysis, series: MMRSeries) -> str:
    """How the record was made, one line (the old title): span, integrator, step, rows, cutoff, epoch, FAIR."""
    regime = series.regime
    span = series.t_yr[-1] - series.t_yr[0]
    divisor, unit = time_unit(span)
    parts = [f'span {span / divisor:.4g} {unit} {regime.direction}']
    if regime.integrator:
        days = f' ({regime.dt / TWO_PI * GAUSSIAN_YEAR_DAYS:.1f} d)' if is_finite(regime.dt) else ''
        parts.append(f'{regime.integrator} dt {regime.dt:g}{days}' if is_finite(regime.dt) else regime.integrator)
    parts.append(f'step {float(np.median(np.diff(series.t_yr))):.4g} yr')
    parts.append(f'{len(series.t_yr)} rows')
    if analysis.prep is not None:
        parts.append(f'cutoff {analysis.prep.cutoff.cutoff_yr:.4g} yr ({analysis.prep.cutoff.binding})')
        if is_finite(analysis.prep.response_cutoff_yr):
            parts.append(f'response cutoff {analysis.prep.response_cutoff_yr:.4g} yr')
    if is_finite(regime.epoch_jd):
        parts.append(f'epoch JD {regime.epoch_jd:.1f}')
    reason = fair_unavailable_reason(analysis, series)
    parts.append('FAIR yes' if reason is None else f'FAIR no ({reason})')
    return '  |  '.join(parts)


# --------------------------------------------------------------------------------------------- time panels


def _style_axis(ax, letter: str, ylabel: str, layout: VerdictFigureLayout):
    """Ticks, the y label, the panel letter, and a full frame (panels stand apart, `panel_gap`)."""
    ax.tick_params(labelsize=layout.font_ticks_pt, length=2.5, pad=1.5)
    ax.set_ylabel(ylabel, fontsize=layout.font_axis_pt, labelpad=2)
    box = {'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75, 'pad': 0.6}
    ax.text(
        PANEL_LETTER_X,
        0.94,
        f'({letter})',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=layout.font_letter_pt,
        fontweight='bold',
        zorder=9,
        bbox=box,
    )
    ax.grid(alpha=0.15, lw=0.5)
    ax.yaxis.set_major_locator(_inside_locator(layout.tick_margin))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME)
        spine.set_linewidth(0.6)


def wave_labels(geometry) -> Dict[int, str]:
    """episode id -> the label after '#id' of an accepted run that is no episode of the angle (`Geometry.forced_waves`):
    ' wave' for a forced wave of another argument (F6), ' forced (strength)' for a run of a 5s row (B2), ' ripple' for
    an unconfirmed ripple (B6b)."""
    out = {}
    for wave in getattr(geometry, 'forced_waves', ()) or ():
        if getattr(wave, 'ripple', False):
            out[wave.episode_id] = RIPPLE_LABEL
        elif getattr(wave, 'by_strength', False):
            out[wave.episode_id] = STRENGTH_WAVE_LABEL
        else:
            out[wave.episode_id] = FORCED_WAVE_LABEL
    return out


def uncounted_tags(geometry) -> Dict[int, str]:
    """episode id -> the tag after '#id' of an accepted run the status does not count and no other mark names
    (`Geometry.eligibility_reason`: below the period floor ' < floor', the center turns ' center turns'; W28 C11,
    `verdict_text.UNCOUNTED_TAGS`)."""
    out = {}
    for run in getattr(geometry, 'runs', ()) or ():
        reason = geometry.eligibility_reason(run) if run.accepted else None
        if reason in verdict_text.UNCOUNTED_TAGS:
            out[run.episode_id] = verdict_text.UNCOUNTED_TAGS[reason]
    return out


def _shade_episodes(ax, frame: _Frame, layout: VerdictFigureLayout):
    geometry = frame.analysis.geometry
    if geometry is None:
        return
    marks = policy_marks(frame.analysis)
    if marks is not None:
        _shade_policy_episodes(ax, frame, layout, marks)
        return
    accepted = {run.episode_id for run in geometry.runs if run.accepted}
    spans = episode_spans(geometry.runs, (float(frame.t[0]), float(frame.t[-1])), frame.settings)
    forced = wave_labels(geometry)
    induced = induced_episode_ids(frame.analysis)
    uncounted = uncounted_tags(geometry)
    pieces = residence_pieces(geometry.runs, spans, (float(frame.t[0]), float(frame.t[-1])), skip=uncounted)
    bracketed = [
        run for run in geometry.runs if run.accepted and run.episode_id not in uncounted and len(getattr(run, 'subregimes', ())) >= 2
    ]
    labelled = set(pieces) | {run.episode_id for run in bracketed}
    shade_runs(
        ax, geometry.runs, accepted, frame.colours, frame.times, frame.px_per_unit, layout, spans, forced, induced, uncounted, labelled
    )
    shade_residences(ax, pieces, frame.colours, frame.times)
    draw_subregime_brackets(ax, bracketed, frame.colours, frame.times, frame.px_per_unit)
    for run in geometry.runs:
        if run.episode_id in labelled:
            draw_run_labels(ax, run, spans[run.episode_id], pieces.get(run.episode_id, ()), frame, layout, forced.get(run.episode_id, ''))


def _shade_policy_episodes(ax, frame: _Frame, layout: VerdictFigureLayout, marks: PolicyMarks) -> None:
    """(a) under the minimal policy (`shade_policy_runs`); a counted 1:1 run keeps its sub-regime brackets, its
    inferred residence and their labels (W28 C13)."""
    geometry = frame.analysis.geometry
    record = (float(frame.t[0]), float(frame.t[-1]))
    counted = [run for run in geometry.runs if run.episode_id in marks.counted]
    # a strength mark the policy does not uphold is the angle's episode again (2012 KL9): no ' forced (strength)'
    forced = {eid: words for eid, words in wave_labels(geometry).items() if not (eid in marks.counted and words == STRENGTH_WAVE_LABEL)}
    bracketed = [run for run in counted if len(getattr(run, 'subregimes', ())) >= 2]
    probe = episode_spans([run for run in geometry.runs if run.episode_id in marks.counted | marks.candidates], record, frame.settings)
    pieces = residence_pieces(counted, probe, record)
    labelled = set(pieces) | {run.episode_id for run in bracketed}
    spans = shade_policy_runs(
        ax,
        geometry.runs,
        marks,
        frame.colours,
        frame.times,
        frame.px_per_unit,
        layout,
        record,
        frame.window,
        frame.settings,
        forced,
        induced_episode_ids(frame.analysis),
        labelled,
    )
    shade_residences(ax, pieces, frame.colours, frame.times)
    draw_subregime_brackets(ax, bracketed, frame.colours, frame.times, frame.px_per_unit)
    for run in counted:
        if run.episode_id in labelled and run.episode_id in spans:
            draw_run_labels(ax, run, spans[run.episode_id], pieces.get(run.episode_id, ()), frame, layout, forced.get(run.episode_id, ''))


def residence_pieces(
    runs, spans: Mapping[int, Tuple[float, float]], record: Tuple[float, float], skip: Collection[int] = ()
) -> Dict[int, List[Tuple[float, float]]]:
    """episode id -> the stretches [yr] of its inferred residence (`Episode.residence`, W28 C8) beyond its fill
    (`spans`, `episode_spans`): before and after it, never past the record's edges nor into the fill of a neighboring
    accepted run; none for a run in `skip` (not counted) or without a residence beyond its fill."""
    ordered = sorted((run for run in runs if run.accepted and run.episode_id in spans), key=lambda run: run.t_start_yr)
    out: Dict[int, List[Tuple[float, float]]] = {}
    for i, run in enumerate(ordered):
        residence = getattr(run, 'residence', None)
        if residence is None or run.episode_id in skip:
            continue
        fill0, fill1 = spans[run.episode_id]
        low = spans[ordered[i - 1].episode_id][1] if i > 0 else record[0]
        high = spans[ordered[i + 1].episode_id][0] if i + 1 < len(ordered) else record[1]
        pieces = []
        start, end = max(float(residence.t0_yr), low, record[0]), min(float(residence.t1_yr), high, record[1])
        if start < fill0:
            pieces.append((start, fill0))
        if end > fill1:
            pieces.append((fill1, end))
        if pieces:
            out[run.episode_id] = pieces
    return out


def shade_residences(ax, pieces: Mapping[int, Sequence[Tuple[float, float]]], colours: Mapping[int, str], times) -> None:
    """(a): each run's inferred residence beyond its fill (`residence_pieces`) in its color, lighter than an episode
    and hatched RESIDENCE_HATCH (gid RESIDENCE_GID)."""
    for episode_id, spans in pieces.items():
        colour = colours.get(episode_id, EPISODE_COLOURS[0])
        for t0, t1 in spans:
            start, end = (float(times(v)) for v in (t0, t1))
            ax.axvspan(start, end, facecolor=colour, alpha=EPISODE_ALPHA * RESIDENCE_ALPHA_SCALE, lw=0, zorder=0, gid=RESIDENCE_GID)
            ax.axvspan(start, end, facecolor='none', edgecolor=colour, hatch=RESIDENCE_HATCH, alpha=0.4, lw=0, zorder=0, gid=RESIDENCE_GID)


def _points_over_top(ax, points: float) -> float:
    """y in axes fraction `points` over the top edge of `ax`."""
    height_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    return 1.0 + points / height_pt if height_pt > 0 else 1.0


def draw_subregime_brackets(ax, runs, colours: Mapping[int, str], times, px_per_unit: float) -> list:
    """(a): one bracket per sub-regime of each run (`Episode.subregimes`, two or more) just over the top edge, in the
    run's color, BRACKET_GAP_PX apart where two meet (gid BRACKET_GID); the names are `draw_run_labels`'."""
    rise, tick = _points_over_top(ax, BRACKET_RISE_PT), _points_over_top(ax, BRACKET_TICK_PT)
    gap = 0.5 * BRACKET_GAP_PX / px_per_unit if px_per_unit > 0 else 0.0
    lines = []
    for run in runs:
        colour = colours.get(run.episode_id, EPISODE_COLOURS[0])
        for sub in run.subregimes:
            x0, x1 = (float(times(v)) for v in (sub.t0_yr, sub.t1_yr))
            if x1 - x0 > 2 * gap:
                x0, x1 = x0 + gap, x1 - gap
            (line,) = ax.plot(
                [x0, x0, x1, x1], [tick, rise, rise, tick], color=colour, lw=0.9, transform=ax.get_xaxis_transform(), clip_on=False
            )
            line.set_gid(BRACKET_GID)
            lines.append(line)
    return lines


def run_label_items(
    run, fill: Tuple[float, float], pieces: Sequence[Tuple[float, float]], suffix: str = ''
) -> List[Tuple[float, float, str, bool]]:
    """The labels over (a) of an accepted 1:1 run with sub-regimes or a residence (W28 C13), (start, end, text, is a
    sub-regime) [yr] in time order: '#id' (with `suffix`, `wave_labels`) and the first well over its first bracket, each
    next well over its own (the same well after an amplitude change NEW_AMPLITUDE_LABEL), else '#id' over the fill;
    'inferred' over each residence piece."""
    subs = list(run.subregimes) if len(run.subregimes) >= 2 else []
    head = f'#{run.episode_id}{suffix}'
    items: List[Tuple[float, float, str, bool]] = []
    if subs:
        for k, sub in enumerate(subs):
            name = verdict_text.well_words(sub.well) if k == 0 or sub.well != subs[k - 1].well else ''
            if k and not name and sub.starts_at == MARK_AMPLITUDE_CHANGE:
                name = NEW_AMPLITUDE_LABEL
            text = ' '.join(part for part in (head if k == 0 else '', name) if part)
            if text:
                items.append((float(sub.t0_yr), float(sub.t1_yr), text, True))
    else:
        items.append((float(fill[0]), float(fill[1]), head, False))
    items += [(float(t0), float(t1), RESIDENCE_LABEL, False) for t0, t1 in pieces]
    return sorted(items, key=lambda item: 0.5 * (item[0] + item[1]))


def merge_label_items(items: Sequence[Tuple[float, float, str, bool]], width_of: Callable[[str], float], gap: float = 0.0):
    """`items` (start, end, text, is a sub-regime) whose text boxes centered over their spans overlap merged into one
    over their joint span, in time order, until none overlap: two sub-regimes' names joined by ' → ', anything else
    by ' · '. `width_of(text)` in the units of the spans."""
    merged: List[Tuple[float, float, str, bool]] = []
    for item in sorted(items, key=lambda item: 0.5 * (item[0] + item[1])):
        merged.append(item)
        while len(merged) > 1:
            (s0, e0, a, sub_a), (s1, e1, b, sub_b) = merged[-2], merged[-1]
            if 0.5 * (s0 + e0) + 0.5 * width_of(a) + gap <= 0.5 * (s1 + e1) - 0.5 * width_of(b):
                break
            merged[-2:] = [(min(s0, s1), max(e0, e1), f'{a}{" → " if sub_a and sub_b else " · "}{b}', sub_a and sub_b)]
    return merged


def draw_run_labels(ax, run, fill, pieces, frame: _Frame, layout: VerdictFigureLayout, suffix: str = '') -> list:
    """The labels of `run_label_items` over (a), merged where they would overlap (`merge_label_items`): over the
    brackets for a run with sub-regimes, else at the '#id' height; an 'inferred' wider than its piece is left out
    (the hatch alone). When a merged name of wells is wider than the run's fill (524522 1V-1 20 kyr: 15 sub-regimes
    in 2.3 kyr), the brackets keep no names and '#id' stands over the fill: the header and the text name the wells.
    Nothing when the fill is narrower than EPISODE_LABEL_MIN_PX, as for '#id'."""
    if (fill[1] - fill[0]) * frame.px_per_unit < EPISODE_LABEL_MIN_PX:
        return []
    units_per_pt = layout.dpi / 72 / frame.px_per_unit if frame.px_per_unit > 0 else 0.0

    def width(text: str) -> float:
        return _text_width_pt(text, layout.font_note_pt) * units_per_pt

    items = []
    for t0, t1, text, sub in run_label_items(run, fill, pieces, suffix):
        x0, x1 = (float(frame.times(v)) for v in (t0, t1))
        if text == RESIDENCE_LABEL and width(text) > x1 - x0:
            continue
        items.append((x0, x1, text, sub))
    bracketed = len(run.subregimes) >= 2
    y = _points_over_top(ax, BRACKET_LABEL_PT) if bracketed else 1 + EPISODE_LABEL_PAD
    colour = frame.colours.get(run.episode_id, EPISODE_COLOURS[0])
    gap = 2 * units_per_pt
    merged = merge_label_items(items, width, gap=gap)
    fill_x0, fill_x1 = (float(frame.times(v)) for v in fill)
    if bracketed and any(text != RESIDENCE_LABEL and width(text) > fill_x1 - fill_x0 for _, _, text, _ in merged):
        head = (fill_x0, fill_x1, f'#{run.episode_id}{suffix}', False)
        merged = merge_label_items([head] + [item for item in items if item[2] == RESIDENCE_LABEL], width, gap=gap)
    texts = []
    for x0, x1, text, _ in merged:
        texts.append(
            ax.text(
                0.5 * (x0 + x1),
                y,
                text,
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='bottom',
                fontsize=layout.font_note_pt,
                color=colour,
                fontweight='normal' if text == RESIDENCE_LABEL else 'bold',
                fontstyle='italic' if text == RESIDENCE_LABEL else 'normal',
                clip_on=False,
                zorder=8,
            )
        )
    return texts


def kick_times(runs) -> List[float]:
    """The kicks of a at the changes inside accepted 1:1 runs that are no boundary (`Episode.marks`, kick@t) [yr]."""
    out = []
    for run in runs:
        if run.accepted:
            out += [t0 for name, t0, _ in verdict_text.episode_marks({'marks': getattr(run, 'marks', '')}) if name == 'kick']
    return sorted(out)


def draw_kicks(ax, frame: _Frame, layout: VerdictFigureLayout) -> list:
    """(d): a ▼ at the top edge at each kick of a met at a change of a 1:1 run that is no boundary (`kick_times`; the
    jump itself is the dotted line of (a)), labeled 'kick' to its right; labels that would overlap merge ('kicks')."""
    geometry = frame.analysis.geometry
    times = kick_times(geometry.runs) if geometry is not None else []
    if not times:
        return []
    xs = [float(frame.times(t)) for t in times]
    top = _points_over_top(ax, -0.5 * KICK_MARK_PT - 1.0)  # the triangle just under the top edge, inside the panel
    marks = ax.plot(
        xs,
        [top] * len(xs),
        ls='none',
        marker='v',
        ms=KICK_MARK_PT,
        color=REFERENCE,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        zorder=9,
    )
    marks[0].set_gid(KICK_GID)
    units_per_pt = layout.dpi / 72 / frame.px_per_unit if frame.px_per_unit > 0 else 0.0
    lead = 0.5 * KICK_MARK_PT * units_per_pt  # the label starts right of its ▼
    reach = lead + _text_width_pt(f' {KICK_LABEL}s', layout.font_note_pt) * units_per_pt
    texts = []
    for x0, x1 in merge_overlapping_labels([(x, x + reach) for x in xs], lambda span: span[1] - span[0]):
        n = sum(1 for x in xs if x0 <= x <= x1)
        texts.append(
            ax.text(
                x0 + lead,
                top,
                f' {KICK_LABEL}s' if n > 1 else f' {KICK_LABEL}',
                transform=ax.get_xaxis_transform(),
                ha='left',
                va='center',
                fontsize=layout.font_note_pt,
                color=REFERENCE,
                zorder=9,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': JUMP_LABEL_PAD_PT},
            )
        )
    return texts


def merge_overlapping_labels(
    spans: Sequence[Tuple[float, float]], width_of: Callable[[Tuple[float, float]], float], gap: float = 0.0
) -> List[Tuple[float, float]]:
    """Labels centered over `spans` [(start, end)] whose text boxes overlap merged into one label over their joint span,
    in time order: a merged label is measured again (`width_of(span)`, its own text) and merged on with its neighbors
    until no two boxes overlap. `gap`: the least free space between two boxes, in the units of the spans."""
    merged: List[Tuple[float, float]] = []
    for span in sorted(spans):
        merged.append(span)
        while len(merged) > 1:
            (s0, e0), (s1, e1) = merged[-2], merged[-1]
            right = 0.5 * (s0 + e0) + 0.5 * width_of((s0, e0))
            left = 0.5 * (s1 + e1) - 0.5 * width_of((s1, e1))
            if right + gap <= left:
                break
            merged[-2:] = [(min(s0, s1), max(e0, e1))]
    return merged


def _text_width_pt(text: str, size: float) -> float:
    """Width of `text` in the default sans font at `size` [pt]."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    return TextToPath().get_text_width_height_descent(text, FontProperties(size=size), ismath=False)[0]


def _jumps_text(count: int) -> str:
    return f'{count} jumps of a'


def _draw_jumps(ax, frame: _Frame, layout: VerdictFigureLayout):
    """Jumps of a: one dotted line each, or one hatched band 'N jumps of a' for jumps closer than 1 % of the record.
    Band labels whose text boxes would overlap become one label over their joint span, counting every jump in it
    (`merge_overlapping_labels`; TG2 4E-1M-3 100 kyr read '3 jum2 ju2 jumps of a')."""
    geometry = frame.analysis.geometry
    if geometry is None or not geometry.jumps:
        return
    _draw_action_steps(ax, frame, layout, [jump for jump in geometry.jumps if jump.action_step])
    jumps = [jump for jump in geometry.jumps if not jump.action_step]
    if not jumps:
        return
    span = frame.t[-1] - frame.t[0]
    min_width = 2.0 / frame.px_per_unit if frame.px_per_unit > 0 else 0.0  # a band is at least 2 px wide
    bands = []
    for first, last, count in jump_groups([jump.time_yr for jump in jumps], span):
        start, end = float(frame.times(first)), float(frame.times(last))
        if count == 1:
            ax.axvline(start, color=REFERENCE, ls=':', lw=0.8, zorder=2)
            continue
        middle, half = 0.5 * (start + end), max(0.5 * (end - start), 0.5 * min_width)
        ax.axvspan(middle - half, middle + half, facecolor='none', edgecolor=REFERENCE, hatch='xxx', lw=0.4, alpha=0.6, zorder=2)
        bands.append((start, end))
    times = frame.times([jump.time_yr for jump in jumps])

    def count_in(piece: Tuple[float, float]) -> int:
        return int(np.count_nonzero((times >= piece[0]) & (times <= piece[1])))

    units_per_pt = layout.dpi / 72 / frame.px_per_unit if frame.px_per_unit > 0 else 0.0
    pad = 2 * JUMP_LABEL_PAD_PT * units_per_pt  # the white box around each label

    def width(piece: Tuple[float, float]) -> float:
        return _text_width_pt(_jumps_text(count_in(piece)), layout.font_note_pt) * units_per_pt + pad

    for start, end in merge_overlapping_labels(bands, width):
        ax.text(
            0.5 * (start + end),
            REJECTED_STRIP + 0.02,
            _jumps_text(count_in((start, end))),
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=layout.font_note_pt,
            color=REFERENCE,
            zorder=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': JUMP_LABEL_PAD_PT},
        )


def _draw_action_steps(ax, frame: _Frame, layout: VerdictFigureLayout, steps) -> None:
    """G2a cuts (`geometry.ActionStep`): a dotted line at the middle of the transitional half-cycle, like a jump of a,
    its half-cycle shaded lightly and labeled 'action step (G2a)'."""
    for jump in steps:
        start, end = (float(x) for x in frame.times([jump.zone_start_yr, jump.zone_end_yr]))
        middle = float(frame.times(jump.time_yr))
        ax.axvspan(start, end, facecolor=REFERENCE, edgecolor='none', alpha=0.08, zorder=1)
        ax.axvline(middle, color=REFERENCE, ls=':', lw=0.8, zorder=2)
        ax.text(
            middle,
            REJECTED_STRIP + 0.02,
            'action step (G2a)',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=layout.font_note_pt,
            color=REFERENCE,
            zorder=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': JUMP_LABEL_PAD_PT},
        )


def regimes(frame: _Frame) -> List[Tuple[float, float]]:
    """The analysis window split at the jumps of a that cut the record (merged as in (a)) [yr]."""
    geometry = frame.analysis.geometry
    start, end = float(frame.t[0]), float(frame.t[-1])
    if geometry is None:
        return [(start, end)]
    cuts = [jump.time_yr for jump in geometry.jumps if not jump.kick]
    pieces, edge = [], start
    for first, last, _ in jump_groups(cuts, end - start):
        if first > edge:
            pieces.append((edge, first))
        edge = max(edge, last)
    pieces.append((edge, end))
    return [piece for piece in pieces if piece[1] > piece[0]]


def draw_angle_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(a) sigma unwrapped with the center, episodes, turning points, jumps and the exit."""
    analysis, prep, geometry = frame.analysis, frame.prep, frame.analysis.geometry
    _shade_episodes(ax, frame, layout)
    if frame.raw_angle is not None and prep is not None:
        _raw(ax, frame.x, frame.raw_angle)
    elif frame.raw_angle is not None:  # nothing filtered: the raw angle is the line to read
        _line(ax, frame.x, frame.raw_angle, color=INK, lw=0.8)
    if prep is not None:
        _line(ax, frame.x, prep.angle_f, color=INK, lw=0.9, zorder=3)
    if frame.centre is not None:
        _line(ax, frame.times(frame.centre_t), frame.centre, color=CENTRE, lw=1.0, zorder=4)
    if geometry is not None:
        _turning_points(ax, frame, geometry.turning_points)
        _draw_jumps(ax, frame, layout)
        if geometry.exit is not None and prep is not None:
            j = min(np.searchsorted(frame.t, geometry.exit.time_yr), len(frame.t) - 1)
            ax.scatter([frame.x[j]], [prep.angle_f[j]], marker='x', s=30, color=REFERENCE, zorder=6, linewidths=1.3)
    low, high = angle_view(prep.angle_f if prep is not None else None, frame.raw_angle)
    ax.set_ylim(low, high)
    if frame.raw_angle is not None:
        outlier_markers(ax, frame.x, frame.raw_angle, low, high)
    _style_axis(ax, 'a', 'σ, rad', layout)
    if analysis.sampling is None:
        _note(ax, 'unwrap not computed', layout.font_note_pt)


def angle_view(filtered, raw, pad: float = ANGLE_VIEW_PAD) -> Tuple[float, float]:
    """y limits of (a): sigma_f (0.2-99.8 percentiles) and the osculating sigma's band (ENVELOPE_PERCENTILES), so the
    whole gray band is in view, padded by `pad` of the span; the samples beyond are `outlier_markers`."""
    ranges = [
        r
        for r in (_view_limits(filtered, pad=0.0) if filtered is not None else None, envelope_range(raw) if raw is not None else None)
        if r
    ]
    if not ranges:
        return -1.0, 1.0
    low, high = min(r[0] for r in ranges), max(r[1] for r in ranges)
    span = high - low or max(abs(high), 1.0) * 1e-6
    return low - pad * span, high + pad * span


def alias_text(analysis: MMRAnalysis) -> str:
    """The one-line ALIAS plaque of (b): the step, the measured period of sigma (slope of the predictive unwrap) and
    the revolutions of the predictive unwrap, e.g. 'ALIAS  step 5.0012 yr ≈ P_σ 5.0015 yr · 3999 rev' (from
    `analysis.sampling` / `context`; the rest is in the right column)."""
    sampling, context = analysis.sampling, analysis.context
    step = f'step {sampling.step_yr:.5g} yr' if sampling is not None else ''
    rate = context.rate_measured_rad_yr if context is not None else float('nan')
    period = f'P_σ {TWO_PI / abs(rate):.5g} yr' if is_finite(rate) and rate != 0 else ''
    revs = f'{sampling.revs_predictive:.0f} rev' if sampling is not None and is_finite(sampling.revs_predictive) else ''
    return 'ALIAS  ' + ' · '.join(part for part in (' ≈ '.join(p for p in (step, period) if p), revs) if part)


def _turning_points(ax, frame: _Frame, tp):
    if tp is None or not len(tp.times_yr) or len(tp.times_yr) > MAX_TP_MARKERS:
        return
    turn_markers(ax, frame.times(tp.times_yr), tp.values_rad, tp.kinds, TP_SIZE, REFERENCE)


def _filtered_angle(frame: _Frame, angle) -> np.ndarray:
    """An angle (mod 2 pi) low-passed at the record's cutoff as a unit vector, unwrapped [rad]."""
    cutoff = frame.prep.cutoff.cutoff_yr if frame.prep is not None else None
    angle = np.asarray(angle, dtype=float)
    if not is_finite(cutoff):
        return np.unwrap(angle)
    return _circular_smooth(frame.t, angle, cutoff, frame.filter_order)


def draw_wrapped_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(b) sigma in a 360-degree window around the center: every osculating sample, sigma_f (when its cycles are
    resolved), the center, the owner / subresonance angle filtered at the cutoff; the ALIAS plaque. Cycles too dense
    for (b) are shown by the strip (b′) under it (`draw_strip`)."""
    low = frame.low
    points = np.degrees(_wrap_window(frame.series.angle_wrapped, low))
    ax.scatter(frame.x, points, s=0.5, color=RAW, alpha=0.5, linewidths=0, rasterized=True, zorder=3)
    cycle_px = frame.cycle_px()
    if frame.prep is not None and cycle_px >= CYCLE_MIN_PX:
        wrapped = _break_jumps(np.degrees(_wrap_window(frame.prep.angle_f, low)), 180)
        _line(ax, frame.x, wrapped, unwrapped=frame.prep.angle_f, color=INK, lw=0.5, zorder=5)
    if frame.centre is not None and cycle_px >= CYCLE_MIN_PX:
        centre = _break_jumps(np.degrees(_wrap_window(frame.centre, low)), 180)
        _line(ax, frame.times(frame.centre_t), centre, unwrapped=frame.centre, color=CENTRE, lw=0.9, zorder=4)
    for line in frame.centre_lines:  # the accepted subresonance or owner: its own angle at the cutoff, under sigma_f
        if line.angle_wrapped is None or len(line.angle_wrapped) != len(frame.t):
            continue
        colour, style = (SUBRESONANCE, '-.') if line.role == 'subresonance' else (OWNER, '--')
        smooth = _filtered_angle(frame, line.angle_wrapped)
        overlay = _break_jumps(np.degrees(_wrap_window(smooth, low)), 180)
        _line(ax, frame.x, overlay, unwrapped=smooth, color=colour, ls=style, lw=0.8, alpha=0.5, zorder=2)  # key: (j), legend
    base = np.degrees(low)
    ax.set_ylim(base, base + 360)
    _style_axis(ax, 'b', 'σ, °', layout)
    ticks = _right_angles(base, inner=True)
    edge = 360 * layout.tick_margin
    ax.set_yticks(ticks[(ticks > base + edge) & (ticks < base + 360 - edge)])
    ax.yaxis.set_major_formatter(_angle_formatter())
    if _is_alias(frame.analysis):  # one short plaque where the illusion is (the numbers are in the right column)
        alias_plaque(ax, alias_text(frame.analysis), layout)


def alias_plaque(ax, text: str, layout: VerdictFigureLayout):
    """The ALIAS plaque in the top right corner of (b)."""
    return ax.text(
        0.995,
        0.94,
        text,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=layout.font_note_pt + 0.5,
        color=CENTRE,
        fontweight='bold',
        zorder=11,
        bbox={'facecolor': 'white', 'edgecolor': CENTRE, 'lw': 0.6, 'pad': 1.2},
    )


def needs_strip(frame: _Frame) -> bool:
    """The strip (b′) is drawn when the cycles of sigma_f are narrower than STRIP_CYCLE_PX (then (b) cannot show them)."""
    return frame.prep is not None and frame.cycle_px() < STRIP_CYCLE_PX


def zoom_window(frame: _Frame) -> Optional[Tuple[float, float]]:
    """[t0, t1] (yr) of the first window of the strip: ZOOM_CYCLES cycles (at least ZOOM_MIN_STEPS steps) in the middle
    of the deciding episode, else in the middle of the window with the circulation period 2 pi / median |sigma_dot_f|."""
    episode = _deciding_episode(frame.analysis)
    if episode is not None and is_finite(episode.period_med_yr):
        return _centred_window(frame, 0.5 * (episode.t_start_yr + episode.t_end_yr), episode.period_med_yr)
    return _circulation_window(frame, frame.window[0], frame.window[1])


def _centred_window(frame: _Frame, middle: float, period: float) -> Tuple[float, float]:
    step = float(np.median(np.diff(frame.t))) if len(frame.t) > 1 else 0.0
    half = 0.5 * max(ZOOM_CYCLES * period, ZOOM_MIN_STEPS * step)
    return middle - half, middle + half


def _circulation_window(frame: _Frame, t0: float, t1: float) -> Optional[Tuple[float, float]]:
    """ZOOM_CYCLES cycles in the middle of [t0, t1] with the period 2 pi / median |sigma_dot_f| there."""
    mask = (frame.t >= t0) & (frame.t <= t1) & frame.inside
    rate = _finite(np.abs(np.asarray(frame.prep.rate_f, dtype=float)[mask]))
    rate_med = float(np.median(rate)) if len(rate) else float('nan')
    if not is_finite(rate_med) or rate_med <= 0:
        return None
    return _centred_window(frame, 0.5 * (t0 + t1), TWO_PI / rate_med)


def strip_windows(frame: _Frame) -> List[Tuple[float, float]]:
    """[(t0, t1)] (yr) of the strip: `zoom_window` and, when the record ends in another regime after a jump of a and
    the deciding episode lies before it, ZOOM_CYCLES cycles in the middle of that last regime."""
    windows = [window for window in (zoom_window(frame),) if window is not None]
    pieces = regimes(frame)
    episode = _deciding_episode(frame.analysis)
    if len(pieces) > 1 and episode is not None and episode.t_end_yr < pieces[-1][0]:
        last = _circulation_window(frame, *pieces[-1])
        if last is not None:
            windows.append(last)
    return windows


def draw_strip(fig, rect: Rect, frame: _Frame, layout: VerdictFigureLayout, wrapped_ax) -> list:
    """The strip (b′) in `rect` (inches): one axes per window of `strip_windows`, each on its own time axis (yr) with
    the samples, the path between them by the predictive unwrap (dashed: a whole revolution may pass between two
    samples) and sigma_f (broken where it turns by more than pi between samples). The windows are marked ▼ on (b)
    (`strip_marks`)."""
    windows = strip_windows(frame)
    width, height = fig.get_size_inches()
    x, y, w, h = rect
    sub_w = (w - STRIP_GAP * (len(windows) - 1)) / max(len(windows), 1)
    axes, marks = [], []
    for i, (t0, t1) in enumerate(windows):
        mask = (frame.t >= t0) & (frame.t <= t1)
        if np.count_nonzero(mask) < 2:
            continue
        ax = fig.add_axes(((x + i * (sub_w + STRIP_GAP)) / width, y / height, sub_w / width, h / height))
        letter = 'b′' if len(windows) == 1 else f'b′{i + 1}'
        _draw_strip_window(ax, frame, layout, mask, letter)
        axes.append(ax)
        marks.append((float(frame.times(0.5 * (t0 + t1))), str(i + 1) if len(windows) > 1 else ''))
    strip_marks(wrapped_ax, marks, layout)
    return axes


def text_box(text, renderer):
    """The pixel box of a text with its white plaque (`bbox`), if it has one."""
    from matplotlib.transforms import Bbox

    box = text.get_window_extent(renderer)
    patch = text.get_bbox_patch()
    if patch is None:
        return box
    text.update_bbox_position_size(renderer)
    return Bbox.union([box, patch.get_window_extent(renderer)])


def _overlap_px2(box, others) -> float:
    """The area [px²] that `box` shares with the boxes `others`."""
    area = 0.0
    for other in others:
        w = min(box.x1, other.x1) - max(box.x0, other.x0)
        h = min(box.y1, other.y1) - max(box.y0, other.y0)
        area += w * h if w > 0 and h > 0 else 0.0
    return area


def strip_marks(ax, marks: Sequence[Tuple[float, str]], layout: VerdictFigureLayout) -> Tuple[list, list]:
    """The ▼ over (b) at the middle of each window of the strip (b′) and its number when there are several (`marks`:
    (x [display unit], number or '')); returns (markers, numbers). The x of a ▼ is the window's middle and never moves
    (W28 g11-8). Neither may hide under the panel letter or the ALIAS plaque (the texts of (b) drawn before): a ▼ whose
    box meets one is raised just over it; a number is tried right of its ▼ hanging from the frame, then level with
    the ▼ (centered on the frame), then left of it level and hanging, and keeps the first place clear of those texts,
    the ▼s and the numbers already placed (the least overlap when none is clear). In pixels at the figure's dpi."""
    from matplotlib.transforms import ScaledTranslation

    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    base = ax.get_xaxis_transform()
    obstacles = [text_box(text, renderer) for text in ax.texts if text.get_visible() and text.get_text().strip()]
    markers, numbers, taken, lifts = [], [], [], []
    for middle, _ in marks:
        (marker,) = ax.plot([middle], [1.0], marker='v', ms=STRIP_MARK_PT, color=REFERENCE, transform=base, clip_on=False, zorder=12)
        box = marker.get_window_extent(renderer)
        hit = [other for other in obstacles if _overlap_px2(box, [other]) > 0]
        lift = (max(other.y1 for other in hit) - box.y0 + 1.0) / fig.dpi if hit else 0.0  # [in]: over what it meets
        marker.set_transform(base + ScaledTranslation(0.0, lift, fig.dpi_scale_trans))  # x unchanged
        markers.append(marker)
        lifts.append(lift)
        taken.append(marker.get_window_extent(renderer))
    gap = STRIP_MARK_GAP_PT / 72
    places = (('left', 'top', gap), ('left', 'center', gap), ('right', 'center', -gap), ('right', 'top', -gap))
    for (middle, number), lift in zip(marks, lifts):
        if not number:
            continue
        best = None
        for ha, va, dx in places:
            text = ax.text(middle, 1.0, number, ha=ha, va=va, fontsize=layout.font_note_pt, color=REFERENCE, zorder=12)
            text.set_transform(base + ScaledTranslation(dx, lift, fig.dpi_scale_trans))
            area = _overlap_px2(text_box(text, renderer), obstacles + taken)
            if best is None or area < best[0]:
                if best is not None:
                    best[1].remove()
                best = (area, text)
            else:
                text.remove()
            if area == 0:
                break
        numbers.append(best[1])
        taken.append(text_box(best[1], renderer))
    return markers, numbers


def _draw_strip_window(ax, frame: _Frame, layout: VerdictFigureLayout, mask, letter: str):
    t = frame.t[mask]
    angle_f = np.asarray(frame.prep.angle_f, dtype=float)[mask]
    low = _circular_mean(angle_f) - np.pi
    spread = np.ptp(_finite(np.degrees(_wrap_window(angle_f, low)))) if len(_finite(angle_f)) else 360.0
    if frame.raw_angle is not None:
        raw = np.asarray(frame.raw_angle, dtype=float)[mask]
        fine = np.linspace(t[0], t[-1], ZOOM_SUBSTEPS * (len(t) - 1) + 1)
        path = np.degrees(_wrap_window(np.interp(fine, t, raw), low))
        ax.plot(fine, _break_jumps(path, 180), color=RAW, lw=0.6, ls='--', zorder=1)
        steps = _finite(np.abs(np.diff(raw)))
        spread = max(spread, np.degrees(steps.max())) if len(steps) else spread  # a revolution between samples
    ax.plot(t, np.degrees(_wrap_window(np.asarray(frame.series.angle_wrapped, dtype=float)[mask], low)), 'o', ms=1.8, color=RAW, zorder=2)
    shown = _break_jumps(np.degrees(_wrap_window(angle_f, low)), 180)
    shown[1:][~(np.abs(np.diff(angle_f)) <= np.pi)] = np.nan  # no line across a revolution between two samples
    ax.plot(t, shown, color=INK, lw=0.8, zorder=3)
    base = np.degrees(low)
    ax.set_xlim(t[0], t[-1])
    _style_axis(ax, letter, 'σ, °' if letter in ('b′', 'b′1') else '', layout)
    ax.texts[-1].set_text(f'({letter}) t, yr')
    if spread < 180:  # a libration: its own range, not the whole circle
        middle_deg, half = base + 180.0, max(0.75 * spread, 30.0)
        ax.set_ylim(middle_deg - half, middle_deg + half)
        ax.yaxis.set_major_locator(_inside_locator(layout.tick_margin + 0.05, nbins=3))
    else:
        ax.set_ylim(base, base + 360)
        ticks = _right_angles(base, inner=True)
        edge = 360 * layout.tick_margin
        ax.set_yticks(ticks[(ticks > base + edge) & (ticks < base + 360 - edge)])
    ax.yaxis.set_major_formatter(_angle_formatter())
    ax.xaxis.set_major_locator(_inside_locator(0.06, nbins=6))
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.tick_params(axis='x', direction='in', pad=-9, labelsize=layout.font_ticks_pt - 1, length=2)
    for label in ax.get_xticklabels():
        label.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 0.2})


def _nu_by_regime(frame: _Frame) -> List[Tuple[float, float, float]]:
    """(t0, t1, nu [rad/yr]) of each regime: nu(<a>) + k (lambda_dot_regime - n(<a>)), the context's
    rate_from_mean_longitude evaluated with the regime's own mean lambda_dot; one regime: the context's value."""
    from resonances.resonance.analysis.mmr.context import mean_longitude_rate

    context, series = frame.analysis.context, frame.series
    if context is None:
        return []
    pieces = regimes(frame)
    full = context.rate_from_mean_longitude_rad_yr if is_finite(context.rate_from_mean_longitude_rad_yr) else context.rate_expected_rad_yr
    if len(pieces) == 1 or series.mean_longitude is None:
        return [(frame.t[0], frame.t[-1], full)] if is_finite(full) else []
    result = []
    k = frame.analysis.resonance.body_coefficient
    for t0, t1 in pieces:
        mask = (frame.t >= t0) & (frame.t <= t1)
        rate = mean_longitude_rate(frame.t[mask], np.asarray(series.mean_longitude)[mask], np.asarray(series.a)[mask])
        if is_finite(rate):
            result.append((t0, t1, context.rate_expected_rad_yr + k * (rate - context.mean_motion_rad_yr)))
    return result


def episode_omega0(analysis: MMRAnalysis) -> List[Tuple[float, float, float]]:
    """(t_start, t_end [yr], omega0 [rad/yr]) of every accepted episode with a measured omega0, in time order."""
    if analysis.geometry is None:
        return []
    physics = {ep.episode_id: ep for ep in analysis.episode_physics}
    result = []
    for run in sorted(accepted_runs(analysis.geometry), key=lambda run: run.t_start_yr):
        ep = physics.get(run.episode_id)
        if ep is not None and is_finite(ep.omega0_rad_yr) and ep.omega0_rad_yr > 0:
            result.append((float(run.t_start_yr), float(run.t_end_yr), float(ep.omega0_rad_yr)))
    return result


def _level(ax, value: float, low: float, high: float, x0: float, x1: float, at: float = 0.5, **style):
    """A horizontal level over [x0, x1] (data), or a triangle at the edge when it is beyond the axis, at the fraction
    `at` of the span (levels that would share the edge are told apart by it)."""
    if low <= value <= high:
        ax.hlines(value, x0, x1, **style)
    else:
        edge_marker(ax, x0 + at * (x1 - x0), value > high, style['color'], data_x=True)


def draw_rate_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(c) sigma_dot at the response cutoff. With an omega0 (`separatrix_omega0`: the deciding episode's, else the
    theory's) as sigma_dot / 2 omega0 on a symlog axis, linear inside +-1 (the speed at the separatrix), logarithmic
    outside: a libration stays inside +-1 and a circulation of any speed is in view on every figure. Over each accepted
    episode +-2 omega0 of that episode (dashed; without an episode +-1 of the theory, thin), nu(lambda_dot) per regime
    and the slips. Without an omega0, sigma_dot in rad/kyr, stretched to 0 only when sigma_dot would still fill
    RATE_DATA_MIN_SHARE of the height. A level beyond the axis is a triangle at the edge."""
    analysis = frame.analysis
    if frame.prep is None:
        _style_axis(ax, 'c', 'dσ/dt\nrad/kyr', layout)
        _note(ax, 'sigma_dot not computed', layout.font_note_pt)
        return
    omega0, source = separatrix_omega0(analysis)
    if is_finite(omega0):
        _draw_rate_symlog(ax, frame, layout, 2 * omega0, source)
        return
    _style_axis(ax, 'c', 'dσ/dt\nrad/kyr', layout)
    rate = _edge_line(ax, frame, _rate(frame.prep) * PER_KYR, color=INK, lw=0.8)
    limits = _view_limits(rate)
    low, high = limits if limits is not None else (-1.0, 1.0)
    if high - low >= RATE_DATA_MIN_SHARE * (max(high, 0.0) - min(low, 0.0)):
        low, high = min(low, 0.0), max(high, 0.0)
        rate_zero(ax)
    ax.set_ylim(low, high)
    edge_overflow_markers(ax, frame, _rate(frame.prep) * PER_KYR)
    _rate_marks(ax, frame, low, high, PER_KYR)


def _draw_rate_symlog(ax, frame: _Frame, layout: VerdictFigureLayout, level: float, source: str):
    """(c) as sigma_dot / `level` (= 2 omega0, rad/yr) on the symlog axis (`draw_rate_panel`)."""
    ratio = _edge_line(ax, frame, _rate(frame.prep) / level, color=INK, lw=0.8)
    ax.set_yscale('symlog', linthresh=1.0, linscale=1.0)
    finite = _finite(ratio)
    lo, hi = np.percentile(finite, [0.2, 99.8]) if len(finite) else (0.0, 0.0)
    low = -RATE_SYMLOG_VIEW if lo >= -RATE_SYMLOG_VIEW else lo * RATE_SYMLOG_PAD
    high = RATE_SYMLOG_VIEW if hi <= RATE_SYMLOG_VIEW else hi * RATE_SYMLOG_PAD
    ax.set_ylim(low, high)
    edge_overflow_markers(ax, frame, _rate(frame.prep) / level)
    rate_zero(ax)
    _separatrix_lines(ax, frame, low, high, level / 2 if source == 'theory' else float('nan'), scale=1 / level)
    _rate_marks(ax, frame, low, high, 1 / level)
    _style_axis(ax, 'c', 'dσ/dt\n/ 2ω₀', layout)
    margin_legend(
        ax,
        layout,
        [(('line', MODEL, '--'), f'2ω₀ {source}'), (None, f'{level * PER_KYR:.3g} rad/kyr'), (('line', RATE_ZERO['color'], '-'), 'σ̇ = 0')],
    )
    ax.set_yticks(symlog_ticks(ax, low, high, layout.tick_margin, layout.font_ticks_pt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: _symlog_label(value)))


def rate_zero(ax, vertical: bool = False):
    """dsigma/dt = 0 (`style.RATE_ZERO`): horizontal in the rate panels and portraits, vertical when sigma_dot is on x."""
    (ax.axvline if vertical else ax.axhline)(0, **RATE_ZERO)


def centre_marks(ax, runs, low: float):
    """+ at (sigma_c, 0) of each accepted episode: its center (`centre_mod_2pi_rad`) in the 360-degree window from
    `low`, where a libration's portrait closes around."""
    centres = [run.centre_mod_2pi_rad for run in runs if is_finite(run.centre_mod_2pi_rad)]
    if centres:
        x = np.degrees(_wrap_window(centres, low))
        ax.plot(x, np.zeros(len(x)), '+', color=RATE_ZERO['color'], ms=9, mew=1.6, zorder=6)


def symlog_ticks(ax, low: float, high: float, margin: float, label_pt: float = 0.0) -> List[float]:
    """0, +-1 and the decades +-10^k of the symlog axis of (c) inside the tick margin (axes fraction); with more than
    three decades on a side, every other one from the outermost. With `label_pt` (the tick font), a decade closer to
    the tick kept before it (outward from 0) than SYMLOG_LABEL_GAP labels is dropped, so the labels never overlap
    (W20 round D: 142694 41J-14 at +-10^8 in a 0.6-inch panel)."""
    to_axes = ax.transData + ax.transAxes.inverted()
    height_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72
    gap = SYMLOG_LABEL_GAP * label_pt / height_pt if label_pt > 0 and height_pt > 0 else 0.0
    ticks = [0.0]
    for sign in (-1.0, 1.0):
        decades = [sign * 10.0**k for k in range(0, 40) if low <= sign * 10.0**k <= high]
        decades = decades[:1] + (decades[1:][::-1][::2][::-1] if len(decades) > 4 else decades[1:])
        decades = [v for v in decades if margin <= to_axes.transform((0.0, v))[1] <= 1 - margin]
        last = to_axes.transform((0.0, 0.0))[1]
        for value in decades:  # outward from 0
            position = to_axes.transform((0.0, value))[1]
            if abs(position - last) >= gap:
                ticks.append(value)
                last = position
    return sorted(v for v in ticks if margin <= to_axes.transform((0.0, v))[1] <= 1 - margin)


def _symlog_label(value: float) -> str:
    if value == 0 or abs(value) < 1000:
        return f'{value:g}'
    return rf"${'-' if value < 0 else ''}10^{{{int(round(np.log10(abs(value))))}}}$"


def _rate_marks(ax, frame: _Frame, low: float, high: float, scale: float):
    """nu(lambda_dot) of each regime (dotted) and the slips (○ at 0), sigma_dot in rad/yr times `scale`."""
    for t0, t1, nu in _nu_by_regime(frame):
        _level(ax, nu * scale, low, high, frame.times(t0), frame.times(t1), color=REFERENCE, ls=':', lw=0.9)
    if frame.analysis.geometry is not None:
        slips = [slip.time_yr for slip in frame.analysis.geometry.slips if not slip.suspect]
        ax.plot(frame.times(slips), np.zeros(len(slips)), 'o', ms=4, mfc='none', mec=REFERENCE, zorder=5)


def _separatrix_lines(ax, frame: _Frame, low: float, high: float, theory_omega0: float, scale: float = PER_KYR):
    """+-2 omega0 dashed over each accepted episode with its own omega0; without an episode +-2 omega0 of the theory
    (finite `theory_omega0`) thin across the panel. Levels in rad/yr times `scale`."""
    episodes = episode_omega0(frame.analysis)
    for t0, t1, omega0 in episodes:
        for sign in (-1, 1):
            _level(ax, sign * 2 * omega0 * scale, low, high, frame.times(t0), frame.times(t1), color=MODEL, ls='--', lw=0.8)
    if not episodes and is_finite(theory_omega0):
        for sign in (-1, 1):
            _level(ax, sign * 2 * theory_omega0 * scale, low, high, frame.x[0], frame.x[-1], color=MODEL, ls='--', lw=0.5)


def osculating_shift(context) -> float:
    """<a> - a(lambda_dot) of the whole record [au]: what moves a level of the mean motion (a*, a_res) into the
    osculating frame of a_f; 0 when a(lambda_dot) or <a> is not measured (the levels are then drawn as they are)."""
    if context is None or not (is_finite(context.axis_from_mean_longitude_rate_au) and is_finite(context.a_mean_au)):
        return 0.0
    return float(context.a_mean_au - context.axis_from_mean_longitude_rate_au)


def axis_shifts(frame: _Frame) -> List[Tuple[float, float, float]]:
    """(t0, t1 [yr], <a> - a(lambda_dot) [au]) of each regime between the jumps of a (`regimes`), as nu of (c): the
    osculating offset follows e and i, which a jump changes. One regime, or no saved lambda: the context's value."""
    from resonances.resonance.analysis.mmr.context import axis_from_mean_longitude_rate

    pieces, series = regimes(frame), frame.series
    whole = osculating_shift(frame.analysis.context)
    if len(pieces) == 1 or series.mean_longitude is None:
        return [(float(frame.t[0]), float(frame.t[-1]), whole)]
    result = []
    for t0, t1 in pieces:
        mask = (frame.t >= t0) & (frame.t <= t1)
        a = np.asarray(series.a, dtype=float)[mask]
        axis = axis_from_mean_longitude_rate(frame.t[mask], np.asarray(series.mean_longitude)[mask], a)
        result.append((t0, t1, float(np.mean(a)) - axis if is_finite(axis) and len(a) else whole))
    return result


def axis_levels(analysis: MMRAnalysis, shift: float) -> List[Tuple[str, float, str, str]]:
    """The levels of (d) in the osculating frame: (name, a [au], color, line style) of a* (dotted) and the catalog
    a_res (dashed), each + `shift` (`axis_shifts`); a level that is not known is left out."""
    context = analysis.context
    if context is None:
        return []
    levels = (('a*', context.exact_axis_au, AXIS_EXACT, ':'), ('a_res', context.nominal_axis_au, AXIS_NOMINAL, '--'))
    return [(name, float(value) + shift, colour, style) for name, value, colour, style in levels if is_finite(value)]


def draw_axis_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(d) a on an absolute axis (au) in three levels: the osculating a light gray, a_f at the cutoff P_c mid gray, and
    in black a filtered at the resonance's own scale (`strong_axis_cutoff`; without one, a_f is the black line); a* and
    a_res in the osculating frame of each regime (`axis_shifts`, `axis_levels`) with the band +- half-width around a*,
    the mean a of every cycle. The filtered lines cover the whole record: in EDGE_SPREAD_STYLE outside the analysis
    window (`_edge_line`). The view (option `combined.osculating_a`): `full` (default) holds the osculating band
    (ENVELOPE_PERCENTILES) and a_f, the samples beyond are triangles at the edge at their time (`outlier_markers`);
    `clipped` holds a_f and the black a only (`_axis_view`), an osculating a beyond is one triangle at the edge;
    `hidden` as clipped, without the osculating a when it exceeds the view. a* and a_res are brought into view when
    within AXIS_REFERENCE_SPANS ranges of a_f; a level beyond the axis is a triangle at the edge. Their values are in
    the compact legend right of the panel (`margin_legend`)."""
    analysis = frame.analysis
    filtered = np.asarray(frame.prep.a_f if frame.prep is not None else frame.series.a, dtype=float)
    filtered_inside = np.where(frame.inside, filtered, np.nan)  # outside the window: the filter's edge spread
    raw = np.asarray(frame.series.a, dtype=float)
    strong_yr = strong_axis_cutoff(frame)
    strong = _strong_axis(frame, raw, strong_yr)
    pieces = [(t0, t1, axis_levels(analysis, shift)) for t0, t1, shift in axis_shifts(frame)]
    viewed = filtered_inside[frame.inside]
    if strong is not None:
        viewed = np.concatenate([viewed, strong[0][strong[1]]])
    values = [value for _, _, levels in pieces for _, value, _, _ in levels]
    low, high = _axis_view(viewed, raw, values)
    mode = frame.options['combined']['osculating_a']
    band = envelope_range(raw)
    if mode == OsculatingAxis.FULL and band is not None:
        span = max(band[1], high) - min(band[0], low)
        low, high = min(low, band[0] - ENVELOPE_VIEW_PAD * span), max(high, band[1] + ENVELOPE_VIEW_PAD * span)
    low, high = hold_levels(low, high, values)
    raw_finite = _finite(raw)  # the whole record: the osculating a is drawn over it, edges included
    raw_range = np.percentile(raw_finite, [0.2, 99.8]) if len(raw_finite) else None
    beyond = (raw_range is not None and raw_range[1] > high, raw_range is not None and raw_range[0] < low)
    if not (mode == OsculatingAxis.HIDDEN and any(beyond)):
        _raw(ax, frame.x, raw, alpha=RAW_LIGHT_ALPHA)
    if strong is not None:  # a_f as a band, lighter in the edge spread
        shown, edge = _edge_split(filtered, frame.inside)
        _raw(ax, frame.x, shown)
        _raw(ax, frame.x, edge, alpha=RAW_ALPHA * EDGE_SPREAD_STYLE['alpha'])
        _edge_line(ax, frame, strong[0], inside=strong[1], color=INK, lw=1.0, zorder=3)
    else:
        _edge_line(ax, frame, filtered, color=INK, lw=0.9, zorder=3)
    _axis_levels(ax, frame, pieces, low, high)
    _cycle_means(ax, frame, filtered_inside)
    ax.set_ylim(low, high)
    if strong is not None:
        edge_overflow_markers(ax, frame, strong[0], inside=strong[1])
    else:
        edge_overflow_markers(ax, frame, filtered)
    if mode == OsculatingAxis.FULL:
        outlier_markers(ax, frame.x, raw, low, high)
    else:
        for above, exceeded in zip((True, False), beyond):
            if exceeded:
                edge_marker(ax, EDGE_X_RAW, above, RAW)
    _style_axis(ax, 'd', 'a, au', layout)
    draw_kicks(ax, frame, layout)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    margin_legend(ax, layout, axis_legend(frame, pieces[-1], strong_yr if strong is not None else float('nan'), high - low))


def _axis_levels(ax, frame: _Frame, pieces, low: float, high: float):
    """a* and a_res of each regime (a triangle at the edge when beyond the axis) and the band +- half-width round a*."""
    half_width = _half_width(frame.analysis)
    for t0, t1, levels in pieces:
        x0, x1 = frame.times(t0), frame.times(t1)
        exact = next((value for name, value, _, _ in levels if name == 'a*'), None)
        if exact is not None and is_finite(half_width):
            ax.fill_between([x0, x1], exact - half_width, exact + half_width, color=MODEL, alpha=0.10, lw=0, zorder=0)
        for (name, value, colour, style), at in zip(levels, AXIS_EDGE_AT):
            width, halo_extra = AXIS_LEVEL_STYLE[name]
            halo = [patheffects.withStroke(linewidth=width + halo_extra, foreground='white', alpha=AXIS_LEVEL_ALPHA)]
            line = dict(color=colour, ls=style, lw=width, alpha=AXIS_LEVEL_ALPHA, zorder=AXIS_LEVEL_ZORDER, path_effects=halo)
            _level(ax, value, low, high, x0, x1, at, **line)


def strong_axis_cutoff(frame: _Frame) -> float:
    """Cutoff [yr] of the black a of (d), from the resonance's own scale. With an accepted episode: the response cutoff
    (max(P_c, MMR_RESPONSE_CUTOFF_CYCLES x P_cyc): the a whose derivative the force law of (e) reads) when it is at least
    AXIS_STRONG_MIN_RATIO x P_c; else, when the cycles are too dense for a_f to be a line (narrower than
    AXIS_DENSE_CYCLE_PX), AXIS_STRONG_CYCLES periods of the deciding episode (the slow a under the libration).
    Without an episode, the center-scale window W_c. nan when the choice is not AXIS_STRONG_MIN_RATIO x P_c or more,
    or not needed (then a_f is the black line)."""
    analysis = frame.analysis
    prep, geometry = getattr(analysis, 'prep', None), analysis.geometry
    cutoff = getattr(getattr(prep, 'cutoff', None), 'cutoff_yr', float('nan'))
    if not is_finite(cutoff):
        return float('nan')
    episode = _deciding_episode(analysis)
    if episode is None:
        value = geometry.centre.window_yr if geometry is not None and geometry.centre is not None else float('nan')
    else:
        value = getattr(prep, 'response_cutoff_yr', float('nan'))
        if not (is_finite(value) and value >= AXIS_STRONG_MIN_RATIO * cutoff):
            value = AXIS_STRONG_CYCLES * episode.period_med_yr if frame.cycle_px() < AXIS_DENSE_CYCLE_PX else float('nan')
    return float(value) if is_finite(value) and value >= AXIS_STRONG_MIN_RATIO * cutoff else float('nan')


def _strong_axis(frame: _Frame, raw: np.ndarray, cutoff_yr: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The osculating a low-passed at `cutoff_yr` over the whole record and the mask of its own analysis window: the
    frame's window less FILTER_EDGE_TRIM_CUTOFFS x `cutoff_yr` at each record end (outside it, the filter's edge
    spread); None without a cutoff, when the window is empty, or when the grid cannot be filtered."""
    if not is_finite(cutoff_yr):
        return None
    trim = getattr(frame.settings, 'filter_edge_trim_cutoffs', 1.0) * cutoff_yr
    keep = frame.inside & (frame.t >= frame.t[0] + trim) & (frame.t <= frame.t[-1] - trim)
    if np.count_nonzero(keep) < 2:
        return None
    try:
        smooth = lowpass(frame.t, raw, cutoff_yr, frame.filter_order)
    except ValueError:
        return None
    return smooth, keep


def axis_legend(frame: _Frame, piece, strong_yr: float, span: float) -> list:
    """Items of the compact legend of (d): a_res, a* (their mean-motion values; the lines are moved by the shift),
    +- half-width, the mean osculating a, the shift <a> - a(lambda_dot) of the last regime, and the cutoffs of the
    gray and the black a. Decimals from the axis span."""
    context, (_, _, levels) = frame.analysis.context, piece
    digits = int(np.clip(np.ceil(-np.log10(span)) + 2, 3, 7)) if is_finite(span) and span > 0 else 4
    shift = next((value - context.exact_axis_au for name, value, _, _ in levels if name == 'a*'), float('nan')) if context else float('nan')
    items = []
    names = {'a_res': ('a_res', getattr(context, 'nominal_axis_au', None)), 'a*': ('a*', getattr(context, 'exact_axis_au', None))}
    for name, _, colour, style in levels:
        label, value = names[name]
        items.append((('line', colour, style), f'{label} {value:.{digits}f}'))
    half_width = _half_width(frame.analysis)
    if is_finite(half_width):
        items.append((('band', MODEL, 0.25), f'±Δa {half_width:.2g}'))
    if context is not None and is_finite(context.a_mean_au):
        items.append((None, rf'$\langle a\rangle$ {context.a_mean_au:.{digits}f}'))
    if is_finite(shift):
        items.append((None, rf'$\langle a\rangle - a(\dot\lambda)$ {shift:+.2g}'))
    cutoff = getattr(getattr(frame.prep, 'cutoff', None), 'cutoff_yr', float('nan'))
    if is_finite(strong_yr):
        items.append((('band', RAW, RAW_ALPHA), f'a_f {cutoff:.3g} yr'))
        items.append((('line', INK, '-'), f'a {strong_yr:.3g} yr'))
    elif is_finite(cutoff):
        items.append((('line', INK, '-'), f'a_f {cutoff:.3g} yr'))
    return items


def _axis_view(filtered, raw, levels: Sequence[float]) -> Tuple[float, float]:
    """View of (d): the range of a_f (of the raw a when a_f is all NaN) padded by AXIS_VIEW_PAD, widened to hold
    the levels within AXIS_REFERENCE_SPANS ranges of it."""
    inside = _finite(filtered)
    band = envelope_range(raw)  # without a_f: the band of the raw a, never its extremes (a spike would set the view)
    lo, hi = (float(inside.min()), float(inside.max())) if len(inside) else (band if band is not None else (0.0, 1.0))
    spread = max(hi - lo, 1e-12 * max(abs(hi), 1.0))
    low, high = lo - AXIS_VIEW_PAD * spread, hi + AXIS_VIEW_PAD * spread
    for value in levels:
        if lo - AXIS_REFERENCE_SPANS * spread <= value <= hi + AXIS_REFERENCE_SPANS * spread:
            low, high = min(low, value - 0.1 * spread), max(high, value + 0.1 * spread)
    return low, high


def hold_levels(low: float, high: float, levels: Sequence[float]) -> Tuple[float, float]:
    """The final view of (d) widened once more to hold a* and a_res within AXIS_REFERENCE_SPANS spans of the view
    itself (the osculating band included), so a_res is a line, not an edge triangle, whenever the view is wider than a_f
    alone (author 2026-10-23: 2897 35S-4, a_res 6 a_f ranges below)"""
    span = high - low
    if not (np.isfinite(span) and span > 0):
        return low, high
    for value in levels:
        if np.isfinite(value) and low - AXIS_REFERENCE_SPANS * span <= value <= high + AXIS_REFERENCE_SPANS * span:
            low, high = min(low, value - 0.05 * span), max(high, value + 0.05 * span)
    return low, high


def _cycle_means(ax, frame: _Frame, values):
    """Mean of `values` over each cycle of sigma_f (maximum to maximum), red dots at the cycle's middle."""
    geometry = frame.analysis.geometry
    if geometry is None or frame.prep is None:
        return
    tp = geometry.turning_points
    maxima = np.asarray(tp.indices)[np.asarray(tp.kinds) > 0]
    if 2 <= len(maxima) <= MAX_CYCLE_MEANS:
        mids = [(frame.x[i] + frame.x[j]) / 2 for i, j in zip(maxima[:-1], maxima[1:])]
        means = [np.mean(_finite(values[i:j])) if len(_finite(values[i:j])) else np.nan for i, j in zip(maxima[:-1], maxima[1:])]
        ax.plot(mids, means, 'o', ms=2.8, color=CENTRE, zorder=4)


def draw_adot_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(e) a_dot as the force law read it with the law that decided, on the angle it read (`law_series`): the single
    law R sin(psi - phi0) + C per window, or the multiplet law (`MultipletFit`, one fit over its windows), and the R² of
    the episode halves; psi = sigma - c(t) on an episode accepted in the frame of its moving well (`well_shift`), else
    sigma. When the line rule raised the cutoff the law read (`Response.force_line_binding_yr`), a_dot and the angle
    are taken at that cutoff and the cutoff and the line are printed. When the law that decided is not a law on sigma
    alone, no curve is drawn and the panel prints its numbers: the N-harmonic law (CS12, `decide.taken_harmonic_law`:
    'harmonic law N=k: R² … (single …)') or, for a 1:1 angle, its own law (`coorbital.CoorbitalLaw`). Under the direct
    averaged-force law (W28 B8, `decide.taken_direct_law`) its own curves are drawn (`draw_direct_law`) with its numbers
    (`direct_law_text`), not the single-sine windows."""
    analysis, prep = frame.analysis, frame.prep
    if prep is None:
        _style_axis(ax, 'e', r'$\dot a$', layout)
        _note(ax, 'da/dt not computed', layout.font_note_pt)
        return
    response = analysis.response
    angle_law, adot_law, raised = law_series(frame)
    adot = np.where(frame.inside, adot_law, np.nan)
    power = _power_of_ten(adot)
    scale = 10.0**-power
    label = r'$\dot a$' + '\n' + _multiplier(power, 'au/yr')
    _edge_line(ax, frame, adot_law * scale, color=INK, lw=0.8, zorder=2)  # the edge spread dashed
    _view(ax, adot * scale, extra_top=0.35)
    edge_overflow_markers(ax, frame, adot_law * scale)
    texts = [cutoff_text(response)] if raised else []
    law = getattr(response, 'coorbital', None) if response is not None else None
    if law is not None and response.force_law == 'coorbital':
        # a 1:1 angle is judged by its own law, not by the harmonic windows (which stay as columns): no sine is drawn
        _style_axis(ax, 'e', label, layout)
        numbers = [f'{value:.2f}' if is_finite(value) else '–' for value in (law.r2, law.r2_h1, law.r2_h2, law.scale)]
        _value_text(ax, f'R² {numbers[0]} ({numbers[1]} / {numbers[2]})   K/K_th {numbers[3]}', layout)
        return
    harmonic = taken_harmonic_law(response)
    if harmonic is not None:
        # the N-harmonic law decided L4: the single-sine windows are not the law that was tested, none is drawn
        _style_axis(ax, 'e', label, layout)
        _value_text(ax, '   '.join([harmonic_law_text(harmonic)] + texts), layout)
        return
    direct = taken_direct_law(response)
    if direct is not None:
        # B8: the term's averaged force decided L4; its windows, not the single-sine ones, are the law that was tested
        drawn = draw_direct_law(ax, frame, scale)
        _style_axis(ax, 'e', label, layout)
        words = [direct_law_text(direct, frame.analysis)] + texts + ([] if drawn else ['(curve not rebuilt)'])
        _value_text(ax, '   '.join(words), layout)
        return
    psi = angle_law - _law_shift(frame)
    multiplet = response.multiplet if response is not None and response.force_law == 'multiplet' else None
    if multiplet is not None:
        model = multiplet_model(frame, multiplet, psi)
        if model is not None:
            _line(ax, frame.x, model * scale, color=MODEL, lw=1.0, alpha=0.9, zorder=3)
        texts.insert(0, multiplet_law_text(multiplet, response))
    else:
        windows = response.windows if response is not None else []
        labels = [_draw_window(ax, frame, window, i, len(windows), scale, psi) for i, window in enumerate(windows)]
        if len(windows) <= WINDOW_LABELS_MAX:
            _window_labels(ax, frame, [label for label in labels if label is not None], layout)
    reserved_px = 0.0
    if texts:
        text = '   '.join(texts)
        _value_text(ax, text, layout)
        reserved_px = VALUE_TEXT_X * layout.panels[2] * layout.dpi + _text_width_pt(text, layout.font_note_pt) / 72 * layout.dpi
    _episode_halves(ax, frame, layout, reserved_px)
    _style_axis(ax, 'e', label, layout)
    if response is None:
        _note(ax, 'force law not computed', layout.font_note_pt, y=0.9)


DIRECT_OBSERVED_STYLE = {'color': INK, 'lw': 1.5, 'alpha': 0.85, 'zorder': 3}  # (e): ȧ low-passed at the direct law's cutoff
DIRECT_MODEL_STYLE = {'color': MODEL, 'lw': 1.2, 'alpha': 0.95, 'zorder': 4}  # (e): s × the term's averaged force there


def draw_direct_law(ax, frame: _Frame, scale: float) -> bool:
    """(e) under the direct averaged-force law that decided L4 (W28 B8/E19a, `response.direct_curves`): the observed
    ȧ low-passed at the law's cutoff (thick black) and, per window, s × the term's averaged force low-passed alike
    (blue), each window on its central half (the windows overlap by half; the first and last to their edges), inside
    the analysis window. False when the curves cannot be rebuilt (the panel then prints the numbers only)."""
    analysis = frame.analysis
    if frame.settings is None or analysis.prep is None:
        return False
    curves = response_module.direct_curves(
        frame.series, analysis.prep, analysis.context, analysis.response, analysis.geometry, frame.settings
    )
    if curves is None:
        return False
    _, observed, fits = curves
    t = np.asarray(analysis.prep.t_yr, dtype=float)
    shown = np.zeros(len(t), dtype=bool)
    for part, _ in fits:
        shown[part] = True
    _line(ax, frame.x, np.where(shown & frame.inside, observed, np.nan) * scale, **DIRECT_OBSERVED_STYLE)
    for i, (part, model) in enumerate(fits):
        times = t[part]
        quarter = (times[-1] - times[0]) / 4 if len(times) else 0.0
        start = times[0] + (quarter if i else 0.0) if len(times) else 0.0
        end = times[-1] - (quarter if i < len(fits) - 1 else 0.0) if len(times) else 0.0
        keep = (times >= start) & (times <= end) & frame.inside[part]
        _line(ax, frame.x[part][keep], np.asarray(model)[keep] * scale, **DIRECT_MODEL_STYLE)
    return True


def direct_law_text(law, analysis) -> str:
    """'direct law 8N-13+0+5 (Neptune's mass) at 3.07 kyr: R² 0.75, s 1.14–1.29' (columns rsp_dforce_*)."""
    spec = analysis.resonance
    planet = f" ({spec.planets[0]}'s mass)" if spec is not None and spec.planets else ''
    cutoff = f'{law.cutoff_yr / 1000:.3g} kyr' if law.cutoff_yr >= 1000 else f'{law.cutoff_yr:.3g} yr'
    r2 = f'{law.r2:.2f}' if is_finite(law.r2) else '–'
    scales = f'{law.scale_min:.2f}–{law.scale_max:.2f}' if is_finite(law.scale_min) and is_finite(law.scale_max) else '–'
    return f'direct law {spec.key if spec is not None else ""}{planet} at {cutoff}: R² {r2}, s {scales}'


def law_series(frame: _Frame) -> Tuple[np.ndarray, np.ndarray, bool]:
    """(sigma, a_dot, raised) as the force law that decided read them (`analysis.response` is rule R's response): at
    the cutoff it was measured at (`Response.force_cutoff_yr`: the response cutoff of the witness episode's own period,
    of the pool, or the line rule's raised cutoff), `response.smoothed_series` with the figure's settings. The
    figure's response series (`prep.angle_response`, `prep.adot_response`, at the decisive period) are that series
    when the cutoffs agree, and stand in when the law's cutoff is unknown. A witness that is not the decisive episode
    has its own (W28 g11-7: 34706 7J-2 87 against 75 yr, 22000 5J-2S-2 637 against 538 yr). `raised`: the line
    rule raised the cutoff (`Response.force_line_binding_yr` finite), so the cutoff and the line are printed."""
    prep, response = frame.prep, frame.analysis.response
    base = _angle_response(prep), _adot(prep)
    cutoff = float(response.force_cutoff_yr) if response is not None else float('nan')
    if not is_finite(cutoff) or frame.settings is None:
        return base[0], base[1], False
    raised = is_finite(response.force_line_binding_yr)
    if is_finite(prep.response_cutoff_yr) and np.isclose(cutoff, prep.response_cutoff_yr, rtol=1e-9, atol=0.0):
        return base[0], base[1], raised
    angle, _, _, adot, _ = response_module.smoothed_series(prep, cutoff, frame.settings)
    return np.asarray(angle, dtype=float), np.asarray(adot, dtype=float), raised


def _law_shift(frame: _Frame):
    """c(t) - its mean on the episodes accepted in the frame of their moving well (`geometry.well_shift`), else 0: the
    force law reads psi = sigma - shift (`response.measure_pieces`)."""
    geometry = frame.analysis.geometry
    shift = well_shift(frame.t, geometry.runs) if geometry is not None else None
    return shift if shift is not None else 0.0


def multiplet_model(frame: _Frame, multiplet, psi: np.ndarray) -> Optional[np.ndarray]:
    """The multiplet law (`response.members_model`, its one fit over the windows) on the samples of its windows inside
    the analysis window, NaN elsewhere; None when the fit does not say what it read (no pooled coefficients, planet
    rate or slow varpi)."""
    slow = frame.analysis.slow
    if multiplet.pooled is None or not multiplet.windows or not is_finite(multiplet.planet_varpi_rate_rad_yr) or slow is None:
        return None
    offset = response_module.multiplet_offset(slow, frame.t, multiplet.planet_varpi_rate_rad_yr)
    if offset is None:
        return None
    covered = response_module.intervals_mask(frame.t, multiplet.windows) & frame.inside
    values = response_module.members_model(psi, offset, multiplet.pooled, multiplet.harmonics, multiplet.constant_au_yr)
    return np.where(covered, values, np.nan)


def multiplet_law_text(multiplet, response) -> str:
    """'multiplet law j = 0, 1: R² 0.85 (single 0.62)': the multiplet law that decided L4 (columns rsp_force_R2_multi,
    rsp_force_multi_harmonics) against the single law (the smaller of rsp_force_R2_h1 / h2)."""
    halves = [value for value in (response.force_r2_h1, response.force_r2_h2) if is_finite(value)]
    r2 = f'{multiplet.r2:.2f}' if is_finite(multiplet.r2) else '–'
    single = f'{min(halves):.2f}' if halves else '–'
    return f'multiplet law j = {", ".join(str(j) for j in multiplet.harmonics)}: R² {r2} (single {single})'


def cutoff_text(response) -> str:
    """'cutoff 480 yr (line 160 yr)': the cutoff the line rule raised the force law to (rsp_response_cutoff_ep_yr) and
    the line of a_dot that raised it (rsp_line_binding_yr)."""
    return f'cutoff {response.force_cutoff_yr:.4g} yr (line {response.force_line_binding_yr:.4g} yr)'


def _draw_window(ax, frame: _Frame, window, i: int, n: int, scale: float, psi: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """The force law of one window on its central half (windows overlap by half; the first and last to the edges), on
    the angle `psi` the law read; returns (start, end [display unit], R²) of its label, None without a finite R²."""
    quarter = (window.t_end_yr - window.t_start_yr) / 4
    start = window.t_start_yr if i == 0 else window.t_start_yr + quarter
    end = window.t_end_yr if i == n - 1 else window.t_end_yr - quarter
    mask = (frame.t >= start) & (frame.t < end) & frame.inside
    model = window_model(window, np.asarray(psi, dtype=float)[mask])
    _line(ax, frame.x[mask], model * scale, color=MODEL, lw=1.0, alpha=0.9, zorder=3)
    if not is_finite(window.r2):
        return None
    return float(frame.times(start)), float(frame.times(end)), float(window.r2)


def window_model(window, angle: np.ndarray) -> np.ndarray:
    """The single force law of one window, R sin(sigma - phi0) + C [au/yr], on `angle` [rad]."""
    return window.amplitude_au_yr * np.sin(angle - window.phi0_rad) + window.offset_au_yr


def harmonic_law_text(law) -> str:
    """'harmonic law N=2: R² 0.65 (single 0.18)': the N-harmonic law that decided L4 against the one-harmonic law,
    both the smaller of the two half medians of the window R² (`models.HarmonicLaw`)."""
    r2, single = (f'{value:.2f}' if is_finite(value) else '–' for value in (law.r2, law.single_r2))
    return f'harmonic law N={law.n_harmonics}: R² {r2} (single {single})'


def _r2_text(values: Sequence[float]) -> str:
    """'0.74' for one window, '0.74–0.81' (lowest–highest) for windows whose labels were merged."""
    low, high = f'{min(values):.2f}', f'{max(values):.2f}'
    return low if low == high else f'{low}–{high}'


def _window_labels(ax, frame: _Frame, labels: Sequence[Tuple[float, float, float]], layout: VerdictFigureLayout):
    """The R² of each window under its law; labels whose boxes would overlap become one 'lowest–highest' label over
    their joint span (`merge_overlapping_labels`, the rule of the jump labels in (a))."""
    units_per_pt = layout.dpi / 72 / frame.px_per_unit if frame.px_per_unit > 0 else 0.0
    centres = np.array([0.5 * (start + end) for start, end, _ in labels])

    def values(piece: Tuple[float, float]) -> List[float]:
        inside = (centres >= piece[0]) & (centres <= piece[1])
        return [label[2] for label, keep in zip(labels, inside) if keep]

    def width(piece: Tuple[float, float]) -> float:
        return _text_width_pt(_r2_text(values(piece)), layout.font_note_pt) * units_per_pt

    spans = [(float(c), float(c)) for c in centres]  # a label sits over the middle of its window's drawn half
    for start, end in merge_overlapping_labels(spans, width, gap=WINDOW_LABEL_GAP_PT * units_per_pt):
        ax.text(
            0.5 * (start + end),
            0.03,
            _r2_text(values((start, end))),
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=layout.font_note_pt,
            color=MODEL,
        )


def _episode_halves(ax, frame: _Frame, layout: VerdictFigureLayout, reserved_px: float = 0.0):
    """'#id ½R² h1 / h2' over the middle of each accepted episode, in its color; labels that would overlap the panel
    letter, the law's numbers (`reserved_px`: their right edge, px from the panel's left) or each other are moved right,
    then to a second row, then left out."""
    geometry = frame.analysis.geometry
    if geometry is None:
        return
    physics = {ep.episode_id: ep for ep in frame.analysis.episode_physics}
    width_px = layout.panels[2] * layout.dpi
    char_px = 0.6 * layout.font_note_pt / 72 * layout.dpi
    x0, span = frame.x[0], frame.x[-1] - frame.x[0]
    letter_px = PANEL_LETTER_X * width_px + _text_width_pt('(e)', layout.font_letter_pt) / 72 * layout.dpi
    first = max(letter_px, reserved_px)
    rows_end = [first, first]  # right edge (px) of the last label on each row: the first starts after '(e)' and the text
    for episode in sorted(accepted_runs(geometry), key=lambda run: run.t_start_yr):
        ep = physics.get(episode.episode_id)
        if ep is None:
            continue
        colour = frame.colours.get(episode.episode_id, EPISODE_COLOURS[0])
        middle = frame.times(0.5 * (episode.t_start_yr + episode.t_end_yr))
        ax.axvline(middle, color=colour, lw=0.5, ls=':')
        r2 = (ep.response.force_r2_h1, ep.response.force_r2_h2) if ep.response is not None else (float('nan'),) * 2
        halves = [f'{value:.2f}' if is_finite(value) else '–' for value in r2]
        measured = any(is_finite(v) for v in r2)
        text = f'#{episode.episode_id} ½R² {halves[0]} / {halves[1]}' if measured else f'#{episode.episode_id} R² –'
        size = len(text) * char_px
        centre_px = (middle - x0) / span * width_px if span > 0 else 0.0
        for row, y in enumerate((0.97, 0.84)):
            left = max(centre_px - size / 2, rows_end[row] + char_px, 0.0)
            if left + size <= width_px:
                rows_end[row] = left + size
                ax.text(
                    x0 + (left + size / 2) / width_px * span,
                    y,
                    text,
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='top',
                    fontsize=layout.font_note_pt,
                    color=colour,
                )
                break


def draw_element_panel(ax, frame: _Frame, layout: VerdictFigureLayout, raw, filtered, letter: str, label: str, fmt: str):
    """(f) e, (g) i: the osculating band, the filtered line, 'min – max' of the filtered series in the window."""
    raw = np.asarray(raw, dtype=float)
    _raw(ax, frame.x, raw, alpha=RAW_ALPHA if filtered is not None else 0.9)
    shown = raw[frame.inside]
    if filtered is not None:
        filtered = np.asarray(filtered, dtype=float)
        _line(ax, frame.x, filtered, color=INK, lw=1.0, zorder=3)
        values = _finite(filtered[frame.inside])
        shown = np.concatenate([shown, values])
        if len(values):
            _value_text(ax, f'{fmt.format(values.min())} – {fmt.format(values.max())}', layout)
    _view(ax, shown, extra_top=0.25)
    _style_axis(ax, letter, label, layout)
    ax.yaxis.set_major_locator(_inside_locator(layout.tick_margin, nbins=3))


def draw_eccentricity_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(f) e."""
    prep = frame.prep
    draw_element_panel(ax, frame, layout, frame.series.e, prep.e_f if prep is not None else None, 'f', 'e', '{:.4g}')


def draw_inclination_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(g) i in degrees."""
    prep = frame.prep
    filtered = np.degrees(prep.inc_f) if prep is not None else None
    draw_element_panel(ax, frame, layout, np.degrees(frame.series.inc), filtered, 'g', 'i, °', '{:.4g}°')


def draw_energy_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(h) h per cycle relative to the separatrix on a log axis, so that deep librations (h << 1) and fast
    circulations (h >> 1) are all in view with the separatrix zone [h_lo, h_hi] (a light band) and h = 1 (dashed):
    libration dots, revolutions open, no labels (`h_view`). h below H_FLOOR (or h <= 0) is a triangle at the bottom
    edge; h = inf (a cycle without a potential top) a triangle at the top edge."""
    energy = frame.analysis.energy
    cycles = energy.cycles if energy is not None else []
    if not cycles:
        _style_axis(ax, 'h', 'h', layout)
        _note(ax, 'h not computed: ' + ('no energy analysis' if energy is None else 'no cycles'), layout.font_note_pt)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        return
    zone = getattr(energy, 'zone', None)
    zone = (zone.h_lo, zone.h_hi) if zone is not None and is_finite(zone.h_lo) and is_finite(zone.h_hi) else None
    x = frame.times([0.5 * (cycle.t_start_yr + cycle.t_end_yr) for cycle in cycles])
    h = np.array([cycle.h for cycle in cycles], dtype=float)
    rotation = np.array([cycle.rotation for cycle in cycles], dtype=bool)
    low, high = h_view(h, zone)
    ax.set_yscale('log')
    if zone is not None:
        ax.axhspan(max(zone[0], low), zone[1], color=ZONE_FILL, lw=0, zorder=0)
    ax.axhline(1, color=REFERENCE, lw=0.9, ls='--', zorder=2)
    shown = (h >= low) & (h <= high)  # NaN, inf, and h below the floor are not points
    ax.plot(x[~rotation & shown], h[~rotation & shown], 'o', ms=3.4, color=INK, zorder=3)
    ax.plot(x[rotation & shown], h[rotation & shown], 'o', ms=3.8, mfc='none', mec=INK, mew=1.0, zorder=4)
    for mask, above in ((h > high, True), (h < low, False)):
        if np.any(mask):
            inset = 0.6 * EDGE_MARKER_PT / 72 / max(ax.get_position().height * ax.figure.get_size_inches()[1], 1e-6)
            y = np.full(int(mask.sum()), 1 - inset if above else inset)
            ax.plot(x[mask], y, '^' if above else 'v', ms=4, color=REFERENCE, transform=ax.get_xaxis_transform(), clip_on=False, zorder=4)
    ax.set_ylim(low, high)
    _style_axis(ax, 'h', 'h', layout)
    keys = [(('line', REFERENCE, '--'), 'h = 1 separatrix')]
    if zone is not None:
        keys.append((('band', ZONE_FILL, 1.0), f'zone {zone[0]:.3g}–{zone[1]:.3g}'))
    margin_legend(ax, layout, keys)
    to_axes = ax.transData + ax.transAxes.inverted()
    decades = [10.0**k for k in range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)]
    inside = [v for v in decades if layout.tick_margin <= to_axes.transform((0.0, v))[1] <= 1 - layout.tick_margin]
    ax.set_yticks(_thin_around_one(inside))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))


def _thin_around_one(decades: Sequence[float], keep: int = 4) -> List[float]:
    """At most about `keep` decades of (h): those whose exponent is a multiple of a common step (h = 1 always)."""
    step = max(1, int(np.ceil((len(decades) - 1) / max(keep - 1, 1))))
    return [v for v in decades if round(np.log10(v)) % step == 0]


def h_view(h, zone: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    """The log axis of (h): every finite h above H_FLOOR, the zone and h = 1, widened by H_VIEW_FACTOR on both sides."""
    values = [float(v) for v in np.asarray(h, dtype=float) if np.isfinite(v) and v >= H_FLOOR]
    values += [1.0] + ([float(v) for v in zone if v > 0] if zone is not None else [])
    return max(H_FLOOR, min(values) / H_VIEW_FACTOR), max(values) * H_VIEW_FACTOR


def _relative_degrees(centre, low_reference) -> np.ndarray:
    """A center track minus its circular mean, in degrees wrapped to +-180: a fixed center is flat."""
    values = np.degrees(wrap_pi(np.asarray(centre, dtype=float) - _circular_mean(low_reference)))
    return _break_jumps(values, 180)


def centre_mode(analysis: MMRAnalysis) -> str:
    """What panel (j) shows: 'subresonance', 'owner', 'kinematic' or 'center'. Kinematic: a kinematic libration, or a
    non_resonant row whose slow scale is a secular cycle (slow_origin kinematic, W28 B10b), with a ϖ term in σ."""
    subresonances, owner = analysis.subresonances, analysis.owner
    if subresonances is not None and subresonances.accepted is not None:
        return 'subresonance'
    if owner is not None and owner.accepted is not None:
        return 'owner'
    slow = analysis.slow_scale
    slow_kinematic = slow is not None and slow.origin == SlowOrigin.KINEMATIC and str(analysis.verdict.status) == 'non_resonant'
    kinematic = analysis.verdict.libration_origin == LibrationOrigin.KINEMATIC or slow_kinematic
    arrays = analysis.prep is not None or analysis.slow is not None
    if kinematic and arrays and analysis.resonance.body_varpi_coefficient:
        return 'kinematic'
    return 'centre'


# --------------------------------------------------------------------------------------------- members


@dataclass
class CentreLine:
    """One line of panel (j): the center of an angle [rad] on the record's time grid.

    `role`: 'sigma0' (the catalog angle), 'member' (another member of the multiplet), 'subresonance'
    (the accepted one) or 'owner'. `angle_wrapped` (accepted subresonance or owner only) is the raw angle
    mod 2 pi for the overlay on (b)."""

    key: str
    centre_rad: np.ndarray
    role: str
    angle_wrapped: Optional[np.ndarray] = None


def planar_members(spec: ResonanceSpec) -> List[ResonanceSpec]:
    """The other members of the catalog angle's multiplet that differ from it in the varpi coefficients only: the
    same lambda part and the same node coefficients, the eccentricity coefficients (body and the angle's planets) of
    one sign summing to the same total (d'Alembert holds since the nodes are kept). For a nodal angle
    (`10E-17+0+5[N+2,NE+0]`) the members keep its nodes. None for a retrograde angle or one without eccentricity
    terms."""
    from resonances.resonance.analysis.mmr.subresonances import member_spec

    if spec.retrograde:
        return []
    total = spec.body_varpi_coefficient + sum(spec.planet_varpi_coefficients)
    if total == 0:
        return []
    sign, order, n = (1 if total > 0 else -1), abs(total), len(spec.planets)
    nodes = (spec.body_node_coefficient,) + tuple(spec.planet_node_coefficient(i) for i in range(n))
    members = []
    for parts in itertools.product(range(order + 1), repeat=1 + n):
        if sum(parts) != order:
            continue
        body, planets = sign * parts[0], tuple(sign * u for u in parts[1:])
        if body == spec.body_varpi_coefficient and planets == tuple(spec.planet_varpi_coefficients):
            continue
        members.append(member_spec(spec, (body,) + planets + nodes))
    return members


def varpi_offset(spec: ResonanceSpec, member: ResonanceSpec) -> Optional[Tuple[int, Tuple[int, ...]]]:
    """Delta_s = sigma_member - sigma0 as (coefficient of the body's varpi, of each planet's varpi); None when the
    lambda or node parts differ (a nodal member lives in another frame and is not drawn)."""
    same_lambda = member.planets == spec.planets and member.planet_coefficients == spec.planet_coefficients
    same_nodes = member.body_node_coefficient == spec.body_node_coefficient and all(
        member.planet_node_coefficient(i) == spec.planet_node_coefficient(i) for i in range(len(spec.planets))
    )
    if not (same_lambda and same_nodes and member.body_coefficient == spec.body_coefficient):
        return None
    planets = tuple(m - s for m, s in zip(member.planet_varpi_coefficients, spec.planet_varpi_coefficients))
    return member.body_varpi_coefficient - spec.body_varpi_coefficient, planets


def spec_angle(spec: ResonanceSpec, series: MMRSeries) -> Optional[np.ndarray]:
    """The angle of `spec` mod 2 pi from the saved longitudes of the body and the planets; None if any is missing
    (or for a retrograde angle, whose body longitudes are in another convention)."""
    planets = [series.planets.get(name) for name in spec.planets]
    if spec.retrograde or series.mean_longitude is None or any(p is None or p.mean_longitude is None for p in planets):
        return None
    angle = spec.body_coefficient * series.mean_longitude + spec.body_varpi_coefficient * series.varpi
    angle = angle + spec.body_node_coefficient * series.Omega
    for index, planet in enumerate(planets):
        angle = angle + spec.planet_coefficients[index] * planet.mean_longitude + spec.planet_varpi_coefficients[index] * planet.varpi
        angle = angle + spec.planet_node_coefficient(index) * planet.Omega
    return np.mod(angle, TWO_PI)


def _settings():
    """The AnalysisSettings of the configuration, None when it cannot be read (a figure is drawn without it too)."""
    try:
        from resonances.resonance.analysis.settings import settings_from_config

        return settings_from_config()
    except Exception:  # noqa: BLE001 - a figure is drawn without a readable configuration too
        return None


def _filter_order() -> int:
    """FILTER_ORDER of the configuration (the analysis' center is smoothed with it)."""
    settings = _settings()
    return int(settings.filter_order) if settings is not None else DEFAULT_FILTER_ORDER


def _smooth(t_yr: np.ndarray, values: np.ndarray, window_yr: float, order: int) -> np.ndarray:
    try:
        return lowpass(t_yr, values, window_yr, order)
    except ValueError:  # a non-uniform grid or a window shorter than two steps: drawn unsmoothed
        return np.asarray(values, dtype=float)


def _circular_smooth(t_yr: np.ndarray, angle: np.ndarray, window_yr: float, order: int) -> np.ndarray:
    """Center of an angle that may circulate: the argument of the smoothed unit vector."""
    cos = _smooth(t_yr, np.cos(angle), window_yr, order)
    sin = _smooth(t_yr, np.sin(angle), window_yr, order)
    return np.unwrap(np.arctan2(sin, cos))


def _flatness(centre: np.ndarray, inside: np.ndarray) -> float:
    values = np.unwrap(np.asarray(centre, dtype=float)[inside])
    values = values[np.isfinite(values)]
    return float(np.ptp(values)) if len(values) else float('inf')


def centre_lines(analysis: MMRAnalysis, series: MMRSeries, filter_order: Optional[int] = None) -> List[CentreLine]:
    """The lines of panel (j): sigma0's center, the members of the multiplet, the accepted subresonance, the owner.

    Members are the scanned candidates when the subresonance step ran (best ranked first), else the planar
    members of the same order (flattest first), at most MAX_MEMBER_LINES; c_s = c0 + smooth_Wc(Delta_s). The
    accepted subresonance is drawn with its own measured center when its derived row exists, and its raw angle
    is kept for (b). The accepted owner's center is its angle smoothed over W_c. Empty without a center."""
    geometry = analysis.geometry
    if geometry is None or geometry.centre is None:
        return []
    t = np.asarray(series.t_yr, dtype=float)
    c0 = np.asarray(geometry.centre.centre_rad, dtype=float)
    if len(c0) != len(t):
        return []
    order = DEFAULT_FILTER_ORDER if filter_order is None else filter_order
    window_yr = float(geometry.centre.window_yr)
    inside = (t >= geometry.window[0]) & (t <= geometry.window[1]) & np.isfinite(c0)
    spec = analysis.resonance
    subresonances = analysis.subresonances
    accepted = subresonances.accepted if subresonances is not None else None
    varpis = _smoothed_varpis(series, spec, t, window_yr, order)
    lines = [CentreLine(spec.key, c0, 'sigma0')] + _member_lines(analysis, c0, varpis, inside)
    if accepted is not None:
        line = _accepted_line(analysis, series, accepted.spec, c0, varpis, t)
        if line is not None:
            lines.append(line)
    owner = analysis.owner.accepted if analysis.owner is not None else None
    angle = spec_angle(owner.spec, series) if owner is not None else None
    key = owner.spec.key if owner is not None else ''
    if owner is not None and angle is None and FLAG_CATALOGUE in analysis.owner.flags:  # found on catalogue planet lines
        angle, key = owner_wrapped_angle(owner.spec, series), f'{key} catalog λ (approx.)'
    if angle is not None:
        lines.append(CentreLine(key, _circular_smooth(t, angle, window_yr, order), 'owner', angle))
    return lines


def _offset_track(offset: Tuple[int, Tuple[int, ...]], varpis: Sequence[np.ndarray]) -> np.ndarray:
    """Delta_s(t) from its varpi coefficients and the (smoothed) varpi of the body and the planets [rad]."""
    return offset[0] * varpis[0] + sum(coefficient * varpi for coefficient, varpi in zip(offset[1], varpis[1:]))


def _member_lines(analysis: MMRAnalysis, c0: np.ndarray, varpis, inside: np.ndarray) -> List[CentreLine]:
    """c0 + smooth(Delta_s) of the scanned candidates (best ranked first) or, without a scan, of the planar
    members (flattest first); the accepted subresonance is drawn apart."""
    if varpis is None:
        return []
    spec, subresonances = analysis.resonance, analysis.subresonances
    scanned = subresonances is not None and bool(subresonances.candidates)
    accepted = subresonances.accepted.spec.key if subresonances is not None and subresonances.accepted is not None else None
    if scanned:
        candidates = [c.spec for c in sorted(subresonances.candidates, key=lambda c: c.rank) if not c.nodal]
    else:
        candidates = planar_members(spec)
    lines = []
    for member in candidates:
        offset = varpi_offset(spec, member)
        if offset is not None and member.key != accepted:
            lines.append(CentreLine(member.key, c0 + _offset_track(offset, varpis), 'member'))
    if not scanned:
        lines.sort(key=lambda line: _flatness(line.centre_rad, inside))
    return lines[:MAX_MEMBER_LINES]


def _smoothed_varpis(series: MMRSeries, spec: ResonanceSpec, t, window_yr: float, order: int) -> Optional[List[np.ndarray]]:
    """[smooth(varpi), smooth(varpi_P1), ...] unwrapped [rad]; None without the planets' varpi."""
    planets = [series.planets.get(name) for name in spec.planets]
    if spec.retrograde or any(p is None or p.varpi is None for p in planets):
        return None
    arrays = [series.varpi] + [p.varpi for p in planets]
    return [_smooth(t, np.unwrap(np.asarray(values, dtype=float)), window_yr, order) for values in arrays]


def _accepted_line(analysis: MMRAnalysis, series: MMRSeries, member: ResonanceSpec, c0, varpis, t) -> Optional[CentreLine]:
    derived = next((item for item in analysis.derived if item.resonance.key == member.key and item.geometry is not None), None)
    if derived is None and len(analysis.derived) == 1 and analysis.derived[0].geometry is not None:
        derived = analysis.derived[0]
    offset = varpi_offset(analysis.resonance, member)
    angle = None
    if derived is not None and derived.derived_series is not None:
        angle = np.mod(np.asarray(derived.derived_series.angle_wrapped, dtype=float), TWO_PI)
    elif offset is not None and series.planets:
        raw = [series.varpi] + [series.planets[name].varpi for name in analysis.resonance.planets if name in series.planets]
        if len(raw) == 1 + len(offset[1]):
            delta = offset[0] * raw[0] + sum(c * np.asarray(v, dtype=float) for c, v in zip(offset[1], raw[1:]))
            angle = np.mod(np.asarray(series.angle_wrapped, dtype=float) + delta, TWO_PI)
    if derived is not None:
        centre = np.asarray(derived.geometry.centre.centre_rad, dtype=float)
        if len(centre) == len(t):
            return CentreLine(member.key, centre, 'subresonance', angle)
    if offset is None or varpis is None:
        return None
    return CentreLine(member.key, c0 + _offset_track(offset, varpis), 'subresonance', angle)


# --------------------------------------------------------------------------------------------- (i), (j)


def centre_pass_matters(centre) -> bool:
    """The largest monotone pass of the center is shaded only when it is a whole revolution or more (the center
    circulates: what G4, G1c, B5 and the secular centre_pass read); a shorter pass decides nothing."""
    times = is_finite(centre.max_pass_t0_yr) and is_finite(centre.max_pass_t1_yr)
    return bool(times and is_finite(centre.max_pass_rad) and centre.max_pass_rad >= TWO_PI)


def draw_centre_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(i) the center of sigma: sigma_f smoothed over W_c drawn over sigma_f, the center's turns and its largest
    monotone pass when it is a revolution or more (`centre_pass_matters`; the numbers geo_centre_window_yr,
    geo_centre_n_tp, geo_centre_max_pass_rev); the slow scale's intervals hatched (`draw_slow_intervals`)."""
    geometry, prep = frame.analysis.geometry, frame.prep
    _style_axis(ax, 'i', 'center\nrad', layout)
    if frame.centre is None or geometry is None:
        _note(ax, 'center not computed', layout.font_note_pt)
        return
    centre = geometry.centre
    if prep is not None:
        _line(ax, frame.x, np.where(frame.inside, prep.angle_f, np.nan), color=RAW, alpha=RAW_ALPHA, lw=0.5, zorder=1)
    centre_in = np.where(frame.inside, frame.centre, np.nan) if len(frame.centre) == len(frame.t) else frame.centre
    _line(ax, frame.times(frame.centre_t), centre_in, color=CENTRE, lw=1.3, zorder=3)
    turns = centre.turns
    if turns is not None and 0 < len(turns.times_yr) <= MAX_TP_MARKERS:
        turn_markers(ax, frame.times(turns.times_yr), turns.values_rad, turns.kinds, CENTRE_TURN_SIZE, CENTRE, white_edge=True)
    if centre_pass_matters(centre):
        ax.axvspan(frame.times(centre.max_pass_t0_yr), frame.times(centre.max_pass_t1_yr), color=CENTRE, alpha=0.1, lw=0, zorder=0)
    _view(ax, prep.angle_f[frame.inside] if prep is not None else centre_in, pad=0.08)
    draw_slow_intervals(ax, frame, layout)


def slow_intervals(analysis: MMRAnalysis) -> List[Tuple[float, float, str]]:
    """(t0, t1, label) [yr] of the slow scale (B10a `SlowScale`; W28 B10b, card g08-7: 'the period and the episodes of
    the slow libration visible and checkable'): each slow episode 'slow #id' (episodes.csv scale='slow'), else the span
    of the slow center's closed cycles 'slow'. Empty without a slow scale or without a slow swing (half-amplitude nan)."""
    slow = analysis.slow_scale
    if slow is None or not slow.measured or not is_finite(slow.halfamp_rad):
        return []
    if slow.episodes:
        return [(float(run.t_start_yr), float(run.t_end_yr), f'{SLOW_LABEL} #{run.episode_id}') for run in slow.episodes]
    if is_finite(slow.t_start_yr) and is_finite(slow.t_end_yr) and slow.t_end_yr > slow.t_start_yr:
        return [(float(slow.t_start_yr), float(slow.t_end_yr), SLOW_LABEL)]
    return []


def draw_slow_intervals(ax, frame: _Frame, layout: VerdictFigureLayout) -> int:
    """The slow intervals on (i) (`slow_intervals`): hatched `SLOW_HATCH` in the center's color, the label at the top
    when the span is at least EPISODE_LABEL_MIN_PX wide. Returns the number drawn."""
    intervals = slow_intervals(frame.analysis)
    for t0, t1, label in intervals:
        x0, x1 = frame.times(t0), frame.times(t1)
        ax.axvspan(x0, x1, facecolor='none', edgecolor=CENTRE, hatch=SLOW_HATCH, alpha=SLOW_HATCH_ALPHA, lw=0, zorder=0.5)
        if (x1 - x0) * frame.px_per_unit >= EPISODE_LABEL_MIN_PX:
            ax.text(
                0.5 * (x0 + x1),
                0.97,
                label,
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=layout.font_note_pt,
                color=CENTRE,
                zorder=9,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.7, 'pad': 0.4},
            )
    return len(intervals)


def draw_centres_panel(ax, frame: _Frame, layout: VerdictFigureLayout):
    """(j) the centers of sigma0, the members of the multiplet and the owner, each minus its circular mean, with
    their keys at the right (no caption: why a check did not run is in the right column); the kinematic terms
    instead when the libration is kinematic."""
    analysis = frame.analysis
    mode = centre_mode(analysis)
    _style_axis(ax, 'j', 'center − mean\n°', layout)
    if frame.centre is None:
        _note(ax, 'centers not computed: no center of σ', layout.font_note_pt)
        return
    if mode == 'kinematic':
        _draw_kinematic(ax, frame, layout)
        return
    items = []
    overlay = varpi_overlay(frame)  # B10b: p varpi where the slow cause is secular (card g08-8, 'here and everywhere')
    for i, line in enumerate(frame.centre_lines):
        colour, width, style, z = _line_style(line, i)
        values = np.where(frame.inside, _relative_degrees(line.centre_rad, line.centre_rad[frame.inside]), np.nan)
        _line(ax, frame.x, values, color=colour, lw=width, ls=style, zorder=z)
        finite = values[np.isfinite(values)]
        if len(finite):
            items.append((finite[-1], line.key, colour))
    if overlay is not None:
        items += draw_varpi_overlay(ax, frame, overlay)
    ax.axhline(0, color=REFERENCE, lw=0.5, alpha=0.5, zorder=0)
    ax.set_ylim(-CENTRES_AXIS_DEG, CENTRES_AXIS_DEG)
    ax.set_yticks([-180, -90, 0, 90, 180])
    _right_labels(ax, [(float(np.clip(value, -180.0, 180.0)), text, colour) for value, text, colour in items], layout.font_note_pt)


def _line_style(line: CentreLine, index: int) -> Tuple[str, float, str, int]:
    if line.role == 'sigma0':
        return CENTRE, 1.3, '-', 5
    if line.role == 'subresonance':
        return SUBRESONANCE, 1.5, '-', 6
    if line.role == 'owner':
        return OWNER, 1.3, '--', 6
    return MEMBER_COLOURS[(index - 1) % len(MEMBER_COLOURS)], 1.1, '-', 3


def _detrended(t, y) -> np.ndarray:
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    return y - np.polyval(np.polyfit(t - t[0], y, 1), t - t[0]) if len(t) > 2 else y - np.mean(y)


@dataclass
class KinematicTerms:
    """The terms of sigma on one scale for panel (j) (W28 B10b, cards g05-3 'σ = λ part + pϖ on this scale', g08-8):
    sigma, p varpi and the lambda part sigma - p varpi [rad] (the planets' slow terms go with it), each low-passed at
    `cutoff_yr` and detrended over each span of `t_yr` (nan between spans). `scale`: 'episode' (σ's decisive episode
    at its response cutoff, where K1e read kin_swing_share), 'slow' (the slow scale, `SlowScale.cutoff_yr`: K3s and
    the p·ϖ overlay) or 'center' (the slow components at W_c, when neither applies). A drawing aid: no status reads it."""

    t_yr: np.ndarray
    sigma: np.ndarray
    p_varpi: np.ndarray
    lambda_part: np.ndarray
    p: int
    scale: str
    cutoff_yr: float


def slow_spans(slow) -> List[Tuple[float, float]]:
    """The spans [yr] of a slow scale the kinematic terms are detrended over, one by one: its episodes (a slow
    libration holds within each; between them σ may slip a revolution, 37087 over 1 Myr), else its whole span."""
    if slow.episodes:
        return [(float(run.t_start_yr), float(run.t_end_yr)) for run in slow.episodes]
    return [(float(slow.t_start_yr), float(slow.t_end_yr))]


def kinematic_scale(analysis: MMRAnalysis, row: Mapping[str, Any], settings) -> Tuple[str, List[Tuple[float, float]], float]:
    """(scale, spans [yr], cutoff [yr]) of the kinematic terms of (j): the slow scale's spans and cutoff when the
    kinematic swing is the slow scale's (`verdict_text.slow_kinematic`, K3s: 37087 3J+8S-3 at 506 yr; or a non_resonant
    row drawn kinematic for its slow scale alone, 285637 over 2 Myr); σ's decisive
    episode (first to last turning point; the analysis window for a libration over the record) at its response cutoff
    max(P_c, mmr_response_cutoff_cycles x P) when K1e measured the swing there (kin_swing_share; 285637 at 1.8 kyr,
    where W_c = 54 kyr would erase its 18-kyr cycle); else ('center', [], nan)."""
    none = ('centre', [], NAN)
    slow = analysis.slow_scale
    slow_swing = verdict_text.slow_kinematic(row, settings.mmr_kin_period_tol)
    # a non_resonant row drawn kinematic for its slow scale alone (`centre_mode`): that scale too
    slow_swing = slow_swing or (
        slow is not None and slow.origin == SlowOrigin.KINEMATIC and analysis.verdict.libration_origin != LibrationOrigin.KINEMATIC
    )
    if slow is not None and slow.measured and slow_swing and is_finite(slow.cutoff_yr):
        return 'slow', slow_spans(slow), float(slow.cutoff_yr)
    prep, geometry, kinematics = analysis.prep, analysis.geometry, analysis.kinematics
    if prep is None or geometry is None or kinematics is None or not is_finite(kinematics.swing_share):
        return none
    run = _deciding_episode(analysis)
    whole = str(analysis.verdict.angle_status) == 'libration'
    if run is not None and not whole and len(run.tp_times_yr) >= 2:
        window = (float(run.tp_times_yr[0]), float(run.tp_times_yr[-1]))
    else:
        window = (float(prep.window[0]), float(prep.window[1]))
    period = run.period_med_yr if run is not None and is_finite(run.period_med_yr) else geometry.period_cycle_yr
    return 'episode', [window], float(response_module.response_cutoff_yr(prep.cutoff.cutoff_yr, period, settings))


def kinematic_terms(
    analysis: MMRAnalysis, series: MMRSeries, settings, spans: Sequence[Tuple[float, float]], cutoff_yr: float, scale: str
) -> Optional[KinematicTerms]:
    """`KinematicTerms` at `cutoff_yr` over `spans`, as `kinematics.episode_kinematics` builds them: prep's unwrapped
    sigma and p x the body's varpi unwrapped with its slow vector (`sampling.body_varpi_slow_unwrap`), low-passed,
    detrended over each span on its own (a nan between two spans, so no line joins them); the lambda part their
    difference. None without a varpi term, for a retrograde angle (no varpi
    convention), without the arrays, or when the low-pass cannot run or the window holds fewer than 3 samples."""
    prep, spec = analysis.prep, analysis.resonance
    p = int(spec.body_varpi_coefficient)
    if prep is None or not p or spec.retrograde or not (is_finite(cutoff_yr) and cutoff_yr > 0):
        return None
    t = np.asarray(prep.t_yr, dtype=float)
    varpi = body_varpi_slow_unwrap(series, settings)[0]
    source = np.asarray(series.t_yr, dtype=float)
    if len(source) != len(t) or not np.array_equal(source, t):
        varpi = np.interp(t, source, varpi)
    sigma_f = lowpass_or_none(t, np.asarray(prep.angle_unwrapped, dtype=float), cutoff_yr, settings)
    p_varpi_f = lowpass_or_none(t, p * varpi, cutoff_yr, settings)
    if sigma_f is None or p_varpi_f is None:
        return None
    pieces = []
    for start, end in spans:
        inside = (t >= start) & (t <= end) & np.isfinite(sigma_f) & np.isfinite(p_varpi_f)
        if np.count_nonzero(inside) < 3:
            continue
        t_in = t[inside]
        if pieces:  # a gap between two spans
            pieces.append((np.array([0.5 * (pieces[-1][0][-1] + t_in[0])]), np.array([np.nan]), np.array([np.nan])))
        pieces.append((t_in, _detrended(t_in, sigma_f[inside]), _detrended(t_in, p_varpi_f[inside])))
    if not pieces:
        return None
    t_all, sigma, p_varpi = (np.concatenate([piece[i] for piece in pieces]) for i in range(3))
    return KinematicTerms(t_all, sigma, p_varpi, sigma - p_varpi, p, scale, float(cutoff_yr))


def _centre_terms(frame: _Frame) -> Optional[KinematicTerms]:
    """The terms of the slow components at W_c (`analysis.slow`), the fallback of `frame_kinematic_terms`."""
    analysis, slow = frame.analysis, frame.analysis.slow
    p = int(analysis.resonance.body_varpi_coefficient)
    if slow is None or not p:
        return None
    t = np.asarray(slow.t_yr, dtype=float)
    mask = (t >= slow.window[0]) & (t <= slow.window[1])
    if np.count_nonzero(mask) < 3:  # a record shorter than two centre windows: the analysis window
        mask = (t >= frame.window[0]) & (t <= frame.window[1])
    if np.count_nonzero(mask) < 3:
        return None
    sigma = _detrended(t[mask], np.asarray(slow.angle_slow, dtype=float)[mask])
    p_varpi = _detrended(t[mask], p * np.asarray(slow.varpi_slow, dtype=float)[mask])
    return KinematicTerms(t[mask], sigma, p_varpi, sigma - p_varpi, p, 'centre', float(slow.window_yr))


def frame_kinematic_terms(frame: _Frame) -> Optional[KinematicTerms]:
    """The kinematic terms of (j) at `kinematic_scale`, else at the center window W_c (`_centre_terms`; also without
    readable settings)."""
    if frame.settings is None:
        return _centre_terms(frame)
    scale, spans, cutoff = kinematic_scale(frame.analysis, frame.row, frame.settings)
    if scale != 'centre':
        terms = kinematic_terms(frame.analysis, frame.series, frame.settings, spans, cutoff, scale)
        if terms is not None:
            return terms
    return _centre_terms(frame)


def varpi_period(frame: _Frame, scale: str) -> float:
    """The secular period [yr] the p·ϖ line of (j) is labeled with, from the columns: on the slow scale the slow
    cause's term (`verdict_text.slow_secular_cause`), else its matched term (slow_secular_term); on σ's own scale the
    secular period a secular_cycle verdict matched (`verdict_text.closest_secular_period`); nan otherwise."""
    row = frame.row
    if scale == 'slow':
        period = verdict_text.slow_secular_cause(row)[1]
        return period if is_finite(period) else verdict_text.secular_term_period(row, row.get('slow_secular_term'))
    if str(frame.analysis.verdict.origin_detail) != 'secular_cycle':
        return NAN
    period = row.get('ep_period_med_yr')
    period = period if verdict_text.finite(period) else row.get('geo_period_cycle_yr')
    return verdict_text.closest_secular_period(row, period)[0]


def varpi_label(p: int, period_yr: float) -> str:
    """'−8ϖ 5.70 kyr': the term and the secular period it is drawn with; the term alone without a period."""
    term = verdict_text.varpi_term(p)
    return f'{term} {verdict_text.years(period_yr)}' if is_finite(period_yr) else term


def _draw_kinematic(ax, frame: _Frame, layout: VerdictFigureLayout):
    """Kinematic mode (W28 B10b): σ = λ part + pϖ on the scale of the kinematic swing (`frame_kinematic_terms`), each
    detrended, in degrees: red σ, orange dashed pϖ labeled with its secular period (`varpi_label`), gray dotted λ part.
    A kinematic libration follows pϖ and its λ part stays flat (kin_swing_share, kin_lambda_swing_share)."""
    terms = frame_kinematic_terms(frame)
    if terms is None:
        _note(ax, 'kinematic terms not computed', layout.font_note_pt)
        return
    x = frame.times(terms.t_yr)
    lines = (
        (np.degrees(terms.sigma), 'σ', CENTRE, {'lw': 1.1, 'zorder': 5}),
        (
            np.degrees(terms.p_varpi),
            varpi_label(terms.p, varpi_period(frame, terms.scale)),
            SUBRESONANCE,
            {'lw': 1.0, 'ls': '--', 'zorder': 6},
        ),
        (np.degrees(terms.lambda_part), LAMBDA_PART_LABEL, LAMBDA_PART, {'lw': 0.9, 'ls': ':', 'zorder': 4}),
    )
    for values, _, colour, style in lines:
        _line(ax, x, values, color=colour, **style)
    ax.axhline(0, color=REFERENCE, lw=0.5, alpha=0.5, zorder=0)
    ax.set_ylabel('detrended\n°', fontsize=layout.font_axis_pt, labelpad=2)
    _view(ax, np.concatenate([values for values, _, _, _ in lines]))
    _right_labels(ax, [(values[-1], text, colour) for values, text, colour, _ in lines if len(values)], layout.font_note_pt)


def varpi_overlay(frame: _Frame) -> Optional[Tuple[KinematicTerms, float]]:
    """(the terms on the slow scale, the secular period [yr]) of the p·ϖ overlay of (j) outside the kinematic mode
    (W28 B10b, card g08-8: 'if the cause is a secular cycle, show it as a line, here and everywhere'): only when the
    slow scale names a secular cause (slow_cause secular_2omega/secular_g_gj, any level) and has a swing; None
    otherwise (nothing is drawn without a slow swing)."""
    slow = frame.analysis.slow_scale
    if slow is None or not slow.measured or str(slow.cause) not in verdict_text.SECULAR_CAUSES or not is_finite(slow.halfamp_rad):
        return None
    if frame.settings is None:
        return None
    terms = kinematic_terms(frame.analysis, frame.series, frame.settings, slow_spans(slow), float(slow.cutoff_yr), 'slow')
    return (terms, varpi_period(frame, 'slow')) if terms is not None else None


def draw_varpi_overlay(ax, frame: _Frame, overlay: Tuple[KinematicTerms, float]) -> List[Tuple[float, str, str]]:
    """The detrended p·ϖ of the slow scale on (j) in degrees (wrapped to ±180°, orange dashed, as in the kinematic
    mode); returns its right label (value, '−9ϖ 7.16 kyr', color)."""
    terms, period = overlay
    values = _break_jumps(np.degrees(wrap_pi(terms.p_varpi)), 180)
    _line(ax, frame.times(terms.t_yr), values, color=SUBRESONANCE, lw=1.0, ls='--', zorder=6)
    finite = values[np.isfinite(values)]
    return [(float(finite[-1]), varpi_label(terms.p, period), SUBRESONANCE)] if len(finite) else []


WRAPPED_INDEX = 1  # (b), the wrapped angle: its time axis is labelled too


def panel_rects(layout: VerdictFigureLayout, n: Optional[int] = None, labelled: Optional[int] = None) -> List[Rect]:
    """Rectangles (inches) of the time panels, top to bottom: `layout.panels` split evenly, `panel_gap` between, and
    `time_label_gap` more under the panel `labeled` (index), whose time axis carries tick labels."""
    n = n or layout.n_panels
    x, y, w, h = layout.panels
    extra = layout.time_label_gap if labelled is not None and 0 <= labelled < n - 1 else 0.0
    height = (h - (n - 1) * layout.panel_gap - extra) / n
    rects, top = [], y + h
    for i in range(n):
        rects.append((x, top - height, w, height))
        top -= height + layout.panel_gap + (extra if i == labelled else 0.0)
    return rects


def stacked_axes(fig, layout: VerdictFigureLayout, n: int, rects: Optional[Sequence[Rect]] = None, labelled: Optional[int] = None) -> list:
    """The time panels as axes sharing t (`panel_rects`, or the given `rects`)."""
    width, height = fig.get_size_inches()
    axes = []
    for x, y, w, h in rects if rects is not None else panel_rects(layout, n, labelled):
        axes.append(fig.add_axes((x / width, y / height, w / width, h / height), sharex=axes[0] if axes else None))
    return axes


def label_time_axes(axes: Sequence, labelled: int, layout: VerdictFigureLayout, unit: str) -> None:
    """Tick labels of t on the last panel (with the axis title) and again under `axes[labeled]`, none elsewhere."""
    for index, ax in enumerate(axes[:-1]):
        ax.tick_params(labelbottom=index == labelled)
    axes[-1].set_xlabel(f't, {unit}', fontsize=layout.font_axis_pt, labelpad=1)


def draw_time_panels(
    fig, analysis: MMRAnalysis, series: MMRSeries, layout: VerdictFigureLayout, options=None, settings=None, row=None
) -> list:
    """The time panels of PANELS sharing the time axis (kyr or Myr by `base.time_unit`); returns their axes. When the
    cycles are too dense for (b) (`needs_strip`), the column is split in eleven and the third slot holds the strip
    (b′) (`draw_strip`, axes created after the ten panels, not returned); otherwise ten equal panels.

    `options`: plot options (`style.DEFAULT_PLOT_OPTIONS`; the figure reads `combined`); `settings`: the
    AnalysisSettings of the episodes' coverage rule (default: from the configuration); `row`: the numbers ⋈ summary
    row whose columns label (j) (default: from the analysis, when needed)."""
    frame = _Frame(analysis, series, layout, options, settings, row)
    strip = needs_strip(frame)
    rects = panel_rects(layout, len(PANELS) + 1, WRAPPED_INDEX) if strip else panel_rects(layout, len(PANELS), WRAPPED_INDEX)
    strip_rect = rects.pop(2) if strip else None
    axes = stacked_axes(fig, layout, len(PANELS), rects)
    drawers = (
        draw_angle_panel,
        draw_wrapped_panel,
        draw_rate_panel,
        draw_axis_panel,
        draw_adot_panel,
        draw_eccentricity_panel,
        draw_inclination_panel,
        draw_energy_panel,
        draw_centre_panel,
        draw_centres_panel,
    )
    for ax, drawer in zip(axes, drawers):
        drawer(ax, frame, layout)
    label_time_axes(axes, WRAPPED_INDEX, layout, frame.unit)
    axes[-1].set_xlim(frame.x[0], frame.x[-1])
    if strip_rect is not None:
        draw_strip(fig, strip_rect, frame, layout, axes[1])
    return axes


# --------------------------------------------------------------------------------------------- bottom row


def _bottom_title(ax, text: str, layout: VerdictFigureLayout):
    ax.text(0.0, 1.02, text, transform=ax.transAxes, ha='left', va='bottom', fontsize=layout.font_letter_pt, fontweight='bold')
    ax.tick_params(labelsize=layout.font_ticks_pt, length=2.5, pad=1.5)


def _empty(ax, text: str, layout: VerdictFigureLayout, crossed: bool = False):
    """An empty frame with the reason; `crossed` draws the cross of a portrait that would mislead."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#dddddd')
    if crossed:
        for ys in ((0, 1), (1, 0)):
            ax.plot((0, 1), ys, transform=ax.transAxes, color='#dddddd', lw=1.0, zorder=0)
    ax.text(
        0.5,
        0.5,
        textwrap.fill(text, 34),
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=layout.font_note_pt + 1,
        color=NOTE,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2},
    )


def _alias_reason(analysis: MMRAnalysis) -> str:
    return f'not drawn: the samples are aliased ({analysis.sampling.verdict}); a portrait of them shows an island that is not there'


def _angle_axis(ax, low: float, layout: VerdictFigureLayout, label: str):
    base = np.degrees(low)
    ax.set_xlim(base, base + 360)
    ax.set_xticks(_right_angles(base))
    ax.xaxis.set_major_formatter(_angle_formatter())
    rate_zero(ax)
    ax.set_xlabel(label, fontsize=layout.font_axis_pt, labelpad=1)
    ax.set_ylabel('dσ/dt, rad/kyr', fontsize=layout.font_axis_pt, labelpad=1)
    ax.grid(alpha=0.15, lw=0.5)


def portrait_cycles(run, tp_times, tp_kinds, max_cycles: int = PORTRAIT_MAX_CYCLES) -> List[Tuple[float, float]]:
    """The spans of an episode drawn in (l): the whole episode when it holds at most `max_cycles` cycles (maximum to
    maximum of sigma_f), so no sample of it is left out (its partial cycles at either end included); else
    `max_cycles` of its cycles, evenly spaced over it."""
    maxima = np.asarray(tp_times, dtype=float)[np.asarray(tp_kinds) > 0]
    maxima = maxima[(maxima >= run.t_start_yr) & (maxima <= run.t_end_yr)]
    bounds = list(zip(maxima[:-1], maxima[1:]))
    if len(bounds) <= max_cycles:
        return [(run.t_start_yr, run.t_end_yr)]
    chosen = np.unique(np.linspace(0, len(bounds) - 1, min(max_cycles, len(bounds))).astype(int))
    return [bounds[i] for i in chosen]


def portrait_span(run, spans: Mapping[int, Tuple[float, float]], window: Tuple[float, float]) -> Tuple[float, float]:
    """(start, end) [yr] of an accepted run in the portraits: its fill span in (a) (`episode_spans`; first to last
    turning point when it has none) clipped to the analysis window of (k) (W28 g03-6, the author 2026-10-31: (l) over
    the interval of the fill, the extended edges in another style). An episode closed by its returns over the whole
    record is the whole window, so (l) holds the samples of (k). (end <= start: nothing inside the window.)"""
    start, end = spans.get(run.episode_id, (run.t_start_yr, run.t_end_yr))
    return max(float(start), float(window[0])), min(float(end), float(window[1]))


def portrait_pieces(run, span: Tuple[float, float], tp_times, tp_kinds, max_cycles: int = PORTRAIT_MAX_CYCLES):
    """The pieces of an episode drawn in (l), each (t0, t1, extended): the cycles of `portrait_cycles` inside the
    turning points (clipped to `span`, `portrait_span`), and the extensions of the fill beyond the first and the last
    turning point (extended=True, drawn dashed), so the drawn interval is the fill's."""
    start, end = span
    if end <= start:
        return []
    core0, core1 = max(start, run.t_start_yr), min(end, run.t_end_yr)
    pieces = [(t0, t1, False) for t0, t1 in portrait_cycles(run, tp_times, tp_kinds, max_cycles)]
    pieces = [(max(t0, core0), min(t1, core1), False) for t0, t1, _ in pieces if min(t1, core1) > max(t0, core0)]
    if start < core0:
        pieces.insert(0, (start, min(core0, end), True))
    if end > core1:
        pieces.append((max(core1, start), end, True))
    return pieces


def _episodes_window_low(geometry, runs, t, angle) -> float:
    """Lower edge of the 360-degree window of (l): around the center track over the episodes (a horseshoe's samples
    crowd its two ends, so their own mean would split it), else around sigma_f there."""
    centres = []
    for run in runs:
        track = geometry.centre_covering(run.t_start_yr)
        track_t = np.asarray(track.t_yr, dtype=float)
        centres.append(np.asarray(track.centre_rad, dtype=float)[(track_t >= run.t_start_yr) & (track_t <= run.t_end_yr)])
    values = np.concatenate(centres) if centres else np.array([])
    if not len(_finite(values)):
        covered = np.zeros(len(t), dtype=bool)
        for run in runs:
            covered |= (t >= run.t_start_yr) & (t <= run.t_end_yr)
        values = angle[covered]
    return _circular_mean(values) - np.pi


def _portrait_episodes(accepted, counted: Optional[frozenset] = None) -> list:
    """The PORTRAIT_MAX_EPISODES longest accepted runs, in time order (the episodes of (l)); with the minimal
    policy's `counted` ids its counted episodes first."""
    counted = counted or frozenset()
    longest = sorted(accepted, key=lambda run: (run.episode_id not in counted, run.t_start_yr - run.t_end_yr))
    return sorted(longest[:PORTRAIT_MAX_EPISODES], key=lambda run: run.t_start_yr)


def _policy_tag(episode_id, marks: PolicyMarks) -> str:
    """' cand' / ' not counted' for an accepted run the minimal policy does not count, '' for a counted one."""
    if episode_id in marks.counted:
        return ''
    return ' cand' if episode_id in marks.candidates else ' not counted'


def portrait_window_low(analysis: MMRAnalysis) -> Optional[float]:
    """Lower edge [rad] of the 360-degree window shared by (k) and (l), so the two portraits compare: around the center
    over the episodes of (l) (`_episodes_window_low`) when there are accepted episodes, else around the center of the
    whole record; None without a filtered angle."""
    prep, geometry = analysis.prep, analysis.geometry
    if prep is None:
        return None
    accepted = accepted_runs(geometry)
    if accepted:
        t, angle = np.asarray(prep.t_yr, dtype=float), np.asarray(prep.angle_f, dtype=float)
        marks = policy_marks(analysis)
        return _episodes_window_low(geometry, _portrait_episodes(accepted, marks.counted if marks is not None else None), t, angle)
    t = np.asarray(prep.t_yr, dtype=float)
    mask = (t >= prep.window[0]) & (t <= prep.window[1])
    return _window_low(geometry.centre.centre_rad if geometry is not None else np.asarray(prep.angle_f)[mask])


def _record(prep, record: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    """The record's edges [yr] the fill spans reach (`episode_spans`): `record`, else the first and last sample."""
    if record is not None:
        return float(record[0]), float(record[-1])
    t = np.asarray(prep.t_yr, dtype=float)
    return float(t[0]), float(t[-1])


def episode_rate_limits(
    analysis: MMRAnalysis, settings=None, record: Optional[Tuple[float, float]] = None
) -> Optional[Tuple[float, float]]:
    """The y view [rad/kyr] of every sample of every accepted episode over its portrait span (`portrait_span`: the
    fill of (a) inside the window of (k); sigma_dot at the response cutoff, padded as the portraits are): the least
    (k) may show, whatever cycles (l) draws. `settings`/`record`: as for `episode_spans` (without `settings`, P/4
    beyond the turns). None without a filtered angle or an accepted episode."""
    prep, geometry = analysis.prep, analysis.geometry
    if prep is None or geometry is None:
        return None
    t = np.asarray(prep.t_yr, dtype=float)
    spans = episode_spans(geometry.runs, _record(prep, record), settings)
    covered = np.zeros(len(t), dtype=bool)
    for run in geometry.runs:
        if run.accepted:
            start, end = portrait_span(run, spans, prep.window)
            covered |= (t >= start) & (t <= end)
    if not covered.any():
        return None
    return _view_limits(_rate(prep)[covered] * PER_KYR, pad=PORTRAIT_VIEW_PAD)


def share_portrait_limits(axes, episodes: Optional[Tuple[float, float]] = None) -> None:
    """One y range for the portraits that were drawn, (k) then (l) (their x window is already shared), so they read on
    the same scales. The episodes' view is (l)'s, widened to `episodes` (`episode_rate_limits`: every sample of every
    accepted episode), so (k) is never clipped tighter than its own episodes. Both get the union of (k)'s and the
    episodes' views; when the whole record spans more than PORTRAIT_SHARE_MAX times the episodes' view (a circulation
    far faster than the libration would squash the island to a line), the episodes' view, the whole record's samples
    beyond it marked ▲▼ at the edge of (k)."""
    drawn = [ax for ax in axes if ax.get_xlabel()]
    if len(drawn) < 2:
        return
    (k_low, k_high), (l_low, l_high) = drawn[0].get_ylim(), drawn[1].get_ylim()
    if episodes is not None:
        l_low, l_high = min(l_low, episodes[0]), max(l_high, episodes[1])
    if k_high - k_low <= PORTRAIT_SHARE_MAX * (l_high - l_low):
        limits = (min(k_low, l_low), max(k_high, l_high))
    else:
        limits = (l_low, l_high)
        for above, beyond in ((True, k_high > l_high), (False, k_low < l_low)):
            if beyond:
                edge_marker(drawn[0], 0.5, above, REFERENCE)
    for ax in drawn:
        ax.set_ylim(*limits)


def draw_episode_portrait(
    ax,
    analysis: MMRAnalysis,
    layout: VerdictFigureLayout = LAYOUT_V1,
    divisor: float = 1.0,
    unit: str = 'yr',
    low: Optional[float] = None,
    settings=None,
    record: Optional[Tuple[float, float]] = None,
    letter: str = 'l',
):
    """(l) the accepted episodes only: sigma_f against sigma_dot at the response cutoff, the PORTRAIT_MAX_EPISODES
    longest, each over its fill span of (a) inside the window of (k) (`portrait_span`, `portrait_pieces`): whole or,
    when it holds more, PORTRAIT_MAX_CYCLES of its cycles (`portrait_cycles`), the stretches beyond its first and last
    turning point dashed; one color per episode (the colors of (a)); a forced wave is labeled '#id wave', a run the
    status does not count '#id < floor', as on (a); the label's times are the turning points'. `low`: the lower edge of the 360-degree window [rad]
    (`portrait_window_low`, shared with (k)); default: around the episodes. `settings`/`record`: as for
    `episode_spans` (the figure passes its own). Under the minimal policy (`policy_marks`) the counted episodes come
    first and solid; an accepted run it does not count is dashed, '#id cand' or '#id not counted' (138537). `letter`:
    the panel's letter (the 'panels' figure re-letters the bottom row)."""
    marks = policy_marks(analysis)
    _bottom_title(ax, f'({letter}) accepted episodes' if marks is None else f'({letter}) episodes, counted solid', layout)
    prep, geometry = analysis.prep, analysis.geometry
    if _aliased(analysis):
        _empty(ax, _alias_reason(analysis), layout, crossed=True)
        return
    if prep is None or geometry is None:
        _empty(ax, 'not computed: no filtered angle', layout)
        return
    accepted = accepted_runs(geometry)
    if not accepted:
        _empty(ax, 'no accepted episode', layout)
        return
    colours = episode_colours(geometry.runs)
    chosen = _portrait_episodes(accepted, marks.counted if marks is not None else None)
    waves = wave_labels(geometry)  # '#id wave', 'forced (strength)', 'ripple', as on (a): no episode of the angle
    induced = induced_episode_ids(analysis)  # '#id induced', hatched on (a)
    uncounted = uncounted_tags(geometry)  # '#id < floor', grey on (a): the status does not count it (W28 C11)
    t = np.asarray(prep.t_yr, dtype=float)
    rate = _rate(prep) * PER_KYR
    angle = np.asarray(prep.angle_f, dtype=float)
    low = _episodes_window_low(geometry, chosen, t, angle) if low is None else low
    tp = geometry.turning_points
    spans = episode_spans(geometry.runs, _record(prep, record), settings)
    shown = []
    for run in chosen:
        colour = colours.get(run.episode_id, EPISODE_COLOURS[0])
        counted = marks is not None and run.episode_id in marks.counted
        label = f'#{run.episode_id}' + ('' if counted else waves.get(run.episode_id, ''))
        label += INDUCED_LABEL if run.episode_id in induced else ''
        if marks is None:
            label += uncounted.get(run.episode_id, '')
        elif counted or run.episode_id not in waves:  # a wave's label already says it is not the angle's
            label += _policy_tag(run.episode_id, marks)
        label += f' {run.t_start_yr / divisor:.4g}–{run.t_end_yr / divisor:.4g} {unit}'
        label += ' (before jump)' if run.before_last_jump else ''
        pieces = portrait_pieces(run, portrait_span(run, spans, prep.window), tp.times_yr, tp.kinds)
        labelled = next((n for n, piece in enumerate(pieces) if not piece[2]), 0)  # the legend shows the solid line
        for n, (t0, t1, extended) in enumerate(pieces):
            mask = (t >= t0) & (t <= t1)
            x = np.degrees(_wrap_window(angle[mask], low))
            y = rate[mask].copy()
            y[1:][np.abs(np.diff(x)) > 180] = np.nan
            faint = marks is not None and run.episode_id not in marks.counted
            look = PORTRAIT_EXTENSION_STYLE if extended or faint else PORTRAIT_CORE_STYLE
            ax.plot(x, y, color=colour, label=label if n == labelled else None, rasterized=True, **look)
            if n == 0 and len(x):
                ax.plot(x[:1], y[:1], 'o', ms=3, color=colour)
            shown.append(y)
    _angle_axis(ax, low, layout, 'σ_f, °')
    centre_marks(ax, chosen, low)
    _view(ax, np.concatenate(shown) if shown else [0.0], pad=PORTRAIT_VIEW_PAD, extra_top=0.25)
    ax.legend(fontsize=layout.font_note_pt, loc='upper right', frameon=False, handlelength=1.2)
    if len(accepted) > len(chosen):
        ax.text(
            0.01,
            0.01,
            f'{len(chosen)} longest of {len(accepted)} episodes',
            transform=ax.transAxes,
            fontsize=layout.font_note_pt,
            color=NOTE,
            va='bottom',
        )


def draw_portrait(
    ax, analysis: MMRAnalysis, norm, layout: VerdictFigureLayout = LAYOUT_V1, divisor: float = 1.0, low: Optional[float] = None
) -> None:
    """(k) sigma_f mod 360 around the center against sigma_dot over the whole record, colored by time, every sample
    (rasterized); crossed out for aliased samples.

    `divisor` converts years into the unit of `norm` (the figure's time unit); `low`: the lower edge of the 360-degree
    window [rad] (`portrait_window_low`, shared with (l)); default: around the center."""
    from matplotlib.collections import LineCollection

    _bottom_title(ax, '(k) phase portrait, whole record', layout)
    prep = analysis.prep
    if _aliased(analysis):
        _empty(ax, _alias_reason(analysis), layout, crossed=True)
        return
    if prep is None:
        _empty(ax, 'not computed: no filtered angle', layout)
        return
    t = np.asarray(prep.t_yr, dtype=float)
    mask = (t >= prep.window[0]) & (t <= prep.window[1])
    if low is None:
        low = _window_low(analysis.geometry.centre.centre_rad if analysis.geometry is not None else prep.angle_f[mask])
    x = np.degrees(_wrap_window(prep.angle_f[mask], low))
    y = _rate(prep)[mask] * PER_KYR
    times = t[mask] / divisor
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    keep = np.abs(np.diff(x)) <= 180
    collection = LineCollection(segments[keep], cmap=time_cmap(), norm=norm, linewidths=0.7, rasterized=True)
    collection.set_array(times[:-1][keep])
    ax.add_collection(collection)
    _angle_axis(ax, low, layout, 'σ_f, °')
    if analysis.geometry is not None:
        centre_marks(ax, accepted_runs(analysis.geometry), low)
    _view(ax, y, pad=PORTRAIT_VIEW_PAD)


def fair_unavailable_reason(analysis: MMRAnalysis, series: MMRSeries, max_step_fraction: float = FAIR_STEP_FRACTION) -> Optional[str]:
    """Why FAIR is not drawn, or None: prograde two-body angles with the planet's longitudes and dt <= P/4.2 only.

    FAIR is not generalised to three-body angles: in (M, k1(lambda1 - lambda) + k2(lambda2 - lambda)) a libration
    gives |q| strips, but that plane needs k1, k2, so it is sigma shifted by qM and cannot identify p:q."""
    spec = analysis.resonance
    if spec.kind == ResonanceKind.RETROGRADE:
        return 'retrograde resonance'
    if spec.kind != ResonanceKind.TWO_BODY:
        return 'three-body angle'
    if series.mean_anomaly is None or series.mean_longitude is None:
        return 'no mean anomaly or mean longitude'
    planet = series.planets.get(spec.planets[0]) if spec.planets else None
    if planet is None or planet.mean_longitude is None:
        return 'no planet longitudes'
    fraction = analysis.context.fair_step_fraction if analysis.context is not None else float('nan')
    if not is_finite(fraction) or fraction > max_step_fraction:
        return f'step {fraction:.2g} of the orbital period > 1/{1 / max_step_fraction:.2g}'
    return None


# The warning line of a figure without FAIR (W28 g03-7, the author 2026-10-31: no empty slot, a warning with the reason
# where FAIR would be): each `fair_unavailable_reason` in words; the header's 'FAIR no (<reason>)' stays as it is.
FAIR_WARNING_PREFIX = 'FAIR not drawn: '
FAIR_REASON_WORDS = {
    'retrograde resonance': 'it is defined for prograde orbits only (for i > 90° the saved longitude and ϖ are in other ' 'conventions)',
    'three-body angle': 'it is a two-body diagnostic; for a three-body angle its plane is σ shifted by qM and cannot ' 'identify p:q',
    'no mean anomaly or mean longitude': 'the body\'s mean anomaly or mean longitude is not in the record',
    'no planet longitudes': 'planet longitudes not saved (SAVE_PLANETS)',
}
FAIR_COORBITAL_WORDS = '; for a 1:1 FAIR = (M, σ) repeats (b)'  # q = 0: one strip, the range of sigma itself


def fair_warning(analysis: MMRAnalysis, series: MMRSeries, max_step_fraction: float = FAIR_STEP_FRACTION) -> Optional[str]:
    """The warning line printed at the top of the legend when FAIR is not drawn (`fair_unavailable_reason`), or None
    when it is: 'FAIR not drawn: ' + the reason in words (`FAIR_REASON_WORDS`); a coarse step with its numbers; a 1:1
    without the planet's longitudes adds that FAIR would only repeat (b) there."""
    reason = fair_unavailable_reason(analysis, series, max_step_fraction)
    if reason is None:
        return None
    if reason.startswith('step '):
        fraction = analysis.context.fair_step_fraction if analysis.context is not None else float('nan')
        words = f'output step {fraction:.2g} of the orbital period, coarser than 1/{1 / max_step_fraction:.2g}: its strips are not resolved'
    else:
        words = FAIR_REASON_WORDS.get(reason, reason)
    if reason == 'no planet longitudes' and coorbital_planet(analysis.resonance) is not None:
        words += FAIR_COORBITAL_WORDS
    return FAIR_WARNING_PREFIX + words


def draw_fair(
    ax, analysis: MMRAnalysis, series: MMRSeries, norm, layout: VerdictFigureLayout = LAYOUT_V1, divisor: float = 1.0, letter: str = 'm'
) -> None:
    """(m) FAIR plane of a prograde two-body angle with dt <= P/4.2 (`fair_unavailable_reason` is None)."""
    _bottom_title(ax, f'({letter}) FAIR', layout)
    planet = series.planets[analysis.resonance.planets[0]]
    inner = osc.is_inner(series.a, planet.a)
    x, y = osc.fair_coordinates(series.mean_anomaly, series.mean_longitude, planet.mean_longitude, inner)
    ax.scatter(x, y, c=series.t_yr / divisor, cmap=time_cmap(), norm=norm, s=1.2, linewidths=0, rasterized=True)
    ticks = [0, 90, 180, 270, 360]
    ax.set(xlim=(0, 360), ylim=(0, 360), xticks=ticks, yticks=ticks)
    name = analysis.resonance.planets[0]
    ax.set_xlabel('M, °', fontsize=layout.font_axis_pt, labelpad=1)
    ax.set_ylabel(f'λ_{name} − λ, °' if inner else f'λ − λ_{name}, °', fontsize=layout.font_axis_pt, labelpad=1)


def legend_rect(layout: VerdictFigureLayout, third_slot_free: bool) -> Rect:
    """The legend's box: `layout.legend`, widened over the third slot of the bottom row when nothing is drawn there
    (no FAIR)."""
    x, y, w, h = layout.legend
    if not third_slot_free:
        return layout.legend
    left = layout.bottom[2][0]
    return (left, y, x + w - left, h)


def draw_legend(
    fig, layout: VerdictFigureLayout, lines: Sequence[str] = LEGEND, rect: Optional[Rect] = None, warning: Optional[str] = None
) -> None:
    """What every line and mark of the panels means, wrapped to the legend box (`rect`, default `layout.legend`);
    the font shrinks (to LEGEND_MIN_FONT_PT) until the text fits the box. `warning` (`fair_warning`): one line in the
    warning color at the top of the box, above the heading, wrapped and shrunk with the legend."""
    width, height = fig.get_size_inches()
    x, y, w, h = rect or layout.legend
    top = y + h
    renderer = fig.canvas.get_renderer()
    font = layout.font_legend_pt
    while True:
        chars = max(int(w * 72 / (LEGEND_CHAR_WIDTH * font)), 20)
        artists, used = [], 0.0  # inches from the top of the box
        if warning:
            note = fig.text(
                x / width,
                top / height,
                textwrap.fill(warning, chars, subsequent_indent='   '),
                fontsize=font,
                va='top',
                color=FAIR_WARNING_COLOUR,
                fontweight='bold',
                linespacing=1.2,
            )
            artists.append(note)
            used = note.get_window_extent(renderer).height / fig.dpi + LEGEND_WARNING_GAP
        heading = fig.text(x / width, (top - used) / height, 'Legend', fontsize=layout.font_legend_pt + 1.5, fontweight='bold', va='top')
        text = '\n'.join(textwrap.fill(line, chars, subsequent_indent='   ') for line in lines)
        artist = fig.text(x / width, (top - used - 0.22) / height, text, fontsize=font, va='top', color='#333333', linespacing=1.2)
        artists += [heading, artist]
        boxes = [a.get_window_extent(renderer) for a in artists]
        fits = max(b.width for b in boxes) <= w * fig.dpi + 0.5 and boxes[-1].height <= (h - used - 0.22) * fig.dpi + 0.5
        if fits or font <= LEGEND_MIN_FONT_PT:
            return
        for a in artists:
            a.remove()
        font -= 0.25


def _colour_bar(fig, norm, layout: VerdictFigureLayout, unit: str, label_on_top: bool = False):
    """The time color bar right of the first portrait; `label_on_top`: its label 't, kyr' above the bar instead of
    beside it, where the next portrait's y label and wide tick labels reach (the ZLK free eccentricity vector, 591986)."""
    from matplotlib.cm import ScalarMappable

    width, height = fig.get_size_inches()
    x, y, w, h = layout.colour_bar
    cax = fig.add_axes((x / width, y / height, w / width, h / height))
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=time_cmap()), cax=cax)
    if label_on_top:
        cax.set_title(f't, {unit}', fontsize=layout.font_ticks_pt, pad=3, loc='left')
    else:
        bar.set_label(f't, {unit}', fontsize=layout.font_ticks_pt, labelpad=2)
    bar.ax.tick_params(labelsize=layout.font_ticks_pt - 0.5)
    bar.outline.set_linewidth(0.5)


# --------------------------------------------------------------------------------------------- the figure


def _rect_axes(fig, rect: Rect):
    width, height = fig.get_size_inches()
    x, y, w, h = rect
    return fig.add_axes((x / width, y / height, w / width, h / height))


def figure_rows(analysis: MMRAnalysis) -> Tuple[Mapping[str, Any], Mapping[str, Any], List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """The CSV rows the text is printed from: summary, numbers, episodes and constants of the analysis."""
    constants = analysis.context.constants_rows() if analysis.context is not None else []
    return analysis.summary_row(), analysis.numbers_row(), analysis.episode_rows(), constants


def episode_portrait_drawn(analysis: MMRAnalysis) -> bool:
    """Whether (l) draws an episode: not when it would hold only its reason (aliased samples, no filtered angle, no
    accepted episode; `draw_episode_portrait`). The 'panels' figure leaves such a slot out."""
    return not _aliased(analysis) and analysis.prep is not None and bool(accepted_runs(analysis.geometry))


# The 'panels' figure (plot option combined.figure='panels'), inches: room over (a) for its episode labels and sub-regime
# brackets; right of `separator_x`, where the margin legends end in the full figure, for their last glyphs
PANELS_TOP_PAD_IN = 0.25
PANELS_RIGHT_PAD_IN = 0.15


def panels_layout(layout: VerdictFigureLayout) -> VerdictFigureLayout:
    """The layout of the 'panels' figure (for a paper): the same rectangles, the page cropped to what it draws, the time
    panels with their margin legends (up to `separator_x`, + PANELS_RIGHT_PAD_IN) and the bottom row; the header's band and the right column
    are cut off. Every rect is measured from the lower-left corner, so nothing else moves."""
    x, y, w, h = layout.panels
    return replace(layout, size=(layout.separator_x + PANELS_RIGHT_PAD_IN, y + h + PANELS_TOP_PAD_IN))


def panels_stamp(fig, layout: VerdictFigureLayout, stamp: Optional[str]) -> None:
    """The 'panels' figure has no header: a `stamp` (a re-run analysis that disagrees with the saved summary) goes in
    red to its top right corner, so a mismatch is never hidden. Shared by the MMR and the secular/ZLK figures."""
    if not stamp:
        return
    width, height = layout.size
    fig.text(
        1 - 0.05 / width, 1 - 0.05 / height, stamp, fontsize=layout.font_letter_pt, va='top', ha='right', color='#b2182b', fontweight='bold'
    )


def draw_combined(
    analysis: MMRAnalysis,
    series: MMRSeries,
    layout: VerdictFigureLayout = LAYOUT_V1,
    stamp: Optional[str] = None,
    rows: Optional[Tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = None,
    options: Optional[Mapping[str, Any]] = None,
    settings=None,
):
    """The whole V1 figure; `stamp` (e.g. 'recomputed != summary') is printed in red in the header. Returns the Figure.
    `settings`: the AnalysisSettings the explanation compares with (default: from the configuration).

    `rows` = (summary, numbers, episodes, constants) as read from the CSV files (a restore without
    recomputation); by default they come from the analysis itself (`figure_rows`). `options`: plot options
    (`style.DEFAULT_PLOT_OPTIONS`, e.g. {'combined': {'osculating_a': 'hidden'}}).

    With {'combined': {'figure': 'panels'}} (a paper figure): the time panels with their margin legends and the bottom
    row only, no header, legend or right column (`fig.text_stats` is empty); (l) is left out when it would draw no
    episode (`episode_portrait_drawn`), FAIR when it is not drawn (no warning line either); the remaining portraits
    move left and are lettered on from (k); the page is cropped (`panels_layout`). A `stamp` is still printed (top
    right), so a mismatch with the saved summary is never hidden.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    summary, numbers, episodes, constants = rows if rows is not None else figure_rows(analysis)
    settings = settings if settings is not None else _settings()  # read once for the panels and the right column
    resolved = resolve_plot_options(options)
    panels = resolved['combined']['figure'] == FigureMode.PANELS
    prepared = None
    if panels:
        layout = panels_layout(layout)
    elif resolved['combined']['numbers'] != NumbersMode.ALL:
        # the column is measured before the figure exists: the figure grows down when it needs (decision D8)
        layout, prepared = prepare_column(analysis, (summary, numbers, episodes, constants), layout, settings)
    fig = plt.figure(figsize=layout.size, dpi=layout.dpi)
    if not panels:
        draw_header(fig, analysis, layout, series, stamp, header_role(analysis, (summary, numbers, episodes), settings))
    else:
        panels_stamp(fig, layout, stamp)
    draw_time_panels(fig, analysis, series, layout, options, settings, {**numbers, **summary})
    divisor, unit = time_unit(series.t_yr[-1] - series.t_yr[0])
    norm = Normalize(series.t_yr[0] / divisor, series.t_yr[-1] / divisor)
    slots, letters = list(layout.bottom), iter('lmn')  # the 'panels' figure packs the drawn portraits left
    low = portrait_window_low(analysis)
    portraits = [_rect_axes(fig, slots.pop(0))]
    draw_portrait(portraits[0], analysis, norm, layout, divisor, low)
    _colour_bar(fig, norm, layout, unit)
    record = (float(series.t_yr[0]), float(series.t_yr[-1]))  # the edges the fill of (a) reaches (`_shade_episodes`)
    if not panels or episode_portrait_drawn(analysis):
        portraits.append(_rect_axes(fig, slots.pop(0)))
        draw_episode_portrait(portraits[1], analysis, layout, divisor, unit, low, settings, record, next(letters))
    share_portrait_limits(portraits, episode_rate_limits(analysis, settings, record))
    warning = fair_warning(analysis, series)
    if warning is None:
        draw_fair(_rect_axes(fig, slots.pop(0)), analysis, series, norm, layout, divisor, next(letters))
    if panels:
        fig.text_stats = {}
        return fig
    draw_legend(fig, layout, legend_lines(analysis), rect=legend_rect(layout, third_slot_free=warning is not None), warning=warning)
    width, height = layout.size
    fig.add_artist(plt.Line2D([layout.separator_x / width] * 2, [0.3 / height, (height - 0.2) / height], color='#dddddd', lw=0.8))
    fig.text_stats = draw_right_column(fig, analysis, (summary, numbers, episodes, constants), layout, options, prepared)
    return fig


def prepare_column(analysis: MMRAnalysis, rows, layout: VerdictFigureLayout = LAYOUT_V1, settings=None):
    """The explained right column before the figure is made: (the layout to build the figure on, the
    `verdict_column_draw.Prepared` column). The layout is `layout` grown down by what the column needs, capped at
    `style.COLUMN_MAX_GROWTH_IN` (`verdict_column_draw.prepare`, decision D8); the left panels keep their geometry at
    the top."""
    from resonances.plotting.verdict_column_draw import prepare
    from resonances.plotting.verdict_column_mmr import figure_column

    return prepare(figure_column(analysis, rows, settings), layout)


def draw_right_column(fig, analysis: MMRAnalysis, rows, layout: VerdictFigureLayout, options=None, prepared=None) -> Dict[str, Any]:
    """The right column by the plot option `combined.numbers`: 'explained' (the ten blocks of `verdict_column`,
    `verdict_column_draw.draw_prepared`; `prepared` = `prepare_column`'s column, measured for the layout the figure was
    built on) or 'all' (every number, `draw_text`). Returns the statistics of the column (for the caller and the
    tests)."""
    from resonances.plotting.verdict_column_draw import draw_prepared

    if resolve_plot_options(options)['combined']['numbers'] == NumbersMode.ALL:
        summary, numbers, episodes, constants = rows
        return draw_text(fig, text_blocks(summary, numbers, analysis.verdict.decision_path, episodes, constants), layout)
    return draw_prepared(fig, prepared, layout)
