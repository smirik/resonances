<!-- Generated from docs/reading-figures.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# How to read the figures

The package draws figures to help a person check a verdict by eye. This page explains every panel, line and mark,
in plain words first; the formulas are in the folded "The maths" blocks, with links to the pages that explain each
quantity in depth. Nothing drawn on a figure changes a status: the verdict is decided before anything is drawn. The
numbers in the right column are columns of the [output files](output.md); a few annotations in the data panels (the
$R^2$ of each force window in (e)) are computed for the drawing and have no column of their own.

## Which figures there are

| figure | when it is drawn | section |
|---|---|---|
| the combined figure: one page per body and resonance, the verdict and its evidence | by default (`plots=['combined']`) | [MMR](#the-combined-mmr-figure), [secular and Lidov-Kozai](#the-secular-and-lidov-kozai-figure) |
| standalone figures: `evolution`, `recurrence`, `fair`, `portrait`, `cycles`, `angle`, `rate`, `ecc_vector`, `cross_spectrum`, `free_omega`, `clones` | when listed in `plots` | [Standalone figures](#standalone-figures) |

Which rows get figures is set by the status filter `plot` (by default every row except a plain non-resonant one).
The file names, folders and drawing options are on the [Plots](plots.md) page. One combined figure for every status
is in the [Gallery](physics/gallery.md).

## Shared conventions

These hold for the combined figures; most hold for the standalone ones too.

- **Black and grey.** Grey is the osculating series, exactly as integrated; black is the same series with the
  short periods filtered out. When the samples are too dense to draw one by one, grey becomes a band between the
  smallest and the largest value.
- **One colour per quantity.** Red is the centre of the angle and the line $\dot\sigma = 0$; blue is a fitted or
  model quantity; dark grey marks references (turning points, jumps, $h = 1$); cyan is the catalogue resonant
  semi-major axis. The full palette: [Plots](plots.md#colours-and-units).
- **Time** runs left to right in yr, kyr or Myr, whichever suits the span (below $10^4$ yr: yr; below $10^6$ yr:
  kyr). Time colour bars use one colour map (viridis without its palest yellow): dark is early, yellow-green late.
- **Angles** folded into one turn are labelled from 0° to 360° (270°, not −90°). The unwrapped angle of MMR panels
  (a) and (i) is in radians; (j) shows offsets in degrees (its kinematic version: detrended terms, in degrees).
- **▲ and ▼ at the edge of a panel** mean "beyond the axis". The view holds 99 % of the osculating samples; the rare
  samples outside it are grouped in time, and each group is one small triangle at the time it goes furthest out (at
  most 8 per edge). A reference level outside the view (for example the resonant semi-major axis) is a triangle in
  the level's colour.
- **The ends of the record.** A filter spreads near the ends of a record, so the analysis does not read the first
  and last `FILTER_EDGE_TRIM_CUTOFFS` cutoff periods. In panels (c), (d), (e) of the MMR figure and (c), (f) of the
  secular figure, the filtered line is pale and dashed there; a filtered line that would leave the view there is a
  black triangle. The other panels draw their filtered lines solid to the ends: do not read them near the ends either.
- **No words in the panels.** The panels hold only data and a few data numbers. What a line means is in the
  legend at the bottom of the figure; the values of the reference lines of (c), (d) and (h) are printed right of
  those panels.

??? info "The maths: the filter and the cutoff"
    $\sigma_f$ is $\sigma$ passed forward and backward through a Butterworth low-pass filter of order
    `FILTER_ORDER`: a period $P$ keeps a share $G(P)$ of its amplitude, $G(P_c) = \tfrac12$ at the cutoff $P_c$ and
    $G \to 1$ for $P \gg P_c$. For a mean-motion resonance $P_c$ is the larger of a multiple of the longest synodic
    period of the angle's planets and a share of the shortest libration the analysis allows. Details:
    [the cutoff](physics/signal.md#the-cutoff-what-is-libration-and-what-is-noise) and
    [the filter](physics/signal.md#the-filter).

## The combined MMR figure

![The combined figure of (1921) Pala in the 2:1 resonance with Jupiter](https://smirik.github.io/resonances/img/gallery/mmr2__1921__2J-1+0-1.png)

*(1921) Pala in the 2:1 resonance with Jupiter, a textbook libration ([Gallery](physics/gallery.md#resonant-two-body)).
Click the image to open it at full size.*

The page is 24 × 13.5 inches. The left half holds ten time panels (a)–(j) on one time axis and, below them, two
phase portraits (k), (l), the FAIR plane (m) and the legend. The right half explains the verdict.

### The header

The first line names the body, the resonance key, its type (`mmr2`, `mmr3` or `mmr2 retrograde`) and the kind of
orbit in the librating episodes (for example `tadpole L4`, `horseshoe`, `interior asymmetric libration about 72°`).
For a non-resonant or uncertain verdict whose episodes come from somewhere else it reads
`episodes about X° (not this resonance)`. A row found as a subresonance adds `subresonance of <parent>`.

When another resonance or term makes the cycles, the second line names it by its role
([roles](physics/response.md#roles)):

- `owner: 5E-16+0+11 (librates in the episode)`: the owner librates and drives $a$;
- `driver: 3N-5+0+2 (circulates, P 4.51 kyr): forced wave`: the other angle circulates, its forced wave makes the
  cycles; `candidate driver: X` when the owner's measured regimes do not cover σ's deciding episode;
- `6E-11+0+5 librates after the episode (13.7 kyr–17.9 kyr), outside it`: a neighbour, a resonance the body is in
  before or after these episodes, not the cause (`… during … but does not make σ's cycles` inside them);
- `resonant argument: 9M-16+2+5 (one resonance 9M-16)`: the member of the multiplet that librates;
- `kinematic: −4ϖ term (2ω cycle, 18 kyr); a does not follow`: σ's perihelion term makes the swing, at the secular
  period σ's period matched;
- `also in: 5J+6S-2+0+0-9 (both resonances at once)`: origin joint, the other resonance librates at the same time
  and keeps its own row ([both resonances at once](physics/response.md)).

The next line describes the record: its span and direction, the integrator and its step, the output step and
number of rows, the filter cutoff and what set it, the cutoff used for rates, the epoch, and whether the FAIR
plane (m) could be drawn.

A red stamp in the top right corner, `recomputed != summary`, appears only when the figure had to run the analysis
again (after a restore) and got a different answer from `summary.csv`.

### (a) The angle over the whole record

The resonant angle $\sigma$, unwrapped: a circulation runs off upwards or downwards, a libration stays in a band.

- grey: the osculating angle; black: the filtered angle $\sigma_f$; red: its slowly moving centre;
- ▼ and ▲: the turning points of $\sigma_f$ (a maximum, a minimum), marked only when there are at most 80;
- coloured fills labelled `#1`, `#2`, …: the accepted librating episodes. A fill runs to a record edge the episode
  covers, else a quarter period beyond its first and last turning points. A hatched fill is an episode before the
  last jump of the semi-major axis (`////`); a fill hatched the other way (`\\\\`) is an episode another
  resonance induces (the header names it). Three labels mark accepted runs that are no libration of this angle:
  `#3 wave`, a forced wave of another argument; `#3 forced (strength)`, a run of a row the strength decided (this
  angle's force cannot make that period; the label names that ground, not a driver); `#3 ripple`, dotted, an
  unconfirmed ripple below 45° on an unfinished slow swing, an auxiliary interval that decides nothing;
- a grey strip along the bottom: runs of turning points that were rejected, with a letter for the reason:
  **T** over the top of the well, **F** too few turning points, **C** the centre moves, **O** no closed cycle;
- a grey strip with a grey label `#0 < floor` (or `#0 centre turns`) above the panel: a run the shape accepted but
  the status does not count, because it is faster than the angle's shortest libration (the period floor) or its
  centre turns a whole revolution inside it (episodes.csv `eligible` False). It is drawn like a rejected run, never
  as an episode, so the figure and the verdict agree (2022 AP1: `#0 < floor`, verdict circulation);
- a turning point inside the filter's pale edge of a 1:1 angle: a turn read on the orbit-averaged raw angle; it is
  added only when it closes a cycle (2006 BZ147, 960 yr);
- dotted lines (or a hatched band with "N jumps of a"): jumps of the semi-major axis; a shaded half-cycle with
  "action step (G2a)": a sudden change of the libration's amplitude, treated like a jump;
- 1:1 only, one episode in parts ([one episode of a 1:1 angle](physics/geometry.md)): brackets just above the panel,
  one per sub-regime, named by its well (`#0 horseshoe`, `tadpole L4`; `new amplitude`: the same well after an
  amplitude change, the gap between two brackets is the transitional half-cycle); names that would overlap merge
  (`#1 compound → quasi-satellite`). A lighter fill hatched `--` labelled `inferred`: the stay in the co-orbital
  region beyond the measured turning points (episodes.csv `residence_*`), up to the record's edge or a whole turn of
  the orbit-averaged angle. It is inferred, not measured: coverage, the edges and the status read only the turning
  points, and (l) does not draw it (2020 CB 1E-1: measured 443–601 yr, hatched from 601 yr to the end: 601 yr is no
  exit);
- ×: where the analysis places the exit from the resonance. It may be only suspected: then the right column shows
  `check_mmr_exit` failed, and the cap `exit_suspected` when the verdict rests on that edge.

What to look for: fills that cover the whole record are a libration; fills in part of it are episodes.

??? info "The maths: angle, turning points, episodes"
    $$
    \sigma = k\lambda + \sum_j k_j\lambda_j + p\varpi + \sum_j p_j\varpi_j + v\Omega + \sum_j v_j\Omega_j
    $$

    (body without index, planets with index $j$). Turning points are the extrema of $\sigma_f$ that stand out by at
    least a prominence set from the noise of $\sigma_f$. Three alternating turning points close a cycle when the angle
    comes back by less than its swing ($Q_i < 1$, [cycles and their closure](physics/geometry.md#cycles-and-their-closure)).
    An episode normally needs at least three turning points and one closed cycle; two opposite turning points also
    make one cycle when the angle and $a$ come back to the same state on either side of them (closure by phase
    return, confidence at most `medium`). The centre is $\sigma_f$ smoothed again over a few libration periods.
    Details: [turning points](physics/geometry.md#turning-points),
    [runs and episodes](physics/geometry.md#runs-and-episodes),
    [jumps and action steps](physics/geometry.md#a-step-of-the-libration-action).

### (b) The angle modulo 360°

The same angle folded into one turn, in a 360° window around the centre: grey dots are the osculating samples,
black is $\sigma_f$, red the centre (the lines are left out when the cycles are too dense to draw). An accepted owner
resonance is drawn purple dashed, an accepted subresonance orange dash-dot.

A red boxed note `ALIAS step … ≈ P_σ …` means that the output step is close to a multiple of the angle's own fast
period: the samples, like a stroboscope, make a fast circulation look slow
([Gallery: alias](physics/gallery.md#alias)).

**The strip (b′).** When the cycles are too dense to see in (b), a strip below it shows about five cycles from the
middle of the deciding episode on their own time axis (in yr): ● the samples, a grey dashed line the path the
analysis' unwrap assumes between them (a whole turn may pass between two samples), black $\sigma_f$. A ▼ on the top
edge of (b) marks the middle of the strip's time window; a second strip (b′2) shows the last regime after a jump, and
the ▼s are then numbered. A ▼ never moves sideways: next to the panel letter or the ALIAS plaque it is raised just
above it, and its number moves level with the ▼ or to its left.

??? info "The maths: sampling"
    With the output step $\Delta t$, a rate $\nu$ is seen as $\nu - 2\pi m/\Delta t$ for a whole number $m$: the
    sampled angle cannot tell them apart. The analysis follows the angle sample to sample with a predicted rate, so a
    turn per step is still counted. Details: [what the step can resolve](physics/signal.md#what-the-output-step-can-resolve),
    [the predictive unwrap](physics/signal.md#counting-the-turns-the-predictive-unwrap).

### (c) The rate of the angle

How fast $\sigma$ moves, in units of the speed a pendulum has at the bottom of its well on the separatrix. A
libration swings between positive and negative values and changes sign at every turning point; a clean circulation
keeps one sign. Brief reversals do not rule out a circulation: what decides it is the centre in (i), which completes
a turn or keeps advancing over several. The axis is linear between −1 and +1 and logarithmic outside, so a slow
libration and a fast circulation are both in view.

- red line: $\dot\sigma = 0$;
- blue dashed: $\pm 2\omega_0$ of each accepted episode, over its span, with the episode's own $\omega_0$ (they need
  not sit at ±1, since the axis uses one reference); without an episode, a thin pair from the theoretical $\omega_0$;
- dotted: the rate the mean motion predicts in each regime between jumps;
- ○ on the zero line: a slip, where the angle's midpoint steps by a full turn.

When no $\omega_0$ is known at all, the panel shows $\dot\sigma$ in rad/kyr on an ordinary axis.

??? info "The maths: the separatrix speed"
    The black line is $\dot\sigma / 2\omega_{0,\rm ref}$, with $\dot\sigma$ taken at the rate cutoff
    $\max(P_c, 0.1\,P_{\rm cyc})$. $\omega_{0,\rm ref}$ is the deciding episode's $\omega_0$, else the longest
    other accepted episode's, else the theoretical $\omega_{0,\rm th} = \lvert k\rvert\, n\sqrt{3\mu_{\rm eff,th}}$,
    with $\mu_{\rm eff,th}$ the resonance's effective strength (the three-body strength scale for a three-body angle;
    [how strong must the resonance be](physics/pendulum.md#how-strong-must-the-resonance-be)). An episode's
    $\omega_0$ comes from its cycles: a cycle of period $P_i$ and amplitude $A_i$ gives the small-amplitude period
    $T_{0,i} = P_i / F(A_i)$ with

    $$
    F(A) = \frac{2}{\pi}\,\mathsf K\!\left(\sin^2\frac A2\right), \qquad \omega_0 = \frac{2\pi}{\operatorname{median} T_0},
    $$

    $\mathsf K$ the complete elliptic integral of the first kind. For the cosine pendulum $\dot\psi^2 =
    4\omega_0^2(h - \sin^2\frac\psi2)$, so the separatrix ($h = 1$) passes the bottom at $\dot\psi = \pm 2\omega_0$:
    ±1 on this axis. This is a cosine-equivalent scale: a well of another shape has its own separatrix speed
    $\sqrt{2(E_{\rm sep} - V_{\min})}$, so ±1 bounds a libration exactly only for a cosine well. The dotted rate is
    $\nu = \nu(\langle a\rangle) + k\,(\dot\lambda - n(\langle a\rangle))$ with $\dot\lambda$ measured in the
    regime. Details: [the pendulum](physics/pendulum.md#the-pendulum).

### (d) The semi-major axis

Three levels of $a$ in au: light grey the osculating $a$, mid grey $a$ filtered at the cutoff, black a slower
version that shows the resonant swing when the filtered one is a dense band (when there is no slower version, the
filtered $a$ itself is black).

- dotted: $a_*$, where the angle would stand still; dashed cyan: $a_{\rm res}$, the value of the resonance catalogue;
- light blue band: $a_* \pm \Delta a$, the half-width of the resonance measured from the episodes;
- red dots: the mean of $a$ over each cycle of $\sigma_f$ (maximum to maximum), drawn when there are 2 to 200 maxima;
- ▼ `kick` at the top edge (1:1): a jump of $a$ at a change inside an episode that is no boundary, a change of well or
  of amplitude (episodes.csv `marks`, `kick@t`; the jump itself is the dotted line of (a)).

$a_*$ and $a_{\rm res}$ are mean-motion values, while the curves are osculating. To compare like with like, both
lines are moved by the difference between the mean osculating $a$ and the $a$ that the measured mean motion gives,
separately in each regime between jumps. The margin to the right prints the unshifted values and the shift.

What to look for: in a libration $a$ swings around $a_*$ in step with $\sigma$, inside the band.

The option `combined.osculating_a` (`full`, `clipped`, `hidden`) chooses how much of the osculating $a$ stays in view
([Plots](plots.md#options)).

??? info "The maths: exact resonance and half-width"
    $a_*$ solves $\nu(a_*) = 0$, where $\nu(a) = k\,n(a) + \sum_j k_j n_j + p\dot\varpi + \dots$ and $n(a) =
    2\pi a^{-3/2}$ (yr, au). The shift uses $a(\dot\lambda) = (GM_\odot/\dot\lambda^2)^{1/3}$. The half-width is

    $$
    \Delta a = \frac{2\omega_0}{\lvert c\rvert}, \qquad c = \frac{\partial\nu}{\partial a} = -\frac{3kn}{2a},
    $$

    the median over the accepted episodes, and a libration of amplitude $A$ swings by $\delta a = \Delta a\sin\frac
    A2$. Like (c), $\Delta a$ is the cosine-equivalent width. Details:
    [the resonant angle and its rate](physics/pendulum.md#the-resonant-angle-and-its-rate),
    [the pendulum](physics/pendulum.md#the-pendulum).

### (e) The response of the semi-major axis

Black: the rate $\dot a$ (scaled by the power of ten in the label). Blue: the force law fitted in short sliding
windows, $\dot a$ as a sine of the angle; the blue numbers along the bottom are each window's $R^2$, written only
when there are at most 12 windows (close numbers merge into a range such as `0.74–0.81`). For each accepted episode
a dotted line at its middle carries `#id` and the $R^2$ of its two halves.

What to look for: when the resonance itself drives $a$, $\dot a$ follows the sine of $\sigma$ and the $R^2$ are
high. A low $R^2$ with a clear libration in (a) suggests that something else bounds the angle.

The blue curve is the law that decided (column `rsp_force_law`), on the series that law read:

- `single`: each window's sine, evaluated on $\sigma$ smoothed at the response cutoff; on an episode accepted in the
  frame of its moving well, on $\psi = \sigma - c(t)$, as the fit read it;
- `multiplet`: one curve, the multiplet law (its one fit over its windows, drawn over them), with
  `multiplet law j = …: R² … (single …)` at the top: the multiplet's $R^2$ against the single sine's;
- `harmonic` (several harmonics) or a 1:1 law: no blue curve; the panel prints that law's numbers
  (`harmonic law N=k: R² … (single …)`);
- `direct_force` (the [direct averaged-force law](physics/response.md#the-direct-averaged-force-law)): a thick black
  $\dot a$ and, per window, a blue curve $s\times$ the term's averaged force with the planet's real mass, both
  low-passed at the law's cutoff ($f\,P_\text{large}$), with `direct law 8N-13+0+5 (Neptune's mass) at 3.07 kyr: R² 0.75,
  s 1.14–1.29` at the top. The single-sine windows are not drawn: they are the law that failed.

When the line rule raised the cutoff (a line of $\dot a$ rivalled the force), the black $\dot a$ and the angle are
taken at that raised cutoff, and the panel prints `cutoff … yr (line … yr)` (columns `rsp_response_cutoff_ep_yr`,
`rsp_line_binding_yr`). The law drawn is the one the status read: when a witness episode decided (rule R, column
`rsp_force_source = witness`), its windows and law, on $\sigma$ and $\dot a$ smoothed at its own response cutoff
(`rsp_response_cutoff_ep_yr`, which can differ from the record's); otherwise the fit pooled over the episodes.

??? info "The maths: the force law"
    In each window of about 3.5 cycles,

    $$
    \dot a_f = R\sin(\sigma_f - \phi_0) + C, \qquad R^2 = 1 - \frac{SS_{\rm res}}{SS_{\rm tot}},
    $$

    fitted as a linear model in $\sin\sigma_f$, $\cos\sigma_f$ and 1. Lagrange's equation gives this law for a
    resonance with one dominant term. Details: [the force law](physics/response.md#the-force-law).

### (f), (g) Eccentricity and inclination

Grey the osculating value, black the filtered value, $i$ in degrees. The grey numbers in the top left corner are the
smallest and largest filtered values in the analysed window.

### (h) The energy of each cycle

One point per cycle of $\sigma$: its energy $h$ in the pendulum fitted to the angle, on a logarithmic axis. $h = 0$
is the bottom of the well, $h = 1$ the separatrix (dashed).

- ● a libration cycle, ○ a revolution (a full turn);
- light blue band: the separatrix zone, where trial orbits in the fitted potential may stay trapped or go over the
  top: a cycle there can cross the separatrix;
- triangles at the bottom or top: $h$ below $10^{-3}$, or no top to the potential.

What to look for: a deep libration sits well below 1; points in or near the band mean the orbit is at the border of
the resonance.

??? info "The maths: energy and the separatrix zone"
    The potential $V$ is fitted to $\dot\sigma^2 = 2E - 2V(\sigma)$ cycle by cycle; each cycle's energy is

    $$
    h = \frac{E - V_{\min}}{V_{\max} - V_{\min}}, \qquad \text{for the cosine well } h = \frac{\dot\psi^2}{4\omega_0^2} + \sin^2\frac\psi2 .
    $$

    The zone $[h_{\rm lo}, h_{\rm hi}]$ comes from trial orbits in the fitted potential. With two harmonics a
    well's own separatrix can lie below $h = 1$. Details: [energy and potential](physics/response.md#energy-and-potential),
    [the separatrix zone](physics/geometry.md#the-separatrix-zone).

### (i) The centre of the angle

Grey: $\sigma_f$ in the analysed window; red: its centre, $\sigma_f$ smoothed over a few libration periods; large
red ▼▲: turns of the centre; light red shading: the centre's longest one-way run, drawn only when it reaches a full
turn. Red vertical hatching labelled `slow #id` (or `slow`): the intervals of the slow scale, a slower libration of
the centre stored apart from the main one (its period, cycles and cause are the `slow_*` columns and a `Slow scale:`
line of the right column); nothing is hatched without a slow swing.

What to look for: a centre that runs a full turn one way is a circulation, however the angle swings around it.

??? info "The maths: the centre"
    $\sigma_c = \mathrm{lowpass}_{W_c}(\sigma_f)$ with $W_c = 3P_{\rm cyc}$ (`LIBRATION_CENTRE_CYCLES`), and the
    centre's turns are found with a larger prominence than those of $\sigma_f$. Details:
    [the centre and circulation](physics/geometry.md#the-centre-and-circulation).

### (j) The centres of related angles

Each line is the centre of an angle minus its mean, in degrees, with its key at the right end: red the catalogue
angle, pale colours other members of its multiplet (the same commensurability with other perihelion terms), orange
an accepted subresonance, purple dashed the owner resonance.

What to look for: **a flat line is the angle that librates.** If the catalogue angle wanders while a member's line
is flat, the libration belongs to that member.

When the slow scale's cause is a secular cycle (`slow_cause` 2ω or $g - g_j$, at any level), the panel adds
$p\varpi$ of that scale, detrended, orange dashed, labelled with $p$ and the secular period from its column
(`−9ϖ 7.11 kyr`): the line that would explain the slow swing of the red centre.

For a kinematic libration (or a non-resonant row whose slow scale is a secular cycle) the panel shows instead
$\sigma = (\lambda\ \text{part}) + p\varpi$ on the scale of that swing, in degrees, each term with its straight-line
trend removed: red $\sigma$, orange dashed $p\varpi$ with its secular period (`−4ϖ 18 kyr`), grey dotted the
$\lambda$ part. The scale is the deciding episode at its response cutoff (285637: the 18-kyr cycle at 1.8 kyr), or
the slow scale, detrended over each slow episode (37087: 506 yr). When $\sigma$ follows $p\varpi$ and the $\lambda$
part stays flat, the perihelion's own precession bounds the angle, not the resonance; compare the swings, not the mean
rates.

??? info "The maths: the members' centres"
    A member $\sigma_s$ differs from the catalogue angle $\sigma_0$ by perihelion terms, $\Delta_s = \sigma_s -
    \sigma_0$, computed from the smoothed $\varpi$ of the body and the planets; its centre is drawn as
    $c_s = c_0 + \mathrm{smooth}_{W_c}(\Delta_s)$. Details: [subresonances](physics/response.md#subresonances),
    [another resonance owns the oscillation](physics/response.md#another-resonance-owns-the-oscillation),
    [kinematic origins](physics/response.md#kinematic-origins).

### (k), (l) Phase portraits

(k) is the whole record, (l) the accepted episodes (the five longest, each in its episode colour, up to 12 cycles
each, a dot at its first sample; `#2 induced` in its legend is an episode another resonance induces, `#0 < floor` a
run the status does not count). (l) draws each episode over the same interval as its fill in (a), cut to the
window of (k): solid between its first and last turning point, dashed beyond them. An episode that covers the whole
record is the whole window, so its (l) holds the same samples as (k). Both show $\sigma_f$ (modulo 360°) against $\dot\sigma$ in rad/kyr, with the same
scales; colour is time in (k). Red line: $\dot\sigma = 0$; red +: the centre of each accepted episode.

What to look for: a libration draws closed loops around the +; a circulation draws a band across the whole width.
A frame crossed out says why the samples cannot be drawn (an aliased step). When (k) is much wider than the
episodes, it uses the episodes' view and marks the clipped part with ▲▼.

### (m) The FAIR plane

The mean anomaly $M$ against the longitude difference to the planet, coloured by time (Forgács-Dajka, Sándor &
Érdi 2018). A libration draws slanted bands with empty gaps between them, since the body avoids some phases
relative to the planet; a circulation fills the square over time. It is drawn only for a prograde two-body angle, with the planet's longitudes saved and an output step of at
most 1/4.2 of the orbital period. Otherwise there is no (m) frame: the header says `FAIR no (reason)`, the legend
takes the slot, and one orange line at its top gives the reason in words, e.g. `FAIR not drawn: planet longitudes
not saved (SAVE_PLANETS); for a 1:1 FAIR = (M, σ) repeats (b)` (for a 1:1, $q = 0$: the plane would only show
the range of σ again).

??? info "The maths: FAIR coordinates"
    $x = M \bmod 360°$, $y = (\lambda_P - \lambda) \bmod 360°$ for a body inside the planet's orbit and
    $(\lambda - \lambda_P) \bmod 360°$ for one outside.

### The right column

The right column says in words what was decided and why. From top to bottom:

1. **The plaque**: `STATUS · angle_status (subtype) · CONFIDENCE`, coloured by the status; a tag `ends: …` beside
   it when the record ends in another regime (for example a capture, then circulation). The status is printed only
   here. The line under it gives the reason, the origin of the libration, the role of the other resonance as in the
   header (`driver 3N-5+0+2`, `owner …`, `neighbour …`, `resonant argument 9M-16+2+5`, `kinematic −4ϖ`, `also in …`) and the
   deciding rule by its meaning, with its ID in parentheses (`decided: circulation (C)`; every ID and its name:
   [names on the figure](libration-analysis.md#names)).
2. **The explanation**. On an MMR figure, short lines in a fixed order: the verdict in plain words, in bold
   (`NOT THIS RESONANCE: σ librates, but 3J+9S-2 does not make the swing.`); then • bullets: what $\sigma$ does
   (turning points, period, amplitude); the argument that decided, with its rule ID; an objection, a check that
   failed but was not read by this verdict (`Objection, not read by this verdict: …`); the cause; what limits the
   level (with the calibration cell and the action); the radial overlap of the body's band $q$…$Q$ with the planets'
   bands (`Radial overlap with Mars.`, `none`, or `no data`; needed for an encounter, not enough for one). The cause
   line is one of three kinds. `Driven by 11J-4+0-7 [driver: …]` stands first, right under the plaque, and only when
   the owner's measured regimes cover σ's deciding episode and Ψ stays bounded there (the decision's own test);
   otherwise `Candidate driver: X (its regimes; why it was not taken): not shown to make these cycles`. Without a
   driver: `Other resonances: not tested (reason)`, with `; not needed: a is not driven` at a kinematic leaf. A
   kinematic libration at a secular period puts `Driven by: secular cycle −4ϖ (the 2ω cycle π/|ω̇|, P 18 kyr)
   [measured]` first, where an owner would be (`measured` when the period matches and the swing is shown to be that
   term's: margin K1e on the episode, or the identified cause on the slow scale; else `candidate`); its cause line
   names σ's perihelion term, its share of σ's swing against the λ part (`over the episode −10ϖ spans 1 of σ's swing,
   the λ part 0.08`) and the swing the free and forced $e$ predict against the measured one (`the free and forced e
   predict ±28.7° for −4ϖ, the episode's half-amplitude is ±25.6°`); on the slow scale (rule K3s, 37087) the slow
   period, the matched cycle, both swings and the failed identity there. Any row with a slow scale prints it at its
   cause level: `Slow scale: σ's centre swings by ±22.2° with period 2.64 kyr (36 slow cycles, …): the e cycle
   2π/|g − g5| = 2.62 kyr (identified).` (153), `probably …` with `a slow libration of another resonance is not
   excluded`, or `cause not identified`; an induced row prints the identity (`σ = φ/8 + Ψ/8: …`: σ is in both
   resonances only if Ψ/8 stays bounded too), and, when its cause is found, the slow swing of σ's centre against the
   owner (`σ's centre swings against the owner with period 6.6 kyr (±11.5°): the 2ω cycle π/|ω̇| = 7.11 kyr.`). A rule C
   row names its resonant argument and the term by which the two angles differ (`Resonant argument: 9M-16+2+5, one
   resonance 9M-16 with this angle; the two differ by −2ϖ + 2ϖ_M, …`); a member the multiplet test rejected as too
   weak is called a frame, not a force (`A steadier centre of a member such as 10E-17+6+1 would be a frame, not a
   force: its term is 1e-9 of the strongest`; without a scan the cheap member list names the flattest weak members,
   `Members 10E-17+5+2 and 10E-17+6+1 give a flatter centre (W/W0 0.42, 0.45), but their terms are 7.6e-9 and
   1.8e-10 of the strongest: a frame, not a force`). The resonant argument says how it was chosen (`chosen by the
   multiplet theory: its centre π − arg P + jβ is the flattest over the episode`, or the flattest measured centre),
   `composite` when no single member stands, and what the other episodes chose in their own windows (`#1 76.2–99.8
   kyr chose no member in its own window`); a gap between the episodes where the composite term cancels is named
   (`The gap 48.7–76.2 kyr … |P|/max |P| falls to 8.84e-5 there (0.142 and 0.195 at its edges) as ϖ_M − ϖ passes its
   cancellation phase`, 12218). A diagnostic member scan (trigger T8) says what it found and that it changes
   nothing. A joint row says `Both resonances at once: x = Ψ/u: x closes 18 cycles of its own well (period …, off
   every secular period by …) while … holds φ`. Where the force law fails or only the direct law holds, a cause line
   names the forced lines that carry $\dot a$ (`ȧ's oscillation is mostly the 107 yr line 57 P_M = P_conj (7M-10):
   93% of its variance`; columns `rsp_line_*`). An uncertain row whose swing its perihelion term spans more than the λ
   part does reads `Suspected kinematic: …` with both shares (2003 EH16): a suspicion, not a status. A row the
   strength decided says that its episodes are drawn `forced (strength)` and that the driver is not identified; a row
   decided by the order-of-magnitude scale (rule 5x) says the scale is out of range and that this is a diagnostic,
   not an objection. When the action is a length-limited extend, the line after it says what the extension serves
   (`The extension serves the late regime: 2 force windows of 3.5 P for #0 (…); the failed potential stays
   measured`). A flag `extend_time` adds where and why (`Extend: a longer integration forward (next check: 2 kyr) would
   show how the record ends (σ_f slows to an arc after circulating; a clean 1:1 leg is cut by the end).`), and a
   circulation that ends in an arc says from when (`from 361 yr to the end σ_f slows to an arc`). An accepted run the
   status does not count is named under what σ does (`Not counted: #0 263–299 yr, P 74 yr < its floor 75.7 yr`).
   A 1:1 circulation judged on one leg (leaf G4c) says what the leg showed: `one horseshoe leg 586–665 yr, force and
   energy conserved: its local 1:1 law holds (K/K_theory 1.24), its energy integral passes and both turns reflect
   (out/in 0.836 and 1.2)` (transient, 2023 OE5), `one passing leg 779–934 yr between passages: it laps past the
   planet …, so no capture is shown and none excluded` (uncertain, 2018 KS), a clean leg cut by the record's edge
   (extend), or, at the circulation leaf, `its best 1:1 leg 552–613 yr turns by kicks (out/in 0.605 and 1.59), not
   reflections` (1991 VG, the near-episode cap). A non-1:1 half-swing between passages (leaf G4h) reads the same way
   with its one-harmonic law (42946 43S-6). Secular and Lidov-Kozai figures use the same slots, plus the time scales
   ([below](#the-secular-and-lidov-kozai-figure)).
3. **How it was decided**: first, over both columns, where the tree stopped and what it gave, for example
   `Decided at 5d → 6: force measured, then judged by behaviour → TRANSIENT`. Then the steps of the
   [status tree](physics/status-tree.md#steps-in-order) in their fixed order, each a question with the IDs it reads:
   `0 · Angle shape [G2]` (with the G rules of the [angle word](physics/geometry.md#from-geometry-to-angle_status)),
   `1 · Does a sibling angle hold the centre? [S]`, `2 · Does another resonance drive a? [W]`, `3 · Does a swing with
   σ? [B1]`, `4 · Does the centre drift without a cause?`, `5 · Is it σ's own force? [L4, R0]`, `6 · How does σ
   behave? [Z]`. The step where the tree took a branch carries its letter (`5d`). A step that ran says why it is asked
   ("Why: …"), then its answer. ● decided here, ↓ passed on, ◌ not run (a gap: the reason is given), ○ not needed (it
   did not run and the leaf decides without it: at a kinematic leaf the owner search is "n/a: no force on a"),
   · n/a (the owner search is not applied to a 1:1; no drift step without a drift). The steps after the leaf are one
   pale line, `4–6 · drift, own force, behaviour: not reached`. Step 0 names the angle rule in words: a two-body
   cycle below the 150-yr floor kept by its own potential (`G1f`), a ripple set aside (`F6r`), the slow centre turning
   back twice with the half-swing of this resonance's well (`G5w`, with the slow identity, the $a$ and period ratios
   and, beyond a planet's 1:2, `the 379 yr cycles are the orbital period`). For an `insufficient_time` angle the
   header names that rule (`Decided at G5w: slow swing in the well`). A circulation decided on one swing has two
   steps: step 0 the circulation (`→ a circulation judged on its best single leg → episodic`) and the leaf, `G4c · Is
   its one leg near the planet a capture?` or `G4h · Is its one half-swing the resonance's own?`. On a 1:1 angle step
   0 also names its own period floor (`1:1 floor of #0: … the loop through them takes 119 yr, so its own floor is
   59.4 yr (the tadpole floor 111 yr)`, 2003 YN107) and a turn read in the edge trim (`… it only closes the cycle`). Secular and Lidov-Kozai figures use the same form
   (the free-ω gate is step 1 of a Lidov-Kozai row). When the column is crowded, the "Why" lines are left out.
4. **Confidence: how firm the verdict is**. Every row reads `name [ID]: what was measured`. The limits come first,
   as ▼ `LIMIT: earlier a-wave [O1] → LOW`: a cap, with the failed margin that explains it printed inside it once (a
   failed cap margin is never drawn ✗), what it means, the other reading it leaves open and what on the figure would
   decide between them; `→ at most MEDIUM` when the cap sits above the shown level. Then the support margins: ✗ failed,
   ? not measured (K1 without a value says why), ⚠ thin, – skipped by design (with the reason); the full ones ✓ in one
   line `full: …`, the ones that do not apply · in one grey line without numbers. The O1 row prints its three
   parts when the ratio reaches its fail level: coherence at the episode's period, the phase gap and the share of
   the episode's own law (`… → not a foreign wave`); only a foreign wave fails it. R0 counts `k of n eligible
   windows` (a window counts with a half-swing ≥ 45° or a revolution) on the visited arc. L4 decided by the direct
   law reads `ȧ follows the averaged force of the term [L4]` with its R², scale band and null. The margins are
   explained on the [Confidence](physics/confidence.md#margins) page.
5. **The physics block**: A, "Is σ the own pendulum of …?" (the period and strength the cycles need against what
   the averaged potential gives, and the width in $a$); B, "What makes the cycles?" (the owner resonance and its
   evidence, or why there is none). Its row `both resonances at once?` asks whether σ is held by its own term too
   while the owner pins φ: x = Ψ/u over the tested window either moved ≥ 180° following no potential, or its
   acceleration follows the secular harmonics (2ω, ϖ − ϖ_j) and not sin x: `excluded`; else `not ruled out`, with
   each comparison printed the way it went; `both resonances at once` when x closes cycles of its own well, off every
   secular period, and `one cycle of x: not decided yet (extend)` with only one. The grey row `slow x` gives the slow swing of x and its cause at its
   level (identified, probably, or not identified: then a slow libration of another resonance is not excluded).
6. **The arithmetic**, two lines, for example `Support: no counted failures, 4 thin, 5 not measured → HIGH` and
   `Limits: earlier a-wave [O1] (a-wave before/inside 0.865 ≥ 0.5) → LOW`. The first gives the raw level from the
   support margins: two or more failed give low, one gives medium, any thin or unmeasured one high, all full certain.
   The second lists the limits, which can only lower it, and gives the shown level
   ([from margins to a level](physics/confidence.md#from-margins-to-a-level)).
   "Also measured, not read by this verdict" lists the numbers that exist but did not count. Under the legend, one
   line names the docs pages that hold the steps, the rules, the margins and the names.
7. **Numbers behind this verdict** (MMR), a short block over the numbers: one framed line per number that matters for
   this case, tagged in words (the frame's line style repeats the tag, so it survives a grey print). `DECISIVE` (solid
   frame): the comparisons of the rule that set the status, for example `a carries the swing [B1]: b1_ratio 0.0954 <
   0.1`; `LIMIT` (dashed): the cap that sets the shown level, with its comparison when it has one (`earlier a-wave
   [O1]: a-wave before/inside 0.865 ≥ 0.5 → LOW`); `CONFLICT` (dash-dot): a measured check that fails and that the
   verdict did not read; `UNMEASURED` (dotted): a test of the deciding rule that has no value. At most five lines;
   on a crowded figure the lines without a number go first (they repeat a LIMIT row), and on the most crowded the
   block goes. A number of the block is printed once: its row in the table below keeps its label and points up,
   `b1_ratio ↑ decisive [B1]`; a number the pendulum block above already prints stays there and the line names it.
8. **The numbers**, in monospaced columns: the episodes table (`EPISODES (1 accepted, 0 counted, 2 rejected runs)`:
   the runs the shape accepted and those the status counts; `*` marks the deciding episode, a grey `not counted: below
   the period floor` a run it does not count, ✗ a rejected run with its reason) and groups such as LIBRATION, RESPONSE OF a, PENDULUM ENERGY ZONE, SUBRESONANCES OWNER, RESONANCE
   RATES, STEP ALIAS FILTER, STRENGTH, KINEMATICS, ORBIT, RECORD. A bold number was compared by the decision; a grey
   one marked "(not used)" belongs to a test skipped by design. When the column is full it ends with "… N more
   lines: every number is in summary.csv". The pendulum numbers are labelled in words with their source:
   `observed cycle period P` is measured on σ; `small-swing period T0 (from observed cycles)` is inferred from those
   cycles (T0 = P/F(A), the amplitude correction of the cosine pendulum; never a theory), and so are
   `well half-width in a (from T0)` and `strength needed μ_eff (effective, not a mass)`; `theory period (strength)`
   is the period the planet's mass predicts (block A compares it with T0). `cycle energy h (1 = top of the fitted V)`
   is normalised to the fitted potential, the cosine separatrix only when that has one harmonic. Two columns of one
   quantity print as one range (`separatrix layer in h 0.9–1.1`). Every other number keeps its column name without
   the prefix and the unit; the `all` form uses the same labels.

With `plot_options={'combined': {'numbers': 'all'}}` the right column lists every number of the decision in blocks
instead of the explanation.

## Co-orbital (1:1) figures

![An Earth co-orbital, 2002 AA29](https://smirik.github.io/resonances/img/gallery/coorbital__2002_AA29__1E-1+0+0.png)

*2002 AA29, an Earth co-orbital alternating between horseshoe and quasi-satellite motion
([Gallery](physics/gallery.md#resonant-coorbital)).*

The figure is the same, with these differences:

- the header names the regime (`tadpole L4`, `horseshoe`, `quasi-satellite`, `compound (horseshoe ↔ quasi-satellite)`),
  or for one episode in several wells `orbit: …`, the wells of its sub-regimes in time order (`orbit: horseshoe →
  tadpole L4`, 2016 JA; more than four are shortened with `…` and their count; summary `geo_orbit_type_sequence`);
- (a) brackets the sub-regimes and hatches the inferred residence, (d) marks the kicks of $a$ (above); the first
  bullet of the right column says the same in words: `σ librates, measured, from 443 yr to 601 yr …; stays in the
  quasi-satellite well from 601 yr to the end (inferred, not measured)` (2020 CB), `horseshoe 129–339 yr → tadpole
  L4 339–537 yr` (2016 JA), `its amplitude changes at ~326–446 yr and ~708–809 yr` (2021 XO3, one quasi-satellite
  episode, not three), `kick of a at ~4.69 kyr` (2024 TY11 20 kyr); the EPISODES block lists the stay (`stay
  start–end (inferred)`) and the marks (`well 339 yr`, `amplitude 326–446 yr`, `kick 4.69 kyr`) under the episode;
- (e) draws no sine windows: a co-orbital is judged by its own force law, and the panel prints its $R^2$ (with the
  two halves) and $K/K_{\rm th}$, the fitted strength over the planet's (1 means the planet's own mass);
- block A of the right column names the 1:1 force law; block B is left out, and the owner step says that a 1:1 angle
  has no owner;
- FAIR is drawn when the output step allows it.

??? info "The maths: the co-orbital force law"
    $$
    \dot a = 2\mu n a\,\tilde F(\theta), \qquad \dot a_{\rm data} = K\,\dot a_{\rm model} + C,
    $$

    with $\tilde F$ from the averaged direct and indirect potential of the planet. Details:
    [co-orbital force law](physics/response.md#co-orbital-force-law), [co-orbitals](physics/pendulum.md#co-orbitals-11).

## The secular and Lidov-Kozai figure

![The ν6 resonance of (1222) Tina](https://smirik.github.io/resonances/img/gallery/secular__1222__g-g6.png)

*(1222) Tina in the ν6 secular resonance ([Gallery](physics/gallery.md#secular-nu6)).*

The page and the right column follow the MMR figure, but the time panels are fewer and **their letters differ**.
The header gives the span, step, number of rows, the cutoff $P_c$ and what set it, and the scale the verdict rests
on (`fine`, or `centre` when the slower centre scale decided). A Lidov-Kozai figure adds the plane $\omega$, $i$ and
$\Omega$ are measured from (`ω, i, Ω from the invariable plane`, `ZLK_ANGLE_FRAME`): (a), (a′), (e) (labelled
`i (invariable)`), (f), (h) and (i) all read that frame, the free elements included.

| panel | what it shows |
|---|---|
| (a) | $\sigma$ in degrees: grey the angle the classifier reads (the reduced proper angle by default), black the angle filtered at the scale the verdict rests on, a thin grey line the angle at $P_c$ when the centre scale decided, red the centre, ▼▲ turning points, fills the librating episodes, grey strips with letters the other runs (as in MMR (a)) |
| (a′) | Lidov-Kozai, an induced libration only (attributed to a forced mode, or a centre whose slow line is a beat of the forced vector): $\omega_f$ with the forced direction against the node and the dominant mode against the node (below); other rows have no (a′) and keep the seven panels in place |
| (b) | $\sigma$ modulo 360° around the centre: grey dots and black $\sigma_f$ |
| (c) | ${\rm d}\sigma_f/{\rm d}t$, in ″/yr for a secular angle and °/kyr for Lidov-Kozai; red line at 0 |
| (d), (e) | $e$ and $i$ (degrees): grey osculating, black filtered at $P_c$, with min – max |
| (f) | secular: the rate of the element the angle contains against a sine of $\sigma_f$ (below); Lidov-Kozai: the drift of the free and the osculating $\omega$ in turns (below) |
| (g) | the centre of $\sigma_f$, its turns, and its longest one-way run when it reaches a full turn (as MMR (i)) |
| (h) | the portrait $(e\cos\sigma, e\sin\sigma)$, or $(\sin\frac i2\cos\sigma, \sin\frac i2\sin\sigma)$ for an angle of $\Omega$ alone; colour: time; red + at the origin |
| (i) | secular: $\sigma_f$ against ${\rm d}\sigma_f/{\rm d}t$; Lidov-Kozai: the free eccentricity vector, or $\sigma_f$ against its rate when the free elements were not built (the axis labels say which) |

What to look for: in (a) a libration needs at least three turning points, one full cycle
(`LIBRATION_MIN_TURNING_POINTS`); a drift is a ramp of less than a turn; a single U is half a cycle; circulation is
a ramp of several turns. In (h) a libration draws loops around the resonance centre: a ring off the origin, or a
crescent when the libration is wide and the centre close to the origin. A ring around the origin is circulation.

**(f) for a secular angle.** Black is $\dot e_f$ for an angle with $\varpi$, or ${\rm d}\sin(i_f/2)/{\rm d}t$ for an
angle of $\Omega$ alone (both, $i$ dashed on a right axis, when the angle has both); blue is the fit
$A\sin\sigma_f + B\cos\sigma_f + C$, with its $R^2$ written in blue. That $R^2$ is a column of the output and changes
no status. Black, blue and $R^2$ are one fit, the analysis' own: the element and $\sigma_f$ filtered at the decisive
cutoff, on its window. When the centre scale decided (header `scale centre`) that cutoff is the centre's, longer
than the $P_c$ of (d) and (e), and the panel says `at the centre scale … kyr` after the $R^2$.

**(f) for Lidov-Kozai.** The drift of $\omega$ from its first value, in turns (as the standalone
[free-ω drift](#free_omega-lidov-kozai) figure), with a legend: grey the osculating $\omega$, blue the free $\omega$
with the forced terms of the planets removed; the free $\omega$'s net turns are written top right (the panel says
`free elements not built` when there are none). A free $\omega$ that winds through full turns while the osculating
one stays near zero points to a kinematic libration: the planets' forcing holds $\omega$, not the Lidov-Kozai
mechanism (591986: 83 turns; [Gallery: kinematic](physics/gallery.md#zlk-kinematic)); one that stays within a
fraction of a turn supports Lidov-Kozai. The picture is not the verdict: the right column states the free-ω gate's
outcome (confirmed, kinematic or no verdict) with $T/P_{\rm cyc}$, the cycles of the osculating $\omega$ in the
record, and the minimum needed. Too short a record, an undefined phase, a planetary frequency left in the free
elements or a co-orbital of Mercury to Mars give no verdict, and an angle that does not librate throughout is demoted
only under the extra condition $\rho_{\max} < 1$ ([the free-ω gate](physics/secular-zlk.md#zlk-free-omega)).

**(a′) for an induced Lidov-Kozai libration.** Grey $\omega_f$; orange $\arg z_{\rm forced}(t) - \Omega(t)$, the
direction of the fitted forced eccentricity vector against the node: the point the free circle turns about, so
$\omega_f$ oscillates around it with the free–forced beat ($P_{\rm cyc}$, written bottom right), and it carries the beat
of the forced modes (the legend names the two largest $|c_j|$ and their beat: 591986, g5 + g6, 54 kyr); purple dashed
$\varpi_j - \Omega$, the longitude of the dominant mode against the node, with the node lock when there is one
(591986 over 4 Myr: locked to g5, 50–130°, a slow wave of 1.4 Myr) or the rate $\dot\Omega - g_j$. A libration of
$\omega$ that follows the orange curve is the forcing's, not the Lidov-Kozai mechanism's. When the curves leave a
turn (624 Hektor: $\omega$ circulates between its episodes, the centre $\varpi_5 - \Omega$ turns with the node) the
panel is drawn modulo 360° about the forced direction, and $\omega_f$ only inside the librating episodes.

**The explanation** of the right column is the MMR figure's bullets in the same slots, in plain physics:

- the verdict in words (`UNCERTAIN: ω librates, but the free-ω gate gives no verdict.`);
- what the angle does. An episodic angle counts its librating episodes only (those not faster than the type's
  shortest libration) and names the deciding one: `ω librates in 3 separate episodes …; the deciding one 8.08–18.1
  kyr, … ; 2 more accepted runs not counted`;
- the argument that decided:
    - the free-ω gate's outcome, said physically: `The free node is not yet separated from s6: |b_Ω − s6| 1.85″/yr
      < 2.59″/yr (1/T …); they separate after 700 kyr` (1981 Midas: not a lack of cycles), or `the free-element model
      does not apply to a co-orbital of Earth` (2020 PP1);
    - for a secular angle, its shape: `The swing 21.9° ≤ 33.9°, the phase uncertainty of g2, g4, g6 (mostly g4) …`
      (5026: rerun with measured phases), or `The centre turned back within less than one revolution: … not a drift`
      (2306);
- `Objection, not deciding the status: …`: $e$ (or $\sin(i/2)$) does not follow the angle; only one independent
  centre estimate; a ZLK libration without the $e$–$i$ exchange at its deciding period. These leave the grey line;
- for a Lidov-Kozai row, the cause:
    - its physics: the ranges of $i$ and $e$, where $i$ stands against Kozai's 39.2° (the classical quadrupole ZLK
      under an outer planet; a literature reference printed for reading, not a test);
    - how many times the free $\omega$ turns in the record, $|g - s|/(1/T)$ (555527: about 8);
- the time scales:
    - $1/T$, the record's frequency resolution in ″/yr;
    - for a secular angle $|\dot\sigma|/(1/T)$, the turns of $\sigma$'s mean rate in the record (5026: 0.05, below
      the resolution; 2306: 1.24);
    - the search window, from $2P_c$ (or the type's shortest libration) to one full cycle inside the trimmed record;
    - the period seen on the angle with the FREQUENCIES rows within $1/T$ of it (`≈ g − g5`), or `unidentified` with
      the nearest row;
    - one sentence: *planetary periods are precession times; they are not the period of a libration*;
- what limits the level, the cell and the action; the radial overlap with the planets' bands.

A number a bullet prints is not printed again: its row in the numbers keeps the label and reads `↑ in the text`.

**The cause** of a kinematic libration is also a bullet of the right column, from the measured columns only: the
mode that holds $\varpi$, the beat the free vector circles it with, what the centre's slow line is, and the node
(locked, or following the mode with no full cycle of $\Omega - \varpi_j$ in the record: a lock not resolved; for an
episodic row, the centre's turns). Not attributed: "Cause not attributed" and why.

**The ZLK lines of the right column** ("Also measured, not read by this verdict"), each only when it was measured and
not already in the cause sentence: the $e$–$i$ coupling at the decisive period (`zlk`: antiphase at the constant-$H$
amplitude; `forced beat`: in phase at the amplitude of a forced-eccentricity beat; with the phase of $i$ against $e$,
the band correlation $r$ and the amplitude over each law; coupling is not libration), the forced mode that holds the
libration, the node lock (a flag), and the slow line of the centre with its origin (591986: 53.8 kyr, ±21.6°, the
g6−g5 beat of the forced vector). What they mean: [Free elements: beyond the gate](free-elements.md#beyond-the-gate).

**The numbers of the right column** add two groups to those of the MMR figure. ORBIT gives $a$ (median, min, max,
au) with $e$ and $i$. FREQUENCIES is a table of rates in ″/yr with their periods $1\,296\,000''/|f|$ (the time a
rate needs for a full turn), to read the periods seen on $\sigma(t)$:

- secular: the body's $g$ and $s$ (the mean rates of its free vectors), $\sigma$'s rate from them and the modes,
  the modes g5, g6, s6 and those of the angle (`in σ −1`: the mode and its coefficient), and for an angle of two or
  more modes the forced waves $g - g_j$, $s - s_k$ (2306 $g-g_6+s-s_6$: 43.9 and 43.6 kyr);
- Lidov-Kozai: the free rates $g$, $s$ and $g - s$ (the free $\omega$), every fitted forced mode with its amplitude
  $|c_j|$ (◀ the mode that holds $\varpi$), the beats $g - g_j$ of the two largest forced modes (`kinematic`: the
  cycle of a kinematic $\omega$), their mutual beat (`beat`: the slow wave of the forced vector) and, when flagged, the
  node lock $\Omega - \varpi_j$ with its period and range. 591986 over 4 Myr: g5 304 kyr |c| 0.045 ◀, $g - g_5$
  6.11 kyr kinematic, g6 − g5 54 kyr beat, $\Omega - \varpi_5$ lock 1.4 Myr 50–130°.

The planetary rates are the package's constants (`PLANETARY_FREQUENCIES`), the body's are measured on the record:
their precision is about one turn over the record, $1\,296\,000''/T$.

The margins of the right column are N, S, F, P and G ([secular and Lidov-Kozai margins](physics/confidence.md#secular-and-lidov-kozai-margins)).

??? info "The maths: the secular angle and the free ω"
    A secular angle combines the body's proper longitudes of perihelion and node with the planets' mode phases,
    $\sigma = p\varpi + q\Omega + \sum_j(p_j\ell_{g_j} + q_j\ell_{s_j})$, on the body's vectors with the forced terms
    at the other planetary frequencies removed ([the secular angle](physics/secular-zlk.md#the-secular-angle)). The
    cutoff is the largest of the great-inequality term, the forced waves near the angle's frequency and half the
    shortest libration allowed ([secular signal](physics/secular-zlk.md#signal-what-is-not-the-libration)).

    For Lidov-Kozai, $z_e = e\,e^{i\varpi}$ and $z_i = \sin\frac i2\,e^{i\Omega}$ are split into forced parts (fitted
    at the planets' frequencies) and free parts, and $\omega_{\rm free} = \varpi_{\rm free} - \Omega_{\rm free}$
    ([what is computed](free-elements.md#what-is-computed), [the gate](free-elements.md#the-gate),
    [why a bounded ω is not enough](physics/secular-zlk.md#zlk-free-omega)).

## Standalone figures

Each kind below is drawn when it is listed in `plots`. The examples are drawn by the package from the saved
benchmark series.

### evolution

![evolution figure of (1222) Tina in g-g6](https://smirik.github.io/resonances/img/figures/evolution.png)

Stacked panels (preset `PLOT_CONFIG`, `full` by default): the angle as integrated and filtered (wrapped and
unwrapped, radians), the filtered $a$ (au), $e$, $i$ (radians), then Lomb-Scargle periodograms of the filtered
angle, $a$ and $e$. The time axes are in years; **the periodograms are drawn against period**, from 0 to the length
of the record, not against time. Blue × marks a peak, grey dashed lines its width; the red and green dashed lines
are the power levels 0.05 and 0.1. The semi-major axis here is smoothed by the body-level filter
(`LIBRATION_FILTER_CUTOFF`: by default periods shorter than 2000 yr are removed), not at the analysis' cutoff.

### recurrence (MMR)

![recurrence of (463) Lola in 4J-2S-1](https://smirik.github.io/resonances/img/figures/mmr_recurrence.png)

Each pixel compares the state of the angle at two times $t_1$ and $t_2$: dark means the angle and its rate are
almost the same at both times. A libration gives a pattern of dark diagonal stripes, one per period, as here for
(463) Lola (period about 10 kyr); a circulation never returns to a state it left, so it stays light away from the
diagonal. The white band along the diagonal is left out on purpose.

??? info "The maths: the recurrence distance"
    $$
    D(t_1, t_2) = \sqrt{\left(\frac{\Delta\sigma}{s_\sigma}\right)^2 + \left(\frac{\Delta\dot\sigma}{s_{\dot\sigma}}\right)^2}
    $$

    on the unwrapped $\sigma_f$, with $s_\sigma$ half the median diameter of a cycle and $s_{\dot\sigma}$ half the
    5–95 % spread of $\dot\sigma$. These are the object's own scales: D compares states within one figure, not
    between objects. The matrix is thinned to about 8 samples per cycle (700–2500 points).

### fair (MMR)

![FAIR plane of (1921) Pala in 2J-1](https://smirik.github.io/resonances/img/figures/mmr_fair.png)

The same plane as panel (m) of the combined figure, for a prograde two-body angle. A libration leaves empty gaps
between slanted bands, as for (1921) Pala in the 2:1 resonance here; a circulation fills the square over time. Unlike (m), this figure is drawn even when the output step
is too coarse: above `fair.max_step_fraction` (¼) of the orbital period it warns that the strips are aliased.

### portrait (MMR)

![phase portrait of (463) Lola in 4J-2S-1](https://smirik.github.io/resonances/img/figures/mmr_portrait.png)

$\sigma_f$ (shifted by whole turns only) against $\dot\sigma$ in degrees per display unit, coloured by time; grey
dots are the osculating values; red line: $\dot\sigma = 0$. With `portrait.y='axis'` the y axis is $a - a_0$ in
$10^{-3}$ au: the analysis' own $a_f$, as in the cycles below, $a_0$ its median over the focus (the osculating $a$
when the analysis' arrays are not held). A libration draws a closed loop. The first and last 1/`LIBRATION_FILTER_CUTOFF` years (2000 yr by
default) are left out, where the filter is distorted.

### cycles (MMR)

![cycles of (463) Lola in 4J-2S-1](https://smirik.github.io/resonances/img/figures/mmr_cycles.png)

Every cycle of $\sigma_f$ from one maximum to the next, stretched to [0, 1] and laid on top of each other (left),
and $a$ over the same cycles minus each cycle's median (right, $10^{-3}$ au), coloured by the cycle's midpoint; $n$
is the number of cycles. Cycles that lie on top of each other are a steady libration, and $a$ swings in step with
$\sigma$, a quarter cycle apart. A cycle here is a candidate oscillation, not a confirmed libration. The right panel
uses the analysis' own filtered $a_f$ (at the angle's cutoff, as panel (d) of the combined figure); when the
analysis' arrays are not held (a row not selected for plotting) it uses the osculating $a$, never the body-level
filter of `evolution`, which would erase a libration faster than 1/`LIBRATION_FILTER_CUTOFF` (2000 yr by default).

### angle, rate, portrait (secular and Lidov-Kozai)

![angle of (1222) Tina in g-g6](https://smirik.github.io/resonances/img/figures/secular_angle.png)

**angle**: $\sigma(t)$ in degrees; grey the osculating angle (folded into the band of the black curve when that one
is bounded), black the angle at the scale the verdict rests on ($P_c$, or the centre scale), red circles the turning
points. Read it as panel (a) of the combined figure: (1222) Tina librates through 33 turning points in 10 Myr. A
slow libration needs a long record: (1020) Arcadia in z1 turns back three times in 10 Myr and is resonant
([Gallery](physics/gallery.md#secular-z1-resonant)); (3701) Purkyne, also in z1 with a period of about 8 Myr, turns
back only twice in 10 Myr and is `uncertain` (`insufficient_time`).

![rate of (1222) Tina in g-g6](https://smirik.github.io/resonances/img/figures/secular_rate.png)

**rate**: the rate of the black curve in ″/yr (also for Lidov-Kozai, unlike the combined figure), the red line at
zero and a dashed line at the mean rate, the "divisor" printed in the title. A libration swings about zero; a drift
runs parallel to it.

![portrait of (1222) Tina in g-g6](https://smirik.github.io/resonances/img/figures/secular_portrait.png)

**portrait**: $(r\cos\sigma, r\sin\sigma)$; grey from the osculating elements, coloured by time from the classified
angle; red + at the origin. Here $r$ is $e$ when the angle contains $\varpi$, otherwise $i$ in degrees from the
invariable plane (the combined figure's (h) uses $\sin\frac i2$ instead). Tina's ring sits off the origin: a
libration.

### ecc_vector

![eccentricity vector of (591986)](https://smirik.github.io/resonances/img/figures/zlk_kinematic_ecc_vector.png)

The plane $(k, h) = (e\cos\omega, e\sin\omega)$, coloured by time, in a square window centred on the origin; one
figure per body. A cloud that does not enclose the origin keeps $\omega$ bounded whatever the dynamics: it explains
why $\omega$ librates, not whether the libration is a resonance. Here (591986)'s cloud is off the origin, yet its
libration is kinematic (below).

??? info "The maths: forced and free eccentricity"
    The cloud's centre is offset from the origin by $e_{\rm forced}$ and its radius is about $e_{\rm free}$, the
    columns `e_forced` and `e_free` of summary.csv
    ([forced and free eccentricity](coherence.md#forced-and-free-eccentricity)).

### free_omega (Lidov-Kozai)

Three figures from the forced/free split ([Free elements](free-elements.md#files-and-figures)), in the frame of the
Lidov-Kozai row (`ZLK_ANGLE_FRAME`, the invariable plane by default): the osculating ω of the drift figure is the
row's classified angle.

![free omega drift of (3040) Kozai](https://smirik.github.io/resonances/img/figures/zlk_free_omega_drift.png)

![free omega drift of (591986)](https://smirik.github.io/resonances/img/figures/zlk_kinematic_free_omega_drift.png)

**drift**: how far $\omega$ has moved since the start: blue the free $\omega$, a thick grey line the osculating one,
in degrees, or in turns when the drift exceeds three turns. A confined band is a libration, a ramp a circulation. Top:
(3040) Kozai, where both stay within a band of some 70°: a Lidov-Kozai libration. Bottom: (591986), where the osculating
$\omega$ stays bounded but the free $\omega$ winds about 80 turns in 500 kyr: kinematic.

![free omega portrait of (3040) Kozai](https://smirik.github.io/resonances/img/figures/zlk_free_omega_portrait.png)

**portrait**: free $e$ against free $\omega$ (0°–360°), coloured by time, with red dotted lines at 90° and 270°. A
librator draws closed loops around one of them (the centre can shift inside a mean-motion resonance); a circulator
draws a wave through every $\omega$.

![free eccentricity vector of (591986)](https://smirik.github.io/resonances/img/figures/zlk_kinematic_free_omega_vector.png)

**vector**: the free eccentricity vector $(e_{\rm free}\cos\omega_{\rm free}, e_{\rm free}\sin\omega_{\rm free})$
on exactly the window of the osculating `ecc_vector` figure, so the two can be compared by eye. For (591986) the
free vector circles the origin while the osculating cloud is offset from it: the forced eccentricity, not a
resonance, kept the osculating $\omega$ bounded.

### cross_spectrum

Coherence and phase of two series (for example $e$ and $i$) against period on a log axis, and below them the two
series band-passed at the decisive line over 16 of its cycles. Everything is cropped to the search band, and the
axes pick their own unit (yr, kyr, Myr).

- **Coherence panel.** Each segmentation level has its own shade of blue: its full spectrum faint, a dotted line at
  its significance threshold, and a stem at each detected line; open red rings mark up to five strongest lines of
  each level. A red dashed vertical line marks the decisive line (the strongest at the coarsest valid level; it
  need not lie in a yellow band).
- **Yellow bands** mark where to look: every level whose validity band covers that period is above its own
  threshold (at least two such levels; a level that cannot see the period abstains), and the band holds one of the
  decisive level's three strongest lines. Agreement alone would not be selective: when the Kozai integral
  $\sqrt{1-e^2}\cos i$ is conserved, $i$ is a smooth function of $e$ and the two are coherent at every period with
  power, so a highlight of all agreement would say only that the integral holds.
- **Phase panel.** Each line is a marker sized by its cross-amplitude, filled when every level agrees; light green
  bands are the expected phase ± tolerance (180° ± 30° for the exchange of $e$ and $i$).
- **Band-passed panel.** This one settles it: two curves in antiphase are an exchange, two curves drifting past each
  other are not.

The method: [Cross-spectral coherence](coherence.md#how-it-is-measured). No example image: the benchmark examples
are saved without coherence.

### clones

Three panels for a body integrated with clones ([Clones](clones.md)): (a) the angle (0°–360°) of every clone,
coloured by its status, the nominal orbit in black; (b) each clone's status against its initial $a$ (minus the
nominal's), the nominal a black ★, with the catalogue resonant axis $a_{\rm res}$ of an MMR (dotted, `a_res
(catalogue)`: from the mean motions, not moved into the osculating frame of the x axis) and $a_{\rm res}$ ± the
median half-width of the clones' episodes (blue dashed); (c) the share of each status
with its Wilson 95 % interval, ★ at the nominal's status. The title gives the number of clones, the source of the
orbit solution, its epoch and the most common status. No example image: drawing it needs an integration of the
clones.

## What to look at first

Start from the plaque, then check the panels the verdict rests on. The [Gallery](physics/gallery.md) has a figure
for each case below.

- **resonant** ([two-body](physics/gallery.md#resonant-two-body), [three-body](physics/gallery.md#resonant-three-body)):
  in (a) the fills cover the record from edge to edge; in (c) the line changes sign at each turning point and stays
  near its episode's blue dashed limits (not necessarily ±1, and only roughly for a well that is not a cosine);
  in (h) the dots lie below 1; (k) and (l) show closed loops around the +; (e) shows high $R^2$.
- **transient** ([capture and release](physics/gallery.md#transient-recurrent),
  [after a jump](physics/gallery.md#transient-before-jump)): fills in part of (a) only; circulation or a drift between
  them; hatched fills and jump lines when a jump of $a$ ended the libration; the tag `ends: …` says how the record ends.
- **a subresonance** ([example](physics/gallery.md#transient-subresonance)): in (j) a member's line is flat while the
  catalogue angle wanders.
- **near_separatrix** ([example](physics/gallery.md#near-separatrix)): no accepted episode, but the angle stalls: ○
  slips on the zero line of (c), or revolutions in (h) close to the separatrix zone
  ([near the separatrix](physics/geometry.md#near-the-separatrix)).
- **non_resonant, circulation** ([example](physics/gallery.md#circulation)): the centre in (i) runs a full turn
  one way (shaded); the recurrence figure stays light away from its diagonal.
- **non_resonant with a libration** ([kinematic](physics/gallery.md#kinematic-libration),
  [induced](physics/gallery.md#induced-owner-librates)): the angle librates in (a), but in (j) $\sigma$ follows $p\varpi$
  (kinematic) or the owner's purple line is the flat one (induced); the header reads `episodes about X° (not this
  resonance)`.
- **an alias** ([example](physics/gallery.md#alias)): the red ALIAS note in (b), and the portraits crossed out.
- **uncertain** ([too short](physics/gallery.md#insufficient-time), [origin unknown](physics/gallery.md#uncertain-origin)):
  the sentences and the action say what is missing (a longer record, a finer step) and how long to integrate.
- **secular** ([ν6](physics/gallery.md#secular-nu6), [z1](physics/gallery.md#secular-z1-resonant)): read (a) with the
  turning-point rule above and (h) for a ring off the origin.
- **Lidov-Kozai** ([confirmed](physics/gallery.md#zlk-resonant), [kinematic](physics/gallery.md#zlk-kinematic)): in
  (f) a free $\omega$ that winds points to kinematic, a confined one supports the libration; the gate's outcome in
  the right column is the verdict. An offset eccentricity-vector cloud alone proves nothing.
