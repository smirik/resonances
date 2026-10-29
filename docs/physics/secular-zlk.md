# Secular and Lidov-Kozai angles

Secular and Lidov-Kozai (ZLK) angles evolve over $10^4$–$10^7$ yr instead of $10^2$–$10^4$ yr, and they involve only
the slow angles $\varpi$, $\Omega$, $\omega$. They go through the same [geometry](geometry.md) as mean-motion
resonances (turning points, cycles, episodes, coverage), with their own signal step and their own physics. The
status tree is short ([below](#geometry-and-status)): there is no semi-major axis to respond, no owner and no
subresonance.

`analysis.secular.analyse_secular(series, settings)` and `analysis.zlk.analyse_zlk(series, settings, free_elements=...)`
read a `LongPeriodSeries`: the classified angle (the reduced proper angle of a secular resonance, $\omega$ for ZLK) with
$a$, $e$, $i$, $\varpi$, $\Omega$ on a uniform grid of years. `Simulation.identify_librations` calls them on every path
(a run, a batch worker, `restore(recompute_librations=True)`).

??? info "The steps of the analysis"

    | step | what |
    |---|---|
    | 0 input | an unphysical orbit ($e > 1.3$, $a < 0$, non-finite $a$ or $e$) → `invalid` / `unphysical`; fewer than three samples, or non-finite times or angles → `uncertain` / `unusable_series` (angle `undefined` in both) |
    | 1 cutoff | $P_c$ and the turning-point prominence (below) |
    | 2 filter | $\sigma_f$ at $P_c$, $P_c/2$ and $2P_c$; the window $[t_0 + P_c,\ t_1 - P_c]$. When $\sigma$ cannot be filtered at $P_c$ (a grid not uniform and increasing within `FILTER_GRID_RTOL`, or $P_c \le 2\Delta t$) → `uncertain` / `unusable_series`, no geometry (an MMR gives `invalid` / `undersampled` there) |
    | 3 geometry | turning points, runs, episodes, the centre and its passes, exit and entry (shared with MMRs) |
    | 4 angle | `angle_status` from the geometry (below); without a libration at $P_c$, the same at the scale of the centre |
    | 5 physics | secular: the $e$/$i$ response (a column); ZLK: the free-$\omega$ gate |
    | 6 status | `analysis.secular.decide.decide_status` |
    | 7 confidence | the margins N, S, F, P, G and the caps (below) |

## The secular angle

**What should librate?** A linear secular resonance locks a precession frequency of the body to one of the
planetary system's eigenfrequencies: $g \approx g_6$ ($\nu_6$), $s \approx s_6$ ($\nu_{16}$), or a combination such
as $g + s - g_6 - s_6$ ($z_1$). In the Laplace-Lagrange theory the planets' eccentricity and inclination vectors are
sums of eigenmodes, each turning at its own rate $g_j$ or $s_j$ (Murray & Dermott 1999, eqs. 7.28–7.29, p. 278).
The resonant angle therefore compares the body's $\varpi$ and $\Omega$ with the planets' **mode longitudes**, not
with their osculating elements, after removing the forced terms that are not in the angle. Why both matter, and the
measured phases: [Secular resonances](../secular.md#how-the-angle-is-built).

??? info "The angle σ and the reduced vectors"

    The mode longitudes are

    $$
    \ell_{g_j}(t) = g_j\,(t + t_{\rm epoch}) + \phi_{g_j}, \qquad \ell_{s_j}(t) = s_j\,(t + t_{\rm epoch}) + \phi_{s_j},
    $$

    with the phases quoted at J2000 ($t_{\rm epoch}$ the offset of the run's start from J2000).

    **Units.** In the formulas of this page $g$, $s$ and every frequency $f$ are angular frequencies in rad/yr. The
    stored `PLANETARY_FREQUENCIES` are in ″/yr and are multiplied by $\pi/648\,000$; one cycle per year is
    1 296 000 ″/yr. A record of length $T$ resolves frequencies $2\pi/T$ rad/yr apart (one Rayleigh width, $1/T$ in
    cycles/yr).

    The body enters through its complex eccentricity and inclination vectors, built from the saved (ecliptic)
    elements, $z_e = e\,e^{i\varpi}$ and $z_i = \sin\tfrac{i}{2}\,e^{i\Omega}$ (Murray & Dermott use $I$ in radians;
    since $\sin\frac i2 \simeq I/2$, phases and frequencies agree, while inclination amplitudes here are **half**
    theirs). The forced terms at the fundamentals that are **not** in the angle only shake it, so they are removed by a
    least-squares fit at their known frequencies $f_j$:

    $$
    z_{\rm red} = z - \sum_{j \in B} C_j\, e^{i f_j t}.
    $$

    The basis $B$ is that of the [free elements](../free-elements.md): $g_5$–$g_8$ plus a constant for $z_e$, and the
    constant ($s_5 = 0$, the invariable plane) plus $s_6$–$s_8$ for $z_i$; inner-planet modes are not subtracted. Modes
    closer than two Rayleigh widths cannot be separated, so the candidates are first clustered and one representative
    of each cluster is kept; the representatives whose labels occur in the angle are dropped (their terms are the
    resonance), and the rest is fitted and removed. With $\varpi = \arg z_{e,\rm red}$ and $\Omega = \arg z_{i,\rm red}$:

    $$
    \sigma = p\,\varpi + q\,\Omega + \sum_j \big(p_j\,\ell_{g_j} + q_j\,\ell_{s_j}\big),
    \qquad p + q + \sum_j (p_j + q_j) = 0 .
    $$

### Signal: what is not the libration

The filter must remove everything faster than the shortest-period libration allowed, and nothing slower:

$$
P_c = \max\Big(F_{\rm GI}\,P_{\rm GI},\ \ F_w \max_j \frac{2\pi}{\lvert f - f_j\rvert},\ \ \frac{P_{\min}}{f_{\min}}\Big).
$$

- $P_{\rm GI}$ = 937 yr, the Jupiter-Saturn great inequality, the longest short-period term of an asteroid's
  elements; $F_{\rm GI}$ = `SECULAR_FILTER_GI_FACTOR` = 2.
- For an angle with two or more planetary modes, each mode $j$ makes a **forced wave**: its forced term turns
  relative to the free one at $f - f_j$ ($f$ the body's own $g$ or $s$). These waves are not librations; $F_w$ =
  `SECULAR_FILTER_FORCED_FACTOR` = 2. A single-mode angle ($\nu_6$) has none: its forced term is the resonance.
- $P_{\min}$ = `SECULAR_LIBRATION_PERIOD_MIN_YR` = 200 kyr, the shortest secular libration accepted (a convention:
  the widest linear resonances are a few ″/yr wide, so their periods exceed ~1 Myr); $f_{\min}$ =
  `FILTER_MIN_LIBRATION_CUTOFFS` = 2.

The output step does not enter. The filter acts on the unwrapped angle with its trend removed and restored, so it
cannot turn a circulation into a libration. A turning point must stand out by at least $\max(5°,\ 4r)$, where $r$ is
the 95th percentile of what the filtered angle still holds faster than $P_{\rm floor} = \max(P_{\min},\ 2P_c)$, the
same shortest period as for an episode (`SECULAR_TP_PROMINENCE_DEG`, `SECULAR_TP_RIPPLE_FACTOR`; $r = 0$ when
$P_{\rm floor} \le P_c$). The two filters split softly, so a libration slower than $P_{\rm floor}$ leaves
$G(P; P_c) - G(P; P_{\rm floor})$ of its own swing in that difference; its image through the same filters is taken out
first (as for an MMR, [Geometry: turning points](geometry.md); `geometry.libration_image`), or a libration close to the floor raises its own threshold
(591986 over 4 Myr: $r$ = 4.35° and 17.4° against a 16° libration, cut into three episodes where its centre moves
fastest; 1.17° and 5° without the image, one episode).

### Geometry and status

A librating episode needs `LIBRATION_MIN_TURNING_POINTS` = 3 turning points (max–min–max: one full cycle), a period
not shorter than $\max(P_{\min},\ 2P_c)$ (a faster one is a forced wave), and a centre that does not run a full turn
inside it. One that reaches both ends of the record, within `LIBRATION_COVERAGE_PERIODS` = 1 period, makes
`libration`; otherwise the record is `episodic`.

**The centre scale.** A slow secular libration often carries faster waves that break its turning points into
fragments. When the fine scale shows no libration, the angle is filtered again at the scale of its centre,
$W_c$ = `LIBRATION_CENTRE_CYCLES` × $P_{\rm cyc}$ = $3P_{\rm cyc}$; if the centre itself librates there (or shows episodes
the fine scale lacks), that scale decides (`sec_scale` = `centre`). There the shortest libration is
$\max(P_{\min},\ 2W_c)$ and a turning point needs the prominence of a centre turn.

??? info "Every `angle_status` rule"

    | condition | angle_status |
    |---|---|
    | an accepted episode covering the record (both ends within `LIBRATION_COVERAGE_PERIODS` periods) | `libration` (`stationary` or `modulated` centre) |
    | accepted episodes, none covering | `episodic` |
    | no episode; the centre's largest monotone pass ≥ 1 turn | `circulation` |
    | no turning point; the filtered angle turns ≥ 1 time | `circulation` |
    | no turning point, less than a turn | `insufficient_time` (`arc` when it spans at least half a turn, else `stall`) |
    | runs rejected for a moving centre cover ≥ `LIBRATION_DRIFT_MIN_WINDOW_FRACTION` (0.5) of the window | `drift` |
    | otherwise (fewer turning points than one cycle) | `insufficient_time` (`few_cycles`) |

| angle_status | status | reason |
|---|---|---|
| libration | resonant | dynamic_libration |
| episodic | transient | dynamic_episode |
| circulation | non_resonant | circulation |
| drift, insufficient_time | uncertain | baseline_needed (action extend) |
| undefined | invalid / uncertain | unphysical ($e > 1.3$, $a < 0$) / unusable_series |

**The response of $e$ and $i$.** To leading order in $e$ and $i$, a secular term
$K\,e^{\lvert p\rvert}\sin^{\lvert q\rvert}(i/2)\cos\sigma$ of the averaged disturbing function drives
$\dot e \propto p\sin\sigma$ and $\frac{d}{dt}\sin\frac i2 \propto q\sin\sigma$ (Lagrange's equations; Murray & Dermott
1999, eqs. 6.146 and 6.150, pp. 251–252). At finite inclination the equation for $i$ also contains the perihelion
derivative, so even $q = 0$ need not give a zero response of $\sin\frac i2$. The package fits
$\dot x = A\sin\sigma_f + B\cos\sigma_f + C$ for $x = e$ and/or $x = \sin\frac i2$, whichever the angle contains, with
free amplitudes: a phase and shape diagnostic that does not enforce the theoretical coefficient. It is reported as
`check_secular_e_response` (pass at $R^2 \ge$ `SECULAR_RESPONSE_R2_MIN` = 0.5), a column only: it changes no status
and no confidence.

**How long to integrate.** An `uncertain` verdict names the next check in `extend_to_years`: the baseline
$\tfrac32 P + 2P_c$ with $P$ at least the shortest libration (500 kyr for a secular angle at the defaults), capped at
four and at least twice the record; for ZLK the free-ω gate's cycles or frequency resolution, or `review` when no
length would help ([How long to extend](confidence.md#actions)). Before a run, the dry-run warns about a record
shorter than that baseline ([Dry-run](../dry-run.md#secular-and-lidov-kozai-rows)).

Examples: [1222 in $\nu_6$](gallery.md#secular-nu6), [1020 in $z_1$](gallery.md#secular-z1-resonant),
[2306 transient](gallery.md#secular-z1-transient), [847 circulates](gallery.md#secular-z1-circulation),
[363 too short](gallery.md#secular-insufficient-time).

## The Lidov-Kozai angle

**What should librate?** For a highly inclined orbit perturbed by a distant planet, the quadrupole problem makes the
argument of perihelion $\omega$ librate about 90° or 270° ([Lidov's integrals](../lidov-kozai.md#lidovs-integrals)).
The package reports $c_1$, $c_2$ and $h_{\rm quad}$ as diagnostics only; no threshold on them enters the status.

The geometry of $\omega$ is that of a secular angle with $P_c = \max(F_{\rm GI} P_{\rm GI},\ P_{\min}/f_{\min})$ and
no forced waves: $F_{\rm GI}$ = `ZLK_FILTER_GI_FACTOR` = 2, $P_{\min}$ = `ZLK_LIBRATION_PERIOD_MIN_YR` = 2 kyr, the
prominence from `ZLK_TP_PROMINENCE_DEG` and `ZLK_TP_RIPPLE_FACTOR`. The shortest libration is then
$\max(P_{\min},\ 2P_c)$ = 3748 yr, not 2 kyr.

### Why a bounded $\omega$ is not enough {#zlk-free-omega}

The osculating $\omega$ can be bounded without any Lidov-Kozai dynamics. When the forced eccentricity vector is longer
than the free one, the point $(e\cos\varpi, e\sin\varpi)$ circles around the tip of the forced vector without
enclosing the origin, so $\varpi$ follows the forced vector's phase instead of turning on its own. The forced vector
itself turns at the planets' frequencies, so this does not fix $\varpi$; but when its phase and the node move
together, $\omega = \varpi - \Omega$ stays confined by the forcing alone. A true ZLK libration lives in the body's
**free** elements, which is why the gate tests the free perihelion against the free node.

**The free-$\omega$ gate** removes the forced parts of $z_e$ and $z_i$ at the planetary fundamentals and asks whether
$\omega_{\rm free} = \arg z_{e,\rm free} - \arg z_{i,\rm free}$ stays within a turn. Its outcome is `confirmed`
(confined), `kinematic` (the free $\omega$ winds) or no verdict (`short_window`, `undefined_phase`,
`planetary_residual`, `unavailable`); the conditions are in [Free elements: the gate](../free-elements.md#the-gate).
It judges every $\omega$ that stays within a turn (`libration`, `episodic`, `drift`, `insufficient_time`).

**From the outcome to the status** (`decide_status`, check `check_zlk_free_omega`):

| angle_status | confirmed | kinematic | no verdict |
|---|---|---|---|
| libration | resonant / dynamic_libration (check `pass`) | non_resonant / kinematic_libration, origin `kinematic`, detail `free_omega_winds` (check `fail`) | uncertain / gate_unavailable, origin `unknown` |
| episodic, drift, insufficient_time | unchanged | non_resonant / kinematic_libration **only if** $\rho_{\max} < 1$ (episodic: or every episode pinned, below), else unchanged | unchanged |

With `FREE_GATE_ENABLED` off the outcome is `disabled` and the geometry's status stands. On a record that does not
librate throughout, free elements fitted over the whole record cannot judge a change of regime, so only the
strongest case changes it:

$$
\rho_{\max} = \max_t \frac{\operatorname{rollmed}_W \lvert z_{e,\rm free}\rvert}{\operatorname{rollmed}_W \lvert z_{e,\rm forced}\rvert} < 1,
\qquad W = \max\bigl(\tfrac{1}{10}\text{ of the shortest beat of the fitted modes},\ k\,P_{\rm cyc}\bigr),
$$

$k$ = `ZLK_RHO_WINDOW_CYCLES` (1, a convention). Each radius is a rolling median over about one libration cycle, so
the free radius of every part of the record is compared with the forced one of the same part: a free circle that grows
in part of the record (and there encloses the origin) is seen, which one median free radius for the whole record
missed. Below one, $\varpi$ stays tied to the forced phase throughout; the condition is applied only together with a
`kinematic` gate outcome (the free $\omega$ winds). 555527: 0.66 at 100 kyr, 0.63 at 500 kyr (`non_resonant`, as the
author labelled it). The former median ratio is the column `zlk_gate_rho_global`.

An **episodic** $\omega$ is also kinematic when every librating episode is the forced vector's: inside each, the median
of $\lvert z_{e,\rm free}\rvert / \lvert z_{e,\rm forced}\rvert$ (instantaneous: a window as long as a forced beat would
mix in the surroundings) is below 1, so the free circle stays off the origin there, and over the episode's whole
cycles (first turning point to the last of the same kind) the free $\omega$ advances at least
`ZLK_EPISODE_FREE_TURNS_PER_CYCLE` = 0.5 turns per cycle (a convention: a kinematic cycle is one turn of the free
vector about the forced one, since $P_{\rm cyc}$ is their beat; a free $\omega$ that librates with the osculating one
returns after whole cycles, about 0). An episode the free elements do not measure (outside their baseline, a masked
sample in it, fewer than two samples per cycle) pins nothing. Columns `zlk_gate_rho_episodes` (the largest median)
and `zlk_gate_free_turns_episodes` (the fewest turns per cycle). 624 Hektor: its $\omega$ circulates but swings back
for one to three cycles wherever the free radius dips just below the forced one (0.99 at most inside, 2.0 over the
record; 0.94 turns per cycle at least): non_resonant, episodic, kinematic. The real ZLK transients of the benchmark
have 3.0–13.7 and at most 0.04.

`kinematic_libration` is a measurement and `gate_unavailable` the absence of one: keep them apart
([legacy codes](status-tree.md#legacy-integer-codes)). A co-orbital of a planet whose own modes the split does not
fit (Mercury to Mars) gets no verdict (`unavailable`, detail `coorbital_unfitted_forcing`); a Jupiter Trojan is
covered, its forced vector being Jupiter's g5 + g6 ([Free elements](../free-elements.md#where-the-method-stops-working)).

**The scale.** The gate's period is $P_{\rm cyc}$ of the decisive geometry and `zlk_gate_n_cycles` = $T/P_{\rm cyc}$
counts cycles of the osculating $\omega$. A libration at the centre scale whose period is within a Rayleigh width of a
beat $2\pi/|g_k - g_j|$ of two fitted forced modes with nonzero amplitudes, whose reconstructed forced phase wave
explains the centre (correlation $\ge$ `ZLK_CENTRE_MODEL_R_MIN`, swing within a factor `ZLK_CENTRE_MODEL_AMP_FACTOR`),
while $\rho_{\max} < 1$ at the fine period, is the forced vector turning its direction, not a libration of $\omega$:
the fine scale stays decisive (`sec_centre_rejected` = `forced_beat:<gk>-<gj>`). Without the per-mode amplitudes (an
old file) there is no measured beat and no rejection. The guard acts only where the gate does: with the gate disabled,
without a split or for a co-orbital whose forcing the split does not fit, the geometric scale stands.

**Columns that change no status** ([Free elements: beyond the gate](../free-elements.md#beyond-the-gate)): the $e$–$i$
coupling at the decisive period (`zlk_ei_coupling`, every row; `check_zlk_ei` and `confirmed_ei` are `zlk` coupling),
the attribution of a kinematic libration to a forced mode (`zlk_forcing_*`), the node lock (`zlk_node_lock*`) and the
slow line of the centre (`zlk_centre_*`, 591986: the g6−g5 beat). The strongest coherent line of the $e$–$i$ cross-spectrum
([Cross-spectral coherence](../coherence.md)) is the columns `zlk_coherent`, `zlk_check`.

Examples: [3040 librates](gallery.md#zlk-resonant), [591986 kinematic](gallery.md#zlk-kinematic),
[162474 over 1.5 Myr](gallery.md#zlk-transient), [1981 without a verdict](gallery.md#zlk-gate-unavailable),
[15527 circulates](gallery.md#zlk-circulation), [2010 HY22 unphysical](gallery.md#zlk-invalid).

## Confidence and output

The level follows the shared rule ([Confidence](confidence.md)); a visible exit or entry caps it at medium. A secular
angle has no physics margin; for ZLK it is G.

??? info "The margins and their classes"

    S = support, D = decisive ([Confidence](confidence.md#margins)).

    | margin | what | full when |
    |---|---|---|
    | N (S) | turning points of the decisive episode | above the minimum (thin at it, with the cap `three_tp_episode`, ≤ high) |
    | S (S) | its centre | every cycle's band overlaps, with enough independent cycles; not measured with one; not applicable to a modulated centre |
    | F (S) | the filter | the libration period ≥ `FILTER_MIN_LIBRATION_CUTOFFS` × $P_c$; else not measured (flag `filter_too_slow`) |
    | P (D) | a circulation | its largest centre pass ≥ 1 turn at every smoothing (×0.5, ×1, ×2); else thin |
    | G (D) | ZLK: the gate | confirmed with its cycle count clear of `FREE_GATE_MIN_CYCLES` by `CONFIDENCE_THIN_BAND`; a kinematic outcome that decided the status is full |

The summary row carries the analysis' columns (`flt_*`, `geo_*`, `sec_*`, `rsp_*`, `zlk_gate*`, `ep_*`;
[Output data](../output.md)); the runs of turning points go to episodes.csv; the figure is the one-page `combined`
figure ([Plots](../plots.md)).
