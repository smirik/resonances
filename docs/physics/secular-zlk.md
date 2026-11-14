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
    | 3 geometry | turning points, runs, episodes, the center and its passes, exit and entry (shared with MMRs) |
    | 4 angle | `angle_status` from the geometry (below); without a libration at $P_c$, the same at the scale of the center |
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

**What periods to expect, and why the filter is not stronger.** A local resonant Hamiltonian
$H = H_0(I) + A(I)\cos\sigma$ gives the small-amplitude period $P_0 = 2\pi/\sqrt{\lvert KA\rvert}$, with $K$ the
torsion $\partial\delta/\partial I$ of the divisor along the angle's canonical action and $A$ the resonant
coefficient (Morbidelli 2002, pp. 72–74; Morbidelli & Henrard 1991, Eqs. 35–39); the period grows towards the
separatrix. Neither $K$ nor $A$ follows from one body's proper elements and frequencies, and the package measures
neither, so no period is predicted per body. Published cycles span sub-Myr to ~8 Myr ($z_1$: (339) Dorothea 4.15 Myr,
(221) Eos 3.4 Myr, Milani & Knežević 1992, p. 227; (3701) Purkyne ~8 Myr, Vokrouhlický et al. 2006; $z_2$: (8089)
Yukar ~6.1 Myr, Carruba et al. 2016). The $\nu_6$ librators of the benchmark (0.24–0.57 Myr) repeat at the
free/forced beat $2\pi/\lvert g_{\rm free} - g_6\rvert$ within 1 %: a forced oscillator, not a pendulum. No general
lower bound on the period exists, so a filter set above a real libration's period would erase it ($G(P) = 1/[1 +
(P_c/P)^4]$ keeps half at $P = P_c$): the slower scales are tried only where the record shows a slow motion (the center
scale below).

The output step does not enter. The filter acts on the unwrapped angle with its trend removed and restored, so it
cannot turn a circulation into a libration. A turning point must stand out by at least $\max(5°,\ 4r)$, where $r$ is
the 95th percentile of what the filtered angle still holds faster than $P_{\rm floor} = \max(P_{\min},\ 2P_c)$, the
same shortest period as for an episode (`SECULAR_TP_PROMINENCE_DEG`, `SECULAR_TP_RIPPLE_FACTOR`; $r = 0$ when
$P_{\rm floor} \le P_c$). The two filters split softly, so a libration slower than $P_{\rm floor}$ leaves
$G(P; P_c) - G(P; P_{\rm floor})$ of its own swing in that difference; its image through the same filters is taken out
first (as for an MMR, [Geometry: turning points](geometry.md); `geometry.libration_image`), or a libration close to the floor raises its own threshold
(591986 over 4 Myr: $r$ = 4.35° and 17.4° against a 16° libration, cut into three episodes where its center moves
fastest; 1.17° and 5° without the image, one episode).

That $r$ is the record's, and a circulation elsewhere in the record inflates it. So, as for an MMR, each run of
turning points found at the floor (5°) is selected at the prominence measured on its own samples, its ends trimmed by
$P_{\rm floor}$ and the rest at least $P_{\rm floor}$ long, and never above the record's
(`SECULAR_TP_LOCAL_PROMINENCE`, `ZLK_TP_LOCAL_PROMINENCE`). 19289 s−s6 over 20 Myr: 76.8° from the record
(26.8° outside the libration, 8.0° inside), 31.7° on the run: the libration from 5.1 Myr to the end is one episode
instead of seven pieces. The cap: where faster waves hide the libration's period from the turning points at the
floor, its image is not taken out and its own swing would count as fast on the run (2020 PP1, ZLK: 101° against 52°).
A lower threshold may add libration but never remove it: every librating episode found at the record's prominence
must lie inside one librating episode of the local reading, else the runs it overlaps keep the record's prominence.
A run of provisional turning points can span a libration and the slow approach to it, and the approach's wiggles
then become turning points too: 455327 g−s−g5+s6 (ias15, 20 Myr), 39.0° from the record and 13.8° on the run, a
32° wiggle at 9.1–9.5 Myr would start the run on the drifting approach (centre_moves) and lose the crescent episode
10.2–12.5 Myr.

### Geometry and status

A librating episode needs `LIBRATION_MIN_TURNING_POINTS` = 3 turning points (max–min–max: one full cycle), a period
not shorter than $\max(P_{\min},\ 2P_c)$ (a faster one is a forced wave), and a center that does not run a full turn
inside it. One that reaches both ends of the record, within `LIBRATION_COVERAGE_PERIODS` = 1 period, makes
`libration`; otherwise the record is `episodic`.

**A bounded run whose center turns back.** The center tests are those of [Geometry](geometry.md) (the bands of the
cycles share a point, or the center turns back `LIBRATION_CENTRE_RETURNS_MODULATED` = 3 times), but a secular
libration makes only a few cycles in a record, too few for three returns. So a run of a secular or ZLK angle that fails
only for its moving center is accepted as `modulated` (center stationarity not measured) when it closes a cycle, its
turning points span less than a turn, its cycle center turns back at least once within less than $\pi$, and one of its
cycles swings over every center it has (`SECULAR_TURNING_CENTRE`, `ZLK_TURNING_CENTRE`; `geometry.accept_turning_centre`).
A center that moves one way stays a drift (101703 g−2g6+g7, a run at 13.4–19 Myr: −231°, no return), and so does a
small wave carried by a slower motion: no cycle of it covers the track (363 g−g6+s−s6: half-amplitudes 4–6° on a center
that travels 42°). 42946 g−g6+s−s6 over 20 Myr: the run 1.3–14.6 Myr (17 turning points, $P$ = 1.6 Myr, centers
116–228° turning back twice, bands missing by 18°, the cycle about 183° of half-amplitude 93° covering them all) is
the early libration episode.

**A ZLK island is half a turn wide.** The averaged Lidov-Kozai Hamiltonian depends on $2\omega$, so its islands
($\omega$ about 90° and 270°) are at most half a turn wide. A run of $\omega$ therefore breaks where a half-cycle swings
`ZLK_RUN_BREAK_HALF_AMPLITUDE_DEG` = 90° or more, or its midpoint jumps by more (a secular or MMR angle: 180°). This
only splits runs; which island an episode is in decides nothing. 88899 over 500 kyr: two islands (67–169 kyr about
90°, 191–294 kyr about 270°) that one run at 180° joined into a circulation (rule `zlk_half_turn_cut`).

**Drift needs a monotone center.** A drift is a slow center that keeps going one way. When the runs are rejected for
their moving center but the slow center turns back inside the record (away from the edges of its own low-pass) and its
largest pass stays below a turn, the record holds part of one slow swing: `insufficient_time` / `few_cycles`, not
`drift` (2306 $g-g_6+s-s_6$: a turn at 2.04 Myr, largest pass 0.92 rev in 10 Myr; rule `centre_turned_insufficient`).
Status and extension are the same as for a drift.

**A swing must exceed the uncertainty of the mode phases.** Under `SECULAR_PLANET_PHASES=linear` the angle uses
$\ell_j = g_j t + \phi_j$ with constant phases, while the true phase $\phi_j(t)$ of an inner mode wanders (Laskar
1990). Over a record of length $T$ the angle then carries an error whose swing is at most

$$
\Delta\phi(T) = \sum_j \lvert k_j\rvert\, \mathrm{range}_j(T),
$$

$k_j$ the angle's coefficient of mode $j$, $\mathrm{range}_j(T)$ the largest max − min of $\phi_j$ over a window of
length $T$, measured on the package's own model (`resonances/data/mode_phase_uncertainty.json`, at 10 Myr: $g_4$ 30°,
$g_3$ 24°, $s_4$ 81°, $s_2$ 87°, $g_2$ 2.7°, $g_6$ 0.9°, $s_6$ 9.5°; [how it is measured](../secular.md#mode-phases)).
A libration or episode whose swing $2A_{\max}$ (the deciding episode's largest half-amplitude) is not larger than
`SECULAR_PHASE_UNCERTAINTY_FACTOR` × $\Delta\phi$ (1, a convention) is not resolved from that error: its turning
points may be the phases'. It becomes `insufficient_time` / `phase_unresolved`, action `review`: rerun with
`SECULAR_PLANET_PHASES=measured` (`SAVE_PLANETS=all`), where the measured modes add nothing (rule `phase_unresolved`;
columns `sec_phase_uncertainty_deg`, `sec_phase_swing_deg`, `sec_phase_swing_ratio`). 5026 $g+g_2-g_4-g_6$ over
10 Myr: $2A_{\max}$ = 22° against $\Delta\phi$ = 34°. The outer modes stray too little to matter: the narrowest
$z_1$/$z_2$ libration of the benchmark swings 4.6 times its $\Delta\phi$ (52773), $\nu_6$ 78 times.

**The center scale.** A slow secular libration often carries faster waves that break its turning points into
fragments. When the fine scale shows no libration, the angle is filtered again at the scale of its center,
$W_c$ = `LIBRATION_CENTRE_CYCLES` × $P_{\rm cyc}$ = $3P_{\rm cyc}$; if the center itself librates there (or shows episodes
the fine scale lacks), that scale decides (`sec_scale` = `centre`). There the shortest libration is
$\max(P_{\min},\ 2W_c)$ and a turning point needs the prominence of a center turn. When $W_c$ shows no more libration
than the fine scale, a secular angle tries the scale of its center's own turns (the fine turning points at the
prominence of a center turn, $W$ = their median period / `FILTER_MIN_LIBRATION_CUTOFFS`; the MMR second scale,
`SECULAR_CENTRE_TURNS_SCALE`): $P_{\rm cyc}$ of fast waves sets $W_c$, and its edge trim of $W_c$ per end, far above
what the slow turns need. 101703 g−2g6+g7 over 20 Myr: $W_c$ = 3.1 Myr from 1-Myr waves trims 3–17 Myr and loses the
turn at 18.2 Myr; the turns give 1.6 Myr and one libration of 12 Myr from 6.3 to 18.25 Myr (`resonant`, extend).

**A wave on a slower motion is no episode.** An accepted episode can be a ripple that the slow motion of the angle
carries (`SECULAR_SLOW_WAVES`; `secular.decide.slow_waves`). The slow motion around it is $\sigma_f$ between the turns of
the center track that bracket it (or the record's edges), of range $R$:

- $R$ below a turn (a slow swing): the episode is a ripple on it when $A_{\max} \le$
  `SECULAR_SLOW_SWING_RIPPLE_RATIO` × $R/2$ (0.5, the MMR ratio, a convention) and the swing carries it: the slow
  track $S$ (below) moves inside the episode at least `SECULAR_SLOW_SWING_CARRY_RATIO` (1/16, a convention) of its
  fastest rate beside it. A wave at a smooth turn of the slow motion reads at least 1/8 (a parabola centred on the
  episode: $(D/2)^2$ inside against $(3D/2)^2-(D/2)^2$ on an arm as long again); a libration on a plateau of the slow
  motion, which moves only beside it, reads the filter's leak and stays an episode (two stationary librations joined
  by a monotone transition: ≤ 0.04). 101703 g−2g6+g7: 8–19° waves of 1 Myr at the turns of the 12-Myr libration
  ($R$ = 315°, carried 0.16 and 0.12); 56700 s+s7−2s6: a 13° ripple on a 322° arc of 20 Myr (0.51), which becomes
  `insufficient_time`, extend (possibly a slow libration longer than the 20-Myr record);
- $R$ a turn or more (a circulation): a capture stops the circulation, a stall only slows it. The slow track $S$
  ($\sigma$ low-passed at `LIBRATION_CENTRE_CYCLES` × the episode's period) must carry the angle out of the episode's
  band beside it ($\ge A_{\max}$ over as long again before or after; an arm the record cuts below
  `SECULAR_STALL_ARM_MIN_FRACTION` of the episode's span is not read) and keep inside the episode at least
  `SECULAR_STALL_RATE_RATIO` (0.5, a convention) of that rate. 348018 g−g5+s−s7: a 4° ripple on 4.4 turns, 0.73:
  `non_resonant`; the captures of 19289 and 278067 s−s6 read 0.04–0.05, 455327 s−s3 0.24.

A record-covering episode is never a wave (no motion beside it). The rule runs at every scale (a ripple filtered at
the center scale is still one), and the exit and entry are read again on the episodes left.

**Exit and entry.** As for an MMR ([Geometry](geometry.md#coverage-exits-and-entries)), an arc of more than a turn is
no piece of libration: when $\sigma_f$ runs at least `LIBRATION_EDGE_ARC_REV` = 1 turn monotonically after the last
turning point (before the first), the exit (entry) is observed, that end is not covered however short the gap, the
angle is `episodic` and `final_regime` is a circulation from the exit. A secular angle has no orbit to average, so
$\sigma_f$ itself is read, against the episode's mean center; the center scale is then not tried. 2012 AX10 and 33325
in 2s − s4 − s6 over 20 Myr leave their libration near 16 Myr and run 1.9 and 1.7 turns to the end: `transient`
(the end gap of 0.8 P does not count as covered). An exit that runs less than a turn stays
`exit_suspected`.

??? info "Every `angle_status` rule"

    | condition | angle_status |
    |---|---|
    | an accepted episode covering the record (both ends within `LIBRATION_COVERAGE_PERIODS` periods, no observed exit or entry) | `libration` (`stationary` or `modulated` center) |
    | accepted episodes, none covering | `episodic` |
    | no episode; the center's largest monotone pass ≥ 1 turn | `circulation` |
    | no turning point; the filtered angle turns ≥ 1 time | `circulation` |
    | no turning point, less than a turn | `insufficient_time` (`arc` when it spans at least half a turn, else `stall`) |
    | runs rejected for a moving center cover ≥ `LIBRATION_DRIFT_MIN_WINDOW_FRACTION` (0.5) of the window, the slow center monotone | `drift` |
    | the same, the slow center turned back (largest pass < 1 turn) | `insufficient_time` (`few_cycles`) |
    | otherwise (fewer turning points than one cycle) | `insufficient_time` (`few_cycles`) |
    | a libration or episode under linear phases swinging no more than `SECULAR_PHASE_UNCERTAINTY_FACTOR` × $\Delta\phi(T)$ | `insufficient_time` (`phase_unresolved`) |

| angle_status | status | reason |
|---|---|---|
| libration | resonant | dynamic_libration |
| episodic | transient | dynamic_episode |
| circulation | non_resonant | circulation |
| drift, insufficient_time | uncertain | baseline_needed (action extend; `phase_unresolved`: review) |
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
[2306 one unfinished slow swing](gallery.md#secular-z1-unfinished-swing), [847 circulates](gallery.md#secular-z1-circulation),
[363 too short](gallery.md#secular-insufficient-time).

## The Lidov-Kozai angle

**What should librate?** For a highly inclined orbit perturbed by a distant planet, the quadrupole problem makes the
argument of perihelion $\omega$ librate about 90° or 270° ([Lidov's integrals](../lidov-kozai.md#lidovs-integrals)).
The package reports $c_1$, $c_2$ and $h_{\rm quad}$ as diagnostics only; no threshold on them enters the status.

**From which plane?** The quadrupole problem's reference plane is the perturbers' orbital plane; the package measures
$\omega$, $i$ and $\Omega$ from the invariable plane (`ZLK_ANGLE_FRAME`, default `invariable`; `ecliptic` is kept for
comparison), by rotating the orbit's angular momentum $\mathbf h$ and perihelion vector $\mathbf P$ exactly, never
by subtracting angles. The free elements of a ZLK body, the gate, the attribution, the node lock, the e–i coupling and
the figure use the same frame. The ecliptic is tilted by $I = 1.58°$ from that plane: the ecliptic node of an orbit
with $i_{\rm inv} < I$ cannot turn (the orbit's pole circles the invariable pole without enclosing the ecliptic pole),
so it librates about the plane's node, and with a bounded $\varpi$ the ecliptic $\omega$ looks confined while the
invariable one circulates (555527, $i_{\rm inv} \approx 1.6°$: `non_resonant` / `circulation` / `certain` on both
records). At $i \gtrsim 20°$ the two frames give the same number of turns. Rule `zlk_invariable_frame` marks a row whose
two $\omega$ part by a whole turn or more (`zlk_frame_shift_turns`).

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
| episodic | unchanged (also when confirmed inside its episodes, below) | non_resonant / kinematic_libration **only if** $\rho_{\max} < 1$ or every episode pinned (below), else read inside the episodes | transient only if $e$ and $i$ trade at constant $H$ inside the deciding episode, else uncertain / gate_unavailable (below) |
| drift, insufficient_time | unchanged | non_resonant / kinematic_libration **only if** $\rho_{\max} < 1$, else unchanged | unchanged |

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
author labeled it). The former median ratio is the column `zlk_gate_rho_global`.

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

**The gate inside the episodes** (`ZLK_GATE_EPISODES`). Between its episodes an episodic $\omega$ circulates, so
its free $\omega$ winds over the record whatever it does inside them, and the record's outcome (`kinematic`, or a
`planetary_residual` on the winding branch: over 500 kyr some $g_j - s_k$ always lies within a width of $b_\omega$)
says nothing about the episodes. Such a row is read inside its librating episodes with the confirmation test: in every
one the free vector is the longer one (median $\rho \ge 1$, column `zlk_gate_rho_episodes_min`) and the free $\omega$
advances fewer than `ZLK_EPISODE_FREE_TURNS_PER_CYCLE` turns per cycle (`zlk_gate_free_turns_episodes_max`), and
$b_\varpi$, $b_\Omega$ stay off the $g$ and $s$ modes → `confirmed`, `zlk_gate_detail` = `episodes`; a residual at a
mode → `planetary_residual`; an episode the free elements do not measure keeps the record's outcome. 88899 over
500 kyr: 3.0 turns over the record, 0.005 turns per cycle and $\rho \ge 8.2$ inside both episodes → confirmed; its
$e$–$i$ integral is not held there, an objection (transient/low, flag `ei_integral_not_held`). 1866, 373136, 3752 and
797204 (500 kyr) are confirmed the same way (origin dynamic).

`kinematic_libration` is a measurement and `gate_unavailable` the absence of one: keep them apart
([legacy codes](status-tree.md#legacy-integer-codes)). A co-orbital of a planet whose own modes the split does not
fit (Mercury to Mars) gets no verdict (`unavailable`, detail `coorbital_unfitted_forcing`): `uncertain` /
`gate_unavailable` whatever bounded $\omega$ it has, an episodic one, a drift or too few cycles included, since the
model does not apply (2020 PP1 and 10563 over 500 kyr; rule `zlk_coorbital_episodic`). A Jupiter Trojan is
covered, its forced vector being Jupiter's g5 + g6 ([Free elements](../free-elements.md#where-the-method-stops-working)).

**An episodic $\omega$ without a gate verdict** (any other no-verdict outcome) needs the physics of a ZLK cycle inside
its deciding episode: $e$ and $i$ must trade so that the quadrupole integral $H = \sqrt{1-e^2}\cos i$ stays constant.
At the episode's own period $P$, on its samples only (the whole record also holds stretches where $\omega$
circulates, which trade $e$ for $i$ near $P$ as well), $e$, $i$, $G = \sqrt{1-e^2}$, $H$ and $K = G(1-\cos i)$ are
high-passed by a one-period running mean over the whole record and demodulated at $P$ cycle by cycle; then

$$
\eta_H = \frac{\operatorname{rms}\,\delta H}{\lvert\cos i\rvert\,\operatorname{rms}\,\delta G}
$$

is 0 when $H$ is held and 1 as if $i$ were frozen. The coupling `zlk_ei_episode_coupling` is `zlk` when $i$ is in
antiphase with $e$ (within `ZLK_EI_PHASE_TOL_DEG`), the demodulated coherence is $\le -$`ZLK_EI_EPISODE_MIN_COHERENCE`
and $\eta_H \le$ `ZLK_EI_H_RESIDUAL_MAX` (0.5, a convention: the benchmark's ZLK episodes measure at most 0.28, the
others at least 0.58); `forced_beat` the same in phase with $K$; `none`; `not_measured` when the episode spans fewer than
`ZLK_EI_MIN_CYCLES` = 3 periods (two independent estimators agree only by chance below that). In the full tree
(`v2_verdict`) the row stays `transient` only on `zlk`, otherwise it is `uncertain` / `gate_unavailable` (−9), rule
`zlk_gate_unconfirmed`; the minimal policy reads the same coupling as the e–i integral (below).
373136, 3752 and 797204 over 500 kyr stay transient ($\eta_H$ 0.05–0.23 over 4.6–8 periods); 88899 over 500 kyr
($e$ and $i$ out of antiphase, $\eta_H$ 1.2) and 797204 over 100 kyr and 2023 RS2 (deciding episodes of 1.5 and 2.4
periods) become uncertain. The record's coupling (`zlk_ei_coupling`) stays a column.

**Under the minimal policy** (the table above is the full tree's, kept in `v2_verdict`): the quadrupole problem, its integral thresholds ($h^2 < 3/5$, $c_1$,
Kozai's 39.2°) and its half-turn symmetry hold only for a hierarchical, axisymmetric perturber, outside their domain
for main-belt and planet-crossing asteroids (Gronchi & Milani 1999; Gallardo et al. 2012); none of them decides. The
geometry of $\omega$ gives the status (one revolution is $2\pi$), and the physics reads two measurements on the
deciding episode:

- the **$e$–$i$ integral** above, measured on every bounded $\omega$ (a libration too): broken (measured, not `zlk`)
  where the forced terms are small, $\rho_{\max} \ge$ `ZLK_RHO_KINEMATIC_EXCLUDED` (3, a convention: a non-resonant
  forced $e$ is at most ~0.1), is a veto, non_resonant / kinematic_libration, detail `ei_integral_not_held` (88899 over
  500 kyr, $\eta_H$ 1.24, $\rho_{\max}$ 13); with larger forced terms it is an objection (flag
  `ei_integral_not_held`, level low), since the forcing itself breaks $\sqrt{1-e^2}\cos i$;
- the **forced vector**: a gate `kinematic` on a libration is a veto (`zlk_kinematic`), and so is $\rho_{\max} < 1$
  over the record together with a `kinematic` gate on an episodic angle, a drift or too few cycles (`zlk_free_e`); the
  per-episode pin (a median) and an undefined free phase veto only together with a broken integral (624 Hektor:
  $\eta_H$ 1.04), otherwise they are objections. $\rho_{\max}$ is a ratio of two parts of one split, not a fact
  independent of it: 591986 refitted with g7 for the unresolved g5 gives 2.9 instead of 0.43, a planetary residual
  0.83″/yr from g5 and resonant/low with `gate_unavailable`; the default split (g5, the one the 4-Myr record resolves)
  is kinematic.

A gate without a verdict keeps the geometry's status with the flag `gate_unavailable` at most medium, except when the
integral holds and $\rho_{\max} \ge$ `ZLK_RHO_KINEMATIC_EXCLUDED`: the forced vector cannot hold the perihelion, so
the abstention (1981 Midas: its free node near $s_6$) is no cap and asks no extension (flag `kinematic_excluded`). Only
a `confirmed` gate names a dynamic origin; otherwise the reason is `geometric_libration`/`geometric_episode`.

**The scale.** The gate's period is that of the deciding librating episode (the covering one, or the one with the most
turning points; `sec_decisive_period_yr`), and `zlk_gate_n_cycles` = $T/P$ counts cycles of the osculating $\omega$.
F, the $e$–$i$ coupling, the $\rho$ window and the extension read the same $P$. Without a librating episode it is the
median full period of the runs not faster than the floor. The median over every episode ($P_{\rm cyc}$,
`geo_period_cycle_yr`) is a column only: 2020 PP1's five episodes run from 2.3 to 34 kyr. The attribution of an $\omega$
pinned inside every episode tests the beat against the episodes' pooled period, since each episode is a cycle of the same
beat (624 Hektor). A libration at the center scale whose period is within a Rayleigh width of a
beat $2\pi/|g_k - g_j|$ of two fitted forced modes with nonzero amplitudes, whose reconstructed forced phase wave
explains the center (correlation $\ge$ `ZLK_CENTRE_MODEL_R_MIN`, swing within a factor `ZLK_CENTRE_MODEL_AMP_FACTOR`),
while $\rho_{\max} < 1$ at the fine period, is the forced vector turning its direction, not a libration of $\omega$:
the fine scale stays decisive (`sec_centre_rejected` = `forced_beat:<gk>-<gj>`). Without the per-mode amplitudes (an
old file) there is no measured beat and no rejection. The guard acts only where the gate does: with the gate disabled,
without a split or for a co-orbital whose forcing the split does not fit, the geometric scale stands.

**Columns that change no status** ([Free elements: beyond the gate](../free-elements.md#beyond-the-gate)): the $e$–$i$
coupling of the whole record at the decisive period (`zlk_ei_coupling`, every row; `check_zlk_ei` and `confirmed_ei`
are `zlk` coupling; the deciding episode's own coupling `zlk_ei_episode_*` is a column too, except for an episodic row
without a gate verdict),
the attribution of a kinematic libration to a forced mode (`zlk_forcing_*`), the node lock (`zlk_node_lock*`) and the
slow line of the center (`zlk_centre_*`, 591986: the g6−g5 beat). The strongest coherent line of the $e$–$i$ cross-spectrum
([Cross-spectral coherence](../coherence.md)) is the columns `zlk_coherent`, `zlk_check`.

Examples: [3040 librates](gallery.md#zlk-resonant), [591986 kinematic](gallery.md#zlk-kinematic),
[162474 over 1.5 Myr](gallery.md#zlk-transient), [1981 without a verdict](gallery.md#zlk-gate-unavailable),
[15527 circulates](gallery.md#zlk-circulation), [2010 HY22 unphysical](gallery.md#zlk-invalid).

## Confidence and output

The level follows the shared rule ([Confidence](confidence.md)); a suspected exit or entry caps it at medium. A secular
angle has no physics margin; for ZLK it is G.

??? info "The margins and their classes"

    S = support, D = decisive ([Confidence](confidence.md#margins)).

    | margin | what | full when |
    |---|---|---|
    | N (S) | turning points of the decisive episode | above the minimum (thin at it, with the cap `three_tp_episode`, ≤ high) |
    | S (S) | its center | every cycle's band overlaps, with enough independent cycles; not measured with one; not applicable to a modulated center |
    | F (S) | the filter | the libration period ≥ `FILTER_MIN_LIBRATION_CUTOFFS` × $P_c$; else not measured (flag `filter_too_slow`) |
    | P (D) | a circulation | its largest center pass ≥ 1 + `CONFIDENCE_THIN_BAND` = 1.1 turns at every smoothing (×0.5, ×1, ×2); else thin |
    | G (D) | ZLK: the gate | confirmed with its cycle count clear of `FREE_GATE_MIN_CYCLES` by `CONFIDENCE_THIN_BAND`; a kinematic outcome that decided the status is full |

The summary row carries the analysis' columns (`flt_*`, `geo_*`, `sec_*`, `rsp_*`, `zlk_gate*`, `ep_*`;
[Output data](../output.md)); the runs of turning points go to episodes.csv; the figure is the one-page `combined`
figure ([Plots](../plots.md)).
