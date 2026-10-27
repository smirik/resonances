<!-- Generated from docs/physics/secular-zlk.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Secular and Lidov-Kozai angles

Secular and Lidov-Kozai (ZLK) angles evolve over $10^4$–$10^7$ yr instead of $10^2$–$10^4$ yr, and they involve only
the slow angles $\varpi$, $\Omega$, $\omega$. They go through the same [geometry](geometry.md) as mean-motion
resonances (turning points, cycles, episodes, coverage), with their own signal step and their own physics. The
status tree is short ([Status tree](status-tree.md#secular-and-lidov-kozai-angles)): there is no semi-major axis
to respond, no owner and no subresonance.

## The secular angle

**What should librate?** A linear secular resonance locks a precession frequency of the body to one of the
planetary system's eigenfrequencies: $g \approx g_6$ ($\nu_6$), $s \approx s_6$ ($\nu_{16}$), or a combination such
as $g + s - g_6 - s_6$ ($z_1$). In the Laplace-Lagrange theory the planets' eccentricity and inclination vectors are
sums of eigenmodes, each turning at its own rate $g_j$ or $s_j$ (Murray & Dermott 1999, eqs. 7.28–7.29, p. 278;
the vector variables themselves are defined by their eqs. 7.18–7.19, p. 277). The resonant angle compares the body's
$\varpi$ and $\Omega$ with the **mode longitudes**

$$
\ell_{g_j}(t) = g_j\,(t + t_{\rm epoch}) + \phi_{g_j}, \qquad \ell_{s_j}(t) = s_j\,(t + t_{\rm epoch}) + \phi_{s_j},
$$

with the phases quoted at J2000 ($t_{\rm epoch}$ the offset of the run's start from J2000).

**Units.** In the formulas of this page $g$, $s$ and every frequency $f$ are angular frequencies in rad/yr. The
stored `PLANETARY_FREQUENCIES` are in ″/yr and are multiplied by $\pi/648\,000$; one cycle per year is
1 296 000 ″/yr. A record of length $T$ resolves frequencies $2\pi/T$ rad/yr apart (one Rayleigh width, $1/T$ in
cycles/yr).

not with the planets' osculating elements. Seen from the ecliptic, no planet's node circulates (the 1.58° tilt of
the invariable plane dominates), and only Mercury's, Jupiter's and Saturn's perihelia follow their own mode. An
angle built from osculating planetary elements would therefore be wrong for most terms.

**The body's vectors.** The body enters through its complex eccentricity and inclination vectors, built from the
saved (ecliptic) elements:

$$
z_e = e\,e^{i\varpi}, \qquad z_i = \sin\tfrac{i}{2}\,e^{i\Omega}.
$$

Murray & Dermott use the inclination $I$ in radians; since $\sin\frac i2 \simeq I/2$ at small inclination, phases and
frequencies agree, while inclination-vector amplitudes and fitted coefficients here are **half** theirs. Each vector is
a forced part, made by the planetary modes, plus the body's free part. The forced terms at the fundamentals that are
**not** in the angle only shake it, so they are removed by a least-squares fit at their known frequencies $f_j$:

$$
z_{\rm red} = z - \sum_{j \in B} C_j\, e^{i f_j t}.
$$

The basis $B$ is that of the [free elements](../free-elements.md): $g_5$–$g_8$ plus a constant for $z_e$, and the
constant ($s_5 = 0$, the invariable plane) plus $s_6$–$s_8$ for $z_i$; inner-planet modes are not subtracted. Modes
closer than two Rayleigh widths, $4\pi/T$ rad/yr, cannot be separated over a record of length $T$, so the candidates
are first clustered and one representative of each cluster is kept; then the representatives whose labels occur in
the angle are dropped (their terms are the resonance), and the rest is fitted and removed. Removing the constant of
$z_i$ refers the node to the invariable plane; this is a vector reduction, not an exact rotation of the orbit into
that plane. With $\varpi = \arg z_{e,\rm red}$ and $\Omega = \arg z_{i,\rm red}$ the angle is

$$
\sigma = p\,\varpi + q\,\Omega + \sum_j \big(p_j\,\ell_{g_j} + q_j\,\ell_{s_j}\big),
\qquad p + q + \sum_j (p_j + q_j) = 0 .
$$

The frequencies and phases are those of the package's own planetary model (`PLANETARY_FREQUENCIES`,
`PLANETARY_PHASES` in `resonances/data/const.py`, `SECULAR_PLANET_PHASES` = linear). The linear law drifts by tens
of degrees over 20 Myr for the inner modes and $s_6$; with `SECULAR_PLANET_PHASES` = measured the mode longitudes
are fitted in the integration itself, in sliding windows of `SECULAR_PHASE_WINDOW` = 5 Myr (needs the planets
saved). Details: [Secular resonances](../secular.md).

### Signal: what is not the libration

The filter must remove everything faster than the slowest possible libration and nothing slower:

$$
P_c = \max\Big(F_{\rm GI}\,P_{\rm GI},\ \ F_w \max_j \frac{2\pi}{\lvert f - f_j\rvert},\ \ \frac{P_{\min}}{2}\Big).
$$

- $P_{\rm GI}$ = 937 yr, the Jupiter-Saturn great inequality, the longest short-period term of an asteroid's
  elements; $F_{\rm GI}$ = `SECULAR_FILTER_GI_FACTOR` = 2.
- For an angle with two or more planetary modes, each mode $j$ makes a **forced wave**: its forced term turns
  relative to the free one at $f - f_j$ ($f$ the body's own $g$ or $s$). These waves are not librations; $F_w$ =
  `SECULAR_FILTER_FORCED_FACTOR` = 2. A single-mode angle ($\nu_6$) has none: its forced term is the resonance.
- $P_{\min}$ = `SECULAR_LIBRATION_PERIOD_MIN_YR` = 200 kyr, the shortest secular libration accepted (a convention:
  the widest linear resonances are a few ″/yr wide, so their periods exceed ~1 Myr). The divisor 2 is
  `FILTER_MIN_LIBRATION_CUTOFFS`.

The filter acts on the unwrapped angle with its trend removed and restored, so it cannot turn a circulation into a
libration. A turning point must stand out by at least $\max(5°,\ 4r)$, where $r$ is the 95th percentile of what the
filtered angle still holds faster than $P_{\rm floor} = \max(P_{\min},\ 2P_c)$, the same shortest period as for an
episode (`SECULAR_TP_PROMINENCE_DEG`, `SECULAR_TP_RIPPLE_FACTOR`).

### Geometry and status

A librating episode needs `LIBRATION_MIN_TURNING_POINTS` = 3 turning points (max–min–max: one full cycle), a period
not shorter than $\max(P_{\min},\ 2P_c)$, and a centre that does not run a full turn inside it. One that reaches both
ends of the record, within `LIBRATION_COVERAGE_PERIODS` = 1 period, makes `libration`; otherwise the record is
`episodic`.

**The centre scale.** A slow secular libration often carries faster waves that break its turning points into
fragments. When the fine scale shows no libration, the angle is filtered again at the scale of its centre,
$W_c$ = `LIBRATION_CENTRE_CYCLES` × $P_{\rm cyc}$ = $3P_{\rm cyc}$; if the centre itself librates there (or shows episodes
the fine scale lacks), that scale decides.

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

**How long to integrate.** A libration of period $P$ needs at least

$$
T \ge \frac{N_{\rm TP}}{2}\,P + 2\,f_{\rm trim}\,P_c ,
$$

with $N_{\rm TP}$ = `LIBRATION_MIN_TURNING_POINTS` = 3 and $f_{\rm trim}$ = `FILTER_EDGE_TRIM_CUTOFFS` = 1: one and a half
periods for the turning points of one cycle, plus the stretches the filter trims at both ends. A degree-4 resonance
(half-width ~0.3″/yr) librates in ~9 Myr, a degree-6 one in ~26 Myr. The package does not compute this length for
secular angles (an `uncertain` verdict asks to extend without a number); use the rule to plan the integration.

Examples: [1222 in $\nu_6$](gallery.md#secular-nu6), [1020 in $z_1$](gallery.md#secular-z1-resonant),
[2306 transient](gallery.md#secular-z1-transient), [847 circulates](gallery.md#secular-z1-circulation),
[363 too short](gallery.md#secular-insufficient-time).

## The Lidov-Kozai angle

**What should librate?** For a highly inclined orbit perturbed by a distant planet, the double-averaged quadrupole
problem has two integrals (Lidov 1962, eqs. 58–60, p. 741; Kozai 1962, eqs. 26–28, p. 594):

$$
c_1 = (1 - e^2)\cos^2 i_{\rm rel}, \qquad c_2 = e^2\Big(\frac25 - \sin^2 i_{\rm rel}\,\sin^2\omega_{\rm rel}\Big),
$$

with $i_{\rm rel}$ the inclination to the perturber's orbital plane and $\omega_{\rm rel}$ measured in that plane; the
argument of perihelion librates about 90° or 270° when $c_1 < 3/5$ and $c_2 < 0$. The package reports $c_1$, $c_2$ and
$h_{\rm quad} = 6c_1 - 2 + 30c_2$ as **diagnostics only**, evaluated on the body's initial saved (ecliptic) $e$, $i$,
$\omega$, not in the perturber's plane: the classical invariants additionally assume a single distant perturber, exact
for $a/a_{\rm pert} \to 0$, while a main-belt asteroid has $a/a_J \approx 0.5$ (1373 Cincinnati librates at
$c_2 > 0$). No threshold on them enters the status. See [Lidov-Kozai resonances](../lidov-kozai.md).

The geometry of $\omega$ is that of a secular angle with $P_c = \max(F_{\rm GI} P_{\rm GI},\ P_{\min}/2)$, $P_{\min}$ =
`ZLK_LIBRATION_PERIOD_MIN_YR` = 2 kyr.

### Why a bounded $\omega$ is not enough

The osculating $\omega$ can be bounded without any Lidov-Kozai dynamics. When the forced eccentricity vector is longer
than the free one, the point $(e\cos\varpi, e\sin\varpi)$ circles around the forced vector without enclosing the
origin; $\varpi$ cannot turn, and $\omega = \varpi - \Omega$ looks confined by geometry alone. A true ZLK libration
lives in the body's **free** elements.

**The free-$\omega$ gate.** The forced parts at the planetary fundamentals ($g_5$–$g_8$ for $z_e$; the invariable
plane $s_5 = 0$ and $s_6$–$s_8$ for $z_i$) are fitted and removed ([Free elements](../free-elements.md)), and

$$
\omega_{\rm free} = \arg z_{e,\rm free} - \arg z_{i,\rm free}.
$$

Samples where $\lvert z_{\rm free}\rvert$ falls below `FREE_GATE_MASK_QUANTILE` = 0.15 of its median are dropped (the
argument is undefined near the origin). The free $\omega$ is **confined** when its range, trimmed by half a percent
at each end, is less than one turn:

$$
R_\omega = Q_{0.995}(\omega_{\rm free}) - Q_{0.005}(\omega_{\rm free}) < 2\pi .
$$

The threshold is topological: inside a mean-motion resonance the ZLK centre moves away from 90°/270°, so no
narrower window is safe. The outcome:

| outcome | condition | meaning |
|---|---|---|
| `confirmed` | confined, and no planetary match (below) | a ZLK libration of the body's own elements |
| `kinematic` | $\omega_{\rm free}$ winds, and no planetary match | the osculating libration is geometry: the forced vector holds $\varpi$ |
| `short_window` | the record holds fewer than `FREE_GATE_MIN_CYCLES` = 3 libration periods | no verdict |
| `undefined_phase` | more than half of the samples masked | no verdict |
| `planetary_residual` | a planetary match | no verdict: a leak of the forcing, or a linear secular resonance where the split is undefined |
| `unavailable` | the free elements could not be built | no verdict |

**Planetary match.** The signed free rates $b_\varpi$, $b_\Omega$, $b_\omega$ (slopes of the unwrapped free angles)
are compared with targets; a match is a difference smaller than one Rayleigh width ($2\pi/T$ rad/yr, 1 296 000/$T$
″/yr). On both branches $b_\varpi$ is compared with $g_5$–$g_8$ and $b_\Omega$ with $s_6$–$s_8$. On the winding branch
$b_\omega$ is also compared with every difference $g_j - s_k$ ($j = 5\ldots8$, $k = 5\ldots8$, $s_5 = 0$), since
$\omega = \varpi - \Omega$ carries an unsubtracted term of either vector at that difference. On the confined branch
$b_\varpi$ is also compared with zero when the constant of $z_e$ was clustered out of the fit (an unsubtracted constant
is exactly what makes an angle look confined).

The libration period is $P_{\rm cyc}$, the median full period of the accepted episodes; the number of cycles is
$T/P_{\rm cyc}$. There is no amplitude criterion: the frequency is the observable, because unsubtracted forcing lives
only at known rates.

**From the outcome to the status** (`check_zlk_free_omega`):

| angle_status | confirmed | kinematic | no verdict |
|---|---|---|---|
| libration | resonant | non_resonant / kinematic_libration (detail `free_omega_winds`) | uncertain / gate_unavailable |
| episodic, drift, insufficient_time | unchanged | non_resonant / kinematic_libration **only if** $\rho_{\max} < 1$, else unchanged | unchanged |

Here $e_{\rm free} = \operatorname{median}_t\lvert z_{e,\rm free}(t)\rvert$ is one radius for the whole record, and
$F(t)$ is $\lvert z_{e,\rm forced}(t)\rvert$ smoothed by a rolling median over a tenth of the shortest beat period
between the fitted modes (no smoothing for a single line); $\rho_{\max} = e_{\rm free}/\min_t F(t)$. This is the
envelope of a circle of the median free radius, not the largest instantaneous ratio. $\rho_{\max} < 1$ means that this
circle stays inside the forced offset over the whole record: $\varpi$ cannot turn and $\omega$ is pinned by the
forced vector throughout. Without it, free elements fitted over the whole record
cannot judge a record that changes regime, and the geometry's status stands.

!!! warning "Kinematic is not the same as no verdict"
    `non_resonant / kinematic_libration` (legacy code −2) is a measurement: the libration was found to be geometry.
    `uncertain / gate_unavailable` (−9) is the absence of a measurement; it asks for a longer record. The two must
    never be merged in a statistic.

**Co-orbitals.** The method does not apply to a body in 1:1 resonance with a planet: its forced term is the
perturber's own vector, not a planetary fundamental, and a confident-looking result there is not a measurement.

**The e–i exchange.** The cross-spectral coherence of $e$ and $i$ ([Cross-spectral coherence](../coherence.md))
gives the column `check_zlk_ei`; it changes no status.

**Confidence.** The margin G is full when the gate confirmed with its number of cycles clear of the minimum by more
than the thin band (10 %), or when it measured a kinematic libration ([Confidence](confidence.md)).

Examples: [3040 librates](gallery.md#zlk-resonant), [591986 kinematic](gallery.md#zlk-kinematic),
[162474 over 1.5 Myr](gallery.md#zlk-transient), [1981 without a verdict](gallery.md#zlk-gate-unavailable),
[15527 circulates](gallery.md#zlk-circulation), [2010 HY22 unphysical](gallery.md#zlk-invalid).
More detail: [Secular and ZLK classification details](../classify.md).
