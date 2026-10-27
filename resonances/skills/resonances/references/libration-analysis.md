<!-- Generated from docs/libration-analysis.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Libration analysis: full reference

!!! note "Start with the physics"
    This page lists every rule of the libration analysis with its formula and settings, for readers who want the
    details after the section [How the status is determined](physics/overview.md). The physics pages explain *why*
    each test exists; this page says exactly *what* is computed. One example figure per status is in the
    [Gallery](physics/gallery.md); every column of the output files is in [Output data](output.md); every setting is in
    [Configuration](config.md) and `resonances/.env.dist`.

The analysis lives in `resonances/resonance/analysis/`. MMRs (two-body, three-body, retrograde, co-orbital) go
through `analyse_mmr` (`analysis/mmr/`); secular and Lidov-Kozai angles share its geometry, ledger and confidence
rule and have their own signal step and physics ([Secular and ZLK classification details](classify.md)). The
analysis reads only the saved series: a run can be classified again without integrating
(`SimulationSerializer.restore(..., recompute_librations=True)`, `benchmark/tools/reclassify.py`).

Every comparison is written to a ledger (`decision_path`, the right-hand column of the combined figure); every
threshold is a setting. Units: time in years, angles in radians (degrees in the columns named `*_deg`), $a$ in au.
Notation as on the [overview](physics/overview.md#notation).

## 1. The verdict {#the-verdict}

| column | values | meaning |
|---|---|---|
| `status` | `resonant`, `transient`, `near_separatrix`, `non_resonant`, `uncertain`, `invalid` | see [Overview](physics/overview.md#the-verdict) |
| `angle_status` | `libration`, `episodic`, `near_separatrix`, `circulation`, `drift`, `insufficient_time`, `undefined` | the angle over the whole record (§7) |
| `angle_subtype` | `stationary`, `modulated`, `moving_centre`, `regime_change`; `recurrent`; `pattern_a`, `pattern_b`; `alias`; `arc`, `stall`, `few_cycles`; `unphysical`, `undefined_phase` | the subtype of `angle_status` |
| `final_regime`, `final_regime_start_yr` | `circulation`, `drift`, `near_separatrix`, `insufficient_time`, `libration`, empty | how the record ends: the last regime after librating episodes (G1c, G1e, an observed exit), or `libration` from the start of the last librating episode when it reaches the record's end (a capture, G1l, and also a libration over the whole record); empty when none of these holds or when the status tree set the word (kinematic drift) |
| `orbit_type` | `tadpole L4`, `horseshoe`, `quasi-satellite`, `interior, symmetric libration about 0°`, `libration about a moving centre (mean 21°)`, `episodes about 75° (not this resonance)`, … | the header of the figure; descriptive, decides nothing |
| `reason` | see below | the one main explanation |
| `libration_origin` | `dynamic`, `induced`, `subresonance`, `kinematic`, `unknown`, empty | who makes the bounded oscillation |
| `origin_detail` | `varpi_term`, `secular_cycle`, `planet_ripple`, `owner_libration`, `owner_forced_wave`, `fixed_subresonance`, `composite_multiplet`, `free_omega_winds` | the concrete cause |
| `owner` | e.g. `11M-20+0+9` | the resonance that carries the oscillation (induced, subresonance) |
| `confidence`, `confidence_raw` | `certain`, `high`, `medium`, `low`; empty for `uncertain` and `invalid` | §14 |
| `action` | `none`, `spot_check`, `review`, `extend`, `resample` | §14 |
| `check_mmr_*` | `pass`, `fail`, `not_measured`, `not_applicable` | `pass` supports the resonant reading, `fail` refutes it |
| `flags`, `caps`, `thin_margins`, `failed_support` | `;`-separated lists | what lowered the confidence and why |

**Reasons.** With a measured force: `dynamic_libration`, `dynamic_episode` (origin `dynamic`). With the geometry
alone, no force measured: `geometric_libration`, `geometric_episode` (origin `unknown`, leaf 5u/6 of §12; files
written before these names existed say `dynamic_*` with origin `unknown` and are read back as `geometric_*`). Near
the separatrix: `separatrix_zone`. No libration: `circulation`, `alias`, `no_resonant_potential`,
`modulated_circulation`. Another cause: `kinematic_libration`, `kinematic_drift`, `induced_libration`,
`subresonance_librates`, `implausible_strength`. Undecided: `baseline_needed`, `drift_mechanism_unknown`,
`origin_unknown`, `gate_unavailable` (Lidov-Kozai, [classify](classify.md)), `unusable_series` (secular and ZLK:
too few samples, non-finite values). Not measurable: `unphysical`, `undefined_phase`, `undersampled`,
`invalid_input`.

**Legacy integer.** `status_code` keeps the old integer for comparison with old runs only. MMR: resonant 2,
transient 1, near_separatrix −3, non_resonant 0 (kinematic, induced, subresonance and alias included), uncertain
with `baseline_needed` −6, other uncertain −9, invalid −99. Secular and ZLK: kinematic libration −2, near_separatrix
−3 (−5 with the flag `probable_near_separatrix`, read for compatibility), uncertain `insufficient_time` −4,
uncertain drift −6, other uncertain −9. The −2 (a measurement) and the −9 (no measurement) must never be merged.

**The angle word may change after the tree.** `angle_status` is decided by the geometry (§7), and three rules of
the status tree then rewrite only the reported word: a union of episodes that the tree promotes is reported as
`libration`/`regime_change` (§12, 6u); a kinematic drift of the slow centre reports the centre's word (`drift` or
`circulation`, §8.9); an angle that circulates once its forced waves are removed keeps the word that the forced
waves' loops give (§6.6). The status is never changed by these words. A body can therefore be `non_resonant` with
`angle_status = libration` (a kinematic or induced libration).

## 2. The model {#the-model}

The resonant angle (Murray & Dermott 1999, eq. 8.18, p. 329) and its rate model (eq. 8.22, p. 331):

$$
\sigma = k\lambda + \sum_j k_j\lambda_j + p\varpi + \sum_j p_j\varpi_j + v\Omega + \sum_j v_j\Omega_j ,
\qquad
\nu(a) = k\,n(a) + \sum_j k_j n_j + p\dot\varpi + \sum_j p_j\dot\varpi_j + \dots
$$

with $n(a) = 2\pi a^{-3/2}$ rad/yr. The $n_j$ are the package model's own mean motions (the slope of the saved
$\lambda_j$, else `PLANET_N_DEG_PER_GAUSS_YR`), and the slow rates are fitted on the record: $\nu(a)$ is a rate
model, not an exact identity at every sample. The exact resonance $a_*$ solves $\nu(a_*) = 0$; near it, with the slow
terms held fixed,

$$
\dot\sigma \simeq c\,(a - a_*), \qquad c = \frac{\partial\nu}{\partial a} = -\frac{3kn}{2a}
\qquad\text{(the identity; tested as L3, §8.1).}
$$

If the resonant term has a force, $c\,\dot a = -V'(\sigma)$ and the angle is a pendulum (M&D 1999, eqs. 8.45–8.48,
pp. 335–336):

$$
\ddot\sigma = -V'(\sigma), \qquad E = \tfrac12\dot\sigma^2 + V(\sigma), \qquad
h = \frac{E - V_{\min}}{V_{\max} - V_{\min}} .
$$

$h$ is normalised by the **global** extrema of the fitted $V$ (`Potential.energy_h`): $h < 1$ is libration and
$h > 1$ circulation for a single-well potential. With two harmonics a well may be bounded by a lower, local barrier;
then the separatrix of that well has $h_{\text{sep}} = (E_{\text{barrier}} - V_{\min})/(V_{\max} - V_{\min}) < 1$, and
$h < 1$ does not by itself mean confinement in that well. The per-cycle period and action use the well that holds
the cycle (below); the separatrix zone of §8.6 has its own normalisation.

For one harmonic, $V = -\omega_0^2\cos(\sigma - \sigma_s)$, a libration of half-amplitude $A$ has
(M&D 1999, eqs. 8.49–8.52, p. 337)

$$
P = T_0\,F(A), \qquad F(A) = \frac{2}{\pi}\,\mathsf K(m),\quad m = \sin^2\frac{A}{2},
\qquad \omega_0 = \frac{2\pi}{T_0} = |k|\,n\sqrt{3\mu_{\text{eff}}},
$$

$F(90^\circ) = 1.180$, $F(150^\circ) = 1.762$ ($\mathsf K$ with the parameter $m$, as in SciPy). Its half-width in $a$
(M&D 1999, eqs. 8.57–8.58, p. 338) and the half-excursion of $a$ at amplitude $A$ are

$$
\Delta a = \frac{2\omega_0}{|c|}, \qquad \delta a(A) = \Delta a\,\sin\frac{A}{2} .
$$

For a well that is not a cosine, $2\omega_0/|c|$ (with $\omega_0^2 = V''(\sigma_c)$) is only the
*cosine-equivalent* scale of its curvature; the physical half-width is $\sqrt{2(E_{\text{sep}} - V_{\min})}/|c|$ with
the barrier that bounds that well (`pendulum.py` keeps both).

The action of a libration with energy $E$ is the area of its loop divided by $2\pi$, i.e. the integral over one
branch between the turning points $\sigma_\pm$ (Wisdom 1985, §III, p. 277):

$$
J(E) = \frac{1}{2\pi}\oint p\,d\sigma = \frac{1}{\pi}\int_{\sigma_-}^{\sigma_+}\sqrt{2\,(E - V(\sigma))}\,d\sigma ,
\qquad \frac{dJ}{dE} = \frac{P}{2\pi} .
$$

For the cosine well $J_s = 8\omega_0/\pi$ on the separatrix and $J/J_s = \mathsf E(h) - (1-h)\,\mathsf K(h)$. For a
fitted $V$ with two harmonics, $J$ and $J_s$ of a libration cycle are quadratures in the well whose basin holds the
cycle's fastest sample (`pendulum.occupied_well`), up to the lower of the two maxima that bound it
(`pendulum.well_separatrix`), as its period (`pendulum.cycle_period`); $d(J/J_s)/dh$ uses that well's $J_s$
(`CycleEnergy.well_rad`, nan for the deepest well). With two harmonics the two wells differ in $J$ by a constant at one
energy, so the deepest well's ratio of a shallow-well cycle would be too large (W25, [The resonant
pendulum](physics/pendulum.md)). The revolution period above the separatrix is $2\mathsf K(1/h)/(\omega_0\sqrt h)$, and the ratio of the
slowest to the mean rate of a revolution is $r_{\text{pend}}(h) = \frac{2}{\pi}\sqrt{(h-1)/h}\,\mathsf K(1/h)$.

These are model diagnostics fitted to the record, not exact integrals of the N-body motion: a well that moves or
changes depth does not conserve $E$ exactly.

## 3. The steps {#steps}

`analyse_mmr(series, settings)` (`analysis/mmr/analyse.py`) runs in a fixed order: input check → context (rates,
$a_*$, $c$, synodic periods, the great inequality) → sampling (predictive unwrap, alias) → preprocess (ripple, cutoff,
filter) → geometry → slow parts → response of $a$ → energy and zone → kinematics → `angle_status` → filter guard →
subresonances → owner → strength → status → margins, caps, confidence.

A record outside the model ($k = 0$, $a \le 0$, $e \notin [0, 1)$, NaN) is `invalid` before anything is computed
(`invalid_input` for $k = 0$, else `unphysical`); so is an orbit with $a < 0$ or $e > 1.3$.

## 4. Sampling: unwrap and alias {#sampling}

**Predictive unwrap.** $\sigma \bmod 2\pi$ is exact at every sample; only the unwrap can err. The analysis unwraps
the $\lambda$ part $\psi_\lambda = \sigma - s$, $s = p\varpi + v\Omega + \sum_j(p_j\varpi_j + v_j\Omega_j)$, by the rate
model (`sampling.predictive_unwrap`):

$$
\psi_i = \psi_{i-1} + \bar\nu_{\lambda,i}\,\Delta t + \operatorname{wrap}\!\left(\psi_i^{\bmod} - \psi_{i-1} - \bar\nu_{\lambda,i}\,\Delta t\right),
\qquad \bar\nu_{\lambda,i} = \nu\!\left(\tfrac{a_{i-1}+a_i}{2}\right) - \langle\dot s\rangle,\quad \operatorname{wrap}\in[-\pi,\pi),
$$

i.e. the rate of the $\lambda$ part alone, $k\,n(\bar a) + \sum_j k_j n_j$ ($\langle\dot s\rangle$ is the mean rate of
the slow part measured on the record, column `ctx_slow_rate_arcsec_yr`); then $s$ is put back. When a planetary slow element of the angle was not saved, the
whole angle is unwrapped the same way with the full $\nu$.

The record is stitchable when the residual has a 99th percentile ≤ `MMR_UNWRAP_MAX_RESID_P99_DEG` (45°) and a maximum
≤ `MMR_UNWRAP_MAX_RESID_DEG` (135°); otherwise `undefined/alias`. The slow part $s$ is saved exactly but $\varpi$ can
turn by up to $\pi$ between samples where $e$ passes near 0; that is the flag `varpi_fast_low_e`, never an alias. The
body's $\varpi$ is wound by the low-passed vector $z_f$ of $e\,e^{i\varpi}$ at `MMR_JITTER_BAND_YR` (50 yr), which is
still exact modulo $2\pi$ but ignores fast loops around the origin.

**Undefined phase.** When the perihelion jitter $|p|\,\delta e/\bar e$ exceeds `MMR_JITTER_MAX_RAD` (0.785), the
whole angle exceeds the unwrap limits, and $\varpi$'s own scatter about $z_f$ exceeds that limit on more than
`MMR_VARPI_UNDEFINED_MAX_SHARE` (0.5) of the record, the angle is `undefined/undefined_phase` (invalid). The jitter
alone, with $\varpi$ defined, is a flag and the cap `jitter_low_e` (≤ medium).

**The step verdict** comes from $a$ alone. On the range of $a$ in the record (1–99 %), the branch is
$m = \operatorname{round}(\nu\Delta t/2\pi)$ and the apparent rate $\nu - 2\pi m/\Delta t$:

| `smp_verdict` | condition |
|---|---|
| `resolved` | $\lvert\nu\rvert\,\Delta t \le$ `MMR_STEP_RESOLVED_DEG` (90°) everywhere |
| `marginal` | ≤ 180°, one branch |
| `aliased_unambiguous` | one branch $m \neq 0$: a true circulation folded by the step |
| `ambiguous` | several branches |

Also recorded: the ghost $a_m$ where branch $m$ looks like an exact resonance ($\nu(a_m) = 2\pi m/\Delta t$), the
Nyquist half-width $\delta a_N = a^{5/2}/(3|k|\Delta t)$, and the far ratio
`smp_far_ratio` $= |\langle a\rangle - a_*| / \max(\tfrac12\text{range}(a),\ |a(\dot\lambda) - \langle a\rangle|)$.
A libration needs $a$ to reach $a_*$, so a large far ratio excludes it. An aliased circulation is
`non_resonant/alias` when the branch is unique and the far ratio ≥ `MMR_C5_FAR_RATIO_MIN` (3), otherwise
`invalid/alias` with `action = resample`. With an aliased step the identity and the force law are not applicable
($\dot\sigma$ is then the model rate).

## 5. The signal: ripple, cutoff, filter {#filter}

**Ripple.** The planets' longitudes enter $\sigma$ linearly, so their slow inequalities (the Jupiter–Saturn great
inequality, about 900 yr) are subtracted when the planets are saved:
$\sigma_{\text{clean}} = \sigma - \sum_j k_j\,\delta\lambda_j^{\text{slow}}$ (`MMR_RIPPLE_REMOVE`, band
`MMR_RIPPLE_BAND_YR`). Without saved planets the flag `ripple_not_removed` is set and the predicted ripple raises the
turning-point prominence. A co-orbital angle keeps its planet's longitude as it is (the body follows the planet's slow
longitude).

**Shortest libration** (`flt_period_floor_yr`): `MMR_LIBRATION_PERIOD_MIN_YR` (150 yr, a convention); for a 1:1 angle
`MMR_COORBITAL_PERIOD_FLOOR_FACTOR` (0.5) × the linear tadpole period $P_{\text{orb}}/\sqrt{27\mu/4}$
(M&D 1999, eq. 3.151, p. 94).

**Cutoff** (physics only, the output step does not enter):

$$
P_c = \max\!\left(f_{\text{syn}}\max_j P_{\text{syn},j},\ \frac{P_{\text{floor}}}{f_{\min}}\right),
\qquad P_{\text{syn},j} = \frac{2\pi}{|n - n_j|}\ \text{(prograde)},\quad \frac{2\pi}{n + n_j}\ \text{(retrograde)},
$$

with $n, n_j > 0$ (the same periods enter the short-period folding checks),
$f_{\text{syn}}$ = `MMR_FILTER_SYNODIC_FACTOR` (1.5), $f_{\min}$ = `FILTER_MIN_LIBRATION_CUTOFFS` (2);
`MMR_FILTER_CUTOFF_YR` overrides. A co-orbital planet has no synodic wave of its own and does not enter. The usual
result is $P_c = 75$ yr for an MMR whose synodic periods are below 50 yr. Filtering before decimating follows
Nesvorný & Morbidelli (1998, §2, p. 3030).

**Filter.** A Butterworth filter of order `FILTER_ORDER` (2), applied forward and backward to the detrended unwrapped
angle (the trend is put back, so the number of turns is kept). Its gain is

$$
G(P) = \left[1 + \left(\frac{\tan(\pi\Delta t/P)}{\tan(\pi\Delta t/P_c)}\right)^{2N}\right]^{-1}
\;\to\; \left[1 + (P_c/P)^{4}\right]^{-1},
$$

so $G(2P_c) = 0.94$, $G(P_c) = 0.5$. Turning points and passes are taken on $[t_0 + P_c,\ t_1 - P_c]$
(`FILTER_EDGE_TRIM_CUTOFFS` = 1).

**The step as a condition.** $P_c \le 2\Delta t$: no filter of the record exists, the verdict is
`invalid/undersampled` before any measurement, action `resample` at the largest of `MMR_RESAMPLE_STEPS_YR` ≤
$P_{\text{floor}}$ / `MMR_RESAMPLE_STEPS_PER_CYCLE` (20) within the alias bounds.

**Short-period folding.** When $\Delta t \ge \min(P_{\text{orb}}, P_{\text{syn}})/2$, or a harmonic $m/P$
($m \le$ `MMR_SHORT_PERIOD_HARMONICS` = 2) of $P_{\text{orb}}$ or $P_{\text{syn}}$ folds to a period slower than the
period floor (`sampling.short_period_trigger`, the dry-run's predicate too), the short-period terms of the elements
fold into slow lines that no filter removes. The flag `short_period_aliased` is set, with the cap of the same name
(≤ medium). The folded lines barely move $\sigma_f$ but can sit in $a$ at up to its whole half-excursion, so they
break the $a$-based checks. The cap stays on every such record except a circulation (reason `circulation` or `alias`)
without forced-wave runs (`MMR_SHORT_PERIOD_CAP_CIRCULATION` = False); the record never lifts it
(`MMR_SHORT_PERIOD_CAP_RATIO` = 0; a positive value enables an opt-in lift). The scatter of $a$ against the decisive
episode's half-excursion above `MMR_SHORT_PERIOD_A_RATIO` (0.5) sets the warning flag `a_short_period_scatter` and
gates the rule of §8.2 (folded short periods). `smp_fold_free_step_yr` is the largest step of the grid below the
record's step that folds nothing (`sampling.fold_free_step`); when this cap holds the level down
(`confidence.cap_binds`), the action is `resample` at that step.

**Derivatives** (identity, force law, potentials) are read after a second smoothing at
the larger of $P_c$ and `MMR_RESPONSE_CUTOFF_CYCLES` (0.1) × $P_{\text{cyc}}$: a derivative multiplies what leaks through
$P_c$. The energy and the zone read $\sigma$ at the larger of $P_c$ and `SEPARATRIX_CUTOFF_CYCLES` (0.2) × $P_{\text{cyc}}$.

**Filter too slow.** A libration period with gain $G(P_{\text{cyc}})$ below `MMR_L8_GAIN_FULL` (0.94) is shaped by
the filter: flag `filter_too_slow`, the shape checks (L2, L5, L6, one pendulum, R0, the force law) are not measured.

**Quasi-satellite floor** (G1q, `MMR_COORBITAL_QS_FLOOR`). A quasi-satellite librates about $\theta = 0$ much faster
than a tadpole. Its period comes from the 1:1 potential (§8.3): the local minimum of $\tilde R(\theta)$ nearest 0 whose
basin holds $\theta = 0$, $\omega_{QS}^2 = 3\mu n^2\tilde R''(\theta_c)$, $P_{QS} = 2\pi/\omega_{QS}$. The pass at this
floor is described in §7. The 1:1 force law is then tabulated on blocks of $\min(P_c, P_{QS})$, at most
`MMR_COORBITAL_QS_MAX_BLOCKS` (4000); an episode takes the quasi-satellite floor only where the well exists on at
least `MMR_COORBITAL_QS_WELL_SHARE` (0.5) of its samples.

**The ×2 filter guard.** The decisive numbers (turning points, episode boundaries, stalls, the energy per cycle,
amplitude and centre shift) are measured again at $2P_c$ on the interval both cutoffs see. The decisive episode is
compared with its counterpart at $2P_c$, the accepted episode that overlaps it most in time (none: the episode is not
reproduced); without an episode, the whole interval. A change beyond `MMR_FILTER_GUARD_*` (turning points by more
than `MMR_FILTER_GUARD_TP_REL` / `_TP_ABS`, amplitude by `MMR_FILTER_GUARD_AMP_REL`, the energy across $h = 1$ or by
more than `MMR_FILTER_GUARD_H_TOL` × max(1, the smaller $|h|$)) sets `filter_sensitive` and empties the checks and
margins that read it. The guard runs **before** the tree: a passed R0 (with $a$) that it withdraws is not measured for
the tree as well (`decide.energy_for_tree`, column `flt_withdrawn_before_tree`), so it can change the branch the tree
takes; a failed R0 stays an objection.

## 6. Geometry: turning points, cycles, episodes {#geometry}

### 6.1 Turning points

A **turning point (TP)** is a local extremum of $\sigma_f$ with prominence

$$
p_{\text{TP}} = \max\!\left(p_{\text{floor}},\ f_r\,r\right),
$$

$p_{\text{floor}}$ = `MMR_TP_PROMINENCE_FLOOR_DEG` (1°), $f_r$ = `MMR_TP_RIPPLE_FACTOR` (4), $r$ the unremoved ripple or
the 95th percentile of what $\sigma_f$ holds faster than the shortest libration (a wave of half-amplitude $r$ makes
extrema of prominence up to $2r$). Extrema alternate (of two maxima in a row the higher is kept). Before $r$ is
measured, the image of each provisional libration run through the same two filters is subtracted, so that a libration
near the period floor does not become its own noise. Each run of ≥ 3 provisional TPs gets its own $p_{\text{TP}}$
measured on its own samples (a regime change elsewhere does not inflate it), and each regime between jumps of $a$
(§6.9) its own too.

The **cycle period** $P_{\text{cyc}}$ is the median of the full periods $t_{i+2} - t_i$: three alternating TPs
(max–min–max) make **one full cycle**.

### 6.2 The centre and its passes

The centre $\sigma_c(t)$ is $\sigma_f$ smoothed over $W_c$ = `LIBRATION_CENTRE_CYCLES` × $P_{\text{cyc}}$ (3;
$3P_c$ without cycles). Its turns have prominence the larger of
`LIBRATION_CENTRE_TP_PROMINENCE_DEG` (10°) and `LIBRATION_CENTRE_TP_FACTOR` (2) × $p_{\text{TP}}$. A *monotone pass* is the stretch between two turns; a
pass ≥ $2\pi$ is a revolution, and an arc longer than $2\pi$ is never a piece of libration.

### 6.3 Runs and episodes

TPs are cut into **runs** where a gap exceeds `LIBRATION_EPISODE_GAP_FACTOR` (3) × the median half-period, a
half-amplitude reaches $\pi$ (over the top), or two midpoints differ by more than $\pi$ (a slip).

An **episode** is a run of ≥ `LIBRATION_MIN_TURNING_POINTS` (3 = one full cycle) TPs whose centre is

- *stationary*: the bands $[m_i - A_i,\ m_i + A_i]$ of all its full cycles share a point ($A_i$ half-amplitudes,
  $m_i$ midpoints). With fewer than `LIBRATION_CENTRE_MIN_ESTIMATES` (2) independent cycles the episode is accepted
  but its centre is not measured (thin L6, cap `three_tp_episode`, ≤ high);
- or *modulated*: the centre turns back ≥ `LIBRATION_CENTRE_RETURNS_MODULATED` (3) times within a range
  < `LIBRATION_MODULATED_RANGE_MAX_RAD` ($\pi$). This is a measured property of the record, not a theorem: the centre
  of a multiplet with one dominant member stays bounded only under conditions on the weaker members
  (Morbidelli 2002, §9.4);
- or *moving_centre*: the bands share a point in $\psi = \sigma - \sigma_c(t)$, with $\sigma_c(t)$ the moving minimum of
  the averaged potential (§6.5), and every band holds $\psi = 0$;

and at least one of its full cycles is **closed** (in $\psi$ for a moving centre): its net advance is below the mean
of its two half-swings,

$$
Q_i = \frac{|\sigma_{i+2} - \sigma_i|}{\tfrac12\left(|\sigma_{i+1} - \sigma_i| + |\sigma_{i+2} - \sigma_{i+1}|\right)} < 1 .
$$

One closed cycle is enough (entry, exit and a drifting centre leave a real libration's cycle open); a circulation that
stalls and turns back a little closes none.

An episode counts as a libration of this angle only when its period is not below the shortest libration (else it
is a forced wave) and its centre makes no revolution inside it. A run whose centre moves is judged again on its regular
stretch (full periods ≤ `LIBRATION_REGULAR_PERIOD_RATIO` (2) × the median), and failing that without at most one slow
edge half-cycle at each end. Rejected runs keep their reason in episodes.csv (`too_few_tp`, `centre_moves`,
`no_closed_cycle`, `over_the_top`, `edge`).

### 6.4 Closure by phase return (MMR)

A run of only two opposite TPs ($v_1$ at $t_1$, $v_2$ at $t_2$, $\Delta = t_2 - t_1$) is one closed cycle when
$\sigma_f$ crosses a level $\sigma_m$ inside the swing before $t_1$ (at $t_a$) and after $t_2$ (at $t_b$) in the same
direction and returns to the same state:

1. $|a(t_b) - a(t_a)| \le$ `LIBRATION_RETURN_TOL` (0.05) × the swing of $a$ on $[t_a, t_b]$;
2. $\dot\sigma(t_b)/\dot\sigma(t_a)$ within $1 \pm$ that tolerance;
3. $(t_b - t_a)/(2\Delta)$ within $1 \pm$ `LIBRATION_RETURN_PERIOD_TOL` (0.25), the quarter-period symmetry of a cycle;
4. $a$ carries the swing: $\operatorname{corr}(\dot\sigma,\ \operatorname{sign}(c)(a - \bar a)) \ge$
   `LIBRATION_RETURN_IDENTITY_CORR_MIN` (0.9);
5. the swing is symmetric: between the TPs $\sigma_f$ crosses $\sigma_m$ back as fast as at $t_a$,
   $|\dot\sigma(t_m)/\dot\sigma(t_a) + 1| \le$ `LIBRATION_RETURN_TOL`.

The mid-level and `LIBRATION_RETURN_LEVELS` (19) further levels are tried; a level counts only inside a band of
consecutive closing levels, ≥ `LIBRATION_RETURN_MID_BAND_LEVELS` (2) for the mid-level and ≥
`LIBRATION_RETURN_BAND_LEVELS` (2) for another (a real return holds over a band of sections; a match at one isolated
level is a coincidence). Levels whose crossing may lie beyond the record neither close nor break a band. The crossings
are sought inside the run's own regime between jumps of $a$. Such a run is an accepted stationary episode of one cycle
of period $t_b - t_a$; a verdict that rests on it has the cap `one_return_cycle` (≤ medium,
`LIBRATION_RETURN_CYCLE_CAPS_MEDIUM`).

### 6.5 Libration about a moving centre (two-body)

A two-body angle librates about the minimum of the averaged disturbing function $\langle D\rangle(\sigma)$ of its
planet, and the minimum moves with the slow elements (Gallardo 2006, §3, p. 30). When the well changes much more slowly
than one libration and the orbit stays away from the separatrix (the adiabatic condition; Wisdom 1985, pp. 274, 279),
$\sigma$ librates about a slowly moving centre. Only for a run that fails in $\sigma$ (and is not modulated), and only
for a two-body angle with its planet saved (1:1 and retrograde included): $\sigma_c(t)$ is the minimum of the full
$\langle D\rangle$ (`averaged_potential`) at the elements averaged over $P_c$ at nodes the larger of $P_c$ and the span /
`MMR_CENTRE_TRACK_NODES` (400) apart, followed by descent from the previous node. A node without a well, without a
converged quadrature or within `MMR_STRENGTH_MIN_DISTANCE_HILL` (3) Hill radii of the planet leaves the run in
$\sigma$. The track counts only when the observed full-cycle centres follow it: their RMS about $\sigma_c(t)$ is below
that about the best fixed centre and below that about the same track run backwards in time. $\psi$ is read at the TPs
of $\sigma$, and the force law of such an episode is fitted in $\psi$ (§8.2). This implementation has no averaged
potential for three-body angles, so they stay in $\sigma$.

### 6.6 Forced waves (MMR)

Once each accepted episode's own response is measured (§8.4), an episode is a forced wave of another argument, not an
episode of the angle, when its identity fails with the wrong sign (slope/$c < 0$ at $|\text{corr}| \ge$
`MMR_IDENTITY_CORR_MIN`) or when its period sits on a forced line of the angle's lattice (§8.2) and its identity
fails (the line alone suffices only with `MMR_FORCED_WAVE_LINE_ALONE`). Forced waves are kept in
`Geometry.forced_waves` (episodes.csv `forced_wave`), and the response, the energy, the exit and the entry are read
again without them. When the status' angle then circulates or drifts, the reported word is the one the forced waves'
loops give (`decide.forced_wave_word`): they are loops of $\sigma$; status, reason and confidence stay those without
them.

### 6.7 Regime changes (MMR)

A run still rejected for its moving centre is cut where its regime changes, and kept cut only when the regimes the cut
claims are accepted (`LIBRATION_REGIME_CUTS`): (1) a *centre step*, a half-cycle with ≥ 3 TPs and a closed cycle on
each side, each side passing the band test while the two together do not, each side with a measured centre, and the
cycle centres differing across the step by more than within either side; failing that, (2) a *stall of the slow
centre*: the longest stretch of ≥ `LIBRATION_STALL_MIN_CYCLES` (3) independent cycles whose midpoints move less per
period than the cycles swing, at the run's own period, cut out as its own run. A steady drift has no stall and is not
cut.

### 6.8 Coverage, union, exit and entry

- **Coverage.** An episode covers an edge when it starts (ends) within `LIBRATION_COVERAGE_PERIODS` (1) of its
  periods from the record's edge, or when no TP is missing at the edge of the analysis window (the gap no longer than
  the longest TP-to-TP step inside the episode, nor than that many periods) and the stretch the filter trims off is
  itself within that many periods. Covering both edges → libration, else episodic (`recurrent` with ≥ 2 episodes).
- **Union of episodes** (`geometry.covering_union`). Consecutive librating episodes join when nothing between them
  goes round (the range of $\sigma_f$'s TPs and of the orbit-averaged raw angle in the gap both below $2\pi$) and the
  gap is ≤ `LIBRATION_UNION_GAP_PERIODS` (1) × $(P_{\text{left}} + P_{\text{right}})$. A chain of ≥ 2 episodes counts
  when the TPs and the averaged raw angle also stay within $2\pi$ over its whole span, and each record edge is covered
  or reached. Such a chain is a libration with subtype `regime_change`: rule **G2** for a 1:1 angle, **G2u** for any
  other angle, which the status tree reads with a guard (§12, 6u).
- **Suspected exit/entry.** After the last TP $\sigma_f$ leaves the episode's centre by more than $\pi$ with less than
  one period left: flag and cap `exit_suspected` (≤ medium) on a verdict that rests on that edge; the same for
  `entry_suspected`.
- **Observed exit/entry** (MMR). The raw angle averaged over one orbital period runs ≥ `LIBRATION_EDGE_ARC_REV`
  (1) turn monotonically from the episode's last TP to the record's end (entry: from the start to the first TP). The
  edge is then not covered, there is no `exit_suspected`, and `final_regime` is a circulation from the exit. The
  centre-scale pass (G2c) does not look for a slower libration behind such an arc.

### 6.9 Slips, stalls, arcs, jumps of $a$

- **Slips (pattern A):** steps of $2\pi$ (within ± `LIBRATION_SLIP_TOLERANCE` = 0.2 of it) of the oscillation midpoint that
  take at least `LIBRATION_SLIP_MIN_DURATION_T0` (0.5) × $T_0$.
- **Stalls (pattern B):** for each revolution the phase and depth of the rate minimum and its energy; pattern B when
  the turns lie in the zone the separatrix reaches and ≥ `LIBRATION_STALL_REV_FRACTION` (0.75) of the minima sit within
  `LIBRATION_STALL_PHASE_TOL_DEG` (30°) of the potential top.
- **No TPs:** `arc` (range of $\sigma_f \ge \pi$) or `stall` (< $\pi$), with a lower bound of the libration period.
  A libration about a centre 0 or $\pi$ moves $a$ at $|\dot a| = K|\sin\sigma|$, so over the analysis window $W$ of
  length $\tau$ with the range $\delta a = \max_W a_f - \min_W a_f$ and $s_{\min} = \min_W|\sin\sigma_f|$:
  $K_{\max} = \delta a/(\tau\,s_{\min})$ and
  $P_{\text{lib,min}} = 2\pi/\sqrt{|c|K_{\max}} = 2\pi\sqrt{\tau\,s_{\min}/(|c|\,\delta a)}$. The least $|\sin|$ gives the most cautious bound; $s_{\min} = 0$ gives no positive bound.
- **Jumps of $a$.** A candidate is a step of $a$ over $\tau = \max(2P_c, 5\Delta t)$ (`LIBRATION_JUMP_TAU_CUTOFFS`,
  `LIBRATION_JUMP_TAU_STEPS`) above `LIBRATION_JUMP_NOISE_FACTOR` (8) robust standard deviations, faster than
  $2\pi/\omega_0$. It is a **jump** when the energy of `LIBRATION_JUMP_CYCLES` (2) cycles on each side differs by more
  than the larger of $2\sigma_h$ and `LIBRATION_JUMP_DH_MIN` (0.1); with fewer cycles, when the step exceeds the half-width
  (flag `jump_by_a_only`). A step without an energy change is a `kick`; one closer than $\max(\tau, P_{\text{cyc}})$ to
  an edge cuts nothing (`jump_at_edge`). After a jump the geometry is measured again on the last regime (flag
  `last_regime_after_jump`); each regime has its own centre, its own prominence, and no run crosses a jump.

### 6.10 Steps of the libration action (MMR, not 1:1)

Away from the separatrix the action is an adiabatic invariant; an amplitude that changes by a large factor within one
half-cycle is a candidate impulse onto another energy level, possibly in another well of the same potential
(`LIBRATION_ACTION_STEP`). On every accepted run in the $\sigma$ frame that no jump of $a$ cuts, the half-amplitudes
$A_j$ are split at the index of largest band separation; the transitional half-cycle belongs to neither side. The
split cuts the record when all hold:

1. each side has ≥ `LIBRATION_ACTION_STEP_MIN_HALF_CYCLES` (2) half-cycles;
2. min(larger side) / max(smaller side) ≥ `LIBRATION_ACTION_STEP_SEPARATION` (2);
3. the side medians differ by ≥ `LIBRATION_ACTION_STEP_RATIO` (3);
4. the swing of $a$ changes the same way by ≥ `LIBRATION_ACTION_STEP_A_RATIO` (1.5), since $\delta a \propto \dot\sigma$;
5. the transitional half-cycle lasts ≤ `LIBRATION_ACTION_STEP_MAX_PERIODS` (1) shorter-side period;
6. ≥ `LIBRATION_ACTION_STEP_CYCLES` (1) energy cycle per side, and $h$ changes the way the amplitude does by more than
   the larger of $2\sigma_h$ and `LIBRATION_JUMP_DH_MIN`;
7. **both regimes librate in one potential, each at its own energy** (`regime_periods`): the median periods of the two
   sides differ by more than `LIBRATION_ACTION_STEP_PERIOD_TOL` (0.2) (else one forcing frequency explains both and the
   period law cannot tell); and at every shape cutoff $f\times$ the shorter side period, $f$ in
   `LIBRATION_ACTION_STEP_CUTOFFS` (0.075, 0.1, 0.15), on the disjoint cycles $(TP_i, TP_{i+2})$ of each side: $V_E$
   fitted on the cycles of the larger swings explains ≥ `SEPARATRIX_VE_R2_MIN` (0.7) of $\dot\sigma^2$ and covers the
   $\sigma$ range of the smaller ones; each cycle (held out from the fit when possible) has a loop of its energy around
   the minimum that holds its fastest sample (`pendulum.loop_through`), $T_{\text{obs}}/T_{\text{pred}}$ within
   `LIBRATION_ACTION_STEP_PERIOD_TOL` of 1, and observed extremes within `LIBRATION_ACTION_STEP_TP_TOL` (0.2) × its
   half-swing of the loop's turning points; each side's cycle energies spread by ≤ `LIBRATION_ACTION_STEP_ENERGY_DRIFT`
   (0.5) × the step of the energy; and $V$ has the same number of wells at every cutoff.

The cut is a confirmed jump (`jump_action_step` = 1) at the middle of the transitional half-cycle, whose zone is that
half-cycle; the rules of §7 then read the regimes as after a jump of $a$. With fewer than `LIBRATION_JUMP_CYCLES`
energy cycles on a side, or one disjoint cycle of the larger swings, the cut is provisional: cap
`action_step_provisional` (≤ medium). **Guard:** a record cut anywhere is also analysed whole; when the cut leaves a
check that failed on the whole record (identity, force law, R0, zone, strength) unmeasured and changes the status or the
confidence, the whole record's analysis is kept and the step is reported as withheld
(`geo_action_step_withheld_checks`). Columns `geo_action_step_*`; panel (a) of the figure marks the cut.

## 7. angle_status {#angle-status}

The first rule that fires (`decide.decide_angle`):

| rule | condition | angle_status |
|---|---|---|
| G0 | unphysical orbit; unwrap residual over its limits; phase undefined (§4); $P_c \le 2\Delta t$ | `undefined` (`unphysical`, `alias`, `undefined_phase`); `invalid/undersampled` |
| G1 | a confirmed jump of $a$ (or an action step) | the rules below run on the last regime |
| G1c | after the last TP the centre completes a revolution, for longer than one period of the last episode | `episodic` (`recurrent` with ≥ 2 episodes), `final_regime = circulation` from the last TP |
| G2 | a librating episode | covering → `libration` (`stationary`, `modulated`, `moving_centre`; a union, §6.8 → `regime_change`), else `episodic` |
| G1l | a libration of the last regime after a jump, but no librating episode from the record's start to the jump | `episodic`, `final_regime = libration` from the start of the last episode (a capture); a libration on both sides of the jump stays `libration` |
| G1q | a prograde 1:1 angle without a libration whose orbit-averaged angle stays in the quasi-satellite well for ≥ (`LIBRATION_MIN_TURNING_POINTS`/2) × $P_{QS}$, with the quasi-satellite floor below the tadpole floor | analysed again at the floor `MMR_COORBITAL_PERIOD_FLOOR_FACTOR` × $P_{QS}$; that verdict is kept when it holds a librating episode centred in the well and no fewer episodes outside the well than the first pass. When the step cannot filter that cutoff, action `resample` or `review`, never `extend` |
| G2c | no libration at $P_c$ (and no observed exit or entry), but the centre librates | the record is analysed again at $W_c$ = `LIBRATION_CENTRE_CYCLES` × $P_{\text{cyc}}$ (failing that, $P_t$/`FILTER_MIN_LIBRATION_CUTOFFS`, $P_t$ the period of the centre's own turns), with turning points at the centre-turn prominence that qualified the scale (`LIBRATION_CENTRE_SCALE_TP_AT_CENTRE`); kept when it is a libration, or an episode that covers the record but for an observed exit or entry. A libration too fast for $W_c$ is analysed once more at $P$ / `FILTER_MIN_LIBRATION_CUTOFFS`. A G2u union the status at $W_c$ did not promote is not kept; the rejected analysis is in `geo_cs_rej_*` |
| G3 | no episode, the $\sigma$-only potential holds (R0σ), and ≥ `LIBRATION_PATTERN_A_MIN_SLIPS` (2) slips of pieces reaching the separatrix (median amplitude ≥ `LIBRATION_PATTERN_A_MIN_AMP_DEG`, 60°) or pattern B | `near_separatrix` (`pattern_a`, `pattern_b`) |
| G4 | the largest centre pass ≥ $2\pi$ | `circulation` (`alias` when the step is aliased) |
| G4b | the slow centre advances ≥ `LIBRATION_CIRCULATION_NET_REV` (2) revolutions net, though no single pass reaches one | `circulation` |
| G5 | rejected `centre_moves` runs cover ≥ `LIBRATION_DRIFT_MIN_WINDOW_FRACTION` (0.5) of the window | `drift`; G5a: when the slow centre turns back exactly once, farther than $W_c$ from the window's edges, and stays within one revolution: `insufficient_time/arc` (part of one slow swing) |
| G6 | TPs but no episode / no TPs | `insufficient_time` (`few_cycles` / `arc` / `stall`) |
| G1e | the last regime holds no episode, but librating episodes precede the jump | `episodic`, `final_regime` = the last regime's word from the jump; the status reads the earlier episodes with their own response and no energy (cap `episode_before_jump`, ≤ medium) |

## 8. Is it this resonance? The response of $a$ {#response}

### 8.1 Identity (L3, `check_mmr_identity`)

Regression of $\dot\sigma_f$ on $a_f$: pass when $|\text{slope}/c - 1| \le$ `MMR_IDENTITY_SLOPE_TOL` (0.2) and the
correlation (taken in the direction of $c$) ≥ `MMR_IDENTITY_CORR_MIN` (0.9). It almost always holds; a failure points
to an alias or to a $\varpi$ share of the oscillation.

### 8.2 Force law (L4, `check_mmr_a_response`)

In windows of `MMR_FORCE_WINDOW_CYCLES` (3.5) cycles (step `MMR_FORCE_WINDOW_STEP`, 0.5), least squares

$$
\dot a_f = R\sin(\sigma_f - \phi_0) + C .
$$

Since $\ddot\sigma = c\,\dot a$, the stable centre is $\phi_0$ when $c < 0$ and $\phi_0 + \pi$ when $c > 0$ (most
prograde catalogue angles have $k < 0$, so $c > 0$). Pass: $R^2 \ge$ `MMR_FORCE_R2_FULL` (0.7) in both halves of the
record (thin for $R^2 \in [0.4, 0.7)$); fail: $R^2 <$ `MMR_FORCE_R2_FAIL` (0.4) under every law tried. Not measured
below `MMR_A_RESPONSE_MIN_AMP_DEG` (45°): there $\sin\sigma$ and $\sigma$ differ by less than 10 % and any oscillation
carried by $a$ fits. With a free $C$ the spread of $\phi_0$ is degenerate with $C$; it is a column only.

- **Multiplet law.** $\dot a_f = \sum_j [A_j\sin\sigma_j + B_j\cos\sigma_j] + C$, $\sigma_j = \sigma + j(\varpi - \varpi_P)$
  (the members of one commensurability; Morbidelli 2002, §9.3), taken when it gains ≥ `MMR_FORCE_MULTI_GAIN` (0.1) in
  $R^2$; its leading coefficients must stay coherent (≥ `MMR_FORCE_COEF_COH_MIN`, 0.8).
- **Several harmonics of $\psi$.** A pendulum in a well that is not a sinusoid has
  $\dot a = \sum_{k=1}^{N}(A_k\sin k\psi + B_k\cos k\psi) + C$. When L4 fails or passes thin (never for a 1:1 angle,
  never inside a null test), $N$ grows from 1 while $R^2$ gains ≥ `MMR_FORCE_MULTI_GAIN`, up to
  `MMR_FORCE_HARMONICS_MAX` (2). The law is taken only with $N \ge 2$ and when a shifted null exists, the leading
  coefficients are coherent, $R^2$ beats the null by `MMR_FORCE_NULL_GAIN`, $R^2 \ge$ `MMR_FORCE_R2_FAIL`, and the law
  fitted on one half of the samples predicts the other half, both ways, with $R^2 \ge$ `MMR_FORCE_R2_FAIL`. It can
  lift L4, never lower it. A kinematic linear swing $\dot a \propto -\sigma$ wider than about ±150° can pass it too,
  so L4 stays necessary, not sufficient. Columns `rsp_force_harm_*`.
- **Lines above the cutoff and the null test.** Each measurement first fits the law at the base cutoff, takes $K$,
  and reads the amplitude spectrum of $\dot a$ from `MMR_RESPONSE_LINE_BAND_CUTOFFS` (0.5) × that cutoff to $P/2$,
  leaving out the libration's harmonics. Every peak ≥ `MMR_RESPONSE_LINE_AMP_FRAC` (0.5) × $K$ raises the cutoff to
  `MMR_RESPONSE_LINE_MARGIN` (1.5) × its period, at most $P$ / `MMR_RESPONSE_MAX_FRACTION` (4). A pass then needs its
  $R^2$ at least `MMR_FORCE_NULL_GAIN` (0.2) above the largest $R^2$ of the same law with $\dot a$ shifted by
  $(m + \tfrac14)P$ and $(m + \tfrac34)P$ (`MMR_FORCE_NULL_SHIFTS` = 8 shifts, up to half the record); below the gain
  L4 fails, without a null (half the record shorter than $1.75P$) it is not measured.
- **Jumps of $a$.** Windows that overlap a confirmed jump ± ($\tau/2$ + `FILTER_EDGE_TRIM_CUTOFFS` × the response
  cutoff) are dropped; fewer than `MMR_FORCE_MIN_WINDOWS` (2) clean windows leave L4 not measured. Kicks do not count.
- **Contamination by a forced line** (`MMR_FORCE_CONTAMINATION = lattice`). Inside a libration every argument
  $m\lambda + \sum_j l_j\lambda_j$ of the angle's planets turns at $\nu = m\,n_c + \sum l_j n_j$, and through the identity
  a wave of $a$ at that frequency gives $\dot a = -(\nu^2/c)(\sigma - \sigma_c)$, a law with no resonant force. When a
  line of the lattice ($1 \le m \le |k|$, $|l_j| \le |k_j|$) lies within the window's resolution
  $|1/T - 1/P| < 1/(WP)$ of the measured period, a measured L4 becomes not measured (`rsp_force_contaminated`); it never
  makes a libration dynamic (leaf 5f).
- **Folded short periods.** On a record with `short_period_aliased`, a failed L4 and a failed L3 are read as not
  measured by the tree and the confidence when the strength F by direct averaging passes and the short-period scatter
  of $a$ exceeds `MMR_SHORT_PERIOD_A_RATIO` (0.5) × the decisive half-excursion (`MMR_SHORT_PERIOD_A_UNMEASURED`). A
  pass is never changed, the cap stays, and the check columns keep the measured fail (`rsp_short_period_unmeasured`).

### 8.3 The co-orbital law (1:1)

A 1:1 angle cannot be described by a few harmonics: its averaged potential is singular at $\theta = 0$ for $e = i = 0$.
The direct and indirect disturbing function (M&D 1999, eq. 6.13, p. 227),
$R = Gm_P\left(|\mathbf r_P - \mathbf r|^{-1} - \mathbf r\cdot\mathbf r_P/r_P^3\right)$, is averaged over the planet's
longitude at fixed $\theta = \lambda - \lambda_P$ (tabulated per slowly varying $e, i, \omega, \Omega$; softened at
`MMR_COORBITAL_SOFTENING_HILL`, 0.1 Hill radius), and Lagrange's equation (eq. 6.145, p. 251) gives

$$
\dot a = \frac{2}{na}\frac{\partial\langle R\rangle}{\partial\theta} = 2\mu n a\,\tilde F(\theta),
\qquad \tfrac12\dot\theta^2 + 3\mu n^2\tilde R(\theta) = E,
\qquad \tilde R = \frac{a\langle R\rangle}{Gm_P} .
$$

For $e = i = 0$, $\tilde R = 1/(2|\sin(\theta/2)|) - \cos\theta$ (M&D 1999, eqs. 3.178–3.179, p. 105), so
$\omega_{\text{tad}}^2 = \tfrac{27}{4}\mu n^2$. The model is evaluated on the unfiltered angle and passed through the
data's filters; $\dot a_{\text{data}} = K\dot a_{\text{model}} + C$. L4 passes with $R^2 \ge 0.4$ in both halves (thin below 0.7;
`MMR_FORCE_R2_FAIL`, `MMR_FORCE_R2_FULL`) and $K/K_{\text{theory}}$ within [1/3, 3] overall and in both halves
(`MMR_COORBITAL_SCALE_RATIO_MAX`; $K_{\text{theory}} = 1$ is the planet's own mass). R0 is the energy integral fitted per cycle, from the same law the tree reads. The 45° floor of the harmonic law
does not apply. The strength check F is not applicable to a 1:1 angle, and no owner exists for it.

The retrograde 1/−1 (`1J-1r<k>`, $i > 90°$) is judged by the same law, with the body placed at
$M = \sigma + \lambda' - ((2k-1)\omega + \Omega)$ (the angle of Li et al. 2019, built from $\Omega, \omega, M$ only).

### 8.4 Per episode: the witness (rule R)

The force law, the identity and B1 are also measured on every librating episode with its own period $P_e$. A
**witness** is an episode whose force law passes with its full margin while B1 does not fail and the identity is not
measured to fail, or passes thin with the identity and B1 passing on the same episode. With a witness the status tree
reads the three checks from the witness with the most turning points; otherwise the pooled measurement (under G1e,
pooled over the episodes before the jump). A witness next to an episode whose force law fails gives the cap
`episode_force_law_mixed` (≤ medium).

### 8.5 Potential and energy (R0, `check_mmr_potential`)

$V_E$ is fitted from $\dot\sigma^2 = 2E_w - 2V(\sigma)$ in each cycle (one $E$ per cycle, ≤ `SEPARATRIX_HARMONICS_MAX`
= 2 harmonics), $V_F$ from $c\,\dot a = -V'(\sigma)$.

- **R0σ ($\sigma$ only):** the period law $|\operatorname{median} T_{\text{obs}}/T_{\text{pred}} - 1| \le$
  `SEPARATRIX_PERIOD_TOL` (0.2) and coherent wells (phase coherence ≥ `SEPARATRIX_PHASE_COHERENCE`, 0.9). In a frame
  whose well moves (the multiplet members or the centre track) the coherence of the per-cycle wells is reported, not
  required: the frame carries the motion of the well, and the period law and the $R^2$ of $V_E$ carry the test.
- **$T_{\text{pred}}$ of a cycle:** for a two-body angle (not 1:1) with its planet saved, its period in the full averaged
  well $V = 3k^2\mu_P n^2 a\langle D\rangle(\sigma)$ at the cycle's middle time, through its first two TPs (nothing
  fitted), when the swing librates in that well; else that in $V_E$: the loop of the cycle's energy around the minimum
  whose basin holds its fastest sample (`pendulum.cycle_period`), not necessarily the deepest well. A TP in a moving well
  is moved to the well at the cycle's middle time with the kinetic energy of the well's drift.
- **R0 with $a$:** R0σ and $V_E \approx V_F$ (correlation ≥ `SEPARATRIX_VEVF_R_MIN`, 0.9) in ≥
  `SEPARATRIX_VEVF_FRACTION` (2/3) of the windows.
- **Order** (`zone.r0_checks`): both parts are not measured when $V_E$ explains less than `SEPARATRIX_VE_R2_MIN`
  (0.7) of $\dot\sigma^2$ or under `filter_too_slow`. Otherwise R0σ is decided first. If R0σ fails (or is not
  measured), R0 with $a$ takes the same value: a failed period law stays an objection even at small amplitude. Only
  when R0σ passes is R0 with $a$ not measured below `MMR_A_RESPONSE_MIN_AMP_DEG` (45°) without a turn (there the
  identity alone makes $V_E \approx V_F$); above it the window fraction decides.

### 8.6 Separatrix zone (flags only)

The zone is the band of energies the separatrix of a **measured, time-dependent pendulum** sweeps (Morbidelli 2002,
§9.4, pp. 224–225), not a fixed tolerance about $h = 1$ (`analysis/mmr/zone.py`).

*The model.* The multiplet force (§8.2) is one harmonic of $\sigma$ with a slowly turning complex amplitude,
$W(t) = \sum_j (A_j + iB_j)\,e^{ij(\varpi - \varpi_P)}$, so

$$
\ddot\sigma = -\bar\omega^2 m(t)\sin(\sigma - \sigma_s(t)), \qquad \bar\omega^2 m(t) = |c|\,|W(t)|,\qquad
\sigma_s = -\arg W\ (+\pi \text{ for } c > 0),
$$

with $\bar\omega$ the frequency of $V_E$. Without resolved members the strength follows the bounded oscillatory part
of $e$, $m \propto (e_{\text{osc}}/\bar e)^{|p|}$, with a fixed $\sigma_s$ (flag `zone_eq_only`).

*The edges.* Trial librations with $J_0/\langle J_s\rangle$ from `MMR_ZONE_J_MIN` (0.1) to 1 and trial revolutions
with $h_0$ from 1 to `MMR_ZONE_H_MAX` (2), both at step `MMR_ZONE_J_STEP` (0.025), start at `MMR_ZONE_PHASES` (16)
times spread over one modulation period (the record when it is shorter or unknown) and are integrated to the record's
end (DOP853, `MMR_ZONE_RTOL`). Then

- $J_{\text{lo}}$ = the smallest $J_0/\langle J_s\rangle$ that ever goes over the top ($|\sigma - \sigma_s| > \pi$);
- $h_{\text{hi}}$ = the largest $h_0$ that is ever trapped (its rate relative to the well changes sign);

each is 1 when no start reaches the separatrix. With `MMR_ZONE_METHOD=adiabatic`, $J_{\text{lo}} =
\omega_{\min}/\bar\omega$ and $h_{\text{hi}} = (\omega_{\max}/\bar\omega)^2$, $\omega = \bar\omega\sqrt m$; when the
modulation is slow ($\lambda = \nu_{\text{mod}}/\bar\omega <$ `MMR_ZONE_ADIABATIC_LAMBDA_MAX`, 0.1) and the two
$J_{\text{lo}}$ differ by more than `MMR_ZONE_ADIABATIC_TOL` (0.05), the flag `zone_numeric_mismatch` is set. No zone
is computed without a finite $\bar\omega > 0$, or when $2\pi/\bar\omega$ is shorter than the smoothing of $\sigma$
($V_E$ is then degenerate).

*The scatter of the energy.* From the finite cycle energies $h_j$,

$$
\sigma_h = \max\!\left(\sigma_{\text{floor}},\ \frac{1.4826\,\operatorname{MAD}(h_{j+1} - h_j)}{\sqrt2}\right),
$$

$\sigma_{\text{floor}}$ = `SEPARATRIX_SIGMA_FLOOR` (0.02), MAD about the median of the differences; with fewer than
two finite energies $\sigma_h$ is not measured and the floor is used. The same $\sigma_h$ enters the jump and
action-step tests (§6.9–6.10).

*The classes.* With the cycle's mean strength $m_j$, its action ratio $j_j = (J/J_s)\sqrt{m_j}$, its energy
$h_{\text{mean}} = h\,m_j$ in the mean normalisation, $z$ = `SEPARATRIX_SIGMA_MARGIN` (2) and

$$
\sigma_J = \frac{d(J/J_s)}{dh}\sqrt{m_j}\,\sigma_h, \qquad \frac{d(J/J_s)}{dh} = \frac{(V_{\max} - V_{\min})\,T(h)}{2\pi J_s}
\ \ \left(= \tfrac12\mathsf K(h) \text{ for one harmonic}\right),
$$

a libration cycle is `deep` when $j_j + z\sigma_J < J_{\text{lo}}$, a revolution `above` when
$h_{\text{mean}} - z\sigma_h > h_{\text{hi}}$, any other cycle `zone`; a cycle within one $\sigma$ of an edge sets
`zone_edge_cycle`. ($J/J_s$ of a two-harmonic $V$ is that of the well the cycle occupies, §2; $h_{\text{lo}}$, a property
of the zone, is read in the deepest well.) For a two-body angle with its
planet saved, an episode's cycle is instead read at its turning points in the averaged well,
$h_{TP} = (\langle D\rangle(\sigma_{TP}) - \langle D\rangle_{\min})/(\langle D\rangle_{\text{barrier}} - \langle D\rangle_{\min})$,
the barrier being the lower of the two tops around the well: `zone` when any $h_{TP} \ge h_{\text{lo}}$ (the energy of
$J_{\text{lo}}$ in $V_E$, over the barrier of $V_E$'s deepest well, `pendulum.h_in_barrier_units`; over $V_E$'s
global depth before W25, which a two-well $V_E$ makes smaller), `deep` when every $h_{TP}$ is measured and below it, otherwise the class above stays; no
$\sigma_h$ enters there. These are classes of the fitted model, not a measurement of chaos. The zone gives the flags
`libration_in_separatrix_zone`, `episodes_in_separatrix_zone`, `near_separatrix_energy`, the margins L2 and Z, and
the cap `episodes_in_separatrix_zone`. It changes no status, with one exception: when the episodes lie in the zone,
O1 is not read as an objection in the unmeasured branch (§12).

Crossings are read from $\sigma$ (`pendulum.regime_transitions`): an exit is a libration half-cycle followed by a slip
or a turn, a capture the reverse. The crossings get one mechanism (`sep_mechanism`), the first that applies:

| `sep_mechanism` | condition |
|---|---|
| `kick` | an energy-changing step of $a$ within one cycle period of a crossing |
| `trend` | the slope of $J/\langle J_s\rangle$ over the cycles differs from 0 by more than `SEPARATRIX_TREND_SIGMA` (3) standard errors and its total change exceeds $1 - J_{\text{lo}}$ |
| `pulsation` | the crossings favour the weak phase of $m(t)$: cyclic-shift p-value ≤ `SEPARATRIX_PERMUTATION_P_MAX` (0.05) |
| `layer` | $\lambda = \nu_{\text{mod}}/\bar\omega >$ `SEPARATRIX_LAYER_ADIABATICITY_MIN` (0.5): a fast modulation, a chaotic layer |
| `unexplained` | none of these |
| (empty) | no crossing |

An episodic angle whose mechanism is `unexplained` and whose episodes are not in the zone gets the cap
`transient_unexplained` (not for 1:1 under `MMR_COORBITAL_LIFTS_UNEXPLAINED`). The mechanism names a reading of the
crossing; it sets no status. The mark `exit_and_capture` (an episodic, non-co-orbital angle with ≥ 1 exit and
≥ 1 capture, no kick, its own force measured, no filter or sampling caveat) lifts the cap `transient_unexplained`
(`MMR_EXIT_AND_CAPTURE_LIFTS_UNEXPLAINED`).

### 8.7 Amplitude (B1, `check_mmr_a_amplitude`)

In the band $[P/2, 2P]$ (`MMR_B1_BAND_FACTOR` = 2), the measured $\sqrt2\,\operatorname{std}(a)$ against the required
$\sqrt2\,\operatorname{std}(\dot\sigma)/|c|$: a ratio below `MMR_B1_RATIO_MIN` (0.1) puts the oscillation in the
$\varpi$ part of the angle, not in $a$.

### 8.8 Kinematics (`check_mmr_kinematic`)

Without an episode: the slow $\sigma$ follows $p\varpi$ (correlation ≥ `MMR_KIN_CORR_MIN`, 0.9) while the slow
$\dot a$ does not answer $\sin\sigma$ ($R^2 \le$ `MMR_KIN_SLOW_R2_MAX`, 0.2): the $\lambda$ part cancels $p\dot\varpi$.
For an episode the cycle itself is judged: it is kinematic when its period matches $2\pi/|g - g_5|$ or the $2\omega$
cycle within `MMR_KIN_PERIOD_TOL` (0.1), the identity fails and the force law stays below `MMR_FORCE_R2_FAIL` in both
halves and under the multiplet law. With $z = e\,e^{i\varpi} = e_f e^{igt} + e_F e^{ig_5t}$,

$$
\dot\varpi = g_5 + (g - g_5)\,\frac{e_f^2 + e_f e_F\cos\theta}{e_f^2 + e_F^2 + 2e_f e_F\cos\theta} .
$$

`origin_detail`: `varpi_term`, `secular_cycle` or `planet_ripple`.

### 8.9 Kinematic drift of the slow centre

An episodic or librating angle whose episodes are a ripple on a drifting centre: the force law fails in both halves and
under the multiplet law, the largest episode swing is below `MMR_KIN_RIPPLE_SHARE_MAX` (0.5) × the range of the slow
centre, and the rate of the slow centre correlates (≥ `MMR_KIN_CORR_MIN`) with the rate of the secular part
$s = \sigma - \psi_\lambda$ low-passed at the same $W_c$. The verdict is `non_resonant/kinematic_drift` and is graded
as a drift; the reported word is the centre's (`MMR_KINEMATIC_DRIFT_WORD`): `drift`, or `circulation` when the
centre's largest pass reaches a revolution or its net shift reaches `LIBRATION_CIRCULATION_NET_REV`. A kinematic
libration whose decisive episode moves its centre by more than `MMR_KIN_DRIFT_SHIFT_RATIO` (1) × its mean amplitude is
decided the same way.

### 8.10 Slow libration (B5)

A drifting centre at rate $\nu_c$ could still be a libration slower than the record. On the analysis window of
length $T_w$ fit the centre by a line, $\sigma_c(t) = b + \nu_c t$, with residual standard deviation $s_c$; then

$$
\delta\nu_c = \frac{2\sqrt3\,s_c}{T_w},\qquad S = |\nu_c|\,T_w,\qquad
K_{\max} = \frac{|\nu_c|\,\delta a}{1 - \cos(\min(S, 2\pi)/2)},\qquad \omega_{0,\max} = \sqrt{|c|K_{\max}},
$$

$\delta a$ the range of the slow $a$ on the window: $K_{\max}$ is the largest force a libration through the observed
arc could have without moving $a$ by more than observed (worst case: the arc centred on the stable point). Excluded
when $|\nu_c| - \delta\nu_c \ge 2\omega_{0,\max}$ (full at ratio ≥ `MMR_B5_RATIO_FULL`, 2), when the centre completes a revolution at every smoothing over an interval
that overlaps the deciding episode, or by an owner. Not excluded: cap `slow_libration_not_excluded` (≤ medium).

## 9. Strength (F, `check_mmr_strength`) {#strength}

The decisive episode's small-amplitude period is $T_0 = \operatorname{median}_i P_i/F(A_i)$. The strength it requires is

$$
\mu_{\text{req}} = \frac13\left(\frac{2\pi}{|k|\,n\,T_0}\right)^2 = \frac13\left(\frac{P_{\text{orb}}}{|k|\,T_0}\right)^2 .
$$

($T_0$ already divides out $F(A)$; it must not be applied a second time.)

**Two-body, not 1:1: direct averaging.** The disturbing function, direct and indirect parts, is averaged over the fast
angle at fixed $\sigma$ with the planet on its eccentric inclined orbit (Gallardo 2006, eqs. 1–6, p. 30). From
Lagrange's equation $\ddot\sigma = -(3k^2/a^2)\,d\langle R\rangle/d\sigma$, so $\sigma$ librates about a minimum
$\sigma_c$ of $\langle R\rangle$, and

$$
\omega_0^2 = \frac{3k^2}{a^2}\langle R\rangle''(\sigma_c)
\quad\Rightarrow\quad
\mu_{\text{eff,th}} = \frac{a\,\langle R\rangle''(\sigma_c)}{GM_\odot} .
$$

$\sigma_c$ is the minimum whose basin holds the measured centre; the curvature, not the first harmonic, is used. An
evaluation counts only when the observed half-cycle nearest to it librates in that well (the swing contains
$\sigma_c$ and does not pass the global maximum); with no consistent evaluation F fails (`centre_outside_well`). The
theory is taken at `MMR_STRENGTH_SNAPSHOTS` (3) times in the decisive episode with the simultaneous elements. F passes
when the range of $\mu_{\text{req}}/\mu_{\text{th}}$ overlaps $[1/x, x]$, $x$ = `MMR_STRENGTH_CHECK_RATIO` (10); it fails
when the whole range lies outside; it is not measured without an episode, a stable well, a converged quadrature, or
with the body within `MMR_STRENGTH_MIN_DISTANCE_HILL` (3) Hill radii of the planet. `implausible_strength` means off
by ≥ `MMR_STRENGTH_IMPLAUSIBLE_RATIO` (10³).

**High order: the harmonic in extended precision** (`MMR_STRENGTH_MP = auto`). When the double-precision profile is
rounding (curvature below `MMR_STRENGTH_DOUBLE_FLOOR` × $|D_0|$), the theory is the resonant harmonic itself,
$C = \langle D\,e^{-i(k_P\lambda_P + kM)}\rangle$ over the two mean anomalies, so that
$\langle D\rangle(\sigma) = D_0 + 2|C|\cos(\sigma - \sigma_1) + \dots$, $\langle D\rangle'' = 2|C|$ at the minimum and
$\mu_{\text{eff,th}} = 2\mu_P a|C|$. $C$ is a 2-D trapezoidal sum in mpmath (`MMR_STRENGTH_MP_*`), checked against two
grids shifted by half a node.

**Three-body angles** keep the order of magnitude $\mu_i\mu_j e^q$ (Nesvorný & Morbidelli 1998, §3, pp. 3031–3032),
which misses the coefficient of the term (`str_method = eq_scale`); F is then a support margin only. A second-order
term may be enhanced by a small divisor up to $B^2$, $B = n/|\Delta|$, $\Delta$ the rate of a two-body angle with a
planet of the term (order ≤ 20, $k_1 \le 60$, $|\Delta|$ within $1.5\times2\pi/P$ somewhere in the $a$ range), counted at
least $2\pi/\min(P, P_{\text{cyc}})$, so $B \le n\min(P, P_{\text{cyc}})/2\pi$
(`strength.small_divisor_envelope`; `str_divisor_residual` = ratio $/B^2$). This envelope is a generous diagnostic,
not a bound.

**What the figure prints** (block A, "is $\sigma$ the own pendulum"): $T_{0,\text{th}} = 2\pi/(|k|n\sqrt{3\mu_{\text{th}}})$
against the observed $T_0$, the ratio $\mu_{\text{req}}/\mu_{\text{th}} = (T_{0,\text{th}}/T_0)^2$, and the
cosine-equivalent half-width $2\omega_0/|c| = \tfrac43 a\sqrt{3\mu_{\text{th}}}$ against the $a$-wave of B1 (a scale,
not a bound for a well that is not a cosine; the tolerance becomes $\sqrt x$ since the width goes as $\sqrt\mu$).

## 10. Subresonances (`subresonances.py`) {#subresonances}

Members $\sigma_s = \sigma_0 + \Delta_s$ of the same commensurability share the fast part and differ by a slow angle
$\Delta_s = m(\varpi_P - \varpi)$ (plus nodes): if member $s$ librates about $c_s$, the catalogue angle's centre drifts as
$c_s - \Delta_s(t)$. The scan runs on a trigger (a drift; a moving, modulated or unclosed centre; $a$ responding with a
rotating $\phi_0$; a kinematic or unexplained-drift verdict; another member leading by the averaged coefficients).

Members are every d'Alembert-valid split of the slow coefficients, up to the catalogue's largest order
(`MATRIX_2BODY_ORDER_MAX`, `MATRIX_3BODY_ORDER_MAX`); nodal members join when the inclination to the invariable plane is
≥ `MMR_SUBRESONANCE_NODAL_INC_MIN_DEG` (10°), and $\sigma \pm 2m\omega$ ($m \le$ `MMR_SUBRESONANCE_OMEGA_HARMONICS`, 2)
always. Nodal members are built wholly in the invariable plane (with $\sigma_0$'s own change of frame).

A member is accepted by its centre track (range ≤ `MMR_SUBRESONANCE_RANGE_RATIO` (0.5) × that of $\sigma_0$, no drift, no
revolution) and the tests: E′ (its d'Alembert factor ≥ `MMR_SUBRESONANCE_STRENGTH_REL_MIN` (10⁻³) of the strongest
term), L (the member librates itself in the scan window), A (a coherent force-law phase relative to the member, ≥ 0.8
and 0.2 above $\sigma_0$; not measured below a 45° swing), B (centre near 0 or $\pi$), C (unique; a tie goes to the lower
degree), E (a resonance at all). Test D compares the centre with the composite multiplet theory
$c_0(\beta) = \pi - \arg\sum_j C_j e^{ij\beta}$ (Morbidelli 2002, §9.3–9.4) and only raises the confidence. When several
members pass and only A could tell them apart, no member is accepted (cap `subresonance_undetermined`). An accepted
planar member is analysed as a row of its own (`added_by = subresonance_check`, `parent`).

`sub_top` lists the 10 best members (`MMR_SUBRESONANCE_TOP_N`) as `key:W/W0:E:N:A/W`: the range $W$ of the member's
slow centre against $W_0$ of the catalogue angle, $E = 1 - \operatorname{var}(c_s)/\operatorname{var}(c_0)$, $N$ = how
many of its slow cycles the window holds, $A/W$ the swing over $W$. The list decides nothing.

## 11. Owner: another resonance (`owner.py`) {#owner}

Two angles of the body, $\sigma = k\lambda + s$ and $\varphi_o = k_o\lambda + s_o$, satisfy the exact identity
$\sigma = (v/u)\varphi_o + \Psi/u$ with $u = k_o/d$, $v = k/d$, $d = \gcd$, and $\Psi = u\sigma - v\varphi_o$ free of the
body's $\lambda$. The identity holds whoever drives $a$, so a shared oscillation is necessary, not proof. B owns
$\sigma$'s oscillation on an interval when B's force explains $\dot a$ there:

- **Potential** (two-body B): $\dot a_B = \frac{2k_o}{na}\,\partial\langle R_B\rangle/\partial\varphi_o$
  (M&D 1999, eq. 6.145), averaged at fixed $\varphi_o$ with the planet's actual orbit, predicts the band-passed
  $\dot a$ with $R^2 \ge$ `MMR_OWNER_R2_MIN` (0.7) and scale $s \in [1/2, 2]$ (`MMR_OWNER_POTENTIAL_SCALE_MAX`).
- **Phase** (any B): the phase of $\dot a = R\sin(\varphi_o - \phi_0)$ is fixed (coherence $C_B \ge$
  `MMR_OWNER_PHI0_COH_MIN`, 0.8) and exceeds what $\sigma$'s own term could produce:
  $C_B \ge L_B$ + `MMR_OWNER_PHI0_COH_GAIN` (0.2) with $L_B = |\langle e^{i\Psi_w/v}\rangle|$.
- **$R^2$ gain** ≥ `MMR_OWNER_R2_GAIN` (0.2) over $\sigma$'s own law.
- **Identity** (three-body $\sigma$ only): $\sigma$ needs ≥ `MMR_OWNER_IDENTITY_RATIO_MIN` (10³) × $\mu_i\mu_j e^q$, and a
  two-body $\varphi$ follows it on the decisive window ($\sigma_f = c\varphi_f + a + bt$, $|c/(k_\sigma/k_\varphi) - 1| \le$
  `MMR_OWNER_IDENTITY_COEF_TOL`, residual ≤ `MMR_OWNER_IDENTITY_RESID_MAX` × the half-amplitude), $\varphi$ librates
  there with its own analysis, and $\varphi$'s directly averaged strength passes. Nothing is measured on $a$.

The owner takes $\sigma$'s episode only when its intervals cover ≥ `MMR_OWNER_COVER_MIN` (0.5) of it (else flag
`owner_outside_episode`, and the tree goes on). With the multiplet law (`MMR_OWNER_LAW = multiplet`) an owner whose
single phase drifts is fitted with all its members $\varphi_j = \varphi_o + j(\varpi - \varpi_P)$ and must then beat
$\sigma$'s own constant-coefficient $R^2$ by 0.2.

**Second capture.** Both resonances hold at once only if $\Psi$ is bounded. With $x = \Psi/u$ low-passed at
`MMR_OWNER_CAPTURE_LOWPASS_PERIODS` (3) $P$: when $x$ sweeps ≥ `MMR_OWNER_CAPTURE_SWEEP_DEG` (180°) and $\dot x^2$ is
not a function of $x$ ($R^2 < 0.7$ on `MMR_OWNER_CAPTURE_HARMONICS` harmonics), the second capture is excluded;
otherwise it is unresolved (cap `second_capture_unresolved`, ≤ medium).

A planet whose $\lambda$ was not saved enters the owner search (and nothing else) as the line
$\lambda_{j0} + n_jt$; candidates whose decision lies within the phase error this causes are `catalogue_ambiguous`, and
every such search carries the cap `catalogue_longitudes` (≤ medium). A 1:1 angle has no owner. MMR, secular and ZLK
verdicts are separate: no verdict demotes another.

## 12. The status tree {#status-tree}

`decide.decide_status`, the only place a status is set (G1c, G1e, G1l and a G2u union are read as episodic). The steps
in order:

| step | condition | status / reason |
|---|---|---|
| — | `undefined` | `invalid` / `alias`, `unphysical`, `undefined_phase`, `undersampled` |
| — | `insufficient_time` | `uncertain` / `baseline_needed` (action `extend`; `review` when the period already fits the record) |
| C | `circulation` | aliased: `non_resonant/alias` (unique branch, far) or `invalid/alias`; otherwise first rule 1 below (an accepted member with its own verdict gives its status); then with TPs and R0 failing `non_resonant/no_resonant_potential`; else `non_resonant/circulation` |
| 1 | a subresonance accepted | the member's status / `subresonance_librates`, origin `subresonance`, owner = the member (an `invalid` member → `uncertain`; a member without a verdict of its own → `non_resonant`); when the catalogue angle and the member both librate over the record, each row keeps its own verdict |
| 2 | an owner accepted covering ≥ `MMR_OWNER_COVER_MIN` of the decisive episode | `non_resonant/induced_libration`; cap `second_capture_unresolved` unless excluded |
| 3 | B1 or the kinematic test fails; or the kinematic drift of §8.9 (rule 3k2) | `non_resonant/kinematic_libration` or `kinematic_drift` (`uncertain/baseline_needed` when B5 finds an exact commensurability) |
| 4 | `drift` | `uncertain/drift_mechanism_unknown` |
| 5f | L4 contaminated by a forced line, and R0σ not failing | `uncertain/origin_unknown`, flag `force_law_contaminated`; a failed R0σ is an objection of its own: the tree goes on to 5s and the leaves below |
| 5s | a librating angle, L4 and R0 not passing, and the period needs ≥ 10³ × a computed strength (two-body direct or harmonic, not `centre_outside_well` unless `MMR_STRENGTH_FIRST_OUTSIDE_WELL`; or three-body $\mu_i\mu_j e^q$ with the small-divisor residual ≥ 10³) | `non_resonant/implausible_strength`, origin `unknown`, cap ≤ low |
| 5a | L4 fails, R0 does not pass | `uncertain/origin_unknown` (from near_separatrix: `non_resonant`) |
| 5b | L4 passes, R0 fails | `uncertain/origin_unknown`, flag `force_law_disagrees` |
| 5e | R0 fails, L4 not measured | as 5a |
| 5g | L4 not measured, R0 not passing, the identity failing as the status reads it and on an episode of its own, passing on none | as 5a |
| 5c, 5d → 6 | R0 passes or is not measured (and no 5g) | by the observed behaviour: libration → `resonant`, episodic → `transient`, near_separatrix → `near_separatrix`; origin `dynamic` (5c: cap `force_law_disagrees`) |
| 6u | a G2u union whose episodic reading reached `transient/dynamic_episode` with origin `dynamic` | `resonant/dynamic_libration` (`decide.promote_union`); the reported word is libration/`regime_change` in every case |

**L4 and R0 both not measured** (a librating angle, `decide.unmeasured_branch`). The first leaf that holds:

| leaf | condition | status |
|---|---|---|
| 5d → 6 | F by direct averaging (or the harmonic) passes | as above, origin `dynamic` |
| 5s | the period needs ≥ `MMR_STRENGTH_IMPLAUSIBLE_RATIO` × the directly averaged strength, or (three-body) × $\mu_i\mu_j e^q$ with the envelope still leaving that factor | `non_resonant/implausible_strength` |
| 5s* | the three-body ratio ≥ 10³ but the envelope can close the gap | `uncertain/origin_unknown` |
| 5o | a measured objection: O1 fails (not read when the episodes lie in the separatrix zone), or F fails | `uncertain/origin_unknown` |
| 5n | the decisive episode has fewer than `LIBRATION_CENTRE_MIN_ESTIMATES` independent cycles | `uncertain/origin_unknown` |
| 5u → 6 | nothing measured | the geometry's status with origin `unknown` and reason `geometric_libration` / `geometric_episode` (`separatrix_zone` stays); cap `geometry_only` (≤ low), or `strength_scale_only` (≤ medium) when F on the order-of-magnitude scale passes $[1/x, x]$ |

A `non_resonant` verdict next to a run one TP short of an episode gets `near_episode` (cap ≤ medium,
`LIBRATION_NEAR_EPISODE_TP_DEFICIT`).

## 13. Baseline and the next check {#baseline}

The record needs
$T_{\text{req}} = (N_{\text{TP}}/2)\,P + 2f_{\text{edge}}P_c$, $N_{\text{TP}}$ = `LIBRATION_MIN_TURNING_POINTS`,
$f_{\text{edge}}$ = `FILTER_EDGE_TRIM_CUTOFFS`; the physical estimate is $N$ = the round value ≥ $\max(T_{\text{req}},\ f_{\min}T)$, $f_{\min}$ =
`MMR_EXTEND_MIN_SPAN_FACTOR` (2), $T$ the record's length (`base_extend_to_years`). `extend_to_years` of a drift or
another uncertain verdict is $N$. For an `insufficient_time` angle it is the **next check**,
round_up$(\max(f_{\min}T,\ \min(N,\ f_{\text{next}}T)))$, $f_{\text{next}}$ = `MMR_NEXT_CHECK_SPAN_FACTOR` (4): a budget
after which the record is analysed again, not a promise that it is enough. $P$ is the time scale that decides
(`base_period_source`): for `few_cycles` twice the median TP spacing and the slow scales; for `arc` twice the longest
stretch without a turn of the slow centre, the $2\omega$ or $e$-cycle when the kinematic test fails, B5's bound; for
`stall` one revolution at the stall's mean rate and the force bound $P_{\text{lib,min}}$. When $T_{\text{req}} \le T$
time is not what is missing: `base_cause` names it and the action is `review`.

## 14. Confidence and action {#confidence}

Each applicable margin is `full`, `thin`, `fail`, `not_measured` or `not_applicable`, of class decisive (D: cannot
fail in the final verdict), support (S: each failure lowers the level) or cap.

| margin | what | full when |
|---|---|---|
| L1 (D) | TPs of the decisive episode at $P_c\times$ 0.5, 1, 2 | ≥ `MMR_L1_TP_FULL` (7) at every cutoff |
| L2 (D) | distance to the separatrix | no cycle in the zone; without a zone, amplitude ≤ `MMR_L2_AMP_FULL_DEG` (120°; thin to 160°) |
| L3 (S) | identity | pass (not applicable to a kinematic verdict) |
| L4 (D) | force law | $R^2 \ge 0.7$ in both halves |
| L5 (S) | one pendulum: $r$ = $T_0$ max/min over the amplitude regimes | $r \le b$ = `MMR_L5_RATIO_FULL` (1.15) × $\omega_{\max}/\omega_{\min}$ (1 without a measured modulation); thin to $r \le$ `MMR_L5_RATIO_FAIL` (2) × $b$, fail above; above $b$ only thin when the median $e$ > `MMR_L5_E_CAVEAT` (0.2) |
| L6 (D) | centre shift / mean amplitude | ≤ `MMR_L6_SHIFT_FULL` (0.5) |
| L7 (D) | coverage gaps of a libration | ≤ `MMR_L7_GAP_FULL` (0.7) $P$ |
| L8 (S) | step and filter | rotation per step < `MMR_L8_STEP_FULL_DEG` (10°), $G(P_{\text{cyc}}) \ge 0.94$ (fail < 0.5), no aliased line; not graded for a circulation ≥ `MMR_FAR_FROM_RES_FACTOR` (10) $a$-scales from $a_*$ |
| O1 (cap) | onset: the $a$-wave before / inside an episode of an episodic verdict that claims a capture | ratio < `MMR_O1_RATIO_THIN` (0.25); ≥ `MMR_O1_RATIO_FAIL` (0.5): cap `onset_failed` (≤ low). Not applicable to a forced wave, a kinematic libration, `implausible_strength`, or an induced libration whose owner librates over ≥ `MMR_O1_OWNER_COVER_MIN` of the stretch before |
| Z, R0 (D) | zone measured without caveats; potential with $a$ | pass |
| S, W (D) | subresonance scan; owner search | W: no candidate reaches $R^2$ `MMR_W_R2_NONE` (0.4), or an accepted owner with $R^2 \ge$ `MMR_W_R2_FULL` (0.85) and a reachable force; not applicable to a 1:1 angle |
| C1–C5 | circulation: centre pass ≥ `MMR_C1_REV_FULL` (10) rev, no TPs at any cutoff, smooth rate, the force against the rate (C4, not measured when no force law explains anything), the alias margin (C5) | |
| K1, B5, F, N | kinematic correlation ≥ `MMR_K1_FULL` (0.97); slow libration excluded ×`MMR_B5_RATIO_FULL` (2); strength; near-episode | |
| U:&lt;name&gt; | a decision within `CONFIDENCE_THIN_BAND` (10 %) of a convention | always thin |

**Rule 1's exception.** A catalogue row that takes the status of its accepted member (`subresonance_librates`) also
takes that member's confidence, raw level, thin and failed margins and caps; it keeps its own calibration cell and its
own `mrg_*` columns. Read the member's row (`owner`) to see why; do not recompute it from the catalogue row's margins.

**The rule** (`confidence.raw_level`): **certain**: every margin full, no cap; **high**: no failure, any number of
thin or unmeasured margins; **medium**: one failed support margin; **low**: two or more. When none of the physics
margins (L4, R0, Z, W, F) was measured the level is at most medium (`all_physics_not_measured`). Caps then lower it:

- ≤ high: `three_tp_episode`, `zone_not_measured`;
- ≤ medium: `exit_suspected`, `entry_suspected`, `one_return_cycle`, `transient_unexplained` (not for a 1:1 angle,
  `MMR_COORBITAL_LIFTS_UNEXPLAINED`), `force_law_disagrees`, `slow_libration_not_excluded`, `planets_missing`,
  `owner_search_skipped`, `catalogue_longitudes`, the subresonance caps (`nodal_subresonance_candidate`,
  `subresonance_tie`, `subresonance_t6_only`, `subresonance_theory_unconfirmed`, `subresonance_undetermined`),
  `jitter_low_e`, `l2_without_zone`, `episodes_in_separatrix_zone`, `near_separatrix_without_zone`,
  `episode_before_jump`, `episode_force_law_mixed`, `near_episode`, `short_period_aliased`,
  `second_capture_unresolved`, `strength_scale_only`, `one_cycle_energy_not_conserved` (a 1:1 verdict on one cycle
  whose co-orbital energy integral fails), `action_step_provisional`;
- ≤ low: `onset_failed`, `geometry_only`, `implausible_strength`.

`uncertain` and `invalid` have no level. Every cap at medium or low names in the figure the other reading and what
decides between the two (`verdict_text.CAP_ALTERNATIVES`).

**Calibration.** How often a level was right is reported next to it, per cell (status, angle_status, reason):
`cell_n_examples` author-labelled verdicts of the cell at high or certain, `cell_n_errors` of them wrong, and the Wilson
95 % interval of the accuracy (`cell_accuracy_low_95`, `cell_accuracy_high_95`; Wilson 1927), from
`resonances/data/confidence_cells.json` (written by `benchmark/tools/calibrate.py`). The counts never change a level:
the level is ordinal evidence, not a probability.

**Action.** invalid alias or undersampled → `resample` (`resample_step_yr`: the largest of `MMR_RESAMPLE_STEPS_YR`
within the safe steps); other invalid → `review`; uncertain → `extend` (`origin_unknown` → `review`); otherwise by the
level: certain and high `none`, medium `spot_check`, low `review`; a medium held there by `short_period_aliased` alone →
`resample` at `smp_fold_free_step_yr`.

## 15. Reading the combined figure {#combined-figure}

The one-page figure (`plots=['combined']`, [Plots](plots.md)) has time panels on the left and the decision on the
right.

| panel | shows |
|---|---|
| (a) | $\sigma$ unwrapped: osculating band, $\sigma_f$, the centre, turning points; accepted episodes filled and numbered (`#id wave` for a forced wave); rejected runs as grey strips with a letter (T over the top, F too few TPs, C centre moves, O no closed cycle); jumps of $a$; an action step; × an exit |
| (b) | $\sigma$ in a 360° window around the centre, tick labels in [0°, 360°); owner or subresonance angle dashed; (b′) a zoom on about five cycles when they are too short to see |
| (c) | $\dot\sigma/2\omega_0$ on a symlog axis, $\omega_0$ of the decisive episode (else the longest librating one, else theory); dashed: each episode's own $\pm2\omega_0$. Cosine-equivalent scales: ±1 bounds a libration only for a fixed cosine well of that $\omega_0$; a general well's separatrix speed is $\sqrt{2(E_{\text{sep}} - V_{\min})}$ (§2) |
| (d) | $a$: osculating, $a_f$, the slow $a$; $a_*$ and the catalogue $a_{\text{res}}$; the band $\pm\Delta a$ |
| (e) | $\dot a$ with the force law per window and its $R^2$ |
| (f), (g) | $e$ and $i$ |
| (h) | $h$ per cycle with the separatrix zone |
| (i), (j) | the centre with its turns and largest pass; the centres of the multiplet's members, subresonance and owner |
| (k)–(m) | phase portraits of the record and of the episodes; FAIR for a two-body angle |

Right: the plaque `STATUS · angle_status (subtype) · CONFIDENCE`; the explanation (what $\sigma$ does, the rule that
decided, the confidence and the action); block A (is $\sigma$ the own pendulum, §9) and block B (what makes the
cycles: owner, identity, forced wave, second capture); the steps of the tree (● decided here, ↓ passed on);
the margins and caps (✗ failed, ? not measured, ⚠ thin, ✓ full, – not measurable by design, · not applicable, ▼ cap);
the numbers. The secular and ZLK figures have the same right column.

## 16. Output {#output}

summary.csv carries the verdict columns above, every check, margin and cap, and the analysis' own columns by prefix:
`smp_*` sampling, `flt_*` filter, `geo_*` geometry, `rsp_*` response, `sep_*` energy and zone, `kin_*` kinematics,
`sub_*` subresonances, `own_*` owner, `str_*` strength, `base_*` baseline, `ctx_*` context. episodes.csv has one row
per run of turning points. See [Output data](output.md).
