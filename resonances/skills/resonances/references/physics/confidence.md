<!-- Generated from docs/physics/confidence.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Confidence

The status says *what* the record shows; the confidence says *how close the record came to showing something
else*. It is built from the comparisons the analysis made on the way to the status, not from a statistical model of
asteroid orbits. `uncertain` and `invalid` verdicts have no level: they claim nothing to be confident about.

## Margins

Every test that can support or undermine the status is graded as a **margin**:

| margin | meaning |
|---|---|
| `full` | the test passed with room to spare |
| `thin` | it passed, but close to its threshold, or it rests on too little (one cycle, one window) |
| `fail` | it did not pass |
| `not_measured` | it applies, but the record could not measure it; counts as `thin` (a check that could not be made is not a pass) |
| `not_applicable` | the test has no meaning for this verdict (e.g. the onset of a capture for a forced wave) |

Each margin belongs to a class:

- **D, decisive**: a test the status tree itself relied on. In a final verdict it cannot fail (the verdict would be
  a different one); it can only be full, thin or unmeasured.
- **S, support**: a test that may fail without changing the status; each failure lowers the level.
- **cap**: a test whose failure limits the level directly (below).

**Margin U.** Many thresholds are conventions rather than physics: chosen values such as $R^2 \ge 0.7$. When a
decision of the tree compared a value with a convention and the value lay within `CONFIDENCE_THIN_BAND` = 10 % of
it, $\lvert x/x_{\rm thr} - 1\rvert < 0.1$, that decision would flip under a slightly different convention. It becomes a
thin margin `U:<name of the comparison>`. Thresholds that follow from physics (one full turn, $h = 1$ at the
separatrix) never make a margin U. Which settings are conventions is marked "(convention)" in
`resonances/.env.dist`.

### The margins of a mean-motion resonance

| id | class | physical question | full when |
|---|---|---|---|
| L1 | D | Does the libration survive a different filter? | the decisive episode keeps ≥ `MMR_L1_TP_FULL` = 7 turning points at $P_c/2$, $P_c$ and $2P_c$ |
| L2 | D | Is it far from the separatrix? | no cycle lies in the separatrix zone; without a measured zone, amplitude ≤ `MMR_L2_AMP_FULL_DEG` = 120° (thin to `MMR_L2_AMP_THIN_DEG` = 160°) |
| L3 | S | Does $\dot\sigma \simeq c\,(a - a_*)$ hold? | the identity passes (not applicable to a kinematic verdict) |
| L4 | D | Does $\dot a$ follow the force law? | $R^2 \ge$ `MMR_FORCE_R2_FULL` = 0.7 in both halves of the record |
| L5 | S | Is it one pendulum? | $r$ = max/min of $T_0$ over the amplitude regimes ≤ $b$ = `MMR_L5_RATIO_FULL` = 1.15 × $\omega_{\max}/\omega_{\min}$ (the modulation of the force; 1 without one); thin to `MMR_L5_RATIO_FAIL` = 2 × $b$, failed above; above $b$ only thin when the median $e$ > `MMR_L5_E_CAVEAT` = 0.2 |
| L6 | D | Is the centre fixed? | the centre shifts by ≤ `MMR_L6_SHIFT_FULL` = 0.5 of the mean amplitude |
| L7 | D | Does the libration reach both ends of the record? | the gaps at the edges are ≤ `MMR_L7_GAP_FULL` = 0.7 periods |
| L8 | S | Do the step and the filter resolve the cycle? | $\sigma$ turns by < `MMR_L8_STEP_FULL_DEG` = 10° per step, the filter keeps ≥ `MMR_L8_GAIN_FULL` = 0.94 of the cycle (failed below 0.5), no aliased line in the band; not graded for a circulation ≥ `MMR_FAR_FROM_RES_FACTOR` = 10 $a$-scales from $a_*$ |
| O1 | cap | Is this a capture, not a wave that was already there? | the $a$-wave before the episode is < `MMR_O1_RATIO_THIN` = 0.25 of the one inside (≥ `MMR_O1_RATIO_FAIL` = 0.5: cap low) |
| Z | D | Is the separatrix zone measured? | measured without caveats |
| R0 | D | Is the energy from $a$ conserved in a potential of $\sigma$? | passes |
| S | D | Does a member of the multiplet own the libration? | see below |
| W | D | Does another resonance own it? | see below |
| C1–C5 | D/S | (circulation) Is it really a circulation? | see below |
| K1 | D | (kinematic) Does the angle follow the geometry? | rate correlation ≥ `MMR_K1_FULL` = 0.97 |
| B5 | S | Is a libration slower than the record excluded? | the centre's rate is ≥ `MMR_B5_RATIO_FULL` = 2 × the slowest rate a libration in the record's range of $a$ would allow |
| F | S | Is the required strength available? | required / theory within [1/x, x], x = `MMR_STRENGTH_CHECK_RATIO` = 10 (inverted for a kinematic verdict, below) |
| N | cap | Was the record one turn short of an episode? | no near-episode |

What each test measures is explained in [Is it this resonance?](response.md) and [What the angle does](geometry.md).

**S and W** support whichever origin the verdict names; they are not tests of absence only.

- S, an accepted member: full when the member's theoretical check confirms it and no tie between members was broken by
  order alone; otherwise thin. No accepted member: the closest member's centre range over $\sigma$'s,
  full above `MMR_S_CLOSE_RATIO` = 0.8 (a member nearly as still as $\sigma$ is a near miss: thin).
- W, an accepted owner: full when its median $R^2 \ge$ `MMR_W_R2_FULL` = 0.85 and its force is reachable (for the
  potential route: every window scale within $[1/1.5,\ 1.5]$, `MMR_OWNER_POTENTIAL_SCALE_FULL`); an owner covering less
  than `MMR_OWNER_COVER_MIN` of the episode is thin. No owner: full while the best candidate's $R^2$ stays below
  `MMR_W_R2_NONE` = 0.4. W is not measured when the phases cannot separate the best candidate or the search could not
  run, and not applicable to 1:1.

**C1–C5** grade a circulation:

- C1: the centre passes $\ge$ `MMR_C1_REV_FULL` = 10 turns (not for an alias);
- C2: full when no turning point is found at $P_c$, $P_c/2$ or $2P_c$; otherwise thin;
- C3: full when the rate never runs against its mean direction, its modulation is below `MMR_C3_MODULATION_MAX` = 0.3
  and pattern B is absent; otherwise thin. The modulation is $\sqrt{b_c^2 + b_s^2}/\lvert b_0\rvert$ from the fit
  $\dot\sigma = b_0 + b_c\cos\sigma + b_s\sin\sigma$;
- C4: $r = \langle\lvert\dot\sigma\rvert\rangle / (2\sqrt{\lvert c\rvert K})$, $K$ the fitted force amplitude: the rate
  against the fastest a pendulum of that force could turn at the separatrix. Full for $r \ge$ `MMR_C4_RATIO_FULL` = 1.5,
  thin for $1 < r < 1.5$, failed for $r \le 1$. A failure counts as not measured when no force law explains $\dot a$
  anywhere (best $R^2 <$ `MMR_FORCE_R2_FAIL` = 0.4). Not for an alias;
- C5, resolved step: full when the measured rate and the rate expected from the mean motions have the same sign and
  agree within `MMR_C5_RATE_TOL` = 0.2; otherwise thin. On an alias: full only when the far ratio reaches
  `MMR_FAR_FROM_RES_FACTOR` = 10 and the distance from $\nu\Delta t$ to the nearest half-turn exceeds
  `MMR_C5_BRANCH_RESID_FACTOR` = 2 × the largest unwrap residual of a step; otherwise thin.

**F for a kinematic verdict.** For a verdict that claims this resonance's own dynamics, F grades the agreement of
required and theory. For a `non_resonant` verdict of kinematic origin with a directly computed strength (direct
averaging or the extended-precision harmonic), F reads in the inverted sense: **full when the strength check fails**
(the force available cannot make the observed cycle, which supports the kinematic reading), not measured otherwise.
The order-of-magnitude $e^q$ scale gives no such margin. The column `check_mmr_strength` keeps its ordinary meaning.

### Secular and Lidov-Kozai margins

N (turning points of the decisive episode; thin at the minimum of three), S (the centre stationary over enough
independent cycles), F (the libration period ≥ `FILTER_MIN_LIBRATION_CUTOFFS` = 2 × $P_c$), P (a circulation's
centre passes a full turn at every smoothing) and, for Lidov-Kozai, G (the free-$\omega$ gate decided with its
number of cycles clear of `FREE_GATE_MIN_CYCLES` = 3 by more than the thin band).

## From margins to a level

The raw level, `confidence_raw`:

| level | rule |
|---|---|
| **certain** | every applicable margin full: no thin, no unmeasured and no U margin |
| **high** | no failure; any number of thin (or unmeasured) margins |
| **medium** | exactly one failed support margin |
| **low** | two or more failed support margins |

The shown level, `confidence`, is the raw level lowered by every **cap** the verdict carries:
$\text{confidence} = \min(\text{raw},\ \text{ceilings of the caps})$.

A catalogue row that takes the status of its accepted member (`subresonance_librates`) takes the member's
confidence too, with its raw level, thin and failed margins and caps; its `mrg_*` columns and calibration cell stay
its own. Read the member's row (`owner`) to see why.

One cap is general: when **none of the physics** of the angle's own force could be measured (for an MMR: the force
law L4, the potential R0, the zone Z, the owner search W, and the strength F by direct averaging), the level is at
most medium (`all_physics_not_measured`). The identity and the response of $a$ (L3, B1, L5) are not in this list:
they show that $a$ carries the oscillation, which is just as true of a libration induced by another resonance.

## Caps and the other reading

A cap names a specific way the record could be read differently. Every cap that brings the level down to medium
or low names, in the right column of the [combined figure](../plots.md), the verdict the record might hold instead
and the panel that decides between the two.

| cap | ceiling | the other reading |
|---|---|---|
| `three_tp_episode` | high | no libration yet: three turning points are one cycle |
| `zone_not_measured` | high | the cycles may sit in the separatrix zone |
| `exit_suspected`, `entry_suspected` | medium | the libration goes on past the record's edge, or $\sigma$ leaves (enters) only here |
| `one_return_cycle` | medium | no libration yet: one cycle closed only by its phase return (two turning points) |
| `transient_unexplained` | medium | near_separatrix: $\sigma$ wanders across the separatrix without a capture or an escape |
| `episodes_in_separatrix_zone` | medium | the edges of the librating stretches are crossings of the zone, not a capture or an escape |
| `force_law_disagrees` | medium | uncertain: the potential holds while $\dot a$ does not follow $\sin\sigma$ |
| `episode_force_law_mixed` | medium | only one episode librates in this resonance; the others carry another force |
| `episode_before_jump` | medium | the episodes belong to a regime that ended with the jump of $a$ |
| `action_step_provisional` | medium | one libration over the whole window: the step in amplitude is a large modulation, not two energy levels |
| `slow_libration_not_excluded` | medium | a slower libration of $\sigma$ about a moving centre |
| `near_episode` | medium | transient: one more turning point would make a librating episode |
| `near_separatrix_without_zone`, `l2_without_zone` | medium | a circulation, or a libration close to the separatrix |
| `second_capture_unresolved` | medium | $\sigma$ is held by its own term as well: both resonances at once |
| `owner_search_skipped`, `catalogue_longitudes` | medium | another resonance drives $a$ (rerun with the search on, the planets saved) |
| `planets_missing`, `nodal_subresonance_candidate`, `subresonance_tie`, `subresonance_t6_only`, `subresonance_theory_unconfirmed`, `subresonance_undetermined` | medium | a member of the multiplet holds the centre |
| `jitter_low_e` | medium | invalid: $\varpi$ is undefined at small $e$ |
| `short_period_aliased` | medium | the turning points are partly the output step's: short-period terms folded into $\sigma_f$ and $a_f$ |
| `one_cycle_energy_not_conserved` | medium | (1:1) the one cycle may be a passage, not motion in the averaged well |
| `strength_scale_only` | medium | uncertain: only the order-of-magnitude strength agrees, the force is not computed |
| `all_physics_not_measured` | medium | uncertain: nothing measured shows this angle's own force |
| `geometry_only` | low | uncertain: only the shape of $\sigma$ says libration |
| `implausible_strength` | low | a libration whose force the first-order average misses |
| `onset_failed` | low | non_resonant: the cycles continue an $a$-wave that began before the libration |

## Calibration

How often a level was right is reported next to it, never mixed into it. The verdicts are grouped into **cells**
(status, angle_status, reason). For each cell, the benchmark ([Benchmark](https://smirik.github.io/resonances/benchmark/)) counts the
author-labelled verdicts at `high` or `certain`, $n$, and the wrong ones among them, $n_{\rm err}$. With
$\hat p = (n - n_{\rm err})/n$ and $z = 1.96$, the Wilson (1927) 95 % interval of the accuracy is

$$
\frac{\hat p + \dfrac{z^2}{2n} \pm z\sqrt{\dfrac{\hat p(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}} .
$$

The columns `cell_n_examples`, `cell_n_errors`, `cell_accuracy_low_95` and `cell_accuracy_high_95` carry them
(from `resonances/data/confidence_cells.json`, written by `benchmark/tools/calibrate.py`). With tens of labels per
cell the interval is wide. The level is therefore **ordinal evidence**: `certain` is stronger evidence than
`high`, which is stronger than `medium`; it is not a probability that the status is right.

## Actions

Each verdict ends with an `action`, the cheapest step that would make it stronger:

| action | when |
|---|---|
| `none` | `certain` or `high` |
| `spot_check` | `medium`: look at the figure |
| `review` | `low`; `uncertain / origin_unknown`; `invalid` other than an alias or an undersampled step; a record whose period already fits it while it is still too short to decide (more time cures nothing) |
| `extend` | other `uncertain` verdicts: integrate longer, to `extend_to_years` |
| `resample` or `review` | an `uncertain` 1:1 stall that stays long enough in the quasi-satellite well ([G1q](pendulum.md#co-orbitals-11)) with $P_{QS} < P_c$: the filter erased a possible quasi-satellite libration, so more time at the same step cures nothing. `resample` when the record's step cannot hold the quasi-satellite cutoff, else `review` |
| `resample` | `invalid / alias` or `undersampled`, or a level held down only by `short_period_aliased`: integrate again with the output step `resample_step_yr` |

**How long to extend.** A libration needs at least $N_{\rm TP}/2$ periods plus the edges the filter trims:

$$
T_{\rm req} = \frac{N_{\rm TP}}{2}\,P + 2\,f_{\rm trim}\,P_c ,
$$

with $N_{\rm TP}$ = `LIBRATION_MIN_TURNING_POINTS` = 3, $f_{\rm trim}$ = `FILTER_EDGE_TRIM_CUTOFFS` = 1 and $P$ the
time scale that decides: the measured cycle, or for a too-short record the slowest lower bound its shape allows
(twice the spacing of its turning points, a slow arc of the centre, the rate of a stall). The physical estimate is
$N$, a round value $\ge \max(T_{\rm req},\ f_{\min} T)$, $f_{\min}$ = `MMR_EXTEND_MIN_SPAN_FACTOR` = 2, $T$ the record
length (column `base_extend_to_years`). For `insufficient_time` the action asks only for the **next check**,

$$
T_{\rm next} = \mathrm{round\ up}\big(\max(f_{\min}T,\ \min(N,\ f_{\rm next}T))\big),
$$

with $f_{\rm next}$ = `MMR_NEXT_CHECK_SPAN_FACTOR` = 4: a budget after which the record is analysed again, not a promise that it is enough.
A drift or another `uncertain` verdict asks for $N$ itself. When $T_{\rm req} \le T$, time is not what is missing:
`base_cause` names what is, and the action is `review`.

??? info "Which time scale decides (`base_period_source`)"
    For `few_cycles`: twice the median spacing of the turning points, and the slow scales. For `arc`: twice the
    longest stretch without a turn of the slow centre, the $2\omega$ or $e$-cycle when the kinematic test fails, and
    B5's bound. For `stall`: one revolution at the stall's mean rate, and the force bound $P_\text{lib,min}$
    ([The lower bound of an arc or a stall](geometry.md#the-centre-and-circulation)). The column `base_period_source`
    names the one that decided.

**Which step to resample at.** For an alias or a coarse step, `resample_step_yr` is the largest value of
`MMR_RESAMPLE_STEPS_YR` that keeps one alias branch over the range of $a$, samples the libration at least
`MMR_RESAMPLE_STEPS_PER_CYCLE` = 20 times per cycle and lets the filter exist ($P_c > 2\Delta t$); `review` when
the record's step already meets these bounds. When the cap `short_period_aliased` alone holds the level down, the
step is `smp_fold_free_step_yr`: the largest grid step below the record's at which no short period folds.

**Secular and Lidov-Kozai.** The same table, and every extension is a next check $T_{\rm next}$ with an $N$ by
cause (column `base_extend_cause`; module `analysis/secular/extend.py`). No status changes:

| cause | $N$ | action |
|---|---|---|
| `baseline` (`baseline_needed`: drift, `insufficient_time`) | $T_{\rm req}$ with $P = \max(P_{\rm floor}, P_{\rm cyc})$, $P_c$ the decisive cutoff (secular with no cycle, $P_c$ = 100 kyr: 500 kyr) | `extend` |
| `gate_cycles` (gate `short_window`) | $n_{\min} P_{\rm cyc}$ + what the free elements lose of the record ($T - T_{\rm free}$), $n_{\min}$ = `FREE_GATE_MIN_CYCLES` | `extend` |
| `gate_resolution` (gate `planetary_residual`) | a free rate $d_{\min}$″/yr from a planetary one, inside the Rayleigh width $1\,296\,000/T_{\rm free}$: $m \cdot 1\,296\,000/d_{\min} + (T - T_{\rm free})$, $m$ = `ZLK_EXTEND_RESOLUTION_MARGIN` = 1.25 (373136 at 100 kyr, $d$ = 3.5: 400 kyr; 1981 at 500 kyr, $d$ = 1.9: 1 Myr) | `extend` |
| `gate_unresolvable` | that baseline above `ZLK_EXTEND_MAX_YR` = 100 Myr, or $d_{\min} = 0$ | `review` |
| `gate_undefined_phase`, `gate_unavailable`, `unusable_series` | a longer record does not define the free phase, build missing free elements or repair a series | `review` |

The geometric $T_{\rm req}$ joins every gate cause (the larger $N$). Columns as for an MMR: `base_required_kyr`,
`base_period_yr`, `base_period_source` (`cycle` or `period_floor`), `base_extend_to_years`.
