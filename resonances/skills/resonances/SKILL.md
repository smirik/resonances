---
name: resonances
description: >-
  Set up, run and interpret resonance studies with the `resonances` Python package: mean-motion (two-body, three-body,
  retrograde, co-orbital), secular and Lidov-Kozai (ZLK) resonances of asteroids and other small bodies. Covers
  choosing check/find/Simulation, data sources (NASA Horizons, AstDyS, MPC), integrator, span and output step (the
  dry-run), clones, batch runs and restore, and reading the verdicts in summary.csv (status, angle_status, reason,
  libration_origin, confidence, caps, action) and the combined figure. Use it whenever the user mentions the
  resonances package, a resonant angle or its libration/circulation, keys such as 4J-2S-1, 3J-1, 1E-1, 1J-3r0, g-g6,
  z1 or LK, a summary.csv or combined figure from such a run, or asks whether an asteroid is in a resonance, even if
  the package is not named.
---

# resonances: resonance studies of small bodies

The package integrates a body with the planets (REBOUND), builds the resonant angle σ of every resonance it checks
and gives each angle a **verdict**: what σ does over the record and whether this resonance makes it do so. Your job
with it is usually one of three: plan a run that *can* give a verdict, run it, and report what the verdict does and
does not say.

`references/` is a copy of the package documentation generated from its `docs/` (same paths, links between the pages
work; images and the omitted pages link to https://smirik.github.io/resonances/). This file holds the workflow and the
judgment; formulas, every column, every setting and every threshold are in the references. Look things up there
(grep is fine) instead of guessing a column name, a default or a threshold.

## 1. Choose the entry point

| goal | call | read |
|---|---|---|
| given bodies, given resonances | `resonances.check(463, '4J-2S-1', **kwargs)`; CLI `resonances check --asteroids=463 --resonances=4J-2S-1 --integration-years=100000` | quick-start.md, core.md |
| search the resonances near each body | `resonances.find([463, 490], planets=['Jupiter', 'Saturn'], type='mmr')` (MMRs within `sigma2` = 0.1 au two-body, `sigma3` = 0.02 au three-body; `type=None` also adds secular candidates from AstDyS proper frequencies, and `LK`) | core.md, examples.md |
| full control (several resonance types, own elements) | `sim = resonances.Simulation(...)`, `sim.create_solar_system()`, `sim.add_body(id_or_elements, ['4J-2S-1', 'g-g6'], name=...)`, `sim.run()` | core.md |
| candidates only, no integration | `resonances.find_mmrs(a, planets=[...])`; `resonances secular-candidates g-g6` | matrix.md, secular.md |
| select bodies from the MPC catalog | `resonances.mpcorb.query(a=(2.3, 2.45), U=(None, 2), numbered=True)` → pass the DataFrame to `find`/`check` with `source='mpc'`; CLI `resonances mpc-search ... --ids` | config.md (Minor Planet Center) |

Keys: two-body `3J-1` (full key `3J-1+0-2`), three-body `4J-2S-1` (`4J-2S-1+0+0-1`), co-orbital `1E-1`, retrograde
`1J-3r0` (p, planet, q, k; only for i > 90°), secular `g-g6` (ν6), `s-s6` (ν16), `g-g6+s-s6` (z1), Lidov-Kozai `LK`.
The output files use the full key. Notation: references/matrix.md, references/core.md.

Data source (`source=`): `nasa` (default; Horizons elements at `date`, which defaults to **today's midnight**),
`astdys` and `mpc` (the catalog epoch is the default date; `mpc_lookup='file_then_api'` takes an object missing from
the catalog, e.g. an alias or one outside an `MPC_CATALOG` subset, from MPC's get-orb). The planets always come from Horizons at `date`, so a
run needs the network once per date. Pin `date=` whenever the result must be reproducible or a batch run may need a
resume. A bare number sent to Horizons is an asteroid number. Your own elements: heliocentric J2000 ecliptic, angles in
radians; an `epoch` that differs from `date` is refused.

## 2. Plan the record: span, output step, integrator

A verdict can only be as good as the record. Plan these three before integrating anything long.

**Span.** A libration is credited only after ≥ 3 turning points (one full cycle) inside the filter's edges; a record
sure to hold them whatever the phase needs T_req = 1.5 P + 2 P_c (references/physics/confidence.md, "How long to
extend"). Aim for several libration periods,
not one. Typical records of the package's own examples (references/physics/gallery.md):

| case | span, output step, integrator |
|---|---|
| main-belt two- and three-body MMR (463 Lola: P ≈ 10 kyr) | 100 kyr, 1 yr, SABA(10,6,4) |
| slow or weak MMR (385458 9N-17) | 300 kyr: at 100 kyr it is `uncertain / insufficient_time` |
| TNO, outer Centaur (a > 30 au) | ≥ 300 kyr (outer-Solar-System classification uses 10 Myr): beyond a planet's 1:2 the cutoff is ≥ 1.5 P_orb (`MMR_FILTER_ORBIT_FACTOR`, `flt_cutoff_binding = orbit`), several hundred years, and librations are slow |
| NEA, planet-crosser MMR | 20–50 kyr, 0.25–5 yr, ias15 |
| Earth co-orbital `1E-1` | 1 kyr, 0.05 yr, ias15 with `solar_system_moon=True` |
| secular (ν6, z1) | 2–10 Myr, step ~500 yr |
| Lidov-Kozai | 500 kyr, 10–100 yr |

**Output step.** The default (span/100, 15.9 yr for 100 kyr) folds the short periods of every main-belt angle: the
verdicts then carry the flag `short_period_aliased`, and a check the folded periods spoil is withdrawn (R-ALIAS-A,
`rsp_short_period_unmeasured`: confidence ≤ medium, `action=resample`). For any run with MMRs use
`output_step='auto'`: the dry-run walks down to a step that folds no short period (about 1 yr in the main belt, 2.5 yr
for Hildas and Jupiter Trojans, 0.5 yr for NEAs). `auto` is not defined for purely secular/ZLK runs: set the step.

**Integrator.** `SABA(10,6,4)` with `dt = 1.0` (REBOUND units, 58 days) suits main-belt orbits. Orbits that approach a
planet (NEAs, crossers, co-orbitals) need `integrator='ias15'`; the Moon as a separate body (`solar_system_moon=True`)
only with ias15, for Earth encounters. Highly eccentric ZLK bodies may need `dt = 0.5` with SABA (one was ejected at
dt = 1). Time units: 2π REBOUND units = 1 yr (`tmax`, `dt`, `sim.times`); `integration_years` and `output_step` are
in years; years = `sim.times / (2 * np.pi)`.

**Dry-run before a long run** (seconds, no integration at the static level; references/dry-run.md):

```bash
resonances dry-run --asteroids=463 --resonances=4J-2S-1 --years=100000 --step=auto --pilot
```

or `report = sim.dry_run(level='pilot'); report.summary(); report.apply()`. Read the alias verdict and `action` per
row (`none`, `predictive_unwrap`, `resample`), `integrator_recommended`, the short-period warnings and the cost
(`cpu_hours`, `disk_GB`, `ram_per_worker_GB`). Secular and ZLK angles get advisory rows (`report.long_period`): the
baseline the analysis needs (secular 500 kyr, ZLK ~9.4 kyr plus the gate's 3 ZLK times at the defaults) and a warning
when the span is shorter. `sim.run()` runs the static dry-run itself and logs its warnings.

**Cost.** Integration is cheap; the analysis is not (1–2 CPU-min per body × resonance for a 1-Myr record). Before
anything longer than a few minutes, tell the user the estimate, then run it in the background with `nice`.

## 3. Run

- Scripts that start worker processes need `if __name__ == '__main__':` on macOS and Windows: batch runs (more than
  `BATCH_THRESHOLD` = 100 bodies) and every run with clones.
- `name=` sets the output folder `SAVE_PATH/<name>` (`cache/<name>`); `find`/`check` use `resonance_find`.
- `save` and `plot` select rows by the verdict (default `review`: everything but a confident plain circulation, i.e.
  `nonzero` plus the review queue, `review_required`: non_resonant at medium/low confidence). `nonzero` keeps the old,
  smaller selection for mass runs (the log says how many more rows `review` keeps). Use `save='all'` when
  the rows may be analyzed again later: only saved series can be re-analyzed. `save_planets='analysis'` (default)
  must stay on for the owner search and the strength; without the planets those checks cap the confidence.
- Clones (virtual asteroids from the orbit's covariance; references/clones.md): `resonances.check(463, '4J-2S-1',
  clones=100)`, CLI `--clones=100`, or `resonances.orbit_solution(463, source='astdys').sample(n=100, seed=0)` and
  `sim.add_ensemble(ensemble, resonances=[...])`. 100 clones give a share to ±10 %, 400 to ±5 %, 1000 to ±3 % (Wilson
  95 %). `clones_source='auto'` follows the data source (nasa → JPL SBDB); for NEAs `neodys` (the fit's epoch);
  MPC refuses U ≥ 3. A numbered main-belt asteroid has σ_a ~ 10⁻⁹ au: its clones differ only through chaos.
- Resume an interrupted batch run, or any run with clones: the same script, bodies, settings, `save_path` and
  **date**; a changed hash raises `ValueError` naming what changed. A run with clones skips its finished clone batches
  and its finished ordinary part (checkpoint `clones_state/`; references/clones.md, references/serialization.md).
- Restore a finished run (written by this version; earlier formats are not read):
  `resonances.SimulationSerializer.restore('<save_path>/simulation.json')` reads the verdicts from summary.csv, and with clones `sim.ensembles` and `sim.clone_summary`; `recompute_librations=True` analyses the
  saved series again without integrating (`current_settings=True`: with today's `.env`); clones keep their stored
  verdicts, only the nominal is reclassified (its clones.csv columns and summary.csv rows are rewritten together).
- Settings: argument of `Simulation`/`check`/`find` (lower case) > `.env` in the working directory > environment >
  `resonances/.env.dist` (references/config.md).

## 4. Read the verdict

Where: `summary.csv` at the top of `save_path` (one row per body × resonance, every row); in Python
`body.librations[key]` (a `Verdict`) and `body.statuses[key]`; `sim.data_manager.get_simulation_summary(sim.bodies)`.
The numbers behind an MMR verdict are in `mmr_numbers.csv`, the episodes in `episodes.csv`, clone shares in
`clones.csv`, the figure in `combined/<confidence>/<status>/<body>-<key>-combined.png`. Every column:
references/output.md; every value: references/physics/overview.md and references/physics/status-tree.md.

| field | read it as |
|---|---|
| `status` | the answer: `resonant`, `transient`, `near_separatrix`, `non_resonant`, `uncertain`, `invalid` |
| `angle_status` | what σ does over the whole record (`libration`, `episodic`, `near_separatrix`, `circulation`, `drift`, `insufficient_time`, `undefined`); not the answer |
| `final_regime` | how the record ends after librating episodes (`circulation`, `libration` = captured at the end), or `arc`/`stall` when σ slows down after circulating (flag `extend_time`) |
| `reason`, `libration_origin` | why; `dynamic` = this resonance's force was measured; `kinematic`, `induced`, `subresonance`, `unknown` otherwise |
| `confidence` | `certain` > `high` > `medium` > `low`: ordinal evidence, not a probability; empty for `uncertain`/`invalid`. Minimal policy: from the evidence alone (below), `confidence_basis` says on what |
| `caps`, `flags` | `caps` is empty: every cap of the W28 tree is a flag and changes neither status nor level (references/physics/confidence.md) |
| `policy`, `v2_verdict` | `minimal`; `v2_verdict` is what the W28 tree (the policy's input) said (`status/confidence/reason/action/angle_status`), for comparison only |
| `veto`, `n_cycles`, `n_counted_episodes`, `counted_episode_ids`, `candidate_episode_ids` | minimal policy: the measured physics that made a bounded angle non_resonant (`owner`, `kinematic`, `forced_wave`, `strength`, `zlk_kinematic`, `zlk_free_e`; '' none), the complete cycles N behind the level, the counted episodes and the candidates (episodes.csv `counted`, `candidate`) |
| `lib_time_yr`, `lib_share` | the time in libration: years of the record inside the counted episodes (a run covering a record edge counts to that edge) and their share of the record; empty without a counted episode |
| `action` | `none`, `spot_check` (look at the figure), `review`, `extend` (to `extend_to_years`), `resample` (at `resample_step_yr`) |
| `rules_fired` | the named rules that acted on the row (a cut, a join, a cap, an outcome), even when the status stayed; each has a row id in references/libration-analysis.md (`frame_join` → G2q). A flag list, never a status |
| `review_required`, `review_reasons` | the review queue: rows a human should look at (medium/low level, a spot_check/review action, origin unknown, a 1:1 coverage-audit flag) and why (`confidence:`, `action:`, `origin_unknown`, `veto_thin`, `uncovered_bounded_run`, `floor_above_own_loop`); no status reads it |

**The minimal policy** (the W28 tree's verdict is kept in `v2_verdict`;
references/physics/status-tree.md, references/physics/confidence.md). The status is the geometry of σ_f: bounded over
the whole record → resonant, bounded stretches between circulation → transient, ambiguous → near_separatrix or
uncertain, never bounded → non_resonant. A bounded angle becomes non_resonant **only** on a measured contradiction
(`veto`); a check that was not measured, or is thin, is a flag (`gate_unavailable`, `measured_objection`,
`veto_needs_length`, `one_coorbital_leg`, …), never a status. The level: `certain` needs N ≥ 3 complete cycles (3 observed inside
the decisive episode), a clear amplitude margin, every needed objection measured and none found, and a filter-stable
record; `high` N ≥ 2; `medium` 1.5 ≤ N < 2 or a needed objection not measured; `low` N < 1.5 or a measured objection on
a libration (`measured_objection`). An MMR libration whose own force is not shown (no force law, energy law or strength
passes: `F=none` in `confidence_basis`) is at most `low`: the angle is bounded, the mechanism is not shown. A measured
veto is `high`; a non_resonant row is `medium` when a candidate episode was rejected (it is plotted in review mode) or
the veto was thin (`veto_thin`), or a secular/ZLK record too short for one cycle ends in circulation by less than
1.1 turns of σ_f (`pol_level_inputs` `end_travel:thin`: perhaps part of a slow libration). `extend` only for an unfinished libration at the record's end, a deciding period longer than T/1.5, or a
needed veto not measured for lack of length (`veto_needs_length`, clause `C`) whose record `pol_veto_need_yr` is within
`MMR_NEXT_CHECK_SPAN_FACTOR` × T (beyond it: `review`); a record that ends in circulation is decided now. So more rows are resonant/transient at `low`/`medium` than under v2
(about 31 benchmark rows v2 left uncertain): the `review` and `nonzero` save/plot modes select them, and a
`medium` row is a "look at the figure" row, not a weak yes.

**What a verdict claims.** It describes the saved interval, at the saved step, in the run's model (planets, integrator,
initial elements, date). A longer span, a finer step or another epoch can change it; chaotic bodies change verdict
with weeks of start epoch. The thresholds are conventions tested on a benchmark labeled by eye.

Distinctions that matter when you report:

- **Librating is not resonant.** `non_resonant` with `angle_status = libration` (or `episodic`) means σ is bounded
  by something else: `kinematic_libration`/`kinematic_drift` (the perihelion term or a secular cycle, a does not carry
  σ), `induced_libration` (another resonance drives a; `owner` names it), `implausible_strength`.
  A kinematic libration reaches `high` only when its cause is measured (margin K1e: the episode's σ follows pϖ,
  `kin_k1_episode` ≥ 0.9, a flat a with B1 < `MMR_KIN_B1_CLEAR` and a secular period; rule `kinematic_high`), else
  `medium` (`all_physics_not_measured`). `kin_swing_share` ≈ 1 says pϖ makes the swing; `kin_pred_swing_deg` (what
  the free and forced e make pϖ swing) against the measured swing says whether the size fits too.
- **A slower libration of the center** is stored at its own scale (`slow_*`, episodes.csv `scale` = `slow`): its period,
  cycles, swing, the identity on that scale and a cause with a level (`slow_cause`, `slow_cause_term`,
  `slow_cause_level` identified / possible / unknown). A secular cause (`secular_2omega`, `secular_g_gj`) is a
  modulation of the perihelion term, not a second resonance; `unknown` means a slow libration of another resonance is
  not excluded, and `slow_origin` = `dynamic_candidate` (the identity holds there) changes no status. Rule K3s makes a
  circulation whose slow swing is a secular cycle with a failed identity `kinematic_libration` (37087 3J+8S-3).
- **Roles of the other resonance.** `owner` holds only an owner (librates, `cause_role` owner) or a driver (circulates,
  its forced wave makes the cycles) that covers the deciding episode (≥ 0.5) and, when it librates, can make its
  cycles (P2). An accepted owner that does not (`owner_outside_episode`, `owner_psi_too_fast`) is in `neighbours`
  (`KEY t0–t1 before|during|after`): the body sits in it before or after, not in two resonances at once, and the row's
  own episodes decide; episodes.csv marks each induced episode (`libration_origin` = `induced`). A librating member of
  the same multiplet is `resonant_argument` (rule C: one resonance), a kinematic cause is in `cause_term` (`-4varpi`).
  Count a body once per `commensurability` (`9M-16`), not per catalog row.
- **Capture or a wave that was already there?** The cap `onset_failed` (≤ low) means σ rides a coherent a-wave at
  its period that began before the episode, continues into it in phase and is not this resonance's own force
  (`ep_onset_coherence_r2`, `ep_onset_phase_gap_deg`, `ep_onset_own_law_r2` vs `ep_onset_own_law_r2_episode`). A large
  a-wave before a capture alone is not an objection: near the separatrix the same force makes it.
- **High needs a full physical support.** An own-dynamics resonant/transient verdict whose force law (or 1:1 law),
  potential and strength all pass only thin is at most `medium` (`no_full_physics_support`).
- **A co-orbital stay beyond its measured episode is inferred.** episodes.csv `core_t0_yr`/`core_t1_yr` are the
  measured turning points that coverage and the status read (of an episode continued past cycles that miss the well,
  rule `coorbital_well_core`, only the well-holding ones; the rest is marked `unheld@t0:t1`: a 1:1 episode ends only
  where σ crosses the planet or L3); `residence_t0_yr`/`residence_t1_yr` are the inferred stay
  around them, until the orbit-averaged angle sweeps a whole turn (`residence_bounded_by_*` = `circulation`) or the
  record ends (`edge`: no exit, the continuation is undetermined; extend the run to know). A change of well inside one
  episode is a sub-regime (`subregimes`, `subregime_checks`; summary `orbit_type` `compound: horseshoe → tadpole L4`,
  the figure's header `orbit: horseshoe → tadpole L4`), not a second episode; an amplitude change inside one well and
  a kick of a there are marks (`marks`). On the figure the residence is the lighter fill hatched `--` labeled
  `inferred`, the sub-regimes are brackets over (a), the kicks ▼ on (d); the text says `measured … ; stays … (inferred,
  not measured)`.
- **One co-orbital leg is not a cycle.** A 1:1 angle without an episode is read on its single legs between passages
  (`rec_rule` G4c; `geo_leg_read`, in mmr_numbers.csv `geo_leg_outcome`/`geo_leg_discrepancy`): a clean leg with the
  local 1:1 law, its energy and two reflecting turns is `transient` capped `one_coorbital_leg` (≤ medium); one
  failing test is `uncertain / one_coorbital_leg` (action review: look at the figure, a longer record rarely helps);
  a clean leg cut by the record edge asks to extend; two kicks keep `non_resonant` with `near_episode` (≤ medium).
  For another MMR a lone half-swing between passages that follows its own one-harmonic law is
  `uncertain / one_half_libration` (G4h). Do not count these as captures in a statistic.
- **Force law from theory only.** `rsp_force_law` = `direct_force` with the cap `force_law_theory_only` (≤ medium):
  the fitted sine failed (a lattice line in ȧ, e.g. the conjunction cycle), but the term's own averaged force with
  the planet's real mass explains the slow ȧ at the libration scale (`rsp_dforce_R2`, scale `rsp_dforce_scale_*`
  near 1, `geo_period_large_yr`). It confirms the force, not the libration; `rsp_dforce_reason` says why it was
  refused elsewhere (`no_independent_windows`: the episode is too short for two disjoint windows).
- **Slow libration at a shorter cutoff.** `rules_fired` with `scale_ladder` and the cap `slow_scale_provisional`
  (≤ medium): the center scale left fewer than two slow periods in its window, so the verdict was read at the shortest
  cutoff that keeps its slow extrema (`geo_scale_ladder_cutoff_yr`, `geo_scale_ladder_n_tp` against
  `_n_tp_centre`), and neither the force law nor R0 was measured there: only the theoretical strength supports it.
  Ask for a record of ≥ 5.25 P (TNOs: ≥ 300 kyr) before trusting the level.
- **Both resonances at once?** An induced row is never "in both": the flag `second_capture_unresolved` (≤ medium)
  means it was not ruled out. It is ruled out when x = Ψ/u sweeps ≥ 180° without an energy integral, or when ẍ follows
  the 2ω / ϖ − ϖ_j cycle and not sin x (`own_second_capture_route` = `secular_forcing`, rule `second_capture_f3`).
  A slow wobble of the center is described in `own_x_*` (period, half-amplitude, `own_x_cause` with
  `own_x_cause_level` identified/possible): read it as the secular cycle, not as a slow three-body libration.
  Only `own_second_capture` = `confirmed` says "in both" (origin `joint`, `owner` = the other resonance, role
  `joint`): x is bounded, conserves an energy, sits in a well of its own at a non-secular period and closes ≥ 2
  cycles (`own_x_cycles`, `own_x_secular_offset`); the row then keeps its own status. `one_cycle` makes it uncertain
  (extend). "Both angles librate" alone is not "in both resonances".
- **Resonant with origin unknown.** `geometric_libration` (resonant) and `geometric_episode` (transient) have origin
  `unknown`: the shape of σ says libration but no force of this resonance could be measured. Say "librates
  geometrically; the dynamical origin was not established", not "confirmed resonant".
- **`subresonance_librates`.** The catalog angle takes the status of a member of its multiplet (σ ± 2mω, nodal
  terms) that holds the center. When counting bodies, count each commensurability once: deduplicate on
  (`name`, `parent` or `resonance`).
- **`transient`** = librating episodes that do not cover the record (capture, escape, jump of a). Check
  `final_regime` before saying "left the resonance" or "was captured".
- **`uncertain`** is not `non_resonant`. `baseline_needed` (angle `insufficient_time`) and `drift_mechanism_unknown`:
  extend to `extend_to_years`, which is the budget of the next check, not a promise. A small loop (< 45°) whose
  identity fails, at the turn of an unfinished slow swing, is set aside (episodes.csv `forced_wave` =
  `unconfirmed_ripple`): the slow swing decides, so expect `insufficient_time`, not an episode. Two slow turns whose
  half-swing obeys this resonance's pendulum (identity, a swing, period: `geo_slow_swing_*`, rule `slow_swing_in_well`)
  are `insufficient_time` / `arc`, not `drift`, and ask for three slow periods. `origin_unknown`: σ librates but
  a measurement contradicts this resonance's pendulum (leaves 5a, 5b, 5e's R0, 5g, 5o: more time may not help, review
  the figure), or nothing measured it (5n one independent cycle, 5x the e^q strength scale out of range: a diagnostic,
  not an objection). When the missing measurement lacks only length (the force law has fewer than two windows, or
  one cycle, and the last episode reaches the record's end) the action is `extend` (rule `extend_length_limited`);
  a swing below 45° alone keeps `review`.
- **`extend_time` is a flag, not an action.** It marks a record whose edge a longer integration would resolve, in
  `extend_direction` (`forward`, `backward`, `both`) to `extend_to_years`: σ slows to an arc or a stall after
  circulating at the end (`final_regime` `arc`/`stall`; a `non_resonant` circulation there is capped
  `record_ends_unresolved`, ≤ medium), the same at the start, one reflection cut by an edge, a clean 1:1 edge leg.
  It changes no status or level and keeps the row's action (often `review` or `none`). In a mass run, collect these
  rows for a second, longer integration (backward ones from the start epoch into the past) rather than reporting
  them as settled; do not count them as captures.
- **`invalid`** says nothing about the resonance: `alias`/`undersampled` → resample at `resample_step_yr`;
  `unphysical` → the orbit left the ellipse (encounter, ejection: check the integrator); `undefined_phase` → ϖ is
  undefined at e ≈ 0. A non_resonant `alias` (one branch, far from a*) is a real circulation seen through the step.
- **Secular episodes.** A secular libration makes few cycles in a record (published and measured periods run from
  ~0.25 Myr for ν6 to ~12 Myr; no period is predicted per body), so read the figure's (a) and the filtered Charlier
  portraits (k), (l) before trusting a short episode. σ_f running a turn or more after the last turning point is an
  observed exit: `transient`, `final_regime` circulation. A small ripple on a slower swing, or a short stall of a
  circulation, is not an episode (secular angles only): the slower scale decides, often `uncertain` + `extend` or a
  slow `resonant` at `medium` + `extend` (101703 g−2g6+g7: one 12-Myr cycle in 20 Myr). Integrate 40 Myr or more
  before reporting such rows.
- **Lidov-Kozai, minimal policy.** The gate can only veto: a measured `kinematic` outcome (`veto` `zlk_kinematic`), or
  free e below forced e (`zlk_free_e`), gives `non_resonant` (code −2, exactly these rows). A gate without a verdict
  (short window, planetary residual, undefined phase, co-orbital of Mercury to Mars) or a disabled one leaves the
  geometric status (resonant/transient) at ≤ `medium` with the flag `gate_unavailable` / `gate_disabled`; −9 is then
  only an `uncertain` row. An episodic ω is judged inside its librating episodes when its free ω winds over the record
  (it circulates between them): confirmed there (`zlk_gate_detail` = `episodes`) when the free ω stays put in every
  episode (`ZLK_GATE_EPISODES`). A confirmed and a geometric ZLK librator share the status code: **split every ZLK
  statistic by `check_zlk_free_omega`** (`pass` = confirmed, `not_measured` = no verdict). The text below is the W28
  tree (`v2_verdict`).
- **Lidov-Kozai (the W28 tree, `v2_verdict`).** A libration of the osculating ω is not enough; the free-ω gate decides. `non_resonant /
  kinematic_libration` is a **measurement** (the free ω winds: geometry); `uncertain / gate_unavailable` is the
  **absence** of one (`extend_to_years` for a short window or a residual near a planetary rate; `review` for an
  undefined free phase or missing free elements). Never merge them in a statistic (legacy codes −2 and −9). A co-orbital
  of Mercury to Mars gets no gate verdict (`zlk_gate_detail` = `coorbital_unfitted_forcing`: its planet's modes are not
  in the split), so it is uncertain/gate_unavailable even when ω is episodic (rule `zlk_coorbital_episodic`); Jupiter Trojans are covered (forced vector g5 + g6), and 591986 stays kinematic. Any other
  episodic ω without a gate verdict is `transient` only when e and i trade at constant H inside its deciding episode
  (`zlk_ei_episode_coupling` = `zlk`; needs ≥ 3 periods in the episode), else uncertain/gate_unavailable with rule
  `zlk_gate_unconfirmed`. Read `zlk_ei_episode_reason`
  (`short_episode`: a longer record with a longer episode may settle it). ω, i, Ω (and the
  free elements) are measured from the invariable plane (`ZLK_ANGLE_FRAME`, column `zlk_angle_frame`): from the
  ecliptic a nearly coplanar orbit (i_inv ≲ 1.6°) shows a false bounded ω; rule `zlk_invariable_frame` marks rows where
  the two frames disagree by a turn.
  Columns that change no status (references/free-elements.md, "Beyond the gate"): `zlk_ei_coupling` (e–i of the
  whole record at the decisive period, every row: `zlk` antiphase at constant H, `forced_beat` in phase; coupling is not libration — a
  circulating ω can show it), `zlk_forcing_*` (which forced mode holds a kinematic libration, or `not_attributed` and
  why; `zlk_attribution_scope` `episodes` for an episodic row demoted inside its episodes), `zlk_node_lock` (a flag:
  the mode the node is locked to, or empty with `zlk_node_lock_reason`, `unresolved` when no full cycle of Ω − ϖ_j fits
  the record), `zlk_centre_*` (the center's slow line and its origin, e.g. `forced_beat:g6-g5`, `forced_direction`).
  The combined figure names the cause of a kinematic libration in its third sentence and adds panel (a′) (ω_f with
  the forced direction and the mode against the node) only for such rows.
- **Clones.** `clones.csv` gives the share of each status with its Wilson interval: a probability under the
  provider's covariance, the model and the span, not of the classifier's errors. The nominal has no special weight.
  The shares are over the bound draws only (`n_classified`); draws that are no bound orbit are `n_draw_invalid`,
  never integrated, and `invalid` is only the classifier's verdict on a clone. With `cloud_usable` False
  (`f_draw_invalid` > `CLONES_MAX_INVALID`, 5 %) the covariance is no model of the orbit (a few-day arc): the row has
  no shares; do not read its counts as evidence.
- `confirmed_ei` (= `zlk_ei_coupling == zlk`), `zlk_coherent` (the Welch line) and the diagnostic plots (recurrence,
  cycles, FAIR) never change a status.
- `status_code` is the legacy integer; use the string columns.

## 5. Traps

- **Default output step.** 15.9 yr flags every main-belt MMR `short_period_aliased`; where the folded periods withdraw
  a check (`rsp_short_period_unmeasured`) the level is ≤ medium and the action `resample` at `smp_fold_free_step_yr`.
  Prevent it with `output_step='auto'`.
- **Too short a record** gives `uncertain / baseline_needed`, not "not resonant". Compare the span with the period
  (`geo_period_cycle_yr`, `base_period_yr` in mmr_numbers.csv).
- **Retrograde orbits (i > 90°).** Use `RetrogradeTwoBody` keys (`1J-3r0`); `find` gives such bodies only retrograde
  candidates. Never build an angle of such an orbit from REBOUND's `orbit.l` or `orbit.pomega` (they are −λ*, −ϖ* and
  jump when i crosses 90°); their `longitude`/`varpi` data columns are in other conventions; no FAIR plot.
- **Secular angles.** The package compares the body's ϖ and Ω with the planets' secular mode longitudes g_j t + φ_j
  (`secular_angle_mode='proper'`). Do not rebuild a secular angle from the planets' osculating elements in
  `data-planet-*.csv`: from the ecliptic no planet's node circulates and most perihelia do not follow their mode.
  With the default linear phases the inner modes' phases stray by tens of degrees over 10 Myr (g4 30°, s4 81°): a
  swing within that (`sec_phase_swing_ratio` ≤ 1) is `insufficient_time` / `phase_unresolved`, action review; rerun
  with `SECULAR_PLANET_PHASES=measured` and `SAVE_PLANETS=all` (a record ≥ 5 Myr) instead of extending.
- **Stale MMR catalog.** A log warning "MMR catalog ... was built with other settings" means the cached
  `cache/mmr-*.csv` (paths relative to the working directory) has other `MATRIX_*` limits or planets. Delete it (or
  `TwoBodyMatrix.load(reload=True)`) before trusting `find`.
- **The ends of the record are not classified.** About one filter cutoff at each end is trimmed: an episode under
  way at the epoch whose turns fall there is not counted, and the status describes the record, not the state at the
  epoch. For the present state put the epoch at least one cutoff inside the record (integrate backward too).
- **1:1 audit flags.** `uncovered_bounded_run` (a bounded stretch of the averaged angle no counted episode covers,
  `aud_unc_*`) and `floor_above_own_loop` (a floor above the loop through the turns of the run it rejects) change no
  status: open the figure (hatched strip in (a)) before you report such a row as non-resonant.
- **Today's date.** A `nasa` run without `date=` starts at today's midnight: two days give two runs, and a resume on
  another day fails its hash.
- **MPC epochs.** Rows at another epoch than the run's date are integrated to it (`MPC_EPOCH_POLICY=propagate`, with
  a warning) or refused (`strict`). The catalog is refreshed only by `resonances mpc-update`.
- **Planets not saved** (`save_planets='none'`) or the owner search off: caps `catalogue_longitudes`,
  `owner_search_skipped`, `planets_missing` (≤ medium): rerun with the planets saved.

## 6. Read the combined figure

Each plotted row gets one page. The figures are explained panel by panel in references/reading-figures.md, with
examples in references/physics/gallery.md and the files in references/plots.md.

The right column is the verdict, in ten blocks that come in the same order for every family:

1. **Header**: the status plaque, the level as four dots (`no level` for uncertain and invalid), and one action chip:
   `no action needed`, `glance at (b), (i)` (the panels of whatever holds the level), `review figure + numbers`,
   `extend to 200 kyr` or `resample to 0.5 yr`.
2. **Subtitle**: the resonance and the subtype in words (`its own libration…`, `induced: borrowed from X`, `kinematic
   (forced) libration…`), then the type and the step that decided.
3. **Key facts**: σ, the period and amplitude, the episodes (◆ marks the deciding one), the resonance or driver, the
   orbit, the radial overlaps and the run.
4. **Why**: at most five sentences, each test with its number, the last on what limits the level; then one line,
   `Needed: …` (uncertain, invalid), `Would change the verdict: …`, `Next check: …` or `A longer record cannot make it
   resonant: …`. This line comes only from computed conditions, such as the extend length or a veto that needs a
   longer record.
5. **Confidence**: `Confidence: high — not certain because …`, then one pill per input of the level
   (`pol_level_inputs`):
   - ✓ the input is in its best state;
   - – it is limited but does not bind;
   - ⚠ it holds the level;
   - ? it is not measured and holds the level;
   - ✗ a measured objection that the level reads (`pol_objection_check`).
6. **Checks**: `Q · Can the data show it?`, then the steps 0–6 of the tree as questions, with each margin nested under
   its step and a small gray rule ID. ◆ marks the one deciding step. When the minimal policy decided
   (`pol_decided_by` ≠ `tree`), ◆ is on a `P · …` line and the tree's leaf reads `overruled`. The steps not reached
   are one gray line. The three-body e^q strength scale is a gray `– scale only`, never an objection.
7. **Evidence**: `is this X's own pendulum?` and `what makes the cycles?` (owner search, identity, `both resonances at
   once?`: `excluded`, `not ruled out` or confirmed). Secular and ZLK rows ask `what holds ω?`, answered with the
   forced vector, the center wave, the node lock and the e–i coupling.
8. **Caveats**: one sentence per flag the reader must know (`verdict_vocab.FLAG_CAVEATS`), or `none`. A failed check
   that the level does not read is said here, with the reason it is not read.
9. **Numbers**: the DECISIVE lines with rule IDs, then labeled groups. Numbers are printed wherever they are needed;
   there are no pointers.
10. **Footer**: the codes (status code, leaf · angle rule, reason/origin/veto, flags), the settings hash and the docs.
    The full tree's verdict before the minimal policy is only in the `v2_verdict` column, for comparison: never a
    second verdict.

For a paper, `plot_options={'combined': {'figure': 'panels'}}` draws the panels only (no header, legend, right
column or empty slot); the caption must then carry the verdict. After a restore,
`sim.data_manager.mmr_combined_figure(body, resonance, sim, options=...)` returns that figure unsaved.

A long column makes the page taller. For the full text, use
`verdict_column.column_text(verdict_column_mmr.figure_column(analysis, rows, settings))`, or
`verdict_column_long.figure_column(analysis, row, free_elements, settings)` for secular and ZLK rows.

The header line above the left panels names the role of another resonance: `owner: …`; `driver: … (circulates, P …):
forced wave`; a neighbor `… librates after the episode (…), outside it`; `resonant argument: …`; `kinematic: −4ϖ
term (…)`. Induced episodes are hatched `\\\\` on (a).

For an MMR the left panels are:
- (a) σ raw and filtered, with turning points and numbered episodes;
- (b) σ wrapped;
- (c) σ̇/2ω₀: ±1 is the separatrix speed of a cosine well, and each episode's own limits are dashed;
- (d) a, with a* and the resonance's half-width;
- (e) ȧ with the fitted force law and its R²;
- (f), (g) e and i;
- (h) the energy of each cycle against the separatrix h = 1;
- (i) the slow center (hatched `slow #id`: the slow scale);
- (j) the centers of other multiplet members and of an owner (a flat line librates), plus pϖ with its secular period
  when the slow cause is a secular cycle. For a kinematic libration, (j) is σ = λ part + pϖ on the scale of the swing:
  σ follows pϖ and the λ part stays flat.

Labels to know:
- On (a), accepted runs that are not a libration of this angle are labeled `#id wave` (a forced wave), `#id forced
  (strength)` (a 5s row: no driver identified) or `#id ripple`.
- A 1:1 episode in parts has brackets per sub-regime and a hatched `inferred` stay, which is inferred, not measured.
- Without FAIR (m), a dark-orange line says why.
- (a) of both figures fills only the counted episodes (`#id`; a counted 1:1 leg is `leg`), with the candidates faint
  and dashed (`cand`). episodes.csv still lists every run.

For secular and ZLK rows:
- (d) is a (with the planet's a and the co-orbital zone's edges for a co-orbital);
- (e), (f) are e and i;
- (g) is the rate fit (secular), or the drift of the free and the osculating ω in turns (ZLK: a free ω that winds
  makes the libration kinematic).

Their FREQUENCIES table lists the body's rates, the planetary modes (g5, g6, s6 always) and the beats in ″/yr with
their periods, so you can name a period seen on σ(t) (e.g. 591986: g − g5 6.11 kyr, the kinematic ω cycle; g6 − g5
54 kyr).

If you can view images, open the figure of every `medium`/`low` verdict before reporting it.

## 7. Before you report a status

1. Read the whole row: `status`, `angle_status`, `final_regime`, `reason`, `libration_origin`, `confidence`, `caps`,
   `flags`, `action`, `extend_to_years`, `extend_direction`, `resample_step_yr`.
2. `invalid`: report that the record could not be measured and how to repair it; never as non-resonant.
3. Is the span long enough (several periods)? If not, or `uncertain`, or caps `three_tp_episode`/`one_return_cycle`:
   "not decided yet; integrate to N yr".
4. Is the step fine enough? `short_period_aliased` with `rsp_short_period_unmeasured`: the sampling withdrew a check and
   holds the level down; propose the resample step.
5. Name the origin: `dynamic` vs `unknown` (geometric), `kinematic`, `induced` (with `owner`), `subresonance`.
   `subfloor_pass` in `rules_fired`: a two-body libration faster than the 150-yr floor, accepted because its own
   averaged potential, center and force law confirm it (G1f; `geo_subfloor_*`).
6. `certain`/`high`: state it. `medium`: look at the figure and name what limits it (the ⚠/?/✗ pills of the
   Confidence block, `pol_level_inputs`; the flags). `low`: a measured objection (`pol_objection_check`), N < 1.5 or
   an MMR own force not shown: review.
7. ZLK: give the gate's outcome; keep kinematic and gate_unavailable apart (minimal: split by `check_zlk_free_omega`).
8. Statistics over many bodies: deduplicate multiplet rows, keep −2/−9 style outcomes apart, do not pool runs with
   different settings (`settings_hash`) or steps.
9. State what the verdict covers: span, output step, integrator, planets, date and source.

A compact report line:

```text
463 4J-2S-1+0+0-1: resonant (libration, dynamic_libration, origin dynamic), confidence high, no caps.
Record 100 kyr at 1 yr, SABA(10,6,4), Horizons 2026-01-01. Next: none.
```

## References

| file | read when |
|---|---|
| quick-start.md, core.md, examples.md, install.md | writing the code of a run; installation |
| dry-run.md, physics/signal.md | choosing the output step and integrator; alias, short periods, cutoff |
| physics/overview.md, physics/status-tree.md | the meaning of every status, reason and origin; the order of the decision |
| physics/confidence.md | margins, caps and the other reading of each cap, actions, how long to extend, which step to resample at |
| physics/gallery.md | one example figure of every status, with its record (span, step, integrator) |
| physics/pendulum.md, physics/geometry.md, physics/response.md | the physics of a single check (turning points, episodes, force law, identity, owner, strength) |
| physics/secular-zlk.md, secular.md, lidov-kozai.md, free-elements.md, coherence.md | secular and Lidov-Kozai angles, the free-ω gate, e–i coupling, attribution |
| matrix.md | resonance notation, catalogs, retrograde resonances, the catalog cache |
| clones.md | clones: sources, epochs, how many, clones.csv |
| output.md | every file and column of a run |
| plots.md | figure kinds and options, redrawing after a run |
| reading-figures.md | how to read every figure, panel by panel, and what to look at first for each status |
| serialization.md | simulation.json, restore, resuming a batch run |
| config.md | every setting and threshold with its default |
| libration-analysis.md | the rule index: what a rule ID (G2a, L4, 5s, …) means, its settings, the page that explains it |
