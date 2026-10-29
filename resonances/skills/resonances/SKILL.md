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
judgement; formulas, every column, every setting and every threshold are in the references. Look things up there
(grep is fine) instead of guessing a column name, a default or a threshold.

## 1. Choose the entry point

| goal | call | read |
|---|---|---|
| given bodies, given resonances | `resonances.check(463, '4J-2S-1', **kwargs)`; CLI `resonances check --asteroids=463 --resonances=4J-2S-1 --integration-years=100000` | quick-start.md, core.md |
| search the resonances near each body | `resonances.find([463, 490], planets=['Jupiter', 'Saturn'], type='mmr')` (MMRs within `sigma2` = 0.1 au two-body, `sigma3` = 0.02 au three-body; `type=None` also adds secular candidates from AstDyS proper frequencies, and `LK`) | core.md, examples.md |
| full control (several resonance types, own elements) | `sim = resonances.Simulation(...)`, `sim.create_solar_system()`, `sim.add_body(id_or_elements, ['4J-2S-1', 'g-g6'], name=...)`, `sim.run()` | core.md |
| candidates only, no integration | `resonances.find_mmrs(a, planets=[...])`; `resonances secular-candidates g-g6` | matrix.md, secular.md |
| select bodies from the MPC catalogue | `resonances.mpcorb.query(a=(2.3, 2.45), U=(None, 2), numbered=True)` → pass the DataFrame to `find`/`check` with `source='mpc'`; CLI `resonances mpc-search ... --ids` | config.md (Minor Planet Center) |

Keys: two-body `3J-1` (full key `3J-1+0-2`), three-body `4J-2S-1` (`4J-2S-1+0+0-1`), co-orbital `1E-1`, retrograde
`1J-3r0` (p, planet, q, k; only for i > 90°), secular `g-g6` (ν6), `s-s6` (ν16), `g-g6+s-s6` (z1), Lidov-Kozai `LK`.
The output files use the full key. Notation: references/matrix.md, references/core.md.

Data source (`source=`): `nasa` (default; Horizons elements at `date`, which defaults to **today's midnight**),
`astdys` and `mpc` (the catalogue epoch is the default date; `mpc_lookup='file_then_api'` takes an object missing from
the catalogue, e.g. an alias or one outside an `MPC_CATALOG` subset, from MPC's get-orb). The planets always come from Horizons at `date`, so a
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
| NEA, planet-crosser MMR | 20–50 kyr, 0.25–5 yr, ias15 |
| Earth co-orbital `1E-1` | 1 kyr, 0.05 yr, ias15 with `solar_system_moon=True` |
| secular (ν6, z1) | 2–10 Myr, step ~500 yr |
| Lidov-Kozai | 500 kyr, 10–100 yr |

**Output step.** The default (span/100, 15.9 yr for 100 kyr) folds the short periods of every main-belt angle: the
verdicts then carry the cap `short_period_aliased` (confidence ≤ medium). For any run with MMRs use
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
- `save` and `plot` select rows by status (default `nonzero`: everything but a plain circulation). Use `save='all'` when
  the rows may be analysed again later: only saved series can be re-analysed. `save_planets='analysis'` (default)
  must stay on for the owner search and the strength; without the planets those checks cap the confidence.
- Clones (virtual asteroids from the orbit's covariance; references/clones.md): `resonances.check(463, '4J-2S-1',
  clones=100)`, CLI `--clones=100`, or `resonances.orbit_solution(463, source='astdys').sample(n=100, seed=0)` and
  `sim.add_ensemble(ensemble, resonances=[...])`. 100 clones give a share to ±10 %, 400 to ±5 %, 1000 to ±3 % (Wilson
  95 %). `clones_source='auto'` follows the data source (nasa → JPL SBDB); for NEAs `neodys` (the fit's epoch);
  MPC refuses U ≥ 3. A numbered main-belt asteroid has σ_a ~ 10⁻⁹ au: its clones differ only through chaos.
- Resume an interrupted batch run, or any run with clones: the same script, bodies, settings, `save_path` and
  **date**; a changed hash raises `ValueError` naming what changed. A run with clones skips its finished clone batches
  and its finished ordinary part (checkpoint `clones_state/`; references/clones.md, references/serialization.md).
- Restore a finished run: `resonances.SimulationSerializer.restore('<save_path>/simulation.json')` reads the verdicts
  from summary.csv, and with clones `sim.ensembles` and `sim.clone_summary`; `recompute_librations=True` analyses the
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
| `final_regime` | how the record ends after librating episodes (`circulation`, `libration` = captured at the end) |
| `reason`, `libration_origin` | why; `dynamic` = this resonance's force was measured; `kinematic`, `induced`, `subresonance`, `unknown` otherwise |
| `confidence` | `certain` > `high` > `medium` > `low`: ordinal evidence, not a probability; empty for `uncertain`/`invalid` |
| `caps`, `flags` | what holds the level down; each cap names another reading of the record (references/physics/confidence.md, "Caps and the other reading") |
| `action` | `none`, `spot_check` (look at the figure), `review`, `extend` (to `extend_to_years`), `resample` (at `resample_step_yr`) |

**What a verdict claims.** It describes the saved interval, at the saved step, in the run's model (planets, integrator,
initial elements, date). A longer span, a finer step or another epoch can change it; chaotic bodies change verdict
with weeks of start epoch. The thresholds are conventions tested on a benchmark labelled by eye.

Distinctions that matter when you report:

- **Librating is not resonant.** `non_resonant` with `angle_status = libration` (or `episodic`) means σ is bounded
  by something else: `kinematic_libration`/`kinematic_drift` (the perihelion term or a secular cycle, a does not carry
  σ), `induced_libration` (another resonance drives a; `owner` names it), `implausible_strength`.
- **Resonant with origin unknown.** `geometric_libration` (resonant) and `geometric_episode` (transient) have origin
  `unknown`: the shape of σ says libration but no force of this resonance could be measured. Say "librates
  geometrically; the dynamical origin was not established", not "confirmed resonant".
- **`subresonance_librates`.** The catalogue angle takes the status of a member of its multiplet (σ ± 2mω, nodal
  terms) that holds the centre. When counting bodies, count each commensurability once: deduplicate on
  (`name`, `parent` or `resonance`).
- **`transient`** = librating episodes that do not cover the record (capture, escape, jump of a). Check
  `final_regime` before saying "left the resonance" or "was captured".
- **`uncertain`** is not `non_resonant`. `baseline_needed` (angle `insufficient_time`) and `drift_mechanism_unknown`:
  extend to `extend_to_years`, which is the budget of the next check, not a promise. `origin_unknown`: σ librates but
  a measurement contradicts this resonance's pendulum; more time may not help, review the figure.
- **`invalid`** says nothing about the resonance: `alias`/`undersampled` → resample at `resample_step_yr`;
  `unphysical` → the orbit left the ellipse (encounter, ejection: check the integrator); `undefined_phase` → ϖ is
  undefined at e ≈ 0. A non_resonant `alias` (one branch, far from a*) is a real circulation seen through the step.
- **Lidov-Kozai.** A libration of the osculating ω is not enough; the free-ω gate decides. `non_resonant /
  kinematic_libration` is a **measurement** (the free ω winds: geometry); `uncertain / gate_unavailable` is the
  **absence** of one (`extend_to_years` for a short window or a residual near a planetary rate; `review` for an
  undefined free phase or missing free elements). Never merge them in a statistic (legacy codes −2 and −9). A co-orbital
  of Mercury to Mars gets no gate verdict (`zlk_gate_detail` = `coorbital_unfitted_forcing`: its planet's modes are not
  in the split); Jupiter Trojans are covered (forced vector g5 + g6), and 591986 stays kinematic.
  Columns that change no status (references/free-elements.md, "Beyond the gate"): `zlk_ei_coupling` (e–i at the
  decisive period, every row: `zlk` antiphase at constant H, `forced_beat` in phase; coupling is not libration — a
  circulating ω can show it), `zlk_forcing_*` (which forced mode holds a kinematic libration, or `not_attributed` and
  why; `zlk_attribution_scope` `episodes` for an episodic row demoted inside its episodes), `zlk_node_lock` (a flag:
  the mode the node is locked to, or empty with `zlk_node_lock_reason`, `unresolved` when no full cycle of Ω − ϖ_j fits
  the record), `zlk_centre_*` (the centre's slow line and its origin, e.g. `forced_beat:g6-g5`, `forced_direction`).
  The combined figure names the cause of a kinematic libration in its third sentence and adds panel (a′) (ω_f with
  the forced direction and the mode against the node) only for such rows.
- **Clones.** `clones.csv` gives the share of each status with its Wilson interval: a probability under the
  provider's covariance, the model and the span, not of the classifier's errors. The nominal has no special weight.
- `confirmed_ei` (= `zlk_ei_coupling == zlk`), `zlk_coherent` (the Welch line) and the diagnostic plots (recurrence,
  cycles, FAIR) never change a status.
- `status_code` is the legacy integer; use the string columns.

## 5. Traps

- **Default output step.** 15.9 yr makes every main-belt MMR `short_period_aliased` (≤ medium). The action is then
  `resample` at `smp_fold_free_step_yr`; prevent it with `output_step='auto'`.
- **Too short a record** gives `uncertain / baseline_needed`, not "not resonant". Compare the span with the period
  (`geo_period_cycle_yr`, `base_period_yr` in mmr_numbers.csv).
- **Retrograde orbits (i > 90°).** Use `RetrogradeTwoBody` keys (`1J-3r0`); `find` gives such bodies only retrograde
  candidates. Never build an angle of such an orbit from REBOUND's `orbit.l` or `orbit.pomega` (they are −λ*, −ϖ* and
  jump when i crosses 90°); their `longitude`/`varpi` data columns are in other conventions; no FAIR plot.
- **Secular angles.** The package compares the body's ϖ and Ω with the planets' secular mode longitudes g_j t + φ_j
  (`secular_angle_mode='proper'`). Do not rebuild a secular angle from the planets' osculating elements in
  `data-planet-*.csv`: from the ecliptic no planet's node circulates and most perihelia do not follow their mode.
- **Stale MMR catalogue.** A log warning "MMR catalog ... was built with other settings" means the cached
  `cache/mmr-*.csv` (paths relative to the working directory) has other `MATRIX_*` limits or planets. Delete it (or
  `TwoBodyMatrix.load(reload=True)`) before trusting `find`.
- **Today's date.** A `nasa` run without `date=` starts at today's midnight: two days give two runs, and a resume on
  another day fails its hash.
- **MPC epochs.** Rows at another epoch than the run's date are integrated to it (`MPC_EPOCH_POLICY=propagate`, with
  a warning) or refused (`strict`). The catalogue is refreshed only by `resonances mpc-update`.
- **Planets not saved** (`save_planets='none'`) or the owner search off: caps `catalogue_longitudes`,
  `owner_search_skipped`, `planets_missing` (≤ medium): rerun with the planets saved.

## 6. Read the combined figure

One page per plotted row (references/reading-figures.md, panel by panel; examples in references/physics/gallery.md; files in references/plots.md). The right
column is the verdict: the plaque `STATUS · angle_status · CONFIDENCE` (and `ends: …`), the verdict in words, the
steps of the status tree, the margins and caps (✓ passes, ⚠ thin, ✗ fails, – skipped by design, · not applicable, ? not measured, ▼ cap) with
the other reading each cap leaves open, and the next check in years. For an MMR the left panels are: (a) σ raw and
filtered with turning points and numbered episodes, (b) σ wrapped, (c) σ̇/2ω₀ (±1 = the separatrix speed of a cosine well; each episode's own limits are dashed), (d) a with
a* and the resonance's half-width, (e) ȧ with the fitted force law and its R², (f), (g) e and i, (h) the energy of
each cycle against the separatrix h = 1, (i) the slow centre, (j) the centres of other multiplet members and of an
owner (a flat line librates). For secular/ZLK rows, (f) is the rate fit (secular) or the drift of the free and the
osculating ω in turns (ZLK: a free ω that winds makes the libration kinematic). If you can view images, open the figure of every
`medium`/`low` verdict before reporting it.

## 7. Before you report a status

1. Read the whole row: `status`, `angle_status`, `final_regime`, `reason`, `libration_origin`, `confidence`, `caps`,
   `flags`, `action`, `extend_to_years`, `resample_step_yr`.
2. `invalid`: report that the record could not be measured and how to repair it; never as non-resonant.
3. Is the span long enough (several periods)? If not, or `uncertain`, or caps `three_tp_episode`/`one_return_cycle`:
   "not decided yet; integrate to N yr".
4. Is the step fine enough? `short_period_aliased`: the level is held down by the sampling; propose the resample step.
5. Name the origin: `dynamic` vs `unknown` (geometric), `kinematic`, `induced` (with `owner`), `subresonance`.
6. `certain`/`high`: state it. `medium`: look at the figure and name the cap and its other reading. `low`: review.
7. ZLK: give the gate's outcome; keep kinematic and gate_unavailable apart.
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
| matrix.md | resonance notation, catalogues, retrograde resonances, the catalogue cache |
| clones.md | clones: sources, epochs, how many, clones.csv |
| output.md | every file and column of a run |
| plots.md | figure kinds and options, redrawing after a run |
| reading-figures.md | how to read every figure, panel by panel, and what to look at first for each status |
| serialization.md | simulation.json, restore, resuming a batch run |
| config.md | every setting and threshold with its default |
| libration-analysis.md | the rule index: what a rule ID (G2a, L4, 5s, …) means, its settings, the page that explains it |
