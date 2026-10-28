# resonance/analysis — libration analysis v2

The only classifier, wired into `Simulation.identify_librations`. It gives the verdict of every type (MMR, secular,
ZLK) on shared geometry. Pure: no Body/Simulation imports, and no bare numbers in decision code. Formulas and the status
tree: `docs/physics/` (one home per topic; the rule IDs with their settings and modules: `docs/libration-analysis.md`; an example figure per status in `docs/physics/gallery.md`, figures in `docs/img/gallery/`, rebuilt by `make docs-figures`). MMR steps and rule IDs: `mmr/CLAUDE.md`. Secular and ZLK science:
`resonances/secular/CLAUDE.md`. Reasons, rejected variants and traps: `dev/`.

## Shared modules

- `verdict`: string `Status` (resonant, transient, near_separatrix, non_resonant, uncertain, invalid) /
  `AngleStatus`/`Reason`, `libration_origin` + `owner` + `origin_detail`, `final_regime` (how the record ends),
  `status_code` (a column for comparison with old runs; `verdict_from_status_code` only converts old integer summary
  files). `angle_status` describes the whole record: librating episodes followed by another regime, or a capture after
  one, are `episodic`. Old files: `angle_status_from_text` (angle `transient` → `episodic`, 2026-10-17), `reason_from_text`
  (`dynamic_*` with origin unknown → `geometric_*`, 2026-10-26). `not_measured` is never `fail`.
- `ledger`: every decisive comparison with its margin U. It feeds the CSV columns, the right column of the figure,
  the verdict text and the confidence; the decision path and the printed numbers must come from it, not be recomputed.
- `settings`: `AnalysisSettings` holds every threshold. Fields = `SimulationConfig` attributes = lower-case `.env.dist`
  keys. Shared names `LIBRATION_*`/`FILTER_*`/`SEPARATRIX_*`/`CONFIDENCE_*`; type prefixes `MMR_`/`SECULAR_`/`ZLK_`. A
  new threshold is a field + an `.env.dist` key with a comment (a convention says so, with its physical meaning) + a
  `docs/config.md` line. `settings_hash` goes into simulation.json and summary.csv. Trap: `settings_from_config()`
  without a source returns the dataclass defaults, so environment variables do not reach the MMR reclassification
  (`benchmark/tools/reclassify.py`); the secular/ZLK one builds a `Simulation` and does read the configuration.
- `filtering` (`resonance/analysis/filtering.py`): Butterworth in second-order sections, forward-backward (never (b, a) form). W23 memo: `record_memo` /
  `memoised` memoise `band_pass` and `response.smoothed_series` per `analyse_mmr` record (2 entries, keyed by object
  identity plus scalars, read-only arrays; never unscoped: a session-wide memo raised one worker from 1280 to 2084 MB).
  `LOW_MEMORY_ALLOCATOR` (`config.low_memory_allocator`, `worker_allocator`: `MallocLargeCache=0` for spawned workers
  on macOS).
- `pendulum` (`resonance/analysis/pendulum.py`): the potential, energies and periods. W24 review: `extrema` (exact, roots of z^N V'(z)),
  `loop_through`/`period_through` (turning points bracketed between extrema; at the separatrix → inf),
  `well_holding`, `cycle_period`: R0's period law reads each cycle in the well that holds its fastest sample (one
  well: the old law; two harmonics: every loop of one energy has one period in either well, so only a cycle below its
  well's bottom changes). For a multi-harmonic V the action J and J_s come from quadratures in the selected well; the
  half-width 2√(V'')/|c| and the cosine action hold for one harmonic only. W25: a libration cycle's J/J_s and its slope
  are read in the well it occupies (`occupied_well`, `CycleEnergy.well_rad`, nan = the deepest; `WellActions` caches
  each well's separatrix and J_s), not the deepest: with two harmonics J_deep(E) − J_shallow(E) is a constant, so the
  deepest well's ratio of a shallow-well cycle is too large. One-well records keep the deepest path exactly.
  W25 (topology): above its well's lower saddle in a V of several wells a cycle's loop encloses both wells: no
  single-well J/J_s (nan, `libration_topology` merged_wells/rotation, class `not_measured`, never "in the zone";
  `_libration_action` never regenerates a clipped 1); its own J_loop (`loop_action_through`) and T_loop/2π are columns.
  A one-well V keeps J/J_s = 1 above its top (separatrix). The averaged route compares actions: J/J_s of each turning
  point's loop in the averaged profile itself (`averaged_potential.action_in_well`) against J_lo, never an h mapped
  through V_E (`h_in_barrier_units` is gone). J_lo is J0/<J_s> (the record's mean separatrix action), so each local
  ratio is carried by J_s(t)/<J_s> of the averaged well, the scale 3k²μn²a included (`zone.tp_energies`,
  `centre_track.mean_separatrix_action`; Codex W25 #3: at a quarter of the mean strength 0.6 local is 0.3); no mean
  (a failed profile on the record's grid) → the pendulum's classes stay. The pendulum route does the same with
  √m_cycle. The zone classes reach L2, Z, caps and leaf 5o: not columns only.
- `confidence`: the level from margins and caps; `certain` is shown. certain needs every applicable margin full; high
  allows thin or unmeasured evidence but no failed support; one failed support → medium, two → low; nothing physical
  measured and explicit caps limit it; uncertain/invalid have no level. The calibration cell's author-label counts are
  only reported (`cell_*`, `resonances/data/confidence_cells.json`, rebuilt by `benchmark/tools/calibrate.py`); they
  never cap a level. Every cap down to medium or low names its alternative (`verdict_text.CAP_ALTERNATIVES`).
- `orbit_type`: the header's tadpole/horseshoe/… text, descriptive only (`centre_moves` is shared with the text).
  Subtype `stationary` means the cycle bands overlap, not zero centre motion (2003 TX9 keeps it while its centre
  moves; the text says so).

## Geometry (`resonance/analysis/geometry.py`)

- Each regime between jumps of a is measured on its own samples, with its own turning-point prominence; smoothing
  stays inside the regime. Turning points and centre per run.
- An episode must close at least one full cycle (`closure_ratios`), or one cycle by its phase return between two TPs
  (`phase_return`: at the mid-level or a band of levels, `LIBRATION_RETURN_LEVELS`, `LIBRATION_RETURN_BAND_LEVELS`);
  a return-closed cycle caps the level at medium and cannot cross a confirmed jump. ≥ 3 TPs and a closed cycle, never a
  TP count, separate an episode from noise.
- MMR only: an exit/entry is observed, not covered, when the orbit-averaged raw angle runs ≥ 1 turn monotonically from
  the last TP (edge arc). A run whose centre moves is cut at a centre step or a stall of the slow centre (regime cuts).
- Episodes without circulation between them cover the record together (`regime_change`, `covering_union`): 1:1 rule
  G2, any other MMR G2u; gaps ≤ `LIBRATION_UNION_GAP_PERIODS` × (P_l + P_r); edges reached within a period.
- After any cut (a jump of a or an action step) the record's entry is read on its first run (`Geometry.record_entry`,
  columns `record_entry_*`, flag `entry_suspected`; `Geometry.entry_suspected()` covers both).
- Derived views (`Geometry.derived`, held in `derived_views`) are emptied by `forget_derived` wherever runs or their
  edges change.
- **G2a action step (W24)**: `action_steps`/`ActionStep`. An accepted σ-frame run (not 1:1, no jump of a inside)
  whose half-amplitudes step between separated bands, with the a swing and the cycle energy changing the same way
  within one half-cycle (`LIBRATION_ACTION_STEP_*`), is cut by a confirmed `Jump(action_step=True)` whose zone is the
  transitional half-cycle (`Jump.zone`; regimes, rule 10, G1e/G1l, `JumpGuard` read it).
  - Test 7 `regime_periods` (made robust after the W24 review): the two sides' periods differ by >
    `LIBRATION_ACTION_STEP_PERIOD_TOL` (else `isochronous`: one forcing frequency explains both), and at every cutoff
    of `LIBRATION_ACTION_STEP_CUTOFFS` (0.075/0.1/0.15 × the shorter period, same number of wells) V_E of the larger
    swings (each of their disjoint cycles held out; one cycle: in sample, provisional) holds every disjoint cycle of
    both sides in a loop of its own energy around the well of its fastest sample (`pendulum.loop_through`, no loop
    vetoes), with its period (`_PERIOD_TOL`) and its turning points (`_TP_TOL` × half-swing), each side's energy
    spread ≤ `_ENERGY_DRIFT` × the step; `shape_failed` = 'subtest@factor'; V_F reported only.
  - Provisional (≤ medium, cap `action_step_provisional`) with one energy cycle on a side or one larger-side cycle.
  - `analyse._action_step_guard`: a row that cut a step of its own (`ActionStep.key`, `_ROW_KEY`) and leaves a check
    failed on the whole record unmeasured, R0 included (no exemption), with another verdict → the whole record's
    analysis is kept, `geo_action_step_failed` = withheld. Columns `geo_action_step_*`, `jump_action_step`.
  - 2003 TX9 3E+2J-10: tests 1–6 pass, test 7 fails (late cycles: law 1.61, turning points 0.74 off), not cut,
    uncertain; the author accepted uncertain (2026-10-27).

## Contracts

- Analysis arrays and Body arrays can share storage: never mutate them in place (planet memmaps and memo arrays are
  read-only). Copy instead (e.g. the 1:1 R0 through `response.with_coorbital_potential`).
- An analysis' arrays are dropped right after its verdict when the row will not be plotted (`drop_arrays` freezes the
  CSV rows first).
- Tests of decision code build real result objects (`tests/resonances/analysis/builders.py`), not stubs; decision
  code reads dataclass fields directly.
- Secular and ZLK: `analysis/secular/` (`analyse_secular`: cutoff from the forced waves, centre scale,
  `decide_status` with the free-omega gate's outcome, `extend.recommend`: the action and `extend_to_years` of an uncertain verdict, `slow`: the centre's slow line, a column) and `analysis/zlk/` (`analyse_zlk`; `diagnostics`: e-i coupling, attribution and the centre-scale guard `centre_guard`, columns only; the guard runs only with the gate enabled, a split and no unfitted co-orbital, and rejects only a measured forced beat that explains the centre). Gate disabled ≠ gate
  unavailable: disabled keeps the geometric status with `check_zlk_free_omega` not_measured.

## Do not retry here

Jump candidates from cycle energy / peak distance / steepest a; deleting small TP pairs before P_cyc; rule A (every
max above every min); TP counts as a physical criterion; a session-wide memo; G2a cuts that hide a failed check. Why:
`dev/rejected.md`.
