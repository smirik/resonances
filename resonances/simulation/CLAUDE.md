# simulation — the run pipeline

User docs: `docs/output.md`, `docs/dry-run.md`, `docs/serialization.md`, `docs/config.md`.

## Hooks and their contracts

- `Simulation.run`: `prepare_run` (dry-run) → integration → `prepare_angles` → `identify_librations` →
  `build_periodograms` → save/plot. `identify_librations` is the only hook on every path (single run, batch workers,
  `SimulationSerializer.restore(recompute_librations=True)`); `Simulation._finalise_verdict` is the only writer of
  `body.librations[key]` (a `Verdict`) and `body.statuses[key]` (its status string). No hook after the analysis.
- `restore` reads the statuses from summary.csv by default; with `recompute_librations=True` it classifies only the
  saved series and rebuilds their periodograms; it writes no table, except for ensembles' nominals (clones.csv and
  their summary.csv rows, together). It finds the series through the manifest (old flat runs too), reads them with `float_precision='round_trip'` (the
  analysis must read the bits the run read) and takes the saved years (`times`) as they are.
- Derived rows (subresonance members) keep `parent`/`added_by` through workers, simulation.json and restore; they are
  emitted once (never recursively through `.derived` after the body's own resonances).
- `_release_unplotted`: an analysis' arrays are dropped right after its verdict when its row will not be plotted
  (`drop_arrays` freezes the CSV rows first).
- `mmr_adapter`: Body → `MMRSeries`, analysis series → Body. The analysis always gets its arrays in increasing time
  (`time_order`: a reversed view for a backward run); backward order exists only at the simulation/output boundary.

## Modules

- `simulation/config.py`: `SimulationConfig` (incl. `analysis_settings`), `SavePlotMode`, `status_selected` (save/plot filter on
  status strings: `resonant` = resonant + transient; `candidates` adds near_separatrix and uncertain; `extended` adds
  non_resonant with a libration origin; `nonzero` = everything except non_resonant without an origin). Default figure layout `DEFAULT_PLOT_LAYOUT` = `confidence/status` (author 2026-10-27); `PLOTS=combined`.
- `simulation/data_manager.py`: summary.csv (string statuses, verdict + ledger columns, every type), episodes.csv (MMR, secular,
  ZLK), mmr_numbers/candidates/constants CSVs, coherence.csv — tables at the top of save_path; series CSVs in
  `DATA_FOLDER` (`data/`), figures in `PLOT_FIGURES_FOLDER` (`figures/`) and `PLOT_COMBINED_FOLDER` (`combined/`),
  split by `PLOT_LAYOUT`/`PLOT_COMBINED_LAYOUT` presets (`simulation/layout.py`: flat, status, confidence, confidence/status, …;
  `figure_folder`; the benchmark tree uses `benchmark/tools/sort_figures.py`). Table writes lock `<table>.lock`,
  removed by its holder (`_locked`: no lock left after a run; after locking, `samestat` checks the path still names the
  locked file, else retry). `save_simulation_summary` replaces the rows of the bodies in memory (`_replace_rows_of`);
  batch workers only append, the parent drops old rows first — this is what prevents duplicate rows on retry.
- `simulation/dry_run.py`: static/pilot dry-run (SPEC §9, `docs/dry-run.md`): alias verdict and ghosts of the output step, the
  cutoff (physics only) and whether the step can filter it (undersampled) or folds short periods
  (`short_period_aliased`, by the analysis' own predicate `sampling.short_period_trigger`; the pilot's leak of σ and ȧ
  accepts or rejects such a step), integrator recommendation, cost; `OUTPUT_STEP=auto` (walked down to a step that
  folds no harmonic, `fold_free_step`); `prepare_run` is the first step of `Simulation.run`. Secular/ZLK angles get
  advisory static rows only (`long_period`, `dryrun_long_period.csv`: the analysis' cutoff, period floor and
  baseline, the ZLK gate's cycles of the quadrupole time); they never set the step.
- `batch_manager.py`, `parallel_executor.py`: spawn workers, `max_tasks_per_child=1`, BLAS threads pinned, the
  low-memory allocator on macOS. Workers get element dicts and the serialised config; they never open a catalogue.
  The resume hash includes date, source, body list, settings, the MPC snapshot and every body's initial elements
  (`clone_store.canonical_elements`, shared with the clone-run identity; a get-orb body's `getorb_sha256` with them);
  simulation_state.json's `body_elements` lets a refusal name the changed bodies. States of the hash without elements
  resume with a warning (`_get_config_dict(elements=False)`).
- `body_manager.py`: elements from AstDyS, Horizons or MPC; `check_epoch` raises when elements and the run date differ
  by > 1e-4 d. `mpc_elements` (see `resonances/mpcorb/CLAUDE.md`).
- `mpc_source.py`: MPC_* keys → `mpcorb.Settings`, registered as its defaults; `epoch_refusal`.
- `propagate.py`: integrates a body given at another epoch to the run date (IAS15 among the run's planets taken at the
  body's own epoch; never a linear covariance map). Used by the MPC `propagate` policy and by clones.
- `ensemble_run.py`: clones in a run (see `resonances/clones/CLAUDE.md`); with ensembles `Simulation.run` goes
  through `run_with_clones` (ordinary part, then clones, one checkpoint). `clone_store.py`: the checkpoint
  `clones_state/` (identity, starts, `batch-<i>.npz` published by tmp + `os.replace`, existence after validation =
  done), the results archives (JSON header as uint8 + numeric arrays, `allow_pickle=False`), simulation.json's
  `ensembles` and their restore. Only the coordinating simulation owns run-level state: workers have no ensembles and
  never create or clear it. `_verify_existing_path` treats `clones_state/clones_state.json` as a resume, like
  simulation_state.json.
- `mpc_getorb.py`: `MPC_LOOKUP=file_then_api`, the get-orb nominal (KEP, else CAR; no covariance) of an identifier the
  snapshot lacks, through the clones' cache; called from `BodyManager.mpc_elements` in the parent only.
- `integration.py`: rebound; planets from Horizons (`cache/solar-<timestamp>.bin`), the asteroid added with
  `primary=Sun`; clones and asteroids are test particles (`N_active` = the massive bodies).

## Units and epochs

- `INTEGRATION_DT` is in REBOUND units: 1.0 = 1/(2π) yr ≈ 58 d. The year is Gaussian (`GAUSSIAN_YEAR_DAYS`).
- `times / (2 * np.pi)` once per call, and the `save_body(body, times)` signature: do not change (root rule).
- Elements and planets must share an epoch. Horizons reads the bare calendar date as TDB; AstDyS/MPC epochs are TT
  (TT − TDB ≈ 0.7 ms): consistent. Horizons numeric names need care ('401' is Phobos: `horizons.py` guards it).
- The Moon option requires ias15.
- Integrator choice: ias15 for NEAs (q < 1.3 au) and planet-crossers; SABA with dt ≤ min(58 d, P/10, τ_peri/10).
  Jupiter Trojans at a 15.9-yr step are undersampled; the dry-run warns before the run.

## Traps

- `find()` without a `name` writes to `<SAVE_PATH>/resonance_find`; in tests SAVE_PATH is the process' own
  `tests.tools.TEST_CACHE`, so parallel sessions and xdist workers do not collide.
- Old `*.lock` files from runs before W24 are reused and removed; `benchmark/tools/integrate.py` still deletes them in
  task folders.
- Selective saving: restore after `SAVE≠all` classifies only the saved angles and reads the other verdicts from summary.
