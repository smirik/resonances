# clones — virtual asteroids

User docs and literature: `docs/clones.md`. Decided by the author on 2026-10-27: interface B (orbit solution object +
ensemble) with the short `CLONES*` keys, N = 100 by default, weak NEAs at the fit epoch (`.eq0`), non-gravitational
parameters only flagged, `find` takes the whole cloud's candidates.

## Modules

- `orbit_solution` → `OrbitSolution` (covariance from AstDyS/NEODyS OEF, JPL SBDB, MPC get-orb; cached in
  `CLONES_CACHE_DIR`). `clones/solution.py` parsers: OEF λ and its COV in **degrees**; SBDB at the solution epoch with its own
  nominal (fall back to `orbit.elements` when the covariance block has none); MPC CAR, refused for U ≥ 3.
- `elements.py` conversions; `ensemble.py`: `OrbitSolution.sample` → `Ensemble` (multivariate normal in the provider's
  published covariance coordinates, exact conversion of each clone, d² stored, prefix-stable seeds), save/load, `from_elements` for the
  user's own clones; `stats`: Wilson shares.
- Package side: `Simulation.add_ensemble` / `add_body(clones=)`. Clones are drawn at the covariance epoch; nominal and
  clones are integrated to `date` (`simulation/propagate.py`, never a linear covariance map). The nominal is an
  ordinary body (its verdict does not change); clones `<name>~c001` run after the ordinary bodies in `CLONES_BATCH`
  batches by the batch workers (`_clone_batch`, `worker_results`, `simulation/ensemble_run.py`).
- Output: clones.csv (six shares + Wilson, modal, nominal agreement, `nominal_offset_sigma`),
  `data/ensemble-<name>.csv/.json`, figure kind `clones` (`plotting/clone_plots.py`). Not resumed, not in
  simulation.json.

## Rules (why: `dev/rejected.md`, `docs/clones.md`)

- Sample the full covariance in the coordinates it is published in (native fit coordinates are established for the
  OEF only; MPC CAR and SBDB cometary are linear maps of the fit); never independent per-element errors (a cometary diagonal gets
  σ_a wrong by 4–889×). Exceeding an ill-conditioned (1e13–1e19) ellipsoid in another metric is not a dynamical
  failure; compare the marginals that matter (σ_a, σ_e, σ_λ).
- Draws that are no bound orbit (e ≥ 1) stay in the sample as invalid clones; never redraw them (that is silent
  truncation).
- OEF: use `COV`, factor it without inverting; for ill-conditioned orbits `NOR` and `COV` disagree (‖N·C − I‖ up to
  2e8). `WEA`/`EIG` hold only in the radian-scaled metric, while `COV` has λ in degrees.
- The nominal runs in its own simulation, as without clones: with SABA `safe_mode = 1` round-off makes a trajectory
  depend on the number of particles (shifts ~5e-10 au), so a chaotic nominal would change verdict with the clone
  batching. The same holds for ordinary batches: reproduce a nominal only with the same batch composition.
- Never propagate a covariance linearly over years (`.eq1` for weak NEAs inflates σ_λ by hundreds through encounters).
- The nominal has no special weight ("without any special significance"); `nominal_agreement` is a diagnostic.
- Keep ZLK −2 and −9 apart in the shares, as everywhere.
- Cost is the classification, not the integration: 1–2 CPU-min per clone for a 1-Myr record (the benchmark median of
  1.15 s per row is for short records); memory per worker follows the analysis of the longest record.
