# Pitfalls and traps

Things that cost a session hours. Each says what happens and what to do instead.

## Tooling and environment

- **`reclassify.py --figures` writes into the examples.** It writes `combined_v2.png` into
  `benchmark/data/examples/<id>/` even with `--out`. Trial renders go elsewhere, or back up and restore.
- **Environment variables do not reach the benchmark's settings.** `settings_from_config()` without a source returns
  the dataclass defaults, so an env-var experiment through `reclassify.py` is a no-op. Pass overrides explicitly and
  check `settings_hash` in `run.json` before reading a run.
- **Spawned workers import the current working tree.** Editing code during a paired run mixes versions (the W11/W12
  base and fix runs were contaminated this way; the first W23 baseline was discarded). Freeze a copy of the code and
  load it through `PYTHONPATH`; compare frozen trees and settings hashes, not run-folder names.
- **Temporary directories do not survive a reboot.** Durable scratch goes in `tmp/<programme>/` (git-ignored).
- **Parallel pytest sessions share `cache/tests`.** The autouse cleanup of one session deletes another session's files
  (`FileNotFoundError` under `cache/tests/...`); `find()` without a `name` always writes to `cache/tests/resonance_find`.
  Never run two test sessions from the same working directory at once.
- **Heavy parallel runs can make a workstation unresponsive.** Run heavy jobs with `nice -n 10`, at most 4 workers,
  and no heavy suites in parallel agents.
- **macOS keeps freed large blocks.** libmalloc retains them, so RSS is not live-array memory (a 400 MB / free /
  800 MB sequence reached 1163 MB; `MallocLargeCache=0` peaked at 781 MB and fell to 17 MB). Hence
  `LOW_MEMORY_ALLOCATOR`. Session-long `lru_cache`/`*_CACHE` helpers in tests held about 0.5 GB each.
- **macOS is case-insensitive.** `Omega.npy` and `omega.npy` collide; the column cache escapes capitals (`^` + lower).
- **OpenBLAS threads spin** with several workers: pin `OPENBLAS_NUM_THREADS=1`; recycle pool workers
  (`max_tasks_per_child=1`; reclassify `--tasks-per-worker`, default 5).
- **Wall-time asserts flake under load.** A 5-s wall-time assert failed under contention; time CPU, with a margin.
- **Horizons can refuse a date.** On Oct 26 it answered `Var not declared: IP_ADDR` for 2026-06-09 for every object;
  tests pinned to that date passed only from the astroquery cache. Pin to a date that answers (2026-06-10).
- **Horizons reads a numeric string as a major body** ('401' is Phobos); `horizons.py` guards it.
- **`tests/conftest.py` writes `cache/tests`** even in a read-only review. An RSS watchdog in one review could not set
  an OS memory limit; treat such a watchdog as advisory unless the limit is shown to be enforced.
- **Comparing runs byte for byte:** exclude the wall-clock column `sep_zone_seconds`; key derived rows by
  (example_id, resonance), not example_id alone (repeated ids are subresonance children, not duplicates).
- **Saved figures are not proof that a row is current.** An old child figure (2003 UG22 +7+0, Oct 16) outlived its
  row. Regenerate before showing figures.

## Units, epochs, data

- **`INTEGRATION_DT` is in REBOUND units:** 1.0 = 1/(2π) yr ≈ 58 d, not one year. A 33128 rerun with "1 yr" diverged
  by 48.6 kyr; the correct one matched σ and a to 5e-11. `meta` dt and the output `step_yr` use different units.
- **The package's year is Gaussian** (2π/k = 365.2568983 d), not Julian. Epoch arithmetic in days divides by 1/k days
  per time unit. Planet frequencies and phases are measured per model year.
- **Never mix element epochs.** AstDyS elements and planets 126 days apart flipped 463 Lola 4J-2S-1 from libration to
  circulation (σ₀ 151.5° vs 118.1°). `BodyManager.check_epoch` raises at more than 1e-4 d. Never use `datetime.now`
  where the configured date is meant.
- **Time scales.** Horizons reads a bare calendar date as TDB for vectors and elements; AstDyS and MPC epochs are TT;
  TT − TDB ≈ −0.7 ms (metres). Sending the TT calendar value is consistent. With `TIME_TYPE=UT` the EM barycentre moves
  by ~2000 km (69 s).
- **MPC near-Earth elements** (neam00) are wrong for bodies discovered at ~20,000 km Earth passes. MPCORB holds ~1.9k
  rows at other epochs (one-opposition orbits each at its own epoch): see `resonances/mpcorb/CLAUDE.md`.
- **The 1:1 sign.** The benchmark's saved 1:1 `angle_raw` is λ_planet − λ; `coorbital.py` handles the sign through
  `angle_sign`. Check the sign before fitting a force by hand.
- **Retrograde orbits:** never REBOUND's `orbit.l`/`orbit.pomega` (see the root `CLAUDE.md`).
- **CSV append after a failed or retried batch** used to duplicate rows (all 269 benchmark rows were doubled once).
  Fixed: `save_simulation_summary` replaces the rows of the bodies in memory (`_replace_rows_of`), the batch parent
  drops old rows before dispatching. Keep it that way when touching `data_manager.py`.

## Numerics

- **Butterworth in (b, a) form breaks** at normalised cutoffs ≲ 1e-3 (errors near 1e11 at 1-yr output); analysis v2
  uses second-order sections. The body-level `resonance/filtering.filter_angle` (the filtered series of the figures)
  still uses (b, a) with `filtfilt`: keep it out of decisions. The digital filtfilt gain uses tan(π dt/P), not the continuous approximation.
- **`np.unwrap` aliases the trend.** The predictive unwrap is valid while |δν|Δt < π, and wrapped residuals cannot
  expose every extra-turn error. Do not certify an alias from a turn count the predictive unwrap itself inserted.
- **A forced-line search can match the angle's own λ family** (42355 matched itself at 2.1 kyr) and raise the cutoff
  until it erases the evidence: exclude the angle's own family and its sisters, and ask for more than a frequency
  coincidence when many lines are scanned.
- **The flat-a force bound** is K_max = |ν|Δa/(1 − cos(S/2)) for an observed arc S ≤ 2π, else |ν|Δa/2
  (`response.py`); the old Δa/T_window bound was too strong. Do not revert it.
- **A libration can raise its own turning-point threshold:** its image leaking through the two filters (2013 AS27:
  48.9°, alternate TPs lost, period doubled to 310 yr). `analyse.fast_part` subtracts `libration_image` first.
- **Lomb-Scargle in [2/T, 1/500 yr] hides long periods**, so T/P passes a "three cycles" gate spuriously (2023 RS2:
  2.1 real cycles, 3.68 inferred). Count cycles from turning points.
- **Cubic potential extrapolation beyond the record** drove DOP853 to ~1e-6 steps (385266: 4.4 M calls, > 6 GB per
  piece). Clamp model times; store bounded `t_eval` samples.
- **One force window copied into an empty half is not two independent halves** (24820); one centre compared with
  itself does not show stationarity. `min(finite, nan)` silently passes a one-sided check: `independent_halves_r2`
  returns nan unless both directions are measured.
- **A phase-error sum ≥ 1:** arcsin(clamp(sum, 1)) invents a bound; return not_measured.
- **Moving well:** at a σ turning point ψ̇ = −ċ, not zero; R0 must include ċ²/2 (a translating pendulum gave a period
  ratio 0.77 instead of 1.00003 without it).
- **Quadrature self-consistency is not convergence.** Force and potential derivative on the same grid agreed, yet at
  e = 0.2 the 720-node force was 167.3 against 144.1 adaptive (16 %).
- **Edge coverage with a huge TP gap as its period** falsely covered 154634 (25.6-kyr centre turns hid a 308-yr
  ripple); the edge step is capped by `LIBRATION_COVERAGE_PERIODS` × P.
- **Prominence is excursion, not frequency.** Raising p_c can hide a genuine 4° slow child libration; not_measured is
  the honest outcome there.
- **Free vectors near their origin** create false 180° turns when unwrapped; mask each free vector near its own origin
  first.
- **Output step count:** `nout_for_step = ceil(span/step) + 1`; plain rounding gave Nout = 1 for a 1-yr auto run.
- **Lyapunov times** come from MEGNO or log-shadow growth slopes, not t/(2Y) at one plateau; a MEGNO plateau near 5
  does not prove sustained chaos.
- **Jump noise from the total a swing** can hide a real step (0.083 AU vs a 0.157 AU threshold); repeated 250-yr
  reversals merge under 2τ = 300 yr into a fake 9.7-kyr drift. These were real strict xfails, not bad tests.

## Process and statistics

- **A label belongs to its exact record** (integrator, span, step, epoch). Thinning or resampling changes the alias
  and cutoff evidence.
- **Old `author_verdict` values** in `candidates.csv` are overridden by `author_label_decisions_2026-10-15.csv`
  (later rows win); `labels.csv` buckets are not author labels; labels produced by Claude or an LLM never count as
  the author's.
- **Benchmark windows are not independent experiments.** 0 errors in 30 gives an upper 95 % bound of 9.5 %. A
  targeted hard sample cannot estimate population error rates; parity of 490 numbers proves arithmetic, not physics.
- **A test oracle that omits L4/R0 can pass a wrong leaf.** Decision tests need an independent truth table and
  fixtures that fail on the broken implementation.
- **Explain a check only after reading what it compares.** An explanation to the author once said a failed C4 meant
  "near the separatrix"; C4 compared σ̇ with a K from a fit the package had itself rejected (R² 0.005–0.18), so it
  measured nothing. A cap's ceiling is not a level; do not attribute to the author words he did not write.
- **A held-out check must be held out.** A predicted centre built from the observed upper TP is partly fitted by
  construction (2003 UG22); use the held-out lower TP and matched nulls.
- **Missing owner-planet series** make benchmark and production verdicts differ (catalogue longitudes cap the owner
  search at medium).
