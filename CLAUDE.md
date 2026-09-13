## Overview

This package can identify mean-motion resonances (MMRs), secular resonances and Lidov-Kozai resonances in the Solar system. It loads the data from AstDyS catalog or NASA Horizon, set up a simulation, perform numerical integration using rebound python package (integrator), build resonant angles and periodograms for some times series, classify the results, depending on the configuration, save the data and plots to files.

Core modules in `resonances/`:

| Module | Purpose |
|--------|---------|
| `cli` | Command line interface (Click-based `resonances` command) |
| `data` | Utility functions, constants |
| `finder` | High-level interface to resonance finders (MMR, secular, Lidov-Kozai) |
| `lidov_kozai` | Lidov-Kozai resonance class, invariant computation |
| `matrix` | MMR catalogs — `TwoBodyMatrix`, `ThreeBodyMatrix` with resonant semi-major axes |
| `mmr` | MMR classes (`TwoBody`, `ThreeBody`), finder functions |
| `plotting` | Time series, periodogram, phase, cross-spectrum and free-omega plotting with flexible configurator |
| `resonance` | Core: classify (`classify_resonance` → `classify_from_data` → `classify_from_metrics`), factory, filtering (Butterworth low-pass; feeds periodograms and classification, not just plots), periodogram (Lomb-Scargle via astropy), cross_spectrum (Welch coherence), coherence_analysis (pair presets + the e-i exchange flag), omega_free_gate (the Lidov-Kozai status decision), planets_mappings |
| `secular` | Secular resonance class, formula parser, finder, proper angle builder, free (proper) element decomposition |
| `simulation` | Simulation engine: `batch_manager` (multicore), `body_manager`, `data_manager` (I/O), `integration` (rebound), `serializer`, `state_manager` |

Key files in `resonances/`:

| File | Purpose |
|------|---------|
| `body.py` | `Body` class — simulation data structure |
| `config.py` | Loads `.env.dist` + `.env`, exposes `SimulationConfig` |
| `horizons.py` | Fetch Keplerian elements from NASA Horizons |
| `logger.py` | Logging (info/warning/error levels) |
| `simulation/config.py` | `SimulationConfig`, `SavePlotMode` enum |
| `simulation/data_manager.py` | Save/plot logic, summary.csv + segments.csv + coherence.csv generation |
| `resonance/classify/classify.py` | Whole classifier: `classify_resonance` → `classify_from_data` → `classify_from_metrics` (window/segment TTO decision tree) |
| `resonance/classify/models.py` | `ResonanceStatus` (IntEnum), `SegmentMetrics`, `SegmentCounts`, `ClassifyParams`, `ResonanceClassifyResult` |
| `resonance/classify/util.py` | `is_unphysical_orbit`, `check_chaos`, `merge_intervals` |
| `resonance/cross_spectrum.py` | Welch coherence + cross-phase, line extraction, phase criterion, e_forced/e_free. Pure math, no Body/Simulation |
| `resonance/coherence_analysis.py` | Per-resonance-type pair presets, `analyse_resonance`, the `confirmed_ei` flag, CSV row builders. Changes no status |
| `resonance/omega_free_gate.py` | The free-omega gate: confirms status 2, demotes to -2 (kinematic libration) or withholds at -9. The only post-classification status change |
| `secular/free_elements.py` | Forced/free split by least squares at the planetary fundamentals: cluster the candidates at 2/T, fit one representative each, mask the origin, measure the three rates. Pure signal processing, no Body or status |

Detailed documentation lives in `docs/` — see `docs/classify.md` for classification logic, `docs/coherence.md` for cross spectra and the ZLK gate, `docs/free-elements.md` for the forced/free split and the free omega, `docs/config.md` for all config options.

## Important notes

- This is a scientific project. Methods and functions must be accurately tested and validated.
- `ResonanceStatus` (IntEnum in `resonance/classify/models.py`) is the central status type: values range from 2 (LIBRATION) to -99 (CHAOTIC). Save/plot modes (`SavePlotMode`) filter by these values.
- Classification has side effects: `classify_resonance` sets `chaos_flag`/`chaos_comment` on the result and may short-circuit to CHAOTIC for unphysical orbits.
- `Simulation.identify_librations()` is the only hook that runs on every path (single run, batch workers, and `SimulationSerializer.restore()`); post-classification status changes belong there. It must update BOTH `libration.status` and `body.statuses[key]` — summary.csv reads the former (through `to_flat_dict()`), save/plot filters read the latter.
- The Lidov-Kozai status is decided by `omega_free_gate.apply_gate`, and only from LIBRATION(2): confirmed at 2, lowered to LIBRATION_UNCERTAIN(-2) when the libration is kinematic, or UNCERTAIN(-9) when no verdict is possible. -2 and -9 must never be merged in a statistic: -2 is a measurement, -9 is the absence of one.
- The e-i coherence check no longer changes any status; it produces `confirmed_ei`. Its validity band (`T/(5·n_seg)`) sits below the real Lidov-Kozai periods on the short baselines the package now targets (see `docs/coherence.md`).
- The cluster representative in `free_elements` (`PRIORITY_E`, `PRIORITY_I`) is a modelling assumption, not a free convention — a different member changes the verdict on 3752 and 591986. What must stay true, and is tested, is that no choice ever turns a non-librator into a libration.
- The method does not apply to co-orbitals: their forced term is the perturber's own vector, not a planetary fundamental. A confident-looking straight line there is not a measurement (see `docs/free-elements.md`).
- Do NOT change `times / (2 * np.pi)` conversions or the `save_body(body, times)` signature. The per-call conversion is intentional.

## Documenting

- In /docs there is actual documentation of the project. It's written for humans. Keep it up to date. Keep it short and concise but clear for understanding. Don't write too much.
- In functions or classes, you may add comments for the files and functions if this is not obvious from the code or the names. E.g., the function sum(a,b) does not require any comments, whereas the function resolve_mmr_status requires a comment explaining what are the statuses and how they are resolved. But really keep it simple.
- If you change a function or a class, update the documentation inside the code accordingly.
- Update overall documentation at the end of the task.

## How to run

Python >=3.11 required. Uses `uv` for dependency management.

```bash
uv sync                # install dependencies
uv run resonances      # CLI entry point
```

## Configuration

- `resonances/.env.dist` contains all default config values with comments.
- Place a `.env` file next to `.env.dist` to override values at runtime.
- `config.py` loads both files and exposes `SimulationConfig`.

## Testing

Any functionality except temporary should be tested. The package has the following conventions:

- Code quality checks (run every time before saying "done"):
  ```bash
  uv run black . --check    # formatting
  uv run flake8 --count     # linting
  ```
- `make test-fast` to run all fast tests that do not require numerical integration, which takes time.
- `make test-slow` to run all slow tests that require numerical integration, which takes enough time. Run them at the end of the task when all other fast tests and formatting are done and passed.
- `make test` to run all tests - fast and slow.
- Never change real (slow) tests unless you are absolutely sure that you know what you do; ask me if you are not sure.

For tests:

- Prefer functional tests over unit tests and mocks.
- When you create a new test, make sure that it tests a case obtained from different sources. For example, if you need to test a sum(a,b) function, calculate first a few examples independently (e.g., 2+3=5, 4+9=13) and then use these examples to test the function in a functional way.
- `KEEP_TEST_CACHE=1` (or `make <target> KEEP=1`) prevents cleanup of `cache/tests/` folder after tests.

## Code style

- Formatting: `black` (default settings). Linting: `flake8`.
- Use `StrEnum`/`IntEnum` for typed constants (e.g., `SavePlotMode`, `ResonanceStatus`).
- Plotting classes inherit `BasePlotter` (shared `save`/`show`/`close`).
- `experiments/` contains one-off research scripts — not part of the package, not tested.
