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
| `plotting` | Time series and periodogram plotting with flexible configurator |
| `resonance` | Core: classify (3-layer: `classify_resonance` → `classify_from_data` → `classify_from_metrics`), factory, filtering (scipy), periodogram (Lomb-Scargle via astropy), resolver (final MMR status from periodogram peak overlap), planets_mappings |
| `secular` | Secular resonance class, formula parser, finder, proper angle builder |
| `simulation` | Simulation engine: `batch_manager` (multicore), `body_manager`, `data_manager` (I/O), `integration` (rebound), `serializer`, `state_manager` |

Key files in `resonances/`:

| File | Purpose |
|------|---------|
| `body.py` | `Body` class — simulation data structure |
| `config.py` | Loads `.env.dist` + `.env`, exposes `SimulationConfig` |
| `horizons.py` | Fetch Keplerian elements from NASA Horizons |
| `logger.py` | Logging (info/warning/error levels) |

## Important notes

- Note that this is a scientific project. The methods and functions must be very accurately tested and validated and stay reliable.

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
