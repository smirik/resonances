## Overview

This package can identify mean-motion resonances (MMRs), secular resonances and Lidov-Kozai resonances in the Solar system. It loads the data from AstDyS catalog, NASA Horizons or the MPC, set up a simulation, perform numerical integration using rebound python package (integrator), build resonant angles and periodograms for some times series, classify the results, depending on the configuration, save the data and plots to files.

Core modules in `resonances/` (* = own `CLAUDE.md` with detail, rules, traps):

| Module | Purpose |
|--------|---------|
| `cli` | Command line interface (Click-based `resonances` command) |
| `data` | Utility functions, constants (`const.py`) |
| `clones` * | Virtual asteroids (orbit covariance → `Ensemble`); run side `simulation/ensemble_run.py` |
| `finder` | High-level interface to resonance finders (MMR, secular, Lidov-Kozai) |
| `lidov_kozai` | Lidov-Kozai resonance class, invariant computation |
| `matrix` | MMR catalog base, cached with a build record (`Matrix.check_cached`); see `mmr/CLAUDE.md` |
| `mmr` * | MMR classes (`TwoBody`, `ThreeBody`, `RetrogradeTwoBody`), catalogues (`TwoBodyMatrix`, `ThreeBodyMatrix`, `RetrogradeTwoBodyMatrix`), finder functions |
| `mpcorb` * | `DATA_SOURCE=mpc`: the MPC catalogue, self-contained; package side `simulation/mpc_source.py` |
| `plotting` * | Evolution plots, combined verdict figures, MMR/secular diagnostics, ecc-vector, cross-spectrum, free-omega, clones |
| `resonance` | `analysis` * (libration analysis v2, the only classifier; MMR in `analysis/mmr` *), classify (`resonance/classify/util.py` only: `is_unphysical_orbit`, `check_chaos`, `merge_intervals`, `calc_sigma_derivative`), factory, filtering, periodogram, cross spectra and coherence, `oscillations`, `omega_free_gate` |
| `secular` * | Secular resonances (also the ZLK science): formula parser, finder, `proper_angle`, `planet_modes`, `combinations`, `free_elements` |
| `simulation` * | Simulation engine: batches, bodies, I/O and layout, `dry_run`, integration (rebound), `propagate`, serializer, `state_manager` |
| `skills` | The agent skill (`skills/resonances/`: SKILL.md + generated `references/`) and `install_skill` behind `resonances skill install` |

Also `body.py` (`Body`), `config.py` (loads `.env.dist` + `.env`, exposes `SimulationConfig`), `horizons.py`, `logger.py`.

## Where knowledge lives

- The `CLAUDE.md` of the code you touch (* above, `benchmark/`, `tests/`): read it first.
- `dev/` (development notes, not on the docs site): history, decisions by topic, rejected approaches (do not retry), pitfalls, physics notes, verified facts, process, status (open questions).
- `docs/` (user docs): start at `physics/overview.md` (the physics pages hold the formulas and the status tree; `libration-analysis.md` is the rule index), `output.md`, `config.md`.
- `benchmark/DECISIONS.md`: the author's decision log (Russian, append-only; a later entry overrides an earlier one).
- `tmp/` is local and git-ignored (working notes, the author's verbatim answers); never cite it from code, tests or docs.

## Important notes

- This is a scientific project. Methods and functions must be accurately tested and validated.
- Statuses are strings (`analysis.verdict.Status`, with `angle_status`, `final_regime`, `confidence`, `reason`, `libration_origin`; the model: `resonance/analysis/CLAUDE.md`). `body.librations[key]` is always a `Verdict`, `body.statuses[key]` its status string; `Simulation._finalise_verdict` is their only writer. There is no integer status any more; `verdict.status_code` is a column for comparison with old runs.
- No hook changes a status after the analysis: the secular centre scale and the free-omega gate (Lidov-Kozai) are steps of `analyse_secular`/`analyse_zlk`, decided in `analysis.secular.decide.decide_status`. Save/plot modes (`SavePlotMode`) filter on status strings through `simulation.config.status_selected`.
- `Simulation.identify_librations()` is the only hook that runs on every path (single run, batch workers, `SimulationSerializer.restore(recompute_librations=True)`). It must update BOTH `body.librations[key]` and `body.statuses[key]` (through `_finalise_verdict`) — summary.csv reads the former (`Verdict.to_flat_dict()`), save/plot filters read the latter. Arrays of a row that will not be plotted are dropped right after its verdict (`_release_unplotted`; `drop_arrays` freezes the CSV rows first).
- The Lidov-Kozai status takes the free-omega gate's outcome in `decide_status`: confirmed (resonant), kinematic (non_resonant, kinematic_libration, code -2) or without a verdict (uncertain, gate_unavailable, code -9); a transient, drift or short record changes only on a kinematic outcome. -2 and -9 must never be merged in a statistic: -2 is a measurement, -9 is the absence of one.
- The e-i coupling changes no status; `confirmed_ei` is `zlk_ei_coupling == zlk`, measured at the row's decisive period for every ZLK row (why: `secular/CLAUDE.md`).
- The cluster representative in `free_elements` (`PRIORITY_E`, `PRIORITY_I`) is a modelling assumption, not a free convention. What must stay true, and is tested, is that no choice ever turns a non-librator into a libration.
- The free-element method does not apply to co-orbitals of a planet whose modes are not in the fitted basis (Jupiter Trojans are covered: their forced vector is Jupiter's, g5+g6); a confident-looking straight line there is not a measurement. Such a body (Mercury to Mars, `ZLK_COORBITAL_HILL_FACTOR`) gets the gate outcome unavailable (`coorbital_unfitted_forcing`), never not_applicable.
- Secular angles must not be built from the planets' osculating elements; `secular_angle_mode='proper'` (default) is what the classifier reads. The inner-planet frequencies and phases in `const.py` are measured on the package's own Newtonian model, not taken from the literature (why: `secular/CLAUDE.md`).
- The MMR diagnostic plots (recurrence D, cycles) are for reading figures: D uses the object's own scales and is not comparable between objects. Nothing there may feed a status.
- A 1:1 (co-orbital) MMR angle is judged by its own force law (`analysis/mmr/coorbital.py`, K/K_theory = 1 is the planet's own mass), not the harmonic law; the owner search does not apply to it.
- Never build an angle of an orbit with i > 90° from REBOUND's `orbit.l` or `orbit.pomega`: there they are −λ* and −ϖ* and jump when i crosses 90°. `RetrogradeTwoBody.calc_angle` uses Ω, ω, M only. The `longitude`/`varpi` data columns of such bodies are in different conventions, hence no FAIR for retrograde resonances.
- Do NOT change `times / (2 * np.pi)` conversions or the `save_body(body, times)` signature. The per-call conversion is intentional.

## Before changing the classifier

- Read the latest `benchmark/DECISIONS.md` entry on the topic, `dev/rejected.md` and `dev/pitfalls.md`.
- Every threshold is a setting: an `AnalysisSettings` field, an `.env.dist` key with a comment (say when it is a convention) and a `docs/config.md` line. Physical criteria, not counts.
- Run `make test-benchmark`; explain every changed verdict row against the previous run.

## Working with the author

- His labels are tentative: argue on the merits, disagree openly; agreement with him is not evidence. Specifications from outside are measured before they are implemented; report which claims survived.
- Questions go on one cheap-to-answer page (`dev/process.md`). No new iteration while disagreements are undiscussed; start on his explicit go.
- Heavy runs: `nice -n 10`, at most 4 workers, no heavy suites in parallel agents; durable scratch in `tmp/<programme>/`.
- Watch memory and CPU: before a heavy step and every 10–15 min while one runs (and whenever a command takes longer
  than expected), check `memory_pressure | tail -1` and `ps -A -o %cpu,rss,command | sort -rn | head`. Stop what is
  yours and runaway (free memory below ~25 %, unexpected worker swarms); report what is not yours, do not kill it.

## Documenting

- In /docs there is actual documentation of the project. It's written for humans. Keep it up to date. Keep it short and concise but clear for understanding. Don't write too much.
- In functions or classes, you may add comments for the files and functions if this is not obvious from the code or the names. E.g., the function sum(a,b) does not require any comments, whereas the function resolve_mmr_status requires a comment explaining what are the statuses and how they are resolved. But really keep it simple.
- If you change a function or a class, update the documentation inside the code accordingly.
- A change to what a figure shows (panels, lines, markers, colours, text, file names) updates `docs/reading-figures.md` and `docs/plots.md` in the same change and regenerates the docs images (`make docs-figures`).
- Update overall documentation at the end of the task (including the relevant `CLAUDE.md` and `dev/` page).
- The agent skill shipped with the package (`resonances/skills/resonances/`: `SKILL.md` + `references/`, installed by `resonances skill install [--user] [--path DIR] [--force]` into `.claude/skills/`; docs/agents.md) explains how to set up runs and read the results. `SKILL.md` holds the workflow and judgement and is written by hand: when a change alters the interface, the statuses/reasons/caps, the settings or the output layout in a way a user would notice, update it too. `references/` is GENERATED from the docs pages listed in `docs/tools/build_skill.py` (`make skill`; snippets inlined, links to omitted pages and images made absolute): never edit it by hand, run `make skill` after editing those pages — `tests/resonances/skills/test_skill.py` fails when it is out of date.

## How to run

Python >=3.11 required. Uses `uv` for dependency management.

```bash
uv sync                # install dependencies
uv run resonances      # CLI entry point
```

## Configuration

- `resonances/.env.dist` contains all default config values with comments.
- Place a `.env` file in the working directory to override values at runtime (precedence: `.env.dist` < environment variables < `.env`).
- `config.py` loads both files and exposes `SimulationConfig`.

## Testing

Any functionality except temporary should be tested. The package has the following conventions (detail: `tests/CLAUDE.md`, `benchmark/CLAUDE.md`):

- Code quality checks (run every time before saying "done"):
  ```bash
  uv run black . --check    # formatting
  uv run flake8 --count     # linting
  ```
  Never pass `--exclude` to black or flake8: it replaces their default excludes (and black's .gitignore), and they
  then walk `.venv/` and `tmp/` (on 2026-10-29 black ran 16 workers, ~13 GB, for 10 min). Use `--extend-exclude`
  (e.g. `.claude` for agent worktrees), and `black -W 4`.
- `make test-fast`: every test of at most 1 s (about 3 min in all), synthetic data and small committed fixtures only.
- `make test-slow` (marker `slow`): integrations (real bodies at fixed Horizons dates) and any test of more than 1 s,
  such as a long synthetic series (about 15 min). Run them at the end of the task when all other fast tests and
  formatting are done and passed.
- test-fast and test-slow must pass on any computer: nothing outside git. A test that reads local-only data
  (`benchmark/data`, the saved series, the cached solar system) is marked `benchmark`; `tests/conftest.py` hides that
  data from every other test and fails one that skips for it. `make test-portable` runs test-fast on a fresh copy of
  the files git knows. Committed fixtures: at most a few tens of KB each and 1 MB in all; anything larger goes to
  `benchmark/data` and its tests to `benchmark` (decided by the author 2026-10-29).
- `make test-long` (pytest marker `long`): the integrations of 10 min and more (z1 over 10 Myr); outside `test`, `test-fast` and `test-slow`.
- `make test` to run all tests - fast and slow (not `long`, not `benchmark`).
- `make test-benchmark`: every test that reads local-only data, and the exact gate on the agreement with the author's labels (every type; no integration); without `benchmark/data` it stops with one message. It fails on any disagreement with the author that is not in `benchmark/KNOWN_DISAGREEMENTS.csv`, on a listed one that now agrees, and on a listed one whose values changed (`benchmark/tools/known_disagreements.py <run> [--secular-zlk <run>] [--write]`). A change that moves an author row updates the list with a reason.
- The episode gate (same session, `benchmark/EPISODE_EXPECTATIONS.csv`): a tier A/B case that fails and is not `known_fail` fails the gate, so does a `known_fail` case that passes (clear the flag); C/D never gate.
- Never change real (slow) tests unless you are absolutely sure that you know what you do; ask me if you are not sure.
- Tests of the decision code build real result objects (`tests/resonances/analysis/builders.py`), not `SimpleNamespace` stubs: the decision reads dataclass fields directly, without `getattr` defaults.

For tests:

- Prefer functional tests over unit tests and mocks.
- When you create a new test, make sure that it tests a case obtained from different sources. For example, if you need to test a sum(a,b) function, calculate first a few examples independently (e.g., 2+3=5, 4+9=13) and then use these examples to test the function in a functional way.
- `KEEP_TEST_CACHE=1` (or `make <target> KEEP=1`) prevents cleanup of `cache/tests/` folder after tests.

## Classification benchmark

- The benchmark lives in `benchmark/` (tracked; `benchmark/CLAUDE.md`); series, figures and runs (`benchmark/data/`) stay outside git.
- Whenever a session meets an object worth keeping as an example (a new regime, an edge case, a classifier error the author points out), propose adding it: name, resonance, integration settings, the author's verdict, why it is distinct. Add it only after the author's explicit confirmation, with `label_source=author:<date>`.
- The author's visual verdict overrides old labels and algorithmic statuses. Labels produced by Claude or by an LLM judge are marked as such and never count as the author's.
- The category and reason in `KNOWN_DISAGREEMENTS.csv` and `known_fail`/`known_reason` in `EPISODE_EXPECTATIONS.csv` are Claude's reading, not labels; a change that makes an episode case pass or fail updates its flag with a reason.

## Code style

- Formatting: `black` (default settings). Linting: `flake8`.
- Use `StrEnum`/`IntEnum` for typed constants (e.g., `SavePlotMode`, `Status`, `AngleStatus`).
- Plotting classes inherit `BasePlotter` (shared `save`/`show`/`close`).
- `experiments/` contains one-off research scripts — not part of the package, not tested.
