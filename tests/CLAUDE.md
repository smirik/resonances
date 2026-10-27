# tests — conventions

The root `CLAUDE.md` has the rules (commands, order, "never change real slow tests without asking"); this file has
the detail. Benchmark gates: `benchmark/CLAUDE.md`.

## Layout and markers

- `make test-fast`: no integration. `make test-slow` (marker `slow`): real bodies at fixed Horizons dates, about
  6 minutes. `make test-long` (marker `long`): integrations of 10 min and more (z1 over 10 Myr); outside `test`,
  `test-fast` and `test-slow`. `make test-benchmark` (marker `benchmark`). `make test` = fast + slow.
- `tests/resonances/independent_wN/`: tests written by a context-free subagent for wave N. When one fails, check the
  code first (on 2026-10-15, 6 of 9 failures were the code's fault); record a dispute with its reason.
- Decision tests build real result objects (`tests/resonances/analysis/builders.py`: `make_geometry`,
  `make_episode`, `make_response`, `make_sampling`, ...), not `SimpleNamespace` stubs: the decision reads dataclass
  fields directly, without `getattr` defaults.
- A strict xfail marks a known physics loss (e.g. TX9's centre scale at p_c 7°/16°), not a bad test; when it starts
  passing, the flag goes and the reason is recorded.

## Writing tests

- Prefer functional tests with expectations computed independently (papers, hand calculations, another tool).
- A test must fail on the broken implementation: an oracle that omits a check (L4/R0) can pass a wrong leaf.
- Time CPU, not wall clock, with a margin (a 5-s wall assert flaked under load).
- Do not cite `tmp/` files in tests; cite docs.

## Slow tests and dates

- Slow tests depend on the Horizons epoch: `create_solar_system()` fetches the planets at the run date. Pin the date
  (2026-06-10, 2026-09-29, 2023-02-25 are in use; Horizons refused 2026-06-09 on 2026-10-26, so tests at that date
  pass only from the astroquery cache).
- 3040 ZLK changes verdict with a few weeks of start epoch (chaotic); 42355 3N-4 was removed (its "transient"
  expectation was stale; the benchmark gates its author label).
- Before blaming the code for a slow failure, run the same test on the previous commit with the same `cache/`.
- Never run two pytest sessions from one working directory: the autouse cleanup deletes `cache/tests` under the other.

## Caches and memory

- `tests/conftest.py`: `SAVE_PATH`/`PLOT_PATH` = `cache/tests`, removed after each test unless `KEEP_TEST_CACHE=1`
  (`make <target> KEEP=1`); benchmark series load through `benchmark/tools/column_cache.py`; each test module's own
  caches (`lru_cache` helpers, `*_CACHE` dicts) are cleared at the module's end (`release_module_caches`).
- `make test*` run with the low-memory allocator on macOS (`MallocLargeCache=0`); the fast suite peaks at ~1.9 GB.
  Avoid several 1-Myr analyses at once in one module.
