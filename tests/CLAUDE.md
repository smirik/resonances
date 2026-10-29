# tests — conventions

The root `CLAUDE.md` has the rules (commands, order, "never change real slow tests without asking"); this file has
the detail. Benchmark gates: `benchmark/CLAUDE.md`.

## Layout and markers

- `make test-fast`: tests of at most 1 s, about 1 min. `make test-slow` (marker `slow`): integrations and any test of
  more than 1 s (a shared module setup of more than 1 s: `pytestmark = pytest.mark.slow` on the module), about
  5 min. `make test-benchmark` about 15 min. All on pytest-xdist (`XDIST`, default `-n 4 --dist loadfile`: a module
  stays on one worker; the benchmark reclassification runs once per session under a file lock). `make test-long` (marker `long`): integrations of 10 min and more (z1 over 10 Myr); outside `test`,
  `test-fast` and `test-slow`. `make test-benchmark` (marker `benchmark`): local-only data and the author gates.
  `make test` = fast + slow. Measure with `--durations=0` before marking; on 2026-10-29 the split moved 234 functions
  to benchmark and 46 functions + 6 modules to slow (test-fast 30 min → 3 min).
- Local-only data (`benchmark/data`, `RESONANCES_BENCHMARK_DATA`; the cached solar system in `cache/`): the test is
  marked `benchmark` (a parameter: `pytest.param(..., marks=pytest.mark.benchmark)`), and its paths are resolved when
  it runs (`tests/resonances/local_data.py`), not at import. `tests/conftest.py` points every other test at a folder
  that does not exist and turns a skip for missing data into a failure; `make test-portable` runs test-fast on a
  fresh copy (no `benchmark/data`, no `cache/`). Never `pytest.skip` a fast or slow test for missing local data.
- Committed fixtures (`tests/fixtures/`): a few tens of KB each, 1 MB in all; larger ones live in
  `benchmark/data/fixtures/` (the free-omega series, ~147 KB each, moved there on 2026-10-29).
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
- Test output goes to `tests.tools.TEST_CACHE` = `cache/tests/<xdist worker | session-pid>`, cleaned after each test: workers
  and parallel sessions never touch each other's files (before W27 one shared `cache/tests` was deleted under the
  other session). Write test output under `TEST_CACHE`, never a literal `cache/tests`.

## Caches and memory

- `tests/conftest.py`: `SAVE_PATH`/`PLOT_PATH` = `TEST_CACHE`, removed after each test unless `KEEP_TEST_CACHE=1`
  (`make <target> KEEP=1`); benchmark series load through `benchmark/tools/column_cache.py`; each test module's own
  caches (`lru_cache` helpers, `*_CACHE` dicts) are cleared at the module's end (`release_module_caches`).
- `make test*` run with the low-memory allocator on macOS (`MallocLargeCache=0`); the fast suite peaks at ~1.9 GB.
  Avoid several 1-Myr analyses at once in one module.
