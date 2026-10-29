import gc
import os
import re
from pathlib import Path
import shutil
import sys

os.environ.setdefault('LOG_LEVEL', 'warning')

import pytest  # noqa: E402

import resonances  # noqa: E402


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
BENCHMARK_TOOLS = os.path.join(ROOT_DIR, 'benchmark', 'tools')


@pytest.fixture(autouse=True, scope="session")
def setup_test_config():
    resonances.config.set('SAVE_PATH', 'cache/tests')
    resonances.config.set('PLOT_PATH', 'cache/tests')


@pytest.fixture(autouse=True, scope="function")
def cleanup_test_files():
    Path('cache/tests').mkdir(parents=True, exist_ok=True)
    """Clean up test cache after each test."""
    yield  # Test runs

    if os.environ.get('KEEP_TEST_CACHE') not in (None, '', '0'):
        return  # skip clean when run with KEEP_TEST_CACHE=1 (or `make ... KEEP=1`)

    cache_path = Path('cache/tests')
    if cache_path.exists():
        shutil.rmtree(cache_path)


@pytest.fixture(autouse=True, scope="session")
def benchmark_column_cache():
    """The saved benchmark examples load through the binary copy of their CSVs (benchmark/tools/column_cache.py, the
    planets as read-only memory maps), as in reclassify: the tests called load_series 282 times, 147 s of CSV parsing
    (W23 perf). Only files under the benchmark data are kept; BENCHMARK_COLUMN_CACHE=0 turns it off."""
    if BENCHMARK_TOOLS not in sys.path:
        sys.path.insert(0, BENCHMARK_TOOLS)
    import column_cache

    installed = column_cache.install()
    yield
    if installed:
        column_cache.uninstall()


@pytest.fixture(autouse=True, scope="module")
def release_module_caches(request):
    """At the end of a test module: clear the caches it defined at its top level, the functools caches of its helpers
    and its `*_CACHE` dicts, which keep whole analyses until the session ends (one helper alone kept +511 MB, W23 perf).
    A cache only spares recomputation inside its own module, so no result changes."""
    yield
    module = request.module
    for name, value in list(vars(module).items()):
        clear = getattr(value, 'cache_clear', None)
        if callable(clear) and getattr(value, '__module__', None) == module.__name__:
            clear()
        elif name.endswith('_CACHE') and isinstance(value, dict):
            value.clear()
    gc.collect()


# --- local-only data (tests/CLAUDE.md): test-fast and test-slow must pass the same way on a fresh clone

LOCAL_DATA_MISSING = re.compile(r'benchmark (example )?data|benchmark examples|cached solar system', re.IGNORECASE)
NO_LOCAL_DATA = Path(ROOT_DIR) / 'cache' / 'no-local-data'  # never created


def _is_local(item) -> bool:
    return item.get_closest_marker('benchmark') is not None


@pytest.fixture(autouse=True)
def hide_local_data(request, monkeypatch):
    """A test not marked `benchmark` runs as on a fresh clone: the benchmark data (`benchmark/data`, not in git) is
    pointed at a folder that does not exist, so a test that reads it fails here too instead of passing only on the
    machines that hold the data (on 2026-10-29, 572 tests of test-fast were skipped on a clean copy)."""
    if _is_local(request.node):
        return
    monkeypatch.setenv('RESONANCES_BENCHMARK_DATA', str(NO_LOCAL_DATA))
    bio = sys.modules.get('bench_io')
    if bio is not None:
        for name in ('DATA', 'EXAMPLES', 'RUNS', 'REVIEW'):
            monkeypatch.setattr(bio, name, NO_LOCAL_DATA if name == 'DATA' else NO_LOCAL_DATA / name.lower())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """A test not marked `benchmark` that skips for missing local data fails: that skip passes silently elsewhere."""
    outcome = yield
    report = outcome.get_result()
    if not report.skipped or _is_local(item) or hasattr(report, 'wasxfail'):
        return
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    if LOCAL_DATA_MISSING.search(str(reason)):
        report.outcome = 'failed'
        report.longrepr = f'{item.nodeid} needs local-only data ({reason}): mark it benchmark (tests/CLAUDE.md)'


def pytest_collection_finish(session):
    """`make test-benchmark` without the benchmark data stops with one message, not hundreds of skips."""
    if not any(_is_local(item) for item in session.items):
        return
    if BENCHMARK_TOOLS not in sys.path:
        sys.path.insert(0, BENCHMARK_TOOLS)
    import bench_io

    if not (bench_io.DATA / 'examples').is_dir():
        pytest.exit(
            f'{sum(_is_local(i) for i in session.items)} selected tests are marked benchmark and need the benchmark data, '
            f'which is not in git and not found at {bench_io.DATA} (set RESONANCES_BENCHMARK_DATA); test-fast and '
            'test-slow do not need it',
            returncode=4,
        )
