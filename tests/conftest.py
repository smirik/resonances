import gc
import os
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
