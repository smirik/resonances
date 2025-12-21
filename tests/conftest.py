import os
from pathlib import Path
import shutil
import sys

import pytest

import resonances


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@pytest.fixture(autouse=True, scope="session")
def setup_test_config():
    resonances.config.set('SAVE_PATH', 'cache/tests')
    resonances.config.set('PLOT_PATH', 'cache/tests')


@pytest.fixture(autouse=True, scope="function")
def cleanup_test_files():
    Path('cache/tests').mkdir(parents=True, exist_ok=True)
    """Clean up test cache after each test."""
    yield  # Test runs

    cache_path = Path('cache/tests')
    if cache_path.exists():
        shutil.rmtree(cache_path)
