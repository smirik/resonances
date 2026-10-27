"""The runs of `make test-benchmark`, shared by its gates: one reclassification of every saved MMR and co-orbital example
(`benchmark/tools/reclassify.py` into benchmark/data/test-benchmark) and one of every secular and Lidov-Kozai example
(`benchmark/tools/reclassify_secular_zlk.py` into benchmark/data/test-benchmark-secular-zlk, a few seconds) per session,
read by the disagreement gate (test_classify_benchmark.py) and the episode gate (test_episode_expectations.py). Skipped
without the saved series in benchmark/data or $RESONANCES_BENCHMARK_DATA."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import bench_io as bio  # noqa: E402

RUN = 'test-benchmark'
SECULAR_ZLK_RUN = 'test-benchmark-secular-zlk'


def _reclassify(script: str, run: str) -> Path:
    """Run a reclassification tool into benchmark/data/<run> (skipped without the saved examples)."""
    if not (bio.DATA / 'examples').is_dir():
        pytest.skip(f'no saved benchmark examples in {bio.DATA}')
    out = bio.DATA / run
    workers = bio.resolve_workers()
    start = time.perf_counter()
    subprocess.run(
        [sys.executable, str(TOOLS / script), '--workers', str(workers), '--out', str(out)],
        check=True,
        cwd=TOOLS.parents[1],
        env={**os.environ, 'PYTHONUNBUFFERED': '1'},
    )
    print(f'\n{script}: {workers} workers, {time.perf_counter() - start:.0f} s')
    return out


@pytest.fixture(scope='session')
def reclassified_run() -> Path:
    """The folder of this session's MMR and co-orbital reclassification (written on first use)."""
    return _reclassify('reclassify.py', RUN)


@pytest.fixture(scope='session')
def secular_zlk_run() -> Path:
    """The folder of this session's secular and Lidov-Kozai reclassification (written on first use)."""
    return _reclassify('reclassify_secular_zlk.py', SECULAR_ZLK_RUN)
