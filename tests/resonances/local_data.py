"""Paths of the local-only test data: the benchmark data (`benchmark/data`, or `RESONANCES_BENCHMARK_DATA`), not in
git. A test that reads them is marked `benchmark` (tests/CLAUDE.md); the paths are resolved when the test runs, so
`tests/conftest.py` can hide the data from every other test."""

from pathlib import Path


def benchmark_data() -> Path:
    import bench_io  # benchmark/tools, on sys.path through tests/conftest.py

    return bench_io.DATA


def free_omega_fixture(name: str) -> Path:
    """A 500-kyr series of omega and the free elements of a reference body (591986, 3040, 162474, 4257), moved out of
    tests/fixtures on 2026-10-29: ~147 KB each, over the size budget of committed fixtures."""
    return benchmark_data() / 'fixtures' / 'free-omega' / f'{name}.csv.gz'
