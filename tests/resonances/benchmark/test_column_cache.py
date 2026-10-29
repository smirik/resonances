"""benchmark/tools/column_cache.py: the binary copy gives the loader's arrays bit for bit, follows the CSV when it
changes, and the worker count rule of bench_io. Fast: synthetic CSVs (plus one saved example when the data is there)."""

import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import bench_io as bio  # noqa: E402
import column_cache as cc  # noqa: E402

from resonances.resonance.analysis.mmr import io  # noqa: E402

SERIES = ['t_yr', 'a', 'e', 'inc', 'Omega', 'omega', 'varpi', 'M', 'lam', 'angle_raw']  # legacy `lam`


def bits(values):
    return np.ascontiguousarray(values).view(np.uint64).tobytes()


def assert_identical(got, ref):
    assert set(got) == set(ref)
    for name in ref:
        assert got[name].dtype == np.float64
        assert bits(got[name]) == bits(ref[name]), name


def write_csv(path, header, rows):
    with gzip.open(path, 'wt') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(row) + '\n')


def series_rows(n=50, empty_m=False):
    rng = np.random.default_rng(1)
    values = rng.normal(size=(n, len(SERIES))) * 10.0 ** rng.integers(-8, 8, size=(n, len(SERIES)))
    values[:, 0] = np.arange(n) * 0.5
    text = [[f'{v:.12g}' for v in row] for row in values]
    if empty_m:
        for row in text:
            row[SERIES.index('M')] = ''
    return text


@pytest.mark.parametrize('empty_m', [False, True])  # numpy's reader, and pandas' when a column is empty
def test_cached_columns_are_bit_identical(tmp_path, empty_m):
    path = tmp_path / 'series.csv.gz'
    write_csv(path, SERIES, series_rows(empty_m=empty_m))
    root = tmp_path / 'cache'
    ref = io._read_columns(path, io.SERIES_COLUMNS)
    assert 'mean_longitude' in ref and 'lam' not in ref
    first = cc.read_columns(path, io.SERIES_COLUMNS, root=root)  # writes the copy
    assert cc.cache_path(path, root).exists()
    second = cc.read_columns(path, io.SERIES_COLUMNS, root=root, reader=pytest.fail)  # must not parse the CSV
    assert_identical(first, ref)
    assert_identical(second, ref)
    assert np.isnan(second['M']).all() == empty_m


def test_subset_absent_and_uncached_columns(tmp_path):
    header = ['t_yr'] + [f'{p}.{f}' for p in ('Mars', 'Jupiter') for f in ('a', 'e', 'inc', 'Omega', 'omega', 'M', 'l', 'varpi')]
    rng = np.random.default_rng(2)
    write_csv(tmp_path / 'planets.csv.gz', header, [[f'{v:.12g}' for v in rng.normal(size=len(header))] for _ in range(30)])
    path = tmp_path / 'planets.csv.gz'
    root = tmp_path / 'cache'
    wanted = ['t_yr', 'Jupiter.a', 'Jupiter.mean_longitude', 'Saturn.a']  # Saturn is not in the file
    got = cc.read_columns(path, wanted, root=root)
    assert_identical(got, io._read_columns(path, wanted))
    assert_identical(cc.read_columns(path, wanted, root=root, reader=pytest.fail), got)
    # omega is not a field the loader reads: not kept, so the CSV answers
    assert_identical(cc.read_columns(path, ['Mars.omega'], root=root), io._read_columns(path, ['Mars.omega']))


def test_a_changed_csv_is_read_again(tmp_path):
    path = tmp_path / 'series.csv.gz'
    root = tmp_path / 'cache'
    write_csv(path, SERIES, series_rows(40))
    cc.read_columns(path, io.SERIES_COLUMNS, root=root)
    write_csv(path, SERIES, series_rows(60))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    got = cc.read_columns(path, io.SERIES_COLUMNS, root=root)
    assert len(got['t_yr']) == 60
    assert_identical(got, io._read_columns(path, io.SERIES_COLUMNS))


def test_install_routes_the_loader_and_can_be_switched_off(monkeypatch):
    session = cc._original is not None  # tests/conftest.py installs the copy for the session
    cc.uninstall()
    original = io._read_columns
    try:
        monkeypatch.setenv(cc.ENV, '0')
        assert cc.install() is False and io._read_columns is original
        monkeypatch.setenv(cc.ENV, '1')
        assert cc.install() is True and io._read_columns is cc.read_columns
        assert cc.install() is True and cc._original is original  # idempotent
    finally:
        cc.uninstall()
        assert io._read_columns is original
        if session:
            cc.install()


@pytest.mark.benchmark
def test_a_saved_example_loads_identically(tmp_path):
    ids = io.list_examples('mmr')
    if not ids:
        pytest.skip('no saved benchmark examples')
    example = next((i for i in ids if (io.example_path(i) / 'planets.csv.gz').exists() and i.startswith('coorbital')), ids[0])
    folder = io.example_path(example)
    for name in ('series.csv.gz', 'planets.csv.gz'):
        path = folder / name
        if not path.exists():
            continue
        header = list(pd.read_csv(path, nrows=0).columns)
        wanted = [n for n in io._current_names(header).values() if cc.is_cacheable(n)]
        ref = io._read_columns(path, wanted)
        cc.read_columns(path, wanted, root=tmp_path)
        assert_identical(cc.read_columns(path, wanted, root=tmp_path, reader=pytest.fail), ref)


def test_auto_workers_rule(monkeypatch):
    gb = 1024**3
    for allocator in ('off', 'auto', 'on'):
        assert bio.auto_workers(cpus=18, ram_bytes=64 * gb, allocator=allocator) == 8  # the cap
        assert bio.auto_workers(cpus=4, ram_bytes=64 * gb, allocator=allocator) == 2  # CPUs - 2
        assert bio.auto_workers(cpus=2, ram_bytes=1 * gb, allocator=allocator) == 1  # at least one
    assert bio.auto_workers(cpus=18, ram_bytes=20 * gb, allocator='off') == 6  # floor(20 / 3)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    # the memory limits the count at 3 GB a worker: with the low-memory allocator 1.6 GB a worker, floor(20 / 1.6)
    assert bio.auto_workers(cpus=18, ram_bytes=20 * gb, allocator='auto') == 8
    assert bio.auto_workers(cpus=18, ram_bytes=12 * gb, allocator='auto') == 7
    assert bio.auto_workers(cpus=8, ram_bytes=8 * gb, allocator='auto') == 5  # an 8 GB Mac: 5 workers instead of 2
    assert bio.workers_allocator(6, ram_bytes=64 * gb, allocator='auto') is False  # 18 GB fit
    assert bio.workers_allocator(6, ram_bytes=12 * gb, allocator='auto') is True
    assert bio.workers_allocator(6, ram_bytes=64 * gb, allocator='on') is True
    monkeypatch.setattr(sys, 'platform', 'linux')  # no policy measured there
    assert bio.auto_workers(cpus=18, ram_bytes=20 * gb, allocator='on') == 6
    assert bio.workers_allocator(6, ram_bytes=12 * gb, allocator='on') is False
    assert bio.resolve_workers('5') == 5
    with pytest.raises(ValueError):
        bio.resolve_workers('0')


def test_planet_columns_are_read_only_maps_and_series_columns_are_arrays(tmp_path):
    """W23 P5: a planet's column comes back as a read-only memory map of its .npy (a write raises), the same bits on the
    first read and the later ones; the body's columns are ordinary arrays."""
    header = ['t_yr'] + [f'Jupiter.{f}' for f in ('a', 'e', 'inc', 'Omega', 'omega', 'M', 'l', 'varpi')]
    rng = np.random.default_rng(3)
    path = tmp_path / 'planets.csv.gz'
    write_csv(path, header, [[f'{v:.12g}' for v in rng.normal(size=len(header))] for _ in range(40)])
    wanted = ['t_yr', 'Jupiter.a', 'Jupiter.Omega', 'Jupiter.varpi']
    ref = io._read_columns(path, wanted)
    root = tmp_path / 'cache'
    for got in (cc.read_columns(path, wanted, root=root), cc.read_columns(path, wanted, root=root, reader=pytest.fail)):
        assert_identical(got, ref)
        assert type(got['Jupiter.a']) is np.ndarray and not got['Jupiter.a'].flags.writeable
        with pytest.raises(ValueError):
            got['Jupiter.a'][0] = 0.0
        assert got['t_yr'].flags.writeable
    series = tmp_path / 'series.csv.gz'
    write_csv(series, SERIES, series_rows())
    got = cc.read_columns(series, io.SERIES_COLUMNS, root=root)
    assert all(values.flags.writeable for values in got.values())


def test_columns_that_differ_only_in_case_stay_apart(tmp_path):
    """Omega and omega are two files on a case-insensitive file system too (macOS): the names are escaped."""
    path = tmp_path / 'series.csv.gz'
    write_csv(path, SERIES, series_rows())
    root = tmp_path / 'cache'
    cc.read_columns(path, io.SERIES_COLUMNS, root=root)
    got = cc.read_columns(path, io.SERIES_COLUMNS, root=root, reader=pytest.fail)
    assert_identical(got, io._read_columns(path, io.SERIES_COLUMNS))
    assert bits(got['Omega']) != bits(got['omega'])
    for name in ('Omega', 'omega', 'Jupiter.Omega', 't_yr', 'mean_longitude'):
        assert cc._column_name(cc._file_stem(name)) == name
    assert cc._file_stem('Omega').lower() != cc._file_stem('omega').lower()


def test_the_default_cache_keeps_only_the_benchmark_files(tmp_path):
    """A CSV outside the data root (a test's temporary file) is read directly when no cache root is given."""
    path = tmp_path / 'series.csv.gz'
    write_csv(path, SERIES, series_rows())
    seen = []

    def reader(p, wanted):
        seen.append(p)
        return io._read_columns(p, wanted)

    for _ in range(2):
        assert_identical(cc.read_columns(path, io.SERIES_COLUMNS, reader=reader), io._read_columns(path, io.SERIES_COLUMNS))
    assert len(seen) == 2 and not cc.cache_path(path).exists()


def test_longest_first_from_the_previous_run(tmp_path):
    """W23 P7: the pool takes the examples longest first by the timings of a previous run.json; untimed ones first,
    ties in their order; a run that timed fewer than half of the ids is not used."""
    import json

    run = tmp_path / 'run.json'
    run.write_text(json.dumps({'timings': [{'example_id': i, 'seconds': s} for i, s in (('a', 1.0), ('b', 9.0), ('c', 1.0), ('d', 4.0))]}))
    assert bio.longest_first(['a', 'b', 'c', 'd', 'new'], runs=[run]) == ['new', 'b', 'd', 'a', 'c']
    assert bio.previous_timings(['a', 'b', 'c', 'd'], runs=[run]) == {'a': 1.0, 'b': 9.0, 'c': 1.0, 'd': 4.0}
    few = tmp_path / 'few.json'
    few.write_text(json.dumps({'timings': [{'example_id': 'a', 'seconds': 1.0}]}))
    ids = ['x', 'y', 'z', 'a']
    assert bio.previous_timings(ids, runs=[few, run]) == {}  # no run (nor one under DATA) timed half of these ids
