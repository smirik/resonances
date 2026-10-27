"""W23 S1, memory and speed without changing a bit: the record memo of the filtered series (P2), the owner's member
columns formed one at a time (P3), the co-orbital table in preallocated buffers (P4), one read of a run's planets file
(perf candidate 30) and the low-memory allocator of the workers (P1). Each is checked against an independent
computation: the plain expression, a second filter call outside the memo, a hand-written file."""

import gzip
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pytest

from resonances.config import low_memory_allocator, worker_allocator
from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.mmr import coorbital, io, owner, response
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()


# --- P2: the record memo ------------------------------------------------------------------------------------------------


def _record(n=4000, step=0.5):
    """A pendulum-like angle with a 7-yr ripple, and a with the same ripple (t in yr)."""
    t = np.arange(n) * step
    angle = 0.8 * np.sin(2 * np.pi * t / 300.0) + 0.05 * np.sin(2 * np.pi * t / 7.0)
    a = 2.5 + 1e-4 * np.cos(2 * np.pi * t / 300.0) + 1e-6 * np.sin(2 * np.pi * t / 7.0)
    return t, angle, a


def test_band_pass_in_the_memo_is_the_band_pass_outside_it():
    t, _, a = _record()
    outside = filtering.band_pass(t, a, 300.0, 2.0, S.filter_order)
    assert outside.flags.writeable  # no memo: an ordinary array
    with filtering.record_memo():
        first = filtering.band_pass(t, a, 300.0, 2.0, S.filter_order)
        again = filtering.band_pass(t, a, 300.0, 2.0, S.filter_order)
        other = filtering.band_pass(t, a, 150.0, 2.0, S.filter_order)
        copy = filtering.band_pass(t, a.copy(), 300.0, 2.0, S.filter_order)  # equal values, another object: computed
        assert again is first and other is not first and copy is not first
        assert not first.flags.writeable  # what a later caller reads cannot be changed
        with pytest.raises(ValueError):
            first[0] = 0.0
    np.testing.assert_array_equal(first, outside)
    np.testing.assert_array_equal(copy, outside)
    # the independent computation: the band is the difference of the two low-passes
    low = filtering.lowpass(t, a, 150.0, S.filter_order) - filtering.lowpass(t, a, 600.0, S.filter_order)
    np.testing.assert_array_equal(first, low)


def test_the_memo_keeps_two_results_and_ends_with_its_scope():
    calls = []

    def compute(value):
        calls.append(value)
        return np.full(3, value)

    key = object()
    with filtering.record_memo():
        with filtering.record_memo():  # re-entrant: the inner scope is the outer one
            for value in (1.0, 2.0, 1.0, 3.0, 1.0):
                filtering.memoised('test', (key,), (value,), lambda v=value: compute(v), lambda r: (r,))
        filtering.memoised('test', (key,), (1.0,), lambda: compute(1.0), lambda r: (r,))
    # 1, 2 computed; 1 hit; 3 computed (2 out: two entries); 1 hit; after the inner scope still the same memo: 1 hit
    assert calls == [1.0, 2.0, 3.0]
    filtering.memoised('test', (key,), (1.0,), lambda: compute(1.0), lambda r: (r,))  # no scope: computed
    assert calls == [1.0, 2.0, 3.0, 1.0]
    assert filtering._RECORD_MEMO is None


class _Prep:
    """The fields of `Preprocessed` that `smoothed_series` reads."""

    def __init__(self, t, angle, a, cutoff_yr):
        self.t_yr, self.angle_f, self.a_f = t, angle, a
        self.rate_f, self.adot_f = filtering.time_derivative(t, angle), filtering.time_derivative(t, a)
        self.cutoff = type('Cutoff', (), {'cutoff_yr': cutoff_yr})()


def test_smoothed_series_in_the_memo():
    t, angle, a = _record()
    prep = _Prep(t, angle, a, 5.0)
    expected = response.smoothed_series(prep, 30.0, S)
    with filtering.record_memo():
        got = response.smoothed_series(prep, 30.0, S)
        assert response.smoothed_series(prep, 30.0, S) is got
        unfiltered = response.smoothed_series(prep, 3.0, S)  # below P_c: prep's own series, left as they are
    for x, y in zip(got[:4], expected[:4]):
        np.testing.assert_array_equal(x, y)
        assert not x.flags.writeable
    # the independent computation: the low-pass of prep's series at 30 yr and its derivative
    np.testing.assert_array_equal(got[0], filtering.lowpass_with(t, angle, 30.0, S))
    np.testing.assert_array_equal(got[3], filtering.time_derivative(t, filtering.lowpass_with(t, a, 30.0, S)))
    assert unfiltered[0] is prep.angle_f and prep.angle_f.flags.writeable and prep.adot_f.flags.writeable


# --- P3: the owner's member columns, one at a time -----------------------------------------------------------------------


def test_resolved_windows_reads_columns_from_a_generator():
    t = np.arange(0.0, 200.0, 1.0)
    base, theta = 0.05 * t, 0.01 * t
    windows = [(0.0, 99.0), (100.0, 199.0)]
    harmonics = (0, 1, 2, 400)  # j = 400: theta steps by 4 rad, the member is not resolved
    listed = owner.resolved_windows(t, [base + j * theta for j in harmonics], windows)
    streamed = owner.resolved_windows(t, (base + j * theta for j in harmonics), windows)
    assert listed.tolist() == streamed.tolist() == [False, False]
    assert owner.resolved_windows(t, (base + j * theta for j in (0, 1, 2)), windows).tolist() == [True, True]


# --- P4: the co-orbital table in preallocated buffers --------------------------------------------------------------------


def _table_by_expression(body, planet, n, softening):
    """The averaged potential and force written as the plain expressions (the table before W23), all rows at once."""
    nodes = 2 * np.pi * np.arange(n) / n
    r_body, dr_body = coorbital.kepler_position(body, nodes - body.phase)
    r_planet, _ = coorbital.kepler_position(planet, nodes - planet.phase)
    gx, gy, gz = (r_planet / (np.sum(r_planet**2, axis=1) ** 1.5)[:, None]).T
    xp, yp, zp = r_planet.T
    xb, yb, zb, tx, ty, tz = (np.lib.stride_tricks.sliding_window_view(np.concatenate([x, x]), n)[:n] for x in (*r_body.T, *dr_body.T))
    with np.errstate(divide='ignore', invalid='ignore'):
        dx, dy, dz = xp - xb, yp - yb, zp - zb
        inverse = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz + softening**2)
        potential = np.mean(inverse - (xb * gx + yb * gy + zb * gz), axis=1)
        force = np.mean((dx * tx + dy * ty + dz * tz) * inverse**3 - (gx * tx + gy * ty + gz * tz), axis=1)
    return potential, force


@pytest.mark.parametrize('n', [720, 721, 100, 64])  # whole chunks of 64 rows and a last partial one
def test_coorbital_table_is_the_plain_expression_bit_for_bit(n):
    rng = np.random.default_rng(n)
    for k in range(6):
        body = coorbital.Orbit(rng.uniform(0, 0.6), rng.uniform(0, 0.6), rng.uniform(0, 6.3), rng.uniform(0, 6.3))
        planet = coorbital.Orbit(rng.uniform(0, 0.1), rng.uniform(0, 0.05), rng.uniform(0, 6.3), rng.uniform(0, 6.3))
        softening = coorbital.hill_softening(10 ** rng.uniform(-6, -3), 0.5) if k % 2 else 0.0
        table = coorbital.averaged_table(body, planet, n, softening)
        potential, force = _table_by_expression(body, planet, n, softening)
        assert table.potential.tobytes() == potential.tobytes() and table.force.tobytes() == force.tobytes()


def test_coorbital_table_of_a_planet_on_its_own_orbit():
    """e = i = 0 without softening: the body sits on the planet at theta = 0 (inf/nan there, as the expression gives)."""
    orbit = coorbital.Orbit(0.0, 0.0, 0.0, 0.0)
    table = coorbital.averaged_table(orbit, orbit, 72, 0.0)
    potential, force = _table_by_expression(orbit, orbit, 72, 0.0)
    assert not np.isfinite(table.potential[0])
    assert table.potential.tobytes() == potential.tobytes() and table.force.tobytes() == force.tobytes()


# --- perf candidate 30: the run's planets file is read once ---------------------------------------------------------------


def _write_csv(path, columns, rows):
    with gzip.open(path, 'wt') as f:
        f.write(','.join(columns) + '\n')
        for row in rows:
            f.write(','.join(repr(float(v)) for v in row) + '\n')


def _example_with_run_planets(root, t, run_t):
    """An example without planets whose run's planets file (Jupiter: a = 5.2, lambda = 0.529 t) has the rows `run_t`."""
    series = ['t_yr', 'a', 'e', 'inc', 'Omega', 'omega', 'varpi', 'M', 'mean_longitude', 'angle_raw']
    folder = root / 'examples' / 'synthetic'
    folder.mkdir(parents=True)
    _write_csv(folder / 'series.csv.gz', series, [[x, 5.2, 0.1, 0.02, 1.0, 2.0, 3.0, 0.5, 0.25 * x, 0.01 * x] for x in t])
    meta = {'name': 'synthetic', 'resonance': '5J-2+0-3', 'resonance_type': 'mmr', 'provenance': {'run_id': 'r1'}}
    (folder / 'meta.json').write_text(json.dumps(meta))
    run = root / 'runs' / 'r1'
    run.mkdir(parents=True)
    fields = ('a', 'e', 'inc', 'Omega', 'omega', 'M', 'mean_longitude', 'varpi')
    _write_csv(
        run / 'planets.csv.gz',
        ['t_yr'] + [f'Jupiter.{f}' for f in fields],
        [[x, 5.2, 0.048, 0.02, 1.75, 4.8, 0.3, 0.529 * x, 0.25] for x in run_t],
    )
    return folder, run / 'planets.csv.gz'


@pytest.mark.parametrize('backward', [False, True])
def test_the_run_planets_file_is_read_once(tmp_path, monkeypatch, backward):
    monkeypatch.setenv(io.DATA_ENV, str(tmp_path))
    t = np.arange(0.0, 40.0, 1.0)
    folder, _ = _example_with_run_planets(tmp_path, t, t[::-1] if backward else t)  # the run's rows in either order
    reads = []
    original = io._read_columns

    def counting(path, wanted):
        reads.append(os.path.basename(str(path)))
        return original(path, wanted)

    monkeypatch.setattr(io, '_read_columns', counting)
    series = io.load_series(folder)
    assert reads.count('planets.csv.gz') == 1
    jupiter = series.planets['Jupiter']
    np.testing.assert_array_equal(jupiter.t_yr, t)
    np.testing.assert_allclose(jupiter.mean_longitude, 0.529 * t)
    np.testing.assert_array_equal(jupiter.a, np.full(t.size, 5.2))


def test_run_planets_on_another_grid_are_not_used(tmp_path, monkeypatch):
    monkeypatch.setenv(io.DATA_ENV, str(tmp_path))
    t = np.arange(0.0, 40.0, 1.0)
    folder, _ = _example_with_run_planets(tmp_path, t, t * 1.5)
    assert io.load_series(folder).planets == {}


# --- P1: the low-memory allocator of the workers -------------------------------------------------------------------------


@pytest.mark.parametrize(
    'platform, mode, ram_limited, expected',
    [
        ('darwin', 'on', False, True),
        ('darwin', 'off', True, False),
        ('darwin', 'auto', True, True),  # memory limits the workers
        ('darwin', 'auto', False, False),  # the CPUs do
        ('linux', 'on', True, False),  # glibc: not measured, no policy
    ],
)
def test_low_memory_allocator_policy(monkeypatch, platform, mode, ram_limited, expected):
    monkeypatch.setattr(sys, 'platform', platform)
    assert low_memory_allocator(ram_limited, mode) is expected
    with pytest.raises(ValueError):
        low_memory_allocator(ram_limited, 'sometimes')


def _child_environment(_):
    return os.environ.get('MallocLargeCache')


@pytest.mark.skipif(sys.platform != 'darwin', reason='the policy exists on macOS only')
def test_spawned_workers_inherit_the_allocator_and_the_parent_is_restored(monkeypatch):
    monkeypatch.delenv('MallocLargeCache', raising=False)
    with worker_allocator(True):
        with mp.get_context('spawn').Pool(1) as pool:
            inside = pool.map(_child_environment, [0])
    with worker_allocator(False):
        with mp.get_context('spawn').Pool(1) as pool:
            off = pool.map(_child_environment, [0])
    assert inside == ['0'] and off == [None]
    assert 'MallocLargeCache' not in os.environ
