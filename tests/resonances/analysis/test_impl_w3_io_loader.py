"""The benchmark loader reads only the columns it needs (W3-M3: the whole DataFrame of 1e6 rows x 73 planet columns
peaked at 2.6 GB). Written by hand here: a forward and a backward run, legacy column names, and an empty column
(numpy cannot parse it, pandas reads it as NaN)."""

import gzip
import json

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import io

SERIES = ('t_yr', 'a', 'e', 'inc', 'Omega', 'omega', 'varpi', 'M', 'lam', 'angle_raw')
PLANET_FIELDS = ('a', 'e', 'inc', 'Omega', 'omega', 'M', 'l', 'varpi')


def _write(folder, t, empty_mean_anomaly=False):
    """Columns are simple functions of t so every value can be checked: a = 5.2 + t/1e6, angle = 0.01 t, ..."""
    folder.mkdir()
    rows = [','.join(SERIES)]
    for x in t:
        values = [x, 5.2 + x / 1e6, 0.1, 0.02, 1.0, 2.0, 3.0, None if empty_mean_anomaly else 0.5, 0.25 * x, 0.01 * x]
        rows.append(','.join('' if v is None else repr(float(v)) for v in values))
    with gzip.open(folder / 'series.csv.gz', 'wt') as f:
        f.write('\n'.join(rows) + '\n')
    columns = ['t_yr'] + [f'{p}.{field}' for p in ('Jupiter', 'Saturn') for field in PLANET_FIELDS]
    rows = [','.join(columns)]
    for x in t:
        jupiter = [5.2, 0.048, 0.02, 1.75, 4.8, 0.3, 0.529 * x, 0.25]
        saturn = [9.5, 0.054, 0.04, 1.98, 5.9, 0.4, 0.213 * x, 1.6]
        rows.append(','.join(repr(float(v)) for v in [x] + jupiter + saturn))
    with gzip.open(folder / 'planets.csv.gz', 'wt') as f:
        f.write('\n'.join(rows) + '\n')
    meta = {'name': 'synthetic', 'resonance': '5J-2+0-3', 'resonance_type': 'mmr', 'regime': {'integrator': 'SABA', 'dt': 0.1}}
    (folder / 'meta.json').write_text(json.dumps(meta))
    return folder


@pytest.mark.parametrize('backward', [False, True])
def test_columns_by_hand(tmp_path, backward):
    t = np.arange(0.0, 50.0, 1.0)
    folder = _write(tmp_path / 'example', -t if backward else t)
    series = io.load_series(folder)
    expected_t = -t[::-1] if backward else t  # a backward run comes back in increasing time
    np.testing.assert_array_equal(series.t_yr, expected_t)
    np.testing.assert_allclose(series.a, 5.2 + expected_t / 1e6, rtol=0, atol=1e-15)
    np.testing.assert_allclose(series.angle_unwrapped_numpy, 0.01 * expected_t, rtol=0, atol=1e-15)
    np.testing.assert_allclose(series.mean_longitude, 0.25 * expected_t)  # legacy `lam`
    assert series.mean_anomaly is not None and np.all(series.mean_anomaly == 0.5)
    assert sorted(series.planets) == ['Jupiter', 'Saturn']
    np.testing.assert_allclose(series.planets['Jupiter'].mean_longitude, 0.529 * expected_t)  # legacy `.l`
    np.testing.assert_array_equal(series.planets['Saturn'].varpi, np.full(t.size, 1.6))
    assert series.regime.n_rows == t.size and series.regime.step_yr == 1.0


def test_an_empty_column_is_read_as_missing(tmp_path):
    t = np.arange(0.0, 30.0, 2.0)
    series = io.load_series(_write(tmp_path / 'example', t, empty_mean_anomaly=True))
    assert series.mean_anomaly is None  # the column is all NaN: not saved
    np.testing.assert_allclose(series.a, 5.2 + t / 1e6, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(series.planets['Jupiter'].a, np.full(t.size, 5.2))
