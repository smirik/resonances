"""The build record (`<file>.meta.json`) of a cached MMR catalog and the warning when the cache was built otherwise.

The expected numbers come from resonances/.env.dist (MATRIX_2BODY_PRIMARY_MAX=12, COEF_MAX=11, ORDER_MAX=10) and from
cache/mmr-2body-order10.csv, a catalog built with those defaults earlier: 666 lines = stamp + header + 664 rows.
"""

import json
import logging

import pandas as pd
import pytest

import resonances
from resonances.matrix.matrix import Matrix
from resonances.mmr.retrograde_two_body_matrix import RetrogradeTwoBodyMatrix
from resonances.mmr.three_body_matrix import ThreeBodyMatrix
from resonances.mmr.two_body_matrix import TwoBodyMatrix

SOLAR_SYSTEM = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
DEFAULT_2BODY = {'MATRIX_2BODY_PRIMARY_MAX': 12, 'MATRIX_2BODY_COEF_MAX': 11, 'MATRIX_2BODY_ORDER_MAX': 10}


@pytest.fixture
def catalog(tmp_path, caplog):
    """Each matrix class reads its catalog from tmp_path; the config, planets and loaded tables are restored after."""
    classes = (TwoBodyMatrix, ThreeBodyMatrix, RetrogradeTwoBodyMatrix)
    keys = [cls.catalog_file for cls in classes] + [key for cls in classes for key in cls.limit_keys]
    saved = {key: resonances.config.get(key) for key in keys}
    saved_planets = {cls: cls.planets for cls in classes}
    for cls in classes:
        resonances.config.set(cls.catalog_file, str(tmp_path / f'{cls.__name__}.csv'))
        cls.planets = None
        cls.matrix = None
    caplog.set_level(logging.WARNING)
    yield tmp_path
    for key, value in saved.items():
        resonances.config.set(key, value)
    for cls in classes:
        cls.planets = saved_planets[cls]
        cls.matrix = None


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and 'MMR catalog' in r.getMessage()]


def test_a_catalog_built_with_the_defaults_is_read_without_warning(catalog, caplog):
    TwoBodyMatrix.load()  # no file yet: built and dumped with its record
    path = catalog / 'TwoBodyMatrix.csv'
    meta = json.loads((catalog / 'TwoBodyMatrix.csv.meta.json').read_text())
    assert meta['catalog'] == 'TwoBodyMatrix'
    assert meta['catalog_version'] == 2
    assert meta['limits'] == DEFAULT_2BODY
    assert meta['planets'] == SOLAR_SYSTEM
    assert meta['package_version'] == resonances.__version__
    assert len(TwoBodyMatrix.matrix) == 664
    assert list(catalog.glob('*.tmp')) == []

    TwoBodyMatrix.matrix = None
    TwoBodyMatrix.load()  # read from the cache
    assert len(TwoBodyMatrix.matrix) == 664
    assert _warnings(caplog) == []
    assert TwoBodyMatrix.check_cached(path) is None


def test_other_limits_warn_and_the_cached_candidates_are_kept(catalog, caplog):
    TwoBodyMatrix.load()
    path = catalog / 'TwoBodyMatrix.csv'
    before = path.read_bytes()
    near_hildas = sorted(m.to_short() for m in TwoBodyMatrix.find_resonances(3.97, sigma=0.1))
    assert '3J-2' in near_hildas and '1E-8' in near_hildas  # 1E-8: a = 8^(2/3) = 4 au, order 7, beyond ORDER_MAX=4

    resonances.config.set('MATRIX_2BODY_ORDER_MAX', '4')
    TwoBodyMatrix.matrix = None
    TwoBodyMatrix.load()

    (message,) = _warnings(caplog)
    assert str(path) in message
    assert 'MATRIX_2BODY_ORDER_MAX: cached 10, current 4' in message
    assert 'MATRIX_2BODY_PRIMARY_MAX' not in message and 'MATRIX_2BODY_COEF_MAX' not in message
    assert 'load(reload=True)' in message and 'MATRIX_2BODY_FILE' in message
    # Only a warning: the same file, the same table, the same candidates.
    assert path.read_bytes() == before
    assert sorted(m.to_short() for m in TwoBodyMatrix.find_resonances(3.97, sigma=0.1)) == near_hildas
    assert len(TwoBodyMatrix.matrix) == 664
    assert json.loads((catalog / 'TwoBodyMatrix.csv.meta.json').read_text())['limits'] == DEFAULT_2BODY

    # A rebuild on request writes the current limits: no warning any more.
    TwoBodyMatrix.load(reload=True)
    assert '1E-8' not in TwoBodyMatrix.matrix['mmr'].tolist()
    assert TwoBodyMatrix.check_cached(path) is None


def test_other_planets_warn(catalog, caplog):
    ThreeBodyMatrix.planets = ['Jupiter', 'Saturn']
    ThreeBodyMatrix.load()
    rows = len(ThreeBodyMatrix.matrix)

    ThreeBodyMatrix.planets = ['Saturn', 'Jupiter']  # the same set: no warning
    ThreeBodyMatrix.matrix = None
    ThreeBodyMatrix.load()
    assert _warnings(caplog) == []

    ThreeBodyMatrix.planets = None
    ThreeBodyMatrix.matrix = None
    ThreeBodyMatrix.load()
    (message,) = _warnings(caplog)
    assert f"planets: cached ['Jupiter', 'Saturn'], current {SOLAR_SYSTEM}" in message
    assert '_MAX: cached' not in message  # no limit differs
    assert len(ThreeBodyMatrix.matrix) == rows
    assert set(ThreeBodyMatrix.matrix['planet1']) | set(ThreeBodyMatrix.matrix['planet2']) == {'Jupiter', 'Saturn'}


def test_a_catalog_without_a_record_warns_once_that_its_limits_are_unknown(catalog, caplog):
    path = catalog / 'TwoBodyMatrix.csv'
    # A cache written before the build records: the current stamp, a single row, no .meta.json.
    path.write_text('# TwoBodyMatrix v2\n,mmr,planet,m1,m,q,a\n0,5J-2,Jupiter,5,-2,3,2.8253\n')

    TwoBodyMatrix.load()
    TwoBodyMatrix.matrix = None
    TwoBodyMatrix.load()

    (message,) = _warnings(caplog)
    assert str(path) in message and 'unknown' in message
    assert 'MATRIX_2BODY_PRIMARY_MAX=12, MATRIX_2BODY_COEF_MAX=11, MATRIX_2BODY_ORDER_MAX=10' in message
    assert [m.to_short() for m in TwoBodyMatrix.find_resonances(2.8, sigma=0.1)] == ['5J-2']
    assert not (catalog / 'TwoBodyMatrix.csv.meta.json').exists()


def test_an_unreadable_record_counts_as_none(catalog):
    path = catalog / 'TwoBodyMatrix.csv'
    path.write_text('# TwoBodyMatrix v2\n,mmr,planet,m1,m,q,a\n')
    Matrix.meta_filename(path).write_text('{not json')
    assert TwoBodyMatrix.build_mismatches(path) is None
    assert 'unknown' in TwoBodyMatrix.check_cached(path)


def test_the_retrograde_catalog_is_rebuilt_on_other_limits_and_its_record_follows(catalog, caplog):
    """Its stamp holds the limits: a change rebuilds it (as before), and the new record matches."""
    RetrogradeTwoBodyMatrix.planets = ['Jupiter']
    RetrogradeTwoBodyMatrix.load()
    assert len(RetrogradeTwoBodyMatrix.matrix) == 1198  # Li et al. (2019), see test_retrograde_two_body_matrix.py

    resonances.config.set('MATRIX_2BODY_RETRO_ORDER_MAX', '4')
    RetrogradeTwoBodyMatrix.matrix = None
    RetrogradeTwoBodyMatrix.load()
    meta = json.loads((catalog / 'RetrogradeTwoBodyMatrix.csv.meta.json').read_text())
    assert meta['limits'] == {'MATRIX_2BODY_RETRO_COEF_MAX': 15, 'MATRIX_2BODY_RETRO_ORDER_MAX': 4}
    # p + q <= 4, coprime: (1,1) k=0..1, (1,2) and (2,1) k=0..1, (1,3) and (3,1) k=0..2: 2 + 4 + 6 = 12 rows
    assert len(RetrogradeTwoBodyMatrix.matrix) == 12

    RetrogradeTwoBodyMatrix.matrix = None
    RetrogradeTwoBodyMatrix.load()
    assert _warnings(caplog) == []


def test_the_csv_read_is_the_cached_file(catalog):
    """The check reads the record only: the table loaded is the cached CSV as it is."""
    TwoBodyMatrix.load()
    path = catalog / 'TwoBodyMatrix.csv'
    resonances.config.set('MATRIX_2BODY_COEF_MAX', '3')
    TwoBodyMatrix.matrix = None
    TwoBodyMatrix.load()
    pd.testing.assert_frame_equal(TwoBodyMatrix.matrix, pd.read_csv(path, skiprows=1))
