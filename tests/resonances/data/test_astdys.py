import resonances
from resonances.data.astdys import astdys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import shutil


@pytest.fixture(autouse=True)
def run_around_tests():
    resonances.config.set('catalog', 'cache/tests/small.csv')
    resonances.config.set('astdys.catalog', 'tests/fixtures/small.cat')
    Path('cache/tests').mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree('cache/tests')


def test_required_config_values():
    assert True == resonances.config.has('catalog')
    assert True == resonances.config.has('catalog.date')
    assert True == resonances.config.has('astdys.catalog.url')
    assert True == resonances.config.has('astdys.catalog')
    assert True == resonances.config.has('astdys.date')


def test_transform_astdys_catalog():
    cat = astdys.transform_astdys_catalog()
    assert 'a' in cat
    assert 'e' in cat
    assert 'inc' in cat
    assert 'omega' in cat
    assert 'Omega' in cat
    assert 'M' in cat

    assert 10 == len(cat)

    assert 2.766 == pytest.approx(cat['a'].iloc[0], 0.01)
    assert 0.07816 == pytest.approx(cat['e'].iloc[0], 0.01)

    assert '6' == cat['num'].iloc[5]
    assert 2.42456 == pytest.approx(cat['a'].iloc[5], 0.01)
    assert 0.20328 == pytest.approx(cat['e'].iloc[5], 0.01)
    assert 14.73973 == pytest.approx(cat['inc'].iloc[5] / np.pi * 180, 0.01)
    assert 138.64293 == pytest.approx(cat['Omega'].iloc[5] / np.pi * 180, 0.01)
    assert 239.70765 == pytest.approx(cat['omega'].iloc[5] / np.pi * 180, 0.01)
    assert 242.94481 == pytest.approx(cat['M'].iloc[5] / np.pi * 180, 0.01)


def test_check_or_build_catalog():
    astdys.check_or_build_catalog()
    assert True == Path(resonances.config.get('catalog')).is_file()

    cat = pd.read_csv('tests/fixtures/small.csv')
    assert 10 == len(cat)
    assert 2.766 == pytest.approx(cat['a'].iloc[0], 0.01)
    assert 0.07816 == pytest.approx(cat['e'].iloc[0], 0.01)
    assert 6 == cat['num'].iloc[5]
    assert 2.42456 == pytest.approx(cat['a'].iloc[5], 0.01)


def test_search():
    resonances.config.set('catalog', 'tests/fixtures/small.csv')
    obj = astdys.search(6)
    assert 2.42456 == pytest.approx(obj['a'], 0.01)
    assert 0.20328 == pytest.approx(obj['e'], 0.01)
    assert 14.73973 == pytest.approx(obj['inc'] / np.pi * 180, 0.01)
    assert 138.64293 == pytest.approx(obj['Omega'] / np.pi * 180, 0.01)
    assert 239.70765 == pytest.approx(obj['omega'] / np.pi * 180, 0.01)
    assert 242.94481 == pytest.approx(obj['M'] / np.pi * 180, 0.01)

    obj = astdys.search(10)
    assert not None == obj
    obj = astdys.search(11)
    assert None == obj
    obj = astdys.search(123456789)
    assert None == obj
