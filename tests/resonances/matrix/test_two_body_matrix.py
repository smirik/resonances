import pytest
import pandas as pd
from pathlib import Path
import shutil

import resonances
from resonances.matrix.two_body_matrix import TwoBodyMatrix


@pytest.fixture(autouse=True)
def run_around_tests():
    resonances.config.set(TwoBodyMatrix.catalog_file, 'cache/tests/mmr-2body-test.csv')
    Path('cache/tests').mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree('cache/tests')


def test_build():
    TwoBodyMatrix.planets = ['Jupiter']  # for performance
    df = TwoBodyMatrix.build()
    assert isinstance(df, pd.DataFrame) is True
    assert 0 == len(df.loc[df['planet'] == 'Mars'])

    asteroid = df.loc[(df['mmr'] == '1J-1')].iloc[0]
    assert 5.2043 == pytest.approx(asteroid['a'], 0.01)

    asteroid = df.loc[(df['m1'] == 1) & (df['m'] == -1) & (df['planet'] == 'Jupiter')].iloc[0]
    assert 5.2043 == pytest.approx(asteroid['a'], 0.01)

    asteroid = df.loc[(df['mmr'] == '3J-2')].iloc[0]
    assert 3.9716 == pytest.approx(asteroid['a'], 0.01)

    asteroid = df.loc[(df['mmr'] == '11J-5')].iloc[0]
    assert 3.0766 == pytest.approx(asteroid['a'], 0.01)


def test_dump():
    TwoBodyMatrix.planets = ['Jupiter']  # for performance
    TwoBodyMatrix.dump()
    assert Path(resonances.config.get(TwoBodyMatrix.catalog_file)).is_file() is True


def test_load():
    TwoBodyMatrix.planets = ['Jupiter']  # for performance
    TwoBodyMatrix.load()
    df = TwoBodyMatrix.matrix
    asteroid = df.loc[(df['mmr'] == '1J-1')].iloc[0]
    assert 5.2043 == pytest.approx(asteroid['a'], 0.01)

    del df
    TwoBodyMatrix.matrix = None
    TwoBodyMatrix.load(reload=True)
    df = TwoBodyMatrix.matrix
    asteroid = df.loc[(df['mmr'] == '1J-1')].iloc[0]
    assert 5.2043 == pytest.approx(asteroid['a'], 0.01)


def test_find_resonances():
    mmrs = TwoBodyMatrix.find_resonances(5.20, sigma=0.1)
    mmrs_list = [mmr.to_short() for mmr in mmrs]
    assert '1J-1' in mmrs_list

    mmrs = TwoBodyMatrix.find_resonances(5.20, sigma=0.1, planets=['Mercury', 'Venus', 'Earth', 'Mars', 'Uranus', 'Neptune'])
    mmrs_list = [mmr.to_short() for mmr in mmrs]
    assert '1J-1' not in mmrs_list

    mmrs = TwoBodyMatrix.find_resonances(3.97, sigma=0.1)
    mmrs_list = [mmr.to_short() for mmr in mmrs]
    assert '3J-2' in mmrs_list
    assert '1J-1' not in mmrs_list
