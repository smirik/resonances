import pytest
import pandas as pd
from pathlib import Path
import shutil

import resonances
from resonances.matrix.three_body_matrix import ThreeBodyMatrix


@pytest.fixture(autouse=True)
def run_around_tests():
    resonances.config.set('matrix.3body.file', 'cache/tests/mmr-3body-test.csv')
    Path('cache/tests').mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree('cache/tests')


def test_build():
    ThreeBodyMatrix.planets = ['Jupiter', 'Saturn']  # for performance
    df = ThreeBodyMatrix.build()
    assert isinstance(df, pd.DataFrame) is True
    assert 0 == len(df.loc[df['planet1'] == 'Mars'])

    asteroid = df.loc[(df['mmr'] == '4J-2S-1')].iloc[0]
    assert 2.3981 == pytest.approx(asteroid['a'], 0.01)

    asteroid = df.loc[
        (df['m1'] == 4) & (df['m2'] == -2) & (df['m'] == -1) & (df['planet1'] == 'Jupiter') & (df['planet2'] == 'Saturn')
    ].iloc[0]
    assert 2.3981 == pytest.approx(asteroid['a'], 0.01)

    asteroid = df.loc[(df['mmr'] == '5J-2S-2')].iloc[0]
    assert 3.1746 == pytest.approx(asteroid['a'], 0.01)

    asteroid = df.loc[(df['mmr'] == '3J-2S-1')].iloc[0]
    assert 3.0801 == pytest.approx(asteroid['a'], 0.01)


def test_dump():
    ThreeBodyMatrix.planets = ['Jupiter', 'Saturn']  # for performance
    ThreeBodyMatrix.dump()
    assert Path(resonances.config.get(ThreeBodyMatrix.catalog_file)).is_file() is True


def test_load():
    ThreeBodyMatrix.planets = ['Jupiter', 'Saturn']  # for performance
    ThreeBodyMatrix.load()
    df = ThreeBodyMatrix.matrix
    asteroid = df.loc[(df['mmr'] == '4J-2S-1')].iloc[0]
    assert 2.3981 == pytest.approx(asteroid['a'], 0.01)

    del df
    ThreeBodyMatrix.matrix = None
    ThreeBodyMatrix.load(reload=True)
    df = ThreeBodyMatrix.matrix
    asteroid = df.loc[(df['mmr'] == '4J-2S-1')].iloc[0]
    assert 2.3981 == pytest.approx(asteroid['a'], 0.01)


def test_find_resonances():
    mmrs = ThreeBodyMatrix.find_resonances(2.39, sigma=0.1)
    mmrs_list = [mmr.to_short() for mmr in mmrs]
    assert '4J-2S-1' in mmrs_list

    mmrs = ThreeBodyMatrix.find_resonances(2.39, sigma=0.1, planets=['Mercury', 'Venus', 'Earth', 'Mars', 'Uranus', 'Neptune'])
    mmrs_list = [mmr.to_short() for mmr in mmrs]
    assert '4J-2S-1' not in mmrs_list

    mmrs = ThreeBodyMatrix.find_resonances(3.17, sigma=0.1)
    mmrs_list = [mmr.to_short() for mmr in mmrs]
    assert '5J-2S-2' in mmrs_list
    assert '4J-2S-1' not in mmrs_list
