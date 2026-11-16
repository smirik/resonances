import pytest
import pandas as pd
from pathlib import Path

import resonances
from resonances.mmr.two_body_matrix import TwoBodyMatrix
from tests.tools import TEST_CACHE


@pytest.fixture(autouse=True)
def run_around_tests():
    resonances.config.set(TwoBodyMatrix.catalog_file, f'{TEST_CACHE}/mmr-2body-test.csv')
    yield


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

    TwoBodyMatrix.planets = []  # testing empty states
    df = TwoBodyMatrix.build()
    assert isinstance(df, pd.DataFrame) is True
    assert 0 != len(df.loc[df['planet'] == 'Mars'])

    TwoBodyMatrix.planets = ['Jupiter']


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

    TwoBodyMatrix.matrix = None
    mmrs = TwoBodyMatrix.find_resonances(3.97, sigma=0.1)
    mmrs_list = [mmr.to_short() for mmr in mmrs]
    assert '3J-2' in mmrs_list
    assert '1J-1' not in mmrs_list


def test_only_prograde_resonances():
    TwoBodyMatrix.planets = []
    df = TwoBodyMatrix.build()
    assert (df['m'] < 0).all()

    TwoBodyMatrix.matrix = df
    near_hildas = [mmr.to_short() for mmr in TwoBodyMatrix.find_resonances(3.97, sigma=0.01, planets=['Jupiter'])]
    assert '3J-2' in near_hildas
    assert '3J+2' not in near_hildas
    near_kirkwood = [mmr.to_short() for mmr in TwoBodyMatrix.find_resonances(2.50, sigma=0.01, planets=['Jupiter'])]
    assert '3J-1' in near_kirkwood
    assert '3J+1' not in near_kirkwood
    TwoBodyMatrix.planets = ['Jupiter']
    TwoBodyMatrix.matrix = None


def test_a_stale_cache_is_rebuilt():
    TwoBodyMatrix.planets = ['Jupiter']
    path = Path(TwoBodyMatrix.catalog_full_filename())
    path.parent.mkdir(parents=True, exist_ok=True)
    # A catalog written before the versioned cache: no stamp, and a retrograde twin in it.
    path.write_text(',mmr,planet,m1,m,q,a\n0,3J+2,Jupiter,3,2,5,3.97\n')

    TwoBodyMatrix.matrix = None
    TwoBodyMatrix.load()

    assert path.read_text().splitlines()[0] == '# TwoBodyMatrix v2'
    assert '3J+2' not in TwoBodyMatrix.matrix['mmr'].tolist()
    assert '3J-2' in TwoBodyMatrix.matrix['mmr'].tolist()


def test_a_current_cache_is_read_not_rebuilt():
    path = Path(TwoBodyMatrix.catalog_full_filename())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# TwoBodyMatrix v2\n,mmr,planet,m1,m,q,a\n0,5J-2,Jupiter,5,-2,3,2.8253\n')

    TwoBodyMatrix.matrix = None
    TwoBodyMatrix.load()

    assert TwoBodyMatrix.matrix['mmr'].tolist() == ['5J-2']
    TwoBodyMatrix.matrix = None
    path.unlink()


def test_dump_replaces_the_file_and_leaves_no_temporary_one():
    TwoBodyMatrix.planets = ['Jupiter']
    path = Path(TwoBodyMatrix.catalog_full_filename())
    path.parent.mkdir(parents=True, exist_ok=True)
    # A cache shared through a symlink (e.g. from a git worktree) must not be written through.
    shared = path.with_name('shared-mmr-2body.csv')
    shared.write_text('shared\n')
    path.unlink(missing_ok=True)
    path.symlink_to(shared)

    TwoBodyMatrix.matrix = None
    TwoBodyMatrix.dump()

    assert shared.read_text() == 'shared\n'
    assert not path.is_symlink()
    assert path.read_text().splitlines()[0] == '# TwoBodyMatrix v2'
    assert list(path.parent.glob(path.name + '.*.tmp')) == []
    TwoBodyMatrix.matrix = None
    path.unlink()
    shared.unlink()
