from pathlib import Path

import pytest

import resonances
from resonances.mmr.retrograde_two_body import RetrogradeTwoBody
from resonances.mmr.retrograde_two_body_matrix import RetrogradeTwoBodyMatrix

STAMP = '# RetrogradeTwoBodyMatrix v1 MATRIX_2BODY_RETRO_COEF_MAX=15 MATRIX_2BODY_RETRO_ORDER_MAX=30'


@pytest.fixture(autouse=True)
def jupiter_only():
    resonances.config.set(RetrogradeTwoBodyMatrix.catalog_file, 'cache/tests/mmr-2body-retro-test.csv')
    RetrogradeTwoBodyMatrix.planets = ['Jupiter']
    RetrogradeTwoBodyMatrix.matrix = None
    yield
    RetrogradeTwoBodyMatrix.planets = None
    RetrogradeTwoBodyMatrix.matrix = None


def test_the_catalog_of_li_et_al():
    """p, q <= 15, p + q <= 30, coprime: 143 resonances per planet (Li et al. 2019), each with
    every k = 0..(p+q)//2: sum over the pairs of ((p+q)//2 + 1) = 1198 rows."""
    df = RetrogradeTwoBodyMatrix.build()
    assert len(df[['p', 'q']].drop_duplicates()) == 143
    assert len(df) == 1198
    assert set(df['planet']) == {'Jupiter'}
    assert (df['order'] == df['p'] + df['q']).all()

    names = set(df['mmr'])
    assert {'1J-3r0', '1J-3r1', '1J-3r2', '1J-1r0', '1J-1r1', '1J-15r8', '3J-14r0'} <= names
    assert not {'1J-3r3', '1J-1r2', '2J-4r0'} & names
    assert df.loc[df['mmr'] == '1J-3r0', 'a'].iloc[0] == pytest.approx(10.8216, abs=1e-4)


def test_find_resonances_near_2000_dg8():
    """a = 10.76 +- 0.5 au: 1/-3 (10.82) and 5/-14 (5.2025 * 2.8^(2/3) = 10.34), every k."""
    mmrs = RetrogradeTwoBodyMatrix.find_resonances(10.76, 0.5, ['Jupiter'])
    expected = {f'1J-3r{k}' for k in range(3)} | {f'5J-14r{k}' for k in range(10)}
    assert {m.to_s() for m in mmrs} == expected
    assert all(isinstance(m, RetrogradeTwoBody) for m in mmrs)
    assert RetrogradeTwoBodyMatrix.find_resonances(10.76, 0.5, ['Saturn']) == []


def test_find_mmrs_returns_only_the_retrograde_ones():
    retro = resonances.find_mmrs(10.76, planets=['Jupiter'], retrograde=True)
    assert {m.to_s() for m in retro} == {m.to_s() for m in resonances.find_mmrs(10.76, planets=['Jupiter'], mmr_type='two-body-retrograde')}
    assert all(m.retrograde for m in retro)


def test_find_mmrs_sigma_reaches_the_retrograde_search():
    """a = 10.76: 1/-3 at 10.8216 is 0.06 au away, 5/-14 at 10.34 is 0.42 au away (see above)."""
    narrow = {m.to_s() for m in resonances.find_mmrs(10.76, planets=['Jupiter'], retrograde=True, sigma=0.1)}
    assert narrow == {f'1J-3r{k}' for k in range(3)}
    assert narrow == {m.to_s() for m in resonances.find_mmrs(10.76, planets=['Jupiter'], retrograde=True, sigma_retro=0.1)}


@pytest.mark.parametrize('mmr_type', ['two-body', 'three-body'])
def test_find_mmrs_refuses_retrograde_with_a_prograde_type(mmr_type):
    with pytest.raises(ValueError, match='only retrograde'):
        resonances.find_mmrs(10.76, planets=['Jupiter'], retrograde=True, mmr_type=mmr_type)


def test_dump_and_load_with_the_stamp():
    RetrogradeTwoBodyMatrix.dump()
    path = Path(RetrogradeTwoBodyMatrix.catalog_full_filename())
    assert path.read_text().splitlines()[0] == STAMP

    RetrogradeTwoBodyMatrix.matrix = None
    RetrogradeTwoBodyMatrix.load()
    df = RetrogradeTwoBodyMatrix.matrix
    assert len(df) == 1198
    assert df.loc[df['mmr'] == '7J-9r6', 'k'].iloc[0] == 6


def test_a_stale_cache_is_rebuilt():
    path = Path(RetrogradeTwoBodyMatrix.catalog_full_filename())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(',mmr,planet,p,q,k,order,a\n0,1J-3r0,Jupiter,1,3,0,4,10.8\n')
    RetrogradeTwoBodyMatrix.load()
    assert path.read_text().splitlines()[0] == STAMP
    assert len(RetrogradeTwoBodyMatrix.matrix) == 1198


def test_a_cache_built_with_other_limits_is_rebuilt():
    """p, q <= 3, p + q <= 30: the coprime pairs (1,1) (1,2) (1,3) (2,1) (2,3) (3,1) (3,2), with
    (p+q)//2 + 1 = 2, 2, 3, 2, 3, 3, 3 values of k: 18 rows."""
    RetrogradeTwoBodyMatrix.dump()
    path = Path(RetrogradeTwoBodyMatrix.catalog_full_filename())
    old = resonances.config.get('MATRIX_2BODY_RETRO_COEF_MAX')
    resonances.config.set('MATRIX_2BODY_RETRO_COEF_MAX', 3)
    try:
        RetrogradeTwoBodyMatrix.matrix = None
        RetrogradeTwoBodyMatrix.load()
        assert path.read_text().splitlines()[0] == STAMP.replace('COEF_MAX=15', 'COEF_MAX=3')
        assert len(RetrogradeTwoBodyMatrix.matrix) == 18
    finally:
        resonances.config.set('MATRIX_2BODY_RETRO_COEF_MAX', old)
        RetrogradeTwoBodyMatrix.matrix = None
    RetrogradeTwoBodyMatrix.load()
    assert path.read_text().splitlines()[0] == STAMP
    assert len(RetrogradeTwoBodyMatrix.matrix) == 1198
