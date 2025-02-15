import pytest

from resonances.resonance.three_body import ThreeBody
from resonances.resonance.two_body import TwoBody
from resonances.resonance.mmr import MMR
from resonances.resonance.factory import create_mmr


def test_create_mmr():
    mmr = create_mmr('4J-2S-1')
    assert isinstance(mmr, ThreeBody) is True

    mmr = create_mmr(mmr)
    assert isinstance(mmr, MMR) is True
    assert isinstance(mmr, ThreeBody) is True

    mmr1 = create_mmr('2J-1')
    assert isinstance(mmr1, TwoBody) is True
    mmr2 = create_mmr([4, -2, -1, 0, 0, -1], planets_names=['Jupiter', 'Saturn'])
    assert isinstance(mmr2, ThreeBody) is True
    mmr3 = create_mmr([2, -1, 0, -1], planets_names=['Jupiter'])
    assert isinstance(mmr3, TwoBody) is True

    mmrs = create_mmr([mmr1, mmr2, mmr3])
    assert 3 == len(mmrs)
    assert isinstance(mmrs[0], TwoBody) is True
    assert isinstance(mmrs[1], ThreeBody) is True
    assert isinstance(mmrs[2], TwoBody) is True

    mmrs = create_mmr(['4J-2S-1', '1J-1'])
    assert 2 == len(mmrs)
    assert isinstance(mmrs[0], ThreeBody) is True
    assert isinstance(mmrs[1], TwoBody) is True

    with pytest.raises(Exception) as exception:
        mmr = create_mmr('5J-2S-1U-1')
    assert 'notation is wrong' in str(exception.value)

    with pytest.raises(Exception) as exception:
        mmr = create_mmr('5J')
    assert 'notation is wrong' in str(exception.value)

    with pytest.raises(Exception) as exception:
        mmr = create_mmr([4, -2, -1, 0, 0, 0, -1], planets_names=['Jupiter', 'Saturn'])
    assert 'number of coefficients' in str(exception.value)

    with pytest.raises(Exception) as exception:
        mmr = create_mmr([4, -2, -1], planets_names=['Jupiter', 'Saturn'])
    assert 'number of coefficients' in str(exception.value)

    with pytest.raises(Exception) as exception:
        mmr = create_mmr([4, -2], planets_names=['Jupiter'])
    assert 'number of coefficients' in str(exception.value)

    with pytest.raises(Exception) as exception:
        mmr = create_mmr(42, planets_names=['Jupiter'])
    assert 'should be either' in str(exception.value)
