import pytest

from resonances.resonance.three_body import ThreeBody
from resonances.resonance.two_body import TwoBody
from resonances.resonance.factory import create_mmr


def test_create_mmr():
    mmr = create_mmr('4J-2S-1')
    assert isinstance(mmr, ThreeBody) is True
    mmr = create_mmr('2J-1')
    assert isinstance(mmr, TwoBody) is True
    mmr = create_mmr([4, -2, -1, 0, 0, -1], planets_names=['Jupiter', 'Saturn'])
    assert isinstance(mmr, ThreeBody) is True
    mmr = create_mmr([2, -1, 0, -1], planets_names=['Jupiter'])
    assert isinstance(mmr, TwoBody) is True

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
