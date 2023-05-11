import resonances
import pytest


def test_get_letter_from_planet_name():
    mmr = resonances.MMR([2, -1, 0, -1])
    assert 'E' == mmr.get_letter_from_planet_name('Earth')
    assert 'J' == mmr.get_letter_from_planet_name('Jupiter')
    assert 'R' == mmr.get_letter_from_planet_name('Mercury')
    assert 'M' == mmr.get_letter_from_planet_name('Mars')


def test_initialisation():
    mmr = resonances.MMR([4, -2, -1, 0, 0, -1])
    assert 4 == mmr.coeff[0]
    assert -2 == mmr.coeff[1]

    mmr = resonances.MMR([2, -1, 0, -1])
    assert 2 == mmr.coeff[0]
    assert -1 == mmr.coeff[1]

    with pytest.raises(Exception) as exception:
        mmr = resonances.MMR([-1, -1, 0, 0])
        assert 'primary coefficient' in str(exception.value)


def test_dalambert():
    with pytest.raises(Exception) as exception:
        resonances.MMR([4, -2, -1, 0, 0, 1])
        assert 'Sum of integers' in str(exception.value)

    with pytest.raises(Exception) as exception:
        resonances.MMR([2, -1, 0, 0])
        assert 'Sum of integers' in str(exception.value)


def test_number_of_bodies():
    mmr = resonances.MMR([4, -2, -1, 0, 0, -1])
    assert 3 == mmr.number_of_bodies()

    mmr = resonances.MMR([2, -1, 0, -1])
    assert 2 == mmr.number_of_bodies()
