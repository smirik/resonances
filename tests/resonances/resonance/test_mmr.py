import resonances
import pytest


def test_get_letter_from_planet_name():
    mmr = resonances.TwoBody([2, -1, 0, -1])
    assert 'E' == mmr.get_letter_from_planet_name('Earth')
    assert 'J' == mmr.get_letter_from_planet_name('Jupiter')
    assert 'R' == mmr.get_letter_from_planet_name('Mercury')
    assert 'M' == mmr.get_letter_from_planet_name('Mars')
    assert 'U' == mmr.get_letter_from_planet_name('Uranus')
    assert 'N' == mmr.get_letter_from_planet_name('Neptune')


def test_get_planet_name_from_letter():
    mmr = resonances.TwoBody([2, -1, 0, -1])
    assert 'Earth' == mmr.get_planet_name_from_letter('E')
    assert 'Jupiter' == mmr.get_planet_name_from_letter('J')
    assert 'Mercury' == mmr.get_planet_name_from_letter('R')
    assert 'Mars' == mmr.get_planet_name_from_letter('M')
    assert 'Uranus' == mmr.get_planet_name_from_letter('U')
    assert 'Neptune' == mmr.get_planet_name_from_letter('N')


def test_str():
    mmr = resonances.TwoBody([2, -1, 0, -1])
    assert str(mmr) == "MMR(coeff=[2, -1, 0, -1])"

    mmr = resonances.ThreeBody([4, -2, -1, 0, 0, -1])
    assert str(mmr) == "MMR(coeff=[4, -2, -1, 0, 0, -1])"


def test_initialisation():
    mmr = resonances.ThreeBody([4, -2, -1, 0, 0, -1])
    assert 4 == mmr.coeff[0]
    assert -2 == mmr.coeff[1]

    mmr = resonances.TwoBody([2, -1, 0, -1])
    assert 2 == mmr.coeff[0]
    assert -1 == mmr.coeff[1]

    with pytest.raises(Exception) as exception:
        mmr = resonances.TwoBody([-1, -1, 0, 0])
        assert 'primary coefficient' in str(exception.value)


def test_dalambert():
    with pytest.raises(Exception) as exception:
        resonances.ThreeBody([4, -2, -1, 0, 0, 1])
        assert 'Sum of integers' in str(exception.value)

    with pytest.raises(Exception) as exception:
        resonances.TwoBody([2, -1, 0, 0])
        assert 'Sum of integers' in str(exception.value)


def test_number_of_bodies():
    mmr = resonances.ThreeBody([4, -2, -1, 0, 0, -1])
    assert 3 == mmr.number_of_bodies()

    mmr = resonances.TwoBody([2, -1, 0, -1])
    assert 2 == mmr.number_of_bodies()
