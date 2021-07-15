from resonances.resonance.three_body import ThreeBody
import resonances
import pytest


def test_dalambert():
    with pytest.raises(Exception) as exception:
        mmr = ThreeBody([4, -2, -1, 0, 0, 1])
    assert 'Sum of integers' in str(exception.value)


def test_full_create():
    mmr = ThreeBody([4, -2, -1, 0, 0, -1])
    assert 4 == mmr.coeff[0]
    assert -1 == mmr.coeff[2]
    assert 'Jupiter' not in mmr.planets_names
    assert 'Saturn' not in mmr.planets_names

    mmr = ThreeBody([4, -2, -1, 0, 0, -1], ['Earth', 'Mars'])
    assert 'Jupiter' not in mmr.planets_names
    assert 'Saturn' not in mmr.planets_names
    assert 'Earth' in mmr.planets_names
    assert 'Mars' in mmr.planets_names


def test_short_notation():
    mmr = ThreeBody(s='4V-2E-1')
    assert 4 == mmr.coeff[0]
    assert -2 == mmr.coeff[1]
    assert -1 == mmr.coeff[2]
    assert 0 == mmr.coeff[3]
    assert 0 == mmr.coeff[4]
    assert -1 == mmr.coeff[5]
    assert 'Venus' in mmr.planets_names
    assert 'Earth' in mmr.planets_names
    assert 'Jupiter' not in mmr.planets_names
    assert 'Saturn' not in mmr.planets_names

    mmr = ThreeBody(s='2J+2S-1')
    assert 2 == mmr.coeff[0]
    assert 2 == mmr.coeff[1]
    assert -1 == mmr.coeff[2]
    assert 0 == mmr.coeff[3]
    assert 0 == mmr.coeff[4]
    assert -3 == mmr.coeff[5]

    mmr = ThreeBody('2J-5S+1')
    assert 2 == mmr.coeff[0]
    assert -5 == mmr.coeff[1]
    assert 1 == mmr.coeff[2]
    assert 0 == mmr.coeff[3]
    assert 0 == mmr.coeff[4]
    assert 2 == mmr.coeff[5]

    with pytest.raises(Exception) as exception:
        mmr = ThreeBody('4A-2S-1')
    assert 'notation' in str(exception.value)

    with pytest.raises(Exception) as exception:
        mmr = ThreeBody('4A-2S-1-5')
    assert 'three' in str(exception.value)


def test_to_s_and_to_short():
    mmr = ThreeBody([4, -2, -1, 0, 0, -1], ['Earth', 'Mars'])
    assert '4E-2M-1+0+0-1' == mmr.to_s()
    assert '4E-2M-1' == mmr.to_short()

    mmr = ThreeBody('2J-5S+1')
    assert '2J-5S+1+0+0+2' == mmr.to_s()
    assert '2J-5S+1' == mmr.to_short()
