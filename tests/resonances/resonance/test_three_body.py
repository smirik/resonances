import resonances
import pytest


def test_full_create():
    mmr = resonances.ThreeBody([4, -2, -1, 0, 0, -1])
    assert 4 == mmr.coeff[0]
    assert -1 == mmr.coeff[2]
    assert 'Jupiter' not in mmr.planets_names
    assert 'Saturn' not in mmr.planets_names

    mmr = resonances.ThreeBody([4, -2, -1, 0, 0, -1], ['Earth', 'Mars'])
    assert 'Jupiter' not in mmr.planets_names
    assert 'Saturn' not in mmr.planets_names
    assert 'Earth' in mmr.planets_names
    assert 'Mars' in mmr.planets_names

    with pytest.raises(Exception) as exception:
        mmr = resonances.ThreeBody([4, -2, -1, 0, 0, 0], ['Jupiter', 'Saturn'])
    assert 'Alembert' in str(exception.value)

    with pytest.raises(Exception) as exception:
        mmr = resonances.ThreeBody([4, -2, -2, 0, 0, 0], ['Jupiter', 'Saturn'])
    assert 'gcd' in str(exception.value)


def test_short_notation():
    mmr = resonances.ThreeBody('4V-2E-1')
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

    mmr = resonances.ThreeBody('2J+2S-1')
    assert 2 == mmr.coeff[0]
    assert 2 == mmr.coeff[1]
    assert -1 == mmr.coeff[2]
    assert 0 == mmr.coeff[3]
    assert 0 == mmr.coeff[4]
    assert -3 == mmr.coeff[5]

    mmr = resonances.ThreeBody('2J-5S+1')
    assert 2 == mmr.coeff[0]
    assert -5 == mmr.coeff[1]
    assert 1 == mmr.coeff[2]
    assert 0 == mmr.coeff[3]
    assert 0 == mmr.coeff[4]
    assert 2 == mmr.coeff[5]

    with pytest.raises(Exception) as exception:
        mmr = resonances.ThreeBody('4A-2S-1')
    assert 'notation' in str(exception.value)

    with pytest.raises(Exception) as exception:
        mmr = resonances.ThreeBody('4A-2S-1-5')
    assert 'three' in str(exception.value)


def test_to_s_and_to_short():
    mmr = resonances.ThreeBody([4, -2, -1, 0, 0, -1], ['Earth', 'Mars'])
    assert '4E-2M-1+0+0-1' == mmr.to_s()
    assert '4E-2M-1' == mmr.to_short()

    mmr = resonances.ThreeBody('2J-5S+1')
    assert '2J-5S+1+0+0+2' == mmr.to_s()
    assert '2J-5S+1' == mmr.to_short()


def test_resonant_axis_getter_and_setter():
    mmr = resonances.ThreeBody('4J-2S-1')
    assert mmr._resonant_axis is None
    assert 2.3981 == pytest.approx(mmr.resonant_axis, rel=0.1)
    assert mmr._resonant_axis is not None
    mmr.resonant_axis = 1.0
    assert 1.0 == mmr.resonant_axis


def test_calculate_axis():
    # see Smirnov, Shevchenko 2013
    mmr = resonances.ThreeBody('4J-2S-1')
    axis = mmr.calculate_resonant_axis()
    assert 2.3981 == pytest.approx(axis, rel=0.1)

    mmr = resonances.ThreeBody('5J-2S-2')
    axis = mmr.calculate_resonant_axis()
    assert 3.1746 == pytest.approx(axis, rel=0.1)

    mmr = resonances.ThreeBody('3J-2S-1')
    axis = mmr.calculate_resonant_axis()
    assert 3.0801 == pytest.approx(axis, rel=0.1)

    del mmr
    mmr = resonances.ThreeBody('4J-2S-1')
    assert 2.3981 == pytest.approx(mmr.resonant_axis, rel=0.1)


def test_calc_angle():
    mmr = resonances.ThreeBody('4J-2S-1')

    body1 = resonances.Body()
    body1.l, body1.Omega, body1.omega = 0.4, 0.1, 0.1
    body2 = resonances.Body()
    body2.l, body2.Omega, body2.omega = 0.1, 0.3, 0.2
    body = resonances.Body()
    body.l, body.Omega, body.omega = 0.3, 0.1, 0.1

    angle = mmr.calc_angle(body, [body1, body2])
    # 4*0.4 + (-2)*0.1 + (-1)*0.3 + 0 + 0 + (-1)*(0.1+0.1)
    assert 0.9 == pytest.approx(angle)
