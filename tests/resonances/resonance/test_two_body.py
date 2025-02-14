import resonances
import pytest


def test_order():
    mmr = resonances.TwoBody('4J-1')
    assert 3 == mmr.order()

    mmr = resonances.TwoBody('1J-1')
    assert 0 == mmr.order()

    mmr = resonances.TwoBody('1J+1')
    assert 2 == mmr.order()


def test_full_create():
    mmr = resonances.TwoBody([2, -1, 0, -1])
    assert 2 == mmr.coeff[0]
    assert -1 == mmr.coeff[1]
    assert 'Jupiter' not in mmr.planets_names
    assert 'Saturn' not in mmr.planets_names

    mmr = resonances.TwoBody([2, -1, 0, -1], ['Mars'])
    assert 'Jupiter' not in mmr.planets_names
    assert 'Saturn' not in mmr.planets_names
    assert 'Mars' in mmr.planets_names

    with pytest.raises(Exception) as exception:
        mmr = resonances.TwoBody([2, -1, 0, 0], ['Jupiter'])
    assert 'Alembert' in str(exception.value)

    with pytest.raises(Exception) as exception:
        mmr = resonances.TwoBody([4, -2, 0, -2], ['Jupiter'])
    assert 'gcd' in str(exception.value)


def test_short_notation():
    mmr = resonances.TwoBody('2M-1')
    assert 2 == mmr.coeff[0]
    assert -1 == mmr.coeff[1]
    assert 'Mars' in mmr.planets_names
    assert 'Mercury' not in mmr.planets_names
    assert 'Jupiter' not in mmr.planets_names

    mmr = resonances.TwoBody('2R-1')
    assert 'Mercury' in mmr.planets_names
    assert 'Mars' not in mmr.planets_names

    mmr = resonances.TwoBody('2J+1')
    assert 2 == mmr.coeff[0]
    assert 1 == mmr.coeff[1]
    assert 'Jupiter' in mmr.planets_names

    with pytest.raises(Exception) as exception:
        mmr = resonances.TwoBody('4A-1')
    assert 'notation' in str(exception.value)

    with pytest.raises(Exception) as exception:
        mmr = resonances.TwoBody('4A-2S-1')
    assert 'two' in str(exception.value)


def test_to_s_and_to_short():
    mmr = resonances.TwoBody([2, -1, 0, -1], ['Mars'])
    assert '2M-1+0-1' == mmr.to_s()
    assert '2M-1' == mmr.to_short()

    mmr = resonances.TwoBody([2, -1, 0, -1], ['Mercury'])
    assert '2R-1+0-1' == mmr.to_s()
    assert '2R-1' == mmr.to_short()

    mmr = resonances.TwoBody('2J-1')
    assert '2J-1+0-1' == mmr.to_s()
    assert '2J-1' == mmr.to_short()


def test_resonant_axis_getter_and_setter():
    mmr = resonances.TwoBody('1J-1')
    assert mmr._resonant_axis is None
    assert 5.2043 == pytest.approx(mmr.resonant_axis, rel=0.1)
    assert mmr._resonant_axis is not None
    mmr.resonant_axis = 1.0
    assert 1.0 == mmr.resonant_axis


def test_calculate_axis():
    # see Smirnov, Shevchenko 2013
    mmr = resonances.TwoBody('3J-2')
    axis = mmr.calculate_resonant_axis()
    assert 3.9716 == pytest.approx(axis, rel=0.1)

    mmr = resonances.TwoBody('11J-5')
    axis = mmr.calculate_resonant_axis()
    assert 3.0766 == pytest.approx(axis, rel=0.1)

    mmr = resonances.TwoBody('2J-1')
    axis = mmr.calculate_resonant_axis()
    assert 3.2785 == pytest.approx(axis, rel=0.1)

    exception_text = 'You must specify two integers only for a short notation, i.e., 2J-1.'
    try:
        mmr = resonances.TwoBody('1J-2S-0')
        raise AssertionError(exception_text)
    except Exception as e:
        assert exception_text in str(e)

    mmr.planets_names = ['Jupiter', 'Saturn']
    exception_text = 'Cannot calculate resonant axis if the only planet is not specified!'
    try:
        axis = mmr.calculate_resonant_axis()
        raise AssertionError(exception_text)
    except Exception as e:
        assert exception_text in str(e)


def test_calc_angle():
    mmr = resonances.TwoBody('2J-1')

    body1 = resonances.Body()
    body1.l, body1.Omega, body1.omega = 0.4, 0.2, 0.1
    body = resonances.Body()
    body.l, body.Omega, body.omega = 0.3, 0.1, 0.2

    angle = mmr.calc_angle(body, [body1])
    # 2*0.4 + (-1)*0.3 + 0 + (-1)*(0.1+0.2)
    assert 0.2 == pytest.approx(angle)
