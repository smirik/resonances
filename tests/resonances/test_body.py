import resonances
import numpy as np


def test_checkers():
    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    body.mmrs = [mmr]
    body.statuses[mmr.to_s()] = 2

    assert body.status(mmr) == 2
    assert body.in_resonance(mmr) is True
    assert body.in_pure_resonance(mmr) is True

    body.statuses[mmr.to_s()] = 1
    assert body.status(mmr) == 1
    assert body.in_resonance(mmr) is True
    assert body.in_pure_resonance(mmr) is False

    body.statuses[mmr.to_s()] = 0
    assert body.status(mmr) == 0
    assert body.in_resonance(mmr) is False
    assert body.in_pure_resonance(mmr) is False


def test_setup():
    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    body.mmrs = [mmr]
    body.setup_vars_for_simulation(5)
    assert 5 == len(body.axis)
    assert 5 == len(body.ecc)
    assert 5 == len(body.angles[mmr.to_s()])
    assert 5 == len(body.varpi)
    assert 5 == len(body.longitude)


def test_str():
    body = resonances.Body()
    body.name = '463'
    body.mmrs = [resonances.create_mmr('4J-2S-1')]
    body.mass = 1.0
    body.type = 'asteroid'

    assert 'Body(type=asteroid, name=463, mass=1.0)\n Resonances: 4J-2S-1+0+0-1, ' == str(body)


def test_mmr_to_dict():
    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    times = np.array([0, 1, 2, 3, 4])
    body.mmrs = [mmr]
    body.angles[mmr.to_s()] = np.array([0, 1, 2, 3, 4])
    body.periodogram_peaks[mmr.to_s()] = np.array([0, 1, 2, 3, 4])

    result = body.mmr_to_dict(mmr, times)
    assert result is None

    body.periodogram_power[mmr.to_s()] = np.array([0, 1, 2, 3, 4])

    result = body.mmr_to_dict(mmr, times)
    print(result)

    assert isinstance(result, dict)
    assert len(result['angle']) == len(times)
    assert 'angle' in result
    assert 'a' in result
    assert 'e' in result


def test_angle():
    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    body.mmrs = [mmr]
    body.angles[mmr.to_s()] = np.array([0, 1, 2, 3, 4])

    result = body.angle(mmr)

    assert isinstance(result, np.ndarray)
    assert len(result) == len(body.angles[mmr.to_s()])
    assert np.array_equal(result, body.angles[mmr.to_s()])

    mmr = resonances.create_mmr('5J-2S-2')
    exception_text = 'The angle for the resonance {} does not exist in the body {}.'.format(mmr.to_s(), body.name)
    try:
        body.angle(mmr)
        assert False, exception_text
    except Exception as e:
        assert str(e) == exception_text


def test_in_resonance():
    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    body.mmrs = [mmr]
    body.statuses[mmr.to_s()] = 2

    assert body.in_resonance(mmr) is True

    body.statuses[mmr.to_s()] = 1
    assert body.in_resonance(mmr) is True

    body.statuses[mmr.to_s()] = 0
    assert body.in_resonance(mmr) is False


def test_is_particle():
    body = resonances.Body()
    body.type = 'particle'
    assert body.is_particle() is True

    body.type = 'planet'
    assert body.is_particle() is False
