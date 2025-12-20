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

    assert 'Body(type=asteroid, name=463, mass=1.0)\nMMR Resonances: 4J-2S-1+0+0-1, \n' == str(body)


def test_mmr_to_dict():
    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    body.mmrs = [mmr]
    body.angles[mmr.to_s()] = np.array([0, 1, 2, 3, 4])
    body.periodogram_peaks[mmr.to_s()] = np.array([0, 1, 2, 3, 4])

    result = body.mmr_to_dict(mmr)
    assert result is not None

    angle_key = mmr.to_s() + '_angle'
    assert angle_key in result
    assert isinstance(result[angle_key], np.ndarray)
    assert len(result[angle_key]) == 5

    # Should NOT contain Keplerian elements
    assert 'a' not in result
    assert 'e' not in result
    assert 'times' not in result

    # Test with filtered angles
    body.angles_filtered[mmr.to_s()] = np.array([0.1, 1.1, 2.1, 3.1, 4.1])
    result = body.mmr_to_dict(mmr)

    assert isinstance(result, dict)
    filtered_key = mmr.to_s() + '_angle_filtered'
    assert filtered_key in result
    assert len(result[filtered_key]) == 5


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
        raise AssertionError(exception_text)
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


def test_keplerian_elements_to_dict():
    """Test that keplerian_elements_to_dict returns all orbital elements."""
    body = resonances.Body()
    body.axis = np.array([2.5, 2.6, 2.7])
    body.ecc = np.array([0.1, 0.2, 0.3])
    body.inc = np.array([5.0, 6.0, 7.0])
    body.Omega = np.array([100.0, 110.0, 120.0])
    body.omega = np.array([200.0, 210.0, 220.0])
    body.M = np.array([300.0, 310.0, 320.0])
    body.longitude = np.array([400.0, 410.0, 420.0])
    body.varpi = np.array([500.0, 510.0, 520.0])

    result = body.keplerian_elements_to_dict()

    assert isinstance(result, dict)
    assert 'a' in result
    assert 'e' in result
    assert 'inc' in result
    assert 'Omega' in result
    assert 'omega' in result
    assert 'M' in result
    assert 'longitude' in result
    assert 'varpi' in result

    # Check values
    assert np.array_equal(result['a'], body.axis)
    assert np.array_equal(result['e'], body.ecc)
    assert np.array_equal(result['inc'], body.inc)

    # Test with filtered axis
    body.axis_filtered = np.array([2.51, 2.61, 2.71])
    result = body.keplerian_elements_to_dict()
    assert 'a_filtered' in result
    assert np.array_equal(result['a_filtered'], body.axis_filtered)


def test_resonances_helper():
    """Test that resonances() returns all resonances combined."""
    body = resonances.Body()

    # Add different types of resonances
    mmr = resonances.create_mmr('4J-2S-1')
    secular = resonances.create_resonance('g-2g6+g5')
    lkr = resonances.create_resonance('lkr')
    # Note: LidovKozaiResonance might need special setup, skipping for now

    body.mmrs = [mmr]
    body.secular_resonances = [secular]
    body.lidov_kozai_resonances = [lkr]

    all_resonances = body.resonances()

    assert isinstance(all_resonances, list)
    assert len(all_resonances) == 3
    assert mmr in all_resonances
    assert secular in all_resonances
    assert lkr in all_resonances


def test_resonance_to_dict_dispatcher():
    """Test that resonance_to_dict correctly dispatches to specific methods."""
    body = resonances.Body()

    # Test MMR dispatch
    mmr = resonances.create_mmr('4J-2S-1')
    body.mmrs = [mmr]
    body.angles[mmr.to_s()] = np.array([0, 1, 2, 3, 4])

    result = body.resonance_to_dict(mmr)
    assert result is not None
    assert mmr.to_s() + '_angle' in result

    # Test Secular dispatch
    secular = resonances.create_resonance('g-2g6+g5')
    body.secular_resonances = [secular]
    body.secular_angles[secular.to_s()] = np.array([10, 11, 12])
    body.secular_angles_osculating[secular.to_s()] = np.array([10.1, 11.1, 12.1])
    body.secular_angles_proper[secular.to_s()] = np.array([10.2, 11.2, 12.2])

    result = body.resonance_to_dict(secular)
    assert result is not None
    assert secular.to_s() + '_angle' in result
    assert secular.to_s() + '_angle_osculating' in result
    assert secular.to_s() + '_angle_proper' in result


def test_secular_to_dict():
    """Test secular_to_dict with new signature (no times parameter)."""
    body = resonances.Body()
    secular = resonances.create_resonance('g-2g6+g5')

    body.secular_resonances = [secular]
    body.secular_angles[secular.to_s()] = np.array([10, 11, 12, 13])
    body.secular_angles_osculating[secular.to_s()] = np.array([10.1, 11.1, 12.1, 13.1])
    body.secular_angles_proper[secular.to_s()] = np.array([10.2, 11.2, 12.2, 13.2])

    result = body.secular_to_dict(secular)

    assert result is not None
    assert isinstance(result, dict)

    # Check prefixed columns
    angle_key = secular.to_s() + '_angle'
    osculating_key = secular.to_s() + '_angle_osculating'
    proper_key = secular.to_s() + '_angle_proper'

    assert angle_key in result
    assert osculating_key in result
    assert proper_key in result

    assert len(result[angle_key]) == 4
    assert np.array_equal(result[angle_key], body.secular_angles[secular.to_s()])

    # Should NOT contain Keplerian elements
    assert 'a' not in result
    assert 'e' not in result
    assert 'times' not in result

    # Test with filtered angles
    body.secular_angles_filtered[secular.to_s()] = np.array([10.3, 11.3, 12.3, 13.3])
    result = body.secular_to_dict(secular)

    filtered_key = secular.to_s() + '_angle_filtered'
    assert filtered_key in result
    assert np.array_equal(result[filtered_key], body.secular_angles_filtered[secular.to_s()])


def test_lidov_kozai_to_dict():
    """Test lidov_kozai_to_dict with new signature (no times parameter)."""
    body = resonances.Body()
    lk = resonances.create_resonance('lkr')
    body.lidov_kozai_resonances = [lk]
    body.lidov_kozai_angles[lk.to_s()] = np.array([20, 21, 22, 23, 24])

    result = body.lidov_kozai_to_dict(lk)

    assert result is not None
    assert isinstance(result, dict)

    # Check that it uses generic 'LK_angle' key (not prefixed)
    assert lk.to_s() + '_angle' in result
    assert len(result[lk.to_s() + '_angle']) == 5
    assert np.array_equal(result[lk.to_s() + '_angle'], body.lidov_kozai_angles[lk.to_s()])

    # Should NOT contain Keplerian elements
    assert 'a' not in result
    assert 'e' not in result
    assert 'times' not in result

    # Test with filtered angles
    body.angles_filtered[lk.to_s()] = np.array([20.1, 21.1, 22.1, 23.1, 24.1])
    result = body.lidov_kozai_to_dict(lk)

    assert lk.to_s() + '_angle_filtered' in result
    assert np.array_equal(result[lk.to_s() + '_angle_filtered'], body.angles_filtered[lk.to_s()])
