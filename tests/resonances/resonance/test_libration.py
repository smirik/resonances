import numpy as np
import pytest
import resonances


def test_pure():
    arr = [1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1]
    assert resonances.libration.pure(arr) is True

    arr = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
    assert resonances.libration.pure(arr) is False

    arr = [6, 0, 1, 0, 6, 5, 6, 0, 1, 2]
    assert resonances.libration.pure(arr) is True

    # test pure and rewrite it after new normal example


def test_shift():
    data = [2 * np.pi - 0.01, 0.01, 4.0]
    shift = resonances.libration.shift(data)
    assert shift[0] == pytest.approx((data[0] - 2 * np.pi))
    assert shift[1] == pytest.approx(0.01)
    assert shift[2] == pytest.approx(data[2] - 2 * np.pi)


def test_resolve():
    overlapping = [[100, 102]]
    empty = []

    lib_crit = float(resonances.config.get('LIBRATION_PERIOD_CRITICAL'))
    mon_crit = [float(x.strip()) for x in resonances.config.get('LIBRATION_MONOTONY_CRITICAL').split(",")]

    # Create mock resonances for testing
    class MockMMR:
        def __init__(self):
            pass

    class MockSecularResonance(resonances.resonance.secular.SecularResonance):
        def __init__(self):
            super().__init__('test', 'Jupiter')

    mmr = MockMMR()

    # pure libration with libration for both angle and axis at the same frequency
    assert 2 == resonances.libration.resolve(mmr, True, overlapping, 100000, lib_crit, 0.5, mon_crit)
    # pure libration with libration for both angle and axis but not at the same frequency
    assert -2 == resonances.libration.resolve(mmr, True, empty, 100000, lib_crit, 0.5, mon_crit)
    # case of very slow circulation
    assert -2 == resonances.libration.resolve(mmr, True, empty, 100000, lib_crit, 0.1, mon_crit)

    assert 1 == resonances.libration.resolve(mmr, False, overlapping, 30000, lib_crit, 0.4, mon_crit)
    # monotony does not apply if libration length is greater than critical and overlapping
    assert 1 == resonances.libration.resolve(mmr, False, overlapping, 30000, lib_crit, 0.1, mon_crit)
    # uncertain status if there is no overlapping but still acceptable libration length and monotony
    assert -1 == resonances.libration.resolve(mmr, False, empty, 30000, lib_crit, 0.4, mon_crit)
    # no resonance if no overlapping and libration period less than critical or no monotony if no overlapping
    assert 0 == resonances.libration.resolve(mmr, False, empty, 10000, lib_crit, 0.5, mon_crit)
    assert 0 == resonances.libration.resolve(mmr, False, empty, 30000, lib_crit, 0.3, mon_crit)

    secular = MockSecularResonance()

    # For SecularResonance: if pure, always status = 2 (no overlapping check)
    assert 2 == resonances.libration.resolve(secular, True, empty, 100000, lib_crit, 0.5, mon_crit)
    assert 2 == resonances.libration.resolve(secular, True, overlapping, 100000, lib_crit, 0.5, mon_crit)

    # For SecularResonance: if not pure but max_libration_length > critical, status = 1
    assert 1 == resonances.libration.resolve(secular, False, empty, 30000, lib_crit, 0.4, mon_crit)
    assert 1 == resonances.libration.resolve(secular, False, overlapping, 30000, lib_crit, 0.4, mon_crit)

    # For SecularResonance: if not pure and max_libration_length <= critical, status = 0
    assert 0 == resonances.libration.resolve(secular, False, empty, 10000, lib_crit, 0.5, mon_crit)
    assert 0 == resonances.libration.resolve(secular, False, overlapping, 10000, lib_crit, 0.5, mon_crit)


def test_monotony_estimation():
    data = [1, 2, 3, 4, 5]
    assert 0.0 == resonances.libration.monotony_estimation(data)

    data = [1, 2, 3, 4, 5]
    assert 0.0 == resonances.libration.monotony_estimation(data)

    data = [5, 4, 3, 2, 1]
    assert 1.0 == resonances.libration.monotony_estimation(data)

    data = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    assert 0.0 == resonances.libration.monotony_estimation(data)

    data = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    assert 0.5 == resonances.libration.monotony_estimation(data)

    data = [1, 2, 3, 4, 5, 6, 1, 6, 1, 6, 1]
    assert 0.2 == resonances.libration.monotony_estimation(data)

    data = [1]
    assert 0.0 == resonances.libration.monotony_estimation(data)
