import resonances
import pytest


def test_get_and_has():
    assert resonances.config.has('SAVE_PATH') is True
    assert 'cache/tests' == resonances.config.get('SAVE_PATH')  # cache/tests set in conftest.py

    assert resonances.config.has('PLOT_PATH') is True
    assert 'cache/tests' == resonances.config.get('PLOT_PATH')

    assert resonances.config.has('SAVE') is True
    assert resonances.config.get('SAVE') == 'nonzero'

    assert resonances.config.has('PLOT') is True
    assert resonances.config.get('PLOT') == 'nonzero'

    assert bool(resonances.config.has('SAVE_SUMMARY')) is True
    assert bool(resonances.config.get('SAVE_SUMMARY')) is True

    assert resonances.config.has('INTEGRATION_DT')
    assert 1.0 == float(resonances.config.get('INTEGRATION_DT'))

    assert resonances.config.has('CATALOG_PATH') is True
    assert 'cache/allnum.csv' == resonances.config.get('CATALOG_PATH')

    assert resonances.config.has('This is the house that Jack built') is False


def test_default():
    assert resonances.config.get('This is the house that Jack built', 'John Galt') == 'John Galt'

    exception_text = 'There is no config with key = This is the house that Jack built'
    try:
        resonances.config.get('This is the house that Jack built')
        raise AssertionError(exception_text)
    except Exception as e:
        assert exception_text in str(e)

    try:
        resonances.config.set('This is the house that Jack built', 'hello')
        raise AssertionError(exception_text)
    except Exception as e:
        assert exception_text in str(e)


def test_set():
    catalog = 'cache/allnum.csv'
    assert catalog == resonances.config.get('CATALOG_PATH')
    resonances.config.set('CATALOG_PATH', 'tests/fixtures/small.csv')
    assert 'tests/fixtures/small.csv' == resonances.config.get('CATALOG_PATH')


def test_config_exception():
    with pytest.raises(Exception) as exception:
        resonances.config.get('This is the house that Jack built')
        assert 'no config' in str(exception.value)


def test_planetary_frequencies():
    """The fundamental frequencies live in `PLANETARY_FREQUENCIES`, not in the .env config (the old
    g5..s8 keys there were never read and were removed)."""
    from resonances.data.const import PLANETARY_FREQUENCIES as F, PLANETARY_PHASES

    # AstDyS values (Knezevic & Milani 2000) for the outer planets
    assert F['g5'] == 4.25749319 and F['g6'] == 28.24552984 and F['g7'] == 3.08675577 and F['g8'] == 0.67255084
    assert F['s5'] == 0 and F['s6'] == -26.34496354 and F['s7'] == -2.99266093 and F['s8'] == -0.69251386
    # the inner planets, measured on the package's model; within 0.4 arcsec/yr of Laskar (1990), whose
    # values differ by general relativity (Mercury) and less for the others
    la90 = {'g1': 5.5964, 'g2': 7.4559, 'g3': 17.3646, 'g4': 17.9154, 's1': -5.6174, 's2': -7.0798, 's3': -18.8512, 's4': -17.7482}
    for label, value in la90.items():
        assert abs(F[label] - value) < 0.4, label
    assert set(PLANETARY_PHASES) == set(F)
    assert not resonances.config.has('g5')
