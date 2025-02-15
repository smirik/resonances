import resonances
import pytest


def test_get_and_has():
    assert resonances.config.has('SAVE_PATH') is True
    assert 'cache' == resonances.config.get('SAVE_PATH')

    assert resonances.config.has('PLOT_PATH') is True
    assert 'cache' == resonances.config.get('PLOT_PATH')

    assert resonances.config.has('SAVE_MODE') is True
    assert resonances.config.get('SAVE_MODE') == 'nonzero'

    assert resonances.config.has('PLOT_MODE') is True
    assert resonances.config.get('PLOT_MODE') == 'nonzero'

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
