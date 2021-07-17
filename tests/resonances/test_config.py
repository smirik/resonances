import resonances
import pytest


def test_get_and_has():
    assert resonances.config.has('save.path') is True
    assert 'cache' == resonances.config.get('save.path')

    assert resonances.config.has('plot.path') is True
    assert 'cache' == resonances.config.get('plot.path')

    assert resonances.config.has('plot') is True
    assert resonances.config.get('plot') is True

    assert resonances.config.has('save') is True
    assert resonances.config.get('save') is True

    assert resonances.config.has('save.summary') is True
    assert resonances.config.get('save.summary') is True

    assert resonances.config.has('integration.dt')
    assert 0.1 == resonances.config.get('integration.dt')

    assert resonances.config.has('catalog') is True
    assert 'cache/allnum.csv' == resonances.config.get('catalog')

    assert resonances.config.has('This is the house that Jack built') is False


def test_set():
    catalog = 'cache/allnum.csv'
    assert catalog == resonances.config.get('catalog')
    resonances.config.set('catalog', 'tests/fixtures/small.csv')
    assert 'tests/fixtures/small.csv' == resonances.config.get('catalog')


def test_config_exception():
    with pytest.raises(Exception) as exception:
        resonances.config.get('This is the house that Jack built')
        assert 'no config' in str(exception.value)
