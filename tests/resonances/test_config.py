import resonances
import pytest


def test_get_and_has():
    assert True == resonances.config.has('save.path')
    assert 'cache' == resonances.config.get('save.path')

    assert True == resonances.config.has('plot.path')
    assert 'cache' == resonances.config.get('plot.path')

    assert True == resonances.config.has('plot')
    assert True == resonances.config.get('plot')

    assert True == resonances.config.has('save')
    assert True == resonances.config.get('save')

    assert True == resonances.config.has('save.summary')
    assert True == resonances.config.get('save.summary')

    assert True == resonances.config.has('integration.dt')
    assert 0.1 == resonances.config.get('integration.dt')

    assert True == resonances.config.has('catalog')
    assert 'cache/allnum.csv' == resonances.config.get('catalog')

    assert False == resonances.config.has('This is the house that Jack built')


def test_set():
    catalog = 'cache/allnum.csv'
    assert catalog == resonances.config.get('catalog')
    resonances.config.set('catalog', 'tests/fixtures/small.csv')
    assert 'tests/fixtures/small.csv' == resonances.config.get('catalog')


def test_config_exception():
    with pytest.raises(Exception) as exception:
        tmp = resonances.config.get('This is the house that Jack built')
        assert 'no config' in str(exception.value)
