import pytest
import resonances
import resonances.horizons
import tests.tools as tools
from astroquery.jplhorizons import Horizons
from astropy.table import Table


def test_astroquery():
    obj = Horizons(id='Hektor', location='500@10', epochs=2458133.33546)
    elems = obj.elements()
    assert isinstance(elems, Table)

    assert elems['a'][0] > 5.0
    assert elems['a'][0] < 5.5

    assert 'a' in elems.keys()
    assert 'e' in elems.keys()
    assert 'incl' in elems.keys()
    assert 'w' in elems.keys()
    assert 'Omega' in elems.keys()
    assert 'M' in elems.keys()


def test_add_body_nasa():
    sim = tools.create_test_simulation_for_solar_system(save=True)
    sim.add_body('99942', resonances.create_mmr('7E-6'), name='Apophis')


def test_horizon():
    elem = resonances.horizons.get_body_keplerian_elements('Hektor', '2025-01-01')
    assert elem['a'] > 5.0
    assert elem['a'] < 5.5
    assert 'a' in elem
    assert 'e' in elem
    assert 'inc' in elem
    assert 'omega' in elem
    assert 'Omega' in elem
    assert 'M' in elem
    assert 5.2 == pytest.approx(elem['a'], 0.2)
