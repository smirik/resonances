import numpy as np
import pytest
import resonances
import resonances.horizons
import tests.tools as tools
from astroquery.jplhorizons import Horizons
from astropy.table import Table


@pytest.mark.slow
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


@pytest.mark.slow
def test_add_body_nasa():
    sim = tools.create_test_simulation_for_solar_system(save=True)
    sim.add_body('99942', resonances.create_mmr('7E-6'), name='Apophis')


@pytest.mark.slow
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


@pytest.mark.parametrize(
    's, expected',
    [
        (401, '401;'),
        ('401', '401;'),
        (' 502 ', '502;'),
        (502.0, '502;'),
        (np.int64(463), '463;'),
        ('401;', '401;'),
        ('Hektor', 'Hektor'),
        ('2017 HV1', '2017 HV1'),
    ],
)
def test_horizons_id_numbers_are_small_bodies(s, expected):
    assert resonances.horizons.horizons_id(s) == expected


def test_horizons_id_rejects_fractional_number():
    with pytest.raises(ValueError):
        resonances.horizons.horizons_id(401.5)


@pytest.mark.slow
def test_horizons_number_as_string_is_the_asteroid():
    """'401' is 401 Ottilia (a = 3.35 AU, JPL SBDB), not Phobos (a = 1.58 AU around the Sun); the epoch is the requested date."""
    by_str = resonances.horizons.get_body_keplerian_elements('401', '2026-06-09')
    by_int = resonances.horizons.get_body_keplerian_elements(401, '2026-06-09')
    assert by_str['a'] == pytest.approx(3.346, abs=0.01)
    assert by_str == by_int
    assert by_str['epoch'] == pytest.approx(61200.0, abs=1e-6)
