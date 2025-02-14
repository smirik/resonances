import pytest
import resonances
import resonances.horizons
import tests.tools as tools


def test_add_body_nasa():
    sim = tools.create_test_simulation_for_solar_system(save=True)
    sim.add_body('99942', resonances.create_mmr('7E-6'), name='Apophis')


def test_horizon():
    sim = tools.create_test_simulation_for_solar_system(save=True)
    elem = resonances.horizons.get_body_keplerian_elements('Hektor', sim.sim, '2025-01-01')
    assert 'a' in elem
    assert 'e' in elem
    assert 'inc' in elem
    assert 'omega' in elem
    assert 'Omega' in elem
    assert 'M' in elem
    assert 5.2 == pytest.approx(elem['a'], 0.2)
