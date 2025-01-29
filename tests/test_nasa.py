import pytest
import resonances
import resonances.horizons
import tests.tools as tools


def test_add_body_nasa():
    sim = tools.create_test_simulation_for_solar_system(save=True)
    sim.add_body('99942', resonances.create_mmr('7E-6'), name='Apophis')


def test_horizon():
    sim = tools.create_test_simulation_for_solar_system(save=True)
    elem = resonances.horizons.get_body_keplerian_elements('Hektor', sim.sim)
    assert 5.2 == pytest.approx(elem['a'], 0.2)


def test_463():
    sim = resonances.find(463, ['Jupiter', 'Saturn'], sigma3=0.1)
    sim.run()

    summary = sim.get_simulation_summary()

    assert 2 == summary.loc[summary['mmr'] == '4J-2S-1+0+0-1', 'status'].values[0]
    assert 0 == summary.loc[summary['mmr'] == '5J-4S-1+0+0+0', 'status'].values[0]
