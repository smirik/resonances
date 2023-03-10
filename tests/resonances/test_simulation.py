import numpy as np
import rebound
import tests.tools as tools
import shutil
from pathlib import Path
import pytest

import resonances


@pytest.fixture(autouse=True)
def run_around_tests():
    Path('cache/tests').mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree('cache/tests')


def test_init():
    resonances.config.set('integration.tmax', 100)
    sim = resonances.Simulation()
    sim.Nout = 10
    sim.tmax_yrs = 100 / (2 * np.pi)


def test_solar_system():
    sim = tools.create_test_simulation_for_solar_system()
    assert isinstance(sim.sim, rebound.Simulation) is True
    assert 10 == len(sim.sim.particles)


def test_add_body():
    sim = tools.create_test_simulation_for_solar_system()
    elem = tools.get_3body_elements_sample()
    mmr = resonances.ThreeBody('4J-2S-1')

    sim.add_body(elem, mmr)
    assert 10 == len(sim.sim.particles)  # because it is not added to rebound simulation yet
    assert 1 == len(sim.bodies)
    assert sim.bodies[0].initial_data['a'] == elem['a']
    assert sim.bodies[0].mmr.coeff[0] == mmr.coeff[0]
    assert 5 == sim.bodies[0].index_of_planets[0]
    assert 6 == sim.bodies[0].index_of_planets[1]

    sim.add_body(6, mmr)  # add from astdys
    assert 2 == len(sim.bodies)
    sim.add_body('7', mmr)  # add from astdys
    assert 3 == len(sim.bodies)


def test_add_bodies_to_simulation():
    sim = tools.create_test_simulation_for_solar_system()
    tools.add_test_asteroid_to_simulation(sim)

    body = sim.bodies[0]
    sim.add_body_to_simulation(body)
    assert 11 == len(sim.sim.particles)


def test_run():
    sim = tools.create_test_simulation_for_solar_system()
    tools.add_test_asteroid_to_simulation(sim)
    sim.save = True
    sim.plot = True
    sim.save_summary = True
    sim.run()

    assert 10 == len(sim.bodies[0].angle)
    assert 10 == len(sim.bodies[0].axis)
    assert 10 == len(sim.bodies[0].ecc)
    assert sim.bodies[0].status is not None
    assert sim.bodies[0].axis_filtered is not None
    assert sim.bodies[0].angle_filtered is not None

    assert Path('cache/tests/data-10-asteroid.csv').exists() is True
    assert Path('cache/tests/10-asteroid.png').exists() is True
    assert Path('cache/tests/summary.csv').exists() is True


@pytest.fixture
def saving_fixtures():
    # files index: data and plot for well determined and data and plot for undetermined.
    return [
        {'save': True, 'plot': True, 'undetermined': True, 'files': [True, True, True, True]},
        {'save': True, 'plot': True, 'undetermined': False, 'files': [True, True, True, True]},
        {'save': True, 'plot': False, 'undetermined': True, 'files': [True, False, True, True]},
        {'save': False, 'plot': True, 'undetermined': True, 'files': [False, True, True, True]},
        {'save': True, 'plot': False, 'undetermined': False, 'files': [True, False, True, False]},
        {'save': False, 'plot': True, 'undetermined': False, 'files': [False, True, False, True]},
        {'save': False, 'plot': False, 'undetermined': True, 'files': [False, False, True, True]},
        {'save': False, 'plot': False, 'undetermined': False, 'files': [False, False, False, False]},
    ]


def test_shall_save_and_plot_body(saving_fixtures):
    for saving_fixture in saving_fixtures:
        sim = tools.create_test_simulation_for_solar_system(
            save=saving_fixture['save'], plot=saving_fixture['plot'], save_only_undetermined=saving_fixture['undetermined']
        )

        elem = tools.get_3body_elements_sample()
        mmr = resonances.ThreeBody('4J-2S-1')
        sim.add_body(elem, mmr, name='asteroid')
        sim.add_body(elem, mmr, name='asteroid2')
        sim.bodies[0].status = 2
        sim.bodies[1].status = -1

        assert saving_fixture['files'][0] == sim.shall_save_body(sim.bodies[0])
        assert saving_fixture['files'][1] == sim.shall_plot_body(sim.bodies[0])
        assert saving_fixture['files'][2] == sim.shall_save_body(sim.bodies[1])
        assert saving_fixture['files'][3] == sim.shall_plot_body(sim.bodies[1])


def test_saving_summary():
    sim = tools.create_test_simulation_for_solar_system(save=True, save_summary=True)
    tools.add_test_asteroid_to_simulation(sim)
    sim.run()

    assert Path('cache/tests/summary.csv').exists() is True
    shutil.rmtree('cache/tests')

    del sim
    sim = tools.create_test_simulation_for_solar_system(save=True, save_summary=False)
    tools.add_test_asteroid_to_simulation(sim)
    sim.run()

    assert Path('cache/tests/summary.csv').exists() is False


def test_get_body_data():
    sim = tools.create_test_simulation_for_solar_system(save_additional_data=False)
    tools.add_test_asteroid_to_simulation(sim)
    sim.run()

    data = sim.get_body_data(sim.bodies[0])
    assert ('times' in data) is True
    assert ('angle' in data) is True
    assert ('a' in data) is True
    assert ('e' in data) is True
    assert ('periodogram' in data) is False
    assert ('a_filtered' in data) is False
    assert ('a_periodogram' in data) is False

    del sim
    sim = tools.create_test_simulation_for_solar_system(save_additional_data=True, save=True)
    tools.add_test_asteroid_to_simulation(sim)
    sim.run()

    data = sim.get_body_data(sim.bodies[0])
    assert ('angle' in data) is True
    assert ('periodogram' in data) is True
    assert ('a_filtered' in data) is True
    assert ('a_periodogram' in data) is True


def test_get_simulation_summary():
    sim = tools.create_test_simulation_for_solar_system(save=True)
    tools.add_test_asteroid_to_simulation(sim)
    sim.run()

    df = sim.get_simulation_summary()
    assert 1 == len(df)
    assert 13 == len(df.columns)
    assert 'asteroid' == df['name'].iloc[0]


def test_list_of_planets():
    sim = tools.create_test_simulation_for_solar_system(save=True)
    assert 10 == len(sim.list_of_planets())  # Yo, Pluto! And Sun... Am I really an astronomer?


@pytest.fixture
def planets_and_indexes():
    return [[['Jupiter', 'Saturn'], [5, 6]], [['Mercury', 'Venus', 'Mars'], [1, 2, 4]]]


def test_get_index_of_planets(planets_and_indexes):
    sim = tools.create_test_simulation_for_solar_system()

    for data in planets_and_indexes:
        planets_names = data[0]
        planets_indexes = data[1]
        assert all([a == b for a, b in zip(planets_indexes, sim.get_index_of_planets(planets_names))])
