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
    # shutil.rmtree('cache/tests')


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
    assert sim.bodies[0].mmrs[0].coeff[0] == mmr.coeff[0]
    assert 5 == sim.bodies[0].mmrs[0].index_of_planets[0]
    assert 6 == sim.bodies[0].mmrs[0].index_of_planets[1]

    sim.add_body(6, mmr)  # add from astdys
    assert 2 == len(sim.bodies)
    sim.add_body('7', mmr)  # add from astdys
    assert 3 == len(sim.bodies)

    sim.add_body(1, '4J-2S-1')
    assert 4 == len(sim.bodies)

    elem['mass'] = 2.0
    sim.add_body(elem, '5J-2S-2')
    assert 5 == len(sim.bodies)
    assert 2.0 == sim.bodies[4].initial_data['mass']

    exception_text = 'You can add body only by its number or all orbital elements'
    try:
        sim.add_body(None, '5J-2S-2')
        raise AssertionError(exception_text)
    except Exception as e:
        assert str(e) == exception_text

    exception_text = 'If input is a list, it should contain a string representation of MMRs, MMR objects, or coefficients.'
    try:
        sim.add_body(2, [])
        raise AssertionError(exception_text)
    except Exception as e:
        assert str(e) == exception_text


def test_add_bodies_to_simulation():
    sim = tools.create_test_simulation_for_solar_system()
    tools.add_test_asteroid_to_simulation(sim)

    body = sim.bodies[0]
    sim.add_body_to_simulation(body)
    assert 11 == len(sim.sim.particles)


def test_run():
    sim = tools.create_test_simulation_for_solar_system()
    tools.add_test_asteroid_to_simulation(sim)
    mmr = sim.bodies[0].mmrs[0]
    sim.save = 'all'
    sim.plot = 'all'
    sim.save_summary = True
    sim.run()

    assert 10 == len(sim.bodies[0].angles[mmr.to_s()])
    assert 10 == len(sim.bodies[0].axis)
    assert 10 == len(sim.bodies[0].ecc)
    assert sim.bodies[0].statuses[mmr.to_s()] is not None
    assert sim.bodies[0].axis_filtered is not None
    assert sim.bodies[0].angles_filtered[mmr.to_s()] is not None

    assert Path(f'cache/tests/data-asteroid-{mmr.to_s()}.csv').exists() is True
    assert Path(f'cache/tests/asteroid_{mmr.to_s()}.png').exists() is True
    assert Path('cache/tests/summary.csv').exists() is True


@pytest.fixture
def saving_fixtures():
    # files index: data and plot for well determined and data and plot for undetermined.
    return [
        {'save': 'all', 'plot': 'all', 'files': [True, True, True, True]},
        {'save': 'nonzero', 'plot': 'nonzero', 'files': [True, True, True, True]},
        {'save': 'all', 'plot': 'resonant', 'files': [True, True, True, False]},
        {'save': 'resonant', 'plot': 'all', 'files': [True, False, True, True]},
        {'save': 'resonant', 'plot': 'resonant', 'files': [True, False, True, False]},
        {'save': 'all', 'plot': None, 'files': [True, True, False, False]},
        {'save': None, 'plot': 'all', 'files': [False, False, True, True]},
        {'save': None, 'plot': None, 'files': [False, False, False, False]},
        {'save': 'resonant', 'plot': None, 'files': [True, False, False, False]},
        {'save': None, 'plot': 'resonant', 'files': [False, False, True, False]},
    ]


def test_shall_save_and_plot_body(saving_fixtures):
    for saving_fixture in saving_fixtures:
        sim = tools.create_test_simulation_for_solar_system(save=saving_fixture['save'], plot=saving_fixture['plot'])

        elem = tools.get_3body_elements_sample()
        mmr = resonances.ThreeBody('4J-2S-1')
        sim.add_body(elem, mmr, name='asteroid')
        sim.add_body(elem, mmr, name='asteroid2')
        sim.bodies[0].statuses[mmr.to_s()] = 2
        sim.bodies[1].statuses[mmr.to_s()] = -1

        assert saving_fixture['files'][0] == sim.shall_save_body_in_mmr(sim.bodies[0], mmr)
        assert saving_fixture['files'][1] == sim.shall_save_body_in_mmr(sim.bodies[1], mmr)
        assert saving_fixture['files'][2] == sim.shall_plot_body_in_mmr(sim.bodies[0], mmr)
        assert saving_fixture['files'][3] == sim.shall_plot_body_in_mmr(sim.bodies[1], mmr)


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


def test_add_body_astdys():
    sim = tools.create_test_simulation_for_solar_system(save=True)
    sim.add_body('1', resonances.create_mmr('4J-2S-1'), name='asteroid', source='astdys')
    assert 'asteroid' == sim.bodies[0].name


def test_get_simulation_summary():
    sim = tools.create_test_simulation_for_solar_system(save=True)
    tools.add_test_asteroid_to_simulation(sim)
    sim.run()

    df = sim.get_simulation_summary()
    assert 1 == len(df)
    assert 14 == len(df.columns)
    assert 'asteroid' == df['name'].iloc[0]
    assert '4J-2S-1+0+0-1' == df['mmr'].iloc[0]


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


def test_process_status():
    sim = tools.create_test_simulation_for_solar_system()

    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    body.statuses[mmr.to_s()] = 2
    sim.save = 'all'
    assert sim.process_status(body, mmr, sim.save) is True
    sim.save = 'resonant'
    assert sim.process_status(body, mmr, sim.save) is True
    sim.save = 'nonzero'
    assert sim.process_status(body, mmr, sim.save) is True
    sim.save = 'candidates'
    assert sim.process_status(body, mmr, sim.save) is False

    body.statuses[mmr.to_s()] = 0
    sim.save = 'all'
    assert sim.process_status(body, mmr, sim.save) is True
    sim.save = 'resonant'
    assert sim.process_status(body, mmr, sim.save) is False
    sim.save = 'nonzero'
    assert sim.process_status(body, mmr, sim.save) is False
    sim.save = 'candidates'
    assert sim.process_status(body, mmr, sim.save) is False

    body.statuses[mmr.to_s()] = -2
    sim.save = 'all'
    assert sim.process_status(body, mmr, sim.save) is True
    sim.save = 'resonant'
    assert sim.process_status(body, mmr, sim.save) is False
    sim.save = 'nonzero'
    assert sim.process_status(body, mmr, sim.save) is True
    sim.save = 'candidates'
    assert sim.process_status(body, mmr, sim.save) is True

    body.statuses[mmr.to_s()] = 0
    sim.save = None
    assert sim.process_status(body, mmr, sim.save) is False
    sim.save = 'resonant'
    assert sim.process_status(body, mmr, sim.save) is False
    sim.save = 'nonzero'
    assert sim.process_status(body, mmr, sim.save) is False
    sim.save = 'candidates'
    assert sim.process_status(body, mmr, sim.save) is False
