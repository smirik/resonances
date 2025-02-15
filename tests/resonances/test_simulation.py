import datetime
import astdys
import numpy as np
import pandas as pd
import rebound
import tests.tools as tools
import shutil
from pathlib import Path
import pytest
import os

import resonances


@pytest.fixture(autouse=True)
def run_around_tests():
    Path('cache/tests').mkdir(parents=True, exist_ok=True)
    yield
    # shutil.rmtree('cache/tests')


def test_solar_system_full_filename():
    fixed_date_str = "2020-01-01 00:00:00"
    sim = resonances.Simulation(date=fixed_date_str)

    expected_timestamp = int(sim.date.timestamp())
    expected_filename = f"{os.getcwd()}/{resonances.config.get('SOLAR_SYSTEM_FILE')}".replace(".bin", f"-{expected_timestamp}.bin")
    actual_filename = sim.solar_system_full_filename()
    print(expected_filename)
    print(actual_filename)
    assert actual_filename == expected_filename, f"Expected '{expected_filename}', got '{actual_filename}'"


def test_create_solar_system_file_exists():
    """
    Covers the 'if solar_file.exists() and not force:' branch.
    We artificially create the file so that solar_system_full_filename()
    points to an existing file.
    """
    resonances.config.set('SOLAR_SYSTEM_FILE', 'tests/solar_test.bin')
    sim = resonances.Simulation()
    path = Path(sim.solar_system_full_filename())

    path.parent.mkdir(parents=True, exist_ok=True)
    sim.create_solar_system(force=True)
    assert sim.sim is not None
    assert path.exists()

    path.unlink()
    resonances.config.set('SOLAR_SYSTEM_FILE', 'cache/solar.bin')


def test_astdys_catalog_mismatch():
    # Force a date we know won't match the current astdys.datetime()
    mismatch_date = "1872-01-01"
    sim = resonances.Simulation(date=mismatch_date, source="astdys")
    assert sim.date.strftime("%Y-%m-%d") == "1872-01-01"


def test_integrator_safe_mode_default():
    sim = resonances.Simulation()
    assert sim.integrator_safe_mode == 1


def test_get_simulation_summary_exception():
    sim = resonances.Simulation()
    mmr = resonances.create_mmr('4J-2S-1')

    body = resonances.Body()
    body.mmrs = [mmr]
    body.periodogram_peaks_overlapping = {}  # missing mmr key => KeyError
    body.libration_metrics = {}  # missing mmr key => KeyError

    sim.bodies.append(body)

    df = sim.get_simulation_summary()
    assert isinstance(df, pd.DataFrame)


def test_identify_librations_exception(monkeypatch):
    """
    Covers the except-block inside identify_librations(), verifying we log an error
    and re-raise. We'll mock resonances.libration.body to raise an exception.
    """

    def mock_libration_body(simulation, body):
        raise ValueError("Mock error from libration.body()")

    sim = resonances.Simulation()
    sim.bodies.append(resonances.Body())  # A single body is enough
    monkeypatch.setattr(resonances.libration, "body", mock_libration_body)

    # Because identify_librations re-raises, we expect ValueError
    with pytest.raises(ValueError, match="Mock error"):
        sim.identify_librations()


def test_tmax_deleter():
    """
    Covers the lines in the tmax deleter by explicitly calling 'del sim.tmax'.
    """
    sim = resonances.Simulation(tmax=1000)
    assert sim.tmax == 1000
    del sim.tmax

    # Now, accessing sim.tmax should raise an AttributeError
    with pytest.raises(AttributeError):
        _ = sim.tmax


def test_init():
    tmax_default = resonances.config.get('INTEGRATION_TMAX')
    resonances.config.set('INTEGRATION_TMAX', 100)
    sim = resonances.Simulation()
    sim.Nout = 10
    sim.tmax_yrs = 100 / (2 * np.pi)
    resonances.config.set('INTEGRATION_TMAX', tmax_default)


def test_simulation_init():
    # Test default initialization
    sim = resonances.Simulation()
    assert sim.name.startswith('20')  # Current date format
    assert sim.Nout == 6283
    assert sim.source == resonances.config.get('DATA_SOURCE')
    assert sim.date.date() == datetime.datetime.today().date()
    assert sim.tmax == int(resonances.config.get('INTEGRATION_TMAX'))
    assert sim.integrator == resonances.config.get('INTEGRATION_INTEGRATOR')
    assert sim.dt == float(resonances.config.get('INTEGRATION_DT'))
    assert sim.integrator_corrector == int(resonances.config.get('INTEGRATION_CORRECTOR'))
    assert sim.save == resonances.config.get('SAVE_MODE')
    assert resonances.config.get('SAVE_PATH') in sim.save_path
    assert sim.save_summary == bool(resonances.config.get('SAVE_SUMMARY'))
    assert sim.plot == resonances.config.get('PLOT_MODE')
    assert sim.plot_type == resonances.config.get('PLOT_TYPE')
    assert resonances.config.get('PLOT_PATH') in sim.plot_path
    assert sim.image_type == resonances.config.get('PLOT_IMAGE_TYPE')
    assert sim.integrator_safe_mode == 1
    assert len(sim.planets) == 10  # Sun + 9 planets
    assert len(sim.bodies) == 0
    assert len(sim.times) == 0
    assert len(sim.particles) == 0

    # Test initialization with custom parameters
    custom_date = "2023-01-01"
    custom_sim = resonances.Simulation(
        name="test_sim",
        date=custom_date,
        source="nasa",
        tmax=1000,
        integrator="whfast",
        integrator_safe_mode=0,
        integrator_corrector=3,
        dt=0.1,
        save="all",
        save_path="custom_path",
        save_summary=True,
        plot="resonant",
        plot_path="plot_path",
        plot_type="custom",
        image_type="png",
    )

    assert custom_sim.name == "test_sim"
    assert custom_sim.date.strftime("%Y-%m-%d") == custom_date
    assert custom_sim.source == "nasa"
    assert custom_sim.tmax == 1000
    assert custom_sim.Nout == 10
    assert custom_sim.integrator == "whfast"
    assert custom_sim.integrator_safe_mode == 0
    assert custom_sim.integrator_corrector == 3
    assert custom_sim.dt == 0.1
    assert custom_sim.save == "all"
    assert custom_sim.save_path == "custom_path"
    assert custom_sim.save_summary is True
    assert custom_sim.plot == "resonant"
    assert custom_sim.plot_path == "plot_path"
    assert custom_sim.plot_type == "custom"
    assert custom_sim.image_type == "png"

    # Test AstDys source
    astdys_sim = resonances.Simulation(source="astdys")
    assert astdys_sim.source == "astdys"
    assert astdys_sim.bodies_date == astdys.datetime()

    # Test libration settings
    assert isinstance(custom_sim.oscillations_cutoff, float)
    assert isinstance(custom_sim.oscillations_filter_order, int)
    assert isinstance(custom_sim.periodogram_frequency_min, float)
    assert isinstance(custom_sim.periodogram_frequency_max, float)
    assert isinstance(custom_sim.periodogram_critical, float)
    assert isinstance(custom_sim.periodogram_soft, float)
    assert isinstance(custom_sim.libration_period_critical, int)
    assert isinstance(custom_sim.libration_monotony_critical, list)
    assert isinstance(custom_sim.libration_period_min, int)

    # Test tmax property
    custom_sim = resonances.Simulation(tmax=2000)
    assert custom_sim.tmax == 2000
    assert custom_sim.tmax_yrs == 2000 / (2 * np.pi)
    assert custom_sim.Nout == int(2000 / 100)


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
    sim.source = 'astdys'
    sim.add_body('1', resonances.create_mmr('4J-2S-1'), name='asteroid')
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
