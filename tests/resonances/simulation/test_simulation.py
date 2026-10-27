#!/usr/bin/env python3
import datetime  # noqa: F401 - used in patches
import rebound  # noqa: F401 - used in patches
import tests.tools as tools
import pytest
from unittest.mock import Mock, patch

import resonances
from resonances.simulation import Simulation


def test_solar_system_full_filename():
    """Test solar system filename generation via integration engine."""
    sim = Simulation()
    filename = sim.integration_engine._solar_system_filename()
    assert filename.endswith('.bin')
    assert 'solar' in filename.lower()


def test_create_solar_system_file_exists():
    """Test solar system creation when file exists."""
    sim = Simulation()
    with patch('pathlib.Path.exists', return_value=True):
        with patch('rebound.Simulation') as mock_sim:
            mock_instance = Mock()
            mock_sim.return_value = mock_instance
            mock_instance.load = Mock()
            sim.create_solar_system()
            # Just verify no exception was raised


def test_integrator_safe_mode_default():
    sim = Simulation()
    assert sim.config.integration_safe_mode is None


def test_get_simulation_summary_exception():
    """Test simulation summary generation with exception handling via data manager."""
    sim = tools.create_test_simulation_for_solar_system()
    tools.add_test_asteroid_to_simulation(sim)

    # Mock an exception in the summary generation
    with patch.object(sim.bodies[0], 'mmrs', []):
        summary = sim.data_manager.get_simulation_summary(sim.bodies)
        assert len(summary) == 0  # Should handle gracefully


def test_identify_librations_exception():
    """Test libration identification with exception handling."""
    sim = tools.create_test_simulation_for_solar_system()
    tools.add_test_asteroid_to_simulation(sim)

    # identify_librations analyses an MMR with analyse_mmr on the angles computed during sim.run()
    with patch('resonances.simulation.simulation.analyse_mmr', side_effect=RuntimeError("Test error")):
        with pytest.raises(RuntimeError):
            sim.run()


def test_tmax_access():
    """Test tmax config access."""
    sim = Simulation(tmax=1000)
    assert sim.config.tmax == 1000


def test_init():
    """Test simulation initialization."""
    sim = Simulation()
    assert sim.config is not None
    assert sim.body_manager is not None
    assert sim.integration_engine is not None
    assert sim.data_manager is not None


def test_simulation_init():
    # Test default initialization
    sim = Simulation()
    assert sim.config.name.startswith('20')  # Current date format
    assert sim.config.Nout == 6283
    assert sim.config.source == resonances.config.get('DATA_SOURCE')

    # Test custom initialization
    custom_sim = Simulation(name='test_sim', tmax=10000, dt=0.5, integrator='WHFAST', save='resonant', plot='all')

    assert custom_sim.config.name == 'test_sim'
    assert custom_sim.config.tmax == 10000
    assert custom_sim.config.dt == 0.5
    assert custom_sim.config.integrator == 'WHFAST'
    assert custom_sim.config.save == 'resonant'
    assert custom_sim.config.plot == 'all'


def test_libration_properties():
    """Test that libration properties are accessible via config."""
    sim = Simulation()

    # Test that all libration properties are accessible via config
    assert isinstance(sim.config.oscillations_cutoff, float)
    assert isinstance(sim.config.libration_period_min, float)
    assert isinstance(sim.config.periodogram_frequency_min, float)
    assert isinstance(sim.config.periodogram_frequency_max, float)
    assert isinstance(sim.config.periodogram_soft, float)


def test_solar_system():
    sim = tools.create_test_simulation_for_solar_system()
    assert sim.integration_engine.sim is not None
    assert sim.integration_engine.sim.N > 0  # Should have planets


def test_add_body():
    sim = tools.create_test_simulation_for_solar_system()

    # Test adding body with MMR
    elem = tools.get_3body_elements_sample()
    mmr = resonances.ThreeBody('4J-2S-1')
    sim.add_body(elem, mmr, name='test_asteroid')

    assert len(sim.bodies) == 1
    assert sim.bodies[0].name == 'test_asteroid'
    assert len(sim.bodies[0].mmrs) == 1


def test_add_bodies():
    sim = tools.create_test_simulation_for_solar_system()

    # Test adding multiple bodies
    mmr = resonances.ThreeBody('4J-2S-1')
    sim.add_bodies(['463', '490'], mmr, prefix='test')

    assert len(sim.bodies) == 2
    assert sim.bodies[0].name == 'test_463'
    assert sim.bodies[1].name == 'test_490'


def test_add_bodies_to_simulation():
    sim = tools.create_test_simulation_for_solar_system()
    tools.add_test_asteroid_to_simulation(sim)

    # Bodies should be added to the REBOUND simulation
    initial_n = sim.integration_engine.sim.N
    sim.body_manager.add_bodies_to_simulation(sim.integration_engine.sim)

    # Should have added the asteroid
    assert sim.integration_engine.sim.N > initial_n


def test_run():
    sim = tools.create_test_simulation_for_solar_system()
    tools.add_test_asteroid_to_simulation(sim)
    sim.config.save = 'all'
    sim.config.plot = 'all'
    sim.config.save_summary = True
    sim.run()


def test_data_manager_methods():
    """Test that data manager methods work correctly."""
    sim = tools.create_test_simulation_for_solar_system(save='all', plot='resonant')

    elem = tools.get_3body_elements_sample()
    mmr = resonances.ThreeBody('4J-2S-1')
    sim.add_body(elem, mmr, name='asteroid')
    sim.add_body(elem, mmr, name='asteroid2')
    sim.bodies[0].statuses[mmr.to_s()] = 'resonant'
    sim.bodies[1].statuses[mmr.to_s()] = 'uncertain'

    # Test data manager methods directly
    assert sim.data_manager.should_save_body(sim.bodies[0], mmr) is True
    assert sim.data_manager.should_plot_body(sim.bodies[0], mmr) is True
    assert sim.data_manager.should_save_body(sim.bodies[1], mmr) is True
    assert sim.data_manager.should_plot_body(sim.bodies[1], mmr) is False


def test_saving_summary():
    sim = tools.create_test_simulation_for_solar_system()
    sim.config.save = True
    sim.config.save_summary = True
    tools.add_test_asteroid_to_simulation(sim)
    sim.run()


def test_add_body_by_number():
    """Test adding body by number (simplified test without external dependencies)."""
    sim = tools.create_test_simulation_for_solar_system()

    # Test adding body with elements dict (this is what the body manager does internally)
    elem = tools.get_3body_elements_sample()
    mmr = resonances.ThreeBody('4J-2S-1')
    sim.add_body(elem, mmr, name='Lola')

    assert len(sim.bodies) == 1
    assert sim.bodies[0].name == 'Lola'


def test_get_simulation_summary():
    sim = tools.create_test_simulation_for_solar_system()
    sim.config.save = True
    tools.add_test_asteroid_to_simulation(sim)
    sim.run()

    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    assert len(summary) > 0
    assert 'name' in summary.columns
    assert 'resonance' in summary.columns
    assert 'c1' in summary.columns
    assert 'c2' in summary.columns
    assert 'h_quad' in summary.columns


def test_list_of_planets():
    sim = Simulation()
    planets = sim.body_manager.planets
    assert isinstance(planets, list)
    assert len(planets) > 0


def test_get_index_of_planets():
    sim = Simulation()
    indices = sim.body_manager.get_index_of_planets(['Jupiter', 'Saturn'])
    assert isinstance(indices, list)
    assert len(indices) == 2


def test_data_manager_universal_methods():
    """Test data manager universal methods."""
    sim = tools.create_test_simulation_for_solar_system(save='all', plot='resonant')

    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    body.statuses[mmr.to_s()] = 'resonant'

    # Test universal methods
    assert sim.data_manager.should_save_body(body, mmr) is True
    assert sim.data_manager.should_plot_body(body, mmr) is True

    # Test with a candidate that is not resonant
    body.statuses[mmr.to_s()] = 'uncertain'
    assert sim.data_manager.should_save_body(body, mmr) is True  # save='all'
    assert sim.data_manager.should_plot_body(body, mmr) is False  # plot='resonant' and status uncertain


def test_a_lost_orbit_is_chaotic_and_the_run_goes_on(tmp_path):
    """An orbit that turns NaN halfway (what the elements of a particle the integrator lost look like) must not
    stop the run at the filter: it is saved and every resonance of it is invalid (the old CHAOTIC -99: non-finite
    elements), while the other body of the same run is classified as usual."""
    import numpy as np
    import pandas as pd
    import astdys.util

    sim = Simulation(
        date=astdys.util.convert_mjd_to_datetime(60000),
        integration_years=2000,
        Nout=400,
        libration_period_min=10,
        save='all',
        plot=None,
        save_summary=True,
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
    )
    sim.create_solar_system()
    elements = tools.get_3body_elements_sample()
    resonances_list = ['4J-2S-1', 'g-g6', 'LK']
    sim.add_body(elements, resonances_list, name='kept')
    sim.add_body(elements, resonances_list, name='lost')
    sim.times = np.linspace(0.0, sim.config.tmax, sim.config.Nout)
    sim.body_manager.add_bodies_to_simulation(sim.integration_engine.sim)
    sim.integration_engine.run_integration(sim.bodies, sim.times)

    lost = sim.bodies[1]
    for series in [lost.axis, lost.ecc, lost.inc, lost.Omega, lost.omega, lost.M, lost.longitude, lost.varpi]:
        series[200:] = np.nan
    for series in list(lost.angles_unwrapped.values()) + list(lost.angles.values()):
        series[200:] = np.nan

    # The steps of Simulation._run_single after the integration
    sim.prepare_angles()
    sim.identify_librations()
    sim.build_periodograms()
    sim.data_manager.save_data(sim.bodies, sim.times, sim)

    summary = pd.read_csv(tmp_path / 'summary.csv')
    assert sorted(summary.loc[summary['name'] == 'lost', 'status']) == ['invalid', 'invalid', 'invalid']
    assert sorted(summary.loc[summary['name'] == 'lost', 'status_code']) == [-99, -99, -99]
    assert len(summary.loc[summary['name'] == 'kept']) == 3
    assert (summary.loc[summary['name'] == 'kept', 'status'] != 'invalid').all()
    assert np.isfinite(sim.bodies[0].angles_unwrapped['4J-2S-1+0+0-1']).all()
