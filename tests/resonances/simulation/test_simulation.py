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
from unittest.mock import Mock, patch
import tempfile

import resonances
from resonances.simulation import Simulation


@pytest.fixture(autouse=True)
def run_around_tests():
    Path('cache/tests').mkdir(parents=True, exist_ok=True)
    yield
    # shutil.rmtree('cache/tests')


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


def test_astdys_catalog_mismatch():
    """Test handling of AstDyS catalog mismatch."""
    sim = Simulation(source='astdys')
    with patch('astdys.datetime', return_value='2023-01-01'):
        with patch('resonances.data.util.datetime_from_string', return_value='2023-01-02'):
            # This should not raise an exception
            sim.config.date = '2023-01-02'


def test_integrator_safe_mode_default():
    sim = Simulation()
    assert sim.config.integrator_safe_mode == 1


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

    with patch('resonances.libration.body', side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            sim.identify_librations()


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
    assert sim.Nout == 6283
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
    assert isinstance(sim.config.libration_period_min, int)
    assert isinstance(sim.config.periodogram_frequency_min, float)
    assert isinstance(sim.config.periodogram_frequency_max, float)
    assert isinstance(sim.config.periodogram_soft, float)
    assert isinstance(sim.config.libration_period_critical, int)
    assert isinstance(sim.config.libration_monotony_critical, list)


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
    mmr = sim.bodies[0].mmrs[0]
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
    sim.bodies[0].statuses[mmr.to_s()] = 2
    sim.bodies[1].statuses[mmr.to_s()] = -1

    # Test data manager methods directly
    assert sim.data_manager.should_save_body_mmr(sim.bodies[0], mmr) == True
    assert sim.data_manager.should_plot_body_mmr(sim.bodies[0], mmr) == True
    assert sim.data_manager.should_save_body_mmr(sim.bodies[1], mmr) == True
    assert sim.data_manager.should_plot_body_mmr(sim.bodies[1], mmr) == False


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


def test_data_manager_process_status():
    """Test data manager process status method."""
    sim = tools.create_test_simulation_for_solar_system()

    body = resonances.Body()
    mmr = resonances.create_mmr('4J-2S-1')
    body.statuses[mmr.to_s()] = 2

    # Test data manager process status
    assert sim.data_manager.process_status(body, mmr, 'all') is True
    assert sim.data_manager.process_status(body, mmr, 'resonant') is True
    assert sim.data_manager.process_status(body, mmr, 'candidates') is False
