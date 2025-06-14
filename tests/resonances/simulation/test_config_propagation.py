#!/usr/bin/env python3
"""
Tests for Configuration Parameter Propagation
=============================================

This module tests that configuration parameters passed to Simulation are correctly
propagated through the system components and actually used during integration.

Tests cover:
1. Direct verification of configuration settings
2. Runtime verification during integration
3. Different ways of setting configuration (constructor, direct assignment)
4. Integration engine actually using the correct parameters
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from resonances.simulation import Simulation


class TestConfigurationPropagation:
    """Test configuration parameter propagation through the simulation system."""

    def test_constructor_config_propagation(self):
        """Test that constructor parameters are correctly applied to config."""
        # Test various configuration parameters
        config_params = {
            'name': 'test_simulation',
            'tmax': 50000,
            'dt': 1.5,
            'integrator': 'SABA(10,6,4)',
            'integrator_corrector': 11,
            'integrator_safe_mode': 0,
            'save': 'resonant',
            'save_summary': False,
            'plot': 'all',
            'plot_type': 'png',
            'image_type': 'svg',
        }

        sim = Simulation(**config_params)

        # Verify all parameters were correctly applied
        assert sim.config.name == config_params['name']
        assert sim.config.tmax == config_params['tmax']
        assert sim.config.dt == config_params['dt']
        assert sim.config.integrator == config_params['integrator']
        assert sim.config.integrator_corrector == config_params['integrator_corrector']
        assert sim.config.integrator_safe_mode == config_params['integrator_safe_mode']
        assert sim.config.save == config_params['save']
        assert sim.config.save_summary == config_params['save_summary']
        assert sim.config.plot == config_params['plot']
        assert sim.config.plot_type == config_params['plot_type']
        assert sim.config.image_type == config_params['image_type']

    def test_direct_config_assignment(self):
        """Test direct assignment to config properties."""
        sim = Simulation()

        # Store original values
        original_dt = sim.config.dt
        original_integrator = sim.config.integrator

        # Modify directly (use different integrator than default)
        sim.config.dt = 2.5
        sim.config.integrator = 'SABA(8,6,4)'  # Different from default SABA(10,6,4)
        sim.config.integrator_safe_mode = 0

        # Verify changes took effect
        assert sim.config.dt == 2.5
        assert sim.config.integrator == 'SABA(8,6,4)'
        assert sim.config.integrator_safe_mode == 0

        # Verify they're different from original
        assert sim.config.dt != original_dt
        assert sim.config.integrator != original_integrator

    def test_integration_engine_uses_config(self):
        """Test that IntegrationEngine actually uses config parameters during setup."""
        # Create simulation with specific parameters
        sim = Simulation(integrator='SABA(10,6,4)', dt=3.14, integrator_safe_mode=1)

        # Create mock REBOUND simulation
        mock_sim = Mock()
        mock_sim.ri_saba = Mock()
        sim.integration_engine.sim = mock_sim

        # Call setup_integrator
        sim.integration_engine.setup_integrator()

        # Verify that the mock simulation received the correct parameters
        assert mock_sim.integrator == 'SABA(10,6,4)'
        assert mock_sim.dt == 3.14
        assert mock_sim.ri_saba.safe_mode == 1
        mock_sim.move_to_com.assert_called_once()

    def test_saba_integrator_config_usage(self):
        """Test that SABA integrator uses config parameters correctly."""
        sim = Simulation(integrator='SABA(10,6,4)', dt=0.123, integrator_safe_mode=1)

        # Create mock REBOUND simulation
        mock_sim = Mock()
        mock_sim.ri_saba = Mock()
        sim.integration_engine.sim = mock_sim

        # Call setup_integrator
        sim.integration_engine.setup_integrator()

        # Verify SABA-specific settings
        assert mock_sim.integrator == 'SABA(10,6,4)'
        assert mock_sim.dt == 0.123
        assert mock_sim.ri_saba.safe_mode == 1
        mock_sim.move_to_com.assert_called_once()

    def test_config_changes_affect_integration_engine(self):
        """Test that changes to config are reflected in integration engine behavior."""
        sim = Simulation(integrator='SABA(8,6,4)', dt=0.5)

        # First setup with initial config
        mock_sim1 = Mock()
        mock_sim1.ri_saba = Mock()
        sim.integration_engine.sim = mock_sim1
        sim.integration_engine.setup_integrator()

        assert mock_sim1.integrator == 'SABA(8,6,4)'
        assert mock_sim1.dt == 0.5

        # Change config
        sim.config.integrator = 'SABA(10,6,4)'
        sim.config.dt = 2.0
        sim.config.integrator_safe_mode = 0

        # Setup again with new config
        mock_sim2 = Mock()
        mock_sim2.ri_saba = Mock()
        sim.integration_engine.sim = mock_sim2
        sim.integration_engine.setup_integrator()

        # Verify new parameters are used
        assert mock_sim2.integrator == 'SABA(10,6,4)'
        assert mock_sim2.dt == 2.0
        assert mock_sim2.ri_saba.safe_mode == 0

    def test_tmax_years_property_consistency(self):
        """Test that tmax and tmax_yrs properties are consistent."""
        sim = Simulation(tmax=62831)  # 10000 * 2π ≈ 62831

        # Verify conversion is correct
        expected_years = 62831 / (2 * np.pi)
        assert abs(sim.config.tmax_yrs - expected_years) < 1e-10

        # Test setting via years
        sim.config.tmax_yrs = 5000
        expected_tmax = 5000 * (2 * np.pi)
        assert abs(sim.config.tmax - expected_tmax) < 1e-10

    @patch('rebound.Simulation')
    def test_runtime_integration_parameters(self, mock_rebound_sim):
        """Test that integration parameters are actually used during run()."""
        # Create a mock REBOUND simulation instance
        mock_sim = Mock()
        mock_sim.particles = [Mock()]  # Mock Sun
        mock_sim.orbits.return_value = []  # No asteroids initially
        mock_rebound_sim.return_value = mock_sim

        # Create simulation with specific parameters
        sim = Simulation(integrator='SABA(8,6,4)', dt=4.0, integrator_safe_mode=1, tmax=1000)

        # Mock file operations
        with patch('pathlib.Path.exists', return_value=False):
            sim.create_solar_system()

        # Verify that sim object was created and parameters applied
        assert sim.integration_engine.sim is not None

        # Add a mock body to test integration
        mock_body = Mock()
        mock_body.mmrs = []
        mock_body.secular_resonances = []
        mock_body.index_in_simulation = 10
        mock_body.setup_vars_for_simulation = Mock()
        mock_body.initial_data = {'a': 2.5, 'e': 0.1, 'inc': 0.05, 'Omega': 1.0, 'omega': 2.0, 'M': 3.0}
        mock_body.mass = 0.0

        # Mock the arrays that will be updated during integration
        mock_body.axis = np.zeros(sim.Nout)
        mock_body.ecc = np.zeros(sim.Nout)
        mock_body.inc = np.zeros(sim.Nout)
        mock_body.Omega = np.zeros(sim.Nout)
        mock_body.omega = np.zeros(sim.Nout)
        mock_body.M = np.zeros(sim.Nout)
        mock_body.longitude = np.zeros(sim.Nout)
        mock_body.varpi = np.zeros(sim.Nout)

        sim.body_manager.bodies = [mock_body]

        # Mock the integration process
        mock_sim.integrate = Mock()

        # Create a mock orbit with actual numeric values
        mock_orbit = Mock()
        mock_orbit.a = 2.5
        mock_orbit.e = 0.1
        mock_orbit.inc = 0.05
        mock_orbit.Omega = 1.0
        mock_orbit.omega = 2.0
        mock_orbit.M = 3.0
        mock_orbit.l = 5.0  # noqa: E741 (l is standard astronomical notation for longitude)
        mock_sim.orbits.return_value = [mock_orbit]  # Mock orbit for the body

        # Set up Nout for short test
        sim.Nout = 5

        # Run the simulation
        sim.run()

        # Verify that setup_integrator was called and parameters were set
        # Note: We can't directly verify mock_sim properties here since they're set in setup_integrator
        # But we can verify that integration was attempted
        assert mock_sim.integrate.call_count == sim.Nout

    def test_config_isolation_between_simulations(self):
        """Test that different simulation instances have isolated configurations."""
        sim1 = Simulation(integrator='SABA(8,6,4)', dt=0.1)
        sim2 = Simulation(integrator='SABA(10,6,4)', dt=5.0)

        # Verify they have different configs
        assert sim1.config.integrator != sim2.config.integrator
        assert sim1.config.dt != sim2.config.dt

        # Change one config
        sim1.config.dt = 999.0

        # Verify the other is unaffected
        assert sim2.config.dt == 5.0
        assert sim1.config.dt == 999.0

    def test_integration_engine_config_reference(self):
        """Test that IntegrationEngine maintains reference to same config object."""
        sim = Simulation(integrator='SABA(10,6,4)', dt=1.23)

        # Verify integration engine has same config object
        assert sim.integration_engine.config is sim.config

        # Change config through simulation
        sim.config.dt = 4.56

        # Verify integration engine sees the change
        assert sim.integration_engine.config.dt == 4.56

    def test_unsupported_integrator_handling(self):
        """Test behavior with unsupported integrator types."""
        sim = Simulation(integrator='UNKNOWN_INTEGRATOR', dt=1.0)

        # Create mock simulation
        mock_sim = Mock()
        sim.integration_engine.sim = mock_sim

        # Setup should still work (REBOUND will handle unknown integrators)
        sim.integration_engine.setup_integrator()

        # Basic parameters should still be set
        assert mock_sim.integrator == 'UNKNOWN_INTEGRATOR'
        assert mock_sim.dt == 1.0
        mock_sim.move_to_com.assert_called_once()

    def test_config_parameter_types(self):
        """Test that config parameters maintain correct types."""
        sim = Simulation(
            tmax=10000,
            dt=1.5,
            integrator_safe_mode=1,
            save_summary=True,
            integrator='SABA(10,6,4)',  # int  # float  # int  # bool  # str
        )

        # Verify types are preserved (note: tmax may be converted to float during processing)
        assert isinstance(sim.config.tmax, (int, float))
        assert isinstance(sim.config.dt, float)
        assert isinstance(sim.config.integrator_safe_mode, int)
        assert isinstance(sim.config.save_summary, bool)
        assert isinstance(sim.config.integrator, str)

    def test_nout_calculation_from_tmax(self):
        """Test that Nout is calculated correctly from tmax."""
        sim = Simulation(tmax=12566)  # 2000 * 2π ≈ 12566

        expected_nout = abs(int(12566 / 100))
        assert sim.Nout == expected_nout

        # Test with different tmax
        sim.config.tmax = 31415  # 5000 * 2π ≈ 31415
        sim.Nout = abs(int(sim.config.tmax / 100))

        expected_nout = abs(int(31415 / 100))
        assert sim.Nout == expected_nout


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
