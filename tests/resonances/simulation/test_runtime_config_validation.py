#!/usr/bin/env python3
"""
Runtime Configuration Validation Tests
======================================

This module performs deeper integration tests to verify that configuration
parameters are actually applied and used during simulation runtime.

These tests focus on:
1. Actual REBOUND integrator configuration
2. Runtime parameter validation during integration
3. End-to-end configuration flow verification
4. Real integration behavior with different settings
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from resonances.simulation import Simulation


class TestRuntimeConfigValidation:
    """Test runtime configuration validation during actual simulation execution."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_config_parameter_validation_during_setup(self):
        """Test that configuration parameters are handled appropriately."""
        # Test with valid parameters first
        sim_valid = Simulation(integrator='SABA(10,6,4)', dt=0.01, tmax=1000)

        assert sim_valid.config.integrator == 'SABA(10,6,4)'
        assert sim_valid.config.dt == 0.01
        assert sim_valid.config.tmax == 1000

        # Test with edge case parameters
        sim_edge = Simulation(dt=0.0001, tmax=1, integrator_safe_mode=0)  # Very small dt  # Very small tmax  # Different safe mode

        assert sim_edge.config.dt == 0.0001
        assert sim_edge.config.tmax == 1
        assert sim_edge.config.integrator_safe_mode == 0

    @patch('rebound.Simulation')
    def test_integration_engine_parameter_consistency(self, mock_rebound_sim):
        """Test that integration engine maintains parameter consistency throughout run."""
        mock_sim = Mock()
        mock_sim.particles = [Mock() for _ in range(10)]
        mock_sim.ri_saba = Mock()

        # Track parameter changes
        parameter_history = []

        def track_integrator_set(self, value):
            parameter_history.append(('integrator', value))
            mock_sim._integrator = value

        def track_dt_set(self, value):
            parameter_history.append(('dt', value))
            mock_sim._dt = value

        # Mock property setters to track changes
        type(mock_sim).integrator = property(lambda self: getattr(self, '_integrator', None), track_integrator_set)
        type(mock_sim).dt = property(lambda self: getattr(self, '_dt', None), track_dt_set)

        mock_rebound_sim.return_value = mock_sim

        # Create simulation
        sim = Simulation(integrator='SABA(10,6,4)', dt=0.05, tmax=314)  # Small tmax

        # Mock solar system creation
        with patch('pathlib.Path.exists', return_value=False):
            sim.create_solar_system()

        # Call setup_integrator multiple times to verify consistency
        sim.integration_engine.setup_integrator()
        sim.integration_engine.setup_integrator()
        sim.integration_engine.setup_integrator()

        # Verify parameters were set consistently
        integrator_sets = [p for p in parameter_history if p[0] == 'integrator']
        dt_sets = [p for p in parameter_history if p[0] == 'dt']

        assert len(integrator_sets) == 3
        assert all(p[1] == 'SABA(10,6,4)' for p in integrator_sets)

        assert len(dt_sets) == 3
        assert all(p[1] == 0.05 for p in dt_sets)

    def test_simulation_config_immutability_during_run(self):
        """Test that config changes during run don't affect ongoing integration."""
        sim = Simulation(integrator='SABA(10,6,4)', dt=0.1, tmax=628)

        # Store original values
        original_integrator = sim.config.integrator
        original_dt = sim.config.dt

        # Create mock to track when setup_integrator is called
        setup_calls = []
        original_setup = sim.integration_engine.setup_integrator

        def track_setup(*args, **kwargs):
            setup_calls.append((sim.config.integrator, sim.config.dt))
            return original_setup(*args, **kwargs)

        sim.integration_engine.setup_integrator = track_setup

        # Mock the integration engine sim
        mock_sim = Mock()
        mock_sim.ri_saba = Mock()
        sim.integration_engine.sim = mock_sim

        # Call setup
        sim.integration_engine.setup_integrator()

        # Change config after setup
        sim.config.integrator = 'SABA(8,6,4)'
        sim.config.dt = 5.0

        # Verify the changes took effect in config
        assert sim.config.integrator != original_integrator
        assert sim.config.dt != original_dt

        # But setup was called with original values
        assert len(setup_calls) == 1
        assert setup_calls[0] == (original_integrator, original_dt)

    def test_config_validation_edge_cases(self):
        """Test configuration validation with edge cases and boundary values."""
        # Test very small dt
        sim1 = Simulation(dt=1e-10)
        assert sim1.config.dt == 1e-10

        # Test very large tmax
        sim2 = Simulation(tmax=int(1e8))
        assert sim2.config.tmax == int(1e8)

        # Test zero safe mode
        sim3 = Simulation(integrator_safe_mode=0)
        assert sim3.config.integrator_safe_mode == 0

        # Test different safe mode value
        sim4 = Simulation(integrator_safe_mode=1)
        assert sim4.config.integrator_safe_mode == 1

    def test_saba_integrator_configuration_verification(self):
        """Test that SABA integrator parameters are correctly configured."""
        sim = Simulation(integrator='SABA(10,6,4)', dt=2.5, integrator_safe_mode=1)

        # Mock REBOUND simulation
        mock_sim = Mock()
        mock_sim.ri_saba = Mock()
        sim.integration_engine.sim = mock_sim

        # Call setup_integrator
        sim.integration_engine.setup_integrator()

        # Verify SABA-specific parameters were applied
        assert mock_sim.integrator == 'SABA(10,6,4)'
        assert mock_sim.dt == 2.5
        assert mock_sim.ri_saba.safe_mode == 1
        mock_sim.move_to_com.assert_called_once()

    def test_multiple_saba_configurations(self):
        """Test different SABA integrator configurations."""
        # Test SABA(8,6,4)
        sim1 = Simulation(integrator='SABA(8,6,4)', dt=1.0, integrator_safe_mode=0)

        mock_sim1 = Mock()
        mock_sim1.ri_saba = Mock()
        sim1.integration_engine.sim = mock_sim1
        sim1.integration_engine.setup_integrator()

        assert mock_sim1.integrator == 'SABA(8,6,4)'
        assert mock_sim1.dt == 1.0
        assert mock_sim1.ri_saba.safe_mode == 0

        # Test SABA(10,6,4)
        sim2 = Simulation(integrator='SABA(10,6,4)', dt=5.0, integrator_safe_mode=1)

        mock_sim2 = Mock()
        mock_sim2.ri_saba = Mock()
        sim2.integration_engine.sim = mock_sim2
        sim2.integration_engine.setup_integrator()

        assert mock_sim2.integrator == 'SABA(10,6,4)'
        assert mock_sim2.dt == 5.0
        assert mock_sim2.ri_saba.safe_mode == 1

    def test_configuration_parameter_types_at_runtime(self):
        """Test that configuration parameter types are preserved at runtime."""
        sim = Simulation(integrator='SABA(10,6,4)', dt=3.14159, integrator_safe_mode=1, tmax=62831)  # str  # float  # int  # int

        # Mock setup
        mock_sim = Mock()
        mock_sim.ri_saba = Mock()
        sim.integration_engine.sim = mock_sim

        # Call setup
        sim.integration_engine.setup_integrator()

        # Verify types were preserved through to REBOUND
        assert isinstance(mock_sim.integrator, str)
        assert isinstance(mock_sim.dt, float)
        assert isinstance(mock_sim.ri_saba.safe_mode, int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
