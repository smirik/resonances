#!/usr/bin/env python3
"""
Core Configuration Validation Tests
===================================

Focused tests to verify the core issue: whether configuration parameters
passed to Simulation constructor are actually used by the integration engine.

This addresses the specific concern about configuration parameter propagation
through the refactored system.
"""

import pytest
from unittest.mock import Mock

from resonances.simulation import Simulation


class TestCoreConfigValidation:
    """Core tests to verify configuration parameters flow through the system."""

    def test_config_constructor_to_integration_engine_flow(self):
        """
        TEST 1: Verify config parameters flow from constructor to integration engine.
        This is the core issue - are constructor parameters actually used?
        """
        # Create simulation with specific parameters
        sim = Simulation(integrator='SABA(10,6,4)', dt=3.14159, integrator_corrector=17, integrator_safe_mode=0, tmax=50000)

        # Verify parameters are in config
        assert sim.config.integrator == 'SABA(10,6,4)'
        assert sim.config.dt == 3.14159
        assert sim.config.integrator_corrector == 17
        assert sim.config.integrator_safe_mode == 0
        assert sim.config.tmax == 50000

        # Verify integration engine has reference to same config
        assert sim.integration_engine.config is sim.config
        assert sim.integration_engine.config.integrator == 'SABA(10,6,4)'
        assert sim.integration_engine.config.dt == 3.14159

    def test_setup_integrator_uses_config_parameters(self):
        """
        TEST 2: Verify setup_integrator actually uses config parameters.
        This tests the critical path where config should affect REBOUND.
        """
        # Test SABA integrator
        sim_saba = Simulation(integrator='SABA(8,6,4)', dt=2.5, integrator_safe_mode=1)

        # Mock REBOUND simulation
        mock_sim_saba = Mock()
        mock_sim_saba.ri_saba = Mock()
        sim_saba.integration_engine.sim = mock_sim_saba

        # Call setup_integrator
        sim_saba.integration_engine.setup_integrator()

        # Verify REBOUND simulation received correct parameters
        assert mock_sim_saba.integrator == 'SABA(8,6,4)'
        assert mock_sim_saba.dt == 2.5
        assert mock_sim_saba.ri_saba.safe_mode == 1
        mock_sim_saba.move_to_com.assert_called_once()

        # Test SABA integrator (second instance)
        sim_saba2 = Simulation(integrator='SABA(10,6,4)', dt=0.01, integrator_safe_mode=0)

        # Mock REBOUND simulation
        mock_sim_saba2 = Mock()
        mock_sim_saba2.ri_saba = Mock()
        sim_saba2.integration_engine.sim = mock_sim_saba2

        # Call setup_integrator
        sim_saba2.integration_engine.setup_integrator()

        # Verify SABA-specific parameters
        assert mock_sim_saba2.integrator == 'SABA(10,6,4)'
        assert mock_sim_saba2.dt == 0.01
        assert mock_sim_saba2.ri_saba.safe_mode == 0
        mock_sim_saba2.move_to_com.assert_called_once()

    def test_config_changes_propagate_to_integration_engine(self):
        """
        TEST 3: Verify that changes to config are reflected in integration engine.
        """
        sim = Simulation(integrator='IAS15', dt=1.0)

        # Initial verification
        assert sim.config.integrator == 'IAS15'
        assert sim.config.dt == 1.0
        assert sim.integration_engine.config.integrator == 'IAS15'
        assert sim.integration_engine.config.dt == 1.0

        # Change config
        sim.config.integrator = 'SABA(10,6,4)'
        sim.config.dt = 0.05
        sim.config.integrator_safe_mode = 0

        # Verify integration engine sees changes
        assert sim.integration_engine.config.integrator == 'SABA(10,6,4)'
        assert sim.integration_engine.config.dt == 0.05
        assert sim.integration_engine.config.integrator_safe_mode == 0

    def test_multiple_simulations_isolated_configs(self):
        """
        TEST 4: Verify different simulations have isolated configurations.
        """
        sim1 = Simulation(integrator='SABA(8,6,4)', dt=0.01, tmax=1000)
        sim2 = Simulation(integrator='SABA(10,6,4)', dt=5.0, tmax=50000)

        # Verify they have different configs
        assert sim1.config.integrator != sim2.config.integrator
        assert sim1.config.dt != sim2.config.dt
        assert sim1.config.tmax != sim2.config.tmax

        # Verify integration engines have correct configs
        assert sim1.integration_engine.config.integrator == 'SABA(8,6,4)'
        assert sim2.integration_engine.config.integrator == 'SABA(10,6,4)'

        # Change one config
        sim1.config.integrator = 'IAS15'

        # Verify other is unaffected
        assert sim1.config.integrator == 'IAS15'
        assert sim2.config.integrator == 'SABA(10,6,4)'

    def test_all_configurable_parameters_propagate(self):
        """
        TEST 5: Comprehensive test of all integration-related config parameters.
        """
        config_params = {
            'integrator': 'SABA(10,6,4)',
            'dt': 0.123,
            'integrator_safe_mode': 1,
            'tmax': 31415,
            'save': 'resonant',
            'plot': 'all',
        }

        sim = Simulation(**config_params)

        # Verify all parameters are correctly applied
        for param, value in config_params.items():
            config_value = getattr(sim.config, param)
            if isinstance(value, (int, float)) and isinstance(config_value, (int, float)):
                # Handle floating point precision for numeric values
                assert abs(config_value - value) < 1e-10
            else:
                assert config_value == value

        # Verify integration engine has same config reference
        assert sim.integration_engine.config is sim.config

        # Test that integration engine actually uses these parameters
        mock_sim = Mock()
        mock_sim.ri_saba = Mock()
        sim.integration_engine.sim = mock_sim

        sim.integration_engine.setup_integrator()

        # Verify critical parameters were applied to REBOUND
        assert mock_sim.integrator == config_params['integrator']
        assert mock_sim.dt == config_params['dt']
        assert mock_sim.ri_saba.safe_mode == config_params['integrator_safe_mode']

    def test_config_parameter_types_preserved(self):
        """
        TEST 6: Verify parameter types are preserved through the config system.
        """
        sim = Simulation(
            tmax=10000,
            dt=1.5,
            integrator_safe_mode=1,
            save_summary=True,
            integrator='SABA(10,6,4)',  # int  # float  # int  # bool  # str
        )

        # Verify types are preserved appropriately
        assert isinstance(sim.config.tmax, (int, float))  # May be converted to float
        assert isinstance(sim.config.dt, float)
        assert isinstance(sim.config.integrator_safe_mode, int)
        assert isinstance(sim.config.save_summary, bool)
        assert isinstance(sim.config.integrator, str)

    def test_integration_engine_consistent_parameter_usage(self):
        """
        TEST 7: Verify setup_integrator consistently uses config parameters.
        """
        sim = Simulation(integrator='SABA(10,6,4)', dt=1.23, integrator_safe_mode=0)

        # Track calls to verify consistency
        mock_sim = Mock()
        mock_sim.ri_saba = Mock()
        sim.integration_engine.sim = mock_sim

        # Call setup multiple times
        sim.integration_engine.setup_integrator()
        sim.integration_engine.setup_integrator()
        sim.integration_engine.setup_integrator()

        # Verify parameters are consistently applied
        assert mock_sim.integrator == 'SABA(10,6,4)'
        assert mock_sim.dt == 1.23
        assert mock_sim.ri_saba.safe_mode == 0
        assert mock_sim.move_to_com.call_count == 3  # Called each time

    def test_demonstrable_configuration_issue(self):
        """
        TEST 8: Demonstrate the specific issue and verify it's fixed.
        This test explicitly shows the problem would occur if config wasn't used.
        """
        # Create simulation with non-default parameters
        sim = Simulation(integrator='SABA(8,6,4)', dt=7.5, integrator_safe_mode=0)  # Non-default  # Non-default  # Non-default

        # These should NOT be default values if config is working
        assert sim.config.integrator != 'IAS15'  # Default would be IAS15
        assert sim.config.dt != 1.0  # Default would be 1.0
        assert sim.config.integrator_safe_mode != 1  # Default would be 1

        # Integration engine should use the custom values
        mock_sim = Mock()
        mock_sim.ri_saba = Mock()
        sim.integration_engine.sim = mock_sim

        sim.integration_engine.setup_integrator()

        # If config wasn't working, these would be default values
        assert mock_sim.integrator == 'SABA(8,6,4)'  # Not default
        assert mock_sim.dt == 7.5  # Not default
        assert mock_sim.ri_saba.safe_mode == 0  # Not default


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
