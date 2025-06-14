#!/usr/bin/env python3
"""
Tests for SimulationConfig Component
===================================

This module tests the SimulationConfig class.
"""

import pytest
import numpy as np

from resonances.simulation import SimulationConfig


class TestSimulationConfig:
    """Test the SimulationConfig component."""

    def test_default_config(self):
        """Test default configuration."""
        config = SimulationConfig()

        # Check that basic attributes are set
        assert config.name is not None
        assert config.date is not None
        assert config.source is not None
        assert config.integrator is not None
        assert config.dt > 0
        assert config.tmax > 0

    def test_custom_config(self):
        """Test custom configuration."""
        custom_name = "test_simulation"
        custom_tmax = 20000
        custom_integrator = "SABA(10,6,4)"

        config = SimulationConfig(name=custom_name, tmax=custom_tmax, integrator=custom_integrator)

        assert config.name == custom_name
        assert config.tmax == custom_tmax
        assert config.integrator == custom_integrator

    def test_tmax_property(self):
        """Test tmax property calculation."""
        config = SimulationConfig()
        tmax_value = 31415  # Arbitrary value
        config.tmax = tmax_value

        assert abs(config.tmax - tmax_value) < 1e-10
        assert abs(config.tmax_yrs - tmax_value / (2 * np.pi)) < 1e-10

    def test_libration_params_setup(self):
        """Test libration parameters setup."""
        config = SimulationConfig()

        # Check that libration parameters are set
        assert hasattr(config, 'oscillations_cutoff')
        assert hasattr(config, 'oscillations_filter_order')
        assert hasattr(config, 'periodogram_frequency_min')
        assert hasattr(config, 'periodogram_frequency_max')
        assert hasattr(config, 'libration_period_critical')


if __name__ == '__main__':
    pytest.main([__file__])
