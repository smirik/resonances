#!/usr/bin/env python3
"""
Tests for IntegrationEngine Component
====================================

This module tests the IntegrationEngine class.
"""

import pytest
from unittest.mock import Mock, patch

from resonances.simulation import SimulationConfig, IntegrationEngine


class TestIntegrationEngine:
    """Test the IntegrationEngine component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SimulationConfig()
        self.engine = IntegrationEngine(self.config)

    def test_solar_system_filename(self):
        """Test solar system filename generation."""
        filename = self.engine._solar_system_filename()
        assert filename.endswith('.bin')
        assert 'solar' in filename.lower()  # More flexible check for 'solar' in filename

    @patch('rebound.Simulation')
    def test_create_solar_system_new(self, mock_simulation):
        """Test creating new solar system."""
        # Mock the simulation
        mock_sim = Mock()
        mock_simulation.return_value = mock_sim

        # Mock file not existing
        with patch('pathlib.Path.exists', return_value=False):
            self.engine.create_solar_system()

        # Verify simulation was created
        mock_simulation.assert_called()
        mock_sim.add.assert_called()
        mock_sim.save_to_file.assert_called()

    @patch('rebound.Simulation')
    def test_create_solar_system_load(self, mock_simulation):
        """Test loading existing solar system."""
        # Mock the simulation
        mock_sim = Mock()
        mock_simulation.return_value = mock_sim

        # Mock file existing
        with patch('pathlib.Path.exists', return_value=True):
            self.engine.create_solar_system()

        # Verify simulation was loaded from file
        mock_simulation.assert_called_once()

    def test_setup_integrator_saba_alternative(self):
        """Test integrator setup for SABA with different parameters."""
        # Create mock simulation
        mock_sim = Mock()
        mock_sim.ri_saba = Mock()
        self.engine.sim = mock_sim
        self.config.integrator = 'SABA(8,6,4)'
        self.config.dt = 0.05
        self.config.integrator_safe_mode = 1

        self.engine.setup_integrator()

        assert mock_sim.integrator == 'SABA(8,6,4)'
        assert mock_sim.dt == 0.05
        assert mock_sim.ri_saba.safe_mode == 1
        mock_sim.move_to_com.assert_called_once()

    def test_setup_integrator_saba(self):
        """Test integrator setup for SABA."""
        # Create mock simulation
        mock_sim = Mock()
        mock_sim.ri_saba = Mock()
        self.engine.sim = mock_sim
        self.config.integrator = 'SABA(10,6,4)'
        self.config.dt = 5.0

        self.engine.setup_integrator()

        assert mock_sim.integrator == 'SABA(10,6,4)'
        assert mock_sim.dt == 5.0
        mock_sim.move_to_com.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
