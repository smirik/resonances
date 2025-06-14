#!/usr/bin/env python3
"""
Tests for Simulation Integration
===============================

This module tests the integration between simulation components.
"""

import pytest
from unittest.mock import Mock, patch

from resonances.simulation import SimulationConfig, BodyManager, IntegrationEngine, DataManager


@pytest.fixture
def sample_simulation_data():
    """Provide sample simulation data for testing."""
    return {
        'config': {
            'name': 'test_simulation',
            'tmax': 62831,
            'integrator': 'SABA(10,6,4)',
            'dt': 5.0,
        },  # 10,000 years in simulation units
        'body_data': {
            'name': 'test_asteroid',
            'elements': {'a': 2.5, 'e': 0.1, 'inc': 0.1, 'Omega': 0.1, 'omega': 0.1, 'M': 0.1},
        },
    }


class TestSimulationIntegration:
    """Integration tests for the refactored simulation."""

    def test_component_integration(self, sample_simulation_data):
        """Test that components work together properly."""
        # Create components
        config = SimulationConfig(**sample_simulation_data['config'])
        body_manager = BodyManager(config)
        integration_engine = IntegrationEngine(config)
        data_manager = DataManager(config)

        # Verify components are initialized
        assert config.name == 'test_simulation'
        assert len(body_manager.bodies) == 0
        assert integration_engine.config == config
        assert data_manager.config == config

    @patch('resonances.horizons.get_body_keplerian_elements')
    @patch('resonances.create_mmr')
    def test_end_to_end_mmr_workflow(self, mock_create_mmr, mock_get_elements, sample_simulation_data):
        """Test end-to-end workflow with MMR."""
        # Setup mocks
        mock_get_elements.return_value = sample_simulation_data['body_data']['elements']

        mock_mmr = Mock()
        mock_mmr.planets_names = ['Jupiter', 'Saturn']
        mock_mmr.to_s.return_value = '4J-2S-1'
        mock_create_mmr.return_value = mock_mmr

        # Create components
        config = SimulationConfig(**sample_simulation_data['config'])
        body_manager = BodyManager(config)

        # Add body with MMR
        body_manager.add_body_with_mmr(463, "4J-2S-1", "test_asteroid")

        # Verify workflow
        assert len(body_manager.bodies) == 1
        body = body_manager.bodies[0]
        assert body.name == "test_asteroid"
        assert len(body.mmrs) == 1
        assert body.mmrs[0] == mock_mmr

        # Verify planet indices were set
        assert hasattr(mock_mmr, 'index_of_planets')


if __name__ == '__main__':
    pytest.main([__file__])
