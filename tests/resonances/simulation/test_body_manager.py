#!/usr/bin/env python3
"""
Tests for BodyManager Component
==============================

This module tests the BodyManager class.
"""

import pytest
from unittest.mock import Mock, patch

from resonances.simulation import BodyManager, SimulationConfig


class TestBodyManager:
    """Test the BodyManager component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SimulationConfig()
        self.body_manager = BodyManager(self.config)

    def test_planet_indices(self):
        """Test planet index lookup."""
        indices = self.body_manager.get_index_of_planets(['Jupiter', 'Saturn'])
        assert indices == [5, 6]  # Jupiter and Saturn indices

    def test_planet_list(self):
        """Test planet list."""
        planets = self.body_manager.planets
        assert 'Sun' in planets
        assert 'Jupiter' in planets
        assert 'Saturn' in planets
        assert len(planets) == 10

    @patch('resonances.horizons.get_body_keplerian_elements')
    def test_get_body_elements_nasa(self, mock_get_elements):
        """Test getting body elements from NASA."""
        mock_elements = {'a': 2.5, 'e': 0.1, 'inc': 0.1, 'Omega': 0.1, 'omega': 0.1, 'M': 0.1}
        mock_get_elements.return_value = mock_elements

        elements = self.body_manager.get_body_elements(463)

        assert elements == mock_elements
        mock_get_elements.assert_called_once()

    def test_get_body_elements_dict(self):
        """Test getting body elements from dictionary."""
        test_elements = {'a': 2.5, 'e': 0.1, 'inc': 0.1, 'Omega': 0.1, 'omega': 0.1, 'M': 0.1}

        elements = self.body_manager.get_body_elements(test_elements)
        assert elements == test_elements

    def test_get_body_elements_invalid(self):
        """Test error handling for invalid element input."""
        with pytest.raises(ValueError):
            self.body_manager.get_body_elements([1, 2, 3])  # Invalid type

    @patch('resonances.create_mmr')
    @patch.object(BodyManager, 'get_body_elements')
    def test_add_body_with_mmr(self, mock_get_elements, mock_create_mmr):
        """Test adding a body with MMR."""
        # Setup mocks
        mock_elements = {'a': 2.5, 'e': 0.1, 'inc': 0.1, 'Omega': 0.1, 'omega': 0.1, 'M': 0.1}
        mock_get_elements.return_value = mock_elements

        mock_mmr = Mock()
        mock_mmr.planets_names = ['Jupiter', 'Saturn']
        mock_create_mmr.return_value = mock_mmr

        # Test adding body
        self.body_manager.add_body_with_mmr(463, "4J-2S-1", "test_asteroid")

        # Verify body was added
        assert len(self.body_manager.bodies) == 1
        body = self.body_manager.bodies[0]
        assert body.name == "test_asteroid"
        assert body.initial_data == mock_elements
        assert len(body.mmrs) == 1
        assert body.mmrs[0] == mock_mmr

    @patch('resonances.create_secular_resonance')
    @patch.object(BodyManager, 'get_body_elements')
    def test_add_body_with_secular(self, mock_get_elements, mock_create_secular):
        """Test adding a body with secular resonance."""
        # Setup mocks
        mock_elements = {'a': 2.5, 'e': 0.1, 'inc': 0.1, 'Omega': 0.1, 'omega': 0.1, 'M': 0.1}
        mock_get_elements.return_value = mock_elements

        mock_secular = Mock()
        mock_secular.planets_names = ['Saturn']
        mock_create_secular.return_value = mock_secular

        # Test adding body
        self.body_manager.add_body_with_secular(463, "nu6", "test_asteroid")

        # Verify body was added
        assert len(self.body_manager.bodies) == 1
        body = self.body_manager.bodies[0]
        assert body.name == "test_asteroid"
        assert body.initial_data == mock_elements
        assert len(body.secular_resonances) == 1
        assert body.secular_resonances[0] == mock_secular


if __name__ == '__main__':
    pytest.main([__file__])
