#!/usr/bin/env python3
"""
Tests for DataManager Component
==============================

This module tests the DataManager class.
"""

import pytest
from unittest.mock import Mock, patch

from resonances.simulation import SimulationConfig, DataManager


class TestDataManager:
    """Test the DataManager component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SimulationConfig()
        self.data_manager = DataManager(self.config)

    def test_process_status_all(self):
        """Test process status with 'all' mode."""
        # Test various statuses with 'all' mode
        assert self.data_manager._process_status(2, 'all') is True
        assert self.data_manager._process_status(1, 'all') is True
        assert self.data_manager._process_status(0, 'all') is True
        assert self.data_manager._process_status(-1, 'all') is True

    def test_process_status_resonant(self):
        """Test process status with 'resonant' mode."""
        assert self.data_manager._process_status(2, 'resonant') is True
        assert self.data_manager._process_status(1, 'resonant') is True
        assert self.data_manager._process_status(0, 'resonant') is False
        assert self.data_manager._process_status(-1, 'resonant') is False

    def test_process_status_candidates(self):
        """Test process status with 'candidates' mode."""
        assert self.data_manager._process_status(2, 'candidates') is False
        assert self.data_manager._process_status(1, 'candidates') is False
        assert self.data_manager._process_status(0, 'candidates') is False
        assert self.data_manager._process_status(-1, 'candidates') is True
        assert self.data_manager._process_status(-2, 'candidates') is True

    def test_process_status_nonzero(self):
        """Test process status with 'nonzero' mode."""
        assert self.data_manager._process_status(2, 'nonzero') is True
        assert self.data_manager._process_status(1, 'nonzero') is True
        assert self.data_manager._process_status(0, 'nonzero') is False
        assert self.data_manager._process_status(-1, 'nonzero') is True
        assert self.data_manager._process_status(-2, 'nonzero') is True

    def test_process_status_none(self):
        """Test process status with None mode."""
        assert self.data_manager._process_status(2, None) is False
        assert self.data_manager._process_status(0, None) is False

    @patch('pathlib.Path.mkdir')
    def test_ensure_save_path_exists(self, mock_mkdir):
        """Test ensuring save paths exist."""
        self.data_manager.ensure_save_path_exists()

        # Should be called twice - once for save_path, once for plot_path
        assert mock_mkdir.call_count == 2

    def test_should_save_body(self):
        """Test should save body decision."""
        # Create mock body and resonance
        mock_body = Mock()
        mock_body.statuses = {'test_resonance': 2}

        mock_resonance = Mock()
        mock_resonance.to_s.return_value = 'test_resonance'

        # Test with different save modes
        self.config.save = 'all'
        assert self.data_manager.should_save_body(mock_body, mock_resonance) is True

        self.config.save = 'resonant'
        assert self.data_manager.should_save_body(mock_body, mock_resonance) is True

        self.config.save = None
        assert self.data_manager.should_save_body(mock_body, mock_resonance) is False

    def test_should_plot_body(self):
        """Test should plot body decision."""
        # Create mock body and resonance
        mock_body = Mock()
        mock_body.statuses = {'test_resonance': 2}

        mock_resonance = Mock()
        mock_resonance.to_s.return_value = 'test_resonance'

        # Test with different plot modes
        self.config.plot = 'all'
        assert self.data_manager.should_plot_body(mock_body, mock_resonance) is True

        self.config.plot = 'resonant'
        assert self.data_manager.should_plot_body(mock_body, mock_resonance) is True

        self.config.plot = None
        assert self.data_manager.should_plot_body(mock_body, mock_resonance) is False


if __name__ == '__main__':
    pytest.main([__file__])
