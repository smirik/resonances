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
        assert self.data_manager._process_status(2, 'all') == True
        assert self.data_manager._process_status(1, 'all') == True
        assert self.data_manager._process_status(0, 'all') == True
        assert self.data_manager._process_status(-1, 'all') == True

    def test_process_status_resonant(self):
        """Test process status with 'resonant' mode."""
        assert self.data_manager._process_status(2, 'resonant') == True
        assert self.data_manager._process_status(1, 'resonant') == True
        assert self.data_manager._process_status(0, 'resonant') == False
        assert self.data_manager._process_status(-1, 'resonant') == False

    def test_process_status_candidates(self):
        """Test process status with 'candidates' mode."""
        assert self.data_manager._process_status(2, 'candidates') == False
        assert self.data_manager._process_status(1, 'candidates') == False
        assert self.data_manager._process_status(0, 'candidates') == False
        assert self.data_manager._process_status(-1, 'candidates') == True
        assert self.data_manager._process_status(-2, 'candidates') == True

    def test_process_status_none(self):
        """Test process status with None mode."""
        assert self.data_manager._process_status(2, None) == False
        assert self.data_manager._process_status(0, None) == False

    @patch('pathlib.Path.mkdir')
    def test_ensure_save_path_exists(self, mock_mkdir):
        """Test ensuring save paths exist."""
        self.data_manager.ensure_save_path_exists()

        # Should be called twice - once for save_path, once for plot_path
        assert mock_mkdir.call_count == 2

    def test_should_save_body_mmr(self):
        """Test should save body MMR decision."""
        # Create mock body and MMR
        mock_body = Mock()
        mock_body.statuses = {'test_mmr': 2}

        mock_mmr = Mock()
        mock_mmr.to_s.return_value = 'test_mmr'

        # Test with different save modes
        self.config.save = 'all'
        assert self.data_manager.should_save_body_mmr(mock_body, mock_mmr) == True

        self.config.save = 'resonant'
        assert self.data_manager.should_save_body_mmr(mock_body, mock_mmr) == True

        self.config.save = None
        assert self.data_manager.should_save_body_mmr(mock_body, mock_mmr) == False


if __name__ == '__main__':
    pytest.main([__file__])
