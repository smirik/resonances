#!/usr/bin/env python3
"""
Tests for DataManager Component
==============================

This module tests the DataManager class.
"""

import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from resonances.simulation import SimulationConfig, DataManager
from resonances.body import Body
import resonances


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

    def test_save_body_consolidated_format(self):
        sim_name = f"random-{time.time()}"
        tmpdir = f"cache/tests/{sim_name}"
        """Test that save_body creates consolidated files with proper structure."""
        # Setup config with temp directory
        config = SimulationConfig(save_path=tmpdir, save='all', plot=None)  # Disable plotting for this test
        data_manager = DataManager(config)

        # Create a body with multiple resonances
        body = Body()
        body.name = 'test_asteroid_463'

        # Add Keplerian elements
        body.axis = np.array([2.5, 2.6, 2.7])
        body.ecc = np.array([0.1, 0.2, 0.3])
        body.inc = np.array([5.0, 6.0, 7.0])
        body.Omega = np.array([100.0, 110.0, 120.0])
        body.omega = np.array([200.0, 210.0, 220.0])
        body.M = np.array([300.0, 310.0, 320.0])
        body.longitude = np.array([400.0, 410.0, 420.0])
        body.varpi = np.array([500.0, 510.0, 520.0])

        # Add MMR
        mmr = resonances.create_mmr('4J-2S-1')
        body.mmrs = [mmr]
        body.angles_unwrapped[mmr.to_s()] = np.array([0.1, 0.2, 0.3])
        body.angles[mmr.to_s()] = np.array([0.1, 0.2, 0.3])
        body.statuses[mmr.to_s()] = 2
        body.periodogram_frequency[mmr.to_s()] = np.array([1.0, 2.0, 3.0])
        body.periodogram_power[mmr.to_s()] = np.array([10.0, 20.0, 30.0])

        # Add Secular resonance
        secular = resonances.create_resonance('g-2g6+g5')
        body.secular_resonances = [secular]
        body.angles[secular.to_s()] = np.array([1.1, 1.2, 1.3])
        body.angles_unwrapped[secular.to_s()] = np.array([1.1, 1.2, 1.3])
        body.secular_angles_osculating[secular.to_s()] = np.array([1.11, 1.21, 1.31])
        body.secular_angles_proper[secular.to_s()] = np.array([1.12, 1.22, 1.32])
        body.statuses[secular.to_s()] = 1

        lk = resonances.create_resonance('lkr')
        body.lidov_kozai_resonances = [lk]
        body.angles[lk.to_s()] = np.array([2.1, 2.2, 2.3])
        body.angles_unwrapped[lk.to_s()] = np.array([2.1, 2.2, 2.3])
        body.statuses[lk.to_s()] = 1

        # Create times array
        times = np.array([0, 100, 200]) * (2 * np.pi)

        # Save body data
        data_manager.save_body(body, times)

        # Verify single consolidated data file exists
        data_file = Path(tmpdir) / f'data-{body.name}.csv'
        assert data_file.exists(), "Consolidated data file should exist"

        # Load and verify data file structure
        df = pd.read_csv(data_file)

        # Check Keplerian elements are present
        assert 'a' in df.columns
        assert 'e' in df.columns
        assert 'inc' in df.columns
        assert 'times' in df.columns

        # Check MMR angle data with prefixed column name
        mmr_angle_col = mmr.to_s() + '_angle'
        assert mmr_angle_col in df.columns, f"MMR angle column '{mmr_angle_col}' should be present"
        assert len(df[mmr_angle_col]) == 3

        # Check Secular angle data with prefixed column names
        sec_angle_col = secular.to_s() + '_angle'
        sec_osc_col = secular.to_s() + '_angle_osculating'
        sec_prop_col = secular.to_s() + '_angle_proper'

        assert sec_angle_col in df.columns, f"Secular angle column '{sec_angle_col}' should be present"
        assert sec_osc_col in df.columns
        assert sec_prop_col in df.columns

        lkr_angle_col = lk.to_s() + '_angle'
        assert lkr_angle_col in df.columns, f"Lidov-Kozai angle column '{lkr_angle_col}' should be present"
        assert len(df[lkr_angle_col]) == 3

        # Verify periodogram file
        periodogram_file = Path(tmpdir) / f'data-{body.name}-periodograms.csv'
        assert periodogram_file.exists(), "Consolidated periodogram file should exist"

        # Load and verify periodogram structure
        df_periodogram = pd.read_csv(periodogram_file)

        # Check semi-major axis periodogram is NOT present (since we didn't set it)
        # But check MMR periodogram is present with prefixed column names
        mmr_freq_col = mmr.to_s() + '_frequency'
        mmr_power_col = mmr.to_s() + '_power'
        mmr_period_col = mmr.to_s() + '_period'

        assert mmr_freq_col in df_periodogram.columns
        assert mmr_power_col in df_periodogram.columns
        assert mmr_period_col in df_periodogram.columns

        # Verify no old-style separate files were created
        old_style_files = list(Path(tmpdir).glob(f'data-{body.name}-{mmr.to_s()}.csv'))
        assert len(old_style_files) == 0, "Old-style separate resonance files should not be created"


if __name__ == '__main__':
    pytest.main([__file__])
