#!/usr/bin/env python3
"""
Tests for DataManager Component
==============================

This module tests the DataManager class.
"""

import fcntl
import multiprocessing
import re
import threading
import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from resonances.simulation import SimulationConfig, DataManager
from resonances.resonance.analysis.verdict import Status
from resonances.body import Body
import resonances
from tests.tools import TEST_CACHE


def _append_csv_chunks(filename, worker, chunks=10, rows=25):
    """Spawn-safe helper that repeatedly appends uniquely identified rows."""
    for chunk in range(chunks):
        start = chunk * rows
        frame = pd.DataFrame(
            {
                "worker": [worker] * rows,
                "row": range(start, start + rows),
            }
        )
        DataManager._append_csv(frame, filename)


def _try_lock(path):
    """Whether a new open file description can take the lock of `path` at once (it is free)."""
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return True


class TestDataManager:
    """Test the DataManager component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = SimulationConfig()
        self.data_manager = DataManager(self.config)

    def test_process_status_all(self):
        """Test process status with 'all' mode."""
        # Test various statuses with 'all' mode
        assert self.data_manager._process_status('resonant', 'all') is True
        assert self.data_manager._process_status('transient', 'all') is True
        assert self.data_manager._process_status('non_resonant', 'all') is True
        assert self.data_manager._process_status('invalid', 'all') is True

    @pytest.mark.slow
    def test_append_csv_is_atomic_across_two_processes(self, tmp_path):
        """Two batch workers produce one header and every row exactly once."""
        filename = str(tmp_path / "summary.csv")
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=_append_csv_chunks, args=(filename, worker)) for worker in range(2)]

        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0

        result = pd.read_csv(filename)
        expected = pd.MultiIndex.from_product([range(2), range(250)], names=["worker", "row"])
        actual = pd.MultiIndex.from_frame(result[["worker", "row"]])
        assert len(result) == 500
        assert actual.is_unique
        assert set(actual) == set(expected)
        assert not list(tmp_path.glob("*.lock")), "the holder removes the lock file"

    @pytest.mark.slow
    def test_append_csv_with_lock_files_removed_stays_atomic(self, tmp_path):
        """Four workers race on a lock file each holder removes: still one header and every row once, no lock left."""
        filename = str(tmp_path / "episodes.csv")
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=_append_csv_chunks, args=(filename, worker, 20, 5)) for worker in range(4)]

        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
            assert process.exitcode == 0

        result = pd.read_csv(filename)
        assert list(result.columns) == ["worker", "row"]
        assert len(result) == 4 * 20 * 5
        assert pd.MultiIndex.from_frame(result[["worker", "row"]]).is_unique
        assert sorted(tmp_path.iterdir()) == [tmp_path / "episodes.csv"]

    def test_a_waiter_on_a_removed_lock_file_locks_the_path_again(self, tmp_path, monkeypatch):
        """A writer blocked on the lock file its holder then removes re-locks the path before writing.

        Without that it would write holding a lock on a deleted file, beside a third writer locking a new one."""
        filename = str(tmp_path / "summary.csv")
        lock_path = Path(f"{filename}.lock")
        seen = {}
        flock, waiting = fcntl.flock, threading.Event()

        def flock_of_the_waiter(fd, operation):
            if threading.current_thread().name == "waiter":
                waiting.set()  # it has opened the holder's lock file and now asks for its lock
            return flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", flock_of_the_waiter)

        def waiter():
            with DataManager._locked(filename):
                seen["lock_file_at_path"] = lock_path.exists()
                seen["third_writer_blocked"] = not _try_lock(lock_path)

        with DataManager._locked(filename):
            thread = threading.Thread(target=waiter, name="waiter")
            thread.start()
            assert waiting.wait(timeout=10)
            time.sleep(0.05)  # let it block in flock on the old file
            assert "lock_file_at_path" not in seen
        thread.join(timeout=10)

        assert seen == {"lock_file_at_path": True, "third_writer_blocked": True}
        assert not lock_path.exists()

    def test_a_stale_lock_file_is_reused_and_removed(self, tmp_path):
        """A .lock left by a run before this change (or a killed worker) does not block the next write."""
        filename = tmp_path / "constants.csv"
        (tmp_path / "constants.csv.lock").write_text("")
        DataManager._append_csv(pd.DataFrame({"planet": ["Jupiter"]}), str(filename))

        assert pd.read_csv(filename)["planet"].tolist() == ["Jupiter"]
        assert not (tmp_path / "constants.csv.lock").exists()

    def test_append_csv_narrow_batch_first_keeps_wide_rows(self, tmp_path):
        """segments.csv: a batch without segment metrics may finish first (seen on a real 10 kyr run)."""
        filename = str(tmp_path / "segments.csv")
        DataManager._append_csv(pd.DataFrame({"body": ["9068"], "resonance": ["g-g6"]}), filename)
        wide = pd.DataFrame({"body": ["1741"], "resonance": ["g-g6"], "0.1_0.00-0.10_R": [0.25], "0.1_0.00-0.10_amp": [1.5]})
        DataManager._append_csv(wide, filename)
        DataManager._append_csv(pd.DataFrame({"body": ["4765"], "resonance": ["s-s6"]}), filename)

        result = pd.read_csv(filename, dtype={"body": str})
        assert list(result.columns) == ["body", "resonance", "0.1_0.00-0.10_R", "0.1_0.00-0.10_amp"]
        assert list(result["body"]) == ["9068", "1741", "4765"]
        assert result.loc[1, "0.1_0.00-0.10_R"] == 0.25
        assert result.loc[1, "0.1_0.00-0.10_amp"] == 1.5
        assert result.loc[[0, 2], "0.1_0.00-0.10_R"].isna().all()

    def test_append_csv_wide_batch_first_pads_narrow_rows(self, tmp_path):
        """Rows missing some columns are aligned to the header, not shifted."""
        filename = str(tmp_path / "segments.csv")
        DataManager._append_csv(pd.DataFrame({"body": ["a"], "resonance": ["r1"], "x": [1.0], "y": [2.0]}), filename)
        DataManager._append_csv(pd.DataFrame({"body": ["b"], "y": [5.0], "resonance": ["r2"]}), filename)

        result = pd.read_csv(filename)
        assert list(result.columns) == ["body", "resonance", "x", "y"]
        assert result.loc[1, "resonance"] == "r2"
        assert np.isnan(result.loc[1, "x"])
        assert result.loc[1, "y"] == 5.0

    def test_process_status_resonant(self):
        """Test process status with 'resonant' mode."""
        assert self.data_manager._process_status('resonant', 'resonant') is True
        assert self.data_manager._process_status('transient', 'resonant') is True
        assert self.data_manager._process_status('non_resonant', 'resonant') is False
        assert self.data_manager._process_status('near_separatrix', 'resonant') is False

    # Every status the package defines, against the author's sets (SPEC §11.5): candidates keep
    # uncertain ("re-check on a longer baseline"), extended adds non_resonant rows with a libration origin.
    MODE_TABLE = {
        #             resonant transient near_sep non_res uncertain invalid  non_res+kinematic
        'candidates': (True, True, True, False, True, False, False),
        'extended': (True, True, True, False, True, False, True),
    }
    STATUSES = (
        ('resonant', ''),
        ('transient', ''),
        ('near_separatrix', ''),
        ('non_resonant', ''),
        ('uncertain', ''),
        ('invalid', ''),
        ('non_resonant', 'kinematic'),
    )

    @pytest.mark.parametrize('mode', ['candidates', 'extended'])
    def test_process_status_table(self, mode):
        for (status, origin), expected in zip(self.STATUSES, self.MODE_TABLE[mode]):
            assert self.data_manager._process_status(status, mode, origin) is expected, (mode, status, origin)

    def test_the_table_covers_every_status(self):
        assert sorted({status for status, _ in self.STATUSES}) == sorted(str(s) for s in Status)

    def test_extended_is_a_superset_of_candidates(self):
        for status, origin in self.STATUSES:
            if self.data_manager._process_status(status, 'candidates', origin):
                assert self.data_manager._process_status(status, 'extended', origin), status

    def test_docs_list_the_same_sets(self):
        # docs/config.md states the sets by status name; they must not drift from the code.
        text = (Path(__file__).parents[3] / 'docs' / 'config.md').read_text()
        listed = re.search(r'^\| `candidates` \| (.*) \|$', text, re.MULTILINE).group(1)
        named = set(re.findall(r'`([a-z_]+)`', listed))
        assert named == {status for (status, origin), keep in zip(self.STATUSES, self.MODE_TABLE['candidates']) if keep}

    def test_process_status_negative(self):
        """Test process status with 'negative' mode (near_separatrix, uncertain, invalid)."""
        assert self.data_manager._process_status('resonant', 'negative') is False
        assert self.data_manager._process_status('transient', 'negative') is False
        assert self.data_manager._process_status('non_resonant', 'negative') is False
        assert self.data_manager._process_status('near_separatrix', 'negative') is True
        assert self.data_manager._process_status('uncertain', 'negative') is True
        assert self.data_manager._process_status('invalid', 'negative') is True

    def test_process_status_nonzero(self):
        """Test process status with 'nonzero' mode (everything but non_resonant without a libration origin)."""
        assert self.data_manager._process_status('resonant', 'nonzero') is True
        assert self.data_manager._process_status('transient', 'nonzero') is True
        assert self.data_manager._process_status('non_resonant', 'nonzero') is False
        assert self.data_manager._process_status('non_resonant', 'nonzero', 'induced') is True
        assert self.data_manager._process_status('uncertain', 'nonzero') is True

    def test_process_status_none(self):
        """Test process status with None mode."""
        assert self.data_manager._process_status('resonant', None) is False
        assert self.data_manager._process_status('non_resonant', None) is False

    @patch('pathlib.Path.mkdir')
    def test_ensure_save_path_exists(self, mock_mkdir):
        """Test ensuring save paths exist."""
        self.data_manager.ensure_save_path_exists()

        # Should be called twice - once for save_path, once for plot_path (the data folder when a file is written)
        assert mock_mkdir.call_count == 2

    def test_should_save_body(self):
        """Test should save body decision."""
        # Create mock body and resonance
        mock_body = Mock()
        mock_body.statuses = {'test_resonance': 'resonant'}
        mock_body.librations = {}

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
        mock_body.statuses = {'test_resonance': 'resonant'}
        mock_body.librations = {}

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
        tmpdir = f"{TEST_CACHE}/{sim_name}"
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
        body.statuses[mmr.to_s()] = 'resonant'
        body.periodogram_frequency[mmr.to_s()] = np.array([1.0, 2.0, 3.0])
        body.periodogram_power[mmr.to_s()] = np.array([10.0, 20.0, 30.0])

        # Add Secular resonance
        secular = resonances.create_resonance('g-2g6+g5')
        body.secular_resonances = [secular]
        body.angles[secular.to_s()] = np.array([1.1, 1.2, 1.3])
        body.angles_unwrapped[secular.to_s()] = np.array([1.1, 1.2, 1.3])
        body.secular_angles_osculating[secular.to_s()] = np.array([1.11, 1.21, 1.31])
        body.secular_angles_proper[secular.to_s()] = np.array([1.12, 1.22, 1.32])
        body.statuses[secular.to_s()] = 'transient'

        lk = resonances.create_resonance('lkr')
        body.lidov_kozai_resonances = [lk]
        body.angles[lk.to_s()] = np.array([2.1, 2.2, 2.3])
        body.angles_unwrapped[lk.to_s()] = np.array([2.1, 2.2, 2.3])
        body.statuses[lk.to_s()] = 'transient'

        # Create times array
        times = np.array([0, 100, 200]) * (2 * np.pi)

        # Save body data
        data_manager.save_body(body, times)

        # Verify single consolidated data file exists
        data_file = Path(tmpdir) / 'data' / f'data-{body.name}.csv'  # the series go to the data folder
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
        periodogram_file = Path(tmpdir) / 'data' / f'data-{body.name}-periodograms.csv'
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
        old_style_files = list(Path(tmpdir).rglob(f'data-{body.name}-{mmr.to_s()}.csv'))
        assert len(old_style_files) == 0, "Old-style separate resonance files should not be created"


class TestPlotSubfolderStrategy:
    """The old PLOT_SUBFOLDER_STRATEGY ('' or 'status') is read as the layout of both figure folders."""

    def test_default_is_confidence_status(self):
        config = SimulationConfig()  # the benchmark's tree (author 2026-10-27, W24 Q8)
        assert config.plot_subfolder_strategy is None
        assert (config.plot_layout, config.plot_combined_layout) == ('confidence/status', 'confidence/status')
        flat = SimulationConfig(plot_layout='flat')
        assert (flat.plot_subfolder_strategy, flat.plot_layout, flat.plot_combined_layout) == (None, 'flat', 'flat')

    def test_status_is_the_status_layout(self):
        config = SimulationConfig(plot_subfolder_strategy='status')
        assert config.plot_subfolder_strategy == 'status'
        assert (config.plot_layout, config.plot_combined_layout) == ('status', 'status')

    def test_empty_string_is_flat(self):
        config = SimulationConfig(plot_subfolder_strategy='')
        assert config.plot_subfolder_strategy is None
        assert config.plot_layout == 'flat'

    def test_invalid_raises_error(self):
        with pytest.raises(ValueError, match="Invalid plot_subfolder_strategy"):
            SimulationConfig(plot_subfolder_strategy='invalid')

    def test_setting_the_old_attribute_sets_both_layouts(self):
        config = SimulationConfig(plot_layout='confidence/status')
        config.plot_subfolder_strategy = 'status'
        assert (config.plot_layout, config.plot_combined_layout) == ('status', 'status')

    @pytest.mark.parametrize('status', [str(s) for s in Status] + [None])
    def test_status_folder_of_every_status(self, status, tmp_path):
        """A body's figure folder under the old 'status' strategy: plot_path/figures/<status>; no status: unclassified."""
        config = SimulationConfig(plot_subfolder_strategy='status', plot_path=str(tmp_path), save_path=str(tmp_path))
        body = Body()
        body.name = 'b'
        mmr = resonances.create_mmr('4J-2S-1')
        if status is not None:
            body.statuses[mmr.to_s()] = status
        folder = status if status is not None else 'unclassified'
        assert DataManager(config).figure_folder(body, mmr) == f'{tmp_path}/figures/{folder}'
        assert DataManager(config).figure_folder(body, mmr, combined=True) == f'{tmp_path}/combined/{folder}'


if __name__ == '__main__':
    pytest.main([__file__])


def test_plot_always_overrides_status_for_listed_types():
    """plot_always draws the listed resonance types whatever their status; the plot mode still rules the rest.

    With plot='resonant' (resonant, transient) a Lidov-Kozai angle at status uncertain is not drawn unless
    'lidov_kozai' is in plot_always; an MMR at uncertain stays undrawn either way; plot=None draws nothing.
    """
    import resonances
    from resonances.simulation.config import SimulationConfig
    from resonances.simulation.data_manager import DataManager
    from resonances.body import Body

    lk = resonances.create_resonance('LK')
    mmr = resonances.create_mmr('1E-1')
    body = Body()
    body.statuses = {lk.to_s(): 'uncertain', mmr.to_s(): 'uncertain'}

    plain = DataManager(SimulationConfig(plot='resonant', save=None, save_path=f'{TEST_CACHE}/pa1', plot_path=f'{TEST_CACHE}/pa1'))
    assert plain.config.plot_always == []
    assert plain.should_plot_body(body, lk) is False
    assert plain.should_plot_body(body, mmr) is False

    always = DataManager(
        SimulationConfig(
            plot='resonant', plot_always=['lidov_kozai'], save=None, save_path=f'{TEST_CACHE}/pa2', plot_path=f'{TEST_CACHE}/pa2'
        )
    )
    assert always.config.plot_always == ['lidov_kozai']
    assert always.should_plot_body(body, lk) is True
    assert always.should_plot_body(body, mmr) is False

    as_string = SimulationConfig(
        plot='resonant', plot_always='lidov_kozai, mmr', save=None, save_path=f'{TEST_CACHE}/pa3', plot_path=f'{TEST_CACHE}/pa3'
    )
    assert as_string.plot_always == ['lidov_kozai', 'mmr']

    off = DataManager(
        SimulationConfig(plot=None, plot_always=['lidov_kozai'], save=None, save_path=f'{TEST_CACHE}/pa4', plot_path=f'{TEST_CACHE}/pa4')
    )
    assert off.should_plot_body(body, lk) is False
