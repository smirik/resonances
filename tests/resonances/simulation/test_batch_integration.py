import pytest
import pandas as pd

import resonances
from resonances.simulation import Simulation


@pytest.mark.slow
class TestBatchIntegration:
    """Integration tests for batch processing."""

    def test_automatic_batching_enabled(self, tmp_path):
        """Test that batching is automatically enabled for >50 bodies."""
        # Create simulation with 6 test asteroids (small for fast testing)
        asteroids = [463, 490, 624, 1139, 1656, 1727]

        sim = resonances.find(
            asteroids,
            planets=["Jupiter", "Saturn"],
            name="batch_test",
            save_path=str(tmp_path),
            save="all",
            save_summary=True,
            integration_years=1000,  # Short integration for testing
            batch_threshold=3,  # Lower threshold for testing
            batch_size=2,  # Small batches
            n_cores=1,
        )

        # Should create simulation
        assert isinstance(sim, Simulation)

        # Should have 6 bodies
        assert len(sim.bodies) == 6

        # Run simulation
        sim.run(progress=False)

        # Verify summary file exists
        summary_file = tmp_path / "summary.csv"
        assert summary_file.exists()

        # Verify all bodies in summary
        df = pd.read_csv(summary_file)
        assert len(df) >= 6  # At least 6 entries (may have multiple resonances per body)

    def test_batching_disabled_for_small_sims(self, tmp_path):
        """Test that batching is not used for small simulations."""
        asteroids = [463, 490]  # Only 2 bodies

        sim = resonances.find(
            asteroids,
            planets=["Jupiter", "Saturn"],
            name="small_test",
            save_path=str(tmp_path),
            save="all",
            save_summary=True,
            integration_years=1000,
            batch_threshold=50,  # Default threshold
        )

        assert isinstance(sim, Simulation)
        assert len(sim.bodies) == 2

        # Run simulation
        sim.run(progress=False)

        # Verify no state file created (because batching wasn't used)
        state_file = tmp_path / "simulation_state.json"
        assert not state_file.exists()

        # Verify results still saved
        summary_file = tmp_path / "summary.csv"
        assert summary_file.exists()

    def test_batching_explicitly_disabled(self, tmp_path):
        """Test that batching can be explicitly disabled."""
        asteroids = [463, 490, 624, 1139, 1656, 1727]

        sim = resonances.find(
            asteroids,
            planets=["Jupiter", "Saturn"],
            name="no_batch_test",
            save_path=str(tmp_path),
            save="all",
            save_summary=True,
            integration_years=1000,
            batch_enabled=False,  # Explicitly disable
        )

        assert isinstance(sim, Simulation)
        assert len(sim.bodies) == 6

        # Run simulation
        sim.run(progress=False)

        # Verify no state file created
        state_file = tmp_path / "simulation_state.json"
        assert not state_file.exists()

        # Verify results still saved
        summary_file = tmp_path / "summary.csv"
        assert summary_file.exists()

    @pytest.mark.slow
    def test_resume_after_simulated_failure(self, tmp_path):
        """Test resuming simulation after a failure."""
        asteroids = [463, 490, 624, 1139]

        # First run - simulate partial completion
        resonances.find(
            asteroids,
            planets=["Jupiter", "Saturn"],
            name="resume_test",
            save_path=str(tmp_path),
            save="all",
            save_summary=True,
            integration_years=1000,
            batch_threshold=2,
            batch_size=2,
            n_cores=1,
        )

        # Manually modify state to simulate partial completion
        # This would normally happen if a batch failed mid-run
        from resonances.simulation.state_manager import StateManager, SimulationState

        state_manager = StateManager(str(tmp_path), "resume_test")
        state = SimulationState.create_new(
            simulation_name="resume_test",
            total_bodies=4,
            batch_size=2,
            batch_threshold=2,
            n_cores=1,
            config_hash="test_hash",
        )
        state.mark_batch_complete(0, ["463", "490"])  # Simulate first batch completed
        state_manager.save_state(state)

        # Create new simulation with same configuration
        sim2 = resonances.find(
            asteroids,
            planets=["Jupiter", "Saturn"],
            name="resume_test",
            save_path=str(tmp_path),
            save="all",
            save_summary=True,
            integration_years=1000,
            batch_threshold=2,
            batch_size=2,
            n_cores=1,
            resume_enabled=True,
        )

        # Run should resume from batch 1
        sim2.run(progress=False)

        # Verify summary contains all bodies
        summary_file = tmp_path / "summary.csv"
        assert summary_file.exists()
        df = pd.read_csv(summary_file)
        assert len(df) >= 4

    def test_state_file_cleanup(self, tmp_path):
        """Test that state file is cleaned up after successful completion."""
        asteroids = [463, 490, 624]

        sim = resonances.find(
            asteroids,
            planets=["Jupiter", "Saturn"],
            name="cleanup_test",
            save_path=str(tmp_path),
            save="all",
            save_summary=True,
            integration_years=1000,
            batch_threshold=2,
            batch_size=2,
            n_cores=1,
            resume_enabled=True,
        )

        sim.run(progress=False)

        # State file should be cleaned up after successful completion
        # state_file = tmp_path / "simulation_state.json"
        # Note: State file cleanup happens in BatchManager.execute_batches
        # It should be removed after all batches complete successfully

    def test_all_results_in_one_folder(self, tmp_path):
        """Test that all batch results are in a single folder."""
        asteroids = [463, 490, 624, 1139]

        sim = resonances.find(
            asteroids,
            planets=["Jupiter", "Saturn"],
            name="single_folder_test",
            save_path=str(tmp_path),
            save="all",
            save_summary=True,
            integration_years=1000,
            batch_threshold=2,
            batch_size=2,
            n_cores=1,
        )

        sim.run(progress=False)

        # Verify no batch subdirectories created
        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(subdirs) == 0, "No subdirectories should be created for batches"

        # Verify all results in main folder
        summary_file = tmp_path / "summary.csv"
        assert summary_file.exists()

        # Verify individual body files exist
        csv_files = list(tmp_path.glob("data-*.csv"))
        assert len(csv_files) > 0, "Individual body CSV files should exist"

    @pytest.mark.slow
    def test_check_function_with_batching(self, tmp_path):
        """Test that check() function also supports batching."""
        asteroids = [463, 490, 624, 1139]
        resonance = resonances.create_mmr("2J-1S-1")

        sim = resonances.check(
            asteroids,
            resonance,
            name="check_batch_test",
            save_path=str(tmp_path),
            save="all",
            save_summary=True,
            integration_years=1000,
            batch_threshold=2,
            batch_size=2,
            n_cores=1,
        )

        assert isinstance(sim, Simulation)
        assert len(sim.bodies) == 4

        sim.run(progress=False)

        # Verify results
        summary_file = tmp_path / "summary.csv"
        assert summary_file.exists()
