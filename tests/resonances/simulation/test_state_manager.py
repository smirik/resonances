import pytest

from resonances.simulation.state_manager import StateManager, SimulationState


class TestSimulationState:
    """Tests for SimulationState dataclass."""

    def test_create_new(self):
        """Test creating a new simulation state."""
        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=4,
            config_hash="abc123",
        )

        assert state.simulation_name == "test_sim"
        assert state.batch_config["total_bodies"] == 100
        assert state.batch_config["batch_size"] == 50
        assert state.batch_config["total_batches"] == 2
        assert state.batch_config["n_cores"] == 4
        assert state.progress["completed_batches"] == []
        assert state.progress["failed"] is False
        assert state.config_hash == "abc123"

    def test_mark_batch_complete(self):
        """Test marking a batch as completed."""
        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=1,
            config_hash="abc123",
        )

        state.mark_batch_complete(0, ["body1", "body2"])

        assert 0 in state.progress["completed_batches"]
        assert "body1" in state.progress["completed_bodies"]
        assert "body2" in state.progress["completed_bodies"]
        assert state.progress["current_batch"] == 1

    def test_mark_failed(self):
        """Test marking simulation as failed."""
        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=1,
            config_hash="abc123",
        )

        state.mark_failed("Test error")

        assert state.progress["failed"] is True
        assert state.progress["error_message"] == "Test error"

    def test_get_completed_bodies(self):
        """Test getting completed bodies as a set."""
        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=1,
            config_hash="abc123",
        )

        state.mark_batch_complete(0, ["body1", "body2"])
        state.mark_batch_complete(1, ["body3"])

        completed = state.get_completed_bodies()
        assert completed == {"body1", "body2", "body3"}

    def test_get_pending_batches(self):
        """Test getting pending batch indices."""
        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=150,
            batch_size=50,
            batch_threshold=50,
            n_cores=1,
            config_hash="abc123",
        )

        # Mark batch 0 and 2 as complete
        state.mark_batch_complete(0, ["body1"])
        state.mark_batch_complete(2, ["body3"])

        pending = state.get_pending_batches()
        assert pending == [1]

    def test_is_complete(self):
        """Test checking if all batches are completed."""
        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=1,
            config_hash="abc123",
        )

        assert not state.is_complete()

        state.mark_batch_complete(0, ["body1"])
        assert not state.is_complete()

        state.mark_batch_complete(1, ["body2"])
        assert state.is_complete()


class TestStateManager:
    """Tests for StateManager class."""

    def test_init(self, tmp_path):
        """Test StateManager initialization."""
        manager = StateManager(str(tmp_path), "test_sim")

        assert manager.save_path == tmp_path
        assert manager.simulation_name == "test_sim"
        assert manager.state_file == tmp_path / "simulation_state.json"

    def test_check_existing_simulation_none(self, tmp_path):
        """Test checking for existing simulation when none exists."""
        manager = StateManager(str(tmp_path), "test_sim")
        existing = manager.check_existing_simulation()
        assert existing is None

    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading state."""
        manager = StateManager(str(tmp_path), "test_sim")

        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=4,
            config_hash="abc123",
        )

        manager.save_state(state)

        assert manager.state_file.exists()

        loaded_state = manager.load_state()
        assert loaded_state.simulation_name == state.simulation_name
        assert loaded_state.batch_config == state.batch_config
        assert loaded_state.config_hash == state.config_hash

    def test_check_existing_simulation_found(self, tmp_path):
        """Test checking for existing simulation when one exists."""
        manager = StateManager(str(tmp_path), "test_sim")

        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=1,
            config_hash="abc123",
        )
        state.mark_batch_complete(0, ["body1"])
        manager.save_state(state)

        existing = manager.check_existing_simulation()
        assert existing is not None
        assert len(existing.progress["completed_batches"]) == 1

    def test_load_state_not_found(self, tmp_path):
        """Test loading state when file doesn't exist."""
        manager = StateManager(str(tmp_path), "test_sim")

        with pytest.raises(FileNotFoundError):
            manager.load_state()

    def test_load_state_corrupted(self, tmp_path):
        """Test loading corrupted state file."""
        manager = StateManager(str(tmp_path), "test_sim")

        # Create corrupted JSON file
        manager.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(manager.state_file, "w") as f:
            f.write("{ invalid json")

        with pytest.raises(ValueError, match="Corrupted state file"):
            manager.load_state()

    def test_clear_state(self, tmp_path):
        """Test clearing state file."""
        manager = StateManager(str(tmp_path), "test_sim")

        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=1,
            config_hash="abc123",
        )
        manager.save_state(state)

        assert manager.state_file.exists()

        manager.clear_state()

        assert not manager.state_file.exists()

    def test_validate_config_match_success(self, tmp_path):
        """Test config validation when configs match."""
        manager = StateManager(str(tmp_path), "test_sim")

        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=1,
            config_hash="abc123",
        )

        assert manager.validate_config_match(state, "abc123")

    def test_validate_config_match_failure(self, tmp_path):
        """Test config validation when configs don't match."""
        manager = StateManager(str(tmp_path), "test_sim")

        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=1,
            config_hash="abc123",
        )

        assert not manager.validate_config_match(state, "different_hash")

    def test_compute_config_hash(self):
        """Test computing configuration hash."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}  # Same but different order
        config3 = {"a": 1, "b": 3}  # Different value

        hash1 = StateManager.compute_config_hash(config1)
        hash2 = StateManager.compute_config_hash(config2)
        hash3 = StateManager.compute_config_hash(config3)

        assert hash1 == hash2  # Order shouldn't matter
        assert hash1 != hash3  # Different values should produce different hashes

    def test_atomic_write(self, tmp_path):
        """Test that state saving uses atomic write (temp file + rename)."""
        manager = StateManager(str(tmp_path), "test_sim")

        state = SimulationState.create_new(
            simulation_name="test_sim",
            total_bodies=100,
            batch_size=50,
            batch_threshold=50,
            n_cores=1,
            config_hash="abc123",
        )

        manager.save_state(state)

        # Verify no .tmp file left behind
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0

        # Verify final file exists
        assert manager.state_file.exists()
