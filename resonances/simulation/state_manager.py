import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Set, List, Dict, Any
from dataclasses import dataclass, asdict, field

from resonances.logger import logger


@dataclass
class SimulationState:
    """Represents the state of a batch simulation."""

    version: str
    simulation_name: str
    started_at: str
    last_updated: str
    batch_config: Dict[str, Any]
    progress: Dict[str, Any]
    config_hash: str
    # One record per body, in batch order: {'name', 'elements' (digest of its canonical initial elements),
    # 'getorb_sha256' (None unless from MPC get-orb)}; lets a refused resume name the bodies whose elements changed.
    # Absent (None) in the states of versions before it.
    body_elements: Optional[List[Dict[str, Any]]] = field(default=None)

    @classmethod
    def create_new(
        cls,
        simulation_name: str,
        total_bodies: int,
        batch_size: int,
        batch_threshold: int,
        n_cores: int,
        config_hash: str,
        body_elements: Optional[List[Dict[str, Any]]] = None,
    ) -> "SimulationState":
        """Create a new simulation state."""
        now = datetime.now().isoformat()
        total_batches = (total_bodies + batch_size - 1) // batch_size

        return cls(
            version="1.0.0",
            simulation_name=simulation_name,
            started_at=now,
            last_updated=now,
            batch_config={
                "total_bodies": total_bodies,
                "batch_size": batch_size,
                "batch_threshold": batch_threshold,
                "n_cores": n_cores,
                "total_batches": total_batches,
            },
            progress={
                "current_batch": 0,
                "total_batches": total_batches,
                "completed_batches": [],
                "completed_bodies": [],
                "failed": False,
                "error_message": None,
            },
            config_hash=config_hash,
            body_elements=body_elements,
        )

    def mark_batch_complete(self, batch_index: int, body_names: List[str]):
        """Mark a batch as completed."""
        if batch_index not in self.progress["completed_batches"]:
            self.progress["completed_batches"].append(batch_index)
            self.progress["completed_batches"].sort()
        # Add only new body names (avoid duplicates)
        existing_bodies = set(self.progress["completed_bodies"])
        new_bodies = [name for name in body_names if name not in existing_bodies]
        self.progress["completed_bodies"].extend(new_bodies)
        self.progress["current_batch"] = batch_index + 1
        self.last_updated = datetime.now().isoformat()

    def mark_failed(self, error_message: str):
        """Mark the simulation as failed."""
        self.progress["failed"] = True
        self.progress["error_message"] = error_message
        self.last_updated = datetime.now().isoformat()

    def get_completed_bodies(self) -> Set[str]:
        """Return set of completed body names."""
        return set(self.progress["completed_bodies"])

    def get_pending_batches(self) -> List[int]:
        """Return list of pending batch indices."""
        completed = set(self.progress["completed_batches"])
        total = self.batch_config["total_batches"]
        return [i for i in range(total) if i not in completed]

    def is_complete(self) -> bool:
        """Check if all batches are completed."""
        return len(self.progress["completed_batches"]) == self.batch_config["total_batches"]


class StateManager:
    """Manages simulation state persistence for batch processing."""

    STATE_FILENAME = "simulation_state.json"

    def __init__(self, save_path: str, simulation_name: str):
        self.save_path = Path(save_path)
        self.simulation_name = simulation_name
        self.state_file = self.save_path / self.STATE_FILENAME

    def check_existing_simulation(self) -> Optional[SimulationState]:
        """The state of an earlier run in save_path, or None (no file, or one that cannot be read)."""
        if not self.state_file.exists():
            return None

        try:
            state = self.load_state()
            logger.info(
                f"Found existing simulation '{state.simulation_name}' "
                f"(started: {state.started_at}). "
                f"Completed: {len(state.progress['completed_batches'])}/{state.batch_config['total_batches']} batches"
            )
            return state
        except Exception as e:
            logger.warning(f"Found state file but failed to load: {e}")
            return None

    def save_state(self, state: SimulationState):
        """Write the state to simulation_state.json (a temporary file renamed into place)."""
        try:
            self.save_path.mkdir(parents=True, exist_ok=True)
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(asdict(state), f, indent=2)
            temp_file.replace(self.state_file)

            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise

    def load_state(self) -> SimulationState:
        """The state in simulation_state.json; FileNotFoundError without it, ValueError when it is invalid."""
        if not self.state_file.exists():
            raise FileNotFoundError(f"State file not found: {self.state_file}")

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            state = SimulationState(**data)
            logger.debug(f"State loaded from {self.state_file}")
            return state
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted state file: {e}")
        except Exception as e:
            raise ValueError(f"Invalid state file: {e}")

    def clear_state(self):
        """Remove state file after successful completion."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info(f"State file removed: {self.state_file}")

    def validate_config_match(self, state: SimulationState, current_config_hash: str) -> bool:
        """Whether the saved state was written with this configuration hash (a warning when not)."""
        if state.config_hash != current_config_hash:
            logger.warning(
                "Configuration mismatch detected! "
                "The current simulation configuration differs from the saved state. "
                "This may lead to inconsistent results."
            )
            return False
        return True

    @staticmethod
    def compute_config_hash(config_dict: Dict[str, Any]) -> str:
        """MD5 of the configuration dict (keys sorted)."""
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
