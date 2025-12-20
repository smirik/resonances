import numpy as np
from typing import List, Dict, Any
import tqdm

from .config import SimulationConfig
from .state_manager import StateManager, SimulationState
from .parallel_executor import ParallelExecutor
from resonances.body import Body
from resonances.logger import logger


class BatchManager:
    """Manages batch processing orchestration for large simulations."""

    def __init__(self, config: SimulationConfig):
        """
        Initialize the BatchManager.

        Parameters
        ----------
        config : SimulationConfig
            Simulation configuration
        """
        self.config = config
        self.state_manager = StateManager(config.save_path, config.name)
        self.parallel_executor = ParallelExecutor(config.n_cores)

    def should_batch(self, num_bodies: int) -> bool:
        """
        Determine if batching is needed.

        Parameters
        ----------
        num_bodies : int
            Number of bodies in the simulation

        Returns
        -------
        bool
            True if batching should be used
        """
        if not self.config.batch_enabled:
            logger.debug("Batching disabled by configuration")
            return False

        if num_bodies <= self.config.batch_threshold:
            logger.debug(f"Number of bodies ({num_bodies}) below threshold ({self.config.batch_threshold})")
            return False

        logger.info(f"Batching enabled: {num_bodies} bodies > threshold ({self.config.batch_threshold})")
        return True

    def create_batches(self, bodies: List[Body]) -> List[List[Dict]]:
        """
        Split bodies into batches.

        Parameters
        ----------
        bodies : List[Body]
            List of bodies to batch

        Returns
        -------
        List[List[Dict]]
            List of batches, where each batch contains body data dictionaries
        """
        batch_size = self._determine_batch_size(len(bodies))
        logger.info(f"Creating batches: {len(bodies)} bodies, batch_size={batch_size}")

        batches = []
        for i in range(0, len(bodies), batch_size):
            batch_bodies = bodies[i : i + batch_size]
            batch_data = []

            for body in batch_bodies:
                resonances = body.mmrs + body.secular_resonances + body.lidov_kozai_resonances
                body_dict = {
                    "elem_or_num": body.initial_data,
                    "resonances": resonances,
                    "name": body.name,
                }
                batch_data.append(body_dict)

            batches.append(batch_data)

        logger.info(f"Created {len(batches)} batches")
        return batches

    def _determine_batch_size(self, num_bodies: int) -> int:
        """
        Determine optimal batch size.

        Parameters
        ----------
        num_bodies : int
            Total number of bodies

        Returns
        -------
        int
            Optimal batch size
        """
        if self.config.batch_size is not None:
            return self.config.batch_size

        # Auto-calculate batch size if not specified
        # Aim for reasonable batch sizes (50-100) while balancing across cores
        default_batch_size = 50
        if self.config.n_cores > 1:
            ideal_batch_size = max(1, num_bodies // (self.config.n_cores * 2))
            return min(default_batch_size, ideal_batch_size)

        return default_batch_size

    def execute_batches(self, simulation, bodies: List[Body], times: np.ndarray, progress: bool = False):
        """
        Execute batches with state management and progress tracking.

        Parameters
        ----------
        simulation : Simulation
            Parent simulation object
        bodies : List[Body]
            List of bodies to process
        times : np.ndarray
            Time array for integration
        progress : bool
            Whether to show progress bar

        Raises
        ------
        RuntimeError
            If batch execution fails
        """
        # Check for existing simulation
        existing_state = self.state_manager.check_existing_simulation()
        if existing_state is not None:
            self._handle_existing_simulation(existing_state, simulation)

        batches = self.create_batches(bodies)
        config_dict = self._get_config_dict()
        config_hash = StateManager.compute_config_hash(config_dict)

        # Create or load state
        if existing_state and self.config.resume_enabled:
            state = existing_state
            if not self.state_manager.validate_config_match(state, config_hash):
                logger.warning("Configuration mismatch - results may be inconsistent")
        else:
            state = SimulationState.create_new(
                simulation_name=self.config.name,
                total_bodies=len(bodies),
                batch_size=len(batches[0]) if batches else 0,
                batch_threshold=self.config.batch_threshold,
                n_cores=self.config.n_cores,
                config_hash=config_hash,
            )

        # Filter to pending batches
        pending_batch_indices = state.get_pending_batches()
        if not pending_batch_indices:
            logger.info("All batches already completed")
            return

        pending_batches = [(i, batches[i]) for i in pending_batch_indices]
        logger.info(f"Processing {len(pending_batches)} pending batches")

        # Prepare simulation kwargs for workers
        simulation_kwargs = self._prepare_simulation_kwargs(simulation)
        try:
            self._execute_with_progress(pending_batches, simulation_kwargs, state, progress)
            logger.info("All batches completed successfully")

            # Clear state file after successful completion
            if self.config.resume_enabled:
                self.state_manager.clear_state()

        except Exception as e:
            state.mark_failed(str(e))
            self.state_manager.save_state(state)
            logger.error(f"Batch execution failed: {e}")
            raise

    def _handle_existing_simulation(self, state: SimulationState, simulation):
        """
        Handle detection of existing simulation.

        Parameters
        ----------
        state : SimulationState
            Existing simulation state
        simulation : Simulation
            Current simulation object
        """
        if state.simulation_name != self.config.name:
            logger.warning(
                f"Found simulation with different name: '{state.simulation_name}' vs '{self.config.name}'. "
                f"This may indicate different simulations in the same directory."
            )

        if state.progress["failed"]:
            logger.warning(f"Previous simulation failed: {state.progress['error_message']}. " f"Resume will retry failed batches.")

        completed = len(state.progress["completed_batches"])
        total = state.batch_config["total_batches"]
        logger.info(f"Resuming simulation: {completed}/{total} batches already completed")

    def _get_config_dict(self) -> Dict[str, Any]:
        """Get configuration dictionary for hashing."""
        return {
            "tmax": self.config.tmax,
            "Nout": self.config.Nout,
            "integrator": self.config.integrator,
            "dt": self.config.dt,
            "batch_threshold": self.config.batch_threshold,
            "secular_angle_mode": self.config.secular_angle_mode,
        }

    def _prepare_simulation_kwargs(self, simulation) -> Dict[str, Any]:
        """
        Prepare simulation keyword arguments for workers.

        Parameters
        ----------
        simulation : Simulation
            Parent simulation object

        Returns
        -------
        Dict[str, Any]
            Simulation configuration for workers
        """
        # Automatically extract all public config attributes
        kwargs = {key: value for key, value in vars(self.config).items() if not key.startswith('_')}  # Skip private/protected attributes

        # Override batch_enabled to prevent recursive batching in workers
        kwargs["batch_enabled"] = False

        return kwargs

    def _execute_with_progress(
        self, pending_batches: List[tuple], simulation_kwargs: Dict[str, Any], state: SimulationState, show_progress: bool
    ):
        """
        Execute batches with progress tracking.

        Parameters
        ----------
        pending_batches : List[tuple]
            List of (batch_index, batch_data) tuples
        simulation_kwargs : Dict[str, Any]
            Simulation configuration for workers
        state : SimulationState
            Current simulation state
        show_progress : bool
            Whether to show progress bar
        """
        if show_progress:
            total_batches = state.batch_config["total_batches"]
            completed_batches = len(state.progress["completed_batches"])
            pbar = tqdm.tqdm(total=total_batches, initial=completed_batches, desc="Batch Processing", unit="batch")
        else:
            pbar = None

        def progress_callback(batch_index: int, result: Dict):
            """Callback for progress updates."""
            # Update state
            state.mark_batch_complete(batch_index, result["body_names"])
            self.state_manager.save_state(state)

            # Update progress bar
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"bodies": result["num_bodies"], "batch": batch_index + 1})

        try:
            # Execute batches
            self.parallel_executor.execute_batches(pending_batches, simulation_kwargs, progress_callback)
        finally:
            if pbar:
                pbar.close()
