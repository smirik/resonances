import os
import multiprocessing as mp
from typing import List, Callable, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

from resonances.logger import logger


def _run_batch_worker(batch_index: int, bodies_data: List[Dict], simulation_kwargs: Dict[str, Any]) -> Dict:
    """
    Worker function for parallel batch execution.

    This function runs in a separate process and must be picklable.

    Parameters
    ----------
    batch_index : int
        Index of the batch being processed
    bodies_data : List[Dict]
        List of body data dictionaries
    simulation_kwargs : Dict[str, Any]
        Simulation configuration parameters

    Returns
    -------
    Dict
        Result containing batch_index, status, body_names, and any errors
    """
    # Import here to avoid circular imports and ensure fresh imports in subprocess
    from resonances.simulation.simulation import Simulation

    try:
        # Create a new simulation for this batch
        sim = Simulation(**simulation_kwargs)
        sim.create_solar_system()

        # Add bodies to simulation
        body_names = []
        for body_data in bodies_data:
            # Use resonance objects directly (passed via pickle)
            resonances_list = body_data["resonances"]

            sim.add_body(body_data["elem_or_num"], resonances_list, name=body_data["name"])
            body_names.append(body_data["name"])

        # Run simulation without progress bar (to avoid conflicts in parallel)
        sim.run(progress=False)

        return {
            "batch_index": batch_index,
            "status": "success",
            "body_names": body_names,
            "num_bodies": len(body_names),
            "error": None,
        }

    except Exception as e:
        logger.error(f"Batch {batch_index} failed: {e}")
        return {
            "batch_index": batch_index,
            "status": "failed",
            "body_names": [],
            "num_bodies": 0,
            "error": str(e),
        }


class ParallelExecutor:
    """Handles multi-core execution of simulation batches."""

    def __init__(self, n_cores: int = 1):
        """
        Initialize the ParallelExecutor.

        Parameters
        ----------
        n_cores : int, optional
            Number of CPU cores to use (default: 1)
        """
        self.n_cores = n_cores
        self._validate_n_cores()

    def _validate_n_cores(self):
        """Validate n_cores configuration."""
        if self.n_cores < 1:
            raise ValueError(f"n_cores must be >= 1, got {self.n_cores}")

        cpu_count = mp.cpu_count()
        if self.n_cores > cpu_count:
            logger.warning(f"Requested n_cores={self.n_cores} exceeds available CPU cores ({cpu_count}). " f"This may degrade performance.")

    def execute_batches(self, batches: List[tuple], simulation_kwargs: Dict[str, Any], progress_callback: Callable = None) -> List[Dict]:
        """
        Execute batches using configured number of cores.

        Parameters
        ----------
        batches : List[tuple]
            List of (batch_index, batch_data) tuples where batch_data is a list of body dictionaries
        simulation_kwargs : Dict[str, Any]
            Simulation configuration parameters
        progress_callback : Callable, optional
            Callback function to report progress (default: None)

        Returns
        -------
        List[Dict]
            List of results from each batch

        Raises
        ------
        RuntimeError
            If any batch fails
        """
        if self.n_cores == 1:
            return self._execute_sequential(batches, simulation_kwargs, progress_callback)
        else:
            return self._execute_parallel(batches, simulation_kwargs, progress_callback)

    def _execute_sequential(
        self, batches: List[tuple], simulation_kwargs: Dict[str, Any], progress_callback: Callable = None
    ) -> List[Dict]:
        """
        Execute batches sequentially (n_cores=1).

        Parameters
        ----------
        batches : List[tuple]
            List of (batch_index, batch_data) tuples to execute
        simulation_kwargs : Dict[str, Any]
            Simulation configuration parameters
        progress_callback : Callable, optional
            Callback function to report progress

        Returns
        -------
        List[Dict]
            List of results from each batch
        """
        results = []

        for batch_index, batch_bodies in batches:
            logger.info(f"Processing batch {batch_index + 1}/{len(batches)} ({len(batch_bodies)} bodies)")

            result = _run_batch_worker(batch_index, batch_bodies, simulation_kwargs)

            if result["status"] == "failed":
                raise RuntimeError(f"Batch {batch_index} failed: {result['error']}")

            results.append(result)

            if progress_callback:
                progress_callback(batch_index, result)

        return results

    def _execute_parallel(self, batches: List[tuple], simulation_kwargs: Dict[str, Any], progress_callback: Callable = None) -> List[Dict]:
        """
        Execute batches in parallel using ProcessPoolExecutor.

        Parameters
        ----------
        batches : List[tuple]
            List of (batch_index, batch_data) tuples to execute
        simulation_kwargs : Dict[str, Any]
            Simulation configuration parameters
        progress_callback : Callable, optional
            Callback function to report progress

        Returns
        -------
        List[Dict]
            List of results from each batch
        """
        results = []

        logger.info(f"Starting parallel execution with {self.n_cores} cores")

        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Submit all batches
            future_to_batch = {}
            for batch_index, batch_bodies in batches:
                future = executor.submit(_run_batch_worker, batch_index, batch_bodies, simulation_kwargs)
                future_to_batch[future] = batch_index

            # Process results as they complete
            for future in as_completed(future_to_batch):
                batch_index = future_to_batch[future]

                try:
                    result = future.result()

                    if result["status"] == "failed":
                        # Cancel remaining futures
                        for f in future_to_batch:
                            f.cancel()
                        raise RuntimeError(f"Batch {batch_index} failed: {result['error']}")

                    results.append(result)

                    if progress_callback:
                        progress_callback(batch_index, result)

                except Exception as e:
                    logger.error(f"Error processing batch {batch_index}: {e}")
                    # Cancel remaining futures
                    for f in future_to_batch:
                        f.cancel()
                    raise

        # Sort results by batch_index to maintain order
        results.sort(key=lambda x: x["batch_index"])

        return results

    @staticmethod
    def configure_environment():
        """
        Configure environment variables for single-threaded execution.

        This prevents numpy/scipy from using internal parallelization,
        which would conflict with our process-level parallelism.
        """
        env_vars = {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }

        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
                logger.debug(f"Set {key}={value}")
