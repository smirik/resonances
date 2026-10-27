import ast
import multiprocessing as mp
import sys
from typing import List, Callable, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, BrokenExecutor

from resonances.config import low_memory_allocator, worker_allocator
from resonances.logger import logger

MAIN_GUARD_MESSAGE = (
    "The worker processes could not start: the script {path} runs the simulation at line {line} outside "
    "`if __name__ == '__main__':`. The workers are spawned processes that import the script again, so the run must "
    "be started under that guard (docs/install.md)."
)


def _is_main_guard(node: ast.AST) -> bool:
    """`if __name__ == '__main__':` (either order of the comparison)."""
    if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1):
        return False
    if not isinstance(node.test.ops[0], ast.Eq):
        return False
    sides = [node.test.left, node.test.comparators[0]]
    names = [side.id for side in sides if isinstance(side, ast.Name)]
    constants = [side.value for side in sides if isinstance(side, ast.Constant)]
    return names == ['__name__'] and constants == ['__main__']


def line_outside_main_guard(source: str, line: int) -> bool:
    """Whether module-level `line` of `source` lies outside the body of every `if __name__ == '__main__':` block."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if _is_main_guard(node) and node.body and node.body[0].lineno <= line <= (node.body[-1].end_lineno or line):
            return False
    return True


def missing_main_guard() -> Optional[tuple]:
    """(script path, line) when this call comes from the module level of a `__main__` script outside its
    `if __name__ == '__main__':` guard, else None (an interactive session, a guarded script, a script not readable).
    A spawned worker imports that script again, runs the simulation itself and fails while bootstrapping, which the
    parent sees only as a broken pool."""
    main = sys.modules.get('__main__')
    path = getattr(main, '__file__', None)
    if not path:
        return None
    frame = sys._getframe(1)
    line = None
    while frame is not None:  # the outermost frame of the script's module level
        if frame.f_code.co_name == '<module>' and frame.f_globals.get('__name__') == '__main__':
            line = frame.f_lineno
        frame = frame.f_back
    if line is None:
        return None
    try:
        with open(path, encoding='utf-8') as handle:
            source = handle.read()
        return (path, line) if line_outside_main_guard(source, line) else None
    except (OSError, SyntaxError, ValueError):
        return None


def _run_batch_worker(batch_index: int, bodies_data: List[Dict], simulation_kwargs: Dict[str, Any]) -> Dict:
    """Run one batch in this (worker) process: a Simulation of its bodies writes its own rows and files.
    Returns batch_index, status ('success' or 'failed'), body_names, num_bodies, the derived rows and the error."""
    from resonances.simulation.simulation import Simulation

    try:
        batch_kwargs = {**simulation_kwargs, '_skip_path_verification': True}
        clone_batch = bool(batch_kwargs.pop('_clone_batch', False))  # clones of the parent's ensembles (ensemble_run)
        sim = Simulation(**batch_kwargs)
        sim.data_manager.skip_simulation_json = True  # Don't save simulation.json per batch
        sim.data_manager.replace_rows = False  # the parent dropped this batch's old rows (execute_batches)
        sim.create_solar_system()

        body_names = []
        for body_data in bodies_data:
            sim.add_body(body_data["elem_or_num"], body_data["resonances"], name=body_data["name"])
            body_names.append(body_data["name"])
        sim.run(progress=False)

        # Rows the MMR analysis added (accepted subresonances): the parent adds them to its bodies before it
        # writes simulation.json, or restore would not know them.
        derived = {str(body.name): [[key, parent] for key, parent in body.derived_resonances.items()] for body in sim.bodies}
        result = {
            "batch_index": batch_index,
            "status": "success",
            "body_names": body_names,
            "num_bodies": len(body_names),
            "derived": {name: rows for name, rows in derived.items() if rows},
            "error": None,
        }
        if clone_batch:  # each clone's verdicts, initial a and thinned angle, for clones.csv and the ensemble figure
            from resonances.simulation.ensemble_run import worker_results

            result["clones"] = worker_results(sim)
        return result

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
        self.n_cores = n_cores
        if self.n_cores < 1:
            raise ValueError(f"n_cores must be >= 1, got {self.n_cores}")

        cpu_count = mp.cpu_count()
        if self.n_cores > cpu_count:
            logger.warning(f"Requested n_cores={self.n_cores} exceeds available CPU cores ({cpu_count}). " f"This may degrade performance.")

    def execute_batches(
        self, batches: List[tuple], simulation_kwargs: Dict[str, Any], progress_callback: Callable = None, before_run: Callable = None
    ) -> List[Dict]:
        """Run `batches` ((index, body dicts) pairs) on n_cores; returns the results; RuntimeError if a batch fails.

        `progress_callback(index, result)` follows each finished batch. `before_run(pending)` is called with the
        batches about to be submitted, before the first round and before each retry after a broken pool: those
        batches may have written rows already."""
        if self.n_cores == 1:
            if before_run:
                before_run(batches)
            return self._execute_sequential(batches, simulation_kwargs, progress_callback)
        return self._execute_parallel(batches, simulation_kwargs, progress_callback, before_run=before_run)

    def _execute_sequential(
        self, batches: List[tuple], simulation_kwargs: Dict[str, Any], progress_callback: Callable = None
    ) -> List[Dict]:
        """The batches one after another in this process (n_cores = 1)."""
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

    def _execute_parallel(
        self,
        batches: List[tuple],
        simulation_kwargs: Dict[str, Any],
        progress_callback: Callable = None,
        max_retries: int = 3,
        before_run: Callable = None,
    ) -> List[Dict]:
        """The batches in a process pool, one batch per process; a broken pool (usually out of memory) is replaced
        and its unfinished batches retried, at most `max_retries` times."""
        results = []
        pending_batches = list(batches)
        completed_indices = set()
        retry_count = 0

        def record(batch_index, result):
            results.append(result)
            completed_indices.add(batch_index)
            if progress_callback:
                progress_callback(batch_index, result)

        logger.info(f"Starting parallel execution with {self.n_cores} cores")

        while pending_batches and retry_count <= max_retries:
            if retry_count > 0:
                logger.warning(f"Retrying with fresh process pool (attempt {retry_count}/{max_retries})")
            if before_run:
                before_run(pending_batches)

            failure = None
            broken = False
            # One batch per process: a reused worker kept the previous batch's memory and
            # grew to 10 GB on its second batch (50 kyr, Nout 83k). Implies the "spawn" start method.
            # The workers inherit the low-memory allocator when LOW_MEMORY_ALLOCATOR is on (n_cores is chosen by the
            # user, not from the memory: auto stays off here; config.low_memory_allocator).
            allocator = worker_allocator(low_memory_allocator(ram_limited=False))
            with allocator, ProcessPoolExecutor(max_workers=self.n_cores, max_tasks_per_child=1) as executor:
                future_to_batch = {
                    executor.submit(_run_batch_worker, batch_index, batch_bodies, simulation_kwargs): batch_index
                    for batch_index, batch_bodies in pending_batches
                }
                for future in as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    try:
                        result = future.result()
                    except BrokenExecutor as e:
                        logger.error(f"Process pool broken while processing batch {batch_index}: {e}")
                        unguarded = missing_main_guard()
                        if unguarded is not None:  # every retry would fail the same way
                            failure = RuntimeError(MAIN_GUARD_MESSAGE.format(path=unguarded[0], line=unguarded[1]))
                            logger.error(str(failure))
                            break
                        logger.error("This usually indicates out-of-memory. Consider reducing batch_size or n_cores.")
                        broken = True
                        break
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_index}: {e}")
                        failure = e
                        break
                    if result["status"] == "failed":
                        failure = RuntimeError(f"Batch {batch_index} failed: {result['error']}")
                        break
                    record(batch_index, result)
                if failure is not None or broken:
                    for f in future_to_batch:
                        f.cancel()
            # Leaving the pool waited for the batches already running: those that finished wrote their
            # files, so they are recorded as complete rather than run again on resume or retry.
            for future, batch_index in future_to_batch.items():
                if batch_index in completed_indices or not future.done() or future.cancelled() or future.exception() is not None:
                    continue
                if future.result()["status"] == "success":
                    record(batch_index, future.result())

            if failure is not None:
                raise failure
            pending_batches = [(idx, bodies) for idx, bodies in batches if idx not in completed_indices]
            if not broken:
                break
            retry_count += 1
            if pending_batches and retry_count <= max_retries:
                logger.info(f"Retrying {len(pending_batches)} remaining batches...")
            elif pending_batches:
                raise RuntimeError(
                    f"Process pool failed {max_retries} times. {len(pending_batches)} batches remaining. "
                    "This is likely due to out-of-memory. Try reducing batch_size or n_cores."
                )

        results.sort(key=lambda x: x["batch_index"])

        return results
