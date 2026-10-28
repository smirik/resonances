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
        self.config = config
        self.state_manager = StateManager(config.save_path, config.name)
        self.parallel_executor = ParallelExecutor(config.n_cores)
        # {body name: [[key, parent], ...]}: derived (subresonance) rows the workers added, for the parent's bodies
        self.derived_resonances: Dict[str, List[List[str]]] = {}

    def should_batch(self, num_bodies: int) -> bool:
        """Whether the bodies are run in batches (batching on and more bodies than `batch_threshold`)."""
        if not self.config.batch_enabled:
            logger.debug("Batching disabled by configuration")
            return False

        if num_bodies <= self.config.batch_threshold:
            logger.debug(f"Number of bodies ({num_bodies}) below threshold ({self.config.batch_threshold})")
            return False

        logger.info(f"Batching enabled: {num_bodies} bodies > threshold ({self.config.batch_threshold})")
        return True

    def create_batches(self, bodies: List[Body]) -> List[List[Dict]]:
        """The bodies split into batches of body dicts (initial elements, resonances, name) for the workers."""
        batch_size = self._determine_batch_size(len(bodies))
        logger.info(f"Creating batches: {len(bodies)} bodies, batch_size={batch_size}")
        batches = [
            [{"elem_or_num": body.initial_data, "resonances": body.resonances(), "name": body.name} for body in bodies[i : i + batch_size]]
            for i in range(0, len(bodies), batch_size)
        ]
        logger.info(f"Created {len(batches)} batches")
        return batches

    def _determine_batch_size(self, num_bodies: int) -> int:
        """`batch_size`, else at most 50 and about two batches per core."""
        if self.config.batch_size is not None:
            return self.config.batch_size
        default_batch_size = 50
        if self.config.n_cores > 1:
            ideal_batch_size = max(1, num_bodies // (self.config.n_cores * 2))
            return min(default_batch_size, ideal_batch_size)

        return default_batch_size

    def execute_batches(self, simulation, bodies: List[Body], times: np.ndarray, progress: bool = False, clear_state: bool = True):
        """Run the pending batches in worker processes, recording each finished batch in simulation_state.json
        (resume); RuntimeError when a batch fails. The state file is removed after the last batch unless
        `clear_state` is False (a run with clones keeps it until its clones are done)."""
        existing_state = self.state_manager.check_existing_simulation()
        if existing_state is not None:
            self._handle_existing_simulation(existing_state, simulation)

        batches = self.create_batches(bodies)
        config_hash = StateManager.compute_config_hash(self._get_config_dict(batches))
        body_elements = self._body_elements(batches)

        # Create or load state. A resume must continue the same run: same epoch, bodies, initial elements, batches
        # and settings; anything else is refused before a completed batch is reused.
        if existing_state and self.config.resume_enabled:
            state = existing_state
            self._check_resume(state, batches, config_hash, body_elements)
        else:
            state = SimulationState.create_new(
                simulation_name=self.config.name,
                total_bodies=len(bodies),
                batch_size=len(batches[0]) if batches else 0,
                batch_threshold=self.config.batch_threshold,
                n_cores=self.config.n_cores,
                config_hash=config_hash,
                body_elements=body_elements,
            )

        # Filter to pending batches
        pending_batch_indices = state.get_pending_batches()
        if not pending_batch_indices:
            logger.info("All batches already completed")
            if self.config.resume_enabled and clear_state:
                self.state_manager.clear_state()
            return

        pending_batches = [(i, batches[i]) for i in pending_batch_indices]
        logger.info(f"Processing {len(pending_batches)} pending batches")

        # Prepare simulation kwargs for workers
        simulation_kwargs = self._prepare_simulation_kwargs(simulation)
        try:
            self._execute_with_progress(simulation, pending_batches, simulation_kwargs, state, progress)
            logger.info("All batches completed successfully")

            # Clear state file after successful completion
            if self.config.resume_enabled and clear_state:
                self.state_manager.clear_state()

        except Exception as e:
            state.mark_failed(str(e))
            self.state_manager.save_state(state)
            logger.error(f"Batch execution failed: {e}")
            raise

    def _check_resume(self, state: SimulationState, batches: List[List[Dict]], config_hash: str, body_elements: List[Dict]):
        """Accept the saved state when its hash is this run's, or one of an older version's (with a warning: less is
        checked); else ValueError (`_resume_refusal`), before any completed batch is reused."""
        if state.config_hash == config_hash:
            return
        if state.config_hash == self._legacy_config_hash():
            logger.warning(
                f"Resuming '{self.config.name}' from a state written by an older version: only tmax, Nout, the "
                "integrator, dt, batch_threshold and secular_angle_mode are checked, and the new batches are "
                "computed by this version"
            )
        elif state.config_hash == StateManager.compute_config_hash(self._get_config_dict(batches, elements=False)):
            logger.warning(
                f"Resuming '{self.config.name}' from a state written before the bodies' initial elements were "
                "recorded: the settings and batch layout match, the elements cannot be checked"
            )
        else:
            logger.warning("Configuration mismatch: the saved simulation_state.json is of another run")
            raise ValueError(self._resume_refusal(state, body_elements))

    def _handle_existing_simulation(self, state: SimulationState, simulation):
        """Log what an existing simulation_state.json says (another name, a failure, the batches done)."""
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

    # Settings that change no saved number: names and paths, the figures, how the work is spread.
    # The clone settings (every `clones*` key) too: the clones run after these batches, outside them, and their own
    # checkpoint covers them (simulation/ensemble_run.py, clone_store.py).
    RESUME_IGNORED = (
        'name',
        'save_path',
        'plot_path',
        'image_type',
        'n_cores',
        'batch_enabled',
        'resume_enabled',
    )
    # data_folder values a resumed run may switch between: the manifest looks in both (serializer.MANIFEST_DATA_FOLDERS),
    # and leaving them out of the hash keeps resuming the runs started before the data folder. Any other value is in
    # the hash: changing it would leave the completed batches' series where the manifest does not look.
    RESUME_DATA_FOLDERS = ('', 'data')

    def _get_config_dict(self, batches: List[List[Dict]] | None = None, elements: bool = True) -> Dict[str, Any]:
        """What a resumed run must share with the interrupted one, for the resume hash.

        The whole serialised config (date and source included: a run without date= starts at today's
        midnight) except RESUME_IGNORED and the plot settings, plus the batch layout as body names and
        resonances, since completed batches are matched by index, and every body's initial elements as
        they were used (`clone_store.canonical_elements`: exact float reprs, with the get-orb sha256 of a
        get-orb body). The elements are what was obtained, not how: a refreshed get-orb answer, a new
        catalogue or a Horizons re-query that returns another solution is another run. `elements=False`
        gives the hash of the versions before the elements were included.
        """
        from .serializer import SimulationSerializer

        config = SimulationSerializer._config_to_dict(self.config)
        config_dict = {
            k: v for k, v in config.items() if k not in self.RESUME_IGNORED and not k.startswith('plot') and not k.startswith('clones')
        }
        if config_dict.get('data_folder') in self.RESUME_DATA_FOLDERS:
            config_dict.pop('data_folder')
        if batches is not None:
            config_dict["batches"] = [[[body["name"], [r.to_s() for r in body["resonances"]]] for body in batch] for batch in batches]
            if elements:
                from .clone_store import canonical_elements

                config_dict["elements"] = [[canonical_elements(body["elem_or_num"]) for body in batch] for batch in batches]
        return config_dict

    @staticmethod
    def _body_elements(batches: List[List[Dict]]) -> List[Dict[str, Any]]:
        """Per body, in batch order: its name, the digest of its canonical initial elements and its get-orb sha256
        (None unless the elements came from MPC get-orb). Kept in simulation_state.json for `_resume_refusal`."""
        from .clone_store import canonical_elements, digest

        records = []
        for batch in batches:
            for body in batch:
                elements = canonical_elements(body["elem_or_num"])
                records.append({"name": str(body["name"]), "elements": digest(elements), "getorb_sha256": elements.get("getorb_sha256")})
        return records

    def _resume_refusal(self, state: SimulationState, body_elements: List[Dict[str, Any]]) -> str:
        """The message of a refused resume: the bodies whose initial elements changed when the state names them,
        else the general list of what the hash covers."""
        where = f"Cannot resume '{self.config.name}' in {self.config.save_path}"
        start_over = "remove simulation_state.json to start over in an empty folder."
        old = {record["name"]: record for record in (state.body_elements or [])}
        changed = []
        for record in body_elements:
            before = old.get(record["name"])
            if before is None or before["elements"] == record["elements"]:
                continue
            if before.get("getorb_sha256") != record["getorb_sha256"]:
                changed.append(f"{record['name']} (get-orb answer sha256 {before.get('getorb_sha256')} -> {record['getorb_sha256']})")
            else:
                changed.append(record["name"])
        if changed:
            return (
                f"{where}: the initial elements of {', '.join(changed)} differ from those of the interrupted run (a "
                "refreshed get-orb answer, another catalogue or a Horizons re-query with a new solution); its completed "
                f"batches were computed from the old elements. Restore the original elements, or {start_over}"
            )
        return (
            f"{where}: the date, source, bodies, their initial elements, batch layout or settings differ from the "
            f"interrupted run (a run without date= starts today). Pass the original settings, or {start_over}"
        )

    def _legacy_config_hash(self) -> str:
        """The resume hash of versions before the full-config one (simulation_state.json written by them)."""
        config_dict = {
            "tmax": self.config.tmax,
            "Nout": self.config.Nout,
            "integrator": self.config.integrator,
            "dt": self.config.dt,
            "batch_threshold": self.config.batch_threshold,
            "secular_angle_mode": self.config.secular_angle_mode,
        }
        if self.config.solar_system_moon:
            config_dict["solar_system_moon"] = True
        return StateManager.compute_config_hash(config_dict)

    def _prepare_simulation_kwargs(self, simulation) -> Dict[str, Any]:
        """The workers' Simulation kwargs: every public config attribute, tmax (a property) and no batching."""
        kwargs = {key: value for key, value in vars(self.config).items() if not key.startswith('_')}
        kwargs["tmax"] = self.config.tmax
        kwargs["batch_enabled"] = False
        kwargs["clones"] = 0  # a worker's bodies are given by their elements: their clones are the parent's (ensemble_run)
        # Every worker keeps the planets of the whole run (save_planets='analysis'), so all write the same files
        kwargs["_planets_needed"] = simulation.planets_needed()

        return kwargs

    def _execute_with_progress(
        self, simulation, pending_batches: List[tuple], simulation_kwargs: Dict[str, Any], state: SimulationState, show_progress: bool
    ):
        """Run `pending_batches` ((index, bodies) pairs) with a progress bar; the parent's DataManager first drops the
        old rows of the batches about to run."""
        if show_progress:
            total_batches = state.batch_config["total_batches"]
            completed_batches = len(state.progress["completed_batches"])
            pbar = tqdm.tqdm(total=total_batches, initial=completed_batches, desc="Batch Processing", unit="batch")
        else:
            pbar = None

        def drop_old_rows(batches):
            # A batch about to run may have written rows before (a crash while plotting, a broken pool,
            # a resume): one pass removes them here, so the workers only append.
            if self.config.save_summary:
                removed = simulation.data_manager.drop_rows([body["name"] for _, batch in batches for body in batch])
                if removed:
                    logger.info(f"Removed {removed} summary rows of bodies that are run again")

        def progress_callback(batch_index: int, result: Dict):
            for name, rows in (result.get("derived") or {}).items():
                self.derived_resonances.setdefault(name, []).extend(rows)
            state.mark_batch_complete(batch_index, result["body_names"])
            self.state_manager.save_state(state)
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"bodies": result["num_bodies"], "batch": batch_index + 1})

        try:
            self.parallel_executor.execute_batches(pending_batches, simulation_kwargs, progress_callback, before_run=drop_old_rows)
        finally:
            if pbar:
                pbar.close()
