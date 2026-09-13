import numpy as np
from typing import List, Union

from resonances.body import Body
from resonances.resonance import coherence_analysis, omega_free_gate
from resonances.resonance.classify import classify_resonance
from resonances.resonance.classify.models import ResonanceStatus
from resonances.resonance.periodogram import Periodogram
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance

from .config import SimulationConfig
from .body_manager import BodyManager
from .integration import IntegrationEngine
from .data_manager import DataManager
from .batch_manager import BatchManager

from resonances.secular.secular_resonance import SecularResonance
from resonances.resonance.resonance import Resonance
from resonances.logger import logger
from resonances.resonance.filtering import filter_angle, wrap


class Simulation:
    """
    Main Simulation class with component-based architecture.

    This class orchestrates the various components to run resonance simulations.
    """

    def __init__(self, **kwargs):
        """Initialize the simulation with component-based architecture."""
        self.config = SimulationConfig(**kwargs)
        self.body_manager = BodyManager(self.config)
        self.integration_engine = IntegrationEngine(self.config)
        self.data_manager = DataManager(self.config)
        self.batch_manager = BatchManager(self.config)

        self.running_time = {
            "start": None,
            "adding_bodies_started": None,
            "integration_started": None,
            "integration_finished": None,
            "librations_identified": None,
            "bodies_saved": None,
            "stop": None,
        }

        self.running_time["start"] = logger.get_current_time()

        self.times = []

    @property
    def bodies(self) -> List[Body]:
        return self.body_manager.bodies

    def create_solar_system(self, force=False):
        """Create or load the Solar System simulation."""
        self.integration_engine.create_solar_system(force)

    def add_body(self, elem_or_num, resonance: Union[Resonance, str, list[Resonance], list[str]], name='asteroid'):  # noqa: C901
        """Add a celestial body with any resonances."""
        self.body_manager.add_body(elem_or_num, resonance, name)

    def add_bodies(self, bodies: List[str], resonance, prefix: str = None):
        """Add multiple celestial bodies to the simulation."""
        if prefix is None:
            prefix = ""
        else:
            prefix = f"{prefix}_"

        for body in bodies:
            self.add_body(body, resonance, f"{prefix}{body}")

    def run(self, progress=False):
        """Run the complete simulation with optional batching."""
        self.times = np.linspace(0.0, self.config.tmax, self.config.Nout)

        if self.batch_manager.should_batch(len(self.bodies)):
            self._run_batched(progress)
        else:
            self._run_single(progress)
        self.running_time["stop"] = logger.get_current_time()

    def _run_single(self, progress=False):
        """Run single-batch execution (original behavior)."""
        self.running_time["adding_bodies_started"] = logger.get_current_time()
        self.body_manager.add_bodies_to_simulation(self.integration_engine.sim)
        self.running_time["integration_started"] = logger.get_current_time()
        self.integration_engine.run_integration(self.bodies, self.times, progress)
        self.running_time["integration_finished"] = logger.get_current_time()
        self.prepare_angles()
        self.build_periodograms()
        self.identify_librations()
        self.running_time["librations_identified"] = logger.get_current_time()
        self.data_manager.save_data(self.bodies, self.times, self)

    def _run_batched(self, progress=False):
        """Run batched execution with multi-core support."""
        self.running_time["adding_bodies_started"] = logger.get_current_time()
        self.running_time["integration_started"] = logger.get_current_time()
        self.batch_manager.execute_batches(self, self.bodies, self.times, progress)
        self.running_time["integration_finished"] = logger.get_current_time()
        self.running_time["librations_identified"] = logger.get_current_time()
        self.running_time["bodies_saved"] = logger.get_current_time()
        self.running_time["stop"] = logger.get_current_time()
        # Save consolidated simulation.json with all bodies after batches complete
        self.data_manager.save_configuration_details(self.bodies, self)

    def prepare_angles(self):
        for body in self.bodies:
            body.axis_filtered = filter_angle(body.axis, self.config)
            for resonance in body.resonances():
                res_key = resonance.to_s()
                # np.unwrap the raw integration angles and store back
                body.angles_unwrapped[res_key] = np.unwrap(body.angle_unwrapped(resonance))
                # Subsequent reads of angle_unwrapped() now return the unwrapped version
                unwrapped = body.angle_unwrapped(resonance)
                body.angles[res_key] = wrap(unwrapped)
                if isinstance(resonance, SecularResonance):
                    body.build_proper_angle(resonance)
                    if self.config.secular_angle_mode == "proper":
                        body.angles[res_key] = body.secular_angles_proper[res_key]
                unwrapped_filtered_angle = filter_angle(unwrapped, self.config)
                body.angles_filtered_unwrapped[res_key] = unwrapped_filtered_angle
                body.angles_filtered[res_key] = wrap(unwrapped_filtered_angle)

    def build_periodograms(self):
        base_periodogram_config = {
            'integration_time_yrs': self.config.tmax_yrs,
            'Nout': self.config.Nout,
            'libration_period_min': self.config.libration_period_min,
            'minimum_frequency': self.config.periodogram_frequency_min,
            'maximum_frequency': self.config.periodogram_frequency_max,
            'threshold': self.config.periodogram_soft,
        }

        for body in self.bodies:
            (
                body.axis_periodogram_frequency,
                body.axis_periodogram_power,
                body.axis_periodogram_peaks,
            ) = Periodogram.periodogram(
                self.times,
                body.axis_filtered,
                label=f'{body.name}:a',
                **base_periodogram_config,
            )
            (
                body.eccentricity_periodogram_frequency,
                body.eccentricity_periodogram_power,
                body.eccentricity_periodogram_peaks,
            ) = Periodogram.periodogram(
                self.times,
                body.ecc,
                label=f'{body.name}:e',
                **base_periodogram_config,
            )

            for resonance in body.resonances():
                (
                    body.periodogram_frequency[resonance.to_s()],
                    body.periodogram_power[resonance.to_s()],
                    body.periodogram_peaks[resonance.to_s()],
                ) = Periodogram.periodogram(
                    self.times,
                    body.angles_filtered[resonance.to_s()],  # if not filtered, easy to skip relevant frequencies
                    label=f'{body.name}:{resonance.to_s()}',
                    **base_periodogram_config,
                )

    def identify_librations(self):
        """Identify librations for all bodies, then run the post-classification checks.

        Both checks run here rather than in a separate step because this method is
        also the entry point used by `SimulationSerializer.restore()`, which
        reclassifies from saved CSVs without replaying `prepare_angles`.

        Order matters: the cross spectra are pure diagnostics and only the free-omega
        gate may change a status, so it runs last and has the final word.
        """
        for body in self.bodies:
            # Welch spectra of a, e and i are shared by every resonance of a body.
            psd_cache = {} if self.config.coherence_enabled else None
            # The forced/free split depends on the orbit alone, so it is done once per
            # body, and only where it is read — the Lidov-Kozai angle omega = varpi - Omega.
            if self.config.free_elements_enabled and body.lidov_kozai_resonances:
                body.build_free_elements(self.config.free_elements_sampling_years, self.config.free_gate_mask_quantile)
            for resonance in body.resonances():
                classification = classify_resonance(
                    body,
                    resonance=resonance,
                    config=self.config,
                )
                libration = classification["result"]
                if "segments" in classification:
                    body.libration_segments[resonance.to_s()] = classification["segments"]
                body.librations[resonance.to_s()] = libration
                body.statuses[resonance.to_s()] = libration.status.value
                self._analyse_coherence(body, resonance, libration, psd_cache)
                self._apply_free_omega_gate(body, resonance, libration)

    def _analyse_coherence(self, body: Body, resonance: Resonance, libration, psd_cache):
        """Run the cross spectrum for a body-resonance and record the e-i exchange flag.

        Diagnostic only: nothing here changes a status. See `coherence_analysis`.
        """
        if not self.config.coherence_enabled:
            return
        if self.config.coherence_skip_non_resonant and libration.status == ResonanceStatus.NON_RESONANT:
            return

        res_key = resonance.to_s()
        analyses = coherence_analysis.analyse_resonance(body, resonance, self.config, psd_cache=psd_cache)
        body.coherence[res_key] = analyses

        if isinstance(resonance, LidovKozaiResonance):
            body.zlk_gates[res_key] = coherence_analysis.evaluate_ei_exchange(analyses, self.config)

    def _apply_free_omega_gate(self, body: Body, resonance: Resonance, libration):
        """Confirm, demote or withhold a Lidov-Kozai libration from the free omega.

        The only hook that changes a status after classification. Both `libration.status`
        and `body.statuses` are updated so that `summary.csv` (which reads the flattened
        classification result) cannot disagree with the save/plot filters (which read
        `body.statuses`), and the decision leaves a "gate:" trail in the comments.
        """
        if not isinstance(resonance, LidovKozaiResonance) or not self.config.free_elements_enabled:
            return

        res_key = resonance.to_s()
        status, gate = omega_free_gate.apply_gate(
            libration.status,
            body.free_elements,
            libration.metrics.libration_period_1,
            self.config,
        )
        body.free_gates[res_key] = gate
        if gate.comment:
            libration.comments = f"{libration.comments} {gate.comment}".strip() if libration.comments else gate.comment
        if status != libration.status:
            logger.info(f"{body.name}/{res_key}: free omega {gate.outcome} — status {libration.status} -> {status}")
            libration.status = status
            body.statuses[res_key] = status.value
