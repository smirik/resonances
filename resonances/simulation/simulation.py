import numpy as np
from typing import List, Union

from resonances.resonance.classify import classify_resonance
from resonances.resonance.periodogram import Periodogram

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
    def bodies(self):
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
        self.batch_manager.execute_batches(self, self.bodies, self.times, progress)

    def prepare_angles(self):
        for body in self.bodies:
            for resonance in body.resonances():
                # make wrapped angle, filter, and record wrapped filtered
                body.angles_unwrapped[resonance.to_s()] = np.unwrap(body.angle_unwrapped(resonance))
                body.angles[resonance.to_s()] = wrap(body.angle_unwrapped(resonance))
                if isinstance(resonance, SecularResonance):
                    body.build_proper_angle(resonance)
                    if self.config.secular_angle_mode == "proper":
                        body.angles[resonance.to_s()] = body.secular_angles_proper[resonance.to_s()]
                unwrapped_filtered_angle = filter_angle(body.angle_unwrapped(resonance), self.config)
                body.angles_filtered_unwrapped[resonance.to_s()] = unwrapped_filtered_angle
                body.angles_filtered[resonance.to_s()] = filter_angle(body.angles[resonance.to_s()], self.config)
                body.axis_filtered = filter_angle(body.axis, self.config)

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
        """Identify librations for all bodies."""
        for body in self.bodies:
            for resonance in body.resonances():
                libration = classify_resonance(
                    body,
                    self.times,
                    resonance=resonance,
                )
                body.librations[resonance.to_s()] = libration
                body.statuses[resonance.to_s()] = libration['status']
