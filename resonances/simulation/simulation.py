import numpy as np
from typing import List, Union

import resonances
from .config import SimulationConfig
from .body_manager import BodyManager
from .integration import IntegrationEngine
from .data_manager import DataManager

from resonances.secular.proper_angle import build_proper_angle_series
from resonances.secular.secular_resonance import SecularResonance


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

        self.times = []

    @property
    def bodies(self):
        return self.body_manager.bodies

    def create_solar_system(self, force=False):
        """Create or load the Solar System simulation."""
        self.integration_engine.create_solar_system(force)

    def add_body(
        self, elem_or_num, resonance: Union[resonances.Resonance, str, list[resonances.Resonance], list[str]], name='asteroid'
    ):  # noqa: C901
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

    # Integration methods
    def run(self, progress=False):
        """Run the complete simulation."""
        self.times = np.linspace(0.0, self.config.tmax, self.config.Nout)
        self.body_manager.add_bodies_to_simulation(self.integration_engine.sim)
        self.integration_engine.run_integration(self.bodies, self.times, progress)
        self.identify_librations()
        self.data_manager.save_data(self.bodies, self.times, self)

    def identify_librations(self):
        """Identify librations for all bodies."""
        for body in self.bodies:
            try:
                if self.config.secular_angle_mode == 'proper':
                    self._rebuild_proper_secular_angles(body)
                resonances.libration.body(self, body)
            except Exception as e:
                resonances.logger.error(f"Error identifying librations for {body.name}: {e}")
                raise

    def _rebuild_proper_secular_angles(self, body: resonances.Body):
        for secular in body.secular_resonances:
            if not isinstance(secular, SecularResonance):
                continue
            try:
                existing = body.secular_angles.get(secular.to_s())
                if existing is not None:
                    body.secular_angles_osculating[secular.to_s()] = existing.copy()
                proper_angle = build_proper_angle_series(
                    times=self.times,
                    body=body,
                    resonance=secular,
                    existing_angle=existing,
                    cutoff_period_years=700_000.0,
                )
                body.secular_angles_proper[secular.to_s()] = proper_angle
                body.secular_angles[secular.to_s()] = proper_angle
            except Exception as exc:
                resonances.logger.warning(f"Failed to build proper secular angle for {body.name} / {secular.to_s()}: {exc}")
