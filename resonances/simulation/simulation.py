import numpy as np
from typing import List, Union

import resonances
from .config import SimulationConfig
from .body_manager import BodyManager
from .integration import IntegrationEngine
from .data_manager import DataManager


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
        self.Nout = abs(int(self.config.tmax / 100))

    @property
    def bodies(self):
        return self.body_manager.bodies

    def create_solar_system(self, force=False):
        """Create or load the Solar System simulation."""
        self.integration_engine.create_solar_system(force)

    def add_body(self, elem_or_num, resonance, name='asteroid'):
        """Add a celestial body with mixed resonance types."""
        # Parse mixed resonance types
        mmr_list = []
        secular_list = []

        if isinstance(resonance, list):
            for res in resonance:
                if isinstance(res, str):
                    if res.lower() in ['nu6', 'nu5', 'nu16']:
                        secular_list.append(res)
                    else:
                        mmr_list.append(res)
                elif isinstance(res, resonances.MMR):
                    mmr_list.append(res)
                elif isinstance(res, resonances.SecularResonance):
                    secular_list.append(res)
        else:
            if isinstance(resonance, str):
                if resonance.lower() in ['nu6', 'nu5', 'nu16']:
                    secular_list.append(resonance)
                else:
                    mmr_list.append(resonance)
            elif isinstance(resonance, resonances.MMR):
                mmr_list.append(resonance)
            elif isinstance(resonance, resonances.SecularResonance):
                secular_list.append(resonance)

        # Add body with appropriate resonances
        if mmr_list and secular_list:
            # Mixed resonances - create body with both types
            body = resonances.Body()
            elem = self.body_manager.get_body_elements(elem_or_num)

            body.initial_data = elem
            body.name = name
            body.mass = elem.get('mass', 0.0)

            # Setup MMRs
            body.mmrs = [resonances.create_mmr(mmr) if isinstance(mmr, str) else mmr for mmr in mmr_list]
            for mmr_elem in body.mmrs:
                mmr_elem.index_of_planets = self.body_manager.get_index_of_planets(mmr_elem.planets_names)

            # Setup secular resonances
            body.secular_resonances = [resonances.create_secular_resonance(s) if isinstance(s, str) else s for s in secular_list]
            for secular_elem in body.secular_resonances:
                secular_elem.index_of_planets = self.body_manager.get_index_of_planets(secular_elem.planets_names)

            self.body_manager.bodies.append(body)

        elif mmr_list:
            self.body_manager.add_body_with_mmr(elem_or_num, mmr_list if len(mmr_list) > 1 else mmr_list[0], name)
        elif secular_list:
            self.body_manager.add_body_with_secular(elem_or_num, secular_list if len(secular_list) > 1 else secular_list[0], name)
        else:
            raise ValueError("If input is a list, it should contain a string representation of MMRs, MMR objects, or coefficients.")

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
        # Setup times
        self.times = np.linspace(0.0, self.config.tmax, self.Nout)

        # Add bodies to simulation
        self.body_manager.add_bodies_to_simulation(self.integration_engine.sim)

        # Run integration
        self.integration_engine.run_integration(self.bodies, self.times, progress)

        # Analyze librations
        self.identify_librations()

        # Save data
        self.data_manager.save_data(self.bodies, self.times, self)

    def identify_librations(self):
        """Identify librations for all bodies."""
        for body in self.bodies:
            try:
                resonances.libration.body(self, body)
            except Exception as e:
                resonances.logger.error(f"Error identifying librations for {body.name}: {e}")
                raise
