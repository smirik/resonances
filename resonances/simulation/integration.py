import os
from pathlib import Path
from typing import List
import tqdm

import rebound
import resonances
from resonances.config import config as c
from .config import SimulationConfig


class IntegrationEngine:
    """Handles the actual numerical integration logic."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.sim = None

    def create_solar_system(self, force=False):
        """Create or load the Solar System REBOUND simulation."""
        solar_file = Path(self._solar_system_filename())

        if solar_file.exists() and not force:
            self.sim = rebound.Simulation(str(solar_file))
        else:
            self.sim = rebound.Simulation()
            planets = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
            self.sim.add(planets, date=self.config.date)
            self.sim.save_to_file(str(solar_file))

    def _solar_system_filename(self) -> str:
        """Generate filename for solar system cache."""
        timestamp = int(self.config.date.timestamp())
        catalog_file = f"{os.getcwd()}/{c.get('SOLAR_SYSTEM_FILE')}"
        return catalog_file.replace('.bin', f'-{timestamp}.bin')

    def setup_integrator(self, N_active=10):
        """Setup the numerical integrator."""
        self.sim.integrator = self.config.integrator
        self.sim.dt = self.config.dt
        self.sim.N_active = N_active

        if 'whfast' == self.config.integrator.lower():
            self.sim.ri_whfast.safe_mode = 0
            if self.config.integrator_corrector is not None:
                self.sim.ri_whfast.corrector = self.config.integrator_corrector
        elif 'SABA' in self.config.integrator:
            self.sim.ri_saba.safe_mode = self.config.integrator_safe_mode

        self.sim.move_to_com()

    def run_integration(self, bodies: List[resonances.Body], times, progress=False):
        """Run the numerical integration."""
        # Setup bodies for simulation
        for body in bodies:
            body.setup_vars_for_simulation(len(times))

        # Setup integrator
        self.setup_integrator()

        # Get particles reference
        ps = self.sim.particles

        # Integration loop
        iterations = list(enumerate(times))
        if progress:
            iterations = tqdm.tqdm(iterations, total=len(iterations))

        for i, time in iterations:
            self.sim.integrate(time)
            os = self.sim.orbits(primary=ps[0])

            # Update body data
            for body in bodies:
                self._update_body_data(body, os, i)

    def _update_body_data(self, body: resonances.Body, orbits, time_index):
        """Update body orbital data and calculate resonant angles."""
        # Get orbital elements
        orbit = orbits[body.index_in_simulation - 1]  # -1 because Sun is not in orbits

        body.axis[time_index] = orbit.a
        body.ecc[time_index] = orbit.e
        body.inc[time_index] = orbit.inc
        body.Omega[time_index] = orbit.Omega
        body.omega[time_index] = orbit.omega
        body.M[time_index] = orbit.M
        body.longitude[time_index] = orbit.l
        body.varpi[time_index] = orbit.Omega + orbit.omega

        # Calculate MMR angles
        for mmr in body.mmrs:
            planets = [orbits[idx - 1] for idx in mmr.index_of_planets]
            body.angle(mmr)[time_index] = mmr.calc_angle(orbit, planets)

        # Calculate secular resonance angles
        for secular in body.secular_resonances:
            planets = [orbits[idx - 1] for idx in secular.index_of_planets]
            body.angle(secular)[time_index] = secular.calc_angle(orbit, planets)
