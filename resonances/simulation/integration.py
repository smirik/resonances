import os
from pathlib import Path
from typing import List
import tqdm
import numpy as np

import rebound
from resonances.config import config as c
from resonances.logger import logger
from .config import SimulationConfig
from resonances.body import Body
from rebound import hash as h


class IntegrationEngine:
    """Handles the actual numerical integration logic."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.sim = None
        self.planets = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']

        self.hash_to_planets = {}

        self.planets_data = {}
        for planet in self.planets:
            planet_cuid = h(planet).value
            self.hash_to_planets[planet_cuid] = planet
            self.planets_data[planet] = []

    def create_solar_system(self, force=False):
        """Create or load the Solar System REBOUND simulation."""
        solar_file = Path(self._solar_system_filename())

        if solar_file.exists() and not force:
            logger.info(f"Loading solar system from cache: {solar_file}")
            self.sim = rebound.Simulation(str(solar_file))
        else:
            self.sim = rebound.Simulation()
            logger.info(f"Creating new solar system simulation. Date = {self.config.date.isoformat()}")
            for planet in self.planets:
                self.sim.add(planet, date=self.config.date, hash=planet)
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
            if self.config.integration_corrector is not None:
                self.sim.ri_whfast.corrector = self.config.integration_corrector
        elif 'SABA' in self.config.integrator:
            self.sim.ri_saba.safe_mode = self.config.integration_safe_mode

        self.sim.move_to_com()

    def run_integration(self, bodies: List[Body], times, progress=False):
        """Run the numerical integration."""
        # Setup bodies for simulation
        for body in bodies:
            body.setup_vars_for_simulation(times)

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

            if self.config.save_planets:
                self._store_planets(time, os)

            # Update body data
            for body in bodies:
                self._update_body_data(body, os, i)

    def _store_planets(self, time, os):
        planets_without_sun = self.planets.copy()
        planets_without_sun.remove('Sun')
        for i, planet in enumerate(planets_without_sun):
            planet_hash = h(planet).value
            self.planets_data[self.hash_to_planets[planet_hash]].append(
                {
                    'times': time / (2 * np.pi),
                    'a': os[i].a,
                    'e': os[i].e,
                    'inc': os[i].inc,
                    'Omega': os[i].Omega,
                    'omega': os[i].omega,
                    'M': os[i].M,
                    'l': os[i].l,
                    'varpi': os[i].Omega + os[i].omega,
                }
            )

    def _update_body_data(self, body: Body, orbits, time_index):
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
            body.angles_unwrapped[mmr.to_s()][time_index] = mmr.calc_angle(orbit, planets)

        # Calculate secular resonance angles
        for secular in body.secular_resonances:
            planets = {idx: orbits[idx - 1] for idx in secular.index_of_planets}
            body.angles_unwrapped[secular.to_s()][time_index] = secular.calc_angle(orbit, planets)

        # Calculate Lidov-Kozai resonant angle (argument of pericenter)
        for lidov in body.lidov_kozai_resonances:
            body.angles_unwrapped[lidov.to_s()][time_index] = lidov.calc_angle(orbit, None)
