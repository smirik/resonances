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
from resonances.data.const import SOLAR_SYSTEM_WITH_SUN


class IntegrationEngine:
    """Handles the actual numerical integration logic."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.sim = None
        self.planets = SOLAR_SYSTEM_WITH_SUN

        self.planets_without_sun = [p for p in self.planets if p != 'Sun']
        self.planets_data = {planet: {} for planet in self.planets_without_sun}

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
        for body in bodies:
            body.setup_vars_for_simulation(times)

        self.setup_integrator()
        ps = self.sim.particles
        n = len(times)

        # Pre-allocate planets data arrays (avoids per-timestep dict allocation)
        save_planets = self.config.save_planets
        if save_planets:
            self._init_planets_arrays(n)
            planet_arrays = [self.planets_data[p] for p in self.planets_without_sun]

        # Pre-resolve per-body array references to avoid repeated
        # dict lookups and to_s() calls in the inner loop
        body_refs = self._build_body_refs(bodies)

        iterations = enumerate(times)
        if progress:
            iterations = tqdm.tqdm(iterations, total=n)

        two_pi = 2 * np.pi

        for i, time in iterations:
            self.sim.integrate(time)
            orbits = self.sim.orbits(primary=ps[0])

            if save_planets:
                self._store_planets(orbits, i, time / two_pi, planet_arrays)

            for body, orbit_idx, mmr_refs, sec_refs, lk_refs in body_refs:
                self._update_body(body, orbits, i, orbit_idx, mmr_refs, sec_refs, lk_refs)

    @staticmethod
    def _build_body_refs(bodies):
        """Build pre-resolved array references for each body's resonances."""
        refs = []
        for body in bodies:
            orbit_idx = body.index_in_simulation - 1
            mmr_refs = [(body.angles_unwrapped[mmr.to_s()], mmr, mmr.index_of_planets) for mmr in body.mmrs]
            sec_refs = [(body.angles_unwrapped[sec.to_s()], sec, sec.index_of_planets) for sec in body.secular_resonances]
            lk_refs = [(body.angles_unwrapped[lk.to_s()], lk) for lk in body.lidov_kozai_resonances]
            refs.append((body, orbit_idx, mmr_refs, sec_refs, lk_refs))
        return refs

    @staticmethod
    def _store_planets(orbits, i, t_yrs, planet_arrays):
        """Store planet orbital data into pre-allocated arrays."""
        for pi, parr in enumerate(planet_arrays):
            orbit = orbits[pi]
            parr['times'][i] = t_yrs
            parr['a'][i] = orbit.a
            parr['e'][i] = orbit.e
            parr['inc'][i] = orbit.inc
            parr['Omega'][i] = orbit.Omega
            parr['omega'][i] = orbit.omega
            parr['M'][i] = orbit.M
            parr['l'][i] = orbit.l
            parr['varpi'][i] = orbit.Omega + orbit.omega

    @staticmethod
    def _update_body(body, orbits, i, orbit_idx, mmr_refs, sec_refs, lk_refs):
        """Update body orbital data and resonant angles using pre-resolved refs."""
        orbit = orbits[orbit_idx]

        body.axis[i] = orbit.a
        body.ecc[i] = orbit.e
        body.inc[i] = orbit.inc
        body.Omega[i] = orbit.Omega
        body.omega[i] = orbit.omega
        body.M[i] = orbit.M
        body.longitude[i] = orbit.l
        body.varpi[i] = orbit.Omega + orbit.omega

        for arr, mmr, planet_indices in mmr_refs:
            planets = [orbits[idx - 1] for idx in planet_indices]
            arr[i] = mmr.calc_angle(orbit, planets)

        for arr, sec, planet_indices in sec_refs:
            planets = {idx: orbits[idx - 1] for idx in planet_indices}
            arr[i] = sec.calc_angle(orbit, planets)

        for arr, lk in lk_refs:
            arr[i] = lk.calc_angle(orbit, None)

    def _init_planets_arrays(self, n):
        """Pre-allocate numpy arrays for planet data."""
        fields = ('times', 'a', 'e', 'inc', 'Omega', 'omega', 'M', 'l', 'varpi')
        for planet in self.planets_without_sun:
            self.planets_data[planet] = {f: np.empty(n) for f in fields}
