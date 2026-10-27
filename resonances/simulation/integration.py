import os
import ssl
from contextlib import contextmanager
from pathlib import Path
from typing import List
import tqdm
import numpy as np

import rebound
from resonances.config import config as c
from resonances.logger import logger
from .config import SimulationConfig, check_integrator_with_moon
from resonances.body import Body
from resonances.data.const import SOLAR_SYSTEM_WITH_SUN, SOLAR_SYSTEM_WITH_MOON, HORIZONS_IDS_WITH_MOON, MOON


@contextmanager
def horizons_tls():
    """Let rebound's Horizons download verify against certifi's CA bundle, for the duration of the block.

    Python's default bundle may lack JPL's root (see docs/config.md, "Horizons and TLS"). rebound's `urlopen`
    gets a certifi context and is restored afterwards. A context rebound passes itself is kept. Nothing
    changes when SSL_CERT_FILE is set or certifi is not installed. Verification is never switched off.
    """
    import rebound.horizons as rh

    original = getattr(rh, 'urlopen', None)
    try:
        import certifi
    except ImportError:
        certifi = None
    if os.environ.get('SSL_CERT_FILE') or certifi is None or original is None:
        yield
        return

    certifi_context = ssl.create_default_context(cafile=certifi.where())

    def urlopen(url, *args, context=None, **kwargs):
        return original(url, *args, context=context or certifi_context, **kwargs)

    rh.urlopen = urlopen
    try:
        yield
    finally:
        rh.urlopen = original


# A fixed step must sample a body's orbit this many times. Measured with SABA(10,6,4) against ias15
# over 2 kyr (non-crossing orbits): 8 steps per orbit misplace the mean longitude by 0.003 deg,
# 3 steps by 39 deg. With the default dt = 1.0 (58 d) this is a < 1.36 AU.
MIN_STEPS_PER_ORBIT = 10


def is_fixed_step(integrator: str) -> bool:
    """Whether the integrator keeps sim.dt (symplectic and leapfrog) rather than adapting it (ias15, bs...)."""
    name = str(integrator).lower()
    return name.startswith('saba') or name in ('whfast', 'whfast512', 'leapfrog', 'janus', 'eos')


class IntegrationEngine:
    """Handles the actual numerical integration logic."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.sim = None
        # The planets in particle order (index 0 is the Sun); resonant angles use these indices.
        self.planets = SOLAR_SYSTEM_WITH_SUN
        # Every massive particle in order: the planets, then the Moon when it is a separate body.
        self.massive_bodies = list(SOLAR_SYSTEM_WITH_SUN)

        self.planets_without_sun = [p for p in self.planets if p != 'Sun']
        # The kept planets' series ({planet: {column: array}}), filled by run_integration (or restore)
        self.planets_data = {}

    def create_solar_system(self, force=False):
        """Create or load the Solar System REBOUND simulation.

        Sun, Mercury..Neptune and Pluto; Earth is the Earth-Moon barycentre with the Earth+Moon mass.
        With `config.solar_system_moon` this hands over to `create_solar_system_with_moon`.
        """
        if self.config.solar_system_moon:
            self.create_solar_system_with_moon(force)
            return
        self._create_or_load(SOLAR_SYSTEM_WITH_SUN, {}, self._solar_system_filename(), force)

    def create_solar_system_with_moon(self, force=False):
        """Create or load the Solar System with a separate Moon.

        Sun, Mercury, Venus, Earth = the geocentre (Horizons 399, hash 'Earth'), Mars..Neptune, Pluto,
        then the Moon (Horizons 301, hash 'Moon') last, so the planets keep their indices. Earth and Moon
        together have the mass and the centre of mass of the barycentre that `create_solar_system` uses.
        Only ias15 is accepted (`check_integrator_with_moon`). A direct call also turns
        `config.solar_system_moon` on, so batch workers, simulation.json and the resume hash describe the
        system that was built.
        """
        check_integrator_with_moon(self.config.integrator)
        self.config.solar_system_moon = True
        self._create_or_load(SOLAR_SYSTEM_WITH_MOON, HORIZONS_IDS_WITH_MOON, self._solar_system_filename(moon=True), force)

    def _create_or_load(self, names, horizons_ids, filename, force):
        """Load `filename` if it exists, else fetch `names` from Horizons at config.date and cache them there."""
        solar_file = Path(filename)
        self.massive_bodies = list(names)

        if solar_file.exists() and not force:
            logger.info(f"Loading solar system from cache: {solar_file}")
            self.sim = rebound.Simulation(str(solar_file))
            return

        self.sim = rebound.Simulation()
        logger.info(f"Creating new solar system simulation. Date = {self.config.date.isoformat()}")
        with horizons_tls():
            for name in names:
                self.sim.add(horizons_ids.get(name, name), date=self.config.date, hash=name)
        self.sim.save_to_file(str(solar_file))

    def _solar_system_filename(self, moon=False) -> str:
        """Generate filename for solar system cache: solar-<timestamp>.bin, or solar-moon-<timestamp>.bin."""
        timestamp = int(self.config.date.timestamp())
        catalog_file = f"{os.getcwd()}/{c.get('SOLAR_SYSTEM_FILE')}"
        suffix = f'-moon-{timestamp}.bin' if moon else f'-{timestamp}.bin'
        return catalog_file.replace('.bin', suffix)

    def setup_integrator(self, N_active=None):
        """Setup the numerical integrator.

        N_active defaults to the number of massive bodies of the solar system built here (10, or 11 with
        the Moon); everything added after them is a test particle.
        """
        if MOON in self.massive_bodies:
            check_integrator_with_moon(self.config.integrator)
        self.sim.integrator = self.config.integrator
        self.sim.dt = self.config.dt
        self.sim.N_active = len(self.massive_bodies) if N_active is None else N_active

        # Unset safe_mode: WHFast 0 (its corrector would otherwise run at every step), SABA 1. Both are what
        # earlier versions used, so their runs are reproduced bit for bit; 0 and 1 differ at round-off.
        safe_mode = self.config.integration_safe_mode
        if 'whfast' == self.config.integrator.lower():
            self.sim.ri_whfast.safe_mode = 0 if safe_mode is None else safe_mode
            if self.config.integration_corrector is not None:
                self.sim.ri_whfast.corrector = self.config.integration_corrector
        elif 'SABA' in self.config.integrator:
            self.sim.ri_saba.safe_mode = 1 if safe_mode is None else safe_mode

        self.sim.move_to_com()

    def run_integration(self, bodies: List[Body], times, progress=False, planets=None):
        """Run the numerical integration.

        `planets` names the planets whose elements are kept in `planets_data` (`Simulation.planets_needed`); None
        keeps all of them unless `save_planets` is 'none'. Only these are held in memory and written.
        """
        for body in bodies:
            body.setup_vars_for_simulation(times)

        self.setup_integrator()
        self.warn_unresolved_orbits(bodies)
        ps = self.sim.particles
        n = len(times)

        # Pre-allocate planets data arrays (avoids per-timestep dict allocation)
        if planets is None:
            planets = self.planets_without_sun if self.config.save_planets else []
        kept = [planet for planet in self.planets_without_sun if planet in set(planets)]
        save_planets = bool(kept)
        self.planets_data = {}
        if save_planets:
            self._init_planets_arrays(n, kept)
            # orbits[i] is particle i + 1: planets_without_sun is the particle order after the Sun
            planet_arrays = [(self.planets_without_sun.index(p), self.planets_data[p]) for p in kept]
            planet_times = self.planets_data[kept[0]]['times']  # one array, shared by the kept planets

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
                planet_times[i] = time / two_pi
                self._store_planets(orbits, i, planet_arrays)

            for body, orbit_idx, mmr_refs, sec_refs, lk_refs in body_refs:
                self._update_body(body, orbits, i, orbit_idx, mmr_refs, sec_refs, lk_refs)

    def warn_unresolved_orbits(self, bodies: List[Body]):
        """Log a warning for every body whose period holds fewer than MIN_STEPS_PER_ORBIT fixed steps.

        Planet-crossing orbits are not detected: a fixed step does not resolve close encounters
        whatever the period, and those need ias15 (docs/config.md).
        """
        if not is_fixed_step(self.config.integrator) or not bodies:
            return
        orbits = self.sim.orbits(primary=self.sim.particles[0])
        for body in bodies:
            orbit = orbits[body.index_in_simulation - 1]
            if orbit.a <= 0:
                continue
            steps = abs(orbit.P / self.sim.dt)
            if steps < MIN_STEPS_PER_ORBIT:
                logger.warning(
                    f"{body.name}: {steps:.1f} steps of dt={self.sim.dt} per orbit (P = {orbit.P / (2 * np.pi):.3f} yr), "
                    f"fewer than {MIN_STEPS_PER_ORBIT}: {self.config.integrator} will misplace it. Use a smaller dt or ias15."
                )

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
    def _store_planets(orbits, i, planet_arrays):
        """Store planet orbital data into pre-allocated arrays ((orbit index, arrays) per kept planet)."""
        for pi, parr in planet_arrays:
            orbit = orbits[pi]
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

    def _init_planets_arrays(self, n, planets=None):
        """Pre-allocate numpy arrays for the kept planets (all by default); they share one `times` array."""
        fields = ('a', 'e', 'inc', 'Omega', 'omega', 'M', 'l', 'varpi')
        times = np.empty(n)
        for planet in self.planets_without_sun if planets is None else planets:
            self.planets_data[planet] = {'times': times, **{f: np.empty(n) for f in fields}}
