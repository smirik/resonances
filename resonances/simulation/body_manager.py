from typing import List, Union

import resonances
import astdys
import resonances.horizons
from .config import SimulationConfig


class BodyManager:
    """Manages addition and setup of celestial bodies."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.bodies: List[resonances.Body] = []
        self.planets = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']

    def get_index_of_planets(self, planets_names):
        """Get indices of planets by name."""
        return [self.planets.index(planet) for planet in planets_names]

    def get_body_elements(self, elem_or_num: Union[int, str, dict]) -> dict:
        """Get orbital elements for a body."""
        if isinstance(elem_or_num, (int, str)):
            if self.config.source == 'astdys':
                return astdys.search(elem_or_num)
            else:
                return resonances.horizons.get_body_keplerian_elements(elem_or_num, date=self.config.date)
        elif isinstance(elem_or_num, dict):
            return elem_or_num
        else:
            raise ValueError('You can add body only by its number or all orbital elements')

    def add_body(self, elem_or_num, resonance: Union[resonances.Resonance, str, list[resonances.Resonance], list[str]], name='asteroid'):
        body = resonances.Body()

        if isinstance(resonance, list):
            resonances_list = [resonances.create_resonance(res) for res in resonance]
        else:
            resonances_list = [resonances.create_resonance(resonance)]

        elem = self.get_body_elements(elem_or_num)

        body.initial_data = elem
        body.name = name

        mmrs_list = []
        secular_list = []

        for res in resonances_list:
            if res.type == 'mmr':
                mmrs_list.append(res)
            elif res.type == 'secular':
                secular_list.append(res)
        body.mmrs = mmrs_list
        body.secular_resonances = secular_list
        body.mass = elem.get('mass', 0.0)

        for mmr_elem in body.mmrs:
            mmr_elem.index_of_planets = self.get_index_of_planets(mmr_elem.planets_names)
        for secular_elem in body.secular_resonances:
            secular_elem.index_of_planets = self.get_index_of_planets(secular_elem.planets_names)

        self.bodies.append(body)

    def add_bodies_to_simulation(self, sim):
        """Add all bodies to the REBOUND simulation."""
        for body in self.bodies:
            self._add_body_to_simulation(body, sim)

    def _add_body_to_simulation(self, body: resonances.Body, sim):
        """Add a single body to the REBOUND simulation."""
        body.index_in_simulation = len(sim.particles)
        sim.add(
            m=body.mass,
            a=body.initial_data['a'],
            e=body.initial_data['e'],
            inc=body.initial_data['inc'],
            Omega=body.initial_data['Omega'],
            omega=body.initial_data['omega'],
            M=body.initial_data['M'],
            date=self.config.get_bodies_date(),
            primary=sim.particles[0],
        )
