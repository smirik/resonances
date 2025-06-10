import datetime
from typing import Union
import rebound
from rebound.units import units_convert_particle, hash_to_unit


def get_body_keplerian_elements(s, sim: rebound.Simulation, date: Union[str, datetime.datetime], G=1) -> dict:
    if isinstance(s, int):
        s = str(s) + ';'

    temp_sim = rebound.Simulation()
    # Use the same units as the main simulation
    temp_sim.units = sim.units
    temp_sim.add("Sun")
    temp_sim.add(s, date=date)
    p: rebound.Particle = temp_sim.particles[1]

    units_convert_particle(
        p,
        'km',
        's',
        'kg',
        hash_to_unit(sim.python_unit_l),
        hash_to_unit(sim.python_unit_t),
        hash_to_unit(sim.python_unit_m),
    )

    orbit = p.orbit(primary=sim.particles[0], G=G)
    elem = {
        'a': orbit.a,
        'e': orbit.e,
        'inc': orbit.inc,
        'Omega': orbit.Omega,
        'omega': orbit.omega,
        'M': orbit.M,
    }
    return elem
