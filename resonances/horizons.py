import rebound.horizons
from rebound.units import units_convert_particle, check_units, convert_G, hash_to_unit


def get_body_keplerian_elements(s: str, sim: rebound.Simulation, G=1) -> dict:
    p: rebound.Particle = rebound.horizons.getParticle(s)
    units_convert_particle(
        p,
        'km',
        's',
        'kg',
        hash_to_unit(sim.python_unit_l),
        hash_to_unit(sim.python_unit_t),
        hash_to_unit(sim.python_unit_m),
    )
    p = p.calculate_orbit(primary=sim.particles[0], G=G)
    elem = {
        'a': p.a,
        'e': p.e,
        'inc': p.inc,
        'Omega': p.Omega,
        'omega': p.omega,
        'M': p.M,
    }
    return elem
