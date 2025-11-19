import numpy as np

import resonances
from resonances.data.util import convert_input_to_list


def check(asteroids, resonance, name: str = None, integration_years: int = 200000, **kwargs) -> resonances.Simulation:
    """
    Dedicated helper for Lidov–Kozai resonances.

    Parameters
    ----------
    asteroids : Union[int, str, list]
        Asteroid identifier(s) compatible with the configured data source.
    resonance : Union[resonances.Resonance, str, list]
        Lidov–Kozai resonance object(s) or identifier(s).
    name : str, optional
        Simulation name.
    integration_years : int, optional
        Total integration time window (default: 200_000 years).
    kwargs :
        Extra parameters forwarded to the Simulation constructor (dt, integrator, etc.).
    """

    sim = resonances.Simulation(
        name=name or "lidov_kozai_check",
        tmax=int(integration_years * 2 * np.pi),
        **kwargs,
    )
    sim.create_solar_system()

    asteroids = convert_input_to_list(asteroids)
    for asteroid in asteroids:
        sim.add_body(asteroid, resonance, name=str(asteroid))

    return sim
