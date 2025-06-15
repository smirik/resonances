import resonances
import numpy as np
from typing import Union, List

from resonances.data.util import convert_input_to_list


def check(
    asteroids: Union[int, str, List[Union[int, str]]],
    secular_resonance: str,
    name: str = None,
    integration_years: int = 1000000,
    integrator: str = 'SABA(10,6,4)',
    dt: float = 5.0,
    nout: int = 10000,
    oscillations_cutoff: float = 0.00005,
    plot: str = 'all',
    save: str = 'all',
) -> resonances.Simulation:
    """
    Check asteroids for secular resonance using SABA integrator.

    Secular resonances operate on much longer timescales than mean motion resonances.
    For reliable detection, integration times of at least 1 million years are recommended.

    Parameters:
    -----------
    asteroids : Union[int, str, List[Union[int, str]]]
        Asteroid ID(s) to check
    secular_resonance : str
        Secular resonance to check (e.g., 'nu6', 'nu16')
    name : str, optional
        Name for the simulation
    integration_years : int, default=1000000
        Integration time in years (minimum 1 Myr recommended for secular resonances)
    integrator : str, default='SABA(10,6,4)'
        Integrator to use
    dt : float, default=15.0
        Time step
    nout : int, default=10000
        Number of output points

    Returns:
    --------
    resonances.Simulation
        Configured simulation ready to run
    """

    # Create simulation with proper configuration for secular resonances
    sim = resonances.Simulation(
        name=name or f"secular_check_{secular_resonance}",
        tmax=int(integration_years * 2 * np.pi),
        integrator=integrator,
        dt=dt,
        save=save,
        plot=plot,
        oscillations_cutoff=oscillations_cutoff,
        libration_period_min=10000,
        libration_period_critical=integration_years * 0.2,
    )

    sim.create_solar_system()
    asteroids = convert_input_to_list(asteroids)

    # Add each asteroid with the secular resonance
    for asteroid in asteroids:
        sim.add_body(asteroid, secular_resonance, name=f"{asteroid}")
        resonances.logger.info('Adding asteroid {} for secular resonance {}'.format(asteroid, secular_resonance))

    sim.config.Nout = nout

    return sim


def analyze_secular_resonance(sim: resonances.Simulation, asteroid_index: int = 0) -> dict:
    """
    Analyze the results of a secular resonance simulation.

    Parameters:
    -----------
    sim : resonances.Simulation
        Completed simulation
    asteroid_index : int, default=0
        Index of asteroid to analyze (for multi-asteroid simulations)

    Returns:
    --------
    dict
        Analysis results including libration status
    """

    if not sim.bodies or len(sim.bodies) <= asteroid_index:
        raise ValueError(f"No body found at index {asteroid_index}")

    body = sim.bodies[asteroid_index]

    if not body.secular_resonances:
        return {'is_librating': False, 'error': 'No secular resonances found'}

    try:
        secular_res = body.secular_resonances[0]
        angles = body.secular_angles[secular_res.to_s()]
        # Calculate angle statistics
        angle_range = np.max(angles) - np.min(angles)
        angle_mean = np.mean(angles)
        # More sophisticated libration criterion:
        # - If angle range < 2π (360°), check if it's significantly less than full circulation
        # - Consider libration if angle range < 3π/2 (270°) to be more permissive
        is_librating = bool(angle_range < (3 * np.pi / 2))

        return {
            'is_librating': is_librating,
            'angle_range_deg': np.degrees(angle_range),
            'angle_mean_deg': np.degrees(angle_mean),
            'angle_range_rad': angle_range,
            'secular_resonance': secular_res.to_s(),
            'semi_major_axis_mean': np.mean(body.axis),
            'eccentricity_mean': np.mean(body.ecc),
        }

    except Exception as e:
        return {'is_librating': False, 'error': str(e)}
