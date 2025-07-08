import resonances
import numpy as np
from typing import Union, List

from resonances.data.util import convert_input_to_list
from resonances.matrix.secular_matrix import SecularMatrix


def check(
    asteroids: Union[int, str, List[Union[int, str]]],
    resonance: str | list[str] | resonances.SecularResonance | list[resonances.SecularResonance],
    name: str = None,
    integration_years: int = 1000000,
    **kwargs,
) -> resonances.Simulation:
    """
    Check asteroids for secular resonance using SABA integrator.

    Secular resonances operate on much longer timescales than mean motion resonances.
    For reliable detection, integration times of at least 1 million years are recommended.

    Parameters:
    -----------
    asteroids : Union[int, str, List[Union[int, str]]]
        Asteroid ID(s) to check
    resonance : str
        Secular resonance to check (e.g., 'nu6', 'nu16', 'g-g6', or any formula)
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

    libration_period_min = kwargs.pop('libration_period_min', 20000)
    libration_period_critical = kwargs.pop('libration_period_critical', integration_years * 0.2)
    periodogram_frequency_min = kwargs.pop('periodogram_frequency_min', 0.000001)
    periodogram_frequency_max = kwargs.pop('periodogram_frequency_max', 0.0002)

    sim = resonances.Simulation(
        name=name or "secular_check",
        tmax=int(integration_years * 2 * np.pi),
        libration_period_min=libration_period_min,
        libration_period_critical=libration_period_critical,
        periodogram_frequency_min=periodogram_frequency_min,
        periodogram_frequency_max=periodogram_frequency_max,
        **kwargs,
    )

    sim.create_solar_system()
    asteroids = convert_input_to_list(asteroids)

    for asteroid in asteroids:
        sim.add_body(asteroid, resonance, name=f"{asteroid}")
        resonances.logger.info('Adding asteroid {} for secular resonance {}'.format(asteroid, resonance))
    sim.config.Nout = kwargs.get('Nout', 10000)

    return sim


def find(
    asteroids: Union[int, str, List[Union[int, str]]],
    formulas: Union[str, List[str]] = None,
    order: int = None,
    name: str = None,
    integration_years: int = 1000000,
    **kwargs,
) -> resonances.Simulation:
    """
    Find secular resonances for asteroids using SecularMatrix.

    This function automatically identifies all secular resonances (or specified ones)
    for the given asteroids. It uses SecularMatrix.build() to get the resonances
    and creates a simulation with all of them.

    Parameters:
    -----------
    asteroids : Union[int, str, List[Union[int, str]]]
        Asteroid ID(s) to check for secular resonances
    formulas : Union[str, List[str]], optional
        Specific secular resonance formulas to check (e.g., ['g-g5', 'g-g6']).
        If None, will find all available resonances or those of specified order.
    order : int, optional
        Order of secular resonances to include (e.g., 2 for linear, 4 for nonlinear).
        Ignored if formulas is specified.
    name : str, optional
        Name for the simulation
    integration_years : int, default=1000000
        Integration time in years (minimum 1 Myr recommended for secular resonances)
    **kwargs
        Additional parameters passed to Simulation constructor (integrator, dt, etc.)

    Returns:
    --------
    resonances.Simulation
        Configured simulation ready to run with all found secular resonances
    """

    secular_resonances = SecularMatrix.build(formulas=formulas, order=order)

    if len(secular_resonances) == 0:
        resonances.logger.warning(f'No secular resonances found for formulas={formulas}, order={order}')
        return resonances.Simulation(name=name or "secular_find")

    libration_period_min = kwargs.pop('libration_period_min', 10000)
    libration_period_critical = kwargs.pop('libration_period_critical', integration_years * 0.2)

    # Remove tmax from kwargs if present to avoid conflict, integration_years takes precedence
    kwargs.pop('tmax', None)

    sim = resonances.Simulation(
        name=name or "secular_find",
        tmax=int(integration_years * 2 * np.pi),
        libration_period_min=libration_period_min,
        libration_period_critical=libration_period_critical,
        **kwargs,
    )
    sim.create_solar_system()

    asteroids = convert_input_to_list(asteroids)

    for asteroid in asteroids:
        sim.add_body(asteroid, secular_resonances, name=f"{asteroid}")
        resonances.logger.info(
            'Adding asteroid {} for {} secular resonances: {}'.format(
                asteroid,
                len(secular_resonances),
                ', '.join([res.to_s() for res in secular_resonances[:5]]) + ('...' if len(secular_resonances) > 5 else ''),
            )
        )

    sim.config.Nout = kwargs.get('Nout', sim.config.Nout)

    return sim
