from datetime import datetime
from typing import List, Union

from resonances.data.util import convert_input_to_list
from resonances.horizons import get_body_keplerian_elements
from resonances.logger import logger
from resonances.mmr.mmr import MMR
from resonances.mmr.mmr_finder import find_mmrs
from resonances.secular.secular_resonance_finder import SecularResonanceFinder
from resonances.secular.secular_resonance import SecularResonance
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance
from resonances.simulation.simulation import Simulation
from resonances.resonance.factory import detect_resonance_type


ResonanceType = Union[
    MMR,
    SecularResonance,
    LidovKozaiResonance,
    str,
    List[Union[MMR, SecularResonance, LidovKozaiResonance, str]],
]


def check(
    asteroids: Union[int, str, List[Union[int, str]]],
    resonance: ResonanceType,
    name: str = None,
    **kwargs,
) -> Union[Simulation, List[Simulation]]:
    """
    Universal check function for MMR, secular, and Lidov-Kozai resonances.

    Parameters:
    -----------
    asteroids : Union[int, str, List[Union[int, str]]]
        Asteroid ID(s) to check
    resonance : Union[MMR, SecularResonance, str, List[...]]
        Resonance(s) to check. Can be:
        - Single resonance (MMR, SecularResonance, or string)
        - List of resonances (mixed types allowed)
    name : str, optional
        Name for the simulation(s)
    **kwargs
        Additional parameters passed to Simulation constructor

    Returns:
    --------
    Union[resonances.Simulation, List[resonances.Simulation]]
        Single simulation if all resonances are same type,
        List of simulations if mixed types
    """

    asteroids = convert_input_to_list(asteroids)
    # Convert single resonance to list for uniform processing
    if not isinstance(resonance, list):
        resonance = [resonance]

    # Categorize resonances by type
    mmr_resonances, secular_resonances, lidov_kozai_resonances = _categorize_resonances(resonance)
    shouldSearchSecular = len(secular_resonances) > 0
    secular_angle_mode = "proper" if shouldSearchSecular else "osculating"

    params = setup_secular_parameters(kwargs, shouldSearchSecular)

    sim = Simulation(name=name or "resonance_find", secular_angle_mode=secular_angle_mode, **params, **kwargs)
    sim.create_solar_system()
    for asteroid in asteroids:
        sim.add_body(asteroid, mmr_resonances + secular_resonances + lidov_kozai_resonances, name=f"{asteroid}")
    return sim


def find(
    asteroids: Union[int, str, List[Union[int, str]]],
    planets=None,
    name: str = None,
    sigma2: float = 0.1,
    sigma3: float = 0.02,
    formulas: Union[str, List[str]] = None,
    type: str | List[str] = None,
    **kwargs,
) -> Union[Simulation, List[Simulation]]:
    """
    Universal find function for both MMR and secular resonances.

    This function determines search type based on the parameter "type":

    Parameters:
    -----------
    asteroids : Union[int, str, List[Union[int, str]]]
        Asteroid ID(s) to search resonances for
    planets : optional
        Planets to consider for MMR search (implies MMR search)
    name : str, optional
        Name for the simulation(s)
    sigma2 : float, default=0.1
        Width parameter for two-body MMR search
    sigma3 : float, default=0.02
        Width parameter for three-body MMR search
    formulas : Union[str, List[str]], optional
        Secular resonance formulas to find (implies secular search)
    type:
        str or List[str], optional
            Type(s) of resonances to search for: 'mmr', 'secular', 'lidov_kozai'
    **kwargs
        Additional parameters passed to Simulation constructor

    Returns:
    --------
    Union[resonances.Simulation, List[resonances.Simulation]]
        Single simulation if single type search,
        List of simulations if both types searched
    """

    shouldSearchMMR = type is None or 'mmr' in (type if isinstance(type, list) else [type])
    shouldSearchSecular = type is None or 'secular' in (type if isinstance(type, list) else [type])
    shouldSearchLidovKozai = (
        type is None
        or 'lidov_kozai' in (type if isinstance(type, list) else [type])
        or 'lk' in (type if isinstance(type, list) else [type])
        or 'lkr' in (type if isinstance(type, list) else [type])
    )

    asteroids = convert_input_to_list(asteroids)
    formulas = convert_input_to_list(formulas) if formulas is not None else None
    now = datetime.now()
    resonances_dict = {}
    elems = {}

    for asteroid in asteroids:
        elem = get_body_keplerian_elements(asteroid, date=now)
        elems[asteroid] = elem
        resonances_dict[asteroid] = []
        if shouldSearchMMR:
            mmrs = find_mmrs(elem['a'], planets=planets, sigma2=sigma2, sigma3=sigma3)
            resonances_dict[asteroid] = mmrs
        if shouldSearchSecular:
            if formulas is None:
                finder = SecularResonanceFinder()
                secular_resonances = finder.find_secular_resonances(asteroid=asteroid)
                resonances_dict[asteroid].extend(secular_resonances.values())
            else:
                secular_resonances = [SecularResonance(formula) for formula in formulas]
                resonances_dict[asteroid].extend(secular_resonances)
        if shouldSearchLidovKozai:
            resonances_dict[asteroid].append(LidovKozaiResonance())

    secular_angle_mode = kwargs.pop('secular_angle_mode', 'proper' if shouldSearchSecular else "osculating")
    params = setup_secular_parameters(kwargs, shouldSearchSecular)
    sim = Simulation(name=name or "resonance_find", secular_angle_mode=secular_angle_mode, **params, **kwargs)
    sim.create_solar_system()

    for asteroid_name, kepler_elements in elems.items():
        res_list = resonances_dict[asteroid_name]
        if len(res_list) == 0:
            logger.warning(f'No resonances found for an asteroid {asteroid_name}')
            continue
        sim.add_body(kepler_elements, res_list, name=f"{asteroid_name}")
        logger.info(
            'Adding a possible resonance for an asteroid {} - {}'.format(
                asteroid_name,
                [elem.to_s() for elem in res_list],
            )
        )

    return sim


def _categorize_resonances(resonance_list):
    """Helper function to categorize resonances by type."""
    mmr_resonances = []
    secular_resonances = []
    lidov_kozai_resonances = []

    for res in resonance_list:
        res_type = detect_resonance_type(res)
        if res_type == 'mmr':
            mmr_resonances.append(res)
        elif res_type == 'secular':
            secular_resonances.append(res)
        elif res_type == 'lidov_kozai':
            lidov_kozai_resonances.append(res)
        else:
            raise ValueError(f"Unknown resonance type: {res_type}")

    return mmr_resonances, secular_resonances, lidov_kozai_resonances


def setup_secular_parameters(params, shouldSearchSecular):
    if not shouldSearchSecular:
        return {}

    integration_years = params.pop('integration_years', 1000000)

    libration_period_min = params.pop('libration_period_min', 20000)
    libration_period_critical = params.pop('libration_period_critical', integration_years * 0.2)
    periodogram_frequency_min = params.pop('periodogram_frequency_min', 0.000001)
    periodogram_frequency_max = params.pop('periodogram_frequency_max', 0.0002)

    return {
        'integration_years': integration_years,
        'libration_period_min': libration_period_min,
        'libration_period_critical': libration_period_critical,
        'periodogram_frequency_min': periodogram_frequency_min,
        'periodogram_frequency_max': periodogram_frequency_max,
    }
