import resonances
from typing import Union, List

from resonances.data.util import convert_input_to_list
from . import mmr_finder, secular_finder


def _categorize_resonances(resonance_list):
    """Helper function to categorize resonances by type."""
    mmr_resonances = []
    secular_resonances = []

    for res in resonance_list:
        res_type = resonances.detect_resonance_type(res)
        if res_type == 'mmr':
            mmr_resonances.append(res)
        elif res_type == 'secular':
            secular_resonances.append(res)
        else:
            raise ValueError(f"Unknown resonance type: {res_type}")

    return mmr_resonances, secular_resonances


def _create_mmr_simulation(asteroids, mmr_resonances, name, **kwargs):
    """Helper function to create MMR simulation."""
    if len(mmr_resonances) == 1:
        mmr_sim = mmr_finder.check(asteroids=asteroids, resonance=mmr_resonances[0], name=name or "mmr_check", **kwargs)
    else:
        # For multiple MMRs, create simulation with first and add others
        mmr_sim = mmr_finder.check(asteroids=asteroids, resonance=mmr_resonances[0], name=name or "mmr_check", **kwargs)
        # Add additional MMRs to the same simulation
        for additional_mmr in mmr_resonances[1:]:
            asteroids_list = convert_input_to_list(asteroids)
            for asteroid in asteroids_list:
                mmr_sim.add_body(asteroid, additional_mmr, name=f"{asteroid}")

    return mmr_sim


ResonanceType = Union[resonances.MMR, resonances.SecularResonance, str, List[Union[resonances.MMR, resonances.SecularResonance, str]]]


def check(
    asteroids: Union[int, str, List[Union[int, str]]],
    resonance: ResonanceType,
    name: str = None,
    **kwargs,
) -> Union[resonances.Simulation, List[resonances.Simulation]]:
    """
    Universal check function for both MMR and secular resonances.

    This function automatically detects resonance types and routes to the
    appropriate finder (mmr_finder or secular_finder). If resonances are
    mixed types, it creates separate simulations for each type.

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

    # Convert single resonance to list for uniform processing
    if not isinstance(resonance, list):
        resonance = [resonance]

    # Categorize resonances by type
    mmr_resonances, secular_resonances = _categorize_resonances(resonance)
    simulations = []

    # Create MMR simulation if we have MMR resonances
    if mmr_resonances:
        mmr_sim = _create_mmr_simulation(asteroids, mmr_resonances, name, **kwargs)
        simulations.append(mmr_sim)

    # Create secular simulation if we have secular resonances
    if secular_resonances:
        res_to_use = secular_resonances[0] if len(secular_resonances) == 1 else secular_resonances
        secular_sim = secular_finder.check(
            asteroids=asteroids,
            resonance=res_to_use,
            name=name or "secular_check",
            **kwargs,
        )
        simulations.append(secular_sim)

    # Return single simulation if only one type, otherwise return list
    return simulations[0] if len(simulations) == 1 else simulations


def find(
    asteroids: Union[int, str, List[Union[int, str]]],
    planets=None,
    name: str = None,
    sigma2: float = 0.1,
    sigma3: float = 0.02,
    formulas: Union[str, List[str]] = None,
    order: int = None,
    integration_years: int = 1000000,
    **kwargs,
) -> Union[resonances.Simulation, List[resonances.Simulation]]:
    """
    Universal find function for both MMR and secular resonances.

    This function determines search type based on parameters:
    - If planets provided: search for MMRs
    - If formulas provided: search for secular resonances
    - If neither provided: search for both types

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
    order : int, optional
        Order of secular resonances to include
    integration_years : int, default=1000000
        Integration time for secular resonances
    **kwargs
        Additional parameters passed to Simulation constructor

    Returns:
    --------
    Union[resonances.Simulation, List[resonances.Simulation]]
        Single simulation if single type search,
        List of simulations if both types searched
    """

    simulations = []

    # Determine search strategy based on parameters
    search_secular = formulas is not None or order is not None
    search_mmr = planets is not None

    # If neither specified, search both
    if not search_secular and not search_mmr:
        search_secular = True
        search_mmr = True
    # If planets is specified, only search MMR (don't also search secular)
    elif search_mmr and not (formulas is not None or order is not None):
        search_secular = False

    # Search for MMRs if requested
    if search_mmr:
        mmr_sim = mmr_finder.find(asteroids=asteroids, planets=planets, name=name or "mmr_find", sigma2=sigma2, sigma3=sigma3, **kwargs)
        simulations.append(mmr_sim)

    # Search for secular resonances if requested
    if search_secular:
        secular_sim = secular_finder.find(
            asteroids=asteroids, formulas=formulas, order=order, name=name or "secular_find", integration_years=integration_years, **kwargs
        )
        simulations.append(secular_sim)

    # Return single simulation if only one type, otherwise return list
    return simulations[0] if len(simulations) == 1 else simulations


def find_asteroids_in_mmr(
    mmr: Union[resonances.MMR, str],
    sigma=0.1,
    per_iteration: int = 500,
    name: str = None,
):
    """
    Find asteroids in a specific MMR using AstDyS catalog.

    This function is specific to MMRs and directly uses mmr_finder.

    Parameters:
    -----------
    mmr : Union[resonances.MMR, str]
        Mean motion resonance to search for
    sigma : float, default=0.1
        Width parameter for search
    per_iteration : int, default=500
        Number of asteroids to process per iteration
    name : str, optional
        Name for the simulation

    Returns:
    --------
    List of simulation data
    """
    return mmr_finder.find_asteroids_in_mmr(mmr=mmr, sigma=sigma, per_iteration=per_iteration, name=name)


def find_mmrs(a: float, planets=None, sigma2=0.1, sigma3=0.02, sigma=None) -> List[resonances.MMR]:
    """
    Find MMRs for a given semi-major axis.

    This function is specific to MMRs and directly uses mmr_finder.

    Parameters:
    ----------
    a : float
        Semi-major axis value to search for resonances around
    planets : List[Planet], optional
        List of planets to consider for resonance search
    sigma2 : float, default=0.1
        Width parameter for two-body resonance search
    sigma3 : float, default=0.02
        Width parameter for three-body resonance search
    sigma : float, optional
        If provided, overrides both sigma2 and sigma3

    Returns:
    -------
    List[resonances.MMR]
        List of found mean motion resonances
    """
    return mmr_finder.find_mmrs(a=a, planets=planets, sigma2=sigma2, sigma3=sigma3, sigma=sigma)
