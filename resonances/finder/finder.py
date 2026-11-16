from typing import List, Union

import astdys

from resonances.data.util import convert_input_to_list
from resonances.horizons import get_body_keplerian_elements
from resonances.logger import logger
from resonances.mmr.mmr import MMR
from resonances.mmr.mmr_finder import candidate_mmrs
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
    asteroids,
    resonance: ResonanceType,
    name: str = None,
    **kwargs,
) -> Union[Simulation, List[Simulation]]:
    """
    Universal check function for MMR, secular, and Lidov-Kozai resonances.

    Parameters:
    -----------
    asteroids : Union[int, str, List[Union[int, str]], DataFrame]
        Asteroid ID(s) to check, or a selection (see find)
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
    _verify_secular_parameters(kwargs)

    sim = Simulation(name=name or "resonance_find", secular_angle_mode=secular_angle_mode, **kwargs)
    sim.create_solar_system()
    for asteroid in asteroids:
        sim.add_body(asteroid, mmr_resonances + secular_resonances + lidov_kozai_resonances, name=f"{asteroid}")
    return sim


def find(  # noqa: C901
    asteroids,
    planets=None,
    name: str = None,
    sigma2: float = 0.1,
    sigma3: float = 0.02,
    sigma_retro: float = 0.5,
    formulas: Union[str, List[str]] = None,
    type: str | List[str] = None,
    **kwargs,
) -> Union[Simulation, List[Simulation]]:
    """
    Universal find function for both MMR and secular resonances.

    This function determines search type based on the parameter "type":

    Parameters:
    -----------
    asteroids : Union[int, str, List[Union[int, str]], DataFrame]
        Asteroid ID(s) to search resonances for, or a selection: a DataFrame of
        resonances.mpcorb.query (for source='mpc') or of astdys (its 'name' column)
    planets : optional
        Planets to consider for MMR search (implies MMR search)
    name : str, optional
        Name for the simulation(s)
    sigma2 : float, default=0.1
        Width parameter for two-body MMR search
    sigma3 : float, default=0.02
        Width parameter for three-body MMR search
    sigma_retro : float, default=0.5
        Width for the retrograde two-body MMR search. A body with i > 90 deg gets only
        retrograde MMRs (all k, e.g. 1J-3r0..1J-3r2), any other body only prograde ones
    formulas : Union[str, List[str]], optional
        Secular resonance formulas to find (implies secular search)
    type:
        str or List[str], optional
            Type(s) of resonances to search for: 'mmr', 'secular', 'lidov_kozai'
    **kwargs
        Additional parameters passed to Simulation constructor. The elements are taken at its
        `date` (from Horizons) or at the AstDyS/MPC catalog epoch, which must then equal `date`
        (with source='mpc' MPC rows at another epoch are integrated to it, with a warning, unless
        MPC_EPOCH_POLICY=strict). Bodies not in the catalog, or whose MPC elements are refused at `date`,
        are left out with a warning.

    Returns:
    --------
    Union[resonances.Simulation, List[resonances.Simulation]]
        Single simulation if single type search,
        List of simulations if both types searched
    """

    type_list = type if isinstance(type, list) else ([type] if type is not None else [])
    shouldSearchMMR = type is None or 'mmr' in type_list
    shouldSearchSecular = type is None or ('secular' in type_list) or (formulas is not None)
    shouldSearchLidovKozai = type is None or 'lidov_kozai' in type_list or 'lk' in type_list or 'lkr' in type_list

    asteroids = convert_input_to_list(asteroids)
    formulas = convert_input_to_list(formulas) if formulas is not None else None
    resonances_dict = {}
    elems = {}
    refused = {}  # MPC rows at another epoch than the date: left out with their reason, as the missing ones

    secular_angle_mode = kwargs.pop('secular_angle_mode', 'proper' if shouldSearchSecular else "osculating")
    _verify_secular_parameters(kwargs)
    sim = Simulation(name=name or "resonance_find", secular_angle_mode=secular_angle_mode, **kwargs)

    # Batch fetch all asteroid elements at once for better performance
    if sim.config.source == 'astdys':
        elems = astdys.search(asteroids)  # Returns dict: {name: elements_dict}
    elif sim.config.source == 'mpc':
        elems = sim.body_manager.mpc_elements(asteroids, refused=refused)  # {str(identifier): elements at the date}
        gaps = [abs(e['epoch'] - e['propagated_from_epoch']) for e in elems.values() if 'propagated_from_epoch' in e]
        if gaps:
            logger.warning(f"{len(gaps)} bodies propagated from other epochs to {sim.config.date}, max gap {max(gaps):.1f} d")
    else:
        # At the simulation date: the planets are built there
        for asteroid in asteroids:
            elems[str(asteroid)] = get_body_keplerian_elements(asteroid, date=sim.config.date)

    for asteroid in asteroids:
        asteroid_key = str(asteroid)
        elem = elems.get(asteroid_key)
        if asteroid_key in refused:
            logger.warning(f'Asteroid {asteroid} left out: {refused[asteroid_key]}')
            continue
        if elem is None:
            logger.warning(f'Asteroid {asteroid} not found in catalog')
            continue
        resonances_dict[asteroid_key] = []
        if shouldSearchMMR:
            mmrs = candidate_mmrs(elem, planets=planets, sigma2=sigma2, sigma3=sigma3, sigma_retro=sigma_retro)
            resonances_dict[asteroid_key] = mmrs
        if shouldSearchSecular:
            if formulas is None:
                finder = SecularResonanceFinder()
                secular_resonances = finder.find_secular_resonances(asteroid=asteroid)
                resonances_dict[asteroid_key].extend(secular_resonances.values())
            else:
                from resonances.resonance.factory import create_secular_resonance

                secular_resonances = [create_secular_resonance(formula) for formula in formulas]
                resonances_dict[asteroid_key].extend(secular_resonances)
        if shouldSearchLidovKozai:
            resonances_dict[asteroid_key].append(LidovKozaiResonance())

    sim.create_solar_system()

    # Clones (CLONES > 0): the clones are checked in the MMR candidates of the whole cloud (docs/clones.md); the nominal
    # keeps its own, so its rows and the auto output step planned from them are those of a run without clones
    clouds = {}
    if sim.config.clones > 0:
        from resonances.simulation import ensemble_run

        for asteroid_name in [key for key in elems if key in resonances_dict]:
            try:
                ensemble = ensemble_run.ensemble_of(sim, asteroid_name, sim.config.clones, asteroid_name)
            except (OSError, ValueError) as error:  # no covariance there (404), a weak MPC orbit: the body runs alone
                logger.warning(f'{asteroid_name}: no clones ({error})')
                continue
            _, clone_elements = ensemble_run.elements_at_date(ensemble, sim.config)
            extra = []
            if shouldSearchMMR:
                cloud = ensemble_run.cloud_candidates(
                    clone_elements, planets=planets, sigma2=sigma2, sigma3=sigma3, sigma_retro=sigma_retro
                )
                known = {r.to_s() for r in resonances_dict[asteroid_name]}
                extra = [r for r in cloud if r.to_s() not in known]
                if extra:
                    logger.info(f'{asteroid_name}: {len(extra)} MMR candidates of its clones only: {[r.to_s() for r in extra]}')
            clouds[asteroid_name] = (ensemble, clone_elements, extra)

    for asteroid_name, kepler_elements in elems.items():
        res_list = resonances_dict[asteroid_name]
        if len(res_list) == 0:
            logger.warning(f'No resonances found for an asteroid {asteroid_name}')
            if asteroid_name in clouds and clouds[asteroid_name][2]:  # its clones have candidates: they run without it
                ensemble, clone_elements, extra = clouds[asteroid_name]
                ensemble_run.attach(sim, ensemble, asteroid_name, extra, clone_elements, float('nan'))
            continue
        sim.add_body(kepler_elements, res_list, name=f"{asteroid_name}")
        if asteroid_name in clouds:
            ensemble, clone_elements, extra = clouds[asteroid_name]
            own = sim.bodies[-1].resonances()
            mmrs = [r for r in own if r.type == 'mmr'] + extra  # the clones: the nominal's rows and the cloud's extra MMRs
            offset = ensemble_run.offset_sigma(sim.bodies[-1].initial_data, clone_elements)
            ensemble_run.attach(sim, ensemble, asteroid_name, mmrs + [r for r in own if r.type != 'mmr'], clone_elements, offset)
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


def _verify_secular_parameters(params):
    if (
        (params.get("integration_years") and params.get("integration_years") < 100000)
        or params.get("tmax")
        and params.get("tmax") < 6283185
    ):
        logger.warning(
            "Integration time for secular resonance search is set to less than 100,000 years. " "This may lead to inaccurate results."
        )
