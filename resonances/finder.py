import resonances
import astdys
from typing import Union, List
from datetime import datetime

import resonances.horizons


def convert_input_to_list(asteroids: Union[int, str, List[Union[int, str]]]) -> List[str]:
    if isinstance(asteroids, str) or isinstance(asteroids, int):
        asteroids = [asteroids]
    elif asteroids is None:
        asteroids = []
    return asteroids


def find(
    asteroids: Union[int, str, List[Union[int, str]]], planets=None, name: str = None, sigma2: float = 0.1, sigma3: float = 0.02
) -> resonances.Simulation:
    now = datetime.now()
    sim = resonances.Simulation(name=name)
    sim.create_solar_system()

    asteroids = convert_input_to_list(asteroids)

    for asteroid in asteroids:
        elem = resonances.horizons.get_body_keplerian_elements(asteroid, sim=sim.sim, date=now)
        mmrs = find_resonances(elem['a'], planets=planets, sigma2=sigma2, sigma3=sigma3)
        if len(mmrs) > 0:
            sim.add_body(elem, mmrs, f"{asteroid}")
            resonances.logger.info(
                'Adding a possible resonance for an asteroid {} - {}'.format(asteroid, ', '.join(map(str, elem.values())))
            )
        else:
            resonances.logger.warning('No resonances found for an asteroid {}'.format(asteroid))

    return sim


def check(asteroids: Union[int, str, List[Union[int, str]]], mmr: Union[resonances.MMR, str]) -> resonances.Simulation:
    if isinstance(mmr, str):
        mmr = resonances.create_mmr(mmr)

    sim = resonances.Simulation()
    sim.create_solar_system()

    asteroids = convert_input_to_list(asteroids)

    for asteroid in asteroids:
        sim.add_body(asteroid, mmr, name=f"{asteroid}")
        resonances.logger.info('Adding a possible resonance for an asteroid {} - {}'.format(asteroid, mmr.to_s()))

    return sim


def find_asteroids_in_mmr(
    mmr: Union[resonances.MMR, str],
    sigma=0.1,
    per_iteration: int = 500,
    name: str = None,
):  # pragma: no cover
    if isinstance(mmr, str):
        mmr = resonances.create_mmr(mmr)

    df = astdys.search_by_axis(mmr.resonant_axis, sigma=sigma)
    numbers = df.index.astype(str).tolist()
    chunks = [numbers[i : i + per_iteration] for i in range(0, len(numbers), per_iteration)]

    num_chunks = len(chunks)
    data = []
    for i, chunk in enumerate(chunks):
        sim = resonances.Simulation(name=name, source='astdys', date=astdys.catalog_time)
        sim.create_solar_system()

        resonances.logger.info(f"Iteration {i+1}/{num_chunks}: Going to process a chunk of {len(chunk)} asteroids.")
        for asteroid in chunk:
            sim.add_body(df.loc[asteroid].to_dict(), mmr, f"{asteroid}")
        sim.run()
        data.append(sim.get_simulation_summary())

    return data


def find_resonances(a: float, planets=None, sigma2=0.1, sigma3=0.02, sigma=None) -> List[resonances.MMR]:
    """Find Two and Three-Body Mean Motion Resonances (MMR) for a given semi-major axis.
    This function identifies both two-body and three-body mean motion resonances
    near the specified semi-major axis value. If a single sigma value is provided,
    it overrides both sigma2 and sigma3 parameters.
    Parameters
    ----------
    a : float
        Semi-major axis value to search for resonances around
    planets : List[Planet], optional
        List of planets to consider for resonance search. If None, uses default planets
    sigma2 : float, default=0.1
        Width parameter for two-body resonance search. Ignored if sigma is provided
    sigma3 : float, default=0.02
        Width parameter for three-body resonance search. Ignored if sigma is provided
    sigma : float, optional
        If provided, overrides both sigma2 and sigma3 with this single width parameter
    Returns
    -------
    List[resonances.MMR]
        Combined list of found two-body and three-body mean motion resonances
    Notes
    -----
    The function uses ThreeBodyMatrix and TwoBodyMatrix classes to identify resonances,
    combining their results into a single list.
    """

    if sigma is not None:
        sigma2 = sigma
        sigma3 = sigma

    mmrs = resonances.ThreeBodyMatrix.find_resonances(a, planets=planets, sigma=sigma3)
    mmrs2 = resonances.TwoBodyMatrix.find_resonances(a, planets=planets, sigma=sigma2)
    mmrs = mmrs + mmrs2
    return mmrs
