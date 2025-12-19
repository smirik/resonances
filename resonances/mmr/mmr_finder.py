import resonances
import astdys
from typing import Union, List
from datetime import datetime

from resonances.data.util import convert_input_to_list
import resonances.horizons


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
        sim = resonances.Simulation(name=name, source='astdys', date=resonances.datetime_from_string(astdys.catalog_time))
        sim.create_solar_system()

        resonances.logger.info(f"Iteration {i+1}/{num_chunks}: Going to process a chunk of {len(chunk)} asteroids.")
        for asteroid in chunk:
            sim.add_body(df.loc[asteroid].to_dict(), mmr, f"{asteroid}")
        sim.run()
        data.append(sim.get_simulation_summary())

    return data


def find_mmrs(a: float, planets=None, sigma2=0.1, sigma3=0.02, sigma=None) -> List[resonances.MMR]:
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
