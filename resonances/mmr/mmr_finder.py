from typing import List
from resonances.mmr.mmr import MMR
from .three_body_matrix import ThreeBodyMatrix
from .two_body_matrix import TwoBodyMatrix


def find_mmrs(a: float, planets=None, sigma2=0.1, sigma3=0.02, sigma=None) -> List[MMR]:
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

    mmrs = ThreeBodyMatrix.find_resonances(a, planets=planets, sigma=sigma3)
    mmrs2 = TwoBodyMatrix.find_resonances(a, planets=planets, sigma=sigma2)
    mmrs = mmrs + mmrs2
    return mmrs
