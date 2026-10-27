from typing import List, Union

import numpy as np

from .three_body import ThreeBody
from .mmr import MMR
from .two_body import TwoBody
from .three_body_matrix import ThreeBodyMatrix
from .two_body_matrix import TwoBodyMatrix
from .retrograde_two_body import RetrogradeTwoBody
from .retrograde_two_body_matrix import RetrogradeTwoBodyMatrix


def find_mmrs(
    a: float,
    planets=None,
    sigma2=0.1,
    sigma3=0.02,
    sigma=None,
    mmr_type: Union[str, None] = None,
    retrograde: bool = False,
    sigma_retro: float = 0.5,
) -> List[MMR]:
    """Find Two and Three-Body Mean Motion Resonances (MMR) for a given semi-major axis.
    This function identifies both two-body and three-body mean motion resonances
    near the specified semi-major axis value. If a single sigma value is provided,
    it overrides sigma2, sigma3 and sigma_retro.
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
        If provided, overrides sigma2, sigma3 and sigma_retro with this single width parameter
    mmr_type : str, optional
        'two-body', 'three-body' or 'two-body-retrograde': only that kind
    retrograde : bool, default=False
        For a body with i > 90 deg: only the retrograde two-body resonances (RetrogradeTwoBodyMatrix,
        all k), and no prograde ones. Also selected by mmr_type='two-body-retrograde'.
        ValueError with mmr_type='two-body' or 'three-body'
    sigma_retro : float, default=0.5
        Width for the retrograde search, the +-0.5 au of Li et al. (2019). Ignored if sigma is provided
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
        sigma_retro = sigma

    if retrograde and mmr_type not in (None, RetrogradeTwoBody.mmr_type):
        raise ValueError(f"retrograde=True gives only retrograde resonances, not mmr_type='{mmr_type}'.")

    if retrograde or mmr_type == RetrogradeTwoBody.mmr_type:
        return RetrogradeTwoBodyMatrix.find_resonances(a, planets=planets, sigma=sigma_retro)

    mmrs, mmrs2 = [], []

    if (mmr_type is None) or (mmr_type == TwoBody.mmr_type):
        mmrs2 = TwoBodyMatrix.find_resonances(a, planets=planets, sigma=sigma2)

    if (mmr_type is None) or (mmr_type == ThreeBody.mmr_type):
        mmrs = ThreeBodyMatrix.find_resonances(a, planets=planets, sigma=sigma3)

    mmrs = mmrs + mmrs2
    return mmrs


def candidate_mmrs(elem: dict, planets=None, sigma2=0.1, sigma3=0.02, sigma_retro=0.5) -> List[MMR]:
    """MMR candidates for a body by its elements: retrograde ones only for i > 90 deg
    (`elem['inc']` in radians), prograde two- and three-body ones only otherwise.

    The 90 deg threshold is strict: a polar orbit just below it is prograde, as in Li et al. (2019).
    """
    retrograde = float(elem['inc']) > np.pi / 2
    return find_mmrs(elem['a'], planets=planets, sigma2=sigma2, sigma3=sigma3, retrograde=retrograde, sigma_retro=sigma_retro)
