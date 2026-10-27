"""The unphysical-orbit rule shared by every analysis (`util.is_unphysical_orbit`, `util.check_chaos`) and the
rate of an angle by central differences (`util.calc_sigma_derivative`, the MMR diagnostic plots).

The TTO classifier that lived here is gone: MMR verdicts come from `analysis.mmr`, secular and Lidov-Kozai ones
from `analysis.secular` and `analysis.zlk`.
"""

from resonances.resonance.classify.util import calc_sigma_derivative, check_chaos, is_unphysical_orbit, merge_intervals

__all__ = ["calc_sigma_derivative", "check_chaos", "is_unphysical_orbit", "merge_intervals"]
