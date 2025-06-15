from typing import List, Optional, Union
from resonances.matrix.secular_resonances import SECULAR_RESONANCES
from resonances.resonance.factory import create_secular_resonance
from resonances.resonance.secular import SecularResonance, GeneralSecularResonance


class SecularMatrix:
    """
    A class for building and managing secular resonances based on mathematical formulas.

    This class provides functionality to create secular resonances from mathematical
    expressions involving planetary frequencies (g, s) and constants (g5, g6, g7, g8, s5, s6, s7, s8).
    """

    @classmethod
    def _build_specific_formulas(cls, formulas: List[str]) -> List[SecularResonance]:
        """Build resonances from specific formulas."""
        resonances = []
        for formula in formulas:
            if formula in SECULAR_RESONANCES:
                resonances.append(create_secular_resonance(formula))
            else:
                # Try to parse as a custom formula
                try:
                    resonances.append(GeneralSecularResonance(formula=formula))
                except Exception as e:
                    print(f"Warning: Could not create resonance for formula '{formula}': {e}")
        return resonances

    @classmethod
    def _build_by_order(cls, order: int) -> List[SecularResonance]:
        """Build resonances by order."""
        resonances = []
        for formula, info in SECULAR_RESONANCES.items():
            if info['order'] == order:
                try:
                    resonances.append(create_secular_resonance(formula))
                except Exception as e:
                    print(f"Warning: Could not create resonance for formula '{formula}': {e}")
        return resonances

    @classmethod
    def _build_all_resonances(cls) -> List[SecularResonance]:
        """Build all available resonances."""
        resonances = []
        for formula in SECULAR_RESONANCES.keys():
            try:
                resonances.append(create_secular_resonance(formula))
            except Exception as e:
                print(f"Warning: Could not create resonance for formula '{formula}': {e}")
        return resonances

    @classmethod
    def build(cls, formulas: Optional[Union[str, List[str]]] = None, order: Optional[int] = None) -> List[SecularResonance]:
        """
        Build secular resonances from formulas or by order.

        Parameters
        ----------
        formulas : str or list of str, optional
            Specific formulas to build. If None, build all or by order.
        order : int, optional
            Build all resonances of a specific order. If None, build all.

        Returns
        -------
        list of SecularResonance
            List of secular resonance objects
        """
        if formulas is not None:
            if isinstance(formulas, str):
                formulas = [formulas]
            return cls._build_specific_formulas(formulas)
        elif order is not None:
            return cls._build_by_order(order)
        else:
            return cls._build_all_resonances()
