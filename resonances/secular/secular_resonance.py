import rebound

from resonances.resonance.resonance import Resonance
from resonances.secular.secular_resonance_formula import SecularResonanceFormula


class SecularResonance(Resonance):
    """
    Secular resonance defined by a linear combination of:
      - g terms: longitude of perihelion  (varpi = Omega + omega)
      - s terms: longitude of node        (Omega)
    """

    def __init__(self, formula: str):
        self.formula_str = formula
        self.formula = SecularResonanceFormula(formula)

        # Forced secular indices involved in this resonance (e.g. 5, 6, 7)
        self.index_of_planets = sorted({t.index for t in self.formula.terms if t.index is not None})

    @property
    def type(self) -> str:
        return "secular"

    def calc_angle(self, body, planets: dict[int, object]) -> float:
        """
        Calculate the secular resonant angle.

        Parameters
        ----------
        body : object
            Must provide Omega and omega.
        planets : dict[int, object]
            Mapping secular index -> orbit object.
            Example: {5: Jupiter_orbit, 6: Saturn_orbit}

        Returns
        -------
        float
            Resonant angle in [0, 2π)
        """
        angle = 0.0

        for term in self.formula.terms:
            if term.mode == "g":
                # longitude of perihelion
                if term.index is None:
                    theta = body.Omega + body.omega
                else:
                    theta = planets[term.index].Omega + planets[term.index].omega

            elif term.mode == "s":
                # longitude of ascending node
                if term.index is None:
                    theta = body.Omega
                else:
                    theta = planets[term.index].Omega

            else:
                raise ValueError(f"Unsupported mode '{term.mode}' in secular formula '{self.formula_str}'")

            angle += term.coefficient * theta

        return rebound.mod2pi(angle)

    def to_s(self) -> str:
        return self.formula_str

    def to_short(self) -> str:
        return self.formula_str

    def order(self) -> int:
        """
        Formal order of the secular resonance:
        sum |coefficients|.
        """
        return sum(abs(t.coefficient) for t in self.formula.terms)


SECULAR_RESONANCE_ALIASES: dict[str, str] = {
    'nu5': 'g-g5',
    'nu6': 'g-g6',
    'nu16': 's-s6',
    'z1': 'g-g6+s-s6',
    'z2': '2g-2g6+s-s6',
    'z3': '3g-3g6+s-s6',
    'z4': '4g-4g6+s-s6',
    '2nu6-nu5': 'g-2g6+g5',
    '3nu6-2nu5': 'g-3g6+2g5',
}
