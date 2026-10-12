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
        # An angle of planetary longitudes only (e.g. 'g5-g6') does not involve the body.
        if all(t.index is not None for t in self.formula.terms):
            raise ValueError(f"Invalid secular resonance {formula!r}: no proper term (g or s) of the body")

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

        return angle

    def to_s(self) -> str:
        return self.formula_str

    def to_short(self) -> str:
        return self.formula_str

    def order(self) -> int:
        """
        Formal order of the secular resonance:
        sum |coefficients|.
        """
        return self.formula.order()


# nu_j = g - g_j and nu_1j = s - s_j for planet j = 1..8 (no nu15: s5 is identically zero).
SECULAR_RESONANCE_ALIASES: dict[str, str] = {
    **{f'nu{j}': f'g-g{j}' for j in range(1, 9)},
    **{f'nu1{j}': f's-s{j}' for j in range(1, 9) if j != 5},
    'z1': 'g-g6+s-s6',
    'z2': '2g-2g6+s-s6',
    'z3': '3g-3g6+s-s6',
    'z4': '4g-4g6+s-s6',
    '2nu6-nu5': 'g-2g6+g5',
    '3nu6-2nu5': 'g-3g6+2g5',
}
