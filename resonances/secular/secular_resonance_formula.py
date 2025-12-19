from dataclasses import dataclass
from typing import Optional
import re


class TermType:
    PROPER = 'proper'
    FORCED = 'forced'


@dataclass
class ResonanceTerm:
    mode: str  # 'g' or 's'
    coefficient: int
    index: Optional[int] = None  # None for proper (g, s)
    term_type: str = TermType.PROPER


class SecularResonanceFormula:
    def __init__(self, formula: str):
        self.formula = formula
        self.terms: list[ResonanceTerm] = []
        self.parse()

    def order(self) -> int:
        return sum(abs(t.coefficient) for t in self.terms)

    def _expand_parentheses(self, expr: str) -> str:
        """
        Expand expressions like 2(g-g6)+(s-s6) → 2g-2g6+s-s6
        """
        pattern = re.compile(r'([+-]?\d*)\(([^()]+)\)')

        while '(' in expr:
            expr = pattern.sub(self._expand_match, expr)

        return expr

    def _expand_match(self, match) -> str:
        factor_str, content = match.groups()
        factor = int(factor_str) if factor_str not in ("", "+", "-") else (-1 if factor_str == "-" else 1)

        expanded = []
        for term in re.finditer(r'([+-]?)(\d*)([gs])(\d*)', content):
            sign, coeff, mode, index = term.groups()

            sign = -1 if sign == '-' else 1
            coeff = int(coeff) if coeff else 1

            total_coeff = factor * sign * coeff
            expanded.append(f"{total_coeff:+d}{mode}{index}")

        return ''.join(expanded)

    def parse(self):
        expr = self._expand_parentheses(self.formula)

        tokens = re.finditer(r'([+-]?)(\d*)([gs])(\d*)', expr)

        for token in tokens:
            sign, coeff, mode, index = token.groups()

            sign = -1 if sign == '-' else 1
            coeff = int(coeff) if coeff else 1
            coefficient = sign * coeff

            if index:
                index = int(index)
                term_type = TermType.FORCED
            else:
                index = None
                term_type = TermType.PROPER

            self.terms.append(ResonanceTerm(mode=mode, coefficient=coefficient, index=index, term_type=term_type))
