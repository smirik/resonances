from __future__ import annotations

from typing import Dict, List
import pandas as pd
import astdys

from resonances.matrix.secular_resonances import load_planetary_frequencies
from resonances.resonance.secular import SecularResonance


class SecularResonanceFinder:
    """
    Evaluate a secular resonance divisor on synthetic proper elements
    and find candidate asteroids.
    """

    def __init__(
        self,
        resonance: SecularResonance,
        threshold: float = 0.4,  # arcsec / yr
        threshold_negative: float = None,
        threshold_positive: float = None,
    ):
        self.resonance = resonance
        self.threshold = threshold

        if (
            (threshold_negative is None)
            and (threshold_positive is not None)
            or (threshold_negative is not None)
            and (threshold_positive is None)
        ):
            raise ValueError("Both threshold_negative and threshold_positive must be provided together.")

        self.threshold_negative = threshold_negative
        self.threshold_positive = threshold_positive

        # Load planetary secular frequencies (arcsec/yr)
        self.planetary_freqs: Dict[str, float] = load_planetary_frequencies()

        # Ensure AstDyS is in synthetic proper elements mode
        astdys.set_type("synthetic")

    def _evaluate_divisor(self, proper_g: float, proper_s: float) -> float:
        """
        Evaluate the secular divisor numerically (arcsec/yr).
        """
        value = 0.0

        for term in self.resonance.formula.terms:
            if term.mode == "g":
                if term.index is None:
                    freq = proper_g
                else:
                    freq = self.planetary_freqs[f"g{term.index}"]

            elif term.mode == "s":
                if term.index is None:
                    freq = proper_s
                else:
                    freq = self.planetary_freqs[f"s{term.index}"]

            else:
                raise ValueError(f"Unsupported mode '{term.mode}'")

            value += term.coefficient * freq

        return value

    def find_candidates(self, limit=None, limit_candidates=None) -> pd.DataFrame:
        """
        Find asteroids satisfying |divisor| < threshold.

        Returns
        -------
        pandas.DataFrame
            DataFrame with AstDyS synthetic elements and an extra
            column 'resonance_value'.
        """
        records: List[dict] = []

        # Iterate over all numbered asteroids in AstDyS

        catalog = astdys.get_catalog()
        if limit:
            catalog = catalog.head(limit)
        count = 0
        for num, row in catalog.iterrows():
            asteroid_id = num

            # Required frequencies
            proper_g = row.get("g")
            proper_s = row.get("s")

            if proper_g is None or proper_s is None:
                continue

            divisor = self._evaluate_divisor(proper_g, proper_s)

            if (
                (self.threshold_negative is None)
                and (abs(divisor) < self.threshold)
                or (self.threshold_negative is not None)
                and (self.threshold_positive is not None)
                and (self.threshold_negative < divisor < self.threshold_positive)
            ):
                count += 1
                if limit_candidates and count > limit_candidates:
                    break
                newRow = {"name": asteroid_id, "resonance_value": divisor}
                newRow.update(dict(row))
                records.append(newRow)

        return pd.DataFrame(records)
