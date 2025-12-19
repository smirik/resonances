from __future__ import annotations

from typing import Dict, List
import pandas as pd
import astdys

from resonances.logger import logger
from resonances.data.const import PLANETARY_FREQUENCIES, SECULAR_FORMULAS
from resonances.secular.secular_resonance import SecularResonance


class SecularResonanceFinder:
    """
    Evaluate a secular resonance divisor on synthetic proper elements
    and find candidate asteroids.
    """

    def __init__(
        self,
        threshold: float = 0.4,  # arcsec / yr
        threshold_negative: float = -6.0,
        threshold_positive: float = 3.0,
    ):
        self.threshold = threshold
        self.resonance = None

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
        self.planetary_freqs: Dict[str, float] = PLANETARY_FREQUENCIES

        # Ensure AstDyS is in synthetic proper elements mode
        astdys.set_type("synthetic")

    def _evaluate_divisor(self, proper_g: float, proper_s: float, resonance: SecularResonance) -> float:
        """
        Evaluate the secular divisor numerically (arcsec/yr).
        """
        value = 0.0

        for term in resonance.formula.terms:
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

    def _matches_threshold(self, divisor: float) -> bool:
        if self.threshold_negative is None:
            return abs(divisor) < self.threshold
        return self.threshold_negative < divisor < self.threshold_positive

    def find_secular_resonances(
        self,
        asteroid: int | str | None = None,
        proper_freqs: Dict[str, float] | None = None,
    ) -> Dict[str, SecularResonance]:
        """
        Find all secular resonances satisfied by the provided asteroid or proper frequencies.

        Parameters
        ----------
        asteroid : str or int, optional
            Asteroid name(s) to query in the AstDyS synthetic catalog.
        proper_freqs : dict, optional
            Explicit proper frequencies with keys "g" and "s".

        Returns
        -------
        dict
            Mapping formula string -> SecularResonance instance.
        """
        if proper_freqs is not None:
            proper_g = proper_freqs.get("g", proper_freqs.get("proper_g"))
            proper_s = proper_freqs.get("s", proper_freqs.get("proper_s"))
            if proper_g is None or proper_s is None:
                raise ValueError("proper_freqs must contain 'g' and 's' values.")
        else:
            if asteroid is None:
                raise ValueError("Provide either asteroid or proper_freqs.")
            row = astdys.search(str(asteroid))
            if row is None:
                logger.warning(f"Asteroid {asteroid} not found in AstDyS catalog.")
                return {}
            proper_g = row.get("g")
            proper_s = row.get("s")
            if proper_g is None or proper_s is None:
                raise ValueError(f"Proper frequencies 'g' and 's' not found for the given asteroid {asteroid}.")

        results: Dict[str, SecularResonance] = {}
        for formula in SECULAR_FORMULAS:
            resonance = SecularResonance(formula)
            divisor = self._evaluate_divisor(proper_g, proper_s, resonance=resonance)
            if self._matches_threshold(divisor):
                results[formula] = resonance

        return results

    def find_secular_resonances_for_asteroids(
        self,
        asteroids: List[int | str],
    ) -> Dict[int | str, Dict[str, SecularResonance]]:
        """
        Find all secular resonances satisfied by the provided list of asteroids.

        Parameters
        ----------
        asteroids : list of str
            List of asteroid names to query in the AstDyS synthetic catalog.

        Returns
        -------
        dict
            Mapping asteroid name -> (mapping formula string -> SecularResonance instance).
        """
        all_results: Dict[int | str, Dict[str, SecularResonance]] = {}
        for asteroid in asteroids:
            results = self.find_secular_resonances(asteroid=str(asteroid))
            all_results[str(asteroid)] = results
        return all_results

    def find_candidates(self, resonance: SecularResonance, limit=None, limit_candidates=None) -> pd.DataFrame:
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

            divisor = self._evaluate_divisor(proper_g, proper_s, resonance)

            if self._matches_threshold(divisor):
                count += 1
                if limit_candidates and count > limit_candidates:
                    break
                newRow = {"name": asteroid_id, "resonance_value": divisor}
                newRow.update(dict(row))
                records.append(newRow)

        return pd.DataFrame(records)
