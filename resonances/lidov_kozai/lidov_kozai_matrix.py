from __future__ import annotations

from pathlib import Path
import pandas as pd

import astdys

from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiParameters


class LidovKozaiMatrix:
    """
    Build a catalog-wide table of Lidov–Kozai invariants (c1, c2, h_quad) for
    every asteroid available in the AstDyS osculating elements file.
    """

    CACHE_FILE = Path('cache/lidov_kozai_matrix.csv')

    @classmethod
    def build(cls, force: bool = False, catalog_type: str = 'osculating') -> pd.DataFrame:
        """
        Generate the matrix (or read from cache if available).

        Parameters
        ----------
        force : bool
            Rebuild even if the cache file already exists.
        catalog_type : str
            AstDyS catalog type (default: 'osculating').
        """

        astdys.set_type(catalog_type)

        if force or not cls._cache_is_current():
            df = cls._generate_dataframe()
            cls.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cls.CACHE_FILE, index=False)
            return df

        return pd.read_csv(cls.CACHE_FILE)

    @classmethod
    def _cache_is_current(cls) -> bool:
        """A cache written before the rename to Lidov's notation has 'c' and holds the
        Hamiltonian under 'c2': rebuild it rather than read it."""
        if not cls.CACHE_FILE.exists():
            return False
        return 'h_quad' in pd.read_csv(cls.CACHE_FILE, nrows=0).columns

    @classmethod
    def _generate_dataframe(cls) -> pd.DataFrame:
        """
        Build dataframe with c1, c2, h_quad for every asteroid in the catalog.
        """
        # Ensure catalog is loaded
        astdys.search(1)
        catalog_df = astdys.get_catalog().reset_index().copy()

        ecc = catalog_df['e'].to_numpy(dtype=float)
        inc = catalog_df['inc'].to_numpy(dtype=float)
        omega = catalog_df['omega'].to_numpy(dtype=float)

        c1, c2, h_quad = LidovKozaiParameters.evaluate(ecc, inc, omega)

        return pd.DataFrame(
            {
                'num': catalog_df['num'].astype(str).str.strip(),
                'a': catalog_df['a'].astype(float),
                'e': ecc,
                'inc': inc,
                'omega': omega,
                'c1': c1,
                'c2': c2,
                'h_quad': h_quad,
            }
        )
