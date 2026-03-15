from resonances.config import config
from resonances.resonance.factory import create_mmr
import pandas as pd
from pathlib import Path
import os


class Matrix:
    catalog_file = ''
    matrix = None
    planets = None

    # Subclasses define which columns hold planet names for filtering
    planet_columns = []

    @classmethod
    def dump(cls):
        if cls.matrix is None:
            cls.build()

        cls.matrix.to_csv(cls.catalog_full_filename())

    @classmethod
    def catalog_full_filename(cls) -> str:
        """
        If the config value is an absolute path (starts with '/'), use it as is.
        Otherwise, interpret it relative to the current working directory.
        """
        filename = config.get(cls.catalog_file)
        path_obj = Path(filename)

        # If it's not already absolute, prepend current working directory
        if not path_obj.is_absolute():
            path_obj = Path(os.getcwd()) / path_obj

        return str(path_obj)

    @classmethod
    def load(cls, reload=False):
        catalog_file = Path(cls.catalog_full_filename())
        if (not catalog_file.exists()) or (reload):
            cls.dump()
        else:
            catalog = pd.read_csv(catalog_file)
            cls.matrix = catalog

    @classmethod
    def find_resonances(cls, a, sigma=0.1, planets=None):
        """Find resonances near semi-major axis `a` within `sigma` AU."""
        if cls.matrix is None:
            cls.load()

        df = cls.matrix[(cls.matrix['a'] >= (a - sigma)) & (cls.matrix['a'] <= (a + sigma))]

        if isinstance(planets, list) and cls.planet_columns:
            for col in cls.planet_columns:
                df = df[df[col].isin(planets)]

        return [create_mmr(mmr) for mmr in df['mmr'].tolist()]
