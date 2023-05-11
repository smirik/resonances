import resonances
import pandas as pd
from pathlib import Path
import os


class Matrix:
    catalog_file = ''
    matrix = None
    planets = None

    @classmethod
    def dump(cls):
        if cls.matrix is None:
            cls.build()

        cls.matrix.to_csv(cls.catalog_full_filename())

    @classmethod
    def catalog_full_filename(cls) -> str:
        catalog_file = Path(f"{os.getcwd()}/{resonances.config.get(cls.catalog_file)}")
        return catalog_file

    @classmethod
    def load(cls, reload=False):
        catalog_file = Path(cls.catalog_full_filename())
        if (not catalog_file.exists()) or (reload):
            cls.dump()
        else:
            catalog = pd.read_csv(catalog_file)
            cls.matrix = catalog
