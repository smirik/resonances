import resonances
import pandas as pd
from pathlib import Path


class Matrix:

    catalog_file = ''
    matrix = None
    planets = None

    @classmethod
    def dump(cls):
        if cls.matrix is None:
            cls.build()

        cls.matrix.to_csv(resonances.config.get(cls.catalog_file))

    @classmethod
    def load(cls, reload=False):
        catalog_file = Path(resonances.config.get(cls.catalog_file))
        if (not catalog_file.exists()) or (reload):
            cls.dump()
        else:
            catalog = pd.read_csv(resonances.config.get(cls.catalog_file))
            cls.matrix = catalog
