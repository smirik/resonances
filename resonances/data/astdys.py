import pandas as pd
import numpy as np
from pathlib import Path
import os
import urllib.request

import resonances.config
import resonances.logger
from resonances.util import convert_mjd_to_date


class astdys:
    catalog = None

    @classmethod
    def search(cls, num):
        num = str(num)
        if cls.catalog is None:
            cls.load()

        if num in cls.catalog['num'].values:
            return cls.catalog.loc[cls.catalog['num'] == num].to_dict('records')[0]

        return None

    @classmethod
    def catalog_time(cls):
        if cls.catalog is None:
            cls.load()

        elems = cls.search(1)
        return convert_mjd_to_date(elems['epoch'])

    @classmethod
    def search_possible_resonant_asteroids(cls, mmr, sigma=0.02):
        if cls.catalog is None:
            cls.load()
        axis = mmr.resonant_axis
        df = cls.catalog[(cls.catalog['a'] >= axis - sigma) & (cls.catalog['a'] <= axis + sigma)]
        return df

    @classmethod
    def astdys_full_filename(cls) -> str:
        filename = f"{os.getcwd()}/{resonances.config.get('astdys.catalog')}"
        return filename

    @classmethod
    def catalog_full_filename(cls) -> str:
        filename = f"{os.getcwd()}/{resonances.config.get('catalog')}"
        return filename

    @classmethod
    def load(cls):
        filename = cls.catalog_full_filename()
        if cls.catalog is None:
            output_file = Path(filename)
            if not output_file.exists():
                cls.build()

        cls.catalog = pd.read_csv(filename)
        cls.catalog['num'] = cls.catalog['num'].astype(str)

    @classmethod
    def rebuild(cls):
        input_file = Path(cls.astdys_full_filename())
        if input_file.exists():
            input_file.unlink()
        cls.build()

    @classmethod
    def build(cls):
        input_file = Path(cls.astdys_full_filename())
        if not input_file.exists():
            resonances.logger.info('Cannot find AstDyS catalog. Trying to download it...')
            try:
                urllib.request.urlretrieve(resonances.config.get('astdys.catalog.url'), 'cache/allnum.cat')
            except Exception:
                raise Exception(
                    "No input catalog available. Cannot download it too. Put AstDys allnum.cat or allnum.csv in the cache directory!"
                )
            resonances.logger.info('Successfully downloaded. Continue working...')

        cat = cls.transform_astdys_catalog()
        cat.to_csv(cls.catalog_full_filename(), index=False)

    @classmethod
    def transform_astdys_catalog(cls):
        catalog = pd.read_csv(cls.astdys_full_filename(), delim_whitespace=True, skiprows=5)
        cat = catalog.rename(
            columns={
                '!': 'num',
                'Name,': 'epoch',
                'Epoch(MJD),': 'a',
                'a,': 'e',
                'e,': 'inc',
                'i,': 'Omega',
                'long.': 'omega',
                'node,': 'M',
            }
        )
        cat['num'] = cat['num'].str.replace("'", "")
        deg_cols = ['inc', 'Omega', 'omega', 'M']
        for col in deg_cols:
            cat[col] = cat[col].map(lambda x: float(x) * np.pi / 180)

        column_to_move = 'epoch'
        cat = cat[[col for col in cat.columns if col != column_to_move] + [column_to_move]]
        cat.drop(cat.columns[[8, 9, 10]], axis=1, inplace=True)
        return cat
