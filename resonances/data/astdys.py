import pandas as pd
import numpy as np
from pathlib import Path

import resonances.config


class astdys:

    catalog = None

    @classmethod
    def search(cls, num):
        num = str(num)
        if cls.catalog is None:
            print('No catalog loaded. Loading...')
            cls.check_or_build_catalog()
            cls.catalog = pd.read_csv(resonances.config.get('catalog'))
            cls.catalog['num'] = cls.catalog['num'].astype(str)

        if num in cls.catalog['num'].values:
            return cls.catalog.loc[cls.catalog['num'] == num].to_dict('records')[0]

        return None

    @classmethod
    def check_or_build_catalog(cls):
        output_file = Path(resonances.config.get('catalog'))
        if not output_file.exists():
            input_file = Path(resonances.config.get('astdys.catalog'))
            if not input_file.exists():
                raise Exception("No input catalog available: put AstDys allnum.cat or allnum.csv in the cache directory!")

            cat = cls.transform_astdys_catalog()
            cat.to_csv(resonances.config.get('catalog'), index=False)

    @classmethod
    def transform_astdys_catalog(cls):
        catalog = pd.read_csv(resonances.config.get('astdys.catalog'), delim_whitespace=True, skiprows=5)
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

        cat.drop(cat.columns[[1, 8, 9, 10, 11]], axis=1, inplace=True)
        return cat
