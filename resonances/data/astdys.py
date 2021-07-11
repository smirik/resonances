import pandas as pd
import numpy as np
from pathlib import Path


class astdys:

    catalog = None

    input_file_src = 'cache/allnum.cat'
    catalog_file_src = 'cache/allnum.csv'

    date = '2020-12-17 00:00'
    jd = '2459200.5'

    @classmethod
    def test(cls):
        print('test from astdys')

    @classmethod
    def search(cls, num):
        if cls.catalog is None:
            print('No catalog loaded. Loading...')
            cls.check_or_build_catalog()
            cls.catalog = pd.read_csv(cls.catalog_file_src)

        return cls.catalog.loc[cls.catalog['num'] == num].to_dict('records')[0]

    @staticmethod
    def check_or_build_catalog():
        output_file = Path(astdys.catalog_file_src)
        if not output_file.exists():
            input_file = Path(astdys.input_file_src)
            if not input_file.exists():
                raise Exception("No input catalog available: put AstDys allnum.cat or allnum.csv in the cache directory!")

            print('No CSV file found. Creating from allnum.cat.')
            astdys.transform_astdys_catalog()

    @staticmethod
    def transform_astdys_catalog():
        catalog = pd.read_csv(astdys.input_file_src, delim_whitespace=True, skiprows=5)
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
        cat.to_csv(astdys.catalog_file_src, index=False)
