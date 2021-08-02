import numpy as np
import pandas as pd

import resonances
from resonances.matrix.matrix import Matrix


class TwoBodyMatrix(Matrix):

    catalog_file = 'matrix.2body.file'

    @classmethod
    def build(cls):
        primary_max = resonances.config.get('matrix.2body.primary_max')
        m_max = resonances.config.get('matrix.2body.coefficients_max')
        q_max = resonances.config.get('matrix.2body.max_order')
        if cls.planets is None:
            planets = resonances.data.const.SOLAR_SYSTEM
        else:
            planets = cls.planets
        data = []
        for planet in planets:
            for m1 in range(1, primary_max):
                for m in range(-m_max, m_max):
                    if np.gcd(m1, m) > 1:
                        continue
                    if m == 0:
                        continue
                    p = 0 - (m1 + m)
                    if abs(p) > q_max:
                        continue
                    mmr = resonances.TwoBody([m1, m, 0, p], [planet])
                    try:
                        axis = mmr.resonant_axis
                        data.append([mmr.to_short(), planet, m1, m, abs(p), axis])
                    except Exception:
                        continue

        df = pd.DataFrame(data, columns=['mmr', 'planet', 'm1', 'm', 'q', 'a'])
        cls.matrix = df
        return df

    @classmethod
    def find_resonances(cls, a, sigma=0.02, planets=None):
        if cls.matrix is None:
            cls.load()

        if isinstance(planets, list):
            df = cls.matrix[(cls.matrix['a'] >= (a - sigma)) & (cls.matrix['a'] <= (a + sigma)) & (cls.matrix['planet'].isin(planets))]
        else:
            df = cls.matrix[(cls.matrix['a'] >= (a - sigma)) & (cls.matrix['a'] <= (a + sigma))]

        mmrs = []
        for mmr in df['mmr'].tolist():
            mmrs.append(resonances.create_mmr(mmr))
        return mmrs
