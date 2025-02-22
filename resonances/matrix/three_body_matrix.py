from typing import List
import numpy as np
import pandas as pd
import itertools

import resonances
from resonances.matrix.matrix import Matrix


class ThreeBodyMatrix(Matrix):

    catalog_file = 'MATRIX_3BODY_FILE'

    @classmethod
    def build(cls):
        primary_max = int(resonances.config.get('MATRIX_3BODY_PRIMARY_MAX'))
        m_max = int(resonances.config.get('MATRIX_3BODY_COEF_MAX'))
        q_max = int(resonances.config.get('MATRIX_3BODY_ORDER_MAX'))
        if (cls.planets is None) or (len(cls.planets) == 0):
            planets = resonances.data.const.SOLAR_SYSTEM
        else:
            planets = cls.planets
        pairs = list(itertools.combinations(planets, 2))
        data = []
        for planet1, planet2 in pairs:
            for m1 in range(1, primary_max):
                for m2 in range(-m_max, m_max):
                    for m in range(-m_max, m_max):
                        if np.gcd(np.gcd(m1, m2), m) > 1:
                            continue
                        if (m2 == 0) or (m == 0):
                            continue
                        p = 0 - (m1 + m2 + m)
                        if abs(p) > q_max:
                            continue
                        mmr = resonances.ThreeBody([m1, m2, m, 0, 0, p], [planet1, planet2])
                        try:
                            axis = mmr.resonant_axis
                            data.append([mmr.to_short(), planet1, planet2, m1, m2, m, abs(p), axis])
                        except Exception:
                            continue

        df = pd.DataFrame(data, columns=['mmr', 'planet1', 'planet2', 'm1', 'm2', 'm', 'q', 'a'])
        cls.matrix = df
        return df

    @classmethod
    def find_resonances(cls, a, sigma=0.02, planets=None) -> List[resonances.MMR]:
        if cls.matrix is None:
            cls.load()

        if isinstance(planets, list):
            df = cls.matrix[
                (cls.matrix['a'] >= (a - sigma))
                & (cls.matrix['a'] <= (a + sigma))
                & (cls.matrix['planet1'].isin(planets))
                & (cls.matrix['planet2'].isin(planets))
            ]
        else:
            df = cls.matrix[(cls.matrix['a'] >= (a - sigma)) & (cls.matrix['a'] <= (a + sigma))]

        mmrs = []
        for mmr in df['mmr'].tolist():
            mmrs.append(resonances.create_mmr(mmr))
        return mmrs
