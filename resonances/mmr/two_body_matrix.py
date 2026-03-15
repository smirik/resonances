import numpy as np
import pandas as pd

from resonances.config import config
from resonances.data import const
from .two_body import TwoBody
from resonances.matrix.matrix import Matrix


class TwoBodyMatrix(Matrix):
    catalog_file = 'MATRIX_2BODY_FILE'
    planet_columns = ['planet']

    @classmethod
    def build(cls):
        primary_max = int(config.get('MATRIX_2BODY_PRIMARY_MAX'))
        m_max = int(config.get('MATRIX_2BODY_COEF_MAX'))
        q_max = int(config.get('MATRIX_2BODY_ORDER_MAX'))
        if (cls.planets is None) or (len(cls.planets) == 0):
            planets = const.SOLAR_SYSTEM
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
                    mmr = TwoBody([m1, m, 0, p], [planet])
                    try:
                        axis = mmr.resonant_axis
                        data.append([mmr.to_short(), planet, m1, m, abs(p), axis])
                    except Exception:  # pragma: no cover
                        continue

        df = pd.DataFrame(data, columns=['mmr', 'planet', 'm1', 'm', 'q', 'a'])
        cls.matrix = df
        return df
