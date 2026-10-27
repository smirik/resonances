import numpy as np
import pandas as pd
import itertools

from resonances.config import config
from .three_body import ThreeBody
from resonances.matrix.matrix import Matrix


class ThreeBodyMatrix(Matrix):

    catalog_file = 'MATRIX_3BODY_FILE'
    limit_keys = ('MATRIX_3BODY_PRIMARY_MAX', 'MATRIX_3BODY_COEF_MAX', 'MATRIX_3BODY_ORDER_MAX')
    planet_columns = ['planet1', 'planet2']

    @classmethod
    # flake8: noqa: C901
    def build(cls):
        primary_max = int(config.get('MATRIX_3BODY_PRIMARY_MAX'))
        m_max = int(config.get('MATRIX_3BODY_COEF_MAX'))
        q_max = int(config.get('MATRIX_3BODY_ORDER_MAX'))
        planets = cls.catalog_planets()
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
                        mmr = ThreeBody([m1, m2, m, 0, 0, p], [planet1, planet2])
                        try:
                            axis = mmr.resonant_axis
                            data.append([mmr.to_short(), planet1, planet2, m1, m2, m, abs(p), axis])
                        except Exception:
                            continue

        df = pd.DataFrame(data, columns=['mmr', 'planet1', 'planet2', 'm1', 'm2', 'm', 'q', 'a'])
        cls.matrix = df
        return df
