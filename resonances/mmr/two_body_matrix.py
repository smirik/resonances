import numpy as np
import pandas as pd

from resonances.config import config
from .two_body import TwoBody
from resonances.matrix.matrix import Matrix


class TwoBodyMatrix(Matrix):
    """Two-body MMRs m1*P + m*body with m < 0: the prograde ones (see TwoBody.calculate_resonant_axis)."""

    catalog_file = 'MATRIX_2BODY_FILE'
    limit_keys = ('MATRIX_2BODY_PRIMARY_MAX', 'MATRIX_2BODY_COEF_MAX', 'MATRIX_2BODY_ORDER_MAX')
    catalog_version = 2  # v2: no m > 0 rows
    planet_columns = ['planet']

    @classmethod
    def build(cls):
        primary_max = int(config.get('MATRIX_2BODY_PRIMARY_MAX'))
        m_max = int(config.get('MATRIX_2BODY_COEF_MAX'))
        q_max = int(config.get('MATRIX_2BODY_ORDER_MAX'))
        planets = cls.catalog_planets()
        data = []
        for planet in planets:
            for m1 in range(1, primary_max):
                for m in range(-m_max, 0):
                    if np.gcd(m1, m) > 1:
                        continue
                    p = 0 - (m1 + m)
                    if abs(p) > q_max:
                        continue
                    mmr = TwoBody([m1, m, 0, p], [planet])
                    data.append([mmr.to_short(), planet, m1, m, abs(p), mmr.resonant_axis])

        df = pd.DataFrame(data, columns=['mmr', 'planet', 'm1', 'm', 'q', 'a'])
        cls.matrix = df
        return df
