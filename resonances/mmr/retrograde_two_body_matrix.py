from math import gcd

import pandas as pd

from resonances.config import config
from resonances.matrix.matrix import Matrix
from .retrograde_two_body import RetrogradeTwoBody


class RetrogradeTwoBodyMatrix(Matrix):
    """Retrograde p/-q resonances (RetrogradeTwoBody): one row per planet, coprime p, q and k.

    p, q <= MATRIX_2BODY_RETRO_COEF_MAX, p + q <= MATRIX_2BODY_RETRO_ORDER_MAX, and every
    k = 0..(p+q)//2, as in Li, Huang & Gong (2019): 143 pairs (p, q) per planet at 15 / 30.
    The cache stamp holds both limits, so changing either rebuilds the cached catalog.
    """

    catalog_file = 'MATRIX_2BODY_RETRO_FILE'
    catalog_version = 1
    planet_columns = ['planet']
    limit_keys = ('MATRIX_2BODY_RETRO_COEF_MAX', 'MATRIX_2BODY_RETRO_ORDER_MAX')

    @classmethod
    def _stamp(cls) -> str:
        """The version and the limits: a cache built with other limits is rebuilt."""
        limits = ' '.join(f'{key}={int(config.get(key))}' for key in cls.limit_keys)
        return f'{super()._stamp()} {limits}'

    @classmethod
    def build(cls):
        coef_max = int(config.get('MATRIX_2BODY_RETRO_COEF_MAX'))
        order_max = int(config.get('MATRIX_2BODY_RETRO_ORDER_MAX'))
        planets = cls.catalog_planets()
        data = []
        for planet in planets:
            for p in range(1, coef_max + 1):
                for q in range(1, coef_max + 1):
                    if gcd(p, q) > 1 or p + q > order_max:
                        continue
                    for k in range(0, (p + q) // 2 + 1):
                        mmr = RetrogradeTwoBody(p=p, q=q, k=k, planet=planet)
                        data.append([mmr.to_s(), planet, p, q, k, mmr.order(), mmr.resonant_axis])

        cls.matrix = pd.DataFrame(data, columns=['mmr', 'planet', 'p', 'q', 'k', 'order', 'a'])
        return cls.matrix
