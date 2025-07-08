import datetime
from typing import Union
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import numpy as np


def get_body_keplerian_elements(s, date: Union[str, datetime.datetime]) -> dict:
    if isinstance(s, int) or isinstance(s, float):
        s = str(s) + ';'

    t = Time(date)
    jd = t.jd

    obj = Horizons(id=s, location='500@10', epochs=jd)
    elems = obj.elements()

    elem = {
        'a': elems['a'][0],
        'e': elems['e'][0],
        'inc': np.radians(elems['incl'][0]),
        'omega': np.radians(elems['w'][0]),
        'Omega': np.radians(elems['Omega'][0]),
        'M': np.radians(elems['M'][0]),
    }

    return elem
