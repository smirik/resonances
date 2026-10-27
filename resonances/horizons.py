import datetime
import numbers
from typing import Union
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import numpy as np

# JD - MJD
MJD_OFFSET = 2400000.5


def horizons_id(s: Union[int, float, str]) -> str:
    """The Horizons id of an asteroid: a number gets ';' (small-body search).

    Without ';' Horizons reads a bare number as a major body: '401' is Phobos, '502' Europa.
    Names and designations ('Hektor', '2017 HV1') are passed as they are.
    """
    if isinstance(s, numbers.Integral):
        return f"{int(s)};"
    if isinstance(s, numbers.Real):
        if not float(s).is_integer():
            raise ValueError(f"An asteroid number must be an integer, got {s}")
        return f"{int(s)};"
    s = str(s).strip()
    if s.isdigit():
        return f"{s};"
    return s


def get_body_keplerian_elements(s, date: Union[str, datetime.datetime]) -> dict:
    """Heliocentric Keplerian elements of an asteroid at `date` (angles in radians, 'epoch' in MJD)."""
    t = Time(date)

    obj = Horizons(id=horizons_id(s), location='500@10', epochs=t.jd)
    elems = obj.elements()

    elem = {
        'a': elems['a'][0],
        'e': elems['e'][0],
        'inc': np.radians(elems['incl'][0]),
        'omega': np.radians(elems['w'][0]),
        'Omega': np.radians(elems['Omega'][0]),
        'M': np.radians(elems['M'][0]),
        'epoch': float(elems['datetime_jd'][0]) - MJD_OFFSET,
    }

    return elem
