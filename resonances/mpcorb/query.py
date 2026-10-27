"""Selections of the catalogue by orbital elements, orbit quality and MPC's orbit-type flags."""

import numbers
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from . import cache
from .parser import FLAG_BITS, ORBIT_TYPE_MASK, ORBIT_TYPES
from .settings import Settings, resolve

Range = Optional[Tuple[Optional[float], Optional[float]]]

# query's range arguments -> the column they bound (q = a(1 - e) and Q = a(1 + e) are computed)
RANGE_COLUMNS = {
    'a': 'a',
    'e': 'e',
    'i': 'inc_deg',
    'q': 'q',
    'Q': 'Q',
    'H': 'H',
    'node': 'Omega_deg',
    'peri': 'omega_deg',
    'M': 'M_deg',
    'n_obs': 'n_obs',
    'n_opp': 'n_opp',
    'arc_years': 'arc_years',
    'U': 'U_value',
}
# Ranges of these may wrap through 360: (350, 10) is 350..360 and 0..10 deg
ANGLE_RANGES = ('node', 'peri', 'M')
EPOCHS = ('any', 'standard')
TABLE_OUTPUT = ('packed', 'number', 'designation', 'name', 'epoch_mjd', 'a', 'e', 'inc_deg', 'Omega_deg', 'omega_deg', 'M_deg')


def query(
    *,
    a: Range = None,
    e: Range = None,
    i: Range = None,
    q: Range = None,
    Q: Range = None,
    H: Range = None,
    node: Range = None,
    peri: Range = None,
    M: Range = None,
    n_obs: Range = None,
    n_opp: Range = None,
    arc_years: Range = None,
    U: Range = None,
    epoch: str = 'any',
    numbered: Optional[bool] = None,
    orbit_type: Union[str, Iterable[str], None] = None,
    neo: Optional[bool] = None,
    km_neo: Optional[bool] = None,
    pha: Optional[bool] = None,
    one_opp_earlier: Optional[bool] = None,
    critical_list: Optional[bool] = None,
    catalog: Optional[str] = None,
    cache_dir: Optional[str] = None,
    settings: Optional[Settings] = None,
    snapshot_dict: Optional[dict] = None,
):
    """The rows of the catalogue that satisfy every criterion given, as a DataFrame (`query(a=(0, 1))`: the orbits
    inside 1 AU).

    Ranges are inclusive tuples (min, max), None for an open end: a, q = a(1 - e), Q = a(1 + e) in AU; e; i, node,
    peri, M in degrees (the angles may wrap: node=(350, 10)); H; n_obs, n_opp (observations, oppositions); arc_years
    (`parser.arc_years`: last minus first year of observation, days / 365.25 for a one-opposition orbit); U, the
    uncertainty parameter 0..9. A row without the value (a blank H, U = E) fails any range on it.
    `epoch`: 'any', or 'standard' for the rows at the catalogue's standard epoch. `numbered`: True for numbered
    objects, False for provisional designations. `orbit_type`: one name or several of `parser.ORBIT_TYPES` (atira,
    aten, apollo, amor, q_lt_1665, hungaria, mpc_internal, hilda, jupiter_trojan, distant), MPC's own classification
    in the 4-hex flags (columns 162-165 of https://minorplanetcenter.net/iau/info/MPOrbitFormat.html, "not 100%
    reliable"). `neo`, `km_neo`, `pha`, `one_opp_earlier`, `critical_list`: True or False to require the flag
    (bits 11-15 of the same field) set or clear.

    Columns: packed, number (0 for a provisional designation), designation, name, epoch_mjd, a, e, inc_deg,
    Omega_deg, omega_deg, M_deg, q, Q, H, U, n_obs, n_opp, arc_years, flags (as in the file), orbit_type ('' when
    the type is 0 or undefined) and the five flag columns.
    """
    ranges = dict(a=a, e=e, i=i, q=q, Q=Q, H=H, node=node, peri=peri, M=M, n_obs=n_obs, n_opp=n_opp, arc_years=arc_years, U=U)
    wanted_flags = dict(neo=neo, km_neo=km_neo, pha=pha, one_opp_earlier=one_opp_earlier, critical_list=critical_list)
    s = resolve(settings, catalog, cache_dir)
    current = snapshot_dict or cache.snapshot(settings=s)
    table = cache.load_table(snapshot_dict=current, settings=s)
    extra = cache.query_columns(snapshot_dict=current, settings=s)

    mask = np.ones(len(table), dtype=bool)
    for name, bounds in ranges.items():
        if bounds is not None:
            mask &= _in_range(name, _column(RANGE_COLUMNS[name], table, extra), bounds)
    mask &= _epoch_mask(epoch, table, current)
    if numbered is not None:
        mask &= (table['number'].to_numpy() > 0) == bool(numbered)
    flags = extra['flags_value'].to_numpy()
    if orbit_type is not None:
        mask &= np.isin(flags & ORBIT_TYPE_MASK, _type_codes(orbit_type))
    for name, wanted in wanted_flags.items():
        if wanted is not None:
            mask &= ((flags >> FLAG_BITS[name]) & 1).astype(bool) == bool(wanted)
    return _selection(table, extra, np.flatnonzero(mask))


def search_by_axis(
    axis: float,
    sigma: float = 0.1,
    catalog: Optional[str] = None,
    cache_dir: Optional[str] = None,
    settings: Optional[Settings] = None,
):
    """The rows with axis - sigma <= a <= axis + sigma (AU), as `astdys.search_by_axis`; ValueError unless both are
    positive."""
    if axis <= 0:
        raise ValueError(f"Semi-major axis must be positive, got {axis}")
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    return query(a=(axis - sigma, axis + sigma), catalog=catalog, cache_dir=cache_dir, settings=settings)


def get_catalog(catalog: Optional[str] = None, cache_dir: Optional[str] = None, settings: Optional[Settings] = None):
    """The whole catalogue with the columns of `query` (a copy: 1.6M rows for MPCORB)."""
    return query(catalog=catalog, cache_dir=cache_dir, settings=settings)


def identifiers(selection) -> List[Union[int, str]]:
    """The identifiers of a selection's rows that `search`/`lookup` read back: the number of a numbered object, else
    its designation (else its packed form)."""
    out = []
    for number, designation, packed in zip(selection['number'], selection['designation'], selection['packed']):
        out.append(int(number) if int(number) > 0 else (designation or packed))
    return out


def _column(name: str, table, extra) -> np.ndarray:
    if name in ('q', 'Q'):
        a, e = table['a'].to_numpy(), table['e'].to_numpy()
        return a * (1.0 - e) if name == 'q' else a * (1.0 + e)
    if name in extra.columns:
        return extra[name].to_numpy(dtype=float)
    return table[name].to_numpy(dtype=float)


def _bounds(name: str, bounds) -> Tuple[Optional[float], Optional[float]]:
    """(min, max) of a range argument, checked."""
    if isinstance(bounds, (str, bytes)) or not hasattr(bounds, '__len__') or len(bounds) != 2:
        raise TypeError(f"{name} is a range (min, max) with None for an open end, got {bounds!r}")
    low, high = bounds
    for value in (low, high):
        if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Real) or np.isnan(value)):
            raise TypeError(f"{name}: the ends of a range are numbers or None, got {bounds!r}")
    if low is not None and high is not None and low > high and name not in ANGLE_RANGES:
        raise ValueError(f"{name}: min {low} > max {high}")
    return low, high


def _in_range(name: str, values: np.ndarray, bounds) -> np.ndarray:
    """Inclusive; nan fails. An angle range with min > max wraps through 360."""
    low, high = _bounds(name, bounds)
    if low is not None and high is not None and low > high:
        return (values >= low) | (values <= high)
    inside = np.ones(len(values), dtype=bool)
    if low is not None:
        inside &= values >= low
    if high is not None:
        inside &= values <= high
    return inside


def _epoch_mask(epoch: str, table, snapshot_dict: dict) -> np.ndarray:
    if epoch not in EPOCHS:
        raise ValueError(f"epoch is one of {', '.join(EPOCHS)}, got {epoch!r}")
    if epoch == 'any':
        return np.ones(len(table), dtype=bool)
    return table['epoch_mjd'].to_numpy() == float(snapshot_dict['standard_epoch_mjd'])


def _type_codes(orbit_type) -> List[int]:
    names = [orbit_type] if isinstance(orbit_type, str) else list(orbit_type)
    codes = {name: code for code, name in ORBIT_TYPES.items()}
    unknown = [name for name in names if name not in codes]
    if unknown:
        raise ValueError(f"Unknown orbit type {', '.join(map(repr, unknown))}: one of {', '.join(codes)}")
    return [codes[name] for name in names]


def _selection(table, extra, rows: np.ndarray):
    """The output frame of `query` for the row positions `rows`."""
    import pandas as pd

    out = table.iloc[rows][list(TABLE_OUTPUT)].reset_index(drop=True)
    a, e = out['a'].to_numpy(), out['e'].to_numpy()
    out['q'], out['Q'] = a * (1.0 - e), a * (1.0 + e)
    picked = extra.iloc[rows].reset_index(drop=True)
    out['H'] = picked['H']
    out['U'] = table['U'].iloc[rows].astype(str).to_numpy()
    out['n_obs'] = picked['n_obs'].astype('Int64')
    out['n_opp'] = picked['n_opp'].astype('Int64')
    out['arc_years'] = picked['arc_years']
    out['perturbers'] = table['perturbers'].iloc[rows].astype(str).to_numpy()
    out['flags'] = picked['flags'].astype(str)
    flags = picked['flags_value'].to_numpy()
    out['orbit_type'] = pd.Series(flags & ORBIT_TYPE_MASK).map(ORBIT_TYPES).fillna('').to_numpy()
    for name, bit in FLAG_BITS.items():
        out[name] = ((flags >> bit) & 1).astype(bool)
    return out
