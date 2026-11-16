"""Finding objects by number, designation or name, and their elements."""

import datetime
import math
import numbers
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from . import cache
from .designations import normalise_provisional, pack_number, pack_provisional, unpack_number, unpack_provisional
from .settings import Settings, resolve

# The rows an ambiguity error lists
AMBIGUOUS_ROWS_SHOWN = 10


def query_keys(identifier) -> Tuple[set, set, set]:
    """(packed forms, designations, name keys) that `identifier` can match: a number (463, '463', '(463)', '00463',
    'A1429', '~02pM'), a provisional designation, packed or not ('2001 ND13', '2001ND13', 'K01N13D', '_QC0000',
    '2040 P-L', in any case), or a name (case-insensitive, 'Ceres', '(1) Ceres'). ValueError for an empty one."""
    if isinstance(identifier, bool) or identifier is None:
        raise ValueError(f"Not an asteroid identifier: {identifier!r}")
    if isinstance(identifier, numbers.Integral):
        return {pack_number(int(identifier))}, set(), set()
    s = str(identifier).strip()
    if not s:
        raise ValueError(f"Not an asteroid identifier: {identifier!r}")
    m = re.fullmatch(r'\(?(\d+)\)?', s)
    if m:
        return {pack_number(int(m.group(1)))}, set(), set()
    m = re.fullmatch(r'\((\d+)\)\s*(.+)', s)
    if m:
        return {pack_number(int(m.group(1)))}, set(), set()
    packed, designations, names = set(), {s}, {s.casefold()}
    if unpack_number(s):
        packed.add(s)
    if unpack_provisional(s):
        packed.add(s)
        designations.add(unpack_provisional(s))
    upper = s.upper()  # '2001 nd13' is 2001 ND13 (packed forms are case-sensitive and are not upper-cased)
    if normalise_provisional(upper):
        designations.add(normalise_provisional(upper))
        if pack_provisional(upper):
            packed.add(pack_provisional(upper))
    return packed, designations, names


def lookup(
    identifiers: Iterable,
    catalog: Optional[str] = None,
    cache_dir: Optional[str] = None,
    snapshot_dict: Optional[dict] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, dict]:
    """{str(identifier): table row as a dict} for the identifiers found, in `snapshot_dict`'s table (else the cached
    snapshot's); ValueError when one matches several rows.

    One pass over the table per column (packed, designation, name), then dictionaries: the cost grows with the table
    plus the identifiers, not with their product (a lookup of every numbered object stays a matter of seconds)."""
    table = cache.load_table(snapshot_dict=snapshot_dict, settings=resolve(settings, catalog, cache_dir))
    keys = {str(ident): query_keys(ident) for ident in identifiers}
    # column -> {value: [row positions]} for the values some identifier asks for
    index = []
    for slot, column in enumerate(('packed', 'designation', 'name_key')):
        wanted = set().union(*(k[slot] for k in keys.values())) if keys else set()
        positions = np.flatnonzero(table[column].isin(wanted).to_numpy()) if wanted else np.empty(0, dtype=int)
        by_value = defaultdict(list)
        for position, value in zip(positions.tolist(), table[column].to_numpy()[positions].tolist()):
            by_value[value].append(position)
        index.append(by_value)

    matched = {}
    for ident, key_sets in keys.items():
        positions = sorted({pos for by_value, values in zip(index, key_sets) for value in values for pos in by_value.get(value, ())})
        if len(positions) > 1:
            packed = table['packed'].to_numpy()[positions[:AMBIGUOUS_ROWS_SHOWN]].tolist()
            more = f" and {len(positions) - AMBIGUOUS_ROWS_SHOWN} more" if len(positions) > AMBIGUOUS_ROWS_SHOWN else ''
            raise ValueError(
                f"{ident!r} matches {len(positions)} MPC rows: {', '.join(packed)}{more}; give the number or the packed designation"
            )
        if positions:
            matched[ident] = positions[0]
    if not matched:
        return {}
    unique = sorted(set(matched.values()))
    records = dict(zip(unique, table.iloc[unique].to_dict('records')))
    return {ident: dict(records[position]) for ident, position in matched.items()}


def row_label(row: dict) -> str:
    """'(463) Lola', '(101429) 1998 VF31' or '2001 ND13' for messages."""
    if int(row['number']):
        return f"({int(row['number'])}) {row['name'] or row['designation']}".strip()
    return row['designation'] or row['packed']


def elements_from_row(row: dict, snapshot_dict: Optional[dict] = None) -> dict:
    """The element dict of a row: a (AU), e, angles in radians, 'epoch' as MJD TT, and the row's provenance.
    ValueError for a row that is not an elliptic orbit (none in the 2026-10-25 MPCORB: max e = 0.9974)."""
    if not (float(row['a']) > 0 and 0 <= float(row['e']) < 1):
        raise ValueError(f"{row_label(row)}: a = {row['a']}, e = {row['e']} is not an elliptic orbit")
    return {
        'a': float(row['a']),
        'e': float(row['e']),
        'inc': math.radians(float(row['inc_deg'])),
        'Omega': math.radians(float(row['Omega_deg'])),
        'omega': math.radians(float(row['omega_deg'])),
        'M': math.radians(float(row['M_deg'])),
        'epoch': float(row['epoch_mjd']),
        'source': 'mpc',
        'packed': row['packed'],
        'number': int(row['number']) or None,
        'designation': row['designation'],
        'mpc_name': row['name'],
        'U': row['U'],
        'perturbers': row['perturbers'],
        'catalog_sha256': (snapshot_dict or {}).get('sha256'),
    }


def quality_warnings(row: dict, warn_u: Optional[int] = None) -> List[str]:
    """Why an orbit may be poor: U >= `warn_u` (default: the settings' one), a non-numeric U (E: assumed e, D/F:
    double/failed), no perturbers."""
    warn_u = resolve().warn_u if warn_u is None else warn_u
    label, u = row_label(row), str(row['U'])
    out = []
    if u.isdigit() and int(u) >= warn_u:
        out.append(f"{label}: MPC uncertainty parameter U = {u} (>= {warn_u}): a poorly determined orbit")
    elif not u.isdigit():
        out.append(f"{label}: MPC uncertainty parameter U = {u or 'blank'}: no measured orbit quality")
    if not row['perturbers']:
        out.append(f"{label}: an unperturbed MPC orbit (recent one-opposition object)")
    return out


def search(
    identifier: Union[int, str, Iterable],
    catalog: Optional[str] = None,
    cache_dir: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Union[Optional[dict], Dict[str, dict]]:
    """The elements (`elements_from_row`) of one object, or None when it is not in the catalogue; for a list,
    {str(identifier): elements} of those found (as `astdys.search`). Any form of `query_keys` identifies an object;
    ValueError when one matches several rows."""
    s = resolve(settings, catalog, cache_dir)
    current = cache.snapshot(settings=s)
    single = isinstance(identifier, (numbers.Integral, str))
    rows = lookup([identifier] if single else list(identifier), snapshot_dict=current, settings=s)
    found = {key: elements_from_row(row, current) for key, row in rows.items()}
    return found.get(str(identifier)) if single else found


def get_catalog_datetime(
    catalog: Optional[str] = None, cache_dir: Optional[str] = None, settings: Optional[Settings] = None
) -> datetime.datetime:
    """The catalogue's standard epoch (the epoch of most rows) as a naive datetime, its TT calendar value."""
    return cache.catalog_datetime(cache.snapshot(catalog, cache_dir, settings))


def load(catalog: Optional[str] = None, cache_dir: Optional[str] = None, settings: Optional[Settings] = None):
    """The catalogue's table, downloaded on first use (as `astdys.load`, but it returns the table)."""
    return cache.load_table(catalog, cache_dir, settings=settings)


def rebuild(catalog: Optional[str] = None, cache_dir: Optional[str] = None, settings: Optional[Settings] = None) -> dict:
    """Download the catalogue again even when the server reports no change; returns the new snapshot."""
    return cache.update_catalog(catalog, cache_dir, force=True, settings=settings)
