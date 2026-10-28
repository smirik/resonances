"""MPC_LOOKUP=file_then_api: the nominal elements of an identifier that is not in the cached MPC catalogue (an alias
such as a secondary designation, an object outside the MPC_CATALOG subset, one newer than the snapshot) from MPC's
get-orb API (docs/config.md, "Minor Planet Center").

Package side only (`resonances.mpcorb` stays self-contained). The answer is fetched and cached by the clones' code
(`clones.solution.fetch`, CLONES_CACHE_DIR/mpc/<identifier>.json with its sha256), in the parent process only. Only the
nominal is read, so no covariance is needed and the clones' U policy (MPC_WEAK_U) does not apply here; a weak orbit
gets the catalogue's quality warning instead. The elements are those of get-orb's KEP block (a, e, i, node, argperi,
mean anomaly at the orbit's epoch, J2000 ecliptic, TT), else of its CAR block converted with mu = k^2; the epoch policy
(MPC_EPOCH_POLICY, MPC_PROPAGATE_MAX_DAYS) then applies as to a catalogue row.
"""

import json
import math
import numpy as np

from resonances.clones import elements as conversions
from resonances.clones.solution import _mpc_frame, cache_dir, fetch, solution_url

LOOKUP_MODES = ('file', 'file_then_api')
KEP_NAMES = ['a', 'e', 'i', 'node', 'argperi', 'mean_anomaly']
CAR_NAMES = ['x', 'y', 'z', 'vx', 'vy', 'vz']
TIME_SYSTEMS = ('TDT', 'TT')


def _orbit(data, identifier) -> dict:
    """The one orbit of a get-orb answer ([record, status] or record); ValueError for none or several."""
    record = data[0] if isinstance(data, list) and data else data
    orbits = record.get('mpc_orb') if isinstance(record, dict) else None
    if not orbits:
        raise ValueError(f'MPC get-orb has no orbit for {identifier!r} (unknown to the MPC?)')
    if len(orbits) != 1:
        raise ValueError(f'MPC get-orb returned {len(orbits)} orbits for {identifier!r}: ambiguous, none is chosen')
    return orbits[0]


def nominal_elements(data, identifier='') -> dict:
    """The package's elements dict (a [AU], e, angles in radians, 'epoch' MJD TT) of a get-orb answer's nominal orbit,
    with its provenance: 'designation' (unpacked primary), 'number', 'mpc_name', 'U', 'nongrav'. No covariance is
    read. ValueError for an answer without one orbit, another frame or time scale, or no elliptic orbit."""
    data = json.loads(data) if isinstance(data, (str, bytes)) else data
    orbit = _orbit(data, identifier)
    _mpc_frame(orbit.get('system_data') or {})
    epoch = orbit.get('epoch_data') or {}
    if str(epoch.get('timeform') or 'MJD').upper() != 'MJD' or str(epoch.get('timesystem') or 'TDT').upper() not in TIME_SYSTEMS:
        raise ValueError(f"MPC get-orb epoch of {identifier!r} in {epoch.get('timeform')}/{epoch.get('timesystem')}: only MJD TT is read")
    kep, car = orbit.get('KEP') or {}, orbit.get('CAR') or {}
    if list(kep.get('coefficient_names') or [])[:6] == KEP_NAMES:
        values = [float(x) for x in kep['coefficient_values'][:6]]
        elem = {'a': values[0], 'e': values[1], **{key: math.radians(v) for key, v in zip(('inc', 'Omega', 'omega', 'M'), values[2:])}}
    elif list(car.get('coefficient_names') or [])[:6] == CAR_NAMES:
        kep_arrays = conversions.cartesian_to_keplerian(np.array(car['coefficient_values'][:6], dtype=float))
        elem = {key: float(kep_arrays[key][0]) for key in conversions.KEPLERIAN_KEYS}
    else:
        raise ValueError(f'MPC get-orb answer for {identifier!r} has neither a KEP nor a CAR block')
    if not all(math.isfinite(v) for v in elem.values()) or not (elem['a'] > 0 and 0 <= elem['e'] < 1):
        raise ValueError(f"MPC get-orb orbit of {identifier!r}: a = {elem['a']}, e = {elem['e']} is not an elliptic orbit")
    for key in ('Omega', 'omega', 'M'):
        elem[key] = elem[key] % (2 * math.pi)
    designation = orbit.get('designation_data') or {}
    stats = orbit.get('orbit_fit_statistics') or {}
    permid = str(designation.get('permid') or '').strip()
    u_param = stats.get('U_param')
    return {
        **elem,
        'epoch': float(epoch['epoch']),
        'source': 'mpc',
        'lookup': 'get-orb',
        'number': int(permid) if permid.isdigit() else None,
        'designation': str(designation.get('unpacked_primary_provisional_designation') or ''),
        'mpc_name': str(designation.get('name') or ''),
        'U': '' if u_param is None else str(u_param),
        'nongrav': bool((orbit.get('non_grav_booleans') or {}).get('non_gravs')),
    }


def label(elem: dict, identifier) -> str:
    """'(463) Lola', '2013 RZ53' or the identifier, for messages."""
    if elem.get('number'):
        return f"({elem['number']}) {elem.get('mpc_name') or elem.get('designation')}".strip()
    return elem.get('designation') or str(identifier)


def quality_warnings(elem: dict, identifier, warn_u: int) -> list:
    """As the catalogue's: U >= warn_u or no U; and a fit with non-gravitational parameters (not modelled)."""
    name, u = label(elem, identifier), str(elem.get('U', ''))
    out = []
    if u.isdigit() and int(u) >= warn_u:
        out.append(f'{name}: MPC uncertainty parameter U = {u} (>= {warn_u}): a poorly determined orbit')
    elif not u.isdigit():
        out.append(f"{name}: MPC uncertainty parameter U = {u or 'blank'}: no measured orbit quality")
    if elem.get('nongrav'):
        out.append(f'{name}: the MPC fit has non-gravitational parameters; they are not modelled')
    return out


def fetch_elements(identifier, refresh: bool = False) -> dict:
    """The nominal elements of `identifier` from get-orb (cached in CLONES_CACHE_DIR/mpc/), with 'getorb_sha256' and
    'getorb_identifier' (what was asked) added; ValueError when the MPC has no usable orbit or the request fails."""
    url, body, filename, _ = solution_url(identifier, 'mpc')
    try:
        content, meta = fetch(url, cache_dir() / 'mpc' / filename, body=body, refresh=refresh)
    except (OSError, ValueError) as error:  # FileNotFoundError (404), URLError, a cache whose sha256 differs
        raise ValueError(f'MPC get-orb request for {identifier!r} failed: {error}') from None
    elem = nominal_elements(content.decode('utf-8', errors='replace'), identifier)
    return {**elem, 'getorb_identifier': str(identifier).strip(), 'getorb_sha256': meta.get('sha256')}
