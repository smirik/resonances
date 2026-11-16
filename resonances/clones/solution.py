"""Orbit solutions with their covariance: the source of virtual asteroids (docs/clones.md).

`orbit_solution(id, source)` downloads (once, then from the cache) an orbit fit with its covariance and returns an
`OrbitSolution`: the nominal and the covariance in the fit's own coordinates, the epoch of the covariance with its time
scale, and the provenance (URL, download time, sha256). Only sources that publish a full covariance are offered:

- 'astdys' (numbered asteroids) and 'neodys' (NEAs): OEF2.0 files, equinoctial (a, h, k, p, q, lambda) with lambda in
  degrees, ecliptic J2000, time TDT. `.eq1` is the orbit at the catalogue's standard epoch, `.eq0` at the fit's own
  epoch (near the middle of the observed arc); `.eq1` is the `.eq0` covariance propagated LINEARLY. Default: `.eq1` for
  AstDyS (good orbits), `.eq0` for NEODyS (author 2026-10-27: weak NEA orbits are drawn at their own epoch, never from a
  linearly propagated covariance).
- 'sbdb' (JPL): cometary (e, q, tp, node, peri, i), angles in degrees, tp in JD TDB, at the orbit-solution epoch (not
  the epoch of the listed elements); the nominal is the covariance block's own.
- 'mpc' (MPC get-orb): Cartesian (x, y, z, vx, vy, vz), AU and AU/day, ecliptic J2000, TDT, at the orbit's epoch, which
  for a weak orbit is far from its arc: refused when U >= MPC_WEAK_U (use 'neodys' or 'sbdb' there).

Only the heliocentric ecliptic J2000 frame of the package is read: a file that states another frame (OEF `refsys`, MPC
`system_data`, SBDB `equinox`) is refused, never rotated. An OEF file holds one record: a second one (a bulk catalogue)
is refused, not mixed.

Non-gravitational parameters (Yarkovsky A2, comet-like A1/A2/A3) are not modelled: the 6x6 orbital block (the marginal
covariance) is used and the solution is flagged (`OrbitSolution.nongrav`).
"""

import datetime
import hashlib
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from resonances.config import config as c
from resonances.logger import logger

from . import elements as conversions

MJD_TO_JD = 2400000.5
SOURCES = ('astdys', 'neodys', 'sbdb', 'mpc')
EPOCHS = ('fit', 'standard')
# The default product of each OEF source: AstDyS numbered orbits at the standard epoch, NEODyS at the fit's own epoch
DEFAULT_EPOCH = {'astdys': 'standard', 'neodys': 'fit'}
OEF_EXTENSION = {'fit': 'eq0', 'standard': 'eq1'}
ASTDYS_URL = 'https://newton.spacedys.com/~astdys2/epoch/numbered/{folder}/{number}.{ext}'
NEODYS_URL = 'https://newton.spacedys.com/~neodys2/epoch/{designation}.{ext}'
SBDB_URL = 'https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={sstr}&cov=mat&full-prec=1'
MPC_URL = 'https://data.minorplanetcenter.net/api/get-orb'
# MPC publishes the covariance at its orbit epoch only, linearly propagated there from the arc. Refusing it from U = 3 on
# (0 best .. 9 worst) is a conservative policy, not a physical boundary: a weak orbit's covariance far from its arc may
# no longer describe its orbits (2013 RZ53, W24 clone verification, E4), and the fit's own epoch is then safer.
MPC_WEAK_U = 3
# The frame the package works in (heliocentric ecliptic J2000): OEF refsys, MPC system_data, the J2000 obliquity (IAU
# 1976, as Horizons and MPC state it), arcsec
OEF_REFSYS = 'ECLM J2000'
MPC_REFSYS = ('ecliptic',)
MPC_REFFRAMES = ('icrf', 'j2000', 'eme2000')
OBLIQUITY_J2000_ARCSEC = 84381.448
OBLIQUITY_TOL_ARCSEC = 1e-3
# OEF tags of an orbit record: a second one of these is a second record
OEF_RECORD_TAGS = ('EQU', 'KEP', 'CAR', 'COM', 'MJD')
USER_AGENT = 'resonances (python)'
TIMEOUT_S = 60

COORDINATES = {
    'equinoctial': (('a', 'h', 'k', 'p', 'q', 'lambda'), ('au', '', '', '', '', 'deg')),
    'cometary': (('e', 'q', 'tp', 'node', 'peri', 'i'), ('', 'au', 'JD', 'deg', 'deg', 'deg')),
    'cartesian': (('x', 'y', 'z', 'vx', 'vy', 'vz'), ('au', 'au', 'au', 'au/d', 'au/d', 'au/d')),
}


@dataclass
class OrbitSolution:
    """An orbit fit: nominal and covariance in the fit's own coordinates (`coordinates`: equinoctial, cometary or
    cartesian; `labels` and `units` name the six), at `epoch_mjd` (`time_scale`: TDT or TDB, ecliptic J2000).

    `nongrav` holds the non-gravitational parameters the fit estimated ({} for none): they are not modelled, the
    covariance here is the orbital 6x6 block. `provenance` holds the URL, download time, sha256 and the source's
    quality fields (U, arc, condition code) when it has them."""

    name: str
    source: str
    coordinates: str
    nominal: np.ndarray
    covariance: np.ndarray
    epoch_mjd: float
    time_scale: str = 'TDT'
    frame: str = 'ecliptic J2000, heliocentric'
    nongrav: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.coordinates not in COORDINATES:
            raise ValueError(f"coordinates is one of {', '.join(COORDINATES)}, got {self.coordinates!r}")
        self.nominal = np.asarray(self.nominal, dtype=float).reshape(6)
        self.covariance = np.asarray(self.covariance, dtype=float).reshape(6, 6)
        if not np.allclose(self.covariance, self.covariance.T, rtol=1e-9, atol=0.0):
            raise ValueError(f'{self.name}: the covariance is not symmetric')
        if not (np.diag(self.covariance) > 0).all():
            raise ValueError(f'{self.name}: the covariance has a non-positive variance')

    @property
    def labels(self) -> tuple:
        return COORDINATES[self.coordinates][0]

    @property
    def units(self) -> tuple:
        return COORDINATES[self.coordinates][1]

    @property
    def sigma(self) -> np.ndarray:
        """The 1-sigma of each coordinate (sqrt of the covariance's diagonal), in `units`."""
        return np.sqrt(np.diag(self.covariance))

    def to_keplerian(self, rows) -> dict:
        """Rows of the fit's coordinates (at the solution's epoch) as Keplerian element arrays and `valid`."""
        if self.coordinates == 'equinoctial':
            return conversions.equinoctial_to_keplerian(rows)
        if self.coordinates == 'cometary':
            return conversions.cometary_to_keplerian(rows, self.epoch_mjd + MJD_TO_JD)
        return conversions.cartesian_to_keplerian(rows)

    def nominal_elements(self) -> dict:
        """The nominal as a package elements dict (radians, 'epoch' MJD)."""
        kep = self.to_keplerian(self.nominal)
        if not kep['valid'][0]:
            raise ValueError(f'{self.name}: the nominal orbit is not a bound ellipse')
        return {**{key: float(kep[key][0]) for key in conversions.KEPLERIAN_KEYS}, 'epoch': float(self.epoch_mjd)}

    def sample(self, n: Optional[int] = None, seed: Optional[int] = None, name: Optional[str] = None):
        """`n` clones (CLONES, else 100) drawn from the multivariate normal of the covariance in the fit's coordinates
        (`ensemble.sample`)."""
        from .ensemble import sample

        return sample(self, n=n, seed=seed, name=name)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'source': self.source,
            'coordinates': self.coordinates,
            'labels': list(self.labels),
            'units': list(self.units),
            'nominal': [float(x) for x in self.nominal],
            'covariance': [[float(x) for x in row] for row in self.covariance],
            'epoch_mjd': float(self.epoch_mjd),
            'time_scale': self.time_scale,
            'frame': self.frame,
            'nongrav': dict(self.nongrav),
            'provenance': dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'OrbitSolution':
        keys = ('name', 'source', 'coordinates', 'nominal', 'covariance', 'epoch_mjd', 'time_scale', 'frame', 'nongrav', 'provenance')
        return cls(**{key: data[key] for key in keys if key in data})

    @classmethod
    def from_file(cls, path: Union[str, Path], name: Optional[str] = None, source: str = 'file') -> 'OrbitSolution':
        """An OEF2.0 file with an EQU record and its COV (AstDyS/NEODyS .eq0/.eq1, OrbFit output)."""
        path = Path(path)
        solution = parse_oef(path.read_text(), source=source)
        if name is not None:
            solution.name = str(name)
        solution.provenance.setdefault('path', str(path.resolve()))
        solution.provenance.setdefault('sha256', hashlib.sha256(path.read_bytes()).hexdigest())
        return solution


# ----------------------------------------------------------------------------------------------- parsers
def _upper_triangle(values, dim: int, row_major: bool = True) -> np.ndarray:
    """A symmetric dim x dim matrix from its upper triangle listed row by row (OEF)."""
    count = dim * (dim + 1) // 2
    if len(values) < count:
        raise ValueError(f'a {dim}x{dim} covariance needs {count} values, got {len(values)}')
    matrix = np.zeros((dim, dim))
    it = iter(values[:count])
    for i in range(dim):
        for j in range(i, dim):
            matrix[i, j] = matrix[j, i] = next(it)
    return matrix


def _oef_records(lines) -> tuple:
    """(records {tag: fields}, COV values in order, the object's name) of the lines after the OEF header. One record
    only: a second element record or epoch (a multi-record file, e.g. a bulk .ctc) raises."""
    records, cov_values, name = {}, [], None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('!'):
            continue
        parts = stripped.split()
        tag = parts[0]
        if tag == 'COV':
            cov_values.extend(float(x.replace('D', 'E')) for x in parts[1:])
        elif tag in ('EQU', 'KEP', 'CAR', 'COM', 'MJD', 'LSP', 'NGR', 'MAG', 'NOR'):
            if tag in OEF_RECORD_TAGS and tag in records:
                raise ValueError(
                    f'OEF file with more than one record (a second {tag}, after {name!r}): only single-record files '
                    '(.eq0/.eq1 of one object) are read'
                )
            records[tag] = parts[1:]
        elif name is None:
            name = stripped.strip("'")
    return records, cov_values, name


def parse_oef(text: str, source: str = 'file') -> OrbitSolution:
    """An OEF2.0 multi-line record (AstDyS/NEODyS .eq0/.eq1): EQU, MJD with its time scale, LSP (non-gravitational
    model, the covariance's dimension), NGR (their values), COV (upper triangle, row by row). The COV units are those of
    EQU: lambda in degrees (the EIG line is reproduced only with lambda converted to radians, W24 clone checks).
    A 7x7 (or larger) covariance of a non-gravitational fit gives its orbital 6x6 block, flagged in `nongrav`."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().startswith('END_OF_HEADER')) + 1
    except StopIteration:
        start = 0
    refsys = re.search(r"refsys\s*=\s*([^!\n]+)", '\n'.join(lines[:start]))
    refsys = ' '.join(refsys.group(1).strip().strip("'").upper().split()) if refsys else OEF_REFSYS
    if refsys != OEF_REFSYS:
        raise ValueError(f'OEF refsys {refsys!r}: only {OEF_REFSYS!r} (heliocentric ecliptic J2000) is read, nothing is rotated')
    records, cov_values, name = _oef_records(lines[start:])
    if 'EQU' not in records:
        other = [tag for tag in ('KEP', 'CAR', 'COM') if tag in records]
        raise ValueError(f'OEF record without EQU (found {other or "none"}): only equinoctial OEF records are read')
    if not cov_values:
        raise ValueError('OEF record without COV: the file has no covariance')
    dim = int(records['LSP'][2]) if 'LSP' in records and len(records['LSP']) > 2 else 6
    if len(cov_values) != dim * (dim + 1) // 2:
        raise ValueError(f'OEF COV has {len(cov_values)} values, a {dim}x{dim} covariance has {dim * (dim + 1) // 2}')
    covariance = _upper_triangle(cov_values, dim)
    nominal = np.array([float(x.replace('D', 'E')) for x in records['EQU'][:6]])
    epoch_mjd = float(records['MJD'][0])
    time_scale = records['MJD'][1] if len(records['MJD']) > 1 else 'TDT'
    nongrav = {}
    if dim > 6 or ('LSP' in records and int(records['LSP'][0]) != 0):
        values = [float(x.replace('D', 'E')) for x in records.get('NGR', [])]
        nongrav = {f'A{i + 1}': value for i, value in enumerate(values)}
        nongrav['model'] = int(records['LSP'][0]) if 'LSP' in records else 0
    frame = refsys + ', heliocentric'
    return OrbitSolution(
        name=name or '',
        source=source,
        coordinates='equinoctial',
        nominal=nominal,
        covariance=covariance[:6, :6],
        epoch_mjd=epoch_mjd,
        time_scale=time_scale,
        frame=frame,
        nongrav=nongrav,
    )


def parse_sbdb(data: Union[str, dict], source: str = 'sbdb') -> OrbitSolution:
    """A JPL SBDB API answer (cov=mat&full-prec=1): the covariance at its own epoch (`orbit.covariance.epoch`, JD TDB),
    labelled e, q, tp, node, peri, i (+ estimated non-gravitational parameters), angles in degrees. The nominal is
    `covariance.elements` (present when the covariance epoch differs from the listed elements' epoch, which it usually
    does), else the listed elements when their epoch is the covariance's."""
    data = json.loads(data) if isinstance(data, str) else data
    if 'orbit' not in data:
        raise ValueError(f"SBDB answer without an orbit: {data.get('message') or data.get('code') or list(data)}")
    orbit = data['orbit']
    covariance = orbit.get('covariance')
    if not covariance or not covariance.get('data'):
        raise ValueError(f"SBDB has no covariance for {data.get('object', {}).get('fullname', '?')}")
    labels = list(covariance['labels'])
    wanted = list(COORDINATES['cometary'][0])
    if labels[:6] != wanted:
        raise ValueError(f'SBDB covariance labels {labels[:6]}, expected {wanted}')
    equinox = str(orbit.get('equinox') or 'J2000').strip()
    if equinox.upper() != 'J2000':
        raise ValueError(f'SBDB elements referred to the equinox {equinox!r}: only the ecliptic J2000 is read')
    matrix = np.array([[float(x) for x in row] for row in covariance['data']])
    epoch_jd = float(covariance['epoch'])
    entries = covariance.get('elements')
    if not entries:
        if abs(float(orbit['epoch']) - epoch_jd) > 1e-9:
            raise ValueError('SBDB covariance without its own elements, at another epoch than the listed elements')
        entries = orbit['elements']
    by_label = {entry['label']: float(entry['value']) for entry in entries}
    nominal = np.array([by_label[label] for label in wanted])
    nongrav = {p['name']: float(p['value']) for p in orbit.get('model_pars') or [] if p.get('kind') == 'EST'}
    obj = data.get('object', {})
    quality = {key: orbit.get(key) for key in ('orbit_id', 'condition_code', 'data_arc', 'first_obs', 'last_obs', 'soln_date')}
    return OrbitSolution(
        name=str(obj.get('des') or obj.get('fullname') or ''),
        source=source,
        coordinates='cometary',
        nominal=nominal,
        covariance=matrix[:6, :6],
        epoch_mjd=epoch_jd - MJD_TO_JD,
        time_scale='TDB',
        frame='ecliptic J2000, heliocentric',
        nongrav=nongrav,
        provenance={key: value for key, value in quality.items() if value is not None},
    )


def _mpc_orbit(data):
    """The mpc_orb record of a get-orb answer ([{...}, status] or {...})."""
    record = data[0] if isinstance(data, list) else data
    orbits = record.get('mpc_orb') if isinstance(record, dict) else None
    if not orbits:
        raise ValueError('MPC get-orb answer without an orbit (unknown designation?)')
    return orbits[0]


def _mpc_frame(system: dict) -> str:
    """The frame MPC states in `system_data` (refsys, refframe, the obliquity), refused unless it is the ecliptic
    J2000 (refframe ICRF/J2000, obliquity 84381.448"); a missing field is taken as the documented default."""
    refsys = str(system.get('refsys') or 'Ecliptic').strip()
    refframe = str(system.get('refframe') or 'ICRF').strip()
    obliquity = system.get('EclipticObliquityArcseconds')
    if refsys.lower() not in MPC_REFSYS or refframe.lower() not in MPC_REFFRAMES:
        raise ValueError(f'MPC orbit in the frame {refsys}/{refframe}: only the ecliptic J2000 (ICRF) is read, nothing is rotated')
    if obliquity not in (None, '') and abs(float(obliquity) - OBLIQUITY_J2000_ARCSEC) > OBLIQUITY_TOL_ARCSEC:
        raise ValueError(f'MPC ecliptic of obliquity {obliquity}" is not the J2000 ecliptic ({OBLIQUITY_J2000_ARCSEC}")')
    return f'{refsys} J2000 ({refframe}), heliocentric'


def parse_mpc(data, source: str = 'mpc', weak_u: Optional[int] = MPC_WEAK_U) -> OrbitSolution:
    """An MPC get-orb answer: the CAR block (heliocentric ecliptic x, y, z [AU], vx, vy, vz [AU/day]) and its covariance
    (keys covIJ, i <= j; a 7th coefficient is a non-gravitational parameter), at the orbit epoch (MJD, TDT).

    Refused when the orbit's U >= `weak_u`: MPC gives the covariance only at its orbit epoch, which for a weak orbit
    is years from its arc (2013 RZ53: a 3-day arc in 2013, the covariance at MJD 61200)."""
    data = json.loads(data) if isinstance(data, str) else data
    orbit = _mpc_orbit(data)
    block = orbit['CAR']
    names = list(block['coefficient_names'])
    if names[:6] != list(COORDINATES['cartesian'][0]):
        raise ValueError(f'MPC CAR coefficients {names[:6]}, expected x, y, z, vx, vy, vz')
    frame = _mpc_frame(orbit.get('system_data') or {})
    cov = block['covariance']
    matrix = np.array([[cov[f'cov{min(i, j)}{max(i, j)}'] for j in range(6)] for i in range(6)], dtype=float)
    stats = orbit.get('orbit_fit_statistics') or {}
    u_param = stats.get('U_param')
    designation = orbit.get('designation_data') or {}
    label = designation.get('permid') or designation.get('unpacked_primary_provisional_designation') or ''
    if weak_u is not None and u_param is not None and int(u_param) >= weak_u:
        raise ValueError(
            f"MPC orbit of {label} has U = {u_param} (>= {weak_u}): MPC gives its covariance only at the epoch MJD "
            f"{orbit['epoch_data']['epoch']}, far from its arc ({stats.get('arc_length_sel')}). Draw its clones at the fit's own "
            "epoch: source='neodys' (.eq0) or 'sbdb'"
        )
    ng = orbit.get('non_grav_booleans') or {}
    nongrav = {}
    if ng.get('non_gravs'):
        coefficients = dict(zip(names, block['coefficient_values']))
        nongrav = {name: float(coefficients[name]) for name in names[6:]} or {'model': 'non_gravs'}
    epoch = orbit['epoch_data']
    quality = {
        'U': u_param,
        'arc': stats.get('arc_length_sel'),
        'n_opp': stats.get('nopp'),
        'fitting_datetime': (orbit.get('software_data') or {}).get('fitting_datetime'),
    }
    return OrbitSolution(
        name=str(label),
        source=source,
        coordinates='cartesian',
        nominal=np.array(block['coefficient_values'][:6], dtype=float),
        covariance=matrix,
        epoch_mjd=float(epoch['epoch']),
        time_scale=str(epoch.get('timesystem') or 'TDT'),
        frame=frame,
        nongrav=nongrav,
        provenance={key: value for key, value in quality.items() if value is not None},
    )


# ----------------------------------------------------------------------------------------------- download and cache
def cache_dir() -> Path:
    return Path(c.get('CLONES_CACHE_DIR') if c.has('CLONES_CACHE_DIR') and c.get('CLONES_CACHE_DIR') else 'cache/clones')


def _ssl_context(url: str):
    if not url.startswith('https://') or os.environ.get('SSL_CERT_FILE'):
        return None
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi comes with the dependencies
        return None


def _download(url: str, body: Optional[bytes] = None) -> bytes:
    """GET `url` (with a JSON body for the MPC API), verified against certifi's CA bundle."""
    headers = {'User-Agent': USER_AGENT}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(url, data=body, headers=headers, method='GET')
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S, context=_ssl_context(url)) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise FileNotFoundError(f'{url}: not found (404)') from None
        raise


def fetch(url: str, cache_path: Path, body: Optional[bytes] = None, refresh: bool = False) -> tuple:
    """(bytes, provenance) of `url`: from `cache_path` when it is there and `refresh` is off, else downloaded and cached
    with `<file>.meta.json` (url, downloaded_at UTC, sha256, bytes). The cache keeps one orbit solution per object: an orbit
    improves with new observations, and the cached one is what a run used until `refresh=True`."""
    meta_path = cache_path.with_name(cache_path.name + '.meta.json')
    if cache_path.exists() and meta_path.exists() and not refresh:
        content = cache_path.read_bytes()
        meta = json.loads(meta_path.read_text())
        if meta.get('sha256') != hashlib.sha256(content).hexdigest():
            raise ValueError(f'{cache_path}: its sha256 differs from {meta_path}; remove both or pass refresh=True')
        return content, {**meta, 'cache_path': str(cache_path), 'from_cache': True}
    content = _download(url, body)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        'url': url,
        'request_body': body.decode() if body is not None else None,
        'downloaded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'sha256': hashlib.sha256(content).hexdigest(),
        'bytes': len(content),
    }
    part = cache_path.with_name(cache_path.name + '.part')
    part.write_bytes(content)
    os.replace(part, cache_path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return content, {**meta, 'cache_path': str(cache_path), 'from_cache': False}


def compact_designation(identifier) -> str:
    """'2013 RZ53' -> '2013RZ53', 463 -> '463' (the AstDyS/NEODyS file names)."""
    return re.sub(r'\s+', '', str(identifier).strip())


def _numbered(identifier) -> Optional[int]:
    text = str(identifier).strip()
    return int(text) if text.isdigit() else None


def solution_url(identifier, source: str, epoch: Optional[str] = None) -> tuple:
    """(url, request body or None, cache file name, product) of an object's orbit solution at a source."""
    if source in ('astdys', 'neodys'):
        epoch = epoch or DEFAULT_EPOCH[source]
        if epoch not in EPOCHS:
            raise ValueError(f"epoch is one of {', '.join(EPOCHS)}, got {epoch!r}")
        ext = OEF_EXTENSION[epoch]
        if source == 'astdys':
            number = _numbered(identifier)
            if number is None:
                raise ValueError(f"source 'astdys' has numbered asteroids only, got {identifier!r} (use 'neodys' or 'sbdb')")
            return ASTDYS_URL.format(folder=number // 1000, number=number, ext=ext), None, f'{number}.{ext}', ext
        designation = compact_designation(identifier)
        return NEODYS_URL.format(designation=designation, ext=ext), None, f'{designation}.{ext}', ext
    if epoch not in (None, 'fit') and source == 'sbdb':
        raise ValueError("source 'sbdb' has the covariance at the orbit-solution epoch only (epoch='fit')")
    if source == 'sbdb':
        sstr = urllib.parse.quote(str(identifier).strip())
        return SBDB_URL.format(sstr=sstr), None, f'{compact_designation(identifier)}.json', 'sbdb'
    if source == 'mpc':
        if epoch not in (None, 'standard'):
            raise ValueError("source 'mpc' has the covariance at the orbit epoch only (epoch='standard')")
        body = json.dumps({'desig': str(identifier).strip()}).encode()
        return MPC_URL, body, f'{compact_designation(identifier)}.json', 'get-orb'
    raise ValueError(f"source is one of {', '.join(SOURCES)}, got {source!r}")


def orbit_solution(identifier, source: str = 'astdys', epoch: Optional[str] = None, refresh: bool = False) -> OrbitSolution:
    """The orbit solution of an asteroid with its covariance, from `source` (astdys, neodys, sbdb, mpc), cached in
    CLONES_CACHE_DIR/<source>/ (downloaded once; `refresh=True` downloads it again).

    `epoch` (AstDyS/NEODyS only): 'fit' = the fit's own epoch (.eq0), 'standard' = the catalogue's epoch (.eq1, the
    covariance propagated linearly); default 'standard' for astdys, 'fit' for neodys."""
    if source not in SOURCES:
        raise ValueError(f"source is one of {', '.join(SOURCES)}, got {source!r}")
    url, body, filename, product = solution_url(identifier, source, epoch)
    content, meta = fetch(url, cache_dir() / source / filename, body=body, refresh=refresh)
    text = content.decode('utf-8', errors='replace')
    if source in ('astdys', 'neodys'):
        solution = parse_oef(text, source=source)
    elif source == 'sbdb':
        solution = parse_sbdb(text, source=source)
    else:
        solution = parse_mpc(text, source=source)
    solution.name = str(identifier).strip()
    solution.provenance = {**solution.provenance, **meta, 'product': product}
    if solution.nongrav:
        logger.warning(
            f"{solution.name}: the {source} orbit fit has non-gravitational parameters {solution.nongrav}; they are not "
            "modelled: the clones use the orbital 6x6 covariance block and the run has no non-gravitational force"
        )
    return solution
