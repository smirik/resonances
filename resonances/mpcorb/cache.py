"""The local copy of the catalogue: download, provenance, generations, the table loaded in this process.

The file is downloaded once into <cache_dir>/<catalog>/ and turned into a compact table (the file's own digits).
snapshot.json records its provenance (URL, ETag, Last-Modified, sha256 of the downloaded bytes, the standard epoch)
and names the generation's files, <sha12>-<file> and table-<sha12>-p<parser>.csv: an update writes new files and
publishes them by replacing snapshot.json, so a failure at any point leaves the previous generation whole. Nothing is
refreshed behind the user's back: `update_catalog` does it, a snapshot older than `Settings.max_age_days` only warns.
A failed update keeps the old snapshot.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import secrets
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .parser import PARSER_VERSION, arc_years, build_table, datetime_from_mjd, flags_value
from .settings import Settings, resolve

logger = logging.getLogger('mpcorb')

CATALOGS = {
    'mpcorb': 'https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz',
    'nea': 'https://minorplanetcenter.net/iau/MPCORB/NEA.txt',
    'neam00': 'https://minorplanetcenter.net/iau/MPCORB/NEAm00.txt',
    'distant': 'https://minorplanetcenter.net/iau/MPCORB/Distant.txt',
    'unusual': 'https://minorplanetcenter.net/iau/MPCORB/Unusual.txt',
}
LOAD_COLUMNS = (
    'packed',
    'number',
    'designation',
    'name',
    'epoch_mjd',
    'a',
    'e',
    'inc_deg',
    'Omega_deg',
    'omega_deg',
    'M_deg',
    'U',
    'perturbers',
    'n_opp',
    'arc',
)
# The columns only a selection reads (`query`), loaded at its first call
EXTRA_COLUMNS = ('H', 'n_obs', 'flags')
# The snapshot fields a host records as the provenance of its elements
PROVENANCE_KEYS = ('catalog', 'url', 'sha256', 'parser_version', 'etag', 'last_modified', 'rows', 'standard_epoch_mjd')
DOWNLOAD_CHUNK = 1 << 20
# A staging file (*.part) older than this is left by a process that died while updating: `_prune` deletes it
STALE_PART_S = 86400.0

_TABLES: Dict[str, object] = {}  # table file (one per sha256 and parser version) -> its loaded frame
_EXTRAS: Dict[str, object] = {}  # table file -> the frame of the columns a selection reads (`query_columns`)
_WARNED_OLD: set = set()  # sha256 of the snapshots already reported as old


def catalog_url(catalog: Optional[str] = None, settings: Optional[Settings] = None) -> Tuple[str, str]:
    """(key, url) of `catalog` (else the settings' one): a preset of CATALOGS, or a URL (http(s) or file://)."""
    catalog = resolve(settings, catalog).catalog
    if catalog in CATALOGS:
        return catalog, CATALOGS[catalog]
    if re.match(r'(https?|file)://', catalog):
        return 'url-' + hashlib.sha256(catalog.encode()).hexdigest()[:12], catalog
    raise ValueError(f"Unknown MPC catalog {catalog!r}: one of {', '.join(CATALOGS)}, or a URL")


def catalog_folder(catalog: Optional[str] = None, cache_dir: Optional[str] = None, settings: Optional[Settings] = None) -> Path:
    """<cache_dir>/<catalog key>/ (a relative cache_dir from the working directory)."""
    s = resolve(settings, catalog, cache_dir)
    key, _ = catalog_url(settings=s)
    return Path(s.cache_dir) / key


def _open_url(url: str, headers: dict, settings: Settings):
    """The response of a GET (HTTPS verified against certifi's CA bundle when it is installed)."""
    request = urllib.request.Request(url, headers={'User-Agent': settings.user_agent, **headers})
    context = None
    if url.startswith('https://') and not os.environ.get('SSL_CERT_FILE'):
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = None
    return urllib.request.urlopen(request, timeout=settings.timeout_s, context=context)


def _write_json(path: Path, data: dict):
    part = path.with_name(path.name + '.part')
    part.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(part, path)


def read_snapshot(catalog: Optional[str] = None, cache_dir: Optional[str] = None, settings: Optional[Settings] = None) -> Optional[dict]:
    """The snapshot.json of the cached catalogue, or None when there is none."""
    path = catalog_folder(catalog, cache_dir, settings) / 'snapshot.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _conditional_headers(old: dict) -> dict:
    """If-None-Match / If-Modified-Since from the cached snapshot's ETag and Last-Modified."""
    headers = {}
    if old.get('etag'):
        headers['If-None-Match'] = old['etag']
    if old.get('last_modified'):
        headers['If-Modified-Since'] = old['last_modified']
    return headers


def _save_response(response, path: Path) -> Tuple[str, int, Optional[str], Optional[str]]:
    """Stream the response into `path`: (sha256, bytes, ETag, Last-Modified); ValueError when the body is shorter or
    longer than its Content-Length."""
    digest, size = hashlib.sha256(), 0
    with response, open(path, 'wb') as out:
        expected = response.headers.get('Content-Length')
        etag, last_modified = response.headers.get('ETag'), response.headers.get('Last-Modified')
        while chunk := response.read(DOWNLOAD_CHUNK):
            digest.update(chunk)
            size += len(chunk)
            out.write(chunk)
    if expected is not None and int(expected) != size:
        raise ValueError(f"incomplete download: {size} of {expected} bytes")
    return digest.hexdigest(), size, etag, last_modified


def update_catalog(
    catalog: Optional[str] = None, cache_dir: Optional[str] = None, force: bool = False, settings: Optional[Settings] = None
) -> dict:
    """Download the catalogue (a conditional GET on the cached ETag/Last-Modified unless `force`), check and parse it,
    and replace the cached snapshot only when everything succeeded. Returns the snapshot (snapshot.json)."""
    s = resolve(settings, catalog, cache_dir)
    key, url = catalog_url(settings=s)
    folder = catalog_folder(settings=s)
    folder.mkdir(parents=True, exist_ok=True)
    old = read_snapshot(settings=s)
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

    fresh = old is not None and not force and (folder / str(old.get('table_file'))).exists()
    headers = _conditional_headers(old) if fresh else {}
    basename = Path(urllib.parse.urlparse(url).path).name or 'catalog'
    parts = []  # the staging files, created only once there is a body to write
    try:
        try:
            response = _open_url(url, headers, s)
        except urllib.error.HTTPError as error:
            if error.code == 304 and old is not None:
                logger.info(f"MPC catalogue {key} not modified since {old.get('last_modified')}")
                old['checked_utc'] = now
                _write_json(folder / 'snapshot.json', old)
                return old
            raise
        parts.extend((_staging(folder), _staging(folder)))
        raw_part, table_part = parts
        sha256, size, etag, last_modified = _save_response(response, raw_part)
        stats = build_table(raw_part, table_part)
        raw_file, table_file = f"{sha256[:12]}-{basename}", _table_name(sha256)
        os.replace(raw_part, folder / raw_file)
        os.replace(table_part, folder / table_file)
        snapshot_dict = {
            'catalog': key,
            'url': url,
            'raw_file': raw_file,
            'table_file': table_file,
            'etag': etag,
            'last_modified': last_modified,
            'sha256': sha256,
            'size_bytes': size,
            'downloaded_utc': now,
            'checked_utc': now,
            'parser_version': PARSER_VERSION,
            **stats,
        }
        _write_json(folder / 'snapshot.json', snapshot_dict)  # the publication
    except Exception as error:
        for part in parts:
            part.unlink(missing_ok=True)
        kept = 'the cached snapshot is kept' if old is not None else 'there is no cached snapshot'
        raise RuntimeError(f"MPC catalogue {key} from {url} could not be updated ({error}); {kept}") from error

    _prune(folder, snapshot_dict, old)
    if stats['unparsed_lines']:
        logger.warning(f"MPC catalogue {key}: {stats['unparsed_lines']} record lines could not be read and are left out")
    logger.info(f"MPC catalogue {key}: {stats['rows']} rows, standard epoch MJD {stats['standard_epoch_mjd']}, sha256 {sha256}")
    return snapshot_dict


def _table_name(sha256: str) -> str:
    return f"table-{sha256[:12]}-p{PARSER_VERSION}.csv"


def _staging(folder: Path) -> Path:
    """A new, unique file to write into before it is renamed (two updates never share one). Created with mode
    0o666 less the umask, as any file the user writes: a cache folder shared between users stays readable (mkstemp
    would give 0o600)."""
    while True:
        path = folder / f"tmp{secrets.token_hex(6)}.part"
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
            return path
        except FileExistsError:
            continue


def _prune(folder: Path, current: dict, previous: Optional[dict], now: Optional[float] = None):
    """Delete the files of generations older than the previous one (a process may still be reading that one), and
    staging files (*.part) older than STALE_PART_S: those of a process that died while updating."""
    keep = {'snapshot.json'}
    for snap in (current, previous or {}):
        keep.update(str(snap.get(name)) for name in ('raw_file', 'table_file'))
    now = time.time() if now is None else now
    for path in folder.iterdir():
        if path.name in keep:
            continue
        if path.name.endswith('.part'):
            try:
                if now - path.stat().st_mtime <= STALE_PART_S:
                    continue  # another update may be writing it
            except FileNotFoundError:
                continue
        path.unlink(missing_ok=True)


def snapshot_age_days(snapshot_dict: dict, now: Optional[datetime.datetime] = None) -> float:
    """Days since the data's Last-Modified (else since the download)."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if snapshot_dict.get('last_modified'):
        stamp = parsedate_to_datetime(snapshot_dict['last_modified'])
    else:
        stamp = datetime.datetime.fromisoformat(snapshot_dict['downloaded_utc'])
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=datetime.timezone.utc)
    return (now - stamp).total_seconds() / 86400.0


def snapshot(catalog: Optional[str] = None, cache_dir: Optional[str] = None, settings: Optional[Settings] = None) -> dict:
    """The cached snapshot, downloaded on first use (never refreshed here). Rebuilds a table written by another parser
    version from the raw file, warns when the data are older than `max_age_days`, and refuses a snapshot whose
    sha256 differs from the settings' `sha256` when that is set."""
    s = resolve(settings, catalog, cache_dir)
    current = read_snapshot(settings=s)
    if current is None:
        key, url = catalog_url(settings=s)
        logger.info(f"No cached MPC catalogue {key}: downloading {url}")
        current = update_catalog(settings=s)
    folder = catalog_folder(settings=s)
    if current.get('parser_version') != PARSER_VERSION or not (folder / str(current.get('table_file'))).exists():
        part = _staging(folder)
        try:
            stats = build_table(folder / current['raw_file'], part)
            os.replace(part, folder / _table_name(current['sha256']))
        finally:
            part.unlink(missing_ok=True)
        current.update(stats, parser_version=PARSER_VERSION, table_file=_table_name(current['sha256']))
        _write_json(folder / 'snapshot.json', current)

    pin = s.sha256
    if pin and pin.lower() != current['sha256']:
        raise ValueError(
            f"The sha256 pin {pin} (Settings.sha256, MPC_SHA256 in resonances) does not match the cached MPC catalogue "
            f"{current['catalog']}, sha256 {current['sha256']} (Last-Modified {current.get('last_modified')}): put the "
            "pinned file in place or clear the pin"
        )
    age = snapshot_age_days(current)
    if s.max_age_days > 0 and age > s.max_age_days and current['sha256'] not in _WARNED_OLD:
        _WARNED_OLD.add(current['sha256'])  # once per snapshot and process, not at every body
        logger.warning(
            f"The MPC catalogue {current['catalog']} is {age:.0f} days old (Last-Modified {current.get('last_modified')}): "
            f"run {s.update_hint} for a fresh one"
        )
    return current


def provenance(snapshot_dict: dict) -> dict:
    """The fields of a snapshot a host records with the elements it took (PROVENANCE_KEYS)."""
    return {key: snapshot_dict.get(key) for key in PROVENANCE_KEYS}


def catalog_datetime(snapshot_dict: dict) -> datetime.datetime:
    """The standard epoch of the catalogue (the epoch of most rows) as a naive datetime holding its TT calendar value."""
    return datetime_from_mjd(snapshot_dict['standard_epoch_mjd'])


def _table_path(s: Settings, snapshot_dict: Optional[dict]) -> Tuple[dict, str]:
    snapshot_dict = snapshot_dict or snapshot(settings=s)  # downloaded on first use, rebuilt by a new parser
    return snapshot_dict, str(catalog_folder(settings=s) / snapshot_dict['table_file'])


def load_table(
    catalog: Optional[str] = None,
    cache_dir: Optional[str] = None,
    snapshot_dict: Optional[dict] = None,
    settings: Optional[Settings] = None,
):
    """The table of `snapshot_dict` (else of the cached snapshot) as a DataFrame of LOAD_COLUMNS, cached in this
    process under its generation's file name: rows and provenance always come from one generation."""
    import pandas as pd

    _, path = _table_path(resolve(settings, catalog, cache_dir), snapshot_dict)
    if path not in _TABLES:
        dtypes = {'packed': str, 'designation': str, 'name': str, 'number': 'int64'}
        dtypes.update({column: 'category' for column in ('U', 'perturbers', 'n_opp', 'arc')})
        frame = pd.read_csv(path, usecols=list(LOAD_COLUMNS), dtype=dtypes, keep_default_na=False, na_values={})
        frame['name_key'] = frame['name'].str.casefold().astype('category')  # ~26k names among 1.6M rows
        _TABLES.clear()  # one catalogue at a time
        _EXTRAS.clear()
        _TABLES[path] = frame
    return _TABLES[path]


def _parsed_categories(column, parse):
    """`parse` applied once per distinct value of a text column (a few thousand among 1.6M rows)."""
    codes = column.astype('category')
    values = [parse(value) for value in codes.cat.categories]
    lookup = np.asarray(values + [parse('')], dtype=float)  # code -1 (missing) reads the last entry
    return lookup[codes.cat.codes.to_numpy()]


def query_columns(
    catalog: Optional[str] = None,
    cache_dir: Optional[str] = None,
    snapshot_dict: Optional[dict] = None,
    settings: Optional[Settings] = None,
):
    """The columns a selection reads besides the table (row for row): H, n_obs, n_opp (nan when blank), arc_years
    (`parser.arc_years`), flags (the file's text) and flags_value (its integer), U_value (nan for a letter or a
    blank U). Loaded at the first call and cached with the table."""
    import pandas as pd

    s = resolve(settings, catalog, cache_dir)
    snapshot_dict, path = _table_path(s, snapshot_dict)
    table = load_table(snapshot_dict=snapshot_dict, settings=s)
    if path not in _EXTRAS:
        raw = pd.read_csv(path, usecols=list(EXTRA_COLUMNS), dtype=str, keep_default_na=False, na_values={})
        extra = pd.DataFrame(
            {
                'H': pd.to_numeric(raw['H'], errors='coerce'),
                'n_obs': pd.to_numeric(raw['n_obs'], errors='coerce'),
                'n_opp': pd.to_numeric(table['n_opp'].astype(str), errors='coerce'),
                'arc_years': _parsed_categories(table['arc'], arc_years),
                'flags': raw['flags'].astype('category'),
                'flags_value': _parsed_categories(raw['flags'], flags_value).astype('int64'),
                'U_value': pd.to_numeric(table['U'].astype(str), errors='coerce'),
            },
            index=table.index,
        )
        _EXTRAS[path] = extra
    return _EXTRAS[path]


def clear_cache():
    """Forget the tables loaded in this process (and which snapshots were reported as old)."""
    _TABLES.clear()
    _EXTRAS.clear()
    _WARNED_OLD.clear()
