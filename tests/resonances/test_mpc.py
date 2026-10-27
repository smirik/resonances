"""DATA_SOURCE=mpc and resonances.mpcorb: designations, packed epochs, the MPCORB parser, the cache, the epoch policy
(propagate by default), the elements of a Body, selections by elements and flags (query), the CLI, and that the
mpcorb folder works as a package of its own.

The fixture tests/fixtures/mpc/MPCORB_sample.DAT holds real lines of MPCORB.DAT (Last-Modified 2026-10-25 12:59 GMT):
its header, numbered objects (1, 2, 4, 10, 463, Sedna, 101429 = A1429, 630872 = ~02pM), multi-opposition and
extended designations (2001 ND13, 2012 TG396, _EH0000, _PO00fh, 1995 YY3 with U = E), orbits at other epochs
(2008 TC3, 2022 LX1) and two one-opposition orbits of the last section (1942 RH, 2021 AQ). Expected values come from
other sources: MPC's documentation of the packed forms, astropy for the dates, AstDyS allnum.cat and MPC's get-orb.
"""

import ast
import datetime
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import pytest
import rebound
from astropy.time import Time
from click.testing import CliRunner

import resonances
import resonances.simulation.propagate as propagate_module
from resonances import mpcorb
from resonances.mpcorb import cache as mpc_cache
from resonances.cli.main import cli
from resonances.simulation.parallel_executor import _run_batch_worker
from resonances.simulation.serializer import SimulationSerializer
from resonances.simulation.simulation import Simulation
from resonances.simulation.state_manager import StateManager

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'mpc' / 'MPCORB_sample.DAT'
FIXTURE_ROWS = 17


# ----------------------------------------------------------------------------------------------- designations
# https://minorplanetcenter.net/iau/info/PackedDes.html (and docs-public packed-designations.md); 630872 = '~02pM' is
# read off the readable column of MPCORB ("(630872) 2006 SX27")
@pytest.mark.parametrize(
    'number, packed',
    [
        (3202, '03202'),
        (50000, '50000'),
        (100345, 'A0345'),
        (360017, 'a0017'),
        (203289, 'K3289'),
        (620000, '~0000'),
        (620061, '~000z'),
        (3140113, '~AZaz'),
        (15396335, '~zzzz'),
        (630872, '~02pM'),
    ],
)
def test_packed_numbers(number, packed):
    assert mpcorb.pack_number(number) == packed
    assert mpcorb.unpack_number(packed) == number


@pytest.mark.parametrize('text', ['K01N13D', '00000', '0463', 'Ceres', '~02p!'])
def test_not_packed_numbers(text):
    assert mpcorb.unpack_number(text) is None


@pytest.mark.parametrize(
    'packed, designation',
    [
        # MPC's provisional-designations page, original scheme
        ('J95X00A', '1995 XA'),
        ('J95X01L', '1995 XL1'),
        ('J95F13B', '1995 FB13'),
        ('J98SA8Q', '1998 SQ108'),
        ('J98SC7V', '1998 SV127'),
        ('J98SG2S', '1998 SS162'),
        ('K99AJ3Z', '2099 AZ193'),
        ('K08Aa0A', '2008 AA360'),
        ('K07Tf8A', '2007 TA418'),
        ('K23B00A', '2023 BA'),
        ('K24C03Z', '2024 CZ3'),
        ('K25Dz9Z', '2025 DZ619'),
        ('K26Cz9Z', '2026 CZ619'),  # the 15,500th of the half-month: still the original scheme
        # extended scheme (the 15,501st, 154,775th and last of 2026 C)
        ('_QC0000', '2026 CA620'),
        ('_QC0aEM', '2026 CZ6190'),
        ('_QCzzzz', '2026 CL591673'),
        # surveys
        ('PLS2040', '2040 P-L'),
        ('T1S3138', '3138 T-1'),
        ('T2S1010', '1010 T-2'),
        ('T3S4101', '4101 T-3'),
        # get-orb designation_data of (463) and (101429); before 1925 the year is written A9xx
        ('J00U00K', 'A900 UK'),
        ('J04O00A', 'A904 OA'),  # by the same rule: year 19xx is J, before 1925 written A9xx
        ('J26T00C', '1926 TC'),
        ('J98V31F', '1998 VF31'),
        ('K01N13D', '2001 ND13'),
        # readable column of MPCORB
        ('_PO00fh', '2025 OL723'),
        ('_EH0000', '2014 HA620'),
        ('K12Td6G', '2012 TG396'),
    ],
)
def test_packed_provisional_designations(packed, designation):
    assert mpcorb.unpack_provisional(packed) == designation
    assert mpcorb.pack_provisional(designation) == packed


@pytest.mark.parametrize(
    'text, normal',
    [('2001ND13', '2001 ND13'), ('2001 ND13', '2001 ND13'), ('1900 UK', 'A900 UK'), ('4150T-3', '4150 T-3'), ('2001 ND013', '2001 ND13')],
)
def test_normalised_designations(text, normal):
    """The AstDyS spelling (no space) and a modern year before 1925 read as MPC writes them."""
    assert mpcorb.normalise_provisional(text) == normal


@pytest.mark.parametrize('text', ['Ceres', '2001 ND0', '2001 IA', '463', '2001 N13'])
def test_not_provisional_designations(text):
    assert mpcorb.normalise_provisional(text) is None
    assert mpcorb.pack_provisional(text) is None


# ----------------------------------------------------------------------------------------------- epochs
@pytest.mark.parametrize(
    'packed, iso, fraction',
    [
        # https://minorplanetcenter.net/iau/info/PackedDates.html
        ('J9611', '1996-01-01', 0.0),
        ('J961A', '1996-01-10', 0.0),
        ('J969U', '1996-09-30', 0.0),
        ('J96A1', '1996-10-01', 0.0),
        ('K01AM', '2001-10-22', 0.0),
        ('J981I73', '1998-01-18', 0.73),
        ('K01AM138303', '2001-10-22', 0.138303),
        # epochs of the fixture
        ('K2669', '2026-06-09', 0.0),
        ('K25BL', '2025-11-21', 0.0),
        ('K20CH', '2020-12-17', 0.0),
        ('J4299', '1942-09-09', 0.0),
        ('K08A7', '2008-10-07', 0.0),
    ],
)
def test_packed_epochs_against_astropy(packed, iso, fraction):
    """MJD of the TT date by astropy, not by the calendar arithmetic of the package."""
    expected = Time(iso, scale='tt').mjd + fraction
    assert mpcorb.unpack_epoch(packed) == pytest.approx(expected, abs=1e-9)
    assert mpcorb.unpack_epoch(packed) + 2400000.5 == pytest.approx(Time(iso, scale='tt').jd + fraction, abs=1e-8)


def test_standard_epoch_is_the_astdys_epoch():
    """K2669 is MJD 61200 TT, the epoch of AstDyS allnum.cat and of get-orb for (463) ({"epoch": 61200.0, "timeform":
    "MJD", "timesystem": "TDT"}), and its calendar value 2026-06-09 0h is what the planets are built at."""
    assert mpcorb.unpack_epoch('K2669') == 61200.0
    assert mpcorb.datetime_from_mjd(61200.0) == datetime.datetime(2026, 6, 9)
    assert mpcorb.datetime_from_mjd(61200.25) == datetime.datetime(2026, 6, 9, 6)


@pytest.mark.parametrize('text', ['K26D0', 'X2669', 'K26DW', 'K2X69', 'K266', 'K2669x'])
def test_bad_packed_epochs_raise(text):
    with pytest.raises(ValueError):
        mpcorb.unpack_epoch(text)


# ----------------------------------------------------------------------------------------------- parser
def _fixture_records():
    with open(FIXTURE) as source:
        return [record for record in map(mpcorb.parse_line, source) if record is not None]


def _record(packed):
    return next(record for record in _fixture_records() if record['packed'] == packed)


def test_parser_reads_every_record_and_only_them():
    records = _fixture_records()
    assert len(records) == FIXTURE_ROWS
    assert [r['packed'] for r in records][:3] == ['00001', '00002', '00004']


def test_parser_fields_of_a_numbered_and_a_one_opposition_row():
    lola = _record('00463')
    assert (lola['number'], lola['name'], lola['designation'], lola['U'], lola['perturbers']) == (463, 'Lola', '', '0', 'M-v 3Ek')
    assert (lola['a'], lola['e'], lola['inc_deg'], lola['Omega_deg'], lola['omega_deg'], lola['M_deg']) == (
        2.3986620,
        0.2194729,
        13.53827,
        36.48765,
        329.45975,
        310.21988,
    )
    rh = _record('J42R00H')
    assert (rh['number'], rh['designation'], rh['U'], rh['perturbers'], rh['epoch_mjd']) == (0, '1942 RH', '', '', 30611.0)
    unnamed = _record('A1429')
    assert (unnamed['number'], unnamed['name'], unnamed['designation']) == (101429, '', '1998 VF31')
    assert _record('J95Y03Y')['U'] == 'E'


# AstDyS allnum.cat (cache/allnum.cat, 2026-10-07), the same objects at the same epoch MJD 61200: a, e, i, Omega,
# omega, M (deg). A different fit: they agree with MPC to |da| <= 1.1e-7 AU, |de| <= 3e-7 and angles <= 5.5e-4 deg.
ASTDYS_61200 = {
    '00001': (2.7655526074353367, 7.9692329086296324e-02, 10.588028739417092, 80.248639494144257, 73.294172914867460, 274.41935409749186),
    '00002': (2.7695590179422447, 2.3070015488221021e-01, 34.932789963971651, 172.88661612209344, 310.96992210812908, 254.24964198662499),
    '00004': (2.3613659708769341, 9.0203724208184538e-02, 7.1439267451811324, 103.70133850484119, 151.46856273625491, 81.190189453273007),
    '00010': (3.1509741119358612, 1.0670935343454307e-01, 3.8295141006880740, 283.11975617926498, 312.42446148695404, 252.03428043870167),
    '00463': (2.3986620775145058, 2.1947277100820548e-01, 13.538290315695448, 36.487495907783284, 329.45992599372408, 310.21982550534904),
    'A1429': (1.5242076383173770, 1.0026706688930165e-01, 31.296024318074721, 221.27812647677507, 310.60670761592399, 131.77702537505803),
}


def _angle_difference(x, y):
    return (x - y + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize('packed', sorted(ASTDYS_61200))
def test_parser_against_astdys(packed):
    record = _record(packed)
    a, e, inc, node, peri, mean = ASTDYS_61200[packed]
    assert record['epoch_mjd'] == 61200.0
    assert record['a'] == pytest.approx(a, abs=2e-7)
    assert record['e'] == pytest.approx(e, abs=3e-7)
    assert abs(_angle_difference(record['inc_deg'], inc)) < 1e-4
    for mine, theirs in ((record['Omega_deg'], node), (record['omega_deg'], peri), (record['M_deg'], mean)):
        assert abs(_angle_difference(mine, theirs)) < 1e-3


def test_parser_against_get_orb_for_463():
    """get-orb KEP of (463) (MPC's orbit API, full precision, the same fit) against its MPCORB line."""
    record = _record('00463')
    a, e, inc, node, peri, mean = (
        2.3986620503012195,
        0.219472884401004,
        13.5382707846348,
        36.4876473431163,
        329.4597515580392,
        310.2198828196205,
    )
    assert record['a'] == pytest.approx(a, abs=6e-8)
    assert record['e'] == pytest.approx(e, abs=6e-8)
    for mine, theirs in ((record['inc_deg'], inc), (record['Omega_deg'], node), (record['omega_deg'], peri), (record['M_deg'], mean)):
        assert abs(_angle_difference(mine, theirs)) < 1e-5


def test_every_row_has_mu_k2():
    """n = k a^-1.5 within the file's rounding on every row (Sedna's n = 0.00007774 included)."""
    assert max(mpcorb.mu_ratio(record) for record in _fixture_records()) <= 1.0


# ----------------------------------------------------------------------------------------------- cache
@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """The fixture gzipped and served by a file:// URL; MPC_CACHE_DIR in tmp_path. Returns (url, gz path)."""
    source = tmp_path / 'served' / 'MPCORB.DAT.gz'
    source.parent.mkdir()
    with open(FIXTURE, 'rb') as raw, gzip.open(source, 'wb') as out:
        shutil.copyfileobj(raw, out)
    url = source.as_uri()
    monkeypatch.setitem(resonances.config.config, 'MPC_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setitem(resonances.config.config, 'MPC_CATALOG', url)
    monkeypatch.setitem(resonances.config.config, 'MPC_SHA256', '')
    monkeypatch.setitem(resonances.config.config, 'MPC_MAX_AGE_DAYS', '0')
    monkeypatch.setitem(resonances.config.config, 'MPC_EPOCH_POLICY', 'propagate')
    monkeypatch.setitem(resonances.config.config, 'MPC_PROPAGATE_MAX_DAYS', '')
    mpcorb.clear_cache()
    yield url, source
    mpcorb.clear_cache()


def test_first_use_downloads_and_records_provenance(catalog):
    url, source = catalog
    snapshot = mpcorb.snapshot()
    assert snapshot['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert snapshot['url'] == url
    assert snapshot['size_bytes'] == source.stat().st_size
    assert snapshot['rows'] == FIXTURE_ROWS
    assert snapshot['standard_epoch_mjd'] == 61200.0
    # 2008 TC3, 2022 LX1, 1942 RH, 2021 AQ are at other epochs; the last two have no perturbers
    assert (snapshot['other_epoch_rows'], snapshot['no_perturber_rows'], snapshot['unparsed_lines']) == (4, 2, 0)
    assert snapshot['last_modified']  # urllib's file handler sends the file's mtime
    sha12 = snapshot['sha256'][:12]
    assert (snapshot['raw_file'], snapshot['table_file']) == (f'{sha12}-MPCORB.DAT.gz', f'table-{sha12}-p{mpcorb.PARSER_VERSION}.csv')
    folder = mpcorb.catalog_folder()
    assert sorted(p.name for p in folder.iterdir()) == sorted([snapshot['raw_file'], 'snapshot.json', snapshot['table_file']])
    assert json.loads((folder / 'snapshot.json').read_text()) == snapshot
    assert mpcorb.provenance(snapshot)['parser_version'] == mpcorb.PARSER_VERSION


def test_corrupt_download_keeps_the_old_snapshot(catalog):
    _, source = catalog
    before = mpcorb.snapshot()
    table_before = (mpcorb.catalog_folder() / before['table_file']).read_bytes()
    data = source.read_bytes()
    source.write_bytes(data[: len(data) // 2])  # a truncated gzip
    with pytest.raises(RuntimeError, match='cached snapshot is kept'):
        mpcorb.update_catalog(force=True)
    assert mpcorb.read_snapshot() == before
    assert (mpcorb.catalog_folder() / before['table_file']).read_bytes() == table_before
    assert not list(mpcorb.catalog_folder().glob('*.part'))


def _changed_lola(source):
    """The served file with 463's M changed from 310.21988 to 123.45678 deg (a new generation)."""
    text = FIXTURE.read_text().replace('K2669 310.21988', 'K2669 123.45678')
    with gzip.open(source, 'wt') as out:
        out.write(text)


def test_failed_publication_keeps_the_old_generation(catalog, monkeypatch):
    """An update that dies when it publishes snapshot.json leaves the old snapshot, its files and its pin working."""
    _, source = catalog
    before = mpcorb.snapshot()
    _changed_lola(source)
    write_json = mpc_cache._write_json

    def failing(path, data):
        if path.name == 'snapshot.json':
            raise OSError('disk full')
        write_json(path, data)

    monkeypatch.setattr(mpc_cache, '_write_json', failing)
    with pytest.raises(RuntimeError, match='cached snapshot is kept'):
        mpcorb.update_catalog(force=True)
    monkeypatch.setattr(mpc_cache, '_write_json', write_json)
    monkeypatch.setitem(resonances.config.config, 'MPC_SHA256', before['sha256'])
    mpcorb.clear_cache()
    sim = Simulation(source='mpc')
    sim.add_body(463, '4J-2S-1')
    assert math.degrees(sim.bodies[0].initial_data['M']) == pytest.approx(310.21988, abs=1e-9)
    assert sim.bodies[0].initial_data['catalog_sha256'] == before['sha256']


def test_rows_and_provenance_come_from_one_generation(catalog):
    """A table cached in this process is not served under a newer snapshot's sha256 (an update by another process)."""
    _, source = catalog
    first = Simulation(source='mpc')
    first.add_body(463, '4J-2S-1')  # loads and caches the first table
    _changed_lola(source)
    after = mpcorb.update_catalog(force=True)
    second = Simulation(source='mpc')
    second.add_body(463, '4J-2S-1')
    elem = second.bodies[0].initial_data
    assert elem['catalog_sha256'] == after['sha256'] != first.bodies[0].initial_data['catalog_sha256']
    assert math.degrees(elem['M']) == pytest.approx(123.45678, abs=1e-9)
    with pytest.raises(ValueError, match='updated in between'):
        first.add_body(463, '4J-2S-1')


def test_older_generations_are_pruned(catalog):
    """After two updates the folder holds the current and the previous generation only."""
    _, source = catalog
    first = mpcorb.snapshot()
    _changed_lola(source)
    second = mpcorb.update_catalog(force=True)
    source.write_bytes(gzip.compress(FIXTURE.read_bytes().replace(b'K2669 310.21988', b'K2669 200.00000')))
    third = mpcorb.update_catalog(force=True)
    names = {p.name for p in mpcorb.catalog_folder().iterdir()}
    assert names == {'snapshot.json', second['raw_file'], second['table_file'], third['raw_file'], third['table_file']}
    assert first['raw_file'] not in names


def test_non_finite_row_is_left_out(catalog):
    """'nan' parses as a float: such a record is not read (and is counted), never turned into elements."""
    _, source = catalog
    source.write_bytes(gzip.compress(FIXTURE.read_bytes().replace(b'K2669 310.21988', b'K2669       nan')))
    snapshot = mpcorb.update_catalog(force=True)
    assert (snapshot['rows'], snapshot['unparsed_lines']) == (FIXTURE_ROWS - 1, 1)
    assert mpcorb.lookup([463]) == {}


def test_not_a_catalogue_is_refused(catalog, tmp_path, monkeypatch):
    bad = tmp_path / 'served' / 'notes.txt'
    bad.write_text("not an orbit file\n" * 10)
    monkeypatch.setitem(resonances.config.config, 'MPC_CATALOG', bad.as_uri())
    with pytest.raises(RuntimeError, match='no MPCORB-format record'):
        mpcorb.update_catalog()
    assert mpcorb.read_snapshot() is None


class _Response:
    """A stand-in for an HTTP response: headers and a body read in chunks."""

    def __init__(self, body: bytes, headers: dict):
        self.headers, self._body = headers, body

    def read(self, size=-1):
        chunk, self._body = (self._body, b'') if size < 0 else (self._body[:size], self._body[size:])
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_etag_hit_skips_the_download(catalog, monkeypatch):
    _, source = catalog
    body = source.read_bytes()
    served = {'ETag': '"59a6103-65ea9c7186bbb"', 'Last-Modified': 'Sun, 25 Oct 2026 12:59:03 GMT', 'Content-Length': str(len(body))}
    requests = []

    def first(url, headers, settings):
        requests.append(headers)
        return _Response(body, served)

    monkeypatch.setattr(mpc_cache, '_open_url', first)
    snapshot = mpcorb.update_catalog()
    assert (snapshot['etag'], snapshot['last_modified']) == (served['ETag'], served['Last-Modified'])

    def not_modified(url, headers, settings):
        requests.append(headers)
        raise urllib.error.HTTPError(url, 304, 'Not Modified', {}, None)

    monkeypatch.setattr(mpc_cache, '_open_url', not_modified)
    again = mpcorb.update_catalog()
    assert requests[0] == {}
    assert requests[1] == {'If-None-Match': served['ETag'], 'If-Modified-Since': served['Last-Modified']}
    assert again['sha256'] == snapshot['sha256'] and again['downloaded_utc'] == snapshot['downloaded_utc']
    for _ in range(3):
        mpcorb.update_catalog()
    names = {p.name for p in mpcorb.catalog_folder().iterdir()}
    assert names == {'snapshot.json', snapshot['raw_file'], snapshot['table_file']}  # no staging file left by a 304


def test_stale_staging_files_are_pruned(catalog):
    """A *.part file of a process that died while updating goes at the next update once older than STALE_PART_S;
    a recent one (another update may be writing it) stays."""
    mpcorb.snapshot()
    folder = mpcorb.catalog_folder()
    stale, fresh = folder / 'tmpdead00000.part', folder / 'tmplive00000.part'
    stale.write_bytes(b'x' * 100)
    fresh.write_bytes(b'y')
    old = time.time() - mpcorb.STALE_PART_S - 60
    os.utime(stale, (old, old))
    mpcorb.update_catalog(force=True)
    assert not stale.exists() and fresh.exists()


def test_cached_files_follow_the_umask(catalog):
    """The raw file and the table get 0o666 less the umask, like snapshot.json: a cache folder shared by users stays
    readable (mkstemp gave 0o600)."""
    previous = os.umask(0o022)
    try:
        snapshot = mpcorb.update_catalog(force=True)
    finally:
        os.umask(previous)
    folder = mpcorb.catalog_folder()
    for name in (snapshot['raw_file'], snapshot['table_file'], 'snapshot.json'):
        assert (folder / name).stat().st_mode & 0o777 == 0o644, name


def test_short_download_is_refused(catalog, monkeypatch):
    _, source = catalog
    body = source.read_bytes()
    monkeypatch.setattr(mpc_cache, '_open_url', lambda url, headers, settings: _Response(body[:-10], {'Content-Length': str(len(body))}))
    with pytest.raises(RuntimeError, match='incomplete download'):
        mpcorb.update_catalog()


def test_sha256_pin(catalog, monkeypatch):
    sha = mpcorb.snapshot()['sha256']
    monkeypatch.setitem(resonances.config.config, 'MPC_SHA256', sha)
    assert mpcorb.snapshot()['sha256'] == sha
    monkeypatch.setitem(resonances.config.config, 'MPC_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='MPC_SHA256'):
        mpcorb.snapshot()


def test_old_catalogue_warns(catalog, monkeypatch):
    _, source = catalog
    forty_days_ago = time.time() - 40 * 86400
    os.utime(source, (forty_days_ago, forty_days_ago))
    warnings = []
    monkeypatch.setattr(mpc_cache.logger, 'warning', warnings.append)
    monkeypatch.setitem(resonances.config.config, 'MPC_MAX_AGE_DAYS', '30')
    mpcorb.snapshot()
    mpcorb.snapshot()  # once per snapshot, not at every body
    assert len(warnings) == 1 and 'mpc-update' in warnings[0] and '40 days old' in warnings[0]


def test_table_of_another_parser_version_is_rebuilt_offline(catalog, monkeypatch):
    mpcorb.snapshot()
    folder = mpcorb.catalog_folder()
    snapshot = json.loads((folder / 'snapshot.json').read_text())
    snapshot['parser_version'] = 0
    (folder / 'snapshot.json').write_text(json.dumps(snapshot))
    (folder / snapshot['table_file']).unlink()
    monkeypatch.setattr(mpc_cache, '_open_url', lambda url, headers, settings: pytest.fail('no download expected'))
    assert mpcorb.snapshot()['parser_version'] == mpcorb.PARSER_VERSION
    assert len(mpcorb.load_table()) == FIXTURE_ROWS


# ----------------------------------------------------------------------------------------------- lookup
@pytest.mark.parametrize(
    'identifier, packed',
    [
        (463, '00463'),
        ('463', '00463'),
        ('(463)', '00463'),
        ('00463', '00463'),
        ('Lola', '00463'),
        ('lola', '00463'),
        ('(463) Lola', '00463'),
        (101429, 'A1429'),
        ('A1429', 'A1429'),
        ('1998 VF31', 'A1429'),  # the principal designation of a numbered object without a name
        ('1998VF31', 'A1429'),
        ('~02pM', '~02pM'),
        (630872, '~02pM'),
        ('2006 SX27', '~02pM'),
        ('2001 ND13', 'K01N13D'),
        ('2001ND13', 'K01N13D'),
        ('K01N13D', 'K01N13D'),
        ('_PO00fh', '_PO00fh'),
        ('2025 OL723', '_PO00fh'),
        ('Sedna', '90377'),
    ],
)
def test_lookup_by_any_form(catalog, identifier, packed):
    found = mpcorb.lookup([identifier])
    assert list(found) == [str(identifier)]
    assert found[str(identifier)]['packed'] == packed


def test_lookup_misses_are_left_out(catalog):
    assert mpcorb.lookup(['2040 P-L', 99999, 'No Such Name', 463]).keys() == {'463'}


@pytest.mark.parametrize('identifier', ['2001 nd13', '2001nd13', '2025 ol723', '1998 vf31'])
def test_lookup_of_a_lower_case_designation(catalog, identifier):
    expected = mpcorb.lookup([identifier.upper()])[identifier.upper()]['packed']
    assert mpcorb.lookup([identifier])[identifier]['packed'] == expected


@pytest.mark.parametrize('identifier', ['', '   '])
def test_empty_identifier_raises(catalog, identifier):
    """An empty identifier would match every row without a designation or a name."""
    with pytest.raises(ValueError, match='Not an asteroid identifier'):
        mpcorb.lookup([identifier])


def test_one_bulk_lookup_equals_single_lookups(catalog):
    """Every row of the fixture by a different form in one call: the same rows as one call per identifier."""
    identifiers = [1, 'Pallas', '(4)', '00010', 'lola', 'A1429', '~02pM', 'Sedna', 'K01N13D', '2012 TG396', '_EH0000',
                   '2025OL723', '1995 YY3', '2008 TC3', '2022 LX1', '1942 RH', '2021 AQ', 'no such body']  # fmt: skip
    bulk = mpcorb.lookup(identifiers)
    single = {str(ident): row for ident in identifiers for row in mpcorb.lookup([ident]).values()}
    assert bulk == single
    assert len({row['packed'] for row in bulk.values()}) == FIXTURE_ROWS


def _numbered_copies(path, count):
    """A catalogue of `count` copies of 463's line numbered 1..count (no names)."""
    lola = next(line for line in FIXTURE.read_text().splitlines(keepends=True) if line.startswith('00463'))
    with open(path, 'w') as out:
        for number in range(1, count + 1):
            out.write(mpcorb.pack_number(number).ljust(7) + lola[7:166] + f'({number})'.ljust(28) + lola[194:])


def test_bulk_lookup_cost_grows_linearly(catalog, tmp_path, monkeypatch):
    """A find over many bodies: 20,000 numbers on a 20,000-row table. One isin per identifier over the candidates
    (the first version) took about 20 s here; one pass per column takes well under a second."""
    path = tmp_path / 'served' / 'numbered.txt'
    _numbered_copies(path, 20000)
    monkeypatch.setitem(resonances.config.config, 'MPC_CATALOG', path.as_uri())
    mpcorb.load_table()
    start = time.perf_counter()
    found = mpcorb.lookup(list(range(1, 20001)) + ['no such body'])
    elapsed = time.perf_counter() - start
    assert len(found) == 20000 and found['20000']['packed'] == '20000' and found['12345']['number'] == 12345
    assert elapsed < 5.0


def test_ambiguous_name_raises(catalog, tmp_path, monkeypatch):
    lines = FIXTURE.read_text().splitlines(keepends=True)
    lola = next(line for line in lines if line.startswith('00463'))
    twin = '00464' + lola[5:166] + '(464) Lola'.ljust(28) + lola[194:]
    path = tmp_path / 'served' / 'twins.txt'
    path.write_text(lola + twin)
    monkeypatch.setitem(resonances.config.config, 'MPC_CATALOG', path.as_uri())
    assert mpcorb.lookup([464])['464']['name'] == 'Lola'
    with pytest.raises(ValueError, match='matches 2 MPC rows'):
        mpcorb.lookup(['Lola'])


def test_ambiguity_message_lists_a_few_rows(catalog, tmp_path, monkeypatch):
    """An identifier matching many rows names the first AMBIGUOUS_ROWS_SHOWN and counts the others."""
    lines = FIXTURE.read_text().splitlines(keepends=True)
    lola = next(line for line in lines if line.startswith('00463'))
    twins = [f'{n:05d}' + lola[5:166] + f'({n}) Lola'.ljust(28) + lola[194:] for n in range(500, 525)]
    path = tmp_path / 'served' / 'twins.txt'
    path.write_text(''.join(twins))
    monkeypatch.setitem(resonances.config.config, 'MPC_CATALOG', path.as_uri())
    with pytest.raises(ValueError) as error:
        mpcorb.lookup(['Lola'])
    message = str(error.value)
    assert 'matches 25 MPC rows: 00500, 00501' in message and '00509 and 15 more' in message and '00510' not in message


# ----------------------------------------------------------------------------------------------- simulation
def test_default_date_is_the_standard_epoch(catalog):
    sim = Simulation(source='mpc')
    assert sim.config.date == datetime.datetime(2026, 6, 9)
    assert sim.config.mpc_snapshot['sha256'] == mpcorb.read_snapshot()['sha256']
    assert (sim.config.mpc_epoch_policy, sim.config.mpc_propagate_max_days) == ('propagate', None)


def test_data_source_mpc_from_config(catalog, monkeypatch):
    monkeypatch.setitem(resonances.config.config, 'DATA_SOURCE', 'mpc')
    assert Simulation().config.source == 'mpc'


# get-orb CAR of (463) at MJD 61200 TDT: heliocentric J2000 ecliptic, AU and AU/day
GET_ORB_463_CAR = (0.819142639298433, -1.9173428169715, -0.488461381415565, 0.0101753486124953, 0.00703237488004754, -9.55535169066704e-05)


def test_body_elements_end_to_end(catalog):
    """add_body(463): the MPCORB line's elements in radians at MJD 61200, placed by the package around a Sun with
    G M = k^2 (AU, day), land on get-orb's Cartesian state (MPC's own conversion) within the line's rounding."""
    sim = Simulation(source='mpc')
    sim.add_body(463, '4J-2S-1', name='463')
    elem = sim.bodies[0].initial_data
    assert elem['epoch'] == 61200.0
    assert elem['inc'] == pytest.approx(math.radians(13.53827), rel=1e-15)
    assert (elem['source'], elem['packed'], elem['number'], elem['mpc_name'], elem['U']) == ('mpc', '00463', 463, 'Lola', '0')
    assert elem['catalog_sha256'] == sim.config.mpc_snapshot['sha256']

    sun = rebound.Simulation()
    sun.G = mpcorb.GAUSSIAN_K**2
    sun.add(m=1.0)
    sim.body_manager._add_body_to_simulation(sim.bodies[0], sun)
    p = sun.particles[1]
    position, velocity = (p.x, p.y, p.z), (p.vx, p.vy, p.vz)
    for mine, theirs in zip(position, GET_ORB_463_CAR[:3]):
        assert mine == pytest.approx(theirs, abs=1e-6)
    for mine, theirs in zip(velocity, GET_ORB_463_CAR[3:]):
        assert mine == pytest.approx(theirs, abs=1e-8)


@pytest.mark.parametrize(
    'identifier, epoch, words', [('2022 LX1', 'MJD 61000.00000', 'MPC_EPOCH_POLICY'), ('1942 RH', 'MJD 30611.00000', 'one-opposition')]
)
def test_other_epoch_is_refused_by_strict(catalog, identifier, epoch, words):
    sim = Simulation(source='mpc', mpc_epoch_policy='strict')
    with pytest.raises(ValueError, match=words) as error:
        sim.add_body(identifier, '4J-2S-1')
    assert epoch in str(error.value)
    assert sim.bodies == []


def test_refusal_offers_only_the_ways_out_that_apply(catalog):
    """At the default date (the standard epoch) the message does not say 'leave date unset'; at another date it does.
    Under propagate with a too long span it does not suggest the propagate policy."""
    with pytest.raises(ValueError) as at_default:
        Simulation(source='mpc', mpc_epoch_policy='strict').add_body('2022 LX1', '4J-2S-1')
    assert 'leave date unset' not in str(at_default.value).lower()
    assert "Set date to the body's epoch" in str(at_default.value) and 'MPC_EPOCH_POLICY=propagate' in str(at_default.value)

    with pytest.raises(ValueError) as at_other_date:
        Simulation(source='mpc', date='2026-01-01', mpc_epoch_policy='strict').add_body(463, '4J-2S-1')
    assert "Leave date unset (the catalogue's standard epoch, 2026-06-09, is the body's epoch)" in str(at_other_date.value)

    with pytest.raises(ValueError) as too_long:
        Simulation(source='mpc', mpc_epoch_policy='propagate', mpc_propagate_max_days=100).add_body('2022 LX1', '4J-2S-1')
    message = str(too_long.value)
    assert 'exceeds MPC_PROPAGATE_MAX_DAYS = 100' in message and 'MPC_EPOCH_POLICY=propagate' not in message


def test_find_leaves_out_refused_rows_like_missing_ones(catalog, monkeypatch):
    """Under strict, a find over a list with a row at another epoch (2022 LX1) adds the others and warns, as for a
    missing body."""
    monkeypatch.setattr(Simulation, 'create_solar_system', lambda self, force=False: None)
    warnings = []
    monkeypatch.setattr(resonances.finder.finder.logger, 'warning', warnings.append)
    sim = resonances.find([463, '2022 LX1', 'NoSuch', 10], type='lidov_kozai', source='mpc', mpc_epoch_policy='strict')
    assert [body.name for body in sim.bodies] == ['463', '10']
    assert any(w.startswith('Asteroid 2022 LX1 left out: The MPC elements of 2022 LX1 are at epoch MJD 61000') for w in warnings)
    assert 'Asteroid NoSuch not found in catalog' in warnings


def test_body_at_an_explicit_date_equal_to_its_epoch(catalog):
    """2022 LX1 is at MJD 61000 = 2025-11-21: accepted at that date, and 463 (MJD 61200) is refused there (strict)."""
    sim = Simulation(source='mpc', date='2025-11-21', mpc_epoch_policy='strict')
    sim.add_body('2022 LX1', '4J-2S-1')
    assert sim.bodies[0].initial_data['epoch'] == 61000.0
    with pytest.raises(ValueError, match='epoch'):
        sim.add_body(463, '4J-2S-1')


def test_non_elliptic_row_raises():
    row = {**_record('00463'), 'e': 1.2}
    with pytest.raises(ValueError, match='not an elliptic orbit'):
        mpcorb.elements_from_row(row)


def test_unknown_body_raises(catalog):
    with pytest.raises(ValueError, match='not in the MPC catalogue'):
        Simulation(source='mpc').add_body('2040 P-L', '4J-2S-1')


def test_quality_warnings(catalog, monkeypatch):
    warnings = []
    monkeypatch.setattr(resonances.simulation.body_manager.logger, 'warning', warnings.append)
    sim = Simulation(source='mpc')
    sim.add_body('1995 YY3', '4J-2S-1')  # U = E
    sim.add_body('2014 HA620', '4J-2S-1')  # U = 8
    sim.add_body(463, '4J-2S-1')  # U = 0
    assert len(warnings) == 2
    assert 'U = E' in warnings[0] and 'U = 8' in warnings[1]


def test_catalogue_updated_after_the_config_is_refused(catalog):
    sim = Simulation(source='mpc')
    sim.config.mpc_snapshot = {**sim.config.mpc_snapshot, 'sha256': 'f' * 64}
    with pytest.raises(ValueError, match='updated in between'):
        sim.add_body(463, '4J-2S-1')


def test_worker_config_does_not_open_the_catalogue(catalog, monkeypatch):
    """A batch worker (or restore) gets mpc_snapshot with the config, and the elements as dicts."""
    parent = Simulation(source='mpc')
    parent.add_body(463, '4J-2S-1', name='463')
    kwargs = SimulationSerializer._config_to_dict(parent.config)
    assert kwargs['mpc_snapshot']['sha256'] == parent.config.mpc_snapshot['sha256']
    for module in (mpcorb, mpc_cache):
        monkeypatch.setattr(module, 'snapshot', lambda *a, **k: pytest.fail('the worker must not read the catalogue'))
    worker = Simulation(**{**kwargs, 'date': parent.config.date, '_skip_path_verification': True})
    worker.add_body(parent.bodies[0].initial_data, '4J-2S-1', name='463')
    assert worker.config.date == datetime.datetime(2026, 6, 9)
    assert worker.bodies[0].initial_data == parent.bodies[0].initial_data


def test_resume_hash_follows_the_snapshot(catalog):
    """Two runs that differ only in the catalogue's sha256 (or parser version) do not resume each other."""
    first = Simulation(source='mpc')
    second = Simulation(source='mpc')
    assert StateManager.compute_config_hash(first.batch_manager._get_config_dict()) == StateManager.compute_config_hash(
        second.batch_manager._get_config_dict()
    )
    for key, value in (('sha256', 'f' * 64), ('parser_version', mpcorb.PARSER_VERSION + 1)):
        second.config.mpc_snapshot = {**first.config.mpc_snapshot, key: value}
        assert StateManager.compute_config_hash(first.batch_manager._get_config_dict()) != StateManager.compute_config_hash(
            second.batch_manager._get_config_dict()
        )


def test_batch_worker_never_opens_the_catalogue(catalog, monkeypatch):
    """A real batch worker (`_run_batch_worker` with the parent's kwargs and batches) runs with the catalogue out of
    reach: it builds its bodies from the parent's element dicts and its config from the parent's mpc_snapshot."""
    parent = Simulation(source='mpc')
    parent.add_body(463, '4J-2S-1', name='463')
    parent.add_body('2001 ND13', '4J-2S-1', name='2001 ND13')
    kwargs = parent.batch_manager._prepare_simulation_kwargs(parent)
    batches = parent.batch_manager.create_batches(parent.bodies)

    for module in (mpcorb, mpc_cache):
        monkeypatch.setattr(module, 'snapshot', lambda *a, **k: pytest.fail('the worker must not read the catalogue'))
        monkeypatch.setattr(module, 'load_table', lambda *a, **k: pytest.fail('the worker must not read the catalogue'))
    monkeypatch.setattr(Simulation, 'create_solar_system', lambda self, force=False: None)
    workers = []
    monkeypatch.setattr(Simulation, 'run', lambda self, progress=False: workers.append(self))
    results = [_run_batch_worker(index, batch, kwargs) for index, batch in enumerate(batches)]
    assert [r['status'] for r in results] == ['success'] * len(batches), [r['error'] for r in results]
    bodies = {body.name: body.initial_data for worker in workers for body in worker.bodies}
    assert bodies == {body.name: body.initial_data for body in parent.bodies}
    assert all(worker.config.mpc_snapshot == parent.config.mpc_snapshot for worker in workers)
    assert all(worker.config.date == parent.config.date for worker in workers)


def test_find_and_check_agree(catalog, monkeypatch):
    monkeypatch.setattr(Simulation, 'create_solar_system', lambda self, force=False: None)
    found = resonances.find([463, '2001 ND13'], type='lidov_kozai', source='mpc')
    checked = resonances.check([463, '2001 ND13'], '4J-2S-1', source='mpc')
    by_name = {body.name: body.initial_data for body in found.bodies}
    assert by_name == {body.name: body.initial_data for body in checked.bodies}
    assert by_name['2001 ND13']['packed'] == 'K01N13D'


# ----------------------------------------------------------------------------------------------- propagate
class _KeplerEngine:
    """The Sun alone (G = 1, M = 1, time unit 1/k days as in the package): propagation is then Kepler's."""

    dates = []

    def __init__(self, config):
        self.config = config

    def create_solar_system(self):
        _KeplerEngine.dates.append(self.config.date)
        self.sim = rebound.Simulation()
        self.sim.add(m=1.0)


@pytest.mark.parametrize('date, identifier, days', [('2026-06-09', '2022 LX1', 200.0), ('2025-11-21', 463, -200.0)])
def test_propagate_policy_is_kepler_without_planets(catalog, monkeypatch, date, identifier, days):
    """Forward (2022 LX1, MJD 61000 -> 61200) and backward (463, 61200 -> 61000): M grows by the file's own n (deg/day)
    times the span, the other elements stay; the planets are built at the body's epoch."""
    _KeplerEngine.dates = []
    monkeypatch.setattr(propagate_module, 'IntegrationEngine', _KeplerEngine)
    record = _record('K22L01X' if identifier == '2022 LX1' else '00463')
    sim = Simulation(source='mpc', date=date, mpc_epoch_policy='propagate')
    sim.add_body(identifier, '4J-2S-1')
    elem = sim.bodies[0].initial_data
    assert _KeplerEngine.dates == [mpcorb.datetime_from_mjd(record['epoch_mjd'])]
    assert (elem['epoch'], elem['propagated_from_epoch']) == (record['epoch_mjd'] + days, record['epoch_mjd'])
    expected_m = (record['M_deg'] + record['n_deg_day'] * days) % 360.0
    assert abs(_angle_difference(math.degrees(elem['M']), expected_m)) < 1e-5
    assert elem['a'] == pytest.approx(record['a'], rel=1e-10)
    assert elem['e'] == pytest.approx(record['e'], abs=1e-10)
    for key, deg in (('inc', 'inc_deg'), ('Omega', 'Omega_deg'), ('omega', 'omega_deg')):
        assert abs(_angle_difference(math.degrees(elem[key]), record[deg])) < 1e-7


def test_propagate_refuses_long_spans(catalog):
    sim = Simulation(source='mpc', mpc_epoch_policy='propagate', mpc_propagate_max_days=100)
    with pytest.raises(ValueError, match='MPC_PROPAGATE_MAX_DAYS'):
        sim.add_body('2022 LX1', '4J-2S-1')


def test_unknown_epoch_policy_raises(catalog):
    with pytest.raises(ValueError, match='mpc_epoch_policy'):
        Simulation(source='mpc', mpc_epoch_policy='advance-M')


# ----------------------------------------------------------------------------------------------- CLI
def test_cli_mpc_update(catalog):
    url, _ = catalog
    result = CliRunner().invoke(cli, ['mpc-update', f'--catalog={url}'])
    assert result.exit_code == 0, result.output
    assert f'{FIXTURE_ROWS} rows' in result.output and 'MJD 61200.0' in result.output and '2026-06-09' in result.output
    assert mpcorb.read_snapshot(url)['sha256'] in result.output


def test_cli_find_with_mpc_source(catalog, monkeypatch):
    runs = []
    monkeypatch.setattr(Simulation, 'create_solar_system', lambda self, force=False: None)
    monkeypatch.setattr(Simulation, 'run', lambda self, progress=False: runs.append(self))
    result = CliRunner().invoke(
        cli, ['find', '--asteroids=463', '--source=mpc', '--type=mmr', '--planets=Jupiter,Saturn', '--save=none', '--plot=none']
    )
    assert result.exit_code == 0, result.output
    assert runs[0].config.date == datetime.datetime(2026, 6, 9)
    assert runs[0].bodies[0].initial_data['packed'] == '00463'


# ----------------------------------------------------------------------------------------------- default propagation
def test_find_propagates_other_epochs_by_default(catalog, monkeypatch):
    """The default policy (propagate, no cap): 2022 LX1 (epoch K25BL = MJD 61000) is carried 200 days to the standard
    epoch with one warning naming it, the gap, U, arc and oppositions (its line: U 0, 2 oppositions, 2022-2026), and
    find adds one summary line."""
    _KeplerEngine.dates = []
    monkeypatch.setattr(propagate_module, 'IntegrationEngine', _KeplerEngine)
    monkeypatch.setattr(Simulation, 'create_solar_system', lambda self, force=False: None)
    warnings = []
    monkeypatch.setattr(resonances.finder.finder.logger, 'warning', warnings.append)
    sim = resonances.find([463, '2022 LX1', 'NoSuch'], type='lidov_kozai', source='mpc')
    assert [body.name for body in sim.bodies] == ['463', '2022 LX1']
    lx1 = sim.bodies[1].initial_data
    assert (lx1['epoch'], lx1['propagated_from_epoch']) == (61200.0, 61000.0)
    per_body = [w for w in warnings if w.startswith('2022 LX1: MPC epoch MJD 61000.00000 is +200.0 days')]
    assert len(per_body) == 1 and 'U = 0, arc 2022-2026, 2 oppositions' in per_body[0]
    assert '1 bodies propagated from other epochs to 2026-06-09 00:00:00, max gap 200.0 d' in warnings
    assert not any('463' in w and 'propagated' in w for w in warnings)


def test_default_policy_has_no_span_limit(catalog, monkeypatch):
    """1942 RH (epoch J4299 = MJD 30611, 30589 days before the standard epoch 61200) is propagated without a cap:
    with the Sun alone M grows by the file's n = 0.28463408 deg/day times the span."""
    monkeypatch.setattr(propagate_module, 'IntegrationEngine', _KeplerEngine)
    sim = Simulation(source='mpc')
    sim.add_body('1942 RH', '4J-2S-1')
    elem = sim.bodies[0].initial_data
    assert (elem['epoch'], elem['propagated_from_epoch']) == (61200.0, 30611.0)
    expected_m = (4.75163 + 0.28463408 * 30589.0) % 360.0
    assert abs(_angle_difference(math.degrees(elem['M']), expected_m)) < 1e-3


# ----------------------------------------------------------------------------------------------- flags and arcs
@pytest.mark.parametrize(
    'flags, orbit_type, set_bits',
    [
        # MPOrbitFormat.html, columns 162-165: bits 0-5 the type, 2048 NEO, 4096 1-km NEO, 8192 1-opposition seen
        # at an earlier opposition, 16384 critical list, 32768 PHA
        ('8803', 'apollo', {'neo', 'pha'}),  # 2022 LX1: 32768 + 2048 + 3
        ('2803', 'apollo', {'neo', 'one_opp_earlier'}),  # 2008 TC3: 8192 + 2048 + 3
        ('0803', 'apollo', {'neo'}),  # 2001 ND13
        ('4000', '', {'critical_list'}),  # Ceres
        ('0005', 'q_lt_1665', set()),  # (101429): q = 1.371 AU
        ('000A', 'distant', set()),  # Sedna
        ('0007', 'mpc_internal', set()),  # Lola
        ('1001', 'atira', {'km_neo'}),
        ('', '', set()),
    ],
)
def test_decode_flags(flags, orbit_type, set_bits):
    decoded = mpcorb.decode_flags(flags)
    assert decoded['orbit_type'] == orbit_type
    assert {name for name in mpcorb.FLAG_BITS if decoded[name]} == set_bits


@pytest.mark.parametrize('arc, years', [('1801-2026', 225.0), ('2001-2025', 24.0), ('1 days', 1 / 365.25), ('10 days', 10 / 365.25)])
def test_arc_years(arc, years):
    assert mpcorb.arc_years(arc) == pytest.approx(years, rel=1e-12)


def test_blank_arc_is_nan():
    assert math.isnan(mpcorb.arc_years(''))


# ----------------------------------------------------------------------------------------------- query
# The expected rows are read off the fixture's lines by hand (a: cols 93-103, e: 71-79, i: 60-68, node: 49-57,
# H: 9-13, U: 106, oppositions: 124-126, arc: 128-136, flags: 162-165)
@pytest.mark.parametrize(
    'criteria, packed',
    [
        ({'a': (0, 2)}, {'A1429', 'K01N13D', 'K08T03C'}),  # a = 1.5242076, 1.5070852, 1.3082282
        # q = a(1 - e): 2001 ND13 0.851, 2008 TC3 0.900, 2022 LX1 0.971; next is (101429) at 1.371
        ({'q': (None, 1.3)}, {'K01N13D', 'K08T03C', 'K22L01X'}),
        ({'neo': True}, {'K01N13D', 'K08T03C', 'K22L01X'}),
        ({'Q': (40, None)}, {'90377', 'J95Y03Y'}),  # Q = a(1 + e): Sedna 1011.2, 1995 YY3 48.40
        ({'orbit_type': 'apollo'}, {'K01N13D', 'K08T03C', 'K22L01X'}),
        ({'orbit_type': ['distant', 'q_lt_1665']}, {'90377', 'J95Y03Y', 'A1429'}),
        ({'pha': True}, {'K22L01X'}),
        ({'critical_list': True}, {'00001', '00002', '00004'}),
        ({'km_neo': True}, set()),
        ({'one_opp_earlier': True}, {'K08T03C', '_EH0000', '_PO00fh', 'J42R00H', 'K21A00Q'}),
        ({'n_opp': (None, 1)}, {'K08T03C', '_EH0000', '_PO00fh', 'J42R00H', 'K21A00Q'}),
        ({'U': (5, None)}, {'90377', '_EH0000', '_PO00fh'}),  # U 5, 8, 7; E and blank fail any U range
        ({'H': (19, 21)}, {'_EH0000', '_PO00fh', 'K22L01X'}),  # H 19.05, 19.58, 20.02
        ({'i': (30, None)}, {'00002', 'A1429', 'K21A00Q'}),  # i 34.93279, 31.29603, 43.51028
        ({'arc_years': (100, None)}, {'00001', '00002', '00004', '00010', '00463'}),  # Lola 1926-2026: exactly 100
        ({'node': (280, 40)}, {'00010', '_PO00fh', '00463'}),  # wraps: node 283.11987, 305.05803, 36.48765
        ({'a': (2.3, 2.7), 'e': (None, 0.1), 'numbered': False}, {'_EH0000', '_PO00fh'}),
    ],
)
def test_query(catalog, criteria, packed):
    assert set(mpcorb.query(**criteria)['packed']) == packed


def test_query_counts_and_epochs(catalog):
    assert len(mpcorb.query()) == FIXTURE_ROWS
    assert len(mpcorb.query(numbered=True)) == 8 and len(mpcorb.query(numbered=False)) == 9
    # 2008 TC3, 2022 LX1, 1942 RH and 2021 AQ are at other epochs
    assert set(mpcorb.query(epoch='any')['packed']) - set(mpcorb.query(epoch='standard')['packed']) == {
        'K08T03C',
        'K22L01X',
        'J42R00H',
        'K21A00Q',
    }


def test_query_columns_of_a_row(catalog):
    """2022 LX1's line: K22L01X 20.02 ... 0.6251719 ... 2.5896989  0 ... 254 2 2022-2026 ... 8803."""
    row = mpcorb.query(pha=True).iloc[0]
    assert (row['designation'], row['number'], row['U'], row['flags'], row['orbit_type']) == ('2022 LX1', 0, '0', '8803', 'apollo')
    assert (row['n_obs'], row['n_opp'], row['arc_years'], row['H']) == (254, 2, 4.0, 20.02)
    assert row['q'] == pytest.approx(2.5896989 * (1 - 0.6251719), rel=1e-12)
    assert row['Q'] == pytest.approx(2.5896989 * (1 + 0.6251719), rel=1e-12)
    assert (row['neo'], row['pha'], row['km_neo'], row['epoch_mjd']) == (True, True, False, 61000.0)


@pytest.mark.parametrize(
    'criteria, error',
    [
        ({'a': (2, 1)}, ValueError),
        ({'a': 1.0}, TypeError),
        ({'a': (0, 1, 2)}, TypeError),
        ({'e': ('0', 1)}, TypeError),
        ({'orbit_type': 'trojan'}, ValueError),
        ({'epoch': 'old'}, ValueError),
    ],
)
def test_query_rejects_bad_criteria(catalog, criteria, error):
    with pytest.raises(error):
        mpcorb.query(**criteria)


def test_search_by_axis(catalog):
    """As astdys.search_by_axis: a within 2.54..2.66 AU are 2022 LX1 (2.5896989), 2025 OL723 (2.6085249) and
    2014 HA620 (2.6570744)."""
    assert set(mpcorb.search_by_axis(2.6, sigma=0.06)['packed']) == {'K22L01X', '_PO00fh', '_EH0000'}
    with pytest.raises(ValueError):
        mpcorb.search_by_axis(-1.0)


def test_search_returns_elements(catalog):
    lola = mpcorb.search(463)
    assert lola['inc'] == pytest.approx(math.radians(13.53827), rel=1e-15) and lola['epoch'] == 61200.0
    assert mpcorb.search('no such body') is None
    found = mpcorb.search([1, '2001 ND13', 'no such body'])
    assert set(found) == {'1', '2001 ND13'} and found['2001 ND13']['a'] == 1.5070852
    assert mpcorb.get_catalog_datetime() == datetime.datetime(2026, 6, 9)
    assert len(mpcorb.get_catalog()) == FIXTURE_ROWS


def test_identifiers_of_a_selection(catalog):
    assert mpcorb.identifiers(mpcorb.query(neo=True)) == ['2001 ND13', '2008 TC3', '2022 LX1']
    assert mpcorb.identifiers(mpcorb.query(critical_list=True)) == [1, 2, 4]


def test_find_takes_a_selection(catalog, monkeypatch):
    """a within 2.35..2.40 AU among the numbered rows: (4) Vesta at 2.3613660 and (463) Lola at 2.3986620."""
    monkeypatch.setattr(Simulation, 'create_solar_system', lambda self, force=False: None)
    selection = mpcorb.query(a=(2.35, 2.40), numbered=True)
    sim = resonances.find(selection, type='lidov_kozai', source='mpc')
    assert [body.name for body in sim.bodies] == ['4', '463']


def test_cli_mpc_search(catalog, tmp_path):
    result = CliRunner().invoke(cli, ['mpc-search', '--a', '0:2', '--limit', '5'])
    assert result.exit_code == 0, result.output
    assert '3 rows match' in result.output and '2001 ND13' in result.output and '2008 TC3' in result.output
    result = CliRunner().invoke(cli, ['mpc-search', '--orbit-type=apollo', '--pha', '--ids'])
    assert result.output.strip() == '2022 LX1'
    out = tmp_path / 'sel.csv'
    result = CliRunner().invoke(cli, ['mpc-search', '--Q', '40:', '--H', ':10', '--output', str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text().count('\n') == 3  # header + Sedna (H 1.50) + 1995 YY3 (H 8.5)
    assert CliRunner().invoke(cli, ['mpc-search', '--a', '2']).exit_code != 0


# ----------------------------------------------------------------------------------------------- self-containment
MPCORB_PACKAGE = Path(resonances.mpcorb.__file__).parent


def test_mpcorb_imports_nothing_of_the_package():
    """Only relative imports inside the folder, and none of `resonances`."""
    for path in MPCORB_PACKAGE.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not any(alias.name.split('.')[0] == 'resonances' for alias in node.names), path.name
            elif isinstance(node, ast.ImportFrom):
                assert node.level > 0 or node.module.split('.')[0] != 'resonances', path.name


def test_mpcorb_works_as_its_own_package(tmp_path):
    """A copy of the folder, imported as `mpcorb` with the package out of reach, reads the fixture."""
    shutil.copytree(MPCORB_PACKAGE, tmp_path / 'site' / 'mpcorb', ignore=shutil.ignore_patterns('__pycache__'))
    script = (
        "import sys, mpcorb\n"
        "assert not any(name == 'resonances' or name.startswith('resonances.') for name in sys.modules)\n"
        f"mpcorb.configure(catalog={FIXTURE.as_uri()!r}, cache_dir={str(tmp_path / 'cache')!r}, max_age_days=0)\n"
        "print(mpcorb.search('Lola')['a'], ','.join(mpcorb.query(pha=True)['designation']))\n"
    )
    done = subprocess.run(
        [sys.executable, '-c', script],
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(tmp_path / 'site')},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ['2.398662', '2022', 'LX1']


def test_selection_frames_as_asteroid_lists():
    """find/check take an astdys frame by its 'name' column; another frame is refused."""
    import pandas as pd

    from resonances.data.util import convert_input_to_list

    assert convert_input_to_list(pd.DataFrame({'name': ['463', '2001ND13'], 'a': [2.4, 1.5]})) == ['463', '2001ND13']
    with pytest.raises(ValueError, match='selection'):
        convert_input_to_list(pd.DataFrame({'a': [2.4]}))
