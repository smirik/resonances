"""MPC_LOOKUP=file_then_api (docs/config.md, "Minor Planet Center"): an identifier the MPCORB snapshot lacks gets its
nominal orbit from MPC get-orb, without the network here: the answers saved in tests/fixtures/clones/mpc (463 and
2013 RZ53, fetched 2026-10-27) and copies of them under other names.

Independent expectations: MPC's own KEP block (its conversion) against the package's conversion of the CAR block;
the MPCORB line of 463 in tests/fixtures/mpc/MPCORB_sample.DAT (rounded to 1e-7 AU and 1e-5 deg)."""

import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest

import resonances
from resonances import mpcorb
from resonances.clones import solution as solution_module
from resonances.simulation import mpc_getorb
from resonances.simulation.simulation import Simulation

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures'
GETORB = FIXTURES / 'clones' / 'mpc'
# (463) Lola in MPCORB_sample.DAT at MJD 61200: a [AU], e, i, node, peri, M [deg]
MPCORB_463 = (2.3986620, 0.2194729, 13.53827, 36.48765, 329.45975, 310.21988)


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """The MPCORB fixture served by a file:// URL, the caches in tmp_path; get-orb answers from a copy of the
    fixture folder (CLONES_CACHE_DIR), no download allowed."""
    source = tmp_path / 'served' / 'MPCORB.DAT.gz'
    source.parent.mkdir()
    with open(FIXTURES / 'mpc' / 'MPCORB_sample.DAT', 'rb') as raw, gzip.open(source, 'wb') as out:
        shutil.copyfileobj(raw, out)
    clones_cache = tmp_path / 'clones'
    shutil.copytree(FIXTURES / 'clones', clones_cache)
    for key, value in {
        'MPC_CACHE_DIR': str(tmp_path / 'mpc'),
        'MPC_CATALOG': source.as_uri(),
        'MPC_SHA256': '',
        'MPC_MAX_AGE_DAYS': '0',
        'MPC_EPOCH_POLICY': 'propagate',
        'MPC_PROPAGATE_MAX_DAYS': '',
        'CLONES_CACHE_DIR': str(clones_cache),
    }.items():
        monkeypatch.setitem(resonances.config.config, key, value)

    def no_network(url, body=None):
        raise FileNotFoundError(f'{url}: not found (404)')

    monkeypatch.setattr(solution_module, '_download', no_network)
    mpcorb.clear_cache()
    yield clones_cache
    mpcorb.clear_cache()


def getorb(name='463.json'):
    return json.loads((GETORB / name).read_text())


def cache_copy(clones_cache: Path, source: str, identifier: str, data=None):
    """A get-orb answer cached under another identifier, with its meta (sha256) as `fetch` writes it."""
    content = json.dumps(data if data is not None else getorb(source)).encode()
    path = clones_cache / 'mpc' / f"{identifier.replace(' ', '')}.json"
    path.write_bytes(content)
    meta = {'url': solution_module.MPC_URL, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    path.with_name(path.name + '.meta.json').write_text(json.dumps(meta))
    return meta['sha256']


def test_nominal_from_kep_agrees_with_the_car_conversion_and_mpcorb():
    data = getorb()
    kep = mpc_getorb.nominal_elements(data, '463')
    orbit = data[0]['mpc_orb'][0]
    values = orbit['KEP']['coefficient_values']
    assert kep['a'] == values[0] and kep['e'] == values[1] and kep['inc'] == math.radians(values[2])
    assert (kep['epoch'], kep['number'], kep['designation'], kep['mpc_name'], kep['U']) == (61200.0, 463, 'A900 UK', 'Lola', '0')
    del orbit['KEP']  # the CAR block alone, converted by the package (mu = k^2)
    car = mpc_getorb.nominal_elements(data, '463')
    # measured: a differs by 2.2e-10 relative (the CAR block's 15 digits and MPC's own constants), e by < 1e-10
    assert car['a'] == pytest.approx(kep['a'], rel=1e-9) and car['e'] == pytest.approx(kep['e'], abs=1e-9)
    for key in ('inc', 'Omega', 'omega', 'M'):
        assert abs((car[key] - kep[key] + math.pi) % (2 * math.pi) - math.pi) < 1e-8, key
    a, e, inc, node, peri, mean = MPCORB_463
    assert kep['a'] == pytest.approx(a, abs=1e-7) and kep['e'] == pytest.approx(e, abs=1e-7)
    for key, deg in zip(('inc', 'Omega', 'omega', 'M'), (inc, node, peri, mean)):
        assert math.degrees(kep[key]) == pytest.approx(deg, abs=1e-5), key


def test_no_covariance_is_needed():
    data = getorb()
    orbit = data[0]['mpc_orb'][0]
    for block in ('CAR', 'KEP', 'COM'):
        orbit[block].pop('covariance', None)
    orbit['orbit_fit_statistics']['U_param'] = 9  # the clones' U policy is not the nominal's
    assert mpc_getorb.nominal_elements(data, '463')['U'] == '9'


@pytest.mark.parametrize(
    'spoil, words',
    [
        (lambda o, d: d[0].update(mpc_orb=[]), 'no orbit'),
        (lambda o, d: d[0].update(mpc_orb=[o, o]), 'ambiguous'),
        (lambda o, d: o['system_data'].update(refsys='Equatorial'), 'frame'),
        (lambda o, d: o['epoch_data'].update(timesystem='UTC'), 'MJD TT'),
        (lambda o, d: o['KEP']['coefficient_values'].__setitem__(1, 1.2), 'not an elliptic orbit'),
    ],
)
def test_unusable_answers_are_refused(spoil, words):
    data = getorb()
    spoil(data[0]['mpc_orb'][0], data)
    with pytest.raises(ValueError, match=words):
        mpc_getorb.nominal_elements(data, '463')


def test_an_object_outside_the_catalogue_comes_from_getorb(catalog):
    """2013 RZ53 is not in the MPCORB fixture: refused with MPC_LOOKUP=file, read from get-orb (U = 6: a warning)
    with file_then_api, at its epoch MJD 61200 = the default date."""
    with pytest.raises(ValueError, match='not in the MPC catalogue'):
        Simulation(source='mpc').add_body('2013 RZ53', '4J-2S-1')
    warnings = []
    sim = Simulation(source='mpc', mpc_lookup='file_then_api')
    logger = resonances.simulation.body_manager.logger
    original, logger.warning = logger.warning, warnings.append
    try:
        sim.add_body('2013 RZ53', '4J-2S-1', name='rz53')
    finally:
        logger.warning = original
    elem = sim.bodies[0].initial_data
    values = getorb('2013RZ53.json')[0]['mpc_orb'][0]['KEP']['coefficient_values']
    assert (elem['a'], elem['e'], elem['inc'], elem['epoch']) == (values[0], values[1], math.radians(values[2]), 61200.0)
    assert elem['lookup'] == 'get-orb' and elem['designation'] == '2013 RZ53' and elem['number'] is None
    meta = json.loads((catalog / 'mpc' / '2013RZ53.json.meta.json').read_text())
    assert elem['getorb_sha256'] == meta['sha256']
    assert any('U = 6' in w for w in warnings)


def test_an_alias_comes_from_getorb_and_matches_the_catalogue_row(catalog):
    """1926 TC, a secondary designation of (463), is not a key of MPCORB: get-orb (here the 463 answer cached under
    that name) gives the elements of the catalogue's row within its rounding."""
    sha = cache_copy(catalog, '463.json', '1926 TC')
    sim = Simulation(source='mpc', mpc_lookup='file_then_api')
    sim.add_body(463, '4J-2S-1', name='from_mpcorb')
    sim.add_body('1926 TC', '4J-2S-1', name='alias')
    row, alias = sim.bodies[0].initial_data, sim.bodies[1].initial_data
    assert alias['getorb_identifier'] == '1926 TC' and alias['getorb_sha256'] == sha and alias['number'] == 463
    assert alias['a'] == pytest.approx(row['a'], abs=1e-7) and alias['e'] == pytest.approx(row['e'], abs=1e-7)
    for key in ('inc', 'Omega', 'omega', 'M'):
        assert math.degrees(alias[key]) == pytest.approx(math.degrees(row[key]), abs=1e-5)


def test_find_leaves_out_what_getorb_cannot_give(catalog, monkeypatch):
    """No answer (here: no network, a 404) is a refusal: find leaves the body out with its message."""
    warnings = []
    monkeypatch.setattr(resonances.finder.finder.logger, 'warning', warnings.append)
    sim = resonances.find([463, 'NoSuch', '2013 RZ53'], type='lidov_kozai', source='mpc', mpc_lookup='file_then_api')
    assert [body.name for body in sim.bodies] == ['463', '2013 RZ53']
    assert any(w.startswith('Asteroid NoSuch left out: NoSuch is not in the MPC catalogue') and 'get-orb' in w for w in warnings)


def test_the_epoch_policy_applies_to_getorb_elements(catalog):
    """At 2026-06-10, one day after the get-orb epoch: strict refuses, propagate with a 0.5-day limit refuses."""
    for options, words in (
        (dict(mpc_epoch_policy='strict'), 'MPC_EPOCH_POLICY=strict'),
        (dict(mpc_propagate_max_days=0.5), 'MPC_PROPAGATE_MAX_DAYS'),
    ):
        sim = Simulation(source='mpc', mpc_lookup='file_then_api', date='2026-06-10', **options)
        with pytest.raises(ValueError, match=words):
            sim.add_body('2013 RZ53', '4J-2S-1')


def test_unknown_lookup_mode_raises(catalog):
    with pytest.raises(ValueError, match='mpc_lookup'):
        Simulation(source='mpc', mpc_lookup='api')
