"""Orbit solutions read from real files saved once (tests/fixtures/clones, downloaded 2026-10-27 by orbit_solution).

The units are checked against numbers each provider prints beside its covariance, computed by the provider:
- OEF (AstDyS/NEODyS): the comment lines RMS = sqrt(diag COV) and EIG = sqrt(eig COV) with lambda in RADIANS, while
  COV and EQU carry lambda in degrees;
- MPC get-orb: `eigenvalues` = sqrt(eig) of the CAR covariance;
- JPL SBDB: the covariance epoch (JD TDB) differs from the epoch of the listed elements; the nominal is the covariance
  block's own.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import resonances
from resonances.clones import OrbitSolution, orbit_solution, parse_mpc, parse_oef, parse_sbdb

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures' / 'clones'


def comment_values(path: Path, tag: str) -> np.ndarray:
    line = next(line for line in path.read_text().splitlines() if line.strip().startswith(f'! {tag}'))
    return np.array([float(x) for x in line.split()[2:]])


@pytest.fixture
def fixture_cache():
    """orbit_solution reads the saved files: CLONES_CACHE_DIR -> tests/fixtures/clones (no download)."""
    previous = resonances.config.get('CLONES_CACHE_DIR')
    resonances.config.set('CLONES_CACHE_DIR', str(FIXTURES))
    yield
    resonances.config.set('CLONES_CACHE_DIR', previous)


@pytest.mark.parametrize('relative', ['astdys/463.eq1', 'astdys/463.eq0', 'neodys/2013RZ53.eq0', 'neodys/2013RZ53.eq1'])
def test_oef_lambda_covariance_is_in_degrees(relative):
    path = FIXTURES / relative
    solution = parse_oef(path.read_text(), source='astdys')
    rms, eig = comment_values(path, 'RMS'), comment_values(path, 'EIG')
    # RMS is sqrt(diag COV) in COV's own units (lambda in degrees)
    np.testing.assert_allclose(solution.sigma, rms, rtol=1e-4)
    # EIG is sqrt(eig COV) only with lambda converted to radians; in degrees the largest one is far off. Compared where
    # float64 resolves them (eigenvalue >= 1e-12 of the largest): NEODyS's .eq1 of 2013 RZ53, the .eq0 covariance
    # propagated linearly over 13 yr, is printed with a negative smallest EIG (numerically indefinite)
    to_radians = np.diag([1, 1, 1, 1, 1, np.pi / 180])
    radians = np.sqrt(np.abs(np.linalg.eigvalsh(to_radians @ solution.covariance @ to_radians)))
    resolved = eig**2 >= 1e-12 * eig.max() ** 2
    np.testing.assert_allclose(radians[resolved], eig[resolved], rtol=2e-4)
    assert not np.allclose(np.sqrt(np.abs(np.linalg.eigvalsh(solution.covariance))), np.abs(eig), rtol=1e-2)


def test_oef_records_of_463():
    solution = parse_oef((FIXTURES / 'astdys/463.eq1').read_text(), source='astdys')
    assert solution.name == '463'
    assert solution.coordinates == 'equinoctial'
    assert solution.epoch_mjd == 61200.0 and solution.time_scale == 'TDT'
    np.testing.assert_allclose(
        solution.nominal,
        [2.3986620775145058, 0.022740843562434, 0.218291436497379, 0.070582641340646, 0.095430514455147, 316.1672474068565],
    )
    # first COV values: C_aa, C_ah, C_ak (upper triangle, row by row)
    assert solution.covariance[0, 0] == pytest.approx(2.423447768835744e-17)
    assert solution.covariance[0, 1] == solution.covariance[1, 0] == pytest.approx(6.700246467665252e-18)
    assert solution.covariance[0, 2] == pytest.approx(5.709905489515595e-17)
    assert solution.nongrav == {}


def test_oef_seven_by_seven_nongrav_fit_gives_the_orbital_block():
    """99942 Apophis (NEODyS .eq0): LSP 1 2 7, NGR 0, -2.90037e-4: a 7x7 covariance with A2; the 6x6 orbital block is
    used and the parameters flagged."""
    path = FIXTURES / 'neodys/99942.eq0'
    solution = parse_oef(path.read_text(), source='neodys')
    assert solution.covariance.shape == (6, 6)
    assert solution.nongrav['A2'] == pytest.approx(-2.90037208427322e-04)
    assert solution.nongrav['model'] == 1
    # the 6 orbital RMS of the 7 printed
    np.testing.assert_allclose(solution.sigma, comment_values(path, 'RMS')[:6], rtol=1e-4)


def test_oef_without_covariance_is_refused():
    text = "END_OF_HEADER\n463\n EQU 2.39 0.02 0.21 0.07 0.09 316.1\n MJD 61200.0 TDT\n"
    with pytest.raises(ValueError, match='no covariance'):
        parse_oef(text)


def test_oef_in_another_frame_is_refused():
    """The package's frame is the J2000 ecliptic: an equatorial OEF file (refsys EQUM J2000) would give the same numbers
    as elements in the wrong plane, so it is refused, never read unrotated. The header's case and spacing do not matter."""
    text = (FIXTURES / 'astdys/463.eq1').read_text()
    with pytest.raises(ValueError, match="'EQUM J2000'"):
        parse_oef(text.replace('ECLM J2000', 'EQUM J2000'))
    assert parse_oef(text.replace('ECLM J2000', 'eclm   J2000')).frame == 'ECLM J2000, heliocentric'


def test_oef_with_two_records_is_refused():
    """A bulk OEF (two objects or two epochs under one header) is refused: read as one, it took the last record's
    nominal and epoch with the first record's covariance (review 2026-10-27). Here 463's .eq0 record followed by its
    .eq1 record."""
    first = (FIXTURES / 'astdys/463.eq0').read_text()
    second = (FIXTURES / 'astdys/463.eq1').read_text().split('END_OF_HEADER', 1)[1]
    with pytest.raises(ValueError, match='more than one record'):
        parse_oef(first + second)
    # a COV block of 21 + 21 values without a second EQU is refused too
    cov_lines = [line for line in first.splitlines() if line.strip().startswith('COV')]
    with pytest.raises(ValueError, match='42 values'):
        parse_oef(first + '\n'.join(cov_lines) + '\n')


def test_mpc_and_sbdb_in_another_frame_are_refused():
    data = json.loads((FIXTURES / 'mpc/463.json').read_text())
    system = data[0]['mpc_orb'][0]['system_data']
    assert (system['refsys'], system['refframe'], system['EclipticObliquityArcseconds']) == ('Ecliptic', 'ICRF', '84381.448')
    assert parse_mpc(data).frame == 'Ecliptic J2000 (ICRF), heliocentric'
    for change, match in (({'refsys': 'Equatorial'}, 'Equatorial/ICRF'), ({'EclipticObliquityArcseconds': '84404.8'}, '84404.8')):
        other = json.loads(json.dumps(data))
        other[0]['mpc_orb'][0]['system_data'].update(change)
        with pytest.raises(ValueError, match=match):
            parse_mpc(other)
    sbdb = json.loads((FIXTURES / 'sbdb/463.json').read_text())
    assert sbdb['orbit']['equinox'] == 'J2000'
    sbdb['orbit']['equinox'] = 'B1950'
    with pytest.raises(ValueError, match='B1950'):
        parse_sbdb(sbdb)


def test_sbdb_covariance_at_its_own_epoch():
    data = json.loads((FIXTURES / 'sbdb/463.json').read_text())
    solution = parse_sbdb(data)
    assert solution.coordinates == 'cometary' and solution.time_scale == 'TDB'
    assert solution.epoch_mjd == pytest.approx(float(data['orbit']['covariance']['epoch']) - 2400000.5)
    assert float(data['orbit']['epoch']) != float(data['orbit']['covariance']['epoch'])
    by_label = {e['label']: float(e['value']) for e in data['orbit']['covariance']['elements']}
    np.testing.assert_allclose(solution.nominal, [by_label[k] for k in ('e', 'q', 'tp', 'node', 'peri', 'i')])
    np.testing.assert_allclose(solution.sigma, [e['sigma'] for e in data['orbit']['covariance']['elements']], rtol=1e-6)


def test_sbdb_nongrav_parameters_are_flagged():
    solution = parse_sbdb((FIXTURES / 'sbdb/99942.json').read_text())
    assert set(solution.nongrav) == {'A1', 'A2'}
    assert solution.covariance.shape == (6, 6)


def test_mpc_cartesian_covariance_matches_its_eigenvalues():
    data = json.loads((FIXTURES / 'mpc/463.json').read_text())
    solution = parse_mpc(data)
    car = data[0]['mpc_orb'][0]['CAR']
    assert solution.coordinates == 'cartesian' and solution.epoch_mjd == 61200.0
    np.testing.assert_allclose(np.sqrt(np.linalg.eigvalsh(solution.covariance)), car['eigenvalues'], rtol=1e-4)
    np.testing.assert_allclose(solution.sigma, car['coefficient_uncertainties'][:6], rtol=1e-4)


def test_mpc_weak_orbit_is_refused():
    """2013 RZ53: U = 6, a 3-day arc in 2013, MPC's covariance at MJD 61200 (a linear propagation over 13 yr)."""
    with pytest.raises(ValueError, match="neodys"):
        parse_mpc((FIXTURES / 'mpc/2013RZ53.json').read_text())


def test_orbit_solution_reads_the_cache_and_records_provenance(fixture_cache):
    solution = orbit_solution(463, source='astdys')
    assert solution.provenance['product'] == 'eq1'
    assert solution.provenance['from_cache'] is True
    assert solution.provenance['url'].endswith('/numbered/0/463.eq1')
    assert len(solution.provenance['sha256']) == 64
    fit = orbit_solution('2013 RZ53', source='neodys')  # NEODyS default: the fit's own epoch (.eq0)
    assert fit.provenance['product'] == 'eq0' and fit.epoch_mjd == pytest.approx(56549.369032888)
    assert orbit_solution('2013 RZ53', source='neodys', epoch='standard').epoch_mjd == 61200.0


def test_orbit_solution_round_trips_through_a_dict(fixture_cache):
    solution = orbit_solution(463, source='sbdb')
    again = OrbitSolution.from_dict(json.loads(json.dumps(solution.to_dict())))
    np.testing.assert_array_equal(again.covariance, solution.covariance)
    np.testing.assert_array_equal(again.nominal, solution.nominal)
    assert again.epoch_mjd == solution.epoch_mjd and again.provenance == solution.provenance


def test_unknown_source_and_unnumbered_astdys():
    with pytest.raises(ValueError):
        orbit_solution(463, source='horizons')
    with pytest.raises(ValueError, match='numbered'):
        orbit_solution('2013 RZ53', source='astdys')


@pytest.mark.slow
@pytest.mark.parametrize('identifier, source', [(463, 'astdys'), ('2013 RZ53', 'neodys'), (463, 'sbdb'), (463, 'mpc')])
def test_real_download_matches_the_saved_file(tmp_path, identifier, source):
    """A fresh download parses and describes the same object as the saved one (an orbit may have been refitted since:
    a within 1e-6 AU, the epoch of the AstDyS/MPC standard orbit unchanged)."""
    previous = resonances.config.get('CLONES_CACHE_DIR')
    resonances.config.set('CLONES_CACHE_DIR', str(tmp_path))
    try:
        fresh = orbit_solution(identifier, source=source)
    finally:
        resonances.config.set('CLONES_CACHE_DIR', previous)
    assert fresh.provenance['from_cache'] is False
    resonances.config.set('CLONES_CACHE_DIR', str(FIXTURES))
    try:
        saved = orbit_solution(identifier, source=source)
    finally:
        resonances.config.set('CLONES_CACHE_DIR', previous)
    assert fresh.coordinates == saved.coordinates
    assert fresh.nominal_elements()['a'] == pytest.approx(saved.nominal_elements()['a'], abs=1e-3 if source == 'neodys' else 1e-6)
