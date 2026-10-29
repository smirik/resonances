"""Runs with clones (virtual asteroids), end to end over a few centuries (seconds): the orbit solution of 463 saved in
tests/fixtures/clones (AstDyS .eq1 at MJD 61200 = 2026-06-09), a few clones in batches of two.

What must hold: every clone has its summary row; clones.csv has the shares of its clones; the manifest keeps every draw;
the ensemble figure is drawn; the nominal's verdict is the one it gets without clones; a covariance at another epoch is
carried to the run's date (here one day) by integration."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import resonances
from resonances.clones import orbit_solution
from resonances.clones.elements import KEPLERIAN_KEYS
from resonances.data.const import K
from resonances.data.util import mjd_from_datetime
from resonances.simulation import ensemble_run
from tests.tools import TEST_CACHE

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures' / 'clones'
KEY = '4J-2S-1+0+0-1'
VERDICT_COLUMNS = ('status', 'angle_status', 'confidence', 'reason', 'libration_origin', 'flt_cutoff_yr')


@pytest.fixture(autouse=True)
def fixture_cache():
    previous = resonances.config.get('CLONES_CACHE_DIR')
    resonances.config.set('CLONES_CACHE_DIR', str(FIXTURES))
    yield
    resonances.config.set('CLONES_CACHE_DIR', previous)


def simulation(tmp_name: str, date: str = '2026-06-09', **kwargs) -> resonances.Simulation:
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = f'{TEST_CACHE}/{tmp_name}_{stamp}'
    options = dict(
        date=date,
        save_path=path,
        plot_path=path,
        integration_years=300,
        Nout=301,
        n_cores=1,
        clones_batch=2,
        save='all',
        plot='all',
        plots=['clones'],
        save_summary=True,
        coherence_enabled=False,
    )
    options.update(kwargs)
    sim = resonances.Simulation(**options)
    sim.create_solar_system()
    return sim


@pytest.fixture(scope='module')
def ensemble():
    resonances.config.set('CLONES_CACHE_DIR', str(FIXTURES))
    return orbit_solution(463, source='astdys').sample(n=3, seed=0, name='463')


@pytest.mark.slow
def test_run_with_an_ensemble(ensemble):
    sim = simulation('clones_run')
    run = sim.add_ensemble(ensemble, resonances=['4J-2S-1'])
    assert run.names == ['463~c001', '463~c002', '463~c003']
    assert [b.name for b in sim.bodies] == ['463']  # the clones are not bodies of the ordinary run
    sim.run()

    summary = pd.read_csv(Path(sim.config.save_path) / 'summary.csv', dtype=str, keep_default_na=False)
    assert sorted(summary['name']) == ['463', '463~c001', '463~c002', '463~c003']

    clones = pd.read_csv(Path(sim.config.save_path) / 'clones.csv', keep_default_na=False, dtype={'name': str})
    assert len(clones) == 1
    row = clones.iloc[0]
    assert (row['name'], row['resonance'], row['source'], row['product']) == ('463', KEY, 'astdys', 'eq1')
    assert row['n_clones'] == 3 and row['n_classified'] == 3 and row['n_draw_invalid'] == 0
    assert not row['propagated'] and row['nominal_offset_sigma'] == 0
    shares = [row[f'f_{s}'] for s in ('resonant', 'transient', 'near_separatrix', 'non_resonant', 'uncertain', 'invalid')]
    assert sum(shares) == pytest.approx(1.0)
    clone_statuses = summary[summary['name'].str.contains('~c')]['status']
    assert row[f"n_{row['modal_status']}"] == clone_statuses.value_counts().max()
    assert row['nominal_status'] == summary.loc[summary['name'] == '463', 'status'].item()
    assert sim.clone_summary.iloc[0]['n_clones'] == 3

    manifest = pd.read_csv(Path(sim.config.data_path) / 'ensemble-463.csv')
    assert manifest['name'].tolist() == run.names
    for key in KEPLERIAN_KEYS:  # at the covariance's epoch = the run's date: the start is the draw
        np.testing.assert_allclose(manifest[f'{key}_start'], manifest[key], rtol=0, atol=0)
    assert ensemble_run_loaded(sim).solution.provenance['product'] == 'eq1'

    figures = list(Path(sim.config.plot_path).rglob('463-4J-2S-1+0+0-1-clones.png'))
    assert len(figures) == 1
    assert figures[0].parent == Path(sim.data_manager.figure_folder(sim.bodies[0], sim.bodies[0].mmrs[0]))


def ensemble_run_loaded(sim):
    return resonances.Ensemble.load(Path(sim.config.data_path) / 'ensemble-463.json')


def test_the_nominal_verdict_does_not_change(ensemble):
    """The nominal runs in its own simulation, as without clones: its summary row is the same."""
    with_clones = simulation('clones_nominal_a', plot=None)
    with_clones.add_ensemble(ensemble, resonances=['4J-2S-1'])
    with_clones.run()
    alone = simulation('clones_nominal_b', plot=None)
    alone.add_body(dict(ensemble.nominal), '4J-2S-1', name='463')
    alone.run()
    rows = []
    for sim in (with_clones, alone):
        summary = pd.read_csv(Path(sim.config.save_path) / 'summary.csv', dtype=str, keep_default_na=False)
        rows.append(summary[summary['name'] == '463'].iloc[0])
    for column in VERDICT_COLUMNS:
        assert rows[0][column] == rows[1][column], column
    a = [np.asarray(sim.bodies[0].axis) for sim in (with_clones, alone)]
    np.testing.assert_array_equal(a[0], a[1])


@pytest.mark.slow
def test_covariance_at_another_epoch_is_carried_to_the_date(ensemble):
    sim = simulation('clones_carried', date='2026-06-10', plot=None)  # MJD 61201: one day after the covariance
    run = sim.add_ensemble(ensemble, resonances=['4J-2S-1'])
    nominal = sim.bodies[0].initial_data
    assert nominal['epoch'] == pytest.approx(61201.0) and nominal['propagated_from_epoch'] == pytest.approx(61200.0)
    # one day on a 3.7-yr orbit: M advances by ~ n = k a^-3/2 rad/day; a changes only by the planets' short-period terms
    for elem, draw in zip(run.elements, ensemble.clone_elements()):
        assert elem['epoch'] == pytest.approx(61201.0)
        assert elem['M'] - draw['M'] == pytest.approx(0.0172020989484 * draw['a'] ** -1.5, rel=1e-2)
        assert abs(elem['a'] - draw['a']) < 1e-5
    sim.run()
    row = pd.read_csv(Path(sim.config.save_path) / 'clones.csv').iloc[0]
    assert row['propagated'] and row['epoch_mjd'] == 61200.0 and row['run_date_mjd'] == pytest.approx(61201.0)


def test_cloud_candidates_are_the_union_of_the_members():
    near_lola = {'a': 2.3985, 'e': 0.22, 'inc': 0.24, 'Omega': 0.6, 'omega': 5.7, 'M': 0.1}
    near_hecuba = {'a': 3.2785, 'e': 0.1, 'inc': 0.05, 'Omega': 0.6, 'omega': 5.7, 'M': 0.1}
    from resonances.mmr.mmr_finder import candidate_mmrs

    expected = []
    for elem in (near_lola, near_hecuba):
        expected += [m.to_s() for m in candidate_mmrs(elem) if m.to_s() not in expected]
    found = [m.to_s() for m in ensemble_run.cloud_candidates([near_lola, None, near_hecuba, near_lola])]
    assert found == expected
    assert '4J-2S-1+0+0-1' in found and '4J-5S-1+0+0+2' in found  # one from each member


def test_offset_of_a_nominal_from_its_cloud():
    """Clones at a = 2.0 +- 1e-6 (sd 1e-6 with ddof=1 for -1, 0, +1), lambda spread 1e-3 rad: a reference 3e-6 AU off
    lies 3 widths away; lambda dominates when it is 5e-3 rad off."""
    clones = [{'a': 2.0 + d * 1e-6, 'e': 0.1, 'inc': 0.1, 'Omega': 0.0, 'omega': 0.0, 'M': 1.0 + d * 1e-3} for d in (-1, 0, 1)]
    reference = {'a': 2.0 + 3e-6, 'e': 0.1, 'inc': 0.1, 'Omega': 0.0, 'omega': 0.0, 'M': 1.0}
    assert ensemble_run.offset_sigma(reference, clones) == pytest.approx(3.0)
    assert ensemble_run.offset_sigma({**reference, 'a': 2.0, 'M': 1.005}, clones) == pytest.approx(5.0)
    assert np.isnan(ensemble_run.offset_sigma(reference, clones[:2]))


def test_clone_settings_and_refusals(ensemble):
    with pytest.raises(ValueError, match='clones_source'):
        resonances.Simulation(clones_source='horizons', date='2026-06-09')
    sim = simulation('clones_refusals')
    assert sim.config.clones == 0 and sim.config.clones_batch == 2
    with pytest.raises(ValueError, match='number or designation'):
        sim.add_body(dict(ensemble.nominal), '4J-2S-1', name='463', clones=5)
    sim.add_ensemble(ensemble, resonances=['4J-2S-1'])
    with pytest.raises(ValueError, match='ensemble in this run'):
        sim.add_ensemble(ensemble, resonances=['4J-2S-1'])


@pytest.mark.parametrize('epoch', [56549.369032888, 57676.116189571, 61200.0, 57827.434632048])
def test_epoch_aligned_to_the_second_horizons_is_asked_for(epoch):
    """rebound asks Horizons for '%Y-%m-%d %H:%M:%S': the clones are drifted to that second (< 1 s earlier), and the
    aligned MJD must read back as that second, not a microsecond before it."""
    from resonances.mpcorb import datetime_from_mjd

    aligned = ensemble_run._aligned_epoch(epoch)
    assert 0 <= (epoch - aligned) * 86400 < 1.0
    moment = datetime_from_mjd(aligned)
    assert moment.strftime('%Y-%m-%d %H:%M:%S') == datetime_from_mjd(epoch).replace(microsecond=0).strftime('%Y-%m-%d %H:%M:%S')


# ----------------------------------------------------------------------------------------------- review 2026-10-27
def bare(date='2026-06-09', **kwargs) -> resonances.Simulation:
    """A run without the planets (nothing integrated): adding, transporting and bookkeeping only."""
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = f'{TEST_CACHE}/clones_bare_{stamp}'
    options = dict(date=date, save_path=path, plot_path=path, save=None, plot=None, n_cores=1)
    return resonances.Simulation(**{**options, **kwargs})


def kepler_mean_anomaly(elem: dict, days: float) -> float:
    """M after `days` on the Sun-only orbit, by REBOUND (G = 1, the Sun's mass 1, time unit 1/k day)."""
    import rebound

    sim = rebound.Simulation()
    sim.add(m=1.0)
    sim.add(m=0.0, **{key: elem[key] for key in KEPLERIAN_KEYS})
    sim.integrate(days * K)
    return sim.particles[1].orbit(primary=sim.particles[0]).M


def test_a_gap_of_seconds_moves_the_clones_on_their_ellipses(ensemble):
    """Five seconds after the covariance's epoch (below EPOCH_TOLERANCE_DAYS = 8.6 s): M advances by n x 5 s (3.8
    sigma_lambda of 463's .eq1), it is not relabelled; clones.csv says so (propagation 'kepler', 5 s in days)."""
    sim = bare(date='2026-06-09 00:00:05')
    run = sim.add_ensemble(ensemble, resonances=['4J-2S-1'])
    gap = 5 / 86400
    nominal = sim.bodies[0].initial_data
    assert nominal['epoch'] == pytest.approx(61200.0 + gap, abs=1e-9)
    for moved, draw in zip([nominal] + run.elements, [ensemble.nominal] + ensemble.clone_elements()):
        expected = kepler_mean_anomaly(draw, gap)
        assert abs((moved['M'] - expected + np.pi) % (2 * np.pi) - np.pi) < 1e-12
        assert moved['a'] == draw['a'] and moved['M'] != draw['M']
    shift = np.degrees(kepler_mean_anomaly(ensemble.nominal, gap) - ensemble.nominal['M'])
    assert shift / ensemble.solution.sigma[-1] == pytest.approx(3.83, abs=0.01)
    row = ensemble_run.clone_rows(sim)[0]
    assert row['propagated'] and row['propagation'] == 'kepler'
    assert row['propagation_days'] == pytest.approx(gap, abs=1e-9)


def test_a_long_span_is_recorded_and_warned_about(ensemble, monkeypatch):
    """A covariance 1 day from the date is integrated (IAS15; replaced here by a stub: only the bookkeeping is tested):
    clones.csv records the span, and MPC_PROPAGATE_MAX_DAYS = 0.5 (set) makes it a warning, not a refusal. Fresh draws:
    the module's ensemble memoises its real transport to that date."""
    ensemble = ensemble.solution.sample(n=3, seed=0, name='463')
    carried = []
    monkeypatch.setattr(
        ensemble_run,
        '_propagate',
        lambda elements, epoch, config: carried.append(len(elements)) or [dict(e, epoch=61201.0) for e in elements],
    )
    warnings = []
    monkeypatch.setattr(ensemble_run.logger, 'warning', warnings.append)
    previous = resonances.config.get('MPC_PROPAGATE_MAX_DAYS')
    resonances.config.set('MPC_PROPAGATE_MAX_DAYS', '0.5')
    try:
        sim = bare(date='2026-06-10')
        sim.add_ensemble(ensemble, resonances=['4J-2S-1'])
    finally:
        resonances.config.set('MPC_PROPAGATE_MAX_DAYS', previous)
    assert carried == [1, 3]  # the nominal alone, the clones together
    assert any('over 1.0 days, beyond MPC_PROPAGATE_MAX_DAYS = 0.5' in w for w in warnings)
    row = ensemble_run.clone_rows(sim)[0]
    assert row['propagated'] and row['propagation'] == 'ias15' and row['propagation_days'] == pytest.approx(1.0)
    warnings.clear()
    bare(date='2026-06-10').add_ensemble(ensemble.solution.sample(n=3, seed=0, name='463'), resonances=['4J-2S-1'])
    assert not any('MPC_PROPAGATE_MAX_DAYS' in w for w in warnings)  # empty by default: no limit


def test_transported_states_are_copies(ensemble):
    """Two runs sharing one ensemble never share element dicts: changing what one got changes neither the memo nor the
    other's."""
    sim = bare()
    nominal, clones = ensemble_run.elements_at_date(ensemble, sim.config)
    a = clones[0]['a']
    clones[0]['a'] = 99.0
    nominal['a'] = 99.0
    again_nominal, again = ensemble_run.elements_at_date(ensemble, sim.config)
    assert again[0]['a'] == a and again_nominal['a'] == ensemble.nominal['a']
    assert ensemble.clones.loc[0, 'a'] == a


def test_a_refused_duplicate_leaves_the_run_unchanged(ensemble):
    sim = bare()
    sim.add_ensemble(ensemble, resonances=['4J-2S-1'])
    with pytest.raises(ValueError, match='ensemble in this run'):
        sim.add_ensemble(ensemble, resonances=['4J-2S-1'])
    assert [b.name for b in sim.bodies] == ['463'] and len(sim.ensembles) == 1
    # add_body(..., clones=N) is refused before the body is fetched or added
    with pytest.raises(ValueError, match='ensemble in this run'):
        sim.add_body(463, '4J-2S-1', name='463', clones=3)
    assert [b.name for b in sim.bodies] == ['463']


def test_a_loaded_manifest_can_be_saved_again(ensemble, tmp_path):
    """A run's manifest has a `name` column; an ensemble loaded from it runs again and its manifest is written again."""
    sim = bare(save='all')
    run = sim.add_ensemble(ensemble, resonances=['4J-2S-1'])
    csv_path, _ = ensemble_run.write_manifest(sim, run)
    loaded = resonances.Ensemble.load(csv_path)
    assert loaded.clones['name'].tolist() == run.names
    again = bare(save='all')
    rerun = again.add_ensemble(loaded, resonances=['4J-2S-1'])
    csv_again, _ = ensemble_run.write_manifest(again, rerun)
    manifest = pd.read_csv(csv_again)
    assert manifest['name'].tolist() == run.names and list(manifest.columns).count('name') == 1
    np.testing.assert_array_equal(manifest['a_start'], pd.read_csv(csv_path)['a_start'])


def test_an_aware_run_date_is_one_instant(ensemble):
    """2026-06-09 02:00+02:00 is 2026-06-09 00:00 UTC: the date becomes that naive value, so the planets' query
    (REBOUND formats the calendar without the offset), the MJD and the clones' transport agree; a naive date is kept."""
    aware = datetime.datetime(2026, 6, 9, 2, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
    sim = bare(date=aware)
    assert sim.config.date == datetime.datetime(2026, 6, 9) and sim.config.date.tzinfo is None
    assert sim.config.date.strftime('%Y-%m-%d %H:%M:%S') == '2026-06-09 00:00:00'
    assert mjd_from_datetime(sim.config.date) == 61200.0
    run = sim.add_ensemble(ensemble, resonances=['4J-2S-1'])
    assert ensemble_run.clone_rows(sim)[0]['propagation'] == 'none'
    assert [e['M'] for e in run.elements] == [e['M'] for e in ensemble.clone_elements()]
    naive = datetime.datetime(2026, 6, 9, 2, 0)
    assert bare(date=naive).config.date == naive


def stub_candidates(monkeypatch):
    """The MMR candidates by a fixed rule: '4J-2S-1' for a > 1 AU, the Mercury co-orbital '1R-1' below. The real
    candidates come from the catalogue MATRIX_2BODY_FILE (cache/mmr-2body.csv, relative to the working directory), which
    differs between checkouts (a catalogue built with other limits gives 463 its own Mercury rows): the tests below
    must not depend on it."""
    from resonances.finder import finder as finder_module
    from resonances.mmr import mmr_finder

    def candidates(elem, planets=None, **kwargs):
        return [resonances.create_resonance('1R-1' if elem['a'] < 1 else '4J-2S-1')]

    monkeypatch.setattr(mmr_finder, 'candidate_mmrs', candidates)
    monkeypatch.setattr(finder_module, 'candidate_mmrs', candidates)


def planned_step(elements: dict, rows: list) -> tuple:
    """(step, Nout) the static dry-run plans for one body with these rows under OUTPUT_STEP=auto over 1 kyr."""
    from resonances.simulation.dry_run import prepare_run

    probe = bare(source='nasa', output_step='auto', integration_years=1000)
    probe.add_body(dict(elements), rows, name='463')
    prepare_run(probe)
    return probe.config.output_step, probe.config.Nout


def test_find_with_clones_runs_the_nominal_as_without_them(monkeypatch):
    """find with clones and OUTPUT_STEP=auto: a clone near Mercury's orbit brings the Mercury co-orbital 1R-1, which
    makes the static dry-run plan 0.1 yr instead of 1 yr for 463 (review 2026-10-27). It goes to the clones only: the
    nominal's rows and its step (1 yr, Nout 1001) are those of the same find without clones."""
    from resonances.finder import finder as finder_module
    from resonances.simulation.dry_run import prepare_run

    stub_candidates(monkeypatch)
    nominal = orbit_solution(463, source='astdys').nominal_elements()
    cloud = resonances.Ensemble.from_elements('463', [dict(nominal), {**nominal, 'a': 0.3871}], epoch_mjd=61200.0, nominal=nominal)
    monkeypatch.setattr(finder_module, 'get_body_keplerian_elements', lambda asteroid, date=None: dict(nominal))
    monkeypatch.setattr(resonances.Simulation, 'create_solar_system', lambda self, force=False: None)
    monkeypatch.setattr(ensemble_run, 'ensemble_of', lambda sim, identifier, n, name: cloud)
    found = {}
    for clones in (0, 2):
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        sim = resonances.find(
            463,
            type='mmr',
            source='nasa',
            date='2026-06-09',
            output_step='auto',
            integration_years=1000,
            clones=clones,
            save=None,
            plot=None,
            save_path=f'{TEST_CACHE}/clones_find_{stamp}',
            plot_path=f'{TEST_CACHE}/clones_find_{stamp}',
        )
        prepare_run(sim)
        found[clones] = sim
    alone, with_clones = found[0], found[2]
    assert [r.to_s() for r in alone.bodies[0].resonances()] == [r.to_s() for r in with_clones.bodies[0].resonances()] == [KEY]
    assert (with_clones.config.output_step, with_clones.config.Nout) == (alone.config.output_step, alone.config.Nout) == (1.0, 1001)
    assert with_clones.ensembles[0].keys == [KEY, '1R-1+0+0']
    assert planned_step(nominal, ['4J-2S-1', '1R-1']) == (0.1, 10001)  # what the extra row would have done to the nominal


def test_add_ensemble_without_resonances_runs_the_nominal_as_without_clones(monkeypatch):
    """add_ensemble(ens) with resonances=None follows find's rule: the nominal gets only its own candidates, so its rows
    and the auto step (1 yr, Nout 1001) are those of add_body alone; the clone near Mercury's orbit brings 1R-1 (0.1 yr
    for the nominal) to the clones only."""
    from resonances.simulation.dry_run import prepare_run

    stub_candidates(monkeypatch)
    nominal = orbit_solution(463, source='astdys').nominal_elements()
    cloud = resonances.Ensemble.from_elements('463', [dict(nominal), {**nominal, 'a': 0.3871}], epoch_mjd=61200.0, nominal=nominal)
    with_clones = bare(source='nasa', output_step='auto', integration_years=1000)
    run = with_clones.add_ensemble(cloud)
    prepare_run(with_clones)
    assert [r.to_s() for r in with_clones.bodies[0].resonances()] == [KEY]
    assert (with_clones.config.output_step, with_clones.config.Nout) == planned_step(nominal, ['4J-2S-1']) == (1.0, 1001)
    assert run.keys == [KEY, '1R-1+0+0']
    assert planned_step(nominal, ['4J-2S-1', '1R-1']) == (0.1, 10001)
