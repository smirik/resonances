"""Resume and restore of runs with clones (docs/clones.md "Resume and restore", docs/serialization.md).

Fast tests (no integration): the results archive keeps exact floats and refuses what is not a result; a torn or
foreign batch file counts as not done; the run identity follows what was run; a refusal names what changed; an
ensemble written by hand (manifest, results, clones.csv, simulation.json) is restored without integrating anything; an
old simulation.json without ensembles restores as before.

Slow tests (a few clones of 463 over 300 yr, the AstDyS solution in tests/fixtures/clones): a run stopped in clone
batch 1 (a directory where a clone's series file goes, as in test_batch_integration.py) and resumed leaves the same
clones.csv as an uninterrupted run, without running the nominal or batch 0 again; a changed setting refuses the resume;
the finished run restores with the same parents, names, starts and rows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import resonances
from resonances.clones import orbit_solution
from resonances.clones.elements import KEPLERIAN_KEYS
from resonances.simulation import clone_store, ensemble_run
from resonances.simulation.serializer import SimulationSerializer

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures' / 'clones'
KEY = '4J-2S-1+0+0-1'


@pytest.fixture(autouse=True)
def fixture_cache():
    previous = resonances.config.get('CLONES_CACHE_DIR')
    resonances.config.set('CLONES_CACHE_DIR', str(FIXTURES))
    yield
    resonances.config.set('CLONES_CACHE_DIR', previous)


def draws(n=3, seed=0):
    return orbit_solution(463, source='astdys').sample(n=n, seed=seed, name='463')


def bare(path, **kwargs) -> resonances.Simulation:
    """A run without the planets (nothing integrated): bookkeeping only."""
    options = dict(date='2026-06-09', save_path=str(path), plot_path=str(path), save='all', plot=None, n_cores=1, clones_batch=2)
    return resonances.Simulation(**{**options, **kwargs})


def a_result(a0, status='resonant', with_series=True):
    t = np.linspace(0.0, 300.0, 7)
    return {
        'a0': a0,
        'rows': {KEY: {'status': status, 'confidence': 'high', 'angle_status': 'libration', 'reason': 'r', 'libration_origin': ''}},
        'half_width': {KEY: 0.1 + 1e-17 * a0},
        'derived': [],
        'series': {KEY: (t, np.array([1.0, 359.9, 2.5, 180.0, 0.1, 33.3, 1e-7]))} if with_series else {},
    }


# ----------------------------------------------------------------------------------------------- results archives
def test_results_archive_keeps_exact_floats_and_thumbnails():
    a0 = 2.398662050276307  # 17 digits: the JSON header must give back the same bits
    arrays = clone_store.encode_results(
        {'463~c001': a_result(a0)}, {'kind': 'x'}, {'463': {KEY: (np.arange(3.0), np.array([1.0, 2.0, 3.0]))}}
    )
    header, results, nominal = clone_store.decode_results(arrays)
    got = results['463~c001']
    assert got['a0'] == a0 and got['half_width'][KEY] == 0.1 + 1e-17 * a0
    assert got['rows'] == a_result(a0)['rows']
    t, deg = got['series'][KEY]
    np.testing.assert_array_equal(t, np.linspace(0.0, 300.0, 7))  # years: float64, exact
    np.testing.assert_array_equal(deg, np.array([1.0, 359.9, 2.5, 180.0, 0.1, 33.3, 1e-7], dtype=np.float32).astype(float))
    assert header['series_precision'].startswith('years float64, degrees float32')
    np.testing.assert_array_equal(nominal['463'][KEY][1], [1.0, 2.0, 3.0])
    assert arrays['header'].dtype == np.uint8 and all(a.dtype != object for a in arrays.values())


@pytest.mark.parametrize(
    'spoil, words',
    [
        (lambda h, a: h['clones']['463~c001']['rows'][KEY].update(status='libration'), 'not a verdict row'),
        (lambda h, a: h.update(schema=99), 'schema'),
        (lambda h, a: a.update(deg_0=np.zeros(3, dtype=np.float32)), 'does not match'),
        (lambda h, a: h['clones']['463~c001'].update(a0=float('nan')), 'not finite'),
    ],
)
def test_results_archive_refuses_what_is_not_a_result(spoil, words):
    arrays = clone_store.encode_results({'463~c001': a_result(2.4)}, {'kind': 'x'})
    header = json.loads(bytes(arrays['header']).decode())
    spoil(header, arrays)
    arrays['header'] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
    with pytest.raises(ValueError, match=words):
        clone_store.decode_results(arrays)


def started_checkpoint(path, sim=None):
    sim = sim or bare(path)
    if not sim.ensembles:
        sim.add_ensemble(draws(), resonances=['4J-2S-1'])
    layout = ensemble_run.clone_batches(sim)
    identity = clone_store.run_identity(sim, layout, batched=False)
    starts = {run.parent: (run.names, clone_store.starts_array(run.elements)) for run in sim.ensembles}
    checkpoint = clone_store.CloneCheckpoint(path)
    resumed = checkpoint.open(identity, starts, resume=True, name='t')
    return sim, layout, identity, starts, checkpoint, resumed


def test_a_batch_file_counts_only_when_valid(tmp_path):
    """Written: loaded back. A '.tmp' left by a crash, a truncated file, another run's or another membership's file:
    not done (the batch runs again)."""
    _, layout, _, _, checkpoint, resumed = started_checkpoint(tmp_path)
    assert not resumed
    names = ['463~c001', '463~c002']
    keys = {name: [KEY] for name in names}
    assert [body['name'] for body in layout[0][1]] == names
    assert checkpoint.load_batch(0, names, keys) is None  # nothing yet

    checkpoint.batch_file(0).with_name('batch-00000.npz.tmp').write_bytes(b'half')  # a crash before the rename
    assert checkpoint.load_batch(0, names, keys) is None

    checkpoint.write_batch(0, names, keys, {n: a_result(2.4 + i) for i, n in enumerate(names)})
    loaded = checkpoint.load_batch(0, names, keys)
    assert sorted(loaded) == names and loaded['463~c002']['a0'] == 3.4
    assert checkpoint.load_batch(0, ['463~c001', '463~c003'], keys) is None  # other members
    assert checkpoint.load_batch(1, names, keys) is None  # another index: no file

    data = checkpoint.batch_file(0).read_bytes()
    checkpoint.batch_file(0).write_bytes(data[: len(data) // 2])  # torn
    assert checkpoint.load_batch(0, names, keys) is None

    own = checkpoint.state['hash']
    checkpoint.state['hash'] = 'another run'
    checkpoint.write_batch(0, names, keys, {n: a_result(2.4) for n in names})
    checkpoint.state['hash'] = own  # the archive is another run's
    assert checkpoint.load_batch(0, names, keys) is None


def test_the_identity_follows_what_is_run(tmp_path):
    """The same inputs give the same hash; CLONES_BATCH (layout), the seed of the draws (ensemble), the clone order
    and an ordinary body's elements change it, each in its own component."""

    def identity(sim):
        return clone_store.run_identity(sim, ensemble_run.clone_batches(sim), batched=False)

    def with_ensemble(ensemble=None, **kwargs):
        sim = bare(tmp_path, **kwargs)
        sim.add_ensemble(ensemble or draws(), resonances=['4J-2S-1'])
        return sim

    base = identity(with_ensemble())
    assert identity(with_ensemble()) == base
    changed = {
        'layout': identity(with_ensemble(clones_batch=1)),
        'settings': identity(with_ensemble(clones_seed=5)),
    }
    assert changed['layout']['components']['layout'] != base['components']['layout']
    assert changed['settings']['components']['settings'] != base['components']['settings']
    other_draws = identity(with_ensemble(draws(seed=1)))
    assert other_draws['components']['ensembles']['463'] != base['components']['ensembles']['463']
    assert other_draws['ensembles']['463']['starts'] != base['ensembles']['463']['starts']
    reordered = draws()
    reordered.clones = reordered.clones.iloc[::-1].reset_index(drop=True)
    assert identity(with_ensemble(reordered))['ensembles']['463']['starts'] != base['ensembles']['463']['starts']
    sim = with_ensemble()
    sim.bodies[0].initial_data = {**sim.bodies[0].initial_data, 'a': sim.bodies[0].initial_data['a'] + 1e-12}
    assert identity(sim)['components']['ordinary'] != base['components']['ordinary']
    assert identity(sim)['hash'] != base['hash'] and changed['layout']['hash'] != base['hash']


def test_a_resume_of_another_run_is_refused_and_says_why(tmp_path):
    sim, layout, identity, starts, checkpoint, _ = started_checkpoint(tmp_path)
    again = clone_store.CloneCheckpoint(tmp_path)
    assert again.open(identity, starts, resume=True, name='t')  # the same run: resumed

    refreshed = draws()
    refreshed.solution.provenance['sha256'] = 'f' * 64  # the provider's file fetched again
    other = bare(tmp_path)
    other.add_ensemble(refreshed, resonances=['4J-2S-1'])
    other_identity = clone_store.run_identity(other, ensemble_run.clone_batches(other), batched=False)
    with pytest.raises(ValueError, match=r'refresh=True') as error:
        clone_store.CloneCheckpoint(tmp_path).open(other_identity, starts, resume=True, name='t')
    assert 'provider_sha256' in str(error.value) and 'remove' in str(error.value)
    assert checkpoint.state_file.exists()  # nothing removed by a refusal

    moved = {parent: (names, matrix + 1e-15) for parent, (names, matrix) in starts.items()}
    with pytest.raises(ValueError, match='bit for bit'):
        clone_store.CloneCheckpoint(tmp_path).open(identity, moved, resume=True, name='t')

    assert not clone_store.CloneCheckpoint(tmp_path).open(identity, starts, resume=False, name='t')  # resume off: fresh


def test_ensemble_manifest_reads_back_to_the_bit(tmp_path):
    """MPC CAR draws (x_* columns) and the elements: '%.17g' read with round_trip gives the same floats; the default
    parser changes some of them by 1 ulp (review measurement: 2 of 6 CAR coefficients)."""
    solution = orbit_solution(463, source='mpc')
    ensemble = solution.sample(n=50, seed=3, name='463')
    csv_path, _ = ensemble.save(tmp_path / 'e')
    loaded = resonances.Ensemble.load(csv_path)
    for column in [c for c in ensemble.clones.columns if c not in ('clone', 'valid')]:
        assert ensemble.clones[column].to_numpy().tobytes() == loaded.clones[column].to_numpy().tobytes(), column
    plain = pd.read_csv(csv_path)
    assert any(
        not np.array_equal(plain[c].to_numpy(), ensemble.clones[c].to_numpy()) for c in ('x_x', 'x_y', 'x_z', 'x_vx', 'x_vy', 'x_vz', 'a')
    )


# ----------------------------------------------------------------------------------------------- restore by hand
def hand_written_run(path, save='all'):
    """A finished run's files without integrating: the ensemble added at the covariance's date, its results set by
    hand (a resonant, a transient, and a clone without a verdict), the nominal's summary row written by hand."""
    sim = bare(path, save=save, save_summary=True)
    run = sim.add_ensemble(draws(), resonances=['4J-2S-1'])
    statuses = {'463~c001': 'resonant', '463~c002': 'transient'}
    run.results = {name: a_result(elem['a'], statuses[name]) for name, elem in zip(run.names, run.elements) if name in statuses}
    run.complete = True
    body = sim.bodies[0]
    body.statuses[KEY] = 'resonant'
    body.librations[KEY] = SimulationSerializer._verdict_of_row(
        {'status': 'resonant', 'resonance_type': 'mmr', 'angle_status': 'libration', 'confidence': 'medium'}, KEY, True
    )
    Path(path).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                'name': '463',
                'resonance': KEY,
                'resonance_type': 'mmr',
                'status': 'resonant',
                'angle_status': 'libration',
                'confidence': 'medium',
            }
        ]
    ).to_csv(Path(path) / 'summary.csv', index=False)
    ensemble_run.write_outputs(sim)
    return sim, run


def test_a_hand_written_run_restores_its_ensemble(tmp_path):
    sim, run = hand_written_run(tmp_path)
    entry = json.loads((tmp_path / 'simulation.json').read_text())['simulation']['ensembles']
    assert entry['version'] == clone_store.SCHEMA
    item = entry['items'][0]
    assert (item['parent'], item['complete'], item['n_clones'], item['n_valid']) == ('463', True, 3, 3)
    assert (item['manifest'], item['results']) == ('data/ensemble-463.csv', 'data/ensemble-463-results.npz')

    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
    again = restored.ensembles[0]
    assert (again.parent, again.names, again.keys, again.clone_verdicts) == ('463', run.names, run.keys, 'stored')
    for mine, theirs in zip(again.elements, run.elements):
        assert [mine[k] for k in KEPLERIAN_KEYS] == [theirs[k] for k in KEPLERIAN_KEYS]  # to the bit
    assert again.results['463~c002']['rows'] == run.results['463~c002']['rows'] and '463~c003' not in again.results
    clones_csv = pd.read_csv(tmp_path / 'clones.csv', dtype={'name': str})
    pd.testing.assert_frame_equal(restored.clone_summary, clones_csv)
    # the rows rebuilt from the restored ensemble are the table's: 1 resonant, 1 transient, 1 missing
    row = ensemble_run.clone_rows(restored)[0]
    assert (row['n_resonant'], row['n_transient'], row['n_missing'], row['n_classified']) == (1, 1, 1, 2)
    assert row['nominal_agreement'] == 0.5 and clones_csv.loc[0, 'nominal_agreement'] == 0.5


def test_restore_without_the_manifest_gives_the_summary_only(tmp_path):
    """SAVE off: no manifest nor results archive; restore gives sim.clone_summary, no ensemble (a warning)."""
    hand_written_run(tmp_path, save=None)
    item = json.loads((tmp_path / 'simulation.json').read_text())['simulation']['ensembles']['items'][0]
    assert item['manifest'] is None and item['results'] is None
    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
    assert restored.ensembles == [] and restored.clone_summary['name'].tolist() == ['463']


def test_an_old_simulation_json_restores_as_before(tmp_path):
    hand_written_run(tmp_path)
    data = json.loads((tmp_path / 'simulation.json').read_text())
    del data['simulation']['ensembles']
    (tmp_path / 'simulation.json').write_text(json.dumps(data))
    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
    assert restored.ensembles == [] and restored.clone_summary is None
    assert [b.name for b in restored.bodies] == ['463'] and restored.bodies[0].statuses[KEY] == 'resonant'


def test_nominal_columns_from_the_counts_of_a_summary_only_row(tmp_path):
    """The agreement of a nominal reclassified as transient, from the clones.csv counts alone (1 resonant, 1
    transient, 1 missing): 1/2 with Wilson's interval for 1 of 2, computed here from the formula."""
    hand_written_run(tmp_path, save=None)
    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
    restored.bodies[0].statuses[KEY] = 'transient'
    clone_store.recompute_clone_rows(restored, tmp_path, ['463'])
    row = pd.read_csv(tmp_path / 'clones.csv', dtype={'name': str}).iloc[0]
    z = 1.959963984540054
    centre, half = (0.5 + z * z / 4) / (1 + z * z / 2), z / (1 + z * z / 2) * np.sqrt(0.25 / 2 + z * z / 16)
    assert row['nominal_status'] == 'transient' and row['nominal_agreement'] == 0.5
    assert row['nominal_agreement_lo'] == pytest.approx(centre - half, abs=1e-15)
    assert row['nominal_agreement_hi'] == pytest.approx(centre + half, abs=1e-15)
    assert (row['n_resonant'], row['n_transient'], row['n_missing']) == (1, 1, 1)


def test_a_recomputed_nominal_rewrites_both_tables(tmp_path):
    """restore(recompute_librations=True) changes the nominal's verdict: summary.csv and clones.csv must both carry it.
    The recompute is played as `Simulation._analyse_mmr` does it (the analysis held, `_finalise_verdict`), with a record
    whose e >= 1 (invalid/unphysical, from the input check: no integration). By hand: the clones are 1 resonant, 1
    transient, 1 missing, so an invalid nominal agrees with 0 of the 2 classified; another body's row stays as it was."""
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
    from resonances.resonance.analysis.settings import settings_from_config

    hand_written_run(tmp_path)
    other = {'name': 'other', 'resonance': KEY, 'resonance_type': 'mmr', 'status': 'transient', 'angle_status': 'episodic'}
    other.update(confidence='low')
    summary = pd.read_csv(tmp_path / 'summary.csv', dtype=str, keep_default_na=False)
    pd.concat([summary, pd.DataFrame([other])], ignore_index=True).to_csv(tmp_path / 'summary.csv', index=False)
    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))

    n, zero = 200, np.zeros(200)
    t = np.arange(n, dtype=float)
    record = MMRSeries(
        t, np.full(n, 3.2), np.full(n, 1.1), zero, zero, zero, zero, zero, ResonanceSpec.from_key(KEY), Regime(1.0, t[-1], n)
    )
    analysis = analyse_mmr(record, settings_from_config())
    body = restored.bodies[0]
    body.mmr_analyses[KEY] = analysis
    resonances.Simulation._finalise_verdict(body, KEY, analysis.verdict)
    clone_store.recompute_clone_rows(restored, tmp_path, [])

    summary = pd.read_csv(tmp_path / 'summary.csv', dtype=str, keep_default_na=False)
    nominal = summary[summary['name'] == '463']
    assert len(nominal) == 1 and (nominal.iloc[0]['status'], nominal.iloc[0]['reason']) == ('invalid', 'unphysical')
    clones_csv = pd.read_csv(tmp_path / 'clones.csv', dtype={'name': str, 'nominal_confidence': str}, keep_default_na=False)
    assert clones_csv.loc[0, 'nominal_status'] == nominal.iloc[0]['status'] == 'invalid'
    assert clones_csv.loc[0, 'nominal_confidence'] == nominal.iloc[0]['confidence'] == ''
    assert clones_csv.loc[0, 'nominal_agreement'] == 0.0
    kept = summary[summary['name'] == 'other'].iloc[0]
    assert (kept['status'], kept['angle_status'], kept['confidence']) == ('transient', 'episodic', 'low')


# ----------------------------------------------------------------------------------------------- real runs (slow)
def real(path, ensemble=None, **kwargs) -> resonances.Simulation:
    options = dict(
        date='2026-06-09',
        save_path=str(path),
        plot_path=str(path),
        name='clones_resume',
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
    sim.add_ensemble(ensemble or draws(), resonances=['4J-2S-1'])
    return sim


def summary_rows(path):
    summary = pd.read_csv(path / 'summary.csv', dtype=str, keep_default_na=False)
    return summary.set_index(['name', 'resonance']).sort_index()


def clones_table(path):
    return pd.read_csv(path / 'clones.csv', dtype={'name': str}).sort_values(['name', 'resonance']).reset_index(drop=True)


@pytest.fixture(scope='module')
def stopped_and_resumed(tmp_path_factory):
    """An uninterrupted run, and the same run stopped in clone batch 1 (a directory where 463~c003's series file
    goes) and resumed after the obstacle is gone; what each step left."""
    resonances.config.set('CLONES_CACHE_DIR', str(FIXTURES))  # module scope: before the function fixtures
    reference = tmp_path_factory.mktemp('reference')
    real(reference).run()
    run = tmp_path_factory.mktemp('stopped')
    sim = real(run)
    obstacle = run / 'data' / 'data-463~c003.csv'
    obstacle.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='Batch 1 failed'):
        sim.run()
    stopped = {
        'state': json.loads((run / 'clones_state' / 'clones_state.json').read_text()),
        'files': sorted(p.name for p in (run / 'clones_state').iterdir()),
        'summary': sorted(set(summary_rows(run).index.get_level_values('name'))),
        'clones_csv': (run / 'clones.csv').exists(),
        'mtimes': {name: (run / 'data' / f'data-{name}.csv').stat().st_mtime_ns for name in ('463', '463~c001', '463~c002')},
        'batch0': (run / 'clones_state' / 'batch-00000.npz').stat().st_mtime_ns,
    }
    refusals = {}
    for label, change in (('batch', dict(clones_batch=1)), ('seed', dict(ensemble=draws(seed=1)))):
        with pytest.raises(ValueError) as error:
            real(run, **change).run()
        refusals[label] = str(error.value)
    obstacle.rmdir()
    resumed = real(run)
    resumed.run()
    return reference, run, stopped, refusals, resumed


@pytest.mark.slow
def test_a_stopped_clone_run_resumes_to_the_uninterrupted_result(stopped_and_resumed):
    reference, run, stopped, refusals, _ = stopped_and_resumed
    # stopped: batch 0 published, batch 1 not; the nominal and batch 0 wrote their rows; no clones.csv yet
    assert stopped['state']['failed'] and stopped['state']['ordinary']['complete'] and 'Batch 1 failed' in stopped['state']['error']
    assert stopped['files'] == ['batch-00000.npz', 'clones_state.json', 'ordinary.npz', 'starts.npz']
    assert stopped['summary'] == ['463', '463~c001', '463~c002'] and not stopped['clones_csv']
    # a changed setting refuses the resume and names it
    assert 'clone batches' in refusals['batch'] and 'ensemble of 463' in refusals['seed']
    # resumed: the nominal and batch 0 were not run again
    assert {name: (run / 'data' / f'data-{name}.csv').stat().st_mtime_ns for name in stopped['mtimes']} == stopped['mtimes']
    rows = summary_rows(run)
    assert not rows.index.duplicated().any()
    assert set(rows.index) == {(name, KEY) for name in ('463', '463~c001', '463~c002', '463~c003')}
    # the same clones.csv and the same verdict rows as the uninterrupted run (the independent expectation)
    pd.testing.assert_frame_equal(clones_table(run), clones_table(reference))
    columns = ['status', 'angle_status', 'confidence', 'reason', 'libration_origin']
    pd.testing.assert_frame_equal(rows[columns], summary_rows(reference)[columns])
    # finished: the checkpoint is gone, simulation.json lists the complete ensemble, its figure is drawn
    assert not (run / 'clones_state').exists() and not (run / 'simulation_state.json').exists()
    item = json.loads((run / 'simulation.json').read_text())['simulation']['ensembles']['items'][0]
    assert item['complete'] and item['manifest'] == 'data/ensemble-463.csv' and item['results'] == 'data/ensemble-463-results.npz'
    assert len(list(run.rglob(f'463-{KEY}-clones.png'))) == 1


@pytest.mark.slow
def test_a_finished_run_restores_its_clones(stopped_and_resumed, tmp_path):
    reference, run, _, _, resumed = stopped_and_resumed
    restored = SimulationSerializer.restore(str(run / 'simulation.json'))
    original, again = resumed.ensembles[0], restored.ensembles[0]
    assert (again.parent, again.names, again.keys) == (original.parent, original.names, original.keys)
    for mine, theirs in zip(again.elements, original.elements):
        assert [mine[k] for k in KEPLERIAN_KEYS] == [theirs[k] for k in KEPLERIAN_KEYS]
    assert {n: r['rows'] for n, r in again.results.items()} == {n: r['rows'] for n, r in original.results.items()}
    # the rows rebuilt from the restored ensemble are clones.csv's
    pd.DataFrame(ensemble_run.clone_rows(restored)).to_csv(tmp_path / 'clones.csv', index=False)
    pd.testing.assert_frame_equal(clones_table(tmp_path), clones_table(run))
    # the figure is drawn again from what was restored
    restored.config.plot_path = str(tmp_path / 'figures-again')
    ensemble_run.plot_ensembles(restored, ensemble_run.clone_rows(restored))
    assert len(list((tmp_path / 'figures-again').rglob(f'463-{KEY}-clones.png'))) == 1
    # recomputing the nominal with the run's own settings and bits leaves clones.csv as it was
    before = clones_table(run)
    again_sim = SimulationSerializer.restore(str(run / 'simulation.json'), recompute_librations=True)
    pd.testing.assert_frame_equal(clones_table(run), before)
    assert again_sim.clone_summary['nominal_status'].tolist() == before['nominal_status'].tolist()


@pytest.mark.slow
def test_batched_ordinary_bodies_keep_their_state_until_the_clones_are_done(tmp_path):
    """The bug: the ordinary batches' state was cleared before the clones ran, so a stop in the clones ran every
    ordinary batch again. Two ordinary bodies in batches of one, then the clones stopped in batch 1."""
    ensemble = draws()

    def batched(path):
        sim = real(path, ensemble=ensemble, batch_enabled=True, batch_threshold=1, batch_size=1, resume_enabled=True)
        second = dict(ensemble.nominal, a=ensemble.nominal['a'] + 1e-4)
        sim.add_body(second, '4J-2S-1', name='second')
        return sim

    obstacle = tmp_path / 'data' / 'data-463~c003.csv'
    sim = batched(tmp_path)
    obstacle.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='Batch 1 failed'):
        sim.run()
    assert (tmp_path / 'simulation_state.json').exists()
    written = {name: (tmp_path / 'data' / f'data-{name}.csv').stat().st_mtime_ns for name in ('463', 'second', '463~c001')}
    obstacle.rmdir()
    batched(tmp_path).run()
    assert {name: (tmp_path / 'data' / f'data-{name}.csv').stat().st_mtime_ns for name in written} == written
    assert not (tmp_path / 'simulation_state.json').exists() and not (tmp_path / 'clones_state').exists()
    assert not summary_rows(tmp_path).index.duplicated().any()
    assert clones_table(tmp_path)['n_classified'].tolist() == [3]
