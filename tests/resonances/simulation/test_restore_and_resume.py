"""Restore, summary rewrites and resume, on bodies built by hand (no integration).

The series are chosen so that their verdict needs no classifier to predict:
- LIB (4J-2S-1): sigma = pi + 0.6 sin(2 pi t / 8 kyr) at the catalogue axis, with a(t) answering by the identity
  sigma_dot = (3/2) n delta_a / a (k = -1): a pendulum libration -> resonant;
- CIRC (6J-7S-1+0+0+2): the same a(t) puts the body 0.006 au from this resonance, so its angle turns at
  6 n_J - 7 n_S - n(a), once in 981 yr (about 100 turns in 100 kyr) -> non_resonant, no bounded oscillation.
With SAVE=nonzero only LIB's series is written, and only for the body that has it.
"""

import gzip
import json
import os
import pathlib
import time

import numpy as np
import pandas as pd
import pytest

import resonances
from resonances.resonance.analysis.mmr.context import table_mean_motion
from resonances.resonance.filtering import filter_angle, wrap
from resonances.simulation import Simulation, SimulationSerializer, DataManager
from resonances.simulation.parallel_executor import ParallelExecutor
from resonances.simulation.state_manager import StateManager, SimulationState

LIB, CIRC = '4J-2S-1+0+0-1', '6J-7S-1+0+0+2'
YEARS, N = 1e5, 10001
A_LIB = resonances.create_mmr(LIB).resonant_axis  # 2.3985 au
AMPLITUDE, LIBRATION_PERIOD = 0.6, 8000.0
ELEMENTS = {'a': 2.6, 'e': 0.1, 'inc': 0.1, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}
FIXTURES = pathlib.Path(__file__).parents[2] / 'fixtures' / 'free-omega'


def _simulation(path, **config):
    kwargs = dict(
        name='restore',
        date='2020-01-01',
        save='nonzero',
        plot=None,
        save_summary=True,
        save_path=str(path),
        plot_path=str(path),
        integration_years=YEARS,
        Nout=N,
    )
    kwargs.update(config)
    return Simulation(**kwargs)


def _saved_run(path, **config):
    """librator (LIB and CIRC) and circulator (CIRC only), classified and saved."""
    sim = _simulation(path, **config)
    sim.body_manager.add_body(ELEMENTS, [LIB, CIRC], name='librator')
    sim.body_manager.add_body(ELEMENTS, [CIRC], name='circulator')
    sim.times = np.linspace(0.0, sim.config.tmax, N)
    t = sim.times / (2 * np.pi)
    axis, angles = _series(t)
    for body in sim.bodies:
        body.setup_vars_for_simulation(sim.times)
        body.axis[:] = axis
        body.ecc[:] = 0.1
        for resonance in body.resonances():
            body.angles_unwrapped[resonance.to_s()] = angles[resonance.to_s()].copy()
    sim.prepare_angles()
    sim.identify_librations()
    sim.build_periodograms()
    sim.data_manager.save_data(sim.bodies, sim.times, sim)
    return sim


def _series(t):
    """a(t) of the LIB pendulum and both angles (see the module docstring); times in years."""
    omega = 2 * np.pi / LIBRATION_PERIOD
    n_lib = 2 * np.pi / A_LIB**1.5
    axis = A_LIB + (2.0 / 3.0) * A_LIB * AMPLITUDE * omega * np.cos(omega * t) / n_lib
    rate = 6 * table_mean_motion('Jupiter') - 7 * table_mean_motion('Saturn') - 2 * np.pi / axis**1.5
    circulation = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    return axis, {LIB: np.pi + AMPLITUDE * np.sin(omega * t), CIRC: circulation}


def _statuses(sim):
    return {body.name: dict(body.statuses) for body in sim.bodies}


EXPECTED = {'librator': {LIB: 'resonant', CIRC: 'non_resonant'}, 'circulator': {CIRC: 'non_resonant'}}


class TestRestore:
    def test_the_run_itself(self, tmp_path):
        sim = _saved_run(tmp_path)

        assert _statuses(sim) == EXPECTED
        # SAVE=nonzero: one data file, holding LIB's angle and not CIRC's
        # the series in the data folder, nothing of them at the top
        assert not list(tmp_path.glob('data-*.csv'))
        assert sorted(p.name for p in (tmp_path / 'data').glob('data-*.csv')) == ['data-librator-periodograms.csv', 'data-librator.csv']
        columns = pd.read_csv(tmp_path / 'data' / 'data-librator.csv', nrows=0).columns
        assert f'{LIB}_angle' in columns and f'{LIB}_angle_unwrapped' in columns and f'{CIRC}_angle' not in columns
        assert f'{LIB}_angle_filtered' not in columns  # an MMR's filtered angle is reproduced by the analysis

    def test_the_manifest_lists_only_files_on_disk(self, tmp_path):
        _saved_run(tmp_path)

        files = json.loads((tmp_path / 'simulation.json').read_text())['simulation']['data_files']

        assert files['bodies'] == ['data/data-librator.csv']
        assert files['periodograms'] == ['data/data-librator-periodograms.csv']
        assert files['summary'] == 'summary.csv' and files['mmr_numbers'] == 'mmr_numbers.csv'
        # MMRs have no TTO segments, so no segments.csv; LIB's a and sigma are coherent, so coherence.csv is written
        assert 'segments' not in files and files['coherence'] == 'coherence.csv'
        assert files['planets'] == []  # no integration here: no planet was kept
        tables = [files[name] for name in ('summary', 'coherence', 'mmr_numbers', 'episodes', 'constants') if name in files]
        for name in files['bodies'] + files['periodograms'] + tables:
            assert (tmp_path / name).exists()

    def test_default_restore_reads_the_statuses_from_the_summary(self, tmp_path):
        _saved_run(tmp_path)

        sim = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))

        assert _statuses(sim) == EXPECTED
        # the verdicts are rebuilt from the summary rows, without any analysis in memory
        assert all(str(v.status) == body.statuses[k] for body in sim.bodies for k, v in body.librations.items())
        assert all(body.mmr_analyses == {} for body in sim.bodies)
        # the saved series are back; the unsaved body has none
        librator, circulator = sim.bodies
        assert LIB in librator.angles and CIRC not in librator.angles
        assert circulator.times is None
        # the years are the saved column itself (the analysis reads them, not times / 2 pi); a live run has none
        saved = pd.read_csv(tmp_path / 'data' / 'data-librator.csv', float_precision='round_trip')['times'].to_numpy()
        np.testing.assert_array_equal(librator.times_years, saved)
        assert circulator.times_years is None
        assert resonances.Body().times_years is None

    def test_saving_after_a_default_restore_keeps_the_summary(self, tmp_path):
        _saved_run(tmp_path)
        before = pd.read_csv(tmp_path / 'summary.csv')

        sim = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
        df = sim.data_manager.get_simulation_summary(sim.bodies)
        sim.data_manager.save_simulation_summary(sim.bodies)

        # nothing is classified in memory: nothing to write, and the rows on disk stay
        assert df.empty
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'summary.csv'), before)

    def test_saving_after_a_recompute_replaces_only_the_reclassified_rows(self, tmp_path):
        _saved_run(tmp_path)
        summary = pd.read_csv(tmp_path / 'summary.csv')
        summary.loc[summary['resonance'] == LIB, 'status'] = 'uncertain'
        summary.to_csv(tmp_path / 'summary.csv', index=False)

        sim = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=True)
        sim.data_manager.save_simulation_summary(sim.bodies)

        # LIB (saved series) is classified again and written once; the CIRC rows (not saved) are kept
        after = pd.read_csv(tmp_path / 'summary.csv')
        statuses = sorted(zip(after['name'], after['resonance'], after['status']))
        assert statuses == [('circulator', CIRC, 'non_resonant'), ('librator', LIB, 'resonant'), ('librator', CIRC, 'non_resonant')]

    def test_recompute_classifies_only_the_saved_series(self, tmp_path):
        _saved_run(tmp_path)
        # Mark LIB uncertain in the summary: a recomputed status overwrites it, a restored one keeps it.
        summary = pd.read_csv(tmp_path / 'summary.csv')
        summary.loc[summary['resonance'] == LIB, 'status'] = 'uncertain'
        summary.to_csv(tmp_path / 'summary.csv', index=False)

        kept = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
        recomputed = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=True)

        assert kept.bodies[0].statuses[LIB] == 'uncertain'
        assert _statuses(recomputed) == EXPECTED
        librator, circulator = recomputed.bodies
        assert list(librator.mmr_analyses) == [LIB]  # analysed again: only the saved series
        assert librator.librations[LIB].status == 'resonant'
        assert circulator.mmr_analyses == {}

    @pytest.mark.parametrize('content', ['', '\n'])
    def test_an_empty_periodogram_file_is_skipped(self, tmp_path, content):
        _saved_run(tmp_path)
        (tmp_path / 'data' / 'data-librator-periodograms.csv').write_text(content)

        sim = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))

        assert sim.bodies[0].periodogram_frequency == {}
        assert sim.bodies[0].statuses[LIB] == 'resonant'

    def test_no_periodogram_writes_no_file(self, tmp_path):
        sim = _simulation(tmp_path, save='all')
        sim.body_manager.add_body(ELEMENTS, [LIB], name='bare')
        body = sim.bodies[0]
        body.axis_periodogram_frequency = None
        body.periodogram_frequency[LIB] = None

        sim.data_manager._save_periodogram_data(body)

        assert not list(tmp_path.rglob('data-bare-periodograms.csv'))


def _secular_drift_run(path):
    """The drifting secular angle of test_secular_plot_pipeline (an arc that never turns back: uncertain, -4), saved."""
    from tests.resonances.simulation.test_secular_plot_pipeline import DRIFT_G, synthetic_secular

    sim, body, _ = synthetic_secular(path, 'g-g6-2g7+2g8', g=DRIFT_G, forced6=0.05, free=0.1)
    sim.config.save, sim.config.plot, sim.config.save_summary = 'all', None, True
    sim.data_manager.save_data(sim.bodies, sim.times, sim)
    return body.statuses['g-g6-2g7+2g8'], body.librations['g-g6-2g7+2g8'].status_code


def _trojan_run(path):
    """591986 from the free-omega fixture: its osculating omega librates, the gate finds it kinematic (-2: non_resonant,
    kinematic libration)."""
    with gzip.open(FIXTURES / '591986.csv.gz', 'rt') as handle:
        data = pd.read_csv(handle)
    sim = _simulation(path, save='all', integration_years=5e5, Nout=len(data), coherence_enabled=False)
    sim.body_manager.add_body(ELEMENTS, 'LK', name='591986')
    body = sim.bodies[0]
    sim.times = data['times'].values * 2 * np.pi
    body.setup_vars_for_simulation(sim.times)
    body.axis, body.ecc, body.inc = data['a'].values, data['e'].values, data['inc'].values
    body.Omega, body.omega = data['Omega'].values, data['omega'].values
    body.axis_filtered = filter_angle(body.axis, sim.config)
    unwrapped = np.unwrap(data['LK_angle_unwrapped'].values)
    body.angles_unwrapped['LK'] = unwrapped
    body.angles['LK'] = wrap(unwrapped)
    body.angles_filtered_unwrapped['LK'] = filter_angle(unwrapped, sim.config)
    body.angles_filtered['LK'] = wrap(body.angles_filtered_unwrapped['LK'])
    sim.identify_librations()
    sim.data_manager.save_data(sim.bodies, sim.times, sim)
    return body.statuses['LK'], body.librations['LK'].status_code


@pytest.mark.parametrize(
    'run, key, expected, code',
    [(_secular_drift_run, 'g-g6-2g7+2g8', 'uncertain', -4), (_trojan_run, 'LK', 'non_resonant', -2)],
    ids=['secular-arc', 'free-omega-gate'],
)
def test_recompute_keeps_the_post_classification_statuses(tmp_path, run, key, expected, code):
    assert run(tmp_path) == (expected, code)

    sim = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=True)

    body = sim.bodies[0]
    assert body.statuses[key] == expected
    assert str(body.librations[key].status) == expected and body.librations[key].status_code == code


class TestSummaryRewrites:
    """A batch retried or resumed after writing its rows must replace them, not repeat them."""

    def test_drop_rows_removes_only_these_bodies(self, tmp_path):
        manager = _simulation(tmp_path).data_manager
        pd.DataFrame({'name': ['b1', 'b2', 'b2'], 'status': [2, 0, -4]}).to_csv(tmp_path / 'summary.csv', index=False)
        pd.DataFrame({'body': ['b2', 'b1'], 'line': [1, 2]}).to_csv(tmp_path / 'coherence.csv', index=False)

        assert manager.drop_rows(['b2', 'b3']) == 3

        summary = pd.read_csv(tmp_path / 'summary.csv')
        assert list(summary['name']) == ['b1'] and list(summary['status']) == [2]
        assert list(pd.read_csv(tmp_path / 'coherence.csv')['body']) == ['b1']

    def test_a_resumed_batch_leaves_one_row_per_body(self, tmp_path):
        # Batch 0 (librator) completes; batch 1 (circulator) writes its rows and dies while plotting.
        # On resume the parent drops batch 1's rows once, and the worker appends them again.
        worker = DataManager(_simulation(tmp_path / 'resumed').config)
        librator, circulator = _saved_run(tmp_path / 'run').bodies
        worker.replace_rows = False
        worker.save_simulation_summary([librator])
        worker.save_simulation_summary([circulator])

        worker.drop_rows(['circulator'])  # BatchManager, before dispatching the pending batch
        worker.save_simulation_summary([circulator])

        summary = pd.read_csv(tmp_path / 'resumed' / 'summary.csv')
        assert sorted(zip(summary['name'], summary['resonance'])) == [('circulator', CIRC), ('librator', LIB), ('librator', CIRC)]
        assert sorted(zip(summary['name'], summary['status'])) == [
            ('circulator', 'non_resonant'),
            ('librator', 'non_resonant'),
            ('librator', 'resonant'),
        ]

    def test_saving_a_run_twice_leaves_one_row_per_body_and_resonance(self, tmp_path):
        sim = _saved_run(tmp_path)
        sim.data_manager.save_data(sim.bodies, sim.times, sim)

        summary = pd.read_csv(tmp_path / 'summary.csv')
        numbers = pd.read_csv(tmp_path / 'mmr_numbers.csv')

        assert sorted(zip(summary['name'], summary['resonance'])) == [('circulator', CIRC), ('librator', LIB), ('librator', CIRC)]
        # MMR rows have no TTO segments; the per-angle table of the MMR analysis keeps one row each as well
        assert not (tmp_path / 'segments.csv').exists()
        assert not numbers.duplicated(subset=['name', 'resonance']).any()


def _failing_worker(batch_index, bodies_data, simulation_kwargs):
    """Stands in for the batch worker: batch 0 succeeds after a while, batch 1 fails at once."""
    if batch_index == 1:
        return {"batch_index": 1, "status": "failed", "body_names": [], "num_bodies": 0, "error": "boom"}
    time.sleep(2.0)
    return {"batch_index": batch_index, "status": "success", "body_names": bodies_data, "num_bodies": len(bodies_data), "error": None}


def _dying_worker(batch_index, bodies_data, simulation_kwargs):
    """Batch 0 kills its process the first time (as the OOM killer would), batch 1 is still running then."""
    marker = pathlib.Path(bodies_data[0])
    if batch_index == 0 and not marker.exists():
        marker.touch()
        os._exit(1)
    if batch_index == 1:
        time.sleep(2.0)
    return {"batch_index": batch_index, "status": "success", "body_names": [], "num_bodies": 0, "error": None}


def test_a_batch_still_running_when_another_fails_is_recorded(monkeypatch):
    # Leaving the pool waits for the running batch 0, which writes its files: it must be marked
    # complete, or a resume runs it (and appends its rows) a second time.
    monkeypatch.setattr('resonances.simulation.parallel_executor._run_batch_worker', _failing_worker)
    recorded = []

    with pytest.raises(RuntimeError, match='Batch 1 failed'):
        ParallelExecutor(n_cores=2).execute_batches([(0, ['a']), (1, ['b'])], {}, lambda index, result: recorded.append(index))

    assert recorded == [0]


def test_the_batches_retried_after_a_broken_pool_are_announced(monkeypatch, tmp_path):
    monkeypatch.setattr('resonances.simulation.parallel_executor._run_batch_worker', _dying_worker)
    marker = str(tmp_path / 'died')
    announced, recorded = [], []

    ParallelExecutor(n_cores=2).execute_batches(
        [(0, [marker]), (1, [marker])],
        {},
        lambda index, result: recorded.append(index),
        before_run=lambda pending: announced.append([index for index, _ in pending]),
    )

    # both rounds start with every unfinished batch: batch 1 was killed with the pool
    assert announced == [[0, 1], [0, 1]]
    assert sorted(recorded) == [0, 1]


class TestResumeHash:
    @staticmethod
    def _hash(path, bodies=('a', 'b', 'c'), **config):
        kwargs = dict(name='resume', date='2020-01-01', save_path=str(path), plot_path=str(path), batch_size=2, n_cores=1)
        kwargs.update(config)
        sim = Simulation(**kwargs)
        for name in bodies:
            sim.body_manager.add_body(ELEMENTS, [LIB], name=name)
        manager = sim.batch_manager
        return StateManager.compute_config_hash(manager._get_config_dict(manager.create_batches(sim.bodies)))

    def test_the_same_run_has_the_same_hash(self, tmp_path):
        assert self._hash(tmp_path) == self._hash(tmp_path)

    def test_names_paths_cores_and_plots_do_not_count(self, tmp_path):
        other = tmp_path / 'other'
        assert self._hash(tmp_path) == self._hash(other, name='renamed', n_cores=2, plot='all', image_type='pdf')

    @pytest.mark.parametrize(
        'change',
        [
            {'date': '2020-01-02'},
            {'source': 'astdys', 'date': '2020-01-01'},
            {'bodies': ('a', 'c', 'b')},
            {'bodies': ('a', 'b')},
            {'batch_size': 1},
            {'oscillations_cutoff': 0.001},
            {'libration_min_turning_points': 4},
            {'dt': 0.5},
        ],
        ids=['date', 'source', 'body-order', 'body-list', 'batch-layout', 'filter', 'classifier', 'dt'],
    )
    def test_what_changes_the_results_changes_the_hash(self, tmp_path, change):
        assert self._hash(tmp_path) != self._hash(tmp_path, **change)

    def test_without_batch_size_the_cores_set_the_layout(self, tmp_path):
        bodies = tuple(f'a{i}' for i in range(12))
        # 12 bodies: 12 // (2 * 2) = 3 per batch on two cores, 12 // (2 * 3) = 2 on three
        assert self._hash(tmp_path, bodies=bodies, batch_size=None, n_cores=2) != self._hash(
            tmp_path, bodies=bodies, batch_size=None, n_cores=3
        )

    def test_a_resume_of_another_run_is_refused(self, tmp_path):
        sim = Simulation(name='resume', date='2020-01-02', save_path=str(tmp_path), plot_path=str(tmp_path), batch_size=2)
        for name in ('a', 'b', 'c'):
            sim.body_manager.add_body(ELEMENTS, [LIB], name=name)
        # the interrupted run started a day earlier
        state = SimulationState.create_new('resume', 3, 2, 100, 1, self._hash(tmp_path / 'first', date='2020-01-01'))
        state.mark_batch_complete(0, ['a', 'b'])
        sim.batch_manager.state_manager.save_state(state)

        with pytest.raises(ValueError, match='Cannot resume'):
            sim.batch_manager.execute_batches(sim, sim.bodies, np.linspace(0, 10, 5))

    def test_a_state_of_the_previous_version_is_resumed_with_a_warning(self, tmp_path, caplog):
        import hashlib

        sim = Simulation(name='resume', date='2020-01-01', save_path=str(tmp_path), plot_path=str(tmp_path), batch_size=2)
        for name in ('a', 'b', 'c'):
            sim.body_manager.add_body(ELEMENTS, [LIB], name=name)
        # The hash the previous version wrote (its BatchManager._get_config_dict), rebuilt here by hand
        c = sim.config
        legacy = {'tmax': c.tmax, 'Nout': c.Nout, 'integrator': c.integrator, 'dt': c.dt}
        legacy.update({'batch_threshold': c.batch_threshold, 'secular_angle_mode': c.secular_angle_mode})
        legacy_hash = hashlib.md5(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
        state = SimulationState.create_new('resume', 3, 2, 100, 1, legacy_hash)
        state.mark_batch_complete(0, ['a', 'b'])
        state.mark_batch_complete(1, ['c'])
        sim.batch_manager.state_manager.save_state(state)

        with caplog.at_level('WARNING'):
            sim.batch_manager.execute_batches(sim, sim.bodies, np.linspace(0, 10, 5))
        assert any('older version' in r.getMessage() for r in caplog.records)


class TestResumeElements:
    """The resume hash covers the initial elements each body was run with (exact floats, with the sha256 of a get-orb
    answer), so that completed batches are never mixed with batches computed from a refreshed orbit
    (docs/serialization.md, "Resuming interrupted batch simulations")."""

    # A get-orb body as BodyManager._getorb_elements gives it: the elements and the sha256 of the cached answer
    OLD = {**ELEMENTS, 'a': 2.5, 'getorb_identifier': 'API-only', 'getorb_sha256': 'old-get-orb'}
    NEW = {**ELEMENTS, 'a': 2.6, 'getorb_identifier': 'API-only', 'getorb_sha256': 'new-get-orb'}

    @staticmethod
    def _sim(path, first):
        sim = Simulation(name='resume', date='2020-01-01', save_path=str(path), plot_path=str(path), batch_size=2, n_cores=1)
        sim.body_manager.add_body(dict(first), [LIB], name='API-only')
        for name in ('b', 'c'):
            sim.body_manager.add_body(ELEMENTS, [LIB], name=name)
        return sim

    @staticmethod
    def _run(sim, monkeypatch, stop_after=None):
        """execute_batches with workers that finish every batch they get (none integrated), or that stop the run
        after finishing batch `stop_after`; returns the batch indices the workers were given."""
        given = []

        def workers(batches, kwargs, progress_callback=None, before_run=None):
            for index, batch in batches:
                given.append(index)
                progress_callback(index, {'body_names': [body['name'] for body in batch], 'num_bodies': len(batch)})
                if index == stop_after:
                    raise RuntimeError('interrupted')
            return []

        monkeypatch.setattr(sim.batch_manager.parallel_executor, 'execute_batches', workers)
        if stop_after is None:
            sim.batch_manager.execute_batches(sim, sim.bodies, np.linspace(0, 10, 5))
        else:
            with pytest.raises(RuntimeError, match='interrupted'):
                sim.batch_manager.execute_batches(sim, sim.bodies, np.linspace(0, 10, 5))
        return given

    @staticmethod
    def _hash(sim):
        manager = sim.batch_manager
        return StateManager.compute_config_hash(manager._get_config_dict(manager.create_batches(sim.bodies)))

    def test_the_same_elements_give_the_same_hash(self, tmp_path):
        assert self._hash(self._sim(tmp_path, self.OLD)) == self._hash(self._sim(tmp_path, dict(self.OLD)))

    @pytest.mark.parametrize(
        'change',
        [{'a': 2.6, 'getorb_sha256': 'new-get-orb'}, {'getorb_sha256': 'new-get-orb'}, {'a': float(np.nextafter(2.5, 3.0))}],
        ids=['refreshed-get-orb', 'get-orb-answer-only', 'one-ulp-in-a'],
    )
    def test_changed_elements_change_the_hash(self, tmp_path, change):
        # the hash of the previous version (names and resonances only) was the same for all of these
        assert self._hash(self._sim(tmp_path, self.OLD)) != self._hash(self._sim(tmp_path, {**self.OLD, **change}))

    def test_a_refreshed_get_orb_orbit_is_not_mixed_with_completed_batches(self, tmp_path, monkeypatch):
        # batch 0 = API-only and b, completed from the old get-orb answer; then the run is interrupted
        assert self._run(self._sim(tmp_path, self.OLD), monkeypatch, stop_after=0) == [0]
        state = StateManager(str(tmp_path), 'resume').load_state()
        assert state.progress['completed_batches'] == [0]
        assert [record['name'] for record in state.body_elements] == ['API-only', 'b', 'c']

        refreshed = self._sim(tmp_path, self.NEW)
        with pytest.raises(ValueError, match='Cannot resume') as refusal:
            self._run(refreshed, monkeypatch)
        assert 'API-only (get-orb answer sha256 old-get-orb -> new-get-orb)' in str(refusal.value)
        assert ' b' not in str(refusal.value).split('differ')[0]  # the unchanged bodies are not named
        assert StateManager(str(tmp_path), 'resume').load_state().progress['completed_batches'] == [0]  # left as it was

    def test_the_same_elements_resume_the_pending_batches_only(self, tmp_path, monkeypatch):
        assert self._run(self._sim(tmp_path, self.OLD), monkeypatch, stop_after=0) == [0]
        assert self._run(self._sim(tmp_path, dict(self.OLD)), monkeypatch) == [1]
        assert not (tmp_path / 'simulation_state.json').exists()

    def test_a_state_written_before_the_elements_were_recorded_is_resumed_with_a_warning(self, tmp_path, monkeypatch, caplog):
        sim = self._sim(tmp_path, self.OLD)
        manager = sim.batch_manager
        # The hash of the version before this one: config and batch layout (names, resonances), no elements
        previous = StateManager.compute_config_hash(manager._get_config_dict(manager.create_batches(sim.bodies), elements=False))
        state = SimulationState.create_new('resume', 3, 2, 100, 1, previous)
        state.mark_batch_complete(0, ['API-only', 'b'])
        manager.state_manager.save_state(state)
        assert StateManager(str(tmp_path), 'resume').load_state().body_elements is None

        with caplog.at_level('WARNING'):
            assert self._run(sim, monkeypatch) == [1]
        assert any('cannot be checked' in r.getMessage() for r in caplog.records)


class TestSafeMode:
    def test_the_env_value_is_read(self):
        original = resonances.config.get('INTEGRATION_SAFE_MODE')
        try:
            resonances.config.set('INTEGRATION_SAFE_MODE', '1')
            assert Simulation().config.integration_safe_mode == 1
            resonances.config.set('INTEGRATION_SAFE_MODE', '0')
            assert Simulation().config.integration_safe_mode == 0
            resonances.config.set('INTEGRATION_SAFE_MODE', '')
            assert Simulation().config.integration_safe_mode is None
        finally:
            resonances.config.set('INTEGRATION_SAFE_MODE', original)

    @staticmethod
    def _orbits(integrator, safe_mode, check_default=None):
        import rebound

        sim = Simulation(integrator=integrator, dt=1.0, integration_safe_mode=safe_mode)
        engine = sim.integration_engine
        engine.sim = rebound.Simulation()
        engine.sim.add(m=1.0)
        engine.sim.add(m=9.5e-4, a=5.2, e=0.048, inc=0.02)
        engine.sim.add(a=2.6, e=0.1, inc=0.1, Omega=1.0, omega=2.0, M=3.0)
        engine.setup_integrator(N_active=2)
        state = engine.sim.ri_saba if integrator.startswith('SABA') else engine.sim.ri_whfast
        if check_default is not None:
            assert state.safe_mode == check_default
        out = []
        for t in np.linspace(0.0, 2000.0 * 2 * np.pi, 41)[1:]:
            engine.sim.integrate(t)
            assert state.is_synchronized
            orbit = engine.sim.particles[2].orbit(primary=engine.sim.particles[0])
            out.append([orbit.a, orbit.e, orbit.l, orbit.pomega])
        return np.array(out)

    @pytest.mark.parametrize('integrator, rebound_default', [('SABA(10,6,4)', 1), ('whfast', 0)])
    def test_unset_safe_mode_is_the_previous_default_bit_for_bit(self, integrator, rebound_default):
        # Earlier versions ran SABA with 1 and WHFast with 0: an unset value must give the same orbits exactly.
        default = self._orbits(integrator, None, check_default=rebound_default)
        np.testing.assert_array_equal(default, self._orbits(integrator, rebound_default))

    @pytest.mark.parametrize('integrator', ['SABA(10,6,4)', 'whfast'])
    def test_safe_mode_changes_the_orbits_at_round_off(self, integrator):
        # Measured with plain rebound (no package code): 0 and 1 are not bitwise equal after 2 kyr,
        # but agree to 1e-11 AU in a and 1e-9 rad in the mean longitude.
        zero, one = self._orbits(integrator, 0), self._orbits(integrator, 1)
        assert not np.array_equal(zero, one)
        np.testing.assert_allclose(zero[:, 0], one[:, 0], rtol=0, atol=1e-11)
        np.testing.assert_allclose(zero[:, 2], one[:, 2], rtol=0, atol=1e-9)


class TestStepWarning:
    def test_an_orbit_with_too_few_steps_is_reported(self, caplog):
        import rebound

        sim = Simulation(integrator='SABA(10,6,4)', dt=1.0)
        engine = sim.integration_engine
        engine.sim = rebound.Simulation()
        engine.sim.add(m=1.0)
        sim.body_manager.add_body({'a': 1.0, 'e': 0.0, 'inc': 0.0, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}, [LIB], name='fast')
        sim.body_manager.add_body({'a': 2.6, 'e': 0.0, 'inc': 0.0, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}, [LIB], name='slow')
        sim.body_manager.add_bodies_to_simulation(engine.sim)
        engine.setup_integrator(N_active=1)

        with caplog.at_level('WARNING'):
            engine.warn_unresolved_orbits(sim.bodies)

        # P = 2 pi a^1.5 in these units: 6.28 steps of dt = 1 at 1 AU, 26.2 at 2.6 AU
        messages = [r.getMessage() for r in caplog.records if 'steps of dt' in r.getMessage()]
        assert len(messages) == 1
        assert messages[0].startswith('fast: 6.3 steps')

    def test_an_adaptive_integrator_is_not_checked(self, caplog):
        import rebound

        sim = Simulation(integrator='ias15', dt=1.0)
        engine = sim.integration_engine
        engine.sim = rebound.Simulation()
        engine.sim.add(m=1.0)
        sim.body_manager.add_body({'a': 0.5, 'e': 0.0, 'inc': 0.0, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}, [LIB], name='fast')
        sim.body_manager.add_bodies_to_simulation(engine.sim)
        engine.setup_integrator(N_active=1)

        with caplog.at_level('WARNING'):
            engine.warn_unresolved_orbits(sim.bodies)

        assert not [r for r in caplog.records if 'steps of dt' in r.getMessage()]
