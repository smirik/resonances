"""Batch processing: when a run is split into batches, and one real batched run in two worker processes that stops on
a failed batch and is resumed.

The stop is a real one: one body's series file cannot be written (a directory is in its place), so its batch fails in
its worker and the parent records the batches that finished; the resume, once the obstacle is gone, must run only the
others and leave the same rows and verdicts as one unbatched run of the same bodies.
"""

import pandas as pd
import pytest

import resonances
from resonances.simulation.batch_manager import BatchManager
from resonances.simulation.config import SimulationConfig
from resonances.simulation.state_manager import StateManager

ASTEROIDS = ['463', '490', '624', '1139', '1656', '1727']
RESONANCES = ['4J-2S-1', 'lidov-kozai']  # an MMR and a ZLK row per body
KEYS = {'4J-2S-1+0+0-1', 'LK'}
FAILING = '624'  # the first body of batch 1 (batches of 2 in this order)
VERDICT = ['status', 'angle_status', 'confidence', 'reason']


@pytest.mark.parametrize(
    'config, bodies, batched',
    [
        ({'batch_enabled': True, 'batch_threshold': 3}, 4, True),
        ({'batch_enabled': True, 'batch_threshold': 3}, 3, False),  # at the threshold: one run
        ({'batch_enabled': False, 'batch_threshold': 3}, 100, False),
    ],
)
def test_when_a_run_is_batched(config, bodies, batched):
    assert BatchManager(SimulationConfig(**config)).should_batch(bodies) is batched


def _check(path, **config):
    return resonances.check(
        [int(name) for name in ASTEROIDS],
        RESONANCES,
        name='batch_resume',
        source='nasa',
        date='2026-09-29',  # fixed: the elements come from Horizons at this date
        save_path=str(path),
        plot_path=str(path),
        save='all',
        plot=None,
        save_summary=True,
        integration_years=1000,
        **config,
    )


def _catalogue_rows(path):
    summary = pd.read_csv(path / 'summary.csv', dtype=str, keep_default_na=False)
    return summary[summary['parent'] == ''].set_index(['name', 'resonance']).sort_index()


@pytest.mark.slow
def test_a_batched_run_in_two_processes_resumes_after_a_stop(tmp_path):
    run = tmp_path / 'batched'
    batched = dict(batch_enabled=True, batch_threshold=2, batch_size=2, n_cores=2, resume_enabled=True)
    sim = _check(run, **batched)
    assert [b.name for b in sim.bodies] == ASTEROIDS
    obstacle = run / 'data' / f'data-{FAILING}.csv'
    obstacle.mkdir(parents=True)  # after the Simulation chose its folder: an empty folder would be kept
    with pytest.raises(RuntimeError, match='Batch 1 failed'):
        sim.run(progress=False)

    state = StateManager(str(run), 'batch_resume').load_state()
    completed = state.progress['completed_batches']
    assert state.batch_config['total_batches'] == 3
    assert state.progress['failed'] and 0 in completed and 1 not in completed  # batch 0 ran next to the failing one
    done = [name for index in completed for name in ASTEROIDS[2 * index : 2 * index + 2]]
    written = {name: (run / 'data' / f'data-{name}.csv').stat().st_mtime_ns for name in done}
    assert set(_catalogue_rows(run).index.get_level_values('name')) == set(done)  # the failed batch wrote no row

    obstacle.rmdir()
    _check(run, **batched).run(progress=False)

    assert not (run / 'simulation_state.json').exists()  # removed after the last batch
    # the batches that had finished were not run again
    assert {name: (run / 'data' / f'data-{name}.csv').stat().st_mtime_ns for name in done} == written
    rows = _catalogue_rows(run)
    assert not rows.index.duplicated().any()
    assert set(rows.index) == {(name, key) for name in ASTEROIDS for key in KEYS}
    # every body's series in the data folder of the one run folder: no folder per batch
    assert all((run / 'data' / f'data-{name}.csv').is_file() for name in ASTEROIDS)
    assert sorted(p.name for p in run.iterdir() if p.is_dir()) == ['data']

    # one unbatched run of the same bodies: no state file, the same verdicts
    single = tmp_path / 'single'
    _check(single, batch_enabled=False).run(progress=False)
    assert not (single / 'simulation_state.json').exists()
    pd.testing.assert_frame_equal(rows[VERDICT], _catalogue_rows(single)[VERDICT])
