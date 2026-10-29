"""W3 simcore on a real integration: 463 Lola, 2 kyr of SABA(10,6,4) at dt = 1 (about 1 s per run).

A fixed-step symplectic integrator moves a test particle independently of the others, so the single run and a batch
worker integrating the same body give the same series bit for bit, and so the same rows; a restore with
recomputation reads the saved series back and must give them again. SAVE_PLANETS='analysis' keeps exactly the
planets of the angle and of the owner search (Venus..Neptune: no Mercury, no Pluto) in memory and on disk.
"""

import json

import astdys.util
import pandas as pd
import pytest

from resonances.simulation import Simulation, SimulationSerializer
from tests.tools import get_3body_elements_sample

KEYS = ['4J-2S-1+0+0-1', 'LK']
OWNER_PLANETS = ['Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']


def run(path, **config):
    kwargs = dict(
        name='lola',
        date=astdys.util.convert_mjd_to_datetime(60000),
        integration_years=2000,
        Nout=2001,
        integrator='SABA(10,6,4)',
        dt=1.0,
        save='all',
        plot=None,
        save_summary=True,
        save_path=str(path),
        plot_path=str(path),
    )
    kwargs.update(config)
    sim = Simulation(**kwargs)
    sim.create_solar_system()
    sim.add_body(get_3body_elements_sample(), ['4J-2S-1', 'LK'], name='Lola')
    sim.run()
    return sim


def rows(path):
    frame = pd.read_csv(path / 'summary.csv', dtype=str, keep_default_na=False)
    return frame.sort_values('resonance').reset_index(drop=True)


@pytest.fixture(scope='module')
def single(tmp_path_factory):
    path = tmp_path_factory.mktemp('w3-lola') / 'single'
    return run(path), path


def test_the_run_keeps_only_the_planets_of_the_analysis(single):
    sim, path = single

    assert sorted(sim.integration_engine.planets_data) == sorted(OWNER_PLANETS)
    assert sorted(p.name for p in (path / 'data').glob('data-planet-*.csv')) == sorted(f'data-planet-{p}.csv' for p in OWNER_PLANETS)
    assert json.loads((path / 'simulation.json').read_text())['simulation']['planets_saved'] == OWNER_PLANETS


def test_every_row_has_a_string_verdict(single):
    sim, path = single
    body = sim.bodies[0]
    summary = rows(path)

    assert sorted(body.statuses) == sorted(KEYS) == list(summary['resonance'])
    for _, row in summary.iterrows():
        assert row['status'] == body.statuses[row['resonance']] == str(body.librations[row['resonance']].status)
    assert summary.set_index('resonance').loc['LK', 'resonance_type'] == 'lidov_kozai'
    assert summary.set_index('resonance').loc['4J-2S-1+0+0-1', 'resonance_type'] == 'mmr'


def test_batch_and_restore_give_the_rows_of_the_single_run(single, tmp_path):
    _, single_path = single
    expected = rows(single_path)

    run(tmp_path / 'batch', batch_threshold=0, batch_size=1, n_cores=1)
    restored = SimulationSerializer.restore(str(tmp_path / 'batch' / 'simulation.json'), recompute_librations=True)
    restored.data_manager.save_simulation_summary(restored.bodies)

    batch = rows(tmp_path / 'batch')
    assert list(batch.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(batch.drop(columns=['comments'], errors='ignore'), expected.drop(columns=['comments'], errors='ignore'))
    assert restored.bodies[0].statuses == {key: status for key, status in zip(expected['resonance'], expected['status'])}


@pytest.mark.slow
def test_a_finished_run_leaves_no_lock_files(single, tmp_path):
    """The writers lock summary.csv, episodes.csv, ... through `<file>.lock`: none is left, also after two workers."""
    _, single_path = single
    sim = Simulation(
        name='lola-pair',
        date=astdys.util.convert_mjd_to_datetime(60000),
        integration_years=2000,
        Nout=2001,
        integrator='SABA(10,6,4)',
        dt=1.0,
        save='all',
        plot=None,
        save_summary=True,
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        batch_threshold=0,
        batch_size=1,
        n_cores=2,
    )
    sim.create_solar_system()
    for name in ('Lola', 'Lola-twin'):
        sim.add_body(get_3body_elements_sample(), ['4J-2S-1'], name=name)
    sim.run()

    for path in (single_path, tmp_path):
        assert (path / 'summary.csv').exists() and (path / 'constants.csv').exists()
        assert not list(path.rglob('*.lock')), path
    assert sorted(rows(tmp_path)['name']) == ['Lola', 'Lola-twin']
