"""W3 simcore: one MMR record through every path of the package gives the same rows.

The record is a saved benchmark example put on a Body exactly as the integration leaves it (raw angle in
`angles_unwrapped`, elements, the kept planets in `integration_engine.planets_data`), so the package path
prepare_angles -> identify_librations -> build_periodograms -> save_data -> restore runs on a real record in about
a second, without integrating. 2003 UG22 10E-17+0+7 (4000 rows, 20 kyr at 5 yr, all planets) is used because,
with test E' switched off (mmr_subresonance_strength_rel_min=0), its analysis accepts the planar subresonance
10E-17+7+0 as a row of its own (W2-REPORT): the derived row has to survive the single run, the batch worker,
simulation.json and both kinds of restore. The default rejects that member (its factor (e_E/e)^7 is 1e-10, W6 C3);
the override also checks that a setting reaches every path.

The batch path runs the real BatchManager/ParallelExecutor (sequential, in this process) with the integration
replaced by the same record (`fill_bodies`), which is the one stand-in; everything else is the package's code.
"""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from resonances.resonance.analysis.mmr import io
from resonances.simulation import Simulation, SimulationSerializer
from resonances.simulation.body_manager import BodyManager
from resonances.simulation.integration import IntegrationEngine

# 2003 UG22 10E-17+0+7: 4000 rows (20 kyr at 5 yr), all planets saved; without test E' its analysis accepts the
# planar subresonance 10E-17+7+0 as a row of its own (W2-REPORT), so it carries a derived row through every path.
WITHOUT_E_PRIME = {'mmr_subresonance_strength_rel_min': 0.0}
UG22 = 'mmr2__2003_UG22__10E-17+0+7'


def example_available(example_id: str = UG22) -> bool:
    return (io.example_path(example_id) / 'meta.json').exists()


def simulation_from_example(path, example_id: str = UG22, **config) -> Simulation:
    """A Simulation (not run) with one body holding the example's record, ready for `fill_from_example`."""
    series = io.load_series(example_id)
    kwargs = dict(
        name='w3-simcore',
        date='2026-06-09',
        save='all',
        plot=None,
        save_summary=True,
        save_path=str(path),
        plot_path=str(path),
        integration_years=float(series.t_yr[-1] - series.t_yr[0]),
        Nout=len(series.t_yr),
        coherence_enabled=False,
        **WITHOUT_E_PRIME,
    )
    kwargs.update(config)
    sim = Simulation(**kwargs)
    elements = {'a': float(series.a[0]), 'e': float(series.e[0]), 'inc': float(series.inc[0])}
    elements.update({'Omega': float(series.Omega[0]), 'omega': float(series.omega[0]), 'M': 0.0})
    sim.body_manager.add_body(elements, [series.resonance.key], name=series.name.replace(' ', '_'))
    sim.times = np.linspace(0.0, sim.config.tmax, sim.config.Nout)  # as Simulation.run() builds them
    return sim


def fill_from_example(sim: Simulation, example_id: str = UG22) -> None:
    """What `IntegrationEngine.run_integration` leaves: elements, the raw angle, the kept planets."""
    fill_bodies(sim.bodies, sim.times, sim.integration_engine, sim.planets_needed(), example_id)


def fill_bodies(bodies, times, engine, kept, example_id: str = UG22) -> None:
    """Stand-in for `IntegrationEngine.run_integration(bodies, times, planets=kept)` on the example's record."""
    series = io.load_series(example_id)
    for body in bodies:
        body.setup_vars_for_simulation(times)
        body.axis[:], body.ecc[:], body.inc[:] = series.a, series.e, series.inc
        body.Omega[:], body.omega[:], body.varpi[:] = series.Omega, series.omega, series.varpi
        body.M[:] = series.mean_anomaly if series.mean_anomaly is not None else 0.0
        body.longitude[:] = series.mean_longitude if series.mean_longitude is not None else 0.0
        for resonance in body.resonances():
            body.angles_unwrapped[resonance.to_s()][:] = series.angle_unwrapped_numpy
    engine.planets_data = {}
    for name, planet in series.planets.items():
        if name in kept:
            engine.planets_data[name] = {
                'times': np.asarray(times) / (2 * np.pi),
                'a': planet.a,
                'e': planet.e,
                'inc': planet.inc,
                'Omega': planet.Omega,
                'varpi': planet.varpi,
                'l': planet.mean_longitude,
            }


def run_example(sim: Simulation, example_id: str = UG22) -> Simulation:
    """The steps of `Simulation._run_single` after the integration, on the example's record."""
    fill_from_example(sim, example_id)
    sim.prepare_angles()
    sim.identify_librations()
    sim.build_periodograms()
    sim.data_manager.save_data(sim.bodies, sim.times, sim)
    return sim


pytestmark = pytest.mark.skipif(not example_available(), reason='benchmark example data not present (benchmark/data)')

PARENT, DERIVED = '10E-17+0+7', '10E-17+7+0'
VERDICT_COLUMNS = [
    'name',
    'resonance',
    'resonance_type',
    'added_by',
    'parent',
    'status',
    'angle_status',
    'reason',
    'libration_origin',
    'owner',
    'confidence',
    'multiplet_status',
    'status_code',
    'flags',
    'action',
]


def summary_rows(path) -> pd.DataFrame:
    frame = pd.read_csv(path / 'summary.csv', dtype=str, keep_default_na=False)
    return frame.sort_values(['name', 'resonance']).reset_index(drop=True)


def assert_same_rows(left: pd.DataFrame, right: pd.DataFrame):
    """The same rows, text for text: the paths read the same record bit for bit (restore parses the saved series
    with float_precision='round_trip'), so not even the last digit of a number may differ."""
    assert list(left['resonance']) == list(right['resonance'])
    assert left[VERDICT_COLUMNS].equals(right[VERDICT_COLUMNS])
    pd.testing.assert_frame_equal(left, right)


@pytest.fixture(scope='module')
def single(tmp_path_factory):
    """The single run, once for the module; tests that write into a run folder work on a copy (`copy_of`)."""
    path = tmp_path_factory.mktemp('w3-simcore') / 'single'
    return run_example(simulation_from_example(path)), path


def copy_of(path, tmp_path):
    target = tmp_path / 'copy'
    shutil.copytree(path, target)
    simulation_json = target / 'simulation.json'
    data = json.loads(simulation_json.read_text())
    data['config']['save_path'] = data['config']['plot_path'] = str(target)
    simulation_json.write_text(json.dumps(data))
    return target


@pytest.mark.benchmark
def test_the_single_run_adds_the_derived_row(single):
    sim, path = single
    body = sim.bodies[0]

    assert [r.to_s() for r in body.resonances()] == [PARENT, DERIVED]
    assert body.derived_resonances == {DERIVED: PARENT}
    assert body.resonance_origin(DERIVED) == ('subresonance_check', PARENT)
    for key in (PARENT, DERIVED):  # one writer: the verdict and the status agree, and the status is a string
        assert body.statuses[key] == str(body.librations[key].status)
        assert body.mmr_analyses[key].verdict is body.librations[key]
    # both angles librate here (rule C, W9: sub_members_locked = 1): the catalogue row keeps its own verdict
    parent = body.mmr_analyses[PARENT]
    assert parent.extra_columns['sub_members_locked'] == 1
    assert body.librations[PARENT].libration_origin != 'subresonance' and body.librations[PARENT].owner != DERIVED
    # the derived row's own angle is saved (it is sigma0 + Delta_s, not the parent's)
    columns = pd.read_csv(path / 'data' / 'data-2003_UG22.csv', nrows=0).columns
    assert f'{DERIVED}_angle' in columns and f'{DERIVED}_angle_unwrapped' in columns
    rows = summary_rows(path)
    assert list(zip(rows['resonance'], rows['added_by'], rows['parent'])) == [
        (PARENT, '', ''),
        (DERIVED, 'subresonance_check', PARENT),
    ]
    entries = json.loads((path / 'simulation.json').read_text())['simulation']['bodies'][0]['resonances']
    assert [(e['key'], e.get('parent')) for e in entries] == [(PARENT, None), (DERIVED, PARENT)]


@pytest.mark.benchmark
def test_a_default_restore_gives_the_same_strings(single, tmp_path):
    sim, path = single
    restored = SimulationSerializer.restore(str(copy_of(path, tmp_path) / 'simulation.json'))
    body, original = restored.bodies[0], sim.bodies[0]

    assert body.derived_resonances == {DERIVED: PARENT}
    assert body.statuses == original.statuses
    for key in (PARENT, DERIVED):
        assert body.librations[key].to_flat_dict()['status'] == original.librations[key].to_flat_dict()['status']
        assert body.librations[key].libration_origin == original.librations[key].libration_origin


@pytest.mark.benchmark
def test_a_restore_with_recomputation_gives_the_same_rows(single, tmp_path):
    _, original = single
    expected = summary_rows(original)
    path = copy_of(original, tmp_path)

    restored = SimulationSerializer.restore(str(path / 'simulation.json'), recompute_librations=True)
    body = restored.bodies[0]
    assert set(body.mmr_analyses) == {PARENT, DERIVED}  # analysed again from the raw saved angle and elements
    restored.data_manager.save_simulation_summary(restored.bodies)

    assert_same_rows(summary_rows(path), expected)


@pytest.mark.benchmark
def test_the_batch_path_gives_the_same_rows(single, tmp_path, monkeypatch):
    _, single_path = single

    def integrate(engine, bodies, times, progress=False, planets=None):
        fill_bodies(bodies, times, engine, planets)

    monkeypatch.setattr(IntegrationEngine, 'run_integration', integrate)
    monkeypatch.setattr(IntegrationEngine, 'create_solar_system', lambda engine, force=False: None)
    monkeypatch.setattr(BodyManager, 'add_bodies_to_simulation', lambda manager, rebound_sim: None)
    sim = simulation_from_example(tmp_path / 'batch', batch_enabled=True, batch_threshold=0, batch_size=1, n_cores=1)
    sim.run()

    body = sim.bodies[0]
    # the parent never analysed: the derived row came back from the worker, the verdicts from summary.csv
    assert body.derived_resonances == {DERIVED: PARENT}
    assert body.mmr_analyses == {}
    assert body.statuses == {key: row for key, row in zip(summary_rows(single_path)['resonance'], summary_rows(single_path)['status'])}
    entries = json.loads((tmp_path / 'batch' / 'simulation.json').read_text())['simulation']['bodies'][0]['resonances']
    assert [(e['key'], e.get('parent')) for e in entries] == [(PARENT, None), (DERIVED, PARENT)]
    assert_same_rows(summary_rows(tmp_path / 'batch'), summary_rows(single_path))


@pytest.mark.benchmark
def test_a_resumed_batch_brings_the_derived_row_back_from_the_summary(single, tmp_path):
    """A batch completed before an interruption returns nothing to the resumed parent: restore_statuses re-adds the
    derived row from the summary's `parent` column."""
    _, single_path = single
    sim = simulation_from_example(tmp_path / 'resumed')
    body = sim.bodies[0]

    assert SimulationSerializer.restore_statuses(sim.bodies, single_path / 'summary.csv') == 2
    assert body.derived_resonances == {DERIVED: PARENT}
    assert [r.to_s() for r in body.resonances()] == [PARENT, DERIVED]


@pytest.mark.benchmark
def test_the_analysis_runs_again_without_duplicating_the_derived_row(tmp_path):
    sim = run_example(simulation_from_example(tmp_path))
    sim.identify_librations()

    body = sim.bodies[0]
    assert [r.to_s() for r in body.resonances()] == [PARENT, DERIVED]
    assert set(body.mmr_analyses) == {PARENT, DERIVED}
