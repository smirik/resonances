"""W3 simcore: every resonance type ends as a Verdict with a string status, the same in every place.

The places are `body.librations[key]` (the verdict), `body.statuses[key]` (the save/plot filters), the `status`
column of summary.csv and the statuses `restore_statuses` reads back. The rows are built without integration:
- MMR: the pendulum record of test_restore_and_resume (4J-2S-1 librating with a answering by the identity:
  resonant; 6J-7S-1+0+0+2 turning once per 981 yr: non_resonant);
- secular: the drifting g-g6-2g7+2g8 of test_secular_plot_pipeline (an arc that never turns back:
  uncertain, insufficient_time, baseline_needed, code -4);
- ZLK: 591986 from the free-omega fixture (omega librates, the free-omega gate finds it kinematic: -2, non_resonant).
The expected codes are those the hooks' own tests establish; here they must arrive as strings everywhere.
"""

import numpy as np
import pandas as pd
import pytest

from resonances.resonance.analysis.verdict import Status, Verdict
from resonances.simulation import DataManager, Simulation, SimulationSerializer
from resonances.simulation.config import SavePlanets, parse_save_planets, status_selected
from tests.resonances.simulation.test_restore_and_resume import CIRC, LIB, _saved_run, _secular_drift_run, _trojan_run

STATUS_STRINGS = {str(s) for s in Status}


def _mmr(path):
    sim = _saved_run(path)
    return sim, {('librator', LIB): ('resonant', 2), ('librator', CIRC): ('non_resonant', 0), ('circulator', CIRC): ('non_resonant', 0)}


def _secular(path):
    _secular_drift_run(path)
    return None, {('synthetic', 'g-g6-2g7+2g8'): ('uncertain', -4)}


def _zlk(path):
    _trojan_run(path)
    return None, {('591986', 'LK'): ('non_resonant', -2)}


@pytest.mark.parametrize(
    'build',
    [_mmr, _secular, pytest.param(_zlk, marks=pytest.mark.benchmark)],  # _zlk reads the 591986 series (local only)
    ids=['mmr', 'secular', 'lidov_kozai'],
)
def test_one_status_string_in_every_place(tmp_path, build):
    _, expected = build(tmp_path)

    summary = pd.read_csv(tmp_path / 'summary.csv', dtype=str, keep_default_na=False)
    in_file = {(row['name'], row['resonance']): (row['status'], int(row['status_code'])) for _, row in summary.iterrows()}
    assert in_file == expected

    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
    recomputed = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=True)
    for sim in (restored, recomputed):
        for body in sim.bodies:
            for key, status in body.statuses.items():
                verdict = body.librations[key]
                assert isinstance(verdict, Verdict) and isinstance(status, str) and status in STATUS_STRINGS
                assert status == str(verdict.status) == expected[(str(body.name), key)][0]
                assert verdict.status_code == expected[(str(body.name), key)][1]


def test_the_run_writes_strings_in_memory(tmp_path):
    sim, expected = _mmr(tmp_path)

    for body in sim.bodies:
        for key, status in body.statuses.items():
            assert (status, body.librations[key].status_code) == expected[(body.name, key)]
            assert body.mmr_analyses[key].verdict is body.librations[key]


def test_coherence_is_skipped_for_a_non_resonant_mmr(tmp_path):
    """`_analyse_coherence` reads the verdict's Status: the circulating row is skipped, the librating one is not."""
    sim, _ = _mmr(tmp_path)
    librator, circulator = sim.bodies

    assert LIB in librator.coherence and CIRC not in librator.coherence
    assert circulator.coherence == {}


def test_an_old_summary_with_integer_statuses(tmp_path):
    """A summary.csv of the previous version: secular/ZLK integers are translated (SPEC §5.7), an MMR integer has no
    translation and is skipped, with one warning."""
    sim = Simulation(name='old', date='2020-01-01', save_path=str(tmp_path), plot_path=str(tmp_path))
    sim.body_manager.add_body({'a': 2.6, 'e': 0.1, 'inc': 0.1, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}, ['4J-2S-1', 'g-g6', 'LK'], name='old')
    pd.DataFrame({'name': ['old', 'old', 'old'], 'resonance': ['4J-2S-1+0+0-1', 'g-g6', 'LK'], 'status': [2, -6, -2]}).to_csv(
        tmp_path / 'summary.csv', index=False
    )

    count = SimulationSerializer.restore_statuses(sim.bodies, tmp_path / 'summary.csv')

    body = sim.bodies[0]
    assert count == 2
    assert body.statuses == {'g-g6': 'uncertain', 'LK': 'non_resonant'}
    assert body.librations['g-g6'].status_code == -6 and body.librations['LK'].status_code == -2


# --------------------------------------------------------------------------------------------- save/plot modes

# SPEC §11.5, by hand: (status, libration_origin) -> selected under each mode
ROWS = [
    ('resonant', 'dynamic'),
    ('transient', 'dynamic'),
    ('near_separatrix', 'dynamic'),
    ('uncertain', 'unknown'),
    ('uncertain', ''),
    ('non_resonant', 'kinematic'),
    ('non_resonant', 'induced'),
    ('non_resonant', ''),
    ('invalid', ''),
]
SELECTED = {
    'all': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    'none': [0, 0, 0, 0, 0, 0, 0, 0, 0],
    'resonant': [1, 1, 0, 0, 0, 0, 0, 0, 0],
    'candidates': [1, 1, 1, 1, 1, 0, 0, 0, 0],
    'extended': [1, 1, 1, 1, 1, 1, 1, 0, 0],
    'nonzero': [1, 1, 1, 1, 1, 1, 1, 0, 1],
    'negative': [0, 0, 1, 1, 1, 0, 0, 0, 1],
}


@pytest.mark.parametrize('mode', sorted(SELECTED))
def test_save_and_plot_modes_on_strings(mode):
    assert [int(status_selected(mode, status, origin)) for status, origin in ROWS] == SELECTED[mode]


def test_no_verdict_is_selected_only_by_all():
    assert status_selected('all', None) and not status_selected('nonzero', None) and not status_selected(None, 'resonant')


# --------------------------------------------------------------------------------------------- planets


@pytest.mark.parametrize(
    'value, expected',
    [(True, 'all'), (False, 'none'), ('True', 'all'), ('False', 'none'), ('analysis', 'analysis'), ('ALL', 'all'), (None, 'none')],
)
def test_save_planets_accepts_the_old_bool(value, expected):
    assert parse_save_planets(value) == expected


def test_save_planets_none_is_falsy_and_a_typo_is_an_error():
    assert not SavePlanets.NONE and SavePlanets.ANALYSIS and SavePlanets.ALL
    with pytest.raises(ValueError, match='all, analysis, none'):
        parse_save_planets('some')


def test_measured_phases_need_every_planet():
    with pytest.raises(ValueError, match="save_planets='all'"):
        Simulation(secular_planet_phases='measured', save_planets='analysis')
    assert Simulation(secular_planet_phases='measured', save_planets=True).config.save_planets == 'all'


OWNER_PLANETS = ['Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
ELEMENTS = {'a': 2.6, 'e': 0.1, 'inc': 0.1, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}


@pytest.mark.parametrize(
    'resonances, config, expected',
    [
        (['4J-2S-1'], {}, OWNER_PLANETS),  # the angle's planets and the owner candidates (no Mercury, no Pluto)
        (['4J-2S-1'], {'mmr_owner_search': 'off'}, ['Jupiter', 'Saturn']),
        (['2J-1', 'g-g6'], {'mmr_owner_search': 'off'}, ['Jupiter', 'Saturn']),  # nu6 reads Saturn's mode
        (['g-g6'], {}, ['Saturn']),  # no MMR: no owner planets
        (['LK'], {}, []),
        (['LK'], {'save_planets': 'all'}, ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']),
        (['4J-2S-1'], {'save_planets': False}, []),
    ],
)
def test_the_planets_the_analysis_needs(resonances, config, expected):
    sim = Simulation(name='planets', date='2020-01-01', save_planets=config.pop('save_planets', 'analysis'), **config)
    sim.body_manager.add_body(ELEMENTS, resonances, name='body')

    assert sim.planets_needed() == expected


def test_a_batch_worker_keeps_the_planets_of_the_whole_run():
    sim = Simulation(name='planets', date='2020-01-01', _planets_needed=['Saturn', 'Jupiter'])
    sim.body_manager.add_body(ELEMENTS, ['LK'], name='body')

    assert sim.planets_needed() == ['Jupiter', 'Saturn']  # the parent's list, in the planets' order


def test_the_mmr_periodogram_reads_the_analysis_series(tmp_path, monkeypatch):
    """Run order prepare -> identify -> periodograms: the MMR periodogram is computed on sigma_f of the analysis
    (the figures and the release of the analysis' arrays after them are skipped)."""
    monkeypatch.setattr(DataManager, 'plot_body', lambda manager, body, simulation=None: None)
    monkeypatch.setattr(DataManager, 'release_arrays', staticmethod(lambda body: None))
    sim = _saved_run(tmp_path, plot='all')  # a row to be plotted keeps its arrays (an unplotted one is released)
    librator = sim.bodies[0]
    prep = librator.mmr_analyses[LIB].prep

    assert np.array_equal(librator.angles_filtered_unwrapped[LIB], prep.angle_f)
    assert not np.array_equal(librator.angles_filtered_unwrapped[LIB], sim._filter(librator.angles_unwrapped[LIB]))
    assert librator.periodogram_frequency[LIB] is not None


def test_a_restore_analyses_with_the_saved_settings_unless_asked_otherwise(tmp_path):
    """SPEC §2.6: recomputation uses the settings of simulation.json; `current_settings=True` the current ones."""
    sim = _saved_run(tmp_path, mmr_filter_synodic_factor=3.0)
    assert sim.config.mmr_filter_synodic_factor == 3.0
    default = Simulation(name='default').config.mmr_filter_synodic_factor

    saved = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
    current = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), current_settings=True)

    assert saved.config.mmr_filter_synodic_factor == 3.0
    assert current.config.mmr_filter_synodic_factor == default == 1.5
    assert current.config.tmax == saved.config.tmax  # the rest of the configuration is the saved one


def _pendulum_body(path, backward: bool):
    """The LIB record of test_restore_and_resume on a body, forward (t = 0..100 kyr) or as a backward run would
    store it (t = 0, -10, ..., -100 kyr: the same record shifted by -100 kyr, rows in decreasing time)."""
    from tests.resonances.simulation.test_restore_and_resume import N, YEARS, _series, _simulation

    sim = _simulation(path, integration_years=-YEARS if backward else YEARS, plot='all')  # keep the arrays
    sim.body_manager.add_body({'a': 2.4, 'e': 0.1, 'inc': 0.1, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}, [LIB], name='pendulum')
    t = np.linspace(0.0, YEARS, N)
    axis, angles = _series(t)
    order = slice(None, None, -1) if backward else slice(None)
    sim.times = (t[order] - (YEARS if backward else 0.0)) * 2 * np.pi
    body = sim.bodies[0]
    body.setup_vars_for_simulation(sim.times)
    body.axis[:], body.ecc[:] = axis[order], 0.1
    body.angles_unwrapped[LIB] = angles[LIB][order].copy()
    sim.prepare_angles()
    sim.identify_librations()
    return body


def test_a_backward_run_is_analysed_in_time_order_and_stored_in_its_own(tmp_path):
    forward = _pendulum_body(tmp_path / 'forward', backward=False)
    backward = _pendulum_body(tmp_path / 'backward', backward=True)

    assert forward.statuses[LIB] == backward.statuses[LIB] == 'resonant'
    f, b = forward.mmr_analyses[LIB], backward.mmr_analyses[LIB]
    assert b.verdict.reason == f.verdict.reason and b.verdict.angle_status == f.verdict.angle_status
    # the analysis read the same record (times shifted by -100 kyr); the body keeps its own row order
    assert np.allclose(b.prep.angle_f, f.prep.angle_f, atol=1e-9)
    assert np.allclose(backward.angles_filtered_unwrapped[LIB], forward.angles_filtered_unwrapped[LIB][::-1], atol=1e-9)


def test_a_derived_row_never_takes_over_a_resonance_of_the_body():
    from resonances.body import Body

    body = Body()
    body.mmrs = [resonances_create('9M-16+0+7'), resonances_create('9M-16+2+5')]

    assert body.add_derived_resonance('9M-16+2+5', '9M-16+0+7') is None  # the user's own row
    assert body.derived_resonances == {}
    added = body.add_derived_resonance('9M-16+1+6', '9M-16+0+7', angle_wrapped=[0.1, 0.2])
    assert added.to_s() == '9M-16+1+6' and body.derived_resonances == {'9M-16+1+6': '9M-16+0+7'}
    assert body.add_derived_resonance('9M-16+1+6', '9M-16+0+7') is added  # idempotent
    assert [r.to_s() for r in body.mmrs] == ['9M-16+0+7', '9M-16+2+5', '9M-16+1+6']
    body.remove_derived_resonance('9M-16+1+6')
    assert [r.to_s() for r in body.mmrs] == ['9M-16+0+7', '9M-16+2+5'] and '9M-16+1+6' not in body.angles


def resonances_create(key):
    from resonances.resonance.factory import create_mmr

    return create_mmr(key)
