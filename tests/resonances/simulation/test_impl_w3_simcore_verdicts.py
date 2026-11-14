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

import dataclasses

import numpy as np
import pandas as pd
import pytest

from resonances.resonance.analysis.verdict import Action, AngleStatus, Confidence, LibrationOrigin, Reason, ResonanceType, Status, Verdict
from resonances.simulation import DataManager, Simulation, SimulationSerializer
from resonances.simulation.config import SavePlanets, SavePlotMode, parse_save_planets, status_selected
from tests.resonances.simulation.test_restore_and_resume import (
    CIRC,
    LIB,
    N,
    _saved_run,
    _secular_drift_run,
    _series,
    _simulation,
    _trojan_run,
)

pytestmark = pytest.mark.usefixtures('epoch_solar_cache')  # the catalogue longitudes of 2020-01-01: tests/conftest.py

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
    'review': [1, 1, 1, 1, 1, 1, 1, 0, 1],  # without a level: what nonzero selects (the level: REVIEW_ROWS below)
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


# --------------------------------------------------------------------------------------------- the review mode (W28 E1)

# By hand from the card g10-1b (the author 2026-10-31): review = nonzero, plus a non_resonant row at medium or low
# confidence (medium = spot check, low = review, the author's q1 of 19.10); the level decides on its own, so a medium
# row whose action became resample stays (benchmark 1036 9J-2S-3); certain/high without an origin is not selected.
# (status, origin, confidence, action) -> (nonzero, review)
REVIEW_ROWS = [
    (('resonant', 'dynamic', 'certain', 'none'), (1, 1)),
    (('transient', 'dynamic', 'medium', 'spot_check'), (1, 1)),
    (('non_resonant', '', 'certain', 'none'), (0, 0)),
    (('non_resonant', '', 'high', 'none'), (0, 0)),
    (('non_resonant', '', 'medium', 'spot_check'), (0, 1)),
    (('non_resonant', '', 'low', 'review'), (0, 1)),
    (('non_resonant', '', 'medium', 'resample'), (0, 1)),
    (('non_resonant', 'kinematic', 'high', 'none'), (1, 1)),
    (('uncertain', '', '', 'extend'), (1, 1)),
    (('uncertain', 'unknown', '', 'review'), (1, 1)),
    (('invalid', '', '', 'resample'), (1, 1)),
]


@pytest.mark.parametrize('row, expected', REVIEW_ROWS, ids=['-'.join(filter(None, row)) for row, _ in REVIEW_ROWS])
def test_review_selects_nonzero_and_the_doubtful_negatives(row, expected):
    status, origin, confidence, action = row
    fields = (status, origin, action, confidence)  # the order of status_selected
    assert (int(status_selected('nonzero', *fields)), int(status_selected('review', *fields))) == expected
    # the level and the action are read by review only: every other mode selects as from the status and origin alone
    for mode in set(SELECTED) - {'review'}:
        assert status_selected(mode, *fields) == status_selected(mode, status, origin), mode


def test_an_audit_flag_puts_a_certain_negative_in_review_only():
    """The 1:1 coverage audit's flags (verdict.REVIEW_FLAGS) select a non_resonant/certain row in review, as a list or
    as the ';'-joined column; another flag does not, and no other mode reads flags. The DataManager passes the
    verdict's flags (`_selection_inputs`)."""
    fields = ('non_resonant', '', 'none', 'certain')
    assert not status_selected('review', *fields)
    for flags in (['uncovered_bounded_run'], 'jump_at_edge;floor_above_own_loop'):
        assert status_selected('review', *fields, flags)
        assert not status_selected('nonzero', *fields, flags) and not status_selected('negative', *fields, flags)
    assert not status_selected('review', *fields, 'jump_at_edge')
    from resonances.body import Body

    body = Body()
    body.name = 'audit'
    key = '1E-1+0+0'
    body.statuses[key] = 'non_resonant'
    body.librations[key] = _verdict('non_resonant', '', 'certain', 'none', flags=['uncovered_bounded_run'])
    assert DataManager._selection_inputs(body, key) == ('non_resonant', '', 'none', 'certain', 'uncovered_bounded_run')
    assert DataManager._process_status('non_resonant', 'review', *DataManager._selection_inputs(body, key)[1:])


def test_review_without_a_level_is_nonzero():
    """A caller that passes no level and no action: review selects what nonzero does."""
    for status, origin in ROWS:
        assert status_selected('review', status, origin) == status_selected('nonzero', status, origin)
    assert not status_selected('review', None)


def _verdict(status, origin, confidence, action, **extra):
    return Verdict(
        resonance_type=ResonanceType.MMR,
        status=Status(status),
        angle_status=AngleStatus.CIRCULATION if status == 'non_resonant' else AngleStatus.LIBRATION,
        confidence=Confidence(confidence),
        action=Action(action),
        libration_origin=LibrationOrigin(origin),
        **extra,
    )


def test_review_columns_say_why():
    # the minimal policy leaves no caps and no failed supports: a thin veto is the last reason (`Verdict.review_reasons`)
    medium = _verdict('non_resonant', '', 'medium', 'spot_check', flags=['veto_thin'])
    assert medium.review_required
    assert medium.review_reasons == ['confidence:medium', 'veto_thin']
    unknown = _verdict('uncertain', 'unknown', '', 'review', reason=Reason.ORIGIN_UNKNOWN)
    assert unknown.review_required and unknown.review_reasons == ['action:review', 'origin_unknown']
    sure = _verdict('non_resonant', '', 'high', 'none', flags=['veto_thin'])
    assert not sure.review_required and sure.review_reasons == []
    row = medium.to_flat_dict()
    assert row['review_required'] is True and row['review_reasons'] == 'confidence:medium;veto_thin'
    # the columns follow from the others: they are not read back
    assert Verdict.from_flat_dict(row).review_reasons == medium.review_reasons


def _classified(path, **config):
    """librator (LIB and CIRC) and circulator (CIRC), classified but not saved (the arrays of plotted rows kept)."""
    sim = _simulation(path, **config)
    sim.body_manager.add_body(ELEMENTS, [LIB, CIRC], name='librator')
    sim.body_manager.add_body(ELEMENTS, [CIRC], name='circulator')
    sim.times = np.linspace(0.0, sim.config.tmax, N)
    axis, angles = _series(sim.times / (2 * np.pi))
    for body in sim.bodies:
        body.setup_vars_for_simulation(sim.times)
        body.axis[:] = axis
        body.ecc[:] = 0.1
        for resonance in body.resonances():
            body.angles_unwrapped[resonance.to_s()] = angles[resonance.to_s()].copy()
    sim.prepare_angles()
    sim.identify_librations()
    return sim


def test_save_release_and_plot_select_the_same_rows(tmp_path):
    """The circulator's CIRC row is a clear circulation, non_resonant/certain (W34, the minimal policy; under the W28
    tree it was medium by the cap catalogue_longitudes, no planets saved, now a flag). A medium copy of the same row:
    review keeps its arrays after the release and selects it for saving and plotting; a high copy is released."""
    sim = _classified(tmp_path, save='review', plot='all')
    circulator = sim.bodies[1]
    resonance = circulator.resonances()[0]
    key = resonance.to_s()
    verdict, analysis = circulator.librations[key], circulator.mmr_analyses[key]
    # W34: the minimal policy reads a clear circulation (C1, C2 full) as certain; the cap is a flag
    assert (str(verdict.status), str(verdict.confidence), str(verdict.libration_origin)) == ('non_resonant', 'certain', '')
    assert 'catalogue_longitudes' in verdict.flags and verdict.v2_verdict.split('/')[:2] == ['non_resonant', 'medium']
    assert not DataManager._arrays_dropped(analysis)

    sim.config.plot = SavePlotMode.REVIEW
    manager = sim.data_manager
    assert not manager.should_save_body(circulator, resonance) and not manager.should_plot_body(circulator, resonance)
    sim._finalise_verdict(circulator, key, dataclasses.replace(verdict, confidence=Confidence.MEDIUM, action=Action.SPOT_CHECK))
    assert manager.should_save_body(circulator, resonance) and manager.should_plot_body(circulator, resonance)
    sim._release_unplotted(circulator, resonance, analysis)
    assert not DataManager._arrays_dropped(analysis)

    sim._finalise_verdict(circulator, key, dataclasses.replace(verdict, confidence=Confidence.HIGH, action=Action.NONE))
    assert not manager.should_save_body(circulator, resonance) and not manager.should_plot_body(circulator, resonance)
    sim._release_unplotted(circulator, resonance, analysis)
    assert DataManager._arrays_dropped(analysis)

    # every row: save and plot agree under the same mode
    sim.config.save = sim.config.plot = SavePlotMode.REVIEW
    for body in sim.bodies:
        for res in body.resonances():
            assert manager.should_save_body(body, res) == manager.should_plot_body(body, res)


# W34: the circulator's row is non_resonant/certain under the minimal policy (a clear circulation; the W28 tree's
# medium by the cap catalogue_longitudes is a flag now), so review saves what nonzero saves; was {'librator', 'circulator'}
@pytest.mark.parametrize('mode, saved', [('review', {'librator'}), ('nonzero', {'librator'})])
def test_review_saves_the_queue_and_restores_it(tmp_path, mode, saved):
    sim = _saved_run(tmp_path, save=mode)
    assert {p.name for p in (tmp_path / 'data').glob('data-*.csv') if 'periodogram' not in p.name} == {f'data-{n}.csv' for n in saved}

    summary = pd.read_csv(tmp_path / 'summary.csv', dtype=str, keep_default_na=False)
    queue = {(r['name'], r['resonance']): (r['review_required'], r['review_reasons']) for _, r in summary.iterrows()}
    assert queue[('circulator', CIRC)] == ('False', '')  # W34: was ('True', 'confidence:medium;cap:catalogue_longitudes')
    assert queue[('librator', LIB)] == ('True', 'origin_unknown')  # a geometric libration: no mechanism shown
    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
    for body in restored.bodies:
        for key, verdict in body.librations.items():
            assert (str(verdict.review_required), ';'.join(verdict.review_reasons)) == queue[(str(body.name), key)]
            original = next(b for b in sim.bodies if b.name == body.name)
            assert restored.data_manager._selection_inputs(body, key) == sim.data_manager._selection_inputs(original, key)


def test_the_log_counts_the_cells_and_the_queue(tmp_path, caplog):
    sim = _classified(tmp_path, save='review', plot='review')
    with caplog.at_level('INFO'):
        sim.data_manager.log_selection(sim.bodies)
    text = caplog.text
    # W34: the minimal policy reads the libration certain and the two clear circulations certain (the W28 tree's caps,
    # catalogue_longitudes among them, are flags): was resonant/medium 1; non_resonant/medium 2 and a queue of 3
    assert 'Verdicts by status/confidence (3 rows): resonant/certain 1; non_resonant/certain 2' in text
    assert 'Review queue (summary.csv review_required): 1 of 3 rows; SAVE=review selects 1 rows, PLOT=review selects 1' in text
    assert 'SAVE=review keeps the series' not in text
    # the warning on a doubtful negative: the circulator's row at medium
    circulator = sim.bodies[1]
    key = circulator.resonances()[0].to_s()
    medium = dataclasses.replace(circulator.librations[key], confidence=Confidence.MEDIUM, action=Action.SPOT_CHECK)
    sim._finalise_verdict(circulator, key, medium)
    caplog.clear()
    with caplog.at_level('INFO'):
        sim.data_manager.log_selection(sim.bodies)
    assert 'Verdicts by status/confidence (3 rows): resonant/certain 1; non_resonant/certain 1; non_resonant/medium 1' in caplog.text
    assert 'SAVE=review keeps the series of 1 rows more than SAVE=nonzero would (2 rather than 1)' in caplog.text
