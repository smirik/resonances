"""W3 'output': what the package writes from the verdicts (SPEC §2.6, §11; DELTA 3/5/9).

summary.csv with string statuses and `resonance_type` (the old `type` overwrite is gone), mmr_numbers.csv,
episodes.csv, candidates.csv and constants.csv (one per run), the save/plot modes on strings (§11.5), a derived
subresonance row with its own summary row and figure, and the verdict figure in combined/<status>/, redrawn after
a restore by re-running the analysis and checking it against summary.csv.

The body is built by hand, not integrated (fast suite): a 2:1 inner resonance with Jupiter whose angle librates
about pi with half-amplitude 0.6 rad and period 400 yr while a follows (the libration case), or an angle that
circulates once per 700 yr (the circulation case). Jupiter moves on a circle at a = 5.2026 au, so its mean motion
is 360 / 5.2026**1.5 = 30.336 deg/yr (Kepler's third law in Gaussian years), independent of the analysis.
"""

import dataclasses

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from resonances.resonance.analysis.mmr.models import ResonanceSpec  # noqa: E402
from resonances.resonance.analysis.settings import settings_hash  # noqa: E402
from resonances.resonance.analysis.verdict import (  # noqa: E402
    MMR_CHECKS,
    CheckValue,
    LibrationOrigin,
    ResonanceType,
    Status,
    Verdict,
    AngleStatus,
    verdict_from_status_code,
)
from resonances.simulation import data_manager as data_manager_module  # noqa: E402
from resonances.simulation.data_manager import DataManager  # noqa: E402
from resonances.simulation.serializer import SimulationSerializer  # noqa: E402
from resonances.simulation.simulation import Simulation  # noqa: E402

A_BODY, A_JUPITER = 3.2776, 5.2026
LIBRATION_PERIOD = 400.0  # yr
LIBRATION_CENTRE = 0.0  # rad
CIRCULATION_PERIOD = 700.0  # yr
KEY = '2J-1+0-1'  # the package key of 2J-1
DERIVED_KEY = '2J-1-1+0'  # its member with Jupiter's perihelion: sigma0 + varpi - varpi_J (both 0 here)
JUPITER_N_DEG_YR = 360.0 / A_JUPITER**1.5  # 30.336 deg/yr
# a follows sigma through sigma_dot = 2 n_J - n(a): |dn/da| = 1.5 n / a with n = 2 pi / a^1.5 rad/yr, so the swing of
# a that carries sigma_dot = 0.6 (2 pi / 400) cos(phase) is 0.6 (2 pi / 400) / (1.5 n / a) = 0.0194 au (W16: the
# identity is read, a 10x smaller swing makes it fail)
A_SWING = 0.6 * (2 * np.pi / LIBRATION_PERIOD) / (1.5 * (2 * np.pi / A_BODY**1.5) / A_BODY)


def synthetic(tmp_path, circulate=False, n=4000, years=4000.0, name='synthetic', **config):
    """A simulation with one hand-built body in 2J-1 (see the module docstring); nothing integrated."""
    kwargs = dict(
        name='w3_output',
        save='all',
        plot=None,
        plots=['combined'],
        save_summary=True,
        save_planets='all',
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        integration_years=years,
        Nout=n,
    )
    if 'plot_layout' not in config and 'plot_subfolder_strategy' not in config:
        kwargs['plot_layout'] = 'flat'  # the figure paths of these tests are written for it (default confidence/status)
    kwargs.update(config)
    sim = Simulation(**kwargs)
    add_synthetic_body(sim, n, circulate, name)
    return sim


def add_synthetic_body(sim, n, circulate, name):
    sim.body_manager.add_body({'a': A_BODY, 'e': 0.1, 'inc': 0.05, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}, '2J-1', name=name)
    sim.times = np.linspace(0.0, sim.config.tmax, n)
    body = sim.bodies[-1]
    body.times = sim.times
    t = sim.times / (2 * np.pi)
    phase = 2 * np.pi * t / LIBRATION_PERIOD
    # the libration is about 0, where Jupiter's directly averaged potential has its minimum for 2J-1 (about 352 deg):
    # about pi it would lie outside that well, and with L4 and R0 not measured the failed strength check would make
    # the row uncertain (W18)
    sigma = 2 * np.pi * t / CIRCULATION_PERIOD if circulate else LIBRATION_CENTRE + 0.6 * np.sin(phase)
    jupiter_longitude = 2 * np.pi * t / A_JUPITER**1.5
    longitude = 2 * jupiter_longitude - sigma  # sigma = 2 lambda_J - lambda - varpi with varpi = 0
    body.axis = A_BODY + (0.0 if circulate else A_SWING) * np.cos(phase)
    body.ecc = np.full(n, 0.1)
    body.inc = np.full(n, 0.05)
    for element in ('Omega', 'omega', 'varpi'):
        setattr(body, element, np.zeros(n))
    body.longitude = longitude
    body.M = np.mod(longitude, 2 * np.pi)
    body.angles[KEY] = np.mod(sigma, 2 * np.pi)
    body.angles_unwrapped[KEY] = sigma
    zeros = np.zeros(n)
    sim.integration_engine.planets_data = {
        'Jupiter': {
            'times': t,
            'a': np.full(n, A_JUPITER),
            'e': np.full(n, 0.048),
            'inc': zeros,
            'Omega': zeros,
            'omega': zeros,
            'M': np.mod(jupiter_longitude, 2 * np.pi),
            'l': jupiter_longitude,
            'varpi': zeros,
        }
    }
    return body


def analysed(sim):
    """What Simulation.run() does after integrating: angles, verdicts, then periodograms."""
    sim.prepare_angles()
    sim.identify_librations()
    sim.build_periodograms()
    return sim


def add_derived_row(body):
    """A derived subresonance row of KEY, as the analysis would add it (depth 1). With varpi = varpi_J = 0 the member
    angle equals sigma0, so its analysis is the parent's under the member's key."""
    parent = body.mmr_analyses[KEY]
    resonance = body.add_derived_resonance(DERIVED_KEY, KEY, angle_wrapped=body.angles[KEY], angle_unwrapped=body.angles_unwrapped[KEY])
    derived = dataclasses.replace(parent, resonance=ResonanceSpec.from_key(DERIVED_KEY), parent=KEY, derived=[], derived_series=None)
    body.mmr_analyses[DERIVED_KEY] = derived
    body.librations[DERIVED_KEY] = derived.verdict
    body.statuses[DERIVED_KEY] = str(derived.verdict.status)
    return resonance


def read(tmp_path, table):
    return pd.read_csv(tmp_path / f'{table}.csv', dtype=str, keep_default_na=False)


# --------------------------------------------------------------------------- save/plot modes on strings (§11.5)

ALL_STATUSES = [str(s) for s in Status]
ROWS = [(status, origin) for status in ALL_STATUSES for origin in ('', 'kinematic')]
# Written from the table of SPEC §11.5, not from the code
EXPECTED = {
    'all': set(ROWS),
    'resonant': {(s, o) for s, o in ROWS if s in ('resonant', 'transient')},
    'candidates': {(s, o) for s, o in ROWS if s in ('resonant', 'transient', 'near_separatrix', 'uncertain')},
    'extended': {(s, o) for s, o in ROWS if s in ('resonant', 'transient', 'near_separatrix', 'uncertain')}
    | {('non_resonant', 'kinematic')},
    'nonzero': set(ROWS) - {('non_resonant', '')},
    'negative': {(s, o) for s, o in ROWS if s in ('near_separatrix', 'uncertain', 'invalid')},
    'none': set(),
}


@pytest.mark.parametrize('mode', sorted(EXPECTED))
def test_modes_select_the_rows_of_the_spec_table(mode):
    selected = {(s, o) for s, o in ROWS if DataManager._process_status(s, mode, o)}
    assert selected == EXPECTED[mode]
    assert not DataManager._process_status('resonant', None)


def test_an_integer_status_is_refused():
    with pytest.raises(TypeError, match='strings'):
        DataManager._process_status(2, 'all')
    with pytest.raises(TypeError, match='strings'):
        DataManager.status_folder(-9)


def test_should_save_reads_the_origin_of_the_verdict(tmp_path):
    """extended and nonzero keep a non_resonant row only when its verdict names a libration origin."""
    sim = synthetic(tmp_path, save='extended')
    body = sim.bodies[0]
    resonance = body.resonances()[0]
    verdict = Verdict(ResonanceType.MMR, Status.NON_RESONANT, AngleStatus.LIBRATION, libration_origin=LibrationOrigin.KINEMATIC)
    body.librations[KEY], body.statuses[KEY] = verdict, 'non_resonant'
    assert sim.data_manager.should_save_body(body, resonance)
    body.librations[KEY] = dataclasses.replace(verdict, angle_status=AngleStatus.CIRCULATION, libration_origin=LibrationOrigin.NONE)
    assert not sim.data_manager.should_save_body(body, resonance)
    sim.config.save = 'nonzero'
    assert not sim.data_manager.should_save_body(body, resonance)
    sim.config.plot = 'resonant'
    body.statuses[KEY] = 'transient'
    assert sim.data_manager.should_plot_body(body, resonance)


@pytest.mark.parametrize('status', ALL_STATUSES)
def test_status_folder_is_the_status_string(tmp_path, status):
    sim = synthetic(tmp_path, plot_subfolder_strategy='status')
    body = sim.bodies[0]
    body.statuses[KEY] = status
    assert sim.data_manager.figure_folder(body, body.resonances()[0], combined=True) == f'{tmp_path}/combined/{status}'
    assert sim.data_manager.figure_folder(body, body.resonances()[0]) == f'{tmp_path}/figures/{status}'


# --------------------------------------------------------------------------- one run, every table


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """A librating body (with a derived subresonance row) and a circulating one, saved with their figures."""
    tmp_path = tmp_path_factory.mktemp('w3_output_run')
    sim = synthetic(tmp_path, plot='all', plots=['combined', 'portrait'], plot_subfolder_strategy='status')
    add_synthetic_body(sim, 4000, True, 'circulator')
    analysed(sim)
    librator, circulator = sim.bodies
    add_derived_row(librator)
    sim.data_manager.save_data(sim.bodies, sim.times, sim)
    return sim, tmp_path


def test_the_verdicts_of_the_synthetic_bodies(run):
    """The physical verdicts the construction implies: a librating angle with a responding a, and a circulation."""
    sim, _ = run
    librator, circulator = sim.bodies
    assert librator.statuses[KEY] == 'resonant'
    assert circulator.statuses[KEY] == 'non_resonant'
    assert str(circulator.librations[KEY].libration_origin) == ''


def test_summary_columns_and_string_statuses(run):
    sim, tmp_path = run
    summary = read(tmp_path, 'summary')
    assert list(summary.columns[:5]) == ['name', 'resonance', 'resonance_type', 'added_by', 'parent']
    assert 'type' not in summary.columns
    row = summary[(summary['name'] == 'synthetic') & (summary['resonance'] == KEY)].iloc[0]
    assert (row['status'], row['angle_status'], row['resonance_type'], row['status_code']) == ('resonant', 'libration', 'mmr', '2')
    assert row['libration_origin'] == 'dynamic' and row['added_by'] == '' and row['parent'] == ''
    for column in ('owner', 'origin_detail', 'flags', 'action', 'confidence', 'reason', 'decision_path', 'multiplet_status'):
        assert column in summary.columns, column
    for check in MMR_CHECKS:
        assert row[check] in {str(v) for v in CheckValue if v != CheckValue.NONE}, check
    assert row['check_zlk_free_omega'] == '' and row['check_secular_e_response'] == ''
    assert row['settings_hash'] == settings_hash(sim.config.analysis_settings)
    assert row['package_commit'] == data_manager_module.package_commit() != ''
    assert float(row['a0']) == pytest.approx(A_BODY) and float(row['e0']) == pytest.approx(0.1)
    assert summary.columns[-1] == 'comments'
    circulator = summary[summary['name'] == 'circulator'].iloc[0]
    assert (circulator['status'], circulator['status_code'], circulator['libration_origin']) == ('non_resonant', '0', '')


def test_the_derived_row_has_its_own_rows(run):
    sim, tmp_path = run
    summary = read(tmp_path, 'summary')
    derived = summary[summary['resonance'] == DERIVED_KEY]
    assert len(derived) == 1
    assert (derived.iloc[0]['added_by'], derived.iloc[0]['parent'], derived.iloc[0]['name']) == ('subresonance_check', KEY, 'synthetic')
    numbers = read(tmp_path, 'mmr_numbers')
    assert sorted(numbers.loc[numbers['name'] == 'synthetic', 'resonance']) == sorted([KEY, DERIVED_KEY])


def test_mmr_numbers_one_row_per_body_and_mmr(run):
    sim, tmp_path = run
    numbers = read(tmp_path, 'mmr_numbers')
    summary = read(tmp_path, 'summary')
    assert len(numbers) == 3 and not numbers.duplicated(['name', 'resonance']).any()
    assert len(numbers.columns) > 100
    joined = summary.merge(numbers, on=['name', 'resonance'], suffixes=('', '_numbers'))
    assert len(joined) == 3
    np.testing.assert_allclose(joined['flt_cutoff_yr'].astype(float), joined['flt_cutoff_yr_numbers'].astype(float))


def test_episodes_and_candidates(run):
    sim, tmp_path = run
    episodes = read(tmp_path, 'episodes')
    librator = sim.bodies[0]
    expected = len(librator.mmr_analyses[KEY].episode_rows())
    assert expected >= 1
    mine = episodes[(episodes['name'] == 'synthetic') & (episodes['resonance'] == KEY)]
    assert len(mine) == expected
    assert set(mine['accepted']) <= {'True', 'False'}
    # The circulating angle has no turning points, hence no episode rows
    assert episodes[episodes['name'] == 'circulator'].empty


def test_constants_one_table_per_run(run):
    sim, tmp_path = run
    constants = read(tmp_path, 'constants')
    jupiter = constants[constants['planet'] == 'Jupiter']
    assert len(jupiter) == 1
    assert float(jupiter.iloc[0]['n_deg_yr']) == pytest.approx(JUPITER_N_DEG_YR, rel=1e-3)
    # Saving again (another batch of the same run) adds no row
    before = len(constants)
    sim.data_manager.save_constants(sim.data_manager.get_mmr_tables(sim.bodies)['constants'])
    assert len(read(tmp_path, 'constants')) == before


def test_data_file_keeps_raw_and_unwrapped_angle_only(run):
    sim, tmp_path = run
    data = pd.read_csv(tmp_path / 'data' / 'data-synthetic.csv')
    assert {f'{KEY}_angle', f'{KEY}_angle_unwrapped', f'{DERIVED_KEY}_angle'} <= set(data.columns)
    assert not [c for c in data.columns if c.startswith(('2J-', KEY)) and 'filtered' in c]
    # the unwrapped column is the analysis' unwrap: sigma itself (no branch jump)
    t = data['times'].to_numpy()
    expected = LIBRATION_CENTRE + 0.6 * np.sin(2 * np.pi * t / LIBRATION_PERIOD)
    np.testing.assert_allclose(data[f'{KEY}_angle_unwrapped'], expected, atol=1e-9)


def test_figures_one_per_row_in_status_folders(run):
    sim, tmp_path = run
    figures = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*.png'))
    assert f'combined/resonant/synthetic-{KEY}-combined.png' in figures
    assert f'combined/resonant/synthetic-{DERIVED_KEY}-combined.png' in figures
    assert f'figures/resonant/synthetic-{KEY}-portrait.png' in figures
    assert f'combined/non_resonant/circulator-{KEY}-combined.png' in figures
    assert len([f for f in figures if f.endswith('-combined.png')]) == 3


def test_arrays_are_released_after_saving(run):
    sim, _ = run
    for analysis in sim.bodies[0].mmr_analyses.values():
        assert analysis.prep is None and analysis.geometry is not None
    # the scalars stay: the summary row can be written again, identical
    summary = read(run[1], 'summary')
    again = sim.data_manager.get_simulation_summary(sim.bodies)
    assert list(again['status']) == list(summary['status'])


def test_saving_again_replaces_the_rows(run):
    sim, tmp_path = run
    sizes = {table: len(read(tmp_path, table)) for table in ('summary', 'mmr_numbers', 'episodes')}
    sim.data_manager.save_simulation_summary(sim.bodies)
    assert {table: len(read(tmp_path, table)) for table in sizes} == sizes


# --------------------------------------------------------------------------- secular and ZLK rows


def test_a_zlk_row_carries_its_verdict_and_classifier_numbers(tmp_path):
    """-2 (the free-omega gate's kinematic libration) is non_resonant with a kinematic origin; the numbers of its
    analysis follow (no TTO columns: chaos_*, n_good_*, legacy_type), and the MMR and ZLK rows keep their own
    resonance_type."""
    sim = synthetic(tmp_path)
    analysed(sim)
    sim.body_manager.add_body({'a': 2.5, 'e': 0.2, 'inc': 0.5, 'Omega': 0.0, 'omega': 1.0, 'M': 0.0}, 'LK', name='zlk')
    body = sim.bodies[-1]
    lk = body.resonances()[0]
    key = lk.to_s()
    verdict = verdict_from_status_code(-2, 'lidov_kozai', 'high', checks={'check_zlk_free_omega': CheckValue.FAIL})
    body.librations[key], body.statuses[key] = verdict, str(verdict.status)
    body.classifier_numbers[key] = {'flt_cutoff_yr': 4000.0, 'geo_n_tp': 7, 'zlk_gate_outcome': 'kinematic'}
    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    zlk = summary[summary['name'] == 'zlk'].iloc[0]
    assert (zlk['resonance_type'], zlk['status'], zlk['libration_origin'], zlk['status_code']) == (
        'lidov_kozai',
        'non_resonant',
        'kinematic',
        -2,
    )
    assert zlk['check_zlk_free_omega'] == 'fail' and zlk['geo_n_tp'] == 7 and zlk['zlk_gate_outcome'] == 'kinematic'
    assert summary.columns[-1] == 'comments'
    assert summary.loc[summary['name'] == 'synthetic', 'resonance_type'].tolist() == ['mmr']
    for legacy in ('type', 'legacy_type', 'chaos_flag', 'chaos_comment', 'n_good_total'):
        assert legacy not in summary.columns


def test_a_verdict_read_back_from_summary_is_not_rewritten(tmp_path):
    """After a restore without recomputation a row has a verdict but not its numbers: it stays as summary.csv has it."""
    sim = synthetic(tmp_path)
    sim.body_manager.add_body({'a': 2.5, 'e': 0.2, 'inc': 0.5, 'Omega': 0.0, 'omega': 1.0, 'M': 0.0}, 'LK', name='zlk')
    body = sim.bodies[-1]
    key = body.resonances()[0].to_s()
    body.librations[key] = verdict_from_status_code(-9, 'lidov_kozai')
    body.statuses[key] = 'uncertain'
    assert not DataManager._in_memory(body, key)
    assert sim.data_manager._classified([body]) == ([], set())


# --------------------------------------------------------------------------- figures after a restore


def test_figure_differences_by_hand():
    reference = {'status': 'resonant', 'angle_status': 'libration', 'confidence': 'medium', 'flt_cutoff_yr': '40.004'}
    same = {'status': 'resonant', 'angle_status': 'libration', 'confidence': 'medium', 'flt_cutoff_yr': 40.004000001}
    assert DataManager.figure_differences(reference, same) == []
    moved = dict(same, status='transient', flt_cutoff_yr=41.0)
    assert DataManager.figure_differences(reference, moved) == ['status resonant -> transient', 'flt_cutoff_yr 40.004 -> 41.0']
    assert DataManager.figure_differences({**reference, 'flt_cutoff_yr': ''}, {**same, 'flt_cutoff_yr': None}) == []


def test_restored_run_redraws_and_checks_against_summary(tmp_path, monkeypatch):
    sim = synthetic(tmp_path)
    analysed(sim)
    sim.data_manager.save_data(sim.bodies, sim.times, sim)
    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=False)
    body = restored.bodies[0]
    resonance = body.resonances()[0]
    assert body.statuses[KEY] == 'resonant' and not body.mmr_analyses

    warnings = []
    monkeypatch.setattr(data_manager_module.logger, 'warning', lambda message: warnings.append(message))
    analysis, series, stamp = restored.data_manager._figure_analysis(body, resonance, restored)
    assert stamp is None and str(analysis.verdict.status) == 'resonant', warnings
    restored.config.plot, restored.config.plots = 'all', ['combined']
    restored.data_manager.plot_body(body, restored)
    assert (tmp_path / 'combined' / f'synthetic-{KEY}-combined.png').exists()

    # A summary that disagrees with the re-run analysis: the figure is stamped and a warning is logged
    summary = pd.read_csv(tmp_path / 'summary.csv', dtype=str, keep_default_na=False)
    summary.loc[summary['resonance'] == KEY, 'status'] = 'transient'
    summary.to_csv(tmp_path / 'summary.csv', index=False)
    _, _, stamp = restored.data_manager._figure_analysis(body, resonance, restored)
    assert stamp == 'recomputed != summary: status transient -> resonant'
    assert any('differs from summary.csv' in w for w in warnings)


def test_mmr_panels_keep_their_kinds_without_the_old_combined():
    from resonances.plotting import MMRPlotter

    assert 'combined' not in MMRPlotter.KINDS
    assert set(MMRPlotter.KINDS) == {'recurrence', 'fair', 'portrait', 'cycles'}
