"""Independent functional tests of the W3 package wiring (SPEC §2.5, §2.6, §11; delta items 3-5).

Written from API.md, W3-PACKAGE-CONTRACT.md and the public docstrings only. No integration: the bodies get
synthetic series with a known truth.

The MMR body is in 2J-1 (sigma = 2 lambda_J - lambda - varpi, coefficients [2, -1, 0, -1]):

- Jupiter's mean motion n_J = 30.349646 deg/yr (const.PLANET_N_DEG_PER_GAUSS_YR), so the exact resonance
  has n* = 2 n_J = 1.059398 rad/yr and a* = (2 pi / n*)^(2/3) = 3.27652 au.
- The identity sigma_dot = c (a - a*) with c = -1.5 k n*/a*, k = -1 (the body's lambda coefficient):
  c = 1.5 * 1.059398 / 3.27652 = 0.48499 rad/yr/au.
- "libration" body: sigma - pi solves the pendulum sigma'' = -w0^2 sin(sigma - pi), w0 = 2 pi / 500 yr,
  amplitude A = 60 deg. Closed form period T = 4 K(m) / w0 with m = sin^2(A/2) = 0.25, K(0.25) = 1.685750,
  T = 4 * 1.685750 * 500 / (2 pi) = 536.6 yr. a = a* + sigma_dot / c, lambda = 2 lambda_J - varpi - sigma,
  which makes n(a) = lambda_dot to first order in (a - a*).
- "circulation" body: sigma = 0.4 + 2 pi t / 300 yr, a = a* + nu / c (constant): a circulating angle with no
  bounded oscillation (status non_resonant, libration_origin '').

Filter cutoff (API §2.3): P_c = max(1.5 max_j P_syn,j, 20 dt), P_syn = 2 pi / |n - n_J| = 2 pi / n_J
= 11.862 yr (since n = 2 n_J). With dt = 1 yr: max(17.79, 20) = 20 yr exactly; with dt = 0.5 yr:
max(17.79, 10) = 17.79 yr.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ellipk

import astdys.util
import resonances
from resonances.resonance.analysis.verdict import (
    ALL_CHECKS,
    MMR_CHECKS,
    SECULAR_CHECKS,
    ZLK_CHECKS,
    Status,
    Verdict,
    verdict_from_status_code,
)
from resonances.simulation.config import SavePlotMode, status_selected
from resonances.simulation.data_manager import DataManager
from resonances.simulation.serializer import SimulationSerializer

MMR_KEY = '2J-1+0-1'
SEC_KEY = 'g-g6'
LK_KEY = 'LK'

N_J = np.radians(30.349646)  # rad/yr
N_STAR = 2 * N_J
A_STAR = (2 * np.pi / N_STAR) ** (2.0 / 3.0)
C_ID = 1.5 * N_STAR / A_STAR  # rad / yr / au
W0 = 2 * np.pi / 500.0
AMP = np.radians(60.0)
# T = 4 K(sin^2(A/2)) / w0 (scipy's ellipk takes m = k^2); hand value 536.6 yr
PENDULUM_PERIOD = 4 * ellipk(np.sin(AMP / 2) ** 2) / W0
CIRC_RATE = 2 * np.pi / 300.0

ELEM = {"a": 3.27, "e": 0.1, "inc": 0.05, "Omega": 1.0, "omega": 2.0, "M": 0.5, "epoch": 60000.0}
STATUS_VALUES = {s.value for s in Status}


def _pendulum(ty):
    sol = solve_ivp(lambda t, y: [y[1], -(W0**2) * np.sin(y[0])], (ty[0], ty[-1]), [AMP, 0.0], t_eval=ty, rtol=1e-10, atol=1e-12)
    return np.pi + sol.y[0], sol.y[1]


def _fill_body(body, kind, ty):
    """Synthetic elements and raw angles of one body (see the module docstring)."""
    n = len(ty)
    if kind == 'libration':
        sig, sigd = _pendulum(ty)
    else:
        sig, sigd = 0.4 + CIRC_RATE * ty, np.full(n, CIRC_RATE)
    lam_j = N_J * ty + 0.3
    varpi = np.full(n, 1.2)
    lam = 2 * lam_j - varpi - sig
    body.times = ty * 2 * np.pi
    body.axis = A_STAR + sigd / C_ID
    body.ecc = np.full(n, 0.1)
    body.inc = np.full(n, 0.05)
    body.Omega = np.full(n, 1.0)
    body.omega = varpi - 1.0
    body.varpi = varpi
    body.longitude = lam
    body.M = lam - varpi
    keys = [r.to_s() for r in body.resonances()]
    if MMR_KEY in keys:
        body.angles_unwrapped[MMR_KEY] = sig.copy()
    if SEC_KEY in keys:
        body.angles_unwrapped[SEC_KEY] = varpi.copy()
    if LK_KEY in keys:
        body.angles_unwrapped[LK_KEY] = body.omega.copy()
    return sig


def make_sim(path, bodies=(('lib', 'libration'),), save='all', dt=1.0, years=10000.0, types=('mmr', 'sec', 'lk')):
    sim = resonances.Simulation(
        date=astdys.util.convert_mjd_to_datetime(60000),
        save_path=str(path),
        plot_path=str(path),
        save=save,
        plot='none',
        save_summary=True,
    )
    sim.config.save_planets = 'none'
    truth = {}
    for name, _kind in bodies:
        res = []
        if 'mmr' in types:
            res.append(resonances.create_mmr('2J-1'))
        if 'sec' in types:
            res.append(resonances.create_secular_resonance(SEC_KEY))
        if 'lk' in types:
            res.append(resonances.LidovKozaiResonance())
        sim.add_body(ELEM, res, name=name)
    ty = np.arange(0.0, years + dt / 2, dt)
    for body, (name, kind) in zip(sim.bodies, bodies):
        truth[name] = _fill_body(body, kind, ty)
    sim.times = sim.bodies[0].times
    return sim, truth


def run_pipeline(sim, save=True):
    sim.prepare_angles()
    sim.identify_librations()
    if save:
        sim.data_manager.save_data(sim.bodies, sim.times, sim)
    return sim


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _angle_diff(x, y):
    return np.angle(np.exp(1j * (np.asarray(x) - np.asarray(y))))


# ---------------------------------------------------------------- identify_librations


def test_identify_librations_writes_verdict_and_status_together(tmp_path):
    sim, _ = make_sim(tmp_path)
    run_pipeline(sim, save=False)
    body = sim.bodies[0]
    assert set(body.librations) == {MMR_KEY, SEC_KEY, LK_KEY}
    for key, verdict in body.librations.items():
        assert isinstance(verdict, Verdict)
        assert isinstance(body.statuses[key], str) and not isinstance(body.statuses[key], Status)
        assert body.statuses[key] == str(verdict.status)
        assert body.statuses[key] in STATUS_VALUES
    assert str(body.librations[MMR_KEY].resonance_type) == 'mmr'
    assert str(body.librations[SEC_KEY].resonance_type) == 'secular'
    assert str(body.librations[LK_KEY].resonance_type) == 'lidov_kozai'
    # Secular/ZLK classifier numbers carry no status of their own (contract §3)
    for key in (SEC_KEY, LK_KEY):
        assert not {'status', 'type', 'subtype', 'confidence'} & set(body.classifier_numbers[key])
    # no MMR analysis for secular/ZLK rows and no classifier numbers for the MMR
    assert set(body.mmr_analyses) == {MMR_KEY}
    assert MMR_KEY not in body.classifier_numbers


def test_pendulum_libration_is_resonant_with_closed_form_period(tmp_path):
    sim, _ = make_sim(tmp_path, types=('mmr',))
    run_pipeline(sim, save=False)
    body = sim.bodies[0]
    res = body.mmrs[0]
    verdict = body.librations[MMR_KEY]
    assert body.statuses[MMR_KEY] == 'resonant'
    assert str(verdict.angle_status) == 'libration'
    assert str(verdict.libration_origin) == 'dynamic'
    assert verdict.status_code == 2  # API §2.4: MMR resonant -> 2
    assert body.in_resonance(res) and body.in_pure_resonance(res)
    assert body.status(res) == 'resonant'
    assert body.verdict(res) is verdict
    assert body.mmr_analyses[MMR_KEY].verdict is verdict
    row = body.mmr_analyses[MMR_KEY].summary_row()
    assert row['status'] == 'resonant'
    assert row['name'] == 'lib'
    # T = 4 K(0.25) / w0 = 536.6 yr; the cycle period is measured on a 1-yr grid
    assert float(row['geo_period_cycle_yr']) == pytest.approx(PENDULUM_PERIOD, abs=2.0)
    # P_c = max(1.5 * 11.86, 150 / 2) = 75 yr exactly (the shortest libration's bound; the step does not enter)
    assert float(row['flt_cutoff_yr']) == pytest.approx(75.0, abs=1e-9)
    assert float(row['smp_step_yr']) == pytest.approx(1.0)
    # mean a = a* + mean(sigma_dot)/c, |mean(sigma_dot)| <= 2A/T_total -> within 5e-4 au of a*
    assert float(row['ctx_a_mean_au']) == pytest.approx(np.mean(body.axis), abs=1e-3)
    assert float(row['ctx_a_mean_au']) == pytest.approx(A_STAR, abs=1e-3)


def test_filter_cutoff_is_the_same_at_a_half_year_step(tmp_path):
    # dt = 0.5 yr: P_c = max(1.5 * 2 pi / n_J = 17.79 yr, 150 / 2 = 75 yr) = 75 yr, as at 1 yr
    sim, _ = make_sim(tmp_path, types=('mmr',), dt=0.5, years=6000.0)
    run_pipeline(sim, save=False)
    row = sim.bodies[0].mmr_analyses[MMR_KEY].numbers_row()
    assert 1.5 * 2 * np.pi / N_J == pytest.approx(17.79, abs=0.01)
    assert float(row['flt_cutoff_yr']) == pytest.approx(75.0) and row['flt_cutoff_binding'] == 'floor'


def test_circulation_is_non_resonant_without_origin(tmp_path):
    sim, truth = make_sim(tmp_path, bodies=(('circ', 'circulation'),), types=('mmr',))
    run_pipeline(sim, save=False)
    body = sim.bodies[0]
    verdict = body.librations[MMR_KEY]
    assert body.statuses[MMR_KEY] == 'non_resonant'
    assert str(verdict.libration_origin) == ''
    assert verdict.status_code == 0
    assert not body.in_resonance(body.mmrs[0])
    assert not body.in_pure_resonance(body.mmrs[0])
    # the predictive unwrap of a slow (1.2 deg/step) circulation is the raw angle up to 2 pi m
    shift = body.angles_unwrapped[MMR_KEY] - truth['circ']
    assert np.allclose(shift, shift[0], atol=1e-9)
    assert abs(shift[0] / (2 * np.pi) - round(shift[0] / (2 * np.pi))) < 1e-9


def test_body_angles_after_analysis(tmp_path):
    sim, truth = make_sim(tmp_path, types=('mmr',))
    run_pipeline(sim, save=False)
    body = sim.bodies[0]
    sig = truth['lib']
    # contract §2: angles[key] is the raw sigma mod 2 pi (exact)
    assert np.all((body.angles[MMR_KEY] >= 0) & (body.angles[MMR_KEY] < 2 * np.pi))
    assert np.max(np.abs(_angle_diff(body.angles[MMR_KEY], sig))) < 1e-12
    # the libration stays in [pi - 60 deg, pi + 60 deg]: no wrap, so the unwrap is sigma itself
    assert np.max(np.abs(body.angles_unwrapped[MMR_KEY] - sig)) < 1e-9
    # sigma_f exists (the analysis went past G0) and the low-pass keeps a 537-yr libration (cutoff 20 yr)
    flt = body.angles_filtered_unwrapped[MMR_KEY]
    assert len(flt) == len(sig)
    interior = slice(200, -200)
    assert np.max(np.abs(flt[interior] - sig[interior])) < np.radians(2.0)


def test_identify_librations_only_limits_the_pairs(tmp_path):
    sim, _ = make_sim(tmp_path)
    sim.prepare_angles()
    sim.identify_librations(only={'lib': [LK_KEY]})
    body = sim.bodies[0]
    assert set(body.librations) == {LK_KEY}
    assert set(body.statuses) == {LK_KEY}
    assert body.verdict(body.mmrs[0]) is None
    with pytest.raises(Exception) as excinfo:
        body.status(body.mmrs[0])
    assert excinfo.type is not AssertionError


def test_stale_derived_row_is_dropped_by_parent_analysis(tmp_path):
    # no planets saved -> no subresonance scan can accept a member; a derived row left from a previous
    # analysis must go (docstring of _analyse_mmr) and never be classified on its own
    sim, truth = make_sim(tmp_path, types=('mmr',))
    body = sim.bodies[0]
    body.add_derived_resonance('2J-1-1+0', MMR_KEY, np.mod(truth['lib'], 2 * np.pi), truth['lib'])
    assert body.resonance_origin('2J-1-1+0') == ('subresonance_check', MMR_KEY)
    assert body.resonance_origin(MMR_KEY) == ('', '')
    run_pipeline(sim, save=False)
    assert '2J-1-1+0' not in body.derived_resonances
    assert '2J-1-1+0' not in body.librations
    assert '2J-1-1+0' not in body.statuses


# ---------------------------------------------------------------- DataManager CSV outputs

VERDICT_BLOCK = [
    'status',
    'angle_status',
    'angle_subtype',
    'confidence',
    'confidence_raw',
    'reason',
    'flags',
    'action',
    'extend_to_years',
    'resample_step_yr',
    'libration_origin',
    'owner',
    'multiplet_status',
    'status_code',
]


@pytest.fixture
def saved_run(tmp_path):
    sim, truth = make_sim(tmp_path, bodies=(('lib', 'libration'), ('circ', 'circulation')))
    run_pipeline(sim)
    return tmp_path, sim, truth


def test_summary_csv_layout_and_string_statuses(saved_run):
    path, sim, _ = saved_run
    df = read_csv(path / 'summary.csv')
    cols = list(df.columns)
    assert cols[:5] == ['name', 'resonance', 'resonance_type', 'added_by', 'parent']
    pos = [cols.index(c) for c in VERDICT_BLOCK]
    assert pos == sorted(pos) and pos[0] == 5
    for c in ALL_CHECKS + ('decision_path', 'thin_margins', 'failed_support', 'caps', 'settings_hash', 'origin_detail'):
        assert c in cols, c
    for c in ('a0', 'e0', 'inc0', 'Omega0', 'omega0', 'M0'):
        assert c in cols, c
    assert cols[-1] == 'comments'
    assert len(df) == 6  # 2 bodies x 3 resonances, each once
    assert set(zip(df['name'], df['resonance'])) == {(n, k) for n in ('lib', 'circ') for k in (MMR_KEY, SEC_KEY, LK_KEY)}
    types = dict(zip(df['resonance'], df['resonance_type']))
    assert types == {MMR_KEY: 'mmr', SEC_KEY: 'secular', LK_KEY: 'lidov_kozai'}
    assert set(df['status']) <= STATUS_VALUES
    for _, row in df.iterrows():
        body = next(b for b in sim.bodies if b.name == row['name'])
        assert row['status'] == body.statuses[row['resonance']]
        assert int(row['status_code']) == body.librations[row['resonance']].status_code
        assert row['added_by'] == '' and row['parent'] == ''
    mmr = df[df['resonance_type'] == 'mmr'].set_index('name')
    assert mmr.loc['lib', 'status'] == 'resonant' and mmr.loc['lib', 'status_code'] == '2'
    assert mmr.loc['circ', 'status'] == 'non_resonant' and mmr.loc['circ', 'status_code'] == '0'
    assert float(mmr.loc['lib', 'a0']) == pytest.approx(ELEM['a'])
    # flags are ';'-separated, never ','
    assert not any(',' in f for f in df['flags'])


def test_summary_check_columns_by_resonance_type(saved_run):
    path, _, _ = saved_run
    df = read_csv(path / 'summary.csv')
    allowed = {'pass', 'fail', 'not_measured', 'not_applicable'}
    for _, row in df.iterrows():
        own = {'mmr': MMR_CHECKS, 'secular': SECULAR_CHECKS, 'lidov_kozai': ZLK_CHECKS}[row['resonance_type']]
        for c in ALL_CHECKS:
            if c in own:
                assert row[c] in allowed, (row['resonance'], c, row[c])
            else:
                assert row[c] == '', (row['resonance'], c, row[c])
        if row['resonance_type'] == 'secular':
            assert row['check_secular_e_response'] == 'not_measured'  # contract §3


def test_mmr_tables_and_segments(saved_run):
    path, sim, _ = saved_run
    numbers = read_csv(path / 'mmr_numbers.csv')
    assert list(numbers.columns[:2]) == ['name', 'resonance']
    assert sorted(zip(numbers['name'], numbers['resonance'])) == [('circ', MMR_KEY), ('lib', MMR_KEY)]
    lib = numbers.set_index('name').loc['lib']
    assert float(lib['geo_period_cycle_yr']) == pytest.approx(PENDULUM_PERIOD, abs=2.0)
    assert float(lib['flt_cutoff_yr']) == pytest.approx(75.0)
    summary = read_csv(path / 'summary.csv').set_index(['name', 'resonance'])
    assert summary.loc[('lib', MMR_KEY), 'geo_period_cycle_yr'] == lib['geo_period_cycle_yr']
    # segments.csv has no MMR rows (contract §3)
    seg_path = path / 'segments.csv'
    if seg_path.exists() and seg_path.stat().st_size > 0:
        seg = read_csv(seg_path)
        if 'resonance' in seg.columns:
            assert MMR_KEY not in set(seg['resonance'])


def test_data_file_has_raw_and_unwrapped_mmr_angle_only(saved_run):
    path, _, truth = saved_run
    df = pd.read_csv(path / 'data' / 'data-lib.csv', float_precision='round_trip')
    mmr_cols = [c for c in df.columns if c.startswith(MMR_KEY)]
    assert sorted(mmr_cols) == sorted([f'{MMR_KEY}_angle', f'{MMR_KEY}_angle_unwrapped'])
    assert np.max(np.abs(_angle_diff(df[f'{MMR_KEY}_angle'].to_numpy(), truth['lib']))) < 1e-9
    assert np.max(np.abs(df[f'{MMR_KEY}_angle_unwrapped'].to_numpy() - truth['lib'])) < 1e-9


def test_simulation_json_records_planets_saved(saved_run):
    path, sim, _ = saved_run
    with open(path / 'simulation.json') as f:
        data = json.load(f)
    assert data['simulation']['planets_saved'] == []
    assert sim.planets_saved() == []


# ---------------------------------------------------------------- SimulationSerializer restore


def test_restore_default_reads_verdicts_from_summary(saved_run):
    path, sim, _ = saved_run
    restored = SimulationSerializer.restore(str(path / 'simulation.json'))
    orig = {b.name: b for b in sim.bodies}
    assert {b.name for b in restored.bodies} == {'lib', 'circ'}
    for body in restored.bodies:
        ref = orig[body.name]
        assert body.statuses == ref.statuses
        assert all(isinstance(s, str) for s in body.statuses.values())
        assert body.mmr_analyses == {}
        assert body.classifier_numbers == {}
        for key, v in body.librations.items():
            r = ref.librations[key]
            assert isinstance(v, Verdict)
            assert str(v.status) == body.statuses[key]
            for attr in ('resonance_type', 'status', 'angle_status', 'angle_subtype', 'confidence', 'reason', 'libration_origin', 'owner'):
                assert str(getattr(v, attr)) == str(getattr(r, attr)), (body.name, key, attr)
            assert list(v.flags) == list(r.flags)
            assert v.status_code == r.status_code
            assert {k: str(c) for k, c in v.checks.items() if str(c)} == {k: str(c) for k, c in r.checks.items() if str(c)}


def test_restore_recompute_gives_the_same_rows(saved_run):
    path, sim, truth = saved_run
    before = read_csv(path / 'summary.csv').set_index(['name', 'resonance'])
    restored = SimulationSerializer.restore(str(path / 'simulation.json'), recompute_librations=True)
    for body in restored.bodies:
        for key in (MMR_KEY, SEC_KEY, LK_KEY):
            assert body.statuses[key] == before.loc[(body.name, key), 'status'], (body.name, key)
            assert body.statuses[key] == str(body.librations[key].status)
        row = body.mmr_analyses[MMR_KEY].summary_row()
        ref = before.loc[(body.name, MMR_KEY)]
        for c in ('angle_status', 'angle_subtype', 'confidence', 'reason', 'libration_origin'):
            assert str(row[c] or '') == ref[c], (body.name, c)
        for c in ('flt_cutoff_yr', 'geo_period_cycle_yr', 'ctx_a_mean_au'):
            if ref[c] == '':  # not measured (a circulation has no cycle): empty, never 0
                assert row[c] is None or np.isnan(float(row[c])), (body.name, c)
            else:
                assert float(row[c]) == pytest.approx(float(ref[c]), rel=1e-12, abs=1e-12), (body.name, c)
        assert np.max(np.abs(body.angles_unwrapped[MMR_KEY] - truth[body.name])) < 1e-9


def _write_old_summary(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_restore_statuses_of_old_integer_file(tmp_path):
    sim, _ = make_sim(tmp_path)
    csv = tmp_path / 'old-summary.csv'
    _write_old_summary(
        csv,
        [
            {'name': 'lib', 'resonance': SEC_KEY, 'status': 2},
            {'name': 'lib', 'resonance': LK_KEY, 'status': -2},
            {'name': 'lib', 'resonance': MMR_KEY, 'status': 2},
        ],
    )
    n = SimulationSerializer.restore_statuses(sim.bodies, str(csv))
    body = sim.bodies[0]
    assert n == 2  # the integer MMR row has no translation and is skipped
    assert MMR_KEY not in body.statuses
    assert body.statuses[SEC_KEY] == 'resonant'  # old LIBRATION
    assert body.librations[SEC_KEY].status_code == 2
    assert body.librations[LK_KEY].status_code == -2
    assert body.statuses[LK_KEY] == str(body.librations[LK_KEY].status)


@pytest.mark.parametrize(
    'code,status',
    [
        (2, 'resonant'),
        (1, 'transient'),
        (0, 'non_resonant'),
        (-3, 'near_separatrix'),
        (-5, 'near_separatrix'),
        (-4, 'uncertain'),
        (-6, 'uncertain'),
        (-9, 'uncertain'),
        (-99, 'invalid'),
    ],
)
def test_old_secular_codes_round_trip_through_restore(tmp_path, code, status):
    sim, _ = make_sim(tmp_path, types=('sec',))
    csv = tmp_path / 'old.csv'
    _write_old_summary(csv, [{'name': 'lib', 'resonance': SEC_KEY, 'status': code}])
    assert SimulationSerializer.restore_statuses(sim.bodies, str(csv)) == 1
    v = sim.bodies[0].librations[SEC_KEY]
    assert sim.bodies[0].statuses[SEC_KEY] == status
    assert v.status_code == code


def test_minus_two_and_minus_nine_stay_distinct():
    # CLAUDE.md: -2 is a measurement (kinematic libration), -9 the absence of one
    a = verdict_from_status_code(-2, 'lidov_kozai')
    b = verdict_from_status_code(-9, 'lidov_kozai')
    assert (str(a.status), str(a.reason)) != (str(b.status), str(b.reason))
    assert a.status_code == -2 and b.status_code == -9
    with pytest.raises(Exception) as excinfo:
        verdict_from_status_code(-1, 'secular')
    assert excinfo.type is not AssertionError


def test_restore_statuses_re_adds_derived_rows(tmp_path):
    sim, _ = make_sim(tmp_path, types=('mmr',))
    v = Verdict.from_flat_dict({'resonance_type': 'mmr', 'status': 'resonant', 'angle_status': 'libration'})
    parent = {'name': 'lib', 'resonance': MMR_KEY, 'resonance_type': 'mmr', 'added_by': '', 'parent': ''}
    derived = {'name': 'lib', 'resonance': '2J-1-1+0', 'resonance_type': 'mmr', 'added_by': 'subresonance_check', 'parent': MMR_KEY}
    rows = [{**parent, **v.to_flat_dict()}, {**derived, **v.to_flat_dict()}]
    csv = tmp_path / 'summary.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    n = SimulationSerializer.restore_statuses(sim.bodies, str(csv))
    body = sim.bodies[0]
    assert n == 2
    assert body.derived_resonances == {'2J-1-1+0': MMR_KEY}
    assert body.resonance_origin('2J-1-1+0') == ('subresonance_check', MMR_KEY)
    assert [r.to_s() for r in body.resonances()] == [MMR_KEY, '2J-1-1+0']
    assert body.statuses == {MMR_KEY: 'resonant', '2J-1-1+0': 'resonant'}
    assert isinstance(body.librations['2J-1-1+0'], Verdict)


# ---------------------------------------------------------------- SavePlotMode on string statuses

# SPEC §11.5, written out by hand: (status, origin) -> set of modes that select it
_O = 'kinematic'
SELECTION = {
    ('resonant', ''): {'all', 'resonant', 'candidates', 'extended', 'nonzero'},
    ('resonant', 'dynamic'): {'all', 'resonant', 'candidates', 'extended', 'nonzero'},
    ('transient', 'dynamic'): {'all', 'resonant', 'candidates', 'extended', 'nonzero'},
    ('near_separatrix', ''): {'all', 'candidates', 'extended', 'nonzero', 'negative'},
    ('uncertain', ''): {'all', 'candidates', 'extended', 'nonzero', 'negative'},
    ('uncertain', 'unknown'): {'all', 'candidates', 'extended', 'nonzero', 'negative'},
    ('invalid', ''): {'all', 'nonzero', 'negative'},
    ('non_resonant', ''): {'all'},
    ('non_resonant', _O): {'all', 'extended', 'nonzero'},
    ('non_resonant', 'induced'): {'all', 'extended', 'nonzero'},
    ('non_resonant', 'subresonance'): {'all', 'extended', 'nonzero'},
}
MODES = ['none', 'all', 'resonant', 'candidates', 'extended', 'nonzero', 'negative']


@pytest.mark.parametrize('status,origin', list(SELECTION))
def test_status_selected_truth_table(status, origin):
    for mode in MODES:
        expected = mode in SELECTION[(status, origin)]
        assert status_selected(mode, status, origin) is expected, (mode, status, origin)
        assert status_selected(SavePlotMode(mode), Status(status), origin) is expected, (mode, status, origin)
        assert DataManager._process_status(status, mode, origin) is expected, (mode, status, origin)


def test_status_selected_without_verdict_and_integer_statuses():
    assert status_selected('all', None, '') is True
    for mode in MODES[2:] + ['none']:
        assert status_selected(mode, None, '') is False
    with pytest.raises(TypeError):
        DataManager._process_status(2, 'resonant', '')
    assert DataManager.status_folder('near_separatrix') == 'near_separatrix'
    assert DataManager.status_folder(Status.RESONANT) == 'resonant'
    with pytest.raises(TypeError):
        DataManager.status_folder(2)


@pytest.mark.parametrize(
    'mode,expected',
    [
        ('all', {'lib', 'circ'}),
        ('resonant', {'lib'}),
        ('candidates', {'lib'}),
        ('extended', {'lib'}),
        ('none', set()),
    ],
)
def test_save_mode_selects_data_files_by_mmr_status(tmp_path, mode, expected):
    # only the MMR so that the secular/ZLK verdicts do not select the bodies; lib resonant, circ non_resonant ''
    sim, _ = make_sim(tmp_path, bodies=(('lib', 'libration'), ('circ', 'circulation')), save=mode, types=('mmr',), years=4000.0)
    run_pipeline(sim)
    written = {n for n in ('lib', 'circ') if (tmp_path / 'data' / f'data-{n}.csv').exists()}
    assert written == expected
    for body in sim.bodies:
        assert sim.data_manager.should_save_body(body, body.mmrs[0]) is (body.name in expected)
    summary = read_csv(tmp_path / 'summary.csv')
    assert set(summary['name']) == {'lib', 'circ'}  # the summary has every row whatever the mode


def test_plot_folder_is_the_status_string(tmp_path):
    sim, _ = make_sim(tmp_path, bodies=(('lib', 'libration'), ('circ', 'circulation')), types=('mmr',), years=4000.0)
    run_pipeline(sim, save=False)
    sim.config.plot_subfolder_strategy = 'status'
    lib, circ = sim.bodies
    p_lib = sim.data_manager.figure_folder(lib, lib.mmrs[0])
    p_circ = sim.data_manager.figure_folder(circ, circ.mmrs[0])
    assert os.path.basename(os.path.normpath(p_lib)) == 'resonant'
    assert os.path.basename(os.path.normpath(p_circ)) == 'non_resonant'
