"""Debug tests of package 5 (decide, confidence, analyse, io). The independent tests come from the
context-free test writers; these pin the rules the orchestration relies on."""

import gzip
import itertools
import json
import math

import numpy as np
import pytest

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.decide import AngleDecision
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.builders import (
    make_energy,
    make_geometry,
    make_kinematics,
    make_response,
    make_sampling,
    make_turning_points,
)
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    MarginClass,
    MarginValue,
    Reason,
    Status,
)

S = settings_from_config()
D, SUP = MarginClass.DECISIVE, MarginClass.SUPPORT


def _m(margin, cls=D):
    return MarginValue(Margin(margin), 1.0, cls)


# --- confidence -----------------------------------------------------------------------------------


def test_raw_level_counts_failed_support_margins_and_thin_ones():
    assert conf.raw_level({'L1': _m('full'), 'L3': _m('full', SUP)}, ('L4',))[0] == Confidence.CERTAIN
    level, thin, failed, missing = conf.raw_level({'L1': _m('thin'), 'L2': _m('thin'), 'L4': _m('full')}, ('L4',))
    assert (level, thin, failed, missing) == (Confidence.HIGH, ['L1', 'L2'], [], False)
    assert conf.raw_level({'L3': _m('fail', SUP), 'L4': _m('full')}, ('L4',))[0] == Confidence.MEDIUM
    assert conf.raw_level({'L3': _m('fail', SUP), 'L5': _m('fail', SUP)}, ())[0] == Confidence.LOW
    # a failed cap margin limits through its cap, it is not counted
    assert conf.raw_level({'O1': _m('fail', MarginClass.CAP)}, ())[0] == Confidence.CERTAIN
    # every physics margin not measured: thin, and the flag that caps the level at medium
    level, thin, _, missing = conf.raw_level({'L4': _m('not_measured'), 'R0': _m('not_measured')}, ('L4', 'R0', 'Z'))
    assert level == Confidence.HIGH and missing and thin == ['L4', 'R0']


def test_caps_and_calibration_cells(tmp_path):
    assert conf.apply_caps(Confidence.CERTAIN, ['three_tp_episode']) == Confidence.HIGH
    assert conf.apply_caps(Confidence.HIGH, ['three_tp_episode', 'onset_failed']) == Confidence.LOW
    assert conf.apply_caps(Confidence.NONE, ['onset_failed']) == Confidence.NONE
    path = tmp_path / 'cells.json'
    cells = [
        {'status': 'resonant', 'angle_status': 'libration', 'reason': 'dynamic_libration', 'n_examples': 120, 'n_errors': 0},
        {'status': 'non_resonant', 'angle_status': 'circulation', 'reason': 'circulation', 'n_examples': 150, 'n_errors': 1},
        {'status': 'transient', 'angle_status': 'episodic', 'reason': 'dynamic_episode', 'n_examples': 40, 'n_errors': 0},
    ]
    path.write_text(json.dumps({'version': 1, 'cells': cells}))
    loaded = conf.load_cells(path)
    assert set(loaded.records) == {(c['status'], c['angle_status'], c['reason']) for c in cells}
    assert loaded.sha == conf.file_sha(path)
    full = {'L1': _m('full')}
    certain = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', full, [], [], ('L4',), loaded)
    assert certain.confidence == Confidence.CERTAIN and certain.cell.n_examples == 120
    # 40 labels: the level is shown all the same (author 2026-10-16), the cell only reported
    few = conf.assess(Status.TRANSIENT, 'episodic', 'dynamic_episode', full, [], [], ('L4',), loaded)
    assert few.confidence == Confidence.CERTAIN and (few.cell.n_examples, few.cell.n_errors) == (40, 0)
    high = conf.assess(Status.TRANSIENT, 'episodic', 'dynamic_episode', {'L1': _m('thin')}, [], [], (), loaded)
    assert high.confidence == Confidence.HIGH
    by_u = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', full, [], ['U:x'], (), loaded)
    assert by_u.confidence_raw == Confidence.HIGH and by_u.thin_margins == ['U:x']
    none = conf.assess(Status.UNCERTAIN, 'drift', 'drift_mechanism_unknown', full, [], [], (), loaded)
    assert none.confidence == Confidence.NONE == none.confidence_raw


# --- the status tree reaches a leaf for every combination ------------------------------------------


def _response(a_response, a_amplitude, slow_libration):
    return make_response(
        a_response=a_response,
        a_amplitude=a_amplitude,
        slow_libration=slow_libration,
        b1_ratio=0.5,
        slow_adot_r2=0.1,
        force_r2_h1=0.8,
        force_r2_h2=0.8,
        centre_rate_rad_yr=1e-6,
        centre_rate_err_rad_yr=1e-5,
        exact_commensurability=True,
        force_law='single',
        multiplet=None,
        phi0_max_dev_rad=0.1,
        phi0_tol_rad=0.5,
    )


VALUES = (CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_MEASURED, CheckValue.NOT_APPLICABLE)


@pytest.mark.parametrize('status', [AngleStatus.LIBRATION, AngleStatus.EPISODIC, AngleStatus.NEAR_SEPARATRIX, AngleStatus.DRIFT])
def test_every_combination_of_checks_reaches_a_leaf(status):
    geometry = make_geometry(turning_points=make_turning_points([1, 2, 3]))
    subtype = AngleSubtype.PATTERN_B if status == AngleStatus.NEAR_SEPARATRIX else AngleSubtype.NONE
    angle = AngleDecision(status, subtype, 'G')
    leaves = set()
    for l4, b1, b5, kin, r0 in itertools.product(VALUES, repeat=5):
        energy = make_energy(with_slow=r0, ve_vf_agree_fraction=0.5)
        kinematics = make_kinematics(check=kin, corr_rate=0.95, detail='')
        result = decide.decide_status(
            angle, make_sampling(), geometry, energy, _response(l4, b1, b5), kinematics, None, None, None, S, Ledger.from_settings(S)
        )
        assert isinstance(result.status, Status) and result.reason
        leaves.add((result.status, result.reason))
    if status == AngleStatus.DRIFT:
        assert (Status.NON_RESONANT, Reason.KINEMATIC_DRIFT) in leaves and (Status.UNCERTAIN, Reason.DRIFT_MECHANISM_UNKNOWN) in leaves
    if status == AngleStatus.LIBRATION:
        assert (Status.RESONANT, Reason.DYNAMIC_LIBRATION) in leaves and (Status.UNCERTAIN, Reason.ORIGIN_UNKNOWN) in leaves
        assert (Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION) in leaves  # B1 fail
        assert (Status.UNCERTAIN, Reason.BASELINE_NEEDED) in leaves  # B1 fail + B5 fail at exact commensurability


def test_force_law_and_potential_combinations():
    angle = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2')

    def run(l4, r0):
        energy = make_energy(with_slow=r0, ve_vf_agree_fraction=0.5)
        kinematics = make_kinematics(check=CheckValue.FAIL, corr_rate=0.99, detail='')  # ignored for an episode
        response = _response(l4, CheckValue.PASS, CheckValue.PASS)
        return decide.decide_status(
            angle, make_sampling(), None, energy, response, kinematics, None, None, None, S, Ledger.from_settings(S)
        )

    assert run(CheckValue.FAIL, CheckValue.FAIL).rule == '5a'
    assert run(CheckValue.PASS, CheckValue.FAIL).flags == [str(Flag.FORCE_LAW_DISAGREES)]
    c = run(CheckValue.FAIL, CheckValue.PASS)
    assert (c.status, c.rule, c.origin) == (Status.RESONANT, '5c/6', LibrationOrigin.DYNAMIC) and 'force_law_disagrees' in c.caps
    assert run(CheckValue.NOT_MEASURED, CheckValue.NOT_MEASURED).status == Status.RESONANT
    assert run(CheckValue.NOT_MEASURED, CheckValue.FAIL).status == Status.UNCERTAIN


def test_round_up_and_baseline():
    assert decide.round_up(37000.0) == 40000.0 and decide.round_up(200000.0) == 200000.0 and math.isnan(decide.round_up(0.0))
    geometry = make_geometry(period_cycle_yr=20000.0)
    base = decide.baseline(geometry, 75.0, 20000.0, float('nan'), S)
    # (3 / 2) x 20 000 + 2 x 1 x 75 = 30 150 yr > 20 000 yr: extend to the round value >= 40 000
    assert base.required_yr == pytest.approx(30150.0) and base.check == CheckValue.FAIL and base.extend_to_years == 40000.0
    assert decide.baseline(geometry, 75.0, 100000.0, float('nan'), S).check == CheckValue.PASS


# --- io -------------------------------------------------------------------------------------------


def test_load_series_reads_legacy_columns_and_orders_time(tmp_path, monkeypatch):
    import pandas as pd

    folder = tmp_path / 'examples' / 'mmr2__x__2J-1+0-1'
    folder.mkdir(parents=True)
    t = -np.arange(0, 50.0, 0.5)  # a backward run
    n = len(t)
    angle = 0.01 * t + 0.3
    series = pd.DataFrame(
        {
            't_yr': t,
            'a': 3.27,
            'e': 0.1,
            'inc': 0.05,
            'Omega': 1.0,
            'omega': 2.0,
            'varpi': 3.0,
            'M': 0.1 * t,
            'lam': 0.2 * t,
            'angle_raw': angle,
        }
    )
    series.to_csv(folder / 'series.csv.gz', index=False)
    planets = pd.DataFrame(
        {'t_yr': t, **{f'Jupiter.{f}': np.full(n, v) for f, v in (('a', 5.2), ('e', 0.05), ('inc', 0.0), ('Omega', 0.0), ('varpi', 0.2))}}
    )
    planets['Jupiter.l'] = 0.53 * t
    planets.to_csv(folder / 'planets.csv.gz', index=False)
    meta = {
        'name': 'x',
        'resonance': '2J-1+0-1',
        'resonance_type': 'mmr',
        'regime': {'integrator': 'SABA', 'dt': 1.0, 'epoch_jd': 2461200.5},
    }
    (folder / 'meta.json').write_text(json.dumps(meta))
    monkeypatch.setenv(io.DATA_ENV, str(tmp_path))
    assert io.list_examples() == ['mmr2__x__2J-1+0-1'] and io.list_examples('secular') == []
    loaded = io.load_series('mmr2__x__2J-1+0-1')
    assert np.all(np.diff(loaded.t_yr) > 0) and loaded.t_yr[0] == pytest.approx(-49.5)
    assert loaded.mean_longitude == pytest.approx(0.2 * loaded.t_yr)
    assert loaded.planets['Jupiter'].mean_longitude == pytest.approx(0.53 * loaded.t_yr)
    assert loaded.angle_wrapped == pytest.approx(np.mod(0.01 * loaded.t_yr + 0.3, 2 * np.pi))
    assert loaded.regime.step_yr == 0.5 and loaded.regime.n_rows == n and loaded.resonance.key == '2J-1+0-1'
    assert gzip.open(folder / 'series.csv.gz').read(4)  # still a gzip file, untouched


# --- real records (benchmark data, git-ignored; skipped when absent) ---------------------------------

EXPECTED = {  # (status, angle_status or None, libration_origin)
    # author: resonant/medium; L4 and R0 not measured, the e^q scale passes but measures nothing: rule 5u/6 (W18).
    # W20 edge rule (variant a, an author-accepted side effect): the raw angle runs 1.35 turns past the last turning
    # point, the exit is observed: transient
    'mmr3__24820__3J-1S-1+0+0-1': (Status.TRANSIENT, AngleStatus.EPISODIC, LibrationOrigin.UNKNOWN),
    'mmr2__153__3J-2+0-1': (Status.RESONANT, AngleStatus.LIBRATION, LibrationOrigin.DYNAMIC),  # Hilda
    'mmr3__1999_UQ__7E+2M-9+0+0+0': (Status.NON_RESONANT, AngleStatus.CIRCULATION, LibrationOrigin.NONE),  # alias at 5 yr
    # author: non_resonant/drift. One 17-kyr swing in 20 kyr: one turning point, no cycle to judge, so 'extend'.
    # The records at 1 and 5 yr are the same orbit and must agree (with P_c = 75 yr the 1-yr record alone read
    # kinematic: 3 TPs at x1, 57 at x0.5; TEST-DISPUTES.md).
    'mmr3__1999_UQ__8M-9J-4+0+0+5__step1yr': (Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, LibrationOrigin.NONE),
    # with the run's planets (W9 link) the great-inequality ripple is removed at 5 yr: the TP prominence falls from
    # the ripple's to 4.2 deg and the 17-kyr swing shows three TPs, 38.9, -97.5, -69.6 deg (136 deg down, 28 back).
    # That cycle does not close (net 108 deg > mean swing 82 deg, W13), so no episode: the same verdict as at 1 yr
    # (TEST-DISPUTES.md, W9 and W13)
    'mmr3__1999_UQ__8M-9J-4+0+0+5': (Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, LibrationOrigin.NONE),
}


@pytest.mark.benchmark
@pytest.mark.parametrize('example', sorted(EXPECTED))
def test_real_records(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(io.load_series(example), S)
    status, angle_status, origin = EXPECTED[example]
    assert analysis.verdict.status == status and analysis.verdict.libration_origin == origin
    assert angle_status is None or analysis.verdict.angle_status == angle_status
    summary = analysis.summary_row()
    for entry in analysis.verdict.decision_path:  # every number the decision read is in the summary row
        assert not entry.column or entry.column in summary, entry.column
