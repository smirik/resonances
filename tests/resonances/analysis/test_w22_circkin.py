"""W22 R-CIRC-KIN (author 2026-10-24): circulations and kinematic verdicts.

1. R-C4: a failed C4 counts only when K comes from a force law accepted at least in part (best R^2 of the halves, the
   multiplet and the one fit over the whole window, the new column rsp_force_R2_whole, >= mmr_force_r2_fail).
2. R-L3K: L3 not_applicable for a kinematic verdict (kinematic_libration, kinematic_drift).
3. R-FK: margin F of a kinematic non_resonant verdict from the directly averaged strength (full when it fails).
4. R-KDRIFT: a kinematic libration whose centre moves more than mmr_kin_drift_shift_ratio x its amplitude (L6) is K2.
5. R-NET (G4b): the angle circulates when the slow centre advances >= libration_circulation_net_rev revolutions net,
   in G4 and in K2's angle word.

Expected values: the circulating pendulum's force law and C4 ratio from its own energy integral (quadrature, not the
package), the R^2 / L6 / net-rev values of the benchmark records as measured in SP/dis/impact/R-CIRC-KIN (37908: R^2
0.005 whole, 0.016 multiplet; 62243 L6 1.82; 74729 L6 1.57, pass 1.039 rev; 92336 L6 1.06; 142694 200 kyr net 2.75;
21028 net 1.81), the author's labels of 2026-10-24 for the records.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import integrate

from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr import response as r
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.decide import AngleDecision, StatusDecision
from resonances.resonance.analysis.mmr.models import MultipletFit, StepVerdict
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, LibrationOrigin, Reason, Status
from resonances.plotting import verdict_text
from tests.resonances.analysis.impl_response_pendulum_synthetic import make, pendulum_record
from tests.resonances.analysis.builders import make_centre, make_geometry, make_response, make_sampling

S = settings_from_config()
NAN = float('nan')
TWO_PI = 2 * math.pi
CIRCULATION = AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4')
LIBRATION = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', 0)
EPISODIC = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)


def test_new_settings_and_shared_ledger_names():
    assert S.mmr_kin_drift_shift_ratio == 1.0 and S.libration_circulation_net_rev == 2.0
    assert {'mmr_kin_drift_shift_ratio', 'libration_circulation_net_rev'} <= set(Ledger.from_settings(S).conventions)
    with pytest.raises(ValueError):
        settings_from_config(libration_circulation_net_rev=0.0)
    # the verdict text reads the decision's ledger entries by name
    assert (verdict_text.K2B_SHIFT, verdict_text.K2_NET, verdict_text.G4B_NET) == (decide.K2B_SHIFT, decide.K2_NET, decide.G4B_NET)


# --- the whole-record force fit (rsp_force_R2_whole) -----------------------------------------------------------------


def _circulating_pendulum():
    # h = 2: above the separatrix (h = 1), sigma circulates; no turning points, so no force windows
    return pendulum_record(h=2.0, span=6000.0)


def test_whole_record_fit_of_a_circulating_pendulum():
    rec = _circulating_pendulum()
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, S, spec=rec.spec)
    omega0 = 2 * np.pi / 300.0
    assert not resp.windows
    assert resp.force_r2_whole == pytest.approx(1.0, abs=1e-6)  # a_dot = (omega0^2 / c) sin sigma exactly
    assert resp.force_amplitude_au_yr == pytest.approx(omega0**2 / abs(rec.c), rel=1e-3)
    # C4 = <|sigma_dot|> / (2 omega0): the circulation period from the energy integral
    # sigma_dot^2 = 4 omega0^2 (h - sin^2(x / 2)), x = sigma - well: T = int_0^2pi dx / sigma_dot
    period = integrate.quad(lambda x: 1 / (2 * omega0 * math.sqrt(2.0 - math.sin(x / 2) ** 2)), 0, TWO_PI)[0]
    assert resp.circulation_strength_ratio == pytest.approx((TWO_PI / period) / (2 * omega0), rel=2e-3)
    assert r.response_columns(resp)['rsp_force_R2_whole'] == resp.force_r2_whole


def test_whole_record_fit_explains_nothing_when_a_dot_is_noise():
    rec = _circulating_pendulum()
    rec.prep.adot_f = 1e-6 * np.random.default_rng(5).standard_normal(len(rec.t))  # a does not answer sigma
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, S, spec=rec.spec)
    assert resp.force_r2_whole < 0.01  # 2 / N_samples on average for two fitted terms
    assert decide.force_law_rejected(resp, S, Ledger.from_settings(S))


def test_whole_record_fit_is_not_made_with_windows():
    rec = pendulum_record()  # a libration: the windowed law, no whole fit
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, S, spec=rec.spec)
    assert resp.windows and math.isnan(resp.force_r2_whole)


# --- part 1: C4 --------------------------------------------------------------------------------------------------------


def _response(ratio, h1=NAN, h2=NAN, multi=NAN, whole=NAN):
    multiplet = make(MultipletFit, r2=multi) if not math.isnan(multi) else None
    return make_response(circulation_strength_ratio=ratio, force_r2_h1=h1, force_r2_h2=h2, multiplet=multiplet, force_r2_whole=whole)


@pytest.mark.parametrize(
    'laws, rejected',
    [
        ({'multi': 0.016, 'whole': 0.005}, True),  # 37908 23J-8
        ({'h1': 0.054, 'h2': 0.02}, True),  # 1221: best 0.054
        ({'h1': 0.179}, True),  # 1992 SZ
        ({}, True),  # nothing fitted: K is no measured force
        ({'h1': 0.1, 'h2': 0.576}, False),  # 2000 YG4 keeps its failed C4
        ({'multi': 0.625}, False),  # 37087
        ({'whole': 0.4}, False),  # at the threshold the law is accepted in part
    ],
)
def test_force_law_rejected(laws, rejected):
    assert decide.force_law_rejected(_response(0.6, **laws), S, Ledger.from_settings(S)) == rejected


def _circulation_margins(resp):
    geometry = make_geometry(
        centre=make_centre(max_pass_rad=5.0 * TWO_PI),
        n_tp_cut_half=0,
        n_tp_cut_double=0,
        sigmadot_negative_fraction=0.0,
        rate_modulation=0.1,
    )
    context = SimpleNamespace(rate_from_mean_longitude_rad_yr=NAN, rate_expected_rad_yr=NAN, rate_measured_rad_yr=NAN)
    return decide._circulation_margins(CIRCULATION, context, None, geometry, resp, Ledger.from_settings(S), S)


def test_a_failed_c4_counts_only_under_an_accepted_force_law():
    assert _circulation_margins(_response(0.6, multi=0.016, whole=0.005))['C4'].margin == Margin.NOT_MEASURED
    assert _circulation_margins(_response(0.6, h1=0.576))['C4'].margin == Margin.FAIL
    # a passed C4 stays under a rejected law: K is then a ceiling of the force and C4 a lower bound
    assert _circulation_margins(_response(2.0, whole=0.005))['C4'].margin == Margin.FULL
    assert _circulation_margins(_response(1.2, whole=0.005))['C4'].margin == Margin.THIN


# --- parts 2 and 3: L3 and F of a kinematic verdict ---------------------------------------------------------------------


def _episode_margins(reason):
    episode = make(geo.Episode, episode_id=0, tp_times_yr=np.arange(6.0), half_amp_max_rad=0.5, centre_shift_over_amp=0.2)
    geometry = make_geometry(runs=[episode])
    resp = make_response(
        identity=CheckValue.FAIL, identity_corr=0.1, force_r2_h1=0.01, force_r2_h2=0.01, a_response=CheckValue.FAIL, a_response_thin=False
    )
    origin = LibrationOrigin.KINEMATIC if reason != Reason.DYNAMIC_LIBRATION else LibrationOrigin.DYNAMIC
    status = Status.NON_RESONANT if origin == LibrationOrigin.KINEMATIC else Status.RESONANT
    decision = StatusDecision(status, reason, origin, '', '', rule='3')
    context = SimpleNamespace(e_med=0.1)
    return decide._episode_margins(LIBRATION, decision, context, geometry, None, resp, [], S, Ledger.from_settings(S), None)


@pytest.mark.parametrize('reason', [Reason.KINEMATIC_LIBRATION, Reason.KINEMATIC_DRIFT])
def test_l3_does_not_apply_to_a_kinematic_verdict(reason):
    margins = _episode_margins(reason)
    assert margins['L3'].margin == Margin.NOT_APPLICABLE and margins['L3'].value == 0.1


def test_l3_still_fails_for_a_dynamic_verdict():
    assert _episode_margins(Reason.DYNAMIC_LIBRATION)['L3'].margin == Margin.FAIL


def _strength(check, method=st.METHOD_DIRECT, ratio=40.0):
    return SimpleNamespace(check=check, method=method, ratio=ratio)


@pytest.mark.parametrize(
    'check, level',
    [(CheckValue.FAIL, Margin.FULL), (CheckValue.PASS, Margin.NOT_MEASURED), (CheckValue.NOT_MEASURED, Margin.NOT_MEASURED)],
)
def test_f_of_a_kinematic_verdict(check, level):
    for method in decide.DIRECT_METHODS:
        margin = decide.kinematic_strength_margin(_strength(check, method))
        assert margin.margin == level and margin.value == 40.0


# --- part 4: K2b ------------------------------------------------------------------------------------------------------


def _geometry(shift, max_pass_rev, net_rev=0.0):
    episode = make(geo.Episode, episode_id=0, centre_shift_over_amp=shift)
    return make_geometry(runs=[episode], centre=make_centre(max_pass_rad=max_pass_rev * TWO_PI, net_rev=net_rev))


def _kinematic(angle, geometry, settings=S):
    resp = SimpleNamespace(
        a_amplitude=CheckValue.FAIL, b1_ratio=0.05, slow_libration=CheckValue.PASS, identity=CheckValue.FAIL, force_r2_h1=0.01
    )
    ledger = Ledger.from_settings(settings)
    return decide._kinematic(angle, geometry, resp, None, settings, ledger), ledger


@pytest.mark.parametrize(
    'shift, max_pass_rev, word',
    [
        (1.82, 0.94, AngleStatus.DRIFT),  # 62243 1M-4J-1: the author's drift
        (1.57, 1.039, AngleStatus.CIRCULATION),  # 74729: the centre passes a revolution
        (1.06, 0.3, AngleStatus.DRIFT),  # 92336
    ],
)
def test_a_kinematic_libration_whose_centre_moves_is_a_kinematic_drift(shift, max_pass_rev, word):
    decision, ledger = _kinematic(LIBRATION, _geometry(shift, max_pass_rev))
    assert (decision.status, decision.reason, decision.origin, decision.rule) == (
        Status.NON_RESONANT,
        Reason.KINEMATIC_DRIFT,
        LibrationOrigin.KINEMATIC,
        decide.RULE_K2,
    )
    assert decision.angle_word == word and decision.origin_detail == 'varpi_term'
    entry = next(e for e in ledger if e.name == decide.K2B_SHIFT)
    assert entry.result and entry.value == shift and entry.threshold == S.mmr_kin_drift_shift_ratio


@pytest.mark.parametrize('angle, shift', [(LIBRATION, 0.8), (LIBRATION, 1.0), (LIBRATION, NAN), (EPISODIC, 1.82)])
def test_a_kinematic_libration_about_a_fixed_centre_stays(angle, shift):
    decision, _ = _kinematic(angle, _geometry(shift, 0.94))
    assert (decision.reason, decision.rule, decision.angle_word) == (Reason.KINEMATIC_LIBRATION, '3', None)


def test_the_shift_ratio_is_a_setting():
    strict = settings_from_config(mmr_kin_drift_shift_ratio=2.0)
    decision, _ = _kinematic(LIBRATION, _geometry(1.82, 0.94), strict)
    assert decision.reason == Reason.KINEMATIC_LIBRATION


# --- part 5: G4b ------------------------------------------------------------------------------------------------------


def _centre_geometry(max_pass_rev, net_rev):
    turns = SimpleNamespace(times_yr=np.array([20000.0, 60000.0]))  # two interior turns: a drift, not an arc (G5a)
    centre = SimpleNamespace(
        max_pass_rad=max_pass_rev * TWO_PI, net_rev=net_rev, turns=turns, window_yr=1000.0, range_rad=abs(net_rev) * TWO_PI
    )
    return SimpleNamespace(centre=centre, drift_fraction=0.9, window=(0.0, 200000.0), turning_points=[])


SAMPLING = make_sampling(verdict=StepVerdict.RESOLVED, alias_numpy_differs=False)


@pytest.mark.parametrize(
    'max_pass_rev, net_rev, word, rule',
    [
        (0.9, 2.75, AngleStatus.CIRCULATION, 'G4b'),  # 142694 1V+3S-7 200 kyr: the author's circulation
        (0.9, -2.2, AngleStatus.CIRCULATION, 'G4b'),  # either direction
        (0.9, 2.0, AngleStatus.CIRCULATION, 'G4b'),  # at the threshold
        (0.78, 1.81, AngleStatus.DRIFT, 'G5'),  # 21028 7E-10M-6: the author's drift
        (0.9, NAN, AngleStatus.DRIFT, 'G5'),  # no number decides nothing
        (1.2, 0.5, AngleStatus.CIRCULATION, 'G4'),  # one pass of a revolution: G4 as before
    ],
)
def test_g4b_net_shift_of_the_slow_centre(max_pass_rev, net_rev, word, rule):
    angle = decide._g4_to_g6(SAMPLING, _centre_geometry(max_pass_rev, net_rev), S, Ledger.from_settings(S))
    assert (angle.angle_status, angle.rule) == (word, rule)


@pytest.mark.parametrize('net_rev, word', [(2.75, AngleStatus.CIRCULATION), (1.81, AngleStatus.DRIFT), (NAN, AngleStatus.DRIFT)])
def test_k2_angle_word_reads_the_net_shift(net_rev, word):
    ledger = Ledger.from_settings(S)
    decision = decide._k2_decision(_centre_geometry(0.78, net_rev), [], S, ledger)
    assert decision.angle_word == word and decision.rule == decide.RULE_K2


# --- the benchmark records (skipped without the benchmark data) ---------------------------------------------------------


def _verdict(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), S)


@pytest.mark.benchmark
def test_37908_circulation_without_a_force_law_reaches_high():
    analysis = _verdict('mmr2__37908__23J-8+0-15')
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.confidence)) == ('non_resonant', 'circulation', 'high')
    assert str(verdict.margins['C4'].margin) == 'not_measured'
    assert analysis.response.force_r2_whole == pytest.approx(0.005, abs=0.005)


@pytest.mark.benchmark
def test_16017_kinematic_episode_reaches_high():
    verdict = _verdict('mmr2__16017__7S-1+0-6').verdict
    assert (str(verdict.status), str(verdict.reason), str(verdict.confidence)) == ('non_resonant', 'kinematic_libration', 'high')
    assert str(verdict.margins['L3'].margin) == 'not_applicable' and str(verdict.margins['F'].margin) == 'full'


def _explained(analysis):
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    return verdict_text.explain_mmr(analysis, S, row, analysis.episode_rows())


@pytest.mark.benchmark
def test_62243_kinematic_libration_with_a_moving_centre_is_a_drift():
    analysis = _verdict('mmr3__62243__1M-4J-1+0+0+4')
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.reason)) == ('non_resonant', 'drift', 'kinematic_drift')
    assert str(verdict.confidence) == 'medium'  # the author's label of 2026-10-24
    shape = next(step for step in _explained(analysis).steps if step.key == 'shape')
    assert 'moves 1.82 > 1 × its amplitude' in shape.outcome


@pytest.mark.benchmark
def test_142694_staircase_over_200_kyr_circulates():
    analysis = _verdict('mmr3__142694__1V+3S-7+0+0+3__200kyr')
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.confidence)) == ('non_resonant', 'circulation', 'medium')
    assert analysis.geometry.centre.max_pass_rev < 1 and abs(analysis.geometry.centre.net_rev) == pytest.approx(2.75, abs=0.01)
    assert 'advances 2.75 ≥ 2 revolutions net' in _explained(analysis).sentences[0]


def test_f_only_for_a_kinematic_non_resonant_verdict_with_a_direct_strength():
    kinematic = StatusDecision(Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, LibrationOrigin.KINEMATIC, '', '', rule='3')
    dynamic = StatusDecision(Status.RESONANT, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC, '', '', rule='5a')
    failed = _strength(CheckValue.FAIL)
    assert decide._kinematic_f(kinematic, failed, {})['F'].margin == Margin.FULL
    assert decide._kinematic_f(dynamic, failed, {}) == {}
    assert decide._kinematic_f(kinematic, _strength(CheckValue.FAIL, st.METHOD_SCALE), {}) == {}  # e^q scale (285637)
    assert decide._kinematic_f(kinematic, None, {}) == {}
    assert decide._kinematic_f(kinematic, failed, {'F': 'kept'}) == {}
