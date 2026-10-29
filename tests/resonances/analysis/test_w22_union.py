"""W22 union rule (author 2026-10-24, item 8; round 1024 safe variant): librating episodes of any MMR without a turn
in a gap or over the whole chain, no gap longer than LIBRATION_UNION_GAP_PERIODS x (P_left + P_right), both edges
reached, cover the record together -> libration/regime_change. For an angle that is not 1:1 (rule G2u) G2c is tried
first, the status tree reads the union as episodic and only a dynamic transient becomes resonant.

Independent numbers (the records are built from their turning points, cosine half-cycles between them):
- two regimes, +-30 deg about 0 (half-period 90 yr, P 180) then 55/115 deg (half-period 85, P 170), joined by one
  half-cycle of `gap` yr: gap / (P_left + P_right) = 80 / 350 = 0.229; 340 / 350 = 0.971 joins, 360 / 350 = 1.029 does
  not; the turning points in the gap range from -30 to 115 deg = 145 deg;
- regimes of +-30 deg about 0, 100, 200 (, 300, 400) deg, 300-yr gaps (above 3 x the 90-yr half-period, so every
  regime is its own run; 300 / 360 = 0.83 joins): three regimes span -30..230 deg = 260 deg < 360 (one libration),
  five span -30..430 deg = 460 deg >= 360 (the centre goes round: no union);
- the edge: window 0-1000 yr, the outer episode ends at 800 yr, P = 100 yr: a turning point at 920 yr leaves 0.8 P,
  one at 850 yr 1.5 P, none 2.0 P;
- the benchmark records (round 1024, n1_changes_final_719rows.csv and its REPORT): 2004 TK14 31M-34 at 0.5 yr becomes
  resonant/libration/high, 2002 XA 5J-2 keeps two episodes (gap 1.05), 136108 7N-12 stays resonant/stationary at the
  scale of its centre although the fine scale holds a union, 2003 TX9 stays uncertain with the word libration.
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.filtering import edge_window, lowpass
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.analyse import _centre_scale_union_word
from resonances.resonance.analysis.mmr.decide import AngleDecision, StatusDecision
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Flag, LibrationOrigin, Margin, Reason, Status
from tests.resonances.analysis.builders import make_response, make_sampling

S = settings_from_config()
ONE_DEG = math.radians(1.0)
SAMPLING = make_sampling(
    unwrap_resid_p99_rad=0.0, unwrap_resid_max_rad=0.0, unwrap_ok=True, undefined_phase=False, verdict='resolved', alias_numpy_differs=False
)
MMR_RULES = dict(edge_arc_rule=True, return_closure=True, split_regimes=True, cycle_travel=True)


def through(times, values, t):
    """sigma(t) through the turning points (times, values [deg]): a half-cosine between consecutive ones, mirrored
    beyond the first and the last [radians]."""
    times, values = np.asarray(times, float), np.radians(np.asarray(values, float))
    times = np.concatenate([[2 * times[0] - times[1]], times, [2 * times[-1] - times[-2]]])
    values = np.concatenate([[values[1]], values, [values[-2]]])
    out = np.full(t.shape, values[0])
    for (t0, v0), (t1, v1) in zip(zip(times, values), zip(times[1:], values[1:])):
        inside = (t >= t0) & (t <= t1)
        out[inside] = v0 + (v1 - v0) * (1 - np.cos(np.pi * (t[inside] - t0) / (t1 - t0))) / 2
    out[t > times[-1]] = values[-1]
    return out


def decided(times, values, coorbital=False):
    t = np.arange(0.0, times[-1] + 60.0, 0.5)
    sigma = through(times, values, t)
    angle_f = lowpass(t, sigma, 10.0, 2)
    rate = np.gradient(angle_f, t)
    angle = g.FilteredAngle(
        t_yr=t,
        angle_raw=sigma,
        angle_f=angle_f,
        rate_f=rate,
        angle_f_half=lowpass(t, sigma, 5.0, 2),
        angle_f_double=lowpass(t, sigma, 20.0, 2),
        slow_f=rate / -30.0,
        identity_coefficient=-30.0,
        cutoff_yr=10.0,
        step_yr=0.5,
        window=edge_window(t, 10.0),
        coorbital=coorbital,
        **MMR_RULES,
    )
    geometry = g.measure_geometry(angle, ONE_DEG, S)
    ledger = Ledger.from_settings(S)
    return decide.decide_angle(SAMPLING, False, geometry, None, S, ledger), geometry, ledger


def two_regimes(gap):
    first_t = 60.0 + 90.0 * np.arange(6)
    second_t = first_t[-1] + gap + 85.0 * np.arange(6)
    return np.concatenate([first_t, second_t]), [30.0, -30.0] * 3 + [115.0, 55.0] * 3


def regimes_about(centres, gap=300.0):
    times, values, start = [], [], 60.0
    for centre in centres:
        times += list(start + 90.0 * np.arange(6))
        values += [centre + 30.0, centre - 30.0] * 3
        start = times[-1] + gap
    return np.asarray(times), values


# --- geometry -------------------------------------------------------------------------------------------------------


def test_two_regimes_of_another_resonance_join_as_g2u():
    angle, geometry, ledger = decided(*two_regimes(80.0))
    assert (angle.angle_status, angle.subtype, angle.rule) == (AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE, decide.RULE_UNION)
    columns = decide.union_columns(geometry, S)
    assert columns['geo_union_n_episodes'] == 2
    assert columns['geo_union_gap_max_P'] == pytest.approx(80.0 / 350.0, abs=0.01)
    assert columns['geo_union_tp_range_rad'] == pytest.approx(math.radians(145.0), abs=math.radians(1.0))
    assert columns['geo_union_span_tp_range_rad'] == pytest.approx(math.radians(145.0), abs=math.radians(1.0))
    assert 'geo_union_gap_max_P' in ledger.columns() and 'geo_union_span_raw_range_rad' in ledger.columns()
    # the same record of a 1:1 angle: the plain rule G2, read as libration by the status
    one_to_one, _, _ = decided(*two_regimes(80.0), coorbital=True)
    assert (one_to_one.angle_status, one_to_one.rule) == (AngleStatus.LIBRATION, 'G2')
    assert decide.status_basis(one_to_one) is one_to_one


@pytest.mark.parametrize('gap, joins', [(340.0, True), (360.0, False)])
def test_the_gap_may_miss_at_most_one_period_on_each_side(gap, joins):
    angle, geometry, _ = decided(*two_regimes(gap))
    assert len(decide.librating_episodes(geometry, S)) == 2
    assert (angle.angle_status == AngleStatus.LIBRATION) is joins
    if joins:
        assert decide.union_columns(geometry, S)['geo_union_gap_max_P'] == pytest.approx(gap / 350.0, abs=0.01)
    else:
        assert (angle.angle_status, angle.subtype) == (AngleStatus.EPISODIC, AngleSubtype.RECURRENT)


def test_a_centre_that_goes_round_over_the_chain_is_no_union():
    three, geometry, _ = decided(*regimes_about([0.0, 100.0, 200.0]))
    assert len(decide.librating_episodes(geometry, S)) == 3
    assert (three.angle_status, three.rule) == (AngleStatus.LIBRATION, decide.RULE_UNION)
    span = decide.union_columns(geometry, S)['geo_union_span_tp_range_rad']
    assert span == pytest.approx(math.radians(260.0), abs=math.radians(1.0))
    five, geometry, _ = decided(*regimes_about([0.0, 100.0, 200.0, 300.0, 400.0]))
    assert len(decide.librating_episodes(geometry, S)) == 5
    assert (five.angle_status, five.subtype) == (AngleStatus.EPISODIC, AngleSubtype.RECURRENT)
    assert decide.union_columns(geometry, S)['geo_union_n_episodes'] == 0


def _edge_geometry(tp_times, tp_values=None, raw=(0.0, 0.0)):
    tp_values = tp_values if tp_values is not None else [0.0] * len(tp_times)
    t = np.array([800.0, 1000.0])
    return SimpleNamespace(
        window=(0.0, 1000.0),
        turning_points=SimpleNamespace(times_yr=np.array(tp_times, float), values_rad=np.array(tp_values, float)),
        averaged_raw=np.array(raw, float),
        averaged_raw_t_yr=t,
    )


@pytest.mark.parametrize('tp_times, shortfall', [([800.0, 920.0], 0.8), ([800.0, 850.0], 1.5), ([800.0], 2.0)])
def test_the_turning_points_beyond_the_episode_reach_the_edge(tp_times, shortfall):
    outer = SimpleNamespace(t_start_yr=500.0, t_end_yr=800.0, period_med_yr=100.0, exit_observed=False)
    assert g._union_edge(_edge_geometry(tp_times), outer, 'end') == pytest.approx(shortfall)


def test_an_observed_exit_or_a_turn_closes_the_edge():
    outer = SimpleNamespace(t_start_yr=500.0, t_end_yr=800.0, period_med_yr=100.0, exit_observed=True)
    assert g._union_edge(_edge_geometry([800.0, 920.0]), outer, 'end') == math.inf
    outer.exit_observed = False
    assert g._union_edge(_edge_geometry([800.0, 920.0], raw=(0.0, 7.0)), outer, 'end') == math.inf  # 7 rad > 2 pi
    assert g._union_edge(_edge_geometry([800.0, 920.0], [0.0, 7.0]), outer, 'end') == math.inf


# --- the physics guard ----------------------------------------------------------------------------------------------

UNION = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE, decide.RULE_UNION, 1)


def _decision(status, reason, origin, rule='5d/6'):
    return StatusDecision(status, reason, origin, '', '', rule=rule)


def test_a_dynamic_transient_of_a_union_becomes_resonant():
    decision = _decision(Status.TRANSIENT, Reason.DYNAMIC_EPISODE, LibrationOrigin.DYNAMIC)
    ledger = Ledger.from_settings(S)
    decide.promote_union(UNION, decision, None, ledger)
    assert (decision.status, decision.reason, decision.union_promoted) == (Status.RESONANT, Reason.DYNAMIC_LIBRATION, True)
    assert decision.rule == '5d/6u'  # the final rule names the promotion (rec_rule)
    assert next(e for e in ledger if e.name == decide.UNION_PROMOTED).result is True
    read = decide.union_read(UNION, decision)
    assert read.rule == 'G2' and decide.status_basis(read).angle_status == AngleStatus.LIBRATION


def test_a_promoted_5c_union_keeps_its_branch_for_the_margins():
    """5c/6 (the potential carries a failed force law) promoted: rule 5c/6u, and L4 stays thin as on 5c/6."""
    decision = _decision(Status.TRANSIENT, Reason.DYNAMIC_EPISODE, LibrationOrigin.DYNAMIC, rule='5c/6')
    decide.promote_union(UNION, decision, None, Ledger.from_settings(S))
    assert decision.rule == '5c/6u' and decision.status == Status.RESONANT
    response = make_response(a_response=CheckValue.FAIL, force_r2_h1=0.2, force_r2_h2=0.3)
    assert decide._l4(response, decision).margin == Margin.THIN


def test_the_zone_flag_of_a_promoted_libration():
    decision = _decision(Status.TRANSIENT, Reason.DYNAMIC_EPISODE, LibrationOrigin.DYNAMIC)
    decide.promote_union(UNION, decision, SimpleNamespace(cycle_classes=['zone', 'deep']), Ledger.from_settings(S))
    assert str(Flag.LIBRATION_IN_SEPARATRIX_ZONE) in decision.flags


@pytest.mark.parametrize(
    'status, reason, origin, rule',
    [
        (Status.TRANSIENT, Reason.GEOMETRIC_EPISODE, LibrationOrigin.UNKNOWN, '5u/6'),  # the geometry alone
        (Status.UNCERTAIN, Reason.ORIGIN_UNKNOWN, LibrationOrigin.UNKNOWN, '5a'),  # 2003 TX9: the force law fails
        (Status.NON_RESONANT, Reason.IMPLAUSIBLE_STRENGTH, LibrationOrigin.UNKNOWN, '5s'),
        (Status.NON_RESONANT, Reason.INDUCED_LIBRATION, LibrationOrigin.INDUCED, '2'),
        (Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, LibrationOrigin.KINEMATIC, '3'),
    ],
)
def test_every_other_reading_stays(status, reason, origin, rule):
    decision = _decision(status, reason, origin, rule)
    decide.promote_union(UNION, decision, None, Ledger.from_settings(S))
    assert (decision.status, decision.reason, decision.union_promoted, decision.rule) == (status, reason, False, rule)
    assert decide.union_read(UNION, decision) is UNION
    assert decide.status_basis(UNION).angle_status == AngleStatus.EPISODIC
    reported = decide.reported_angle(UNION, None, S, decision)
    assert (reported.angle_status, reported.subtype) == (AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE)


def test_a_one_to_one_union_is_not_guarded():
    one_to_one = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE, 'G2', 1)
    decision = _decision(Status.RESONANT, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC)
    ledger = Ledger.from_settings(S)
    decide.promote_union(one_to_one, decision, None, ledger)
    assert not decision.union_promoted and not any(e.name == decide.UNION_PROMOTED for e in ledger)


def test_the_word_of_a_union_at_the_centre_scale():
    episodic = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.RECURRENT, 'G2', 0)
    uncertain = _decision(Status.UNCERTAIN, Reason.ORIGIN_UNKNOWN, LibrationOrigin.UNKNOWN, '5a')
    ledger = Ledger.from_settings(S)
    word = _centre_scale_union_word(episodic, True, uncertain, ledger)
    assert (word.angle_status, word.subtype) == (AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE)
    assert 'geo_union_word_centre_scale' in ledger.columns()
    assert _centre_scale_union_word(episodic, False, uncertain, ledger) is episodic
    circulation = AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4', None)
    assert _centre_scale_union_word(circulation, True, uncertain, ledger) is circulation
    k2 = StatusDecision(Status.NON_RESONANT, Reason.KINEMATIC_DRIFT, LibrationOrigin.KINEMATIC, '', '', angle_word=AngleStatus.DRIFT)
    assert _centre_scale_union_word(episodic, True, k2, ledger) is episodic


# --- the benchmark records (skipped without the benchmark data) ------------------------------------------------------


def _analysis(example, settings=S, **options):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), settings, **options)


def _word(verdict):
    return str(verdict.status), str(verdict.angle_status), str(verdict.angle_subtype), str(verdict.confidence)


@pytest.mark.benchmark
def test_tk14_half_year_is_resonant():
    analysis = _analysis('mmr2__2004_TK14__31M-34+0+3__step0.5yr')
    assert _word(analysis.verdict) == ('resonant', 'libration', 'regime_change', 'high')
    assert str(analysis.verdict.reason) == 'dynamic_libration'
    assert analysis.extra_columns['geo_union_read_episodic'] == 0
    assert next(e for e in analysis.verdict.decision_path if e.name == decide.UNION_PROMOTED).result is True


@pytest.mark.benchmark
def test_tk14_five_years_is_resonant_at_medium():
    """The author's resonant/medium/libration: the union rule and R-ALIAS-A together (the cap holds medium)."""
    verdict = _analysis('mmr2__2004_TK14__31M-34+0+3').verdict
    assert _word(verdict) == ('resonant', 'libration', 'regime_change', 'medium')


# the G2c analysis the union guard rejected, as the W23 design's control run persisted it (episodes/full-d, a monkeypatch
# of the W22 code): W_c, TP prominence at the 1-deg floor, status, reason, caps, accepted episodes, the decisive one
TX9_REJECTED = {
    'mmr3__2003_TX9__3E+2J-10+0+0+5': (1107.77, 'geometry_only;short_period_aliased', '2211-7167;7992-14564;15249-17489', '7992-14564'),
    'mmr3__2003_TX9__3E+2J-10+0+0+5__step0.5yr': (1108.27, 'geometry_only', '2211-7168;7994-14563;15250-17492', '7994-14563'),
}


@pytest.mark.benchmark
@pytest.mark.parametrize('example', sorted(TX9_REJECTED))
def test_tx9_keeps_its_status_with_the_word_libration(example):
    """The W22 path, now only without the W23 centre-scale TP rule (with it G2c keeps TX9: test_w23_centre_scale.py):
    the union found at W_c is rejected, the fine verdict stays with the word libration; the rejected analysis is in the
    geo_cs_rej_* columns (W23 d)."""
    # G2a off: these mechanisms are read on the single episode (W24, test_w23_centre_scale.test_tx9_g2a_*)
    analysis = _analysis(example, replace(S, libration_centre_scale_tp_at_centre=False, libration_action_step=False))
    assert (str(analysis.verdict.status), str(analysis.verdict.angle_status)) == ('uncertain', 'libration')
    assert analysis.extra_columns['geo_union_word_centre_scale'] == 1
    assert analysis.prep.cutoff.binding != 'centre_scale'  # the fine verdict stays
    cutoff, caps, episodes, decisive = TX9_REJECTED[example]
    columns = analysis.extra_columns
    assert columns['geo_cs_rej_cutoff_yr'] == pytest.approx(cutoff, abs=0.01)
    assert columns['geo_cs_rej_tp_prominence_deg'] == pytest.approx(S.mmr_tp_prominence_floor_deg)
    rejected = [columns[f'geo_cs_rej_{k}'] for k in ('status', 'angle_status', 'reason', 'origin', 'caps')]
    # the control run wrote dynamic_episode: origin unknown is geometric_episode since 2026-10-26 (author, Q3)
    assert rejected == ['transient', 'libration', 'geometric_episode', 'unknown', caps]
    assert (columns['geo_cs_rej_n_accepted'], columns['geo_cs_rej_union_n']) == (3, 3)
    assert (columns['geo_cs_rej_episodes'], columns['geo_cs_rej_decisive']) == (episodes, decisive)


@pytest.mark.benchmark
def test_136108_g2c_comes_before_the_union():
    fine = _analysis('mmr2__136108__7N-12+0+5', centre_scale=False)
    assert (str(fine.verdict.angle_status), str(fine.verdict.angle_subtype)) == ('libration', 'regime_change')
    analysis = _analysis('mmr2__136108__7N-12+0+5')
    assert _word(analysis.verdict) == ('resonant', 'libration', 'stationary', 'high')
    assert analysis.prep.cutoff.binding == 'centre_scale'


@pytest.mark.benchmark
def test_2002_xa_gap_is_longer_than_the_two_periods():
    analysis = _analysis('mmr2__2002_XA__5J-2+0-3')
    left, right = sorted(decide.librating_episodes(analysis.geometry, S), key=lambda e: e.t_start_yr)
    ratio = (right.t_start_yr - left.t_end_yr) / (left.period_med_yr + right.period_med_yr)
    assert ratio == pytest.approx(1.05, abs=0.02) and ratio > S.libration_union_gap_periods
    assert _word(analysis.verdict) == ('transient', 'episodic', 'recurrent', 'medium')


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, word',
    [
        ('mmr2__138537__13N-25+0+12', ('non_resonant', 'episodic', 'recurrent', 'low')),  # 48 episodes, centre goes round
        ('mmr2__2001_WK15__32E-39+0+7', ('transient', 'episodic', 'recurrent', 'medium')),
        ('mmr3__2003_TG2__4E-1M-3+0+0+0__step0.5yr', ('transient', 'episodic', 'recurrent', 'low')),  # a stall of 12.6 P
        ('mmr2__15875__2N-3+0+1', ('resonant', 'libration', 'stationary', 'high')),
    ],
)
def test_records_the_union_must_not_change(example, word):
    analysis = _analysis(example)
    assert _word(analysis.verdict) == word
    assert analysis.extra_columns['geo_union_n_episodes'] == 0
