"""Independent functional tests of rule R (W10: the force law, the identity and B1 per librating episode).

Expected values are derived here, not read from the implementation:

- rule R (impl/w10/per-episode/REPORT.md): a witness is a librating episode whose force law (L4) passes strictly
  while B1 does not fail and L3 does not fail (W11: not_measured allowed), or passes thin with L3 and B1 passing on
  the same episode; with witnesses the three checks
  come from the one with the most turning points (the earlier on a tie), a not_measured never replaces a measured
  value; without a witness the pooled measurement stays (under G1e the pool over the episodes before the last jump);
  a witness next to an episode whose L4 is measured and fails gives the flag and cap `episode_force_law_mixed`
  (confidence <= medium);
- synthetic records of 2J-1: a pendulum sigma_ddot = -w0^2 sin(sigma - pi) with the body's n = 2 n_J - sigma_dot and
  a from Kepler's third law, so the identity and the force law a_dot ~ sin(sigma - pi) hold by construction on a
  "genuine" episode. On a "shuffled" episode a - <a> is replaced by its Hilbert transform: every Fourier component
  of a is turned by 90 degrees, so a_dot is in quadrature with sin(sigma - phi0) for any phi0 (only odd harmonics of
  the libration frequency there, while cos(sigma - pi) carries only even ones) and R^2 ~ cos^2(90 deg) = 0;
- two episodes are separated by a rest of sigma at the centre (half-amplitude 0.3 deg: no turning point at the
  libration's prominence, a gap > 3 half-periods), a jump of a by a circulation of energy 8 w0^2 (sigma_dot ~ 4.2 w0
  shifts n by ~4 % of 2 n_J - the jump of a ~ 2.5 %).
"""

import dataclasses
import math
from functools import lru_cache

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.signal import hilbert

from resonances.data import const
from resonances.plotting.verdict_text import MMRFacts, measured_on
from resonances.resonance.analysis.mmr import response as rsp
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import (
    EpisodeResponse,
    MMRSeries,
    Regime,
    ResonanceSpec,
    Response,
    ResponseSource,
)
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CONFIDENCE_ORDER, CheckValue, Confidence, Reason, Status

SETTINGS = AnalysisSettings()
N_J = math.radians(const.PLANET_N_DEG_PER_GAUSS_YR["Jupiter"])
P0_YR = 600.0
W0 = 2 * np.pi / P0_YR
PASS, FAIL, NM, NA = CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_MEASURED, CheckValue.NOT_APPLICABLE
MIXED = 'episode_force_law_mixed'


# ------------------------------------------------------------------------------------------- dataclass builders

_EMPTY = {
    'float': lambda: float('nan'),
    'int': lambda: 0,
    'bool': lambda: False,
    'str': lambda: '',
    'CheckValue': lambda: CheckValue.NOT_MEASURED,
    'np.ndarray': lambda: np.array([]),
}


def blank(cls, **values):
    """An instance with every required field not in `values` set to 'not measured' (nan, None, empty)."""
    kwargs = dict(values)
    for f in dataclasses.fields(cls):
        has_default = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        if f.name not in values and not has_default:
            kind = str(f.type)
            if kind.startswith('List'):
                kwargs[f.name] = []
            elif kind.startswith('Tuple'):
                kwargs[f.name] = ()
            else:
                kwargs[f.name] = _EMPTY.get(kind, lambda: None)()
    return cls(**kwargs)


def episode(episode_id, n_tp=20, l4=PASS, thin=False, l3=PASS, b1=PASS, t0=0.0, t1=1000.0, r2=0.9, slope=1.0, b1_ratio=1.0):
    """A judged episode response; its numbers are tagged so that their origin can be traced."""
    return blank(
        EpisodeResponse,
        episode_id=episode_id,
        before_jump=False,
        n_tp=n_tp,
        t_start_yr=t0,
        t_end_yr=t1,
        period_yr=P0_YR,
        force_r2_h1=r2,
        force_r2_h2=r2,
        identity_slope_ratio=slope,
        identity_corr=0.99,
        b1_ratio=b1_ratio,
        a_response=l4,
        a_response_thin=thin,
        identity=l3,
        a_amplitude=b1,
    )


def record_response(episodes=(), before_jump=None, l4=PASS, thin=False, l3=PASS, b1=PASS, r2=0.55, slope=1.5):
    """The record's pooled response with tagged numbers (r2 0.55, slope ratio 1.5, b1 ratio 7)."""
    return blank(
        Response,
        a_response=l4,
        a_response_thin=thin,
        identity=l3,
        a_amplitude=b1,
        force_r2_h1=r2,
        force_r2_h2=r2,
        identity_slope_ratio=slope,
        identity_corr=0.5,
        b1_ratio=7.0,
        slow_libration=FAIL,
        episode_responses=list(episodes),
        before_jump_response=before_jump,
    )


# ------------------------------------------------------------------------------------------- the witness


@pytest.mark.parametrize(
    'l4, thin, l3, b1, expected',
    [
        (PASS, False, PASS, PASS, True),
        (PASS, False, FAIL, PASS, False),  # W11: strict L4 with a failed identity is no witness
        (PASS, False, NM, NM, True),  # strict L4 and B1 not failing
        (PASS, False, PASS, FAIL, False),  # B1 fails
        (PASS, True, PASS, PASS, True),  # thin L4 with L3 and B1 passing
        (PASS, True, PASS, NM, False),
        (PASS, True, NM, PASS, False),
        (PASS, True, FAIL, PASS, False),
        (FAIL, False, PASS, PASS, False),
        (NM, False, PASS, PASS, False),
        (NA, False, PASS, PASS, False),
    ],
)
def test_witness_truth_table(l4, thin, l3, b1, expected):
    assert episode(0, l4=l4, thin=thin, l3=l3, b1=b1).witness is expected


def test_unjudged_run_and_pool_are_never_witnesses():
    unjudged = dataclasses.replace(episode(3), a_response=None, identity=None, a_amplitude=None)
    pooled = dataclasses.replace(episode(0), episode_id=None, before_jump=True)
    assert not unjudged.witness
    assert not pooled.witness
    assert unjudged.to_columns()['check_a_response'] == ''
    assert unjudged.to_columns()['witness'] is False
    cols = episode(1, l4=FAIL).to_columns()
    assert cols['check_a_response'] == 'fail' and cols['witness'] is False


def test_witnesses_and_conflict():
    good, bad, unmeasured = episode(0), episode(1, l4=FAIL), episode(2, l4=NM)
    thin_no_b1 = episode(3, thin=True, b1=NM)  # passes thin, no witness, but its force law does not fail
    assert [e.episode_id for e in rsp.witnesses(record_response([good, bad]))] == [0]
    assert rsp.force_law_mixed(record_response([good, bad]))
    assert rsp.force_law_mixed(record_response([bad, good]))
    assert not rsp.force_law_mixed(record_response([good, unmeasured]))
    assert not rsp.force_law_mixed(record_response([good, thin_no_b1]))
    assert not rsp.force_law_mixed(record_response([good, episode(4)]))
    assert not rsp.force_law_mixed(record_response([bad, unmeasured]))  # no witness: nothing to conflict with
    assert rsp.witnesses(record_response()) == []


# ------------------------------------------------------------------------------------------- deciding_response


def test_no_witness_leaves_the_pooled_response_unchanged():
    response = record_response([episode(0, l4=FAIL, r2=0.1), episode(1, thin=True, b1=FAIL, r2=0.5)])
    d = rsp.deciding_response(response, before_jump=False)
    assert d.a_response == PASS and d.identity == PASS and d.a_amplitude == PASS
    assert d.force_r2_h1 == 0.55 and d.identity_slope_ratio == 1.5 and d.b1_ratio == 7.0
    assert (d.force_source, d.identity_source, d.b1_source) == (ResponseSource.RECORD,) * 3
    assert d.witness_episode_id is None
    assert d.slow_libration == FAIL  # outside G1e B5 stays the record's


def test_a_pool_before_the_jump_is_not_read_outside_g1e():
    pool = dataclasses.replace(episode(None, l4=FAIL, r2=0.05), before_jump=True)
    d = rsp.deciding_response(record_response([episode(0, l4=FAIL)], before_jump=pool), before_jump=False)
    assert d.force_source == ResponseSource.RECORD and d.a_response == PASS and d.force_r2_h1 == 0.55


def test_witness_with_most_turning_points_decides_all_three_checks():
    short = episode(0, n_tp=10, r2=0.81, slope=1.01, b1_ratio=1.1, t0=0.0, t1=3000.0)
    long = episode(1, n_tp=14, thin=True, r2=0.62, slope=1.02, b1_ratio=1.2, t0=5000.0, t1=9000.0)
    record = record_response([short, long], l4=FAIL, l3=FAIL, b1=FAIL)
    d = rsp.deciding_response(record, before_jump=False)
    assert d.witness_episode_id == 1
    assert (d.force_source, d.identity_source, d.b1_source) == (ResponseSource.WITNESS,) * 3
    assert d.a_response == PASS and d.a_response_thin  # the witness's thin pass with its flag
    assert d.force_r2_h1 == 0.62 and d.identity_slope_ratio == 1.02 and d.b1_ratio == 1.2
    assert d.identity == PASS and d.a_amplitude == PASS
    cols = d.to_columns()
    assert cols['rsp_force_source'] == 'witness'
    assert cols['rsp_witness_episode_id'] == 1
    assert cols['rsp_n_witnesses'] == 2
    assert cols['rsp_n_episode_force_fail'] == 0


def test_tie_in_turning_points_takes_the_earlier_episode():
    early = episode(0, n_tp=12, r2=0.91, t0=0.0, t1=4000.0)
    late = episode(1, n_tp=12, r2=0.92, t0=6000.0, t1=10000.0)
    d = rsp.deciding_response(record_response([late, early]), before_jump=False)
    assert d.witness_episode_id == 0 and d.force_r2_h1 == 0.91


def test_not_measured_of_the_witness_never_replaces_a_measured_value():
    witness = episode(0, n_tp=16, r2=0.95, l3=NM, b1=NM, slope=float('nan'), b1_ratio=float('nan'))
    d = rsp.deciding_response(record_response([witness], l3=FAIL, b1=PASS), before_jump=False)
    assert d.force_source == ResponseSource.WITNESS and d.force_r2_h1 == 0.95
    assert d.identity == FAIL and d.identity_source == ResponseSource.RECORD and d.identity_slope_ratio == 1.5
    assert d.a_amplitude == PASS and d.b1_source == ResponseSource.RECORD and d.b1_ratio == 7.0


def test_measured_value_of_the_witness_replaces_a_measured_pooled_one():
    d = rsp.deciding_response(record_response([episode(0, l3=PASS)], l3=FAIL), before_jump=False)
    assert d.identity == PASS and d.identity_source == ResponseSource.WITNESS and d.identity_slope_ratio == 1.0


def test_input_response_is_not_modified():
    record = record_response([episode(0, r2=0.97)], l4=FAIL)
    rsp.deciding_response(record, before_jump=False)
    assert record.a_response == FAIL and record.force_r2_h1 == 0.55 and record.force_source == ResponseSource.RECORD


def test_g1e_without_witness_reads_the_pool_before_the_jump():
    pool = dataclasses.replace(episode(None, l4=FAIL, l3=PASS, b1=PASS, r2=0.03, slope=0.98), before_jump=True)
    eps = [dataclasses.replace(episode(0, l4=FAIL), before_jump=True)]
    # the last regime's measurement (a strict pass) says nothing about the episodes before the jump
    d = rsp.deciding_response(record_response(eps, before_jump=pool, l4=PASS), before_jump=True)
    assert (d.force_source, d.identity_source, d.b1_source) == (ResponseSource.BEFORE_JUMP,) * 3
    assert d.a_response == FAIL and d.force_r2_h1 == 0.03 and d.identity_slope_ratio == 0.98
    assert d.witness_episode_id is None
    assert d.slow_libration in (NM, NA)  # B5 of the last regime is blanked


def test_g1e_without_any_measurement_gives_none():
    assert rsp.deciding_response(record_response([], before_jump=None), before_jump=True) is None
    assert rsp.deciding_response(None, before_jump=False) is None


def test_g1e_witness_not_measured_falls_back_to_the_pool_not_to_the_last_regime():
    pool = dataclasses.replace(episode(None, l3=PASS, b1=PASS, slope=0.99, b1_ratio=1.3), before_jump=True)
    witness = dataclasses.replace(episode(0, n_tp=18, l3=NM, b1=NM, r2=0.93), before_jump=True)
    record = record_response([witness], before_jump=pool, l3=FAIL, b1=FAIL)
    d = rsp.deciding_response(record, before_jump=True)
    assert d.force_source == ResponseSource.WITNESS and d.force_r2_h1 == 0.93
    assert d.identity == PASS and d.identity_source == ResponseSource.BEFORE_JUMP and d.identity_slope_ratio == 0.99
    assert d.a_amplitude == PASS and d.b1_source == ResponseSource.BEFORE_JUMP and d.b1_ratio == 1.3


# ------------------------------------------------------------------------------------------- synthetic records


def _pendulum(t, sigma0, rate0):
    sol = solve_ivp(
        lambda _, y: [y[1], -(W0**2) * np.sin(y[0] - np.pi)],
        (t[0], t[-1]),
        [sigma0, rate0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        max_step=t[1] - t[0],
    )
    return sol.y


def build_record(segments, span=20000.0, step=1.0):
    """2J-1 from `segments` (t0, t1, kind, value, shuffled): 'lib' a libration of half-amplitude `value` deg about pi,
    'rest' sigma at the centre (0.3 deg), 'circ' a circulation of energy `value` w0^2; `shuffled`: a - <a> of the
    segment replaced by its Hilbert transform."""
    t = np.arange(0.0, span + step / 2, step)
    sigma, rate = np.empty_like(t), np.empty_like(t)
    previous = np.pi
    for t0, t1, kind, value, _ in segments:
        m = (t >= t0) & (t < t1)
        if kind == 'lib':
            s0, r0 = np.pi, W0 * 2 * math.sin(math.radians(value) / 2)
        elif kind == 'rest':
            s0, r0 = np.pi + math.radians(0.3), 0.0
        else:
            s0, r0 = np.pi, W0 * math.sqrt(2 * (value + 1))
        turns = 2 * np.pi * np.round((previous - np.pi) / (2 * np.pi))
        s, r = _pendulum(t[m], s0, r0)
        sigma[m], rate[m] = s + turns, r
        previous = sigma[m][-1]
    varpi = 0.7
    lam = 2 * (0.3 + N_J * t) - varpi - sigma  # sigma = 2 lambda_J - lambda - varpi
    a = (4 * np.pi**2 / (2 * N_J - rate) ** 2) ** (1.0 / 3.0)
    for t0, t1, _, _, shuffled in segments:
        if shuffled:
            m = (t >= t0) & (t < t1)
            mean = a[m].mean()
            a[m] = mean + np.imag(hilbert(a[m] - mean))
    one = np.ones_like(t)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=0.15 * one,
        inc=0.05 * one,
        Omega=1.1 * one,
        omega=(varpi - 1.1) * one,
        varpi=varpi * one,
        angle_wrapped=np.mod(sigma, 2 * np.pi),
        resonance=ResonanceSpec.from_key('2J-1'),
        regime=Regime(step_yr=step, span_yr=span, n_rows=len(t), integrator='ias15'),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name='rule-R',
    )


RECORDS = {
    # two genuine episodes, the first one longer (0-9 kyr vs 12-20 kyr: ~30 vs ~26 TPs)
    'two_genuine': [(0, 9000, 'lib', 90, False), (9000, 12000, 'rest', 0, False), (12000, 20001, 'lib', 90, False)],
    # a long genuine episode and a short shuffled one: R0 (pooled, >= 2/3 of the windows) still holds
    'genuine_shuffled': [(0, 14000, 'lib', 90, False), (14000, 16000, 'rest', 0, False), (16000, 20001, 'lib', 90, True)],
    'genuine_genuine_control': [
        (0, 14000, 'lib', 90, False),
        (14000, 16000, 'rest', 0, False),
        (16000, 20001, 'lib', 90, False),
    ],
    # the shuffled episode has more turning points than the genuine one
    'shuffled_long_genuine_short': [
        (0, 11000, 'lib', 90, True),
        (11000, 13000, 'rest', 0, False),
        (13000, 20001, 'lib', 90, False),
    ],
    # G1e: two episodes before a jump of a, the second shuffled, then a circulation to the end
    'g1e_mixed': [
        (0, 5000, 'lib', 90, False),
        (5000, 7000, 'rest', 0, False),
        (7000, 12000, 'lib', 90, True),
        (12000, 20001, 'circ', 8, False),
    ],
    # G1e: both episodes before the jump shuffled: no witness
    'g1e_no_witness': [
        (0, 5000, 'lib', 90, True),
        (5000, 7000, 'rest', 0, False),
        (7000, 12000, 'lib', 90, True),
        (12000, 20001, 'circ', 8, False),
    ],
}


@lru_cache(maxsize=None)
def analysed(name):
    series = build_record(RECORDS[name])
    return series, analyse_mmr(series, SETTINGS)


def _strs(values):
    return {str(v) for v in values}


def _rank(level):
    return CONFIDENCE_ORDER.index(Confidence(level))


def test_hilbert_shuffled_piece_fails_the_force_law_and_genuine_piece_is_a_witness():
    """measure_pieces on two pieces of one record: R^2 ~ 1 on the genuine half, ~ cos^2(90 deg) = 0 on the other."""
    series = build_record([(0, 10000, 'lib', 90, False), (10000, 20001, 'lib', 90, True)])
    analysis = analyse_mmr(series, SETTINGS)
    judge = rsp.episode_judge(P0_YR, analysis.prep, analysis.sampling, series, SETTINGS)
    amp = math.radians(90.0)
    measured = {}
    for episode_id, piece in enumerate([(1000.0, 9000.0), (11000.0, 19000.0)]):
        measured[episode_id] = rsp.measure_pieces(
            analysis.prep,
            analysis.context,
            SETTINGS,
            [piece],
            P0_YR,
            (amp, amp),
            judge=judge,
            spec=series.resonance,
            episode_id=episode_id,
            n_tp=26,
        )
    genuine, shuffled = measured[0], measured[1]
    assert genuine.a_response == PASS and not genuine.a_response_thin
    assert min(genuine.force_r2_h1, genuine.force_r2_h2) > 0.99
    assert abs(genuine.identity_slope_ratio - 1) < 0.05 and genuine.identity == PASS
    assert abs(genuine.b1_ratio - 1) < 0.1 and genuine.a_amplitude == PASS
    assert genuine.witness
    assert shuffled.a_response == FAIL and max(shuffled.force_r2_h1, shuffled.force_r2_h2) < 0.05
    assert shuffled.identity == FAIL  # sigma_dot no longer follows a
    assert not shuffled.witness
    # without a judge only the numbers
    bare = rsp.measure_pieces(analysis.prep, analysis.context, SETTINGS, [(1000.0, 9000.0)], P0_YR, (amp, amp))
    assert bare.a_response is None and not bare.witness and bare.force_r2_h1 > 0.99


def test_two_genuine_episodes_the_one_with_more_turning_points_is_read():
    _, analysis = analysed('two_genuine')
    responses = {e.episode_id: e for e in analysis.response.episode_responses}
    assert len(responses) == 2 and all(e.witness for e in responses.values())
    most = max(responses.values(), key=lambda e: (e.n_tp, -e.t_start_yr))
    assert most.t_start_yr < 9000.0  # the first, longer episode
    cols = analysis.response.to_columns()
    assert cols['rsp_witness_episode_id'] == most.episode_id
    assert cols['rsp_n_witnesses'] == 2 and cols['rsp_n_episode_force_fail'] == 0
    assert cols['rsp_force_source'] == cols['rsp_identity_source'] == cols['rsp_b1_source'] == 'witness'
    v = analysis.verdict
    assert v.status == Status.TRANSIENT
    assert MIXED not in _strs(v.flags) and MIXED not in _strs(v.caps)


def test_witness_and_conflict_cap_medium_on_a_genuine_and_a_shuffled_episode():
    _, analysis = analysed('genuine_shuffled')
    by_start = sorted(analysis.response.episode_responses, key=lambda e: e.t_start_yr)
    genuine, shuffled = by_start
    assert genuine.witness and genuine.a_response == PASS
    assert shuffled.a_response == FAIL and not shuffled.witness
    cols = analysis.response.to_columns()
    assert cols['rsp_witness_episode_id'] == genuine.episode_id
    assert cols['rsp_n_witnesses'] == 1 and cols['rsp_n_episode_force_fail'] == 1
    v = analysis.verdict
    assert v.status == Status.TRANSIENT and v.reason == Reason.DYNAMIC_EPISODE
    assert v.checks['check_mmr_a_response'] == PASS
    assert MIXED in _strs(v.flags) and MIXED in _strs(v.caps)
    assert _rank(v.confidence) >= _rank(Confidence.MEDIUM)
    # episodes.csv: the checks of each episode, which one is the witness the status read
    rows = {row['episode_id']: row for row in analysis.episode_rows()}
    assert rows[genuine.episode_id]['witness'] and rows[genuine.episode_id]['deciding_witness']
    assert not rows[shuffled.episode_id]['witness'] and not rows[shuffled.episode_id]['deciding_witness']
    assert rows[shuffled.episode_id]['check_a_response'] == 'fail'
    # docs/output.md: the sources in mmr_numbers.csv, the counts the ledger compared also in summary.csv
    numbers, summary = analysis.numbers_row(), analysis.summary_row()
    assert numbers['rsp_force_source'] == 'witness' and numbers['rsp_witness_episode_id'] == genuine.episode_id
    assert summary['rsp_n_witnesses'] == 1 and summary['rsp_n_episode_force_fail'] == 1


def test_control_without_the_shuffle_has_no_conflict_cap():
    _, control = analysed('genuine_genuine_control')
    v = control.verdict
    assert v.status == Status.TRANSIENT
    assert MIXED not in _strs(v.flags) and MIXED not in _strs(v.caps)
    assert _rank(v.confidence) < _rank(Confidence.MEDIUM)  # the medium of the mixed record comes from the cap


def test_shuffled_episode_with_more_turning_points_never_becomes_the_witness():
    _, analysis = analysed('shuffled_long_genuine_short')
    by_start = sorted(analysis.response.episode_responses, key=lambda e: e.t_start_yr)
    shuffled, genuine = by_start
    assert shuffled.n_tp > genuine.n_tp
    assert not shuffled.witness and genuine.witness
    cols = analysis.response.to_columns()
    assert cols['rsp_witness_episode_id'] == genuine.episode_id and cols['rsp_force_source'] == 'witness'
    assert analysis.response.a_response == PASS
    assert MIXED in _strs(analysis.verdict.flags)


def test_g1e_witness_before_the_jump_with_conflict():
    _, analysis = analysed('g1e_mixed')
    assert len(analysis.geometry.jumps) >= 1 and not analysis.geometry.episodes  # nothing librates after the jump
    responses = sorted(analysis.response.episode_responses, key=lambda e: e.t_start_yr)
    assert all(e.before_jump for e in responses) and len(responses) == 2
    assert responses[0].witness and responses[1].a_response == FAIL
    cols = analysis.response.to_columns()
    assert cols['rsp_force_source'] == 'witness' and cols['rsp_witness_episode_id'] == responses[0].episode_id
    v = analysis.verdict
    assert v.status == Status.TRANSIENT
    assert {MIXED, 'episode_before_jump'} <= _strs(v.caps)
    assert _rank(v.confidence) >= _rank(Confidence.MEDIUM)
    assert v.checks['check_mmr_potential'] == NM  # the energy is not measured per episode


@pytest.mark.slow
def test_g1e_without_witness_reads_the_pool_before_the_jump_functionally():
    _, analysis = analysed('g1e_no_witness')
    jump = analysis.geometry.jumps[-1].time_yr
    pool = analysis.response.before_jump_response
    assert pool is not None and pool.episode_id is None and pool.before_jump
    assert pool.t_end_yr <= jump + 2 * P0_YR
    assert pool.a_response == FAIL
    d = analysis.response
    assert (d.force_source, d.identity_source, d.b1_source) == (ResponseSource.BEFORE_JUMP,) * 3
    assert d.a_response == FAIL and d.force_r2_h1 == pool.force_r2_h1 and d.force_r2_h2 == pool.force_r2_h2
    assert d.witness_episode_id is None
    cols = d.to_columns()
    assert cols['rsp_n_witnesses'] == 0 and cols['rsp_n_episode_force_fail'] == 2
    v = analysis.verdict
    # L4 fails and R0 is not measured under G1e: step 5a
    assert v.status == Status.UNCERTAIN and v.reason == Reason.ORIGIN_UNKNOWN
    assert v.checks['check_mmr_a_response'] == FAIL
    assert v.checks['check_mmr_slow_libration'] in (NM, NA)
    assert MIXED not in _strs(v.flags)


def _facts(analysis):
    row = {**analysis.summary_row(), **analysis.numbers_row()}
    return MMRFacts(analysis, SETTINGS, row, analysis.episode_rows())


def test_right_column_names_where_the_force_law_was_measured():
    _, two = analysed('two_genuine')
    witness_id = two.response.to_columns()['rsp_witness_episode_id']
    text = measured_on(_facts(two))
    assert f'witness #{witness_id}' in text and 'of 2 witnesses' in text
    _, mixed = analysed('genuine_shuffled')
    text = measured_on(_facts(mixed))
    assert f"witness #{mixed.response.witness_episode_id}" in text and 'witnesses with most' not in text
    _, pool = analysed('g1e_no_witness')
    assert 'pooled over the 2 episodes before the jump' in measured_on(_facts(pool))


def test_ledger_records_the_witness_and_the_conflict():
    _, mixed = analysed('genuine_shuffled')
    names = [entry.name for entry in mixed.verdict.decision_path]
    assert any(name.startswith('R witness episodes') for name in names)
    assert any(name.startswith('R no other episode whose force law fails') for name in names)
    _, pool = analysed('g1e_no_witness')
    names = [entry.name for entry in pool.verdict.decision_path]
    assert not any(name.startswith('R no other episode') for name in names)  # only with a witness
