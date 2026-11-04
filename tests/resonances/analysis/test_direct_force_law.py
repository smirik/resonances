"""W28 B8 (card g07-2 A): the direct averaged-force law at the libration scale (`response.direct_force_law`).

The synthetic record is a pendulum sigma_ddot = -w^2 sin sigma (scipy, not the package) with the identity
sigma_dot = c (a - a*), so a_dot = -(w^2 / c) sin sigma exactly: the theory profile coefficient x d<D>/dsigma is
-(w^2 / c) x sin sigma, and the true scale is s = 1. A lattice line (period 2100 yr, as the conjunction cycle 13 P_N
of 59358 8N-13) is added to a with 77 % of the variance of a_dot: the harmonic force law at the response cutoff
(0.1 P) then explains about 1 - 0.77 = 0.23 of a_dot, below MMR_FORCE_R2_FAIL. Low-passed at 0.35 P the line keeps
the gain 1 / (1 + (0.35 P / 2100)^4) ~ 0.1, so the direct law recovers s ~ 1. Independent closed forms checked: the
quarter-cycle shift (R^2 of sin against cos: 0), a prediction x4 (s = 1/4, outside [1/sqrt(10), sqrt(10)]).
The real records (59358 transient, GQ107 uncertain) are benchmark tests.
"""

from functools import lru_cache

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.mmr import decide, response as R, strength as strength_module
from resonances.resonance.analysis.mmr.models import DirectForceLaw, ResonanceSpec, Strength
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, LibrationOrigin, Status
from tests.resonances.analysis.builders import make, make_response, tree_verdict

S = settings_from_config()
STEP = 5.0
T = np.arange(0.0, 100000.0, STEP)
T0 = 8000.0  # small-amplitude period [yr]
W2 = (2 * np.pi / T0) ** 2
C = 10.0  # identity coefficient [rad / yr / au]
LINE_YR = 2100.0
LINE_SHARE = 0.77  # share of the a_dot variance carried by the line


@lru_cache(maxsize=None)
def _pendulum():
    """(sigma, a, period): the pendulum started at sigma = 0 with 0.75 of the separatrix energy (half-amplitude about
    120 deg) and a = a* + sigma_dot / c plus the lattice line."""
    v0 = 0.75 * 2 * np.sqrt(W2)
    sol = solve_ivp(lambda _, y: [y[1], -W2 * np.sin(y[0])], (0, T[-1]), [0.0, v0], t_eval=T, rtol=1e-10, atol=1e-12)
    sigma, rate = sol.y
    force = -W2 * np.sin(sigma) / C  # the exact a_dot
    # line amplitude B in a: B (2 pi / L) / sqrt(2) = sqrt(share / (1 - share)) x rms(force)
    amp = np.sqrt(LINE_SHARE / (1 - LINE_SHARE)) * np.std(force) * np.sqrt(2) * LINE_YR / (2 * np.pi)
    a = 40.0 + rate / C + amp * np.sin(2 * np.pi * T / LINE_YR)
    crossings = np.flatnonzero(np.diff(np.sign(rate)) != 0)
    return sigma, a, 2 * float(np.median(np.diff(T[crossings])))


def _profiles(coefficient=-W2 / C, shift=0.0):
    grid = 2 * np.pi * np.arange(720) / 720
    return [(T.mean(), (grid, coefficient, np.sin(grid - shift)))]


def _judge(sigma, a, period, profiles):
    cutoff = S.mmr_direct_force_cutoff_cycles * period
    blank = dict(period_yr=period, period_large_yr=period, survived=True, cutoff_yr=cutoff)
    windows = R.direct_windows(T, [(T[0], T[-1])], period, cutoff, S)
    return R.judge_direct(T, a, np.mod(sigma, 2 * np.pi), windows, profiles, blank, S)


def test_the_harmonic_force_law_fails_under_the_line():
    sigma, a, period = _pendulum()
    cutoff = S.mmr_response_cutoff_cycles * period
    angle_f = filtering.lowpass_with(T, sigma, cutoff, S)
    adot_f = filtering.time_derivative(T, filtering.lowpass_with(T, a, cutoff, S))
    fit = R.ForceFit(T, [(T[0], T[-1])], S.mmr_force_window_cycles * period, None, None, None, None, np.ptp(sigma) / 2, S)
    measured = fit.measure(angle_f, adot_f, [])
    assert measured.check[0] == CheckValue.FAIL
    assert measured.law_r2 < S.mmr_force_r2_fail


def test_the_direct_law_recovers_the_real_scale_under_the_line():
    sigma, a, period = _pendulum()
    law = _judge(sigma, a, period, _profiles())
    assert law.taken and law.reason == ''
    assert law.r2 >= S.mmr_force_r2_full
    assert law.scale_min == pytest.approx(1.0, abs=0.1) and law.scale_max == pytest.approx(1.0, abs=0.1)
    assert law.null_gain >= S.mmr_force_null_gain


def test_a_quarter_cycle_phase_shift_fails():
    sigma, a, period = _pendulum()
    law = _judge(sigma, a, period, _profiles(shift=np.pi / 2))  # d<D>/dsigma ~ -cos sigma: orthogonal to sin sigma
    assert not law.taken and law.reason == R.DIRECT_BELOW_FAIL
    assert law.r2 < S.mmr_force_r2_fail


def test_a_prediction_four_times_too_strong_falls_outside_the_band():
    sigma, a, period = _pendulum()
    law = _judge(sigma, a, period, _profiles(coefficient=4 * (-W2 / C)))
    assert law.scale_max == pytest.approx(0.25, abs=0.03)  # s = 1/4 < 1/sqrt(10) = 0.316
    assert not law.taken and law.reason == R.DIRECT_SCALE
    # x2 lies inside [1/sqrt(10), sqrt(10)]: taken (the band is the strength check's, not a model error)
    assert _judge(sigma, a, period, _profiles(coefficient=2 * (-W2 / C))).taken


def test_overlapping_windows_are_not_two_measurements():
    sigma, a, period = _pendulum()
    cutoff = S.mmr_direct_force_cutoff_cycles * period
    blank = dict(period_yr=period, period_large_yr=period, survived=True, cutoff_yr=cutoff)
    window = S.mmr_force_window_cycles * period
    short = [(10000.0, 10000.0 + 1.9 * window)]  # two windows half overlapping, none disjoint
    windows = R.direct_windows(T, short, period, cutoff, S)
    assert len(windows) == 2
    law = R.judge_direct(T, a, np.mod(sigma, 2 * np.pi), windows, _profiles(), blank, S)
    assert not law.taken and law.reason == R.DIRECT_NOT_INDEPENDENT


def test_period_large_reads_the_swing_through_a_faster_wiggle():
    # sigma = 100 deg sin(2 pi t / 8800) + 25 deg sin(2 pi t / 2100): the wiggle adds turns (73 in 90 kyr, period
    # 2.1 kyr unfiltered); started at the mixed 4350 yr (59358's P_e) the ladder reaches the swing, whose turns
    # survive the doubled cutoff (G = 1 / (1 + 0.7^4) = 0.81 of the 8.8-kyr cycle)
    t = np.arange(0.0, 100000.0, 1.0)
    slow, fast = 8800.0, 2100.0
    angle = np.radians(100) * np.sin(2 * np.pi * t / slow) + np.radians(25) * np.sin(2 * np.pi * t / fast)
    pieces = [(5000.0, 95000.0)]
    assert R._turn_periods(t, angle, np.radians(1.0), pieces)[0] == pytest.approx(fast, rel=0.05)
    period, survived, amplitude = R.period_large(t, angle, np.radians(1.0), pieces, 4350.0, S)
    assert period == pytest.approx(slow, rel=0.05) and survived
    assert np.degrees(amplitude) == pytest.approx(100.0, rel=0.15)
    # a clean libration keeps its own period at once
    clean = np.radians(100) * np.sin(2 * np.pi * t / slow)
    assert R.period_large(t, clean, np.radians(1.0), pieces, slow, S)[:2] == (pytest.approx(slow, rel=0.01), True)


def test_three_body_and_coorbital_angles_are_never_eligible():
    direct = make(Strength, method=strength_module.METHOD_DIRECT)
    assert R.direct_force_eligible(ResonanceSpec.from_key('4J-2S-1'), None, direct, S) == 'three_body_owner'
    assert R.direct_force_eligible(ResonanceSpec.from_key('1J-1+0+0'), None, direct, S) == 'coorbital_owner'
    import dataclasses

    off = dataclasses.replace(S, mmr_direct_force_law=False)
    assert R.direct_force_eligible(ResonanceSpec.from_key('8N-13+0+5'), None, direct, off) == 'off'
    scale = make(Strength, method=strength_module.METHOD_SCALE)
    assert R.direct_force_eligible(ResonanceSpec.from_key('8N-13+0+5'), None, scale, S) == R.DIRECT_NO_PLANET


def _law(taken=True, r2=0.75, survived=True):
    return DirectForceLaw(
        4353.5,
        8781.5,
        survived,
        3073.5,
        n_windows=4,
        r2=r2,
        scale_min=1.1,
        scale_max=1.3,
        null_max=0.02,
        small_period_large_yr=7000.0,
        mu_eff_large=2.0e-6,
        taken=taken,
        reason='' if taken else R.DIRECT_BELOW_FAIL,
    )


def test_a_taken_law_lifts_a_failed_l4_and_never_lowers():
    failed = make_response(a_response=CheckValue.FAIL, a_response_thin=False, force_law=R.LAW_SINGLE)
    lifted = R._lifted(failed, _law(), S)
    assert lifted.a_response == CheckValue.PASS and not lifted.a_response_thin and lifted.force_law == R.LAW_DIRECT
    assert R._lifted(failed, _law(r2=0.5), S).a_response_thin  # below MMR_FORCE_R2_FULL: thin
    refused = R._lifted(failed, _law(taken=False, r2=0.2), S)
    assert refused.a_response == CheckValue.FAIL and refused.force_direct is not None
    thin = make_response(a_response=CheckValue.PASS, a_response_thin=True, force_law=R.LAW_SINGLE)
    assert R._lifted(thin, _law(r2=0.5), S).force_law == R.LAW_SINGLE  # a thin direct law gains nothing over it
    assert R._lifted(thin, _law(r2=0.8), S).force_law == R.LAW_DIRECT
    assert not R._consults_direct(CheckValue.NOT_MEASURED, False) and not R._consults_direct(CheckValue.PASS, False)


def test_the_cap_force_law_theory_only_and_the_l4_margin():
    response = make_response(
        a_response=CheckValue.PASS, a_response_thin=False, force_law=R.LAW_DIRECT, force_direct=_law(), force_r2_h1=0.16, force_r2_h2=0.17
    )
    assert decide.taken_direct_law(response).r2 == 0.75
    decision = make(decide.StatusDecision, status=Status.TRANSIENT, origin=LibrationOrigin.DYNAMIC, rule='5d/6')
    margin = decide._l4(response, decision)
    assert margin.margin == Margin.FULL and margin.value == 0.75
    assert decide.theory_only_caps(decision, {'L4': margin}, response) == [str(Cap.FORCE_LAW_THEORY_ONLY)]
    harmonic = make_response(a_response=CheckValue.PASS, a_response_thin=False, force_law=R.LAW_SINGLE, force_direct=_law())
    assert decide.taken_direct_law(harmonic) is None
    assert decide.theory_only_caps(decision, {'L4': decide._l4(harmonic, decision)}, harmonic) == []
    columns = R.direct_force_columns(_law())
    assert columns['rsp_dforce_reason'] == 'taken' and columns['geo_period_large_yr'] == 8781.5
    assert np.isnan(R.direct_force_columns(_law(survived=False))['geo_period_large_yr'])


def test_f_reads_the_strength_of_the_large_turns():
    # 59358's numbers: theory 1.2e-6..1.3e-6; the episode needs 5.9 x, the large turns 2.0e-6 / 1.25e-6 ~ 1.6
    strength = make(
        Strength,
        method=strength_module.METHOD_DIRECT,
        mu_eff_theory_min=1.2e-6,
        mu_eff_theory_max=1.3e-6,
        mu_eff_required=7.4e-6,
        ratio=5.9,
        check=CheckValue.PASS,
        not_measured_reason='',
    )
    read = strength_module.at_large_period(strength, _law(), S)
    assert read.ratio == pytest.approx(2.0e-6 / 1.3e-6) and read.mu_eff_required == 2.0e-6
    assert read.check == CheckValue.PASS and read.t_obs_yr == 7000.0 and read.t_obs_source == 'period_large'
    assert strength_module.at_large_period(strength, _law(taken=False), S) is strength
    assert strength_module.at_large_period(strength, _law(survived=False), S) is strength


# --- real records (benchmark data; W28 B8 research numbers, not the package's output) -----------------------------
# 59358 8N-13 (author 2026-10-31, g07-2 A: transient/medium): L4 halves 0.163/0.168, the direct law at 0.35 x 8.8 kyr
# R^2 0.75, s 1.14-1.29, null 0.02 (Codex g07: 0.82/0.72 at a 3-kyr cutoff, s 1.25/1.24). 2005 GQ107 40M-57 (author
# 14.10, kept 31.10, g07-3 A: uncertain/episodic): its E01 (31.7-38.8 kyr) holds two overlapping windows only.


@lru_cache(maxsize=None)
def _analysis(example):
    from resonances.resonance.analysis.mmr import analyse as analyse_module, io

    return analyse_module.analyse_mmr(io.load_series(example), S)


@pytest.mark.benchmark
def test_59358_is_a_transient_whose_own_force_drives_a():
    analysis = _analysis('mmr2__59358__8N-13+0+5')
    verdict, columns = analysis.verdict, analysis.response.to_columns()
    # the W28 tree's verdict and cap (W34: the minimal policy keeps transient, the cap is a flag, the level its own)
    tree = tree_verdict(verdict)
    assert (tree.status, tree.angle_status, tree.confidence) == ('transient', 'episodic', 'medium')
    assert str(Cap.FORCE_LAW_THEORY_ONLY) in verdict.flags and 'direct_force_law' in verdict.rules_fired
    assert str(verdict.status) == 'transient' and verdict.caps == []
    assert columns['rsp_dforce_reason'] == 'taken' and columns['rsp_dforce_R2'] >= S.mmr_force_r2_full
    assert 1.0 < columns['rsp_dforce_scale_min'] and columns['rsp_dforce_scale_max'] < 1.4
    assert columns['geo_period_large_yr'] == pytest.approx(8700, rel=0.05)
    assert analysis.strength.ratio < 2.0  # F at the large turns (5.9 at the episode's 4.4 kyr)


@pytest.mark.benchmark
def test_gq107_stays_uncertain():
    analysis = _analysis('mmr2__2005_GQ107__40M-57+0+17')
    # the W28 tree's leaf 5g/origin_unknown; W34: the minimal policy gives the geometry's transient back with the
    # objection as a flag (measured_objection, low; the author's label of 2026-11-03)
    assert tree_verdict(analysis.verdict).status == 'uncertain'
    assert (str(analysis.verdict.status), str(analysis.verdict.confidence)) == ('transient', 'low')
    assert 'measured_objection' in analysis.verdict.flags
    e01 = next(e for e in analysis.response.episode_responses if e.episode_id is not None and 31000 < e.t_start_yr < 32500)
    assert e01.force_direct.reason == R.DIRECT_NOT_INDEPENDENT and e01.a_response == CheckValue.FAIL
