"""W28 B10a (cards g08-7, g08-8, g05-2, g05-1): the slower libration of the centre stored at its own scale, its
origin and its cause named with a level, and rule K3s (MMR_SLOW_SCALE_REPORT, MMR_SECULAR_CYCLE_KINEMATIC).

The synthetic records are built here from the linear secular solution (Murray & Dermott 1999, eqs. 7.18-7.19): the
body's eccentricity vector z = e_f exp(i g t) + e_F exp(i g5 t), varpi = arg z. A kinematic angle is sigma = pi +
p (varpi - g t) (the lambda part cancels p g on average): it swings with the period 2 pi / (g - g5) and half-amplitude
|p| asin(e_F / e_f), while a stays flat. A dynamic one is a slow pendulum whose a follows the identity
sigma_dot = c (a - a*), c = -1.5 k n / a*. The expected periods, turn times and swings are computed here with numpy
from these closed forms, not by the package. 37087 3J+8S-3 (benchmark): the author's g08-7 numbers (slow period
5.0-6.5 kyr, slow intervals about 17.2-62.4 and 65.8-96.9 kyr) and g08-8 A (non_resonant / kinematic_libration,
secular_cycle; level medium kept)."""

import math

import numpy as np
import pytest

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import analyse, decide, io, kinematics as kin, models
from resonances.resonance.analysis.mmr.models import CauseLevel, MMRSeries, Regime, ResonanceSpec, SlowOrigin, SlowScale
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue, Confidence, LibrationOrigin, OriginDetail, Reason, Status
from tests.resonances.analysis.builders import make, make_centre, make_episode, make_geometry, make_kinematics

S = settings_from_config()
TWO_PI = 2 * math.pi
ARCSEC = math.pi / (180 * 3600)
G_FREE = 231.0 * ARCSEC  # rad/yr, the free frequency of 37087's varpi
G5 = PLANETARY_FREQUENCIES['g5'] * ARCSEC
P_SECULAR = TWO_PI / (G_FREE - G5)  # ~ 5.7 kyr
KEY = '3J+8S-3+0+0-8'  # p = -8, k = -3
A_STAR = 3.17  # au
N_STAR = TWO_PI * A_STAR**-1.5  # rad/yr
C = -1.5 * -3 * N_STAR / A_STAR  # identity coefficient [rad yr^-1 au^-1]
STEP, SPAN = 10.0, 60_000.0


def _varpi(t, e_free=0.05, e_forced=0.005):
    z = e_free * np.exp(1j * G_FREE * t) + e_forced * np.exp(1j * G5 * t)
    return np.unwrap(np.angle(z))


def _series(t, varpi, sigma):
    spec = ResonanceSpec.from_key(KEY)
    zeros = np.zeros_like(t)
    return MMRSeries(
        t_yr=t,
        a=A_STAR + zeros,
        e=0.05 + zeros,
        inc=0.02 + zeros,
        Omega=1.0 + zeros,
        omega=np.mod(varpi - 1.0, TWO_PI),
        varpi=np.mod(varpi, TWO_PI),
        angle_wrapped=np.mod(sigma, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=STEP, span_yr=float(t[-1] - t[0]), n_rows=len(t)),
        name='synthetic-slow',
    )


def _turns(t, sigma):
    """Times and values of the extrema of sigma (sign changes of its rate), first and last 2 kyr left out."""
    rate = np.gradient(sigma, t)
    rising = rate >= 0
    idx = np.where(rising[1:] != rising[:-1])[0]
    idx = idx[(t[idx] > 2000.0) & (t[idx] < t[-1] - 2000.0)]
    return t[idx], sigma[idx]


def _track(t, sigma, a, turns, values):
    return kin.SlowTrack(
        source='centre_scale',
        cutoff_yr=500.0,
        t_yr=t,
        rate_rad_yr=np.gradient(sigma, t),
        a_au=a,
        spans=((float(turns[0]), float(turns[-1])),),
        turn_times=(turns,),
        turn_values=(values,),
        identity_coefficient=C,
    )


def _measured(sigma, varpi, a, centre_corr):
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    series = _series(t, varpi, sigma)
    prep = make(models.Preprocessed, t_yr=t, a_f=a, angle_f=sigma)
    turns, values = _turns(t, sigma)
    track = _track(t, sigma, a, turns, values)
    return kin.measure_slow_scale(series, prep, make_kinematics(centre_corr_rate=centre_corr), track, S), turns, values


def _noise(t, scale=1e-7):
    return scale * np.random.default_rng(37087).standard_normal(len(t))


def _kinematic():
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    varpi = _varpi(t)
    sigma = math.pi - 8 * (varpi - G_FREE * t)
    rate_sigma, rate_pvarpi = np.gradient(sigma, t), -8 * np.gradient(varpi, t)
    corr = float(np.corrcoef(rate_sigma, rate_pvarpi)[0, 1])  # 1 by construction: sigma_dot - p varpi_dot = const
    return _measured(sigma, varpi, A_STAR + _noise(t), corr)


def _circulation(rule='C'):
    return decide.StatusDecision(Status.NON_RESONANT, Reason.CIRCULATION, LibrationOrigin.NONE, '', '', rule=rule)


def _slipping_geometry(max_pass_rev=1.05):
    return make_geometry(centre=make_centre(max_pass_rad=max_pass_rev * TWO_PI))


# --------------------------------------------------------------------------------------------- closed forms


def test_predicted_swing_is_p_times_the_wobble_of_varpi():
    # e_f = 2 e_F: the smaller vector seen from the tip of the larger at most asin(1/2) = 30 deg; 4 varpi: 120 deg
    assert np.degrees(kin.predicted_swing_rad(-4, 0.06, 0.03)) == pytest.approx(120.0)
    assert np.degrees(kin.predicted_swing_rad(4, 0.03, 0.06)) == pytest.approx(120.0)  # the branch does not matter
    assert np.isnan(kin.predicted_swing_rad(0, 0.06, 0.03))
    assert np.isnan(kin.predicted_swing_rad(3, float('nan'), 0.03))


def test_turn_cycles_one_cycle_has_no_period():
    # three turns: one closed cycle, one estimate of the period (nothing to compare): nan, n = 1
    cycles, period, half = kin.turn_cycles([np.array([0.0, 2.0, 4.0])], [np.array([1.0, -1.0, 1.0])], 2)
    assert (cycles, np.isnan(period), half) == (1, True, 1.0)
    # two runs of five turns (2 + 2 cycles), full periods 4 and 6 yr pooled: median 5
    times = [np.arange(5) * 2.0, 100 + np.arange(5) * 3.0]
    values = [np.array([1, -1, 1, -1, 1.0]), np.array([2, -2, 2, -2, 2.0])]
    cycles, period, half = kin.turn_cycles(times, values, 2)
    assert (cycles, period, half) == (4, 5.0, 1.5)


def test_slower_than_the_main_libration():
    track = _track(np.arange(10.0), np.zeros(10), np.zeros(10), np.array([0.0, 181.0, 362.0, 543.0]), np.zeros(4))
    assert not kin.slower_than(track, 362.0, S)  # 6489 3J-1: the centre-scale run is the libration again
    assert kin.slower_than(track, 100.0, S)  # 362 > 3 x 100
    assert kin.slower_than(track, float('nan'), S)  # no main libration (37087: forced waves only)
    assert not kin.slower_than(None, float('nan'), S)


# --------------------------------------------------------------------------------------------- the slow scale


def test_kinematic_secular_cycle_is_identified():
    slow, turns, values = _kinematic()
    expected_period = float(np.median(turns[2:] - turns[:-2]))
    assert expected_period == pytest.approx(P_SECULAR, rel=0.01)  # the closed form: the e-cycle 2 pi / (g - g5)
    assert slow.period_yr == pytest.approx(expected_period)
    assert slow.n_cycles == (len(turns) - 1) // 2
    # half-amplitude |p| asin(e_F / e_f) = 8 asin(0.1) = 45.9 deg
    assert np.degrees(slow.halfamp_rad) == pytest.approx(8 * math.degrees(math.asin(0.1)), rel=0.02)
    assert slow.identity == CheckValue.FAIL  # a flat: sigma_dot does not follow a
    assert slow.secular_term == 'g-g5'  # g - g7 is 0.5 % away too; c_dot follows g - g5
    assert slow.secular_offset < S.mmr_kin_period_tol
    assert (slow.origin, slow.cause, slow.cause_level) == (SlowOrigin.KINEMATIC, kin.CAUSE_G_GJ, CauseLevel.IDENTIFIED)


def test_true_slow_pendulum_at_a_secular_period_is_a_dynamic_candidate():
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    sigma = math.pi + 0.8 * np.sin(TWO_PI * t / P_SECULAR)
    a = A_STAR + np.gradient(sigma, t) / C  # the identity holds exactly
    slow, _, _ = _measured(sigma, _varpi(t), a, 0.1)
    assert slow.identity == CheckValue.PASS
    assert slow.origin == SlowOrigin.DYNAMIC_CANDIDATE
    decision = _circulation()
    decide.secular_cycle_kinematic(decision, slow, _slipping_geometry(), S, Ledger.from_settings(S))
    assert (decision.reason, decision.origin) == (Reason.CIRCULATION, LibrationOrigin.NONE)  # K3s needs a failed identity


def test_unrelated_period_without_fit_is_unknown():
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    varpi = G_FREE * t  # one vector: no e-cycle wobble; omega = varpi - Omega turns at g: 2 omega at pi / g
    sigma = math.pi + 0.8 * np.sin(TWO_PI * t / 4000.0)
    slow, _, _ = _measured(sigma, varpi, A_STAR + _noise(t), 0.1)
    assert slow.period_yr == pytest.approx(4000.0, rel=0.01)
    assert slow.secular_offset > S.mmr_kin_period_tol  # 4000 / 5709 - 1 = -0.30; pi / g = 2.8 kyr: 0.42
    assert (slow.cause, slow.cause_level) == ('', CauseLevel.UNKNOWN)
    assert slow.origin == SlowOrigin.UNRESOLVED  # identity fails, no secular period


def test_one_slow_cycle_has_no_period_and_one_cycle():
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    sigma = math.pi + 0.8 * np.sin(TWO_PI * t / P_SECULAR)
    series = _series(t, _varpi(t), sigma)
    prep = make(models.Preprocessed, t_yr=t, a_f=A_STAR + _noise(t), angle_f=sigma)
    turns, values = _turns(t, sigma)
    track = _track(t, sigma, A_STAR + _noise(t), turns[:3], values[:3])
    slow = kin.measure_slow_scale(series, prep, make_kinematics(centre_corr_rate=0.1), track, S)
    assert (slow.n_cycles, np.isnan(slow.period_yr), np.isnan(slow.b1_ratio)) == (1, True, True)


def test_owner_row_names_the_cause_of_its_owner_x_first():
    slow, _, _ = _kinematic()
    t = np.arange(0.0, SPAN + STEP / 2, STEP)
    varpi = _varpi(t)
    sigma = math.pi - 8 * (varpi - G_FREE * t)
    turns, values = _turns(t, sigma)
    owner_x = models.SlowX(period_yr=7200.0, term='2omega', cause=kin.CAUSE_2OMEGA, level='possible')
    prep = make(models.Preprocessed, t_yr=t, a_f=A_STAR + _noise(t), angle_f=sigma)
    track = _track(t, sigma, A_STAR + _noise(t), turns, values)
    named = kin.measure_slow_scale(_series(t, varpi, sigma), prep, make_kinematics(centre_corr_rate=1.0), track, S, owner_x)
    assert (named.cause, named.cause_term, named.cause_level, named.cause_source) == (kin.CAUSE_2OMEGA, '2omega', 'possible', 'owner_x')
    assert named.period_yr == slow.period_yr  # the period stays the centre's own


# --------------------------------------------------------------------------------------------- K3s


def test_k3s_makes_a_circulation_a_kinematic_libration():
    slow, _, _ = _kinematic()
    decision = _circulation()
    ledger = Ledger.from_settings(S)
    decide.secular_cycle_kinematic(decision, slow, _slipping_geometry(), S, ledger)
    assert decision.status == Status.NON_RESONANT
    assert (decision.reason, decision.origin) == (Reason.KINEMATIC_LIBRATION, LibrationOrigin.KINEMATIC)
    assert (decision.origin_detail, decision.rule) == (OriginDetail.SECULAR_CYCLE, decide.RULE_K3S)
    assert str(Rule.SECULAR_CYCLE_KINEMATIC) in decision.rules_fired
    assert any(entry.name == decide.K3S_OFFSET for entry in ledger)


@pytest.mark.parametrize(
    'decision, geometry, settings',
    [
        # a libration is never demoted: the slow kinematic component of a resonant row is reported only
        (decide.StatusDecision(Status.RESONANT, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC, '', ''), None, S),
        # an induced row keeps its owner
        (decide.StatusDecision(Status.NON_RESONANT, Reason.INDUCED_LIBRATION, LibrationOrigin.INDUCED, '', 'x'), None, S),
        # the centre circulates 4.7 turns in a row (285637 3M+7J-6 2 Myr): a circulation with slow interludes
        (_circulation(), _slipping_geometry(4.7), S),
        # the switch off
        (_circulation(), None, S.__class__(**{**S.__dict__, 'mmr_secular_cycle_kinematic': False})),
    ],
)
def test_k3s_leaves_other_rows(decision, geometry, settings):
    slow, _, _ = _kinematic()
    before = (decision.status, decision.reason, decision.origin, decision.rule)
    decide.secular_cycle_kinematic(decision, slow, geometry or _slipping_geometry(), settings, Ledger.from_settings(settings))
    assert (decision.status, decision.reason, decision.origin, decision.rule) == before


def test_k3s_is_graded_as_the_circulation_it_read():
    # no K1/K1e/B5 margin and no K1e in the physics list: the level stays the circulation's (37087 medium)
    decision = _circulation(rule=decide.RULE_K3S)
    decision.reason, decision.origin = Reason.KINEMATIC_LIBRATION, LibrationOrigin.KINEMATIC
    angle = make(decide.AngleDecision, angle_status=AngleStatus.CIRCULATION)
    margins = decide._origin_margins(angle, decision, None, make_kinematics(corr_rate=1.0), Ledger.from_settings(S), S)
    assert margins == {}
    assert decide.physics_margins_of(decision, ('L4',)) == ('L4',)


# --------------------------------------------------------------------------------------------- episodes.csv


def test_slow_runs_are_episode_rows_of_scale_slow_only_on_request():
    main = make_episode(5, episode_id=1, tp_times_yr=np.arange(5) * 100.0)
    slow_run = make_episode(5, episode_id=1, tp_times_yr=17_000 + np.arange(5) * 2800.0)
    analysis = make(
        models.MMRAnalysis,
        resonance=ResonanceSpec.from_key(KEY),
        geometry=make_geometry(runs=[main], window=(0.0, 20_000.0)),
        episode_physics=[],
        slow_scale=SlowScale(source='centre_scale', cutoff_yr=500.0, episodes=(slow_run,), t_start_yr=17_000.0, t_end_yr=28_200.0),
    )
    assert [row['scale'] for row in analysis.episode_rows()] == ['main']
    rows = analysis.episode_rows(include_slow=True)
    assert [row['scale'] for row in rows] == ['main', 'slow']
    assert (rows[1]['t_start_yr'], rows[1]['cutoff_yr']) == (17_000.0, 500.0)


# --------------------------------------------------------------------------------------------- benchmark records


def _analysed(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    return analyse.analyse_mmr(io.load_series(example), S)


@pytest.mark.benchmark
def test_37087_slow_scale_and_k3s():
    analysis = _analysed('mmr3__37087__3J+8S-3+0+0-8')
    verdict, numbers = analysis.verdict, analysis.numbers_row()
    assert (verdict.status, verdict.confidence) == (Status.NON_RESONANT, Confidence.MEDIUM)
    assert (verdict.reason, verdict.origin_detail) == (Reason.KINEMATIC_LIBRATION, OriginDetail.SECULAR_CYCLE)
    assert 5000 <= numbers['slow_period_yr'] <= 6500  # g08-7: 5.0-6.5 kyr
    assert (numbers['slow_secular_term'], numbers['slow_cause_level']) == ('g-g5', 'identified')
    slow = [(row['t_start_yr'], row['t_end_yr']) for row in analysis.slow_episode_rows() if row['accepted']]
    near = [(pytest.approx(t0, abs=2000), pytest.approx(t1, abs=2000)) for t0, t1 in ((17_200, 62_400), (65_800, 96_900))]
    assert slow == near


@pytest.mark.benchmark
def test_285637_two_myr_stays_a_circulation():
    verdict = _analysed('mmr3__285637__3M+7J-6+0+0-4__2Myr').verdict
    assert (verdict.status, verdict.reason, verdict.confidence) == (Status.NON_RESONANT, Reason.CIRCULATION, Confidence.HIGH)
