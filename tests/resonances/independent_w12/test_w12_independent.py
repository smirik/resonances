"""Independent functional tests of W12: the response cutoff at 0.1 P_cyc and the closure of a libration cycle.

Expected values are derived here, not read from the implementation.

Response cutoff (docs/libration-analysis.md §2, `MMR_RESPONSE_CUTOFF_CYCLES` = 0.1): the derivatives of the force law
(L4) are read after a second smoothing at P_r = max(P_c, 0.1 P_cyc).

- A 2J-1 pendulum sigma_ddot = -w0^2 sin(sigma - pi), w0 = 2 pi / 600 yr, half-amplitude 90 deg, has the period
  T = (2/pi) K(m = 1/2) 600 = 708.2 yr. A forced line sigma += A_f sin(2 pi t / T_f) with T_f = 0.07 T = 49.6 yr is put
  on top, and a follows from Kepler's law with n = 2 n_J - sigma_dot, so a_dot = (2/3)(a/n) sigma_ddot.
- The pendulum part of a_dot is k w0^2 sin sigma, the forced one k A_f w_f^2 sin(w_f t): relative amplitude
  A_f (w_f / w0)^2 = 146.5 A_f. The filters (forward-backward Butterworth of order 2) pass it with the gain
  G(P) = 1/(1 + (P_cut/P)^4) each: at P_c = 1.5 P_syn = 17.8 yr G = 0.98; at 0.1 T = 70.8 yr G = 0.194, at the old
  0.05 T = 35.4 yr G = 0.794. The forced line is uncorrelated with sin sigma, hence
  R^2 = S / (S + x^2 / 2), S = <sin^2 sigma> over the pendulum, x the filtered relative amplitude.
  For A_f = 1 deg: R^2 = 0.85 at 0.1 (a strict L4 pass, >= 0.7) and 0.25 at 0.05 (a fail, < 0.4).
- The line rule (W20 CS6, `response.line_margin_cutoff`) sees the forced line at the old cutoff: in a_dot it has
  x = 146.5 A_f 0.98 x 0.794 = 1.99 of the force K (>= 0.5), so the cutoff rises to 1.5 T_f = 74.4 yr, where
  G = 0.165, x = 0.414 and R^2 = S / (S + x^2 / 2) = 0.886 (S = 0.669 from the 0.85 above). At 0.1 T the line keeps
  x = 0.486, at the rule's threshold (0.5): whether it fires there depends on the fitted K, and both outcomes pass.
- Phase-shuffled a (Fourier phases of a randomised, the spectrum kept) breaks the link between a and sigma: the force
  law must not pass strictly on any seed.

Cycle closure (`geometry.closure_ratios`, W13; the author made it mandatory on 2026-10-17): a full cycle
TP_i, TP_i+1, TP_i+2 is closed when the net shift |sigma(i+2) - sigma(i)| is smaller than the mean of its two
half-swings. A run of turning points that closes no cycle is a saw, not a libration.

- Circulation (+300 deg per kyr) with two reversals of 5 deg: half-swings 5, 300, 5 deg; both cycles have net shift
  295 deg against a mean half-swing of 152.5 deg (ratio 1.93): no closed cycle, so not a libration run.
- A circulation that turns back by 2 deg after a 20-deg swing (three TPs 0, 20, 18 deg): shift 18 against 11 (1.64).
- Libration about 72 deg with a slowly modulated half-amplitude 20-60 deg and unequal half-cycles (600/400 yr), a
  horseshoe about 180 deg (half-amplitude ~146 deg, quick turns), and a libration whose centre drifts by 80 deg over
  20 cycles (net shift 4 deg per cycle against 60-deg half-swings) close every cycle: they stay libration runs with
  all their turning points.
"""

import dataclasses
import math
from functools import lru_cache

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.data import const
from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import response as rsp
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

SETTINGS = AnalysisSettings()
PASS, FAIL = CheckValue.PASS, CheckValue.FAIL

# ------------------------------------------------------------------------------------------- response cutoff

N_J = math.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'])
W0 = 2 * np.pi / 600.0
PERIOD_YR = (2 / np.pi) * 1.854075 * 600.0  # K(m = 1/2) = 1.854075: 708.2 yr
FORCED_PERIOD_YR = 0.07 * PERIOD_YR
FORCED_AMP_RAD = math.radians(1.0)
STEP_YR = 0.5
SPAN_YR = 10000.0
P_SYN_YR = 2 * np.pi / N_J  # 2:1 with Jupiter: n - n_J = n_J
# MMR_FILTER_SYNODIC_FACTOR x P_syn, held as the record's cutoff: the floor bound 150 / 2 = 75 yr would itself damp the
# 49.6-yr forced line (G = 0.16) and leave the response cutoff nothing to do; these tests are about the latter
CUTOFF_YR = 1.5 * P_SYN_YR


def gain(period_yr, cutoff_yr):
    return 1.0 / (1.0 + (cutoff_yr / period_yr) ** 4)


@lru_cache(maxsize=None)
def pendulum(forced):
    """(t, sigma, sigma_dot) of the 90-deg pendulum about pi, with or without the forced line."""
    t = np.arange(0.0, SPAN_YR + STEP_YR / 2, STEP_YR)
    sol = solve_ivp(
        lambda _, y: [y[1], -(W0**2) * np.sin(y[0] - np.pi)],
        (0.0, SPAN_YR),
        [np.pi, W0 * 2 * math.sin(math.radians(90.0) / 2)],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        max_step=STEP_YR,
    )
    sigma, rate = sol.y
    if forced:
        wf = 2 * np.pi / FORCED_PERIOD_YR
        sigma = sigma + FORCED_AMP_RAD * np.sin(wf * t)
        rate = rate + FORCED_AMP_RAD * wf * np.cos(wf * t)
    return t, sigma, rate


def kepler_a(rate):
    return (4 * np.pi**2 / (2 * N_J - rate) ** 2) ** (1.0 / 3.0)


def make_series(t, sigma, a):
    varpi = 0.7
    lam = 2 * (0.3 + N_J * t) - varpi - sigma  # sigma = 2 lambda_J - lambda - varpi
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
        regime=Regime(step_yr=STEP_YR, span_yr=float(t[-1]), n_rows=len(t), integrator='ias15'),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name='w12-independent',
    )


def predicted_r2(fraction):
    """R^2 = S / (S + x^2/2) of the single force law with the uncorrelated forced line (module docstring)."""
    _, sigma, _ = pendulum(False)
    s = float(np.mean(np.sin(sigma) ** 2))
    response_cutoff = max(CUTOFF_YR, fraction * PERIOD_YR)
    x = FORCED_AMP_RAD * (600.0 / FORCED_PERIOD_YR) ** 2 * gain(FORCED_PERIOD_YR, CUTOFF_YR)
    x *= gain(FORCED_PERIOD_YR, response_cutoff)
    return s / (s + x**2 / 2)


@lru_cache(maxsize=None)
def analysed(fraction, shuffle_seed=None, forced=True, line_rule=True):
    t, sigma, rate = pendulum(forced)
    a = kepler_a(rate)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        spectrum = np.fft.rfft(a - a.mean())
        phases = rng.uniform(0.0, 2 * np.pi, len(spectrum))
        phases[0] = phases[-1] = 0.0
        a = a.mean() + np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=len(a))
    settings = dataclasses.replace(SETTINGS, mmr_response_cutoff_cycles=fraction, mmr_filter_cutoff_yr=CUTOFF_YR)
    if not line_rule:  # no line of a_dot reaches 1e9 K: the rule of W20 CS6 never raises the cutoff
        settings = dataclasses.replace(settings, mmr_response_line_amp_frac=1e9)
    return analyse_mmr(make_series(t, sigma, a), settings)


def strict_pass(check, thin):
    return check == PASS and not thin


def judged_episodes(response):
    return [e for e in response.episode_responses if e.a_response is not None]


def test_default_response_cutoff_is_a_tenth_of_the_cycle():
    assert SETTINGS.mmr_response_cutoff_cycles == pytest.approx(0.1)
    assert rsp.response_cutoff_yr(17.8, 708.2, SETTINGS) == pytest.approx(70.82)
    assert rsp.response_cutoff_yr(100.0, 708.2, SETTINGS) == pytest.approx(100.0)  # P_c binds
    assert rsp.response_cutoff_yr(17.8, float('nan'), SETTINGS) == pytest.approx(17.8)  # no cycle: P_c


def test_predicted_r2_brackets_the_two_cutoffs():
    """The synthetic sits where the choice matters: strict pass at 0.1, fail at 0.05 (own numbers)."""
    assert predicted_r2(0.1) == pytest.approx(0.85, abs=0.02)
    assert predicted_r2(0.05) == pytest.approx(0.25, abs=0.02)


def test_exact_pendulum_passes_the_force_law_almost_perfectly():
    response = analysed(0.1, forced=False).response
    assert strict_pass(response.a_response, response.a_response_thin)
    assert min(response.force_r2_h1, response.force_r2_h2) > 0.99


def test_forced_line_below_the_response_cutoff_passes_the_force_law_strictly():
    analysis = analysed(0.1)
    response = analysis.response
    assert analysis.prep.cutoff.cutoff_yr == pytest.approx(CUTOFF_YR, rel=1e-3)
    assert analysis.geometry.period_cycle_yr == pytest.approx(PERIOD_YR, rel=0.03)  # TPs jitter by the forced line
    assert strict_pass(response.a_response, response.a_response_thin)
    for r2 in (response.force_r2_h1, response.force_r2_h2):
        assert r2 >= SETTINGS.mmr_force_r2_full
        assert r2 == pytest.approx(predicted_r2(0.1), abs=0.05)
    episodes = judged_episodes(response)
    assert episodes and all(strict_pass(e.a_response, e.a_response_thin) for e in episodes)


def test_same_record_fails_the_force_law_at_the_old_cutoff():
    response = analysed(0.05, line_rule=False).response
    assert response.a_response == FAIL
    for r2 in (response.force_r2_h1, response.force_r2_h2):
        assert r2 < SETTINGS.mmr_force_r2_fail
        assert r2 == pytest.approx(predicted_r2(0.05), abs=0.05)


def test_the_line_rule_lifts_the_old_cutoff_above_the_forced_line():
    response = analysed(0.05).response
    assert response.force_line_binding_yr == pytest.approx(FORCED_PERIOD_YR, rel=0.01)
    assert response.force_cutoff_yr == pytest.approx(1.5 * FORCED_PERIOD_YR, rel=0.01)
    assert response.a_response == PASS and response.force_r2_null_max < 0.3
    for r2 in (response.force_r2_h1, response.force_r2_h2):
        assert r2 == pytest.approx(0.886, abs=0.03)


@pytest.mark.parametrize('seed', range(5))
def test_phase_shuffled_a_does_not_pass_the_force_law_strictly(seed):
    response = analysed(0.1, shuffle_seed=seed).response
    assert not strict_pass(response.a_response, response.a_response_thin)
    assert not any(strict_pass(e.a_response, e.a_response_thin) for e in judged_episodes(response))


# ------------------------------------------------------------------------------------------- cycle closure

FINE_YR = 0.5


def knots(points):
    """sigma [rad] on a FINE_YR grid through (t [yr], sigma [deg]) knots, a half-cosine between each pair: zero rate at
    every knot, so the interior knots that change direction are the turning points and nothing else is one."""
    tk = np.array([p[0] for p in points], dtype=float)
    vk = np.radians([p[1] for p in points])
    t = np.arange(tk[0], tk[-1] + FINE_YR / 2, FINE_YR)
    x = np.empty_like(t)
    for i in range(len(tk) - 1):
        m = (t >= tk[i]) & (t <= tk[i + 1])
        u = (t[m] - tk[i]) / (tk[i + 1] - tk[i])
        x[m] = vk[i] + (vk[i + 1] - vk[i]) * (1 - np.cos(np.pi * u)) / 2
    return t, x


def episodes_of(t, x):
    tp = g.find_turning_points(t, x, math.radians(1.0), (t[0], t[-1]))
    return tp, g.find_episodes(t, tp, (t[0], t[-1]), SETTINGS)


def closure_ratios(values):
    """|sigma(i+2) - sigma(i)| / mean of the two half-swings, per full cycle (own definition)."""
    v = np.asarray(values, dtype=float)
    swings = np.abs(np.diff(v))
    return np.abs(v[2:] - v[:-2]) / ((swings[:-1] + swings[1:]) / 2)


REVERSALS = [(0, -600), (1000, 0), (1500, -5), (2000, 295), (2500, 290), (3500, 900)]
SAW = [(0, 600), (1000, 0), (1500, 20), (2000, 18), (3000, 600)]


def asymmetric_libration():
    points, t = [(0, 72)], 0
    for k in range(30):
        amp = 40 + 20 * math.sin(2 * math.pi * k / 30)
        t += 600 if k % 2 == 0 else 400
        points.append((t, 72 + (amp if k % 2 == 0 else -amp)))
    return knots(points)


def horseshoe():
    t = np.arange(0.0, 20000.0 + FINE_YR / 2, FINE_YR)
    return t, np.radians(180 + 156 * (2 / np.pi) * np.arcsin(0.995 * np.sin(2 * np.pi * t / 1000)))


def drifting_centre():
    t = np.arange(0.0, 20000.0 + FINE_YR / 2, FINE_YR)
    return t, np.radians(180 + 80 * t / 20000 + 60 * np.sin(2 * np.pi * t / 1000))


@pytest.mark.parametrize(
    'points, n_tp, ratio',
    [(REVERSALS, 4, 295 / 152.5), (SAW, 3, 18 / 11)],
    ids=['two-small-reversals', 'one-turn-back'],
)
def test_synthetic_saws_close_no_cycle(points, n_tp, ratio):
    tp, _ = episodes_of(*knots(points))
    assert len(tp) == n_tp
    ratios = closure_ratios(np.degrees(tp.values_rad))
    assert np.all(ratios > 1) and ratios.min() == pytest.approx(ratio, rel=1e-3)


@pytest.mark.parametrize('points', [REVERSALS, SAW], ids=['two-small-reversals', 'one-turn-back'])
def test_circulation_with_a_small_reversal_is_not_a_libration_run(points):
    _, runs = episodes_of(*knots(points))
    assert runs and not any(run.accepted for run in runs)


def test_circulation_with_three_reversals_is_not_a_libration_run():
    _, runs = episodes_of(*knots([(0, -600), (1000, 0), (1500, -20), (2000, 280), (2500, 260), (3000, 560), (3500, 540), (4500, 1200)]))
    assert runs and not any(run.accepted for run in runs)


@pytest.mark.parametrize(
    'build, n_tp',
    [(asymmetric_libration, 29), (horseshoe, 40), (drifting_centre, 40)],
    ids=['asymmetric', 'horseshoe', 'drifting-centre'],
)
def test_closed_librations_stay_libration_runs(build, n_tp):
    tp, runs = episodes_of(*build())
    assert len(tp) == n_tp
    assert np.all(closure_ratios(tp.values_rad) < 1)
    assert len(runs) == 1
    run = runs[0]
    assert run.accepted and len(run.tp_times_yr) == n_tp
    assert run.reject_reason == g.RejectReason.NONE
