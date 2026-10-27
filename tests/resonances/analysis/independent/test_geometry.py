"""Independent tests of the shared geometry, the MMR preprocessing and the angle tree (decide_angle).

Written from tmp/classifier-v2-2026-10-15/impl/API.md, SPEC §4.3, §4.4, §5.2, DELTA item 1 and
definitions.md only (no implementation bodies read). Every expected value comes from a closed form
(pendulum: F(A) = 2K(sin^2(A/2))/pi, r_pend(h), Butterworth gain), from a synthetic series whose truth
is constructed here (pendulum integrated with scipy, sinusoids with known drifts, a 2:1 libration
built from the identity), or from a hand calculation written in a comment.
"""

import dataclasses
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ellipk

from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.geometry import (
    EpisodeShape,
    FilteredAngle,
    Jump,
    JumpCandidate,
    RejectReason,
    Revolution,
    TurningPoints,
)
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, preprocess
from resonances.resonance.analysis.mmr.context import build_context
from resonances.resonance.analysis.mmr.models import (
    GreatInequality,
    MMRSeries,
    PlanetSeries,
    Regime,
    ResonanceKind,
    ResonanceSpec,
    Sampling,
    StepVerdict,
)
from resonances.resonance.analysis.mmr.sampling import analyse_sampling
from resonances.resonance.analysis.pendulum import EnergyAnalysis, Potential, Zone, ZoneMethod
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue

TWO_PI = 2.0 * np.pi
DEG = np.pi / 180.0
NAN = float('nan')


@pytest.fixture(scope='module')
def settings():
    return settings_from_config()


# ----------------------------------------------------------------------------------------------
# independent references
# ----------------------------------------------------------------------------------------------


def gain(period, cutoff, order=2):
    """Forward-backward Butterworth amplitude gain 1 / (1 + (P_c/P)^(2N)) (analytic, DELTA item 1)."""
    return 1.0 / (1.0 + (cutoff / period) ** (2 * order))


def amplitude_factor_ref(amplitude_rad):
    """F(A) = (2/pi) K(m = sin^2(A/2)), Murray & Dermott eqs. 8.50-8.52."""
    return 2.0 / np.pi * ellipk(np.sin(amplitude_rad / 2.0) ** 2)


def r_pend_ref(h):
    """min/mean rate of a pendulum rotation at energy h: (2/pi) sqrt((h-1)/h) K(m = 1/h) (SPEC §4.3.5)."""
    return 2.0 / np.pi * np.sqrt((h - 1.0) / h) * ellipk(1.0 / h)


def pendulum(omega0, sigma0, rate0, t):
    """sigma_ddot = -omega0^2 sin(sigma), DOP853 at rtol 1e-11 (the truth of the pendulum tests)."""
    sol = solve_ivp(
        lambda _, y: [y[1], -(omega0**2) * np.sin(y[0])],
        (t[0], t[-1]),
        [sigma0, rate0],
        t_eval=t,
        method='DOP853',
        rtol=1e-11,
        atol=1e-12,
    )
    assert sol.success
    return sol.y[0], sol.y[1]


def make_angle(t, sigma, rate=None, cutoff=50.0, window=None, c=-0.5, slow=None):
    """A FilteredAngle whose filtered versions are the given (already smooth) truth."""
    if rate is None:
        rate = np.gradient(sigma, t)
    if slow is None:
        slow = 2.5 + rate / c  # identity sigma_dot = c (a - a*)
    if window is None:
        window = (t[0] + cutoff, t[-1] - cutoff)
    return FilteredAngle(
        t_yr=t,
        angle_raw=sigma,
        angle_f=sigma,
        rate_f=rate,
        angle_f_half=sigma,
        angle_f_double=sigma,
        slow_f=slow,
        identity_coefficient=c,
        cutoff_yr=cutoff,
        step_yr=float(np.median(np.diff(t))),
        window=window,
    )


def make_tp(times, values, prominence=0.01):
    values = np.asarray(values, dtype=float)
    # alternating sequence: a TP above its neighbour is a maximum
    neighbour = np.r_[values[1:2], values[:-1]]
    kinds = np.where(values > neighbour, 1, -1)
    return TurningPoints(
        indices=np.arange(len(values)),
        times_yr=np.asarray(times, dtype=float),
        values_rad=values,
        kinds=kinds,
        prominence_rad=prominence,
    )


def make_episode(times, accepted, reject, shape=EpisodeShape.NONE):
    times = np.asarray(times, dtype=float)
    return geo.Episode(
        episode_id=0,
        tp_times_yr=times,
        tp_values_rad=np.zeros_like(times),
        accepted=accepted,
        shape=shape,
        reject_reason=reject,
        period_med_yr=NAN,
        period_min_yr=NAN,
        period_max_yr=NAN,
        half_amp_min_rad=NAN,
        half_amp_med_rad=NAN,
        half_amp_max_rad=NAN,
        centre_mod_2pi_rad=NAN,
        centre_shift_rad=NAN,
        centre_shift_over_amp=NAN,
        bands_overlap=False,
        centre_returns=0,
        coverage_start_gap_p=NAN,
        coverage_end_gap_p=NAN,
    )


def make_sampling(n=1000, step=1.0, **over):
    values = dict(
        step_yr=step,
        span_yr=step * (n - 1),
        n_rows=n,
        nyquist_rad_yr=np.pi / step,
        step_rotation_min_rad=0.0,
        step_rotation_max_rad=0.01,
        branches=(0,),
        verdict=StepVerdict.RESOLVED,
        branch_margin_turns=0.45,
        nyquist_boundary_da_au=0.1,
        next_branch_da_au=0.2,
        far_ratio=0.1,
        far_from_resonance=False,
        rate_apparent_pred_rad_yr=0.0,
        rate_apparent_meas_rad_yr=0.0,
        nyquist_halfwidth_au=0.1,
        ghost_axis_au=NAN,
        ghost_distance_au=NAN,
        angle_unwrapped=np.zeros(n),
        residual_rad=np.zeros(n - 1),
        unwrap_resid_p99_rad=0.0,
        unwrap_resid_max_rad=0.0,
        unwrap_ok=True,
        suspect_times_yr=(),
        revs_predictive=0.0,
        revs_numpy=0.0,
        alias_numpy_differs=False,
        jitter=0.0,
        jitter_low_e=False,
        undefined_phase=False,
        safe_step_plain_yr=10.0,
        safe_step_pred_yr=10.0,
        angle_reconstruction_rad=0.0,
        aliased_lines_yr=(),
        resample_step_yr=None,
    )
    values.update(over)
    return Sampling(**values)


def sine(t, period, amp, centre=0.0):
    return centre + amp * np.sin(TWO_PI * t / period)


# ----------------------------------------------------------------------------------------------
# small closed-form helpers
# ----------------------------------------------------------------------------------------------


def test_reference_values_of_the_spec():
    # The spec's numbers, checked against the closed forms used as truth below.
    assert amplitude_factor_ref(90 * DEG) == pytest.approx(1.180, abs=1e-3)
    assert amplitude_factor_ref(120 * DEG) == pytest.approx(1.373, abs=1e-3)
    assert amplitude_factor_ref(150 * DEG) == pytest.approx(1.762, abs=1e-3)
    assert r_pend_ref(1.14) == pytest.approx(0.554, abs=2e-3)
    assert r_pend_ref(1.012) == pytest.approx(0.25, abs=3e-3)
    assert gain(10.0, 1.0) == pytest.approx(0.9999, abs=1e-4)
    assert gain(1.0, 1.0) == pytest.approx(0.5)
    assert gain(1.0 / 1.5, 1.0) == pytest.approx(0.165, abs=1e-3)


def test_wrap_pi_half_open_interval_and_negative_angles():
    # [-pi, pi): +pi maps to -pi; 3pi/2 -> -pi/2; -3pi/2 -> pi/2; 7pi -> -pi
    x = np.array([np.pi, -np.pi, 1.5 * np.pi, -1.5 * np.pi, 7 * np.pi, 0.3, -0.3 - 4 * np.pi])
    w = geo.wrap_pi(x)
    np.testing.assert_allclose(w, [-np.pi, -np.pi, -0.5 * np.pi, 0.5 * np.pi, -np.pi, 0.3, -0.3], atol=1e-12)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)


def test_turning_prominence_is_max_of_floor_and_ripple_term():
    # p_TP = max(1 deg, 4 x r): r = 0.5 deg -> 2 deg; r = 0 -> 1 deg; r = 0.1 deg -> floor
    assert geo.turning_prominence(1 * DEG, 4.0, 0.5 * DEG) == pytest.approx(2 * DEG)
    assert geo.turning_prominence(1 * DEG, 4.0, 0.0) == pytest.approx(1 * DEG)
    assert geo.turning_prominence(1 * DEG, 4.0, 0.1 * DEG) == pytest.approx(1 * DEG)


# ----------------------------------------------------------------------------------------------
# turning points
# ----------------------------------------------------------------------------------------------


def test_turning_points_of_a_sine():
    # sigma = 0.5 sin(2 pi t / 10): maxima at 2.5 + 10 n, minima at 7.5 + 10 n.
    t = np.arange(0.0, 100.0 + 1e-9, 0.01)
    tp = geo.find_turning_points(t, sine(t, 10.0, 0.5), 0.01, (0.0, 100.0))
    assert len(tp) == 20
    np.testing.assert_allclose(tp.times_yr, 2.5 + 5.0 * np.arange(20), atol=0.011)
    np.testing.assert_allclose(np.abs(tp.values_rad), 0.5, atol=1e-6)
    assert tp.kinds[0] == 1
    assert np.all(np.diff(tp.kinds) != 0)
    # inside the window (20, 60): 22.5, 27.5, ..., 57.5 -> 8 TPs
    tpw = geo.find_turning_points(t, sine(t, 10.0, 0.5), 0.01, (20.0, 60.0))
    assert len(tpw) == 8
    assert tpw.times_yr[0] == pytest.approx(22.5, abs=0.011)
    assert tpw.times_yr[-1] == pytest.approx(57.5, abs=0.011)


def test_turning_points_ripple_removed_by_prominence():
    # A ripple of half-amplitude r makes extrema of prominence <= 2r (SPEC §4.3.1). With r = 0.06
    # rad at 0.5 yr the ripple slope 0.75 rad/yr exceeds the main slope 0.31 rad/yr near every
    # extremum, so a tiny prominence finds extra TPs; 4r = 0.24 rad keeps only the 20 main ones.
    t = np.arange(0.0, 100.0 + 1e-9, 0.005)
    sigma = sine(t, 10.0, 0.5) + 0.06 * np.sin(TWO_PI * t / 0.5 + 0.3)
    loose = geo.find_turning_points(t, sigma, 1e-4, (0.0, 100.0))
    strict = geo.find_turning_points(t, sigma, 4 * 0.06, (0.0, 100.0))
    assert len(loose) > 20
    assert len(strict) == 20
    assert np.all(np.diff(strict.kinds) != 0)


def test_turning_points_alternate_on_noisy_series():
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 400.0, 0.1)
    sigma = sine(t, 20.0, 1.0) + 0.05 * rng.standard_normal(t.size)
    tp = geo.find_turning_points(t, sigma, 0.3, (0.0, 400.0))
    assert np.all(np.diff(tp.kinds) != 0)
    # 400/20 = 20 cycles -> 40 extrema (+-1 at the edges)
    assert abs(len(tp) - 40) <= 1


@pytest.mark.parametrize('sigma_fn', [lambda t: 0.01 * t, lambda t: np.full_like(t, 2.0), lambda t: -0.3 * t])
def test_no_turning_points_on_monotone_or_constant(sigma_fn):
    t = np.arange(0.0, 200.0, 1.0)
    tp = geo.find_turning_points(t, sigma_fn(t), 1 * DEG, (0.0, 199.0))
    assert len(tp) == 0
    assert tp.times_yr.size == 0 and tp.values_rad.size == 0 and tp.kinds.size == 0


def test_turning_point_of_a_three_sample_series():
    t = np.array([0.0, 1.0, 2.0])
    tp = geo.find_turning_points(t, np.array([0.0, 1.0, 0.0]), 0.5, (0.0, 2.0))
    assert len(tp) == 1
    assert tp.times_yr[0] == 1.0 and tp.kinds[0] == 1


def test_periods_amplitudes_midpoints_by_hand():
    # values 1, -1, 1.2, -0.8, 1 at t = 0, 5, 10, 15, 20
    # full periods: 10-0, 15-5, 20-10 = 10, 10, 10
    # A_i = |diff|/2: 1, 1.1, 1.0, 0.9; m_i = mean of neighbours: 0, 0.1, 0.2, 0.1
    tp = make_tp([0, 5, 10, 15, 20], [1, -1, 1.2, -0.8, 1])
    np.testing.assert_allclose(geo.full_periods(tp), [10, 10, 10])
    np.testing.assert_allclose(geo.half_amplitudes(tp), [1.0, 1.1, 1.0, 0.9])
    np.testing.assert_allclose(geo.midpoints(tp), [0.0, 0.1, 0.2, 0.1], atol=1e-12)


def test_exactly_three_turning_points_give_one_full_period():
    tp = make_tp([0, 4, 9], [0.5, -0.5, 0.4])
    np.testing.assert_allclose(geo.full_periods(tp), [9.0])
    np.testing.assert_allclose(geo.half_amplitudes(tp), [0.5, 0.45])
    two = make_tp([0, 4], [0.5, -0.5])
    assert geo.full_periods(two).size == 0


# ----------------------------------------------------------------------------------------------
# centre and passes
# ----------------------------------------------------------------------------------------------


def test_monotone_passes_by_hand():
    # centre 0 -> 3 (t 0..10), 3 -> 1 (10..20), 1 -> 5 (20..30): passes +3, -2, +4
    t = np.arange(0.0, 30.0 + 1e-9, 0.5)
    c = np.interp(t, [0, 10, 20, 30], [0, 3, 1, 5])
    turns = TurningPoints(
        indices=np.array([20, 40]),
        times_yr=np.array([10.0, 20.0]),
        values_rad=np.array([3.0, 1.0]),
        kinds=np.array([1, -1]),
        prominence_rad=0.1,
    )
    passes = geo.monotone_passes(t, c, turns, (0.0, 30.0))
    assert [p.delta_rad for p in passes] == pytest.approx([3.0, -2.0, 4.0])
    assert [p.t_start_yr for p in passes] == pytest.approx([0.0, 10.0, 20.0])
    assert [p.t_end_yr for p in passes] == pytest.approx([10.0, 20.0, 30.0])
    assert geo.largest_monotone_pass(passes).delta_rad == pytest.approx(4.0)


def test_largest_pass_by_absolute_value_and_empty():
    passes = [geo.MonotonePass(0, 1, 1.0), geo.MonotonePass(1, 2, -7.0), geo.MonotonePass(2, 3, 3.0)]
    assert geo.largest_monotone_pass(passes).delta_rad == -7.0
    assert geo.largest_monotone_pass([]) is None


def test_centre_going_one_and_a_quarter_turns_and_back(settings):
    # WP2 truth: the centre c = 2.5 pi sin(pi t / T) goes +1.25 turns and back: the largest pass
    # is a circulation (>= 2pi) while the net is ~0. Window (300, 19700):
    # c(300) = 2.5 pi sin(0.0471) = 0.370, c(10000) = 7.854 -> largest pass 7.484 rad = 1.191 rev.
    t = np.arange(0.0, 20000.0 + 1e-9, 1.0)
    centre = 2.5 * np.pi * np.sin(np.pi * t / 20000.0)
    sigma = centre + sine(t, 100.0, 0.5)
    track = geo.centre_track(t, sigma, 100.0, 50.0, 1 * DEG, (300.0, 19700.0), settings)
    assert track.window_yr == pytest.approx(3 * 100.0)
    assert abs(track.max_pass_rad) == pytest.approx(7.484, abs=0.1)
    assert track.max_pass_rev == pytest.approx(7.484 / TWO_PI, abs=0.02)
    assert abs(track.net_rev) < 0.02
    assert track.range_rad == pytest.approx(7.484, abs=0.1)
    assert len(track.turns) == 1
    assert 1.1 < track.max_pass_rev_lo <= track.max_pass_rev_hi < 1.3


def test_centre_of_a_steady_libration_and_fallback_window(settings):
    t = np.arange(0.0, 10000.0 + 1e-9, 1.0)
    sigma = sine(t, 200.0, 1.0, centre=np.pi)
    track = geo.centre_track(t, sigma, 200.0, 50.0, 1 * DEG, (100.0, 9900.0), settings)
    # residual of the wiggle after a 600-yr smoothing: 1 x G(200; 600) = 1/82 rad
    inside = (track.t_yr >= 1000) & (track.t_yr <= 9000)
    np.testing.assert_allclose(track.centre_rad[inside], np.pi, atol=0.03)
    assert track.max_pass_rev < 0.1
    # a stationary centre: net below the 10-deg floor of a centre turn (0.02 rev = 7 deg)
    assert abs(track.net_rev) < 0.02
    # without a measured period: W_c = 3 P_c
    fallback = geo.centre_track(t, sigma, NAN, 50.0, 1 * DEG, (100.0, 9900.0), settings)
    assert fallback.window_yr == pytest.approx(150.0)


def test_centre_of_a_negative_circulation(settings):
    # sigma = -2pi t / 1000: centre = sigma; window (100, 9900) -> -9.8 turns
    t = np.arange(0.0, 10000.0 + 1e-9, 1.0)
    sigma = -TWO_PI * t / 1000.0
    track = geo.centre_track(t, sigma, NAN, 50.0, 1 * DEG, (100.0, 9900.0), settings)
    assert track.net_rev == pytest.approx(-9.8, abs=0.05)
    assert abs(track.max_pass_rad) == pytest.approx(9.8 * TWO_PI, rel=0.01)


# ----------------------------------------------------------------------------------------------
# runs and episodes
# ----------------------------------------------------------------------------------------------


def test_steady_libration_is_one_covering_stationary_episode(settings):
    # sigma = pi + 1 sin(2 pi t / 200), 20 kyr: TPs every 100 yr, A = 1, P = 200, centre pi.
    # gaps: first TP at 50 -> 0.25 P, last at 19950 -> 0.25 P (record 0..20000).
    t = np.arange(0.0, 20000.0 + 1e-9, 1.0)
    sigma = sine(t, 200.0, 1.0, centre=np.pi)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 20000.0))
    episodes = geo.find_episodes(t, tp, (0.0, 20000.0), settings)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.accepted and ep.shape == EpisodeShape.STATIONARY and ep.reject_reason == RejectReason.NONE
    assert ep.n_tp == 200
    assert ep.period_med_yr == pytest.approx(200.0, abs=1.0)
    assert ep.half_amp_med_rad == pytest.approx(1.0, abs=1e-3)
    assert geo.wrap_pi(ep.centre_mod_2pi_rad - np.pi) == pytest.approx(0.0, abs=1e-3)
    assert abs(ep.centre_shift_rad) < 1e-3
    assert ep.bands_overlap
    assert ep.coverage_start_gap_p == pytest.approx(0.25, abs=0.01)
    assert ep.coverage_end_gap_p == pytest.approx(0.25, abs=0.01)
    assert geo.covering_episode(episodes, settings) is ep


def test_centre_near_minus_pi_reported_mod_2pi(settings):
    # wrap-around: centre -pi + 0.1 -> mod 2pi = pi + 0.1
    t = np.arange(0.0, 5000.0 + 1e-9, 1.0)
    sigma = sine(t, 200.0, 0.8, centre=-np.pi + 0.1)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 5000.0))
    ep = geo.find_episodes(t, tp, (0.0, 5000.0), settings)[0]
    assert 0.0 <= ep.centre_mod_2pi_rad < TWO_PI
    assert ep.centre_mod_2pi_rad == pytest.approx(np.pi + 0.1, abs=1e-3)


def test_exactly_three_turning_points_make_an_episode(settings):
    # t in [0, 1500], P = 1000: extrema 250, 750, 1250 -> 3 TPs = one full cycle; gaps 0.25 P.
    t = np.arange(0.0, 1500.0 + 1e-9, 1.0)
    sigma = sine(t, 1000.0, 0.7)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 1500.0))
    assert len(tp) == 3
    runs = geo.split_runs(tp, settings)
    assert len(runs) == 1 and len(runs[0]) == 3
    ep = geo.classify_run(0, tp, runs[0], t, (0.0, 1500.0), settings)
    assert ep.accepted and ep.n_tp == 3
    assert ep.period_med_yr == pytest.approx(1000.0, abs=1.0)
    assert ep.coverage_start_gap_p == pytest.approx(0.25, abs=0.01)
    assert ep.coverage_end_gap_p == pytest.approx(0.25, abs=0.01)


def test_two_turning_points_are_too_few(settings):
    t = np.arange(0.0, 1000.0 + 1e-9, 1.0)
    tp = geo.find_turning_points(t, sine(t, 1000.0, 0.7), 1 * DEG, (0.0, 1000.0))
    assert len(tp) == 2
    episodes = geo.find_episodes(t, tp, (0.0, 1000.0), settings)
    assert len(episodes) == 1
    assert not episodes[0].accepted
    assert episodes[0].reject_reason == RejectReason.TOO_FEW_TP
    assert geo.covering_episode(episodes, settings) is None


def test_split_runs_at_a_gap():
    # TPs every 5 yr (half-period 5); a 100-yr gap > 3 x 5 splits the run.
    times = [0, 5, 10, 15, 20, 120, 125, 130, 135]
    values = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5]
    tp = make_tp(times, values)
    runs = geo.split_runs(tp, settings_from_config())
    assert len(runs) == 2
    assert list(runs[0]) == [0, 1, 2, 3, 4]
    assert list(runs[1]) == [5, 6, 7, 8]


def test_split_runs_over_the_top_and_slip():
    # TP4 = -0.5 -> TP5 = 6.5: half-amplitude 3.5 >= pi and midpoint jump 0 -> 3 -> 6 > pi.
    times = np.arange(9) * 5.0
    values = [-0.5, 0.5, -0.5, 0.5, -0.5, 6.5, 5.5, 6.5, 5.5]
    tp = make_tp(times, values)
    runs = geo.split_runs(tp, settings_from_config())
    assert len(runs) == 2
    assert runs[0][0] == 0 and runs[-1][-1] == 8
    assert {0, 1, 2, 3} <= set(runs[0])
    assert {6, 7, 8} <= set(runs[1])
    assert not ({3, 6} <= set(runs[0])) and not ({3, 6} <= set(runs[1]))


def test_drift_saw_on_a_slope(settings):
    # WP2 truth: a small wave (0.3 rad, P = 100) on a slope moving 0.4 turn over 10 kyr.
    # first band ~0.01, last ~2.50: |diff| 2.49 > A_first + A_last = 0.6; centre never turns back.
    t = np.arange(0.0, 10000.0 + 1e-9, 0.5)
    nu = 0.4 * TWO_PI / 10000.0
    sigma = sine(t, 100.0, 0.3) + nu * t
    window = (50.0, 9950.0)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, window)
    assert len(tp) > 100
    runs = geo.find_episodes(t, tp, window, settings)
    assert not any(r.accepted for r in runs)
    big = [r for r in runs if r.n_tp >= settings.libration_min_turning_points]
    assert big and all(r.reject_reason == RejectReason.CENTRE_MOVES for r in big)
    assert all(not r.bands_overlap for r in big)
    assert geo.drift_fraction(runs, window, settings) >= 0.95


def test_bounded_centre_with_four_returns_is_modulated(settings):
    # WP2 truth: a centre +-60 deg with 4 returns is modulated. c = 60 deg cos(2 pi t / 2000),
    # T = 2.5 x 2000: turns at 1000, 2000, 3000, 4000 (4 returns); first band +1.04, last -1.04,
    # A = 0.2: the bands do not overlap, so only the returns make it a libration.
    t = np.arange(0.0, 5000.0 + 1e-9, 0.5)
    sigma = 60 * DEG * np.cos(TWO_PI * t / 2000.0) + sine(t, 100.0, 0.2)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 5000.0))
    episodes = geo.find_episodes(t, tp, (0.0, 5000.0), settings)
    accepted = [e for e in episodes if e.accepted]
    assert len(accepted) == 1
    assert accepted[0].shape == EpisodeShape.MODULATED
    assert accepted[0].centre_returns == 4
    assert not accepted[0].bands_overlap


def test_bounded_centre_with_two_returns_moves(settings):
    # same, T = 1.5 x 2000: returns at 1000, 2000 only (< 3), bands +1.04 / -1.04 apart -> centre_moves
    t = np.arange(0.0, 3000.0 + 1e-9, 0.5)
    sigma = 60 * DEG * np.cos(TWO_PI * t / 2000.0) + sine(t, 100.0, 0.2)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 3000.0))
    episodes = geo.find_episodes(t, tp, (0.0, 3000.0), settings)
    assert len(episodes) == 1
    assert not episodes[0].accepted
    assert episodes[0].reject_reason == RejectReason.CENTRE_MOVES
    assert episodes[0].centre_returns == 2


def test_episode_in_the_middle_does_not_cover(settings):
    # libration only on [3000, 7000] of a 10-kyr record: start gap ~ 30 periods -> no covering episode
    t = np.arange(0.0, 10000.0 + 1e-9, 1.0)
    inside = (t >= 3000) & (t <= 7000)
    sigma = np.where(inside, sine(t, 100.0, 0.5), 0.0)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 10000.0))
    episodes = geo.find_episodes(t, tp, (0.0, 10000.0), settings)
    accepted = [e for e in episodes if e.accepted]
    assert len(accepted) == 1
    assert accepted[0].coverage_start_gap_p == pytest.approx(30.25, abs=0.1)
    assert geo.covering_episode(episodes, settings) is None


def test_drift_fraction_counts_only_long_centre_moves_runs(settings):
    # window 0..1000: centre_moves run 0..600 (5 TPs) counts 0.6; a 2-TP centre_moves run, a
    # too_few_tp run and an accepted run do not count.
    runs = [
        make_episode([0, 150, 300, 450, 600], False, RejectReason.CENTRE_MOVES),
        make_episode([650, 700], False, RejectReason.CENTRE_MOVES),
        make_episode([720, 760], False, RejectReason.TOO_FEW_TP),
        make_episode([800, 850, 900, 950], True, RejectReason.NONE, EpisodeShape.STATIONARY),
    ]
    assert geo.drift_fraction(runs, (0.0, 1000.0), settings) == pytest.approx(0.6)
    assert geo.drift_fraction(runs[1:], (0.0, 1000.0), settings) == pytest.approx(0.0)


# ----------------------------------------------------------------------------------------------
# slips, revolutions, pattern B
# ----------------------------------------------------------------------------------------------


def _over_the_top(t, t_start, period):
    """0 -> 1 in half a cycle, (1 - cos)/2: added x 2pi at a maximum of A sin with A < pi, sigma
    keeps rising through the top (d/dt = (2 pi/P) sin(2 pi tau/P) (pi - A) > 0), i.e. one pass
    over the top, as a pendulum near the separatrix does (pattern A)."""
    x = np.clip((t - t_start) / (period / 2.0), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def test_slips_of_plus_and_minus_one_turn(settings):
    # A = 2.5 rad, P = 100 yr (T0 = P / F(2.5 rad) = 60 yr). +2pi over the top at the maximum
    # t = 3025, -2pi under the bottom at the minimum t = 6075. The midpoint steps 0 -> pi -> 2pi over
    # the TPs at t_s - 50 and t_s + 100: ~150 yr >= 0.5 T0 = 30 yr.
    t = np.arange(0.0, 9000.0 + 1e-9, 0.5)
    sigma = sine(t, 100.0, 2.5) + TWO_PI * _over_the_top(t, 3025.0, 100.0) - TWO_PI * _over_the_top(t, 6075.0, 100.0)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 9000.0))
    slips, n_short = geo.find_slips(t, sigma, tp, 60.0, [], settings)
    assert n_short == 0
    assert len(slips) == 2
    first, second = sorted(slips, key=lambda s: s.time_yr)
    assert first.size_rev == pytest.approx(1.0, abs=0.2)
    assert second.size_rev == pytest.approx(-1.0, abs=0.2)
    assert first.time_yr == pytest.approx(3050.0, abs=100.0)
    assert second.time_yr == pytest.approx(6100.0, abs=100.0)
    assert 30.0 <= first.duration_yr <= 200.0
    assert not first.suspect and not second.suspect
    # the same step at an unwrap-suspect time is marked
    slips_s, _ = geo.find_slips(t, sigma, tp, 60.0, [6100.0], settings)
    marked = {round(s.time_yr, -2): s.suspect for s in slips_s}
    assert marked.get(6100.0) is True
    assert marked.get(3000.0) is False
    # with T0 = 2000 yr the same 150-yr steps are faster than 0.5 T0: rejected as unwrap errors
    none, n_short = geo.find_slips(t, sigma, tp, 2000.0, [], settings)
    assert none == [] and n_short == 2


def test_instant_steps_are_unwrap_errors_not_slips(settings):
    # A 2pi jump within one sample (an unwrap error) accumulates over <= one cycle (100 yr), far
    # below 0.5 T0 = 1000 yr: both are rejected as too short. The down jump at 6010 (rising phase)
    # adds a TP pair: midpoints 0, -0.55, -1.70, -4.30, -2pi over 5925..6025 (hand check), i.e. a
    # 2pi step of the midpoint within 100 yr, which must be counted like the up jump.
    t = np.arange(0.0, 9000.0 + 1e-9, 0.5)
    sigma = sine(t, 100.0, 2.5) + np.where(t >= 3010.0, TWO_PI, 0.0) - np.where(t >= 6010.0, TWO_PI, 0.0)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 9000.0))
    slips, n_short = geo.find_slips(t, sigma, tp, 2000.0, [], settings)
    assert slips == []
    assert n_short == 2


def _rotation(h, direction, t):
    # pendulum rotation with energy h (V = -Omega^2 cos): sigma_dot(0) = 2 Omega sqrt(h) at sigma = 0
    omega0 = TWO_PI / 100.0
    sigma, rate = pendulum(omega0, 0.0, direction * 2.0 * omega0 * np.sqrt(h), t)
    return omega0, sigma, rate


@pytest.mark.parametrize('direction', [1.0, -1.0])
def test_revolutions_of_a_pendulum_rotation(direction):
    # h = 1.14: rate minimum at the top sigma = pi, min/mean rate = r_pend(1.14) = 0.554.
    # The turn period 2K(1/h)/(Omega sqrt(h)) = 73.6 yr; in 1500 yr ~ 20 turns.
    h = 1.14
    t = np.arange(0.0, 1500.0 + 1e-9, 0.05)
    omega0, sigma, rate = _rotation(h, direction, t)
    t_rot = 2.0 * ellipk(1.0 / h) / (omega0 * np.sqrt(h))
    revs = geo.revolutions(t, sigma, rate, (0.0, 1500.0))
    expected = int(np.floor(abs(sigma[-1] - sigma[0]) / TWO_PI))
    assert expected - 1 <= len(revs) <= expected
    for rev in revs:
        assert rev.t_end_yr - rev.t_start_yr == pytest.approx(t_rot, rel=2e-3)
        # the stall is at the top whatever the direction of the circulation
        assert geo.wrap_pi(rev.phase_min_rate_rad - np.pi) == pytest.approx(0.0, abs=0.02)
        assert rev.rate_min_over_mean == pytest.approx(r_pend_ref(h), abs=0.01)


def _revs(phases, h, rmin):
    return [
        Revolution(t_start_yr=100.0 * i, t_end_yr=100.0 * (i + 1), phase_min_rate_rad=p, rate_min_over_mean=rmin, energy_h=h)
        for i, p in enumerate(phases)
    ]


def test_pattern_b_detected_at_the_top_with_wrap_around(settings):
    # top = pi; phases alternate +3.1 and -3.1 (both 0.04 rad from pi across the wrap)
    h = 1.05
    revs = _revs([3.1, -3.1] * 4, h, r_pend_ref(h))
    pb = geo.pattern_b(revs, np.pi, 1.1, 0.02, settings)
    assert pb.n_revolutions == 8
    assert pb.stall_fraction == pytest.approx(1.0)
    assert pb.in_zone and pb.detected
    assert not pb.rate_inconsistent
    assert geo.wrap_pi(pb.stall_phase_mean_rad - np.pi) == pytest.approx(0.0, abs=1e-6)
    assert pb.stall_phase_r == pytest.approx(np.cos(np.pi - 3.1), abs=1e-6)
    assert pb.rate_min_over_mean_pend_med == pytest.approx(r_pend_ref(h), abs=1e-3)
    assert pb.h_rev_med == pytest.approx(h) and pb.h_rev_max == pytest.approx(h)


def test_pattern_b_fraction_threshold(settings):
    # 6 of 8 at the top = 0.75 (>= threshold) -> detected; 5 of 8 = 0.625 -> not
    h = 1.05
    six = _revs([np.pi] * 6 + [0.0, 1.0], h, r_pend_ref(h))
    five = _revs([np.pi] * 5 + [0.0, 1.0, -1.0], h, r_pend_ref(h))
    pb6 = geo.pattern_b(six, np.pi, 1.1, 0.02, settings)
    pb5 = geo.pattern_b(five, np.pi, 1.1, 0.02, settings)
    assert pb6.stall_fraction == pytest.approx(0.75) and pb6.detected
    assert pb5.stall_fraction == pytest.approx(0.625) and not pb5.detected


def test_pattern_b_needs_every_turn_in_the_zone(settings):
    # one turn at h = 1.5 > h_hi + 2 sigma_h = 1.14: out of the separatrix zone
    revs = _revs([np.pi] * 8, 1.05, r_pend_ref(1.05))
    revs[3] = dataclasses.replace(revs[3], energy_h=1.5)
    pb = geo.pattern_b(revs, np.pi, 1.1, 0.02, settings)
    assert pb.stall_fraction == pytest.approx(1.0)
    assert not pb.in_zone and not pb.detected
    assert pb.h_rev_max == pytest.approx(1.5)


def test_pattern_b_rate_inconsistent_flag(settings):
    h = 1.14
    revs = _revs([np.pi] * 8, h, r_pend_ref(h) + 0.3)  # 0.3 > libration_stall_rate_tol 0.2
    pb = geo.pattern_b(revs, np.pi, 1.2, 0.02, settings)
    assert pb.rate_inconsistent
    assert pb.rate_min_over_mean_pend_med == pytest.approx(0.554, abs=2e-3)


# ----------------------------------------------------------------------------------------------
# arc and stall
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize('direction', [1.0, -1.0])
def test_stall_strength_bound_by_hand(direction):
    # sigma from pi/2 - 0.1 to pi/2 + 0.1 over tau = 2000 yr (range 0.2 < pi -> stall); |sin| at
    # sigma_c in {0, pi} lies in [cos 0.1, 1] = [0.995, 1]. a rises by da = 1e-4 au; c = -0.06.
    # range bound K = da/(tau |sin|) = 5e-8 au/yr; P_lib,min = 2 pi sqrt(tau |sin| / (|c| da))
    #   = 2 pi sqrt(2000 / 6e-6) = 2 pi x 18257 = 114 712 yr (within 0.5 % for |sin| in [0.995, 1]).
    t = np.arange(0.0, 2000.0 + 1e-9, 1.0)
    sigma = np.pi / 2 + direction * (-0.1 + 0.2 * t / 2000.0)
    a = 2.5 + 1e-4 * t / 2000.0
    arc = geo.arc_or_stall(t, sigma, a, -0.06, (0.0, 2000.0))
    assert arc.kind == 'stall'
    assert arc.tau_yr == pytest.approx(2000.0, abs=1.0)
    assert arc.range_rad == pytest.approx(0.2, abs=1e-6)
    assert arc.rate_rad_yr == pytest.approx(direction * 1e-4, rel=1e-3)
    assert abs(arc.acc_rad_yr2) < 1e-10
    assert arc.slow_ptp == pytest.approx(1e-4, rel=1e-6)
    assert arc.slow_trend_per_yr == pytest.approx(5e-8, rel=1e-3)
    assert arc.force_bound_range == pytest.approx(5e-8, rel=0.006)
    assert arc.period_min_yr == pytest.approx(114712.0, rel=0.006)
    assert arc.wrapped_min_rad == pytest.approx(np.pi / 2 - 0.1, abs=1e-3)
    assert arc.wrapped_max_rad == pytest.approx(np.pi / 2 + 0.1, abs=1e-3)


def test_arc_when_range_exceeds_pi():
    # sigma = 0.001 t over 5000 yr: range 5 rad >= pi -> arc
    t = np.arange(0.0, 5000.0 + 1e-9, 1.0)
    arc = geo.arc_or_stall(t, 0.001 * t, 2.5 + 0 * t + 1e-6 * np.sin(t), -0.5, (0.0, 5000.0))
    assert arc.kind == 'arc'
    assert arc.range_rad == pytest.approx(5.0, abs=1e-6)


# ----------------------------------------------------------------------------------------------
# visible exit and entry
# ----------------------------------------------------------------------------------------------


def _exit_series(t):
    # libration (A 0.5, P 100) until 5000, then sigma rises at 0.01 rad/yr: |sigma - 0| > pi at 5314.2
    return np.where(t < 5000.0, sine(t, 100.0, 0.5), 0.01 * (t - 5000.0))


def test_visible_exit_after_the_last_episode(settings):
    t = np.arange(0.0, 6000.0 + 1e-9, 1.0)
    sigma = _exit_series(t)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 6000.0))
    ep = [e for e in geo.find_episodes(t, tp, (0.0, 6000.0), settings) if e.accepted][-1]
    edge = geo.visible_edge(t, sigma, sigma, ep, 'exit', settings)
    assert edge is not None and edge.kind == 'exit'
    assert edge.time_yr == pytest.approx(5000.0 + np.pi / 0.01, abs=30.0)
    # max |sigma - c| = 0.01 x 1000 = 10 rad -> beyond = 10 - pi
    assert edge.beyond_rad == pytest.approx(10.0 - np.pi, abs=0.05)
    # (6000 - 5314) / 100 = 6.86 periods left
    assert edge.remaining_periods == pytest.approx(6.86, abs=0.35)
    assert not edge.suspected
    assert edge.raw_monotonic


def test_visible_exit_near_the_end_is_suspected(settings):
    t = np.arange(0.0, 5350.0 + 1e-9, 1.0)
    sigma = _exit_series(t)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 5350.0))
    ep = [e for e in geo.find_episodes(t, tp, (0.0, 5350.0), settings) if e.accepted][-1]
    edge = geo.visible_edge(t, sigma, sigma, ep, 'exit', settings)
    assert edge is not None
    assert edge.remaining_periods < 1.0  # (5350 - 5314) / 100 = 0.36
    assert edge.suspected


def test_visible_entry_mirrors_the_exit(settings):
    # sigma = 0.01 (t - 1000) before 1000, libration after: |sigma| > pi until 1000 - 314.2 = 685.8
    t = np.arange(0.0, 6000.0 + 1e-9, 1.0)
    sigma = np.where(t < 1000.0, 0.01 * (t - 1000.0), sine(t - 1000.0, 100.0, 0.5))
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 6000.0))
    ep = [e for e in geo.find_episodes(t, tp, (0.0, 6000.0), settings) if e.accepted][0]
    edge = geo.visible_edge(t, sigma, sigma, ep, 'entry', settings)
    assert edge is not None and edge.kind == 'entry'
    assert edge.time_yr == pytest.approx(1000.0 - np.pi / 0.01, abs=30.0)
    assert edge.remaining_periods == pytest.approx(6.86, abs=0.35)
    assert not edge.suspected


def test_no_exit_for_a_steady_libration(settings):
    t = np.arange(0.0, 5000.0 + 1e-9, 1.0)
    sigma = sine(t, 100.0, 0.5)
    tp = geo.find_turning_points(t, sigma, 1 * DEG, (0.0, 5000.0))
    ep = geo.find_episodes(t, tp, (0.0, 5000.0), settings)[0]
    assert geo.visible_edge(t, sigma, sigma, ep, 'exit', settings) is None
    assert geo.visible_edge(t, sigma, sigma, ep, 'entry', settings) is None


# ----------------------------------------------------------------------------------------------
# jumps and the last regime
# ----------------------------------------------------------------------------------------------


def test_jump_baseline_by_hand(settings):
    # tau = max(2 P_c, 5 dt): (100, 1) -> 200; (10, 5) -> 25
    assert geo.jump_baseline_yr(100.0, 1.0, settings) == pytest.approx(200.0)
    assert geo.jump_baseline_yr(10.0, 5.0, settings) == pytest.approx(25.0)


def test_jump_candidate_at_a_fast_step(settings):
    # a = 2.5 + 1e-3 step at 3000 over ~20 yr + noise 1e-6: |da_tau| = 1e-3 >> 8 robust std (~1e-5)
    rng = np.random.default_rng(7)
    t = np.arange(0.0, 10000.0 + 1e-9, 1.0)
    a = 2.5 + 1e-3 * 0.5 * (1 + np.tanh((t - 3000.0) / 5.0)) + 1e-6 * rng.standard_normal(t.size)
    cands = geo.jump_candidates(t, a, 50.0, 1000.0, settings)
    assert len(cands) >= 1
    assert all(abs(c.time_yr - 3000.0) <= 50.0 for c in cands)
    assert max(abs(c.size) for c in cands) == pytest.approx(1e-3, rel=0.05)
    assert all(c.duration_yr < 1000.0 for c in cands)


def test_slow_drift_of_a_is_not_a_jump(settings):
    # the same 1e-3 accumulated over 2000 yr, slower than the 500-yr reference period: adiabatic drift
    rng = np.random.default_rng(8)
    t = np.arange(0.0, 10000.0 + 1e-9, 1.0)
    a = 2.5 + np.interp(t, [0, 2000, 4000, 10000], [0, 0, 1e-3, 1e-3]) + 1e-6 * rng.standard_normal(t.size)
    assert geo.jump_candidates(t, a, 50.0, 500.0, settings) == []


def test_confirm_jump_by_energy_and_kick(settings):
    cand = [JumpCandidate(time_yr=3000.0, size=1e-3, duration_yr=20.0)]
    times = np.arange(100.0, 6000.0, 200.0)
    # h 0.3 -> 0.8: dh = 0.5 > max(2 x 0.02, 0.1)
    jumps = geo.confirm_jumps(cand, times, np.where(times < 3000, 0.3, 0.8), 0.02, 5e-4, settings)
    assert len(jumps) == 1
    assert jumps[0].dh == pytest.approx(0.5, abs=1e-9)
    assert not jumps[0].kick and not jumps[0].by_slow_only
    # unchanged energy: a kick that does not cut the record
    kicks = geo.confirm_jumps(cand, times, np.full(times.size, 0.3), 0.02, 5e-4, settings)
    assert len(kicks) == 1 and kicks[0].kick


def test_confirm_jump_by_slow_variable_with_one_cycle_before(settings):
    # only one cycle before the candidate (< 2): |size| 1e-3 > half-width 5e-4 decides; dh = nan
    cand = [JumpCandidate(time_yr=3000.0, size=1e-3, duration_yr=20.0)]
    times = np.array([2900.0, 3100.0, 3300.0, 3500.0])
    jumps = geo.confirm_jumps(cand, times, np.array([0.3, 0.3, 0.3, 0.3]), 0.02, 5e-4, settings)
    assert len(jumps) == 1
    assert jumps[0].by_slow_only and not jumps[0].kick
    assert math.isnan(jumps[0].dh)


def test_last_regime_start_by_hand():
    t = np.arange(0.0, 10000.0 + 1e-9, 1.0)
    jumps = [
        Jump(time_yr=3000.0, size=1e-3, duration_yr=10.0, dh=0.5, by_slow_only=False, kick=False),
        Jump(time_yr=6000.0, size=1e-3, duration_yr=10.0, dh=0.0, by_slow_only=False, kick=True),
    ]
    # the last non-kick jump at 3000 + tau/2 = 3050
    assert geo.last_regime_start(t, jumps, 100.0) == pytest.approx(3050.0)
    assert geo.last_regime_start(t, jumps[1:], 100.0) == pytest.approx(0.0)
    assert geo.last_regime_start(t, [], 100.0) == pytest.approx(0.0)


# ----------------------------------------------------------------------------------------------
# wiggle
# ----------------------------------------------------------------------------------------------


def test_wiggle_half_amplitude_and_period():
    # sqrt(2) std(A sin) = A over whole periods; T = 50 x 137 puts 137 yr exactly on an FFT bin
    t = np.arange(0.0, 50 * 137.0, 0.5)
    centre = 1.0 + 1e-4 * t
    amp, period = geo.wiggle(t, centre + sine(t, 137.0, 0.3), centre, (0.0, t[-1]))
    assert amp == pytest.approx(0.3, rel=0.01)
    assert period == pytest.approx(137.0, rel=0.03)


# ----------------------------------------------------------------------------------------------
# measure_geometry / finish_geometry on synthetic truths
# ----------------------------------------------------------------------------------------------


def test_pendulum_libration_period_and_amplitude(settings):
    # A = 150 deg, T0 = 100 yr: P = T0 F(150 deg) = 176.2 yr (closed form, M&D 8.50-8.52)
    omega0 = TWO_PI / 100.0
    amp = 150 * DEG
    t = np.arange(0.0, 20000.0 + 1e-9, 0.25)
    sigma, rate = pendulum(omega0, amp, 0.0, t)
    g = geo.measure_geometry(make_angle(t, sigma, rate, cutoff=20.0), 1 * DEG, settings)
    expected = 100.0 * amplitude_factor_ref(amp)
    assert g.period_cycle_yr == pytest.approx(expected, rel=2e-3)
    assert len(g.episodes) == 1
    ep = g.episodes[0]
    assert ep.half_amp_med_rad == pytest.approx(amp, abs=1e-3)
    assert ep.shape == EpisodeShape.STATIONARY
    assert geo.wrap_pi(ep.centre_mod_2pi_rad) == pytest.approx(0.0, abs=1e-3)


def test_measure_geometry_steady_libration(settings):
    # sigma = pi + sin(2 pi t / 200), window (60, 19940): extrema at 150, 250, ..., 19850 -> 198 TPs
    t = np.arange(0.0, 20000.0 + 1e-9, 1.0)
    sigma = sine(t, 200.0, 1.0, centre=np.pi)
    angle = make_angle(t, sigma, cutoff=50.0, window=(60.0, 19940.0))
    g = geo.measure_geometry(angle, 1 * DEG, settings)
    assert len(g.turning_points) == 198
    assert g.n_tp_cut_half == 198 and g.n_tp_cut_double == 198
    assert g.period_cycle_yr == pytest.approx(200.0, abs=1.0)
    assert len(g.episodes) == 1 and g.episodes[0].shape == EpisodeShape.STATIONARY
    assert g.bands_overlap
    assert g.centre.max_pass_rev < 0.1
    # net of sigma_f itself is the endpoint difference: sin(2pi 99.7) - sin(2pi 0.3) = -2 sin(0.6 pi)
    # = -1.902 rad -> -0.3027 rev (a libration's net is bounded by 2A, not zero)
    assert g.net_rev_filtered == pytest.approx(-2 * np.sin(0.6 * np.pi) / TWO_PI, abs=1e-3)
    assert g.wiggle_half_amp_rad == pytest.approx(1.0, rel=0.03)
    assert g.wiggle_period_yr == pytest.approx(200.0, rel=0.05)
    assert g.arc_stall is None and g.exit is None and g.entry is None
    assert g.slips == [] and g.jumps == [] and g.pattern_b is None
    cols = g.to_columns()
    assert cols['geo_n_tp'] == 198
    assert cols['geo_tp_prominence_deg'] == pytest.approx(1.0)
    assert cols['geo_period_cycle_yr'] == pytest.approx(200.0, abs=1.0)
    assert cols['geo_centre_max_pass_rev'] == pytest.approx(g.centre.max_pass_rev)


@pytest.mark.parametrize('direction', [1.0, -1.0])
def test_measure_geometry_circulation(settings, direction):
    # nu = +-2pi/1000 rad/yr with a wiggle 0.1 sin(2pi t/200): the wiggle slope 0.0031 < 0.0063,
    # sigma never reverses (no TPs). Window (50, 19950): 19.9 turns.
    t = np.arange(0.0, 20000.0 + 1e-9, 1.0)
    sigma = direction * TWO_PI * t / 1000.0 + sine(t, 200.0, 0.1)
    g = geo.measure_geometry(make_angle(t, sigma, cutoff=50.0), 1 * DEG, settings)
    assert len(g.turning_points) == 0
    assert g.episodes == []
    assert g.centre.max_pass_rev == pytest.approx(19.9, abs=0.1)
    assert g.net_rev_filtered == pytest.approx(direction * 19.9, abs=0.1)
    assert g.net_rev_raw == pytest.approx(direction * 19.9, abs=0.2)
    assert 18 <= len(g.revolutions) <= 20
    assert g.arc_stall is None or g.arc_stall.kind == 'arc'


def test_measure_geometry_constant_angle(settings):
    t = np.arange(0.0, 2000.0 + 1e-9, 1.0)
    g = geo.measure_geometry(make_angle(t, np.full(t.size, 1.0), rate=np.zeros(t.size)), 1 * DEG, settings)
    assert len(g.turning_points) == 0
    assert g.episodes == []
    assert g.centre.max_pass_rev == pytest.approx(0.0, abs=1e-6)
    assert g.arc_stall is not None and g.arc_stall.kind == 'stall'
    assert g.arc_stall.range_rad == pytest.approx(0.0, abs=1e-9)


def test_finish_geometry_without_events_keeps_the_window(settings):
    t = np.arange(0.0, 20000.0 + 1e-9, 1.0)
    sigma = sine(t, 200.0, 1.0, centre=np.pi)
    angle = make_angle(t, sigma, cutoff=50.0, window=(60.0, 19940.0))
    g = geo.measure_geometry(angle, 1 * DEG, settings)
    cyc = np.arange(100.0, 20000.0, 200.0)
    g = geo.finish_geometry(g, angle, 190.0, 0.0, 1.2, 0.02, cyc, np.full(cyc.size, 0.23), 0.01, 200.0, [], settings)
    assert g.regime_start_yr == pytest.approx(0.0)
    assert g.window == pytest.approx((60.0, 19940.0))
    assert g.slips == [] and [j for j in g.jumps if not j.kick] == []


def test_finish_geometry_cuts_at_a_jump(settings):
    # Amplitude 0.5 -> 1.5 rad at 10000 (h = sin^2(A/2): 0.061 -> 0.465, dh = 0.40 > 0.1) with a
    # fast 1e-3 au step of a. tau = max(2 x 50, 5 x 1) = 100 -> regime start ~ 10000 + 50.
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 20000.0 + 1e-9, 1.0)
    amp = np.where(t < 10000.0, 0.5, 1.5)
    sigma = np.pi + amp * np.sin(TWO_PI * t / 200.0)
    a = 2.5 + 1e-3 * 0.5 * (1 + np.tanh((t - 10000.0) / 5.0)) + 1e-7 * rng.standard_normal(t.size)
    angle = make_angle(t, sigma, cutoff=50.0, slow=a)
    g = geo.measure_geometry(angle, 1 * DEG, settings)
    cyc = np.arange(100.0, 20000.0, 200.0)
    h = np.where(cyc < 10000.0, np.sin(0.25) ** 2, np.sin(0.75) ** 2)
    g = geo.finish_geometry(g, angle, 190.0, 0.0, 1.2, 0.02, cyc, h, 0.05, 2000.0, [], settings)
    real = [j for j in g.jumps if not j.kick]
    assert len(real) == 1
    assert real[0].time_yr == pytest.approx(10000.0, abs=100.0)
    assert 10000.0 <= g.regime_start_yr <= 10200.0
    assert g.window[0] >= g.regime_start_yr - 1.0


# ----------------------------------------------------------------------------------------------
# preprocess
# ----------------------------------------------------------------------------------------------


def _kepler_period(a):
    return a**1.5  # Gaussian years


def test_cutoff_hilda_floor(settings):
    # Hilda a = 3.97 (P = 7.91 yr), Jupiter 11.86 yr: P_syn = 1/(1/7.91 - 1/11.86) = 23.7 yr
    # P_c = max(1.5 x 23.7 = 35.6 yr, 150 yr / 2 = 75 yr) -> binding floor; the step only counts samples
    p_syn = 1.0 / (1.0 / _kepler_period(3.97) - 1.0 / 11.862)
    assert p_syn == pytest.approx(23.7, abs=0.1)
    for step in (1.0, 16.0):
        cut = preprocess.choose_cutoff({'Jupiter': p_syn}, step, settings)
        assert cut.cutoff_yr == pytest.approx(75.0)
        assert cut.binding == 'floor'
        assert cut.synodic_bound_yr == pytest.approx(1.5 * p_syn)
        assert cut.floor_bound_yr == pytest.approx(75.0)
        assert cut.samples_per_cutoff == pytest.approx(75.0 / step)
    # a 250-yr libration keeps G(250; 75) = 1 / (1 + 0.3^4) = 0.992 of its amplitude
    assert gain(250.0, 75.0) == pytest.approx(1 / (1 + 0.3**4)) and gain(250.0, 75.0) >= 0.94


def test_cutoff_griqua_floor(settings):
    # 2J-1 at 1 yr: synodic bound 1.5 x 11.9 = 17.85 yr < floor bound 75 yr -> floor
    cut = preprocess.choose_cutoff({'Jupiter': 11.9}, 1.0, settings)
    assert cut.cutoff_yr == pytest.approx(75.0)
    assert cut.binding == 'floor'
    assert cut.synodic_bound_yr == pytest.approx(17.85)


def test_cutoff_longest_synodic_no_planets_and_user(settings):
    two = preprocess.choose_cutoff({'Jupiter': 20.0, 'Saturn': 60.0}, 1.0, settings)
    assert two.cutoff_yr == pytest.approx(90.0) and two.binding == 'synodic'
    none = preprocess.choose_cutoff({}, 2.0, settings)
    assert none.synodic_bound_yr == pytest.approx(0.0)
    assert none.cutoff_yr == pytest.approx(75.0) and none.binding == 'floor'
    assert none.samples_per_cutoff == pytest.approx(37.5)
    user = preprocess.choose_cutoff({'Jupiter': 23.7}, 1.0, settings_from_config(mmr_filter_cutoff_yr=500.0))
    assert user.cutoff_yr == pytest.approx(500.0) and user.binding == 'user'


@pytest.mark.parametrize(
    'synodic, step, cutoff, binding',
    [({'Jupiter': 23.7}, 1.0, 35.55, 'synodic'), ({'Jupiter': 11.9}, 1.0, 17.85, 'synodic'), ({'Jupiter': 11.9}, 15.9, 17.85, 'synodic')],
)
def test_cutoff_rule_without_the_libration_period_floor(synodic, step, cutoff, binding):
    # With MMR_LIBRATION_PERIOD_MIN_YR off (0) only the synodic bound is left, whatever the step.
    off = settings_from_config(mmr_libration_period_min_yr=0.0)
    cut = preprocess.choose_cutoff(synodic, step, off)
    assert cut.cutoff_yr == pytest.approx(cutoff)
    assert cut.binding == binding


def test_filter_guard(settings):
    # (filter_too_slow = G(P_cyc) < MMR_L8_GAIN_FULL = 0.94, G(P_cyc)); P_c = 35.55
    slow, g = preprocess.filter_guard(250.0, 35.55, settings)
    assert slow is False or slow == False  # noqa: E712
    assert g == pytest.approx(gain(250.0, 35.55))
    fast, g2 = preprocess.filter_guard(60.0, 35.55, settings)
    assert fast
    assert g2 == pytest.approx(1.0 / (1.0 + (35.55 / 60.0) ** 4))  # 0.8675
    # at exactly 2 P_c the guard does not fire: G = 16/17 = 0.941 >= 0.94
    edge, g3 = preprocess.filter_guard(71.1, 35.55, settings)
    assert not edge
    assert g3 == pytest.approx(16.0 / 17.0)
    none, g4 = preprocess.filter_guard(NAN, 35.55, settings)
    assert not none and math.isnan(g4)


def _spec_2j1():
    return ResonanceSpec(
        key='2J-1+0-1',
        kind=ResonanceKind.TWO_BODY,
        planets=('Jupiter',),
        planet_coefficients=(2,),
        body_coefficient=-1,
        body_varpi_coefficient=-1,
        planet_varpi_coefficients=(0,),
        nominal_axis_au=3.278,
    )


def test_predicted_ripple_by_hand():
    # r_GI = |k_J dlJ - k_S dlS| G(P_GI) (antiphased waves): 3J-1S: (3 x 0.34 + 1 x 0.83) deg x G(883; 100)
    gi = GreatInequality(period_yr=883.0, dlambda_jupiter_rad=0.34 * DEG, dlambda_saturn_rad=0.83 * DEG, measured=False)
    spec3 = ResonanceSpec(
        key='3J-1S-1+0+0-1',
        kind=ResonanceKind.THREE_BODY,
        planets=('Jupiter', 'Saturn'),
        planet_coefficients=(3, -1),
        body_coefficient=-1,
        body_varpi_coefficient=-1,
        planet_varpi_coefficients=(0, 0),
    )
    expected = (3 * 0.34 + 0.83) * DEG * gain(883.0, 100.0)
    assert preprocess.predicted_ripple(spec3, gi, 100.0, 2) == pytest.approx(expected, rel=1e-6)
    assert preprocess.predicted_ripple(_spec_2j1(), gi, 100.0, 2) == pytest.approx(2 * 0.34 * DEG * gain(883.0, 100.0), rel=1e-6)
    mars = ResonanceSpec(
        key='9M-16+0+7',
        kind=ResonanceKind.TWO_BODY,
        planets=('Mars',),
        planet_coefficients=(9,),
        body_coefficient=-16,
        body_varpi_coefficient=7,
        planet_varpi_coefficients=(0,),
    )
    assert preprocess.predicted_ripple(mars, gi, 100.0, 2) == pytest.approx(0.0, abs=1e-15)


N_JUPITER = 30.3496 * DEG  # rad / Gaussian yr
DLAMBDA_J = 0.34 * DEG
P_GI = 883.0


def _jupiter_longitude(t):
    # n_J t + great inequality + a fast synodic-like term the cutoff removes (G(5.93; 100) ~ 1e-5)
    return N_JUPITER * t + DLAMBDA_J * np.sin(TWO_PI * t / P_GI) + 0.002 * np.sin(TWO_PI * t / 5.93)


def test_planet_ripple_recovers_the_great_inequality():
    # 2J-1: ripple = k_J dlambda_J^slow = 2 x 0.34 deg sin(2 pi t / 883) x G(883; 100)
    t = np.arange(0.0, 20000.0 + 1e-9, 1.0)
    wrapped = np.mod(_jupiter_longitude(t), TWO_PI)
    ripple = preprocess.planet_ripple(t, _spec_2j1(), {'Jupiter': wrapped}, {'Jupiter': N_JUPITER}, 100.0, 2)
    truth = 2 * DLAMBDA_J * gain(P_GI, 100.0) * np.sin(TWO_PI * t / P_GI)
    inside = (t > 2000.0) & (t < 18000.0)
    resid = (ripple - truth)[inside]
    resid -= resid.mean()
    assert np.max(np.abs(resid)) < 0.03 * 2 * DLAMBDA_J


def test_remove_planet_ripple_is_exact_and_the_correlation_diagnostic():
    # Contract changed after API.md (DELTA item 1, TEST-DISPUTES.md): lambda_j enters sigma linearly, so
    # the subtraction is exact and never undone by the band correlation; that correlation is a column.
    t = np.arange(0.0, 40000.0 + 1e-9, 1.0)
    clean = np.pi + 0.3 * np.sin(TWO_PI * t / 20000.0)  # outside the band [300, 3000]
    ripple = 2 * DLAMBDA_J * np.sin(TWO_PI * t / P_GI)
    out, half_amp, corr = preprocess.remove_planet_ripple(t, clean + ripple, ripple, (300.0, 3000.0), 2)
    assert corr >= 0.8  # nothing else in the band (the bound of the original test; edges of the band-pass lower it)
    assert half_amp == pytest.approx(2 * DLAMBDA_J, rel=1e-3)  # ptp / 2 of the sine (1-yr sampling of 883 yr)
    np.testing.assert_allclose(out, clean, atol=1e-12)
    # a 400-yr libration of 30 deg in the band: it decorrelates from the ripple (|corr| ~ 0.006 measured,
    # requests-preprocess-geometry.md item 4), yet the subtraction is still exactly the ripple
    libration = clean + np.radians(30.0) * np.sin(TWO_PI * t / 400.0 + 1.0)
    out2, _, corr2 = preprocess.remove_planet_ripple(t, libration + ripple, ripple, (300.0, 3000.0), 2)
    assert abs(corr2) < 0.1
    np.testing.assert_allclose(out2, libration, atol=1e-12)


def test_ripple_subtracted_only_when_the_angle_is_the_saved_longitudes(settings):
    # The gate of the exact subtraction is the angle reconstruction (MMR_RIPPLE_RECONSTRUCTION_MAX_DEG = 1):
    # sigma_saved = 2 lambda_J - lambda - varpi by construction in _libration_series (error 0 up to rounding).
    series, _, _, _ = _libration_series()
    context = build_context(series, settings)
    sampling = analyse_sampling(series, context, settings)
    assert sampling.angle_reconstruction_rad < np.radians(1e-6)
    prep = preprocess.preprocess(series, context, sampling, settings)
    assert prep.ripple_removed and prep.ripple_rad is not None
    # a saved lambda_J off by 5 deg sin(2 pi t / 4000) (five whole cycles, offset 0): the angle no longer is the combination
    # (error 2 x 5 = 10 deg > 1 deg after removing the constant offset) -> nothing subtracted
    jupiter = series.planets['Jupiter']
    bad = dataclasses.replace(
        jupiter, mean_longitude=np.mod(jupiter.mean_longitude + np.radians(5.0) * np.sin(TWO_PI * series.t_yr / 4000.0), TWO_PI)
    )
    broken = dataclasses.replace(series, planets={'Jupiter': bad})
    sampling_b = analyse_sampling(broken, context, settings)
    assert sampling_b.angle_reconstruction_rad == pytest.approx(np.radians(10.0), rel=0.01)
    prep_b = preprocess.preprocess(broken, context, sampling_b, settings)
    assert not prep_b.ripple_removed and prep_b.ripple_rad is None
    np.testing.assert_array_equal(prep_b.angle_clean, prep_b.angle_unwrapped)


def _libration_series():
    # 2:1 with Jupiter, sigma = 2 lambda_J - lambda - varpi = pi + 0.5 sin(2 pi t / 500).
    # Identity: n = 2 n_J - g - sigma_dot, a = (2 pi / n)^(2/3); lambda = (2 n_J - g) t - sigma - varpi0.
    t = np.arange(0.0, 20000.0 + 1e-9, 1.0)
    sigma = sine(t, 500.0, 0.5, centre=np.pi)
    sigma_dot = 0.5 * TWO_PI / 500.0 * np.cos(TWO_PI * t / 500.0)
    g = 20.0 / 3600.0 * DEG
    varpi = 0.4 + g * t
    n = 2 * N_JUPITER - g - sigma_dot
    a = (TWO_PI / n) ** (2.0 / 3.0)
    lam_j = N_JUPITER * t
    lam = (2 * N_JUPITER - g) * t - sigma - 0.4
    node = 1.0 - g * t
    ones = np.ones_like(t)
    jupiter = PlanetSeries(
        name='Jupiter',
        t_yr=t,
        a=5.2026 * ones,
        e=0.0484 * ones,
        inc=0.0228 * ones,
        Omega=1.75 * ones,
        varpi=0.257 * ones,
        mean_longitude=np.mod(lam_j, TWO_PI),
    )
    series = MMRSeries(
        t_yr=t,
        a=a,
        e=0.15 * ones,
        inc=0.05 * ones,
        Omega=np.mod(node, TWO_PI),
        omega=np.mod(varpi - node, TWO_PI),
        varpi=np.mod(varpi, TWO_PI),
        angle_wrapped=np.mod(sigma, TWO_PI),
        resonance=_spec_2j1(),
        regime=Regime(step_yr=1.0, span_yr=20000.0, n_rows=t.size, planets_saved=('Jupiter',)),
        planets={'Jupiter': jupiter},
        mean_longitude=np.mod(lam, TWO_PI),
        name='synthetic-2J-1',
    )
    return series, sigma, sigma_dot, a


def test_preprocess_end_to_end_on_a_constructed_libration(settings):
    series, sigma, sigma_dot, a = _libration_series()
    context = build_context(series, settings)
    sampling = analyse_sampling(series, context, settings)
    assert sampling.unwrap_ok
    prep = preprocess.preprocess(series, context, sampling, settings)
    # n ~ 2 n_J -> P_syn = 2 pi / n_J = 11.86 yr (the cutoff rule itself is tested above)
    assert context.synodic_periods_yr['Jupiter'] == pytest.approx(TWO_PI / N_JUPITER, rel=1e-3)
    cutoff = preprocess.choose_cutoff(context.synodic_periods_yr, 1.0, settings).cutoff_yr
    assert prep.cutoff.cutoff_yr == pytest.approx(cutoff)
    assert prep.window == pytest.approx((cutoff, 20000.0 - cutoff))
    inside = (prep.t_yr >= prep.window[0]) & (prep.t_yr <= prep.window[1])
    # G(500; P_c <= 75) >= 0.9995: the filtered angle is the angle
    # each version away from its own filter edges (2 x its cutoff), error <= the gain loss + 0.005
    for name, factor in (('angle_f', 1.0), ('angle_f_half', 0.5), ('angle_f_double', 2.0)):
        trim = 2 * factor * cutoff
        sel = (prep.t_yr >= trim) & (prep.t_yr <= 20000.0 - trim)
        diff = geo.wrap_pi(getattr(prep, name)[sel] - sigma[sel])
        assert np.max(np.abs(diff)) < 0.5 * (1 - gain(500.0, factor * cutoff)) + 0.005, name
    np.testing.assert_allclose(prep.rate_f[inside], sigma_dot[inside], atol=0.02 * 0.5 * TWO_PI / 500.0)
    np.testing.assert_allclose(prep.a_f[inside], a[inside], atol=1e-4)
    np.testing.assert_allclose(prep.e_f[inside], 0.15, atol=1e-6)

    angle = preprocess.filtered_angle(prep, context, sampling.step_yr)
    assert angle.window == pytest.approx(prep.window)
    assert angle.cutoff_yr == pytest.approx(cutoff)
    assert angle.step_yr == pytest.approx(1.0)
    assert angle.identity_coefficient == pytest.approx(context.identity_coefficient)
    np.testing.assert_allclose(angle.slow_f, prep.a_f)
    np.testing.assert_allclose(angle.angle_f, prep.angle_f)
    # c = -1.5 k n / a with k = -1, n = 2 n_J, a = a*: by hand
    a_star = (TWO_PI / (2 * N_JUPITER)) ** (2.0 / 3.0)
    assert context.identity_coefficient == pytest.approx(1.5 * 2 * N_JUPITER / a_star, rel=0.01)

    slow = preprocess.slow_components(series, prep, 500.0, settings)
    assert slow.window_yr == pytest.approx(1500.0)
    assert slow.window == pytest.approx((1500.0, 18500.0))
    mid = (slow.t_yr >= slow.window[0]) & (slow.t_yr <= slow.window[1])
    # residual of the 500-yr libration after a 1500-yr smoothing: 0.5 x G(500; 1500) = 0.006 rad
    assert np.max(np.abs(geo.wrap_pi(slow.angle_slow[mid] - np.pi))) < 0.05
    np.testing.assert_allclose(slow.a_slow[mid], np.mean(a), atol=2e-4)


# ----------------------------------------------------------------------------------------------
# decide_angle (G0-G6)
# ----------------------------------------------------------------------------------------------


def _decide(settings, sigma_t, sampling=None, energy=None, cutoff=50.0, unphysical=False, geometry=None):
    t, sigma, rate = sigma_t
    if geometry is None:
        geometry = geo.measure_geometry(make_angle(t, sigma, rate, cutoff=cutoff), 1 * DEG, settings)
    sampling = sampling or make_sampling(n=t.size, step=float(t[1] - t[0]))
    ledger = Ledger.from_settings(settings)
    dec = decide.decide_angle(sampling, unphysical, geometry, energy, settings, ledger)
    return dec, ledger, geometry


def _series(fn, t_end, step=1.0):
    t = np.arange(0.0, t_end + 1e-9, step)
    sigma = fn(t)
    return t, sigma, np.gradient(sigma, t)


def test_g0_unphysical_before_everything(settings):
    ledger = Ledger.from_settings(settings)
    dec = decide.decide_angle(make_sampling(unwrap_ok=False), True, None, None, settings, ledger)
    assert (dec.angle_status, dec.subtype, dec.rule) == (AngleStatus.UNDEFINED, AngleSubtype.UNPHYSICAL, 'G0')


def test_g0_unwrap_failed_alias_and_undefined_phase(settings):
    ledger = Ledger.from_settings(settings)
    dec = decide.decide_angle(make_sampling(unwrap_ok=False), False, None, None, settings, ledger)
    assert (dec.angle_status, dec.subtype, dec.rule) == (AngleStatus.UNDEFINED, AngleSubtype.ALIAS, 'G0')
    jitter = make_sampling(unwrap_ok=False, jitter=1.2, jitter_low_e=True, undefined_phase=True)
    dec2 = decide.decide_angle(jitter, False, None, None, settings, Ledger.from_settings(settings))
    assert (dec2.angle_status, dec2.subtype) == (AngleStatus.UNDEFINED, AngleSubtype.UNDEFINED_PHASE)


def test_g2_steady_libration(settings):
    dec, ledger, _ = _decide(settings, _series(lambda t: sine(t, 200.0, 1.0, np.pi), 20000.0))
    assert (dec.angle_status, dec.subtype, dec.rule) == (AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2')
    assert len(ledger) >= 1  # the coverage comparison goes through the ledger


def test_g2_exactly_three_turning_points_is_a_libration(settings):
    dec, _, g = _decide(settings, _series(lambda t: sine(t, 1000.0, 0.7), 1500.0), cutoff=20.0)
    assert len(g.turning_points) == 3
    assert (dec.angle_status, dec.subtype) == (AngleStatus.LIBRATION, AngleSubtype.STATIONARY)


def test_g2_modulated_libration(settings):
    # MMR-like periods: libration 200 yr, centre +-60 deg at 4000 yr over 2.5 modulation periods
    fn = lambda t: 60 * DEG * np.cos(TWO_PI * t / 4000.0) + sine(t, 200.0, 0.2)  # noqa: E731
    dec, _, _ = _decide(settings, _series(fn, 10000.0), cutoff=20.0)
    assert (dec.angle_status, dec.subtype) == (AngleStatus.LIBRATION, AngleSubtype.MODULATED)


def test_g2_transient_and_recurrent(settings):
    one = lambda t: np.where((t >= 3000) & (t <= 7000), sine(t, 200.0, 0.5), 0.0)  # noqa: E731
    dec, _, _ = _decide(settings, _series(one, 10000.0))
    assert (dec.angle_status, dec.subtype, dec.rule) == (AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2')
    inside = lambda t: ((t >= 2000) & (t <= 6000)) | ((t >= 10000) & (t <= 14000))  # noqa: E731
    two = lambda t: np.where(inside(t), sine(t, 200.0, 0.5), 0.0)  # noqa: E731
    dec2, _, _ = _decide(settings, _series(two, 20000.0))
    assert (dec2.angle_status, dec2.subtype) == (AngleStatus.EPISODIC, AngleSubtype.RECURRENT)


@pytest.mark.parametrize('direction', [1.0, -1.0])
def test_g4_circulation(settings, direction):
    fn = lambda t: direction * TWO_PI * t / 1000.0 + sine(t, 200.0, 0.1)  # noqa: E731
    dec, _, _ = _decide(settings, _series(fn, 20000.0))
    assert (dec.angle_status, dec.subtype, dec.rule) == (AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4')


def test_g4_aliased_circulation(settings):
    # nu = 71.98 deg/yr sampled at 5.0012 yr (Knezevic & Milani): the predictive unwrap is the true
    # line, 0.19994 rev/yr x ~99 800 yr ~ 19 954 revolutions; the step verdict is aliased.
    step = 5.0012
    nu = 71.98 * DEG
    t = np.arange(0.0, 100000.0, step)
    sigma = nu * t
    s_alias = make_sampling(
        n=t.size, step=step, verdict=StepVerdict.ALIASED_UNAMBIGUOUS, branches=(1,), rate_apparent_pred_rad_yr=-0.0039 * DEG
    )
    dec, _, g = _decide(settings, (t, sigma, np.full(t.size, nu)), sampling=s_alias, cutoff=20 * step)
    assert g.net_rev_filtered == pytest.approx(nu * (t[-1] - t[0] - 40 * step) / TWO_PI, rel=0.01)
    assert (dec.angle_status, dec.subtype) == (AngleStatus.CIRCULATION, AngleSubtype.ALIAS)
    s_amb = make_sampling(n=t.size, step=step, verdict=StepVerdict.AMBIGUOUS, branches=(0, 1))
    dec2, _, _ = _decide(settings, (t, sigma, np.full(t.size, nu)), sampling=s_amb, cutoff=20 * step, geometry=g)
    assert (dec2.angle_status, dec2.subtype) == (AngleStatus.CIRCULATION, AngleSubtype.ALIAS)


def test_g4_numpy_unwrap_differs_marks_alias(settings):
    fn = lambda t: TWO_PI * t / 1000.0  # noqa: E731
    sampling = make_sampling(n=20001, alias_numpy_differs=True)
    dec, _, _ = _decide(settings, _series(fn, 20000.0), sampling=sampling)
    assert (dec.angle_status, dec.subtype) == (AngleStatus.CIRCULATION, AngleSubtype.ALIAS)


def test_g4_centre_one_and_a_quarter_turns_and_back(settings):
    # No wiggle: sigma itself goes +1.25 turns and back (a single TP at the top). The largest
    # monotone pass 7.48 rad >= 2pi -> circulation although the net is ~0 (WP2 truth, rule 3).
    fn = lambda t: 2.5 * np.pi * np.sin(np.pi * t / 20000.0)  # noqa: E731
    dec, _, g = _decide(settings, _series(fn, 20000.0), cutoff=100.0)
    assert len(g.turning_points) == 1
    assert abs(g.centre.net_rev) < 0.05
    assert (dec.angle_status, dec.rule) == (AngleStatus.CIRCULATION, 'G4')


def test_centre_travelling_one_and_a_quarter_turns_is_not_a_libration(settings):
    # The same centre with a small libration wiggle on it (A 0.5, P 100). The first and last bands
    # overlap only because the centre came back; in between it went through every phase, so the
    # centre is not stationary (definitions.md) and the largest pass is >= 2pi (WP2 truth).
    fn = lambda t: 2.5 * np.pi * np.sin(np.pi * t / 20000.0) + sine(t, 100.0, 0.5)  # noqa: E731
    dec, _, g = _decide(settings, _series(fn, 20000.0))
    assert g.centre.max_pass_rev > 1.0
    assert dec.angle_status == AngleStatus.CIRCULATION


def test_g5_drift(settings):
    nu = 0.4 * TWO_PI / 10000.0
    dec, _, g = _decide(settings, _series(lambda t: sine(t, 100.0, 0.3) + nu * t, 10000.0, 0.5))
    assert g.centre.max_pass_rev == pytest.approx(0.4, abs=0.05)
    assert (dec.angle_status, dec.subtype, dec.rule) == (AngleStatus.DRIFT, AngleSubtype.NONE, 'G5')


def test_g6_few_cycles(settings):
    # P = 800 on 1000 yr: TPs at 200 and 600 only -> no episode, no drift
    dec, _, g = _decide(settings, _series(lambda t: sine(t, 800.0, 0.7), 1000.0), cutoff=20.0)
    assert len(g.turning_points) == 2
    assert (dec.angle_status, dec.subtype, dec.rule) == (AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, 'G6')


@pytest.mark.parametrize(
    'slope, subtype',
    [(0.001, AngleSubtype.ARC), (-0.001, AngleSubtype.ARC), (0.0005, AngleSubtype.STALL), (0.0, AngleSubtype.STALL)],
)
def test_g6_arc_and_stall(settings, slope, subtype):
    # no TPs; window (20, 4980): range |slope| x 4960 = 4.96 rad (arc, < 2pi pass) or 2.48 (stall)
    dec, _, _ = _decide(settings, _series(lambda t: 1.0 + slope * t, 5000.0), cutoff=20.0)
    assert (dec.angle_status, dec.subtype, dec.rule) == (AngleStatus.INSUFFICIENT_TIME, subtype, 'G6')


def test_slow_arc_of_more_than_a_turn_is_circulation(settings):
    # 1.2 turns over 10 kyr, window (50, 9950): 1.19 turns >= 1 -> G4
    dec, _, _ = _decide(settings, _series(lambda t: 1.2 * TWO_PI * t / 10000.0, 10000.0))
    assert dec.angle_status == AngleStatus.CIRCULATION


def _rotation_geometry(settings):
    h = 1.14
    t = np.arange(0.0, 3000.0 + 1e-9, 0.05)
    omega0, sigma, rate = _rotation(h, 1.0, t)
    angle = make_angle(t, sigma, rate, cutoff=5.0)
    g = geo.measure_geometry(angle, 1 * DEG, settings)
    revs = [dataclasses.replace(r, energy_h=h) for r in g.revolutions]
    g.pattern_b = geo.pattern_b(revs, np.pi, 1.2, 0.02, settings)
    potential = Potential(
        n_harmonics=1,
        cos_coeffs=np.array([-(omega0**2)]),
        sin_coeffs=np.array([0.0]),
        r2=0.99,
        well_rad=0.0,
        top_rad=np.pi,
        omega_rad_yr=omega0,
        v_min=-(omega0**2),
        v_max=omega0**2,
    )
    zone = Zone(method=ZoneMethod.NUMERIC, j_lo=0.8, h_lo=0.9, h_hi=1.2, adiabatic_lo=0.9, adiabatic_hi=1.1, mismatch=False)
    return t, sigma, rate, g, potential, zone


def test_g3_pattern_b_of_a_rotation_just_above_the_separatrix(settings):
    # pendulum rotation at h = 1.14: every turn stalls at the top pi with min/mean 0.554
    t, sigma, rate, g, potential, zone = _rotation_geometry(settings)
    assert len(g.turning_points) == 0 and g.episodes == []
    assert g.pattern_b.detected
    energy = EnergyAnalysis(
        potential_e=potential,
        potential_f=None,
        ve_vf_agree_fraction=NAN,
        well_coherence=0.99,
        zone=zone,
        sigma_only=CheckValue.PASS,
    )
    dec, _, _ = _decide(settings, (t, sigma, rate), energy=energy, geometry=g)
    assert (dec.angle_status, dec.subtype, dec.rule) == (AngleStatus.NEAR_SEPARATRIX, AngleSubtype.PATTERN_B, 'G3')
    # without a sigma-only potential the same turns are a circulation (G3 needs R0 sigma)
    no_potential = dataclasses.replace(energy, sigma_only=CheckValue.FAIL)
    dec2, _, _ = _decide(settings, (t, sigma, rate), energy=no_potential, geometry=g)
    assert (dec2.angle_status, dec2.rule) == (AngleStatus.CIRCULATION, 'G4')
