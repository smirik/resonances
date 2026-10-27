"""Independent functional tests of W16 (physical filter cutoff, step validity, co-orbital ripple, filter_too_slow by
gain, per-run turning-point thresholds, the L4 jump guard, O1 inside its regime, the zone e-fit on a short segment,
the subresonance swing gate).

Every expected number below is derived here from the documented formulas (docs/libration-analysis.md §2-§5,
docs/dry-run.md) or measured on synthetic records whose answer is known by construction, not read from the code.

Synthetic records: sigma is a pendulum (or a prescribed kinematic angle) and the body's mean longitude and a follow
from it exactly (sigma = k_J lambda_J + k lambda + p varpi, n = d lambda / dt, a = (2 pi / n)^(2/3) au, yr), so the
identity sigma_dot = c (a - a*) holds by construction.
"""

import dataclasses
import functools
import types

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import filtering, geometry
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import preprocess, response, subresonances, zone
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.context import table_mean_motion
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings

SETTINGS = AnalysisSettings()

# Jupiter: mass ratio 1/1047.3486 (IAU), sidereal period 11.862 yr. Linear tadpole period P_J / sqrt(27 mu / 4)
# (Murray & Dermott eq. 3.151) = 147.76 yr; floor 0.5 x that, cutoff floor / 2.
MU_JUPITER = 1.0 / 1047.3486
P_JUPITER_YR = 11.862
TADPOLE_YR = P_JUPITER_YR / np.sqrt(27.0 * MU_JUPITER / 4.0)
TROJAN_CUTOFF_YR = 0.5 * TADPOLE_YR / 2.0  # 36.94 yr


# --------------------------------------------------------------------------------------------------------------------
# synthetic records


def _pendulum(t, period0_yr, amp_deg, centre_rad=0.0):
    w0 = 2 * np.pi / period0_yr
    sol = solve_ivp(
        lambda _, y: [y[1], -(w0**2) * np.sin(y[0] - centre_rad)],
        (t[0], t[-1]),
        [centre_rad + np.radians(amp_deg), 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        method='DOP853',
    )
    return sol.y[0], sol.y[1]


def _series(t, sigma, sigma_dot, key, planet_wave_deg=0.0, body_follows_planet=False):
    """MMRSeries of a body whose angle is `sigma`; the planet's saved lambda carries a 900-yr wave of
    `planet_wave_deg` (a great-inequality-like slow inequality). `body_follows_planet`: the body's lambda follows
    the planet's true (waved) longitude, as a Trojan does; otherwise it follows the planet's mean line, so the
    saved angle carries k_J x the wave."""
    spec = ResonanceSpec.from_key(key)
    planet = spec.planets[0]
    n_planet = table_mean_motion(planet)
    lam_line = n_planet * t + 0.3
    wave = np.radians(planet_wave_deg) * np.sin(2 * np.pi * t / 900.0)
    wave_dot = np.radians(planet_wave_deg) * 2 * np.pi / 900.0 * np.cos(2 * np.pi * t / 900.0)
    lam_planet = lam_line + wave
    varpi = np.full_like(t, 1.0)
    k_j, k, p = spec.planet_coefficients[0], spec.body_coefficient, spec.body_varpi_coefficient
    reference = lam_planet if body_follows_planet else lam_line
    lam = (sigma - k_j * reference - p * varpi) / k
    n = (sigma_dot - k_j * n_planet - (k_j * wave_dot if body_follows_planet else 0.0)) / k
    a = (2 * np.pi / n) ** (2.0 / 3.0)
    angle = np.mod(k_j * lam_planet + k * lam + p * varpi, 2 * np.pi)
    ones = np.ones_like(t)
    planet_series = PlanetSeries(
        planet, t, 5.2 * ones, 0.048 * ones, 0.02 * ones, 1.75 * ones, 0.25 * ones, mean_longitude=np.mod(lam_planet, 2 * np.pi)
    )
    step = float(t[1] - t[0])
    return MMRSeries(
        t_yr=t,
        a=a,
        e=0.15 * ones,
        inc=0.05 * ones,
        Omega=0.4 * ones,
        omega=np.mod(varpi - 0.4, 2 * np.pi),
        varpi=varpi,
        angle_wrapped=angle,
        resonance=spec,
        regime=Regime(step, float(t[-1] - t[0]), len(t)),
        planets={planet: planet_series},
        mean_longitude=np.mod(lam, 2 * np.pi),
        name='synthetic',
    )


@functools.lru_cache(maxsize=None)
def _three_one(step_yr, span_yr=20000.0, planet_wave_deg=0.0):
    """3J-1 pendulum, small period 400 yr, half-amplitude 50 deg: period 400 x 2K(sin 25 deg)/pi = 419.7 yr."""
    t = np.arange(0.0, span_yr + step_yr / 2, step_yr)
    sigma, sigma_dot = _pendulum(t, 400.0, 50.0)
    series = _series(t, sigma, sigma_dot, '3J-1', planet_wave_deg=planet_wave_deg)
    return analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False), sigma


@functools.lru_cache(maxsize=None)
def _trojan(step_yr, span_yr=5000.0):
    """1J-1 tadpole about 60 deg, half-amplitude 15 deg, period ~150 yr; Jupiter's lambda carries a 0.34-deg 900-yr
    wave that the Trojan follows."""
    t = np.arange(0.0, span_yr + step_yr / 2, step_yr)
    sigma, sigma_dot = _pendulum(t, 150.0, 15.0, centre_rad=np.pi / 3)
    series = _series(t, sigma, sigma_dot, '1J-1', planet_wave_deg=0.34, body_follows_planet=True)
    return analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False), sigma


# --------------------------------------------------------------------------------------------------------------------
# Q9: the cutoff from physics only, the step as a condition


def test_cutoff_formula_ignores_the_step():
    # 3J-1 at a = 2.5 au: n = 360/2.5^1.5 = 91.07 deg/yr, n_J = 30.35 deg/yr -> P_syn = 5.93 yr; 1.5 P_syn = 8.9 yr
    # is below P_floor / 2 = 75 yr, so P_c = 75 yr at every step.
    synodic = {'Jupiter': 360.0 / (360.0 / 2.5**1.5 - 360.0 / 11.862)}
    for step in (0.25, 1.0, 8.0, 15.9, 30.0):
        cutoff = preprocess.choose_cutoff(synodic, step, SETTINGS, 150.0)
        assert cutoff.cutoff_yr == pytest.approx(75.0)
    # a long synodic period binds: 1.5 x 60 = 90 yr > 75 yr, still independent of the step
    for step in (1.0, 8.0):
        assert preprocess.choose_cutoff({'Saturn': 60.0}, step, SETTINGS, 150.0).cutoff_yr == pytest.approx(90.0)


def test_pendulum_at_1_and_8_yr_gives_the_same_turning_points_and_period():
    fine, _ = _three_one(1.0)
    coarse, _ = _three_one(8.0)
    row_fine, row_coarse = fine.numbers_row(), coarse.numbers_row()
    assert row_fine['flt_cutoff_yr'] == pytest.approx(75.0)
    assert row_coarse['flt_cutoff_yr'] == pytest.approx(75.0)
    tp_fine = fine.geometry.turning_points.times_yr
    tp_coarse = coarse.geometry.turning_points.times_yr
    # 20 kyr minus 2 x 75 yr of filter edges at half-periods of 209.8 yr: 94 turning points
    assert len(tp_fine) == len(tp_coarse) == 94
    assert np.max(np.abs(tp_fine - tp_coarse)) <= 8.0  # within one coarse step
    period = 400.0 * 1.0492  # 2K(k)/pi for k = sin 25 deg
    assert fine.geometry.period_cycle_yr == pytest.approx(period, abs=2.0)
    assert coarse.geometry.period_cycle_yr == pytest.approx(period, abs=8.0)
    assert fine.verdict.status == coarse.verdict.status


def test_short_period_flag_follows_the_orbital_period():
    # min(P_orb = 3.95 yr, P_syn = 5.93 yr) / 2 = 1.98 yr: 1 yr is below, 8 yr above
    fine, _ = _three_one(1.0)
    coarse, _ = _three_one(8.0)
    assert fine.numbers_row()['smp_short_period_step_yr'] == pytest.approx(2.5**1.5 / 2, rel=0.01)
    assert 'short_period_aliased' not in fine.verdict.flags
    assert 'short_period_aliased' in coarse.verdict.flags
    # since W20 round D the record never lifts the cap: flagged -> capped (medium at most), however small sigma's fast
    # residual. At 8 yr P_orb = 3.953 yr folds to |1/3.953 - 2/8|^-1 = 336 yr (its harmonic to 168 yr, P_syn to 22.9
    # and 26.5 yr)
    row = coarse.numbers_row()
    assert row['smp_fast_over_half_amp'] < 0.1 and 'short_period_aliased' in coarse.verdict.caps
    assert str(coarse.verdict.confidence) in ('medium', 'low')
    # P_orb from the record's mean a (3.9538 yr against 3.9528 at 2.5 au): a fold this close to 2/dt amplifies it
    assert row['smp_folded_slowest_yr'] == pytest.approx(1 / abs(1 / 2.5**1.5 - 2 / 8.0), rel=0.03)
    # the opt-in lift (MMR_SHORT_PERIOD_CAP_RATIO = 0.1, the W20 F5 rule) needs the scatter of a below 0.5 of its swing
    lift = dataclasses.replace(SETTINGS, mmr_short_period_cap_ratio=0.1)
    t = np.arange(0.0, 20000.0 + 4.0, 8.0)
    lifted = analyse_mmr(_series(t, *_pendulum(t, 400.0, 50.0), '3J-1'), lift, scan_subresonances=False, scan_owner=False)
    ratio = lifted.numbers_row()['smp_da_sp_over_a_excursion']
    assert ('short_period_aliased' in lifted.verdict.caps) is (not ratio <= lift.mmr_short_period_a_ratio)


def test_filter_realizable_boundary():
    # P_c <= 2 dt: no filter; 75 yr at 37.5 yr is exactly the limit
    assert not preprocess.filter_realizable(75.0, 37.5)
    assert preprocess.filter_realizable(75.0, 37.4)
    assert not preprocess.filter_realizable(TROJAN_CUTOFF_YR, 18.5)
    assert preprocess.filter_realizable(TROJAN_CUTOFF_YR, 18.0)


def test_undersampled_ordinary_mmr_recommends_5_yr():
    # 40 yr: 2 dt = 80 >= 75 -> invalid/undersampled; the resample step is the largest of the list <= P_floor / 20
    # = 7.5 yr (and < 37.5 yr): 5 yr.
    analysis, _ = _three_one(40.0)
    verdict = analysis.verdict
    assert str(verdict.status) == 'invalid'
    assert str(verdict.reason) == 'undersampled'
    assert str(verdict.action) == 'resample'
    assert verdict.resample_step_yr == pytest.approx(5.0)


def test_step_just_below_the_limit_is_analysed():
    # 37 yr: 2 dt = 74 < 75, the filter exists: no undersampled verdict, the same cutoff and the same 94 turning
    # points (a 420-yr cycle is 11 steps), but the short-period terms fold: confidence <= medium.
    analysis, _ = _three_one(37.0)
    assert str(analysis.verdict.reason) != 'undersampled'
    assert analysis.numbers_row()['flt_cutoff_yr'] == pytest.approx(75.0)
    assert len(analysis.geometry.turning_points.times_yr) == 94
    assert 'short_period_aliased' in analysis.verdict.flags


# --------------------------------------------------------------------------------------------------------------------
# Trojans: co-orbital cutoff, no ripple subtraction, undersampled at 19 yr


def test_trojan_cutoff_is_half_the_tadpole_floor():
    analysis, _ = _trojan(1.0)
    row = analysis.numbers_row()
    assert row['flt_period_floor_yr'] == pytest.approx(0.5 * TADPOLE_YR, rel=0.005)
    assert row['flt_cutoff_yr'] == pytest.approx(TROJAN_CUTOFF_YR, rel=0.005)
    assert row['flt_cutoff_binding'] == 'floor'


def test_coorbital_angle_keeps_the_planet_longitude():
    analysis, sigma = _trojan(1.0)
    prep = analysis.prep
    assert not prep.ripple_removed
    assert np.array_equal(prep.angle_clean, prep.angle_unwrapped)
    # the saved angle is the true libration (the body follows Jupiter's waved longitude): up to whole turns
    offset = prep.angle_unwrapped - sigma
    turns = np.round(np.median(offset) / (2 * np.pi)) * 2 * np.pi
    assert np.max(np.abs(offset - turns)) < np.radians(0.01)


def test_non_coorbital_angle_subtracts_the_ripple():
    # contrast: 3J-1 with the same 0.34-deg wave in the saved lambda_J; the body does not follow it, so the angle
    # holds 3 x 0.34 = 1.02 deg of ripple (gain ~1 at 900 yr), and the subtraction leaves the pendulum.
    analysis, sigma = _three_one(1.0, span_yr=10000.0, planet_wave_deg=0.34)
    prep = analysis.prep
    assert prep.ripple_removed
    # ptp / 2 of lowpass(lambda_J - its least-squares line): over 11 cycles the line takes a slope of order
    # 3 / (pi N) of the amplitude, which adds a few per cent at the ends
    assert np.degrees(prep.ripple_half_amp_rad) == pytest.approx(1.02, rel=0.1)
    offset = prep.angle_clean - sigma
    middle = slice(len(sigma) // 10, -len(sigma) // 10)
    residual = offset[middle] - np.median(offset[middle])
    assert np.max(np.abs(residual)) < np.radians(0.1)


@pytest.mark.parametrize('step, undersampled', [(18.0, False), (19.0, True)])
def test_trojan_step_limit(step, undersampled):
    # 2 x 18 = 36 < 36.94 yr; 2 x 19 = 38 >= 36.94 yr. Resample: largest of the list <= 147.8 / 2 / 20 = 3.69 yr: 2.5.
    analysis, _ = _trojan(step)
    verdict = analysis.verdict
    if undersampled:
        assert (str(verdict.status), str(verdict.reason)) == ('invalid', 'undersampled')
        assert verdict.resample_step_yr == pytest.approx(2.5)
    else:
        assert str(verdict.reason) != 'undersampled'
        assert str(verdict.status) == 'resonant'
        assert 'short_period_aliased' in verdict.flags  # 18 yr >= P_J / 2


# --------------------------------------------------------------------------------------------------------------------
# filter_too_slow by the digital gain


def _measured_gain(period_yr, cutoff_yr, step_yr):
    t = np.arange(0.0, 400 * period_yr, step_yr)
    x = np.sin(2 * np.pi * t / period_yr)
    y = filtering.lowpass(t, x, cutoff_yr, SETTINGS.filter_order, detrend=False)
    middle = slice(len(t) // 4, 3 * len(t) // 4)
    return float(np.std(y[middle]) / np.std(x[middle]))


@pytest.mark.parametrize('period, cutoff, step', [(100.0, 75.0, 1.0), (150.0, 75.0, 1.0), (143.0, 73.9, 15.9)])
def test_gain_formula_matches_the_filter(period, cutoff, step):
    assert filtering.butterworth_gain(period, cutoff, SETTINGS.filter_order, step) == pytest.approx(
        _measured_gain(period, cutoff, step), abs=0.01
    )


def test_filter_too_slow_reads_the_gain_not_the_ratio():
    # analog limit: G(P) = 1 / (1 + (P_c / P)^4); G(2 P_c) = 16/17 = 0.941 >= 0.94, G(1.9 P_c) = 0.929 < 0.94
    slow, gain = preprocess.filter_guard(2.0 * 75.0, 75.0, SETTINGS)
    assert not slow and gain == pytest.approx(16.0 / 17.0, rel=1e-6)
    slow, gain = preprocess.filter_guard(1.9 * 75.0, 75.0, SETTINGS)
    assert slow and gain == pytest.approx(1.0 / (1.0 + 1.9**-4), rel=1e-6)
    # digital: a 143-yr cycle at 15.9 yr against P_c = 73.9 yr. The ratio 143 / 73.9 = 1.93 < 2, but
    # G = 1 / (1 + (tan(pi 15.9/143) / tan(pi 15.9/73.9))^4) = 0.960 (measured above too): not too slow.
    expected = 1.0 / (1.0 + (np.tan(np.pi * 15.9 / 143.0) / np.tan(np.pi * 15.9 / 73.9)) ** 4)
    slow, gain = preprocess.filter_guard(143.0, 73.9, SETTINGS, 15.9)
    assert not slow
    assert gain == pytest.approx(expected, rel=1e-6)
    assert gain == pytest.approx(0.960, abs=0.002)
    # at or above Nyquist (P <= 2 dt) the filter cannot carry the cycle at all
    slow, _ = preprocess.filter_guard(30.0, 73.9, SETTINGS, 15.9)
    assert slow


def test_real_pendulum_is_not_filter_too_slow():
    analysis, _ = _three_one(8.0)
    row = analysis.numbers_row()
    assert not row['flt_filter_too_slow']
    assert row['flt_gain_at_pcyc'] > 0.99  # 420 yr against 75 yr


# --------------------------------------------------------------------------------------------------------------------
# Q11: each run of turning points at its own threshold


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@functools.lru_cache(maxsize=None)
def _quiet_libration_then_escape():
    """Libration of half-amplitude 10 deg and period 200 yr up to 3 kyr; over 3-4 kyr the centre starts to circulate
    smoothly (no jump of a) at 1.2 deg/yr with a 100-yr wave of 15 deg, monotone (wave rate 0.7 < 1.2 deg/yr).

    Record threshold by hand: the 100-yr wave keeps G(100; 75) = 0.76, 11.4 deg in sigma_f, of which 0.835 is faster
    than 150 yr: 9.5 deg over 60 % of the record, 95th percentile ~9.4 deg, p_TP = 4 x 9.4 ~ 38 deg, above the
    libration's 20-deg swings. On the libration alone 9.8 deg x (1 - G(200; 150) = 0.24) = 2.4 deg was the libration's
    own image (p_TP ~ 9 deg before W18); with that image taken out (`analyse.libration_image`) what stays is
    9.8 deg x 0.24 x (1 - G(200; 75) = 0.02) = 0.05 deg: the floor."""
    t = np.arange(0.0, 10000.5, 1.0)
    s = _smoothstep((t - 3000.0) / 1000.0)
    rate = np.radians(1.2) * s
    centre = np.concatenate([[0.0], np.cumsum((rate[1:] + rate[:-1]) / 2 * np.diff(t))])
    sigma = centre + np.radians(10.0) * (1 - s) * np.sin(2 * np.pi * t / 200.0) + np.radians(15.0) * s * np.sin(2 * np.pi * t / 100.0)
    series = _series(t, sigma, np.gradient(sigma, t), '3J-1')
    return analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False)


def test_record_threshold_would_drop_the_quiet_libration():
    analysis = _quiet_libration_then_escape()
    tp = analysis.geometry.turning_points
    assert 25.0 < np.degrees(tp.prominence_rad) < 50.0
    record_only = geometry.find_turning_points(analysis.prep.t_yr, analysis.prep.angle_f, tp.prominence_rad, analysis.geometry.window)
    assert np.count_nonzero(record_only.times_yr < 3000.0) < 3


def test_run_threshold_recovers_the_libration_before_the_exit():
    analysis = _quiet_libration_then_escape()
    tp = analysis.geometry.turning_points
    assert len(tp.local) >= 1
    run = tp.local[0]
    assert run.start_yr < 300.0 and run.end_yr > 2800.0
    assert np.degrees(run.prominence_rad) == pytest.approx(SETTINGS.mmr_tp_prominence_floor_deg)
    accepted = [e for e in analysis.geometry.episodes if e.accepted]
    assert len(accepted) == 1
    episode = accepted[0]
    # first extremum of 10 sin(2 pi t / 200) inside [75, ...] is the minimum at 150 yr; half-periods of 100 yr
    assert episode.t_start_yr == pytest.approx(150.0, abs=5.0)
    assert 2900.0 < episode.t_end_yr < 3500.0
    assert episode.n_tp >= 28
    assert np.median(np.diff(episode.tp_times_yr[::2])) == pytest.approx(200.0, abs=5.0)
    assert str(analysis.verdict.angle_status) == 'episodic'
    assert analysis.geometry.jumps == [] or all(j.kick for j in analysis.geometry.jumps)


def test_l1_counts_at_half_and_double_cutoff_use_the_run_thresholds():
    # at 2 P_c = 150 yr the 200-yr libration keeps G = 1/(1 + 0.75^4) = 0.76: swings of ~15 deg, above the run's ~9 deg
    # and below the record's ~35 deg. The x0.5 / x2 counts are taken at the x1 thresholds, so all cycles stay.
    analysis = _quiet_libration_then_escape()
    n_tp = len(analysis.geometry.turning_points.times_yr)
    assert abs(analysis.geometry.n_tp_cut_double - n_tp) <= 2
    assert abs(analysis.geometry.n_tp_cut_half - n_tp) <= 2


def test_quiet_libration_is_not_filter_sensitive():
    row = _quiet_libration_then_escape().numbers_row()
    assert row['flt_x2_n_tp'] >= 20
    assert not row['flt_filter_sensitive']


# --------------------------------------------------------------------------------------------------------------------
# Q6: L4 windows next to a confirmed jump of a


def _jump_record():
    """Pendulum (period 420 yr) with a step of a of 10 a-wave amplitudes at 5 kyr, filtered at P_c = 75 yr."""
    t = np.arange(0.0, 10000.5, 1.0)
    sigma, sigma_dot = _pendulum(t, 400.0, 50.0)
    c = 1.5 * (2 * np.pi / 2.5**1.5) / 2.5  # |dn/da| at 2.5 au
    a = 2.5 + sigma_dot / c
    a = a + 10.0 * np.ptp(a) / 2 * (t >= 5000.0)
    a_f = filtering.lowpass(t, a, 75.0, SETTINGS.filter_order)
    angle_f = filtering.lowpass(t, sigma, 75.0, SETTINGS.filter_order)
    windows = response.force_windows(t, angle_f, np.gradient(a_f, t), 3.5 * 420.0, 0.5)
    return windows


def test_jump_zone_is_tau_half_plus_one_cutoff():
    guard = response.JumpGuard((5000.0,), 150.0)
    zones = response.jump_zones(guard, 75.0, SETTINGS)
    x = SETTINGS.filter_edge_trim_cutoffs
    assert zones == [pytest.approx((5000.0 - 75.0 - x * 75.0, 5000.0 + 75.0 + x * 75.0))]
    assert response.jump_zones(response.JumpGuard(), 75.0, SETTINGS) == []


def test_windows_across_a_confirmed_jump_are_dropped_the_rest_kept():
    windows = _jump_record()
    zones = response.jump_zones(response.JumpGuard((5000.0,), 150.0), 75.0, SETTINGS)
    low, high = zones[0]
    kept = response.outside_zones(windows, zones)
    expected = [w for w in windows if w.t_end_yr < low or w.t_start_yr > high]
    assert [(w.t_start_yr, w.t_end_yr) for w in kept] == [(w.t_start_yr, w.t_end_yr) for w in expected]
    dropped = [w for w in windows if w not in expected]
    assert dropped
    assert min(w.r2 for w in dropped) < 0.5  # the step's peak of a_dot spoils the window
    inner = [w for w in kept if w.t_start_yr > 100.0 and w.t_end_yr < 9900.0]
    assert min(w.r2 for w in inner) > 0.95  # the pendulum law away from the jump


def test_kicks_do_not_guard():
    kick = geometry.Jump(time_yr=3000.0, size=0.01, duration_yr=50.0, dh=0.0, by_slow_only=False, kick=True)
    jump = geometry.Jump(time_yr=5000.0, size=0.05, duration_yr=50.0, dh=0.5, by_slow_only=False, kick=False)
    fake = types.SimpleNamespace(jumps=[kick, jump], jump_tau_yr=150.0)
    guard = response.JumpGuard.from_geometry(fake)
    assert guard.times_yr == (5000.0,)
    assert guard.half_tau_yr == pytest.approx(75.0)
    windows = [(2500.0, 3500.0), (4400.0, 4900.0), (6000.0, 7000.0)]
    kept = response.outside_zones(windows, response.jump_zones(guard, 75.0, SETTINGS), bounds=tuple)
    assert kept == [(2500.0, 3500.0), (6000.0, 7000.0)]
    assert response.JumpGuard.from_geometry(None).times_yr == ()


# --------------------------------------------------------------------------------------------------------------------
# Q7: O1 only inside its regime, away from the record's ends and from confirmed jumps


T_O1 = np.arange(0.0, 20000.5, 1.0)
P_O1 = 200.0
REACH = SETTINGS.filter_edge_trim_cutoffs * 2 * P_O1  # 400 yr with the default 1.0


def test_onset_near_the_record_start_is_not_measured():
    (low, high), why = response.onset_clear(T_O1, P_O1, 300.0, 3000.0, None, SETTINGS)
    assert (low, high) == pytest.approx((REACH, 20000.0 - REACH))
    assert why == 'record_edge'


def test_onset_far_from_edges_and_jumps_is_measured():
    (low, high), why = response.onset_clear(T_O1, P_O1, 5000.0, 8000.0, None, SETTINGS)
    assert why == ''
    assert (low, high) == pytest.approx((REACH, 20000.0 - REACH))


@pytest.mark.parametrize(
    'jump, expected_why, expected_clear',
    [
        (4700.0, 'jump', (4700.0 + 75.0 + REACH, 20000.0 - REACH)),  # 'before' stretch [4600, 5000] lies in the jump zone
        (4000.0, '', (4000.0 + 75.0 + REACH, 20000.0 - REACH)),  # 4475: the 400 yr before the episode stay
        (9000.0, '', (REACH, 9000.0 - 75.0 - REACH)),  # a jump after the episode cuts only the far side
    ],
)
def test_onset_after_a_jump(jump, expected_why, expected_clear):
    guard = response.JumpGuard((jump,), 150.0)
    clear, why = response.onset_clear(T_O1, P_O1, 5000.0, 8000.0, guard, SETTINGS)
    assert why == expected_why
    assert clear == pytest.approx(expected_clear)


def test_onset_bands_leave_out_a_stretch_cut_by_the_clear_range():
    # an a-wave of period 200 yr only inside the episode [5000, 8000]; the record ends at 8300, so the 'after' stretch
    # [8000, 8400] is cut at 8300 - 400 = 7900 and is absent; 'before' has no wave.
    t = np.arange(0.0, 8300.5, 1.0)
    a_f = 2.5 + 1e-3 * np.sin(2 * np.pi * t / P_O1) * ((t >= 5000.0) & (t <= 8000.0))
    clear, why = response.onset_clear(t, P_O1, 5000.0, 8000.0, None, SETTINGS)
    assert why == ''
    before, inside, after = response.onset_bands(t, a_f, P_O1, 5000.0, 8000.0, SETTINGS.filter_order, clear)
    assert np.isnan(after)
    # band gain at P: G(P; P/2) - G(P; 2P) = 1/(1 + 2^-4) - 1/(1 + 2^4) = 0.882
    band_gain = 1.0 / (1.0 + 2.0**-4) - 1.0 / (1.0 + 2.0**4)
    assert inside == pytest.approx(band_gain * 1e-3 / np.sqrt(2), rel=0.05)
    assert before < 0.25 * inside


# --------------------------------------------------------------------------------------------------------------------
# Q12: the zone's e-fit takes no sinusoid on a segment shorter than its period


def test_e_cycle_much_longer_than_the_segment_gives_no_modulation():
    # 6 Myr e-cycle seen over 40 kyr: e is a straight line to 9e-6. dK/K = q de / e_bar ~ 2e-5; the slope is the
    # cycle's derivative at mid-segment: 0.01 x 2 pi / 6 Myr x cos(0.7 + 0.0209) = 7.86e-6 per kyr.
    t = np.arange(0.0, 40000.5, 10.0)
    nu = 2 * np.pi / 6.0e6
    e = 0.39 + 0.01 * np.sin(nu * t + 0.7)
    omega2, dk, trend = zone.modulation_from_eccentricity(t, e, 2, 1.0, nu)
    assert dk < 1e-3
    assert trend == pytest.approx(0.01 * nu * 1000.0 * np.cos(0.7 + nu * 20000.0), rel=0.02)
    assert np.ptp(omega2) < 1e-3


def test_e_cycle_covered_by_the_segment_is_fitted_with_its_sinusoid():
    # 1.5 cycles of 50 kyr plus a trend of 2e-4 per kyr. Only a fit with the sinusoid recovers the trend (a line alone
    # takes the unfinished cycle's tilt, checked below); de = 0.02 about e_bar = 0.2 + 2e-4 x 37.5 = 0.2075:
    # dK/K = 2 x 0.02 / 0.2075 = 0.193.
    t = np.arange(0.0, 75000.5, 10.0)
    nu = 2 * np.pi / 50000.0
    e = 0.2 + 0.02 * np.sin(nu * t + 1.5) + 2e-4 * t / 1000.0
    line_only = np.polyfit(t / 1000.0, e, 1)[0]
    assert abs(line_only / 2e-4 - 1) > 0.2
    omega2, dk, trend = zone.modulation_from_eccentricity(t, e, 2, 1.0, nu)
    assert trend == pytest.approx(2e-4, rel=0.03)
    assert dk == pytest.approx(2 * 0.02 / 0.2075, rel=0.03)
    assert np.max(omega2) / np.min(omega2) == pytest.approx(((0.2075 + 0.02) / (0.2075 - 0.02)) ** 2, rel=0.03)


# --------------------------------------------------------------------------------------------------------------------
# Q8: the subresonance scan does not read phase coherence below a 45-deg swing


LAWS = {'member_a': (0.95, 0.8, 5), 'member_b': (0.55, 0.3, 4)}


def _gate(swing_deg):
    ledger = Ledger(thin_band=SETTINGS.confidence_thin_band)
    return subresonances.swing_gate(0.6, dict(LAWS), np.radians(swing_deg), SETTINGS, ledger), ledger


def test_small_swing_leaves_coherence_unmeasured():
    # 2013 AS27 4E+9S-5 swings 3-4 deg: sin(sigma) and sigma agree to 0.1 %, every member's phase is coherent
    (sigma0, laws, small), ledger = _gate(4.0)
    assert small
    assert np.isnan(sigma0)
    assert set(laws) == set(LAWS)
    for key, (coherence, r2, n_windows) in laws.items():
        assert np.isnan(coherence)
        assert (r2, n_windows) == LAWS[key][1:]
    assert any(entry.value == pytest.approx(4.0) for entry in ledger.entries)


@pytest.mark.parametrize('swing_deg', [45.0, 60.0, 120.0])
def test_swing_at_or_above_45_deg_keeps_the_coherence(swing_deg):
    (sigma0, laws, small), _ = _gate(swing_deg)
    assert not small
    assert sigma0 == 0.6
    assert laws == LAWS


def test_unknown_swing_changes_nothing():
    ledger = Ledger(thin_band=SETTINGS.confidence_thin_band)
    assert subresonances.swing_gate(0.6, dict(LAWS), np.nan, SETTINGS, ledger) == (0.6, LAWS, False)
