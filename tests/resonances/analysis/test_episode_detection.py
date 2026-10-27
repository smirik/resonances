"""Episode detection on synthetic records, one per mechanism of the author's episode cases (benchmark/
EPISODE_EXPECTATIONS.csv, W23 S3). Every record is a closed form, so the expected episodes follow from the construction,
not from the package:

(i)   a pendulum libration, sigma = pi + 2 arcsin(k sn(omega_0 t, k^2)) with k = sin(A/2), A = 90 deg, T = 4 K(k^2) /
      omega_0 = 1000 yr, plus a ripple of r = 3 deg at 37 yr left in sigma_f. Its turning points are the extremes at
      t = (2n + 1) K / omega_0 (n = 0..9 in 10..9990 yr: 20 of them). The ripple makes extrema of prominence <= 2r
      near the slow turns, below p_TP = MMR_TP_RIPPLE_FACTOR x r = 12 deg: one episode covers the record (2001 HW7, E02:
      three episodes were an artefact of the prominence). At the 1-deg floor the ripple's own loops are turns.
(ii)  two librations about pi (A = 60 deg, P = 1000 yr, 5 cycles each) joined by a circulation at the libration's top
      rate A 2 pi / P for exactly one turn: tau = 2 pi / (A 2 pi / P) = 3 P / pi = 955 yr. The gap is shorter than
      P_l + P_r, but the turning points across it span 2 pi + 2A: two episodes, no union (author 2026-10-19: a
      regime change *without* circulation is not an exit).
(iii) a 1:1 regime change without circulation: centre 60 deg, then 150 deg (A = 25 deg, P = 180 yr, 8 cycles each), the
      angle gliding monotonically from 60 to 150 deg over D yr between a rising centre crossing and the next. The
      episodes end and start at the turns P / 4 before and after the glide: gap = D + P / 2, against P_l + P_r = 2P.
      D = 198 yr: gap / 2P = 0.8 <= LIBRATION_UNION_GAP_PERIODS = 1, a union (469219, 164207, 524522); D = 342 yr:
      1.2, two episodes.
(iv)  a slow drift with a ripple, sigma = 2 pi t / 20 kyr + r sin(2 pi t / P_r): the ripple turns sigma back
      (r 2 pi / P_r above the drift rate), but the centre moves 2 pi P_r / 20 kyr per period, more than r: no episode
      (r = 30 deg at 4 kyr: 72 deg per period; r = 3 deg at 214 yr, the KL9 a-wave shape of E15: 3.9 deg).
(v)   2003 TX9 shape (E26, E27): sigma = pi + A(t) sin(2 pi t / 40 kyr) + 0.1 sin(2 pi t / 13 kyr) + 0.15 sin(2 pi t /
      3.5 kyr), A = 2.6 rad for the first slow cycle, then 0.26 rad (the switch at a zero of the sine), a from the
      identity. The fine scale sees the 3.5-kyr waves (P_cyc = 3.5 kyr), so G2c reads sigma at W_c = 3 x 3.5 =
      10.5 kyr (LIBRATION_CENTRE_CYCLES), where the 13-kyr wave makes loops of a few degrees, below the centre
      prominence p_c = 10 deg. With the centre-scale TP rule (W23 S4, LIBRATION_CENTRE_SCALE_TP_AT_CENTRE) the turns
      at W_c are taken at p_c: one episode of the 40-kyr libration covers the record. Without it the loops count, G2c
      is rejected and the fine episodes are reported.
(vii) G2a, a step of the libration action inside one run (W24, 2003 TX9 3E+2J-10): an exact pendulum about pi with
      A = 140 deg, P = 1000 yr for 3 cycles, then (from the upward centre crossing at 3000 yr, an impulse) A = 15 deg,
      P = 700 yr for 6 cycles. Its turning points are at 250 + 500 n up to 2750 yr and at 3175 + 350 n after; inside
      the window (trimmed by P_c = 200 yr) the geometry sees one run of 17 TPs whose half-cycles are 140 deg (5),
      77.5 deg (the transitional one) and 15 deg (10). The cycle energies are those of the construction, h = sin^2(A/2) (the pendulum's energy over the
      separatrix): 0.883 and 0.0170. G2a cuts at the middle of the transitional half-cycle, (2750 + 3175) / 2 =
      2962.5 yr, both turning points staying with their regimes. Both regimes' periods follow one potential (test 7,
      every disjoint cycle at each shape cutoff 0.075, 0.1, 0.15 x the shorter period, with its turning points; the
      periods 1000 and 700 yr differ, so one forcing frequency does not explain them):
      the pendulum of 140 deg and 1000 yr gives T0 = 1000 / F(140 deg) = 627.2 yr, so 15 deg after the kick takes
      T0 F(15 deg) = 629.9 yr, against 700 observed: T_obs / T_pred = 0.7 F(140) / F(15) = 1.111 (F(A) = 2 K(sin^2(A/2))
      / pi), within 0.2. A double well, V = Omega^2 (-cos s + 0.6 cos 2s - 0.25 sin s) with Omega = 2 pi / 1000 yr (wells at 68
      and 298 deg, h 0 and 0.171, barrier at 349.5 deg, h 0.249), integrated at h = 0.7 across both wells (period 1192 yr)
      and kicked to h = 0.19 in the second well (829 yr), is cut too: V fitted on the large swings holds the small one.
      Not cut: an adiabatic decay of the amplitude (140 -> 15 deg over 10 cycles: neighbouring half-cycles across the
      excluded one differ by e^(P/tau) = 1.25 < 2), a smooth modulation of one pendulum's amplitude, beats of two waves
      (60 and 50 deg at 1000 and 1100 yr) of constant energy, the same step with h unchanged, the same step with a
      swinging as before it (a step of sigma alone, 74729 1J+6S-1), a step after which the period is not the
      potential's (400 yr: 0.4 F(140) / F(15) = 0.635), and the step of a 1:1 angle. Nor, after the W24 review: a
      forced wave whose envelope steps 60 -> 10 deg at one period (isochronous), a small swing off the minimum of V
      (its turning points are not the loop's), a cycle the potential does not give among cycles that pass (the worst
      cycle, not the median, decides).
(vi)  one cycle closed by its phase return: sigma = pi + 40 deg sin(2 pi t / 400 yr - 0.2 pi) on 0-520 yr, a from the
      identity. Two turning points (140 and 340 yr), sigma crosses pi upwards at 0.1 P = 40 yr and 1.1 P = 440 yr with
      the same a and rate: one cycle. With a flat a (a kinematic back-swing) the cycle does not close.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ellipj, ellipk

from resonances.resonance.analysis import geometry as g, pendulum as pendulum_module
from resonances.resonance.analysis.filtering import edge_window, lowpass
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr import analyse as analyse_module
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Confidence
from tests.resonances.analysis.builders import make_sampling

S = settings_from_config()
ONE_DEG = math.radians(1.0)
TWO_PI = 2 * np.pi
IDENTITY = -30.0  # sigma_dot = c (a - a*): the slow variable is rate / c
SAMPLING = make_sampling(
    unwrap_resid_p99_rad=0.0, unwrap_resid_max_rad=0.0, unwrap_ok=True, undefined_phase=False, verdict='resolved', alias_numpy_differs=False
)
MMR_RULES = dict(edge_arc_rule=True, return_closure=True, split_regimes=True, cycle_travel=True)


def filtered(t, sigma, cutoff, rate=None, smooth=True, slow=None, coorbital=False):
    """The angle as the geometry reads it: sigma_f = sigma low-passed at `cutoff` (or sigma itself, `smooth` False, with
    its exact `rate`), the slow variable from the identity (or `slow`)."""
    angle_f = lowpass(t, sigma, cutoff, 2) if smooth else sigma
    rate_f = np.gradient(angle_f, t) if smooth else rate
    return g.FilteredAngle(
        t_yr=t,
        angle_raw=sigma,
        angle_f=angle_f,
        rate_f=rate_f,
        angle_f_half=angle_f,
        angle_f_double=angle_f,
        slow_f=rate_f / IDENTITY if slow is None else slow,
        identity_coefficient=IDENTITY,
        cutoff_yr=cutoff,
        step_yr=float(t[1] - t[0]),
        window=edge_window(t, cutoff),
        coorbital=coorbital,
        **MMR_RULES,
    )


def decided(angle, prominence=ONE_DEG):
    geometry = g.measure_geometry(angle, prominence, S)
    return decide.decide_angle(SAMPLING, False, geometry, None, S, Ledger.from_settings(S)), geometry


def covering(episode):
    return all(g.covers_edges(episode, S))


# --- (i) a pendulum with a ripple below the TP prominence ------------------------------------------------------------


def pendulum(t, amplitude_deg, period_yr):
    """The exact pendulum about pi: sigma, sigma_dot = 2 k omega_0 cn, and the times of its extremes."""
    k = math.sin(math.radians(amplitude_deg) / 2)
    quarter = ellipk(k * k)
    omega_0 = 4 * quarter / period_yr
    sn, cn, _, _ = ellipj(omega_0 * t, k * k)
    extremes = (2 * np.arange(int(t[-1] / period_yr * 2) + 2) + 1) * quarter / omega_0
    return np.pi + 2 * np.arcsin(k * sn), 2 * k * omega_0 * cn, extremes


def test_i_a_ripple_below_the_prominence_leaves_one_covering_episode():
    t = np.arange(0.0, 10_000.5, 0.5)
    sigma, rate, extremes = pendulum(t, 90.0, 1000.0)
    ripple = math.radians(3.0)
    sigma = sigma + ripple * np.sin(TWO_PI * t / 37.0)
    rate = rate + ripple * TWO_PI / 37.0 * np.cos(TWO_PI * t / 37.0)
    angle = filtered(t, sigma, 10.0, rate=rate, smooth=False)
    prominence = g.turning_prominence(ONE_DEG * S.mmr_tp_prominence_floor_deg, S.mmr_tp_ripple_factor, ripple)
    assert prominence == pytest.approx(math.radians(12.0))
    decision, geometry = decided(angle, prominence)
    (episode,) = decide.librating_episodes(geometry, S)
    inside = extremes[(extremes >= angle.window[0]) & (extremes <= angle.window[1])]
    assert len(inside) == 20 and episode.n_tp == 20 and covering(episode)
    assert episode.period_med_yr == pytest.approx(1000.0, abs=2 * 37.0)
    assert math.degrees(episode.half_amp_med_rad) == pytest.approx(90.0, abs=3.0)
    assert (decision.angle_status, decision.subtype) == (AngleStatus.LIBRATION, AngleSubtype.STATIONARY)
    # the ripple is in sigma_f: at the 1-deg floor its loops are turning points and the libration falls apart
    _, fine = decided(angle, ONE_DEG)
    assert len(fine.turning_points.times_yr) > 3 * len(inside)
    assert not any(covering(e) for e in decide.librating_episodes(fine, S))


# --- (ii) a circulation of one turn between two librations ----------------------------------------------------------


def test_ii_one_turn_of_circulation_separates_two_episodes():
    period, amplitude = 1000.0, math.radians(60.0)
    omega = TWO_PI / period
    t1 = 5 * period
    tau = TWO_PI / (amplitude * omega)  # one turn at the libration's top rate
    t2 = t1 + tau
    assert tau == pytest.approx(3 * period / np.pi)
    t = np.arange(0.0, t2 + 5 * period + 0.25, 0.5)
    sigma = np.where(
        t < t1,
        np.pi + amplitude * np.sin(omega * t),
        np.where(t < t2, np.pi + amplitude * omega * (t - t1), 3 * np.pi + amplitude * np.sin(omega * (t - t2))),
    )
    decision, geometry = decided(filtered(t, sigma, 10.0))
    first, second = decide.librating_episodes(geometry, S)
    assert first.t_end_yr < t1 < t2 < second.t_start_yr
    assert (second.t_start_yr - first.t_end_yr) < first.period_med_yr + second.period_med_yr  # not the gap: the turn
    assert decide.librating_union(geometry, S).episodes == ()
    assert decision.angle_status == AngleStatus.EPISODIC


# --- (iii) a 1:1 regime change without circulation ------------------------------------------------------------------


@pytest.mark.parametrize('glide_yr, gap_ratio, joins', [(198.0, 0.8, True), (342.0, 1.2, False)])
def test_iii_a_regime_change_without_circulation_is_a_union_within_the_gap(glide_yr, gap_ratio, joins):
    period, amplitude = 180.0, math.radians(25.0)
    omega = TWO_PI / period
    c1, c2 = math.radians(60.0), math.radians(150.0)
    t1 = 8 * period
    t2 = t1 + glide_yr
    t = np.arange(0.0, t2 + 8 * period + 0.25, 0.5)
    sigma = np.where(
        t < t1,
        c1 + amplitude * np.sin(omega * t),
        np.where(t < t2, c1 + (c2 - c1) * (t - t1) / glide_yr, c2 + amplitude * np.sin(omega * (t - t2))),
    )
    assert (glide_yr + period / 2) / (2 * period) == pytest.approx(gap_ratio)
    decision, geometry = decided(filtered(t, sigma, 10.0, coorbital=True))
    left, right = decide.librating_episodes(geometry, S)
    assert (right.t_start_yr - left.t_end_yr) / (left.period_med_yr + right.period_med_yr) == pytest.approx(gap_ratio, abs=0.01)
    union = decide.librating_union(geometry, S)
    assert (len(union.episodes) == 2) is joins
    assert gap_ratio <= S.libration_union_gap_periods or not joins
    if joins:
        assert (decision.angle_status, decision.subtype, decision.rule) == (AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE, 'G2')
    else:
        assert decision.angle_status == AngleStatus.EPISODIC


# --- (iv) a drift with a ripple -------------------------------------------------------------------------------------


@pytest.mark.parametrize('ripple_deg, ripple_period_yr, step_yr', [(30.0, 4000.0, 10.0), (3.0, 214.0, 1.0)])
def test_iv_a_drift_with_a_ripple_has_no_episode(ripple_deg, ripple_period_yr, step_yr):
    drift = 20_000.0
    t = np.arange(0.0, 100_000.0, step_yr)
    ripple = math.radians(ripple_deg)
    sigma = TWO_PI * t / drift + ripple * np.sin(TWO_PI * t / ripple_period_yr)
    assert ripple * TWO_PI / ripple_period_yr > TWO_PI / drift  # the ripple turns sigma back
    assert TWO_PI * ripple_period_yr / drift > ripple  # the centre moves more than the swing per period
    decision, geometry = decided(filtered(t, sigma, ripple_period_yr / 8))
    assert abs(len(geometry.turning_points.times_yr) - 2 * 100_000.0 / ripple_period_yr) <= 4  # a back-swing per period
    assert decide.librating_episodes(geometry, S) == []
    assert decision.angle_status not in (AngleStatus.LIBRATION, AngleStatus.EPISODIC)


# --- (v) the TX9 shape: loops below the centre prominence at the scale of the centre ---------------------------------

CENTRE_RULE = 'libration_centre_scale_tp_at_centre'  # W23 S4 (LIBRATION_CENTRE_SCALE_TP_AT_CENTRE)
P_SLOW, P_FAST = 40_000.0, 3500.0
N_NEPTUNE = math.radians(2.184902)  # rad / yr


WAVES = ((0.1, 13_000.0), (0.15, P_FAST))  # (half-amplitude [rad], period [yr]) of the fast component of (v)


def tx9_shape() -> MMRSeries:
    """sigma of (v) and a from the identity sigma_dot = k_N n_N + k n(a) (two-body 3N-4, constant varpi)."""
    t = np.arange(0.0, 200_000.0 + 25.0, 50.0)
    slow = TWO_PI / P_SLOW
    amplitude = np.where(t < P_SLOW, 2.6, 0.26)
    sigma = np.pi + amplitude * np.sin(slow * t) + sum(a * np.sin(TWO_PI * t / p) for a, p in WAVES)
    rate = amplitude * slow * np.cos(slow * t) + sum(a * TWO_PI / p * np.cos(TWO_PI * t / p) for a, p in WAVES)
    return identity_series(t, sigma, rate, 'tx9-shape')


def identity_series(t, sigma, rate, name) -> MMRSeries:
    """The 3N-4 record of sigma(t) with a from the identity (constant varpi)."""
    spec = ResonanceSpec.from_key('3N-4+0+1')
    k, k_planet = spec.body_coefficient, spec.planet_coefficients[0]
    n = (rate - k_planet * N_NEPTUNE) / k
    omega, node = np.full_like(t, 0.8), np.full_like(t, 2.1)
    longitude = (sigma - k_planet * N_NEPTUNE * t - spec.body_varpi_coefficient * (omega + node)) / k
    return MMRSeries(
        t_yr=t,
        a=(TWO_PI / n) ** (2 / 3),
        e=np.full_like(t, 0.05),
        inc=np.full_like(t, 0.05),
        Omega=node,
        omega=omega,
        varpi=np.mod(omega + node, TWO_PI),
        angle_wrapped=np.mod(sigma, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=50.0, span_yr=float(t[-1] - t[0]), n_rows=len(t)),
        name=name,
        mean_longitude=np.mod(longitude, TWO_PI),
    )


def with_centre_rule(on: bool):
    """The settings with the centre-scale TP rule on or off."""
    return replace(S, **{CENTRE_RULE: on})


def test_v_at_the_scale_of_the_centre_turns_are_taken_at_the_centre_prominence():
    analysis = analyse_mmr(tx9_shape(), with_centre_rule(True), scan_subresonances=False, scan_owner=False)
    assert analysis.extra_columns.get('geo_centre_scale_cutoff_yr') == pytest.approx(S.libration_centre_cycles * P_FAST, rel=0.02)
    (episode,) = decide.librating_episodes(analysis.geometry, S)
    assert covering(episode)
    assert episode.period_med_yr == pytest.approx(P_SLOW, rel=0.05)
    assert abs((math.degrees(episode.centre_mean_rad) - 180.0 + 180.0) % 360.0 - 180.0) <= 15.0
    assert analysis.verdict.angle_status == AngleStatus.LIBRATION


def test_v_without_the_rule_the_loops_reject_the_centre_scale():
    analysis = analyse_mmr(tx9_shape(), with_centre_rule(False), scan_subresonances=False, scan_owner=False)
    assert np.isnan(analysis.extra_columns.get('geo_centre_scale_cutoff_yr', np.nan))
    assert analysis.geometry.period_cycle_yr == pytest.approx(P_FAST, rel=0.02)  # the fine scale: the 3.5-kyr waves
    episodes = decide.librating_episodes(analysis.geometry, S)
    assert len(episodes) > 1 and not any(covering(e) for e in episodes)
    assert analysis.verdict.angle_status == AngleStatus.EPISODIC


def test_v_a_circulation_with_the_same_waves_does_not_librate_at_any_scale():
    """Control of (v): the centre goes round once per 60 kyr (3.3 turns over the record) under the same waves; the
    waves turn sigma back (0.15 x 2 pi / 3.5 kyr = 0.27 rad/kyr above the drift 0.10 rad/kyr), but no scale holds a
    libration, with the centre-scale TP rule as without it."""
    t = np.arange(0.0, 200_000.0 + 25.0, 50.0)
    drift = TWO_PI / 60_000.0
    sigma = np.pi + drift * t + sum(a * np.sin(TWO_PI * t / p) for a, p in WAVES)
    rate = drift + sum(a * TWO_PI / p * np.cos(TWO_PI * t / p) for a, p in WAVES)
    for on in (True, False):
        analysis = analyse_mmr(
            identity_series(t, sigma, rate, 'circulation'), with_centre_rule(on), scan_subresonances=False, scan_owner=False
        )
        assert np.isnan(analysis.extra_columns.get('geo_centre_scale_cutoff_yr', np.nan))
        assert analysis.verdict.angle_status != AngleStatus.LIBRATION and str(analysis.verdict.status) == 'non_resonant'


# --- (vi) one cycle closed by its phase return ----------------------------------------------------------------------


@pytest.mark.parametrize('flat_a', [False, True])
def test_vi_one_cycle_closes_by_its_phase_return(flat_a):
    period, amplitude = 400.0, math.radians(40.0)
    t = np.arange(0.0, 1.3 * period, 0.5)
    phase = TWO_PI * t / period - 0.2 * np.pi
    sigma, rate = np.pi + amplitude * np.sin(phase), amplitude * TWO_PI / period * np.cos(phase)
    slow = np.zeros_like(t) if flat_a else None
    decision, geometry = decided(filtered(t, sigma, 10.0, rate=rate, smooth=False, slow=slow))
    (run,) = geometry.runs
    assert run.n_tp == 2 and run.tp_times_yr == pytest.approx([0.35 * period, 0.85 * period], abs=1.0)
    if flat_a:  # a does not carry the swing (condition 5): no cycle
        assert not run.closed_by_return and not run.accepted
        return
    assert run.accepted and run.closed_by_return
    assert (run.return_t_a_yr, run.return_t_b_yr) == pytest.approx((0.1 * period, 1.1 * period), abs=1.0)
    assert decision.angle_status == AngleStatus.LIBRATION


# --- (vii) G2a: a step of the libration action inside one run --------------------------------------------------------

P_BEFORE, P_AFTER, A_BEFORE, A_AFTER = 1000.0, 700.0, 140.0, 15.0
T_KICK = 3 * P_BEFORE  # an upward centre crossing of the first pendulum
# the angle's cutoff: the a-locator's baseline tau = 2 P_c = 400 yr >= T0 / pi is aliased (flag jump_locator_aliased),
# as on TX9, so only G2a can cut (at P_c = 10 yr the step of a_f at the kick is itself a confirmed jump of a); the window
# ends at 7200 - 200 yr, so the last regime's turning points run 3175 ... 6675 yr (11)
CUTOFF_STEP = 200.0


def h_of(amplitude_deg):
    """The pendulum's energy over the separatrix energy at half-amplitude A: (1 - cos A) / 2."""
    return math.sin(math.radians(amplitude_deg) / 2) ** 2


def step_record(after_deg=A_AFTER, after_period=P_AFTER):
    """The pendulum of (vii): sigma, its exact rate and the half-amplitude of each sample's regime [deg]."""
    t = np.arange(0.0, T_KICK + 6 * P_AFTER + 0.25, 0.5)
    s1, r1, _ = pendulum(t, A_BEFORE, P_BEFORE)
    s2, r2, _ = pendulum(t - T_KICK, after_deg, after_period)
    before = t < T_KICK
    return t, np.where(before, s1, s2), np.where(before, r1, r2), np.where(before, A_BEFORE, after_deg)


def finished(angle, energy_at, settings=S):
    """`finish_geometry` with the cycle energies of the construction: `energy_at(t)` at each cycle's middle, the cycles
    (TP_i, TP_i+2) of each run as the energy analysis forms them."""
    geometry = g.measure_geometry(angle, ONE_DEG, settings)
    spans = []
    for run in geometry.runs:
        times = run.tp_times_yr
        spans += [(times[i], times[i + 2]) for i in range(0, len(times) - 2, 2)]
    spans = np.array(spans, dtype=float).reshape(-1, 2)
    middles = spans.mean(axis=1)
    return g.finish_geometry(
        geometry,
        angle,
        geometry.period_cycle_yr,
        np.pi,
        1.0,
        0.01,
        middles,
        np.array([energy_at(m) for m in middles]),
        np.nan,
        geometry.period_cycle_yr,
        (),
        settings,
        cycle_spans_yr=spans,
    )


# T_obs / T_pred after the kick of (vii) in the pendulum of the large swings: P_AFTER F(A_BEFORE) / (P_BEFORE F(A_AFTER))
LAW_AFTER = 1.1113328957


def regime_energy(t_mid):
    return h_of(A_BEFORE if t_mid < T_KICK else A_AFTER)


def test_vii_an_impulsive_drop_of_the_amplitude_cuts_the_run_at_the_transitional_half_cycle():
    t, sigma, rate, _ = step_record()
    angle = filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False)
    (whole,) = g.measure_geometry(angle, ONE_DEG, S).runs  # one run before G2a
    assert whole.accepted and whole.n_tp == 17
    geometry = finished(angle, regime_energy)
    (step,) = geometry.action_steps
    assert step.cuts and not step.provisional
    assert (step.t_from_yr, step.t_to_yr, step.time_yr) == pytest.approx((2750.0, 3175.0, 2962.5), abs=0.5)
    assert (step.n_before, step.n_after) == (5, 10) and (step.cycles_before, step.cycles_after) == (2, 5)
    assert math.degrees(step.amp_before_rad) == pytest.approx(A_BEFORE, abs=0.1)
    assert math.degrees(step.amp_after_rad) == pytest.approx(A_AFTER, abs=0.1)
    assert step.separation == pytest.approx(A_BEFORE / A_AFTER, rel=1e-3) and step.ratio == pytest.approx(A_BEFORE / A_AFTER, rel=1e-3)
    assert step.dh == pytest.approx(h_of(A_AFTER) - h_of(A_BEFORE), abs=1e-9)  # -0.866
    assert step.a_ratio > S.libration_action_step_a_ratio
    # test 7: one pendulum; after the kick 1.111 in closed form on every disjoint cycle (2 before, held out, 5 after),
    # at each shape cutoff 0.075, 0.1, 0.15 x 700 yr; the periods differ (700 / 1000: not one forcing frequency)
    assert step.shape_cutoff_yr == pytest.approx(0.1 * P_AFTER) and step.shape_wells == 1 and step.shape_failed == ''
    assert step.law_before == pytest.approx(1.0, abs=0.02) and step.law_after == pytest.approx(LAW_AFTER, abs=0.02)
    assert (step.shape_cycles_before, step.shape_cycles_after, step.shape_held_out) == (2, 5, True)
    assert step.shape_period_ratio == pytest.approx(P_AFTER / P_BEFORE, abs=1e-3) and step.shape_tp_mismatch < 0.1
    assert step.shape_drift_before == pytest.approx(0.0, abs=1e-3) and step.shape_drift_after == pytest.approx(0.0, abs=1e-3)
    (jump,) = [jump for jump in geometry.jumps if not jump.kick]
    assert jump.action_step and jump.zone(geometry.jump_tau_yr) == pytest.approx((2750.0, 3175.0), abs=0.5)
    before, after = [run for run in geometry.runs if run.accepted and run.before_last_jump], geometry.episodes
    assert [(r.n_tp, r.t_start_yr, r.t_end_yr) for r in before] == [(6, pytest.approx(250.0, abs=0.5), pytest.approx(2750.0, abs=0.5))]
    assert [(r.n_tp, r.t_start_yr, r.t_end_yr) for r in after] == [(11, pytest.approx(3175.0, abs=0.5), pytest.approx(6675.0, abs=0.5))]
    assert {'action_step', 'last_regime_after_jump', 'jump_locator_aliased'} <= set(geometry.flags)
    assert 'action_step_provisional' not in geometry.flags
    columns = geometry.to_columns()
    assert columns['geo_action_step_n_cuts'] == 1 and columns['geo_action_step_failed'] == ''
    assert columns['jump_action_step'] == '1'


def test_vii_the_period_of_a_loop_in_whichever_well_holds_it():
    """`pendulum.period_through`: V = -cos s (Omega = 1) at E = -cos A takes 4 K(sin^2(A/2)) (A = 90 deg: 7.4163, the
    deepest well's `period_in_potential` too). V = -cos s + cos 2s: V' = sin s (1 - 4 cos s), wells at cos s = 1/4
    (+-75.52 deg) with V'' = cos s - 4 cos 2s = 1/4 + 7/2 = 3.75, so a small loop in either well takes 2 pi / sqrt(3.75)
    = 3.2446; above the barrier at 0 (V = 0) one loop holds both wells, above V(pi) = 2 nothing librates (nan), and
    where V > E nothing moves (nan)."""
    cosine = pendulum_module.potential_from_coefficients([-1.0], [0.0])
    assert pendulum_module.period_through(cosine, -math.cos(math.pi / 2), 0.0) == pytest.approx(7.416298709205, rel=1e-9)
    assert pendulum_module.period_through(cosine, -math.cos(math.pi / 2), 0.0) == pytest.approx(
        pendulum_module.period_in_potential(cosine, 0.5), rel=1e-9
    )
    double = pendulum_module.potential_from_coefficients([-1.0, 1.0], [0.0, 0.0])
    well = math.acos(0.25)
    bottom = float(double.value(np.array([well]))[0])
    small = 2 * math.pi / math.sqrt(3.75)
    for centre in (well, -well):
        assert pendulum_module.period_through(double, bottom + 1e-6, centre) == pytest.approx(small, rel=1e-4)
    both = pendulum_module.period_through(double, 0.5, well)
    assert both > small and both == pytest.approx(pendulum_module.period_through(double, 0.5, -well), rel=1e-9)
    assert math.isnan(pendulum_module.period_through(double, 2.5, 0.0))  # circulates
    assert math.isnan(pendulum_module.period_through(double, bottom + 1e-6, math.pi))  # V(pi) = 2 > E
    assert pendulum_module.well_count(double) == 2 and pendulum_module.well_count(cosine) == 1


def double_well_record():
    """The double well of (vii), integrated (DOP853, rtol 1e-11) at h = 0.7 across both wells from the deep one (68
    deg), then kicked at the first passage through the second well's bottom (298 deg) after 4 kyr to h = 0.19 inside
    that well: (t, sigma, rate, kick time, V)."""
    omega_sq = (TWO_PI / 1000.0) ** 2
    cos_c, sin_c = [-omega_sq, 0.6 * omega_sq], [-0.25 * omega_sq, 0.0]
    potential = pendulum_module.potential_from_coefficients(cos_c, sin_c)

    def rhs(_, y):
        force = sum(k * (cos_c[k - 1] * math.sin(k * y[0]) - sin_c[k - 1] * math.cos(k * y[0])) for k in (1, 2))
        return [y[1], force]

    def energy(h):
        return potential.v_min + h * potential.depth

    def speed(h, sigma):
        return math.sqrt(2 * (energy(h) - float(potential.value(np.array([sigma]))[0])))

    first = np.arange(0.0, 5000.0, 0.5)
    start = math.radians(68.0)
    s1, r1 = solve_ivp(rhs, (0.0, first[-1]), [start, speed(0.7, start)], t_eval=first, rtol=1e-11, atol=1e-13, method='DOP853').y
    phase = np.mod(s1 - math.radians(298.0) + np.pi, TWO_PI) - np.pi
    kick = next(i for i in range(1, len(first)) if first[i] > 4000.0 and phase[i - 1] * phase[i] <= 0 and abs(phase[i]) < 1)
    second = np.arange(first[kick], first[kick] + 6000.0, 0.5)
    y0 = [s1[kick], math.copysign(speed(0.19, s1[kick]), r1[kick])]
    s2, r2 = solve_ivp(rhs, (second[0], second[-1]), y0, t_eval=second, rtol=1e-11, atol=1e-13, method='DOP853').y
    t = np.concatenate([first[:kick], second])
    return t, np.concatenate([s1[:kick], s2]), np.concatenate([r1[:kick], r2]), float(first[kick]), potential


def test_vii_a_kick_into_the_second_well_of_a_double_well_cuts():
    """The double well of (vii): the large swings (1192 yr across both wells) and, after the kick, the 829-yr loop in
    the second well are one potential at two energies. V fitted on the large swings has two wells and predicts every
    small loop: T_obs / T_pred 0.998, 0.993, 0.970 at C = 0.075, 0.1, 0.15 P_short (1.000 on the exact series, which
    recovers V; 0.911 at 0.2 and the second well lost at 0.3: the filter flattens it), with the loop's turning points;
    the run is cut at the kick at every cutoff of the list."""
    t, sigma, rate, kick, potential = double_well_record()
    assert pendulum_module.well_count(potential) == 2
    small = pendulum_module.period_through(potential, potential.v_min + 0.19 * potential.depth, math.radians(298.0))
    large = pendulum_module.period_through(potential, potential.v_min + 0.7 * potential.depth, math.radians(68.0))
    assert (large, small) == pytest.approx((1192.3, 829.3), abs=0.5)
    angle = filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False)
    geometry = finished(angle, lambda m: 0.7 if m < kick else 0.19)
    (step,) = geometry.action_steps
    assert step.cuts and step.t_from_yr < kick < step.t_to_yr and step.shape_wells == 2 and step.shape_failed == ''
    assert step.shape_cutoff_yr == pytest.approx(0.1 * small, rel=0.01)
    assert step.law_before == pytest.approx(1.0, abs=0.03) and step.law_after == pytest.approx(0.993, abs=0.01)
    laws = dict(item.split(':') for item in step.shape_laws.split(';'))
    assert set(laws) == {'0.075', '0.1', '0.15'}
    for factor, after in (('0.075', 0.998), ('0.1', 0.993), ('0.15', 0.970)):
        assert float(laws[factor].split('/')[1]) == pytest.approx(after, abs=0.01)
    assert step.shape_tp_mismatch < S.libration_action_step_tp_tol and step.shape_held_out
    assert any(jump.action_step for jump in geometry.jumps) and 'action_step' in geometry.flags
    exact = pendulum_module.fit_potential_from_rate(t, sigma, rate, [(step.run_t_start_yr, step.t_from_yr)], S, with_windows=False)[0]
    assert exact.cos_coeffs / (TWO_PI / 1000.0) ** 2 == pytest.approx([-1.0, 0.6], abs=1e-4)


def test_vii_a_smooth_modulation_of_one_pendulum_is_not_split():
    """sigma = pi + A(t) sin(2 pi t / 1000 yr), A from 140 deg down to 15 deg and back over 8 kyr (a cosine in time):
    neighbouring half-cycles never differ by the separation 2 across one excluded half-cycle."""
    t = np.arange(0.0, 16_000.25, 0.5)
    low, high = math.radians(A_AFTER), math.radians(A_BEFORE)
    amplitude = low + (high - low) * (1 + np.cos(TWO_PI * t / 16_000.0)) / 2
    phase = TWO_PI * t / P_BEFORE
    sigma = np.pi + amplitude * np.sin(phase)
    rate = amplitude * TWO_PI / P_BEFORE * np.cos(phase) - (high - low) * TWO_PI / 32_000.0 * np.sin(TWO_PI * t / 16_000.0) * np.sin(phase)
    geometry = finished(filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False), lambda m: h_of(math.degrees(np.interp(m, t, amplitude))))
    assert geometry.action_steps and all(step.failed == g.STEP_SEPARATION for step in geometry.action_steps)
    assert not any(jump.action_step for jump in geometry.jumps) and 'action_step' not in geometry.flags


def test_vii_a_period_the_potential_does_not_give_is_not_split():
    """The step of (vii) with 400 yr after the kick: the pendulum of the large swings gives 629.9 yr at 15 deg, T_obs /
    T_pred = 0.4 F(140) / F(15) = 0.635 (regime_periods), though h, a and the amplitudes step as before."""
    t, sigma, rate, _ = step_record(after_period=400.0)
    geometry = finished(filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False), regime_energy)
    (step,) = geometry.action_steps
    assert step.failed == g.STEP_REGIME_PERIODS and step.law_after == pytest.approx(0.635, abs=0.03)
    assert step.law_before == pytest.approx(1.0, abs=0.02)
    assert not any(jump.action_step for jump in geometry.jumps) and 'action_step' not in geometry.flags
    assert geometry.to_columns()['geo_action_step_failed'] == 'regime_periods'


@pytest.mark.parametrize('lead_yr, observed', [(180.0, False), (300.0, True)])
def test_vii_after_a_cut_the_record_entry_is_still_read(lead_yr, observed):
    """A circulation runs up into the libration of (vii) for `lead_yr` (sigma = pi + rate(0) t, continuous with the
    pendulum at t = 0): it leaves the centre by more than pi before t = -160 yr. 180 yr: the averaged angle travels
    3.5 + 2.4 = 5.8 rad < 1 turn to the first TP, the entry 0.01 P after the record's start is suspected; 300 yr: 8.1
    rad >= 1 turn, observed (the edge rule). After the cut the record's first run lies before the jump and the last
    regime's entry is None: the entry is read on that run as without a cut (`Geometry.record_entry`)."""
    t, sigma, rate, _ = step_record()
    lead = np.arange(-lead_yr, 0.0, 0.5)
    t = np.concatenate([lead, t])
    sigma, rate = np.concatenate([np.pi + rate[0] * lead, sigma]), np.concatenate([np.full_like(lead, rate[0]), rate])
    angle = filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False)
    whole, cut = finished(angle, lambda m: 0.5), finished(angle, regime_energy)
    assert not any(step.cuts for step in whole.action_steps) and [step.cuts for step in cut.action_steps] == [True]
    assert cut.entry is None and whole.entry is not None and whole.record_entry is None
    event = cut.record_entry
    assert event is not None and (event.observed, event.suspected) == (observed, not observed) == (
        whole.entry.observed,
        whole.entry.suspected,
    )
    assert event.raw_arc_rad == pytest.approx(whole.entry.raw_arc_rad, rel=1e-9)
    assert cut.entry_suspected() is (not observed) and ('entry_suspected' in cut.flags) is (not observed)
    first = min((run for run in cut.runs if run.accepted and run.before_last_jump), key=lambda run: run.t_start_yr)
    assert first.entry_observed is observed and g.covers_edges(first, S)[0] is (not observed)
    assert cut.to_columns()['record_entry_suspected'] is (not observed)


def test_vii_one_energy_cycle_on_a_side_makes_the_split_provisional():
    """Two cycles before the step: the side keeps 3 half-cycles of 140 deg and one energy cycle (TP 0-2)."""
    t, sigma, rate, _ = step_record()
    keep = t >= P_BEFORE  # the record starts one cycle later: TPs 1250 ... 2750 before the step
    angle = filtered(t[keep], sigma[keep], CUTOFF_STEP, rate=rate[keep], smooth=False)
    geometry = finished(angle, regime_energy)
    (step,) = geometry.action_steps
    assert step.cuts and step.provisional and (step.n_before, step.cycles_before) == (3, 1)
    assert 'action_step_provisional' in geometry.flags


def test_vii_the_pipeline_reads_two_regimes_of_one_libration():
    """The same shape at 20 kyr -> 14 kyr through `analyse_mmr` (3N-4, a from the identity): the runs before and after
    the step are the two regimes, the word stays a libration (G1l: the first covers the record start and lasts to the
    step), and with the rule off one run covers both."""
    p1, p2 = 20_000.0, 14_000.0
    t = np.arange(0.0, 3 * p1 + 6 * p2 + 25.0, 50.0)
    s1, r1, _ = pendulum(t, A_BEFORE, p1)
    s2, r2, _ = pendulum(t - 3 * p1, A_AFTER, p2)
    sigma, rate = np.where(t < 3 * p1, s1, s2), np.where(t < 3 * p1, r1, r2)
    series = identity_series(t, sigma, rate, 'action-step')
    on = analyse_mmr(series, S, scan_subresonances=False, scan_owner=False)
    (step,) = on.geometry.action_steps
    assert step.cuts and step.time_yr == pytest.approx((3 * p1 - p1 / 4 + 3 * p1 + p2 / 4) / 2, abs=100.0)
    # both regimes in the pendulum of the large swings (test 7: LAW_AFTER in closed form), and the guard keeps the cut
    assert step.law_before == pytest.approx(1.0, abs=0.05) and step.law_after == pytest.approx(LAW_AFTER, abs=0.05)
    assert len(on.geometry.episodes_before_jump) == 1 and len(on.geometry.episodes) == 1
    assert on.verdict.angle_status == AngleStatus.LIBRATION
    off = analyse_mmr(series, replace(S, libration_action_step=False), scan_subresonances=False, scan_owner=False)
    assert not off.geometry.action_steps and len([run for run in off.geometry.runs if run.accepted]) == 1


def pipeline_analyses():
    """The record of `test_vii_the_pipeline_reads_two_regimes_of_one_libration` analysed with G2a on and off."""
    p1, p2 = 20_000.0, 14_000.0
    t = np.arange(0.0, 3 * p1 + 6 * p2 + 25.0, 50.0)
    s1, r1, _ = pendulum(t, A_BEFORE, p1)
    s2, r2, _ = pendulum(t - 3 * p1, A_AFTER, p2)
    series = identity_series(t, np.where(t < 3 * p1, s1, s2), np.where(t < 3 * p1, r1, r2), 'action-step')
    options = dict(scan_subresonances=False, scan_owner=False)
    return analyse_mmr(series, S, **options), analyse_mmr(series, replace(S, libration_action_step=False), **options)


def with_checks(analysis, checks, confidence=None, **fields):
    """A copy of a real analysis with its verdict's checks overridden (and its confidence), its geometry copied."""
    verdict = replace(analysis.verdict, checks={**analysis.verdict.checks, **checks}, decision_path=list(analysis.verdict.decision_path))
    if confidence is not None:
        verdict = replace(verdict, confidence=confidence)
    geometry = replace(analysis.geometry) if analysis.geometry is not None else None
    return replace(analysis, verdict=verdict, geometry=geometry, **fields)


def test_vii_a_failed_r0_the_cut_leaves_unmeasured_withholds_the_cut():
    """The guard has no exemption for R0 any more (W24 review: its law reads each cycle in the well that holds it, so a
    whole record over two energy levels of one potential is a valid R0, not a pooled one): R0 failed on the whole
    record and not measured after the cut, with another confidence, withholds the cut like any guarded check."""
    on, off = pipeline_analyses()
    steps = [step for step in on.geometry.action_steps if step.cuts]
    assert steps and all(step.key == on.resonance.key for step in steps)
    same = {name: on.verdict.checks.get(name) for name in analyse_module.CUT_GUARDED_CHECKS}  # only R0 differs
    cut = with_checks(on, {**same, 'check_mmr_potential': CheckValue.NOT_MEASURED}, confidence=Confidence.LOW)
    whole = with_checks(off, {**same, 'check_mmr_potential': CheckValue.FAIL}, confidence=Confidence.MEDIUM)
    kept = analyse_module._action_step_guard(cut, whole, steps, S)
    assert kept is whole
    (step,) = [step for step in kept.geometry.action_steps if step.failed == g.STEP_WITHHELD]
    assert step.withheld_checks == 'check_mmr_potential' and step.withheld_verdict.endswith('/low')
    assert 'G2a cut withheld' in kept.verdict.decision_path[-1].name


def test_vii_a_derived_row_does_not_borrow_another_rows_step():
    """A derived row (another key) whose analysis cut nothing: R0 failed on its whole record and not measured on the
    cut one, another confidence. The step belongs to the catalogue row, whose verdict the cut does not change: the cut
    stands and the derived row's decision path gets nothing of it."""
    on, off = pipeline_analyses()
    steps = [step for step in on.geometry.action_steps if step.cuts]
    member = replace(on.resonance, key='derived-member-without-a-step')
    child_cut = with_checks(on, {'check_mmr_potential': CheckValue.NOT_MEASURED}, confidence=Confidence.LOW, resonance=member, derived=[])
    child_cut.geometry.action_steps = []
    child_whole = with_checks(off, {'check_mmr_potential': CheckValue.FAIL}, confidence=Confidence.MEDIUM, resonance=member, derived=[])
    parent_cut = with_checks(on, {}, derived=[child_cut])
    parent_whole = with_checks(off, {}, derived=[child_whole])
    parent_whole.verdict = replace(on.verdict, decision_path=list(on.verdict.decision_path))  # the catalogue row: no change
    path = list(child_cut.verdict.decision_path)
    assert analyse_module._action_step_guard(parent_cut, parent_whole, steps, S) is parent_cut
    assert child_cut.verdict.decision_path == path


def test_vii_an_adiabatic_decay_is_not_split():
    period, start, end, cycles = 1000.0, A_BEFORE, A_AFTER, 10
    tau = cycles * period / math.log(start / end)
    t = np.arange(0.0, cycles * period + 0.25, 0.5)
    amplitude = np.radians(start) * np.exp(-t / tau)
    phase = TWO_PI * t / period
    sigma = np.pi + amplitude * np.sin(phase)
    rate = amplitude * (TWO_PI / period * np.cos(phase) - np.sin(phase) / tau)
    angle = filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False)
    geometry = finished(angle, lambda m: h_of(start * math.exp(-m / tau)))
    (step,) = geometry.action_steps
    assert step.failed == g.STEP_SEPARATION and step.separation == pytest.approx(math.exp(period / tau), rel=0.02)
    assert not any(jump.action_step for jump in geometry.jumps) and 'action_step' not in geometry.flags


def test_vii_beats_of_constant_energy_are_not_split():
    t = np.arange(0.0, 22_000.25, 0.5)
    waves = ((math.radians(60.0), 1000.0), (math.radians(50.0), 1100.0))
    sigma = np.pi + sum(a * np.sin(TWO_PI * t / p) for a, p in waves)
    rate = sum(a * TWO_PI / p * np.cos(TWO_PI * t / p) for a, p in waves)
    geometry = finished(filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False), lambda m: 0.5)
    assert geometry.action_steps and not any(step.cuts for step in geometry.action_steps)
    assert not any(jump.action_step for jump in geometry.jumps)


def test_vii_a_step_without_an_energy_change_is_not_split():
    t, sigma, rate, _ = step_record()
    geometry = finished(filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False), lambda m: 0.5)
    (step,) = geometry.action_steps
    assert step.failed == g.STEP_ENERGY and step.dh == pytest.approx(0.0, abs=1e-12)


def test_vii_a_sigma_step_that_a_does_not_show_is_not_split():
    """74729 1J+6S-1: sigma's half-amplitude steps, the swing of a stays (a ratio 1 < 1.5)."""
    t, sigma, rate, _ = step_record()
    slow = 1e-4 * np.sin(TWO_PI * t / P_BEFORE)
    geometry = finished(filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False, slow=slow), regime_energy)
    (step,) = geometry.action_steps
    assert step.failed == g.STEP_A_SWING and step.a_ratio == pytest.approx(1.0, abs=0.1)


def test_vii_a_one_to_one_angle_is_never_split():
    t, sigma, rate, _ = step_record()
    geometry = finished(filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False, coorbital=True), regime_energy)
    assert geometry.action_steps == [] and not any(jump.action_step for jump in geometry.jumps)


# --- (vii) test 7 made robust (W24 review): what one potential must show, and what it must not take ---------------------


def forced_sine_step(offset_deg=0.0):
    """The review's forced wave (Codex, W24): sigma = pi + A sin(2 pi t / 1000 yr), A 60 deg then 10 deg from 3000 yr
    (an envelope switched at a zero of the sine), the small swing moved by `offset_deg` off the centre: one forcing
    frequency, no pendulum."""
    t = np.arange(0.0, 9000.25, 0.5)
    omega = TWO_PI / 1000.0
    amplitude = np.where(t < 3000.0, math.radians(60.0), math.radians(10.0))
    sigma = np.pi + amplitude * np.sin(omega * t) + np.where(t < 3000.0, 0.0, math.radians(offset_deg))
    return t, sigma, amplitude * omega * np.cos(omega * t)


def test_vii_a_forced_wave_whose_envelope_steps_is_not_split():
    """Tests 1-6 pass (the amplitude, a and h step at 3000 yr) and V of the 60-deg swings is nearly harmonic, so it
    predicts the 10-deg period too (law 1.045): but both regimes keep the forcing period, T_small / T_big = 1, and the
    isochrony null explains them as well (in a nearly harmonic V every loop has one period): not split."""
    t, sigma, rate = forced_sine_step()
    geometry = finished(filtered(t, sigma, 200.0, rate=rate, smooth=False), lambda m: h_of(60.0 if m < 3000.0 else 10.0))
    (step,) = geometry.action_steps
    assert step.failed == g.STEP_REGIME_PERIODS and step.shape_failed == g.T7_ISOCHRONOUS
    assert step.shape_period_ratio == pytest.approx(1.0, abs=1e-3)
    assert abs(step.law_after - 1) <= S.libration_action_step_period_tol  # the law alone would have passed
    assert not any(jump.action_step for jump in geometry.jumps) and 'action_step' not in geometry.flags


@pytest.mark.parametrize('offset_deg', [20.0, 40.0])
def test_vii_a_small_swing_off_the_minimum_of_the_potential_is_not_split(offset_deg):
    """The step of (vii) (140 deg / 1000 yr, then 15 deg / 700 yr: periods that differ) with the small swing about
    pi + offset instead of pi, inside the large swing's range: no loop of V (well at pi) has turning points there. The
    loop of its energy is centred on pi, so its extremes miss the observed ones by about the offset over the 15-deg
    half-swing (>= 20/15 - 0.2 > 1)."""
    t, sigma, rate, _ = step_record()
    sigma = sigma + np.where(t < T_KICK, 0.0, math.radians(offset_deg))
    geometry = finished(filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False), regime_energy)
    (step,) = geometry.action_steps
    assert step.failed == g.STEP_REGIME_PERIODS and step.shape_failed.split('@')[0] in (g.T7_LAW, g.T7_TURNING_POINTS)
    assert step.shape_tp_mismatch > offset_deg / A_AFTER - S.libration_action_step_tp_tol
    assert not any(jump.action_step for jump in geometry.jumps)


def test_vii_a_cycle_the_potential_does_not_give_vetoes_whatever_the_median():
    """The step of (vii) whose last 700 yr swing at 350 yr instead of 700 (law 0.56: V of the large swings gives 630
    yr): most cycles after the kick pass (1.11), the median would; the worst cycle fails."""
    t, sigma, rate, _ = step_record()
    late = T_KICK + 5 * P_AFTER  # an upward centre crossing: the last 700 yr swing at 350 yr
    s3, r3, _ = pendulum(t - late, A_AFTER, 350.0)
    sigma, rate = np.where(t < late, sigma, s3), np.where(t < late, rate, r3)
    geometry = finished(filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False), regime_energy)
    (step,) = geometry.action_steps
    assert step.failed == g.STEP_REGIME_PERIODS and step.shape_failed.split('@')[0] == g.T7_LAW
    assert step.law_after < 1 - S.libration_action_step_period_tol


def test_vii_one_disjoint_cycle_of_the_larger_swings_reads_in_sample_and_is_provisional():
    """With one disjoint cycle of 140 deg (the record starting at 1000 yr: TPs 1250 ... 2750) V is fitted on the cycle
    it then reads: the law is in sample (held_out False), the cut provisional."""
    t, sigma, rate, _ = step_record()
    keep = t >= P_BEFORE
    geometry = finished(filtered(t[keep], sigma[keep], CUTOFF_STEP, rate=rate[keep], smooth=False), regime_energy)
    (step,) = geometry.action_steps
    assert step.cuts and step.provisional and not step.shape_held_out and step.shape_cycles_before == 1


def test_vii_period_through_brackets_turning_points_between_the_extrema():
    """`pendulum.loop_through` finds each turning point between two extrema of V (`pendulum.extrema`, the roots of a
    polynomial in e^{i sigma}): near a separatrix, where V = E only in a hair's breadth around the top, a 0.5-deg grid
    saw none (nan). V = -cos s at E = 1 - 1e-8: the exact 4 K(m), m = (1 + E) / 2 (mpmath, 40 digits) is 43.77283;
    E = 1 is the separatrix (inf). V = -cos s + cos 2s at E = -1e-7, just below its inner barrier V(0) = 0: the loop
    of the well at +75.5 deg stops at s = acos((1 + sqrt(9 + 8E)) / 4) = 2.582e-4 rad (the barrier, not the other
    well) and at acos((1 - sqrt(9 + 8E)) / 4) = 2 pi / 3, its period 12.5753 by an mpmath quadrature."""
    mp = pytest.importorskip('mpmath')
    mp.mp.dps = 40
    cosine = pendulum_module.potential_from_coefficients([-1.0], [0.0])
    energy = 1 - 1e-8
    exact = float(4 * mp.ellipk((1 + mp.mpf(energy)) / 2))
    assert exact == pytest.approx(43.77283, abs=1e-5)
    assert pendulum_module.period_through(cosine, energy, 0.00123) == pytest.approx(exact, rel=1e-8)
    assert pendulum_module.loop_through(cosine, 1.0, 0.3).reason == pendulum_module.LOOP_SEPARATRIX
    assert math.isinf(pendulum_module.period_through(cosine, 1.0, 0.3))
    double = pendulum_module.potential_from_coefficients([-1.0, 1.0], [0.0, 0.0])
    energy = -1e-7
    e = mp.mpf(energy)
    lower, upper = mp.acos((1 + mp.sqrt(9 + 8 * e)) / 4), mp.acos((1 - mp.sqrt(9 + 8 * e)) / 4)
    period = 2 * mp.quad(lambda s: 1 / mp.sqrt(2 * (e + mp.cos(s) - mp.cos(2 * s))), [lower, mp.mpf('0.01'), upper])
    loop = pendulum_module.loop_through(double, energy, math.acos(0.25))
    assert (loop.lower_rad, loop.upper_rad) == pytest.approx((float(lower), float(upper)), rel=1e-9)
    assert loop.period_yr == pytest.approx(float(mp.re(period)), rel=1e-7) and float(lower) == pytest.approx(2.582e-4, rel=1e-3)
    extrema = pendulum_module.extrema(double)
    assert np.degrees(extrema.sigma_rad[extrema.is_min]) == pytest.approx([75.5225, 284.4775], abs=1e-3)


def test_r0_reads_a_cycle_in_the_well_that_holds_it():
    """R0's period law (`pendulum.cycle_energies`, `cycle_period`) in V = -cos s + 0.6 cos 2s - 0.1 cos 3s - 0.25 sin s
    (Omega = 1): wells at 72.3 deg (h 0) and 296.0 deg (h 0.175), barrier between them at h 0.192. A cycle integrated
    at h = 0.18 in the second well takes 7.401 yr; the deepest well's loop at that energy (the law before W24) 8.465:
    the cycle's law is 1.000 in its own well, 0.874 in the deepest. With two harmonics every loop of one energy has one
    period, whichever well holds it (t = tan(s/2) makes T an integral over the two real ovals of one elliptic curve):
    the double well of (vii) at h = 0.19, 829.30 yr in either well. With one well the loop around the well holding the
    cycle is the deepest well's, from any fastest sample: the law is unchanged to 1e-12."""
    cos_c, sin_c = [-1.0, 0.6, -0.1], [-0.25, 0.0, 0.0]
    potential = pendulum_module.potential_from_coefficients(cos_c, sin_c)
    found = pendulum_module.extrema(potential)
    assert np.degrees(found.sigma_rad[found.is_min]) == pytest.approx([72.3, 296.0], abs=0.1)

    def rhs(_, y):
        return [y[1], sum(k * (cos_c[k - 1] * math.sin(k * y[0]) - sin_c[k - 1] * math.cos(k * y[0])) for k in (1, 2, 3))]

    well = float(found.sigma_rad[found.is_min][1])
    energy = potential.v_min + 0.18 * potential.depth
    start = [well, math.sqrt(2 * (energy - float(potential.value(np.array([well]))[0])))]
    t = np.arange(0.0, 30.0, 0.001)
    angle, speed = solve_ivp(rhs, (0.0, t[-1]), start, t_eval=t, rtol=1e-12, atol=1e-14, method='DOP853').y
    tops = np.flatnonzero(np.diff(np.sign(speed)) < 0)  # maxima of sigma: one cycle between two of them
    bounds = [(float(t[tops[0]]), float(t[tops[1]]), False)]
    (cycle,) = pendulum_module.cycle_energies(t, angle, speed, bounds, potential)
    assert cycle.h == pytest.approx(0.18, abs=1e-6) and bounds[0][1] - bounds[0][0] == pytest.approx(7.4007, abs=2e-3)
    assert cycle.period_law_ratio == pytest.approx(1.0, abs=5e-4)
    deepest = pendulum_module.period_in_potential(potential, cycle.h)
    assert deepest == pytest.approx(8.4653, abs=1e-3)
    two = pendulum_module.potential_from_coefficients([-1.0, 0.6], [-0.25, 0.0])
    loops = [pendulum_module.period_through(two, two.v_min + 0.19 * two.depth, math.radians(x)) for x in (68.0, 298.0)]
    assert loops[0] == pytest.approx(loops[1], rel=1e-10) and loops[0] * 1000.0 / TWO_PI == pytest.approx(829.30, abs=0.01)
    single = pendulum_module.potential_from_coefficients([-1.0, -0.2], [0.1, 0.0])
    assert pendulum_module.well_count(single) == 1
    for h in (0.05, 0.5, 0.95):
        for fastest in (single.well_rad, single.well_rad + 1.0, single.well_rad - 2.0):
            assert pendulum_module.cycle_period(single, h, fastest) == pytest.approx(
                pendulum_module.period_in_potential(single, h), rel=1e-12
            )


def test_vii_a_cycle_the_potential_cannot_hold_has_no_loop():
    """A 15-deg swing of 100 yr in V = -1e-6 cos s (depth 2e-6 rad^2/yr^2): its mean sigma_dot^2 / 2 = 3.4e-5 lies above
    the top of V, so no loop holds it (the orbit would circulate): law and turning-point mismatch inf, a veto."""
    potential = pendulum_module.potential_from_coefficients([-1e-6], [0.0])
    t = np.arange(0.0, 100.0, 0.05)
    sigma = math.radians(15.0) * np.sin(TWO_PI * t / 100.0)
    rate = math.radians(15.0) * TWO_PI / 100.0 * np.cos(TWO_PI * t / 100.0)
    fit = g._cycle_fit(t, sigma, rate, potential, (0.0, 99.95))
    assert fit.energy > potential.v_max and math.isinf(fit.law) and math.isinf(fit.tp_mismatch)
