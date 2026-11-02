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
      D = 198, 342 and 1710 yr (gap / 2P = 0.8, 1.2, 5): a union (469219, 164207, 524522) however long the glide, since
      sigma never leaves the band about the centre interpolated from 60 to 150 deg (A7, W28 g06-5: the length of the
      gap, LIBRATION_UNION_GAP_PERIODS, no longer decides).
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
(viii) rule q, a half-cycle over the top in sigma but not in psi (W28 g11-2, 82831 13M-24): psi = sigma - c(t)
      swings +-60 deg about a well moving at c_dot = 60 deg/kyr, half-periods of 500 yr (24 TPs, 0.25-11.75 kyr), except
      one half-cycle near the separatrix, psi from -170 to +170 deg (TPs 10 and 11, 5.25 and 5.75 kyr). In sigma that
      half-cycle rises 340 + 30 = 370 deg, a half-amplitude of 185 deg >= pi: `split_runs` breaks there. In psi its
      half-amplitude is 170 deg < pi, so the two runs are one libration about the moving centre (the well moves 720 deg
      over the record, so sigma's bands fail and psi accepts). A true crossing, psi from -170 to +420 deg (295 deg >= pi)
      and then a libration about 360 deg, stays two episodes; so does a track the full-cycle centres do not follow.
(x)   A7, the bounded gap (W28 g06-5, 2001 WK15 32E-39, 2002 XA 5J-2): pi + 60 deg sin(2 pi t / 1000 yr), then
      n periods of a 2-deg swing about pi (below the 12-deg TP prominence: the turning points vanish), then the same
      libration: gap = (n + 1/2) P, 1.75 to 12.25 (P_l + P_r) for n = 3..24, and sigma_f stays within A of pi: one
      union. An excursion to pi + 200 deg over the top and back (turning points 0, 200, -40, 0 deg: no 2 pi in the
      gap) overshoots the band by > 100 deg: two episodes.
(ix)  rule s, a stall and its leftover (W28 g11-3, 385266 16N-27): A = 30 deg, P = 4.4 kyr; a centre drifting 79 deg
      per period for six cycles (1.1-23.1 kyr: the run's centre moves), then held for nine (25.3-67.1 kyr, the stall),
      then two cycles about c + 60 deg swinging 60 deg (69.3-73.7 kyr): their centre steps away faster than a stall,
      so the stall cut leaves them a piece of their own, but their bands [c, c + 120] share [c, c + 30] with the
      stall's [c - 30, c + 30] and their period is the stall's: one episode 25.3-73.7 kyr. A leftover of ripple, 8 TPs
      of +-5 deg about c + 5 deg at 300-yr half-periods (P = 0.6 kyr, 7.3x shorter, outside
      LIBRATION_REGULAR_PERIOD_RATIO = 2; no cycle with two of its TPs has the stall's period), passes the band test
      with the stall but is another oscillation: not joined.
(xii) C4, rule coorbital_edge_turn (W28 g02-2 A, 2006 BZ147): an Earth horseshoe of the averaged circular problem
      (M&D eqs. 3.178-3.179) turning 6 deg from the planet, integrated by solve_ivp (turns every 85.9 yr), recorded from
      100 yr to 10 yr after its third turn and filtered at 20 yr: sigma_f turns twice in the window, the third
      reflection lies in the trim and closes the cycle when read on the orbit-averaged raw angle. Not added: the same
      record circulating past the planet after L3 with a 30-deg ripple at the 1-yr orbit (the raw angle turns back
      every orbit, its 3-orbit average does not), and a kick that leaves |a - a_P| at 0.4 of its mirror (out/in 0.4).
(xiii) C7, rule coorbital_well_cut (W28 g03-1 A, g03-9 A): a 1:1 episode ends only at a separatrix, the wells read
      on the averaged Earth problem (circular: L4/L5 at 60/300 deg, M&D; a body of e = 0.1: a quasi-satellite well at
      0). A tadpole whose amplitude drops 5x at an impulse (40 -> 8 deg) while the forced centre drifts 18 deg is one
      episode with an amplitude_change mark on the transitional half-cycle 750-850 yr (bands fail, every swing holds
      L4 within 1 deg); a quasi-satellite gliding into a tadpole without a turn is one episode, wells qs > compound >
      tadpole_L4, with well_change marks at the two turning points of the glide's half-cycle; one turn of circulation between two
      quasi-satellites is a crossing: two episodes; a ripple drifting 60 deg past L5 holds no minimum: no episode.
(xv)  C11, episodes.csv eligible / ineligible_reason (W28 g02-9 A): a 1:1 tadpole of 30 deg at P = 100 yr with the
      angle's floor set by hand to 150 yr is an accepted run the status does not count (below_period_floor, no
      libration), counted at a 50-yr floor; a forced wave marked after the verdict reads forced_wave; the G2a step of
      (vii) gives before_last_jump to the run before the cut (P = 1000 yr) and, at a floor of 800 yr, below_period_floor
      to the one after it (P = 700 yr).
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


@pytest.mark.parametrize('glide_yr, gap_ratio', [(198.0, 0.8), (342.0, 1.2), (1710.0, 5.0)])
def test_iii_a_regime_change_without_circulation_is_a_union_however_long(glide_yr, gap_ratio):
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
    assert len(union.episodes) == 2
    assert math.degrees(union.band_overshoot_rad) < S.libration_union_band_margin_deg
    assert (decision.angle_status, decision.subtype, decision.rule) == (AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE, 'G2')


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


def test_vii_period_through_brackets_turning_points_between_the_extrema(monkeypatch):
    """`pendulum.loop_through` finds each turning point between two extrema of V (`pendulum.extrema`, the roots of a
    polynomial in e^{i sigma}): near a separatrix, where V = E only in a hair's breadth around the top, a 0.5-deg grid
    saw none (nan). V = -cos s at E = 1 - 1e-8: the exact 4 K(m), m = (1 + E) / 2 (mpmath, 40 digits) is 43.77283;
    E = 1 is the separatrix (inf). V = -cos s + cos 2s at E = -1e-7, just below its inner barrier V(0) = 0: the loop
    of the well at +75.5 deg stops at s = acos((1 + sqrt(9 + 8E)) / 4) = 2.582e-4 rad (the barrier, not the other
    well) and at acos((1 - sqrt(9 + 8E)) / 4) = 2 pi / 3, its period 12.5753 by an mpmath quadrature."""
    mp = pytest.importorskip('mpmath')
    monkeypatch.setattr(mp.mp, 'dps', 40)  # restored after the test: a global 40 broke the 30-digit roots of other modules
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


# --- (viii) rule q: over the top in sigma, not in psi; (ix) rule s: a stall and its leftover --------------------------


def through(times, values_deg, t):
    """sigma(t) through the turning points (times [yr], values [deg]): a half-cosine between consecutive ones, held
    beyond the last [rad]. Every given point is an extremum."""
    times, values = np.asarray(times, float), np.radians(np.asarray(values_deg, float))
    times = np.concatenate([[2 * times[0] - times[1]], times, [2 * times[-1] - times[-2]]])
    values = np.concatenate([[values[1]], values, [values[-2]]])
    out = np.full(t.shape, values[0])
    for (t0, v0), (t1, v1) in zip(zip(times, values), zip(times[1:], values[1:])):
        inside = (t >= t0) & (t <= t1)
        out[inside] = v0 + (v1 - v0) * (1 - np.cos(np.pi * (t[inside] - t0) / (t1 - t0))) / 2
    out[t > times[-1]] = values[-1]
    return out


WELL_RATE = math.radians(60.0) / 1000.0  # c_dot of (viii) [rad/yr]


class TrackModel:
    """A geometry.WellModel: the track c(t) = rate t on sigma's branch (the turn nearest to the run's mean psi)."""

    def __init__(self, t, rate):
        self.well = g.MovingWell(g.WELL_OK, t, rate * t)

    def __call__(self, times, values):
        branch = TWO_PI * np.round(np.mean(values - self.well.at(times)) / TWO_PI)
        return replace(self.well, centre_rad=self.well.centre_rad + branch)


def moving_well_record(top_deg, after_deg=0.0):
    """(viii): turning points of psi = +-60 deg every 500 yr, TPs 10 and 11 at -170 and `top_deg`, every later TP
    shifted by `after_deg`; sigma = c(t) + psi. Returns t and the turning points of sigma at a 1-deg prominence."""
    times = 250.0 + 500.0 * np.arange(24)
    psi = np.array([60.0 if i % 2 else -60.0 for i in range(24)])
    psi[10], psi[11] = -170.0, top_deg
    psi[12:] += after_deg
    t = np.arange(0.0, times[-1] + 250.0, 1.0)
    sigma = through(times, psi + math.degrees(WELL_RATE) * times, t)
    return t, g.find_turning_points(t, sigma, ONE_DEG, (t[0], t[-1]))


def joined_runs(t, tp, model, on=True):
    fired = set()
    record = (float(t[0]), float(t[-1]))
    runs = g._find_episodes(tp, record, record, replace(S, libration_frame_join=on), well_model=model, fired=fired)
    return [(round(run.t_start_yr), round(run.t_end_yr)) for run in runs if run.accepted], fired


def test_viii_over_the_top_in_sigma_but_not_in_psi_is_one_episode():
    t, tp = moving_well_record(170.0)
    halves = np.degrees(np.abs(np.diff(tp.values_rad)) / 2)
    assert len(tp) == 24 and halves.max() == pytest.approx(185.0, abs=0.5)  # the half-cycle that breaks the run
    episodes, fired = joined_runs(t, tp, TrackModel(t, WELL_RATE))
    assert episodes == [(250, 11750)] and fired == {g.rules.Rule.FRAME_JOIN}
    run = g._find_episodes(tp, (t[0], t[-1]), (t[0], t[-1]), S, well_model=TrackModel(t, WELL_RATE))[0]
    assert run.frame == g.FRAME_PSI
    # off: the over-the-top half-cycle breaks the run, each side librates in psi on its own
    assert joined_runs(t, tp, TrackModel(t, WELL_RATE), on=False) == ([(250, 5250), (5750, 11750)], set())


def test_viii_a_crossing_of_the_separatrix_in_psi_stays_a_break():
    t, tp = moving_well_record(420.0, after_deg=360.0)
    episodes, fired = joined_runs(t, tp, TrackModel(t, WELL_RATE))
    assert episodes == [(250, 5250), (5750, 11750)] and not fired


def test_viii_a_track_the_centres_do_not_follow_joins_nothing():
    t, tp = moving_well_record(170.0)
    _, fired = joined_runs(t, tp, TrackModel(t, -WELL_RATE))  # the well run the wrong way
    assert not fired


def stall_angle(leftover):
    """(ix): drift, stall and `leftover` ('same' or 'ripple') as an MMR angle (sigma_f = sigma, the regime cuts on)."""
    times, values, centre, sign = [], [], 0.0, -1.0
    for i in range(12):
        times.append(1100.0 + 2200.0 * i)
        values.append(centre + sign * 30.0)
        sign, centre = -sign, centre + 79.0 / 2
    for _ in range(18):
        times.append(times[-1] + 2200.0)
        values.append(centre + sign * 30.0)
        sign = -sign
    step, offset, swing, count = (2200.0, 60.0, 60.0, 4) if leftover == 'same' else (300.0, 5.0, 5.0, 8)
    for _ in range(count):
        times.append(times[-1] + step)
        values.append(centre + offset + sign * swing)
        sign = -sign
    t = np.arange(0.0, times[-1] + 1100.0, 5.0)
    sigma = through(times, values, t)
    return filtered(t, sigma, 10.0, rate=np.gradient(sigma, t), smooth=False, slow=np.zeros_like(t))


def stall_runs(leftover, on=True, marks=True):
    settings = replace(S, libration_stall_join=on, libration_regime_cut_marks=marks)
    geometry = g.measure_geometry(stall_angle(leftover), ONE_DEG, settings)
    accepted = [(round(run.t_start_yr / 100) / 10, round(run.t_end_yr / 100) / 10) for run in geometry.runs if run.accepted]
    return accepted, set(geometry.rules_fired)


def test_ix_a_stall_and_a_leftover_of_its_own_regime_are_one_episode():
    assert stall_runs('same') == ([(25.3, 73.7)], {g.rules.Rule.STALL_JOIN})
    assert stall_runs('same', on=False, marks=False) == ([(25.3, 67.1), (69.3, 73.7)], set())
    # without rule s the accepted leftover still shares the stall's band and period: rule regime_cut_mark (W28 C16)
    # joins it, with the stall as a mark
    assert stall_runs('same', on=False) == ([(25.3, 73.7)], {g.rules.Rule.REGIME_CUT_MARK})


def test_ix_a_leftover_of_another_period_is_not_joined():
    accepted, fired = stall_runs('ripple')
    assert accepted == [(25.3, 65.2), (65.5, 67.3)] and not fired
    # the band test alone would join it: the period of the stall's own oscillation is what refuses the ripple
    angle = stall_angle('ripple')
    geometry = g.measure_geometry(angle, ONE_DEG, S)
    tp, run = geometry.turning_points, g.split_runs(geometry.turning_points, S)[0]
    cut = g.split_at_regime_changes(tp, run, S)
    assert cut.stall and [len(piece) for piece in cut.pieces] == [11, 20, 7]
    joined = np.concatenate(cut.pieces[1:])
    record = (float(angle.t_yr[0]), float(angle.t_yr[-1]))
    assert g._classify_run(0, tp, joined, record, S).accepted
    assert not g._same_period(np.asarray(tp.times_yr)[joined], 20, False, S)


@pytest.mark.parametrize('step_yr, joins', [(550.0, True), (162.0, False), (300.0, True), (250.0, False)])
def test_ix_a_leftover_of_one_turning_point_is_read_by_its_half_cycle(step_yr, joins):
    """A stall of 1100-yr cycles (TPs every 550 yr) and one leftover TP `step_yr` after it: its period is 2 x step_yr,
    within LIBRATION_REGULAR_PERIOD_RATIO = 2 of 1100 yr for 550 and 300 yr (1100, 600), not for 162 and 250 yr (324,
    500 < 550). Before the stall the same holds."""
    stall = 550.0 * np.arange(9)
    assert g._same_period(np.append(stall, stall[-1] + step_yr), 9, False, S) is joins
    assert g._same_period(np.concatenate([[-step_yr], stall]), 9, True, S) is joins


# --- (x) A7: a bounded gap joins two episodes however long it is; an excursion over the top in it does not ------------

P_GAP, A_GAP_DEG, QUIET_DEG = 1000.0, 60.0, 2.0
TP_PROMINENCE = math.radians(12.0)


def half_cosine(t, t0, t1, v0, v1):
    """From v0 at t0 to v1 at t1 with zero slope at both ends [rad]."""
    return v0 + (v1 - v0) * (1 - np.cos(np.pi * (t - t0) / (t1 - t0))) / 2


def gap_record(n_gap, excursion_deg=None):
    """sigma = pi + A sin(2 pi t / P) up to t1 = 5.5 P (a descending centre crossing), then n_gap periods of a 2-deg
    swing about pi (below the 12-deg TP prominence: the turning points vanish), then 6 cycles of
    pi - A sin(2 pi (t - t2) / P) from t2 = t1 + n_gap P. The last turning point before the gap is the maximum at
    5.25 P, the first after it the minimum at t2 + P / 4: gap = (n_gap + 1/2) P, against P_left + P_right = 2 P.

    With `excursion_deg` (E) the gap is an excursion over the top instead (half-cosines): sigma rises from pi to
    pi + E in 2 P, falls to pi - 40 deg in P / 2 and comes back to pi in 3 P (n_gap = 5.5), then the second libration
    starts downwards. Its turning points: the left run ends at the minimum pi at t1 (P / 4 after its last maximum),
    pi + E and pi - 40 deg are a run of two (500 yr apart, 2000 and 3000 yr from their neighbours: more than
    LIBRATION_EPISODE_GAP_FACTOR = 3 x 500-yr half-periods), and the right run starts with the maximum pi at t2."""
    omega, amplitude = TWO_PI / P_GAP, math.radians(A_GAP_DEG)
    t1 = 5.5 * P_GAP
    t2 = t1 + n_gap * P_GAP
    t = np.arange(0.0, t2 + 6 * P_GAP + 0.25, 1.0)
    if excursion_deg is None:
        middle = np.pi - math.radians(QUIET_DEG) * np.sin(omega * (t - t1))
    else:
        top, low = t1 + 2 * P_GAP, t1 + 2.5 * P_GAP
        excursion, back = math.radians(excursion_deg), math.radians(-40.0)
        middle = (
            np.where(
                t < top,
                half_cosine(t, t1, top, 0.0, excursion),
                np.where(t < low, half_cosine(t, top, low, excursion, back), half_cosine(t, low, t2, back, 0.0)),
            )
            + np.pi
        )
    sigma = np.where(t < t1, np.pi + amplitude * np.sin(omega * t), np.where(t < t2, middle, np.pi - amplitude * np.sin(omega * (t - t2))))
    return filtered(t, sigma, 10.0)


@pytest.mark.parametrize('n_gap', [3, 8, 24])
def test_x_turning_points_vanishing_inside_the_band_leave_one_union(n_gap):
    decision, geometry = decided(gap_record(n_gap), TP_PROMINENCE)
    left, right = decide.librating_episodes(geometry, S)
    assert left.t_end_yr == pytest.approx(5.25 * P_GAP, abs=5.0)
    assert right.t_start_yr == pytest.approx((5.75 + n_gap) * P_GAP, abs=5.0)
    gap_p = (right.t_start_yr - left.t_end_yr) / (left.period_med_yr + right.period_med_yr)
    assert gap_p == pytest.approx((n_gap + 0.5) / 2, abs=0.02)  # 1.75 to 12.25: the gap length never decides
    # sigma_f stays inside the band: |sigma_f - c| <= A, so its swing beyond the neighbours' amplitude is <= 0
    overshoot = g.gap_band_overshoot(geometry, left, right)
    assert math.degrees(overshoot) <= 0.5 < S.libration_union_band_margin_deg
    union = decide.librating_union(geometry, S)
    assert union.episodes == (left, right)
    assert union.band_overshoot_rad == pytest.approx(overshoot)
    assert g.rules.Rule.BOUNDED_GAP_MERGE in geometry.rules_fired
    assert (decision.angle_status, decision.subtype, decision.rule) == (
        AngleStatus.LIBRATION,
        AngleSubtype.REGIME_CHANGE,
        decide.RULE_UNION,
    )


def test_x_an_excursion_over_the_top_in_the_gap_is_no_union():
    # pi + 200 deg is beyond the top of a pendulum about pi; sigma comes back without a full turn, so neither range
    # reaches 2 pi (turning points 0, 200, -40, 0 deg: 240 deg): the band alone refuses the join
    decision, geometry = decided(gap_record(5.5, excursion_deg=200.0), TP_PROMINENCE)
    left, right = decide.librating_episodes(geometry, S)
    assert left.t_end_yr == pytest.approx(5.5 * P_GAP, abs=5.0) and right.t_start_yr == pytest.approx(11.0 * P_GAP, abs=5.0)
    assert math.degrees(max(left.half_amp_max_rad, right.half_amp_max_rad)) == pytest.approx(A_GAP_DEG, abs=1.0)
    assert math.degrees(g.break_range(geometry, left, right)) == pytest.approx(240.0, abs=1.0)
    assert geometry.gap_raw_range(left, right) < TWO_PI
    assert math.degrees(g.gap_band_overshoot(geometry, left, right)) > 3 * S.libration_union_band_margin_deg
    assert decide.librating_union(geometry, S).episodes == ()
    assert g.rules.Rule.BOUNDED_GAP_MERGE not in geometry.rules_fired
    assert (decision.angle_status, decision.subtype) == (AngleStatus.EPISODIC, AngleSubtype.RECURRENT)
    # a gap as long with sigma inside the band joins
    assert len(decide.librating_union(decided(gap_record(5.5), TP_PROMINENCE)[1], S).episodes) == 2


def test_x_without_sigma_f_nothing_shows_the_angle_stayed():
    _, geometry = decided(gap_record(3), TP_PROMINENCE)
    left, right = decide.librating_episodes(geometry, S)
    assert math.isnan(g.gap_band_overshoot(replace(geometry, angle_f=None), left, right))
    assert decide.librating_union(replace(geometry, angle_f=None), S).episodes == ()


def with_band(episode, centre_deg, amplitude_deg):
    """`episode` with its turning points alternating about `centre_deg` by +-`amplitude_deg` (its edge centres are then
    centre_deg) and half_amp_max = amplitude_deg."""
    signs = np.where(np.arange(len(episode.tp_values_rad)) % 2, 1.0, -1.0)
    values = np.radians(centre_deg + signs * amplitude_deg)
    return replace(episode, tp_values_rad=values, half_amp_max_rad=math.radians(amplitude_deg))


@pytest.mark.parametrize(
    'gap_sigma, expected_deg',
    [
        # the step 0 -> 90 deg along the carried centre: |sigma - c(t)| = 0, 60 deg inside max(A) = 60 deg
        ('glide', -60.0),
        # held at -20 deg: inside the left band (|-20 - 0| - 30 = -10 deg) though 110 - 60 = 50 deg off the carried
        # centre at the right edge (the swing back of 524522 1V-1)
        (-20.0, -10.0),
        # held at 170 deg: 80 - 60 = 20 deg beyond the right band (and the carried one at the right edge), 140 deg beyond
        # the left one: the gap leaves the band
        (170.0, 20.0),
    ],
)
def test_x_the_gap_is_read_against_both_bands_and_the_step_between_them(gap_sigma, expected_deg):
    _, geometry = decided(gap_record(3), TP_PROMINENCE)
    left, right = decide.librating_episodes(geometry, S)
    left, right = with_band(left, 0.0, 30.0), with_band(right, 90.0, 60.0)
    assert math.degrees(g.edge_centre(left, 'end')) == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(g.edge_centre(right, 'start')) == pytest.approx(90.0, abs=1e-9)
    t = np.asarray(geometry.angle_f_t_yr, dtype=float)
    inside = (t >= left.t_end_yr) & (t <= right.t_start_yr)
    sigma = np.full(len(t), np.nan)
    if gap_sigma == 'glide':
        sigma[inside] = np.radians(90.0) * (t[inside] - left.t_end_yr) / (right.t_start_yr - left.t_end_yr)
    else:
        sigma[inside] = math.radians(gap_sigma)
    overshoot = g.gap_band_overshoot(replace(geometry, angle_f=sigma), left, right)
    assert math.degrees(overshoot) == pytest.approx(expected_deg, abs=0.05)


# --- (xi) C9: the edge rule reads the local centre, not the episode's mean centre -------------------------------------
#
# 2016 JA 1E-1 shape (W28 g03-2): a horseshoe half-swing from 45 to 320 deg (about 182.5 deg), then tadpole cycles of
# +-30 deg about 290 deg. Turning points at 130 yr (45 deg), 340 yr (320 deg), then every 100 yr 260 and 320 deg, 12
# more; half-cosines between them. Before the first one sigma comes down from the horseshoe maximum 320 deg at -80 yr
# (outside the record, which starts at 0 at 320 - 275 (1 - cos(80 pi / 210)) / 2 = 232.7 deg): no circulation. The
# mean of its 13 midpoints, (182.5 + 12 x 290) / 13 = 281.7 deg, lies 236.7 deg from the first turning point, so
# the mean reading sees sigma beyond pi of it at that turning point (an entry, suspected: 0.6 tadpole periods of record
# before it); the edge swing's own midpoint, (45 + 320) / 2 = 182.5 deg, holds every sample before it within 137.5 deg.

HS_TPS = [(130.0, 45.0), (340.0, 320.0)] + [(340.0 + 100.0 * (i + 1), 260.0 if i % 2 == 0 else 320.0) for i in range(12)]


def horseshoe_then_tadpole():
    t = np.arange(0.0, HS_TPS[-1][0] + 60.0, 0.25)
    times = np.array([-80.0] + [tp[0] for tp in HS_TPS] + [HS_TPS[-1][0] + 100.0])
    values = np.radians([320.0] + [tp[1] for tp in HS_TPS] + [260.0])
    k = np.clip(np.searchsorted(times, t, 'right') - 1, 0, len(times) - 2)
    sigma = half_cosine(t, times[k], times[k + 1], values[k], values[k + 1])
    return filtered(t, sigma, 5.0)


def test_xi_a_horseshoe_swing_before_tadpole_cycles_is_no_entry():
    angle = horseshoe_then_tadpole()
    geometry = g.measure_geometry(angle, ONE_DEG, S)
    (run,) = [run for run in geometry.runs if run.accepted]
    assert run.t_start_yr == pytest.approx(130.0, abs=1.0) and run.t_end_yr == pytest.approx(HS_TPS[-1][0], abs=1.0)
    assert math.degrees(run.edge_centre_rad('entry')) == pytest.approx(182.5, abs=0.5)
    assert math.degrees(run.edge_centre_rad('exit')) == pytest.approx(290.0, abs=0.5)
    assert math.degrees(run.centre_mean_rad) == pytest.approx(281.7, abs=0.1)
    # the mean centre's reading: an entry at the first turning point, suspected (0.6 periods of record before it)
    t, raw = np.asarray(angle.t_yr), np.asarray(angle.angle_raw)
    averaged = g.orbit_averaged(t, raw, angle.orbit_period_yr)
    record = (float(t[0]), float(t[-1]))
    old = g._edge_event(t, np.asarray(angle.angle_f), raw, run, 'entry', record, S, averaged, run.centre_mean_rad)
    assert old is not None and old.suspected and old.time_yr == pytest.approx(130.0, abs=5.0)
    # the local centre's: none, and the rule that changed the reading fired
    assert geometry.entry is None and not run.entry_observed
    assert 'entry_suspected' not in geometry.flags and not geometry.entry_suspected()
    assert g.rules.Rule.LOCAL_CENTRE_ENTRY in geometry.rules_fired


# A pendulum about pi (A = 60 deg, P = 1000 yr: k = 1/2, omega_0 = 4 K(1/4) / P = 6.743e-3 / yr) entered from a
# circulation at its top rate 2 k omega_0 = omega_0 for 1500 yr: sigma = pi + omega_0 t before 0, so sigma - pi < -pi
# before t = -pi / omega_0 = -465.9 yr, and the averaged angle travels 1500 omega_0 + 60 deg = 11.2 rad >= one turn to
# the first turning point (250 yr): an observed entry. Both centres are pi: the reading does not change.
# The same with a drifting centre, sigma = pi + c_dot t + A sin(2 pi t / P), c_dot = 10 deg per period, 8 periods,
# entered at the rate c_dot + 2 pi A / P = 0.3870 deg/yr: the edge swing's midpoint is c(P / 2) = pi + 5 deg, the mean
# of the midpoints pi + 40 deg; against the local centre sigma leaves it by pi at t = -175 / 0.3870 = -452.2 yr
# (against the mean one at -361.8 yr).


def circulation_into(t_lead_yr, centre_drift_deg_per_period):
    period, amplitude = 1000.0, math.radians(60.0)
    t = np.arange(-t_lead_yr, 8 * period, 0.5)
    inside = t >= 0
    sigma = np.empty(len(t))
    if centre_drift_deg_per_period is None:
        lib, rate, _ = pendulum(t[inside], 60.0, period)
        sigma[inside] = lib
        sigma[~inside] = np.pi + rate[0] * t[~inside]
    else:
        drift = math.radians(centre_drift_deg_per_period) / period
        sigma[inside] = np.pi + drift * t[inside] + amplitude * np.sin(TWO_PI * t[inside] / period)
        sigma[~inside] = np.pi + (drift + TWO_PI * amplitude / period) * t[~inside]
    return filtered(t, sigma, 10.0)


def test_xi_a_true_entry_from_circulation_is_observed():
    geometry = g.measure_geometry(circulation_into(1500.0, None), ONE_DEG, S)
    run = min((run for run in geometry.runs if run.accepted), key=lambda run: run.t_start_yr)
    assert math.degrees(run.edge_centre_rad('entry')) == pytest.approx(180.0, abs=0.5)
    event = geometry.entry
    assert event is not None and event.observed and not event.suspected and run.entry_observed
    assert event.time_yr == pytest.approx(-465.9, abs=5.0)
    assert event.raw_arc_rad == pytest.approx(11.2, abs=0.2)
    assert not event.differs_from_mean_centre and g.rules.Rule.LOCAL_CENTRE_ENTRY not in geometry.rules_fired


def test_xi_a_drifting_centre_entered_from_circulation_is_observed_at_its_local_centre():
    geometry = g.measure_geometry(circulation_into(1500.0, 10.0), ONE_DEG, S)
    run = min((run for run in geometry.runs if run.accepted), key=lambda run: run.t_start_yr)
    assert math.degrees(run.edge_centre_rad('entry')) == pytest.approx(185.0, abs=1.0)
    assert math.degrees(run.centre_mean_rad) == pytest.approx(220.0, abs=3.0)
    event = geometry.entry
    assert event is not None and event.observed and not event.suspected and run.entry_observed
    assert event.time_yr == pytest.approx(-452.2, abs=5.0)  # the mean centre's crossing would be at -361.8 yr


def test_xi_in_the_frame_of_its_moving_well_the_local_centre_is_the_well_at_the_edge_turning_point():
    # the well's track 100 -> 200 deg linear over 0-1000 yr; the run's edge turning points at 130 yr and 1540 yr (held
    # at the last node beyond it): c = 113 and 200 deg, whatever the turning points' values
    geometry = g.measure_geometry(horseshoe_then_tadpole(), ONE_DEG, S)
    (run,) = [run for run in geometry.runs if run.accepted]
    well = g.MovingWell(g.WELL_OK, np.array([0.0, 1000.0]), np.radians([100.0, 200.0]))
    moving = replace(run, frame=g.FRAME_PSI, well=well)
    assert math.degrees(moving.edge_centre_rad('entry')) == pytest.approx(100.0 + 100.0 * run.t_start_yr / 1000.0, abs=1e-9)
    assert math.degrees(moving.edge_centre_rad('exit')) == pytest.approx(200.0, abs=1e-9)
    assert math.degrees(replace(moving, well=g.MovingWell(g.WELL_INVALID)).edge_centre_rad('entry')) == pytest.approx(182.5, abs=0.5)


# --- (xii) C4: a 1:1 turning point the edge trim hides, read on the orbit-averaged raw angle -----------------------

MU_EARTH, N_EARTH = 3.0404e-6, 2 * np.pi  # Earth + Moon / Sun; n of a 1-yr orbit [rad/yr] (a_P = 1 au)
HS_TURN_DEG = 6.0  # the horseshoe's turning angle: turns 42.9 + 85.9 n yr, at 6 and 354 deg


def _softening():
    from resonances.resonance.analysis.mmr import coorbital as co

    return co.hill_softening(MU_EARTH, S.mmr_coorbital_softening_hill)


def _closed_r(theta):
    """R~ of the circular coplanar problem (M&D eqs. 3.178-3.179), softened as the package's table."""
    return 1.0 / np.sqrt(4 * np.sin(theta / 2) ** 2 + _softening() ** 2) - np.cos(theta)


def horseshoe(span_yr):
    """theta(t), theta_dot and a of the averaged circular problem integrated by solve_ivp from L3 towards the planet
    (theta_ddot = -3 mu n^2 R~', a = a_P (1 - (2/3) theta_dot / n)), and the turn times (theta_dot = 0)."""

    def rhs(_, y):
        d2 = 4 * np.sin(y[0] / 2) ** 2 + _softening() ** 2
        return [y[1], -3 * MU_EARTH * N_EARTH**2 * np.sin(y[0]) * (1.0 - d2**-1.5)]

    def turn(_, y):
        return y[1]

    level = _closed_r(np.radians(HS_TURN_DEG)) - _closed_r(np.pi)
    t = np.arange(0.0, span_yr, 0.05)
    run = solve_ivp(
        rhs, (0.0, span_yr), [np.pi, -np.sqrt(6 * MU_EARTH * N_EARTH**2 * level)], t_eval=t, events=turn, rtol=1e-11, atol=1e-13
    )
    theta, rate = run.y
    return t, theta, rate, 1.0 - (2.0 / 3.0) * rate / N_EARTH, run.t_events[0]


def coorbital_angle(t, sigma, a, cutoff, edge_rule=True):
    """sigma (= theta, k = +1) of a 1:1 angle with the C4 edge model on the circular Earth's averaged potential. The
    closure by phase return is off: the record's half-cycle before the first turn would close the two-TP run by
    itself, and only turning points are tested here."""
    from resonances.resonance.analysis.mmr import coorbital as co
    from resonances.resonance.analysis.mmr import coorbital_geometry as cg

    angle = replace(filtered(t, sigma, cutoff, coorbital=True), slow_f=lowpass(t, a, cutoff, 2), orbit_period_yr=1.0, return_closure=False)
    if not edge_rule:
        return angle
    circular = co.Orbit(0.0, 0.0, 0.0, 0.0)
    orbits = S.mmr_coorbital_edge_turn_orbits
    model = cg.CoorbitalEdgeTurn(
        series=None,
        context=None,
        t_yr=t,
        averaged=g.orbit_averaged(t, sigma, orbits),
        a_averaged=g.orbit_averaged(t, a, orbits),
        body_coefficient=1,
        settings=S,
        profile=cg.profile_from_orbits(circular, circular, MU_EARTH, N_EARTH, S),
    )
    return replace(angle, edge_turn=model)


def _record(theta, rate, a, t, start_yr, end_yr):
    keep = (t >= start_yr) & (t <= end_yr)
    return t[keep], theta[keep], rate[keep], a[keep]


def test_xii_a_third_reflection_10_yr_before_the_end_closes_the_cycle():
    """Record 100 yr -> the third turn + 10 yr (310.5 yr), cutoff 20 yr: sigma_f turns at 128.8 and 214.7 yr inside
    the window [120, 290.5]; the third reflection (300.5 yr, 6 deg from the planet) lies in the trim. In 10 yr it swings
    back 34.7 deg (solve_ivp), above MMR_COORBITAL_EDGE_TURN_MIN_DEG, and a mirrors |a - a_P| (out/in = 1 by time
    reversal): the run of two turning points becomes one closed horseshoe cycle."""
    t, theta, rate, a, turns = horseshoe(320.0)
    third = turns[3]
    t, theta, rate, a = _record(theta, rate, a, t, 100.0, third + 10.0)
    without = g.measure_geometry(coorbital_angle(t, theta, a, 20.0, edge_rule=False), ONE_DEG, S)
    assert [run.accepted for run in without.runs] == [False]
    assert np.allclose(without.runs[0].tp_times_yr, turns[1:3], atol=2.0)

    geometry = g.measure_geometry(coorbital_angle(t, theta, a, 20.0), ONE_DEG, S)
    (run,) = geometry.runs
    assert run.accepted and run.edge_raw_tp == 'exit'
    assert np.allclose(run.tp_times_yr, turns[1:4], atol=2.0)
    (edge,) = geometry.edge_turns
    assert edge.added and edge.site == 'planet' and edge.a_crosses
    assert edge.out_in == pytest.approx(1.0, abs=0.05)
    assert math.degrees(edge.prominence_rad) > S.mmr_coorbital_edge_turn_min_deg
    assert math.degrees(abs(geometry.turning_points.values_rad[-1] - 2 * np.pi)) == pytest.approx(HS_TURN_DEG, abs=1.5)
    assert g.rules.Rule.COORBITAL_EDGE_TURN in geometry.rules_fired
    assert geometry.to_columns()['geo_edge_raw_tp_t'] == f'{edge.t_yr:.6g}'


def test_xii_a_circulating_edge_with_a_30_deg_ripple_gives_no_turning_point():
    """The same two turns, then from L3 (257.6 yr) the angle keeps its L3 rate and passes the planet (no third turn),
    with a ripple of 30 deg at the 1-yr orbit on the raw angle. The raw angle turns back by ~2 x 30 deg every orbit in
    the trim; averaged over MMR_COORBITAL_EDGE_TURN_ORBITS orbits it runs on: no turning point, no episode."""
    t, theta, rate, a, turns = horseshoe(320.0)
    at_l3 = turns[2] + (turns[2] - turns[1]) / 2
    linear = t > at_l3
    speed = float(np.interp(at_l3, t, rate))
    theta = np.where(linear, np.interp(at_l3, t, theta) + speed * (t - at_l3), theta)
    a = np.where(linear, 1.0 - (2.0 / 3.0) * speed / N_EARTH, a)
    sigma = theta + np.radians(30.0) * np.sin(2 * np.pi * t)
    t, sigma, _, a = _record(sigma, rate, a, t, 100.0, turns[3] + 10.0)
    trim = t > t[-1] - 20.0
    peaks, _ = g.signal.find_peaks(sigma[trim], prominence=np.radians(S.mmr_coorbital_edge_turn_min_deg))
    assert len(peaks) > 0  # the raw angle itself does turn back in the trim

    geometry = g.measure_geometry(coorbital_angle(t, sigma, a, 20.0), ONE_DEG, S)
    assert not any(turn.added for turn in geometry.edge_turns)
    assert [run.accepted for run in geometry.runs] == [False] and geometry.runs[0].edge_raw_tp == ''
    assert g.rules.Rule.COORBITAL_EDGE_TURN not in geometry.rules_fired


def test_xii_an_edge_kick_with_out_in_0_4_is_rejected():
    """The third turn 20 yr before the end (cutoff 25 yr), but |a - a_P| after it is 0.4 x the mirrored one and theta
    follows that a (theta_dot = -(3/2) n (a - a_P)): it still swings back 0.4 x 78 = 31 deg, a still crosses a_P, but
    |ln 0.4| = 0.92 > MMR_COORBITAL_TURN_ENERGY_TOL: a kick, not a reflection. No turning point is added."""
    t, theta, rate, a, turns = horseshoe(330.0)
    third = turns[3]
    after = t > third
    rate = np.where(after, 0.4 * rate, rate)
    swept = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])  # trapezoid integral of rate
    theta = np.where(after, np.interp(third, t, theta) + swept - np.interp(third, t, swept), theta)
    a = 1.0 - (2.0 / 3.0) * rate / N_EARTH
    t, theta, rate, a = _record(theta, rate, a, t, 100.0, third + 20.0)
    geometry = g.measure_geometry(coorbital_angle(t, theta, a, 25.0), ONE_DEG, S)
    (edge,) = geometry.edge_turns
    assert edge.outcome == 'energy' and edge.site == 'planet' and edge.a_crosses
    assert edge.out_in == pytest.approx(0.4, abs=0.03)
    assert math.degrees(edge.prominence_rad) > S.mmr_coorbital_edge_turn_min_deg
    assert [run.accepted for run in geometry.runs] == [False]
    assert len(geometry.turning_points) == 2


# --- (xiii) C7, rule coorbital_well_cut: a 1:1 episode ends only at a separatrix --------------------------------------


def wells_model(e=0.0):
    """The wells of the averaged Earth problem for a body of eccentricity e on the circular Earth's orbit (sigma =
    theta): circular, L4/L5 at 60/300 deg and no quasi-satellite well; e = 0.1, a quasi-satellite well about 0."""
    from resonances.resonance.analysis.mmr import coorbital as co, coorbital_geometry as cg

    profile = cg.profile_from_orbits(co.Orbit(e, 0.0, 0.0, 0.0), co.Orbit(0.0, 0.0, 0.0, 0.0), MU_EARTH, N_EARTH, S)
    return cg.CoorbitalWells(series=None, context=None, body_coefficient=1, settings=S, profile=profile)


def minimum_deg(model, name):
    return next(math.degrees(x.theta_rad) for x in model.profile.extremes if x.name == name)


def wells_angle(t, sigma, rate, model):
    """The 1:1 angle read exactly (sigma_f = sigma), with the wells of `model` (None: the band test and the cuts)."""
    return replace(filtered(t, sigma, 10.0, rate=rate, smooth=False, coorbital=True), coorbital_wells=model)


def wells_geometry(angle, prominence_deg):
    geometry = g.measure_geometry(angle, math.radians(prominence_deg), S)
    g.coorbital_marks(geometry, angle, S)
    return geometry


def marks_of(run, name):
    """[(t0, t1)] of the marks `name@t0:t1` of a run."""
    spans = []
    for item in filter(None, run.marks.split(';')):
        mark, _, span = item.partition('@')
        if mark == name:
            start, _, end = span.partition(':')
            spans.append((float(start), float(end or start)))
    return spans


def amplitude_jump_record(model):
    """A tadpole about L4 (60 deg): 4 cycles of 40 deg at P = 200 yr about L4 + 12 deg, then (an impulse at the zero
    crossing at 800 yr) 6 cycles of 8 deg, 5x smaller, whose centre drifts from L4 + 12 to L4 - 12 deg (the forced
    centre moves). The small cycles' bands [c + 4, c + 20] and [c - 20, c - 4] share no point: the band test fails,
    while every swing holds the minimum to within 4 deg, below the 6-deg prominence. Turning points at 50 + 100 n yr;
    the transitional half-cycle runs from the last large one (750 yr, L4 + 12 - 40) to the first small one (850 yr)."""
    centre, omega = math.radians(minimum_deg(model, 'tadpole_L4')), TWO_PI / 200.0
    t = np.arange(0.0, 2000.0 + 0.25, 0.25)
    drift = math.radians(12.0) - math.radians(24.0) * np.clip((t - 800.0) / 1200.0, 0.0, 1.0)
    big, small = math.radians(40.0), math.radians(8.0)
    sigma = np.where(t < 800.0, centre + math.radians(12.0) + big * np.sin(omega * t), centre + drift + small * np.sin(omega * (t - 800.0)))
    rate = np.where(t < 800.0, big * omega * np.cos(omega * t), small * omega * np.cos(omega * (t - 800.0)) - math.radians(24.0) / 1200.0)
    return t, sigma, rate


def test_xiii_a_fivefold_amplitude_jump_in_one_tadpole_well_is_one_episode_with_a_mark():
    model = wells_model()
    assert minimum_deg(model, 'tadpole_L4') == pytest.approx(60.0, abs=0.5)  # M&D: L4 of the circular problem
    t, sigma, rate = amplitude_jump_record(model)
    without = wells_geometry(wells_angle(t, sigma, rate, None), 6.0)
    assert not any(run.accepted and covering(run) for run in without.runs)  # the band test fails, no cut stands

    geometry = wells_geometry(wells_angle(t, sigma, rate, model), 6.0)
    (episode,) = [run for run in geometry.runs if run.accepted]
    assert covering(episode) and episode.shape == g.EpisodeShape.REGIME_CHANGE
    assert episode.wells == 'tadpole_L4' and not episode.several_wells
    ((start, end),) = marks_of(episode, 'amplitude_change')
    assert start == pytest.approx(750.0, abs=5.0) and end == pytest.approx(850.0, abs=5.0)
    assert marks_of(episode, 'well_change') == []
    assert g.rules.Rule.COORBITAL_WELL_CUT in geometry.rules_fired


def qs_to_tadpole_record(model, glide_yr=400.0):
    """A quasi-satellite (e = 0.1) of 10 deg about 0 at P = 100 yr for 6 cycles, a monotone glide from 0 to L4 over
    `glide_yr` (no turn: sigma moves less than 2 pi), then a tadpole of 20 deg about L4 at 200 yr for 5 cycles. The
    glide half-cycle (575 yr, -10 deg, to 600 + glide + 50 yr, L4 + 20 deg) is 9.5 x the quasi-satellite's half-period:
    `split_runs` breaks it into two runs. The cycles holding both ends hold the quasi-satellite minimum and L4
    (compound)."""
    l4 = math.radians(minimum_deg(model, 'tadpole_L4'))
    t2 = 600.0 + glide_yr
    t = np.arange(0.0, t2 + 1000.0 + 0.25, 0.25)
    qs, tadpole = math.radians(10.0), math.radians(20.0)
    sigma = np.where(
        t < 600.0,
        qs * np.sin(TWO_PI * t / 100.0),
        np.where(t < t2, l4 * (t - 600.0) / glide_yr, l4 + tadpole * np.sin(TWO_PI * (t - t2) / 200.0)),
    )
    rate = np.where(
        t < 600.0,
        qs * TWO_PI / 100.0 * np.cos(TWO_PI * t / 100.0),
        np.where(t < t2, l4 / glide_yr, tadpole * TWO_PI / 200.0 * np.cos(TWO_PI * (t - t2) / 200.0)),
    )
    return t, sigma, rate, t2


def test_xiii_a_quasi_satellite_turning_into_a_tadpole_is_one_episode_with_sub_regimes():
    model = wells_model(0.1)
    assert minimum_deg(model, 'qs') == pytest.approx(0.0, abs=0.5)
    t, sigma, rate, t2 = qs_to_tadpole_record(model)
    without = wells_geometry(wells_angle(t, sigma, rate, None), 2.0)
    assert len([run for run in without.runs if run.accepted]) == 2  # two runs; the union joins them for the status

    geometry = wells_geometry(wells_angle(t, sigma, rate, model), 2.0)
    (episode,) = [run for run in geometry.runs if run.accepted]
    assert covering(episode) and episode.n_tp == sum(run.n_tp for run in without.runs if run.accepted)
    assert episode.wells == 'qs>compound>tadpole_L4' and episode.several_wells
    assert episode.shape == g.EpisodeShape.REGIME_CHANGE
    # the glide's half-cycle (575 yr -> t2 + 50) swings through both wells: a change at each of its turning points
    assert marks_of(episode, 'well_change') == [pytest.approx((575.0, 575.0), abs=2.0), pytest.approx((t2 + 50.0,) * 2, abs=2.0)]
    decision, _ = decided(wells_angle(t, sigma, rate, model), math.radians(2.0))
    assert (decision.angle_status, decision.subtype) == (AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE)
    # its floor is the quasi-satellite well's: it holds the well through its quasi-satellite cycles
    well = g.FloorWell(t, np.zeros(len(t)), np.full(len(t), math.radians(20.0)), 111.0, 'x')
    assert well.holds(episode) and not well.holds(replace(episode, wells='qs'))


def test_xiii_one_turn_of_circulation_between_quasi_satellites_separates_two_episodes():
    """QS of 10 deg (P = 100 yr, 6 cycles), one turn of circulation in 150 yr, then the QS about 360 deg: the half-cycle
    from -10 to 370 deg swings 190 deg, over the top: a separatrix crossing, two episodes, nothing joined."""
    model = wells_model(0.1)
    qs = math.radians(10.0)
    t = np.arange(0.0, 1350.0 + 0.25, 0.25)
    t2 = 750.0
    sigma = np.where(
        t < 600.0,
        qs * np.sin(TWO_PI * t / 100.0),
        np.where(t < t2, TWO_PI * (t - 600.0) / 150.0, TWO_PI + qs * np.sin(TWO_PI * (t - t2) / 100.0)),
    )
    rate = np.gradient(sigma, t)
    geometry = wells_geometry(wells_angle(t, sigma, rate, model), 2.0)
    first, second = [run for run in geometry.runs if run.accepted]
    assert first.t_end_yr == pytest.approx(575.0, abs=1.0) and second.t_start_yr == pytest.approx(t2 + 25.0, abs=1.0)
    assert first.wells == second.wells == 'qs' and first.marks == second.marks == ''
    assert g.rules.Rule.COORBITAL_WELL_CUT not in geometry.rules_fired
    assert decide.librating_union(geometry, S).episodes == ()  # the turn: no union either


def test_xiii_a_ripple_drifting_past_l5_is_no_episode():
    """8 deg of ripple at 100 yr on a centre drifting from L5 by 60 deg over 1 kyr (6 deg per period, slower than the
    ripple turns it): the cycles beyond 8 + 2 deg (the prominence) from L5 hold no minimum, so the wells reject what
    the bands reject (2022 BB2 1V-1)."""
    model = wells_model()
    l5 = math.radians(minimum_deg(model, 'tadpole_L5'))
    t = np.arange(0.0, 1000.0 + 0.25, 0.25)
    sigma = l5 - math.radians(60.0) * t / 1000.0 + math.radians(8.0) * np.sin(TWO_PI * t / 100.0)
    rate = -math.radians(60.0) / 1000.0 + math.radians(8.0) * TWO_PI / 100.0 * np.cos(TWO_PI * t / 100.0)
    geometry = wells_geometry(wells_angle(t, sigma, rate, model), 2.0)
    assert not any(run.accepted for run in geometry.runs)
    assert any(str(run.reject_reason) == 'centre_moves' for run in geometry.runs)


# --- (xiv) C8, rule coorbital_residence: the core, the inferred residence and the sub-regimes ------------------------


def residence_geometry(angle, prominence_deg, settings=S):
    """The geometry as `finish_geometry` leaves a 1:1 angle: the runs, their marks and sub-regimes, their residences."""
    geometry = wells_geometry(angle, prominence_deg)
    g.coorbital_residences(geometry, angle, settings)
    geometry.forget_derived()
    return geometry


def capture_into_qs_record():
    """Two turns of circulation in 300 yr (sigma from -720 deg up to 0), then a quasi-satellite of 15 deg about 0 at
    P = 100 yr for 4 cycles (turning points 325, 375, ..., 675 yr), then swings of 0.5 deg to the end (1500 yr): below the
    2-deg turning-point prominence, so the core is 325-675 yr, and below the 20-deg reversal of the residence's legs,
    so its last leg runs 675 yr -> the end sweeping 15.5 deg (no turn): the stay goes on, inferred, to the record's end.
    The first leg (0 -> 325 yr) sweeps 735 deg, 2.04 turns: the capture bounds the residence at the core's start."""
    t = np.arange(0.0, 1500.0 + 0.25, 0.25)
    circulation = -2 * TWO_PI + 2 * TWO_PI * t / 300.0
    qs, tail = math.radians(15.0), math.radians(0.5)
    w = TWO_PI / 100.0
    sigma = np.where(t < 300.0, circulation, np.where(t < 700.0, qs * np.sin(w * (t - 300.0)), tail * np.sin(w * (t - 700.0))))
    rate = np.where(
        t < 300.0, 2 * TWO_PI / 300.0, np.where(t < 700.0, qs * w * np.cos(w * (t - 300.0)), tail * w * np.cos(w * (t - 700.0)))
    )
    return t, sigma, rate


def test_xiv_a_capture_into_the_qs_well_has_a_core_and_a_residence_to_the_end():
    model = wells_model(0.1)
    t, sigma, rate = capture_into_qs_record()
    angle = wells_angle(t, sigma, rate, model)
    geometry = residence_geometry(angle, 2.0)
    (episode,) = [run for run in geometry.runs if run.accepted]
    assert (episode.core_t0_yr, episode.core_t1_yr) == (pytest.approx(325.0, abs=1.0), pytest.approx(675.0, abs=1.0))
    residence = episode.residence
    assert residence.t0_yr == pytest.approx(325.0, abs=1.0) and residence.bounded_by_start == g.RESIDENCE_CIRCULATION
    assert residence.t1_yr == pytest.approx(1500.0) and residence.bounded_by_end == g.RESIDENCE_EDGE
    assert math.isnan(residence.zone_max)  # no a in a model without a series: not measured, never 0
    assert not episode.exit_observed and not covering(episode)  # 825 yr of record after the core: the end is not covered
    assert episode.coverage_end_gap_p > 5  # the core's gap, as without the residence
    assert g.rules.Rule.COORBITAL_RESIDENCE in geometry.rules_fired
    assert episode.subregimes == (g.SubRegime('qs', episode.t_start_yr, episode.t_end_yr),)  # one sub-regime, no law read


def test_xiv_the_residence_changes_no_decision():
    """The same record decided with and without the residences: the angle's word, subtype, flags, the episodes' edges
    and coverage and the union are identical (nothing reads the residence)."""
    model = wells_model(0.1)
    t, sigma, rate = capture_into_qs_record()
    angle = wells_angle(t, sigma, rate, model)
    off = replace(S, mmr_coorbital_residence=False)
    seen = []
    for settings in (S, off):
        geometry = residence_geometry(angle, 2.0, settings)
        decision = decide.decide_angle(SAMPLING, False, geometry, None, settings, Ledger.from_settings(settings))
        runs = [
            (r.accepted, r.t_start_yr, r.t_end_yr, r.coverage_start_gap_p, r.coverage_end_gap_p, r.exit_observed, r.entry_observed)
            for r in geometry.runs
        ]
        union = decide.librating_union(geometry, settings).episodes
        seen.append(
            (decision.angle_status, decision.subtype, tuple(geometry.flags), runs, len(union), [r.residence is None for r in geometry.runs])
        )
    (on_word, on_sub, on_flags, on_runs, on_union, on_none), (off_word, off_sub, off_flags, off_runs, off_union, off_none) = seen
    assert (on_word, on_sub, on_flags, on_runs, on_union) == (off_word, off_sub, off_flags, off_runs, off_union)
    assert on_none == [False] and off_none == [True]  # the switch removes the residence and nothing else


def horseshoe_to_tadpole_record():
    """A horseshoe of the circular Earth problem, sigma = 180 + 170 sin(2 pi t / 400) deg (turns at 10 and 350 deg,
    holding L4 and L5), to its minimum at 1100 yr, then a tadpole 60 - 50 cos(2 pi (t - 1100) / 200) deg about L4
    (10-110 deg) for 5 cycles: the same turn at 10 deg starts both. One libration of two sub-regimes, no separatrix."""
    t = np.arange(0.0, 2100.0 + 0.25, 0.25)
    hs = np.radians(180.0 + 170.0 * np.sin(TWO_PI * t / 400.0))
    tadpole = np.radians(60.0 - 50.0 * np.cos(TWO_PI * (t - 1100.0) / 200.0))
    sigma = np.where(t < 1100.0, hs, tadpole)
    rate = np.where(
        t < 1100.0,
        np.radians(170.0) * TWO_PI / 400.0 * np.cos(TWO_PI * t / 400.0),
        np.radians(50.0) * TWO_PI / 200.0 * np.sin(TWO_PI * (t - 1100.0) / 200.0),
    )
    return t, sigma, rate


def test_xiv_b_horseshoe_then_tadpole_is_one_episode_with_two_sub_regimes():
    from resonances.resonance.analysis import orbit_type as ot

    model = wells_model()
    t, sigma, rate = horseshoe_to_tadpole_record()
    geometry = residence_geometry(wells_angle(t, sigma, rate, model), 2.0)
    (episode,) = [run for run in geometry.runs if run.accepted]
    assert episode.wells == 'horseshoe>tadpole_L4'
    assert marks_of(episode, 'well_change') == [pytest.approx((1100.0, 1100.0), abs=1.0)]
    first, second = episode.subregimes
    assert (first.well, second.well) == ('horseshoe', 'tadpole_L4')
    assert first.t0_yr == pytest.approx(100.0, abs=1.0) and first.t1_yr == pytest.approx(1100.0, abs=1.0)
    assert second.t0_yr == pytest.approx(1100.0, abs=1.0) and second.starts_at == g.MARK_WELL_CHANGE
    assert (first.force, first.energy) == ('not_measured', 'not_measured')  # read, but no series to fit the law on
    assert g.well_sequence(geometry.runs) == 'horseshoe>tadpole_L4'
    spec = ResonanceSpec.from_key('1E-1+0+0')
    assert ot.mmr_orbit_type(spec, geometry, episode.episode_id, None, S, 'libration') == 'compound: horseshoe → tadpole L4'


def test_xiv_subregimes_split_at_well_and_amplitude_changes():
    """Hand-made: TPs at 0, 100, ..., 600 yr; a well change at 200 yr, an amplitude change over the half-cycle
    400-500 yr (it belongs to neither side): three sub-regimes [0, 200], [200, 400], [500, 600]."""
    times = np.arange(0.0, 700.0, 100.0)
    halves = ('qs', 'qs', 'tadpole_L4', 'tadpole_L4', 'tadpole_L4', 'tadpole_L4')
    spans = [(g.MARK_AMPLITUDE_CHANGE, 400.0, 500.0), (g.MARK_WELL_CHANGE, 200.0, 200.0)]
    pieces = g.subregimes_of(times, halves, spans)
    assert [(p.well, p.t0_yr, p.t1_yr, p.starts_at) for p in pieces] == [
        ('qs', 0.0, 200.0, g.SUBREGIME_START),
        ('tadpole_L4', 200.0, 400.0, g.MARK_WELL_CHANGE),
        ('tadpole_L4', 500.0, 600.0, g.MARK_AMPLITUDE_CHANGE),
    ]
    assert g.subregimes_text(pieces) == 'qs@0.0:200.0;tadpole_L4@200.0:400.0;tadpole_L4@500.0:600.0'
    assert g.subregime_checks_text(pieces) == ''  # no law read


def test_xiv_the_residence_walks_the_legs_until_a_whole_turn():
    """Hand-made legs: reversals at 100, 200, 300, 400 yr, sweeps 0.3, 1.2, 0.2, 0.4, 0.9 turns; a core 200-300 yr:
    the leg 100-200 yr sweeps 1.2 turns (circulation) and bounds the start at the core, the end runs to the record's end
    (1000 yr). The core 330-390 yr inside the leg 300-400: start at 200 yr (the leg before 200 sweeps 1.2). An observed
    exit keeps the core's end."""
    legs = g.AveragedLegs(np.array([100.0, 200.0, 300.0, 400.0]), np.array([0.3, 1.2, 0.2, 0.4, 0.9]), (0.0, 1000.0))
    run = make_wells_run(200.0, 300.0)
    assert g.residence_of(run, legs, S) == g.Residence(200.0, 1000.0, g.RESIDENCE_CIRCULATION, g.RESIDENCE_EDGE)
    assert g.residence_of(make_wells_run(330.0, 390.0), legs, S) == g.Residence(200.0, 1000.0, g.RESIDENCE_CIRCULATION, g.RESIDENCE_EDGE)
    exits = replace(make_wells_run(330.0, 390.0), exit_observed=True)
    assert g.residence_of(exits, legs, S) == g.Residence(200.0, 390.0, g.RESIDENCE_CIRCULATION, g.RESIDENCE_CIRCULATION)
    one_turn = g.AveragedLegs(
        np.array([100.0, 200.0, 300.0, 400.0]), np.array([0.3, 0.2, 0.2, 0.4, S.libration_edge_arc_rev]), (0.0, 1000.0)
    )
    assert g.residence_of(run, one_turn, S) == g.Residence(0.0, 400.0, g.RESIDENCE_EDGE, g.RESIDENCE_CIRCULATION)


def make_wells_run(t0, t1):
    from tests.resonances.analysis.builders import make_episode

    return make_episode(tp_times_yr=np.array([t0, (t0 + t1) / 2, t1]), tp_values_rad=np.array([0.1, -0.1, 0.1]))


def test_xiv_a_long_sub_regime_sequence_is_shortened_in_the_header_only():
    from resonances.resonance.analysis import orbit_type as ot

    four = ('tadpole_L5', 'horseshoe', 'tadpole_L4', 'compound')  # 2009 BE81 1M-1 20 kyr: shown whole
    assert ot.coorbital_text(four) == 'compound: tadpole L5 → horseshoe → tadpole L4 → compound'
    six = ('quasi_satellite', 'compound', 'tadpole_L5', 'compound', 'horseshoe', 'compound')
    assert ot.coorbital_text(six) == 'compound: quasi-satellite → compound → tadpole L5 → … compound (6 sub-regimes)'


# --- (xv) C11: an accepted run faster than the period floor is not counted (W28 g02-9 A) ---------------------------


def tadpole_record(floor_yr):
    """A 1:1 tadpole about L4: sigma = 60 deg + 30 deg sin(2 pi t / 100 yr) on 0-330 yr, filtered at 10 yr, the angle's
    period floor set to `floor_yr` (as `preprocess.libration_period_floor` sets a co-orbital's). Its turning points are
    the extremes at 25 + 50 n yr: one run whose full periods are 100 yr by construction."""
    t = np.arange(0.0, 330.0, 0.25)
    sigma = np.radians(60.0) + np.radians(30.0) * np.sin(2 * np.pi * t / 100.0)
    angle = replace(filtered(t, sigma, 10.0, coorbital=True), period_floor_yr=floor_yr, period_floor_setting='mmr_libration_period_min_yr')
    return decided(angle)


def test_xv_an_accepted_cycle_faster_than_the_floor_is_ineligible_below_period_floor():
    """Floor 150 yr against the 100-yr cycles: the run stays accepted (geometry), but the status does not count it
    (no librating episode, no libration), and episodes.csv says why: eligible False, below_period_floor; the rule
    eligible_reason fires. The same record at a 50-yr floor counts: eligible True, '' and no rule."""
    from resonances.resonance.analysis.mmr.models import eligibility_columns

    decision, geometry = tadpole_record(150.0)
    (run,) = [run for run in geometry.runs if run.accepted]
    assert run.period_med_yr == pytest.approx(100.0, rel=0.02)
    assert decide.librating_episodes(geometry, S) == []
    assert decision.angle_status != AngleStatus.LIBRATION
    assert geometry.eligibility_reason(run) == g.RejectReason.BELOW_PERIOD_FLOOR
    assert eligibility_columns(geometry, run) == {'eligible': False, 'ineligible_reason': 'below_period_floor'}
    assert run.reject_reason == g.RejectReason.NONE  # accepted keeps its meaning
    assert g.rules.Rule.ELIGIBLE_REASON in geometry.rules_fired

    decision, geometry = tadpole_record(50.0)
    (run,) = [run for run in geometry.runs if run.accepted]
    assert decide.librating_episodes(geometry, S) == [run]
    assert decision.angle_status == AngleStatus.LIBRATION
    assert eligibility_columns(geometry, run) == {'eligible': True, 'ineligible_reason': ''}
    assert g.rules.Rule.ELIGIBLE_REASON not in geometry.rules_fired


def test_xv_a_forced_wave_is_ineligible_whatever_its_period_and_a_rejected_run_is_not_counted():
    """The 50-yr-floor record with its run marked a forced wave after the decision (as rule 5s marks its waves): the
    reason follows `forced_waves` (forced_wave), though the stored eligibility said it counted. A rejected run: False,
    '' (its reason is reject_reason); before any decision: unknown (None)."""
    from resonances.resonance.analysis.mmr.models import ForcedWave, eligibility_columns

    _, geometry = tadpole_record(50.0)
    (run,) = [run for run in geometry.runs if run.accepted]
    wave = ForcedWave(
        run.episode_id, run.n_tp, run.period_med_yr, float('nan'), float('nan'), '', float('nan'), False, False, by_strength=True
    )
    geometry.forced_waves = (wave,)
    assert eligibility_columns(geometry, run) == {'eligible': False, 'ineligible_reason': 'forced_wave'}
    assert eligibility_columns(geometry, replace(run, accepted=False, reject_reason=g.RejectReason.CENTRE_MOVES)) == {
        'eligible': False,
        'ineligible_reason': '',
    }
    fresh = replace(geometry, forced_waves=())  # a replace starts without the decision's reading
    assert eligibility_columns(fresh, run) == {'eligible': None, 'ineligible_reason': ''}


def test_xv_a_librating_run_before_the_last_jump_is_counted_by_g1e_only():
    """The G2a step of (vii): the run before the cut (250-2750 yr, P = 1000 yr) librates but lies before the last jump,
    the run after it (P = 700 yr) is the last regime's. Default floor (MMR_LIBRATION_PERIOD_MIN_YR, below both
    periods): before_last_jump and NONE, `librating_episodes` with and without before_jump returns each, and no rule
    fires for the earlier run. A floor of 800 yr between the two periods: the later run is below_period_floor."""
    t, sigma, rate, _ = step_record()
    geometry = finished(filtered(t, sigma, CUTOFF_STEP, rate=rate, smooth=False), regime_energy)
    before = [run for run in geometry.runs if run.accepted and run.before_last_jump]
    after = geometry.episodes
    assert S.mmr_libration_period_min_yr < P_AFTER < P_BEFORE
    reasons = decide.episode_eligibility(geometry, S)
    assert [reasons[run.episode_id] for run in before + after] == [g.RejectReason.BEFORE_LAST_JUMP, g.RejectReason.NONE]
    assert decide.librating_episodes(geometry, S, before_jump=True) == before
    assert decide.librating_episodes(geometry, S) == after
    assert g.rules.Rule.ELIGIBLE_REASON not in geometry.rules_fired

    slow = replace(geometry, period_floor_yr=800.0, period_floor_setting='mmr_libration_period_min_yr')
    reasons = decide.episode_eligibility(slow, S)
    assert [reasons[run.episode_id] for run in before + after] == [g.RejectReason.BEFORE_LAST_JUMP, g.RejectReason.BELOW_PERIOD_FLOOR]
    assert decide.librating_episodes(slow, S) == [] and g.rules.Rule.ELIGIBLE_REASON in slow.rules_fired
