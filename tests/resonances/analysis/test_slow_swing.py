"""W28 B6c, G5w (cards g07-5 and g07-6, the author's answers A of 2026-10-31): an unfinished slow swing that is a swing
in the resonance's own well reads insufficient_time, not drift (rule `slow_swing_in_well`, MMR_SLOW_SWING_IN_WELL).

523683 2U-9 over 100 kyr: the slow centre turns at 11.9 and 67.7 kyr (one half-swing of 114 deg), the slow identity
holds (slope / c 1.00), the a swing between the turns is 1.07 x and the period 2 (t2 - t1) = 111.6 kyr is 1.00 x the
directly averaged pendulum at that amplitude (T0_th 84 kyr): too short to tell, extend to 400 kyr (three slow cycles).
The author's drift of 101703 and 142694 1V+3S-7 (also two turns) fails the slow identity (slope / c -0.27/-0.25).

The synthetic record is a pendulum integrated here (scipy, independent of the package): sigma_ddot = -omega0^2 sin
sigma with T0 = 80 kyr, released at rest at A = 114 deg, seen from 0.05 to 0.65 of its period (0.6 cycle, the turns
at 0.05 T and 0.55 T); a = a* + sigma_dot / c with c = -1.5 k n / a* for k = -9, a* = 52.384 au (523683's 2U-9). Its
period, measured from the integration's own sigma_dot = 0 events, checks the closed form T0 F(A) =
T0 (2/pi) K(sin^2(A/2)). The geometry is built from these arrays (`builders`), with the turns where the integration
puts them.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, slow_swing
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Status
from tests.resonances.analysis.builders import make_centre, make_geometry, make_turning_points
from tests.resonances.analysis.test_w20_regime_cuts import S, SAMPLING

T0_YR = 80e3
AMPLITUDE = math.radians(114.0)
A_STAR = 52.384
K = -9
C = -1.5 * K * (2 * math.pi * A_STAR**-1.5) / A_STAR  # rad yr^-1 au^-1, = 4.8e-4 for these numbers
STEP_YR = 50.0
W_C_YR = 3000.0  # the centre's window (3 P_cyc of a 1-kyr wave): the slow swing passes it unchanged
CUTOFF_YR = 500.0


def pendulum():
    """(t, sigma, sigma_dot, period T measured from the sigma_dot = 0 events) of the released pendulum."""
    omega0 = 2 * math.pi / T0_YR

    def rhs(_, y):
        return [y[1], -(omega0**2) * math.sin(y[0])]

    def turn(_, y):
        return y[1]

    solution = solve_ivp(rhs, (0, 2.5 * T0_YR), [AMPLITUDE, 0.0], events=turn, rtol=1e-10, atol=1e-12, dense_output=True)
    events = solution.t_events[0]
    period = 2 * float(np.median(np.diff(events[events > 1.0])))  # half-periods between the turns
    t = np.arange(0.0, 0.6 * period + STEP_YR / 2, STEP_YR)
    sigma, rate = solution.sol(t + 0.45 * period)  # the record starts 0.05 T before the second turn (0.5 T)
    return t, sigma, rate, period


def record(a_factor=1.0, drift_fraction=0.0, flat_a=False):
    """(geometry, t, a) of the synthetic record; `a_factor` scales the a swing, `flat_a` keeps a at a*, and
    `drift_fraction` stands for runs rejected for their moving centre (G5 holds from 0.5)."""
    t, sigma, rate, period = pendulum()
    a = np.full_like(t, A_STAR) if flat_a else A_STAR + a_factor * rate / C
    turns = np.array([0.05, 0.55]) * period  # sigma_dot = 0 inside the record (the release's turns at 0.5 T and T)
    centre = make_centre(
        t_yr=t,
        centre_rad=sigma,
        window_yr=W_C_YR,
        turns=make_turning_points(turns, np.interp(turns, t, sigma)),
        range_rad=float(np.ptp(sigma)),
        max_pass_rad=2 * AMPLITUDE,
        net_rev=float((sigma[-1] - sigma[0]) / (2 * math.pi)),
    )
    fast = np.arange(1000.0, t[-1] - 1000.0, 1700.0)  # turning points of a fast wave on the swing (no episode)
    geometry = make_geometry(
        window=(CUTOFF_YR, float(t[-1]) - CUTOFF_YR),
        turning_points=make_turning_points(fast, np.interp(fast, t, sigma)),
        centre=centre,
        drift_fraction=drift_fraction,
        period_cycle_yr=1000.0,
    )
    return geometry, t, a, period


def measured(geometry, t, a, t0_theory=T0_YR):
    """`measure_slow_swing` without the direct averaging: the half-swing, then the pendulum of T0_th."""
    return slow_swing.with_theory(slow_swing.half_swing(geometry, t, a, C, S), t0_theory, C)


def decide_angle(geometry, swing):
    ledger = Ledger.from_settings(S)
    return decide.decide_angle(SAMPLING, False, geometry, None, S, ledger, swing), ledger


def test_the_integrated_period_is_the_closed_form():
    *_, period = pendulum()
    # F(A) = (2/pi) K(m) = 1 / AGM(1, cos(A/2)) (Gauss: K(m) = pi / (2 AGM(1, sqrt(1 - m))), m = sin^2(A/2))
    x, y = 1.0, math.cos(AMPLITUDE / 2)
    for _ in range(30):
        x, y = (x + y) / 2, math.sqrt(x * y)
    assert 1.180 < 1 / x < 1.373  # between the tabulated F(90 deg) and F(120 deg)
    assert period == pytest.approx(T0_YR / x, rel=1e-6)
    geometry, t, a, _ = record()
    swing = measured(geometry, t, a)
    assert swing.period_theory_yr == pytest.approx(period, rel=2e-3)


def test_a_slow_pendulum_over_0_6_cycle_reads_insufficient_time():
    geometry, t, a, period = record()
    swing = measured(geometry, t, a)
    assert swing.reason == '' and swing.identity == CheckValue.PASS
    assert swing.identity_slope_ratio == pytest.approx(1.0, abs=0.02) and swing.identity_corr > 0.99
    assert math.degrees(swing.amplitude_rad) == pytest.approx(114.0, abs=0.5)
    assert swing.period_yr == pytest.approx(period, rel=1e-3)
    # a swing between the turns = the half-excursion sigma_dot_max / |c| = 2 omega0 sin(A/2) / |c|
    assert swing.a_swing_au == pytest.approx(2 * (2 * math.pi / T0_YR) * math.sin(AMPLITUDE / 2) / abs(C), rel=0.01)
    assert swing.a_ratio == pytest.approx(1.0, abs=0.01) and swing.period_ratio == pytest.approx(1.0, abs=0.005)
    assert slow_swing.holds(swing, S)
    for drift in (0.0, 1.0):  # without G5 (the fast TPs would read few_cycles) and with it (it would read drift)
        decision, ledger = decide_angle(replace(geometry, drift_fraction=drift), swing)
        assert (decision.angle_status, decision.subtype, decision.rule) == (AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, 'G5w')
        assert any(entry.name.startswith('G5w') for entry in ledger)
    status = decide.decide_status(decision, SAMPLING, geometry, None, None, None, None, None, None, S, Ledger.from_settings(S))
    assert status.status == Status.UNCERTAIN


def test_the_extension_asks_for_three_slow_cycles():
    """By hand: the longest stretch without a turn is the half-swing, P = 2 x 0.5 T = T = 105.7 kyr (arc_slow);
    N = 3 P + 2 P_c = 317.6 kyr -> 400 kyr; the next check round_up(max(2 T_rec, min(N, 4 T_rec))) with T_rec = 63.4 -
    1.0 = 62.4 kyr: min(400, 249.7) = 249.7 -> 300 kyr. One cycle (1.5 P + 2 P_c = 159.6 kyr) would give 200 kyr."""
    geometry, t, a, period = record()
    decision, _ = decide_angle(geometry, measured(geometry, t, a))
    span = geometry.window[1] - geometry.window[0]
    base = decide.baseline(geometry, CUTOFF_YR, span, float('nan'), S, decision)
    assert (base.period_source, base.period_yr) == ('arc_slow', pytest.approx(period, rel=1e-3))
    assert base.required_yr == pytest.approx(3 * period + 2 * CUTOFF_YR)
    assert base.extend_to_years == 400e3
    assert decide.extend_years(Status.UNCERTAIN, decision.angle_status, base, span, S) == 300e3
    one_cycle = decide.baseline(geometry, CUTOFF_YR, span, float('nan'), replace(S, mmr_extend_slow_cycles=1.5), decision)
    assert one_cycle.extend_to_years == 200e3


def test_a_flat_a_is_no_slow_swing_and_stays_a_drift():
    geometry, t, a, _ = record(flat_a=True, drift_fraction=1.0)
    swing = measured(geometry, t, a)
    assert swing.identity != CheckValue.PASS  # sigma moves, a does not: no identity (rounding noise at most)
    assert not slow_swing.holds(swing, S)
    decision, _ = decide_angle(geometry, swing)
    assert (decision.angle_status, decision.rule) == (AngleStatus.DRIFT, 'G5')


def test_an_a_swing_three_times_too_large_fails_the_slow_identity_and_stays_a_drift():
    geometry, t, a, _ = record(a_factor=3.0, drift_fraction=1.0)
    swing = measured(geometry, t, a)
    assert swing.identity == CheckValue.FAIL and swing.identity_slope_ratio == pytest.approx(1 / 3, abs=0.01)
    decision, _ = decide_angle(geometry, swing)
    assert (decision.angle_status, decision.rule) == (AngleStatus.DRIFT, 'G5')


def test_a_well_too_weak_for_the_swing_stays_a_drift_and_sqrt_x_is_the_bound():
    """Theory T0_th k times the pendulum's own T0 (a well k^2 weaker): a ratio k, period ratio 1/k. sqrt(10) = 3.16
    (MMR_STRENGTH_CHECK_RATIO, the strength check's bound on mu): k = 3 lies inside, k = 4 outside."""
    geometry, t, a, _ = record(drift_fraction=1.0)
    inside = measured(geometry, t, a, t0_theory=3 * T0_YR)
    assert inside.a_ratio == pytest.approx(3.0, rel=0.01) and inside.period_ratio == pytest.approx(1 / 3, rel=0.01)
    assert decide_angle(geometry, inside)[0].rule == 'G5w'
    outside = measured(geometry, t, a, t0_theory=4 * T0_YR)
    assert outside.a_ratio == pytest.approx(4.0, rel=0.01)
    assert decide_angle(geometry, outside)[0].rule == 'G5'


def test_unmeasured_cases_change_nothing():
    geometry, t, a, _ = record(drift_fraction=1.0)
    one_turn = replace(geometry, centre=replace(geometry.centre, turns=make_turning_points([geometry.centre.turns.times_yr[0]], [0.0])))
    assert slow_swing.half_swing(one_turn, t, a, C, S).reason == slow_swing.REASON_TURNS
    assert slow_swing.half_swing(replace(geometry, coorbital=True), t, a, C, S).reason == slow_swing.REASON_COORBITAL
    off = replace(S, mmr_slow_swing_in_well=False)
    assert slow_swing.half_swing(geometry, t, a, C, off).reason == slow_swing.REASON_OFF
    no_theory = slow_swing.with_theory(slow_swing.half_swing(geometry, t, a, C, S), float('nan'), C, 'not_two_body')
    assert no_theory.reason == 'not_two_body' and not slow_swing.holds(no_theory, S)
    for swing in (None, no_theory):
        assert decide_angle(geometry, swing)[0].rule == 'G5'
