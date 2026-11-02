"""W28 C5 (cards g01-2, g02-8): the G1q trigger, lowered to a stay of one P_QS in the quasi-satellite well that holds a
turn of the orbit-averaged angle (`quasi_satellite.measure`, `QuasiSatellite.turns`, `analyse._quasi_satellite_pass`).

The synthetic records come from the averaged co-orbital ODE theta'' = -3 mu n^2 F~(theta) with F~ from a brute-force
average written in the independent W4 tests (not the package's table), a from n = n_P + theta_dot, and kicks of
theta_dot (jumps of a, as at an encounter) between a circulation and the quasi-satellite loop.
"""

import math
from functools import lru_cache

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis.mmr import coorbital
from resonances.resonance.analysis.mmr import quasi_satellite as qs_module
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config

from tests.resonances.independent_w4.test_status_coorbital import MU_EARTH, N_EARTH, _qs_force_spline

S = settings_from_config()
DEG = math.pi / 180


# --------------------------------------------------------------------------------------------- counting turns


@pytest.mark.parametrize(
    'values_deg, min_reversal_deg, expected',
    [
        # sin over 1.2 periods from 0: a maximum at 0.25 and a minimum at 0.75 of the period
        (10 * np.sin(2 * np.pi * np.linspace(0.0, 1.2, 1201)), 1.0, 2),
        # the same at amplitude 0.4 deg: every reversal (0.8 deg) is below 1 deg
        (0.4 * np.sin(2 * np.pi * np.linspace(0.0, 1.2, 1201)), 1.0, 0),
        # a monotone drift: no turn
        (np.linspace(30.0, -30.0, 500), 1.0, 0),
        # a drift with a 0.3-deg wiggle every 10 samples: the wiggle reverses by < 1 deg
        (np.linspace(30.0, -30.0, 500) + 0.3 * np.sin(2 * np.pi * np.arange(500) / 10), 1.0, 0),
        # down by 10, up by 5, down by 20: two turns (at -10 and at -5)
        (np.concatenate([np.linspace(0, -10, 11), np.linspace(-9, -5, 5), np.linspace(-6, -25, 20)]), 1.0, 2),
        # a single overshoot at the end: up 6 then down 2 at the last sample -> one turn
        (np.array([0.0, 2.0, 4.0, 6.0, 4.0]), 1.0, 1),
    ],
)
def test_count_turns(values_deg, min_reversal_deg, expected):
    assert qs_module.count_turns(np.radians(values_deg), math.radians(min_reversal_deg)) == expected


def _flat_well(t, halfwidth_deg=20.0, period_yr=50.0):
    return coorbital.QuasiSatelliteWell(t, np.zeros_like(t), np.full_like(t, math.radians(halfwidth_deg)), np.full_like(t, period_yr))


def test_visit_prefers_the_longest_stay_that_turns():
    """A long slow passage through the well (0-200 yr, no turn) and a shorter loop (500-560 yr, two turns): G1q reads
    the loop; without any turn the longest stay is read with 0 turns."""
    t = np.arange(0.0, 1000.0, 1.0)
    theta = np.full_like(t, math.radians(90.0))
    theta[:200] = np.radians(np.linspace(19.0, -19.0, 200))
    theta[500:560] = np.radians(15 * np.sin(2 * np.pi * (t[500:560] - 500) / 50.0))
    first, last, turns = qs_module._visit(t, theta, _flat_well(t), math.radians(1.0))
    assert (t[first], t[last], len(turns)) == (500.0, 559.0, 2)
    assert np.degrees(turns) == pytest.approx([15.0, -15.0], abs=0.2)  # the loop's top and bottom
    theta[500:560] = math.radians(90.0)
    first, last, turns = qs_module._visit(t, theta, _flat_well(t), math.radians(1.0))
    assert (t[first], t[last], turns) == (0.0, 199.0, [])


# --------------------------------------------------------------------------------------------- synthetic records


def _track(theta0, rate0, t0, t1, step, force):
    """The averaged ODE from (theta0, rate0) at t0 to t1 [rad, rad/yr]: (theta, theta_dot) on the grid."""
    t = np.arange(t0, t1 + step / 2, step)
    sol = solve_ivp(
        lambda _, y: [y[1], -3 * MU_EARTH * N_EARTH**2 * force(y[0])],
        (t0, t[-1]),
        [theta0, rate0],
        t_eval=t,
        rtol=1e-11,
        atol=1e-13,
        max_step=step,
    )
    return sol.y


def _series(theta, theta_dot, t, name):
    """A 1E-1 record of a planar e = 0.2 body (varpi 0.3 rad) with theta = lambda - lambda_E: sigma = -theta."""
    nr = len(t)
    a = (2 * np.pi / (N_EARTH + theta_dot)) ** (2.0 / 3.0)
    lam_p = N_EARTH * t
    lam = lam_p + theta
    varpi = np.full(nr, 0.3)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full(nr, 0.2),
        inc=np.full(nr, 1e-3),
        Omega=np.zeros(nr),
        omega=varpi.copy(),
        varpi=varpi,
        angle_wrapped=np.mod(-theta, 2 * np.pi),
        resonance=ResonanceSpec.from_key('1E-1'),
        regime=Regime(step_yr=float(t[1] - t[0]), span_yr=float(t[-1] - t[0]), n_rows=nr),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name=name,
    )


# theta_dot of the circulation and of the loop where it starts at theta = 0 [rad/yr]: the entry needs no kick, the
# circulation turns once in 480 yr
CIRCULATION_RATE = math.radians(-0.75)
STEP_YR = 0.25


@lru_cache(maxsize=None)
def _loop_period_yr() -> float:
    """The period of the synthetic loop, from the truth itself: the mean spacing of the downward crossings of 0 over
    400 yr of the ODE started at (0, CIRCULATION_RATE)."""
    theta, _ = _track(0.0, CIRCULATION_RATE, 0.0, 400.0, 0.05, _qs_force_spline())
    t = np.arange(0.0, 400.0 + 0.025, 0.05)[: len(theta)]
    down = t[1:][(theta[:-1] > 0) & (theta[1:] <= 0)]
    return float(np.mean(np.diff(down)))


def _qs_inside_circulation(loop_periods: float):
    """Circulation down to theta = 0 at t1 = 400 yr, where the quasi-satellite force takes over (the loop starts at
    the circulation's rate), the loop for loop_periods of its own period, then a kick back to the circulation rate to
    1000 yr (an encounter, as 2002 PN's after its loop). Entry and exit both run downwards, as PN's."""
    force = _qs_force_spline()
    t1 = 400.0
    t2 = t1 + loop_periods * _loop_period_yr()
    before = np.arange(0.0, t1, STEP_YR)
    theta_before = CIRCULATION_RATE * (before - t1)
    theta_loop, rate_loop = _track(0.0, CIRCULATION_RATE, t1, t2, STEP_YR, force)
    after = np.arange(t1 + len(theta_loop) * STEP_YR, 1000.0 + STEP_YR / 2, STEP_YR)
    theta_after = theta_loop[-1] + CIRCULATION_RATE * (after - after[0] + STEP_YR)
    t = np.concatenate([before, t1 + STEP_YR * np.arange(len(theta_loop)), after])
    theta = np.concatenate([theta_before, theta_loop, theta_after])
    rate = np.concatenate([np.full(len(before), CIRCULATION_RATE), rate_loop, np.full(len(after), CIRCULATION_RATE)])
    return t, theta, rate


def test_the_synthetic_loop_is_a_quasi_satellite():
    """The truth of the synthetic record (no package code): theta stays in the quasi-satellite well about 0 (|theta| <
    17 deg, the e = 0.2 well of the W4 tests) and turns twice in its 1.2 periods, half a period apart."""
    t, theta, _ = _qs_inside_circulation(1.2)
    period = _loop_period_yr()
    loop = (t > 400.0) & (t <= 400.0 + 1.2 * period)
    th = theta[loop]
    assert np.max(np.abs(np.degrees(th))) < 17.0
    turns = t[loop][1:-1][np.diff(np.sign(np.diff(th))) != 0]
    assert len(turns) == 2
    assert turns[1] - turns[0] == pytest.approx(period / 2, rel=0.02)


def test_a_loop_inside_circulation_runs_the_pass_and_is_accepted():
    """1.2 periods of the loop (83 yr each; 1.1 small-amplitude P_QS of the package's well, 90.6 yr) inside
    circulation, two turns.
    The trigger measures the stay against the loop through its turns: the package's loop period (its own table) is the
    truth's (the independent ODE) within 1 %. The first pass at the 111-yr tadpole floor keeps no libration; the
    quasi-satellite pass keeps one librating episode centred in the well, and the row is transient (circulation before
    and after)."""
    t, theta, rate = _qs_inside_circulation(1.2)
    analysis = analyse_mmr(_series(theta, rate, t, 'qs-1.2'), S, scan_subresonances=False, scan_owner=False)
    columns = analysis.extra_columns
    assert columns['geo_qs_visit_loop_yr'] == pytest.approx(_loop_period_yr(), rel=0.01)
    assert columns['geo_qs_visit_needed_yr'] == pytest.approx(S.mmr_coorbital_qs_trigger_periods * columns['geo_qs_visit_loop_yr'])
    assert columns['geo_qs_visit_turns'] == 2
    assert columns['geo_qs_visit_yr'] >= columns['geo_qs_visit_needed_yr']
    assert columns['geo_qs_pass'] == 'kept'
    assert 'qs_trigger' in analysis.verdict.rules_fired
    names = [entry.name for entry in analysis.verdict.decision_path[:3]]
    assert names == [qs_module.QS_VISIT, qs_module.QS_FLOOR, qs_module.QS_TURN]
    assert str(analysis.verdict.status) == 'transient'
    well = analysis.geometry.floor_well
    accepted = [e for e in analysis.geometry.runs if e.accepted and well.holds(e)]
    assert len(accepted) == 1 and 400.0 <= accepted[0].t_start_yr and accepted[0].t_end_yr <= 400.0 + 1.2 * _loop_period_yr()


def test_a_slow_passage_through_the_well_without_a_turn_runs_no_pass():
    """theta drifts through the well at 0.25 deg/yr (a stay of more than one P_QS) and never turns: no pass, no rule."""
    t = np.arange(0.0, 1000.0 + STEP_YR / 2, STEP_YR)
    rate = math.radians(-0.25)
    theta = math.radians(60.0) + rate * t
    analysis = analyse_mmr(_series(theta, np.full_like(t, rate), t, 'slow-pass'), S, scan_subresonances=False, scan_owner=False)
    columns = analysis.extra_columns
    assert columns['geo_qs_visit_yr'] >= columns['geo_qs_visit_needed_yr']
    assert columns['geo_qs_visit_turns'] == 0
    assert columns['geo_qs_pass'] == ''
    assert 'qs_trigger' not in analysis.verdict.rules_fired
    assert not any(entry.name == qs_module.QS_TURN for entry in analysis.verdict.decision_path)
