"""W28 C16, rule regime_cut_mark (the author's q-5 i, the W28 C15 audit): a centre step or a stall of an MMR angle that
is not 1:1 is no boundary when sigma stays in one libration region across it; it becomes a mark of the joined episode.

The records are built from their turning points (half-cosines between them, `through`), so every number follows from
the construction:
- the joinable step: half-amplitude 20 deg, half-period 90 yr; 9 TPs whose half-cycle midpoints ramp 0 -> 15 deg, one
  step half-cycle of 90 yr, 9 TPs ramping 35 -> 50 deg. Each side passes the band test (its centres move 15 < 2 x 20
  deg); the whole run fails it (first band [-20, 20] against the last [30, 70] when the cycles of 20 deg about 0 and 50
  deg are compared: 50 > 40 deg), so the run is cut at the step (the centre jumps 20 deg there, more than the 15-deg
  range inside either side). The last full cycle before the step (centre ~13, half-amplitude 20) and the first after it
  (centre ~37) share 16 deg of band and one period (180 yr): one libration region, one episode with a mark.
- the same with a 50-deg jump: the facing cycles are ~45 deg apart, more than 2 x 20: two regions, the cut stays.
- the same with the right side's half-period 200 yr: a period ratio 400 / 180 = 2.2 > LIBRATION_REGULAR_PERIOD_RATIO
  (2): the cut stays; with 170 yr (340 / 180 = 1.9) it becomes a mark.
- the joinable step with every angle multiplied by `scale` (the same shape in units of the half-amplitude, so the
  facing cycles still share a band and a period): its turning points span from TP 1 (15/8 - 20 = -18.125 deg) to TP
  16 (35 + 15 x 7/8 + 20 = 68.125 deg), 86.25 deg; x4 is 345 deg, less than a turn, one episode; x4.5 is 388.1 deg,
  sigma takes every value across the step, no libration region: the cut stays (FIX-c16, 55390 1J+3S-1).
"""

import math

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.test_w20_regime_cuts import MMR_RULES, ONE_DEG, drift_record, filtered, through, two_regimes

S = settings_from_config()
OFF = settings_from_config(libration_regime_cut_marks=False)


def step_record(jump_deg: float = 20.0, right_half_period: float = 90.0, n: int = 9, scale: float = 1.0):
    """The step record of the module docstring: (t, sigma) with the step half-cycle between TP n-1 and TP n; every
    angle multiplied by `scale`."""
    left_mid = np.linspace(0.0, 15.0, n)
    right_mid = np.linspace(15.0 + jump_deg, 30.0 + jump_deg, n)
    left = [m + (20.0 if k % 2 == 0 else -20.0) for k, m in enumerate(left_mid)]
    right = [m + (20.0 if (k + n) % 2 == 0 else -20.0) for k, m in enumerate(right_mid)]
    left_t = 60.0 + 90.0 * np.arange(n)
    right_t = left_t[-1] + 90.0 + right_half_period * np.arange(n)
    times = np.concatenate([left_t, right_t])
    t = np.arange(0.0, times[-1] + 60.0, 0.5)
    return t, through(times, [scale * v for v in left + right], t)


def runs_of(t, sigma, settings):
    geometry = g.measure_geometry(filtered(t, sigma, 10.0, **MMR_RULES), ONE_DEG, settings)
    return geometry, [run for run in geometry.runs if run.accepted]


def test_edge_cycle_follows_the_full_cycle_convention():
    """Centre (v0 + 2 v1 + v2) / 4, half-amplitude the mean half-swing over two, period t2 - t0, by hand."""
    # first cycle of 10, -30, 14 at 0, 50, 100 yr: centre (10 - 60 + 14) / 4 = -9, half-swings 20 and 22 -> 21
    m, a, p = g.edge_cycle([0.0, 50.0, 100.0, 160.0], np.radians([10.0, -30.0, 14.0, -20.0]), last=False)
    assert (math.degrees(m), math.degrees(a), p) == pytest.approx((-9.0, 21.0, 100.0))
    # last cycle -30, 14, -20 at 50, 100, 160 yr: centre (-30 + 28 - 20) / 4 = -5.5, half-swings 22 and 17 -> 19.5
    m, a, p = g.edge_cycle([0.0, 50.0, 100.0, 160.0], np.radians([10.0, -30.0, 14.0, -20.0]), last=True)
    assert (math.degrees(m), math.degrees(a), p) == pytest.approx((-5.5, 19.5, 110.0))
    # two TPs: their half-cycle, the period twice it
    m, a, p = g.edge_cycle([0.0, 40.0], np.radians([10.0, -30.0]), last=True)
    assert (math.degrees(m), math.degrees(a), p) == pytest.approx((-10.0, 20.0, 80.0))
    assert np.isnan(g.edge_cycle([0.0], [0.0], last=True)[0])


def test_a_step_inside_one_libration_region_becomes_a_mark():
    t, sigma = step_record()
    _, apart = runs_of(t, sigma, OFF)
    assert len(apart) == 2 and all(str(run.shape) == 'stationary' for run in apart)  # the cut as it stood
    assert g.cycles_continue(apart[0], apart[1], S)
    geometry, (episode,) = runs_of(t, sigma, S)
    assert episode.t_start_yr == pytest.approx(apart[0].t_start_yr) and episode.t_end_yr == pytest.approx(apart[1].t_end_yr)
    # the joined run fails the band test only for its moving centre: accepted as a regime change, L6 not measured
    assert str(episode.shape) == 'regime_change' and episode.frame == g.FRAME_SIGMA
    assert episode.centre_stationarity == 'not_measured'
    # the mark is the step half-cycle: the last TP before the cut to the first after it
    assert episode.marks == f'centre_step@{apart[0].t_end_yr:.1f}:{apart[1].t_start_yr:.1f}'
    assert 775.0 < apart[0].t_end_yr < 785.0 and 865.0 < apart[1].t_start_yr < 875.0  # the TPs at 780 and 870 yr
    assert Rule.REGIME_CUT_MARK in geometry.rules_fired


def test_disjoint_bands_across_the_step_stay_two_episodes():
    """Two regions (the 2013 VA14 60 kyr stall, card g11-4 A): the facing cycles do not share a band."""
    t, sigma = step_record(jump_deg=50.0)
    geometry, accepted = runs_of(t, sigma, S)
    assert len(accepted) == 2 and not g.cycles_continue(accepted[0], accepted[1], S)
    assert not any(run.marks for run in geometry.runs) and Rule.REGIME_CUT_MARK not in geometry.rules_fired
    # the W20 two regimes (bands [-30, 30] and [55, 115]) likewise
    t, sigma = two_regimes()
    geometry, accepted = runs_of(t, sigma, S)
    assert len(accepted) == 2 and Rule.REGIME_CUT_MARK not in geometry.rules_fired


@pytest.mark.parametrize('scale, span_deg, joined', [(4.0, 345.0, True), (4.5, 388.125, False)])
def test_a_centre_that_travels_the_circle_across_the_cut_stays_cut(scale, span_deg, joined):
    """The facing cycles agree at any scale, but a joined stretch whose turning points span a turn holds no libration
    region (`within_one_turn`): the step stays a boundary at x4.5 and is a mark at x4."""
    t, sigma = step_record(scale=scale)
    _, apart = runs_of(t, sigma, OFF)
    assert len(apart) == 2 and g.cycles_continue(apart[0], apart[1], S)
    span = np.ptp(np.concatenate([apart[0].tp_values_rad, apart[1].tp_values_rad]))
    assert math.degrees(span) == pytest.approx(span_deg, abs=1.0)
    assert g.within_one_turn(np.concatenate([apart[0].tp_values_rad, apart[1].tp_values_rad])) is joined
    geometry, accepted = runs_of(t, sigma, S)
    assert len(accepted) == (1 if joined else 2)
    assert (Rule.REGIME_CUT_MARK in geometry.rules_fired) is joined
    if not joined:
        assert not any(run.marks for run in geometry.runs)
        assert [(run.t_start_yr, run.t_end_yr) for run in accepted] == [(run.t_start_yr, run.t_end_yr) for run in apart]


def test_within_one_turn_is_the_circle():
    assert g.within_one_turn(np.radians([0.0, 359.0, 10.0]))
    assert not g.within_one_turn(np.radians([-180.0, 180.0]))
    assert not g.within_one_turn(np.zeros(0))


@pytest.mark.parametrize('half_period, joined', [(200.0, False), (170.0, True)])
def test_the_period_must_continue_across_the_cut(half_period, joined):
    t, sigma = step_record(right_half_period=half_period)
    _, apart = runs_of(t, sigma, OFF)
    assert len(apart) == 2
    _, a_l, p_l = g.edge_cycle(apart[0].tp_times_yr, apart[0].tp_values_rad, last=True)
    _, a_r, p_r = g.edge_cycle(apart[1].tp_times_yr, apart[1].tp_values_rad, last=False)
    assert p_r / p_l == pytest.approx(2 * half_period / 180.0, rel=0.02)  # 2.2 or 1.9 against the ratio 2
    _, accepted = runs_of(t, sigma, S)
    assert len(accepted) == (1 if joined else 2)


def test_a_stall_between_drifts_stays_cut():
    """An accepted | rejected boundary always stays: the stall of the W20 drift record (40-80 kyr) between two drifts
    keeps its own episode with the switch on (385266 16N-27's drift)."""
    t, sigma = drift_record(stall_kyr=40.0)
    geometry = g.measure_geometry(filtered(t, sigma, 500.0, **MMR_RULES), ONE_DEG, S)
    assert [run.accepted for run in geometry.runs] == [False, True, False]
    assert not any(run.marks for run in geometry.runs) and Rule.REGIME_CUT_MARK not in geometry.rules_fired


def test_one_stall_mark_names_the_stall():
    """The mark of a stall cut is the stall's own span (its first and last TP), whichever side it joins."""
    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    tp = g.find_turning_points(
        np.arange(-5.0, 46.0, 0.5), through(times, [10.0, -10.0, 10.0, -10.0, 10.0], np.arange(-5.0, 46.0, 0.5)), ONE_DEG, (-5.0, 45.0)
    )
    assert len(tp) == 5
    stall = g.RegimeCut([np.arange(0, 2), np.arange(2, 5)], (1,))
    step = g.RegimeCut([np.arange(0, 2), np.arange(2, 5)], (0, 1))
    t2, t4 = tp.times_yr[2], tp.times_yr[4]
    assert g._regime_cut_mark(tp, stall, stall.pieces[0], stall.pieces[1]) == f'stall@{t2:.1f}:{t4:.1f}'
    assert g._regime_cut_mark(tp, step, step.pieces[0], step.pieces[1]) == f'centre_step@{tp.times_yr[1]:.1f}:{t2:.1f}'
