"""W28 B7 (card g07-4 A): the G2c scale ladder (`geometry.slow_scale_rung`, `analyse._at_ladder_scale`) and the cap
slow_scale_provisional, plus the libration period the retry at the libration scale reads (`analyse._libration_period`).

The synthetic angle: sigma = 1.3 + 1.9 sin(2 pi t / 45 kyr) + 0.05 sin(2 pi t / 3 kyr + 0.4) rad over 100 kyr, every 50
yr, at the fine cutoff 600 yr (385533 7N-12: a 45-kyr libration of 110 deg, a 2-kyr wave of 3 deg). The slow extrema
are at t = 45 kyr (1/4 + k/2) = 11.25, 33.75, 56.25 and 78.75 kyr. At the centre scale W_c = P / 2 = 22.5 kyr the edge
trim keeps 22.5-77.5 kyr = 1.22 P: two extrema (33.75, 56.25). The 0.05-rad wave is below p_c = 10 deg everywhere.
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.mmr import analyse, io, models
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import (
    Action,
    AngleStatus,
    Confidence,
    MarginClass,
    MarginValue,
    Reason,
    Status,
    Verdict,
)
from tests.resonances.analysis.builders import make, make_episode, make_geometry, make_turning_points

S = settings_from_config()
P_SLOW, W_C, P_C = 45_000.0, 22_500.0, 600.0
PC = np.radians(S.libration_centre_tp_prominence_deg)
SLOW_EXTREMA = [11_250.0, 33_750.0, 56_250.0, 78_750.0]


def _angle(span=100_000.0, p_fast=3000.0, a_fast=0.05, extra=None):
    t = np.arange(0.0, span + 25.0, 50.0)
    sigma = 1.3 + 1.9 * np.sin(2 * np.pi * t / P_SLOW) + a_fast * np.sin(2 * np.pi * t / p_fast + 0.4)
    if extra is not None:
        sigma = sigma + extra[1] * np.sin(2 * np.pi * t / extra[0])
    f = filtering.lowpass(t, sigma, P_C, S.filter_order)
    return geo.FilteredAngle(
        t_yr=t,
        angle_raw=sigma,
        angle_f=f,
        rate_f=np.gradient(f, t),
        angle_f_half=f,
        angle_f_double=f,
        slow_f=np.zeros_like(t),
        identity_coefficient=float('nan'),
        cutoff_yr=P_C,
        step_yr=50.0,
        window=filtering.edge_window(t, P_C),
    )


def _fast(angle):
    return geo.fast_period_yr(geo.find_turning_points(angle.t_yr, angle.angle_f, np.radians(1.0), tuple(angle.window)))


# --- pure helpers, by hand ----------------------------------------------------------------------------------------


def test_fast_period_is_the_shortest_full_period():
    # full periods t_{i+2} - t_i: 1950, 19 000, 21 050 -> 1950 (385533's fine TPs 43.5/44.2/45.5 kyr at a flat top)
    assert geo.fast_period_yr(make_turning_points([0.0, 1000.0, 1950.0, 20_000.0, 23_000.0])) == 1950.0
    assert np.isnan(geo.fast_period_yr(make_turning_points([0.0, 1000.0])))


def test_slow_extrema_shift_in_the_common_window():
    a = make_turning_points([10_000.0, 30_000.0, 50_000.0, 70_000.0])  # max, min, max, min
    b = make_turning_points([31_000.0, 49_100.0])  # built as a max, then a min
    # in b's window (25-55 kyr) a holds 30 (min) and 50 (max), b 31 (max) and 49.1 (min): kinds differ -> inf
    assert geo.slow_extrema_shift(a, (0.0, 80_000.0), b, (25_000.0, 55_000.0), 40_000.0) == np.inf
    c = replace(b, kinds=np.array([-1, 1]))
    # paired in order: |30 - 31| = 1 kyr, |50 - 49.1| = 0.9 kyr -> 1 / 40 = 0.025
    assert geo.slow_extrema_shift(a, (0.0, 80_000.0), c, (25_000.0, 55_000.0), 40_000.0) == pytest.approx(0.025)
    # one more TP in the common window -> inf; no TP in it -> 0
    assert geo.slow_extrema_shift(a, (0.0, 80_000.0), c, (5_000.0, 55_000.0), 40_000.0) == np.inf
    assert geo.slow_extrema_shift(a, (0.0, 80_000.0), c, (80_500.0, 90_000.0), 40_000.0) == 0.0


def test_ladder_cutoffs_by_hand():
    # 385533: top 22 717 yr, floor 3 x 1950 = 5850 yr, below W_c 22 744 yr: 0.71, 0.5, 0.35 x 22 717 (0.25 x = 5679 < 5850)
    rungs = geo.ladder_cutoffs(22_717.0, 5850.0, 22_744.0, (0.71, 0.5, 0.35, 0.25, 0.18, 0.125))
    assert rungs == pytest.approx([16_129.07, 11_358.5, 7_950.95])
    assert geo.ladder_cutoffs(22_717.0, 30_000.0, 22_744.0, (0.71, 0.5)) == []


# --- the ladder on the synthetic slow libration ----------------------------------------------------------------------


def test_the_ladder_recovers_the_slow_extrema_the_centre_scale_trims():
    angle = _angle()
    coarse = geo.angle_at_centre_window(angle, W_C, S)
    reference = geo.find_turning_points(coarse.t_yr, coarse.angle_f, PC, tuple(coarse.window))
    assert reference.times_yr == pytest.approx(SLOW_EXTREMA[1:3], abs=500)  # W_c sees two extrema only
    fast = _fast(angle)
    assert 2000 < fast < 3000  # the 3-kyr wave's TPs near the slow extrema (measured 2550 yr)
    rung = geo.slow_scale_rung(angle, coarse, PC, P_SLOW, fast, S)
    assert rung is not None
    # floor 3 x 2550 = 7650 yr; rungs 0.71/0.5/0.35 x 22.5 kyr; 0.35 x 22.5 = 7875 yr is the last one above the floor
    assert rung.floor_yr == pytest.approx(S.libration_centre_cycles * fast)
    assert rung.cutoff_yr == pytest.approx(0.35 * W_C)
    assert (rung.n_tp, rung.n_tp_centre_scale) == (4, 2)
    turns = geo.find_turning_points(rung.angle.t_yr, rung.angle.angle_f, PC, tuple(rung.angle.window))
    assert turns.times_yr == pytest.approx(SLOW_EXTREMA, abs=0.02 * P_SLOW)
    assert rung.shift <= S.mmr_scale_tp_time_tol


def test_the_walk_stops_where_a_wave_reaches_the_centre_prominence():
    # a 4-kyr wave of 0.4 rad: below 4.05 kyr it adds turning points at p_c (16 at 4050 yr), so with P_fast given as
    # 900 yr (floor 2.7 kyr) the walk stops there and keeps 0.25 x 22.5 = 5625 yr, the last rung with the 4 extrema
    angle = _angle(extra=(4000.0, 0.4))
    coarse = geo.angle_at_centre_window(angle, W_C, S)
    rung = geo.slow_scale_rung(angle, coarse, PC, P_SLOW, 900.0, S)
    assert rung is not None and rung.cutoff_yr == pytest.approx(0.25 * W_C) and rung.n_tp == 4
    below = geo.angle_at_centre_window(angle, 0.18 * W_C, S)
    assert len(geo.find_turning_points(below.t_yr, below.angle_f, PC, tuple(below.window))) > 4


def test_fast_and_slow_too_close_keep_nothing():
    # a fast wave of 8 kyr: 3 x 8 kyr = 24 kyr is above the top rung P / 2 = 22.5 kyr, no rung suppresses it
    angle = _angle()
    coarse = geo.angle_at_centre_window(angle, W_C, S)
    assert geo.slow_scale_rung(angle, coarse, PC, P_SLOW, 8000.0, S) is None
    # no fast wave measured (fewer than three fine TPs): nothing to place the floor, no ladder
    assert geo.slow_scale_rung(angle, coarse, PC, P_SLOW, float('nan'), S) is None
    # the ladder off
    assert geo.slow_scale_rung(angle, coarse, PC, P_SLOW, 2550.0, replace(S, mmr_scale_ladder=())) is None


def test_a_centre_scale_that_holds_two_slow_periods_keeps_its_scale():
    # 200 kyr: W_c's window 22.5-177.5 kyr holds 3.4 P >= libration_centre_min_estimates: the scale withholds nothing
    angle = _angle(span=200_000.0)
    coarse = geo.angle_at_centre_window(angle, W_C, S)
    assert (coarse.window[1] - coarse.window[0]) / P_SLOW > S.libration_centre_min_estimates
    assert geo.slow_scale_rung(angle, coarse, PC, P_SLOW, _fast(angle), S) is None


def test_the_ladder_settings_validate():
    with pytest.raises(ValueError):
        settings_from_config(mmr_scale_ladder=(0.5, 1.0))  # a rung as long as its top
    with pytest.raises(ValueError):
        settings_from_config(mmr_scale_tp_time_tol=1.5)
    assert settings_from_config(mmr_scale_ladder=()).mmr_scale_ladder == ()


# --- the period the retry reads and the cap ------------------------------------------------------------------------


def _analysis(period_cycle_yr, episode_period_yr):
    episode = make_episode(2, episode_id=3, period_med_yr=episode_period_yr)
    geometry = make_geometry(runs=[episode], period_cycle_yr=period_cycle_yr)
    return make(models.MMRAnalysis, geometry=geometry, decisive_episode_id=3)


def test_the_return_period_reaches_the_retry():
    # 385533 at W_c 22.7 kyr: two TPs, no P_cyc, its one cycle closed by the phase return 33.30 -> 78.74 kyr
    assert analyse._libration_period(_analysis(float('nan'), 78_737.95 - 33_303.43)) == pytest.approx(45_434.52)
    assert analyse._libration_period(_analysis(27_935.0, 45_434.52)) == 27_935.0  # P_cyc first
    assert np.isnan(analyse._libration_period(_analysis(float('nan'), float('nan'))))


def _verdict(l4, r0, confidence=Confidence.HIGH):
    margins = {
        'L4': MarginValue(l4, None, MarginClass.DECISIVE),
        'R0': MarginValue(r0, None, MarginClass.DECISIVE),
        'F': MarginValue(Margin.FULL, 1.0, MarginClass.SUPPORT),
    }
    status = Status.RESONANT if confidence != Confidence.NONE else Status.UNCERTAIN
    reason = Reason.DYNAMIC_LIBRATION if confidence != Confidence.NONE else Reason.ORIGIN_UNKNOWN
    action = Action.NONE if confidence != Confidence.NONE else Action.REVIEW
    return make(Verdict, status=status, reason=reason, confidence=confidence, confidence_raw=confidence, margins=margins, action=action)


def test_the_cap_holds_a_ladder_libration_without_its_own_force_law_at_medium():
    analysis = make(models.MMRAnalysis, verdict=_verdict(Margin.NOT_MEASURED, Margin.NOT_MEASURED))
    analyse._slow_scale_cap(analysis)
    verdict = analysis.verdict
    assert str(Cap.SLOW_SCALE_PROVISIONAL) in verdict.caps
    assert (verdict.confidence, verdict.confidence_raw, verdict.action) == (Confidence.MEDIUM, Confidence.HIGH, Action.SPOT_CHECK)


@pytest.mark.parametrize('l4, r0', [(Margin.FULL, Margin.NOT_MEASURED), (Margin.NOT_MEASURED, Margin.THIN)])
def test_a_measured_force_law_or_potential_lifts_the_cap(l4, r0):
    analysis = make(models.MMRAnalysis, verdict=_verdict(l4, r0))
    analyse._slow_scale_cap(analysis)
    assert analysis.verdict.caps == [] and analysis.verdict.confidence == Confidence.HIGH


def test_an_uncertain_ladder_verdict_keeps_no_level():
    analysis = make(models.MMRAnalysis, verdict=_verdict(Margin.NOT_MEASURED, Margin.NOT_MEASURED, Confidence.NONE))
    analyse._slow_scale_cap(analysis)
    assert analysis.verdict.confidence == Confidence.NONE and analysis.verdict.action == Action.REVIEW


# --- benchmark records (local data) --------------------------------------------------------------------------------


def _analysed(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    return analyse.analyse_mmr(io.load_series(example), S)


@pytest.mark.benchmark
def test_385533_is_read_at_the_ladder_cutoff_and_capped_at_medium():
    analysis = _analysed('mmr2__385533__7N-12+0+5')
    verdict, numbers = analysis.verdict, analysis.numbers_row()
    assert (verdict.status, verdict.angle_status, verdict.confidence) == (Status.RESONANT, AngleStatus.LIBRATION, Confidence.MEDIUM)
    assert str(Cap.SLOW_SCALE_PROVISIONAL) in verdict.caps and 'scale_ladder' in verdict.rules_fired
    # W_c 22 744 yr (two extrema) -> 0.35 x 22 717 = 7951 yr above 3 x 1950 yr (four extrema, 19.5-89.5 kyr)
    assert numbers['geo_scale_ladder_above_yr'] == pytest.approx(22_744.25, rel=1e-3)
    assert numbers['geo_scale_ladder_cutoff_yr'] == pytest.approx(7951.0, rel=1e-3)
    assert (numbers['geo_scale_ladder_n_tp'], numbers['geo_scale_ladder_n_tp_centre']) == (4, 2)
    episode = geo.run_of(analysis.geometry, analysis.decisive_episode_id)
    assert (episode.t_start_yr, episode.t_end_yr) == (pytest.approx(20_183, abs=2000), pytest.approx(89_059, abs=2000))


@pytest.mark.benchmark
@pytest.mark.parametrize('example', ['mmr2__136108__7N-12+0+5', 'mmr2__15875__2N-3+0+1', 'mmr2__182223__7N-19+0+12'])
def test_centre_scale_controls_keep_their_scale(example):
    # 136108 (2.6 P in W_c's window) and 15875 (3.5 P) need no ladder; 182223 (1.7 P) has a rung at 5976 yr whose
    # analysis decides no libration: each keeps the verdict at its centre scale
    numbers = _analysed(example).numbers_row()
    assert np.isnan(numbers['geo_scale_ladder_cutoff_yr'])
