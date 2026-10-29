"""W20 CS4: runs cut at regime changes (F2: a centre step; K3: a stall of the slow centre), 1:1 coverage by the union
of regimes (F2) and G5a, a turning arc of the slow centre instead of a drift (K1).

The records are built from their turning points (cosine half-cycles between them) or from a centre track plus a
sine, so every expected number follows from the construction:
- two regimes: 6 TPs alternating +30/-30 deg (centre 0, half-amplitude 30, half-period 90 yr), a transitional
  half-cycle from -30 to 115 deg, then 6 TPs alternating 115/55 deg (centre 85, half-period 85 yr; a full period of
  170 yr, above the 150-yr shortest libration). The full-cycle bands are
  [-30, 30] and [55, 115]: each regime passes the band test, the two together do not (30 < 55).
- the drift: c(t) = 2 pi t / 20 kyr plus 30 deg sin(2 pi t / 4.4 kyr): the centre moves 2 pi x 4.4 / 20 = 79 deg per
  period against a 30-deg swing, no stall; sigma runs back while cos(2 pi t / 4.4 kyr) < -(2 pi / 20) / (pi / 6 x 2 pi
  / 4.4) = -0.42, 37 % of each period, so the half-periods (1.6 and 2.8 kyr) keep one run. With c held for 40 kyr the
  stall holds 40 / 4.4 = 9 cycles.
"""

import math

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.filtering import edge_window, lowpass
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype
from tests.resonances.analysis.builders import make_sampling

S = settings_from_config()
ONE_DEG = math.radians(1.0)
SAMPLING = make_sampling(
    unwrap_resid_p99_rad=0.0, unwrap_resid_max_rad=0.0, unwrap_ok=True, undefined_phase=False, verdict='resolved', alias_numpy_differs=False
)
MMR_RULES = dict(edge_arc_rule=True, return_closure=True, split_regimes=True, cycle_travel=True)


def through(times, values, t):
    """sigma(t) through the turning points (times, values [deg]): a half-cosine between consecutive ones, mirrored
    beyond the first and the last [radians]. Every given point is an extremum."""
    times, values = np.asarray(times, float), np.radians(np.asarray(values, float))
    times = np.concatenate([[2 * times[0] - times[1]], times, [2 * times[-1] - times[-2]]])
    values = np.concatenate([[values[1]], values, [values[-2]]])
    out = np.full(t.shape, values[0])
    for (t0, v0), (t1, v1) in zip(zip(times, values), zip(times[1:], values[1:])):
        inside = (t >= t0) & (t <= t1)
        out[inside] = v0 + (v1 - v0) * (1 - np.cos(np.pi * (t[inside] - t0) / (t1 - t0))) / 2
    out[t > times[-1]] = values[-1]
    return out


def filtered(t, angle, cutoff, c=-30.0, **rules):
    angle_f = lowpass(t, angle, cutoff, 2)
    rate = np.gradient(angle_f, t)
    return g.FilteredAngle(
        t_yr=t,
        angle_raw=angle,
        angle_f=angle_f,
        rate_f=rate,
        angle_f_half=lowpass(t, angle, cutoff / 2, 2),
        angle_f_double=lowpass(t, angle, 2 * cutoff, 2),
        slow_f=rate / c,
        identity_coefficient=c,
        cutoff_yr=cutoff,
        step_yr=float(t[1] - t[0]),
        window=edge_window(t, cutoff),
        **rules,
    )


def two_regimes(circulation_turns: float = 0.0):
    """Centre 0 (+-30 deg, 90-yr half-periods) then centre 85 (+-30 deg, 85 yr), joined by one half-cycle of 80 yr;
    `circulation_turns` > 0: that many turns upwards between them instead (sigma then joins the second regime
    shifted by them)."""
    first_t = 60.0 + 90.0 * np.arange(6)
    first_v = [30.0, -30.0] * 3
    shift = 360.0 * circulation_turns
    gap = 80.0 if circulation_turns == 0 else 80.0 + 300.0 * circulation_turns
    second_t = first_t[-1] + gap + 85.0 * np.arange(6)
    second_v = [115.0 + shift, 55.0 + shift] * 3
    times, values = np.concatenate([first_t, second_t]), first_v + second_v
    t = np.arange(0.0, second_t[-1] + 60.0, 0.5)
    return t, through(times, values, t)


def test_short_regimes_are_not_cut():
    """Two regimes of 4 TPs each (one independent cycle, one centre each): nothing shows that either centre stays."""
    times = list(60.0 + 90.0 * np.arange(4)) + list(60.0 + 90.0 * 3 + 80.0 + 85.0 * np.arange(4))
    values = [30.0, -30.0] * 2 + [115.0, 55.0] * 2
    t = np.arange(0.0, times[-1] + 60.0, 0.5)
    geometry = g.measure_geometry(filtered(t, through(times, values, t), 10.0, **MMR_RULES), ONE_DEG, S)
    assert [str(run.reject_reason) for run in geometry.runs] == ['centre_moves']


def test_a_centre_step_cuts_the_run_into_two_regimes():
    t, sigma = two_regimes()
    geometry = g.measure_geometry(filtered(t, sigma, 10.0, **MMR_RULES), ONE_DEG, S)
    accepted = [run for run in geometry.runs if run.accepted]
    assert len(accepted) == 2 and all(run.n_tp == 6 for run in accepted)
    assert [round(math.degrees(run.centre_mean_rad)) for run in accepted] == [0, 85]
    # without the rule (secular, ZLK) the run is one, its centre moves
    plain = g.measure_geometry(filtered(t, sigma, 10.0), ONE_DEG, S)
    assert [str(run.reject_reason) for run in plain.runs] == ['centre_moves']


def test_a_swing_into_a_short_ripple_is_no_step():
    """A slow swing of 79 deg into 4 TPs of 2.8-deg ripple (2003 KQ18 6J+2S-1): the side of one open cycle is the swing,
    not a regime, and neither side has a measured centre (two independent cycles); nothing is cut."""
    times = [100.0, 500.0, 900.0, 1100.0, 1300.0, 1500.0, 1700.0]
    values = [186.0, 265.0, 263.3, 274.0, 269.7, 278.7, 273.0]
    t = np.arange(0.0, 1800.0, 0.5)
    tp = g.find_turning_points(t, through(times, values, t), ONE_DEG / 2, (50.0, 1750.0))
    assert len(tp) == 7
    assert g.split_at_regime_changes(tp, np.arange(7), S) is None


def drift_record(stall_kyr: float = 0.0):
    """c(t) = 2 pi t / 20 kyr, held for `stall_kyr` from 40 kyr, plus 30 deg sin(2 pi t / 4.4 kyr)."""
    t = np.arange(0.0, 100e3 + stall_kyr * 1e3, 10.0)
    start, stop = 40e3, 40e3 + stall_kyr * 1e3
    moving = np.where(t < start, t, np.where(t < stop, start, t - (stop - start)))
    sigma = 2 * np.pi * moving / 20e3 + math.radians(30.0) * np.sin(2 * np.pi * t / 4400.0)
    return t, sigma


def test_a_steady_drift_is_not_cut():
    t, sigma = drift_record()
    geometry = g.measure_geometry(filtered(t, sigma, 500.0, **MMR_RULES), ONE_DEG, S)
    (run,) = geometry.runs
    assert not run.accepted and str(run.reject_reason) == 'centre_moves' and run.n_tp > 40
    tp = geometry.turning_points
    assert g.split_at_regime_changes(tp, np.arange(len(tp)), S) is None


def test_a_stall_of_the_slow_centre_is_cut_out():
    t, sigma = drift_record(stall_kyr=40.0)
    geometry = g.measure_geometry(filtered(t, sigma, 500.0, **MMR_RULES), ONE_DEG, S)
    (episode,) = [run for run in geometry.runs if run.accepted]
    # the stall at 40-80 kyr: 40 / 4.4 = 9 cycles, so >= 19 turning points; the drift on either side stays rejected
    assert episode.n_tp >= 19 and (episode.n_tp - 1) // 2 >= 9
    assert 40e3 - 2200 <= episode.t_start_yr and episode.t_end_yr <= 80e3 + 2200  # within half a loop period
    assert [str(run.reject_reason) for run in geometry.runs if not run.accepted] == ['centre_moves', 'centre_moves']


# --- F2, 1:1 coverage by the union of regimes ------------------------------------------------------------------------


def decided(t, sigma, coorbital=True):
    geometry = g.measure_geometry(filtered(t, sigma, 10.0, coorbital=coorbital, **MMR_RULES), ONE_DEG, S)
    ledger = Ledger.from_settings(S)
    return decide.decide_angle(SAMPLING, False, geometry, None, S, ledger), geometry, ledger


def test_one_to_one_regimes_without_a_circulation_cover_the_record():
    angle, geometry, ledger = decided(*two_regimes())
    assert (angle.angle_status, angle.subtype) == (AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE)
    columns = decide.union_columns(geometry, S)
    assert columns['geo_union_n_episodes'] == 2
    # the turning points between the regimes: -30 deg (the last of the first) and 115 deg (the first of the second)
    assert columns['geo_union_tp_range_rad'] == pytest.approx(math.radians(145.0), abs=math.radians(1.0))
    assert 'geo_union_raw_range_rad' in ledger.columns()
    assert angle.rule == 'G2'
    # W22: an angle of another resonance joins too (the gap of 80 yr is below P_left + P_right = 180 + 170 yr), as
    # rule G2u, which the status tree reads as the two episodes it saw before (`decide.status_basis`)
    other, _, _ = decided(*two_regimes(), coorbital=False)
    assert (other.angle_status, other.subtype, other.rule) == (AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE, decide.RULE_UNION)
    basis = decide.status_basis(other)
    assert (basis.angle_status, basis.subtype, basis.episode_id) == (AngleStatus.EPISODIC, AngleSubtype.RECURRENT, other.episode_id)


def test_a_circulation_between_the_regimes_breaks_the_union():
    t, sigma = two_regimes(circulation_turns=1.5)
    angle, geometry, _ = decided(t, sigma)
    assert len([run for run in geometry.runs if run.accepted]) == 2
    assert angle.angle_status == AngleStatus.EPISODIC
    assert decide.union_columns(geometry, S)['geo_union_n_episodes'] == 0


# --- K1, G5a: a turning arc, not a drift ---------------------------------------------------------------------------


def ripple_on(centre_rad, t):
    """10-deg ripple of 900 yr on a slow centre: turning points everywhere, runs whose centre moves."""
    return centre_rad + math.radians(10.0) * np.sin(2 * np.pi * t / 900.0)


def g4_to_g6(sigma, t):
    geometry = g.measure_geometry(filtered(t, sigma, 300.0, **MMR_RULES), ONE_DEG, S)
    ledger = Ledger.from_settings(S)
    return decide._g4_to_g6(SAMPLING, geometry, S, ledger), geometry


def test_a_centre_that_turns_back_within_a_turn_is_an_arc():
    """2003 KQ18: the slow centre goes down 0.18 rev and back within 20 kyr."""
    t = np.arange(0.0, 20e3, 5.0)
    angle, geometry = g4_to_g6(ripple_on(-0.18 * 2 * np.pi * np.sin(np.pi * t / 20e3), t), t)
    assert geometry.drift_fraction >= S.libration_drift_min_window_fraction
    assert len(geometry.centre.turns) >= 1 and geometry.centre.range_rad < 2 * np.pi
    assert (angle.angle_status, angle.subtype, angle.rule) == (AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, 'G5a')


def test_a_centre_that_turns_twice_stays_a_drift():
    """101703 1V+3S-7 type: the centre goes down 0.18 rev, back up and down again within the record (two turns): the
    record holds a whole half-swing between the turns, not part of one."""
    t = np.arange(0.0, 30e3, 5.0)
    angle, geometry = g4_to_g6(ripple_on(-0.18 * 2 * np.pi * np.sin(np.pi * t / 15e3), t), t)
    assert len(geometry.centre.turns) == 2 and geometry.centre.range_rad < 2 * np.pi
    assert (angle.angle_status, angle.rule) == (AngleStatus.DRIFT, 'G5')


def test_a_monotone_drift_stays_a_drift():
    """0.8 rev in one direction over 20 kyr: 360 x 0.8 x 0.9 / 20 = 13 deg per ripple period, above its 10-deg swing,
    so no stall either; no turn of the centre and a pass of 0.8 rev < 1: G5 drift. (A monotone pass of a turn or more
    is G4 circulation, before G5.)"""
    t = np.arange(0.0, 20e3, 5.0)
    angle, geometry = g4_to_g6(ripple_on(-0.8 * 2 * np.pi * t / 20e3, t), t)
    assert len(geometry.centre.turns) == 0 and geometry.centre.max_pass_rad < 2 * np.pi
    assert (angle.angle_status, angle.rule) == (AngleStatus.DRIFT, 'G5')


# --- benchmark ----------------------------------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, expected',
    [
        ('coorbital__524522__1V-1+0+0', ('resonant', 'libration', 'regime_change', 'high')),
        ('mmr3__2003_KQ18__6J+2S-1+0+0-7', ('uncertain', 'insufficient_time', 'arc', '')),
        ('coorbital__2017_CY35__1V-1+0+0', ('transient', 'episodic', '', 'high')),
    ],
)
def test_benchmark_regime_rows(example, expected):
    """524522 1V-1 at 1 kyr (author 2026-10-19: QS -> tadpole without circulation, resonant/high), 2003 KQ18 (author:
    uncertain/insufficient_time, an arc), 2017 CY35 (a circulation into one episode: its own midpoints move ~12 deg)."""
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    v = analyse_mmr(io.load_series(example), S).verdict
    assert (str(v.status), str(v.angle_status), str(v.angle_subtype), str(v.confidence)) == expected


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example',
    [
        'coorbital__2017_UM52__1E-1+0+0',  # G2 exit observed
        'coorbital__2026_FW12__1E-1+0+0',  # G2 entry observed
        'coorbital__524522__1V-1+0+0',  # G2 1:1 union
        'mmr3__2003_KQ18__6J+2S-1+0+0-7',  # G5a
        'coorbital__2014_OL339__1E-1+0+0',  # G2c not after an observed exit
    ],
)
def test_the_new_ledger_numbers_reach_the_csv_and_the_figure(example):
    from tests.resonances.analysis.test_impl_w3_guards_ledger_columns import missing_numbers

    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    assert missing_numbers(analyse_mmr(io.load_series(example), S)) == []


def test_a_drift_of_three_half_swings_is_no_step():
    """sigma = 30 deg sin(2 pi t / 200 yr) + a centre moving 90 deg uniformly over 20 cycles: the whole run fails the
    band test (90 > 2 x 30), each half passes it (45 < 60), but no half-cycle carries the change: nothing is cut."""
    t = np.arange(0.0, 4000.0, 1.0)
    sigma = np.radians(30.0) * np.sin(2 * np.pi * t / 200.0) + np.radians(90.0) * t / 4000.0
    geometry = g.measure_geometry(filtered(t, sigma, 20.0, **MMR_RULES), ONE_DEG, S)
    assert [str(run.reject_reason) for run in geometry.runs] == ['centre_moves']
