"""W36 F4: an accepted episode of a secular angle that is a wave on a slower motion of the angle is no episode of it
(`secular.decide.slow_waves`, SECULAR_SLOW_WAVES), and the slow scale may be that of the centre's own turns
(SECULAR_CENTRE_TURNS_SCALE, the MMR G2c second scale).

The author 2026-11-07: 101703 g-2g6+g7 ("the system caught short librations; is there not a global libration?"),
348018 g-g5+s-s7 (non_resonant), 56700 s+s7-2s6 (extend: a slow libration we cannot see).

Every expectation below comes from the formula of the synthetic angle, not from the package: the turning points of a
cosine, the range of an arc, the rate of a ramp.
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.secular import LongPeriodSeries
from resonances.resonance.analysis.secular import decide
from resonances.resonance.analysis.secular.analyse import analyse_long_period, analyse_secular
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import Action, AngleStatus, ResonanceType, Status
from tests.resonances.analysis.builders import make_episode

SETTINGS = settings_from_config()
NO_WAVES = replace(SETTINGS, secular_slow_waves=False)
T = np.arange(0.0, 20.0e6 + 1000.0, 2000.0)  # 20 Myr at 2 kyr: P_c = 100 kyr (no planetary modes), P_floor 200 kyr
DEG = np.radians


def _series(sigma) -> LongPeriodSeries:
    one = np.ones_like(T)
    return LongPeriodSeries(ResonanceType.SECULAR, 's-s6', 'synthetic', T, sigma, 2.4 * one, 0.1 * one, 0.3 * one)


# a 12-Myr libration of half-amplitude 150 deg (maxima at 5.5 and 17.5 Myr, minimum at 11.5 Myr) under a 1-Myr wave
# of 10 deg; the wave turns sigma back only near the slow turns (slow rate 150 x 2 pi / 12 = 79 deg/Myr against the
# wave's 63 deg/Myr), so the fine scale sees three 3-TP 'episodes' there (101703 g-2g6+g7)
SWING = DEG(160.0) + DEG(150.0) * np.cos(2 * np.pi * (T - 5.5e6) / 12.0e6) + DEG(10.0) * np.sin(2 * np.pi * T / 1.0e6)


def _slow_swing_deg(t):  # the slow motion of SWING, without its wave [deg]
    return 160.0 + 150.0 * np.cos(2 * np.pi * (t - 5.5e6) / 12.0e6)


# an arc of 300 deg, 150 (1 - cos(pi t / 20 Myr)), under a 1.5-Myr wave of 6 deg: the arc is slow at both ends, where
# the wave turns sigma back (56700 s+s7-2s6: a 13-deg ripple on a 322-deg arc)
ARC = DEG(150.0) * (1.0 - np.cos(np.pi * T / 20.0e6)) + DEG(6.0) * np.sin(2 * np.pi * T / 1.5e6)
# a circulation at 80 deg/Myr (4.4 turns) whose 0.35-Myr wave grows to 19 deg near 7 Myr (348018 g-g5+s-s7)
STALL = DEG(80.0) * T / 1.0e6 + (DEG(16.0) * np.exp(-(((T - 7.0e6) / 0.4e6) ** 2)) + DEG(3.0)) * np.sin(2 * np.pi * T / 0.35e6)
# a capture: 900 deg/Myr of circulation until 5 Myr, then a libration of 50 deg and 0.25 Myr to the end (19289 s-s6)
CAPTURE = np.where(T < 5.0e6, -DEG(900.0) * (T - 5.0e6) / 1.0e6, 0.0) + np.pi
CAPTURE = CAPTURE + DEG(50.0) * (T >= 5.0e6) * np.sin(2 * np.pi * (T - 5.0e6) / 0.25e6)
# Codex's W36 review: 0.2-rad librations of 1 Myr about 0 (0-4 Myr) and about 3 rad (10-20 Myr), joined by the smooth
# monotone transition 1.5 (1 - cos(pi (t - 4 Myr) / 6 Myr)) without a libration
_STEP = np.clip((T - 4.0e6) / 6.0e6, 0.0, 1.0)
CODEX = 1.5 * (1.0 - np.cos(np.pi * _STEP)) + 0.2 * np.sin(2 * np.pi * T / 1.0e6) * ((T <= 4.0e6) | (T >= 10.0e6))


def _entries(analysis, name):
    return [entry for entry in analysis.ledger if entry.name == name]


def test_ripples_at_the_turns_of_a_slow_libration_yield_to_it():
    off = analyse_long_period(_series(SWING), NO_WAVES)
    ripples = [run for run in (off.fine_geometry or off.geometry).runs if run.accepted]
    assert [round(float(np.degrees(run.half_amp_max_rad))) for run in ripples] == [10, 10, 10]
    analysis = analyse_long_period(_series(SWING), SETTINGS)
    assert analysis.fine_geometry.slow_wave_ids == frozenset(run.episode_id for run in ripples)
    # R = the 300-deg swing (+ the wave): A_max / (R / 2) = 10 / 150 within the wave's share
    ratios = [entry.value for entry in _entries(analysis, decide.SLOW_SWING_RIPPLE)]
    assert ratios == pytest.approx([10.0 / 155.0] * 3, abs=0.005)
    # the slow scale decides: one libration from the maximum at 5.5 Myr to the one at 17.5 Myr
    assert analysis.scale == 'centre'
    (run,) = [run for run in analysis.geometry.runs if run.accepted]
    assert run.tp_times_yr[0] == pytest.approx(5.5e6, abs=0.1e6) and run.tp_times_yr[-1] == pytest.approx(17.5e6, abs=0.1e6)
    assert run.period_med_yr == pytest.approx(12.0e6, rel=0.02)
    verdict = analyse_secular(_series(SWING), SETTINGS).verdict
    assert verdict.status == Status.RESONANT and verdict.angle_status == AngleStatus.LIBRATION


def _carry_by_formula(slow_deg, first: float, last: float) -> float:
    """The carry of an episode by the slow motion `slow_deg(t)` (the formula of the synthetic angle, no filter): its
    range over the episode per year, over its fastest range per year on an arm of as long again on either side (inside
    the record, an arm cut below a quarter of the span not read)."""
    span = last - first

    def rate(start, end):
        start, end = max(start, T[0]), min(end, T[-1])
        if end - start < 0.25 * span:
            return np.nan
        grid = np.linspace(start, end, 2001)
        return float(np.ptp(slow_deg(grid))) / (end - start)

    return rate(first, last) / np.nanmax([rate(first - span, first), rate(last, last + span)])


def test_the_slow_swing_carries_its_ripples():
    """A wave at a smooth turn of the slow motion is carried by it: the slow cosine moves inside each ripple at a share
    of its rate beside it that the cosine itself gives (for a parabola centred on the episode (1/2)^2 / ((3/2)^2 -
    (1/2)^2) = 1/8, more off centre), far above SECULAR_SLOW_SWING_CARRY_RATIO. The package reads the slow track
    low-passed at 3 P (2.7 Myr): the last ripple's arm ends 0.9 Myr from the record's end, inside the filter's edge
    spread, and reads 0.30 against the cosine's 0.24; the two others match the cosine."""
    off = analyse_long_period(_series(SWING), NO_WAVES)
    ripples = [run for run in (off.fine_geometry or off.geometry).runs if run.accepted]
    expected = [_carry_by_formula(_slow_swing_deg, run.tp_times_yr[0], run.tp_times_yr[-1]) for run in ripples]
    assert all(value >= 1 / 8 for value in expected)
    carried = [entry.value for entry in _entries(analyse_long_period(_series(SWING), SETTINGS), decide.SLOW_SWING_CARRIED)]
    assert len(carried) == 3 and carried[:2] == pytest.approx(expected[:2], abs=0.01)
    assert min(carried) >= 1 / 8 >= 2 * SETTINGS.secular_slow_swing_carry_ratio


def test_two_stationary_librations_joined_by_a_transition_stay_episodes():
    """Codex's W36 review: 0.2-rad librations of 1 Myr about 0 for 0-4 Myr and about 3 rad for 10-20 Myr, joined by a
    smooth monotone transition. The swing between the record's edges is R = 3 + 2 x 0.2 = 3.4 rad, so A_max / (R/2) =
    0.2 / 1.7 = 0.118 <= 0.5, but no slow motion carries either libration: on its plateau the slow track holds still
    (only the filter's leak), beside it the transition runs (after the first: 1.5 (1 - cos(pi 3.25 / 6)) = 1.70 rad
    over the 3.5-Myr arm). Both stay episodes, as with the rule off: transient, episodic."""
    on, off = analyse_long_period(_series(CODEX), SETTINGS), analyse_long_period(_series(CODEX), NO_WAVES)
    assert on.geometry.slow_wave_ids == frozenset()
    runs = [run for run in on.geometry.runs if run.accepted]
    assert [round(run.tp_times_yr[0] / 1.0e6, 2) for run in runs] == [0.25, 10.25]
    assert [entry.value for entry in _entries(on, decide.SLOW_SWING_RIPPLE)] == pytest.approx([0.2 / 1.7] * 2, abs=0.002)
    carried = [entry.value for entry in _entries(on, decide.SLOW_SWING_CARRIED)]
    assert len(carried) == 2 and all(value < SETTINGS.secular_slow_swing_carry_ratio for value in carried)
    verdicts = [analyse_secular(_series(CODEX), settings).verdict for settings in (SETTINGS, NO_WAVES)]
    assert [(v.status, v.angle_status) for v in verdicts] == [(Status.TRANSIENT, AngleStatus.EPISODIC)] * 2
    assert on.verdict.angle_status == off.verdict.angle_status == AngleStatus.EPISODIC


def test_an_arm_the_record_cuts_is_read_from_its_setting():
    """SECULAR_STALL_ARM_MIN_FRACTION: an episode with turning points from 1 to 19 Myr of a 20-Myr ramp has arms of 1 Myr
    on either side, 1/18 = 0.056 of its span: not read at the default 0.25 (nor at 0.06), read at 0.05 (the ramp's
    own rate on both sides, so its rate ratio is 1)."""
    episode = make_episode(tp_times_yr=np.arange(1.0e6, 19.0e6 + 1.0, 1.0e6), period_med_yr=2.0e6)
    ramp = DEG(80.0) * T / 1.0e6
    assert SETTINGS.secular_stall_arm_min_fraction == 0.25
    for fraction, read in ((0.25, False), (0.06, False), (0.05, True)):
        ratio, sweep = decide.stall_reading(T, ramp, episode, replace(SETTINGS, secular_stall_arm_min_fraction=fraction))
        assert np.isfinite(ratio) == read and np.isfinite(sweep) == read
    ratio, _ = decide.stall_reading(T, ramp, episode, replace(SETTINGS, secular_stall_arm_min_fraction=0.05))
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_the_scale_of_the_centre_turns_keeps_the_last_slow_turn():
    """W_c = 3 P_cyc of the 1-Myr ripples (2.7 Myr) trims the record to 2.7-17.3 Myr and loses the maximum at 17.5 Myr:
    two turning points, too few; the turns of the centre give W = 2.5 Myr, whose window keeps it."""
    analysis = analyse_long_period(_series(SWING), replace(SETTINGS, secular_centre_turns_scale=False))
    assert analysis.verdict.angle_status == AngleStatus.INSUFFICIENT_TIME
    turns = analyse_long_period(_series(SWING), SETTINGS)
    assert turns.angle.cutoff_yr < 2.7e6 and turns.angle.window[1] > 17.3e6
    (run,) = [run for run in turns.geometry.runs if run.accepted]
    assert run.n_tp == 3 and run.tp_times_yr[-1] > 17.3e6


def test_a_ripple_on_an_unfinished_arc_is_no_episode():
    off = analyse_secular(_series(ARC), NO_WAVES).verdict
    assert off.status == Status.TRANSIENT and off.angle_status == AngleStatus.EPISODIC
    analysis = analyse_long_period(_series(ARC), SETTINGS)
    ratios = [entry.value for entry in _entries(analysis, decide.SLOW_SWING_RIPPLE)]
    # R = the 300-deg arc + the wave (<= 312 deg); a half-swing is the wave's 6 deg plus half the arc's rise over half a
    # wave period (<= 23.6 deg/Myr x sin(pi 2.5 / 20) x 0.75 Myr / 2 = 3.4 deg near the ends): 6/156 <= ratio <= 9.4/150
    assert len(ratios) == 2 and all(6.0 / 156.0 <= ratio <= 9.4 / 150.0 for ratio in ratios)
    verdict = analyse_secular(_series(ARC), SETTINGS).verdict
    assert verdict.status == Status.UNCERTAIN and verdict.angle_status == AngleStatus.INSUFFICIENT_TIME
    assert verdict.action == Action.EXTEND


def test_a_stall_of_a_circulation_is_no_episode():
    off = analyse_long_period(_series(STALL), NO_WAVES)
    (stall,) = [run for run in off.geometry.runs if run.accepted]
    assert 6.5e6 < stall.tp_times_yr[0] and stall.tp_times_yr[-1] < 7.5e6
    assert analyse_secular(_series(STALL), NO_WAVES).verdict.status == Status.TRANSIENT
    # the slow track of a ramp is the ramp: 80 deg/Myr inside the episode and beside it
    angle = off.angle
    ratio, beside = decide.stall_reading(np.asarray(angle.t_yr), np.asarray(angle.angle_raw), stall, SETTINGS)
    assert ratio == pytest.approx(1.0, abs=0.05)
    span_myr = (stall.tp_times_yr[-1] - stall.tp_times_yr[0]) / 1.0e6
    assert np.degrees(beside) == pytest.approx(80.0 * span_myr, rel=0.05)  # beside: the ramp over as long again
    assert np.degrees(beside) > np.degrees(stall.half_amp_max_rad)  # the ramp carries sigma out of the band
    verdict = analyse_secular(_series(STALL), SETTINGS).verdict
    assert verdict.status == Status.NON_RESONANT and verdict.angle_status == AngleStatus.CIRCULATION


def test_a_capture_stops_the_circulation_and_stays():
    on, off = analyse_long_period(_series(CAPTURE), SETTINGS), analyse_long_period(_series(CAPTURE), NO_WAVES)
    (capture,) = [run for run in on.geometry.runs if run.accepted]
    assert capture.tp_times_yr[0] == pytest.approx(5.0e6, abs=0.2e6) and capture.n_tp > 100
    assert on.geometry.slow_wave_ids == frozenset()
    # inside: a still slow track; beside: 900 deg/Myr
    ratio, _ = decide.stall_reading(np.asarray(on.angle.t_yr), np.asarray(on.angle.angle_raw), capture, SETTINGS)
    assert ratio < 0.05
    assert (on.verdict.status, on.verdict.angle_status) == (off.verdict.status, off.verdict.angle_status)


def test_a_slow_wave_is_excluded_from_the_episodes_and_named():
    analysis = analyse_long_period(_series(STALL), SETTINGS)
    geometry = analysis.geometry
    (wave,) = [run for run in geometry.runs if run.accepted]
    assert geometry.slow_wave_ids == {wave.episode_id} and wave.episode_id in geometry.excluded_ids
    assert geometry.episodes == [] and decide.librating_episodes(geometry) == []
    eligibility = g.episode_eligibility(geometry, lambda episode: 0.0)
    assert eligibility[wave.episode_id] == g.RejectReason.SLOW_WAVE
    assert g.Geometry.__dataclass_fields__['slow_wave_ids'].default == frozenset()  # MMR and ZLK never set it


def test_the_rule_reads_the_secular_switch_only():
    off = replace(SETTINGS, secular_slow_waves=False)
    analysis = analyse_long_period(_series(STALL), off)
    assert analysis.geometry.slow_wave_ids == frozenset() and not _entries(analysis, decide.STALL_RATE)
