"""W36 F1: an exit (entry) the record shows ends a secular or ZLK libration (the edge rule on sigma_f).

DECISIONS 2026-10-15: "an arc > 2 pi is not a piece of libration". The MMR reads it on the orbit-averaged raw angle
(E1, `LIBRATION_EDGE_ARC_REV`); a secular angle has no orbit to average, so sigma_f itself is read. 2012 AX10 and
33325 2s-s4-s6 (the author 2026-11-07): one 3-TP episode ends 0.79 and 0.87 P before the end of 20 Myr, so the end
counted as covered, though sigma_f then runs 1.9 and 1.7 turns one way: transient, the record ends circulating.

The synthetic angles (built here, not by the package): a libration c + A sin(2 pi t / P), A = 60 deg, over ten
periods T = 10 P; its last maximum is at t_L = 9.25 P, 0.75 P before the end. After it sigma falls monotonically by
`turns` turns, c + A - 2 pi turns s^2 with s = (t - t_L) / (T - t_L) (no kink at the turning point). It leaves the
band |sigma - c| = pi at s^2 = (180 + 60) / (360 turns): s = 0.667 for 1.5 turns (t = 9.75 P, 0.25 P before the end).
- 1.5 turns: the exit is observed (arc 1.5 rev >= 1): episodic, transient, final_regime circulation from 9.75 P;
- 0.8 turn: the exit is only suspected (it crosses at s = 0.913, but the arc is < 1 turn, the 2026-10-15 rule):
  libration, flag exit_suspected;
- the 1.5-turn record reversed in time: the entry is observed, episodic, and the record ends librating.
Secular: P = 2 Myr (above the 200-kyr floor), ZLK omega about 90 deg: P = 100 kyr (floor 2 kyr).
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis.geometry import measure_geometry
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.secular import LongPeriodSeries
from resonances.resonance.analysis.secular import decide as secular_decide
from resonances.resonance.analysis.secular import preprocess
from resonances.resonance.analysis.secular.analyse import analyse_long_period, analyse_secular
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, Flag, ResonanceType, Status

SETTINGS = settings_from_config()
AMPLITUDE = np.radians(60.0)
SECULAR = (ResonanceType.SECULAR, 'g-g6', 2.0e6, 2000.0, 1.0)  # type, name, P [yr], step [yr], centre [rad]
ZLK = (ResonanceType.LIDOV_KOZAI, 'LK', 1.0e5, 100.0, np.pi / 2)


def _angle(period: float, step: float, centre: float, turns: float) -> tuple:
    """(t, sigma, t_L): ten periods of libration, the last maximum at t_L = 9.25 P, then `turns` turns down."""
    t = np.arange(0.0, 10 * period + step / 2, step)
    t_last = 9.25 * period
    sigma = centre + AMPLITUDE * np.sin(2 * np.pi * t / period)
    after = t > t_last
    s = (t[after] - t_last) / (t[-1] - t_last)
    sigma[after] = centre + AMPLITUDE - 2 * np.pi * turns * s**2
    return t, sigma, t_last


def _series(kind, turns: float, reverse: bool = False) -> LongPeriodSeries:
    resonance_type, name, period, step, centre = kind
    t, sigma, _ = _angle(period, step, centre, turns)
    if reverse:
        sigma = sigma[::-1].copy()
    one = np.ones_like(t)
    return LongPeriodSeries(resonance_type, name, 'synthetic', t, sigma, 2.4 * one, 0.1 * one, 0.3 * one)


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_a_turn_and_a_half_after_the_last_turning_point_is_an_observed_exit(kind):
    period = kind[2]
    analysis = analyse_long_period(_series(kind, 1.5), SETTINGS)
    geometry, verdict = analysis.geometry, analysis.verdict
    episode = max(secular_decide.librating_episodes(geometry), key=lambda e: e.t_end_yr)
    # the episode ends at the last maximum, 0.75 P before the end: by the gap alone that end is covered
    assert episode.t_end_yr == pytest.approx(9.25 * period, abs=0.02 * period)
    assert episode.coverage_end_gap_p <= SETTINGS.libration_coverage_periods
    # sigma_f runs 1.5 turns from there, monotonically: the exit is observed, not suspected
    assert geometry.exit.observed and not geometry.exit.suspected and episode.exit_observed
    assert geometry.exit.raw_arc_rad / (2 * np.pi) == pytest.approx(1.5, abs=0.05)
    assert geometry.exit.time_yr == pytest.approx(9.75 * period, abs=0.03 * period)
    assert str(Flag.EXIT_SUSPECTED) not in verdict.flags
    assert verdict.angle_status == AngleStatus.EPISODIC
    assert str(verdict.final_regime) == str(AngleStatus.CIRCULATION)
    assert verdict.final_regime_start_yr == pytest.approx(geometry.exit.time_yr)
    arc = next(entry for entry in verdict.decision_path if entry.name == secular_decide.EDGE_ARC_ENTRIES['exit'])
    assert arc.result is True and arc.threshold == SETTINGS.libration_edge_arc_rev


def test_the_secular_status_is_transient_and_the_last_episode_does_not_reach_the_end():
    verdict = analyse_secular(_series(SECULAR, 1.5), SETTINGS).verdict
    assert verdict.status == Status.TRANSIENT
    assert verdict.pol_last_reaches_end is False
    assert str(verdict.final_regime) == str(AngleStatus.CIRCULATION)


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_without_the_rule_the_same_angle_is_a_libration_with_a_suspected_exit(kind):
    """The edge rule is what changes the word: the same filtered angle with `edge_arc_filtered` off (as before W36)."""
    series = _series(kind, 1.5)
    cutoff = preprocess.choose_cutoff(series, SETTINGS)
    angle = replace(preprocess.filtered_angle(series, cutoff, SETTINGS), edge_arc_filtered=False)
    geometry = measure_geometry(angle, cutoff.prominence_rad, SETTINGS)
    assert geometry.exit.suspected and not geometry.exit.observed
    assert secular_decide.decide_angle(geometry, SETTINGS, Ledger.from_settings(SETTINGS)).angle_status == AngleStatus.LIBRATION


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_less_than_a_turn_after_the_last_turning_point_stays_a_suspected_exit(kind):
    """0.8 turn: sigma_f leaves the band (288 deg - A > 180 deg) but runs less than a turn: libration, <= medium."""
    analysis = analyse_long_period(_series(kind, 0.8), SETTINGS)
    geometry, verdict = analysis.geometry, analysis.verdict
    assert geometry.exit.suspected and not geometry.exit.observed
    assert geometry.exit.raw_arc_rad / (2 * np.pi) == pytest.approx(0.8, abs=0.05)
    assert verdict.angle_status == AngleStatus.LIBRATION
    assert str(Flag.EXIT_SUSPECTED) in verdict.flags
    assert str(verdict.final_regime) == ''


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_a_turn_and_a_half_before_the_first_turning_point_is_an_observed_entry(kind):
    """The 1.5-turn record reversed in time: the libration starts 0.75 P after the start, after 1.5 turns."""
    period = kind[2]
    analysis = analyse_long_period(_series(kind, 1.5, reverse=True), SETTINGS)
    geometry, verdict = analysis.geometry, analysis.verdict
    first = min(secular_decide.librating_episodes(geometry), key=lambda e: e.t_start_yr)
    assert first.t_start_yr == pytest.approx(0.75 * period, abs=0.02 * period)
    assert geometry.entry.observed and first.entry_observed
    assert geometry.entry.raw_arc_rad / (2 * np.pi) == pytest.approx(1.5, abs=0.05)
    assert verdict.angle_status == AngleStatus.EPISODIC
    assert str(verdict.final_regime) == ''  # the record ends librating
    assert any(entry.name == secular_decide.EDGE_ARC_ENTRIES['entry'] for entry in verdict.decision_path)
