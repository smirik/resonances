"""W28 D6 (card g09-4): a secular or ZLK drift needs a monotone centre; a centre that turned back is insufficient_time.

A drift is a centre that keeps going one way. A slow centre that turns back inside the record (away from the edges of
its own low-pass, `mmr.decide.interior_turns`) with its largest pass below one turn holds part of a slow swing: too
little to tell a drift from a slow libration (2306 g-g6+s-s6 over 10 Myr: a turn at 2.04 Myr, largest pass 0.92 rev).
The status (uncertain, baseline_needed) and the extension stay those of a drift.

The synthetic angles: a 40-deg wiggle whose runs are rejected for a moving centre on a centre c(t) that
- rises 0.45 turn and falls back (c = 0.9 pi sin(pi t / T): one turn at T/2, largest pass 0.45 rev),
- rises 0.9 turn monotonically (no turn, one pass of 0.9 rev),
- rises 1.3 turns monotonically (a pass >= 1 turn: circulation).
"""

import numpy as np
import pytest

from resonances.resonance.analysis.mmr.decide import interior_turns
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.secular import LongPeriodAnalysis, LongPeriodSeries
from resonances.resonance.analysis.secular.analyse import analyse_long_period
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, Reason, ResonanceType, Status

SETTINGS = settings_from_config()
FIRED = str(Rule.CENTRE_TURNED_INSUFFICIENT)


def _secular(centre_turns) -> LongPeriodAnalysis:
    """A secular angle over 20 Myr: a 400-kyr wiggle of 40 deg on the centre `centre_turns(t / T)` [turns]."""
    t = np.arange(0.0, 20_000_001.0, 2000.0)
    one = np.ones_like(t)
    omega = 1.0 + 2 * np.pi * centre_turns(t / t[-1]) + np.radians(40.0) * np.sin(2 * np.pi * t / 400_000.0)
    series = LongPeriodSeries(ResonanceType.SECULAR, 'g-g6', 'synthetic', t, omega, 2.4 * one, 0.1 * one, 0.1 * one)
    return analyse_long_period(series, SETTINGS)


def _zlk(centre_turns) -> LongPeriodAnalysis:
    """A Lidov-Kozai omega over 1 Myr: a 10-kyr wiggle of 40 deg on the centre `centre_turns(t / T)` [turns]."""
    t = np.arange(0.0, 1_000_001.0, 100.0)
    one = np.ones_like(t)
    omega = 1.0 + 2 * np.pi * centre_turns(t / t[-1]) + np.radians(40.0) * np.sin(2 * np.pi * t / 10_000.0)
    series = LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, 5.2 * one, 0.05 * one, 0.3 * one)
    return analyse_long_period(series, SETTINGS)


def back(x):
    return 0.45 * np.sin(np.pi * x)


def monotone(x):
    return 0.9 * x


def winding(x):
    return 1.3 * x


@pytest.mark.parametrize('analyse', [_secular, _zlk])
def test_a_centre_that_turned_back_is_insufficient_time(analyse):
    analysis = analyse(back)
    geometry, verdict = analysis.geometry, analysis.verdict
    # the runs are rejected for their moving centre over the window: what made it a drift before D6
    assert geometry.drift_fraction >= SETTINGS.libration_drift_min_window_fraction
    assert interior_turns(geometry) == 1
    assert geometry.centre.max_pass_rev == pytest.approx(0.45, abs=0.03)
    assert verdict.angle_status == AngleStatus.INSUFFICIENT_TIME
    assert verdict.angle_subtype == AngleSubtype.FEW_CYCLES
    assert verdict.status == Status.UNCERTAIN and verdict.reason == Reason.BASELINE_NEEDED
    assert FIRED in verdict.rules_fired
    entry = next(e for e in verdict.decision_path if e.name == 'centre_monotone')
    assert entry.result is False


@pytest.mark.parametrize('analyse', [_secular, _zlk])
def test_a_monotone_centre_below_a_turn_is_a_drift(analyse):
    analysis = analyse(monotone)
    assert interior_turns(analysis.geometry) == 0
    assert analysis.geometry.centre.max_pass_rev == pytest.approx(0.9, abs=0.03)
    assert analysis.verdict.angle_status == AngleStatus.DRIFT
    assert FIRED not in analysis.verdict.rules_fired


@pytest.mark.parametrize('analyse', [_secular, _zlk])
def test_a_centre_pass_of_a_turn_or_more_is_circulation(analyse):
    analysis = analyse(winding)
    assert analysis.geometry.centre.max_pass_rev >= 1.0
    assert analysis.verdict.angle_status == AngleStatus.CIRCULATION
    assert FIRED not in analysis.verdict.rules_fired


def test_status_and_extension_are_those_of_a_drift():
    """Only the angle word changes (the author, g09-4): uncertain, extend, the same baseline."""
    turned, drift = _secular(back).verdict, _secular(monotone).verdict
    assert turned.status == drift.status == Status.UNCERTAIN
    assert turned.reason == drift.reason == Reason.BASELINE_NEEDED
    assert turned.action == drift.action == 'extend'
    assert turned.extend_to_years == drift.extend_to_years
