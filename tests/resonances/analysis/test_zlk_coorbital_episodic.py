"""W28 D4 (card g09-6, the author's answer B): the ZLK row of a co-orbital whose planet's modes the free-element split
does not fit has no gate verdict whatever bounded angle it has, and the period the physics reads is the deciding
librating episode's.

Expectations are independent of the package: the status code -9 against -2 (SPEC 5.7, the CLAUDE.md rule that the two
are never merged), periods counted by hand from the turning points the test places, the free-forced beat
1296000 / (222 - g5) arcsec/yr from the synthetic's own rates, and the synthetic librator's 40-kyr cycle.
"""

import numpy as np
import pytest

from resonances.resonance import omega_free_gate
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.secular import decide
from resonances.resonance.analysis.secular.models import GateResult
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    CheckValue,
    Flag,
    LibrationOrigin,
    Margin,
    Reason,
    ResonanceType,
    Status,
    status_code,
)
from resonances.resonance.analysis.zlk import analyse_zlk, diagnostics
from tests.resonances.analysis.builders import make_episode, make_geometry
from tests.resonances.analysis.test_zlk_double_scale import AS, G5, G6, RAD, zlk_series

SETTINGS = settings_from_config()
LK = ResonanceType.LIDOV_KOZAI
COORBITAL = omega_free_gate.COORBITAL_UNFITTED


def _status(angle_status, gate, episode_id=None):
    angle = decide.AngleDecision(angle_status, decisive_episode_id=episode_id)
    return decide.decide_status(LK, angle, gate, Ledger.from_settings(SETTINGS))


def _code(decision, angle_status):
    return status_code(LK, decision.status, angle_status, decision.reason)


# ------------------------------------------------------------------------------- part 1: the status (choice B)


def test_the_decision_and_the_gate_name_the_same_detail():
    assert decide.DETAIL_COORBITAL_UNFITTED == omega_free_gate.COORBITAL_UNFITTED
    assert decide.GATE_UNAVAILABLE == str(omega_free_gate.GateOutcome.UNAVAILABLE)


@pytest.mark.parametrize('angle_status', [AngleStatus.EPISODIC, AngleStatus.DRIFT, AngleStatus.INSUFFICIENT_TIME])
def test_an_unfitted_coorbital_without_a_libration_has_no_gate_verdict(angle_status):
    # 2020 PP1 (episodic), 10563 over 100 kyr (too few cycles): the free-element model does not apply, the same as for
    # a libration; the code is -9 (the absence of a measurement), never -4/-6 (a length would not help) or 1
    gate = GateResult('unavailable', min_cycles=3.0, rho_max=30.0, detail=COORBITAL)
    decision = _status(angle_status, gate)
    assert (decision.status, decision.reason) == (Status.UNCERTAIN, Reason.GATE_UNAVAILABLE)
    assert decision.origin == LibrationOrigin.UNKNOWN
    assert decision.checks['check_zlk_free_omega'] == CheckValue.NOT_MEASURED
    assert Rule.ZLK_COORBITAL_EPISODIC in decision.rules_fired
    assert _code(decision, angle_status) == -9


def test_an_unfitted_coorbital_libration_is_unchanged():
    gate = GateResult('unavailable', n_cycles=10.0, min_cycles=3.0, detail=COORBITAL)
    decision = _status(AngleStatus.LIBRATION, gate)
    assert (decision.status, decision.reason) == (Status.UNCERTAIN, Reason.GATE_UNAVAILABLE)
    assert Rule.ZLK_COORBITAL_EPISODIC not in decision.rules_fired  # the rule is the non-libration branch


def test_a_kinematic_episodic_row_stays_non_resonant_and_apart_from_minus_nine():
    # free circle never around the origin: the measurement says geometry (-2); the co-orbital says nothing (-9)
    kinematic = _status(AngleStatus.EPISODIC, GateResult('kinematic', n_cycles=10.0, min_cycles=3.0, rho_max=0.5))
    assert (kinematic.status, kinematic.reason) == (Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION)
    coorbital = _status(AngleStatus.EPISODIC, GateResult('unavailable', min_cycles=3.0, detail=COORBITAL))
    assert (_code(kinematic, AngleStatus.EPISODIC), _code(coorbital, AngleStatus.EPISODIC)) == (-2, -9)


@pytest.mark.parametrize(
    'gate',
    [
        GateResult('planetary_residual', n_cycles=15.0, min_cycles=3.0, rho_max=13.0, episodic_policy='keep'),  # 88899
        GateResult('unavailable', min_cycles=3.0, episodic_policy='keep'),  # no free elements: not the co-orbital detail
        GateResult('short_window', n_cycles=2.0, min_cycles=3.0, episodic_policy='keep'),
    ],
)
def test_other_unverdicted_outcomes_are_not_the_coorbital_rule(gate):
    # with ZLK_EPISODIC_NO_VERDICT=keep the geometric status stands; the default (ei) is D10's
    # (test_zlk_episodic_no_verdict)
    decision = _status(AngleStatus.EPISODIC, gate)
    assert (decision.status, decision.reason) == (Status.TRANSIENT, Reason.DYNAMIC_EPISODE)
    assert decision.checks['check_zlk_free_omega'] == CheckValue.NOT_MEASURED
    assert Rule.ZLK_COORBITAL_EPISODIC not in decision.rules_fired


def _episodic_librator(a):
    """omega_free = 90 + 25 sin(2 pi t / 40 kyr) until 240 kyr, then circulating one turn per 30 kyr; e 0.30, i 40 deg,
    the node at -20 arcsec/yr, small forced terms at g5 and g6 (the librator of `test_zlk_double_scale`, cut short)."""
    t = np.arange(0.0, 500_001.0, 100.0)
    switch = 240_000.0
    librating = np.radians(90.0) + np.radians(25.0) * np.sin(2 * np.pi * t / 40_000.0)
    circulating = np.radians(90.0) + 2 * np.pi * (t - switch) / 30_000.0
    omega_free = np.where(t < switch, librating, circulating)
    Omega = -20.0 * RAD * t + 0.7
    z_e = (0.30 + 0.04 * np.cos(2 * np.pi * t / 40_000.0)) * np.exp(1j * (omega_free + Omega))
    z_e = z_e + 0.03 * np.exp(1j * (G5 * RAD * t + 1.0)) + 0.007 * np.exp(1j * (G6 * RAD * t + 2.0))
    return zlk_series(t, z_e, Omega, np.full_like(t, np.radians(40.0)), a=a)


def test_an_episodic_earth_coorbital_is_uncertain_end_to_end():
    series, elements = _episodic_librator(a=1.0)
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    verdict = analysis.verdict
    assert verdict.angle_status == AngleStatus.EPISODIC
    assert (verdict.status, verdict.reason) == (Status.UNCERTAIN, Reason.GATE_UNAVAILABLE)
    assert analysis.gate.detail == COORBITAL and 'zlk_coorbital_episodic' in verdict.rules_fired
    assert verdict.status_code == -9
    # the episode's own cycle, 40 kyr within a grid step per turning point, is what the gate and the e-i coupling read
    assert analysis.gate.libration_period_yr == pytest.approx(40_000.0, abs=400.0)
    assert analysis.zlk.ei.period_yr == analysis.gate.libration_period_yr
    assert analysis.numbers_row()['sec_decisive_period_yr'] == analysis.gate.libration_period_yr
    # the same orbit away from any planet: the co-orbital rule does not act; without a gate verdict D10 decides (i is
    # frozen here, so e and i do not trade inside the episode: uncertain, rule zlk_gate_unconfirmed)
    series, elements = _episodic_librator(a=2.5)
    far = analyse_zlk(series, SETTINGS, free_elements=elements).verdict
    assert far.angle_status == AngleStatus.EPISODIC and 'zlk_coorbital_episodic' not in far.rules_fired
    assert far.status == Status.UNCERTAIN and 'zlk_gate_unconfirmed' in far.rules_fired


# ------------------------------------------------------------------- part 2: the deciding episode's period


def _fast_and_slow(accepted=True):
    """A run faster than the ZLK floor (2 kyr): 9 TPs 500 yr apart, full periods 1000 yr; a slow run: 5 TPs at 100,
    119, 138, 161, 180 kyr, full periods 38, 42, 42 kyr (median 42 kyr). Pooled the way P_cyc pools them, the seven
    1-kyr periods outnumber the three slow ones: P_cyc = 1000 yr."""
    fast = make_episode(episode_id=0, tp_times_yr=np.arange(9) * 500.0, period_med_yr=1000.0, accepted=accepted)
    slow_tp = np.array([100_000.0, 119_000.0, 138_000.0, 161_000.0, 180_000.0])
    slow = make_episode(
        episode_id=1,
        tp_times_yr=slow_tp,
        period_med_yr=42_000.0,
        accepted=accepted,
        coverage_start_gap_p=2.4,
        coverage_end_gap_p=7.6,
    )
    return make_geometry(
        window=(0.0, 500_000.0), runs=[fast, slow], period_cycle_yr=1000.0, period_floor_yr=SETTINGS.zlk_libration_period_min_yr
    )


def test_a_floor_rejected_fast_run_does_not_set_the_period():
    geometry = _fast_and_slow()
    angle = decide.decide_angle(geometry, SETTINGS, Ledger.from_settings(SETTINGS))
    assert (angle.angle_status, angle.decisive_episode_id) == (AngleStatus.EPISODIC, 1)
    assert decide.decisive_librating_episode(geometry, angle.decisive_episode_id).episode_id == 1
    assert decide.decisive_librating_episode(geometry, 0) is None  # the fast run is no librating episode
    assert decide.decisive_period_yr(geometry, angle.decisive_episode_id) == 42_000.0
    # F at P_c = 5 kyr: 42/5 = 8.4 >= FILTER_MIN_LIBRATION_CUTOFFS (2), full; the pooled 1 kyr would give 0.2, not
    # measured with filter_too_slow
    margins, _, flags = decide.margins_and_caps(angle, geometry, 5_000.0, None, SETTINGS, Ledger.from_settings(SETTINGS))
    assert margins['F'].margin == Margin.FULL and margins['F'].value == pytest.approx(8.4)
    assert str(Flag.FILTER_TOO_SLOW) not in flags


def test_without_a_librating_episode_the_period_pools_only_runs_above_the_floor():
    geometry = _fast_and_slow(accepted=False)  # a drift: every run rejected
    assert decide.decisive_period_yr(geometry, None) == 42_000.0  # median of 38, 42, 42 kyr
    fast_only = make_geometry(runs=geometry.runs[:1], period_cycle_yr=1000.0, period_floor_yr=2000.0)
    assert np.isnan(decide.decisive_period_yr(fast_only, None))
    assert np.isnan(decide.decisive_period_yr(None, None))


def test_the_attribution_of_episodes_reads_their_pooled_period():
    # forced 0.03 at g5, free 0.035 at 222 arcsec/yr: the beat 1296000 / (222 - g5); the strict pin fails
    # (0.035 > 0.03), so only the episodes' predicate attributes, with the scope episodes
    t = np.arange(0.0, 500_001.0, 100.0)
    z_e = 0.03 * np.exp(1j * (G5 * RAD * t + 1.0)) + 0.035 * np.exp(1j * 222.0 * RAD * t)
    _, elements = zlk_series(t, z_e, G5 * RAD * t + 0.3, np.full_like(t, np.radians(39.0)))
    beat = AS / (222.0 - G5)
    args = (elements, 'kinematic', 1.1 * beat, 2.0, 0.5, 1.0, 0.5, 100.0)  # the deciding episode 10 % off the beat
    pooled = diagnostics.attribute_forcing(*args, episodes_period_yr=beat)
    assert pooled.attributed and pooled.scope == 'episodes'
    assert pooled.beat_period_yr == pytest.approx(beat, rel=1e-3)
    assert diagnostics.attribute_forcing(*args).reason == 'beat_period_mismatch'
