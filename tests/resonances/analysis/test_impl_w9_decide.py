"""W9 decide package (AUTHOR-FEEDBACK 2026-10-16b, DECISIONS 2026-10-16; impl/w7 re-checks).

- The symmetric last-regime rule (G1l): a capture after a jump of a whose libration does not cover the record is a
  transient (angle_status episodic over the record, final_regime libration). A libration on both sides of the jump
  stays a libration.
- L4 (W7 'phi0', variant D): the single force law is judged by R^2 alone; the spread of phi0 is a column only. A
  modulated pendulum a_dot = K [sin sigma + eps sin(2 sigma + beta(t))] with beta turning once per 8 kyr moves the
  free-offset phase of the fit window by window (+-48 deg for A = 60 deg, eps = 0.3) at R^2 >= 0.96.
- Rule C for a multiplet (W7 'ug22'): a catalogue row whose libration an accepted member of the same commensurability
  takes gets the member's status (reason subresonance_librates, owner = the member, its own angle_status) and the
  member's confidence; both librating: each keeps its own verdict; an owner of another commensurability is unchanged.
- 37087 (W7 '37087'): the cycle of an episode is kinematic when its period matches 2 pi / |g - g5| (or the 2 omega
  cycle) within MMR_KIN_PERIOD_TOL, the identity fails and the force law stays below MMR_FORCE_R2_FAIL in both halves
  and under the multiplet law. By hand: g = 231.3"/yr, g5 = 4.257"/yr give 1296000 / 227.043 = 5708 yr; the cycle of
  5314 yr is 6.9 % off. B5 by geometry: a centre that completes a revolution (at every smoothing) while the deciding
  episode lasts.

The synthetic records: a circulation of one turn per 1000 yr (or a libration of 40 deg, 500 yr) up to a jump of a by
1e-3 au at 10 kyr, then a libration of 100 deg and 500 yr (TPs every 250 yr) around another centre to 20 kyr. The
cycle energies are 1.5 (rotation) or 0.2 before and 0.8 after, so the jump is confirmed by the energy step.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr import response as response_module
from resonances.resonance.analysis.mmr import kinematics as kinematics_module
from resonances.resonance.analysis.mmr import subresonances as sub
from resonances.resonance.analysis.mmr.models import ForceWindow, MultipletFit, ResonanceSpec, SubresonanceCandidate
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)
from tests.resonances.analysis.builders import make, make_response
from tests.resonances.analysis.test_impl_preprocess_geometry_geometry import ONE_DEG, filtered
from tests.resonances.analysis.test_impl_w2_zone_rules_decide import SAMPLING

S = settings_from_config()
PERIOD = 500.0
JUMP_YR = 10000.0


def _ledger():
    return Ledger.from_settings(S)


def capture_record(before: str):
    """`before` up to the jump at 10 kyr: 'circulation' (one turn per 1000 yr), 'libration' (40 deg around pi) or
    'libration_then_circulation' (the libration to 4 kyr, then the circulation); after it a 100-deg libration around
    pi + 2 to the end. The geometry finished with the cycle energies (`finish_geometry`)."""
    t = np.arange(0, 20000, 1.0)
    libration = np.pi + np.radians(40) * np.sin(2 * np.pi * t / PERIOD)
    circulation = np.pi + 2 * np.pi * (t - JUMP_YR) / 1000
    early = {
        'circulation': circulation,
        'libration': libration,
        # the same centre mod 2 pi, six turns down, so the circulation reaches pi at the jump without a step
        'libration_then_circulation': np.where(t < 4000, libration - 12 * np.pi, circulation),
    }[before]
    late = np.pi + 2 + np.radians(100) * np.sin(2 * np.pi * t / PERIOD)
    angle = np.where(t < JUMP_YR, early, late)
    slow = 1.0 + 1e-4 * np.cos(2 * np.pi * t / PERIOD) + 1e-3 * (t >= JUMP_YR)
    filtered_angle = filtered(t, angle, 50.0, slow=slow)
    geometry = g.measure_geometry(filtered_angle, ONE_DEG, S)
    cycles = np.arange(250, 20000, PERIOD)
    rotating = (cycles < JUMP_YR) & ((before == 'circulation') | ((before == 'libration_then_circulation') & (cycles > 4000)))
    energies = np.where(rotating, 1.5, np.where(cycles < JUMP_YR, 0.2, 0.8))
    return g.finish_geometry(geometry, filtered_angle, PERIOD, np.nan, np.nan, 0.02, cycles, energies, 5e-4, PERIOD, [], S)


def decide_all(geometry):
    ledger = _ledger()
    angle = decide.decide_angle(SAMPLING, False, geometry, None, S, ledger)
    decision = decide.decide_status(angle, SAMPLING, geometry, None, None, None, None, None, None, S, ledger)
    return angle, decision, ledger


# --- (1) the symmetric last-regime rule ------------------------------------------------------------------------------


def test_a_capture_after_a_circulation_and_a_jump_is_a_transient_that_ends_in_libration():
    geometry = capture_record('circulation')
    assert str(Flag.LAST_REGIME_AFTER_JUMP) in geometry.flags and JUMP_YR < geometry.regime_start_yr < JUMP_YR + 1000
    assert not decide.librating_episodes(geometry, S, before_jump=True)  # the circulation holds no episode
    angle, decision, ledger = decide_all(geometry)
    # the last regime itself librates (G2 libration of its window), the record does not
    assert (angle.angle_status, angle.rule) == (AngleStatus.LIBRATION, 'G1l')
    assert decide.status_basis(angle).angle_status == AngleStatus.EPISODIC
    # no response, energy or strength: L4 and R0 not measured, nothing else either (rule 5u/6, W18)
    assert (decision.status, decision.origin) == (Status.TRANSIENT, LibrationOrigin.UNKNOWN)
    reported = decide.reported_angle(angle, geometry, S)
    assert (reported.angle_status, reported.subtype) == (AngleStatus.EPISODIC, AngleSubtype.NONE)
    regime, start = decide.final_regime(angle, geometry, S)
    # libration from the start of the episode after the jump: its first TP lies within one period of the regime start
    assert regime == 'libration' and geometry.regime_start_yr <= start < geometry.regime_start_yr + PERIOD
    names = [entry.name for entry in ledger]
    assert 'G1l librating episodes before the last jump of a' in names


def test_a_libration_on_both_sides_of_the_jump_stays_a_libration():
    geometry = capture_record('libration')
    covers_start, end_gap = decide.before_jump_coverage(geometry, S)
    # the earlier episode starts within a period of the record's start and ends at the jump zone (gap <= 0 P)
    assert covers_start == 1.0 and end_gap <= S.libration_coverage_periods
    angle, decision, _ = decide_all(geometry)
    assert (angle.angle_status, angle.rule, decision.status) == (AngleStatus.LIBRATION, 'G2', Status.RESONANT)
    assert decide.reported_angle(angle, geometry, S) is angle


def test_an_early_libration_that_stops_long_before_the_jump_does_not_cover_the_record():
    geometry = capture_record('libration_then_circulation')
    (earlier,) = decide.librating_episodes(geometry, S, before_jump=True)
    assert earlier.t_end_yr < 4500  # the libration stops at 4 kyr, the jump comes 6 kyr = 12 P later
    covers_start, end_gap = decide.before_jump_coverage(geometry, S)
    assert covers_start == 1.0 and end_gap > 10
    angle, decision, _ = decide_all(geometry)
    assert (angle.rule, decision.status) == ('G1l', Status.TRANSIENT)
    # two librating episodes over the record: before the jump and the capture
    assert decide.reported_angle(angle, geometry, S).subtype == AngleSubtype.RECURRENT


def _analysed(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), S)


@pytest.mark.benchmark
def test_277617_quasi_satellite_after_the_jump_is_a_transient():
    # DECISIONS 2026-10-16: circulation up to the jump of a at 9.2 kyr, then a quasi-satellite from 13.1 kyr to the end
    # of the 20-kyr record. The run that crossed the jump keeps only its turning point before the last regime (8627 yr;
    # W9 integration): no librating episode before the jump, so the capture is one transient, not recurrent
    analysis = _analysed('coorbital__277617__1M-1+0+0__20kyr')
    verdict = analysis.verdict
    assert (verdict.status, verdict.angle_status, verdict.reason) == (Status.TRANSIENT, AngleStatus.EPISODIC, 'dynamic_episode')
    assert verdict.angle_subtype == AngleSubtype.NONE
    assert verdict.final_regime == 'libration' and verdict.final_regime_start_yr == pytest.approx(13086, abs=5)
    assert analysis.extra_columns['geo_n_episodes_before_jump'] == 0
    assert all(run.t_end_yr < analysis.geometry.regime_start_yr for run in analysis.geometry.runs if run.before_last_jump)


# --- (2) L4 without the phase of the single law ----------------------------------------------------------------------


def test_a_modulated_pendulum_whose_phase_moves_passes_the_force_law():
    t = np.arange(0, 20000, 1.0)
    sigma = np.radians(60) * np.sin(2 * np.pi * t / 300.0)
    beta = 2 * np.pi * t / 8000.0
    adot = 1e-5 * (np.sin(sigma) + 0.3 * np.sin(2 * sigma + beta))
    windows = response_module.force_windows(t, sigma, adot, 1500.0, 0.5)
    _, _, spread = response_module.phase_spread(windows)
    # the phase swings beyond the old floor of the tolerance while the law fits well
    assert np.degrees(spread) > S.mmr_force_phase_tol_deg and min(w.r2 for w in windows) > 0.95
    assert response_module.a_response_check(windows, None, np.radians(60), S) == (CheckValue.PASS, False, 'single')


def test_a_force_law_with_unbounded_modulation_passes_on_r2():
    # sum eps = 1.1 >= 1: the old rule had no bound on the phase and left L4 not_measured; R^2 0.8 now passes
    windows = [ForceWindow(1000.0 * i, 1000.0 * (i + 1), 0.8, 1e-6, phase, 0.0) for i, phase in enumerate([0.0, 2.0, -2.0, 1.0])]
    multiplet = MultipletFit((0, 1, -1), np.ones((4, 3), dtype=complex), 0.85, 0.9, (0.6, 0.5))  # gains 0.05: not taken
    assert np.isnan(response_module.phase_tolerance(multiplet.eps, 0.0, np.radians(S.mmr_force_phase_tol_deg)))
    assert response_module.a_response_check(windows, multiplet, np.radians(90), S) == (CheckValue.PASS, False, 'single')


@pytest.mark.benchmark
def test_7343_three_body_libration_is_resonant_without_the_phase_rule():
    # W7 'phi0': the eight L4 failures were all phase spreads; 7343 1M+1J-2 goes uncertain (5a*) -> resonant
    verdict = _analysed('mmr3__7343__1M+1J-2+0+0+0').verdict
    assert (verdict.status, verdict.angle_status, verdict.checks['check_mmr_a_response']) == (
        Status.RESONANT,
        AngleStatus.LIBRATION,
        CheckValue.PASS,
    )
    assert '5 L4 phi0 spread within tolerance' not in [entry.name for entry in verdict.decision_path]


# --- (3) rule C: the catalogue row takes its member's status --------------------------------------------------------


def _accepted_scan():
    member = SubresonanceCandidate(
        ResonanceSpec.from_key('9M-16+2+5'), 1, False, 0.1, 0.0, 0.0, 0.0, np.nan, 1.0, np.nan, np.nan, np.nan, 1, False, ''
    )
    return replace(sub._empty((), CheckValue.FAIL, []), candidates=[member], accepted=member, theory_measured=True)


def _member_verdict(status, angle_status):
    return Verdict(ResonanceType.MMR, status, angle_status, confidence=Confidence.HIGH, confidence_raw=Confidence.HIGH)


TRANSIENT_ANGLE = decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
LIBRATING_ANGLE = decide.AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', 0)


@pytest.mark.parametrize(
    'member_status, expected',
    [
        (Status.TRANSIENT, Status.TRANSIENT),
        (Status.RESONANT, Status.RESONANT),
        (Status.NEAR_SEPARATRIX, Status.NEAR_SEPARATRIX),
        (Status.UNCERTAIN, Status.UNCERTAIN),
        (Status.NON_RESONANT, Status.NON_RESONANT),
        (Status.INVALID, Status.UNCERTAIN),  # an unmeasurable member leaves the commensurability open
    ],
)
def test_the_catalogue_row_takes_the_status_of_its_accepted_member(member_status, expected):
    member = _member_verdict(member_status, AngleStatus.EPISODIC)
    decision = decide._foreign(TRANSIENT_ANGLE, _accepted_scan(), None, S, _ledger(), member)
    assert (decision.status, decision.reason, decision.origin, decision.owner) == (
        expected,
        Reason.SUBRESONANCE_LIBRATES,
        LibrationOrigin.SUBRESONANCE,
        '9M-16+2+5',
    )


def test_a_circulating_catalogue_angle_takes_the_member_status_and_keeps_its_angle():
    circulation = decide.AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4')
    member = _member_verdict(Status.TRANSIENT, AngleStatus.EPISODIC)
    decision = decide._circulation(circulation, SAMPLING, None, None, _accepted_scan(), None, S, _ledger(), member)
    assert (decision.status, decision.reason, decision.owner) == (Status.TRANSIENT, Reason.SUBRESONANCE_LIBRATES, '9M-16+2+5')


def test_both_librating_keep_their_own_verdicts():
    member = _member_verdict(Status.RESONANT, AngleStatus.LIBRATION)
    ledger = _ledger()
    assert decide.members_locked(LIBRATING_ANGLE, member) == 1.0
    assert decide._foreign(LIBRATING_ANGLE, _accepted_scan(), None, S, ledger, member) is None  # the tree goes on
    assert any(entry.column == 'sub_members_locked' for entry in ledger)
    # a librating catalogue angle next to a transient member: the member takes it
    transient_member = _member_verdict(Status.TRANSIENT, AngleStatus.EPISODIC)
    assert decide.members_locked(LIBRATING_ANGLE, transient_member) == 0.0
    assert decide._foreign(LIBRATING_ANGLE, _accepted_scan(), None, S, _ledger(), transient_member).status == Status.TRANSIENT
    assert np.isnan(decide.members_locked(TRANSIENT_ANGLE, member))


def test_a_member_without_its_own_row_keeps_the_old_leaf():
    # a nodal member has no derived row: non_resonant/subresonance_librates as before
    decision = decide._foreign(TRANSIENT_ANGLE, _accepted_scan(), None, S, _ledger(), None)
    assert (decision.status, decision.reason) == (Status.NON_RESONANT, Reason.SUBRESONANCE_LIBRATES)


@pytest.mark.benchmark
def test_31935_catalogue_row_is_the_transient_of_its_member():
    # W7 'ug22' spec: 9M-16+0+7 circulates while 9M-16+2+5 librates for a while; the commensurability is transient
    analysis = _analysed('mmr2__31935__9M-16+0+7')
    verdict, (member,) = analysis.verdict, analysis.derived
    assert (verdict.status, verdict.reason, verdict.owner) == (Status.TRANSIENT, Reason.SUBRESONANCE_LIBRATES, '9M-16+2+5')
    # the catalogue angle's own behaviour: it librates about the moving minimum of the averaged potential at 31-82 kyr
    # (W18 moving centre; the author's angle label since 2026-10-16: episodic, DECISIONS.md)
    assert verdict.angle_status == AngleStatus.EPISODIC
    assert member.verdict.status == Status.TRANSIENT and verdict.confidence == member.verdict.confidence
    assert analysis.multiplet_status == Status.TRANSIENT


# --- (4) 37087: the cycle of an episode at a secular period -------------------------------------------------------

ARCSEC_RAD = np.pi / 180 / 3600


def _kinematics(modulation_yr, two_omega_yr=np.nan):
    return SimpleNamespace(modulation_period_yr=modulation_yr, two_omega_period_yr=two_omega_yr, check=CheckValue.NOT_MEASURED)


def _response(identity=CheckValue.FAIL, r2=(3e-4, 5e-4, 2e-3), slope=-0.62):
    multiplet = make(MultipletFit, r2=r2[2]) if r2[2] is not None else None
    return make_response(
        identity=identity, identity_slope_ratio=slope, identity_corr=-0.3, force_r2_h1=r2[0], force_r2_h2=r2[1], multiplet=multiplet
    )


def test_the_secular_period_of_37087_by_hand():
    modulation = kinematics_module._modulation_period(231.3 * ARCSEC_RAD, 4.257 * ARCSEC_RAD)
    assert modulation == pytest.approx(1296000 / 227.043, rel=1e-6)  # 5708.2 yr
    assert decide.secular_period_offset(5314.0, _kinematics(modulation)) == pytest.approx(1 - 5314 / 5708.2, abs=1e-4)
    # the closer of the two secular periods counts
    assert decide.secular_period_offset(5314.0, _kinematics(40000.0, 5300.0)) == pytest.approx(14 / 5300, rel=1e-6)


def test_an_episode_at_the_secular_period_that_a_does_not_carry_is_kinematic():
    kinematics = _kinematics(5708.2)
    assert decide.kinematic_check(TRANSIENT_ANGLE, kinematics, _response(), 5314.0, S) == CheckValue.FAIL
    # each condition alone keeps the episode out of it (not_applicable, as for every episode before)
    open_cases = [
        (_response(), 4500.0),  # 21 % off the secular period
        (_response(identity=CheckValue.PASS, slope=1.02), 5314.0),  # a Hilda: the identity holds
        (_response(identity=CheckValue.NOT_MEASURED), 5314.0),
        (_response(r2=(3e-4, 0.5, 2e-3)), 5314.0),  # a responds in the second half
        (_response(r2=(3e-4, np.nan, 2e-3)), 5314.0),  # not measured
        (_response(r2=(3e-4, 5e-4, None)), 5314.0),  # no multiplet law
    ]
    for response, period in open_cases:
        assert decide.kinematic_check(TRANSIENT_ANGLE, kinematics, response, period, S) == CheckValue.NOT_APPLICABLE
    # a non-episodic angle keeps the slow-angle test of the kinematics module
    drift = decide.AngleDecision(AngleStatus.DRIFT, AngleSubtype.NONE, 'G5')
    assert decide.kinematic_check(drift, kinematics, _response(), 5314.0, S) == CheckValue.NOT_MEASURED


def test_slow_libration_is_excluded_by_a_centre_turning_round_during_the_episode():
    # 37087-like: the centre passes > 1 revolution at every smoothing inside the episode
    assert response_module.centre_turns_in_episode((1.4, 20000.0, 70000.0), (10000.0, 90000.0))
    # 48789: the pass at 11-38 kyr, the deciding episode at 54-98 kyr: no overlap, a capture there is not excluded
    assert not response_module.centre_turns_in_episode((1.4, 11000.0, 38000.0), (54000.0, 98000.0))
    # below one revolution at one of the smoothings, or without an episode
    assert not response_module.centre_turns_in_episode((0.8, 20000.0, 70000.0), (10000.0, 90000.0))
    assert not response_module.centre_turns_in_episode((1.4, 20000.0, 70000.0), (np.nan, np.nan))
    bound = {'centre_rate': 1e-4, 'centre_rate_err': 1e-5, 'ratio': 0.3}
    check = response_module.slow_libration_check(bound, np.nan, np.nan, np.nan, S, (1.4, 20000.0, 70000.0), (10000.0, 90000.0))
    assert check == (CheckValue.PASS, 'geometry', False)
    assert response_module.slow_libration_check(bound, np.nan, np.nan, np.nan, S)[:2] == (CheckValue.FAIL, 'none')


@pytest.mark.benchmark
@pytest.mark.xfail(
    strict=True,
    reason=(
        'W20 CS7: the 168-yr forced wave (line 2J-1, identity -0.35 c) is no episode now: non_resonant/circulation/'
        'medium (test_w20_cs7). The 5.7-kyr kinematic loops are no episodes at the 75-yr cutoff: the 168-yr '
        'turning points fill the record; above the line (cutoff 505 yr) they are two episodes of P 5.73 kyr whose '
        'force law passes thin (R2 0.58/0.61). Reason kinematic_libration and the word episodic: open (author)'
    ),
)
def test_37087_is_a_kinematic_libration_at_the_secular_cycle():
    verdict = _analysed('mmr3__37087__3J+8S-3+0+0-8').verdict
    assert (verdict.status, verdict.reason, verdict.libration_origin, verdict.origin_detail) == (
        Status.NON_RESONANT,
        Reason.KINEMATIC_LIBRATION,
        LibrationOrigin.KINEMATIC,
        'secular_cycle',
    )
    assert verdict.angle_status == AngleStatus.EPISODIC  # the geometry of the angle stays
    assert verdict.checks['check_mmr_kinematic'] == CheckValue.FAIL and verdict.checks['check_mmr_slow_libration'] == CheckValue.PASS


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, status, reason',
    [
        # W16: at the physical P_c (75 yr) a 7-TP episode of ~14 kyr is found; identity and F pass, L4 and R0 are not
        # measured: transient (the author's bucket), not kinematic (TEST-DISPUTES W16); only the geometry decided
        # (rule 5u/6): geometric_episode since 2026-10-26 (author, Q3)
        ('mmr3__48789__3J-1S-1+0+0-1', Status.TRANSIENT, Reason.GEOMETRIC_EPISODE),
        # 11 % off the secular period; transient since W20 (its exit observed: the edge rule)
        ('mmr3__24820__3J-1S-1+0+0-1', Status.TRANSIENT, Reason.GEOMETRIC_EPISODE),
        ('mmr2__153__3J-2+0-1', Status.RESONANT, Reason.DYNAMIC_LIBRATION),  # a Hilda: the identity holds
    ],
)
def test_the_kinematic_episode_rule_leaves_its_neighbours(example, status, reason):
    verdict = _analysed(example).verdict
    assert (verdict.status, verdict.reason) == (status, reason)
