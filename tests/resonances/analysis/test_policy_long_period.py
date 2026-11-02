"""W29 minimal policy on secular and Lidov-Kozai rows (`policy.apply_long_period`, spec §8), built from real result
objects. The expectations follow PRINCIPLES §7 by hand: the gate's kinematic outcome or a free e below the forced e
is a measured veto (-2); a gate without a verdict is a flag, the row keeps its geometry at <= medium; a secular swing
inside the mode-phase uncertainty is uncertain, never non_resonant; a ZLK omega's own revolution is pi."""

import numpy as np
import pytest

from resonances.resonance.analysis import policy
from resonances.resonance.analysis.secular.models import GateResult
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    Action,
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    MarginValue,
    Reason,
    ResonanceType,
    Status,
)
from tests.resonances.analysis.builders import (
    make_episode,
    make_filtered_angle,
    make_geometry,
    make_long_period_analysis,
    make_turning_points,
    make_verdict,
)

MINIMAL = AnalysisSettings(classifier_policy='minimal')
ZLK, SECULAR = ResonanceType.LIDOV_KOZAI, ResonanceType.SECULAR
SPAN = 1.0e6
PERIOD = 5.0e4


def geometry(n_tp=9, gap=0.2, accepted=True):
    """One run of `n_tp` turning points half a period apart from 1e5 yr in a window [0, 1e6] yr."""
    times = 1.0e5 + np.arange(n_tp) * PERIOD / 2
    run = make_episode(
        n_tp,
        episode_id=0,
        tp_times_yr=times,
        accepted=accepted,
        period_med_yr=PERIOD,
        half_amp_max_rad=0.6,
        coverage_start_gap_p=gap,
        coverage_end_gap_p=gap,
    )
    return make_geometry(window=(0.0, SPAN), runs=[run], turning_points=make_turning_points(times), period_cycle_yr=PERIOD)


def gate(outcome, **fields):
    fields.setdefault('n_cycles', 18.0)
    fields.setdefault('min_cycles', 3.0)
    return GateResult(outcome, **fields)


def zlk(word=AngleStatus.LIBRATION, outcome='confirmed', subtype=AngleSubtype.NONE, n_tp=9, **gate_fields):
    status = {AngleStatus.LIBRATION: Status.RESONANT, AngleStatus.EPISODIC: Status.TRANSIENT}.get(word, Status.UNCERTAIN)
    verdict = make_verdict(status, word, angle_subtype=subtype, resonance_type=ZLK, reason=Reason.DYNAMIC_LIBRATION)
    analysis = make_long_period_analysis(verdict, geometry(n_tp), gate(outcome, **gate_fields), span_yr=SPAN, decisive_episode_id=0)
    analysis.decisive_period_yr = PERIOD
    return policy.apply_long_period(analysis, MINIMAL)


def test_a_libration_without_a_gate_verdict_is_resonant_at_medium():
    v = zlk(outcome='planetary_residual', min_distance_arcsec_yr=2.0, baseline_yr=SPAN).verdict
    assert (v.status, v.reason, v.confidence, v.status_code) == (Status.RESONANT, Reason.GEOMETRIC_LIBRATION, Confidence.MEDIUM, 2)
    assert {str(Flag.GATE_UNAVAILABLE), str(Flag.VETO_NEEDS_LENGTH)} <= set(v.flags)
    assert v.checks['check_zlk_free_omega'] == CheckValue.NOT_MEASURED
    # 1.25 x 1296000 / 2 arcsec/yr = 810 kyr of free baseline, nothing lost (ZLK_EXTEND_RESOLUTION_MARGIN 1.25)
    assert v.pol_veto_need_yr == pytest.approx(MINIMAL.zlk_extend_resolution_margin * 1296000 / 2.0)
    assert v.pol_veto_need_yr == pytest.approx(810000.0)
    # <= MMR_NEXT_CHECK_SPAN_FACTOR x T = 4 Myr: clause C, next check round_up(max(2 T, 810 kyr)) = 2 Myr (W29 Q5)
    assert (v.action, v.pol_extend_clause, v.extend_to_years) == (Action.EXTEND, 'C', 2.0e6)


def test_a_gate_that_needs_more_than_the_budget_goes_to_review():
    # 1.25 x 1296000 / 0.25 arcsec/yr = 6.48 Myr > 4 T = 4 Myr: no extend (1981 Midas on 1 Myr), the row keeps its
    # status and level and goes to review
    v = zlk(outcome='planetary_residual', min_distance_arcsec_yr=0.25, baseline_yr=SPAN).verdict
    assert v.pol_veto_need_yr == pytest.approx(6.48e6)
    assert (v.status, v.confidence, v.action, v.pol_extend_clause) == (Status.RESONANT, Confidence.MEDIUM, Action.REVIEW, '')
    # d_min = 0: no baseline resolves the planets' lines (inf): review as well
    v = zlk(outcome='planetary_residual', min_distance_arcsec_yr=0.0, baseline_yr=SPAN).verdict
    assert v.pol_veto_need_yr == float('inf') and v.action == Action.REVIEW


def test_an_unfitted_co_orbital_episode_is_transient_at_medium():
    v = zlk(AngleStatus.EPISODIC, 'unavailable', detail='coorbital_unfitted_forcing').verdict
    assert (v.status, v.reason, v.confidence, v.status_code) == (Status.TRANSIENT, Reason.GEOMETRIC_EPISODE, Confidence.MEDIUM, 1)


def test_a_disabled_gate_keeps_the_geometry_at_medium():
    v = zlk(outcome='disabled').verdict
    assert (v.status, v.confidence) == (Status.RESONANT, Confidence.MEDIUM)
    assert str(Flag.GATE_DISABLED) in v.flags and v.checks['check_zlk_free_omega'] == CheckValue.NOT_MEASURED


def test_a_confirmed_gate_reads_n():
    v = zlk().verdict  # 9 TPs: 4 cycles + covered gaps 0.2 + 0.2
    assert (v.status, v.confidence, v.n_cycles) == (Status.RESONANT, Confidence.HIGH, pytest.approx(4.4))
    assert v.checks['check_zlk_free_omega'] == CheckValue.PASS


def test_a_kinematic_libration_is_minus_two():
    analysis = zlk(outcome='kinematic', rho_max=0.4)
    v = analysis.verdict
    assert (v.status, v.reason, v.veto, v.status_code) == (Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, policy.VETO_ZLK_KINEMATIC, -2)
    assert v.confidence == Confidence.HIGH and v.n_counted_episodes == 0
    assert analysis.policy.candidate_episode_ids == frozenset({0})


def test_a_kinematic_outcome_that_pins_nothing_on_few_cycles_is_an_objection():
    v = zlk(AngleStatus.INSUFFICIENT_TIME, 'kinematic', AngleSubtype.FEW_CYCLES, n_tp=2, rho_max=1.4).verdict
    assert v.status == Status.UNCERTAIN and str(Flag.MEASURED_OBJECTION) in v.flags and v.veto == ''


def test_an_episode_with_few_free_turns_is_transient_low():
    v = zlk(AngleStatus.EPISODIC, 'kinematic', rho_max=1.4, rho_episodes=0.8, free_turns_episodes=0.1, free_turns_min=0.5).verdict
    assert (v.status, v.confidence, v.veto) == (Status.TRANSIENT, Confidence.LOW, '')


def test_an_episode_the_forced_vector_pins_is_a_free_e_veto():
    # 624 Hektor: rho_episodes 0.991 against 1 (within the thin band 0.1): the veto moves, at medium
    v = zlk(AngleStatus.EPISODIC, 'kinematic', rho_max=1.4, rho_episodes=0.991, free_turns_episodes=0.9, free_turns_min=0.5).verdict
    assert (v.status, v.veto, v.confidence, v.status_code) == (Status.NON_RESONANT, policy.VETO_ZLK_FREE_E, Confidence.MEDIUM, -2)
    assert str(Flag.VETO_THIN) in v.flags


def test_an_undefined_phase_with_the_free_circle_off_the_origin_is_a_free_e_veto():
    v = zlk(outcome='undefined_phase', rho_max=0.5).verdict
    assert (v.veto, v.status_code) == (policy.VETO_ZLK_FREE_E, -2)
    w = zlk(outcome='undefined_phase', rho_max=1.5).verdict
    assert (w.status, w.status_code) == (Status.RESONANT, 2) and str(Flag.GATE_UNAVAILABLE) in w.flags


def test_minus_two_and_gate_unavailable_never_meet():
    cases = [
        zlk(outcome=outcome, rho_max=rho, rho_episodes=0.5, free_turns_episodes=1.0, free_turns_min=0.5)
        for outcome in ('confirmed', 'kinematic', 'undefined_phase', 'short_window', 'unavailable', 'disabled', 'planetary_residual')
        for rho in (0.5, 1.5)
    ]
    for analysis in cases:
        v = analysis.verdict
        unavailable = str(Flag.GATE_UNAVAILABLE) in v.flags or str(Flag.GATE_DISABLED) in v.flags
        assert not (v.status_code == -2 and unavailable)
        assert (v.status_code == -2) == (v.veto in policy.ZLK_VETOES)
        if unavailable:
            assert v.confidence in (Confidence.MEDIUM, Confidence.LOW, Confidence.NONE)


def secular_row(word, subtype, angle_f, n_tp=2):
    verdict = make_verdict(Status.UNCERTAIN, word, angle_subtype=subtype, resonance_type=SECULAR, reason=Reason.BASELINE_NEEDED)
    t = np.linspace(0.0, SPAN, 1001)
    analysis = make_long_period_analysis(verdict, geometry(n_tp, accepted=False), angle=make_filtered_angle(t, angle_f(t)), span_yr=SPAN)
    return policy.apply_long_period(analysis, MINIMAL)


def circulating_after(t, t_from=1.25e5, turns=3.0):
    """Flat until `t_from`, then `turns` revolutions to the end."""
    return np.where(t < t_from, 0.0, 2 * np.pi * turns * (t - t_from) / (SPAN - t_from))


def test_phase_unresolved_is_uncertain_review_even_when_it_ends_in_circulation():
    v = secular_row(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.PHASE_UNRESOLVED, circulating_after).verdict
    assert v.pol_end_travel_rev >= 1
    assert (v.status, v.action, v.pol_extend_clause) == (Status.UNCERTAIN, Action.REVIEW, '')


def test_too_few_cycles_ending_in_circulation_is_decided_now():
    v = secular_row(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, circulating_after).verdict
    assert (v.status, v.reason, v.action) == (Status.NON_RESONANT, Reason.CIRCULATION, Action.NONE)
    assert str(Flag.REJECTED_CANDIDATE) in v.flags
    bounded = secular_row(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, lambda t: 0.3 * np.sin(t / 4e4)).verdict
    assert (bounded.status, bounded.action, bounded.pol_extend_clause) == (Status.UNCERTAIN, Action.EXTEND, 'A')


def test_zlk_travel_counts_half_turns():
    # omega runs 0.7 turn (1.4 pi) after its last turning point: one own revolution of a ZLK omega, not of a secular angle
    def angle(t):
        return circulating_after(t, turns=0.7)

    verdict = make_verdict(Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, angle_subtype=AngleSubtype.FEW_CYCLES, resonance_type=ZLK)
    t = np.linspace(0.0, SPAN, 1001)
    analysis = make_long_period_analysis(
        verdict, geometry(2, accepted=False), gate('short_window'), angle=make_filtered_angle(t, angle(t)), span_yr=SPAN
    )
    v = policy.apply_long_period(analysis, MINIMAL).verdict
    assert v.pol_end_travel_rev == pytest.approx(1.4, rel=1e-3) and v.status == Status.NON_RESONANT
    secular = secular_row(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, angle).verdict
    assert secular.pol_end_travel_rev == pytest.approx(0.7, rel=1e-3) and secular.status == Status.UNCERTAIN


def test_a_clear_circulation_is_certain():
    verdict = make_verdict(Status.NON_RESONANT, AngleStatus.CIRCULATION, resonance_type=SECULAR, reason=Reason.CIRCULATION)
    verdict.margins = {'P': MarginValue(margin='full', value=3.0)}
    geo = make_geometry(window=(0.0, SPAN), n_tp_cut_half=0, n_tp_cut_double=0)
    v = policy.apply_long_period(make_long_period_analysis(verdict, geo, span_yr=SPAN), MINIMAL).verdict
    assert (v.status, v.confidence) == (Status.NON_RESONANT, Confidence.CERTAIN)
    geo.n_tp_cut_double = 2  # a turning point at the doubled cutoff: high
    verdict = make_verdict(Status.NON_RESONANT, AngleStatus.CIRCULATION, resonance_type=SECULAR, reason=Reason.CIRCULATION)
    verdict.margins = {'P': MarginValue(margin='full', value=3.0)}
    assert policy.apply_long_period(make_long_period_analysis(verdict, geo, span_yr=SPAN), MINIMAL).verdict.confidence == Confidence.HIGH


def test_policy_v2_only_marks_the_row():
    verdict = make_verdict(Status.RESONANT, AngleStatus.LIBRATION, resonance_type=ZLK)
    analysis = make_long_period_analysis(verdict, geometry(), gate('planetary_residual'), span_yr=SPAN)
    before = verdict.to_flat_dict()
    after = policy.apply_long_period(analysis, AnalysisSettings(classifier_policy='v2')).verdict.to_flat_dict()
    assert after.pop('policy') == 'v2' and before.pop('policy') == ''
    assert before == after and analysis.policy is None
