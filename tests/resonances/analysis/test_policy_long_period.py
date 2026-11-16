"""W29 minimal policy on secular and Lidov-Kozai rows (`policy.apply_long_period`, spec §8), built from real result
objects. The expectations follow PRINCIPLES §7 by hand: the gate's kinematic outcome or a free e below the forced e
is a measured veto (-2); a gate without a verdict is a flag, the row keeps its geometry at <= medium; a secular swing
inside the mode-phase uncertainty is uncertain, never non_resonant. W30 (the author's D1-D6, 2026-11-03): a broken
e-i integral is a veto where rho_max >= ZLK_RHO_KINEMATIC_EXCLUDED, an objection otherwise; a held one with rho_max
that large lifts the gate's cap; episode-median and undefined-phase vetoes need the integral broken too; omega's
revolution is 2 pi; the geometry alone gives a geometric reason."""

import numpy as np
import pytest

from resonances.resonance.analysis import policy
from resonances.resonance.analysis.secular.models import EpisodeCoupling, GateResult
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

MINIMAL = AnalysisSettings()
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


BROKEN = EpisodeCoupling('none', '', 0, PERIOD, 4.0, 263.0, -0.2, 1.04, h_residual_max=0.5)  # 624, 88899 500 kyr
HELD = EpisodeCoupling('zlk', '', 0, PERIOD, 4.0, 180.0, -0.99, 0.05, h_residual_max=0.5)  # 1981 Midas


def test_an_episode_the_forced_vector_pins_is_a_free_e_veto_only_with_the_integral_broken():
    # 624 Hektor: rho_episodes 0.991 (a median) and the e-i integral broken (eta_H 1.04): the veto moves, at medium
    pins = dict(rho_max=1.4, rho_episodes=0.991, free_turns_episodes=0.9, free_turns_min=0.5)
    v = zlk(AngleStatus.EPISODIC, 'kinematic', episode_ei=BROKEN, **pins).verdict
    assert (v.status, v.veto, v.confidence, v.status_code) == (Status.NON_RESONANT, policy.VETO_ZLK_FREE_E, Confidence.MEDIUM, -2)
    assert str(Flag.VETO_THIN) in v.flags
    # the median alone is no pointwise measurement (W30 D4): an objection, the geometry stays at low
    w = zlk(AngleStatus.EPISODIC, 'kinematic', **pins).verdict
    assert (w.status, w.veto, w.confidence) == (Status.TRANSIENT, '', Confidence.LOW) and str(Flag.MEASURED_OBJECTION) in w.flags


def test_an_undefined_phase_with_the_free_circle_off_the_origin_vetoes_only_with_the_integral_broken():
    v = zlk(outcome='undefined_phase', rho_max=0.5, episode_ei=BROKEN).verdict
    assert (v.veto, v.status_code) == (policy.VETO_ZLK_FREE_E, -2)
    u = zlk(outcome='undefined_phase', rho_max=0.5).verdict  # a missing phase is no contradiction (W30 D4)
    assert (u.status, u.veto, u.confidence) == (Status.RESONANT, '', Confidence.LOW)
    w = zlk(outcome='undefined_phase', rho_max=1.5).verdict
    assert (w.status, w.status_code) == (Status.RESONANT, 2) and str(Flag.GATE_UNAVAILABLE) in w.flags


def test_a_broken_integral_is_a_veto_where_the_forced_terms_are_small():
    # 88899 500 kyr: no gate verdict, rho_max 13 >= 3, eta_H 1.24: non_resonant / kinematic, veto zlk_integral (D1 C)
    v = zlk(
        AngleStatus.EPISODIC, 'planetary_residual', rho_max=13.3, episode_ei=BROKEN, min_distance_arcsec_yr=2.0, baseline_yr=SPAN
    ).verdict
    assert (v.status, v.veto, v.status_code, v.origin_detail) == (Status.NON_RESONANT, policy.VETO_ZLK_INTEGRAL, -2, 'ei_integral_not_held')
    # rho_max 2 < 3: the forced terms may break the integral themselves: an objection, low
    w = zlk(
        AngleStatus.EPISODIC, 'planetary_residual', rho_max=2.0, episode_ei=BROKEN, min_distance_arcsec_yr=2.0, baseline_yr=SPAN
    ).verdict
    assert (w.status, w.veto, w.confidence) == (Status.TRANSIENT, '', Confidence.LOW) and str(Flag.EI_INTEGRAL_NOT_HELD) in w.flags


def test_a_held_integral_with_a_small_forced_vector_lifts_the_gate_cap():
    # 1981 Midas: node near s6 (planetary_residual), rho_max 69, eta_H 0.05: high by cycles, no extension (D2 C)
    v = zlk(outcome='planetary_residual', rho_max=69.0, episode_ei=HELD, min_distance_arcsec_yr=0.25, baseline_yr=SPAN).verdict
    assert (v.status, v.confidence, v.action) == (Status.RESONANT, Confidence.HIGH, Action.NONE)
    assert {str(Flag.GATE_UNAVAILABLE), str(Flag.KINEMATIC_EXCLUDED)} <= set(v.flags) and str(Flag.VETO_NEEDS_LENGTH) not in v.flags
    # the same with rho_max 2: the gate's abstention keeps the cap
    w = zlk(outcome='planetary_residual', rho_max=2.0, episode_ei=HELD, min_distance_arcsec_yr=0.25, baseline_yr=SPAN).verdict
    assert w.confidence == Confidence.MEDIUM and str(Flag.KINEMATIC_EXCLUDED) not in w.flags


def test_only_a_confirmed_gate_names_a_dynamic_origin():
    v = zlk().verdict
    assert (v.reason, v.libration_origin) == (Reason.DYNAMIC_LIBRATION, 'dynamic')
    w = zlk(outcome='disabled').verdict
    assert (w.reason, w.libration_origin) == (Reason.GEOMETRIC_LIBRATION, 'unknown')


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


def test_zlk_travel_counts_whole_turns():
    # omega runs 0.7 turn after its last turning point: less than one revolution for a ZLK omega too (W30 D3: the
    # quadrupole's half-turn symmetry is not general; 88899 100 kyr)
    def angle(t):
        return circulating_after(t, turns=0.7)

    verdict = make_verdict(Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, angle_subtype=AngleSubtype.FEW_CYCLES, resonance_type=ZLK)
    t = np.linspace(0.0, SPAN, 1001)
    analysis = make_long_period_analysis(
        verdict, geometry(2, accepted=False), gate('short_window'), angle=make_filtered_angle(t, angle(t)), span_yr=SPAN
    )
    v = policy.apply_long_period(analysis, MINIMAL).verdict
    assert v.pol_end_travel_rev == pytest.approx(0.7, rel=1e-3) and v.status == Status.UNCERTAIN
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


@pytest.mark.parametrize(
    'turns, level, action, thin',
    [
        # 143662 s+g5-g8-s3 (W36 F5): 1.09 rev with 2 turning points; within CONFIDENCE_THIN_BAND = 0.1 of one turn the
        # record still ends in circulation (non_resonant) but at most medium (spot_check), as a thin veto
        (1.05, Confidence.MEDIUM, Action.SPOT_CHECK, True),
        (1.09, Confidence.MEDIUM, Action.SPOT_CHECK, True),
        # 1.12 and 3 rev are clear of the band: high, as before (no candidate, no clear circulation: 2 TPs exist)
        (1.12, Confidence.HIGH, Action.NONE, False),
        (3.0, Confidence.HIGH, Action.NONE, False),
    ],
)
def test_a_thin_end_travel_is_non_resonant_at_most_medium(turns, level, action, thin):
    # no accepted run: the travel is read over the whole window, max - min of the ramp = `turns` exactly
    v = secular_row(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, lambda t: circulating_after(t, turns=turns)).verdict
    assert v.pol_end_travel_rev == pytest.approx(turns, rel=1e-9)
    assert (v.status, v.reason, v.confidence, v.action) == (Status.NON_RESONANT, Reason.CIRCULATION, level, action)
    assert v.pol_decided_by == policy.DECIDED_INSUFFICIENT
    assert (policy.END_TRAVEL_INPUT in v.pol_level_inputs) == thin
    if thin:
        assert v.pol_level_inputs.endswith('end_travel:thin:medium')
    entry = next(e for e in v.decision_path if e.name == policy.END_TRAVEL_THIN_ENTRY)
    assert entry.threshold == pytest.approx(1.1) and entry.result is (not thin)


def test_a_thin_end_travel_caps_a_zlk_row_too_and_never_an_uncertain_one():
    def angle(t):
        return circulating_after(t, turns=1.04)

    verdict = make_verdict(Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, angle_subtype=AngleSubtype.FEW_CYCLES, resonance_type=ZLK)
    t = np.linspace(0.0, SPAN, 1001)
    analysis = make_long_period_analysis(
        verdict, geometry(2, accepted=False), gate('short_window'), angle=make_filtered_angle(t, angle(t)), span_yr=SPAN
    )
    v = policy.apply_long_period(analysis, MINIMAL).verdict
    assert (v.status, v.confidence) == (Status.NON_RESONANT, Confidence.MEDIUM)
    assert 'end_travel:thin:medium' in v.pol_level_inputs
    # 0.96 rev, as thin below one turn: the record does not end in circulation, uncertain has no level to cap
    bounded = secular_row(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, lambda t: circulating_after(t, turns=0.96)).verdict
    assert (bounded.status, bounded.confidence) == (Status.UNCERTAIN, Confidence.NONE)
    assert policy.END_TRAVEL_INPUT not in bounded.pol_level_inputs
    assert all(e.name != policy.END_TRAVEL_THIN_ENTRY for e in bounded.decision_path)
