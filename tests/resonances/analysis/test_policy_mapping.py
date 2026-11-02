"""W29 minimal policy on MMR rows (`policy.apply_mmr`): one case per row of the leaf table (spec §3.1, §3.2), built
from real result objects (`builders.py`). Each expected status and level is read off the PRINCIPLES by hand:
a bounded angle stays bounded unless a measured veto holds (P2), a measured objection gives level low (P3), a single
1:1 leg counts half a cycle, a half-swing between passages is a candidate, a drift is a circulation when its centre
passes a whole revolution (P1)."""

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis import policy
from resonances.resonance.analysis.ledger import Ledger, Op
from resonances.resonance.analysis.mmr import decide, models
from resonances.resonance.analysis.mmr.strength import DIRECT_METHODS
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    Action,
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    MarginValue,
    Reason,
    Status,
)
from tests.resonances.analysis.builders import (
    make,
    make_centre,
    make_episode,
    make_geometry,
    make_mmr_analysis,
    make_turning_points,
    make_verdict,
)

MINIMAL = AnalysisSettings(classifier_policy='minimal')
V2 = AnalysisSettings(classifier_policy='v2')
DECIDE = Path(decide.__file__)


def episode_geometry(n_tp=7, period=400.0, start_gap=0.2, end_gap=0.2, **fields):
    """A geometry over [0, 2000] yr holding one accepted episode of `n_tp` turning points half a period apart (the period
    above the MMR floor MMR_LIBRATION_PERIOD_MIN_YR = 150 yr)."""
    times = 100.0 + np.arange(n_tp) * period / 2
    run = make_episode(
        n_tp,
        episode_id=0,
        tp_times_yr=times,
        period_med_yr=period,
        half_amp_max_rad=0.5,
        coverage_start_gap_p=start_gap,
        coverage_end_gap_p=end_gap,
    )
    return make_geometry(window=(0.0, 2000.0), runs=[run], period_cycle_yr=period, **fields)


def bounded(rule, word=AngleStatus.LIBRATION, status=Status.UNCERTAIN, reason=Reason.ORIGIN_UNKNOWN, geometry=None, **fields):
    verdict = make_verdict(status, word, reason=reason, libration_origin=LibrationOrigin.UNKNOWN, **fields.pop('verdict', {}))
    return make_mmr_analysis(verdict, geometry or episode_geometry(), rule, decisive_episode_id=0, **fields)


def test_every_leaf_of_the_tree_has_a_mapping():
    text = DECIDE.read_text()
    found = set(re.findall(r"rule='([^']*)'", text))
    found |= set(re.findall(r"_unknown\('([^']*)'", text))
    found |= set(re.findall(r"_no_dynamics\(angle, '([^']*)'", text))
    found |= {'5a', '5a*'}  # `_no_dynamics(angle, '5a' if ... else '5a*')`
    found |= set(re.findall(r"^RULE_[A-Z0-9]+ = '([^']*)'", text, re.M)) - {decide.RULE_UNION, decide.RULE_G5W}
    found -= set(re.findall(r"replace\(angle, rule='([^']*)'", text))  # angle rules (G1l, G2), not leaves
    found |= {f'{leaf}/6' for leaf in ('5c', '5d')} | {f'{leaf}/6u' for leaf in ('5c', '5d')}  # f'{rule}/6', promote_union
    assert len(found) > 25
    missing = sorted(found - set(policy.LEAVES))
    assert not missing, missing


def test_an_unknown_leaf_takes_the_geometry_with_a_flag():
    analysis = policy.apply_mmr(bounded('9z'), MINIMAL)
    v = analysis.verdict
    assert (v.status, v.reason) == (Status.RESONANT, Reason.GEOMETRIC_LIBRATION)
    assert str(Flag.POLICY_UNKNOWN_LEAF) in v.flags


@pytest.mark.parametrize('rule', ['5g', '5a', '5e', '5b', '5f'])
def test_a_measured_objection_keeps_the_libration_at_low(rule):
    v = policy.apply_mmr(bounded(rule), MINIMAL).verdict
    assert (v.status, v.confidence, v.reason) == (Status.RESONANT, Confidence.LOW, Reason.GEOMETRIC_LIBRATION)
    assert str(Flag.MEASURED_OBJECTION) in v.flags and v.action == Action.REVIEW
    assert v.v2_verdict.startswith('uncertain/')
    assert v.counted_episode_ids == '0' and v.n_counted_episodes == 1


def test_5o_on_an_episode_is_transient_low():
    v = policy.apply_mmr(bounded('5o', word=AngleStatus.EPISODIC), MINIMAL).verdict
    assert (v.status, v.confidence, v.reason) == (Status.TRANSIENT, Confidence.LOW, Reason.GEOMETRIC_EPISODE)


def test_5a_on_near_separatrix_stays_as_the_tree_read_it():
    analysis = bounded('5a', word=AngleStatus.NEAR_SEPARATRIX, status=Status.NON_RESONANT, reason=Reason.NO_RESONANT_POTENTIAL)
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.reason, v.confidence) == (Status.NON_RESONANT, Reason.NO_RESONANT_POTENTIAL, Confidence.MEDIUM)


def strength(method, ratio, three_body=False):
    return make(models.Strength, method=method, ratio=ratio, three_body=three_body, eq_legal=True)


def test_5s_two_body_is_a_strength_veto():
    analysis = bounded(
        '5s',
        status=Status.NON_RESONANT,
        reason=Reason.IMPLAUSIBLE_STRENGTH,
        strength=strength(DIRECT_METHODS[0], 1.0e6),
        extra={'ep_n_tp_cut_half': 2, 'ep_n_tp_cut_double': 7},  # L1 = 2: not stable, no candidate (138537)
    )
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.veto, v.confidence) == (Status.NON_RESONANT, policy.VETO_STRENGTH, Confidence.HIGH)
    assert v.n_counted_episodes == 0 and v.candidate_episode_ids == ''
    assert analysis.policy.candidate_episode_ids == frozenset()


def test_a_stable_vetoed_episode_is_a_candidate_at_medium():
    analysis = bounded(
        '5s',
        status=Status.NON_RESONANT,
        reason=Reason.IMPLAUSIBLE_STRENGTH,
        strength=strength(DIRECT_METHODS[0], 1.0e6),
        extra={'ep_n_tp_cut_half': 5, 'ep_n_tp_cut_double': 7},
    )
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.veto, v.confidence, v.candidate_episode_ids) == (policy.VETO_STRENGTH, Confidence.MEDIUM, '0')


def test_5s_three_body_is_an_objection():
    analysis = bounded(
        '5s',
        word=AngleStatus.EPISODIC,
        status=Status.NON_RESONANT,
        reason=Reason.IMPLAUSIBLE_STRENGTH,
        strength=strength('eq_scale', 1e6, True),
    )
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.confidence, v.veto) == (Status.TRANSIENT, Confidence.LOW, '')
    assert str(Flag.STRENGTH_SCALE_IMPLAUSIBLE) in v.flags


def test_induced_with_a_thin_fit_is_a_medium_veto():
    owner = make(models.OwnerResult, accepted=make(models.OwnerCandidate))
    analysis = bounded(
        '2',
        status=Status.NON_RESONANT,
        reason=Reason.INDUCED_LIBRATION,
        owner=owner,
        verdict={'margins': {'W': MarginValue(margin='thin', value=0.5)}},
    )
    analysis.verdict.libration_origin = LibrationOrigin.INDUCED
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.reason, v.veto, v.confidence) == (
        Status.NON_RESONANT,
        Reason.INDUCED_LIBRATION,
        policy.VETO_OWNER,
        Confidence.MEDIUM,
    )
    assert str(Flag.VETO_THIN) in v.flags and 'veto_thin' in v.review_reasons
    assert v.confidence_basis.endswith('veto=owner:thin')


def test_a_kinematic_leaf_without_a_kinematic_origin_is_no_veto():
    v = policy.apply_mmr(bounded('3', status=Status.NON_RESONANT, reason=Reason.KINEMATIC_LIBRATION), MINIMAL).verdict
    assert v.status == Status.RESONANT and v.veto == '' and v.confidence == Confidence.LOW


def drift_geometry(pass_rev):
    centre = make_centre(max_pass_rad=pass_rev * 2 * np.pi)
    return make_geometry(window=(0.0, 1000.0), centre=centre, period_cycle_yr=np.nan)


def test_a_drift_passing_a_revolution_is_a_modulated_circulation():
    analysis = bounded('4', word=AngleStatus.DRIFT, reason=Reason.DRIFT_MECHANISM_UNKNOWN, geometry=drift_geometry(1.3))
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.reason, v.confidence, v.action) == (Status.NON_RESONANT, Reason.MODULATED_CIRCULATION, Confidence.HIGH, Action.NONE)


def test_a_drift_short_of_a_revolution_extends():
    # 0.9 rev in 1000 yr: P_dec = 1000 / 0.9 = 1111 yr > 2 T / 3 = 667 yr (clause B)
    analysis = bounded('4', word=AngleStatus.DRIFT, reason=Reason.DRIFT_MECHANISM_UNKNOWN, geometry=drift_geometry(0.9))
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.reason, v.action, v.pol_extend_clause) == (Status.UNCERTAIN, Reason.DRIFT_MECHANISM_UNKNOWN, Action.EXTEND, 'B')
    assert v.pol_period_dec_yr == pytest.approx(1000 / 0.9)
    assert v.extend_to_years is not None and v.extend_to_years > 1000


def strength_marked(geometry, ids):
    """Rule 5s's marks: the accepted runs it named forced waves by strength after the verdict."""
    nan = float('nan')
    waves = tuple(
        models.ForcedWave(
            episode_id=i,
            n_tp=7,
            period_yr=400.0,
            identity_slope_ratio=nan,
            identity_corr=nan,
            force_line='',
            force_line_offset=nan,
            wrong_sign=False,
            on_line=False,
            by_strength=True,
        )
        for i in ids
    )
    return replace(geometry, forced_waves=waves)


def test_5s_three_body_counts_the_episodes_rule_5s_marked():
    # W29 integration (2002 BG, 2011 SR158, ...): rule 5s marks the episodes as forced waves by strength after its
    # verdict; when the policy does not uphold the veto, they are this angle's episode again: transient with 1 counted
    analysis = bounded(
        '5s',
        word=AngleStatus.EPISODIC,
        status=Status.NON_RESONANT,
        reason=Reason.IMPLAUSIBLE_STRENGTH,
        strength=strength('eq_scale', 1e6, True),
        geometry=strength_marked(episode_geometry(), [0]),
    )
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.confidence, v.n_counted_episodes, v.counted_episode_ids) == (Status.TRANSIENT, Confidence.LOW, 1, '0')


def test_an_upheld_strength_veto_keeps_its_marks_out_of_the_candidates():
    analysis = bounded(
        '5s',
        status=Status.NON_RESONANT,
        reason=Reason.IMPLAUSIBLE_STRENGTH,
        strength=strength(DIRECT_METHODS[0], 1.0e6),
        geometry=strength_marked(episode_geometry(), [0]),
        extra={'ep_n_tp_cut_half': 5, 'ep_n_tp_cut_double': 7},
    )
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.veto, v.confidence, v.n_counted_episodes, v.candidate_episode_ids) == (policy.VETO_STRENGTH, Confidence.HIGH, 0, '')


def turn(t, out_in=1.0, zone=0.5):
    return g.LegTurn(t, 'planet', out_in, zone)


def leg(t0, t1, edge='', energy='pass', topology=g.LEG_HORSESHOE):
    entry = None if edge == g.LEG_EDGE_START else turn(t0)
    exit_ = None if edge == g.LEG_EDGE_END else turn(t1)
    return g.SingleLeg(t0, t1, edge, topology, 2.0, entry, exit_, force='pass', force_scale=1.0, energy=energy)


def leg_analysis(rule, legs, status=Status.UNCERTAIN, reason=Reason.ONE_COORBITAL_LEG):
    tps = make_turning_points([300.0, 450.0])  # the leg's turns: the record has turning points
    geometry = make_geometry(window=(0.0, 1000.0), coorbital=True, single_legs=tuple(legs), turning_points=tps)
    verdict = make_verdict(status, AngleStatus.EPISODIC, reason=reason)
    return make_mmr_analysis(verdict, geometry, rule)


def test_one_clean_interior_leg_is_a_transient_counted_as_half_a_cycle():
    analysis = policy.apply_mmr(leg_analysis('G4c', [leg(300.0, 450.0)], Status.TRANSIENT, Reason.DYNAMIC_EPISODE), MINIMAL)
    v = analysis.verdict
    assert (v.status, v.confidence, v.n_cycles, v.counted_episode_ids) == (Status.TRANSIENT, Confidence.MEDIUM, 0.5, 'leg')
    assert analysis.policy.counted_spans_yr == ((300.0, 450.0),) and v.n_counted_episodes == 1
    assert str(Flag.ONE_COORBITAL_LEG) in v.flags


def test_an_uncertain_leg_with_an_end_edge_leg_extends():
    legs = [leg(300.0, 450.0, energy='fail'), leg(900.0, 1000.0, edge=g.LEG_EDGE_END)]
    v = policy.apply_mmr(leg_analysis('G4c', legs), MINIMAL).verdict
    assert (v.status, v.reason, v.action, v.pol_extend_clause) == (Status.UNCERTAIN, Reason.ONE_COORBITAL_LEG, Action.EXTEND, 'A1')
    assert v.candidate_episode_ids == 'leg;leg'


def test_an_uncertain_leg_alone_is_reviewed():
    v = policy.apply_mmr(leg_analysis('G4c', [leg(300.0, 450.0, energy='fail')]), MINIMAL).verdict
    assert (v.status, v.action) == (Status.UNCERTAIN, Action.REVIEW)


def test_a_start_edge_leg_alone_is_a_circulation_without_extend():
    analysis = leg_analysis('G4c', [leg(0.0, 120.0, edge=g.LEG_EDGE_START)], reason=Reason.BASELINE_NEEDED)
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.reason, v.angle_status, v.confidence) == (
        Status.NON_RESONANT,
        Reason.CIRCULATION,
        AngleStatus.CIRCULATION,
        Confidence.MEDIUM,
    )
    assert v.action == Action.SPOT_CHECK and not v.pol_extend_clause
    assert {str(Flag.ONE_COORBITAL_LEG), str(Flag.EXTEND_TIME), str(Flag.REJECTED_CANDIDATE)} <= set(v.flags)


def test_a_leg_beside_an_accepted_libration_changes_nothing():
    # W29 integration (2009 SE, 2015 TL144, 2022 BL5: resonant/certain read as their start-edge leg): on a kept leaf
    # the accepted episode decides; legs decide only on G4c and on rule C
    geometry = replace(episode_geometry(23, period=160.0), coorbital=True, single_legs=(leg(0.0, 90.0, edge=g.LEG_EDGE_START),))
    analysis = bounded('5d/6', status=Status.RESONANT, reason=Reason.DYNAMIC_LIBRATION, geometry=geometry)
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.reason, v.counted_episode_ids) == (Status.RESONANT, Reason.DYNAMIC_LIBRATION, '0')
    assert str(Flag.ONE_COORBITAL_LEG) not in v.flags and v.action != Action.EXTEND


def test_an_end_edge_leg_beside_a_libration_reaching_the_end_does_not_extend():
    # W29 integration (624 1J-1, 643 cycles, certain, was extended by clause A1): A1 needs the legs to decide
    geometry = replace(episode_geometry(23, period=160.0), coorbital=True, single_legs=(leg(1950.0, 2000.0, edge=g.LEG_EDGE_END),))
    analysis = bounded('5d/6', status=Status.RESONANT, reason=Reason.DYNAMIC_LIBRATION, geometry=geometry)
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.pol_extend_clause) == (Status.RESONANT, '') and v.action != Action.EXTEND


def half(t0, t1, edge=''):
    return g.HalfLibration(t0, t1, edge, 1.0, 2.0, r2_h1=0.7, r2_h2=0.65, restoring=True)


def half_analysis(halves, reason):
    tps = make_turning_points([h.t0_yr for h in halves] + [h.t1_yr for h in halves])
    geometry = make_geometry(window=(0.0, 1000.0), half_librations=tuple(halves), turning_points=tps)
    return make_mmr_analysis(make_verdict(Status.UNCERTAIN, AngleStatus.EPISODIC, reason=reason), geometry, 'G4h')


def test_a_half_swing_between_passages_is_a_circulation_with_a_candidate():
    v = policy.apply_mmr(half_analysis([half(400.0, 500.0)], Reason.ONE_HALF_LIBRATION), MINIMAL).verdict
    assert (v.status, v.reason, v.angle_status, v.confidence) == (
        Status.NON_RESONANT,
        Reason.CIRCULATION,
        AngleStatus.CIRCULATION,
        Confidence.MEDIUM,
    )
    assert v.candidate_episode_ids == 'half' and v.action == Action.SPOT_CHECK
    assert {str(Flag.ONE_HALF_LIBRATION), str(Flag.REJECTED_CANDIDATE)} <= set(v.flags)


def test_a_half_swing_at_the_end_extends():
    analysis = half_analysis([half(900.0, 1000.0, g.LEG_EDGE_END)], Reason.BASELINE_NEEDED)
    analysis.verdict.angle_status = AngleStatus.INSUFFICIENT_TIME
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.action, v.pol_extend_clause) == (Status.UNCERTAIN, Action.EXTEND, 'A2')


def test_a_dynamic_libration_of_many_cycles_is_certain():
    # 23 TPs: 11 cycles + covered gaps 0.2 + 0.2 = 11.4 >= 10; L1, L2 full, identity passes, no owner found
    analysis = bounded('5d/6', status=Status.RESONANT, reason=Reason.DYNAMIC_LIBRATION, geometry=episode_geometry(23, period=160.0))
    full = MarginValue(margin='full', value=1.0)
    analysis.verdict.margins = {'L1': full, 'L2': full}
    analysis.verdict.checks = {'check_mmr_identity': CheckValue.PASS, 'check_mmr_owner': CheckValue.PASS}
    analysis.verdict.caps = ['no_full_physics_support']
    v = policy.apply_mmr(analysis, MINIMAL).verdict
    assert (v.status, v.confidence, v.n_cycles) == (Status.RESONANT, Confidence.CERTAIN, pytest.approx(11.4))
    assert v.caps == [] and 'no_full_physics_support' in v.flags and v.confidence_raw == v.confidence
    assert v.confidence_basis == 'N=11.4(cover);M=full;V=clean;S=full;veto=-'
    assert any(entry.name == 'policy:N >= confidence_certain_cycles' for entry in v.decision_path)


def test_policy_v2_only_marks_the_row():
    analysis = bounded('5g')
    before = analysis.verdict.to_flat_dict()
    after = policy.apply_mmr(analysis, V2).verdict.to_flat_dict()
    new = {'policy'}
    assert after['policy'] == 'v2' and analysis.policy is None
    assert {k: v for k, v in before.items() if k not in new} == {k: v for k, v in after.items() if k not in new}


def test_episode_rows_carry_counted_and_candidate():
    spec = make(models.ResonanceSpec, key='5J-2+0-3')
    analysis = policy.apply_mmr(bounded('5g', resonance=spec), MINIMAL)
    rows = analysis.episode_rows()
    assert [(row['episode_id'], row['counted'], row['candidate']) for row in rows] == [(0, True, False)]
    assert policy.apply_mmr(bounded('5g', resonance=spec), V2).episode_rows()[0]['counted'] is None


def test_the_policy_ledger_names_its_comparisons():
    ledger = Ledger.from_settings(MINIMAL)
    end = policy.EndReading(0.0, 0.2, 800.0, 2.0, True)
    assert policy.extend_clause(Status.RESONANT, end, 1000.0, MINIMAL, ledger=ledger) == 'A'
    assert [entry.name for entry in ledger][0].startswith('policy:A4')
    assert policy.extend_clause(Status.RESONANT, replace(end, n_last=5.0, period_dec_yr=100.0), 1000.0, MINIMAL, ledger=ledger) == ''
    assert ledger.entries[-1].op == Op.GT and ledger.entries[-1].threshold == pytest.approx(2000.0 / 3)


def test_a_measured_alias_circulation_is_certain():
    # W29 Q18 (the author): C5 measures a far from the resonance with a unique branch; with C2 full (no turning point
    # at any cutoff) that is a measured non-resonance, certain, as C1 + C2 full; C1 is not defined for an alias
    full, thin = MarginValue(margin='full', value=5.0), MarginValue(margin='thin', value=1.0)

    def alias(margins):
        verdict = make_verdict(Status.NON_RESONANT, AngleStatus.CIRCULATION, reason=Reason.ALIAS, angle_subtype=AngleSubtype.ALIAS)
        verdict.margins = margins
        return policy.apply_mmr(make_mmr_analysis(verdict, make_geometry(window=(0.0, 2000.0)), 'C5', span_yr=2000.0), MINIMAL).verdict

    v = alias({'C5': full, 'C2': full})
    assert (v.status, v.reason, v.confidence, v.action) == (Status.NON_RESONANT, Reason.ALIAS, Confidence.CERTAIN, Action.NONE)
    assert alias({'C5': thin, 'C2': full}).confidence == Confidence.HIGH  # far or branch not cleared
    assert alias({'C5': full, 'C2': thin}).confidence == Confidence.HIGH  # turning points at some cutoff
    assert alias({'C1': full, 'C2': full}).confidence == Confidence.HIGH  # an alias reads C5, never a stray C1
    # a circulation that is no alias still needs C1: C5 there is the rate check
    verdict = make_verdict(Status.NON_RESONANT, AngleStatus.CIRCULATION, reason=Reason.CIRCULATION)
    verdict.margins = {'C5': full, 'C2': full}
    analysis = make_mmr_analysis(verdict, make_geometry(window=(0.0, 2000.0)), 'C', span_yr=2000.0)
    assert policy.apply_mmr(analysis, MINIMAL).verdict.confidence == Confidence.HIGH
