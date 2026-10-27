"""W18 decision rules and marks.

- The branch where the force law (L4) and the potential with a (R0) are both not measured at decision time
  (`decide.unmeasured_branch`, author 2026-10-17): direct averaging confirms the strength -> as before; implausible
  strength -> non_resonant (three-body only when the small-divisor envelope cannot close the gap); a measured objection
  or one independent cycle -> uncertain; nothing measured -> the geometry's status, origin unknown, confidence <= low.
- The small-divisor envelope (`strength.small_divisor_envelope`) on hand-built cases and against Codex's independent
  numbers for the benchmark rows (W17 Q3 codex-fix/small_divisor_envelopes.csv).
- near_separatrix of the geometry (G3) without a measured zone: confidence <= medium (564160 3N-4).
- The directed transitions of `pendulum.regime_transitions`, the mark exit_and_capture and the rename of
  separatrix_chaos to episodes_in_separatrix_zone (old files still read).

Expected values come from the constructions (hand-computed rates, ratios and moments) or from the W17 reports, not
from the package.
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis import pendulum
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.decide import AngleDecision, StatusDecision
from resonances.resonance.analysis.mmr.models import ResonanceSpec, Strength
from resonances.resonance.analysis.pendulum import EnergyAnalysis, SeparatrixMechanism
from resonances.resonance.analysis.settings import AnalysisSettings, settings_from_config
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    MarginClass,
    MarginValue,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)
from tests.resonances.analysis.builders import make_episode, make_geometry, make_response, make_sampling

S = settings_from_config()
LIMIT = S.mmr_strength_implausible_ratio  # 1000
EPISODIC = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
LIBRATION = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', 0)
SAMPLING = make_sampling(
    unwrap_resid_p99_rad=0.0, unwrap_resid_max_rad=0.0, unwrap_ok=True, undefined_phase=False, verdict='resolved', alias_numpy_differs=False
)


def _ledger():
    return Ledger.from_settings(S)


def _strength(ratio, method=st.METHOD_SCALE, check=CheckValue.NOT_MEASURED, legal=False, three_body=False, **divisor):
    return Strength(
        mu_scale=1e-9,
        mu_scale_no_eq=1e-6,
        eq_legal=legal,
        mu_eff_required=ratio * 1e-9,
        ratio=ratio,
        omega0_theory_rad_yr=float('nan'),
        period_theory_yr=float('nan'),
        implausible=bool(np.isfinite(ratio) and ratio >= LIMIT),
        check=check,
        method=method,
        three_body=three_body,
        **divisor,
    )


def _geometry(centre_estimates=2):
    return make_geometry(runs=[make_episode(episode_id=0, centre_estimates=centre_estimates)])


def _physics(before=0.1, inside=1.0):
    return [SimpleNamespace(episode_id=0, a_band_std_before_au=before, a_band_std_in_au=inside)]


def _branch(angle=EPISODIC, strength=None, physics=None, centre_estimates=2):
    ledger = _ledger()
    physics = _physics() if physics is None else physics
    decision = decide.unmeasured_branch(angle, _geometry(centre_estimates), None, strength, physics, S, ledger)
    return decision, ledger


# --- the branch -------------------------------------------------------------------------------------------------


def test_direct_averaging_that_confirms_the_strength_keeps_the_dynamic_leaf():
    decision, _ = _branch(LIBRATION, _strength(1.3, st.METHOD_DIRECT, CheckValue.PASS, legal=True))
    assert (decision.status, decision.reason, decision.origin, decision.rule) == (
        Status.RESONANT,
        Reason.DYNAMIC_LIBRATION,
        LibrationOrigin.DYNAMIC,
        '5d/6',
    )
    assert not decision.caps


@pytest.mark.parametrize('ratio, implausible', [(1.5e7, True), (LIMIT, True), (999.0, False)])
def test_two_body_strength_far_above_the_directly_averaged_one_is_non_resonant(ratio, implausible):
    # 105247 29M-52: the period needs 1.5e7 x the directly averaged strength (W17 Q3)
    decision, ledger = _branch(strength=_strength(ratio, st.METHOD_DIRECT, CheckValue.FAIL, legal=True))
    if implausible:
        assert (decision.status, decision.reason, decision.origin, decision.rule) == (
            Status.NON_RESONANT,
            Reason.IMPLAUSIBLE_STRENGTH,
            LibrationOrigin.UNKNOWN,
            '5s',
        )
        assert decision.caps == [str(conf.Cap.IMPLAUSIBLE_STRENGTH)]
    else:  # F fails: a measured objection
        assert (decision.status, decision.rule) == (Status.UNCERTAIN, '5o')
    entry = next(e for e in ledger if e.name.startswith('5s required / directly averaged'))
    assert entry.value == ratio and entry.threshold == LIMIT


@pytest.mark.parametrize(
    'kind, residual, rule',
    [
        (st.DIVISOR_NEAREST, 2.97e9, '5s'),  # 2012 KL9: small divisors leave 3e9
        (st.DIVISOR_NEAREST, 22.3, '5s*'),  # 46596: they leave only 22
        (st.DIVISOR_NO_CANDIDATE, 140.2, '5s*'),  # 501710: at least 140, not >= 1000
        (st.DIVISOR_CROSSES_ZERO, 0.0, '5s*'),  # 8306: 14M-25 passes its exact commensurability
        (st.DIVISOR_NO_PERIOD, float('nan'), '5s*'),
    ],
)
def test_three_body_strength_is_implausible_only_when_small_divisors_cannot_close_the_gap(kind, residual, rule):
    strength = _strength(1e9, check=CheckValue.FAIL, legal=True, three_body=True, divisor_kind=kind, divisor_residual=residual)
    decision, _ = _branch(strength=strength)
    expected = (Status.NON_RESONANT, Reason.IMPLAUSIBLE_STRENGTH) if rule == '5s' else (Status.UNCERTAIN, Reason.ORIGIN_UNKNOWN)
    assert (decision.status, decision.reason, decision.origin, decision.rule) == expected + (LibrationOrigin.UNKNOWN, rule)
    assert str(Flag.IMPLAUSIBLE_STRENGTH) in decision.flags


def test_an_illegal_three_body_estimate_is_no_implausibility():
    # 1998 QA62 crosses Venus and Mars: the ratio 1.7e5 is against mu alone, no rule 5s; nothing else measured
    decision, _ = _branch(strength=_strength(1.7e5, three_body=True), centre_estimates=2)
    assert (decision.status, decision.origin, decision.rule) == (Status.TRANSIENT, LibrationOrigin.UNKNOWN, '5u/6')
    assert decision.reason == Reason.GEOMETRIC_EPISODE


@pytest.mark.parametrize('before, objects', [(0.6, True), (0.5, True), (0.4, False)])
def test_a_strong_wave_before_the_episode_is_a_measured_objection(before, objects):
    # O1 = before / inside against MMR_O1_RATIO_FAIL = 0.5: 0.6 and 0.5 fail, 0.4 is only thin
    decision, _ = _branch(physics=_physics(before, 1.0))
    assert decision.rule == ('5o' if objects else '5u/6')
    assert decision.status == (Status.UNCERTAIN if objects else Status.TRANSIENT)


def test_o1_is_not_an_objection_for_a_libration_over_the_whole_record():
    decision, _ = _branch(LIBRATION, physics=_physics(5.0, 1.0))
    assert (decision.status, decision.rule) == (Status.RESONANT, '5u/6')


@pytest.mark.parametrize('ratio, check', [(50.0, CheckValue.FAIL), (0.05, CheckValue.FAIL)])
def test_a_failed_strength_check_is_a_measured_objection(ratio, check):
    decision, _ = _branch(strength=_strength(ratio, check=check, legal=True, three_body=True))
    assert (decision.status, decision.reason, decision.rule) == (Status.UNCERTAIN, Reason.ORIGIN_UNKNOWN, '5o')


def test_centre_outside_the_well_is_a_measured_objection():
    strength = _strength(7.8, st.METHOD_DIRECT, CheckValue.FAIL, legal=True)
    strength = Strength(**{**strength.__dict__, 'not_measured_reason': st.REASON_CENTRE_OUTSIDE_WELL, 'n_consistent': 0})
    decision, ledger = _branch(strength=strength)  # 482824 19N-32: ratio 7.8, no swing in the well
    assert decision.rule == '5o'
    assert any(e.name.startswith('5o F an observed swing') and e.result is False for e in ledger)


@pytest.mark.parametrize('estimates, rule', [(1, '5n'), (S.libration_centre_min_estimates, '5u/6')])
def test_one_independent_cycle_is_uncertain(estimates, rule):
    # 2003 ND: a libration over the record with one centre estimate (LIBRATION_CENTRE_MIN_ESTIMATES = 2)
    decision, _ = _branch(LIBRATION, centre_estimates=estimates)
    assert decision.rule == rule
    assert decision.status == (Status.UNCERTAIN if rule == '5n' else Status.RESONANT)


def test_nothing_measured_keeps_the_geometry_with_origin_unknown_and_at_most_low():
    decision, _ = _branch(LIBRATION)
    assert (decision.status, decision.origin, decision.caps) == (Status.RESONANT, LibrationOrigin.UNKNOWN, ['geometry_only'])
    assert decision.reason == Reason.GEOMETRIC_LIBRATION  # only the geometry decided (author 2026-10-26, Q3)
    # every margin full and the owner search measured (W full): the cap geometry_only still holds the level at low
    margins = {key: MarginValue(Margin.FULL, 1.0, MarginClass.DECISIVE) for key in ('L1', 'L2', 'L6', 'L7', 'W')}
    assessment = conf.assess(
        decision.status, 'libration', 'geometric_libration', margins, decision.caps, [], ('L4', 'R0', 'W'), conf.ConfidenceCells()
    )
    assert (assessment.confidence_raw, assessment.confidence) == (Confidence.CERTAIN, Confidence.LOW)
    assert decide.own_dynamics_claimed(decision)


def test_the_branch_reads_l4_and_r0_at_decision_time():
    """117306 11M-20: R0 passed at decision time, the filter guard later made check_mmr_potential not_measured; the
    branch must not take it. decide_status reads the energy it is given (the guard runs after)."""
    response = make_response(
        a_response=CheckValue.NOT_MEASURED,
        a_amplitude=CheckValue.PASS,
        slow_libration=CheckValue.NOT_APPLICABLE,
        identity=CheckValue.PASS,
        b1_ratio=0.9,
        force_r2_h1=float('nan'),
        force_r2_h2=float('nan'),
        force_contaminated=False,
    )
    energy = EnergyAnalysis(None, None, 1.0, 1.0, with_slow=CheckValue.PASS)
    args = (SAMPLING, _geometry(1), energy, response, None, None, None, _strength(float('nan')), S, _ledger())
    decision = decide.decide_status(EPISODIC, *args, episodes=_physics())
    assert (decision.rule, decision.origin) == ('5d/6', LibrationOrigin.DYNAMIC)
    energy.with_slow = CheckValue.NOT_MEASURED
    decision = decide.decide_status(EPISODIC, *args[:-1], _ledger(), episodes=_physics())
    assert (decision.rule, decision.status) == ('5n', Status.UNCERTAIN)


# --- the small-divisor envelope -----------------------------------------------------------------------------------

N_J = 2 * math.pi / 11.862  # rad/yr


def _context(n_lo, n_hi, n_mean):
    a = lambda n: (2 * math.pi / n) ** (2 / 3)  # noqa: E731  n = 2 pi a^-1.5
    return SimpleNamespace(
        a_p01_au=a(n_lo),
        a_p99_au=a(n_hi),
        body_varpi_rate_rad_yr=0.0,
        mean_motion_rad_yr=n_mean,
        # Saturn nearly at rest: none of its two-body angles comes near a commensurability with the body
        planet_mean_motion_rad_yr={'Jupiter': N_J, 'Saturn': 1e-9},
    )


SPEC = ResonanceSpec.from_key('2J-1S-1+0+0+0')


def test_envelope_takes_the_nearest_endpoint_rate():
    # n from n_J/2 (1 + 1e-3) to n_J/2 (1 + 2e-3): the 1:2 angle with Jupiter turns at n_J - 2 n = -1e-3 n_J and
    # -2e-3 n_J; 2 pi / P = 0.8e-3 n_J bounds the rates at 1.5 x 2 pi / P = 1.2e-3 n_J, which only 1:2 reaches here,
    # and 1:2 turns faster than sigma swings (1e-3 > 0.8e-3 n_J): the swing cap does not bind
    n_lo, n_hi = N_J / 2 * (1 + 1e-3), N_J / 2 * (1 + 2e-3)
    period = 2 * math.pi / (0.8e-3 * N_J)
    envelope = st.small_divisor_envelope(SPEC, _context(n_lo, n_hi, (n_lo + n_hi) / 2), 1e9, period, period, S)
    b = ((n_lo + n_hi) / 2) / (1e-3 * N_J)  # = 500.75
    assert envelope.kind == st.DIVISOR_NEAREST and envelope.candidate == '1J-2' and not envelope.capped
    assert envelope.b == pytest.approx(b, rel=1e-6)
    assert envelope.residual == pytest.approx(1e9 / b**2, rel=1e-6)


def test_envelope_caps_a_divisor_slower_than_the_swing():
    # W22 R-DIV (author 2026-10-24): the same 1:2 angle, but 2 pi / P = 2e-3 n_J: it turns slower than sigma swings,
    # so |Delta| counts 2 pi / P and B = n P / 2 pi = n / (2e-3 n_J) = 250.375 instead of 500.75
    n_lo, n_hi = N_J / 2 * (1 + 1e-3), N_J / 2 * (1 + 2e-3)
    period = 1.5 * 2 * math.pi / (3e-3 * N_J)
    envelope = st.small_divisor_envelope(SPEC, _context(n_lo, n_hi, (n_lo + n_hi) / 2), 1e9, period, period, S)
    b = ((n_lo + n_hi) / 2) / (2e-3 * N_J)
    assert b == pytest.approx(250.375, rel=1e-9)
    assert envelope.kind == st.DIVISOR_NEAREST and envelope.candidate == '1J-2' and envelope.capped
    assert envelope.b == pytest.approx(b, rel=1e-6)
    assert envelope.residual == pytest.approx(1e9 / b**2, rel=1e-6)


def test_envelope_of_a_range_across_the_commensurability_is_capped_by_the_swing():
    # the range passes 1:2: |Delta| = 0 is no longer an unbounded B (W22 R-DIV); B = n min(P, P_cyc) / 2 pi with the
    # shorter of the two periods (here P_cyc = P / 2)
    n_lo, n_hi = N_J / 2 * (1 - 1e-3), N_J / 2 * (1 + 1e-3)
    period = 1.5 * 2 * math.pi / (3e-3 * N_J)
    envelope = st.small_divisor_envelope(SPEC, _context(n_lo, n_hi, N_J / 2), 1e9, period, period / 2, S)
    b = (N_J / 2) * (period / 2) / (2 * math.pi)  # = n_J/2 / (4e-3 n_J) = 125
    assert b == pytest.approx(125.0, rel=1e-9)
    assert (envelope.kind, envelope.candidate, envelope.capped) == (st.DIVISOR_CROSSES_ZERO, '1J-2', True)
    assert envelope.b == pytest.approx(b, rel=1e-9)
    assert envelope.residual == pytest.approx(1e9 / 125.0**2, rel=1e-9)


def test_envelope_without_a_candidate_is_bounded_by_the_rate_bound():
    # one value of n, far from any k1:k2 with k1 <= 60, |k2 - k1| <= 20 by more than the bound 1e-9 rad/yr
    n = N_J * 0.6123456
    period = 1.5 * 2 * math.pi / 1e-9
    envelope = st.small_divisor_envelope(SPEC, _context(n, n, n), 1e6, period, period, S)
    assert envelope.kind == st.DIVISOR_NO_CANDIDATE and envelope.candidate == ''
    assert envelope.b == pytest.approx(n / 1e-9, rel=1e-9)
    assert envelope.residual == pytest.approx(1e6 * (1e-9 / n) ** 2, rel=1e-9)


# Codex's independent envelope (W17 Q3 codex-fix/small_divisor_envelopes.csv, review_numbers.py): B at the most
# favourable endpoint, the residual ratio / B^2 and the candidate; 8306: a candidate crosses zero (B infinite). These
# are before the swing cap of W22 R-DIV (author 2026-10-24), which the benchmark test applies on top: B = min(B_codex,
# n min(P, P_cyc) / 2 pi). The R-3B impact run (an independent monkeypatch, SP/dis/impact/R-3B) gave 8306 B = 53.1,
# residual 1.98e6 (5s, non_resonant).
CODEX_ENVELOPES = {
    'mmr3__2012_KL9__5E-2M-10+0+0+7': (st.DIVISOR_NEAREST, 747.90, 2.9717e9, '57M-77'),
    'mmr3__2012_TK84__1M-9J+1+0+0+7': (st.DIVISOR_NEAREST, 48.260, 6.5289e7, '8J-3'),
    'mmr3__46596__8J+5S-3+0+0-10': (st.DIVISOR_NEAREST, 1357.6, 22.32, '10J-3'),
    'mmr3__501710__1E+7S-4+0+0-4__500kyr': (st.DIVISOR_NO_CANDIDATE, 112.63, 140.2, ''),
    'mmr3__8306__2M-5S-3+0+0+6': (st.DIVISOR_CROSSES_ZERO, math.inf, 0.0, '14M-25'),
}
R3B_IMPACT = {'mmr3__8306__2M-5S-3+0+0+6': (53.1, 1.98e6)}

# the verdicts the author chose for the branch (ANSWERS 2026-10-17c, Q3) and the rule that gives them
EXPECTED_BRANCH = {
    'mmr3__2012_KL9__5E-2M-10+0+0+7': (Status.NON_RESONANT, '5s'),
    'mmr3__2012_TK84__1M-9J+1+0+0+7': (Status.NON_RESONANT, '5s'),
    # W22 R-DIV (author 2026-10-24): the swing cap bounds B, the residual reaches 1e3 -> 5s (the author's non_resonant)
    'mmr3__8306__2M-5S-3+0+0+6': (Status.NON_RESONANT, '5s'),
    # without the W20 cuts (below) min(P, P_cyc) = 913 yr: B 1358 -> 257, residual 22 -> 624 < 1e3, still 5s*; with
    # the cuts (the package) P_cyc = 286 yr of the ripple episodes caps B at 80.5: residual 6355, non_resonant (R-3B)
    'mmr3__46596__8J+5S-3+0+0-10': (Status.UNCERTAIN, '5s*'),
    'mmr3__501710__1E+7S-4+0+0-4__500kyr': (Status.UNCERTAIN, '5s*'),
    'mmr2__105247__29M-52+0+23': (Status.NON_RESONANT, '5s'),
    'mmr3__2003_ND__6J-6S-1+0+0+1': (Status.UNCERTAIN, '5n'),
    'mmr3__2013_VA14__2J+6S-1+0+0-7': (Status.UNCERTAIN, '5n'),
    # W21 D (Q2): O1's background no longer lies on episode #0 (no circulation between #0 and #1: one group), O1
    # 0.705 (fail) -> 0.450 (thin), so 5o no longer decides; 5n (one independent cycle) gives the same status
    'mmr3__1998_QA62__1V-5M+3+0+0+1': (Status.UNCERTAIN, '5n'),
}


# W20 cuts long runs whose centre moves at a stall of the slow centre (LIBRATION_REGIME_CUTS): in 46596 8J+5S-3 and
# 501710 1E+7S-4 500 kyr that exposes ripple episodes of ~1 deg (18 TPs at 286 yr; 20 TPs at 547 yr) that become the
# decisive ones, and the branch then reads 5g instead of 5s*, the status the same. Codex's envelopes were computed on the
# episodes before W20, so these benchmark checks run without the cuts.
BEFORE_W20_CUTS = {'mmr3__46596__8J+5S-3+0+0-10', 'mmr3__501710__1E+7S-4+0+0-4__500kyr'}
# W20 CS7: the slow centre of 501710 500 kyr drifts with 4 varpi under ~1-deg ripple episodes (K2: corr 0.98, slope 1.03,
# share 0.005, force law R2 0.10/0.11), and step 3 makes it non_resonant/kinematic_drift before the branch is reached;
# the branch itself (the author's 5s* of 2026-10-17) is checked with K2 off (no swing passes a share of 0)
BEFORE_W20_K2 = {'mmr3__501710__1E+7S-4+0+0-4__500kyr'}


def _analyse(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    settings = replace(S, libration_regime_cuts=False) if example in BEFORE_W20_CUTS else S
    if example in BEFORE_W20_K2:
        settings = replace(settings, mmr_kin_ripple_share_max=0.0)
    return analyse_mmr(io.load_series(example), settings)


@pytest.mark.parametrize('example', sorted(CODEX_ENVELOPES))
def test_envelope_agrees_with_codex_on_the_benchmark(example):
    kind, b_codex, residual_codex, candidate = CODEX_ENVELOPES[example]
    analysis = _analyse(example)
    strength = analysis.strength
    assert (strength.divisor_kind, strength.divisor_candidate) == (kind, candidate)
    if kind == st.DIVISOR_NO_CANDIDATE:  # n / bound < the swing cap: unchanged
        assert strength.divisor_b == pytest.approx(b_codex, rel=1e-3) and not strength.divisor_capped
        assert strength.divisor_residual == pytest.approx(residual_codex, rel=2e-3)
        return
    run = next(r for r in analysis.geometry.runs if r.episode_id == analysis.decisive_episode_id)
    periods = [p for p in (run.period_med_yr, analysis.geometry.period_cycle_yr) if math.isfinite(p) and p > 0]
    b_cap = analysis.context.mean_motion_rad_yr * min(periods) / (2 * math.pi)
    b = min(b_codex, b_cap)
    assert strength.divisor_capped == (b_cap < b_codex)
    assert strength.divisor_b == pytest.approx(b, rel=1e-3)
    assert strength.divisor_residual == pytest.approx(strength.ratio / b**2, rel=2e-3)
    if math.isfinite(b_codex):  # Codex's residual is ratio / B_codex^2: the same ratio
        assert strength.ratio == pytest.approx(residual_codex * b_codex**2, rel=3e-3)
    if example in R3B_IMPACT:
        b_impact, residual_impact = R3B_IMPACT[example]
        assert strength.divisor_b == pytest.approx(b_impact, rel=1e-3)
        assert strength.divisor_residual == pytest.approx(residual_impact, rel=1e-2)


@pytest.mark.parametrize('example', sorted(EXPECTED_BRANCH))
def test_branch_verdicts_on_the_benchmark(example):
    status, rule = EXPECTED_BRANCH[example]
    analysis = _analyse(example)
    v = analysis.verdict
    assert (v.status, analysis.extra_columns['rec_rule']) == (status, rule)
    assert v.libration_origin == LibrationOrigin.UNKNOWN
    if status == Status.NON_RESONANT:
        assert v.reason == Reason.IMPLAUSIBLE_STRENGTH and v.confidence == Confidence.LOW


# --- near_separatrix without a zone -------------------------------------------------------------------------------


def _zone_energy(zone, cycles=(1,)):
    return EnergyAnalysis(None, None, 1.0, 1.0, cycles=list(cycles), zone=zone)


@pytest.mark.parametrize(
    'zone, z_margin, capped',
    [(object(), Margin.FULL, False), (object(), Margin.THIN, True), (None, Margin.NOT_MEASURED, True)],
)
def test_near_separatrix_of_the_geometry_is_capped_without_a_measured_zone(zone, z_margin, capped):
    angle = AngleDecision(AngleStatus.NEAR_SEPARATRIX, AngleSubtype.PATTERN_B, 'G3')
    decision = StatusDecision(Status.NEAR_SEPARATRIX, Reason.SEPARATRIX_ZONE, LibrationOrigin.DYNAMIC, '', '', rule='5d/6')
    margins = {'Z': MarginValue(z_margin, 1.0)}
    caps = decide.collect_caps(decide.tree_view(angle, None, _zone_energy(zone)), decision, make_geometry(), margins, S)
    assert (str(conf.Cap.NEAR_SEPARATRIX_WITHOUT_ZONE) in caps) == capped
    assert conf.CAP_CEILING[conf.Cap.NEAR_SEPARATRIX_WITHOUT_ZONE] == Confidence.MEDIUM
    # a non_resonant pattern B (5a) is no near_separatrix verdict: no cap
    other = StatusDecision(Status.NON_RESONANT, Reason.MODULATED_CIRCULATION, LibrationOrigin.NONE, '', '', rule='5a')
    assert str(conf.Cap.NEAR_SEPARATRIX_WITHOUT_ZONE) not in decide.collect_caps(
        decide.tree_view(angle, None, None), other, make_geometry(), margins, S
    )


def test_564160_is_near_separatrix_at_medium():
    v = _analyse('mmr2__564160__3N-4+0+1').verdict
    assert (v.status, v.confidence) == (Status.NEAR_SEPARATRIX, Confidence.MEDIUM)
    assert str(conf.Cap.NEAR_SEPARATRIX_WITHOUT_ZONE) in v.caps and v.checks['check_mmr_separatrix_zone'] != CheckValue.PASS


# --- exits and captures -------------------------------------------------------------------------------------------


def _slip_record():
    """Librations swinging between +-1 rad, a slip over the top between t = 20 and 24 (sigma from -1 to 2 pi + 1),
    librations again; the angle is linear between its turning points."""
    times = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0])
    values = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 2 * np.pi + 1, 2 * np.pi - 1, 2 * np.pi + 1, 2 * np.pi - 1])
    t = np.arange(0.02, 36.0, 0.05)  # samples at ... 21.97, 22.02 ...
    return t, np.interp(t, times, values), times, values


def test_a_slip_between_librations_is_one_exit_and_one_capture_at_one_moment():
    t, angle, times, values = _slip_record()
    # the slip passes the middle of its swing, (-1 + 2 pi + 1) / 2 = pi, at t = 20 + 4 (pi + 1) / (2 pi + 2) = 22.0;
    # the moment is the first sample after the crossing, 22.02
    moment = 22.02
    transitions = pendulum.regime_transitions(t, angle, times, values, [])
    assert [kind for _, kind in transitions] == [pendulum.EXIT, pendulum.CAPTURE]
    assert [m for m, _ in transitions] == pytest.approx([moment, moment])


def test_a_libration_that_ends_in_turns_is_an_exit_and_turns_into_a_libration_a_capture():
    t = np.linspace(0, 40, 401)
    angle = np.zeros_like(t)
    # half-cycles centred at 2 and 6 (TPs at 0, 4, 8), a turn centred at 20, then half-cycles centred at 32 and 36
    exit_only = pendulum.regime_transitions(t, angle, [0.0, 4.0, 8.0], [1.0, -1.0, 1.0], [20.0])
    assert exit_only == [(pytest.approx(13.0), pendulum.EXIT)]  # midpoint of 6 and 20
    capture_only = pendulum.regime_transitions(t, angle, [30.0, 34.0, 38.0], [1.0, -1.0, 1.0], [20.0])
    assert capture_only == [(pytest.approx(26.0), pendulum.CAPTURE)]  # midpoint of 20 and 32
    energy = EnergyAnalysis(None, None, 1.0, 1.0, transitions=exit_only + capture_only, crossing_times_yr=[13.0, 26.0])
    columns = energy.to_columns()
    assert (columns['sep_n_exits'], columns['sep_n_captures'], columns['sep_n_crossings']) == (1, 1, 2)


def _mark(**changes):
    energy = EnergyAnalysis(
        None, None, 1.0, 1.0, transitions=[(10.0, pendulum.EXIT), (10.0, pendulum.CAPTURE)], mechanism=SeparatrixMechanism.UNEXPLAINED
    )
    args = dict(
        spec=ResonanceSpec.from_key('6N-7+0+1'),
        angle_status=AngleStatus.EPISODIC,
        energy=energy,
        checks={'check_mmr_a_response': CheckValue.PASS, 'check_mmr_potential': CheckValue.NOT_MEASURED},
        flags=['zone_eq_only'],
        caps=['transient_unexplained'],
        reason=Reason.DYNAMIC_EPISODE,
        subtype=AngleSubtype.RECURRENT,
    )
    args.update(changes)
    return decide.exit_and_capture(ledger=_ledger(), **args)


def test_exit_and_capture_mark_and_what_keeps_it_off():
    assert _mark()
    exit_only = EnergyAnalysis(None, None, 1.0, 1.0, transitions=[(10.0, pendulum.EXIT)])
    assert not _mark(energy=exit_only)
    kick = EnergyAnalysis(
        None, None, 1.0, 1.0, transitions=[(1.0, pendulum.EXIT), (2.0, pendulum.CAPTURE)], mechanism=SeparatrixMechanism.KICK
    )
    assert not _mark(energy=kick)
    assert not _mark(angle_status=AngleStatus.LIBRATION)
    assert not _mark(spec=ResonanceSpec.from_key('1J-1+0+0'))  # co-orbital
    assert not _mark(checks={'check_mmr_a_response': CheckValue.NOT_MEASURED, 'check_mmr_potential': CheckValue.FAIL})
    for blocker in ('filter_sensitive', 'filter_too_slow', 'short_period_aliased'):
        assert not _mark(flags=[blocker])
    assert not _mark(caps=['episode_before_jump'])
    # the mark lifts transient_unexplained only (author:2026-10-18), and only when the convention is on
    lift = decide.transient_unexplained_after_exit_and_capture
    caps = ['three_tp_episode', 'transient_unexplained']
    assert lift(caps, True, AnalysisSettings()) == ['three_tp_episode']
    assert lift(caps, False, AnalysisSettings()) == caps
    assert lift(caps, True, AnalysisSettings(mmr_exit_and_capture_lifts_unexplained=False)) == caps


def test_33128_exits_and_captures_as_in_the_q12_report():
    # W17 Q12: the slip over the top at 69.3 kyr is an exit and a capture, 89.2 kyr an exit; the mark is set and lifts
    # the cap transient_unexplained (author:2026-10-18: "поднять до high? Да")
    analysis = _analyse('mmr2__33128__6N-7+0+1')
    transitions = analysis.energy.transitions
    assert [kind for _, kind in transitions] == [pendulum.EXIT, pendulum.CAPTURE, pendulum.EXIT]
    assert [round(m / 1e3, 1) for m, _ in transitions] == [69.3, 69.3, 89.2]
    assert str(Flag.EXIT_AND_CAPTURE) in analysis.verdict.flags
    assert str(conf.Cap.TRANSIENT_UNEXPLAINED) not in analysis.verdict.caps
    assert (str(analysis.verdict.status), str(analysis.verdict.confidence)) == ('transient', 'high')
    assert analysis.extra_columns['sep_unexplained_lifted'] == 1.0


def test_33128_keeps_the_cap_with_the_convention_off():
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path('mmr2__33128__6N-7+0+1').exists():
        pytest.skip('benchmark data not available')
    off = settings_from_config(None, mmr_exit_and_capture_lifts_unexplained=False)
    analysis = analyse_mmr(io.load_series('mmr2__33128__6N-7+0+1'), off)
    assert str(Flag.EXIT_AND_CAPTURE) in analysis.verdict.flags  # the mark itself does not depend on the convention
    assert analysis.verdict.caps == [str(conf.Cap.TRANSIENT_UNEXPLAINED)]
    assert (str(analysis.verdict.status), str(analysis.verdict.confidence)) == ('transient', 'medium')
    assert analysis.extra_columns['sep_unexplained_lifted'] == 0.0


# --- the rename ---------------------------------------------------------------------------------------------------


def test_old_summary_files_with_separatrix_chaos_still_read():
    row = Verdict(ResonanceType.MMR, Status.TRANSIENT, AngleStatus.EPISODIC, confidence=Confidence.MEDIUM).to_flat_dict()
    row.update(flags='separatrix_chaos;zone_eq_only', caps='separatrix_chaos;three_tp_episode', angle_status='transient')
    verdict = Verdict.from_flat_dict(row)
    assert verdict.flags == ['episodes_in_separatrix_zone', 'zone_eq_only']
    assert verdict.caps == ['episodes_in_separatrix_zone', 'three_tp_episode']
    assert conf.apply_caps(Confidence.HIGH, verdict.caps) == Confidence.MEDIUM
    assert str(Flag.EPISODES_IN_SEPARATRIX_ZONE) == pendulum.EPISODES_IN_SEPARATRIX_ZONE == 'episodes_in_separatrix_zone'
