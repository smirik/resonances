"""The G2c guard (LIBRATION_CENTRE_SCALE_OBJECTION_GUARD, rule centre_scale_displaced): a reading of the angle at the
scale of its centre replaces the fine reading only when it is at least as well established as the resonance's own
libration.

The expectations come from docs/physics/status-tree.md ("Two kinds of uncertain / origin_unknown leaves": a
measurement contradicts the angle's own dynamics in 5a, 5b, 5e, 5g, 5o, and 5f's forced line, which the minimal
policy counts as an objection; nothing was measured in 5n, 5x, 5u) and from the two independent readings of 709123
2J-1+0-1 (2012 UO134) by Claude and Codex on 2026-11-20 (the author: resonant/high): the 2:1 libration proper has a
period of about 460-490 yr (Codex's fast band: median cycle 472 yr, 5-95 % 363-673 yr) and a half-amplitude of about
14 deg; the larger, slow motion of its centre is forced at g - g5 (3.5-3.7 kyr) with its harmonic near 1.9 kyr.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis import policy, rules
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import analyse as analyse_module, decide, io
from resonances.resonance.analysis.mmr.decide import StatusDecision
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue, LibrationOrigin, Reason, Status
from tests.resonances.analysis.builders import make_mmr_analysis, make_verdict, tree_and_final

S = settings_from_config()
OFF = replace(S, libration_centre_scale_objection_guard=False)
EXAMPLE = 'mmr2__709123__2J-1+0-1'

# status-tree.md: the leaves where a measurement contradicts the angle's own dynamics (5f: the forced line, an
# objection of the minimal policy), and those where nothing was measured or the leaf is no objection at all
CONTRADICTED = ('5a', '5a*', '5b', '5e', '5f', '5g', '5o')
NOT_CONTRADICTED = ('5n', '5x', '5u/6', '5u/6u', '5s', '5s*', '5d/6', '5d/6u', '5c/6', '4', 'G6', '3', '2', '')


def _fine(origin):
    """The fine pass's decision as `decide_status` returns it (a real StatusDecision)."""
    status = Status.RESONANT if origin == LibrationOrigin.DYNAMIC else Status.UNCERTAIN
    reason = Reason.DYNAMIC_LIBRATION if origin == LibrationOrigin.DYNAMIC else Reason.ORIGIN_UNKNOWN
    return StatusDecision(status, reason, origin, '', '', rule='5d/6u' if origin == LibrationOrigin.DYNAMIC else '5g')


def test_the_guard_is_on_by_default():
    assert S.libration_centre_scale_objection_guard is True


def test_one_set_of_objection_leaves_for_the_guard_and_the_policy():
    """The guard and the minimal policy read the same leaves as a measured objection (decide.OBJECTION_LEAVES); every
    one but 5o names its failed checks in policy.LEAF_OBJECTIONS (5o reads its own)."""
    assert set(decide.OBJECTION_LEAVES) == set(CONTRADICTED)
    assert set(policy.LEAF_OBJECTIONS) | {'5o'} == set(decide.OBJECTION_LEAVES)
    assert {leaf for leaf, kind in policy.LEAVES.items() if kind == policy.OBJECTION} == set(CONTRADICTED)


@pytest.mark.parametrize('leaf', CONTRADICTED)
def test_a_contradicted_coarse_reading_does_not_displace_a_dynamic_fine_one(leaf):
    assert decide.objection_leaf(leaf)
    assert decide.centre_scale_displaced(_fine(LibrationOrigin.DYNAMIC), leaf, LibrationOrigin.UNKNOWN)


@pytest.mark.parametrize('leaf', NOT_CONTRADICTED)
def test_an_unmeasured_or_established_coarse_reading_keeps_its_precedence(leaf):
    """5n/5x/5u: not_measured is no contradiction; 5d/5c and 3: a measured origin (dynamic, kinematic) is out of scope."""
    assert not decide.objection_leaf(leaf)
    assert not decide.centre_scale_displaced(_fine(LibrationOrigin.DYNAMIC), leaf, LibrationOrigin.UNKNOWN)


@pytest.mark.parametrize(
    'fine, coarse_origin',
    [
        (LibrationOrigin.UNKNOWN, LibrationOrigin.UNKNOWN),  # the fine reading is no better established (2003 EH16, TX9)
        (LibrationOrigin.KINEMATIC, LibrationOrigin.UNKNOWN),
        (LibrationOrigin.INDUCED, LibrationOrigin.UNKNOWN),
        (LibrationOrigin.SUBRESONANCE, LibrationOrigin.UNKNOWN),
        (LibrationOrigin.DYNAMIC, LibrationOrigin.DYNAMIC),  # a coarse measurement (the Neptune G2c rows)
        (LibrationOrigin.DYNAMIC, LibrationOrigin.KINEMATIC),  # 481085 7E-10M-6
        (LibrationOrigin.DYNAMIC, LibrationOrigin.NONE),  # a near_separatrix 5g reads non_resonant, origin none
    ],
)
def test_only_a_dynamic_fine_origin_against_an_unknown_coarse_one_displaces(fine, coarse_origin):
    assert not decide.centre_scale_displaced(_fine(fine), '5g', coarse_origin)


def _coarse(rule='5g', origin=LibrationOrigin.UNKNOWN, cutoff=1374.0, **checks):
    """A G2c analysis as `_at_centre_scale` returns it: the tree's verdict at W_c (raw v2: uncertain/origin_unknown on an
    objection leaf), its leaf and the G2c columns."""
    verdict = make_verdict(
        Status.UNCERTAIN, AngleStatus.LIBRATION, reason=Reason.ORIGIN_UNKNOWN, libration_origin=origin, checks=dict(checks)
    )
    return make_mmr_analysis(verdict, rec_rule=rule, extra={'geo_centre_scale_cutoff_yr': cutoff})


def test_only_a_contradicted_g2c_analysis_is_held():
    coarse = _coarse()
    assert analyse_module._held_centre_scale(coarse, S).analysis is coarse
    assert analyse_module._held_centre_scale(coarse, OFF) is None  # the switch off: G2c always takes precedence
    assert analyse_module._held_centre_scale(None, S) is None
    assert analyse_module._held_centre_scale(_coarse(rule='5n'), S) is None  # nothing measured
    assert analyse_module._held_centre_scale(_coarse(origin=LibrationOrigin.DYNAMIC, rule='5d/6'), S) is None
    assert analyse_module._held_centre_scale(_coarse(cutoff=math.nan), S) is None  # G1q / G1f: no W_c, not judged


def test_a_dynamic_fine_decision_keeps_the_fine_pass_and_reports_the_displaced_reading():
    coarse = _coarse(check_mmr_identity=CheckValue.FAIL, check_mmr_strength=CheckValue.FAIL, check_mmr_potential=CheckValue.PASS)
    held = analyse_module._held_centre_scale(coarse, S)
    decision, ledger = _fine(LibrationOrigin.DYNAMIC), Ledger.from_settings(S)
    kept, columns = analyse_module._centre_scale_guard(held, decision, ledger, 75.0)
    assert kept is None
    assert str(Rule.CENTRE_SCALE_DISPLACED) in {str(rule) for rule in decision.rules_fired}
    assert columns['geo_cs_displaced_cutoff_yr'] == 1374.0 and columns['geo_cs_displaced_leaf'] == '5g'
    assert (columns['geo_cs_displaced_status'], columns['geo_cs_displaced_origin']) == ('uncertain', 'unknown')
    assert columns['geo_cs_displaced_failed'] == 'check_mmr_identity;check_mmr_strength'  # the objection stays with it
    assert set(columns) == set(analyse_module.CS_DISPLACED_NONE)
    (entry,) = list(ledger)
    assert entry.name == analyse_module.CENTRE_SCALE_DISPLACED and entry.column == 'geo_cs_displaced_cutoff_yr'


def test_a_fine_decision_that_is_not_dynamic_returns_the_held_analysis_as_g2c_kept_it():
    """The held analysis comes back unchanged and the G2a cuts the fine pass made after it was held are dropped: without
    the guard the fine pass would have stopped at G2c."""
    coarse = _coarse()
    token = analyse_module._ACTION_STEP_CUTS.set(['cut at G2c'])
    try:
        held = analyse_module._held_centre_scale(coarse, S)
        analyse_module._ACTION_STEP_CUTS.get().append('cut of the fine pass')
        decision, ledger = _fine(LibrationOrigin.UNKNOWN), Ledger.from_settings(S)
        kept, columns = analyse_module._centre_scale_guard(held, decision, ledger, 75.0)
        assert kept is coarse and columns == analyse_module.CS_DISPLACED_NONE
        assert analyse_module._ACTION_STEP_CUTS.get() == ['cut at G2c']
    finally:
        analyse_module._ACTION_STEP_CUTS.reset(token)
    assert not decision.rules_fired and len(ledger) == 0


def test_no_held_analysis_writes_empty_columns():
    kept, columns = analyse_module._centre_scale_guard(None, _fine(LibrationOrigin.DYNAMIC), Ledger.from_settings(S), 75.0)
    assert kept is None and columns == analyse_module.CS_DISPLACED_NONE


def test_the_rule_is_registered_under_g2c():
    info = rules.RULES[Rule.CENTRE_SCALE_DISPLACED]
    assert info.doc_id == 'G2c' and info.doc_id != rules.PENDING


def test_a_record_stopped_before_the_geometry_has_empty_displaced_columns():
    t = np.arange(0.0, 1000.0, 1.0)
    one = np.full_like(t, 0.1)
    series = MMRSeries(
        t_yr=t,
        a=np.full_like(t, -1.0),
        e=one,
        inc=one,
        Omega=one,
        omega=one,
        varpi=2 * one,
        angle_wrapped=one,
        resonance=ResonanceSpec.from_key('3N-4+0+1'),
        regime=Regime(step_yr=1.0, span_yr=999.0, n_rows=len(t)),
        name='invalid',
    )
    columns = analyse_module.analyse_mmr(series, S).numbers_row()
    assert set(analyse_module.CS_DISPLACED_NONE) <= set(columns)
    assert columns['geo_cs_displaced_leaf'] == '' and np.isnan(columns['geo_cs_displaced_cutoff_yr'])


# --------------------------------------------------------------------------------------------- 709123 (benchmark)


def _load():
    if not io.example_path(EXAMPLE).exists():
        pytest.skip('benchmark data not available')
    return io.load_series(EXAMPLE)


@pytest.mark.benchmark
def test_709123_reads_the_2_1_libration_not_the_wave_of_its_centre():
    """The fine 2J-1 libration (a union of episodes, origin dynamic) stays; the g - g5 wave of the centre at W_c, where
    the identity fails (leaf 5g), is reported as the slow scale and in geo_cs_displaced_*."""
    tree, final = tree_and_final(_load(), S)
    verdict = final.verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.confidence)) == ('resonant', 'libration', 'high')
    assert str(verdict.libration_origin) == 'dynamic' and str(Rule.CENTRE_SCALE_DISPLACED) in verdict.rules_fired
    assert 360.0 <= tree.geometry.period_cycle_yr <= 673.0  # Codex's fast band, 5-95 % of the cycle periods
    assert tree.prep.cutoff.binding != 'centre_scale'
    columns = tree.numbers_row()
    assert columns['geo_cs_displaced_leaf'] == '5g' and 'check_mmr_identity' in columns['geo_cs_displaced_failed']
    assert columns['geo_cs_displaced_cutoff_yr'] == pytest.approx(3 * tree.geometry.period_cycle_yr, rel=0.01)  # W_c = 3 P_cyc
    slow = tree.summary_row()['slow_period_yr']
    assert 1.8e3 <= slow <= 3.8e3  # between the 2(g - g5) harmonic and the g - g5 period of the centre's wave


@pytest.mark.benchmark
def test_709123_without_the_guard_is_read_at_the_scale_of_its_centre():
    tree, final = tree_and_final(_load(), OFF)
    assert (str(final.verdict.status), str(final.verdict.confidence)) == ('resonant', 'low')
    assert tree.extra_columns['rec_rule'] == '5g' and tree.prep.cutoff.binding == 'centre_scale'
    assert tree.geometry.period_cycle_yr > 1.8e3  # the wave of the centre, not the 2:1 libration
    assert tree.extra_columns['geo_cs_displaced_leaf'] == ''
