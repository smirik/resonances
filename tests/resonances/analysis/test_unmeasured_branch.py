"""W28 B9 (card g08-6 A): in the branch where L4 and R0 are both not measured (`decide.unmeasured_branch`) an
order-of-magnitude strength (eq_scale, required / mu_i mu_j e^q) outside [1/x, x] is no measured objection: leaf 5x,
uncertain with origin unknown ('origin not established'), before 5n and the geometry's leaf 5u/6, so it never becomes
the geometry's resonant/transient. A failed strength computed by direct averaging stays the objection of leaf 5o.

The ratios are the benchmark rows' (79653 3M+3J-7: 39.5 x the e^q scale; x = MMR_STRENGTH_CHECK_RATIO = 10, so the
scale fails above 10 and below 0.1; 1000 x is rule 5s), not values the package computes here.
"""

from types import SimpleNamespace

import pytest

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.decide import AngleDecision
from resonances.resonance.analysis.mmr.models import Strength
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, LibrationOrigin, Reason, Status
from tests.resonances.analysis.builders import make, make_episode, make_geometry

S = settings_from_config()
EPISODIC = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
LIBRATION = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', 0)


def test_the_ratios_bracket_the_default_thresholds():
    assert (S.mmr_strength_check_ratio, S.mmr_strength_implausible_ratio, S.mmr_o1_ratio_fail) == (10.0, 1000.0, 0.5)


def _strength(ratio, check, method=st.METHOD_SCALE, **fields):
    three_body = method == st.METHOD_SCALE
    return make(Strength, ratio=ratio, check=check, method=method, eq_legal=True, three_body=three_body, **fields)


def _branch(strength, angle=EPISODIC, centre_estimates=2, before=0.1):
    ledger = Ledger.from_settings(S)
    geometry = make_geometry(runs=[make_episode(episode_id=0, centre_estimates=centre_estimates)])
    physics = [SimpleNamespace(episode_id=0, a_band_std_before_au=before, a_band_std_in_au=1.0)]
    return decide.unmeasured_branch(angle, geometry, None, strength, physics, S, ledger), ledger


@pytest.mark.parametrize('ratio', [39.47, 0.05])
@pytest.mark.parametrize('angle', [EPISODIC, LIBRATION])
@pytest.mark.parametrize('estimates', [1, 2])
def test_an_eq_scale_out_of_range_is_leaf_5x_never_the_geometry(ratio, angle, estimates):
    decision, ledger = _branch(_strength(ratio, CheckValue.FAIL), angle, estimates)
    assert (decision.status, decision.reason, decision.origin, decision.rule) == (
        Status.UNCERTAIN,
        Reason.ORIGIN_UNKNOWN,
        LibrationOrigin.UNKNOWN,
        '5x',
    )
    assert Rule.EQ_SCALE_NOT_OBJECTION in decision.rules_fired and not decision.caps
    labelled = [e for e in ledger if e.name.startswith(decide.SCALE_OUT_OF_RANGE)]
    assert len(labelled) == 2 and all(e.value == ratio for e in labelled)
    assert not any(e.name.startswith('5o F') for e in ledger)  # not called a measured objection


def test_the_same_scale_inside_the_range_still_lets_the_geometry_decide():
    decision, _ = _branch(_strength(3.0, CheckValue.PASS), LIBRATION)
    assert (decision.status, decision.rule, decision.caps) == (Status.RESONANT, '5u/6', ['strength_scale_only'])
    assert Rule.EQ_SCALE_NOT_OBJECTION not in decision.rules_fired


def test_a_failed_direct_averaging_is_still_a_measured_objection():
    # 50 x the directly averaged strength (below the 1000 x of 5s): F fails as a measurement, leaf 5o
    decision, ledger = _branch(_strength(50.0, CheckValue.FAIL, method=st.METHOD_DIRECT))
    assert (decision.status, decision.rule) == (Status.UNCERTAIN, '5o')
    assert any(e.name.startswith('5o F') and e.result is False for e in ledger)
    assert not decision.rules_fired


def test_an_onset_objection_decides_before_5x():
    # O1 = 0.6 >= 0.5 fails: 5o, whatever the scale says (1865, 1916, 1917 ... keep 5o through their O1)
    decision, ledger = _branch(_strength(39.47, CheckValue.FAIL), before=0.6)
    assert decision.rule == '5o'
    assert not any(e.name.startswith(decide.SCALE_OUT_OF_RANGE) for e in ledger)


def test_an_implausible_three_body_scale_keeps_rule_5s():
    # 2e4 x the e^q scale, the small-divisor residual 3e9: rule 5s before 5x
    strength = _strength(2.0e4, CheckValue.FAIL, divisor_kind=st.DIVISOR_NEAREST, divisor_residual=2.97e9)
    decision, _ = _branch(strength)
    assert (decision.status, decision.reason, decision.rule) == (Status.NON_RESONANT, Reason.IMPLAUSIBLE_STRENGTH, '5s')
