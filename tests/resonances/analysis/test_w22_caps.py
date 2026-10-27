"""W22 R-CAPS (author 2026-10-24): three caps of the MMR confidence.

- R-F: leaf 5u/6 whose strength on the order-of-magnitude scale (required / mu_i mu_j e^q) lies within [1/x, x]:
  cap strength_scale_only (<= medium) instead of geometry_only (<= low).
- R-WJ: a 1:1 verdict on one cycle (three_tp_episode) whose co-orbital energy integral, read from rule R's response,
  has R^2 < mmr_force_r2_fail: cap one_cycle_energy_not_conserved (<= medium).
- R-O1-OWNER: an induced libration (owner_libration) whose owner's libration intervals cover at least
  mmr_o1_owner_cover_min of O1's background window: O1 not_applicable, so no cap onset_failed.

Expected values come from the constructions (hand-computed fractions and ratios), from the R-CAPS measurement
(SP/dis/impact/R-CAPS: F of 2348 1.59, R^2 of 2024 WJ 0.043 at the witness and 0.824 pooled, GK3's intervals) and, for
the benchmark records, from the author's labels of 2026-10-24 (all three medium), not from the package.
"""

from types import SimpleNamespace

import pytest

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.decide import AngleDecision, StatusDecision
from resonances.resonance.analysis.mmr.models import Strength
from resonances.resonance.analysis.mmr.owner import INTERVAL_FORCED, INTERVAL_LIBRATION
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.builders import make_episode, make_geometry
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    LibrationOrigin,
    OriginDetail,
    Reason,
    Status,
)

S = settings_from_config()
LIBRATION = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', 0)
EPISODIC = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
SCALE_ONLY, GEOMETRY_ONLY = str(conf.Cap.STRENGTH_SCALE_ONLY), str(conf.Cap.GEOMETRY_ONLY)
ONE_CYCLE = str(conf.Cap.ONE_CYCLE_ENERGY_NOT_CONSERVED)


def test_the_new_caps_hold_the_level_at_medium():
    for cap in (SCALE_ONLY, ONE_CYCLE):
        assert conf.apply_caps(Confidence.CERTAIN, [cap]) == Confidence.MEDIUM
        assert conf.apply_caps(Confidence.LOW, [cap]) == Confidence.LOW
    assert conf.apply_caps(Confidence.HIGH, [GEOMETRY_ONLY]) == Confidence.LOW  # unchanged


# --- R-F: leaf 5u/6 --------------------------------------------------------------------------------------------------


def _scale_strength(ratio, check):
    return Strength(
        mu_scale=1e-9,
        mu_scale_no_eq=1e-6,
        eq_legal=False,
        mu_eff_required=ratio * 1e-9,
        ratio=ratio,
        omega0_theory_rad_yr=float('nan'),
        period_theory_yr=float('nan'),
        implausible=False,
        check=check,
        method=st.METHOD_SCALE,
        three_body=True,
    )


def _branch(strength):
    ledger = Ledger.from_settings(S)
    geometry = SimpleNamespace(runs=[SimpleNamespace(episode_id=0, centre_estimates=S.libration_centre_min_estimates)])
    physics = [SimpleNamespace(episode_id=0, a_band_std_before_au=0.1, a_band_std_in_au=1.0)]
    return decide.unmeasured_branch(LIBRATION, geometry, None, strength, physics, S, ledger), ledger


@pytest.mark.parametrize('ratio', [1.59, 1.32, 1.88, 0.2, 9.0])  # 2348, 68436, 48789 (R-CAPS); both ends of [1/10, 10]
def test_a_passed_strength_scale_caps_the_geometry_leaf_at_medium(ratio):
    assert 1 / S.mmr_strength_check_ratio <= ratio <= S.mmr_strength_check_ratio
    decision, ledger = _branch(_scale_strength(ratio, CheckValue.PASS))
    assert (decision.status, decision.origin, decision.rule, decision.caps) == (
        Status.RESONANT,
        LibrationOrigin.UNKNOWN,
        '5u/6',
        [SCALE_ONLY],
    )
    entries = [e for e in ledger if e.name.startswith('5u/6 F on the order-of-magnitude scale')]
    assert len(entries) == 2 and all(e.result and e.value == ratio for e in entries)
    # every margin full: the cap alone sets the level (geometry_only would give low)
    assert conf.apply_caps(Confidence.CERTAIN, decision.caps) == Confidence.MEDIUM


@pytest.mark.parametrize('check', [CheckValue.NOT_MEASURED, CheckValue.NOT_APPLICABLE])
def test_without_a_passed_scale_the_geometry_leaf_stays_at_low(check):
    decision, _ = _branch(_scale_strength(float('nan'), check))
    assert (decision.rule, decision.caps) == ('5u/6', [GEOMETRY_ONLY])
    decision, _ = _branch(None)
    assert (decision.rule, decision.caps) == ('5u/6', [GEOMETRY_ONLY])


def test_a_failed_scale_is_an_objection_not_the_geometry_leaf():
    decision, _ = _branch(_scale_strength(64.0, CheckValue.FAIL))  # 10 < 64 < 1000: rule 5o
    assert (decision.status, decision.rule, decision.caps) == (Status.UNCERTAIN, '5o', [])


# --- R-WJ: a 1:1 on one cycle ----------------------------------------------------------------------------------------


def _response(energy_r2):
    return SimpleNamespace(coorbital=SimpleNamespace(energy_r2=energy_r2))


def _caps(n_tp, energy_r2, one_to_one=True):
    geometry = make_geometry(runs=[make_episode(n_tp, episode_id=0, closed_by_return=False)], exit=None, entry=None)
    decision = StatusDecision(Status.TRANSIENT, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC, '', '')
    return decide.collect_caps(
        decide.tree_view(EPISODIC, None, None), decision, geometry, {}, S, None, True, one_to_one=one_to_one, response=_response(energy_r2)
    )


@pytest.mark.parametrize(
    'energy_r2, capped',
    [
        (0.043, True),  # 2024 WJ, rule R's witness episode
        (0.386, True),  # 2019 YB4: just below 0.4
        (0.0002, True),  # 2025 NL
        (0.824, False),  # 2024 WJ's pooled law: not what the tree read
        (S.mmr_force_r2_fail, False),  # at the threshold the integral does not fail
        (float('nan'), False),  # not measured is no failure
    ],
)
def test_one_cycle_without_a_conserved_energy_caps_a_coorbital_at_medium(energy_r2, capped):
    caps = _caps(S.libration_min_turning_points, energy_r2)
    assert str(conf.Cap.THREE_TP_EPISODE) in caps
    assert (ONE_CYCLE in caps) == capped
    assert conf.apply_caps(Confidence.HIGH, caps) == (Confidence.MEDIUM if capped else Confidence.HIGH)


def test_the_energy_cap_needs_one_cycle_and_a_coorbital():
    assert ONE_CYCLE not in _caps(S.libration_min_turning_points + 2, 0.043)  # two cycles: no three_tp_episode
    assert ONE_CYCLE not in _caps(S.libration_min_turning_points, 0.043, one_to_one=False)


# --- R-O1-OWNER --------------------------------------------------------------------------------------------------------


def _owner(*intervals):
    return SimpleNamespace(intervals=list(intervals))


def _induced(detail=OriginDetail.OWNER_LIBRATION):
    return StatusDecision(Status.NON_RESONANT, Reason.INDUCED_LIBRATION, LibrationOrigin.INDUCED, detail, '47M-44')


GK3 = _owner((455.0, 11636.0, INTERVAL_LIBRATION), (11636.0, 12407.0, INTERVAL_LIBRATION))  # R-CAPS log


@pytest.mark.parametrize(
    'owner, window, cover',
    [
        (GK3, (11068.0, 11718.0), 1.0),  # 1997 GK3 3M+5S-3: the window lies inside the libration
        # libration up to 11300 yr, then a forced wave: (11300 - 11068) / 650 = 0.357
        (_owner((0.0, 11300.0, INTERVAL_LIBRATION), (11300.0, 11718.0, INTERVAL_FORCED)), (11068.0, 11718.0), 232.0 / 650.0),
        (_owner((1000.0, 1500.0, INTERVAL_LIBRATION)), (1000.0, 2000.0), 0.5),
        # two overlapping pieces count once: [1200, 1800] of [1000, 2000]
        (_owner((1200.0, 1600.0, INTERVAL_LIBRATION), (1400.0, 1800.0, INTERVAL_LIBRATION)), (1000.0, 2000.0), 0.6),
    ],
)
def test_owner_background_cover(owner, window, cover):
    assert decide.owner_background_cover(owner, window) == pytest.approx(cover)
    physics = SimpleNamespace(onset_before_t0_yr=window[0], onset_before_t1_yr=window[1])
    ledger = Ledger.from_settings(S)
    applies = decide.owner_librates_before(_induced(), owner, physics, S, ledger)
    assert applies == (cover >= S.mmr_o1_owner_cover_min)
    assert ledger.entries[-1].name == decide.O1_OWNER_COVER and ledger.entries[-1].threshold == S.mmr_o1_owner_cover_min


def test_o1_still_applies_without_an_owner_libration():
    physics = SimpleNamespace(onset_before_t0_yr=11068.0, onset_before_t1_yr=11718.0)
    ledger = Ledger.from_settings(S)
    assert not decide.owner_librates_before(_induced(OriginDetail.OWNER_FORCED_WAVE), GK3, physics, S, ledger)
    assert not decide.owner_librates_before(_induced(), None, physics, S, ledger)
    assert not decide.owner_librates_before(_induced(), _owner((455.0, 12407.0, INTERVAL_FORCED)), physics, S, ledger)
    nan = SimpleNamespace(onset_before_t0_yr=float('nan'), onset_before_t1_yr=float('nan'))
    assert not decide.owner_librates_before(_induced(), GK3, nan, S, ledger)
    # a stricter convention than GK3's cover keeps O1 (the fraction is a setting)
    strict = settings_from_config(mmr_o1_owner_cover_min=1.0)
    half = _owner((11068.0, 11393.0, INTERVAL_LIBRATION))
    assert not decide.owner_librates_before(_induced(), half, physics, strict, Ledger.from_settings(strict))
    assert decide.owner_librates_before(_induced(), half, physics, S, Ledger.from_settings(S))


# --- the benchmark records the author labelled (skipped without the benchmark data) -----------------------------------


@pytest.mark.parametrize(
    'example, cap',
    [
        ('mmr3__24820__3J-1S-1+0+0-1', SCALE_ONLY),  # R-F, F 1.67
        ('coorbital__2024_WJ__1E-1+0+0', ONE_CYCLE),  # R-WJ
        ('mmr3__1997_GK3__3M+5S-3+0+0-5', None),  # R-O1-OWNER: no onset_failed
    ],
)
def test_benchmark_records_reach_the_authors_medium(example, cap):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    verdict = analyse_mmr(io.load_series(example), S).verdict
    assert str(verdict.confidence) == 'medium'  # the author's label of 2026-10-24
    if cap is not None:
        assert cap in verdict.caps and GEOMETRY_ONLY not in verdict.caps
    else:
        assert str(verdict.origin_detail) == 'owner_libration'
        assert str(conf.Cap.ONSET_FAILED) not in verdict.caps
        assert str(verdict.margins['O1'].margin) == 'not_applicable'
