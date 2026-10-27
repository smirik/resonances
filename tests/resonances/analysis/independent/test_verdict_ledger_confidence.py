"""Independent tests of analysis/verdict.py, analysis/ledger.py and analysis/confidence.py.

Written from the public API (impl/API.md), the docstrings, SPEC §2.2, §5.6, §5.7, §7 and the delta, without reading
the implementation. Every expected value below comes from a table of the specification or from a hand calculation
shown in a comment.
"""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis.ledger import Ledger, LedgerEntry, Margin, Op
from resonances.resonance.analysis.settings import conventions, settings_from_config
from resonances.resonance.analysis.verdict import (
    ALL_CHECKS,
    MMR_CHECKS,
    SECULAR_CHECKS,
    ZLK_CHECKS,
    Action,
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    MarginClass,
    MarginValue,
    OriginDetail,
    Reason,
    ResonanceType,
    Status,
    Verdict,
    best_status,
    default_action,
    join_list,
    lower_confidence,
    split_list,
    status_code,
    verdict_from_status_code,
)

PHYSICS = ('L4', 'Z', 'R0')  # SPEC §7.2: the decisive physics margins of an MMR verdict


def mv(margin, value=None, cls=MarginClass.DECISIVE):
    return MarginValue(margin=margin, value=value, margin_class=cls)


def no_cells():
    return conf.ConfidenceCells()


# ---------------------------------------------------------------------------------------------------------------------
# verdict.status_code: the MMR table of SPEC §2.2 (every row)
# ---------------------------------------------------------------------------------------------------------------------

MMR_CODE_TABLE = [
    (Status.RESONANT, AngleStatus.LIBRATION, Reason.DYNAMIC_LIBRATION, 2),
    (Status.TRANSIENT, AngleStatus.EPISODIC, Reason.DYNAMIC_EPISODE, 1),
    (Status.NEAR_SEPARATRIX, AngleStatus.NEAR_SEPARATRIX, Reason.SEPARATRIX_ZONE, -3),
    (Status.NON_RESONANT, AngleStatus.CIRCULATION, Reason.CIRCULATION, 0),
    (Status.NON_RESONANT, AngleStatus.LIBRATION, Reason.KINEMATIC_LIBRATION, 0),
    (Status.NON_RESONANT, AngleStatus.DRIFT, Reason.KINEMATIC_DRIFT, 0),
    (Status.NON_RESONANT, AngleStatus.LIBRATION, Reason.INDUCED_LIBRATION, 0),
    (Status.NON_RESONANT, AngleStatus.LIBRATION, Reason.SUBRESONANCE_LIBRATES, 0),
    (Status.NON_RESONANT, AngleStatus.CIRCULATION, Reason.ALIAS, 0),
    (Status.NON_RESONANT, AngleStatus.CIRCULATION, Reason.MODULATED_CIRCULATION, 0),
    (Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, Reason.BASELINE_NEEDED, -6),
    (Status.UNCERTAIN, AngleStatus.DRIFT, Reason.BASELINE_NEEDED, -6),
    (Status.UNCERTAIN, AngleStatus.LIBRATION, Reason.ORIGIN_UNKNOWN, -9),
    (Status.UNCERTAIN, AngleStatus.DRIFT, Reason.DRIFT_MECHANISM_UNKNOWN, -9),
    (Status.INVALID, AngleStatus.UNDEFINED, Reason.ALIAS, -99),
    (Status.INVALID, AngleStatus.UNDEFINED, Reason.UNPHYSICAL, -99),
    (Status.INVALID, AngleStatus.UNDEFINED, Reason.UNDEFINED_PHASE, -99),
]


@pytest.mark.parametrize('status,angle_status,reason,expected', MMR_CODE_TABLE)
def test_mmr_status_code_table(status, angle_status, reason, expected):
    assert status_code(ResonanceType.MMR, status, angle_status, reason) == expected
    assert status_code('mmr', status, angle_status, reason) == expected


def test_mmr_kinematic_is_zero_not_minus_two():
    # SPEC §2.2: -2 stays only for ZLK; an MMR kinematic libration is 0, like the old non_resonant rows.
    code = status_code(ResonanceType.MMR, Status.NON_RESONANT, AngleStatus.LIBRATION, Reason.KINEMATIC_LIBRATION)
    assert code == 0 and code != -2


def test_verdict_status_code_property_matches_function():
    v = Verdict(
        resonance_type=ResonanceType.MMR,
        status=Status.UNCERTAIN,
        angle_status=AngleStatus.INSUFFICIENT_TIME,
        angle_subtype=AngleSubtype.ARC,
        reason=Reason.BASELINE_NEEDED,
    )
    assert v.status_code == -6


# ---------------------------------------------------------------------------------------------------------------------
# verdict_from_status_code: the secular / ZLK table of SPEC §5.7
# ---------------------------------------------------------------------------------------------------------------------

LEGACY_CODES = [2, 1, 0, -2, -3, -4, -5, -6, -9, -99]

# code -> (status, angle_status, reason or None when type-dependent)
SECTION_57 = {
    2: (Status.RESONANT, AngleStatus.LIBRATION, Reason.DYNAMIC_LIBRATION),
    1: (Status.TRANSIENT, AngleStatus.EPISODIC, Reason.DYNAMIC_EPISODE),
    0: (Status.NON_RESONANT, AngleStatus.CIRCULATION, Reason.CIRCULATION),
    -2: (Status.NON_RESONANT, AngleStatus.LIBRATION, Reason.KINEMATIC_LIBRATION),
    -3: (Status.NEAR_SEPARATRIX, AngleStatus.NEAR_SEPARATRIX, Reason.SEPARATRIX_ZONE),
    -4: (Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, Reason.BASELINE_NEEDED),
    -5: (Status.NEAR_SEPARATRIX, AngleStatus.NEAR_SEPARATRIX, Reason.SEPARATRIX_ZONE),
    -6: (Status.UNCERTAIN, AngleStatus.DRIFT, Reason.BASELINE_NEEDED),
    -9: (Status.UNCERTAIN, AngleStatus.UNDEFINED, None),
    -99: (Status.INVALID, AngleStatus.UNDEFINED, Reason.UNPHYSICAL),
}


@pytest.mark.parametrize('rtype', [ResonanceType.SECULAR, ResonanceType.LIDOV_KOZAI])
@pytest.mark.parametrize('code', LEGACY_CODES)
def test_legacy_code_translation_table(rtype, code):
    v = verdict_from_status_code(code, rtype, confidence=Confidence.MEDIUM)
    status, angle_status, reason = SECTION_57[code]
    assert v.resonance_type == rtype
    assert v.status == status
    assert v.angle_status == angle_status
    if reason is None:
        expected = Reason.GATE_UNAVAILABLE if rtype == ResonanceType.LIDOV_KOZAI else Reason.UNUSABLE_SERIES
        assert v.reason == expected
    else:
        assert v.reason == reason


@pytest.mark.parametrize('rtype', [ResonanceType.SECULAR, ResonanceType.LIDOV_KOZAI, 'secular', 'lidov_kozai'])
@pytest.mark.parametrize('code', LEGACY_CODES)
def test_legacy_code_is_exact_inverse(rtype, code):
    # API §2.4: verdict_from_status_code and status_code are inverses on {2, 1, 0, -2, -3, -4, -5, -6, -9, -99}.
    v = verdict_from_status_code(code, rtype, confidence=Confidence.HIGH)
    assert v.status_code == code
    assert status_code(rtype, v.status, v.angle_status, v.reason, v.flags) == code


def test_legacy_minus_two_is_kinematic_origin():
    v = verdict_from_status_code(-2, ResonanceType.LIDOV_KOZAI, confidence=Confidence.HIGH)
    assert v.libration_origin == LibrationOrigin.KINEMATIC


def test_legacy_minus_four_is_arc():
    v = verdict_from_status_code(-4, ResonanceType.SECULAR, confidence=Confidence.HIGH)
    assert v.angle_subtype == AngleSubtype.ARC


def test_legacy_minus_five_low_and_flag_minus_three_not():
    # -5 and -3 share status/angle/reason; only the flag (and level low) keeps them apart.
    v5 = verdict_from_status_code(-5, ResonanceType.SECULAR, confidence=Confidence.HIGH)
    v3 = verdict_from_status_code(-3, ResonanceType.SECULAR, confidence=Confidence.HIGH)
    assert v5.confidence == Confidence.LOW
    assert Flag.PROBABLE_NEAR_SEPARATRIX in v5.flags
    assert Flag.PROBABLE_NEAR_SEPARATRIX not in v3.flags


def test_legacy_minus_two_and_minus_nine_never_merge():
    # CLAUDE.md: -2 is a measurement, -9 the absence of one.
    v2 = verdict_from_status_code(-2, ResonanceType.LIDOV_KOZAI, confidence=Confidence.HIGH)
    v9 = verdict_from_status_code(-9, ResonanceType.LIDOV_KOZAI, confidence=Confidence.HIGH)
    assert (v2.status, v2.reason) != (v9.status, v9.reason)
    assert v2.status_code == -2 and v9.status_code == -9


@pytest.mark.parametrize('code', [2, 1, 0, -2, -3])
def test_legacy_confidence_uncalibrated_and_never_certain(code):
    v = verdict_from_status_code(code, ResonanceType.SECULAR, confidence=Confidence.CERTAIN)
    assert v.confidence != Confidence.CERTAIN
    assert Flag.CONFIDENCE_UNCALIBRATED in v.flags


def test_legacy_keeps_old_level():
    v = verdict_from_status_code(2, ResonanceType.SECULAR, confidence='medium')
    assert v.confidence == Confidence.MEDIUM


@pytest.mark.parametrize('code', [-4, -6, -9, -99])
def test_legacy_uncertain_and_invalid_have_no_level(code):
    v = verdict_from_status_code(code, ResonanceType.LIDOV_KOZAI, confidence=Confidence.HIGH)
    assert v.confidence == Confidence.NONE


# SPEC §5.7, column action (2 and 1 "by confidence": high -> none per §5.6)
SECTION_57_ACTION = {
    2: Action.NONE,
    1: Action.NONE,
    0: Action.NONE,
    -2: Action.NONE,
    -4: Action.EXTEND,
    -5: Action.REVIEW,
    -6: Action.EXTEND,
    -9: Action.EXTEND,
    -99: Action.REVIEW,
}


@pytest.mark.parametrize('code', sorted(SECTION_57_ACTION))
def test_legacy_action_table(code):
    v = verdict_from_status_code(code, ResonanceType.SECULAR, confidence=Confidence.HIGH)
    assert v.action == SECTION_57_ACTION[code]


def test_legacy_minus_three_spot_check_at_medium():
    v = verdict_from_status_code(-3, ResonanceType.SECULAR, confidence=Confidence.MEDIUM)
    assert v.action == Action.SPOT_CHECK


def test_legacy_checks_are_kept():
    checks = {'check_zlk_free_omega': CheckValue.PASS, 'check_zlk_ei': CheckValue.NOT_MEASURED}
    v = verdict_from_status_code(2, ResonanceType.LIDOV_KOZAI, confidence=Confidence.HIGH, checks=checks)
    assert v.checks['check_zlk_free_omega'] == CheckValue.PASS
    assert v.checks['check_zlk_ei'] == CheckValue.NOT_MEASURED


@pytest.mark.parametrize('code', [-1, 3, -7, -100])
def test_legacy_unknown_codes_refused(code):
    # -1 is produced by no classifier and is refused; codes outside the old set do not exist either.
    with pytest.raises((ValueError, KeyError)):
        verdict_from_status_code(code, ResonanceType.SECULAR, confidence=Confidence.HIGH)


# ---------------------------------------------------------------------------------------------------------------------
# default_action (SPEC §5.6)
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'status,reason,level,expected',
    [
        (Status.INVALID, Reason.ALIAS, Confidence.NONE, Action.RESAMPLE),
        (Status.INVALID, Reason.UNPHYSICAL, Confidence.NONE, Action.REVIEW),
        (Status.INVALID, Reason.UNDEFINED_PHASE, Confidence.NONE, Action.REVIEW),
        (Status.UNCERTAIN, Reason.BASELINE_NEEDED, Confidence.NONE, Action.EXTEND),
        (Status.UNCERTAIN, Reason.DRIFT_MECHANISM_UNKNOWN, Confidence.NONE, Action.EXTEND),
        (Status.UNCERTAIN, Reason.ORIGIN_UNKNOWN, Confidence.NONE, Action.REVIEW),
        (Status.RESONANT, Reason.DYNAMIC_LIBRATION, Confidence.CERTAIN, Action.NONE),
        (Status.RESONANT, Reason.DYNAMIC_LIBRATION, Confidence.HIGH, Action.NONE),
        (Status.TRANSIENT, Reason.DYNAMIC_EPISODE, Confidence.MEDIUM, Action.SPOT_CHECK),
        (Status.NEAR_SEPARATRIX, Reason.SEPARATRIX_ZONE, Confidence.LOW, Action.REVIEW),
        (Status.NON_RESONANT, Reason.CIRCULATION, Confidence.HIGH, Action.NONE),
        (Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, Confidence.MEDIUM, Action.SPOT_CHECK),
        (Status.NON_RESONANT, Reason.ALIAS, Confidence.LOW, Action.REVIEW),
        # uncertain overrides any level (no level is given to uncertain)
        (Status.UNCERTAIN, Reason.BASELINE_NEEDED, Confidence.HIGH, Action.EXTEND),
        (Status.INVALID, Reason.ALIAS, Confidence.HIGH, Action.RESAMPLE),
    ],
)
def test_default_action_table(status, reason, level, expected):
    assert default_action(status, reason, level) == expected


# ---------------------------------------------------------------------------------------------------------------------
# lower_confidence, best_status, join/split
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'levels,expected',
    [
        ((Confidence.HIGH, Confidence.LOW), Confidence.LOW),
        ((Confidence.CERTAIN, Confidence.MEDIUM, Confidence.HIGH), Confidence.MEDIUM),
        ((Confidence.NONE, Confidence.HIGH), Confidence.HIGH),
        ((Confidence.HIGH, Confidence.NONE, Confidence.CERTAIN), Confidence.HIGH),
        ((Confidence.CERTAIN,), Confidence.CERTAIN),
        ((Confidence.NONE, Confidence.NONE), Confidence.NONE),
        ((), Confidence.NONE),
    ],
)
def test_lower_confidence(levels, expected):
    assert lower_confidence(*levels) == expected


@pytest.mark.parametrize(
    'statuses,expected',
    [
        ([Status.NON_RESONANT, Status.TRANSIENT], Status.TRANSIENT),
        ([Status.NON_RESONANT, Status.UNCERTAIN], Status.UNCERTAIN),  # rank: uncertain before non_resonant
        ([Status.INVALID, Status.NEAR_SEPARATRIX], Status.NEAR_SEPARATRIX),
        ([Status.TRANSIENT, Status.RESONANT, Status.NEAR_SEPARATRIX], Status.RESONANT),
        ([Status.INVALID], Status.INVALID),
        ([], None),
    ],
)
def test_best_status(statuses, expected):
    assert best_status(statuses) == expected
    assert best_status(iter(statuses)) == expected  # any iterable, consumed once


def test_join_split_lists():
    assert join_list(['L2', 'L4', 'U:force_r2_full']) == 'L2;L4;U:force_r2_full'
    assert ',' not in join_list(['a', 'b', 'c'])
    assert join_list([]) == ''
    assert split_list('L2;L4;U:force_r2_full') == ['L2', 'L4', 'U:force_r2_full']
    assert split_list('') == []
    assert split_list(None) == []
    assert split_list(float('nan')) == []
    assert split_list(np.nan) == []
    assert split_list('single') == ['single']
    items = ['exit_suspected', 'three_tp_episode']
    assert split_list(join_list(items)) == items


# ---------------------------------------------------------------------------------------------------------------------
# Verdict.to_flat_dict / from_flat_dict
# ---------------------------------------------------------------------------------------------------------------------


def sample_mmr_verdict():
    ledger = Ledger(thin_band=0.1, conventions={'mmr_l1_tp_full'})
    ledger.compare('l1_tp', 6.6, Op.GE, 7.0, 'TP', 'geo_tp_count', setting='mmr_l1_tp_full')
    ledger.compare('amplitude_below_pi', 2.0, Op.LT, math.pi, 'rad', 'geo_amplitude_deg')
    return Verdict(
        resonance_type=ResonanceType.MMR,
        status=Status.NON_RESONANT,
        angle_status=AngleStatus.LIBRATION,
        angle_subtype=AngleSubtype.MODULATED,
        confidence=Confidence.MEDIUM,
        confidence_raw=Confidence.HIGH,
        reason=Reason.KINEMATIC_LIBRATION,
        flags=[Flag.EXIT_SUSPECTED.value, Flag.THREE_TP_EPISODE.value],
        thin_margins=['L2', 'U:l1_tp'],
        failed_support=['L5'],
        caps=['exit_suspected'],
        action=Action.SPOT_CHECK,
        libration_origin=LibrationOrigin.KINEMATIC,
        origin_detail=OriginDetail.VARPI_TERM,
        owner='',
        checks={'check_mmr_kinematic': CheckValue.FAIL, 'check_mmr_a_response': CheckValue.NOT_MEASURED},
        margins={
            'L1': mv(Margin.THIN, 6.6),
            'K1': mv(Margin.FULL, 0.985),
            'L4': mv(Margin.NOT_MEASURED, None),
        },
        decision_path=list(ledger),
    )


def test_flat_dict_has_every_check_column_and_foreign_ones_empty():
    row = sample_mmr_verdict().to_flat_dict()
    for name in ALL_CHECKS:
        assert name in row
    for name in SECULAR_CHECKS + ZLK_CHECKS:
        assert row[name] == ''
    assert row['check_mmr_kinematic'] == 'fail'
    assert row['check_mmr_a_response'] == 'not_measured'


def test_flat_dict_values():
    row = sample_mmr_verdict().to_flat_dict()
    assert row['status'] == 'non_resonant'
    assert row['angle_status'] == 'libration'
    assert row['confidence'] == 'medium'
    assert row['confidence_raw'] == 'high'
    assert row['libration_origin'] == 'kinematic'
    assert row['origin_detail'] == 'varpi_term'
    assert row['flags'] == 'exit_suspected;three_tp_episode'
    assert row['thin_margins'] == 'L2;U:l1_tp'
    assert row['failed_support'] == 'L5'
    assert row['caps'] == 'exit_suspected'
    assert row['decision_path'] == 'l1_tp;amplitude_below_pi'
    assert row['status_code'] == 0
    assert row['mrg_L1'] == 'thin'
    assert row['mrg_L1_value'] == pytest.approx(6.6)
    assert row['mrg_L4'] == 'not_measured'
    assert row['mrg_L4_value'] is None  # not measured = empty, never 0
    assert row['extend_to_years'] is None
    assert row['resample_step_yr'] is None


def _assert_same_verdict(a, b):
    assert b.resonance_type == a.resonance_type
    assert b.status == a.status
    assert b.angle_status == a.angle_status
    assert b.angle_subtype == a.angle_subtype
    assert b.confidence == a.confidence
    assert b.confidence_raw == a.confidence_raw
    assert b.reason == a.reason
    assert list(b.flags) == list(a.flags)
    assert list(b.thin_margins) == list(a.thin_margins)
    assert list(b.failed_support) == list(a.failed_support)
    assert list(b.caps) == list(a.caps)
    assert b.action == a.action
    assert b.libration_origin == a.libration_origin
    assert b.origin_detail == a.origin_detail
    assert b.owner == a.owner
    for key, value in a.checks.items():
        assert b.checks[key] == value
    assert set(a.margins) <= set(b.margins)
    for key, value in a.margins.items():
        assert b.margins[key].margin == value.margin
        if value.value is None:
            assert b.margins[key].value is None
        else:
            assert b.margins[key].value == pytest.approx(value.value)
        assert b.margins[key].margin_class == MarginClass.DECISIVE
    assert [e.name for e in b.decision_path] == [e.name for e in a.decision_path]
    assert b.status_code == a.status_code


def test_flat_dict_roundtrip_in_memory():
    v = sample_mmr_verdict()
    _assert_same_verdict(v, Verdict.from_flat_dict(v.to_flat_dict()))


def test_flat_dict_roundtrip_through_csv(tmp_path):
    # summary.csv path: empty cells come back as NaN from pandas.
    v = sample_mmr_verdict()
    v.extend_to_years = 2.0e6
    path = tmp_path / 'summary.csv'
    pd.DataFrame([v.to_flat_dict()]).to_csv(path, index=False)
    row = pd.read_csv(path).iloc[0].to_dict()
    back = Verdict.from_flat_dict(row)
    _assert_same_verdict(v, back)
    assert back.extend_to_years == pytest.approx(2.0e6)
    assert back.resample_step_yr is None
    # decision path names only: numbers are not restored
    assert all(entry.result is None for entry in back.decision_path)


def test_flat_dict_roundtrip_subresonance_row():
    v = Verdict(
        resonance_type=ResonanceType.MMR,
        status=Status.NON_RESONANT,
        angle_status=AngleStatus.LIBRATION,
        reason=Reason.SUBRESONANCE_LIBRATES,
        libration_origin=LibrationOrigin.SUBRESONANCE,
        origin_detail=OriginDetail.FIXED_SUBRESONANCE,
        owner='4J-2S-1+0+0-1',
        confidence=Confidence.HIGH,
        confidence_raw=Confidence.HIGH,
        action=Action.NONE,
        multiplet_status=Status.RESONANT,
    )
    back = Verdict.from_flat_dict(v.to_flat_dict())
    _assert_same_verdict(v, back)
    assert back.multiplet_status == Status.RESONANT
    assert back.status_code == 0


def test_flat_dict_roundtrip_legacy_zlk_verdict():
    for code in LEGACY_CODES:
        v = verdict_from_status_code(code, ResonanceType.LIDOV_KOZAI, confidence=Confidence.HIGH)
        back = Verdict.from_flat_dict(v.to_flat_dict())
        assert back.status_code == code
        row = v.to_flat_dict()
        for name in MMR_CHECKS + SECULAR_CHECKS:
            assert row[name] == ''


# ---------------------------------------------------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------------------------------------------------


def test_compare_holds_and_full_margin():
    ledger = Ledger(thin_band=0.1)
    entry = ledger.compare('amp', 5.0, Op.LT, 10.0, 'deg', 'geo_amplitude_deg')
    assert entry.result is True and bool(entry)
    assert entry.margin == Margin.FULL
    assert entry.relative_distance == pytest.approx(0.5)  # |5/10 - 1|
    assert entry.mark == '✓'
    assert len(ledger) == 1 and list(ledger)[0] == entry


@pytest.mark.parametrize(
    'value,op,threshold,expected',
    [
        (3.0, '<', 3.0, False),
        (3.0, '<=', 3.0, True),
        (3.0, '>', 3.0, False),
        (3.0, '>=', 3.0, True),
        (3.0, '==', 3.0, True),
        (3.0, '!=', 3.0, False),
        (-2.0, '>', -3.0, True),
        (-4.0, '>=', -3.0, False),
    ],
)
def test_compare_operators_as_strings(value, op, threshold, expected):
    entry = Ledger(thin_band=0.1).compare('x', value, op, threshold, '', 'col')
    assert entry.result is expected
    assert bool(entry) is expected
    assert entry.op == Op(op)


def test_failed_full_comparison_mark():
    entry = Ledger(thin_band=0.1).compare('c4', 0.5, Op.GE, 1.5, '', 'circ_rate_ratio')
    assert entry.result is False and not entry
    assert entry.margin == Margin.FULL  # |0.5/1.5 - 1| = 0.667: far from the threshold
    assert entry.mark == '✗'


@pytest.mark.parametrize('missing', [None, float('nan'), np.nan, np.float64('nan')])
@pytest.mark.parametrize('op', list(Op))
def test_missing_value_never_compares(missing, op):
    # NaN != 1 is True in Python; the ledger must still give result None (not measured).
    ledger = Ledger(thin_band=0.1, conventions={'mmr_l1_tp_full'})
    entry = ledger.compare('x', missing, op, 1.0, '', 'col', setting='mmr_l1_tp_full')
    assert entry.result is None
    assert not entry
    assert entry.margin == Margin.NOT_MEASURED
    assert entry.relative_distance is None
    assert entry.mark == '–'
    assert ledger.thin_margins() == []


def test_convention_close_is_thin_whatever_the_outcome():
    # |6.6/7 - 1| = 0.0571 < 0.1 (fails); |7.4/7 - 1| = 0.0571 < 0.1 (holds): both thin.
    ledger = Ledger(thin_band=0.1, conventions={'mmr_l1_tp_full'})
    below = ledger.compare('l1_below', 6.6, Op.GE, 7.0, 'TP', 'geo_tp_count', setting='mmr_l1_tp_full')
    above = ledger.compare('l1_above', 7.4, Op.GE, 7.0, 'TP', 'geo_tp_count', setting='mmr_l1_tp_full')
    assert below.result is False and below.margin == Margin.THIN
    assert above.result is True and above.margin == Margin.THIN
    assert below.relative_distance == pytest.approx(0.4 / 7.0)
    assert above.convention is True
    assert above.mark == '~'
    assert ledger.thin_margins() == ['U:l1_below', 'U:l1_above']


def test_convention_far_is_full():
    # |8.5/7 - 1| = 0.214 > 0.1
    ledger = Ledger(thin_band=0.1, conventions={'mmr_l1_tp_full'})
    entry = ledger.compare('l1', 8.5, Op.GE, 7.0, 'TP', 'geo_tp_count', setting='mmr_l1_tp_full')
    assert entry.margin == Margin.FULL
    assert ledger.thin_margins() == []


def test_physics_threshold_never_thin():
    # pi for the amplitude is derived from physics: setting '' never makes U even at 1 % distance.
    ledger = Ledger(thin_band=0.1, conventions={'mmr_l1_tp_full'})
    entry = ledger.compare('amplitude_below_pi', 0.99 * math.pi, Op.LT, math.pi, 'rad', 'geo_amplitude_deg')
    assert entry.result is True
    assert entry.convention is False
    assert entry.margin == Margin.FULL
    # a setting that is not a convention: also no U
    entry = ledger.compare('e_caveat', 0.199, Op.LE, 0.2, '', 'ctx_e_mean', setting='mmr_l5_e_caveat')
    assert entry.margin == Margin.FULL
    assert ledger.thin_margins() == []


def test_negative_threshold_relative_distance():
    # value -0.95, threshold -1: |(-0.95)/(-1) - 1| = 0.05 < 0.1 -> thin
    ledger = Ledger(thin_band=0.1, conventions={'conv'})
    entry = ledger.compare('neg', -0.95, Op.LT, -1.0, '', 'col', setting='conv')
    assert entry.result is False
    assert entry.relative_distance == pytest.approx(0.05)
    assert entry.margin == Margin.THIN


def test_zero_threshold_has_no_relative_distance():
    ledger = Ledger(thin_band=0.1, conventions={'conv'})
    entry = ledger.compare('rate_sign', 1e-9, Op.GT, 0.0, 'rad/yr', 'rate', setting='conv')
    assert entry.result is True
    assert entry.relative_distance is None
    assert entry.margin != Margin.THIN
    assert ledger.thin_margins() == []


def test_numpy_scalars_are_compared():
    entry = Ledger(thin_band=0.1).compare('x', np.float64(2.5), Op.GT, np.float32(2.0), '', 'col')
    assert entry.result is True
    assert entry.relative_distance == pytest.approx(0.25)


def test_thin_margins_dedup_and_order():
    ledger = Ledger(thin_band=0.1, conventions={'conv'})
    ledger.compare('b', 1.05, Op.GE, 1.0, '', 'col_b', setting='conv')
    ledger.compare('a', 0.97, Op.GE, 1.0, '', 'col_a', setting='conv')
    ledger.compare('b', 1.02, Op.GE, 1.0, '', 'col_b', setting='conv')
    assert ledger.thin_margins() == ['U:b', 'U:a']


def test_decision_path_and_columns():
    ledger = Ledger(thin_band=0.1)
    ledger.compare('first', 1.0, Op.GT, 0.5, '', 'col_x')
    ledger.compare('second', 1.0, Op.GT, 0.5, '', 'col_y')
    ledger.compare('third', 1.0, Op.GT, 0.5, '', 'col_x')
    assert ledger.decision_path() == 'first;second;third'
    assert ledger.columns() == ['col_x', 'col_y']
    assert split_list(ledger.decision_path()) == ['first', 'second', 'third']


def test_extend_appends_in_order():
    main = Ledger(thin_band=0.1)
    main.compare('a', 1.0, Op.GT, 0.0, '', 'c1')
    sub = Ledger(thin_band=0.1, conventions={'conv'})
    sub.compare('b', 0.98, Op.GT, 1.0, '', 'c2', setting='conv')
    main.extend(sub)
    assert [e.name for e in main] == ['a', 'b']
    assert len(main) == 2
    # the thin margin of the sub-decision is kept (the entry carries its own margin)
    assert main.thin_margins() == ['U:b']


def test_name_only_entry():
    entry = LedgerEntry.name_only('l1_tp')
    assert entry.name == 'l1_tp'
    assert entry.result is None
    assert not entry


def test_describe_has_name_unit_op_and_mark():
    entry = Ledger(thin_band=0.1).compare('amp', 5.0, Op.LT, 10.0, 'deg', 'geo_amplitude_deg')
    text = entry.describe()
    assert 'amp' in text and 'deg' in text and '<' in text and '✓' in text
    assert '5' in text and '10' in text


def test_ledger_from_settings_defaults():
    settings = settings_from_config()
    ledger = Ledger.from_settings(settings)
    assert ledger.thin_band == pytest.approx(settings.confidence_thin_band)
    assert set(ledger.conventions) == set(conventions())
    # mmr_l5_ratio_full (1.15) is a convention: 1.20 is |1.2/1.15 - 1| = 0.043 away -> thin
    entry = ledger.compare('l5', 1.20, Op.LE, 1.15, '', 'mrg_L5_value', setting='mmr_l5_ratio_full')
    assert entry.margin == Margin.THIN
    assert ledger.thin_margins() == ['U:l5']


def test_ledger_from_settings_override_band():
    # band 0.25: |1.2/1.0 - 1| = 0.2 is thin; with the default 0.1 it is full.
    wide = Ledger.from_settings(settings_from_config(confidence_thin_band=0.25))
    narrow = Ledger.from_settings(settings_from_config())
    assert wide.compare('x', 1.2, Op.GE, 1.0, '', 'c', setting='mmr_force_r2_full').margin == Margin.THIN
    assert narrow.compare('x', 1.2, Op.GE, 1.0, '', 'c', setting='mmr_force_r2_full').margin == Margin.FULL


# ---------------------------------------------------------------------------------------------------------------------
# confidence: caps
# ---------------------------------------------------------------------------------------------------------------------


def test_every_cap_has_a_ceiling():
    assert set(conf.CAP_CEILING) == set(conf.Cap)


@pytest.mark.parametrize(
    'level,caps,expected',
    [
        (Confidence.CERTAIN, [conf.Cap.THREE_TP_EPISODE], Confidence.HIGH),
        (Confidence.CERTAIN, [conf.Cap.ZONE_NOT_MEASURED], Confidence.HIGH),
        (Confidence.HIGH, [conf.Cap.EXIT_SUSPECTED], Confidence.MEDIUM),
        (Confidence.HIGH, ['planets_missing'], Confidence.MEDIUM),
        (Confidence.MEDIUM, [conf.Cap.ONSET_FAILED], Confidence.LOW),
        (Confidence.LOW, [conf.Cap.THREE_TP_EPISODE], Confidence.LOW),  # a cap never raises
        (Confidence.CERTAIN, [conf.Cap.THREE_TP_EPISODE, conf.Cap.ONSET_FAILED], Confidence.LOW),
        (Confidence.CERTAIN, [conf.Cap.ENTRY_SUSPECTED, conf.Cap.THREE_TP_EPISODE], Confidence.MEDIUM),
        (Confidence.HIGH, [], Confidence.HIGH),
        (Confidence.NONE, [conf.Cap.ONSET_FAILED], Confidence.NONE),
    ],
)
def test_apply_caps(level, caps, expected):
    assert conf.apply_caps(level, caps) == expected
    assert conf.apply_caps(level, iter(caps)) == expected


# ---------------------------------------------------------------------------------------------------------------------
# confidence: raw level (SPEC §7.2)
# ---------------------------------------------------------------------------------------------------------------------


def full_physics():
    return {name: mv(Margin.FULL, 1.0) for name in PHYSICS}


def test_raw_all_full_is_certain():
    margins = {**full_physics(), 'L1': mv(Margin.FULL, 9), 'L5': mv(Margin.FULL, 1.0, MarginClass.SUPPORT)}
    level, thin, failed, all_nm = conf.raw_level(margins, PHYSICS)
    assert level == Confidence.CERTAIN
    assert thin == [] and failed == [] and all_nm is False


def test_raw_not_applicable_is_ignored():
    margins = {**full_physics(), 'C1': mv(Margin.NOT_APPLICABLE), 'K1': mv(Margin.NOT_APPLICABLE)}
    level, thin, failed, _ = conf.raw_level(margins, PHYSICS)
    assert level == Confidence.CERTAIN
    assert thin == [] and failed == []


def test_raw_thin_decisive_is_high_any_number():
    margins = {**full_physics(), 'L1': mv(Margin.THIN, 5), 'L2': mv(Margin.THIN), 'L7': mv(Margin.THIN)}
    level, thin, failed, _ = conf.raw_level(margins, PHYSICS)
    assert level == Confidence.HIGH
    assert set(thin) == {'L1', 'L2', 'L7'}
    assert failed == []


def test_raw_one_support_fail_is_medium():
    margins = {**full_physics(), 'L5': mv(Margin.FAIL, 3.0, MarginClass.SUPPORT)}
    level, thin, failed, _ = conf.raw_level(margins, PHYSICS)
    assert level == Confidence.MEDIUM
    assert failed == ['L5']


def test_raw_fail_dominates_thin():
    margins = {
        **full_physics(),
        'L2': mv(Margin.THIN),
        'L3': mv(Margin.FAIL, 0.4, MarginClass.SUPPORT),
    }
    level, thin, failed, _ = conf.raw_level(margins, PHYSICS)
    assert level == Confidence.MEDIUM
    assert 'L2' in thin and failed == ['L3']


@pytest.mark.parametrize('n_fail', [2, 3, 5])
def test_raw_two_or_more_support_fails_is_low(n_fail):
    support = {f'S{i}': mv(Margin.FAIL, None, MarginClass.SUPPORT) for i in range(n_fail)}
    level, _, failed, _ = conf.raw_level({**full_physics(), **support}, PHYSICS)
    assert level == Confidence.LOW
    assert sorted(failed) == sorted(support)


def test_raw_one_physics_not_measured_counts_as_thin():
    margins = {**full_physics(), 'L4': mv(Margin.NOT_MEASURED)}
    level, thin, failed, all_nm = conf.raw_level(margins, PHYSICS)
    assert level == Confidence.HIGH
    assert 'L4' in thin
    assert failed == [] and all_nm is False


def test_raw_support_not_measured_counts_as_thin():
    margins = {**full_physics(), 'F': mv(Margin.NOT_MEASURED, None, MarginClass.SUPPORT)}
    level, thin, _, _ = conf.raw_level(margins, PHYSICS)
    assert level == Confidence.HIGH
    assert 'F' in thin


def test_raw_all_physics_not_measured_flag():
    margins = {name: mv(Margin.NOT_MEASURED) for name in PHYSICS}
    margins['L1'] = mv(Margin.FULL, 9)
    level, thin, failed, all_nm = conf.raw_level(margins, PHYSICS)
    assert all_nm is True
    assert level in (Confidence.HIGH, Confidence.MEDIUM)  # never certain; the medium ceiling may come as a cap
    assert failed == []


def test_raw_cap_margin_failure_not_counted():
    # O1 is a cap margin: its failure limits the level (cap onset_failed) instead of being counted.
    margins = {**full_physics(), 'O1': mv(Margin.FAIL, 0.7, MarginClass.CAP)}
    level, _, failed, _ = conf.raw_level(margins, PHYSICS)
    assert failed == []
    assert level in (Confidence.CERTAIN, Confidence.HIGH)


def test_raw_decisive_fail_counted_like_support():
    margins = {**full_physics(), 'L6': mv(Margin.FAIL, 0.9)}
    level, _, failed, _ = conf.raw_level(margins, PHYSICS)
    assert level == Confidence.MEDIUM
    assert failed == ['L6']


# ---------------------------------------------------------------------------------------------------------------------
# confidence: assess
# ---------------------------------------------------------------------------------------------------------------------

RESONANT_CELL = ('resonant', 'libration', 'dynamic_libration')


def certain_margins():
    return {**full_physics(), 'L1': mv(Margin.FULL, 9), 'L7': mv(Margin.FULL, 0.1)}


def test_assess_certain_is_shown_without_a_cell():
    # author 2026-10-16 (C2, decision 7): no minimum of labels, the level from the margins and caps is shown
    a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', certain_margins(), [], [], PHYSICS, no_cells())
    assert a.confidence_raw == Confidence.CERTAIN
    assert a.confidence == Confidence.CERTAIN
    assert a.cell is None and a.cell_fields() == {}


def test_assess_reports_the_cell_and_never_changes_the_level():
    # a cell with errors (2 of 4) is reported, the level stays what the margins give
    record = conf.CellRecord(n_examples=4, n_errors=2, accuracy_low_95=0.15, accuracy_high_95=0.85)
    cells = conf.ConfidenceCells({RESONANT_CELL: record}, 'abc')
    a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', certain_margins(), [], [], PHYSICS, cells)
    assert a.confidence == Confidence.CERTAIN and a.cell == record
    assert a.cell_fields() == dict(cell_n_examples=4, cell_n_errors=2, cell_accuracy_low_95=0.15, cell_accuracy_high_95=0.85)
    margins = {**certain_margins(), 'L2': mv(Margin.THIN)}
    a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', margins, [], [], PHYSICS, cells)
    assert a.confidence == Confidence.HIGH and 'L2' in a.thin_margins and a.cell == record


def test_assess_cell_must_match_all_three_fields():
    cells = conf.ConfidenceCells({RESONANT_CELL: conf.CellRecord(100, 0)}, 'abc')
    a = conf.assess(Status.RESONANT, 'libration', 'dynamic_episode', certain_margins(), [], [], PHYSICS, cells)
    assert a.cell is None


def test_assess_ledger_u_margins_make_high():
    a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', certain_margins(), [], ['U:l1_tp'], PHYSICS, no_cells())
    assert a.confidence_raw == Confidence.HIGH
    assert 'U:l1_tp' in a.thin_margins


@pytest.mark.parametrize('status', [Status.UNCERTAIN, Status.INVALID])
def test_assess_uncertain_invalid_no_level(status):
    cells = conf.ConfidenceCells({(str(status), 'insufficient_time', 'baseline_needed'): conf.CellRecord(5, 0)}, 'abc')
    a = conf.assess(status, 'insufficient_time', 'baseline_needed', certain_margins(), [], [], PHYSICS, cells)
    assert a.confidence == Confidence.NONE
    assert a.confidence_raw == Confidence.NONE
    assert a.cell is None


def test_assess_cap_lowers_final_and_is_listed():
    a = conf.assess(
        Status.TRANSIENT,
        'episodic',
        'dynamic_episode',
        certain_margins(),
        [conf.Cap.EXIT_SUSPECTED],
        [],
        PHYSICS,
        no_cells(),
    )
    assert a.confidence == Confidence.MEDIUM
    assert 'exit_suspected' in a.caps
    assert a.confidence_raw in (Confidence.CERTAIN, Confidence.HIGH)


def test_assess_support_fail_plus_onset_cap_is_low():
    margins = {**certain_margins(), 'L5': mv(Margin.FAIL, 2.0, MarginClass.SUPPORT)}
    a = conf.assess(Status.TRANSIENT, 'episodic', 'dynamic_episode', margins, ['onset_failed'], [], PHYSICS, no_cells())
    assert a.confidence_raw == Confidence.MEDIUM
    assert a.confidence == Confidence.LOW
    assert a.failed_support == ['L5']


def test_assess_two_support_fails_low():
    margins = {
        **certain_margins(),
        'L5': mv(Margin.FAIL, 2.0, MarginClass.SUPPORT),
        'C3': mv(Margin.FAIL, None, MarginClass.SUPPORT),
    }
    a = conf.assess(Status.NON_RESONANT, 'circulation', 'circulation', margins, [], [], PHYSICS, no_cells())
    assert a.confidence == Confidence.LOW
    assert sorted(a.failed_support) == ['C3', 'L5']


def test_assess_all_physics_not_measured_at_most_medium():
    margins = {name: mv(Margin.NOT_MEASURED) for name in PHYSICS}
    margins['L1'] = mv(Margin.FULL, 9)
    a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', margins, [], [], PHYSICS, no_cells())
    assert a.confidence == Confidence.MEDIUM


def test_assess_one_physics_not_measured_stays_high():
    margins = {**certain_margins(), 'L4': mv(Margin.NOT_MEASURED)}
    a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', margins, [], [], PHYSICS, no_cells())
    assert a.confidence == Confidence.HIGH
    assert 'L4' in a.thin_margins


# ---------------------------------------------------------------------------------------------------------------------
# confidence: cells file
# ---------------------------------------------------------------------------------------------------------------------


def write_cells(path, cells):
    path.write_text(json.dumps({'version': 1, 'cells': cells}))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(status, angle_status, reason, n_examples, n_errors, accuracy_95=None):
    entry = dict(status=status, angle_status=angle_status, reason=reason, n_examples=n_examples, n_errors=n_errors)
    if accuracy_95 is not None:
        entry['accuracy_95'] = accuracy_95
    return entry


def test_load_cells_reads_every_cell(tmp_path):
    path = tmp_path / 'cells.json'
    sha = write_cells(
        path,
        [
            cell('resonant', 'libration', 'dynamic_libration', 24, 0, [0.862, 1.0]),
            cell('non_resonant', 'circulation', 'circulation', 101, 1, [0.946, 0.9983]),
            cell('transient', 'episodic', 'dynamic_episode', 0, 0),  # no example: no interval
        ],
    )
    loaded = conf.load_cells(path)
    assert loaded.sha == sha
    assert loaded.records[RESONANT_CELL] == conf.CellRecord(24, 0, 0.862, 1.0)
    assert loaded.records[('non_resonant', 'circulation', 'circulation')] == conf.CellRecord(101, 1, 0.946, 0.9983)
    empty = loaded.records[('transient', 'episodic', 'dynamic_episode')]
    assert (empty.n_examples, empty.n_errors) == (0, 0) and math.isnan(empty.accuracy_low_95)
    assert empty.verdict_fields() == dict(cell_n_examples=0, cell_n_errors=0, cell_accuracy_low_95=None, cell_accuracy_high_95=None)
    assert conf.load_cells(str(path)).records == loaded.records  # the same file read by a string path


def test_load_cells_missing_file(tmp_path):
    loaded = conf.load_cells(tmp_path / 'absent.json')
    assert dict(loaded.records) == {}
    assert loaded.sha == ''


@pytest.mark.parametrize('path', [None, ''])
def test_load_cells_package_file(path):
    loaded = conf.load_cells(path)
    assert loaded.sha == hashlib.sha256(conf.PACKAGE_CELLS.read_bytes()).hexdigest()
    package = json.loads(conf.PACKAGE_CELLS.read_text())
    assert {key: (r.n_examples, r.n_errors) for key, r in loaded.records.items()} == {
        (c['status'], c['angle_status'], c['reason']): (c['n_examples'], c['n_errors']) for c in package['cells']
    }


def test_file_sha(tmp_path):
    path = tmp_path / 'x.bin'
    path.write_bytes(b'abc')
    # sha256('abc'), FIPS 180-2 test vector
    assert conf.file_sha(path) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert conf.file_sha(tmp_path / 'absent') == ''


def test_cell_of_is_plain_strings():
    got = conf.cell_of(Status.RESONANT, 'libration', 'dynamic_libration')
    assert got == RESONANT_CELL
    assert all(type(x) is str for x in got)
    assert conf.cell_of('transient', 'episodic', 'dynamic_episode') == ('transient', 'episodic', 'dynamic_episode')


def test_assess_with_loaded_cells_end_to_end(tmp_path):
    path = tmp_path / 'cells.json'
    write_cells(path, [cell('resonant', 'libration', 'dynamic_libration', 100, 1, [0.946, 0.998])])
    a = conf.assess(Status.RESONANT, 'libration', 'dynamic_libration', certain_margins(), [], [], PHYSICS, conf.load_cells(path))
    # one error in the cell no longer removes certain: the counts are reported only
    assert a.confidence == Confidence.CERTAIN
    assert (a.cell.n_examples, a.cell.n_errors) == (100, 1)
