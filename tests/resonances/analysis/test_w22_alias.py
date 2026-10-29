"""W22 R-ALIAS-A (author 2026-10-24): on a step that folds short periods (flag short_period_aliased), a failed force
law (L4) and identity (L3), which read a, count as not_measured in the tree and the level, only when the strength by
direct averaging passes and the short-period scatter of a exceeds mmr_short_period_a_ratio x the decisive a
half-excursion. A fail never becomes a pass; the cap stays.

Independent numbers:
- scatter ratios by hand: 1.21e-3 / 1.0e-3 = 1.21 (> 0.5, applies); 0.089e-3 / 1.0e-3 = 0.089 (2001 ND13's measured
  ratio, < 0.5, keeps the fail); a missing excursion gives no ratio and no change;
- the level by the rule of confidence.py: L3 and L4 failed = two failed support margins = low; read as not_measured
  they are thin, so high, and the cap short_period_aliased holds it at medium;
- the benchmark records: their 0.5-yr twins (no folding) and the W20 round D fine reruns, where L3 and L4 pass, give
  433 47M-44 transient/medium (the author's status) and 2003 UC22 16M-27 resonant; the fine rerun of 2001 ND13 1V-3
  stays uncertain (SP/dis/impact/R-ALIAS-A, rows_changed.csv).
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.decide import AngleDecision, StatusDecision
from resonances.resonance.analysis.settings import settings_from_config
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
    Status,
)

S = settings_from_config()
OFF = settings_from_config(mmr_short_period_a_unmeasured=False)
EPISODIC = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
FAIL, PASS, NM = CheckValue.FAIL, CheckValue.PASS, CheckValue.NOT_MEASURED
FOLDED = str(Flag.SHORT_PERIOD_ALIASED)


@dataclass
class _Response:  # the two a-based checks the rule reads; dataclasses.replace needs a dataclass
    a_response: CheckValue
    identity: CheckValue


def _call(response, flags=(FOLDED,), da=1.21e-3, excursion=1.0e-3, method=st.METHOD_DIRECT, check=PASS, settings=S):
    sampling = SimpleNamespace(flags=list(flags), short_period_da_au=da)
    sampling.a_scatter_ratio = decide.a_scatter_ratio(sampling, excursion)  # as the analysis sets it
    strength = SimpleNamespace(method=method, check=check)
    ledger = Ledger.from_settings(settings)
    read, withdrawn = decide.short_period_unmeasured(sampling, response, strength, settings, ledger)
    return read, tuple(item.check for item in withdrawn.values()), ledger


@pytest.mark.parametrize(
    'l4, l3, expected',
    [
        (FAIL, FAIL, (NM, NM)),  # TK14, 433: both a-based checks read as not measured
        (FAIL, PASS, (NM, PASS)),  # a pass stays a pass
        (PASS, FAIL, (PASS, NM)),
        (NM, FAIL, (NM, NM)),
    ],
)
def test_failed_a_checks_read_as_not_measured(l4, l3, expected):
    read, unmeasured, ledger = _call(_Response(l4, l3))
    assert (read.a_response, read.identity) == expected
    assert set(unmeasured) == {name for name, value in (('a_response', l4), ('identity', l3)) if value == FAIL}
    entry = next(e for e in ledger if e.name == decide.ALIAS_A)
    assert entry.result is True and entry.value == pytest.approx(1.21)
    assert entry.setting == 'mmr_short_period_a_ratio'


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(flags=()),  # the step folds nothing: no cap, no rule
        dict(check=FAIL),  # F fails: nothing else measures the force
        dict(check=NM),
        dict(method=st.METHOD_SCALE),  # the e^q scale is an estimate, not the direct strength
        dict(da=0.089e-3),  # 2001 ND13: scatter 0.089 of the a swing, its fail is real
        dict(excursion=float('nan')),  # no decisive excursion, no ratio
        dict(settings=OFF),
    ],
)
def test_the_rule_needs_the_fold_a_passed_f_and_the_scatter(kwargs):
    response = _Response(FAIL, FAIL)
    read, unmeasured, _ = _call(response, **kwargs)
    assert read is response and unmeasured == ()


def test_nothing_failed_nothing_changes():
    response = _Response(PASS, NM)
    read, unmeasured, ledger = _call(response)
    assert read is response and unmeasured == ()
    assert not any(e.name == decide.ALIAS_A for e in ledger)


def test_the_scatter_at_the_threshold_does_not_apply():
    read, unmeasured, _ = _call(_Response(FAIL, FAIL), da=S.mmr_short_period_a_ratio * 1.0e-3)
    assert unmeasured == () and read.a_response == FAIL


def _margins():
    return {
        'L3': MarginValue(Margin.FAIL, 0.2, MarginClass.SUPPORT),
        'L4': MarginValue(Margin.FAIL, 0.1, MarginClass.DECISIVE),
        'L5': MarginValue(Margin.FULL, 1.0, MarginClass.SUPPORT),
    }


def test_the_level_reads_l3_and_l4_as_the_tree_did():
    margins = _margins()
    plain = StatusDecision(Status.TRANSIENT, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC, '', '')
    withdrawn = {margin: decide.Withdrawal(decide.WITHDRAWN_ALIAS_A, name) for name, margin in decide.ALIAS_A_CHECKS.items()}
    applied = StatusDecision(Status.TRANSIENT, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC, '', '', withdrawn=withdrawn)
    assert decide.margins_as_tree_read(margins, plain) is margins
    read = decide.margins_as_tree_read(margins, applied)
    assert read['L3'].margin == Margin.NOT_MEASURED and read['L4'].margin == Margin.NOT_MEASURED
    assert read['L5'] == margins['L5'] and margins['L4'].margin == Margin.FAIL  # the column keeps the fail
    assert conf.raw_level(margins, decide.PHYSICS_MARGINS)[0] == Confidence.LOW  # two failed support margins
    level, thin, failed, _ = conf.raw_level(read, decide.PHYSICS_MARGINS)
    assert level == Confidence.HIGH and failed == [] and {'L3', 'L4'} <= set(thin)
    assert conf.apply_caps(level, [str(conf.Cap.SHORT_PERIOD_ALIASED)]) == Confidence.MEDIUM


# --- the benchmark records (skipped without the benchmark data) ------------------------------------------------------


def _verdict(example, settings=S):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), settings).verdict


def _alias_entry(verdict):
    return next((e for e in verdict.decision_path if e.name == decide.ALIAS_A), None)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, status, confidence',
    [
        ('mmr2__433__47M-44+0-3', 'transient', 'medium'),  # the author's transient; its 0.5-yr twin transient/medium
        ('mmr2__2003_UC22__16M-27+0+11', 'resonant', 'medium'),  # 0.5-yr twin resonant/high; the cap holds medium
    ],
)
def test_folded_records_take_the_verdict_of_their_fine_twins(example, status, confidence):
    verdict = _verdict(example)
    assert (str(verdict.status), str(verdict.confidence)) == (status, confidence)
    assert _alias_entry(verdict).result is True
    assert str(conf.Cap.SHORT_PERIOD_ALIASED) in verdict.caps
    assert not {'L3', 'L4'} & set(verdict.failed_support)


@pytest.mark.benchmark
def test_tk14_reads_its_a_checks_as_not_measured():
    """2004 TK14 31M-34 at 5 yr: the rule applies (scatter 0.79); its status is the union rule's (W22 item 8)."""
    verdict = _verdict('mmr2__2004_TK14__31M-34+0+3')
    assert _alias_entry(verdict).result is True
    assert str(verdict.status) != 'uncertain' and not {'L3', 'L4'} & set(verdict.failed_support)


@pytest.mark.benchmark
def test_nd13_keeps_its_failed_force_law():
    """2001 ND13 1V-3: scatter 0.089 of the a swing; its fine rerun is uncertain as well. With one harmonic only (W23
    CS12 off): the two-harmonic law lifts its L4 (test_w23_harmonic_force_law), and this test is about R-ALIAS-A."""
    verdict = _verdict('mmr2__2001_ND13__1V-3+0+2', replace(S, mmr_force_harmonics_max=1))
    entry = _alias_entry(verdict)
    assert entry is not None and entry.result is False and entry.value == pytest.approx(0.089, abs=0.01)
    assert str(verdict.status) == 'uncertain'


@pytest.mark.benchmark
def test_switched_off_433_stays_uncertain():
    verdict = _verdict('mmr2__433__47M-44+0-3', OFF)
    assert str(verdict.status) == 'uncertain' and _alias_entry(verdict) is None
