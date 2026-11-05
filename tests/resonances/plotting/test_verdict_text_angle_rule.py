"""The subtitle and the steps' header of an insufficient_time row name the angle rule the analysis saved
(geo_angle_rule), and the leaf G6 its own name.

The records are those of `tests/resonances/analysis/test_angle_rule_column.py` (their rules worked out there by hand):
a slow arc (G5a) and the same swing with two turning points outside the ripple (G6 few_cycles; the ledger holds G5a's
entry, written by the ripple test: 1992 SZ 1M-10J+1 and 2002 BG 1E-10S-1 were headed G5a). A file older than the column
falls back on the ledger, and that fallback no longer names G5a on a few_cycles row."""

import matplotlib

matplotlib.use('Agg')

from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.plotting import verdict_vocab as vv  # noqa: E402
from resonances.resonance.analysis.settings import AnalysisSettings  # noqa: E402
from tests.resonances.analysis.test_angle_rule_column import analysed, slow_swing  # noqa: E402

S = AnalysisSettings()


def explained(sigma, row=None, drop=()):
    analysis = analysed(sigma)
    columns = {**analysis.numbers_row(), **analysis.summary_row(), **(row or {})}
    for column in drop:
        columns.pop(column)
    facts = vt.MMRFacts(analysis, S, columns, analysis.episode_rows())
    return facts, vt.explain_mmr(analysis, S, columns, analysis.episode_rows())


def test_a_few_cycles_row_is_headed_by_the_leaf_g6_not_g5a():
    facts, text = explained(slow_swing(dip_deg=30.0))
    assert facts.angle_rule == 'G6' and vt.shown_rule(facts) == 'G6'
    assert text.subtitle.endswith('decided: too short to decide (G6)')
    assert text.decided_at == 'Decided at G6: too short to decide → UNCERTAIN'
    assert vt.shape_rule_ids(facts) == ('F6r', 'G6')
    assert 'G5a' not in text.decided_at and 'one slow arc' not in text.subtitle


def test_without_the_column_a_few_cycles_row_still_never_names_g5a():
    facts, text = explained(slow_swing(dip_deg=30.0), drop=('geo_angle_rule',))
    assert facts.angle_rule == '' and vt._slow_arc(facts) is None
    assert vt.shown_rule(facts) == 'G6' and text.decided_at == 'Decided at G6: too short to decide → UNCERTAIN'


def test_a_slow_arc_row_is_headed_by_g5a():
    facts, text = explained(slow_swing())
    assert facts.angle_rule == 'G5a' and vt.shape_rule_ids(facts) == ('F6r', 'G5a')
    assert text.subtitle.endswith('decided: one slow arc (G5a)')
    assert text.decided_at == 'Decided at G5a: one slow arc → UNCERTAIN'


def test_the_header_reads_the_column():
    # the column, not the ledger: a G5a record whose row says G5w is headed G5w
    facts, text = explained(slow_swing(), row={'geo_angle_rule': 'G5w'})
    assert vt.shown_rule(facts) == 'G5w' and vt.slow_swing_decided(facts)
    assert text.decided_at == 'Decided at G5w: slow swing in the well → UNCERTAIN'


def test_g6_names_the_leaf_and_the_angle_rule_apart():
    assert vv.rule_name('G6', leaf=True) == 'too short to decide'
    assert vv.decided('G6') == 'decided: too short to decide (G6)'
    assert vv.rule_name('G6') == 'too few cycles' == vv.rule_name('G6', subtype='few_cycles')
    assert vv.rule_name('G6', subtype='arc') == 'arc without turning points'
    assert vv.rule_name('G6', subtype='stall') == 'stall without turning points'
    assert vv.rule_name('G6', leaf=True, subtype='arc') == 'too short to decide'  # the leaf has one name
    assert vv.rule_name('G5a', subtype='arc') == 'one slow arc'  # only G6 sets several words
