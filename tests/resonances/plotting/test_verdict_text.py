"""The explanation of the verdict figures (plotting/verdict_text.py) and the explained right column.

The formatting helpers are checked on hand-computed values. The steps, checks and physics rows of ten benchmark
records of different statuses (skipped without the benchmark data) are checked against the verdict they explain: every
'value op threshold' comparison printed there, the leaf, the margins. The sentences, plaque, arithmetic and slotted
bullets of the old column were removed with it (2026-11-14).
"""

import math
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.plotting import verdict_column as vc  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.resonance.analysis.ledger import Margin  # noqa: E402
from resonances.resonance.analysis.mmr import io  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from tests.resonances.analysis.builders import tree_and_final  # noqa: E402

S = settings_from_config()
# 41588 3J+9S-2 is these tests' record of an owner search that could not run (too few force windows) and its cap
# all_physics_not_measured; since W33 the search runs again on short windows (MMR_OWNER_SHORT_WINDOW_CYCLES) and passes,
# so the record is read here without that retry
BEFORE_W33_OWNER = {'mmr3__41588__3J+9S-2+0+0-10'}


def settings_of(example: str):
    return replace(S, mmr_owner_short_window_cycles=0.0) if example in BEFORE_W33_OWNER else S


# --------------------------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    'value, expected',
    [(0.6691, '0.669'), (3959.2, '3 959'), (152.95, '153'), (2.7e-4, '0.00027'), (3e-6, '≈0'), (None, '—'), (0, '0')],
)
def test_fmt(value, expected):
    assert vt.fmt(value) == expected


@pytest.mark.parametrize('value, expected', [(193.4, '193 yr'), (19581.0, '19.6 kyr'), (6644500.0, '6.64 Myr'), (3.38e9, '3 380 Myr')])
def test_years(value, expected):
    assert vt.years(value) == expected


@pytest.mark.parametrize(
    'value, op, threshold, expected',
    [
        (0.669, '>=', 0.7, '0.669 < 0.7'),  # a failed '≥' prints the comparison as it came out
        (0.81, '<=', 0.5, '0.81 > 0.5'),
        (1.07, '<=', 1.0, '1.07 > 1'),
        (6.0, '>=', 7, '6 < 7'),
        (None, '>=', 0.7, '— (need ≥ 0.7)'),
    ],
)
def test_compare(value, op, threshold, expected):
    assert vt.compare(value, op, threshold) == expected


def test_compare_percent_and_short_reason():
    assert vt.compare_percent(1.0, '>=', 0.666) == '100% ≥ 67%'
    assert vt.short_reason('half-amplitude 30.1° < 45°: sin σ ≈ σ, the law cannot be told') == 'half-amplitude 30.1° < 45°'
    assert vt.short_reason('a lasts 19.6 kyr = 3.2 cycles < 3.5 (P_cyc 6.14 kyr); the test needs 2 windows') == (
        'a lasts 19.6 kyr = 3.2 cycles < 3.5 (P_cyc 6.14 kyr)'
    )


# --------------------------------------------------------------------------------------------- benchmark records

# (example, a short description of why it is here); statuses as of v2-w8
RECORDS = (
    'mmr2__153__3J-2+0-1',  # resonant, force law below its amplitude floor (Hilda)
    'coorbital__2002_AA29__1E-1+0+0',  # resonant 1:1, its own force law
    'mmr3__52356__5J-2S-2+0+0-1',  # transient in the chaotic layer, capped; force law: 3.2 cycles < 3.5
    'mmr3__7343__1M+1J-2+0+0+0',  # uncertain: the force law's phase moves
    'mmr3__1999_UQ__7E+2M-9+0+0+0',  # non_resonant alias
    'mmr3__117306__1V+5J-7+0+0+1',  # non_resonant with librating stretches
    'coorbital__524522__1V-1+0+0__20kyr',  # transient that ends in circulation
    'mmr2__2003_UG22__10E-17+0+7',  # resonant, medium by a cap, thin S
    'mmr3__1998_QA62__1V-5M+3+0+0+1',  # uncertain (5o): L4, R0 not measured, onset failed before the filter guard
    'mmr2__82831__13M-24+0+11',  # non_resonant, recurrent
    'mmr3__2012_KL9__5E-2M-10+0+0+7',  # non_resonant/implausible_strength (5s): small divisors leave 3e9
    'mmr3__8306__2M-5S-3+0+0+6',  # uncertain (5s*): a two-body angle crosses its commensurability
    'mmr3__2003_ND__6J-6S-1+0+0+1',  # uncertain (5n): one independent cycle
    'mmr3__24820__3J-1S-1+0+0-1',  # transient by the geometry alone (5u/6), cap strength_scale_only (W22)
    'mmr2__33128__6N-7+0+1',  # transient with the mark exit_and_capture
    'mmr3__501710__1E+7S-4+0+0-4__500kyr',  # uncertain (5s*): no two-body candidate, the residual a lower bound
    # near_separatrix of the pattern (G3), zone not measured: cap medium; 564160 3N-4 until 2026-11-20, a circulation
    # since the per-turn pattern B (its turns after 50 kyr pass the top on their own branch)
    'mmr2__523899__3N-5+0+2',
    'mmr2__30963__14M-25+0+11',  # non_resonant/certain far circulation: L8's step part not graded (W20 CS10)
    'mmr2__29744__23M-49+0+26',  # non_resonant/certain alias: C5 branch clearance 19.7° > 2 × 8.84° (W20 CS10)
    'mmr2__433__47M-44+0-3',  # transient (5 yr): R-ALIAS-A reads the failed L3/L4 as not measured (W22)
    'mmr2__2003_UC22__16M-27+0+11',  # resonant (5 yr): R-ALIAS-A reads the failed L3 as not measured (W22)
    'mmr2__2004_TK14__31M-34+0+3',  # resonant (5 yr): R-ALIAS-A and a union of episodes (W22)
    'mmr2__2004_TK14__31M-34+0+3__step0.5yr',  # resonant: a union promoted from transient by rule 6u (W22)
    'mmr3__2003_TX9__3E+2J-10+0+0+5',  # uncertain at the scale of the centre (5e: R0 measured and failed; W23 rule a)
    'mmr2__2001_ND13__1V-3+0+2',  # transient: L4 lifted by the two-harmonic force law (W23 CS12)
)
NUMBER = r'(?:−|-)?[0-9][0-9 ]*(?:\.[0-9]+)?'
COMPARISON = re.compile(rf'(?<![\w.^])(?<!e-)({NUMBER})(°|%| P| yr| kyr| rev)? (≤|≥|<|>|=) ({NUMBER})(°|%| P| yr| kyr| rev)?(?![\w.])')
HOLDS = {'≤': lambda a, b: a <= b, '≥': lambda a, b: a >= b, '<': lambda a, b: a < b, '>': lambda a, b: a > b, '=': lambda a, b: a == b}


def _number(text: str) -> float:
    return float(text.replace(' ', '').replace('−', '-'))


def comparisons(text: str):
    """Every 'value op threshold' printed in `text` (numbers as formatted, units on both sides equal or absent)."""
    for match in COMPARISON.finditer(text):
        yield match.group(0), match.group(1), match.group(2), match.group(3), match.group(4), match.group(5)


def _all_text(text: vt.VerdictText) -> str:
    parts = [step.outcome for step in text.steps]
    parts += [f'{row.detail} {row.note}' for row in text.checks]
    parts += [row.text for section in text.physics for row in section.rows]  # W20 CS11: the physics block too
    return ' | '.join(parts)


def _explain(analysis, settings=S) -> vt.VerdictText:
    return vt.explain_mmr(analysis, settings, {**analysis.numbers_row(), **analysis.summary_row()}, analysis.episode_rows())


class Record(tuple):
    """(example, analysis, episodes, text) of the final verdict; `tree` = (analysis, text) of the W28 tree's verdict from
    the same analysis (W34: the minimal policy re-reads it; a figure explains a tree verdict as it did before W29)."""

    tree = None


@pytest.fixture(scope='module', params=RECORDS)
def record(request):
    """(analysis, text) of one benchmark record, the final one and the tree's (`Record.tree`)."""
    example = request.param
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    tree, analysis = tree_and_final(io.load_series(example), S)
    out = Record((example, analysis, analysis.episode_rows(), _explain(analysis)))
    out.tree = (tree, _explain(tree))
    return out


def assert_comparisons_hold(text: vt.VerdictText) -> None:
    """No step, check or physics row claims 'a op b' when the numbers say otherwise (rounding: equal printed values
    pass a strict comparison only when they print the same)."""
    for whole, left, unit_left, op, right, unit_right in comparisons(_all_text(text)):
        if unit_left != unit_right and (op == '=' or (unit_left and unit_right)):
            continue  # '19.6 kyr = 3.2 cycles': a conversion, not a comparison
        a, b = _number(left), _number(right)
        assert HOLDS[op](a, b) or (op in '<>' and left == right), whole


@pytest.mark.benchmark
def test_every_printed_comparison_holds(record):
    assert_comparisons_hold(record[-1])


@pytest.mark.benchmark
def test_checks_follow_the_margins(record):
    _, analysis, _, text = record
    v = analysis.verdict
    if not str(v.confidence):
        return
    rows = {}
    for row in text.checks:
        rows.setdefault(row.key[:2] if row.key.startswith('L8') else row.key, []).append(row)
        if row.merged:  # W28 E4: a failed cap-class margin is printed inside its cap's LIMIT row
            rows.setdefault(row.merged, []).append(row)
    expected = {
        Margin.FULL: {vt.CheckState.PASS},
        Margin.THIN: {vt.CheckState.THIN, vt.CheckState.PASS},  # L8 is the worst of its parts
        Margin.FAIL: {vt.CheckState.FAIL},
        Margin.NOT_MEASURED: {vt.CheckState.SKIPPED, vt.CheckState.MISSING},
        Margin.NOT_APPLICABLE: {vt.CheckState.NOT_APPLICABLE},
    }
    # R-ALIAS-A (W22): a failed a-based check the level read as not measured is shown so, never as a pass
    unmeasured = {ALIAS_MARGINS[name] for name in str(analysis.extra_columns.get('rsp_short_period_unmeasured', '')).split(';') if name}
    for key, margin in v.margins.items():
        assert key in rows, key
        states = {row.state for row in rows[key]}
        read = Margin.NOT_MEASURED if key in unmeasured and margin.margin == Margin.FAIL else margin.margin
        assert states <= expected[read] | ({vt.CheckState.THIN} if key == 'L8' else set()), (key, margin.margin, states)
    assert {row.key[4:] for row in text.checks if row.is_cap} == set(v.caps)
    assert sum(1 for row in text.checks if row.counts == vt.COUNTS_FAILED) == len(v.failed_support)


ALIAS_MARGINS = {'a_response': 'L4', 'identity': 'L3'}


@pytest.mark.benchmark
def test_one_leaf_and_nothing_after_it(record):
    *_, text = record
    states = [step.state for step in text.steps]
    assert states.count(vt.StepState.LEAF) == 1
    after = text.steps[states.index(vt.StepState.LEAF) + 1 :]
    assert len(after) <= 1 and all(step.state == vt.StepState.NOT_REACHED and not step.outcome for step in after)  # W28 E3
    from resonances.plotting.verdict_vocab import RULE_NAMES

    assert all(rule_id in RULE_NAMES for step in text.steps for rule_id in step.ids)


@pytest.mark.benchmark
def test_52356_force_law_windows():
    """The author's question (16b): the last episode has 8 turning points, yet no force window: it lasts 19.6 kyr =
    3.2 cycles of P_cyc 6.14 kyr, and a window needs 3.5 cycles."""
    example = 'mmr3__52356__5J-2S-2+0+0-1'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(io.load_series(example), S)
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    longest = max((r for r in analysis.episode_rows() if r['accepted']), key=lambda r: r['t_end_yr'] - r['t_start_yr'])
    cycles = (longest['t_end_yr'] - longest['t_start_yr']) / row['geo_period_cycle_yr']  # 19581 / 6138
    assert math.isclose(cycles, 3.19, abs_tol=0.01)


@pytest.mark.benchmark
def test_secular_record_over_the_whole_record():
    """1020 g-g6+s-s6 (author, 16b): the whole record librates; the figure prints the status once and its FREQUENCIES."""
    tools = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
    example = 'secular__1020__g-g6+s-s6'
    sys.path.insert(0, str(tools))
    try:
        import bench_io as bio
        import reclassify_secular_zlk as reclassify
    except ImportError:
        pytest.skip('benchmark tools not available')
    if not bio.example_dir(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.plotting.secular_plots import draw_long_period_combined

    sim, body, resonance = reclassify.build_simulation(bio.load_example(example, with_planets=False))
    sim.identify_librations()
    analysis = sim.long_period_analysis(body, resonance)
    assert str(analysis.verdict.angle_status) == 'libration'
    fig = draw_long_period_combined(analysis)
    assert fig.text_stats['overflow'] == 0
    texts = [artist.get_text() for artist in fig.texts]
    # the status once, the header's word (the right column, 2026-11-04); the W28 tree's verdict is not printed (2026-11-14)
    assert texts.count('RESONANT') == 1 and 'W28 tree' not in ' '.join(texts)
    # the FREQUENCIES table (W27): the mode g6 and the forced wave g - g6 of the angle's own modes, one drawn row each
    (table,) = [group for group in fig.text_stats['column'].numbers.groups if group.table]
    names = [cells[0] for cells in table.cells]
    assert any(name.startswith('g6') for name in names) and any(name.startswith('g − g6') for name in names)
    assert any(t.startswith('g6') for t in texts) and any(t.startswith('g − g6') for t in texts)
    plt.close(fig)


def test_characteristic_periods():
    """Periods computed by hand: 2π/|ϖ̇ − g5| = 1296000/(500 − 4.2575) = 2614 yr; g5: 1296000/4.2575 = 304.4 kyr;
    g6: 45.88 kyr; s6: 1296000/26.345 = 49.19 kyr; the great inequality as constants.csv gives it."""
    from resonances.plotting.mmr_verdict_panel import period_lines

    row = {'ctx_P_syn_yr': 'J:10.8802;S:6.99893', 'ctx_P_orb_yr': 7.78, 'ctx_varpi_dot_arcsec_yr': 500.0}
    constants = [
        {'planet': 'Jupiter', 'g_arcsec_yr': 4.25749319, 's_arcsec_yr': 0.0},
        {'planet': 'Saturn', 'g_arcsec_yr': 28.24552984, 's_arcsec_yr': -26.34496354},
        {'planet': 'great_inequality', 'period_yr': 938.0, 'measured': True},
        {'planet': 'uranus_neptune_1_2', 'period_yr': 4233.0},
    ]
    printed = {line.label: line.value for line in period_lines(row, constants)}
    assert printed['synodic with Jup'] == '10.9 yr' and printed['synodic with Sat'] == '7 yr'
    assert printed['ϖ − ϖ_Jup (g5)'] == '2.61 kyr'
    assert printed['g5 (Jup)'] == '304 kyr' and printed['g6 (Sat)'] == '45.9 kyr' and printed['s6 (Sat)'] == '49.2 kyr'
    assert printed['great ineq. 2J−5S'] == '938 yr  measured' and printed['U–N 1:2'] == '4.23 kyr'
    assert 's5 (Jup)' not in printed  # s5 = 0: no period


@pytest.mark.benchmark
def test_force_law_after_the_jump_names_the_regime():
    """W9 (TODO 5) and W10 rule R: when the status rests on episodes before the last jump of a (G1e), the force law is
    measured on them; the column names the episode it was read on (the witness with most turning points). Without any
    measurement before the jump (the record's response only), it says the force law was measured on the last regime
    only instead of counting force windows. 524522 1V-1 20 kyr: episodes before the jumps at 17.7-19 kyr, then
    circulation."""
    from resonances.resonance.analysis.mmr.models import ResponseSource
    from resonances.resonance.analysis.mmr.response import witnesses

    example = 'coorbital__524522__1V-1+0+0__20kyr'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(io.load_series(example), S)
    geometry = analysis.geometry
    assert not geometry.episodes and geometry.episodes_before_jump  # G1e: nothing librates after the last jump
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    facts = vt.MMRFacts(analysis, S, row, analysis.episode_rows())
    response = analysis.response
    found = witnesses(response)
    witness = next(e for e in found if e.episode_id == response.witness_episode_id)
    where = f'measured on episode #{witness.episode_id} ({vt.interval(witness.t_start_yr, witness.t_end_yr)}'
    assert vt.measured_on(facts).startswith(where) and f'of {len(found)} eligible episodes the one with most' in vt.measured_on(facts)
    assert vt._force_detail(facts)[0].endswith(vt.measured_on(facts)) and vt.earlier_regime_text(facts) == ''

    analysis.response = replace(response, force_source=ResponseSource.RECORD)  # nothing measured before the jump
    facts = vt.MMRFacts(analysis, S, row, analysis.episode_rows())
    state, detail = vt.why_not_measured(facts, 'L4')
    n = len(geometry.episodes_before_jump)
    assert state == vt.CheckState.SKIPPED
    assert detail.startswith('it is measured only in the last regime after the jump')
    assert 'where σ circulates' in detail and f'{n} episodes before the jump are not measured' in detail
    assert 'force window' not in detail


# --------------------------------------------------------------------------------------------- W22 review fixes


def _explained(example, settings=S):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(io.load_series(example), settings)
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    return analysis, vt.explain_mmr(analysis, settings, row, analysis.episode_rows())


def _tree_explained(example, settings=S, text_settings=None):
    """(tree analysis, its text, final analysis, its text) of one record from a single analysis (W34: the W28 tree's
    verdict and the minimal policy's; `tree_and_final`); the texts read `text_settings` (default `settings`)."""
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    tree, final = tree_and_final(io.load_series(example), settings)
    text_settings = text_settings or settings
    return tree, _explain(tree, text_settings), final, _explain(final, text_settings)


# R-ALIAS-A records at 5 yr (the W22 R-ALIAS-A impact run, merged-scat+margins): both a-based checks fail as measured
# and are read as not measured; the scatter of a over the a swing 0.73 (TK14) and 0.79 (433) > 0.5
ALIAS_RECORDS = {'mmr2__2004_TK14__31M-34+0+3': 0.73, 'mmr2__433__47M-44+0-3': 0.79}


@pytest.mark.benchmark
@pytest.mark.parametrize('example', sorted(ALIAS_RECORDS))
def test_alias_text_names_the_rule_that_decided(example):
    """W34: the confidence part is the W28 tree's (the raw level, its support count); the minimal policy keeps the
    tree's transient/resonant and gives the level from the evidence."""
    analysis, text, final, _ = _tree_explained(example)
    assert final.verdict.status == analysis.verdict.status
    v = analysis.verdict
    assert analysis.extra_columns['rsp_short_period_unmeasured'] == 'a_response;identity'
    assert v.margins['L4'].margin == Margin.FAIL and v.margins['L3'].margin == Margin.FAIL  # the column keeps the fail
    assert str(v.confidence_raw) == 'high' and not v.failed_support
    direct = next(e for e in v.decision_path if e.name == vt.DIRECT_STRENGTH_ENTRY)
    assert direct.result
    force = next(step for step in text.steps if step.key == 'force')
    assert 'R-ALIAS-A' in force.outcome and 'strength by direct averaging decides' in force.outcome
    assert 'fails' not in force.outcome
    for key in ('L3', 'L4'):
        (row,) = [r for r in text.checks if r.key == key]
        assert row.state == vt.CheckState.MISSING and row.counts == vt.COUNTS_NOT_MEASURED
        assert row.detail.endswith(': not used)') and 'R-ALIAS-A' in row.note


@pytest.mark.benchmark
def test_a_fail_the_rule_did_not_touch_stays_failed():
    """2001 ND13 1V-3: the scatter 0.09 < 0.5 keeps its failed force law (R-ALIAS-A does not apply). With one harmonic
    only (W23 CS12 off): the two-harmonic law lifts its L4 (`test_nd13_harmonic_force_law_sentence`)."""
    analysis, text, final, final_text = _tree_explained('mmr2__2001_ND13__1V-3+0+2', replace(S, mmr_force_harmonics_max=1))
    assert not analysis.extra_columns['rsp_short_period_unmeasured']
    assert str(analysis.verdict.status) == 'uncertain'
    assert 'R-ALIAS-A' not in ' '.join(step.outcome for step in text.steps)
    # W34: the minimal policy gives the tree's 5a* the geometry's transient back with the flag measured_objection at
    # low (docs/physics/status-tree.md#minimal-policy); still without R-ALIAS-A
    v = final.verdict
    assert (str(v.status), str(v.confidence)) == ('transient', 'low') and 'measured_objection' in v.flags
    assert 'R-ALIAS-A' not in ' '.join(step.outcome for step in final_text.steps)


@pytest.mark.benchmark
def test_nd13_harmonic_force_law_sentence():
    """2001 ND13 1V-3 (W23 CS12): one harmonic of σ explains 0.18 of ȧ, two explain 0.65 (thin); the L4 detail names the
    law and its four guards, each as the record's numbers say."""
    analysis, text = _explained('mmr2__2001_ND13__1V-3+0+2')
    law = analysis.response.force_harmonic
    facts = vt.MMRFacts(analysis, S, {**analysis.numbers_row(), **analysis.summary_row()}, analysis.episode_rows())
    detail, need = vt._force_detail(facts)
    assert detail.startswith(f'2-harmonic law R² {vt.compare(law.r2, ">=", S.mmr_force_r2_fail)}, thin: ')
    assert f'thin: {vt.compare(law.r2, ">=", S.mmr_force_r2_full)} (one harmonic: R² ' in detail
    assert f'coefficients coherent {vt.compare(law.coherence, ">=", S.mmr_force_coef_coh_min)}' in detail
    assert f'shifted in time by {vt.compare(law.null_gain, ">=", S.mmr_force_null_gain)}' in detail
    assert f'one half predicts the other with R² {vt.compare(law.oos_r2, ">=", S.mmr_force_r2_fail)}' in detail
    assert '0.646 ≥ 0.4, thin: 0.646 < 0.7' in detail and 'up to 2 harmonics of σ' in need
    assert_comparisons_hold(text)


@pytest.mark.benchmark
def test_promoted_union_reads_episodic_then_resonant():
    """TK14 31M-34 at 0.5 yr: three accepted episodes (episodes.csv) without a circulation between them; the tree read
    them as episodic, reached transient/dynamic_episode and rule 6u promoted the union."""
    analysis, text = _explained('mmr2__2004_TK14__31M-34+0+3__step0.5yr')
    accepted = [r for r in analysis.episode_rows() if str(r.get('accepted')).lower() in ('true', '1')]
    assert len(accepted) == 3
    steps = {step.key: step.outcome for step in text.steps}
    assert steps['shape'].startswith('3 episodes without a circulation between them cover the record (union rule')
    assert steps['zone'].startswith('episodic → transient; the union covers the record → resonant (6u)')
    assert analysis.numbers_row()['rec_rule'] == '5d/6u'  # the final rule names the promotion (`decide.promote_union`)


@pytest.mark.benchmark
@pytest.mark.parametrize('example', ['mmr3__2003_TX9__3E+2J-10+0+0+5', 'mmr3__2003_TX9__3E+2J-10+0+0+5__step0.5yr'])
def test_centre_scale_union_does_not_lend_the_fine_episode(example):
    """2003 TX9 without the W23 center-scale TP rule (with it G2c keeps the slow libration): the union was found at the
    scale of the center only; the fine deciding episode ends before 14 kyr of a 20 kyr record, so its numbers are not
    printed as the whole record's libration."""
    # G2a off: it cuts this record at the Mars kick (W24, `test_tx9_names_the_action_step_and_its_one_potential`)
    w22 = replace(S, libration_centre_scale_tp_at_centre=False, libration_action_step=False)
    # W28 C16 regime_cut_mark (q-5 i) makes the W_c pieces one episode that G2c keeps: no union word, the deciding
    # episode is the whole 2.2-17.5 kyr run; the union-only text below is read with C16 and B1's stall join off
    joined, _ = _explained(example, w22)
    assert joined.extra_columns['geo_union_word_centre_scale'] == 0 and joined.prep.cutoff.binding == 'centre_scale'
    assert next(r for r in joined.episode_rows() if r['episode_id'] == joined.decisive_episode_id)['t_end_yr'] > 17000.0
    analysis, text = _explained(example, replace(w22, libration_regime_cut_marks=False, libration_stall_join=False))
    assert analysis.extra_columns['geo_union_word_centre_scale'] == 1
    decisive = next(r for r in analysis.episode_rows() if r['episode_id'] == analysis.decisive_episode_id)
    assert decisive['t_end_yr'] < 14000.0
    shape = next(step for step in text.steps if step.key == 'shape')
    assert 'at the scale of the center' in shape.outcome


@pytest.mark.benchmark
def test_tx9_one_joined_episode_is_not_a_union():
    """2003 TX9 on the W22 path (without the center-scale TP rule and G2a) with C16 on: the three W_c pieces are one
    episode 2.21–17.5 kyr with the mark stall@7992.0:14563.6 (test_w22_union; 7997.0 before W32: one output step,
    the W_c pieces read with the thin band of filter_too_slow), subtype regime_change."""
    w22 = replace(S, libration_centre_scale_tp_at_centre=False, libration_action_step=False)
    analysis, _ = _explained('mmr3__2003_TX9__3E+2J-10+0+0+5', w22)
    assert str(analysis.verdict.angle_subtype) == 'regime_change'
    (episode,) = [row for row in analysis.episode_rows() if vt.episode_accepted(row)]
    assert episode['marks'] == 'stall@7992.0:14563.6'


# --------------------------------------------------------------------------------------------- W23 S7 (F5, F6)


@pytest.mark.benchmark
def test_header_and_sentence_read_one_centre_test(record):
    """F5: the header's ', moving center' and the words 'σ librates ... with a fixed center' / 'the center is
    stationary' come from one test (`orbit_type.centre_moves`): no figure says both of σ's own center (a subresonance
    member's 'fixed center', 82831, is another angle's)."""
    _, analysis, _, text = record
    words = ' '.join(step.outcome for step in text.steps)
    if 'moving center' in str(analysis.verdict.orbit_type):
        assert 'the whole record with a fixed center' not in words and 'the center is stationary' not in words


@pytest.mark.benchmark
@pytest.mark.parametrize('example', ['mmr3__2003_TX9__3E+2J-10+0+0+5', 'mmr3__2003_TX9__3E+2J-10+0+0+5__step0.5yr'])
def test_tx9_names_its_moving_centre_and_the_scale_of_the_centre(example):
    """2003 TX9 3E+2J-10 after W23 rule (a): one episode at the scale of the center whose cycles' bands overlap (subtype
    stationary) while its center shifts by 1.45 (5 yr) of the mean amplitude, above MMR_L6_SHIFT_FULL 0.5: the header
    says 'moving center' (F5). The shape step names the scale of the
    center and the turning-point prominence it used (F6)."""
    analysis, text = _explained(example)
    v = analysis.verdict
    decisive = next(r for r in analysis.episode_rows() if r['episode_id'] == analysis.decisive_episode_id)
    shift = decisive['centre_shift_over_amp']
    assert str(v.angle_subtype) == 'stationary' and shift > S.mmr_l6_shift_full and v.orbit_type.endswith(', moving center')
    steps = {step.key: step.outcome for step in text.steps}
    assert steps['sub'] == 'not triggered: the cycles’ bands overlap (no trigger fired)'
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    window, fine, prominence = row['geo_centre_scale_cutoff_yr'], row['geo_fine_cutoff_yr'], row['geo_tp_prominence_deg']
    assert window > fine and prominence >= S.mmr_tp_prominence_floor_deg
    assert (
        f'at the scale of the center: σ smoothed at W_c {vt.years(window)}, longer than the fine cutoff {vt.years(fine)}' in steps['shape']
    )
    assert f'its turning points are turns of the center that stand out by ≥ {vt.fmt(prominence)}°' in steps['shape']
    assert_comparisons_hold(text)
    _, without = _explained(example, replace(S, libration_centre_scale_tp_at_centre=False))
    assert 'turns of the center' not in ' '.join(step.outcome for step in without.steps)


# --------------------------------------------------------------------------------------------- rule names (g10-3)
# The author's card g10-3 (2026-10-31, option C): every printed id reads 'short name [id]', a prefix only where ids
# collide (SZ. for the secular/ZLK margins, cap: for the caps); the subtitle reads 'decided: circulation (C)'.

RULE_ID = re.compile(r'^(G\d[a-z]?|C5?|R0|[1-6][a-z0-9]*\*?(/6u?)?)$')
# The leaves of decide_status as the column rec_rule writes them (mmr/decide.py, read on 2026-10-31): the paths
# 5c/6 and 5d/6 and their promotion by 6u (`promote_union`) are built there, not written as literals
REC_RULES = {
    'G0', 'G6', 'C', 'C5', 'R0', '1', '2', '3', '3k2', '3b5', '4', '5a', '5a*', '5b', '5e', '5f', '5g', '5n', '5o',
    '5s', '5s*', '5u/6', '5c/6', '5d/6', '5c/6u', '5d/6u',
}  # fmt: skip
G_RULES = {'G0', 'G1', 'G1c', 'G1e', 'G1l', 'G1q', 'G2', 'G2u', 'G2a', 'G2c', 'G3', 'G4', 'G4b', 'G5', 'G5a', 'G6'}
# the ids the WP names besides the margins of decide.MARGIN_CLASSES
OTHER_IDS = {'B1', 'F6', 'F7', 'K1', 'K2', 'B5', 'R', 'L8a', 'L8b', 'L8c', 'U'}


def _source(module) -> str:
    return Path(module.__file__).read_text()


def _rules_in_decide():
    """Every rule id decide.py writes as a literal: rule='X', rule = 'X', _unknown('X'), _no_dynamics(angle, 'X' if …
    else 'Y'), the constants RULE_*; a new leaf added there without a name fails the test below."""
    from resonances.resonance.analysis.mmr import decide

    text = _source(decide)
    found = set(re.findall(r"\brule\s*=\s*'([^']+)'", text)) | set(re.findall(r"_unknown\('([^']+)'", text))
    for call in re.findall(r"_no_dynamics\(angle, ([^)]*)\)", text):
        found |= set(re.findall(r"'([^']+)'", call))
    found |= {decide.RULE_UNION, decide.RULE_K2}
    return {rule for rule in found if RULE_ID.match(rule)}


def test_every_rule_id_has_a_short_name():
    """Completeness (g10-3): every rec_rule, G rule, margin, check and cap id the column can print has a name, so no id
    is printed bare or as 'unmapped check'."""
    from resonances.plotting import verdict_vocab as vv
    from resonances.resonance.analysis.confidence import Cap
    from resonances.resonance.analysis.mmr import decide
    from resonances.resonance.analysis.secular import decide as secular_decide

    scanned = _rules_in_decide()
    assert {'C', 'C5', 'R0', '1', '2', '3', '3b5', '4', '5f', '5g', '5a', '5a*', '5s*', '5o', '5n', 'G2u', '3k2'} <= scanned
    for rule in REC_RULES | scanned:
        assert vv.rule_name(rule, leaf=True), rule
        assert 'unmapped' not in vv.decided(rule), rule
    geometry_ids = set(re.findall(r"'(G\d[a-z]?)'", _source(decide))) | {decide.RULE_UNION}
    for rule in G_RULES | geometry_ids:
        assert rule in vv.RULE_NAMES, rule
    margins = set(decide.MARGIN_CLASSES) | set(vt.MMR_MARGIN_NAMES) | OTHER_IDS
    margins |= set(re.findall(r"CheckRow\(\s*'([A-Z][A-Za-z0-9]*)'", _source(vt)))  # L8, L8a, L8c rows
    for key in margins:
        assert key in vv.RULE_NAMES, key
    long_period = set(re.findall(r"margins\['([A-Z][A-Za-z0-9]*)'\]", _source(secular_decide))) | set(vt.LONG_PERIOD_MARGIN_NAMES)
    assert long_period == {'N', 'S', 'F', 'P', 'G'}
    for key in long_period:
        assert vv.long_period_id(key) in vv.RULE_NAMES, key
    for cap in Cap:
        (row,) = vt.cap_rows([str(cap)], 'medium')
        assert row.key == f'cap:{cap}' and row.key in vv.RULE_NAMES, cap
        assert str(cap) in vv.CAPS  # its CapText (CS11) is in the registry too


def test_short_names_do_not_collide():
    """No two ids print the same name (a leaf's outcome name included), and every name fits the column with its id."""
    from resonances.plotting import verdict_vocab as vv

    printed = list(vv.RULE_NAMES.values()) + list(vv.LEAF_NAMES.values())
    printed += [name for names in vv.ANGLE_SUBTYPE_NAMES.values() for name in names.values()]
    lowered = [name.lower() for name in printed]
    assert len(set(lowered)) == len(lowered), sorted({n for n in lowered if lowered.count(n) > 1})
    assert all(0 < len(name) <= 40 for name in printed)  # the column is about 60 characters wide, and the id follows
    assert set(vv.LEAF_NAMES) <= set(vv.RULE_NAMES)  # a leaf outcome only for an id that also names a check
    # the ids that collide are prefixed: the secular/ZLK margins and the caps, nothing else
    assert {key for key in vv.RULE_NAMES if '.' in key or ':' in key} == {f'SZ.{k}' for k in 'NSFPG'} | {f'cap:{c}' for c in vv.CAP_NAMES}


@pytest.mark.parametrize(
    'rule_id, expected',
    [
        ('B1', 'a carries the swing [B1]'),
        ('R0', 'energy fits the potential [R0]'),  # the check; the leaf R0 is its failure (below)
        ('cap:onset_failed', 'earlier a-wave [cap:onset_failed]'),
        ('SZ.N', 'enough reversals [SZ.N]'),
        ('U:5 L4 null test: R2 above the binding line - max R2 with a shifted', 'near its convention [U]'),
        ('X9', 'unmapped check [X9]'),  # never guessed
    ],
)
def test_named(rule_id, expected):
    from resonances.plotting import verdict_vocab as vv

    assert vv.named(rule_id) == expected


@pytest.mark.parametrize(
    'rec_rule, expected',
    [
        ('C', 'decided: circulation (C)'),
        ('R0', 'decided: no resonant potential (R0)'),
        ('C5', 'decided: aliased circulation (C5)'),
        ('5d/6', 'decided: force measured → judged by behavior (5d/6)'),
        ('5c/6u', 'decided: potential supports → union covers the record (5c/6u)'),
        ('3k2', 'decided: kinematic drift of the center (3k2)'),
        ('', ''),
    ],
)
def test_decided(rec_rule, expected):
    from resonances.plotting import verdict_vocab as vv

    assert vv.decided(rec_rule) == expected


def _circulating(key: str, a_au: float):
    """A record whose σ circulates: λ at the mean motion of `a_au`, ϖ turning once in 25 kyr, Jupiter on its mean
    longitude; the 2:1 rate at 3.35 au is 2 n_J − n − ϖ̇ ≈ −0.5°/yr, a revolution in ~700 yr, three over 2 kyr."""
    from resonances.data import const
    from resonances.resonance.analysis.mmr import context as ctx
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec

    spec = ResonanceSpec.from_key(key)
    t = np.arange(0.0, 2000.0 + 0.25, 0.5)
    n_jupiter = const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'] * math.pi / 180
    lam = float(ctx.mean_motion(a_au)) * t + 0.3
    varpi = 2 * math.pi * t / 25000.0 + 0.4
    sigma = spec.planet_coefficients[0] * (n_jupiter * t + 1.1) + spec.body_coefficient * lam + spec.body_varpi_coefficient * varpi
    ones = np.ones_like(t)
    a = a_au + 1e-5 * np.sin(2 * math.pi * t / 7.3)
    series = MMRSeries(
        t, a, 0.1 * ones, 0.05 * ones, 0.7 * ones, np.mod(varpi - 0.7, 2 * math.pi), np.mod(varpi, 2 * math.pi),
        np.mod(sigma, 2 * math.pi), spec, Regime(0.5, float(t[-1]), len(t)), {}, np.mod(lam, 2 * math.pi), name='g10-3',
    )  # fmt: skip
    analysis = analyse_mmr(series, S, scan_owner=False, scan_subresonances=False)
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    return analysis, vt.explain_mmr(analysis, S, row, analysis.episode_rows())


def test_a_circulating_record_is_decided_at_c():
    """The circulating synthetic reaches the leaf C (`test_circulation_steps_and_every_step_id_has_a_name` reads its
    steps)."""
    analysis, _ = _circulating('2J-1+0-1', 3.35)
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.reason), analysis.numbers_row()['rec_rule']) == ('non_resonant', 'circulation', 'C')


def test_the_vocabulary_is_one_registry():
    """verdict_text reads the registry, it keeps no copy: a cap text added to verdict_vocab is the one the column prints."""
    from resonances.plotting import verdict_vocab as vv

    assert vt.LEDGER_WORDS is vv.LEDGER_WORDS and vt.CAP_WORDS is vv.CAP_WORDS
    assert vt.OWNER_FLAG_REASONS['owner_search_off'] == (vt.CheckState.SKIPPED, vv.OWNER_FLAG_WORDS['owner_search_off'])
    assert set(vt.OWNER_FLAG_REASONS) == set(vv.OWNER_FLAG_WORDS)


def test_the_docs_table_lists_every_name():
    """docs/libration-analysis.md#names prints the same short name for every id (a renamed id updates the page)."""
    from resonances.plotting import verdict_vocab as vv

    page = Path(__file__).resolve().parents[3] / 'docs' / 'libration-analysis.md'
    text = page.read_text()
    section = text[text.index('{#names}') :]
    rows = dict(re.findall(r'^\| `([^`]+)` \| ([^|]+?) \|', section, flags=re.M))
    for rule_id, name in vv.RULE_NAMES.items():
        assert rows.get(rule_id) == name, rule_id
    for rule_id, name in vv.LEAF_NAMES.items():
        assert rows.get(f'{rule_id} (leaf)') == name, rule_id
    for rule_id, names in vv.ANGLE_SUBTYPE_NAMES.items():
        for subtype, name in names.items():
            assert rows.get(f'{rule_id} ({subtype})') == name, (rule_id, subtype)


# --------------------------------------------------------------------------------------------- W28 E3: the steps as questions


def _kinematic(owner=None):
    """A record whose σ = 2λ_J − λ − ϖ librates only through ϖ (ϖ swings ±0.9 rad in 400 yr), with a flat a (a swing
    of 1e-6 au) at the exact 2:1 mean motion plus a center rate of one turn in 60 kyr (so that an exact commensurability
    is excluded, rule 3 and not 3b5): a carries none of the swing σ needs, B1 ≈ 0 < 0.1, non_resonant /
    kinematic_libration (41588 3J+9S-2 in miniature). `owner` replaces the owner search (not run in the synthetic)."""
    from scipy.optimize import brentq

    from resonances.data import const
    from resonances.resonance.analysis.mmr import context as ctx
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec

    spec = ResonanceSpec.from_key('2J-1+0-1')
    t = np.arange(0.0, 3000.0 + 0.25, 0.5)
    n_jupiter = const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'] * math.pi / 180
    a_au = brentq(lambda a: float(ctx.mean_motion(a)) - 2 * n_jupiter, 2.0, 5.0)
    lam = (2 * n_jupiter + 2 * math.pi / 60000.0) * t + 0.3
    varpi = 0.4 + 0.9 * np.sin(2 * math.pi * t / 400.0)
    sigma = spec.planet_coefficients[0] * (n_jupiter * t + 1.1) + spec.body_coefficient * lam + spec.body_varpi_coefficient * varpi
    ones = np.ones_like(t)
    a = a_au + 1e-6 * np.sin(2 * math.pi * t / 7.3)
    series = MMRSeries(
        t, a, 0.1 * ones, 0.05 * ones, 0.7 * ones, np.mod(varpi - 0.7, 2 * math.pi), np.mod(varpi, 2 * math.pi),
        np.mod(sigma, 2 * math.pi), spec, Regime(0.5, float(t[-1]), len(t)), {}, np.mod(lam, 2 * math.pi), name='e3',
    )  # fmt: skip
    analysis = analyse_mmr(series, S, scan_owner=False, scan_subresonances=False)
    if owner is not None:
        analysis = replace(analysis, owner=owner)
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    return analysis, row, vt.explain_mmr(analysis, S, row, analysis.episode_rows())


def _owner_without_verdict(*flags):
    from resonances.resonance.analysis.mmr.models import OwnerResult
    from resonances.resonance.analysis.verdict import CheckValue
    from tests.resonances.analysis.builders import make

    return make(OwnerResult, check=CheckValue.NOT_MEASURED, flags=list(flags))


def _steps(text):
    return {step.key: step for step in text.steps}


def test_kinematic_leaf_needs_no_owner():
    """Card g04-7 (author A): at a kinematic leaf (B1 fails: a does not swing) the owner search, which fits a force on
    a, is not a gap: 'n/a: no force on a; not needed: step 3 decides without it', never 'could not run', even when the
    search had too few force windows (41588); the physics block B says the same. The same search at a leaf that reads a
    force is a gap ('not run')."""
    analysis, row, text = _kinematic(_owner_without_verdict('owner_too_few_windows'))
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.reason), row['rec_rule']) == ('non_resonant', 'kinematic_libration', '3')
    steps = _steps(text)
    assert steps['owner'].state == vt.StepState.NOT_NEEDED
    assert steps['owner'].outcome == 'n/a: no force on a; not needed: step 3 decides without it'
    assert steps['owner'].heading == '2 · Does another resonance drive a? [W]' and steps['owner'].why == ''
    assert not any('could not run' in step.outcome or 'not run:' in step.outcome for step in text.steps)
    leaf = steps['kin']
    assert leaf.state == vt.StepState.LEAF and leaf.heading == '3 · Does a (semi-major axis) swing with σ? [B1]'
    assert leaf.why.startswith('a resonance turns σ back through a') and leaf.outcome.endswith('→ non_resonant')
    (cycles,) = [section for section in text.physics if section.title == 'What makes the cycles?']  # the same predicate
    assert cycles.subtitle == 'owner search not needed' and cycles.rows[0].text.startswith('n/a: no force on a (step 3)')
    facts = vt.MMRFacts(analysis, S, {**row, 'rec_rule': '5d/6'}, analysis.episode_rows())  # a leaf that reads the force
    assert vt.step_not_needed(facts, vt._owner_step(facts)) == ''
    assert vt._owner_step(facts).outcome.startswith('not run: ')


def test_nothing_is_said_after_the_leaf():
    """The steps after the leaf are one pale line without an answer: '4–6 · drift, own force, behavior: not reached'."""
    *_, text = _kinematic()
    keys = [step.key for step in text.steps]
    assert keys == ['shape', 'sub', 'owner', 'kin', 'rest']
    rest = text.steps[-1]
    assert rest.state == vt.StepState.NOT_REACHED and rest.outcome == '' and rest.why == ''
    assert rest.heading == '4–6 · drift, own force, behavior: not reached'
    assert text.steps[0].heading == '0 · Does σ librate? [G2]'


def test_coorbital_owner_wording():
    """A 1:1: 'the owner search is not applied to a 1:1' (the project's rule; 'a 1:1 has no owner' misleads, Codex on
    card g10-4b), shown n/a (·), also where a kinematic leaf would make it 'not needed'."""
    *_, text = _kinematic(_owner_without_verdict('owner_coorbital'))
    owner = _steps(text)['owner']
    assert owner.state == vt.StepState.NOT_APPLICABLE
    assert owner.outcome == 'n/a: the owner search is not applied to a 1:1'
    words = ' '.join([step.outcome for step in text.steps] + [f'{row.detail} {row.note}' for row in text.checks])
    assert 'has no owner' not in words


def test_tree_status_reads_the_w28_tree():
    """2024 JR16 1E-1 (2026-11-02): the tree's leaf 5a (force law fails) gave uncertain, the minimal policy resonant;
    `tree_status` is the tree's word (v2_verdict), the verdict's own without one."""
    from resonances.resonance.analysis.verdict import POLICY_MINIMAL, AngleStatus, ResonanceType, Status, Verdict

    mmr, libration = ResonanceType.MMR, AngleStatus.LIBRATION

    v = Verdict(mmr, Status.RESONANT, libration, policy=POLICY_MINIMAL, v2_verdict='uncertain//origin_unknown/review/libration')
    v.flags = ('zone_eq_only', 'measured_objection')
    assert vt.tree_status(v) == 'uncertain'
    unmoved = Verdict(mmr, Status.RESONANT, libration, policy=POLICY_MINIMAL, v2_verdict='resonant/high/geometric_libration//libration')
    assert vt.tree_status(unmoved) == 'resonant'
    v2 = Verdict(mmr, Status.UNCERTAIN, libration, v2_verdict='')  # the v2 policy: the tree's status is the verdict's
    assert vt.tree_status(v2) == 'uncertain'


def test_collapse_after_leaf():
    keys = ('sub', 'owner', 'kin', 'drift', 'force', 'zone')
    steps = [vt.Step('shape', 'angle shape', vt.StepState.PASSED)] + [
        vt.Step(key, key, vt.StepState.LEAF if key == 'sub' else vt.StepState.NOT_REACHED, 'text') for key in keys
    ]
    out = vt.collapse_after_leaf(steps)
    assert [step.key for step in out] == ['shape', 'sub', 'rest']
    assert out[-1].heading == '2–6 · another resonance, a swings, drift, own force, behavior: not reached'
    last = [replace(step, state=vt.StepState.PASSED) for step in steps[:-2]] + [replace(steps[-2], state=vt.StepState.LEAF), steps[-1]]
    assert vt.collapse_after_leaf(last)[-1].heading == '6 · behavior: not reached'
    assert vt.collapse_after_leaf(steps[:2]) == steps[:2]  # nothing after the leaf


def test_circulation_steps_and_every_step_id_has_a_name():
    """A circulation: step 0 with its G rule and the leaf C; every printed id of the steps is a named id (card g10-3)."""
    from resonances.plotting.verdict_vocab import RULE_NAMES

    _, circulating = _circulating('2J-1+0-1', 3.35)
    assert [step.heading for step in circulating.steps] == ['0 · Does σ librate? [G4]', 'C · Circulation']
    *_, kinematic = _kinematic()
    for text in (circulating, kinematic):
        assert all(rule_id in RULE_NAMES for step in text.steps for rule_id in step.ids)


def test_the_footer_names_existing_docs_pages():
    """The footer line names pages that exist, with the anchors it gives."""
    from resonances.plotting import verdict_column_long as vl
    from resonances.plotting import verdict_column_mmr as vm

    docs = Path(__file__).resolve().parents[3] / 'docs'
    for text in (vm.DOCS, vl.DOCS):
        for page, anchor in re.findall(r'((?:physics/)?[a-z-]+\.md)(?:#([a-z-]+))?', text):
            source = (docs / page).read_text()
            assert not anchor or f'{{#{anchor}}}' in source or f'# {anchor.replace("-", " ").capitalize()}' in source, (page, anchor)


# 41588 3J+9S-2 (card g04-7: kinematic, owner n/a) and 2020 CB (cards g10-4/g10-4b: a 1:1 decided at 5d → 6)
E3_RECORDS = {
    'mmr3__41588__3J+9S-2+0+0-10': {
        'headings': ['0 · Does σ librate? [G2]', '1 · Does a sibling angle hold the center? [S]', '2 · Does another resonance drive a? [W]',
                     '3 · Does a (semi-major axis) swing with σ? [B1]', '4–6 · drift, own force, behavior: not reached'],
        'owner': (vt.StepState.NOT_NEEDED, 'n/a: no force on a; not needed: step 3 decides without it'),
    },
    'coorbital__2020_CB__1E-1+0+0': {
        'headings': ['0 · Does σ librate? [G1q, G2]', '1 · Does a sibling angle hold the center? [S]', '2 · Does another resonance drive a? [W]',
                     '3 · Does a (semi-major axis) swing with σ? [B1]', '4 · Does the center drift without a cause?',
                     '5d · Is it σ’s own force? [L4, R0]', '6 · How does σ behave? [Z]'],
        'owner': (vt.StepState.NOT_APPLICABLE, 'n/a: the owner search is not applied to a 1:1'),
    },
}  # fmt: skip


@pytest.mark.benchmark
@pytest.mark.parametrize('example', sorted(E3_RECORDS))
def test_the_steps_of_41588_and_2020_cb(example):
    """The questions with their ids and the branch letter, the owner step (not needed at 41588's kinematic
    leaf, n/a for the 1:1 of 2020 CB), no 'could not run' and no answer after the leaf; the figure does not overflow."""
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.plotting import mmr_combined as mc
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = io.load_series(example)
    analysis = analyse_mmr(series, settings_of(example))
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    text = vt.explain_mmr(analysis, S, row, analysis.episode_rows())
    expected = E3_RECORDS[example]
    assert [step.heading for step in text.steps] == expected['headings']
    owner = next(step for step in text.steps if step.key == 'owner')
    assert (owner.state, owner.outcome) == expected['owner']
    assert not any('could not run' in step.outcome for step in text.steps)
    fig = mc.draw_combined(analysis, series, settings=S)
    # the right column (2026-11-04): one ◆ step, its rule named in the footer too; the docs pages in the footer
    column = fig.text_stats['column']
    assert fig.text_stats['overflow'] == 0 and vc.problems(column) == []
    texts = ' '.join(artist.get_text() for artist in fig.texts)
    decides = next(line for line in column.checks if line.decides)
    assert vc.plain(decides.question) in texts and 'physics/status-tree.md' in texts
    plt.close(fig)


# --------------------------------------------------------------------------------------------- W28 E4: the confidence block


def _margin_row(key, margin, margin_class, detail='', need=''):
    """A margin's row as `verdict_text._margin_row` builds it (state, counts and kind from a real `MarginValue`)."""
    from resonances.resonance.analysis.verdict import MarginValue

    value = MarginValue(margin, None, margin_class)
    return vt.CheckRow(key, vt.MMR_MARGIN_NAMES.get(key, key), vt.margin_state(margin), detail, need,
                       vt.margin_counts(margin, margin_class), kind=vt.margin_kind(value))  # fmt: skip


def _cb_like():
    """The confidence block of 2020 CB 1E-1 in miniature (card g10-4b): O1, a cap-class margin, failed (a-wave
    before/inside 0.865 ≥ 0.5) and its cap onset_failed holds the level at low; 4 thin (L1, L2, L4, Z), 5 not measured
    (L5, L6, R0, S, L3), 2 full (L8a, L8b), W n/a. By hand: 0 failed support → raw high (thin and not measured allow
    high, `confidence.raw_level`); onset_failed ≤ low → low."""
    from resonances.resonance.analysis.verdict import Confidence, MarginClass, Verdict
    from tests.resonances.analysis.builders import make

    D, S_, CAP = MarginClass.DECISIVE, MarginClass.SUPPORT, MarginClass.CAP
    rows = [_margin_row('O1', Margin.FAIL, CAP, 'a-wave before/inside 0.865 ≥ 0.5', '< 0.25 (fail ≥ 0.5)')]
    rows += [_margin_row(key, Margin.THIN, D, 'x') for key in ('L1', 'L2', 'L4', 'Z')]
    rows += [_margin_row(key, Margin.NOT_MEASURED, cls) for key, cls in (('L5', S_), ('L6', D), ('R0', D), ('S', D), ('L3', S_))]
    rows += [_margin_row(key, Margin.FULL, S_, 'y') for key in ('L8a', 'L8b')]
    rows += [_margin_row('W', Margin.NOT_APPLICABLE, D)]
    verdict = make(Verdict, confidence=Confidence.LOW, confidence_raw=Confidence.HIGH, caps=['onset_failed'])
    return verdict, vt.confidence_rows(verdict, S, rows, whole_record=False)


def test_a_failed_limit_is_printed_once_as_limit():
    """Cards g03-5, g10-4/g10-4b: the failed O1 is one ▼ LIMIT row (the cap onset_failed with O1's numbers), first,
    never ✗; the support margins after it worst first; every row prints an id that has a short name."""
    from resonances.plotting import verdict_vocab as vv

    verdict, checks = _cb_like()
    o1 = [row for row in checks if 'O1' in (row.key, row.merged)]
    assert len(o1) == 1 and checks[0] is o1[0]
    (limit,) = o1
    assert limit.is_limit and limit.icon == '▼' and limit.label == 'earlier a-wave [O1]'
    assert limit.detail == 'a-wave before/inside 0.865 ≥ 0.5' and limit.need == '< 0.25 (fail ≥ 0.5)'
    assert not any(row.icon == '✗' for row in checks)
    assert [row.kind for row in checks].count(vt.KIND_LIMIT) == 1 and checks[-1].kind == vt.KIND_NA
    assert all(row.printed_id in vv.RULE_NAMES for row in checks)


def test_a_cap_above_the_level_holds_its_failed_margin():
    """1 failed support (L3) → raw medium (`confidence.raw_level`); the cap three_tp_episode (≤ high) explained by the
    failed cap-class N, merged into it, comes first."""
    from resonances.resonance.analysis.verdict import Confidence, MarginClass, Verdict
    from tests.resonances.analysis.builders import make

    rows = [
        _margin_row('L3', Margin.FAIL, MarginClass.SUPPORT, 'slope/c 3'),
        _margin_row('N', Margin.FAIL, MarginClass.CAP, '3 turning points'),
    ]
    verdict = make(Verdict, confidence=Confidence.MEDIUM, confidence_raw=Confidence.MEDIUM, caps=['three_tp_episode'])
    checks = vt.confidence_rows(verdict, S, rows, whole_record=False)
    assert [row.label for row in checks] == ['one closed cycle [N]', f'{vt.MMR_MARGIN_NAMES["L3"]} [L3]']


@pytest.mark.parametrize(
    'code, expected',
    [
        ('too_few_points', 'no value — the record holds fewer than {n} independent center windows; compare σ and p·ϖ in (j)'),
        ('no_body_varpi_term', 'no value — σ has no ϖ term of the body (p = 0); compare σ and p·ϖ in (j)'),
        ('', 'not measured'),
    ],
)
def test_k1_without_a_value_says_why(code, expected):
    """Card g10-4b (285637): K1 without a number prints kin_check_reason (A5) in words, and K1e its own reason."""
    analysis, row, _ = _kinematic()
    analysis = replace(analysis, kinematics=replace(analysis.kinematics, check_reason=code, k1_episode_reason=code))
    facts = vt.MMRFacts(analysis, S, row, analysis.episode_rows())
    for key in ('K1', 'K1e'):
        assert vt.why_not_measured(facts, key) == (vt.CheckState.MISSING, expected.format(n=S.mmr_kin_min_points))
    from resonances.resonance.analysis.mmr import kinematics as kin

    assert set(vt.KIN_REASON_WORDS) == {getattr(kin, name) for name in dir(kin) if name.startswith('K1E_')}


E4_RECORDS = ('coorbital__2020_CB__1E-1+0+0', 'mmr3__41588__3J+9S-2+0+0-10', 'mmr3__285637__3M+7J-6+0+0-4')


@pytest.mark.benchmark
def test_the_confidence_blocks_of_cb_41588_and_285637():
    """Card g10-4b on the benchmark: 2020 CB prints O1 once (since W28 B4 a support row; its limit is the q-2 b
    ceiling); 41588's limit is the cap own force unmeasured, first; 285637's K1 says why it has no
    value; no figure overflows. W34: the limits are the W28 tree's (its text); under the minimal policy both caps are
    flags and the final block has no LIMIT row (docs/reading-figures.md: 'the confidence block lists no caps')."""
    from resonances.plotting import mmr_combined as mc

    if not all(io.example_path(example).exists() for example in E4_RECORDS):
        pytest.skip('benchmark data not available')
    texts, finals = {}, {}
    for example in E4_RECORDS:
        series = io.load_series(example)
        tree, analysis = tree_and_final(series, settings_of(example))
        texts[example], finals[example] = _explain(tree), (analysis, _explain(analysis))
        fig = mc.draw_combined(analysis, series, settings=S)
        assert fig.text_stats['overflow'] == 0, example
        plt.close(fig)
    for example, cap in (
        ('coorbital__2020_CB__1E-1+0+0', 'no_full_physics_support'),
        ('mmr3__41588__3J+9S-2+0+0-10', 'all_physics_not_measured'),
    ):
        analysis, text = finals[example]
        assert cap in analysis.verdict.flags and not analysis.verdict.caps and not any(row.is_limit for row in text.checks)
    cb, k41588, k285637 = (texts[example] for example in E4_RECORDS)
    o1 = [row for row in cb.checks if 'O1' in (row.key, row.merged)]
    # W28 B4 (g03-4 A): CB's background is no foreign wave (coherence 0.02), so O1 is no limit and is printed once as a
    # support row (withdrawn by the filter guard: not measured); the q-2 b ceiling is the limit and holds medium
    assert len(o1) == 1 and not o1[0].is_limit and o1[0].state == vt.CheckState.MISSING
    limit = cb.checks[0]
    assert limit.is_limit and limit.label == 'no full physical support [cap:no_full_physics_support]'
    first = k41588.checks[0]
    assert first.is_limit and first.label == 'own force unmeasured [cap:all_physics_not_measured]'
    for k285637_text in (k285637, finals['mmr3__285637__3M+7J-6+0+0-4'][1]):
        (k1,) = [row for row in k285637_text.checks if row.key == 'K1']
        assert k1.state == vt.CheckState.MISSING and k1.note.startswith('no value — the record holds fewer than')
    from resonances.plotting import verdict_vocab as vv

    for text in list(texts.values()) + [text for _, text in finals.values()]:
        assert all(row.printed_id in vv.RULE_NAMES for row in text.checks)


# --------------------------------------------------------------------------------------------- W28 E11 pendulum labels


def test_pendulum_labels_name_their_source():
    """W28 E11 (card g10-7, author C): words, unit, source. T0 = P/F(A) comes from the observed cycles and is never
    called theory; the theory period is the strength's; the unit is still read from the column suffix."""
    from resonances.plotting import combined_numbers as numbers
    from resonances.plotting import mmr_verdict_panel as panel

    assert 'from observed cycles' in panel.label_of('ep_T0_min_kyr') and 'from observed cycles' in panel.label_of('ep_T0_max_kyr')
    assert 'theory' in panel.label_of('str_P_lib_theory_kyr') and 'theory' in panel.label_of('str_P_lib_theory_min_kyr')
    assert panel.label_of('ep_period_med_yr').startswith('observed')
    assert 'from T0' in panel.label_of('ep_half_width_au') and 'not a mass' in panel.label_of('ep_mu_eff')
    assert panel.unit_of('ep_T0_min_kyr') == 'kyr' and panel.unit_of('ep_half_width_au') == 'au'
    labels = list(numbers.COLUMN_LABELS.values()) + list(numbers.RANGE_LABELS.values())
    assert not any('T0' in label and 'theory' in label for label in labels)
    assert all(lo in numbers.COLUMN_LABELS and hi in numbers.COLUMN_LABELS for lo, hi in numbers.RANGE_LABELS)
    assert panel.label_of('smp_unwrap_resid_p99_deg') == 'unwrap_resid_p99'  # a column without words keeps its stem


@pytest.mark.benchmark
def test_137295_pendulum_numbers_say_their_source():
    """137295 1N-2 (card g10-7): T0 20.7–21.1 kyr is labeled 'from observed cycles'; block A compares the theory period
    T0_th (str_P_lib_theory, 18–18.2 kyr) with T0 from the observed cycles (str_T_obs 20.8 kyr), and nothing prints
    'T0 theory'."""
    from resonances.plotting import mmr_combined as mc
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    example = 'mmr2__137295__1N-2+0+1'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    series = io.load_series(example)
    analysis = analyse_mmr(series, S)
    fig = mc.draw_combined(analysis, series, settings=S)
    drawn = ' '.join(' '.join(artist.get_text().split()) for artist in fig.texts)
    overflow = fig.text_stats['overflow']
    plt.close(fig)
    assert overflow == 0
    assert 'T0 theory' not in drawn and '(theory)' not in drawn
    assert re.search(r'small-swing period T0 \(from observed cycles\) 2\d\.\d+–2\d\.\d+ kyr', drawn)
    # the right column prints the theory period in symbols (2026-11-04 redesign: 'T₀ theory', never the ASCII T0_th)
    assert re.search(r'T₀ theory 18(\.0)?–18\.2 kyr .*?against T₀ 20\.8 kyr from the observed cycles', drawn)


# --------------------------------------------------------------------------------------------- W28 E8 the explanation's records


def _kinematic_with_w():
    """The kinematic synthetic with the owner search that could not run (too few force windows, 41588) and its W
    margin as the decision writes it then (`decide._w_margin`: not_measured): the synthetic skips the search, so the
    margin is added to the verdict here."""
    from resonances.resonance.analysis.verdict import Margin, MarginValue

    analysis, *_ = _kinematic(_owner_without_verdict('owner_too_few_windows'))
    margins = {**analysis.verdict.margins, 'W': MarginValue(Margin.NOT_MEASURED)}
    analysis = replace(analysis, verdict=replace(analysis.verdict, margins=margins))
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    return analysis, row, vt.explain_mmr(analysis, S, row, analysis.episode_rows())


def test_kinematic_leaf_w_row():
    """Cards g10-8, g04-7 on the 2:1 kinematic synthetic (σ = 2λ_J − λ − ϖ, ϖ swings ±0.9 rad, a flat): σ's −ϖ term
    is the only oscillating part of σ, so it spans ~1 of the swing and the λ part ~0; the W row says 'other resonances
    not tested … not needed' (the search did not answer), never the claim 'no other resonance'."""
    analysis, row, text = _kinematic_with_w()
    assert 0.9 <= float(row['kin_swing_share']) <= 1.1 and float(row['kin_lambda_swing_share']) < 0.1
    words = ' | '.join(f'{r.name} {r.label} {r.note}' for r in text.checks)
    assert 'no other resonance' not in words
    (w,) = [r for r in text.checks if r.key == 'W']
    assert w.name == vt.OTHER_NOT_TESTED and w.note.startswith('not needed: no force on a (step 3); not run: ')
    assert vt.CIRCULATION_QUESTIONS['R0'] == 'Do the turning points come from a potential?'


def _with_owner(cover: float, psi_turns: float = float('nan'), flags=()):
    """The kinematic synthetic with an accepted owner 5E-16+0+11 named by the verdict (cause_role owner): its measured
    regime 1–2 kyr, its F7 cover of σ's window `cover` and P2 turns `psi_turns` (A1's `EpisodeCover`)."""
    from resonances.resonance.analysis.mmr.models import EpisodeCover, OwnerCandidate, OwnerResult, ResonanceSpec
    from resonances.resonance.analysis.verdict import CauseRole, CheckValue
    from tests.resonances.analysis.builders import make

    analysis, row, _ = _kinematic()
    accepted = make(OwnerCandidate, spec=ResonanceSpec.from_key('5E-16+0+11'), r2_median=0.94, accepted=True)
    window = EpisodeCover(None, 0.0, 3000.0, 400.0, cover, psi_turns)
    owner = make(OwnerResult, check=CheckValue.FAIL, accepted=accepted, n_candidates=1, candidates=[accepted],
                 intervals=[(1000.0, 2000.0, 'libration')], episode_covers=(window,), cover=cover)  # fmt: skip
    verdict = replace(analysis.verdict, cause_role=CauseRole.OWNER, owner='5E-16+0+11', cause_term='5E-16+0+11', flags=list(flags))
    analysis = replace(analysis, owner=owner, verdict=verdict)
    return vt.MMRFacts(analysis, S, {**row, 'own_psi_def': '8*sigma-1*phi'}, analysis.episode_rows())


@pytest.mark.parametrize(
    'psi_def, expected',
    [
        ('8*sigma-1*phi', 'σ = φ/8 + Ψ/8'),  # HW7 5J+6S-2 / 5E-16: Ψ = 8σ − φ
        ('4*sigma-13*phi', 'σ = (13/4)φ + Ψ/4'),  # 719 36J-13 / 11J-4: Ψ = 4σ − 13φ
        ('3*sigma+2*phi', 'σ = −(2/3)φ + Ψ/3'),  # Ψ = 3σ + 2φ → σ = (Ψ − 2φ)/3
        ('1*sigma-2*phi', 'σ = 2φ + Ψ/1'),
    ],
)
def test_identity_line(psi_def, expected):
    facts = _with_owner(0.8)
    facts.row = {**facts.row, 'own_psi_def': psi_def}
    assert vt.identity_line(facts).startswith(expected + ':')


@pytest.mark.parametrize(
    'overlap, source, expected',
    [
        ('Venus;Earth', 'J2000', 'Radial overlap with Venus, Earth (planets taken at their J2000 orbits).'),
        ('Mars', 'saved', 'Radial overlap with Mars.'),
        ('none', 'saved:Mercury+Venus+Earth+Mars+Jupiter+Saturn', 'Radial overlap: none (Uranus, Neptune taken at their J2000 orbits).'),
        (None, None, 'Radial overlap: no data.'),
    ],
)
def test_orbit_line(overlap, source, expected):
    """Card g10-6 (E6 columns): 'none' (compared, no overlap) differs from blank (no data); the source when not saved."""
    assert vt.orbit_line({'ctx_orbit_overlap': overlap, 'ctx_orbit_overlap_source': source}) == expected


E8_RECORDS = (
    'mmr3__41588__3J+9S-2+0+0-10',
    'mmr3__285637__3M+7J-6+0+0-4',
    'coorbital__2020_CB__1E-1+0+0',
    'mmr2__2000_SF8__8M-9+0+1',
    'mmr2__719__36J-13+0-23',
    'mmr3__2001_HW7__5J+6S-2+0+0-9',
    'mmr2__1627__13E-33+0+20',  # the densest right column (CS11 survey)
)


@pytest.mark.benchmark
def test_the_explanation_records_on_the_benchmark():
    """Cards g10-2/g10-8/g05-1 on their records (the reading of Claude and Codex): 41588's −10ϖ spans 0.9–1.1 of σ's
    swing; 285637's −4ϖ 1.13 (as A5 measured it: the λ part carries a slow ±7° wave); 2000 SF8 never says 'drives a'
    without 'candidate' in its steps and checks; no figure overflows."""
    from resonances.plotting import mmr_combined as mc
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not all(io.example_path(example).exists() for example in E8_RECORDS):
        pytest.skip('benchmark data not available')
    texts, rows = {}, {}
    for example in E8_RECORDS:
        series = io.load_series(example)
        analysis = analyse_mmr(series, settings_of(example))
        rows[example] = {**analysis.numbers_row(), **analysis.summary_row()}
        texts[example] = vt.explain_mmr(analysis, S, rows[example], analysis.episode_rows())
        fig = mc.draw_combined(analysis, series, settings=S)
        assert fig.text_stats['overflow'] == 0, example
        plt.close(fig)
    assert 0.9 <= float(rows['mmr3__41588__3J+9S-2+0+0-10']['kin_swing_share']) <= 1.1
    assert 1.05 <= float(rows['mmr3__285637__3M+7J-6+0+0-4']['kin_swing_share']) <= 1.2
    sf8 = texts['mmr2__2000_SF8__8M-9+0+1']
    pieces = [s.outcome for s in sf8.steps] + [f'{r.name} {r.detail}' for r in sf8.checks]
    assert all('candidate' in piece for piece in pieces if 'drives a' in piece and 'no other resonance' not in piece)


# --------------------------------------------------------------------------------------------- W28 E10: numbers behind


@pytest.mark.parametrize(
    'name, rule_id',
    [  # names as the benchmark's decision paths print them (v2-w26-final summary.csv)
        ('G4 largest center pass', 'G4'),
        ('G1c center turns after the last turning point', 'G1c'),
        ('C5 alias: far from the resonance', 'C5'),
        ('3 B1 a carries the amplitude', 'B1'),
        ('3 K2b kinematic libration whose center moves more than its amplitude (L6)', 'K2b'),
        ('3 kinematic (episode): identity correlation', 'K1'),
        ('5 L4 force law R2 (half 1)', 'L4'),
        ('5 R0 V_E ~ V_F windows', 'R0'),
        ('5o O1 a-wave before / inside the episode', 'O1'),
        ('5o F: required / theory <= x', 'F'),
        ('5d (i) F by direct averaging: required / theory <= x', 'F'),
        ('5u/6 F on the order-of-magnitude scale: required / theory <= x', 'F'),
        ('5s required / directly averaged strength (two-body)', 'F'),
        ('5n independent cycles (center estimates) of the decisive episode', '5n'),
        ('1 subresonance phi0 coherence', 'S'),
        ('2 owner R2', 'W'),
        ('2 second capture: x sweep', 'capture'),
    ],
)
def test_entry_rule_id(name, rule_id):
    assert vt.entry_rule_id(name) == rule_id


def test_column_comparison_as_it_came_out():
    """41588's B1 (0.0954 against ≥ 0.1, failed) prints the way it went; 2020 CB's K/K_theory 1.037 inside both bounds
    [1/3, 3] is one interval."""
    from resonances.resonance.analysis.ledger import Ledger

    ledger = Ledger(thin_band=0.1)
    b1 = ledger.compare('3 B1 a carries the amplitude', 0.09542, '>=', 0.1, '', 'rsp_b1_ratio')
    assert vt._column_comparison([b1]) == '0.0954 < 0.1'
    high = ledger.compare('5 L4 co-orbital K/K_theory <= max', 1.037, '<=', 3.0, '', 'rsp_coorb_K_over_K_theory')
    low = ledger.compare('5 L4 co-orbital K/K_theory >= 1/max', 1.037, '>=', 1 / 3, '', 'rsp_coorb_K_over_K_theory')
    assert vt._column_comparison([high, low]) == '1.04 within [0.333, 3]'


E10_RECORDS = ('mmr3__41588__3J+9S-2+0+0-10', 'coorbital__2020_CB__1E-1+0+0', 'mmr3__285637__3M+7J-6+0+0-4')


@pytest.mark.benchmark
def test_numbers_behind_the_verdicts_of_41588_cb_and_285637():
    """Card g10-5 on its records: the figure frames the DECISIVE lines first in Numbers, no '↑' pointer (D1); every
    printed comparison holds; no figure overflows."""
    from resonances.plotting import mmr_combined as mc

    if not all(io.example_path(example).exists() for example in E10_RECORDS):
        pytest.skip('benchmark data not available')
    for example in E10_RECORDS:
        series = io.load_series(example)
        _, analysis = tree_and_final(series, settings_of(example))
        assert_comparisons_hold(_explain(analysis))
        fig = mc.draw_combined(analysis, series, settings=S)
        drawn = [artist.get_text() for artist in fig.texts]
        # the right column (2026-11-04): DECISIVE lines framed first in Numbers, no '↑' pointer (D1)
        assert fig.text_stats['overflow'] == 0 and 'DECISIVE' in ' '.join(drawn) and not any('↑' in t for t in drawn), example
        assert fig.text_stats['column'].numbers.decisive, example
        plt.close(fig)


# --------------------------------------------------------------------------------------------- W28 C13: 1:1 parts in words
# The episodes.csv rows of the records the cards name, as one.py printed them on 2026-11-01 (C7 marks, C8 residence and
# sub-regimes). Every expected string is worked out by hand: a time in yr with 3 significant digits ('601 yr', '2.65
# kyr'), a span in the unit of its end ('129–339 yr'), a number of 100-999 rounded to an integer by Python's round
# half to even (326.5 → '326'); the first and last sub-regime reach the core's turning points (128.502635 → '129').

CB_ROW = {  # 2020 CB 1E-1 (card g03-3 C): core 443-601 yr, residence to the end of the record
    'episode_id': 1, 'accepted': True, 't_start_yr': 442.859082, 't_end_yr': 600.512315, 'n_tp': 4,
    'period_med_yr': 117.827416, 'half_amp_med_deg': 13.293307, 'half_amp_max_deg': 23.770135,
    'coverage_start_gap_P': 3.758540, 'coverage_end_gap_P': 3.390197, 'marks': '', 'wells': 'qs',
    'core_t0_yr': 442.859082, 'core_t1_yr': 600.512315, 'residence_t0_yr': 442.859082, 'residence_t1_yr': 999.970507,
    'residence_bounded_by_start': 'circulation', 'residence_bounded_by_end': 'edge', 'subregimes': 'qs@442.9:600.5',
}  # fmt: skip
JA_ROW = {  # 2016 JA 1E-1 (card g03-2 A): one episode, horseshoe -> tadpole L4 at 338.7 yr, residence over the record
    'episode_id': 0, 'accepted': True, 't_start_yr': 128.502635, 't_end_yr': 536.561004, 'marks': 'well_change@338.7',
    'core_t0_yr': 128.502635, 'core_t1_yr': 536.561004, 'residence_t0_yr': 0.0, 'residence_t1_yr': 999.970507,
    'residence_bounded_by_start': 'edge', 'residence_bounded_by_end': 'edge',
    'subregimes': 'horseshoe@128.5:338.7;tadpole_L4@338.7:536.6',
}  # fmt: skip
XO3_ROW = {  # 2021 XO3 1V-1 (card g03-1 A): one quasi-satellite episode, two amplitude changes
    'episode_id': 0, 'accepted': True, 't_start_yr': 89.501836, 't_end_yr': 944.51937, 'n_tp': 14,
    'period_med_yr': 119.002441, 'half_amp_med_deg': 21.800995, 'half_amp_max_deg': 65.015706, 'centre_mod_2pi_deg': 2.400548,
    'marks': 'amplitude_change@326.5:446.0;amplitude_change@708.4:809.2', 'wells': 'qs',
    'core_t0_yr': 89.501836, 'core_t1_yr': 944.51937, 'residence_t0_yr': 0.0, 'residence_t1_yr': 999.970507,
    'residence_bounded_by_start': 'edge', 'residence_bounded_by_end': 'edge',
    'subregimes': 'qs@89.5:326.5;qs@446.0:708.4;qs@809.2:944.5',
}  # fmt: skip
TY11_ROW = {  # 2024 TY11 1M-1 20 kyr (card g03-8 A): core 2.65-8.69 kyr, a kick at the change to the quasi-satellite
    'episode_id': 1, 'accepted': True, 't_start_yr': 2652.2, 't_end_yr': 8687.2,
    'marks': 'well_change@4654.2;kick@4687.9;well_change@5118.5', 'core_t0_yr': 2652.2, 'core_t1_yr': 8687.2,
    'residence_t0_yr': 0.0, 'residence_t1_yr': 8687.2, 'residence_bounded_by_start': 'edge',
    'residence_bounded_by_end': 'circulation', 'subregimes': 'compound@2652.2:4654.2;qs@4654.2:5118.5;compound@5118.5:8687.2',
}  # fmt: skip


def test_c13_the_episode_lines_list_the_stay_and_the_marks():
    """The EPISODES block of the numbers: under an accepted 1:1 run its inferred stay ('start'/'end' at a record edge)
    and its marks, at most three, then '…'."""
    from resonances.plotting import mmr_verdict_panel as vp

    def under(row):
        return [line.value for line in vp.explained_episode_lines([row], row['episode_id'])][1:]

    assert under(CB_ROW) == ['  stay 443 yr–end (inferred)']
    assert under(JA_ROW) == ['  stay start–end (inferred)', '  well 339 yr']
    assert under(TY11_ROW) == ['  stay start–8.69 kyr (inferred)', '  well 4.65 kyr · kick 4.69 kyr · well 5.12 kyr']
    assert under(XO3_ROW) == ['  stay start–end (inferred)', '  amplitude 326–446 yr · amplitude 708–809 yr']
    many = {**JA_ROW, 'marks': 'well_change@100;well_change@200;well_change@300;well_change@400'}
    assert under(many)[1] == '  well 100 yr · well 200 yr · well 300 yr · …'
