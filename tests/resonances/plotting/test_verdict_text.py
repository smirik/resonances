"""The explanation of the verdict figures (plotting/verdict_text.py) and the explained right column.

The formatting helpers are checked on hand-computed values. The text of ten benchmark records of different statuses
(skipped without the benchmark data) is checked sentence by sentence against the verdict it explains: the plaque, the
wording of the record's coverage ('over the whole record' only for a libration, 'episodes' only for several), the
status and confidence sentences, every 'value op threshold' comparison printed anywhere, the steps and the checks.
"""

import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.resonance.analysis.ledger import Margin  # noqa: E402
from resonances.resonance.analysis.mmr import io  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402

S = settings_from_config()

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
    'mmr2__564160__3N-4+0+1',  # near_separatrix of the pattern (G3), zone not measured: cap medium
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
    parts = text.sentences + [step.outcome for step in text.steps]
    parts += [f'{row.detail} {row.note}' for row in text.checks]
    parts += [row.text for section in text.physics for row in section.rows]  # W20 CS11: the physics block too
    return ' | '.join(parts + [text.arithmetic])


@pytest.fixture(scope='module', params=RECORDS)
def record(request):
    """(analysis, text) of one benchmark record; the analysis keeps only what the checks below read."""
    example = request.param
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(io.load_series(example), S)
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    episodes = analysis.episode_rows()
    return example, analysis, episodes, vt.explain_mmr(analysis, S, row, episodes)


def assert_comparisons_hold(text: vt.VerdictText) -> None:
    """No sentence, step or check claims 'a op b' when the numbers say otherwise (rounding: equal printed values
    pass a strict comparison only when they print the same)."""
    for whole, left, unit_left, op, right, unit_right in comparisons(_all_text(text)):
        if unit_left != unit_right and (op == '=' or (unit_left and unit_right)):
            continue  # '19.6 kyr = 3.2 cycles': a conversion, not a comparison
        a, b = _number(left), _number(right)
        assert HOLDS[op](a, b) or (op in '<>' and left == right), whole


def test_every_printed_comparison_holds(record):
    assert_comparisons_hold(record[-1])


def test_plaque_holds_the_verdict(record):
    _, analysis, _, text = record
    v = analysis.verdict
    status, angle, *rest = text.badge.split(' · ')
    assert status == str(v.status).replace('_', '-').upper()
    assert angle.split(' (')[0] == str(v.angle_status)
    assert rest == ([str(v.confidence).upper()] if str(v.confidence) else [])
    # the final regime is the tag beside the plaque, never inside it
    assert 'ends:' not in text.badge
    if str(v.final_regime) and str(v.final_regime) != str(v.angle_status):
        assert text.ends == f'ends: {v.final_regime}'
    else:
        assert text.ends == ''


def test_plaque_wording():
    """Written out by hand (author, 2026-10-17): status · angle_status · confidence, all three always, even when the
    angle_status repeats the status; the final regime is a separate tag 'ends: …'. An angle with librating episodes
    reads 'episodic' (author, 2026-10-17 10:56: not the status's word 'transient')."""
    plaque = {
        ('transient', 'episodic (recurrent)', 'medium', ''): ('TRANSIENT · episodic (recurrent) · MEDIUM', ''),
        ('transient', 'episodic (recurrent)', 'low', 'circulation'): (
            'TRANSIENT · episodic (recurrent) · LOW',
            'ends: circulation',
        ),
        ('non_resonant', 'episodic (recurrent)', 'medium', 'circulation'): (
            'NON-RESONANT · episodic (recurrent) · MEDIUM',
            'ends: circulation',
        ),
        ('resonant', 'libration (stationary)', 'high', ''): ('RESONANT · libration (stationary) · HIGH', ''),
        ('near_separatrix', 'near_separatrix', 'low', ''): ('NEAR-SEPARATRIX · near_separatrix · LOW', ''),
        ('uncertain', 'undefined', '', ''): ('UNCERTAIN · undefined', ''),
    }
    for (status, angle, confidence, final), (badge, ends) in plaque.items():
        text = vt.VerdictText(status, angle, confidence, '', [], [], [], '', final_regime=final)
        assert (text.badge, text.ends) == (badge, ends)

    class Verdict:
        angle_status, angle_subtype, final_regime = 'episodic', 'recurrent', 'circulation'

    assert vt.angle_text(Verdict()) == 'episodic (recurrent)'
    assert vt.final_regime_text(Verdict()) == 'circulation'
    Verdict.final_regime = 'episodic'
    assert vt.final_regime_text(Verdict()) == ''
    Verdict.angle_subtype = ''
    assert vt.angle_text(Verdict()) == 'episodic'


def test_coverage_wording(record):
    """'over the whole record' only when the angle librates over it; 'episodes' only when there are several."""
    _, analysis, episodes, text = record
    v = analysis.verdict
    angle = text.sentences[0]
    accepted = [row for row in episodes if str(row.get('accepted')).lower() in ('true', '1')]
    assert ('over the whole record' in angle) == (str(v.angle_status) == 'libration')
    if 'episode' in angle and str(v.angle_status) != 'libration':  # a union of episodes (W22) librates over the record
        assert len(accepted) >= 2 and f'{len(accepted)} separate episodes' in angle
    covered = re.search(r'covering ([0-9]+)% of the record|\(([0-9]+)% of the record\)', angle)
    if covered:
        span = analysis.numbers_row()['smp_span_yr']
        expected = 100 * sum(row['t_end_yr'] - row['t_start_yr'] for row in accepted) / span
        assert int(covered.group(1) or covered.group(2)) == round(expected)


def test_status_and_confidence_sentences(record):
    _, analysis, _, text = record
    v = analysis.verdict
    head = {'non_resonant': 'Non-resonant', 'near_separatrix': 'Near-separatrix'}.get(str(v.status), str(v.status).capitalize())
    assert text.sentences[1].startswith(head)
    if not str(v.confidence):
        assert not text.checks and 'No confidence level' in text.arithmetic
        return
    confidence = text.sentences[2]
    assert confidence.startswith(str(v.confidence).capitalize())
    capped = str(v.confidence) != str(v.confidence_raw)
    assert ('capped' in confidence) == capped
    if capped:
        assert f'it would be {v.confidence_raw}' in confidence
    assert text.arithmetic.endswith(f'→  {v.confidence}') and f'= {v.confidence_raw}' in text.arithmetic


def test_checks_follow_the_margins(record):
    _, analysis, _, text = record
    v = analysis.verdict
    if not str(v.confidence):
        return
    rows = {}
    for row in text.checks:
        rows.setdefault(row.key[:2] if row.key.startswith('L8') else row.key, []).append(row)
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
RAW_CEILING = {2: 'low', 1: 'medium'}  # failed support tests -> the highest raw level (confidence.assess)
LEVELS = ('certain', 'high', 'medium', 'low')


def test_arithmetic_agrees_with_the_raw_level(record):
    """The printed count of failed tests never allows less than the raw level printed next to it (433 47M-44 at 5 yr
    printed '2 failed (≥ 2 failed → low) = high' before W22's fix)."""
    _, analysis, _, text = record
    v = analysis.verdict
    if not str(v.confidence):
        return
    failed = int(re.match(r'raw: ([0-9]+) failed', text.arithmetic).group(1))
    ceiling = RAW_CEILING.get(min(failed, 2), 'certain')
    assert LEVELS.index(str(v.confidence_raw)) >= LEVELS.index(ceiling), text.arithmetic


def test_one_leaf_and_nothing_after_it(record):
    *_, text = record
    states = [step.state for step in text.steps]
    assert states.count(vt.StepState.LEAF) == 1
    after = states[states.index(vt.StepState.LEAF) + 1 :]
    assert all(state == vt.StepState.NOT_REACHED for state in after)


def test_52356_force_law_windows():
    """The author's question (16b): the last episode has 8 turning points, yet no force window: it lasts 19.6 kyr =
    3.2 cycles of P_cyc 6.14 kyr (3.3 of its own 5.93 kyr), and a window needs 3.5 cycles."""
    example = 'mmr3__52356__5J-2S-2+0+0-1'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(io.load_series(example), S)
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    text = vt.explain_mmr(analysis, S, row, analysis.episode_rows())
    longest = max((r for r in analysis.episode_rows() if r['accepted']), key=lambda r: r['t_end_yr'] - r['t_start_yr'])
    cycles = (longest['t_end_yr'] - longest['t_start_yr']) / row['geo_period_cycle_yr']  # 19581 / 6138
    assert math.isclose(cycles, 3.19, abs_tol=0.01)
    assert '19.6 kyr = 3.2 cycles < 3.5 of one force window' in text.sentences[1]
    assert '3.3 of its own 5.93 kyr' in text.sentences[1]


def test_secular_record_over_the_whole_record():
    """1020 g-g6+s-s6 (author, 16b): the whole record librates — no 'episode' in its text."""
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
    text = vt.explain_long_period(analysis, S, analysis.episode_rows())
    assert 'over the whole record' in text.sentences[0] and 'episode' not in ' '.join(text.sentences)
    assert_comparisons_hold(text)
    fig = draw_long_period_combined(analysis)
    assert fig.text_stats['overflow'] == 0
    texts = [artist.get_text() for artist in fig.texts]
    assert sum(1 for t in texts if 'RESONANT ·' in t or 'resonant ·' in t) == 1  # the status once, in the plaque
    assert any(t.startswith('g6') for t in texts) and any('forced wave g6' in t for t in texts)
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


def _long_explanation():
    checks = [vt.CheckRow(f'M{i}', f'margin number {i}', vt.CheckState.PASS, 'detail ' * 12, 'need', vt.COUNTS_FULL) for i in range(14)]
    return vt.VerdictText('resonant', 'libration', 'high', 'reason x', ['A sentence. ' * 30] * 3, [], checks, 'raw: …')


def _draw_numbers(n_lines: int):
    from resonances.plotting import combined_numbers as numbers
    from resonances.plotting.style import LAYOUT_V1

    lines = [numbers.NumberLine('numbers', f'n{i}', str(i), group='G') for i in range(n_lines)]
    fig = plt.figure(figsize=LAYOUT_V1.size, dpi=40)
    stats = numbers.draw_explained(fig, _long_explanation(), lines, LAYOUT_V1)
    texts = [artist.get_text() for artist in fig.texts]
    plt.close(fig)
    return stats, texts


def test_long_explanation_collapses_the_full_margins():
    """When the numbers would not fit even at the smallest font in five columns, the full margins become one line of
    names (the last resort, W23 F1)."""
    stats, texts = _draw_numbers(300)
    assert stats['compact'] and stats['n_columns'] == 5 and stats['overflow'] == 0
    assert any(t.startswith('full: margin number 0; margin number 1') for t in texts)
    assert sum(1 for t in texts if 'RESONANT · libration · HIGH' in t) == 1


@pytest.mark.parametrize(
    'n_lines, n_columns, legend_beside',
    [(200, 4, False), (240, 5, True)],  # 200 fit as they are; 240 need the legend under the steps and a fifth column
)
def test_numbers_fit_by_layout_before_the_explanation_turns_compact(n_lines, n_columns, legend_beside):
    """W23 F1 (1862 2M-6J-1 was 21 lines over): the legend under the steps and a narrow fifth number column come before
    the compact explanation; every margin keeps its own line and nothing overflows."""
    stats, texts = _draw_numbers(n_lines)
    assert not stats['compact'] and stats['overflow'] == 0
    assert stats['n_columns'] == n_columns and stats['legend_beside'] == legend_beside
    assert not any(t.startswith('full: ') for t in texts)
    assert all(any(t == f'margin number {i}' for t in texts) for i in range(14))
    assert sum(1 for t in texts if t.startswith('✓ full')) == 1  # the legend, once


def test_branch_and_mark_wording(record):
    """The leaves of the branch where L4 and R0 are not measured name their rule's numbers; the mark
    exit_and_capture is listed with its counts."""
    example, analysis, _, text = record
    rule = analysis.extra_columns['rec_rule']
    status = text.sentences[1]
    if rule == '5s':
        assert 'not measured' in status and '× μ_i μ_j e^q' in status and 'B²' in status
        assert f'{vt.fmt(analysis.strength.divisor_b)}' in status
    elif rule == '5s*' and analysis.strength.divisor_kind == 'crosses_zero':
        assert 'passes its exact commensurability' in status and analysis.strength.divisor_candidate in status
    elif rule == '5s*' and analysis.strength.divisor_kind == 'no_candidate':
        # Codex W18: the no-candidate residual is a lower bound, never printed as the residual itself
        assert f'only bounded below (140 < {vt.fmt(S.mmr_strength_implausible_ratio)} ×)' in status
        assert 'cannot rule out the gap closing' in status
        assert 'leave only' not in status
    elif rule == '5n':
        assert 'one centre only (independent cycles 1 < 2)' in status
    elif rule == '5u/6':
        assert ': by the geometry alone,' in status and 'origin stays unknown' in status
        scale = str(analysis.strength.check) == 'pass' if analysis.strength is not None else False
        assert ('strength_scale_only' if scale else 'geometry_only') in analysis.verdict.caps
    if 'near_separatrix_without_zone' in analysis.verdict.caps:
        assert 'the separatrix zone is not measured' in status and 'cycles lie in the separatrix zone' not in status
    if 'exit_and_capture' in analysis.verdict.flags:
        energy = analysis.energy
        line = next(item for item in text.info if item.startswith('exit and capture'))
        assert f'{energy.n_exits} times' in line and f'came back {energy.n_captures} time' in line
        lifted = analysis.extra_columns['sep_unexplained_lifted'] == 1.0
        assert ('lifts the cap "no mechanism explains the transitions"' in line) == lifted
        if example == 'mmr2__33128__6N-7+0+1':  # the mark lifts the cap (author:2026-10-18)
            assert lifted and 'transient_unexplained' not in analysis.verdict.caps


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
    where = f'measured on witness #{witness.episode_id} ({vt.years(witness.t_start_yr)}–{vt.years(witness.t_end_yr)}'
    assert vt.measured_on(facts).startswith(where) and f'the one of {len(found)} witnesses with most' in vt.measured_on(facts)
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


# R-ALIAS-A records at 5 yr (the W22 R-ALIAS-A impact run, merged-scat+margins): both a-based checks fail as measured
# and are read as not measured; the scatter of a over the a swing 0.73 (TK14) and 0.79 (433) > 0.5
ALIAS_RECORDS = {'mmr2__2004_TK14__31M-34+0+3': 0.73, 'mmr2__433__47M-44+0-3': 0.79}


@pytest.mark.parametrize('example', sorted(ALIAS_RECORDS))
def test_alias_text_names_the_rule_that_decided(example):
    analysis, text = _explained(example)
    v = analysis.verdict
    assert analysis.extra_columns['rsp_short_period_unmeasured'] == 'a_response;identity'
    assert v.margins['L4'].margin == Margin.FAIL and v.margins['L3'].margin == Margin.FAIL  # the column keeps the fail
    assert str(v.confidence_raw) == 'high' and not v.failed_support
    status, confidence = text.sentences[1], text.sentences[2]
    assert 'potential with a carries' not in status  # R0 is not measured there (V_E below its floor)
    assert 'R-ALIAS-A' in status and f'{ALIAS_RECORDS[example]:.2f}' in status
    direct = next(e for e in v.decision_path if e.name == vt.DIRECT_STRENGTH_ENTRY)
    assert direct.result and f'the strength by direct averaging decides (required / theory {vt.fmt(direct.value)}' in status
    assert 'support test' not in confidence and 'nothing failed' in confidence
    force = next(step for step in text.steps if step.key == 'force')
    assert 'R-ALIAS-A' in force.outcome and 'strength by direct averaging decides' in force.outcome
    assert 'fails' not in force.outcome
    for key in ('L3', 'L4'):
        (row,) = [r for r in text.checks if r.key == key]
        assert row.state == vt.CheckState.MISSING and row.counts == vt.COUNTS_NOT_MEASURED
        assert row.detail.endswith(': not used)') and 'R-ALIAS-A' in row.note
    assert text.arithmetic.startswith('raw: 0 failed') and 'thin or not measured → high) = high' in text.arithmetic


def test_a_fail_the_rule_did_not_touch_stays_failed():
    """2001 ND13 1V-3: the scatter 0.09 < 0.5 keeps its failed force law (R-ALIAS-A does not apply). With one harmonic
    only (W23 CS12 off): the two-harmonic law lifts its L4 (`test_nd13_harmonic_force_law_sentence`)."""
    analysis, text = _explained('mmr2__2001_ND13__1V-3+0+2', replace(S, mmr_force_harmonics_max=1))
    assert not analysis.extra_columns['rsp_short_period_unmeasured']
    assert str(analysis.verdict.status) == 'uncertain' and 'ȧ does not follow the force law' in text.sentences[1]
    assert 'R-ALIAS-A' not in ' '.join(text.sentences + [step.outcome for step in text.steps])


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
    sentences = ' '.join(text.sentences)
    assert 'a responds to σ as a pendulum (ȧ = Σ_k [A_k sin kσ + B_k cos kσ] + C' in sentences and detail in sentences
    assert_comparisons_hold(text)


def test_promoted_union_reads_episodic_then_resonant():
    """TK14 31M-34 at 0.5 yr: three accepted episodes (episodes.csv) without a circulation between them; the tree read
    them as episodic, reached transient/dynamic_episode and rule 6u promoted the union."""
    analysis, text = _explained('mmr2__2004_TK14__31M-34+0+3__step0.5yr')
    accepted = [r for r in analysis.episode_rows() if str(r.get('accepted')).lower() in ('true', '1')]
    assert len(accepted) == 3
    steps = {step.key: step.outcome for step in text.steps}
    assert steps['shape'].startswith('3 episodes without a circulation between them cover the record (union rule')
    assert steps['zone'].startswith('episodic → transient; the union covers the record → resonant (6u)')
    assert 'through 3 episodes without a circulation between them (union rule); the deciding one' in text.sentences[0]


@pytest.mark.parametrize('example', ['mmr3__2003_TX9__3E+2J-10+0+0+5', 'mmr3__2003_TX9__3E+2J-10+0+0+5__step0.5yr'])
def test_centre_scale_union_does_not_lend_the_fine_episode(example):
    """2003 TX9 without the W23 centre-scale TP rule (with it G2c keeps the slow libration): the union was found at the
    scale of the centre only; the fine deciding episode ends before 14 kyr of a 20 kyr record, so its numbers are not
    printed as the whole record's libration."""
    # G2a off: it cuts this record at the Mars kick (W24, `test_tx9_names_the_action_step_and_its_one_potential`)
    analysis, text = _explained(example, replace(S, libration_centre_scale_tp_at_centre=False, libration_action_step=False))
    assert analysis.extra_columns['geo_union_word_centre_scale'] == 1
    decisive = next(r for r in analysis.episode_rows() if r['episode_id'] == analysis.decisive_episode_id)
    assert decisive['t_end_yr'] < 14000.0
    angle = text.sentences[0]
    assert angle.startswith('σ librates over the whole record at the scale of the centre')
    assert f'at the fine scale the deciding episode runs {vt.years(decisive["t_start_yr"])}–{vt.years(decisive["t_end_yr"])}' in angle
    shape = next(step for step in text.steps if step.key == 'shape')
    assert 'at the scale of the centre' in shape.outcome


# --------------------------------------------------------------------------------------------- W23 S7 (F5, F6)


def test_header_and_sentence_read_one_centre_test(record):
    """F5: the header's ', moving centre' and the words 'σ librates ... with a fixed centre' / 'the centre is
    stationary' come from one test (`orbit_type.centre_moves`): no figure says both of σ's own centre (a subresonance
    member's 'fixed centre', 82831, is another angle's)."""
    _, analysis, _, text = record
    words = ' '.join(text.sentences + [step.outcome for step in text.steps])
    if 'moving centre' in str(analysis.verdict.orbit_type):
        assert 'the whole record with a fixed centre' not in words and 'the centre is stationary' not in words


@pytest.mark.parametrize('example', ['mmr3__2003_TX9__3E+2J-10+0+0+5', 'mmr3__2003_TX9__3E+2J-10+0+0+5__step0.5yr'])
def test_tx9_names_its_moving_centre_and_the_scale_of_the_centre(example):
    """2003 TX9 3E+2J-10 after W23 rule (a): one episode at the scale of the centre whose cycles' bands overlap (subtype
    stationary) while its centre shifts by 1.45 (5 yr) of the mean amplitude, above MMR_L6_SHIFT_FULL 0.5: the header
    says 'moving centre', so the sentence must not say 'fixed centre' (F5). The shape step names the scale of the
    centre and the turning-point prominence it used (F6)."""
    analysis, text = _explained(example)
    v = analysis.verdict
    decisive = next(r for r in analysis.episode_rows() if r['episode_id'] == analysis.decisive_episode_id)
    shift = decisive['centre_shift_over_amp']
    assert str(v.angle_subtype) == 'stationary' and shift > S.mmr_l6_shift_full and v.orbit_type.endswith(', moving centre')
    angle = text.sentences[0]
    assert f'with a moving centre (shift / mean amplitude {vt.compare(shift, ">", S.mmr_l6_shift_full)}; every cycle' in angle
    assert 'fixed centre' not in angle
    steps = {step.key: step.outcome for step in text.steps}
    assert steps['sub'] == 'not triggered: the cycles’ bands overlap (no trigger fired)'
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    window, fine, prominence = row['geo_centre_scale_cutoff_yr'], row['geo_fine_cutoff_yr'], row['geo_tp_prominence_deg']
    assert window > fine and prominence >= S.mmr_tp_prominence_floor_deg
    assert f'at the scale of the centre: σ filtered at W_c {vt.compare(window, ">", fine, " yr")}' in steps['shape']
    assert f'its turning points are turns of the centre (prominence ≥ {vt.fmt(prominence)}°)' in steps['shape']
    assert_comparisons_hold(text)
    _, without = _explained(example, replace(S, libration_centre_scale_tp_at_centre=False))
    assert 'turns of the centre' not in ' '.join(step.outcome for step in without.steps)


@pytest.mark.parametrize('example', ['mmr3__2003_TX9__3E+2J-10+0+0+5', 'mmr3__2003_TX9__3E+2J-10+0+0+5__step0.5yr'])
def test_tx9_names_the_action_step_it_did_not_cut_and_why(example):
    """W24 G2a after the review: the step at the Mars kick passes tests 1-6 but not 7 (one potential does not hold the
    late cycles): the angle sentence names the step, the sub-test that failed with its cutoff and the worst laws, from
    the geo_action_step_* columns; the badge says what the header says (moving centre); with the rule off the text says
    nothing of G2a."""
    analysis, text = _explained(example)
    row = analysis.numbers_row()
    angle = text.sentences[0]
    assert f'Rule G2a found a step of the libration at {vt.years(row["geo_action_step_time_yr"])}' in angle
    assert "and did not cut: one potential does not hold both regimes (a cycle's period is not the loop's" in angle
    laws = f'worst-cycle T_obs/T_pred {vt.fmt(row["geo_action_step_law_before"])} before, {vt.fmt(row["geo_action_step_law_after"])} after'
    assert laws in angle and 'librates over the whole record' in angle
    assert analysis.verdict.orbit_type.endswith(', moving centre') and text.angle == 'libration (stationary bands, moving centre)'
    assert_comparisons_hold(text)
    _, without = _explained(example, replace(S, libration_action_step=False))
    assert 'G2a' not in ' '.join(without.sentences)


def test_a_libration_cut_into_regimes_is_not_said_to_cover_the_record_as_one():
    """The record of test_episode_detection (vii) through `analyse_mmr` (3N-4): G2a cuts it into two episodes, so the
    angle sentence lists both and the cut, not 'over the whole record'."""
    from tests.resonances.analysis import test_episode_detection as te

    on, _ = te.pipeline_analyses()
    text = vt.explain_mmr(on, S, {**on.numbers_row(), **on.summary_row()}, on.episode_rows())
    angle = text.sentences[0]
    assert angle.startswith('σ librates in 2 episodes, ') and 'cut by a step of its action (G2a); the deciding one' in angle
    assert 'over the whole record' not in angle and 'Rule G2a cut the libration at' in angle


def test_the_withheld_action_step_sentence_names_the_hidden_check():
    """A step the guard withheld (`analyse._action_step_guard`): the sentence
    names the check and the verdict the cut would have given."""
    row = {
        **_action_step_row(140.0, 15.0),
        'geo_action_step_n_cuts': 0,
        'geo_action_step_failed': 'withheld',
        'geo_action_step_withheld_checks': 'check_mmr_strength',
        'geo_action_step_withheld_verdict': 'resonant/low',
    }
    sentence = vt.action_step_sentence(SimpleNamespace(row=row, s=S))
    assert sentence.startswith(f'Rule G2a found a step of the libration at {vt.years(2962.5)} (140° to 15°)')
    assert 'the cut leaves check_mmr_strength, failed on the whole record, unmeasured, and would have made it resonant/low' in sentence


def _action_step_row(before_deg, after_deg, provisional=0):
    return {
        'geo_action_step_n_cuts': 1,
        'geo_action_step_failed': '',
        'geo_action_step_time_yr': 2962.5,
        'geo_action_step_t_from_yr': 2750.0,
        'geo_action_step_t_to_yr': 3175.0,
        'geo_action_step_amp_before_deg': before_deg,
        'geo_action_step_amp_after_deg': after_deg,
        'geo_action_step_a_ratio': 7.16,
        'geo_action_step_dh': -0.866 if after_deg < before_deg else 0.866,
        'geo_action_step_dh_threshold': 0.1,
        'geo_action_step_cycles_before': 2,
        'geo_action_step_cycles_after': 5,
        'geo_action_step_provisional': provisional,
    }


@pytest.mark.parametrize('before_deg, after_deg, change', [(140.0, 15.0, 'shrank 7.16×'), (15.0, 140.0, 'grew 7.16×')])
def test_the_action_step_sentence_says_which_way_the_swing_of_a_went(before_deg, after_deg, change):
    """geo_action_step_a_ratio is the larger side's swing of a over the smaller's (`geometry.action_step_of`): after a
    step down of the half-amplitude (140 -> 15 deg) the swing of a shrank by it, after a step up it grew (W24 review:
    TX9 printed '×7.16' for a swing that fell 7 times)."""
    facts = SimpleNamespace(row=_action_step_row(before_deg, after_deg), s=S)
    sentence = vt.action_step_sentence(facts)
    assert sentence.startswith('Rule G2a cut the libration at') and f'the swing of a {change},' in sentence
