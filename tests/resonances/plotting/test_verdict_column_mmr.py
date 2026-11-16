"""The right column of an MMR verdict as ten blocks (`plotting/verdict_column_mmr.py`, the redesign of 2026-11-04).

Fast tests: the words of a resonance and the helpers on hand-worked values, the flag registry against the package's
flags, the Confidence pills of the policy's level inputs (an input holds the level when its ceiling is the level), and
whole columns built from synthetic analyses (a slow arc near 2J-1, `test_angle_rule_column`; the 2:1 kinematic
synthetic, `test_mmr_combined`). Benchmark tests: the real rows the author reviewed in the mockups and the spec review
(their verdicts as of the step-1 run, 2026-11-04), each column holding its contract (`verdict_column.problems`) and the
contents the review named.
"""

import functools
import re

import matplotlib

matplotlib.use('Agg')

import pytest  # noqa: E402

from resonances.plotting import verdict_column as vc  # noqa: E402
from resonances.plotting import verdict_column_common as common  # noqa: E402
from resonances.plotting import verdict_column_mmr as vm  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.plotting import verdict_vocab as vv  # noqa: E402
from resonances.plotting import combined_numbers as numbers  # noqa: E402
from resonances.resonance.analysis.confidence import Cap  # noqa: E402
from resonances.resonance.analysis.mmr.models import ResonanceSpec  # noqa: E402
from resonances.resonance.analysis.settings import AnalysisSettings, settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import Flag, Status  # noqa: E402

S = AnalysisSettings()


def _column(analysis, settings=S):
    from resonances.plotting.mmr_combined import figure_rows

    summary, numbers_row, episodes, constants = figure_rows(analysis)
    return vm.mmr_column(analysis, settings, {**numbers_row, **summary}, episodes, constants)


def _all_text(column) -> str:
    return vc.column_text(column)


# --------------------------------------------------------------------------------------------- words


@pytest.mark.parametrize(
    'key, expected',
    [
        # n : n_J = 2 : 1, so P : P_J = 1 : 2 (sigma = 2 lambda_J - lambda ...: q = |2 - 1| = 1)
        ('2J-1+0-1', 'two-body, Jupiter 2:1 (period ratio 1:2), order 1'),
        # n : n_N = 9 : 17, the outer belt's 17:9 (q = |9 - 17| = 8)
        ('9N-17+0+8', 'two-body, Neptune 9:17 (period ratio 17:9), order 8'),
        ('3M+3J-7+0+0+1', 'three-body Mars–Jupiter, order 1'),  # q = |3 + 3 - 7| = 1
        ('1E-1+0+0', 'co-orbital 1:1 with Earth'),  # q = 0, one planet
    ],
)
def test_type_words(key, expected):
    assert vm.type_words(ResonanceSpec.from_key(key)) == expected


def test_words_drop_codes():
    assert vm.words('episodic → transient (episodes_in_separatrix_zone)') == 'episodic → transient'
    assert vm.words('B1 6e-05 < 0.1 → non_resonant') == 'B1 6e-05 < 0.1 → non-resonant'
    assert vm.words('insufficient_time (arc)') == 'insufficient time (arc)'
    assert vm.words('near_separatrix (pattern_b)') == 'near-separatrix (pattern b)'


def test_power_of_ten_and_lists():
    assert vm._power(1000.0) == '10³' and vm._power(1e6) == '10⁶'
    assert vm._power(39.5) == '39.5'
    assert vt.and_list(['force law']) == 'force law'
    assert vt.and_list(['force law', 'potential', 'strength']) == 'force law, potential and strength'


def test_split_answer_moves_a_closing_parenthesis_to_the_numbers():
    assert vm._split_answer('38 candidates, none drives a (best 5S-1+0-4: R² 0.059 < 0.7)') == (
        '38 candidates, none drives a',
        'best 5S-1+0-4: R² 0.059 < 0.7',
    )
    assert vm._split_answer('n/a: the center does not drift (libration)') == ('n/a: the center does not drift', 'libration')
    assert vm._split_answer('libration → resonant') == ('libration → resonant', '')


@pytest.mark.parametrize(
    'text, outer, expected',
    [
        # strings of the exported columns (1921 L2, 79653 step 5, 3040 SZ.H and step 0, 2020 PP1 gate cycles)
        ('0 of 322 cycles in the zone (max half-amplitude 138°)', True, '0 of 322 cycles in the zone, max half-amplitude 138°'),
        (
            'the force law is not measured (2 force windows of 3.5 cycles (step 0.5 window) (cycles of P_cyc 9.7 kyr)); '
            'the potential holds',
            False,
            'the force law is not measured (2 force windows of 3.5 cycles, step 0.5 window, cycles of P_cyc 9.7 kyr); '
            'the potential holds',
        ),
        ('η_H 0.067 ≤ 0.5 (√(1−e²) cos i kept)', True, 'η_H 0.067 ≤ 0.5, √(1−e²) cos i kept'),
        ('gaps to the record’s ends 0.36 and 0.48 P (≤ 1 covers)', True, 'gaps to the record’s ends 0.36 and 0.48 P, ≤ 1 covers'),
        ('not measured (need ≥ 3)', True, 'not measured, need ≥ 3'),
        ('x; (y z)', True, 'x; y z'),
        # a formula's own parentheses stay
        ('μ needed/μ theory = (T₀ theory/T₀)² = 2.43', True, 'μ needed/μ theory = (T₀ theory/T₀)² = 2.43'),
        ('dσ/dt = c (a − a*): the rate (slope/c 1)', False, 'dσ/dt = c (a − a*): the rate (slope/c 1)'),
        ('σ = (1/8)φ + Ψ/8', True, 'σ = (1/8)φ + Ψ/8'),
        ('no value; compare σ and p·ϖ in (j)', True, 'no value; compare σ and p·ϖ in (j)'),  # 285637 K1: a panel
        ('ȧ = K sin(σ − φ₀) (R² 0.8)', False, 'ȧ = K sin(σ − φ₀) (R² 0.8)'),
    ],
)
def test_unnest_flattens_a_gloss_inside_parentheses(text, outer, expected):
    assert common.unnest(text, outer=outer) == expected


def test_tidy_checks_drops_a_rule_id_from_the_words_and_nested_numbers():
    line = vc.CheckLine(
        vc.THIN,
        '3',
        'Does a swing with σ?',
        'yes',
        'B1 (thin)',
        'B1',
        (vc.SubCheck(vc.THIN, 'the slow center follows the perihelion term (K2)', '1 independent center estimate(s)', 'K2'),),
    )
    (tidy,) = common.tidy_checks([line])
    assert tidy.numbers == 'B1, thin'
    assert tidy.children[0].text == 'the slow center follows the perihelion term'
    assert tidy.children[0].numbers == '1 independent center estimate'


def test_an_evidence_value_keeps_a_formula_and_a_panel_in_its_statement():
    """The gray part takes glosses only (`verdict_column_common.unnest`'s rule): the identity's 'c (a − a*)' and a
    panel letter '(j)' stay in the statement, the slope goes gray."""
    assert vm.evidence_value('identity σ̇ = c (a − a*) holds (slope 1.0)') == 'identity σ̇ = c (a − a*) holds [[slope 1.0]]'
    assert vm.evidence_value('compare σ and p·ϖ (j)') == 'compare σ and p·ϖ (j)'
    assert vm._flat('the libration of X (owner search: 3 candidates)') == 'the libration of X, owner search: 3 candidates'


def test_a_flag_caveat_needs_its_status_and_its_slots():
    """`flag_caveat` fills the registry's slots; a sentence whose slot has no value, or a status the flag does not
    apply to, is not said (exit_suspected: librating statuses, slot {exit_P})."""
    sentence = vm.flag_caveat('exit_suspected', 'resonant', {'exit_P': '0.5'})
    assert sentence.startswith('The exit lies 0.5 libration periods before the record’s end')
    assert vm.flag_caveat('exit_suspected', 'resonant', {}) == ''
    assert vm.flag_caveat('exit_suspected', 'non_resonant', {'exit_P': '0.5'}) == ''
    assert vm.flag_caveat('not_a_flag', 'resonant', {}) == ''
    assert vm.slots({'exit_suspected': 'True', 'exit_remaining_P': 0.5, 'regime_start_yr': 12_000.0}) == {
        'jump': '12 kyr',
        'exit_P': '0.5',
    }


def test_the_caveat_registry_names_real_flags_and_statuses():
    flags = {str(flag) for flag in Flag} | {str(cap) for cap in Cap}  # every former cap is a flag now
    statuses = {str(status) for status in Status}
    assert set(vv.FLAG_CAVEATS) <= flags
    assert set(vv.MEMBER_CAVEATS) <= set(vv.FLAG_CAVEATS)
    for flag, (applies, sentence) in vv.FLAG_CAVEATS.items():
        assert set(applies) <= statuses, flag
        assert sentence.endswith('.') and not vc.CODE_NAME.search(sentence), flag
    # level inputs are pills, never caveats (the spec review's flags table)
    for pill in ('measured_objection', 'veto_thin', 'second_capture_unresolved', 'filter_sensitive', 'owner_probable'):
        assert pill not in vv.FLAG_CAVEATS


def test_the_decisive_line_names_its_numbers_in_words():
    line = vt.Emphasis(
        vt.EMPHASIS_DECISIVE,
        'L4',
        'a obeys the force law',
        (vt.EmphasisPart('rsp_force_R2_h1', '0.823 ≥ 0.7'), vt.EmphasisPart('rsp_force_R2_h2', '0.814 ≥ 0.7')),
    )
    assert vm.decisive_line(line, ()) == (
        'a obeys the force law [L4]: force law R², first half 0.823 ≥ 0.7, force law R², second half 0.814 ≥ 0.7'
    )
    conflict = vt.Emphasis(
        vt.EMPHASIS_CONFLICT, 'L3', 'σ̇ follows a − a*', (vt.EmphasisPart('rsp_identity_corr', '0.777'),), vt.OBJECTION_NOTE
    )
    assert vm.decisive_line(conflict, ('identity',)) == 'fails, read as an objection: σ̇ follows a − a* [L3]: σ̇ against a: correlation 0.777'
    assert vm.decisive_line(conflict, ()).startswith('fails, not read: ')
    missing = vt.Emphasis(vt.EMPHASIS_UNMEASURED, 'L4', 'a obeys the force law', (), 'not measured (half-amplitude 0.783° < 45°)')
    assert vm.decisive_line(missing, ()) == 'not measured: a obeys the force law [L4]: half-amplitude 0.783° < 45°'


@pytest.mark.parametrize(
    'value, expected',
    [
        (999.95, '1 kyr'),  # 3 significant digits round it to 1000 yr: the unit follows the rounded value
        (999.4, '999 yr'),
        (999_950.0, '1 Myr'),  # 999.95 kyr rounds to 1000 kyr
        (19_581.0, '19.6 kyr'),  # unchanged: 19 581 / 1000 = 19.581 -> 19.6
        (193.4, '193 yr'),
    ],
)
def test_years_never_prints_an_exponent(value, expected):
    assert vt.years(value) == expected


@pytest.mark.parametrize(
    'text, expected',
    [
        ('kinematic: slow sigma follows p varpi', 'kinematic: slow σ follows p ϖ'),  # 523899 3N-5's K1 ledger entry
        ('8*sigma-1*phi', '8σ − φ'),  # an own_psi_def: 8σ − 1·φ
        ('5*sigma-27*phi', '5σ − 27φ'),
        ('Omega - varpi_5', 'Ω − ϖ₅'),
        ('μ_req/μ_th = (T0_th/T_obs)² = 905', 'μ needed/μ theory = (T₀ theory/T₀)² = 905'),
        ('zone half-width 2ω0/|c|', 'zone half-width 2ω₀/|c|'),
        ('B5 ratio -0.0753', 'B5 ratio -0.0753'),  # a number keeps its sign
    ],
)
def test_symbols_replace_the_ascii_names(text, expected):
    assert vm.symbols(text) == expected


def test_an_evidence_value_is_short_with_its_numbers_grey():
    """1921 2J-1's own-period and strength rows (verdict_text `_direct_rows`): the details in parentheses go gray, a
    formula's own parentheses stay, nothing nests and the gray part never ends in ']' (the markup would close early)."""
    period = (
        'T0_th 344–386 yr (141 yr–99.8 kyr: 3 snapshots, 3 of 3 in the well, ⟨e⟩ 0.373–0.485) against T0 214 yr from the '
        'observed cycles (episode #0)'
    )
    assert vm.evidence_value(period) == (
        'T₀ theory 344–386 yr against T₀ 214 yr from the observed cycles '
        '[[141 yr–99.8 kyr: 3 snapshots, 3 of 3 in the well, ⟨e⟩ 0.373–0.485; episode #0]]'
    )
    strength = 'μ_req 0.000256 against μ_th 7.6e-5–0.000105: μ_req/μ_th = (T0_th/T_obs)² = 2.43–3.37 overlaps [0.1, 10]'
    value = vm.evidence_value(strength)
    assert (
        value
        == 'μ needed 0.000256 against μ theory 7.6e-5–0.000105 [[μ needed/μ theory = (T₀ theory/T₀)² = 2.43–3.37 overlaps band 0.1–10]]'
    )
    assert vc.plain(value).count('(') == vc.plain(value).count(')')
    assert vm.evidence_value('owner search found none') == 'owner search found none'
    # a title's gloss (verdict_text `cycles_section`: '<what> of <owner> (owner search: N candidates)') does not nest
    section = vt.PhysicsSection('What makes the cycles?', 'the libration of 5E-16+0+11 (owner search: 3 candidates)', [])
    assert vm._evidence_title(section, '5J+6S-2+0+0-9') == (
        'Evidence: what makes the cycles? [[the libration of 5E-16+0+11, owner search: 3 candidates]]'
    )


# --------------------------------------------------------------------------------------------- the Confidence pills


def test_a_pill_holds_the_level_only_when_its_ceiling_is_the_level():
    """79653's inputs at level high: N and core allow certain (✓), M not measured holds it (?), S thin holds it (⚠)."""
    row = {'pol_level_inputs': 'N:4.12>=3:certain;core:4>=3:certain;M:not_measured:high;S:thin:high;V:clean:certain'}
    pills = {pill.key: pill for pill in common.pills(row, 'high')}
    assert pills['N'].icon == vc.OK and pills['core'].icon == vc.OK and pills['V'].icon == vc.OK
    assert pills['M'].icon == vc.MISSING and pills['M'].text == 'separatrix margin not measured'
    assert pills['S'].icon == vc.THIN and pills['S'].text == 'filter-sensitive' and pills['S'].ceiling == 'high'
    block = common.confidence(row, 'high')
    assert block.because == 'not certain because separatrix margin not measured, filter-sensitive'


def test_the_cycle_pills_say_how_they_count_and_which_level_their_bar_opens():
    """1866 ZLK (pol_level_inputs 'N:1.99<2:medium;core:1.5<3:high', confidence_basis N=1.99(episode)): N is held at
    medium by the bar CONFIDENCE_HIGH_CYCLES 2 (so 'high needs ≥ 2'), core only by CONFIDENCE_CERTAIN_CORE_CYCLES 3; the
    two counts differ by the covered edges, which the glosses say."""
    row = {'pol_level_inputs': 'N:1.99<2:medium;core:1.5<3:high;M:full:certain', 'confidence_basis': 'N=1.99(episode);M=full'}
    pills = {pill.key: pill for pill in common.pills(row, 'medium')}
    assert pills['N'].icon == vc.THIN
    assert pills['N'].text == '1.99 cycles counted [[from the deciding episode’s turning points, covered edges added; high needs ≥ 2]]'
    assert pills['core'].icon == vc.QUIET
    assert pills['core'].text == (
        '1.5 cycles between the deciding episode’s turning points [[edges not added; certain needs ≥ 3]]'
        ' — alone it would cap the level at high'
    )
    assert common.confidence(row, 'medium').because == 'not certain because 1.99 cycles counted'


def test_the_circulation_pill_names_what_the_level_reads():
    """847 g−g6+s−s6 (10 Myr): the center passes 1.92 turns at every smoothing (SZ.P full), yet the circulation is not
    clear: 239 turning points at the cutoff ×½ (geo_n_tp_cut_half), none at ×1 and ×2 (`policy._long_circulation_clear`).
    The pill says that, not 'not clear at every scale' beside a ✓ 'at every smoothing'."""
    row = {
        'pol_level_inputs': 'candidate:none:certain;circulation:not_clear:high',
        'geo_n_tp': 0,
        'geo_n_tp_cut_half': 239,
        'geo_n_tp_cut_double': 0,
    }
    (pill,) = [pill for pill in common.pills(row, 'high') if pill.key == 'circulation']
    assert (pill.icon, pill.text) == (vc.THIN, 'σ turns back at some filter cutoff [[turning points: 239 at cutoff ×½]]')
    assert common.confidence(row, 'high').because == 'not certain because σ turns back at some filter cutoff'
    # every cutoff clean: what is missing is the centre's whole turns (P thin)
    no_turns = {**row, 'geo_n_tp_cut_half': 0, 'resonance_type': 'secular', 'mrg_P': 'thin'}
    assert common.pill_text('circulation', 'not_clear', '', no_turns) == 'its center passes too few whole turns for a clear circulation'
    assert common.pill_text('circulation', 'clear', '', row) == 'clear circulation [[no turning point at any cutoff]]'


def test_an_unfitted_coorbital_glances_at_the_shape_not_the_free_split():
    """2020 PP1 ZLK, a co-orbital of Earth (zlk_gate_detail coorbital_unfitted_forcing): its free ω and free e vector
    (g, j) are the split the column calls no measurement, so the V pill points at (a) and (d); any other ZLK row keeps
    (g), (j)."""
    v = vc.Pill(vc.MISSING, 'free-ω gate: no verdict', 'V', 'medium')
    assert common.glance_letters('zlk', [v], {'zlk_gate_detail': 'coorbital_unfitted_forcing'}) == ['a', 'd']
    assert common.glance_letters('zlk', [v], {'zlk_gate_detail': 'node_unresolved'}) == ['g', 'j']
    assert common.glance_letters('secular', [v], {'zlk_gate_detail': 'coorbital_unfitted_forcing'}) == ['g', 'j']


@pytest.mark.parametrize(
    'key, detail, fails',
    [
        ('L6', 'shift/amplitude 5.77 > 0.5', True),  # 164207: the centre moves
        ('L6', 'shift/amplitude 0.0651 ≤ 0.5', False),  # 1921
        ('L7', 'larger edge gap 0.954 > 0.7 P', True),
        ('L8a', 'turn per step 17.9° ≥ 10°; σ turns more than the limit per output step', True),  # 2013 AS27
        ('L8a', 'turn per step 2.55° < 10°', False),
        ('L8a', 'turn per step 365°, not graded: ⟨a⟩ 66.2 ≥ 10 a-scales from a*', False),  # 1999 UQ: nothing compared
        ('L5', 'T0 max/min 2.06 ≤ 3.28', False),  # 2003 UG22: ⚠ with a passing comparison keeps its name
        ('O1', 'a-wave before/inside 1.03 ≥ 0.5; coherent at P: R² 0.952 ≥ 0.5', True),  # 2897
        ('W', 'best 5S-1+0-4 R² 0.059 < 0.4', False),  # no passing operator on file for W: read by its state
    ],
)
def test_a_failed_comparison_is_read_from_the_printed_operator(key, detail, fails):
    assert vm._printed_fails(key, detail) is fails


def test_a_failed_margin_is_named_by_what_was_measured():
    """The sub-check of a failed comparison says what the numbers show ('center moves (5.77 > 0.5)', not 'center
    stationary'), whatever its icon; a passing one keeps its name; a ⚠ with a passing comparison is not renamed."""
    analysis = _slow_arc()
    f = vt.MMRFacts(analysis, S, {}, [])
    assert vm._reading_name(f, 'L6', 'center stationary', vc.THIN, 'shift/amplitude 5.77 > 0.5') == 'center moves'
    assert vm._reading_name(f, 'L6', 'center stationary', vc.OK, 'shift/amplitude 0.0651 ≤ 0.5') == 'center stationary'
    assert vm._reading_name(f, 'L5', 'one pendulum: T₀ stable', vc.THIN, 'T0 max/min 2.06 ≤ 3.28') == 'one pendulum: T₀ stable'
    assert vm._reading_name(f, 'O1', 'no a-wave', vc.QUIET, 'a-wave before/inside 1.03 ≥ 0.5') == 'a swings already before the libration'
    assert (
        vm._reading_name(f, 'L2', 'cycles away from the separatrix', vc.THIN, '5 of 5 cycles in the zone') == 'cycles near the separatrix'
    )
    assert vm._reading_name(f, 'L3', 'the rate follows a', vc.FAIL, 'slope/c -2.9, corr -0.307').startswith('dσ/dt ≠ c (a − a*)')
    assert vm._reading_name(f, 'L3', 'the rate follows a', vc.THIN, 'slope/c 0.9, corr 0.95') == 'the rate follows a'


def test_years_from_ten_thousand_print_in_kyr():
    """Numbers in one unit per scale: 64409 yr is 64.41 kyr (385458 300 kyr), 1e5 yr 100 kyr, 2.5e7 yr 25 Myr; a cell
    below 10⁴ yr and a column in another unit are as before."""
    assert numbers.cell_with_unit('ep_period_med_yr', 64409.0) == '64.41 kyr'
    assert numbers.cell_with_unit('smp_span_yr', 100000.0) == '100 kyr'
    assert numbers.cell_with_unit('smp_span_yr', 2.5e7) == '25 Myr'
    assert numbers.cell_with_unit('flt_cutoff_yr', 75.0) == '75 yr'
    assert numbers.cell_with_unit('ep_half_amp_med_deg', 128.4) == numbers.format_cell('ep_half_amp_med_deg', 128.4) + ' °'


def test_an_eccentricity_of_one_or_more_marks_the_forced_e_model():
    """1997 GK3 3M+5S-3 (e 0.26–0.30, planet-crossing): free e 1.857 and forced e 2.127 are no ellipse's, so the linear
    model is said to fail there, and its predicted half-swing comes from it; below 1 nothing is added."""
    rows = ((numbers.column_words('kin_e_free'), '1.857'), (numbers.column_words('kin_pred_swing_deg'), '303.9 °'))
    marked = dict(vm._kinematics_rows({'kin_e_free': 1.857, 'kin_e_forced': 2.127}, rows))
    assert marked[numbers.column_words('kin_e_free')] == '1.857 (≥ 1: the linear forced-e model fails here)'
    assert marked[numbers.column_words('kin_pred_swing_deg')] == '303.9 ° (from that model)'
    assert vm._kinematics_rows({'kin_e_free': 0.333, 'kin_e_forced': 0.171}, rows) == rows


# --------------------------------------------------------------------------------------------- synthetic columns


@functools.lru_cache(maxsize=None)
def _slow_arc():
    from tests.resonances.analysis.test_angle_rule_column import analysed, slow_swing

    return analysed(slow_swing())


@functools.lru_cache(maxsize=None)
def _kinematic():
    from tests.resonances.plotting.test_mmr_combined import _synthetic

    return _synthetic()[0]


def test_a_slow_arc_is_decided_at_step_0_by_its_angle_rule():
    analysis = _slow_arc()
    column = _column(analysis)
    assert vc.problems(column) == []
    assert (column.header.status, column.header.level, column.header.chip.kind) == ('uncertain', '', vc.CHIP_EXTEND)
    decides = [line for line in column.checks if line.decides]
    assert len(decides) == 1 and decides[0].number == '0' and decides[0].rule_id == 'G5a' == column.decided_rule
    assert column.checks[0].number == vm.QUALITY_NUMBER  # Q first
    assert 'leaf G6 · angle rule G5a' in ' · '.join(column.footer)
    assert column.subtitle.title == '2J-1 · too short to decide: one slow arc'
    assert column.subtitle.detail.startswith('two-body, Jupiter 2:1 (period ratio 1:2), order 1 · decided at step 0 (angle shape')
    assert column.confidence.level == '' and column.confidence.pills  # what is missing, as '?' pills
    assert all(pill.icon == vc.MISSING for pill in column.confidence.pills)
    # one slow arc: what is missing is the slow swing closing, never 'the libration at the end' (no libration yet)
    assert column.confidence.pills[0].text.startswith('the slow swing must close: 3 slow periods')
    assert not any('libration at the end' in pill.text for pill in column.confidence.pills)
    # steps 1-6 are not reached: one quiet line after the ◆ at step 0
    rest = column.checks[-1]
    assert (rest.number, rest.quiet, rest.icon) == ('1–6', True, vc.QUIET) and rest.answer.startswith('— not reached')
    # G5a compares nothing at the slow scale: Evidence A is quiet (no cycles of a rejected run)
    assert not any(block.title.startswith('Evidence: is this') for block in column.evidence)


def test_a_kinematic_synthetic_names_its_term_and_the_veto():
    analysis = _kinematic()
    column = _column(analysis)
    v = analysis.verdict
    assert vc.problems(column) == []
    assert (str(v.status), str(v.reason), str(v.veto)) == ('non_resonant', 'kinematic_libration', 'kinematic')
    assert column.subtitle.title.startswith('2J-1+0-1 · kinematic (forced) libration: bounded by −ϖ')
    decides = next(line for line in column.checks if line.decides)
    assert decides.number == '3' and decides.rule_id == '3' == column.decided_rule
    rest = column.checks[-1]
    assert rest.quiet and rest.icon == vc.QUIET and 'not reached' in rest.answer
    assert column.confidence.level == 'high' and column.confidence.pills[0].key == 'veto'
    # every number of Numbers comes with a label in words: one of the tables, never a column stem
    words = set(numbers.COLUMN_LABELS.values()) | set(numbers.COLUMN_WORDS.values()) | set(numbers.RANGE_LABELS.values())
    words |= {'', 'e', 'i', 'q_min / Q_max', 'orbital period'}
    for group in column.numbers.groups:
        if group.title in ('REFERENCE PERIODS',):  # labelled by `mmr_verdict_panel.period_lines`
            continue
        for label, _ in group.rows:
            assert label in words and not vc.CODE_NAME.search(label), (group.title, label)


def test_flag_caveats_follow_the_registry_and_the_status():
    """A flag of the registry on its status is said, with its slot from the row; on another status, or with its slot
    missing, it is not."""
    from dataclasses import replace

    analysis = _slow_arc()
    flags = ('last_regime_after_jump', 'transient_unexplained', 'zone_eq_only')
    analysis = replace(analysis, verdict=replace(analysis.verdict, flags=flags))
    row = {**analysis.numbers_row(), **analysis.summary_row(), 'regime_start_yr': 12000.0}
    facts = vt.MMRFacts(analysis, S, row, analysis.episode_rows())
    # uncertain: the jump sentence (any status) with 12 kyr; transient_unexplained only on transient rows; zone_eq_only
    # is a diagnostic (footer only)
    assert vm.flag_caveats(facts, row) == [
        'a jumps at 12 kyr; the record before and after the jump is measured as two regimes, and the one after it gives the '
        'final regime.'
    ]
    row.pop('regime_start_yr')
    assert vm.flag_caveats(vt.MMRFacts(analysis, S, row, analysis.episode_rows()), row) == []


def test_edge_caveats_print_the_distance_in_periods():
    """exit_suspected / entry_suspected are said with the record left after the exit (before the entry) in the
    episode's periods; 2019 GM1 1E-1 (test-benchmark, 2026-11-04): exit_remaining_P 0.159977, entry_remaining_P
    0.118776 -> '0.16', '0.119'; after a cut the record's entry (2025 PN7 20 kyr: record_entry_remaining_P 0.284501).
    Without the number the sentence is not said."""
    from dataclasses import replace

    analysis = _slow_arc()
    verdict = replace(analysis.verdict, status=Status.TRANSIENT, flags=('exit_suspected', 'entry_suspected'))
    analysis = replace(analysis, verdict=verdict)
    base = {**analysis.numbers_row(), **analysis.summary_row()}
    row = {**base, 'exit_suspected': True, 'exit_remaining_P': 0.159977, 'entry_suspected': True, 'entry_remaining_P': 0.118776}
    said = vm.flag_caveats(vt.MMRFacts(analysis, S, row, analysis.episode_rows()), row)
    assert said == [
        'The exit lies 0.16 libration periods before the record’s end, so the libration may just continue past the record.',
        'The entry lies 0.119 libration periods after the record’s start, so the libration may have begun before the ' 'record.',
    ]
    row = {**base, 'exit_suspected': False, 'entry_suspected': False, 'record_entry_suspected': True}
    row['record_entry_remaining_P'] = 0.284501
    said = vm.flag_caveats(vt.MMRFacts(analysis, S, row, analysis.episode_rows()), row)
    assert said == [
        'The entry lies 0.285 libration periods after the record’s start, so the libration may have begun before the ' 'record.'
    ]


# --------------------------------------------------------------------------------------------- benchmark rows

SB = settings_from_config()


@functools.lru_cache(maxsize=None)
def _real(example: str):
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), SB)
    return analysis, _column(analysis, SB)


EXAMPLES = (
    'mmr2__1921__2J-1+0-1',
    'mmr3__2001_HW7__5J+6S-2+0+0-9',
    'mmr3__79653__3M+3J-7+0+0+1',
    'mmr2__385458__9N-17+0+8',
    'mmr3__2012_KL9__5E-2M-10+0+0+7',
    'coorbital__2016_JA__1E-1+0+0__5kyr',
    'mmr2__523899__3N-5+0+2',
    'mmr3__285637__3M+7J-6+0+0-4',
    'mmr3__1999_UQ__7E+2M-9+0+0+0',
    'mmr3__142694__1V+3S-7+0+0+3',
    'coorbital__2007_VU6__1E-1+0+0',
    'mmr2__2897__35S-4+0-31',
)


def _decides(column) -> vc.CheckLine:
    return next(line for line in column.checks if line.decides)


def _step(column, number: str) -> vc.CheckLine:
    return next(line for line in column.checks if line.number == number)


def _pill(column, key: str) -> vc.Pill:
    return next(pill for pill in column.confidence.pills if pill.key == key)


@pytest.mark.benchmark
@pytest.mark.parametrize('example', EXAMPLES)
def test_every_example_holds_the_contract(example):
    analysis, column = _real(example)
    v = analysis.verdict
    assert vc.problems(column) == []
    assert (column.header.status, column.header.level) == (str(v.status), str(v.confidence or ''))
    assert column.checks[0].number == vm.QUALITY_NUMBER
    decides = _decides(column)
    assert decides.rule_id == column.decided_rule
    footer = ' · '.join(column.footer)
    if column.decided_rule == vm.POLICY_NUMBER:
        assert f'decided by {analysis.numbers_row().get("pol_decided_by") or analysis.summary_row()["pol_decided_by"]}' in footer
    else:
        assert column.decided_rule in footer  # the leaf or the angle rule the footer names
    assert 'W28 tree' not in footer  # not printed (the author 2026-11-14)
    # Evidence in symbols, never the ledger's ASCII names; its grey numbers never nest parentheses in parentheses
    for block in column.evidence:
        for item in block.rows:
            assert not any(name in item.value for name in ('T0_th', 'mu_req', 'μ_req', 'μ_th', '*sigma', 'T_obs')), item
    for block in column.evidence:  # a gloss inside the grey numbers is unnested: never '…))' nor '((…'
        assert not any('))' in vc.plain(item.value) or '((' in vc.plain(item.value) for item in block.rows), block
    # nor in a title's gloss or a key fact ('(the libration of X (owner search: …))', '(libration (stationary))')
    for text in [block.title for block in column.evidence] + [fact.value for fact in column.key_facts]:
        assert '))' not in vc.plain(text), text
    # nor in the grey numbers of Checks (`common.unnest`): never '(…(gloss))'
    for line in column.checks:
        for numbers_ in [line.numbers] + [child.numbers for child in line.children]:
            assert '((' not in numbers_ and not re.search(r'\([^()]*\([^()]*\)[^()²/\w]*\)', f'({numbers_})'), numbers_


@pytest.mark.benchmark
def test_1921_is_the_clean_reference():
    _, column = _real('mmr2__1921__2J-1+0-1')
    assert (column.header.status, column.header.level, column.header.chip.kind) == ('resonant', 'certain', vc.CHIP_NONE)
    assert column.subtitle.title == '2J-1+0-1 · its own libration over the whole record'
    assert _decides(column).number == '6' and column.decided_rule == '5d/6'
    assert all(pill.icon == vc.OK for pill in column.confidence.pills)
    assert column.caveats == ()
    assert column.evidence[0].title.startswith('Evidence: is this 2J-1+0-1’s own pendulum?')
    # one R² (0.059) against two bars, each named: MMR_OWNER_R2_MIN 0.7 at step 2, MMR_W_R2_NONE 0.4 in the margin
    step = _step(column, '2')
    assert step.numbers.endswith('R² 0.059 < 0.7 needed to accept an owner')
    (w,) = [child for child in step.children if child.rule_id == 'W']
    assert w.numbers.endswith('R² 0.059 < 0.4, too low even for a partial fit')


@pytest.mark.benchmark
def test_hw7_is_induced_and_held_at_medium_by_the_second_capture():
    _, column = _real('mmr3__2001_HW7__5J+6S-2+0+0-9')
    assert column.subtitle.title == '5J+6S-2+0+0-9 · induced: borrowed from 5E-16+0+11'
    assert _decides(column).number == '2'
    rest = column.checks[-1]
    assert rest.quiet and rest.number == '3–6' and rest.question == 'a swings · drift · own force · behavior'
    pill = _pill(column, 'second_capture')
    assert (pill.icon, pill.ceiling, pill.text) == (vc.THIN, 'medium', 'both resonances at once: not ruled out')
    assert column.header.chip.kind == vc.CHIP_GLANCE
    # the veto's pill says what was measured (the owner's F7 cover and P2 hold: `owner_confirmed`), not 'the owner found'
    assert _pill(column, 'veto').text == 'driven by 5E-16+0+11: measured'


@pytest.mark.benchmark
def test_79653_transient_high_with_the_strength_scale_grey():
    _, column = _real('mmr3__79653__3M+3J-7+0+0+1')
    assert (column.header.status, column.header.level) == ('transient', 'high')
    assert column.header.chip.text == 'extend to 200 kyr'
    assert _pill(column, 'M').text == 'separatrix margin not measured' and _pill(column, 'M').icon == vc.MISSING
    # S names the part that fails (`policy._mmr_stable`: filter_sensitive, L1 is full here)
    assert _pill(column, 'S').text == 'filter-sensitive [[a deciding number changes at cutoff ×2]]'
    assert _pill(column, 'S').icon == vc.THIN
    assert column.confidence.because == 'not certain because separatrix margin not measured, filter-sensitive'
    strength = next(child for child in _step(column, '5').children if child.rule_id == 'F')
    assert strength.icon == vc.QUIET
    assert strength.text.startswith('strength, order of magnitude: scale only (×39.5; an objection only from ×10³')
    assert strength.numbers == ''  # the ratio is in its words; the margin's band [0.1, 10] is no bar of a scale
    # K2 is not measured here since 2026-11-14 (one force window, 1.4 independent center points < MMR_KIN_MIN_POINTS):
    # no K2 condition under 3 (its display: test_k2_near_threshold_condition_is_named_by_what_it_compared)
    assert not [child for child in _step(column, '3').children if child.rule_id == 'K2']
    assert any(c.startswith('The three-body strength is known only as the scale') for c in column.caveats)
    assert _decides(column).number == '6'
    # the filter guard withdrew the zone (Z, L2 not measured; flag episodes_in_separatrix_zone gone): step 6 follows
    # the surviving flags, not the energy's raw reading '7 of 7 cycles in the separatrix zone and the angle crosses it'
    analysis, _ = _real('mmr3__79653__3M+3J-7+0+0+1')
    assert 'episodes_in_separatrix_zone' not in analysis.verdict.flags and 'zone_not_measured' in analysis.verdict.flags
    step = _decides(column)
    assert step.answer.endswith('zone not measured') and step.numbers == 'withdrawn by the filter guard'
    assert 'separatrix zone' not in step.answer and 'crosses' not in step.answer


@pytest.mark.benchmark
def test_k2_near_threshold_condition_is_named_by_what_it_compared():
    """1998 QA62 1V-5M+3: K2 is measured (8 force windows) and stops at the centre correlation; its force law R² 0.361
    is within the thin band of 0.4, so step 3 names that condition by what it compared, tagged K2."""
    _, column = _real('mmr3__1998_QA62__1V-5M+3+0+0+1')
    (k2,) = [child for child in _step(column, '3').children if child.rule_id == 'K2']
    assert 'a does not answer σ, force law R²' in k2.text and '(K2)' not in k2.text and k2.numbers.startswith('0.361 < 0.4')


@pytest.mark.benchmark
def test_385458_names_the_leaf_and_the_angle_rule_apart():
    _, column = _real('mmr2__385458__9N-17+0+8')
    assert 'leaf G6 · angle rule G5w' in ' · '.join(column.footer)
    decides = _decides(column)
    assert decides.number == '0' and decides.rule_id == 'G5w' == column.decided_rule
    assert column.header.level == '' and column.confidence.level == ''
    assert column.subtitle.title == '9N-17+0+8 · too short to decide: slow swing in the well'
    # what is missing in the words of the slow swing, with the extension's length (extend_to_years 300 kyr)
    assert [pill.text for pill in column.confidence.pills] == ['the slow swing must close: 3 slow periods, a record of 300 kyr']
    # steps 1-6 collapsed into one quiet line
    assert (column.checks[-1].number, column.checks[-1].icon) == ('1–6', vc.QUIET)
    # Evidence A speaks of the slow swing (ledger G5w: period ratio 0.9655, a ratio 1.0128; T0_th 47 657 yr, the period
    # 76 258 yr), never of the 1.55-kyr rejected run against T0_th
    analysis, _ = _real('mmr2__385458__9N-17+0+8')
    row = analysis.numbers_row()
    assert vt.fmt(row['geo_slow_swing_period_ratio']) == '0.966' and vt.years(row['geo_slow_swing_t0_theory_yr']) == '47.7 kyr'
    block = column.evidence[0]
    assert block.title.startswith('Evidence: is the slow swing 9N-17+0+8’s own pendulum?')
    rows = {item.label: item for item in block.rows}
    assert rows['own period'].icon == vc.OK
    assert rows['own period'].value.startswith('slow swing 76.3 kyr = 0.966 × T₀ theory F(A) [[T₀ theory 47.7 kyr')
    assert rows['swing in a'].value.startswith('a swings 1.01 × the pendulum’s')
    assert rows['σ̇ follows a'].value == 'the slow identity holds [[|slope/c − 1| 0.00144 ≤ 0.2, corr 1 ≥ 0.9]]'
    assert '1.55 kyr' not in vc.column_text(column).split('— Caveats')[0]


@pytest.mark.benchmark
def test_kl9_circulates_and_its_runs_are_the_forced_wave():
    _, column = _real('mmr3__2012_KL9__5E-2M-10+0+0+7')
    assert column.subtitle.title == '5E-2M-10+0+0+7 · σ circulates; its short runs are the background’s forced wave'
    said = ' '.join([column.subtitle.title, column.subtitle.detail, *column.why.sentences])
    assert 'librates' not in said and 'σ librates' not in said
    decides = _decides(column)
    assert decides.number == vm.POLICY_NUMBER and decides.question == 'Is it the background’s forced wave?'
    leaf = _step(column, '5')
    assert leaf.icon == vc.FAIL and 'overruled: the policy decides at step P' in leaf.answer
    assert column.numbers.decisive[0].startswith('is it the background’s forced wave [P]: counted episodes the background')
    assert column.numbers.decisive[1].startswith('not upheld: strength implausible for this period [F]')


@pytest.mark.benchmark
def test_2016_ja_is_resonant_low_by_the_objection():
    _, column = _real('coorbital__2016_JA__1E-1+0+0__5kyr')
    assert (column.header.status, column.header.level) == ('resonant', 'low')
    assert _pill(column, 'V').icon == vc.FAIL and _pill(column, 'V').text.startswith('objection: the force law fails')
    assert _decides(column).number == vm.POLICY_NUMBER
    # the overruled leaf shows its own result (the force law fails: ✗), never '–' (the legend's 'not needed')
    assert 'overruled: the policy decides at step P' in _step(column, '5').answer and _step(column, '5').icon == vc.FAIL
    assert column.family == vm.FAMILY_COORBITAL and column.subtitle.detail.startswith('co-orbital 1:1 with Earth')
    # what V read: leaf 5a, R0 failed for the tree (the 1:1 energy integral: R² 0.586 passes 0.4, its scale K_E/K_th
    # -0.0457 is outside [1/3, 3]); the filter guard emptied the published check afterwards. Step 5 and DECISIVE say it.
    analysis, _ = _real('coorbital__2016_JA__1E-1+0+0__5kyr')
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    assert row['pol_objection_check'] == 'force_law;potential' and row['check_mmr_potential'] == 'not_measured'
    scale = row['rsp_coorb_energy_K_over_K_theory']
    assert not (1 / 3 <= scale <= 3) and row['rsp_coorb_energy_R2'] >= 0.4
    outside = f'its scale K_E/K_th {vt.fmt(scale)} lies outside [0.333, 3]'
    step = _step(column, '5')
    assert step.answer.startswith('the energy does not fit the potential from a') and 'not measured' not in step.answer
    assert outside in step.answer
    r0 = next(line for line in column.numbers.decisive if '[R0]' in line)
    assert r0.startswith('fails, read as an objection: energy fits the potential [R0]') and r0.endswith(outside)


@pytest.mark.benchmark
def test_523899_reads_the_identity_failure_as_its_level():
    _, column = _real('mmr2__523899__3N-5+0+2')
    pill = _pill(column, 'V')
    assert (pill.icon, pill.text) == (vc.FAIL, 'objection: σ̇ does not follow a (identity fails)')
    assert not any('identity' in caveat for caveat in column.caveats)  # read by the level: a pill, not a caveat
    assert any(line.startswith('fails, read as an objection: σ̇ follows a − a* [L3]') for line in column.numbers.decisive)
    # the K1 ledger entry '3 K1 kinematic: slow sigma follows p varpi' named by its rule, in symbols
    k1 = next(child for child in _step(column, '3').children if child.rule_id == 'K1')
    assert k1.text == 'slow σ follows pϖ near its convention'


@pytest.mark.benchmark
def test_285637_is_bounded_by_its_perihelion_term():
    _, column = _real('mmr3__285637__3M+7J-6+0+0-4')
    assert column.subtitle.title.endswith('kinematic (forced) libration: bounded by −4ϖ, not this resonance’s force')
    assert _decides(column).number == '3' and _pill(column, 'veto').text == 'overruled by the kinematic bound'


@pytest.mark.benchmark
def test_1999_uq_is_an_alias():
    _, column = _real('mmr3__1999_UQ__7E+2M-9+0+0+0')
    assert (column.header.status, column.header.level) == ('non_resonant', 'certain')
    decides = _decides(column)
    assert decides.number == vm.CIRCULATION_NUMBER and decides.rule_id == 'C5'
    assert 'an alias' in column.subtitle.title
    # the alias leaf's margin: full at MMR_FAR_FROM_RES_FACTOR (10), while the leaf needs MMR_C5_FAR_RATIO_MIN (3)
    c5 = next(child for child in column.checks[0].children if child.rule_id == 'C5')
    assert c5.text == 'margin of the alias leaf: far from a*, one branch'
    assert c5.numbers.startswith('⟨a⟩ 66.2 ≥ 10 a-scales from a* for a full margin, the leaf needs ≥ 3;')


@pytest.mark.benchmark
def test_142694_drift_says_what_is_missing():
    _, column = _real('mmr3__142694__1V+3S-7+0+0+3')
    assert column.header.status == 'uncertain' and _decides(column).number == '4'
    assert 'a mechanism for the drift of the center' in column.confidence.because


@pytest.mark.benchmark
def test_2007_vu6_is_one_leg():
    _, column = _real('coorbital__2007_VU6__1E-1+0+0')
    decides = _decides(column)
    assert decides.number == vm.CIRCULATION_NUMBER and decides.rule_id == 'G4c'
    assert column.key_facts[0].value.startswith('circulates; judged on one 1:1 leg')
    assert 'the 1:1 leg at the record’s end must close' in column.confidence.because


@pytest.mark.benchmark
def test_2897_is_too_weak():
    _, column = _real('mmr2__2897__35S-4+0-31')
    assert column.subtitle.title.startswith('35S-4+0-31 · too weak')
    assert _decides(column).rule_id == '5s'
    assert column.numbers.decisive[0].startswith('strength implausible for this period [F]: needed / theory strength')
