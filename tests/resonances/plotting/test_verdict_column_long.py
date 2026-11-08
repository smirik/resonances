"""The right column of a secular or Lidov-Kozai figure as a `verdict_column.Column` (`verdict_column_long.long_column`).

Every built column must keep the contract (`verdict_column.problems` == []); the contents are checked against numbers
worked out apart from the code:

- a rate of f ″/yr turns once in 1 296 000 / |f| yr: g − s = 217.5″/yr is 5.96 kyr, the g6 − g5 beat
  28.2455 − 4.2575 = 23.988″/yr is 54.0 kyr, σ̇ = 2.182″/yr of 1222 g − g6 is 594 kyr;
- 1/T of a 500-kyr record is 1 296 000 / 500 000 = 2.592″/yr (1981 Midas: the free node 1.85″/yr from s6 is below it);
- the gate's cycles are T / P_cyc: 500 kyr / 6 kyr = 83.3 for 591986;
- G5p compares the swing 2 A_max with the phase error: 1222 has A_max 35.15°, so 2 A_max = 70.3°;
- the synthetic Lidov-Kozai body (591986's analogue: free circle at 222″/yr about a forced vector at g5) turns its
  free ω (222 − 4.26) × 500 000 / 1 296 000 = 84 times, a kinematic libration;
- the synthetic secular angle g − g6 + s − s6 with g = 57.8, s = −56.1″/yr has σ̇ = 57.8 − 28.2455 − 56.1 + 26.3450 =
  −0.2005″/yr, a period of 6.46 Myr.

Benchmark records (marker `benchmark`) are analyzed as `benchmark/tools/reclassify_secular_zlk.py` does.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from resonances.plotting.verdict_text import fmt as vt_fmt

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.plotting import verdict_column as vc
from resonances.plotting.secular_plots import long_period_frequencies
from resonances.plotting.verdict_column_long import long_column, resonance_words
from resonances.resonance.analysis.secular import AngleTerm, LongPeriodSeries, analyse_secular
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import ResonanceType
from resonances.resonance.analysis.zlk import analyse_zlk
from resonances.secular import free_elements as fe

SETTINGS = settings_from_config()
RAD = 2 * np.pi / 1296000.0  # rad/yr per arcsec/yr
G5, G6, S6 = PLANETARY_FREQUENCIES['g5'], PLANETARY_FREQUENCIES['g6'], PLANETARY_FREQUENCIES['s6']
NUMBER = r'([0-9]+(?:\.[0-9]+)?)'
COLUMN_CACHE = {}


def _text(column: vc.Column) -> str:
    return vc.column_text(column)


def _block(column: vc.Column, name: str) -> str:
    return '\n'.join(vc.block_text(column, name))


def _decides(column: vc.Column) -> vc.CheckLine:
    return next(line for line in column.checks if line.decides)


# --------------------------------------------------------------------------------------------- synthetic (fast)


@pytest.fixture(scope='module')
def secular_column():
    t = np.linspace(0.0, 2e6, 4001)
    g, s = 57.8, -56.1
    varpi, Omega = g * RAD * t + 0.3, s * RAD * t + 1.1
    sigma = varpi - G6 * RAD * t + Omega - S6 * RAD * t
    n = len(t)
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -1), AngleTerm('s', None, 1), AngleTerm('s', 6, -1))
    series = LongPeriodSeries(
        ResonanceType.SECULAR,
        'g-g6+s-s6',
        'synthetic',
        t,
        sigma,
        2.73 + 0.002 * np.sin(2 * np.pi * t / 3e4),
        np.full(n, 0.05),
        np.full(n, np.radians(5.0)),
        varpi=varpi,
        Omega=Omega,
        terms=terms,
        planet_frequencies=PLANETARY_FREQUENCIES,
    )
    analysis = analyse_secular(series, SETTINGS)
    row = {**analysis.summary_row(), 'resonance': 'g-g6+s-s6'}
    frequencies = long_period_frequencies(analysis, row)
    return analysis, long_column(analysis, SETTINGS, row, analysis.episode_rows(), frequencies, 'secular')


@pytest.fixture(scope='module')
def zlk_column():
    t = np.arange(0.0, 500_001.0, 100.0)
    z_e = 0.045 * np.exp(1j * (G5 * RAD * t + np.radians(83.0))) + 0.017 * np.exp(1j * (G6 * RAD * t + np.radians(3.2)))
    z_e = z_e + 0.011 * np.exp(1j * 222.0 * RAD * t)
    Omega = G5 * RAD * t + np.radians(146.0) + np.radians(8.0) * np.sin(2 * np.pi * t / 1.5e6)
    e, varpi = np.abs(z_e), np.angle(z_e)
    inc = np.full_like(t, np.radians(39.0))
    omega = np.unwrap(varpi - Omega)
    series = LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, 5.2 + 0.05 * np.sin(2 * np.pi * t / 1e4), e, inc)
    elements = fe.free_elements(t, e, inc, Omega, omega)
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    row = {**analysis.summary_row(), **fe.summary_fields(elements)}
    frequencies = long_period_frequencies(analysis, row, elements)
    return analysis, long_column(analysis, SETTINGS, row, analysis.episode_rows(), frequencies, 'zlk')


def test_synthetic_secular_column_keeps_the_contract(secular_column):
    analysis, column = secular_column
    assert vc.problems(column) == []
    assert column.family == 'secular' and column.subtitle.title.startswith('g − g6 + s − s6 · ')
    frequencies = next(g for g in column.numbers.groups if g.title.startswith('FREQUENCIES'))
    sigma = next(cells for cells in frequencies.cells if cells[0] == 'σ')
    # by hand −0.2005″/yr and 1 296 000 / 0.2005 yr = 6.46 Myr (g and s are the measured mean rates of the free vectors)
    assert float(sigma[1].replace('−', '-')) == pytest.approx(-0.2005, abs=3e-4) and sigma[2] == '6.46 Myr'
    assert 'origin' not in column.subtitle.title + column.subtitle.detail  # P7: no 'origin' on a secular row
    assert column.checks[0].number == 'Q' and column.checks[1].number == '0'


def test_synthetic_zlk_kinematic_column(zlk_column):
    analysis, column = zlk_column
    assert vc.problems(column) == []
    assert column.header.status == 'non_resonant' and column.decided_rule == 'SZ.G'
    assert 'the free ω winds' in column.subtitle.title and 'step 1 (free-ω gate)' in column.subtitle.detail
    facts = {fact.label: vc.plain(fact.value) for fact in column.key_facts}
    turns = float(re.search(rf'winds\*?\*?: {NUMBER} net turns', facts['free ω (g − s)']).group(1))
    assert turns == pytest.approx((222.0 - 4.26) * 500_000 / 1_296_000, abs=1.5)
    # one count of the free ω's turns in the main zone (the net one), the mean-rate count only in Numbers, labelled
    why = _block(column, 'why')
    assert f'winds instead: {vt_fmt(turns)} net turns over the record' in why
    gate = next(group for group in column.numbers.groups if group.title == 'FREE-ω GATE')
    rows = dict(gate.rows)
    assert float(rows['net turns of the free ω (record)']) == pytest.approx(turns)
    mean = rows['turns at the mean rate |g − s|·T']
    assert 'mean rate' not in why + _block(column, 'key_facts') and mean  # the mean-rate count only here, labelled
    assert facts['forcing'].startswith('mode g5') and '|c_g5| 0.045' in facts['forcing']
    assert _decides(column).number == '1' and 'it winds' in _decides(column).answer
    assert 'forced vector' in _block(column, 'evidence')


def test_synthetic_zlk_decisive_line_is_the_gate_comparison(zlk_column):
    """DECISIVE of a gate that decided: 'free-ω gate: kinematic · ρ_max … · cycles of ω T/P ≥ 3 [SZ.G]', the numbers
    those of the gate's columns, none recomputed: T / P_cyc of the 500-kyr record, ρ_max of zlk_gate_rho_max."""
    analysis, column = zlk_column
    row = analysis.summary_row()
    line = column.numbers.decisive[0]
    assert line.startswith('free-ω gate: kinematic · ρ_max ') and line.endswith(' [SZ.G]')
    rho = float(re.search(rf'ρ_max {NUMBER}', line).group(1))
    assert rho == pytest.approx(row['zlk_gate_rho_max'], rel=5e-3) and rho < 1  # the free circle inside the forced one
    cycles = float(re.search(rf'cycles of ω in the record T / P_cyc {NUMBER} ≥ 3', line).group(1))
    assert cycles == pytest.approx(500_000 / row['zlk_gate_period_yr'], rel=5e-3)
    assert not any(text.startswith('step 1:') for text in column.numbers.decisive)  # the comparison is said once


def test_node_words_follow_the_lock_columns():
    """'locked' only for a resolved lock; a rate within 1/T without a full cycle of Ω − ϖ_j only 'as closely as the
    record can resolve'; another mode or no candidate: nothing about the node (591986: Ω − ϖ5 within 53–93°, no full
    cycle)."""
    from resonances.plotting.verdict_column_why import _node_words
    from resonances.resonance.analysis.zlk.diagnostics import NodeLock

    unresolved = NodeLock(reason='unresolved', candidate='g5', rate_diff_arcsec_yr=-0.17, min_deg=53.0, max_deg=93.0)
    assert _node_words(unresolved, 'g5') == 'The node moves at the rate of g5 as closely as the record can resolve'
    locked = NodeLock(mode='g5', candidate='g5', min_deg=50.0, max_deg=130.0, cycles=1.5, period_yr=1.46e6)
    assert _node_words(locked, 'g5') == 'The node is locked to g5 too'
    assert _node_words(unresolved, 'g6') == '' and _node_words(NodeLock(reason='rate_mismatch', candidate='g5'), 'g5') == ''


def test_secular_generic_reason_says_no_origin():
    """D9: 'origin' is hidden on secular rows; the generic template words the reason origin_unknown without it."""
    from resonances.plotting.verdict_column_why import LONG_REASON_WORDS

    assert 'origin' not in LONG_REASON_WORDS['origin_unknown'].lower()


def test_resonance_words():
    assert resonance_words('g-g6') == 'g − g6'
    assert resonance_words('g-g6+s-s6') == 'g − g6 + s − s6'
    assert resonance_words('2_g-g6_+_s-s6_') == '2(g − g6) + (s − s6)'
    assert resonance_words('LK') == 'Lidov–Kozai'


def test_synthetic_zlk_key_facts_name_each_number(zlk_column):
    """Review 2 (#26, #17, #33): the free ω's period is the mean rate's 360°/|g − s| (not a libration period), the
    forcing ratio is against its one term, the gate's count is T / P_cyc. By hand: 360°/|g − s| of the synthetic body's
    free ω (222 − 4.26″/yr) is 1 296 000 / 217.7 = 5.95 kyr; T / P_cyc = 500 kyr / 5.95 kyr ≈ 84."""
    analysis, column = zlk_column
    facts = {fact.label: vc.plain(fact.value) for fact in column.key_facts}
    period = re.search(r'360°/\|g − s\| ([0-9.]+) kyr', facts['free ω (g − s)'])
    assert period and float(period.group(1)) == pytest.approx(1_296_000 / (222.0 - 4.26) / 1000, abs=0.05)
    assert facts['forcing'].endswith('(against this term alone)') and 'e_free/|c_g5| ' in facts['forcing']
    gate = next(line for line in column.checks if line.number == '1')
    cycles = float(re.search(rf'cycles of ω in the record T / P_cyc {NUMBER} ≥ 3', gate.numbers).group(1))
    assert cycles == pytest.approx(500_000 / (1_296_000 / 217.7), rel=0.03)


def test_synthetic_secular_resonance_fact_and_no_repeated_mean_rate(secular_column):
    """Review 2 (#24, #22): the modes of σ with their coefficients and σ̇ once, in Key facts (−0.2005″/yr, 6.46 Myr by
    hand, module docstring); Evidence has no second mean-rate row; the cutoff says what set it."""
    analysis, column = secular_column
    facts = {fact.label: vc.plain(fact.value) for fact in column.key_facts}
    assert facts['resonance'].startswith('modes in σ (coefficient): g6 Sat −1, s6 Sat −1 · mean rate σ̇ −0.2')
    assert facts['resonance'].endswith('(360°/|σ̇| 6.46 Myr)')
    assert 'mean rate' not in _block(column, 'evidence')
    assert re.search(r'filter cutoff \S+ \S+ \(set by ', facts['run'])


def test_circulating_zlk_coupling_is_no_support():
    """Review 2 (#15): constant √(1−e²) cos i holds for a circulating ω too, so the record's ZLK coupling of a circulating
    row is shown neutral (–), never ✓, under 'what moves with ω?'. ω circulates once in 4 kyr; e = 0.2 + 0.05 cos 2ω
    and cos i = H/√(1−e²) keep H, so e and i trade at π/|ω̇| = 2 kyr, half the circulation period."""
    t = np.arange(0.0, 200_001.0, 20.0)
    omega = 2 * np.pi * t / 4000.0 + 0.3
    e = 0.2 + 0.05 * np.cos(2 * omega)
    inc = np.arccos(np.sqrt(1 - 0.2**2) * np.cos(np.radians(30.0)) / np.sqrt(1 - e**2))
    series = LongPeriodSeries(
        ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, np.full_like(t, 2.5), e, inc, Omega=-2 * np.pi * t / 60000.0
    )
    analysis = analyse_zlk(series, SETTINGS)
    row = analysis.summary_row()
    column = long_column(analysis, SETTINGS, row, analysis.episode_rows(), long_period_frequencies(analysis, row), 'zlk')
    assert vc.problems(column) == [] and column.header.status == 'non_resonant'
    block = column.evidence[0]
    assert block.title == 'Evidence: what moves with ω?'
    ei = next(r for r in block.rows if r.label == 'e-i coupling')
    assert ei.icon == vc.QUIET and 'at 2 kyr' in ei.value and '(half the circulation period of ω)' in ei.value
    assert 'ZLK' in ei.value  # the coupling itself is measured as ZLK
    facts = {fact.label: vc.plain(fact.value) for fact in column.key_facts}
    assert facts['orbit'].startswith('a 2.5 au · ')  # a constant a: one number, not '2.500000–2.500000'


def test_sub_check_names_say_what_the_numbers_show(zlk_column):
    """No row states the passing condition beside failing numbers (plotting/CLAUDE.md). The synthetic body's deciding
    episode: i against e at 10.3° from antiphase (the bar is 45°), coherence 0.004 (needs ≤ −0.5) and η_H 1.0 (≤ 0.5)
    fail all three e-i tests; its free circle (0.011) lies inside the forced one (0.045 at g5): ρ_max 0.388 < 1. With
    ρ_max set to 1.31 the comparison fails and the name says the free e reaches the forced e."""
    from dataclasses import replace

    from resonances.plotting import verdict_column_long as vl

    analysis, column = zlk_column
    ei = next(line for line in column.checks if line.number == '2')
    assert [child.text for child in ei.children][:3] == [
        'i against e not in antiphase',
        'e and i not coherent',
        '√(1−e²) cos i not kept',
    ]
    gate = next(line for line in column.checks if line.number == '1')
    assert any(child.text == 'free e below the forced e over the record' for child in gate.children)
    reached = replace(analysis, gate=replace(analysis.gate, rho_max=1.31))
    child = next(c for c in vl._gate_subs(reached, analysis.summary_row(), SETTINGS) if 'forced e over the record' in c.text)
    assert child.text == 'free e reaches the forced e over the record' and 'ρ_max 1.31 ≥ 1' in child.numbers


def _variant(analysis, verdict=None, gate=None):
    from dataclasses import replace

    return replace(
        analysis,
        verdict=replace(analysis.verdict, **(verdict or {})),
        gate=replace(analysis.gate, **(gate or {})),
    )


def test_a_free_e_veto_on_an_undefined_phase_is_no_kept_status(zlk_column):
    """policy._zlk_measured: an undefined phase with ρ_max < 1 and the e-i integral failed is the veto zlk_free_e
    (non_resonant, kinematic). Step 1 says so, never 'the shape’s status is kept, at most medium'."""
    from resonances.plotting import verdict_column_long as vl

    analysis, _ = zlk_column
    vetoed = _variant(analysis, {'veto': 'zlk_free_e'}, {'outcome': 'undefined_phase'})
    row = {**analysis.summary_row(), 'pol_decided_by': 'veto:zlk_free_e'}
    assert vl.deciding_step(vetoed, row, 'zlk') == '1'
    line = vl._gate_step(vetoed, row, SETTINGS, decides=True)
    assert line.decides and 'the free e stays below the forced e over the record → non-resonant, kinematic' in line.answer
    assert 'at most medium' not in line.answer


def test_a_thin_veto_pill_reads_as_the_mmr_one(zlk_column):
    """The veto's pill is `verdict_column_common.pill_text`'s for every family: 'overruled by the winding free ω (by a
    thin margin)', and the Confidence line names it without its gloss."""
    from resonances.plotting import verdict_column_long as vl

    analysis, column = zlk_column
    assert [vc.plain(p.text) for p in column.confidence.pills if p.key == 'veto'] == ['overruled by the winding free ω']
    thin = _variant(analysis, {'confidence': 'medium'})
    row = {**analysis.summary_row(), 'pol_level_inputs': 'veto:thin(zlk_kinematic):medium;second_capture:none:certain'}
    block = vl.confidence(thin, row, 'zlk')
    assert [vc.plain(p.text) for p in block.pills] == ['overruled by the winding free ω (by a thin margin)']
    assert block.because == 'not certain because overruled by the winding free ω'


def test_long_caveats_override_the_registry_for_the_edges(zlk_column):
    """An edge of a long-period row is said without the registry's distance in periods (LONG_CAVEATS), and only on the
    statuses LONG_CAVEATS names; any other flag takes the registry's sentence with its slots (a jump at 12 000 yr is
    '12 kyr')."""
    from resonances.plotting import verdict_column_long as vl

    analysis, _ = zlk_column
    row = {**analysis.summary_row(), 'entry_suspected': True, 'entry_remaining_P': 0.4, 'regime_start_yr': 12_000.0}
    flags = ['entry_suspected', 'last_regime_after_jump']
    transient = vl.caveats(_variant(analysis, {'status': 'transient', 'flags': flags}), row, 'zlk', analysis.episode_rows())
    assert transient[0].startswith('An entry is suspected: before the first librating episode') and '0.4' not in transient[0]
    assert transient[1].startswith('a jumps at 12 kyr;')
    near = vl.caveats(_variant(analysis, {'status': 'near_separatrix', 'flags': flags}), row, 'zlk', analysis.episode_rows())
    assert not any('entry' in text for text in near) and near[0].startswith('a jumps at 12 kyr;')


def test_the_coorbital_zone_reads_the_settings_given(zlk_column):
    """⟨a⟩ = 5.2 au lies 0.07% from Jupiter's 5.2034 au: inside the co-orbital zone of half-width k (m_J/3)^(1/3) =
    0.068 k for k = 3, outside it for k = 1e-6; the subtitle and Key facts read the settings passed to the column."""
    from dataclasses import replace

    analysis, column = zlk_column
    assert 'co-orbital zone of Jupiter' in column.subtitle.detail
    row = analysis.summary_row()
    frequencies = long_period_frequencies(analysis, row)
    narrow = replace(SETTINGS, zlk_coorbital_hill_factor=1e-6)
    outside = long_column(analysis, narrow, row, analysis.episode_rows(), frequencies, 'zlk')
    assert 'co-orbital zone' not in outside.subtitle.detail + _block(outside, 'key_facts')


@pytest.mark.parametrize(
    'lo, hi, expected',
    [
        (2.788208, 2.794853, '2.788–2.795'),  # 1020 z1: '2.79–2.79' at 3 digits
        (2.787643, 2.801222, '2.79–2.80'),  # 1222 ν6: one number of decimals, not '2.79–2.8'
        (5.166113, 5.249178, '5.17–5.25'),  # 591986
        (0.831, 1.08, '0.83–1.08'),
        (-505.8465, 135.4489, 'from −506 to 135'),  # 2010 HY22, unbound: a U+2212 minus, no '-506–135'
    ],
)
def test_a_range_tells_its_ends_apart(lo, hi, expected):
    from resonances.plotting.verdict_column_long import _a_range

    assert _a_range(lo, hi) == expected


# --------------------------------------------------------------------------------------------- benchmark records


def _column(example: str) -> vc.Column:
    """The column of a secular/ZLK benchmark record, analyzed as `reclassify_secular_zlk` does (cached per module)."""
    if example in COLUMN_CACHE:
        return COLUMN_CACHE[example]
    tools = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
    sys.path.insert(0, str(tools))
    try:
        import bench_io as bio
        import reclassify_secular_zlk as reclassify
    except ImportError:
        pytest.skip('benchmark tools not available')
    if not bio.example_dir(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.plotting.mmr_verdict_panel import config_settings

    sim, body, resonance = reclassify.build_simulation(bio.load_example(example, with_planets=False))
    sim.identify_librations()
    analysis = sim.long_period_analysis(body, resonance)
    row = {**analysis.summary_row(), **reclassify._extra_columns(reclassify.summary_row(example, body, resonance))}
    elements = body.free_elements if resonance.type == 'lidov_kozai' else None
    frequencies = long_period_frequencies(analysis, row, elements)
    family = 'zlk' if example.startswith('zlk') else 'secular'
    column = long_column(analysis, config_settings(None), row, analysis.episode_rows(), frequencies, family)
    assert vc.problems(column) == []
    COLUMN_CACHE[example] = column
    return column


BENCHMARK = [
    'secular__1222__g-g6',
    'secular__2306__g-g6+s-s6',
    'secular__363__g-g6+s-s6',
    'secular__847__g-g6+s-s6',
    'zlk__3040__LK',
    'zlk__591986__LK',
    'zlk__88899__LK__500kyr',
    'zlk__162474__LK',
    'zlk__1981__LK',
    'zlk__2010_HY22__LK',
    'zlk__10563__LK__500kyr',
]


@pytest.mark.benchmark
@pytest.mark.parametrize('example', BENCHMARK)
def test_benchmark_columns_keep_the_contract(example):
    column = _column(example)
    assert vc.problems(column) == []
    assert sum(line.decides for line in column.checks) == 1
    assert column.footer[-1] == 'docs: physics/secular-zlk.md'
    assert 'response' not in _block(column, 'caveats')  # D7: the secular e-response is never a caveat


@pytest.mark.benchmark
def test_1222_nu6_is_certain_by_its_shape():
    column = _column('secular__1222__g-g6')
    assert (column.header.status, column.header.level, column.header.chip.text) == ('resonant', 'certain', 'no action needed')
    assert _decides(column).number == '0' and 'decided at step 0 (shape)' in column.subtitle.detail
    phase = next(line for line in column.checks if line.number == '2')
    assert phase.rule_id == 'G5p' and 'largest full swing 70.3° > 0.847°' in phase.numbers  # 2 × 35.15°
    frequencies = next(g for g in column.numbers.groups if g.title.startswith('FREQUENCIES'))
    assert next(cells for cells in frequencies.cells if cells[0] == 'σ')[2] == '594 kyr'  # 1 296 000 / 2.182
    assert column.caveats == () and 'e-response pass' in ' · '.join(column.footer)
    # Key facts count the cycles the level counts (input N), as the Why and the pill do: 16.8, not the episode's 16
    facts = {fact.label: vc.plain(fact.value) for fact in column.key_facts}
    assert (
        ' · 16.8 cycles counted · ' in facts['σ']
        and facts['σ'].startswith('librates over the whole record (')
        and '16.8 cycles counted' in ' '.join(column.why.sentences)
    )


@pytest.mark.benchmark
def test_2306_is_an_unfinished_slow_swing():
    column = _column('secular__2306__g-g6+s-s6')
    assert column.header.status == 'uncertain' and column.header.level == ''
    assert column.header.chip.text == 'extend to 20 Myr'
    assert column.decided_rule == 'G5s' and _decides(column).number == '0'
    assert 'turned back' in _decides(column).answer and 'one full slow swing' in column.confidence.because
    assert 'e-response fail' in ' · '.join(column.footer) and column.caveats == ()  # D7


@pytest.mark.benchmark
def test_363_too_short_and_the_later_steps_not_reached():
    column = _column('secular__363__g-g6+s-s6')
    assert column.header.chip.text == 'extend to 20 Myr'
    assert any(line.quiet and line.answer == '— not reached' for line in column.checks)


@pytest.mark.benchmark
def test_847_circulates():
    column = _column('secular__847__g-g6+s-s6')
    assert (column.header.status, column.header.level) == ('non_resonant', 'high')
    assert 'circulates' in column.subtitle.title and 'circulates' in _decides(column).answer
    assert 'center’s largest one-way pass 1.92 turns ≥ 1 turn' in _decides(column).numbers
    held = [pill for pill in column.confidence.pills if pill.icon == vc.THIN]
    assert [(pill.key, pill.ceiling) for pill in held] == [('circulation', 'high')]


@pytest.mark.benchmark
@pytest.mark.parametrize('example', ['zlk__3040__LK', 'zlk__162474__LK'])
def test_confirmed_gate_decides(example):
    column = _column(example)
    assert (column.header.status, column.header.level) == ('resonant', 'certain')
    assert column.decided_rule == 'SZ.G' and 'rule SZ.G' in column.footer
    assert 'confirmed' in column.subtitle.title and _decides(column).answer.startswith('Yes')
    ei = next(line for line in column.checks if line.number == '2')
    assert ei.icon == vc.OK and len(ei.children) == 3
    facts = {fact.label: vc.plain(fact.value) for fact in column.key_facts}
    assert facts['free ω (g − s)'].split(' · ')[-1].startswith('librates: full range ')


@pytest.mark.benchmark
def test_591986_trojan_kinematic():
    column = _column('zlk__591986__LK')
    assert (column.header.status, column.header.level) == ('non_resonant', 'high')
    assert 'only kinematically: the free ω winds' in column.subtitle.title
    assert 'co-orbital zone of Jupiter' in column.subtitle.detail
    facts = {fact.label: vc.plain(fact.value) for fact in column.key_facts}
    assert '360°/|g − s| 5.96 kyr' in facts['free ω (g − s)'] and 'winds: 83.1 net turns' in facts['free ω (g − s)']  # 1 296 000 / 217.5
    assert facts['forcing'].startswith('mode g5')
    assert '83.3 ≥ 3' in _decides(column).numbers  # 500 kyr / 6 kyr
    assert [pill.key for pill in column.confidence.pills] == ['veto'] and column.confidence.pills[0].icon == vc.THIN
    # D9 'overruled by X', as on the MMR rows; the Why's limit names the same veto
    assert column.confidence.pills[0].text == 'overruled by the winding free ω'
    assert column.confidence.because == 'not certain because overruled by the winding free ω'
    assert 'forced beat g6 − g5' in _block(column, 'evidence') and 'Ω − ϖ₅' in _block(column, 'evidence')
    evidence = _block(column, 'evidence')
    assert 'forced vector' in evidence and 'center wave' in evidence and 'node lock' in evidence
    frequencies = next(g for g in column.numbers.groups if g.title.startswith('FREQUENCIES'))
    assert next(cells for cells in frequencies.cells if cells[0] == 'g6 − g5')[2] == '54 kyr'  # 1 296 000 / 23.988
    assert any('does not read it' in text for text in column.caveats)  # the e-i of the episode, not read


@pytest.mark.benchmark
def test_591986_one_free_turn_count_and_the_node_within_the_resolution():
    """The Why says of the node only what the lock columns support (Ω − ϖ5 closes no cycle: within the resolution, not
    locked) and prints the free ω's net turns (83.1, as Key facts); |g − s|·T = 217.5 × 500 000 / 1 296 000 = 83.9 is
    printed only in Numbers, under its own label. DECISIVE is the gate's comparison: 500 kyr / 6 kyr = 83.3 cycles."""
    column = _column('zlk__591986__LK')
    why = _block(column, 'why')
    assert 'The node moves at the rate of g5 as closely as the record can resolve' in why and 'locked' not in why
    assert 'winds instead: 83.1 net turns over the record' in why
    main = '\n'.join(_block(column, name) for name in vc.BLOCKS if name not in ('numbers', 'footer'))
    assert '83.9' not in main
    rows = dict(next(group for group in column.numbers.groups if group.title == 'FREE-ω GATE').rows)
    assert rows['net turns of the free ω (record)'] == '83.1' and rows['turns at the mean rate |g − s|·T'] == '83.9'
    assert column.numbers.decisive[0] == 'free-ω gate: kinematic · ρ_max 0.431 · cycles of ω in the record T / P_cyc 83.3 ≥ 3 [SZ.G]'


@pytest.mark.benchmark
def test_88899_500kyr_low_by_the_e_i_integral():
    column = _column('zlk__88899__LK__500kyr')
    assert (column.header.status, column.header.level, column.header.chip.kind) == ('transient', 'low', vc.CHIP_REVIEW)
    assert column.confidence.pills[0].icon == vc.FAIL and 'constant √(1−e²) cos i' in column.confidence.pills[0].text
    ei = next(line for line in column.checks if line.number == '2')
    assert ei.icon == vc.FAIL and any('η_H 1.24 > 0.5' in child.numbers for child in ei.children)
    # flag entry_suspected: said without the registry's distance in periods (a long row's edge is read about the mean
    # centre, not the episode's gap that step 0 prints: 0.919 P against 2 P)
    entry = [text for text in column.caveats if 'may have begun before the record' in text]
    assert entry and 'libration periods' not in entry[0]


@pytest.mark.benchmark
def test_1981_kinematic_excluded_without_a_gate_verdict():
    column = _column('zlk__1981__LK')
    assert (column.header.status, column.header.level) == ('resonant', 'certain')
    # pol_decided_by gate_unavailable: the policy's line P decides, the shape's step is answered (✓), not ◆
    assert _decides(column).number == 'P' and column.decided_rule == 'P'
    assert (
        'kinematic libration is excluded' in _decides(column).answer
        and 'is kept, at the level its evidence gives' in _decides(column).answer
    )
    assert next(line for line in column.checks if line.number == '0').icon == vc.OK
    gate = next(line for line in column.checks if line.number == '1')
    assert gate.icon == vc.MISSING and gate.answer == 'No verdict.'
    node = next(child for child in gate.children if 'not yet separated from s6' in child.text)
    assert '1.85″/yr < 2.59″/yr' in node.numbers  # 1/T = 1 296 000 / 500 000
    assert any('kinematic libration excluded' in pill.text for pill in column.confidence.pills)


@pytest.mark.benchmark
def test_2010_hy22_is_invalid():
    column = _column('zlk__2010_HY22__LK')
    assert (column.header.status, column.header.level, column.header.chip.kind) == ('invalid', '', vc.CHIP_REVIEW)
    assert _decides(column).number == 'Q' and 'e reaches 2.11' in _decides(column).answer
    assert column.evidence == () and 'FREQUENCIES' not in _block(column, 'numbers')


@pytest.mark.benchmark
def test_gate_unavailable_keeps_the_shape_at_medium():
    column = _column('zlk__10563__LK__500kyr')
    assert (column.header.status, column.header.level) == ('transient', 'medium')
    # an unfitted co-orbital: its free split (g, j) is no measurement, the chip points at the episodes and a
    assert column.header.chip.text == 'glance at (a), (d)'
    assert _decides(column).number == 'P' and 'at most medium' in _decides(column).answer
    v = next(pill for pill in column.confidence.pills if pill.key == 'V')
    assert v.icon == vc.MISSING and 'co-orbital of Earth' in v.text
    assert 'co-orbital zone of Earth' in column.subtitle.detail


@pytest.mark.benchmark
def test_2020_pp1_policy_line_and_an_unfitted_forcing():
    """2020 PP1 (Earth co-orbital, pol_decided_by gate_unavailable): ◆ on the policy's line, the shape's step answered
    (✓/⚠, not ◆), rule P in the footer; FREQUENCIES says that g − s is no free rate: Earth's forcing is not fitted."""
    column = _column('zlk__2020_PP1__LK')
    assert (column.header.status, column.header.level) == ('transient', 'medium')
    line = _decides(column)
    assert (line.number, line.rule_id, column.decided_rule) == ('P', 'P', 'P')
    assert line.question == 'Does the free-ω gate give a verdict?'
    assert vc.plain(line.answer) == (
        'No (the free-element model does not apply to a co-orbital of Earth) → the shape’s status (transient) is kept, at most medium'
    )
    assert next(line for line in column.checks if line.number == '0').icon in (vc.OK, vc.THIN)
    assert 'rule P' in column.footer and 'decided without a gate verdict' in column.subtitle.detail
    cells = {cells[0]: cells for cells in next(g for g in column.numbers.groups if g.title.startswith('FREQUENCIES')).cells}
    assert cells['g − s'][3] == 'not free: Earth’s forcing not fitted'
    assert all('turns' not in label for group in column.numbers.groups for label, _ in group.rows)


@pytest.mark.benchmark
@pytest.mark.parametrize('example', [name for name in BENCHMARK if name.startswith('secular')])
def test_secular_main_zone_has_no_origin(example):
    """D9: 'origin' is hidden on secular rows: no block but the footer (raw codes) names it."""
    column = _column(example)
    main = '\n'.join(_block(column, name) for name in vc.BLOCKS if name != 'footer')
    assert 'origin' not in main.lower()


@pytest.mark.parametrize('k, b, thin', [(0.95, 0.9, True), (0.90, 1.3, False)])
def test_a_short_record_decided_by_the_policy_shows_step_p(k, b, thin):
    """W36 F5 (143662 s+g5−g8−s3): the policy, not the tree, decided a too-short record that ends in circulation
    (`pol_decided_by` insufficient_time). ◆ moves to P with MMR's question and the end travel σ_f spans (1.04 and 1.11
    turns by hand, `test_secular_thin_end_travel`); the shape's 'Not yet' stays, overruled; a thin travel says at most
    medium."""
    from tests.resonances.analysis.test_secular_thin_end_travel import short_record_series

    analysis = analyse_secular(short_record_series(k, b), SETTINGS)
    row = {**analysis.summary_row(), 'resonance': 's+g5-g8-s3'}
    column = long_column(analysis, SETTINGS, row, analysis.episode_rows(), long_period_frequencies(analysis, row), 'secular')
    assert vc.problems(column) == []
    assert (column.header.status, column.header.level) == ('non_resonant', 'medium' if thin else 'high')
    line = _decides(column)
    assert (line.number, line.rule_id, column.decided_rule) == ('P', 'P', 'P')
    assert line.question == 'Is the record long enough?'
    answer, numbers = vc.plain(line.answer), vc.plain(line.numbers)
    assert answer.startswith('No, but σ_f ends in circulation → non-resonant')
    assert ('at most medium' in answer) == thin
    travel = '1.04' if thin else '1.11'
    assert numbers.startswith(f'σ_f spans {travel} turns ≥ 1 turn over the window')
    assert ('thin: 1.04 turns < 1.1 turns' in numbers) == thin
    shape = next(line for line in column.checks if line.number == '0')
    assert not shape.decides and shape.answer == 'Not yet: fewer turning points than one cycle needs; the policy decides at step P.'
    assert 'decided by the policy' in column.subtitle.detail and 'rule P' in column.footer
    assert 'the record ends in circulation' in _block(column, 'why')
    pills = [p.key for p in column.confidence.pills]
    assert ('end_travel' in pills) == thin
    if thin:
        assert column.header.chip.text == 'glance at (a), (b)'
