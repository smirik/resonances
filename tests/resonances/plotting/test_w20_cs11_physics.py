"""W20 CS11: the physics block of the right column, the caps' alternatives and the W20 ledger entries in words.

- The physics block (R1 §3–4, R5 §5): section A 'Is σ the own pendulum of <key>?' and section B 'What makes the
  cycles?'. Every number it prints is one of the record's columns (format-tolerant), every row says why when nothing
  was measured, and no σ amplitude is ever 'predicted' from the identity alone.
- The author's q1 (2026-10-19): every cap that brings the level down to medium (or low) names the other verdict and
  what on the figure decides it.
- No number twice: the columns the physics block prints leave the numbers below it; the S2 table of the multiplet's
  members replaces the bare list of keys.

Expected values come from other sources than the code under test: the R1 prototype (scratchpad w20-design/R1, 10
snapshots of its own: 385266 T0_th 209–260 kyr, K/K_th 0.96–1.02, |Δφ0| ≤ 1°, δa 0.019–0.020 au predicted against
0.0199 measured, σ band 12.6°; 82157 K/K_th 0.98–1.01), the identity written by hand from the two angles' coefficients,
and arithmetic done by hand in the docstrings.
"""

import math
import re
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from resonances.plotting import mmr_verdict_panel as panel  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.resonance.analysis.ledger import LedgerEntry, Op  # noqa: E402
from resonances.resonance.analysis.mmr import io  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from tests.resonances.analysis.builders import tree_and_final  # noqa: E402

S = settings_from_config()

# --------------------------------------------------------------------------------------------- pure


@pytest.mark.parametrize(
    'value, expected',
    [
        (5.802848e-10, '5.8e-10'),
        (2.7374972e-05, '2.74e-5'),
        (0.0196286, '0.0196'),
        (1.394, '1.39'),
        (2853.06, '2\u2009853'),
        (3.1e5, '3.1e5'),
    ],
)
def test_sci(value, expected):
    assert vt.sci(value) == expected


W20_ENTRIES = {
    '5 L4 null test: R2 above the binding line - max R2 with a shifted': 'the force law beats a shifted in time (null test)',
    'cap short_period_aliased: fast residual of sigma_f / median half-amplitude of the decisive episode': (
        'the fast spread of σ_f against the swing (the step folds short periods)'
    ),
    'short_period_aliased: a harmonic folded slower than the period floor': (
        'a harmonic of the short periods folded slower than the period floor'
    ),
    'short-period scatter of a / a half-excursion of the decisive episode': (
        'the short-period scatter of a against the libration’s a swing'
    ),
    'G2 exit observed: the raw angle runs on beyond the episode': 'exit seen: the orbit-averaged σ runs on beyond the episode',
    'G2 entry observed: the raw angle moves monotonically': 'entry seen: the orbit-averaged σ moved one way',
    '2 owner cover of the decisive episode': 'the owner’s intervals cover σ’s decisive episode',
    '2 owner route (potential)': 'the owner is accepted by its route',
    '2 second capture: x sweep': 'x = Ψ/u sweeps far enough to answer “both resonances at once?”',
    '2 second capture: energy R2': 'x = Ψ/u follows no potential (not both resonances at once)',
    'G1q quasi-satellite floor from the averaged potential': 'the quasi-satellite floor lies below the tadpole floor',
    'G1q the orbit-averaged angle stays in the quasi-satellite well': 'θ stays in the quasi-satellite well long enough',
}


def test_w20_ledger_entries_in_words():
    """The new entries of W20 parts A and B read as words; the longest matching start wins (the lift is not the cap);
    an unknown name stays as it is."""
    for name, words in W20_ENTRIES.items():
        assert vt.ledger_words(name) == words, name
    assert vt.ledger_words('G2 accepted episodes') == 'G2 accepted episodes'


def test_near_threshold_row_prints_words():
    """237743 5J-2S-2 (CS10 open item): own_cover 0.538 against 0.5 is inside the convention band."""
    name = '2 owner cover of the decisive episode'
    entry = LedgerEntry(name, 0.538, Op.GE, 0.5, '', 'own_cover', 'mmr_owner_cover_min', True, True)
    verdict = SimpleNamespace(thin_margins=[f'U:{name}'], decision_path=[entry])
    (row,) = vt.near_threshold_rows(verdict, S)
    assert row.name == 'the owner’s intervals cover σ’s decisive episode near its convention'
    assert row.detail == '0.538 ≥ 0.5'


def test_owner_intervals_and_decisive_period():
    """own_intervals in words; the period the owner's bound reads, back from own_sigma_period_min_yr = T/√(|v/u| b):
    385266, Ψ = 5σ − 27φ, b = 0.010454, 11 567.97 yr → T = 11 567.97·√(5.4·0.010454) = 11 567.97·0.23760 = 2 748.5 yr."""
    assert vt._intervals('9060-88093:forced_wave;100-200:slip') == 'during 9.06–88.1 kyr as a forced wave, 100–200 yr as slip'
    # one regime for every block (1997 GK3's owner): said once, before the spans
    assert vt._intervals('455-11600:libration;12800-18600:libration') == 'librating during 0.455–11.6 kyr, 12.8–18.6 kyr'
    assert vt._intervals('') == ''
    row = {'own_psi_def': '5*sigma-27*phi', 'own_sigma_period_min_yr': 11567.97}
    facts = SimpleNamespace(row=row)
    assert math.isclose(vt._decisive_period(facts, 0.010454), 2748.5, rel_tol=2e-4)
    facts.row = {'own_psi_def': '8*sigma-1*phi', 'own_sigma_period_min_yr': 1700.0}  # HW7: T = 1700·√(0.125·0.257)
    assert math.isclose(vt._decisive_period(facts, 0.257), 1700.0 * math.sqrt(0.125 * 0.257), rel_tol=1e-12)
    facts.row = {'own_psi_def': '', 'own_sigma_period_min_yr': 1700.0}
    assert math.isnan(vt._decisive_period(facts, 0.257))


def test_subresonance_table():
    """S2 (CS9 columns) as a table, written out by hand: W/W0, E, N, A/W of the first three members, the catalog
    angle's W0 and A/W0, the potential minimum's W/W0 = 13.2/36 = 0.37; a long key on its own line; N < 1 in words.
    Every cell after a space (W20 round C: 82157 printed '0.990.00190.0064'), the line as wide as before; a number
    longer than the 5-character cell in a short exponent (0.0019 -> 2e-3; 17346 2J-1: E = -4 020 658 -> -4e6)."""
    row = {
        'sub_rank1_key': '16N-27+3+8',
        'sub_rank1_W_ratio': 0.9257,
        'sub_rank1_E': 0.1212,
        'sub_rank1_N': 0.0793,
        'sub_rank1_AW': 0.0597,
        'sub_rank2_key': '31E-36+1+2[N+2,NE+0]',
        'sub_rank2_W_ratio': 0.37,
        'sub_rank2_E': 0.86,
        'sub_rank2_N': 1.7,
        'sub_rank2_AW': 2.2,
        'sub_rank3_key': '20N-29+1+8',
        'sub_rank3_W_ratio': 0.99,
        'sub_rank3_E': 0.0019,
        'sub_rank3_N': 0.0064,
        'sub_rank3_AW': -4020658.0,
        'sub_cat_W_deg': 36.0,
        'sub_cat_AW': 0.81,
        'sub_pot_W_deg': 13.2,
        'sub_pot_E': 0.99,
        'sub_strength_caveat': 'e;crossing',
    }
    lines = [line.value for line in panel.ranked_members_lines(row, 'G')]
    assert lines == [
        'member          W/W0     E     N   A/W',
        '16N-27+3+8      0.93  0.12 0.079  0.06',
        '31E-36+1+2[N+2,NE+0]',
        '                0.37  0.86   1.7   2.2',
        '20N-29+1+8      0.99  2e-3  6e-3  -4e6',
        'σ W0 36°           1     0     —  0.81',
        'pot. minimum    0.37  0.99     —     —',
        'N < 1: less than one slow cycle in the window, only a rate matched',
        "d'Alembert factors unreliable: e, crossing",
    ]
    assert panel.ranked_members_lines({}, 'G') == []


# --------------------------------------------------------------------------------------------- benchmark records

PHYSICS_RECORDS = (
    'mmr2__385266__16N-27+0+11',  # circulation driven by 3N-5: implausible own pendulum, forced wave confirmed
    'mmr2__82157__20N-29+0+9',  # the same with 2N-3
    'mmr2__119066__4N-7+0+3',  # resonant: its own pendulum, owner search without windows
    'mmr2__153__3J-2+0-1',  # Hilda: own pendulum, no owner after a valid search
    'mmr3__52356__5J-2S-2+0+0-1',  # three-body: the e^q scale only
    'coorbital__2016_CA138__1E-1+0+0',  # 1:1: its own force law, no owner section
    'mmr2__385533__7N-12+0+5',  # an episode without a required strength
    'mmr3__2001_HW7__5J+6S-2+0+0-9',  # two-body owner of a three-body angle, second capture unresolved
    'mmr3__237743__5J-2S-2+0+0-1',  # three-body owner: no direct theory
)
_CACHE = {}
TREE_TEXT_CACHE = {}


def _analysis(example):
    """(series, final analysis, its row, its text); the W28 tree's text of the same analysis goes to `_tree_text`."""
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    if example not in _CACHE:
        series = io.load_series(example)
        tree, analysis = tree_and_final(series, S)
        TREE_TEXT_CACHE[example] = vt.explain_mmr(tree, S, {**tree.numbers_row(), **tree.summary_row()}, tree.episode_rows())
        row = {**analysis.numbers_row(), **analysis.summary_row()}
        _CACHE[example] = (series, analysis, row, vt.explain_mmr(analysis, S, row, analysis.episode_rows()))
    return _CACHE[example]


def _tree_text(example):
    """The text of the W28 tree's verdict (W34: its caps, which the minimal policy turns into flags, and their words)."""
    _analysis(example)
    return TREE_TEXT_CACHE[example]


def _parse(text: str) -> float:
    return float(text.replace('\u2009', '').replace(' ', '').replace('−', '-'))


def _rows(text):
    return {(section.title.split()[0], row.key): row for section in text.physics for row in section.rows}


@pytest.mark.benchmark
@pytest.mark.parametrize('example', PHYSICS_RECORDS)
def test_every_physics_number_is_its_column(example):
    """Each number of the block, as printed, is its column times the unit factor (3 digits: 0.5 %; 2 digits and
    below 1e-3: 5 %; integers: 0.5); and it is printed in the row's text."""
    *_, row, text = _analysis(example)
    for section in text.physics:
        for physics_row in section.rows:
            for column, printed, scale in physics_row.numbers:
                assert printed in physics_row.text, (column, printed, physics_row.text)
                value = float(row[column]) * scale
                if printed == '≈0':
                    assert abs(value) < 1e-5
                    continue
                shown = _parse(printed)
                digits = len(re.sub(r'[^0-9]', '', printed.split('e')[0]).lstrip('0'))
                tolerance = (
                    0.51
                    if abs(value) >= 100 and '.' not in printed and 'e' not in printed
                    else abs(value) * (0.0051 if digits >= 3 else 0.051)
                )
                assert abs(shown - value) <= tolerance, (example, column, printed, value)


@pytest.mark.benchmark
@pytest.mark.parametrize('example', PHYSICS_RECORDS)
def test_no_sigma_amplitude_predicted_from_the_identity(example):
    """Only the a-wave is predicted (δa = K/ν); σ's wave is measured only, and no row compares a predicted σ."""
    *_, text = _analysis(example)
    for section in text.physics:
        for physics_row in section.rows:
            if physics_row.key == 'sigma_wave':
                assert 'measured only' in physics_row.text and 'predicted' not in physics_row.text
            elif 'predicted' in physics_row.text:
                assert physics_row.key in ('amplitude', 'theory') and 'σ' not in physics_row.text.split('predicted')[0][-12:]


@pytest.mark.benchmark
def test_385266_block_against_the_r1_prototype():
    """385266 16N-27: T0_th within 200–270 kyr (prototype 209–260 over 10 snapshots), μ_req/μ_th ≥ 1000 (prototype
    2.2–2.8e3), the a-wave wider than the zone (prototype 0.0196 against 0.0023–0.0028 au); owner 3N-5: K/K_th within
    [0.9, 1.1], |Δφ0| ≤ 3°, δa predicted within 10 % of measured, σ's band 12.6° (all prototype), the identity by
    hand: σ = 16λ_N − 27λ + 11ϖ, φ = 3λ_N − 5λ + 2ϖ, (27/5)φ = (81/5)λ_N − 27λ + (54/5)ϖ, so σ − (27/5)φ =
    (−1/5)λ_N + (1/5)ϖ = (ϖ − λ_N)/5; the second capture excluded."""
    *_, row, text = _analysis('mmr2__385266__16N-27+0+11')
    rows = _rows(text)
    assert [section.title for section in text.physics] == ['Is σ the own pendulum of 16N-27+0+11?', 'What makes the cycles?']
    assert 200 <= row['str_P_lib_theory_min_kyr'] <= row['str_P_lib_theory_max_kyr'] <= 270
    assert rows[('Is', 'period')].icon == '✗' and rows[('Is', 'strength')].icon == '✗' and rows[('Is', 'width')].icon == '✗'
    assert 'a rejected run of turning points, not an episode' in rows[('Is', 'period')].text
    assert row['str_ratio_all_min'] >= 1000 and '≥ 1\u2009000 × is implausible' in rows[('Is', 'strength')].text
    assert 0.9 <= row['own_K_over_K_direct_min'] <= row['own_K_over_K_direct_max'] <= 1.1 and row['own_phi0_diff_max_deg'] <= 3
    assert abs(row['own_wave_da_pred_au'] / row['own_wave_da_meas_au'] - 1) <= 0.1
    assert rows[('What', 'identity')].text.startswith('σ = (27/5)φ + (ϖ − λ_N)/5')
    assert rows[('What', 'theory')].icon == '✓' and rows[('What', 'amplitude')].icon == '✓'
    assert 'a linear forced wave' in rows[('What', 'amplitude')].text
    assert rows[('What', 'sigma_wave')].text.startswith('σ on sin/cos φ: 12.6°')
    assert rows[('What', 'capture')].icon == '✓' and rows[('What', 'capture')].text.endswith('→ excluded')
    assert 'model limit' in rows[('What', 'own_term')].text  # bound 0.0105 < MMR_OWNER_POTENTIAL_MODEL_ERROR


@pytest.mark.benchmark
def test_82157_owner_theory():
    """82157 20N-29 owner 2N-3: K/K_th within [0.9, 1.1] (prototype 0.98–1.01)."""
    *_, row, text = _analysis('mmr2__82157__20N-29+0+9')
    assert 0.9 <= row['own_K_over_K_direct_min'] <= row['own_K_over_K_direct_max'] <= 1.1
    assert _rows(text)[('What', 'theory')].icon == '✓'


@pytest.mark.benchmark
def test_rows_say_why_when_not_measured():
    """Three-body: the e^q scale only; 1:1: its own law, no owner section; an owner search that could not run says why;
    no owner after a valid search names the best candidate; an episode without a required strength says so."""
    *_, text = _analysis('mmr3__52356__5J-2S-2+0+0-1')
    assert text.physics[0].subtitle == 'order-of-magnitude scale' and 'is a scale only' in text.physics[0].rows[0].text
    *_, text = _analysis('coorbital__2016_CA138__1E-1+0+0')
    assert len(text.physics) == 1 and text.physics[0].subtitle == '1:1' and 'follows the 1:1 force law' in text.physics[0].rows[0].text
    *_, text = _analysis('mmr2__119066__4N-7+0+3')
    owner = _rows(text)[('What', 'owner')]
    assert owner.icon == '–' and owner.text == 'the libration holds too few force windows for the owner fit'
    assert _rows(text)[('Is', 'strength')].icon == '✓'
    *_, text = _analysis('mmr2__153__3J-2+0-1')
    owner = _rows(text)[('What', 'owner')]
    assert owner.icon == '✗' and owner.text.startswith('18 candidates, none accepted; best ')
    # 385533 lost its 'no cycle to read a required strength from' at W_c 22.7 kyr: since W28 B7 the scale ladder reads
    # it at 7.95 kyr, where F by direct averaging measures T0_th 33.9-36.8 kyr (Codex's scratch at 2 kyr: 33.8-36.7)
    *_, text = _analysis('mmr2__385533__7N-12+0+5')
    assert text.physics[0].subtitle == 'directly averaged potential' and text.physics[0].rows[0].text.startswith('T0_th 33.9')
    *_, text = _analysis('mmr3__237743__5J-2S-2+0+0-1')
    assert _rows(text)[('What', 'theory')].text.startswith('no direct theory for a three-body owner (second order)')


@pytest.mark.benchmark
def test_hw7_second_capture_and_its_cap():
    """2001 HW7 5J+6S-2 (R5: x sweep 46°, energy R² 0.04–0.06 → unresolved): the block says unresolved with the
    comparisons. W34: under the minimal policy the W28 tree's cap is the flag second_capture_unresolved."""
    *_, analysis, row, text = _analysis('mmr3__2001_HW7__5J+6S-2+0+0-9')
    capture = _rows(text)[('What', 'capture')]
    assert capture.icon == '?' and capture.text.endswith('→ not ruled out') and capture.label == 'both resonances at once?'
    assert f'x moved {vt.fmt(row["own_psi_sweep_deg"])}° < 180°' in capture.text
    assert 'second_capture_unresolved' in analysis.verdict.flags


W20_WORDING = {
    'coorbital__2017_UM52__1E-1+0+0': ('steps', 'exit seen: the orbit-averaged σ runs 1.97 rev ≥ 1 rev beyond the episode, one way'),
    'coorbital__2016_CA138__1E-1+0+0': ('steps', 'one cycle closed by its phase return: σ_f leaves the 0.5 level of the swing at'),
    # W28 C5 (g01-2): the stay needs one loop through its own turns (69.8 yr), no longer 1.5 small-amplitude P_QS,max
    'coorbital__164207__1E-1+0+0': ('steps', 'the orbit-averaged angle stays in the quasi-satellite well for 566 yr ≥ 69.8 yr'),
    # W20 round D: no lift by the record; the cap names the scatter of a (0.597 of the a swing, MEASURE: 0.60); W34: the
    # W28 tree's cap ('tree cap'), a flag under the minimal policy
    'mmr3__2001_HW7__5J+6S-2+0+0-9': ('tree cap', 'short-period scatter of a 0.597 > 0.5 × the a swing'),
}


@pytest.mark.benchmark
@pytest.mark.parametrize('example', sorted(W20_WORDING))
def test_w20_rules_in_words(example):
    """The W20 rules that decided a record are said in words with their numbers (the ledger values of the CS10 run:
    UM52 arc 1.97 rev, 164207 visit 566 yr against 145 yr, 69.8 yr since W28 C5, HW7 scatter of a 0.597)."""
    *_, text = _analysis(example)
    where, words = W20_WORDING[example]
    if where == 'tree cap':
        where, text = 'cap', _tree_text(example)
    parts = {
        'steps': [step.outcome for step in text.steps],
        'cap': [row.detail for row in text.checks if row.is_cap],
    }[where]
    assert any(words in part for part in parts), (words, parts)


# --------------------------------------------------------------------------------------------- layout

# 1627 13E-33: the densest right column of the 16 records ranked highest by caps, thin margins, episodes, owner and
# member table (CS11 survey: the only one of them that still overflowed before the physics block was placed beside
# the checks)
LAYOUT_RECORDS = ('mmr2__385266__16N-27+0+11', 'mmr2__82157__20N-29+0+9', 'mmr2__119066__4N-7+0+3', 'mmr2__1627__13E-33+0+20')


@pytest.mark.benchmark
@pytest.mark.parametrize('example', LAYOUT_RECORDS)
def test_block_is_drawn_and_nothing_overflows(example):
    from resonances.plotting import mmr_combined as mc

    from resonances.plotting import verdict_column as vc

    series, analysis, *_ = _analysis(example)
    fig = mc.draw_combined(analysis, series, settings=S)
    texts = ' '.join(artist.get_text() for artist in fig.texts)
    # the right column (2026-11-04): the physics sections are its Evidence block, drawn whole; nothing overflows
    column = fig.text_stats['column']
    assert fig.text_stats['overflow'] == 0 and fig.text_stats['not_drawn'] == 0 and vc.problems(column) == []
    assert column.evidence and all(vc.plain(evidence.title).split()[-1] in texts for evidence in column.evidence)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- W20 round C: figure text


@pytest.mark.benchmark
def test_ca138_return_times_are_printed_once():
    """2016 CA138: the return 79 -> 453 yr is said in the angle-shape step; the cap one_return_cycle names the cycle's
    length (453 - 79 = 374 yr) and points there instead of printing the times again. W34: the cap is the W28 tree's
    (its text); under the minimal policy it is a flag and the final text has no cap row."""
    *_, analysis, _, text = _analysis('coorbital__2016_CA138__1E-1+0+0')
    for shown in (text, _tree_text('coorbital__2016_CA138__1E-1+0+0')):
        everything = [step.outcome for step in shown.steps] + [row.detail for row in shown.checks]
        assert sum('crosses it back at' in part for part in everything) == 1
    assert 'one_return_cycle' in analysis.verdict.flags and not any(row.is_cap for row in text.checks)
    (cap,) = [row for row in _tree_text('coorbital__2016_CA138__1E-1+0+0').checks if row.key == 'cap:one_return_cycle']
    assert cap.detail.startswith('one cycle of 37') and 'angle-shape step' in cap.detail


def _event(kind, time_yr, observed, arc_rev):
    from resonances.resonance.analysis.geometry import EdgeEvent

    return EdgeEvent(kind, time_yr, 0.1, 3.0, False, observed, arc_rev * 2 * math.pi, observed)


def test_a_seen_exit_is_said_where_no_exit_was_printed():
    """check_mmr_exit pass means no edge is suspected, not that none is seen: an exit observed by the edge rule
    (UM52: at 949 yr, 1.97 rev) is said, pointing to the step that prints its arc; with no event, 'no exit or entry seen'."""
    geometry = SimpleNamespace(exit=_event('exit', 949.0, True, 1.97), entry=None)
    assert vt._edges_seen(SimpleNamespace(a=SimpleNamespace(geometry=geometry))) == 'exit observed (the angle-shape step), not suspected'
    geometry = SimpleNamespace(exit=None, entry=_event('entry', 300.0, False, 0.4))
    assert vt._edges_seen(SimpleNamespace(a=SimpleNamespace(geometry=geometry))) == 'entry at 300 yr, not suspected'
    geometry = SimpleNamespace(exit=None, entry=None)
    assert vt._edges_seen(SimpleNamespace(a=SimpleNamespace(geometry=geometry))) == 'no exit or entry seen'


def _facts(owner, flags=(), origin='induced', entries=()):
    path = {entry.name: entry for entry in entries}
    v = SimpleNamespace(flags=list(flags), libration_origin=origin, status='transient')
    return SimpleNamespace(a=SimpleNamespace(owner=owner), s=S, v=v, derived=False, entry=path.get)


def _owner(accepted, candidates=(), check='fail', flags=()):
    return SimpleNamespace(
        accepted=accepted, candidates=list(candidates), n_candidates=len(candidates) or 1, check=check, flags=list(flags)
    )


def test_the_owner_step_follows_the_decision():
    """433 1R-5V+7: the best candidate has R² 0.73 but an incoherent phase (0.41): the step names that test. (The
    refused owner of 1627 13E-33 and the induced leaf are worded by role since W28 A3b and tested on real result
    objects in test_verdict_text_roles.py.)"""
    best = SimpleNamespace(spec=SimpleNamespace(key='47M-44+0-3'), r2_median=0.73, phi0_coherence=0.41, reason='phase_incoherent', rank=1)
    step = vt._owner_step(_facts(_owner(None, [best]), origin='unknown'))
    assert step.outcome == '1 candidate, none drives a (best 47M-44+0-3: R² 0.73, but its phase φ0 does not hold, coherence 0.41)'


def test_an_arc_of_the_slow_centre_is_explained():
    """2003 KQ18: 7 turning points on a slow center that turns back once within 0.18 rev (G5a) is insufficient_time
    (arc), and the step says so instead of 'too few turning points'."""
    entry = LedgerEntry(
        'G5a slow centre stays within one revolution',
        0.18 * 2 * math.pi,
        Op.LT,
        2 * math.pi,
        'rad',
        'geo_centre_range_rad',
        '',
        False,
        True,
    )
    f = SimpleNamespace(row={'geo_n_tp': 7}, entry={entry.name: entry}.get)
    assert vt._insufficient_text(f, 'arc') == (
        '7 turning points ride on a slow center that turns back once, within 0.18 rev: part of one slow swing, where a '
        'drift and a slow libration look alike → insufficient_time (arc)'
    )
    f = SimpleNamespace(row={'geo_n_tp': 2}, entry={}.get)
    assert vt._insufficient_text(f, 'few_cycles') == '2 turning points, too few for a libration → insufficient_time'


def test_forced_waves_are_marked_in_the_episode_lists():
    """37087 1 Myr type: an accepted run judged a forced wave (episodes.csv forced_wave) is no episode of the angle."""
    runs = [
        {
            'episode_id': 0,
            'accepted': True,
            'forced_wave': 'lattice_line',
            't_start_yr': 89.0,
            't_end_yr': 483.0,
            'n_tp': 5,
            'period_med_yr': 168.0,
            'half_amp_med_deg': 20.0,
            'half_amp_max_deg': 25.0,
            'subtype': 'stationary',
        },
        {
            'episode_id': 1,
            'accepted': True,
            'forced_wave': '',
            't_start_yr': 600.0,
            't_end_yr': 900.0,
            'n_tp': 5,
            'period_med_yr': 150.0,
            'half_amp_med_deg': 20.0,
            'half_amp_max_deg': 25.0,
            'subtype': 'stationary',
        },
    ]
    lines = panel.explained_episode_lines(runs, 1)
    assert lines[0].group == 'EPISODES (2 accepted (1 forced wave), 0 rejected runs)'
    assert lines[0].value.endswith(': forced wave') and lines[0].style == 'grey'
    assert not lines[1].value.endswith('forced wave') and lines[1].style == 'bold'
    assert panel._episode_text(runs[0]).endswith(' forced wave')
