"""The Why block of the right column (`verdict_column_why`, brief block 4): every reason has a template, the sentences
carry no codes, and on real benchmark rows each claim is the record's (the expected numbers are read from the row's
columns or computed here from them independently, e.g. the free ω's turns |g − s| T / 1 296 000″)."""

import functools
import re
import sys
from pathlib import Path

import pytest

from resonances.plotting import verdict_column as vc
from resonances.plotting import verdict_column_why as why
from resonances.plotting import verdict_text as vt
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import Reason

S = settings_from_config()

# what a Why sentence must never print: column stems, W-labels (verdict_column.CODE_NAME, W_CODE), snake_case flag or
# reason names, rule ids ('G5w', 'B1', 'L4', '5d/6'), the pointer, and the chips' words (the action is said once)
SNAKE = re.compile(r'\b[a-z]+_[a-z_]+\b')
# (a Lagrange point, 'tadpole L4', is no rule id: the 1:1 regimes of 2016 JA)
RULE_ID = re.compile(r'(?<![\w+−-])(?<!tadpole )(?:[GBCKLROSFWZN]\d[a-z0-9*]*|\d[a-z]\*?(?:/\d[a-z]?)?)(?![\w+−-])')
CHIP_WORDS = ('extend to', 'resample to', 'glance at', 'review figure', 'spot check')


def _codes(text: str):
    return (
        vc.CODE_NAME.findall(text)
        + vc.W_CODE.findall(text)
        + SNAKE.findall(text)
        + RULE_ID.findall(text)
        + [word for word in CHIP_WORDS if word in text]
        + ([vc.POINTER] if vc.POINTER in text else [])
    )


def test_the_settings_the_expected_strings_assume():
    assert S.mmr_extend_slow_cycles == 3 and S.libration_coverage_periods == 1


# --------------------------------------------------------------------------------------------- templates


# the reasons the benchmark runs give (test-benchmark: 732 MMR rows, test-benchmark-secular-zlk: 89 rows, 2026-11-04)
MMR_OCCURRING = (
    'alias',
    'separatrix_zone',
    'circulation',
    'implausible_strength',
    'induced_libration',
    'kinematic_drift',
    'kinematic_libration',
    'no_resonant_potential',
    'dynamic_libration',
    'geometric_libration',
    'dynamic_episode',
    'geometric_episode',
    'subresonance_librates',
    'baseline_needed',
    'drift_mechanism_unknown',
    'one_coorbital_leg',
)
LONG_OCCURRING = (
    'unphysical',
    'circulation',
    'kinematic_libration',
    'dynamic_libration',
    'geometric_libration',
    'dynamic_episode',
    'geometric_episode',
    'baseline_needed',
)


def test_every_reason_has_an_mmr_template():
    """Every Reason value has an MMR template of its own (the occurring ones and the live code's others)."""
    templates = {str(reason) for reason in why.MMR_TEMPLATES}
    assert set(MMR_OCCURRING) <= templates
    assert {str(reason) for reason in Reason} <= templates


def test_every_occurring_long_period_reason_has_a_template():
    templates = {str(reason) for reason in why.LONG_TEMPLATES}
    assert set(LONG_OCCURRING) <= templates
    assert {'gate_unavailable', 'modulated_circulation', 'drift_mechanism_unknown'} <= templates


@pytest.mark.parametrize(
    'table',
    [why.FORCE_SEEN, why.VETO_WORDS, why.GATE_NO_VERDICT, why.OBJECTION_SEEN],
    ids=['force', 'veto', 'gate', 'objection'],
)
def test_the_word_tables_carry_no_codes(table):
    for value in table.values():
        # a value is its words, or a tuple of words and panel letters (GATE_NO_VERDICT: the short and the long words)
        for words in value if isinstance(value, tuple) else (value,):
            if isinstance(words, str) and len(words) > 1:
                assert not _codes(words), words


# --------------------------------------------------------------------------------------------- the limit and the head


@pytest.mark.parametrize(
    'inputs, level, expected',
    [
        # 79653 3M+3J-7 (test-benchmark): certain is blocked by M not measured and S thin, not by N or the force
        (
            'N:4.12>=3:certain;core:4>=3:certain;M:not_measured:high;S:thin:high;V:clean:certain;F:shown:certain',
            'high',
            'The confidence stops at high: the separatrix margin is not measured and the result depends on the filter cutoff.',
        ),
        # 591986 LK: a full veto allows at most high
        (
            'veto:full(zlk_kinematic):high;second_capture:none:certain',
            'high',
            'The confidence stops at high: the status rests on a veto, the winding free ω, which allows at most high.',
        ),
        # 2012 KL9 5E-2M-10: a thin veto, at most medium
        (
            'veto:thin(forced_wave):medium;second_capture:none:certain',
            'medium',
            'The confidence stops at medium: the status rests on a veto, the background’s forced wave with a thin margin, '
            'which allows at most medium.',
        ),
        # 2001 HW7 5J+6S-2: the full veto allows high, 'both at once' unresolved holds medium
        (
            'veto:full(owner):high;second_capture:unresolved:medium',
            'medium',
            'The confidence stops at medium: both resonances may act at once.',
        ),
        # 1997 GK3 3M+5S-3: a thin owner veto and 'both at once' both hold medium; the veto goes last
        (
            'veto:thin(owner):medium;second_capture:unresolved:medium',
            'medium',
            'The confidence stops at medium: both resonances may act at once; the status rests on a veto, another resonance '
            'that owns the libration with a thin margin, which allows at most medium.',
        ),
        # 115978 11M-19: one cycle counted holds low
        ('N:1<2:low;M:thin:high;V:clean:certain;F:shown:certain', 'low', 'The confidence stops at low: only 1 cycle counted.'),
        # 1921 2J-1: certain, the author's mockup sentence
        ('N:323>=3:certain;M:full:certain;S:full:certain;V:clean:certain;F:shown:certain', 'certain', why.NOTHING_LIMITS),
        ('', '', ''),
    ],
)
def test_the_limit_names_the_inputs_that_hold_the_level(inputs, level, expected):
    assert why.limit_sentence({'pol_level_inputs': inputs}, level) == expected


def test_an_owner_veto_is_named_by_what_was_measured():
    """The veto of an owner holds the level at high: the limit says which resonance drives a (the row's owner column,
    2001 HW7 5J+6S-2: 5E-16+0+11), never 'the owner found'; another veto keeps its words."""
    row = {'pol_level_inputs': 'veto:full(owner):high', 'owner': '5E-16+0+11'}
    assert why.limit_sentence(row, 'high') == (
        'The confidence stops at high: the status rests on a veto, driven by 5E-16+0+11, which allows at most high.'
    )
    row = {'pol_level_inputs': 'veto:full(kinematic):high'}
    assert why.limit_sentence(row, 'high') == (
        'The confidence stops at high: the status rests on a veto, the kinematic bound, which allows at most high.'
    )
    # 62243 1M-4J-1: the kinematic veto of a drift is no bound (review #52)
    row = {'pol_level_inputs': 'veto:full(kinematic):high', 'reason': 'kinematic_drift'}
    assert why.limit_sentence(row, 'high') == (
        'The confidence stops at high: the status rests on a veto, the kinematic drift, which allows at most high.'
    )


def test_an_objection_said_before_is_only_referred_to():
    row = {'pol_level_inputs': 'N:10.3>=3:certain;M:thin:high;V:ambiguous:low;F:none:low', 'pol_objection_check': 'force_law;potential'}
    assert why.limit_sentence(row, 'low', said=True) == (
        'The confidence stops at low: that measured objection and no test measured its own force.'
    )
    alone = why.limit_sentence(row, 'low')
    assert alone == (
        'The confidence stops at low: the measured objection that ȧ does not follow its force law and the energy of the '
        'cycles does not fit the potential built from a; no test measured its own force.'
    )


@pytest.mark.parametrize(
    'status, head',
    [('uncertain', why.NEEDED), ('invalid', why.NEEDED), ('transient', why.WOULD_CHANGE), ('non_resonant', why.WOULD_CHANGE)],
)
def test_the_would_change_head(status, head):
    assert why._would(status, ['a record of 300 kyr.', '']) == f'{head}a record of 300 kyr.'
    assert why._would(status, ['', ' ']) == ''


@pytest.mark.parametrize('fraction, words', [(0.6512, ', 65% of the record'), (0.004, ', under 1% of the record'), (float('nan'), '')])
def test_the_share_of_the_record(fraction, words):
    assert why._share(fraction) == words
    if words:  # with the record's span (52356: 35% of the 100 kyr record)
        assert why._share(fraction, 100_000.0) == words.replace('the record', 'the 100 kyr record')


@pytest.mark.parametrize(
    'spans, words',
    [
        # 52356 5J-2S-2: three episodes, every one printed (review #48), one unit at the end
        ([(2390, 10100), (26100, 33600), (51100, 70700)], '2.39–10.1, 26.1–33.6 and 51.1–70.7 kyr'),
        # 2020 PP1 ZLK: no 'and 1 more' for a third span (review #29)
        ([(8080, 18100), (43000, 56400), (283000, 329000)], '8.08–18.1, 43–56.4 and 283–329 kyr'),
        # the units differ: each span keeps its own
        ([(586, 665), (2000, 3000)], '586–665 yr and 2–3 kyr'),
        ([(1000, 2000), (3000, 4000), (5000, 6000), (7000, 8000)], '1–2 kyr, 3–4 kyr, 5–6 kyr and one more'),
    ],
)
def test_the_episode_spans(spans, words):
    assert why._spans([{'t_start_yr': t0, 't_end_yr': t1} for t0, t1 in spans]) == words


@pytest.mark.parametrize(
    'value, words',
    [(0.939, '0.939 of a revolution'), (-0.525, '0.525 of a revolution'), (1.0, '1 revolution'), (3.2, '3.2 revolutions')],
)
def test_a_drift_in_revolutions(value, words):
    """62243 and 142694 drift 0.939 and 0.525 of a revolution, never '0.939 revolution' (review #52, #72)."""
    assert why._revolutions(value) == words


@pytest.mark.parametrize(
    'definition, formula',
    [
        ('17*sigma-3*phi', 'Ψ = 17σ − 3φ'),  # 1997 GK3: σ = (3/17)φ + Ψ/17, so Ψ = 17σ − 3φ
        ('8*sigma-1*phi', 'Ψ = 8σ − φ'),  # 2001 HW7: σ = φ/8 + Ψ/8
        ('1*sigma+2*phi', 'Ψ = σ + 2φ'),
        ('', ''),
    ],
)
def test_psi_is_defined_where_the_identity_names_it(definition, formula):
    assert why.psi_formula(definition) == formula


def test_the_last_line_heads():
    """A transient row whose σ circulates between its episodes cannot become resonant by a longer record, and says so
    under its own head, never under 'Would change the verdict' (review #50, #90); an extension that only tests is a
    next check; an uncertain row says all under 'Needed'."""
    line = why._last_line('transient', 'σ does not librate during 32.5–61.3 kyr', (), ['a record of 200 kyr.'])
    assert line == ('A longer record cannot make it resonant: σ does not librate during 32.5–61.3 kyr. Next check: a record of 200 kyr.')
    assert why._last_line('transient', 'σ does not librate after 42.9 kyr') == (
        'A longer record cannot make it resonant: σ does not librate after 42.9 kyr.'
    )
    assert why._last_line('uncertain', '', (), ['a record of 20 Myr']) == 'Needed: a record of 20 Myr.'
    assert why._last_line('non_resonant', '', ['if Ψ/8 were shown to stay bounded as well, …']).startswith(why.WOULD_CHANGE)
    assert why._last_line('resonant') == ''


def test_an_optional_sentence_goes_first():
    """Six sentences: the three-body strength scale (OPTIONAL) is dropped, the limit sentence stays last."""
    sentences = ['One.', 'Two.', why.OPTIONAL + 'The scale.', 'Three.', 'Four.', why.NOTHING_LIMITS]
    assert why._fit(sentences) == ('One.', 'Two.', 'Three.', 'Four.', why.NOTHING_LIMITS)
    assert why._fit(['One.', why.OPTIONAL + 'the scale.']) == ('One.', 'The scale.')


@pytest.mark.parametrize(
    'value, words',
    [(359.7, '0°'), (359.4, '359°'), (270.0, '270°'), (-90.0, '270°'), (45.26, '45.3°'), (720.5, '0.5°'), (float('nan'), '')],
)
def test_a_centre_prints_in_zero_to_360(value, words):
    """Angle labels in [0°, 360°) (plotting/CLAUDE.md): 359.7° would round to '360°', so it is 0°; −90° is 270°."""
    assert why.centre_text(value) == words


def test_the_longest_gap_between_episodes():
    """Episodes 0–10, 15–40 and 70–80 yr: the gaps are 5 and 30 yr, the longest from 40 to 70 yr."""
    rows = [{'t_start_yr': t0, 't_end_yr': t1} for t0, t1 in ((0.0, 10.0), (15.0, 40.0), (70.0, 80.0))]
    assert why.longest_gap(rows) == 'during 40–70 yr'
    assert why.longest_gap(rows[:1]) == '' and why.longest_gap([rows[1], {'t_start_yr': 30.0, 't_end_yr': 50.0}]) == ''


def test_the_figure_tables_name_every_policy_value():
    """The words are keyed on the policy's own names (`policy.OBJECTION_*`, `DECIDED_*`, the vetoes) and the gate's
    no-verdict outcomes (`secular.decide.GATE_NO_VERDICT`): a renamed value fails here instead of printing its code."""
    from resonances.plotting import verdict_column_common as common
    from resonances.plotting import verdict_column_mmr as mmr
    from resonances.resonance.analysis import policy as pol
    from resonances.resonance.analysis.secular import decide

    objections = {value for name, value in vars(pol).items() if name.startswith('OBJECTION_')}
    assert objections <= set(common.OBJECTION_WORDS)
    assert objections - {pol.OBJECTION_ZLK_INTEGRAL} <= set(why.OBJECTION_SEEN)  # the e-i objection: `_ei_sentence`
    decided = {value for name, value in vars(pol).items() if name.startswith('DECIDED_')} - {pol.DECIDED_TREE, pol.DECIDED_VETO}
    mmr_vetoes = {pol.VETO_OWNER, pol.VETO_KINEMATIC, pol.VETO_FORCED_WAVE, pol.VETO_STRENGTH}
    assert decided | {pol.DECIDED_VETO + veto for veto in mmr_vetoes} <= set(mmr.POLICY_QUESTIONS)
    assert set(pol.ZLK_VETOES) <= set(common.VETO_WORDS)
    assert set(decide.GATE_NO_VERDICT) <= set(why.GATE_NO_VERDICT)


def test_an_unmeasured_alias_distance_is_said_so():
    """An invalid alias whose far ratio is not measured (NaN: a* unknown) says the distance is not measured, never
    '⟨a⟩ lies only far from the resonance'; a measured one says how far."""
    from resonances.resonance.analysis.ledger import LedgerEntry

    def alias(far_value, far_result):
        m = why._Mmr.__new__(why._Mmr)
        m.status = 'invalid'
        entries = {
            vt.C5_BRANCH: LedgerEntry(vt.C5_BRANCH, 1.0, result=True),
            vt.C5_FAR: LedgerEntry(vt.C5_FAR, far_value, result=far_result),
        }
        m.entry = entries.get
        return why._t_alias(m)[0]

    unmeasured = alias(float('nan'), None)
    assert 'the distance of ⟨a⟩ from the resonance is not measured' in unmeasured and 'lies only far' not in unmeasured
    assert '⟨a⟩ lies only 1.5 a-scales from the resonance' in alias(1.5, False)


@pytest.fixture(scope='module')
def zlk_kinematic():
    """The synthetic Lidov-Kozai body of test_verdict_column_long (591986's analogue): a free circle of 0.011 at
    222″/yr about a forced vector of 0.045 at g5, so ρ_max ≈ 0.011/0.028 < 1 and the gate is kinematic."""
    import numpy as np

    from resonances.data.const import PLANETARY_FREQUENCIES
    from resonances.resonance.analysis.secular import LongPeriodSeries
    from resonances.resonance.analysis.verdict import ResonanceType
    from resonances.resonance.analysis.zlk import analyse_zlk
    from resonances.secular import free_elements as fe

    rad = 2 * np.pi / 1296000.0
    g5, g6 = PLANETARY_FREQUENCIES['g5'], PLANETARY_FREQUENCIES['g6']
    t = np.arange(0.0, 500_001.0, 100.0)
    z_e = 0.045 * np.exp(1j * (g5 * rad * t + np.radians(83.0))) + 0.017 * np.exp(1j * (g6 * rad * t + np.radians(3.2)))
    z_e = z_e + 0.011 * np.exp(1j * 222.0 * rad * t)
    Omega = g5 * rad * t + np.radians(146.0) + np.radians(8.0) * np.sin(2 * np.pi * t / 1.5e6)
    e, varpi, inc = np.abs(z_e), np.angle(z_e), np.full_like(t, np.radians(39.0))
    omega = np.unwrap(varpi - Omega)
    series = LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, np.full_like(t, 5.2), e, inc)
    elements = fe.free_elements(t, e, inc, Omega, omega)
    analysis = analyse_zlk(series, S, free_elements=elements)
    return analysis, {**analysis.summary_row(), **fe.summary_fields(elements)}


def _zlk(analysis, verdict=None, gate=None):
    from dataclasses import replace

    return replace(analysis, verdict=replace(analysis.verdict, **(verdict or {})), gate=replace(analysis.gate, **(gate or {})))


def test_a_measured_gate_objection_is_named_not_called_no_verdict(zlk_kinematic):
    """policy._zlk_objection: a kinematic gate that pins no episode (zlk_gate_kinematic), or the free e below the forced
    e with an undefined phase (zlk_free_e), is a measured objection on the geometric status. The Why names it (with
    ρ_max for the free e) and the limit refers to it; it never says the gate gives no verdict."""
    analysis, row = zlk_kinematic
    episodic = {'status': 'transient', 'reason': Reason.GEOMETRIC_EPISODE, 'veto': '', 'confidence': 'low', 'angle_status': 'episodic'}
    base = {**row, 'pol_level_inputs': 'V:ambiguous:low', 'pol_decided_by': 'objection'}
    kinematic = _all(
        why.long_why(_zlk(analysis, episodic), S, {**base, 'pol_objection_check': 'zlk_gate_kinematic'}, analysis.episode_rows())
    )
    assert 'The free-ω gate measured an objection: the free ω winds' in kinematic and 'no verdict' not in kinematic
    assert 'The confidence stops at low: that measured objection.' in kinematic
    rho = vt.fmt(row['zlk_gate_rho_max'])
    undefined = _zlk(analysis, episodic, {'outcome': 'undefined_phase'})
    free_e = _all(why.long_why(undefined, S, {**base, 'pol_objection_check': 'zlk_free_e'}, analysis.episode_rows()))
    assert f'measured an objection: the free e stays below the forced e over the record (ρ_max {rho})' in free_e
    assert 'no verdict' not in free_e and float(rho) < 1
    # a gate without a verdict keeps its sentence
    short = _zlk(analysis, episodic, {'outcome': 'short_window'})
    unmeasured = {**base, 'pol_level_inputs': 'V:not_measured:medium', 'pol_decided_by': 'gate_unavailable'}
    assert 'The free-ω gate gives no verdict here: the record holds too few' in _all(
        why.long_why(short, S, unmeasured, analysis.episode_rows())
    )


def test_the_free_e_veto_says_where_it_pinned_the_free_e(zlk_kinematic):
    """policy._free_e_pins: ρ_max < 1 pins the free e over the whole record; else (ρ_max ≥ 1) ρ_episodes < 1 with the
    free ω turning ≥ ZLK_EPISODE_FREE_TURNS_PER_CYCLE per cycle pins it inside the episodes, and the Why says that, with
    those numbers (0.62, 2.5 turns per cycle), never 'over the whole record (at most 1.4 of it)'."""
    analysis, row = zlk_kinematic
    vetoed = _zlk(analysis, {'veto': 'zlk_free_e'})
    record = _all(why.long_why(vetoed, S, row, analysis.episode_rows()))
    assert f'stays below the forced e over the whole record (at most {vt.fmt(row["zlk_gate_rho_max"])} of it)' in record
    inside = {**row, 'zlk_gate_rho_max': 1.4, 'zlk_gate_rho_episodes': 0.62, 'zlk_gate_free_turns_episodes': 2.5}
    episodes = _all(why.long_why(vetoed, S, inside, analysis.episode_rows()))
    assert (
        'Inside every librating episode the free e stays below the forced e (at most 0.62 of it) while the free ω turns 2.5 times per cycle'
        in episodes
    )
    assert 'over the whole record (at most' not in episodes


# --------------------------------------------------------------------------------------------- benchmark rows


@functools.lru_cache(maxsize=None)
def _mmr(example: str):
    """(row, why) of a benchmark MMR row, analyzed as the figure is (`mmr_combined.figure_rows`)."""
    from resonances.plotting.mmr_combined import figure_rows
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), S)
    summary, numbers_row, episodes, _ = figure_rows(analysis)
    row = {**numbers_row, **summary}
    f = vt.MMRFacts(analysis, S, row, episodes)
    return row, why.mmr_why(f, row)


@functools.lru_cache(maxsize=None)
def _long(example: str):
    """(row, why) of a secular/ZLK benchmark record, analyzed as `reclassify_secular_zlk` does."""
    tools = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
    sys.path.insert(0, str(tools))
    try:
        import bench_io as bio
        import reclassify_secular_zlk as reclassify
    except ImportError:
        pytest.skip('benchmark tools not available')
    if not bio.example_dir(example).exists():
        pytest.skip('benchmark data not available')
    sim, body, resonance = reclassify.build_simulation(bio.load_example(example, with_planets=False))
    sim.identify_librations()
    analysis = sim.long_period_analysis(body, resonance)
    row = {**analysis.summary_row(), **reclassify._extra_columns(reclassify.summary_row(example, body, resonance))}
    return row, why.long_why(analysis, S, row, analysis.episode_rows()), analysis


def _all(w: vc.Why) -> str:
    return ' '.join(w.sentences + (w.would_change,))


def _check_form(w: vc.Why):
    """2–5 sentences, each ending with a full stop, no codes, at most one last line with one of its heads."""
    assert 2 <= len(w.sentences) <= why.MAX_SENTENCES
    for sentence in w.sentences + ((w.would_change,) if w.would_change else ()):
        assert sentence.endswith('.') and not _codes(vc.plain(sentence)), sentence
    assert not w.would_change or w.would_change.startswith((why.WOULD_CHANGE, why.NEEDED, why.NEXT_CHECK, why.NO_CHANGE))
    column = vc.Column('mmr2', vc.Header('resonant', 'certain', vc.Chip('none', '')), vc.Subtitle('', ''), why=w)
    assert [p for p in vc.problems(column) if p.startswith('why')] == []


MMR_CASES = (
    'mmr2__1921__2J-1+0-1',
    'mmr3__2001_HW7__5J+6S-2+0+0-9',
    'mmr3__79653__3M+3J-7+0+0+1',
    'mmr2__385458__9N-17+0+8',
    'mmr3__2012_KL9__5E-2M-10+0+0+7',
    'mmr3__285637__3M+7J-6+0+0-4',
    'mmr3__1999_UQ__7E+2M-9+0+0+0',
    'mmr2__2897__35S-4+0-31',
    'mmr3__142694__1V+3S-7+0+0+3',
    'coorbital__2007_VU6__1E-1+0+0',
    'coorbital__2016_JA__1E-1+0+0__5kyr',
    'coorbital__2016_JA__1E-1+0+0',
    'mmr2__523899__3N-5+0+2',
)
LONG_CASES = ('secular__1222__g-g6', 'secular__2306__g-g6+s-s6', 'zlk__3040__LK', 'zlk__591986__LK', 'zlk__88899__LK__500kyr')


@pytest.mark.benchmark
@pytest.mark.parametrize('example', MMR_CASES)
def test_mmr_why_form(example):
    _check_form(_mmr(example)[1])


@pytest.mark.benchmark
@pytest.mark.parametrize('example', LONG_CASES)
def test_long_why_form(example):
    _check_form(_long(example)[1])


@pytest.mark.benchmark
def test_1921_own_pendulum_and_what_was_ruled_out():
    """The author's case (2026-11-05): each test by what it means physically, with its number; the strength as the
    Evidence prints it (T₀ theory 344–386 yr against T₀ 214 yr from the observed cycles, ratio 2.43–3.37 in [0.1, 10]),
    never 'within a factor of two'; what was excluded as positive sentences, never 'Ruled out by measurement: …'."""
    row, w = _mmr('mmr2__1921__2J-1+0-1')
    assert w.sentences[0] == 'σ librates about 0° over the whole 100 kyr: 323 cycles counted, period 308 yr <<b, k>>.'
    assert w.sentences[1].startswith('The semi-major axis swings in step with σ as the pendulum of 2J-1 predicts: ȧ follows')
    assert 'R² 0.814 in the weaker half' in w.sentences[1]  # the halves 0.823 and 0.814 of step 5
    assert 'in 322 of 322 windows and stays clear of its separatrix' in w.sentences[1]
    assert w.sentences[2] == (
        'Its averaged potential predicts a small-swing period of 344–386 yr, against 214 yr derived from the observed '
        'cycles, so these cycles need 2.43–3.37 times its theoretical strength, inside the accepted band 0.1–10: 2J-1 is '
        'strong enough to make them.'
    )
    assert w.sentences[3].startswith('None of 38 neighboring resonances drives a (the best, 5S-1+0-4, fits a with R² ')
    assert 'Ruled out' not in _all(w) and 'factor of two' not in _all(w)
    assert w.sentences[4] == why.NOTHING_LIMITS and w.would_change == ''


@pytest.mark.benchmark
def test_2001_hw7_induced_names_the_owner_and_both_at_once():
    row, w = _mmr('mmr3__2001_HW7__5J+6S-2+0+0-9')
    text = _all(w)
    assert 'The libration is borrowed from 5E-16+0+11' in text and 'σ = φ/8 + Ψ/8' in text  # own_psi_def 8*sigma-1*phi
    assert 'Ψ = 8σ − φ' in text
    assert 'both resonances may act at once' in w.sentences[-1]  # the input that holds medium
    assert w.would_change.startswith(why.WOULD_CHANGE) and 'if Ψ/8 were shown to stay bounded as well' in w.would_change


@pytest.mark.benchmark
def test_79653_cannot_become_resonant_by_extension():
    """The circulation between 32.5 and 61.3 kyr stays in any longer record (the critic, spec review); the 200 kyr
    serve the force law of the last episode (clause C, pol_veto_need_yr)."""
    row, w = _mmr('mmr3__79653__3M+3J-7+0+0+1')
    assert row['pol_extend_clause'] == 'C' and row['pol_veto_need_yr'] == 200000.0
    assert w.would_change.count('resonant') == 1  # only in the head: a longer record cannot make it resonant
    assert w.would_change.startswith(why.NO_CHANGE + 'σ does not librate during 32.5–61.3 kyr.')
    assert why.NEXT_CHECK + 'a record of 200 kyr, long enough to test the force law in the last episode' in w.would_change
    assert w.sentences[-1] == (
        'The confidence stops at high: the separatrix margin is not measured and the result depends on the filter cutoff.'
    )
    # R0 was read inside episode #0 only (W34 `zone.episode_r0`), not the deciding one, so the sentence names it
    assert 'inside episode #0 (4.81–32.5 kyr), not the deciding one, the cycle energy fits' in w.sentences[1]
    assert 'force law' not in w.sentences[1]


@pytest.mark.benchmark
def test_385458_needs_three_slow_periods():
    row, w = _mmr('mmr2__385458__9N-17+0+8')
    period, required = row['base_period_yr'], row['base_required_kyr'] * 1e3
    assert required >= 3 * period  # required = 3 P + 2 trims of P_c (decide.baseline)
    assert w.would_change == (
        f'Needed: 3 slow periods of {vt.years(period)}, about {vt.years(required)} in all; next check: a record of '
        f'{vt.years(row["extend_to_years"])}.'
    )
    assert w.would_change == 'Needed: 3 slow periods of 76.3 kyr, about 230 kyr in all; next check: a record of 300 kyr.'
    assert 'swings 144° in between' in w.sentences[0]


@pytest.mark.benchmark
def test_2012_kl9_one_name_for_the_forced_wave():
    """Invariant 3 (one cause, one name): the forced-wave veto keeps reason kinematic_libration (D6), σ circulates."""
    row, w = _mmr('mmr3__2012_KL9__5E-2M-10+0+0+7')
    text = _all(w)
    assert 'σ circulates' in w.sentences[0] and 'σ librates' not in text
    assert 'They are the background’s forced wave, not librations' in text and 'strength' not in text


@pytest.mark.benchmark
def test_285637_kinematic_term_and_a():
    row, w = _mmr('mmr3__285637__3M+7J-6+0+0-4')
    text = _all(w)
    assert 'the swing comes from σ’s −4ϖ part, not from 3M+7J-6’s force; that part swings with the 2ω cycle' in text
    assert 'it carries only 6.04e-5 of the swing' in text  # one exponent style (`verdict_text.sci`)
    assert 'whether another resonance drives a' not in text  # a does not move: no driver of a to look for


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, words',
    [
        ('mmr3__1999_UQ__7E+2M-9+0+0+0', ('Only one alias branch fits the range of a', '66.2 a-scales')),
        ('mmr2__2897__35S-4+0-31', ('9.63e35 times the strength of 35S-4’s averaged resonant term',)),
        ('coorbital__2007_VU6__1E-1+0+0', ('its local energy integral fails', 'Needed: a record of 2 kyr')),
        ('coorbital__2016_JA__1E-1+0+0__5kyr', ('not shown to be 1E-1’s own', 'that measured objection')),
        ('coorbital__2016_JA__1E-1+0+0', (why.NO_CHANGE + 'σ does not librate after 537 yr',)),
        ('mmr2__523899__3N-5+0+2', ('near the separatrix', 'σ̇ does not follow a')),
    ],
)
def test_mmr_claims(example, words):
    text = _all(_mmr(example)[1])
    for word in words:
        assert word in text, (word, text)


@pytest.mark.benchmark
def test_142694_needs_the_deciding_period():
    row, w = _mmr('mmr3__142694__1V+3S-7+0+0+3')
    assert row['pol_extend_clause'] == 'B'
    assert w.would_change == (
        f'Needed: a record of {vt.years(row["extend_to_years"])}, since the deciding period '
        f'({vt.years(row["pol_period_dec_yr"])}) is too long for this record.'
    )


@pytest.mark.benchmark
def test_591986_free_omega_winds_and_the_forced_vector():
    row, w, analysis = _long('zlk__591986__LK')
    rate = abs(row['free_b_omega']) * vt.record_span_yr(analysis) / 1296000.0  # |g − s| T in turns, ~83.9
    net = abs(row['free_omega_revolutions'])  # the measured net turns, the number Key facts prints
    text = _all(w)
    assert f'winds instead: {vt.fmt(net)} net turns over the record' in text and net > 80 and abs(net - rate) < 2
    # the lock columns: Ω − ϖ5 closes no cycle in the record, so the node follows g5 only within the resolution
    assert 'free/forced 0.252' in text and 'The node moves at the rate of g5 as closely as the record can resolve' in text
    assert 'locked to g5' not in text
    # the libration period is the free–forced beat, the gate sub-check's number
    assert f'2π/|g − g5| = {vt.years(analysis.zlk.attribution.beat_period_yr)}, is the libration period of ω' in text
    assert w.would_change == ''


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, words',
    [
        ('secular__1222__g-g6', ('librates about 180° over the whole 10 Myr', 'rests on the shape of σ alone', '70.3°')),
        ('secular__2306__g-g6+s-s6', ('Needed: a record of 20 Myr', 'has turned back only once')),
        ('zlk__3040__LK', ('It is a Lidov–Kozai (ZLK) libration', 'stays within a 63.6° range', 'trade in antiphase')),
        # the gate excludes the forced libration inside the episodes, but e and i object: never called Lidov–Kozai
        ('zlk__88899__LK__500kyr', ('Inside each episode the free ω', 'e and i do not trade', 'that measured objection')),
    ],
)
def test_long_claims(example, words):
    text = _all(_long(example)[1])
    for word in words:
        assert word in text, (word, text)


@pytest.mark.benchmark
def test_88899_is_not_called_lidov_kozai_while_e_and_i_object():
    """Review #5: the gate excludes only the forced libration; with e and i not trading as a ZLK cycle (η_H 1.24 > 0.5)
    the paragraph never says 'Lidov–Kozai libration', and the free ω's winding over the whole record is said."""
    row, w, _ = _long('zlk__88899__LK__500kyr')
    text = _all(w)
    assert 'It is a Lidov–Kozai' not in text and 'η_H 1.24 > 0.5' in text
    assert f'over the whole record the free ω winds ({vt.fmt(abs(row["free_omega_revolutions"]))} net turns)' in text
