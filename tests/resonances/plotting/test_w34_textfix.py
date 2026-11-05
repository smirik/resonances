"""W34 text fixes of the right column (no status changes): the words of the verdict figure that disagreed with the
numbers they print (the W34 text fix, items A2–A12 without A10, and the W28 Codex defects S5, S8, S9; A1, A10 and the
flag words C reached only the old, undrawn engine and were dropped).

Every expected string is worked out by hand from the numbers given (the benchmark rows they come from are named;
their own tests are in the benchmark-marked tests of test_verdict_text.py)."""

import functools
import math
import re
from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

import pytest  # noqa: E402

from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.resonance.analysis.ledger import LedgerEntry, Margin, Op  # noqa: E402
from resonances.resonance.analysis.mmr.decide import C5_BRANCH, C5_FAR  # noqa: E402
from resonances.resonance.analysis.pendulum import CycleClass, EnergyAnalysis  # noqa: E402
from resonances.resonance.analysis.secular.models import EpisodeCoupling  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import MarginClass, MarginValue  # noqa: E402

S = settings_from_config()


def test_the_settings_the_expected_strings_assume():
    assert (S.mmr_l2_amp_full_deg, S.mmr_w_r2_none, S.mmr_owner_r2_min) == (120.0, 0.4, 0.7)
    assert (S.zlk_ei_phase_tol_deg, S.zlk_ei_episode_min_coherence, S.zlk_ei_h_residual_max) == (45.0, 0.5, 0.5)
    assert S.zlk_rho_kinematic_excluded == 3.0 and S.mmr_c5_far_ratio_min == 3.0


# --------------------------------------------------------------------------------------------- A3: large numbers


@pytest.mark.parametrize(
    'value, expected',
    [
        (963013857750307201977827653771591680.0, '9.63e35'),  # 2897: the ratio printed in full before (36 digits)
        (3294877922321982.0, '3.29e15'),  # 2897: the width ratio
        (2.94363697e17, '2.94e17'),
        (999999.0, '999\u2009999'),  # below 1e6: grouped by a thin space, as before
        (1e6, '1e6'),
        (-2.5e7, '-2.5e7'),
        (9.9999e6, '1e7'),  # rounding carries into the exponent
    ],
)
def test_numbers_from_a_million_are_scientific(value, expected):
    assert vt.fmt(value) == expected


@pytest.mark.parametrize(
    'value, expected',
    [
        (999999.7, '1e6'),  # the grouped form would round it to 1 000 000: the exponent form instead
        (999999.4, '999\u2009999'),
        (-999999.7, '-1e6'),
    ],
)
def test_the_exponent_form_is_chosen_on_the_rounded_value(value, expected):
    assert vt.fmt(value) == expected


@pytest.mark.parametrize(
    't0, t1, expected',
    [
        (500.0, 999.7, '0.5–1 kyr'),  # 999.7 yr rounds to 1 kyr, as years() prints it: never '500–1000 yr'
        (20000.0, 999990.0, '0.02–1 Myr'),  # never '20–1000 kyr'
        (586.0, 665.0, '586–665 yr'),
        (28500.0, 32300.0, '28.5–32.3 kyr'),
        (139806.0, 140182.0, '139.8–140.2 kyr'),  # ends that round alike get the digits that tell them apart
        (1000.0, 1000.4, '1.0000–1.0004 kyr'),
        (140000.0, 140000.0, '140–140 kyr'),  # equal ends stay as they are
    ],
)
def test_an_interval_takes_its_unit_from_the_rounded_end(t0, t1, expected):
    assert vt.interval(t0, t1) == expected
    assert vt.years(999.7) == '1 kyr' and vt.years(999990.0) == '1 Myr'


def test_a_huge_comparison_reads_as_it_came_out():
    assert vt.compare(9.63013857750307e35, '>=', 1000.0) == '9.63e35 ≥ 1\u2009000'
    assert vt.years(2.94363697e20) == '2.94e14 Myr'  # 2.94e20 yr = 2.94e14 Myr


# --------------------------------------------------------------------------------------------- A2: the DECISIVE names


def _named_facts(sub_accepted=None, owner_accepted=None, flags=(), cause_role='owner', strength='pass'):
    owner = SimpleNamespace(accepted=owner_accepted, flags=[], identity=None) if owner_accepted is not None else None
    sub = SimpleNamespace(accepted=sub_accepted) if sub_accepted is not None else None
    verdict = SimpleNamespace(flags=list(flags), cause_role=cause_role, angle_subtype='')  # G6 is named by the subtype
    analysis = SimpleNamespace(subresonances=sub, owner=owner, response=None)
    return SimpleNamespace(a=analysis, v=verdict, checks={'strength': strength}, coorbital=False, f_kinematic=False)


def _key(key):
    return SimpleNamespace(spec=SimpleNamespace(key=key), route='')


def test_the_decisive_line_names_what_the_comparison_found():
    """82831 (parent): 'no member holds the center [S]' while 13M-24+2+9 holds it; 117306 and 1997 GK3: 'no other
    resonance drives a [W]' while a driver (11M-20+0+9) or an owner (10E-17+0+7) drives it; 2897: F failed."""
    assert vt.decisive_name(_named_facts(sub_accepted=_key('13M-24+2+9')), 'S') == 'member 13M-24+2+9 holds the center'
    assert vt.decisive_name(_named_facts(owner_accepted=_key('11M-20+0+9'), cause_role='driver'), 'W') == 'driver 11M-20+0+9 drives a'
    assert vt.decisive_name(_named_facts(owner_accepted=_key('10E-17+0+7')), 'W') == 'owner 10E-17+0+7 drives a'
    refused = _named_facts(owner_accepted=_key('5M-2+0+1'), flags=['owner_outside_episode'])
    assert vt.decisive_name(refused, 'W') == 'candidate driver 5M-2+0+1'
    assert vt.decisive_name(_named_facts(), 'S') == vt.rule_name('S') == 'no member holds the center'
    assert vt.decisive_name(_named_facts(), 'W') == 'no other resonance drives a'
    assert vt.decisive_name(_named_facts(strength='fail'), 'F') == vt.STRENGTH_FAILED_NAME
    assert vt.decisive_name(_named_facts(), 'F') == vt.rule_name('F')


# --------------------------------------------------------------------------------------------- A4: the alias


def _entry(name, value=None, threshold=None, result=None, op=Op.GE):
    return LedgerEntry(name, value=value, op=op, threshold=threshold, result=result)


def _alias_facts(far_value, far_ok, branch_ok, status='invalid'):
    path = {C5_FAR: _entry(C5_FAR, far_value, 3.0, far_ok), C5_BRANCH: _entry(C5_BRANCH, 1.0, 1.0, branch_ok, Op.EQ)}
    return SimpleNamespace(v=SimpleNamespace(status=status), s=S, path=path, entry=path.get)


def test_the_alias_says_which_part_failed():
    """1864 51M-50 (invalid): ⟨a⟩ is 14.4 a-scales from a* (far: passes), but several branches fit: the old text read
    'or ⟨a⟩ near a* (14.4 ≥ 3 a-scales)', as if 14.4 were near."""
    assert vt._alias_sentence(_alias_facts(14.4, True, False)) == (
        'Invalid: several alias branches of ν·Δt fit the a range (the folding is ambiguous), although ⟨a⟩ lies 14.4 ≥ 3 '
        'a-scales from a*: a libration hidden by the step cannot be excluded.'
    )
    assert vt._alias_sentence(_alias_facts(1.2, False, True)) == (
        'Invalid: ⟨a⟩ lies only 1.2 < 3 a-scales from a*: a libration hidden by the step cannot be excluded.'
    )
    both = vt._alias_sentence(_alias_facts(1.2, False, False))
    assert both.startswith('Invalid: several alias branches') and '⟨a⟩ lies only 1.2 < 3' in both


# --------------------------------------------------------------------------------------------- A5, A6: L2


def _energy(classes):
    return EnergyAnalysis(None, None, math.nan, math.nan, zone=object(), cycle_classes=[CycleClass(c) for c in classes])


def test_l2_graded_by_the_amplitude_prints_the_amplitude():
    """2003 UG22 40 kyr: the tree read the episode before the jump (no zone there), so L2 was graded by 138.47° >
    120° (thin); the record's zone holds none of the 181 cycles, which the row printed alone."""
    f = SimpleNamespace(a=SimpleNamespace(energy=_energy(['above'] * 181)), s=S)
    margin = MarginValue(Margin.THIN, 138.46924470167357, MarginClass.DECISIVE)
    detail, need, note = vt._detail_l2(f, margin)
    assert detail == (
        'max half-amplitude 138° > 120° (graded without a zone: the record’s zone, 0 of 181 cycles in the zone, is not ' 'this regime’s)'
    )
    assert (need, note) == ('≤ 120°', '')
    # graded by the zone: the cycles, as before
    full = MarginValue(Margin.FULL, 96.1, MarginClass.DECISIVE)
    assert vt._detail_l2(SimpleNamespace(a=SimpleNamespace(energy=_energy(['above'] * 125)), s=S), full)[0] == (
        '0 of 125 cycles in the zone (max half-amplitude 96.1°)'
    )
    # no zone at all: the amplitude
    none = SimpleNamespace(a=SimpleNamespace(energy=None), s=S)
    assert vt._detail_l2(none, MarginValue(Margin.FULL, 21.4, MarginClass.DECISIVE))[0] == 'max half-amplitude 21.4° ≤ 120°'


def test_a_coorbital_l2_says_m_reads_the_zone():
    """2002 AA29 (1:1, resonant/certain): L2 '? not measured' beside 'amplitude clear of the separatrix' in the basis:
    the policy's M of a 1:1 reads the zone (no zone cycle → clear), not L2."""
    f = SimpleNamespace(
        a=SimpleNamespace(energy=_energy(['not_measured'] * 4), geometry=SimpleNamespace(coorbital=True)),
        v=SimpleNamespace(policy='minimal'),
    )
    state, why = vt._skip_l2(f)
    assert state == vt.CheckState.SKIPPED
    assert why == (
        '4 cycles above a lower saddle, which this test cannot grade; the level reads the 1:1 zone instead: 0 of 4 cycles '
        'in it → clear of the separatrix'
    )
    f.a.energy = _energy(['zone', 'above'])
    assert vt._skip_l2(f)[1] == 'not graded on a 1:1; the level reads the 1:1 zone instead: 1 of 2 cycles in it → near the separatrix'
    f.v.policy = ''
    assert vt._skip_l2(f) == (vt.CheckState.MISSING, 'not measured')


# --------------------------------------------------------------------------------------------- A7: one K2 row


def test_two_entries_of_one_family_print_one_row():
    """79653: K2 near its convention on both halves of the force law (0.382 < 0.4 each) printed twice."""
    from resonances.resonance.analysis.verdict import Verdict
    from tests.resonances.analysis.builders import make

    names = [f'3 kinematic (slow centre): a does not answer sigma, force law R2 (half {i})' for i in (1, 2)]
    path = [LedgerEntry(name, value=0.382, op=Op.LT, threshold=0.4, result=True) for name in names]
    verdict = make(Verdict, decision_path=path, thin_margins=[f'U:{name}' for name in names])
    rows = vt.near_threshold_rows(verdict, S)
    assert len(rows) == 1 and rows[0].detail == '0.382 < 0.4'
    path[1] = LedgerEntry(names[1], value=0.39, op=Op.LT, threshold=0.4, result=True)
    verdict = make(Verdict, decision_path=path, thin_margins=[f'U:{name}' for name in names])
    assert [row.detail for row in vt.near_threshold_rows(verdict, S)] == ['0.382 < 0.4; 0.39 < 0.4']


# --------------------------------------------------------------------------------------------- A8, A9: ZLK e-i


def _coupling(**fields):
    base = dict(phase_tol_deg=45.0, min_coherence=0.5, h_residual_max=0.5, min_cycles=3.0)
    return EpisodeCoupling(**{**base, **fields})


def test_the_episode_ei_tests_as_they_came_out():
    """88899 500 kyr, the deciding episode #1 (summary.csv): phase 261.92°, coherence -0.7769, eta_H 1.244. By hand:
    |261.92 - 180| = 81.92° > 45°; -0.777 ≤ -0.5; 1.24 > 0.5."""
    ei = _coupling(
        coupling='none', episode_id=1, period_yr=33730.0, n_periods=3.0427, phase_deg=261.9213, coherence=-0.77686, eta_h=1.24427
    )
    assert vt.episode_ei_parts(ei) == [
        'i against e 262° (81.9° > 45° from antiphase)',
        'coherence -0.777 ≤ -0.5',
        'η_H 1.24 > 0.5 (√(1−e²) cos i not kept)',
    ]
    gate = SimpleNamespace(episode_ei=ei, rho_max=13.27)
    analysis = SimpleNamespace(verdict=SimpleNamespace(flags=['ei_integral_not_held'], caps=[], confidence='low'), gate=gate)
    assert vt.episode_ei_objection(analysis) == (
        'the e-i integral is not held in the deciding episode #1 (at its period 33.7 kyr, 3.04 periods): i against e 262° '
        '(81.9° > 45° from antiphase), coherence -0.777 ≤ -0.5, η_H 1.24 > 0.5 (√(1−e²) cos i not kept): neither law; '
        'a measured objection, so the level is low (the record’s e-i line below is not what the level read)'
    )
    analysis.verdict.flags = []
    assert vt.episode_ei_objection(analysis) == ''


def test_kinematic_excluded_says_why_the_level_holds():
    """1981 Midas (resonant/certain, no gate verdict): rho_max 69.28 ≥ 3 and the episode's integral holds (phase
    180.06°, coherence -0.996, eta_H 0.0539)."""
    ei = _coupling(coupling='zlk', episode_id=0, period_yr=26260.0, phase_deg=180.0557, coherence=-0.99609, eta_h=0.05395)
    gate = SimpleNamespace(episode_ei=ei, rho_max=69.2845)
    analysis = SimpleNamespace(
        verdict=SimpleNamespace(flags=['gate_unavailable', 'kinematic_excluded'], caps=[], confidence='certain'), gate=gate
    )
    assert vt.kinematic_excluded_line(analysis, S) == (
        'Why no cap without a gate verdict: the forced terms are too small to hold the perihelion (ρ_max 69.3 ≥ 3) and '
        'the e-i integral holds in the deciding episode #0 (i against e 180° (0.0557° ≤ 45° from antiphase), coherence '
        '-0.996 ≤ -0.5, η_H 0.0539 ≤ 0.5 (√(1−e²) cos i kept)), so a kinematic libration is excluded (flag '
        'kinematic_excluded): the level follows the evidence (CERTAIN).'
    )
    analysis.verdict.flags = ['gate_unavailable']
    assert vt.kinematic_excluded_line(analysis, S) == ''


# --------------------------------------------------------------------------------------------- A12: unphysical


def test_an_unphysical_orbit_says_why_with_the_value():
    """2010 HY22 (invalid, unphysical 'hyperbolic'): ctx_e_max 2.1097, ctx_a_min_au -505.85."""
    analysis = SimpleNamespace(unphysical=True, unphysical_reason='hyperbolic', series=None)
    row = {'ctx_e_max': 2.10974085077, 'ctx_a_min_au': -505.846537229}
    assert vt.unphysical_text(analysis, row) == 'e reaches 2.11 > 1 (hyperbolic)'
    analysis.unphysical_reason = 'negative_sma'
    assert vt.unphysical_text(analysis, row) == 'a falls to -506 au < 0 au (unbound)'
    analysis.unphysical_reason = 'non_finite'
    assert vt.unphysical_text(analysis, row) == 'a or e is not finite (the integration lost the orbit)'
    series = SimpleNamespace(e=[0.1, float('nan'), 1.5], a=[2.0, 2.1, 2.2])  # without the row: the series' own maximum
    assert vt.unphysical_text(SimpleNamespace(unphysical=True, unphysical_reason='hyperbolic', series=series), {}) == (
        'e reaches 1.5 > 1 (hyperbolic)'
    )
    assert vt.unphysical_text(SimpleNamespace(unphysical=False, unphysical_reason='', series=None), {}) == ''


# --------------------------------------------------------------------------------------------- A11: the W thin band


def _w_facts(r2, reason='r2_low', coherence=float('nan')):
    candidate = SimpleNamespace(spec=SimpleNamespace(key='34M-21+0-13'), r2_median=r2, reason=reason, rank=0, phi0_coherence=coherence)
    owner = SimpleNamespace(accepted=None, candidates=[candidate], n_candidates=3, flags=[])
    return SimpleNamespace(a=SimpleNamespace(owner=owner), s=S)


def test_the_w_row_states_the_thin_band():
    """2013 AS27: best R² 0.63 printed '0.63 ≥ 0.4 · need best R² < 0.4' (thin, but why?); 2897: 0.97, refused for
    its phase (coherence 0.31)."""
    need = 'best R² < 0.4 (thin from 0.4; an owner needs ≥ 0.7)'
    assert vt._w_detail(_w_facts(0.63)) == (
        '3 candidates; best 34M-21+0-13 R² 0.63 in the thin band 0.4–0.7: it fits a, not accepted as an owner (thin)',
        need,
    )
    assert vt._w_detail(_w_facts(0.97, 'phase_incoherent', 0.31)) == (
        '3 candidates; best 34M-21+0-13 R² 0.97 ≥ 0.7, but its phase φ0 does not hold, coherence 0.31: it fits a, not '
        'accepted as an owner (thin)',
        need,
    )
    assert vt._w_detail(_w_facts(0.28)) == ('3 candidates; best 34M-21+0-13 R² 0.28 < 0.4', need)


# --------------------------------------------------------------------------------------------- B: S5, S8, S9


def test_span_tells_close_ends_apart():
    """2001 HW7 150 kyr: its late episode 139.806–140.182 kyr printed '140 kyr–140 kyr'."""
    assert vt._span(139806.0, 140182.0) == '139.8–140.2 kyr'
    assert vt._span(494.0, 46300.0) == '494 yr–46.3 kyr'
    assert vt._span(1000.0, 1000.4) == '1.0000–1.0004 kyr'


def test_the_capture_window_names_the_owner_blocks_it_spans():
    """2001 HW7 150 kyr (S5): x = Ψ/u tested on 494 yr–140.18 kyr, the hull of the owner's blocks 494 yr–46.34 kyr and
    139.81–140.18 kyr: by hand 140182 - 494 - (46340 - 494) - (140182 - 139806) = 93466 yr between them."""
    capture = SimpleNamespace(t_start_yr=494.0, t_end_yr=140182.0)
    owner = SimpleNamespace(capture=capture, intervals=[(494.0, 46340.0, 'libration'), (139806.0, 140182.0, 'libration')])
    f = SimpleNamespace(a=SimpleNamespace(owner=owner))
    assert vt.capture_hull(f) == ', over both stretches of the owner, including the 93.5 kyr between them'
    owner.intervals = [(494.0, 140182.0, 'libration')]
    assert vt.capture_hull(f) == ''


def _leg_row(key, state=vt.CheckState.THIN):
    return vt.CheckRow(key, vt.MMR_MARGIN_NAMES[key], state, 'x', counts='thin')


def test_a_single_leg_capture_does_not_read_the_circulation_margins():
    """2024 XP16, 2023 OE5, 2025 YU11 (G4c, transient) (S8): C1–C4 measure the circulation around the leg; they go to
    the gray n/a line with that note, C5 (the sampling) stays."""
    f = SimpleNamespace(v=SimpleNamespace(policy='minimal', status='transient'), rule='G4c')
    rows = vt._around_leg(f, [_leg_row(key) for key in ('C1', 'C2', 'C3', 'C4', 'C5')])
    assert [row.kind for row in rows] == [vt.KIND_NA] * 4 + [vt.KIND_SUPPORT]
    assert vt.na_line([row for row in rows if row.kind == vt.KIND_NA]) == (
        'the circulation around the leg, not the leg: center passes ≥ 10 rev [C1], no turning points at any cutoff [C2], '
        'dσ/dt smooth (one sign, no stalls) [C3], a-wave as a circulation predicts [C4]'
    )
    f.v.status = 'non_resonant'  # the circulation leaf: they are its support
    assert {row.kind for row in vt._around_leg(f, [_leg_row('C1')])} == {vt.KIND_SUPPORT}
    plain = vt.CheckRow('L3', 'σ̇ follows a − a*', vt.CheckState.NOT_APPLICABLE, kind=vt.KIND_NA)
    assert vt.na_line([plain]) == 'not for this record: σ̇ follows a − a* [L3]'


def test_filter_too_slow_is_said_where_it_is_the_cause():
    """2026 AC4 (S9): P_cyc 94.2 yr at P_c 55.52 yr passes with gain 0.892 < 0.911 (the 0.911 of
    `preprocess.gain_too_slow`'s docstring: G = 1/(1 + 1.8^-4) for N = 2 at 0.9 × 2 P_c)."""
    f = SimpleNamespace(
        a=SimpleNamespace(prep=SimpleNamespace(filter_too_slow=True, gain_at_period_cycle=0.892)),
        s=S,
        row={'geo_period_cycle_yr': 94.2, 'flt_cutoff_yr': 55.52},
    )
    assert vt.filter_too_slow_text(f) == 'the filter is too slow for the period (P_cyc 94.2 yr at P_c 55.5 yr, gain 0.892 < 0.911)'
    assert math.isclose(1.0 / (1.0 + 1.8**-4), 0.9130, abs_tol=5e-4)  # the analog gain the threshold reads at N = 2
    f.a.prep.filter_too_slow = False
    assert vt.filter_too_slow_text(f) == ''


# --------------------------------------------------------------------------------------------- the benchmark rows


@functools.lru_cache(maxsize=None)
def _mmr_text(example: str, derived: str = ''):
    """(analysis, text) of a benchmark MMR row as the figure explains it (`mmr_combined.figure_rows`); `derived`: the
    key of a derived subresonance row."""
    from resonances.plotting.mmr_combined import figure_rows
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), S)
    if derived:
        analysis = next(a for a in analysis.derived if a.resonance.key == derived)
    summary, numbers_row, episodes, _ = figure_rows(analysis)
    return analysis, vt.explain_mmr(analysis, S, {**numbers_row, **summary}, episodes)


def _slot(text, slot):
    return ' | '.join(bullet.text for bullet in text.bullets if bullet.slot == slot)


def _emphasis(text, rule_id):
    return next(item for item in text.emphasis if item.rule_id == rule_id)


@pytest.mark.benchmark
def test_79653_prints_k2_once_and_r0_as_decisive():
    analysis, text = _mmr_text('mmr3__79653__3M+3J-7+0+0+1')
    k2 = [row for row in text.checks if 'perihelion term (K2)' in row.name]
    assert len(k2) == 1
    r0 = _emphasis(text, 'R0')
    assert r0.kind == vt.EMPHASIS_DECISIVE and 'inside episode #0' in r0.parts[0].text


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example, derived, rule_id, name',
    [
        ('mmr2__82831__13M-24+0+11', '', 'S', 'member 13M-24+2+9 holds the center'),
        ('mmr3__117306__1V+5J-7+0+0+1', '', 'W', 'driver 11M-20+0+9 drives a'),
        ('mmr3__1997_GK3__3M+5S-3+0+0-5', '', 'W', 'owner 10E-17+0+7 drives a'),
        ('mmr3__2001_HW7__5J+6S-2+0+0-9__150kyr', '', 'W', 'owner 5E-16+0+11 drives a'),
    ],
)
def test_the_decisive_label_names_the_accepted_member_or_owner(example, derived, rule_id, name):
    _, text = _mmr_text(example, derived)
    assert _emphasis(text, rule_id).name == name


@pytest.mark.benchmark
def test_2897_the_strength_leaf_and_its_numbers():
    analysis, text = _mmr_text('mmr2__2897__35S-4+0-31')
    leaf = next(step for step in text.steps if step.state == vt.StepState.LEAF)
    assert leaf.number == '5s' and leaf.ids == ('L4', 'R0', 'F') and 'would need 9.63e35 × the strength' in leaf.outcome
    assert _emphasis(text, 'F').name == vt.STRENGTH_FAILED_NAME
    assert not any(item.rule_id == 'G1c' for item in text.emphasis)
    decisive = _slot(text, vt.SLOT_DECISIVE)
    assert 'not counted (not drawn on (a); (l) labels them forced (strength))' in decisive and 'drawn as' not in decisive
    printed = ' '.join([decisive] + [row.text for section in text.physics for row in section.rows])
    assert not re.search(r'\d \d{3} \d{3} \d{3}', printed)  # no number of 1e9 or more in full
    w = next(row for row in text.checks if row.key == 'W')
    assert 'R² 0.97 ≥ 0.7, but its phase φ0 does not hold' in w.detail


@pytest.mark.benchmark
def test_1864_alias_says_several_branches():
    _, text = _mmr_text('mmr2__1864__51M-50+0-1')
    assert _slot(text, vt.SLOT_DECISIVE).startswith('several alias branches of ν·Δt fit the a range (the folding is ambiguous), although')


@pytest.mark.benchmark
def test_ug22_40kyr_l2_prints_the_amplitude_that_made_it_thin():
    _, text = _mmr_text('mmr2__2003_UG22__10E-17+0+7__40kyr')
    l2 = next(row for row in text.checks if row.key == 'L2')
    assert l2.detail.startswith('max half-amplitude 138° > 120° (graded without a zone')


@pytest.mark.benchmark
@pytest.mark.parametrize('example', ['coorbital__2002_AA29__1E-1+0+0', 'coorbital__164207__1E-1+0+0'])
def test_a_coorbital_l2_row_says_m_reads_the_zone(example):
    _, text = _mmr_text(example)
    l2 = next(row for row in text.checks if row.key == 'L2')
    assert 'reads the 1:1 zone instead' in l2.note and l2.note.endswith('cycles in it → clear of the separatrix')
    assert 'amplitude clear of the separatrix' in _slot(text, vt.SLOT_POLICY)


@pytest.mark.benchmark
def test_as27_w_row_states_the_band():
    _, text = _mmr_text('mmr2__2013_AS27__31E-36+0+5')
    w = next(row for row in text.checks if row.key == 'W')
    assert 'in the thin band 0.4–0.7' in w.detail


@pytest.mark.benchmark
def test_hw7_150kyr_capture_window_and_spans():
    _, text = _mmr_text('mmr3__2001_HW7__5J+6S-2+0+0-9__150kyr')
    capture = next(row.text for section in text.physics for row in section.rows if row.label == vt.CAPTURE_LABEL)
    assert 'over both stretches of the owner' in capture and 'over the whole window, not inside one stretch' in capture
    assert '140 kyr–140 kyr' not in _slot(text, vt.SLOT_CAUSE) and '139.5–140.4 kyr' in _slot(text, vt.SLOT_CAUSE)


@pytest.mark.benchmark
def test_xp16_leg_does_not_read_the_circulation_margins():
    _, text = _mmr_text('coorbital__2024_XP16__1E-1+0+0')
    rows = {row.key: row for row in text.checks}
    assert all(rows[key].kind == vt.KIND_NA and rows[key].note == vt.AROUND_LEG for key in ('C1', 'C2', 'C3', 'C4'))


@pytest.mark.benchmark
def test_ac4_says_the_filter_is_too_slow():
    _, text = _mmr_text('coorbital__2026_AC4__1E-1+0+0')
    r0 = _emphasis(text, 'R0')
    assert r0.kind == vt.EMPHASIS_UNMEASURED and 'the filter is too slow for the period' in r0.note
    assert 'force window' in _emphasis(text, 'L4').note
