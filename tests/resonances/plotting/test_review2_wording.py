"""Wording of the right column after the second independent review (2026-11-05): numbers printed with what they mean,
no derivation that is not one, no rule id in a sentence. No status changes.

The inputs are the numbers of the benchmark figures the review named (24820 3J-1S-1, 2012 KL9, 2001 HW7, 2002 AA29,
3040 ZLK, 1921 2J-1); every expected string is worked out by hand from them."""

from types import SimpleNamespace

from resonances.plotting import verdict_text as vt
from resonances.resonance.analysis.ledger import LedgerEntry, Margin, Op
from resonances.resonance.analysis.mmr.decide import STRENGTH_THREE_BODY
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import MarginClass, MarginValue
from resonances.resonance.analysis.zlk.diagnostics import EICoupling

S = settings_from_config()
THIN_SPACE = ' '  # fmt groups thousands with a thin space


def test_the_settings_the_expected_strings_assume():
    assert (S.libration_coverage_periods, S.mmr_l7_gap_full, S.libration_min_turning_points) == (1.0, 0.7, 3)


def test_coverage_within_the_bar_says_so_and_names_the_other_cause():
    """24820: gaps 0.33 and 0.72 P, both ≤ 1: the gaps alone would cover the record, so the text says they are within
    the bar (the exit seen after it is what makes the angle episodic); 2012 KL9: 11 and 2.7 P, the bar in brackets."""
    covered = {'coverage_start_gap_P': 0.33, 'coverage_end_gap_P': 0.72}
    assert vt.coverage_within(covered, S)
    assert vt.coverage_text(covered, S) == (
        'it misses the start by 0.33 and the end by 0.72 libration periods, both within the 1 period a libration that '
        'covers the record may miss at each end'
    )
    missed = {'coverage_start_gap_P': 11.0, 'coverage_end_gap_P': 2.7}
    assert not vt.coverage_within(missed, S)
    assert vt.coverage_text(missed, S) == (
        'it misses the start by 11 and the end by 2.7 libration periods (a libration that covers the record may miss '
        'each end by up to 1 period)'
    )


def test_l7_between_the_two_bars_is_covered_without_the_full_margin():
    """2001 HW7: larger gap 0.705 P > 0.7 (full margin) but ≤ 1 (covered): both bars printed; 1921's 0.5 only the one."""
    f = SimpleNamespace(s=S)
    thin = MarginValue(Margin.THIN, 0.705, MarginClass.DECISIVE)
    assert vt._detail_l7(f, thin)[0] == ('larger edge gap 0.705 > 0.7 P for a full margin; the record counts as covered up to 1 P')
    full = MarginValue(Margin.FULL, 0.5, MarginClass.DECISIVE)
    assert vt._detail_l7(f, full)[0] == 'larger edge gap 0.5 ≤ 0.7 P'


def test_a_strength_ratio_is_a_multiple_of_its_scale_with_the_bar_apart():
    """2012 KL9: required / μ_i μ_j e^q = 1.66e15, implausible from ×1000; never '1.66e15 ≥ 1 000 × μ_i μ_j e^q'."""
    entry = LedgerEntry(f'5s {STRENGTH_THREE_BODY}', 1.66e15, Op.GE, 1000.0, '', 'str_ratio', 'mmr_strength_implausible_ratio', True, True)
    assert vt.times_scale(entry, 'the scale μ_i μ_j e^q') == (f'1.66e15 × the scale μ_i μ_j e^q (×1{THIN_SPACE}000 or more is implausible)')
    assert vt.times_scale(None, 'the scale') == ''


def test_sci_prints_small_numbers_with_an_exponent():
    """1921: μ needed 2.558e-4 against μ theory 7.6e-5–1.05e-4: one notation in the range and beside it."""
    assert [vt.sci(v) for v in (2.558e-4, 7.6e-5, 1.05e-4, 9.996e-4, 0.0197, 12345.0)] == [
        '2.56e-4',
        '7.6e-5',
        '1.05e-4',
        '1e-3',
        '0.0197',
        '1.23e4',
    ]


def test_long_period_n_at_the_minimum_and_s_on_one_cycle():
    """1866 ZLK: 3 turning points (the minimum, thin): no 'need > 3' beside a libration; one center estimate: one
    sentence, no doubled bracket."""
    n_thin = MarginValue(Margin.THIN, 3, MarginClass.DECISIVE)
    detail, need, note = vt._long_n(n_thin, S, 'over the whole record')
    assert (detail, need) == ('3 turning points over the whole record', '')
    assert note == '3, the fewest that make a libration; a full margin needs more than 3'
    assert vt._long_n(MarginValue(Margin.FULL, 24, MarginClass.DECISIVE), S, 'over the whole record')[1] == '> 3'
    s_one = MarginValue(Margin.NOT_MEASURED, 1, MarginClass.DECISIVE)
    row = vt._long_period_margin_row('S', s_one, S, True)
    assert row.detail == '' and row.need == ''
    assert row.note == 'not measured: one cycle gives one center estimate, two are needed to compare'


def test_ei_line_names_both_slopes_once():
    """3040 ZLK: at 39.5 kyr, i against e 179°, r −1.00, 1.14 × the constant-H slope, 3.28 × the forced-beat slope;
    'whole record' once, a true minus sign."""
    ei = EICoupling(coupling='zlk', period_yr=39_500.0, phase_deg=179.0, band_r=-1.0, ratio_h=1.14, ratio_beat=3.28)
    assert vt._ei_line(ei) == (
        'e-i at 39.5 kyr, whole record: ZLK (antiphase, constant H): i against e 179°, r −1.00, δi/δe 1.14 × the '
        'constant-H slope (H = √(1−e²) cos i), 3.28 × the forced-beat slope (says how e and i trade, not whether ω '
        'librates)'
    )


def test_owner_intervals_said_once_per_regime():
    """1997 GK3's owner librates in three blocks (455 yr–11.6 kyr, 11.6–12.4 kyr, 12.8–18.6 kyr)."""
    text = '455-11600:libration;11600-12400:libration;12800-18600:libration'
    assert vt._intervals(text) == 'librating during 0.455–11.6 kyr, 11.6–12.4 kyr, 12.8–18.6 kyr'
