"""W28 B10b (cards g08-8, g05-2, g05-3): the right column says why a kinematic libration is not the resonance, with the
predicted and the measured swing, names the slow scale's cause at its level, and puts 'Driven by: secular cycle −4ϖ
(…, P …) [measured | candidate]' first under the plaque (the author on g05-3: show the term where an owner would be).

The rows carry 37087 3J+8S-3's slow scale as B10a measured it (benchmark, mmr_numbers.csv of 2026-11-01): slow
period 5689 yr, 13 slow cycles over 17.3-96.6 kyr at W_c 505 yr, half-swing 123.6°, the e cycle 2π/|g − g5| =
5705 yr (term g-g5, offset 0.0028), identity slope ratio 2.78 / corr 0.51 (fail); predicted swing 96.67° (8 asin
(0.0286/0.1365)). The sentences below are written by hand from those numbers. The synthetic analysis is the −4ϖ
kinematic libration of `test_kinematic_panel` (5:1 with Jupiter, ϖ = 0.4 + 0.25 sin(2π t / 400 yr)): its
half-amplitude is 4 × 0.25 rad = 57.3°."""

import copy
import dataclasses
import math

import pytest

from resonances.plotting import verdict_text as vt
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.plotting.test_kinematic_panel import _record

S = settings_from_config()
TOL = S.mmr_kin_period_tol

SLOW_37087 = dict(
    slow_source='centre_scale',
    slow_cutoff_yr=505.0,
    slow_t0_yr=17289.0,
    slow_t1_yr=96577.0,
    slow_n_cycles=13,
    slow_period_yr=5689.0,
    slow_halfamp_deg=123.6,
    slow_identity='fail',
    slow_identity_slope_ratio=2.78,
    slow_identity_corr=0.512,
    slow_secular_offset=0.0028,
    slow_secular_period_yr=5704.97,
    slow_secular_term='g-g5',
    slow_origin='kinematic',
    slow_cause='secular_g_gj',
    slow_cause_term='g-g5',
    slow_cause_level='identified',
    slow_cause_source='centre',
    kin_pred_swing_deg=96.67,
    kin_meas_swing_deg=123.6,
    kin_cycle_secular_offset=0.912,
)
K3S_VERDICT = dict(origin_detail='secular_cycle', cause_role='kinematic', cause_term='-4varpi')


def _facts(verdict=None, **columns):
    analysis, _ = _record()
    analysis = copy.copy(analysis)
    if verdict:
        analysis.verdict = dataclasses.replace(analysis.verdict, **verdict)
    row = {**analysis.numbers_row(), **analysis.summary_row(), **columns}
    return vt.MMRFacts(analysis, S, row, analysis.episode_rows()), analysis, row


# --------------------------------------------------------------------------------------------- the columns


def test_secular_term_period_reads_the_column_that_measured_the_term():
    row = {
        'slow_secular_term': 'g-g6',
        'slow_secular_period_yr': 46000.0,
        'own_x_secular_term': '2omega',
        'own_x_secular_period_yr': 7110.0,
        'kin_two_omega_period_yr': 7156.0,
        'kin_modulation_period_kyr': 5.7086,
    }
    assert vt.secular_term_period(row, 'g-g6') == 46000.0
    assert vt.secular_term_period(row, '2omega') == 7110.0  # A2's slow x measured it on this row
    assert vt.secular_term_period({**row, 'own_x_secular_term': ''}, '2omega') == 7156.0
    assert vt.secular_term_period(row, 'g-g5') == pytest.approx(5708.6)
    assert math.isnan(vt.secular_term_period(row, 'g-g7')) and math.isnan(vt.secular_term_period(row, ''))


@pytest.mark.parametrize(
    'origin, offset, expected',
    [('kinematic', 0.912, True), ('kinematic', float('nan'), True), ('kinematic', 0.004, False), ('', 0.912, False)],
)
def test_the_kinematic_swing_is_the_slow_scales_only_when_sigmas_own_period_matches_none(origin, offset, expected):
    # 37087: σ's 168-yr forced wave matches no secular period (0.912), its slow scale does; 285637: σ's own 18 kyr does
    assert vt.slow_kinematic({'slow_origin': origin, 'kin_cycle_secular_offset': offset}, TOL) is expected


# --------------------------------------------------------------------------------------------- K3s: the slow scale


def test_k3s_cause_names_term_period_swings_and_the_failed_identity():
    f, _, _ = _facts(K3S_VERDICT, **SLOW_37087)
    assert vt.kinematic_cause(f) == (
        'The slow swing is σ’s −4ϖ term: on the slow scale (17.3 kyr–96.6 kyr, cutoff 505 yr) its period 5.69 kyr '
        'matches the e cycle 2π/|g − g5| = 5.7 kyr; the free and forced e predict ±96.7° for −4ϖ, the slow swing is '
        '±124°; σ̇ = c(a − a*) fails there (slope ratio 2.78, corr 0.51).'
    )
    # the header: the slow scale's cycle, not the 2ω cycle nearest σ's forced wave (37087 before B10b: '2ω cycle, 1.92 kyr')
    assert '(e cycle g − g5, 5.7 kyr)' in vt.cause_line(f.v, f)[1]


@pytest.mark.parametrize('level, word', [('identified', 'measured'), ('possible', 'candidate'), ('unknown', 'candidate')])
def test_k3s_driver_line_is_measured_only_at_the_identified_level(level, word):
    f, analysis, row = _facts(K3S_VERDICT, **{**SLOW_37087, 'slow_cause_level': level})
    expected = f'Driven by: secular cycle −4ϖ (the e cycle 2π/|g − g5|, P 5.7 kyr) [{word}].'
    assert vt.secular_driver_line(f) == (expected, ('K3s',))
    text = vt.explain_mmr(analysis, S, row, analysis.episode_rows())
    # W34: the minimal policy opens the explanation with its four SLOT_POLICY bullets; the driver line is the first after
    # them, where a confirmed owner would be
    body = [bullet for bullet in text.bullets if bullet.slot != vt.SLOT_POLICY]
    assert [bullet.slot for bullet in text.bullets[:4]] == [vt.SLOT_POLICY] * 4
    assert body[0].text == expected
    assert vt.slow_scale_line(f) == ''  # the cause line says it


# --------------------------------------------------------------------------------------------- the main scale


def test_main_scale_cause_prints_the_predicted_swing_and_the_episodes_half_amplitude():
    f, _, row = _facts()
    assert str(f.v.reason) == 'kinematic_libration'
    half = f.episodes.decisive['half_amp_med_deg']
    assert half == pytest.approx(math.degrees(1.0), rel=0.03)  # 4 × 0.25 rad
    words = vt.swing_words(f, '−4ϖ')
    assert (
        words
        == f'the free and forced e predict ±{vt.fmt(row["kin_pred_swing_deg"])}° for −4ϖ, the episode’s half-amplitude is ±{vt.fmt(half)}°'
    )
    assert words in vt.kinematic_cause(f)


def test_no_driver_line_without_a_secular_cycle():
    f, _, _ = _facts()  # origin_detail varpi_term: no secular period matched
    assert vt.secular_driver_line(f) == ('', ())


# --------------------------------------------------------------------------------------------- the slow-scale line


HEAD = 'Slow scale: σ’s center swings by ±124° with period 5.69 kyr (13 slow cycles, 17.3 kyr–96.6 kyr)'
NOT_EXCLUDED = 'a slow libration of another resonance is not excluded'


@pytest.mark.parametrize(
    'columns, expected',
    [
        ({}, f'{HEAD}: the e cycle 2π/|g − g5| = 5.7 kyr (identified).'),
        ({'slow_cause_level': 'possible'}, f'{HEAD}: probably the e cycle 2π/|g − g5| = 5.7 kyr; {NOT_EXCLUDED}.'),
        ({'slow_cause': '', 'slow_cause_term': '', 'slow_cause_level': 'unknown'}, f'{HEAD}: cause not identified; {NOT_EXCLUDED}.'),
        (
            {'slow_cause': 'pvarpi_term', 'slow_cause_term': '-8varpi', 'slow_cause_level': 'possible'},
            f'{HEAD}: probably σ’s −8ϖ term (the center follows it at no secular period); {NOT_EXCLUDED}.',
        ),
        (
            {'slow_origin': 'dynamic_candidate', 'slow_identity': 'pass'},
            f'{HEAD}: the e cycle 2π/|g − g5| = 5.7 kyr (identified); σ̇ = c(a − a*) holds on that scale, so a slow '
            'libration is not excluded.',
        ),
        (
            {'slow_period_yr': float('nan'), 'slow_n_cycles': 1, 'slow_halfamp_deg': 10.9, 'slow_t0_yr': 1105.0, 'slow_t1_yr': 11988.0},
            'Slow scale: σ’s center swings by ±10.9° over 1.1 kyr–12 kyr (1 slow cycle: no period): the e cycle '
            '2π/|g − g5| = 5.7 kyr (identified).',
        ),
        ({'slow_halfamp_deg': float('nan')}, ''),  # no slow swing: nothing said
        ({'slow_source': ''}, ''),
    ],
)
def test_slow_scale_line_at_its_cause_level(columns, expected):
    f, analysis, row = _facts({'reason': 'dynamic_libration'}, **{**SLOW_37087, 'slow_origin': 'unresolved', **columns})
    assert vt.slow_scale_line(f) == expected
    if expected:
        bullets = [bullet.text for bullet in vt.explain_mmr(analysis, S, row, analysis.episode_rows()).bullets]
        assert expected in bullets
