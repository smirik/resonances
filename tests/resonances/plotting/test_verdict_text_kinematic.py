"""W28 B10b (cards g08-8, g05-2, g05-3): the predicates the right column reads to say why a kinematic libration is
not the resonance (`slow_kinematic`, `cause_line`).

The rows carry 37087 3J+8S-3's slow scale as B10a measured it (benchmark, mmr_numbers.csv of 2026-11-01): slow
period 5689 yr, 13 slow cycles over 17.3-96.6 kyr at W_c 505 yr, half-swing 123.6°, the e cycle 2π/|g − g5| =
5705 yr (term g-g5, offset 0.0028), identity slope ratio 2.78 / corr 0.51 (fail). The synthetic analysis is the −4ϖ
kinematic libration of `test_kinematic_panel` (5:1 with Jupiter, ϖ = 0.4 + 0.25 sin(2π t / 400 yr))."""

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


def test_k3s_cause_line_names_the_slow_scales_cycle():
    f, _, _ = _facts(K3S_VERDICT, **SLOW_37087)
    # the header: the slow scale's cycle, not the 2ω cycle nearest σ's forced wave (37087 before B10b: '2ω cycle, 1.92 kyr')
    assert '(e cycle g − g5, 5.7 kyr)' in vt.cause_line(f.v, f)[1]
