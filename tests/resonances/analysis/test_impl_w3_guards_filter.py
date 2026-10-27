"""W3 guards, F1: the x2 guard compares decisive NUMBERS with tolerances, a vanished measurement is a change, and a
guard with nothing to compare says 'not checked' (sensitive None), never 'unchanged'.

Counterexamples (Codex astra PHYSICS-MATH §3, sol REPORT P1): 30 TPs -> 3 TPs and h 0.2 -> 0.9 kept every yes/no
conclusion and passed as unchanged; 1221 1V-2E+1 was sensitive yet kept L5/L6 full; 31935 at 2 P_c had nothing to
compare and still reported filter_sensitive False; the synthetic pendulum of P = 1066 yr at P_c = 400 / 800 yr.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.filtering import edge_window, lowpass, time_derivative
from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.mmr import decide, io, preprocess
from resonances.resonance.analysis.mmr.models import Cutoff, MMRSeries, Preprocessed, Regime, ResonanceSpec
from resonances.resonance.analysis.mmr.preprocess import DecisiveQuantities, compare_quantities
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, Confidence, Flag, MarginClass, MarginValue, Status
from tests.resonances.analysis.impl_response_pendulum_synthetic import make

S = settings_from_config()
DEG = np.pi / 180


def quantities(n_tp=30, n_episodes=1, h_med=0.2, cutoff=20.0, period=1000.0, bounds=((1000.0, 29000.0),), amp=40.0, shift=0.1):
    return DecisiveQuantities(cutoff, n_tp, n_episodes, False, 0.0, h_med, (0.0, 30000.0), period, tuple(bounds), np.radians(amp), shift)


# --- the comparison by numbers ----------------------------------------------------------------------------------------


def test_numbers_that_keep_every_conclusion_still_change():
    assert compare_quantities(quantities(), quantities(n_tp=3, cutoff=40.0), S).changed == ('turning_points',)  # 30 -> 3
    assert compare_quantities(quantities(), quantities(h_med=0.9, cutoff=40.0), S).changed == ('energy',)  # 0.2 -> 0.9


@pytest.mark.parametrize(
    'second, changed',
    [
        (quantities(n_tp=27), ()),  # 3 = max(MMR_FILTER_GUARD_TP_ABS 2, MMR_FILTER_GUARD_TP_REL 0.1 x 30)
        (quantities(n_tp=26), ('turning_points',)),  # 4 > 3
        (quantities(h_med=0.3), ()),  # dh = 0.1 = MMR_FILTER_GUARD_H_TOL
        (quantities(h_med=0.31), ('energy',)),
        (quantities(bounds=((1500.0, 29000.0),)), ()),  # half a period
        (quantities(bounds=((3000.0, 29000.0),)), ('episodes',)),  # two periods (MMR_FILTER_GUARD_BOUNDARY_PERIODS 1)
        (quantities(n_episodes=0, bounds=()), ('episodes',)),  # the episode vanished
        (quantities(amp=33.0), ()),  # -17.5 % (MMR_FILTER_GUARD_AMP_REL 0.2)
        (quantities(amp=31.0), ('shape',)),  # -22.5 %
        (quantities(shift=0.6), ('shape',)),  # L6 full (<= MMR_L6_SHIFT_FULL) -> not full
    ],
)
def test_tolerances_of_the_settings(second, changed):
    second = DecisiveQuantities(**{**second.__dict__, 'cutoff_yr': 40.0})
    assert compare_quantities(quantities(), second, S).changed == changed


def test_a_measurement_at_one_cutoff_only_is_a_change():
    vanished = compare_quantities(quantities(), quantities(h_med=float('nan'), cutoff=40.0), S)
    assert vanished.changed == ('energy',) and vanished.sensitive is True
    appeared = compare_quantities(quantities(h_med=float('nan')), quantities(cutoff=40.0), S)
    assert appeared.changed == ('energy',)


def test_shape_and_energy_need_a_doubled_cutoff_that_lets_the_libration_through():
    # P_cyc = 1000 yr < FILTER_MIN_LIBRATION_CUTOFFS (2) x 2 P_c = 1200 yr: 2 P_c itself damps the libration
    first, second = quantities(cutoff=300.0, h_med=0.2, amp=40.0), quantities(cutoff=600.0, h_med=0.9, amp=20.0)
    result = compare_quantities(first, second, S)
    assert result.changed == () and result.not_checked == ('energy', 'shape') and result.sensitive is False


def test_nothing_to_compare_is_not_checked():
    result = compare_quantities(quantities(), None, S)
    assert result.sensitive is None and result.not_checked == preprocess.GUARD_QUANTITIES
    assert result.to_columns()['flt_filter_sensitive'] is None
    checks = {'check_mmr_filter': CheckValue.PASS, 'check_mmr_one_pendulum': CheckValue.PASS}
    margins = {'L8': MarginValue(Margin.FULL, 0.98, MarginClass.SUPPORT), 'L5': MarginValue(Margin.FULL, 1.0)}
    decide.apply_filter_sensitivity(checks, margins, result)
    assert checks == {'check_mmr_filter': CheckValue.NOT_MEASURED, 'check_mmr_one_pendulum': CheckValue.PASS}
    assert margins['L8'].margin == Margin.THIN and margins['L8'].value == 0.98 and margins['L5'].margin == Margin.FULL


def test_the_guard_never_raises_the_confidence():
    """A changed quantity empties the margins that read it, but a failed one stays failed (1221: O1 fail)."""
    margins = {'O1': MarginValue(Margin.FAIL, 2.0, MarginClass.SUPPORT), 'L5': MarginValue(Margin.FULL, 1.1, MarginClass.SUPPORT)}
    checks = {'check_mmr_onset': CheckValue.FAIL}
    decide.apply_filter_sensitivity(checks, margins, compare_quantities(quantities(), quantities(n_tp=3), S))
    assert margins['O1'].margin == Margin.FAIL and margins['L5'].margin == Margin.NOT_MEASURED
    assert checks['check_mmr_onset'] == CheckValue.NOT_MEASURED


def test_filter_too_slow_empties_the_shape_margins():
    margins = {m: MarginValue(Margin.FULL, 1.0) for m in ('L2', 'L5', 'L6', 'L3')}
    checks = {'check_mmr_one_pendulum': CheckValue.PASS, 'check_mmr_identity': CheckValue.PASS}
    decide.apply_filter_too_slow(checks, margins, True)
    assert [m for m, v in margins.items() if v.margin == Margin.NOT_MEASURED] == ['L2', 'L5', 'L6']
    assert checks == {'check_mmr_one_pendulum': CheckValue.NOT_MEASURED, 'check_mmr_identity': CheckValue.PASS}


# --- the measurement on the interval both cutoffs see ------------------------------------------------------------------

CUTOFF = 20.0


def prepared(t, angle):
    angle_f = lowpass(t, angle, CUTOFF, 2)
    return make(
        Preprocessed,
        cutoff=Cutoff(CUTOFF, 'floor', 0.0, CUTOFF, CUTOFF / float(t[1] - t[0])),
        t_yr=t,
        angle_unwrapped=angle,
        angle_clean=angle,
        angle_f=angle_f,
        rate_f=time_derivative(t, angle_f),
        angle_f_half=lowpass(t, angle, CUTOFF / 2, 2),
        angle_f_double=lowpass(t, angle, 2 * CUTOFF, 2),
        a_f=np.zeros_like(t),
        adot_f=np.zeros_like(t),
        e_f=np.zeros_like(t),
        inc_f=np.zeros_like(t),
        window=edge_window(t, CUTOFF),
    )


def plain_angle(prep):
    """The x1 angle of `measured` (no MMR rules)."""
    return g.FilteredAngle(
        prep.t_yr,
        prep.angle_unwrapped,
        prep.angle_f,
        prep.rate_f,
        prep.angle_f_half,
        prep.angle_f_double,
        prep.a_f,
        -30.0,
        CUTOFF,
        0.5,
        prep.window,
    )


def measured(prep):
    return g.measure_geometry(plain_angle(prep), 1 * DEG, S)


def test_counts_on_the_common_interval():
    """A 100-yr libration (5 P_c): the x2 window is trimmed 20 yr more at each end; counted on the common interval the
    TPs agree exactly (extrema at 25 + 50 k yr: 75, 125, ..., 5925 in [40, 5959.5], 118 at both cutoffs)."""
    t = np.arange(0.0, 6000.0, 0.5)
    prep = prepared(t, np.pi + np.radians(60) * np.sin(2 * np.pi * t / 100.0))
    result = preprocess.filter_sensitivity(prep, plain_angle(prep), measured(prep), None, 1 * DEG, S)
    assert result.at_cutoff.interval == (40.0, 5959.5)
    assert result.at_cutoff.n_tp == result.at_double.n_tp == (5925 - 75) // 50 + 1
    assert result.sensitive is False and 'turning_points' not in result.changed


def test_no_common_interval_is_not_checked():
    t = np.arange(0.0, 120.0, 0.5)  # the x2 window [40, 79.5] is still there, the regime starts after it
    prep = prepared(t, np.pi + np.radians(60) * np.sin(2 * np.pi * t / 100.0))
    geometry = measured(prep)
    geometry.regime_start_yr = 90.0
    result = preprocess.filter_sensitivity(prep, plain_angle(prep), geometry, None, 1 * DEG, S)
    assert result.at_double is None and result.sensitive is None


# --- whole analyses -----------------------------------------------------------------------------------------------------


def pendulum_series(period=1000.0, amplitude=1.0, n=30001):
    """Codex astra's synthetic 2J-1 pendulum: phi_ddot = -w^2 sin(phi), 1 rad, P_0 = 1000 yr (P = 1066 yr at 1 rad)."""
    t = np.arange(n, dtype=float)
    n_jupiter = np.radians(30.349652)
    a0 = (np.pi / n_jupiter) ** (2 / 3)
    c = 3 * np.pi * a0**-2.5
    w = 2 * np.pi / period
    solution = solve_ivp(
        lambda _, y: [y[1], -w * w * np.sin(y[0])], (0, t[-1]), [amplitude, 0.0], t_eval=t, method='DOP853', rtol=1e-10, atol=1e-12
    )
    angle, rate = solution.y
    zero = np.zeros(n)
    return MMRSeries(
        t,
        a0 + rate / c,
        np.full(n, 0.1),
        zero,
        zero,
        zero,
        zero,
        angle % (2 * np.pi),
        ResonanceSpec.from_key('2J-1'),
        Regime(1.0, t[-1], n),
    )


def test_synthetic_pendulum_at_400_and_800_yr():
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = pendulum_series()
    at_400 = analyse_mmr(series, settings_from_config(mmr_filter_cutoff_yr=400.0), scan_owner=False, scan_subresonances=False)
    # P = 1066 yr >= 2 x 400: measured; 2 P_c = 800 yr is itself too slow for the shape: those are not checked
    assert at_400.verdict.status == Status.RESONANT and not at_400.prep.filter_too_slow
    assert at_400.extra_columns['flt_filter_sensitive'] is False
    assert at_400.extra_columns['flt_unchecked_quantities'] == 'energy;shape'
    at_800 = analyse_mmr(series, settings_from_config(mmr_filter_cutoff_yr=800.0), scan_owner=False, scan_subresonances=False)
    # P < 2 x 800 (filter_too_slow): the shape of the oscillation is not measured, whatever the geometry says
    assert at_800.prep.filter_too_slow and str(Flag.FILTER_TOO_SLOW) in at_800.verdict.flags
    for margin_id in ('L2', 'L5', 'L6'):
        if margin_id in at_800.verdict.margins:
            assert at_800.verdict.margins[margin_id].margin in (Margin.NOT_MEASURED, Margin.FAIL)
    assert at_800.verdict.checks['check_mmr_one_pendulum'] in (CheckValue.NOT_MEASURED, CheckValue.NOT_APPLICABLE)


def _real(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), S, scan_owner=False, scan_subresonances=False)


def test_1221_is_sensitive_and_loses_its_shape_margins():
    # with the owner search: its owner 3E-8 makes the row non_resonant/induced, a verdict with margins (without it, L4
    # and R0 are not measured and O1 fails at P_c: uncertain by rule 5o, W18, and an uncertain verdict has no margins)
    if not io.example_path('mmr3__1221__1V-2E+1+0+0+0').exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(io.load_series('mmr3__1221__1V-2E+1+0+0+0'), S, scan_subresonances=False)
    verdict, numbers = analysis.verdict, analysis.numbers_row()
    # P_c = 150 yr / 2 = 75 yr at the 5-yr step. Before W18: 70 TPs against 64 at 150 yr and an episode at 315-505 yr
    # only at P_c (sensitive: episodes). Without the libration's own image in the threshold (1.99 deg instead of 5.31)
    # all 92 TPs form one run that circulates (centre_moves; the author: circulation), 72 at 2 P_c: 20 > max(2, 0.1 x 92)
    assert numbers['flt_cutoff_yr'] == 75.0 and (numbers['flt_x1_n_tp'], numbers['flt_x2_n_tp']) == (92, 72)
    assert numbers['flt_filter_sensitive'] is True and 'turning_points' in numbers['flt_sensitive_quantities']
    assert str(Flag.FILTER_SENSITIVE) in verdict.flags and verdict.checks['check_mmr_filter'] == CheckValue.NOT_MEASURED
    for margin_id in ('L5', 'L6'):  # were full (sol); no episode now: not there at all
        assert margin_id not in verdict.margins or verdict.margins[margin_id].margin == Margin.NOT_MEASURED
    # the guard raises nothing; W22 R-C4 (author 2026-10-24): the failed C4 under a force law rejected everywhere (best
    # R^2 0.054) no longer counts, so the level is the author's high
    assert verdict.margins['C4'].margin == Margin.NOT_MEASURED and verdict.confidence == Confidence.HIGH


def test_31935_compares_the_same_turning_points():
    analysis = _real('mmr2__31935__9M-16+0+7')
    numbers = analysis.numbers_row()
    assert numbers['flt_filter_sensitive'] is False
    # W21 Q9: the TPs of the decisive episode (31410-82320 yr) and of its counterpart at 2 P_c; the record has 60
    assert numbers['flt_x1_n_tp'] == numbers['flt_x2_n_tp'] == 51
    assert numbers['flt_decisive_overlap'] > 0.99
    assert numbers['flt_guard_t0_yr'] < numbers['flt_guard_t1_yr']
