"""W20 CS8: the baseline reads the slow time scale of an insufficient_time angle, and its text names it (F8, author
2026-10-19: the "about N yr needed" of RB7, 3753, CG236, OM35, FA13 and KQ18 was computed from the wrong period).

required = (LIBRATION_MIN_TURNING_POINTS / 2) P + 2 FILTER_EDGE_TRIM_CUTOFFS P_c, extend = the round value (one
significant digit, up) >= max(required, 2 T). With the defaults 3 and 1, by hand:

- 3753 / 2025 RB7 1 kyr: turning points at 265.6 and 649.3 yr, P = 2 x 383.7 = 767.4 yr, P_c 55.5 yr:
  required = 1.5 x 767.4 + 2 x 55.5 = 1151.1 + 111.0 = 1262.1 yr; extend max(1262.1, 2000) -> 2000 yr;
- a stall at 0.12 rad/kyr (2016 FA13): P = 2 pi / 0.12 = 52.36 kyr, required = 78.54 + 0.15 = 78.69 kyr -> 80 kyr;
- 2003 KQ18 6J+2S-1 (window 75-19925 yr, the slow centre turns once at 9037 yr): stretches 8962 and 10888 yr, the
  2 omega cycle 22.79 kyr is not below 2 x 10888 = 21.78 kyr, B5 18.14 kyr: P = 22.79 kyr, required = 34.185 + 0.15 =
  34.335 kyr -> 40 kyr.
"""

import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.plotting import verdict_text as vt
from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.decide import AngleDecision
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Reason, Status

S = settings_from_config()
NAN = float('nan')
FEW = AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, 'G6')
ARC = AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, 'G5a')
STALL = AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.STALL, 'G6')


PROMINENCE = math.radians(5)


def _tps(times, prominence=PROMINENCE):
    n = len(times)
    return g.TurningPoints(np.arange(n), np.asarray(times, float), np.zeros(n), np.ones(n), prominence)


def _run(times, reason=g.RejectReason.TOO_FEW_TP, episode_id=0):
    times = np.asarray(times, float)
    return SimpleNamespace(
        episode_id=episode_id,
        accepted=False,
        before_last_jump=False,
        reject_reason=reason,
        tp_times_yr=times,
        t_start_yr=float(times[0]),
        t_end_yr=float(times[-1]),
    )


def _geometry(window, tp_times=(), runs=(), centre_turns=(), centre_window=NAN, arc_stall=None, cycle=NAN):
    return SimpleNamespace(
        window=window,
        turning_points=_tps(tp_times),
        runs=list(runs),
        forced_wave_ids=frozenset(),
        centre=SimpleNamespace(turns=_tps(centre_turns), window_yr=centre_window),
        arc_stall=arc_stall,
        period_cycle_yr=cycle,
    )


def _stall(rate_rad_kyr, period_min_yr):
    return SimpleNamespace(rate_rad_yr=rate_rad_kyr / 1000.0, period_min_yr=period_min_yr)


def _kinematics(check, two_omega_yr, modulation_yr):
    return SimpleNamespace(check=check, two_omega_period_yr=two_omega_yr, modulation_period_yr=modulation_yr)


# --- the period P and what it needs ----------------------------------------------------------------------------------


def test_few_cycles_reads_the_turning_point_spacing():
    # 3753 1E-1, 1 kyr: two TPs, the centre (W_c 167 yr) follows them: its turns at 265.6 / 649.3 bound the same
    # half-period, so the spacing decides
    geometry = _geometry((55.5, 944.5), (265.6, 649.3), [_run((265.6, 649.3))], centre_turns=(265.6, 649.3), centre_window=166.6)
    base = decide.baseline(geometry, 55.5, 1000.0, 435.0, S, FEW, None)
    assert base.period_source == 'tp_spacing' and base.period_yr == pytest.approx(767.4)
    assert base.required_yr == pytest.approx(1262.1) and base.check == CheckValue.FAIL and base.extend_to_years == 2000.0
    assert base.cause == ''


def test_few_cycles_reads_the_slow_centre_when_the_window_can_show_its_turn():
    # 385458 9N-17 (100 kyr): runs whose centre moves, P_cyc 1546 yr; the centre turns at 28 913 and 67 042 yr inside
    # the window 523.5-99 476.5 yr (W_c 4635 yr): stretches 28 389.5, 38 129, 32 434.5 -> P = 76 258 yr; B5 47.68 kyr
    runs = [_run(np.arange(20359.0, 36881.0, 773.0), g.RejectReason.CENTRE_MOVES)]
    geometry = _geometry((523.5, 99476.5), np.arange(20359.0, 36881.0, 773.0), runs, (28913.0, 67042.0), 4635.0, cycle=1546.0)
    base = decide.baseline(geometry, 523.5, 100000.0, 47680.0, S, FEW, None)
    assert base.period_source == 'arc_slow' and base.period_yr == pytest.approx(76258.0)
    # 1.5 x 76 258 + 2 x 523.5 = 114 387 + 1047 = 115 434 yr -> 200 kyr (its 300-kyr record is resonant)
    assert base.required_yr == pytest.approx(115434.0) and base.extend_to_years == 200000.0
    # the same record with a centre that cannot turn inside the window (W <= 2 W_c, 1999 UQ: W_c 50.6 kyr) keeps the
    # spacing: 2 x 773 = 1546 yr
    blind = _geometry((523.5, 99476.5), np.arange(20359.0, 36881.0, 773.0), runs, (), 60000.0)
    assert decide.baseline_period(FEW, blind, None, NAN)[:2] == (pytest.approx(1546.0), 'tp_spacing')


def test_few_cycles_that_already_fit_the_record_name_what_is_missing():
    # two TPs 170 yr apart in a 20-kyr record whose centre cannot show a turn: 1.5 x 340 + 150 = 660 yr <= 20 kyr
    geometry = _geometry((75.0, 19925.0), (10087.5, 10257.5), [_run((10087.5, 10257.5))], (), 20000.0)
    base = decide.baseline(geometry, 75.0, 20000.0, NAN, S, FEW, None)
    assert base.required_yr == pytest.approx(660.0) and base.check == CheckValue.PASS and base.extend_to_years is None
    assert base.cause == 'too_few_tp'
    # an episode's angle has enough time: no cause (the check only)
    libration = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2')
    assert decide.baseline(geometry, 75.0, 20000.0, NAN, S, libration, None).cause == ''


def test_arc_reads_the_slow_swing():
    geometry = _geometry((75.0, 19925.0), np.arange(100.0, 19900.0, 450.0), centre_turns=(9037.0,), centre_window=2700.7)
    kq18 = _kinematics(CheckValue.FAIL, 22790.0, 146400.0)
    base = decide.baseline(geometry, 75.0, 20000.0, 18140.0, S, ARC, kq18)
    assert base.period_source == 'two_omega' and base.period_yr == pytest.approx(22790.0)
    assert base.required_yr == pytest.approx(34335.0) and base.extend_to_years == 40000.0
    # the kinematic test passes: the cycle of varpi is no time scale of this angle; twice the longer stretch
    passing = _kinematics(CheckValue.PASS, 22790.0, 146400.0)
    assert decide.baseline_period(ARC, geometry, passing, 18140.0)[:2] == (pytest.approx(21776.0), 'arc_slow')
    # a secular cycle below the bound is no candidate: the shortest one above it (the e-cycle) is
    short = _kinematics(CheckValue.FAIL, 15000.0, 146400.0)
    assert decide.baseline_period(ARC, geometry, short, 18140.0)[:2] == (pytest.approx(146400.0), 'modulation')


def test_stall_reads_one_revolution_at_its_rate():
    # 2016 FA13 8M-7J-5: 0.12 rad/kyr over 19.85 kyr = 2.38 rad, far above the prominence; P_lib,min 114 yr
    geometry = _geometry((75.0, 19925.0), arc_stall=_stall(0.12, 114.0), centre_window=225.0)
    base = decide.baseline(geometry, 75.0, 20000.0, 10760.0, S, STALL, None)
    assert base.period_source == 'stall_rate' and base.period_yr == pytest.approx(2 * math.pi / 0.12e-3)
    assert base.period_yr == pytest.approx(52359.88, rel=1e-6)
    assert base.required_yr == pytest.approx(78689.82, rel=1e-6) and base.extend_to_years == 80000.0
    # a travel below the turning points' prominence (here 1.0 deg in 363 yr against 5 deg) measures no rate: the
    # largest of twice the window without a turn (726 yr) and the force bound (4886 yr)
    still = _geometry((318.6, 681.4), arc_stall=_stall(math.radians(1.0) / 0.3628, 4886.0), centre_window=955.8)
    assert decide.baseline_period(STALL, still, None, NAN)[:2] == (pytest.approx(4886.0), 'stall_force')
    fast_force = _geometry((318.6, 681.4), arc_stall=_stall(math.radians(1.0) / 0.3628, 500.0), centre_window=955.8)
    assert decide.baseline_period(STALL, fast_force, None, NAN)[:2] == (pytest.approx(725.6), 'arc_slow')


def test_other_angles_keep_the_cycle():
    geometry = _geometry((75.0, 19925.0), cycle=20000.0, arc_stall=_stall(0.12, 114.0))
    libration = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2')
    assert decide.baseline_period(libration, geometry, None, 5000.0)[:2] == (20000.0, 'cycle')
    geometry.period_cycle_yr = NAN
    assert decide.baseline_period(libration, geometry, None, 5000.0)[:2] == (114.0, 'stall_force')
    assert decide.baseline_period(None, _geometry((0.0, 1.0)), None, 5000.0)[:2] == (5000.0, 'b5')


# --- the sentence ----------------------------------------------------------------------------------------------------


class _Facts(SimpleNamespace):
    def entry(self, name):
        return self.path.get(name)


def _sentence(row, path=None):
    return vt._baseline_sentence(_Facts(row=row, path=path or {}, s=S))


def test_sentence_names_the_period_and_its_source():
    row = {'base_required_kyr': 34.335, 'base_period_yr': 22790.0, 'base_period_source': 'two_omega', 'base_cause': ''}
    text = _sentence(row)
    assert text == 'Uncertain: the record is too short to decide: about 34.3 kyr needed (P = 22.8 kyr from the 2ω cycle the angle follows).'
    row = {'base_required_kyr': 1.2621, 'base_period_yr': 767.4, 'base_period_source': 'tp_spacing', 'base_cause': ''}
    assert 'about 1.26 kyr needed (P = 767 yr from twice the median spacing of the turning points)' in _sentence(row)


def test_sentence_names_the_next_check():
    """N8: the next check first; the budget clause only when it cuts the estimate (613440: 4 kyr against 200 kyr)."""
    row = {
        'base_required_kyr': 104.7,
        'base_period_yr': 69800.0,
        'base_period_source': 'stall_rate',
        'base_cause': '',
        'base_extend_to_years': 200000.0,
        'extend_to_years': 4000.0,
    }
    assert _sentence(row) == (
        'Uncertain: the record is too short to decide; next check: 4 kyr (at most 4 × the record; about 105 kyr '
        'estimated from P = 69.8 kyr from one revolution at the rate of the stall).'
    )
    row = {
        'base_required_kyr': 34.335,
        'base_period_yr': 22790.0,
        'base_period_source': 'two_omega',
        'base_cause': '',
        'base_extend_to_years': 40000.0,
        'extend_to_years': 40000.0,
    }
    assert _sentence(row) == (
        'Uncertain: the record is too short to decide; next check: 40 kyr (about 34.3 kyr estimated from P = 22.8 kyr '
        'from the 2ω cycle the angle follows).'
    )


def test_action_names_the_next_check_of_insufficient_time_only():
    short = SimpleNamespace(action='extend', angle_status='insufficient_time', extend_to_years=4000.0)
    assert vt.action_sentence(short) == 'Action: extend the integration; next check: 4 kyr.'
    drift = SimpleNamespace(action='extend', angle_status='drift', extend_to_years=40000.0)
    assert vt.action_sentence(drift) == 'Action: extend the integration to 40 kyr.'


def test_sentence_when_time_is_not_what_is_missing():
    row = {
        'base_required_kyr': 0.66,
        'base_period_yr': 340.0,
        'base_period_source': 'tp_spacing',
        'base_cause': 'too_few_tp',
        'smp_span_yr': 20000.0,
    }
    text = _sentence(row)
    assert text.startswith('Uncertain, but not for lack of time: the record (20 kyr) already holds the 660 yr')
    assert text.endswith('the run of turning points stops before a cycle closes.')
    qs = SimpleNamespace(value=8.0, threshold=55.5, result=True)
    text = _sentence(row, {vt.QS_STALL_ENTRY: qs})
    assert '8 yr < 55.5 yr' in text and 'longer record would not decide' in text


# --- the benchmark rows (independent: the slope of the saved angle or of omega) --------------------------------------

KQ18 = 'mmr3__2003_KQ18__6J+2S-1+0+0-7'
FA13 = 'mmr3__2016_FA13__8M-7J-5+0+0+4'
OM35 = 'mmr3__2015_OM35__1R-9E+6+0+0+2'


def _analysis(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = io.load_series(example)
    analysis = analyse_mmr(series, S)
    return series, analysis, {**analysis.numbers_row(), **analysis.summary_row()}


def _slope_rad_yr(series, values, window):
    t = np.asarray(series.t_yr, float)
    inside = (t >= window[0]) & (t <= window[1])
    return float(np.polyfit(t[inside], np.unwrap(np.asarray(values, float))[inside], 1)[0])


@pytest.mark.parametrize('example', [FA13, OM35])
def test_stalls_extend_by_one_revolution(example):
    series, analysis, row = _analysis(example)
    v = analysis.verdict
    assert (v.status, v.reason, str(v.angle_subtype), str(v.action)) == (Status.UNCERTAIN, Reason.BASELINE_NEEDED, 'stall', 'extend')
    window = (row['geo_window_t0_yr'], row['geo_window_t1_yr'])
    rate = abs(_slope_rad_yr(series, series.angle_wrapped, window))  # the raw angle's mean rate, outside the package
    assert row['base_period_source'] == 'stall_rate'
    assert row['base_period_yr'] == pytest.approx(2 * math.pi / rate, rel=0.1)
    required = 1.5 * 2 * math.pi / rate + 2 * row['flt_cutoff_yr']
    assert row['base_required_kyr'] * 1000 == pytest.approx(required, rel=0.1)
    estimate = decide.round_up(row['base_required_kyr'] * 1000)
    assert row['base_extend_to_years'] == estimate and estimate >= 80000.0
    # N8: the next check is capped at 4 T = 80 kyr (T = 20 kyr): FA13 asks for 80 kyr, OM35 for 200 kyr
    assert v.extend_to_years == 80000.0


def test_kq18_extends_by_its_two_omega_cycle():
    series, analysis, row = _analysis(KQ18)
    v = analysis.verdict
    assert (v.status, str(v.angle_subtype), str(v.action), v.extend_to_years) == (Status.UNCERTAIN, 'arc', 'extend', 40000.0)
    # pi / |omega_dot| over the record, outside the package (the author's report: 22.7-22.8 kyr)
    two_omega = math.pi / abs(_slope_rad_yr(series, series.omega, (series.t_yr[0], series.t_yr[-1])))
    assert row['base_period_source'] == 'two_omega' and row['base_period_yr'] == pytest.approx(two_omega, rel=0.05)
    text = vt.explain_mmr(analysis, S, row, analysis.episode_rows()).sentences[1]
    printed = re.search(r'next check: 40 kyr \(about ([0-9.]+) kyr estimated from P = ([0-9.]+) kyr from the 2ω cycle', text)
    assert printed is not None, text
    assert float(printed.group(1)) == pytest.approx(row['base_required_kyr'], abs=0.05)
    assert float(printed.group(2)) == pytest.approx(row['base_period_yr'] / 1000, abs=0.05)


# --- N8: the next check of an insufficient_time record (Codex, author 2026-10-24) ------------------------------------
# N_next = round_up(max(2 T, min(N, 4 T))); the cases are the rows of Codex's proposed_checkpoints.csv (round-1024/
# codex-n8), computed there from v2-w21b: T = 999.970507446 yr (1 kyr co-orbitals), 19 999.8876138 yr, 49 999.96 yr.

T1, T20, T50 = 999.970507446, 19999.8876138, 49999.9577668


@pytest.mark.parametrize(
    'estimate, span, expected',
    [
        (200000.0, T1, 4000.0),  # 613440: 4 T = 3999.88 -> 4 kyr
        (5000.0, T1, 4000.0),  # 2015 TL144, 2017 XG62
        (4000.0, T1, 4000.0),  # 2016 NZ55 unchanged
        (3000.0, T1, 3000.0),  # 2020 HO8 unchanged
        (2000.0, T1, 2000.0),  # few_cycles without a period: 2 T
        (1500.0, T1, 2000.0),  # never below 2 T
        (NAN, T1, 2000.0),  # no estimate: 2 T
        (300000.0, T20, 80000.0),  # 2014 QH365
        (80000.0, T20, 80000.0),  # 2016 FA13 unchanged
        (60000.0, T20, 60000.0),  # 2014 TF17 unchanged
        (500000.0, T50, 200000.0),  # 26420
    ],
)
def test_next_check_by_hand(estimate, span, expected):
    assert decide.next_check(estimate, span, S) == expected


def test_next_check_reads_its_settings():
    wide = settings_from_config(mmr_next_check_span_factor=10.0)
    assert decide.next_check(200000.0, T1, wide) == 10000.0  # 10 T = 9999.7 -> 10 kyr
    with pytest.raises(ValueError):
        settings_from_config(mmr_next_check_span_factor=1.0)  # below MMR_EXTEND_MIN_SPAN_FACTOR = 2


def test_613440_asks_for_the_next_check_and_keeps_its_estimate():
    """613440 1M-1, 1 kyr: a stall at the rate that needs ~200 kyr; the 20-kyr twin and its 4-kyr prefix decide
    (Codex N8: non_resonant/circulation/high at 4 kyr)."""
    series, analysis, row = _analysis('coorbital__613440__1M-1+0+0')
    v = analysis.verdict
    assert (v.status, str(v.angle_status), str(v.action)) == (Status.UNCERTAIN, 'insufficient_time', 'extend')
    assert row['base_extend_to_years'] == 200000.0 and v.extend_to_years == 4000.0
    assert vt.action_sentence(v) == 'Action: extend the integration; next check: 4 kyr.'
    assert 'next check: 4 kyr (at most 4 × the record;' in vt.explain_mmr(analysis, S, row, analysis.episode_rows()).sentences[1]


def test_extend_years_caps_insufficient_time_only():
    from resonances.resonance.analysis.mmr.models import Baseline

    far = Baseline(300000.0, 300000.0, CheckValue.FAIL, 200000.0, 'stall_rate')
    assert decide.extend_years(Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, far, T20, S) == 80000.0  # 4 T
    assert decide.extend_years(Status.UNCERTAIN, AngleStatus.DRIFT, far, T20, S) == 300000.0  # not insufficient_time
    assert decide.extend_years(Status.NON_RESONANT, AngleStatus.INSUFFICIENT_TIME, far, T20, S) is None
    enough = Baseline(1000.0, None, CheckValue.PASS, 600.0, 'cycle')
    assert decide.extend_years(Status.UNCERTAIN, AngleStatus.DRIFT, enough, T20, S) == 40000.0  # round_up(2 T)
