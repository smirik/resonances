"""The 1:1 coverage audit (`mmr.coverage_audit`): bounded motion the verdict does not count, as flags only.

The synthetic controls are the survey sweep's (followup/qs-sweep/missed_runs_audit.py self-test): positives with the
regime read here from the envelope of the turning points (an independent rule: which of the meridians 0 and 180 deg
the swing crosses), negatives that must give no run (Codex's drifting wave 4t + 20 sin(2 pi t / 20) among them). The
turn times of a sine are P/4 + k P/2 by hand.
"""

import math

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import coverage_audit as ca
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import Flag, needs_review
from tests.resonances.analysis.builders import make_verdict

S = settings_from_config()
STEP_YR = 0.05
T = np.arange(0.0, 1000.0 + STEP_YR / 2, STEP_YR)
WINDOW = (0.0, 1000.0)


def _noisy(deg, seed=1):
    """deg + a 15-deg term of one orbit (1 yr) + 1-deg white noise, in radians, unwrapped as the package keeps it."""
    rng = np.random.default_rng(seed)
    return np.radians(deg + 15.0 * np.sin(2 * np.pi * T) + rng.normal(0.0, 1.0, len(T)))


def _audit(raw, spans=(), window=WINDOW, settings=S):
    return ca.audit(T, raw, 1.0, window, list(spans), settings)


def _regime(stretch) -> str:
    """The meridians the swing [centre - A, centre + A] crosses: 0 -> QS, 180 -> HS, both -> compound, none -> TP."""
    lo, hi = math.degrees(stretch.centre_rad - stretch.half_amp_rad), math.degrees(stretch.centre_rad + stretch.half_amp_rad)
    host = math.floor(hi / 360.0) * 360.0 > lo
    l3 = math.floor((hi - 180.0) / 360.0) * 360.0 + 180.0 > lo
    if host and l3:
        return 'compound'
    if host:
        return 'QS'
    if l3:
        return 'HS'
    return 'TP60' if math.degrees(stretch.centre_rad) % 360.0 < 180.0 else 'TP300'


@pytest.mark.parametrize(
    'deg, regime',
    [
        (25 * np.sin(2 * np.pi * T / 48.0), 'QS'),
        (300 + 30 * np.sin(2 * np.pi * T / 150.0), 'TP300'),
        (60 + 30 * np.sin(2 * np.pi * T / 150.0), 'TP60'),
        (180 + 150 * np.sin(2 * np.pi * T / 200.0), 'HS'),
        (90 + 120 * np.sin(2 * np.pi * T / 120.0), 'compound'),
        ((15 + 0.03 * T) * np.sin(2 * np.pi * T / 40.0), 'QS'),
    ],
    ids=['qs_48yr', 'tp300', 'tp60', 'horseshoe', 'compound', 'qs_growing'],
)
def test_a_bounded_run_is_found_whatever_its_well(deg, regime):
    """Every positive control: one bounded run over the record, of the regime its envelope says; uncounted, inside the
    window, so one missed stretch and the flag."""
    found = _audit(_noisy(deg))
    assert len(found.runs) == 1 and _regime(found.runs[0]) == regime
    assert found.runs[0].t1_yr - found.runs[0].t0_yr > 800.0
    assert len(found.missed) == 1 and found.flag == ca.FLAG_UNCOVERED


@pytest.mark.parametrize(
    'raw',
    [
        np.radians(4 * T + 20 * np.sin(2 * np.pi * T / 20.0)),
        np.radians(4 * T + 60 * np.sin(2 * np.pi * T / 20.0)),
        _noisy(1.0 * T + 30 * np.sin(2 * np.pi * T / 60.0)),
        _noisy(-2.0 * T),
        np.radians(np.random.default_rng(3).normal(0.0, 3.0, len(T))),
    ],
    ids=['codex_drifting_wave', 'drifting_wave_x3', 'slow_drift_wave', 'circulation', 'noise_only'],
)
def test_no_bounded_run_in_a_drift_a_circulation_or_noise(raw):
    found = _audit(raw)
    assert found.runs == [] and found.missed == [] and found.flag == ''


def test_the_centre_test_alone_rejects_the_drifting_wave():
    """Without a swing floor and noise factor the centre test still rejects 4t + 20 sin(2 pi t / 20) (its centre moves
    80 deg per cycle, the swing is 20 deg); without the centre test the envelope alone takes it (Codex's mechanism)."""
    raw = np.radians(4 * T + 20 * np.sin(2 * np.pi * T / 20.0))
    bare = settings_from_config(mmr_coorbital_audit_min_swing_deg=1e-6, mmr_coorbital_audit_noise_factor=1e-9)
    assert _audit(raw, settings=bare).runs == []
    loose = settings_from_config(
        mmr_coorbital_audit_min_swing_deg=1e-6, mmr_coorbital_audit_noise_factor=1e-9, mmr_coorbital_audit_centre_tol=1e9
    )
    assert len(_audit(raw, settings=loose).runs) > 0


def test_turning_points_of_a_sine_by_hand():
    """sin(2 pi t / 48) over 1000 yr: turns at 12 + 24 k yr (P/4 + k P/2) up to 972 yr; the one at 996 yr retraces only
    3.3 deg before the end and is not confirmed."""
    x = np.radians(25.0) * np.sin(2 * np.pi * T / 48.0)
    turns = ca.hysteresis_turns(x, math.radians(10.0))
    expected = 12.0 + 24.0 * np.arange(len(turns))
    assert len(turns) == 41 and T[turns] == pytest.approx(expected, abs=STEP_YR)


# --- coverage -------------------------------------------------------------------------------------------------------


def test_a_counted_run_gives_no_flag():
    raw = _noisy(25 * np.sin(2 * np.pi * T / 48.0))
    found = _audit(raw, spans=[(0.0, 1000.0)])
    assert len(found.runs) == 1 and found.missed == [] and found.flag == ''
    assert found.columns['aud_n_bounded_runs'] == 1 and found.columns['aud_n_uncovered'] == 0


def test_a_half_counted_run_gives_one_stretch_on_the_other_half():
    """Counted over 0-500 yr: the missed stretch starts at the first turn whose half-cycle's midpoint lies after 500
    (turn at 492 yr, midpoint 504) and ends at the last confirmed turn, 972 yr (the one at 996 yr retraces only
    25 (1 - cos 30 deg) = 3.3 deg before the end, under the 10-deg swing)."""
    found = _audit(_noisy(25 * np.sin(2 * np.pi * T / 48.0)), spans=[(0.0, 500.0)])
    (stretch,) = found.missed
    assert stretch.t0_yr == pytest.approx(492.0, abs=1.0) and stretch.t1_yr == pytest.approx(972.0, abs=1.0)
    assert stretch.period_yr == pytest.approx(48.0, abs=0.5) and math.degrees(stretch.half_amp_rad) == pytest.approx(25.0, abs=1.0)
    assert found.columns['aud_unc_t0_yr'] == stretch.t0_yr and found.columns['aud_unc_half_cycles'] == stretch.half_cycles


def test_a_run_at_the_record_end_is_an_edge_stretch():
    """A QS loop of 48 yr only over the last 120 yr (5 half-cycles), the window ending 60 yr before the end: 2
    half-cycles inside it, fewer than MMR_COORBITAL_AUDIT_MIN_HALF_CYCLES: columns only, no flag."""
    deg = np.where(T > 880.0, 25 * np.sin(2 * np.pi * (T - 880.0) / 48.0), 0.0) - np.where(T <= 880.0, 0.75 * (T - 880.0), 0.0)
    found = _audit(_noisy(deg), window=(55.0, 940.0))
    assert found.missed == [] and found.flag == ''
    assert found.columns['aud_edge_half_cycles'] >= S.mmr_coorbital_audit_min_half_cycles


def test_the_columns_of_a_missed_stretch_name_the_caller_reasons():
    found = ca.audit(
        T,
        _noisy(25 * np.sin(2 * np.pi * T / 48.0)),
        1.0,
        WINDOW,
        [],
        S,
        well_of=lambda s: 'qs',
        law_of=lambda s: (1.05, 0.99, ca.LAW_PASS),
        overlap_of=lambda s: 'below_period_floor',
    )
    columns = found.columns
    assert (columns['aud_unc_well'], columns['aud_unc_overlap_reason']) == ('qs', 'below_period_floor')
    assert (columns['aud_unc_law_K'], columns['aud_unc_law_R2'], columns['aud_unc_law_check']) == (1.05, 0.99, 'pass')
    assert abs(columns['aud_unc_centre_deg']) < 2.0
    assert found.flag == ca.FLAG_UNCOVERED and columns['aud_n_law_confirmed'] == 1.0


# review 2: a slow drift under a wave passes the pairwise centre test (12.5 deg per cycle against A = 20 deg); its law
# read K -0.8 and 0.3 there; 2016 JA's tadpole stretch K 2.34, R^2 0.46 (a thin pass at best)
@pytest.mark.parametrize('check', [ca.LAW_THIN, 'fail', ca.LAW_NOT_MEASURED])
def test_a_missed_stretch_whose_law_does_not_pass_is_no_flag(check):
    """The same QS stretch, its law thin, failed or not measured: the stretch stays in the columns (aud_n_uncovered,
    aud_unc_*), the flag needs a passing law (the force is the planet's)."""
    found = ca.audit(T, _noisy(25 * np.sin(2 * np.pi * T / 48.0)), 1.0, WINDOW, [], S, law_of=lambda s: (0.3, 0.46, check))
    assert found.missed and not found.confirmed and found.flag == ''
    assert found.columns['aud_n_uncovered'] >= 1 and found.columns['aud_n_law_confirmed'] == 0.0
    assert found.columns['aud_unc_law_check'] == check


def test_the_flag_reports_the_longest_stretch_whose_law_passes():
    """Two missed QS stretches (0-400 and 600-1000 yr; 400-600 yr counted): the law fails on the longer one only, so
    the flag stands and the columns describe the confirmed one."""
    raw = _noisy(25 * np.sin(2 * np.pi * T / 48.0))
    found = ca.audit(T, raw, 1.0, WINDOW, [(410.0, 590.0)], S, law_of=lambda s: (1.0, 0.99, 'fail' if s.t0_yr < 300 else ca.LAW_PASS))
    assert len(found.missed) == 2 and len(found.confirmed) == 1 and found.flag == ca.FLAG_UNCOVERED
    assert found.columns['aud_unc_t0_yr'] > 500.0 and found.columns['aud_unc_law_check'] == ca.LAW_PASS


def test_a_coarse_step_or_a_gap_is_not_measured():
    """A step not below half the 3-orbit window (orbit_averaged would return the raw samples, whose 15-deg orbital term
    would make turns) and a nan sample (the running sum would carry it on) leave Layer II unmeasured: no runs, no flag,
    the reason in aud_not_measured."""
    t = np.arange(0.0, 1000.0, 4.0)
    coarse = ca.audit(t, np.radians(25 * np.sin(2 * np.pi * t / 48.0)), 1.0, WINDOW, [], S)
    assert coarse.columns['aud_not_measured'] == ca.NOT_MEASURED_STEP and coarse.flag == '' and coarse.runs == []
    raw = _noisy(25 * np.sin(2 * np.pi * T / 48.0))
    raw[len(raw) // 3] = np.nan
    gap = ca.audit(T, raw, 1.0, WINDOW, [], S)
    assert gap.columns['aud_not_measured'] == ca.NOT_MEASURED_SAMPLES and gap.flag == ''
    assert np.isnan(gap.columns['aud_n_bounded_runs'])


# --- Layer I: a floor above the run's own loop ------------------------------------------------------------------------


def test_a_rejected_run_whose_floor_exceeds_its_loop_is_a_contradiction():
    """(i): 2015 YA's QS run, P 69.3 yr, loop 66.1 yr, rejected at the 73.1-yr QS floor; a run whose floor is below its
    loop (0.5 x 232 = 116 yr) or that counts is none."""
    found = ca.floor_contradiction([(0, 'below_period_floor', 73.1, 66.1), (1, '', 116.1, 232.2)], {})
    assert found == ca.FloorContradiction('run:0', 73.1, 66.1)
    assert ca.floor_contradiction([(1, 'below_period_floor', 116.1, 232.2), (2, '', 73.1, 66.1)], {}) is None


@pytest.mark.parametrize(
    'columns, fires',
    [
        # 2010 JK1: the pass rejected below the floor, the loop through the stay's 3 turns 28.6 yr below the 32.7-yr QS floor
        (
            {
                'geo_qs_pass': 'rejected',
                'geo_qs_reject': 'below_period_floor',
                'geo_qs_visit_turns': 3.0,
                'geo_qs_visit_loop_yr': 28.6,
                'flt_qs_floor_yr': 32.7,
            },
            True,
        ),
        # 54509 c046 without the rule: no turning points at the floor's prominence, stay's loop 48.1 < 55.7 yr
        (
            {
                'geo_qs_pass': 'rejected',
                'geo_qs_reject': 'no_turning_points',
                'geo_qs_visit_turns': 9.0,
                'geo_qs_visit_loop_yr': 48.1,
                'flt_qs_floor_yr': 55.7,
            },
            True,
        ),
        # the same numbers rejected for reasons the floor does not cause (review 1): the in-well episode was found
        # (keeps_outside); the geometry rejected the run itself (2022 VG too_few_tp, 2020 FN3 over_the_top)
        (
            {
                'geo_qs_pass': 'rejected',
                'geo_qs_reject': 'keeps_outside',
                'geo_qs_visit_turns': 3.0,
                'geo_qs_visit_loop_yr': 28.6,
                'flt_qs_floor_yr': 32.7,
            },
            False,
        ),
        (
            {
                'geo_qs_pass': 'rejected',
                'geo_qs_reject': 'not_accepted:too_few_tp',
                'geo_qs_visit_turns': 3.0,
                'geo_qs_visit_loop_yr': 28.6,
                'flt_qs_floor_yr': 32.7,
            },
            False,
        ),
        (
            {
                'geo_qs_pass': 'rejected',
                'geo_qs_reject': 'not_accepted:over_the_top',
                'geo_qs_visit_turns': 3.0,
                'geo_qs_visit_loop_yr': 28.6,
                'flt_qs_floor_yr': 32.7,
            },
            False,
        ),
        # 2023 CU: one turn, the orbit at rest, no loop
        (
            {
                'geo_qs_pass': 'rejected',
                'geo_qs_reject': 'below_period_floor',
                'geo_qs_visit_turns': 1.0,
                'geo_qs_visit_loop_yr': 25.2,
                'flt_qs_floor_yr': 30.3,
            },
            False,
        ),
        # 2015 SO2: the loop 30.2 yr above the 19.1-yr QS floor
        (
            {
                'geo_qs_pass': 'rejected',
                'geo_qs_reject': 'no_run_in_well',
                'geo_qs_visit_turns': 8.0,
                'geo_qs_visit_loop_yr': 30.2,
                'flt_qs_floor_yr': 19.1,
            },
            False,
        ),
        # a kept pass
        ({'geo_qs_pass': 'kept', 'geo_qs_visit_turns': 9.0, 'geo_qs_visit_loop_yr': 48.1, 'flt_qs_floor_yr': 55.7}, False),
    ],
    ids=['jk1', 'c046_no_tp', 'keeps_outside', 'too_few_tp', 'over_the_top', 'cu_one_turn', 'so2', 'kept'],
)
def test_a_rejected_qs_pass_below_its_own_loop(columns, fires):
    """(ii): only a rejection the pass's floor can cause (`FLOOR_REJECTIONS`); `where` names it."""
    found = ca.floor_contradiction([], columns)
    assert (found is not None) == fires
    if fires:
        assert found.where == f"qs_pass:{columns['geo_qs_reject']}"


# --- the verdict ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('flag', [Flag.UNCOVERED_BOUNDED_RUN, Flag.FLOOR_ABOVE_OWN_LOOP])
def test_an_audit_flag_puts_a_certain_row_in_the_review_queue_and_changes_nothing_else(flag):
    """A non_resonant/high circulation is no review row; with an audit flag it is, the flag is its review reason, and
    status, confidence and action stay."""
    plain = make_verdict(confidence='high', action='none')
    flagged = make_verdict(confidence='high', action='none', flags=[str(flag)])
    assert not plain.review_required and flagged.review_required
    assert flagged.review_reasons == [str(flag)]
    assert (flagged.status, flagged.confidence, flagged.action) == (plain.status, plain.confidence, plain.action)
    assert needs_review('high', 'none', '', '', f'jump_at_edge;{flag}') and not needs_review('high', 'none', '', '', 'jump_at_edge')


def test_the_uncovered_stretch_is_a_hatched_strip_of_panel_a():
    """(a) of the verdict figure: the stretch the audit found as a hatched gray strip with its label, in display units
    (kyr here); none without a missed stretch."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from resonances.plotting import mmr_combined as mc

    fig, ax = plt.subplots()
    try:
        columns = {'aud_n_uncovered': 1.0, 'aud_unc_t0_yr': 405.0, 'aud_unc_t1_yr': 669.5}
        mc.shade_audit(ax, columns, lambda t: np.asarray(t) / 1000.0, mc.LAYOUT_V1)
        strips = [p for p in ax.patches if p.get_gid() == mc.AUDIT_GID]
        assert len(strips) == 1 and strips[0].get_hatch() == mc.AUDIT_HATCH
        x0 = strips[0].get_x()
        assert (x0, x0 + strips[0].get_width()) == pytest.approx((0.405, 0.6695))
        assert [t.get_text() for t in ax.texts if t.get_gid() == mc.AUDIT_GID] == [mc.AUDIT_LABEL]
        before = len(ax.patches)
        mc.shade_audit(ax, {'aud_n_uncovered': 0.0, 'aud_unc_t0_yr': np.nan}, lambda t: t, mc.LAYOUT_V1)
        assert len(ax.patches) == before
    finally:
        plt.close(fig)
