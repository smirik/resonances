"""How long an uncertain secular or Lidov-Kozai verdict asks to extend (`analysis.secular.extend`).

The expected lengths are computed by hand from the rules in docs/physics/confidence.md ("How long to extend") and the
.env.dist defaults: f_min = MMR_EXTEND_MIN_SPAN_FACTOR = 2, f_next = MMR_NEXT_CHECK_SPAN_FACTOR = 4,
LIBRATION_MIN_TURNING_POINTS = 3, FILTER_EDGE_TRIM_CUTOFFS = 1, ZLK_EXTEND_RESOLUTION_MARGIN = 1.25,
ZLK_EXTEND_MAX_YR = 1e8; the gate cases are the review's (373136 at 100 kyr with d = 3.5"/yr, 1981 at 500 kyr with
d = 1.9"/yr). No status changes: the same series with and without the extension rule gives the same status.
"""

import numpy as np
import pytest

from resonances.resonance.analysis.secular import GateResult, LongPeriodSeries, analyse_secular
from resonances.resonance.analysis.secular.analyse import analyse_long_period
from resonances.resonance.analysis.secular.extend import (
    CAUSE_BASELINE,
    CAUSE_GATE_CYCLES,
    CAUSE_GATE_PHASE,
    CAUSE_GATE_RESOLUTION,
    CAUSE_GATE_UNAVAILABLE,
    CAUSE_GATE_UNRESOLVABLE,
    recommend,
)
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Action, AngleStatus, Reason, ResonanceType, Status

SETTINGS = AnalysisSettings()  # the .env.dist defaults (test_skeleton checks they agree)
ZLK_CUTOFF = 2 * 937.0  # ZLK P_c: twice the great inequality
ZLK_FLOOR = 2 * ZLK_CUTOFF  # FILTER_MIN_LIBRATION_CUTOFFS x P_c = 3748 yr > ZLK_LIBRATION_PERIOD_MIN_YR


def residual(span_yr, d, baseline_yr=None, period=10_000.0):
    return GateResult(
        'planetary_residual',
        n_cycles=span_yr / period,
        min_cycles=3.0,
        libration_period_yr=period,
        baseline_yr=span_yr if baseline_yr is None else baseline_yr,
        min_distance_arcsec_yr=d,
    )


def zlk_recommend(gate, span_yr, period=10_000.0, reason=Reason.GATE_UNAVAILABLE):
    return recommend(Status.UNCERTAIN, reason, gate, ZLK_FLOOR, period, ZLK_CUTOFF, span_yr, SETTINGS)


# --- the gate's planetary residual --------------------------------------------------------------------------


def test_373136_a_residual_of_3_5_arcsec_per_yr_at_100_kyr_asks_for_400_kyr():
    # 1.25 x 1296000 / 3.5 = 462 857 yr needed; the next check is capped at 4T = 400 kyr, above 2T = 200 kyr
    extension = zlk_recommend(residual(1e5, 3.5), 1e5)
    assert extension.action == Action.EXTEND and extension.cause == CAUSE_GATE_RESOLUTION
    assert extension.required_yr == pytest.approx(1.25 * 1296000 / 3.5)
    assert extension.extend_to_years == pytest.approx(4e5)
    assert extension.estimate_yr == pytest.approx(5e5)  # round_up(462 857)


def test_1981_a_residual_of_1_9_arcsec_per_yr_at_500_kyr_asks_for_1_myr():
    # 1.25 x 1296000 / 1.9 = 852 632 yr < 2T = 1 Myr: the floor of every extension
    extension = zlk_recommend(residual(5e5, 1.9), 5e5)
    assert extension.action == Action.EXTEND
    assert extension.extend_to_years == pytest.approx(1e6)


def test_the_free_elements_lost_span_is_added():
    # T = 100 kyr, free baseline 60 kyr, d = 20"/yr: 1.25 x 64 800 + 40 000 = 121 000 yr -> round_up 200 kyr
    extension = zlk_recommend(residual(1e5, 20.0, baseline_yr=6e4), 1e5)
    assert extension.required_yr == pytest.approx(1.25 * 64800 + 40000)
    assert extension.extend_to_years == pytest.approx(2e5)


@pytest.mark.parametrize('d', [0.0, 0.01])
def test_a_residual_no_practical_run_resolves_is_reviewed(d):
    # 1.25 x 1296000 / 0.01 = 1.6e8 yr > ZLK_EXTEND_MAX_YR = 1e8 (and d = 0 never resolves)
    extension = zlk_recommend(residual(5e5, d), 5e5)
    assert extension.action == Action.REVIEW and extension.extend_to_years is None
    assert extension.cause == CAUSE_GATE_UNRESOLVABLE


# --- the gate's other outcomes ------------------------------------------------------------------------------


def test_a_short_window_asks_for_its_cycles():
    # P_cyc = 40 kyr over T = 40 kyr (1 cycle < 3): 3 x 40 kyr + (40 000 - 39 500) lost = 120 500 yr;
    # geometry 1.5 x 40 kyr + 2 x 1874 = 63 748 yr is less; next check round_up(max(80 kyr, min(120.5, 160 kyr)))
    gate = GateResult('short_window', n_cycles=39_500 / 4e4, min_cycles=3.0, libration_period_yr=4e4, baseline_yr=39_500.0)
    extension = zlk_recommend(gate, 4e4, period=4e4)
    assert extension.action == Action.EXTEND and extension.cause == CAUSE_GATE_CYCLES
    assert extension.required_yr == pytest.approx(120_500.0)
    assert extension.extend_to_years == pytest.approx(2e5)


@pytest.mark.parametrize('outcome, cause', [('undefined_phase', CAUSE_GATE_PHASE), ('unavailable', CAUSE_GATE_UNAVAILABLE)])
def test_an_undefined_phase_or_no_free_elements_is_reviewed(outcome, cause):
    extension = zlk_recommend(GateResult(outcome, n_cycles=10.0, min_cycles=3.0), 1e5)
    assert extension.action == Action.REVIEW and extension.extend_to_years is None and extension.cause == cause


def test_a_verdict_that_is_not_uncertain_keeps_its_default_action():
    extension = recommend(Status.RESONANT, Reason.DYNAMIC_LIBRATION, None, 2e5, 1e6, 1e5, 1e7, SETTINGS)
    assert extension.action is None and extension.extend_to_years is None


# --- the geometric baseline ---------------------------------------------------------------------------------


def test_a_secular_angle_without_a_period_asks_for_the_floor_baseline():
    # T = 100 kyr, P_c = 100 kyr, floor 200 kyr, no cycle: N = 1.5 x 200 + 2 x 100 = 500 kyr;
    # next check round_up(max(200, min(500, 400))) = 400 kyr; the estimate round_up(max(500, 200)) = 500 kyr
    extension = recommend(Status.UNCERTAIN, Reason.BASELINE_NEEDED, None, 2e5, float('nan'), 1e5, 1e5, SETTINGS)
    assert extension.action == Action.EXTEND and extension.cause == CAUSE_BASELINE
    assert extension.period_source == 'period_floor'
    assert extension.required_yr == pytest.approx(5e5)
    assert extension.extend_to_years == pytest.approx(4e5) and extension.estimate_yr == pytest.approx(5e5)


def test_a_measured_cycle_longer_than_the_floor_sets_the_period():
    # P_cyc = 1 Myr over T = 2 Myr: N = 1.5 Myr + 200 kyr = 1.7 Myr < T: the next check is 2T = 4 Myr
    extension = recommend(Status.UNCERTAIN, Reason.BASELINE_NEEDED, None, 2e5, 1e6, 1e5, 2e6, SETTINGS)
    assert extension.period_source == 'cycle' and extension.period_yr == pytest.approx(1e6)
    assert extension.required_yr == pytest.approx(1.7e6)
    assert extension.extend_to_years == pytest.approx(4e6)


def test_a_partial_cycle_of_a_secular_angle_end_to_end():
    # a quarter of a 2-Myr cycle on 400 kyr (no turning point): uncertain, baseline_needed; no period is measured, so
    # the floor sets N = 500 kyr and the next check is round_up(max(2 x 400, min(500, 4 x 400))) = 800 kyr
    t = np.linspace(0.0, 4e5, 801)
    angle = 0.8 * np.sin(2 * np.pi * t / 2e6)
    n = len(t)
    series = LongPeriodSeries(ResonanceType.SECULAR, 'test', 'arc', t, angle, np.full(n, 2.5), np.full(n, 0.1), np.full(n, 0.1))
    analysis = analyse_secular(series, SETTINGS)
    verdict = analysis.verdict
    assert (verdict.status, verdict.reason, verdict.angle_status) == (
        Status.UNCERTAIN,
        Reason.BASELINE_NEEDED,
        AngleStatus.INSUFFICIENT_TIME,
    )
    assert verdict.action == Action.EXTEND and verdict.extend_to_years == pytest.approx(8e5)
    row = analysis.summary_row()
    assert row['extend_to_years'] == pytest.approx(8e5)
    assert row['base_required_kyr'] == pytest.approx(500.0) and row['base_extend_cause'] == CAUSE_BASELINE


def test_the_gate_changes_the_action_not_the_status():
    # a ZLK libration of 10 kyr over 100 kyr whose gate finds a residual 3.5"/yr from a planetary rate: uncertain
    # (gate_unavailable) as before, now with action extend to 400 kyr; an undefined phase: review
    t = np.linspace(0.0, 1e5, 10001)
    n = len(t)
    angle = np.pi / 2 + 0.5 * np.sin(2 * np.pi * t / 1e4)
    series = LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'zlk', t, angle, np.full(n, 2.5), np.full(n, 0.3), np.full(n, 0.6))

    def run(result):
        return analyse_long_period(series, SETTINGS, gate=lambda decision, geometry: result).verdict

    resolved = run(residual(1e5, 3.5, period=1e4))
    assert (resolved.status, resolved.reason, resolved.angle_status) == (Status.UNCERTAIN, Reason.GATE_UNAVAILABLE, AngleStatus.LIBRATION)
    assert resolved.action == Action.EXTEND and resolved.extend_to_years == pytest.approx(4e5)
    phase = run(GateResult('undefined_phase', n_cycles=10.0, min_cycles=3.0, libration_period_yr=1e4))
    assert (phase.status, phase.reason) == (Status.UNCERTAIN, Reason.GATE_UNAVAILABLE)
    assert phase.action == Action.REVIEW and phase.extend_to_years is None
