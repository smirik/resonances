"""W29 minimal policy: the level from the evidence (`policy.level`) and the cycle count N (`policy.episode_cycles`).

Every expected level is read off the table of the spec (§6) by hand for the defaults CONFIDENCE_HIGH_CYCLES = 3 and
CONFIDENCE_CERTAIN_CYCLES = 10; every N is counted by hand from the turning points and the edge gaps."""

import itertools
import math

import numpy as np
import pytest

from resonances.resonance.analysis import policy
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CONFIDENCE_ORDER, Confidence, Status
from tests.resonances.analysis.builders import make_episode

SETTINGS = AnalysisSettings(classifier_policy='minimal')
NAN = math.nan
C, H, M, L = Confidence.CERTAIN, Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW


def ev(n, margin=policy.M_FULL, veto_state=policy.V_CLEAN, stable=True, **fields):
    return policy.Evidence(n_cycles=n, margin=margin, veto_state=veto_state, stable=stable, **fields)


@pytest.mark.parametrize(
    'n, margin, veto_state, stable, expected',
    [
        (0.5, policy.M_FULL, policy.V_CLEAN, True, M),  # half a cycle (a 1:1 leg): medium
        (1.0, policy.M_FULL, policy.V_CLEAN, True, M),
        (1.5, policy.M_FULL, policy.V_CLEAN, True, M),
        (2.5, policy.M_FULL, policy.V_CLEAN, True, M),  # TX1: 2.5 < 3
        (3.0, policy.M_FULL, policy.V_CLEAN, True, H),  # exactly the high count
        (9.5, policy.M_FULL, policy.V_CLEAN, True, H),
        (10.0, policy.M_FULL, policy.V_CLEAN, True, C),  # exactly the certain count, all clean
        (40.0, policy.M_FULL, policy.V_CLEAN, True, C),
        (40.0, policy.M_THIN, policy.V_CLEAN, True, H),  # no amplitude margin: high, not certain
        (40.0, policy.M_FULL, policy.V_CLEAN, False, H),  # the filter not stable
        (40.0, policy.M_FULL, policy.V_PARTIAL, True, H),  # the owner search not measured: high at most
        (40.0, policy.M_FULL, policy.V_NOT_MEASURED, True, M),  # a needed veto not measured: medium
        (40.0, policy.M_FULL, policy.V_AMBIGUOUS, True, L),  # a measured objection: low whatever N
        (0.5, policy.M_NOT_MEASURED, policy.V_AMBIGUOUS, False, L),
        (3.0, policy.M_NOT_MEASURED, policy.V_PARTIAL, False, H),
    ],
)
def test_librating_levels_follow_the_table(n, margin, veto_state, stable, expected):
    for status in (Status.RESONANT, Status.TRANSIENT):
        assert policy.level(status, ev(n, margin, veto_state, stable), SETTINGS) == expected


def test_near_separatrix_uncertain_and_invalid():
    assert policy.level(Status.NEAR_SEPARATRIX, ev(40.0), SETTINGS) == M
    assert policy.level(Status.NEAR_SEPARATRIX, ev(40.0, veto_state=policy.V_AMBIGUOUS), SETTINGS) == L
    assert policy.level(Status.UNCERTAIN, ev(40.0), SETTINGS) == Confidence.NONE
    assert policy.level(Status.INVALID, ev(40.0), SETTINGS) == Confidence.NONE


def test_the_non_resonant_ladder():
    nr = Status.NON_RESONANT
    assert policy.level(nr, ev(NAN, veto='owner'), SETTINGS) == H  # a full veto
    assert policy.level(nr, ev(NAN, veto='owner', veto_thin=True), SETTINGS) == M  # 624, GK3
    assert policy.level(nr, ev(NAN, veto='owner', second_capture=True), SETTINGS) == M
    assert policy.level(nr, ev(NAN, veto='strength', candidate=True), SETTINGS) == M  # a stable candidate drawn
    assert policy.level(nr, ev(NAN, candidate=True), SETTINGS) == M  # a leg, a half-swing, a near-episode
    assert policy.level(nr, ev(NAN, circulation_clear=True), SETTINGS) == C  # C1 and C2 full (32929)
    assert policy.level(nr, ev(NAN), SETTINGS) == H  # a drift or any other circulation
    assert policy.level(nr, ev(NAN, veto_state=policy.V_AMBIGUOUS), SETTINGS) == H  # low is for librating rows only


def test_the_level_never_falls_as_n_rises():
    grid = np.arange(0.0, 15.0, 0.25)
    rank = {level: CONFIDENCE_ORDER.index(level) for level in CONFIDENCE_ORDER}
    for margin, veto_state, stable in itertools.product(
        (policy.M_FULL, policy.M_THIN, policy.M_NOT_MEASURED), policy.V_ORDER, (True, False)
    ):
        levels = [rank[policy.level(Status.RESONANT, ev(n, margin, veto_state, stable), SETTINGS)] for n in grid]
        assert all(later <= earlier for earlier, later in zip(levels, levels[1:])), (margin, veto_state, stable)


def test_the_cycle_counts_follow_the_settings():
    stricter = AnalysisSettings(classifier_policy='minimal', confidence_high_cycles=5.0, confidence_certain_cycles=20.0)
    assert policy.level(Status.RESONANT, ev(4.0), stricter) == M
    assert policy.level(Status.RESONANT, ev(12.0), stricter) == H
    with pytest.raises(ValueError):
        AnalysisSettings(confidence_high_cycles=10.0, confidence_certain_cycles=10.0)


def test_episode_cycles_count_turning_points_and_covered_edges():
    # 7 turning points: 3 complete cycles
    assert policy.episode_cycles(make_episode(7, coverage_start_gap_p=5.0, coverage_end_gap_p=5.0), SETTINGS) == 3.0
    # covered edges (gaps 0.4 and 0.25 <= LIBRATION_COVERAGE_PERIODS = 1): 3 + 0.4 + 0.25
    covered = make_episode(7, coverage_start_gap_p=0.4, coverage_end_gap_p=0.25)
    assert policy.episode_cycles(covered, SETTINGS) == pytest.approx(3.65)
    # an observed exit at the end: that gap is no libration
    exits = make_episode(7, coverage_start_gap_p=0.4, coverage_end_gap_p=0.25, exit_observed=True)
    assert policy.episode_cycles(exits, SETTINGS) == pytest.approx(3.4)
    assert policy.episode_cycles(covered, SETTINGS, entry_seen=True) == pytest.approx(3.25)
    # a 2-TP run its phase return closed: one cycle
    assert policy.episode_cycles(make_episode(2, closed_by_return=True, coverage_start_gap_p=3.0, coverage_end_gap_p=3.0), SETTINGS) == 1.0
    # 14 turning points with gaps 0.41 and 3.96 (33128 6N-7, episode 0): 6.5 + 0.41
    e33128 = make_episode(14, coverage_start_gap_p=0.41179, coverage_end_gap_p=3.963141)
    assert policy.episode_cycles(e33128, SETTINGS) == pytest.approx(6.91179)


def test_union_counts_an_overlap_once():
    first = make_episode(
        5, tp_times_yr=np.array([0.0, 50.0, 100.0, 150.0, 200.0]), period_med_yr=100.0, coverage_start_gap_p=9.0, coverage_end_gap_p=9.0
    )
    second = make_episode(
        5, tp_times_yr=np.array([150.0, 200.0, 250.0, 300.0, 350.0]), period_med_yr=100.0, coverage_start_gap_p=9.0, coverage_end_gap_p=9.0
    )
    # 2 + 2 cycles, the second starts 50 yr (half a period) before the first ends: 2 + 1.5
    assert policy._union_cycles([second, first], SETTINGS) == pytest.approx(3.5)


def test_the_basis_column_grammar():
    text = policy.Evidence(6.5, policy.N_COVER, policy.M_FULL, policy.V_PARTIAL, True, 'owner', True).basis()
    assert text == 'N=6.5(cover);M=full;V=partial;S=full;veto=owner:thin'
    assert policy.Evidence().basis() == 'N=nan(-);M=not_measured;V=not_measured;S=thin;veto=-'
    assert policy.worst_veto_state([policy.V_PARTIAL, policy.V_AMBIGUOUS, policy.V_CLEAN]) == policy.V_AMBIGUOUS
    assert policy.worst_veto_state([]) == policy.V_CLEAN
