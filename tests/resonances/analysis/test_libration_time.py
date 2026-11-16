"""`lib_time_yr` / `lib_share` (`policy.libration_time`): the record's time inside the counted episodes."""

import numpy as np
import pytest

from resonances.resonance.analysis import policy
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Reason, Status

from tests.resonances.analysis.builders import make_episode, make_geometry

RECORD = (0.0, 1000.0)


def episode(episode_id, times, start_gap_p=5.0, end_gap_p=5.0):
    """An episode with turning points at `times`, `start_gap_p` / `end_gap_p` periods from the record's edges (within
    LIBRATION_COVERAGE_PERIODS = 1 it covers that edge)."""
    return make_episode(
        episode_id=episode_id, tp_times_yr=np.asarray(times, dtype=float), coverage_start_gap_p=start_gap_p, coverage_end_gap_p=end_gap_p
    )


def outcome(counted=(), spans=()):
    return policy._Outcome(Status.TRANSIENT, Reason.GEOMETRIC_EPISODE, '', counted=frozenset(counted), counted_spans=tuple(spans))


def test_union_of_counted_runs_and_spans_carried_to_a_covered_edge():
    # run 0 covers the record's start: 0-206; run 1 500-600 and the span 580-700 overlap: 500-700; run 2 is not counted
    geometry = make_geometry(runs=[episode(0, [96, 150, 206], start_gap_p=0.5), episode(1, [500, 550, 600]), episode(2, [800, 850, 900])])
    total, share = policy.libration_time(geometry, outcome({0, 1}, [(580.0, 700.0)]), RECORD, AnalysisSettings())
    assert total == pytest.approx(206.0 + 200.0)
    assert share == pytest.approx(0.406)


def test_a_run_covering_both_edges_is_the_whole_record():
    geometry = make_geometry(runs=[episode(0, [132, 400, 929], start_gap_p=0.9, end_gap_p=0.5)])
    assert policy.libration_time(geometry, outcome({0}), RECORD, AnalysisSettings()) == pytest.approx((1000.0, 1.0))


def test_nothing_counted_has_no_time():
    geometry = make_geometry(runs=[episode(0, [96, 150, 206])])
    assert policy.libration_time(geometry, outcome(), RECORD, AnalysisSettings()) == (None, None)
