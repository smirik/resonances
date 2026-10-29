"""W3 guards, G2: three turning points are one centre estimate; a stationary centre is checked over the whole track.

Codex astra (PHYSICS-MATH §7): TPs [2, 0, 5.9] rad at [0, 500, 1000] yr have half-cycle midpoints [1, 2.95] that
average to one full-cycle centre, so shift = 0 and 'stationary' by construction - no second estimate. Codex sol (CASES):
15874 5N-23 had an accepted 10-TP episode whose smoothed centre moved 262 deg; its first and last cycle bands
overlapped (half-amplitudes up to 155 deg). The author keeps the 3-TP rule (the episode stays accepted), but the
centre is then not measured (L6 thin), and every band of the track must share a point.
"""

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.geometry import EpisodeShape, RejectReason, TurningPoints
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue, Status

S = settings_from_config()


def run_of(times, values) -> geo.Episode:
    values = np.asarray(values, dtype=float)
    kinds = np.where(values > np.r_[values[1:2], values[:-1]], 1, -1)
    tp = TurningPoints(np.arange(len(values)), np.asarray(times, dtype=float), values, kinds, 0.01)
    return geo.classify_run(0, tp, np.arange(len(values)), np.asarray([times[0], times[-1]], dtype=float), (times[0], times[-1]), S)


def test_three_turning_points_are_one_centre_estimate():
    episode = run_of([0.0, 500.0, 1000.0], [2.0, 0.0, 5.9])
    assert episode.accepted and episode.shape == EpisodeShape.STATIONARY  # the 3-TP rule stays (author)
    assert episode.centre_estimates == 1  # (3 - 1) // 2
    assert episode.centre_stationarity == CheckValue.NOT_MEASURED
    margin = decide._l6(episode, Ledger.from_settings(S), S)
    assert margin.margin == Margin.NOT_MEASURED  # a thin margin for the confidence, never 'full'


def test_five_turning_points_measure_the_centre():
    episode = run_of([0.0, 500.0, 1000.0, 1500.0, 2000.0], [1.0, -1.0, 1.0, -1.0, 1.0])
    assert episode.centre_estimates == 2 and episode.centre_stationarity == CheckValue.PASS
    assert decide._l6(episode, Ledger.from_settings(S), S).margin == Margin.FULL  # shift 0


def test_bands_by_hand():
    # bands [m - A, m + A]: [-1, 1], [0.5, 2.5], [-1, 1]: every pair overlaps; [-1, 1], [2, 4] do not
    assert geo.centre_bands_overlap(np.array([0.0, 1.5, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert not geo.centre_bands_overlap(np.array([0.0, 3.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    assert not geo.centre_bands_overlap(np.array([]), np.array([]))


def test_a_centre_that_goes_away_and_comes_back_once_is_not_stationary():
    """Half-cycles of 1 rad about 0, then about 3 rad, then about 0 again. The first and last full cycles share
    their band ([-1, 1]), the middle ones ([2, 4]) do not: max(m - A) = 2 > min(m + A) = 1. The centre turns back
    once (fewer than LIBRATION_CENTRE_RETURNS_MODULATED 3): rejected as centre_moves, not a stationary libration."""
    values = [1.0, -1.0, 1.0, -1.0, 4.0, 2.0, 4.0, 2.0, 1.0, -1.0, 1.0]
    episode = run_of([100.0 * i for i in range(len(values))], values)
    assert not episode.bands_overlap and episode.centre_stationarity == CheckValue.FAIL
    assert (episode.accepted, episode.reject_reason) == (False, RejectReason.CENTRE_MOVES)


@pytest.mark.benchmark
def test_15874_has_no_episode_with_a_wandering_centre():
    example = 'mmr2__15874__5N-23+0+18'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(io.load_series(example), S, scan_owner=False, scan_subresonances=False)
    for episode in analysis.geometry.episodes:
        assert episode.bands_overlap  # every accepted episode: the whole track inside one band
        assert decide.centre_travel_in(analysis.geometry, episode.t_start_yr, episode.t_end_yr) < np.radians(262)
    # the author's verdict (non_resonant): the 10-TP 'episode' is gone, the angle circulates
    assert (analysis.verdict.status, analysis.verdict.angle_status) == (Status.NON_RESONANT, AngleStatus.CIRCULATION)
