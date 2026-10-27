"""One slow edge half-cycle trimmed off a run whose centre moves (`geometry._trim_slow_edges`, W7 rule R1).

The numbers are worked out by hand: the full cycles of a uniform drift s_j = j b + (-1)^j A over N TPs have midpoints
b, 2b, ..., (N - 2) b and half-amplitudes A, so the band test max(m - A) < min(m + A) passes when (N - 3) b < 2 A."""

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
RATIO = S.libration_regular_period_ratio  # 2.0 in .env.dist: every slow/regular time below is written against it
RECORD = (0.0, 1.0e6)

# 2005 GQ107 40M-57+0+17 (benchmark, 50 kyr): the run at 31.7-40.2 kyr, half-cycles 15-26 of the W7 dump (midpoint
# and half-amplitude rounded to 0.1 deg); the TPs alternate from a minimum
GQ107_TIMES = [31716.1, 32836.9, 33554.5, 34342.3, 34835.5, 35401.3, 35916.1, 36372.7, 36973.3, 37460.5, 38146.9, 38850.7, 40185.1]
GQ107_MIDS = [86.4, 106.5, 111.1, 118.4, 118.9, 116.6, 109.0, 96.3, 80.4, 56.8, 30.4, -18.8]
GQ107_AMPS = [67.6, 47.5, 52.0, 44.7, 45.2, 47.4, 39.8, 52.5, 36.7, 60.3, 33.9, 83.1]
GQ107_RECORD = (0.0, 49999.96)
GQ107_WINDOW = (12.0, 49987.96)


def turning_points(times, values_deg) -> g.TurningPoints:
    values = np.radians(np.asarray(values_deg, dtype=float))
    kinds = np.where(np.arange(len(values)) % 2 == 0, np.sign(values[0] - values[1]), np.sign(values[1] - values[0]))
    return g.TurningPoints(np.arange(len(values)), np.asarray(times, dtype=float), values, kinds.astype(int), np.radians(1.0))


def gq107() -> g.TurningPoints:
    values = [m + (a if i % 2 else -a) for i, (m, a) in enumerate(zip(GQ107_MIDS, GQ107_AMPS))]
    values.append(2 * GQ107_MIDS[-1] - values[-1])
    return turning_points(GQ107_TIMES, values)


def drift(n: int, b_deg: float, a_deg: float, half_periods) -> g.TurningPoints:
    """s_j = j b + (-1)^j A at the given half-periods."""
    times = np.concatenate([[0.0], np.cumsum(half_periods)])
    assert len(times) == n
    return turning_points(times, [j * b_deg + (-1) ** j * a_deg for j in range(n)])


def episodes(tp, record=RECORD, window=RECORD, first_id=0):
    return g._find_episodes(tp, record, window, S, first_id)


def test_gq107_slow_last_half_cycle_is_trimmed():
    tp = gq107()
    half = np.diff(GQ107_TIMES)
    full = np.asarray(GQ107_TIMES[2:]) - np.asarray(GQ107_TIMES[:-2])
    # the last half-cycle, 1334.4 yr, is 2.07 x the 643.4-yr median half-period; its full period 2038.2 yr only 1.74 x
    # the 1174.2-yr median period, so the split at the regular stretch does not see it
    assert half[-1] / np.median(half) == pytest.approx(2.07, abs=0.01)
    assert full[-1] / np.median(full) == pytest.approx(1.74, abs=0.01)
    whole = g._classify_run(0, tp, np.arange(13), GQ107_RECORD, S, GQ107_WINDOW)
    assert whole.reject_reason == g.RejectReason.CENTRE_MOVES
    assert g._split_regular(tp, np.arange(13), S) == [pytest.approx(np.arange(13))]

    found = episodes(tp, GQ107_RECORD, GQ107_WINDOW, first_id=2)
    assert [(e.episode_id, e.n_tp, e.accepted) for e in found] == [(2, 12, True), (3, 1, False)]
    assert found[0].shape == g.EpisodeShape.STATIONARY
    assert (found[0].t_start_yr, found[0].t_end_yr) == (31716.1, 38850.7)
    assert found[1].reject_reason == g.RejectReason.EDGE and found[1].t_start_yr == 40185.1


def test_uniform_drift_with_equal_half_periods_stays_centre_moves():
    # N = 13, b = 10, A = 45: (13 - 3) x 10 = 100 > 2A = 90 fails, and no half-cycle is slow
    tp = drift(13, 10.0, 45.0, [100.0] * 12)
    assert g._trim_slow_edges(tp, np.arange(13), RECORD, RECORD, S) is None
    found = episodes(tp)
    assert len(found) == 1 and found[0].reject_reason == g.RejectReason.CENTRE_MOVES and found[0].n_tp == 13


def test_slow_edge_whose_remainder_still_drifts_changes_nothing():
    # b = 10, A = 42.5: 12 TPs fail (9 x 10 = 90 > 85, only N <= 11 passes); the last half-cycle, 2.8 x the median, is slow
    tp = drift(13, 10.0, 42.5, [100.0] * 11 + [280.0])
    assert not g._classify_run(0, tp, np.arange(12), RECORD, S).accepted
    assert g._trim_slow_edges(tp, np.arange(13), RECORD, RECORD, S) is None
    found = episodes(tp)
    assert len(found) == 1 and found[0].reject_reason == g.RejectReason.CENTRE_MOVES


def test_only_one_half_cycle_is_trimmed_per_edge():
    # the first two half-cycles slow (250 yr against a 100-yr median): 11 TPs from TP 2 would pass, but one cut leaves 12
    tp = drift(13, 10.0, 42.5, [250.0, 250.0] + [100.0] * 10)
    assert g._classify_run(0, tp, np.arange(2, 13), RECORD, S).accepted
    assert g._trim_slow_edges(tp, np.arange(13), RECORD, RECORD, S) is None
    found = episodes(tp)
    assert len(found) == 1 and found[0].reject_reason == g.RejectReason.CENTRE_MOVES and found[0].n_tp == 13


def test_most_turning_points_then_longer_then_earlier():
    # 12 TPs, b = 10, A = 42.5: the whole run fails (9 x 10 = 90 > 85), 11 TPs pass (80 < 85); both edges are slow,
    # so the left, the right and both cuts pass, and one cut keeps more TPs than two
    left_slower = drift(12, 10.0, 42.5, [280.0] + [100.0] * 9 + [250.0])
    pieces = g._trim_slow_edges(left_slower, np.arange(12), RECORD, RECORD, S)
    assert [len(p) for p in pieces] == [11, 1]  # without the faster right half-cycle the core lasts longer
    right_slower = drift(12, 10.0, 42.5, [250.0] + [100.0] * 9 + [280.0])
    assert [len(p) for p in g._trim_slow_edges(right_slower, np.arange(12), RECORD, RECORD, S)] == [1, 11]
    equal = drift(12, 10.0, 42.5, [250.0] + [100.0] * 9 + [250.0])
    assert [len(p) for p in g._trim_slow_edges(equal, np.arange(12), RECORD, RECORD, S)] == [11, 1]  # the earlier core

    found = episodes(right_slower, first_id=5)
    assert [(e.episode_id, e.n_tp, e.accepted, str(e.reject_reason)) for e in found] == [(5, 1, False, 'edge'), (6, 11, True, '')]


def test_accepted_run_is_untouched():
    # a stationary run with a slow last half-cycle stays one episode: the trim acts only on centre_moves
    tp = drift(13, 0.0, 45.0, [100.0] * 11 + [280.0])
    found = episodes(tp)
    assert len(found) == 1 and found[0].accepted and found[0].n_tp == 13
