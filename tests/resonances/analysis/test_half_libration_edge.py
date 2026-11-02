"""W29 codex S1: a half-swing of an angle that is not 1:1 (`geometry.half_librations`, leaf G4h) with no neighbour run on
one side is cut by the record's edge only when sigma_f stays bounded from its turning point to the window's edge; a
turn of >= one revolution there is a passage (`geometry.edge_travel_rev`). Before S1 the side counted as the record's
edge whatever followed, so 42946 43S-6 (about 8 turns after its half-swing) asked for a futile extend.

The travel of `edge_travel_rev` is checked on ramps whose travel is known in closed form: sigma = 2 pi t / P advances
(t1 - t0) / P revolutions on [t0, t1]; a ZLK omega counts its revolution as pi, so the same ramp is twice as many.
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.settings import AnalysisSettings
from tests.resonances.analysis.builders import make, make_episode, make_geometry

S = AnalysisSettings()
TWO_PI = 2 * np.pi
C = -0.05  # identity coefficient sigma_dot = c (a - a*) [rad yr^-1 au^-1]: finite, so the angle is an MMR's
WINDOW = (0.0, 10000.0)
T0, T1 = 4100.0, 4500.0  # the half-swing's two turning points [yr]
SIGMA0 = 1.0  # sigma at the upper turning point [rad]; the lower one is SIGMA0 - 2


# --- edge_travel_rev ---------------------------------------------------------------------------------------------------


def test_edge_travel_counts_revolutions_of_a_ramp():
    """sigma = 2 pi t / 100 yr on 0..1000 yr: 2.5 revolutions on [200, 450]; clipped by the window (0, 300) to
    [200, 300], 1.0; as a ZLK omega (revolution pi) 2.5 x 2 = 5; an empty or one-sample interval travels 0."""
    t = np.arange(0.0, 1001.0, 1.0)
    sigma = TWO_PI * t / 100.0
    assert g.edge_travel_rev(t, sigma, 200.0, 450.0, (0.0, 1000.0)) == pytest.approx(2.5)
    assert g.edge_travel_rev(t, sigma, 200.0, 450.0, (0.0, 300.0)) == pytest.approx(1.0)
    assert g.edge_travel_rev(t, sigma, 200.0, 450.0, (0.0, 1000.0), rev=np.pi) == pytest.approx(5.0)
    assert g.edge_travel_rev(t, sigma, 500.0, 400.0, (0.0, 1000.0)) == 0.0
    assert g.edge_travel_rev(t, sigma, 500.0, 500.2, (0.0, 1000.0)) == 0.0


def test_edge_travel_is_the_swing_not_the_net_advance():
    """A libration of half-amplitude 2 rad that ends where it began: its net advance is 0 but max - min is 4 rad,
    4 / 2 pi = 0.637 revolutions; nan samples are skipped."""
    t = np.linspace(0.0, 100.0, 1001)
    sigma = 2.0 * np.sin(TWO_PI * t / 50.0)
    sigma[10] = np.nan
    assert g.edge_travel_rev(t, sigma, 0.0, 100.0, (0.0, 100.0)) == pytest.approx(4.0 / TWO_PI, rel=1e-3)


# --- the S1 side of a half-swing -----------------------------------------------------------------------------------------


def angle_with_half_swing(before_rev: float, after_rev: float, step: float = 1.0) -> g.FilteredAngle:
    """sigma_f rising by `before_rev` revolutions to the upper turning point at T0, a cosine half-swing down by 2 rad to
    T1, then rising by `after_rev` revolutions to the window's end: one rejected 2-TP run with no neighbour run."""
    t = np.arange(WINDOW[0], WINDOW[1] + step, step)
    sigma = np.empty_like(t)
    head, swing, tail = t <= T0, (t > T0) & (t < T1), t >= T1
    sigma[head] = SIGMA0 - TWO_PI * before_rev * (T0 - t[head]) / (T0 - WINDOW[0])
    sigma[swing] = SIGMA0 - 1.0 + np.cos(np.pi * (t[swing] - T0) / (T1 - T0))
    sigma[tail] = SIGMA0 - 2.0 + TWO_PI * after_rev * (t[tail] - T1) / (WINDOW[1] - T1)
    rate = np.gradient(sigma, t)
    return make(
        g.FilteredAngle,
        t_yr=t,
        angle_raw=sigma,
        angle_f=sigma,
        rate_f=rate,
        angle_f_half=sigma,
        angle_f_double=sigma,
        slow_f=1.0 + rate / C,
        identity_coefficient=C,
        cutoff_yr=50.0,
        step_yr=step,
        window=WINDOW,
    )


def half_swings(before_rev: float, after_rev: float, settings: AnalysisSettings = S):
    run = make_episode(2, tp_times_yr=np.array([T0, T1]), tp_values_rad=np.array([SIGMA0, SIGMA0 - 2.0]), accepted=False)
    geometry = make_geometry(runs=[run], window=WINDOW, period_floor_yr=100.0)
    g.half_librations(geometry, angle_with_half_swing(before_rev, after_rev), settings)
    return geometry.half_librations


def test_a_half_swing_between_two_circulations_without_neighbour_runs_is_between_passages():
    """41 turns before and 8 after (42946 43S-6 has about 8.1 after): both sides are passages, so the half-swing is
    not at the record's edge (edge ''), and leaf G4h never asks for an extend for it. Before S1 both sides read as
    the record's edge and the half-swing was not found at all; with a run on one side only it read as an edge one."""
    found = half_swings(41.0, 8.0)
    assert len(found) == 1
    assert found[0].edge == ''
    assert (found[0].t0_yr, found[0].t1_yr) == (T0, T1)
    assert found[0].half_amp_rad == pytest.approx(1.0)


@pytest.mark.parametrize('after_rev, edge', [(0.0, g.LEG_EDGE_END), (0.9, g.LEG_EDGE_END), (1.1, '')])
def test_the_end_side_is_the_record_edge_only_below_one_revolution(after_rev, edge):
    """The tail starts at the lower turning point and rises monotonically, so its travel to the window's end is exactly
    `after_rev`: below one revolution the record's end cuts the half-swing (edge 'end', G4h extend), from one
    revolution on it is a passage."""
    found = half_swings(41.0, after_rev)
    assert [half.edge for half in found] == [edge]


def test_bounded_on_both_sides_is_no_half_swing_between_passages():
    """Bounded before and after (0.5 revolutions each): neither side is a passage, so no half-libration (the rule
    needs a passage on at least one side), as before S1."""
    assert half_swings(0.5, 0.5) == ()


def test_the_start_side_is_mirrored():
    """Bounded before (0.3 revolutions to the window's start) and a passage after: edge 'start'."""
    assert [half.edge for half in half_swings(0.3, 8.0)] == [g.LEG_EDGE_START]


def test_the_switch_off_finds_nothing():
    assert half_swings(41.0, 8.0, replace(S, mmr_single_half_libration=False)) == ()


# --- the benchmark record (local data) ---------------------------------------------------------------------------------


@pytest.mark.benchmark
def test_42946_43s6_half_swing_is_between_passages_and_asks_no_extend():
    """42946 43S-6 (50 kyr): the half-swing 28476-32285 yr is followed by about 8 turns of sigma_f, so it lies between
    passages (no ':end' in geo_half_libration_read) and the row does not extend. Under the v2 tree it is
    uncertain/one_half_libration/review; under the minimal policy (W29 §3.2) non_resonant/circulation."""
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    example = 'mmr2__42946__43S-6+0-37'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    series = io.load_series(example)
    v2 = analyse_mmr(series, replace(S, classifier_policy='v2'))
    row = v2.summary_row()
    assert row['geo_half_libration_read'].startswith('28476.1-32285.5:')
    assert ':end' not in row['geo_half_libration_read']
    assert (row['status'], row['reason'], row['action']) == ('uncertain', 'one_half_libration', 'review')
    minimal = analyse_mmr(series, replace(S, classifier_policy='minimal')).verdict
    assert (str(minimal.status), str(minimal.reason)) == ('non_resonant', 'circulation')
    assert str(minimal.action) != 'extend'
