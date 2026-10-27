"""W5: the centre of each regime before the last jump of a is low-passed on that regime's samples only (codex astra W5
findings 1 and 7).

The record: a libration of 40 deg and 500 yr about pi until `end` (3 or 6 cycles), a jump of a, then a strong
circulation of one turn per 40 yr; with `second`, another jump at 10 kyr (energies 0.2, 1.5, 3). A centre
low-passed over [t0, last jump] (the W4 fix with one cut) carries the circulation between the jumps into the
episode before the first one: its centre then passes 7-16 rad > 2 pi and G1e drops the episode (non_resonant).
Each regime's own centre leaves the episode at its true centre (travel < 0.2 rad) and the status transient.
"""

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.verdict import Status
from tests.resonances.analysis.test_impl_w4_status_episodes import ONE_DEG, S, decide_all, filtered

PERIOD = 500.0
SPAN = 20000.0
SECOND_JUMP = 10000.0


def two_regime_record(end: float, second: bool, turn_yr: float = 40.0):
    t = np.arange(0, SPAN, 1.0)
    early = np.pi + np.radians(40) * np.sin(2 * np.pi * t / PERIOD)
    angle = np.where(t < end, early, np.pi + 2 * np.pi * (t - end) / turn_yr)
    slow = 1 + 1e-4 * np.cos(2 * np.pi * t / PERIOD) + 1e-3 * (t >= end) + (2e-3 * (t >= SECOND_JUMP) if second else 0.0)
    fa = filtered(t, angle, 50.0, slow=slow)
    geometry = g.measure_geometry(fa, ONE_DEG, S)
    cycles = np.arange(PERIOD / 2, SPAN, PERIOD)
    energies = np.where(cycles < end, 0.2, np.where((cycles < SECOND_JUMP) | (not second), 1.5, 3.0))
    return g.finish_geometry(geometry, fa, PERIOD, np.nan, np.nan, 0.02, cycles, energies, 5e-4, PERIOD, [], S), fa


@pytest.mark.parametrize('second', [True, False])
@pytest.mark.parametrize('end', [1500.0, 3000.0])
def test_an_episode_before_two_jumps_keeps_its_centre(end, second):
    geometry, _ = two_regime_record(end, second)
    cuts = sorted(jump.time_yr for jump in geometry.jumps if not jump.kick)
    assert len(cuts) == (2 if second else 1) and abs(cuts[0] - end) < 50
    assert len(geometry.record_centres) == len(cuts)
    assert geometry.record_centres[0].t_yr[-1] < cuts[0]  # the first regime's centre ends before its jump
    (episode,) = geometry.episodes_before_jump
    assert episode.t_end_yr < end
    assert decide.centre_travel_in(geometry, episode.t_start_yr, episode.t_end_yr) < 0.2
    _, decision, _ = decide_all(geometry)
    assert decision.status == Status.TRANSIENT


@pytest.mark.parametrize('end', [1500.0, 3000.0])
def test_a_centre_over_the_whole_record_before_the_last_jump_would_lose_the_episode(end):
    """The control: the same record read with one centre up to the last jump (or the whole window) is non_resonant,
    so the test above fails if the per-regime centres are replaced by either."""
    geometry, fa = two_regime_record(end, second=True)
    tau = geometry.jump_tau_yr
    last = max(jump.time_yr for jump in geometry.jumps if not jump.kick)
    t = np.asarray(fa.t_yr)
    before = t <= last - tau / 2
    one_cut = g.centre_track(
        t[before],
        np.asarray(fa.angle_f)[before],
        PERIOD,
        fa.cutoff_yr,
        geometry.turning_points.prominence_rad,
        (fa.window[0], last - tau / 2),
        S,
    )
    whole_window = g.measure_geometry(fa, ONE_DEG, S).centre  # before `finish_geometry` cuts at the jumps
    (episode,) = geometry.episodes_before_jump
    for centre in (one_cut, whole_window):
        geometry.record_centres = [centre]
        assert decide.centre_travel_in(geometry, episode.t_start_yr, episode.t_end_yr) > 2 * np.pi
        _, decision, _ = decide_all(geometry)
        assert decision.status == Status.NON_RESONANT
