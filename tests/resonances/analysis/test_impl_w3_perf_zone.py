"""W3 'perf' package, zone.py: energy cycles inside runs only (W3-G4) and the zone integration's cost.

Codex sol: the W2 pairing (TP_i, TP_i+2) over all turning points of the window joined runs across gaps (52356:
10049-19539 yr; 182223: 92 of 105 records). Expected cycles below are listed by hand.
"""

import math
import time
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import zone
from resonances.resonance.analysis.pendulum import CycleEnergy
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()


def _geometry(times, runs, revolutions=()):
    """What `labelled_cycle_bounds` reads: the window's turning points, the runs (accepted or not), the turns."""
    return SimpleNamespace(
        turning_points=SimpleNamespace(times_yr=np.asarray(times, float)),
        runs=[
            SimpleNamespace(tp_times_yr=np.asarray(r[0], float), accepted=r[1], before_last_jump=r[2] if len(r) > 2 else False)
            for r in runs
        ],
        revolutions=[SimpleNamespace(t_start_yr=a, t_end_yr=b) for a, b in revolutions],
    )


def test_cycles_never_join_two_runs():
    # run A (accepted): 0..4, run B (rejected): 10..13. W2 paired over all: (0,2) (2,4) (4,11) (11,13)
    times = [0, 1, 2, 3, 4, 10, 11, 12, 13]
    geometry = _geometry(times, [([0, 1, 2, 3, 4], True), ([10, 11, 12, 13], False)])
    assert zone.labelled_cycle_bounds(geometry) == [
        (0.0, 2.0, False, 'episode'),
        (2.0, 4.0, False, 'episode'),
        (10.0, 12.0, False, 'diagnostic'),
    ]
    assert zone.cycle_bounds(geometry) == [(0.0, 2.0, False), (2.0, 4.0, False), (10.0, 12.0, False)]


def test_pairing_restarts_at_every_run():
    # odd first run: the second run's pairing starts at its own first turning point, not at the parity of the window
    times = [0, 1, 2, 5, 6, 7, 8]
    geometry = _geometry(times, [([0, 1, 2], True), ([5, 6, 7, 8], True)])
    assert zone.cycle_bounds(geometry) == [(0.0, 2.0, False), (5.0, 7.0, False)]


def test_turns_and_runs_cut_off_by_the_last_regime():
    times = [20, 21, 22]
    geometry = _geometry(times, [([20, 21, 22], True, True)], revolutions=[(0.0, 8.0), (8.0, 16.0), (19.0, 23.0)])
    # the run before the last jump is not an episode of the window; a turn holding turning points is not a turn
    assert zone.labelled_cycle_bounds(geometry) == [
        (0.0, 8.0, True, 'turn'),
        (8.0, 16.0, True, 'turn'),
        (20.0, 22.0, False, 'diagnostic'),
    ]


def test_turning_points_outside_every_run_are_diagnostic():
    geometry = _geometry([0, 1, 2, 3], [])
    assert zone.labelled_cycle_bounds(geometry) == [(0.0, 2.0, False, 'diagnostic')]


def test_cycle_kinds_follow_the_measured_cycles():
    geometry = _geometry([0, 1, 2, 3, 4, 10, 11, 12], [([0, 1, 2, 3, 4], True), ([10, 11, 12], False)], [(30.0, 40.0)])
    cycles = [CycleEnergy(a, b, rot, 0.5, 0.5, b - a, 1.0) for a, b, rot in [(2.0, 4.0, False), (10.0, 12.0, False), (30.0, 40.0, True)]]
    assert zone.cycle_kinds(geometry, cycles) == ['episode', 'diagnostic', 'turn']


@pytest.mark.slow
def test_numeric_zone_of_a_long_record_is_bounded_in_memory_and_time():
    """1e5 samples, 1200 small periods, 16 phases x 77 starts: trajectories kept only as their state and the samples
    of one piece. A pendulum without modulation: nothing reaches the separatrix (J_lo = h_hi = 1)."""
    t = np.linspace(0.0, 2.0e5, 100001)
    omega_bar = 2 * math.pi / 1000.0
    coarse = settings_from_config(mmr_zone_j_step=0.1, mmr_zone_phases=4, mmr_zone_rtol=1e-7)
    tracemalloc.start()
    began = time.perf_counter()
    try:
        j_lo, h_hi = zone.numeric_zone(
            t, lambda x: np.full_like(np.asarray(x, float), omega_bar**2), lambda x: np.zeros_like(x), coarse, np.nan, omega_bar
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (j_lo, h_hi) == (1.0, 1.0)
    assert peak < 200e6
    assert time.perf_counter() - began < 20.0


def test_numeric_zone_finds_the_modulated_separatrix():
    # strength 1 + 0.3 cos(nu t) at nu / Omega_bar = 0.02: J_lo near the adiabatic sqrt(0.7) = 0.837
    t = np.linspace(0.0, 30000.0, 30001)
    omega_bar = 2 * math.pi / 100.0
    nu = 0.02 * omega_bar
    coarse = settings_from_config(mmr_zone_j_step=0.025, mmr_zone_phases=4, mmr_zone_rtol=1e-8)
    j_lo, _ = zone.numeric_zone(
        t,
        lambda x: omega_bar**2 * (1 + 0.3 * np.cos(nu * np.asarray(x, float))),
        lambda x: np.zeros_like(x),
        coarse,
        2 * math.pi / nu,
        omega_bar,
    )
    assert j_lo == pytest.approx(math.sqrt(0.7), abs=0.06)


def test_no_zone_for_a_pendulum_faster_than_the_series():
    """46162 6V-41: a degenerate V_E (R^2 ~ 0) gave Omega_bar = 1.1e6 rad/yr, a 6e-6-yr period over 50 kyr: the zone
    integration never ended. A small period shorter than the smoothing of sigma (here 20 yr) gives no zone."""
    t = np.linspace(0.0, 5000.0, 1001)
    modulation = zone.Modulation(zone.SOURCE_ECCENTRICITY, t, np.ones_like(t), np.zeros_like(t))
    window = np.ones_like(t, dtype=bool)
    began = time.perf_counter()
    assert zone._zone_of(modulation, window, 1.1e6, 0.01, S, 20.0)[0] is None
    assert time.perf_counter() - began < 1.0
    # 2 pi / 0.3 = 20.9 yr > 20 yr: measured (a fixed pendulum: nothing reaches the separatrix)
    coarse = settings_from_config(mmr_zone_j_step=0.1, mmr_zone_phases=2, mmr_zone_rtol=1e-6, mmr_zone_method='numeric')
    found, _ = zone._zone_of(modulation, window, 0.3, 0.01, coarse, 20.0)
    assert found is not None and found.j_lo == 1.0
