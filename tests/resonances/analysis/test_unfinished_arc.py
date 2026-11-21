"""The candidate arc of an uncertain row (`policy.unfinished_arc`): a whole window without a revolution of sigma_f.

523683 2U-9 over 50 kyr (the author 2026-11-21): sigma_f swings slowly from ~95 to ~269 deg and never crosses 2pi;
the slow cycle (~90 kyr in the 2U-9 well) does not close, so the row stays uncertain / insufficient_time, and the
whole window is shown as one candidate span, tag 'arc'. A record whose sigma_f turns once is no arc.
"""

import math
from types import SimpleNamespace

import numpy as np

from resonances.resonance.analysis import policy


def _geometry(angle_deg, raw_range=0.0):
    t = np.linspace(0.0, 50_000.0, len(angle_deg))
    return SimpleNamespace(
        window=(500.0, 49_500.0),
        angle_f=np.radians(np.asarray(angle_deg, float)),
        angle_f_t_yr=t,
        averaged_raw=np.array([0.0, raw_range]),
        averaged_raw_t_yr=np.array([500.0, 49_500.0]),
    )


def test_a_slow_swing_below_a_turn_is_one_arc_over_the_window():
    swing = 182.0 - 87.0 * np.cos(np.linspace(0.0, math.pi, 2001))  # 95 -> 269 deg, half a slow cycle
    assert policy.unfinished_arc(_geometry(swing)) == (500.0, 49_500.0)


def test_a_turn_of_sigma_or_of_the_averaged_raw_angle_is_no_arc():
    assert policy.unfinished_arc(_geometry(np.linspace(0.0, 400.0, 2001))) is None  # one turn and more of sigma_f
    assert policy.unfinished_arc(_geometry(np.linspace(100.0, 200.0, 2001), raw_range=7.0)) is None  # 7 rad > 2 pi
    assert policy.unfinished_arc(None) is None


def test_sigma_f_comes_from_the_prep_when_the_geometry_has_none():
    geometry = _geometry([100.0, 200.0])
    geometry.angle_f = geometry.angle_f_t_yr = None
    prep = SimpleNamespace(t_yr=np.linspace(0.0, 50_000.0, 11), angle_f=np.radians(np.linspace(100.0, 250.0, 11)))
    assert policy.unfinished_arc(geometry, prep) == (500.0, 49_500.0)
    assert policy.unfinished_arc(geometry) is None
