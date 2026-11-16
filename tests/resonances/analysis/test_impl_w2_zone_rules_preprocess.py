"""W2 zone-rules: the filter guard at the cutoff x2 (DELTA 13a, preprocess.filter_sensitivity).

A circulation carrying a forced wave just slower than P_c: at P_c the wave reverses sigma_dot and makes turning points,
at 2 P_c the filter damps it (Butterworth gain 1/(1 + (P_c/P)^4): 0.83 at P = 1.5 P_c, 0.24 at P = 0.75 x 2 P_c) and
the turning points disappear - the conclusion 'a cycle exists' changes. A libration slower than both keeps its
conclusions.
"""

import numpy as np

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.filtering import edge_window, lowpass, time_derivative
from resonances.resonance.analysis.mmr import preprocess
from resonances.resonance.analysis.mmr.models import Cutoff, Preprocessed
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.impl_response_pendulum_synthetic import make

S = settings_from_config()
ONE_DEG = np.radians(1.0)
CUTOFF = 20.0


def prepared(t, angle):
    angle_f = lowpass(t, angle, CUTOFF, 2)
    return make(
        Preprocessed,
        cutoff=Cutoff(CUTOFF, 'floor', 0.0, CUTOFF, CUTOFF / float(t[1] - t[0])),
        t_yr=t,
        angle_unwrapped=angle,
        angle_clean=angle,
        angle_f=angle_f,
        rate_f=time_derivative(t, angle_f),
        angle_f_half=lowpass(t, angle, CUTOFF / 2, 2),
        angle_f_double=lowpass(t, angle, 2 * CUTOFF, 2),
        a_f=np.zeros_like(t),
        adot_f=np.zeros_like(t),
        e_f=np.zeros_like(t),
        inc_f=np.zeros_like(t),
        window=edge_window(t, CUTOFF),
    )


def plain_angle(prep):
    """The x1 angle of `measured` (no MMR rules)."""
    return g.FilteredAngle(
        t_yr=prep.t_yr,
        angle_raw=prep.angle_unwrapped,
        angle_f=prep.angle_f,
        rate_f=prep.rate_f,
        angle_f_half=prep.angle_f_half,
        angle_f_double=prep.angle_f_double,
        slow_f=prep.a_f,
        identity_coefficient=-30.0,
        cutoff_yr=CUTOFF,
        step_yr=float(prep.t_yr[1] - prep.t_yr[0]),
        window=prep.window,
    )


def measured(prep):
    return g.measure_geometry(plain_angle(prep), ONE_DEG, S)


def test_a_forced_wave_that_the_cutoff_lets_through_is_filter_sensitive():
    t = np.arange(0.0, 6000.0, 0.5)
    angle = 2 * np.pi * t / 2000 + 0.05 * np.sin(2 * np.pi * t / 30.0)  # turns every 2 kyr, a 30-yr forced wave
    prep = prepared(t, angle)
    geometry = measured(prep)
    assert len(geometry.turning_points) >= S.libration_min_turning_points  # the wave reverses sigma_dot at P_c
    result = preprocess.filter_sensitivity(prep, plain_angle(prep), geometry, None, ONE_DEG, S)
    assert result.sensitive and 'turning_points' in result.changed
    assert result.at_double.n_tp < S.libration_min_turning_points
    assert result.at_cutoff.cutoff_yr == CUTOFF and result.at_double.cutoff_yr == 2 * CUTOFF


def test_a_slow_libration_keeps_its_conclusions_at_the_doubled_cutoff():
    t = np.arange(0.0, 6000.0, 0.5)
    angle = np.pi + np.radians(60) * np.sin(2 * np.pi * t / 500)  # P = 25 P_c
    prep = prepared(t, angle)
    geometry = measured(prep)
    result = preprocess.filter_sensitivity(prep, plain_angle(prep), geometry, None, ONE_DEG, S)
    assert not result.sensitive
    assert result.at_cutoff.n_episodes == result.at_double.n_episodes == 1
    assert abs(result.at_cutoff.n_tp - result.at_double.n_tp) <= 2


def test_without_the_doubled_series_nothing_is_compared():
    t = np.arange(0.0, 6000.0, 0.5)
    prep = prepared(t, np.pi + np.radians(60) * np.sin(2 * np.pi * t / 500))
    prep.angle_f_double = np.full_like(t, np.nan)
    result = preprocess.filter_sensitivity(prep, plain_angle(prep), measured(prep), None, ONE_DEG, S)
    assert result.at_double is None and not result.sensitive
