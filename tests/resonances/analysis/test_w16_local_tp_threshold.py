"""W16: each run of turning points is selected at the TP prominence measured on its own stretch.

The record's prominence (4 x the fast residual of sigma_f over the whole record) is inflated by a regime change
elsewhere and then drops the half-cycles of a quiet libration; the gap rule cuts the run at the hole that leaves
(2001 PG14 33S-2+0-31: 21 deg against 2.7 deg on the libration, the episode ended at 700 yr instead of 2.8 kyr).
`geometry.local_thresholds` finds provisional turning points at the floor, splits them into runs by the usual rules
and gives every run of >= 3 TPs the prominence `analyse.regime_prominence` measures on its own samples, once.

Expected values come from elsewhere: the analytic extrema of a synthetic sine, an independent Butterworth low-pass for
the record's prominence, and for 2001 PG14 the author's reading of the figure (libration to about 2.8 kyr) with the
independent what-if of Codex (345-2816 yr, 26 TPs, median period 187.5 yr at a supported prominence of 10.0 deg).
"""

import numpy as np
import pytest
from scipy import signal

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.mmr import models as m
from resonances.resonance.analysis.mmr.analyse import regime_prominence
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.impl_response_pendulum_synthetic import make

S = settings_from_config()

STEP_YR = 1.0
CUTOFF_YR = 5.0
FLOOR_PERIOD_YR = 150.0  # the shortest libration (the MMR default)
LIBRATION_PERIOD_YR = 1200.0  # slow against the floor: no fast residual of its own
LIBRATION_HALF_AMP_RAD = 0.2  # a full swing of 0.4 rad (23 deg)
ESCAPE_YR = 9300.0  # at a minimum of the libration
END_YR = 18000.0
WAVE_PERIOD_YR = 40.0  # a forced wave on the circulation after the escape, faster than the floor
WAVE_RAD = 0.5


def _lowpass(x, period_yr):
    """An independent Butterworth low-pass (order 2, forward-backward) at `period_yr`."""
    sos = signal.butter(2, (1 / period_yr) / (0.5 / STEP_YR), output='sos')
    return signal.sosfiltfilt(sos, x)


def _synthetic():
    """sigma = pi + 0.2 sin(2 pi t / 1200 yr) up to 9.3 kyr, then a circulation of 60 yr per turn carrying a 40-yr wave of
    0.5 rad that never reverses it (rate 2 pi/60 - 0.5 x 2 pi/40 > 0): no turning point after the escape, but a fast
    residual that sets the record's prominence far above the libration's swing."""
    t = np.arange(0.0, END_YR + STEP_YR, STEP_YR)
    libration = np.pi + LIBRATION_HALF_AMP_RAD * np.sin(2 * np.pi * t / LIBRATION_PERIOD_YR)
    after = t - ESCAPE_YR
    start = np.pi - LIBRATION_HALF_AMP_RAD
    circulation = start + 2 * np.pi * after / 60.0 + WAVE_RAD * np.sin(2 * np.pi * after / WAVE_PERIOD_YR)
    sigma = np.where(t < ESCAPE_YR, libration, circulation)
    window = (t[0] + CUTOFF_YR, t[-1] - CUTOFF_YR)
    angle = g.FilteredAngle(
        t_yr=t,
        angle_raw=sigma,
        angle_f=sigma,
        rate_f=np.gradient(sigma, t),
        angle_f_half=sigma,
        angle_f_double=sigma,
        slow_f=np.ones_like(t),
        identity_coefficient=-1.0,
        cutoff_yr=CUTOFF_YR,
        step_yr=STEP_YR,
        window=window,
    )
    inside = (t >= window[0]) & (t <= window[1])
    record = S.mmr_tp_ripple_factor * float(np.percentile(np.abs(sigma - _lowpass(sigma, FLOOR_PERIOD_YR))[inside], 95))
    prep = make(
        m.Preprocessed,
        cutoff=m.Cutoff(CUTOFF_YR, 'floor', 0.0, CUTOFF_YR, CUTOFF_YR / STEP_YR),
        t_yr=t,
        angle_clean=sigma,
        angle_f=sigma,
        window=window,
        unremoved_ripple_rad=0.0,
        period_floor_yr=FLOOR_PERIOD_YR,
    )
    return angle, prep, record


# the extrema of the sine up to the escape (the last one): 300 + 600 k yr
ANALYTIC_TP_YR = 300.0 + 600.0 * np.arange(16)


def test_the_record_s_prominence_loses_the_libration():
    """The defect: the wave after the escape sets a prominence above the libration's full swing."""
    angle, _, record = _synthetic()
    assert record > 2 * LIBRATION_HALF_AMP_RAD
    geometry = g.measure_geometry(angle, record, S)
    assert not geometry.episodes


def test_the_libration_run_gets_its_own_prominence_and_every_turning_point():
    angle, prep, record = _synthetic()
    local = regime_prominence(prep, S)
    geometry = g.measure_geometry(angle, record, S, local)
    (stretch,) = geometry.turning_points.local
    # the provisional run spans the analytic extrema; on it sigma has nothing faster than the floor: the floor itself
    assert stretch.start_yr == ANALYTIC_TP_YR[0] and stretch.end_yr == ANALYTIC_TP_YR[-1]
    assert stretch.prominence_rad == pytest.approx(np.radians(S.mmr_tp_prominence_floor_deg))
    (episode,) = geometry.episodes
    np.testing.assert_array_equal(episode.tp_times_yr, ANALYTIC_TP_YR)
    assert episode.period_med_yr == pytest.approx(LIBRATION_PERIOD_YR)
    assert episode.tp_prominence_rad == stretch.prominence_rad
    # the record's prominence stays the record's (the centre and the CSV column read it)
    assert geometry.turning_points.prominence_rad == record
    # the counts at x0.5 and x2 use the same thresholds (here the same series): they cannot differ by the threshold
    assert geometry.n_tp_cut_half == geometry.n_tp_cut_double == len(geometry.turning_points)


def test_a_run_too_short_to_measure_keeps_the_record_s_prominence():
    """A run of three provisional TPs over less than P_f + 2 x the trim margin has no prominence of its own."""
    angle, prep, record = _synthetic()
    t = angle.t_yr
    span = FLOOR_PERIOD_YR + 2 * S.filter_edge_trim_cutoffs * FLOOR_PERIOD_YR
    period = 0.9 * span  # max, min, max: the three TPs span one period
    start = 1000.0
    phase = 2 * np.pi * (t - start) / period
    quick = np.pi + 0.2 * np.sin(np.clip(phase, -np.pi / 2, 3 * np.pi))  # flat before the first minimum, after 1.5 cycles
    provisional = g.find_turning_points(t, quick, np.radians(S.mmr_tp_prominence_floor_deg), angle.window)
    np.testing.assert_allclose(provisional.times_yr, start + period * np.array([0.25, 0.75, 1.25]), atol=STEP_YR)
    assert g.local_thresholds(t, quick, angle.window, regime_prominence(prep, S), S) == ()
    tp = g.turning_points_with_local(t, quick, record, angle.window, S, regime_prominence(prep, S))
    assert len(tp) == 0 and tp.prominence_rad == record


def test_without_a_local_prominence_nothing_changes():
    angle, _, record = _synthetic()
    plain = g.find_turning_points(angle.t_yr, angle.angle_f, record, angle.window)
    same = g.turning_points_with_local(angle.t_yr, angle.angle_f, record, angle.window, S, None)
    np.testing.assert_array_equal(plain.indices, same.indices)
    assert same.local == ()


# ------------------------------------------------------------------ 2001 PG14 33S-2+0-31 (benchmark)


PG14 = 'mmr2__2001_PG14__33S-2+0-31'


@pytest.fixture(scope='module')
def pg14():
    if not io.example_path(PG14).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(PG14), S)


@pytest.mark.benchmark
def test_pg14_the_episode_runs_to_the_escape(pg14):
    """The author: several cycles after 700 yr; Codex's what-if: 345-2816 yr, 26 TPs, period 187.5 yr. Without the
    libration's own image in the threshold (W18) the quiet half-cycles of 4.5-6 deg at 1.9-2.2 kyr count too: 28 TPs,
    one every ~90 yr there instead of a 276-yr hole (1915 -> 2191 yr), period 180 yr."""
    episodes = [run for run in pg14.geometry.runs if run.accepted]
    (episode,) = episodes
    assert episode.t_start_yr == pytest.approx(345.0, abs=10.0)
    assert episode.t_end_yr == pytest.approx(2816.0, abs=10.0)
    assert episode.n_tp == 28
    assert episode.period_med_yr == pytest.approx(187.5, rel=0.05)
    # selected at a prominence well below the record's, which dropped the 19.6-20.1 deg half-cycles
    assert np.degrees(episode.tp_prominence_rad) < 15.0
    assert pg14.verdict.angle_status == 'episodic'


def test_l1_recounts_an_episode_spanning_two_stretches_at_its_own_profile():
    """An episode may join TPs of two stretches (a provisional run split by the gap rule at the floor, joined again at
    the local thresholds). Its L1 recount must use both thresholds: the largest alone drops the quiet half (here the
    same series at every cutoff, so the counts must agree exactly)."""
    from types import SimpleNamespace

    from resonances.resonance.analysis.mmr.analyse import _episode_tp_counts

    t = np.arange(0.0, 4000.0 + STEP_YR, STEP_YR)
    amplitude = np.where(t < 1950.0, 0.3, 0.05)  # half-swings of 0.3 rad, then (from a zero of the sine) 0.05 rad
    sigma = amplitude * np.sin(2 * np.pi * t / 300.0)
    local = (g.LocalThreshold(0.0, 1950.0, 0.2), g.LocalThreshold(1950.0 + STEP_YR, 4000.0, 0.05))
    window = (t[0], t[-1])
    tp = g.find_turning_points(t, sigma, 0.2, window, local=local)
    (episode,) = g.find_episodes(t, tp, window, S)
    # analytic extrema: 75 + 150 k yr, all of them
    np.testing.assert_array_equal(episode.tp_times_yr, 75.0 + 150.0 * np.arange(26))
    assert episode.tp_prominence_rad == 0.2
    prep = SimpleNamespace(t_yr=t, angle_f_half=sigma, angle_f_double=sigma)
    counts = _episode_tp_counts(prep, SimpleNamespace(runs=[episode]), episode.episode_id)
    assert counts == {'one': 26, 'half': 26, 'double': 26}
    # the largest threshold alone would see only the loud half
    assert len(g.find_turning_points(t, sigma, episode.tp_prominence_rad, (episode.t_start_yr, episode.t_end_yr))) < 26
