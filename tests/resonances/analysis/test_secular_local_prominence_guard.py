"""W36 F2 guard: a run's own turning-point prominence (`secular.preprocess.run_prominence`) may add libration, never
remove an episode the record's prominence accepts (`secular.analyse.measure_fine`).

455327 g-s-g5+s6 (ias15, 20 Myr): at the record's 39.0 deg the crescent episode 10.23-12.47 Myr (5 TPs) is accepted;
the run of provisional turning points also spans the slow approach before it, and at the run's 13.8 deg a 32-deg
wiggle of that approach (9.14, 9.49 Myr) joined it: 9.14-12.47 Myr, 7 TPs, rejected centre_moves.

The synthetic angle (built here, not by the package), secular, step 2 kyr, 20 Myr, P_c = 100 kyr, P_min = 200 kyr:
- 0-8 Myr: a circulation of 200 kyr per turn downwards with a wave of 40 deg and 150 kyr (as in
  test_secular_local_prominence: its rate stays one-signed, and it sets the record's p_TP = 93.4 deg);
- 8-10 Myr: the approach, the centre drifting down by D = 180 deg at a constant rate with a wiggle of 20 deg and
  250 kyr;
- 10-20 Myr: a libration pi + 55 deg sin(2 pi (t - 10 Myr) / 250 kyr), continuous with the approach.
Expected by hand:
- the approach's half-cycles turn by 2 x 20 deg -+ the drift over half a period, 180 x 125 / 2000 = 11.25 deg: 28.75 and
  51.25 deg, above the 5-deg floor and below the record's 93.4 deg; the libration's by 110 deg, above both;
- the wiggle (250 kyr) is slower than P_min: nothing fast on the run, its own prominence is the floor (5 deg);
- at the record's prominence the first turning point is the minimum at 10.1875 Myr: the maximum before it (10.0625)
  rises from the approach's last minimum at pi + 5.625 - 20 deg (the centre 180 x 62.5 / 2000 deg above pi), by
  69.4 deg < 93.4. The last is the maximum at 19.8125 Myr (the window ends at 20 Myr - P_c; the next extremum, 19.9375,
  is beyond it). One episode, (19.8125 - 10.1875) / 0.125 + 1 = 78 turning points;
- at the run's prominence alone the run starts at the approach (8.0 Myr), and its centre drifts 180 deg: centre_moves;
- with the guard: the stretch keeps the record's prominence, the 78-TP episode stays and the angle is a transient.
The ZLK case is the same angle with every time scaled to P_c = 1874 yr.
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis.geometry import local_thresholds, measure_geometry
from resonances.resonance.analysis.secular import LongPeriodSeries
from resonances.resonance.analysis.secular import decide as secular_decide
from resonances.resonance.analysis.secular import preprocess
from resonances.resonance.analysis.secular.analyse import analyse_long_period, analyse_secular, measure_fine
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import ResonanceType, Status

SETTINGS = settings_from_config()
OFF = replace(SETTINGS, secular_tp_local_prominence=False, zlk_tp_local_prominence=False)
SECULAR = (ResonanceType.SECULAR, 's-s6', 1.0)  # type, name, time scale
ZLK = (ResonanceType.LIDOV_KOZAI, 'LK', 1874.0 / 1.0e5)
APPROACH_YR, CAPTURE_YR = 8.0e6, 10.0e6
DRIFT_DEG, WIGGLE_DEG, AMPLITUDE_DEG, PERIOD_YR = 180.0, 20.0, 55.0, 250.0e3
RECORD_PROMINENCE_DEG = 93.4  # test_secular_local_prominence: the same circulation over the same 0.4 of the window
FIRST_TP_MYR, LAST_TP_MYR, N_TP = 10.1875, 19.8125, 78


def _series(kind) -> LongPeriodSeries:
    resonance_type, name, scale = kind
    step = 2000.0 * scale
    t = np.arange(0.0, 20.0e6 * scale + step / 2, step)
    capture, approach, period = CAPTURE_YR * scale, APPROACH_YR * scale, PERIOD_YR * scale
    sigma = np.empty_like(t)
    lib = t >= capture
    sigma[lib] = np.pi + np.radians(AMPLITUDE_DEG) * np.sin(2 * np.pi * (t[lib] - capture) / period)
    drift = (t >= approach) & ~lib
    centre = np.pi + np.radians(DRIFT_DEG) * (capture - t[drift]) / (capture - approach)
    sigma[drift] = centre + np.radians(WIGGLE_DEG) * np.sin(2 * np.pi * (t[drift] - capture) / period)
    before = t[t < approach] - approach
    sigma[t < approach] = (
        np.pi
        + np.radians(DRIFT_DEG)
        - 2 * np.pi * before / (200.0e3 * scale)
        + np.radians(40.0) * np.sin(2 * np.pi * before / (150.0e3 * scale))
    )
    one = np.ones_like(t)
    return LongPeriodSeries(resonance_type, name, 'synthetic', t, sigma, 2.4 * one, 0.1 * one, 0.3 * one)


def _prepared(kind):
    series = _series(kind)
    cutoff = preprocess.choose_cutoff(series, SETTINGS)
    return series, cutoff, preprocess.filtered_angle(series, cutoff, SETTINGS)


def _runs(geometry, scale):
    """(start, end) [Myr of the unscaled record], number of TPs, accepted, reject reason of every run."""
    return [(r.t_start_yr / 1e6 / scale, r.t_end_yr / 1e6 / scale, r.n_tp, r.accepted, str(r.reject_reason)) for r in geometry.runs]


def _assert_the_episode(geometry, scale):
    (start, end, n_tp, accepted, _), *rest = _runs(geometry, scale)
    assert rest == []
    assert accepted
    assert start == pytest.approx(FIRST_TP_MYR, abs=0.01)
    assert end == pytest.approx(LAST_TP_MYR, abs=0.01)
    assert n_tp == N_TP


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_the_run_prominence_reaches_the_approach(kind):
    """One run of provisional turning points spans the approach and the libration, at the floor (nothing fast on it)."""
    series, cutoff, angle = _prepared(kind)
    assert np.degrees(cutoff.prominence_rad) == pytest.approx(RECORD_PROMINENCE_DEG, abs=1.5)
    local = preprocess.run_prominence(series, cutoff, angle, SETTINGS)
    (stretch,) = local_thresholds(angle.t_yr, np.asarray(angle.angle_f), tuple(angle.window), local, SETTINGS)
    assert stretch.start_yr / kind[2] / 1e6 == pytest.approx(APPROACH_YR / 1e6, abs=0.1)
    assert np.degrees(stretch.prominence_rad) == pytest.approx(SETTINGS.secular_tp_prominence_deg, abs=0.5)


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_the_record_prominence_accepts_the_libration(kind):
    _, cutoff, angle = _prepared(kind)
    _assert_the_episode(measure_geometry(angle, cutoff.prominence_rad, SETTINGS), kind[2])


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_the_run_prominence_alone_loses_it_to_the_drift(kind):
    """The mechanism (F2 before its guard): the wiggles of the approach become turning points, the run starts there and
    its centre moves."""
    series, cutoff, angle = _prepared(kind)
    local = preprocess.run_prominence(series, cutoff, angle, SETTINGS)
    (start, _, n_tp, accepted, reason), *rest = _runs(measure_geometry(angle, cutoff.prominence_rad, SETTINGS, local), kind[2])
    assert rest == []
    assert start == pytest.approx(APPROACH_YR / 1e6, abs=0.1)
    assert n_tp > N_TP + 2 * (CAPTURE_YR - APPROACH_YR) / PERIOD_YR  # the approach's 16 turns joined the libration's
    assert not accepted and reason == 'centre_moves'


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_the_guard_keeps_the_episode(kind):
    series, cutoff, angle = _prepared(kind)
    geometry = measure_fine(series, cutoff, angle, SETTINGS)
    _assert_the_episode(geometry, kind[2])
    assert geometry.turning_points.local == ()  # the stretch fell back to the record's prominence
    analysis = analyse_long_period(series, SETTINGS)
    episodes = secular_decide.librating_episodes(analysis.geometry)
    assert [(e.t_start_yr, e.t_end_yr) for e in episodes] == [(r.t_start_yr, r.t_end_yr) for r in geometry.runs]


def test_the_secular_verdict_is_the_transient_of_the_record_reading():
    on, off = analyse_secular(_series(SECULAR), SETTINGS).verdict, analyse_secular(_series(SECULAR), OFF).verdict
    assert on.status == off.status == Status.TRANSIENT
    assert on.confidence == off.confidence
