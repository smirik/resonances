"""W36 F3: a run of a secular or ZLK angle rejected only for its moving centre is a libration with a modulated centre
when it closes a cycle, stays within one turn, its centre turns back at least once and one of its cycles swings over
every centre it has (`geometry.accept_turning_centre`, SECULAR_TURNING_CENTRE, ZLK_TURNING_CENTRE).

42946 g-g6+s-s6 (the author 2026-11-07: the episode is identified wrongly; the early libration 0-14.6 Myr was missing):
17 TPs, P 1.6 Myr, its cycle centres between 116 and 228 deg turning back twice, below the MMR count of three.

Every number below is computed by hand from the turning points (cycle centre = the mean of two neighbouring
half-cycle midpoints, `geometry.cycle_centres`) or from the formula of the synthetic angle, not by the package.
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.secular import LongPeriodSeries
from resonances.resonance.analysis.secular import preprocess
from resonances.resonance.analysis.secular.analyse import analyse_long_period, analyse_secular
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, ResonanceType, Status
from tests.resonances.analysis.builders import make_turning_points

SETTINGS = settings_from_config()
OFF = replace(SETTINGS, secular_turning_centre=False, zlk_turning_centre=False)

# cycle centres 0, 10, 15, 20, 40, 55, 70, 70, 62.5, 62.5, 57.5 deg (one turn back, at 70 deg, by 12.5 deg >= p_c = 10
# deg), half-amplitudes 20, 30, 45, 60, 60, 55, 50, 30, 17.5, 12.5, 12.5 deg: the band of the cycle about 70 deg
# (40-100) misses the first (-20-20) by 20 deg, the bands of the cycles about 62.5 deg (50-75) by 30 deg (band margin
# -30 deg); the cycle about 40 deg (-20-100) holds every centre; TPs from -40 to 120 deg
TURNING = [-20, 20, -20, 60, -40, 100, 0, 120, 40, 80, 50, 70, 40]
# the same with a wider middle: cycle centres 0, 10, 0, 0, 7.5, 12.5, 77.5, 92.5, 62.5, 62.5, 57.5 deg (two turns
# back), the cycle of half-amplitude 162.5 deg holds them all, but the TPs span -155 to 210 deg: 365 deg
OVER_A_TURN = [-20, 20, -20, 60, -100, 140, -155, 210, 40, 80, 50, 70, 40]
# a wave on a slower motion (363 g-g6+s-s6 at 1.1-3.6 Myr): cycle centres 12.5 ... 35 ... 22.5 deg (one turn back),
# half-amplitudes 7.5 deg (5 at the turn): no band 15 deg wide holds a centre track 22.5 deg long
WAVE = [0, 20, 10, 30, 20, 40, 30, 40, 20, 30, 10]
# a drift (101703 g-2g6+g7, the run at 13.4-19.0 Myr, centre -231 deg, no return): centres 22.5, 27.5, ..., 62.5 deg
DRIFT = [0, 40, 10, 50, 20, 60, 30, 70, 40, 80, 50]


def _run(values_deg):
    values = np.radians(np.asarray(values_deg, dtype=float))
    times = np.arange(len(values)) * 1.0e5
    tp = make_turning_points(times, values, np.radians(1.0))
    return g._classify_run(0, tp, np.arange(len(values)), (0.0, float(times[-1])), SETTINGS)


def test_the_cycle_band_holds_every_centre():
    assert g.swing_covers_centres(np.radians([0.0, 40.0, 70.0]), np.radians([20.0, 60.0, 30.0])) is True  # -20..100
    assert g.swing_covers_centres(np.radians([0.0, 40.0, 70.0]), np.radians([20.0, 35.0, 30.0])) is False  # 5..75
    assert g.swing_covers_centres(np.zeros(0), np.zeros(0)) is False


def test_a_bounded_run_whose_centre_turns_back_is_accepted():
    run = _run(TURNING)
    assert run.reject_reason == g.RejectReason.CENTRE_MOVES  # the MMR test: the bands miss and one return < 3
    assert run.centre_returns == 1
    assert np.degrees(run.margin_sigma_rad) == pytest.approx(-30.0)
    accepted = g.accept_turning_centre(run, SETTINGS)
    assert accepted.accepted and accepted.reject_reason == g.RejectReason.NONE
    assert accepted.shape == g.EpisodeShape.MODULATED
    assert accepted.centre_stationarity == 'not_measured'
    assert accepted.frame == g.FRAME_SIGMA
    assert accepted.n_tp == len(TURNING) and accepted.closed_cycles == run.closed_cycles


@pytest.mark.parametrize(
    'values, returns',
    [(OVER_A_TURN, 2), (WAVE, 1), (DRIFT, 0)],
    ids=['over_a_turn', 'wave_on_a_slow_motion', 'drift'],
)
def test_what_stays_rejected(values, returns):
    run = _run(values)
    assert run.reject_reason == g.RejectReason.CENTRE_MOVES
    assert run.centre_returns == returns
    assert g.accept_turning_centre(run, SETTINGS) is run


def test_an_open_run_or_another_reason_stays_as_it_is():
    run = _run(TURNING)
    assert g.accept_turning_centre(replace(run, closed_cycles=0), SETTINGS).accepted is False
    too_few = replace(run, reject_reason=g.RejectReason.TOO_FEW_TP)
    assert g.accept_turning_centre(too_few, SETTINGS) is too_few


# ---- the whole path on a synthetic secular angle (and the same, time-scaled, for omega) ----
# sigma = pi + 60 deg sin(2 pi t / 16 Myr) + A(t) sin(2 pi t / 1.6 Myr), A = 40 + 40 cos^2(2 pi t / 16 Myr) deg, 20 Myr:
# the centre turns back at 4 and 12 Myr (two returns), spans 120-240 deg; the cycles at the extremes (A = 40 deg:
# 200-280 and 80-160 deg) share no point, the one about 180 deg (A = 80 deg: 100-260) holds every centre; the TPs stay
# within 80-280 deg. A 0.8-Myr half-period over the window 0.1-19.9 Myr (P_c = 100 kyr): 25 TPs.
SECULAR = (ResonanceType.SECULAR, 's-s6', 1.0)
ZLK = (ResonanceType.LIDOV_KOZAI, 'LK', 1874.0 / 1.0e5)  # P_c = 1874 yr for omega: every time scaled


def _series(kind, drift: bool = False) -> LongPeriodSeries:
    resonance_type, name, scale = kind
    t = np.arange(0.0, 20.0e6 * scale + 1000.0 * scale, 2000.0 * scale)
    slow = 2 * np.pi * t / (16.0e6 * scale)
    if drift:  # the centre moves 100 deg one way at a constant half-amplitude of 40 deg
        centre, amplitude = np.pi + np.radians(100.0) * t / (20.0e6 * scale), np.radians(40.0) * np.ones_like(t)
    else:
        centre, amplitude = np.pi + np.radians(60.0) * np.sin(slow), np.radians(40.0 + 40.0 * np.cos(slow) ** 2)
    sigma = centre + amplitude * np.sin(2 * np.pi * t / (1.6e6 * scale))
    one = np.ones_like(t)
    return LongPeriodSeries(resonance_type, name, 'synthetic', t, sigma, 2.4 * one, 0.1 * one, 0.3 * one)


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_the_modulated_libration_is_one_episode(kind):
    (run,) = analyse_long_period(_series(kind), SETTINGS).geometry.runs
    assert run.accepted and run.shape == g.EpisodeShape.MODULATED
    assert run.n_tp == 25 and run.centre_returns == 2
    _, mids, amps = g.cycle_centres(run.tp_times_yr, run.tp_values_rad)
    assert np.degrees(mids.min()) == pytest.approx(120.0, abs=3.0) and np.degrees(mids.max()) == pytest.approx(240.0, abs=3.0)
    assert np.degrees(np.ptp(run.tp_values_rad)) < 360.0
    (off,) = analyse_long_period(_series(kind), OFF).geometry.runs
    assert off.reject_reason == g.RejectReason.CENTRE_MOVES


def test_the_secular_verdict():
    verdict = analyse_secular(_series(SECULAR), SETTINGS).verdict
    assert verdict.status == Status.RESONANT and verdict.angle_status == AngleStatus.LIBRATION
    off = analyse_secular(_series(SECULAR), OFF).verdict
    assert off.status == Status.UNCERTAIN and off.angle_status == AngleStatus.INSUFFICIENT_TIME


def test_a_drifting_centre_stays_a_drift():
    analysis = analyse_long_period(_series(SECULAR, drift=True), SETTINGS)
    (run,) = analysis.geometry.runs
    assert run.reject_reason == g.RejectReason.CENTRE_MOVES and run.centre_returns == 0
    assert analyse_secular(_series(SECULAR, drift=True), SETTINGS).verdict.angle_status == AngleStatus.DRIFT


def test_only_the_secular_preprocessing_sets_the_rule():
    """An MMR `FilteredAngle` keeps the default (off); the secular and ZLK ones follow their own settings."""
    assert g.FilteredAngle.__dataclass_fields__['turning_centre'].default is False
    for kind, key in ((SECULAR, 'secular_turning_centre'), (ZLK, 'zlk_turning_centre')):
        series = _series(kind)
        cutoff = preprocess.choose_cutoff(series, SETTINGS)
        assert preprocess.filtered_angle(series, cutoff, SETTINGS).turning_centre is True
        assert preprocess.filtered_angle(series, cutoff, replace(SETTINGS, **{key: False})).turning_centre is False
