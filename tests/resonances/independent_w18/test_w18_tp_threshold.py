"""Independent tests of W18 item 2: the turning-point threshold no longer takes the libration's own swing for noise
(docs/libration-analysis.md §3 "The libration is not its own noise"; W17 Q8, 2013 AS27 31E-36+0+5).

p_TP = max(floor, 4 r), r the 95th percentile of |sigma_f - lowpass(sigma, P_f)|. For a sine of period P and
half-amplitude A the filters are Butterworth of order 2 applied forward and backward: gain G(P; P_c) =
1 / (1 + (P_c / P)^4) (the digital correction at a 1-yr step is below 1e-3 here). Before W18 a libration with
P >= P_f = 150 yr left (G(P; 75) - G(P; 150)) A in r: 44 % of A at 150 yr, enough for 4 r to reach its own swing 2A.
With the run's own image taken out, (G(P; 75) - G(P; 150)) (1 - G(P; 75)) A remains. A forced wave faster than P_f is
not a libration and keeps its image. The 95th percentile of |A sin| is A sin(0.95 pi / 2) = 0.99692 A.
"""

import functools

import numpy as np
import pytest

from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.context import table_mean_motion
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings

SETTINGS = AnalysisSettings()
P_CUT, P_FLOOR, SWING_DEG, SPAN = 75.0, 150.0, 25.0, 20000.0
P95 = np.sin(0.95 * np.pi / 2)


def _gain(period, cutoff):
    return 1.0 / (1.0 + (cutoff / period) ** 4)


@functools.lru_cache(maxsize=None)
def _sine(period_yr):
    """3J-1 at 2.5 au whose angle is A sin(2 pi t / P); a follows from sigma_dot."""
    t = np.arange(0.0, SPAN + 0.5, 1.0)
    w = 2 * np.pi / period_yr
    sigma, rate = np.radians(SWING_DEG) * np.sin(w * t), np.radians(SWING_DEG) * w * np.cos(w * t)
    spec = ResonanceSpec.from_key('3J-1')
    ones = np.ones_like(t)
    n_j = table_mean_motion('Jupiter')
    lam_j = n_j * t + 0.3
    lam = (sigma - 3 * lam_j + 2 * 1.0) / -1
    n = (rate - 3 * n_j) / -1
    series = MMRSeries(
        t_yr=t,
        a=(2 * np.pi / n) ** (2.0 / 3.0),
        e=0.15 * ones,
        inc=0.05 * ones,
        Omega=0.4 * ones,
        omega=0.6 * ones,
        varpi=1.0 * ones,
        angle_wrapped=np.mod(3 * lam_j - lam - 2 * 1.0, 2 * np.pi),
        resonance=spec,
        regime=Regime(1.0, SPAN, len(t)),
        planets={
            'Jupiter': PlanetSeries(
                'Jupiter', t, 5.2026 * ones, 0.048 * ones, 0.0228 * ones, 1.753 * ones, 0.257 * ones, np.mod(lam_j, 2 * np.pi)
            )
        },
        mean_longitude=np.mod(lam, 2 * np.pi),
        name='synthetic',
    )
    return analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False)


def _extrema_in(period, window):
    """Extrema of A sin(2 pi t / P): t = P/4 + m P/2, strictly inside the analysis window."""
    times = period / 4 + np.arange(0, int(SPAN / period * 2) + 2) * period / 2
    return int(np.count_nonzero((times > window[0]) & (times < window[1])))


def test_the_cutoffs_are_the_documented_ones():
    analysis = _sine(160.0)
    row = analysis.numbers_row()
    assert row['flt_cutoff_yr'] == pytest.approx(P_CUT)
    assert row['flt_period_floor_yr'] == pytest.approx(P_FLOOR)


@pytest.mark.parametrize('period', [150.0, 155.0, 160.0, 170.0, 180.0])
def test_a_libration_of_150_to_180_yr_keeps_every_turning_point(period):
    analysis = _sine(period)
    tp = analysis.geometry.turning_points
    old = 4 * P95 * (_gain(period, P_CUT) - _gain(period, P_FLOOR)) * SWING_DEG
    new = max(
        SETTINGS.mmr_tp_prominence_floor_deg,
        4 * P95 * (_gain(period, P_CUT) - _gain(period, P_FLOOR)) * (1 - _gain(period, P_CUT)) * SWING_DEG,
    )
    assert old > 0.5 * 2 * SWING_DEG  # what the record's threshold was: at least half the swing's prominence 2A
    assert np.degrees(tp.prominence_rad) == pytest.approx(new, rel=0.1, abs=0.1)
    assert len(tp.times_yr) == _extrema_in(period, analysis.geometry.window)
    accepted = [e for e in analysis.geometry.episodes if e.accepted]
    assert len(accepted) == 1
    assert accepted[0].n_tp == len(tp.times_yr)
    assert str(analysis.verdict.angle_status) == 'libration'


def test_a_forced_wave_of_115_yr_still_raises_the_threshold_above_its_swing():
    """115 yr < P_f: no libration, the image stays: 4 x 0.997 x (0.847 - 0.257) x 25 = 58.8 deg > 2A = 50 deg."""
    analysis = _sine(115.0)
    tp = analysis.geometry.turning_points
    expected = 4 * P95 * (_gain(115.0, P_CUT) - _gain(115.0, P_FLOOR)) * SWING_DEG
    assert expected == pytest.approx(58.8, abs=0.1)
    assert np.degrees(tp.prominence_rad) == pytest.approx(expected, rel=0.03)
    assert expected > 2 * SWING_DEG
    assert len(tp.times_yr) == 0
    assert not [e for e in analysis.geometry.episodes if e.accepted]
    assert str(analysis.verdict.angle_status) != 'libration'
