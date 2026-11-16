"""W36 F2: each run of turning points of a secular or ZLK angle is selected at the prominence measured on its own
samples (`secular.preprocess.run_prominence`, the MMR rule `geometry.local_thresholds`).

19289 and 278067 s-s6 (the author 2026-11-07: "starting from about 5 Myr to the end there is one libration episode,
and you split it into pieces"): the record's p_TP = 4 r = 76.8 deg came from the circulation before the capture
(r = 26.8 deg there, 8.0 deg on the libration); the libration's smaller swings fell below it and the gaps cut it.

The synthetic angle (built here, not by the package), secular, step 2 kyr, 20 Myr, P_c = 100 kyr, floor P_min = 200 kyr:
- 0-8 Myr: a circulation of 200 kyr per turn carrying a wave of 40 deg and 150 kyr (its rate stays one-signed: no
  turning points);
- 8-20 Myr: a libration pi + A(t) sin(2 pi t / 250 kyr), A = 40 + 20 cos(2 pi t / 1.5 Myr) deg (20-60 deg).
Expected by hand. The fast part of the wave (filtfilt order 2: gain 1/(1 + (P_c/P)^4)) is (0.835 - 0.240) x 40 =
23.8 deg; it fills 0.4 of the window, so its 95th percentile there is the |sin| level exceeded 1/8 of the time,
23.8 sin(0.875 pi/2) = 23.3 deg, and p_TP = 4 x 23.3 = 93.4 deg. A half-cycle turns by 2A, so the half-cycles of
A < 46.7 deg (cos < 0.34: 61 % of each 1.5-Myr modulation) are lost and the libration falls apart into pieces around
the maxima of A (9.5, 11, ... Myr). On the libration itself the fast part is its own image (taken out) and the floor
binds: one episode from the first turning point after 8 Myr to the last before the window's end (19.9 Myr),
11.9 Myr / 125 kyr = 95 turning points. The ZLK case is the same angle with every time scaled to P_c = 1874 yr.
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis.secular import LongPeriodSeries
from resonances.resonance.analysis.secular import decide as secular_decide
from resonances.resonance.analysis.secular import preprocess
from resonances.resonance.analysis.secular.analyse import analyse_long_period, analyse_secular
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import ResonanceType, Status

SETTINGS = settings_from_config()
OFF = replace(SETTINGS, secular_tp_local_prominence=False, zlk_tp_local_prominence=False)
CAPTURE_YR = 8.0e6
SECULAR = (ResonanceType.SECULAR, 's-s6', 1.0)  # type, name, time scale
ZLK = (ResonanceType.LIDOV_KOZAI, 'LK', 1874.0 / 1.0e5)
RECORD_PROMINENCE_DEG = 93.4  # 4 x 23.3 deg (module docstring)
N_TP = 95  # 11.9 Myr of a 250-kyr libration, two turning points per period


def _series(kind) -> LongPeriodSeries:
    resonance_type, name, scale = kind
    step = 2000.0 * scale
    t = np.arange(0.0, 20.0e6 * scale + step / 2, step)
    since = t - CAPTURE_YR * scale
    sigma = np.empty_like(t)
    lib = since >= 0
    amplitude = np.radians(40.0 + 20.0 * np.cos(2 * np.pi * since[lib] / (1.5e6 * scale)))
    sigma[lib] = np.pi + amplitude * np.sin(2 * np.pi * since[lib] / (250.0e3 * scale))
    before = since[~lib]
    sigma[~lib] = np.pi + 2 * np.pi * before / (200.0e3 * scale) + np.radians(40.0) * np.sin(2 * np.pi * before / (150.0e3 * scale))
    one = np.ones_like(t)
    return LongPeriodSeries(resonance_type, name, 'synthetic', t, sigma, 2.4 * one, 0.1 * one, 0.3 * one)


def _episodes(analysis, scale):
    """(start, end) [Myr of the unscaled record] and the number of TPs of every librating episode."""
    return [(e.t_start_yr / 1e6 / scale, e.t_end_yr / 1e6 / scale, e.n_tp) for e in secular_decide.librating_episodes(analysis.geometry)]


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_the_record_prominence_comes_from_the_circulation(kind):
    analysis = analyse_long_period(_series(kind), SETTINGS)
    assert np.degrees(analysis.cutoff.prominence_rad) == pytest.approx(RECORD_PROMINENCE_DEG, abs=1.5)
    assert analysis.cutoff.prominence_binding == 'ripple'


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_the_libration_is_one_episode_at_its_own_prominence(kind):
    scale = kind[2]
    analysis = analyse_long_period(_series(kind), SETTINGS)
    (start, end, n_tp), *rest = _episodes(analysis, scale)
    assert rest == []
    assert start == pytest.approx(8.0, abs=0.25)  # the first turning point after the capture
    assert end == pytest.approx(19.9, abs=0.15)  # the last one before the window's end (P_c from the end)
    assert n_tp == pytest.approx(N_TP, abs=2)
    # the run's own threshold: the floor binds (no wave faster than P_min on the libration)
    (local,) = analysis.geometry.turning_points.local
    assert np.degrees(local.prominence_rad) == pytest.approx(SETTINGS.secular_tp_prominence_deg, abs=0.5)


@pytest.mark.parametrize('kind', [SECULAR, ZLK], ids=['secular', 'zlk'])
def test_at_the_record_prominence_the_libration_falls_apart(kind):
    """The rule off (as before W36): pieces around the maxima of A, 9.5, 11, ..., 18.5 Myr."""
    scale = kind[2]
    episodes = _episodes(analyse_long_period(_series(kind), OFF), scale)
    assert len(episodes) == 7
    centres = [(start + end) / 2 for start, end, _ in episodes]
    assert centres == pytest.approx([9.5 + 1.5 * k for k in range(7)], abs=0.1)
    assert all(end - start < 0.6 for start, end, _ in episodes)


def test_the_secular_verdict_is_a_transient_that_reaches_the_end():
    verdict = analyse_secular(_series(SECULAR), SETTINGS).verdict
    assert verdict.status == Status.TRANSIENT
    assert verdict.pol_last_reaches_end is True


def test_a_run_never_gets_more_than_the_record_and_a_short_run_keeps_the_record():
    """On the circulation alone the fast part is the whole wave: 23.8 sin(0.95 pi/2) = 23.7 deg, 4 r = 94.9 deg >
    93.4 deg of the record, so the run gets the record's (the cap). A run of 300 kyr has nothing left after its ends
    are trimmed by P_min each: no measurement (NaN, the record's prominence stays)."""
    series = _series(SECULAR)
    cutoff = preprocess.choose_cutoff(series, SETTINGS)
    angle = preprocess.filtered_angle(series, cutoff, SETTINGS)
    local = preprocess.run_prominence(series, cutoff, angle, SETTINGS)
    t = angle.t_yr
    circulation = (t >= 1.0e6) & (t <= 7.0e6)
    part = preprocess._fast_part(
        t, angle.angle_raw, angle.angle_f, tuple(angle.window), cutoff.cutoff_yr, angle.period_floor_yr, 0.0, SETTINGS
    )
    supported = circulation & (t >= 1.2e6) & (t <= 6.8e6)
    uncapped = SETTINGS.secular_tp_ripple_factor * np.degrees(np.percentile(part[supported], 95.0))
    assert uncapped == pytest.approx(94.9, abs=1.5) and uncapped > np.degrees(cutoff.prominence_rad)
    assert local(circulation) == cutoff.prominence_rad
    assert np.isnan(local((t >= 12.0e6) & (t <= 12.3e6)))


def test_the_rule_off_gives_no_local_prominence():
    series = _series(SECULAR)
    cutoff = preprocess.choose_cutoff(series, SETTINGS)
    angle = preprocess.filtered_angle(series, cutoff, SETTINGS)
    assert preprocess.run_prominence(series, cutoff, angle, OFF) is None
