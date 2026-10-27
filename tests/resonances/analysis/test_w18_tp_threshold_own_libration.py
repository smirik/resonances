"""W18: the fast residual no longer counts the libration itself (`analyse.fast_part`, `analyse.libration_image`).

The residual sigma_f - lowpass(sigma, P_floor) splits softly at P_floor: a libration of period P >= P_floor keeps
G(P; P_c) - G(P; P_floor) of its swing there (0.44 at 150 yr with P_c = 75 yr), and the TP prominence 4 x r reached
its own swing (2013 AS27 31E-36+0+5: 48.9 deg, 107 of 260 TPs in 19 runs). The image of each libration run through
the same two filters is taken out first; a wave faster than P_floor is no libration run and still raises the threshold.

Expected values come from elsewhere: the gains of an independent scipy Butterworth (sosfreqz), the analytic extrema of
a sine, and for 2013 AS27 the count of Codex (w17 q8-as27: 260 TPs in one run at the 1 deg floor, 95-19890 yr) and
the agreed label of the 4E+9S-5 row (non_resonant, induced by 31:36).
"""

import numpy as np
import pytest
from scipy import signal

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import models as m
from resonances.resonance.analysis.mmr.analyse import _prominence, fast_residual, libration_image, regime_prominence
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.impl_response_pendulum_synthetic import make

S = settings_from_config()

STEP_YR = 1.0
CUTOFF_YR = 75.0  # P_floor / 2, the MMR cutoff where the floor sets it
FLOOR_YR = 150.0  # mmr_libration_period_min_yr
END_YR = 20000.0
HALF_AMP_DEG = 25.0


def _lowpass(x, period_yr, step=STEP_YR):
    """An independent Butterworth (order 2, forward-backward, linear trend removed and put back)."""
    t = np.arange(len(x)) * step
    trend = np.polyval(np.polyfit(t, x, 1), t)
    sos = signal.butter(2, 1 / period_yr, fs=1 / step, output='sos')
    return signal.sosfiltfilt(sos, x - trend) + trend


def _gain(period_yr, cutoff_yr, step=STEP_YR):
    """|H|^2 of that Butterworth at `period_yr` (forward-backward): scipy's frequency response, not the package's."""
    sos = signal.butter(2, 1 / cutoff_yr, fs=1 / step, output='sos')
    _, h = signal.sosfreqz(sos, worN=[1 / period_yr], fs=1 / step)
    return float(abs(h[0]) ** 2)


def _record(sigma, step=STEP_YR, window=None):
    t = np.arange(len(sigma)) * step
    angle_f = _lowpass(sigma, CUTOFF_YR, step)
    window = (t[0] + CUTOFF_YR, t[-1] - CUTOFF_YR) if window is None else window
    angle = g.FilteredAngle(
        t_yr=t,
        angle_raw=sigma,
        angle_f=angle_f,
        rate_f=np.gradient(angle_f, t),
        angle_f_half=angle_f,
        angle_f_double=angle_f,
        slow_f=np.ones_like(t),
        identity_coefficient=-1.0,
        cutoff_yr=CUTOFF_YR,
        step_yr=step,
        window=window,
    )
    prep = make(
        m.Preprocessed,
        cutoff=m.Cutoff(CUTOFF_YR, 'floor', 0.0, CUTOFF_YR, CUTOFF_YR / step),
        t_yr=t,
        angle_clean=sigma,
        angle_f=angle_f,
        window=window,
        unremoved_ripple_rad=0.0,
        period_floor_yr=FLOOR_YR,
    )
    inside = (t >= window[0]) & (t <= window[1])
    # the rule before W18, computed independently: 4 x the 95th percentile of |sigma_f - lowpass(sigma, P_floor)|
    old = S.mmr_tp_ripple_factor * float(np.percentile(np.abs(angle_f - _lowpass(sigma, FLOOR_YR, step))[inside], 95))
    return angle, prep, old


def _geometry(angle, prep):
    return g.measure_geometry(angle, _prominence(prep, S), S, regime_prominence(prep, S))


def _sine(period_yr, half_amp_deg=HALF_AMP_DEG, modulation_deg=0.0, step=STEP_YR):
    t = np.arange(0.0, END_YR + step, step)
    amplitude = np.radians(half_amp_deg + modulation_deg * np.sin(2 * np.pi * t / 7000.0))
    return np.pi + amplitude * np.sin(2 * np.pi * t / period_yr)


def _analytic_extrema(period_yr, window):
    """Extrema of sin(2 pi t / P) inside the window: t = P/4 + k P/2."""
    times = period_yr / 4 + period_yr / 2 * np.arange(int(END_YR / period_yr * 2) + 2)
    return times[(times >= window[0]) & (times <= window[1])]


@pytest.mark.parametrize('period_yr', [150.0, 160.0, 180.0])
def test_a_libration_of_150_to_180_yr_keeps_every_turning_point(period_yr):
    angle, prep, old = _record(_sine(period_yr))
    geometry = _geometry(angle, prep)
    tp = geometry.turning_points
    np.testing.assert_allclose(tp.times_yr, _analytic_extrema(period_yr, angle.window), atol=STEP_YR)
    assert len(geometry.runs) == 1
    # the leak that stays: (G(P; P_c) - G(P; P_floor)) (1 - G(P; P_c)) of the swing, far below the old one
    leak_old = _gain(period_yr, CUTOFF_YR) - _gain(period_yr, FLOOR_YR)
    leak_new = leak_old * (1 - _gain(period_yr, CUTOFF_YR))
    assert np.degrees(old) == pytest.approx(S.mmr_tp_ripple_factor * leak_old * HALF_AMP_DEG, rel=0.02)
    (stretch,) = tp.local
    assert np.degrees(stretch.prominence_rad) < 2 * S.mmr_tp_ripple_factor * leak_new * HALF_AMP_DEG + S.mmr_tp_prominence_floor_deg


@pytest.mark.parametrize('period_yr', [152.0, 160.0])
def test_a_modulated_libration_lost_half_of_its_turning_points_and_keeps_them_now(period_yr):
    """Half-amplitude 21-29 deg (2013 AS27: 20-28 deg): 4 x the old residual of the loud cycles passes the swing of the
    quiet ones."""
    sigma = _sine(period_yr, modulation_deg=4.0)
    angle, prep, old = _record(sigma)
    swing_min = 2 * np.radians(HALF_AMP_DEG - 4.0) * _gain(period_yr, CUTOFF_YR)
    assert old > swing_min  # the defect, from the independent rule
    every = g.find_turning_points(angle.t_yr, angle.angle_f, np.radians(S.mmr_tp_prominence_floor_deg), angle.window)
    geometry = _geometry(angle, prep)
    np.testing.assert_array_equal(geometry.turning_points.indices, every.indices)
    assert len(geometry.runs) == 1 and geometry.runs[0].accepted


@pytest.mark.parametrize('period_yr', [151.0, 155.0])
@pytest.mark.parametrize('step', [5.0012222, 5.1])
def test_a_modulated_libration_keeps_its_turning_points_on_a_coarse_grid(period_yr, step):
    """On a coarse grid the full periods t_i+2 - t_i are whole numbers of steps (a 150-yr libration at 5.1 yr: median
    147.9 yr, Codex W18); the vertex times of the turns measure the period itself. P = P_floor exactly stays a coin
    toss (149.998 yr against 150): the floor is a convention, not a measured boundary."""
    angle, prep, old = _record(_sine(period_yr, modulation_deg=4.0, step=step), step)
    every = g.find_turning_points(angle.t_yr, angle.angle_f, np.radians(S.mmr_tp_prominence_floor_deg), angle.window)
    geometry = _geometry(angle, prep)
    np.testing.assert_array_equal(geometry.turning_points.indices, every.indices)
    assert old > 2 * np.radians(HALF_AMP_DEG - 4.0) * _gain(period_yr, CUTOFF_YR, step)


@pytest.mark.parametrize('period_yr', [145.0, 147.0, 148.0])
def test_a_wave_just_below_the_floor_keeps_its_image_on_a_coarse_grid(period_yr):
    """On the 5.0012-yr grid of the NEA runs the full periods of a 148-yr sine round to 150.04 yr: the vertex times keep
    it below the floor, nothing is taken out, the threshold is the old one."""
    step = 5.0012222
    angle, prep, old = _record(_sine(period_yr, step=step), step)
    assert _prominence(prep, S) == pytest.approx(old, rel=1e-6)


def test_a_run_of_two_turning_points_has_no_period_and_keeps_its_image():
    """libration_min_turning_points = 2 is allowed; two TPs give no full period, so no run of them is a libration."""
    settings = settings_from_config(libration_min_turning_points=2)
    t = np.arange(0.0, 131.0, 1.0)
    sigma = np.pi + np.radians(HALF_AMP_DEG) * np.sin(2 * np.pi * t / 115.0)
    _, prep, _ = _record(sigma, window=(0.0, 130.0))
    assert len(g.find_turning_points(t, prep.angle_f, np.radians(1.0), prep.window)) == 2
    assert not np.any(libration_image(prep, FLOOR_YR, settings))


@pytest.mark.parametrize('half_amp_deg', [20.0, 25.0])
def test_a_115_yr_wave_still_raises_the_threshold(half_amp_deg):
    """Faster than P_floor: no libration run, nothing is taken out; the threshold is the old one, above the wave's swing."""
    angle, prep, old = _record(_sine(115.0, half_amp_deg))
    leak = _gain(115.0, CUTOFF_YR) - _gain(115.0, FLOOR_YR)
    assert np.degrees(fast_residual(prep, S)) == pytest.approx(leak * half_amp_deg, rel=0.02)
    assert _prominence(prep, S) == pytest.approx(old, rel=1e-6)
    assert old > 2 * np.radians(half_amp_deg) * _gain(115.0, CUTOFF_YR)
    geometry = _geometry(angle, prep)
    assert len(geometry.turning_points) == 0


def test_a_115_yr_wave_on_a_libration_stays_in_the_residual():
    """Only the run's own oscillation is taken out: a 6-deg 115-yr wave on a 160-yr libration keeps part of its image
    in the residual (F1 of w17 dropped it with the libration), and its turns, none of its own here, add no TP."""
    t = np.arange(0.0, END_YR + STEP_YR, STEP_YR)
    libration = _sine(160.0)
    wave = np.radians(6.0) * np.sin(2 * np.pi * t / 115.0 + 0.3)
    angle, prep, _ = _record(libration + wave)
    pure = _record(libration)[1]
    assert fast_residual(prep, S) > 2 * fast_residual(pure, S)
    geometry = _geometry(angle, prep)
    # the wave shifts the turns but makes none of its own: one per analytic extremum of the libration
    assert len(geometry.turning_points) == len(_analytic_extrema(160.0, angle.window))
    assert len(geometry.runs) == 1


def _amplitude_at(t, y, period_yr, other_yr, keep):
    """Least-squares half-amplitude [deg] of the line at `period_yr` in y (with the line at `other_yr` and a constant)."""
    columns = [f(2 * np.pi * t[keep] / p) for p in (period_yr, other_yr) for f in (np.sin, np.cos)]
    coefficients = np.linalg.lstsq(np.c_[np.column_stack(columns), np.ones(keep.sum())], y[keep], rcond=None)[0]
    return float(np.degrees(np.hypot(coefficients[0], coefficients[1])))


@pytest.mark.parametrize('wave_deg', [3.0, 6.0])
@pytest.mark.parametrize('phase', [0.3, 1.5])
def test_a_riding_wave_is_taken_out_only_in_part(wave_deg, phase):
    """The curve through the turns passes through the shifts the wave gives them: about 70 % of a 115-yr wave without
    turns of its own leaves the residual with the libration (Codex W18: 2.48 of 3.54 deg for 6 deg), the rest stays."""
    t = np.arange(0.0, END_YR + STEP_YR, STEP_YR)
    sigma = _sine(160.0) + np.radians(wave_deg) * np.sin(2 * np.pi * t / 115.0 + phase)
    _, prep, _ = _record(sigma)
    residual = prep.angle_f - _lowpass(sigma, FLOOR_YR)
    keep = (t > 1000.0) & (t < 19000.0)
    image = wave_deg * (_gain(115.0, CUTOFF_YR) - _gain(115.0, FLOOR_YR))
    assert _amplitude_at(t, residual, 115.0, 160.0, keep) == pytest.approx(image, rel=0.02)
    left = _amplitude_at(t, residual - libration_image(prep, FLOOR_YR, S), 115.0, 160.0, keep)
    assert 0.2 * image < left < 0.4 * image


# ------------------------------------------------------------------ 2013 AS27 31E-36+0+5 (benchmark)


AS27 = 'mmr3__2013_AS27__4E+9S-5+0+0-8'


@pytest.fixture(scope='module')
def as27():
    from dataclasses import replace

    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(AS27).exists():
        pytest.skip('benchmark data not available')
    catalogue = io.load_series(AS27)
    earth = catalogue.planets['Earth']
    # 31 lambda_E - 36 lambda + 5 varpi on the same series (planets of bench2-R-ias15-19.9999kyr-5.00122yr-jd2461297.5)
    angle = 31 * np.asarray(earth.mean_longitude) - 36 * np.asarray(catalogue.mean_longitude) + 5 * np.asarray(catalogue.varpi)
    series = replace(
        catalogue,
        resonance=m.ResonanceSpec.from_key('31E-36+0+5'),
        angle_wrapped=np.mod(angle, 2 * np.pi),
        angle_unwrapped_numpy=np.unwrap(angle),
    )
    return analyse_mmr(series, S), analyse_mmr(catalogue, S)


def test_as27_31_36_is_one_run_of_260_turning_points(as27):
    analysis, _ = as27
    tp = analysis.geometry.turning_points
    assert len(tp) == 260
    assert len(analysis.geometry.runs) == 1
    (stretch,) = tp.local
    assert stretch.start_yr == pytest.approx(95.0, abs=6.0) and stretch.end_yr == pytest.approx(19890.0, abs=6.0)
    # below the smallest swing of the record (half-amplitudes 20-28 deg): no cycle can fall under it
    assert np.degrees(stretch.prominence_rad) < 2 * 20.0 / 4


def test_as27_4e_9s_5_is_induced_by_31_36(as27):
    """The three-body angle repeats 31:36 with factor 5/36 (identity to 1e-9 deg, w17 q8): its 150-yr turns are the
    owner's libration, no longer cut by the threshold that half of that libration raised."""
    _, catalogue = as27
    verdict = catalogue.verdict
    assert str(verdict.status) == 'non_resonant'
    assert str(verdict.libration_origin) == 'induced'
    assert verdict.owner == '31E-36+0+5'
