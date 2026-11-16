"""W11: the turning points of each regime between jumps of a are found at that regime's own prominence.

The TP prominence covers what sigma_f holds faster than any libration (`analyse.fast_residual`). Measured over the
whole record, a stretch where sigma circulates between many jumps of a sets it for the libration before those jumps
too, and the libration disappears (2003 UG22 10E-17+0+7, 40 kyr: 80 deg over the record against 26 deg before the
jumps, a prominence of 322 deg and no turning point before 26.7 kyr). `geometry.finish_geometry` now measures each
regime's prominence on its own samples (`analyse.regime_prominence`).

Expected values come from elsewhere: the analytic extrema of a synthetic sine, and the analysis of the same UG22
series cut before the jumps (no jump there, so no regime logic at all).
"""

from dataclasses import fields, is_dataclass, replace

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

# ------------------------------------------------------------------ synthetic: a libration, then a fast circulation


STEP_YR = 1.0
CUTOFF_YR = 5.0
LIBRATION_PERIOD_YR = 300.0
LIBRATION_HALF_AMP_RAD = 1.2
CAPTURE_END_YR = 6000.0  # the first jump of a
JUMP_EVERY_YR = 400.0
FLOOR_PERIOD_YR = 150.0  # the shortest libration (the MMR default)
RIPPLE_FACTOR = 4.0


def _lowpass(x, period_yr):
    """An independent Butterworth low-pass (order 2, forward-backward) at `period_yr`."""
    sos = signal.butter(2, (1 / period_yr) / (0.5 / STEP_YR), output='sos')
    return signal.sosfiltfilt(sos, x)


def _synthetic():
    """sigma = pi + 1.2 sin(2 pi t / 300 yr) up to 6 kyr; then a circulation of 60 yr per turn carrying a 40-yr wave of
    3 rad, while a steps by 0.01 au every 400 yr (the jumps). Returns the angle and the prominence of a stretch:
    4 x the 95th percentile of |sigma - lowpass(sigma, 150 yr)| on it (the MMR rule, computed here independently)."""
    t = np.arange(0.0, 12000.0 + STEP_YR, STEP_YR)
    libration = np.pi + LIBRATION_HALF_AMP_RAD * np.sin(2 * np.pi * t / LIBRATION_PERIOD_YR)
    after = t - CAPTURE_END_YR
    circulation = np.pi + 2 * np.pi * after / 60.0 + 3.0 * np.sin(2 * np.pi * after / 40.0)
    sigma = np.where(t < CAPTURE_END_YR, libration, circulation)
    a = 1.0 + 0.01 * np.where(t < CAPTURE_END_YR, 0, 1 + np.floor(after / JUMP_EVERY_YR))
    window = (t[0] + CUTOFF_YR, t[-1] - CUTOFF_YR)
    angle = g.FilteredAngle(
        t_yr=t,
        angle_raw=sigma,
        angle_f=sigma,
        rate_f=np.gradient(sigma, t),
        angle_f_half=sigma,
        angle_f_double=sigma,
        slow_f=a,
        identity_coefficient=-1.0,
        cutoff_yr=CUTOFF_YR,
        step_yr=STEP_YR,
        window=window,
    )
    fast = np.abs(sigma - _lowpass(sigma, FLOOR_PERIOD_YR))
    inside = (t >= window[0]) & (t <= window[1])

    def prominence(own):
        return RIPPLE_FACTOR * float(np.percentile(fast[own & inside], 95))

    return angle, prominence, prominence(np.ones_like(t, dtype=bool))


def _finished(angle, record_prominence, regime_prominence):
    geometry = g.measure_geometry(angle, record_prominence, S)
    nan = float('nan')
    local = None if regime_prominence is None else g.LocalProminence(np.radians(S.mmr_tp_prominence_floor_deg), regime_prominence)
    return g.finish_geometry(geometry, angle, nan, nan, nan, nan, np.zeros(0), np.zeros(0), 1e-4, nan, [], S, regime_prominence=local)


def test_the_libration_before_the_jumps_keeps_its_turning_points():
    angle, prominence, record = _synthetic()
    # the circulation inflates the record's prominence beyond the libration's full swing of 2.4 rad
    assert record > 2 * LIBRATION_HALF_AMP_RAD
    geometry = _finished(angle, record, prominence)
    assert geometry.regime_start_yr > CAPTURE_END_YR
    first_jump = min(jump.time_yr for jump in geometry.jumps if not jump.kick)
    assert first_jump == pytest.approx(CAPTURE_END_YR, abs=geometry.jump_tau_yr)
    (episode,) = geometry.episodes_before_jump
    # the extrema of the sine: 75 + 150 k yr (k = 0..39) before 6 kyr
    analytic = 75.0 + 150.0 * np.arange(40)
    assert set(episode.tp_times_yr) <= set(analytic)
    assert episode.n_tp >= len(analytic) - 1
    assert episode.tp_prominence_rad < record
    # the circulation after the capture makes no episode at its own prominence either
    assert all(run.t_start_yr < CAPTURE_END_YR for run in geometry.runs if run.accepted)


def test_with_the_record_s_prominence_the_libration_is_lost():
    """The defect itself: one prominence for every regime leaves no turning point of the libration."""
    angle, _, record = _synthetic()
    geometry = _finished(angle, record, None)
    assert geometry.regime_start_yr > CAPTURE_END_YR
    assert not geometry.episodes_before_jump
    assert not any(len(run.tp_times_yr) and run.t_start_yr < CAPTURE_END_YR for run in geometry.runs)


def test_a_record_without_a_jump_never_asks_for_a_regime_prominence():
    angle, _, _ = _synthetic()
    t = angle.t_yr
    keep = t < CAPTURE_END_YR
    libration = g.FilteredAngle(
        **{
            name: (value[keep] if isinstance(value, np.ndarray) and value.shape == t.shape else value)
            for name, value in vars(angle).items()
        }
    )
    libration = replace(libration, window=(CUTOFF_YR, CAPTURE_END_YR - STEP_YR - CUTOFF_YR))

    def never(own):
        raise AssertionError('no regime without a jump')

    geometry = _finished(libration, 0.5, never)
    assert geometry.regime_start_yr == 0.0 and len(geometry.episodes) == 1
    assert geometry.episodes[0].tp_prominence_rad == 0.5


def _prep(sigma, t):
    return make(
        m.Preprocessed,
        cutoff=m.Cutoff(CUTOFF_YR, 'floor', 0.0, CUTOFF_YR, CUTOFF_YR / STEP_YR),
        t_yr=t,
        angle_clean=sigma,
        angle_f=sigma,
        window=(t[0] + CUTOFF_YR, t[-1] - CUTOFF_YR),
        unremoved_ripple_rad=0.0,
        period_floor_yr=FLOOR_PERIOD_YR,
    )


def test_the_regime_prominence_does_not_see_the_next_regime():
    """The first regime (to 6 kyr, minus the jump zone) gets the same prominence whatever follows the jump: a fast
    circulation or the same libration going on. The low-pass at the period floor leaks across the jump only within
    FILTER_EDGE_TRIM_CUTOFFS x P_f, and those samples do not count."""
    angle, _, _ = _synthetic()
    t = angle.t_yr
    calm = np.pi + LIBRATION_HALF_AMP_RAD * np.sin(2 * np.pi * t / LIBRATION_PERIOD_YR)
    first = t <= CAPTURE_END_YR - 5.0
    after_chaos = regime_prominence(_prep(angle.angle_f, t), S)(first)
    after_calm = regime_prominence(_prep(calm, t), S)(first)
    assert np.isfinite(after_chaos) and after_chaos == pytest.approx(after_calm, rel=0.05)
    # and far below the record's prominence of the chaotic record
    assert after_chaos < 0.1 * _synthetic()[2]


def test_a_regime_shorter_than_the_floor_after_the_trim_has_no_prominence_of_its_own():
    angle, _, record = _synthetic()
    t = angle.t_yr
    prominence = regime_prominence(_prep(angle.angle_f, t), S)
    trim = S.filter_edge_trim_cutoffs * FLOOR_PERIOD_YR
    # between two jumps: (2 trim + P_f) of own samples are needed
    short = (t >= 7000.0) & (t <= 7000.0 + 2 * trim + 0.9 * FLOOR_PERIOD_YR)
    long = (t >= 7000.0) & (t <= 7000.0 + 2 * trim + 1.1 * FLOOR_PERIOD_YR)
    assert np.isnan(prominence(short)) and np.isfinite(prominence(long))
    # the geometry then keeps the record's prominence for it
    regimes = g._regimes(t, [7500.0, 7900.0], 10.0, record, prominence)
    assert regimes[1].prominence_rad == record


# ------------------------------------------------------------------ 2003 UG22 10E-17+0+7, 40 kyr (ias15, step 0.25 yr)


UG22 = 'mmr2__2003_UG22__10E-17+0+7__40kyr'


def _cut(obj, keep, n):
    """The series restricted to the samples `keep` (every per-sample array of every nested dataclass)."""
    if isinstance(obj, np.ndarray):
        return obj[keep] if obj.shape[:1] == (n,) else obj
    if isinstance(obj, dict):
        return {key: _cut(value, keep, n) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)) and obj and is_dataclass(obj[0]):
        return type(obj)(_cut(value, keep, n) for value in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return replace(obj, **{f.name: _cut(getattr(obj, f.name), keep, n) for f in fields(obj) if f.init})
    return obj


@pytest.fixture(scope='module')
def ug22():
    if not io.example_path(UG22).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = io.load_series(UG22)
    whole = analyse_mmr(series, S)
    cut = analyse_mmr(_cut(series, series.t_yr < 20000.0, series.t_yr.size), S)
    return whole, cut


@pytest.mark.benchmark
def test_ug22_the_libration_before_the_jumps_is_the_one_of_the_cut_record(ug22):
    whole, cut = ug22
    geometry = whole.geometry
    first_jump = min(jump.time_yr for jump in geometry.jumps if not jump.kick)
    # the jumps of a start where a leaves the libration's level (1.426 au -> 1.36 au at 31.0-31.9 kyr). At P_c = 5 yr
    # (20 steps, before W16) the locator also took a 0.002-au step at 28.9 kyr, 1.05 x its threshold; at the physical
    # P_c = 75 yr that step is 0.3 of the threshold (TEST-DISPUTES W16)
    assert 28000.0 < first_jump < 31500.0
    early = [episode for episode in geometry.episodes_before_jump if episode.t_start_yr < 1000.0]
    assert len(early) == 1
    episode = early[0]
    # W18: at the run's own threshold the exit's last swings at 20.5-20.7 kyr join the run, whose centre follows the
    # averaged potential by 237 deg: it is a libration about the moving well (psi), cut where the swings leave the well
    assert episode.frame == 'psi' and 20000.0 < episode.t_end_yr < 20600.0
    # the record cut at 20 kyr has no jump: one episode of 170 TPs from 169 yr, P 221 yr
    (reference,) = cut.geometry.episodes
    assert reference.n_tp >= 150 and episode.n_tp >= reference.n_tp
    assert episode.period_med_yr == pytest.approx(reference.period_med_yr, rel=0.02)
    # away from the cut's edge both find the same turning points
    inner = (reference.tp_times_yr > 1000.0) & (reference.tp_times_yr < 19000.0)
    ours = episode.tp_times_yr[(episode.tp_times_yr > 1000.0) & (episode.tp_times_yr < 19000.0)]
    np.testing.assert_allclose(ours, reference.tp_times_yr[inner])
    # its prominence is the one of the regime before the jumps, far below the record's (259 deg at P_c = 75 yr)
    record_prominence_deg = S.mmr_tp_ripple_factor * whole.numbers_row()['flt_fast_residual_deg']
    assert record_prominence_deg > 180.0  # above the ceiling of any libration's turning points
    assert np.degrees(episode.tp_prominence_rad) < record_prominence_deg / 2
    assert np.degrees(episode.tp_prominence_rad) == pytest.approx(np.degrees(cut.geometry.turning_points.prominence_rad), rel=0.05)
