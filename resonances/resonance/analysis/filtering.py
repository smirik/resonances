"""Low-pass wrappers of the analysis: Butterworth in second-order sections, applied forward-backward.

The (b, a) form of order 4 errs by 7.5e-2 already at a 5000-yr cutoff and a 1-yr step, and
derivatives amplify it; the sos form does not (SEPARATRIX §4.4). An angle is filtered after its
linear trend is removed (and the trend is put back). Times in years, uniformly sampled.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from resonances.resonance.analysis.settings import AnalysisSettings

# Relative spread of the time steps still called uniform (FILTER_GRID_RTOL): saved times come from REBOUND
# units divided by 2 pi and carry rounding, a decimated or reversed record does not change the step.
DEFAULT_GRID_RTOL = AnalysisSettings.__dataclass_fields__['filter_grid_rtol'].default
NAN = float('nan')
# Results each memoised function keeps inside one record's `record_memo` (most recent first out of the store last).
# Two cover the repeats (37087: response_series 206 of 212 calls repeated, band_pass 203 of 211); each entry holds
# up to four N-arrays, so the store stays per record and small (W23 P2: unscoped, 88710_c3 rose 1280 -> 2084 MB).
RECORD_MEMO_ENTRIES = 2

_RECORD_MEMO: Optional[Dict[str, OrderedDict]] = None  # the memo of the record analysed now (`record_memo`)


@contextmanager
def record_memo():
    """Remember the last RECORD_MEMO_ENTRIES results of `memoised` calls (`band_pass`, `response.smoothed_series`)
    until the analysis of one record ends (re-entrant: the nested passes of the record use the outer memo). The episode
    physics asks again for the same filtered series run after run (the runs' median periods repeat on a uniform grid)
    and the second episode pass after the forced waves repeats them all; the memo returns the arrays it computed, made
    read-only so that no caller can change what a later caller reads."""
    global _RECORD_MEMO
    if _RECORD_MEMO is not None:
        yield
        return
    _RECORD_MEMO = {}
    try:
        yield
    finally:
        _RECORD_MEMO = None


def memoised(name: str, objects: Sequence[Any], scalars: Tuple, compute: Callable[[], Any], fresh: Callable[[Any], Sequence]):
    """`compute()`, remembered inside `record_memo` under `name` by the identity of `objects` (kept alive with the
    entry, so an id is not reused) and the values `scalars`; outside a memo scope it is computed every time. `fresh`
    names the arrays of a result that `compute` created (they are made read-only when stored)."""
    if _RECORD_MEMO is None:
        return compute()
    store = _RECORD_MEMO.setdefault(name, OrderedDict())
    key = (tuple(id(obj) for obj in objects), scalars)
    hit = store.get(key)
    if hit is not None and all(kept is obj for kept, obj in zip(hit[0], objects)):
        store.move_to_end(key)
        return hit[1]
    value = compute()
    for array in fresh(value):
        array.flags.writeable = False
    store[key] = (tuple(objects), value)
    while len(store) > RECORD_MEMO_ENTRIES:
        store.popitem(last=False)
    return value


def butterworth_gain(period_yr, cutoff_yr: float, order: int, step_yr: Optional[float] = None):
    """Steady-state amplitude gain of the forward-backward digital Butterworth at period P [yr].

    The sos filter of `lowpass` is the bilinear transform of the analog Butterworth with the cutoff prewarped,
    so |H|^2 = 1 / (1 + (tan(pi dt / P) / tan(pi dt / P_c))^(2N)) and filtfilt applies |H|^2:

        G(P) = 1 / (1 + (tan(pi dt / P) / tan(pi dt / P_c))^(2N)),   P > 2 dt (nan at and above Nyquist).

    `step_yr=None` is the limit dt -> 0, the analog G(P) = 1 / (1 + (P_c / P)^(2N)): exact only for P_c >> dt
    (dt = 1, P_c = 5, N = 2: G(10) = 0.9615 digital against 0.9412 analog, G(10/3) = 0.0720 against 0.1649).
    Accepts a scalar or an array of periods.
    """
    period = np.asarray(period_yr, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        if step_yr is None:
            ratio = cutoff_yr / period
        else:
            above_nyquist = ~(period > 2 * step_yr)
            ratio = np.tan(np.pi * step_yr / np.where(above_nyquist, np.inf, period)) / np.tan(np.pi * step_yr / cutoff_yr)
            ratio = np.where(above_nyquist, np.nan, ratio)
        gain = 1.0 / (1.0 + ratio ** (2 * order))
    return float(gain) if gain.ndim == 0 else gain


def _uniform_step(t_yr: np.ndarray, grid_rtol: float) -> float:
    steps = np.diff(t_yr)
    if len(steps) == 0:
        raise ValueError('a filter needs at least two samples')
    step = float(np.median(steps))
    if step <= 0 or max(steps.max() - step, step - steps.min()) > grid_rtol * step:
        raise ValueError('the time grid is not uniform and increasing')
    return step


def r2(target: np.ndarray, residual: np.ndarray) -> float:
    """1 - sum(residual^2) / sum((target - mean)^2); nan for a constant target."""
    total = float(np.sum((target - target.mean()) ** 2))
    return 1 - float(np.sum(residual**2)) / total if total > 0 else NAN


def circular_mean(angles_rad) -> float:
    """Mean direction of the finite angles [rad, -pi..pi]; nan for none."""
    angles = np.asarray(angles_rad, dtype=float)
    angles = angles[np.isfinite(angles)]
    return float(np.angle(np.mean(np.exp(1j * angles)))) if len(angles) else NAN


def circular_resultant(angles_rad) -> float:
    """|<e^{i phi}>| of the finite angles (0..1; None counts as missing); nan for none."""
    angles = np.asarray(angles_rad, dtype=float)
    angles = angles[np.isfinite(angles)]
    return float(np.abs(np.mean(np.exp(1j * angles)))) if len(angles) else NAN


def linear_fit(t_yr: np.ndarray, series: np.ndarray) -> Tuple[float, float]:
    """(slope [unit/yr], value at t_yr[0]) of the least-squares line: the closed form on centred times and values
    (`np.polyfit(t - t0, x, 1)` without its N x 2 Vandermonde matrix). nan for fewer than two samples."""
    t = np.asarray(t_yr, dtype=float)
    x = np.asarray(series, dtype=float)
    if len(t) < 2:
        return NAN, NAN
    t_mean, x_mean = float(t.mean()), float(x.mean())
    centred = t - t_mean
    spread = float(np.dot(centred, centred))
    slope = float(np.dot(centred, x - x_mean) / spread) if spread > 0 else NAN
    return slope, x_mean - slope * (t_mean - float(t[0]))


def lowpass(
    t_yr: np.ndarray,
    series: np.ndarray,
    cutoff_yr: float,
    order: int,
    detrend: bool = True,
    grid_rtol: Optional[float] = None,
) -> np.ndarray:
    """Zero-phase low-pass of `series` (any unit) sampled at `t_yr`: sos Butterworth with period cutoff P_c [yr].

    `butter(order, 1/cutoff_yr, fs=1/dt, output='sos')` + `sosfiltfilt`; with `detrend` the least-squares
    line is removed first and added back (angles, slow drifts). Same length and unit as the input.
    Raises ValueError for a non-uniform grid (a step off the median by more than `grid_rtol`, default
    FILTER_GRID_RTOL) or a cutoff shorter than two steps.
    """
    t = np.asarray(t_yr, dtype=float)
    x = np.asarray(series, dtype=float)
    if x.shape != t.shape:
        raise ValueError('series and times differ in length')
    step = _uniform_step(t, DEFAULT_GRID_RTOL if grid_rtol is None else grid_rtol)
    if not np.isfinite(cutoff_yr) or cutoff_yr <= 2 * step:
        raise ValueError(f'cutoff {cutoff_yr} yr is not longer than two steps ({2 * step} yr)')
    sos = signal.butter(order, 1.0 / cutoff_yr, btype='low', fs=1.0 / step, output='sos')
    # the default padding needs more samples than a very short record has
    padlen = min(3 * (2 * len(sos) + 1), len(x) - 1)
    if not detrend:
        return signal.sosfiltfilt(sos, x, padlen=padlen)
    # np.polyfit, not `linear_fit`: the zone's members fit turns a 1e-12 change of sigma_f into a different V_E
    slope, intercept = np.polyfit(t - t[0], x, 1)
    trend = intercept + slope * (t - t[0])
    filtered = signal.sosfiltfilt(sos, x - trend, padlen=padlen)
    filtered += trend
    return filtered


def band_pass(t_yr: np.ndarray, values: np.ndarray, period_yr: float, factor: float, filter_order: int) -> np.ndarray:
    """values at periods [P / factor, factor P]: lowpass(P / factor) - lowpass(factor P) (the owner search's band).
    Remembered inside `record_memo` (read-only there)."""

    def compute():
        low = lowpass(t_yr, values, period_yr / factor, filter_order)
        return low - lowpass(t_yr, values, period_yr * factor, filter_order)

    scalars = (float(period_yr), float(factor), int(filter_order))
    return memoised('band_pass', (t_yr, values), scalars, compute, lambda band: (band,))


def lowpass_with(t_yr: np.ndarray, series: np.ndarray, cutoff_yr: float, settings: AnalysisSettings) -> np.ndarray:
    """`lowpass` with the order and grid tolerance of `settings` (the analysis' own filter); raises as `lowpass`."""
    return lowpass(t_yr, series, cutoff_yr, settings.filter_order, grid_rtol=settings.filter_grid_rtol)


def lowpass_or_none(t_yr: np.ndarray, series: np.ndarray, cutoff_yr: float, settings: AnalysisSettings) -> Optional[np.ndarray]:
    """`lowpass_with`; None where it raises (a grid that is not uniform, a cutoff not longer than two steps)."""
    try:
        return lowpass_with(t_yr, series, cutoff_yr, settings)
    except ValueError:
        return None


def is_finite(value) -> bool:
    """A measured scalar: not None, not NaN, not infinite."""
    return value is not None and np.isfinite(value)


def time_derivative(t_yr: np.ndarray, series: np.ndarray) -> np.ndarray:
    """Central differences d(series)/dt [unit/yr] (`np.gradient(series, t_yr)`)."""
    return np.gradient(np.asarray(series, dtype=float), np.asarray(t_yr, dtype=float))


def edge_window(t_yr: np.ndarray, trim_yr: float) -> Tuple[float, float]:
    """The analysis window [t0 + trim, t1 - trim] [yr] where the filter has no edge spread; (nan, nan) if empty."""
    t = np.asarray(t_yr, dtype=float)
    if len(t) == 0 or not np.isfinite(trim_yr):
        return float('nan'), float('nan')
    start, end = float(t[0] + trim_yr), float(t[-1] - trim_yr)
    if start > end:
        return float('nan'), float('nan')
    return start, end


def window_mask(t_yr: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    """Boolean mask of the samples inside the closed window [t0, t1] [yr]."""
    t = np.asarray(t_yr, dtype=float)
    start, end = window
    if not (np.isfinite(start) and np.isfinite(end)):
        return np.zeros(t.shape, dtype=bool)
    return (t >= start) & (t <= end)
