"""Oscillation diagnostics of a resonant angle: cycles, recurrence and the FAIR plane.

Pure signal processing on arrays (times in years and increasing, angles in radians;
a backward integration is passed in time order, see `MMRDiagnostics`). Nothing here
reads a Body or changes a status: these quantities feed the MMR diagnostic plots and
are not calibrated decision rules.

- A *cycle* runs from one maximum of the filtered, unwrapped angle to the next. It is a
  candidate oscillation, not a confirmed libration: a slow circulation with ripples
  yields cycles too, which is exactly what the drift of their centres shows.
- The *recurrence* distance compares states (sigma, sigma_dot) at two times, in units
  of the object's own oscillation. It is readable within one object and must not be
  compared between objects or thresholded.
- The *FAIR* plane (Forgács-Dajka, Sándor & Érdi 2018) is the asteroid mean anomaly
  against the mean-longitude difference to one planet. It is a two-body diagnostic.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Cycle:
    """One maximum-to-maximum interval of an angle (times in years, angle in radians)."""

    start: float
    end: float
    diameter: float  # mean of the two maxima minus the minimum
    centre: float  # halfway between that mean and the minimum

    @property
    def period(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Recurrence:
    """Distance matrix of decimated states; NaN on the excluded near-diagonal band."""

    times: np.ndarray
    distance: np.ndarray


def default_prominence(sigma: np.ndarray) -> float:
    """Minimum peak prominence: 5% of the angle's range, kept within [0.005, 0.1] rad.

    The floor stops sampling noise from being counted as cycles; the cap keeps small
    oscillations detectable on top of a large drift.
    """
    return min(0.1, max(0.005, 0.05 * float(np.ptp(sigma))))


def find_cycles(times: np.ndarray, sigma: np.ndarray, prominence: Optional[float] = None) -> List[Cycle]:
    """Maximum-to-maximum cycles of the (filtered, unwrapped) angle.

    Incomplete fragments at both ends are dropped. An interval is skipped when its
    minimum sits on one of its maxima (no oscillation between them).
    """
    times = np.asarray(times, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if prominence is None:
        prominence = default_prominence(sigma)
    peaks, _ = find_peaks(sigma, prominence=prominence)
    cycles = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        bottom = left + int(np.argmin(sigma[left : right + 1]))
        upper = (sigma[left] + sigma[right]) / 2
        diameter = upper - sigma[bottom]
        if diameter <= 0 or bottom in (left, right):
            continue
        cycles.append(
            Cycle(
                start=float(times[left]),
                end=float(times[right]),
                diameter=float(diameter),
                centre=float((upper + sigma[bottom]) / 2),
            )
        )
    return cycles


def _cycle_slice(times: np.ndarray, cycle: Cycle) -> slice:
    """Samples of one cycle, ends included (times are increasing)."""
    return slice(int(np.searchsorted(times, cycle.start, 'left')), int(np.searchsorted(times, cycle.end, 'right')))


def cycle_centres(cycles: List[Cycle], times: np.ndarray, axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cycle: mid time, angle centre and the midpoint of the semi-major axis range."""
    axis = np.asarray(axis, dtype=float)
    mid = np.array([(c.start + c.end) / 2 for c in cycles])
    centre = np.array([c.centre for c in cycles])
    axis_mid = np.array([(axis[m].max() + axis[m].min()) / 2 for m in (_cycle_slice(times, c) for c in cycles)])
    return mid, centre, axis_mid


def fold_cycle(cycle: Cycle, times: np.ndarray, series: np.ndarray, subtract_median: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """`series` over one cycle against the cycle fraction 0 (first maximum) .. 1 (next)."""
    part = _cycle_slice(times, cycle)
    values = np.asarray(series, dtype=float)[part]
    if subtract_median and values.size:
        values = values - np.median(values)
    return (np.asarray(times)[part] - cycle.start) / cycle.period, values


def recurrence_scales(sigma: np.ndarray, rate: np.ndarray, cycles: Optional[List[Cycle]] = None) -> Tuple[float, float]:
    """The object's own units for the recurrence distance (rad, rad/yr).

    s_sigma is half the median cycle diameter (half the angle's range without cycles),
    s_rate half the 5-95 percentile spread of the rate.
    """
    if cycles:
        sigma_scale = float(np.median([c.diameter for c in cycles]) / 2)
    else:
        sigma_scale = float(np.ptp(sigma) / 2)
    rate_scale = float((np.percentile(rate, 95) - np.percentile(rate, 5)) / 2)
    return max(sigma_scale, 0.01), max(rate_scale, 1e-6)


def recurrence(
    times: np.ndarray,
    sigma: np.ndarray,
    rate: np.ndarray,
    sigma_scale: float,
    rate_scale: float,
    max_points: int,
    exclude_samples: int = 2,
) -> Recurrence:
    """Distance between the states (sigma, sigma_dot) at every pair of decimated times.

    D = hypot(d_sigma / sigma_scale, d_rate / rate_scale), see `recurrence_scales`. The
    angle is taken on the real line, so a full circulation never returns at zero
    distance. Decimation is display only: pass the rate computed at full resolution.
    Pairs at most `exclude_samples` decimated steps apart are NaN (0 = the diagonal
    only), since neighbouring samples are trivially close.
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    idx = np.unique(np.round(np.linspace(0, n - 1, min(max_points, n))).astype(int))
    y = np.asarray(sigma, dtype=float)[idx] / sigma_scale
    r = np.asarray(rate, dtype=float)[idx] / rate_scale
    # In place: the matrix reaches 2500^2, so no further n^2 temporaries.
    distance = np.subtract.outer(y, y)
    np.square(distance, out=distance)
    rate_part = np.subtract.outer(r, r)
    np.square(rate_part, out=rate_part)
    distance += rate_part
    del rate_part
    np.sqrt(distance, out=distance)
    m = len(idx)
    for offset in range(min(exclude_samples, m - 1) + 1):
        rows = np.arange(m - offset)
        distance[rows, rows + offset] = distance[rows + offset, rows] = np.nan
    return Recurrence(times=times[idx], distance=distance)


def is_inner(axis: np.ndarray, planet_axis: np.ndarray) -> bool:
    """Whether the asteroid orbits inside the planet (median semi-major axes)."""
    return bool(np.median(axis) < np.median(planet_axis))


def fair_coordinates(
    mean_anomaly: np.ndarray, longitude: np.ndarray, planet_longitude: np.ndarray, inner: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """FAIR plane in degrees: x = M mod 360, y = (lambda_p - lambda) mod 360 for an inner
    asteroid, (lambda - lambda_p) mod 360 for an outer one."""
    delta = np.asarray(planet_longitude, dtype=float) - np.asarray(longitude, dtype=float)
    if not inner:
        delta = -delta
    return np.degrees(np.mod(mean_anomaly, 2 * np.pi)), np.degrees(np.mod(delta, 2 * np.pi))


def fair_step_fraction(times: np.ndarray, axis: np.ndarray) -> float:
    """Output step as a fraction of the asteroid's orbital period (Kepler, years).

    FAIR needs the mean anomaly sampled densely within an orbit; above ~1/4 the plane
    fills with aliased points and its strips are not resolved.
    """
    step = float(np.median(np.diff(np.asarray(times, dtype=float))))
    period = float(np.median(np.asarray(axis, dtype=float))) ** 1.5
    return abs(step) / period
