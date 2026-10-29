"""The slow line of the centre of a secular or Lidov-Kozai angle: its period, half-amplitude and what makes it.

The centre c(t) is the angle smoothed over W_c = LIBRATION_CENTRE_CYCLES x P_cyc (`geometry.CentreTrack`). Under a
fast libration it can carry a slow wave of its own (591986: 6-kyr cycles on a centre that swings +-22 deg at 54 kyr).
The verdict rests on one scale; this measures the other one as a column, even when the fine scale decided:

    period, half-amplitude  the strongest sinusoid of the centre between W_c and span / ZLK_CENTRE_MIN_CYCLES, with a
                            quadratic trend in the model (least squares on a grid of 1/10 of a Rayleigh width); only
                            for a libration or a drift (a circulating or episodic centre has no slow line of its own)
    ZLK origin              forced_beat:<gk>-<gj>  the period is within a Rayleigh width of a beat 2 pi/|g_k - g_j| of
                                                   two fitted forced eccentricity modes with nonzero amplitudes, and the
                                                   phase wave of that forced vector, smoothed like the centre, explains
                                                   the centre: correlation >= ZLK_CENTRE_MODEL_R_MIN and swing (gain,
                                                   std of the model / std of the centre, both less a quadratic trend)
                                                   within a factor ZLK_CENTRE_MODEL_AMP_FACTOR (591986: g6-g5, r 0.99,
                                                   model half-swing arcsin(|c_g6|/|c_g5|) = 22.8 deg). Correlation alone
                                                   is blind to size: a g6 term of 1e-6 has the shape of any centre wave
                                                   at that period (a node's, say). Without the per-mode amplitudes (an
                                                   old file) no forced beat is named.
                            node                   the node's own wave (Omega less its linear trend) does instead
                            forced_direction       the direction of the whole fitted forced vector less the node,
                                                   arg z_forced(t) - Omega(t), does (the same r and gain test): with the
                                                   forced vector holding varpi this is the centre of omega itself, at
                                                   every scale (591986 over 4 Myr: the 1.43-Myr wave of a node locked to
                                                   g5 plus the 54-kyr g6-g5 beat, r 0.998, gain 1.07; the node alone
                                                   gives r 0.82, the beat alone is not the strongest line)
                            unexplained            none of these
    secular origin          forced_wave:<mode>     within a Rayleigh width of a forced wave of the angle's own modes
                            unexplained

Nothing here changes a status. Pure: arrays in, `CentreLine` out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from resonances.resonance.analysis.filtering import lowpass_or_none
from resonances.resonance.analysis.secular.models import ARCSEC_PER_TURN, CentreLine

NAN = float('nan')
GRID_OVERSAMPLING = 10  # frequency grid step = 1 / (this x span)
MAX_TRACK_POINTS = 4000  # the track is smoothed over W_c: thinning it to this many points loses nothing
MIN_TRACK_POINTS = 16


def strongest_line(t: np.ndarray, x: np.ndarray, period_min: float, period_max: float) -> Tuple[float, float]:
    """(period [yr], amplitude) of the strongest sinusoid of x(t) between the two periods, with a quadratic trend in the
    model (structure slower than the record); (nan, nan) when the band is empty or the series too short."""
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    finite = np.isfinite(t) & np.isfinite(x)
    t, x = t[finite], x[finite]
    if len(t) < MIN_TRACK_POINTS or not (0 < period_min < period_max):
        return NAN, NAN
    stride = max(1, len(t) // MAX_TRACK_POINTS)
    t, x = t[::stride], x[::stride]
    span = float(t[-1] - t[0])
    tau = (t - t[0]) / span - 0.5
    frequencies = np.arange(1.0 / period_max, 1.0 / period_min, 1.0 / (GRID_OVERSAMPLING * span))
    best = (NAN, 0.0)
    for frequency in frequencies:
        phase = 2 * np.pi * frequency * t
        design = np.column_stack([np.cos(phase), np.sin(phase), np.ones_like(t), tau, tau**2])
        coefficients, *_ = np.linalg.lstsq(design, x, rcond=None)
        amplitude = float(np.hypot(coefficients[0], coefficients[1]))
        if amplitude > best[1]:
            best = (1.0 / frequency, amplitude)
    return (best[0], best[1]) if np.isfinite(best[0]) else (NAN, NAN)


def measure_line(track, settings, prefix: str) -> CentreLine:
    """The slow line of a centre track (`geometry.CentreTrack`), origin not yet named."""
    if track is None or len(track.centre_rad) < MIN_TRACK_POINTS:
        return CentreLine(prefix)
    t, centre = np.asarray(track.t_yr, dtype=float), np.asarray(track.centre_rad, dtype=float)
    span = float(t[-1] - t[0])
    period, amplitude = strongest_line(t, centre, float(track.window_yr), span / settings.zlk_centre_min_cycles)
    return CentreLine(prefix, period_yr=period, half_amp_rad=amplitude)


def _within_rayleigh(period_yr: float, other_yr: float, resolution_arcsec_yr: float) -> bool:
    """Whether two periods are one line at this record's resolution: |f1 - f2| < 1296000 / T [arcsec/yr]."""
    if not (np.isfinite(period_yr) and np.isfinite(other_yr) and period_yr > 0 and other_yr > 0):
        return False
    return abs(ARCSEC_PER_TURN / period_yr - ARCSEC_PER_TURN / other_yr) < resolution_arcsec_yr


TREND_DEGREE = 2  # the centre and a model are compared less a quadratic trend: what is slower than the record


def _model_fit(track, t_series: np.ndarray, model: np.ndarray, settings) -> Tuple[float, float]:
    """(r, gain) of a model against the centre on the series grid: the model smoothed over the centre's W_c, both less a
    quadratic trend; r their correlation (591986, g6-g5: 0.991; with a linear trend only, the record's slower bend
    leaves 0.93), gain = std(model) / std(centre), whether the model's swing is the centre's size. (nan, nan) when
    either is flat or the model cannot be smoothed."""
    smoothed = lowpass_or_none(t_series, model, float(track.window_yr), settings)
    if smoothed is None:
        return NAN, NAN
    t = np.asarray(track.t_yr, dtype=float)
    centre = np.asarray(track.centre_rad, dtype=float)
    predicted = np.interp(t, t_series, smoothed)
    residual_c = centre - np.polyval(np.polyfit(t, centre, TREND_DEGREE), t)
    residual_m = predicted - np.polyval(np.polyfit(t, predicted, TREND_DEGREE), t)
    if np.std(residual_c) == 0 or np.std(residual_m) == 0:
        return NAN, NAN
    return float(np.corrcoef(residual_c, residual_m)[0, 1]), float(np.std(residual_m) / np.std(residual_c))


def _explains(r: float, gain: float, settings) -> bool:
    """A model explains the centre: shape (r >= ZLK_CENTRE_MODEL_R_MIN) and size (gain within a factor
    ZLK_CENTRE_MODEL_AMP_FACTOR of one)."""
    factor = settings.zlk_centre_model_amp_factor
    return bool(np.isfinite(r) and np.isfinite(gain) and r >= settings.zlk_centre_model_r_min and 1 / factor <= gain <= factor)


@dataclass
class ForcedBeatModel:
    """The forced beat that a centre's slow wave is compared with (`forced_beat_model`)."""

    label: str  # '<gk>-<gj>'
    r: float = NAN
    gain: float = NAN
    half_amp_rad: float = NAN  # arcsin(|c_k| / |c_j|)
    explains: bool = False


def forced_beat_model(track, t_series: np.ndarray, elements, period_yr: float, settings) -> Optional[ForcedBeatModel]:
    """The forced beat of `elements` within a Rayleigh width of `period_yr` whose phase wave explains the centre track
    (`_explains`); else the nearest such beat that does not (explains False); None when no measured beat lies there
    (no elements, no per-mode amplitudes, no pair of nonzero amplitudes at that period)."""
    if track is None or elements is None or not np.isfinite(period_yr):
        return None
    t_series = np.asarray(t_series, dtype=float)
    forced = elements.forced_vector(t_series)
    if forced is None:
        return None
    candidate = None
    for label, beat in elements.forced_beats().items():
        if not _within_rayleigh(period_yr, beat, elements.resolution):
            continue
        small, big = label.split('-')
        rate = elements.basis_e[big] * 2 * np.pi / ARCSEC_PER_TURN
        wave = np.unwrap(np.angle(forced)) - (rate * t_series + np.angle(elements.amplitudes_e[big]))
        r, gain = _model_fit(track, t_series, wave, settings)
        amplitude_small, amplitude_big = abs(elements.amplitudes_e[small]), abs(elements.amplitudes_e[big])
        half = float(np.arcsin(min(1.0, amplitude_small / amplitude_big)))
        model = ForcedBeatModel(label, r, gain, half, _explains(r, gain, settings))
        if model.explains:
            return model
        candidate = candidate or model
    return candidate


def zlk_centre_line(track, series, elements, settings) -> CentreLine:
    """The slow line of the centre of omega and its origin (module docstring). `elements` is the body's FreeElements
    (None, or without amplitudes: no forced_beat test); `series` the LongPeriodSeries (its Omega for the node)."""
    line = measure_line(track, settings, 'zlk_centre')
    if not np.isfinite(line.period_yr):
        return line
    t_series = np.asarray(series.t_yr, dtype=float)
    model = forced_beat_model(track, t_series, elements, line.period_yr, settings)
    if model is not None and model.explains:
        line.origin, line.model_r, line.model_gain = f'forced_beat:{model.label}', model.r, model.gain
        line.model_half_amp_rad = model.half_amp_rad
        return line
    if series.Omega is not None:
        node = np.unwrap(np.asarray(series.Omega, dtype=float))
        wave = -(node - np.polyval(np.polyfit(t_series, node, 1), t_series))  # omega = varpi - Omega
        r, gain = _model_fit(track, t_series, wave, settings)
        if _explains(r, gain, settings):
            line.origin, line.model_r, line.model_gain = 'node', r, gain
            return line
        forced = elements.forced_vector(t_series) if elements is not None else None
        if forced is not None:
            r, gain = _model_fit(track, t_series, np.unwrap(np.angle(forced) - np.asarray(series.Omega, dtype=float)), settings)
            if _explains(r, gain, settings):
                line.origin, line.model_r, line.model_gain = 'forced_direction', r, gain
                return line
    line.origin = 'unexplained'
    return line


def secular_centre_line(track, forced_waves_yr: Mapping[str, float], span_yr: float, settings) -> CentreLine:
    """The slow line of the centre of a modulated secular angle: forced_wave:<mode> when its period is a forced wave of
    the angle's own modes within a Rayleigh width, else unexplained (the proper angle uses free vectors, so the
    forced-vector beat of ZLK does not apply)."""
    line = measure_line(track, settings, 'sec_centre')
    if not np.isfinite(line.period_yr):
        return line
    resolution = ARCSEC_PER_TURN / span_yr if span_yr > 0 else NAN
    match: Optional[str] = next(
        (label for label, wave in forced_waves_yr.items() if _within_rayleigh(line.period_yr, wave, resolution)), None
    )
    line.origin = f'forced_wave:{match}' if match else 'unexplained'
    return line
