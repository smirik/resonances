"""The e-i coupling of a Lidov-Kozai omega inside its deciding librating episode (W28 D10, the method validated by two
blind estimators in W28 D9).

Why inside the episode: the record's coupling (`diagnostics.measure_ei_coupling`, column zlk_ei_coupling) fits the
whole record at the decisive period, and the stretches where omega circulates also trade e for i at constant H near
that period (the quadrupole Hamiltonian does so whether omega librates or not: 15527 circulates and shows zlk). In
88899 over 500 kyr both librating episodes have e and i in phase with H not conserved (eta_H 1.1-1.4), and 2020 PP1's
deciding episode shows no coupling, while the whole record says zlk for PP1. The status of an episodic omega without a
gate verdict (`decide.decide_status`, the W28 tree; the author's answer q-1 E, 2026-10-31) reads this
measurement.

Method, on the deciding episode [first TP, last TP] at its own median full period P:

1. e, the tilt i (min(i, pi - i): an orbit that crosses 90 deg is not measured), G = sqrt(1 - e^2), H = G cos i and
   K = G (1 - cos i) are high-passed on the WHOLE record by subtracting their centred running mean over one period
   (the boxcar has a zero at 1/P, so the line at P passes whole and the planets' slower secular waves go; no edge
   effect inside an episode away from the record's ends);
2. inside the episode each is demodulated at P: x(t) exp(-2 pi i t / P) averaged over a sliding window of one period
   (removes the 2/P image and the mean) gives local complex amplitudes, cycle by cycle, so a period that drifts
   over many cycles keeps its phase (one fixed-P fit lost it over 13 cycles of 162474 at 1.5 Myr);
3. the phase of i against e is arg sum E* I, the coherence |sum E* I| / sqrt(sum |E|^2 sum |I|^2) signed by
   cos(phase), the amplitudes the rms of E and I times sqrt 2, and the integrals' residuals
       eta_H = rms(H) / |cos i rms(G)|,    eta_K = rms(K) / |(1 - cos i) rms(G)|
   (0: the integral is conserved by the trade; 1: as if i were frozen).

Class:
    zlk          phase within ZLK_EI_PHASE_TOL_DEG of 180, coherence <= -ZLK_EI_EPISODE_MIN_COHERENCE,
                 eta_H <= ZLK_EI_H_RESIDUAL_MAX (H = sqrt(1-e^2) cos i conserved: the quadrupole integral)
    forced_beat  phase within the tolerance of 0, coherence >= the minimum, eta_K <= the same maximum (the momentum
                 conjugate to Omega when the forcing acts on varpi only)
    none         every number finite, neither
    not_measured no deciding librating episode (no_episode), fewer than ZLK_EI_MIN_CYCLES periods between its first
                 and last TPs (short_episode: at 1.5-3 periods the two D9 estimators agree 3 times in 11), the band
                 [(1 - h)/P, (1 + h)/P] at or above the Nyquist frequency (undersampled, h = ZLK_EI_BAND_HALFWIDTH),
                 an orbit flip, an e that does not move (flat_series; a frozen i with a moving e is none, eta_H = 1)
                 or a law that is not finite (law_undefined: i at 0 or 90 deg).
The slope |di|/|de| over the constant-H law is a column (ratio_h): it fails on large cycles (1866: 3.0 x the law
with H cancelled to 5 %), which eta_H does not.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from resonances.resonance.analysis.geometry import Episode
from resonances.resonance.analysis.secular.models import EpisodeCoupling, LongPeriodSeries
from resonances.resonance.analysis.settings import AnalysisSettings

NAN = float('nan')
# an oscillation of e or sqrt(1 - e^2) below this is round-off of the high-pass, not a line (a definition, not a
# threshold: e itself is of order 0.01-1)
NUMERICAL_ZERO = 1e-12


def running_mean_highpass(t: np.ndarray, x: np.ndarray, period: float) -> np.ndarray:
    """x less its centred running mean over one period (an odd number of samples; shorter windows at the edges)."""
    step = float(np.median(np.diff(t)))
    n = max(int(round(period / step)) | 1, 1)
    half = n // 2
    cumulative = np.concatenate([[0.0], np.cumsum(x)])
    index = np.arange(len(x))
    low, high = np.clip(index - half, 0, len(x)), np.clip(index + half + 1, 0, len(x))
    return x - (cumulative[high] - cumulative[low]) / (high - low)


def demodulate(t: np.ndarray, x: np.ndarray, period: float) -> np.ndarray:
    """Local complex amplitudes of x at `period`: x exp(-2 pi i t / P) averaged over a sliding one-period window."""
    step = float(np.median(np.diff(t)))
    n = max(int(round(period / step)), 1)
    return np.convolve((x - x.mean()) * np.exp(-2j * np.pi * t / period), np.ones(n) / n, mode='valid')


def _rms(z: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(z) ** 2))) if len(z) else NAN


def phase_off_deg(phase_deg: float, target_deg: float) -> float:
    """The angular distance of a phase from a target [deg], in [0, 180]."""
    return abs((phase_deg - target_deg + 180.0) % 360.0 - 180.0)


def measure_episode_coupling(series: LongPeriodSeries, episode: Optional[Episode], settings: AnalysisSettings) -> EpisodeCoupling:
    """The coupling of e and i inside `episode` (the deciding librating episode, `decide.decisive_librating_episode`)
    at its median full period (module docstring); not_measured / no_episode without one. `series.inc` and
    `series.Omega` are those of the ZLK angle's frame (the invariable plane by default): never rotated here."""
    thresholds = dict(
        min_cycles=settings.zlk_ei_min_cycles,
        phase_tol_deg=settings.zlk_ei_phase_tol_deg,
        min_coherence=settings.zlk_ei_episode_min_coherence,
        h_residual_max=settings.zlk_ei_h_residual_max,
    )
    if episode is None:
        return EpisodeCoupling(**thresholds)
    period = float(episode.period_med_yr)
    tp = np.asarray(episode.tp_times_yr, dtype=float)
    result = EpisodeCoupling(reason='', episode_id=episode.episode_id, period_yr=period, **thresholds)
    if not (np.isfinite(period) and period > 0 and len(tp) >= 2):
        result.reason = 'no_episode'
        return result
    result.n_periods = float((tp[-1] - tp[0]) / period)
    t_all = np.asarray(series.t_yr, dtype=float)
    step = float(np.median(np.diff(t_all))) if len(t_all) > 1 else NAN
    halfwidth = settings.zlk_ei_band_halfwidth
    if not (np.isfinite(step) and step > 0 and (1.0 + halfwidth) / period < 0.5 / step):
        result.reason = 'undersampled'  # the band around 1/P reaches the Nyquist frequency: aliased
        return result
    if result.n_periods < settings.zlk_ei_min_cycles:
        result.reason = 'short_episode'
        return result
    e_all, inc_all = np.asarray(series.e, dtype=float), np.asarray(series.inc, dtype=float)
    if np.nanmin(inc_all) < np.pi / 2 < np.nanmax(inc_all):
        result.reason = 'orbit_flip'
        return result
    tilt_all = np.pi - inc_all if np.nanmedian(inc_all) > np.pi / 2 else inc_all
    inside = (t_all >= tp[0]) & (t_all <= tp[-1])
    g_all = np.sqrt(1.0 - e_all**2)
    raw = {'e': e_all, 'i': tilt_all, 'G': g_all, 'H': g_all * np.cos(tilt_all), 'K': g_all * (1.0 - np.cos(tilt_all))}
    t = t_all[inside]
    local = {name: demodulate(t, running_mean_highpass(t_all, x, period)[inside], period) for name, x in raw.items()}
    rms = {name: _rms(z) for name, z in local.items()}
    if not (rms['e'] > NUMERICAL_ZERO and rms['G'] > NUMERICAL_ZERO):
        result.reason = 'flat_series'  # e does not move at P: nothing to trade (a frozen i with a moving e is none)
        return result
    cross = np.sum(np.conj(local['e']) * local['i'])
    result.phase_deg = float(np.degrees(np.angle(cross)) % 360.0)
    coherence = float(np.abs(cross) / np.sqrt(np.sum(np.abs(local['e']) ** 2) * np.sum(np.abs(local['i']) ** 2)))
    result.coherence = coherence * float(np.sign(np.cos(np.radians(result.phase_deg))))
    result.amp_e, result.amp_i_deg = rms['e'] * np.sqrt(2.0), float(np.degrees(rms['i'] * np.sqrt(2.0)))
    e_mean, i_mean = float(np.mean(e_all[inside])), float(np.mean(tilt_all[inside]))
    cos_i = np.cos(i_mean)
    law_h = -e_mean / np.tan(i_mean) / (1.0 - e_mean**2) if np.tan(i_mean) != 0 else NAN
    result.ratio_h = float(rms['i'] / rms['e'] / abs(law_h)) if np.isfinite(law_h) and law_h != 0 else NAN
    result.eta_h = float(rms['H'] / abs(cos_i * rms['G'])) if cos_i != 0 else NAN
    result.eta_k = float(rms['K'] / abs((1.0 - cos_i) * rms['G'])) if cos_i != 1 else NAN
    if not all(np.isfinite(v) for v in (result.phase_deg, result.coherence, result.eta_h, result.eta_k)):
        result.reason = 'law_undefined'
        return result
    tolerance, coherent, residual = result.phase_tol_deg, result.min_coherence, result.h_residual_max
    if phase_off_deg(result.phase_deg, 180.0) <= tolerance and result.coherence <= -coherent and result.eta_h <= residual:
        result.coupling = 'zlk'
    elif phase_off_deg(result.phase_deg, 0.0) <= tolerance and result.coherence >= coherent and result.eta_k <= residual:
        result.coupling = 'forced_beat'
    else:
        result.coupling = 'none'
    return result
