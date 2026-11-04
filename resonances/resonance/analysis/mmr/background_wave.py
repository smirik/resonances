"""W31: is a small accepted episode a libration of the angle, or the same forced wave the record carries everywhere?

Below MMR_A_RESPONSE_MIN_AMP_DEG a forced wave of a (sigma_dot ~ -(omega^2/c)(sigma - c)) and a pendulum
(-(omega0^2/c) sin(sigma - c)) have the same sign, phase and a-participation: the episode alone cannot tell them
apart. Its background can. A libration adds its own swing to sigma and to a while it lasts and starts and ends at a
separatrix; a forced wave runs on unchanged before and after it (Nesvorny & Morbidelli 1998: short-period a
oscillations forced by a neighbouring resonance are removed before reading a three-body angle). So the episode is a
forced wave when, at its own period P, the envelopes of sigma and of a inside it lie within the band the same wave
shows outside it (`measure_background_wave`). The author asked for a measured criterion (W30 page, group three B,
2026-11-03: 2012 KL9 "рябь a-волны на дрейфе — не эпизод"). Pure: no decision is taken here; `policy` reads it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.signal import hilbert

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.geometry import Episode
from resonances.resonance.analysis.settings import AnalysisSettings

NAN = float('nan')
FORCED, OWN, NOT_MEASURED = 'forced', 'own', 'not_measured'
COHERENCE_WINDOW_PERIODS = 3.0  # the windows the steadiness of the background wave is read in (a measurement scale)


@dataclass(frozen=True)
class BackgroundWave:
    """The background test of one accepted episode (episodes.csv `bw_*`). outcome forced | own | not_measured; reason
    of not_measured: no_period, short_reference, incoherent, large_background. sigma_hi = the median sigma envelope
    inside over the outside quantile q (MMR_FORCED_WAVE_ENVELOPE_QUANTILE), sigma_lo = the same over the outside 1 - q,
    a_hi = the median a envelope inside over the outside quantile q; reference_periods = the outside samples in P."""

    episode_id: int
    outcome: str = NOT_MEASURED
    reason: str = ''
    sigma_hi: float = NAN
    sigma_lo: float = NAN
    a_hi: float = NAN
    reference_periods: float = NAN
    coherence_r2: float = NAN

    @property
    def forced(self) -> bool:
        return self.outcome == FORCED

    def thin(self) -> bool:
        """A forced outcome that needed the CONFIDENCE_THIN_BAND widening: inside the episode sigma or a above the
        background's q quantile, or sigma below its 1 - q quantile."""
        return bool(self.forced and (self.sigma_hi > 1.0 or self.a_hi > 1.0 or self.sigma_lo < 1.0))


def _envelope(band: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(band - np.mean(band)))


def _sinusoid_r2(t: np.ndarray, x: np.ndarray, period: float) -> float:
    """R^2 of one sinusoid at `period` (with an offset) fitted to `x`."""
    if len(x) < 6:
        return NAN
    w = 2.0 * math.pi / period
    design = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    total = float(np.sum((x - np.mean(x)) ** 2))
    return float(1.0 - np.sum((x - design @ coef) ** 2) / total) if total > 0 else NAN


def _coherence_r2(t: np.ndarray, x: np.ndarray, mask: np.ndarray, period: float) -> float:
    """How steady the background wave is: the median R^2 of one sinusoid at `period` over windows of
    COHERENCE_WINDOW_PERIODS periods (half overlapping) that lie wholly in `mask`; a slow drift of the wave's phase
    over the whole reference is no unsteadiness, so each window fits its own phase."""
    length, values = COHERENCE_WINDOW_PERIODS * period, []
    start = float(t[mask][0]) if mask.any() else NAN
    stop = float(t[mask][-1]) if mask.any() else NAN
    while math.isfinite(start) and start + length <= stop:
        inside = (t >= start) & (t < start + length)
        if inside.any() and mask[inside].all():
            r2 = _sinusoid_r2(t[inside], x[inside], period)
            if math.isfinite(r2):
                values.append(r2)
        start += length / 2
    return float(np.median(values)) if values else NAN


def measure_background_wave(
    t_yr: np.ndarray,
    sigma: np.ndarray,
    a: np.ndarray,
    episode: Episode,
    accepted: Sequence[Episode],
    window: Tuple[float, float],
    settings: AnalysisSettings,
) -> BackgroundWave:
    """The background test of `episode` (accepted, below MMR_A_RESPONSE_MIN_AMP_DEG; the caller selects it):

    1. sigma (unwrapped, before filtering) and a band-passed at [P / f, f P], f = MMR_OWNER_BAND_FACTOR, P the
       episode's median period;
    2. the reference: within MMR_FORCED_WAVE_REFERENCE_PERIODS x P of the episode, inside the edge-trimmed window,
       outside every accepted run +- P/2; at least MMR_FORCED_WAVE_MIN_REFERENCE_PERIODS periods, else not_measured;
    3. one sinusoid at P explains R^2 >= MMR_O1_COHERENCE_R2_MIN of the background sigma in windows of three periods
       (median; a steady wave), else
       not_measured; the background's sigma envelope stays below the small-amplitude limit at its quantile q, else
       not_measured (no small wave to compare with);
    4. forced when the median sigma envelope inside lies in [(1 - b) q_{1-q}, (1 + b) q_q] of the outside envelope and
       the median a envelope inside is <= (1 + b) q_q of the outside (b = CONFIDENCE_THIN_BAND); otherwise own."""
    eid = episode.episode_id
    period = float(episode.period_med_yr)
    if not (math.isfinite(period) and period > 0):
        return BackgroundWave(eid, NOT_MEASURED, 'no_period')
    t = np.asarray(t_yr, dtype=float)
    factor, order = settings.mmr_owner_band_factor, settings.filter_order
    sigma_band = filtering.band_pass(t, np.asarray(sigma, dtype=float), period, factor, order)
    a_band = filtering.band_pass(t, np.asarray(a, dtype=float), period, factor, order)
    lo, hi = window
    reach = settings.mmr_forced_wave_reference_periods * period
    near = (t >= max(lo, episode.t_start_yr - reach)) & (t <= min(hi, episode.t_end_yr + reach))
    outside = near.copy()
    for run in accepted:
        outside &= ~((t >= run.t_start_yr - period / 2) & (t <= run.t_end_yr + period / 2))
    inside = (t >= episode.t_start_yr) & (t <= episode.t_end_yr)
    step = float(np.median(np.diff(t))) if len(t) > 1 else NAN
    reference = float(np.count_nonzero(outside) * step / period) if math.isfinite(step) else NAN
    if not (math.isfinite(reference) and reference >= settings.mmr_forced_wave_min_reference_periods) or not inside.any():
        return BackgroundWave(eid, NOT_MEASURED, 'short_reference', reference_periods=reference)
    r2 = _coherence_r2(t, sigma_band, outside, period)
    if not (math.isfinite(r2) and r2 >= settings.mmr_o1_coherence_r2_min):
        return BackgroundWave(eid, NOT_MEASURED, 'incoherent', reference_periods=reference, coherence_r2=r2)
    q = settings.mmr_forced_wave_envelope_quantile
    sigma_env, a_env = _envelope(sigma_band), _envelope(a_band)
    out_sigma, out_a = sigma_env[outside], a_env[outside]
    s_top, s_low, a_top = float(np.quantile(out_sigma, q)), float(np.quantile(out_sigma, 1.0 - q)), float(np.quantile(out_a, q))
    if s_top >= math.radians(settings.mmr_a_response_min_amp_deg):
        return BackgroundWave(eid, NOT_MEASURED, 'large_background', reference_periods=reference, coherence_r2=r2)
    s_in, a_in = float(np.median(sigma_env[inside])), float(np.median(a_env[inside]))
    sigma_hi = s_in / s_top if s_top > 0 else NAN
    sigma_lo = s_in / s_low if s_low > 0 else NAN
    a_hi = a_in / a_top if a_top > 0 else NAN
    band = settings.confidence_thin_band
    forced = bool(
        math.isfinite(sigma_hi)
        and math.isfinite(sigma_lo)
        and math.isfinite(a_hi)
        and sigma_hi <= 1.0 + band
        and sigma_lo >= 1.0 - band
        and a_hi <= 1.0 + band
    )
    return BackgroundWave(eid, FORCED if forced else OWN, '', sigma_hi, sigma_lo, a_hi, reference, r2)


def background_waves(
    t_yr: np.ndarray, sigma: np.ndarray, a: np.ndarray, runs: Sequence[Episode], window, settings: AnalysisSettings, coorbital: bool = False
) -> Tuple[BackgroundWave, ...]:
    """The test on every accepted run below MMR_A_RESPONSE_MIN_AMP_DEG of a non-1:1 angle (the 1:1 law is measured at
    any amplitude); () with MMR_FORCED_WAVE_BACKGROUND off. A run measured forced is part of the background: the other
    runs are measured again with it in their reference, until no new run is forced (W32 redundancy review: 2012 DE61
    ep 4, whose reference lay between the forced runs 3 and 5)."""
    if not settings.mmr_forced_wave_background or coorbital:
        return ()
    accepted = [run for run in runs if run.accepted]
    limit = math.radians(settings.mmr_a_response_min_amp_deg)
    small = [run for run in accepted if math.isfinite(run.half_amp_max_rad) and run.half_amp_max_rad < limit]
    waves = {run.episode_id: measure_background_wave(t_yr, sigma, a, run, accepted, window, settings) for run in small}
    forced, added = forced_ids(waves.values()), True
    while added:  # each pass adds at least one forced run or ends: at most len(small) passes
        masked = [run for run in accepted if run.episode_id not in forced]
        again = {
            run.episode_id: measure_background_wave(t_yr, sigma, a, run, masked + [run], window, settings)
            for run in small
            if run.episode_id not in forced
        }
        waves.update(again)
        added = forced_ids(again.values())
        forced = forced | added
    return tuple(waves[run.episode_id] for run in small)


def forced_ids(waves: Optional[Sequence[BackgroundWave]]) -> frozenset:
    return frozenset(w.episode_id for w in waves or () if w.forced)
