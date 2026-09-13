"""Cross-spectral coherence between two time series of a simulated body.

Answers the question "do these two quantities share a frequency, and with what
phase shift?" — e.g. does the resonant angle share the libration frequency with
the semi-major axis (MMR), or do e and i exchange in antiphase (Lidov-Kozai).

The estimator is Welch's method (`scipy.signal.coherence` + `scipy.signal.csd`)
on **non-overlapping** segments. Non-overlapping is deliberate: the significance
threshold

    gamma2_crit = 1 - alpha ** (1 / (n_seg - 1))

is exact only when the segments are statistically independent. With 50% overlap
the same threshold is roughly 100x too conservative (measured false-alarm rate
1e-4 instead of the nominal alpha = 1e-2).

Two limits bound what can be believed:

    P_min = 4 * dt          — safety margin over Nyquist
    P_max = T / (5 * n_seg) — at least 5 cycles inside every segment

Because P_max shrinks as n_seg grows, several segmentation levels are scanned
(`n_segments`, default 4/8/16) and the verdict is taken from the coarsest level
whose validity band is non-empty: it reaches the longest periods and carries the
strictest threshold. Levels that cannot be computed (too few samples per segment)
are reported as such rather than silently skipped.

Contiguous runs of significant bins are grouped into *lines*; lines are ranked by
integrated cross-amplitude sum(|Pxy|), and the phase criterion is applied to the
top one. Ranking by coherence alone does not work: a weak noise line can reach
gamma2 ~ 0.95 and outrank the physically dominant term.

Phase convention: `phase_deg > 0` means the SECOND series leads the first.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from resonances.logger import logger

# Series that can take part in a pair. 'sigma' is the raw (unfiltered) unwrapped
# resonant angle; the others are the osculating elements, used as they are.
SERIES_KEYS = ('sigma', 'a', 'e', 'i')

# Welch needs a segment long enough for the FFT to resolve anything at all.
MIN_NPERSEG = 512
# Coherence is identically 1 for a single segment — that is an algebraic identity,
# not a measurement.
MIN_SEGMENTS = 2

# Default pairs and phase targets per resonance type. A phase target is
# (target_deg, tolerance_deg); the criterion passes when the circular distance
# between the measured phase and the target is within the tolerance.
DEFAULT_PAIRS: Dict[str, List[str]] = {
    'lidov_kozai': ['e-i'],
    'mmr': ['sigma-a'],
    'secular': ['sigma-e'],
}

DEFAULT_PHASE_TARGETS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # ZLK exchanges e for i at constant H, so the two are in antiphase.
    'lidov_kozai': {'e-i': (180.0, 30.0)},
    # da/dt ~ dR/dsigma puts the axis a quarter period ahead of the angle.
    'mmr': {'sigma-a': (90.0, 30.0)},
    'secular': {},
}

# Longest period that is physically interesting per resonance type. Without a
# lower bound on the period, short-period terms flood the ZLK search band with
# real but irrelevant lines.
DEFAULT_PERIOD_MIN: Dict[str, Optional[float]] = {
    'lidov_kozai': 10_000.0,
    'mmr': None,
    'secular': 10_000.0,
}


@dataclass
class CoherenceLine:
    """One coherent spectral line: a contiguous run of significant frequency bins."""

    period: float  # years, at the bin with the largest |Pxy| in the run
    frequency: float  # 1/years
    gamma2: float  # magnitude-squared coherence at that bin
    phase_deg: float  # circular mean phase over the run; > 0 means y leads x
    phase_std: float  # circular spread of the phase over the run, degrees
    amp_ratio: float  # sqrt(Pyy/Pxx) at that bin — how much y moves per unit of x
    cross_amplitude: float  # sum(|Pxy|) over the run; the ranking key
    n_bins: int
    n_seg: int
    # Time-domain cross-check of the same claim, filled for the strongest line only
    # (see bandpassed_correlation). NaN on the others: it costs two FFTs per line.
    r_band: float = float('nan')


@dataclass
class CrossSpectrum:
    """Result of one (pair, n_seg) cross-spectral estimate.

    `valid` is False when the estimate could not be made at all (non-uniform grid,
    too few samples, empty validity band). `lines` may legitimately be empty on a
    valid estimate — that means "no coherent line", which is a real answer.
    """

    pair: str
    n_seg: int
    valid: bool
    reason: str = ''
    nperseg: int = 0
    gamma2_crit: float = np.nan
    period_min: float = np.nan
    period_max: float = np.nan
    frequency: Optional[np.ndarray] = None
    coherence: Optional[np.ndarray] = None
    phase_deg: Optional[np.ndarray] = None
    in_band: Optional[np.ndarray] = None  # boolean mask of bins inside [period_min, period_max]
    lines: List[CoherenceLine] = field(default_factory=list)

    @property
    def top_line(self) -> Optional[CoherenceLine]:
        """Strongest line by integrated cross-amplitude, or None if there is none."""
        return self.lines[0] if self.lines else None


def parse_pair(pair: str) -> Tuple[str, str]:
    """Split a pair identifier such as 'e-i' into its two series keys.

    Pairs are strings rather than tuples so that they can be used as dict keys in
    `simulation.json` (JSON object keys cannot be tuples).
    """
    parts = pair.split('-')
    if len(parts) != 2 or any(p not in SERIES_KEYS for p in parts):
        raise ValueError(f"Invalid coherence pair '{pair}'. Expected '<x>-<y>' with x, y in {SERIES_KEYS}.")
    return parts[0], parts[1]


def phase_distance(phase_deg: float, target_deg: float) -> float:
    """Smallest absolute circular distance between two angles, in degrees [0, 180]."""
    return abs((phase_deg - target_deg + 180.0) % 360.0 - 180.0)


def phase_matches(phase_deg: float, target_deg: float, tolerance_deg: float) -> bool:
    """Whether a measured phase sits within `tolerance_deg` of `target_deg`."""
    return phase_distance(phase_deg, target_deg) <= tolerance_deg


def _circular_mean_std(angles_deg: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Weighted circular mean and spread of a set of angles, in degrees."""
    rad = np.radians(angles_deg)
    if weights is None:
        weights = np.ones_like(rad)
    total = weights.sum()
    if total <= 0:
        return float(np.degrees(rad[0])), 0.0
    c = float(np.sum(weights * np.cos(rad)) / total)
    s = float(np.sum(weights * np.sin(rad)) / total)
    mean = float(np.degrees(np.arctan2(s, c)))
    r = min(max(np.hypot(c, s), 1e-12), 1.0)
    # abs() keeps a perfectly concentrated set from reporting -0.0.
    return mean, abs(float(np.degrees(np.sqrt(-2.0 * np.log(r)))))


BAND_WIDTH = 1.3  # multiplicative half-width of the band-pass around a line


def bandpass(series: np.ndarray, dt: float, frequency: float, width: float = BAND_WIDTH) -> np.ndarray:
    """Keep only the content of `series` within a factor `width` of `frequency`."""
    spectrum = np.fft.rfft(series - series.mean())
    freqs = np.fft.rfftfreq(len(series), d=dt)
    spectrum[(freqs < frequency / width) | (freqs > frequency * width)] = 0.0
    return np.fft.irfft(spectrum, n=len(series))


def bandpassed_correlation(x: np.ndarray, y: np.ndarray, dt: float, frequency: float, width: float = BAND_WIDTH) -> float:
    """Pearson correlation of `x` and `y` band-passed around `frequency`.

    An independent, time-domain reading of what the cross-phase claims: two series in
    antiphase at that frequency correlate at -1, in phase at +1, in quadrature at 0.

    It matters because the cross-phase is read off a single frequency bin, and a bin can
    be misleading when several close lines are blended into it — a real case on asteroid
    591986, where an unresolved 53.9/54.0/54.2 kyr triplet reports antiphase on a 5 Myr
    window and +124 degrees on longer ones. The band-passed correlation uses the whole
    window and does not flip with the baseline.
    """
    band_x, band_y = bandpass(x, dt, frequency, width), bandpass(y, dt, frequency, width)
    if band_x.std() == 0 or band_y.std() == 0:
        return float('nan')
    return float(np.corrcoef(band_x, band_y)[0, 1])


def is_uniform_grid(times: np.ndarray, rtol: float = 1e-4) -> bool:
    """Whether the sampling grid is uniform to within `rtol` of the median step."""
    if len(times) < 3:
        return False
    steps = np.diff(times)
    dt = float(np.median(steps))
    return bool(dt > 0 and np.ptp(steps) <= rtol * dt)


def cross_spectrum(
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_seg: int,
    pair: str = '',
    alpha: float = 0.01,
    period_min: Optional[float] = None,
    period_max: Optional[float] = None,
    window: str = 'hann',
    max_lines: Optional[int] = None,
    psd_cache: Optional[dict] = None,
) -> CrossSpectrum:
    """Estimate the cross spectrum of `x` and `y` on a uniform time grid.

    Parameters
    ----------
    times : np.ndarray
        Sampling times in years. Must be uniform.
    x, y : np.ndarray
        The two series. Angles must already be unwrapped by the caller; a, e and i
        must not be.
    n_seg : int
        Number of non-overlapping Welch segments. Sets both the resolution and the
        significance threshold.
    pair : str
        Identifier such as 'e-i', carried through into the result for reporting.
    alpha : float
        Per-bin false-alarm probability for the coherence threshold.
    period_min, period_max : float, optional
        Extra narrowing of the search band, in years. The intrinsic limits
        4*dt and T/(5*n_seg) always apply on top of these.
    max_lines : int, optional
        Keep only this many strongest lines. Dropped lines are logged.
    psd_cache : dict, optional
        Reused across pairs of the same body to avoid recomputing Welch spectra of
        shared series. Keyed by (id(array), nperseg, window).
    """
    n = len(x)
    result = CrossSpectrum(pair=pair, n_seg=n_seg, valid=False)

    if n_seg < MIN_SEGMENTS:
        result.reason = f"n_seg={n_seg} < {MIN_SEGMENTS}; coherence is identically 1 for a single segment"
        return result
    if not (n == len(y) == len(times)):
        # Silently mismatched lengths would yield a plausible but wrong period scale.
        result.reason = f"length mismatch: times={len(times)}, x={n}, y={len(y)}"
        return result
    if not is_uniform_grid(times):
        result.reason = "sampling grid is not uniform"
        return result

    nperseg = int(n // n_seg)
    if nperseg < MIN_NPERSEG:
        result.reason = f"nperseg={nperseg} < {MIN_NPERSEG}; not enough samples per segment"
        return result
    result.nperseg = nperseg

    dt = float(np.median(np.diff(times)))
    total_time = float(times[-1] - times[0])
    p_lo = 4.0 * dt if period_min is None else max(4.0 * dt, period_min)
    p_hi = total_time / (5.0 * n_seg) if period_max is None else min(total_time / (5.0 * n_seg), period_max)
    result.period_min, result.period_max = p_lo, p_hi
    result.gamma2_crit = 1.0 - alpha ** (1.0 / (n_seg - 1))

    if p_hi <= p_lo:
        result.reason = f"empty validity band: P_min={p_lo:.4g} yr >= P_max={p_hi:.4g} yr (baseline too short)"
        return result

    # coherence, csd and welch must share identical parameters, otherwise their
    # frequency grids stop lining up index by index.
    kwargs = dict(fs=1.0 / dt, window=window, nperseg=nperseg, noverlap=0, detrend='linear')
    try:
        frequency, coherence_xy = signal.coherence(x, y, **kwargs)
        _, pxy = signal.csd(x, y, **kwargs)
        pxx = _welch_cached(x, kwargs, psd_cache)
        pyy = _welch_cached(y, kwargs, psd_cache)
    except Exception as e:  # pragma: no cover - scipy failures are environment-specific
        logger.error(f"Cross spectrum failed for pair '{pair}' (n_seg={n_seg}): {e}")
        result.reason = f"scipy failure: {e}"
        return result

    with np.errstate(divide='ignore'):
        period = np.where(frequency > 0, 1.0 / np.where(frequency > 0, frequency, 1.0), np.inf)
    in_band = (frequency > 0) & (period >= p_lo) & (period <= p_hi)

    result.valid = True
    result.frequency = frequency
    result.coherence = coherence_xy
    result.phase_deg = np.degrees(np.angle(pxy))
    result.in_band = in_band
    result.lines = _extract_lines(
        n_seg=n_seg,
        period=period,
        frequency=frequency,
        coherence_xy=coherence_xy,
        phase=result.phase_deg,
        pxy=pxy,
        pxx=pxx,
        pyy=pyy,
        significant=in_band & (coherence_xy >= result.gamma2_crit),
    )

    if max_lines is not None and len(result.lines) > max_lines:
        logger.info(f"Cross spectrum '{pair}' (n_seg={n_seg}): keeping {max_lines} of {len(result.lines)} coherent lines")
        result.lines = result.lines[:max_lines]

    if result.lines:
        result.lines[0].r_band = bandpassed_correlation(x, y, dt, result.lines[0].frequency)

    return result


def _welch_cached(series: np.ndarray, kwargs: dict, cache: Optional[dict]) -> np.ndarray:
    """Welch power spectrum, memoised per (array, segmentation) when a cache is given.

    A body's a, e and i are shared by all of its resonances, so without the cache the
    same spectra get recomputed for every resonance and every pair. The cached entry
    keeps a reference to the array so that its id() cannot be recycled by another
    object while the cache is alive.
    """
    if cache is None:
        return signal.welch(series, **kwargs)[1]
    key = (id(series), kwargs['nperseg'], kwargs['window'])
    if key not in cache:
        cache[key] = (series, signal.welch(series, **kwargs)[1])
    return cache[key][1]


def _extract_lines(
    n_seg: int,
    period: np.ndarray,
    frequency: np.ndarray,
    coherence_xy: np.ndarray,
    phase: np.ndarray,
    pxy: np.ndarray,
    pxx: np.ndarray,
    pyy: np.ndarray,
    significant: np.ndarray,
) -> List[CoherenceLine]:
    """Group contiguous significant bins into lines, ranked by integrated cross-amplitude."""
    indices = np.where(significant)[0]
    if len(indices) == 0:
        return []

    lines = []
    amplitude = np.abs(pxy)
    for group in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1):
        # Represent the group by its strongest bin, not by its most coherent one:
        # wide groups can be coherent across a whole band while the physically
        # meaningful frequency is the one carrying the power.
        peak = group[np.argmax(amplitude[group])]
        mean_phase, std_phase = _circular_mean_std(phase[group], amplitude[group])
        lines.append(
            CoherenceLine(
                period=float(period[peak]),
                frequency=float(frequency[peak]),
                gamma2=float(coherence_xy[peak]),
                phase_deg=mean_phase,
                phase_std=std_phase,
                amp_ratio=float(np.sqrt(pyy[peak] / pxx[peak])) if pxx[peak] > 0 else float('nan'),
                cross_amplitude=float(amplitude[group].sum()),
                n_bins=int(len(group)),
                n_seg=n_seg,
            )
        )

    lines.sort(key=lambda line: -line.cross_amplitude)
    return lines


def cross_spectrum_levels(
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_segments: Sequence[int] = (4, 8, 16),
    **kwargs,
) -> List[CrossSpectrum]:
    """Estimate the cross spectrum at several segmentation levels, coarsest first."""
    return [cross_spectrum(times, x, y, n_seg, **kwargs) for n_seg in sorted(n_segments)]


def decisive_level(levels: Sequence[CrossSpectrum]) -> Optional[CrossSpectrum]:
    """The level a verdict should be based on: the coarsest one that is valid.

    Coarsest means the fewest segments, hence the longest reachable period and the
    strictest coherence threshold. Levels are not skipped for merely having found
    no lines — "no line at the most permissive period range" is itself the answer.
    """
    for level in sorted(levels, key=lambda lvl: lvl.n_seg):
        if level.valid:
            return level
    return None


def levels_agreeing(levels: Sequence[CrossSpectrum], line: CoherenceLine) -> int:
    """How many valid levels reproduce `line` as their own strongest line.

    Levels have different frequency grids, so lines are matched within 1.5 bin
    widths of the level being tested rather than by exact frequency.
    """
    agreeing = 0
    for level in levels:
        top = level.top_line
        if top is None or level.frequency is None or len(level.frequency) < 2:
            continue
        df = float(level.frequency[1] - level.frequency[0])
        if abs(top.frequency - line.frequency) <= 1.5 * df:
            agreeing += 1
    return agreeing


def eccentricity_vector(ecc: np.ndarray, omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Non-singular eccentricity components k = e cos(omega), h = e sin(omega)."""
    return ecc * np.cos(omega), ecc * np.sin(omega)


def forced_free_eccentricity(ecc: np.ndarray, omega: np.ndarray) -> Tuple[float, float]:
    """Split the eccentricity into its forced and free parts.

    In the (k, h) plane the forced term is the offset of the centroid from the
    origin and the free term is the radius the vector traces around it:

        e_forced = |(<k>, <h>)|
        e_free   = sqrt(var(k) + var(h))

    The sum of the two variances is what makes e_free the radius: a free vector of
    radius R sweeping the circle uniformly puts R^2/2 into each coordinate.
    """
    k, h = eccentricity_vector(ecc, omega)
    return float(np.hypot(k.mean(), h.mean())), float(np.sqrt(k.var() + h.var()))
