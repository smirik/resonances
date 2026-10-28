"""Response of a to the angle and the force law (SPEC §4.5).

Window: the union of the accepted episodes (libration, episodic), otherwise the analysis window.
With an aliased step (aliased_unambiguous, ambiguous) sigma_dot is the model rate, a function of a
by construction: identity and force law are not_applicable there. With `filter_too_slow` (the cycle
shorter than FILTER_MIN_LIBRATION_CUTOFFS x P_c) the shape checks (identity, force law, B1) are
not_measured; their numbers are still written.

Conventions. The force law a_dot_f = R sin(sigma_f - phi0) + C has phi0 = atan2(-B, A) for
A sin sigma + B cos sigma; sigma_ddot = c a_dot puts the stable point at phi0 when c < 0 and at
phi0 + pi when c > 0 (most catalogue angles have k < 0, c > 0). The identity correlation is taken
in the direction of c (corr x sign c), so that 1 means 'sigma_dot follows c (a - a*)'.

The derivative-based quantities (identity, force law, multiplet, C4 rate) are measured on the series smoothed
again at max(P_c, MMR_RESPONSE_CUTOFF_CYCLES x P_cyc) (`response_cutoff_yr`): the record cutoff P_c is chosen for
sigma, and a derivative of what leaks through it swamps the resonant a_dot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.filtering import circular_mean, r2
from resonances.resonance.analysis.filtering import circular_resultant as circular_coherence  # |<e^{i phi}>|, nan for none
from resonances.resonance.analysis.geometry import (
    Episode,
    Geometry,
    gap_circulates,
    half_amplitude_values,
    half_amplitudes,
    well_shift,
    wrap_pi,
)
from resonances.resonance.analysis.mmr import coorbital as coorbital_module
from resonances.resonance.analysis.mmr import forced_lines as forced_lines_module
from resonances.resonance.analysis.mmr import strength as strength_module
from resonances.resonance.analysis.mmr.context import mean_motion, table_mean_motion
from resonances.resonance.analysis.mmr.coorbital import LAW_COORBITAL, CoorbitalInput, CoorbitalLaw
from resonances.resonance.analysis.mmr.preprocess import coorbital_inputs, coorbital_planet, filter_guard, law_planet
from resonances.resonance.analysis.mmr.models import (
    Context,
    EpisodePhysics,
    EpisodeResponse,
    ForceWindow,
    HarmonicLaw,
    harmonic_reason,
    MMRSeries,
    MultipletFit,
    Preprocessed,
    ResonanceSpec,
    Response,
    ResponseSource,
    Sampling,
    SlowComponents,
    StepVerdict,
)
from resonances.resonance.analysis.pendulum import (
    CycleClass,
    EnergyAnalysis,
    excursion,
    half_width,
    small_period,
)
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

NAN = float('nan')
TWO_PI = 2 * np.pi
ALIASED = (StepVerdict.ALIASED_UNAMBIGUOUS, StepVerdict.AMBIGUOUS)
LAW_SINGLE = 'single'
LAW_MULTIPLET = 'multiplet'
LAW_HARMONIC = 'harmonic'  # CS12: N harmonics of the angle (`harmonic_law`)
ROUTE_DATA, ROUTE_GEOMETRY, ROUTE_OWNER, ROUTE_NONE = 'data', 'geometry', 'owner', 'none'
REVOLUTION = 1.0  # a centre pass of one revolution is a circulation of the centre (its definition, not a threshold)
MIN_FIT_SAMPLES = 4  # a three-parameter least-squares fit needs more points than parameters
ONSET_BAND_FACTOR = 2  # O1's band [P / 2, 2 P] (`onset_bands`)
# a window whose normal matrix has det / (product of its diagonal) below this (Hadamard ratio, 1 for orthogonal columns,
# 0 for dependent ones) is solved by lstsq: numerical safety, not a threshold of a decision
SINE_FIT_HADAMARD_MIN = 1.0e-10
LINE_GRID_OVERSAMPLE = 4  # zero padding of a_dot's amplitude spectrum: a peak within 3 % of its height (numerical)
# a window of the N-harmonic law whose normal matrix has a condition number above this is solved by lstsq instead
# (numerical safety, not a threshold of a decision)
HARMONIC_FIT_COND_MAX = 1.0e12
# independent halves of the N-harmonic law: an angle counts as visited by the fitting half when its bin of this width
# holds one of that half's samples (the resolution of 'visited', a definition)
VISITED_BIN_DEG = 2.0
# why the N-harmonic law was not taken (`HarmonicLaw.reason`), in the order the guards are read
HARMONIC_NO_GAIN = 'no_gain'  # N = 2 gains < mmr_force_multi_gain over one harmonic
HARMONIC_NULL_UNMEASURED = 'null_not_measured'  # no shifted null with enough fully covered windows
HARMONIC_INCOHERENT = 'incoherent'  # the leading C_k turn from window to window
HARMONIC_BELOW_NULL = 'below_null'  # R^2 - null < mmr_force_null_gain
HARMONIC_BELOW_FAIL = 'below_fail'  # R^2 < mmr_force_r2_fail
HARMONIC_HALVES = 'halves_disagree'  # one half's law does not predict the other


# --- small numerical helpers --------------------------------------------------------------------


def _finite(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; nan below three samples or for a constant series."""
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return NAN
    return float(np.corrcoef(x, y)[0, 1])


def _fit(design: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef, r2(target, target - design @ coef)


def sine_coefficients(angle: np.ndarray, target: np.ndarray) -> Tuple[float, float, float, float]:
    """target = A sin(angle) + B cos(angle) + C by least squares: (A, B, C, R^2); K = hypot(A, B), phi0 =
    arctan2(-B, A). No sample guard: each caller keeps its own."""
    coef, r2 = _fit(np.column_stack([np.sin(angle), np.cos(angle), np.ones_like(angle)]), target)
    return float(coef[0]), float(coef[1]), float(coef[2]), r2


def _sine_fit(angle: np.ndarray, target: np.ndarray) -> Tuple[float, float, float, float]:
    """`sine_coefficients` on the finite samples; all nan below MIN_FIT_SAMPLES of them."""
    mask = _finite(angle, target)
    if np.count_nonzero(mask) < MIN_FIT_SAMPLES:
        return NAN, NAN, NAN, NAN
    return sine_coefficients(angle[mask], target[mask])


def members_fit(angle: np.ndarray, offset: np.ndarray, target: np.ndarray, harmonics: Sequence[int]) -> Tuple[np.ndarray, float]:
    """target = sum_j [A_j sin sigma_j + B_j cos sigma_j] + C, sigma_j = angle + j offset, by least squares:
    (C_j = A_j + i B_j in the order of `harmonics`, R^2). No sample guard: each caller keeps its own."""
    n_terms = len(harmonics)
    design = np.empty((len(target), 2 * n_terms + 1))
    for i, j in enumerate(harmonics):
        member = angle + j * offset
        design[:, 2 * i], design[:, 2 * i + 1] = np.sin(member), np.cos(member)
    design[:, -1] = 1.0
    coef, r2 = _fit(design, target)
    return coef[0 : 2 * n_terms : 2] + 1j * coef[1 : 2 * n_terms : 2], r2


def intervals_mask(t_yr: np.ndarray, intervals: Sequence[Tuple[float, float]]) -> np.ndarray:
    """True inside any of the closed `intervals` [start, end] (`samples_between`; a non-finite bound adds nothing)."""
    mask = np.zeros(len(t_yr), dtype=bool)
    for start, end in intervals:
        mask[samples_between(t_yr, start, end)] = True
    return mask


def samples_between(t_yr: np.ndarray, start: float, end: float) -> slice:
    """The samples of an increasing `t_yr` inside [start, end] as a slice (a mask over the whole record per window cost
    O(N x windows)); empty for a non-finite bound."""
    if not (np.isfinite(start) and np.isfinite(end)):
        return slice(0, 0)
    return slice(int(np.searchsorted(t_yr, start, side='left')), int(np.searchsorted(t_yr, end, side='right')))


def sliding_windows(t_yr: np.ndarray, length_yr: float, step_fraction: float) -> List[Tuple[float, float]]:
    """Windows [t0, t0 + L] stepping by step_fraction L inside [t_yr[0], t_yr[-1]] (the last may end half a step late)."""
    if len(t_yr) < 2 or not (np.isfinite(length_yr) and length_yr > 0 and step_fraction > 0):
        return []
    tolerance = np.median(np.diff(t_yr)) / 2
    windows, start = [], float(t_yr[0])
    while start + length_yr <= t_yr[-1] + tolerance:
        windows.append((start, start + length_yr))
        start += step_fraction * length_yr
    return windows


def _response_pieces(geometry: Geometry) -> List[Tuple[float, float]]:
    """The accepted episodes clipped to the analysis window, else the analysis window."""
    start, end = geometry.window
    pieces = [(max(e.t_start_yr, start), min(e.t_end_yr, end)) for e in geometry.episodes]
    pieces = [(a, b) for a, b in pieces if b > a]
    return pieces or [(start, end)]


def _amplitudes(geometry: Geometry) -> Tuple[float, float]:
    """(A_max, A_med) [rad] over the accepted episodes, else over every turning point of the window; nan without TPs."""
    if geometry.episodes:
        return (
            float(np.nanmax([e.half_amp_max_rad for e in geometry.episodes])),
            float(np.nanmedian([e.half_amp_med_rad for e in geometry.episodes])),
        )
    amplitudes = half_amplitudes(geometry.turning_points)
    if not len(amplitudes):
        return NAN, NAN
    return float(np.max(amplitudes)), float(np.median(amplitudes))


def multiplet_harmonics(body_varpi_coefficient: int, planet_varpi_coefficient: int) -> Tuple[int, ...]:
    """The members sigma_j = sigma + j (varpi - varpi_P) of the same order: body varpi coefficient p + j and
    planet p_P - j with |p + j| + |p_P - j| = |p| + |p_P| (j = 0 is the angle itself)."""
    p, p_planet = body_varpi_coefficient, planet_varpi_coefficient
    order = abs(p) + abs(p_planet)
    return tuple(j for j in range(-order, order + 1) if abs(p + j) + abs(p_planet - j) == order)


def response_cutoff_yr(cutoff_yr: float, period_cycle_yr: float, settings: AnalysisSettings) -> float:
    """The cutoff of the derivative-based response quantities [yr]: max(P_c, mmr_response_cutoff_cycles x P_cyc).

    P_c removes the synodic waves from sigma to <= 0.5 deg, but a derivative multiplies what leaks by P_cyc / P_syn:
    at P_c = 20 yr the residual 7-yr wave of the osculating a dominates a_dot 50 times over the resonant force of 24820,
    and the identity and force law fail on any record. Smoothing at a small fraction of the cycle keeps the cycle and its
    first harmonics (G = 0.94 at the 5th for 0.1) and removes that leak. Without a cycle: P_c.

    The fraction 0.1 (not 0.05): the slowest forced lines of the angle (T_k = |k_b| P_planet, e.g. 9.8 yr in 88710 9V-16)
    leak through 0.05 P_cyc and fail the force law of 16 of 78 measured harmonic episodes (8 at 0.1); the null test of
    shifted and reversed a (312 trials) is unchanged and an exact pendulum loses R^2 < 0.01 up to 178 deg. The whole
    record is filtered, then cut into episodes: a jump of a leaks into an episode within ~ one cutoff of it, less at 0.1
    than at 0.05 when the jump is at the first turning point (the derivative of the smoothed step has energy ~ jump^2 /
    cutoff), more only when it lies between the two reaches (R^2 0.99 -> 0.83 for a jump of 4 peak libration rates
    0.025 P_cyc before it); no check went from pass to fail on 288 synthetic captures of 6-8 cycles.
    """
    fraction = settings.mmr_response_cutoff_cycles
    if not (np.isfinite(period_cycle_yr) and period_cycle_yr > 0):
        return cutoff_yr
    return max(cutoff_yr, fraction * period_cycle_yr)


def response_series(
    prep: Preprocessed, period_cycle_yr: float, settings: AnalysisSettings
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """(sigma, sigma_dot, a, a_dot, cutoff) for the response: prep's filtered series, low-passed again at
    `response_cutoff_yr` when that is longer than P_c (sigma_dot and a_dot are derivatives of the smoothed series)."""
    return smoothed_series(prep, response_cutoff_yr(prep.cutoff.cutoff_yr, period_cycle_yr, settings), settings)


def smoothed_series(
    prep: Preprocessed, cutoff_yr: float, settings: AnalysisSettings
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """`response_series` at a given cutoff [yr]: prep's series low-passed again when the cutoff is longer than P_c and
    two steps. Remembered inside the record's `filtering.record_memo` (the arrays it filtered are read-only there)."""

    def compute():
        t = np.asarray(prep.t_yr, float)
        angle, rate, a, adot = (np.asarray(x, float) for x in (prep.angle_f, prep.rate_f, prep.a_f, prep.adot_f))
        cutoff = cutoff_yr
        if cutoff > prep.cutoff.cutoff_yr and cutoff > 2 * np.median(np.diff(t)) and np.all(np.isfinite(angle)) and np.all(np.isfinite(a)):
            angle = filtering.lowpass_with(t, angle, cutoff, settings)
            a = filtering.lowpass_with(t, a, cutoff, settings)
            rate, adot = filtering.time_derivative(t, angle), filtering.time_derivative(t, a)
        return angle, rate, a, adot, cutoff

    def fresh(result):  # the arrays made here, not prep's own
        return [x for x, own in zip(result[:4], (prep.angle_f, prep.rate_f, prep.a_f, prep.adot_f)) if x is not own]

    return filtering.memoised('smoothed_series', (prep, settings), (float(cutoff_yr),), compute, fresh)


def adot_lines(
    t_yr: np.ndarray, adot: np.ndarray, pieces: Sequence[Tuple[float, float]], period_yr: float, shortest_yr: float
) -> List[Tuple[float, float]]:
    """(period [yr], amplitude [au/yr]) of every spectral line of a_dot in each piece: the local maxima of the amplitude
    spectrum 2 |DFT| / N (the least-squares sine amplitude on a uniform grid, zero-padded LINE_GRID_OVERSAMPLE times)
    with a period in [shortest_yr, P/2], leaving out the libration's own harmonics k/P within 1/T of the piece (its
    resolution). The libration of period P puts a_dot = K sin sigma(t) at P and its harmonics; anything else there is a
    wave the force law cannot hold."""
    t_yr, adot = np.asarray(t_yr, float), np.asarray(adot, float)
    if not (np.isfinite(period_yr) and period_yr > 0 and np.isfinite(shortest_yr) and 0 < shortest_yr < period_yr / 2):
        return []
    lines = []
    for start, end in pieces:
        part = samples_between(t_yr, start, end)
        times, values = t_yr[part], adot[part]
        if len(values) < MIN_FIT_SAMPLES or not np.all(np.isfinite(values)) or times[-1] <= times[0]:
            continue
        span = float(times[-1] - times[0])
        n_fft = int(2 ** np.ceil(np.log2(LINE_GRID_OVERSAMPLE * len(values))))
        amplitude = 2 * np.abs(np.fft.rfft(values - np.mean(values), n_fft)) / len(values)
        frequency = np.fft.rfftfreq(n_fft, span / (len(values) - 1))
        band = (frequency >= 2 / period_yr) & (frequency <= 1 / shortest_yr)
        band &= np.abs(frequency - np.rint(frequency * period_yr) / period_yr) >= 1 / span
        peak = band[1:-1] & (amplitude[1:-1] >= amplitude[:-2]) & (amplitude[1:-1] >= amplitude[2:])
        lines += [(float(1 / frequency[i]), float(amplitude[i])) for i in np.flatnonzero(peak) + 1]
    return lines


def line_margin_cutoff(
    t_yr: np.ndarray,
    adot: np.ndarray,
    pieces: Sequence[Tuple[float, float]],
    period_yr: float,
    base_cutoff_yr: float,
    force_au_yr: float,
    settings: AnalysisSettings,
) -> Tuple[float, float]:
    """(cutoff [yr], binding line [yr]) of the harmonic force law on `pieces` read with period P (R3 §2, W20 F5).

    The base cutoff max(P_c, mmr_response_cutoff_cycles P) leaves in the lines of a_dot between it and P/2, and those
    just below it (the filter's slope; the derivative weights a line by 2 pi / its period): 2004 TK14 31M-34 keeps a
    64-yr line at 0.9 K under a 75-yr cutoff, 433 Eros 47M-44 one at 83 yr at 1.5 K, and the force law fails there (R^2
    0.03-0.16) while it holds above them. Each line of `adot` (the series of the first fit, `adot_lines` from
    mmr_response_line_band_cutoffs x the base cutoff) with an amplitude of at least mmr_response_line_amp_frac x K (the
    force of that fit) raises the cutoff to mmr_response_line_margin x its period, the longest such line binding,
    at most P / mmr_response_max_fraction. (base cutoff, nan) when no line raises it."""
    if not all(np.isfinite(x) and x > 0 for x in (period_yr, base_cutoff_yr, force_au_yr)):
        return base_cutoff_yr, NAN
    shortest = settings.mmr_response_line_band_cutoffs * base_cutoff_yr
    lines = adot_lines(t_yr, adot, pieces, period_yr, shortest)
    strong = [line for line, amplitude in lines if amplitude >= settings.mmr_response_line_amp_frac * force_au_yr]
    if not strong:
        return base_cutoff_yr, NAN
    binding = max(strong)
    cutoff = min(settings.mmr_response_line_margin * binding, period_yr / settings.mmr_response_max_fraction)
    return (cutoff, binding) if cutoff > base_cutoff_yr else (base_cutoff_yr, NAN)


# --- L3 identity --------------------------------------------------------------------------------


def identity(
    rate_f: np.ndarray, a_f: np.ndarray, identity_coefficient: float, settings: AnalysisSettings
) -> Tuple[float, float, CheckValue]:
    """L3: regression of sigma_dot_f on a_f; (slope / c, correlation, check): pass when |slope/c - 1| <=
    mmr_identity_slope_tol and corr >= mmr_identity_corr_min. The correlation is signed by c."""
    rate_f, a_f = np.asarray(rate_f, float), np.asarray(a_f, float)
    mask = _finite(rate_f, a_f)
    rate_f, a_f = rate_f[mask], a_f[mask]
    if len(a_f) < MIN_FIT_SAMPLES or np.var(a_f) <= 0 or np.var(rate_f) <= 0 or not np.isfinite(identity_coefficient):
        return NAN, NAN, CheckValue.NOT_MEASURED
    slope = float(np.cov(a_f, rate_f, bias=True)[0, 1] / np.var(a_f))
    corr = float(np.corrcoef(a_f, rate_f)[0, 1] * np.sign(identity_coefficient))
    ratio = float(slope / identity_coefficient)
    holds = abs(ratio - 1) <= settings.mmr_identity_slope_tol and corr >= settings.mmr_identity_corr_min
    return ratio, corr, CheckValue.PASS if holds else CheckValue.FAIL


def identity_wave(
    t_yr: np.ndarray,
    a_f: np.ndarray,
    body_coefficient: int,
    angle_f: np.ndarray,
    centre: np.ndarray,
    smoothing_yr: float = NAN,
    filter_order: Optional[int] = None,
) -> Tuple[float, float]:
    """pred = k int (n(a_f) - <n>) dt in the band of sigma_f - c; (correlation, amplitude ratio std(obs)/std(pred)).
    Necessary, not sufficient.

    The band: pred minus its low-pass at `smoothing_yr` (the centre window W_c, the same smoothing that made c);
    without it only the least-squares line of pred is removed.
    """
    t_yr, a_f, angle_f, centre = (np.asarray(x, dtype=float) for x in (t_yr, a_f, angle_f, centre))
    if len(t_yr) < MIN_FIT_SAMPLES:
        return NAN, NAN
    n = mean_motion(a_f)
    pred = body_coefficient * np.concatenate([[0.0], np.cumsum((n[1:] + n[:-1]) / 2 * np.diff(t_yr))])
    pred -= body_coefficient * np.mean(n) * (t_yr - t_yr[0])
    if np.isfinite(smoothing_yr) and smoothing_yr > 2 * np.median(np.diff(t_yr)):
        order = filter_order if filter_order is not None else AnalysisSettings().filter_order
        pred = pred - filtering.lowpass(t_yr, pred, smoothing_yr, order)
    else:
        pred = pred - np.polyval(np.polyfit(t_yr, pred, 1), t_yr)
    observed = angle_f - centre
    mask = _finite(pred, observed)
    if np.count_nonzero(mask) < MIN_FIT_SAMPLES or np.std(pred[mask]) <= 0 or np.std(observed[mask]) <= 0:
        return NAN, NAN
    corr = correlation(observed[mask], pred[mask])
    return corr, float(np.std(observed[mask]) / np.std(pred[mask]))


# --- L4 force law -------------------------------------------------------------------------------


def window_bounds(t_yr: np.ndarray, windows: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """(first, stop) sample index of each window [start, end] (both ends included) on an increasing `t_yr`."""
    starts = np.array([w[0] for w in windows], dtype=float)
    ends = np.array([w[1] for w in windows], dtype=float)
    return np.searchsorted(t_yr, starts, side='left'), np.searchsorted(t_yr, ends, side='right')


def _cumulative(values: np.ndarray) -> np.ndarray:
    """Running sums with a leading 0: the sum over samples [i, j) is out[j] - out[i]."""
    running = np.empty(len(values) + 1)
    running[0] = 0.0
    np.cumsum(values, out=running[1:])
    return running


def sine_fits_in_windows(
    t_yr: np.ndarray, angle: np.ndarray, target: np.ndarray, windows: Sequence[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """target = A sin(angle) + B cos(angle) + C by least squares in every window; (A, B, C, R^2) arrays, nan where a window
    holds fewer than MIN_FIT_SAMPLES finite samples.

    Closed form: the 3x3 normal equations of each window come from running sums of sin, cos, 1, target and their
    products, so the cost is one pass over the record plus a 3x3 solve per window, whatever the number of windows
    (99942: 1.65 million per-window lstsq calls before). The target is centred on its mean over the record first (the
    running sums then cancel less). Nearly singular windows (an angle that does not move: SINE_FIT_HADAMARD_MIN) fall
    back to lstsq, the minimum-norm solution of the per-window fit. `t_yr` must increase."""
    t_yr, angle, target = (np.asarray(x, dtype=float) for x in (t_yr, angle, target))
    n_windows = len(windows)
    blank = np.full(n_windows, NAN)
    if not n_windows or not len(t_yr):
        return blank, blank.copy(), blank.copy(), blank.copy()
    finite = _finite(angle, target)
    weight = finite.astype(float)
    shift = float(np.mean(target[finite])) if np.any(finite) else 0.0
    angle0 = np.where(finite, angle, 0.0)
    y = np.where(finite, target - shift, 0.0)
    sin, cos = np.sin(angle0) * weight, np.cos(angle0) * weight
    first, stop = window_bounds(t_yr, windows)
    sums = {}
    products = {'ss': (sin, sin), 'cc': (cos, cos), 'sc': (sin, cos), 'sy': (sin, y), 'cy': (cos, y), 'yy': (y, y)}
    for name, values in (('n', weight), ('s', sin), ('c', cos), ('y', y)):
        running = _cumulative(values)
        sums[name] = running[stop] - running[first]
    for name, (left, right) in products.items():  # one product at a time (memory)
        running = _cumulative(left * right)
        sums[name] = running[stop] - running[first]
    del running
    count = np.rint(sums['n'])
    matrix = np.empty((n_windows, 3, 3))
    matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2] = sums['ss'], sums['sc'], sums['s']
    matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2] = sums['sc'], sums['cc'], sums['c']
    matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2] = sums['s'], sums['c'], count
    rhs = np.stack([sums['sy'], sums['cy'], sums['y']], axis=1)
    usable = count >= MIN_FIT_SAMPLES
    coef = np.full((n_windows, 3), NAN)
    conditioned = np.zeros(n_windows, dtype=bool)
    if np.any(usable):
        diagonal = np.prod(np.diagonal(matrix, axis1=1, axis2=2), axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            hadamard = np.where(diagonal > 0, np.linalg.det(matrix) / diagonal, 0.0)
        conditioned = usable & (hadamard > SINE_FIT_HADAMARD_MIN)
        coef[conditioned] = np.linalg.solve(matrix[conditioned], rhs[conditioned][:, :, None])[:, :, 0]
    total = sums['yy'] - np.where(count > 0, sums['y'] ** 2 / np.maximum(count, 1), 0.0)
    residual = sums['yy'] - np.einsum('ij,ij->i', coef, rhs)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = np.where(total > 0, 1 - residual / total, NAN)
    for i in np.flatnonzero(usable & ~conditioned):  # rare: the per-window lstsq of the degenerate window
        part = slice(first[i], stop[i])
        a_sin, b_cos, offset, r2[i] = _sine_fit(angle[part], target[part])  # unshifted: the minimum norm is not shift-free
        coef[i] = (a_sin, b_cos, offset - shift)
    r2[~usable] = NAN
    return coef[:, 0], coef[:, 1], coef[:, 2] + shift, r2


def force_windows(t_yr: np.ndarray, angle_f: np.ndarray, adot_f: np.ndarray, window_yr: float, step_fraction: float) -> List[ForceWindow]:
    """a_dot_f = R sin(sigma_f - phi0) + C by linear least squares on (sin, cos, 1) in sliding windows [au/yr]
    (`sine_fits_in_windows`, closed form); `t_yr` increases."""
    t_yr, angle_f, adot_f = (np.asarray(x, dtype=float) for x in (t_yr, angle_f, adot_f))
    windows = sliding_windows(t_yr, window_yr, step_fraction)
    a_sin, b_cos, offset, r2 = sine_fits_in_windows(t_yr, angle_f, adot_f, windows)
    return [
        ForceWindow(start, end, float(r2[i]), float(np.hypot(a_sin[i], b_cos[i])), float(np.arctan2(-b_cos[i], a_sin[i])), float(offset[i]))
        for i, (start, end) in enumerate(windows)
        if np.isfinite(a_sin[i])
    ]


def multiplet_fit(
    t_yr: np.ndarray,
    angle_f: np.ndarray,
    adot_f: np.ndarray,
    slow_offset: np.ndarray,
    harmonics: Sequence[int],
    windows: Sequence[Tuple[float, float]],
    power_share: float,
) -> MultipletFit:
    """a_dot_f = sum_j [A_j sin sigma_j + B_j cos sigma_j] + C, sigma_j = sigma_f + j slow_offset (slow_offset = varpi - varpi_P);
    coefficient coherence over the windows for the terms carrying `power_share` of the power.

    R^2 and eps_j = |C_j|/|C_0| (j != 0, in the order of `harmonics`) come from one fit over the union of the
    windows; the coherence |<C_j/|C_j|>| from a fit in each window, its minimum over the leading terms.
    """
    t_yr, angle_f, adot_f, slow_offset = (np.asarray(x, dtype=float) for x in (t_yr, angle_f, adot_f, slow_offset))
    harmonics = tuple(int(j) for j in harmonics)
    n_terms = len(harmonics)
    finite = _finite(angle_f, adot_f, slow_offset)

    def solve(rows) -> Tuple[np.ndarray, float]:
        """`members_fit` on the finite samples of `rows` (a mask or a slice)."""
        keep = finite[rows]
        angle, offset, target = angle_f[rows][keep], slow_offset[rows][keep], adot_f[rows][keep]
        if len(target) <= 2 * n_terms + 1:
            return np.full(n_terms, np.nan + 0j), NAN
        return members_fit(angle, offset, target, harmonics)

    overall, r2 = solve(intervals_mask(t_yr, windows))
    per_window = np.array([solve(samples_between(t_yr, a, b))[0] for a, b in windows]).reshape(len(windows), n_terms)
    magnitude = np.abs(overall)
    eps: Tuple[float, ...] = ()
    if 0 in harmonics:
        main = magnitude[harmonics.index(0)]
        eps = tuple(float(magnitude[i] / main) if main > 0 else NAN for i, j in enumerate(harmonics) if j != 0)
    coherence = leading_coherence(magnitude, per_window, power_share)
    return MultipletFit(harmonics, per_window, r2, coherence, eps)


def leading_coherence(magnitude: np.ndarray, per_window: np.ndarray, power_share: float) -> float:
    """Coherence of a law's complex coefficients over its windows: min over the leading terms (those carrying
    `power_share` of the pooled power, `magnitude` = |C| of the pooled fit) of |<C/|C|>| over the windows
    (`per_window`: windows x terms); nan without power or windows. The multiplet law's and the N-harmonic law's test."""
    if not (np.all(np.isfinite(magnitude)) and np.sum(magnitude**2) > 0 and len(per_window)):
        return NAN
    order = np.argsort(-magnitude)
    share = np.cumsum(magnitude[order] ** 2) / np.sum(magnitude**2)
    leading = order[: int(np.searchsorted(share, power_share)) + 1]
    values = []
    for i in leading:
        column = per_window[:, i]
        column = column[np.isfinite(column) & (np.abs(column) > 0)]
        values.append(float(np.abs(np.mean(column / np.abs(column)))) if len(column) else NAN)
    return float(np.min(values)) if values else NAN


def phase_tolerance(eps: Sequence[float], phase_sigma_rad: float, floor_rad: float) -> float:
    """phi_tol = max(floor, arcsin(sum eps_j) + 2 sigma_phi) [rad]: the centre of a modulated pendulum swings that much
    (the column rsp_phi0_tol_deg next to the spread rsp_phi0_max_dev_deg; it decides nothing, `a_response_check`).
    Missing eps or sigma_phi count as 0.

    The members' force sum_j C_j e^{i j theta} = C_0 (1 + z) with |z| <= r = sum eps_j turns its phase by at most
    arcsin(r) only while r < 1 (the disk 1 + z then leaves out the origin). From r = 1 on the disk holds the origin
    and the phase can turn by pi (1 + 1.5 e^{i pi} = -0.5): there is no bound, nan (W3-G6)."""
    total = float(np.nansum(np.asarray(eps, dtype=float))) if len(eps) else 0.0
    if total >= 1.0:
        return NAN
    sigma = phase_sigma_rad if np.isfinite(phase_sigma_rad) else 0.0
    return float(max(floor_rad, np.arcsin(total) + 2 * sigma))


@dataclass(frozen=True)
class JumpGuard:
    """The confirmed jumps of a (`Geometry.jumps` without kicks) and the locator's baseline tau [yr]: where the force law
    and the onset test must not look (`jump_zones`, `onset_bands`). A kick (a step of a without a change of the cycle
    energy) is not a jump here: inside a libration it is the libration itself (2001 ND13 1V-3), and guarding it erases
    the measurement."""

    times_yr: Tuple[float, ...] = ()
    tau_yr: float = NAN
    # the half-width of each jump's zone (`geometry.Jump.zone`) when it is not tau/2: an action step (G2a) owns its
    # transitional half-cycle; empty = tau/2 for every jump
    half_widths_yr: Tuple[float, ...] = ()

    @classmethod
    def from_geometry(cls, geometry: Optional[Geometry]) -> 'JumpGuard':
        if geometry is None:
            return cls()
        tau = float(getattr(geometry, 'jump_tau_yr', NAN))
        cuts = sorted((j for j in getattr(geometry, 'jumps', ()) if not j.kick), key=lambda j: j.time_yr)
        zones = [j.zone(tau) for j in cuts]
        times = tuple(float((lo + hi) / 2) for lo, hi in zones)
        halves = tuple(float((hi - lo) / 2) for lo, hi in zones) if any(j.action_step for j in cuts) else ()
        return cls(times, tau, halves)

    @property
    def half_tau_yr(self) -> float:
        """tau / 2, the locator's uncertainty of a jump's time (a regime starts at jump + tau/2); 0 when unknown."""
        return self.tau_yr / 2 if np.isfinite(self.tau_yr) and self.tau_yr > 0 else 0.0

    def half_width_yr(self, index: int) -> float:
        """The half-width of jump `index`'s zone: its own (`half_widths_yr`), else tau/2."""
        return self.half_widths_yr[index] if index < len(self.half_widths_yr) else self.half_tau_yr


def jump_zones(guard: Optional[JumpGuard], cutoff_yr: float, settings: AnalysisSettings) -> List[Tuple[float, float]]:
    """[jump - tau/2 - x P_cut, jump + tau/2 + x P_cut] around each confirmed jump, x = filter_edge_trim_cutoffs, P_cut the
    response cutoff (`response_cutoff_yr`): the smoothed a turns a step into a rise about P_cut wide, and its derivative
    into a peak of a_dot that a force-law window fits instead of the pendulum (a jump of 10 amplitudes of the a-wave:
    window R^2 0.02 at the window's edge, 0.998 one P_cut away; inv-06-07 Q6). tau/2 covers the locator's error of the
    jump time. Empty without jumps."""
    if guard is None or not guard.times_yr:
        return []
    trim = settings.filter_edge_trim_cutoffs * (cutoff_yr if np.isfinite(cutoff_yr) else 0.0)
    return [(t - guard.half_width_yr(i) - trim, t + guard.half_width_yr(i) + trim) for i, t in enumerate(guard.times_yr)]


def outside_zones(windows: Sequence, zones: Sequence[Tuple[float, float]], bounds=lambda w: (w.t_start_yr, w.t_end_yr)) -> list:
    """The windows (ForceWindow, or (start, end) with `bounds=tuple`) that overlap none of `zones`."""
    if not zones:
        return list(windows)
    kept = []
    for window in windows:
        start, end = bounds(window)
        if not any(start <= hi and end >= lo for lo, hi in zones):
            kept.append(window)
    return kept


def zones_mask(t_yr: np.ndarray, zones: Sequence[Tuple[float, float]]) -> np.ndarray:
    """True outside every zone (the samples the 1:1 law may use)."""
    return ~intervals_mask(t_yr, zones)


def _halves(windows: Sequence[ForceWindow]) -> Tuple[float, float]:
    """Median R^2 of the first and the second half of the windows (in time order)."""
    if not windows:
        return NAN, NAN
    ordered = sorted(windows, key=lambda w: w.t_start_yr)
    first, second = np.array_split(np.array([w.r2 for w in ordered], dtype=float), 2)
    first = first if len(first) else second
    second = second if len(second) else first
    return finite_median(first), finite_median(second)


def finite_median(values) -> float:
    """Median of the finite values (an array, or a sequence that may hold None); nan for none."""
    values = np.asarray([v for v in values if v is not None] if not isinstance(values, np.ndarray) else values, float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else NAN


def phase_spread(windows: Sequence[ForceWindow]) -> Tuple[float, float, float]:
    """(circular mean phi0, coherence, largest |phi0 - mean|) over the windows [rad]."""
    phases = [w.phi0_rad for w in windows]
    mean = circular_mean(phases)
    if not np.isfinite(mean):
        return NAN, NAN, NAN
    deviation = float(np.max(np.abs(wrap_pi(np.asarray(phases) - mean))))
    return mean, circular_coherence(phases), deviation


def a_response_check(
    windows: Sequence[ForceWindow],
    multiplet: Optional[MultipletFit],
    amplitude_max_rad: float,
    settings: AnalysisSettings,
) -> Tuple[CheckValue, bool, str]:
    """L4 (check_mmr_a_response): pass with R^2 >= mmr_force_r2_full in both halves (thin for R^2 in [mmr_force_r2_fail,
    mmr_force_r2_full)); the multiplet law (taken when R^2 gains >= mmr_force_multi_gain) needs stable coefficients as
    well; fail with R^2 < mmr_force_r2_fail under both laws or unstable multiplet coefficients; not_measured below
    mmr_a_response_min_amp_deg or with fewer than mmr_force_min_windows windows. Returns (check, thin, law).

    The spread of the single law's phase phi0 is a column only (rsp_phi0_*, W7 'phi0', variant D): with a free offset C
    the phase is degenerate with C on a libration arc (its error grows as sqrt(32)/A^2; a second harmonic eps shifts
    it by ~4 eps), and a stable phase is no criterion of a resonance (a modulated pendulum's phase moves: Tsiganis 2010
    eq. 27; Morbidelli 2002 §9.3).

    No amplitude (no turning points) is not_applicable. The single law's R^2 is the smaller of the two half medians; the
    multiplet's is its fit over all windows.
    """
    if not np.isfinite(amplitude_max_rad):
        return CheckValue.NOT_APPLICABLE, False, ''
    if amplitude_max_rad < np.radians(settings.mmr_a_response_min_amp_deg) or len(windows) < settings.mmr_force_min_windows:
        return CheckValue.NOT_MEASURED, False, ''
    halves = [x for x in _halves(windows) if np.isfinite(x)]
    single_r2 = min(halves) if halves else NAN
    multi_r2 = multiplet.r2 if multiplet is not None else NAN
    take_multiplet = np.isfinite(multi_r2) and np.isfinite(single_r2) and multi_r2 - single_r2 >= settings.mmr_force_multi_gain
    law, r2, stable = LAW_SINGLE, single_r2, True
    if take_multiplet:
        law, r2 = LAW_MULTIPLET, multi_r2
        stable = np.isfinite(multiplet.coefficient_coherence) and multiplet.coefficient_coherence >= settings.mmr_force_coef_coh_min
    if not np.isfinite(r2):
        return CheckValue.NOT_MEASURED, False, law
    weak_under_both = single_r2 < settings.mmr_force_r2_fail and not (np.isfinite(multi_r2) and multi_r2 >= settings.mmr_force_r2_fail)
    if weak_under_both or r2 < settings.mmr_force_r2_fail or not stable:
        return CheckValue.FAIL, False, law
    return CheckValue.PASS, r2 < settings.mmr_force_r2_full, law


# --- CS12: the N-harmonic force law -------------------------------------------------------------------------------
#
# sigma_dot = c (a - a*) and sigma_ddot = c a_dot = -V'(sigma) for the periodic V that R0 fits as V_E with up to
# SEPARATRIX_HARMONICS_MAX harmonics (`pendulum.fit_potential_from_rate`), so the force law of a pendulum in any such
# well is a_dot = sum_{k=1..N} [A_k sin k psi + B_k cos k psi] + C, psi the angle the single law reads (sigma, or
# sigma - c(t) in a moving well). The one-harmonic law (N = 1) is exact only for a sinusoidal well: a double well or a
# flat bottom swung at large amplitude fails it (2001 ND13 1V-3: R^2 0.18 for N = 1, 0.65 for N = 2).


def _harmonic_min_samples(n: int) -> int:
    """Samples a fit of the N-harmonic law needs: its 2N + 1 parameters and two degrees of freedom left."""
    return 2 * n + 3


def _harmonic_design(angle: np.ndarray, n: int) -> np.ndarray:
    """Columns sin psi, cos psi, sin 2 psi, cos 2 psi, ... up to N (no constant)."""
    return np.column_stack([f(k * angle) for k in range(1, n + 1) for f in (np.sin, np.cos)])


def harmonic_fits(
    t_yr: np.ndarray, angle: np.ndarray, target: np.ndarray, windows: Sequence[Tuple[float, float]], n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """target = sum_{k=1..n} [A_k sin k angle + B_k cos k angle] + C by least squares in every window: (coefficients
    [windows x (A_1, B_1, ..., A_n, B_n, C)], R^2, count of finite samples); nan where a window holds fewer than
    `_harmonic_min_samples` finite samples. The closed form of `sine_fits_in_windows` with 2n + 1 columns (running
    sums, one pass); an ill-conditioned window (HARMONIC_FIT_COND_MAX) falls back to lstsq. `t_yr` must increase."""
    t_yr, angle, target = (np.asarray(x, dtype=float) for x in (t_yr, angle, target))
    n_windows, m = len(windows), 2 * n + 1
    if not n_windows:
        return np.zeros((0, m)), np.zeros(0), np.zeros(0)
    finite = _finite(angle, target)
    weight = finite.astype(float)
    shift = float(np.mean(target[finite])) if np.any(finite) else 0.0
    y = np.where(finite, target - shift, 0.0)
    columns = list((_harmonic_design(np.where(finite, angle, 0.0), n) * weight[:, None]).T) + [weight]
    first, stop = window_bounds(t_yr, windows)

    def sums(values: np.ndarray) -> np.ndarray:
        running = _cumulative(values)
        return running[stop] - running[first]

    gram = np.empty((n_windows, m, m))
    for i in range(m):
        for j in range(i, m):
            gram[:, i, j] = gram[:, j, i] = sums(columns[i] * columns[j])
    rhs = np.stack([sums(column * y) for column in columns], axis=1)
    syy, sy, count = sums(y * y), sums(y), sums(weight)
    coef = np.full((n_windows, m), NAN)
    usable = count >= _harmonic_min_samples(n)
    with np.errstate(all='ignore'):
        condition = np.where(usable, np.linalg.cond(np.where(usable[:, None, None], gram, np.eye(m))), np.inf)
    good = usable & (condition < HARMONIC_FIT_COND_MAX)
    if np.any(good):
        coef[good] = np.linalg.solve(gram[good], rhs[good][:, :, None])[:, :, 0]
    for i in np.flatnonzero(usable & ~good):  # rare: the per-window lstsq of a degenerate window
        part = slice(first[i], stop[i])
        keep = finite[part]
        design = np.column_stack([column[part][keep] for column in columns])
        coef[i] = np.linalg.lstsq(design, y[part][keep], rcond=None)[0]
    total = syy - np.where(count > 0, sy**2 / np.maximum(count, 1), 0.0)
    residual = syy - np.einsum('ij,ij->i', np.nan_to_num(coef), rhs)
    with np.errstate(all='ignore'):
        r2 = np.where(usable & (total > 0), 1 - residual / total, NAN)
    coef[:, -1] += shift
    return coef, r2, count


def lower_half_median(starts: Sequence[float], r2: np.ndarray) -> float:
    """The single law's statistic on (window start, R^2): the smaller of the medians of the two halves of the windows
    in time order (`_halves`); nan when neither half has a finite value."""
    values = np.asarray(r2, dtype=float)[np.argsort(np.asarray(starts, dtype=float))]
    if not len(values):
        return NAN
    first, second = np.array_split(values, 2)
    first = first if len(first) else second
    second = second if len(second) else first
    medians = [m for m in (finite_median(first), finite_median(second)) if np.isfinite(m)]
    return float(min(medians)) if medians else NAN


def shifted_rates(t_yr: np.ndarray, adot: np.ndarray, period_yr: float, n_shifts: int):
    """a_dot shifted in time by n_shifts lags (m + 1/4) P and (m + 3/4) P, alternately forward and back, whole periods
    m >= 1 spread evenly up to half the record, the samples shifted out dropped (nan), not wrapped: the null of
    `ForceFit.null_r2` and `harmonic_null`. Nothing when half the record is shorter than 1.75 P."""
    span = float(t_yr[-1] - t_yr[0]) if len(t_yr) > 1 else NAN
    if not (np.isfinite(period_yr) and period_yr > 0 and np.isfinite(span)):
        return
    last = int(np.floor(span / 2 / period_yr - 0.75))  # the largest m with (m + 3/4) P <= T / 2
    if last < 1:
        return
    step = float(np.median(np.diff(t_yr)))
    wholes = np.round(np.linspace(1, last, n_shifts)).astype(int)
    for j, whole in enumerate(wholes):
        lag = int(round((whole + (0.25 if j % 2 == 0 else 0.75)) * period_yr / step))
        moved = np.full(len(adot), NAN)
        if j % 2 == 0:
            moved[lag:] = adot[: len(adot) - lag]
        else:
            moved[: len(adot) - lag] = adot[lag:]
        yield moved


def harmonic_coherence(
    t_yr: np.ndarray,
    angle: np.ndarray,
    adot: np.ndarray,
    windows: Sequence[Tuple[float, float]],
    coef: np.ndarray,
    n: int,
    power_share: float,
) -> float:
    """`leading_coherence` of the complex C_k = A_k + i B_k of the N-harmonic law over its windows; the leading
    harmonics from one fit over the union of the windows (the multiplet law's test)."""
    mask = intervals_mask(t_yr, windows) & _finite(angle, adot)
    if mask.sum() < _harmonic_min_samples(n):
        return NAN
    design = np.column_stack([_harmonic_design(angle[mask], n), np.ones(int(mask.sum()))])
    pooled = np.linalg.lstsq(design, adot[mask], rcond=None)[0]
    magnitude = np.hypot(pooled[0 : 2 * n : 2], pooled[1 : 2 * n : 2])
    return leading_coherence(magnitude, coef[:, 0 : 2 * n : 2] + 1j * coef[:, 1 : 2 * n : 2], power_share)


def harmonic_null(
    fit: 'ForceFit', angle: np.ndarray, adot: np.ndarray, windows: Sequence[Tuple[float, float]], n: int, period_yr: float
) -> float:
    """The largest R^2 (`lower_half_median`) of the same N-harmonic law with a_dot shifted in time (`shifted_rates`),
    counting only the windows every sample of which the shift keeps: a window the shift cuts is a shorter fit, not a
    null (on TX9 at 0.5 yr cut windows gave a multiplet null of 0.96). nan when no shift keeps mmr_force_min_windows
    full windows."""
    settings = fit.settings
    first, stop = window_bounds(fit.t_yr, windows)
    starts = np.array([w[0] for w in windows], dtype=float)
    values = []
    for moved in shifted_rates(fit.t_yr, adot, period_yr, settings.mmr_force_null_shifts):
        _, r2, count = harmonic_fits(fit.t_yr, angle, moved, windows, n)
        full = np.flatnonzero(np.rint(count) >= stop - first)
        if len(full) < settings.mmr_force_min_windows:
            continue
        value = lower_half_median(starts[full], r2[full])
        if np.isfinite(value):
            values.append(value)
    return float(max(values)) if values else NAN


def independent_halves_r2(t_yr: np.ndarray, angle: np.ndarray, adot: np.ndarray, windows: Sequence[Tuple[float, float]], n: int) -> float:
    """Out of sample: the N-harmonic law fitted on the first half in time of the windows' samples predicts the second
    half and the reverse, scored only where the fitting half visited the angle (VISITED_BIN_DEG bins: a law says nothing
    about angles it never saw) with the offset refitted (a slow drift of a* is no part of the force); the smaller R^2.
    nan when either direction is not measured (fewer than `_harmonic_min_samples` samples, or a target without spread
    there): one direction alone is no out-of-sample test. A more flexible law needs this: the shape it fits must be the
    same on independent cycles."""
    mask = intervals_mask(t_yr, windows) & _finite(angle, adot)
    t, psi, y = t_yr[mask], angle[mask], adot[mask]
    if len(t) < 2 * _harmonic_min_samples(n):
        return NAN
    middle = 0.5 * (t[0] + t[-1])
    n_bins = int(round(360.0 / VISITED_BIN_DEG))
    bins = np.linspace(0.0, TWO_PI, n_bins + 1)
    values = []
    for fitting, tested in ((t <= middle, t > middle), (t > middle, t <= middle)):
        design = np.column_stack([_harmonic_design(psi[fitting], n), np.ones(int(fitting.sum()))])
        coef = np.linalg.lstsq(design, y[fitting], rcond=None)[0]
        visited = np.histogram(np.mod(psi[fitting], TWO_PI), bins=bins)[0] > 0
        inside = tested.copy()
        inside[tested] = visited[np.clip(np.digitize(np.mod(psi[tested], TWO_PI), bins) - 1, 0, n_bins - 1)]
        if inside.sum() < _harmonic_min_samples(n):
            return NAN
        residual = y[inside] - _harmonic_design(psi[inside], n) @ coef[:-1]
        residual = residual - np.mean(residual)
        total = np.sum((y[inside] - np.mean(y[inside])) ** 2)
        if not total > 0:
            return NAN
        values.append(1 - np.sum(residual**2) / total)
    return float(min(values))


def harmonic_law(
    fit: 'ForceFit', angle: np.ndarray, adot: np.ndarray, zones: Sequence[Tuple[float, float]], single_r2: float
) -> Optional[HarmonicLaw]:
    """CS12 (W23): the N-harmonic force law on psi (`angle`, already shifted in a moving well) in the single law's
    windows (MMR_FORCE_WINDOW_CYCLES x P stepped by MMR_FORCE_WINDOW_STEP inside the pieces, none overlapping `zones`),
    its R^2 the smaller of the two half medians as the single law's (`single_r2`).

    N grows from 2 while R^2 gains at least mmr_force_multi_gain over N - 1 (N = 1 is `single_r2`), up to
    mmr_force_harmonics_max. The law is taken when N >= 2 gained and, read in this order (the first refusal is
    `reason`): a shifted null exists (`harmonic_null`), the leading C_k are coherent over the windows
    (>= mmr_force_coef_coh_min, `harmonic_coherence`), R^2 beats the null by mmr_force_null_gain, R^2 >=
    mmr_force_r2_fail, and the law of one half predicts the other with R^2 >= mmr_force_r2_fail
    (`independent_halves_r2`). None when the law is off (mmr_force_harmonics_max < 2), with fewer than
    mmr_force_min_windows windows or without a single-law R^2.

    An exact one-harmonic pendulum gains <= 0.001 and is never taken; a kinematic linear swing wider than about
    +-150 deg is (sin psi fits a line less well there), which is why L4 is never sufficient alone."""
    settings, t = fit.settings, fit.t_yr
    if settings.mmr_force_harmonics_max < 2:
        return None
    windows = []
    for start, end in fit.pieces:
        windows += sliding_windows(t[samples_between(t, start, end)], fit.window_yr, settings.mmr_force_window_step)
    windows = outside_zones(windows, zones, bounds=tuple)
    if len(windows) < settings.mmr_force_min_windows or not np.isfinite(single_r2):
        return None
    starts = [w[0] for w in windows]
    previous, best, first_try = single_r2, None, None
    for n in range(2, settings.mmr_force_harmonics_max + 1):
        coef, r2, _ = harmonic_fits(t, angle, adot, windows, n)
        value = lower_half_median(starts, r2)
        first_try = first_try or (n, value)
        if not (np.isfinite(value) and value - previous >= settings.mmr_force_multi_gain):
            break
        best, previous = (n, coef, value), value
    if best is None:
        n, value = first_try
        return HarmonicLaw(n, value, single_r2, len(windows), reason=HARMONIC_NO_GAIN)
    n, coef, value = best
    period = fit.window_yr / settings.mmr_force_window_cycles
    coherence = harmonic_coherence(t, angle, adot, windows, coef, n, settings.mmr_force_power_share)
    null = harmonic_null(fit, angle, adot, windows, n, period)
    oos = independent_halves_r2(t, angle, adot, windows, n)
    c1 = np.hypot(finite_median(coef[:, 0]), finite_median(coef[:, 1]))
    c2 = np.hypot(finite_median(coef[:, 2]), finite_median(coef[:, 3]))
    guards = (
        (HARMONIC_NULL_UNMEASURED, np.isfinite(null)),
        (HARMONIC_INCOHERENT, coherence >= settings.mmr_force_coef_coh_min),
        (HARMONIC_BELOW_NULL, value - null >= settings.mmr_force_null_gain),
        (HARMONIC_BELOW_FAIL, value >= settings.mmr_force_r2_fail),
        (HARMONIC_HALVES, np.isfinite(oos) and oos >= settings.mmr_force_r2_fail),
    )
    reason = next((name for name, holds in guards if not holds), '')
    return HarmonicLaw(
        n,
        value,
        single_r2,
        len(windows),
        coherence=coherence,
        null_max=null,
        oos_r2=oos,
        c2_over_c1=float(c2 / c1) if c1 > 0 else NAN,
        taken=not reason,
        reason=reason,
    )


def omega0_from_force(identity_coefficient: float, force_au_yr: float) -> float:
    """omega0_K = sqrt(|c| K) [rad/yr]."""
    return float(np.sqrt(abs(identity_coefficient) * force_au_yr)) if np.isfinite(force_au_yr) and force_au_yr >= 0 else NAN


# --- B1, C4, slow a_dot, B5 ---------------------------------------------------------------------


def b1_amplitude(
    t_yr: np.ndarray,
    a_f: np.ndarray,
    period_yr: float,
    half_amplitude_rad: float,
    identity_coefficient: float,
    window: Tuple[float, float],
    window_cycles: Optional[float] = None,
    angle_f: Optional[np.ndarray] = None,
    settings: Optional[AnalysisSettings] = None,
) -> Tuple[float, float, float]:
    """B1: required and measured wave of a [au]; returns (required, measured, ratio).

    With `angle_f` and `settings` (W3-G11): a against sigma_dot / |c| in the band [P/f, f P] (`b1_in_band`):
    sigma_dot = c (a - a*) holds for the lambda part harmonic by harmonic, a_n = n omega sigma_n / |c|, so in the band
    the two agree whatever the shape of the libration and however its period wanders; an oscillation carried by p varpi
    leaves a flat there. Where the band cannot be formed (the window clear of its edges is shorter than a period: a
    secular cycle in a short record) the first harmonics at omega = 2 pi / P are compared instead, a1 against
    omega sigma1 / |c| (`b1_fixed_frequency`). Both are approximations: whatever else moves a or sigma there counts as
    the response, and one frequency per piece dephases when the period wanders.

    Without them (W2): required = omega A / |c| from the half-amplitude A of the turning points (exact for a harmonic
    libration only: a near-separatrix cycle has sigma1 ~ 4A/pi)."""
    if not (np.isfinite(period_yr) and period_yr > 0 and np.isfinite(half_amplitude_rad)):
        return NAN, NAN, NAN
    if angle_f is not None and settings is not None:
        in_band = b1_in_band(t_yr, a_f, angle_f, period_yr, identity_coefficient, window, settings)
        if np.isfinite(in_band[2]):
            return in_band
    use_angle = angle_f if settings is not None else None
    return b1_fixed_frequency(t_yr, a_f, period_yr, half_amplitude_rad, identity_coefficient, window, window_cycles, use_angle)


def b1_fixed_frequency(
    t_yr: np.ndarray,
    a_f: np.ndarray,
    period_yr: float,
    half_amplitude_rad: float,
    identity_coefficient: float,
    window: Tuple[float, float],
    window_cycles: Optional[float] = None,
    angle_f: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """B1 at omega = 2 pi / P: measured = the median sine-regression amplitude a1 of a_f (with a constant and a line)
    over consecutive windows of `window_cycles` periods (one fixed frequency over hundreds of cycles dephases as soon as
    the period wanders by more than 1/N_cycles: a Hilda over 100 kyr, 0.0009 au against 0.013 au per window); required
    = omega median(sigma1) / |c| with sigma1 the same regression of `angle_f`, or omega A / |c| without it."""
    omega = TWO_PI / period_yr
    required = omega * half_amplitude_rad / abs(identity_coefficient)
    t_yr, a_f = np.asarray(t_yr, float), np.asarray(a_f, float)
    angle = np.asarray(angle_f, float) if angle_f is not None else None
    pieces = [window]
    if window_cycles is not None and np.isfinite(window_cycles) and window[1] - window[0] > 2 * window_cycles * period_yr:
        length = window_cycles * period_yr
        starts = np.arange(window[0], window[1] - length / 2, length)
        pieces = [(start, min(start + length, window[1])) for start in starts]
    amplitudes, sigma_amplitudes = [], []
    for start, end in pieces:
        mask = (t_yr >= start) & (t_yr <= end) & np.isfinite(a_f)
        if angle is not None:
            mask &= np.isfinite(angle)
        if np.count_nonzero(mask) < MIN_FIT_SAMPLES + 1:
            continue
        tc = t_yr[mask] - t_yr[mask].mean()
        design = np.column_stack([np.sin(omega * tc), np.cos(omega * tc), np.ones_like(tc), tc])
        coef, _ = _fit(design, a_f[mask])
        amplitudes.append(float(np.hypot(coef[0], coef[1])))
        if angle is not None:
            coef, _ = _fit(design, angle[mask])
            sigma_amplitudes.append(float(np.hypot(coef[0], coef[1])))
    if sigma_amplitudes:
        required = omega * float(np.median(sigma_amplitudes)) / abs(identity_coefficient)
    if not amplitudes:
        return required, NAN, NAN
    measured = float(np.median(amplitudes))
    return float(required), measured, float(measured / required) if required > 0 else NAN


def b1_in_band(
    t_yr: np.ndarray,
    a_f: np.ndarray,
    angle_f: np.ndarray,
    period_yr: float,
    identity_coefficient: float,
    window: Tuple[float, float],
    settings: AnalysisSettings,
) -> Tuple[float, float, float]:
    """(required, measured, ratio) of B1 in the band [P/f, f P]: measured = sqrt 2 std(a_band), required = sqrt 2
    std(d sigma_band / dt) / |c| (the amplitudes of equal-power sines) over `window` clear of the band-pass edges
    (filter_edge_trim_cutoffs f P). The band is filtered on a slice of the record around the window (cost: the window,
    not the record). nan when the band cannot be formed or the cleared window is shorter than one period."""
    factor = settings.mmr_b1_band_factor
    t_yr, a_f, angle_f = (np.asarray(x, float) for x in (t_yr, a_f, angle_f))
    if not (factor > 1 and np.isfinite(identity_coefficient) and identity_coefficient != 0):
        return NAN, NAN, NAN
    trim = settings.filter_edge_trim_cutoffs * factor * period_yr
    first = int(np.searchsorted(t_yr, window[0] - trim, side='left'))
    stop = int(np.searchsorted(t_yr, window[1] + trim, side='right'))
    t, a, angle = t_yr[first:stop], a_f[first:stop], angle_f[first:stop]
    if len(t) < MIN_FIT_SAMPLES + 1 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(angle))):
        return NAN, NAN, NAN

    def band(values: np.ndarray) -> np.ndarray:
        low = filtering.lowpass_with(t, values, period_yr / factor, settings)
        return low - filtering.lowpass_with(t, values, period_yr * factor, settings)

    try:
        a_band, rate_band = band(a), filtering.time_derivative(t, band(angle))
    except ValueError:  # P / f within two steps: the band does not exist on this grid
        return NAN, NAN, NAN
    edge = filtering.edge_window(t, trim)
    if not np.all(np.isfinite(edge)):  # the slice is shorter than the two edge trims
        return NAN, NAN, NAN
    start, end = max(window[0], edge[0]), min(window[1], edge[1])
    inside = (t >= start) & (t <= end)
    if not (end - start >= period_yr and np.count_nonzero(inside) > MIN_FIT_SAMPLES):
        return NAN, NAN, NAN
    measured = float(np.sqrt(2) * np.std(a_band[inside]))
    required = float(np.sqrt(2) * np.std(rate_band[inside]) / abs(identity_coefficient))
    return required, measured, float(measured / required) if required > 0 else NAN


def circulation_strength(rate_f: np.ndarray, omega0_force_rad_yr: float, force_au_yr: float) -> Tuple[float, float]:
    """C4: (<|sigma_dot_f|> / (2 omega0_K), predicted forced wave K / <|sigma_dot|> [au])."""
    rate = np.asarray(rate_f, float)
    rate = rate[np.isfinite(rate)]
    if not len(rate):
        return NAN, NAN
    mean_rate = float(np.mean(np.abs(rate)))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = mean_rate / (2 * omega0_force_rad_yr) if omega0_force_rad_yr > 0 else NAN
        wave = force_au_yr / mean_rate if mean_rate > 0 else NAN
    return float(ratio), float(wave)


def slow_adot_r2(slow: SlowComponents) -> float:
    """R^2 of a_dot_slow on sin/cos sigma_slow (kinematics), inside the trimmed slow window."""
    mask = filtering.window_mask(slow.t_yr, slow.window)
    return _sine_fit(np.asarray(slow.angle_slow, float)[mask], np.asarray(slow.adot_slow, float)[mask])[3]


def slow_libration_bound(
    t_yr: np.ndarray, centre: np.ndarray, a_slow: np.ndarray, identity_coefficient: float, window: Tuple[float, float]
) -> dict:
    """B5 from data: nu_c (linear fit of the centre), its error 2 std(resid) sqrt(3) / T_w, arc S = |nu_c| T_w, da = ptp a_slow,
    K_max = |nu_c| da / (1 - cos(S/2)) for S <= 2pi else |nu_c| da / 2, omega0_max = sqrt(|c| K_max),
    ratio = (|nu_c| - dnu_c) / (2 omega0_max). Keys: centre_rate, centre_rate_err, arc, force_max, omega0_max, period_min, ratio.

    K_max is the largest force a slow libration through the observed arc could have without moving a by more than
    observed: a = a0 + (K/nu_c)[cos sigma - cos sigma0], worst case the arc centred on the stable point.
    """
    keys = ('centre_rate', 'centre_rate_err', 'arc', 'force_max', 'omega0_max', 'period_min', 'ratio')
    out = dict.fromkeys(keys, NAN)
    t_yr, centre, a_slow = (np.asarray(x, dtype=float) for x in (t_yr, centre, a_slow))
    mask = (t_yr >= window[0]) & (t_yr <= window[1]) & _finite(centre, a_slow)
    if np.count_nonzero(mask) < MIN_FIT_SAMPLES:
        return out
    t, c_t, a = t_yr[mask], centre[mask], a_slow[mask]
    span = float(t[-1] - t[0])
    if span <= 0:
        return out
    slope, intercept = np.polyfit(t, c_t, 1)
    residual = c_t - (slope * t + intercept)
    rate = abs(float(slope))
    error = 2 * float(np.std(residual)) * np.sqrt(3) / span
    arc = rate * span
    da = float(np.ptp(a))
    denominator = 1 - np.cos(arc / 2) if arc <= TWO_PI else 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        force_max = rate * da / denominator if denominator > 0 else np.inf
    omega0_max = float(np.sqrt(abs(identity_coefficient) * force_max))
    with np.errstate(divide='ignore', invalid='ignore'):
        period_min = TWO_PI / omega0_max if omega0_max > 0 else np.inf
        ratio = (rate - error) / (2 * omega0_max) if omega0_max > 0 else np.inf
    out.update(
        centre_rate=float(slope),
        centre_rate_err=error,
        arc=arc,
        force_max=float(force_max),
        omega0_max=omega0_max,
        period_min=float(period_min),
        ratio=float(ratio),
    )
    return out


def slow_libration_check(
    bound: dict,
    omega0_theory_rad_yr: float,
    owner_halfwidth_au: float,
    a_slow_range_au: float,
    settings: AnalysisSettings,
    centre_pass: Tuple[float, float, float] = (NAN, NAN, NAN),
    episode_yr: Tuple[float, float] = (NAN, NAN),
) -> Tuple[CheckValue, str, bool]:
    """B5: excluded by data (ratio >= 1, full at >= mmr_b5_ratio_full), by geometry or by the owner; returns (check,
    route 'data'|'geometry'|'owner'|'none', exact_commensurability = |nu_c| < mmr_b5_exact_sigma dnu_c).

    The geometry route (W7 '37087'): the centre itself completes a revolution, at every smoothing
    (`centre_pass` = (geo_centre_max_pass_rev_lo, its t0, its t1): the smallest of the largest passes at x0.5 and x2
    is >= 1 revolution), over an interval that overlaps the deciding episode (`episode_yr`): a centre that turns
    round while the episode lasts does not librate slowly there. Only that overlap counts: the largest pass of the
    whole record does not exclude a capture elsewhere (48789 3-body: pass at 11-38 kyr, the episode at 54-98 kyr).
    The owner route: a found owner holds a and the half-width of a slow libration at omega0_max
    (`owner_halfwidth_au`) is below half the slow range of a. Without a centre rate: not_measured.
    The full/thin margin of the data route is read from `ratio` by the caller.

    No route by theory (W3-F3): omega0_th = |k| n sqrt(3 mu e^q) leaves out the coefficient of the disturbing function,
    an order of magnitude, not an upper bound of the libration frequency (Gallardo 2006 p. 30; Nesvorny & Morbidelli
    1998 p. 3031), so |nu_c| > 2 omega0_th excludes nothing. `omega0_theory_rad_yr` stays in the signature for the
    callers; the scale is the column `str_omega0_theory_rad_yr`.
    """
    rate = bound.get('centre_rate', NAN)
    error = bound.get('centre_rate_err', NAN)
    if not np.isfinite(rate):
        return CheckValue.NOT_MEASURED, ROUTE_NONE, False
    exact = bool(np.isfinite(error) and abs(rate) < settings.mmr_b5_exact_sigma * error)
    ratio = bound.get('ratio', NAN)
    if np.isfinite(ratio) and ratio >= 1:
        return CheckValue.PASS, ROUTE_DATA, exact
    if centre_turns_in_episode(centre_pass, episode_yr):
        return CheckValue.PASS, ROUTE_GEOMETRY, exact
    if np.isfinite(owner_halfwidth_au) and np.isfinite(a_slow_range_au) and owner_halfwidth_au < a_slow_range_au / 2:
        return CheckValue.PASS, ROUTE_OWNER, exact
    return CheckValue.FAIL, ROUTE_NONE, exact


def centre_turns_in_episode(centre_pass: Tuple[float, float, float], episode_yr: Tuple[float, float]) -> bool:
    """B5 by geometry: the largest centre pass is >= one revolution at every smoothing (`centre_pass[0]`, the
    revolutions at x0.5/x2) and its interval [t0, t1] overlaps the deciding episode `episode_yr`."""
    revolutions, start, end = centre_pass
    episode_start, episode_end = episode_yr
    if not all(np.isfinite(x) for x in (revolutions, start, end, episode_start, episode_end)):
        return False
    return bool(revolutions >= REVOLUTION and start < episode_end and episode_start < end)


# --- L4, L3 and B1 on pieces of the record (pooled, one episode, the episodes before a jump) ------------------


MEASURED = (CheckValue.PASS, CheckValue.FAIL)


@dataclass(frozen=True)
class ResponseJudge:
    """What judging L4, L3 and B1 on some pieces needs besides their numbers: an aliased step (sigma_dot is the model
    rate there: not_applicable), a filter too slow for the period (the shape checks not_measured, their numbers still
    written), and for the 1:1 law the sampling and the body's elements (`coorbital_response`)."""

    aliased: bool
    too_slow: bool
    sampling: Sampling
    series: Optional[MMRSeries] = None


def episode_judge(period_yr: float, prep: Preprocessed, sampling: Sampling, series: Optional[MMRSeries], settings: AnalysisSettings):
    """The judge of an episode or of the episodes before a jump: the filter guard at their own period P_e."""
    too_slow = filter_guard(period_yr, prep.cutoff.cutoff_yr, settings, sampling.step_yr)[0]
    return ResponseJudge(sampling.verdict in ALIASED, too_slow, sampling, series)


@dataclass(frozen=True)
class ForceMeasurement:
    """The harmonic force law on the pieces at one response cutoff: the clean windows, the multiplet law, the check
    (`a_response_check`: value, thin, law; law 'harmonic' when the N-harmonic law lifted it), the windows dropped next
    to a confirmed jump of a and the N-harmonic law when it was consulted (`harmonic_law`)."""

    windows: List[ForceWindow]
    multiplet: Optional[MultipletFit]
    check: Tuple[CheckValue, bool, str]
    jump_dropped: int
    harmonic: Optional[HarmonicLaw] = None

    @property
    def force_au_yr(self) -> float:
        """K, the median window amplitude [au/yr] (nan without windows)."""
        return float(np.median([w.amplitude_au_yr for w in self.windows])) if self.windows else NAN

    @property
    def law_r2(self) -> float:
        """The R^2 the check reads: the multiplet's fit, the N-harmonic law's, else the smaller of the two half medians."""
        if self.check[2] == LAW_MULTIPLET and self.multiplet is not None:
            return float(self.multiplet.r2)
        if self.check[2] == LAW_HARMONIC and self.harmonic is not None:
            return float(self.harmonic.r2)
        halves = [x for x in _halves(self.windows) if np.isfinite(x)]
        return min(halves) if halves else NAN


@dataclass(frozen=True)
class ForceFit:
    """The harmonic force law (L4) of `measure_pieces` with everything but the series fixed: measured at the base
    response cutoff, again above a binding line of a_dot (`line_margin_cutoff`), and on a_dot shifted in time (the null
    test)."""

    t_yr: np.ndarray
    pieces: Sequence[Tuple[float, float]]
    window_yr: float
    angle_shift: Optional[np.ndarray]
    spec: Optional[ResonanceSpec]
    slow: Optional[SlowComponents]
    context: Context
    amplitude_max_rad: float
    settings: AnalysisSettings

    def measure(self, angle: np.ndarray, adot: np.ndarray, zones: Sequence[Tuple[float, float]], harmonic: bool = True) -> ForceMeasurement:
        """L4 on the pieces; with `harmonic` (not inside the null test) a failed or thin check consults the N-harmonic
        law (`harmonic_law`, CS12), which lifts it to pass (thin below mmr_force_r2_full), never lowers it; not for a
        1:1 angle, which has its own law (`coorbital_response`)."""
        t, settings = self.t_yr, self.settings
        force_angle = angle - self.angle_shift if self.angle_shift is not None else angle
        every_window = _force_law(t, force_angle, adot, self.pieces, self.window_yr, settings)
        windows = outside_zones(every_window, zones)
        multiplet = None
        if self.pieces:
            multiplet = _best_multiplet(
                t, force_angle, adot, self.pieces, self.window_yr, self.spec, self.slow, self.context, settings, zones
            )
        check = a_response_check(windows, multiplet, self.amplitude_max_rad, settings)
        law = None
        if harmonic and self._consults_harmonic(check):
            halves = [x for x in _halves(windows) if np.isfinite(x)]
            law = harmonic_law(self, force_angle, adot, zones, min(halves) if halves else NAN)
            if law is not None and law.taken:
                check = (CheckValue.PASS, law.r2 < settings.mmr_force_r2_full, LAW_HARMONIC)
        return ForceMeasurement(windows, multiplet, check, len(every_window) - len(windows), law)

    def _consults_harmonic(self, check: Tuple[CheckValue, bool, str]) -> bool:
        """The N-harmonic law is read when L4 fails or passes thin, never for a 1:1 angle."""
        value, thin, _ = check
        if self.spec is not None and law_planet(self.spec) is not None:
            return False
        return value == CheckValue.FAIL or (value == CheckValue.PASS and thin)

    def null_r2(
        self, angle: np.ndarray, adot: np.ndarray, measured: ForceMeasurement, zones: Sequence[Tuple[float, float]], period_yr: float
    ) -> float:
        """The largest R^2 of the law `measured` read (single or multiplet; the N-harmonic law's own `harmonic_null`,
        which counts full windows only) with a_dot shifted in time (`shifted_rates`) by
        mmr_force_null_shifts lags (m + 1/4) P and (m + 3/4) P, alternately forward and back, with whole periods m >= 1
        spread evenly up to half the record (the samples shifted out are dropped, not wrapped): the shift moves a to
        another stretch of the record and a quarter of a cycle out of phase with sigma, so what the law then explains
        is chance. Whole periods alone would not do: a steady libration shifted by m P is the same a (the exact
        pendulum of the tests: null R^2 0.94 at lags spread evenly over [P, T - P]), by half a period only its sign
        changes, which the free amplitude absorbs, and a circular shift puts the wrapped part at another phase. nan
        when half the record is shorter than 1.75 P or no shift gives a value."""
        if measured.check[2] == LAW_HARMONIC and measured.harmonic is not None:
            return float(measured.harmonic.null_max)
        values = []
        for moved in shifted_rates(self.t_yr, adot, period_yr, self.settings.mmr_force_null_shifts):
            shifted = self.measure(angle, moved, zones, harmonic=False)
            if measured.check[2] == LAW_MULTIPLET and shifted.multiplet is not None:
                values.append(float(shifted.multiplet.r2))
            else:
                halves = [x for x in _halves(shifted.windows) if np.isfinite(x)]
                values.append(min(halves) if halves else NAN)
        values = [v for v in values if np.isfinite(v)]
        return float(max(values)) if values else NAN


def _line_rule_applies(judge: ResponseJudge, spec: Optional[ResonanceSpec], force: ForceMeasurement, settings: AnalysisSettings) -> bool:
    """The line rule (`line_margin_cutoff`) reads a harmonic force law that was measured (enough clean windows, a
    force K) on a step that resolves the angle (not aliased) and a filter that keeps the cycle; a 1:1 angle has its
    own law (`coorbital_response`)."""
    if judge.aliased or judge.too_slow or (spec is not None and law_planet(spec) is not None):
        return False
    return len(force.windows) >= settings.mmr_force_min_windows and np.isfinite(force.force_au_yr) and force.force_au_yr > 0


def measure_pieces(
    prep: Preprocessed,
    context: Context,
    settings: AnalysisSettings,
    pieces: Sequence[Tuple[float, float]],
    period_yr: float,
    amplitudes_rad: Tuple[float, float],
    cycle_edges_yr: Sequence[float] = (),
    judge: Optional[ResponseJudge] = None,
    spec: Optional[ResonanceSpec] = None,
    slow: Optional[SlowComponents] = None,
    filtered: Optional[Tuple[np.ndarray, ...]] = None,
    episode_id: Optional[int] = None,
    before_jump: bool = False,
    n_tp: int = 0,
    guard: Optional[JumpGuard] = None,
    angle_shift: Optional[np.ndarray] = None,
) -> EpisodeResponse:
    """L4 (force law), L3 (identity) and B1 on `pieces` [(start, end)] read with one period `period_yr`: force windows
    of MMR_FORCE_WINDOW_CYCLES periods inside each piece and the medians of their two halves, the multiplet law, the
    identity on the union of the pieces, B1 on the longest; `amplitudes_rad` = (A_max, A_med) of their turning points,
    `cycle_edges_yr` every second turning point (the 1:1 law's cycles). `filtered` is `response_series` at that period
    when the caller has it. The record's pooled response, one episode (rule R) and the episodes before the last jump
    of a are this one measurement on different pieces.

    `guard` (the confirmed jumps of a): the force-law windows of both harmonic laws (the multiplet's fallback halves
    too) that overlap a `jump_zones` zone are dropped, fewer than mmr_force_min_windows clean windows make the harmonic
    L4 not_measured; the 1:1 law is not windowed and leaves out the samples in the zones instead. This comes before the lattice-line contamination, which
    then judges only the clean windows' L4. L3 and B1 are not guarded.

    Without `judge` only the numbers (the check values None: a run nothing judges). With it, as SPEC §4.5: an aliased
    step makes L4 and L3 not_applicable and B1 not_measured, a filter too slow for the period makes the measured ones
    not_measured; a 1:1 angle takes L4, its thin flag and the two half R^2 from its own law (`coorbital_check`, the
    phase spread not measured); L3 is not_measured on pieces shorter than two periods in all; a measured harmonic L4
    becomes not_measured (force_contaminated) when a forced line of the angle's lattice lies at P (`force_line`).

    The line rule (with `judge`, a harmonic law measured on a resolving step, W20 F5): when a line of a_dot rivals the
    force of the first fit (`line_margin_cutoff`), the force law is measured again on sigma and a smoothed above it
    (the identity and B1 keep the base series) and passes only with its R^2 at least mmr_force_null_gain above the
    largest R^2 of the same law with a shifted in time (`ForceFit.null_r2`); below that it fails, and without a null
    value it is not_measured.

    `angle_shift` (`geometry.well_shift`): both harmonic force laws are fitted on sigma - shift, psi = sigma - c(t) up to
    a constant on an episode accepted in the frame of its moving well, a_dot = K sin(sigma - c(t) - phi0) + C (a
    single phase on sigma drifts with the well: phi0 coherence 0.28-0.78 in sigma against 0.94-1.00 in psi on six such
    rows, W17 Q3); the identity and B1 read sigma (sigma_dot = c (a - a*) holds for sigma itself)."""
    t = np.asarray(prep.t_yr, float)
    c = context.identity_coefficient
    amplitude_max, amplitude_med = amplitudes_rad
    angle_f, rate_f, a_f, adot_f, smoothing = filtered if filtered is not None else response_series(prep, period_yr, settings)
    window_yr = settings.mmr_force_window_cycles * period_yr if np.isfinite(period_yr) else NAN
    fit = ForceFit(t, pieces, window_yr, angle_shift, spec, slow, context, amplitude_max, settings)
    zones = jump_zones(guard, smoothing, settings)
    force = fit.measure(angle_f, adot_f, zones)
    force_cutoff, line, null_max, adot_force = smoothing, NAN, NAN, adot_f
    if judge is not None and _line_rule_applies(judge, spec, force, settings):
        raised, line = line_margin_cutoff(t, adot_f, pieces, period_yr, smoothing, force.force_au_yr, settings)
        if raised > smoothing and raised > 2 * np.median(np.diff(t)):
            angle_r, _, _, adot_r, _ = smoothed_series(prep, raised, settings)
            zones_raised = jump_zones(guard, raised, settings)
            force = fit.measure(angle_r, adot_r, zones_raised)
            null_max = fit.null_r2(angle_r, adot_r, force, zones_raised, period_yr)
            force_cutoff, adot_force = raised, adot_r
        else:
            line = NAN
    windows, multiplet = force.windows, force.multiplet
    r2_h1, r2_h2 = _halves(windows)
    phi0, phi0_coherence, phi0_dev = phase_spread(windows)
    eps = multiplet.eps if multiplet is not None else ()
    phase_tol = phase_tolerance(eps, _phase_sigma(windows, t, adot_force, force_cutoff), np.radians(settings.mmr_force_phase_tol_deg))
    a_response, thin, law = force.check
    null_gain = force.law_r2 - null_max if np.isfinite(line) else NAN
    if np.isfinite(line) and a_response == CheckValue.PASS:
        # a law read above a line: no better than the same law with a shifted a is no law (R3 §2, mmr_force_null_gain)
        if not np.isfinite(null_gain):
            a_response, thin = CheckValue.NOT_MEASURED, False
        elif null_gain < settings.mmr_force_null_gain:
            a_response, thin = CheckValue.FAIL, False
    mask = intervals_mask(t, pieces)
    slope_ratio, corr, identity_check = identity(rate_f[mask], a_f[mask], c, settings)
    longest = max(pieces, key=lambda p: p[1] - p[0]) if pieces else None
    b1_req, b1_meas, b1_ratio = (NAN, NAN, NAN)
    if longest is not None:
        b1_req, b1_meas, b1_ratio = b1_amplitude(
            t, a_f, period_yr, amplitude_med, c, longest, settings.mmr_force_window_cycles, angle_f, settings
        )
    numbers = dict(
        episode_id=episode_id,
        before_jump=before_jump,
        n_tp=n_tp,
        t_start_yr=float(pieces[0][0]) if pieces else NAN,
        t_end_yr=float(pieces[-1][1]) if pieces else NAN,
        period_yr=period_yr,
        windows=windows,
        force_window_yr=window_yr,
        force_r2_h1=r2_h1,
        force_r2_h2=r2_h2,
        force_jump_dropped=force.jump_dropped,
        force_cutoff_yr=force_cutoff,
        force_line_binding_yr=line,
        force_r2_null_max=null_max,
        force_null_gain=null_gain,
        force_harmonic=force.harmonic,
        multiplet=multiplet,
        force_law=law,
        force_amplitude_au_yr=float(np.median([w.amplitude_au_yr for w in windows])) if windows else NAN,
        phi0_rad=phi0,
        phi0_coherence=phi0_coherence,
        phi0_max_dev_rad=phi0_dev,
        phi0_tol_rad=phase_tol,
        identity_slope_ratio=slope_ratio,
        identity_corr=corr,
        b1_required_au=b1_req,
        b1_measured_au=b1_meas,
        b1_ratio=b1_ratio,
    )
    if judge is None:
        return EpisodeResponse(**numbers)
    coorbital = coorbital_response(
        judge.series, prep, context, judge.sampling, smoothing, prep.angle_clean, adot_f, pieces, cycle_edges_yr, settings, zones
    )
    checks = {'a_response': a_response, 'a_response_thin': thin, 'identity': identity_check}
    line = force_line(spec, context, period_yr, settings)
    if line is not None:
        numbers.update(force_line_period_yr=line[0].period_yr, force_line=line[0].label, force_line_offset=line[1])
    return _judged(numbers, checks, coorbital, judge, pieces, amplitude_max, settings)


def _judged(
    numbers: dict,
    checks: dict,
    coorbital: Optional[CoorbitalLaw],
    judge: ResponseJudge,
    pieces: Sequence[Tuple[float, float]],
    amplitude_max: float,
    settings: AnalysisSettings,
) -> EpisodeResponse:
    """`measure_pieces` with its judge: the check values of L4 (the 1:1 law for a co-orbital), L3 and B1."""
    a_response, thin, identity_check = checks['a_response'], checks['a_response_thin'], checks['identity']
    if judge.aliased:
        a_response, thin = CheckValue.NOT_APPLICABLE, False
    elif judge.too_slow and a_response in MEASURED:
        a_response, thin = CheckValue.NOT_MEASURED, False
    if coorbital is not None:  # a 1:1 angle: its own law decides L4 (the harmonic windows stay as columns)
        a_response, thin = coorbital_check(coorbital, amplitude_max)
        if judge.too_slow and a_response in MEASURED:
            a_response, thin = CheckValue.NOT_MEASURED, False
        numbers = {**numbers, 'force_law': LAW_COORBITAL, 'force_r2_h1': coorbital.r2_h1, 'force_r2_h2': coorbital.r2_h2}
        numbers.update(phi0_max_dev_rad=NAN, phi0_tol_rad=NAN)
    elif a_response in MEASURED and np.isfinite(numbers.get('force_line_period_yr', NAN)):
        # the harmonic fit cannot tell the libration from the forced wave at its period (forced_lines)
        numbers = {**numbers, 'force_contaminated': True, 'force_raw_check': str(a_response)}
        a_response, thin = CheckValue.NOT_MEASURED, False

    period = numbers['period_yr']
    span = sum(b - a for a, b in pieces)
    if judge.aliased:
        identity_check = CheckValue.NOT_APPLICABLE
    elif judge.too_slow or (np.isfinite(period) and span < 2 * period):
        identity_check = CheckValue.NOT_MEASURED

    b1_ratio = numbers['b1_ratio']
    if not np.isfinite(amplitude_max):
        a_amplitude = CheckValue.NOT_APPLICABLE
    elif judge.aliased or judge.too_slow or not np.isfinite(b1_ratio):
        a_amplitude = CheckValue.NOT_MEASURED
    else:
        a_amplitude = CheckValue.FAIL if b1_ratio < settings.mmr_b1_ratio_min else CheckValue.PASS
    return EpisodeResponse(
        **numbers,
        coorbital=coorbital,
        a_response=a_response,
        a_response_thin=thin,
        identity=identity_check,
        a_amplitude=a_amplitude,
        filter_too_slow=judge.too_slow,
    )


def before_jump_response(
    prep: Preprocessed,
    context: Context,
    sampling: Sampling,
    settings: AnalysisSettings,
    episodes: Sequence[Episode],
    spec: Optional[ResonanceSpec] = None,
    slow: Optional[SlowComponents] = None,
    series: Optional[MMRSeries] = None,
    guard: Optional[JumpGuard] = None,
) -> Optional[EpisodeResponse]:
    """L4, L3 and B1 pooled over the librating `episodes` before the last jump of a (G1e, rule R): as the record's
    pooled response, with the median of their periods, A_max and the median A_med over them, B1 on the longest one;
    `guard` keeps the force law off the confirmed jumps (`measure_pieces`). None without episodes or a period."""
    if not episodes:
        return None
    period = float(np.median([e.period_med_yr for e in episodes]))
    if not (np.isfinite(period) and period > 0):
        return None
    ordered = sorted(episodes, key=lambda e: e.t_start_yr)
    return measure_pieces(
        prep,
        context,
        settings,
        [(e.t_start_yr, e.t_end_yr) for e in ordered],
        period,
        (float(np.nanmax([e.half_amp_max_rad for e in ordered])), float(np.nanmedian([e.half_amp_med_rad for e in ordered]))),
        np.concatenate([np.asarray(e.tp_times_yr, float)[::2] for e in ordered]),
        judge=episode_judge(period, prep, sampling, series, settings),
        spec=spec,
        slow=slow,
        before_jump=True,
        n_tp=sum(e.n_tp for e in ordered),
        guard=guard,
        angle_shift=well_shift(prep.t_yr, ordered),
    )


# --- rule R: the response the status tree reads ------------------------------------------------------------------

# the fields of each check, its check value first (`deciding_response` takes a check with its numbers, never apart)
RESPONSE_PARTS = {
    'force': (
        'a_response',
        'a_response_thin',
        'windows',
        'force_window_yr',
        'force_r2_h1',
        'force_r2_h2',
        'force_jump_dropped',
        'multiplet',
        'force_law',
        'phi0_rad',
        'phi0_coherence',
        'phi0_max_dev_rad',
        'phi0_tol_rad',
        'coorbital',
        'force_line_period_yr',
        'force_line',
        'force_line_offset',
        'force_contaminated',
        'force_raw_check',
        'force_cutoff_yr',
        'force_line_binding_yr',
        'force_r2_null_max',
        'force_null_gain',
        'force_harmonic',
    ),
    'identity': ('identity', 'identity_slope_ratio', 'identity_corr'),
    'b1': ('a_amplitude', 'b1_required_au', 'b1_measured_au', 'b1_ratio'),
}
SOURCE_FIELDS = {'force': 'force_source', 'identity': 'identity_source', 'b1': 'b1_source'}
UNMEASURED_PART = {
    'force': dict(
        a_response=CheckValue.NOT_MEASURED,
        a_response_thin=False,
        windows=[],
        force_window_yr=NAN,
        force_r2_h1=NAN,
        force_r2_h2=NAN,
        force_jump_dropped=0,
        multiplet=None,
        force_law='',
        phi0_rad=NAN,
        phi0_coherence=NAN,
        phi0_max_dev_rad=NAN,
        phi0_tol_rad=NAN,
        coorbital=None,
        force_line_period_yr=NAN,
        force_line='',
        force_line_offset=NAN,
        force_contaminated=False,
        force_raw_check='',
        force_cutoff_yr=NAN,
        force_line_binding_yr=NAN,
        force_r2_null_max=NAN,
        force_null_gain=NAN,
        force_harmonic=None,
    ),
    'identity': dict(identity=CheckValue.NOT_MEASURED, identity_slope_ratio=NAN, identity_corr=NAN),
    'b1': dict(a_amplitude=CheckValue.NOT_MEASURED, b1_required_au=NAN, b1_measured_au=NAN, b1_ratio=NAN),
}


def witnesses(response) -> List[EpisodeResponse]:
    """The witness episodes of the response (`EpisodeResponse.witness`); none for a response without episodes."""
    return [e for e in getattr(response, 'episode_responses', None) or () if e.witness]


def force_law_mixed(response) -> bool:
    """Rule R conflict: a witness exists and another librating episode's force law is measured and fails."""
    episodes = getattr(response, 'episode_responses', None) or ()
    return bool(witnesses(response)) and any(e.a_response == CheckValue.FAIL for e in episodes)


def _take(response: Response, measured: EpisodeResponse, parts: Sequence[str], source: ResponseSource) -> Response:
    fields: Dict[str, object] = {}
    for part in parts:
        fields.update({name: getattr(measured, name) for name in RESPONSE_PARTS[part]})
        fields[SOURCE_FIELDS[part]] = source
    return replace(response, **fields)


def deciding_response(response: Optional[Response], before_jump: bool) -> Optional[Response]:
    """Rule R (W10): the response whose force law (L4), identity (L3) and B1 the status tree reads.

    - A witness is a librating episode (of the window or before the last jump of a) whose force law passes with its
      full margin while B1 does not fail and the identity is not measured to fail there (W11), or passes thin with the
      identity and B1 passing on the same episode (`EpisodeResponse.witness`). With witnesses, the three checks and
      their numbers come from the one with most turning points (the earlier on a tie), except that its not_measured
      never replaces a measured value.
    - Without one the pooled measurement stays: the record's, or under G1e (`before_jump`: the status rests on the
      episodes before the last jump of a) the one pooled over those episodes (`Response.before_jump_response`); the last
      regime's checks and its B5 describe no episode there and are not_measured. None under G1e without any measurement.
    Nothing is counted over windows or episodes and no probabilities are multiplied: one episode decides, or the pool.
    Every other number (K, omega0, C4, B5 outside G1e, slow a_dot) stays the record's."""
    if response is None:
        return None
    found = witnesses(response)
    chosen = response
    if before_jump:
        pooled = getattr(response, 'before_jump_response', None)
        if pooled is None and not found:
            return None
        blank = {name: value for part in UNMEASURED_PART.values() for name, value in part.items()}
        chosen = replace(response, slow_libration=CheckValue.NOT_MEASURED, **blank)
        chosen = _take(chosen, pooled, tuple(RESPONSE_PARTS), ResponseSource.BEFORE_JUMP) if pooled is not None else chosen
    if found:
        witness = max(found, key=lambda e: (e.n_tp, -e.t_start_yr))
        parts = [
            part
            for part in RESPONSE_PARTS
            if not (
                getattr(witness, RESPONSE_PARTS[part][0]) == CheckValue.NOT_MEASURED
                and getattr(chosen, RESPONSE_PARTS[part][0]) in MEASURED
            )
        ]
        chosen = replace(_take(chosen, witness, parts, ResponseSource.WITNESS), witness_episode_id=witness.episode_id)
    return chosen


# --- the record ---------------------------------------------------------------------------------


def force_line(
    spec: Optional[ResonanceSpec], context: Context, period_yr: float, settings: AnalysisSettings
) -> Optional[Tuple[forced_lines_module.ForcedLine, float]]:
    """(line, offset) of the lowest-order forced line of the angle's lattice within the force window's resolution of
    1/P (`forced_lines.coincident_line`, W = mmr_force_window_cycles); None with mmr_force_contamination = 'off',
    without `spec` or for a co-orbital (its own law) or retrograde angle. Planets: the context's mean motions, else
    the table; n_c with the context's slow rate."""
    if settings.mmr_force_contamination == 'off' or spec is None or spec.retrograde or coorbital_planet(spec) is not None:
        return None
    planet_n = {}
    for name in spec.planets:
        value = context.planet_mean_motion_rad_yr.get(name, NAN)
        planet_n[name] = value if np.isfinite(value) else table_mean_motion(name)
    lines = forced_lines_module.lattice_lines(
        spec, planet_n, context.slow_rate_rad_yr, settings.matrix_2body_order_max, settings.matrix_3body_order_max
    )
    return forced_lines_module.coincident_line(lines, period_yr, settings.mmr_force_window_cycles)


def _phase_sigma(windows: Sequence[ForceWindow], t_yr: np.ndarray, adot_f: np.ndarray, cutoff_yr: float) -> float:
    """Median formal error of phi0 [rad]: sigma_resid sqrt(2 / N_eff) / R with N_eff = window / P_c independent samples."""
    errors = []
    for w in windows:
        values = adot_f[samples_between(t_yr, w.t_start_yr, w.t_end_yr)]
        values = values[np.isfinite(values)]
        if not len(values) or not np.isfinite(w.r2) or w.amplitude_au_yr <= 0:
            continue
        residual = np.sqrt(max(1 - w.r2, 0.0) * np.var(values))
        independent = max((w.t_end_yr - w.t_start_yr) / cutoff_yr, 1.0) if cutoff_yr > 0 else 1.0
        errors.append(residual * np.sqrt(2 / independent) / w.amplitude_au_yr)
    return float(np.median(errors)) if errors else NAN


def _varpi_offset(slow: Optional[SlowComponents], t_yr: np.ndarray, planet_rate_rad_yr: float) -> Optional[np.ndarray]:
    """varpi - varpi_P on `t_yr` up to a constant: the body's slow varpi minus g_P t (the planet's phase only rotates C_j)."""
    if slow is None or not np.isfinite(planet_rate_rad_yr):
        return None
    varpi = np.unwrap(np.asarray(slow.varpi_slow, dtype=float))
    if not np.all(np.isfinite(varpi)):
        return None
    return np.interp(t_yr, np.asarray(slow.t_yr, float), varpi) - planet_rate_rad_yr * t_yr


def _best_multiplet(
    t_yr: np.ndarray,
    angle_f: np.ndarray,
    adot_f: np.ndarray,
    pieces: Sequence[Tuple[float, float]],
    force_window_yr: float,
    spec: Optional[ResonanceSpec],
    slow: Optional[SlowComponents],
    context: Context,
    settings: AnalysisSettings,
    zones: Sequence[Tuple[float, float]] = (),
) -> Optional[MultipletFit]:
    """The multiplet law over each planet of the term with a known varpi rate; the best R^2 wins.

    Windows: the modulation period 2 pi / |<varpi_dot> - g_P| (or the force window, if longer), stepped by half,
    inside the response pieces; the two halves of the union when that gives fewer than two windows. Windows that
    overlap `zones` (`jump_zones`) are dropped first; when the halves are needed and one of them overlaps a zone, the
    planet gives no fit.
    """
    if spec is None or spec.retrograde or slow is None:
        return None
    best = None
    planet_varpi = spec.planet_varpi_terms
    for name, planet_p in zip(spec.planets, planet_varpi):
        rate = context.planet_varpi_rate_rad_yr.get(name, NAN)
        harmonics = multiplet_harmonics(spec.body_varpi_coefficient, planet_p)
        offset = _varpi_offset(slow, t_yr, rate)
        if offset is None or len(harmonics) < 2:
            continue
        nu = abs(np.polyfit(t_yr, offset, 1)[0]) if len(t_yr) > 1 else NAN
        modulation = TWO_PI / nu if np.isfinite(nu) and nu > 0 else NAN
        length = np.nanmax([modulation, force_window_yr]) if np.isfinite(modulation) or np.isfinite(force_window_yr) else NAN
        windows = []
        for start, end in pieces:
            windows += sliding_windows(t_yr[samples_between(t_yr, start, end)], length, settings.mmr_force_window_step)
        windows = outside_zones(windows, zones, bounds=tuple)
        if len(windows) < settings.mmr_force_min_windows:
            start, end = pieces[0][0], pieces[-1][1]
            middle = (start + end) / 2
            windows = [(start, middle), (middle, end)]
            if len(outside_zones(windows, zones, bounds=tuple)) < len(windows):
                continue
        fit = multiplet_fit(t_yr, angle_f, adot_f, offset, harmonics, windows, settings.mmr_force_power_share)
        if np.isfinite(fit.r2) and (best is None or fit.r2 > best.r2):
            best = fit
    return best


def _window_centre_coherence(windows: Sequence[ForceWindow], t_yr: np.ndarray, angle_f: np.ndarray) -> float:
    """Coherence of phi0 - (circular mean of sigma_f in the window) over the windows."""
    return circular_coherence([w.phi0_rad - circular_mean(angle_f[samples_between(t_yr, w.t_start_yr, w.t_end_yr)]) for w in windows])


def _force_law(
    t_yr: np.ndarray,
    angle_f: np.ndarray,
    adot_f: np.ndarray,
    pieces: Sequence[Tuple[float, float]],
    window_yr: float,
    settings: AnalysisSettings,
) -> List[ForceWindow]:
    windows = []
    for start, end in pieces:
        piece = samples_between(t_yr, start, end)
        windows += force_windows(t_yr[piece], angle_f[piece], adot_f[piece], window_yr, settings.mmr_force_window_step)
    return windows


def coorbital_response(
    series: Optional[MMRSeries],
    prep: Preprocessed,
    context: Context,
    sampling: Sampling,
    response_cutoff_yr: float,
    angle: np.ndarray,
    adot_f: np.ndarray,
    pieces: Sequence[Tuple[float, float]],
    cycle_edges_yr: Sequence[float],
    settings: AnalysisSettings,
    zones: Sequence[Tuple[float, float]] = (),
) -> Optional[CoorbitalLaw]:
    """The 1:1 force law and energy integral (`coorbital.coorbital_law`) of a 1:1 angle (`law_planet`: prograde or
    retrograde 1/-1); None for any other angle,
    without the body's elements (`series`) or the planet's mass and mean motion, and for an aliased step (sigma_dot is
    the model rate there). `angle` is the unfiltered unwrapped angle (the caller passes `prep.angle_clean`: the model
    is nonlinear in it), `adot_f` the response's a_dot, `cycle_edges_yr` every second turning point (one energy per
    cycle). The model goes through the data's filters: P_c, then the response cutoff when `response_series` applied it.
    The samples inside `zones` (`jump_zones`: around the confirmed jumps of a) are left out of the fit."""
    spec = series.resonance if series is not None else None
    planet = law_planet(spec) if spec is not None else None
    if planet is None or sampling.verdict in ALIASED:
        return None
    inputs = coorbital_inputs(series, context, planet)
    if inputs is None:
        return None
    mu, n_planet, planet_elements = inputs
    t = np.asarray(prep.t_yr, float)
    cutoffs = (prep.cutoff.cutoff_yr,)
    if response_cutoff_yr > prep.cutoff.cutoff_yr and response_cutoff_yr > 2 * np.median(np.diff(t)):
        cutoffs += (response_cutoff_yr,)
    data = CoorbitalInput(
        t_yr=t,
        angle=np.asarray(angle, float),
        a=np.asarray(series.a, float),
        adot=adot_f,
        mask=intervals_mask(t, pieces) & zones_mask(t, zones),
        pieces=pieces,
        e=series.e,
        inc=series.inc,
        omega=series.omega,
        Omega=series.Omega,
        planet=planet,
        planet_mass=mu,
        planet_mean_motion_rad_yr=n_planet,
        body_coefficient=spec.body_coefficient,
        cutoffs_yr=cutoffs,
        planet_elements=planet_elements,
        cycle_edges_yr=tuple(np.asarray(cycle_edges_yr, float)),
        retrograde=spec.retrograde,
        omega_coefficient=spec.body_node_coefficient - 1 if spec.retrograde else 1,  # 2k - 1 of 1J-1r<k>
        block_yr=float(getattr(prep, 'elements_block_yr', NAN)),
    )
    return coorbital_module.coorbital_law(data, settings)


def coorbital_check(law: CoorbitalLaw, amplitude_max_rad: float) -> Tuple[CheckValue, bool]:
    """check_mmr_a_response of a 1:1 angle from its own law: not_applicable without turning points, else the law's
    check (`coorbital.force_check`) at any amplitude. The harmonic floor mmr_a_response_min_amp_deg does not apply: it
    exists because a fitted sine and a line differ by < 10 % below it, so a harmonic law with a free K says nothing
    there. The 1:1 law has no free shape and its scale K_theory is the planet's mass, so even a small (linear) tadpole is
    measured: K / K_theory ~ 1 means the period is the one the planet's mass gives (omega^2 = 27/4 mu n^2,
    Murray & Dermott 1999 §3.9), which a kinematic oscillation of sigma does not reproduce."""
    if not np.isfinite(amplitude_max_rad):
        return CheckValue.NOT_APPLICABLE, False
    return law.force_check, law.force_thin


def with_coorbital_potential(
    energy: Optional[EnergyAnalysis], response: Optional[Response], geometry: Geometry, filter_too_slow: bool
) -> Optional[EnergyAnalysis]:
    """The energy with R0 with a of a 1:1 angle: the co-orbital potential check of `response`'s law
    (`coorbital.potential_check`: the energy integral, not_measured when only the conservation of E fails) in place of
    the harmonic V_E ~ V_F test, whose potentials (a few harmonics of sigma) cannot hold the 1:1 potential (singular at
    0 for e = i = 0, far from harmonic for horseshoes); the law goes to `EnergyAnalysis.potential_law`. Only on
    accepted episodes (the integral is fitted cycle by cycle there; a circulation keeps its R0). `energy` itself is
    not changed (a copy; it keeps the harmonic result), and it is returned as it is for another angle, without an
    energy analysis, or under filter_too_slow (R0 is shape-dependent: not_measured, DELTA item 1).

    The orchestration passes rule R's response (`deciding_response`) once the episodes are judged: the 1:1 law
    carries the force law and the energy integral together, and the status tree reads the witness's law, so R0 is read
    from that same law (W20 round D: 2024 WJ took R0 pass from the pooled law, energy R^2 0.82 on cycles cut at the
    record's turning points, and printed the witness's 0.043); a deciding response without a 1:1 law keeps the
    harmonic R0 (W20 round E, review D #2: the pooled law is not the one the tree reads)."""
    law = response.coorbital if response is not None else None
    if energy is None or law is None or not geometry.episodes or filter_too_slow:
        return energy
    return replace(energy, with_slow=law.potential_check, potential_law=law)


def _deciding_episode_yr(geometry: Geometry, settings: AnalysisSettings) -> Tuple[float, float]:
    """[start, end] of the episode the verdict rests on (`decide.decisive_episode`), (nan, nan) without one."""
    from resonances.resonance.analysis.mmr.decide import decisive_episode  # decide imports owner, owner this module

    episode = decisive_episode(geometry, settings)
    return (episode.t_start_yr, episode.t_end_yr) if episode is not None else (NAN, NAN)


def analyse_response(
    prep: Preprocessed,
    slow: SlowComponents,
    geometry: Geometry,
    context: Context,
    sampling: Sampling,
    settings: AnalysisSettings,
    spec: Optional[ResonanceSpec] = None,
    series: Optional[MMRSeries] = None,
) -> Response:
    """Every number of SPEC §4.5 on the episode window (owner-based B5 completed later by the orchestration).

    `spec` (optional) gives k and the varpi coefficients of the multiplet law; without it k comes back from c and
    the multiplet law is not fitted. B5 is excluded by data only here (the owner route is completed by the
    orchestration); there is no route by theory (W3-F3, `slow_libration_check`).

    A 1:1 angle (`law_planet`: prograde, or retrograde 1/-1) with `series` (the body's e, i, omega, Omega) is judged by its own law
    (`coorbital_response`): check_mmr_a_response, its thin flag and the two half R^2 come from it, `force_law` is
    'coorbital' and the phase spread is not measured (the shape is fixed by the theory, only the scale K is fitted);
    the harmonic windows stay as columns. The law is `response.coorbital` (None for other angles).
    """
    t = np.asarray(prep.t_yr, float)
    c = context.identity_coefficient
    k = strength_module.body_coefficient(context, spec)
    period = geometry.period_cycle_yr
    filtered = response_series(prep, period, settings)
    angle_f, rate_f, a_f, adot_f, smoothing = filtered
    amplitude_max, amplitude_med = _amplitudes(geometry)
    pieces = _response_pieces(geometry)
    mask = intervals_mask(t, pieces)
    shift = well_shift(t, geometry.runs)  # psi on the episodes accepted in the frame of their moving well
    pooled = measure_pieces(
        prep,
        context,
        settings,
        pieces,
        period,
        (amplitude_max, amplitude_med),
        geometry.turning_points.times_yr[::2],
        judge=ResponseJudge(sampling.verdict in ALIASED, prep.filter_too_slow, sampling, series),
        spec=spec,
        slow=slow,
        filtered=filtered,
        guard=JumpGuard.from_geometry(geometry),
        angle_shift=shift,
    )

    # the a-wave of the identity
    longest = max(pieces, key=lambda p: p[1] - p[0])
    in_longest = samples_between(t, *longest)  # contiguous: the band-pass needs a uniform grid
    centre = np.interp(t[in_longest], np.asarray(geometry.centre.t_yr, float), np.asarray(geometry.centre.centre_rad, float))
    wave_corr, wave_ratio = identity_wave(
        t[in_longest], a_f[in_longest], k, angle_f[in_longest], centre, geometry.centre.window_yr, settings.filter_order
    )

    # strength of the force: K over the windows, or one fit over the whole response window (circulation)
    force, force_r2_whole = pooled.force_amplitude_au_yr, NAN
    if not pooled.windows:
        whole = force_windows(t[mask], angle_f[mask], adot_f[mask], float(np.ptp(t[mask])) if np.any(mask) else NAN, 1.0)
        force = whole[0].amplitude_au_yr if whole else NAN
        force_r2_whole = whole[0].r2 if whole else NAN  # how much of a_dot that K explains (C4 reads it, W22 R-C4)
    omega0_force = omega0_from_force(c, force)
    t0 = small_period(period, amplitude_med) if np.isfinite(period) and np.isfinite(amplitude_med) else NAN
    omega0_ratio = omega0_force / (TWO_PI / t0) if np.isfinite(t0) and t0 > 0 else NAN

    # C4 forced wave of a circulation
    circ_ratio, wave_pred = circulation_strength(rate_f[mask], omega0_force, force)
    a_sin, b_cos, _, _ = _sine_fit(angle_f[mask], a_f[mask])
    wave_meas = float(np.hypot(a_sin, b_cos))

    # slow a_dot and B5
    adot_r2 = slow_adot_r2(slow)
    slow_t = np.asarray(slow.t_yr, float)
    centre_slow = np.interp(slow_t, np.asarray(geometry.centre.t_yr, float), np.asarray(geometry.centre.centre_rad, float))
    bound = slow_libration_bound(slow_t, centre_slow, slow.a_slow, c, slow.window)
    in_slow = filtering.window_mask(slow_t, slow.window)
    a_slow_range = float(np.ptp(np.asarray(slow.a_slow, float)[in_slow])) if np.any(in_slow) else NAN
    track = geometry.centre
    centre_pass = (track.max_pass_rev_lo, track.max_pass_t0_yr, track.max_pass_t1_yr)
    episode_yr = _deciding_episode_yr(geometry, settings)
    slow_check, route, exact = slow_libration_check(bound, NAN, NAN, a_slow_range, settings, centre_pass, episode_yr)

    response = Response(
        **{name: getattr(pooled, name) for names in RESPONSE_PARTS.values() for name in names},
        identity_wave_corr=wave_corr,
        identity_amp_ratio=wave_ratio,
        force_amplitude_au_yr=force,
        phi0_minus_centre_coherence=_window_centre_coherence(pooled.windows, t, angle_f - shift if shift is not None else angle_f),
        omega0_force_rad_yr=omega0_force,
        omega0_ratio=omega0_ratio,
        circulation_strength_ratio=circ_ratio,
        forced_wave_pred_au=wave_pred,
        forced_wave_meas_au=wave_meas,
        slow_adot_r2=adot_r2,
        b5_arc_rad=bound['arc'],
        b5_force_max_au_yr=bound['force_max'],
        b5_omega0_max_rad_yr=bound['omega0_max'],
        b5_period_min_yr=bound['period_min'],
        centre_rate_rad_yr=bound['centre_rate'],
        centre_rate_err_rad_yr=bound['centre_rate_err'],
        b5_ratio=bound['ratio'],
        b5_owner_halfwidth_au=NAN,
        b5_owner_ratio=NAN,
        b5_route=route,
        slow_libration=slow_check,
        exact_commensurability=exact,
        force_r2_whole=force_r2_whole,
    )
    return response


def harmonic_columns(law: Optional[HarmonicLaw]) -> dict:
    """`rsp_force_harm_*`: the N-harmonic law the response's L4 consulted (CS12; empty when it was not)."""
    blank = law is None
    return {
        'rsp_force_harm_N': NAN if blank else law.n_harmonics,
        'rsp_force_harm_R2': NAN if blank else law.r2,
        'rsp_force_harm_gain': NAN if blank else law.gain,
        'rsp_force_harm_coherence': NAN if blank else law.coherence,
        'rsp_force_harm_null_max': NAN if blank else law.null_max,
        'rsp_force_harm_null_gain': NAN if blank else law.null_gain,
        'rsp_force_harm_oos': NAN if blank else law.oos_r2,
        'rsp_force_harm_c2_over_c1': NAN if blank else law.c2_over_c1,
        'rsp_force_harm_reason': '' if blank else harmonic_reason(law),
    }


def response_columns(response: Response) -> dict:
    """CSV columns `rsp_*` (the body of `Response.to_columns`)."""
    deg = np.degrees
    multi = response.multiplet
    return {
        'rsp_identity_slope_ratio': response.identity_slope_ratio,
        'rsp_identity_corr': response.identity_corr,
        'rsp_identity_wave_corr': response.identity_wave_corr,
        'rsp_identity_amp_ratio': response.identity_amp_ratio,
        'rsp_force_window_yr': response.force_window_yr,
        'rsp_force_n_windows': len(response.windows),
        'rsp_force_R2_h1': response.force_r2_h1,
        'rsp_force_R2_h2': response.force_r2_h2,
        'rsp_force_n_windows_jump': response.force_jump_dropped,
        'rsp_force_R2_multi': multi.r2 if multi is not None else NAN,
        'rsp_force_R2_whole': response.force_r2_whole,
        'rsp_force_multi_harmonics': ';'.join(str(j) for j in multi.harmonics) if multi is not None else '',
        'rsp_force_multi_coherence': multi.coefficient_coherence if multi is not None else NAN,
        'rsp_force_law': response.force_law,
        'rsp_force_K_au_yr': response.force_amplitude_au_yr,
        'rsp_force_phi0_deg': float(deg(response.phi0_rad) % 360) if np.isfinite(response.phi0_rad) else NAN,
        'rsp_phi0_coherence': response.phi0_coherence,
        'rsp_phi0_max_dev_deg': float(deg(response.phi0_max_dev_rad)),
        'rsp_phi0_tol_deg': float(deg(response.phi0_tol_rad)),
        'rsp_phi0_minus_centre_coherence': response.phi0_minus_centre_coherence,
        'rsp_force_thin': response.a_response_thin,
        'rsp_force_line_period_yr': response.force_line_period_yr,
        'rsp_force_line': response.force_line,
        'rsp_force_line_offset': response.force_line_offset,
        'rsp_force_contaminated': response.force_contaminated,
        'rsp_force_raw_check': response.force_raw_check,
        'rsp_response_cutoff_ep_yr': response.force_cutoff_yr,
        'rsp_line_binding_yr': response.force_line_binding_yr,
        'rsp_force_R2_null_max': response.force_r2_null_max,
        'rsp_force_null_gain': response.force_null_gain,
        **harmonic_columns(response.force_harmonic),
        'rsp_omega0_K_rad_yr': response.omega0_force_rad_yr,
        'rsp_omega0_ratio': response.omega0_ratio,
        'rsp_b1_req_au': response.b1_required_au,
        'rsp_b1_meas_au': response.b1_measured_au,
        'rsp_b1_ratio': response.b1_ratio,
        'rsp_circ_strength_ratio': response.circulation_strength_ratio,
        'rsp_forced_wave_pred_au': response.forced_wave_pred_au,
        'rsp_forced_wave_meas_au': response.forced_wave_meas_au,
        'rsp_slow_adot_R2': response.slow_adot_r2,
        'rsp_b5_arc_rad': response.b5_arc_rad,
        'rsp_b5_K_max_au_yr': response.b5_force_max_au_yr,
        'rsp_b5_omega0_max_rad_yr': response.b5_omega0_max_rad_yr,
        'rsp_b5_P_min_kyr': response.b5_period_min_yr / 1e3,
        'rsp_centre_rate_rad_kyr': response.centre_rate_rad_yr * 1e3,
        'rsp_centre_rate_err_rad_kyr': response.centre_rate_err_rad_yr * 1e3,
        'rsp_b5_ratio': response.b5_ratio,
        'rsp_b5_owner_halfwidth_au': response.b5_owner_halfwidth_au,
        'rsp_b5_owner_ratio': response.b5_owner_ratio,
        'rsp_b5_route': response.b5_route,
        'rsp_exact_commensurability': response.exact_commensurability,
        'rsp_flags': ';'.join(response.flags),
        'rsp_force_source': str(response.force_source),
        'rsp_identity_source': str(response.identity_source),
        'rsp_b1_source': str(response.b1_source),
        'rsp_witness_episode_id': response.witness_episode_id,
        'rsp_n_witnesses': len(witnesses(response)),
        'rsp_n_episode_force_fail': sum(1 for e in response.episode_responses if e.a_response == CheckValue.FAIL),
        **coorbital_module.coorbital_columns(response.coorbital),
    }


# --- episodes -----------------------------------------------------------------------------------


def onset_bands(
    t_yr: np.ndarray,
    a_f: np.ndarray,
    period_yr: float,
    t_start_yr: float,
    t_end_yr: float,
    filter_order: Optional[int] = None,
    clear: Tuple[float, float] = (-np.inf, np.inf),
    before: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float]:
    """O1: std of a_f in the band [P/2, 2P] in stretches of length 2P before, inside and after an episode [au] (nan if absent).

    The band-pass is lowpass(P/2) - lowpass(2P) on the whole record; 'inside' is the episode; 'before' is
    [start - 2P, start] unless `before` gives the background (`onset_group`: before the episode's group, cut at an
    earlier episode); every stretch is cut to `clear` (`onset_clear`: the episode's regime clear of the band-pass
    edges); a stretch covering less than one period is absent.
    """
    t_yr, a_f = np.asarray(t_yr, float), np.asarray(a_f, float)
    if not (np.isfinite(period_yr) and period_yr > 0) or len(t_yr) < MIN_FIT_SAMPLES:
        return NAN, NAN, NAN
    order = filter_order if filter_order is not None else AnalysisSettings().filter_order
    step = float(np.median(np.diff(t_yr)))
    if period_yr / ONSET_BAND_FACTOR <= 2 * step:
        return NAN, NAN, NAN
    band = filtering.band_pass(t_yr, a_f, period_yr, ONSET_BAND_FACTOR, order)
    stretch = 2 * period_yr

    def std(start, end):
        values = band[samples_between(t_yr, start, end)]
        covered = (min(end, t_yr[-1]) - max(start, t_yr[0])) if len(values) else 0.0
        return float(np.std(values)) if covered >= period_yr else NAN

    low, high = clear

    def cut(start, end):
        return std(max(start, low), min(end, high))

    background = before if before is not None else (t_start_yr - stretch, t_start_yr)
    return cut(*background), cut(t_start_yr, t_end_yr), cut(t_end_yr, t_end_yr + stretch)


ONSET_EDGE, ONSET_JUMP, ONSET_EPISODE = 'record_edge', 'jump', 'episode'


@dataclass(frozen=True)
class OnsetBackground:
    """O1's stretch before an episode (`onset_group`): [start, end] [yr], the earlier episodes of its group (ids,
    latest first) and the bound an earlier episode with a circulation after it cuts the stretch at (-inf: none)."""

    start_yr: float
    end_yr: float
    group_ids: Tuple[int, ...] = ()
    episode_cut_yr: float = -np.inf


def onset_group(
    run: Episode,
    period_yr: float,
    accepted: Sequence[Episode],
    circulates: Callable[[Episode, Episode], bool],
    low_yr: float = -np.inf,
) -> OnsetBackground:
    """O1's background before `run` (author 2026-10-23, Q2; 2013 VA14 60 kyr: the 2P before #5 held all of #4): never
    an accepted episode of the same angle. The latest accepted episode of `accepted` that ends inside the 2P before
    the group's first episode (and after `low_yr`, the regime's clear bound: `onset_clear`) joins the group when the
    gap between them holds no observed circulation (`circulates`: a full turn, `geometry.gap_circulates`), and the
    search repeats before it; one with a circulation after it cuts the background at its end. The background is
    [max(first start - 2P, cut), first start], the period that of `run`. The episodes themselves stay apart: only the
    stretch O1 reads changes."""
    first, joined, cut = run, [], -np.inf
    if not (np.isfinite(period_yr) and period_yr > 0 and np.isfinite(run.t_start_yr)):
        return OnsetBackground(run.t_start_yr - 2 * period_yr, run.t_start_yr)
    others = [e for e in accepted if e.episode_id != run.episode_id and np.isfinite(e.t_end_yr)]
    while True:
        reach = max(first.t_start_yr - 2 * period_yr, low_yr)
        inside = [e for e in others if e.t_start_yr < first.t_start_yr and reach < e.t_end_yr <= first.t_start_yr]
        if not inside:
            break
        previous = max(inside, key=lambda e: e.t_end_yr)
        if circulates(previous, first):
            cut = float(previous.t_end_yr)
            break
        joined.append(previous.episode_id)
        first = previous
    start = max(first.t_start_yr - 2 * period_yr, cut)
    return OnsetBackground(float(start), float(first.t_start_yr), tuple(joined), cut)


def onset_clear(
    t_yr: np.ndarray,
    period_yr: float,
    start_yr: float,
    end_yr: float,
    guard: Optional[JumpGuard],
    settings: AnalysisSettings,
    background: Optional[OnsetBackground] = None,
) -> Tuple[Tuple[float, float], str]:
    """((low, high), why O1 cannot be measured) for `onset_bands`: the samples of the episode's regime (between the
    confirmed jumps of a on either side of the episode's middle, a regime starting tau/2 after its jump and ending
    tau/2 before the next) kept filter_edge_trim_cutoffs x 2P from both ends of the record and from those jumps, 2P
    being the band-pass's slowest cutoff (the reach B1 clears too). The reason is '' while at least one period of the
    stretch before the episode (`background`, `onset_group`: before its group, cut at an earlier episode; default
    [start - 2P, start]) and of the episode itself is left, else ONSET_JUMP, ONSET_EDGE or ONSET_EPISODE (an earlier
    episode of the angle with a circulation after it), the bound that cut it most. A step of a just before the episode
    (277617 1M-1: a series of Mars encounters) breaks the test's premise, a capture from a regime without that wave;
    the filter's ramp at the record's start makes a wave of its own (385533 7N-12: O1 7.18)."""
    t = np.asarray(t_yr, float)
    if not (len(t) and np.isfinite(period_yr) and period_yr > 0 and np.isfinite(start_yr) and np.isfinite(end_yr)):
        return (-np.inf, np.inf), ''
    reach = settings.filter_edge_trim_cutoffs * 2 * period_yr
    zones = [(j - guard.half_width_yr(i), j + guard.half_width_yr(i)) for i, j in enumerate(guard.times_yr)] if guard is not None else []
    middle = (start_yr + end_yr) / 2
    prev_end = max((hi for lo, hi in zones if (lo + hi) / 2 <= middle), default=NAN)
    next_start = min((lo for lo, hi in zones if (lo + hi) / 2 > middle), default=NAN)
    edge_low, edge_high = float(t[0]) + reach, float(t[-1]) - reach
    jump_low = prev_end + reach if np.isfinite(prev_end) else -np.inf
    jump_high = next_start - reach if np.isfinite(next_start) else np.inf
    low, high = max(edge_low, jump_low), min(edge_high, jump_high)
    low_reason = ONSET_JUMP if jump_low >= edge_low else ONSET_EDGE
    high_reason = ONSET_JUMP if jump_high <= edge_high else ONSET_EDGE
    group_start = background.end_yr if background is not None else start_yr
    episode_cut = background.episode_cut_yr if background is not None else -np.inf
    reason = ''
    for start, end, cut in ((group_start - 2 * period_yr, group_start, episode_cut), (start_yr, end_yr, -np.inf)):
        if min(end, high) - max(start, low, cut) < period_yr:
            low_loss, high_loss, cut_loss = max(low - start, 0.0), max(end - high, 0.0), max(cut - start, 0.0)
            if cut_loss > max(low_loss, high_loss):
                reason = ONSET_EPISODE
            else:
                reason = low_reason if low_loss >= high_loss else high_reason
            break
    return (low, high), reason


def _episode_zone_class(classes: Sequence[CycleClass]) -> str:
    """'zone' if any cycle is in the zone, else 'not_measured' if a cycle is not measured (a loop around several wells:
    it may be the one in the zone), else the common class of the cycles ('' without cycles)."""
    if not classes:
        return ''
    if CycleClass.ZONE in classes:
        return str(CycleClass.ZONE)
    if CycleClass.NOT_MEASURED in classes:
        return str(CycleClass.NOT_MEASURED)
    return str(CycleClass.DEEP) if CycleClass.DEEP in classes else str(CycleClass.ABOVE)


def _onset(t, a_f, run: Episode, period_med: float, guard, settings: AnalysisSettings, accepted, circulates):
    """O1's numbers for `episode_physics`: ((before, inside, after) band stds, why not measured, the background
    (`onset_group` when `circulates` is given), its clear bounds (nan when nothing of it is clear))."""
    times = np.asarray(run.tp_times_yr, float)
    start, end = (float(times[0]), float(times[-1])) if len(times) else (NAN, NAN)
    clear, short = onset_clear(t, period_med, start, end, guard, settings)
    background = OnsetBackground(start - 2 * period_med, start)
    if circulates is not None and np.isfinite(start):
        background = onset_group(run, period_med, accepted, circulates, clear[0])
        clear, short = onset_clear(t, period_med, start, end, guard, settings, background)
    bands = (NAN,) * 3
    if np.isfinite(start):
        stretch = (background.start_yr, background.end_yr)
        bands = onset_bands(t, a_f, period_med, start, end, settings.filter_order, clear, stretch)
    bounds = (max(background.start_yr, clear[0]), min(background.end_yr, clear[1]))
    if not bounds[1] > bounds[0]:  # nothing of the stretch is clear (a group starting at the record's edge)
        bounds = (NAN, NAN)
    return bands, short, background, bounds


def episode_physics(
    run: Episode,
    prep: Preprocessed,
    context: Context,
    energy: Optional[EnergyAnalysis],
    settings: AnalysisSettings,
    spec: Optional[ResonanceSpec] = None,
    slow: Optional[SlowComponents] = None,
    judge: Optional[ResponseJudge] = None,
    guard: Optional[JumpGuard] = None,
    accepted: Sequence[Episode] = (),
    circulates: Optional[Callable[[Episode, Episode], bool]] = None,
) -> EpisodePhysics:
    """The pendulum, energy and response numbers of one run of turning points (accepted or rejected); `judge` (a
    librating episode, rule R) adds the check values of its force law, identity and B1 (`measure_pieces`). `guard`
    (the confirmed jumps of a) keeps the force law and the onset stretches off them (`jump_zones`, `onset_clear`).
    `accepted` (the angle's accepted episodes) and `circulates` (a full turn between two of them) keep them out of
    O1's background (`onset_group`); without `circulates` the background is the 2P before the run."""
    t = np.asarray(prep.t_yr, float)
    c = context.identity_coefficient
    k = strength_module.body_coefficient(context, spec)
    times, values = np.asarray(run.tp_times_yr, float), np.asarray(run.tp_values_rad, float)
    half = half_amplitude_values(values)
    periods = times[2:] - times[:-2]
    period_med = float(np.median(periods)) if len(periods) else NAN
    angle_f, rate_f, a_f, adot_f, smoothing = response_series(prep, period_med, settings)
    cycle_amp = (half[1:] + half[:-1]) / 2 if len(half) > 1 else np.array([])
    usable = (cycle_amp < np.pi) & np.isfinite(periods) if len(periods) else np.array([], dtype=bool)
    t0_cycles = small_period(periods[usable], cycle_amp[usable]) if np.any(usable) else np.array([])
    t0_cycles = np.atleast_1d(t0_cycles)
    t0_min = float(np.min(t0_cycles)) if len(t0_cycles) else NAN
    t0_max = float(np.max(t0_cycles)) if len(t0_cycles) else NAN
    t0 = float(np.median(t0_cycles)) if len(t0_cycles) else NAN
    omega0 = TWO_PI / t0 if np.isfinite(t0) and t0 > 0 else NAN

    start, end = (float(times[0]), float(times[-1])) if len(times) else (NAN, NAN)
    mask = samples_between(t, start, end)
    inside = mask.stop > mask.start
    amp_max = float(np.max(half)) if len(half) else NAN
    amp_med = float(np.median(half)) if len(half) else NAN
    peak_rate = float(np.nanmax(np.abs(rate_f[mask]))) if inside else NAN
    omega0_vmax = float(peak_rate / (2 * np.sin(amp_max / 2))) if np.isfinite(amp_max) and amp_max > 0 else NAN
    vmax_ratio = peak_rate / (2 * omega0) if np.isfinite(omega0) and omega0 > 0 else NAN

    excursion_meas = NAN
    if inside and len(times) > 2:
        cycles = [a_f[samples_between(t, a, b)] for a, b in zip(times[:-2], times[2:])]
        cycle_ptp = [np.ptp(cycle) / 2 for cycle in cycles if len(cycle)]
        excursion_meas = float(np.median(cycle_ptp)) if cycle_ptp else NAN
    (before, inside, after), onset_short, background, (before_t0, before_t1) = _onset(
        t, a_f, run, period_med, guard, settings, accepted, circulates
    )

    h_values, j_values, classes = [], [], []
    if energy is not None:
        cycle_classes = list(energy.cycle_classes) + [None] * (len(energy.cycles) - len(energy.cycle_classes))
        for cycle, cls in zip(energy.cycles, cycle_classes):
            middle = (cycle.t_start_yr + cycle.t_end_yr) / 2
            if start <= middle <= end:
                h_values.append(cycle.h)
                j_values.append(cycle.action_ratio)
                if cls is not None:
                    classes.append(CycleClass(cls))

    pieces = [(start, end)] if np.isfinite(start) else []
    response = measure_pieces(
        prep,
        context,
        settings,
        pieces,
        period_med,
        (amp_max, amp_med),
        times[::2],
        judge=judge,
        spec=spec,
        slow=slow,
        filtered=(angle_f, rate_f, a_f, adot_f, smoothing),
        episode_id=run.episode_id,
        before_jump=bool(run.before_last_jump),
        n_tp=run.n_tp,
        guard=guard,
        angle_shift=well_shift(t, [run]),
    )

    def finite_stat(values, function):
        values = [v for v in values if np.isfinite(v)]
        return float(function(values)) if values else NAN

    return EpisodePhysics(
        episode_id=run.episode_id,
        small_period_min_yr=t0_min,
        small_period_max_yr=t0_max,
        small_period_ratio=t0_max / t0_min if np.isfinite(t0_min) and t0_min > 0 else NAN,
        omega0_rad_yr=omega0,
        omega0_vmax_rad_yr=omega0_vmax,
        omega0_vmax_ratio=vmax_ratio,
        half_width_au=half_width(omega0, c),
        a_excursion_pred_au=excursion(omega0, amp_med, c),
        a_excursion_meas_au=excursion_meas,
        mu_eff=strength_module.mu_eff_required(t0, k, context.mean_motion_rad_yr),
        a_band_std_before_au=before,
        a_band_std_in_au=inside,
        a_band_std_after_au=after,
        onset_short=onset_short if not (np.isfinite(before) and np.isfinite(inside)) else '',
        onset_before_t0_yr=before_t0,
        onset_before_t1_yr=before_t1,
        onset_group_ids=background.group_ids,
        h_min=finite_stat(h_values, np.min),
        h_med=finite_stat(h_values, np.median),
        h_max=finite_stat(h_values, np.max),
        action_ratio_min=finite_stat(j_values, np.min),
        action_ratio_max=finite_stat(j_values, np.max),
        zone_class=_episode_zone_class(classes),
        response=response,
    )


def analyse_episodes(
    prep: Preprocessed,
    geometry: Geometry,
    context: Context,
    energy: Optional[EnergyAnalysis],
    settings: AnalysisSettings,
    spec: Optional[ResonanceSpec] = None,
    slow: Optional[SlowComponents] = None,
    sampling: Optional[Sampling] = None,
    series: Optional[MMRSeries] = None,
    judged_ids: Sequence[int] = (),
) -> List[EpisodePhysics]:
    """Per episode (accepted or rejected): T0 = P/F(A) per amplitude regime, omega0 (period and peak rate), half-width
    2 omega0/|c|, predicted and measured a excursion, mu_eff, onset bands, h and J/J_s range and zone class, force law and
    identity on the episode window, B1 ratio.

    An amplitude regime is one full cycle (three TPs): T0_i = P_i / F(A_i) with A_i the mean of its two half-amplitudes;
    cycles over the top (A >= pi) are left out. omega0 = 2 pi / median T0; omega0_vmax = max|sigma_dot| / (2 sin(A_max/2));
    omega0_vmax_ratio = max|sigma_dot| / (2 omega0) (SPEC §4.4). The measured a excursion is the median half-range of
    a_f over the cycles. `spec` and `slow` (optional) add the multiplet R^2. The runs in `judged_ids` (the librating
    episodes, rule R) are judged with `sampling` and `series`: L4, L3 and B1 with the filter guard at their own period.
    The confirmed jumps of a in `geometry` guard the force law and the onset stretches (`JumpGuard`); its accepted
    runs but the forced waves stay out of every run's onset background (`onset_group`, a full turn of sigma_f between
    two of them: `geometry.gap_circulates`).
    """
    judged = set(judged_ids) if sampling is not None else set()
    guard = JumpGuard.from_geometry(geometry)
    forced = geometry.forced_wave_ids
    accepted = [run for run in geometry.runs if run.accepted and run.episode_id not in forced]

    def circulates(left: Episode, right: Episode) -> bool:
        return gap_circulates(geometry, prep.t_yr, prep.angle_f, left, right)

    return [
        episode_physics(
            run,
            prep,
            context,
            energy,
            settings,
            spec,
            slow,
            episode_judge(run.period_med_yr, prep, sampling, series, settings) if run.episode_id in judged else None,
            guard,
            accepted,
            circulates,
        )
        for run in geometry.runs
    ]
