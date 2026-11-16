"""Forced lines of the angle's lattice at a libration's period: the contamination of the force law (L4, W15).

Inside a libration of sigma = sum_j k_j lambda_j + k lambda + (slow) the mean sigma_dot is zero, so the body's mean
motion sits at the commensurate value n_c = -(sum_j k_j n_j + slow_rate) / k, and every argument
m lambda + sum_j l_j lambda_j turns there at nu = m n_c + sum_j l_j n_j. A term of the disturbing function with m != 0
moves a at that frequency (a_dot = 2 / (n a) dR/dlambda) whether or not sigma librates, and through the identity
(sigma_ddot = c a_dot) it moves sigma as a free libration at 2 pi / P would: a_dot = -(nu^2 / c) (sigma - sigma_c), the
harmonic law. Over a window of W = mmr_force_window_cycles periods the force-law fit resolves frequencies only to
1 / (W P), so a line with

    |1/T - 1/P| < 1 / (W P)        (`coincident_line`; the offset in units of 1 / (W P) below RESOLUTION)

makes the fit unable to tell the libration from the forced wave (Codex, W10 cutoff/codex/ANSWER.md: a strong flag of
non-identifiability, not a proof of a kinematic origin). 101703 41J-14: P_e = 166.2 yr, the line lambda - 3 lambda_J
(the 3:1, 3 x 14 - 41 = 1) at 164.4 yr; its 'libration' is that forced wave.

The lattice S1 (`lattice_lines`): the angle's own planets, 1 <= m <= |k|, |l_j| <= |k_j|, d'Alembert order
|m + sum_j l_j| up to the catalogue's (matrix_2body_order_max for a term with one planet, matrix_3body_order_max with
two), not proportional to sigma's lambda vector (sigma itself, its harmonics and subresonances). Harmonics of a line
are kept: j phi is a term of the expansion as well. The wider grid of the owner search hits about half of all periods
by chance and touched real resonances (1162, 1362: W12); S1 hit 39 of 231 measured harmonic force laws in 17 records,
for three-body angles with a chance of 0.06 (q90 0.26) of a random period (impl/w12/forced). Retrograde angles have no
lines here (lambda* conventions). Pure: numbers in, numbers out.

Line shares (`line_share`, W28 B8b, card g07-3): which lines of S1 carry a_dot on an episode, for the right column when
the own force law fails. a_dot - <a_dot> is fitted by least squares with sin/cos of every line in the band at once; a
line's share is the variance of its fitted part over the variance of a_dot. Lines closer than the episode's resolution
1/T are one regressor and are all named (with 'P_lib' when the libration's own period coincides); the strongest
spectral peak of what the lines leave is fitted with them, named 'P_lib' when it is the libration's own period, and
when it carries more than any line the column says that name or 'unidentified'. `share_all` is R^2 of the lattice
lines alone. A two-body line m lambda + l lambda_P has
nu = j n_P / |k| (j = l k - m k_P, up to the slow rate), i.e. the period P_conj / |j| with the conjunction cycle
P_conj = |k| P_P (the configuration repeats after |k_P| orbits of the body): 2005 GQ107 40M-57 carries the 107-yr line
7M-10 = 57 P_M = P_conj. The periods are taken at the body's mean motion measured on the episode (n_c inside a
libration; a circulating body is not at n_c), the names at the commensurability. Columns only: nothing here decides a
status.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis.mmr.models import LineShare, ResonanceSpec, planet_letter, planet_terms

TWO_PI = 2 * math.pi
RESOLUTION = 1.0  # |1/T - 1/P| in units of the window's frequency resolution 1 / (W P): its definition, not a threshold
# line shares (numerical, not thresholds): samples per shortest period of the band after decimation (twice Nyquist),
# rows per block of the normal equations, zero padding of the residual's spectrum, the fewest samples of a fit
SAMPLES_PER_SHORTEST = 4
GRAM_BLOCK_ROWS = 8192
SPECTRUM_OVERSAMPLE = 4
MIN_SHARE_SAMPLES = 8
REFINE_POINTS = 21  # frequencies tried inside a line's resolution element (`refined_frequency`)
UNIDENTIFIED = 'unidentified'


@dataclass(frozen=True)
class ForcedLine:
    """One argument m lambda + sum_j l_j lambda_j of the lattice: its period at n_c and its d'Alembert order."""

    period_yr: float
    body_coefficient: int  # m
    planet_coefficients: Tuple[int, ...]  # l_j, in the order of the angle's planets
    order: int
    planets: Tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """The argument in package notation with its first planet coefficient positive: (1, -3) of Jupiter -> '3J-1'."""
        sign = -1 if next((c for c in self.planet_coefficients if c != 0), 0) < 0 else 1
        kept = [(name, c) for name, c in zip(self.planets, self.planet_coefficients) if c != 0]
        head = planet_terms([name for name, _ in kept], [sign * c for _, c in kept])
        return f'{head}{sign * self.body_coefficient:+d}'


def commensurate_mean_motion(spec: ResonanceSpec, planet_n: Mapping[str, float], slow_rate_rad_yr: float) -> float:
    """n_c = -(sum_j k_j n_j + slow_rate) / k [rad/yr]: the body's mean motion where sigma_dot = 0 (slow rate nan = 0)."""
    planets = sum(k * planet_n[name] for name, k in zip(spec.planets, spec.planet_coefficients))
    slow = slow_rate_rad_yr if np.isfinite(slow_rate_rad_yr) else 0.0
    return float(-(planets + slow) / spec.body_coefficient)


def _proportional(u: Sequence[int], v: Sequence[int]) -> bool:
    u, v = np.asarray(u, float), np.asarray(v, float)
    return bool(np.allclose(u * np.dot(v, v), v * np.dot(u, v)))


@functools.lru_cache(maxsize=64)
def _lattice(
    body_coefficient: int, planet_coefficients: Tuple[int, ...], planet_n: Tuple[float, ...], n_c: float, order_pair: int, order_triple: int
) -> Tuple[Tuple[float, int, Tuple[int, ...], int], ...]:
    """(period, m, (l_j), order) of every line of S1 with nu != 0 (cached: the response measures several pieces)."""
    sigma = (body_coefficient,) + planet_coefficients
    ranges = [range(1, abs(body_coefficient) + 1)] + [range(-abs(k), abs(k) + 1) for k in planet_coefficients]
    lines = []
    for combination in product(*ranges):
        m, ls = combination[0], combination[1:]
        n_planets = sum(1 for value in ls if value != 0)
        if n_planets == 0:
            continue
        order = abs(m + sum(ls))
        if order > (order_pair if n_planets == 1 else order_triple) or _proportional(combination, sigma):
            continue
        nu = m * n_c + sum(value * n for value, n in zip(ls, planet_n))
        if nu != 0:
            lines.append((TWO_PI / abs(nu), m, tuple(ls), order))
    return tuple(lines)


def lattice_lines(
    spec: ResonanceSpec, planet_n: Mapping[str, float], slow_rate_rad_yr: float, order_max_pair: int, order_max_triple: int
) -> Tuple[ForcedLine, ...]:
    """S1 (module docstring) at n_c; none for a retrograde angle or a planet without a mean motion."""
    if spec.retrograde or spec.body_coefficient == 0 or any(not np.isfinite(planet_n.get(name, np.nan)) for name in spec.planets):
        return ()
    n_c = commensurate_mean_motion(spec, planet_n, slow_rate_rad_yr)
    return lattice_lines_at(spec, planet_n, n_c, order_max_pair, order_max_triple)


def lattice_lines_at(
    spec: ResonanceSpec, planet_n: Mapping[str, float], body_n_rad_yr: float, order_max_pair: int, order_max_triple: int
) -> Tuple[ForcedLine, ...]:
    """S1 with the body at a given mean motion [rad/yr] (n_c inside a libration; the measured one outside it)."""
    if spec.retrograde or spec.body_coefficient == 0 or not np.isfinite(body_n_rad_yr):
        return ()
    if any(not np.isfinite(planet_n.get(name, np.nan)) for name in spec.planets):
        return ()
    rates = tuple(float(planet_n[name]) for name in spec.planets)
    raw = _lattice(
        spec.body_coefficient, tuple(spec.planet_coefficients), rates, float(body_n_rad_yr), int(order_max_pair), int(order_max_triple)
    )
    return tuple(ForcedLine(period, m, ls, order, tuple(spec.planets)) for period, m, ls, order in raw)


def line_offset(line: ForcedLine, period_yr: float, window_cycles: float) -> float:
    """|1/T - 1/P| in units of the window's frequency resolution 1 / (W P)."""
    return float(abs(1 / line.period_yr - 1 / period_yr) * window_cycles * period_yr)


def coincident_line(lines: Sequence[ForcedLine], period_yr: float, window_cycles: float) -> Optional[Tuple[ForcedLine, float]]:
    """(line, offset) of the lowest-order line within RESOLUTION of 1/P (the nearest among equal orders); None
    without one or without a period."""
    if not lines or not (np.isfinite(period_yr) and period_yr > 0 and np.isfinite(window_cycles) and window_cycles > 0):
        return None
    hits = [(line.order, offset, line) for line in lines if (offset := line_offset(line, period_yr, window_cycles)) < RESOLUTION]
    if not hits:
        return None
    order, offset, line = min(hits, key=lambda hit: (hit[0], hit[1]))
    return line, offset


def line_groups(lines: Sequence[ForcedLine], shortest_yr: float, span_yr: float) -> List[Tuple[float, Tuple[ForcedLine, ...]]]:
    """(frequency [1/yr], members) of the lines in [shortest_yr, span_yr], members closer than 1/span to the group's
    first (lowest-frequency) line merged: the episode cannot tell them apart. The frequency is the members' mean."""
    inside = sorted((line for line in lines if shortest_yr <= line.period_yr <= span_yr), key=lambda line: 1 / line.period_yr)
    groups: List[List[ForcedLine]] = []
    for line in inside:
        if groups and 1 / line.period_yr - 1 / groups[-1][0].period_yr < RESOLUTION / span_yr:
            groups[-1].append(line)
        else:
            groups.append([line])
    return [(float(np.mean([1 / line.period_yr for line in group])), tuple(group)) for group in groups]


def harmonic_shares(t_yr: np.ndarray, y: np.ndarray, frequencies: Sequence[float]) -> Tuple[np.ndarray, float, np.ndarray]:
    """Least squares y = C + sum_g (A_g sin 2 pi f_g t + B_g cos 2 pi f_g t): (share of var(y) of each fitted component,
    R^2 of the whole fit, residual). The normal equations are summed over blocks of rows (memory O(p^2), not O(N p))."""
    t = np.asarray(t_yr, float) - float(t_yr[0])
    y = np.asarray(y, float)
    omega = 2 * np.pi * np.asarray(frequencies, float)
    p = 2 * len(omega) + 1

    def design(rows: slice) -> np.ndarray:
        phase = np.outer(t[rows], omega)
        return np.column_stack([np.ones(len(t[rows]))] + [np.sin(phase), np.cos(phase)])

    gram, moment = np.zeros((p, p)), np.zeros(p)
    for start in range(0, len(t), GRAM_BLOCK_ROWS):
        rows = slice(start, start + GRAM_BLOCK_ROWS)
        x = design(rows)
        gram += x.T @ x
        moment += x.T @ y[rows]
    coefficients = np.linalg.lstsq(gram, moment, rcond=None)[0]
    n = len(y)
    total = float(np.sum((y - np.mean(y)) ** 2))
    shares = np.zeros(len(omega))
    for g in range(len(omega)):
        idx = [1 + g, 1 + len(omega) + g]
        c = coefficients[idx]
        mean = float(gram[0, idx] @ c) / n
        shares[g] = (float(c @ gram[np.ix_(idx, idx)] @ c) - n * mean**2) / total if total > 0 else np.nan
    residual = np.concatenate(
        [y[rows] - design(rows) @ coefficients for rows in (slice(s, s + GRAM_BLOCK_ROWS) for s in range(0, n, GRAM_BLOCK_ROWS))]
    )
    r2_all = 1 - float(np.sum(residual**2)) / total if total > 0 else np.nan
    return shares, r2_all, residual


def refined_frequency(t_yr: np.ndarray, y: np.ndarray, frequency: float, span_yr: float) -> float:
    """The frequency within half a resolution element (0.5 / span) of `frequency` where y's single-sine amplitude
    |sum y e^(-2 pi i f t)| is largest: a line's period is known to the episode's resolution (the body's mean motion
    on it and the line's secular terms), the data place it inside that element."""
    grid = frequency + np.linspace(-0.5, 0.5, REFINE_POINTS) * RESOLUTION / span_yr
    t = np.asarray(t_yr, float) - float(t_yr[0])
    y = np.asarray(y, float) - float(np.mean(y))
    power = [abs(np.sum(y * np.exp(-2j * np.pi * f * t))) for f in grid]
    return float(grid[int(np.argmax(power))])


def refined_groups(
    t_yr: np.ndarray, y: np.ndarray, groups: Sequence[Tuple[float, Tuple[ForcedLine, ...]]], span_yr: float
) -> List[Tuple[float, Tuple[ForcedLine, ...]]]:
    """`groups` with each frequency refined (`refined_frequency`); two that end closer than half a resolution element
    are one regressor (their sines are nearly collinear on the episode), refined again from their mean."""
    refined: List[Tuple[float, Tuple[ForcedLine, ...]]] = []
    for frequency, members in groups:
        frequency = refined_frequency(t_yr, y, frequency, span_yr)
        if refined and frequency - refined[-1][0] < 0.5 * RESOLUTION / span_yr:
            merged = refined.pop()[1] + tuple(members)
            frequency = refined_frequency(t_yr, y, float(np.mean([1 / line.period_yr for line in merged])), span_yr)
            members = merged
        refined.append((frequency, tuple(members)))
    return refined


def strongest_peak(t_yr: np.ndarray, y: np.ndarray, shortest_yr: float, span_yr: float) -> float:
    """Frequency [1/yr] of the largest local maximum of y's amplitude spectrum (zero-padded) in [1/span, 1/shortest];
    nan without one (y on a uniform grid)."""
    n_fft = int(2 ** np.ceil(np.log2(SPECTRUM_OVERSAMPLE * len(y))))
    amplitude = np.abs(np.fft.rfft(y - np.mean(y), n_fft))
    frequency = np.fft.rfftfreq(n_fft, float(np.median(np.diff(t_yr))))
    band = (frequency >= 1 / span_yr) & (frequency <= 1 / shortest_yr)
    peak = np.zeros_like(band)
    peak[1:-1] = band[1:-1] & (amplitude[1:-1] >= amplitude[:-2]) & (amplitude[1:-1] >= amplitude[2:])
    if not np.any(peak):
        return float('nan')
    return float(frequency[np.flatnonzero(peak)[np.argmax(amplitude[peak])]])


def period_name(line: ForcedLine, spec: ResonanceSpec) -> str:
    """A two-body line's period as a multiple of the planet's: P_conj / |j| = (|k| / |j|) P_P ('57 P_M = P_conj',
    '13/2' is written 'P_conj/2'); '' for a three-body line (no common conjunction cycle)."""
    if len(spec.planets) != 1 or len(line.planet_coefficients) != 1:
        return ''
    k, k_planet = spec.body_coefficient, spec.planet_coefficients[0]
    j = abs(line.planet_coefficients[0] * k - line.body_coefficient * k_planet)
    if j == 0:
        return ''
    common = math.gcd(abs(k), abs(k_planet))
    cycle, divisor = abs(k) // common, j // common
    letter = planet_letter(spec.planets[0])
    ratio = Fraction(cycle, divisor)
    if ratio.denominator == 1:
        head = f'P_{letter}' if ratio == 1 else f'{ratio.numerator} P_{letter}'
        return f'{head} = P_conj' if divisor == 1 else head
    return f'P_conj/{divisor}'


def own_period(frequency: float, own_period_yr: float, span_yr: float) -> List[str]:
    """['P_lib'] when the frequency lies within 1/span of the libration's own 1/P (the episode cannot tell that line
    from the libration: `coincident_line` at the force window's resolution), else []. Its harmonics h/P are not named:
    on a short episode they are dense (spacing 1/P against the resolution 1/T), so a line near one is a coincidence
    (GQ107: 107 yr against P/11 at 6 cycles)."""
    if not (np.isfinite(own_period_yr) and own_period_yr > 0) or abs(frequency - 1 / own_period_yr) >= RESOLUTION / span_yr:
        return []
    return ['P_lib']


def group_ids(frequency: float, members: Sequence[ForcedLine], spec: ResonanceSpec, own_period_yr: float, span_yr: float) -> str:
    """The names of one group: the period(s) as planet-period multiples, every member's label (lowest order first) and
    'P_lib' when the libration's own period coincides, '; '-joined."""
    ordered = sorted(members, key=lambda line: (line.order, line.label))
    periods = list(dict.fromkeys(name for name in (period_name(line, spec) for line in ordered) if name))
    return '; '.join(periods + [line.label for line in ordered] + own_period(frequency, own_period_yr, span_yr))


def line_share(
    t_yr: np.ndarray,
    adot: np.ndarray,
    lines: Sequence[ForcedLine],
    spec: ResonanceSpec,
    own_period_yr: float,
    shortest_yr: float,
    cutoff_yr: float = float('nan'),
    orbital_period_yr: float = float('nan'),
) -> LineShare:
    """Which lines of S1 dominate a_dot on one episode (module docstring): `t_yr`, `adot` the episode's samples (a
    uniform grid, low-passed), the band [shortest_yr, T]. The body's own orbital period at n_c (the term m = 1 without
    a planet, which S1 leaves out: the short-period wave of the osculating a, 379 yr in 523683 2U-9) is a known line
    too, 'P_orb', fitted with the lattice but outside `share_all`. The series is decimated to SAMPLES_PER_SHORTEST
    samples per shortest period first (it is low-passed above that)."""
    t_yr, adot = np.asarray(t_yr, float), np.asarray(adot, float)
    finite = np.isfinite(t_yr) & np.isfinite(adot)
    t_yr, adot = t_yr[finite], adot[finite]
    nan = float('nan')
    start, end = (float(t_yr[0]), float(t_yr[-1])) if len(t_yr) else (nan, nan)

    def blank(reason: str) -> LineShare:
        return LineShare(nan, nan, nan, '', start, end, cutoff_yr, reason)

    if len(t_yr) < MIN_SHARE_SAMPLES or end <= start:
        return blank('no_samples')
    span = end - start
    if not (np.isfinite(shortest_yr) and 0 < shortest_yr < span):
        return blank('no_band')
    stride = max(1, int(shortest_yr / (SAMPLES_PER_SHORTEST * float(np.median(np.diff(t_yr))))))
    t, y = t_yr[::stride], adot[::stride] - float(np.mean(adot[::stride]))
    if len(t) < MIN_SHARE_SAMPLES:
        return blank('no_samples')
    groups = refined_groups(t, y, line_groups(lines, shortest_yr, span), span)
    frequencies = [f for f, _ in groups]
    names = [group_ids(f, members, spec, own_period_yr, span) for f, members in groups]
    if 2 * len(frequencies) + 5 >= len(t):
        return blank('no_samples')
    lattice_r2 = harmonic_shares(t, y, frequencies)[1] if frequencies else 0.0
    if np.isfinite(orbital_period_yr) and shortest_yr <= orbital_period_yr <= span:
        f_orb = 1 / orbital_period_yr
        near = [i for i, f in enumerate(frequencies) if abs(f - f_orb) < RESOLUTION / span]
        if near:
            names[near[0]] = f'{names[near[0]]}; P_orb'
        else:
            frequencies, names = frequencies + [refined_frequency(t, y, f_orb, span)], names + ['P_orb']
    residual = harmonic_shares(t, y, frequencies)[2] if frequencies else y
    extra = strongest_peak(t, residual, shortest_yr, span)
    if np.isfinite(extra) and all(abs(extra - f) >= RESOLUTION / span for f in frequencies):
        # off the lattice: the libration's own period when it lies there, else unidentified
        frequencies, names = frequencies + [extra], names + ['; '.join(own_period(extra, own_period_yr, span)) or UNIDENTIFIED]
    if not frequencies:
        return blank('no_lines')
    shares, _, _ = harmonic_shares(t, y, frequencies)
    best = int(np.nanargmax(shares)) if np.any(np.isfinite(shares)) else 0
    return LineShare(1 / frequencies[best], float(shares[best]), float(lattice_r2), names[best], start, end, cutoff_yr)
