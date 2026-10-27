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
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis.mmr.models import ResonanceSpec, planet_terms

TWO_PI = 2 * math.pi
RESOLUTION = 1.0  # |1/T - 1/P| in units of the window's frequency resolution 1 / (W P): its definition, not a threshold


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
    rates = tuple(float(planet_n[name]) for name in spec.planets)
    raw = _lattice(spec.body_coefficient, tuple(spec.planet_coefficients), rates, n_c, int(order_max_pair), int(order_max_triple))
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
