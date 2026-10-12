"""Every admissible secular combination close to a body's proper frequencies.

`SECULAR_FORMULAS` is a catalogue: it holds the resonances that the literature found to matter
somewhere in the main belt. A body can still sit in a combination that is not listed, so the
complete check works from the other side: take the body's proper frequencies g and s, build
every angle allowed by the d'Alembert rules up to a given order,

    sigma = p*varpi + q*Omega + sum_j (p_j*varpi_j + q_j*Omega_j)
    p + q + sum p_j + sum q_j = 0             (invariance under a rotation of the axes)
    q + sum q_j even                          (symmetry with respect to the reference plane)

and keep the ones whose divisor p*g + q*s + sum(p_j*g_j + q_j*s_j) is small. The order is
the sum of |coefficients|, as in `SecularResonanceFormula.order`; the lowest power of e and
sin(i) in the term is at least that large, so the strength falls quickly with it.

s5 is left out: it is identically zero and its "longitude" is the fixed node of the
invariable plane, so a term with s5 only shifts another combination by a constant.
An angle and its negative are the same resonance; the sign is fixed so that the first
non-zero of (p, q) is positive.

Pure arithmetic: no Body, no integration.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from resonances.data.const import PLANETARY_FREQUENCIES

# Every planetary fundamental that can appear in an angle, in the order used for formulas.
PLANETARY_MODES = ('g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 's1', 's2', 's3', 's4', 's6', 's7', 's8')

# Default divisor windows, arcsec/yr, by order. Wider than any resonance width of that order in
# the main belt, so that a missed combination cannot be blamed on the window.
DEFAULT_THRESHOLDS = {2: 5.0, 4: 2.0, 6: 1.0}


@dataclass(frozen=True)
class SecularCombination:
    formula: str
    order: int
    divisor: float  # arcsec/yr
    key: Tuple[Tuple[str, int], ...]  # sorted (label, coefficient) with 'g'/'s' for the body


def coefficient_key(resonance_or_formula) -> Tuple[Tuple[str, int], ...]:
    """Canonical coefficient vector of an angle, the same for every spelling and for -sigma.

    Labels are 'g', 's' for the body and 'g5', 's6', ... for the planets. 'g-g6+s-s6' and
    'g+s-s6-g6' have one key; so do '-g+s+g5-s7' and 'g-s-g5+s7'.
    """
    from resonances.secular.secular_resonance_formula import SecularResonanceFormula

    if isinstance(resonance_or_formula, str):
        terms = SecularResonanceFormula(resonance_or_formula).terms
    else:
        terms = resonance_or_formula.formula.terms
    coeffs: Dict[str, int] = {}
    for t in terms:
        label = t.mode if t.index is None else f"{t.mode}{t.index}"
        coeffs[label] = coeffs.get(label, 0) + t.coefficient
    coeffs = {label: c for label, c in coeffs.items() if c != 0}
    lead = coeffs.get('g') or coeffs.get('s') or next((coeffs[k] for k in sorted(coeffs)), 1)
    sign = 1 if lead > 0 else -1
    return tuple(sorted((label, sign * c) for label, c in coeffs.items()))


def formula_from_coefficients(p: int, q: int, planetary: Dict[str, int]) -> str:
    """'g-g6+s-s6'-style formula: the body's g, s first, then the planets in PLANETARY_MODES order."""
    parts = [(p, 'g'), (q, 's')] + [(planetary.get(label, 0), label) for label in PLANETARY_MODES]
    out = ''
    for c, label in parts:
        if c == 0:
            continue
        sign = '-' if c < 0 else '+'
        mag = '' if abs(c) == 1 else str(abs(c))
        out += f"{sign}{mag}{label}"
    return out[1:] if out.startswith('+') else out


@lru_cache(maxsize=None)
def _planetary_parts(max_planetary_order: int) -> np.ndarray:
    """All integer vectors over PLANETARY_MODES with 1 <= L1 norm <= max_planetary_order, plus zero (cached, read-only)."""
    n = len(PLANETARY_MODES)
    signed = [(j, s) for j in range(n) for s in (1, -1)]
    rows = {tuple([0] * n)}
    for size in range(1, max_planetary_order + 1):
        for combo in combinations_with_replacement(range(len(signed)), size):
            v = [0] * n
            for idx in combo:
                j, s = signed[idx]
                v[j] += s
            if sum(abs(x) for x in v) == size:  # +x and -x of one mode would cancel: that is a lower order
                rows.add(tuple(v))
    parts = np.array(sorted(rows), dtype=np.int8)  # coefficients are small; ~250k rows at order 6
    parts.flags.writeable = False
    return parts


def near_resonant(
    g: float,
    s: float,
    max_order: int = 6,
    thresholds: Optional[Dict[int, float]] = None,
    frequencies: Optional[Dict[str, float]] = None,
    modes: Sequence[str] = PLANETARY_MODES,
) -> List[SecularCombination]:
    """Admissible combinations of order <= max_order with |divisor| below the window for their order.

    g, s are the body's proper frequencies in arcsec/yr. `thresholds` maps an order to its window
    (orders are even; a missing order uses the window of the closest lower one). `modes` restricts
    the planetary fundamentals that may appear. Sorted by order, then |divisor|.
    """
    thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    source = PLANETARY_FREQUENCIES if frequencies is None else frequencies
    allowed = np.array([label in modes for label in PLANETARY_MODES])
    freqs = np.array([source.get(label, np.nan) for label in PLANETARY_MODES])
    if np.any(np.isnan(freqs[allowed])):
        missing = [label for label, ok, f in zip(PLANETARY_MODES, allowed, freqs) if ok and np.isnan(f)]
        raise ValueError(f"Missing planetary frequencies {missing}")

    parts = _planetary_parts(max_order - 1)
    parts = parts[~np.any(parts[:, ~allowed] != 0, axis=1)]
    s_index = np.array([label.startswith('s') for label in PLANETARY_MODES])
    part_l1 = np.abs(parts).sum(axis=1)
    part_sum = parts.sum(axis=1)
    part_s_sum = parts[:, s_index].sum(axis=1)
    part_freq = parts[:, allowed] @ freqs[allowed]

    # Window by order: the one given for that order, or for the closest lower order listed.
    by_order = np.zeros(max_order + 1)
    for order in range(max_order + 1):
        known = [o for o in thresholds if o <= order]
        by_order[order] = thresholds[max(known)] if known else 0.0

    found: List[SecularCombination] = []
    for p in range(0, max_order + 1):
        for q in range(-max_order, max_order + 1):
            if (p == 0 and q <= 0) or abs(p) + abs(q) > max_order:
                continue  # the sign convention: the first non-zero of (p, q) is positive
            order = abs(p) + abs(q) + part_l1
            ok = (order <= max_order) & (part_sum == -(p + q)) & ((q + part_s_sum) % 2 == 0)
            divisor = p * g + q * s + part_freq
            hits = np.nonzero(ok & (np.abs(divisor) < by_order[np.minimum(order, max_order)]))[0]
            for i in hits:
                planetary = {label: int(c) for label, c in zip(PLANETARY_MODES, parts[i]) if c != 0}
                formula = formula_from_coefficients(p, q, planetary)
                # p > 0, or p == 0 and q > 0: already the canonical sign of `coefficient_key`
                key = tuple(sorted([(label, c) for label, c in (('g', p), ('s', q)) if c] + list(planetary.items())))
                found.append(SecularCombination(formula, int(order[i]), float(divisor[i]), key))
    found.sort(key=lambda c: (c.order, abs(c.divisor)))
    return found
