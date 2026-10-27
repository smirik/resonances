"""The free-omega gate: does a bounded Lidov-Kozai angle survive the forced/free split?

`secular/free_elements.py` does the signal processing; this module reads its output and returns an outcome
(`evaluate_gate`). The ZLK analysis (`analysis.zlk`, `analysis.secular.decide.decide_status`) turns the outcome into
the verdict: on a libration, confirmed keeps it resonant, kinematic makes it non_resonant (kinematic_libration),
any other outcome uncertain (gate_unavailable) — the last two never merged; on an episodic angle, a drift or too
few cycles only a kinematic outcome with the free circle never around the origin changes the status.

Why it is needed. Libration of the osculating omega does not prove a Lidov-Kozai
resonance. When the forced eccentricity vector is longer than the free one, the (k, h)
cloud does not enclose the origin, varpi cannot turn, and omega = varpi - Omega is pinned
by geometry alone. 591986 is the worked example: its osculating omega is confined for
30 Myr while the free one leaves confinement after 6 kyr.

The outcomes:

    no free elements (too short, or failed)      -> unavailable       (never measured)
    baseline shorter than min_cycles * P_lib     -> short_window      (never measured)
    more than half the samples masked            -> undefined_phase   (no usable argument)
    free omega spans less than a full turn:
        b_varpi or b_Omega at a planetary rate   -> planetary_residual (residual is forcing)
        otherwise                                -> confirmed
    free omega winds:
        any rate at a planetary rate             -> planetary_residual
        otherwise                                -> kinematic

The same frequency check stands on both branches, and that symmetry is the point. A
Lidov-Kozai libration lives on the object's *own* rates, so it can only be confirmed when
varpi_free and Omega_free turn at speeds that are not planetary. A residual sitting at a
planetary rate is either a defect of the subtraction or a genuine linear secular resonance
(g close to some g_j, the nu5/nu6 family) where the forced/free split is undefined in the
first place; both make the verdict unsafe, and telling them apart is not required.

There is no amplitude criterion anywhere in the gate. An earlier design compared the free
eccentricity against an estimated floor of unsubtracted forcing, but that floor cannot be
measured: estimating how much forcing sits at a frequency the baseline cannot resolve
requires resolving it. Frequency is the direct observable — unsubtracted forcing lives, by
construction, only at known rates.

The honest cost: an object whose own g or s happens to sit within one Rayleigh width of a
planetary fundamental is withheld rather than confirmed. That is a real ambiguity, not a
missing feature. For screening, uncertain means "re-check on a longer baseline", so it
belongs in the candidate list next to resonant, not in the discard pile.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from resonances.secular.free_elements import FreeElements

# Modes the residual rates are compared against. 'const' is deliberately absent from the
# single-rate lists: a residual that barely turns always sits within tolerance of zero,
# which is what a librator looks like, not what a leak looks like. It joins the list only
# on the confirmation branch, and only when a constant term was actually left unsubtracted.
E_MODES = ('g5', 'g6', 'g7', 'g8')
I_MODES = ('s6', 's7', 's8')

DEFAULT_MIN_CYCLES = 3.0
MAX_MASKED_FRACTION = 0.5


class GateOutcome(StrEnum):
    """Why the gate reached the status it did."""

    CONFIRMED = 'confirmed'  # free omega confined and turning at non-planetary rates
    KINEMATIC = 'kinematic'  # free omega winds: the osculating libration was geometry
    SHORT_WINDOW = 'short_window'  # fewer than min_cycles libration periods in the baseline
    UNDEFINED_PHASE = 'undefined_phase'  # too many samples near the origin to unwrap
    PLANETARY_RESIDUAL = 'planetary_residual'  # a residual rate matches a planetary one
    NOT_APPLICABLE = 'not_applicable'  # the angle does not librate, nothing to check
    DISABLED = 'disabled'
    UNAVAILABLE = 'unavailable'  # the decomposition could not be built: nothing was measured


@dataclass
class OmegaFreeGate:
    """Verdict of the free-omega check for one body-resonance."""

    outcome: GateOutcome
    matches: List[str] = field(default_factory=list)  # e.g. ['b_varpi~g8 (d=0.4)']
    n_cycles: Optional[float] = None  # baseline / P_lib


def _matches(label: str, value: float, targets: Dict[str, float], tolerance: float) -> List[str]:
    """Which targets the signed rate `value` cannot be told apart from.

    Signed, not absolute: every s_j is retrograde, so a prograde free node is not s8 no
    matter how close the magnitudes are.
    """
    if not np.isfinite(value):
        return []
    return [f'{label}~{name} (d={abs(value - target):.1f})' for name, target in targets.items() if abs(value - target) < tolerance]


def _mode_targets(names: Tuple[str, ...], frequencies: Dict[str, float]) -> Dict[str, float]:
    return {name: frequencies[name] for name in names if name in frequencies}


def _confirmation_targets(elements: FreeElements, frequencies: Dict[str, float]) -> Tuple[dict, dict]:
    """Planetary rates a *confined* residual must stay away from.

    Zero joins the eccentricity list when the constant term was clustered away and so left
    in the residual, because an unsubtracted constant is exactly what makes an angle look
    confined — the dangerous direction. The inclination list never needs it: 'const' heads
    PRIORITY_I, so it is always its cluster's representative and always fitted.
    """
    g_targets = _mode_targets(E_MODES, frequencies)
    if 'const' in elements.dropped_e:
        g_targets = dict(g_targets, const=0.0)
    return g_targets, _mode_targets(I_MODES, frequencies)


def _demotion_targets(frequencies: Dict[str, float]) -> Tuple[dict, dict, dict]:
    """Planetary rates a *winding* residual must stay away from, including differences.

    omega = varpi - Omega, so an unsubtracted term in either vector shows up in omega at a
    difference frequency; all g_j - s_k are checked, with s5 = 0 included as a real mode.
    """
    g_targets = _mode_targets(E_MODES, frequencies)
    s_targets = _mode_targets(I_MODES, frequencies)
    combos = {
        f'{g}-{s}': g_value - s_value
        for g, g_value in g_targets.items()
        for s, s_value in dict(s_targets, s5=frequencies.get('s5', 0.0)).items()
    }
    return g_targets, s_targets, combos


def evaluate_gate(elements: Optional[FreeElements], libration_period: Optional[float], config) -> OmegaFreeGate:
    """The free-omega verdict on a bounded Lidov-Kozai angle (`libration_period` its period in years).

    The caller asks only for an angle that librates (or has libration episodes); the outcome is then confirmed,
    kinematic, or one of the reasons why no verdict is possible (short_window, undefined_phase, planetary_residual,
    unavailable). The two outcomes below confirmed are distinct and must stay that way in any statistics: kinematic
    is a measurement (the libration was found to be geometry), the others are the absence of one.
    """
    if not getattr(config, 'free_gate_enabled', True):
        return OmegaFreeGate(GateOutcome.DISABLED)
    if elements is None:
        return OmegaFreeGate(GateOutcome.UNAVAILABLE)

    # The fundamentals the split actually subtracted, not a fresh lookup: judging a
    # residual against numbers other than the ones removed would be meaningless.
    frequencies = elements.frequencies
    tolerance = elements.resolution
    min_cycles = getattr(config, 'free_gate_min_cycles', DEFAULT_MIN_CYCLES)
    cycles = elements.baseline / libration_period if libration_period and np.isfinite(libration_period) else None

    if cycles is not None and cycles < min_cycles:
        return OmegaFreeGate(GateOutcome.SHORT_WINDOW, n_cycles=cycles)
    if max(elements.mask_fraction_e, elements.mask_fraction_i) > MAX_MASKED_FRACTION:
        return OmegaFreeGate(GateOutcome.UNDEFINED_PHASE, n_cycles=cycles)

    if elements.librates:
        g_targets, s_targets = _confirmation_targets(elements, frequencies)
        matches = _matches('b_varpi', elements.b_varpi, g_targets, tolerance) + _matches('b_Omega', elements.b_Omega, s_targets, tolerance)
        if matches:
            return OmegaFreeGate(GateOutcome.PLANETARY_RESIDUAL, matches, cycles)
        return OmegaFreeGate(GateOutcome.CONFIRMED, n_cycles=cycles)

    g_targets, s_targets, combos = _demotion_targets(frequencies)
    matches = (
        _matches('b_varpi', elements.b_varpi, g_targets, tolerance)
        + _matches('b_Omega', elements.b_Omega, s_targets, tolerance)
        + _matches('b_omega', elements.b_omega, combos, tolerance)
    )
    if matches:
        return OmegaFreeGate(GateOutcome.PLANETARY_RESIDUAL, matches, cycles)
    return OmegaFreeGate(GateOutcome.KINEMATIC, n_cycles=cycles)


SUMMARY_COLUMNS = ('free_gate', 'free_n_cycles', 'free_gate_matches')


def summary_fields(gate: Optional[OmegaFreeGate]) -> dict:
    """The gate columns of one summary.csv row.

    `free_gate` duplicates information that also lands in `comments`, on purpose: counting
    how many librations the gate demoted should not require a regular expression over free
    text.
    """
    if gate is None:
        return {column: None for column in SUMMARY_COLUMNS}
    return {
        'free_gate': str(gate.outcome),
        'free_n_cycles': gate.n_cycles,
        'free_gate_matches': '; '.join(gate.matches) or None,
    }
