"""The free-omega gate: does a librating Lidov-Kozai angle survive the forced/free split?

`secular/free_elements.py` does the signal processing; this module turns its output into a
resonance status. It runs only where the classifier already said LIBRATION, and it is the
only thing that decides a Lidov-Kozai status after classification.

Why it is needed. Libration of the osculating omega does not prove a Lidov-Kozai
resonance. When the forced eccentricity vector is longer than the free one, the (k, h)
cloud does not enclose the origin, varpi cannot turn, and omega = varpi - Omega is pinned
by geometry alone. 591986 is the worked example: its osculating omega is confined for
30 Myr while the free one leaves confinement after 6 kyr.

The decision tree, entered only at LIBRATION:

    baseline shorter than min_cycles * P_lib     -> UNCERTAIN   (never measured)
    more than half the samples masked            -> UNCERTAIN   (no usable argument)
    free omega spans less than a full turn:
        b_varpi or b_Omega at a planetary rate   -> UNCERTAIN   (residual is forcing)
        otherwise                                -> LIBRATION   (confirmed)
    free omega winds:
        any rate at a planetary rate             -> UNCERTAIN   (residual is forcing)
        otherwise                                -> LIBRATION_UNCERTAIN  (kinematic)

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
missing feature. For screening, UNCERTAIN means "re-check on a longer baseline", so it
belongs in the candidate list next to LIBRATION, not in the discard pile.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from resonances.resonance.classify.models import ResonanceStatus
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
    UNAVAILABLE = 'unavailable'  # the decomposition could not be built


@dataclass
class OmegaFreeGate:
    """Verdict of the free-omega check for one body-resonance."""

    outcome: GateOutcome
    matches: List[str] = field(default_factory=list)  # e.g. ['b_varpi~g8 (d=0.4)']
    n_cycles: Optional[float] = None  # baseline / P_lib

    @property
    def comment(self) -> str:
        """The trail left in summary.csv, prefixed so gate decisions can be counted."""
        if self.outcome is GateOutcome.CONFIRMED:
            return 'gate: 2 confirmed'
        if self.outcome is GateOutcome.KINEMATIC:
            return 'gate: 2->-2 kinematic libration'
        if self.outcome is GateOutcome.SHORT_WINDOW:
            cycles = 'unknown' if self.n_cycles is None else f'{self.n_cycles:.1f}'
            return f'gate: 2->-9 window holds {cycles} libration periods'
        if self.outcome is GateOutcome.UNDEFINED_PHASE:
            return 'gate: 2->-9 phase undefined (|z_free| near zero)'
        if self.outcome is GateOutcome.PLANETARY_RESIDUAL:
            return f"gate: 2->-9 residual at planetary freq: {', '.join(self.matches)}"
        return ''


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


def apply_gate(
    status: ResonanceStatus,
    elements: Optional[FreeElements],
    libration_period: Optional[float],
    config,
) -> Tuple[ResonanceStatus, OmegaFreeGate]:
    """Confirm, demote or withhold a Lidov-Kozai libration using the free omega.

    Only LIBRATION is ever changed. The two outcomes below it are distinct and must stay
    that way in any statistics: LIBRATION_UNCERTAIN means the libration was measured and
    found kinematic, UNCERTAIN means no verdict could be reached.
    """
    if not getattr(config, 'free_gate_enabled', True):
        return status, OmegaFreeGate(GateOutcome.DISABLED)
    if status != ResonanceStatus.LIBRATION:
        return status, OmegaFreeGate(GateOutcome.NOT_APPLICABLE)
    if elements is None:
        return status, OmegaFreeGate(GateOutcome.UNAVAILABLE)

    # The fundamentals the split actually subtracted, not a fresh lookup: judging a
    # residual against numbers other than the ones removed would be meaningless.
    frequencies = elements.frequencies
    tolerance = elements.resolution
    min_cycles = getattr(config, 'free_gate_min_cycles', DEFAULT_MIN_CYCLES)
    cycles = elements.baseline / libration_period if libration_period else None

    if cycles is not None and cycles < min_cycles:
        return ResonanceStatus.UNCERTAIN, OmegaFreeGate(GateOutcome.SHORT_WINDOW, n_cycles=cycles)
    if max(elements.mask_fraction_e, elements.mask_fraction_i) > MAX_MASKED_FRACTION:
        return ResonanceStatus.UNCERTAIN, OmegaFreeGate(GateOutcome.UNDEFINED_PHASE, n_cycles=cycles)

    if elements.librates:
        g_targets, s_targets = _confirmation_targets(elements, frequencies)
        matches = _matches('b_varpi', elements.b_varpi, g_targets, tolerance) + _matches('b_Omega', elements.b_Omega, s_targets, tolerance)
        if matches:
            return ResonanceStatus.UNCERTAIN, OmegaFreeGate(GateOutcome.PLANETARY_RESIDUAL, matches, cycles)
        return status, OmegaFreeGate(GateOutcome.CONFIRMED, n_cycles=cycles)

    g_targets, s_targets, combos = _demotion_targets(frequencies)
    matches = (
        _matches('b_varpi', elements.b_varpi, g_targets, tolerance)
        + _matches('b_Omega', elements.b_Omega, s_targets, tolerance)
        + _matches('b_omega', elements.b_omega, combos, tolerance)
    )
    if matches:
        return ResonanceStatus.UNCERTAIN, OmegaFreeGate(GateOutcome.PLANETARY_RESIDUAL, matches, cycles)
    return ResonanceStatus.LIBRATION_UNCERTAIN, OmegaFreeGate(GateOutcome.KINEMATIC, n_cycles=cycles)


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
