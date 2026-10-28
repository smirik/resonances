"""How long to extend a secular or Lidov-Kozai record: the action of an uncertain verdict and its `extend_to_years`.

Changes no status: it reads the decision and fills the action (extend or review) and the total length to integrate.
T is the record's length, N what the cause needs (an estimate, not a guarantee of a verdict); the length asked for is
the MMR rule's next check (`mmr.decide.next_check`): round_up(max(f_min T, min(N, f_next T))), f_min =
MMR_EXTEND_MIN_SPAN_FACTOR, f_next = MMR_NEXT_CHECK_SPAN_FACTOR; the record is analysed again at that length.

    baseline_needed (drift, insufficient_time)  N = (LIBRATION_MIN_TURNING_POINTS / 2) P + 2 FILTER_EDGE_TRIM_CUTOFFS P_c,
                                                P = max(period floor, the measured cycle P_cyc), P_c the decisive cutoff
                                                (the MMR baseline rule); ZLK with the gate's short_window: also the gate's N
    gate short_window                           N = FREE_GATE_MIN_CYCLES P_cyc on the free elements' baseline, plus
                                                what the free elements lose of the record (T - their baseline)
    gate planetary_residual                     the residual is d_min arcsec/yr from a planetary rate, inside one
                                                Rayleigh width 1296000/T_free: N = ZLK_EXTEND_RESOLUTION_MARGIN x
                                                1296000/d_min + (T - T_free); beyond ZLK_EXTEND_MAX_YR (or d_min = 0):
                                                review, no run of any practical length resolves it
    gate undefined_phase, unavailable           review: a longer record does not define the free phase, and missing
                                                free elements are not a matter of length
    unusable_series                             review (non-finite angle, too few samples, a grid the filter refuses)

The geometric N of an angle that needs more cycles is always included (max), so a gate cause never asks for less.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from resonances.resonance.analysis.mmr.decide import min_extension_yr, next_check, round_up
from resonances.resonance.analysis.secular.models import ARCSEC_PER_TURN, GateResult
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Action, Reason, Status

NAN = float('nan')
KYR = 1e3

# what the extension is for (column base_extend_cause); the review causes name why there is none
CAUSE_BASELINE = 'baseline'  # the angle needs more cycles
CAUSE_GATE_CYCLES = 'gate_cycles'  # the gate's short_window
CAUSE_GATE_RESOLUTION = 'gate_resolution'  # planetary_residual: a free rate within a Rayleigh width of a planetary one
CAUSE_GATE_UNRESOLVABLE = 'gate_unresolvable'  # ... closer than a record of ZLK_EXTEND_MAX_YR resolves: review
CAUSE_GATE_PHASE = 'gate_undefined_phase'  # review
CAUSE_GATE_UNAVAILABLE = 'gate_unavailable'  # review
CAUSE_UNUSABLE = 'unusable_series'  # review

PERIOD_CYCLE = 'cycle'  # base_period_source: the measured P_cyc
PERIOD_FLOOR = 'period_floor'  # the type's shortest libration (or 2 P_c)


@dataclass(frozen=True)
class Extension:
    """The action of an uncertain secular/ZLK verdict and what it rests on (columns `base_*`)."""

    action: Optional[Action] = None  # None: the default action stands (the verdict is not uncertain)
    extend_to_years: Optional[float] = None  # with action extend: the total length of the next check [yr]
    required_yr: float = NAN  # N
    period_yr: float = NAN  # P of the geometric N
    period_source: str = ''  # cycle | period_floor
    cause: str = ''  # CAUSE_*
    # round_up(max(N, f_min T)): the estimate behind the next check (as MMR's base_extend_to_years); None without N
    estimate_yr: Optional[float] = None

    def to_columns(self) -> Dict[str, Any]:
        return {
            'base_required_kyr': self.required_yr / KYR if np.isfinite(self.required_yr) else None,
            'base_period_yr': self.period_yr if np.isfinite(self.period_yr) else None,
            'base_period_source': self.period_source,
            'base_extend_to_years': self.estimate_yr,
            'base_extend_cause': self.cause,
        }


def geometric_need(period_floor_yr: float, period_cycle_yr: float, cutoff_yr: float, settings: AnalysisSettings):
    """(N, P, source): N = (LIBRATION_MIN_TURNING_POINTS / 2) P + 2 FILTER_EDGE_TRIM_CUTOFFS P_c, P = max(floor, P_cyc)."""
    cycle = period_cycle_yr if np.isfinite(period_cycle_yr) and period_cycle_yr > 0 else NAN
    floor = period_floor_yr if np.isfinite(period_floor_yr) and period_floor_yr > 0 else NAN
    if np.isfinite(cycle) and not (np.isfinite(floor) and floor >= cycle):
        period, source = cycle, PERIOD_CYCLE
    else:
        period, source = floor, PERIOD_FLOOR if np.isfinite(floor) else ''
    if not (np.isfinite(period) and np.isfinite(cutoff_yr)):
        return NAN, period, source
    need = settings.libration_min_turning_points / 2 * period + 2 * settings.filter_edge_trim_cutoffs * cutoff_yr
    return float(need), float(period), source


def _lost(gate: GateResult, span_yr: float) -> float:
    """What the free elements lose of the record (T minus their baseline) [yr]."""
    baseline = gate.baseline_yr
    return max(0.0, span_yr - baseline) if np.isfinite(baseline) else 0.0


def gate_cycles_need(gate: GateResult, span_yr: float) -> float:
    """short_window: FREE_GATE_MIN_CYCLES x P_cyc on the free baseline, plus what the free elements lose [yr]."""
    period = gate.libration_period_yr
    if not (np.isfinite(period) and period > 0 and np.isfinite(gate.min_cycles)):
        return NAN
    return float(gate.min_cycles * period + _lost(gate, span_yr))


def gate_resolution_need(gate: GateResult, span_yr: float, settings: AnalysisSettings) -> float:
    """planetary_residual: ZLK_EXTEND_RESOLUTION_MARGIN x 1296000/d_min + what the free elements lose [yr]; inf when
    d_min is 0 or that baseline exceeds ZLK_EXTEND_MAX_YR; nan when d_min was not measured."""
    distance = gate.min_distance_arcsec_yr
    if not np.isfinite(distance):
        return NAN
    if distance <= 0:
        return float('inf')
    baseline = settings.zlk_extend_resolution_margin * ARCSEC_PER_TURN / distance
    return float('inf') if baseline > settings.zlk_extend_max_yr else float(baseline + _lost(gate, span_yr))


def recommend(
    status: Status,
    reason: Reason | str,
    gate: Optional[GateResult],
    period_floor_yr: float,
    period_cycle_yr: float,
    cutoff_yr: float,
    span_yr: float,
    settings: AnalysisSettings,
) -> Extension:
    """The extension of a verdict (module docstring); an empty Extension (default action) unless it is uncertain."""
    if status != Status.UNCERTAIN:
        return Extension()
    span = abs(float(span_yr))
    if reason == Reason.UNUSABLE_SERIES:
        return Extension(Action.REVIEW, cause=CAUSE_UNUSABLE)
    geometric, period, source = geometric_need(period_floor_yr, period_cycle_yr, cutoff_yr, settings)
    outcome = gate.outcome if gate is not None else ''
    if reason == Reason.BASELINE_NEEDED:
        # ZLK: the gate's own need joins when its window is short too (the max below)
        need, cause = (gate_cycles_need(gate, span) if outcome == 'short_window' else NAN), CAUSE_BASELINE
    elif reason == Reason.GATE_UNAVAILABLE and outcome == 'short_window':
        need, cause = gate_cycles_need(gate, span), CAUSE_GATE_CYCLES
    elif reason == Reason.GATE_UNAVAILABLE and outcome == 'planetary_residual':
        need, cause = gate_resolution_need(gate, span, settings), CAUSE_GATE_RESOLUTION
        if np.isinf(need):
            return Extension(Action.REVIEW, required_yr=need, period_yr=period, period_source=source, cause=CAUSE_GATE_UNRESOLVABLE)
    elif reason == Reason.GATE_UNAVAILABLE:
        cause = CAUSE_GATE_PHASE if outcome == 'undefined_phase' else CAUSE_GATE_UNAVAILABLE
        return Extension(Action.REVIEW, period_yr=period, period_source=source, cause=cause)
    else:  # an uncertain reason this module does not know: the default action, without a length
        return Extension()
    need = float(np.nanmax([need, geometric])) if np.isfinite(need) or np.isfinite(geometric) else NAN
    extend = next_check(float(need), span, settings)
    estimate = round_up(max(need, min_extension_yr(span, settings))) if np.isfinite(need) else None
    return Extension(Action.EXTEND, extend, float(need), period, source, cause, estimate)
