"""`analyse_zlk`: the Lidov-Kozai angle omega = varpi - Omega.

The geometry and the angle decision are those of a secular angle (`analysis.secular`); the physics is the
free-omega gate (`resonance/omega_free_gate.py`): a bounded osculating omega is a ZLK libration only if the free
omega, after the forced terms at the planetary fundamentals are removed, stays bounded at non-planetary rates.

The gate judges every omega that stays within a turn (`decide.GATED`). The TTO classifier called "libration" any
omega
that stayed within a turn over the record, and the gate judged exactly those; the shared geometry calls them
libration, episodic, drift or too few cycles (555527: 100 kyr, e 0.01-0.05, omega jumps by ~150 deg where e
passes near zero, its centre drifts), and the gate keeps judging them. A libration takes the gate's verdict as it
is;
an episodic omega, a drift or too few cycles is made non_resonant only by a kinematic verdict with
the
free circle never around the origin (rho_max = max e_free/e_forced < 1: varpi cannot turn, omega is pinned by the
forced vector over the whole record), and keeps its status otherwise. The free elements are fitted over the whole
record, so their verdict on a record that changes regime is the record's, not the episode's: 3752 (500 kyr) librates
in one lobe, crosses to the other at 180 kyr and librates there, the gate finds a planetary residual (free omega
winds 1.5 turns), and the transient stands; 1866 and 2020 PP1 (rho 3.4 and 7) keep their transient as well.
Its libration period is P_cyc, the median full period of the accepted episodes ("cycles in the gate by turning
points", DECISIONS, REVIEW §6.1), and n_cycles = baseline / P_cyc. The e-i exchange
(`coherence_analysis`) is a column of the verdict (`check_zlk_ei`) and changes nothing here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from resonances.resonance.analysis.confidence import ConfidenceCells
from resonances.resonance.analysis.secular.analyse import analyse_long_period
from resonances.resonance.analysis.secular.decide import GATED, AngleDecision
from resonances.resonance.analysis.secular.models import GateResult, LongPeriodAnalysis, LongPeriodSeries
from resonances.resonance.analysis.settings import AnalysisSettings


def analyse_zlk(
    series: LongPeriodSeries,
    settings: AnalysisSettings,
    cells: Optional[ConfidenceCells] = None,
    free_elements: Any = None,
    gate_config: Any = None,
) -> LongPeriodAnalysis:
    """The verdict of omega: the shared geometry, then the free-omega gate on a bounded omega.

    `free_elements` is the body's `secular.free_elements.FreeElements` (None: not built, the gate says unavailable);
    `gate_config` anything with `free_gate_enabled` and `free_gate_min_cycles` (a SimulationConfig; None: the gate on
    with `omega_free_gate.DEFAULT_MIN_CYCLES`)."""
    from resonances.resonance import omega_free_gate

    config = gate_config
    if config is None:
        config = SimpleNamespace(free_gate_enabled=True, free_gate_min_cycles=omega_free_gate.DEFAULT_MIN_CYCLES)

    def gate(angle: AngleDecision, geometry) -> Optional[GateResult]:
        if angle.angle_status not in GATED:
            return GateResult(str(omega_free_gate.GateOutcome.NOT_APPLICABLE))
        period = float(geometry.period_cycle_yr)
        verdict = omega_free_gate.evaluate_gate(free_elements, period if np.isfinite(period) else None, config)
        return GateResult(
            outcome=str(verdict.outcome),
            n_cycles=verdict.n_cycles,
            matches=tuple(verdict.matches),
            min_cycles=float(config.free_gate_min_cycles),
            libration_period_yr=period,
            rho_max=float(free_elements.rho_envelope()[1]) if free_elements is not None else float('nan'),
        )

    return analyse_long_period(series, settings, cells, gate=gate)
