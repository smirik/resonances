"""`analyse_zlk`: the Lidov-Kozai angle omega = varpi - Omega.

The geometry and the angle decision are those of a secular angle (`analysis.secular`); the physics is the
free-omega gate (`resonance/omega_free_gate.py`): a bounded osculating omega is a ZLK libration only if the free
omega, after the forced terms at the planetary fundamentals are removed, stays bounded at non-planetary rates.

The gate judges every omega that stays within a turn (`decide.GATED`): libration, episodic, drift or too few cycles
(555527: 100 kyr, e 0.01-0.05, omega jumps by ~150 deg where e passes near zero, its centre drifts). A libration takes
the gate's verdict as it is; an episodic omega, a drift or too few cycles is made non_resonant only by a kinematic
verdict with the free circle never around the origin, and keeps its status otherwise. "Never around the origin" is
rho_max = max over the record of rollmed(|z_free|, W) / rollmed(|z_forced|, W) < 1, W = max(the beat window,
ZLK_RHO_WINDOW_CYCLES x P_cyc) (`FreeElements.rho_local_max`): each radius is read over about one libration cycle, so a
free circle that grows in part of the record is seen (555527: 0.66 at 100 kyr, 0.63 at 500 kyr, both < 1). The former
median e_free / smallest smoothed forced radius stays a column (`zlk_gate_rho_global`). The free elements are fitted
over the whole record, so their verdict on a record that changes regime is the record's, not the episode's: 3752
(500 kyr) librates in one lobe, crosses to the other at 180 kyr and librates there, the gate finds a planetary residual
(free omega winds 1.5 turns), and the transient stands; 1866 and 2020 PP1 keep their transient as well.

Co-orbitals: a body inside the co-orbital region of a planet whose own modes the split does not fit (Mercury to Mars;
ZLK_COORBITAL_HILL_FACTOR) gets the gate outcome unavailable (coorbital_unfitted_forcing): uncertain/gate_unavailable
for a libration, its status kept for an episodic row (2020 PP1, an Earth co-orbital, stays transient). Jupiter's
Trojans are covered: their forced vector is Jupiter's, g5 + g6 (591986 stays kinematic).

The gate's libration period is P_cyc of the decisive geometry, the median full period of the accepted episodes, and
n_cycles = baseline / P_cyc: cycles of the osculating omega, not of the free one. A libration at the centre scale whose
period is a measured beat of the fitted forced modes, whose forced phase wave explains the centre, while rho_max < 1,
never becomes that period (`diagnostics.forced_beat_centre`); the guard acts only where the gate does (enabled, a split,
no unfitted co-orbital): a disabled gate keeps the geometric result.

Columns that change no status (`diagnostics`, `secular.slow`): the e-i coupling at the decisive period for every row
(`zlk_ei_*`, `check_zlk_ei`, `confirmed_ei`), the attribution of a kinematic libration to a forced mode
(`zlk_forcing_*`), and the slow line of the centre (`zlk_centre_*`) even when the fine scale decided.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from resonances.resonance.analysis.confidence import ConfidenceCells
from resonances.resonance.analysis.secular import slow
from resonances.resonance.analysis.secular.analyse import analyse_long_period
from resonances.resonance.analysis.secular.decide import GATED, AngleDecision
from resonances.resonance.analysis.secular.models import CentreLine, GateResult, LongPeriodAnalysis, LongPeriodSeries
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import AngleStatus
from resonances.resonance.analysis.zlk import diagnostics

NAN = float('nan')


def analyse_zlk(
    series: LongPeriodSeries,
    settings: AnalysisSettings,
    cells: Optional[ConfidenceCells] = None,
    free_elements: Any = None,
    gate_config: Any = None,
) -> LongPeriodAnalysis:
    """The verdict of omega: the shared geometry, then the free-omega gate on a bounded omega; then the columns.

    `free_elements` is the body's `secular.free_elements.FreeElements` (None: not built, the gate says unavailable);
    `gate_config` anything with `free_gate_enabled` and `free_gate_min_cycles` (a SimulationConfig; None: the gate on
    with `omega_free_gate.DEFAULT_MIN_CYCLES`)."""
    from resonances.resonance import omega_free_gate

    config = gate_config
    if config is None:
        config = SimpleNamespace(free_gate_enabled=True, free_gate_min_cycles=omega_free_gate.DEFAULT_MIN_CYCLES)
    a = np.asarray(series.a, dtype=float)
    a_mean = float(np.median(a[np.isfinite(a)])) if np.isfinite(a).any() else NAN
    coorbital = omega_free_gate.coorbital_planet(a_mean, free_elements, settings.zlk_coorbital_hill_factor)

    def rho_max(period: float) -> float:
        if free_elements is None:
            return NAN
        return float(free_elements.rho_local_max(period if np.isfinite(period) else None, settings.zlk_rho_window_cycles))

    def gate(angle: AngleDecision, geometry) -> Optional[GateResult]:
        if angle.angle_status not in GATED:
            return GateResult(str(omega_free_gate.GateOutcome.NOT_APPLICABLE))
        period = float(geometry.period_cycle_yr)
        verdict = omega_free_gate.evaluate_gate(free_elements, period if np.isfinite(period) else None, config, coorbital)
        return GateResult(
            outcome=str(verdict.outcome),
            n_cycles=verdict.n_cycles,
            matches=tuple(verdict.matches),
            min_cycles=float(config.free_gate_min_cycles),
            libration_period_yr=period,
            rho_max=rho_max(period),
            rho_global=float(free_elements.rho_envelope()[1]) if free_elements is not None else NAN,
            baseline_yr=float(free_elements.baseline) if free_elements is not None else NAN,
            min_distance_arcsec_yr=float(verdict.min_distance) if verdict.min_distance is not None else NAN,
            detail=verdict.detail,
        )

    def centre_guard(slow_geometry, fine_geometry, ledger) -> str:
        return diagnostics.forced_beat_centre(
            free_elements,
            float(slow_geometry.period_cycle_yr),
            rho_max(float(fine_geometry.period_cycle_yr)),
            fine_geometry.centre,
            series.t_yr,
            settings,
            ledger,
        )

    # The guard reads the forced/free split: only where the gate would judge it. A disabled gate keeps the geometric
    # result; no split, or a co-orbital whose forcing the split does not fit, has no forced beat to refuse with.
    guarded = bool(getattr(config, 'free_gate_enabled', True)) and free_elements is not None and not coorbital
    analysis = analyse_long_period(series, settings, cells, gate=gate, centre_guard=centre_guard if guarded else None)
    _add_columns(analysis, free_elements, settings)
    return analysis


def _add_columns(analysis: LongPeriodAnalysis, free_elements, settings: AnalysisSettings) -> None:
    """The e-i coupling (also check_zlk_ei), the attribution and the slow line of the centre; no status changes."""
    series, verdict = analysis.series, analysis.verdict
    geometry = analysis.geometry
    result = diagnostics.ZlkDiagnostics()
    if geometry is not None:
        period, source = diagnostics.decisive_period(verdict.angle_status, geometry.period_cycle_yr, series.t_yr, series.angle)
        result.ei = diagnostics.measure_ei_coupling(series.t_yr, series.e, series.inc, series.Omega, period, source, settings)
        gate = analysis.gate
        if gate is not None:
            result.attribution = diagnostics.attribute_forcing(free_elements, gate.outcome, gate.libration_period_yr, gate.rho_max)
        # a libration's centre, or one that moves without turning (drift); a circulating or episodic one has no slow
        # line of its own
        if verdict.angle_status in (AngleStatus.LIBRATION, AngleStatus.DRIFT):
            fine = analysis.fine_geometry if analysis.fine_geometry is not None else geometry
            analysis.centre_line = slow.zlk_centre_line(fine.centre, series, free_elements, settings)
    else:
        result.ei.reason = 'no_geometry'
    if analysis.centre_line is None:
        analysis.centre_line = CentreLine('zlk_centre')
    analysis.zlk = result
    verdict.checks['check_zlk_ei'] = result.ei.check
