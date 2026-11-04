"""`analyse_zlk`: the Lidov-Kozai angle omega = varpi - Omega.

omega, i and Omega are measured from the plane ZLK_ANGLE_FRAME names (default the invariable plane, by an exact
rotation of the orbit's vectors: `simulation.long_period_series`, `LongPeriodSeries.frame`); the free elements, the
forcing panel, the attribution, the node lock and the e-i coupling read the same frame (the coupling never rotates an
invariable series again). From the ecliptic, the node of an orbit tilted from the invariable plane by about the plane's
own 1.58 deg librates about the plane's node instead of turning and omega with it: 555527 (i_inv about 1.6 deg) looked
bounded (kinematic) from the ecliptic and circulates from the invariable plane (W28 g09-1). Where the two frames' omega
part by a whole turn or more, the row fires rule zlk_invariable_frame (`frame_shift_turns`, column
zlk_frame_shift_turns).

The geometry and the angle decision are those of a secular angle (`analysis.secular`); the physics is the
free-omega gate (`resonance/omega_free_gate.py`): a bounded osculating omega is a ZLK libration only if the free
omega, after the forced terms at the planetary fundamentals are removed, stays bounded at non-planetary rates.

The gate judges every omega that stays within a turn (`decide.GATED`): libration, episodic, drift or too few cycles
(1866 and 3752 over 500 kyr are episodic, 3752 over 100 kyr has too few cycles). A libration takes
the gate's verdict as it is; an episodic omega, a drift or too few cycles is made non_resonant only by a kinematic
verdict with the free circle never around the origin, and keeps its status otherwise. "Never around the origin" is
rho_max = max over the record of rollmed(|z_free|, W) / rollmed(|z_forced|, W) < 1, W = max(the beat window,
ZLK_RHO_WINDOW_CYCLES x P_cyc) (`FreeElements.rho_local_max`): each radius is read over about one libration cycle, so a
free circle that grows in part of the record is seen (555527 read from the ecliptic: 0.66 at 100 kyr, 0.63 at 500
kyr, both < 1). The former median e_free / smallest smoothed forced radius stays a column (`zlk_gate_rho_global`).
The free elements are fitted over the whole record, so their verdict on a record that changes regime is the
record's, not the episode's: 3752 (500 kyr) librates in one lobe, crosses to the other at 180 kyr and librates there,
the gate finds a planetary residual (free omega winds 1.5 turns); 1866 (kinematic, not pinned) keeps its transient
(2020 PP1 is an unfitted co-orbital, below). W34 (`ZLK_GATE_EPISODES`, `omega_free_gate.episode_confirmation`): an
episodic omega whose free omega winds over the record is read inside its librating episodes, where it is confirmed when
the free omega stays put in every one (3752, 373136, 797204 at 500 kyr, 1866, 88899 500 kyr: 0.001-0.04 turns per
cycle, rho >= 3). An episodic omega without a gate verdict is transient only when e and i
trade at constant H inside its deciding episode (W28 D10, the author's q-1 E; ZLK_EPISODIC_NO_VERDICT): the gate
callback measures that coupling (`episode_ei.measure_episode_coupling`, columns zlk_ei_episode_*) and passes it with
the policy in the GateResult; 3752 (500 kyr: zlk in the 4.6-period episode) stays transient, 88899 (500 kyr: e and i
in phase, H not conserved) becomes uncertain/gate_unavailable under v2 (the minimal policy: confirmed inside its
episodes, the broken integral an objection, transient/low).

Co-orbitals: a body inside the co-orbital region of a planet whose own modes the split does not fit (Mercury to Mars;
ZLK_COORBITAL_HILL_FACTOR) gets the gate outcome unavailable (coorbital_unfitted_forcing): uncertain/gate_unavailable
for any bounded omega, an episodic one too (2020 PP1, an Earth co-orbital; W28 g09-6 B, rule zlk_coorbital_episodic).
Jupiter's Trojans are covered: their forced vector is Jupiter's, g5 + g6 (591986 stays kinematic).

The gate's libration period is the deciding librating episode's median full period (`decide.decisive_period_yr`; the
pooled P_cyc over runs not faster than the floor without one), and n_cycles = baseline / that period: cycles of the
osculating omega, not of the free one. The rho window and the e-i coupling read the same period (W28 D4: the pooled
median of every episode once set them, 2020 PP1). A libration at the centre scale whose
period is a measured beat of the fitted forced modes, whose forced phase wave explains the centre, while rho_max < 1,
never becomes that period (`diagnostics.forced_beat_centre`); the guard acts only where the gate does (enabled, a split,
no unfitted co-orbital): a disabled gate keeps the geometric result.

W29: the minimal policy (`analysis.policy`, the last step) reads the gate as a measured veto (kinematic on a libration;
the free e below the forced e) or a flag (no verdict, disabled: the geometry's status at <= medium, flag
gate_unavailable / gate_disabled); ZLK_EPISODIC_NO_VERDICT acts on the W28 tree only (`v2_verdict`).
Columns that change no status (`diagnostics`, `secular.slow`): the e-i coupling of the whole record at the decisive
period for every row (`zlk_ei_*`, `check_zlk_ei`, `confirmed_ei`; the episode's own coupling is the one a status reads),
the attribution of a kinematic libration to a forced mode (`zlk_forcing_*`), the slow line of the centre
(`zlk_centre_*`) even when the fine scale decided, the node lock
(`zlk_node_lock*`: the node's rate one line with a forced mode's and Omega - varpi_j librating), and the node near s_j
(`free_rho_i_max`, `zlk_gate_node_near*`: a planetary residual that is only the free node within a width of s_j).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from resonances.data.invariable_plane import FRAME_INVARIABLE
from resonances.resonance.analysis import policy as policy_module
from resonances.resonance.analysis.confidence import ConfidenceCells
from resonances.resonance.analysis.rules import Rule, ordered
from resonances.resonance.analysis.secular import slow
from resonances.resonance.analysis.secular.analyse import analyse_long_period
from resonances.resonance.analysis.secular.decide import (
    GATED,
    AngleDecision,
    decisive_librating_episode,
    decisive_period_yr,
    librating_episodes,
)
from resonances.resonance.analysis.secular.models import CentreLine, GateResult, LongPeriodAnalysis, LongPeriodSeries
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import AngleStatus
from resonances.resonance.analysis.zlk import diagnostics
from resonances.resonance.analysis.zlk.episode_ei import measure_episode_coupling

NAN = float('nan')
# rule zlk_invariable_frame: omega(invariable) - omega(ecliptic) spans at least one turn, so the two frames give omega
# different winding numbers (a definition, not a convention: less than a turn moves the angle, never its topology)
FRAME_WINDING_TURNS = 1.0


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
            return GateResult(str(omega_free_gate.GateOutcome.NOT_APPLICABLE), episodic_policy=settings.zlk_episodic_no_verdict)
        period = decisive_period_yr(geometry, angle.decisive_episode_id)
        verdict = omega_free_gate.evaluate_gate(free_elements, period if np.isfinite(period) else None, config, coorbital)
        pinning = _episode_pinning(free_elements, angle, geometry, settings)
        if pinning and settings.zlk_gate_episodes:
            verdict = omega_free_gate.episode_confirmation(
                verdict,
                free_elements,
                pinning['rho_episodes_min'],
                pinning['free_turns_episodes_max'],
                settings.zlk_episode_free_turns_per_cycle,
            )
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
            free_turns_min=settings.zlk_episode_free_turns_per_cycle,
            # the e-i coupling inside the deciding librating episode (a column for a libration; the status of an
            # episodic omega without a gate verdict under ZLK_EPISODIC_NO_VERDICT=ei, `decide_status`)
            episode_ei=measure_episode_coupling(series, decisive_librating_episode(geometry, angle.decisive_episode_id), settings),
            episodic_policy=settings.zlk_episodic_no_verdict,
            **pinning,
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
    if series.frame == FRAME_INVARIABLE and series.frame_shift_turns >= FRAME_WINDING_TURNS:
        # omega from the invariable plane and from the ecliptic part by a whole turn or more: the frame decided how
        # omega winds (555527: bounded from the ecliptic, circulating from the invariable plane)
        analysis.verdict.rules_fired = ordered(analysis.verdict.rules_fired, [Rule.ZLK_INVARIABLE_FRAME])
    if free_elements is not None and free_elements.resample_remainder:
        # the split this row's gate and columns read was resampled with a polyphase leftover pass (D1, 555527)
        analysis.verdict.rules_fired = ordered(analysis.verdict.rules_fired, [Rule.FREE_RESAMPLE_REMAINDER])
    return policy_module.apply_long_period(analysis, settings)  # W29: the policy layer, after the columns


def _episode_pinning(free_elements, angle: AngleDecision, geometry, settings: AnalysisSettings) -> dict:
    """{rho_episodes, free_turns_episodes, rho_episodes_min, free_turns_episodes_max} of an episodic omega
    (`GateResult`): over its librating episodes the largest median of |z_free| / |z_forced| inside one and the fewest
    turns of the free omega per cycle inside one (the kinematic pin), and the smallest median and the most turns (the
    confirmation inside every episode, `omega_free_gate.episode_confirmation`) (`FreeElements.episode_pinning`: nan
    where the free elements do not measure an episode, which then neither pins nor confirms); {} otherwise.

    A librating episode of the osculating omega while the free circle stays off the origin (median ratio < 1) and the
    free vector turns about the forced one once per cycle (a kinematic cycle: P_cyc is their beat; a dynamic one
    leaves the free omega librating, about 0 turns over whole cycles) is the forced vector's, whatever the free circle
    does between the episodes (624 Hektor: eleven 1-3-cycle librations of omega where the free radius dips just below
    the forced one).
    """
    if free_elements is None or angle.angle_status != AngleStatus.EPISODIC:
        return {}
    episodes = librating_episodes(geometry)
    if not episodes:
        return {}
    medians, turns = [], []
    for episode in episodes:
        tp = np.asarray(episode.tp_times_yr, dtype=float)
        cycles = (len(tp) - 1) // 2
        if cycles < 1:
            medians.append(NAN)
            turns.append(NAN)
            continue
        rho, per_cycle = free_elements.episode_pinning(float(tp[0]), float(tp[-1]), float(tp[2 * cycles]), cycles)
        medians.append(rho)
        turns.append(per_cycle)
    return {
        'rho_episodes': float(np.max(medians)),
        'free_turns_episodes': float(np.min(turns)),
        'rho_episodes_min': float(np.min(medians)),
        'free_turns_episodes_max': float(np.max(turns)),
    }


def _librating_period(geometry) -> float:
    """The median full period pooled over the librating episodes [yr] (nan without one with a full cycle): what the
    attribution's beat test reads for an omega pinned inside every episode (`diagnostics.attribute_forcing`)."""
    times = [np.asarray(episode.tp_times_yr, dtype=float) for episode in librating_episodes(geometry)]
    periods = [t[2:] - t[:-2] for t in times if len(t) >= 3]
    return float(np.median(np.concatenate(periods))) if periods else NAN


def _add_columns(analysis: LongPeriodAnalysis, free_elements, settings: AnalysisSettings) -> None:
    """The e-i coupling (also check_zlk_ei), the attribution, the slow line of the centre and the node lock; no status
    changes."""
    series, verdict = analysis.series, analysis.verdict
    geometry = analysis.geometry
    result = diagnostics.ZlkDiagnostics()
    if geometry is not None:
        libration = decisive_period_yr(geometry, analysis.decisive_episode_id)
        period, source = diagnostics.decisive_period(verdict.angle_status, libration, series.t_yr, series.angle)
        result.ei = diagnostics.measure_ei_coupling(
            series.t_yr, series.e, series.inc, series.Omega, period, source, settings, frame=series.frame
        )
        gate = analysis.gate
        if gate is not None:
            result.attribution = diagnostics.attribute_forcing(
                free_elements,
                gate.outcome,
                gate.libration_period_yr,
                gate.rho_max,
                gate.rho_episodes,
                gate.free_turns_episodes,
                gate.free_turns_min,
                series.step_yr,
                episodes_period_yr=_librating_period(geometry),
            )
        # a libration's centre, or one that moves without turning (drift); a circulating or episodic one has no slow
        # line of its own
        if verdict.angle_status in (AngleStatus.LIBRATION, AngleStatus.DRIFT):
            fine = analysis.fine_geometry if analysis.fine_geometry is not None else geometry
            analysis.centre_line = slow.zlk_centre_line(fine.centre, series, free_elements, settings)
    else:
        result.ei.reason = 'no_geometry'
    if analysis.centre_line is None:
        analysis.centre_line = CentreLine('zlk_centre')
    result.node_lock = diagnostics.node_lock(free_elements, settings)
    result.node_near = diagnostics.node_near(free_elements, analysis.gate, settings)
    analysis.zlk = result
    verdict.checks['check_zlk_ei'] = result.ei.check
