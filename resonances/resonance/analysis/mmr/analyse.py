"""`analyse_mmr`: the steps of the MMR analysis in their fixed order (SPEC §2.4).

   1 context        build_context                       9 kinematics     analyse_kinematics
   2 sampling       analyse_sampling (predictive)      10 angle_status   decide_angle (geometry only)
   0 input          MMRSeries.validate (structure) and domain_problems (k = 0, a <= 0, e outside [0, 1), NaN):
                    invalid with its reason before anything is computed, never an exception
   3 G0             unwrap failed -> undefined verdict, the rest not_measured; P_c <= 2 dt (the step
                    is too coarse for the angle's own filter) -> invalid / undersampled, resample
   4 prep           preprocess (ripple, cutoff, filter) 11 subresonances run_subresonances when triggered
   5 geometry       measure_geometry, last regime      12 owner          run_owner when required
   6 slow           slow_components                    13 strength       analyse_strength
   7 response       analyse_response, analyse_episodes 14 status         decide_status (the only place a status is set)
   8 energy         zone.analyse_separatrix, finish_geometry
  15 filter guard   the decisive quantities again at the cutoff x2 (DELTA 13a); runs after step 10, before step 11
  16 confidence     margins, checks, caps, confidence.assess

angle_status depends on the geometry (and the sigma-only potential) alone, so it is decided before the foreign steps:
the subresonance triggers need it (T5 is the reason the tree would give without the scan, T6 asks whether the angle
is a clean stationary libration), and steps 11-13 never change it. The verdict reports it over the whole record
(`decide.reported_angle`: episodes followed by a circulation, G1c, or by a regime without episodes after a jump of a,
G1e, are episodic) and the last regime in `final_regime` (`decide.final_regime`); the calibration cell reads the
reported one. Only the output changes: every step above reads the decision as it is.

Triggers of the foreign steps:
  subresonances  the angle has turning points and one of T1 (drift, moving or modulated centre), T2 (a responds, phi0
                 rotates), T5 (the pending reason is kinematic_* or drift_mechanism_unknown), T6 (the leading member
                 of the multiplet by the averaging is not the angle itself, unless the angle is a clean stationary
                 libration) fires (`subresonances.triggers`);
  owner          the angle oscillates (episode, drift, turning points) or its force law is not measured
                 (`owner.owner_required`), or the angle circulates: the owner then names the origin of a circulation
                 (2000 YG4 2R-8E-1 circulates while the body sits in 7E-23).

Owner per episode (A1, W28): an accepted owner whose measured regimes cover some accepted episodes (with P2) but not
others leaves the row's own episodes to decide the status: the analysis runs once more with the induced episodes taken
out of the geometry (`Geometry.induced_ids`, as F6's forced waves: `_without_episodes`) and the owner of the first pass
(`_InducedPass`); all episodes induced -> rule 2 as before, none -> origin none (F7).

An accepted planar subresonance is analysed once more as a derived row of its own (summary.csv with `added_by` and
`parent`, its own combined figure): its series is sigma0 + Delta_s from the saved elements
(`subresonances.member_series`), depth 1 (no scans of its own), `multiplet_status` shared with the parent. It is
analysed before the catalogue row's status: that row takes the member's status and confidence (rule C,
`decide.member_status`; both librating: each keeps its own verdict).

The filter guard (DELTA 13a): turning points, episode boundaries, stalls, the energy per cycle, amplitude and centre
shift are measured again at 2 P_c on the interval both cutoffs see, on the episode the status reads and its
counterpart at 2 P_c (largest time overlap; none = not reproduced), else on the whole interval; a number that changes
beyond the tolerance of its setting (or is measured at one cutoff only) empties the checks and margins that read it
(not_measured, flag filter_sensitive) and lowers the confidence; nothing comparable = not checked (check_mmr_filter
not_measured), never 'unchanged'. It runs before the tree (W21 Q9, principle b): a passed check it withdraws is
not_measured for the tree too (`decide.energy_for_tree`: R0 with a), a failed one stays an objection; the column
flt_withdrawn_before_tree names them. Under filter_too_slow the shape margins (L2, L5, L6) are not measured either.

The G2a guard (W24 review): a record a step of the libration action cut (`geometry.action_steps`, anywhere in its
analysis) is analysed once more whole (libration_action_step off); when the cut leaves a measurement that failed on the
whole record unmeasured (`CUT_GUARDED_CHECKS`) and changes the verdict, the whole record's analysis is kept and the step
reported as withheld (`_action_step_guard`): geometry does not override a failed measurement (W22). Every check is
guarded, R0 too: its period law reads each cycle in the well that holds it (`pendulum.cycle_period`), so a whole
record of two energy levels in one potential of two wells is a valid R0 measurement, not a pooled one. Each step is
bound to the row whose analysis found it (`ActionStep.key`): a row's hidden failures count only when that row cut.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis import confidence as confidence_module
from resonances.resonance.analysis import policy as policy_module
from resonances.resonance.analysis import filtering, geometry as geometry_module, orbit_type as orbit_type_module, pendulum
from resonances.resonance.analysis.confidence import Cap, ConfidenceCells
from resonances.resonance.analysis.filtering import is_finite
from resonances.resonance.analysis.geometry import Geometry
from resonances.resonance.analysis.ledger import Ledger, Margin, Op
from resonances.resonance.analysis.rules import Rule, fire
from resonances.resonance.analysis.mmr import background_wave as background_wave_module
from resonances.resonance.analysis.mmr import centre_track as centre_track_module
from resonances.resonance.analysis.mmr import context as context_module, coorbital as coorbital_module, coorbital_geometry
from resonances.resonance.analysis.mmr import coverage_audit as coverage_audit_module
from resonances.resonance.analysis.mmr import decide, kinematics as kinematics_module, owner as owner_module
from resonances.resonance.analysis.mmr import owner_identity
from resonances.resonance.analysis.mmr import preprocess as preprocess_module, response as response_module
from resonances.resonance.analysis.mmr import quasi_satellite as quasi_satellite_module
from resonances.resonance.analysis.mmr import roles as roles_module
from resonances.resonance.analysis.mmr import sampling as sampling_module, strength as strength_module
from resonances.resonance.analysis.mmr import slow_swing as slow_swing_module
from resonances.resonance.analysis.mmr import subresonances as subresonances_module, zone as zone_module
from resonances.resonance.analysis.mmr.models import (
    MMRAnalysis,
    MMRSeries,
    OwnerResult,
    Preprocessed,
    Response,
    Sampling,
    DIAGNOSTIC_TRIGGERS,
    SlowScale,
    SubresonanceCheap,
    SubresonanceResult,
    SubresonanceTrigger,
    physics_of,
)
from resonances.resonance.analysis.mmr.preprocess import FilterSensitivity
from resonances.resonance.analysis.pendulum import EnergyAnalysis
from resonances.resonance.analysis.settings import AnalysisSettings, settings_hash
from resonances.resonance.analysis.verdict import (
    Action,
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    Flag,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
    best_status,
    default_action,
    join_list,
    unmeasured_checks,
)

NAN = float('nan')
BYTES_PER_MB = 1e6
_UNSET = object()  # `regime_prominence`: no fast_part given
FAST_RESIDUAL_PERCENTILE = 95  # the peak of a modulated wave without its rare outliers
VERDICT_FLAGS = frozenset(str(flag) for flag in Flag)
# the G2a cuts made anywhere in the analysis of one record (`_finish`); None outside `analyse_mmr`, whose outermost call
# guards them (`_action_step_guard`)
_ACTION_STEP_CUTS: ContextVar[Optional[list]] = ContextVar('action_step_cuts', default=None)
# the measurements a G2a cut may not hide: failed on the whole record, neither passed nor failed after the cut
CUT_GUARDED_CHECKS = (
    'check_mmr_identity',
    'check_mmr_a_response',
    'check_mmr_potential',
    'check_mmr_separatrix_zone',
    'check_mmr_strength',
)
# the resonance key of the row being analysed (`_analyse_mmr`): the G2a steps found in its passes carry it
_ROW_KEY: ContextVar[str] = ContextVar('action_step_row', default='')
# the steps of MMRAnalysis; a record stopped before them (`_verdict_only`) has None for each
MEASUREMENTS = (
    'context',
    'sampling',
    'prep',
    'slow',
    'geometry',
    'energy',
    'response',
    'kinematics',
    'subresonances',
    'owner',
    'strength',
    'baseline',
)


# --------------------------------------------------------------------------------------------- small helpers


def _period_floor(prep: Preprocessed, settings: AnalysisSettings) -> float:
    """The shortest libration period of the angle [yr]: the preprocessing's, else mmr_libration_period_min_yr."""
    return prep.period_floor_yr if np.isfinite(prep.period_floor_yr) else settings.mmr_libration_period_min_yr


# --------------------------------------------------------------------------------------------- geometry and energy


def _energy(series, prep, slow, geometry, response, context, sampling, settings, well=None) -> Optional[EnergyAnalysis]:
    """The energy relative to the separatrix (zone.analyse_separatrix); None for an aliased angle (sigma_dot is the
    model rate there: no potential can be read from it) and without a cycle or a turn. R0's period law reads the full
    averaged well `well` where the direct averaging applies (`strength.direct_applicable`: two-body, not 1:1, whose
    energy integral is its own)."""
    if sampling.verdict in decide.ALIASED:
        return None
    full_well = well if strength_module.direct_applicable(series.resonance) else None
    return zone_module.analyse_separatrix(series, prep, slow, geometry, response, context, settings, full_well)


def fast_part(prep: Preprocessed, settings: AnalysisSettings, run_floor=None) -> Optional[np.ndarray]:
    """|sigma_f - lowpass(sigma, P_floor) - image of the libration| [rad], what sigma_f holds faster than any libration
    of the angle (P_floor: `preprocess.fast_reference`, the floor `preprocess.libration_period_floor`, for a 1:1 angle
    the shortest floor any of its runs can get, W28 C3: a horseshoe turning near the planet is faster than the tadpole
    floor, and as a fast residual it raised the prominence over its own swing, 2003 YN107 233 deg); None when the bound
    is off or cannot be formed.

    The two filters split softly at P_floor: a libration of period P >= P_floor keeps G(P; P_c) - G(P; P_floor) of
    its swing in sigma_f - lowpass(sigma, P_floor) (0.44 at 150 yr, P_c = 75 yr), and that image of the libration
    itself raised the threshold to 4 x 0.44 A, about its own swing 2 A (2013 AS27 31E-36+0+5: 48.9 deg, a libration
    of 260 TPs cut into 19 runs). The image of each libration run through the same two filters
    (`libration_image`) is taken out first; a wave faster than P_floor is no libration run and stays. `run_floor`: the
    1:1 model of the G1q pass (`CoorbitalRunFloor` with qs_pass), whose QS runs below P_floor keep their image when
    they reach their own loop floor (rule coorbital_qs_run_floor)."""
    period = preprocess_module.fast_reference(prep, settings)
    if period <= prep.cutoff.cutoff_yr:
        return None
    t = np.asarray(prep.t_yr, float)
    slow = filtering.lowpass_or_none(t, prep.angle_clean, period, settings)
    if slow is None:
        return None
    return np.abs(np.asarray(prep.angle_f, float) - slow - libration_image(prep, period, settings, run_floor))


# the period floor that is a convention (MMR_LIBRATION_PERIOD_MIN_YR, 150 yr) and so read with its thin band (W31); a
# co-orbital's floor is its planet's tadpole period, a measurement
CONVENTION_FLOOR = 'mmr_libration_period_min_yr'


def _image_floor_of(run_floor):
    """The per-run floor the libration image reads (`geometry.libration_image`, `run_floor_of`): the 1:1 model's
    `floor_of` in the G1q pass (rule coorbital_qs_run_floor), None anywhere else (the first pass, every other angle)."""
    return run_floor.floor_of if run_floor is not None and run_floor.qs_pass else None


def libration_image(prep: Preprocessed, period_floor_yr: float, settings: AnalysisSettings, run_floor=None) -> np.ndarray:
    """`geometry.libration_image` of sigma_f with the turning points at mmr_tp_prominence_floor_deg (the shared rule,
    also the secular/ZLK prominence's); `run_floor` as in `fast_part`."""
    return geometry_module.libration_image(
        prep.t_yr,
        prep.angle_f,
        tuple(prep.window),
        prep.cutoff.cutoff_yr,
        period_floor_yr,
        math.radians(settings.mmr_tp_prominence_floor_deg),
        settings,
        floor_band=settings.confidence_thin_band if prep.period_floor_basis == CONVENTION_FLOOR else 0.0,
        run_floor_of=_image_floor_of(run_floor),
    )


def fast_residual(
    prep: Preprocessed,
    settings: AnalysisSettings,
    own: Optional[np.ndarray] = None,
    part: Optional[np.ndarray] = None,
    run_floor=None,
) -> float:
    """Half-amplitude of what sigma_f holds faster than any libration of the angle [rad]: the 95th percentile of
    `fast_part` in the window (and in the samples `own`, a regime between jumps of a, when given; `part`: `fast_part`
    computed already). A forced wave of a nearby commensurability that the cutoff lets through (52356: 121 yr, 23240:
    115 yr) makes turning points of up to twice this; 0 when the bound is off or cannot be formed. `run_floor` as in
    `fast_part`."""
    inside = filtering.window_mask(np.asarray(prep.t_yr, float), prep.window)
    if own is not None:
        inside &= own
    part = fast_part(prep, settings, run_floor) if part is None and np.any(inside) else part
    if part is None or not np.any(inside):
        return 0.0
    return float(np.percentile(part[inside], FAST_RESIDUAL_PERCENTILE))


def _prominence(prep: Preprocessed, settings: AnalysisSettings, fast_residual_rad: Optional[float] = None) -> float:
    """TP prominence: the floor, or mmr_tp_ripple_factor x the larger of the unremoved planetary ripple and the
    fast residual (`fast_residual`, computed here when not given)."""
    fast = fast_residual(prep, settings) if fast_residual_rad is None else fast_residual_rad
    ripple = max(prep.unremoved_ripple_rad, fast)
    return geometry_module.turning_prominence(math.radians(settings.mmr_tp_prominence_floor_deg), settings.mmr_tp_ripple_factor, ripple)


def regime_prominence(prep: Preprocessed, settings: AnalysisSettings, run_floor=None, part=_UNSET) -> geometry_module.LocalProminence:
    """The TP prominence of a stretch from the mask of its own samples: a regime between jumps of a
    (`geometry.finish_geometry`) or a run of provisional turning points found at the floor `mmr_tp_prominence_floor_deg`
    (`geometry.local_thresholds`). `_prominence` with the fast residual measured on those samples only; the unremoved
    ripple stays the record's.

    The fast residual of the record mixes the regimes: where sigma circulates in a few decades it is the circulation
    itself (2003 UG22 10E-17+0+7, 40 kyr: 80 deg over the record, 26 deg on the 28.9 kyr before the jumps of a). The
    fast part is the record's low-pass at the period floor P_f, whose response to a jump spreads over
    filter_edge_trim_cutoffs x P_f, so the regime's samples within that distance of a jump do not count; what is left
    must span at least P_f (one cycle of the slowest wave the residual holds). A shorter regime has no measurement of
    its own: NaN, and the geometry keeps the record's prominence there (54 of the UG22 regimes are shorter than P_f).
    A run's ends are where its oscillation starts and stops, so they are trimmed the same way (2001 PG14 33S-2+0-31:
    10.0 deg on the run at 0.35-2.8 kyr against 21.0 deg of the record). `run_floor` as in `fast_part`; `part`: `fast_part`
    of the pass when computed already (None included), else computed at the first regime that needs it."""
    t = np.asarray(prep.t_yr, float)
    window = filtering.window_mask(t, prep.window)
    floor = preprocess_module.fast_reference(prep, settings)  # the slowest wave the fast part holds (`fast_part`)
    margin = settings.filter_edge_trim_cutoffs * floor
    cache: List[Optional[np.ndarray]] = [] if part is _UNSET else [part]

    def prominence(own: np.ndarray) -> float:
        samples = np.flatnonzero(own)
        if not len(samples):
            return NAN
        # a side that ends at a jump (not at the record's edge) loses the stretch the low-pass carries across the jump
        start = t[samples[0]] + (margin if samples[0] > 0 else 0.0)
        end = t[samples[-1]] - (margin if samples[-1] < len(t) - 1 else 0.0)
        supported = own & window & (t >= start) & (t <= end)
        if not np.any(supported) or np.ptp(t[supported]) < floor:
            return NAN
        if not cache:
            cache.append(fast_part(prep, settings, run_floor))
        part = cache[0]
        return _prominence(prep, settings, fast_residual(prep, settings, supported, part) if part is not None else 0.0)

    return geometry_module.LocalProminence(math.radians(settings.mmr_tp_prominence_floor_deg), prominence)


def _small_period(geometry: Geometry) -> float:
    """T0 of the decisive cycles: P_cyc / F(A_med) over the accepted episodes (else every TP)."""
    period = geometry.period_cycle_yr
    episodes = geometry.episodes
    amplitude = float(np.nanmedian([e.half_amp_med_rad for e in episodes])) if episodes else decide.tp_half_amplitude_median(geometry)
    if not (is_finite(period) and is_finite(amplitude)) or amplitude >= np.pi:
        return NAN
    return float(pendulum.small_period(period, amplitude))


def _finish(geometry, angle, energy, context, sampling, local, settings) -> Geometry:
    t0 = _small_period(geometry)
    potential = energy.potential_e if energy is not None else None
    cycles = energy.cycles if energy is not None else []
    zone = energy.zone if energy is not None else None
    omega0 = 2 * np.pi / t0 if is_finite(t0) and t0 > 0 else NAN
    finished = geometry_module.finish_geometry(
        geometry,
        angle,
        t0,
        potential.top_rad if potential is not None else NAN,
        zone.h_hi if zone is not None else NAN,
        energy.sigma_h if energy is not None else NAN,
        np.array([(c.t_start_yr + c.t_end_yr) / 2 for c in cycles]),
        np.array([c.h for c in cycles]),
        pendulum.half_width(omega0, context.identity_coefficient),
        geometry.period_cycle_yr,
        sampling.suspect_times_yr,
        settings,
        regime_prominence=local,
        cycle_spans_yr=np.array([(c.t_start_yr, c.t_end_yr) for c in cycles], dtype=float).reshape(-1, 2),
        potential=potential,
        frame=energy.model_frame if energy is not None else None,
    )
    key = _ROW_KEY.get()
    finished.action_steps = [replace(step, key=key) for step in finished.action_steps]
    cuts = _ACTION_STEP_CUTS.get()
    if cuts is not None:
        cuts.extend(step for step in finished.action_steps if step.cuts)
    return finished


def _response_and_energy(series, prep, slow, geometry, context, sampling, settings, well=None):
    """The response of a and the energy. A 1:1 angle gets its own force law (`series` holds the body's e, i, omega,
    Omega for the averaged co-orbital potential); its R0 from the co-orbital energy integral, which replaces the
    harmonic V_E ~ V_F test, is taken once the deciding response is known (`response.with_coorbital_potential`);
    every other angle keeps the harmonic ones."""
    response = response_module.analyse_response(prep, slow, geometry, context, sampling, settings, spec=series.resonance, series=series)
    return response, _energy(series, prep, slow, geometry, response, context, sampling, settings, well)


def _pass_fast(prep: Preprocessed, settings: AnalysisSettings, run_floor) -> Tuple[Optional[np.ndarray], float]:
    """The pass's `fast_part` (read again by `regime_prominence`) and its `fast_residual` (0 without a part)."""
    part = fast_part(prep, settings, run_floor)
    return part, (fast_residual(prep, settings, part=part) if part is not None else 0.0)


def _measure(series, context, sampling, prep, prominence, settings, floor_well, run_floor, part):
    """Geometry, slow parts, response and energy; measured again on the last regime after a jump of a. Every run of
    turning points and every regime between jumps is selected at its own prominence (`regime_prominence`). A two-body
    angle with its planet saved carries the track of its moving well (`centre_track`): a run whose centre moves is
    tested again in psi = sigma - c(t) (`geometry._classify_run`), its force law is fitted there and R0 reads the
    full well's periods. `floor_well`: the quasi-satellite well of the G1q pass (`geometry.FloorWell`); `run_floor`: the
    1:1 run-floor model of this pass (`_run_floor_model`); `part`: the pass's `fast_part`."""
    well = centre_track_module.build(series, context, settings, prep.cutoff.cutoff_yr)
    coorbital = response_module.law_planet(series.resonance) is not None
    angle = preprocess_module.filtered_angle(prep, context, sampling.step_yr, coorbital, settings.libration_regime_cuts, floor_well)
    # the turning point the filter's edge trim hides, read on the orbit-averaged raw 1:1 angle (W28 C4)
    edge_turn = coorbital_geometry.coorbital_edge_turn(series, context, prep.t_yr, prep.angle_clean, context.orbital_period_yr, settings)
    # a 1:1 episode ends only at a separatrix; its changes of well and amplitude are annotations (W28 C7)
    wells = coorbital_geometry.coorbital_wells(series, context, settings)
    angle = replace(angle, well_model=well, run_floor=run_floor, edge_turn=edge_turn, coorbital_wells=wells)
    local = regime_prominence(prep, settings, run_floor, part)
    geometry = geometry_module.measure_geometry(angle, prominence, settings, local)
    prep.filter_too_slow, prep.gain_at_period_cycle = preprocess_module.filter_guard(
        geometry.period_cycle_yr, prep.cutoff.cutoff_yr, settings, sampling.step_yr
    )
    if prep.filter_too_slow and str(Flag.FILTER_TOO_SLOW) not in prep.flags:
        prep.flags.append(str(Flag.FILTER_TOO_SLOW))
    slow = preprocess_module.slow_components(series, prep, geometry.period_cycle_yr, settings)
    response, energy = _response_and_energy(series, prep, slow, geometry, context, sampling, settings, well)
    start = geometry.regime_start_yr
    geometry = _finish(geometry, angle, energy, context, sampling, local, settings)
    if geometry.regime_start_yr > start:
        prep.filter_too_slow, prep.gain_at_period_cycle = preprocess_module.filter_guard(
            geometry.period_cycle_yr, prep.cutoff.cutoff_yr, settings, sampling.step_yr
        )
        slow = preprocess_module.slow_components(series, prep, geometry.period_cycle_yr, settings)
        response, energy = _response_and_energy(series, prep, slow, geometry, context, sampling, settings, well)
    for rule in decide.run_floor_acts(geometry, settings):
        geometry.rules_fired = fire(geometry.rules_fired, rule)
    if run_floor is not None:
        run_floor.release()  # the pass is measured: its law models (record-length arrays) are no longer read
    return angle, geometry, slow, response, energy


def _run_floor_model(series, context, prep, settings, floor_well=None):
    """The 1:1 run-floor model of one pass (`coorbital_geometry.CoorbitalRunFloor`, None for another angle): a run whose
    turns reflect on a top of the averaged potential has its own floor (W28 C3); in the G1q pass (`floor_well` given)
    a run of quasi-satellite turns too (rule coorbital_qs_run_floor). One model per pass, built before the prominence,
    so the libration image and the geometry read the same floors from one cache."""
    return coorbital_geometry.coorbital_run_floor(
        series,
        context,
        prep.t_yr,
        prep.angle_clean,
        context.orbital_period_yr,
        prep.cutoff.cutoff_yr,
        settings,
        qs_pass=floor_well is not None,
        qs_floor_yr=prep.period_floor_yr if floor_well is not None else NAN,
        floor_well=floor_well,
    )


def _episode_responses(series, prep, slow, geometry, context, sampling, energy, response: Response, settings, measured=None) -> list:
    """The physics of every run (`response.analyse_episodes`) with L4, L3 and B1 judged on each librating episode, of the
    window and before the last jump of a; those go to `response.episode_responses`, and the measurement pooled over
    the episodes before the jump to `response.before_jump_response` (rule R, `response.deciding_response`). The forced
    waves (`Geometry.forced_waves`) and the induced episodes (`Geometry.induced_ids`) are judged as well, for
    episodes.csv, but are no episodes of the angle."""
    librating = decide.librating_episodes(geometry, settings)
    before_jump = decide.librating_episodes(geometry, settings, before_jump=True)
    judged = [e.episode_id for e in librating + before_jump]
    episodes = response_module.analyse_episodes(
        prep,
        geometry,
        context,
        energy,
        settings,
        spec=series.resonance,
        slow=slow,
        sampling=sampling,
        series=series,
        judged_ids=judged + sorted(geometry.excluded_ids),
        measured=measured,
    )
    response.episode_responses = [p.response for p in episodes if p.episode_id in judged]
    response.before_jump_response = response_module.before_jump_response(
        prep,
        context,
        sampling,
        settings,
        before_jump,
        spec=series.resonance,
        slow=slow,
        series=series,
        guard=response_module.JumpGuard.from_geometry(geometry),
    )
    return episodes


FORCED_WAVE, INDUCED = 'forced_wave', 'induced'  # the kinds of accepted runs `_without_episodes` takes out


def _without_forced_waves(series, prep, slow, angle, geometry, context, sampling, energy, response, episodes, settings, measured=None):
    """`_without_episodes` of the accepted runs that are forced waves of another argument (`decide.forced_wave_episodes`,
    W20 F6), and the unconfirmed ripples on an unfinished slow swing (`decide.ripple_waves`, W28 B6b, rule
    `ripple_on_arc`): they go to `Geometry.forced_waves`. Unchanged when there is none."""
    waves = decide.forced_wave_episodes(response.episode_responses, settings, geometry)
    if any(wave.ripple for wave in waves):
        geometry.rules_fired = fire(geometry.rules_fired, Rule.RIPPLE_ON_ARC)
    args = (series, prep, slow, angle, geometry, context, sampling, energy, response, episodes, settings)
    return _without_episodes(*args, FORCED_WAVE, waves, measured)


def _without_episodes(
    series, prep, slow, angle, geometry, context, sampling, energy, response, episodes, settings, kind, removed, measured=None
):
    """(geometry, response, energy, episodes) with accepted runs left out of the angle's episodes: `kind` FORCED_WAVE,
    `removed` the ForcedWave objects (F6, `Geometry.forced_waves`), or INDUCED, `removed` the ids an owner induces
    while other episodes are the row's own (A1, `Geometry.induced_ids`). The edge rule is read again on the real first
    and last episode (`geometry.refresh_edges`), and the response, the energy and the episodes' physics are measured
    again without them (the response pieces and the decisive episode change). Unchanged when `removed` is empty.
    `measured`: the runs' `measure_pieces` results of the earlier pass of this record analysis (same prep, context,
    settings, spec and slow), reused where a run's inputs are unchanged (`response.episode_physics`)."""
    if not removed:
        return geometry, response, energy, episodes
    if kind == FORCED_WAVE:
        geometry.forced_waves = tuple(removed)
    else:
        geometry.induced_ids = frozenset(removed)
    geometry_module.refresh_edges(geometry, angle, settings)
    response, energy = _response_and_energy(series, prep, slow, geometry, context, sampling, settings, angle.well_model)
    episodes = _episode_responses(series, prep, slow, geometry, context, sampling, energy, response, settings, measured)
    return geometry, response, energy, episodes


@dataclass(frozen=True)
class _InducedPass:
    """The second pass of a row whose accepted owner induces some of its episodes but not all (A1, card g04-3): the
    owner of the first pass, the ids of the induced episodes, and what the first pass measured at other scales
    (G1q's well and outcome, the G2c columns of a rejected union, G1f's columns, a G2c analysis the guard holds), which
    the second pass does not measure again (the held analysis is judged against the second pass's decision)."""

    owner: OwnerResult
    ids: frozenset
    qs: object
    qs_outcome: str
    rejected: dict
    subfloor: dict  # the first pass's G1f columns (`subfloor_columns`)
    held: Optional['_HeldCentreScale'] = None  # the first pass's G2c analysis the guard holds (`_held_centre_scale`)


@dataclass(frozen=True)
class _HeldCentreScale:
    """A G2c analysis held while the fine pass decides (LIBRATION_CENTRE_SCALE_OBJECTION_GUARD,
    `decide.centre_scale_displaced`): it carries a measured objection (an objection leaf, origin unknown), so it replaces
    the fine verdict only when the fine pass does not establish the origin as dynamic. `cuts`: the G2a cuts of the
    record when it was held; the fine pass's later cuts go when it is returned (`_return_held`), so the returned
    analysis is the one G2c kept before the guard, its arrays included (held until the fine pass decides)."""

    analysis: MMRAnalysis
    cuts: int


def _episode_tp_counts(prep: Preprocessed, geometry: Geometry, episode_id) -> dict:
    """TPs of the decisive episode at the cutoffs x0.5, x1, x2 (L1), at the thresholds its own turning points were
    selected with (those of its runs of provisional TPs, else of its regime between jumps of a: an episode may span two
    stretches); None where the filtered angle is missing."""
    episode = geometry_module.run_of(geometry, episode_id)
    if episode is None:
        return {}
    window = (episode.t_start_yr, episode.t_end_yr)
    prominence, local = episode.selection_thresholds()
    counts = {'one': episode.n_tp}
    for key, series in (('half', prep.angle_f_half), ('double', prep.angle_f_double)):
        values = np.asarray(series, float)
        counts[key] = (
            len(geometry_module.find_turning_points(prep.t_yr, values, prominence, window, local=local))
            if np.all(np.isfinite(values))
            else None
        )
    return counts


# --------------------------------------------------------------------------------------------- foreign steps


def _oscillates(geometry: Geometry, settings: AnalysisSettings) -> bool:
    scratch = Ledger.from_settings(settings)
    drift = scratch.compare('drift', geometry.drift_fraction, '>=', settings.libration_drift_min_window_fraction, '', '')
    return bool(geometry.episodes) or bool(drift) or len(geometry.turning_points) > 0


def _run_subresonances(
    series, prep, context, sampling, geometry, response, energy, kinematics, angle_decision, settings, scan, cheap=None
) -> Tuple[Optional[SubresonanceResult], Optional[SubresonanceCheap], frozenset]:
    """(the scan, the cheap numbers, the rules fired). The scan runs when the angle has turning points and a trigger
    fires (T5 reads the reason the tree would give without the scan, T6 the leading member of the multiplet by the
    averaging, T7 the cheap theory's centre travel `cheap`); T8 alone runs it as a diagnostic (W28 A6a): its columns go
    into `cheap` (sub_diag_*) and the scan is not returned, so the tree never reads it. Rules: subresonance_t7 when T7
    alone made the scan run, subresonance_t8 when the diagnostic ran, and the scan's own (subresonance_theory_select,
    A6b). The scan reads each accepted episode in its own window; the decisive episode's window gives the row's member
    (`subresonances.scan_windows`)."""
    if not scan or series.resonance.retrograde or not len(geometry.turning_points):
        return None, cheap, frozenset()
    scratch = Ledger.from_settings(settings)
    pending = decide.decide_status(angle_decision, sampling, geometry, energy, response, kinematics, None, None, None, settings, scratch)
    leading, _ = subresonances_module.leading_member(series, context, settings)
    stationary_full = (
        angle_decision.angle_status == AngleStatus.LIBRATION
        and angle_decision.subtype == AngleSubtype.STATIONARY
        and not scratch.thin_margins()
    )
    travel = cheap.travel_ratio if cheap is not None else NAN
    triggered, _ = subresonances_module.triggers(geometry, response, str(pending.reason), leading or 0, stationary_full, settings, travel)
    full = [t for t in triggered if t not in DIAGNOSTIC_TRIGGERS]
    decisive_id = decide.status_basis(angle_decision).episode_id  # A6b: the row's member is the decisive episode's
    if full:
        fired = fire((), Rule.SUBRESONANCE_T7) if full == [SubresonanceTrigger.T7] else frozenset()
        result = subresonances_module.run_subresonances(
            series, prep, geometry, response, context, triggered, settings, decisive_id=decisive_id
        )
        return result, cheap, fired | result.rules_fired
    if not triggered or response_module.law_planet(series.resonance) is not None:
        return None, cheap, frozenset()
    diagnostic = subresonances_module.run_subresonances(
        series, prep, geometry, response, context, triggered, settings, diagnostic=True, decisive_id=decisive_id
    )
    if cheap is not None:
        cheap = replace(cheap, diagnostic=subresonances_module.diagnostic_columns(diagnostic))
    return None, cheap, fire((), Rule.SUBRESONANCE_T8)


def _cheap_subresonances(series, prep, geometry, context, settings, scan) -> Optional[SubresonanceCheap]:
    """The cheap theory numbers and candidate list (W28 A6a, card g06-8a A) of a first-pass prograde non-co-orbital row
    (`scan`: not a derived or owner row); None otherwise."""
    if not scan:
        return None
    cheap, _ = subresonances_module.cheap_subresonances(series, prep, geometry, context, settings)
    return cheap


def _owner(
    series, prep, context, geometry, response, strength, angle_decision, settings, cells, scan_owner, ledger
) -> Optional[OwnerResult]:
    """The owner search (SPEC §4.8), its routes in order: the search over the commensurabilities (`_run_owner`), then
    the route 'identity' on a three-body angle without an accepted owner (`_identity_owner`). None for a row without
    one: not asked for (`scan_owner`, a derived row or phi's own analysis), a retrograde resonance, or
    mmr_owner_search off. Each route records its own comparisons (`owner.record_decisive`)."""
    if not scan_owner or series.resonance.retrograde or settings.mmr_owner_search == 'off':
        return None
    owner = _run_owner(series, prep, context, geometry, response, angle_decision, settings)
    found = _identity_owner(series, prep, context, geometry, owner, strength, angle_decision, settings, cells, ledger)
    if found is not None and found is not owner:  # the route 'identity': phi's intervals on every episode (F7, A1)
        basis = decide.status_basis(angle_decision)
        _, period = decisive_window(basis, geometry, prep)
        owner_module.cover_episodes(found, episode_windows(basis, geometry, prep, settings), period)
    return found


def _run_owner(series, prep, context, geometry, response, angle_decision, settings) -> Optional[OwnerResult]:
    """The owner search when it is required (an oscillation, or a force law that cannot be measured) or the angle
    circulates (not an alias): the owner of a circulation goes to the origin columns, the status stays."""
    basis = decide.status_basis(angle_decision)  # the angle the tree reads (G1c -> episodic)
    if not owner_module.owner_search_matters(basis.angle_status, basis.subtype):
        return None  # alias, undefined angle or too little time: the tree never reads the owner (W3-M2)
    amplitude = max((e.half_amp_max_rad for e in geometry.episodes), default=decide.tp_half_amplitude_median(geometry))
    required = owner_module.owner_required(str(response.a_response), amplitude, _oscillates(geometry, settings), settings)
    circulates = angle_decision.angle_status == AngleStatus.CIRCULATION and angle_decision.subtype != AngleSubtype.ALIAS
    if not (required or circulates):
        return None
    window, period = decisive_window(basis, geometry, prep)
    return owner_module.run_owner(
        series,
        prep,
        context,
        geometry.period_cycle_yr,
        settings,
        decisive_window=window,
        decisive_period_yr=period,
        episode_windows=episode_windows(basis, geometry, prep, settings),
    )


def _identity_owner(series, prep, context, geometry, owner, strength, angle_decision, settings, cells, ledger):
    """The owner route 'identity' (`owner_identity`, W22 R-OWNER-ID) on a librating three-body angle whose period needs
    >= mmr_owner_identity_ratio_min x mu_i mu_j e^q and that has no accepted owner: the owner result with phi accepted
    (its comparisons on `ledger`, `owner_identity.record`), else `owner` unchanged. Each shortlisted phi gets its own
    analysis without scans (no recursion)."""
    if not owner_identity.longitudes_saved(series, settings):
        return owner
    basis = decide.status_basis(angle_decision)
    if basis.angle_status not in decide.LIBRATING:
        return owner
    run = geometry_module.run_of(geometry, basis.episode_id)
    if not owner_identity.route_applies(strength, owner, run, settings):
        return owner

    def analyse(phi: MMRSeries) -> MMRAnalysis:
        return analyse_mmr(phi, settings, cells, scan_subresonances=False, scan_owner=False)

    found = owner_identity.search(series, prep, context, geometry, run, owner, settings, analyse)
    if found is None:
        return owner
    fit, phi, route = found
    owner_identity.record(ledger, strength, fit, phi, settings)
    return owner_identity.identity_owner(owner, series, fit, phi, route, run)


def decisive_window(basis, geometry, prep) -> Tuple[Tuple[float, float], float]:
    """((t0, t1), period [yr]) of the episode the status reads (`decide.status_basis`): its run's first and last turning
    point and its median period; for a libration over the record, or without an episode, the analysis window and the
    cycle period (F7: the owner must act there)."""
    run = geometry_module.run_of(geometry, basis.episode_id) if basis.angle_status != AngleStatus.LIBRATION else None
    if run is not None and len(run.tp_times_yr) >= 2:
        return (float(run.tp_times_yr[0]), float(run.tp_times_yr[-1])), float(run.period_med_yr)
    return (float(prep.window[0]), float(prep.window[1])), float(geometry.period_cycle_yr)


def episode_windows(basis, geometry, prep, settings) -> List[Tuple[Optional[int], float, float, float]]:
    """(episode id, t0, t1, period [yr]) of every accepted episode of sigma the owner is compared with (F7 per episode,
    A1): the librating episodes of the last regime and before its jump (`decide.librating_episodes`), from the first
    to the last turning point, with their median period, the decisive one included; for a libration over the record,
    or without such an episode, the decisive window alone (id None, `decisive_window`)."""
    window, period = decisive_window(basis, geometry, prep)
    if basis.angle_status == AngleStatus.LIBRATION:
        return [(None, window[0], window[1], period)]
    runs = decide.librating_episodes(geometry, settings) + decide.librating_episodes(geometry, settings, before_jump=True)
    decisive = geometry_module.run_of(geometry, basis.episode_id)
    if decisive is not None and all(run.episode_id != decisive.episode_id for run in runs):
        runs.append(decisive)
    runs = sorted((run for run in runs if len(run.tp_times_yr) >= 2), key=lambda run: run.tp_times_yr[0])
    if not runs:
        return [(None, window[0], window[1], period)]
    return [(run.episode_id, float(run.tp_times_yr[0]), float(run.tp_times_yr[-1]), float(run.period_med_yr)) for run in runs]


def _second_pass_owner(induced: _InducedPass, basis, geometry, prep) -> OwnerResult:
    """The first pass's owner for the second pass (A1): the induced episodes marked (`OwnerResult.induced_ids`), its
    cover and P2 read again on the episode the second pass decides on (an own episode)."""
    window, period = decisive_window(basis, geometry, prep)
    owner = replace(induced.owner, induced_ids=induced.ids, decisive_window=window)
    owner.cover = owner_module.covered_fraction(owner.intervals, window)
    owner.psi_turns_per_cycle = owner_module.psi_turns(owner, period)
    return owner


def _owner_per_episode(
    series, prep, context, geometry, response, strength, angle_decision, settings, cells, scan_owner, ledger, induced, scales
) -> Tuple[Optional[OwnerResult], Optional[_InducedPass]]:
    """(owner, second pass): the owner search (`_owner`) and, when its owner induces some episodes but not all
    (`_mixed_owner`), the second pass to run instead (A1; `scales` = the first pass's qs, G1q outcome, G2c and G1f
    columns);
    in a second pass (`induced`) the first pass's owner read on the episode it decides on (`_second_pass_owner`)."""
    if induced is not None:
        return _second_pass_owner(induced, decide.status_basis(angle_decision), geometry, prep), None
    owner = _owner(series, prep, context, geometry, response, strength, angle_decision, settings, cells, scan_owner, ledger)
    mixed = _mixed_owner(owner, settings)
    return owner, (_InducedPass(owner, mixed, *scales) if mixed else None)


def _mixed_owner(owner: Optional[OwnerResult], settings: AnalysisSettings) -> frozenset:
    """The ids of the episodes an accepted owner induces when others are the row's own (card g04-3: the own episodes
    decide), empty otherwise (all induced: rule 2; none: F7 outside the episode)."""
    induced, own = owner_module.induced_split(owner, settings)
    ids = frozenset(entry.episode_id for entry in induced if entry.episode_id is not None)
    return ids if ids and own else frozenset()


def _derived(series: MMRSeries, subresonances: Optional[SubresonanceResult], settings, cells) -> List[MMRAnalysis]:
    """The accepted planar subresonance analysed as its own row (depth 1)."""
    if subresonances is None or subresonances.accepted is None or subresonances.accepted.nodal:
        return []
    member_series = subresonances_module.member_series(series, subresonances.accepted.spec)
    derived = analyse_mmr(member_series, settings, cells, scan_subresonances=False, scan_owner=False, parent=series.resonance.key)
    derived.derived_series = member_series
    return [derived]


# --------------------------------------------------------------------------------------------- the analysis


def _undefined(series, context, sampling, settings, cells, parent: str) -> MMRAnalysis:
    """G0: a failed unwrap: the verdict from the sampling alone. The orbit is physical here: one outside the elliptic
    domain never reaches G0 (`MMRSeries.domain_problems` refuses it first, `_invalid_input`), so `decide_angle` gets
    unphysical = False and its G0 line reads 0."""
    ledger = Ledger.from_settings(settings)
    angle = decide.decide_angle(sampling, False, None, None, settings, ledger)
    decision = decide.decide_status(angle, sampling, None, None, None, None, None, None, None, settings, ledger)
    action, resample = _resample_action(default_action(decision.status, decision.reason, Confidence.NONE), sampling)
    verdict = Verdict(
        resonance_type=ResonanceType.MMR,
        status=decision.status,
        angle_status=angle.angle_status,
        angle_subtype=angle.subtype,
        reason=decision.reason,
        flags=list(dict.fromkeys(list(sampling.flags) + decision.flags)),
        action=action,
        resample_step_yr=resample,
        checks=decide.mmr_checks(decide.tree_view(angle, None, None), sampling, None, None, None, None, None, None, None),
        decision_path=list(ledger),
    )
    return _verdict_only(
        series,
        verdict,
        settings,
        cells,
        context=context,
        sampling=sampling,
        parent=parent,
        extra_columns={'rec_unphysical': 0.0},
        filter_order=settings.filter_order,
    )


def _g0(series, context, sampling, settings, cells, parent: str) -> Optional[MMRAnalysis]:
    """G0, the verdicts before any measurement: a failed unwrap (`_undefined`), a step too coarse for the angle's filter
    (`_undersampled`); None when the record goes on to the measurements. An unphysical orbit never gets here: the
    package rule (`classify.util.unphysical_arrays`: non-finite a or e, e > 1.3, a < 0) and non-finite angles or times
    are all inside `MMRSeries.domain_problems` (non-finite, e >= 1, a <= 0), checked first (`_invalid_input`)."""
    if not sampling.unwrap_ok:
        return _undefined(series, context, sampling, settings, cells, parent)
    cutoff, floor, _ = preprocess_module.angle_cutoff(series.resonance, context, sampling.step_yr, settings)
    if not preprocess_module.filter_realizable(cutoff.cutoff_yr, sampling.step_yr):
        return _undersampled(series, context, sampling, settings, cells, parent, cutoff, floor)
    return None


def _undersampled(series, context, sampling, settings, cells, parent: str, cutoff, floor: float) -> MMRAnalysis:
    """G0, the step is too coarse for the angle's filter: P_c <= 2 dt (`preprocess.filter_realizable`), so no low-pass
    of the record keeps its shortest libration (a Jupiter Trojan, P_c = 36.9 yr, at dt >= 18.5 yr; any other MMR at
    dt >= 37.5 yr when the floor bound sets P_c). Invalid / undersampled with every check not_measured (check_mmr_filter
    fails) and resample at the largest round step <= P_floor / mmr_resample_steps_per_cycle that also meets the alias
    bounds and filters P_c (`sampling.resample_step`; Jupiter Trojans 2.5 yr); review when none fits. It comes after
    the unwrap check (`_g0`): a record failing that as well is reported by it."""
    step = float(sampling.step_yr)
    ledger = Ledger.from_settings(settings)
    ledger.compare('G0 sampling: the cutoff is longer than two output steps', cutoff.cutoff_yr, Op.GT, 2 * step, 'yr', 'flt_cutoff_yr')
    resample = sampling_module.resample_step(
        sampling.safe_step_plain_yr, sampling.safe_step_pred_yr, floor, step, settings, cutoff.cutoff_yr
    )
    verdict = Verdict(
        resonance_type=ResonanceType.MMR,
        status=Status.INVALID,
        angle_status=AngleStatus.UNDEFINED,
        angle_subtype=AngleSubtype.NONE,
        reason=Reason.UNDERSAMPLED,
        flags=list(sampling.flags),
        action=Action.RESAMPLE if resample is not None else Action.REVIEW,
        resample_step_yr=resample,
        checks={**unmeasured_checks(ResonanceType.MMR), 'check_mmr_filter': CheckValue.FAIL},
        decision_path=list(ledger),
    )
    columns = {
        'rec_unphysical': 0.0,
        'rec_rule': 'G0',
        'geo_angle_rule': 'G0',
        'flt_cutoff_yr': cutoff.cutoff_yr,
        'flt_cutoff_binding': cutoff.binding,
        'flt_samples_per_cutoff': cutoff.samples_per_cutoff,
        'flt_period_floor_yr': floor,
    }
    return _verdict_only(
        series,
        verdict,
        settings,
        cells,
        context=context,
        sampling=sampling,
        parent=parent,
        extra_columns=columns,
        filter_order=settings.filter_order,
    )


def _verdict_only(series: MMRSeries, verdict: Verdict, settings, cells, **fields) -> MMRAnalysis:
    """The analysis of a record stopped before the measurements: the verdict and what `fields` give (context, sampling,
    parent, columns); every measurement None, no rejected or displaced G2c analysis (geo_cs_rej_*, geo_cs_displaced_*
    empty)."""
    fields['extra_columns'] = {**CS_REJECTED_NONE, **CS_DISPLACED_NONE, **fields.get('extra_columns', {})}
    verdict.commensurability = roles_module.commensurability(series.resonance)
    return MMRAnalysis(
        series_name=series.name,
        resonance=series.resonance,
        episode_physics=[],
        verdict=verdict,
        settings_hash=settings_hash(settings),
        confidence_cells_sha=cells.sha,
        **{**dict.fromkeys(MEASUREMENTS), **fields},
    )


def _length_limited_action(
    action: Action,
    extend: Optional[float],
    decision: decide.StatusDecision,
    view: decide.TreeView,
    geometry,
    response,
    sampling: Sampling,
    prep,
    settings: AnalysisSettings,
) -> Tuple[Action, Optional[float]]:
    """W28 B9 (cards g08-5 B, g08-6 A, g06-6 A, g06-7 A): an uncertain row left at review whose deciding not-measured
    check is length-limited (`decide.length_limited_extension`) asks for extend with that length (rule
    extend_length_limited). Any other action (resample, the F8 review, the quasi-satellite stall) is kept."""
    if action != Action.REVIEW:
        return action, extend
    basis = decide.status_basis(view.angle)
    length = decide.length_limited_extension(decision, basis, geometry, response, sampling.span_yr, prep.cutoff.cutoff_yr, settings)
    if length is None:
        return action, extend
    decision.rules_fired = fire(decision.rules_fired, Rule.EXTEND_LENGTH_LIMITED)
    return Action.EXTEND, length


def _resample_action(action: Action, sampling: Sampling) -> Tuple[Action, Optional[float]]:
    """(action, resample step): resample asks for `sampling.resample_step_yr` (finer than the record's step); without one
    (the step already meets the safe bounds, so a finer one cures nothing: 354652 53J-20 at 0.6 yr) a person looks."""
    if action != Action.RESAMPLE:
        return action, None
    if sampling.resample_step_yr is None:
        return Action.REVIEW, None
    return action, sampling.resample_step_yr


def _invalid_input(series: MMRSeries, problems: List[str], settings, cells, parent: str) -> MMRAnalysis:
    """The verdict of a record outside the MMR model (`MMRSeries.domain_problems`): invalid, every check not_measured.
    An orbit outside the elliptic domain (a <= 0, e outside [0, 1), NaN) is `unphysical`; an angle without the body's
    mean longitude (k = 0) is `invalid_input`. The problems are in rec_input_problems and on the decision path."""
    ledger = Ledger.from_settings(settings)
    orbit = [problem for problem in problems if problem != 'k=0']
    ledger.compare(
        'G0 input: the angle holds the mean longitude (k != 0)', series.resonance.body_coefficient, Op.NE, 0, '', 'rec_body_coefficient'
    )
    ledger.compare('G0 orbit is physical (unphysical = 0)', float(bool(orbit)), Op.EQ, 0, '', 'rec_unphysical')
    reason = Reason.UNPHYSICAL if orbit else Reason.INVALID_INPUT
    verdict = Verdict(
        resonance_type=ResonanceType.MMR,
        status=Status.INVALID,
        angle_status=AngleStatus.UNDEFINED,
        angle_subtype=AngleSubtype.UNPHYSICAL if orbit else AngleSubtype.NONE,
        reason=reason,
        action=default_action(Status.INVALID, reason, Confidence.NONE),
        checks=unmeasured_checks(ResonanceType.MMR),
        decision_path=list(ledger),
    )
    return _verdict_only(
        series,
        verdict,
        settings,
        cells,
        parent=parent,
        extra_columns={
            'rec_unphysical': float(bool(orbit)),
            'rec_body_coefficient': series.resonance.body_coefficient,
            'rec_input_problems': join_list(problems),
            'rec_rule': 'G0',
            'geo_angle_rule': 'G0',
        },
    )


def analyse_mmr(
    series: MMRSeries,
    settings: AnalysisSettings,
    cells: Optional[ConfidenceCells] = None,
    *,
    scan_subresonances: bool = True,
    scan_owner: bool = True,
    parent: str = '',
    centre_scale: bool = True,
    quasi_satellite: Optional[quasi_satellite_module.QuasiSatellite] = None,
    subfloor: Optional[SubfloorPass] = None,
) -> MMRAnalysis:
    """Analyse one angle of one body from its saved series; never integrates. `cells` default: the configured file.

    `parent` marks a derived row (an accepted subresonance of that catalogue angle). A malformed record (lengths,
    times, planet grids: `MMRSeries.validate`) raises ValueError; a well-formed record outside the MMR model
    (`MMRSeries.domain_problems`) gets an invalid verdict. `centre_scale=False` is the analysis at the scale of the centre
    itself (G2c, no second level); `quasi_satellite` the second pass of a 1:1 angle at its quasi-satellite floor
    (G1q, no second level either); `subfloor` the second pass of a two-body angle at the floor of its own averaged
    potential (G1f, `_subfloor_pass`, no second level). The co-orbital tables of the averaged potential are kept for the whole analysis, up
    to mmr_coorbital_record_tables_mb of their arrays (`coorbital.record_tables`), and the filtered series the passes ask
    for again in the record's memo (`filtering.record_memo`).

    W29: the outermost call ends with the policy layer (`policy.apply_mmr`: the minimal policy re-reads the finished
    analysis); a pass inside a record's analysis (a derived row,
    the centre scale, an owner's identity) returns the raw v2 analysis, which the outermost call's layer reads. After
    the policy only the 1:1 coverage audit runs (`coverage_audit`, flags and aud_* columns, never a status)."""
    options = (scan_subresonances, scan_owner, parent, centre_scale, quasi_satellite, subfloor)
    if _ACTION_STEP_CUTS.get() is not None:  # a pass inside a record's analysis: no policy, the outermost call applies it
        return _analyse_mmr_record(series, settings, cells, *options)
    analysis = policy_module.apply_mmr(_analyse_mmr_record(series, settings, cells, *options), settings)
    _coverage_audit(series, analysis, settings)
    return analysis


def _coverage_audit(series: MMRSeries, analysis: MMRAnalysis, settings: AnalysisSettings) -> None:
    """The 1:1 coverage audit after the policy (`coverage_audit`; prograde 1:1 only, MMR_COORBITAL_COVERAGE_AUDIT):
    flags floor_above_own_loop (Layer I) and uncovered_bounded_run (Layer II) into the verdict's flags (they enter
    `review_reasons`), the aud_* columns into mmr_numbers.csv. Changes no status, level, cap or action."""
    analysis.extra_columns.update(coverage_audit_module.empty_columns())
    geometry, prep, context = analysis.geometry, analysis.prep, analysis.context
    if not settings.mmr_coorbital_coverage_audit or preprocess_module.coorbital_planet(series.resonance) is None:
        return
    if geometry is None or prep is None or context is None or analysis.verdict.status == Status.INVALID:
        return
    reader = _AuditReader(series, analysis, settings)
    columns = analysis.extra_columns
    found = coverage_audit_module.audit(
        prep.t_yr,
        prep.angle_unwrapped,
        context.orbital_period_yr,
        tuple(prep.window),
        reader.counted_spans(),
        settings,
        reader.well_of,
        reader.law_of,
        reader.overlap_of,
    )
    columns.update(found.columns)
    contradiction = coverage_audit_module.floor_contradiction(reader.accepted_floors(), columns)
    flags = [found.flag] if found.flag else []
    if contradiction is not None:
        columns.update(aud_floor_above_loop=contradiction.where, aud_floor_yr=contradiction.floor_yr, aud_loop_yr=contradiction.loop_yr)
        flags.append(coverage_audit_module.FLAG_FLOOR_ABOVE_LOOP)
    if flags:
        analysis.verdict.flags = list(dict.fromkeys(list(analysis.verdict.flags) + flags))


class _AuditReader:
    """What the coverage audit reads of a finished 1:1 analysis: the counted spans, the well, the local law and the
    classifier's reason for a stretch, and the floors of the accepted runs (`_coverage_audit`). Reporting only."""

    def __init__(self, series: MMRSeries, analysis: MMRAnalysis, settings: AnalysisSettings):
        self.series, self.analysis, self.settings = series, analysis, settings
        self.geometry, self.prep, self.context = analysis.geometry, analysis.prep, analysis.context
        self.eligibility = decide.episode_eligibility(self.geometry, settings)

    def counted_spans(self) -> List[Tuple[float, float]]:
        """The policy's counted episodes and spans [yr] (`PolicyResult`)."""
        policy = self.analysis.policy
        if policy is None:
            return []
        counted = set(policy.counted_episode_ids)
        spans = [(r.t_start_yr, r.t_end_yr) for r in self.geometry.runs if r.episode_id in counted]
        return spans + [tuple(span) for span in policy.counted_spans_yr]

    def well_of(self, stretch) -> str:
        """The well of the averaged potential that holds the stretch's swing (`coorbital_geometry.well_of`)."""
        profile = coorbital_geometry.series_profile(self.series, self.context, stretch.t0_yr, stretch.t1_yr, self.settings)
        if profile is None:
            return ''
        centre = float(coorbital_geometry.theta_of(np.array([stretch.centre_rad]), self.series.resonance.body_coefficient)[0])
        return coorbital_geometry.well_of(profile, float(np.mod(centre, 2 * np.pi)), stretch.half_amp_rad).name

    def law_of(self, stretch) -> Tuple[float, float, str]:
        """(K / K_theory, R^2, force check) of the 1:1 law on the stretch +- one half-cycle, at the cutoff a floor of
        its own period sets (factor x P / FILTER_MIN_LIBRATION_CUTOFFS: the filter keeps the stretch's swing). The check
        is the law's own `coorbital.force_check` (pass, thin, fail, not_measured): Layer II flags only a stretch it
        passes, not thin (`coverage_audit.audit`)."""
        s = self.settings
        record = coorbital_geometry.series_input(self.series, self.context, self.prep.angle_clean)
        if record is None:
            return NAN, NAN, coverage_audit_module.LAW_NOT_MEASURED
        cutoff = min(self.prep.cutoff.cutoff_yr, s.mmr_coorbital_period_floor_factor * stretch.period_yr / s.filter_min_libration_cutoffs)
        half = stretch.period_yr / 2
        law = coorbital_geometry.local_law(record, (stretch.t0_yr - half, stretch.t1_yr + half), cutoff, s)
        if law is None:
            return NAN, NAN, coverage_audit_module.LAW_NOT_MEASURED
        check = coverage_audit_module.LAW_THIN if law.force_check == CheckValue.PASS and law.force_thin else str(law.force_check)
        return float(law.scale), float(law.r2), check

    def overlap_of(self, stretch) -> str:
        """Why the classifier did not count the stretch: the reject or ineligible reason of the run that overlaps it
        most, else why the G1q pass was rejected, else no_run."""

        def overlap(run) -> float:
            return min(run.t_end_yr, stretch.t1_yr) - max(run.t_start_yr, stretch.t0_yr)

        runs = [r for r in self.geometry.runs if len(r.tp_times_yr) >= 2 and overlap(r) > 0]
        columns = self.analysis.extra_columns
        if runs:
            run = max(runs, key=overlap)
            if not run.accepted:
                return str(run.reject_reason)
            return str(self.eligibility.get(run.episode_id, '')) or coverage_audit_module.OVERLAP_NOT_COUNTED
        if str(columns.get('geo_qs_pass', '') or '') == quasi_satellite_module.PASS_REJECTED:
            return f"{coverage_audit_module.OVERLAP_QS_REJECTED}:{columns.get('geo_qs_reject', '')}"
        return coverage_audit_module.OVERLAP_NO_RUN

    def accepted_floors(self) -> List[Tuple[object, str, float, float]]:
        """(episode_id, ineligible_reason, floor, loop) of every accepted run with a floor reading: the loop is the
        median over its turn pairs, or for a run without a loop floor of its own (not_reflection: mixed sites, or QS
        turns outside the G1q pass) the loop of its first turn pair (`CoorbitalRunFloor.read`, cost; 2015 YA without the
        rule: 66.1 yr against P 69.3 yr)."""
        out = []
        for run in self.geometry.runs:
            reading = self.geometry.run_floor(run)
            if run.accepted and reading is not None:
                loop = float(dict(reading.columns).get('cpl_loop_period_yr', NAN))
                floor = decide.period_floor(self.geometry, self.settings, run)[0]
                out.append((run.episode_id, str(self.eligibility.get(run.episode_id, '')), floor, loop))
        return out


def _analyse_mmr_record(
    series: MMRSeries,
    settings: AnalysisSettings,
    cells: Optional[ConfidenceCells],
    scan_subresonances: bool,
    scan_owner: bool,
    parent: str,
    centre_scale: bool,
    quasi_satellite: Optional[quasi_satellite_module.QuasiSatellite],
    subfloor: Optional[SubfloorPass],
) -> MMRAnalysis:
    """The v2 analysis of one record (`analyse_mmr` without the policy layer), with the G2a guard, which chooses between
    the cut and the whole record on the raw v2 (status, confidence)."""
    options = (scan_subresonances, scan_owner, parent, centre_scale, quasi_satellite, subfloor)
    with coorbital_module.record_tables(settings.mmr_coorbital_record_tables_mb * BYTES_PER_MB), filtering.record_memo():
        row = _ROW_KEY.set(series.resonance.key)
        try:
            if _ACTION_STEP_CUTS.get() is not None:  # a pass inside a record's analysis: the outermost call guards its cuts
                return _analyse_mmr(series, settings, cells, *options)
            token = _ACTION_STEP_CUTS.set([])
            try:
                analysis = _analyse_mmr(series, settings, cells, *options)
                cuts = list(_ACTION_STEP_CUTS.get())
            finally:
                _ACTION_STEP_CUTS.reset(token)
        finally:
            _ROW_KEY.reset(row)
        if not cuts:
            return analysis
        whole = _analyse_mmr_record(
            series,
            replace(settings, libration_action_step=False),
            cells,
            scan_subresonances,
            scan_owner,
            parent,
            centre_scale,
            quasi_satellite,
            subfloor,
        )
        return _action_step_guard(analysis, whole, cuts, settings)


def _decide_angle(series, context, sampling, prep, geometry, energy, settings, ledger):
    """(the angle decision G0-G6, the G5w numbers): G5w (B6c) first measures whether an unfinished slow swing (two
    turns of the slow centre) is a swing in the resonance's own well (`slow_swing.measure_slow_swing`); the rule
    `slow_swing_in_well` fires on the geometry when it decides the angle."""
    swing = slow_swing_module.measure_slow_swing(series, context, prep, geometry, settings)
    angle_decision = decide.decide_angle(sampling, False, geometry, energy, settings, ledger, swing)
    if angle_decision.rule == decide.RULE_G5W:
        geometry.rules_fired = fire(geometry.rules_fired, Rule.SLOW_SWING_IN_WELL)
    return angle_decision, swing


def hidden_failures(cut_checks, whole_checks) -> List[str]:
    """The `CUT_GUARDED_CHECKS` that failed on the whole record and are neither passed nor failed after a G2a cut."""
    measured = (CheckValue.PASS, CheckValue.FAIL)
    return [name for name in CUT_GUARDED_CHECKS if whole_checks.get(name) == CheckValue.FAIL and cut_checks.get(name) not in measured]


def _action_step_guard(cut: MMRAnalysis, whole: MMRAnalysis, steps: list, settings: AnalysisSettings) -> MMRAnalysis:
    """G2a guard: `cut` (the analysis with the G2a `steps` it cut) unless one of its rows that cut a step of its own
    (`ActionStep.key`: the catalogue angle or a derived member; a row never answers for another row's step) hides a
    failed measurement (`hidden_failures`, R0 included) and has another status or confidence than the same row of
    `whole`: then `whole` (libration_action_step off), each of its rows carrying its own steps of the cut analysis,
    those that cut reported as withheld (`geometry.STEP_WITHHELD`, the row's hidden checks and its verdict of the cut in
    geo_action_step_withheld_*). A cut whose regimes measure what the whole record failed (pass or fail) stays; so does
    one that changes no verdict."""
    wholes = {row.resonance.key: row for row in [whole] + whole.derived}
    own: dict = {}
    for step in steps:
        own.setdefault(step.key, []).append(step)
    hidden, verdicts = {}, {}
    for row in [cut] + cut.derived:
        key = row.resonance.key
        other = wholes.get(key)
        if other is None or key not in own:
            continue
        names = hidden_failures(row.verdict.checks, other.verdict.checks)
        mine, theirs = row.verdict, other.verdict
        verdicts[key] = f'{mine.status}/{mine.confidence}' if mine.confidence else str(mine.status)
        if names and (mine.status, mine.confidence) != (theirs.status, theirs.confidence):
            hidden[key] = names
    if not hidden:
        return cut
    cuts = {row.resonance.key: row for row in [cut] + cut.derived}
    for row in [whole] + whole.derived:
        key = row.resonance.key
        source = cuts.get(key)
        final = source.geometry.action_steps if source is not None and source.geometry is not None else []
        reported = final if any(step.cuts for step in final) else own.get(key, [])  # else the cut of a pass inside it
        marks = {'failed': geometry_module.STEP_WITHHELD, 'provisional': False, 'withheld_checks': join_list(hidden.get(key, []))}
        withheld = [replace(step, **marks, withheld_verdict=verdicts.get(key, '')) if step.cuts else step for step in reported]
        row.settings_hash = settings_hash(settings)
        if row.geometry is not None:
            row.geometry.action_steps = withheld
    names = sorted({name for names in hidden.values() for name in names})
    ledger = Ledger.from_settings(settings)
    entry = ledger.compare(
        'G2a cut withheld: measurements failed on the whole record, unmeasured after the cut',
        len(names),
        Op.EQ,
        0,
        '',
        'geo_action_step_n_hidden',
    )
    whole.verdict.decision_path = list(whole.verdict.decision_path) + [entry]
    return whole


def _analyse_mmr(
    series: MMRSeries,
    settings: AnalysisSettings,
    cells: Optional[ConfidenceCells],
    scan_subresonances: bool,
    scan_owner: bool,
    parent: str,
    centre_scale: bool,
    quasi_satellite: Optional[quasi_satellite_module.QuasiSatellite],
    subfloor: Optional[SubfloorPass] = None,
    induced: Optional[_InducedPass] = None,
) -> MMRAnalysis:
    """`analyse_mmr` inside the record's table scope; `induced`: the second pass of a row whose owner induces some of
    its episodes but not all (A1: those episodes out of the geometry, the first pass's owner, no other scales). A G2c
    analysis with a measured objection is held, not returned, until the fine pass has decided: it is returned unless the
    fine origin is dynamic (the G2c guard, `_held_centre_scale`, `decide.centre_scale_displaced`)."""
    series.validate()
    cells = cells if cells is not None else confidence_module.cells_from_settings(settings)
    problems = series.domain_problems()
    if problems:
        return _invalid_input(series, problems, settings, cells, parent)
    context = context_module.build_context(series, settings)
    sampling = sampling_module.analyse_sampling(series, context, settings)
    floor, block, floor_well = quasi_satellite_module.pass_inputs(quasi_satellite)
    flag_folded_harmonic(series.resonance, context, sampling, settings, floor)  # the flag survives G0 (the cap stays below)
    stopped = _g0(series, context, sampling, settings, cells, parent)
    if stopped is not None:
        return stopped

    # G1f: the pass at the floor of the angle's own averaged potential (the flag above keeps the angle's own floor)
    prep = preprocess_module.preprocess(series, context, sampling, settings, subfloor.period_floor if subfloor else floor, block)
    run_floor = _run_floor_model(series, context, prep, settings, floor_well)
    part, fast = _pass_fast(prep, settings, run_floor)
    prominence = _prominence(prep, settings, fast)
    angle, geometry, slow, response, energy = _measure(series, context, sampling, prep, prominence, settings, floor_well, run_floor, part)
    del part
    period = geometry.period_cycle_yr
    sampling.aliased_lines_yr = sampling_module.aliased_lines(
        context.synodic_periods_yr, sampling.step_yr, period, settings.mmr_aliased_line_harmonics, settings.mmr_aliased_line_band_factor
    )
    sampling.resample_step_yr = sampling_module.resample_step(
        sampling.safe_step_plain_yr, sampling.safe_step_pred_yr, period, sampling.step_yr, settings
    )
    measured = {}  # the runs' force law, identity and B1 of this pass, reused by the passes without some runs
    episodes = _episode_responses(series, prep, slow, geometry, context, sampling, energy, response, settings, measured)
    geometry, response, energy, episodes = _without_forced_waves(
        series, prep, slow, angle, geometry, context, sampling, energy, response, episodes, settings, measured
    )
    # A1: in a second pass the episodes the owner induces are no episodes of the angle
    args = (series, prep, slow, angle, geometry, context, sampling, energy, response, episodes, settings)
    geometry, response, energy, episodes = _without_episodes(*args, INDUCED, induced.ids if induced is not None else (), measured)
    del measured
    # a 1:1 angle's R0 from the law the tree reads (rule R: the witness's law carries its energy integral too)
    energy = response_module.with_coorbital_potential(
        energy, response_module.deciding_response(response, False), geometry, prep.filter_too_slow
    )
    kinematics = kinematics_module.analyse_kinematics(series, prep, slow, geometry, context, response, settings)

    ledger = Ledger.from_settings(settings)
    angle_decision, slow_swing = _decide_angle(series, context, sampling, prep, geometry, energy, settings, ledger)
    options = (scan_subresonances, scan_owner, parent, centre_scale, quasi_satellite, subfloor)
    args = (series, context, sampling, prep, angle, geometry, prominence, angle_decision, ledger, settings, cells, options)
    seen = []  # the slow tracks of G2c's centre-scale geometries, for the slow scale (B10a)
    qs, qs_outcome, other_scale, rejected, subfloor_found = _other_scales(*args, induced=induced, seen=seen)
    # the G2c guard: a center-scale reading with a measured objection waits for the fine decision (`_held_centre_scale`)
    held = induced.held if induced is not None else _held_centre_scale(other_scale, settings)
    if other_scale is not None and held is None:
        return other_scale
    del other_scale
    union_word = bool(rejected)  # G2c found a union of episodes read as episodic (`_at_centre_scale`)
    decisive_id = decide.status_basis(angle_decision).episode_id  # G1e: the episode before the last jump of a
    # the short-period scatter of a over the decisive a half-excursion, once: R-ALIAS-A and the flag read it
    decisive_physics = physics_of(episodes, decisive_id)
    excursion = decisive_physics.a_excursion_meas_au if decisive_physics is not None else NAN
    sampling.a_scatter_ratio = decide.a_scatter_ratio(sampling, excursion)
    # the filter guard before the tree (W21 Q9, principle b): the last regime's own prominence, the decisive episode
    # against its counterpart at 2 P_c; a passed check it withdraws is not_measured for the tree (`tree_energy`)
    last_prominence = geometry.turning_points.prominence_rad
    # pattern B's stall is compared only where G3 can read it: a record without librating episodes
    compare_stall = not decide.librating_episodes(geometry, settings)
    sensitivity = preprocess_module.filter_sensitivity(prep, angle, geometry, energy, last_prominence, settings, decisive_id, compare_stall)
    if sensitivity.sensitive and str(Flag.FILTER_SENSITIVE) not in prep.flags:
        prep.flags.append(str(Flag.FILTER_SENSITIVE))
    tree_energy, withdrawn = decide.energy_for_tree(energy, sensitivity, ledger)
    cheap = _cheap_subresonances(series, prep, geometry, context, settings, scan_subresonances)
    subresonances, cheap, subresonance_rules = _run_subresonances(
        series, prep, context, sampling, geometry, response, tree_energy, kinematics, angle_decision, settings, scan_subresonances, cheap
    )
    basis = decide.status_basis(angle_decision)
    _decisive_kinematics(series, prep, context, geometry, response, kinematics, basis, settings)
    episodes = sorted(episodes, key=lambda p: p.episode_id != decisive_id)  # the decisive episode first (strength)
    strength = strength_module.analyse_strength(context, episodes, settings, series, geometry, prep.cutoff.cutoff_yr)
    # W28 B8: the direct averaged-force law where L4 fails or is thin (lifts L4 only); F then reads the large turns
    response = response_module.with_direct_force(response, series, prep, geometry, context, strength, settings, sampling)
    # W28 B8b: the lattice lines that carry a_dot on the decisive window (columns rsp_line_*, no status reads them)
    response = response_module.with_line_share(response, series, prep, context, *decisive_window(basis, geometry, prep), settings)
    strength = strength_module.at_large_period(strength, decide.taken_direct_law(response, angle_decision), settings)
    args = (series, prep, context, geometry, response, strength, angle_decision, settings, cells, scan_owner, ledger)
    owner, again = _owner_per_episode(*args, induced, (qs, qs_outcome, rejected, subfloor_found, held))
    if again is not None:  # A1, card g04-3: some episodes induced, others the row's own -> the own episodes decide
        return _analyse_mmr(series, settings, cells, *options, induced=again)
    derived = _derived(series, subresonances, settings, cells)  # before the status: the catalogue row may take its status
    member = derived[0].verdict if derived else None
    # B10a: the slower libration of the centre at its own scale (K3s reads it), then G2c's arrays go
    slow_scale = _slow_scale(series, prep, geometry, kinematics, context, owner, seen, basis, settings)
    seen.clear()
    decision = decide.decide_status(
        angle_decision,
        sampling,
        geometry,
        tree_energy,
        response,
        kinematics,
        subresonances,
        owner,
        strength,
        settings,
        ledger,
        member,
        episodes=episodes,
        withdrawn=withdrawn,
        slow_scale=slow_scale,
    )
    decision.rules_fired = decision.rules_fired | _slow_scale_rules(slow_scale) | subresonance_rules
    kept, displaced = _centre_scale_guard(held, decision, ledger, prep.cutoff.cutoff_yr)
    if kept is not None:  # the held center-scale reading keeps its precedence: the fine origin is not dynamic
        return kept
    held = None  # a displaced reading's arrays go
    # A3a: the decision's other key split by its role (owner/driver, neighbours, resonant argument, cause term)
    roles = roles_module.verdict_roles(
        series.resonance, decision.origin, decision.origin_detail, decision.owner, decision.flags, owner, settings
    )
    episodes = [_with_origin(p, decision, roles) if p.episode_id == decisive_id else p for p in episodes]
    episodes = _episode_arguments(episodes, decisive_id, roles, subresonances)  # A6b: each episode's own member
    episodes = _induced_origins(episodes, decision, owner, settings)  # A1: each induced episode names its owner
    union_episodic = float(angle_decision.rule == decide.RULE_UNION and not decision.union_promoted)  # (geo_union_read_episodic)
    angle_decision = decide.union_read(angle_decision, decision)  # a promoted union reads as the libration it is
    view = decide.tree_view(angle_decision, response, energy)  # the angle, response and energy the tree read
    read = view.response or response  # the response the tree read (rule R), else the record's

    base = decide.baseline(geometry, prep.cutoff.cutoff_yr, sampling.span_yr, response.b5_period_min_yr, settings, view.angle, kinematics)
    counts = _episode_tp_counts(prep, geometry, decisive_id)
    margins = decide.evaluate_margins(
        view,
        decision,
        context,
        sampling,
        geometry,
        kinematics,
        subresonances,
        owner,
        episodes,
        settings,
        prep=prep,
        episode_tp_counts=counts,
        strength=strength,
    )
    checks = decide.mmr_checks(
        view,
        sampling,
        prep,
        geometry,
        kinematics,
        subresonances,
        owner,
        strength,
        base,
        decision,
        margins,
        settings,
    )
    _not_run_steps(checks, subresonances, owner, settings)
    kept = decide.R0_KEPT if decide.r0_beside_guard(energy, sensitivity) else ()
    decide.apply_filter_sensitivity(checks, margins, sensitivity, keep=kept)  # DELTA 13a: before the caps and the level read them
    decide.apply_filter_too_slow(checks, margins, prep.filter_too_slow)  # DELTA item 1: the shape is not measured
    caps = decide.collect_caps(
        view,
        decision,
        geometry,
        margins,
        settings,
        sampling,
        owner is not None,
        subresonances=subresonances,
        one_to_one=response_module.law_planet(series.resonance) is not None,
        response=read,
    )
    decisive_run = geometry_module.run_of(geometry, decisive_id)
    decisive_half = decisive_run.half_amp_med_rad if decisive_run is not None else NAN
    # the cap short_period_aliased and its fold-free step keep reading the angle's own floor in the G1f pass (card g08-2 B)
    floor = subfloor.own_floor_yr if subfloor is not None else _period_floor(prep, settings)
    decisive_physics = physics_of(episodes, decisive_id)
    reported = decide.reported_angle(angle_decision, geometry, settings, decision)  # G1c/G1e -> episodic; K2 -> drift
    # a circulation by geometry only: kinematic_drift and no_resonant_potential rest on failed a-checks, which folded
    # lines can produce, so they keep the cap (W20 Q1d, independent review 23.10); so does a circulation whose accepted
    # runs F6 removed as forced waves: their failed identity reads a (W21a verify, 1036 9J-2S-3)
    circulation = (
        str(reported.angle_status) == AngleStatus.CIRCULATION and decision.reason in GEOMETRIC_CIRCULATIONS and not geometry.forced_waves
    )
    caps += short_period_caps(sampling, fast, decisive_half, settings, ledger, floor, circulation)
    set_fold_free_step(sampling, context, floor, settings)  # the floor of the cap (the flag may be set just above)
    regime, regime_start = decide.final_regime(angle_decision, geometry, settings, decision)
    flags = _flags(sampling, prep, geometry, read, energy, strength, subresonances, owner, decision)
    flags = decide.surviving_zone_flags(flags, view.energy, margins)  # the zone flags follow the reading that survived
    lifted_caps = _mark_exit_and_capture(series, view, reported, checks, flags, caps, decision, ledger, settings)
    worded = _centre_scale_union_word(reported, union_word, decision, ledger)
    rewritten, reported = worded is not reported, worded  # geo_union_word_centre_scale: only a word it rewrote
    reported = decide.forced_wave_word(reported, decision, sampling, geometry, energy, settings, ledger)  # F6 loops: episodic
    unexplained_lifted, caps = len(lifted_caps) < len(caps), lifted_caps
    # F only when measured by direct averaging; K1e for a kinematic origin (A5)
    physics = decide.physics_margins_of(decision, strength_module.physics_margins(decide.PHYSICS_MARGINS, strength))
    tree_margins = decide.margins_as_tree_read(margins, decision)  # R-ALIAS-A: a failed L3/L4 the tree read as not_measured
    caps += decide.full_support_caps(decision, tree_margins, physics, settings)  # q-2 b: high needs one full support
    # W28 C14 (rule end_arc_extend): an end arc after circulation sets final_regime and caps a non_resonant circulation;
    # an edge a longer record would resolve gives the flag extend_time (no status, no level)
    edges = decide.record_edges(geometry, decision, reported.angle_status, regime, regime_start, sampling.span_yr, settings, ledger)
    regime, regime_start = edges.final_regime, edges.final_regime_start_yr
    caps += list(edges.caps)
    flags += edges.flags
    decision.rules_fired = decision.rules_fired | edges.rules_fired
    assessment = confidence_module.assess(decision.status, tree_margins, caps, ledger.thin_margins(), physics)
    decision.rules_fired = decision.rules_fired | decide.kinematic_high_rules(decision, tree_margins, physics)
    strength_waves = _strength_waves(geometry, decision, settings)  # W28 B2: marked after the verdict and its numbers
    if decision.origin == LibrationOrigin.SUBRESONANCE and member is not None and decision.status == member.status:
        assessment = _member_confidence(member)
        unexplained_lifted = False  # the caps shown are the member's
    action, resample = _resample_action(default_action(decision.status, decision.reason, assessment.confidence), sampling)
    if decision.reason == Reason.BASELINE_NEEDED and base.cause:  # F8: P fits the record, a longer one decides nothing
        action, resample = Action.REVIEW, None
    action, resample = _quasi_satellite_stall_action(qs, angle_decision, decision, prep, sampling, settings, ledger) or (action, resample)
    action, resample = short_period_resample(action, resample, assessment, sampling)  # W21 A: the cap made actionable
    extend = decide.extend_years(decision.status, reported.angle_status, base, sampling.span_yr, settings)
    action, extend = _length_limited_action(action, extend, decision, view, geometry, read, sampling, prep, settings)
    verdict = Verdict(
        resonance_type=ResonanceType.MMR,
        status=decision.status,
        angle_status=reported.angle_status,
        angle_subtype=reported.subtype,
        final_regime=regime,
        final_regime_start_yr=regime_start if is_finite(regime_start) else None,
        orbit_type=orbit_type_module.mmr_orbit_type(
            series.resonance,
            geometry,
            decisive_id,
            response.coorbital,
            settings,
            str(reported.angle_status),
            orbit_type_module.not_this_resonance(decision.status, decision.origin, decision.reason, assessment.caps),
        ),
        confidence=assessment.confidence,
        confidence_raw=assessment.confidence_raw,
        reason=decision.reason,
        flags=flags,
        thin_margins=assessment.thin_margins,
        failed_support=assessment.failed_support,
        caps=assessment.caps,
        action=action,
        extend_to_years=extend if action == 'extend' else edges.extend_to_years,
        extend_direction=edges.direction,
        resample_step_yr=resample,
        libration_origin=decision.origin,
        origin_detail=decision.origin_detail,
        owner=roles.owner,
        neighbours=roles.neighbours,
        resonant_argument=roles.resonant_argument,
        cause_role=roles.cause_role,
        cause_term=roles.cause_term,
        commensurability=roles.commensurability,
        checks=checks,
        margins=margins,
        decision_path=list(ledger),
        rules_fired=decide.rules_fired(
            geometry, decision.rules_fired, preprocess_module.cutoff_rules(prep.cutoff), zone_module.energy_rules(energy)
        ),
    )
    # the series the response read, for the figure: made last, so they are not held through the scans above
    prep.angle_response, prep.rate_response, _, prep.adot_response, prep.response_cutoff_yr = response_module.response_series(
        prep, period, settings
    )
    waves = background_wave_module.background_waves(
        prep.t_yr,
        prep.angle_clean,
        prep.a_f,
        geometry.runs if geometry is not None else (),
        prep.window,
        settings,
        coorbital=geometry is not None and geometry.coorbital,
    )
    analysis = MMRAnalysis(
        series_name=series.name,
        resonance=series.resonance,
        context=context,
        sampling=sampling,
        prep=prep,
        slow=slow,
        geometry=geometry,
        energy=energy,
        response=read,
        kinematics=kinematics,
        subresonances=subresonances,
        owner=owner,
        strength=strength,
        baseline=base,
        episode_physics=episodes,
        verdict=verdict,
        settings_hash=settings_hash(settings),
        confidence_cells_sha=cells.sha,
        parent=parent,
        decisive_episode_id=decisive_id,
        background_waves=waves,
        extra_columns={
            **_extra_columns(
                geometry, read, counts, sensitivity, decision, prep, fast, settings, view.angle, member, kinematics, decisive_half
            ),
            # O1's number of the decisive episode (the branch where L4 and R0 are not measured reads it at decision time)
            'ep_onset_ratio': decide.onset_ratio(decisive_physics),
            # what set the flag extend_time (C14: 'end:slowdown', 'start:reflection', 'leg:end', 'half:start'; '' without it)
            'geo_extend_time_from': join_list(edges.sources),
            # 1 when the mark exit_and_capture took the cap transient_unexplained off the caps the confidence read
            'sep_unexplained_lifted': float(unexplained_lifted),
            'smp_da_sp_over_a_excursion': sampling.a_scatter_ratio,
            # R-ALIAS-A: the failed a-based checks the tree and the level read as not_measured (a_response;identity)
            'rsp_short_period_unmeasured': join_list(decide.withdrawn_checks(decision, decide.WITHDRAWN_ALIAS_A)),
            # the union of episodes (G2u) the status read as episodic, at this scale / at the scale of the centre when
            # it rewrote the word (`_centre_scale_union_word`), and the G2c analysis that union was (geo_cs_rej_*)
            'geo_union_read_episodic': union_episodic,
            'geo_union_word_centre_scale': float(rewritten),
            **(rejected or CS_REJECTED_NONE),
            # the G2c reading the guard did not let displace this verdict (geo_cs_displaced_*, `_displaced_columns`)
            **displaced,
            # the passed checks the filter guard withdrew before the tree read them (W21 Q9)
            'flt_withdrawn_before_tree': join_list(decide.withdrawn_checks(decision, decide.WITHDRAWN_FILTER_GUARD)),
            **quasi_satellite_module.columns(qs, qs_outcome),
            **(subfloor_found or subfloor_columns(None)),
            **slow_swing_module.slow_swing_columns(slow_swing),
        },
        candidate_top_n=(settings.mmr_subresonance_top_n, settings.mmr_owner_top_n),
        filter_order=settings.filter_order,
        slow_scale=slow_scale,
        subresonance_cheap=cheap,
    )
    _mark_strength_waves(geometry, strength_waves)
    analysis.derived = derived
    if analysis.derived:
        _share_multiplet_status(analysis)
    return analysis


def _decisive_kinematics(series, prep, context, geometry, response, kinematics, basis, settings) -> None:
    """The kinematic numbers of the decisive episode of a librating angle (`basis`), set on `kinematics`: the slow-angle
    test does not judge an episode, its cycle is judged instead (`decide.kinematic_check`, origin_detail); A5: K1 and the
    swing shares on that episode, at its response cutoff (the margin K1e reads them). Nothing for another angle."""
    if basis.angle_status not in decide.LIBRATING:
        return
    period = decide.decisive_period(basis, geometry)
    check = decide.kinematic_check(basis, kinematics, response, period, settings, geometry)
    ripple = context.great_inequality.period_yr if context.great_inequality is not None else NAN
    kinematics.detail = kinematics_module.origin_detail(kinematics, response, settings, period, ripple, check)
    window, _ = decisive_window(basis, geometry, prep)
    cutoff = response_module.response_cutoff_yr(prep.cutoff.cutoff_yr, period, settings)
    k1e = kinematics_module.episode_kinematics(series, prep, window, cutoff, settings)
    kinematics.k1_episode, kinematics.swing_share, kinematics.lambda_swing_share, kinematics.k1_episode_reason = k1e


def _slow_scale(series, prep, geometry, kinematics, context, owner, seen, basis, settings) -> Optional[SlowScale]:
    """The slow scale of the record (B10a, W28, cards g08-7, g08-8, g05-2; `kinematics.measure_slow_scale`): the first
    of G2c's centre-scale tracks (`seen`, `kinematics.centre_scale_track`) with an accepted run slower than the
    centre's own scale (`kinematics.slower_than`: a centre-scale run of a librating angle at its own period is that
    libration again, 6489 3J-1 at W_c 182 yr), else, for an angle that librates (as the status reads it, `basis`),
    the turns of its slow centre c(t); None without either or with MMR_SLOW_SCALE_REPORT off. An owner row names the
    cause from A2's slow x first (`SecondCapture.slow_x`)."""
    if not settings.mmr_slow_scale_report:
        return None
    librating = basis.angle_status in decide.LIBRATING
    main = decide.decisive_period(basis, geometry) if librating else NAN
    track = next((track for track in seen if kinematics_module.slower_than(track, main, settings)), None)
    if track is None and librating:
        track = kinematics_module.centre_track_slow(geometry, prep, context.identity_coefficient, settings)
    if track is None:
        return None
    owner_x = owner.capture.slow_x if owner is not None and owner.capture is not None else None
    slow = kinematics_module.measure_slow_scale(series, prep, kinematics, track, settings, owner_x)
    kinematics.meas_swing_rad = slow.halfamp_rad  # kin_meas_swing_deg, next to the swing p varpi predicts
    return slow


def _slow_scale_rules(slow: Optional[SlowScale]) -> frozenset:
    """The rule slow_scale when G2c's slow runs are stored at their own scale (episodes.csv scale='slow'); a slow centre
    read from the turns of c(t) is a description only and fires nothing."""
    return fire(frozenset(), Rule.SLOW_SCALE) if slow is not None and slow.episodes else frozenset()


def _strength_waves(geometry: Geometry, decision: decide.StatusDecision, settings: AnalysisSettings) -> tuple:
    """W28 B2: the episodes of a 5s row as forced waves of route 'strength' (`decide.strength_forced_waves`); the rule
    `forced_strength_mark` fires on the decision when there is one. `_mark_strength_waves` stores them later."""
    waves = decide.strength_forced_waves(geometry, decision, settings)
    if waves:
        decision.rules_fired = fire(decision.rules_fired, Rule.FORCED_STRENGTH_MARK)
    return waves


def _mark_strength_waves(geometry: Geometry, waves: tuple) -> None:
    """W28 B2: the forced waves of rule 5s (`decide.strength_forced_waves`) join `Geometry.forced_waves` after the
    verdict and the numbers the decision read (`_extra_columns`), so that only episodes.csv (`forced_wave` 'strength'),
    the derived geometry columns (`geo_n_episodes`) and the figure's episode marks change; nothing is measured again
    (no `_without_episodes`)."""
    if not waves:
        return
    geometry.forced_waves = tuple(geometry.forced_waves) + tuple(waves)
    geometry.forget_derived()


def _mark_exit_and_capture(series, view, reported, checks, flags, caps, decision, ledger, settings) -> List[str]:
    """Appends the mark exit_and_capture to `flags` when `decide.exit_and_capture` holds (the energy of the regime the
    status reads, `view.energy`) and returns the caps, without transient_unexplained when the mark lifts it
    (`decide.transient_unexplained_after_exit_and_capture`)."""
    marked = decide.exit_and_capture(
        series.resonance, reported.angle_status, view.energy, checks, flags, caps, decision.reason, reported.subtype, ledger
    )
    if marked:
        flags.append(str(Flag.EXIT_AND_CAPTURE))
    return decide.transient_unexplained_after_exit_and_capture(caps, marked, settings)


SHORT_PERIOD_HARMONIC = 'short_period_aliased: a harmonic folded slower than the period floor'
GEOMETRIC_CIRCULATIONS = (Reason.CIRCULATION, Reason.ALIAS)  # circulations that read no a-based check
SHORT_PERIOD_CAP = 'cap short_period_aliased: fast residual of sigma_f / median half-amplitude of the decisive episode'
# ledger names the explained verdict column reads (plotting/verdict_text.py)
CENTRE_SCALE_UNION = 'G2c episodes without a circulation between them cover the record at the scale of the centre (word only)'
QS_STALL_ENTRY = 'G1q a stall with P_QS below the cutoff: a filter artefact, no extension'


def short_period_folds(sampling: Sampling, settings: AnalysisSettings, ledger: Ledger, period_floor_yr: float) -> bool:
    """Whether the step folds short-period terms of the elements into slow lines (`sampling.short_period_trigger`, the
    predicate the dry-run reads too): the sampling flag short_period_aliased (dt >= min(P_orb, P_syn) / 2), or (W20
    round D, review C #3) a harmonic m <= mmr_short_period_harmonics folded slower than the angle's period floor below
    that step (`Sampling.folded_slowest_yr`; P_orb 12 yr at 5.999 yr folds its second harmonic to 36 kyr), which then
    sets the flag as well (already set before G0 by `flag_folded_harmonic`) and is recorded in the ledger. Higher
    harmonics are not searched (a convention)."""
    flagged = str(Flag.SHORT_PERIOD_ALIASED) in sampling.flags
    folded = sampling.folded_slowest_yr
    trigger = _short_period_trigger(sampling, period_floor_yr)
    if flagged and trigger != sampling_module.SHORT_PERIOD_HARMONIC:
        return True  # the Nyquist flag of `analyse_sampling` (the harmonic one set before G0 is compared below)
    if folded > 0 and period_floor_yr > 0:  # the comparison of the harmonic part, whatever it gives
        ledger.compare(SHORT_PERIOD_HARMONIC, folded, Op.GT, period_floor_yr, 'yr', 'smp_folded_slowest_yr')
    if not trigger:
        return False
    if not flagged:
        sampling.flags.append(str(Flag.SHORT_PERIOD_ALIASED))
    return True


def _short_period_trigger(sampling: Sampling, period_floor_yr: float) -> str:
    return sampling_module.short_period_trigger(
        sampling.step_yr, sampling.short_period_step_yr, sampling.folded_slowest_yr, period_floor_yr
    )


def flag_folded_harmonic(spec, context, sampling: Sampling, settings: AnalysisSettings, period_floor=None) -> bool:
    """The harmonic part of `short_period_folds` before G0 (W20 round F, review E #3): a harmonic folded slower than the
    angle's period floor (`preprocess.libration_period_floor`, or the quasi-satellite pass' `period_floor` (floor,
    setting)) sets the flag short_period_aliased on the record, so an invalid/undersampled or undefined verdict keeps
    the flag the dry-run predicts, and the fold-free step of every flagged record (`set_fold_free_step`). Only the flag:
    the cap and its ledger comparison stay in `short_period_caps`."""
    floor = period_floor[0] if period_floor is not None else preprocess_module.libration_period_floor(spec, context, settings)[0]
    harmonic = _short_period_trigger(sampling, floor) == sampling_module.SHORT_PERIOD_HARMONIC
    if harmonic and str(Flag.SHORT_PERIOD_ALIASED) not in sampling.flags:
        sampling.flags.append(str(Flag.SHORT_PERIOD_ALIASED))
    set_fold_free_step(sampling, context, floor, settings)
    return harmonic


def set_fold_free_step(sampling: Sampling, context, period_floor_yr: float, settings: AnalysisSettings) -> float:
    """`Sampling.fold_free_step_yr` (smp_fold_free_step_yr) of a record flagged short_period_aliased: the largest
    MMR_RESAMPLE_STEPS_YR step below its step at which the same predicate (`sampling.short_period_trigger`, the record's
    P_orb, P_syn, MMR_SHORT_PERIOD_HARMONICS and period floor) folds nothing (`sampling.fold_free_step`); nan for an
    unflagged record. Returns the value set."""
    step = NAN
    if str(Flag.SHORT_PERIOD_ALIASED) in sampling.flags:
        step = sampling_module.fold_free_step(
            context.orbital_period_yr,
            context.synodic_periods_yr,
            sampling.step_yr,
            period_floor_yr,
            settings.mmr_resample_steps_yr,
            settings.mmr_short_period_harmonics,
        )
    sampling.fold_free_step_yr = step
    return step


def short_period_resample(
    action: Action, resample: Optional[float], assessment: confidence_module.ConfidenceAssessment, sampling: Sampling
) -> Tuple[Action, Optional[float]]:
    """(action, resample step) when the cap short_period_aliased holds the level down (`confidence.cap_binds`: the level
    with that cap alone dropped is higher; W21 A, the author's Q1 of 23.10): resample at the fold-free step
    (`Sampling.fold_free_step_yr`), where the cap does not arise. Changes no status or confidence; an action that is
    already resample (alias, undersampled, the quasi-satellite stall) and a record without a fold-free step keep theirs."""
    step = sampling.fold_free_step_yr
    if action == Action.RESAMPLE or not np.isfinite(step):
        return action, resample
    if not confidence_module.cap_binds(assessment.confidence, assessment.confidence_raw, assessment.caps, Cap.SHORT_PERIOD_ALIASED):
        return action, resample
    return Action.RESAMPLE, float(step)


def short_period_caps(
    sampling: Sampling,
    fast_residual_rad: float,
    half_amplitude_rad: float,
    settings: AnalysisSettings,
    ledger: Ledger,
    period_floor_yr: float,
    circulation: bool = False,
) -> List[str]:
    """Cap short_period_aliased (medium) when the step folds short-period terms into slow ones (`short_period_folds`):
    how much of them reached sigma_f and a_f is not measurable from the record (inv-09: 0.1-0.3 deg in the Trojans at
    16 yr, 12.5 deg in 62243). Only a finely sampled pilot measures it (dry-run), so by default
    (mmr_short_period_cap_ratio 0) the record never lifts the cap (W20 round D).

    W20 round D measured it on fine reruns of 47 records (dec/clean on one trajectory): the folded lines move sigma_f by
    at most 0.024 of the decisive half-amplitude, but sit in a_f at 0.02-1.65 of its half-excursion; the four records
    whose verdict they change (2003 TG2 4E-1M-3, 2004 TK14 31M-34, 2003 UC22 16M-27, 433 47M-44) lose the identity and
    the force law, which read a. The record's own short-period scatter of a against that excursion
    (`Sampling.a_scatter_ratio`, mmr_short_period_a_ratio) sets the warning flag a_short_period_scatter; it is a proxy, not a
    bound (a line folded to centuries hardly enters a 50-yr high-pass), hence no lift by it alone.

    With mmr_short_period_cap_ratio > 0 (the W20 F5 lift, a convention) the cap is lifted when the fast residual of
    sigma_f (`fast_residual`, p95) is at most that x the decisive episode's median half-amplitude and the scatter of a
    passes mmr_short_period_a_ratio; without an episode, a fast residual or an a excursion the cap stays.

    A `circulation` (angle word circulation, reason circulation or alias: `GEOMETRIC_CIRCULATIONS`, no forced-wave
    run) keeps the flag but not the cap unless mmr_short_period_cap_circulation (W20 Q1d, author 23.10): it reads
    neither the identity nor the force law. kinematic_drift and no_resonant_potential rest on failed a-checks and keep
    it, and so does a circulation whose accepted runs F6 removed as forced waves (their failed identity reads a)."""
    if not short_period_folds(sampling, settings, ledger, period_floor_yr):
        return []
    if circulation and not settings.mmr_short_period_cap_circulation:
        return []
    cap = [str(Cap.SHORT_PERIOD_ALIASED)]
    lift = settings.mmr_short_period_cap_ratio > 0
    a_ratio = sampling.a_scatter_ratio
    if np.isfinite(a_ratio):
        # a warning by default: it is a convention only where it decides the lift
        if ledger.compare_setting(
            decide.SHORT_PERIOD_A, a_ratio, Op.GT, settings, 'mmr_short_period_a_ratio', '', 'smp_da_sp_over_a_excursion', named=lift
        ):
            sampling.flags.append(str(Flag.A_SHORT_PERIOD_SCATTER))
            return cap
    measured = np.isfinite(fast_residual_rad) and fast_residual_rad > 0  # `fast_residual` is 0 where it cannot be formed
    if not (lift and measured and np.isfinite(a_ratio) and np.isfinite(half_amplitude_rad) and half_amplitude_rad > 0):
        return cap
    ratio = fast_residual_rad / half_amplitude_rad
    if ledger.compare_setting(SHORT_PERIOD_CAP, ratio, Op.GT, settings, 'mmr_short_period_cap_ratio', '', 'smp_fast_over_half_amp'):
        return cap
    return []


def _not_run_steps(checks: dict, subresonances, owner, settings: AnalysisSettings) -> None:
    """A foreign step nothing triggered (or a derived row, depth 1) does not apply; not_measured stays for a search
    switched off."""
    if subresonances is None:
        checks['check_mmr_subresonances'] = CheckValue.NOT_APPLICABLE
    if owner is None and settings.mmr_owner_search != 'off':
        checks['check_mmr_owner'] = CheckValue.NOT_APPLICABLE


def _member_confidence(member: Verdict) -> confidence_module.ConfidenceAssessment:
    """Rule C: a catalogue row that takes the status of its accepted member (`decide.member_status`) takes the
    member's confidence with the margins and caps that set it."""
    return confidence_module.ConfidenceAssessment(
        Confidence(member.confidence),
        Confidence(member.confidence_raw),
        list(member.thin_margins),
        list(member.failed_support),
        list(member.caps),
    )


def _share_multiplet_status(analysis: MMRAnalysis) -> None:
    """The best status of a catalogue row and its derived rows, on each of them."""
    status = best_status([analysis.verdict.status] + [d.verdict.status for d in analysis.derived])
    for row in [analysis] + analysis.derived:
        row.multiplet_status = status
        row.verdict.multiplet_status = status


def _at_centre_scale(
    series, angle, geometry, prep, record_prominence, settings, cells, scan_subresonances, scan_owner, parent, seen=None
) -> Tuple[Optional[MMRAnalysis], dict]:
    """G2c, the centre librates: the analysis of the angle filtered at the scale of its centre, when the fine scale shows
    no libration and the centre does (the geometry of the angle at that scale holds a librating episode that covers
    the record), and that analysis decides libration (or a libration bounded by an observed exit or entry, the edge
    rule: `_decide_at_scale`). The fine turning points are then waves on a slow libration
    (2011 KT19 7N-9r6: 3-4 kyr waves on a 43 kyr libration around 180 deg). None otherwise; a union of episodes the
    status there read as episodic (`_union_read_episodic`) is no such libration: the second value holds the last such
    rejected analysis as its geo_cs_rej_* columns (`_rejected_columns`; empty for none), which the fine verdict's word
    reads (`_centre_scale_union_word`) and a kept analysis carries too. The centre-turn prominence
    comes from `record_prominence`, the TP prominence of the whole record: this scale reads the whole record, not the
    last regime after a jump of a.

    Two scales, in order: W_c = libration_centre_cycles x P_cyc (`geometry.centre_scale_angle`, KT19: 12.45 kyr from
    P_cyc 4.15 kyr of the waves), then W_c from the turns of the centre itself (`geometry.centre_turns_window_yr`: the
    fine turning points at the prominence of a centre turn, W_c = their period / filter_min_libration_cutoffs). The
    second covers a P_cyc that mixes waves and slow turns: waves seen only at some slow extremes give P_cyc 29.9 kyr
    for a 40 kyr libration, and 3 P_cyc = 90 kyr filters the libration away (Q12).

    A libration found at W_c whose period P is too short for that cutoff (filter_too_slow: G(P) at W_c below
    mmr_l8_gain_full, about P < filter_min_libration_cutoffs x W_c) is analysed once more at P /
    filter_min_libration_cutoffs (`_at_libration_scale`; the basis stays the one W_c came from, W_c itself goes to
    geo_centre_scale_first_cutoff_yr) and that verdict is kept when it again decides libration; else the one at W_c
    stays (15875 2N-3: W_c 12.3 kyr = 3 x the 4.1-kyr Uranus-Neptune wave, P 21.6 kyr). P is P_cyc, else the period
    of the decisive episode's cycle closed by its phase return (`_libration_period`, W28 B7).

    Then the scale ladder (W28 B7, card g07-4 A; `_at_ladder_scale`, `geometry.slow_scale_rung`): when the window of
    that scale holds fewer than libration_centre_min_estimates slow periods, the shortest shorter cutoff above the fast
    wave at which the slow extrema hold, kept when it decides libration again; neither L4 nor R0 with a measured there
    caps it at medium (`slow_scale_provisional`). 385533 7N-12: W_c 22.7 kyr, one return cycle (uncertain, 5n); at
    7.95 kyr four extrema 19.5-89.5 kyr, F by direct averaging passes: resonant, medium.

    `seen` collects the slow tracks of both scales W_c (`_decide_at_scale`), kept or not: the slow scale of the fine
    verdict reads them (B10a, `_slow_scale`)."""
    prominence = geometry_module.centre_prominence(record_prominence, settings)  # the centre scale reads the whole record
    first = geometry_module.centre_scale_angle(angle, geometry.period_cycle_yr, settings)
    found = _decide_at_scale(series, first, prominence, settings, cells, scan_subresonances, scan_owner, parent, seen)
    found, rejected = _union_guard(found, {})
    basis, turns_window = 'period_cycle', NAN
    if found is None:
        turns_window = geometry_module.centre_turns_window_yr(angle, prominence, settings)
        if first is None or turns_window != first.cutoff_yr:
            second = geometry_module.angle_at_centre_window(angle, turns_window, settings)
            found = _decide_at_scale(series, second, prominence, settings, cells, scan_subresonances, scan_owner, parent, seen)
            basis = 'centre_turns'
            found, rejected = _union_guard(found, rejected)
    if found is None:
        return None, rejected
    coarse, analysis = found
    ledger = Ledger.from_settings(settings)
    first_cutoff = NAN
    retry = _at_libration_scale(series, angle, coarse, analysis, prominence, settings, cells, scan_subresonances, scan_owner, parent)
    retry, rejected = _union_guard(retry, rejected)  # a rejected retry: the verdict at W_c stays, as for no libration
    if retry is not None:
        first_cutoff, (coarse, analysis) = coarse.cutoff_yr, retry
    ladder_from, rung = coarse.cutoff_yr, None  # B7: the scale ladder below the scale kept so far
    ladder_args = (series, angle, geometry, coarse, analysis, prominence, settings, cells, scan_subresonances, scan_owner, parent)
    ladder = _at_ladder_scale(*ladder_args)
    if ladder is not None:
        found, rejected = _union_guard(ladder[0], rejected)  # a rejected rung: the verdict above stays
        if found is not None:
            (coarse, analysis), rung = found, ladder[1]
    name = 'G2c the centre librates: the angle at the scale of its centre'
    entries = [ledger.compare(name, coarse.cutoff_yr, Op.GT, prep.cutoff.cutoff_yr, 'yr', 'geo_centre_scale_cutoff_yr')]
    if retry is not None:
        name = 'G2c W_c too slow for the libration found there: analysed again at P / filter_min_libration_cutoffs'
        entries.append(ledger.compare(name, first_cutoff, Op.GT, ladder_from, 'yr', 'geo_centre_scale_first_cutoff_yr'))
    ladder_rules = frozenset()
    if rung is not None:
        entries += _ladder_entries(rung, settings, ledger)
        ladder_rules = fire(ladder_rules, Rule.SCALE_LADDER)
        _slow_scale_cap(analysis)
    analysis.verdict.decision_path = entries + list(analysis.verdict.decision_path)
    fine_rules = preprocess_module.cutoff_rules(prep.cutoff)  # the fine pass's rules acted too
    analysis.verdict.rules_fired = decide.rules_fired(geometry, analysis.verdict.rules_fired, fine_rules, ladder_rules)
    analysis.prep.cutoff = replace(analysis.prep.cutoff, binding='centre_scale')
    analysis.settings_hash = settings_hash(settings)
    analysis.extra_columns.update(
        {
            'geo_centre_scale_cutoff_yr': coarse.cutoff_yr,
            'geo_centre_scale_basis': basis,
            'geo_centre_scale_first_cutoff_yr': first_cutoff,
            'geo_centre_turns_cutoff_yr': turns_window,
            'geo_fine_cutoff_yr': prep.cutoff.cutoff_yr,
            'geo_fine_n_tp': len(geometry.turning_points),
            'geo_fine_period_cycle_yr': geometry.period_cycle_yr,
            **_ladder_columns(rung, ladder_from),
            **rejected,
        }
    )
    return analysis, rejected


def _libration_period(analysis: MMRAnalysis) -> float:
    """P of the libration an analysis at a coarse scale found [yr]: P_cyc, else the period of its decisive episode (a
    cycle closed by its phase return has two turning points and no P_cyc: 385533 7N-12 at W_c 22.7 kyr, 45.4 kyr;
    W28 B7 (1)); nan without either."""
    geometry = analysis.geometry
    if geometry is None:
        return NAN
    if is_finite(geometry.period_cycle_yr) and geometry.period_cycle_yr > 0:
        return float(geometry.period_cycle_yr)
    run = geometry_module.run_of(geometry, analysis.decisive_episode_id)
    period = run.period_med_yr if run is not None else NAN
    return float(period) if is_finite(period) and period > 0 else NAN


def _at_ladder_scale(series, angle, fine, coarse, analysis, prominence, settings, cells, scan_subresonances, scan_owner, parent):
    """((coarse angle, analysis) of `_decide_at_scale`, the `geometry.LadderRung`) at the rung of the G2c scale ladder
    (W28 B7, `geometry.slow_scale_rung`) below the scale `coarse` whose analysis found a slow libration; P_fast from the
    fine geometry `fine`. None without a rung or when the analysis there does not decide libration (the verdict above
    stays: 182223 7N-19 at 5.98 kyr)."""
    fast = geometry_module.fast_period_yr(fine.turning_points)
    rung = geometry_module.slow_scale_rung(angle, coarse, prominence, _libration_period(analysis), fast, settings)
    if rung is None:
        return None
    found = _decide_at_scale(series, rung.angle, prominence, settings, cells, scan_subresonances, scan_owner, parent)
    return (found, rung) if found is not None else None


# the G2c scale ladder's ledger names (W28 B7; the explained column reads them, E19a)
LADDER_SHIFT = 'G2c scale ladder: the slow extrema of W_c hold at the shorter cutoff (worst shift / P_slow)'
LADDER_FLOOR = 'G2c scale ladder: the cutoff is above the fast wave (>= libration_centre_cycles x P_fast, and P_c)'
LADDER_GAIN = 'G2c scale ladder: slow extrema at the ladder cutoff against those at W_c'


def _ladder_entries(rung: geometry_module.LadderRung, settings: AnalysisSettings, ledger: Ledger) -> list:
    """The ledger entries of a kept rung of the scale ladder: the shift of its slow extrema, its floor, its gain."""
    return [
        ledger.compare(LADDER_SHIFT, rung.shift, Op.LE, settings.mmr_scale_tp_time_tol, 'P', 'geo_scale_ladder_shift'),
        ledger.compare(LADDER_FLOOR, rung.cutoff_yr, Op.GE, rung.floor_yr, 'yr', 'geo_scale_ladder_floor_yr'),
        ledger.compare(LADDER_GAIN, rung.n_tp, Op.GT, rung.n_tp_centre_scale, '', 'geo_scale_ladder_n_tp'),
    ]


def _ladder_columns(rung: Optional[geometry_module.LadderRung], above_yr: float) -> dict:
    """geo_scale_ladder_*: the rung of the scale ladder that decided (nan / empty without one): its cutoff, the scale it
    replaced, the slow extrema there and at that scale, the worst shift [P_slow], P_slow, P_fast and the floor [yr]."""
    if rung is None:
        return dict(LADDER_NONE)
    return {
        'geo_scale_ladder_cutoff_yr': rung.cutoff_yr,
        'geo_scale_ladder_above_yr': above_yr,
        'geo_scale_ladder_n_tp': float(rung.n_tp),
        'geo_scale_ladder_n_tp_centre': float(rung.n_tp_centre_scale),
        'geo_scale_ladder_shift': rung.shift,
        'geo_scale_ladder_slow_yr': rung.slow_period_yr,
        'geo_scale_ladder_fast_yr': rung.fast_period_yr,
        'geo_scale_ladder_floor_yr': rung.floor_yr,
    }


LADDER_NONE = {
    'geo_scale_ladder_cutoff_yr': NAN,
    'geo_scale_ladder_above_yr': NAN,
    'geo_scale_ladder_n_tp': NAN,
    'geo_scale_ladder_n_tp_centre': NAN,
    'geo_scale_ladder_shift': NAN,
    'geo_scale_ladder_slow_yr': NAN,
    'geo_scale_ladder_fast_yr': NAN,
    'geo_scale_ladder_floor_yr': NAN,
}


def slow_scale_provisional(margins) -> bool:
    """The cap slow_scale_provisional applies to a verdict read at a rung of the scale ladder: neither L4 (the force
    law) nor R0 (the potential with a) is measured there, so only the theory (F by direct averaging) supports it."""
    return all(m in margins and margins[m].margin == Margin.NOT_MEASURED for m in ('L4', 'R0'))


def _slow_scale_cap(analysis: MMRAnalysis) -> None:
    """The cap slow_scale_provisional (<= medium) on the verdict of a ladder rung (`slow_scale_provisional`); the level
    is min(level, medium) as `confidence.assess` would give with the cap among the others, the default action follows
    the level (medium: spot_check). Nothing for a verdict without a level (uncertain)."""
    verdict = analysis.verdict
    if not slow_scale_provisional(verdict.margins) or str(Cap.SLOW_SCALE_PROVISIONAL) in verdict.caps:
        return
    verdict.caps = list(verdict.caps) + [str(Cap.SLOW_SCALE_PROVISIONAL)]
    if verdict.confidence == Confidence.NONE:
        return
    before = verdict.confidence
    verdict.confidence = confidence_module.apply_caps(before, [Cap.SLOW_SCALE_PROVISIONAL])
    if verdict.action == default_action(verdict.status, verdict.reason, before):
        verdict.action = default_action(verdict.status, verdict.reason, verdict.confidence)


def _union_read_episodic(analysis: MMRAnalysis) -> bool:
    """The analysis' libration is a union of episodes of an angle that is not 1:1 that its status read as episodic
    (G2u not promoted, `decide.promote_union`): no libration G2c may keep (W22 physics guard)."""
    return analysis.extra_columns.get('geo_union_read_episodic') == 1


def _union_guard(found, rejected: dict):
    """(found, rejected): `found` (coarse angle, analysis) of `_decide_at_scale`, or (None, its geo_cs_rej_* columns)
    when the W22 guard rejects its analysis (`_union_read_episodic`); `rejected` stays otherwise. The columns are taken
    at once, so the rejected analysis is not held."""
    if found is None or not _union_read_episodic(found[1]):
        return found, rejected
    return None, _rejected_columns(found[1])


# the G2c analysis the union guard rejected (W23 d, status-neutral): empty values when there is none
CS_REJECTED_NONE = {
    'geo_cs_rej_cutoff_yr': NAN,
    'geo_cs_rej_tp_prominence_deg': NAN,
    'geo_cs_rej_status': '',
    'geo_cs_rej_angle_status': '',
    'geo_cs_rej_reason': '',
    'geo_cs_rej_origin': '',
    'geo_cs_rej_caps': '',
    'geo_cs_rej_n_accepted': NAN,
    'geo_cs_rej_union_n': NAN,
    'geo_cs_rej_episodes': '',
    'geo_cs_rej_decisive': '',
}


def _rejected_columns(analysis: MMRAnalysis) -> dict:
    """The geo_cs_rej_* columns of a G2c analysis the union guard rejected: its cutoff W_c [yr], TP prominence [deg],
    verdict (status, angle word, reason, origin, caps), accepted episodes (count and 't0-t1' spans [yr]), the number
    in its union and the decisive episode's span."""
    verdict, geometry = analysis.verdict, analysis.geometry
    accepted = [run for run in geometry.runs if run.accepted]
    decisive = geometry_module.run_of(geometry, analysis.decisive_episode_id)
    return {
        'geo_cs_rej_cutoff_yr': analysis.prep.cutoff.cutoff_yr,
        'geo_cs_rej_tp_prominence_deg': math.degrees(geometry.turning_points.prominence_rad),
        'geo_cs_rej_status': str(verdict.status),
        'geo_cs_rej_angle_status': str(verdict.angle_status),
        'geo_cs_rej_reason': str(verdict.reason),
        'geo_cs_rej_origin': str(verdict.libration_origin),
        'geo_cs_rej_caps': join_list(verdict.caps),
        'geo_cs_rej_n_accepted': float(len(accepted)),
        'geo_cs_rej_union_n': analysis.extra_columns.get('geo_union_n_episodes', NAN),
        'geo_cs_rej_episodes': join_list(_span(run) for run in accepted),
        'geo_cs_rej_decisive': _span(decisive) if decisive is not None else '',
    }


def _span(run) -> str:
    return f'{run.t_start_yr:.0f}-{run.t_end_yr:.0f}'


# the G2c guard's ledger name (the fine verdict kept; its column is the displaced reading's W_c)
CENTRE_SCALE_DISPLACED = 'G2c guard: the center-scale reading carries a measured objection; the fine dynamic libration stays'


def _held_centre_scale(analysis: Optional[MMRAnalysis], settings: AnalysisSettings) -> Optional[_HeldCentreScale]:
    """The analysis `_other_scales` kept, held for the fine decision (`_HeldCentreScale`) when the guard is on
    (libration_centre_scale_objection_guard), it is a G2c analysis (geo_centre_scale_cutoff_yr finite: G1q and G1f
    passes are not judged) and the tree left its origin unknown on a leaf where a measurement contradicts it
    (`decide.objection_leaf` of its rec_rule); None otherwise, and the analysis is returned as before."""
    if analysis is None or not settings.libration_centre_scale_objection_guard:
        return None
    columns = analysis.extra_columns
    if not is_finite(columns.get('geo_centre_scale_cutoff_yr', NAN)):
        return None
    if analysis.verdict.libration_origin != LibrationOrigin.UNKNOWN or not decide.objection_leaf(columns.get('rec_rule', '')):
        return None
    cuts = _ACTION_STEP_CUTS.get()
    return _HeldCentreScale(analysis, len(cuts) if cuts is not None else 0)


def _centre_scale_guard(
    held: Optional[_HeldCentreScale], decision: decide.StatusDecision, ledger: Ledger, fine_cutoff_yr: float
) -> Tuple[Optional[MMRAnalysis], dict]:
    """(the analysis to return instead of the fine one, the geo_cs_displaced_* columns): the held G2c analysis when
    the fine decision does not displace it (`decide.centre_scale_displaced`: the fine origin is not dynamic), else None
    and its columns (`_displaced_columns`), with rule centre_scale_displaced on the decision and its ledger entry
    (W_c against the fine cutoff); (None, empty columns) without a held analysis."""
    if held is None:
        return None, CS_DISPLACED_NONE
    coarse = held.analysis
    if not decide.centre_scale_displaced(decision, coarse.extra_columns.get('rec_rule', ''), coarse.verdict.libration_origin):
        return _return_held(held), CS_DISPLACED_NONE
    displaced = _displaced_columns(coarse)
    decision.rules_fired = fire(decision.rules_fired, Rule.CENTRE_SCALE_DISPLACED)
    cutoff = displaced['geo_cs_displaced_cutoff_yr']
    ledger.compare(CENTRE_SCALE_DISPLACED, cutoff, Op.GT, fine_cutoff_yr, 'yr', 'geo_cs_displaced_cutoff_yr')
    return None, displaced


def _return_held(held: _HeldCentreScale) -> MMRAnalysis:
    """The held G2c analysis as G2c kept it: the G2a cuts the fine pass made after it was held are dropped (without the
    guard that pass would have stopped there)."""
    cuts = _ACTION_STEP_CUTS.get()
    if cuts is not None:
        del cuts[held.cuts :]
    return held.analysis


# the G2c reading the guard did not let displace the fine verdict: empty values when there is none
CS_DISPLACED_NONE = {
    'geo_cs_displaced_cutoff_yr': NAN,
    'geo_cs_displaced_leaf': '',
    'geo_cs_displaced_status': '',
    'geo_cs_displaced_angle_status': '',
    'geo_cs_displaced_reason': '',
    'geo_cs_displaced_origin': '',
    'geo_cs_displaced_failed': '',
    'geo_cs_displaced_period_yr': NAN,
    'geo_cs_displaced_decisive': '',
}


def _displaced_columns(analysis: MMRAnalysis) -> dict:
    """The geo_cs_displaced_* columns of a G2c analysis the guard set aside (`decide.centre_scale_displaced`): its cutoff
    W_c [yr], leaf (rec_rule), the tree's verdict there (status, angle word, reason, origin), the checks that failed
    there (the objection kept with its reading), its libration period [yr] (`_libration_period`) and the decisive
    episode's span; its slow runs are the slow scale (`_slow_scale`)."""
    verdict = analysis.verdict
    decisive = geometry_module.run_of(analysis.geometry, analysis.decisive_episode_id) if analysis.geometry is not None else None
    return {
        'geo_cs_displaced_cutoff_yr': analysis.extra_columns.get('geo_centre_scale_cutoff_yr', NAN),
        'geo_cs_displaced_leaf': analysis.extra_columns.get('rec_rule', ''),
        'geo_cs_displaced_status': str(verdict.status),
        'geo_cs_displaced_angle_status': str(verdict.angle_status),
        'geo_cs_displaced_reason': str(verdict.reason),
        'geo_cs_displaced_origin': str(verdict.libration_origin),
        'geo_cs_displaced_failed': join_list(name for name, value in verdict.checks.items() if value == CheckValue.FAIL),
        'geo_cs_displaced_period_yr': _libration_period(analysis),
        'geo_cs_displaced_decisive': _span(decisive) if decisive is not None else '',
    }


def _centre_scale_union_word(reported, union_word: bool, decision, ledger: Ledger):
    """The word of an episodic verdict at the fine scale when G2c found a union of episodes at the scale of the centre
    that the status read as episodic (`_at_centre_scale`): the fine verdict stays, but the record never goes round
    (no turn in a gap nor over the chain, the edges reached), so the angle word is libration/regime_change as for a
    union at the fine scale (W22: 2003 TX9 3E+2J-10, uncertain after its failed force law, the author's libration).
    Other words and a word the tree set (K2) stay."""
    if not union_word or decision.angle_word is not None or reported.angle_status != AngleStatus.EPISODIC:
        return reported
    ledger.compare(CENTRE_SCALE_UNION, 1, Op.EQ, 1, '', 'geo_union_word_centre_scale')
    return replace(reported, angle_status=AngleStatus.LIBRATION, subtype=AngleSubtype.REGIME_CHANGE)


def _at_libration_scale(series, angle, coarse, analysis, prominence, settings, cells, scan_subresonances, scan_owner, parent):
    """(coarse angle, analysis) at min(W_c, P / filter_min_libration_cutoffs) when the libration found at W_c (`coarse`,
    `analysis`) has the flag filter_too_slow (G(P) at W_c below mmr_l8_gain_full: the filter damps the libration it
    found) and the analysis at that shorter cutoff again decides libration (`_decide_at_scale`); None otherwise, and
    the verdict at W_c stays (at P close to W_c the shorter cutoff approaches the waves). P is `_libration_period`; a
    period from the phase return alone is put through the filter guard here (the guard of the analysis had none)."""
    period = _libration_period(analysis)
    if analysis.prep is None or not is_finite(period):
        return None
    too_slow = analysis.prep.filter_too_slow
    if not is_finite(analysis.geometry.period_cycle_yr):  # P from the phase return: the guard had no period (B7 (1))
        too_slow = preprocess_module.filter_guard(period, coarse.cutoff_yr, settings, analysis.sampling.step_yr)[0]
    if not too_slow:
        return None
    window = min(coarse.cutoff_yr, period / settings.filter_min_libration_cutoffs)
    if not window < coarse.cutoff_yr:
        return None
    finer = geometry_module.angle_at_centre_window(angle, window, settings)
    return _decide_at_scale(series, finer, prominence, settings, cells, scan_subresonances, scan_owner, parent)


def _decide_at_scale(series, coarse, prominence, settings, cells, scan_subresonances, scan_owner, parent, seen=None):
    """(coarse angle, analysis) when the angle at the centre scale `coarse` holds a librating episode that covers the
    record and the analysis at that cutoff decides libration, or an episode that covers it but for an observed exit or
    entry (the edge rule, `_bounded_by_observed_edge`: 182223 7N-19, one slow cycle of 38 kyr and then an exit of 1.5
    rev at 86.6 kyr); None otherwise (also for no such scale).

    The analysis at that cutoff takes its turning points at the centre-turn `prominence` p_c [rad] that qualified the
    scale (W23 rule a, libration_centre_scale_tp_at_centre: mmr_tp_prominence_floor_deg raised to p_c, the local
    regime prominences with it): at W_c a turn is a turn of the centre as the precheck reads it (the definition of
    `geometry.centre_turns_window_yr`); a smaller swing is not resolved at this scale (residue of the fast component
    W_c attenuates, or a libration too small for it, which the fine scale judges when G2c keeps nothing). At the 1-deg
    floor 2003 TX9 3E+2J-10 picked up 1-7 deg loops at W_c 1.1 kyr, which failed the fixed band and cut its slow
    libration in three. The analyses run inside (the retry at the libration scale, a derived subresonance row, an
    owner-identity phi) inherit W_c and the raised floor; none runs inside a kept G2c analysis of the benchmark (W23).
    `seen` (a list) collects the slow track of that geometry (`kinematics.centre_scale_track`, None without an accepted
    run) before the covering filter: the slow scale reads its runs (B10a); nothing else of the geometry is kept.
    """
    if coarse is None:
        return None
    measured = geometry_module.measure_geometry(coarse, prominence, settings)
    episode = decide.decisive_episode(measured, settings)
    if seen is not None:
        seen.append(kinematics_module.centre_scale_track(coarse, measured))
    if episode is None or not all(geometry_module.covers_edges(_edges_unobserved(episode), settings)):
        return None
    at_scale = replace(settings, mmr_filter_cutoff_yr=coarse.cutoff_yr)
    if settings.libration_centre_scale_tp_at_centre:
        floor = max(settings.mmr_tp_prominence_floor_deg, math.degrees(prominence))
        at_scale = replace(at_scale, mmr_tp_prominence_floor_deg=floor)
    analysis = analyse_mmr(
        series,
        at_scale,
        cells,
        scan_subresonances=scan_subresonances,
        scan_owner=scan_owner,
        parent=parent,
        centre_scale=False,
    )
    librates = analysis.verdict.angle_status == AngleStatus.LIBRATION or _bounded_by_observed_edge(analysis, settings)
    return (coarse, analysis) if librates else None


def _edges_unobserved(episode):
    return replace(episode, exit_observed=False, entry_observed=False)


def _bounded_by_observed_edge(analysis: MMRAnalysis, settings: AnalysisSettings) -> bool:
    """An episodic verdict whose decisive episode covers the record but for an exit or entry the record shows (the
    edge rule, `geometry.Episode.exit_observed`): the slow libration is there, and so is its end."""
    geometry = analysis.geometry
    if analysis.verdict.angle_status != AngleStatus.EPISODIC or geometry is None:
        return False
    episode = geometry_module.run_of(geometry, analysis.decisive_episode_id)
    if episode is None or not (episode.exit_observed or episode.entry_observed):
        return False
    return all(geometry_module.covers_edges(_edges_unobserved(episode), settings))


def _other_scales(
    series, context, sampling, prep, angle, geometry, prominence, angle_decision, ledger, settings, cells, options, induced=None, seen=None
):
    """(the quasi-satellite well or None, the G1q outcome, the analysis at another scale or None, the geo_cs_rej_*
    columns of a G2c union of episodes read as episodic, empty for none, `_centre_scale_union_word`, the G1f columns,
    empty when G1f was not tried): G1q, the analysis at the quasi-satellite floor of a 1:1 angle
    (`_quasi_satellite_pass`), then G1f, the analysis of a two-body angle at the floor of its own averaged potential
    (`_subfloor_pass`), then G2c, at the scale of the centre (`_at_centre_scale`), each kept when it finds the libration
    (a union of episodes of an angle that is not 1:1 does not pre-empt G2c: it is tried as the episodic angle it is for
    the status); only on the first pass of the record (`options`: scan_subresonances, scan_owner, parent, centre_scale,
    the quasi-satellite of a pass already at its floor, the G1f pass; never in the second pass of the owner per
    episode, `induced`, which returns the first pass's)."""
    scan_subresonances, scan_owner, parent, centre_scale, quasi_satellite, subfloor = options
    if induced is not None:  # the second pass of the owner per episode (A1): measured by the first pass, which kept this scale
        return induced.qs, induced.qs_outcome, None, induced.rejected, induced.subfloor
    if not centre_scale or quasi_satellite is not None or subfloor is not None:
        return quasi_satellite, '', None, {}, {}
    if angle_decision.rule == decide.RULE_UNION:  # G2c before the union of episodes (136108 7N-12 at 100 kyr)
        angle_decision = decide.status_basis(angle_decision)
    # measured only where G1q can fire: a libration keeps its pass (100-kyr Trojans paid 15 s for nothing), an aliased
    # step has no force law (2023 RS2 at 10 yr: 14 s)
    qs = None
    if angle_decision.angle_status != AngleStatus.LIBRATION and sampling.verdict not in response_module.ALIASED:
        qs = quasi_satellite_module.measure(series, context, prep, sampling.step_yr, settings)
    second, outcome, qs = _quasi_satellite_pass(
        series, qs, angle_decision, geometry, settings, cells, scan_subresonances, scan_owner, parent
    )
    if second is not None:
        return qs, outcome, second, {}, {}
    sub_args = (series, context, sampling, prep, geometry, angle_decision, settings, cells, scan_subresonances, scan_owner, parent)
    second, found = _subfloor_pass(*sub_args)
    if second is not None:
        return qs, outcome, second, {}, found
    if angle_decision.angle_status != AngleStatus.LIBRATION and not decide.edge_observed(geometry, ledger):
        args = (series, angle, geometry, prep, prominence, settings, cells, scan_subresonances, scan_owner, parent, seen)
        return (qs, outcome, *_at_centre_scale(*args), found)
    return qs, outcome, None, {}, found


# G1f's ledger names (the explained verdict column reads them, E19a)
SUBFLOOR_FLOOR = 'G1f floor of the angle\'s own averaged potential T0_th / sqrt(MMR_STRENGTH_CHECK_RATIO)'
SUBFLOOR_CYCLE = 'G1f the pass\'s decisive cycle is faster than the angle\'s own floor'
SUBFLOOR_F = 'G1f strength F by direct averaging passes at the pass\'s scale (pass = 1)'
SUBFLOOR_CENTRE = 'G1f the theoretical centre lies inside every observed swing (offset - smallest half-amplitude)'
SUBFLOOR_L4 = 'G1f L4 (a responds to the force) passes at the pass\'s scale (pass = 1)'
# geo_subfloor_failed: the conditions of a pass that was not kept
SUBFLOOR_NO_CYCLE, SUBFLOOR_NO_F, SUBFLOOR_NO_CENTRE, SUBFLOOR_NO_L4 = 'cycle', 'F', 'centre', 'L4'


@dataclass(frozen=True)
class SubfloorPass:
    """The G1f pass of a two-body angle (`_subfloor_pass`): its floor and source, the angle's own floor (the cap
    short_period_aliased and its fold-free step keep reading it) and the theoretical period T0_th the floor comes from."""

    floor_yr: float
    source: str
    own_floor_yr: float
    period_yr: float

    @property
    def period_floor(self) -> Tuple[float, str]:
        return self.floor_yr, self.source


def subfloor_columns(
    found: Optional[SubfloorPass], kept: Optional[bool] = None, failed: Sequence[str] = (), period_yr: float = NAN
) -> dict:
    """G1f's columns: `geo_subfloor_t0_theory_yr` (T0_th), `geo_subfloor_floor_yr` (the pass's floor),
    `geo_subfloor_period_yr` (the period of the pass's decisive episode), `geo_subfloor_kept` (1 the row is the G1f
    pass, 0 the pass ran and was not kept, nan not run) and `geo_subfloor_failed` (the conditions that kept it out:
    cycle, F, centre, L4); nan and empty without a pass."""
    if found is None:
        return {
            'geo_subfloor_t0_theory_yr': NAN,
            'geo_subfloor_floor_yr': NAN,
            'geo_subfloor_period_yr': NAN,
            'geo_subfloor_kept': NAN,
            'geo_subfloor_failed': '',
        }
    return {
        'geo_subfloor_t0_theory_yr': found.period_yr,
        'geo_subfloor_floor_yr': found.floor_yr,
        'geo_subfloor_period_yr': float(period_yr),
        'geo_subfloor_kept': NAN if kept is None else float(kept),
        'geo_subfloor_failed': join_list(failed),
    }


def subfloor_period(second: MMRAnalysis) -> float:
    """The median period of the G1f pass's decisive episode [yr] (nan without an accepted one)."""
    geometry = second.geometry
    run = geometry_module.run_of(geometry, second.decisive_episode_id) if geometry is not None else None
    return float(run.period_med_yr) if run is not None and run.accepted else NAN


def subfloor_eligible(series, context, sampling, prep, angle_decision, settings) -> Optional[SubfloorPass]:
    """G1f's pass when it applies (card g08-2 B): MMR_SUBFLOOR_PASS on, a two-body angle that is not 1:1 with a
    direct theory whose snapshots are consistent with sigma's swings (`preprocess.profile_period_floor`), a step that
    resolves the angle (an aliased one has no force law), no libration over the record at the angle's own floor, and
    a floor max(T0_th / sqrt(MMR_STRENGTH_CHECK_RATIO), 2 dt FILTER_MIN_LIBRATION_CUTOFFS) below that floor whose
    cutoff the step can filter (P_c > 2 dt: where the realisable bound binds, its cutoff is 2 dt and the step cannot
    resolve the pass); None otherwise. Three-body angles have no computed theory, so they are never eligible."""
    if not settings.mmr_subfloor_pass or angle_decision.angle_status == AngleStatus.LIBRATION:
        return None
    if sampling.verdict in response_module.ALIASED or not strength_module.direct_applicable(series.resonance):
        return None
    own = _period_floor(prep, settings)
    profile = preprocess_module.profile_period_floor(series, context, prep, settings)
    if profile is None or not own > 0:
        return None
    floor = max(profile.floor_yr, 2 * sampling.step_yr * settings.filter_min_libration_cutoffs)  # the realisable floor
    if not floor < own:
        return None
    cutoff, _, _ = preprocess_module.angle_cutoff(series.resonance, context, sampling.step_yr, settings, (floor, profile.source))
    if not preprocess_module.filter_realizable(cutoff.cutoff_yr, sampling.step_yr):
        return None
    return SubfloorPass(float(floor), profile.source, float(own), profile.period_yr)


def subfloor_conditions(second: MMRAnalysis, own_floor_yr: float, ledger: Ledger) -> List[str]:
    """The G1f conditions the pass `second` fails (empty: kept), each a ledger entry: its decisive cycle is faster
    than the angle's own floor (otherwise the pass found nothing the floor hid); F passes by direct averaging; the
    theoretical centre lies inside every observed swing (its largest offset from the observed centres, over every
    evaluation, is at most the decisive episode's smallest half-amplitude: no free constant); L4 passes."""
    failed = []
    geometry, strength = second.geometry, second.strength
    run = geometry_module.run_of(geometry, second.decisive_episode_id) if geometry is not None else None
    period = subfloor_period(second)
    if not ledger.compare(SUBFLOOR_CYCLE, period, Op.LT, own_floor_yr, 'yr', 'geo_subfloor_period_yr'):
        failed.append(SUBFLOOR_NO_CYCLE)
    direct = strength is not None and strength.method in strength_module.DIRECT_METHODS
    f_pass = direct and strength.check == CheckValue.PASS
    if not ledger.compare(SUBFLOOR_F, float(f_pass), Op.EQ, 1.0, '', 'check_mmr_strength'):
        failed.append(SUBFLOOR_NO_F)
    offset = strength.centre_offset_max_deg if direct else NAN
    half = math.degrees(run.half_amp_min_rad) if run is not None else NAN
    if not ledger.compare(SUBFLOOR_CENTRE, offset - half, Op.LE, 0.0, 'deg', 'str_centre_offset_max_deg'):
        failed.append(SUBFLOOR_NO_CENTRE)
    l4 = second.verdict.checks.get('check_mmr_a_response') == CheckValue.PASS
    if not ledger.compare(SUBFLOOR_L4, float(l4), Op.EQ, 1.0, '', 'check_mmr_a_response'):
        failed.append(SUBFLOOR_NO_L4)
    return failed


def _subfloor_pass(
    series, context, sampling, prep, geometry, angle_decision, settings, cells, scan_subresonances, scan_owner, parent
) -> Tuple[Optional[MMRAnalysis], dict]:
    """G1f (card g08-2 B, rule `subfloor_pass`): (the analysis at the floor of the angle's own averaged potential, its
    columns) when the pass applies (`subfloor_eligible`) and keeps its decisive cycle (`subfloor_conditions`: a cycle
    below the angle's own floor, F by direct averaging, the centre inside the swings, L4); else (None, the columns,
    empty when the pass did not apply). The floor itself (MMR_LIBRATION_PERIOD_MIN_YR) stays: the pass only accepts a
    cycle the resonance's own potential, its centre and its force confirm."""
    found = subfloor_eligible(series, context, sampling, prep, angle_decision, settings)
    if found is None:
        return None, {}
    second = analyse_mmr(
        series,
        settings,
        cells,
        scan_subresonances=scan_subresonances,
        scan_owner=scan_owner,
        parent=parent,
        centre_scale=False,
        subfloor=found,
    )
    ledger = Ledger.from_settings(settings)
    floor = ledger.compare(SUBFLOOR_FLOOR, found.floor_yr, Op.LT, found.own_floor_yr, 'yr', 'geo_subfloor_floor_yr')
    failed = subfloor_conditions(second, found.own_floor_yr, ledger)
    period = subfloor_period(second)
    if not floor or failed:
        return None, subfloor_columns(found, False, failed, period)
    second.verdict.decision_path = list(ledger) + list(second.verdict.decision_path)
    # the pass decided the row: the first pass's geometry rules acted too (rules contract), and G1f fired
    second.verdict.rules_fired = decide.rules_fired(geometry, second.verdict.rules_fired, fire((), Rule.SUBFLOOR_PASS))
    second.extra_columns.update(subfloor_columns(found, True, (), period))
    return second, second.extra_columns


def _quasi_satellite_pass(
    series, qs, angle_decision, geometry, settings, cells, scan_subresonances, scan_owner, parent
) -> Tuple[Optional[MMRAnalysis], str, object]:
    """G1q (`quasi_satellite`): (the analysis at the quasi-satellite floor, 'kept') when the angle has a quasi-satellite
    well (`qs`), no libration over the record at the first pass (`geometry`, `angle_decision`), visits the well long
    enough, that floor is below the tadpole floor, and the pass at it holds a librating episode centred in the well and
    keeps as many librating episodes outside the well as the first pass (the short cutoff leaves the planet's wiggles
    in a horseshoe, whose cycle closed by its return at P_c may not close there: 2019 GF1); else (None, outcome): ''
    when the rule does not fire, 'resample' when the step cannot hold its cutoff, 'rejected' when the pass fails; the
    third item is `qs`, which on a rejected pass carries why (`QuasiSatellite.rejection`, `qs_rejection`).
    The trigger (W28 C5, rule qs_trigger, fired on the first pass's geometry whenever the pass runs): a stay of
    mmr_coorbital_qs_trigger_periods x P_QS in the well holding a turn of the averaged angle (`QuasiSatellite.turns`)."""
    if qs is None or angle_decision.angle_status == AngleStatus.LIBRATION:
        return None, '', qs
    scratch = Ledger.from_settings(settings)
    if not (qs.visits(settings, scratch) and qs.turns(scratch) and qs.lowers_floor(scratch)):
        return None, '', qs
    if not qs.realizable:
        return None, quasi_satellite_module.PASS_RESAMPLE, qs
    geometry.rules_fired = fire(geometry.rules_fired, Rule.QS_TRIGGER)
    second = analyse_mmr(
        series,
        settings,
        cells,
        scan_subresonances=scan_subresonances,
        scan_owner=scan_owner,
        parent=parent,
        centre_scale=False,
        quasi_satellite=qs,
    )
    well = qs.floor_well()
    found = decide.librating_episodes(second.geometry, settings) if second.geometry is not None else []
    first_outside = [e for e in decide.librating_episodes(geometry, settings) if not well.holds(e)]
    outside = len([e for e in found if not well.holds(e)]) + _kept_in_regime_change(found, first_outside, well)
    before = len(first_outside)
    if not any(well.holds(e) for e in found) or not qs.keeps_outside(outside, before, Ledger.from_settings(settings)):
        rejection = qs_rejection(second.geometry, well, found, settings)
        return None, quasi_satellite_module.PASS_REJECTED, replace(qs, rejection=tuple(rejection.items()))
    ledger = Ledger.from_settings(settings)
    # the turn after the floor: the first two entries stay the visit and the floor (verdict_text reads them)
    entries = [qs.visits(settings, ledger), qs.lowers_floor(ledger), qs.turns(ledger), qs.keeps_outside(outside, before, ledger)]
    entries += _qs_run_floor_entries(second.geometry, well, found, ledger, settings)
    second.verdict.decision_path = entries + list(second.verdict.decision_path)
    second.verdict.rules_fired = decide.rules_fired(geometry, second.verdict.rules_fired)  # the first pass's rules acted too
    second.extra_columns.update(quasi_satellite_module.columns(qs, quasi_satellite_module.PASS_KEPT))
    second.extra_columns['geo_qs_n_outside_episodes'] = float(outside)
    return second, quasi_satellite_module.PASS_KEPT, qs


def qs_rejection(geometry: Optional[Geometry], well, found: List[geometry_module.Episode], settings: AnalysisSettings) -> dict:
    """Why a G1q pass was rejected (reporting only; first-pass row columns `geo_qs_reject`, `geo_qs_run_*`): read on its
    best in-well run (the longest run of >= 2 TPs the well holds, else the longest run): no_turning_points (no such
    run), keeps_outside (an in-well librating episode was found, the pass lost an outside one), no_run_in_well, else
    the run's `ineligible_reason` (below_period_floor, centre_turns, before_last_jump, ...) or, for a run the geometry
    did not accept, 'not_accepted:<reject_reason>'. The run's floor (`decide.period_floor`) and its own reading
    (`cpl_*`: source, loop, K / K_theory, R^2) name the failing sub-test."""
    columns = quasi_satellite_module.rejection_none()
    runs = [r for r in (geometry.runs if geometry is not None else ()) if len(r.tp_times_yr) >= 2]
    if not runs:
        columns['geo_qs_reject'] = quasi_satellite_module.REJECT_NO_TP
        return columns
    inside = [r for r in runs if well.holds(r)]
    run = max(inside or runs, key=lambda r: (r.t_end_yr - r.t_start_yr, len(r.tp_times_yr)))
    if any(well.holds(e) for e in found):
        reason = quasi_satellite_module.REJECT_KEEPS_OUTSIDE
    elif not inside:
        reason = quasi_satellite_module.REJECT_NO_RUN_IN_WELL
    elif not run.accepted:
        reason = f'not_accepted:{run.reject_reason}'
    else:
        reason = str(decide.episode_eligibility(geometry, settings).get(run.episode_id, ''))
    own = geometry.run_floor(run)
    reading = dict(own.columns) if own is not None else {}
    columns.update(
        geo_qs_reject=reason,
        geo_qs_run_t0_yr=float(run.t_start_yr),
        geo_qs_run_t1_yr=float(run.t_end_yr),
        geo_qs_run_n_tp=float(len(run.tp_times_yr)),
        geo_qs_run_period_yr=float(run.period_med_yr) if run.accepted else 2 * float(np.median(np.diff(run.tp_times_yr))),
        geo_qs_run_half_amp_deg=math.degrees(run.half_amp_med_rad) if is_finite(run.half_amp_med_rad) else NAN,
        geo_qs_run_floor_yr=float(decide.period_floor(geometry, settings, run)[0]),
        geo_qs_run_floor_source=str(reading.get('cpl_floor_source', '')),
        geo_qs_run_loop_yr=float(reading.get('cpl_loop_period_yr', NAN)),
        geo_qs_run_K_over_K_theory=float(reading.get('cpl_law_K_over_K_theory', NAN)),
        geo_qs_run_R2=float(reading.get('cpl_law_R2', NAN)),
    )
    return columns


def _qs_run_floor_entries(geometry: Optional[Geometry], well, found, ledger: Ledger, settings: AnalysisSettings) -> list:
    """The ledger entry of rule coorbital_qs_run_floor on a kept G1q pass: the QS floor (column flt_qs_floor_yr) above
    the first in-well librating episode's own loop floor, when that episode reads one (source qs_loop); after G1q's own
    entries (verdict_text reads the first two)."""
    if geometry is None:
        return []
    for episode in found:
        own = geometry.run_floor(episode)
        if well.holds(episode) and own is not None and own.holds and own.source == coorbital_geometry.FLOOR_SOURCE_QS_LOOP:
            entry = ledger.compare(
                quasi_satellite_module.QS_RUN_FLOOR,
                float(geometry.period_floor_yr),
                Op.GT,
                float(own.floor_yr),
                'yr',
                'flt_qs_floor_yr',
                setting=own.setting,
            )
            return [entry]
    return []


def _kept_in_regime_change(found: List[geometry_module.Episode], outside: List[geometry_module.Episode], well) -> int:
    """How many of the first pass's librating episodes outside the quasi-satellite well (`outside`) lie inside an
    episode of the G1q pass that holds the well and others (`Episode.several_wells`, rule coorbital_well_cut, W28 C7): the
    pass keeps them as a sub-regime of one libration (164207 1E-1: the horseshoe after the quasi-satellite)."""
    joined = [e for e in found if e.several_wells and well.holds(e)]
    return len([e for e in outside if any(j.t_start_yr <= e.t_start_yr and e.t_end_yr <= j.t_end_yr for j in joined)])


def _quasi_satellite_stall_action(
    qs, angle_decision, decision, prep, sampling, settings, ledger
) -> Optional[Tuple[Action, Optional[float]]]:
    """(action, resample step) for a stall of a 1:1 angle that stays in its quasi-satellite well long enough (G1q's
    visit) and whose quasi-satellite period P_QS is below the cutoff P_c: the filter erased the libration, a longer
    record cures nothing. resample at a step that holds the quasi-satellite cutoff when the record's step cannot
    (`QuasiSatellite.realizable`), else review; None otherwise."""
    if qs is None or decision.status != Status.UNCERTAIN or angle_decision.subtype != AngleSubtype.STALL:
        return None
    if not qs.visits(settings, ledger):  # a stall outside the well is no erased quasi-satellite (2015 TL144 1M-1)
        return None
    if not ledger.compare(QS_STALL_ENTRY, qs.period_min_yr, Op.LT, prep.cutoff.cutoff_yr, 'yr', 'flt_qs_period_yr'):
        return None
    if qs.realizable:
        return Action.REVIEW, None
    step = sampling_module.resample_step(
        sampling.safe_step_plain_yr, sampling.safe_step_pred_yr, qs.period_min_yr, sampling.step_yr, settings, qs.cutoff_yr
    )
    return (Action.RESAMPLE, step) if step is not None else (Action.REVIEW, None)


def _with_origin(physics, decision, roles: roles_module.Roles):
    """The decisive episode's origin, owner (an owner or driver only) and resonant argument (episodes.csv)."""
    return replace(physics, libration_origin=str(decision.origin), owner=roles.owner, resonant_argument=roles.resonant_argument)


def _episode_arguments(episodes, decisive_id, roles: roles_module.Roles, subresonances: Optional[SubresonanceResult]) -> list:
    """episodes.csv resonant_argument per episode (W28 A6b, card g06-8c A): when the verdict names a resonant argument
    (rule C or 'composite'), every other accepted episode takes the member chosen in its own scan window
    (`SubresonanceResult.episode_arguments`: a key, 'composite', or nothing); the decisive episode keeps the verdict's."""
    if subresonances is None or not roles.resonant_argument:
        return episodes
    chosen = subresonances.episode_arguments
    return [
        replace(p, resonant_argument=chosen[p.episode_id]) if p.episode_id != decisive_id and p.episode_id in chosen else p
        for p in episodes
    ]


def _induced_origins(episodes, decision, owner: Optional[OwnerResult], settings: AnalysisSettings) -> list:
    """episodes.csv origin per episode (A1, card g04-3): libration_origin induced and the owner's key on every episode
    the owner induces, when the row is induced (rule 2: all of them) or its own episodes decided after a second pass
    (`OwnerResult.induced_ids`); the others keep theirs."""
    if owner is None or owner.accepted is None:
        return episodes
    if decision.origin == LibrationOrigin.INDUCED:
        ids = {entry.episode_id for entry in owner_module.induced_split(owner, settings)[0]} | set(owner.induced_ids)
    else:
        ids = set(owner.induced_ids)
    ids.discard(None)
    if not ids:
        return episodes
    key = owner.accepted.spec.key
    return [replace(p, libration_origin=str(LibrationOrigin.INDUCED), owner=key) if p.episode_id in ids else p for p in episodes]


def centre_travel_columns(geometry: Geometry, settings: AnalysisSettings) -> dict:
    """The centre travels G1c and G2 read (ledger columns): the largest monotone travel inside an episode, the travel
    after the last turning point and that stretch in periods of the last librating episode."""
    travel_max = max((geometry_module.episode_centre_travel(geometry, e) for e in geometry.episodes), default=NAN)
    travel, beyond, _ = decide.final_stretch(geometry, settings) or (NAN, NAN, None)
    return {'geo_episode_centre_travel_max_rad': travel_max, 'geo_final_centre_travel_rad': travel, 'geo_final_circulation_P': beyond}


def _extra_columns(
    geometry: Geometry,
    response: Response,
    counts: dict,
    sensitivity: FilterSensitivity,
    decision,
    prep,
    fast_residual_rad,
    settings,
    angle,
    member,
    kinematics,
    decide_half: float,
) -> dict:
    """Numbers the decision read that no result dataclass holds (all in mmr_numbers.csv); `angle` as the status read it
    (`TreeView.angle`); `decide_half` is the median half-amplitude of the decisive episode's run [rad] (nan without
    it)."""
    covers_start, end_gap = decide.before_jump_coverage(geometry, settings)
    fastest = decide.fastest_episode(geometry, settings)
    return {
        'rec_unphysical': 0.0,
        'rec_rule': decision.rule,
        # the angle decision's own rule (`AngleDecision.rule`: G5a, G5w, G6, ...); rec_rule is the status leaf, G6 for
        # every insufficient_time row whatever rule set the word
        'geo_angle_rule': angle.rule,
        'geo_tp_half_amp_med_deg': math.degrees(decide.tp_half_amplitude_median(geometry)),
        'geo_n_librating_episodes': len(decide.librating_episodes(geometry, settings)),
        'geo_n_episodes_before_jump': len(decide.librating_episodes(geometry, settings, before_jump=True)),
        'geo_before_jump_covers_start': covers_start,
        'geo_before_jump_end_gap_P': end_gap,
        'sub_members_locked': decide.members_locked(angle, member),
        'kin_cycle_secular_offset': decide.secular_period_offset(decide.decisive_period(angle, geometry), kinematics),
        'kin_centre_ripple_share': decide.ripple_share(angle, geometry, settings),
        **decide.forced_wave_columns(geometry),
        'kin_drift_word': float(settings.mmr_kinematic_drift_word) if decision.rule == decide.RULE_K2 else NAN,
        'rsp_identity_slope_offset': abs(response.identity_slope_ratio - 1),
        'geo_near_episode_n_tp': decide.near_episode_tp(geometry, settings),
        **decide.single_leg_columns(geometry, settings),  # W28 C6: the 1:1 leg that decides leaf G4c
        'geo_episode_period_min_yr': fastest.period_med_yr if fastest is not None else NAN,
        **centre_travel_columns(geometry, settings),
        **decide.union_columns(geometry, settings),
        'rsp_centre_rate_over_err': decide.centre_rate_significance(response),
        'flt_fast_residual_deg': math.degrees(fast_residual_rad),
        'smp_fast_over_half_amp': fast_residual_rad / decide_half if np.isfinite(decide_half) and decide_half > 0 else NAN,
        'rsp_response_cutoff_yr': response_module.response_cutoff_yr(prep.cutoff.cutoff_yr, geometry.period_cycle_yr, settings),
        **sensitivity.to_columns(),
        'ep_n_tp_cut_half': counts.get('half'),
        'ep_n_tp_cut_double': counts.get('double'),
    }


def _flags(sampling, prep, geometry, response, energy, strength, subresonances, owner, decision) -> List[str]:
    """The verdict's flags: those of the steps (energy, subresonance scan and owner search only where they are verdict
    flags) and the decision's. implausible_strength only where the verdict claims a libration of this
    angle's own force (requests-w3-physics: on a circulation or an induced libration it says nothing new; the
    str_implausible column keeps it everywhere)."""
    flags = list(sampling.flags) + list(prep.flags) + list(geometry.flags) + list(response.flags) + list(decision.flags)
    for step in (energy, subresonances, owner):
        if step is not None:
            flags += [flag for flag in step.flags if flag in VERDICT_FLAGS]
    if subresonances is not None and subresonances.undetermined:
        flags.append(str(Flag.SUBRESONANCE_UNDETERMINED))
    if subresonances is not None:  # A6b: an episode whose own window could not decide (the caps read it too)
        flags += [flag for episode in subresonances.episode_undecided.values() for flag in episode if flag in VERDICT_FLAGS]
    if strength is not None and strength.implausible and decide.own_dynamics_claimed(decision):
        flags.append(str(Flag.IMPLAUSIBLE_STRENGTH))
    return list(dict.fromkeys(flags))
