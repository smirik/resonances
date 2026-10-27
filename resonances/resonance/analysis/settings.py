"""Every number of the libration analysis, as one frozen dataclass.

`AnalysisSettings` is what the pure analysis modules receive; `DryRunSettings` is read by the
dry-run only. Each field is a setting of `SimulationConfig` with the same (lowercase) name and
an upper-case key in `resonances/.env.dist`; the field metadata holds that key, the parser, the
unit and one line of physical meaning (the source of the .env.dist comments and docs/config.md).

Naming: a rule shared by every resonance type has a shared name (`LIBRATION_*`, `FILTER_*`,
`SEPARATRIX_*`, `CONFIDENCE_*`); MMR-only rules have `MMR_`, secular `SECULAR_`, ZLK `ZLK_`.
`convention=True` marks agreed thresholds (not derived from physics): a decision closer than
`confidence_thin_band` to one of them is a thin margin U (see `ledger.py`).

This module imports nothing from Body/Simulation: `settings_from_config` duck-types its source.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import MISSING, Field, asdict, dataclass, field, fields
from typing import Any, Mapping, Optional

KINDS = ('int', 'float', 'bool', 'str', 'floats', 'strs', 'optional_float')


def _setting(default, env: str, kind: str, meaning: str, *, unit: str = '', convention: bool = False):
    """A dataclass field carrying its .env key, parser kind, unit, meaning and convention flag."""
    return field(default=default, metadata={'env': env, 'kind': kind, 'meaning': meaning, 'unit': unit, 'convention': convention})


PI = math.pi
OWNER_PLANETS_DEFAULT = ('Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune')
RESAMPLE_STEPS_DEFAULT = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0)
AUTO_STEPS_DEFAULT = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)


@dataclass(frozen=True)
class AnalysisSettings:
    """All thresholds and conventions of the libration analysis (SPEC §8 with the 2026-10-15 delta)."""

    # --- Filter (shared) and its MMR cutoff rule ---------------------------------------------------
    filter_order: int = _setting(
        2,
        'FILTER_ORDER',
        'int',
        'Butterworth order N of the sos low-pass; filtfilt gain G(P) = 1/(1 + (tan(pi dt/P)/tan(pi dt/P_c))^(2N)) '
        '(-> 1/(1 + (P_c/P)^(2N)) for P_c >> dt)',
    )
    filter_edge_trim_cutoffs: float = _setting(
        1.0,
        'FILTER_EDGE_TRIM_CUTOFFS',
        'float',
        'turning points, passes and portrait use [t0 + x P_c, t1 - x P_c]: filtfilt edges spread ~ +-P_c',
    )
    filter_min_libration_cutoffs: float = _setting(
        2.0,
        'FILTER_MIN_LIBRATION_CUTOFFS',
        'float',
        'the floor term of the cutoff is the shortest libration / this (MMR: P_floor, secular/ZLK: P_min): where it sets P_c, '
        'that libration keeps G = 1/(1 + this^-4) = 0.94; secular/ZLK: a measured libration period below this many P_c sets '
        'filter_too_slow (MMR: G(P_cyc) below MMR_L8_GAIN_FULL); shape checks become not_measured',
    )
    filter_grid_rtol: float = _setting(
        1e-3,
        'FILTER_GRID_RTOL',
        'float',
        'a time grid is uniform for the filter when every step is within this fraction of the median step (REBOUND rounding)',
    )
    mmr_filter_synodic_factor: float = _setting(
        1.5,
        'MMR_FILTER_SYNODIC_FACTOR',
        'float',
        'P_c >= this x the longest synodic period with the planets of the angle (leak <= 0.5 deg)',
    )
    mmr_libration_period_min_yr: float = _setting(
        150.0,
        'MMR_LIBRATION_PERIOD_MIN_YR',
        'float',
        "shortest libration period of a non-co-orbital MMR, a convention (not a physical minimum: Hildas ~260 yr; the "
        "forced waves of 52356 and 23240 are 115-121 yr): faster 'episodes' are forced waves and the TP prominence covers "
        'what sigma_f holds faster than this (0 = off); co-orbitals use MMR_COORBITAL_PERIOD_FLOOR_FACTOR',
        unit='yr',
        convention=True,
    )
    mmr_coorbital_period_floor_factor: float = _setting(
        0.5,
        'MMR_COORBITAL_PERIOD_FLOOR_FACTOR',
        'float',
        'a co-orbital angle (1:1 with its planet) librates no faster than this x the linear tadpole period '
        'P_orb / sqrt(27 mu / 4) of the planet (Jupiter 148 yr, Earth 222 yr, Mars 1274 yr); wide horseshoes are faster '
        '(Earth horseshoes of the benchmark down to 0.65 of it); its P_c is the floor / FILTER_MIN_LIBRATION_CUTOFFS (0 = off)',
        convention=True,
    )
    mmr_coorbital_qs_floor: bool = _setting(
        True,
        'MMR_COORBITAL_QS_FLOOR',
        'bool',
        'rule G1q: a prograde 1:1 angle whose orbit-averaged angle stays inside the quasi-satellite well of the averaged '
        'potential for >= LIBRATION_MIN_TURNING_POINTS / 2 of its small-amplitude periods P_QS, and that has no libration '
        'over the record, is analysed again with the floor MMR_COORBITAL_PERIOD_FLOOR_FACTOR x P_QS (episodes centred '
        'outside the well keep the tadpole floor); kept when that pass finds a libration in the well (2010 CG236: 8 yr; '
        'W20 F1)',
        convention=True,
    )
    mmr_coorbital_qs_well_share: float = _setting(
        0.5,
        'MMR_COORBITAL_QS_WELL_SHARE',
        'float',
        'rule G1q: an episode counts as centred in the quasi-satellite well (and takes its floor) only where the averaged '
        "potential has that well on at least this share of the episode's samples; otherwise it keeps the tadpole floor",
        convention=True,
    )
    mmr_coorbital_qs_max_blocks: int = _setting(
        4000,
        'MMR_COORBITAL_QS_MAX_BLOCKS',
        'int',
        "rule G1q (cost): the 1:1 force law of the quasi-satellite pass reads the elements in blocks of P_QS,min (not longer "
        "than the angle's own P_c), but in no more than this many over the record (longer blocks then): each block may "
        'need its own table of the averaged potential (2010 CG236 over 20 kyr: 3774 blocks of 5.3 yr, 45 s; 1000 blocks of 20 '
        'yr fail its energy integral, R^2 0.11)',
    )
    mmr_filter_guard_tp_abs: int = _setting(
        2,
        'MMR_FILTER_GUARD_TP_ABS',
        'int',
        'x2 guard: the TP counts at P_c and 2 P_c on their common interval may differ by this many (one TP at each end) '
        'or by MMR_FILTER_GUARD_TP_REL of the larger count; more is a change (filter_sensitive)',
        convention=True,
    )
    mmr_filter_guard_tp_rel: float = _setting(
        0.1,
        'MMR_FILTER_GUARD_TP_REL',
        'float',
        'x2 guard: relative tolerance of the TP counts (see MMR_FILTER_GUARD_TP_ABS)',
        convention=True,
    )
    mmr_filter_guard_boundary_periods: float = _setting(
        1.0,
        'MMR_FILTER_GUARD_BOUNDARY_PERIODS',
        'float',
        'x2 guard: the decisive episode (else any accepted episode) whose start or end moves by more than this many '
        'libration periods is a change',
        convention=True,
    )
    mmr_filter_guard_h_tol: float = _setting(
        0.1,
        'MMR_FILTER_GUARD_H_TOL',
        'float',
        'x2 guard: the median cycle energy h (separatrix 1) changing by more than this x max(1, the smaller |h|), or across '
        '1, is a change (relative far above the separatrix: h ~ 1e4-1e7 of a far circulation moves by < 1 %)',
        convention=True,
    )
    mmr_filter_guard_amp_rel: float = _setting(
        0.2,
        'MMR_FILTER_GUARD_AMP_REL',
        'float',
        'x2 guard: the largest half-amplitude of the decisive episode against its counterpart at 2 P_c (else of the '
        'episode with the most TPs) changing by more than this fraction is a change; energy and shape are compared only '
        'when the gain at P_cyc of the 2 P_c filter is >= MMR_L8_GAIN_FULL (else not checked)',
        convention=True,
    )
    mmr_filter_cutoff_yr: Optional[float] = _setting(
        None, 'MMR_FILTER_CUTOFF_YR', 'optional_float', 'explicit cutoff period overriding the rule (empty = rule)', unit='yr'
    )
    mmr_ripple_remove: bool = _setting(
        True, 'MMR_RIPPLE_REMOVE', 'bool', 'subtract sum_j k_j dlambda_j (great inequality) from the angle when planets are saved'
    )
    mmr_ripple_band_yr: tuple = _setting(
        (300.0, 3000.0),
        'MMR_RIPPLE_BAND_YR',
        'floats',
        'period band of the diagnostic correlation of the subtracted ripple with the angle (flt_ripple_corr)',
        unit='yr',
    )
    mmr_ripple_reconstruction_max_deg: float = _setting(
        1.0,
        'MMR_RIPPLE_RECONSTRUCTION_MAX_DEG',
        'float',
        'the ripple is subtracted when the saved angle equals the combination of the saved longitudes within this',
        unit='deg',
    )

    # --- Sampling and predictive unwrap (MMR) ----------------------------------------------------
    mmr_unwrap: str = _setting(
        'predictive', 'MMR_UNWRAP', 'str', "'predictive' = unwrap by the rate model from a; 'numpy' = np.unwrap (diagnostic)"
    )
    mmr_unwrap_max_resid_p99_deg: float = _setting(
        45.0,
        'MMR_UNWRAP_MAX_RESID_P99_DEG',
        'float',
        '99th percentile of the per-step model error of the lambda part for a stitchable record',
        unit='deg',
    )
    mmr_unwrap_max_resid_deg: float = _setting(
        135.0, 'MMR_UNWRAP_MAX_RESID_DEG', 'float', 'largest single per-step model error of the lambda part (pi/4 below pi)', unit='deg'
    )
    mmr_unwrap_suspect_resid_deg: float = _setting(
        90.0, 'MMR_UNWRAP_SUSPECT_RESID_DEG', 'float', 'a step with |residual| above this is unwrap_step_suspect', unit='deg'
    )
    mmr_varpi_ref_window_yr: float = _setting(
        5000.0, 'MMR_VARPI_REF_WINDOW_YR', 'float', 'window of the varpi slope used as the slow term of the rate model', unit='yr'
    )
    mmr_jitter_max_rad: float = _setting(
        0.785,
        'MMR_JITTER_MAX_RAD',
        'float',
        '|p| de_sp/e above this (pi/4) = jitter_low_e (the osculating wobble of p varpi: a flag and a cap <= medium); '
        'undefined_phase also needs varpi itself undefined (MMR_VARPI_UNDEFINED_MAX_SHARE); locally |p| s/|z_f| = '
        'phase_undefined_local',
        unit='rad',
    )
    mmr_varpi_undefined_max_share: float = _setting(
        0.5,
        'MMR_VARPI_UNDEFINED_MAX_SHARE',
        'float',
        "undefined_phase needs varpi's own (p = 1) local scatter s/|z_f| above MMR_JITTER_MAX_RAD on more than this "
        'share of the measured record (not measured: undefined_phase as before, jitter_low_e and the full angle failing)',
        convention=True,
    )
    mmr_jitter_band_yr: float = _setting(
        50.0,
        'MMR_JITTER_BAND_YR',
        'float',
        'periods shorter than this make the short-period e scatter de_sp; the band of the slow vector z_f = <e exp(i varpi)>',
        unit='yr',
    )
    mmr_step_resolved_deg: float = _setting(90.0, 'MMR_STEP_RESOLVED_DEG', 'float', 'step verdict resolved: |nu| dt <= this', unit='deg')
    mmr_far_from_res_factor: float = _setting(
        10.0,
        'MMR_FAR_FROM_RES_FACTOR',
        'float',
        'far_from_resonance: |<a> - a*| >= this x the a scale of the record, max(half the a range, |a(lambda_dot) - <a>|): '
        'a libration about a* needs the slow a to reach a* (no strength enters: mu e^q is not a bound)',
    )
    mmr_axis_range_percentile: float = _setting(
        1.0,
        'MMR_AXIS_RANGE_PERCENTILE',
        'float',
        'the a range of the rate model (nu_lo, nu_hi, alias branches, far ratio) is the [x, 100 - x] percentiles of a',
        unit='%',
    )
    mmr_planet_unwrap_max_step_deg: float = _setting(
        90.0,
        'MMR_PLANET_UNWRAP_MAX_STEP_DEG',
        'float',
        "a saved planet's n_j is the slope of its unwrapped lambda_j only when no step errs from the table rate by more than "
        'this; otherwise the table value',
        unit='deg',
    )
    mmr_alias_numpy_rev_tol: float = _setting(
        0.5, 'MMR_ALIAS_NUMPY_REV_TOL', 'float', 'alias_np_unwrap_differs: predictive and np.unwrap revolutions differ by more', unit='rev'
    )
    mmr_aliased_line_harmonics: int = _setting(
        3, 'MMR_ALIASED_LINE_HARMONICS', 'int', 'harmonics m of synodic waves checked for folding into [P_cyc/2, 2P_cyc]'
    )
    mmr_aliased_line_band_factor: float = _setting(
        2.0, 'MMR_ALIASED_LINE_BAND_FACTOR', 'float', 'a folded synodic line is reported when its period is within [P_cyc/x, x P_cyc]'
    )
    mmr_resample_steps_yr: tuple = _setting(
        RESAMPLE_STEPS_DEFAULT,
        'MMR_RESAMPLE_STEPS_YR',
        'floats',
        'round values a resample step (and smp_fold_free_step_yr) is chosen from',
        unit='yr',
    )
    mmr_resample_steps_per_cycle: float = _setting(
        20.0,
        'MMR_RESAMPLE_STEPS_PER_CYCLE',
        'float',
        'recommended step <= P_cyc / this (an undersampled record and the dry-run: the shortest libration P_floor / this)',
    )
    mmr_extend_min_span_factor: float = _setting(
        2.0,
        'MMR_EXTEND_MIN_SPAN_FACTOR',
        'float',
        'an extension asks for at least this x the record length T (the total length from the start epoch)',
    )
    mmr_next_check_span_factor: float = _setting(
        4.0,
        'MMR_NEXT_CHECK_SPAN_FACTOR',
        'float',
        'insufficient_time/extend: the next check is round_up(max(f_min T, min(N, this x T))), N the estimate from the '
        'period (base_extend_to_years); a budget, not a sufficiency estimate: the record is analysed again after it (N8)',
        convention=True,
    )

    # --- Geometry (shared) ---------------------------------------------------------------------
    libration_min_turning_points: int = _setting(
        3,
        'LIBRATION_MIN_TURNING_POINTS',
        'int',
        'turning points of one full cycle: an episode needs this many; a secular moving centre too',
    )
    libration_near_episode_tp_deficit: int = _setting(
        1,
        'LIBRATION_NEAR_EPISODE_TP_DEFICIT',
        'int',
        'a run that would be a librating episode but for its count (half-cycles below pi, no over-the-top neighbour, '
        'period not below the shortest libration) and lacks at most this many turning points is a near-episode: a '
        'verdict without libration gets flag near_episode, confidence <= medium; 0 = off',
    )
    libration_coverage_periods: float = _setting(
        1.0,
        'LIBRATION_COVERAGE_PERIODS',
        'float',
        'an episode covers a record edge when it starts (ends) within this many periods of it, or when no turning point is '
        'missing at the edge of the analysis window (that gap not longer than the longest TP-to-TP step of the episode, '
        'nor than this many periods) and the stretch the filter trims off that edge is within this many periods',
    )
    libration_union_gap_periods: float = _setting(
        1.0,
        'LIBRATION_UNION_GAP_PERIODS',
        'float',
        'MMR: two consecutive librating episodes join into one libration over the record (G2, `covering_union`) only '
        'when the gap between them is at most this x (P_left + P_right): each side misses at most its own period, the '
        'symmetric reading of LIBRATION_COVERAGE_PERIODS at an edge (W22 union rule, author D1b = a: 2002 XA 5J-2, '
        '1.05, stays two episodes)',
        convention=True,
    )
    libration_episode_gap_factor: float = _setting(
        3.0, 'LIBRATION_EPISODE_GAP_FACTOR', 'float', 'a gap between turning points above this x the median half-period splits a run'
    )
    libration_regular_period_ratio: float = _setting(
        2.0,
        'LIBRATION_REGULAR_PERIOD_RATIO',
        'float',
        'a run whose centre moves is judged again on its longest stretch of full periods <= this x their median (a slow '
        'edge cycle is a change of regime, not part of the stationary libration), and may drop one edge half-cycle '
        'slower than this x the median half-period',
    )
    libration_edge_arc_rev: float = _setting(
        1.0,
        'LIBRATION_EDGE_ARC_REV',
        'float',
        'MMR: an exit (entry) is observed, not suspected, when the orbit-averaged raw angle runs at least this many turns '
        'away from the centre from the last turning point of the last episode to the record end (from the record start '
        'to the first one) and moves monotonically there; that edge is then not covered (an arc > 2 pi is not a piece of '
        'libration; author 2026-10-19, variant a)',
        unit='rev',
        convention=True,
    )
    libration_return_tol: float = _setting(
        0.05,
        'LIBRATION_RETURN_TOL',
        'float',
        'MMR: two opposite turning points are one closed cycle when sigma_f crosses their mid-level before the first and '
        'after the second in the same direction, a returning to within this x its swing there, the rate to within this '
        'fraction, and sigma_f crossing the mid-level between them as fast as at the first crossing, the other way '
        '(closure by phase return; 2016 CA138: 0.08 %, 0.2 %, 0.1 %)',
        convention=True,
    )
    libration_return_period_tol: float = _setting(
        0.25,
        'LIBRATION_RETURN_PERIOD_TOL',
        'float',
        'MMR: the phase return also needs (t_b - t_a) / (2 x the time between the two turning points) within 1 +- this '
        '(the quarter-period symmetry of a cycle; a circulation creeping back between two turns gives ~0.5; 2016 '
        'CA138: 1.00)',
        convention=True,
    )
    libration_return_identity_corr_min: float = _setting(
        0.9,
        'LIBRATION_RETURN_IDENTITY_CORR_MIN',
        'float',
        'MMR: the phase return also needs corr(sigma_dot, sign(c) (a - mean a)) >= this between the two crossings (a '
        'kinematic back-swing with a flat a is no cycle); the value of MMR_IDENTITY_CORR_MIN',
    )
    libration_return_levels: int = _setting(
        19,
        'LIBRATION_RETURN_LEVELS',
        'int',
        'MMR: the phase return of two opposite turning points is tried at the mid-level and at the levels k / (n + 1) of '
        'their swing, k = 1..n, with the same conditions, a level counting only inside a band of closing levels '
        '(LIBRATION_RETURN_MID_BAND_LEVELS for the mid-level, LIBRATION_RETURN_BAND_LEVELS for another); the mid-level is '
        'kept when it counts, else the best match (the mid-level is a convention, any level is a section the cycle crosses '
        'twice; 164207 never crosses its mid-level before the horseshoe turn); 0 or 1 = the mid-level only, alone',
        convention=True,
    )
    libration_return_band_levels: int = _setting(
        2,
        'LIBRATION_RETURN_BAND_LEVELS',
        'int',
        'MMR: a level other than the mid-level closes a cycle by its phase return only inside at least this many '
        'consecutive closing levels of LIBRATION_RETURN_LEVELS (the mismatches change continuously with the level, a real '
        'return holds over a band of sections; 2025 RB7 at 1 kyr closes at 0.20-0.25, 164207 at 0.15-0.20)',
        convention=True,
    )
    libration_return_mid_band_levels: int = _setting(
        2,
        'LIBRATION_RETURN_MID_BAND_LEVELS',
        'int',
        'MMR: the mid-level closes a cycle by its phase return inside at least this many consecutive closing levels of '
        'LIBRATION_RETURN_LEVELS (2: one closing neighbour), counted over the observable levels only: a level whose '
        'crossing may lie beyond the record neither closes nor breaks the band, so a mid-level without an observable '
        'neighbour closes alone (W20 round D: a 400-yr cosine on -104..304 yr); 1 = the mid-level alone',
        convention=True,
    )
    libration_return_cycle_caps_medium: bool = _setting(
        True,
        'LIBRATION_RETURN_CYCLE_CAPS_MEDIUM',
        'bool',
        'MMR: a verdict that rests on a cycle closed only by its phase return (two turning points) has confidence <= '
        'medium (cap one_return_cycle; author question q1, W20)',
        convention=True,
    )
    libration_regime_cuts: bool = _setting(
        True,
        'LIBRATION_REGIME_CUTS',
        'bool',
        'MMR: a run whose centre moves is cut at a centre step between two measured regimes or at a stall of its slow '
        'centre (F2, K3; 524522 1V-1, 385266 16N-27); off: such a run stays one rejected run, as before W20',
        convention=True,
    )
    libration_stall_min_cycles: int = _setting(
        3,
        'LIBRATION_STALL_MIN_CYCLES',
        'int',
        'MMR: a run whose centre moves is cut at a stall of its slow centre: at least this many independent full cycles '
        'whose midpoints move less per period than the cycles swing (385266 16N-27, 55-95 kyr); a steady drift has none',
        convention=True,
    )
    libration_centre_cycles: float = _setting(
        3.0, 'LIBRATION_CENTRE_CYCLES', 'float', 'the centre is the angle smoothed over this many cycles'
    )
    libration_centre_tp_prominence_deg: float = _setting(
        10.0,
        'LIBRATION_CENTRE_TP_PROMINENCE_DEG',
        'float',
        'floor of the prominence of a centre turn (4x the centre noise A/82)',
        unit='deg',
    )
    libration_centre_tp_factor: float = _setting(
        2.0, 'LIBRATION_CENTRE_TP_FACTOR', 'float', 'a centre turn also needs this x the turning-point prominence'
    )
    libration_centre_scale_tp_at_centre: bool = _setting(
        True,
        'LIBRATION_CENTRE_SCALE_TP_AT_CENTRE',
        'bool',
        'MMR G2c: the analysis at the scale of the centre takes its turning points at the centre-turn prominence p_c '
        '(MMR_TP_PROMINENCE_FLOOR_DEG raised to p_c there): the turns the precheck that qualified the scale reads: a swing '
        'below p_c is no turn of the centre there (residue of the fast component W_c attenuates, or a libration too small '
        'for this scale, which the fine scale judges) (W23 rule a; 2003 TX9 3E+2J-10); off: the floor of the fine scale, '
        'as before W23',
    )
    libration_centre_returns_modulated: int = _setting(
        3, 'LIBRATION_CENTRE_RETURNS_MODULATED', 'int', 'a bounded centre turning back this many times is modulated, not drift'
    )
    libration_centre_min_estimates: int = _setting(
        2,
        'LIBRATION_CENTRE_MIN_ESTIMATES',
        'int',
        'a stationary centre needs this many independent full cycles (no shared half-cycle; every band overlapping); '
        'with fewer (3 TPs = one cycle, one centre) the episode is accepted but its centre is not measured (L6 thin)',
    )
    libration_modulated_range_max_rad: float = _setting(
        PI, 'LIBRATION_MODULATED_RANGE_MAX_RAD', 'float', 'modulated: range of the centre below this', unit='rad'
    )
    libration_slip_tolerance: float = _setting(
        0.2, 'LIBRATION_SLIP_TOLERANCE', 'float', 'a slip is a step of 2 pi (1 +- this) of the oscillation midpoint', convention=True
    )
    libration_slip_min_duration_t0: float = _setting(
        0.5,
        'LIBRATION_SLIP_MIN_DURATION_T0',
        'float',
        'a slip faster than this x T0 is an unwrap error (the angle cannot pass the top faster)',
    )
    libration_pattern_a_min_slips: int = _setting(2, 'LIBRATION_PATTERN_A_MIN_SLIPS', 'int', 'pattern A needs at least this many slips')
    libration_pattern_a_min_amp_deg: float = _setting(
        60.0,
        'LIBRATION_PATTERN_A_MIN_AMP_DEG',
        'float',
        'without a measured zone, pattern-A pieces need this amplitude (J/J_s ~ 0.3)',
        unit='deg',
        convention=True,
    )
    libration_stall_phase_tol_deg: float = _setting(
        30.0, 'LIBRATION_STALL_PHASE_TOL_DEG', 'float', 'pattern B: rate minimum within this of the potential top', unit='deg'
    )
    libration_stall_rev_fraction: float = _setting(
        0.75,
        'LIBRATION_STALL_REV_FRACTION',
        'float',
        'pattern B: this fraction of revolutions has the rate minimum at the top',
        convention=True,
    )
    libration_stall_rate_tol: float = _setting(
        0.2, 'LIBRATION_STALL_RATE_TOL', 'float', '|r_min - r_pend(h)| above this = stall_rate_inconsistent (flag only)'
    )
    libration_drift_min_window_fraction: float = _setting(
        0.5, 'LIBRATION_DRIFT_MIN_WINDOW_FRACTION', 'float', 'drift: rejected centre_moves runs cover at least this fraction of the window'
    )
    libration_circulation_net_rev: float = _setting(
        2.0,
        'LIBRATION_CIRCULATION_NET_REV',
        'float',
        'G4b: the angle circulates when the slow centre advances by at least this many revolutions net over the window '
        '(|c(t1) - c(t0)| / 2pi), even when its largest monotone pass stays below one (a staircase with small step-backs); '
        'also in the angle word of K2 (W22 R-NET: 142694 1V+3S-7 200 kyr, 2.75 rev, the author\'s circulation; '
        '21028 7E-10M-6, 1.81 rev, his drift)',
        convention=True,
    )
    libration_jump_noise_factor: float = _setting(
        8.0, 'LIBRATION_JUMP_NOISE_FACTOR', 'float', 'a jump candidate exceeds this many robust std of the slow-variable increments'
    )
    libration_jump_dh_min: float = _setting(
        0.1, 'LIBRATION_JUMP_DH_MIN', 'float', 'a jump changes the cycle energy h by more than max(2 sigma_h, this)'
    )
    libration_jump_cycles: int = _setting(2, 'LIBRATION_JUMP_CYCLES', 'int', 'cycles compared on each side of a jump candidate')
    # --- G2a: a step of the libration action inside one accepted run (MMR, not 1:1; W24, 2003 TX9 3E+2J-10) ------------
    libration_action_step: bool = _setting(
        True,
        'LIBRATION_ACTION_STEP',
        'bool',
        'G2a (MMR, not 1:1): an accepted run whose half-amplitudes step between two separated bands, with the swing of a '
        'and the cycle energy changing the same way within one half-cycle, is cut there as at a confirmed jump of a '
        '(an impulse: a close encounter puts the orbit on another energy level; 2003 TX9 3E+2J-10 at the Mars '
        'encounters of 6.5-6.9 kyr); off: such a run stays one episode, as before W24',
        convention=True,
    )
    libration_action_step_min_half_cycles: int = _setting(
        2,
        'LIBRATION_ACTION_STEP_MIN_HALF_CYCLES',
        'int',
        'G2a: half-cycles on each side of the transitional half-cycle (2 = one closed cycle)',
    )
    libration_action_step_separation: float = _setting(
        2.0,
        'LIBRATION_ACTION_STEP_SEPARATION',
        'float',
        'G2a: the two bands of half-amplitudes are separated: the smallest on the larger side over the largest on the '
        'smaller side is at least this (an adiabatic decay needs an e-folding of 1.5 cycles to reach 2 across one '
        'excluded half-cycle)',
    )
    libration_action_step_ratio: float = _setting(
        3.0, 'LIBRATION_ACTION_STEP_RATIO', 'float', 'G2a: ratio of the median half-amplitudes of the two sides'
    )
    libration_action_step_a_ratio: float = _setting(
        1.5,
        'LIBRATION_ACTION_STEP_A_RATIO',
        'float',
        'G2a: the swing of a (std of a_f on each side at the record\'s cutoff) changes the same way by at least this '
        'factor (delta a ~ sigma_dot; 74729 1J+6S-1: a sigma step that a does not show, ratio 0.37)',
    )
    libration_action_step_cycles: int = _setting(
        1,
        'LIBRATION_ACTION_STEP_CYCLES',
        'int',
        'G2a: cycles of the energy analysis on each side that confirm the step (|dh| > max(2 sigma_h, '
        'LIBRATION_JUMP_DH_MIN), the sign of the amplitude change); fewer than LIBRATION_JUMP_CYCLES on a side makes '
        'the split provisional (cap action_step_provisional, confidence <= medium)',
    )
    libration_action_step_period_tol: float = _setting(
        0.2,
        'LIBRATION_ACTION_STEP_PERIOD_TOL',
        'float',
        'G2a test 7: every disjoint cycle of both sides has T_obs/T_pred within this of 1, T_pred the period of the loop '
        'of the cycle\'s own energy, in the well that holds it, of V(sigma) fitted (as V_E) on the cycles of the larger '
        'swings (held out: each of them in V of the others); and the two sides\' median periods differ by more than '
        'this (else one forcing frequency explains both: in a nearly harmonic V every loop has one period). As '
        'SEPARATRIX_PERIOD_TOL for R0',
    )
    libration_action_step_tp_tol: float = _setting(
        0.2,
        'LIBRATION_ACTION_STEP_TP_TOL',
        'float',
        'G2a test 7: the observed extremes of sigma of every cycle lie within this x its half-swing of the turning points '
        'of the loop of its energy (a swing off the minimum of V is no libration in V). In the harmonic limit the half-'
        'width at a given energy scales as the period, so this equals LIBRATION_ACTION_STEP_PERIOD_TOL',
    )
    libration_action_step_energy_drift: float = _setting(
        0.5,
        'LIBRATION_ACTION_STEP_ENERGY_DRIFT',
        'float',
        'G2a test 7: on each side the cycle energies in V spread by at most this x the step of the energy between the '
        'sides (between impulses the energy of a libration stays put; at 0.5 the two bands are apart by at least half '
        'the step, as the half-amplitude bands of LIBRATION_ACTION_STEP_SEPARATION)',
    )
    libration_action_step_cutoffs: tuple = _setting(
        (0.075, 0.1, 0.15),
        'LIBRATION_ACTION_STEP_CUTOFFS',
        'floats',
        'G2a test 7 holds at every shape cutoff C = f x the shorter side period of this list, with the same number of '
        'wells (the numbers reported at the middle one): a factor 2, lighter than SEPARATRIX_CUTOFF_CYCLES, since the '
        'test reads turning points and a second well, which live in the high harmonics of a swing across a barrier (a '
        'double well integrated and kicked: T_obs/T_pred 0.994 at 0.1, 0.911 at 0.2, the second well lost at 0.3); the '
        'rate\'s noise sets the floor (V_E R^2 >= SEPARATRIX_VE_R2_MIN). A well or a law one cutoff alone gives is the '
        'filter\'s',
    )
    libration_action_step_max_periods: float = _setting(
        1.0,
        'LIBRATION_ACTION_STEP_MAX_PERIODS',
        'float',
        'G2a: the transitional half-cycle lasts at most this many of the shorter median full period of the two sides',
    )
    libration_jump_tau_cutoffs: float = _setting(2.0, 'LIBRATION_JUMP_TAU_CUTOFFS', 'float', 'jump locator baseline tau >= this x P_c')
    libration_jump_tau_steps: float = _setting(
        5.0, 'LIBRATION_JUMP_TAU_STEPS', 'float', 'jump locator baseline tau >= this x the output step'
    )
    libration_jump_edge_periods: float = _setting(
        1.0,
        'LIBRATION_JUMP_EDGE_PERIODS',
        'float',
        'a jump candidate needs max(tau, this x P_cyc) of the window on each side to be confirmed; closer to an edge it '
        'is kept as jump_at_edge and cuts nothing',
    )
    mmr_tp_prominence_floor_deg: float = _setting(
        1.0, 'MMR_TP_PROMINENCE_FLOOR_DEG', 'float', 'turning-point prominence floor: twice the synodic leak after the filter', unit='deg'
    )
    mmr_tp_ripple_factor: float = _setting(
        4.0,
        'MMR_TP_RIPPLE_FACTOR',
        'float',
        'prominence >= this x the unremoved ripple half-amplitude or the part of sigma_f faster than MMR_LIBRATION_PERIOD_MIN_YR '
        '(95th percentile, the image of the libration runs through the two filters taken out first; a wave of half-amplitude r '
        'gives turns <= 2r)',
    )

    # --- Separatrix energy (shared) and the numeric zone (MMR) -----------------------------------
    separatrix_harmonics_max: int = _setting(2, 'SEPARATRIX_HARMONICS_MAX', 'int', 'harmonics of V(sigma): asymmetric external 1:N need 2')
    separatrix_harmonic_gain: float = _setting(
        0.2, 'SEPARATRIX_HARMONIC_GAIN', 'float', 'the second harmonic is kept when R^2 of V_E grows by this'
    )
    separatrix_period_tol: float = _setting(
        0.2, 'SEPARATRIX_PERIOD_TOL', 'float', 'period-energy law: |median T_obs/T_pend - 1| <= this (pendula 0.9-1.1, kinematics 0.61)'
    )
    separatrix_phase_coherence: float = _setting(
        0.9,
        'SEPARATRIX_PHASE_COHERENCE',
        'float',
        'fixed well: coherence of the stable point over windows (potential 0.95-0.99, kinematics 0.71)',
    )
    separatrix_vevf_r_min: float = _setting(
        0.9, 'SEPARATRIX_VEVF_R_MIN', 'float', 'R0 with the slow variable: correlation of V_E and V_F in a window'
    )
    separatrix_vevf_fraction: float = _setting(
        0.666, 'SEPARATRIX_VEVF_FRACTION', 'float', 'R0 with the slow variable: fraction of windows where V_E ~ V_F'
    )
    separatrix_sigma_margin: float = _setting(
        2.0, 'SEPARATRIX_SIGMA_MARGIN', 'float', 'margin in sigma_h when a cycle is put into a zone class'
    )
    separatrix_sigma_floor: float = _setting(0.02, 'SEPARATRIX_SIGMA_FLOOR', 'float', 'floor of the cycle-to-cycle energy scatter sigma_h')
    separatrix_trend_sigma: float = _setting(
        3.0, 'SEPARATRIX_TREND_SIGMA', 'float', 'trend mechanism: slope of J/J_s over cycles significant at this many sigma'
    )
    separatrix_permutation_p_max: float = _setting(
        0.05, 'SEPARATRIX_PERMUTATION_P_MAX', 'float', 'pulsation mechanism: cyclic-shift p-value below this', convention=True
    )
    separatrix_layer_adiabaticity_min: float = _setting(
        0.5,
        'SEPARATRIX_LAYER_ADIABATICITY_MIN',
        'float',
        "layer mechanism: crossings at an adiabaticity lambda above this are the chaotic layer's own (no adiabatic invariant)",
    )
    separatrix_near_zone_widths: float = _setting(
        1.5, 'SEPARATRIX_NEAR_ZONE_WIDTHS', 'float', 'near_separatrix_energy flag: revolutions within this many zone widths above the zone'
    )
    mmr_exit_and_capture_lifts_unexplained: bool = _setting(
        True,
        'MMR_EXIT_AND_CAPTURE_LIFTS_UNEXPLAINED',
        'bool',
        'the mark exit_and_capture lifts the cap transient_unexplained: an angle that left its libration and came back '
        'is the transient the record shows even without a named mechanism (author:2026-10-18)',
        convention=True,
    )
    mmr_coorbital_lifts_unexplained: bool = _setting(
        True,
        'MMR_COORBITAL_LIFTS_UNEXPLAINED',
        'bool',
        'a 1:1 angle gets no cap transient_unexplained: the pendulum separatrix model that names a mechanism does not '
        'apply to it, its crossings are close encounters (2026 FW12; author 2026-10-19, R2)',
        convention=True,
    )
    separatrix_cutoff_cycles: float = _setting(
        0.2,
        'SEPARATRIX_CUTOFF_CYCLES',
        'float',
        'V_E, V_F, the modulation and the zone read sigma and a smoothed at max(P_c, this x P_cyc) (G(P_cyc) 0.998)',
    )
    separatrix_ve_r2_min: float = _setting(
        0.7,
        'SEPARATRIX_VE_R2_MIN',
        'float',
        'R0 is measured only when V_E explains this share of sigma_dot^2 within the cycles',
        convention=True,
    )
    mmr_zone_method: str = _setting(
        'numeric', 'MMR_ZONE_METHOD', 'str', "zone the separatrix reaches: 'numeric' (modulated pendulum), 'eq_only', 'adiabatic'"
    )
    mmr_separatrix_avgpot_tp_max: int = _setting(
        40,
        'MMR_SEPARATRIX_AVGPOT_TP_MAX',
        'int',
        'two-body angle with its planet saved: the energy of each turning point is read in the averaged potential at its '
        'own time, h = (<D>(sigma_TP) - <D>_min) / (<D>_barrier - <D>_min); at most this many evenly chosen turning points '
        'are evaluated (cost: one profile each, ~0.02 s), the others interpolated in time',
    )
    mmr_zone_j_min: float = _setting(0.1, 'MMR_ZONE_J_MIN', 'float', 'smallest initial action J0/<J_s> of the zone grid')
    mmr_zone_j_step: float = _setting(0.025, 'MMR_ZONE_J_STEP', 'float', 'step of the zone grid in J0/<J_s> and in h0')
    mmr_zone_h_max: float = _setting(2.0, 'MMR_ZONE_H_MAX', 'float', 'largest initial energy h0 on the rotation side of the zone grid')
    mmr_zone_phases: int = _setting(16, 'MMR_ZONE_PHASES', 'int', 'start phases per modulation period in the zone grid')
    mmr_zone_rtol: float = _setting(1.0e-9, 'MMR_ZONE_RTOL', 'float', 'relative tolerance of the DOP853 zone trajectories')
    mmr_zone_adiabatic_lambda_max: float = _setting(
        0.1, 'MMR_ZONE_ADIABATIC_LAMBDA_MAX', 'float', 'below this adiabaticity the numeric zone is checked against the adiabatic edge'
    )
    mmr_zone_adiabatic_tol: float = _setting(
        0.05, 'MMR_ZONE_ADIABATIC_TOL', 'float', 'numeric vs adiabatic J_lo differing by more = zone_numeric_mismatch'
    )

    # --- Response of a, kinematics, slow libration (MMR) -----------------------------------------
    mmr_a_response_min_amp_deg: float = _setting(
        45.0,
        'MMR_A_RESPONSE_MIN_AMP_DEG',
        'float',
        'below this amplitude sine and line differ < 10% (A^2/6): harmonic force law not_measured (not the 1:1 law)',
        unit='deg',
        convention=True,
    )
    mmr_force_window_cycles: float = _setting(
        3.5, 'MMR_FORCE_WINDOW_CYCLES', 'float', 'force-law fit window in cycles (phase defined, force ~ constant)'
    )
    mmr_force_window_step: float = _setting(
        0.5, 'MMR_FORCE_WINDOW_STEP', 'float', 'step of the force-law windows as a fraction of the window'
    )
    mmr_force_min_windows: int = _setting(2, 'MMR_FORCE_MIN_WINDOWS', 'int', 'fewer force-law windows = not_measured')
    mmr_force_r2_full: float = _setting(
        0.7, 'MMR_FORCE_R2_FULL', 'float', 'force law passes with R^2 >= this in both halves', convention=True
    )
    mmr_force_r2_fail: float = _setting(
        0.4, 'MMR_FORCE_R2_FAIL', 'float', 'force law fails with R^2 below this under both laws', convention=True
    )
    mmr_force_phase_tol_deg: float = _setting(
        30.0, 'MMR_FORCE_PHASE_TOL_DEG', 'float', 'floor of the expected spread of phi0 (the column rsp_phi0_tol_deg only)', unit='deg'
    )
    mmr_force_multi_gain: float = _setting(
        0.1, 'MMR_FORCE_MULTI_GAIN', 'float', 'the multiplet law is taken when its R^2 exceeds the single one by this'
    )
    mmr_force_coef_coh_min: float = _setting(
        0.8, 'MMR_FORCE_COEF_COH_MIN', 'float', 'multiplet law: coherence of the leading complex coefficients over windows'
    )
    mmr_force_power_share: float = _setting(
        0.8, 'MMR_FORCE_POWER_SHARE', 'float', 'the leading multiplet terms carry this share of the fitted power'
    )
    mmr_force_contamination: str = _setting(
        'lattice',
        'MMR_FORCE_CONTAMINATION',
        'str',
        "'lattice' = a measured harmonic force law is not_measured (contamination) when a forced line of the angle's lattice "
        "lies within 1/(MMR_FORCE_WINDOW_CYCLES P) of 1/P (the fit cannot tell the two apart; it never makes a libration "
        "dynamic); 'off'",
    )
    mmr_response_cutoff_cycles: float = _setting(
        0.1,
        'MMR_RESPONSE_CUTOFF_CYCLES',
        'float',
        'identity, force law, potentials use sigma, a smoothed at max(P_c, this x P_cyc): derivatives amplify what P_c leaks; '
        '0.1 keeps the cycle and its first 5 harmonics (G(2 cutoffs) = 0.94, as FILTER_MIN_LIBRATION_CUTOFFS) and costs an '
        'exact pendulum R^2 < 0.01 up to 178 deg; 0.15-0.2 bend the law near the separatrix (F[sin sigma] != sin F[sigma])',
    )
    mmr_response_line_amp_frac: float = _setting(
        0.5,
        'MMR_RESPONSE_LINE_AMP_FRAC',
        'float',
        "harmonic force law (L4): a line of a_dot (not a harmonic P_e/k of the libration) with an amplitude of at least this "
        'x the force K of the first fit, in the series the law reads, raises the response cutoff of that measurement '
        '(2004 TK14 31M-34: 64 yr at 0.9 K; 433 Eros 47M-44: 83 yr at 1.5 K)',
    )
    mmr_response_line_margin: float = _setting(
        1.5,
        'MMR_RESPONSE_LINE_MARGIN',
        'float',
        'such a line raises the response cutoff to this x its period (G = 0.16 of the line left in a, 2nd-order filtfilt)',
    )
    mmr_response_max_fraction: float = _setting(
        4.0,
        'MMR_RESPONSE_MAX_FRACTION',
        'float',
        'the response cutoff raised by a line stays at or below the period of the measurement / this (P_e/4 keeps G = 0.996 '
        'of the cycle; a slower line is left in)',
    )
    mmr_response_line_band_cutoffs: float = _setting(
        0.5,
        'MMR_RESPONSE_LINE_BAND_CUTOFFS',
        'float',
        'lines of a_dot are searched from this x the base response cutoff up to P_e/2: a line faster than the cutoff leaks '
        'through it (2004 TK14: 64 yr under a 75-yr cutoff keeps 0.9 K); below half the cutoff the filter leaves < 6 %',
    )
    mmr_force_null_shifts: int = _setting(
        8,
        'MMR_FORCE_NULL_SHIFTS',
        'int',
        'null test of a force law whose cutoff a line raised: R^2 of the same law with a_dot shifted in time (not wrapped) '
        'by this many lags (m + 1/4) P_e and (m + 3/4) P_e, alternately forward and back, m >= 1 whole periods spread up to '
        'half the record; its largest value is the null R^2 (whole or half periods alone keep a steady libration in phase)',
    )
    mmr_force_null_gain: float = _setting(
        0.2,
        'MMR_FORCE_NULL_GAIN',
        'float',
        'a force law whose cutoff a line raised passes only with R^2 at least this above the null R^2 (433 Eros 47M-44: '
        '0.6-0.8 against 0.15-0.35)',
        convention=True,
    )
    mmr_force_harmonics_max: int = _setting(
        2,
        'MMR_FORCE_HARMONICS_MAX',
        'int',
        'CS12: a failed or thin harmonic force law (L4) is fitted again with up to this many harmonics of the angle, '
        'a_dot = sum_k [A_k sin k sigma + B_k cos k sigma] + C, the force of the V(sigma) R0 fits (as '
        'SEPARATRIX_HARMONICS_MAX; a double well or a flat bottom): taken when N >= 2 gains MMR_FORCE_MULTI_GAIN, its '
        'coefficients are coherent (MMR_FORCE_COEF_COH_MIN), it beats the shifted null by MMR_FORCE_NULL_GAIN and one '
        'half of the windows predicts the other with R^2 >= MMR_FORCE_R2_FAIL; it lifts L4, never lowers it; 1 = off',
    )
    mmr_short_period_cap_ratio: float = _setting(
        0.0,
        'MMR_SHORT_PERIOD_CAP_RATIO',
        'float',
        'the record itself lifts the cap short_period_aliased only when the fast residual of sigma_f (p95, '
        'flt_fast_residual_deg) is at most this x the decisive episode\'s median half-amplitude and the short-period scatter '
        'of a passes MMR_SHORT_PERIOD_A_RATIO; 0 = never: the record cannot bound a line folded into a, only a finely '
        'sampled pilot measures it (W20 round D: 0.1 lifted it on 2004 TK14 31M-34 and 433 47M-44, whose verdicts the '
        'folded lines in a change)',
        convention=True,
    )
    mmr_short_period_cap_circulation: bool = _setting(
        False,
        'MMR_SHORT_PERIOD_CAP_CIRCULATION',
        'bool',
        'the cap short_period_aliased also on a circulation (angle word circulation, reason circulation or alias; '
        'kinematic_drift, no_resonant_potential and a circulation with forced-wave runs (F6, its failed identity reads a) '
        'keep it); off: a circulation reads neither the identity nor the force '
        'law, which the folded lines in a spoil, and the folded lines move sigma_f by <= 0.024 of a half-amplitude (W20 '
        'round D, 47 fine reruns), so they cannot turn a libration into a circulation; the flag stays '
        '(author, W20 Q1d, 23.10: 1627 3V-4M-7, 285637 3M+7J-6, 1999 UQ 7E+2M-9)',
        convention=True,
    )
    mmr_short_period_a_ratio: float = _setting(
        0.5,
        'MMR_SHORT_PERIOD_A_RATIO',
        'float',
        'flag a_short_period_scatter (a warning, no cap by itself) on a record whose step folds short periods: its '
        'short-period scatter of a (smp_da_short_period_au, periods below MMR_JITTER_BAND_YR) above this x the a '
        'half-excursion of the decisive episode (ep_a_excursion_meas_au); the folded lines reach the a-based checks '
        '(identity, force law) there (W20 round D: 2003 TG2, 2004 TK14, 2003 UC22 and 433 lose their verdict with folded '
        'errors of 0.59-1.65 x that excursion, no row at <= 0.56 does). A proxy, not a bound: a line folded to centuries '
        'is nearly invisible to it. With MMR_SHORT_PERIOD_CAP_RATIO > 0 it also keeps the cap; the same comparison '
        'gates MMR_SHORT_PERIOD_A_UNMEASURED',
        convention=True,
    )
    mmr_short_period_a_unmeasured: bool = _setting(
        True,
        'MMR_SHORT_PERIOD_A_UNMEASURED',
        'bool',
        'R-ALIAS-A: on a record whose step folds short periods (flag short_period_aliased), a failed force law (L4) and '
        'identity (L3), which read a, count as not_measured in the status tree and the confidence when the strength by '
        'direct averaging passes and the short-period scatter of a exceeds MMR_SHORT_PERIOD_A_RATIO x the decisive '
        'a half-excursion; a fail never becomes a pass and the cap stays (W22, author 2026-10-24: 433 47M-44, 2003 UC22 '
        '16M-27 and 2004 TK14 31M-34 at 5 yr keep their verdict at 0.5 yr, where L3 and L4 pass; 2001 ND13 1V-3, scatter '
        '0.09, keeps its failed L4, which its fine rerun confirms)',
        convention=True,
    )
    mmr_short_period_harmonics: int = _setting(
        2,
        'MMR_SHORT_PERIOD_HARMONICS',
        'int',
        'harmonics m / P (m = 1..this) of P_orb and of the P_syn of the angle\'s planets that the step may fold: a line '
        'folded slower than the angle\'s period floor (smp_folded_slowest_yr) sets the flag and the cap '
        'short_period_aliased also below the fundamental step min(P) / 2 (W20 round D, review C #3: P_orb 12 yr at a '
        '5.999-yr step folds its second harmonic to 36 kyr). Only m <= this is searched: P_orb 12 yr at 4.002 yr folds its '
        'third harmonic to 8 kyr and stays unflagged at 2 (the record bounds no harmonic\'s amplitude, so the count is a '
        'convention, W20 round E); 0 = the fundamental step alone',
        convention=True,
    )
    mmr_coorbital_grid_points: int = _setting(
        720,
        'MMR_COORBITAL_GRID_POINTS',
        'int',
        "co-orbital (1:1) law: points of the theta grid and of the trapezoidal average over the planet's longitude "
        '(720 = 0.5 deg; the average converges at 360, the grid interpolates the near-encounter peaks to ~5 % rms)',
    )
    mmr_coorbital_record_tables_mb: float = _setting(
        140.0,
        'MMR_COORBITAL_RECORD_TABLES_MB',
        'float',
        'co-orbital law (memory): the tables of the averaged potential kept for the analysis of one record, in bytes of '
        'their arrays (17.3 kB each at 720 points: 140 MB hold 8101); a table beyond them is computed again when a later '
        'pass needs it (2010 CG236 over 20 kyr needs 7909, 137 MB, at 7 ms each). A bound in bytes holds at any '
        'MMR_COORBITAL_GRID_POINTS (W20 round D, review C #6). It bounds this store only, not the process: the shared '
        'cache of 512 tables, the figure and the FFT workspaces come on top, and a process keeps the memory of its '
        'largest record (the benchmark tools recycle their workers, reclassify --tasks-per-worker)',
        unit='MB',
    )
    mmr_coorbital_softening_hill: float = _setting(
        0.1,
        'MMR_COORBITAL_SOFTENING_HILL',
        'float',
        "co-orbital law: the direct term is softened at this many Hill radii a_P (mu/3)^(1/3), so that an exact orbit crossing "
        "stays finite (the averaging does not describe the motion inside the Hill sphere; at 1 r_H the fitted K / K_theory of "
        "close-approaching horseshoes rises by 10-20 %, at 0.1 it is within 2 % of no softening)",
    )
    mmr_coorbital_retabulate_e: float = _setting(
        0.005,
        'MMR_COORBITAL_RETABULATE_E',
        'float',
        'co-orbital law: a new table of the averaged potential when the eccentricity vector e exp(i omega) of the body or '
        'the planet moved by more than this',
    )
    mmr_coorbital_retabulate_inc_deg: float = _setting(
        0.5,
        'MMR_COORBITAL_RETABULATE_INC_DEG',
        'float',
        'co-orbital law: a new table when the inclination vector sin i exp(i Omega) of the body or the planet moved by more '
        'than sin of this',
        unit='deg',
    )
    mmr_coorbital_scale_ratio_max: float = _setting(
        3.0,
        'MMR_COORBITAL_SCALE_RATIO_MAX',
        'float',
        "co-orbital law passes only with the fitted force K / K_theory in [1/this, this] (K_theory = 1: the model holds the "
        "planet's mass; a negative K pushes a the wrong way)",
        convention=True,
    )
    mmr_identity_slope_tol: float = _setting(0.2, 'MMR_IDENTITY_SLOPE_TOL', 'float', 'identity: |slope/c - 1| <= this')
    mmr_identity_corr_min: float = _setting(0.9, 'MMR_IDENTITY_CORR_MIN', 'float', 'identity: correlation of sigma_dot and a >= this')
    mmr_forced_wave_line_alone: bool = _setting(
        False,
        'MMR_FORCED_WAVE_LINE_ALONE',
        'bool',
        'an accepted episode whose period sits on a forced line of the lattice is a forced wave, no episode of the angle, '
        'even when its identity does not fail (the line alone); off: the line needs the failed identity, since a chance '
        'coincidence with a line happens for 6-17 % of the force laws (W12; 482824 19N-32: line 3N-5 with the identity '
        'passing at slope/c 1.00). Question F6 to the author (W20)',
        convention=True,
    )
    mmr_b1_ratio_min: float = _setting(
        0.1, 'MMR_B1_RATIO_MIN', 'float', 'a carries < this of the needed amplitude: the oscillation sits in the varpi part'
    )
    mmr_b1_band_factor: float = _setting(
        2.0,
        'MMR_B1_BAND_FACTOR',
        'float',
        'B1 compares a and sigma_dot / |c| in the band [P/f, f P] around the cycle period P (the libration and its '
        'wandering period; the identity holds there harmonic by harmonic)',
        convention=True,
    )
    mmr_kin_corr_min: float = _setting(
        0.9, 'MMR_KIN_CORR_MIN', 'float', 'kinematic: slow sigma follows p varpi with correlation >= this', convention=True
    )
    mmr_kin_slow_r2_max: float = _setting(
        0.2, 'MMR_KIN_SLOW_R2_MAX', 'float', 'kinematic: slow a_dot answers sin sigma_slow with R^2 <= this', convention=True
    )
    mmr_kin_min_points: int = _setting(
        3, 'MMR_KIN_MIN_POINTS', 'int', 'independent points (length / centre window) needed for the kinematic correlation'
    )
    mmr_kin_period_tol: float = _setting(
        0.1,
        'MMR_KIN_PERIOD_TOL',
        'float',
        'origin_detail: the angle period matches the e-cycle, the 2 omega cycle or a planetary ripple within this fraction',
    )
    mmr_kin_ripple_share_max: float = _setting(
        0.5,
        'MMR_KIN_RIPPLE_SHARE_MAX',
        'float',
        'kinematic drift of the slow centre (K2): the largest swing of a librating episode (2 x its largest half-amplitude) '
        'stays below this x the range of the slow centre, so the episodes are a ripple on the drift, not the libration '
        '(21028 7E-10M-6: 117 deg against 741 deg, 0.16; the Hilda 153 3J-2: 0.95)',
        convention=True,
    )
    mmr_kinematic_drift_word: bool = _setting(
        True,
        'MMR_KINEMATIC_DRIFT_WORD',
        'bool',
        'a verdict non_resonant/kinematic_drift of K2 reports the angle word of its slow centre, not of the ripple the '
        'geometry reads as episodes: drift, or circulation when the centre passes a revolution in one direction (G4) '
        '(author 2026-10-19 q7: 21028 drift; 2897 35S-4: 10.8 rev, his circulation); off: the geometry word (episodic) stays',
        convention=True,
    )
    mmr_kin_drift_shift_ratio: float = _setting(
        1.0,
        'MMR_KIN_DRIFT_SHIFT_RATIO',
        'float',
        'a kinematic libration (step 3, B1 or the kinematic cycle) whose decisive episode moves its centre by more than '
        'this x its mean amplitude (L6) is decided as K2: non_resonant/kinematic_drift with the slow centre\'s word, drift '
        'or circulation (W22 R-KDRIFT: 62243 1M-4J-1, L6 1.82, the author\'s drift)',
        convention=True,
    )
    mmr_b5_ratio_full: float = _setting(
        2.0, 'MMR_B5_RATIO_FULL', 'float', 'slow libration excluded with a full margin: centre rate >= this x the bound'
    )
    mmr_b5_exact_sigma: float = _setting(2.0, 'MMR_B5_EXACT_SIGMA', 'float', 'exact commensurability: |nu_c| < this x its error')
    mmr_c4_ratio_full: float = _setting(1.5, 'MMR_C4_RATIO_FULL', 'float', 'C4 full: <|sigma_dot|> / (2 omega0_K) >= this')

    # --- Subresonances of the multiplet and the owner (MMR) --------------------------------------
    mmr_subresonance_strength_rel_min: float = _setting(
        1.0e-3,
        'MMR_SUBRESONANCE_STRENGTH_REL_MIN',
        'float',
        "test E': a member's d'Alembert factor prod x^|c| (x = e, e_P, sin i/2, sin i_P/2) is at least this x that of "
        'the strongest term of the same commensurability (its pendulum is then <= 1/sqrt(this) ~ 30 times slower)',
    )
    mmr_subresonance_nodal_inc_min_deg: float = _setting(
        10.0,
        'MMR_SUBRESONANCE_NODAL_INC_MIN_DEG',
        'float',
        'members with node coefficients are enumerated when the median inclination to the invariable plane reaches '
        'this (their factor against the planar one of the same degree is (sin(i/2) / e)^2)',
        unit='deg',
    )
    mmr_subresonance_range_ratio: float = _setting(
        0.5, 'MMR_SUBRESONANCE_RANGE_RATIO', 'float', 'candidate: its centre range <= this x the range of the catalogue angle centre'
    )
    mmr_subresonance_rate_floor_rad_kyr: float = _setting(
        0.02, 'MMR_SUBRESONANCE_RATE_FLOOR_RAD_KYR', 'float', 'candidate: |centre rate| <= max(2 sigma_rate, this)', unit='rad/kyr'
    )
    mmr_subresonance_pass_max_rad: float = _setting(
        PI, 'MMR_SUBRESONANCE_PASS_MAX_RAD', 'float', 'candidate: largest monotone centre pass below this', unit='rad'
    )
    mmr_subresonance_phi0_coh_min: float = _setting(
        0.8, 'MMR_SUBRESONANCE_PHI0_COH_MIN', 'float', 'test A: phi0 coherence of the force law relative to the subresonance'
    )
    mmr_subresonance_phi0_coh_gain: float = _setting(
        0.2, 'MMR_SUBRESONANCE_PHI0_COH_GAIN', 'float', 'test A: and higher than for the catalogue angle by this'
    )
    mmr_subresonance_centre_tol_deg: float = _setting(
        30.0, 'MMR_SUBRESONANCE_CENTRE_TOL_DEG', 'float', 'test B: mean centre within this of 0 or pi', unit='deg'
    )
    mmr_subresonance_centre_e_max: float = _setting(
        0.3, 'MMR_SUBRESONANCE_CENTRE_E_MAX', 'float', 'test B is skipped above this eccentricity'
    )
    mmr_subresonance_centre_inc_max_deg: float = _setting(
        20.0, 'MMR_SUBRESONANCE_CENTRE_INC_MAX_DEG', 'float', 'test B is skipped above this inclination', unit='deg'
    )
    mmr_subresonance_tie_fraction: float = _setting(
        0.2, 'MMR_SUBRESONANCE_TIE_FRACTION', 'float', 'test C: centre ranges within this fraction tie'
    )
    mmr_subresonance_theory: bool = _setting(
        True,
        'MMR_SUBRESONANCE_THEORY',
        'bool',
        'test D: compare with the composite multiplet theory (confidence only: D never rejects a subresonance)',
    )
    mmr_subresonance_theory_rms_deg: float = _setting(
        20.0, 'MMR_SUBRESONANCE_THEORY_RMS_DEG', 'float', 'test D: RMS of the centre about pi - arg P(beta)', unit='deg'
    )
    mmr_subresonance_theory_single_ratio: float = _setting(
        0.5,
        'MMR_SUBRESONANCE_THEORY_SINGLE_RATIO',
        'float',
        'test D: and <= this x the RMS of the best rival single-member hypothesis (sigma0 in, the accepted member out)',
    )
    mmr_subresonance_theory_coef_fraction: float = _setting(
        0.2, 'MMR_SUBRESONANCE_THEORY_COEF_FRACTION', 'float', 'test D: the accepted one has the largest coefficient or within this of it'
    )
    mmr_subresonance_period_corr_min: float = _setting(
        0.8, 'MMR_SUBRESONANCE_PERIOD_CORR_MIN', 'float', 'test D: correlation of the period with T0(t) F(A)'
    )
    mmr_subresonance_period_ratio_max: float = _setting(
        2.0, 'MMR_SUBRESONANCE_PERIOD_RATIO_MAX', 'float', 'test D: median observed/predicted period within this factor'
    )
    mmr_subresonance_t0_ratio_max: float = _setting(
        2.0, 'MMR_SUBRESONANCE_T0_RATIO_MAX', 'float', 'test E: predicted T0 <= this x the observed period (is a resonance at all)'
    )
    mmr_subresonance_t2_r2_min: float = _setting(0.4, 'MMR_SUBRESONANCE_T2_R2_MIN', 'float', 'trigger T2: a responds (R^2 >= this) ...')
    mmr_subresonance_t2_coh_max: float = _setting(
        0.7, 'MMR_SUBRESONANCE_T2_COH_MAX', 'float', 'trigger T2: ... but the phi0 coherence is below this'
    )
    mmr_subresonance_t3_sigma: float = _setting(
        2.0, 'MMR_SUBRESONANCE_T3_SIGMA', 'float', 'support T3: drift rate matches a secular combination within this sigma'
    )
    mmr_subresonance_t3_rate_frac: float = _setting(
        0.25, 'MMR_SUBRESONANCE_T3_RATE_FRAC', 'float', 'support T3: or within this fraction of the drift rate'
    )
    mmr_subresonance_top_n: int = _setting(
        10, 'MMR_SUBRESONANCE_TOP_N', 'int', 'length of the ranked list sub_top; subresonances written to candidates.csv in top mode'
    )
    mmr_subresonance_cycle_period_ratio: float = _setting(
        2.0,
        'MMR_SUBRESONANCE_CYCLE_PERIOD_RATIO',
        'float',
        'the scan window is the longest stretch of the run whose full periods stay <= this x their median (the slow cycles '
        'at the run edges are near the separatrix of the modulated pendulum, where no centre is stationary)',
    )
    mmr_subresonance_omega_harmonics: int = _setting(
        2,
        'MMR_SUBRESONANCE_OMEGA_HARMONICS',
        'int',
        'with the nodal members (inclination >= MMR_SUBRESONANCE_NODAL_INC_MIN_DEG) the members sigma +- 2 m omega '
        '(omega = varpi - Omega), m = 1..this, are scanned and listed whatever their degree (2003 UG22 10E-17+0+7: '
        'sigma + 2 omega has degree 11); 0 = none beyond the degree limit',
        convention=True,
    )
    mmr_subresonance_dalembert_e_max: float = _setting(
        0.2,
        'MMR_SUBRESONANCE_DALEMBERT_E_MAX',
        'float',
        "the ranked list marks the d'Alembert factor unreliable above this eccentricity (display only, sub_strength_caveat)",
        convention=True,
    )
    mmr_subresonance_dalembert_inc_max_deg: float = _setting(
        20.0,
        'MMR_SUBRESONANCE_DALEMBERT_INC_MAX_DEG',
        'float',
        "the ranked list marks the d'Alembert factor unreliable above this inclination to the invariable plane (display only)",
        unit='deg',
        convention=True,
    )
    matrix_2body_order_max: int = _setting(
        10,
        'MATRIX_2BODY_ORDER_MAX',
        'int',
        'largest order of the two-body catalogue; also the largest degree (sum of |slow coefficients|) of the '
        'subresonances scanned and listed (members of degree |q| always)',
    )
    matrix_3body_order_max: int = _setting(
        6,
        'MATRIX_3BODY_ORDER_MAX',
        'int',
        'largest order of the three-body catalogue; also the largest degree (sum of |slow coefficients|) of the '
        'subresonances scanned and listed (members of degree |q| always)',
    )
    mmr_owner_search: str = _setting(
        'auto', 'MMR_OWNER_SEARCH', 'str', "'auto' = search the owner by the rules (always when the force law is not_measured); 'off'"
    )
    mmr_owner_planets: tuple = _setting(OWNER_PLANETS_DEFAULT, 'MMR_OWNER_PLANETS', 'strs', 'planets of the owner candidates')
    mmr_owner_coef_max: int = _setting(60, 'MMR_OWNER_COEF_MAX', 'int', 'largest planet coefficient of a two-body owner')
    mmr_owner_order_max: int = _setting(20, 'MMR_OWNER_ORDER_MAX', 'int', 'largest order of a two-body owner (owners of 7-17 were seen)')
    mmr_owner_coef_max_3b: int = _setting(12, 'MMR_OWNER_COEF_MAX_3B', 'int', 'largest |coefficient| of a three-body owner')
    mmr_owner_rate_factor: float = _setting(
        1.5,
        'MMR_OWNER_RATE_FACTOR',
        'float',
        'owner candidate: |rate| <= this x 2 pi / min(P_a, P_cyc) somewhere in the a range of the record',
    )
    mmr_owner_r2_min: float = _setting(
        0.7,
        'MMR_OWNER_R2_MIN',
        'float',
        'owner accepted: median R^2 of a_dot on its angle in the windows of one regime (0 false owners on 29 measurable '
        'own librations at any value 0.5-0.95 with the ripple test; GK3 and 117306 are lost at 0.9, YG4 at 0.95)',
        convention=True,
    )
    mmr_owner_phi0_coh_min: float = _setting(0.8, 'MMR_OWNER_PHI0_COH_MIN', 'float', 'owner accepted: phi0 coherence')
    mmr_owner_r2_gain: float = _setting(
        0.2,
        'MMR_OWNER_R2_GAIN',
        'float',
        'owner accepted (route r2): R^2 above that of the angle itself by this; routes phase and potential are the alternatives',
    )
    mmr_owner_phi0_coh_gain: float = _setting(
        0.2,
        'MMR_OWNER_PHI0_COH_GAIN',
        'float',
        'owner accepted (route phase, W20): its phi0 coherence C_B >= L_owner + this, L_owner = |<exp(i Psi_w / v)>| the most '
        "it could have if sigma's own term drove a, while sigma's C_sigma < L_sigma + this (L_sigma = |<exp(i Psi_w / u)>|, "
        "corrected for the owner's slips); L_owner > 1 - this: phase_unresolved (at small amplitude both angles fit a_dot "
        'by the identity sigma = (v/u) phi_o + Psi/u)',
    )
    mmr_owner_band_factor: float = _setting(
        2.0,
        'MMR_OWNER_BAND_FACTOR',
        'float',
        'owner test: a_dot band [P/f, f P] around the a-wave period P; P is searched within a factor f of P_cyc; '
        'force-law windows keep f P (x filter_edge_trim_cutoffs) from the record ends',
    )
    mmr_owner_ripple_factor: float = _setting(
        2.0,
        'MMR_OWNER_RIPPLE_FACTOR',
        'float',
        "owner rejected as a ripple: sigma below the owner's band turns back in the regime's windows with a prominence of "
        "at least this x the band's peak-to-peak, more than a circulating owner can move it",
    )
    mmr_owner_psi_share_max: float = _setting(
        0.1, 'MMR_OWNER_PSI_SHARE_MAX', 'float', 'owner accepted: fast share of the remainder psi below this'
    )
    mmr_owner_strength_margin: float = _setting(
        10.0,
        'MMR_OWNER_STRENGTH_MARGIN',
        'float',
        'order-of-magnitude band of K_o / (2 |k| n a prod mu): orders the owner candidates and qualifies owner_probable;'
        ' never accepts or rejects an owner (mu e^q is no bound, W3-F3)',
    )
    mmr_owner_top_n: int = _setting(5, 'MMR_OWNER_TOP_N', 'int', 'owner candidates written to candidates.csv')
    mmr_owner_law: str = _setting(
        'multiplet',
        'MMR_OWNER_LAW',
        'str',
        "owner phase test: 'multiplet' = where the single phi0 drifts, fit phi_o + j (varpi - varpi_P), j from 0 to -p, with "
        "constant coefficients (R^2 >= MMR_OWNER_R2_MIN, coefficient coherence >= MMR_OWNER_PHI0_COH_MIN); 'single' = phi0 only",
    )
    mmr_owner_cover_min: float = _setting(
        0.5,
        'MMR_OWNER_COVER_MIN',
        'float',
        "an accepted owner takes sigma's oscillation (origin induced) only when its intervals (its passing regimes' "
        "windows and the slips between them) cover at least this fraction of sigma's decisive episode (for a libration "
        'over the record or no episode: the analysis window); else origin none, flag owner_outside_episode (F7, W20)',
        convention=True,
    )
    mmr_owner_potential: str = _setting(
        'auto',
        'MMR_OWNER_POTENTIAL',
        'str',
        "owner route 'potential': 'auto' = a two-body candidate with the planet's saved elements may be accepted when its "
        'directly averaged force (2 k_o / (n a)) G m_P d<D>/dphi_o, band-passed as a, explains a_dot (R^2 >= '
        "MMR_OWNER_R2_MIN, every window scale s in [1/MMR_OWNER_POTENTIAL_SCALE_MAX, MMR_OWNER_POTENTIAL_SCALE_MAX]); 'off'",
    )
    mmr_owner_potential_scale_max: float = _setting(
        2.0,
        'MMR_OWNER_POTENTIAL_SCALE_MAX',
        'float',
        "owner route 'potential': the fitted amplitude scale s of the predicted a_dot lies within [1/x, x] in every window "
        '(x = 2: the owner supplies at least half the force; the model error of the frozen average is ~0.1)',
        convention=True,
    )
    mmr_owner_potential_scale_full: float = _setting(
        1.5,
        'MMR_OWNER_POTENTIAL_SCALE_FULL',
        'float',
        "margin W of an owner accepted by 'potential': full when its scale range lies within [1/x, x] and R^2 >= "
        'MMR_W_R2_FULL, else thin',
        convention=True,
    )
    mmr_owner_potential_snapshots: int = _setting(
        5,
        'MMR_OWNER_POTENTIAL_SNAPSHOTS',
        'int',
        "owner route 'potential': the averaged force is evaluated in at most this many of the regime's windows, evenly chosen",
    )
    mmr_owner_potential_top_n: int = _setting(
        3,
        'MMR_OWNER_POTENTIAL_TOP_N',
        'int',
        "owner route 'potential' (cost): tried on the best this many candidates by R^2 that pass R^2, psi share and the ripple "
        'test, when no candidate was accepted by R^2 or phase (the accepted owner gets it too, for its columns)',
    )
    mmr_owner_potential_max_kernels: float = _setting(
        1e8,
        'MMR_OWNER_POTENTIAL_MAX_KERNELS',
        'float',
        "owner route 'potential' (cost): one profile <D>(phi_o) on L nodes per turn evaluates |k_o| L^2 kernels "
        '(L values of phi_o, |k_o| L nodes each); skipped (reason potential_cost) when the next L would exceed this '
        '(about 5 ns per kernel: 1e8 is ~0.5 s per profile, at most MMR_OWNER_POTENTIAL_SNAPSHOTS x '
        'MMR_OWNER_POTENTIAL_TOP_N profiles per row)',
    )
    mmr_owner_potential_confirm: bool = _setting(
        False,
        'MMR_OWNER_POTENTIAL_CONFIRM',
        'bool',
        'a two-body owner accepted by R^2 or phase must also agree with its directly averaged force where that is measured: '
        'every window K/K_direct within [1/MMR_OWNER_POTENTIAL_SCALE_MAX, MMR_OWNER_POTENTIAL_SCALE_MAX] and |phi0 - '
        'phi0_direct| <= MMR_OWNER_POTENTIAL_PHASE_MAX_DEG, else rejected (potential_unconfirmed); off = columns only '
        '(question R1 to the author, W20)',
        convention=True,
    )
    mmr_owner_potential_phase_max_deg: float = _setting(
        20.0,
        'MMR_OWNER_POTENTIAL_PHASE_MAX_DEG',
        'float',
        'MMR_OWNER_POTENTIAL_CONFIRM: the largest |phi0 - phi0_direct| of the fitted and the directly averaged force',
        unit='deg',
        convention=True,
    )
    mmr_owner_capture_lowpass_periods: float = _setting(
        3.0,
        'MMR_OWNER_CAPTURE_LOWPASS_PERIODS',
        'float',
        'second capture: Psi = u sigma - v phi_o is low-passed at this many owner-band periods P (its slow part: the '
        "owner's wave cancels in Psi, what is left is the separation of the two resonances); the filter edges are trimmed",
    )
    mmr_owner_capture_sweep_deg: float = _setting(
        180.0,
        'MMR_OWNER_CAPTURE_SWEEP_DEG',
        'float',
        "second capture (sigma held by its own term while the owner pins phi_o) is excluded when x = Psi_slow / u sweeps at "
        'least this and x_dot^2 is not a function of x (R^2 on [1, cos x, sin x, cos 2x, sin 2x] < MMR_OWNER_R2_MIN); '
        'otherwise unresolved (flag second_capture_unresolved, confidence <= medium)',
        unit='deg',
        convention=True,
    )
    mmr_owner_capture_harmonics: int = _setting(
        2,
        'MMR_OWNER_CAPTURE_HARMONICS',
        'int',
        'second capture: the energy integral x_dot^2 = 2E - 2V(x) is fitted with the harmonics 1 .. this of x (V periodic '
        'in x; 2 = [1, cos x, sin x, cos 2x, sin 2x], R5 §1.7)',
        convention=True,
    )
    mmr_owner_capture_samples_per_term: int = _setting(
        3,
        'MMR_OWNER_CAPTURE_SAMPLES_PER_TERM',
        'int',
        'second capture: the energy fit needs at least this many samples per fitted term (after the filter edges are '
        'trimmed), else the block is not measured',
    )
    mmr_owner_potential_model_error: float = _setting(
        0.1,
        'MMR_OWNER_POTENTIAL_MODEL_ERROR',
        'float',
        "display only (the figure's owner lines): the model error of the frozen first-order average (Codex scatter "
        "0.90-1.12 of s); a bound on sigma's own force max |s - 1| below it is printed as this value ('model limit') and the "
        'own period bound is taken at it',
        convention=True,
    )
    mmr_owner_identity_ratio_min: float = _setting(
        1000.0,
        'MMR_OWNER_IDENTITY_RATIO_MIN',
        'float',
        "owner route 'identity' (W22 R-OWNER-ID): tried on a librating three-body angle without an accepted owner whose "
        'period needs at least this many times mu_i mu_j e^q (str_ratio, before the small-divisor envelope)',
        convention=True,
    )
    mmr_owner_identity_coef_tol: float = _setting(
        0.05,
        'MMR_OWNER_IDENTITY_COEF_TOL',
        'float',
        "owner route 'identity': on sigma's decisive window sigma_f = c phi_f + a + b t with |c / (k_sigma / k_phi) - 1| at "
        'most this (the exact identity passes it for almost any unwrapped candidate: phi librating there and its strength '
        'decide)',
        convention=True,
    )
    mmr_owner_identity_resid_max: float = _setting(
        0.1,
        'MMR_OWNER_IDENTITY_RESID_MAX',
        'float',
        "owner route 'identity': rms of sigma_f - (k_sigma / k_phi) phi_f about its line on sigma's decisive window, over "
        "sigma's median half-amplitude, at most this",
        convention=True,
    )
    mmr_owner_identity_span_max_deg: float = _setting(
        360.0,
        'MMR_OWNER_IDENTITY_SPAN_MAX_DEG',
        'float',
        "owner route 'identity' (cost pre-filter): only a candidate whose low-passed phi_f spans less than this on sigma's "
        'decisive window is analysed in full (a phi running a full turn there does not librate about a fixed centre)',
        unit='deg',
    )
    mmr_owner_identity_top_n: int = _setting(
        3,
        'MMR_OWNER_IDENTITY_TOP_N',
        'int',
        "owner route 'identity' (cost): at most this many candidates passing the identity numbers, smallest phi_f span "
        'first, get their own full analysis (libration, cover of the window, direct-averaging strength)',
    )
    mmr_forced_wave_linear_ratio: float = _setting(
        2.0,
        'MMR_FORCED_WAVE_LINEAR_RATIO',
        'float',
        "display only (the figure's owner lines): the linear forced-wave amplitude delta a = K / nu holds when the owner's "
        'rate |nu| is at least this many times its own small-amplitude frequency omega0 (nu >> omega0)',
        convention=True,
    )

    # --- Strength (MMR) ------------------------------------------------------------------------
    mmr_strength_e_max: float = _setting(
        0.3,
        'MMR_STRENGTH_E_MAX',
        'float',
        'the e^q order-of-magnitude strength scale is used only when the median e of the record is below this and the '
        "body's radial range crosses no planet of the term; the same median e enters e^q",
    )
    mmr_strength_check_ratio: float = _setting(
        10.0,
        'MMR_STRENGTH_CHECK_RATIO',
        'float',
        'check_mmr_strength passes when required/theory is within [1/x, x]: two-body, the range over the snapshots of the '
        'directly averaged theory whose observed swing lies in the well overlaps it (none: fail); three-body, '
        'required / mu e^q (legal e^q only, a scale, not a bound, support only); 1:1: not_applicable, its scale is '
        'judged by MMR_COORBITAL_SCALE_RATIO_MAX',
    )
    mmr_strength_implausible_ratio: float = _setting(
        1000.0,
        'MMR_STRENGTH_IMPLAUSIBLE_RATIO',
        'float',
        'implausible_strength: required/theory >= this or <= 1/this (two-body direct averaging; with the e^q scale of '
        'three-body terms, whose f(alpha) reaches ~1e2 (2003 EH16 1M-3J-1: 254), the lower side only with legal e^q); '
        'rule 5s: a librating angle for which neither L4 nor R0 passes (a measured failure included) and whose period '
        'needs >= this x a computed strength is non_resonant (two-body direct averaging or the harmonic, a centre '
        'outside the well excluded unless MMR_STRENGTH_FIRST_OUTSIDE_WELL; three-body only with legal e^q and when '
        'required / (mu e^q B^2) of the small-divisor envelope stays >= this too)',
    )
    mmr_strength_snapshots: int = _setting(
        3,
        'MMR_STRENGTH_SNAPSHOTS',
        'int',
        'two-body strength: the averaged disturbing function is evaluated with the simultaneous elements of the body and '
        'the planet at this many evenly spaced times inside the decisive episode; the check reads the range over them',
    )
    mmr_strength_unsaved_planet_phases: int = _setting(
        4,
        'MMR_STRENGTH_UNSAVED_PLANET_PHASES',
        'int',
        "two-body strength without the planet's saved elements: its mean J2000 orbit turned about the ecliptic pole by "
        'this many evenly spaced angles (its perihelion and node phase is unknown), all in the range',
    )
    mmr_strength_nodes_per_turn: int = _setting(
        720,
        'MMR_STRENGTH_NODES_PER_TURN',
        'int',
        "two-body strength: trapezoidal nodes per turn of the planet's longitude in the average at fixed sigma (doubled "
        'until the curvature with half of them agrees within MMR_STRENGTH_QUADRATURE_TOL)',
    )
    mmr_strength_max_nodes_per_turn: int = _setting(
        5760,
        'MMR_STRENGTH_MAX_NODES_PER_TURN',
        'int',
        'two-body strength: the largest number of nodes per turn; not converged there = not measured (nearly crossing orbits)',
    )
    mmr_strength_sigma_grid: int = _setting(
        72, 'MMR_STRENGTH_SIGMA_GRID', 'int', 'two-body strength: points of the coarse sigma grid that finds the wells of <R>'
    )
    mmr_strength_fit_half_width_deg: float = _setting(
        2.0,
        'MMR_STRENGTH_FIT_HALF_WIDTH_DEG',
        'float',
        'two-body strength: the curvature at the centre is the quadratic fit over +- this (the 1:1 tadpole 9/4 is 0.08 % off '
        'at 2 deg, 0.5 % at 5 deg)',
        unit='deg',
    )
    mmr_strength_quadrature_tol: float = _setting(
        0.02,
        'MMR_STRENGTH_QUADRATURE_TOL',
        'float',
        'two-body strength: the curvature is converged when the half-node quadrature agrees within this relative difference',
    )
    mmr_strength_min_distance_hill: float = _setting(
        3.0,
        'MMR_STRENGTH_MIN_DISTANCE_HILL',
        'float',
        'two-body strength: at the centre the body must stay this many Hill radii a_P (mu/3)^(1/3) from the planet along the '
        'average, else the averaged problem does not hold there (an encounter) and the strength is not measured',
    )
    mmr_strength_mp: str = _setting(
        'auto',
        'MMR_STRENGTH_MP',
        'str',
        "two-body strength where double precision cannot resolve the well (no stable centre, no converged curvature, or a "
        "curvature below MMR_STRENGTH_DOUBLE_FLOOR x |D0|): 'auto' = mu_P a 2|C| of the resonant harmonic C in extended "
        "precision (mpmath; method direct_harmonic); 'off' = not measured, as before",
    )
    mmr_strength_mp_dps: int = _setting(
        60,
        'MMR_STRENGTH_MP_DPS',
        'int',
        'two-body strength in extended precision: decimal digits of the first attempt (doubled up to MMR_STRENGTH_MP_DPS_MAX '
        'while |C| / |D0| < 10^(MMR_STRENGTH_MP_GUARD_DIGITS - dps))',
    )
    mmr_strength_mp_dps_max: int = _setting(
        130,
        'MMR_STRENGTH_MP_DPS_MAX',
        'int',
        'two-body strength in extended precision: the largest number of digits; a harmonic still unresolved there is not '
        'measured (precision_exhausted)',
    )
    mmr_strength_mp_guard_digits: int = _setting(
        15,
        'MMR_STRENGTH_MP_GUARD_DIGITS',
        'int',
        'two-body strength in extended precision: digits kept beyond the cancellation log10(|D0| / |C|) of the sum',
    )
    mmr_strength_mp_nodes_body: int = _setting(
        512,
        'MMR_STRENGTH_MP_NODES_BODY',
        'int',
        "two-body strength in extended precision: trapezoidal nodes of the body's mean anomaly; the twin grid shifted by half "
        'a node must agree within MMR_STRENGTH_QUADRATURE_TOL',
    )
    mmr_strength_mp_nodes_planet: int = _setting(
        128,
        'MMR_STRENGTH_MP_NODES_PLANET',
        'int',
        "two-body strength in extended precision: trapezoidal nodes of the planet's mean longitude (the same twin check)",
    )
    mmr_strength_double_floor: float = _setting(
        1e-12,
        'MMR_STRENGTH_DOUBLE_FLOOR',
        'float',
        "two-body strength: a double-precision curvature of <D> below this x |D0| is rounding, not a well (the quadratic "
        "fit over +-2 deg divides the 1e-16 rounding of <D> by ~1e-3 rad^2); the harmonic in extended precision is used "
        "instead (numerical, not a convention)",
    )
    mmr_strength_first_outside_well: bool = _setting(
        False,
        'MMR_STRENGTH_FIRST_OUTSIDE_WELL',
        'bool',
        'rule 5s (implausible strength) runs before 5a*/5e/5g/5u for a librating angle whose force law and potential with a '
        'do not pass; with this off a two-body strength whose observed swings all lie outside the theoretical well '
        '(centre_outside_well) keeps the old order and reaches 5s only inside the branch where L4 and R0 are both not '
        'measured (129772 4N-7, 145480 1N-4 1 Myr, 719 36J-13; question to the author, W20 q4)',
        convention=True,
    )
    mmr_centre_track_nodes: int = _setting(
        400,
        'MMR_CENTRE_TRACK_NODES',
        'int',
        'moving centre: the most nodes of the track c(t) of the averaged-potential minimum along a record; the node step '
        'is max(P_c, span / this). Numerical resolution, not a threshold: the well moves on secular time scales (2013 '
        'AS27 31E-36: 56 deg in 20 kyr, 0.2 deg per node), c(t) is interpolated linearly between nodes',
    )

    # --- Margins (MMR) -------------------------------------------------------------------------
    mmr_l1_tp_full: int = _setting(
        7, 'MMR_L1_TP_FULL', 'int', 'L1 full: turning points at cutoffs x0.5, x1, x2 (three cycles)', convention=True
    )
    mmr_l2_amp_full_deg: float = _setting(
        120.0, 'MMR_L2_AMP_FULL_DEG', 'float', 'L2 without a zone: full below this amplitude', unit='deg', convention=True
    )
    mmr_l2_amp_thin_deg: float = _setting(
        160.0,
        'MMR_L2_AMP_THIN_DEG',
        'float',
        'L2 without a zone: thin up to this amplitude, above it a medium cap',
        unit='deg',
        convention=True,
    )
    mmr_l5_ratio_full: float = _setting(
        1.15, 'MMR_L5_RATIO_FULL', 'float', 'L5: T0 max/min within this x Omega_max/Omega_min (T0 resolution ~10%)', convention=True
    )
    mmr_l5_ratio_fail: float = _setting(2.0, 'MMR_L5_RATIO_FAIL', 'float', 'L5 fails beyond this x the full threshold')
    mmr_l5_e_caveat: float = _setting(0.2, 'MMR_L5_E_CAVEAT', 'float', 'above this e the period-amplitude law may fail: L5 at most thin')
    mmr_l6_shift_full: float = _setting(0.5, 'MMR_L6_SHIFT_FULL', 'float', 'L6 full: centre shift <= this x the mean amplitude')
    mmr_l7_gap_full: float = _setting(0.7, 'MMR_L7_GAP_FULL', 'float', 'L7 full: coverage gaps <= this x the period')
    mmr_l8_step_full_deg: float = _setting(
        10.0,
        'MMR_L8_STEP_FULL_DEG',
        'float',
        'L8 full: rotation per step below this (36 points per turn); not graded for a non_resonant circulation whose '
        'slow a stays MMR_FAR_FROM_RES_FACTOR a-scales from a* (the step need not resolve turning points it cannot have)',
        unit='deg',
    )
    mmr_l8_gain_full: float = _setting(
        0.94,
        'MMR_L8_GAIN_FULL',
        'float',
        'L8 full: filter gain at the cycle period >= this; below it the libration is filter_too_slow (shape checks not_measured)',
    )
    mmr_l8_gain_fail: float = _setting(0.5, 'MMR_L8_GAIN_FAIL', 'float', 'L8 fails: filter gain at the cycle period below this')
    mmr_c1_rev_full: float = _setting(
        10.0, 'MMR_C1_REV_FULL', 'float', 'C1 full: largest centre pass of a circulation in revolutions', unit='rev'
    )
    mmr_c3_modulation_max: float = _setting(
        0.3, 'MMR_C3_MODULATION_MAX', 'float', 'C3: rate modulation below this fraction of the mean rate'
    )
    mmr_c5_rate_tol: float = _setting(0.2, 'MMR_C5_RATE_TOL', 'float', 'C5: |measured/expected rate - 1| <= this')
    mmr_c5_branch_resid_factor: float = _setting(
        2.0,
        'MMR_C5_BRANCH_RESID_FACTOR',
        'float',
        'C5 alias full: far from a* (MMR_FAR_FROM_RES_FACTOR) and the branch clearance per step [deg] > this x the largest '
        'per-step unwrap residual (smp_unwrap_resid_max_deg): no step can have been put on the wrong branch',
        convention=True,
    )
    mmr_c5_far_ratio_min: float = _setting(
        3.0, 'MMR_C5_FAR_RATIO_MIN', 'float', 'C5 alias: far ratio below this is not an alias circulation but invalid/alias + resample'
    )
    mmr_o1_ratio_thin: float = _setting(0.25, 'MMR_O1_RATIO_THIN', 'float', 'O1: std of the a-wave before / in the episode thin from this')
    mmr_o1_ratio_fail: float = _setting(0.5, 'MMR_O1_RATIO_FAIL', 'float', 'O1: fails (cap low) from this')
    mmr_o1_owner_cover_min: float = _setting(
        0.5,
        'MMR_O1_OWNER_COVER_MIN',
        'float',
        "O1 not_applicable for an induced libration (origin_detail owner_libration) when the owner's libration intervals "
        "cover at least this fraction of O1's background window (the band before sigma's decisive episode): the a-wave "
        'there is the owner\'s libration, not a regime before a capture (W22 R-O1-OWNER)',
        convention=True,
    )
    mmr_k1_full: float = _setting(0.97, 'MMR_K1_FULL', 'float', 'K1 full: correlation of slow sigma_dot with p varpi_dot')
    mmr_w_r2_none: float = _setting(0.4, 'MMR_W_R2_NONE', 'float', 'W full (no owner): best candidate R^2 below this')
    mmr_w_r2_full: float = _setting(0.85, 'MMR_W_R2_FULL', 'float', 'W full (induced): owner R^2 at least this')
    mmr_s_close_ratio: float = _setting(
        0.8,
        'MMR_S_CLOSE_RATIO',
        'float',
        "S full (none accepted): the closest member's centre range > this x that of sigma0 in the same scan window",
    )

    # --- Secular and Lidov-Kozai angles (analysis/secular, analysis/zlk) --------------------------
    secular_filter_gi_factor: float = _setting(
        2.0,
        'SECULAR_FILTER_GI_FACTOR',
        'float',
        'P_c of a secular angle >= this x the great inequality period (937 yr), the longest short-period term of an '
        "asteroid's elements: the filter passes 1/(1 + x^(2N)) of it (6% for x = 2, N = 2)",
    )
    secular_filter_forced_factor: float = _setting(
        2.0,
        'SECULAR_FILTER_FORCED_FACTOR',
        'float',
        "P_c >= this x the longest forced wave 2 pi/|f - f_j| of the angle's own planetary modes (body f = g or s); "
        'a single-mode angle has none (its forced response is the resonance itself)',
    )
    secular_tp_prominence_deg: float = _setting(
        5.0,
        'SECULAR_TP_PROMINENCE_DEG',
        'float',
        'turning-point prominence floor of a secular angle (see SECULAR_TP_RIPPLE_FACTOR)',
        unit='deg',
        convention=True,
    )
    secular_libration_period_min_yr: float = _setting(
        2.0e5,
        'SECULAR_LIBRATION_PERIOD_MIN_YR',
        'float',
        'shortest libration period of a secular angle, a convention: a pendulum librates no faster than half its rate '
        'width, and the widest linear resonances are a few arcsec/yr wide (nu6: +-1.2"/yr, Carruba et al. 2018), so '
        'periods are >~ 1 Myr; the fastest of the benchmark is Tina around its forced equilibrium (590 kyr). Faster '
        "'episodes' are forced waves, and the TP prominence covers what sigma_f holds faster than this",
        unit='yr',
        convention=True,
    )
    secular_tp_ripple_factor: float = _setting(
        4.0,
        'SECULAR_TP_RIPPLE_FACTOR',
        'float',
        'TP prominence >= this x the 95th percentile of what sigma_f holds faster than SECULAR_LIBRATION_PERIOD_MIN_YR '
        '(a wave of half-amplitude r turns by up to 2 r)',
    )
    secular_response_r2_min: float = _setting(
        0.5,
        'SECULAR_RESPONSE_R2_MIN',
        'float',
        'check_secular_e_response passes when e_dot (and d sin(i/2)/dt for an s-angle) follows A sin(sigma) + B cos(sigma) '
        '+ C with R^2 >= this; a column only, it changes no status and no confidence',
        convention=True,
    )
    zlk_filter_gi_factor: float = _setting(
        2.0,
        'ZLK_FILTER_GI_FACTOR',
        'float',
        'P_c of the Lidov-Kozai angle omega >= this x the great inequality period (937 yr) (6% passes for x = 2, N = 2)',
    )
    zlk_libration_period_min_yr: float = _setting(
        2.0e3,
        'ZLK_LIBRATION_PERIOD_MIN_YR',
        'float',
        'shortest libration period of omega, a convention: the quadrupole ZLK time (2/3pi)(M_sun/m_J) P_J^2/P '
        '(1 - e_J^2)^(3/2) (Antognini 2015, the coefficient of eq. 27) is 3.9 kyr at 4 au and 31 kyr at 1 au, and a '
        "libration lasts several of it; faster 'episodes' are forced waves",
        unit='yr',
        convention=True,
    )
    zlk_tp_ripple_factor: float = _setting(
        4.0,
        'ZLK_TP_RIPPLE_FACTOR',
        'float',
        'TP prominence of omega >= this x the 95th percentile of what omega_f holds faster than ZLK_LIBRATION_PERIOD_MIN_YR',
    )
    zlk_tp_prominence_deg: float = _setting(
        5.0,
        'ZLK_TP_PROMINENCE_DEG',
        'float',
        'turning-point prominence floor of omega (see ZLK_TP_RIPPLE_FACTOR)',
        unit='deg',
        convention=True,
    )

    # --- Confidence (shared) -------------------------------------------------------------------
    confidence_thin_band: float = _setting(
        0.1, 'CONFIDENCE_THIN_BAND', 'float', 'margin U: a decision within this relative distance of a convention'
    )
    confidence_cells: str = _setting(
        '',
        'CONFIDENCE_CELLS',
        'str',
        'calibration cells with their author-label counts, written by benchmark/tools/calibrate.py and reported in the '
        'cell_* columns, never changing a level (empty = resonances/data/confidence_cells.json)',
    )

    def __post_init__(self):
        validate_settings(self)


@dataclass(frozen=True)
class DryRunSettings:
    """Settings of the dry-run (SPEC §9.6); they do not change any verdict, hence apart from the hash."""

    dryrun_fair: str = _setting('optional', 'DRYRUN_FAIR', 'str', "'required' = the step resolves FAIR; 'optional'")
    dryrun_fair_fraction: float = _setting(0.238, 'DRYRUN_FAIR_FRACTION', 'float', 'FAIR needs a step <= this x the orbital period (1/4.2)')
    dryrun_da_static: tuple = _setting(
        (0.005, 0.001, 0.002),
        'DRYRUN_DA_STATIC',
        'floats',
        'spread of a without a pilot: belt, NEA, planet-crossing (provisional)',
        unit='au',
    )
    dryrun_da_sp_static: tuple = _setting(
        (0.003, 0.0003, 0.001), 'DRYRUN_DA_SP_STATIC', 'floats', 'short-period a scatter without a pilot: belt, NEA, crossing', unit='au'
    )
    dryrun_steps_per_orbit: float = _setting(10.0, 'DRYRUN_STEPS_PER_ORBIT', 'float', 'SABA steps per orbit (measured)')
    dryrun_steps_per_peri: float = _setting(10.0, 'DRYRUN_STEPS_PER_PERI', 'float', 'SABA steps per perihelion passage (provisional)')
    dryrun_nea_q: float = _setting(1.3, 'DRYRUN_NEA_Q', 'float', 'ias15 below this perihelion distance', unit='au')
    dryrun_hill_margin: float = _setting(1.0, 'DRYRUN_HILL_MARGIN', 'float', 'crossing band around a planet orbit in Hill radii')
    dryrun_drop_aliased: bool = _setting(False, 'DRYRUN_DROP_ALIASED', 'bool', 'drop guaranteed aliased circulations before integrating')
    dryrun_drop_ghost_factor: float = _setting(
        20.0, 'DRYRUN_DROP_GHOST_FACTOR', 'float', 'drop only beyond this many Nyquist half-widths from a_res'
    )
    dryrun_pilot_step_yr: float = _setting(0.05, 'DRYRUN_PILOT_STEP_YR', 'float', 'output step of the pilot integration', unit='yr')
    dryrun_pilot_min_yr: float = _setting(600.0, 'DRYRUN_PILOT_MIN_YR', 'float', 'shortest pilot', unit='yr')
    dryrun_pilot_max_yr: float = _setting(3000.0, 'DRYRUN_PILOT_MAX_YR', 'float', 'longest pilot', unit='yr')
    dryrun_pilot_cutoffs: float = _setting(10.0, 'DRYRUN_PILOT_CUTOFFS', 'float', 'pilot length >= this x the predicted cutoff')
    dryrun_auto_steps_yr: tuple = _setting(
        AUTO_STEPS_DEFAULT, 'DRYRUN_AUTO_STEPS_YR', 'floats', "round values OUTPUT_STEP='auto' is chosen from", unit='yr'
    )
    dryrun_max_step_yr: Optional[float] = _setting(
        None, 'DRYRUN_MAX_STEP_YR', 'optional_float', "the user's cap on OUTPUT_STEP='auto' (empty = none)", unit='yr'
    )
    dryrun_forced_term_min_deg: float = _setting(
        2.0,
        'DRYRUN_FORCED_TERM_MIN_DEG',
        'float',
        'pilot: report a line of sigma the filter passes (period >= P_c) whose filtered amplitude reaches this',
        unit='deg',
    )
    dryrun_forced_term_max_coefficient: int = _setting(
        12, 'DRYRUN_FORCED_TERM_MAX_COEFFICIENT', 'int', 'pilot: name such a line |p n_j - q n| with p, q up to this'
    )

    def __post_init__(self):
        validate_settings(self)


CHOICES = {
    'mmr_unwrap': ('predictive', 'numpy'),
    'mmr_zone_method': ('numeric', 'eq_only', 'adiabatic'),
    'mmr_owner_search': ('auto', 'off'),
    'mmr_strength_mp': ('auto', 'off'),
    'mmr_owner_law': ('multiplet', 'single'),
    'mmr_owner_potential': ('auto', 'off'),
    'mmr_force_contamination': ('lattice', 'off'),
    'dryrun_fair': ('optional', 'required'),
}

# (smaller, larger): the first must be strictly below the second
ORDERED_PAIRS = (
    ('libration_min_turning_points', 'mmr_l1_tp_full'),
    ('libration_near_episode_tp_deficit', 'libration_min_turning_points'),
    ('mmr_force_r2_fail', 'mmr_force_r2_full'),
    ('mmr_l2_amp_full_deg', 'mmr_l2_amp_thin_deg'),
    ('mmr_o1_ratio_thin', 'mmr_o1_ratio_fail'),
    ('mmr_w_r2_none', 'mmr_w_r2_full'),
    ('mmr_unwrap_max_resid_p99_deg', 'mmr_unwrap_max_resid_deg'),
    ('mmr_l8_gain_fail', 'mmr_l8_gain_full'),
    ('dryrun_pilot_min_yr', 'dryrun_pilot_max_yr'),
    ('mmr_strength_mp_guard_digits', 'mmr_strength_mp_dps'),
    ('mmr_owner_potential_scale_full', 'mmr_owner_potential_scale_max'),
)
# (smaller, larger): the first must not exceed the second
ORDERED_OR_EQUAL = (
    ('mmr_strength_mp_dps', 'mmr_strength_mp_dps_max'),
    ('mmr_extend_min_span_factor', 'mmr_next_check_span_factor'),
)
# Fractions and correlations that must lie in [0, 1]
UNIT_INTERVAL = (
    'libration_stall_rev_fraction',
    'libration_drift_min_window_fraction',
    'separatrix_phase_coherence',
    'separatrix_vevf_r_min',
    'separatrix_vevf_fraction',
    'separatrix_permutation_p_max',
    'mmr_force_window_step',
    'mmr_force_r2_full',
    'mmr_force_r2_fail',
    'mmr_force_coef_coh_min',
    'mmr_force_power_share',
    'mmr_identity_corr_min',
    'mmr_kin_corr_min',
    'mmr_kin_slow_r2_max',
    'mmr_subresonance_phi0_coh_min',
    'mmr_subresonance_period_corr_min',
    'mmr_subresonance_t2_r2_min',
    'mmr_subresonance_t2_coh_max',
    'mmr_subresonance_strength_rel_min',
    'mmr_owner_r2_min',
    'mmr_owner_phi0_coh_min',
    'mmr_owner_psi_share_max',
    'mmr_owner_phi0_coh_gain',
    'mmr_owner_cover_min',
    'mmr_o1_owner_cover_min',
    'mmr_owner_identity_coef_tol',
    'separatrix_ve_r2_min',
    'mmr_filter_guard_tp_rel',
    'mmr_filter_guard_amp_rel',
    'mmr_l8_gain_full',
    'mmr_l8_gain_fail',
    'mmr_k1_full',
    'mmr_w_r2_none',
    'mmr_w_r2_full',
    'confidence_thin_band',
    'dryrun_fair_fraction',
    'mmr_varpi_undefined_max_share',
    'mmr_coorbital_qs_well_share',
    'libration_return_tol',
    'libration_return_identity_corr_min',
    'mmr_force_null_gain',
    'mmr_kin_ripple_share_max',
)

# Factors and ratios that must be strictly positive
POSITIVE = (
    'libration_action_step_tp_tol',
    'libration_action_step_energy_drift',
    'libration_regular_period_ratio',
    'libration_edge_arc_rev',
    'libration_return_tol',  # the return mismatches are ranked over it (`geometry.return_mismatch`)
    'libration_return_period_tol',
    'libration_stall_min_cycles',
    'mmr_owner_band_factor',
    'mmr_owner_ripple_factor',
    'mmr_subresonance_cycle_period_ratio',
    'mmr_strength_fit_half_width_deg',
    'mmr_strength_quadrature_tol',
    'mmr_strength_double_floor',
    'separatrix_cutoff_cycles',
    'mmr_response_line_amp_frac',
    'mmr_response_line_margin',
    'mmr_response_max_fraction',
    'mmr_response_line_band_cutoffs',
    'mmr_c5_branch_resid_factor',
    'mmr_owner_potential_model_error',
    'mmr_forced_wave_linear_ratio',
    'libration_circulation_net_rev',
    'libration_union_gap_periods',
    'mmr_kin_drift_shift_ratio',
    'mmr_owner_identity_resid_max',
    'mmr_owner_identity_span_max_deg',
)


def validate_settings(settings) -> None:
    """Raise ValueError with the reason for an out-of-range or inconsistent value."""
    values = {f.name: getattr(settings, f.name) for f in fields(settings)}
    kinds = {f.name: f.metadata['kind'] for f in fields(settings)}
    _check_choices(values)
    _check_ranges(values, kinds)
    for small, large in ORDERED_PAIRS:
        if small in values and large in values and not values[small] < values[large]:
            raise ValueError(f'{small} ({values[small]}) must be below {large} ({values[large]})')
    for small, large in ORDERED_OR_EQUAL:
        if small in values and large in values and not values[small] <= values[large]:
            raise ValueError(f'{small} ({values[small]}) must not exceed {large} ({values[large]})')


def _check_choices(values: dict) -> None:
    for name, choices in CHOICES.items():
        if name in values and values[name] not in choices:
            raise ValueError(f"{name} must be one of {', '.join(choices)}, got {values[name]!r}")


def _check_positive_list(values: Mapping[str, Any], name: str) -> None:
    """A 'floats' setting that must hold at least one value, every one positive."""
    factors = values.get(name)
    if factors is not None and not (len(factors) and all(f > 0 for f in factors)):
        raise ValueError(f'{name} must hold positive values')


def _check_ranges(values: dict, kinds: dict) -> None:
    for name, value in values.items():
        numbers = value if kinds[name] == 'floats' else (value,) if kinds[name] in ('int', 'float', 'optional_float') else ()
        # nan passes every comparison below; an infinite bound compares as one ("no limit") and stays allowed
        if any(number is not None and math.isnan(number) for number in numbers):
            raise ValueError(f'{name} must be a number, got {value}')
        if kinds[name] in ('int', 'float') and value < 0:
            raise ValueError(f'{name} must not be negative, got {value}')
        if name in UNIT_INTERVAL and value > 1.0:
            raise ValueError(f'{name} must lie in [0, 1], got {value}')
    minimum = {
        'libration_min_turning_points': 2,
        'filter_order': 1,
        'mmr_coorbital_grid_points': 4,  # a periodic grid needs a few points
        'mmr_coorbital_scale_ratio_max': 1.0,  # the band [1/x, x] is empty below 1
        'mmr_strength_snapshots': 1,
        'mmr_strength_unsaved_planet_phases': 1,
        'mmr_force_null_shifts': 1,
        'mmr_force_harmonics_max': 1,  # 1 = one harmonic: the N-harmonic law off
        'mmr_separatrix_avgpot_tp_max': 2,  # a profile per turning point; one cycle needs two
        'mmr_strength_nodes_per_turn': 8,  # L / 2 (the convergence twin) must still be a quadrature
        'mmr_strength_sigma_grid': 8,  # finds a well only with a few points per turn of sigma
        'mmr_strength_mp_dps': 16,  # below double precision the extended route has no point
        'mmr_strength_mp_nodes_body': 4,
        'mmr_strength_mp_nodes_planet': 4,
        'mmr_centre_track_nodes': 2,  # a track needs two nodes to interpolate
        'mmr_owner_potential_scale_full': 1.0,  # the band [1/x, x] is empty below 1
        'mmr_owner_potential_snapshots': 1,
        'mmr_coorbital_qs_max_blocks': 1,
        'libration_return_band_levels': 1,
        'libration_return_mid_band_levels': 1,
        'mmr_owner_capture_lowpass_periods': 1.0,  # below one band period the owner's own wave stays in Psi
        'mmr_owner_capture_harmonics': 1,  # V(x) needs at least its first harmonic
        'mmr_owner_capture_samples_per_term': 1,
        'libration_action_step_min_half_cycles': 2,  # a side of G2a needs one closed cycle
        'libration_action_step_cycles': 1,  # G2a confirms by the energy of at least one cycle per side
        'libration_action_step_separation': 1.0,  # below 1 the bands overlap
        'libration_action_step_ratio': 1.0,
        'libration_action_step_a_ratio': 1.0,
    }
    for name, least in minimum.items():
        if name in values and values[name] < least:
            raise ValueError(f'{name} must be at least {least}, got {values[name]}')
    for name in POSITIVE:
        if name in values and not values[name] > 0:
            raise ValueError(f'{name} must be positive, got {values[name]}')
    _check_positive_list(values, 'libration_action_step_cutoffs')
    cutoff = values.get('mmr_filter_cutoff_yr')
    if cutoff is not None and cutoff <= 0:
        raise ValueError('mmr_filter_cutoff_yr must be positive or empty')


def setting_fields(cls=AnalysisSettings) -> tuple[Field, ...]:
    """The fields of a settings dataclass (each has metadata env, kind, meaning, unit, convention)."""
    return fields(cls)


def conventions(cls=AnalysisSettings) -> frozenset[str]:
    """Names of the settings that are agreed conventions (they take part in margin U)."""
    return frozenset(f.name for f in fields(cls) if f.metadata['convention'])


def _parse_bool(text: str) -> bool:
    if text.lower() in ('true', '1', 'yes'):
        return True
    if text.lower() in ('false', '0', 'no'):
        return False
    raise ValueError(f'expected True or False, got {text!r}')


_FROM_TEXT = {
    'int': lambda text: int(float(text)),
    'float': float,
    'optional_float': float,
    'bool': _parse_bool,
    'floats': lambda text: tuple(float(x) for x in text.split(',') if x.strip()),
    'strs': lambda text: tuple(x.strip() for x in text.split(',') if x.strip()),
    'str': str,
}
_FROM_VALUE = {
    'int': int,
    'float': float,
    'optional_float': float,
    'bool': bool,
    'floats': lambda value: tuple(float(x) for x in value),
    'strs': lambda value: tuple(str(x) for x in value),
    'str': str,
}
_EMPTY = {'optional_float': None, 'str': '', 'floats': (), 'strs': ()}


def parse_value(f: Field, raw: Any) -> Any:
    """Convert a raw value (an .env string, or an already typed kwarg) to the field's type.

    An empty string or None means the default, except for optional floats (None), strings ('')
    and lists (()).
    """
    kind = f.metadata['kind']
    if isinstance(raw, str):
        text = raw.strip()
        if text == '':
            return _EMPTY[kind] if kind in _EMPTY else _default(f)
        return _FROM_TEXT[kind](text)
    if raw is None:
        return None if kind == 'optional_float' else _default(f)
    return _FROM_VALUE[kind](raw)


def _default(f: Field) -> Any:
    if f.default is MISSING:  # pragma: no cover - every setting has a default
        raise ValueError(f'{f.name} has no default')
    return f.default


def format_value(value: Any) -> str:
    """The .env.dist spelling of a value (tuples comma-separated, None empty)."""
    if value is None:
        return ''
    if isinstance(value, tuple):
        return ','.join(format_value(v) for v in value)
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return f'{value:.1f}'
    return str(value)


def read_settings(source: Any = None, cls=AnalysisSettings, **overrides):
    """Build a settings object from a SimulationConfig, a mapping, or nothing (the defaults).

    `source` may be any object with the settings as attributes (SimulationConfig), or a mapping
    keyed by field name or by .env key; missing values take the dataclass default. Keyword
    overrides win. Values are parsed with `parse_value`, then validated.
    """
    values = {}
    for f in fields(cls):
        raw = MISSING
        if f.name in overrides:
            raw = overrides[f.name]
        elif isinstance(source, Mapping):
            if f.name in source:
                raw = source[f.name]
            elif f.metadata['env'] in source:
                raw = source[f.metadata['env']]
        elif source is not None and hasattr(source, f.name):
            raw = getattr(source, f.name)
        if raw is not MISSING:
            values[f.name] = parse_value(f, raw)
    unknown = set(overrides) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
    return cls(**values)


def settings_from_config(source: Any = None, **overrides) -> AnalysisSettings:
    """`AnalysisSettings` from a SimulationConfig, a plain dict of values, or the defaults."""
    return read_settings(source, AnalysisSettings, **overrides)


def dry_run_settings_from_config(source: Any = None, **overrides) -> DryRunSettings:
    """`DryRunSettings` from a SimulationConfig, a plain dict of values, or the defaults."""
    return read_settings(source, DryRunSettings, **overrides)


def settings_hash(settings: AnalysisSettings) -> str:
    """Short sha256 of the settings (written as `settings_hash` to simulation.json and summary.csv)."""
    payload = json.dumps(asdict(settings), sort_keys=True, default=list)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
