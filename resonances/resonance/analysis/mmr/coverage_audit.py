"""The 1:1 coverage audit: bounded motion of a co-orbital angle that the verdict does not count (flags, never a status).

It runs after the policy (`analyse.analyse_mmr`, prograde 1:1 only, MMR_COORBITAL_COVERAGE_AUDIT), the only step there,
and writes flags (`verdict.flags`, which enter `review_reasons` and the review queue) and the `aud_*` columns of
mmr_numbers.csv. Nothing here changes the status, the confidence, the caps or the action. Two layers:

Layer I, `floor_contradiction` (flag floor_above_own_loop, no threshold): a period floor must not exceed the period of
the averaged potential's own loop through the turns of the run it rejects. A run that is a loop of that potential has
the loop's period (the YORP QS runs: observed 50.1 / 57.8 yr against loops 48.3 / 56.3 yr), so a floor above its loop
rejects a libration of the very potential the floor comes from. Fires when (i) an accepted run of the analysis has
ineligible_reason below_period_floor and a floor (`decide.period_floor`) above its loop (`cpl_loop_period_yr`: the
median loop over its turn pairs; for a not_reflection reading the loop of its first pair only, which the reading
computes as a column), or (ii) the G1q pass was rejected for a reason
its floor can cause (`FLOOR_REJECTIONS`: below_period_floor directly; no_turning_points, no_run_in_well through the
pass's cutoff P_c = floor / FILTER_MIN_LIBRATION_CUTOFFS and the libration image of its prominence, 54509 c046) while
the loop through the stay's >= 2 turns is shorter than the QS floor (`geo_qs_visit_loop_yr` < `flt_qs_floor_yr`; one
turn is the orbit at rest, no loop: 2023 CU). A pass rejected as keeps_outside (its in-well episode was found) or
not_accepted:<reason> (the geometry rejected the run itself: too_few_tp, over_the_top) is not the floor's doing.

Layer II, `audit` (flag uncovered_bounded_run): independent of the filter, the prominence and the floor, it reads the
raw angle averaged over MMR_COORBITAL_EDGE_TURN_ORBITS orbital periods (the C4 average, which drops the orbit's own
terms; 3 yr for an Earth co-orbital, the survey sweep's boxcar). Turning points by hysteresis: the averaged angle must
retrace h = max(MMR_COORBITAL_AUDIT_MIN_SWING_DEG, MMR_COORBITAL_AUDIT_NOISE_FACTOR x noise) from its running extreme
(noise = 1.4826 MAD of the averaged angle less its own average over the same window: the jitter the average leaves).
A bounded run is >= MMR_COORBITAL_AUDIT_MIN_HALF_CYCLES consecutive half-cycles whose turning points span less than a
turn and whose same-kind turns stay put, |x_(i+2) - x_i| <= 2 MMR_COORBITAL_AUDIT_CENTRE_TOL min(A_i, A_(i+1)) (a wave
riding on a drift moves its centre by more than its swing each cycle and fails: 4t + 20 sin(2 pi t / 20)). A half-cycle
is uncovered when its midpoint lies outside every counted episode and span of the policy (`PolicyResult`); a missed
stretch is >= MIN_HALF_CYCLES consecutive uncovered half-cycles inside the analysis window. A stretch with fewer inside
the window is an edge stretch (`aud_edge_half_cycles` only): the record's ends are not analysed (docs: limitations).
The flag needs the force too (the caller's `law_of`): a missed stretch counts only when the local 1:1 law on it passes
the co-orbital force check, not thin (R^2 >= MMR_FORCE_R2_FULL in both halves, K / K_theory within
MMR_COORBITAL_SCALE_RATIO_MAX): the pairwise centre test lets a slow drift through (0.5 deg/yr under a 20-deg, 25-yr
wave: 12.5 deg per cycle against A = 20 deg, 49-71 half-cycles, K -0.8 and 0.3), and so does 2016 JA's tadpole
(K 2.34, R^2 0.46). The law is fitted on at most AUDIT_LAW_STRETCHES_MAX stretches, longest first (cost).
Not measured (columns nan, `aud_not_measured`): a step not below half the averaging window (the average would be the
raw samples, the orbit's own terms would make turns) or non-finite samples (the running sum would carry them on).

Measured on the Earth survey (5 clone runs, the sweep `followup/qs-sweep/missed_runs_audit.py` this ports): 453
stretches in 434 clones, 306 of them quasi-satellite or tadpole inside the window; the horseshoe (14), compound (13) and
TP300 (7) hits are marginal or forced waves, which is why the audit is a flag. Pure: arrays in, no Body/Simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis.mmr.quasi_satellite import PASS_REJECTED, REJECT_NO_RUN_IN_WELL, REJECT_NO_TP
from resonances.resonance.analysis.pendulum import MAD_TO_SIGMA
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Flag

NAN = float('nan')
TWO_PI = 2.0 * np.pi
FLAG_UNCOVERED = str(Flag.UNCOVERED_BOUNDED_RUN)
FLAG_FLOOR_ABOVE_LOOP = str(Flag.FLOOR_ABOVE_OWN_LOOP)
# aud_unc_overlap_reason without an overlapping run of the classifier
OVERLAP_NO_RUN = 'no_run'
OVERLAP_QS_REJECTED = 'qs_pass_rejected'
OVERLAP_NOT_COUNTED = 'eligible_not_counted'  # an accepted, eligible run that the policy did not count
# aud_unc_law_check: the local law's force check on the stretch (`coorbital.force_check`; thin = a pass with R^2 below
# mmr_force_r2_full in a half); Layer II flags only LAW_PASS
LAW_PASS, LAW_THIN, LAW_NOT_MEASURED = 'pass', 'thin', 'not_measured'
AUDIT_LAW_STRETCHES_MAX = 3  # numerical (cost): the law is fitted on at most this many missed stretches, longest first
# aud_not_measured
NOT_MEASURED_STEP, NOT_MEASURED_SAMPLES = 'step_not_below_half_window', 'nonfinite_samples'
REJECT_BELOW_FLOOR = 'below_period_floor'  # the run's ineligible_reason (`decide.episode_eligibility`)
# Layer I (ii): the G1q rejections (geo_qs_reject) the pass's floor can cause (module docstring)
FLOOR_REJECTIONS = (REJECT_BELOW_FLOOR, REJECT_NO_TP, REJECT_NO_RUN_IN_WELL)

AUDIT_NUMBERS = (
    'aud_n_bounded_runs',
    'aud_n_uncovered',
    'aud_edge_half_cycles',
    'aud_unc_t0_yr',
    'aud_unc_t1_yr',
    'aud_unc_half_cycles',
    'aud_unc_period_yr',
    'aud_unc_half_amp_deg',
    'aud_unc_centre_deg',
    'aud_unc_law_K',
    'aud_unc_law_R2',
    'aud_n_law_confirmed',
    'aud_noise_deg',
    'aud_swing_deg',
    'aud_floor_yr',
    'aud_loop_yr',
)
AUDIT_TEXTS = ('aud_unc_well', 'aud_unc_overlap_reason', 'aud_unc_law_check', 'aud_floor_above_loop', 'aud_not_measured')


def empty_columns() -> Dict[str, object]:
    """The aud_* columns of a row the audit did not read (another angle, the audit off): nan and ''."""
    return {**{name: NAN for name in AUDIT_NUMBERS}, **{name: '' for name in AUDIT_TEXTS}}


@dataclass(frozen=True)
class BoundedRun:
    """Half-cycles k0..k1 of the averaged angle (turning points k0..k1 + 1) that stay about one centre."""

    k0: int
    k1: int


@dataclass(frozen=True)
class Stretch:
    """A stretch of half-cycles [t0, t1] (turning point times) of a bounded run: its numbers in the averaged angle."""

    t0_yr: float
    t1_yr: float
    half_cycles: int
    period_yr: float  # 2 x the median half-cycle
    half_amp_rad: float  # the median half-amplitude
    centre_rad: float  # the median mid-level of its half-cycles (unwrapped)


@dataclass
class CoverageAudit:
    """What `audit` found: the bounded runs, the missed stretches inside the window (longest first), those whose
    force law passes (`confirmed`; every missed one without a `law_of`), the uncovered half-cycles of edge stretches,
    and the columns (the hysteresis and noise among them); `flag` the word for verdict.flags."""

    runs: List[Stretch] = field(default_factory=list)
    missed: List[Stretch] = field(default_factory=list)
    confirmed: List[Stretch] = field(default_factory=list)
    edge_half_cycles: int = 0
    columns: Dict[str, object] = field(default_factory=empty_columns)

    @property
    def flag(self) -> str:
        return FLAG_UNCOVERED if self.confirmed else ''


def hysteresis_turns(x: np.ndarray, h: float) -> np.ndarray:  # noqa: C901 (one zigzag pass)
    """Indices of the confirmed turning points of `x` with hysteresis `h`: an extreme reached by a move >= h and
    followed by a retrace >= h; the record ends are never turning points, and they alternate max / min."""
    x = np.asarray(x, dtype=float)
    step = np.sign(np.diff(x))
    moving = np.flatnonzero(step)
    if len(moving) < 2:
        return np.array([], dtype=int)
    change = np.flatnonzero(step[moving][1:] != step[moving][:-1])
    candidates = np.concatenate([[0], moving[change] + 1, [len(x) - 1]])  # the local extrema and the ends
    v = x[candidates]
    turns, mode, extreme, high, low, j = [], 0, 0, 0, 0, 1
    while j < len(v) and mode == 0:  # the first move of h from the start fixes the direction
        high = j if v[j] > v[high] else high
        low = j if v[j] < v[low] else low
        if v[high] - v[low] >= h:
            mode, extreme = (1, high) if low < high else (-1, low)
        j += 1
    for k in range(j, len(v)):
        if mode == 1:
            if v[k] > v[extreme]:
                extreme = k
            elif v[extreme] - v[k] >= h:
                turns.append(extreme)
                mode, extreme = -1, k
        else:
            if v[k] < v[extreme]:
                extreme = k
            elif v[k] - v[extreme] >= h:
                turns.append(extreme)
                mode, extreme = 1, k
    return candidates[np.array(turns, dtype=int)] if turns else np.array([], dtype=int)


def bounded_runs(values: np.ndarray, centre_tol: float, min_half_cycles: int) -> List[BoundedRun]:
    """The maximal runs of >= `min_half_cycles` consecutive half-cycles between the turning points `values` [rad] whose
    turns span less than a turn and whose same-kind turns stay within 2 centre_tol x the smaller neighbouring
    half-amplitude of each other (module docstring)."""
    pv = np.asarray(values, dtype=float)
    half = np.abs(np.diff(pv)) / 2.0
    n = len(half)
    same_centre = [abs(pv[k + 2] - pv[k]) <= 2.0 * centre_tol * min(half[k], half[k + 1]) for k in range(n - 1)]
    runs, k = [], 0
    while k < n:
        j = k
        lo, hi = min(pv[k], pv[k + 1]), max(pv[k], pv[k + 1])
        while j + 1 < n and same_centre[j]:
            lo2, hi2 = min(lo, pv[j + 2]), max(hi, pv[j + 2])
            if hi2 - lo2 >= TWO_PI:
                break
            lo, hi = lo2, hi2
            j += 1
        if j - k + 1 >= min_half_cycles:
            runs.append(BoundedRun(k, j))
        k = j + 1
    return runs


def stretch_of(times: np.ndarray, values: np.ndarray, k0: int, k1: int) -> Stretch:
    """The `Stretch` of half-cycles k0..k1 (turning points k0..k1 + 1)."""
    tp, pv = np.asarray(times[k0 : k1 + 2], dtype=float), np.asarray(values[k0 : k1 + 2], dtype=float)
    return Stretch(
        t0_yr=float(tp[0]),
        t1_yr=float(tp[-1]),
        half_cycles=int(k1 - k0 + 1),
        period_yr=float(2.0 * np.median(np.diff(tp))),
        half_amp_rad=float(np.median(np.abs(np.diff(pv)) / 2.0)),
        centre_rad=float(np.median((pv[:-1] + pv[1:]) / 2.0)),
    )


def _covered(t_yr: float, spans: Sequence[Tuple[float, float]]) -> bool:
    return any(lo <= t_yr <= hi for lo, hi in spans)


def audit(
    t_yr: np.ndarray,
    raw_unwrapped: np.ndarray,
    orbital_period_yr: float,
    window: Tuple[float, float],
    counted_spans: Sequence[Tuple[float, float]],
    settings: AnalysisSettings,
    well_of: Optional[Callable[[Stretch], str]] = None,
    law_of: Optional[Callable[[Stretch], Tuple[float, float, str]]] = None,
    overlap_of: Optional[Callable[[Stretch], str]] = None,
) -> CoverageAudit:
    """Layer II on one 1:1 record (module docstring): `raw_unwrapped` sigma [rad] on `t_yr`, averaged over
    mmr_coorbital_edge_turn_orbits x `orbital_period_yr`; `window` the analysis window [yr]; `counted_spans` the
    counted episodes and spans [yr]. The caller's `law_of` ((K / K_theory, R^2, force check) of the local 1:1 law on a
    stretch) gates the flag (module docstring; None: every missed stretch counts); `well_of` (the well a stretch holds)
    and `overlap_of` (the classifier's reason for the run overlapping it) fill the columns of the longest confirmed
    stretch, else of the longest missed one, and decide nothing."""
    from resonances.resonance.analysis.geometry import orbit_averaged  # geometry does not import this module

    result = CoverageAudit()
    t = np.asarray(t_yr, dtype=float)
    raw = np.asarray(raw_unwrapped, dtype=float)
    width = settings.mmr_coorbital_edge_turn_orbits * float(orbital_period_yr)
    columns = result.columns
    step = float(np.median(np.diff(t))) if len(t) >= 2 else NAN
    if not (np.isfinite(width) and np.isfinite(step) and 0 < step and round(width / step / 2) >= 1):
        columns['aud_not_measured'] = NOT_MEASURED_STEP  # orbit_averaged would return the raw samples
        return result
    if not np.all(np.isfinite(raw)):
        columns['aud_not_measured'] = NOT_MEASURED_SAMPLES  # its running sum would carry a nan to the record's end
        return result
    averaged = orbit_averaged(t, raw, width)
    jitter = averaged - orbit_averaged(t, averaged, width)
    jitter = jitter[np.isfinite(jitter)]
    noise = MAD_TO_SIGMA * float(np.median(np.abs(jitter - np.median(jitter)))) if len(jitter) else 0.0
    h = max(math.radians(settings.mmr_coorbital_audit_min_swing_deg), settings.mmr_coorbital_audit_noise_factor * noise)
    turns = hysteresis_turns(averaged, h)
    minimum = int(settings.mmr_coorbital_audit_min_half_cycles)
    columns.update(aud_noise_deg=math.degrees(noise), aud_swing_deg=math.degrees(h), aud_n_bounded_runs=0.0)
    columns.update(aud_n_uncovered=0.0, aud_edge_half_cycles=0.0, aud_n_law_confirmed=0.0)
    if len(turns) < minimum + 1:
        return result
    times, values = t[turns], averaged[turns]
    runs = bounded_runs(values, settings.mmr_coorbital_audit_centre_tol, minimum)
    result.runs = [stretch_of(times, values, r.k0, r.k1) for r in runs]
    result.missed, result.edge_half_cycles = _missed_stretches(times, values, runs, window, counted_spans, minimum)
    laws = {i: law_of(s) for i, s in enumerate(result.missed[:AUDIT_LAW_STRETCHES_MAX])} if law_of is not None else {}
    if law_of is None:
        result.confirmed = list(result.missed)
    else:
        result.confirmed = [result.missed[i] for i, law in laws.items() if law[2] == LAW_PASS]
    columns.update(
        aud_n_bounded_runs=float(len(runs)),
        aud_n_uncovered=float(len(result.missed)),
        aud_edge_half_cycles=float(result.edge_half_cycles),
        aud_n_law_confirmed=float(len(result.confirmed)),
    )
    if result.missed:
        index = next((i for i, law in laws.items() if law[2] == LAW_PASS), 0)
        best = result.missed[index]
        centre = math.degrees(float(np.mod(best.centre_rad + np.pi, TWO_PI) - np.pi))
        k_scale, r2, check = laws.get(index, (NAN, NAN, ''))
        columns.update(
            aud_unc_t0_yr=best.t0_yr,
            aud_unc_t1_yr=best.t1_yr,
            aud_unc_half_cycles=float(best.half_cycles),
            aud_unc_period_yr=best.period_yr,
            aud_unc_half_amp_deg=math.degrees(best.half_amp_rad),
            aud_unc_centre_deg=centre,
            aud_unc_well=well_of(best) if well_of is not None else '',
            aud_unc_overlap_reason=overlap_of(best) if overlap_of is not None else '',
            aud_unc_law_K=float(k_scale),
            aud_unc_law_R2=float(r2),
            aud_unc_law_check=str(check),
        )
    return result


def _missed_stretches(times, values, runs: Sequence[BoundedRun], window, counted_spans, minimum: int) -> Tuple[List[Stretch], int]:
    """(the missed stretches inside the window, longest first; the uncovered half-cycles of edge stretches) of the
    bounded `runs` between the turning points (`times`, `values`): uncovered half-cycles by their midpoint."""
    missed, edge = [], 0
    lo, hi = float(window[0]), float(window[1])
    for run in runs:
        k = run.k0
        while k <= run.k1:
            if _covered(0.5 * (times[k] + times[k + 1]), counted_spans):
                k += 1
                continue
            j = k
            while j + 1 <= run.k1 and not _covered(0.5 * (times[j + 1] + times[j + 2]), counted_spans):
                j += 1
            if j - k + 1 >= minimum:  # an uncovered stretch: its half-cycles inside the window decide
                inside = [i for i in range(k, j + 1) if times[i] >= lo and times[i + 1] <= hi]
                if len(inside) >= minimum:
                    missed.append(stretch_of(times, values, inside[0], inside[-1]))
                else:
                    edge += j - k + 1
            k = j + 1
    missed.sort(key=lambda s: (s.half_cycles, s.t1_yr - s.t0_yr), reverse=True)
    return missed, edge


@dataclass(frozen=True)
class FloorContradiction:
    """Layer I's finding: where ('run:<id>' or 'qs_pass:<geo_qs_reject>'), the floor and the loop it exceeds [yr]."""

    where: str
    floor_yr: float
    loop_yr: float


def floor_contradiction(runs: Sequence[Tuple[object, str, float, float]], qs_columns: Dict[str, object]) -> Optional[FloorContradiction]:
    """Layer I (module docstring): `runs` the accepted runs of the analysis as (episode_id, ineligible_reason, floor
    [yr], its loop [yr]), `qs_columns` the row's G1q columns (`geo_qs_reject` among them). The first
    contradiction, None without one. The largest floor / loop ratio among the runs is reported."""
    found = [
        FloorContradiction(f'run:{episode_id}', float(floor), float(loop))
        for episode_id, reason, floor, loop in runs
        if reason == REJECT_BELOW_FLOOR and np.isfinite(loop) and loop > 0 and np.isfinite(floor) and floor > loop
    ]
    if found:
        return max(found, key=lambda c: c.floor_yr / c.loop_yr)
    turns = _number(qs_columns.get('geo_qs_visit_turns'))
    loop, floor = _number(qs_columns.get('geo_qs_visit_loop_yr')), _number(qs_columns.get('flt_qs_floor_yr'))
    rejected = str(qs_columns.get('geo_qs_pass', '') or '') == PASS_REJECTED
    reason = str(qs_columns.get('geo_qs_reject', '') or '')
    if rejected and reason in FLOOR_REJECTIONS and turns >= 2 and np.isfinite(loop) and loop > 0 and np.isfinite(floor) and loop < floor:
        return FloorContradiction(f'qs_pass:{reason}', floor, loop)
    return None


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN
