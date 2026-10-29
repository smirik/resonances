"""Geometry of a filtered, unwrapped angle over the whole record (shared by every resonance type).

Turning points (TP), the centre and its largest monotone pass, runs of TPs and episodes, slips
(pattern A), revolutions and stalls (pattern B), arc/stall without TPs, visible exit and entry,
jumps of the slow variable (a for an MMR) and the last regime. Pure arrays in, dataclasses out:
times in years (increasing), angles in radians, the slow variable in its own unit.

Definitions (tmp/methodichka-2026-10-14/inputs/definitions.md, SPEC §4.3–4.4): a TP is a local
maximum or minimum of sigma_f with prominence >= p_TP, the sequence made alternating; one full
cycle is `libration_min_turning_points` TPs; an episode is a run of >= that many TPs with a
stationary or modulated centre and at least one closed full cycle (`closure_ratios`).

Two measurement conventions not fixed by the SPEC text:
- a pass of the centre is measured by max/min (AUTHOR-FEEDBACK 2): on each stretch between centre
  turns (and the window edges) delta = +-(max c - min c), signed by which extreme comes first;
- a slip is an over-the-top half-cycle (half-amplitude >= pi) whose neighbouring midpoints differ
  by a whole number of turns (+- tolerance); its duration is the time the angle needs to go from the old midpoint
  level to the new one (a full turn: >= T0/2 for a pendulum with h <= 1.59, while an unwrap error
  smeared by the filter takes ~P_c plus at most a quarter of a libration period).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from resonances.resonance.analysis import pendulum
from resonances.resonance.analysis.filtering import edge_window, linear_fit, lowpass_or_none, time_derivative, window_mask
from resonances.resonance.analysis.ledger import Ledger, Op
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue, Flag, join_list

NAN = float('nan')
TWO_PI = 2 * np.pi


def wrap_pi(x):
    """Wrap angles [rad] into [-pi, pi)."""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


class EpisodeShape(StrEnum):
    STATIONARY = 'stationary'  # bands of the first and last full cycle overlap
    MODULATED = 'modulated'  # bounded centre turning back >= libration_centre_returns_modulated times
    MOVING_CENTRE = 'moving_centre'  # the bands overlap only in psi = sigma - c(t), c the caller's well track (`MovingWell`)
    NONE = ''


FRAME_SIGMA, FRAME_PSI = 'sigma', 'psi'  # the frame whose band test accepted an episode
WELL_OK = 'ok'  # a track of the well's centre exists along the run
WELL_INVALID = 'invalid_track'  # it exists but fails `well_track_valid`


@dataclass(frozen=True)
class MovingWell:
    """The centre c(t) of the potential well a run librates in, from the caller's model of it (MMR:
    `mmr.centre_track`, the minimum of the averaged disturbing function), on nodes in time; `status` WELL_OK, else why
    there is none (the nodes are then empty)."""

    status: str
    t_yr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    centre_rad: np.ndarray = field(default_factory=lambda: np.zeros(0))  # unwrapped on sigma's branch

    @property
    def available(self) -> bool:
        return self.status == WELL_OK and len(self.t_yr) > 0

    def at(self, times_yr) -> np.ndarray:
        """c at `times_yr` [rad]: linear between the nodes, the end values beyond them."""
        return np.interp(np.asarray(times_yr, dtype=float), self.t_yr, self.centre_rad)


# the caller's model: (TP times [yr], TP values [rad]) of a run -> its MovingWell
WellModel = Callable[[np.ndarray, np.ndarray], MovingWell]


class RejectReason(StrEnum):
    TOO_FEW_TP = 'too_few_tp'
    CENTRE_MOVES = 'centre_moves'
    OVER_THE_TOP = 'over_the_top'  # too few TPs because the angle went over the top next to the run
    EDGE = 'edge'  # the slow edge half-cycle trimmed off a run whose centre moves (`_trim_slow_edges`)
    NO_CLOSED_CYCLE = 'no_closed_cycle'  # no full cycle of the run returns about its centre (`closure_ratios`)
    NONE = ''


@dataclass(frozen=True)
class LocalThreshold:
    """A run of provisional turning points [start_yr, end_yr] (`local_thresholds`) whose extrema are selected at the
    prominence measured on its own samples [rad] instead of the record's."""

    start_yr: float
    end_yr: float
    prominence_rad: float


@dataclass(frozen=True)
class LocalProminence:
    """How a type measures the TP prominence of a stretch (MMR: `analyse.regime_prominence`): `floor_rad`, the type's
    prominence floor, selects the provisional turning points (`local_thresholds`); calling it with the mask of a
    stretch's samples gives that stretch's prominence [rad], NaN when the stretch is too short to measure it."""

    floor_rad: float
    of_samples: Callable[[np.ndarray], float]

    def __call__(self, own: np.ndarray) -> float:
        return self.of_samples(own)


@dataclass(frozen=True)
class TurningPoints:
    """Alternating extrema of an angle; `kinds` +1 for a maximum, -1 for a minimum. `prominence_rad` is the threshold
    of the record (or of its regime between jumps); inside a stretch of `local` the stretch's own threshold applies."""

    indices: np.ndarray  # int, into the series arrays
    times_yr: np.ndarray
    values_rad: np.ndarray
    kinds: np.ndarray
    prominence_rad: float
    local: Tuple[LocalThreshold, ...] = ()

    def __len__(self) -> int:
        return len(self.indices)

    def thresholds(self, run: Optional[np.ndarray] = None) -> np.ndarray:
        """The prominence each TP (of `run`, index array into the TPs; all without it) was selected at [rad]."""
        times = np.asarray(self.times_yr, dtype=float)
        times = times if run is None else times[np.asarray(run, dtype=int)]
        return threshold_at(times, self.prominence_rad, self.local)

    def run_prominence(self, run: np.ndarray) -> float:
        """The prominence of a run's turning points: the largest threshold among them [rad] (the record's for none)."""
        values = self.thresholds(run)
        return float(values.max()) if len(values) else float(self.prominence_rad)


@dataclass(frozen=True)
class MonotonePass:
    """A stretch of the centre between two of its turns (or a window edge)."""

    t_start_yr: float
    t_end_yr: float
    delta_rad: float  # c(end) - c(start), signed


@dataclass
class CentreTrack:
    """The centre c(t): the angle smoothed over `libration_centre_cycles` cycles, and its passes."""

    t_yr: np.ndarray
    centre_rad: np.ndarray
    window_yr: float  # smoothing period W_c
    turns: TurningPoints  # turning points of the centre (prominence p_c)
    passes: List[MonotonePass]
    max_pass_rad: float  # largest |delta| over the passes
    max_pass_t0_yr: float
    max_pass_t1_yr: float
    net_rev: float  # (c(t1) - c(t0)) / 2pi
    range_rad: float  # ptp of c
    max_pass_rev_lo: float = float('nan')  # the largest pass at smoothing x0.5 and x2 (min, max), revolutions
    max_pass_rev_hi: float = float('nan')

    @property
    def max_pass_rev(self) -> float:
        return self.max_pass_rad / (2 * np.pi)


@dataclass
class Episode:
    """A run of alternating TPs, accepted as an episode or rejected with a reason (episodes.csv row).

    Geometric part only; the pendulum, energy and response numbers of the episode are added by
    other modules under the same `episode_id`.
    """

    episode_id: int
    tp_times_yr: np.ndarray
    tp_values_rad: np.ndarray
    accepted: bool
    shape: EpisodeShape
    reject_reason: RejectReason
    period_med_yr: float  # full periods t(TP_i+2) - t(TP_i)
    period_min_yr: float
    period_max_yr: float
    half_amp_min_rad: float  # A_i = |sigma(TP_i) - sigma(TP_i+1)| / 2
    half_amp_med_rad: float
    half_amp_max_rad: float
    centre_mod_2pi_rad: float  # mean midpoint m = (sigma(TP_i) + sigma(TP_i+1)) / 2, mod 2pi
    centre_shift_rad: float  # m_last - m_first (full cycles)
    centre_shift_over_amp: float  # |shift| / mean A
    bands_overlap: bool  # the bands m_i +- A_i of every full cycle share a point (`centre_bands_overlap`)
    centre_returns: int  # turns of the full-cycle midpoints inside the run (prominence p_c)
    coverage_start_gap_p: float  # (t_start - t0) / P
    coverage_end_gap_p: float  # (t1 - t_end) / P
    before_last_jump: bool = False  # cut off by the last regime (kept for the CSV)
    # the gaps between the analysis window's edges and the first/last TP, and the longest TP-to-TP step inside the run
    # [yr]: a gap not longer than that step misses no turning point (`covers_edges`)
    window_start_gap_yr: float = NAN
    window_end_gap_yr: float = NAN
    tp_step_max_yr: float = NAN
    # the stretches between the record's and the window's edges, hidden by the filter's trim, in periods of the run
    window_start_hidden_p: float = NAN
    window_end_hidden_p: float = NAN
    # independent estimates of the centre: full cycles without a shared half-cycle, (n_TP - 1) // 2
    centre_estimates: int = 0
    # pass: >= libration_centre_min_estimates estimates and every band overlaps; not_measured: fewer estimates
    # (three TPs are one cycle, one centre: nothing to compare); fail: the bands do not share a point
    centre_stationarity: str = ''
    # full cycles (TP_i, TP_i+1, TP_i+2) in which sigma returns, closure ratio < 1 (`closure_ratios`); an episode needs
    # one. The smallest closure ratio of the run (nan without a full cycle)
    closed_cycles: int = 0
    closure_ratio_min: float = NAN
    # the prominence its turning points were selected with [rad]: that of its run of provisional turning points
    # (`local_thresholds`), else of its regime between jumps of the slow variable (`finish_geometry`), else the record's;
    # the largest of them when its TPs come from several stretches
    tp_prominence_rad: float = NAN
    # the whole selection profile of its turning points: the record's (or regime's) prominence [rad] and the stretches
    # at their own (`TurningPoints.local`); a recount at other cutoffs uses it (`selection_thresholds`)
    tp_base_prominence_rad: float = NAN
    tp_local: Tuple[LocalThreshold, ...] = ()
    # the moving centre (`_classify_run`): the frame whose band test accepted the run (FRAME_SIGMA, FRAME_PSI; '' for a
    # rejected run), the band margins min(m + A) - max(m - A) over the full cycles in sigma and in psi = sigma - c(t)
    # [rad, > 0 passes], the half-swing margin (every maximum above every minimum: a column, never a test), the
    # well's track when it was evaluated (only for a run the sigma test rejects), its status ('' = not evaluated) and
    # the RMS of the full-cycle centres about c(t), about their mean and about the time-reversed track [rad]
    frame: str = ''
    margin_sigma_rad: float = NAN
    margin_psi_rad: float = NAN
    half_swing_margin_rad: float = NAN
    well: Optional[MovingWell] = None
    well_status: str = ''
    well_rms_rad: float = NAN
    well_rms_fixed_rad: float = NAN
    well_rms_reversed_rad: float = NAN
    # MMR edge rule (variant a, `_visible_edge`): the orbit-averaged raw angle runs >= libration_edge_arc_rev turns
    # away from the centre beyond the last (first) turning point, so the record shows the exit (entry): that edge is
    # not covered (`covers_edges`). Set on the last (first) accepted run only
    exit_observed: bool = False
    entry_observed: bool = False
    # MMR closure by phase return (`phase_return`): a 2-TP run accepted as one cycle; the return crossings [yr]
    closed_by_return: bool = False
    return_t_a_yr: float = NAN
    return_t_b_yr: float = NAN
    return_level_fraction: float = NAN  # where between v2 (0) and v1 (1) the return was measured; 0.5 = the mid-level

    @property
    def well_drift_rad(self) -> float:
        """c(t_end) - c(t_start) of the well's track over the run [rad] (nan without a track)."""
        if self.well is None or not len(self.well.t_yr):
            return NAN
        ends = self.well.at([self.t_start_yr, self.t_end_yr])
        return float(ends[1] - ends[0])

    @property
    def n_tp(self) -> int:
        return len(self.tp_times_yr)

    def selection_thresholds(self) -> Tuple[float, Tuple[LocalThreshold, ...]]:
        """(prominence [rad], local stretches) its turning points were selected with, for `find_turning_points`; an
        episode built without a profile: its `tp_prominence_rad` alone."""
        if np.isfinite(self.tp_base_prominence_rad):
            return float(self.tp_base_prominence_rad), tuple(self.tp_local)
        return float(self.tp_prominence_rad), ()

    @property
    def t_start_yr(self) -> float:
        return float(self.tp_times_yr[0])

    @property
    def t_end_yr(self) -> float:
        return float(self.tp_times_yr[-1])

    @property
    def centre_mean_rad(self) -> float:
        """Mean midpoint of the run on the unwrapped branch [rad] (c_ep of the exit test)."""
        values = np.asarray(self.tp_values_rad, dtype=float)
        if len(values) < 2:
            return float(values[0]) if len(values) else NAN
        return float(np.mean(midpoint_values(values)))


@dataclass(frozen=True)
class Slip:
    """A step of 2pi (1 +- tolerance) of the oscillation midpoint (pattern A)."""

    time_yr: float
    size_rev: float  # signed, in revolutions
    duration_yr: float  # time over which the step accumulated
    suspect: bool  # coincides with an unwrap_step_suspect step or a jump: not a slip


@dataclass(frozen=True)
class Revolution:
    """One full turn of a circulating angle (pattern B is read from these)."""

    t_start_yr: float
    t_end_yr: float
    phase_min_rate_rad: float  # sigma mod 2pi where sigma_dot is smallest
    rate_min_over_mean: float  # min sigma_dot / mean sigma_dot over the turn
    energy_h: float = float('nan')  # h of the turn under the sigma-only potential (pendulum.py)


@dataclass(frozen=True)
class PatternB:
    """Stalls of a circulation at the top of the potential (SPEC §4.3.5)."""

    n_revolutions: int
    stall_fraction: float  # fraction of turns with the rate minimum within the tolerance of the top
    stall_phase_mean_rad: float
    stall_phase_r: float  # |mean e^{i phase}|
    rate_min_over_mean_med: float
    rate_min_over_mean_pend_med: float  # r_pend(h) of the same turns
    h_rev_med: float
    h_rev_max: float
    in_zone: bool  # every h_rev <= h_hi + margin * sigma_h
    detected: bool  # in_zone and stall_fraction >= libration_stall_rev_fraction
    rate_inconsistent: bool  # |r_min - r_pend| > libration_stall_rate_tol (flag)


@dataclass(frozen=True)
class ArcStall:
    """No turning points: an arc (range >= pi) or a stall (< pi), with strength bounds (SPEC §4.3.6)."""

    kind: str  # 'arc' | 'stall'
    tau_yr: float
    range_rad: float  # ptp of sigma_f in the window
    rate_rad_yr: float  # mean sigma_dot
    acc_rad_yr2: float  # sigma_ddot of a quadratic fit
    wrapped_min_rad: float
    wrapped_max_rad: float
    slow_ptp: float  # ptp of the slow variable (a [au] for an MMR)
    slow_trend_per_yr: float
    force_bound_range: float  # K bounds [slow unit / yr]; 'range' decides
    force_bound_trend: float
    force_bound_curvature: float
    force_bound_rate: float
    period_min_yr: float  # P_lib,min = 2 pi sqrt(tau |sin| / (|c| d_slow))


@dataclass(frozen=True)
class EdgeEvent:
    """A visible exit after the last TP of the last episode (or entry before the first)."""

    kind: str  # 'exit' | 'entry'
    time_yr: float
    beyond_rad: float  # max |sigma_f - c_ep| - pi
    remaining_periods: float  # record left after the exit (before the entry), in episode periods
    suspected: bool  # remaining < libration_coverage_periods, and the exit (entry) not observed
    # the raw series after the crossing is monotonic (confirms the exit); with the MMR edge rule the orbit-averaged
    # angle from the turning point to the record's edge
    raw_monotonic: bool
    # MMR edge rule: net travel of the orbit-averaged raw angle from the episode's last turning point to the record's
    # end (entry: from the record's start to its first one) [rad]; nan when the rule is off
    raw_arc_rad: float = NAN
    observed: bool = False  # raw_arc >= 2 pi x libration_edge_arc_rev and raw_monotonic (the edge rule)


@dataclass(frozen=True)
class JumpCandidate:
    time_yr: float
    size: float  # slow-variable step over tau
    duration_yr: float


@dataclass(frozen=True)
class Jump:
    """A confirmed jump: the cycle energy changes across it (SPEC §4.3.9)."""

    time_yr: float
    size: float
    duration_yr: float
    dh: float  # h_after - h_before; nan when decided by the slow variable only
    by_slow_only: bool  # fewer than libration_jump_cycles on a side: decided by |step| > half-width
    kick: bool  # does not cut the record: a step without an energy change, or one `at_edge`
    at_edge: bool = False  # closer to a window edge than the data needed to confirm it (`jump_edge_margin_yr`)
    # G2a (`ActionStep`): a step of the libration action, not of a; its zone is the transitional half-cycle
    # [TP_k, TP_k+1] (both turning points stay with their regimes) instead of time -+ tau/2 (`zone`)
    action_step: bool = False
    zone_start_yr: float = NAN
    zone_end_yr: float = NAN

    def zone(self, tau_yr: float) -> Tuple[float, float]:
        """The stretch no regime owns [yr]: the end of the regime before it and the start of the one after it. A jump
        of a is located within +-tau/2; an action step owns its transitional half-cycle."""
        if np.isfinite(self.zone_start_yr) and np.isfinite(self.zone_end_yr):
            return float(self.zone_start_yr), float(self.zone_end_yr)
        half = tau_yr / 2 if np.isfinite(tau_yr) else 0.0
        return float(self.time_yr - half), float(self.time_yr + half)


@dataclass(frozen=True)
class ActionStep:
    """G2a, a step of the libration action inside one accepted run (W24; 2003 TX9 3E+2J-10): the best split of the
    run's half-amplitudes A_j (the split with the largest band separation; the transitional half-cycle A_k between
    TP_k and TP_k+1 belongs to neither side) and the tests it passed or failed (`failed`, '' when it cuts the record).

    Between encounters the libration's action is an adiabatic invariant; a change of the amplitude by a large factor
    within one half-cycle is an impulse that puts the orbit on another energy level. One potential and one period law
    over both levels cannot hold, so the record is cut there as at a confirmed jump of a (rule 10, G1e, G1l); the
    a-locator misses such a step when it changes the swing of a and not its mean (TX9: +2e-5 au against a half-width of
    2.9e-4 au, and its baseline tau >= T0/pi is aliased)."""

    run_t_start_yr: float
    run_t_end_yr: float
    t_from_yr: float  # TP_k: the last turning point of the side before
    t_to_yr: float  # TP_k+1: the first turning point of the side after
    n_before: int  # half-cycles on each side
    n_after: int
    amp_before_rad: float  # median half-amplitude of each side
    amp_after_rad: float
    separation: float  # min(larger side) / max(smaller side)
    ratio: float  # median(larger side) / median(smaller side)
    a_ratio: float  # std of a_f on the larger side over the smaller one (< 1: a disagrees with sigma)
    transition_periods: float  # (t_to - t_from) / the shorter median full period of the two sides
    cycles_before: int  # cycles of the energy analysis inside each side
    cycles_after: int
    dh: float  # mean h of up to libration_jump_cycles cycles after minus before (nan: not measured)
    dh_threshold: float
    failed: str  # the first test that failed ('' = cuts; STEP_WITHHELD: passed, but `analyse` kept the record whole)
    provisional: bool  # cuts with fewer than libration_jump_cycles energy cycles on a side, or test 7 in sample
    h_before: float = NAN  # median h of the energy analysis' cycles on each side
    h_after: float = NAN
    # regime_periods (`regime_periods`): the potential of the larger swings and each side's period law in it
    shape_cutoff_yr: float = NAN  # sigma low-passed at separatrix_cutoff_cycles x the shorter side period [yr]
    shape_r2: float = NAN  # R^2 of that V within the larger side's cycles
    shape_wells: int = 0  # local minima of that V over one turn (2003 TX9: 2)
    law_before: float = NAN  # the worst T_obs / T_pred of each side's disjoint cycles (inf: a cycle the V cannot hold)
    law_after: float = NAN
    shape_failed: str = ''  # the sub-test of test 7 that failed ('name@cutoff factor'; `RegimePeriods.failed`)
    shape_tp_mismatch: float = NAN  # the worst |observed turning point - the loop's| / half-swing
    shape_drift_before: float = NAN  # spread of each side's cycle energies / the step of the energy
    shape_drift_after: float = NAN
    shape_period_ratio: float = NAN  # median T_obs smaller swings / larger swings (the isochrony null)
    shape_cycles_before: int = 0  # disjoint cycles of each side
    shape_cycles_after: int = 0
    shape_held_out: bool = False  # the larger side's laws out of sample
    shape_laws: str = ''  # worst laws per cutoff factor
    shape_vf_r: float = NAN  # V_E ~ V_F over the larger swings (reported)
    shape_vf_law_small: float = NAN
    key: str = ''  # the resonance key of the row whose analysis found the step (`analyse` binds the guard to it)
    # STEP_WITHHELD: the measurements the cut left unmeasured that failed on the whole record, and the cut's verdict
    withheld_checks: str = ''
    withheld_verdict: str = ''

    @property
    def cuts(self) -> bool:
        return not self.failed

    @property
    def time_yr(self) -> float:
        """The middle of the transitional half-cycle [yr]."""
        return 0.5 * (self.t_from_yr + self.t_to_yr)

    def jump(self) -> Jump:
        """The confirmed jump that cuts the record (`kick` False), its zone the transitional half-cycle."""
        return Jump(
            self.time_yr,
            NAN,
            self.t_to_yr - self.t_from_yr,
            self.dh,
            False,
            False,
            action_step=True,
            zone_start_yr=self.t_from_yr,
            zone_end_yr=self.t_to_yr,
        )


# the tests of G2a in their order (`ActionStep.failed`)
STEP_SEPARATION, STEP_RATIO, STEP_A_SWING, STEP_TRANSITION = 'separation', 'ratio', 'a_swing', 'transition'
STEP_ENERGY_CYCLES, STEP_ENERGY, STEP_REGIME_PERIODS = 'energy_cycles', 'energy', 'regime_periods'
# a step that passed every test, not cut: the cut would leave a measurement that failed on the whole record unmeasured
# and change the verdict (`analyse._action_step_guard`)
STEP_WITHHELD = 'withheld'


@dataclass(frozen=True)
class FloorWell:
    """A well of the caller's model inside which the angle's shortest libration is `FilteredAngle.period_floor_yr`;
    an episode centred outside it has the floor `outside_floor_yr` (MMR: a 1:1 angle analysed at its quasi-satellite
    floor keeps the tadpole floor for its tadpole and horseshoe episodes, so their forced waves stay rejected).
    Centre and half-width of the well per sample of `t_yr`, in the angle's own units [rad]; nan where there is none."""

    t_yr: np.ndarray
    centre_rad: np.ndarray
    halfwidth_rad: np.ndarray
    outside_floor_yr: float
    outside_setting: str
    min_share: float = 0.0  # the share of the episode's samples the well must exist on (MMR_COORBITAL_QS_WELL_SHARE)

    def holds(self, episode: 'Episode') -> bool:
        """The episode's mean centre lies within the well's (median) half-width of its (median) centre over the
        episode; False where the well exists on less than `min_share` of the episode's samples (or on none)."""
        inside = (self.t_yr >= episode.t_start_yr) & (self.t_yr <= episode.t_end_yr)
        centre, halfwidth = self.centre_rad[inside], self.halfwidth_rad[inside]
        finite = np.isfinite(centre) & np.isfinite(halfwidth)
        if not np.any(finite) or np.count_nonzero(finite) < self.min_share * len(finite):
            return False
        offset = np.mod(episode.centre_mod_2pi_rad - float(np.median(centre[finite])) + np.pi, TWO_PI) - np.pi
        return bool(abs(offset) < float(np.median(halfwidth[finite])))


@dataclass
class Geometry:
    """Everything geometric about one angle (MMR columns `geo_*`, `stall_*`, `exit_*`, `entry_*`, `jump_*`)."""

    window: Tuple[float, float]  # analysis window after edge trim and the last regime [yr]
    regime_start_yr: float
    turning_points: TurningPoints
    n_tp_cut_half: Optional[int]  # TPs with the cutoff x0.5 / x2 (L1, ripple sign); None = not measurable
    n_tp_cut_double: Optional[int]
    period_cycle_yr: float  # P_cyc: median full period over accepted episodes, else over all runs
    period_min_yr: float
    period_max_yr: float
    centre: CentreTrack
    net_rev_filtered: float
    net_rev_raw: float
    band_first_rad: float  # centre (mean midpoint) of the first full cycle
    band_last_rad: float
    bands_overlap: bool
    wiggle_half_amp_rad: float  # sqrt(2) std(sigma_f - c)
    wiggle_period_yr: float
    runs: List[Episode]  # every run, accepted or not
    slips: List[Slip] = field(default_factory=list)
    n_slips_rejected_short: int = 0
    revolutions: List[Revolution] = field(default_factory=list)
    pattern_b: Optional[PatternB] = None
    arc_stall: Optional[ArcStall] = None
    exit: Optional[EdgeEvent] = None
    entry: Optional[EdgeEvent] = None
    # after a cut (a confirmed jump of a or a G2a step): the entry before the record's first accepted run, which lies
    # before the jump (`entry` is the last regime's); None without a cut (`_mark_record_entry`)
    record_entry: Optional[EdgeEvent] = None
    jumps: List[Jump] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    band_first_half_rad: float = NAN  # mean half-amplitude of the first / last full cycle
    band_last_half_rad: float = NAN
    drift_fraction: float = NAN  # `drift_fraction` of the runs in the window
    sigmadot_negative_fraction: float = NAN  # time with sigma_dot against its mean sign
    rate_modulation: float = NAN  # amplitude of sigma_dot ~ m + r cos(sigma - phi), over |m|
    rate_slow_corr: float = NAN  # corr(sigma_dot_f, slow_f)
    jump_tau_yr: float = NAN
    jump_half_width: float = NAN  # half-width of the slow variable used by `jump_by_a_only`
    jump_edge_margin_yr: float = NAN  # data a jump needs on each side to be confirmed (`jump_edge_margin_yr`)
    # G2a: the best split of every accepted run tested for a step of its libration action (`action_steps`); those
    # that cut the record are jumps too (`Jump.action_step`)
    action_steps: List[ActionStep] = field(default_factory=list)
    # the shortest period an accepted episode may have in the caller's model (FilteredAngle.period_floor_yr); nan =
    # the caller's default. `period_floor_setting` names the setting it comes from (ledger)
    period_floor_yr: float = NAN
    period_floor_setting: str = ''
    floor_well: Optional[FloorWell] = None  # FilteredAngle.floor_well: episodes outside it keep its outside floor
    # the centres of the regimes before the last jump of the slow variable, each low-passed on the samples between its
    # own jumps only (`finish_geometry`, `_centres_before`): the centre turns of the runs before the last jump are
    # measured on them. Empty without such a cut. The last regime's own centre (`centre`) is low-passed on its samples
    # too and drawn on the record's grid with these (`_stitched_centre`)
    record_centres: List[CentreTrack] = field(default_factory=list)
    # MMR rules carried from the FilteredAngle (False for secular/ZLK): a 1:1 angle (the status reads its union of
    # regimes as a libration, `mmr.decide._g2`), and the travel inside an episode read on its own cycles
    # (`episode_centre_travel`)
    coorbital: bool = False
    cycle_travel: bool = False
    # an MMR angle with accepted runs (the edge rule or 1:1): the orbit-averaged raw angle and the record's times, for
    # `gap_raw_range` and `covering_union` (None otherwise; dropped with the other arrays once the verdict is frozen)
    averaged_raw: Optional[np.ndarray] = None
    averaged_raw_t_yr: Optional[np.ndarray] = None
    # MMR: accepted runs judged forced waves of another argument (`mmr.models.ForcedWave`, set by the MMR analysis
    # after their response is measured, `mmr.decide.forced_wave_episodes`); they stay in `runs` (episodes.csv) but are
    # no episodes of the angle. Empty for secular/ZLK
    forced_waves: tuple = ()
    # what a decision derives from this geometry for one settings object (`derived`: the MMR librating episodes, their
    # covering union, the near-episode count), computed once; emptied where the runs or their edges change
    # (`forget_derived`). Not a constructor argument: a `replace` starts with none
    derived_views: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def derived(self, key: tuple, settings, compute: Callable[[], object]):
        """`compute()` once per `key`, `settings` and list of runs (the same objects, the list of the same length): a
        later call with all of them returns that value; another settings object or runs list computes it again."""
        entry = self.derived_views.get(key)
        if entry is not None and entry[0] is settings and entry[1] is self.runs and entry[2] == len(self.runs):
            return entry[3]
        value = compute()
        self.derived_views[key] = (settings, self.runs, len(self.runs), value)
        return value

    def forget_derived(self) -> None:
        """Drop the derived views (`derived`): the runs, their edges or the centre changed."""
        self.derived_views.clear()

    @property
    def forced_wave_ids(self) -> frozenset:
        return frozenset(wave.episode_id for wave in self.forced_waves)

    @property
    def episodes(self) -> List[Episode]:
        """Accepted runs of the last regime, but for the forced waves (`forced_waves`)."""
        forced = self.forced_wave_ids
        return [run for run in self.runs if run.accepted and not run.before_last_jump and run.episode_id not in forced]

    def gap_raw_range(self, left: Episode, right: Episode) -> float:
        """The range (max - min) of the orbit-averaged raw angle from the last sample of `left` to the first of
        `right` (later) [rad], whatever lies between them; 0 when they overlap, nan without the averaged angle. Computed
        when asked: `covering_union` asks only for neighbouring librating episodes (Codex W20 A #4: a table of every
        pair cost O(E^2) memory)."""
        if self.averaged_raw is None or self.averaged_raw_t_yr is None:
            return NAN
        t = self.averaged_raw_t_yr
        start = int(np.searchsorted(t, left.t_end_yr, 'right')) - 1
        stop = int(np.searchsorted(t, right.t_start_yr)) + 1
        if stop <= start:
            return 0.0
        part = self.averaged_raw[start:stop]
        return float(np.max(part) - np.min(part))

    @property
    def episodes_before_jump(self) -> List[Episode]:
        """Accepted runs before the last jump of the slow variable (outside the last regime), but for the forced waves."""
        forced = self.forced_wave_ids
        return [run for run in self.runs if run.accepted and run.before_last_jump and run.episode_id not in forced]

    def centre_covering(self, start_yr: float) -> CentreTrack:
        """The centre track measured on a window that holds `start_yr`: the last regime's, or for a time before it the
        one of the regime before the last jump that holds it (`record_centres`)."""
        if start_yr < self.window[0]:
            for track in reversed(self.record_centres):
                if len(track.t_yr) and start_yr >= track.t_yr[0]:
                    return track
        return self.centre

    def to_columns(self) -> dict:
        """CSV columns (`geo_*`, `stall_*`, `exit_*`, `entry_*`, `jump_*`, `regime_start_yr`) in CSV units."""
        deg = np.degrees
        centre = self.centre
        columns = {
            'geo_window_t0_yr': self.window[0],
            'geo_window_t1_yr': self.window[1],
            'geo_n_tp': len(self.turning_points),
            'geo_tp_prominence_deg': float(deg(self.turning_points.prominence_rad)),
            'geo_n_tp_cut_half': self.n_tp_cut_half,
            'geo_n_tp_cut_double': self.n_tp_cut_double,
            'geo_period_cycle_yr': self.period_cycle_yr,
            'geo_period_min_yr': self.period_min_yr,
            'geo_period_max_yr': self.period_max_yr,
            'geo_n_runs': len(self.runs),
            'geo_n_episodes': len(self.episodes),
            'geo_drift_fraction': self.drift_fraction,
            'geo_centre_window_yr': centre.window_yr,
            'geo_centre_max_pass_rad': centre.max_pass_rad,
            'geo_centre_max_pass_rev': centre.max_pass_rev,
            'geo_centre_max_pass_t0_yr': centre.max_pass_t0_yr,
            'geo_centre_max_pass_t1_yr': centre.max_pass_t1_yr,
            'geo_centre_max_pass_rev_lo': centre.max_pass_rev_lo,
            'geo_centre_max_pass_rev_hi': centre.max_pass_rev_hi,
            'geo_centre_net_rev': centre.net_rev,
            'geo_centre_range_rad': centre.range_rad,
            'geo_centre_n_tp': len(centre.turns),
            'geo_centre_tp_times_yr': join_list(f'{t:.6g}' for t in centre.turns.times_yr),
            'geo_net_rev_filtered': self.net_rev_filtered,
            'geo_net_rev_raw': self.net_rev_raw,
            'geo_band_first_deg': float(deg(self.band_first_rad)),
            'geo_band_first_half_deg': float(deg(self.band_first_half_rad)),
            'geo_band_last_deg': float(deg(self.band_last_rad)),
            'geo_band_last_half_deg': float(deg(self.band_last_half_rad)),
            'geo_bands_overlap': self.bands_overlap,
            'geo_wiggle_half_amp_deg': float(deg(self.wiggle_half_amp_rad)),
            'geo_wiggle_period_yr': self.wiggle_period_yr,
            'geo_sigmadot_negative_fraction': self.sigmadot_negative_fraction,
            'geo_rate_modulation': self.rate_modulation,
            'geo_rate_a_corr': self.rate_slow_corr,
            'geo_n_slips': sum(1 for slip in self.slips if not slip.suspect),
            'geo_n_slips_suspect': sum(1 for slip in self.slips if slip.suspect),
            'geo_slip_times_yr': join_list(f'{slip.time_yr:.6g}' for slip in self.slips),
            'geo_slip_sizes_rev': join_list(f'{slip.size_rev:.4g}' for slip in self.slips),
            'geo_slip_durations_yr': join_list(f'{slip.duration_yr:.6g}' for slip in self.slips),
            'geo_slip_suspect': join_list(int(slip.suspect) for slip in self.slips),
            'geo_n_slips_rejected_short': self.n_slips_rejected_short,
            'geo_n_revolutions': len(self.revolutions),
        }
        columns.update(_pattern_b_columns(self.pattern_b))
        columns.update(_arc_stall_columns(self.arc_stall))
        for kind, event in (('exit', self.exit), ('entry', self.entry), ('record_entry', self.record_entry)):
            columns.update(_edge_columns(kind, event))
        columns.update(
            {
                'jump_times_yr': join_list(f'{jump.time_yr:.6g}' for jump in self.jumps),
                'jump_sizes_au': join_list(f'{jump.size:.4g}' for jump in self.jumps),
                'jump_durations_yr': join_list(f'{jump.duration_yr:.6g}' for jump in self.jumps),
                'jump_dh': join_list(f'{jump.dh:.4g}' for jump in self.jumps),
                'jump_kick': join_list(int(jump.kick) for jump in self.jumps),
                'jump_by_a_only': join_list(int(jump.by_slow_only) for jump in self.jumps),
                'jump_at_edge': join_list(int(jump.at_edge) for jump in self.jumps),
                'jump_edge_margin_yr': self.jump_edge_margin_yr,
                'jump_tau_yr': self.jump_tau_yr,
                'jump_halfwidth_ref_au': self.jump_half_width,
                'jump_action_step': join_list(int(jump.action_step) for jump in self.jumps),
                'regime_start_yr': self.regime_start_yr,
            }
        )
        columns.update(_action_step_columns(self.action_steps))
        return columns

    def entry_suspected(self) -> bool:
        """An entry suspected before the first accepted run of the last regime or, after a cut, of the record."""
        return any(event is not None and event.suspected for event in (self.entry, self.record_entry))

    def last_cut_zone_yr(self) -> Tuple[float, float]:
        """The zone (`Jump.zone`) of the last jump that cuts the record [yr]; (nan, nan) without one."""
        cuts = [jump for jump in self.jumps if not jump.kick]
        if not cuts:
            return NAN, NAN
        return max(cuts, key=lambda jump: jump.zone(self.jump_tau_yr)[1]).zone(self.jump_tau_yr)


def run_of(geometry: Optional[Geometry], episode_id: Optional[int]) -> Optional[Episode]:
    """The run (accepted or not) with this `episode_id`; None for a None id, a None geometry or no such run. A module
    function, not a method: the decision code also reads stand-ins that carry `runs` only."""
    if episode_id is None:
        return None
    return next((run for run in getattr(geometry, 'runs', ()) if run.episode_id == episode_id), None)


def _pattern_b_columns(pattern: Optional[PatternB]) -> dict:
    if pattern is None:
        pattern = PatternB(0, NAN, NAN, NAN, NAN, NAN, NAN, NAN, False, False, False)
    return {
        'geo_stall_fraction': pattern.stall_fraction,
        'geo_stall_phase_mean_deg': float(np.degrees(pattern.stall_phase_mean_rad)) % 360.0,
        'geo_stall_phase_R': pattern.stall_phase_r,
        'geo_rate_min_over_mean_med': pattern.rate_min_over_mean_med,
        'geo_rate_min_over_mean_pend_med': pattern.rate_min_over_mean_pend_med,
        'geo_h_rev_med': pattern.h_rev_med,
        'geo_h_rev_max': pattern.h_rev_max,
        'geo_pattern_b_in_zone': pattern.in_zone,
        'geo_pattern_b_detected': pattern.detected,
        'geo_stall_rate_inconsistent': pattern.rate_inconsistent,
    }


def _arc_stall_columns(arc: Optional[ArcStall]) -> dict:
    if arc is None:
        return {
            'stall_kind': '',
            'geo_arc_range_rad': NAN,
            'stall_tau_yr': NAN,
            'stall_rate_rad_kyr': NAN,
            'stall_acc_rad_kyr2': NAN,
            'stall_wrapped_min_deg': NAN,
            'stall_wrapped_max_deg': NAN,
            'stall_a_ptp_au': NAN,
            'stall_a_trend_au_yr': NAN,
            'stall_K_bound_range_au_yr': NAN,
            'stall_K_bound_trend_au_yr': NAN,
            'stall_K_bound_curvature_au_yr': NAN,
            'stall_K_bound_adotmax_au_yr': NAN,
            'stall_P_lib_min_kyr': NAN,
        }
    kyr = 1e3
    return {
        'stall_kind': arc.kind,
        'geo_arc_range_rad': arc.range_rad,
        'stall_tau_yr': arc.tau_yr,
        'stall_rate_rad_kyr': arc.rate_rad_yr * kyr,
        'stall_acc_rad_kyr2': arc.acc_rad_yr2 * kyr * kyr,
        'stall_wrapped_min_deg': float(np.degrees(arc.wrapped_min_rad)),
        'stall_wrapped_max_deg': float(np.degrees(arc.wrapped_max_rad)),
        'stall_a_ptp_au': arc.slow_ptp,
        'stall_a_trend_au_yr': arc.slow_trend_per_yr,
        'stall_K_bound_range_au_yr': arc.force_bound_range,
        'stall_K_bound_trend_au_yr': arc.force_bound_trend,
        'stall_K_bound_curvature_au_yr': arc.force_bound_curvature,
        'stall_K_bound_adotmax_au_yr': arc.force_bound_rate,
        'stall_P_lib_min_kyr': arc.period_min_yr / kyr,
    }


def reported_action_step(steps: Sequence[ActionStep]) -> Optional[ActionStep]:
    """The step the columns and the figure report: of those that cut, the best separated; else of those withheld
    (`STEP_WITHHELD`); else the best separated evaluated one (None without any)."""
    if not steps:
        return None
    cutting = [step for step in steps if step.cuts] or [step for step in steps if step.failed == STEP_WITHHELD]
    return max(cutting or list(steps), key=lambda step: step.separation)


def _action_step_columns(steps: Sequence[ActionStep]) -> dict:
    """geo_action_step_* (G2a): how many runs it cut and the numbers of the reported step (`reported_action_step`)."""
    step = reported_action_step(steps)
    deg = np.degrees
    return {
        'geo_action_step_n_cuts': sum(1 for s in steps if s.cuts),
        'geo_action_step_n_tested': len(steps),
        'geo_action_step_time_yr': step.time_yr if step else NAN,
        'geo_action_step_t_from_yr': step.t_from_yr if step else NAN,
        'geo_action_step_t_to_yr': step.t_to_yr if step else NAN,
        'geo_action_step_run_t0_yr': step.run_t_start_yr if step else NAN,
        'geo_action_step_run_t1_yr': step.run_t_end_yr if step else NAN,
        'geo_action_step_n_before': step.n_before if step else NAN,
        'geo_action_step_n_after': step.n_after if step else NAN,
        'geo_action_step_amp_before_deg': float(deg(step.amp_before_rad)) if step else NAN,
        'geo_action_step_amp_after_deg': float(deg(step.amp_after_rad)) if step else NAN,
        'geo_action_step_separation': step.separation if step else NAN,
        'geo_action_step_ratio': step.ratio if step else NAN,
        'geo_action_step_a_ratio': step.a_ratio if step else NAN,
        'geo_action_step_transition_P': step.transition_periods if step else NAN,
        'geo_action_step_cycles_before': step.cycles_before if step else NAN,
        'geo_action_step_cycles_after': step.cycles_after if step else NAN,
        'geo_action_step_dh': step.dh if step else NAN,
        'geo_action_step_dh_threshold': step.dh_threshold if step else NAN,
        'geo_action_step_h_before': step.h_before if step else NAN,
        'geo_action_step_h_after': step.h_after if step else NAN,
        'geo_action_step_shape_cutoff_yr': step.shape_cutoff_yr if step else NAN,
        'geo_action_step_shape_R2': step.shape_r2 if step else NAN,
        'geo_action_step_shape_wells': step.shape_wells if step else NAN,
        'geo_action_step_law_before': step.law_before if step else NAN,
        'geo_action_step_law_after': step.law_after if step else NAN,
        'geo_action_step_shape_failed': step.shape_failed if step else '',
        'geo_action_step_tp_mismatch': step.shape_tp_mismatch if step else NAN,
        'geo_action_step_drift_before': step.shape_drift_before if step else NAN,
        'geo_action_step_drift_after': step.shape_drift_after if step else NAN,
        'geo_action_step_period_ratio': step.shape_period_ratio if step else NAN,
        'geo_action_step_shape_cycles_before': step.shape_cycles_before if step else NAN,
        'geo_action_step_shape_cycles_after': step.shape_cycles_after if step else NAN,
        'geo_action_step_held_out': int(step.shape_held_out) if step else NAN,
        'geo_action_step_laws': step.shape_laws if step else '',
        'geo_action_step_vf_r': step.shape_vf_r if step else NAN,
        'geo_action_step_vf_law_small': step.shape_vf_law_small if step else NAN,
        'geo_action_step_provisional': int(step.provisional) if step else NAN,
        'geo_action_step_failed': step.failed if step else '',
        'geo_action_step_withheld_checks': step.withheld_checks if step else '',
        'geo_action_step_withheld_verdict': step.withheld_verdict if step else '',
    }


def _edge_columns(kind: str, event: Optional[EdgeEvent]) -> dict:
    if event is None:
        return {
            f'{kind}_time_yr': NAN,
            f'{kind}_beyond_deg': NAN,
            f'{kind}_remaining_P': NAN,
            f'{kind}_suspected': False,
            f'{kind}_raw_monotonic': None,
            f'{kind}_raw_arc_rev': NAN,
            f'{kind}_observed': False,
        }
    return {
        f'{kind}_time_yr': event.time_yr,
        f'{kind}_beyond_deg': float(np.degrees(event.beyond_rad)),
        f'{kind}_remaining_P': event.remaining_periods,
        f'{kind}_suspected': event.suspected,
        f'{kind}_raw_monotonic': event.raw_monotonic,
        f'{kind}_raw_arc_rev': float(event.raw_arc_rad / TWO_PI),
        f'{kind}_observed': event.observed,
    }


# --- turning points ---------------------------------------------------------------------------


def turning_prominence(floor_rad: float, ripple_factor: float, unremoved_ripple_rad: float) -> float:
    """p_TP = max(floor, factor * r_unremoved) [rad]: ripple of half-amplitude r makes extrema of prominence <= 2r."""
    ripple = unremoved_ripple_rad if np.isfinite(unremoved_ripple_rad) else 0.0
    return float(max(floor_rad, ripple_factor * ripple))


def threshold_at(times_yr: np.ndarray, prominence_rad: float, local: Sequence[LocalThreshold] = ()) -> np.ndarray:
    """The TP prominence threshold at `times_yr` [rad]: `prominence_rad`, or the threshold of the `local` stretch that
    holds the time (the stretches do not overlap: runs of one sequence of turning points)."""
    times = np.asarray(times_yr, dtype=float)
    out = np.full(times.shape, float(prominence_rad))
    for stretch in local:
        out[(times >= stretch.start_yr) & (times <= stretch.end_yr)] = stretch.prominence_rad
    return out


def _empty_turning_points(prominence_rad: float, local: Sequence[LocalThreshold] = ()) -> TurningPoints:
    return TurningPoints(np.zeros(0, dtype=int), np.zeros(0), np.zeros(0), np.zeros(0, dtype=int), float(prominence_rad), tuple(local))


def _alternate(indices: np.ndarray, kinds: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Of two consecutive extrema of the same kind keep the more extreme one."""
    keep_idx: List[int] = []
    keep_kind: List[int] = []
    for idx, kind in zip(indices, kinds):
        if keep_kind and keep_kind[-1] == kind:
            if kind * values[idx] > kind * values[keep_idx[-1]]:
                keep_idx[-1] = int(idx)
            continue
        keep_idx.append(int(idx))
        keep_kind.append(int(kind))
    return np.asarray(keep_idx, dtype=int), np.asarray(keep_kind, dtype=int)


def _base_minima(y: List[float]) -> np.ndarray:
    """For each i: the minimum of y between i and the nearest strictly higher point on its left (the whole left
    side when there is none) - the left base of `scipy.signal.peak_prominences`. +inf when y[i - 1] > y[i].
    One pass with a stack of decreasing values, each entry holding the minimum of what lies between it and the
    entry below it: O(n)."""
    out = np.empty(len(y))
    stack: List[int] = []
    gap_minima: List[float] = []
    for i, value in enumerate(y):
        lowest = math.inf
        while stack and y[stack[-1]] <= value:
            lowest = min(lowest, y[stack.pop()], gap_minima.pop())
        out[i] = lowest
        stack.append(i)
        gap_minima.append(lowest)
    return out


@dataclass(frozen=True)
class Extrema:
    """Every local maximum and minimum of a series with its prominence (the peak over the higher of its two bases)."""

    maxima: np.ndarray  # int, into the series
    maxima_prominence: np.ndarray
    minima: np.ndarray
    minima_prominence: np.ndarray

    def above(self, prominence: float) -> Tuple[np.ndarray, np.ndarray]:
        """(maxima, minima) with prominence >= `prominence`."""
        return self.maxima[self.maxima_prominence >= prominence], self.minima[self.minima_prominence >= prominence]

    def above_at(self, t_yr: np.ndarray, prominence: float, local: Sequence[LocalThreshold]) -> Tuple[np.ndarray, np.ndarray]:
        """(maxima, minima) with prominence >= the threshold at their time (`threshold_at`; `t_yr` the series' times)."""
        if not local:
            return self.above(prominence)
        t = np.asarray(t_yr, dtype=float)
        maxima = self.maxima_prominence >= threshold_at(t[self.maxima], prominence, local)
        minima = self.minima_prominence >= threshold_at(t[self.minima], prominence, local)
        return self.maxima[maxima], self.minima[minima]


def extrema(x: np.ndarray) -> Extrema:
    """The local extrema of x and their prominences, both kinds from one skeleton of local extrema and ends: measured
    once, they serve every prominence threshold (the regimes between jumps, `finish_geometry`)."""
    x = np.asarray(x, dtype=float)
    maxima, _ = signal.find_peaks(x)
    minima, _ = signal.find_peaks(-x)
    skeleton = np.unique(np.concatenate([[0, len(x) - 1], maxima, minima]))
    y = x[skeleton]

    def prominences(peaks: np.ndarray, heights: np.ndarray) -> np.ndarray:
        if not len(peaks):
            return np.zeros(0)
        at = np.searchsorted(skeleton, peaks)  # every extremum is in the skeleton
        values = heights.tolist()
        left = _base_minima(values)[at]
        right = _base_minima(values[::-1])[::-1][at]
        return heights[at] - np.maximum(left, right)

    return Extrema(maxima, prominences(maxima, y), minima, prominences(minima, -y))


def _prominent_extrema(x: np.ndarray, prominence: float) -> Tuple[np.ndarray, np.ndarray]:
    """(maxima, minima) of x with prominence >= `prominence`."""
    return extrema(x).above(prominence)


def prominent_peaks(x: np.ndarray, prominence: float) -> np.ndarray:
    """Indices of the maxima of x with prominence >= `prominence`: the same set as
    `scipy.signal.find_peaks(x, prominence=prominence)` (prominence = the peak over the higher of its two bases).

    A base is a minimum between the peak and the nearest higher point, and every such minimum is a local minimum
    or an end of the series, so the prominences are measured on the skeleton of local extrema and ends (exact).
    scipy searches each side of every peak sample by sample: on 10^6 points with 4 x 10^4 ripple maxima on a
    trend that took 5-11 s per call; the skeleton and `_base_minima` take 0.05-0.1 s."""
    return _prominent_extrema(np.asarray(x, dtype=float), prominence)[0]


def find_turning_points(
    t_yr: np.ndarray,
    angle_rad: np.ndarray,
    prominence_rad: float,
    window: Tuple[float, float],
    found: Optional[Extrema] = None,
    local: Sequence[LocalThreshold] = (),
) -> TurningPoints:
    """Alternating maxima/minima of `angle_rad` inside `window` with prominence >= `prominence_rad` (inside a stretch
    of `local`: >= that stretch's own prominence).

    `_prominent_extrema` (the extrema of `scipy.signal.find_peaks` with that prominence, `prominent_peaks`; `found`:
    `extrema(angle_rad)` measured already); of two consecutive extrema of the same kind the more extreme is kept. Empty
    result has zero-length arrays.
    """
    t = np.asarray(t_yr, dtype=float)
    x = np.asarray(angle_rad, dtype=float)
    local = tuple(local)
    inside = window_mask(t, window)
    if len(x) < 3 or not inside.any() or not np.all(np.isfinite(x)):
        return _empty_turning_points(prominence_rad, local)
    # prominences are measured on the whole series: the window only selects which extrema count
    maxima, minima = (found if found is not None else extrema(x)).above_at(t, prominence_rad, local)
    indices = np.concatenate([maxima, minima])
    kinds = np.concatenate([np.ones(len(maxima), dtype=int), -np.ones(len(minima), dtype=int)])
    order = np.argsort(indices, kind='stable')
    indices, kinds = indices[order], kinds[order]
    selected = inside[indices]
    indices, kinds = _alternate(indices[selected], kinds[selected], x)
    if len(indices) == 0:
        return _empty_turning_points(prominence_rad, local)
    return TurningPoints(indices, t[indices], x[indices], kinds, float(prominence_rad), local)


def full_periods(tp: TurningPoints) -> np.ndarray:
    """Full periods t(TP_i+2) - t(TP_i) [yr] (max->max and min->min)."""
    times = np.asarray(tp.times_yr, dtype=float)
    return times[2:] - times[:-2] if len(times) >= 3 else np.zeros(0)


def half_amplitude_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.abs(np.diff(values)) / 2 if len(values) >= 2 else np.zeros(0)


def midpoint_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values[1:] + values[:-1]) / 2 if len(values) >= 2 else np.zeros(0)


def half_amplitudes(tp: TurningPoints) -> np.ndarray:
    """A_i = |sigma(TP_i) - sigma(TP_i+1)| / 2 [rad]."""
    return half_amplitude_values(tp.values_rad)


def midpoints(tp: TurningPoints) -> np.ndarray:
    """m_i = (sigma(TP_i) + sigma(TP_i+1)) / 2 [rad], one per half-cycle."""
    return midpoint_values(tp.values_rad)


# --- centre -----------------------------------------------------------------------------------


def centre_prominence(tp_prominence_rad: float, settings: AnalysisSettings) -> float:
    """p_c [rad]: the prominence of a turn of the centre, max(libration_centre_tp_prominence_deg,
    libration_centre_tp_factor x p_TP)."""
    return max(np.radians(settings.libration_centre_tp_prominence_deg), settings.libration_centre_tp_factor * tp_prominence_rad)


def monotone_passes(t_yr: np.ndarray, centre_rad: np.ndarray, turns: TurningPoints, window: Tuple[float, float]) -> List[MonotonePass]:
    """The centre between consecutive turns (and window edges); delta = c(end) - c(start) [rad].

    Measured by max/min (AUTHOR-FEEDBACK 2): on each stretch delta = +-(max - min), positive when
    the minimum comes first; t_start/t_end are the times of the two extremes.
    """
    t = np.asarray(t_yr, dtype=float)
    c = np.asarray(centre_rad, dtype=float)
    inside = np.flatnonzero(window_mask(t, window))
    if len(inside) < 2:
        return []
    nodes = [int(inside[0])] + [int(i) for i in turns.indices if inside[0] < i < inside[-1]] + [int(inside[-1])]
    passes = []
    for start, end in zip(nodes[:-1], nodes[1:]):
        piece = c[start : end + 1]
        i_min, i_max = int(np.argmin(piece)), int(np.argmax(piece))
        if i_min == i_max:
            passes.append(MonotonePass(float(t[start]), float(t[end]), 0.0))
            continue
        first, last = sorted((i_min, i_max))
        delta = float(piece[last] - piece[first])
        passes.append(MonotonePass(float(t[start + first]), float(t[start + last]), delta))
    return passes


def _centre_arrays(geometry: 'Geometry', start_yr: float) -> Optional[Tuple['CentreTrack', np.ndarray, np.ndarray]]:
    """(centre track, its times [yr], its centre [rad]) for an interval from `start_yr` (`Geometry.centre_covering`; an
    object with only `centre` reads that); None without a track or samples."""
    covering = getattr(geometry, 'centre_covering', None)
    centre = covering(start_yr) if covering is not None else geometry.centre
    if centre is None or not len(centre.t_yr):
        return None
    return centre, np.asarray(centre.t_yr, dtype=float), np.asarray(centre.centre_rad, dtype=float)


def centre_travel_in(geometry: 'Geometry', start_yr: float, end_yr: float) -> float:
    """Largest monotone travel of the centre inside [start, end] [rad]: each monotone pass of the centre (split at its
    turns) clipped to the interval; 0 without passes. An interval before the last regime reads the centre of the record
    before the last jump (`Geometry.centre_covering`; an object with only `centre` reads that)."""
    arrays = _centre_arrays(geometry, start_yr)
    if arrays is None:
        return 0.0
    centre, t, c = arrays
    travel = 0.0
    for single in centre.passes:
        lo, hi = max(single.t_start_yr, start_yr), min(single.t_end_yr, end_yr)
        if hi > lo:
            travel = max(travel, abs(float(np.interp(hi, t, c) - np.interp(lo, t, c))))
    return travel


def largest_monotone_pass(passes: Sequence[MonotonePass]) -> Optional[MonotonePass]:
    """The pass with the largest |delta_rad| (None for no passes). A pass >= 2pi makes a circulation."""
    if not passes:
        return None
    return max(passes, key=lambda item: abs(item.delta_rad))


def _max_pass_at(t, angle_f, cutoff_yr, prominence_c, window, settings) -> float:
    centre = lowpass_or_none(t, angle_f, cutoff_yr, settings)
    if centre is None:
        return NAN
    turns = find_turning_points(t, centre, prominence_c, window)
    best = largest_monotone_pass(monotone_passes(t, centre, turns, window))
    return abs(best.delta_rad) / TWO_PI if best is not None else NAN


def centre_track(
    t_yr: np.ndarray,
    angle_f: np.ndarray,
    period_cycle_yr: float,
    fallback_cutoff_yr: float,
    tp_prominence_rad: float,
    window: Tuple[float, float],
    settings: AnalysisSettings,
) -> CentreTrack:
    """The centre: low-pass of sigma_f with W_c = libration_centre_cycles * P_cyc (3 P_c without cycles) [yr].

    Centre turns: prominence p_c = max(libration_centre_tp_prominence_deg, libration_centre_tp_factor * p_TP);
    passes between consecutive turns and window edges; the largest |pass| also at W_c x0.5 and x2
    (`max_pass_rev_lo/hi`). Uses `analysis.filtering.lowpass` with `settings.filter_order`.
    """
    t = np.asarray(t_yr, dtype=float)
    angle = np.asarray(angle_f, dtype=float)
    base = period_cycle_yr if np.isfinite(period_cycle_yr) and period_cycle_yr > 0 else fallback_cutoff_yr
    window_yr = float(settings.libration_centre_cycles * base)
    prominence_c = centre_prominence(tp_prominence_rad, settings)
    centre = lowpass_or_none(t, angle, window_yr, settings)
    if centre is None:
        centre = np.full_like(angle, NAN)
    turns = find_turning_points(t, centre, prominence_c, window)
    passes = monotone_passes(t, centre, turns, window)
    best = largest_monotone_pass(passes)
    inside = window_mask(t, window) & np.isfinite(centre)
    if inside.sum() >= 2:
        c_in = centre[inside]
        net_rev = float((c_in[-1] - c_in[0]) / TWO_PI)
        range_rad = float(np.ptp(c_in))
    else:
        net_rev = range_rad = NAN
    by_smoothing = [abs(best.delta_rad) / TWO_PI if best is not None else NAN]
    for factor in (0.5, 2):
        by_smoothing.append(_max_pass_at(t, angle, factor * window_yr, prominence_c, window, settings))
    finite = [value for value in by_smoothing if np.isfinite(value)]
    return CentreTrack(
        t_yr=t,
        centre_rad=centre,
        window_yr=window_yr,
        turns=turns,
        passes=passes,
        max_pass_rad=abs(best.delta_rad) if best is not None else NAN,
        max_pass_t0_yr=best.t_start_yr if best is not None else NAN,
        max_pass_t1_yr=best.t_end_yr if best is not None else NAN,
        net_rev=net_rev,
        range_rad=range_rad,
        max_pass_rev_lo=min(finite) if finite else NAN,
        max_pass_rev_hi=max(finite) if finite else NAN,
    )


# --- runs and episodes ------------------------------------------------------------------------


def _sorted_median(values: List[float]) -> float:
    """The median of an ascending list (as `np.median`)."""
    k = len(values) // 2
    return float(values[k]) if len(values) % 2 else float((values[k - 1] + values[k]) / 2)


def split_runs(tp: TurningPoints, settings: AnalysisSettings) -> List[np.ndarray]:
    """Index arrays (into `tp`) of runs: a run breaks where a gap exceeds libration_episode_gap_factor x
    the run's median half-period, where a half-amplitude reaches pi (over the top), or where
    consecutive midpoints differ by more than pi (a slip).

    The half-cycle that breaks the run belongs to neither side: the old run ends at TP_i, the new
    one starts at TP_i+1.
    """
    n = len(tp)
    if n == 0:
        return []
    times = np.asarray(tp.times_yr, dtype=float)
    values = np.asarray(tp.values_rad, dtype=float)
    runs: List[List[int]] = []
    current = [0]
    half_periods: List[float] = []  # of the current run, kept sorted: the median at each step without re-sorting
    previous_mid: Optional[float] = None
    for i in range(n - 1):
        half_period = times[i + 1] - times[i]
        amplitude = abs(values[i + 1] - values[i]) / 2
        mid = (values[i + 1] + values[i]) / 2
        broken = amplitude >= np.pi
        if not broken and half_periods:
            broken = half_period > settings.libration_episode_gap_factor * _sorted_median(half_periods)
        if not broken and previous_mid is not None:
            broken = abs(mid - previous_mid) > np.pi
        if broken:
            runs.append(current)
            current, half_periods, previous_mid = [i + 1], [], None
        else:
            current.append(i + 1)
            bisect.insort(half_periods, half_period)
            previous_mid = mid
    runs.append(current)
    return [np.asarray(run, dtype=int) for run in runs]


def local_thresholds(
    t_yr: np.ndarray,
    angle_rad: np.ndarray,
    window: Tuple[float, float],
    local_prominence: LocalProminence,
    settings: AnalysisSettings,
    found: Optional[Extrema] = None,
) -> Tuple[LocalThreshold, ...]:
    """The stretches whose turning points are selected at their own prominence, and those prominences.

    The record's prominence covers what sigma_f holds faster than any libration over the whole record, and a regime
    change elsewhere inflates it beyond the swings of a quiet libration (2001 PG14 33S-2+0-31: 21.0 deg from an escape
    at 3-4 kyr and a drop of a at 10.3 kyr, 2.7 deg on the libration at 0.3-2.8 kyr; its half-cycles of 13-24 deg fell
    below it, the gap rule then cut the run at 700 yr). So, once and without iterating: the provisional turning points
    at the type's floor (`local_prominence.floor_rad`) are split into runs by the usual rules (`split_runs`), and every
    run of >= libration_min_turning_points of them gets the prominence measured on its own samples
    (`local_prominence`); a run too short to measure it keeps the record's (no stretch)."""
    t = np.asarray(t_yr, dtype=float)
    provisional = find_turning_points(t, angle_rad, local_prominence.floor_rad, window, found)
    stretches = []
    for run in split_runs(provisional, settings):
        if len(run) < settings.libration_min_turning_points:
            continue
        start, end = float(provisional.times_yr[run[0]]), float(provisional.times_yr[run[-1]])
        prominence = local_prominence((t >= start) & (t <= end))
        if np.isfinite(prominence):
            stretches.append(LocalThreshold(start, end, float(prominence)))
    return tuple(stretches)


def libration_image(
    t_yr: np.ndarray,
    angle_f: np.ndarray,
    window: Tuple[float, float],
    cutoff_yr: float,
    period_floor_yr: float,
    floor_prominence_rad: float,
    settings: AnalysisSettings,
) -> np.ndarray:
    """lowpass(x, P_c) - lowpass(x, P_floor) on the samples of every libration run, 0 elsewhere [rad, signed]: what
    the run's own oscillation x leaves in `fast_part`.

    The runs are those `geometry.local_thresholds` selects at its own prominence: the turning points of sigma_f at the
    floor mmr_tp_prominence_floor_deg split by `geometry.split_runs`, of >= libration_min_turning_points TPs. A run is
    a libration when the median of its full periods t_i+2 - t_i is not shorter than P_floor, the test
    `mmr.decide.librating_episodes` applies to an episode; a run without a full period (two TPs) is none. A run it would
    reject as a forced wave keeps its image (a 115-yr wave: the threshold stays above its swing). The mean full period
    would not do: a slow swing among the turns of a 121-yr wave stretches it (52356 5J-2S-2, 60-67 kyr: median 122 yr,
    mean 155 yr). The TP times here are the vertices of the parabola through the three samples about each extremum
    (`_vertex_times`): on the grid a full period is a whole number of steps, and a 150-yr libration sampled at 5.1 yr
    measured 147.9 yr. Only this test reads them; the episode's period test still reads the sample times.

    x passes through the run's turning points, a half-cosine between neighbours, and holds its end values beyond
    them; its image is formed on the run and filter_edge_trim_cutoffs x P_floor on each side (where a stretch's
    residual is not read: `regime_prominence`). For a steady sine x = G(P; P_c) of it, so (G(P; P_c) - G(P; P_floor))
    (1 - G(P; P_c)) of the swing stays (0.026 at P_floor); a changing amplitude, centre or shape leaves more. A wave
    that rides on the run without turns of its own is partly taken out with it (x passes through the turns it shifts:
    about 70 % of a 6-deg 115-yr wave on a 25-deg 160-yr libration)."""
    t = np.asarray(t_yr, float)
    image = np.zeros(len(t))
    angle_f = np.asarray(angle_f, float)
    turning = find_turning_points(t, angle_f, floor_prominence_rad, tuple(window))
    if len(turning) < settings.libration_min_turning_points:
        return image
    step = float(np.median(np.diff(t)))
    margin = int(math.ceil(settings.filter_edge_trim_cutoffs * period_floor_yr / step))
    vertices = _vertex_times(t, angle_f, np.asarray(turning.indices, int))
    for run in split_runs(turning, settings):
        times = vertices[run]
        periods = times[2:] - times[:-2]
        if len(run) < settings.libration_min_turning_points or not len(periods) or float(np.median(periods)) < period_floor_yr:
            continue
        index = np.asarray(turning.indices, int)[run]
        lo, hi = max(int(index[0]) - margin, 0), min(int(index[-1]) + margin, len(t) - 1)
        span = t[lo : hi + 1]
        x = _through_turning_points(span, times, np.asarray(turning.values_rad, float)[run])
        fast = lowpass_or_none(span, x, cutoff_yr, settings)
        slow = lowpass_or_none(span, x, period_floor_yr, settings)
        if fast is None or slow is None:
            continue
        own = slice(int(index[0]), int(index[-1]) + 1)
        image[own] = (fast - slow)[own.start - lo : own.stop - lo]
    return image


def _vertex_times(t: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """The times of the extrema y[indices] refined to the vertex of the parabola through the samples i - 1, i, i + 1
    (uniform grid); the sample's own time at the record's ends or where the three samples are collinear."""
    times = np.asarray(t, float)[indices]
    inner = (indices > 0) & (indices < len(y) - 1)
    i = indices[inner]
    curvature = y[i - 1] - 2 * y[i] + y[i + 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        shift = np.where(curvature != 0, 0.5 * (y[i - 1] - y[i + 1]) / curvature, 0.0)
    times[inner] += np.clip(shift, -0.5, 0.5) * (t[i + 1] - t[i - 1]) / 2
    return times


def _through_turning_points(t: np.ndarray, times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The curve through the turning points (times, values): a half-cosine between neighbours (its slope is 0 at both),
    the end values before the first and after the last."""
    x = np.interp(t, times, values)
    k = np.clip(np.searchsorted(times, t, side='right') - 1, 0, len(times) - 2)
    t0, t1, v0, v1 = times[k], times[k + 1], values[k], values[k + 1]
    inside = (t > times[0]) & (t < times[-1]) & (t1 > t0)
    phase = np.pi * (t - t0) / np.where(t1 > t0, t1 - t0, 1.0)
    x[inside] = ((v0 + v1) / 2 + (v0 - v1) / 2 * np.cos(phase))[inside]
    return x


def turning_points_with_local(
    t_yr: np.ndarray,
    angle_rad: np.ndarray,
    prominence_rad: float,
    window: Tuple[float, float],
    settings: AnalysisSettings,
    local_prominence: Optional[LocalProminence] = None,
    found: Optional[Extrema] = None,
) -> TurningPoints:
    """`find_turning_points` at `prominence_rad`, each run of provisional turning points at its own prominence
    (`local_thresholds`) when `local_prominence` is given."""
    if local_prominence is None:
        return find_turning_points(t_yr, angle_rad, prominence_rad, window, found)
    x = np.asarray(angle_rad, dtype=float)
    if found is None and len(x) >= 3 and np.all(np.isfinite(x)):
        found = extrema(x)
    local = local_thresholds(t_yr, x, window, local_prominence, settings, found)
    return find_turning_points(t_yr, x, prominence_rad, window, found, local)


def _over_the_top_next_to(tp: TurningPoints, run: np.ndarray) -> bool:
    values = np.asarray(tp.values_rad, dtype=float)
    first, last = int(run[0]), int(run[-1])
    before = first > 0 and abs(values[first] - values[first - 1]) / 2 >= np.pi
    after = last < len(values) - 1 and abs(values[last + 1] - values[last]) / 2 >= np.pi
    return bool(before or after)


def _turn_count(series: np.ndarray, prominence: float) -> int:
    if len(series) < 3:
        return 0
    maxima, _ = signal.find_peaks(series, prominence=prominence)
    minima, _ = signal.find_peaks(-series, prominence=prominence)
    return int(len(maxima) + len(minima))


def centre_bands_overlap(cycle_mids: np.ndarray, cycle_amps: np.ndarray) -> bool:
    """Every band [m_i - A_i, m_i + A_i] of the full cycles shares a point with every other: max(m - A) < min(m + A)
    (all pairs |m_i - m_j| < A_i + A_j, the whole centre track, not only its first and last cycle). False for none."""
    mids, amps = np.asarray(cycle_mids, dtype=float), np.asarray(cycle_amps, dtype=float)
    if not len(mids):
        return False
    return bool(np.max(mids - amps) < np.min(mids + amps))


def band_margin(cycle_mids: np.ndarray, cycle_amps: np.ndarray) -> float:
    """min(m + A) - max(m - A) over the full cycles [rad]: positive when every band shares a point
    (`centre_bands_overlap`); nan for none."""
    mids, amps = np.asarray(cycle_mids, dtype=float), np.asarray(cycle_amps, dtype=float)
    if not len(mids):
        return NAN
    return float(np.min(mids + amps) - np.max(mids - amps))


def half_swing_margin(values: np.ndarray) -> float:
    """min over the half-cycles of their upper TP minus the max of their lower TP [rad]: positive when every maximum
    lies above every minimum (the half-swing rule of W17 Q3, a column only); nan below two TPs."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return NAN
    lower, upper = np.minimum(values[1:], values[:-1]), np.maximum(values[1:], values[:-1])
    return float(upper.min() - lower.max())


def cycle_centres(times: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, centres, half-amplitudes) of the full cycles of a run of TPs: the mean of two neighbouring half-cycles
    cancels the alternation, cycle k (TPs k, k+1, k+2) at the time of its middle TP; one half-cycle (two TPs) is
    its own cycle at its middle time."""
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    mids, amps = midpoint_values(values), half_amplitude_values(values)
    if len(mids) >= 2:
        return times[1:-1], (mids[1:] + mids[:-1]) / 2, (amps[1:] + amps[:-1]) / 2
    return (times[:1] + times[1:2]) / 2 if len(times) >= 2 else np.zeros(0), mids, amps


def well_track_rms(cycle_times: np.ndarray, cycle_mids: np.ndarray, well: MovingWell) -> Tuple[float, float, float]:
    """(about c(t), about the best fixed centre, about the time-reversed track c(t0 + t1 - t)) [rad]: RMS of the
    full-cycle centres of a run, t0 and t1 its first and last cycle time. Nan without cycles or a track."""
    times, mids = np.asarray(cycle_times, dtype=float), np.asarray(cycle_mids, dtype=float)
    if not len(mids) or not well.available:
        return NAN, NAN, NAN
    track = mids - well.at(times)
    reversed_track = mids - well.at(times[0] + times[-1] - times)
    fixed = mids - np.mean(mids)
    return tuple(float(np.sqrt(np.mean(x**2))) for x in (track, fixed, reversed_track))


def well_track_valid(rms_track: float, rms_fixed: float, rms_reversed: float) -> bool:
    """The centre follows the well: the observed full-cycle centres lie closer (RMS) to c(t) than to the best fixed
    centre and than to the same track run backwards in time (one criterion, no number of its own; 2013 AS27
    31E-36+0+5: 1.4 deg against 14.9 and 29 deg). A track that does not move cannot beat the fixed centre, and a wrong
    well, a drift the wrong way or a bad branch loses to one of the two nulls."""
    return bool(np.isfinite(rms_track) and rms_track < rms_fixed and rms_track < rms_reversed)


def closure_ratios(values: np.ndarray) -> np.ndarray:
    """For each full cycle (TP_i, TP_i+1, TP_i+2): its net advance |v_i+2 - v_i| over its swing, the mean of its two
    half-swings. Below 1 the cycle is closed, sigma returns about its centre. By algebra the same as: each of its three
    TPs lies on its own side of the cycle's centre (v_i + 2 v_i+1 + v_i+2) / 4 (the full-cycle midpoint of
    `_classify_run`), or: its two half-swings differ less than threefold. The bound is that of the centre estimate, not
    a law of the pendulum: a strong drift of the centre or a change of mode (a co-orbital's tadpole to horseshoe) can
    open a cycle of a real libration, so an episode needs one closed cycle, not all (the entry and exit half-cycles of
    a capture often leave an edge cycle open). A circulation that stalls and turns back a little closes none, while
    the bands of its averaged cycles may still overlap (1864 3E-5S-5: 311 deg down, 5 deg back, 233 deg down)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.zeros(0)
    swings = np.abs(np.diff(values))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(values[2:] - values[:-2]) / ((swings[1:] + swings[:-1]) / 2)


def _classify_run(
    episode_id: int,
    tp: TurningPoints,
    run: np.ndarray,
    record: Tuple[float, float],
    settings: AnalysisSettings,
    window: Optional[Tuple[float, float]] = None,
    well_model: Optional[WellModel] = None,
) -> Episode:
    """`classify_run` on the record edges `record` and the analysis window `window` (None: the record); `well_model`
    the caller's track of the well's centre (`_moving_frame`), read only when the band test in sigma fails."""
    run = np.asarray(run, dtype=int)
    times = np.asarray(tp.times_yr, dtype=float)[run]
    values = np.asarray(tp.values_rad, dtype=float)[run]
    amps = half_amplitude_values(values)
    mids = midpoint_values(values)
    periods = times[2:] - times[:-2] if len(times) >= 3 else np.zeros(0)
    if len(periods):
        period = float(np.median(periods))
    elif len(times) == 2:
        period = float(2 * (times[1] - times[0]))
    else:
        period = NAN
    # full-cycle midpoints: the mean of two neighbouring half-cycles cancels the alternation
    cycle_times, cycle_mids, cycle_amps = cycle_centres(times, values)
    if len(cycle_mids):
        shift = float(cycle_mids[-1] - cycle_mids[0])
        overlap = centre_bands_overlap(cycle_mids, cycle_amps)
        mean_amp = float(np.mean(amps))
        shift_over_amp = abs(shift) / mean_amp if mean_amp > 0 else NAN
        returns = _turn_count(cycle_mids, centre_prominence(tp.run_prominence(run), settings))
        centre_range = float(np.ptp(cycle_mids))
        centre_mod = float(np.mean(mids) % TWO_PI)
    else:
        shift, overlap, shift_over_amp, returns, centre_range = NAN, False, NAN, 0, NAN
        centre_mod = float(values[0] % TWO_PI) if len(values) else NAN
    modulated = returns >= settings.libration_centre_returns_modulated and centre_range < settings.libration_modulated_range_max_rad
    moving = MovingFrame()
    if not (overlap or modulated) and len(run) >= settings.libration_min_turning_points and len(cycle_mids):
        moving = _moving_frame(times, values, cycle_times, cycle_mids, well_model)
    in_frame = values - moving.well.at(times) if moving.accepted else values  # the frame that accepted the run

    estimates = max(0, (len(run) - 1) // 2)
    if not (overlap or moving.accepted):
        stationarity = CheckValue.FAIL
    elif estimates >= settings.libration_centre_min_estimates:
        stationarity = CheckValue.PASS
    else:
        stationarity = CheckValue.NOT_MEASURED
    closure = closure_ratios(in_frame)
    closed = int(np.count_nonzero(closure < 1))
    shape, reason = _shape_or_reason(tp, run, settings, (overlap, modulated, moving.accepted), closed)
    with np.errstate(invalid='ignore'):
        start_gap = (times[0] - record[0]) / period if period > 0 else NAN
        end_gap = (record[1] - times[-1]) / period if period > 0 else NAN
    window = record if window is None else window
    with np.errstate(invalid='ignore'):
        hidden_start = (window[0] - record[0]) / period if period > 0 else NAN
        hidden_end = (record[1] - window[1]) / period if period > 0 else NAN
    return Episode(
        episode_id=episode_id,
        tp_times_yr=times,
        tp_values_rad=values,
        accepted=reason == RejectReason.NONE,
        shape=shape,
        reject_reason=reason,
        period_med_yr=period if len(periods) else NAN,
        period_min_yr=float(periods.min()) if len(periods) else NAN,
        period_max_yr=float(periods.max()) if len(periods) else NAN,
        half_amp_min_rad=float(amps.min()) if len(amps) else NAN,
        half_amp_med_rad=float(np.median(amps)) if len(amps) else NAN,
        half_amp_max_rad=float(amps.max()) if len(amps) else NAN,
        centre_mod_2pi_rad=centre_mod,
        centre_shift_rad=shift,
        centre_shift_over_amp=shift_over_amp,
        bands_overlap=overlap,
        centre_returns=returns,
        coverage_start_gap_p=float(start_gap),
        coverage_end_gap_p=float(end_gap),
        window_start_gap_yr=float(times[0] - window[0]) if len(times) else NAN,
        window_end_gap_yr=float(window[1] - times[-1]) if len(times) else NAN,
        tp_step_max_yr=float(np.max(np.diff(times))) if len(times) >= 2 else NAN,
        window_start_hidden_p=float(hidden_start),
        window_end_hidden_p=float(hidden_end),
        centre_estimates=estimates,
        centre_stationarity=str(stationarity) if len(cycle_mids) else str(CheckValue.NOT_MEASURED),
        closed_cycles=closed,
        closure_ratio_min=float(np.min(closure[np.isfinite(closure)])) if np.any(np.isfinite(closure)) else NAN,
        tp_prominence_rad=tp.run_prominence(run),
        tp_base_prominence_rad=float(tp.prominence_rad),
        tp_local=tuple(tp.local),
        frame=(FRAME_PSI if moving.accepted and not (overlap or modulated) else FRAME_SIGMA) if reason == RejectReason.NONE else '',
        margin_sigma_rad=band_margin(cycle_mids, cycle_amps),
        margin_psi_rad=moving.margin_rad,
        half_swing_margin_rad=half_swing_margin(values),
        well=moving.well if len(moving.well.t_yr) else None,
        well_status=moving.well.status,
        well_rms_rad=moving.rms[0],
        well_rms_fixed_rad=moving.rms[1],
        well_rms_reversed_rad=moving.rms[2],
    )


def _shape_or_reason(
    tp: TurningPoints, run: np.ndarray, settings: AnalysisSettings, centre: Tuple[bool, bool, bool], closed: int
) -> Tuple[EpisodeShape, RejectReason]:
    """(shape, reject reason) of a run from its count, its centre tests (bands in sigma, modulated, bands in psi) and
    its closed cycles (`_classify_run`)."""
    overlap, modulated, moving = centre
    if len(run) < settings.libration_min_turning_points:
        return EpisodeShape.NONE, RejectReason.OVER_THE_TOP if _over_the_top_next_to(tp, run) else RejectReason.TOO_FEW_TP
    if not (overlap or modulated or moving):
        return EpisodeShape.NONE, RejectReason.CENTRE_MOVES
    if not closed:
        return EpisodeShape.NONE, RejectReason.NO_CLOSED_CYCLE
    if overlap:
        return EpisodeShape.STATIONARY, RejectReason.NONE
    return (EpisodeShape.MODULATED if modulated else EpisodeShape.MOVING_CENTRE), RejectReason.NONE


@dataclass(frozen=True)
class MovingFrame:
    """The band test of a run in psi = sigma - c(t) (`_moving_frame`): the track, its status, the three RMS of
    `well_track_rms`, the band margin in psi [rad] and whether the run is accepted there."""

    well: MovingWell = MovingWell('')
    rms: Tuple[float, float, float] = (NAN, NAN, NAN)
    margin_rad: float = NAN
    accepted: bool = False


def bands_hold_bottom(cycle_mids: np.ndarray, cycle_amps: np.ndarray) -> bool:
    """Every full-cycle band [m - A, m + A] of psi = sigma - c(t) holds 0, the bottom of the well: the common point of
    the bands is the predicted centre itself (a libration in a well swings through its minimum, an asymmetric well or a
    horseshoe around two included). A band test that any point passes would accept a swing beside the well (Codex,
    W18). False for no cycle."""
    mids, amps = np.asarray(cycle_mids, dtype=float), np.asarray(cycle_amps, dtype=float)
    return bool(len(mids) and np.all(np.abs(mids) < amps))


def _moving_frame(
    times: np.ndarray, values: np.ndarray, cycle_times: np.ndarray, cycle_mids: np.ndarray, well_model: Optional[WellModel]
) -> MovingFrame:
    """The band test of a run in the frame of its moving well, psi = sigma - c(t) (the libration about a slowly moving
    centre: Gallardo 2006 sect. 3 p. 30; Wisdom 1985 pp. 274, 279). The caller's track c(t) counts only when the
    observed full-cycle centres follow it (`well_track_valid`); the run is then accepted when the same band test as in
    sigma (`centre_bands_overlap`) passes in psi with the bottom of the well inside every band (`bands_hold_bottom`).
    psi is read at the turning points of sigma: they differ from the extrema of psi by c_dot over a fraction of a
    cycle, which the adiabatic condition keeps small (2013 AS27 31:36: 1.2 deg of drift per 150-yr cycle against a
    25-deg half-swing). No model: nothing (status '')."""
    if well_model is None:
        return MovingFrame()
    well = well_model(times, values)
    if not well.available:
        return MovingFrame(well)
    rms = well_track_rms(cycle_times, cycle_mids, well)
    psi = values - well.at(times)
    _, psi_mids, psi_amps = cycle_centres(times, psi)
    margin = band_margin(psi_mids, psi_amps)
    if not well_track_valid(*rms):
        return MovingFrame(replace(well, status=WELL_INVALID), rms, margin)
    return MovingFrame(well, rms, margin, bool(np.isfinite(margin) and margin > 0 and bands_hold_bottom(psi_mids, psi_amps)))


def well_shift(t_yr: np.ndarray, runs: Sequence[Episode], continuous: bool = False) -> Optional[np.ndarray]:
    """c(t) - its mean over the run, on the samples of every accepted run of `runs` in the frame of its moving well
    (FRAME_PSI), 0 elsewhere [rad]: sigma - shift is psi up to a constant, so a law fitted on it sees the well's centre
    stand still while a stationary episode keeps sigma. `continuous`: linear between such runs and held beyond them
    instead of 0 (a frame for the energy fit, whose time derivative must exist). None when no run is in that frame."""
    moving = [run for run in runs if run.accepted and getattr(run, 'frame', '') == FRAME_PSI and getattr(run, 'well', None) is not None]
    if not moving:
        return None
    t = np.asarray(t_yr, dtype=float)
    shift = np.zeros(len(t))
    inside = np.zeros(len(t), dtype=bool)
    for run in moving:
        own = slice(int(np.searchsorted(t, run.t_start_yr, 'left')), int(np.searchsorted(t, run.t_end_yr, 'right')))
        if own.stop <= own.start:
            continue
        centre = run.well.at(t[own])
        shift[own] = centre - float(np.mean(centre))
        inside[own] = True
    if continuous and np.any(inside) and not np.all(inside):
        shift = np.interp(t, t[inside], shift[inside])
    return shift


def episode_centre_travel(geometry: 'Geometry', episode: Episode) -> float:
    """The centre's travel inside an episode [rad]: `centre_travel_in` (its largest monotone pass), or for an episode
    accepted in the frame of its moving well the range of the low-passed centre minus c(t) over it (an upper bound of
    its largest pass in psi): a centre that runs a full turn with its well still librates about the well (82831
    13M-24: the well of Mars moves 330 deg).

    With the MMR rule `Geometry.cycle_travel` an episode in sigma reads its own full-cycle midpoints instead
    (`cycle_centres` of its turning points, their range: an upper bound of their largest monotone pass): the centre
    low-passed over several cycles carries the circulation next to a short episode into it (2017 CY35 1V-1: 12.35
    rad of the low-passed centre inside an episode whose midpoints move ~12 deg)."""
    if getattr(episode, 'frame', '') != FRAME_PSI or getattr(episode, 'well', None) is None:
        if getattr(geometry, 'cycle_travel', False):
            _, mids, _ = cycle_centres(episode.tp_times_yr, episode.tp_values_rad)
            return float(np.ptp(mids)) if len(mids) else 0.0
        return centre_travel_in(geometry, episode.t_start_yr, episode.t_end_yr)
    arrays = _centre_arrays(geometry, episode.t_start_yr)
    if arrays is None:
        return 0.0
    _, t, c = arrays
    inside = (t >= episode.t_start_yr) & (t <= episode.t_end_yr)
    if not np.any(inside):
        return 0.0
    return float(np.ptp(c[inside] - episode.well.at(t[inside])))


def librating_episodes_with(geometry: 'Geometry', floor_of: Callable[[Episode], float], before_jump: bool = False) -> List[Episode]:
    """The accepted episodes (`before_jump`: of the runs before the last jump of a) that are librations of the angle:
    the period not below `floor_of(episode)` [yr] (a faster 'episode' is a forced wave the cutoff let through) and
    the centre's travel inside (`episode_centre_travel`) below 2 pi. A missing period, floor or travel (nan) rejects
    nothing. The MMR and secular decisions call it with their own period floors."""
    scratch = Ledger(thin_band=0.0)
    kept = []
    for episode in geometry.episodes_before_jump if before_jump else geometry.episodes:
        if scratch.compare('period', episode.period_med_yr, Op.LT, floor_of(episode), 'yr', ''):
            continue
        if scratch.compare('centre turn', episode_centre_travel(geometry, episode), Op.GE, TWO_PI, 'rad', ''):
            continue
        kept.append(episode)
    return kept


def classify_run(
    episode_id: int,
    tp: TurningPoints,
    run: np.ndarray,
    t_yr: np.ndarray,
    window: Tuple[float, float],
    settings: AnalysisSettings,
) -> Episode:
    """Accept a run as an episode (>= libration_min_turning_points TPs, a stationary or modulated
    centre and at least one closed full cycle, `closure_ratios`) or reject it with `too_few_tp`, `centre_moves`,
    `no_closed_cycle` or `over_the_top`; coverage gaps from the
    true record edges in units of the run's median period, and from the edges of the analysis `window` in years
    (`covers_edges`).

    Stationary: the bands (midpoint +- half-amplitude) of every full cycle share a point (`centre_bands_overlap`);
    with fewer than libration_centre_min_estimates independent cycles (three TPs: one cycle, one centre) the run is
    still accepted, but its `centre_stationarity` is not_measured (the decision reads it as a thin margin, L6).
    Modulated: the full-cycle midpoints turn back >= libration_centre_returns_modulated
    times (prominence p_c) within a range below libration_modulated_range_max_rad. `over_the_top`:
    too few TPs because a neighbouring half-cycle went over the top. The TPs were selected inside `window` already.
    """
    t = np.asarray(t_yr, dtype=float)
    return _classify_run(episode_id, tp, run, (float(t[0]), float(t[-1])), settings, tuple(window))


def regular_part(times_yr: np.ndarray, ratio: float) -> Optional[Tuple[int, int]]:
    """(first, last) TP index of the longest stretch whose full periods t[i+2] - t[i] stay <= ratio x their median;
    None for fewer than three TPs or no regular cycle."""
    times = np.asarray(times_yr, dtype=float)
    if len(times) < 3:
        return None
    periods = times[2:] - times[:-2]
    regular = periods <= ratio * float(np.median(periods))
    best, start = None, None
    for i, ok in enumerate(np.append(regular, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if best is None or i - start > best[1] - best[0] - 1:
                best = (start, i + 1)  # cycles start..i-1 span TPs start..i+1
            start = None
    return best


def _split_regular(tp: TurningPoints, run: np.ndarray, settings: AnalysisSettings) -> List[np.ndarray]:
    """A run whose centre moves, split at its regular stretch (`regular_part`): a slow edge cycle is a change of regime,
    not part of the stationary libration, and carries the centre away (31935 9M-16+2+5: 5-10 kyr cycles before 25 kyr
    against a 1.8-kyr median, then a stationary centre to 83 kyr). [run] when there is nothing to split."""
    part = _regular_stretch(tp, run, settings)
    if part is None:
        return [run]
    pieces = [run[: part[0]], run[part[0] : part[1] + 1], run[part[1] + 1 :]]
    return [piece for piece in pieces if len(piece)]


def _regular_stretch(tp: TurningPoints, run: np.ndarray, settings: AnalysisSettings) -> Optional[Tuple[int, int]]:
    """(first, last) TP of the regular stretch `_split_regular` cuts out of the run; None when there is nothing to split."""
    part = regular_part(np.asarray(tp.times_yr, dtype=float)[run], settings.libration_regular_period_ratio)
    if part is None or (part[0] == 0 and part[1] == len(run) - 1) or part[1] - part[0] + 1 < settings.libration_min_turning_points:
        return None
    return part


def _trim_slow_edges(
    tp: TurningPoints,
    run: np.ndarray,
    record: Tuple[float, float],
    window: Tuple[float, float],
    settings: AnalysisSettings,
    well_model: Optional[WellModel] = None,
) -> Optional[List[np.ndarray]]:
    """The pieces [left edge, core, right edge] (empty ones dropped) of a run whose centre moves, after removing at most
    ONE slow half-cycle from each end; None when no such cut gives an accepted core.

    An edge half-cycle is slow when it lasts longer than libration_regular_period_ratio x the median half-period of the
    whole run. The candidates are: the left one removed, the right one, both; a candidate counts when its core keeps
    >= libration_min_turning_points TPs and is accepted by `_classify_run`; of several, the one with most TPs, then the
    longer, then the earlier. Nothing is iterated: removing edges until the test passes would accept any uniform drift
    on a short enough piece (its band margin is 2A - (N - 3)|b|, positive for few TPs).

    Exit and entry are events of one half-cycle (the separatrix map changes the energy once per half-period), and a full
    period dilutes them, so `_split_regular` misses them: 2005 GQ107 40M-57+0+17, the last half-cycle of the run at
    31.7-40.2 kyr lasts 1334 yr, 2.07 x the 643-yr median half-period, but its full period 2038 yr is only 1.74 x the
    median period; without it the 12 TPs to 38.85 kyr are stationary."""
    times = np.asarray(tp.times_yr, dtype=float)[run]
    n = len(times)
    if n < 3:
        return None
    half_periods = np.diff(times)
    slow = settings.libration_regular_period_ratio * float(np.median(half_periods))
    left, right = bool(half_periods[0] > slow), bool(half_periods[-1] > slow)
    candidates = []
    if left:
        candidates.append((1, n - 1))
    if right:
        candidates.append((0, n - 2))
    if left and right:
        candidates.append((1, n - 2))
    passing = [
        (lo, hi)
        for lo, hi in candidates
        if hi - lo + 1 >= settings.libration_min_turning_points
        and _classify_run(0, tp, run[lo : hi + 1], record, settings, window, well_model).accepted
    ]
    if not passing:
        return None
    lo, hi = max(passing, key=lambda cut: (cut[1] - cut[0], times[cut[1]] - times[cut[0]], -times[cut[0]]))
    return [piece for piece in (run[:lo], run[lo : hi + 1], run[hi + 1 :]) if len(piece)]


def _trim_outside_well(
    tp: TurningPoints,
    run: np.ndarray,
    record: Tuple[float, float],
    window: Tuple[float, float],
    settings: AnalysisSettings,
    well_model: WellModel,
) -> Optional[List[np.ndarray]]:
    """The pieces [left edge, core, right edge] (empty ones dropped) of a run whose centre moves, without the
    half-cycles at either end that do not swing through the bottom of its moving well (in psi = sigma - c(t) both of
    their TPs on one side of 0); None when the run has no track, nothing is cut, the core keeps fewer than
    libration_min_turning_points TPs or is not accepted in psi (`_classify_run`). An exit or an entry takes its last swings
    off the well (2003 UG22 10E-17+0+7, 40 kyr: TPs of psi -108, 67, 20, 90 deg at 20.36-20.69 kyr after 170 TPs
    through the bottom). Only the ends: a half-cycle beside the well inside the run rejects it."""
    times = np.asarray(tp.times_yr, dtype=float)[run]
    values = np.asarray(tp.values_rad, dtype=float)[run]
    well = well_model(times, values)
    if not well.available or len(run) < 2:
        return None
    psi = values - well.at(times)
    through = (np.maximum(psi[1:], psi[:-1]) > 0) & (np.minimum(psi[1:], psi[:-1]) < 0)
    inside = np.flatnonzero(through)
    if not len(inside) or (inside[0] == 0 and inside[-1] == len(through) - 1):
        return None
    lo, hi = int(inside[0]), int(inside[-1]) + 1  # the core's first and last TP
    if hi - lo + 1 < settings.libration_min_turning_points:
        return None
    core = _classify_run(0, tp, run[lo : hi + 1], record, settings, window, well_model)
    if not (core.accepted and core.frame == FRAME_PSI):  # never a shorter piece of a drift that passes in sigma
        return None
    return [piece for piece in (run[:lo], run[lo : hi + 1], run[hi + 1 :]) if len(piece)]


@dataclass(frozen=True)
class ReturnCycle:
    """One cycle closed by its phase return (`phase_return`): sigma_f crosses the mid-level of two opposite turning
    points before the first (t_a) and after the second (t_b) in the same direction, back in the same state."""

    t_a_yr: float
    t_b_yr: float
    level_rad: float  # v2 + level_fraction (v1 - v2); the mid-level (v1 + v2) / 2 by default
    slow_mismatch: float  # |slow(t_b) - slow(t_a)| / ptp(slow on [t_a, t_b])
    rate_mismatch: float  # |rate(t_b) / rate(t_a) - 1|
    period_ratio: float  # (t_b - t_a) / (2 (t2 - t1))
    identity_corr: float  # corr(rate, sign(c) (slow - mean)) on [t_a, t_b]
    swing_asymmetry: float = NAN  # |rate(t_m) / rate(t_a) + 1|, t_m the crossing of the level between the TPs
    level_fraction: float = 0.5

    @property
    def period_yr(self) -> float:
        return float(self.t_b_yr - self.t_a_yr)


def _crossing_time(t: np.ndarray, x: np.ndarray, j: int, level: float) -> float:
    """The time x crosses `level` between samples j and j + 1 (linear)."""
    y0, y1 = x[j] - level, x[j + 1] - level
    return float(t[j] + (t[j + 1] - t[j]) * (-y0) / (y1 - y0)) if y1 != y0 else float(t[j])


def return_crossings(
    t: np.ndarray, x: np.ndarray, indices: Tuple[int, int], level: float, direction: float, bounds: Tuple[int, int]
) -> Tuple[float, float]:
    """(t_a, t_b): the last time x crosses `level` going in `direction` before sample indices[0], and the first time after
    indices[1], both within the samples `bounds` (inclusive); nan when there is none."""
    i1, i2 = indices
    lo, hi = bounds
    before = direction * (x[lo : i1 + 1] - level)
    below = np.flatnonzero(before[:-1] < 0)
    t_a = _crossing_time(t, x, lo + int(below[-1]), level) if len(below) else NAN
    after = direction * (x[i2 : hi + 1] - level)
    reached = np.flatnonzero(after[1:] >= 0)
    t_b = _crossing_time(t, x, i2 + int(reached[0]), level) if len(reached) else NAN
    return t_a, t_b


def phase_return(
    angle: 'FilteredAngle',
    tp: TurningPoints,
    pair: np.ndarray,
    settings: AnalysisSettings,
    bounds: Optional[Tuple[float, float]] = None,
) -> Optional[ReturnCycle]:
    """Closure of one cycle by its phase return (author 2026-10-19, E2): two consecutive opposite turning points
    (v1 at t1, v2 at t2, Delta = t2 - t1) are a full cycle when, with sigma_m = (v1 + v2) / 2,

    1. sigma_f crosses sigma_m before t1 (t_a) and after t2 (t_b) in the same direction, inside `bounds` [yr] (the
       record of the run's regime; the whole series without it). The filter's edge trim is not excluded: the
       conditions below test the state at both crossings, so a crossing the filter bent at an edge fails them (3753
       1E-1 at 1 kyr: t_a = 23 yr, inside the 55.5-yr trim, a back within 0.17 % and the rate within 2.4 %);
    2. the slow variable comes back: |a(t_b) - a(t_a)| <= libration_return_tol x ptp(a on [t_a, t_b]);
    3. so does the rate: |rate(t_b) / rate(t_a) - 1| <= libration_return_tol;
    4. (t_b - t_a) / (2 Delta) is 1 within libration_return_period_tol: a libration cycle spends as long from the
       mid-level to a turn as back, while a circulation with a short back-step crosses sigma_m right next to its
       turning points, (t_b - t_a) ~ Delta (1864 3E-5S-5, 5 deg back: ratio ~ 0.5);
    5. a carries the swing on [t_a, t_b]: corr(rate, sign(c) (a - mean a)) >= libration_return_identity_corr_min (c the
       identity coefficient); a kinematic back-swing with a flat a passes 4 but not 5;
    6. the swing is symmetric: sigma_f crosses sigma_m between the two turning points (t_m) as fast as at t_a, the
       other way, |rate(t_m) / rate(t_a) + 1| <= libration_return_tol: the same state at the same sigma on either side
       of the cycle (W20, added to the plan's 1-5: a centre drifting at 2 pi per 20 kyr with 15-deg loops of 4.4 kyr
       passes 1-5 at every loop, rate +1.13e-4 forth against -0.60e-4 rad/yr back, an asymmetry of 0.47).

    2016 CA138 1E-1: sigma_f crosses -180 deg going down at 80.1 and 453.9 yr, a - a_res -0.0035016 and -0.0035124
    au, the rate -1.898 and -1.904 deg/yr: one horseshoe cycle of 374 yr between its two turning points at 169 and
    353 yr. The 3-TP rule cannot see a cycle that starts between two turns. None when a condition fails.

    A symmetric reversal of a circulation passes 1-5 as well (locally it is one small cycle, the identity holds): the
    decision drops it when its return period is below the angle's shortest libration (`mmr.decide.librating_episodes`).

    The mid-level is a convention: any level inside the swing is a section the cycle crosses twice in the same
    direction. The mid-level and the levels v2 + k (v1 - v2) / (n + 1), k = 1..n (n = libration_return_levels; 0 or
    1: the mid-level only) are tried with the same conditions 1-6 (`return_level`). A level counts only inside a band of
    consecutive closing levels, at least libration_return_band_levels of them, the mid-level
    libration_return_mid_band_levels (one closing neighbour; W20 round C, review B #10), counted over the observable
    levels: a level whose crossing may lie beyond the record (`return_observable`) neither closes nor breaks a band, so
    a mid-level without an observable neighbour closes alone (W20 round D, review C #1). The mismatches change
    continuously with the level, so a real return holds over a band of sections, while a match at one isolated
    section of many is a coincidence (2020 CB 1E-1: 0.90 of a 312-deg swing only, next to a turning point; 60608
    1U-1N-2: a 67-deg back-step of a circulation, 0.70 only). The mid-level is kept when it counts; else the level
    with the smallest worst mismatch (each over its tolerance):
    164207 1E-1 never crosses its mid-level before the horseshoe turn at 752 yr, but returns at 0.15-0.20 of the swing
    (611 -> 973 yr, the rate within 0.6 %); 2025 RB7 at 1 kyr returns at 0.20-0.25 (62 -> 771 yr; at the mid-level a
    is 6.8 % and the rate 22 % off). The crossings are sought only within (1 + libration_return_period_tol) 2 Delta of
    the turning points (`return_samples`).
    """
    t = np.asarray(angle.t_yr, dtype=float)
    i1, i2 = (int(i) for i in np.asarray(tp.indices, dtype=int)[np.asarray(pair, dtype=int)])
    samples = return_samples(t, (i1, i2), bounds, settings.libration_return_period_tol)
    n = settings.libration_return_levels
    fractions = sorted({0.5, *(k / (n + 1) for k in range(1, n + 1))})
    cycles = [return_cycle(angle, tp, pair, bounds, fraction, samples) for fraction in fractions]
    closing = [c is not None and closes(c, settings) for c in cycles]
    clipped = return_clipped(t, (i1, i2), bounds, settings.libration_return_period_tol)
    observable = [c is not None or return_observable(angle, tp, pair, fraction, samples, clipped) for c, fraction in zip(cycles, fractions)]
    mismatches = [return_mismatch(c, settings) if shut else NAN for c, shut in zip(cycles, closing)]
    chosen = return_level(
        fractions,
        closing,
        mismatches,
        len(fractions) == 1,
        settings.libration_return_band_levels,
        observable,
        settings.libration_return_mid_band_levels,
    )
    return cycles[chosen] if chosen is not None else None


def return_clipped(t: np.ndarray, indices: Tuple[int, int], bounds: Optional[Tuple[float, float]], period_tol: float) -> Tuple[bool, bool]:
    """(before, after): whether the record (or `bounds`) ends inside the window `return_samples` searches before the
    first turning point and after the second: a crossing missing on such a side may lie beyond the record."""
    i1, i2 = indices
    reach = (1 + period_tol) * 2 * float(t[i2] - t[i1])
    start, end = (t[0], t[-1]) if bounds is None else bounds
    return bool(float(start) > float(t[i1]) - reach), bool(float(end) < float(t[i2]) + reach)


def return_observable(
    angle: 'FilteredAngle', tp: TurningPoints, pair: np.ndarray, fraction: float, samples: Tuple[int, int], clipped: Tuple[bool, bool]
) -> bool:
    """Whether a level without its two crossings (`return_cycle` None) was observable: False when a crossing is missing
    on a side where the record ends inside the search window (`return_clipped`): it may lie beyond the record (W20 round
    D, review C #1: a 400-yr cosine on -104..304 yr crosses its mid-level at -100 and 300 yr, but 0.45 and 0.55 of the
    swing only at -106 and 306 yr)."""
    t = np.asarray(angle.t_yr, dtype=float)
    x = np.asarray(angle.angle_f, dtype=float)
    i1, i2 = (int(i) for i in np.asarray(tp.indices, dtype=int)[np.asarray(pair, dtype=int)])
    v1, v2 = float(x[i1]), float(x[i2])
    lo, hi = samples
    if hi - lo + 1 < 2 or v1 == v2:
        return True
    t_a, t_b = return_crossings(t, x, (i1, i2), v2 + fraction * (v1 - v2), float(np.sign(v1 - v2)), (lo, hi))
    return not ((not np.isfinite(t_a) and clipped[0]) or (not np.isfinite(t_b) and clipped[1]))


def return_level(
    fractions: Sequence[float],
    closing: Sequence[bool],
    mismatches: Sequence[float],
    mid_only: bool,
    least: int,
    observable: Optional[Sequence[bool]] = None,
    mid_least: int = 2,
) -> Optional[int]:
    """Which level of `phase_return` closes the cycle (index into `fractions`, which hold 0.5). The bands of closing
    levels (`closing_band`) are counted over the observable levels only (`observable`, all by default: a level whose
    crossing may lie beyond the record neither closes nor breaks a band). The mid-level when it closes inside a band of
    at least `mid_least` (LIBRATION_RETURN_MID_BAND_LEVELS), or of every observable level when fewer (alone when no
    neighbour is observable, or `mid_only`: no grid); else, of the levels inside a band of at least `least`, the one
    with the smallest mismatch; None when none qualifies."""
    seen = [i for i in range(len(fractions)) if observable is None or observable[i]]
    band = [0] * len(fractions)
    for i, count in zip(seen, closing_band([closing[i] for i in seen])):
        band[i] = count
    middle = list(fractions).index(0.5)
    if closing[middle] and (mid_only or band[middle] >= min(mid_least, len(seen))):
        return middle
    kept = [i for i in seen if closing[i] and i != middle and band[i] >= least]
    return min(kept, key=lambda i: mismatches[i]) if kept else None


def closing_band(closing: Sequence[bool]) -> List[int]:
    """For each level (in order of its fraction), the number of consecutive closing levels it belongs to (0 when it does
    not close)."""
    band = [0] * len(closing)
    i = 0
    while i < len(closing):
        if not closing[i]:
            i += 1
            continue
        j = i
        while j < len(closing) and closing[j]:
            j += 1
        band[i:j] = [j - i] * (j - i)
        i = j
    return band


def return_samples(t: np.ndarray, indices: Tuple[int, int], bounds: Optional[Tuple[float, float]], period_tol: float) -> Tuple[int, int]:
    """The samples [lo, hi] (inclusive) a phase return of the turning points at `indices` can use: inside `bounds` and
    within (1 + period_tol) 2 Delta of the turning points (Delta = t2 - t1). A crossing farther out gives
    (t_b - t_a) / (2 Delta) > 3/2 + period_tol, beyond condition 4: the search stops there (W20 round C, review B #5)."""
    i1, i2 = indices
    reach = (1 + period_tol) * 2 * float(t[i2] - t[i1])
    start, end = (t[0], t[-1]) if bounds is None else bounds
    if not (np.isfinite(start) and np.isfinite(end)):  # no regime: no sample (as `window_mask`)
        return 0, -1
    start, end = max(float(start), float(t[i1]) - reach), min(float(end), float(t[i2]) + reach)
    return int(np.searchsorted(t, start, side='left')), int(np.searchsorted(t, end, side='right')) - 1


def return_mismatch(cycle: ReturnCycle, settings: AnalysisSettings) -> float:
    """The worst of conditions 2, 3, 4 and 6 of `phase_return`, each over its tolerance (<= 1 closes; the tolerances
    are positive, `settings.POSITIVE`)."""
    tol = settings.libration_return_tol
    return max(
        cycle.slow_mismatch / tol,
        cycle.rate_mismatch / tol,
        abs(cycle.period_ratio - 1) / settings.libration_return_period_tol,
        cycle.swing_asymmetry / tol,
    )


def closes(cycle: ReturnCycle, settings: AnalysisSettings) -> bool:
    """Conditions 2-6 of `phase_return` on the measured return."""
    tol = settings.libration_return_tol
    return bool(
        cycle.slow_mismatch <= tol
        and cycle.rate_mismatch <= tol
        and abs(cycle.period_ratio - 1) <= settings.libration_return_period_tol
        and cycle.identity_corr >= settings.libration_return_identity_corr_min
        and cycle.swing_asymmetry <= tol
    )


def return_cycle(
    angle: 'FilteredAngle',
    tp: TurningPoints,
    pair: np.ndarray,
    bounds: Optional[Tuple[float, float]] = None,
    fraction: float = 0.5,
    samples: Optional[Tuple[int, int]] = None,
) -> Optional[ReturnCycle]:
    """The numbers of `phase_return` for two consecutive turning points (`pair`, indices into `tp`) at the level
    v2 + fraction (v1 - v2) (0.5: the mid-level), whether they close or not; None without the two crossings
    (condition 1) or an identity coefficient. `samples` [lo, hi] limit the search (`return_samples`); without them,
    every sample inside `bounds`. Reads only those samples."""
    t = np.asarray(angle.t_yr, dtype=float)
    x = np.asarray(angle.angle_f, dtype=float)
    i1, i2 = (int(i) for i in np.asarray(tp.indices, dtype=int)[np.asarray(pair, dtype=int)])
    v1, v2 = float(x[i1]), float(x[i2])
    if samples is None:
        samples = return_samples(t, (i1, i2), bounds, math.inf)
    lo, hi = samples
    coefficient = float(angle.identity_coefficient)
    if hi - lo + 1 < 2 or v1 == v2 or not (np.isfinite(coefficient) and coefficient != 0):
        return None
    level = v2 + fraction * (v1 - v2)
    t_a, t_b = return_crossings(t, x, (i1, i2), level, float(np.sign(v1 - v2)), (lo, hi))
    if not (np.isfinite(t_a) and np.isfinite(t_b)):
        return None
    slow, rate = np.asarray(angle.slow_f, dtype=float), np.asarray(angle.rate_f, dtype=float)
    between = np.sign(v1 - v2) * (x[i1 : i2 + 1] - level)
    passed = np.flatnonzero(between[1:] < 0)
    t_m = _crossing_time(t, x, i1 + int(passed[0]), level) if len(passed) else NAN
    first, last = int(np.searchsorted(t, t_a, side='left')), int(np.searchsorted(t, t_b, side='right'))
    span = slice(first, last)  # the samples in [t_a, t_b]
    near = slice(max(first - 1, 0), min(last + 1, len(t)))  # with the samples around t_a and t_b
    swing = float(np.ptp(slow[span])) if last - first >= 2 else NAN
    slow_a, slow_b = np.interp([t_a, t_b], t[near], slow[near])
    rate_a, rate_b = np.interp([t_a, t_b], t[near], rate[near])
    rate_m = float(np.interp(t_m, t[i1 : i2 + 1], rate[i1 : i2 + 1])) if np.isfinite(t_m) else NAN
    with np.errstate(divide='ignore', invalid='ignore'):
        slow_mismatch = float(abs(slow_b - slow_a) / swing)
        rate_mismatch = float(abs(rate_b / rate_a - 1))
        asymmetry = float(abs(rate_m / rate_a + 1))
        period_ratio = float((t_b - t_a) / (2 * (t[i2] - t[i1])))
        part, response = rate[span], np.sign(coefficient) * (slow[span] - np.mean(slow[span]))
        corr = float(np.corrcoef(part, response)[0, 1]) if np.std(part) > 0 and np.std(response) > 0 else NAN
    return ReturnCycle(float(t_a), float(t_b), float(level), slow_mismatch, rate_mismatch, period_ratio, corr, asymmetry, fraction)


def _returned(episode: Episode, cycle: ReturnCycle, record: Tuple[float, float], window: Tuple[float, float]) -> Episode:
    """A 2-TP run closed by its phase return as an accepted stationary episode of one cycle: its period is the return's
    t_b - t_a (the coverage gaps in that period), its single centre not measured (`centre_stationarity`)."""
    period = cycle.period_yr
    times = episode.tp_times_yr
    return replace(
        episode,
        accepted=True,
        shape=EpisodeShape.STATIONARY,
        reject_reason=RejectReason.NONE,
        period_med_yr=period,
        period_min_yr=period,
        period_max_yr=period,
        coverage_start_gap_p=float((times[0] - record[0]) / period),
        coverage_end_gap_p=float((record[1] - times[-1]) / period),
        window_start_hidden_p=float((window[0] - record[0]) / period),
        window_end_hidden_p=float((record[1] - window[1]) / period),
        centre_stationarity=str(CheckValue.NOT_MEASURED),
        closed_cycles=1,
        frame=FRAME_SIGMA,
        closed_by_return=True,
        return_t_a_yr=cycle.t_a_yr,
        return_t_b_yr=cycle.t_b_yr,
        return_level_fraction=cycle.level_fraction,
    )


@dataclass(frozen=True)
class RegimeCut:
    """A run cut at a regime change (`split_at_regime_changes`): its pieces in time order and the indices of those the
    cut claims to be regimes (both sides of a centre step, the stall itself); the cut stands only when every one of
    them is accepted."""

    pieces: List[np.ndarray]
    regimes: Tuple[int, ...]


def split_at_regime_changes(tp: TurningPoints, run: np.ndarray, settings: AnalysisSettings) -> Optional[RegimeCut]:
    """A run whose centre moves, cut where its regime changes (F2, K3); None when there is no such place.

    1. A centre step: a half-cycle such that the full cycles on each side of it pass the band test
       (`centre_bands_overlap`) while the two sides together do not, each side with a closed cycle (`closure_ratios`)
       and a measured centre (>= libration_centre_min_estimates independent cycles: a side of one cycle has one centre,
       nothing shows that it stays), and the cycle centres on either side of the half-cycle differ by more than the
       range of the cycle centres inside either side (a drift whose travel is two to four half-swings passes the band
       test on each half and would be cut anywhere: W18's drift along a reversed well, +138 deg). The half-cycle
       belongs to neither side (as in `split_runs`). Of several, the one across which the mean full-cycle centres of
       the two sides differ most (a 1:1 quasi-satellite about 0 deg turning into a tadpole about 83 deg: 524522 1V-1).
    2. Failing that, a stall of the slow centre: the longest stretch of >= libration_stall_min_cycles independent full
       cycles (no shared half-cycle) in which the centre moves less per period than the cycles swing, |delta m| / dt x
       P below the mean half-amplitude of the two cycles of each step (m the full-cycle midpoints, `cycle_centres`; P
       the median full period of the run, which the stall's own must match within libration_regular_period_ratio); it
       is cut out as a piece of its own (385266 16N-27: the slow centre stops at
       55-100 kyr between descents of 7-11 deg/kyr). The criterion is relative to the swing, so a steady drift of the
       centre has no stall and is not cut ("произвольно резать нельзя", D report 2026-10-19).

    Only for the MMR (`FilteredAngle.split_regimes`). The caller keeps the cut only when the regimes it claims are
    accepted: a fragment left over by a cut is no reason to cut (1627 3V-4M-7: a stall of 55 TPs that is no episode,
    with 3 TPs after it that are)."""
    run = np.asarray(run, dtype=int)
    n, m = len(run), settings.libration_min_turning_points
    side = max(m, 2 * settings.libration_centre_min_estimates + 1)  # TPs of a side with a measured centre
    if n < 2 * side:
        return None
    times = np.asarray(tp.times_yr, dtype=float)[run]
    values = np.asarray(tp.values_rad, dtype=float)[run]
    cycle_times, mids, amps = cycle_centres(times, values)
    with np.errstate(invalid='ignore'):
        closed = closure_ratios(values) < 1
    step = _centre_step(mids, amps, closed, n, side)
    if step is not None:
        return RegimeCut([run[: step + 1], run[step + 1 :]], (0, 1))
    stall = _centre_stall(times, cycle_times, mids, amps, float(np.median(times[2:] - times[:-2])), settings)
    if stall is None or (stall[0] == 0 and stall[1] == n - 1):
        return None
    pieces = [run[: stall[0]], run[stall[0] : stall[1] + 1], run[stall[1] + 1 :]]
    return RegimeCut([piece for piece in pieces if len(piece)], (1 if stall[0] > 0 else 0,))


def _centre_step(mids: np.ndarray, amps: np.ndarray, closed: np.ndarray, n: int, m: int) -> Optional[int]:
    """The TP index i of the best centre step (`split_at_regime_changes` 1): the half-cycle (i, i + 1) is the step,
    the left side TPs 0..i (cycles 0..i-2), the right side TPs i+1..n-1 (cycles i+1..n-3), each with >= m TPs and a closed
    cycle (`closed`, per cycle: `closure_ratios` < 1; a side of one open cycle is the swing into or out of the step,
    not a regime: 2003 KQ18 6J+2S-1, a slow 79-deg swing next to 2.8-deg ripple). None for none. Running extremes of
    the band edges over the cycles from each end give every candidate at once."""
    low, high = mids - amps, mids + amps
    steps = np.arange(m - 1, n - m)
    if not len(steps):
        return None
    left_low, left_high = np.maximum.accumulate(low)[steps - 2], np.minimum.accumulate(high)[steps - 2]
    right_low = np.maximum.accumulate(low[::-1])[::-1][steps + 1]
    right_high = np.minimum.accumulate(high[::-1])[::-1][steps + 1]
    left_margin, right_margin = left_high - left_low, right_high - right_low
    across = np.minimum(left_high, right_high) - np.maximum(left_low, right_low)
    left_closed = np.maximum.accumulate(closed)[steps - 2]
    right_closed = np.maximum.accumulate(closed[::-1])[::-1][steps + 1]
    # a step, not a drift cut in two: the centre moves more across the step than anywhere inside either side
    left_range = (np.maximum.accumulate(mids) - np.minimum.accumulate(mids))[steps - 2]
    right_range = (np.maximum.accumulate(mids[::-1]) - np.minimum.accumulate(mids[::-1]))[::-1][steps + 1]
    jump = np.abs(mids[steps + 1] - mids[steps - 2])
    stepped = jump > np.maximum(left_range, right_range)
    valid = (left_margin > 0) & (right_margin > 0) & ~(across > 0) & left_closed & right_closed & stepped
    if not np.any(valid):
        return None
    # of several, where the centre jumps most: the mean full-cycle centres of the two sides farthest apart
    total = np.concatenate([[0.0], np.cumsum(mids)])
    left_mean = total[steps - 1] / (steps - 1)
    right_mean = (total[-1] - total[steps + 1]) / (len(mids) - steps - 1)
    jump = np.where(valid, np.abs(right_mean - left_mean), -np.inf)
    return int(steps[int(np.argmax(jump))])


def _centre_stall(
    times: np.ndarray, cycle_times: np.ndarray, mids: np.ndarray, amps: np.ndarray, period_yr: float, settings: AnalysisSettings
) -> Optional[Tuple[int, int]]:
    """(first, last) TP index of the longest stall of the slow centre (`split_at_regime_changes` 2), None for none. The
    stall is the run's own oscillation with its centre stopped, so its median full period lies within
    libration_regular_period_ratio of the run's (`period_yr`): a stretch of faster ripple inside an irregular run is no
    stall of it (46596 8J+5S-3: 18 TPs of 0.8 deg at a 286-yr period in a run of 6-kyr cycles)."""
    if len(mids) < 2 or not (np.isfinite(period_yr) and period_yr > 0):
        return None
    ratio = settings.libration_regular_period_ratio

    def own_period(first: int, last: int) -> bool:
        piece = times[first : last + 1]
        stall_period = float(np.median(piece[2:] - piece[:-2]))
        return bool(period_yr / ratio <= stall_period <= ratio * period_yr)

    with np.errstate(divide='ignore', invalid='ignore'):
        per_period = np.abs(np.diff(mids)) / np.diff(cycle_times) * period_yr
    slow = per_period < (amps[1:] + amps[:-1]) / 2
    best: Optional[Tuple[int, int]] = None
    start = None
    for k, ok in enumerate(np.append(slow, False)):
        if ok and start is None:
            start = k
        elif not ok and start is not None:
            first, last = start, k + 2  # steps start..k-1 join cycles start..k, i.e. TPs start..k+2
            long_enough = (last - first) // 2 >= settings.libration_stall_min_cycles
            if long_enough and own_period(first, last) and (best is None or last - first > best[1] - best[0]):
                best = (first, last)
            start = None
    return best


def _find_episodes(
    tp: TurningPoints,
    record: Tuple[float, float],
    window: Tuple[float, float],
    settings: AnalysisSettings,
    first_id: int = 0,
    well_model: Optional[WellModel] = None,
    angle: Optional['FilteredAngle'] = None,
    return_bounds: Optional[Tuple[float, float]] = None,
) -> List[Episode]:
    """Every run classified. A run rejected with `centre_moves` is classified again in pieces: first at its regular
    stretch (`_split_regular`, slow full cycles), kept split only when a piece is accepted; failing that, without one
    slow edge half-cycle at either end (`_trim_slow_edges`), the trimmed half-cycles kept as rejected `edge` pieces.
    Neither touches an accepted run.

    The frame of a moving well (`well_model`, `_classify_run`) comes after sigma: on the regular stretch of a split
    (never on its slow edge cycles, a change of regime: 31935 9M-16+2+5, 5-10 kyr cycles before 29 kyr), then, when
    sigma accepts no piece, on the whole run, on the run without the half-cycles at its ends that do not swing through
    the bottom of the well (`_trim_outside_well`) and without one slow edge half-cycle.

    The MMR rules of `angle` (`FilteredAngle`; None: neither): `return_closure`, a run of two turning points closed by
    its phase return (`phase_return`) is one cycle; `split_regimes`, a run still rejected for its moving centre after
    all of the above is cut at a centre step or a stall (`split_at_regime_changes`), kept split only when the regimes
    the cut claims are accepted. A phase return is sought inside `return_bounds` (the run's regime between jumps of the
    slow variable; `record` without it): a crossing beyond a confirmed jump belongs to another regime."""
    episodes: List[Episode] = []
    return_bounds = record if return_bounds is None else return_bounds
    returns = angle is not None and angle.return_closure
    cuts = angle is not None and angle.split_regimes
    for run in split_runs(tp, settings):
        first = first_id + len(episodes)
        episode = _classify_run(first, tp, run, record, settings, window)
        if returns and len(run) == 2 and episode.reject_reason in (RejectReason.TOO_FEW_TP, RejectReason.OVER_THE_TOP):
            cycle = phase_return(angle, tp, run, settings, return_bounds)
            episode = _returned(episode, cycle, record, window) if cycle is not None else episode
        if episode.reject_reason != RejectReason.CENTRE_MOVES:
            episodes.append(episode)
            continue
        pieces = _split_regular(tp, run, settings)
        if len(pieces) > 1:
            regular = int(_regular_stretch(tp, run, settings)[0] > 0)  # the index of the regular piece
            split = [
                _classify_run(first + i, tp, piece, record, settings, window, well_model if i == regular else None)
                for i, piece in enumerate(pieces)
            ]
            if any(piece.accepted for piece in split):
                episodes.extend(split)
                continue
        trimmed = _trim_slow_edges(tp, run, record, window, settings)
        if trimmed is None and well_model is not None:
            episode = _classify_run(first, tp, run, record, settings, window, well_model)  # rejected: keeps the psi numbers
            if episode.accepted:
                episodes.append(episode)
                continue
            trimmed = _trim_outside_well(tp, run, record, window, settings, well_model)
            if trimmed is None:
                trimmed = _trim_slow_edges(tp, run, record, window, settings, well_model)
        if trimmed is None:
            cut = split_at_regime_changes(tp, run, settings) if cuts else None
            split = [
                _classify_run(first + i, tp, piece, record, settings, window, well_model)
                for i, piece in enumerate(cut.pieces if cut else ())
            ]
            episodes.extend(split if cut is not None and all(split[i].accepted for i in cut.regimes) else [episode])
            continue
        split = [_classify_run(first + i, tp, piece, record, settings, window, well_model) for i, piece in enumerate(trimmed)]
        episodes.extend(piece if piece.accepted else replace(piece, reject_reason=RejectReason.EDGE) for piece in split)
    return episodes


def find_episodes(t_yr: np.ndarray, tp: TurningPoints, window: Tuple[float, float], settings: AnalysisSettings) -> List[Episode]:
    """Every run as an `Episode` (accepted or rejected), in time order, ids from 0."""
    t = np.asarray(t_yr, dtype=float)
    return _find_episodes(tp, (float(t[0]), float(t[-1])), tuple(window), settings)


def short_runs(runs: Sequence[Episode]) -> List[Episode]:
    """The runs that missed being an episode by their length alone: rejected too_few_tp (no over-the-top half-cycle
    next to them) with at least one half-cycle, every one below pi. The candidates of a near-episode (2016 CA138
    1E-1: 2 TPs at 169-356 yr, half-amplitude 175 deg: a capture that turned back once)."""
    return [
        run
        for run in runs
        if run.reject_reason == RejectReason.TOO_FEW_TP and np.isfinite(run.half_amp_max_rad) and run.half_amp_max_rad < np.pi
    ]


def window_step_limit_yr(episode: Episode, settings: AnalysisSettings) -> float:
    """The longest gap [yr] between the analysis window's edge and the episode that misses no turning point: the
    longest TP-to-TP step of the episode, at most libration_coverage_periods of its periods (`covers_edges`)."""
    bound = settings.libration_coverage_periods * episode.period_med_yr
    return float(min(episode.tp_step_max_yr, bound)) if np.isfinite(bound) else float(episode.tp_step_max_yr)


def covers_edges(episode: Episode, settings: AnalysisSettings) -> Tuple[bool, bool]:
    """(start, end): the episode reaches that edge of the record. An edge is covered when the gap from the record edge
    is within libration_coverage_periods periods of the episode, or when no turning point is missing between the edge
    of the analysis window and the episode (that gap is not longer than the longest TP-to-TP step inside the episode,
    nor than libration_coverage_periods periods) and the stretch the filter's trim hides between the record's and the
    window's edges is itself within libration_coverage_periods periods.

    The second reading needs no number of its own. The trim counts in the record-edge gap a stretch no turning point can
    be seen in, and TP-to-TP steps are uneven (2003 UG22 10E-17+0+7: steps of 110-200 yr, the next maximum 15 yr past
    the window's end gives 1.07 P from the record's end, 0.68 of the longest step from the window's, 0.45 P hidden; the
    1-kyr co-orbitals 2018 DV1, 2019 JH7, 2019 SB6, 2019 VL5, 2022 GO3 miss a TP and stay episodic). The hidden
    stretch matters at the centre scale (G2c), where the trim of a long cutoff can hide several periods (12499 4M-7:
    4.2 P before the window): no turning point missed in the window says nothing about them. A step longer than the
    limit itself is a hole in the episode, not a step of its libration (154634 1M+1J-2 at the fine scale: TPs of a
    308-yr period seen only near the turns of a 29-kyr centre libration, a 25.6-kyr step, the first TP 71 P after the
    window's start), so the window gap is bounded by libration_coverage_periods periods too."""
    limit = settings.libration_coverage_periods
    step_limit = window_step_limit_yr(episode, settings)

    def covered(gap_p: float, gap_yr: float, hidden_p: float) -> bool:
        return bool(gap_p <= limit or (gap_yr <= step_limit and hidden_p <= limit))

    start = covered(episode.coverage_start_gap_p, episode.window_start_gap_yr, episode.window_start_hidden_p)
    end = covered(episode.coverage_end_gap_p, episode.window_end_gap_yr, episode.window_end_hidden_p)
    # an observed entry (exit) is a regime the record shows beyond the episode (MMR edge rule, `_visible_edge`)
    return start and not getattr(episode, 'entry_observed', False), end and not getattr(episode, 'exit_observed', False)


def covering_episode(episodes: Sequence[Episode], settings: AnalysisSettings) -> Optional[Episode]:
    """The accepted episode (not before the last jump of a) that covers both edges (`covers_edges`); of several, the one
    with most TPs. None if none."""
    covering = [
        episode for episode in episodes if episode.accepted and not episode.before_last_jump and all(covers_edges(episode, settings))
    ]
    return max(covering, key=lambda episode: episode.n_tp) if covering else None


@dataclass(frozen=True)
class UnionCover:
    """The chain of librating episodes that covers the record together (`covering_union`) and the numbers it passed:
    the largest ranges of sigma_f's turning points and of the orbit-averaged raw angle inside one gap and over the
    whole chain [rad], the longest gap over (P_left + P_right), and how far the turning points beyond the outer
    episodes fall short of each record edge [P of that episode] (nan where `covers_edges` covered it). Empty: no chain."""

    episodes: Tuple[Episode, ...] = ()
    tp_range_rad: float = NAN
    raw_range_rad: float = NAN
    span_tp_range_rad: float = NAN
    span_raw_range_rad: float = NAN
    gap_max_p: float = NAN
    start_shortfall_p: float = NAN
    end_shortfall_p: float = NAN


def covering_union(geometry: 'Geometry', episodes: Sequence[Episode], settings: AnalysisSettings) -> UnionCover:
    """Coverage of the record by consecutive librating `episodes` without a circulation between them (author
    2026-10-19 for the 1:1, "a regime change without circulation is not an exit": 469219, 164207, 524522; W22 for every
    MMR, the safe variant of round 1024). Neighbours join when the gap between them holds no turn, the range of
    sigma_f's turning points from the end of one to the start of the next (`break_range`) and the range of the
    orbit-averaged raw angle there (`Geometry.gap_raw_range`) both below 2pi, and lasts at most
    libration_union_gap_periods x (P_left + P_right). A chain of >= 2 counts when nothing goes round over its whole
    span either (a slowly circulating centre under a ripple of small episodes is no libration: 138537 13N-25 at 100 kyr,
    48 episodes of ~1 deg whose centre goes round several times), and both outer episodes reach their record edge
    (`covers_edges`, or `_union_edge` within libration_coverage_periods). Without the orbit-averaged raw angle
    (nan) nothing shows the angle stayed: no join. The first such chain is returned (`UnionCover`; empty when none)."""
    ordered = sorted(episodes, key=lambda episode: episode.t_start_yr)
    if len(ordered) < 2:
        return UnionCover()
    chains: List[List[Episode]] = [[ordered[0]]]
    worst: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for left, right in zip(ordered, ordered[1:]):
        tp_range = break_range(geometry, left, right)
        raw_range = geometry.gap_raw_range(left, right)
        gap_p = (right.t_start_yr - left.t_end_yr) / (left.period_med_yr + right.period_med_yr)
        if tp_range < TWO_PI and raw_range < TWO_PI and gap_p <= settings.libration_union_gap_periods:
            chains[-1].append(right)
            worst[-1] = (max(worst[-1][0], tp_range), max(worst[-1][1], raw_range), max(worst[-1][2], gap_p))
        else:
            chains.append([right])
            worst.append((0.0, 0.0, 0.0))
    for chain, (tp_range, raw_range, gap_p) in zip(chains, worst):
        if len(chain) < 2:
            continue
        span = (chain[0].t_start_yr, chain[-1].t_end_yr)
        span_tp, span_raw = tp_range_in(geometry, *span), raw_range_in(geometry, *span)
        if not (span_tp < TWO_PI and span_raw < TWO_PI):
            continue
        start = None if covers_edges(chain[0], settings)[0] else _union_edge(geometry, chain[0], 'start')
        end = None if covers_edges(chain[-1], settings)[1] else _union_edge(geometry, chain[-1], 'end')
        if all(short is None or short <= settings.libration_coverage_periods for short in (start, end)):
            shortfalls = [NAN if short is None else short for short in (start, end)]
            return UnionCover(tuple(chain), tp_range, raw_range, span_tp, span_raw, gap_p, *shortfalls)
    return UnionCover()


def tp_range_in(geometry: 'Geometry', start_yr: float, end_yr: float) -> float:
    """max - min of sigma_f's turning points inside [start_yr, end_yr] [rad]; 0 with fewer than two."""
    times = np.asarray(geometry.turning_points.times_yr, dtype=float)
    values = np.asarray(geometry.turning_points.values_rad, dtype=float)
    inside = values[(times >= start_yr) & (times <= end_yr)]
    return float(np.ptp(inside)) if len(inside) > 1 else 0.0


def raw_range_in(geometry: 'Geometry', start_yr: float, end_yr: float) -> float:
    """max - min of the orbit-averaged raw angle inside [start_yr, end_yr] [rad]; 0 without samples there, nan
    without the averaged angle."""
    if geometry.averaged_raw is None or geometry.averaged_raw_t_yr is None:
        return NAN
    t = geometry.averaged_raw_t_yr
    part = geometry.averaged_raw[(t >= start_yr) & (t <= end_yr)]
    return float(np.ptp(part)) if len(part) else 0.0


def _union_edge(geometry: 'Geometry', outer: Episode, edge: str) -> float:
    """The edge of a union chain that `covers_edges` does not cover ('start' or 'end'): how far sigma_f's turning
    points, accepted in a run or not, fall short of the analysis window's edge beyond the `outer` episode, in periods
    of that episode [P] (TK14 31M-34 at 0.5 yr: 0.15 after a lone TP 1.74 P past the last episode; 2003 TG2 4E-1M-3:
    12.6, a stall at a* without TPs). inf when the record shows the regime ending there: an observed exit (entry), or
    a turn of sigma_f's turning points or of the orbit-averaged raw angle between the episode and the edge."""
    if getattr(outer, 'exit_observed' if edge == 'end' else 'entry_observed', False):
        return math.inf
    w0, w1 = float(geometry.window[0]), float(geometry.window[1])
    lo, hi = (w0, outer.t_start_yr) if edge == 'start' else (outer.t_end_yr, w1)
    if not (tp_range_in(geometry, lo, hi) < TWO_PI and raw_range_in(geometry, lo, hi) < TWO_PI):
        return math.inf
    times = np.asarray(geometry.turning_points.times_yr, dtype=float)
    inside = times[(times >= lo) & (times <= hi)]
    if edge == 'start':
        reach = (inside.min() if len(inside) else hi) - w0
    else:
        reach = w1 - (inside.max() if len(inside) else lo)
    return float(reach / outer.period_med_yr) if outer.period_med_yr > 0 else math.inf


def break_range(geometry: 'Geometry', left: Episode, right: Episode) -> float:
    """max - min of sigma_f's turning points from the last one of `left` to the first one of `right` [rad] (the
    extrema of sigma_f between two episodes); 0 with fewer than two of them."""
    return tp_range_in(geometry, left.t_end_yr, right.t_start_yr)


def gap_circulates(geometry: 'Geometry', t_yr: np.ndarray, angle_f: np.ndarray, left: Episode, right: Episode) -> bool:
    """An observed circulation between two episodes (`left` ends before `right` starts): sigma_f, the angle the
    turning points are found on, covers 2pi or more from the last TP of `left` to the first TP of `right` (the range of
    its samples there: `break_range` read on the samples, so it holds before the last jump of a too), or, for an MMR
    angle, the orbit-averaged raw angle does (`Geometry.gap_raw_range`). A gap without a finite sigma_f counts as one
    (nothing shows the angle stayed)."""
    t = np.asarray(t_yr, dtype=float)
    part = np.asarray(angle_f, dtype=float)[
        int(np.searchsorted(t, left.t_end_yr, 'left')) : int(np.searchsorted(t, right.t_start_yr, 'right'))
    ]
    part = part[np.isfinite(part)]
    if not len(part):
        return True
    raw = geometry.gap_raw_range(left, right) if geometry is not None else NAN
    return bool(np.ptp(part) >= TWO_PI or (np.isfinite(raw) and raw >= TWO_PI))


def drift_fraction(runs: Sequence[Episode], window: Tuple[float, float], settings: AnalysisSettings) -> float:
    """Fraction of the window covered by rejected `centre_moves` runs with >= libration_min_turning_points TPs."""
    start, end = window
    if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
        return NAN
    covered = 0.0
    for run in runs:
        if run.reject_reason != RejectReason.CENTRE_MOVES or run.n_tp < settings.libration_min_turning_points:
            continue
        covered += max(0.0, min(run.t_end_yr, end) - max(run.t_start_yr, start))
    return float(covered / (end - start))


# --- slips, revolutions, stalls -----------------------------------------------------------------


def _first_crossing(t: np.ndarray, x: np.ndarray, i0: int, i1: int, level: float, direction: float) -> float:
    """First time in [i0, i1] where direction * (x - level) >= 0, linearly interpolated."""
    piece = direction * (x[i0 : i1 + 1] - level)
    above = np.flatnonzero(piece >= 0)
    if len(above) == 0:
        return NAN
    j = int(above[0])
    if j == 0:
        return float(t[i0])
    return _crossing_time(t, x, i0 + j - 1, level)


def _last_crossing(t: np.ndarray, x: np.ndarray, i0: int, i1: int, level: float, direction: float) -> float:
    """Last time in [i0, i1] where direction * (x - level) goes from < 0 to >= 0 (or i0 when never below)."""
    piece = direction * (x[i0 : i1 + 1] - level)
    below = np.flatnonzero(piece < 0)
    if len(below) == 0:
        return float(t[i0])
    j = int(below[-1])
    if j == len(piece) - 1:
        return NAN
    return _crossing_time(t, x, i0 + j, level)


def find_slips(
    t_yr: np.ndarray,
    angle_f: np.ndarray,
    tp: TurningPoints,
    small_period_yr: float,
    suspect_times_yr: Sequence[float],
    settings: AnalysisSettings,
) -> Tuple[List[Slip], int]:
    """Slips: midpoint steps of 2pi (1 +- libration_slip_tolerance) accumulated over
    >= libration_slip_min_duration_t0 * T0 [yr]. Returns (slips, number rejected as too short).
    Steps at `suspect_times_yr` (unwrap suspects, jumps) are marked `suspect`.

    A slip sits on a half-cycle TP_i -> TP_i+1 with half-amplitude >= pi; its size is
    m_i+1 - m_i-1 (the midpoints on either side), accepted within the tolerance of a whole number
    of turns: a modulated pendulum often goes over the top twice before it is caught again, and
    such a step is one slip of 2 rev (an unwrap error inserts exactly one turn). Its duration runs from the last crossing of the
    old level m_i-1 to the first crossing of the new one m_i+1 by sigma_f; with T0 unknown (nan) no
    slip is rejected as short. The time of a slip is the crossing of the top (old level + size/2).

    A step against the phase of the oscillation (an unwrap error while sigma rises, a 2pi drop) adds
    a TP pair instead, one on each side of the step: the midpoints then move m_i-1 -> m_i+3
    monotonically over three half-cycles, none of half-amplitude pi. There the step is the middle
    half-cycle TP_i+1 -> TP_i+2, and its duration is the time between these two TPs.
    """
    t = np.asarray(t_yr, dtype=float)
    x = np.asarray(angle_f, dtype=float)
    values = np.asarray(tp.values_rad, dtype=float)
    mids = midpoint_values(values)
    amps = half_amplitude_values(values)
    suspects = np.asarray([s for s in suspect_times_yr if np.isfinite(s)], dtype=float)
    tolerance = settings.libration_slip_tolerance

    def whole_turns(size: float) -> bool:
        turns = round(size / TWO_PI)
        return turns != 0 and abs(size / TWO_PI - turns) <= tolerance

    # (first half-cycle, number of half-cycles): over the top, then against the phase
    steps = [(i, 1) for i in range(1, len(amps) - 1) if amps[i] >= np.pi and whole_turns(mids[i + 1] - mids[i - 1])]
    used = {i for i, _ in steps}
    for i in range(1, len(amps) - 3):
        span = range(i, i + 3)
        path = np.diff(mids[i - 1 : i + 4])
        if any(j in used for j in span) or np.max(amps[i : i + 3]) >= np.pi or not whole_turns(mids[i + 3] - mids[i - 1]):
            continue
        if np.all(np.sign(path) == np.sign(mids[i + 3] - mids[i - 1])):
            steps.append((i, 3))
            used.update(span)

    slips: List[Slip] = []
    rejected = 0
    for i, n_half in sorted(steps):
        size = float(mids[i + n_half] - mids[i - 1])
        direction = np.sign(size)
        if n_half == 1:
            i0, i1 = int(tp.indices[i]), int(tp.indices[i + 1])
            start = _last_crossing(t, x, i0, i1, mids[i - 1], direction)
            end = _first_crossing(t, x, i0, i1, mids[i + 1], direction)
            duration = float(end - start)
        else:
            i0, i1 = int(tp.indices[i + 1]), int(tp.indices[i + 2])
            duration = float(t[i1] - t[i0])
        if np.isfinite(small_period_yr) and np.isfinite(duration):
            if duration < settings.libration_slip_min_duration_t0 * small_period_yr:
                rejected += 1
                continue
        time = _first_crossing(t, x, i0, i1, mids[i - 1] + size / 2, direction)
        suspect = bool(np.any((suspects >= t[i0]) & (suspects <= t[i1])))
        slips.append(Slip(time, size / TWO_PI, duration, suspect))
    return slips, rejected


def revolutions(t_yr: np.ndarray, angle_f: np.ndarray, rate_f: np.ndarray, window: Tuple[float, float]) -> List[Revolution]:
    """Full turns of sigma_f (each 2pi of travel) with the phase and relative depth of the rate minimum.

    Turn k runs from the first time sigma_f reaches sigma_0 + 2pi (k - 1) to the first time it
    reaches sigma_0 + 2pi k (in the direction of the net motion); the mean rate is 2pi / duration.
    """
    t = np.asarray(t_yr, dtype=float)
    x = np.asarray(angle_f, dtype=float)
    rate = np.asarray(rate_f, dtype=float)
    inside = np.flatnonzero(window_mask(t, window))
    if len(inside) < 2:
        return []
    i0, i1 = int(inside[0]), int(inside[-1])
    direction = np.sign(x[i1] - x[i0])
    n_turns = int(np.floor(abs(x[i1] - x[i0]) / TWO_PI))
    if direction == 0 or n_turns == 0:
        return []
    progress = direction * (x[i0 : i1 + 1] - x[i0])
    reached = np.maximum.accumulate(progress)
    # `reached` never decreases: the first sample at or beyond each whole turn by bisection (O(n log n))
    marks = TWO_PI * np.arange(1, n_turns + 1)
    edges = np.concatenate([[i0], i0 + np.searchsorted(reached, marks, side='left')])
    starts, ends = edges[:-1], edges[1:]
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]
    speed = direction * rate[starts[0] : ends[-1] + 1]
    first = starts - starts[0]  # the turns tile the samples: turn k owns [start_k, end_k), its end is the next start
    lowest = np.minimum(np.minimum.reduceat(speed, first), speed[ends - starts[0]])
    # the first sample of each turn at its lowest speed (argmin): in the turn's own samples, else its end sample
    hits = np.flatnonzero(speed == np.repeat(lowest, np.diff(np.append(first, len(speed)))))
    owner = np.searchsorted(first, hits, side='right') - 1
    turns_hit, at = np.unique(owner, return_index=True)
    slowest = ends - starts[0]
    slowest[turns_hit] = hits[at]
    slowest += starts[0]
    mean_speed = TWO_PI / (t[ends] - t[starts])
    phases = x[slowest] % TWO_PI
    depth = direction * rate[slowest] / mean_speed
    return [
        Revolution(t_start_yr=float(t[a]), t_end_yr=float(t[b]), phase_min_rate_rad=float(phase), rate_min_over_mean=float(ratio))
        for a, b, phase, ratio in zip(starts, ends, phases, depth)
    ]


def pattern_b(
    revs: Sequence[Revolution],
    top_rad: float,
    h_hi: float,
    sigma_h: float,
    settings: AnalysisSettings,
) -> PatternB:
    """Pattern B: every turn reachable by the separatrix (h_rev <= h_hi + separatrix_sigma_margin * sigma_h)
    and the rate minima of >= libration_stall_rev_fraction turns within libration_stall_phase_tol_deg of
    the potential top `top_rad`. `rate_inconsistent` compares r_min with `pendulum.rate_min_over_mean(h)`.

    Unknown energies (nan h or h_hi) leave `in_zone` False: pattern B is then not detected.
    """
    n = len(revs)
    if n == 0:
        return PatternB(0, NAN, NAN, NAN, NAN, NAN, NAN, NAN, False, False, False)
    phases = np.array([rev.phase_min_rate_rad for rev in revs], dtype=float)
    r_min = np.array([rev.rate_min_over_mean for rev in revs], dtype=float)
    h = np.array([rev.energy_h for rev in revs], dtype=float)
    vector = np.mean(np.exp(1j * phases))
    if np.isfinite(top_rad):
        at_top = np.abs(wrap_pi(phases - top_rad)) <= np.radians(settings.libration_stall_phase_tol_deg)
        stall_fraction = float(np.mean(at_top))
    else:
        stall_fraction = NAN
    margin = settings.separatrix_sigma_margin * (sigma_h if np.isfinite(sigma_h) else 0.0)
    in_zone = bool(np.all(np.isfinite(h)) and np.isfinite(h_hi) and np.all(h <= h_hi + margin))
    rotating = np.isfinite(h) & (h > 1)
    r_pend = np.atleast_1d(pendulum.rate_min_over_mean(h[rotating]))
    r_pend_med = float(np.median(r_pend)) if len(r_pend) else NAN
    r_min_med = float(np.median(r_min))
    inconsistent = bool(np.isfinite(r_pend_med) and abs(r_min_med - r_pend_med) > settings.libration_stall_rate_tol)
    detected = bool(in_zone and np.isfinite(stall_fraction) and stall_fraction >= settings.libration_stall_rev_fraction)
    return PatternB(
        n_revolutions=n,
        stall_fraction=stall_fraction,
        stall_phase_mean_rad=float(np.angle(vector) % TWO_PI),
        stall_phase_r=float(np.abs(vector)),
        rate_min_over_mean_med=r_min_med,
        rate_min_over_mean_pend_med=r_pend_med,
        h_rev_med=float(np.nanmedian(h)) if np.isfinite(h).any() else NAN,
        h_rev_max=float(np.nanmax(h)) if np.isfinite(h).any() else NAN,
        in_zone=in_zone,
        detected=detected,
        rate_inconsistent=inconsistent,
    )


def arc_or_stall(
    t_yr: np.ndarray,
    angle_f: np.ndarray,
    slow_f: np.ndarray,
    identity_coefficient: float,
    window: Tuple[float, float],
) -> ArcStall:
    """No TPs: arc if ptp(sigma_f) >= pi else stall; strength bounds K from range, trend, curvature and
    rate; 'range' decides: K |sin| <= d_slow / tau, P_lib,min = 2 pi sqrt(tau |sin| / (|c| d_slow)) [yr]
    (`identity_coefficient` c in rad yr^-1 per slow unit).

    |sin| is |sin(sigma - sigma_c)| at its least over the window with sigma_c in {0, pi} (NUMBERS §5.8):
    the least |sin| gives the largest K and the shortest (most cautious) P_lib,min. Each bound is the
    measured K |sin| divided by that |sin|: range d_slow / tau, trend |slow slope|, curvature
    |sigma_ddot| / |c|, rate max |slow_dot|.
    """
    t = np.asarray(t_yr, dtype=float)
    inside = window_mask(t, window)
    tw = t[inside]
    x = np.asarray(angle_f, dtype=float)[inside]
    slow = np.asarray(slow_f, dtype=float)[inside]
    if len(tw) < 3:
        return ArcStall('stall', NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN)
    tau = float(tw[-1] - tw[0])
    range_rad = float(np.ptp(x))
    rate = float((x[-1] - x[0]) / tau)
    acc = float(2 * np.polyfit(tw - tw[0], x, 2)[0])
    mean_direction = float(np.angle(np.mean(np.exp(1j * x))))
    around = mean_direction + wrap_pi(x - mean_direction)
    wrapped_min, wrapped_max = float(np.min(around) % TWO_PI), float(np.max(around) % TWO_PI)
    slow_ptp = float(np.ptp(slow))
    slow_trend = float(np.polyfit(tw - tw[0], slow, 1)[0])
    slow_rate_max = float(np.max(np.abs(time_derivative(tw, slow))))
    sin_min = np.min(np.abs(np.sin(x)))  # numpy floats: a zero |sin| gives an infinite bound, not an error
    c = np.abs(np.float64(identity_coefficient))
    with np.errstate(divide='ignore', invalid='ignore'):
        bound_range = slow_ptp / (tau * sin_min)
        bound_trend = abs(slow_trend) / sin_min
        bound_curvature = abs(acc) / (c * sin_min)
        bound_rate = slow_rate_max / sin_min
        period_min = TWO_PI / np.sqrt(c * bound_range)
    return ArcStall(
        kind='arc' if range_rad >= np.pi else 'stall',
        tau_yr=tau,
        range_rad=range_rad,
        rate_rad_yr=rate,
        acc_rad_yr2=acc,
        wrapped_min_rad=wrapped_min,
        wrapped_max_rad=wrapped_max,
        slow_ptp=slow_ptp,
        slow_trend_per_yr=slow_trend,
        force_bound_range=float(bound_range),
        force_bound_trend=float(bound_trend),
        force_bound_curvature=float(bound_curvature),
        force_bound_rate=float(bound_rate),
        period_min_yr=float(period_min),
    )


# --- edges, jumps, last regime -----------------------------------------------------------------


def _raw_monotonic(t: np.ndarray, raw: np.ndarray, mask: np.ndarray, block_yr: float, direction: float) -> bool:
    """Means of the raw angle in consecutive blocks of `block_yr` move strictly in `direction`."""
    tt, xx = t[mask], raw[mask]
    if len(tt) < 2 or not np.isfinite(block_yr) or block_yr <= 0:
        return False
    blocks = np.floor((tt - tt[0]) / block_yr).astype(int)  # non-decreasing: each block is a run of samples
    starts = np.flatnonzero(np.diff(blocks, prepend=-1))
    means = np.add.reduceat(xx, starts) / np.diff(np.append(starts, len(xx)))
    if len(means) < 2:
        return bool(direction * (xx[-1] - xx[0]) > 0)
    return bool(np.all(direction * np.diff(means) > 0))


def orbit_averaged(t_yr: np.ndarray, raw: np.ndarray, period_yr: float) -> np.ndarray:
    """The raw angle averaged over one orbital period `period_yr` (a centred running mean, shorter at the record's
    ends) [rad]; the raw samples themselves when the step is not below that period or the period is unknown. It keeps
    every slow motion of the angle up to the record's ends, where the low-pass bends it (2017 UM52 passes its planet
    51 yr before the end, less than the 55.5-yr cutoff), and drops the fast wiggle of one orbit."""
    t, x = np.asarray(t_yr, dtype=float), np.asarray(raw, dtype=float)
    if len(t) < 2 or not (np.isfinite(period_yr) and period_yr > 0):
        return x
    step = float(np.median(np.diff(t)))
    half = int(round(period_yr / step / 2)) if step > 0 else 0
    if half < 1:
        return x
    total = np.concatenate([[0.0], np.cumsum(x)])
    index = np.arange(len(x))
    lo, hi = np.maximum(index - half, 0), np.minimum(index + half + 1, len(x))
    return (total[hi] - total[lo]) / (hi - lo)


def _visible_edge(
    t: np.ndarray,
    angle_f: np.ndarray,
    raw: np.ndarray,
    episode: Episode,
    kind: str,
    record: Tuple[float, float],
    settings,
    averaged: Optional[np.ndarray] = None,
) -> Optional[EdgeEvent]:
    """`visible_edge` on the record `record`. With `averaged` (the orbit-averaged raw angle, `orbit_averaged`; MMR
    only) the edge rule of variant (a) applies (author 2026-10-19; DECISIONS 2026-10-15 "an arc > 2pi is not a piece of
    libration"): on the stretch from the episode's last turning point to the record's end (entry: from the record's
    start to its first one) that angle travels >= libration_edge_arc_rev turns away from the turning point and moves
    monotonically there (means over half-period blocks, `_raw_monotonic`): the exit (entry) is observed, not suspected,
    however little record is left. Arc, direction and monotonicity are read on that one stretch of the averaged angle,
    whether or not sigma_f crosses |sigma_f - c_ep| = pi there (the low-pass may hide the crossing at the record's
    edge; Codex W20 A #10); the event's time is then the averaged angle's own crossing."""
    centre = episode.centre_mean_rad
    period = episode.period_med_yr
    if not np.isfinite(centre):
        return None
    block = period / 2 if np.isfinite(period) else NAN
    in_record = (t >= record[0]) & (t <= record[1])
    if kind == 'exit':
        piece = np.flatnonzero(in_record & (t > episode.t_end_yr))
    else:
        piece = np.flatnonzero(in_record & (t < episode.t_start_yr))
    raw_arc, observed, arc_monotonic, arc_crossing = NAN, False, False, None
    if averaged is not None:
        inside = np.flatnonzero(in_record)
        if kind == 'exit':
            turn, edge = int(np.searchsorted(t, episode.t_end_yr)), int(inside[-1])
        else:
            turn, edge = int(np.searchsorted(t, episode.t_start_yr)), int(inside[0])
        turn = min(turn, len(t) - 1)
        travel = float(averaged[edge] - averaged[turn])
        raw_arc = abs(travel)
        stretch = np.zeros(len(t), dtype=bool)
        stretch[min(turn, edge) : max(turn, edge) + 1] = True
        heading = float(np.sign(travel)) if kind == 'exit' else -float(np.sign(travel))  # in time
        arc_monotonic = bool(travel != 0 and _raw_monotonic(t, averaged, stretch & in_record, block, heading))
        observed = bool(raw_arc >= TWO_PI * settings.libration_edge_arc_rev and arc_monotonic)
        beyond_avg = piece[np.abs(averaged[piece] - centre) > np.pi]
        if observed and len(beyond_avg):
            arc_crossing = int(beyond_avg[0] if kind == 'exit' else beyond_avg[-1])
    deviation = np.abs(angle_f - centre)
    beyond = piece[deviation[piece] > np.pi]
    if len(beyond):
        crossing = int(beyond[0] if kind == 'exit' else beyond[-1])
        beyond_rad = float(np.max(deviation[piece]) - np.pi)
    elif arc_crossing is not None:
        crossing = arc_crossing
        beyond_rad = float(np.max(np.abs(averaged[piece] - centre)) - np.pi)
    else:
        return None
    remaining = record[1] - t[crossing] if kind == 'exit' else t[crossing] - record[0]
    last_piece = in_record & ((t >= t[crossing]) if kind == 'exit' else (t <= t[crossing]))
    remaining_periods = float(remaining / period) if np.isfinite(period) and period > 0 else NAN
    suspected = bool(np.isfinite(remaining_periods) and remaining_periods < settings.libration_coverage_periods)
    if averaged is not None:
        monotonic = arc_monotonic
    else:
        away = float(np.sign(angle_f[crossing] - centre))  # the side the angle leaves the centre to
        direction = away if kind == 'exit' else -away  # entering: the raw angle moves towards the centre
        monotonic = _raw_monotonic(t, raw, last_piece, block, direction)
    return EdgeEvent(kind, float(t[crossing]), beyond_rad, remaining_periods, suspected and not observed, monotonic, raw_arc, observed)


def visible_edge(
    t_yr: np.ndarray,
    angle_f: np.ndarray,
    raw_angle: np.ndarray,
    episode: Episode,
    kind: str,
    settings: AnalysisSettings,
) -> Optional[EdgeEvent]:
    """Exit ('exit', after the last TP) or entry ('entry', before the first): first time |sigma_f - c_ep| > pi;
    suspected when the remaining record is < libration_coverage_periods episode periods. None if none.

    c_ep is the mean midpoint of the episode. The search runs to the true record edge. `raw_monotonic`:
    the raw angle averaged over half-period blocks moves monotonically away from the centre after an
    exit (towards it before an entry).
    """
    t = np.asarray(t_yr, dtype=float)
    return _visible_edge(
        t, np.asarray(angle_f, dtype=float), np.asarray(raw_angle, dtype=float), episode, kind, (float(t[0]), float(t[-1])), settings
    )


def jump_baseline_yr(cutoff_yr: float, step_yr: float, settings: AnalysisSettings) -> float:
    """tau = max(libration_jump_tau_cutoffs * P_c, libration_jump_tau_steps * dt) [yr]."""
    return float(max(settings.libration_jump_tau_cutoffs * cutoff_yr, settings.libration_jump_tau_steps * step_yr))


def _increment_stretches(increments: np.ndarray, edges: np.ndarray, gap: int, threshold: float) -> List[Tuple[int, int]]:
    """The stretches of `jump_candidates`: runs above the threshold (`edges`, pairs of start/stop indices) less than
    `gap` samples apart form a group; a group is one stretch, or its dominant run alone when that run's largest step
    exceeds every other run's by more than the threshold and by more than that run's own step: what stands out is
    larger than the motion around it, as the motion is larger than the noise for any run. A smoothly growing and
    decaying libration has neighbouring swings of nearly its largest one (Codex W16: amplitude 0.01 -> 2.5 rad and back
    within 1 kyr at a 700-yr period, runs 0.0298 and 0.0281 au against a threshold of 0.001 au), a switch to
    circulation stands out by more than the swing (W6: 0.0239 against 0.0009 au; W10: 0.114 against 0.039 au)."""

    def peak(piece: Tuple[int, int]) -> float:
        return float(np.max(np.abs(increments[piece[0] : piece[1]])))

    groups: List[List[Tuple[int, int]]] = []
    for start, stop in zip(edges[::2], edges[1::2]):
        if groups and start - groups[-1][-1][1] < gap:
            groups[-1].append((int(start), int(stop)))
        else:
            groups.append([(int(start), int(stop))])
    stretches = []
    for group in groups:
        peaks = sorted((peak(piece) for piece in group), reverse=True)
        if len(group) > 1 and peaks[0] - peaks[1] > max(threshold, peaks[1]):
            stretches.append(max(group, key=peak))  # a step standing out above the motion around it
        else:
            stretches.append((group[0][0], group[-1][1]))
    return stretches


def jump_candidates(
    t_yr: np.ndarray, slow_f: np.ndarray, tau_yr: float, reference_period_yr: float, settings: AnalysisSettings
) -> List[JumpCandidate]:
    """|slow_f(t + tau/2) - slow_f(t - tau/2)| above libration_jump_noise_factor robust std (1.4826 MAD),
    accumulated faster than `reference_period_yr` (the callers pass the libration cycle period P_cyc,
    `Geometry.period_cycle_yr`; the separatrix zone falls back to the median turn of a circulation): a kick, not an
    adiabatic drift.

    Each stretch above the threshold is one candidate at its largest |step|. A sharp step stays above the threshold
    for ~tau, so its duration is the stretch length minus tau (at least one step); with no finite reference period
    the duration does not filter. Stretches less than tau apart are one group, since an increment over tau couples
    every pair of points within tau (noise near the threshold would otherwise cut a slow drift into one-step pieces
    at its ends). A group is one stretch from its first to its last sample, unless one of its stretches has a largest
    step above every other one's by more than the threshold and more than that other step (`_increment_stretches`):
    then that stretch alone is the candidate, with its own duration, and the rest is the motion around it (W6
    synthetic record: a switch to circulation 50 times the libration's own swing of a, and W10: 3 times a shuffled
    libration's swing, merged at tau = 2 P_c = 150 yr with those swings into one slow 'drift' and were lost). A piece
    at the end of a drift lies below the drift's own largest step, so it still merges. Steps of similar size closer
    than 2 tau stay one group (the 40 reversals 250 yr apart of the W11 synthetic record, TEST-DISPUTES W16).
    """
    t = np.asarray(t_yr, dtype=float)
    slow = np.asarray(slow_f, dtype=float)
    if len(t) < 3:
        return []
    step = float(np.median(np.diff(t)))
    half = max(1, int(round(tau_yr / (2 * step))))
    if len(t) <= 2 * half:
        return []
    increments = slow[2 * half :] - slow[: -2 * half]
    centres = np.arange(half, len(t) - half)
    noise = pendulum.MAD_TO_SIGMA * float(np.median(np.abs(increments - np.median(increments))))
    above = np.abs(increments) > settings.libration_jump_noise_factor * noise
    candidates: List[JumpCandidate] = []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(int), [0]])))
    threshold = settings.libration_jump_noise_factor * noise

    def duration_of(start: int, stop: int) -> float:
        length = float(t[centres[stop - 1]] - t[centres[start]] + step)
        return max(length - 2 * half * step, step)

    for start, stop in _increment_stretches(increments, edges, 2 * half, threshold):
        j = start + int(np.argmax(np.abs(increments[start:stop])))
        duration = duration_of(start, stop)
        if np.isfinite(reference_period_yr) and duration >= reference_period_yr:
            continue
        candidates.append(JumpCandidate(float(t[centres[j]]), float(increments[j]), float(duration)))
    return candidates


def jump_threshold_dh(sigma_h: float, settings: AnalysisSettings) -> float:
    """The energy change that makes a jump: max(2 sigma_h, libration_jump_dh_min) (dh_min alone without sigma_h)."""
    return max(2 * sigma_h, settings.libration_jump_dh_min) if np.isfinite(sigma_h) else settings.libration_jump_dh_min


def confirm_jumps(
    candidates: Sequence[JumpCandidate],
    cycle_times_yr: np.ndarray,
    cycle_h: np.ndarray,
    sigma_h: float,
    half_width: float,
    settings: AnalysisSettings,
) -> List[Jump]:
    """A jump when the energy of libration_jump_cycles cycles on each side differs by
    > max(2 sigma_h, libration_jump_dh_min); with fewer cycles on a side: |size| > `half_width`
    (by_slow_only). A step without an energy change is a `kick`.

    A candidate that cannot be decided (too few cycles and no finite half-width) is kept as a kick.
    """
    times = np.asarray(cycle_times_yr, dtype=float)
    h = np.asarray(cycle_h, dtype=float)
    finite = np.isfinite(times) & np.isfinite(h)
    times, h = times[finite], h[finite]
    n = settings.libration_jump_cycles
    threshold = jump_threshold_dh(sigma_h, settings)
    jumps = []
    for candidate in candidates:
        before = h[times < candidate.time_yr][-n:] if n > 0 else np.zeros(0)
        after = h[times > candidate.time_yr][:n] if n > 0 else np.zeros(0)
        if len(before) >= n and len(after) >= n and n > 0:
            dh = float(np.mean(after) - np.mean(before))
            kick = not abs(dh) > threshold
            jumps.append(Jump(candidate.time_yr, candidate.size, candidate.duration_yr, dh, False, kick))
        else:
            kick = not (np.isfinite(half_width) and abs(candidate.size) > half_width)
            jumps.append(Jump(candidate.time_yr, candidate.size, candidate.duration_yr, NAN, True, kick))
    return jumps


def jump_edge_margin_yr(tau_yr: float, reference_period_yr: float, settings: AnalysisSettings) -> float:
    """Data a jump candidate needs on each side inside the window to be confirmed [yr]: max(tau,
    libration_jump_edge_periods x the reference period). The energy test compares cycles before and after, and the
    a-step itself is an increment over tau: a candidate closer to an edge is only the edge of the filtered series
    (31935 9M-16 at 2 P_c: 'jumps' 20 yr after the start and 20 yr before the end cut the record to [100000, 99980])."""
    period = reference_period_yr if np.isfinite(reference_period_yr) else 0.0
    return float(max(tau_yr, settings.libration_jump_edge_periods * period))


def _window_jumps(
    candidates: Sequence[JumpCandidate],
    window: Tuple[float, float],
    margin_yr: float,
    cycle_times_yr: np.ndarray,
    cycle_h: np.ndarray,
    sigma_h: float,
    half_width: float,
    settings: AnalysisSettings,
) -> List[Jump]:
    """The candidates inside `window`, in time order: confirmed (`confirm_jumps`) when they have `margin_yr` of the
    window on each side, else kept as `at_edge` kicks (they cut nothing)."""
    inside = [c for c in candidates if window[0] <= c.time_yr <= window[1]]
    confirmable = [c for c in inside if c.time_yr - window[0] >= margin_yr and window[1] - c.time_yr >= margin_yr]
    jumps = confirm_jumps(confirmable, cycle_times_yr, cycle_h, sigma_h, half_width, settings)
    jumps += [Jump(c.time_yr, c.size, c.duration_yr, NAN, False, True, at_edge=True) for c in inside if c not in confirmable]
    return sorted(jumps, key=lambda jump: jump.time_yr)


def _median_full_period(times: np.ndarray) -> float:
    periods = np.asarray(times[2:], dtype=float) - np.asarray(times[:-2], dtype=float)
    return float(np.median(periods)) if len(periods) else NAN


def _best_split(amps: np.ndarray, min_side: int) -> Optional[Tuple[int, float]]:
    """(k, separation) of the split of the half-amplitudes `amps` with the largest band separation: the sides are
    amps[:k] and amps[k + 1:] (amps[k], the transitional half-cycle, belongs to neither), each >= `min_side` long; the
    separation is min(larger side) / max(smaller side), the side whose median is larger being the larger. None without a
    split."""
    best = None
    for k in range(min_side, len(amps) - min_side):
        left, right = amps[:k], amps[k + 1 :]
        big, small = (left, right) if np.median(left) >= np.median(right) else (right, left)
        separation = float(np.min(big) / np.max(small)) if np.max(small) > 0 else np.inf
        if best is None or separation > best[1]:
            best = (k, separation)
    return best


def _step_energy(
    times: np.ndarray, k: int, cycle_spans: np.ndarray, cycle_h: np.ndarray, sigma_h: float, settings: AnalysisSettings
) -> Tuple[int, int, float, float, float, float]:
    """(cycles before, cycles after, dh, threshold, median h before, median h after) of a G2a split at TP_k / TP_k+1:
    the cycles of the energy analysis (`cycle_spans` [start, end] per cycle, `cycle_h`) lying inside each side; dh =
    mean h of the libration_jump_cycles nearest the step after minus before (as `confirm_jumps`), nan without a cycle
    on a side (so is that side's median)."""
    spans = np.asarray(cycle_spans, dtype=float).reshape(-1, 2)
    h = np.asarray(cycle_h, dtype=float)
    finite = np.all(np.isfinite(spans), axis=1) & np.isfinite(h)
    spans, h = spans[finite], h[finite]
    order = np.argsort(spans[:, 0], kind='stable')
    spans, h = spans[order], h[order]
    before = h[(spans[:, 0] >= times[0]) & (spans[:, 1] <= times[k])]
    after = h[(spans[:, 0] >= times[k + 1]) & (spans[:, 1] <= times[-1])]
    threshold = jump_threshold_dh(sigma_h, settings)
    n = max(settings.libration_jump_cycles, 1)
    medians = tuple(float(np.median(side)) if len(side) else NAN for side in (before, after))
    if not (len(before) and len(after)):
        return len(before), len(after), NAN, threshold, *medians
    return len(before), len(after), float(np.mean(after[:n]) - np.mean(before[-n:])), threshold, *medians


# the sub-tests of G2a test 7 in their order (`RegimePeriods.failed`, with the cutoff factor where one failed)
T7_ISOCHRONOUS, T7_SHAPE, T7_COVERED, T7_LOOP, T7_LAW = 'isochronous', 'shape', 'covered', 'no_loop', 'period_law'
T7_TURNING_POINTS, T7_ENERGY_DRIFT, T7_WELLS = 'turning_points', 'energy_drift', 'wells_unstable'


@dataclass(frozen=True)
class CycleFit:
    """One disjoint cycle (TP_i, TP_i+2) of a G2a side in the potential V of the larger swings (`_cycle_fit`)."""

    t_start_yr: float
    t_end_yr: float
    energy: float  # <sigma_dot^2 / 2 + V(sigma)> over the cycle [rad^2/yr^2]
    law: float  # T_obs / T_pred of the loop of that energy in the well that holds the cycle (inf: no loop)
    tp_mismatch: float  # largest |observed turning point - the loop's| over the observed half-swing (inf: no loop)


@dataclass(frozen=True)
class RegimePeriods:
    """G2a test 7 (`regime_periods`): both regimes are librations of one potential, each at its own energy. The
    numbers are those at the middle cutoff of libration_action_step_cutoffs (x the shorter side period); `failed`
    names the first sub-test that failed, with the cutoff factor where it did ('' passes)."""

    cutoff_yr: float
    r2: float
    wells: int
    covered: bool  # the smaller swings lie inside the sigma range the larger ones sampled
    law_before: float  # the worst cycle of each side (largest |T_obs / T_pred - 1|)
    law_after: float
    failed: str = ''
    tp_mismatch: float = NAN  # the worst cycle of both sides
    drift_before: float = NAN  # spread of each side's cycle energies over the step of the energy between the sides
    drift_after: float = NAN
    period_ratio: float = NAN  # median T_obs of the smaller swings over that of the larger ones (isochrony null)
    cycles_before: int = 0  # disjoint cycles of each side
    cycles_after: int = 0
    held_out: bool = False  # the larger side's laws out of sample (V of its other cycles); False: one cycle, in sample
    laws: str = ''  # the worst law of each side at each cutoff: 'factor:before/after;...'
    vf_r: float = NAN  # correlation of V_E and V_F (from a) over the larger swings' range (reported, not a test)
    vf_law_small: float = NAN  # the worst law of the smaller side in V_F (reported)

    def passes(self, settings: AnalysisSettings) -> bool:
        return not self.failed and math.isfinite(self.cutoff_yr)


def _disjoint_cycles(times: np.ndarray, k: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """The disjoint full cycles (TP_i, TP_i+2) of each side of a split at TP_k / TP_k+1, from the step outward."""
    before = [(float(times[i]), float(times[i + 2])) for i in range(k - 2, -1, -2)][::-1]
    after = [(float(times[i]), float(times[i + 2])) for i in range(k + 1, len(times) - 2, 2)]
    return before, after


def _cycle_fit(t: np.ndarray, sigma: np.ndarray, rate: np.ndarray, potential: pendulum.Potential, span: Tuple[float, float]) -> CycleFit:
    """The cycle `span` in `potential`: its energy E = <sigma_dot^2 / 2 + V(sigma)> (the window constant of V_E), the
    loop of E around the minimum whose basin holds its fastest sample (`pendulum.well_holding`, `loop_through`), its
    period law and how far the observed extremes of sigma lie from the loop's turning points. No loop (E below that
    well's bottom, or the orbit would circulate): law and mismatch inf."""
    mask = (t >= span[0]) & (t <= span[1])
    if np.count_nonzero(mask) < 3:
        return CycleFit(span[0], span[1], NAN, np.inf, np.inf)
    x, v = sigma[mask], rate[mask]
    energy = float(np.mean(v**2 / 2 + potential.value(x)))
    found = pendulum.extrema(potential)
    well = pendulum.well_holding(potential, float(x[np.argmax(v**2)]), found)
    loop = pendulum.loop_through(potential, energy, well, found)
    if not (np.isfinite(loop.period_yr) and loop.period_yr > 0):
        return CycleFit(span[0], span[1], energy, np.inf, np.inf)
    lo, hi = float(np.min(x)), float(np.max(x))  # sigma is unwrapped and the loop lies about its fastest sample
    half = (hi - lo) / 2
    mismatch = max(abs(lo - loop.lower_rad), abs(hi - loop.upper_rad)) / half if half > 0 else np.inf
    return CycleFit(span[0], span[1], energy, float((span[1] - span[0]) / loop.period_yr), float(mismatch))


def _worst_law(fits: Sequence[CycleFit]) -> float:
    return max((fit.law for fit in fits), key=lambda law: abs(law - 1) if np.isfinite(law) else np.inf) if fits else NAN


def _drift(fits: Sequence[CycleFit], step: float) -> float:
    energies = [fit.energy for fit in fits if np.isfinite(fit.energy)]
    if len(energies) < 2:
        return 0.0 if energies else NAN
    return float((max(energies) - min(energies)) / step) if step > 0 else np.inf


@dataclass(frozen=True)
class _ShapeAtCutoff:
    """Test 7 at one cutoff: V of the larger swings, the cycle fits of both sides, the first failed sub-test."""

    cutoff_yr: float
    potential: Optional[pendulum.Potential]
    covered: bool
    before: Tuple[CycleFit, ...]
    after: Tuple[CycleFit, ...]
    drift_before: float
    drift_after: float
    failed: str
    vf_r: float = NAN
    vf_law_small: float = NAN


def _vf_report(t, sigma, angle, cutoff, big, small, potential, settings) -> Tuple[float, float]:
    """V_F (c a_dot = -V'(sigma)) on the larger side against V_E: their correlation over the sampled range and the
    worst law of the smaller side in V_F. Reported, not tested (`slow_f` carries the record's own cutoff)."""
    slow = lowpass_or_none(t, np.asarray(angle.slow_f, dtype=float), cutoff, settings)
    if slow is None or not np.isfinite(angle.identity_coefficient):
        return NAN, NAN
    mask = (t >= big[0][0]) & (t <= big[-1][1])
    v_f = pendulum.fit_potential_from_force(sigma[mask], time_derivative(t, slow)[mask], angle.identity_coefficient, potential.n_harmonics)
    if v_f is None:
        return NAN, NAN
    grid = np.linspace(float(np.min(sigma[mask])), float(np.max(sigma[mask])), 181)
    r = float(np.corrcoef(potential.value(grid), v_f.value(grid))[0, 1])
    rate = time_derivative(t, sigma)
    return r, _worst_law([_cycle_fit(t, sigma, rate, v_f, span) for span in small])


def _shape_at_cutoff(times, k, down, angle, cutoff, settings) -> _ShapeAtCutoff:
    """Test 7 at the shape cutoff `cutoff` [yr]: V fitted as V_E on the disjoint cycles of the larger swings; each of
    their cycles is read in V of the others (held out; in sample with one cycle), each cycle of the smaller swings in V
    of all. Sub-tests in order: V explains >= separatrix_ve_r2_min (shape), covers the smaller swings (covered), every
    cycle has a loop (no_loop), its period law within libration_action_step_period_tol (period_law), its observed
    extremes at the loop's turning points within libration_action_step_tp_tol of its half-swing (turning_points), and
    each side's cycle energies spread by <= libration_action_step_energy_drift of the step between them
    (energy_drift)."""
    t = np.asarray(angle.t_yr, dtype=float)
    sigma = lowpass_or_none(t, np.asarray(angle.angle_raw, dtype=float), cutoff, settings) if np.isfinite(cutoff) else None
    before, after = _disjoint_cycles(times, k)
    big, small = (before, after) if down else (after, before)
    empty = _ShapeAtCutoff(cutoff, None, False, (), (), NAN, NAN, T7_SHAPE)
    if sigma is None or not big or not small:
        return empty
    rate = time_derivative(t, sigma)
    potential, _ = pendulum.fit_potential_from_rate(t, sigma, rate, big, settings, with_windows=False)
    if potential is None or not potential.r2 >= settings.separatrix_ve_r2_min:
        return replace(empty, potential=potential)
    sampled = sigma[(t >= big[0][0]) & (t <= big[-1][1])]
    held = sigma[(t >= small[0][0]) & (t <= small[-1][1])]
    lo, hi = float(np.min(sampled)), float(np.max(sampled))
    middle = (lo + hi) / 2
    held = middle + wrap_pi(held - middle)
    covered = bool(len(held) and np.min(held) >= lo and np.max(held) <= hi)

    def big_fit(j):
        others = [span for i, span in enumerate(big) if i != j]
        fitted = pendulum.fit_potential_from_rate(t, sigma, rate, others, settings, with_windows=False)[0] if others else potential
        if fitted is None or not fitted.r2 >= settings.separatrix_ve_r2_min:
            return CycleFit(big[j][0], big[j][1], NAN, np.inf, np.inf)
        return replace(_cycle_fit(t, sigma, rate, fitted, big[j]), energy=_cycle_fit(t, sigma, rate, potential, big[j]).energy)

    big_fits = tuple(big_fit(j) for j in range(len(big)))
    small_fits = tuple(_cycle_fit(t, sigma, rate, potential, span) for span in small)
    step = abs(float(np.median([f.energy for f in small_fits])) - float(np.median([f.energy for f in big_fits])))
    fits = big_fits + small_fits
    drifts = (_drift(big_fits, step), _drift(small_fits, step))
    tol = settings.libration_action_step_period_tol
    tests = (
        (T7_COVERED, covered),
        (T7_LOOP, all(np.isfinite(f.law) for f in fits)),
        (T7_LAW, all(abs(f.law - 1) <= tol for f in fits)),
        (T7_TURNING_POINTS, all(f.tp_mismatch <= settings.libration_action_step_tp_tol for f in fits)),
        (T7_ENERGY_DRIFT, all(d <= settings.libration_action_step_energy_drift for d in drifts)),
    )
    failed = next((name for name, passed in tests if not passed), '')
    vf_r, vf_law = _vf_report(t, sigma, angle, cutoff, big, small, potential, settings)
    side_before, side_after = (big_fits, small_fits) if down else (small_fits, big_fits)
    drift_before, drift_after = drifts if down else drifts[::-1]
    return _ShapeAtCutoff(cutoff, potential, covered, side_before, side_after, drift_before, drift_after, failed, vf_r, vf_law)


def regime_periods(times: np.ndarray, k: int, down: bool, angle: FilteredAngle, settings: AnalysisSettings) -> RegimePeriods:
    """Test 7 of G2a: both regimes librate in one potential, each at its own energy (Q1b, author 2026-10-27; made
    robust after the W24 review).

    Between impulses a conservative 1-degree-of-freedom libration keeps its energy; an impulse moves the orbit to
    another loop of the same V, possibly in another of its wells (coexisting resonant modes: Efimov & Sidorenko 2020,
    Eq. 4). What makes the two regimes one potential and not a forced wave whose amplitude steps:
      - isochronous: the median observed periods of the two sides differ by more than libration_action_step_period_tol.
        Otherwise the null of one forcing frequency explains both and the period law cannot tell (in a nearly harmonic
        V every loop has one period);
      - at every cutoff C = f x the shorter side period, f in libration_action_step_cutoffs (a factor 2, lighter than
        the energy analysis' separatrix_cutoff_cycles: the turning points and a second well of a swing across a barrier
        live in its high harmonics, a well or a law that one cutoff alone gives is the filter's), the per-cycle
        sub-tests of `_shape_at_cutoff` on disjoint cycles, every cycle (no median);
      - wells_unstable: V has the same number of wells at every cutoff.
    One disjoint cycle on the side of the larger swings reads its law in sample (`held_out` False): the cut is
    provisional. 2003 TX9 3E+2J-10 (wells at 133 and 272 deg) is the case this was built for."""
    shorter = min(_median_full_period(times[: k + 1]), _median_full_period(times[k + 1 :]))
    factors = sorted(settings.libration_action_step_cutoffs)
    middle = factors[len(factors) // 2]  # the cutoff whose numbers are reported
    central = float(middle * shorter) if np.isfinite(shorter) else NAN
    before, after = _disjoint_cycles(times, k)
    big, small = (before, after) if down else (after, before)
    small_period = float(np.median([b - a for a, b in small])) if small else NAN
    big_period = float(np.median([b - a for a, b in big])) if big else NAN
    period_ratio = small_period / big_period if big_period > 0 else NAN
    base = RegimePeriods(
        central,
        NAN,
        0,
        False,
        NAN,
        NAN,
        failed=T7_SHAPE,
        period_ratio=period_ratio,
        cycles_before=len(before),
        cycles_after=len(after),
        held_out=len(big) >= 2,
    )
    if not (np.isfinite(shorter) and big and small):
        return base
    shapes = {f: _shape_at_cutoff(times, k, down, angle, f * shorter, settings) for f in factors}
    centre = shapes[middle]
    wells = {pendulum.well_count(s.potential) for s in shapes.values()}
    failed = '' if abs(period_ratio - 1) > settings.libration_action_step_period_tol else T7_ISOCHRONOUS
    if not failed:
        failed = next((f'{s.failed}@{f:g}' for f, s in shapes.items() if s.failed), '')
    if not failed and len(wells) > 1:
        failed = T7_WELLS
    laws = ';'.join(f'{f:g}:{_worst_law(s.before):.3f}/{_worst_law(s.after):.3f}' for f, s in shapes.items())
    mismatches = [fit.tp_mismatch for fit in centre.before + centre.after]
    return replace(
        base,
        r2=float(centre.potential.r2) if centre.potential is not None else NAN,
        wells=pendulum.well_count(centre.potential),
        covered=centre.covered,
        law_before=_worst_law(centre.before),
        law_after=_worst_law(centre.after),
        failed=failed,
        tp_mismatch=float(max(mismatches)) if mismatches else NAN,
        drift_before=centre.drift_before,
        drift_after=centre.drift_after,
        laws=laws,
        vf_r=centre.vf_r,
        vf_law_small=centre.vf_law_small,
    )


def _shape_fields(periods: RegimePeriods) -> dict:
    """The ActionStep fields of test 7 beyond the laws."""
    return dict(
        shape_failed=periods.failed,
        shape_tp_mismatch=periods.tp_mismatch,
        shape_drift_before=periods.drift_before,
        shape_drift_after=periods.drift_after,
        shape_period_ratio=periods.period_ratio,
        shape_cycles_before=periods.cycles_before,
        shape_cycles_after=periods.cycles_after,
        shape_held_out=periods.held_out,
        shape_laws=periods.laws,
        shape_vf_r=periods.vf_r,
        shape_vf_law_small=periods.vf_law_small,
    )


def action_step_of(
    run: Episode,
    angle: FilteredAngle,
    cycle_spans: np.ndarray,
    cycle_h: np.ndarray,
    sigma_h: float,
    settings: AnalysisSettings,
) -> Optional[ActionStep]:
    """G2a on one run (`ActionStep`): its best split (`_best_split`) and the tests, in order, the first failure named
    in `failed`:
      1 >= libration_action_step_min_half_cycles half-cycles on each side (else no split: None);
      2 separation >= libration_action_step_separation (an adiabatic decay or a slow modulation has none);
      3 ratio of the side medians >= libration_action_step_ratio;
      4 the swing of a, std of `angle.slow_f` on each side, larger on the larger side by >= libration_action_step_a_ratio
        (delta a ~ sigma_dot: a step of sigma alone is no step of the libration, 74729 1J+6S-1);
      5 the transitional half-cycle lasts <= libration_action_step_max_periods of the shorter side period;
      6 >= libration_action_step_cycles cycles of the energy analysis on each side (energy_cycles), and the energy
        changes the way the amplitude does by more than max(2 sigma_h, libration_jump_dh_min) (energy: beats of a
        constant energy do not);
      7 both regimes librate in one potential (regime_periods, `regime_periods`: the sub-test that failed in
        `shape_failed`): their periods differ (not one forcing frequency), and at every cutoff of
        libration_action_step_cutoffs V of the larger swings explains them, covers the smaller swings and holds every
        disjoint cycle of both sides in a loop of its own energy, with its period and turning points, each side's
        energy staying put. The cycle energies of test 6 come from the record's V_E, whose smoothing may hide a second
        well; this test measures the physics of each regime instead (measured only when 1-6 pass).
    Provisional when a side has fewer than libration_jump_cycles energy cycles, or the larger side one disjoint cycle
    (test 7 read its law in sample)."""
    times = np.asarray(run.tp_times_yr, dtype=float)
    amps = half_amplitude_values(run.tp_values_rad)
    split = _best_split(amps, settings.libration_action_step_min_half_cycles)
    if split is None:
        return None
    k, separation = split
    left, right = amps[:k], amps[k + 1 :]
    down = bool(np.median(left) >= np.median(right))
    big, small = (left, right) if down else (right, left)
    ratio = float(np.median(big) / np.median(small)) if np.median(small) > 0 else np.inf
    t, slow = np.asarray(angle.t_yr, dtype=float), np.asarray(angle.slow_f, dtype=float)
    swing_before = float(np.std(slow[(t >= times[0]) & (t <= times[k])]))
    swing_after = float(np.std(slow[(t >= times[k + 1]) & (t <= times[-1])]))
    big_swing, small_swing = (swing_before, swing_after) if down else (swing_after, swing_before)
    a_ratio = big_swing / small_swing if small_swing > 0 else np.inf
    shorter = min(_median_full_period(times[: k + 1]), _median_full_period(times[k + 1 :]))
    transition = float((times[k + 1] - times[k]) / shorter) if shorter > 0 else NAN
    n_before, n_after, dh, threshold, h_before, h_after = _step_energy(times, k, cycle_spans, cycle_h, sigma_h, settings)
    energy_sign = -1.0 if down else 1.0
    tests = (
        (STEP_SEPARATION, separation >= settings.libration_action_step_separation),
        (STEP_RATIO, ratio >= settings.libration_action_step_ratio),
        (STEP_A_SWING, a_ratio >= settings.libration_action_step_a_ratio),
        (STEP_TRANSITION, transition <= settings.libration_action_step_max_periods),
        (STEP_ENERGY_CYCLES, min(n_before, n_after) >= settings.libration_action_step_cycles),
        (STEP_ENERGY, bool(np.isfinite(dh) and energy_sign * dh > threshold)),
    )
    failed = next((name for name, passed in tests if not passed), '')
    periods = None
    if not failed:
        periods = regime_periods(times, k, down, angle, settings)
        failed = '' if periods.passes(settings) else STEP_REGIME_PERIODS
    return ActionStep(
        run_t_start_yr=run.t_start_yr,
        run_t_end_yr=run.t_end_yr,
        t_from_yr=float(times[k]),
        t_to_yr=float(times[k + 1]),
        n_before=len(left),
        n_after=len(right),
        amp_before_rad=float(np.median(left)),
        amp_after_rad=float(np.median(right)),
        separation=separation,
        ratio=ratio,
        a_ratio=float(a_ratio),
        transition_periods=transition,
        cycles_before=n_before,
        cycles_after=n_after,
        dh=dh,
        dh_threshold=threshold,
        failed=failed,
        provisional=not failed and (min(n_before, n_after) < settings.libration_jump_cycles or not periods.held_out),
        h_before=h_before,
        h_after=h_after,
        shape_cutoff_yr=periods.cutoff_yr if periods else NAN,
        shape_r2=periods.r2 if periods else NAN,
        shape_wells=periods.wells if periods else 0,
        law_before=periods.law_before if periods else NAN,
        law_after=periods.law_after if periods else NAN,
        **(_shape_fields(periods) if periods else {}),
    )


def action_steps(
    geometry: Geometry,
    angle: FilteredAngle,
    jumps: Sequence[Jump],
    cycle_spans: Optional[np.ndarray],
    cycle_h: np.ndarray,
    sigma_h: float,
    settings: AnalysisSettings,
) -> List[ActionStep]:
    """G2a on every accepted run in the sigma frame (a run accepted in psi = sigma - c(t) has its swings measured
    about a moving centre) that no confirmed jump of a (`jumps`) cuts already; nothing for a 1:1 angle (its regime
    changes have their own rules), without cycle energies, or with libration_action_step off."""
    if not settings.libration_action_step or angle.coorbital or cycle_spans is None:
        return []
    cuts = [jump.time_yr for jump in jumps if not jump.kick]
    steps = []
    for run in geometry.runs:
        if not run.accepted or run.frame != FRAME_SIGMA or any(run.t_start_yr <= cut <= run.t_end_yr for cut in cuts):
            continue
        step = action_step_of(run, angle, cycle_spans, cycle_h, sigma_h, settings)
        if step is not None:
            steps.append(step)
    return steps


def _jump_flags(jumps: Sequence[Jump]) -> List[str]:
    """kick (a step without an energy change), jump_at_edge (one too close to an edge to confirm), jump_by_a_only."""
    flags = []
    if any(jump.kick and not jump.at_edge for jump in jumps):
        flags.append(str(Flag.KICK))
    if any(jump.at_edge for jump in jumps):
        flags.append(str(Flag.JUMP_AT_EDGE))
    if any(jump.by_slow_only and not jump.kick for jump in jumps):
        flags.append(str(Flag.JUMP_BY_A_ONLY))
    return flags


def last_regime_start(t_yr: np.ndarray, jumps: Sequence[Jump], tau_yr: float) -> float:
    """Start of the analysis window: the end of the last non-kick jump's zone (`Jump.zone`: jump + tau/2 for a jump
    of a, TP_k+1 for an action step), else t0 [yr]."""
    t = np.asarray(t_yr, dtype=float)
    ends = [jump.zone(tau_yr)[1] for jump in jumps if not jump.kick]
    if not ends:
        return float(t[0])
    return float(min(max(ends), t[-1]))


def wiggle(t_yr: np.ndarray, angle_f: np.ndarray, centre_rad: np.ndarray, window: Tuple[float, float]) -> Tuple[float, float]:
    """(sqrt(2) std(sigma_f - c) [rad], dominant FFT period of sigma_f - c [yr])."""
    t = np.asarray(t_yr, dtype=float)
    inside = window_mask(t, window) & np.isfinite(centre_rad)
    if inside.sum() < 4:
        return NAN, NAN
    tw = t[inside]
    d = np.asarray(angle_f, dtype=float)[inside] - np.asarray(centre_rad, dtype=float)[inside]
    amplitude = float(np.sqrt(2) * np.std(d))
    slope, intercept = linear_fit(tw, d)
    y = (d - (intercept + slope * (tw - tw[0]))) * np.hanning(len(d))
    step = float(np.median(np.diff(tw)))
    freqs = np.fft.rfftfreq(len(y), step)[1:]
    power = np.abs(np.fft.rfft(y))[1:] ** 2
    if len(freqs) == 0 or not np.any(power > 0):
        return amplitude, NAN
    return amplitude, float(1 / freqs[int(np.argmax(power))])


# --- whole-record measurement -------------------------------------------------------------------


@dataclass(frozen=True)
class FilteredAngle:
    """What the geometry reads: an unwrapped angle and its filtered versions on one time grid.

    Built by the type-specific preprocessing (MMR: `analysis.mmr.preprocess`). `angle_f_half` and
    `angle_f_double` are sigma_f at cutoffs x0.5 and x2 (L1 and the ripple sign). `slow_f` is the
    filtered slow variable that answers the angle (a [au] for an MMR) and `identity_coefficient`
    the c of sigma_dot = c (slow - slow*) [rad yr^-1 per slow unit].
    """

    t_yr: np.ndarray
    angle_raw: np.ndarray  # unwrapped, unfiltered [rad]
    angle_f: np.ndarray  # filtered [rad]
    rate_f: np.ndarray  # d angle_f / dt [rad/yr]
    angle_f_half: np.ndarray
    angle_f_double: np.ndarray
    slow_f: np.ndarray
    identity_coefficient: float
    cutoff_yr: float
    step_yr: float
    window: Tuple[float, float]  # edge-trimmed window [yr]
    # the shortest libration period of the resonance (an MMR: `mmr.preprocess.libration_period_floor`); faster
    # 'episodes' are forced waves. Carried to Geometry for the decision; nan = the caller's default
    period_floor_yr: float = NAN
    period_floor_setting: str = ''
    # a well inside which period_floor_yr holds, outside it another floor (`FloorWell`; MMR: the 1:1 quasi-satellite
    # pass, rule G1q). None: period_floor_yr for every episode
    floor_well: Optional[FloorWell] = None
    # the caller's track of the well a run librates in (`WellModel`; MMR: `mmr.centre_track`): a run whose centre moves
    # is tested again in psi = sigma - c(t) (`_classify_run`). None: sigma only (the scale of the centre, secular, ZLK)
    well_model: Optional[WellModel] = None
    # Rules set only by the MMR preprocessing (`mmr.preprocess.filtered_angle`); secular and ZLK keep them off, so their
    # geometry does not depend on them. `orbit_period_yr`: the body's orbital period, over which the raw angle is
    # averaged for the edge rule (`orbit_averaged`)
    orbit_period_yr: float = NAN
    edge_arc_rule: bool = False  # an exit (entry) whose raw arc beyond the episode is >= one turn is observed
    return_closure: bool = False  # a 2-TP run closed by its phase return is one cycle (`phase_return`)
    split_regimes: bool = False  # a run whose centre moves is cut at a centre step or a stall (`split_at_regime_changes`)
    coorbital: bool = False  # a 1:1 angle: the status reads its union of regimes as a libration (`mmr.decide._g2`)
    cycle_travel: bool = False  # an episode's centre travel is read on its own cycles (`episode_centre_travel`)


def _count_at_cutoff(
    angle: FilteredAngle,
    series: np.ndarray,
    factor: float,
    prominence: float,
    window: Tuple[float, float],
    settings,
    local: Sequence[LocalThreshold] = (),
) -> Optional[int]:
    """TPs of the angle filtered at factor x P_c inside `window` (the x1 window), narrowed to that
    version's own edge trim when it is wider: the counts at x0.5 / x1 / x2 then differ only by the
    smoothing, never by a wider window or another threshold (`local`: the x1 stretches' own prominences; L1, ripple
    sign)."""
    series = np.asarray(series, dtype=float)
    if not np.all(np.isfinite(series)):
        return None
    t = np.asarray(angle.t_yr, dtype=float)
    own = edge_window(t, settings.filter_edge_trim_cutoffs * factor * angle.cutoff_yr)
    if np.isfinite(own[0]):
        window = (max(window[0], own[0]), min(window[1], own[1]))
    return len(find_turning_points(t, series, prominence, window, local=local))


def _pooled_periods(runs: Sequence[Episode]) -> np.ndarray:
    """Full periods of every run with at least three TPs, pooled [yr]."""
    pooled = [run.tp_times_yr[2:] - run.tp_times_yr[:-2] for run in runs if run.n_tp >= 3]
    return np.concatenate(pooled).astype(float) if pooled else np.zeros(0)


def _bands(values: np.ndarray) -> Tuple[float, float, float, float, bool]:
    """(centre first, half first, centre last, half last, overlap) of the first and last full cycle."""
    if len(values) < 3:
        return NAN, NAN, NAN, NAN, False
    mids, amps = midpoint_values(values), half_amplitude_values(values)
    first_c, first_h = float(np.mean(mids[:2])), float(np.mean(amps[:2]))
    last_c, last_h = float(np.mean(mids[-2:])), float(np.mean(amps[-2:]))
    return first_c, first_h, last_c, last_h, bool(abs(last_c - first_c) < first_h + last_h)


def _rate_numbers(angle: FilteredAngle, window: Tuple[float, float]) -> Tuple[float, float, float]:
    """(fraction of time against the mean rate sign, rate modulation by cos(sigma - phi), corr(rate, slow))."""
    inside = window_mask(angle.t_yr, window)
    if inside.sum() < 4:
        return NAN, NAN, NAN
    rate = np.asarray(angle.rate_f, dtype=float)[inside]
    sigma = np.asarray(angle.angle_f, dtype=float)[inside]
    slow = np.asarray(angle.slow_f, dtype=float)[inside]
    mean_rate = float(np.mean(rate))
    sign = np.sign(mean_rate) if mean_rate != 0 else 1.0
    negative = float(np.mean(sign * rate < 0))
    design = np.column_stack([np.ones_like(sigma), np.cos(sigma), np.sin(sigma)])
    coef = np.linalg.lstsq(design, rate, rcond=None)[0]
    modulation = float(np.hypot(coef[1], coef[2]) / abs(coef[0])) if coef[0] != 0 else NAN
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = float(np.corrcoef(rate, slow)[0, 1]) if np.std(rate) > 0 and np.std(slow) > 0 else NAN
    return negative, modulation, corr


def _measure(
    angle: FilteredAngle,
    prominence_rad: float,
    settings: AnalysisSettings,
    window: Tuple[float, float],
    record_start: float,
    first_id: int = 0,
    local_prominence: Optional[LocalProminence] = None,
    thresholds: Sequence[LocalThreshold] = (),
) -> Geometry:
    t = np.asarray(angle.t_yr, dtype=float)
    record = (record_start, float(t[-1]))
    angle_f = np.asarray(angle.angle_f, dtype=float)
    if thresholds and local_prominence is None:
        tp = find_turning_points(t, angle_f, prominence_rad, window, local=thresholds)
    else:
        tp = turning_points_with_local(t, angle_f, prominence_rad, window, settings, local_prominence)
    runs = _find_episodes(tp, record, window, settings, first_id, angle.well_model, angle)
    accepted = [run for run in runs if run.accepted]
    periods = _pooled_periods(accepted)
    if len(periods) == 0:
        periods = _pooled_periods(runs)
    period = float(np.median(periods)) if len(periods) else NAN
    # the centre of a regime after a jump is low-passed on its own samples (`finish_geometry` stitches it to the record):
    # a low-pass over the whole record carries the motion before the jump into it, and a circulation before a capture
    # makes the centre of the libration after it run several turns (`_centres_before` for the regimes before)
    own = int(np.searchsorted(t, record_start)) if record_start > t[0] else 0
    centre = centre_track(t[own:], angle_f[own:], period, angle.cutoff_yr, prominence_rad, window, settings)
    inside = np.flatnonzero(window_mask(t, window))
    if len(inside) >= 2:
        net_f = float((angle_f[inside[-1]] - angle_f[inside[0]]) / TWO_PI)
        raw = np.asarray(angle.angle_raw, dtype=float)
        net_raw = float((raw[inside[-1]] - raw[inside[0]]) / TWO_PI)
    else:
        net_f = net_raw = NAN
    first_c, first_h, last_c, last_h, overlap = _bands(tp.values_rad)
    wiggle_amp, wiggle_period = wiggle(t[own:], angle_f[own:], centre.centre_rad, window)
    negative, modulation, corr = _rate_numbers(angle, window)
    geometry = Geometry(
        window=window,
        regime_start_yr=record_start,
        turning_points=tp,
        n_tp_cut_half=_count_at_cutoff(angle, angle.angle_f_half, 0.5, prominence_rad, window, settings, tp.local),
        n_tp_cut_double=_count_at_cutoff(angle, angle.angle_f_double, 2, prominence_rad, window, settings, tp.local),
        period_cycle_yr=period,
        period_min_yr=float(periods.min()) if len(periods) else NAN,
        period_max_yr=float(periods.max()) if len(periods) else NAN,
        centre=centre,
        net_rev_filtered=net_f,
        net_rev_raw=net_raw,
        band_first_rad=first_c,
        band_last_rad=last_c,
        bands_overlap=overlap,
        wiggle_half_amp_rad=wiggle_amp,
        wiggle_period_yr=wiggle_period,
        runs=runs,
        revolutions=revolutions(t, angle_f, angle.rate_f, window),
        band_first_half_rad=first_h,
        band_last_half_rad=last_h,
        drift_fraction=drift_fraction(runs, window, settings),
        sigmadot_negative_fraction=negative,
        rate_modulation=modulation,
        rate_slow_corr=corr,
        period_floor_yr=float(angle.period_floor_yr),
        period_floor_setting=angle.period_floor_setting,
        floor_well=angle.floor_well,
        coorbital=bool(angle.coorbital),
        cycle_travel=bool(angle.cycle_travel),
    )
    if len(tp) == 0 and len(inside) >= 3:
        geometry.arc_stall = arc_or_stall(t, angle_f, angle.slow_f, angle.identity_coefficient, window)
    averaged = _mark_edges(geometry, angle, accepted, record, settings)
    if averaged is not None:  # the union of episodes (`covering_union`) and the onset groups read it (`gap_raw_range`)
        geometry.averaged_raw, geometry.averaged_raw_t_yr = averaged, t
    return geometry


def _mark_edges(
    geometry: Geometry, angle: FilteredAngle, accepted: Sequence[Episode], record: Tuple[float, float], settings
) -> Optional[np.ndarray]:
    """Exit after the last of the `accepted` runs and entry before the first (`_visible_edge`), their observed marks on
    those runs and the flags exit/entry_suspected; returns the orbit-averaged raw angle when it was formed."""
    averaged = None
    if accepted:
        t, angle_f = np.asarray(angle.t_yr, dtype=float), np.asarray(angle.angle_f, dtype=float)
        raw = np.asarray(angle.angle_raw, dtype=float)
        averaged = orbit_averaged(t, raw, angle.orbit_period_yr) if (angle.edge_arc_rule or angle.coorbital) else None
        edge_averaged = averaged if angle.edge_arc_rule else None
        geometry.exit = _visible_edge(t, angle_f, raw, accepted[-1], 'exit', record, settings, edge_averaged)
        geometry.entry = _visible_edge(t, angle_f, raw, accepted[0], 'entry', record, settings, edge_averaged)
        accepted[-1].exit_observed = bool(geometry.exit is not None and geometry.exit.observed)
        accepted[0].entry_observed = bool(geometry.entry is not None and geometry.entry.observed)
    if geometry.exit is not None and geometry.exit.suspected:
        geometry.flags.append(str(Flag.EXIT_SUSPECTED))
    if geometry.entry is not None and geometry.entry.suspected:
        geometry.flags.append(str(Flag.ENTRY_SUSPECTED))
    return averaged


def _mark_record_entry(geometry: Geometry, angle: FilteredAngle, settings) -> None:
    """After a cut (the regimes measured anew): the entry rule (`_visible_edge`) on the record's first accepted run,
    which lies before the jump, on the whole record: `Geometry.record_entry`, the run's entry_observed mark and the flag
    entry_suspected. `_mark_edges` reads the last regime only, yet G1l calls the record a libration from its start
    through that run (`decide.before_jump_coverage`, `covers_edges`) and G1e may decide on it, so its entry is checked
    as it is without a cut (W24 review: 2003 TX9 lost its entry_suspected when G2a cut it)."""
    earlier = [run for run in geometry.runs if run.accepted and run.before_last_jump]
    if not earlier:
        return
    first = min(earlier, key=lambda run: run.t_start_yr)
    t = np.asarray(angle.t_yr, dtype=float)
    raw = np.asarray(angle.angle_raw, dtype=float)
    averaged = None
    if angle.edge_arc_rule:
        averaged = geometry.averaged_raw if geometry.averaged_raw is not None else orbit_averaged(t, raw, angle.orbit_period_yr)
    record = (float(t[0]), float(t[-1]))
    event = _visible_edge(t, np.asarray(angle.angle_f, dtype=float), raw, first, 'entry', record, settings, averaged)
    geometry.record_entry = event
    first.entry_observed = bool(event is not None and event.observed)
    if event is not None and event.suspected:
        geometry.flags.append(str(Flag.ENTRY_SUSPECTED))


def refresh_edges(geometry: Geometry, angle: FilteredAngle, settings) -> None:
    """The edge rule again on the episodes of the last regime once the forced waves are known (`Geometry.forced_waves`,
    set after the response is measured): the exit belongs to the last real episode, the entry to the first (W20 round
    C, review B #9: 145480 1N-4 at 1 Myr, whose real episode 0 is followed by the forced waves 1 and 2). The marks and
    flags the first pass left on the forced waves go; nothing changes without forced waves."""
    if not geometry.forced_waves:
        return
    geometry.forget_derived()
    for run in geometry.runs:
        if not run.before_last_jump:
            run.exit_observed = run.entry_observed = False
    stale = (str(Flag.EXIT_SUSPECTED), str(Flag.ENTRY_SUSPECTED))
    geometry.flags[:] = [flag for flag in geometry.flags if flag not in stale]
    geometry.exit = geometry.entry = None
    record = (float(geometry.regime_start_yr), float(np.asarray(angle.t_yr, dtype=float)[-1]))
    _mark_edges(geometry, angle, geometry.episodes, record, settings)
    if geometry.record_entry is not None and geometry.record_entry.suspected and str(Flag.ENTRY_SUSPECTED) not in geometry.flags:
        geometry.flags.append(str(Flag.ENTRY_SUSPECTED))


def measure_geometry(
    angle: FilteredAngle,
    prominence_rad: float,
    settings: AnalysisSettings,
    local_prominence: Optional[LocalProminence] = None,
    thresholds: Sequence[LocalThreshold] = (),
) -> Geometry:
    """Everything geometric that needs no energy: TPs at x0.5/x1/x2, P_cyc, centre and passes, net
    revolutions, bands, wiggle, runs/episodes, revolutions, arc or stall, exit and entry.

    The TPs are those of prominence >= `prominence_rad`; with `local_prominence` each run of provisional TPs is
    selected at its own prominence instead (`local_thresholds`), the record's elsewhere; with `thresholds` (and no
    `local_prominence`) the given stretches keep their prominences (another cutoff of the same angle measured at the
    x1 profile, as `_count_at_cutoff`). The centre keeps the record's. Slips, pattern B and jumps stay empty: they need
    T0 and cycle energies (`finish_geometry`).
    """
    t = np.asarray(angle.t_yr, dtype=float)
    return _measure(
        angle, prominence_rad, settings, tuple(angle.window), float(t[0]), local_prominence=local_prominence, thresholds=thresholds
    )


def centre_scale_angle(angle: FilteredAngle, period_cycle_yr: float, settings: AnalysisSettings) -> Optional[FilteredAngle]:
    """The angle filtered at the scale of its centre, W_c = libration_centre_cycles x P_cyc (the smoothing of the
    centre track), on the window trimmed by that scale; None without P_cyc, when W_c <= P_c, or when the record is too
    short for W_c.

    Its turning points (prominence `centre_prominence`) are the turns of the centre, so `measure_geometry` on it
    applies the episode rules to the centre: when the centre librates, the slow libration is the angle's and the fine
    turning points are waves on it (2011 KT19 7N-9r6: 3-4 kyr waves on a 43 kyr libration; 518 z1: 400 kyr waves on
    3.3 Myr). The shortest libration is the fine angle's; a caller that decides at this scale sets its own."""
    if not (np.isfinite(period_cycle_yr) and period_cycle_yr > 0):
        return None
    return angle_at_centre_window(angle, float(settings.libration_centre_cycles * period_cycle_yr), settings)


def centre_turns_window_yr(angle: FilteredAngle, centre_prominence_rad: float, settings: AnalysisSettings) -> float:
    """W_c from the turns of the centre [yr]: the turning points of the fine angle at the prominence of a centre turn
    (p_c, `centre_prominence`; a wave whose swing is below p_c is no turn of the centre) have the median full period
    P_t, and W_c = P_t / filter_min_libration_cutoffs, the longest cutoff that still measures a libration of period P_t
    (`mmr.preprocess.filter_guard`). NaN with fewer than three such turns.

    The second scale of G2c: the fine P_cyc mixes the waves and the slow turns when the waves show only at some slow
    extremes (pi + 2.1 sin(2 pi t / 40 kyr) + 0.05 sin(2 pi t / 3.5 kyr + 1.3): P_cyc 29.9 kyr, 3 P_cyc = 90 kyr above
    the libration itself); P_t is the slow period (40 kyr) and W_c = 20 kyr keeps it."""
    turns = find_turning_points(angle.t_yr, angle.angle_f, centre_prominence_rad, tuple(angle.window))
    periods = full_periods(turns)
    return float(np.median(periods) / settings.filter_min_libration_cutoffs) if len(periods) else NAN


def angle_at_centre_window(angle: FilteredAngle, window_yr: float, settings: AnalysisSettings) -> Optional[FilteredAngle]:
    """The angle filtered at the centre scale W_c = `window_yr` (`centre_scale_angle`), on the window trimmed by it;
    None when W_c is not finite or <= P_c, or when the record is too short for W_c."""
    if not (np.isfinite(window_yr) and window_yr > angle.cutoff_yr):
        return None
    t = angle.t_yr
    angle_f = lowpass_or_none(t, angle.angle_raw, window_yr, settings)
    window = edge_window(t, settings.filter_edge_trim_cutoffs * window_yr)
    if angle_f is None or not np.isfinite(window[0]):
        return None

    def filtered(series: np.ndarray, cutoff_yr: float) -> np.ndarray:
        result = lowpass_or_none(t, series, cutoff_yr, settings)
        return result if result is not None else np.full_like(angle_f, NAN)

    # the same angle at another cutoff: its rules and period floor carry over; the centre's scale has no moving well
    return replace(
        angle,
        angle_f=angle_f,
        rate_f=time_derivative(t, angle_f),
        angle_f_half=filtered(angle.angle_raw, window_yr / 2),
        angle_f_double=filtered(angle.angle_raw, 2 * window_yr),
        slow_f=filtered(angle.slow_f, window_yr),
        cutoff_yr=window_yr,
        window=window,
        well_model=None,
    )


def _with_energies(revs: Sequence[Revolution], cycle_times: np.ndarray, cycle_h: np.ndarray) -> List[Revolution]:
    """Attach to each turn the energy of the cycle whose time falls inside it (nearest to its middle; of equally near
    ones the first in `cycle_times`)."""
    if not revs:
        return list(revs)
    valid = np.flatnonzero(np.isfinite(cycle_times) & np.isfinite(cycle_h))
    order = valid[np.argsort(cycle_times[valid], kind='stable')]  # by time, then by position
    times = cycle_times[order]
    start = np.array([rev.t_start_yr for rev in revs])
    end = np.array([rev.t_end_yr for rev in revs])
    middle = (start + end) / 2
    lo, hi = np.searchsorted(times, start, side='left'), np.searchsorted(times, end, side='right')
    found = lo < hi
    right = np.clip(np.searchsorted(times, middle, side='left'), lo, np.maximum(hi - 1, lo))
    left = np.maximum(right - 1, lo)
    # the first entry of each candidate time: equal times are ordered by position
    right_first = np.searchsorted(times, times[np.minimum(right, len(times) - 1)], side='left') if len(times) else right
    left_first = np.searchsorted(times, times[np.minimum(left, len(times) - 1)], side='left') if len(times) else left
    out = list(revs)
    for k in np.flatnonzero(found):
        a, b = order[left_first[k]], order[right_first[k]]
        da, db = abs(cycle_times[a] - middle[k]), abs(cycle_times[b] - middle[k])
        j = a if da < db or (da == db and a < b) else b
        out[k] = replace(revs[k], energy_h=float(cycle_h[j]))
    return out


@dataclass(frozen=True)
class JumpRegime:
    """A stretch of the record between confirmed jumps of the slow variable: its own samples [start, end] (a jump is
    located within +-tau/2, so a regime starts at the jump before it + tau/2 and ends at its jump - tau/2; an action
    step, G2a, owns its transitional half-cycle instead: `Jump.zone`) and the turning-point prominence measured on them
    [rad]. Its turning points are those up to the next regime's start (the jump zone stays with the regime before it)."""

    start_yr: float
    end_yr: float
    next_start_yr: float
    prominence_rad: float


def _regimes(
    t: np.ndarray,
    cuts_yr: Sequence,
    tau_yr: float,
    record_prominence_rad: float,
    regime_prominence: Optional[LocalProminence],
) -> List[JumpRegime]:
    """Every regime of the record, the last one (after the last jump) included; `cuts_yr` sorted, each a jump time
    (its zone jump -+ tau/2) or a zone (start, end) of its own (`Jump.zone`: an action step's transitional
    half-cycle): a regime ends where a zone starts and the next starts where it ends. The prominence of a regime is
    `regime_prominence` of the mask of its own samples, the record's without that function, for a regime with fewer
    than two samples, or when it returns no finite value."""
    zones = [tuple(cut) if np.ndim(cut) else (float(cut) - tau_yr / 2, float(cut) + tau_yr / 2) for cut in cuts_yr]
    starts = [float(t[0])] + [float(zone[1]) for zone in zones]
    ends = [float(zone[0]) for zone in zones] + [float(t[-1])]
    regimes = []
    for k, (start, end) in enumerate(zip(starts, ends)):
        own = (t >= start) & (t <= end)
        prominence = regime_prominence(own) if regime_prominence is not None and np.count_nonzero(own) >= 2 else NAN
        regimes.append(
            JumpRegime(
                start_yr=start,
                end_yr=float(end),
                next_start_yr=starts[k + 1] if k + 1 < len(starts) else np.inf,
                prominence_rad=float(prominence) if np.isfinite(prominence) else float(record_prominence_rad),
            )
        )
    return regimes


def _centres_before(
    angle: FilteredAngle,
    regimes: Sequence[JumpRegime],
    earlier: Sequence[Episode],
    whole: Geometry,
    window0: Tuple[float, float],
    settings: AnalysisSettings,
) -> List[CentreTrack]:
    """The centres of the regimes before the last jump of the slow variable (`regimes`), each low-passed on its own
    samples only, its turns at the centre prominence of its own turning points. A low-pass over the whole window
    (W_c = several cycles) would carry the motion after a jump into the episodes before it, and a circulation after the
    jump would make their centre run a full turn (a libration of 3 cycles followed by a strong kick then reads as a
    moving centre); with two jumps the same holds for the regime between them. P_cyc of a regime is that of its runs
    (accepted first)."""
    t = np.asarray(angle.t_yr, dtype=float)
    angle_f = np.asarray(angle.angle_f, dtype=float)
    tracks = []
    for regime in regimes:
        runs = [run for run in earlier if regime.start_yr <= run.t_start_yr < regime.next_start_yr]
        periods = _pooled_periods([run for run in runs if run.accepted])
        if len(periods) == 0:
            periods = _pooled_periods(runs)
        period = float(np.median(periods)) if len(periods) else whole.period_cycle_yr
        inside = (t >= regime.start_yr) & (t <= regime.end_yr)
        if np.count_nonzero(inside) < 2:
            tracks.append(whole.centre)
            continue
        window = (max(window0[0], regime.start_yr), min(window0[1], regime.end_yr))
        tracks.append(centre_track(t[inside], angle_f[inside], period, angle.cutoff_yr, regime.prominence_rad, window, settings))
    return tracks


def _runs_before(
    angle: FilteredAngle,
    regimes: Sequence[JumpRegime],
    record: Tuple[float, float],
    window: Tuple[float, float],
    settings: AnalysisSettings,
    local_prominence: Optional[LocalProminence] = None,
) -> List[Episode]:
    """The runs of the regimes before the last one, marked `before_last_jump`: each regime's turning points at its own
    prominence, from its start to the next regime's start (excluded; the jump zone stays with the regime before it, as
    the jump is located only within +-tau/2), classified in time order with consecutive ids (`_find_episodes`).

    A regime's turning points are its own: the prominence covers what sigma_f holds faster than any libration
    (MMR: `analyse.regime_prominence`), and a stretch after the jumps where sigma circulates in a few decades would
    otherwise set it for the whole record (2003 UG22 10E-17+0+7, 40 kyr: 79 jumps of a after 28.9 kyr, a fast residual
    of 80 deg over the record against 26 deg before the jumps, a prominence of 322 deg that left no turning point of
    the libration at 0.2-26.7 kyr). Only the threshold is the regime's: the prominence of each extremum is measured on
    the whole series, as without jumps. No run crosses a confirmed jump + tau/2 (a confirmed jump changes the energy, so
    a cycle across it would join extrema of two regimes): the last regime's turning points are measured again there,
    and counting them before the jump too would turn one capture into a libration on both sides of it (2021 BA 1E-1:
    the horseshoe's third TP at 407 yr, 30 yr after the jump at 377 yr, stays before it). Inside a regime each run of
    provisional turning points has its own prominence too (`local_thresholds`), with `local_prominence`. A run's phase
    return (`phase_return`) is sought within its regime only; the coverage gaps keep the whole `record`."""
    t = np.asarray(angle.t_yr, dtype=float)
    angle_f = np.asarray(angle.angle_f, dtype=float)
    found = extrema(angle_f) if np.all(np.isfinite(angle_f)) else None
    runs: List[Episode] = []
    for regime in regimes:
        upto = float(np.nextafter(regime.next_start_yr, -np.inf))  # the next regime's start belongs to it
        selection = (max(window[0], regime.start_yr), min(window[1], upto))
        if not selection[1] >= selection[0]:
            continue
        tp = turning_points_with_local(t, angle_f, regime.prominence_rad, selection, settings, local_prominence, found)
        own = (max(float(t[0]), regime.start_yr), upto)
        for run in _find_episodes(tp, record, window, settings, len(runs), angle.well_model, angle, own):
            run.before_last_jump = True
            runs.append(run)
    return runs


def _stitched_centre(last: CentreTrack, before: Sequence[CentreTrack], whole: CentreTrack) -> CentreTrack:
    """The last regime's centre (`last`, low-passed on the regime's own samples) on the record's grid: every regime's own
    centre on its samples, the whole-record low-pass only in the jump zones between them (tau about each jump), where
    no regime has samples. The turns, passes and numbers stay the last regime's (turn indices shifted to the grid)."""
    t = np.asarray(whole.t_yr, dtype=float)
    centre = np.array(whole.centre_rad, dtype=float)
    for track in [*before, last]:
        if track is not whole and len(track.t_yr):
            start = int(np.searchsorted(t, track.t_yr[0]))
            centre[start : start + len(track.t_yr)] = track.centre_rad
    offset = int(np.searchsorted(t, last.t_yr[0])) if len(last.t_yr) else 0
    turns = replace(last.turns, indices=np.asarray(last.turns.indices, dtype=int) + offset)
    return replace(last, t_yr=t, centre_rad=centre, turns=turns)


def finish_geometry(
    geometry: Geometry,
    angle: FilteredAngle,
    small_period_yr: float,
    top_rad: float,
    h_hi: float,
    sigma_h: float,
    cycle_times_yr: np.ndarray,
    cycle_h: np.ndarray,
    half_width: float,
    reference_period_yr: float,
    suspect_times_yr: Sequence[float],
    settings: AnalysisSettings,
    regime_prominence: Optional[LocalProminence] = None,
    cycle_spans_yr: Optional[np.ndarray] = None,
) -> Geometry:
    """Add slips (needs T0), pattern B (needs the potential top and the zone h_hi) and confirmed jumps
    (needs cycle energies); set `regime_start_yr`, the window of the last regime and mark the runs
    before the last jump. Returns the same object, updated.

    After a jump that changes the energy, everything geometric is measured again on the last regime
    [jump + tau/2, t1 - trim] (rule 10); the earlier runs stay in `runs` marked `before_last_jump`, measured regime by
    regime between the jumps (`_runs_before`). Each regime's turning points are found at the prominence
    `regime_prominence` gives for the mask of its own samples (MMR: the fast residual measured there), the record's
    prominence without it or where it gives no finite value (a regime too short to measure it), and inside a regime
    each run of provisional turning points at its own (`local_thresholds`). A candidate
    with less than `jump_edge_margin_yr` of the window on either side cannot be confirmed: it stays in `jumps` as
    `at_edge` (flag jump_at_edge) and cuts nothing.
    `cycle_times_yr` are the mid-times of the cycles/turns whose energies are `cycle_h`.

    G2a (`action_steps`, given `cycle_spans_yr`, the [start, end] of those cycles): an accepted run whose libration
    action steps within one half-cycle is cut there by a confirmed jump (`ActionStep.jump`) whose zone is that
    half-cycle; the regimes, rule 10, G1e and G1l then read it as a jump of a. Flags action_step and, with fewer than
    libration_jump_cycles energy cycles on a side, action_step_provisional.
    """
    t = np.asarray(angle.t_yr, dtype=float)
    cycle_times = np.asarray(cycle_times_yr, dtype=float)
    energies = np.asarray(cycle_h, dtype=float)
    tau = jump_baseline_yr(angle.cutoff_yr, angle.step_yr, settings)
    reference = reference_period_yr if np.isfinite(reference_period_yr) else NAN
    window0 = tuple(angle.window)
    margin = jump_edge_margin_yr(tau, reference, settings)
    candidates = jump_candidates(t, angle.slow_f, tau, reference, settings)
    jumps = _window_jumps(candidates, window0, margin, cycle_times, energies, sigma_h, half_width, settings)
    steps = action_steps(geometry, angle, jumps, cycle_spans_yr, energies, sigma_h, settings)
    jumps = sorted(jumps + [step.jump() for step in steps if step.cuts], key=lambda jump: jump.time_yr)
    regime_start = last_regime_start(t, jumps, tau)
    if regime_start > t[0] and not min(window0[1], t[-1]) - max(window0[0], regime_start) > 0:
        regime_start = float(t[0])  # never an empty window (the margin keeps it from happening)

    if regime_start > t[0]:
        zones = sorted(jump.zone(tau) for jump in jumps if not jump.kick)
        regimes = _regimes(t, zones, tau, geometry.turning_points.prominence_rad, regime_prominence)
        *before, last = regimes
        regime_window = (max(window0[0], regime_start), window0[1])
        earlier = _runs_before(angle, before, (float(t[0]), float(t[-1])), window0, settings, regime_prominence)
        fresh = _measure(angle, last.prominence_rad, settings, regime_window, regime_start, len(earlier), regime_prominence)
        fresh.runs = earlier + fresh.runs
        fresh.record_centres = _centres_before(angle, before, earlier, geometry, window0, settings)
        fresh.centre = _stitched_centre(fresh.centre, fresh.record_centres, geometry.centre)
        geometry.__dict__.update(fresh.__dict__)
        _mark_record_entry(geometry, angle, settings)
    flags = list(geometry.flags)
    if regime_start > t[0]:
        flags.append(str(Flag.LAST_REGIME_AFTER_JUMP))

    geometry.jumps = jumps
    geometry.action_steps = steps
    if any(step.cuts for step in steps):
        flags.append(str(Flag.ACTION_STEP))
    if any(step.cuts and step.provisional for step in steps):
        flags.append(str(Flag.ACTION_STEP_PROVISIONAL))
    geometry.jump_tau_yr = tau
    geometry.jump_half_width = half_width
    geometry.jump_edge_margin_yr = margin
    flags += _jump_flags(jumps)
    # the pendulum's own swing over tau stays below the half-width only while tau < 2 / Omega = T0 / pi
    if np.isfinite(small_period_yr) and tau >= small_period_yr / np.pi:
        flags.append(str(Flag.JUMP_LOCATOR_ALIASED))

    suspects = list(suspect_times_yr) + [jump.time_yr for jump in jumps]
    geometry.slips, geometry.n_slips_rejected_short = find_slips(
        t, angle.angle_f, geometry.turning_points, small_period_yr, suspects, settings
    )
    if any(slip.suspect for slip in geometry.slips):
        flags.append(str(Flag.UNWRAP_STEP_SUSPECT))

    geometry.revolutions = _with_energies(geometry.revolutions, cycle_times, energies)
    geometry.pattern_b = pattern_b(geometry.revolutions, top_rad, h_hi, sigma_h, settings)
    if geometry.pattern_b.rate_inconsistent:
        flags.append(str(Flag.STALL_RATE_INCONSISTENT))
    geometry.flags = list(dict.fromkeys(flags))
    geometry.forget_derived()  # the runs before the last jump are marked now (and the regime may be measured anew)
    return geometry
