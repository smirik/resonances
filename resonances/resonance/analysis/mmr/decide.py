"""The MMR decision: angle_status from geometry, status from physics, origin, margins, checks (SPEC §5, §7.1).

angle_status (first rule that fires, SPEC §5.2):
  G0 unphysical orbit -> undefined/unphysical (never from `analyse_mmr`: an orbit outside the elliptic domain is refused
     before, invalid/unphysical by `MMRSeries.domain_problems`, so it passes False); unwrap failed -> undefined/alias
     (undefined_phase: jitter_low_e, the full angle over the residual limits and varpi itself undefined; jitter_low_e
     alone is a flag and a cap)
  G1 the window is the last regime after jumps of a (G2a, `geometry.ActionStep`: a step of the libration action inside
     one accepted run, not 1:1, whose regimes' periods follow one potential, is such a jump too, its zone the
     transitional half-cycle; a G2a cut that hides a check failed on the whole record, other than R0 pooled over both
     regimes, and changes the verdict is withheld, `analyse._action_step_guard`)
  G1e (after G3-G6) the last regime holds no episode but librating episodes precede the last jump: angle_status stays
     the last regime's, the status reads the angle as the episodic angle of those episodes (`status_basis`) without the
     last regime's response and energy (`regime_physics`), confidence capped (episode_before_jump)
  G1c after the last turning point the centre runs >= 2pi for longer than the coverage -> circulation (last regime)
  G1l (after G2) the last regime librates but no librating episode runs from the record's start to the jump (a
     capture): angle_status stays libration (the last regime's), the status reads the angle as the episodic one it is
  G2 an accepted episode: covers the record -> libration/<shape>, else episodic ('' | recurrent); an edge is covered
     within LIBRATION_COVERAGE_PERIODS periods of the record edge or with no turning point missed at the window edge
     (`geometry.covers_edges`), never when the raw angle shows the exit (entry) beyond it (the edge rule,
     `Episode.exit_observed`); librating episodes that cover the record together, no turn in any gap nor over the
     chain, sigma_f inside its neighbours' band through every gap (LIBRATION_UNION_BAND_MARGIN_DEG, whatever the
     gap's length: `geometry.gap_band_overshoot`), each record edge covered or reached by the
     turning points within LIBRATION_COVERAGE_PERIODS -> libration/regime_change (`_regimes_cover`; for an angle that
     is not 1:1, rule G2u: G2c is tried first and the status reads it as episodic, `promote_union`); an episode is
     accepted when it is not faster than the angle's
     shortest libration (`period_floor`) and its centre does not turn inside it (`librating_episodes`); a 3-TP episode
     has one centre estimate: L6 not_measured (thin); a 2-TP episode closed by its phase return caps medium; an
     accepted run that is a forced wave of another argument (its identity fails on a lattice line or with the wrong
     sign, `forced_wave_episodes`, W20 F6) is no episode at all (`Geometry.forced_waves`); nor is an unconfirmed
     ripple (W28 B6b, `ripple_waves`, MMR_RIPPLE_ON_ARC): half-amplitude below MMR_A_RESPONSE_MIN_AMP_DEG, identity
     measured and failed, on a record whose slow centre holds an unfinished swing, not covering the record, not 1:1
  G2c (analyse.py) no libration at P_c (nor an observed exit or entry), but the centre librates: the record is analysed
     at the scale of its centre W_c = LIBRATION_CENTRE_CYCLES x P_cyc and that verdict is kept when it is a libration
     (or one bounded by an observed exit or entry) (once more at
     P / FILTER_MIN_LIBRATION_CUTOFFS when W_c is too slow for the libration's period P)
  G3 no episode, sigma-only potential, pattern A or B -> near_separatrix/pattern_a|pattern_b (confidence <= medium
     while the zone is not measured: cap near_separatrix_without_zone)
  G4 largest monotone centre pass >= 2pi -> circulation ('' | alias)
  G5 centre_moves runs cover >= libration_drift_min_window_fraction -> drift; G5a: the slow centre turns back once
     (farther than W_c from the edges) and stays within one revolution -> insufficient_time/arc; G5w (W28 B6c,
     MMR_SLOW_SWING_IN_WELL): it turns back twice within one revolution and the half-swing between the turns is a
     swing in the resonance's own well (slow identity, the a swing and the period against the directly averaged
     pendulum at its amplitude within sqrt(MMR_STRENGTH_CHECK_RATIO), `slow_swing.py`) -> insufficient_time/arc, also
     without G5 (before the ripple and few_cycles reading of G6: the turning points are waves on the slow swing);
     extension MMR_EXTEND_SLOW_CYCLES slow periods (`baseline`)
  G6 TPs without episode or drift -> insufficient_time/few_cycles; no TPs -> arc (range >= pi) | stall; with
     unconfirmed ripples set aside only the TPs outside them count, and without any the slow centre decides: G5a's
     arc (one turn), else arc | stall by its range (`_g6_ripple`)
The verdict reports G1c, G1e and G1l as episodic over the record (`reported_angle`) and their last regime as
final_regime (`final_regime`); the rules above stay as they are.
After the tree, the record's edges (W28 C14, rule end_arc_extend, `record_edges`, MMR_RECORD_EDGE_STRETCH): a record
that ends, after >= one revolution of sigma_f, in a stretch without another one longer than MMR_END_ARC_SLOWDOWN_FACTOR
x the pass's longest revolution and (doubled) MMR_END_ARC_MIN_FLOOR_PERIODS floors, with no centre turn and no
librating episode in it, takes final_regime arc | stall and the flag extend_time forward; a non_resonant circulation
there the cap record_ends_unresolved (<= medium). The start mirror, one reflection cut by either edge, a clean 1:1
edge leg (G4c's extend) and an edge half-libration (G4h) give the flag only. No status changes.
G4c / G4h (W28 C6, q-4 A, q-5 i; leaves of the status, before the circulation leaf): a circulation without a librating
     episode is judged on its single legs. 1:1 (G4c, MMR_COORBITAL_SINGLE_LEG; `Geometry.single_legs`, the legs of the
     orbit-averaged angle outside every accepted run, period 2 L >= the floor less CONFIDENCE_THIN_BAND): a clean leg
     (tadpole, horseshoe, quasi-satellite) whose local 1:1 law (K > 0) and energy integral pass and whose two turns
     reflect (|ln out/in| <= MMR_COORBITAL_TURN_ENERGY_TOL, zone <= MMR_COORBITAL_TURN_ZONE_MAX) -> transient (cap
     one_coorbital_leg); one discrepancy -> uncertain/one_coorbital_leg (review); a clean edge leg whose turn reflects
     -> uncertain/insufficient_time (extend); both turns scatter -> the circulation with the near-episode cap; the
     strongest outcome of the legs decides. Not 1:1 (G4h, MMR_SINGLE_HALF_LIBRATION; `Geometry.half_librations`): a
     half-swing between passages that follows the resonance's one-harmonic law (R^2 >= MMR_FORCE_R2_FAIL per half,
     restoring at both turns) -> uncertain/one_half_libration (review), at the record edge insufficient_time (extend).
     The structural near-episode of a 1:1 angle also counts a 2-TP run rejected over_the_top whose half-swings each
     cross at most one of the planet and L3 (`geometry.short_runs`, rule coorbital_near_episode)
status (SPEC §5.3 with DELTA 13b; G1c, G1e, G1l and G2u are judged as the episodic angle their episodes show,
`status_basis`; a G2u union whose episodic reading is a dynamic transient becomes resonant, `promote_union`):
subresonance accepted -> the member's status/subresonance_librates (the member is the verdict's resonant_argument,
its own row decides; both
librating: each row keeps its own verdict; the member is the decisive episode's, chosen in its own window by the
theory's flattest centre, W28 A6b; a composite regime names no member: an own libration takes origin_detail
composite_multiplet, `composite_regime_detail`); owner accepted (another commensurability) whose measured regimes cover at
least mmr_owner_cover_min of every accepted episode of sigma, with Psi/u turning less than
mmr_owner_psi_turns_per_cycle_max per cycle for a librating owner (F7 per episode and P2, A1) ->
non_resonant/induced_libration (cap second_capture_unresolved unless the second capture is excluded: by the sweep and
energy route, or by F3 when x_ddot follows the secular cycle, rule second_capture_f3; confirmed, A2b: x = Psi/u
bounded, with an energy integral, a well of its own (x_ddot on sin x) at a non-secular period and >=
libration_centre_min_estimates closed cycles -> the angle's own reading below, origin joint when it claims the
libration, rule both_resonances_confirmed; the same on one cycle -> uncertain/baseline_needed, origin joint, 2j); an
episode it does not
induce is the row's own: the own episodes decide (the induced ones out of the geometry in a second pass; none induced:
origin none, flag owner_outside_episode or owner_psi_too_fast, the owner a neighbour, the tree goes on); B1 or kinematic fail ->
kinematic_libration|kinematic_drift (K2, W20: a slow centre that drifts with the secular part under a ripple of
episodes -> kinematic_drift with the slow centre's own angle word, drift or circulation, `slow_centre_is_kinematic`,
`_k2_decision`); drift otherwise ->
uncertain/drift_mechanism_unknown; a force law contaminated by a forced line of the angle's lattice at its period ->
uncertain/origin_unknown (5f, `_contaminated`); a librating angle whose force law and potential with a both do not
pass and whose period needs >= mmr_strength_implausible_ratio x a strength computed from the disturbing function (two-body
direct averaging or the harmonic in extended precision, not centre_outside_well unless mmr_strength_first_outside_well;
three-body legal e^q with the small-divisor residual) -> non_resonant/implausible_strength (5s before 5a/5a*/5e/5g and
the branch, W20); on a step that folds short periods a failed force law or identity reads as not_measured when the
direct strength F passes and the short-period scatter of a exceeds mmr_short_period_a_ratio x the a swing (R-ALIAS-A,
`short_period_unmeasured`; the level reads L3/L4 so too); force law x potential with a (four combinations); a librating angle
whose force law and potential with a are both not measured at decision time takes the branch rule
(`unmeasured_branch`, author 2026-10-17): direct averaging confirms the strength -> as below; an implausible strength
-> non_resonant/implausible_strength (5s; three-body only when the small-divisor envelope cannot close the gap, else
uncertain 5s*); a measured objection (O1, F by direct averaging) -> uncertain (5o); the order-of-magnitude strength
scale (eq_scale) out of range, no objection but no origin either -> uncertain (5x, W28 B9); one independent cycle ->
uncertain (5n); nothing measured -> the geometry's status with origin unknown and the cap geometry_only (5u/6: geometric_libration |
geometric_episode | separatrix_zone; strength_scale_only when the e^q scale agrees); dynamics -> resonant |
transient | near_separatrix by the observed behaviour (the zone and the cycle energies are columns and change no
status, DECISIONS 2026-10-15 evening). After the leaf, K3s (W28 B10a, `secular_cycle_kinematic`): a non_resonant
circulation whose slow scale (`analyse._slow_scale`) has a secular period and fails its own identity, its centre
leaving the slow libration only by slips of one turn, becomes kinematic_libration/secular_cycle; graded as the
circulation it was.
Every comparison goes through the ledger; no bare numbers here, every threshold is a setting.

The check values (pass/fail/...) are the truth of a test; the ledger entries next to them record the numbers
the test read, so that the decision path, the CSV and the figure show why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.filtering import is_finite
from resonances.resonance.analysis.geometry import (
    EDGE_REFLECTION,
    EDGE_SLOWDOWN,
    LEG_EDGE_END,
    LEG_EDGE_START,
    EdgeStretch,
    Episode,
    Geometry,
    HalfLibration,
    LegTurn,
    RejectReason,
    SingleLeg,
    UnionCover,
    break_range,
    centre_travel_in,
    episode_centre_travel,
    covering_episode,
    covering_union,
    covers_edges,
    episode_eligibility as geometry_eligibility,
    half_amplitudes,
    run_of,
    short_runs,
    window_step_limit_yr,
)
from resonances.resonance.analysis.ledger import Ledger, Margin, Op
from resonances.resonance.analysis.mmr.models import (
    Baseline,
    Context,
    DirectForceLaw,
    EpisodePhysics,
    ForcedWave,
    HarmonicLaw,
    Kinematics,
    OwnerResult,
    Preprocessed,
    Response,
    Sampling,
    SlowScale,
    StepVerdict,
    Strength,
    SubresonanceResult,
    physics_of,
)
from resonances.resonance.analysis.mmr import response as response_module
from resonances.resonance.analysis.mmr.forced_lines import RESOLUTION
from resonances.resonance.analysis.mmr.kinematics import K1E_NO_VARPI, K1E_RETROGRADE
from resonances.resonance.analysis.mmr.response import LAW_DIRECT, LAW_HARMONIC
from resonances.resonance.analysis.mmr.owner import (
    CAPTURE_CONFIRMED,
    CAPTURE_EXCLUDED,
    CAPTURE_ONE_CYCLE,
    ROUTE_SECULAR,
    FLAG_CATALOGUE,
    FLAG_COORBITAL,
    GAIN_PHASE,
    GAIN_POTENTIAL,
    INTERVAL_LIBRATION,
    REASON_PHASE_UNRESOLVED,
    bridged_cover,
    covered_fraction,
    induced_split,
    record_decisive,
)
from resonances.resonance.analysis.mmr.strength import DIRECT_METHODS, REASON_CENTRE_OUTSIDE_WELL
from resonances.resonance.analysis.mmr import slow_swing as slow_swing_module
from resonances.resonance.analysis.mmr.slow_swing import SlowSwing
from resonances.resonance.analysis import rules as rules_module
from resonances.resonance.analysis.pendulum import EPISODES_IN_SEPARATRIX_ZONE, EnergyAnalysis, SeparatrixMechanism, has_zone_cycle
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Flag,
    MMR_CHECKS,
    LibrationOrigin,
    MarginClass,
    MarginValue,
    OriginDetail,
    Reason,
    Status,
    Verdict,
    finite_or_none,
    margin_of_check,
)

# Margin ids of SPEC §7.1 and their class; R0 is the potential with a (a decisive physics margin)
MARGIN_CLASSES = {
    'L1': MarginClass.DECISIVE,
    'L2': MarginClass.DECISIVE,
    'L3': MarginClass.SUPPORT,
    'L4': MarginClass.DECISIVE,
    'L5': MarginClass.SUPPORT,
    'L6': MarginClass.DECISIVE,
    'L7': MarginClass.DECISIVE,
    'L8': MarginClass.SUPPORT,
    'O1': MarginClass.CAP,
    'Z': MarginClass.DECISIVE,
    'R0': MarginClass.DECISIVE,
    'S': MarginClass.DECISIVE,
    'W': MarginClass.DECISIVE,
    'C1': MarginClass.DECISIVE,
    'C2': MarginClass.SUPPORT,
    'C3': MarginClass.SUPPORT,
    'C4': MarginClass.SUPPORT,
    'C5': MarginClass.DECISIVE,
    'K1': MarginClass.DECISIVE,
    'K1e': MarginClass.SUPPORT,
    'B5': MarginClass.SUPPORT,
    'F': MarginClass.SUPPORT,
    'N': MarginClass.CAP,
}
# Physics of the angle's own force (confidence.raw_level: when every applicable one is not_measured, the cap
# all_physics_not_measured): the force law, the potential with a, the zone, the owner search and the strength (F only
# when measured by direct averaging: `strength.physics_margins`, applied by analyse_mmr). Identity
# (L3), B1 and one pendulum (L5) are not in it: they show that a carries the oscillation (against a kinematic origin)
# but hold as well for an oscillation another resonance induces in a (PHYSICS N5).
PHYSICS_MARGINS = ('L4', 'R0', 'Z', 'W', 'F')
# A kinematic verdict has one more: K1e (A5, W28, card g05-3), the episode's sigma following p varpi with a flat a at a
# secular period (`kinematic_episode_margin`); read only when the origin is kinematic (`physics_margins_of`)
KINEMATIC_PHYSICS_MARGINS = ('K1e',)

TWO_PI = 2 * math.pi
NAN = float('nan')
# G2, the union of librating episodes of an angle that is not 1:1 (`_g2`): the status tree reads it as episodic, and
# promotes only a dynamic transient (`promote_union`)
RULE_UNION = 'G2u'
UNION_PROMOTED = '6u union of episodes covers the record: the episodic reading reached a dynamic transient'
RULE_K2 = '3k2'  # step 3, the kinematic drift of the slow centre under a ripple of episodes (`slow_centre_is_kinematic`)
RULE_K3S = 'K3s'  # B10a: a circulation whose slow scale is a secular cycle (`secular_cycle_kinematic`)
# ledger names the explained verdict column reads (plotting/verdict_text.py finds the entries by these names)
G4B_NET = 'G4b net shift of the slow centre'
K2_NET = '3 K2 angle word: net shift of the slow centre (G4b)'
K2B_SHIFT = '3 K2b kinematic libration whose centre moves more than its amplitude (L6)'
K2_SHARE = '3 kinematic (slow centre): largest episode swing / range of the slow centre'
K2_CORR = '3 kinematic (slow centre): the centre follows the lambda part + the secular part, corr of the rates'
G5A_ONE_REVOLUTION = 'G5a slow centre stays within one revolution'
# G5w (W28 B6c, `_g5w`): the unfinished slow swing is a swing in the resonance's own well
RULE_G5W = 'G5w'
G5W_IDENTITY_SLOPE = 'G5w slow identity |slope / c - 1| (sigma and a low-passed at W_c)'
G5W_IDENTITY_CORR = 'G5w slow identity correlation'
G5W_A_LOW = 'G5w a swing between the turns / pendulum at its amplitude, lower bound 1/sqrt(MMR_STRENGTH_CHECK_RATIO)'
G5W_A_HIGH = 'G5w a swing between the turns / pendulum at its amplitude, upper bound sqrt(MMR_STRENGTH_CHECK_RATIO)'
G5W_P_LOW = 'G5w period 2 (t2 - t1) / T0_th F(A), lower bound 1/sqrt(MMR_STRENGTH_CHECK_RATIO)'
G5W_P_HIGH = 'G5w period 2 (t2 - t1) / T0_th F(A), upper bound sqrt(MMR_STRENGTH_CHECK_RATIO)'
G6_RANGE = 'G6 range of sigma_f'
C5_BRANCH = 'C5 alias: one branch over the a range'
C5_FAR = 'C5 alias: far from the resonance'
OWNER_COVER = '2 owner cover of the decisive episode'
OWNER_PSI_TURNS = '2 owner Psi/u turns per sigma cycle (P2)'
B1_AMPLITUDE = '3 B1 a carries the amplitude'
KIN_VARPI = '3 kinematic: slow sigma follows p varpi'
NULL_TEST = '5 L4 null test: R2 above the binding line - max R2 with a shifted'
HARMONIC_LAW = '5 L4 N-harmonic law'  # CS12 (`_harmonic_law_entries`)
HARMONIC_NULL = '5 L4 N-harmonic law null test: R2 - max R2 with a shifted (full windows)'
HARMONIC_HALVES = '5 L4 N-harmonic law on independent halves: R2 of one half predicting the other'
DIRECT_BRANCH = '5d (i) F by direct averaging'  # the label of rule 5d/6 (i)'s strength check (`_strength_within`)
INDEPENDENT_CYCLES = '5n independent cycles (centre estimates) of the decisive episode'
O1_ONSET = '5o O1 a-wave before / inside the episode'
F_IN_WELL = '5o F an observed swing librates in the theoretical well (consistent evaluations)'
# rule 5s: each name follows the prefix '5s' or STRENGTH_FIRST
STRENGTH_DIRECT = 'required / directly averaged strength (two-body)'
STRENGTH_THREE_BODY = 'required / mu_i mu_j e^q (three-body, legal e^q)'
STRENGTH_DIVISORS = 'small divisors cannot close it: required / (mu_i mu_j e^q B^2)'
KINEMATIC_REASONS = (Reason.KINEMATIC_LIBRATION, Reason.KINEMATIC_DRIFT)
ALIASED = (StepVerdict.ALIASED_UNAMBIGUOUS, StepVerdict.AMBIGUOUS)
OSCILLATING = (AngleStatus.LIBRATION, AngleStatus.EPISODIC, AngleStatus.NEAR_SEPARATRIX, AngleStatus.DRIFT)
LIBRATING = (AngleStatus.LIBRATION, AngleStatus.EPISODIC)
MEASURED = (CheckValue.PASS, CheckValue.FAIL)


@dataclass(frozen=True)
class AngleDecision:
    angle_status: AngleStatus
    subtype: AngleSubtype
    rule: str  # 'G0'..'G6'
    episode_id: Optional[int] = None  # the decisive episode (G2): the covering one, else the one with most TPs
    # G1e: the last regime after a jump of a has no episode, but a librating episode precedes the jump (the one with
    # most TPs); the status reads it as episodic (`status_basis`), the response and the energy of the last regime
    # do not describe it (`other_regime`)
    earlier_episode_id: Optional[int] = None


# the guards that make the status tree read a measured check as not_measured (`Withdrawal.guard`)
WITHDRAWN_FILTER_GUARD = 'filter_guard'  # a passed check whose quantity changes at 2 P_c (`energy_for_tree`, W21 Q9)
WITHDRAWN_ALIAS_A = 'alias_a'  # a failed a-based check on a folding step (`short_period_unmeasured`, R-ALIAS-A)


@dataclass(frozen=True)
class Withdrawal:
    """A measured check the status tree read as not_measured: the guard that withdrew it (WITHDRAWN_*) and the check
    in that guard's terms (the filter guard's `check_mmr_*`, R-ALIAS-A's Response field: the spellings of the columns
    flt_withdrawn_before_tree and rsp_short_period_unmeasured)."""

    guard: str
    check: str


@dataclass
class StatusDecision:
    status: Status
    reason: Reason
    origin: LibrationOrigin
    origin_detail: str
    # the other key the tree named: the owner or driver (induced), the accepted member (subresonance) or an owner F7/P2
    # refused (its flags); the verdict splits it into owner / resonant_argument / neighbours (`roles.verdict_roles`)
    owner: str
    flags: List[str] = field(default_factory=list)
    caps: List[str] = field(default_factory=list)
    rule: str = ''  # e.g. '1', '3', '5b', 'R4'
    # the angle word the verdict reports instead of the geometry's (`reported_angle`): K2's drift or circulation
    angle_word: Optional[AngleStatus] = None
    # the measured checks the tree read as not_measured, by margin id (both guards: the filter guard and R-ALIAS-A);
    # the confidence reads those margins the same way (`margins_as_tree_read`)
    withdrawn: Dict[str, Withdrawal] = field(default_factory=dict)
    # the union of episodes (G2u) promoted from a dynamic transient to resonant (`promote_union`)
    union_promoted: bool = False
    # rules (`analysis.rules`) that acted on the status (the owner per episode, P2), merged by `rules_fired`
    rules_fired: frozenset = frozenset()
    # A2b: the owner's key when 'both resonances at once?' was confirmed and the tree gave the angle's own reading;
    # `joint_capture_origin` names it origin joint at the end of `decide_status`
    joint_owner: str = ''


# --------------------------------------------------------------------------------------------- helpers


def _flag(value: bool) -> float:
    """A yes/no fact as a number for the ledger (1 = yes)."""
    return 1.0 if value else 0.0


def episodes_in_separatrix_zone(energy: Optional[EnergyAnalysis]) -> bool:
    """The energy's episodes_in_separatrix_zone (False without an energy analysis): crossings with every cycle inside
    the model's separatrix zone ('separatrix_chaos' before 2026-10-17)."""
    return energy is not None and bool(energy.episodes_in_separatrix_zone)


def period_floor(geometry: Geometry, settings: AnalysisSettings, episode=None) -> Tuple[float, str]:
    """The shortest libration period of the angle [yr] and its setting: the one the preprocessing derived
    (geometry.period_floor_yr: a co-orbital's own, `preprocess.libration_period_floor`), else mmr_libration_period_min_yr.
    With `episode` (a run):
    - inside a floor well (`geometry.FloorWell`: the 1:1 quasi-satellite pass, G1q): the angle's (the QS) floor;
    - with a floor of its own (`Geometry.run_floor`: a 1:1 run whose turns reflect on the planet's or L3's top, the
      loop through them, rule coorbital_floor_profile, W28 C3): that floor;
    - outside a floor well: the well's outside floor (the tadpole floor)."""
    well = geometry.floor_well
    if episode is not None and not (well is not None and well.holds(episode)):
        own = geometry.run_floor(episode)
        if own is not None and own.holds:
            return float(own.floor_yr), own.setting
        if well is not None:
            return float(well.outside_floor_yr), well.outside_setting
    floor, setting = geometry.period_floor_yr, geometry.period_floor_setting
    if is_finite(floor) and setting:
        return float(floor), setting
    return float(settings.mmr_libration_period_min_yr), 'mmr_libration_period_min_yr'


def run_floor_acts(geometry: Geometry, settings: AnalysisSettings) -> bool:
    """A run's own floor (`Geometry.run_floor`, rule coorbital_floor_profile) changes what the angle's floor would
    decide for it: an accepted run passes one floor and not the other, or a short run the near-episode test
    (`near_episode_floor`) only by one of them."""
    for run in geometry.runs:
        own = geometry.run_floor(run)
        if own is None or not own.holds or len(run.tp_times_yr) < 2:
            continue
        well = geometry.floor_well
        default = float(well.outside_floor_yr) if well is not None else float(geometry.period_floor_yr)
        if not is_finite(default):
            continue
        period = run.period_med_yr if run.accepted else 2 * float(np.median(np.diff(run.tp_times_yr)))
        band = 1.0 if run.accepted else 1.0 - settings.confidence_thin_band
        if is_finite(period) and (period >= own.floor_yr * band) != (period >= default * band):
            return True
    return False


def _eligible_floor(geometry: Geometry, settings: AnalysisSettings, episode) -> float:
    """The floor an episode's period must reach to count: `period_floor`, less its thin band when it is the
    convention MMR_LIBRATION_PERIOD_MIN_YR (W31, 2001 WK15: 142.9 yr against 150); a measured floor (a co-orbital's
    tadpole or loop period) is read as it is (2022 AP1)."""
    floor, setting = period_floor(geometry, settings, episode)
    return floor * (1.0 - settings.confidence_thin_band) if setting == 'mmr_libration_period_min_yr' else floor


def fastest_episode(geometry: Geometry, settings: AnalysisSettings) -> Optional[Episode]:
    """The accepted episode whose period is the shortest against its own floor (`period_floor`); None without one."""
    return min(geometry.episodes, key=lambda e: e.period_med_yr / period_floor(geometry, settings, e)[0], default=None)


def librating_episodes(geometry: Geometry, settings: AnalysisSettings, before_jump: bool = False) -> List[Episode]:
    """The accepted episodes that are librations of this angle (`before_jump`: of the runs before the last jump of a,
    `Geometry.episodes_before_jump`, instead of the last regime's):

    - the period is not shorter than the shortest libration of the angle (`period_floor`: mmr_libration_period_min_yr,
      a co-orbital's from its planet's tadpole period): a faster 'episode' is a forced wave the cutoff let through;
    - the centre does not complete a monotone revolution inside the episode (`centre_travel_in` < 2pi): turning points
      around a centre that runs a full turn are an oscillation about a moving reference, not a libration of this angle
      (an arc > 2pi is not a piece of libration, DECISIONS 2026-10-15; 82831 13M-24+0+11: 84 TPs at 12-71 kyr while
      the centre runs +7.3 rad monotonically from 30 to 71 kyr - the subresonance librates, this angle circulates). An
      episode accepted in the frame of its moving well reads the travel there (`geometry.episode_centre_travel`): the
      centre running with the well is the libration about a moving centre, not a circulation."""

    reasons = episode_eligibility(geometry, settings)
    wanted = RejectReason.BEFORE_LAST_JUMP if before_jump else RejectReason.NONE
    return [run for run in geometry.runs if run.accepted and reasons.get(run.episode_id) == wanted]


def episode_eligibility(geometry: Geometry, settings: AnalysisSettings) -> Dict[int, RejectReason]:
    """`geometry.episode_eligibility` of the accepted runs at this angle's period floors (`period_floor`), by
    episode_id: NONE for a librating episode of the last regime, BEFORE_LAST_JUMP for one of a regime before the last
    jump of a (G1e, G1l), else why the run does not count (forced_wave, induced, below_period_floor, centre_turns).
    `librating_episodes` reads it; episodes.csv writes it (`eligible`, `ineligible_reason`, W28 C11, card g02-9 A).
    Computed once per geometry and settings (`Geometry.derived`; the forced waves and induced episodes are part of the
    key: F6 and the owner's second pass set them later) and kept as `Geometry.eligibility`. Rule eligible_reason fires
    when an accepted run of the last regime is no librating episode (a run before the last jump is counted by G1e/G1l
    and fires nothing)."""

    def compute() -> Dict[int, RejectReason]:
        reasons = geometry_eligibility(geometry, lambda episode: _eligible_floor(geometry, settings, episode))
        if any(reason not in (RejectReason.NONE, RejectReason.BEFORE_LAST_JUMP) for reason in reasons.values()):
            geometry.rules_fired = rules_module.fire(geometry.rules_fired, rules_module.Rule.ELIGIBLE_REASON)
        return reasons

    reasons = geometry.derived(('episode_eligibility', geometry.excluded_ids), settings, compute)
    geometry.eligibility = reasons
    return reasons


def librating_union(geometry: Geometry, settings: AnalysisSettings) -> UnionCover:
    """`geometry.covering_union` of the window's librating episodes (`librating_episodes`), once per geometry and
    settings (`Geometry.derived`)."""
    key = ('librating_union', geometry.excluded_ids)
    return geometry.derived(key, settings, lambda: covering_union(geometry, librating_episodes(geometry, settings), settings))


def forced_wave_episodes(
    episode_responses: Sequence, settings: AnalysisSettings, geometry: Optional[Geometry] = None
) -> Tuple[ForcedWave, ...]:
    """The accepted episodes that are forced waves of another argument, not librations of this angle (W20 F6; 37087
    3J+8S-3: a 168-yr wave of 2 lambda_J - lambda - varpi at the edge of the 2:1 gap, sigma_dot against c (a - a*) at
    slope -0.35 c, corr -0.85). From each episode's own response (`EpisodeResponse`, rule R's measurement):

    - (i) wrong sign: the identity is measured and fails with slope / c < 0 at |corr| >= mmr_identity_corr_min:
      sigma_dot runs against c (a - a*), which no pendulum of this angle does;
    - (ii) on a line: its period sits on a forced line of the angle's lattice within the window's resolution
      (`EpisodeResponse.force_line`, `response.force_line`) and its identity is measured and fails; with
      mmr_forced_wave_line_alone the line alone (a chance coincidence with a line happens for 6-17 % of the force
      laws, W12, so by default it needs the failed identity).

    `Geometry.forced_waves` holds them (`Geometry.episodes` leaves them out); the analysis measures the response again
    without them. With the window's `geometry`, the unconfirmed ripples on an unfinished slow swing follow
    (`ripple_waves`, W28 B6b)."""
    scratch = Ledger.from_settings(settings)
    found = []
    for episode in episode_responses:
        fails = episode.identity == CheckValue.FAIL
        wrong = bool(
            fails
            and scratch.compare('slope', episode.identity_slope_ratio, Op.LT, 0, '', '')
            and scratch.compare('corr', -episode.identity_corr, Op.GE, settings.mmr_identity_corr_min, '', '')
        )
        on_line = bool(episode.force_line) and (fails or settings.mmr_forced_wave_line_alone)
        if episode.episode_id is None or not (wrong or on_line):
            continue
        found.append(
            ForcedWave(
                episode_id=episode.episode_id,
                n_tp=episode.n_tp,
                period_yr=episode.period_yr,
                identity_slope_ratio=episode.identity_slope_ratio,
                identity_corr=episode.identity_corr,
                force_line=episode.force_line,
                force_line_offset=episode.force_line_offset,
                wrong_sign=wrong,
                on_line=on_line,
            )
        )
    taken = {wave.episode_id for wave in found}
    return tuple(found) + tuple(w for w in ripple_waves(geometry, episode_responses, settings) if w.episode_id not in taken)


def unfinished_slow_swing(geometry: Geometry) -> bool:
    """The record holds part of one slow swing of the angle (W28 B6b, condition iii): its slow centre
    (`Geometry.centre`) stays within one revolution and turns back once or twice away from the edges
    (`interior_turns`). Fewer than three turns: no full slow cycle is seen, so the slow motion is unfinished; with
    three or more the centre has completed a cycle (2003 UC22 16M-27: four turns over 0.9 rad, the modulated centre of
    a 43-TP libration that covers the record). G5w (`_g5w`, B6c) asks more of a two-turn swing: that it is a swing
    in the resonance's own well."""
    turns = interior_turns(geometry)
    return 1 <= turns <= 2 and is_finite(geometry.centre.range_rad) and geometry.centre.range_rad < TWO_PI


def ripple_waves(geometry: Optional[Geometry], episode_responses: Sequence, settings: AnalysisSettings) -> Tuple[ForcedWave, ...]:
    """W28 B6b (card g08-4 A, rule `ripple_on_arc`): the accepted episodes of the window that are ripples without
    measurable physics on an unfinished slow swing, auxiliary intervals that do not decide the angle. All of:

    - (i) the half-amplitude stays below MMR_A_RESPONSE_MIN_AMP_DEG (largest half-amplitude): sin sigma ~ sigma, so
      neither the force law L4 nor R0 with a can be measured on it;
    - (ii) its identity does not pass on it (`EpisodeResponse.identity` fail or not measured): nothing confirms that
      sigma_dot follows c (a - a*) as this angle's libration must (W32, the author's 2002 BG card C: a one-cycle
      ripple whose identity cannot be measured is as unconfirmed as one whose identity fails);
    - (iii) the record holds an unfinished slow swing (`unfinished_slow_swing`);

    and it is no record-covering episode (`covers_edges` on both edges: then there is no arc beside it, the episode is
    the angle's motion over the record) and the angle is not 1:1 (the co-orbital law is measured at any amplitude).
    2004 YG1 9M-10J-4 over 20 kyr: a 3.1 deg ripple of 5 TPs at 13.5-14.9 kyr, identity fail (slope/c 0.65, corr
    0.79), at the one turn (13.7 kyr) of a slow centre that sweeps 281 deg; 2002 BG 1E-10S-1 over 20 kyr: a 2-deg
    735-yr ripple of 3 TPs (identity not measured) at the one turn of a centre that sweeps 1.68 rad, so the record is
    one unfinished slow swing (insufficient_time, extend). A real small libration keeps its passing identity, so it is
    never taken. No strength ratio and no global prominence (the card): () with MMR_RIPPLE_ON_ARC
    off, without a geometry, or when (iii) does not hold. Set before G2 like F6 (`Geometry.forced_waves`, route
    'unconfirmed_ripple'), so G5a/G6 read the slow centre (`_g4_to_g6`)."""
    if not settings.mmr_ripple_on_arc or geometry is None or geometry.coorbital or not unfinished_slow_swing(geometry):
        return ()
    limit = math.radians(settings.mmr_a_response_min_amp_deg)
    scratch = Ledger.from_settings(settings)
    found = []
    for episode in episode_responses:
        run = run_of(geometry, episode.episode_id) if episode.episode_id is not None else None
        if run is None or not run.accepted or run.before_last_jump or episode.identity == CheckValue.PASS:
            continue
        if not scratch.compare('ripple', run.half_amp_max_rad, Op.LT, limit, 'rad', '') or all(covers_edges(run, settings)):
            continue
        found.append(
            ForcedWave(
                episode_id=run.episode_id,
                n_tp=episode.n_tp,
                period_yr=episode.period_yr,
                identity_slope_ratio=episode.identity_slope_ratio,
                identity_corr=episode.identity_corr,
                force_line=episode.force_line,
                force_line_offset=episode.force_line_offset,
                wrong_sign=False,
                on_line=False,
                ripple=True,
                half_amp_max_rad=run.half_amp_max_rad,
            )
        )
    return tuple(found)


def ripples(geometry: Optional[Geometry]) -> Tuple[ForcedWave, ...]:
    """The unconfirmed ripples of the geometry (`ripple_waves`, `ForcedWave.ripple`)."""
    return tuple(wave for wave in geometry.forced_waves if wave.ripple) if geometry is not None else ()


def tp_outside_ripples(geometry: Geometry) -> int:
    """The turning points of the window that belong to no unconfirmed ripple (`ripples`): G6 counts these."""
    times = np.asarray(geometry.turning_points.times_yr, dtype=float)
    runs = [run_of(geometry, wave.episode_id) for wave in ripples(geometry)]
    taken = [np.asarray(run.tp_times_yr, dtype=float) for run in runs if run is not None]
    if not taken:
        return len(times)
    return int(np.count_nonzero(~np.isin(times, np.concatenate(taken))))


def strength_forced_waves(geometry: Optional[Geometry], decision: StatusDecision, settings: AnalysisSettings) -> Tuple[ForcedWave, ...]:
    """W28 B2 (card g11-1, rule `forced_strength_mark`): on a row that rule 5s decided (reason implausible_strength,
    not the uncertain 5s*), every accepted run that is still an episode of the angle (before or after the last jump of
    a; F6 waves and induced episodes already are none) as a forced wave with the route 'strength': the period of its
    loops needs a force >= MMR_STRENGTH_IMPLAUSIBLE_RATIO x the one this angle's disturbing function gives, so they
    are no libration of this angle. A name of the ground, not a driver found: nothing is measured again, and the
    caller adds them to `Geometry.forced_waves` only after the verdict and its numbers (no status, word, reason,
    level or ledger value reads them). Empty with MMR_FORCED_WAVE_STRENGTH off, without a geometry or another reason."""
    if not settings.mmr_forced_wave_strength or geometry is None or decision.reason != Reason.IMPLAUSIBLE_STRENGTH:
        return ()
    excluded = geometry.excluded_ids
    nan = float('nan')
    return tuple(
        ForcedWave(
            episode_id=run.episode_id,
            n_tp=run.n_tp,
            period_yr=run.period_med_yr,
            identity_slope_ratio=nan,
            identity_corr=nan,
            force_line='',
            force_line_offset=nan,
            wrong_sign=False,
            on_line=False,
            by_strength=True,
        )
        for run in geometry.runs
        if run.accepted and run.episode_id not in excluded
    )


def _f6_waves(geometry: Optional[Geometry]) -> tuple:
    """The forced waves F6 found by their identity or lattice line (`forced_wave_episodes`), without the marks of rule
    5s (`ForcedWave.by_strength`) and the unconfirmed ripples (`ForcedWave.ripple`, B6b), which G2's forced-wave
    lines, its columns and the angle word do not read (a ripple has its own G2 lines, `_ripple_entries`)."""
    if geometry is None:
        return ()
    return tuple(wave for wave in geometry.forced_waves if not wave.by_strength and not wave.ripple)


def _main_forced_wave(geometry: Optional[Geometry]) -> Optional[ForcedWave]:
    """The F6 forced wave (`_f6_waves`) with most turning points (the earlier on a tie); None without one."""
    waves = _f6_waves(geometry)
    return max(waves, key=lambda w: (w.n_tp, -w.episode_id)) if waves else None


def forced_wave_columns(geometry: Optional[Geometry]) -> dict:
    """mmr_numbers.csv: the count of F6 forced waves (`forced_wave_episodes`; the strength marks of rule 5s are not
    counted, `_f6_waves`) and the numbers of the one with most turning points, which the ledger of G2 reads (nan
    without one)."""
    wave = _main_forced_wave(geometry)
    nan = float('nan')
    slope = wave.identity_slope_ratio if wave is not None else nan
    return {
        'geo_n_forced_wave_episodes': len(_f6_waves(geometry)),
        'geo_forced_wave_episode_id': wave.episode_id if wave is not None else nan,
        'geo_forced_wave_period_yr': wave.period_yr if wave is not None else nan,
        'geo_forced_wave_line': wave.force_line if wave is not None else '',
        'geo_forced_wave_line_offset': wave.force_line_offset if wave is not None else nan,
        'geo_forced_wave_identity_slope_ratio': slope,
        'geo_forced_wave_identity_slope_offset': abs(slope - 1) if is_finite(slope) else nan,
        'geo_forced_wave_identity_corr': wave.identity_corr if wave is not None else nan,
        'geo_forced_wave_identity_corr_abs': abs(wave.identity_corr) if wave is not None else nan,
    }


def _forced_wave_entries(geometry: Geometry, settings: AnalysisSettings, ledger: Ledger) -> None:
    """G2 on the ledger when accepted runs were judged forced waves (`forced_wave_episodes`): the numbers of the one
    with most turning points (`forced_wave_columns`), and the count."""
    wave = _main_forced_wave(geometry)
    if wave is None:
        return
    columns = forced_wave_columns(geometry)
    prefix = f'G2 forced wave (episode {wave.episode_id}, P {wave.period_yr:.0f} yr)'
    if wave.on_line:
        name = f'{prefix}: on the lattice line {wave.force_line}, |1/T - 1/P| W P'
        ledger.compare(
            name, wave.force_line_offset, Op.LT, RESOLUTION, '', 'geo_forced_wave_line_offset', setting='mmr_force_window_cycles'
        )
    if wave.wrong_sign:
        name = f'{prefix}: identity slope / c'
        ledger.compare(name, wave.identity_slope_ratio, Op.LT, 0, '', 'geo_forced_wave_identity_slope_ratio')
        name = f'{prefix}: |identity corr|'
        ledger.compare_setting(
            name,
            columns['geo_forced_wave_identity_corr_abs'],
            Op.GE,
            settings,
            'mmr_identity_corr_min',
            '',
            'geo_forced_wave_identity_corr_abs',
        )
    elif wave.on_line and not settings.mmr_forced_wave_line_alone:
        name = f'{prefix}: identity |slope / c - 1| (fails above, or by its correlation)'
        value = columns['geo_forced_wave_identity_slope_offset']
        ledger.compare_setting(name, value, Op.LE, settings, 'mmr_identity_slope_tol', '', 'geo_forced_wave_identity_slope_offset')
        name = f'{prefix}: identity correlation'
        ledger.compare_setting(name, wave.identity_corr, Op.GE, settings, 'mmr_identity_corr_min', '', 'geo_forced_wave_identity_corr')
    name = 'G2 accepted runs that are forced waves, no episodes of the angle'
    ledger.compare(name, columns['geo_n_forced_wave_episodes'], Op.EQ, 0, '', 'geo_n_forced_wave_episodes')


def decisive_episode(geometry: Geometry, settings: AnalysisSettings) -> Optional[Episode]:
    """The episode the verdict rests on: the covering one if any (geometry.covering_episode), else the one with most
    turning points of the episodes that cover the record together (`geometry.covering_union`), else the
    one with most turning points among the librating episodes of the window (`librating_episodes`)."""
    episodes = librating_episodes(geometry, settings)
    if not episodes:
        return None
    covering = covering_episode(episodes, settings)
    if covering is not None:
        return covering
    union = librating_union(geometry, settings).episodes
    return max(union or episodes, key=lambda episode: episode.n_tp)


def tp_half_amplitude_median(geometry: Geometry) -> float:
    """Median half-amplitude of every half-cycle of the window [rad] (pattern A without a zone)."""
    values = half_amplitudes(geometry.turning_points)
    return float(np.median(values)) if len(values) else float('nan')


def centre_rate_significance(response: Response) -> float:
    """|nu_c| / dnu_c: the centre rate in units of its error (B5 exact commensurability)."""
    rate, error = response.centre_rate_rad_yr, response.centre_rate_err_rad_yr
    return abs(rate) / error if is_finite(rate) and is_finite(error) and error > 0 else float('nan')


def round_up(value: float) -> float:
    """The next round value >= value: one significant digit, rounded up (37 000 -> 40 000)."""
    if not is_finite(value) or value <= 0:
        return float('nan')
    scale = 10.0 ** math.floor(math.log10(value))
    return float(math.ceil(value / scale - 1e-12) * scale)


def rules_fired(geometry: Optional[Geometry], *fired) -> List[str]:
    """`Verdict.rules_fired` of an MMR row: the rules of the geometry (`Geometry.rules_fired`) and of every other
    frozenset of fired rules given (the result objects' own, added by the work package that fires them), deduplicated,
    in registry order (`rules.ordered`). No geometry (a verdict before the measurements) adds none."""
    return rules_module.ordered(geometry.rules_fired if geometry is not None else (), *fired)


# --------------------------------------------------------------------------------------------- angle_status


def _g0(sampling: Sampling, unphysical: bool, geometry: Optional[Geometry], settings: AnalysisSettings, ledger: Ledger):
    if not ledger.compare('G0 orbit is physical (unphysical = 0)', _flag(unphysical), Op.EQ, 0, '', 'rec_unphysical'):
        return AngleDecision(AngleStatus.UNDEFINED, AngleSubtype.UNPHYSICAL, 'G0')
    p99 = ledger.compare_setting(
        'G0 unwrap residual p99',
        math.degrees(sampling.unwrap_resid_p99_rad),
        Op.LE,
        settings,
        'mmr_unwrap_max_resid_p99_deg',
        'deg',
        'smp_unwrap_resid_p99_deg',
    )
    largest = ledger.compare_setting(
        'G0 unwrap residual max',
        math.degrees(sampling.unwrap_resid_max_rad),
        Op.LE,
        settings,
        'mmr_unwrap_max_resid_deg',
        'deg',
        'smp_unwrap_resid_max_deg',
    )
    if sampling.jitter_low_e:
        _g0_phase(sampling, settings, ledger)
    if p99 and largest and sampling.unwrap_ok and geometry is not None:
        return None
    subtype = AngleSubtype.UNDEFINED_PHASE if sampling.undefined_phase else AngleSubtype.ALIAS
    return AngleDecision(AngleStatus.UNDEFINED, subtype, 'G0')


def _g0_phase(sampling: Sampling, settings: AnalysisSettings, ledger: Ledger) -> None:
    """The three comparisons of undefined_phase (`sampling.analyse_sampling`), recorded so that decision_path explains
    it: the |p|-scaled jitter, the full angle's residual, and whether varpi itself is defined (258640 46J-19+0-27)."""
    ledger.compare_setting('G0 angle jitter', sampling.jitter, Op.LE, settings, 'mmr_jitter_max_rad', 'rad', 'smp_jitter')
    for name, value, setting, column in (
        ('G0 full-angle residual p99', sampling.unwrap_resid_full_p99_rad, 'mmr_unwrap_max_resid_p99_deg', 'smp_unwrap_resid_full_p99_deg'),
        ('G0 full-angle residual max', sampling.unwrap_resid_full_max_rad, 'mmr_unwrap_max_resid_deg', 'smp_unwrap_resid_full_max_deg'),
    ):
        ledger.compare_setting(name, math.degrees(value), Op.LE, settings, setting, 'deg', column)
    share = sampling.varpi_undefined_fraction
    ledger.compare_setting(
        'G0 varpi phase defined', share, Op.LE, settings, 'mmr_varpi_undefined_max_share', '', 'smp_varpi_undefined_fraction'
    )


def final_stretch(geometry: Geometry, settings: AnalysisSettings) -> Optional[Tuple[float, float, Optional[Episode]]]:
    """The stretch after the last turning point that G1c reads: (the centre's travel there [rad], its length in periods
    of the last librating episode [P], that episode); the length nan and the episode None without a librating episode,
    None without turning points."""
    times = np.asarray(geometry.turning_points.times_yr, float)
    if not len(times):
        return None
    start, end = float(times[-1]), float(geometry.window[1])
    episodes = librating_episodes(geometry, settings)
    last = max(episodes, key=lambda episode: episode.t_end_yr) if episodes else None
    period = last.period_med_yr if last is not None else float('nan')
    beyond = (end - start) / period if is_finite(period) and period > 0 else float('nan')
    return centre_travel_in(geometry, start, end), beyond, last


def _g1c(geometry: Geometry, settings: AnalysisSettings, ledger: Ledger) -> Optional[AngleDecision]:
    """G1c, the last regime is a circulation: after the last turning point the window runs on without one, the centre
    completes a monotone revolution there (`centre_travel_in` >= 2pi, the pass measured by max/min) and that stretch is
    longer than libration_coverage_periods periods of the last librating episode (a visible exit inside the last
    period stays the episode's: exit_suspected). The label describes the last regime (rule 10; DECISIONS 2026-10-15:
    >= 1 revolution of the centre is circulation, 82831: 8.9 turns in the last 27 kyr). The earlier episodes stay in
    episodes.csv and the status still reads them (`status_basis`: status by the observed behaviour, DELTA 13b)."""
    stretch = final_stretch(geometry, settings)
    if stretch is None or stretch[2] is None:
        return None
    travel, beyond, last = stretch
    if not ledger.compare('G1c centre turns after the last turning point', travel, Op.GE, TWO_PI, 'rad', 'geo_final_centre_travel_rad'):
        return None
    name = 'G1c circulation lasts beyond the last period'
    if not ledger.compare_setting(name, beyond, Op.GT, settings, 'libration_coverage_periods', 'P', 'geo_final_circulation_P'):
        return None
    return AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G1c', last.episode_id)


def _g1e(angle: AngleDecision, geometry: Geometry, settings: AnalysisSettings, ledger: Ledger) -> AngleDecision:
    """G1e, episodes before the last jump of a: the last regime decides angle_status (rule 10, author 12.4), but when it
    holds no episode (circulation, drift, near_separatrix, too little time) the librating episodes before the jump are
    still observed behaviour, so the status reads the angle as the episodic one they show (DELTA 13b; 2021 BA: a 3-TP
    horseshoe at 134-407 yr, then a jump and circulation; 524522 1V-1 20 kyr: four episodes of 5-25 TPs before the
    jumps at 17.7-19 kyr). An aliased circulation keeps its verdict: the step decides there, not the episodes."""
    if angle.angle_status in LIBRATING or angle.angle_status == AngleStatus.UNDEFINED or angle.subtype == AngleSubtype.ALIAS:
        return angle
    if Flag.LAST_REGIME_AFTER_JUMP not in geometry.flags:
        return angle
    earlier = librating_episodes(geometry, settings, before_jump=True)
    if not ledger.compare('G1e librating episodes before the last jump of a', len(earlier), Op.GE, 1, '', 'geo_n_episodes_before_jump'):
        return angle
    return replace(angle, earlier_episode_id=max(earlier, key=lambda episode: episode.n_tp).episode_id)


def before_jump_coverage(geometry: Geometry, settings: AnalysisSettings) -> Tuple[float, float]:
    """(covers the record's start: 1/0, gap from its end to the last regime [P]) of the earliest librating episode
    before the last jump of a (`librating_episodes(before_jump=True)`); the gap is (start of the last jump's zone -
    episode end) / P: a jump of a owns jump -+ tau/2 (tau = the locator's baseline), an action step (G2a) its
    transitional half-cycle (`Geometry.last_cut_zone_yr`). (nan, nan) without such an episode or a jump."""
    earlier = librating_episodes(geometry, settings, before_jump=True)
    if Flag.LAST_REGIME_AFTER_JUMP not in geometry.flags or not earlier:
        return float('nan'), float('nan')
    first = min(earlier, key=lambda episode: episode.t_start_yr)
    period = first.period_med_yr if is_finite(first.period_med_yr) and first.period_med_yr > 0 else float('nan')
    zone_start = geometry.last_cut_zone_yr()[0]
    if not is_finite(zone_start):  # the regime start without a cutting jump (never with the flag): tau before it
        zone_start = geometry.regime_start_yr - (geometry.jump_tau_yr if is_finite(geometry.jump_tau_yr) else 0.0)
    gap = (zone_start - first.t_end_yr) / period
    return _flag(covers_edges(first, settings)[0]), float(gap)


def _g1l(angle: AngleDecision, geometry: Geometry, settings: AnalysisSettings, ledger: Ledger) -> AngleDecision:
    """G1l, a capture after a jump of a (the mirror of G1c/G1e, DECISIONS 2026-10-16): the last regime after the jump
    librates (G2 libration of its window), but no librating episode runs from the record's start to the jump
    (`before_jump_coverage`: the earliest one covers the start edge and ends within libration_coverage_periods periods
    of the jump zone). The libration then does not cover the record: the status reads the angle as episodic
    (`status_basis`), the verdict reports angle_status episodic and final_regime libration from the start of the
    last episode. A libration on both sides of the jump stays a libration. 277617 1M-1 20 kyr: circulation up to
    the jump at 9.2 kyr, then a quasi-satellite to the end."""
    if angle.angle_status != AngleStatus.LIBRATION or Flag.LAST_REGIME_AFTER_JUMP not in geometry.flags:
        return angle
    earlier = librating_episodes(geometry, settings, before_jump=True)
    if ledger.compare('G1l librating episodes before the last jump of a', len(earlier), Op.GE, 1, '', 'geo_n_episodes_before_jump'):
        covers_start, end_gap = before_jump_coverage(geometry, settings)
        name = 'G1l the earliest of them covers the record start'
        if ledger.compare(name, covers_start, Op.EQ, 1, '', 'geo_before_jump_covers_start'):
            name = 'G1l it lasts to the jump of a'
            if ledger.compare_setting(name, end_gap, Op.LE, settings, 'libration_coverage_periods', 'P', 'geo_before_jump_end_gap_P'):
                return angle
    return replace(angle, rule='G1l')


def other_regime(angle: AngleDecision) -> bool:
    """The status rests on an episode before the last jump of a (G1e): the record's response (force law, B1, identity)
    and energy (potential, zone) were measured on the last regime after the jump and say nothing about that episode."""
    return angle.earlier_episode_id is not None


def regime_physics(angle: AngleDecision, response: Optional[Response], energy: Optional[EnergyAnalysis]):
    """(response, energy) as the status may read them. The response is rule R's (`response.deciding_response`): a
    witness episode's force law, identity and B1, else the pooled ones; under G1e (`other_regime`) the pooled ones are
    those of the episodes before the jump (None when none was measured) and the energy is None: the potential and the
    zone are not measured per episode, so every check and margin built on them is not_measured there."""
    deciding = response_module.deciding_response(response, other_regime(angle))
    return deciding, (None if other_regime(angle) else energy)


@dataclass(frozen=True)
class TreeView:
    """The record as the status tree read it, for what comes after the status (margins, checks, caps, the baseline and
    the confidence's word): the angle as the status reads it (`status_basis`), rule R's response and the energy of
    that regime (`regime_physics`: under G1e the response pooled over the episodes before the jump, or None, and no
    energy). Built once (`tree_view`)."""

    angle: AngleDecision
    response: Optional[Response]
    energy: Optional[EnergyAnalysis]


def tree_view(angle: AngleDecision, response: Optional[Response], energy: Optional[EnergyAnalysis]) -> TreeView:
    """`TreeView` of `angle` (after `union_read`) with the record's `response` and `energy`."""
    deciding, regime_energy = regime_physics(angle, response, energy)
    return TreeView(status_basis(angle), deciding, regime_energy)


def _witness_entries(response: Optional[Response], ledger: Ledger) -> None:
    """Rule R in the ledger: the witnesses among the librating episodes and the episodes whose force law fails."""
    episodes = response.episode_responses if response is not None else ()
    if not episodes:
        return
    name = 'R witness episodes (L4 strict pass, B1 and L3 not fail, or L4 thin with L3 and B1 pass)'
    if ledger.compare(name, len(response_module.witnesses(response)), Op.GE, 1, '', 'rsp_n_witnesses'):
        failing = sum(1 for episode in episodes if episode.a_response == CheckValue.FAIL)
        ledger.compare('R no other episode whose force law fails', failing, Op.EQ, 0, '', 'rsp_n_episode_force_fail')


def status_basis(angle: AngleDecision) -> AngleDecision:
    """The angle the status tree reads: a last-regime circulation after librating episodes (G1c), a last regime
    without episodes after a jump of a that ended librating ones (G1e) and a capture after a jump of a (G1l) are
    judged as the episodic angle those episodes show (status by the observed behaviour, DELTA 13b); so is the union of
    episodes of an angle that is not 1:1 (G2u, the physics guard of the W22 union rule: `promote_union`); every other
    angle as it is."""
    if angle.rule in ('G1c', 'G1l'):
        return AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, angle.rule, angle.episode_id)
    if angle.rule == RULE_UNION:  # the episodes as G2 reads them without the union
        return AngleDecision(AngleStatus.EPISODIC, AngleSubtype.RECURRENT, angle.rule, angle.episode_id)
    if other_regime(angle):
        return AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, angle.rule, angle.earlier_episode_id, angle.earlier_episode_id)
    return angle


def reported_angle(
    angle: AngleDecision, geometry: Optional[Geometry], settings: AnalysisSettings, decision: Optional[StatusDecision] = None
) -> AngleDecision:
    """The angle_status and subtype the verdict reports (and its calibration cell reads): the angle as the status reads
    it (`status_basis`), so episodes followed by a circulation (G1c) or by a regime without episodes after a jump of a
    (G1e), and a capture after a jump (G1l, counting the episodes on both sides of the jump) are `episodic` over the
    record, `recurrent` with >= 2 librating episodes (as in G2); a union of episodes (G2u) is libration/regime_change
    whatever the status (the word describes the geometry). The last regime itself goes to `final_regime`. A word the
    status tree set (`StatusDecision.angle_word`: drift for K2, the kinematic drift of the slow centre under a ripple
    of episodes, author 2026-10-19 q7) replaces it. Only the output: the tree, the
    checks and the margins read `angle` and `status_basis`."""
    if decision is not None and decision.angle_word is not None:
        return AngleDecision(decision.angle_word, AngleSubtype.NONE, angle.rule, angle.episode_id, angle.earlier_episode_id)
    if angle.rule == RULE_UNION:  # the geometry of the record: the angle never went round (libration/regime_change)
        return angle
    basis = status_basis(angle)
    if basis is angle or geometry is None:
        return angle
    episodes = librating_episodes(geometry, settings, before_jump=other_regime(angle))
    if angle.rule == 'G1l':  # the capture and the episodes before the jump
        episodes += librating_episodes(geometry, settings, before_jump=True)
    return replace(basis, subtype=AngleSubtype.RECURRENT if len(episodes) >= 2 else AngleSubtype.NONE)


def forced_wave_word(
    reported: AngleDecision,
    decision: Optional[StatusDecision],
    sampling: Sampling,
    geometry: Optional[Geometry],
    energy: Optional[EnergyAnalysis],
    settings: AnalysisSettings,
    ledger: Ledger,
) -> AngleDecision:
    """The angle word with the forced-wave runs (W20 round C; the author's labels of 385266 16N-27 and 37087 3J+8S-3
    at 100 kyr and 1 Myr: non_resonant, episodic). F6 keeps those runs out of the status (`forced_wave_episodes`:
    neither librating episodes nor the decisive one), but the word says what sigma does geometrically over the record,
    and the loops of a forced wave are librations of sigma: when the reported word is circulation or drift and the angle
    tree run with the forced waves kept in gives a librating word, that word is reported (one ledger line). Only the
    output, after the status, the margins, the caps and the confidence (`analyse`: `confidence.assess` reads the tree's
    word): they keep reading the tree without the forced waves. A word the status tree set (K2) stays, and so does a
    word that is no description of the angle's motion over the record (insufficient_time, undefined: W20 round D,
    review C #5, 129772 4N-7 keeps insufficient_time/arc under its G6 verdict)."""
    waves = _f6_waves(geometry)
    rewritable = reported.angle_status in (AngleStatus.CIRCULATION, AngleStatus.DRIFT)
    if not waves or not rewritable or (decision is not None and decision.angle_word is not None):
        return reported
    kept = replace(geometry, forced_waves=())
    geometric = decide_angle(sampling, False, kept, energy, settings, Ledger.from_settings(settings))
    word = reported_angle(geometric, kept, settings)
    if word.angle_status not in LIBRATING:
        return reported
    name = f'angle word {word.angle_status}: sigma librates in the forced-wave runs (the status reads the angle without them)'
    ledger.compare(name, len(waves), Op.GE, 1, '', 'geo_n_forced_wave_episodes')
    return AngleDecision(word.angle_status, word.subtype, reported.rule, reported.episode_id, reported.earlier_episode_id)


def final_regime(
    angle: AngleDecision, geometry: Optional[Geometry], settings: AnalysisSettings, decision: Optional[StatusDecision] = None
) -> Tuple[str, float]:
    """(final_regime, its start [yr]): how the record ends. G1c: circulation from the last turning point; G1e: the
    angle_status of the last regime from its start (the jump); an observed exit after the last librating episode (the
    edge rule, `Episode.exit_observed`): circulation from the exit; otherwise (G1l among them) libration from the start
    of the last librating episode when that episode reaches the record's end (`geometry.covers_edges`); ('', nan) when
    none of these holds
    (the angle_status already describes the end, or the record ends in no regime of its own), and under a word the
    status tree set (`StatusDecision.angle_word`, K2: the drift describes the whole record)."""
    if geometry is None or (decision is not None and decision.angle_word is not None):
        return '', float('nan')
    if angle.rule == 'G1c':
        return str(AngleStatus.CIRCULATION), float(geometry.turning_points.times_yr[-1])
    if other_regime(angle):
        return str(angle.angle_status), float(geometry.regime_start_yr)
    episodes = librating_episodes(geometry, settings)
    if not episodes:
        return '', float('nan')
    last = max(episodes, key=lambda episode: episode.t_end_yr)
    if last.exit_observed and geometry.exit is not None:  # the edge rule: the exit is observed
        return str(AngleStatus.CIRCULATION), float(geometry.exit.time_yr)
    if covers_edges(last, settings)[1]:
        return str(AngleStatus.LIBRATION), last.t_start_yr
    return '', float('nan')


def _ripple_entries(geometry: Geometry, settings: AnalysisSettings, ledger: Ledger) -> None:
    """G2 on the ledger when accepted runs were judged unconfirmed ripples (`ripple_waves`, B6b): the numbers of the one
    with most turning points (half-amplitude, identity) and the slow centre's unfinished swing."""
    found = ripples(geometry)
    if not found:
        return
    wave = max(found, key=lambda w: (w.n_tp, -w.episode_id))
    prefix = f'G2 unconfirmed ripple (episode {wave.episode_id}, P {wave.period_yr:.0f} yr)'
    ledger.compare_setting(
        f'{prefix}: half-amplitude (L4, R0 not measurable below)',
        math.degrees(wave.half_amp_max_rad),
        Op.LT,
        settings,
        'mmr_a_response_min_amp_deg',
        'deg',
        '',
    )
    name = f'{prefix}: identity |slope / c - 1| (fails above, or by its correlation)'
    offset = abs(wave.identity_slope_ratio - 1) if is_finite(wave.identity_slope_ratio) else float('nan')
    ledger.compare_setting(name, offset, Op.LE, settings, 'mmr_identity_slope_tol', '', '')
    ledger.compare_setting(f'{prefix}: identity correlation', wave.identity_corr, Op.GE, settings, 'mmr_identity_corr_min', '', '')
    name = 'G2 ripple on an unfinished slow swing: slow centre turns back inside the record (1 or 2)'
    ledger.compare(name, interior_turns(geometry), Op.GE, 1, '', 'geo_centre_n_tp_interior')
    ledger.compare(G5A_ONE_REVOLUTION, geometry.centre.range_rad, Op.LT, TWO_PI, 'rad', 'geo_centre_range_rad')


def _g2(geometry: Geometry, settings: AnalysisSettings, ledger: Ledger) -> Optional[AngleDecision]:
    _forced_wave_entries(geometry, settings, ledger)
    _ripple_entries(geometry, settings, ledger)
    episodes = librating_episodes(geometry, settings)
    if len(episodes) < len(geometry.episodes):
        fastest = fastest_episode(geometry, settings)
        floor, setting = period_floor(geometry, settings, fastest)
        name = 'G2 episodes not faster than the shortest libration'
        ledger.compare(name, fastest.period_med_yr, Op.GE, floor, 'yr', 'geo_episode_period_min_yr', setting=setting)
        travel = max(episode_centre_travel(geometry, e) for e in geometry.episodes)
        ledger.compare('G2 centre turns inside no episode', travel, Op.LT, TWO_PI, 'rad', 'geo_episode_centre_travel_max_rad')
    if not ledger.compare('G2 accepted episodes', len(episodes), Op.GE, 1, '', 'geo_n_librating_episodes'):
        return None
    episode = decisive_episode(geometry, settings)
    start = _covers_edge(ledger, episode, 'start', settings, geometry)
    end = _covers_edge(ledger, episode, 'end', settings, geometry)
    if start and end:
        subtype = AngleSubtype(str(episode.shape)) if str(episode.shape) else AngleSubtype.STATIONARY
        return AngleDecision(AngleStatus.LIBRATION, subtype, 'G2', episode.episode_id)
    if _regimes_cover(geometry, episodes, settings, ledger):
        rule = 'G2' if geometry.coorbital else RULE_UNION
        return AngleDecision(AngleStatus.LIBRATION, AngleSubtype.REGIME_CHANGE, rule, episode.episode_id)
    subtype = AngleSubtype.RECURRENT if len(episodes) >= 2 else AngleSubtype.NONE
    return AngleDecision(AngleStatus.EPISODIC, subtype, 'G2', episode.episode_id)


def _regimes_cover(geometry: Geometry, episodes: List[Episode], settings: AnalysisSettings, ledger: Ledger) -> bool:
    """Librating episodes that cover the record together (`geometry.covering_union`): no turn in any gap between them
    nor over the whole chain, sigma_f inside the band of its neighbours through every gap (`gap_band_overshoot` <=
    libration_union_band_margin_deg), both record edges reached. A 1:1 quasi-satellite turning into a horseshoe or a
    tadpole is one libration (author 2026-10-19: "a regime change without circulation is not an exit"; 469219, 164207,
    524522); for every other MMR the same geometry
    (W22, TK14 31M-34) is read by the status as episodic (`status_basis`, `promote_union`). On the ledger only with
    several episodes."""
    if len(episodes) < 2:
        return False
    union = librating_union(geometry, settings)  # `episodes` are the librating ones
    name = 'G2 episodes without a circulation between them cover the record'
    if not ledger.compare(name, len(union.episodes), Op.GE, 2, '', 'geo_union_n_episodes'):
        return False
    tp, raw = union.tp_range_rad, union.raw_range_rad
    ledger.compare('G2 union: range of the turning points in a gap', tp, Op.LT, TWO_PI, 'rad', 'geo_union_tp_range_rad')
    ledger.compare('G2 union: range of the raw angle in a gap', raw, Op.LT, TWO_PI, 'rad', 'geo_union_raw_range_rad')
    name = 'G2 union: sigma_f beyond its neighbours\' amplitude in a gap'  # bounded gap (A7, `geometry.gap_band_overshoot`)
    overshoot = math.degrees(union.band_overshoot_rad)
    ledger.compare_setting(name, overshoot, Op.LE, settings, 'libration_union_band_margin_deg', 'deg', 'geo_union_band_overshoot_deg')
    span_tp, span_raw = union.span_tp_range_rad, union.span_raw_range_rad
    ledger.compare('G2 union: range of the turning points over the chain', span_tp, Op.LT, TWO_PI, 'rad', 'geo_union_span_tp_range_rad')
    ledger.compare('G2 union: range of the raw angle over the chain', span_raw, Op.LT, TWO_PI, 'rad', 'geo_union_span_raw_range_rad')
    for edge, shortfall in (('start', union.start_shortfall_p), ('end', union.end_shortfall_p)):
        if is_finite(shortfall):  # the edge `covers_edges` left open, reached by the turning points beyond the episode
            name = f'G2 union: turning points reach the record {edge}, shortfall'
            ledger.compare_setting(name, shortfall, Op.LE, settings, 'libration_coverage_periods', 'P', f'geo_union_{edge}_shortfall_P')
    return True


def edge_observed(geometry: Optional[Geometry], ledger: Ledger) -> bool:
    """The record shows an exit or an entry beyond the librating episodes (the edge rule, `EdgeEvent.observed`): G2c
    then does not look for a slower libration at the scale of the centre, whose low-pass would only hide the arc the
    raw angle runs (2014 OL339 1E-1: an exit of 1.26 rev observed at 55.5 yr, none visible at 189 yr). On the ledger
    only when it holds."""
    for kind in ('exit', 'entry'):
        event = getattr(geometry, kind) if geometry is not None else None
        if event is not None and event.observed:
            name = f'G2c not when the {kind} is observed at the fine scale'
            return not ledger.compare(name, _flag(event.observed), Op.EQ, 0, '', f'{kind}_observed')
    return False


def union_columns(geometry: Optional[Geometry], settings: AnalysisSettings) -> dict:
    """The numbers `_regimes_cover` reads (mmr_numbers.csv): the episodes of the covering union, the largest ranges of
    the turning points and of the orbit-averaged raw angle in one gap and over the chain, the longest gap over
    (P_left + P_right), the largest swing of sigma_f beyond its neighbours' amplitude in a gap [deg], the edge
    shortfalls (0 / nan without a union); and the turns of the slow centre G5a reads (`interior_turns`)."""
    union = librating_union(geometry, settings) if geometry is not None else UnionCover()
    return {
        'geo_union_n_episodes': len(union.episodes),
        'geo_union_tp_range_rad': union.tp_range_rad,
        'geo_union_raw_range_rad': union.raw_range_rad,
        'geo_union_span_tp_range_rad': union.span_tp_range_rad,
        'geo_union_span_raw_range_rad': union.span_raw_range_rad,
        'geo_union_gap_max_P': union.gap_max_p,
        'geo_union_band_overshoot_deg': math.degrees(union.band_overshoot_rad),
        'geo_union_start_shortfall_P': union.start_shortfall_p,
        'geo_union_end_shortfall_P': union.end_shortfall_p,
        'geo_centre_n_tp_interior': interior_turns(geometry) if geometry is not None else 0,
    }


def edge_arc_name(kind: str) -> str:
    """The ledger name of an observed exit/entry (`kind`): the raw angle runs on beyond the episode (`_covers_edge`)."""
    return f'G2 {kind} observed: the raw angle runs on beyond the episode'


def edge_monotone_name(kind: str) -> str:
    """The ledger name of an observed exit/entry (`kind`): the raw angle moves monotonically (`_covers_edge`)."""
    return f'G2 {kind} observed: the raw angle moves monotonically'


def _covers_edge(ledger: Ledger, episode: Episode, edge: str, settings: AnalysisSettings, geometry: Optional[Geometry] = None) -> bool:
    """`geometry.covers_edges` for one edge ('start' or 'end'), on the ledger: not when the exit (entry) beyond the
    episode is observed (the edge rule: the orbit-averaged raw angle runs >= libration_edge_arc_rev turns away and
    monotonically, `geometry._visible_edge`); else the gap from the record edge within
    libration_coverage_periods periods, else no turning point missed at the analysis window's edge (that gap not longer
    than the longest TP-to-TP step of the episode, nor than libration_coverage_periods periods:
    `geometry.window_step_limit_yr`) and the stretch the trim hides within libration_coverage_periods."""
    kind = 'entry' if edge == 'start' else 'exit'
    event = getattr(geometry, kind) if geometry is not None else None
    if getattr(episode, f'{kind}_observed') and event is not None:
        name = edge_arc_name(kind)
        ledger.compare_setting(name, event.raw_arc_rad / TWO_PI, Op.GE, settings, 'libration_edge_arc_rev', 'rev', f'{kind}_raw_arc_rev')
        ledger.compare(edge_monotone_name(kind), _flag(event.raw_monotonic), Op.EQ, 1, '', f'{kind}_raw_monotonic')
        return False
    gap_p = getattr(episode, f'coverage_{edge}_gap_p')
    if ledger.compare_setting(
        f'G2 coverage gap at the {edge}', gap_p, Op.LE, settings, 'libration_coverage_periods', 'P', f'ep_coverage_{edge}_gap_P'
    ):
        return True
    gap_yr, hidden_p = getattr(episode, f'window_{edge}_gap_yr'), getattr(episode, f'window_{edge}_hidden_p')
    name = f'G2 no turning point missed at the window {edge}'
    if not ledger.compare(name, gap_yr, Op.LE, window_step_limit_yr(episode, settings), 'yr', f'ep_window_{edge}_gap_yr'):
        return False
    name = f'G2 the trim hides little at the {edge}'
    return bool(ledger.compare_setting(name, hidden_p, Op.LE, settings, 'libration_coverage_periods', 'P', f'ep_window_{edge}_hidden_P'))


def _g3(geometry: Geometry, energy: Optional[EnergyAnalysis], settings: AnalysisSettings, ledger: Ledger) -> Optional[AngleDecision]:
    """Patterns near the separatrix; they need the sigma-only potential (a geometric property of the angle)."""
    if energy is None or energy.sigma_only != CheckValue.PASS:
        return None
    ledger.compare_setting(
        'G3 sigma-only: well coherence',
        energy.well_coherence,
        Op.GE,
        settings,
        'separatrix_phase_coherence',
        '',
        'sep_well_coherence',
    )
    n_slips = sum(1 for slip in geometry.slips if not slip.suspect)
    if ledger.compare_setting('G3 pattern A: slips', n_slips, Op.GE, settings, 'libration_pattern_a_min_slips', '', 'geo_n_slips'):
        amplitude = math.degrees(tp_half_amplitude_median(geometry))
        name = 'G3 pattern A: pieces reach the separatrix (A_med, no zone)'
        if ledger.compare_setting(name, amplitude, Op.GE, settings, 'libration_pattern_a_min_amp_deg', 'deg', 'geo_tp_half_amp_med_deg'):
            return AngleDecision(AngleStatus.NEAR_SEPARATRIX, AngleSubtype.PATTERN_A, 'G3')
    pattern = geometry.pattern_b
    if pattern is None:
        return None
    stalls = ledger.compare_setting(
        'G3 pattern B: stalls at the top',
        pattern.stall_fraction,
        Op.GE,
        settings,
        'libration_stall_rev_fraction',
        '',
        'geo_stall_fraction',
    )
    if stalls and pattern.detected:
        return AngleDecision(AngleStatus.NEAR_SEPARATRIX, AngleSubtype.PATTERN_B, 'G3')
    return None


def _g4_to_g6(
    sampling: Sampling, geometry: Geometry, settings: AnalysisSettings, ledger: Ledger, slow_swing: Optional[SlowSwing] = None
) -> AngleDecision:
    alias = sampling.verdict in ALIASED or sampling.alias_numpy_differs
    rule = centre_circulates(geometry, settings, ledger, 'G4 largest centre pass', G4B_NET)
    if rule:
        return AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.ALIAS if alias else AngleSubtype.NONE, rule)
    name = 'G5 drift: centre_moves runs cover the window'
    if ledger.compare_setting(
        name, geometry.drift_fraction, Op.GE, settings, 'libration_drift_min_window_fraction', '', 'geo_drift_fraction'
    ):
        if _g5a(geometry, ledger):
            return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, 'G5a')
        if _g5w(slow_swing, settings, ledger):
            return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, RULE_G5W)
        return AngleDecision(AngleStatus.DRIFT, AngleSubtype.NONE, 'G5')
    if _g5w(slow_swing, settings, ledger):  # the turning points are waves on a slow swing in the well (523683)
        return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, RULE_G5W)
    if ripples(geometry):
        return _g6_ripple(geometry, ledger)
    if ledger.compare('G6 turning points without an episode', len(geometry.turning_points), Op.GE, 1, '', 'geo_n_tp'):
        return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, 'G6')
    arc_range = geometry.arc_stall.range_rad if geometry.arc_stall is not None else float('nan')
    if ledger.compare(G6_RANGE, arc_range, Op.GE, math.pi, 'rad', 'geo_arc_range_rad'):
        return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, 'G6')
    return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.STALL, 'G6')


def _g6_ripple(geometry: Geometry, ledger: Ledger) -> AngleDecision:
    """G6 when unconfirmed ripples were set aside (`ripple_waves`, B6b): turning points outside them are swings of the
    angle (few_cycles, as G6); without any, the angle's own motion is the slow centre's unfinished swing: G5a's arc
    when it turns back once (`_g5a`), else an arc or a stall by the slow centre's range (pi, as G6 reads sigma_f's)."""
    name = 'G6 turning points without an episode, outside the unconfirmed ripples'
    if ledger.compare(name, tp_outside_ripples(geometry), Op.GE, 1, '', ''):
        return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, 'G6')
    if _g5a(geometry, ledger):
        return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, 'G5a')
    if ledger.compare(
        'G6 range of the slow centre (ripples set aside)', geometry.centre.range_rad, Op.GE, math.pi, 'rad', 'geo_centre_range_rad'
    ):
        return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.ARC, 'G6')
    return AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.STALL, 'G6')


def centre_circulates(geometry: Optional[Geometry], settings: AnalysisSettings, ledger: Ledger, pass_name: str, net_name: str) -> str:
    """G4/G4b, the slow centre circulates: 'G4' when its largest monotone pass reaches a revolution, 'G4b' when its net
    shift does (`centre_net_circulates`), else ''; the comparisons named `pass_name` and `net_name` (G4 and K2's word)."""
    passed = geometry.centre.max_pass_rad if geometry is not None else float('nan')
    if ledger.compare(pass_name, passed, Op.GE, TWO_PI, 'rad', 'geo_centre_max_pass_rad'):
        return 'G4'
    return 'G4b' if centre_net_circulates(geometry, settings, ledger, net_name) else ''


def centre_net_circulates(geometry: Optional[Geometry], settings: AnalysisSettings, ledger: Ledger, name: str) -> bool:
    """G4b (W22 R-NET): the slow centre advances by |net_rev| >= libration_circulation_net_rev revolutions over the
    window, a circulation even when its largest monotone pass stays below one revolution (a staircase with small
    step-backs: 142694 1V+3S-7 over 200 kyr, 2.75 rev net, 0.9 rev at most in one pass). A missing number decides
    nothing."""
    net = geometry.centre.net_rev if geometry is not None else float('nan')
    value = abs(net) if is_finite(net) else None
    return bool(ledger.compare_setting(name, value, Op.GE, settings, 'libration_circulation_net_rev', 'rev', 'geo_centre_net_rev'))


def interior_turns(geometry: Geometry) -> int:
    """Turns of the slow centre (`Geometry.centre.turns`) farther than the centre's own smoothing window W_c from both
    edges of the analysis window: nearer, the low-pass of the centre bends at the edge (2003 UG22 8M-4S-7: a 'turn' at
    185 yr of 20 kyr, W_c = 660 yr)."""
    return len(interior_turn_times(geometry))


def interior_turn_times(geometry: Geometry) -> np.ndarray:
    """The times [yr] of the turns counted by `interior_turns`, in order."""
    centre = geometry.centre
    times = np.sort(np.asarray(centre.turns.times_yr, dtype=float))
    start, end = geometry.window
    return times[(times - start > centre.window_yr) & (end - times > centre.window_yr)]


def _g5a(geometry: Geometry, ledger: Ledger) -> bool:
    """G5a, a turning arc, not a drift (K1, author 2026-10-19): the slow centre (`Geometry.centre`) turns back exactly
    once away from the edges (`interior_turns`) and its whole range stays below one revolution: the record holds part
    of one slow swing, so it is too short to tell a drift from a slow libration (2003 KQ18 6J+2S-1: one turn at 8.9
    kyr, 0.18 rev, the e-omega cycle of 22.7 kyr covered 0.88 times). With two turns the record holds a whole half-swing
    between them and stays a drift (the author's drift of 101703 and 142694 1V+3S-7: turns at 5.3/11.5 and 12.9/21.1
    kyr in 50 kyr); a turn within W_c of an edge is the edge of the centre's own low-pass (W20 deviations from the
    plan's "at least one turn")."""
    if not ledger.compare(
        'G5a slow centre turns back once inside the record', interior_turns(geometry), Op.EQ, 1, '', 'geo_centre_n_tp_interior'
    ):
        return False
    return bool(ledger.compare(G5A_ONE_REVOLUTION, geometry.centre.range_rad, Op.LT, TWO_PI, 'rad', 'geo_centre_range_rad'))


def _g5w(swing: Optional[SlowSwing], settings: AnalysisSettings, ledger: Ledger) -> bool:
    """G5w (W28 B6c, cards g07-5/g07-6 A, rule `slow_swing_in_well`): the slow centre turned back twice within one
    revolution and the half-swing between the turns is a swing in the resonance's own well (`slow_swing.holds`): the
    slow identity passes, and the a swing and the period 2 (t2 - t1) agree with the directly averaged pendulum at the
    half-swing's amplitude within [1/sqrt(x), sqrt(x)], x = MMR_STRENGTH_CHECK_RATIO. The record holds part of one slow
    libration of this resonance: too short to tell, not a drift (523683 2U-9: turns at 11.9 and 67.7 kyr, identity
    1.00, a 1.07, period 1.00). Unmeasured (one turn, three turns or more, 1:1, three-body: no direct theory) -> False
    and the row reads as before: the author's drift of 101703/142694 1V+3S-7 (two turns, three-body, slow identity
    slope / c -0.27/-0.25) stays. On the ledger only when measured."""
    if swing is None or not swing.measured:
        return False
    offset = abs(swing.identity_slope_ratio - 1) if is_finite(swing.identity_slope_ratio) else None
    column = 'geo_slow_swing_identity_slope_ratio'
    held = bool(ledger.compare_setting(G5W_IDENTITY_SLOPE, offset, Op.LE, settings, 'mmr_identity_slope_tol', '', column))
    corr, column = swing.identity_corr, 'geo_slow_swing_identity_corr'
    held &= bool(ledger.compare_setting(G5W_IDENTITY_CORR, corr, Op.GE, settings, 'mmr_identity_corr_min', '', column))
    if not held:
        return False
    bound = math.sqrt(settings.mmr_strength_check_ratio)
    ratios = (
        (swing.a_ratio, G5W_A_LOW, G5W_A_HIGH, 'geo_slow_swing_a_ratio'),
        (swing.period_ratio, G5W_P_LOW, G5W_P_HIGH, 'geo_slow_swing_period_ratio'),
    )
    for ratio, low, high, column in ratios:
        held &= bool(ledger.compare(low, ratio, Op.GE, 1 / bound, '', column, setting='mmr_strength_check_ratio'))
        held &= bool(ledger.compare(high, ratio, Op.LE, bound, '', column, setting='mmr_strength_check_ratio'))
    return held and slow_swing_module.holds(swing, settings)


def decide_angle(
    sampling: Sampling,
    unphysical: bool,
    geometry: Optional[Geometry],
    energy: Optional[EnergyAnalysis],
    settings: AnalysisSettings,
    ledger: Ledger,
    slow_swing: Optional[SlowSwing] = None,
) -> AngleDecision:
    """G0–G6 above (G1l after G2: a capture after a jump of a; G1e after G3–G6: episodes before the last jump of a,
    for the status only); `slow_swing` holds G5w's numbers (`slow_swing.measure_slow_swing`; None: G5w unmet)."""
    undefined = _g0(sampling, unphysical, geometry, settings, ledger)
    if undefined is not None:
        return undefined
    if Flag.LAST_REGIME_AFTER_JUMP in geometry.flags:
        name = 'G1 window = last regime after a jump of a'
        ledger.compare(name, geometry.regime_start_yr, Op.GE, geometry.window[0], 'yr', 'regime_start_yr')
    final = _g1c(geometry, settings, ledger)
    if final is not None:
        return final
    episodes = _g2(geometry, settings, ledger)
    if episodes is not None:
        return _g1l(episodes, geometry, settings, ledger)
    pattern = _g3(geometry, energy, settings, ledger)
    if pattern is not None:
        return _g1e(pattern, geometry, settings, ledger)
    return _g1e(_g4_to_g6(sampling, geometry, settings, ledger, slow_swing), geometry, settings, ledger)


# --------------------------------------------------------------------------------------------- status


def zone_rule(
    angle: AngleDecision, geometry: Geometry, energy: Optional[EnergyAnalysis], settings: AnalysisSettings, ledger: Ledger
) -> Tuple[Status, Reason, str, List[str]]:
    """The dynamic leaf (DELTA 13b): the observed behaviour decides; the zone and the cycle energies only add flags.

    libration -> resonant (flag libration_in_separatrix_zone when a cycle lies in the zone), episodic -> transient,
    near_separatrix -> near_separatrix (the a-response was checked by the tree before this leaf). The flag
    episodes_in_separatrix_zone (crossings, every cycle inside the model's zone) goes with any of them; it changes no
    status of this leaf (52356 stays transient, DECISIONS 2026-10-15 evening). Elsewhere the classes are not columns
    only: they set L2 and Z, the cap episodes_in_separatrix_zone, O1's applicability, and withhold the onset objection
    of leaf 5o (uncertain/origin_unknown) when the episodes lie in the zone (W25 review).
    """
    flags = []
    zone_cycles = energy is not None and has_zone_cycle(energy.cycle_classes)
    if episodes_in_separatrix_zone(energy):
        name = 'flag episodes_in_separatrix_zone: crossings with every cycle in the zone'
        ledger.compare(name, energy.n_outside, Op.EQ, 0, '', 'sep_n_outside')
        flags.append(EPISODES_IN_SEPARATRIX_ZONE)
    if angle.angle_status == AngleStatus.LIBRATION:
        if zone_cycles:
            flags.append(str(Flag.LIBRATION_IN_SEPARATRIX_ZONE))
        return Status.RESONANT, Reason.DYNAMIC_LIBRATION, '6', flags
    if angle.angle_status == AngleStatus.EPISODIC:
        return Status.TRANSIENT, Reason.DYNAMIC_EPISODE, '6', flags
    return Status.NEAR_SEPARATRIX, Reason.SEPARATRIX_ZONE, '6', flags


def near_separatrix_energy(energy: Optional[EnergyAnalysis], settings: AnalysisSettings, ledger: Ledger) -> bool:
    """Flag of a circulation (SPEC §5.4 R6): the median turn lies above the zone's upper edge h_hi, but within
    separatrix_near_zone_widths zone widths (h_hi - h_lo) of it (`EnergyAnalysis.turn_zone_widths`). A column only."""
    if energy is None or not np.isfinite(energy.turn_zone_widths):
        return False
    name = 'flag near_separatrix_energy: turns above the zone, in zone widths'
    return bool(
        ledger.compare_setting(name, energy.turn_zone_widths, Op.LE, settings, 'separatrix_near_zone_widths', '', 'sep_turn_zone_widths')
    )


def _check_entry(
    ledger: Ledger, name: str, check: CheckValue, value, op: Op, settings: AnalysisSettings, setting: str, column: str, named: bool = True
):
    """Record the number behind a check (a not-measured entry when the check is neither pass nor fail); `named` False:
    a number the check did not rest on (no convention, never a thin margin)."""
    ledger.compare_setting(name, value if check in MEASURED else None, op, settings, setting, '', column, named=named)


def _undefined(angle: AngleDecision) -> StatusDecision:
    reason = {
        AngleSubtype.ALIAS: Reason.ALIAS,
        AngleSubtype.UNPHYSICAL: Reason.UNPHYSICAL,
        AngleSubtype.UNDEFINED_PHASE: Reason.UNDEFINED_PHASE,
    }[angle.subtype]
    return StatusDecision(Status.INVALID, reason, LibrationOrigin.NONE, '', '', rule='G0')


def _subresonance_origin(subresonances: SubresonanceResult) -> Tuple[str, str]:
    detail = OriginDetail.COMPOSITE_MULTIPLET if subresonances.composite else OriginDetail.FIXED_SUBRESONANCE
    return str(detail), subresonances.accepted.spec.key


COMPOSITE_REGIME_ORIGINS = (LibrationOrigin.DYNAMIC, LibrationOrigin.UNKNOWN)  # own librations a composite regime names
CLAIMING_STATUSES = (Status.RESONANT, Status.TRANSIENT, Status.NEAR_SEPARATRIX)


def composite_regime_detail(decision: StatusDecision, subresonances: Optional[SubresonanceResult]) -> None:
    """W28 A6b, card g06-8c A ('composite' is an allowed answer): the scan read the decisive episode as a composite
    regime (`SubresonanceResult.composite_regime`: nothing admissible, no planar member's theoretical centre stands over
    the episode, the measured centre follows the composite one pi - arg P(beta)). A libration verdict of the catalogue
    angle's own (origin dynamic or unknown, no detail yet) then names origin_detail composite_multiplet, and its
    resonant argument is 'composite' (`roles.verdict_roles`). In place; no status, reason or level changes."""
    if subresonances is None or not subresonances.composite_regime or decision.origin_detail:
        return
    if decision.status in CLAIMING_STATUSES and decision.origin in COMPOSITE_REGIME_ORIGINS:
        decision.origin_detail = str(OriginDetail.COMPOSITE_MULTIPLET)


def owner_in_episode(
    owner: Optional[OwnerResult], settings: AnalysisSettings, ledger: Ledger, decisive: bool = True
) -> Tuple[Optional[OwnerResult], List[str]]:
    """F7 per episode and P2 (W20; A1, W28, cards g04-0..g04-3): an accepted owner takes an episode of sigma only when
    its intervals (its measured regimes, no bridge) cover at least mmr_owner_cover_min of it and, in its libration
    regime, Psi/u turns less than mmr_owner_psi_turns_per_cycle_max per cycle of it (`EpisodeCover.induced`); every
    accepted episode is checked (`OwnerResult.episode_covers`; without them the decisive episode alone). The decisive
    episode's cover (`OwnerResult.cover`, column own_cover) and turns (own_psi_turns_per_cycle) go on the ledger.

    (the owner, []) when it induces every episode still in the geometry (rule 2), or when its cover is not measured (no
    accepted owner, or a result without intervals); (None, flags) otherwise: the row's own episodes decide, the tree
    goes on without the owner and `_decide_status` keeps its key in the verdict's `owner` column, flag
    owner_outside_episode for an own episode the owner does not cover, owner_psi_too_fast for one it covers but whose
    Psi/u turns too fast. A row with induced and own episodes reaches this with the induced ones out of the geometry
    (the analysis' second pass, `OwnerResult.induced_ids`). The route, the phase lever and the potential scale of the
    accepted owner are recorded next to it. `decisive` False (a circulation: the owner only names the origin of a
    non_resonant verdict, 237743 5J-2S-2) records these comparisons without their settings: no convention the verdict
    rests on, never a thin margin U."""
    if owner is None or owner.accepted is None:
        return owner, []
    accepted = owner.accepted
    ledger.compare(f'2 owner route ({accepted.route or "r2"})', 1.0 if accepted.route else None, Op.EQ, 1, '', 'own_route')

    def record(name, value, op, setting, column):
        return ledger.compare_setting(name, value, op, settings, setting, '', column, named=decisive)

    if accepted.route == GAIN_PHASE and is_finite(accepted.lever_owner):
        gain = accepted.phi0_coherence - accepted.lever_owner
        record('2 phase lever: C_B - L_owner', gain, Op.GE, 'mmr_owner_phi0_coh_gain', 'own_lever_gain')
    potential = accepted.potential
    if accepted.route == GAIN_POTENTIAL and potential is not None and potential.measured:
        worst = max(potential.scale_hi, 1.0 / potential.scale_lo) if potential.scale_lo > 0 else float('inf')
        record('2 potential scale: max(s, 1/s)', worst, Op.LE, 'mmr_owner_potential_scale_max', 'own_pot_scale_worst')
    if not is_finite(owner.cover):
        return owner, []
    covers = bool(record(OWNER_COVER, owner.cover, Op.GE, 'mmr_owner_cover_min', 'own_cover'))
    slow = True
    if is_finite(owner.psi_turns_per_cycle):
        turns = owner.psi_turns_per_cycle
        slow = bool(record(OWNER_PSI_TURNS, turns, Op.LT, 'mmr_owner_psi_turns_per_cycle_max', 'own_psi_turns_per_cycle'))
    induced, own = induced_split(owner, settings)
    if not induced and not own:  # no per-episode covers: the decisive episode alone
        return (owner, []) if covers and slow else (None, _refusal_flags(not covers, covers and not slow))
    if not own:
        return owner, []
    outside = any(not entry.cover >= settings.mmr_owner_cover_min for entry in own)
    fast = any(entry.cover >= settings.mmr_owner_cover_min for entry in own)  # an own episode it covers: P2 refused it
    return None, _refusal_flags(outside, fast)


def _refusal_flags(outside: bool, too_fast: bool) -> List[str]:
    """The flags of an owner that does not take the row (`owner_in_episode`), in this order."""
    return [str(flag) for flag, holds in ((Flag.OWNER_OUTSIDE_EPISODE, outside), (Flag.OWNER_PSI_TOO_FAST, too_fast)) if holds]


def owner_rules(owner: Optional[OwnerResult], flags: Sequence[str], settings: AnalysisSettings) -> frozenset:
    """The rules the owner check acted with (`analysis.rules`): owner_per_episode where A1 changed the outcome (the
    induced episodes a second pass took out, or an owner refused for lack of cover that the old slip bridge would have
    given the decisive episode, `owner.bridged_cover`; an owner that acts elsewhere either way, 1627, does not fire
    it), owner_psi_turns when P2 refused one."""
    fired: frozenset = frozenset()
    if owner is None or owner.accepted is None:
        return fired
    outside = str(Flag.OWNER_OUTSIDE_EPISODE) in flags
    bridged = outside and bridged_cover(owner.intervals, owner.decisive_window) >= settings.mmr_owner_cover_min
    if owner.induced_ids or bridged:
        fired = rules_module.fire(fired, rules_module.Rule.OWNER_PER_EPISODE)
    if str(Flag.OWNER_PSI_TOO_FAST) in flags:
        fired = rules_module.fire(fired, rules_module.Rule.OWNER_PSI_TURNS)
    return fired


def second_capture_caps(owner: Optional[OwnerResult], settings: AnalysisSettings, ledger: Ledger) -> Tuple[List[str], List[str]]:
    """(flags, caps) of an induced libration: second_capture_unresolved (cap medium) unless the owner's second-capture
    test excludes it by one of its two routes (`owner.second_capture`): x = Psi / u sweeps >= mmr_owner_capture_sweep_deg
    and the energy R^2 < mmr_owner_r2_min, or (F3) x_ddot follows the secular cycle (R^2 >= mmr_owner_r2_min) and not
    x (R^2 on [1, sin x, cos x] < mmr_owner_capture_self_r2_max). The F3 comparisons are recorded with their settings
    only when F3 is the route that excluded (otherwise they decided nothing and never make a thin margin)."""
    capture = owner.capture if owner is not None else None
    sweep = math.degrees(capture.x_sweep_rad) if capture is not None else None
    ledger.compare_setting('2 second capture: x sweep', sweep, Op.GE, settings, 'mmr_owner_capture_sweep_deg', 'deg', 'own_psi_sweep_deg')
    energy = capture.energy_r2 if capture is not None else None
    ledger.compare_setting('2 second capture: energy R2', energy, Op.LT, settings, 'mmr_owner_r2_min', '', 'own_psi_energy_R2')
    if capture is not None and np.isfinite(capture.secular_r2):
        named = capture.route == ROUTE_SECULAR
        ledger.compare_setting(
            '2 second capture: secular R2', capture.secular_r2, Op.GE, settings, 'mmr_owner_r2_min', '', 'own_psi_secular_R2', named=named
        )
        ledger.compare_setting(
            '2 second capture: self R2',
            capture.self_r2,
            Op.LT,
            settings,
            'mmr_owner_capture_self_r2_max',
            '',
            'own_psi_self_R2',
            named=named,
        )
    if capture is not None and capture.verdict == CAPTURE_EXCLUDED:
        return [], []
    return [str(Flag.SECOND_CAPTURE_UNRESOLVED)], [str(Cap.SECOND_CAPTURE_UNRESOLVED)]


def second_capture_rules(owner: Optional[OwnerResult]) -> frozenset:
    """second_capture_f3 when the route F3 removed the cap second_capture_unresolved (rule 2)."""
    capture = owner.capture if owner is not None else None
    if capture is not None and capture.verdict == CAPTURE_EXCLUDED and capture.route == ROUTE_SECULAR:
        return rules_module.fire(frozenset(), rules_module.Rule.SECOND_CAPTURE_F3)
    return frozenset()


def _foreign_origin(subresonances: Optional[SubresonanceResult], owner: Optional[OwnerResult]) -> Tuple[LibrationOrigin, str, str]:
    """(origin, detail, owner key) when an accepted subresonance or owner carries the oscillation."""
    if subresonances is not None and subresonances.accepted is not None:
        detail, key = _subresonance_origin(subresonances)
        return LibrationOrigin.SUBRESONANCE, detail, key
    if owner is not None and owner.accepted is not None:
        return LibrationOrigin.INDUCED, owner.detail, owner.accepted.spec.key
    return LibrationOrigin.NONE, '', ''


def _circulation(
    angle: AngleDecision,
    sampling: Sampling,
    geometry: Geometry,
    energy: Optional[EnergyAnalysis],
    subresonances: Optional[SubresonanceResult],
    owner: Optional[OwnerResult],
    settings: AnalysisSettings,
    ledger: Ledger,
    member: Optional[Verdict] = None,
) -> StatusDecision:
    """C5 for an alias; an accepted subresonance with its own verdict gives the member's status (`member_status`);
    R0 for a circulation that turns back at times, else circulation (+ near_separatrix_energy).

    Alias (DECISIONS 2026-10-15): non_resonant only when the branch of nu dt is unique over the a range AND the rate
    expected from <a> is far from the resonance (`sampling.far_ratio` >= mmr_c5_far_ratio_min: |<a> - a*| over the a
    range of the record or |a(lambda_dot) - <a>|, from the data and independent of the unwrap); several branches, or
    a unique branch near the resonance, cannot exclude a libration hidden by the step: invalid/alias + resample."""
    if angle.subtype == AngleSubtype.ALIAS:
        unique = ledger.compare(C5_BRANCH, _flag(sampling.verdict == StepVerdict.AMBIGUOUS), Op.EQ, 0, '', 'smp_verdict')
        far = ledger.compare_setting(C5_FAR, sampling.far_ratio, Op.GE, settings, 'mmr_c5_far_ratio_min', '', 'smp_far_ratio')
        if unique and far:
            return StatusDecision(Status.NON_RESONANT, Reason.ALIAS, LibrationOrigin.NONE, '', '', rule='C5')
        return StatusDecision(Status.INVALID, Reason.ALIAS, LibrationOrigin.NONE, '', '', rule='C5')
    shared = member_status(angle, subresonances, member, ledger)
    if shared is not None:
        return shared
    origin, detail, owner_key = _foreign_origin(subresonances, owner)
    potential = energy.with_slow if energy is not None else CheckValue.NOT_MEASURED
    if len(geometry.turning_points) > 0 and potential == CheckValue.FAIL:
        name = 'R0 V_E ~ V_F windows'
        ledger.compare_setting(
            name, energy.ve_vf_agree_fraction, Op.GE, settings, 'separatrix_vevf_fraction', '', 'sep_VE_VF_agree_fraction'
        )
        return StatusDecision(Status.NON_RESONANT, Reason.NO_RESONANT_POTENTIAL, origin, detail, owner_key, rule='R0')
    flags = [str(Flag.NEAR_SEPARATRIX_ENERGY)] if near_separatrix_energy(energy, settings, ledger) else []
    return StatusDecision(Status.NON_RESONANT, Reason.CIRCULATION, origin, detail, owner_key, flags=flags, rule='C')


def members_locked(angle: AngleDecision, member: Optional[Verdict]) -> float:
    """1 when the catalogue angle (as the status reads it) and the accepted member's own row both librate over the
    record, 0 when only the member's does; nan when the catalogue angle does not librate or there is no member row."""
    if member is None or status_basis(angle).angle_status != AngleStatus.LIBRATION:
        return float('nan')
    return _flag(member.angle_status == AngleStatus.LIBRATION)


def member_status(
    angle: AngleDecision, subresonances: Optional[SubresonanceResult], member: Optional[Verdict], ledger: Ledger
) -> Optional[StatusDecision]:
    """Rule C for a multiplet (W7 'ug22'): the members of one commensurability are one resonance (Nesvorny & Morbidelli
    1998; Morbidelli 2002 §9.3), and the body may reside in another component of the multiplet. When an accepted member
    (`subresonances.accepted`) has its own verdict (`member`, the derived row), the catalogue row takes the member's
    status (resonant, transient, near_separatrix, uncertain, non_resonant; an invalid member leaves it uncertain) with
    reason subresonance_librates, origin subresonance and owner = the member; its angle_status stays its own. Both
    angles may librate at once (their difference is then bounded: apsidal corotation, Beaugé et al. 2003): when the
    catalogue angle (as the status reads it) and the member both librate over the record, the catalogue row keeps
    its own verdict (None; the ledger entry says so). None as well without an accepted member or its verdict (a nodal
    member has no derived row: the old leaf, `_foreign`)."""
    if member is None or subresonances is None or subresonances.accepted is None:
        return None
    locked = members_locked(angle, member)
    if is_finite(locked):
        name = '1 the catalogue angle and the accepted member librate together (both keep their verdicts)'
        if ledger.compare(name, locked, Op.EQ, 1, '', 'sub_members_locked'):
            return None
    detail, key = _subresonance_origin(subresonances)
    status = Status.UNCERTAIN if member.status == Status.INVALID else Status(member.status)
    return StatusDecision(status, Reason.SUBRESONANCE_LIBRATES, LibrationOrigin.SUBRESONANCE, detail, key, rule='1')


def _foreign(
    angle: AngleDecision,
    subresonances: Optional[SubresonanceResult],
    owner: Optional[OwnerResult],
    settings: AnalysisSettings,
    ledger: Ledger,
    member: Optional[Verdict] = None,
) -> Optional[StatusDecision]:
    """Steps 1 and 2: an accepted subresonance or owner carries the oscillation. Step 1 gives the member's status
    (`member_status`), or non_resonant/subresonance_librates when the member has no verdict of its own; when the
    catalogue angle and the member both librate, the subresonance does not take the libration and step 2 decides.
    Step 2 (A2b, `joint_capture`): 'both resonances at once?' confirmed -> None (the tree reads the angle as its own and
    `joint_capture_origin` names it origin joint); confirmed on one cycle of x only -> uncertain/baseline_needed, origin
    joint, rule 2j; otherwise non_resonant/induced_libration."""
    if subresonances is not None:
        best = subresonances.accepted or min(subresonances.candidates, key=lambda c: c.rank, default=None)
        value = best.phi0_coherence if best is not None else None
        ledger.compare_setting(
            '1 subresonance phi0 coherence', value, Op.GE, settings, 'mmr_subresonance_phi0_coh_min', '', 'sub_best_phi0_coherence'
        )
    if owner is not None:
        record_decisive(owner, settings, ledger)  # each route records its own comparison
    origin, detail, key = _foreign_origin(subresonances, owner)
    if origin == LibrationOrigin.SUBRESONANCE:
        if member is None:
            return StatusDecision(Status.NON_RESONANT, Reason.SUBRESONANCE_LIBRATES, origin, detail, key, rule='1')
        shared = member_status(angle, subresonances, member, ledger)
        if shared is not None:
            return shared
        origin, detail, key = _foreign_origin(None, owner)
    if origin == LibrationOrigin.INDUCED:
        joint = joint_capture(owner, settings, ledger)
        if joint == CAPTURE_ONE_CYCLE:  # the own well shows, on one cycle only: no reading either way yet
            decision = StatusDecision(
                Status.UNCERTAIN, Reason.BASELINE_NEEDED, LibrationOrigin.JOINT, str(OriginDetail.JOINT_CAPTURE), key, rule='2j'
            )
            decision.rules_fired = rules_module.fire(decision.rules_fired, rules_module.Rule.BOTH_RESONANCES_CONFIRMED)
            return decision
        if joint == CAPTURE_CONFIRMED:  # both resonances at once: the tree reads the angle as its own (`_decide_read`)
            return None
        flags, caps = second_capture_caps(owner, settings, ledger)
        decision = StatusDecision(Status.NON_RESONANT, Reason.INDUCED_LIBRATION, origin, detail, key, flags=flags, caps=caps, rule='2')
        decision.rules_fired = decision.rules_fired | second_capture_rules(owner)
        return decision
    return None


# A2b (card g05-8 B): the ledger names of the five conditions of 'both resonances at once? confirmed'
JOINT_SWEEP = "2j both resonances at once: x sweep < mmr_owner_capture_sweep_deg"
JOINT_ENERGY = '2j both resonances at once: energy R2 of x'
JOINT_SELF = '2j both resonances at once: R2 of x_ddot on sin x (its own well)'
JOINT_OFFSET = '2j both resonances at once: |P_x / P_secular - 1| (all secular periods and harmonics)'
JOINT_CYCLES = '2j both resonances at once: closed cycles of x'


def joint_capture(owner: Optional[OwnerResult], settings: AnalysisSettings, ledger: Ledger) -> str:
    """The third outcome of the second capture (A2b W28, card g05-8 B; `owner.joint_outcome`): CAPTURE_CONFIRMED when
    x = Psi / u is bounded, conserves an energy, has a well of its own at a period that is no secular one and closes at
    least libration_centre_min_estimates cycles; CAPTURE_ONE_CYCLE when it closes one; '' otherwise. The five
    comparisons go on the ledger with their settings only when the outcome is one of these two (elsewhere they decided
    nothing: no thin margin)."""
    capture = owner.capture if owner is not None else None
    if capture is None or capture.route:
        return ''
    outcome = capture.verdict if capture.verdict in (CAPTURE_CONFIRMED, CAPTURE_ONE_CYCLE) else ''
    named = bool(outcome)

    def record(name, value, op, setting, column):
        ledger.compare_setting(name, value, op, settings, setting, '', column, named=named)

    record(JOINT_SWEEP, math.degrees(capture.x_sweep_rad), Op.LT, 'mmr_owner_capture_sweep_deg', 'own_psi_sweep_deg')
    record(JOINT_ENERGY, capture.energy_r2, Op.GE, 'mmr_owner_r2_min', 'own_psi_energy_R2')
    record(JOINT_SELF, capture.self_r2, Op.GE, 'mmr_owner_r2_min', 'own_psi_self_R2')
    record(JOINT_OFFSET, capture.x_secular_offset, Op.GT, 'mmr_kin_period_tol', 'own_x_secular_offset')
    record(JOINT_CYCLES, capture.x_cycles, Op.GE, 'libration_centre_min_estimates', 'own_x_cycles')
    return outcome


def joint_capture_origin(decision: StatusDecision, ledger: Ledger) -> None:
    """A2b: the angle's own reading under a confirmed joint capture (`StatusDecision.joint_owner`) takes origin joint,
    origin_detail joint_capture and the owner's key, when it reads the libration as the angle's own
    (`own_dynamics_claimed`: the status and the reason stay the tree's); rule both_resonances_confirmed. Any other
    reading (kinematic, implausible strength, uncertain, a circulation) stands as it is: the joint capture names no
    mechanism the tree did not find. In place."""
    if not decision.joint_owner:
        return
    name = '2j the own reading under a joint capture claims the libration (claims = 1)'
    if not ledger.compare(name, _flag(own_dynamics_claimed(decision)), Op.EQ, 1, '', ''):
        return
    decision.origin, decision.origin_detail = LibrationOrigin.JOINT, str(OriginDetail.JOINT_CAPTURE)
    decision.owner = decision.joint_owner
    decision.rules_fired = rules_module.fire(decision.rules_fired, rules_module.Rule.BOTH_RESONANCES_CONFIRMED)


def secular_period_offset(period_cycle_yr: float, kinematics: Optional[Kinematics]) -> float:
    """|P_cyc / P_sec - 1| for the closer of the two secular periods of the body: the e-cycle 2 pi / |g - g5|
    (`Kinematics.modulation_period_yr`) and the double omega cycle (`two_omega_period_yr`); nan without either."""
    if kinematics is None or not (is_finite(period_cycle_yr) and period_cycle_yr > 0):
        return float('nan')
    references = (kinematics.modulation_period_yr, kinematics.two_omega_period_yr)
    offsets = [abs(period_cycle_yr / ref - 1) for ref in references if is_finite(ref) and ref > 0]
    return min(offsets) if offsets else float('nan')


def _measured_below(value: float, threshold: float) -> bool:
    return is_finite(value) and value < threshold


def episode_cycle_is_kinematic(
    kinematics: Optional[Kinematics], response: Optional[Response], period_cycle_yr: float, settings: AnalysisSettings
) -> bool:
    """The cycle of an episode is kinematic (W7 '37087'): its period matches a secular period of the body within
    mmr_kin_period_tol (`secular_period_offset`: the e-cycle 2 pi / |g - g5| or the double omega cycle), the identity
    sigma_dot = c (a - a*) is measured and fails, and the force law is measured below mmr_force_r2_fail in both halves
    and under the multiplet law: a does not carry the oscillation that the forced varpi makes (37087 3J+8S-3: cycle
    5314 yr against 2 pi / (g - g5) = 5709 yr, identity slope -0.62 c, R^2 3e-4). A missing number leaves it open."""
    return _episode_cycle_route(kinematics, response, period_cycle_yr, settings, Ledger.from_settings(settings))


def force_laws(response: Response) -> Tuple[Tuple[str, float, str], ...]:
    """(label, R^2, column) of the force laws a kinematic route reads: the harmonic law in both halves, the multiplet
    law (nan when not fitted) and the N-harmonic law when L4 took it (CS12; a refused N-harmonic fit is no law)."""
    multi = response.multiplet.r2 if response.multiplet is not None else float('nan')
    laws = (
        ('half 1', response.force_r2_h1, 'rsp_force_R2_h1'),
        ('half 2', response.force_r2_h2, 'rsp_force_R2_h2'),
        ('multiplet', multi, 'rsp_force_R2_multi'),
    )
    harmonic = taken_harmonic_law(response)
    if harmonic is not None:
        laws += ((f'{harmonic.n_harmonics} harmonics', harmonic.r2, 'rsp_force_harm_R2'),)
    direct = taken_direct_law(response)
    if direct is not None:
        laws += (('direct averaged force', direct.r2, 'rsp_dforce_R2'),)
    return laws


def taken_direct_law(response: Optional[Response], angle: Optional[AngleDecision] = None) -> Optional[DirectForceLaw]:
    """The direct averaged-force law (W28 B8) that decided the response's L4 (`force_law` 'direct_force'), else None.
    With `angle` the response is first read as the tree reads it (rule R, `response.deciding_response`)."""
    if response is not None and angle is not None:
        response = response_module.deciding_response(response, other_regime(angle))
    law = response.force_direct if response is not None else None
    return law if law is not None and law.taken and response.force_law == LAW_DIRECT else None


def taken_harmonic_law(response: Optional[Response]) -> Optional[HarmonicLaw]:
    """The N-harmonic law the response's L4 read (`force_law` 'harmonic'), else None."""
    law = response.force_harmonic if response is not None else None
    return law if law is not None and law.taken and response.force_law == LAW_HARMONIC else None


def force_law_fails(response: Response, settings: AnalysisSettings) -> bool:
    """a does not carry sigma: every law of `force_laws` measured below mmr_force_r2_fail ("fails under both laws");
    a missing number leaves it open."""
    return all(_measured_below(value, settings.mmr_force_r2_fail) for _, value, _ in force_laws(response))


def decisive_period(angle: AngleDecision, geometry: Optional[Geometry]) -> float:
    """The period of the cycle the episode tests read [yr]: the decisive episode's own (`status_basis`, so under G1e
    the episode before the jump), else P_cyc. P_cyc is the median over every accepted run and may mix in another
    wave (37087 3J+8S-3 before W20: 168-yr forced-wave turning points pulled it to 836 yr)."""
    if geometry is None:
        return float('nan')
    run = run_of(geometry, status_basis(angle).episode_id)
    period = run.period_med_yr if run is not None else float('nan')
    return float(period) if is_finite(period) else float(geometry.period_cycle_yr)


def ripple_share(angle: AngleDecision, geometry: Optional[Geometry], settings: AnalysisSettings) -> float:
    """The largest swing of a librating episode (2 x its largest half-amplitude) over the range of the slow centre
    (K2); the episodes before the last jump of a under G1e (`other_regime`). nan without either."""
    if geometry is None:
        return float('nan')
    episodes = librating_episodes(geometry, settings, before_jump=other_regime(angle))
    swing = max((2 * e.half_amp_max_rad for e in episodes if is_finite(e.half_amp_max_rad)), default=float('nan'))
    span = geometry.centre.range_rad
    return float(swing / span) if is_finite(swing) and is_finite(span) and span > 0 else float('nan')


def slow_centre_is_kinematic(
    angle: AngleDecision,
    geometry: Optional[Geometry],
    response: Optional[Response],
    kinematics: Optional[Kinematics],
    settings: AnalysisSettings,
) -> bool:
    """K2, the kinematic drift of the slow centre (W20, author 2026-10-19; 21028 7E-10M-6+0+0+9): the slow centre
    follows the uniform lambda part plus the secular part of the angle (`Kinematics.centre_corr_rate` >=
    mmr_kin_corr_min, `kinematics.slow_centre_kinematics`), a does not answer sigma (`force_law_fails`: below
    mmr_force_r2_fail in both halves and under the multiplet law; 2897 35S-4 with a multiplet R^2 of 0.50 is not
    kinematic) and the episodes are a ripple on that drift: their largest swing stays below mmr_kin_ripple_share_max x
    the centre's range (`ripple_share`). A Hilda's centre follows the forced varpi too (153 3J-2: corr 0.98), but its
    force law holds and its swing is the centre's whole range (0.95). A missing number leaves it open. The decision and
    the ledger are one walk (`_slow_centre_route`)."""
    return _slow_centre_route(angle, geometry, response, kinematics, settings, Ledger.from_settings(settings))


def kinematic_check(
    angle: AngleDecision,
    kinematics: Optional[Kinematics],
    response: Optional[Response],
    period_cycle_yr: float,
    settings: AnalysisSettings,
    geometry: Optional[Geometry] = None,
) -> CheckValue:
    """check_mmr_kinematic as the tree reads it. For an episode the slow parts are smoothed over 3 cycles and hold
    the centre's wobble, not the libration (a Hilda's centre follows the forced varpi at g - g5 while it librates
    dynamically: corr 0.98), so the slow-angle test does not apply; the episode's cycle itself is judged instead
    (`period_cycle_yr`: the decisive episode's, `decisive_period`): fail when it is kinematic
    (`episode_cycle_is_kinematic`: secular period, identity fails, no force law; the Hilda's identity holds) or, with
    `geometry`, when the slow centre drifts kinematically under a ripple of episodes (`slow_centre_is_kinematic`, K2),
    else not_applicable. B1 judges the cycle as well."""
    if angle.angle_status in LIBRATING:
        kinematic = episode_cycle_is_kinematic(kinematics, response, period_cycle_yr, settings) or (
            geometry is not None and slow_centre_is_kinematic(angle, geometry, response, kinematics, settings)
        )
        return CheckValue.FAIL if kinematic else CheckValue.NOT_APPLICABLE
    return kinematics.check if kinematics is not None else CheckValue.NOT_MEASURED


def _episode_cycle_route(
    kinematics: Optional[Kinematics], response: Optional[Response], period_cycle_yr: float, settings: AnalysisSettings, ledger: Ledger
) -> bool:
    """`episode_cycle_is_kinematic` with its comparisons on `ledger`, as far as they go: the period, the identity (slope
    and correlation once it is measured), every force law once the identity fails."""
    offset = secular_period_offset(period_cycle_yr, kinematics)
    if response is None or not is_finite(offset):
        return False
    name = '3 kinematic (episode): cycle at a secular period |P_cyc / P_sec - 1|'
    if not ledger.compare_setting(name, offset, Op.LE, settings, 'mmr_kin_period_tol', '', 'kin_cycle_secular_offset'):
        return False
    if response.identity not in MEASURED:
        return False
    slope = abs(response.identity_slope_ratio - 1) if is_finite(response.identity_slope_ratio) else None
    name = '3 kinematic (episode): identity |slope / c - 1| (fails above, or by its correlation)'
    ledger.compare_setting(name, slope, Op.LE, settings, 'mmr_identity_slope_tol', '', 'rsp_identity_slope_offset')
    ledger.compare_setting(
        '3 kinematic (episode): identity correlation',
        response.identity_corr,
        Op.GE,
        settings,
        'mmr_identity_corr_min',
        '',
        'rsp_identity_corr',
    )
    if response.identity != CheckValue.FAIL:
        return False
    below = []
    for label, value, column in force_laws(response):
        name = f'3 kinematic (episode): a does not carry it, force law R2 ({label})'
        below.append(ledger.compare_setting(name, value, Op.LT, settings, 'mmr_force_r2_fail', '', column))
    return all(below)


def _slow_centre_route(
    angle: AngleDecision,
    geometry: Optional[Geometry],
    response: Optional[Response],
    kinematics: Optional[Kinematics],
    settings: AnalysisSettings,
    ledger: Ledger,
) -> bool:
    """K2's comparisons into `ledger`, in order, stopping at the first that does not hold: the force law (a does not
    answer: the premise, and the comparison that stops a dynamic libration first), the ripple share, the centre against
    the secular part. Returns the route's verdict."""
    corr = kinematics.centre_corr_rate if kinematics is not None else None
    if response is None or not is_finite(corr):
        return False
    for label, value, column in force_laws(response):
        name = f'3 kinematic (slow centre): a does not answer sigma, force law R2 ({label})'
        if not ledger.compare_setting(name, value, Op.LT, settings, 'mmr_force_r2_fail', '', column):
            return False
    name = K2_SHARE
    share = ripple_share(angle, geometry, settings)
    if not ledger.compare_setting(name, share, Op.LT, settings, 'mmr_kin_ripple_share_max', '', 'kin_centre_ripple_share'):
        return False
    name = K2_CORR
    return bool(ledger.compare_setting(name, corr, Op.GE, settings, 'mmr_kin_corr_min', '', 'kin_centre_corr_rate'))


def _k2_decision(geometry: Optional[Geometry], caps: List[str], settings: AnalysisSettings, ledger: Ledger) -> StatusDecision:
    """K2's leaf: non_resonant/kinematic_drift, detail varpi_term, and the angle word of the slow centre itself unless
    mmr_kinematic_drift_word is off (author 2026-10-19 q7: the word follows the mechanism, not the ripple): drift, or
    circulation when the centre's largest monotone pass reaches a revolution or its net shift reaches
    libration_circulation_net_rev, as G4/G4b before G5 read a centre (2897 35S-4: 10.8 rev, the author's circulation;
    21028: 0.78 rev in one pass, 1.81 net, his drift)."""
    name = '3 K2 angle word of the slow centre, not of the ripple (MMR_KINEMATIC_DRIFT_WORD on = 1)'
    word = None
    if ledger.compare(name, _flag(settings.mmr_kinematic_drift_word), Op.EQ, 1, '', 'kin_drift_word'):
        name = '3 K2 angle word: largest centre pass (G4: circulation at a revolution, else drift)'
        circulates = centre_circulates(geometry, settings, ledger, name, K2_NET)
        word = AngleStatus.CIRCULATION if circulates else AngleStatus.DRIFT
    detail = str(OriginDetail.VARPI_TERM)
    return StatusDecision(
        Status.NON_RESONANT, Reason.KINEMATIC_DRIFT, LibrationOrigin.KINEMATIC, detail, '', caps=caps, rule=RULE_K2, angle_word=word
    )


def _kinematic(
    angle: AngleDecision,
    geometry: Optional[Geometry],
    response: Optional[Response],
    kinematics: Optional[Kinematics],
    settings: AnalysisSettings,
    ledger: Ledger,
) -> Optional[StatusDecision]:
    """Step 3: the oscillation sits in the varpi part (B1), the slow angle follows p varpi with a flat a, the cycle
    of an episode is kinematic, or the slow centre drifts kinematically under a ripple of episodes (`kinematic_check`;
    the last gives kinematic_drift, detail varpi_term, and with mmr_kinematic_drift_word the slow centre's own angle
    word, drift or circulation, author 2026-10-19 q7, `_k2_decision`). A kinematic libration whose centre moves more
    than its amplitude is decided the same way (`_kinematic_centre_moves`, K2b)."""
    period = decisive_period(angle, geometry)
    b1 = response.a_amplitude if response is not None else CheckValue.NOT_MEASURED
    kin = kinematic_check(angle, kinematics, response, period, settings, geometry)
    centre_drift = False
    if response is not None:
        _check_entry(ledger, B1_AMPLITUDE, b1, response.b1_ratio, Op.GE, settings, 'mmr_b1_ratio_min', 'rsp_b1_ratio')
    if angle.angle_status in LIBRATING:
        cycle = _episode_cycle_route(kinematics, response, period, settings, ledger)
        centre_drift = not cycle and _slow_centre_route(angle, geometry, response, kinematics, settings, ledger)
    elif kinematics is not None and kin != CheckValue.NOT_APPLICABLE:
        name = KIN_VARPI
        _check_entry(ledger, name, kin, kinematics.corr_rate, Op.GE, settings, 'mmr_kin_corr_min', 'kin_corr_rate')
        if kin in MEASURED and response is not None:
            name = '3 kinematic: slow a_dot answers sin sigma'
            ledger.compare_setting(name, response.slow_adot_r2, Op.LE, settings, 'mmr_kin_slow_r2_max', '', 'rsp_slow_adot_R2')
    if CheckValue.FAIL not in (b1, kin):
        return None
    detail = kinematics.detail if kinematics is not None and kinematics.detail else str(OriginDetail.VARPI_TERM)
    slow_libration_open = response is not None and response.slow_libration == CheckValue.FAIL
    if slow_libration_open:
        name = '3 B5 exact commensurability |nu_c| / dnu_c'
        exact = ledger.compare_setting(
            name, centre_rate_significance(response), Op.LT, settings, 'mmr_b5_exact_sigma', '', 'rsp_centre_rate_over_err'
        )
        if exact:
            return StatusDecision(Status.UNCERTAIN, Reason.BASELINE_NEEDED, LibrationOrigin.KINEMATIC, detail, '', rule='3b5')
    caps = [str(Cap.SLOW_LIBRATION_NOT_EXCLUDED)] if slow_libration_open else []
    if centre_drift:
        return _k2_decision(geometry, caps, settings, ledger)
    if _kinematic_centre_moves(angle, geometry, settings, ledger):
        return replace(_k2_decision(geometry, caps, settings, ledger), origin_detail=detail)
    reason = Reason.KINEMATIC_DRIFT if angle.angle_status == AngleStatus.DRIFT else Reason.KINEMATIC_LIBRATION
    return StatusDecision(Status.NON_RESONANT, reason, LibrationOrigin.KINEMATIC, detail, '', caps=caps, rule='3')


def _kinematic_centre_moves(angle: AngleDecision, geometry: Optional[Geometry], settings: AnalysisSettings, ledger: Ledger) -> bool:
    """K2b (W22 R-KDRIFT): a kinematic libration (word libration) whose decisive episode moves its centre by more than
    mmr_kin_drift_shift_ratio x its mean amplitude (L6) is no libration about a fixed centre: it is decided as K2, a
    kinematic drift of the centre with the centre's own word (62243 1M-4J-1: L6 1.82, the centre rises 6 rad in 100
    kyr, the author's drift). The origin detail of step 3 stays. A dynamic libration with L6 > 1 is not touched."""
    if angle.angle_status != AngleStatus.LIBRATION or geometry is None:
        return False
    episode = run_of(geometry, angle.episode_id)
    shift = episode.centre_shift_over_amp if episode is not None else None
    return bool(ledger.compare_setting(K2B_SHIFT, shift, Op.GT, settings, 'mmr_kin_drift_shift_ratio', '', 'mrg_L6_value'))


def _force_law_stability(response: Response, settings: AnalysisSettings, ledger: Ledger) -> None:
    """The second half of L4: stable coefficients (multiplet law), or for a 1:1 angle the scale of its own law
    K/K_theory within [1/max, max] overall and in both halves (`coorbital.force_check`: the shape is fixed by the
    averaged potential, the planet's mass sets the scale). The single law has none: the spread of its phase phi0 is a
    column only (`response.a_response_check`). Read above a binding line of a_dot (`response.line_margin_cutoff`), the
    law must also beat the same law with a shifted in time (the null test). The N-harmonic law (CS12) records its
    gain over one harmonic, its R^2, the coherence of its coefficients, its own null test (full windows) and the
    independent halves (`response.harmonic_law`)."""
    if is_finite(response.force_line_binding_yr):
        name = NULL_TEST
        ledger.compare_setting(name, response.force_null_gain, Op.GE, settings, 'mmr_force_null_gain', '', 'rsp_force_null_gain')
    harmonic = taken_harmonic_law(response)
    if harmonic is not None:
        _harmonic_law_entries(harmonic, settings, ledger)
        return
    law = response.coorbital
    if response.force_law == 'coorbital' and law is not None:
        for label, value, column in (
            ('', law.scale, 'rsp_coorb_K_over_K_theory'),
            (' (half 1)', law.scale_h1, 'rsp_coorb_K_over_K_theory_h1'),
            (' (half 2)', law.scale_h2, 'rsp_coorb_K_over_K_theory_h2'),
        ):
            name = f'5 L4 co-orbital K/K_theory{label}'
            _within_ratio(ledger, f'{name} <= max', f'{name} >= 1/max', value, settings, 'mmr_coorbital_scale_ratio_max', column)
        return
    if response.force_law == 'multiplet' and response.multiplet is not None:
        value = response.multiplet.coefficient_coherence
        ledger.compare_setting(
            '5 L4 multiplet coefficients coherent',
            value,
            Op.GE,
            settings,
            'mmr_force_coef_coh_min',
            '',
            'rsp_force_multi_coherence',
        )


def _harmonic_law_entries(law: HarmonicLaw, settings: AnalysisSettings, ledger: Ledger) -> None:
    """The comparisons that took the N-harmonic law (`response.harmonic_law`), in the order it reads them."""
    name = HARMONIC_LAW
    ledger.compare_setting(
        f'{name}: R2 gain over one harmonic', law.gain, Op.GE, settings, 'mmr_force_multi_gain', '', 'rsp_force_harm_gain'
    )
    ledger.compare_setting(f'{name} R2', law.r2, Op.GE, settings, 'mmr_force_r2_full', '', 'rsp_force_harm_R2')
    ledger.compare_setting(
        f'{name} coefficients coherent', law.coherence, Op.GE, settings, 'mmr_force_coef_coh_min', '', 'rsp_force_harm_coherence'
    )
    ledger.compare_setting(HARMONIC_NULL, law.null_gain, Op.GE, settings, 'mmr_force_null_gain', '', 'rsp_force_harm_null_gain')
    ledger.compare_setting(HARMONIC_HALVES, law.oos_r2, Op.GE, settings, 'mmr_force_r2_fail', '', 'rsp_force_harm_oos')


def _no_dynamics(angle: AngleDecision, rule: str, flags: Optional[List[str]] = None) -> StatusDecision:
    """5a/5e: the force law and the potential with a do not show the angle's own dynamics."""
    if angle.angle_status == AngleStatus.NEAR_SEPARATRIX:
        reason = Reason.MODULATED_CIRCULATION if angle.subtype == AngleSubtype.PATTERN_B else Reason.NO_RESONANT_POTENTIAL
        return StatusDecision(Status.NON_RESONANT, reason, LibrationOrigin.NONE, '', '', flags=list(flags or []), rule=rule)
    return _unknown(rule, flags)


def _contaminated(response: Response, settings: AnalysisSettings, ledger: Ledger) -> StatusDecision:
    """5f, before the four combinations of step 5: the force law is not_measured because a forced line of the angle's
    lattice lies at its period (`response.force_line`): the fit cannot tell the libration from the forced wave, and R0
    with a reads the same a_dot-sigma relation (V_F from c a_dot; a harmonic V_F agrees with V_E for any wave the
    identity carries), so neither its pass nor its V_E-V_F fail is read; a failed sigma-only R0 (period law, fixed
    well) is independent of a_dot and keeps leaf 5e (the caller). Nothing then shows the angle's own dynamics, and the
    not_measured never makes the libration dynamic: uncertain/origin_unknown, flag force_law_contaminated (an owner
    that drives the line is step 2's; W15). The measured check stays in rsp_force_raw_check."""
    name = f'5f L4 forced line {response.force_line} within the window resolution: |1/T - 1/P| W P'
    ledger.compare(name, response.force_line_offset, Op.LT, RESOLUTION, '', 'rsp_force_line_offset', setting='mmr_force_window_cycles')
    return _unknown('5f', [str(Flag.FORCE_LAW_CONTAMINATED)])


def _identity_objects(angle: AngleDecision, response: Response, settings: AnalysisSettings, ledger: Ledger) -> StatusDecision:
    """5g: neither the force law nor the potential with a is measured, and the identity sigma_dot = c (a - a*), which
    the angle's own pendulum needs, fails, on an episode as well and on none passes (`identity_fails_locally`): the
    one measurement there is contradicts the angle's own dynamics, so the geometry alone cannot make it dynamic (W16:
    at the physical cutoff 2897 35S-4 holds an 11-TP episode of half-amplitude 1.9 deg whose identity slope is
    -2.86 c, -2.90 c pooled; its force law has one window). As 5a."""
    slope = abs(response.identity_slope_ratio - 1) if is_finite(response.identity_slope_ratio) else None
    name = '5g L3 identity |slope / c - 1| (fails above, or by its correlation)'
    ledger.compare_setting(name, slope, Op.LE, settings, 'mmr_identity_slope_tol', '', 'rsp_identity_slope_offset')
    ledger.compare_setting(
        '5g L3 identity correlation', response.identity_corr, Op.GE, settings, 'mmr_identity_corr_min', '', 'rsp_identity_corr'
    )
    return _no_dynamics(angle, '5g')


def _objection(
    angle: AngleDecision, l4: CheckValue, r0: CheckValue, response: Optional[Response], settings: AnalysisSettings, ledger: Ledger
) -> Optional[StatusDecision]:
    """The leaves of step 5 where a measurement speaks against the angle's own dynamics (5a, 5b, 5e, 5g); None when
    none does (5c, 5d -> step 6)."""
    if l4 == CheckValue.FAIL and r0 != CheckValue.PASS:
        return _no_dynamics(angle, '5a' if r0 == CheckValue.FAIL else '5a*')  # 5a*: nothing confirms a potential either
    if l4 == CheckValue.PASS and r0 == CheckValue.FAIL:
        # 5b for every angle class (SPEC §5.3): a_dot follows the force law while the energy disagrees with the
        # potential from a; under the identity both hold together, so the two measurements contradict each other
        # (a varpi share or an error), which neither confirms nor excludes the resonance
        return _unknown('5b', [str(Flag.FORCE_LAW_DISAGREES)])
    if r0 == CheckValue.FAIL:
        return _no_dynamics(angle, '5e')  # the potential with a fails and no force law speaks for the angle
    if l4 not in MEASURED and r0 != CheckValue.PASS and identity_fails_locally(response):
        return _identity_objects(angle, response, settings, ledger)
    return None


def identity_fails_locally(response: Optional[Response]) -> bool:
    """The identity the status reads fails, and so does it on an episode of its own while it passes on none: a pooled
    regression over pieces with different equilibria a* can fail although every piece passes (Codex W16: two pieces
    with slope/c 1.000, their union 0.016), so the pool alone is no objection to the angle's own dynamics."""
    if response is None or response.identity != CheckValue.FAIL:
        return False
    local = [episode.identity for episode in response.episode_responses]
    return CheckValue.FAIL in local and CheckValue.PASS not in local


def _force_and_potential(
    angle: AngleDecision,
    geometry: Geometry,
    response: Optional[Response],
    energy: Optional[EnergyAnalysis],
    settings: AnalysisSettings,
    ledger: Ledger,
    strength: Optional[Strength] = None,
    episodes: Sequence[EpisodePhysics] = (),
) -> StatusDecision:
    """Step 5 (all four combinations of L4 and R0, and 5g: both unmeasured with a failed identity) and step 6
    (dynamics). A librating angle whose L4 and R0 both do not pass first meets rule 5s (`_implausible_strength_first`):
    a strength computed from the disturbing function that the period exceeds by mmr_strength_implausible_ratio says
    more than a failed or missing force law (W20, author 2026-10-19). A librating angle with L4 and R0 both not
    measured then takes `unmeasured_branch`. `energy` is the tree's (`energy_for_tree`): a passed R0 the filter guard
    withdraws is not_measured here (W21 Q9, 117306 11M-20: then rule 5d (i), the direct strength F, decides)."""
    l4 = response.a_response if response is not None else CheckValue.NOT_MEASURED
    r0 = energy.with_slow if energy is not None else CheckValue.NOT_MEASURED
    if response is not None:
        single_decides = taken_harmonic_law(response) is None  # else the one-harmonic halves are diagnostics only
        for half, value in ((1, response.force_r2_h1), (2, response.force_r2_h2)):
            name, column = f'5 L4 force law R2 (half {half})', f'rsp_force_R2_h{half}'
            _check_entry(ledger, name, l4, value, Op.GE, settings, 'mmr_force_r2_full', column, named=single_decides)
    if response is not None and l4 in MEASURED:
        _force_law_stability(response, settings, ledger)
    law = energy.potential_law if energy is not None else None  # the 1:1 law R0 was read from (`response.with_coorbital_potential`)
    if law is not None:
        name = '5 R0 co-orbital energy integral R2'
        _check_entry(ledger, name, r0, law.energy_r2, Op.GE, settings, 'mmr_force_r2_fail', 'rsp_coorb_energy_R2')
    elif energy is not None:
        name = '5 R0 V_E ~ V_F windows'
        _check_entry(ledger, name, r0, energy.ve_vf_agree_fraction, Op.GE, settings, 'separatrix_vevf_fraction', 'sep_VE_VF_agree_fraction')
    objection = _implausible_strength_first(angle, l4, r0, strength, settings, ledger)
    if objection is None:
        objection = _objection(angle, l4, r0, response, settings, ledger)
    if objection is not None:
        return objection
    if l4 not in MEASURED and r0 not in MEASURED and angle.angle_status in LIBRATING:
        return unmeasured_branch(angle, geometry, energy, strength, episodes, settings, ledger)
    flags, caps, rule = [], [], '5d'
    if l4 == CheckValue.FAIL:  # and R0 pass
        rule = '5c'
        flags.append(str(Flag.FORCE_LAW_DISAGREES))
        caps.append(str(Cap.FORCE_LAW_DISAGREES))
    status, reason, _, zone_flags = zone_rule(angle, geometry, energy, settings, ledger)
    return StatusDecision(status, reason, LibrationOrigin.DYNAMIC, '', '', flags=flags + zone_flags, caps=caps, rule=f'{rule}/6')


# --------------------------------------------------------------------------------------------- the unmeasured branch


def unmeasured_branch(
    angle: AngleDecision,
    geometry: Geometry,
    energy: Optional[EnergyAnalysis],
    strength: Optional[Strength],
    episodes: Sequence[EpisodePhysics],
    settings: AnalysisSettings,
    ledger: Ledger,
) -> StatusDecision:
    """Rule 5d/6 of a librating angle whose force law (L4) and potential with a (R0) are both not measured at decision
    time (author 2026-10-17, W17 Q3 P4/P5). The first leaf that holds:

      (i)   5d/6  the strength measured by direct averaging (or the harmonic) passes: as every 5d/6, origin dynamic;
      (ii-a) 5s   the period needs >= mmr_strength_implausible_ratio x the directly averaged strength (two-body), or
                  x mu_i mu_j e^q (three-body, legal e^q) and the small-divisor envelope still leaves that factor
                  (`strength.small_divisor_envelope`): non_resonant/implausible_strength, origin unknown, cap <= low;
            5s*   the same three-body ratio, but the envelope can close the gap (or has no bound): uncertain;
      (ii-b) 5o   a measured objection: O1 fails (a strong a-wave before the episode) or F by direct averaging (or
                  the harmonic) fails (centre_outside_well included): uncertain/origin_unknown;
            5x    the order-of-magnitude scale (eq_scale: required / mu_i mu_j e^q, three-body and 1:1) lies outside
                  [1/x, x]: no measured objection (an estimate known to a factor 1-100 is a diagnostic, W28 B9, card
                  g08-6 A; 79653 3M+3J-7, 39.5 x), but it does not let the geometry decide either:
                  uncertain/origin_unknown, 'origin not established';
            5n    the decisive episode holds fewer than libration_centre_min_estimates independent cycles (one centre,
                  nothing to compare): uncertain/origin_unknown;
      (iii) 5u/6  nothing measured: the geometry's status (`zone_rule`) with origin unknown, reason geometric_libration
                  or geometric_episode (`GEOMETRIC_REASONS`), and the cap geometry_only (<= low), which the owner
                  search cannot lift; strength_scale_only (<= medium) instead when the order-of-magnitude strength
                  scale passes [1/x, x] (`_geometry_cap`, W22 R-F).

    Since W20 the two-body 5s (except centre_outside_well) and the three-body 5s already ran before this branch
    (`_implausible_strength_first`); here (ii-a) still decides centre_outside_well and the three-body 5s*.

    The 1000 x factor is the existing implausibility threshold: among the three-body rows with a legal estimate no
    author resonant/transient needs more than 46 x, 20 of 21 non_resonant need more than 1000 x (four rows support the
    lower side). Expected on the benchmark: 2012 KL9 and 2012 TK84 non_resonant; 8306, 46596, 501710 uncertain;
    105247 non_resonant; 2003 ND, 2013 VA14 and 1998 QA62 uncertain."""
    name = '5 L4 and R0 both not measured at decision time: the branch rule'
    ledger.compare(name, None, Op.EQ, 0, '', '')
    if strength is not None and strength.method in DIRECT_METHODS and strength.check == CheckValue.PASS:
        _strength_within(strength, DIRECT_BRANCH, settings, ledger)
        status, reason, _, flags = zone_rule(angle, geometry, energy, settings, ledger)
        return StatusDecision(status, reason, LibrationOrigin.DYNAMIC, '', '', flags=flags, rule='5d/6')
    implausible = _implausible_strength(strength, settings, ledger)
    if implausible is not None:
        return implausible
    physics = physics_of(episodes, angle.episode_id)
    if _measured_objection(angle, energy, strength, physics, settings, ledger):
        return _unknown('5o')
    if _scale_out_of_range(strength, settings, ledger):
        decision = _unknown('5x')
        decision.rules_fired = rules_module.fire(decision.rules_fired, rules_module.Rule.EQ_SCALE_NOT_OBJECTION)
        return decision
    episode = run_of(geometry, angle.episode_id)
    if episode is not None:
        name = INDEPENDENT_CYCLES
        setting = 'libration_centre_min_estimates'
        if not ledger.compare_setting(name, episode.centre_estimates, Op.GE, settings, setting, '', 'ep_centre_estimates'):
            return _unknown('5n')
    status, reason, _, flags = zone_rule(angle, geometry, energy, settings, ledger)
    reason = GEOMETRIC_REASONS.get(reason, reason)  # no force measured: the geometry's word (author 2026-10-26, Q3)
    return StatusDecision(
        status, reason, LibrationOrigin.UNKNOWN, '', '', flags=flags, caps=[_geometry_cap(strength, settings, ledger)], rule='5u/6'
    )


# leaf 5u/6: the reason of the geometry alone (origin unknown) for the reason a measured force gives (origin dynamic)
GEOMETRIC_REASONS = {Reason.DYNAMIC_LIBRATION: Reason.GEOMETRIC_LIBRATION, Reason.DYNAMIC_EPISODE: Reason.GEOMETRIC_EPISODE}


def _geometry_cap(strength: Optional[Strength], settings: AnalysisSettings, ledger: Ledger) -> str:
    """The cap of leaf 5u/6: strength_scale_only (<= medium) when the strength on the order-of-magnitude scale
    (required / mu_i mu_j e^q, not direct averaging: that one decided leaf (i)) lies within [1/x, x], else
    geometry_only (<= low). The scale agreeing is evidence for the angle's own force, but no computed force (W22 R-F)."""
    if strength is not None and strength.check == CheckValue.PASS and strength.method not in DIRECT_METHODS:
        if _strength_within(strength, '5u/6 F on the order-of-magnitude scale', settings, ledger):
            return str(Cap.STRENGTH_SCALE_ONLY)
    return str(Cap.GEOMETRY_ONLY)


def _unknown(rule: str, flags: Optional[List[str]] = None) -> StatusDecision:
    return StatusDecision(Status.UNCERTAIN, Reason.ORIGIN_UNKNOWN, LibrationOrigin.UNKNOWN, '', '', flags=list(flags or []), rule=rule)


def strength_upper_name(label: str) -> str:
    """The ledger name of the upper end of F (`_strength_within`) under the rule label `label`."""
    return f'{label}: required / theory <= x'


def _strength_within(strength: Strength, label: str, settings: AnalysisSettings, ledger: Ledger) -> bool:
    """F on the ledger: required / theory within [1/x, x], x = mmr_strength_check_ratio (both ends recorded)."""
    lower_name = f'{label}: required / theory >= 1/x'
    return _within_ratio(ledger, strength_upper_name(label), lower_name, strength.ratio, settings, 'mmr_strength_check_ratio', 'str_ratio')


def _within_ratio(ledger: Ledger, upper_name: str, lower_name: str, value, settings: AnalysisSettings, setting: str, column: str) -> bool:
    """`value` within [1/x, x], x the setting `setting`: both ends on the ledger (upper first), True when both hold."""
    upper = ledger.compare_setting(upper_name, value, Op.LE, settings, setting, '', column)
    lower = ledger.compare(lower_name, value, Op.GE, 1 / getattr(settings, setting), '', column, setting=setting)
    return bool(upper and lower)


STRENGTH_FIRST = '5s (before 5a/5e/5g/5u)'  # the ledger prefix of rule 5s where it precedes the leaves of step 5


def _implausible_decision() -> StatusDecision:
    flags = [str(Flag.IMPLAUSIBLE_STRENGTH)]
    caps = [str(Cap.IMPLAUSIBLE_STRENGTH)]
    return StatusDecision(
        Status.NON_RESONANT, Reason.IMPLAUSIBLE_STRENGTH, LibrationOrigin.UNKNOWN, '', '', flags=flags, caps=caps, rule='5s'
    )


def _implausible_strength(strength: Optional[Strength], settings: AnalysisSettings, ledger: Ledger) -> Optional[StatusDecision]:
    """Leaf (ii-a) of `unmeasured_branch`: 5s, 5s* or None. One-sided: only a force far above the theory is
    implausible here (a weak required force is no objection to a libration)."""
    if strength is None or not (strength.method in DIRECT_METHODS or (strength.three_body and strength.eq_legal)):
        return None
    rule = _strength_route(strength, '5s', settings, ledger)
    if rule == '5s*':
        return _unknown('5s*', [str(Flag.IMPLAUSIBLE_STRENGTH)])
    return _implausible_decision() if rule else None


def _strength_route(strength: Strength, prefix: str, settings: AnalysisSettings, ledger: Ledger) -> Optional[str]:
    """The comparisons of rule 5s, named `prefix` ...: the two-body strength (direct averaging or the harmonic) at or
    above mmr_strength_implausible_ratio gives '5s'; the three-body e^q ratio there gives '5s' when the small-divisor
    envelope still leaves the factor, else '5s*'; None below the ratio. The callers check that the route applies."""
    setting = 'mmr_strength_implausible_ratio'
    if strength.method in DIRECT_METHODS:
        name = f'{prefix} {STRENGTH_DIRECT}'
        return '5s' if ledger.compare_setting(name, strength.ratio, Op.GE, settings, setting, '', 'str_ratio') else None
    if not ledger.compare_setting(f'{prefix} {STRENGTH_THREE_BODY}', strength.ratio, Op.GE, settings, setting, '', 'str_ratio'):
        return None
    name = f'{prefix} {STRENGTH_DIVISORS}'
    return '5s' if ledger.compare_setting(name, strength.divisor_residual, Op.GE, settings, setting, '', 'str_divisor_residual') else '5s*'


def _implausible_strength_first(
    angle: AngleDecision, l4: CheckValue, r0: CheckValue, strength: Optional[Strength], settings: AnalysisSettings, ledger: Ledger
) -> Optional[StatusDecision]:
    """Rule 5s before the leaves 5a/5a*/5e/5g and the branch (W20, author 2026-10-19): a librating angle whose force
    law (`l4`) and potential with a (`r0`) do not pass, and whose period needs >= mmr_strength_implausible_ratio x a strength
    computed from the disturbing function, is non_resonant/implausible_strength whatever else failed or is missing
    (2897 35S-4, 16126 2R-35: the harmonic in extended precision, 10^36 and 10^20 x). Two routes, as leaf (ii-a):
    the two-body strength by direct averaging or the harmonic, except a centre outside the theoretical well (its
    ratio is over evaluations whose swing does not librate in the well; it keeps the old order unless
    mmr_strength_first_outside_well); the three-body legal e^q ratio only when the small-divisor envelope still leaves
    the factor (no 5s* here: an envelope that may close the gap leaves the old leaves in charge). None otherwise,
    and nothing is recorded when the route does not apply or no ratio was measured."""
    if angle.angle_status not in LIBRATING or CheckValue.PASS in (l4, r0):
        return None
    if strength is None or not is_finite(strength.ratio):
        return None
    if strength.method in DIRECT_METHODS:
        if strength.not_measured_reason == REASON_CENTRE_OUTSIDE_WELL and not settings.mmr_strength_first_outside_well:
            return None
    elif not (strength.three_body and strength.eq_legal) or not is_finite(strength.divisor_residual):
        return None
    return _implausible_decision() if _strength_route(strength, STRENGTH_FIRST, settings, ledger) == '5s' else None


def onset_ratio(physics: Optional[EpisodePhysics]) -> float:
    """O1's number: the a-wave in the band before the episode over the one inside it (nan when not measured)."""
    if physics is None:
        return float('nan')
    before, inside = physics.a_band_std_before_au, physics.a_band_std_in_au
    return float(before / inside) if is_finite(before) and is_finite(inside) and inside > 0 else float('nan')


def _measured_objection(
    angle: AngleDecision,
    energy: Optional[EnergyAnalysis],
    strength: Optional[Strength],
    physics: Optional[EpisodePhysics],
    settings: AnalysisSettings,
    ledger: Ledger,
) -> bool:
    """Leaf (ii-b) of `unmeasured_branch`: O1 of an episodic angle fails at decision time (before the filter guard;
    not applicable when the episodes lie in the separatrix zone), or the strength check F fails by direct averaging or
    the harmonic (every observed swing outside the theoretical well included). A failed order-of-magnitude scale
    (eq_scale) is no measured objection: leaf 5x reads it (`_scale_out_of_range`, W28 B9, card g08-6 A).

    The leaf reads O1's ratio alone, not the narrowed margin (`foreign_wave`, B4): measured on 2026-10-31, the
    narrowed test would lift the objection from the author's uncertain/episodic labels 2013 VA14 60 kyr (coherent
    0.69 but 135 deg out of phase) and 2003 TG2 100 kyr (coherence 0.23), so leaf 5o keeps the 23.10 rule (Q2: a
    failed O1 is uncertain; dev/status.md)."""
    objects = False
    if angle.angle_status == AngleStatus.EPISODIC and physics is not None and not episodes_in_separatrix_zone(energy):
        name = O1_ONSET
        held = ledger.compare_setting(name, onset_ratio(physics), Op.LT, settings, 'mmr_o1_ratio_fail', '', 'ep_onset_ratio')
        objects = held.result is False
    if strength is not None and strength.check in MEASURED and strength.method in DIRECT_METHODS:
        if strength.not_measured_reason == REASON_CENTRE_OUTSIDE_WELL:
            name = F_IN_WELL
            ledger.compare(name, strength.n_consistent, Op.GE, 1, '', 'str_n_consistent')
        else:
            _strength_within(strength, '5o F', settings, ledger)
        objects = objects or strength.check == CheckValue.FAIL
    return objects


SCALE_OUT_OF_RANGE = '5x F scale out of range'  # the ledger label of leaf 5x (`_strength_within` names its two ends)


def _scale_out_of_range(strength: Optional[Strength], settings: AnalysisSettings, ledger: Ledger) -> bool:
    """Leaf 5x of `unmeasured_branch` (W28 B9, card g08-6 A): the order-of-magnitude strength (eq_scale, required /
    mu_i mu_j e^q or mu e^q; known only to a factor 1-100) is measured and outside [1/x, x], x =
    mmr_strength_check_ratio. Not an objection (that is 5o, direct averaging only): the origin is not established."""
    if strength is None or strength.method in DIRECT_METHODS or strength.check != CheckValue.FAIL:
        return False
    _strength_within(strength, SCALE_OUT_OF_RANGE, settings, ledger)
    return True


def own_dynamics_claimed(decision: StatusDecision) -> bool:
    """The verdict reads the oscillation as this angle's own: origin dynamic, or origin unknown on a resonant,
    transient or near_separatrix status (rule 5u/6: the geometry alone). Its margins and flags are those of a dynamic
    verdict (L4, R0, Z, F; implausible_strength)."""
    claims = decision.status in (Status.RESONANT, Status.TRANSIENT, Status.NEAR_SEPARATRIX)
    own = (LibrationOrigin.UNKNOWN, LibrationOrigin.JOINT)  # joint: the angle's own reading under a joint capture (A2b)
    return decision.origin == LibrationOrigin.DYNAMIC or (claims and decision.origin in own)


def decide_status(
    angle: AngleDecision,
    sampling: Sampling,
    geometry: Optional[Geometry],
    energy: Optional[EnergyAnalysis],
    response: Optional[Response],
    kinematics: Optional[Kinematics],
    subresonances: Optional[SubresonanceResult],
    owner: Optional[OwnerResult],
    strength: Optional[Strength],
    settings: AnalysisSettings,
    ledger: Ledger,
    member: Optional[Verdict] = None,
    episodes: Sequence[EpisodePhysics] = (),
    withdrawn: Optional[Mapping[str, Withdrawal]] = None,
    slow_scale: Optional[SlowScale] = None,
) -> StatusDecision:
    """SPEC §5.3: the only place a status is set. Every combination of check values reaches a leaf.

    `strength` is no step of the tree but for the branch where L4 and R0 are both not measured (`unmeasured_branch`;
    elsewhere it supports a libration whose force law is not measured, margin F); `episodes`: the episodes' physics
    (O1 of the decisive one there). The flag episodes_in_separatrix_zone of the energy goes with every leaf that has an
    oscillation (a column, no status). `member`: the verdict of the accepted subresonance's own row (the derived
    analysis, built before this call), whose status the catalogue row takes (`member_status`). `withdrawn`: the checks
    the filter guard withdrew from `energy` before the tree (`energy_for_tree`), recorded on the decision with those
    of R-ALIAS-A (`StatusDecision.withdrawn`). `slow_scale`: the slower libration of the centre (B10a), which K3s
    reads after the leaf (`secular_cycle_kinematic`)."""
    if other_regime(angle):
        name = 'G1e the response and energy of the last regime describe no episode the status reads'
        ledger.compare(name, None, Op.EQ, 0, '', '')
    _witness_entries(response, ledger)
    response, energy = regime_physics(angle, response, energy)
    decision = _decide_status(
        angle, sampling, geometry, energy, response, kinematics, subresonances, owner, strength, settings, ledger, member, episodes
    )
    promote_union(angle, decision, energy, ledger)
    secular_cycle_kinematic(decision, slow_scale, geometry, settings, ledger)
    composite_regime_detail(decision, subresonances)
    joint_capture_origin(decision, ledger)
    decision.withdrawn = {**dict(withdrawn or {}), **decision.withdrawn}
    if response_module.force_law_mixed(response):  # rule R: the witness decides, the other episode limits the level
        decision.flags.append(str(Flag.EPISODE_FORCE_LAW_MIXED))
        decision.caps.append(str(Cap.EPISODE_FORCE_LAW_MIXED))
    if episodes_in_separatrix_zone(energy) and EPISODES_IN_SEPARATRIX_ZONE not in decision.flags:
        if decision.status not in (Status.INVALID,) and status_basis(angle).angle_status != AngleStatus.INSUFFICIENT_TIME:
            decision.flags.append(EPISODES_IN_SEPARATRIX_ZONE)
    scatters = _claims_no_libration(angle, decision, geometry) and leg_scatters(status_basis(angle), geometry, settings)
    if scatters:  # G4c: the best 1:1 leg is a near miss of a capture, both its turns kicks (1991 VG)
        ledger.compare(G4C_SCATTER, 1.0, Op.EQ, 1, '', 'geo_leg_outcome')
        decision.rules_fired = rules_module.fire(decision.rules_fired, rules_module.Rule.COORBITAL_LEG)
    if near_episode(angle, decision, geometry, settings, ledger) or scatters:
        decision.flags.append(str(Flag.NEAR_EPISODE))
        decision.caps.append(str(Cap.NEAR_EPISODE))
        if structural_near_episode(geometry, settings):  # the near-episode is a 1:1 leg rejected over the top
            decision.rules_fired = rules_module.fire(decision.rules_fired, rules_module.Rule.COORBITAL_NEAR_EPISODE)
    if owner is not None and FLAG_CATALOGUE in owner.flags:  # the search ran on catalogue lines: approximate either way
        decision.caps.append(str(Cap.CATALOGUE_LONGITUDES))
    if taken_direct_law(response) is not None:  # W28 B8: the direct averaged-force law decided L4 the tree read
        decision.rules_fired = rules_module.fire(decision.rules_fired, rules_module.Rule.DIRECT_FORCE_LAW)
    return decision


# K3s (W28 B10a, card g08-8 A): the ledger names; only these reasons, which name no mechanism, may take it
K3S_OFFSET = 'K3s slow scale: |P_slow / P_secular - 1| (pi/|omega_dot|, 2 pi/|g - g_j|)'
K3S_IDENTITY = 'K3s slow scale: its own identity sigma_s_dot = c (a_s - a*) fails (fail = 1)'
K3S_SLIP = 'K3s the centre circulates only by slips: largest monotone pass <= 1 + libration_slip_tolerance'
K3S_REASONS = (Reason.CIRCULATION,)


def secular_cycle_kinematic(
    decision: StatusDecision,
    slow_scale: Optional[SlowScale],
    geometry: Optional[Geometry],
    settings: AnalysisSettings,
    ledger: Ledger,
) -> None:
    """K3s (W28 B10a, card g08-8 A, rule `secular_cycle_kinematic`, MMR_SECULAR_CYCLE_KINEMATIC): a non_resonant
    circulation (no mechanism named, origin none, not 1:1) whose slow scale (`SlowScale`) has the period of a secular
    cycle (secular offset <= mmr_kin_period_tol) and fails its own identity is a kinematic libration with origin
    detail secular_cycle: the slow swing is the secular cycle seen through the angle's varpi part, which a does not
    follow (37087 3J+8S-3: 5.7 kyr = 2 pi/(g - g5), identity slope / c 2.6, corr 0.53). Only where the slow
    libration is the record's regime: its centre leaves it by slips of one turn (the largest monotone pass of the
    centre within a slip, 1 + libration_slip_tolerance rev; 37087 1.09 rev at 100 kyr, 1.16 at 1 Myr); a centre that
    circulates for several turns in a row is a circulation with slow interludes, whatever their cause (285637 3M+7J-6
    2 Myr: 4.7 rev, the author's circulation). B1 and a thin L4 do not block
    it (B1 is pooled on the fast scale, where a forced wave carries a). The status never changes, nor a row that
    librates, an induced or a subresonance one; nothing without a slow scale. The level stays the circulation's
    (`_origin_margins`, `physics_margins_of`). In place, on the ledger when it is read."""
    if not settings.mmr_secular_cycle_kinematic or slow_scale is None or not slow_scale.measured:
        return
    if decision.status != Status.NON_RESONANT or decision.reason not in K3S_REASONS or decision.origin != LibrationOrigin.NONE:
        return
    if geometry is None or geometry.coorbital:
        return
    slips = ledger.compare(
        K3S_SLIP,
        geometry.centre.max_pass_rev,
        Op.LE,
        1 + settings.libration_slip_tolerance,
        'rev',
        'geo_centre_max_pass_rev',
        setting='libration_slip_tolerance',
    )
    if not slips:
        return
    offset = slow_scale.secular_offset if is_finite(slow_scale.secular_offset) else None
    matched = ledger.compare_setting(K3S_OFFSET, offset, Op.LE, settings, 'mmr_kin_period_tol', '', 'slow_secular_offset')
    failed = ledger.compare(K3S_IDENTITY, _flag(slow_scale.identity == CheckValue.FAIL), Op.EQ, 1, '', 'slow_identity')
    if not (matched and failed):
        return
    decision.reason, decision.origin = Reason.KINEMATIC_LIBRATION, LibrationOrigin.KINEMATIC
    decision.origin_detail, decision.rule = str(OriginDetail.SECULAR_CYCLE), RULE_K3S
    decision.rules_fired = rules_module.fire(decision.rules_fired, rules_module.Rule.SECULAR_CYCLE_KINEMATIC)


def promote_union(angle: AngleDecision, decision: StatusDecision, energy: Optional[EnergyAnalysis], ledger: Ledger) -> None:
    """The physics guard of the union rule (W22, round 1024 safe variant): the tree reads a union of episodes of an
    angle that is not 1:1 (G2u) as episodic (`status_basis`); only when that reading reached a dynamic transient (leaf
    6 with origin dynamic: the force law L4, the potential with a R0 or the directly averaged strength F measured and
    passing) the libration over the record makes it resonant/dynamic_libration (TK14 31M-34 at 0.5 yr). Every other
    outcome stays: induced, kinematic, implausible_strength, uncertain (5o, 5n, 5f, 5s*), the geometry alone (5u/6),
    a subresonance (2003 TX9 3E+2J-10 stays uncertain). In place, on the ledger only for a union. The final rule
    (`rec_rule`) names the promotion: the leaf's rule plus 'u' (5d/6 → 5d/6u); its prefix keeps the leaf's branch for
    the readers after the status (`_l4` reads 5c)."""
    if angle.rule != RULE_UNION:
        return
    dynamic = decision.status == Status.TRANSIENT and decision.reason == Reason.DYNAMIC_EPISODE
    dynamic = dynamic and decision.origin == LibrationOrigin.DYNAMIC
    if not ledger.compare(UNION_PROMOTED, _flag(dynamic), Op.EQ, 1, '', ''):
        return
    decision.status, decision.reason, decision.union_promoted = Status.RESONANT, Reason.DYNAMIC_LIBRATION, True
    decision.rule += 'u'
    if energy is not None and has_zone_cycle(energy.cycle_classes):  # as `zone_rule`
        decision.flags.append(str(Flag.LIBRATION_IN_SEPARATRIX_ZONE))


def union_read(angle: AngleDecision, decision: StatusDecision) -> AngleDecision:
    """The angle the margins, checks, caps, baseline and final regime read after the status: a union of episodes the
    status promoted (`promote_union`) as the libration it is (rule G2, as a 1:1 union), else the angle unchanged (a
    union not promoted keeps its episodic reading there)."""
    return replace(angle, rule='G2') if angle.rule == RULE_UNION and decision.union_promoted else angle


def _claims_no_libration(angle: AngleDecision, decision: StatusDecision, geometry: Optional[Geometry]) -> bool:
    """A non_resonant verdict whose angle (as the status reads it) holds no episode."""
    return geometry is not None and decision.status == Status.NON_RESONANT and status_basis(angle).angle_status not in LIBRATING


def near_episode_floor(geometry: Geometry, settings: AnalysisSettings, run) -> float:
    """The period a short run needs to count as a near-episode [yr]: its floor (`period_floor`) x (1 -
    confidence_thin_band), for every MMR (card g01-8 A): a run within the thin margin of the floor is a thin miss of an
    episode too (2023 RX1 1E-1: 0.899 of its floor). No new threshold."""
    return period_floor(geometry, settings, run)[0] * (1.0 - settings.confidence_thin_band)


def near_episode_tp(geometry: Geometry, settings: AnalysisSettings) -> int:
    """TPs of the longest run that would be a librating episode but for its count: a short run (`geometry.short_runs`)
    whose period (twice its median half-cycle: a run of two TPs has no full period) is not shorter than the angle's
    shortest libration less the thin band (`near_episode_floor`) and whose centre does not complete a revolution inside
    it: the criteria of `librating_episodes` but the count. A faster back-and-forth is a forced wave, not half a
    libration (2012 XE133 1V-1: 46 yr against a 76-yr floor). 0 without one. Once per geometry and settings
    (`Geometry.derived`)."""

    def compute() -> int:
        scratch = Ledger.from_settings(settings)
        counts = [
            run.n_tp
            for run in short_runs(geometry.runs, coorbital=geometry.coorbital)
            if scratch.compare(
                'period', 2 * float(np.median(np.diff(run.tp_times_yr))), Op.GE, near_episode_floor(geometry, settings, run), 'yr', ''
            )
            and scratch.compare('centre turn', centre_travel_in(geometry, run.t_start_yr, run.t_end_yr), Op.LT, TWO_PI, 'rad', '')
        ]
        return max(counts, default=0)

    return geometry.derived(('near_episode_tp',), settings, compute)


def structural_near_episode(geometry: Geometry, settings: AnalysisSettings) -> bool:
    """Whether the near-episode of a 1:1 angle rests on a run rejected over_the_top (`short_runs` with coorbital: a
    leg of one co-orbital well between passages, cards g01-5 A, g01-8 A): rule coorbital_near_episode."""
    if not geometry.coorbital:
        return False
    scratch = Ledger.from_settings(settings)
    count = near_episode_tp(geometry, settings)
    return any(
        run.reject_reason == RejectReason.OVER_THE_TOP
        and run.n_tp == count
        and scratch.compare(
            'period', 2 * float(np.median(np.diff(run.tp_times_yr))), Op.GE, near_episode_floor(geometry, settings, run), 'yr', ''
        )
        for run in short_runs(geometry.runs, coorbital=True)
    )


# --------------------------------------------------------------------------------------------- G4c: one 1:1 leg

RULE_G4C = 'G4c'
# the outcomes of one leg (`leg_outcome`), the record takes the strongest (`LEG_RANK`)
LEG_TRANSIENT, LEG_UNCERTAIN, LEG_EXTEND, LEG_SCATTER = 'transient', 'uncertain', 'extend', 'scatter'
LEG_RANK = {LEG_TRANSIENT: 4, LEG_UNCERTAIN: 3, LEG_EXTEND: 2, LEG_SCATTER: 1}
# the first discrepancy of a leg that is not transient (column geo_leg_discrepancy)
LEG_NOT_CLEAN, LEG_FORCE, LEG_ENERGY, LEG_TURN = 'passing', 'force_law', 'energy', 'turn'
G4C_TURN = 'G4c turn |ln out/in| (|a - a_P| after / before: 0 for a reflection)'
G4C_ZONE = 'G4c turn |a - a_P| / da_max on the leg side'
G4C_CLEAN = 'G4c leg is one swing of a co-orbital well (crosses at most one of planet, L3)'
G4C_FORCE = 'G4c local 1:1 law on the leg passes with K > 0'
G4C_ENERGY = 'G4c local energy integral on the leg passes'
G4C_SCATTER = 'G4c best 1:1 leg: both turns scatter (near-episode)'


def turn_reflects(turn: Optional[LegTurn], settings: AnalysisSettings) -> bool:
    """A turn of a 1:1 leg is a reflection by the averaged potential: |a - a_P| mirrored (|ln out/in| <=
    mmr_coorbital_turn_energy_tol) at a level the potential can reverse (zone <= mmr_coorbital_turn_zone_max). A
    turn not measured is no reflection."""
    if turn is None or not (is_finite(turn.out_in) and turn.out_in > 0 and is_finite(turn.zone)):
        return False
    return abs(math.log(turn.out_in)) <= settings.mmr_coorbital_turn_energy_tol and turn.zone <= settings.mmr_coorbital_turn_zone_max


def turn_reflects_within_band(turn: Optional[LegTurn], settings: AnalysisSettings) -> bool:
    """A turn that would reflect but for its zone, which lies within the thin band above MMR_COORBITAL_TURN_ZONE_MAX
    (zone <= (1 + CONFIDENCE_THIN_BAND) x max): the threshold is a convention, so the turn is neither a reflection nor
    a kick (the author, W30 zone C: 2020 FN3 1.11, 255071 1.15 against 1.1)."""
    if turn is None or turn_reflects(turn, settings) or not (is_finite(turn.out_in) and turn.out_in > 0 and is_finite(turn.zone)):
        return False
    mirrored = abs(math.log(turn.out_in)) <= settings.mmr_coorbital_turn_energy_tol
    return bool(mirrored and turn.zone <= settings.mmr_coorbital_turn_zone_max * (1.0 + settings.confidence_thin_band))


def leg_law_holds(leg: SingleLeg) -> bool:
    """The local 1:1 law of the leg passes and pushes a the right way (K / K_theory > 0)."""
    return leg.force == CheckValue.PASS and is_finite(leg.force_scale) and leg.force_scale > 0


def leg_outcome(leg: SingleLeg, settings: AnalysisSettings) -> Tuple[str, str]:
    """(outcome, first discrepancy) of one 1:1 leg (q-4 A, the W28 C1 rule; '' when the leg decides nothing):

    - an edge leg (one turn observed): clean, its turn reflects (with mmr_coorbital_leg_force_gate also its law
      holds on the observed part) -> LEG_EXTEND, the record may hold the rest of the cycle; else '';
    - neither turn reflects (both are kicks, or a level the potential cannot reverse), or with the force gate no 1:1
      law -> LEG_SCATTER: no capture, but a near miss of one; a turn whose zone is within the thin band above the
      threshold (`turn_reflects_within_band`) -> LEG_UNCERTAIN (W30 zone C);
    - clean, the law with K > 0 and the energy integral pass and both turns reflect -> LEG_TRANSIENT;
    - anything else -> LEG_UNCERTAIN, with the first discrepancy (passing, force_law, energy, turn)."""
    gate = settings.mmr_coorbital_leg_force_gate
    if leg.edge:
        turn = leg.exit if leg.edge == LEG_EDGE_START else leg.entry
        held = leg.clean and turn_reflects(turn, settings) and (leg_law_holds(leg) or not gate)
        return (LEG_EXTEND, '') if held else ('', '')
    reflects = [turn_reflects(turn, settings) for turn in (leg.entry, leg.exit)]
    if not any(reflects):
        # a turn whose zone is only within the thin band of the threshold: undecided, not a scatter (W30 zone C)
        if any(turn_reflects_within_band(turn, settings) for turn in (leg.entry, leg.exit)):
            return LEG_UNCERTAIN, LEG_TURN
        return LEG_SCATTER, LEG_TURN
    if gate and not leg_law_holds(leg):
        return LEG_SCATTER, LEG_FORCE
    discrepancies = (
        (not leg.clean, LEG_NOT_CLEAN),
        (not leg_law_holds(leg), LEG_FORCE),
        (leg.energy != CheckValue.PASS, LEG_ENERGY),
        (not all(reflects), LEG_TURN),
    )
    first = next((name for failed, name in discrepancies if failed), '')
    return (LEG_UNCERTAIN, first) if first else (LEG_TRANSIENT, '')


def deciding_leg(geometry: Optional[Geometry], settings: AnalysisSettings) -> Optional[Tuple[SingleLeg, str, str]]:
    """(leg, outcome, discrepancy) of the 1:1 leg that decides leaf G4c: the strongest outcome over
    `Geometry.single_legs` (`LEG_RANK`; the longest leg among equals), None when no leg decides anything."""
    if geometry is None or not geometry.single_legs:
        return None
    read = [(leg, *leg_outcome(leg, settings)) for leg in geometry.single_legs]
    read = [item for item in read if item[1]]
    if not read:
        return None
    return max(read, key=lambda item: (LEG_RANK[item[1]], item[0].t1_yr - item[0].t0_yr))


def single_leg_columns(geometry: Optional[Geometry], settings: AnalysisSettings) -> dict:
    """The deciding leg of leaf G4c (mmr_numbers.csv): span, topology, outcome and its first discrepancy."""
    found = deciding_leg(geometry, settings)
    if found is None:
        return {'geo_leg_t0_yr': NAN, 'geo_leg_t1_yr': NAN, 'geo_leg_topology': '', 'geo_leg_outcome': '', 'geo_leg_discrepancy': ''}
    leg, outcome, discrepancy = found
    return {
        'geo_leg_t0_yr': leg.t0_yr,
        'geo_leg_t1_yr': leg.t1_yr,
        'geo_leg_topology': leg.topology,
        'geo_leg_outcome': outcome,
        'geo_leg_discrepancy': discrepancy,
    }


def _leg_entries(leg: SingleLeg, settings: AnalysisSettings, ledger: Ledger) -> None:
    """The deciding leg's comparisons on the ledger (the decision path and the figure read them)."""
    ledger.compare(G4C_CLEAN, _flag(leg.clean), Op.EQ, 1, '', 'geo_leg_topology')
    for turn in leg.turns:
        out_in = abs(math.log(turn.out_in)) if is_finite(turn.out_in) and turn.out_in > 0 else None
        ledger.compare_setting(G4C_TURN, out_in, Op.LE, settings, 'mmr_coorbital_turn_energy_tol', '', 'geo_leg_read')
        zone = turn.zone if is_finite(turn.zone) else None
        ledger.compare_setting(G4C_ZONE, zone, Op.LE, settings, 'mmr_coorbital_turn_zone_max', '', 'geo_leg_read')
    ledger.compare(G4C_FORCE, _flag(leg_law_holds(leg)), Op.EQ, 1, '', 'geo_leg_read')
    ledger.compare(G4C_ENERGY, _flag(leg.energy == CheckValue.PASS), Op.EQ, 1, '', 'geo_leg_read')


def _g4c(angle: AngleDecision, geometry: Optional[Geometry], settings: AnalysisSettings, ledger: Ledger) -> Optional[StatusDecision]:
    """Leaf G4c (W28 C6, rule coorbital_leg; cards g02-4 A, g02-5 B, g02-6 B, g02-7 A, g01-5 A, q-4 A): a prograde
    1:1 circulation without a librating episode is judged on its best single leg (`deciding_leg`; the legs outside
    every accepted run, `Geometry.single_legs`) before the circulation leaf:

    - LEG_TRANSIENT -> transient/dynamic_episode, origin dynamic, the angle word episodic, cap one_coorbital_leg
      (<= medium: half a cycle, no closed one; 2023 OE5);
    - LEG_UNCERTAIN -> uncertain/one_coorbital_leg, origin unknown, the word episodic, action review (2018 KS: a
      backward lap past the Earth at 779-934 yr);
    - LEG_EXTEND -> uncertain/baseline_needed, the word insufficient_time, extend (a clean leg the record edge cuts);
    - LEG_SCATTER -> None here: the circulation leaf decides, `decide_status` adds the near-episode cap (1991 VG:
      552-613 yr, out/in 0.61 and 1.59).

    None without a deciding leg (2008 EZ84: one reflection, no leg; it stays high) and for an alias or another angle.
    """
    if not settings.mmr_coorbital_single_leg or geometry is None or not geometry.coorbital:
        return None
    if angle.angle_status != AngleStatus.CIRCULATION or angle.subtype == AngleSubtype.ALIAS:
        return None
    found = deciding_leg(geometry, settings)
    if found is None:
        return None
    leg, outcome, _ = found
    _leg_entries(leg, settings, ledger)
    fired = rules_module.fire(frozenset(), rules_module.Rule.COORBITAL_LEG)
    if outcome == LEG_TRANSIENT:
        return StatusDecision(
            Status.TRANSIENT,
            Reason.DYNAMIC_EPISODE,
            LibrationOrigin.DYNAMIC,
            '',
            '',
            caps=[str(Cap.ONE_COORBITAL_LEG)],
            rule=RULE_G4C,
            angle_word=AngleStatus.EPISODIC,
            rules_fired=fired,
        )
    if outcome == LEG_UNCERTAIN:
        return StatusDecision(
            Status.UNCERTAIN,
            Reason.ONE_COORBITAL_LEG,
            LibrationOrigin.UNKNOWN,
            '',
            '',
            rule=RULE_G4C,
            angle_word=AngleStatus.EPISODIC,
            rules_fired=fired,
        )
    if outcome == LEG_EXTEND:
        return StatusDecision(
            Status.UNCERTAIN,
            Reason.BASELINE_NEEDED,
            LibrationOrigin.NONE,
            '',
            '',
            rule=RULE_G4C,
            angle_word=AngleStatus.INSUFFICIENT_TIME,
            rules_fired=fired,
        )
    return None


RULE_G4H = 'G4h'
G4H_LAW = 'G4h half-swing between passages: R2 of the one-harmonic law, the smaller half'
G4H_RESTORING = 'G4h the fitted force turns sigma back at both turning points'


def half_libration_holds(half: HalfLibration, settings: AnalysisSettings) -> bool:
    """The half-swing follows the resonance's one-harmonic law: R^2 >= mmr_force_r2_fail on both halves of its
    window, and the fitted force turns sigma back at both turning points."""
    smaller = min(half.r2_h1, half.r2_h2)
    return bool(half.restoring and is_finite(smaller) and smaller >= settings.mmr_force_r2_fail)


def deciding_half_libration(geometry: Optional[Geometry], settings: AnalysisSettings) -> Optional[HalfLibration]:
    """The half-swing that decides leaf G4h: of `Geometry.half_librations` those whose one-harmonic law reaches
    mmr_force_r2_fail on both halves and turns sigma back at both turning points; one between two passages first,
    then the longest. None without one."""
    if geometry is None or not geometry.half_librations:
        return None
    held = [half for half in geometry.half_librations if half_libration_holds(half, settings)]
    return max(held, key=lambda half: (not half.edge, half.t1_yr - half.t0_yr), default=None)


def _g4h(angle: AngleDecision, geometry: Optional[Geometry], settings: AnalysisSettings, ledger: Ledger) -> Optional[StatusDecision]:
    """Leaf G4h (W28 C6, rule single_half_libration, q-5 i): an angle that is not 1:1, circulating (G4, not an alias)
    without an accepted run, whose bounded half-swing between passages follows the resonance's own one-harmonic law
    (`deciding_half_libration`): between two passages -> uncertain/one_half_libration, the word episodic, action
    review; next to the record's edge -> uncertain/baseline_needed, the word insufficient_time (extend: 42946 43S-6,
    28.5-32.3 kyr to the end, R^2 0.72/0.62). Without the law nothing changes: a slow wide turn of a circulation is no
    reflection by the resonance's potential (the W28 C15 audit: R^2 < 0.06 on most such swings)."""
    if not settings.mmr_single_half_libration or geometry is None or geometry.coorbital:
        return None
    if angle.angle_status != AngleStatus.CIRCULATION or angle.subtype == AngleSubtype.ALIAS:
        return None
    half = deciding_half_libration(geometry, settings)
    if half is None:
        return None
    column = 'geo_half_libration_read'
    ledger.compare_setting(G4H_LAW, min(half.r2_h1, half.r2_h2), Op.GE, settings, 'mmr_force_r2_fail', '', column)
    ledger.compare(G4H_RESTORING, _flag(half.restoring), Op.EQ, 1, '', column)
    fired = rules_module.fire(frozenset(), rules_module.Rule.SINGLE_HALF_LIBRATION)
    if half.edge:
        word, reason, origin = AngleStatus.INSUFFICIENT_TIME, Reason.BASELINE_NEEDED, LibrationOrigin.NONE
    else:
        word, reason, origin = AngleStatus.EPISODIC, Reason.ONE_HALF_LIBRATION, LibrationOrigin.UNKNOWN
    return StatusDecision(Status.UNCERTAIN, reason, origin, '', '', rule=RULE_G4H, angle_word=word, rules_fired=fired)


def leg_scatters(angle: AngleDecision, geometry: Optional[Geometry], settings: AnalysisSettings) -> bool:
    """Leaf G4c's LEG_SCATTER on this angle: the best leg's turns both scatter (the near-episode of a 1:1 leg)."""
    if not settings.mmr_coorbital_single_leg or geometry is None or not geometry.coorbital:
        return False
    if angle.angle_status != AngleStatus.CIRCULATION or angle.subtype == AngleSubtype.ALIAS:
        return False
    found = deciding_leg(geometry, settings)
    return found is not None and found[1] == LEG_SCATTER


def near_episode_threshold(settings: AnalysisSettings) -> int:
    """TPs a short run needs to be a near-episode: libration_min_turning_points - libration_near_episode_tp_deficit."""
    return settings.libration_min_turning_points - settings.libration_near_episode_tp_deficit


def near_episode(
    angle: AngleDecision, decision: StatusDecision, geometry: Optional[Geometry], settings: AnalysisSettings, ledger: Ledger
) -> bool:
    """Flag near_episode and cap at medium (author, W4): a verdict without libration (non_resonant, no episode) next
    to a run short of an episode by at most libration_near_episode_tp_deficit TPs (`near_episode_tp`). The
    episode threshold stays (author's decision); what is thin is the margin by which the record missed it: one more
    turn back and the same record would read transient (2016 CA138: one half-cycle of capture, author transient).
    Recorded in the ledger only when a short run exists."""
    if not _claims_no_libration(angle, decision, geometry) or settings.libration_near_episode_tp_deficit <= 0:
        return False
    count = near_episode_tp(geometry, settings)
    if not count:
        return False
    name = 'near-episode: TPs of the longest short run'
    threshold = near_episode_threshold(settings)
    entry = ledger.compare(name, count, Op.GE, threshold, '', 'geo_near_episode_n_tp', setting='libration_near_episode_tp_deficit')
    return bool(entry)


def _decide_status(
    angle: AngleDecision,
    sampling: Sampling,
    geometry: Optional[Geometry],
    energy: Optional[EnergyAnalysis],
    response: Optional[Response],
    kinematics: Optional[Kinematics],
    subresonances: Optional[SubresonanceResult],
    owner: Optional[OwnerResult],
    strength: Optional[Strength],
    settings: AnalysisSettings,
    ledger: Ledger,
    member: Optional[Verdict] = None,
    episodes: Sequence[EpisodePhysics] = (),
) -> StatusDecision:
    angle = status_basis(angle)
    if angle.angle_status == AngleStatus.UNDEFINED:
        return _undefined(angle)
    if angle.angle_status == AngleStatus.INSUFFICIENT_TIME:
        return StatusDecision(Status.UNCERTAIN, Reason.BASELINE_NEEDED, LibrationOrigin.NONE, '', '', rule='G6')
    kept, outside = owner_in_episode(owner, settings, ledger, decisive=angle.angle_status != AngleStatus.CIRCULATION)
    decision = _decide_read(
        angle, sampling, geometry, energy, response, kinematics, subresonances, kept, strength, settings, ledger, member, episodes
    )
    if outside:  # F7/P2: the owner does not make the own episodes; its key goes to the verdict's neighbours (`roles`)
        decision.flags.extend(outside)
        decision.owner = decision.owner or owner.accepted.spec.key
    decision.rules_fired = decision.rules_fired | owner_rules(owner, outside, settings)
    return decision


def _decide_read(
    angle: AngleDecision,
    sampling: Sampling,
    geometry: Optional[Geometry],
    energy: Optional[EnergyAnalysis],
    response: Optional[Response],
    kinematics: Optional[Kinematics],
    subresonances: Optional[SubresonanceResult],
    owner: Optional[OwnerResult],
    strength: Optional[Strength],
    settings: AnalysisSettings,
    ledger: Ledger,
    member: Optional[Verdict] = None,
    episodes: Sequence[EpisodePhysics] = (),
) -> StatusDecision:
    """The tree below G6 on the angle as the status reads it, with the owner F7 left in (`owner_in_episode`)."""
    if angle.angle_status == AngleStatus.CIRCULATION:
        leg = _g4c(angle, geometry, settings, ledger) or _g4h(angle, geometry, settings, ledger)  # W28 C6: one leg
        if leg is not None:
            return leg
        return _circulation(angle, sampling, geometry, energy, subresonances, owner, settings, ledger, member)
    foreign = _foreign(angle, subresonances, owner, settings, ledger, member)
    if foreign is not None:
        return foreign
    decision = _own_read(angle, sampling, geometry, energy, response, kinematics, strength, settings, ledger, episodes)
    if owner is not None and owner.accepted is not None and owner.capture is not None:
        if owner.capture.verdict == CAPTURE_CONFIRMED and not owner.capture.route:  # A2b: `_foreign` stepped aside
            decision.joint_owner = owner.accepted.spec.key
    return decision


def _own_read(
    angle: AngleDecision,
    sampling: Sampling,
    geometry: Optional[Geometry],
    energy: Optional[EnergyAnalysis],
    response: Optional[Response],
    kinematics: Optional[Kinematics],
    strength: Optional[Strength],
    settings: AnalysisSettings,
    ledger: Ledger,
    episodes: Sequence[EpisodePhysics] = (),
) -> StatusDecision:
    """The tree after steps 1 and 2: the angle's own reading (kinematic, drift, contaminated force law, force law x
    potential)."""
    kinematic = _kinematic(angle, geometry, response, kinematics, settings, ledger)
    if kinematic is not None:
        return kinematic
    if angle.angle_status == AngleStatus.DRIFT:
        return StatusDecision(Status.UNCERTAIN, Reason.DRIFT_MECHANISM_UNKNOWN, LibrationOrigin.UNKNOWN, '', '', rule='4')
    # a sigma-only R0 failure (period law, fixed well: no a_dot in it) stays an objection: 5e (Codex W15)
    sigma_only_fails = energy is not None and energy.sigma_only == CheckValue.FAIL
    if response is not None and response.force_contaminated and not sigma_only_fails:
        return _contaminated(response, settings, ledger)
    response, unmeasured = short_period_unmeasured(sampling, response, strength, settings, ledger)
    decision = _force_and_potential(angle, geometry, response, energy, settings, ledger, strength, episodes)
    decision.withdrawn.update(unmeasured)
    return decision


# --------------------------------------------------------------------------------------------- R-ALIAS-A

SHORT_PERIOD_A = 'short-period scatter of a / a half-excursion of the decisive episode'
ALIAS_A_CHECKS = {'a_response': 'L4', 'identity': 'L3'}  # the a-based checks: Response field -> margin id
ALIAS_A = f'5 R-ALIAS-A folded step, F passed: failed L4/L3 read as not_measured when the {SHORT_PERIOD_A} is above'


def a_scatter_ratio(sampling: Sampling, a_excursion_au: float) -> float:
    """smp_da_sp_over_a_excursion: the record's short-period scatter of a (`Sampling.short_period_da_au`) over the a
    half-excursion of the decisive episode (`EpisodePhysics.a_excursion_meas_au`); nan without that excursion. The
    analysis computes it once (`Sampling.a_scatter_ratio`)."""
    da_short = sampling.short_period_da_au
    return float(da_short / a_excursion_au) if is_finite(a_excursion_au) and a_excursion_au > 0 else math.nan


def short_period_unmeasured(
    sampling: Sampling, response: Optional[Response], strength: Optional[Strength], settings: AnalysisSettings, ledger: Ledger
) -> Tuple[Optional[Response], Dict[str, Withdrawal]]:
    """R-ALIAS-A (W22, author 2026-10-24): on a record whose step folds short periods (flag short_period_aliased, the
    cap's trigger), the folded lines sit in a at up to 1.65 of the decisive half-excursion while they move sigma_f by
    <= 0.024 of its half-amplitude (W20 round D), so they lower R^2 and the slope of the a-based checks: a failed force
    law (L4) and identity (L3) are read as not_measured when the strength by direct averaging passes (F, the one force
    measurement left) and the record's short-period scatter of a exceeds mmr_short_period_a_ratio x the decisive a
    half-excursion (`Sampling.a_scatter_ratio`, the flag a_short_period_scatter's number; 2001 ND13 1V-3 at 0.09 keeps
    its fail, which its fine
    rerun confirms). A fail never becomes a pass, the cap stays. Returns the response the tree reads and the checks
    read as not_measured by margin id (the confidence reads L3/L4 the same way, `margins_as_tree_read`)."""
    unchanged = response, {}
    if not settings.mmr_short_period_a_unmeasured or response is None:
        return unchanged
    if str(Flag.SHORT_PERIOD_ALIASED) not in sampling.flags:
        return unchanged
    failed = tuple(name for name in ALIAS_A_CHECKS if getattr(response, name) == CheckValue.FAIL)
    if not failed or strength is None or strength.method not in DIRECT_METHODS or strength.check != CheckValue.PASS:
        return unchanged
    ratio = sampling.a_scatter_ratio
    if not ledger.compare_setting(ALIAS_A, ratio, Op.GT, settings, 'mmr_short_period_a_ratio', '', 'smp_da_sp_over_a_excursion'):
        return unchanged
    withdrawn = {ALIAS_A_CHECKS[name]: Withdrawal(WITHDRAWN_ALIAS_A, name) for name in failed}
    return replace(response, **{field_name: CheckValue.NOT_MEASURED for field_name in failed}), withdrawn


def withdrawn_checks(decision: StatusDecision, guard: str) -> Tuple[str, ...]:
    """The checks `guard` withdrew from the tree (`StatusDecision.withdrawn`), in its own spelling, in order."""
    return tuple(item.check for item in decision.withdrawn.values() if item.guard == guard)


def margins_as_tree_read(margins: Mapping[str, MarginValue], decision: StatusDecision) -> Mapping[str, MarginValue]:
    """The margins the confidence reads: a failed margin whose check the tree read as not_measured
    (`StatusDecision.withdrawn`: R-ALIAS-A's failed L3 or L4; the filter guard withdraws only passed checks, whose
    margins it empties itself, `apply_filter_sensitivity`) counts as not_measured, as the tree read it; the margins
    column keeps the measured fail. `margins` itself when nothing changes."""
    failed = [
        margin_id for margin_id in decision.withdrawn if margins.get(margin_id) is not None and margins[margin_id].margin == Margin.FAIL
    ]
    if not failed:
        return margins
    read = dict(margins)
    for margin_id in failed:
        read[margin_id] = replace(read[margin_id], margin=Margin.NOT_MEASURED)
    return read


# --------------------------------------------------------------------------------------------- baseline


def baseline(
    geometry: Optional[Geometry],
    cutoff_yr: float,
    span_yr: float,
    fallback_period_yr: float,
    settings: AnalysisSettings,
    angle: Optional[AngleDecision] = None,
    kinematics: Optional[Kinematics] = None,
) -> Baseline:
    """required = C P + 2 filter_edge_trim_cutoffs P_c [yr], C = libration_min_turning_points / 2 (mmr_extend_slow_cycles
    for G5w, `baseline_cycles`); extend to a round value
    >= max(required, f_min T) (`mmr_extend_min_span_factor`, 2; the physical estimate N, column base_extend_to_years:
    an insufficient_time verdict asks for `next_check` instead); check pass when the span suffices, fail -> extend,
    not_measured without a period.

    P is the time scale that decides (`baseline_period`; F8, W20): for an insufficient_time angle the slow one of its
    subtype, else the measured cycle, the shortest period a stall allows, B5's P_min (`fallback_period_yr`). Without any
    period the extension is the round value >= f_min T. An insufficient_time angle whose P fits the record (required <= T)
    is not short of time: `Baseline.cause` names what is missing instead."""
    period, source, run = baseline_period(angle, geometry, kinematics, fallback_period_yr)
    if not is_finite(period) or period <= 0:
        extend = round_up(min_extension_yr(span_yr, settings))
        return Baseline(float('nan'), extend, CheckValue.NOT_MEASURED, float('nan'), '')
    required = baseline_cycles(angle, settings) * period + 2 * settings.filter_edge_trim_cutoffs * cutoff_yr
    enough = Ledger.from_settings(settings).compare('baseline', span_yr, Op.GE, required, 'yr', 'smp_span_yr')
    extend = None if enough else round_up(max(required, min_extension_yr(span_yr, settings)))
    cause = ''
    if enough and angle is not None and angle.angle_status == AngleStatus.INSUFFICIENT_TIME:
        cause = str(run.reject_reason) if run is not None and str(run.reject_reason) else BASE_CAUSE_PERIOD_FITS
    return Baseline(required, extend, CheckValue.PASS if enough else CheckValue.FAIL, period, source, cause)


def baseline_cycles(angle: Optional[AngleDecision], settings: AnalysisSettings) -> float:
    """The cycles of P the record needs: one full cycle (libration_min_turning_points / 2), or for a slow swing in the
    well (G5w, W28 B6c) mmr_extend_slow_cycles slow periods (a budget convention: 523683 2U-9, P 112 kyr in 100 kyr,
    asks for 400 kyr instead of 200 kyr, less than two slow cycles)."""
    if angle is not None and angle.rule == RULE_G5W:
        return float(settings.mmr_extend_slow_cycles)
    return settings.libration_min_turning_points / 2


BASE_CAUSE_PERIOD_FITS = 'period_fits_record'  # Baseline.cause without a run: P fits the record, nothing else known


def extend_years(status, angle_status, base: Baseline, span_yr: float, settings: AnalysisSettings) -> Optional[float]:
    """The total length an uncertain verdict asks for (shown with action extend): the baseline's estimate N, else
    round_up(f_min T); for an insufficient_time angle the next check instead (`next_check`, N8). None otherwise."""
    if status != Status.UNCERTAIN:
        return None
    extend = base.extend_to_years if base.extend_to_years is not None else round_up(min_extension_yr(span_yr, settings))
    return next_check(extend, span_yr, settings) if angle_status == AngleStatus.INSUFFICIENT_TIME else extend


def min_extension_yr(span_yr: float, settings: AnalysisSettings) -> float:
    """The floor of every extension, f_min T [yr] (`mmr_extend_min_span_factor` x the record length), unrounded."""
    return settings.mmr_extend_min_span_factor * span_yr


def next_check(estimate_yr: float, span_yr: float, settings: AnalysisSettings) -> float:
    """N8 (Codex, author 2026-10-24): the total length [yr] of the next check of an insufficient_time record,
    round_up(max(f_min T, min(N, f_next T))) with N the estimate from the period (`baseline`), T the record length,
    f_min = `mmr_extend_min_span_factor`, f_next = `mmr_next_check_span_factor`. A budget, not a sufficiency estimate:
    the record is analysed again at that length (613440: 4 kyr decide where N asked for 200 kyr; 2009 BE81 and
    2017 XG62 need more than their N)."""
    low = min_extension_yr(span_yr, settings)
    if not is_finite(estimate_yr):
        return round_up(low)
    return round_up(max(low, min(estimate_yr, settings.mmr_next_check_span_factor * span_yr)))


# --------------------------------------------------------------------------------------------- B9: length-limited extend

# The not-measured reasons a longer record removes (W28 B9; cards g08-5 B, g08-6 A, g06-6 A, g06-7 A). Names, not
# thresholds: each is read from the measurement's own reason (`force_windows_short`, the leaf 5n itself).
LENGTH_FORCE_WINDOWS = 'force_windows'  # L4 not_measured with fewer than mmr_force_min_windows windows
LENGTH_CYCLES = 'independent_cycles'  # 5n: fewer than libration_centre_min_estimates independent cycles
LENGTH_LIMITED = frozenset({LENGTH_FORCE_WINDOWS, LENGTH_CYCLES})
# The uncertain leaves where nothing measured the angle's own force, and the not-measured check that decides each:
# 5e (R0 failed, L4 missing: TX9), 5g (the identity fails, the force law that would tell a kinematic cycle is missing:
# 2003 EH16), 5x (the scale out of range, nothing measured: 79653), 5n (one independent cycle: 2003 ND). The objection
# leaves (5a, 5a*, 5b, 5f, 5o, 5s*) keep review: a longer record does not remove a measured objection.
EXTEND_LEAVES = {'5e': LENGTH_FORCE_WINDOWS, '5g': LENGTH_FORCE_WINDOWS, '5x': LENGTH_FORCE_WINDOWS, '5n': LENGTH_CYCLES}


def force_windows_short(response: Optional[Response], settings: AnalysisSettings) -> bool:
    """L4 is not measured for lack of length: not_measured with fewer than mmr_force_min_windows windows of
    mmr_force_window_cycles periods, and not for a forced line at its period (5f). L4 not measured with enough windows
    (a swing below mmr_a_response_min_amp_deg, or R-ALIAS-A's withdrawal) is not length-limited."""
    if response is None or response.a_response != CheckValue.NOT_MEASURED or response.force_contaminated:
        return False
    return len(response.windows) < settings.mmr_force_min_windows


def last_episode_at_end(geometry: Geometry, settings: AnalysisSettings) -> Optional[Episode]:
    """The last librating episode (not before a jump of a) that reaches the record's end (`covers_edges`), else None:
    only such an episode gains windows and cycles from a longer record."""
    episodes = [e for e in librating_episodes(geometry, settings) if covers_edges(e, settings)[1]]
    return max(episodes, key=lambda e: e.t_end_yr, default=None)


def force_window_extension(episode: Episode, settings: AnalysisSettings) -> float:
    """A-FW [yr]: the total length that gives the last episode mmr_extend_force_windows force-law windows,
    t_start + mmr_extend_force_windows x mmr_force_window_cycles x P (its median period), rounded up (`round_up`)."""
    windows = settings.mmr_extend_force_windows * settings.mmr_force_window_cycles * episode.period_med_yr
    return round_up(episode.t_start_yr + windows)


def cycles_extension(episode: Episode, cutoff_yr: float, settings: AnalysisSettings) -> float:
    """N of leaf 5n [yr]: the time the episode's turning points need to hold libration_centre_min_estimates independent
    cycles (2 m + 1 TPs, `geometry` counts (n_TP - 1) // 2), at least one more TP, from its last TP at half its period
    each, plus the edge trim filter_edge_trim_cutoffs x P_c that hides the record's end."""
    missing = max(2 * settings.libration_centre_min_estimates + 1 - len(episode.tp_times_yr), 1)
    trim = settings.filter_edge_trim_cutoffs * cutoff_yr
    return float(episode.tp_times_yr[-1] + missing * episode.period_med_yr / 2 + trim)


def length_limited_extension(
    decision: StatusDecision,
    angle: AngleDecision,
    geometry: Optional[Geometry],
    response: Optional[Response],
    span_yr: float,
    cutoff_yr: float,
    settings: AnalysisSettings,
) -> Optional[float]:
    """W28 B9: extend_to [yr] for an uncertain verdict whose deciding not-measured check (`EXTEND_LEAVES`) is
    length-limited, else None (the action stays review). `angle` is the angle the status read (`status_basis`),
    `response` the one the tree read (rule R).

    Only the last episode reaching the record's end gains from a longer record (`last_episode_at_end`). L4 short of
    windows asks for A-FW (`force_window_extension`); leaf 5n, when its decisive episode is that last one, asks for the
    N8 next check of its cycles (`next_check` of `cycles_extension`: round_up(max(f_min T, min(N, f_next T)))). The
    result is the longest of them that exceeds the record; none exceeding it is no length limit."""
    deciding = EXTEND_LEAVES.get(decision.rule)
    if decision.status != Status.UNCERTAIN or deciding is None or geometry is None:
        return None
    late = last_episode_at_end(geometry, settings)
    if late is None or not is_finite(late.period_med_yr):
        return None
    lengths = []
    if force_windows_short(response, settings):
        lengths.append(force_window_extension(late, settings))
    elif deciding == LENGTH_FORCE_WINDOWS:
        return None
    if deciding == LENGTH_CYCLES:
        if angle.episode_id != late.episode_id:
            return None
        lengths.append(next_check(cycles_extension(late, cutoff_yr, settings), span_yr, settings))
    longest = max((x for x in lengths if is_finite(x) and x > span_yr), default=None)
    return float(longest) if longest is not None else None


# --------------------------------------------------------------------------------------------- C14: the record's edges

# `RecordEdges.direction`: the record edge a longer integration would resolve (the flag extend_time)
EXTEND_FORWARD, EXTEND_BACKWARD, EXTEND_BOTH = 'forward', 'backward', 'both'
EDGE_DIRECTION = {LEG_EDGE_END: EXTEND_FORWARD, LEG_EDGE_START: EXTEND_BACKWARD}
# words that already say how the record ends (or describe it whole): no edge stretch is read under them
WORDS_OF_THEIR_OWN = (AngleStatus.LIBRATION, AngleStatus.DRIFT, AngleStatus.INSUFFICIENT_TIME, AngleStatus.UNDEFINED)
# the final regimes an end arc may replace: none, or the circulation of G1c / an observed exit
REPLACEABLE_REGIMES = ('', str(AngleStatus.CIRCULATION))


@dataclass(frozen=True)
class RecordEdges:
    """What the record's edges add to a verdict (W28 C14, rule end_arc_extend, `record_edges`): the final regime (an
    end arc or stall after circulation replaces '' or circulation), the cap record_ends_unresolved, the direction of
    the flag extend_time ('' without it) and its length (`extend_to_years`), and what set the flag (`sources`:
    'end:slowdown', 'start:reflection', 'leg:end', 'half:start', ...)."""

    final_regime: str = ''
    final_regime_start_yr: float = NAN
    caps: Tuple[str, ...] = ()
    direction: str = ''
    extend_to_years: Optional[float] = None
    sources: Tuple[str, ...] = ()

    @property
    def flags(self) -> List[str]:
        """[extend_time] with a direction, else []."""
        return [str(Flag.EXTEND_TIME)] if self.direction else []

    @property
    def rules_fired(self) -> frozenset:
        """end_arc_extend when the rule acted (the flag is set)."""
        return rules_module.fire(frozenset(), rules_module.Rule.END_ARC_EXTEND) if self.direction else frozenset()


def edge_stretch_holds(
    stretch: EdgeStretch, geometry: Geometry, episodes: Sequence[Episode], floor_yr: float, settings: AnalysisSettings, ledger: Ledger
) -> bool:
    """An edge stretch (`Geometry.edge_stretches`) the rule reads: of a kind (slowdown, or a reflection of a 1:1 angle:
    q-5 i reads a non-1:1 turn only through G4h's law), at the record's own edge (not a window start after a jump of
    a), no librating episode in it (the episode rules read those edges), 2 x its length at least
    mmr_end_arc_min_floor_periods floor periods (the stretch holds no turn of the slow centre, so a cycle through it
    lasts at least 2 D, read against the floor as a 1:1 leg's 2 L is, `geometry.coorbital_single_legs`; 2008 EZ84's
    start, 87 yr, is 1.57 floors), and a slowdown longer than mmr_end_arc_slowdown_factor x the longest revolution of
    its pass and, when the slow centre reverses inside the window, than the longest time between two of its turns (a
    circulation near a* that reverses on its own slow scale slows before every reversal: 37908 23J-8, 323879 6V-41)."""
    if not stretch.kind:
        return False
    if stretch.side == LEG_EDGE_START and Flag.LAST_REGIME_AFTER_JUMP in geometry.flags:
        return False  # the window starts at a jump of a, not at the record's start: no backward integration reaches it
    if stretch.kind == EDGE_REFLECTION and not geometry.coorbital:
        return False  # q-5 i: a slow turn of a non-1:1 circulation is no reflection by its potential (G4h reads those)
    edge = geometry.window[1] if stretch.side == LEG_EDGE_END else geometry.window[0]
    lo, hi = sorted((stretch.t_mark_yr, edge))
    if any(episode.t_end_yr > lo and episode.t_start_yr < hi for episode in episodes):
        return False
    prefix = f'{RULE_END_ARC} {stretch.side} {stretch.kind}'
    column = f'geo_{stretch.side}_edge_duration_yr'
    length = 2 * stretch.duration_yr / floor_yr if floor_yr > 0 else math.inf
    if not ledger.compare_setting(f'{prefix}: 2 x length / floor', length, Op.GE, settings, 'mmr_end_arc_min_floor_periods', 'P', column):
        return False
    if stretch.kind != EDGE_SLOWDOWN:
        return True
    ratio = stretch.duration_yr / stretch.last_rev_yr if is_finite(stretch.last_rev_yr) and stretch.last_rev_yr > 0 else None
    name = f'{prefix}: length / the longest revolution'
    if not ledger.compare_setting(name, ratio, Op.GT, settings, 'mmr_end_arc_slowdown_factor', '', column):
        return False
    if not is_finite(stretch.reversal_yr):
        return True
    name = f'{prefix}: length against the longest time between turns of the slow centre'
    return bool(ledger.compare(name, stretch.duration_yr, Op.GT, stretch.reversal_yr, 'yr', f'geo_{stretch.side}_edge_reversal_yr'))


RULE_END_ARC = 'end arc'


def record_edges(
    geometry: Optional[Geometry],
    decision: StatusDecision,
    reported: AngleStatus | str,
    regime: str,
    regime_start_yr: float,
    span_yr: float,
    settings: AnalysisSettings,
    ledger: Ledger,
) -> RecordEdges:
    """Rule end_arc_extend (W28 C14; cards g01-7 A, g01-4 A, g02-4 A; MMR_RECORD_EDGE_STRETCH), after the tree and the
    final regime (`final_regime`: `regime`, `regime_start_yr`); `reported` is the verdict's angle word.

    - End slowdown (`edge_stretch_holds`, the reported word not one of `WORDS_OF_THEIR_OWN`): final_regime arc
      (range >= pi, as G6) or stall from the last completed revolution, when the final regime was '' or circulation;
      flag extend_time forward; a non_resonant circulation gets the cap record_ends_unresolved (<= medium: the record
      ends before it shows whether sigma circulates on or is captured; 2022 BB2's last 500 yr). Its comparisons go to
      `ledger` only then.
    - Start slowdown, and one reflection of a 1:1 angle at either edge (2008 EZ84 at 144 yr): the flag only (backward
      at the start, forward at the end), no cap: one reflection is no cycle (g01-4 A keeps EZ84 high).
    - Without a librating episode: a clean 1:1 edge leg whose turn reflects (leaf G4c's LEG_EXTEND) and an edge
      half-libration that holds its law (G4h) give the flag of their edge as well (q-4 A: cut by the edge -> extend).

    The flag's length: the N8 next check without an estimate, round_up(max(f_min T, ...)) = round_up(f_min T)
    (`next_check` of `min_extension_yr`; no 1:1 exit estimate, D18 rejected). Comparisons that only set the flag are
    read on a scratch ledger: the flag changes no status and no level."""
    unchanged = RecordEdges(regime, regime_start_yr)
    if not settings.mmr_record_edge_stretch or geometry is None:
        return unchanged
    episodes = librating_episodes(geometry, settings)
    floor = period_floor(geometry, settings)[0]
    found = edge_sources(geometry, reported, episodes, floor, settings)
    if not found:
        return unchanged
    caps: Tuple[str, ...] = ()
    arc = next((stretch for side, label, stretch in found if side == LEG_EDGE_END and label == EDGE_SLOWDOWN), None)
    if arc is not None and regime in REPLACEABLE_REGIMES:
        regime, regime_start_yr = str(AngleSubtype.ARC if arc.range_rad >= np.pi else AngleSubtype.STALL), arc.t_mark_yr
        if decision.status == Status.NON_RESONANT and str(reported) == AngleStatus.CIRCULATION:
            edge_stretch_holds(arc, geometry, episodes, floor, settings, ledger)  # the cap's numbers on the record
            caps = (str(Cap.RECORD_ENDS_UNRESOLVED),)
    sides = {side for side, _, _ in found}
    direction = EXTEND_BOTH if len(sides) == 2 else EDGE_DIRECTION[found[0][0]]
    extend = next_check(min_extension_yr(span_yr, settings), span_yr, settings)
    sources = tuple(dict.fromkeys(f'{side}:{label}' if stretch else f'{label}:{side}' for side, label, stretch in found))
    return RecordEdges(regime, regime_start_yr, caps, direction, extend, sources)


def edge_sources(
    geometry: Geometry, reported: AngleStatus | str, episodes: Sequence[Episode], floor_yr: float, settings: AnalysisSettings
) -> List[Tuple[str, str, Optional[EdgeStretch]]]:
    """(edge, what, the stretch or None) of everything that sets the flag extend_time (`record_edges`): the edge
    stretches that hold (`edge_stretch_holds` on a scratch ledger, not under `WORDS_OF_THEIR_OWN`), and without a
    librating episode the 1:1 edge legs of outcome LEG_EXTEND ('leg') and the edge half-librations that hold their
    law ('half')."""
    scratch = Ledger.from_settings(settings)
    found: List[Tuple[str, str, Optional[EdgeStretch]]] = []
    if str(reported) not in WORDS_OF_THEIR_OWN and is_finite(floor_yr):
        held = (
            stretch for stretch in geometry.edge_stretches if edge_stretch_holds(stretch, geometry, episodes, floor_yr, settings, scratch)
        )
        found += [(stretch.side, stretch.kind, stretch) for stretch in held]
    if not episodes:
        found += [(leg.edge, 'leg', None) for leg in geometry.single_legs if leg.edge and leg_outcome(leg, settings)[0] == LEG_EXTEND]
        found += [(half.edge, 'half', None) for half in geometry.half_librations if half.edge and half_libration_holds(half, settings)]
    return found


def baseline_period(
    angle: Optional[AngleDecision], geometry: Optional[Geometry], kinematics: Optional[Kinematics], b5_period_yr: float
) -> Tuple[float, str, Optional[Episode]]:
    """(P [yr], source, the run read for few_cycles or None): the time scale an insufficient_time angle needs (F8,
    author 2026-10-19), else the generic one. P is the largest of lower bounds, each the time a swing needs to come back:

    - few_cycles: twice the median TP spacing of the longest run with >= 2 TPs (`tp_spacing`; 3753 and 2025 RB7 1 kyr:
      2 x 383.7 yr, not B5's 435 yr) and the slow scale (`slow_timescale`): 385458 9N-17 (runs whose centre moves,
      the 300-kyr record decides at the scale of the centre's turns, 76 kyr) and 1566 1R-7J-4 (two TPs 170 yr apart,
      then 10 kyr of a slow arc; the 200-kyr record circulates);
    - arc: `slow_timescale` (2003 KQ18: the 2 omega cycle, 22.8 kyr);
    - stall: `stall_timescale` (2016 FA13: one revolution at 0.12 rad/kyr, 52 kyr); a stall of the slow centre
      under set-aside ripples (B6b, no `arc_stall`): `slow_timescale`;
    - otherwise, or when the subtype's own scale is not measured: the measured cycle (`cycle`), the shortest period
      a stall's force bound allows (`stall_force`), B5's P_min (`b5`)."""
    subtype = angle.subtype if angle is not None and angle.angle_status == AngleStatus.INSUFFICIENT_TIME else None
    run = longest_window_run(geometry) if geometry is not None and subtype == AngleSubtype.FEW_CYCLES else None
    candidates: List[Tuple[float, str]] = []
    if geometry is not None and subtype == AngleSubtype.FEW_CYCLES:
        if run is not None:
            candidates.append((2 * float(np.median(np.diff(run.tp_times_yr))), 'tp_spacing'))
        candidates += slow_timescale(geometry, kinematics, b5_period_yr)
    elif geometry is not None and subtype == AngleSubtype.ARC:
        candidates = slow_timescale(geometry, kinematics, b5_period_yr)
    elif geometry is not None and subtype == AngleSubtype.STALL and geometry.arc_stall is not None:
        candidates = stall_timescale(geometry, b5_period_yr)
    elif geometry is not None and subtype == AngleSubtype.STALL and ripples(geometry):
        candidates = slow_timescale(geometry, kinematics, b5_period_yr)  # B6b: the slow centre under set-aside ripples
    candidates = [(period, source) for period, source in candidates if is_finite(period) and period > 0]
    if candidates:
        return (*max(candidates), run)
    if geometry is not None and is_finite(geometry.period_cycle_yr):
        return geometry.period_cycle_yr, 'cycle', run
    if geometry is not None and geometry.arc_stall is not None and is_finite(geometry.arc_stall.period_min_yr):
        return geometry.arc_stall.period_min_yr, 'stall_force', run
    if is_finite(b5_period_yr):
        return b5_period_yr, 'b5', run
    return float('nan'), '', run


def longest_window_run(geometry: Geometry) -> Optional[Episode]:
    """The longest (in time) run of the analysis window that is no episode and has >= 2 TPs: before the last jump of a
    and forced waves excluded; None without one."""
    forced = geometry.forced_wave_ids
    runs = [
        run
        for run in geometry.runs
        if not run.accepted and not run.before_last_jump and run.episode_id not in forced and len(run.tp_times_yr) >= 2
    ]
    return max(runs, key=lambda run: run.t_end_yr - run.t_start_yr, default=None)


def no_turn_stretch_yr(geometry: Geometry) -> float:
    """The longest stretch of the analysis window in which sigma's slow part does not turn [yr]: the whole window when
    sigma_f has no TP; else the longest stretch between the slow centre's interior turns (`interior_turn_times`) or from
    one to the window's edge, nan when the window cannot show a turn of the centre (W <= 2 W_c: every turn would lie
    within W_c of an edge; 1999 UQ 8M-9J-4, W_c 50.6 kyr in 20 kyr). A libration's half-period is at least that long,
    so twice it bounds the period from below."""
    start, end = geometry.window
    if len(geometry.turning_points) == 0:
        return float(end - start)
    if end - start <= 2 * geometry.centre.window_yr:
        return float('nan')
    edges = np.concatenate(([start], interior_turn_times(geometry), [end]))
    return float(np.max(np.diff(edges)))


# the `base_period_source` values of a slow swing seen in part (`slow_timescale`)
SLOW_SWING_SOURCES = ('arc_slow', 'b5', 'two_omega', 'modulation')


def slow_timescale(geometry: Geometry, kinematics: Optional[Kinematics], b5_period_yr: float) -> List[Tuple[float, str]]:
    """The (P, source) lower bounds of a slow swing seen in part: twice the longest stretch without a turn (`arc_slow`,
    `no_turn_stretch_yr`), the secular cycle the angle follows when it follows p varpi (the kinematic test fails: the
    shorter of the double omega cycle `two_omega` and the e-cycle `modulation` that is not below that bound) and B5's
    P_min (`b5`); unmeasured ones are nan (2003 KQ18 6J+2S-1: stretch 10.9 kyr, 2 omega 22.8 kyr, B5 18.1 kyr)."""
    bound = 2 * no_turn_stretch_yr(geometry)
    candidates = [(bound, 'arc_slow'), (b5_period_yr, 'b5')]
    if kinematics is not None and kinematics.check == CheckValue.FAIL:
        floor = bound if is_finite(bound) else 0.0
        cycles = [
            (period, name)
            for period, name in ((kinematics.two_omega_period_yr, 'two_omega'), (kinematics.modulation_period_yr, 'modulation'))
            if is_finite(period) and period >= floor
        ]
        candidates += [min(cycles)] if cycles else []
    return candidates


def stall_timescale(geometry: Geometry, b5_period_yr: float) -> List[Tuple[float, str]]:
    """The (P, source) lower bounds of a stall: one revolution at its mean rate 2 pi / |sigma_dot| (`stall_rate`; only
    when the stall's travel over the window reaches the turning points' prominence, else the rate is no measurement),
    the shortest libration period its force bound allows (`stall_force`), twice the window without a turn
    (`arc_slow`) and B5's P_min (`b5`): the record must hold either a revolution or the turns of a libration (2016
    FA13 8M-7J-5: 0.122 rad/kyr -> 51.5 kyr, not P_lib,min = 114 yr)."""
    stall = geometry.arc_stall
    start, end = geometry.window
    candidates = [(2 * no_turn_stretch_yr(geometry), 'arc_slow'), (stall.period_min_yr, 'stall_force'), (b5_period_yr, 'b5')]
    rate = abs(stall.rate_rad_yr)
    if is_finite(rate) and rate > 0 and rate * (end - start) >= geometry.turning_points.prominence_rad:
        candidates.append((TWO_PI / rate, 'stall_rate'))
    return candidates


# --------------------------------------------------------------------------------------------- margins


def _margin(margin: Margin, value, margin_id: str) -> MarginValue:
    number = finite_or_none(value)
    return MarginValue(margin, number, MARGIN_CLASSES[margin_id])


def _grade(scratch: Ledger, margin_id: str, value, full: Tuple[Op, float], thin: Optional[Tuple[Op, float]] = None) -> MarginValue:
    """full when `value op full` holds, thin when `value op thin` holds (thin=None: always thin), else fail;
    not_measured without a value."""
    if not is_finite(value):
        return _margin(Margin.NOT_MEASURED, None, margin_id)
    if scratch.compare(margin_id, value, full[0], full[1], '', ''):
        return _margin(Margin.FULL, value, margin_id)
    if thin is None or scratch.compare(margin_id, value, thin[0], thin[1], '', ''):
        return _margin(Margin.THIN, value, margin_id)
    return _margin(Margin.FAIL, value, margin_id)


def _check_margin(check: CheckValue, value, margin_id: str, thin: bool = False) -> MarginValue:
    return _margin(margin_of_check(check, thin), value, margin_id)


def _worst(*margins: MarginValue) -> MarginValue:
    order = (Margin.FAIL, Margin.NOT_MEASURED, Margin.THIN, Margin.FULL, Margin.NOT_APPLICABLE)
    return min(margins, key=lambda m: order.index(m.margin))


def _potential_margin(energy: Optional[EnergyAnalysis]) -> MarginValue:
    check = energy.with_slow if energy is not None else CheckValue.NOT_MEASURED
    return _check_margin(check, energy.ve_vf_agree_fraction if energy is not None else None, 'R0')


def evaluate_margins(
    view: TreeView,
    decision: StatusDecision,
    context: Context,
    sampling: Sampling,
    geometry: Optional[Geometry],
    kinematics: Optional[Kinematics],
    subresonances: Optional[SubresonanceResult],
    owner: Optional[OwnerResult],
    episodes: List[EpisodePhysics],
    settings: AnalysisSettings,
    prep: Optional[Preprocessed] = None,
    episode_tp_counts: Optional[Mapping[str, Optional[int]]] = None,
    strength: Optional[Strength] = None,
) -> Dict[str, MarginValue]:
    """The applicable margins of SPEC §7.1 (classes from MARGIN_CLASSES), each with its number.

    `view`: the angle, response and energy as the tree read them (`tree_view`). `episode_tp_counts`: TPs of the
    decisive episode at the cutoffs x0.5, x1, x2 ({'half', 'one', 'double'}; L1). Uncertain and invalid verdicts get no
    margins (they have no confidence)."""
    if decision.status in (Status.UNCERTAIN, Status.INVALID) or geometry is None:
        return {}
    angle, response, energy = view.angle, view.response, view.energy
    scratch = Ledger.from_settings(settings)
    margins: Dict[str, MarginValue] = {}
    dynamic = own_dynamics_claimed(decision)
    # K2 (rule 3k2) is graded as the drift its verdict reports: the episodes are a ripple on it, no libration the margins
    # L1-L7 could support (its failed identity is part of the kinematic proof, not a failed support)
    if angle.angle_status in LIBRATING and decision.rule != RULE_K2:
        margins.update(
            _episode_margins(angle, decision, context, geometry, energy, response, episodes, settings, scratch, episode_tp_counts, owner)
        )
    if angle.angle_status in (AngleStatus.EPISODIC, AngleStatus.NEAR_SEPARATRIX) and dynamic:
        margins['Z'] = _zone_margin(energy)
    if angle.angle_status == AngleStatus.NEAR_SEPARATRIX and dynamic:
        margins['L4'] = _l4(response, decision)
        margins['R0'] = _potential_margin(energy)
    margins['L8'] = _l8(sampling, prep, scratch, settings, angle, decision)
    # a circulation is non_resonant by its geometry: the subresonance scan and the owner search only name its origin
    oscillating = angle.angle_status != AngleStatus.CIRCULATION
    if subresonances is not None and oscillating:
        margins['S'] = _s_margin(subresonances, scratch, settings)
    if owner is not None and oscillating:
        margins['W'] = _w_margin(owner, scratch, settings)
    if angle.angle_status == AngleStatus.CIRCULATION:
        margins.update(_circulation_margins(angle, context, sampling, geometry, response, scratch, settings))
    margins.update(_origin_margins(angle, decision, response, kinematics, scratch, settings, geometry))
    near = near_episode_tp(geometry, settings)
    if _claims_no_libration(angle, decision, geometry) and near and settings.libration_near_episode_tp_deficit > 0:
        margins['N'] = _grade(scratch, 'N', near, (Op.LT, near_episode_threshold(settings)))
    if 'L4' in margins and margins['L4'].margin == Margin.NOT_MEASURED and dynamic:
        margins['F'] = strength_margin(strength, owner)
    margins.update(_kinematic_f(decision, strength, margins))
    return margins


def kinematic_f_applies(status, origin, strength: Optional[Strength]) -> bool:
    """F of a verdict (`status`, `origin`) reads in its inverted sense (W22 R-FK, `kinematic_strength_margin`): a
    kinematic non_resonant verdict whose strength was computed by direct averaging or the harmonic (the e^q scale
    gives none). The explained column reads the same predicate (`plotting.verdict_text`)."""
    kinematic = status == Status.NON_RESONANT and origin == LibrationOrigin.KINEMATIC
    return bool(kinematic and strength is not None and strength.method in DIRECT_METHODS)


def _kinematic_f(decision: StatusDecision, strength: Optional[Strength], margins: Mapping[str, MarginValue]) -> Dict[str, MarginValue]:
    """{'F': ...} for a verdict without F whose F reads in the inverted sense (`kinematic_f_applies`), else {}."""
    if 'F' not in margins and kinematic_f_applies(decision.status, decision.origin, strength):
        return {'F': kinematic_strength_margin(strength)}
    return {}


def kinematic_strength_margin(strength: Strength) -> MarginValue:
    """F of a kinematic non_resonant verdict, for a strength computed by direct averaging or the harmonic (W22 R-FK;
    the e^q scale gives none): a strength that cannot make the observed period (check fail) supports the verdict
    fully, otherwise not_measured."""
    return _margin(Margin.FULL if strength.check == CheckValue.FAIL else Margin.NOT_MEASURED, strength.ratio, 'F')


def _origin_margins(
    angle: AngleDecision,
    decision: StatusDecision,
    response,
    kinematics,
    scratch: Ledger,
    settings: AnalysisSettings,
    geometry: Optional[Geometry] = None,
) -> Dict[str, MarginValue]:
    """K1 for a kinematic origin, K1e for a kinematic libration, B5 for non_resonant drift and kinematic libration. K2
    (rule 3k2) decided on the slow centre's correlation, so K1 grades that one, and its verdict is a drift of the centre,
    so B5 grades it too. K3s (rule K3s) adds none: its level is the circulation's the tree read (C1-C5 grade the fine
    scale; the slow scale only names the mechanism of the slow swing, and its failed identity is what B5 would ask)."""
    margins = {}
    if decision.rule == RULE_K3S:
        return margins
    k2 = decision.rule == RULE_K2
    if decision.origin == LibrationOrigin.KINEMATIC and kinematics is not None:
        corr = kinematics.centre_corr_rate if k2 else kinematics.corr_rate
        margins['K1'] = _grade(scratch, 'K1', corr, (Op.GE, settings.mmr_k1_full))
        if decision.reason == Reason.KINEMATIC_LIBRATION:
            margins['K1e'] = kinematic_episode_margin(angle, response, kinematics, geometry, scratch, settings)
    drift_or_kinematic = angle.angle_status == AngleStatus.DRIFT or decision.reason == Reason.KINEMATIC_LIBRATION or k2
    if decision.status == Status.NON_RESONANT and drift_or_kinematic:
        margins['B5'] = _b5_margin(response, scratch, settings)
    return margins


def kinematic_episode_margin(
    angle: AngleDecision,
    response: Optional[Response],
    kinematics: Kinematics,
    geometry: Optional[Geometry],
    scratch: Ledger,
    settings: AnalysisSettings,
) -> MarginValue:
    """K1e (A5, W28, card g05-3 B; support, value: K1 on the episode): the measured physics of a kinematic libration.
    Full when three measurements hold together: the decisive episode's sigma follows p varpi (`Kinematics.k1_episode`
    >= mmr_kin_corr_min, `kinematics.episode_kinematics`), a is flat (the B1 ratio of the response the tree read <
    mmr_kin_b1_clear, far below what a pendulum needs) and the episode's period is a secular one
    (`secular_period_offset` <= mmr_kin_period_tol: the e-cycle or the double omega cycle). The conjunction is the
    measurement of the kinematic cause, so a part that does not hold leaves it not_measured: it argues for no other
    status (41588 3J+9S-2: B1 0.095). not_applicable for a retrograde angle or one without the body's varpi.
    In the physics list of a kinematic verdict (`physics_margins_of`), a full K1e lifts all_physics_not_measured
    (285637 3M+7J-6: 0.971, B1 6e-5, offset 0.004; 481085 7E-10M-6: 1.000, 3.2e-3, 0.004)."""
    if kinematics.k1_episode_reason in (K1E_RETROGRADE, K1E_NO_VARPI):
        return _margin(Margin.NOT_APPLICABLE, None, 'K1e')
    corr = kinematics.k1_episode
    b1 = response.b1_ratio if response is not None else None
    offset = secular_period_offset(decisive_period(angle, geometry), kinematics)
    holds = (
        is_finite(corr)
        and is_finite(b1)
        and is_finite(offset)
        and scratch.compare('K1e', corr, Op.GE, settings.mmr_kin_corr_min, '', '')
        and scratch.compare('K1e B1', b1, Op.LT, settings.mmr_kin_b1_clear, '', '')
        and scratch.compare('K1e period', offset, Op.LE, settings.mmr_kin_period_tol, '', '')
    )
    return _margin(Margin.FULL if holds else Margin.NOT_MEASURED, corr, 'K1e')


def physics_margins_of(decision: StatusDecision, physics: Sequence[str]) -> Tuple[str, ...]:
    """The physics margins of a verdict (confidence.raw_level): `physics` (`strength.physics_margins` of
    PHYSICS_MARGINS) and, for a kinematic origin, KINEMATIC_PHYSICS_MARGINS (K1e; not for K3s, graded as the
    circulation it read)."""
    if decision.origin == LibrationOrigin.KINEMATIC and decision.rule != RULE_K3S:
        return tuple(physics) + KINEMATIC_PHYSICS_MARGINS
    return tuple(physics)


def kinematic_high_rules(decision: StatusDecision, margins: Mapping[str, MarginValue], physics: Sequence[str]) -> frozenset:
    """Rule kinematic_high where K1e acted: it is full and without it every physics margin of the verdict would be
    not_measured (the cap all_physics_not_measured it removed)."""
    k1e = margins.get('K1e')
    if decision.origin != LibrationOrigin.KINEMATIC or k1e is None or k1e.margin != Margin.FULL:
        return frozenset()
    others = [margins[m].margin for m in physics if m in margins and m != 'K1e' and margins[m].margin != Margin.NOT_APPLICABLE]
    if others and any(margin != Margin.NOT_MEASURED for margin in others):
        return frozenset()
    return rules_module.fire(frozenset(), rules_module.Rule.KINEMATIC_HIGH)


def _episode_margins(
    angle: AngleDecision,
    decision: StatusDecision,
    context: Context,
    geometry: Geometry,
    energy: Optional[EnergyAnalysis],
    response: Optional[Response],
    episodes: List[EpisodePhysics],
    settings: AnalysisSettings,
    scratch: Ledger,
    episode_tp_counts: Optional[Mapping[str, Optional[int]]],
    owner: Optional[OwnerResult] = None,
) -> Dict[str, MarginValue]:
    """L1–L7, O1 and R0 of the decisive episode (O1 not_applicable where the owner librates over its background,
    `owner_librates_before`; L3 not_applicable for a kinematic verdict: its failure repeats B1)."""
    episode = run_of(geometry, angle.episode_id)
    if episode is None:
        return {}
    physics = physics_of(episodes, angle.episode_id)
    margins = {'L1': _l1(episode, episode_tp_counts, scratch, settings), 'L2': _l2(episode, energy, scratch, settings)}
    identity = response.identity if response is not None else CheckValue.NOT_MEASURED
    margins['L3'] = _check_margin(identity, response.identity_corr if response is not None else None, 'L3')
    if decision.reason in KINEMATIC_REASONS:  # a without the oscillation (B1) cannot carry the identity (W22 R-L3K)
        margins['L3'] = _margin(Margin.NOT_APPLICABLE, margins['L3'].value, 'L3')
    own = own_dynamics_claimed(decision)
    if own or decision.origin == LibrationOrigin.NONE:
        margins['L4'] = _l4(response, decision)
    margins['L5'] = _l5(physics, energy, context, scratch, settings)
    if episode.several_wells:  # one pendulum per well: the small periods of a 1:1 regime change differ (W28 C7)
        margins['L5'] = _margin(Margin.NOT_APPLICABLE, margins['L5'].value, 'L5')
    margins['L6'] = _l6(episode, scratch, settings)
    if angle.angle_status == AngleStatus.LIBRATION:
        gap = max(episode.coverage_start_gap_p, episode.coverage_end_gap_p)
        margins['L7'] = _grade(scratch, 'L7', gap, (Op.LE, settings.mmr_l7_gap_full))
    if angle.angle_status == AngleStatus.EPISODIC and physics is not None:
        applies = capture_claimed(decision) and not owner_librates_before(decision, owner, physics, settings, scratch)
        margins['O1'] = _o1(physics, energy, scratch, settings) if applies else _margin(Margin.NOT_APPLICABLE, None, 'O1')
        if applies and onset_ratio_fails(physics, settings):  # the narrowed O1 decided between fail and thin (B4)
            decision.rules_fired = rules_module.fire(decision.rules_fired, rules_module.Rule.O1_FOREIGN_WAVE)
    if own:
        margins['R0'] = _potential_margin(energy)
    return margins


def _l1(episode: Episode, counts: Optional[Mapping[str, Optional[int]]], scratch: Ledger, settings: AnalysisSettings) -> MarginValue:
    values = [v for v in (counts or {}).values() if v is not None]
    least = min(values) if values else episode.n_tp
    if values and scratch.compare('L1', least, Op.GE, settings.mmr_l1_tp_full, '', ''):
        return _margin(Margin.FULL, least, 'L1')
    return _margin(Margin.THIN, least, 'L1')


def _l2(episode: Episode, energy: Optional[EnergyAnalysis], scratch: Ledger, settings: AnalysisSettings) -> MarginValue:
    """Distance to the separatrix: the cycle classes when the zone is known (thin with a cycle in the zone,
    not_measured with a cycle whose single-well J/J_s does not exist and none in the zone, W25), else the amplitude
    (above the thin band it stays thin and the cap l2_without_zone limits the level)."""
    amplitude = math.degrees(episode.half_amp_max_rad)
    if energy is not None and energy.zone is not None and energy.cycle_classes:
        in_zone = has_zone_cycle(energy.cycle_classes)
        if not in_zone and energy.n_not_measured:
            return _margin(Margin.NOT_MEASURED, amplitude, 'L2')
        return _margin(Margin.THIN if in_zone else Margin.FULL, amplitude, 'L2')
    return _grade(scratch, 'L2', amplitude, (Op.LE, settings.mmr_l2_amp_full_deg))


def capture_claimed(decision: StatusDecision) -> bool:
    """The verdict reads the episode as a capture into a libration (the premise of the onset test O1 and of its cap
    onset_failed): the angle's own (`own_dynamics_claimed`), a subresonance's or the owner's libration
    (owner_libration). Not a forced wave of a circulating owner (owner_forced_wave) nor a kinematic libration: neither
    has a capture, and the same a-wave before and inside the episode is what they predict (385266 16N-27, the forced
    wave of 3N-5 at 55-100 kyr; author 2026-10-19 non_resonant/episodic/medium). Nor a non_resonant verdict of
    unknown origin (implausible_strength): it denies the angle's own libration and names no other (2897 35S-4, W21)."""
    if own_dynamics_claimed(decision) or decision.origin == LibrationOrigin.SUBRESONANCE:
        return True
    return decision.origin == LibrationOrigin.INDUCED and decision.origin_detail == OriginDetail.OWNER_LIBRATION


O1_OWNER_COVER = "O1 the owner's libration covers the background window"  # the ledger name (the figure reads it)


def owner_librates_before(
    decision: StatusDecision, owner: Optional[OwnerResult], physics: EpisodePhysics, settings: AnalysisSettings, ledger: Ledger
) -> bool:
    """An induced libration (origin_detail owner_libration) whose owner's libration intervals cover at least
    mmr_o1_owner_cover_min of O1's background window (onset_before_t0_yr .. t1): the a-wave there is the owner's own
    libration, which carries sigma before the episode as inside it, so O1's premise (a capture from a regime without
    that wave) does not hold and O1 does not apply (W22 R-O1-OWNER; 1997 GK3 3M+5S-3, cover 1.00)."""
    if decision.origin_detail != OriginDetail.OWNER_LIBRATION:
        return False
    cover = owner_background_cover(owner, (physics.onset_before_t0_yr, physics.onset_before_t1_yr))
    return bool(ledger.compare_setting(O1_OWNER_COVER, cover, Op.GE, settings, 'mmr_o1_owner_cover_min', '', ''))


def owner_background_cover(owner: Optional[OwnerResult], window: Tuple[float, float]) -> float:
    """The fraction of the window [yr] covered by the owner's libration intervals (nan without an owner, its
    libration intervals or a finite window)."""
    librations = [piece for piece in (owner.intervals if owner is not None else ()) if piece[2] == INTERVAL_LIBRATION]
    if not librations or not all(is_finite(t) for t in window):
        return float('nan')
    return covered_fraction(librations, window)


# the ledger names of O1's three parts (B4, W28; the figure's O1 rows read them)
O1C_COHERENCE = 'O1c background a-wave coherent at P: R2 of one sinusoid at P'
O1C_PHASE = 'O1c background a-wave continues into the episode in phase: phase gap'
O1L_OWN_LAW = "O1l episode's own law explains the background: R2 there / R2 on the episode"


def onset_ratio_fails(physics: Optional[EpisodePhysics], settings: AnalysisSettings) -> bool:
    """O1's ratio (`onset_ratio`) is measured and at least mmr_o1_ratio_fail (the old failure; leaf 5o still reads it
    alone, `_measured_objection`)."""
    ratio = onset_ratio(physics)
    return is_finite(ratio) and ratio >= settings.mmr_o1_ratio_fail


def foreign_wave(physics: EpisodePhysics, ledger: Ledger, settings: AnalysisSettings) -> bool:
    """O1's objection (B4, W28; cards g03-4 A, g08-3 A, author 2026-10-31): the a-wave before the episode is a foreign
    wave the libration only rides. All three must hold:

      O1c  one sinusoid at the episode's period P explains >= mmr_o1_coherence_r2_min of the background's band-passed
           a (a wave: encounter kicks or a circulation sweeping through the band are not coherent at P; 2020 CB 0.02);
      O1c  its phase, extrapolated into the episode, lies within mmr_o1_phase_gap_max_deg of the one fitted on the
           episode's first 2P (a capture starts its own phase; 2024 JZ64: coherent, 146 deg off);
      O1l  NOT explained: a_dot there under the episode's own law (K, phi0; the 1:1 law's K/K_theory) reaches less than
           mmr_o1_own_law_r2_fraction of the same R^2 on the episode. Capture theory: before a capture the body
           circulates near the separatrix under the same force (Murray & Dermott 1999 §8.12), so a wave that follows
           the episode's own law is the cause of the capture, not a rival explanation (16183: 0.52 of 0.63).

    A part that cannot be measured is no evidence of a foreign wave for O1c (nan never coherent) and no explanation
    for O1l (nan never explains)."""
    coherent = ledger.compare_setting(
        O1C_COHERENCE, physics.onset_coherence_r2, Op.GE, settings, 'mmr_o1_coherence_r2_min', '', 'ep_onset_coherence_r2'
    )
    in_phase = ledger.compare_setting(
        O1C_PHASE, physics.onset_phase_gap_deg, Op.LE, settings, 'mmr_o1_phase_gap_max_deg', 'deg', 'ep_onset_phase_gap_deg'
    )
    episode = physics.onset_own_law_r2_episode
    share = physics.onset_own_law_r2 / episode if is_finite(physics.onset_own_law_r2) and is_finite(episode) and episode > 0 else None
    explained = ledger.compare_setting(O1L_OWN_LAW, share, Op.GE, settings, 'mmr_o1_own_law_r2_fraction', '', '')
    return bool(coherent) and bool(in_phase) and not bool(explained)


def _o1(physics: EpisodePhysics, energy: Optional[EnergyAnalysis], scratch: Ledger, settings: AnalysisSettings) -> MarginValue:
    """Onset: the a-wave in the band before the episode over the one inside it (a capture makes its own wave).

    The ratio grades it (full below mmr_o1_ratio_thin, thin below mmr_o1_ratio_fail); at or above mmr_o1_ratio_fail it
    fails only for a foreign wave (`foreign_wave`: coherent at P, in phase, not the episode's own law) and is thin
    otherwise: the wave before a capture from circulation or after a kick is the capture's cause (B4, W28). The cap
    onset_failed reads this margin (`collect_caps`); leaf 5o keeps the ratio alone (`_measured_objection`).

    not_applicable under episodes_in_separatrix_zone: there the angle crosses the separatrix back and forth inside the
    zone, and the a-wave before the episode is the same force's near-separatrix circulation, not an external wave
    (the premise of the test, a capture from a regime without that wave, does not hold; 52356)."""
    if episodes_in_separatrix_zone(energy):
        return _margin(Margin.NOT_APPLICABLE, None, 'O1')
    ratio = onset_ratio(physics)
    if not is_finite(ratio):
        return _margin(Margin.NOT_MEASURED, None, 'O1')
    margin = _grade(scratch, 'O1', ratio, (Op.LT, settings.mmr_o1_ratio_thin), (Op.LT, settings.mmr_o1_ratio_fail))
    if margin.margin == Margin.FAIL and not foreign_wave(physics, scratch, settings):
        return _margin(Margin.THIN, ratio, 'O1')
    return margin


def _l4(response: Optional[Response], decision: StatusDecision) -> MarginValue:
    if response is None:
        return _margin(Margin.NOT_MEASURED, None, 'L4')
    halves = (response.force_r2_h1, response.force_r2_h2)
    value = min(halves) if all(is_finite(x) for x in halves) else None
    harmonic = taken_harmonic_law(response)
    if harmonic is not None:  # the N-harmonic law took L4 (CS12): its R^2 is the margin's value
        value = harmonic.r2
    direct = taken_direct_law(response)
    if direct is not None:  # the direct averaged-force law took L4 (B8): its R^2 is the margin's value
        value = direct.r2
    if response.a_response == CheckValue.FAIL and decision.rule.startswith('5c'):
        return _margin(Margin.THIN, value, 'L4')  # 5c: the potential with a carries the dynamics
    return _check_margin(response.a_response, value, 'L4', thin=response.a_response_thin)


def _l6(episode: Episode, scratch: Ledger, settings: AnalysisSettings) -> MarginValue:
    """Centre shift over the amplitude; not_measured (a thin margin) when the episode has fewer than
    libration_centre_min_estimates independent cycles: three TPs are one cycle and one centre, nothing to compare."""
    if episode.centre_stationarity == CheckValue.NOT_MEASURED:
        return _margin(Margin.NOT_MEASURED, episode.centre_shift_over_amp, 'L6')
    return _grade(scratch, 'L6', episode.centre_shift_over_amp, (Op.LE, settings.mmr_l6_shift_full))


def _l5(physics: Optional[EpisodePhysics], energy, context: Context, scratch: Ledger, settings: AnalysisSettings) -> MarginValue:
    """One pendulum: T0 max/min over the amplitude regimes against what the modulation of the force allows."""
    if physics is None or not is_finite(physics.small_period_ratio):
        return _margin(Margin.NOT_MEASURED, None, 'L5')
    modulation = 1.0
    if energy is not None and is_finite(energy.omega_max_rel) and is_finite(energy.omega_min_rel) and energy.omega_min_rel > 0:
        modulation = energy.omega_max_rel / energy.omega_min_rel
    threshold = modulation * settings.mmr_l5_ratio_full
    ratio = physics.small_period_ratio
    if scratch.compare('L5', ratio, Op.LE, threshold, '', ''):
        return _margin(Margin.FULL, ratio, 'L5')
    caveat = scratch.compare('L5 e caveat', context.e_med, Op.GT, settings.mmr_l5_e_caveat, '', '')
    if caveat or scratch.compare('L5', ratio, Op.LE, threshold * settings.mmr_l5_ratio_fail, '', ''):
        return _margin(Margin.THIN, ratio, 'L5')
    return _margin(Margin.FAIL, ratio, 'L5')


def _zone_margin(energy: Optional[EnergyAnalysis]) -> MarginValue:
    if energy is None or energy.zone is None:
        return _margin(Margin.NOT_MEASURED, None, 'Z')
    thin_flags = {str(Flag.ZONE_EQ_ONLY), str(Flag.ZONE_NUMERIC_MISMATCH), str(Flag.ZONE_EDGE_CYCLE)}
    return _margin(Margin.THIN if thin_flags & set(energy.flags) else Margin.FULL, energy.zone.h_hi, 'Z')


def far_circulation(
    angle: Optional[AngleDecision], decision: Optional[StatusDecision], sampling: Sampling, scratch: Ledger, settings: AnalysisSettings
) -> bool:
    """A non_resonant circulation (an alias included) whose slow a stays >= mmr_far_from_res_factor a-scales from a*
    (`sampling.far_ratio`): sigma_dot = c (a - a*) keeps its sign, so the angle has no turning point for the output
    step to resolve. Never a libration (W20 CS10, H report: 67 far circulations held at high by the 10-deg step rule)."""
    if angle is None or decision is None or decision.status != Status.NON_RESONANT:
        return False
    if angle.angle_status != AngleStatus.CIRCULATION:
        return False
    return bool(scratch.compare('far circulation', sampling.far_ratio, Op.GE, settings.mmr_far_from_res_factor, '', ''))


def _l8(
    sampling: Sampling,
    prep: Optional[Preprocessed],
    scratch: Ledger,
    settings: AnalysisSettings,
    angle: Optional[AngleDecision] = None,
    decision: Optional[StatusDecision] = None,
) -> MarginValue:
    """Step and filter: rotation per step, aliased lines in the band, the gain at the cycle period. The rotation part
    is not graded for a far circulation (`far_circulation`): the 10-deg rule is there to resolve a libration's turning
    points, and on a circulation it would lower the level the faster the angle turns, i.e. the farther the body is from
    the resonance. With no part graded L8 is not_applicable (the x2 guard's 'nothing compared' still thins it)."""
    rotation = math.degrees(sampling.step_rotation_max_rad)
    parts = []
    if not far_circulation(angle, decision, sampling, scratch, settings):
        parts.append(_grade(scratch, 'L8', rotation, (Op.LT, settings.mmr_l8_step_full_deg)))  # coarser steps: C5 decides
    if sampling.aliased_lines_yr:
        parts.append(_margin(Margin.THIN, rotation, 'L8'))
    gain = prep.gain_at_period_cycle if prep is not None else float('nan')
    if is_finite(gain):
        parts.append(_grade(scratch, 'L8', gain, (Op.GE, settings.mmr_l8_gain_full), (Op.GE, settings.mmr_l8_gain_fail)))
    if not parts:
        return MarginValue(Margin.NOT_APPLICABLE, rotation, MarginClass.SUPPORT)
    worst = _worst(*parts)
    return MarginValue(worst.margin, gain if is_finite(gain) else rotation, MarginClass.SUPPORT)


def _s_margin(subresonances: SubresonanceResult, scratch: Ledger, settings: AnalysisSettings) -> MarginValue:
    """S, the subresonance scan. An accepted member: full when the theory confirms it (test D), thin when only the
    order broke a tie or the theory is not measured. None accepted: the closest member's centre range against that of
    sigma0 in the same scan window (`centre0_range_rad`, the range the acceptance compares with), full above
    mmr_s_close_ratio (a member nearly as still as sigma0 is a near miss: thin); full without a range to compare."""
    if subresonances.check not in MEASURED:
        return _margin(Margin.NOT_MEASURED, None, 'S')
    if subresonances.accepted is not None:
        if subresonances.tie_by_order or not subresonances.theory_measured:
            return _margin(Margin.THIN, None, 'S')
        return _margin(Margin.FULL, subresonances.accepted.phi0_coherence, 'S')
    ranges = [c.centre_range_rad for c in subresonances.candidates if is_finite(c.centre_range_rad)]
    own = subresonances.centre0_range_rad
    if not ranges or not is_finite(own) or own <= 0:
        return _margin(Margin.FULL, None, 'S')
    return _grade(scratch, 'S', min(ranges) / own, (Op.GT, settings.mmr_s_close_ratio))


def _w_margin(owner: OwnerResult, scratch: Ledger, settings: AnalysisSettings) -> MarginValue:
    """W, the owner search: not_applicable for a 1:1 angle (no owner exists: every candidate is k sigma plus a planetary
    combination, `owner.FLAG_COORBITAL`); not_measured when the search could not run (a retrograde angle: not
    implemented) or the best candidate is phase_unresolved; an owner accepted by 'potential' full when its R^2 >=
    mmr_w_r2_full and its scale range lies within [1/mmr_owner_potential_scale_full, mmr_owner_potential_scale_full];
    another accepted owner full when it fits a strongly with a reachable force; none accepted: full while the best
    candidate's R^2 stays below mmr_w_r2_none; an accepted owner refused by F7 (cover) or P2 (Psi/u turns) is thin."""
    if owner.check == CheckValue.NOT_APPLICABLE and FLAG_COORBITAL in owner.flags:
        return _margin(Margin.NOT_APPLICABLE, None, 'W')
    if owner.check not in MEASURED:
        return _margin(Margin.NOT_MEASURED, None, 'W')
    if owner.accepted is not None and scratch.compare('W cover', owner.cover, Op.LT, settings.mmr_owner_cover_min, '', ''):
        # F7: an owner acts elsewhere in the record, not in sigma's episode: a near miss of the foreign explanation
        return _margin(Margin.THIN, owner.cover, 'W')
    turns = owner.psi_turns_per_cycle
    if owner.accepted is not None and scratch.compare('W P2', turns, Op.GE, settings.mmr_owner_psi_turns_per_cycle_max, '', ''):
        # P2: the owner fits a in sigma's episode but cannot make its cycles: the same near miss
        return _margin(Margin.THIN, turns, 'W')
    if owner.accepted is not None:
        accepted = owner.accepted
        potential = accepted.potential
        if accepted.route == GAIN_POTENTIAL and potential is not None and potential.measured:
            strong = scratch.compare('W', potential.r2_median, Op.GE, settings.mmr_w_r2_full, '', '')
            full = settings.mmr_owner_potential_scale_full
            inside = scratch.compare('W scale', max(potential.scale_hi, 1.0 / potential.scale_lo), Op.LE, full, '', '')
            return _margin(Margin.FULL if strong and inside else Margin.THIN, potential.r2_median, 'W')
        strong = scratch.compare('W', accepted.r2_median, Op.GE, settings.mmr_w_r2_full, '', '')
        reachable = scratch.compare('W strength', accepted.strength_ratio, Op.LE, 1, '', '')
        return _margin(Margin.FULL if strong and reachable else Margin.THIN, accepted.r2_median, 'W')
    ranked = sorted(owner.candidates, key=lambda c: c.rank)
    if ranked and ranked[0].reason == REASON_PHASE_UNRESOLVED:  # the phases could not separate the best candidate
        return _margin(Margin.NOT_MEASURED, None, 'W')
    best = max((c.r2_median for c in owner.candidates if is_finite(c.r2_median)), default=None)
    if best is None:
        return _margin(Margin.FULL, None, 'W')
    return _grade(scratch, 'W', best, (Op.LT, settings.mmr_w_r2_none))


def _circulation_margins(
    angle: AngleDecision, context: Context, sampling: Sampling, geometry: Geometry, response, scratch: Ledger, settings: AnalysisSettings
) -> Dict[str, MarginValue]:
    """C1–C5 (C1 and C4 not for an alias: there the model set the turns). A failed C4 counts only when K comes from a
    force law accepted at least in part (`force_law_rejected`); a passed one stays: K is then a ceiling of the force,
    and C4 a lower bound (W22 R-C4)."""
    margins = {}
    alias = angle.subtype == AngleSubtype.ALIAS
    if not alias:
        margins['C1'] = _grade(scratch, 'C1', geometry.centre.max_pass_rev, (Op.GE, settings.mmr_c1_rev_full))
    counts = [len(geometry.turning_points), geometry.n_tp_cut_half, geometry.n_tp_cut_double]
    total = sum(c for c in counts if c is not None)
    margins['C2'] = _margin(Margin.FULL if total == 0 else Margin.THIN, total, 'C2')
    pattern = geometry.pattern_b is not None and geometry.pattern_b.detected
    smooth = (
        scratch.compare('C3 sign', geometry.sigmadot_negative_fraction, Op.EQ, 0, '', '')
        and scratch.compare('C3 modulation', geometry.rate_modulation, Op.LT, settings.mmr_c3_modulation_max, '', '')
        and not pattern
    )
    margins['C3'] = _margin(Margin.FULL if smooth else Margin.THIN, geometry.rate_modulation, 'C3')
    if not alias and response is not None:
        ratio = response.circulation_strength_ratio
        margins['C4'] = _grade(scratch, 'C4', ratio, (Op.GE, settings.mmr_c4_ratio_full), (Op.GT, 1))
        if margins['C4'].margin == Margin.FAIL and force_law_rejected(response, settings, scratch):
            margins['C4'] = _margin(Margin.NOT_MEASURED, ratio, 'C4')
    margins['C5'] = _c5_margin(angle, context, sampling, scratch, settings)
    return margins


def force_law_rejected(response: Response, settings: AnalysisSettings, scratch: Ledger) -> bool:
    """The force law explains nothing anywhere: the largest R^2 of the halves, the multiplet law and the one fit over the
    whole window (`rsp_force_R2_whole`) is below mmr_force_r2_fail, or none was fitted (37908 23J-8: 0.005 whole,
    0.016 multiplet). K is then no measured force and a failed C4 measures nothing."""
    values = [value for _, value, _ in force_laws(response)] + [response.force_r2_whole]
    best = max((value for value in values if is_finite(value)), default=None)
    return best is None or bool(scratch.compare('C4 force law R2 (best)', best, Op.LT, settings.mmr_force_r2_fail, '', ''))


def _c5_margin(angle: AngleDecision, context: Context, sampling: Sampling, scratch: Ledger, settings: AnalysisSettings) -> MarginValue:
    """Alias: far from the resonance (`sampling.far_ratio`: |<a> - a*| over the a range or a(lambda_dot), from the
    data) and the branch clearance (`branch_clearance_deg`: the distance per step from nu dt to the nearest half-turn)
    above mmr_c5_branch_resid_factor x the largest per-step unwrap residual (the model error of a step: with twice
    that clearance no step lands on the wrong branch; W20 CS10, Codex); otherwise the expected rate (from the
    measured mean-longitude rate when the body's lambda is saved, else nu(<a>)) against the measured one."""
    if angle.subtype == AngleSubtype.ALIAS:
        far = scratch.compare('C5 far', sampling.far_ratio, Op.GE, settings.mmr_far_from_res_factor, '', '')
        bound = settings.mmr_c5_branch_resid_factor * math.degrees(sampling.unwrap_resid_max_rad)
        branch = scratch.compare('C5 branch', branch_clearance_deg(sampling), Op.GT, bound, 'deg', '')
        return _margin(Margin.FULL if far and branch else Margin.THIN, sampling.far_ratio, 'C5')
    expected = context.rate_from_mean_longitude_rad_yr
    if not is_finite(expected):
        expected = context.rate_expected_rad_yr
    measured = context.rate_measured_rad_yr
    if not (is_finite(expected) and is_finite(measured)) or expected == 0:
        return _margin(Margin.NOT_MEASURED, None, 'C5')
    ratio = measured / expected
    close = scratch.compare('C5 rate', abs(ratio - 1), Op.LE, settings.mmr_c5_rate_tol, '', '')
    same_sign = scratch.compare('C5 sign', ratio, Op.GT, 0, '', '')
    return _margin(Margin.FULL if close and same_sign else Margin.THIN, ratio, 'C5')


def branch_clearance_deg(sampling: Sampling) -> float:
    """The alias branch margin in degrees per step (`Sampling.branch_margin_turns` x 360)."""
    return 360.0 * sampling.branch_margin_turns


def _b5_margin(response: Optional[Response], scratch: Ledger, settings: AnalysisSettings) -> MarginValue:
    if response is None or response.slow_libration in (CheckValue.NOT_MEASURED, CheckValue.NOT_APPLICABLE):
        return _margin(Margin.NOT_MEASURED, None, 'B5')
    if response.slow_libration == CheckValue.FAIL:
        return _margin(Margin.FAIL, response.b5_ratio, 'B5')
    if response.b5_route == 'data' and scratch.compare('B5', response.b5_ratio, Op.GE, settings.mmr_b5_ratio_full, '', ''):
        return _margin(Margin.FULL, response.b5_ratio, 'B5')
    return _margin(Margin.THIN, response.b5_ratio, 'B5')


def strength_margin(strength: Optional[Strength], owner: Optional[OwnerResult]) -> MarginValue:
    """F (SPEC §7.1): the strength check and the owner search when the force law is not measured."""
    if strength is None or strength.check not in MEASURED:
        return _margin(Margin.NOT_MEASURED, None, 'F')
    if strength.check == CheckValue.FAIL:
        return _margin(Margin.FAIL, strength.ratio, 'F')
    return _margin(Margin.FULL if owner is not None else Margin.THIN, strength.ratio, 'F')


# --------------------------------------------------------------------------------------------- caps and checks


def collect_caps(
    view: TreeView,
    decision: StatusDecision,
    geometry: Optional[Geometry],
    margins: Dict[str, MarginValue],
    settings: Optional[AnalysisSettings] = None,
    sampling: Optional[Sampling] = None,
    owner_searched: bool = True,
    subresonances: Optional[SubresonanceResult] = None,
    one_to_one: bool = False,
    response: Optional[Response] = None,
) -> List[str]:
    """`confidence.Cap` values that apply (three_tp_episode, one_return_cycle, exit_suspected, force_law_disagrees,
    onset_failed, episode_before_jump, action_step_provisional, ...). exit_suspected and entry_suspected only where the verdict rests on that edge
    (`_break_shown`); onset_failed only on a measured O1 failure (not_measured is no failure) of a verdict that claims
    a capture (`capture_claimed`, as the margin O1); transient_unexplained
    not for a 1:1 angle (`one_to_one`, `_episode_caps`); one_cycle_energy_not_conserved on a 1:1 angle from the
    response the tree read (`response`: rule R's, the record's where rule R has none; `one_cycle_energy_caps`). `view`:
    the angle and the energy as the tree read them (`tree_view`)."""
    angle, energy = view.angle, view.energy
    caps = [str(Cap.EPISODE_BEFORE_JUMP)] if other_regime(angle) and decision.status == Status.TRANSIENT else []
    caps += list(decision.caps) + _episode_caps(angle, geometry, energy, settings, one_to_one) + action_step_caps(geometry)
    caps += one_cycle_energy_caps(caps, response, settings, one_to_one)
    if angle.angle_status != AngleStatus.CIRCULATION:  # a circulation's scan only names its origin
        caps += subresonance_caps(decision, subresonances)
    if decision.status == Status.TRANSIENT and episodes_in_separatrix_zone(energy):
        caps.append(str(Cap.EPISODES_IN_SEPARATRIX_ZONE))
    caps += near_separatrix_caps(angle, decision, energy, margins)
    if margins.get('O1') is not None and margins['O1'].margin == Margin.FAIL and capture_claimed(decision):
        caps.append(str(Cap.ONSET_FAILED))
    caps += theory_only_caps(decision, margins, response)
    if margins.get('Z') is not None and margins['Z'].margin == Margin.NOT_MEASURED:
        caps.append(str(Cap.ZONE_NOT_MEASURED))
    no_zone = energy is None or energy.zone is None
    l2 = margins.get('L2')
    if l2 is not None and l2.value is not None and no_zone and settings is not None:
        if Ledger.from_settings(settings).compare('L2 beyond thin', l2.value, Op.GT, settings.mmr_l2_amp_thin_deg, 'deg', ''):
            caps.append(str(Cap.L2_WITHOUT_ZONE))
    if margins.get('B5') is not None and margins['B5'].margin == Margin.FAIL:
        caps.append(str(Cap.SLOW_LIBRATION_NOT_EXCLUDED))
    l4 = margins.get('L4')
    if l4 is not None and l4.margin == Margin.NOT_MEASURED and own_dynamics_claimed(decision) and not owner_searched:
        caps.append(str(Cap.OWNER_SEARCH_SKIPPED))
    if sampling is not None and sampling.jitter_low_e:
        caps.append(str(Cap.JITTER_LOW_E))
    return list(dict.fromkeys(caps))


def theory_only_caps(decision: StatusDecision, margins: Mapping[str, MarginValue], response: Optional[Response]) -> List[str]:
    """force_law_theory_only (<= medium; W28 B8, card g07-2 A): L4 of the response the tree read passes only through
    the direct averaged-force law (`taken_direct_law`) on a verdict that claims the angle's own dynamics: a computed
    first-order average with a fitted scale, not the measured shape of the force law; the harmonic law itself failed
    or was thin (59358 8N-13: transient, medium at most)."""
    l4 = margins.get('L4')
    if l4 is None or l4.margin not in (Margin.FULL, Margin.THIN) or not own_dynamics_claimed(decision):
        return []
    return [str(Cap.FORCE_LAW_THEORY_ONLY)] if taken_direct_law(response) is not None else []


# the physical supports of the angle's own force for the cap no_full_physics_support (q-2 b): the force law (the 1:1
# law of a co-orbital takes L4), the potential with a and the strength (in the physics list only when measured by
# direct averaging: `strength.physics_margins`). The zone (Z) and the owner search (W) exclude rivals, they do not
# show the own force
FULL_SUPPORT_MARGINS = ('L4', 'R0', 'F')


def full_support_caps(
    decision: StatusDecision, margins: Mapping[str, MarginValue], physics: Sequence[str], settings: AnalysisSettings
) -> List[str]:
    """no_full_physics_support (<= medium; author 2026-10-31, q-2 b, MMR_HIGH_NEEDS_FULL_SUPPORT): a resonant,
    transient or near_separatrix verdict that claims the angle's own libration (`own_dynamics_claimed`) is high only
    when at least one of its physical supports (`FULL_SUPPORT_MARGINS` that are in `physics`, the margins as the tree
    read them) passes with its full margin. Not added when every applicable physics margin is not_measured: the cap
    all_physics_not_measured says that already. Kinematic, induced and subresonance verdicts rest on other evidence."""
    claims = decision.status in (Status.RESONANT, Status.TRANSIENT, Status.NEAR_SEPARATRIX)
    if not (settings.mmr_high_needs_full_support and claims and own_dynamics_claimed(decision)):
        return []
    applicable = [margins[m].margin for m in physics if m in margins and margins[m].margin != Margin.NOT_APPLICABLE]
    if not applicable or all(margin == Margin.NOT_MEASURED for margin in applicable):
        return []
    supports = [m for m in FULL_SUPPORT_MARGINS if m in physics and m in margins]
    if any(margins[m].margin == Margin.FULL for m in supports):
        return []
    return [str(Cap.NO_FULL_PHYSICS_SUPPORT)]


def one_cycle_energy_caps(caps: Sequence[str], response: Optional[Response], settings: AnalysisSettings, one_to_one: bool) -> List[str]:
    """one_cycle_energy_not_conserved (<= medium) for a 1:1 verdict on one cycle (cap three_tp_episode) whose
    co-orbital energy integral fails: R² of (1/2) theta_dot^2 on the averaged potential below mmr_force_r2_fail. The R²
    is that of the response the tree read (rule R's witness episode, the column rsp_coorb_energy_R2), not the pooled
    law: one cycle may follow the force law while its energy is not conserved (2024 WJ: witness 0.04, pooled 0.82;
    W22 R-WJ). A not measured R² is no failure."""
    if not one_to_one or str(Cap.THREE_TP_EPISODE) not in caps:
        return []
    law = response.coorbital if response is not None else None
    r2 = law.energy_r2 if law is not None else None
    name = '1:1 one cycle: co-orbital energy integral R2'
    if Ledger.from_settings(settings).compare_setting(name, r2, Op.LT, settings, 'mmr_force_r2_fail', '', 'rsp_coorb_energy_R2'):
        return [str(Cap.ONE_CYCLE_ENERGY_NOT_CONSERVED)]
    return []


def near_separatrix_caps(
    angle: AngleDecision, decision: StatusDecision, energy: Optional[EnergyAnalysis], margins: Mapping[str, MarginValue]
) -> List[str]:
    """near_separatrix_without_zone (<= medium, as l2_without_zone): a near_separatrix of the angle's geometry (G3,
    pattern A or B without an episode) whose zone is not measured (`separatrix_zone_check` not pass). The pattern
    alone does not place the angle at the separatrix: 564160 3N-4 stalls at the top, but its zone is only estimated
    (zone_eq_only) and the frozen model puts it inside the island after 30 kyr (W17 Q15)."""
    near = angle.angle_status == AngleStatus.NEAR_SEPARATRIX and decision.status == Status.NEAR_SEPARATRIX
    if near and separatrix_zone_check(energy, margins) != CheckValue.PASS:
        return [str(Cap.NEAR_SEPARATRIX_WITHOUT_ZONE)]
    return []


def subresonance_caps(decision: StatusDecision, subresonances: Optional[SubresonanceResult]) -> List[str]:
    """SPEC §6.4 step 3 and §6.3, each capping the level at medium:

    - a verdict that rests on an accepted subresonance: the scan ran on the theory trigger T6 alone, test C was decided
      by the order only (subresonance_tie), or test D did not confirm the member (not computed or disagrees);
    - any verdict after a scan that could not decide (check not_measured): no planets of the angle (planets_missing),
      beta = varpi_P - varpi librates (subresonance_undetermined) or a nodal member won (nodal_subresonance_candidate).
      The subresonance mechanism is then not excluded, so a kinematic or dynamic reading is not above medium. Since
      W28 A6b every accepted episode is scanned in its own window: another episode whose window could not decide
      (`SubresonanceResult.episode_undecided`) leaves the mechanism open there, and caps the same way."""
    if subresonances is None:
        return []
    caps = []
    if decision.origin == LibrationOrigin.SUBRESONANCE and subresonances.accepted is not None:
        if tuple(str(t) for t in subresonances.triggers) == ('T6',):
            caps.append(str(Cap.SUBRESONANCE_T6_ONLY))
        if subresonances.tie_by_order:
            caps.append(str(Cap.SUBRESONANCE_TIE))
        if not subresonances.theory_measured:
            caps.append(str(Cap.SUBRESONANCE_THEORY_UNCONFIRMED))
    flags = {flag for episode in subresonances.episode_undecided.values() for flag in episode}
    decisive_open = subresonances.check == CheckValue.NOT_MEASURED
    if decisive_open:
        flags |= set(subresonances.flags)
    if str(Flag.PLANETS_MISSING) in flags:
        caps.append(str(Cap.PLANETS_MISSING))
    if str(Flag.SUBRESONANCE_UNDETERMINED) in flags or (decisive_open and subresonances.undetermined):
        caps.append(str(Cap.SUBRESONANCE_UNDETERMINED))
    if str(Flag.NODAL_SUBRESONANCE_CANDIDATE) in flags:
        caps.append(str(Cap.NODAL_SUBRESONANCE_CANDIDATE))
    return caps


def action_step_caps(geometry: Optional[Geometry]) -> List[str]:
    """action_step_provisional: G2a cut a run at a step of its action with fewer than libration_jump_cycles energy
    cycles on a side, or one disjoint cycle of the larger swings (`geometry.ActionStep.provisional`, W24)."""
    provisional = geometry is not None and str(Flag.ACTION_STEP_PROVISIONAL) in geometry.flags
    return [str(Cap.ACTION_STEP_PROVISIONAL)] if provisional else []


def _episode_caps(
    angle: AngleDecision, geometry: Optional[Geometry], energy, settings: Optional[AnalysisSettings], one_to_one: bool = False
) -> List[str]:
    """The caps of a librating angle's episodes: three_tp_episode, one_return_cycle (a decisive episode closed by its
    phase return, with libration_return_cycle_caps_medium), the edges (`_break_shown`) and transient_unexplained (an
    episodic angle whose crossings the separatrix model cannot explain; not for a 1:1 angle, `one_to_one`, while
    mmr_coorbital_lifts_unexplained: the pendulum model does not apply there, author 2026-10-19 R2)."""
    if geometry is None or angle.angle_status not in LIBRATING:
        return []
    caps = []
    episode = run_of(geometry, angle.episode_id)
    if episode is not None and settings is not None:
        if Ledger.from_settings(settings).compare('three TP episode', episode.n_tp, Op.EQ, settings.libration_min_turning_points, '', ''):
            caps.append(str(Cap.THREE_TP_EPISODE))
        if episode.closed_by_return and settings.libration_return_cycle_caps_medium:
            caps.append(str(Cap.ONE_RETURN_CYCLE))
    edges = []
    if geometry.exit is not None and geometry.exit.suspected:
        edges.append(str(Cap.EXIT_SUSPECTED))
    if geometry.entry_suspected():  # the last regime's entry or, after a cut, the record's (`Geometry.record_entry`)
        edges.append(str(Cap.ENTRY_SUSPECTED))
    if edges and not (angle.angle_status == AngleStatus.EPISODIC and _break_shown(geometry, settings)):
        caps += edges
    lifted = one_to_one and (settings is None or settings.mmr_coorbital_lifts_unexplained)
    if angle.angle_status == AngleStatus.EPISODIC and energy is not None and energy.mechanism == SeparatrixMechanism.UNEXPLAINED:
        if not episodes_in_separatrix_zone(energy) and not lifted:  # crossings inside the zone: not an unexplained change
            caps.append(str(Cap.TRANSIENT_UNEXPLAINED))
    return caps


def break_between_episodes(geometry: Geometry, settings: AnalysisSettings) -> float:
    """The range sigma_f covers between two consecutive librating episodes of the window (`librating_episodes`) [rad]:
    max - min of its turning points from the last one of an episode to the first one of the next (the extrema of
    sigma_f there); the largest over the pairs, 0 with fewer than two episodes. A libration covers less than 2 pi, so
    above 2 pi the angle went over the top between them, in one swing or by a drift with small reversals: the
    libration was broken inside the record (33128 6N-7: 11.4 rad at 63.8-73.9 kyr)."""
    episodes = sorted(librating_episodes(geometry, settings), key=lambda episode: episode.t_start_yr)
    travel = 0.0
    for left, right in zip(episodes, episodes[1:]):
        travel = max(travel, break_range(geometry, left, right))
    return travel


def _break_shown(geometry: Geometry, settings: Optional[AnalysisSettings]) -> bool:
    """An episodic angle whose break the record shows by itself (`break_between_episodes` > 2 pi): exit_suspected and
    entry_suspected doubt only a verdict that rests on that edge (a libration, or an episodic angle whose only break is
    the edge), so they do not cap this one (DECISIONS 2026-10-15; inv-11-12 Q12: 33128)."""
    if settings is None:
        return False
    travel = break_between_episodes(geometry, settings)
    return bool(Ledger.from_settings(settings).compare('revolution between librating episodes', travel, Op.GT, TWO_PI, 'rad', ''))


def mmr_checks(
    view: TreeView,
    sampling: Sampling,
    prep: Optional[Preprocessed],
    geometry: Optional[Geometry],
    kinematics: Optional[Kinematics],
    subresonances: Optional[SubresonanceResult],
    owner: Optional[OwnerResult],
    strength: Optional[Strength],
    base: Optional[Baseline],
    decision: Optional[StatusDecision] = None,
    margins: Optional[Mapping[str, MarginValue]] = None,
    settings: Optional[AnalysisSettings] = None,
) -> Dict[str, CheckValue]:
    """Every `check_mmr_*` of SPEC §11.2 (filter_too_slow makes the shape-dependent ones not_measured).

    check_mmr_filter: fail when G(P_cyc) < mmr_l8_gain_fail, not_measured without cycles, when a decisive number
    changes at the cutoff x2 (flag filter_sensitive, DELTA 13a) or when the x2 guard could compare nothing. `decision` and `margins` give the
    status-dependent not_applicable cases and L5/O1; `view` the angle, response and energy as the tree read them."""
    nm = CheckValue.NOT_MEASURED
    angle, response, energy = view.angle, view.response, view.energy
    checks = {name: nm for name in MMR_CHECKS}
    checks['check_mmr_alias'] = CheckValue.PASS if sampling.unwrap_ok else CheckValue.FAIL
    if prep is None or geometry is None:
        return checks
    checks['check_mmr_filter'] = _filter_check(prep, settings)
    checks['check_mmr_baseline'] = (
        CheckValue.NOT_APPLICABLE if angle.angle_status == AngleStatus.CIRCULATION else (base.check if base is not None else nm)
    )
    if response is not None:
        checks['check_mmr_identity'] = response.identity
        checks['check_mmr_a_response'] = response.a_response
        checks['check_mmr_a_amplitude'] = response.a_amplitude
        checks['check_mmr_slow_libration'] = response.slow_libration
    checks['check_mmr_potential'] = energy.with_slow if energy is not None else nm
    checks['check_mmr_kinematic'] = kinematic_check(angle, kinematics, response, decisive_period(angle, geometry), settings, geometry)
    checks['check_mmr_subresonances'] = subresonances.check if subresonances is not None else nm
    checks['check_mmr_owner'] = owner.check if owner is not None else nm
    if str(Flag.OWNER_OUTSIDE_EPISODE) in decision.flags:  # F7: the owner found acts outside sigma's episode
        checks['check_mmr_owner'] = CheckValue.PASS
    checks['check_mmr_strength'] = strength.check if strength is not None else nm
    checks.update(_shape_checks(angle, geometry, energy, decision, dict(margins or {})))
    return checks


def _filter_check(prep: Preprocessed, settings: Optional[AnalysisSettings]) -> CheckValue:
    gain = prep.gain_at_period_cycle
    if str(Flag.FILTER_SENSITIVE) in prep.flags or not is_finite(gain) or settings is None:
        return CheckValue.NOT_MEASURED
    holds = Ledger.from_settings(settings).compare('filter gain', gain, Op.GE, settings.mmr_l8_gain_fail, '', '')
    return CheckValue.PASS if holds else CheckValue.FAIL


R0_KEPT = ('check_mmr_potential', 'R0')  # what an R0 read beside the guard's episode keeps (`r0_beside_guard`)


def r0_beside_guard(energy: Optional[EnergyAnalysis], sensitivity) -> bool:
    """R0 was read inside another episode than the one the filter guard compared (W34, `zone.episode_r0`): the guard's
    change at 2 P_c is not that episode's, so it withdraws nothing of R0 (79653: R0 from ep0, the guard on ep1)."""
    if energy is None or sensitivity is None or energy.r0_episode_id is None or sensitivity.decisive_episode_id is None:
        return False
    return energy.r0_episode_id != sensitivity.decisive_episode_id


def _unmeasure(checks: Dict[str, CheckValue], margins: Dict[str, MarginValue], quantities, keep=()) -> None:
    """The checks and margins that read `quantities` (`preprocess.SENSITIVE_CHECKS`, `SENSITIVE_MARGINS`) become
    not_measured, keeping their numbers; a not_applicable one stays, and so does a failed margin (emptying it would
    raise the confidence: the guard only ever lowers it; 1221 1V-2E+1, O1 fail)."""
    from resonances.resonance.analysis.mmr.preprocess import SENSITIVE_CHECKS, SENSITIVE_MARGINS

    names = set()
    for quantity in quantities:
        names.update(SENSITIVE_CHECKS.get(quantity, ()))
        for margin_id in SENSITIVE_MARGINS.get(quantity, ()):
            margin = margins.get(margin_id)
            if margin_id not in keep and margin is not None and margin.margin not in (Margin.NOT_APPLICABLE, Margin.FAIL):
                margins[margin_id] = MarginValue(Margin.NOT_MEASURED, margin.value, margin.margin_class)
    for name in names - set(keep):
        if name in checks and checks[name] != CheckValue.NOT_APPLICABLE:
            checks[name] = CheckValue.NOT_MEASURED


def apply_filter_sensitivity(
    checks: Dict[str, CheckValue], margins: Dict[str, MarginValue], sensitivity, flags: Optional[List[str]] = None, keep=()
) -> None:
    """DELTA 13a, in place: when a decisive number changes at the cutoff x2 (`preprocess.filter_sensitivity`),
    check_mmr_filter and the checks and margins that read the changed quantity become not_measured
    (`preprocess.SENSITIVE_CHECKS`, `SENSITIVE_MARGINS`; a not_applicable one stays), and `flags` gets
    filter_sensitive. When nothing could be compared (`sensitivity.sensitive` None: no x2 series or no common
    interval) the filter dependence is not checked: check_mmr_filter is not_measured and L8 at most thin (a far
    circulation's not_applicable L8 as well: only its step part is not graded).
    Here the guard sets how far the verdict can be trusted (confidence); the passed checks the tree itself reads were
    withdrawn before it ran (`energy_for_tree`, W21 Q9). `keep`: checks and margin ids left as they are (`R0_KEPT`
    when `r0_beside_guard`)."""
    if sensitivity is None:
        return
    if sensitivity.sensitive is None:
        if checks.get('check_mmr_filter', CheckValue.NOT_APPLICABLE) != CheckValue.NOT_APPLICABLE:
            checks['check_mmr_filter'] = CheckValue.NOT_MEASURED
        l8 = margins.get('L8')
        if l8 is not None and l8.margin in (Margin.FULL, Margin.NOT_APPLICABLE):  # a far circulation's L8 too (CS10)
            margins['L8'] = MarginValue(Margin.THIN, l8.value, l8.margin_class)
        return
    if not sensitivity.sensitive:
        return
    if flags is not None and str(Flag.FILTER_SENSITIVE) not in flags:
        flags.append(str(Flag.FILTER_SENSITIVE))
    if 'check_mmr_filter' in checks and checks['check_mmr_filter'] != CheckValue.NOT_APPLICABLE:
        checks['check_mmr_filter'] = CheckValue.NOT_MEASURED
    _unmeasure(checks, margins, sensitivity.changed, keep)


# the checks the tree reads that the filter guard can withdraw (`preprocess.SENSITIVE_CHECKS`): check -> (the field of
# the energy, the margin id); R0 with a. The other withdrawn checks and margins (L5, L6, L7, O1, Z, one pendulum, onset,
# exit) are read by the confidence, and O1 by the tree only as an objection (a failure, which stays)
TREE_ENERGY_CHECKS = {'check_mmr_potential': ('with_slow', 'R0')}


def energy_for_tree(
    energy: Optional[EnergyAnalysis], sensitivity, ledger: Ledger
) -> Tuple[Optional[EnergyAnalysis], Dict[str, Withdrawal]]:
    """(energy as the tree reads it, the checks withdrawn from it by margin id, for `decide_status`): W21 Q9,
    principle (b) as both reviews refined it.
    The filter guard runs before the tree; a check the tree reads (`TREE_ENERGY_CHECKS`) that passed at P_c while a
    quantity it rests on changes at 2 P_c (`sensitivity.changed`) is not available to the tree (not_measured, a
    ledger entry names the guard); a failed one stays an objection: the guard only ever lowers the support. The
    energy itself is not changed (its columns and the margins keep the measured result; `apply_filter_sensitivity`
    empties the published checks afterwards). An R0 read inside another episode than the one the guard compared
    (`EnergyAnalysis.r0_episode_id`, W34) is not withdrawn by it."""
    from resonances.resonance.analysis.mmr.preprocess import SENSITIVE_CHECKS

    if energy is None or sensitivity is None or not sensitivity.sensitive:
        return energy, {}
    names = {name for quantity in sensitivity.changed for name in SENSITIVE_CHECKS.get(quantity, ())}
    if r0_beside_guard(energy, sensitivity):
        names -= set(R0_KEPT)
    withdrawn = tuple(
        name for name in TREE_ENERGY_CHECKS if name in names and getattr(energy, TREE_ENERGY_CHECKS[name][0]) == CheckValue.PASS
    )
    if not withdrawn:
        return energy, {}
    quantities = ', '.join(q for q in sensitivity.changed if set(SENSITIVE_CHECKS.get(q, ())) & set(withdrawn))
    for name in withdrawn:
        ledger.compare(
            f'filter guard: {name} passed at P_c, withdrawn before the tree ({quantities} change at 2 P_c)',
            None,
            Op.EQ,
            0,
            '',
            'flt_withdrawn_before_tree',
        )
    read = replace(energy, **{TREE_ENERGY_CHECKS[name][0]: CheckValue.NOT_MEASURED for name in withdrawn})
    return read, {TREE_ENERGY_CHECKS[name][1]: Withdrawal(WITHDRAWN_FILTER_GUARD, name) for name in withdrawn}


def apply_filter_too_slow(checks: Dict[str, CheckValue], margins: Dict[str, MarginValue], filter_too_slow: bool) -> None:
    """DELTA item 1, in place: a libration period below filter_min_libration_cutoffs x P_c leaves the shape of the
    oscillation unmeasured (the filter damps it): the margins and checks of the quantity 'shape' (L2, L5, L6,
    one pendulum) become not_measured."""
    if filter_too_slow:
        _unmeasure(checks, margins, ('shape',))


def _shape_checks(
    angle: AngleDecision, geometry: Geometry, energy, decision: Optional[StatusDecision], margins: Dict[str, MarginValue]
) -> Dict[str, CheckValue]:
    """The checks whose applicability depends on the angle's shape and the status."""
    na = CheckValue.NOT_APPLICABLE
    checks = {}
    if decision is not None and decision.status in (Status.RESONANT, Status.TRANSIENT, Status.NEAR_SEPARATRIX):
        checks['check_mmr_slow_libration'] = na
    if angle.angle_status == AngleStatus.CIRCULATION:
        checks['check_mmr_a_amplitude'] = na
    if angle.angle_status == AngleStatus.INSUFFICIENT_TIME and angle.subtype == AngleSubtype.ARC:
        checks['check_mmr_potential'] = na
    if angle.angle_status not in OSCILLATING:
        checks['check_mmr_owner'] = na
    librating = angle.angle_status in LIBRATING
    checks['check_mmr_one_pendulum'] = _margin_check(margins.get('L5')) if librating else na
    checks['check_mmr_onset'] = _margin_check(margins.get('O1')) if librating else na
    suspected = (geometry.exit is not None and geometry.exit.suspected) or geometry.entry_suspected()
    checks['check_mmr_exit'] = (CheckValue.FAIL if suspected else CheckValue.PASS) if librating else na
    checks['check_mmr_separatrix_zone'] = separatrix_zone_check(energy, margins)
    return checks


def separatrix_zone_check(energy: Optional[EnergyAnalysis], margins: Mapping[str, MarginValue]) -> CheckValue:
    """check_mmr_separatrix_zone: not_applicable without cycles, pass when the zone is measured and its margin Z is
    full (a numeric zone, no edge cycle), not_measured otherwise (an estimated zone or none)."""
    if energy is None or not energy.cycles:
        return CheckValue.NOT_APPLICABLE
    z = margins.get('Z')
    if energy.zone is not None and z is not None and z.margin == Margin.FULL:
        return CheckValue.PASS
    return CheckValue.NOT_MEASURED


def _margin_check(margin: Optional[MarginValue]) -> CheckValue:
    if margin is None or margin.margin == Margin.NOT_MEASURED:
        return CheckValue.NOT_MEASURED
    if margin.margin == Margin.NOT_APPLICABLE:
        return CheckValue.NOT_APPLICABLE
    return CheckValue.FAIL if margin.margin == Margin.FAIL else CheckValue.PASS


# --------------------------------------------------------------------------------------------- marks


# what keeps exit_and_capture off: the record's own numbers are not trusted there (filter, step, phase, orbit) or the
# episodes are judged without their own energy (episode_before_jump)
EXIT_AND_CAPTURE_BLOCKERS = frozenset(
    {
        str(Flag.FILTER_SENSITIVE),
        str(Flag.FILTER_TOO_SLOW),
        str(Flag.SHORT_PERIOD_ALIASED),
        str(AngleSubtype.UNDEFINED_PHASE),
        str(AngleSubtype.UNPHYSICAL),
        str(Cap.EPISODE_BEFORE_JUMP),
    }
)


def exit_and_capture(
    spec,
    angle_status,
    energy: Optional[EnergyAnalysis],
    checks: Mapping[str, CheckValue],
    flags: Sequence[str],
    caps: Sequence[str],
    reason,
    subtype,
    ledger: Ledger,
) -> bool:
    """The mark exit_and_capture (W17 Q12): the angle left a libration at least once and came back at least once
    (`EnergyAnalysis.n_exits`, `n_captures`, counted before the times are merged: a slip between two librations is
    one of each). It names no mechanism and changes no status. Only for an episodic angle (as reported) that is not
    co-orbital, whose crossings are not a kick of the slow variable (sep_mechanism), whose own force is measured
    (check_mmr_a_response or check_mmr_potential pass) and that carries none of `EXIT_AND_CAPTURE_BLOCKERS` in its flags,
    caps, reason or subtype. The comparisons go to the ledger only when the angle is episodic with an energy."""
    from resonances.resonance.analysis.mmr.preprocess import coorbital_planet

    if str(angle_status) != str(AngleStatus.EPISODIC) or energy is None or coorbital_planet(spec) is not None:
        return False
    exits = ledger.compare('mark exit_and_capture: exits from a libration', energy.n_exits, Op.GE, 1, '', 'sep_n_exits')
    captures = ledger.compare('mark exit_and_capture: captures into a libration', energy.n_captures, Op.GE, 1, '', 'sep_n_captures')
    if not (exits and captures) or energy.mechanism == SeparatrixMechanism.KICK:
        return False
    own_force = CheckValue.PASS in (checks.get('check_mmr_a_response'), checks.get('check_mmr_potential'))
    blocked = EXIT_AND_CAPTURE_BLOCKERS & (set(flags) | set(caps) | {str(reason), str(subtype)})
    return bool(own_force and not blocked)


def transient_unexplained_after_exit_and_capture(caps: List[str], marked: bool, settings: AnalysisSettings) -> List[str]:
    """The caps without transient_unexplained when the mark exit_and_capture holds and
    `mmr_exit_and_capture_lifts_unexplained` is on (author:2026-10-18, W17 Q12). A convention, not an explanation of
    the transitions: an angle that left its own libration and came back is the transient the record shows, whatever
    moved it. On the benchmark it raises only 33128 6N-7 (medium -> high); 34706 7J-2, 16183 and 17445 5J-2S-2
    lose the cap but stay low (onset_failed)."""
    if not (marked and settings.mmr_exit_and_capture_lifts_unexplained):
        return caps
    return [cap for cap in caps if cap != str(Cap.TRANSIENT_UNEXPLAINED)]
