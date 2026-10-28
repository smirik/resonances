"""Subresonances of the multiplet: which member owns the oscillation (SPEC §6 with 'subresonance' for 'sister').

All members sigma_s = sigma0 + Delta_s share the fast oscillation and differ by the slow angle Delta_s
(an integer combination of varpi, varpi_P, Omega, Omega_P), i.e. by the centre. If member N librates
about c_N, the catalogue angle's centre moves as c0(t) = c_N - Delta_N(t) and drifts at -Delta_dot_N =
m (g - g_P) for Delta_N = m (varpi_P - varpi). Planar members (varpi, varpi_P only) are in the same ecliptic
frame as sigma0 and become rows of their own (summary.csv and a combined figure); nodal members (enumerated when
the inclination reaches mmr_subresonance_nodal_inc_min_deg) are scanned in one frame, the invariable plane: every
longitude, perihelion and node of the body and the angle's planets is taken from it (lambda' = varpi' + M, M does not
depend on the plane), i.e. Delta_s from the invariable plane plus sigma0's own change of frame (`plane_shift`); without
a planet's saved elements that change is unknown and the old mixed frame stays (scan flag frame_mixed). Nodal members
are only listed, never accepted.

Degree and order. Every member shares the order q = |sum k| of the commensurability; a member's degree is the sum
of |slow coefficients|, the power of e, e_P, sin i/2, sin i_P/2 in its leading d'Alembert factor (degree >= |q|,
same parity). The decision and the ranked list `sub_top` (display only) take the members up to the catalogue's
largest order, MATRIX_2BODY_ORDER_MAX / MATRIX_3BODY_ORDER_MAX, as their degree (`degree_max`; author, 2026-10-16: no
constant of its own), e.g. 3J-1S-1+2-2-1 (degree 5) of 3J-1S-1+0+0-1 (q = 1); the members of degree |q| always; with
the nodal members also sigma +- 2 m omega (omega = varpi - Omega), m = 1..mmr_subresonance_omega_harmonics, whatever
their degree (W20: the centre of 2003 UG22 10E-17+0+7 follows -2 omega, and sigma + 2 omega has degree 11 > 10).
Test E' keeps the many weak high-degree members from being measured; the list (`member_table`) ranks every member by
E, the share of the catalogue centre's motion its slow angle explains, from the covariances of the slow angles,
without a track each.

The check (SPEC §6.4; SISTERS §§3-4; codex B-review §4):
  window   the longest run of >= libration_min_turning_points TPs of sigma0, trimmed to its regular cycles (the slow
           near-separatrix cycles at its edges have no stationary centre), else the analysis window
  step 1   per member the centre track c_s = c0 + smooth_{W_c}(Delta_s): range <= ratio x range(c0), |rate| <=
           max(2 sigma_rate, floor), largest monotone pass < pass_max (`screen`)
  step 2   E' the member's d'Alembert factor is >= mmr_subresonance_strength_rel_min x that of the strongest term
           of the commensurability (always, theory or not: the e_Earth^7 member 10E-17+7+0 of 2003 UG22 has 1e-10);
           L the member librates itself in the window: its own angle sigma0_f + Delta_s,f has an episode its own row
           would count as a libration, at sigma0's prominence (`member_libration`; 63047 3J+7S-2-8+0+0 passed A with
           no TP);
           A force law a_dot = R sin(sigma_s - phi0) + C in windows of mmr_force_window_cycles cycles: phi0 coherent
           (65366: +1+4 0.88-0.94 against 0.08-0.32 for sigma0; 31935: +2+5 0.84-0.91 against 0.24-0.51); not
           measured when sigma0's largest swing in the window is below mmr_a_response_min_amp_deg (there a sine and a
           line differ by < 10 %: every member's phase is coherent with the identity alone; 2013 AS27 4E+9S-5, 3-4 deg);
           more than one member left with A not measured and nothing else to tell them apart -> undetermined
           B centre near 0 or pi; C unique (tie: lower degree, then larger coefficient); E the strongest predicted T0
           is not far slower than the observed period; D composite multiplet theory (`multiplet.py`: c0 = pi - arg
           P(beta), periods against T0(t) F(A)) never rejects, it only raises the confidence (`decide`, `evaluate`)
  outcome  accepted -> check fail (the tree gives non_resonant/subresonance_librates); none -> pass; without the
           planets of the angle -> not_measured (planets_missing); beta = varpi_P - varpi librating ->
           not_measured (subresonance_undetermined), as are several members that only A could tell apart and A is not
           measured; a nodal winner -> not_measured (nodal_subresonance_candidate)

Every number goes to `SubresonanceScan.to_columns()` (columns `sub_*`) and the candidates. `member_series` builds the
accepted member's own series (sigma0 + Delta_s from the saved elements, no integration) for its derived row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from resonances.data import const
from resonances.data.invariable_plane import to_invariable_plane
from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.geometry import (
    Geometry,
    RejectReason,
    find_episodes,
    find_turning_points,
    half_amplitude_values,
    largest_monotone_pass,
    librating_episodes_with,
    monotone_passes,
    wrap_pi,
)
from resonances.resonance.analysis.ledger import Ledger, LedgerEntry, Op
from resonances.resonance.analysis.mmr import decide as decide_module, multiplet
from resonances.resonance.analysis.mmr.context import planet_frequency
from resonances.resonance.analysis.mmr.preprocess import law_planet
from resonances.resonance.analysis.mmr.response import correlation, force_windows, response_series, samples_between
from resonances.resonance.analysis.mmr.models import (
    KYR,
    PLANET_NAMES_ORDER,
    RAD_TO_DEG,
    Context,
    MMRSeries,
    Preprocessed,
    ResonanceKind,
    ResonanceSpec,
    Response,
    SubresonanceCandidate,
    SubresonanceResult,
    SubresonanceTrigger,
    column_number,
    planet_letter,
    planet_terms,
)
from resonances.resonance.analysis.pendulum import amplitude_factor
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue, Flag, Reason, join_list

NAN = float('nan')
TWO_PI = 2 * np.pi
SQRT_12 = math.sqrt(12.0)  # slope error of a least-squares line: std(resid) sqrt(12) / T for a uniform grid
J2000_JD = 2451545.0
GAUSSIAN_YEAR_DAYS = const.GAUSSIAN_YEAR_DAYS  # the year of the analysis time axis (const.py)
MIN_PERIOD_CYCLES = 4  # a correlation of observed and predicted periods needs at least this many cycles
THEORY_GRID_PER_WINDOW = 10  # theory track samples per centre window W_c (c0 is smooth on W_c)
TRACK_POINTS = 2000  # samples of the inclination series behind the median sin(i/2) (`_median_sin_half_inc`)
TABLE_ROWS = 3  # members of the ranked list with columns of their own (sub_rank1..3_*; the figure's table)
SCREEN_POINTS = 4000  # samples of a member's track in step 1 (smooth on W_c; the pass search dominates the cost)
THEORY_MAX_POINTS = 1000  # samples of the theory track
THEORY_E_NODES = 3  # (e, e_P) nodes of the coefficient table; C_j / (e^|q-j| e_P^|j|) is smooth between them
THEORY_EP_NODES = 2
THEORY_J_MARGIN = 2  # members beyond 0..q kept in P(beta) (they enter at higher powers of e)
# rejections of a TP-rich run that make T1: a centre that moves, a saw that never returns
TRIGGER_REJECTIONS = (str(RejectReason.CENTRE_MOVES), str(RejectReason.NO_CLOSED_CYCLE))
PENDING_KINEMATIC = (str(Reason.KINEMATIC_DRIFT), str(Reason.KINEMATIC_LIBRATION), str(Reason.DRIFT_MECHANISM_UNKNOWN))

# candidate reasons (candidates.csv `reason`): why a member was not accepted
REASON_RANGE, REASON_RATE, REASON_PASS = 'range', 'rate', 'pass'
SCREEN_REASONS = (REASON_RANGE, REASON_RATE, REASON_PASS)
REASON_A_NOT_MEASURED, REASON_A_COHERENCE, REASON_A_GAIN = 'A_not_measured', 'A_coherence', 'A_gain'
REASON_A_SMALL_SWING = 'A_small_swing'  # test A not measured: sigma0 swings below mmr_a_response_min_amp_deg
REASON_B, REASON_C, REASON_D, REASON_E = 'B_centre', 'C_tie', 'D_theory', 'E_t0'
REASON_E_STRENGTH = 'E_strength'  # test E': too weak against the strongest term of the commensurability
# test L: the member's own angle has no accepted episode in the scan window, or that angle could not be built
REASON_L, REASON_L_NOT_MEASURED = 'L_not_librating', 'L_not_measured'
# `member_reject` of an accepted run that is no libration of the member (`decide.librating_episodes`)
REJECT_PERIOD, REJECT_CENTRE_TURN = 'below_period_floor', 'centre_turn'
REASON_PLANETS, REASON_UNDETERMINED, REASON_NODAL = 'planets_missing', 'undetermined', 'nodal'
FLAG_TIE = str(Flag.SUBRESONANCE_TIE)
FLAG_THEORY_DISAGREES = str(Flag.SUBRESONANCE_THEORY_DISAGREES)  # test D measured and failed (confidence only)
FLAG_THEORY_CROSSING = str(Flag.SUBRESONANCE_THEORY_CROSSING)  # no theory: the orbits can intersect (2003 UG22 with Earth)
FLAG_THEORY_ROUND_OFF = str(Flag.SUBRESONANCE_THEORY_ROUND_OFF)  # no theory: coefficients below round-off (very high order)
FLAG_THEORY_RANGE = str(Flag.SUBRESONANCE_THEORY_RANGE)  # no theory: e or e_P outside (0, 1) or an empty window
# a scan flag (sub_flags only, no verdict flag): nodal members in the mixed frame, a planet of the angle not saved
FLAG_FRAME_MIXED = 'frame_mixed'
FRAME_SHIFT = 'frame_shift'  # the pseudo slow angle sigma0(invariable) - sigma0(ecliptic) of the invariable frame


# --------------------------------------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class SubresonanceMember(SubresonanceCandidate):
    """A `SubresonanceCandidate` with the numbers of the screen and the tests that the base row does not hold.

    `composite_rms_rad`, `period_corr`, `period_ratio` of the base are record-level theory numbers and are set on
    the accepted (else the best) member only; `single_rms_rad` is the RMS of c0 about this member's hypothesis
    c0 = const - Delta_s."""

    degree: int = 0  # sum of |slow coefficients| (power of e, e_P, sin i/2, sin i_P/2; not the order |q|)
    strength_rel: float = NAN  # prod x^|c| / that of the strongest term of the commensurability (test E')
    kept: bool = False  # passed step 1
    rate_tolerance_rad_yr: float = NAN  # max(2 sigma_rate, floor)
    centre_distance_rad: float = NAN  # mean centre from the nearer of 0 and pi
    force_r2_median: float = NAN
    n_force_windows: int = 0
    single_rms_rad: float = NAN
    coefficient_abs: float = NAN  # |C_j| at the mean e, e_P (two-body planar members)
    offset_rate_rad_yr: float = NAN  # Delta_dot_s over the window (c0 drifts at -this if s librates)
    # test L, the member's own angle in the scan window (`member_libration`): its turning points at sigma0's
    # prominence, its librating episodes, why its longest run is not one when there is none; member_tp < 0: the angle
    # could not be built, None: not a scan's member (a hand-made row; the scan measures every member step 1 kept);
    # (t0, t1) its longest librating episode
    member_tp: Optional[int] = None
    member_episodes: int = 0
    member_reject: str = ''
    member_t0_yr: float = NAN
    member_t1_yr: float = NAN

    def extra_columns(self) -> dict:
        """candidates.csv columns besides those of `SubresonanceCandidate`."""
        return {
            'member_degree': self.degree,
            'strength_rel': column_number(self.strength_rel),
            'kept': self.kept,
            'rate_tol_rad_kyr': column_number(self.rate_tolerance_rad_yr, KYR),
            'rate_resid_tol': column_number(rate_residual(self.centre_rate_rad_yr, self.rate_tolerance_rad_yr)),
            'centre_dist_deg': column_number(self.centre_distance_rad, RAD_TO_DEG),
            'force_R2_med': column_number(self.force_r2_median),
            'n_force_windows': self.n_force_windows,
            'single_rms_deg': column_number(self.single_rms_rad, RAD_TO_DEG),
            'coef_abs': column_number(self.coefficient_abs),
            'offset_rate_rad_kyr': column_number(self.offset_rate_rad_yr, KYR),
            'member_tp': _measured_count(self, 'member_tp'),
            'member_episodes': _measured_count(self, 'member_episodes'),
            'member_reject': self.member_reject,
            'member_t0_yr': column_number(self.member_t0_yr),
            'member_t1_yr': column_number(self.member_t1_yr),
        }


@dataclass(frozen=True)
class TheoryCheck:
    """Composite multiplet theory over the window (test D numbers and test E); nan = not measured."""

    composite_rms_rad: float = NAN  # RMS of c0 about pi - arg P(beta(t)) + j_cat beta + best constant
    composite_offset_rad: float = NAN  # that constant (should be ~0: the sign convention)
    single_rms_rad: float = NAN  # the best single-member hypothesis other than the accepted member (sigma0 included)
    sigma0_single_rms_rad: float = NAN  # c0 about its own mean: the hypothesis 'sigma0 owns the oscillation'
    period_corr: float = NAN  # observed cycle periods against T0(t) F(A)
    period_ratio: float = NAN  # median observed / predicted
    n_period_cycles: int = 0
    t0_ratio: float = NAN  # strongest predicted T0 over the window / median observed period (test E)
    confirms: Optional[bool] = None  # test D on the accepted member (None: not measured or nothing accepted)

    @property
    def measured(self) -> bool:
        return bool(np.isfinite(self.composite_rms_rad))


@dataclass(frozen=True)
class ListedMember:
    """A member in the ranked list `sub_top` (display only: no arrays kept), its numbers over the scan window (S2 of
    W20): W the range of its slow centre c0 + smooth(Delta_s), W/W0 against the catalogue angle's own, E the share of
    the catalogue centre's motion its slow angle explains (1 - var(c_s) / var(c0)), N how many of its own slow cycles
    (Delta_s) the window holds (< 1: only a rate matched), A/W the median fast full swing of sigma0 over W (>= 1 looks
    like a libration, << 1 like a drift), and the d'Alembert factor against the strongest term (`sub_strength_caveat`
    says when it is unreliable). The catalogue angle's own row has E = 0, N = 0."""

    key: str
    degree: int
    nodal: bool
    rate_rad_yr: float  # the least-squares rate of the member's slow centre
    strength_rel: float  # test E' number
    width_rad: float = NAN
    width_ratio: float = NAN
    explained: float = NAN
    slow_cycles: float = NAN
    swing_ratio: float = NAN

    def text(self) -> str:
        return f'{self.key}:{self.width_ratio:.2g}:{self.explained:.2g}:{self.slow_cycles:.2g}:{self.swing_ratio:.2g}'


@dataclass
class SubresonanceScan(SubresonanceResult):
    """`SubresonanceResult` with every number the check read (`to_columns`).

    `theory_measured` of the base is True only when test D was measured AND confirms the accepted member (the tree
    reads it as 'the theory supports the verdict', margin S); `theory.measured` / `sub_theory_computed` says whether
    the composite theory was computed at all."""

    window: Tuple[float, float] = (NAN, NAN)
    n_kept: int = 0
    swing_rad: float = NAN  # sigma0's largest half-amplitude in the window (`window_swing`; test A needs the floor)
    centre0_range_rad: float = NAN
    centre0_rate_rad_yr: float = NAN
    centre0_rate_err_rad_yr: float = NAN
    leading_j: Optional[int] = None  # argmax |C_j| at the mean e, e_P (two-body)
    binomial_ratio: float = NAN  # c of C_j ~ C(q, j) (-c e_P / e)^j from C_1 / C_0
    rouche: Optional[bool] = None  # |C_N| > sum |C_j| for the accepted (else leading) member
    winding: Optional[int] = None
    theory: TheoryCheck = field(default_factory=TheoryCheck)
    spacing_au: float = NAN
    adiabaticity: float = NAN  # |varpi_dot - varpi_dot_P| / (2 pi / P_cyc)
    beta_librates: bool = False
    strongest_factor: float = NAN  # d'Alembert factor of the strongest term of the commensurability (test E')
    nodal_scanned: bool = False  # the inclination reached mmr_subresonance_nodal_inc_min_deg
    listed: List[ListedMember] = field(default_factory=list)  # the best mmr_subresonance_top_n (`rank_members`)
    n_listed: int = 0  # members ranked for the list (up to `degree_max`, with sigma +- 2 m omega for the nodal scan)
    catalogue_row: Optional[ListedMember] = None  # the catalogue angle's own numbers of the list (W0, A/W0)
    fast_swing_rad: float = NAN  # A: sigma0's median full swing between turning points in the window
    potential_width_rad: float = NAN  # W of the two-body 'minimum of the averaged potential' row (`potential_row`)
    potential_explained: float = NAN  # its E
    strength_caveat: str = ''  # why the list's d'Alembert factors are unreliable: e, i, crossing (`strength_caveat`)
    n_weak: int = 0  # members test E' rejected before step 1 (planar ones stay in `candidates`)
    entries: List[LedgerEntry] = field(default_factory=list)  # comparisons of the scan, for the figure card

    def to_columns(self) -> dict:
        columns = super().to_columns()
        ranked = sorted(self.candidates, key=lambda c: c.rank)
        best = self.accepted or (ranked[0] if ranked else None)
        theory = self.theory

        def of(attribute: str, scale: float = 1.0):
            return column_number(getattr(best, attribute, NAN), scale) if best is not None else None

        columns.update(
            {
                'sub_window_t0_yr': column_number(self.window[0]),
                'sub_window_t1_yr': column_number(self.window[1]),
                'sub_n_kept': self.n_kept,
                'sub_swing_deg': column_number(self.swing_rad, RAD_TO_DEG),
                'sub_centre0_range_deg': column_number(self.centre0_range_rad, RAD_TO_DEG),
                'sub_centre0_rate_rad_kyr': column_number(self.centre0_rate_rad_yr, KYR),
                'sub_centre0_rate_err_rad_kyr': column_number(self.centre0_rate_err_rad_yr, KYR),
                'sub_best_rate_rad_kyr': of('centre_rate_rad_yr', KYR),
                'sub_best_rate_tol_rad_kyr': of('rate_tolerance_rad_yr', KYR),
                'sub_best_max_pass_deg': of('centre_max_pass_rad', RAD_TO_DEG),
                'sub_best_centre_deg': (column_number(np.mod(best.centre_mean_rad, TWO_PI), RAD_TO_DEG) if best is not None else None),
                'sub_best_centre_dist_deg': of('centre_distance_rad', RAD_TO_DEG),
                'sub_best_force_R2': of('force_r2_median'),
                'sub_best_member_tp': _measured_count(best, 'member_tp'),
                'sub_best_member_episodes': _measured_count(best, 'member_episodes'),
                'sub_best_coef_rel': of('coefficient_rel'),
                'sub_best_strength_rel': of('strength_rel'),
                'sub_strongest_factor': column_number(self.strongest_factor),
                'sub_nodal_scanned': self.nodal_scanned,
                'sub_n_listed': self.n_listed,
                'sub_n_weak': self.n_weak,
                'sub_top': join_list(m.text() for m in self.listed),
                **self.table_columns(),
                'sub_accepted_centre_range_deg': (
                    column_number(self.accepted.centre_range_rad, RAD_TO_DEG) if self.accepted is not None else None
                ),
                'sub_accepted_phi0_coherence': column_number(self.accepted.phi0_coherence) if self.accepted is not None else None,
                'sub_leading_j': self.leading_j,
                'sub_binomial_ratio': column_number(self.binomial_ratio),
                'sub_rouche': self.rouche,
                'sub_winding': self.winding,
                'sub_theory_computed': theory.measured,
                'sub_theory_confirms': theory.confirms,
                'sub_sigma0_single_rms_deg': column_number(theory.sigma0_single_rms_rad, RAD_TO_DEG),
                'sub_composite_offset_deg': column_number(theory.composite_offset_rad, RAD_TO_DEG),
                'sub_period_corr': column_number(theory.period_corr),
                'sub_period_ratio': column_number(theory.period_ratio),
                'sub_n_period_cycles': theory.n_period_cycles,
                'sub_t0_ratio': column_number(theory.t0_ratio),
                'sub_spacing_au': column_number(self.spacing_au),
                'sub_adiabaticity': column_number(self.adiabaticity),
                'sub_beta_librates': self.beta_librates,
            }
        )
        return columns

    def table_columns(self) -> dict:
        """The ranked list's numbers (S2): the first TABLE_ROWS members (`sub_rank<i>_*`), the catalogue angle's own W0
        and A/W0, A, the potential-minimum row and the strength caveat."""
        columns = {}
        for i in range(TABLE_ROWS):
            member = self.listed[i] if i < len(self.listed) else None

            def number(attribute: str, scale: float = 1.0, member=member):
                return column_number(getattr(member, attribute), scale) if member is not None else None

            columns.update(
                {
                    f'sub_rank{i + 1}_key': member.key if member is not None else '',
                    f'sub_rank{i + 1}_W_deg': number('width_rad', RAD_TO_DEG),
                    f'sub_rank{i + 1}_W_ratio': number('width_ratio'),
                    f'sub_rank{i + 1}_E': number('explained'),
                    f'sub_rank{i + 1}_N': number('slow_cycles'),
                    f'sub_rank{i + 1}_AW': number('swing_ratio'),
                    f'sub_rank{i + 1}_strength_rel': number('strength_rel'),
                }
            )
        catalogue = self.catalogue_row
        columns.update(
            {
                'sub_cat_W_deg': column_number(catalogue.width_rad, RAD_TO_DEG) if catalogue is not None else None,
                'sub_cat_AW': column_number(catalogue.swing_ratio) if catalogue is not None else None,
                'sub_fast_swing_deg': column_number(self.fast_swing_rad, RAD_TO_DEG),
                'sub_pot_W_deg': column_number(self.potential_width_rad, RAD_TO_DEG),
                'sub_pot_E': column_number(self.potential_explained),
                'sub_strength_caveat': self.strength_caveat,
            }
        )
        return columns


def _measured_count(candidate, attribute: str) -> Optional[int]:
    """A test-L count of the candidate for the CSV; None when test L did not measure it."""
    n_tp = getattr(candidate, 'member_tp', None)
    if n_tp is None or n_tp < 0:
        return None
    return int(getattr(candidate, attribute))


# --------------------------------------------------------------------------------------------- the members


def _slow_names(spec: ResonanceSpec) -> List[str]:
    """Names of the slow angles in the order of the coefficient vector: varpi, varpi_j..., Omega, Omega_j..."""
    return ['varpi'] + [f'varpi:{p}' for p in spec.planets] + ['Omega'] + [f'Omega:{p}' for p in spec.planets]


def slow_coefficients(spec: ResonanceSpec) -> Tuple[int, ...]:
    """(p, p_1.., v, v_1..) of an angle, in the order of `_slow_names`."""
    n = len(spec.planets)
    return (
        (spec.body_varpi_coefficient,)
        + tuple(spec.planet_varpi_coefficients)
        + (spec.body_node_coefficient,)
        + tuple(spec.planet_node_coefficient(i) for i in range(n))
    )


def _vectors(n: int, total: int, budget: int) -> Iterator[Tuple[int, ...]]:
    """Integer vectors of length n with sum `total` and sum |c| <= budget."""
    if n == 1:
        if abs(total) <= budget:
            yield (total,)
        return
    for first in range(-budget, budget + 1):
        rest = budget - abs(first)
        if abs(total - first) <= rest:
            for tail in _vectors(n - 1, total - first, rest):
                yield (first,) + tail


def member_key(spec: ResonanceSpec, coefficients: Sequence[int]) -> str:
    """The package notation of a member (`9M-16+2+5`, `3J-1S-1+0+1-2`) and, for a nodal one, its nodes
    `[N+1,NM-1]` (no package notation exists for them)."""
    n = len(spec.planets)
    varpi, planet_varpi = coefficients[0], coefficients[1 : n + 1]
    node, planet_node = coefficients[n + 1], coefficients[n + 2 :]
    letters = [planet_letter(p) for p in spec.planets]
    text = planet_terms(spec.planets, spec.planet_coefficients)
    text += f'{spec.body_coefficient:+d}' + ''.join(f'{c:+d}' for c in planet_varpi) + f'{varpi:+d}'
    if node or any(planet_node):
        text += '[' + ','.join([f'N{node:+d}'] + [f'N{letter}{c:+d}' for letter, c in zip(letters, planet_node)]) + ']'
    return text


def member_spec(spec: ResonanceSpec, coefficients: Sequence[int]) -> ResonanceSpec:
    """The ResonanceSpec of a member with the slow coefficients `coefficients` (order of `_slow_names`)."""
    n = len(spec.planets)
    planet_node = tuple(int(c) for c in coefficients[n + 2 :])
    return replace(
        spec,
        key=member_key(spec, coefficients),
        body_varpi_coefficient=int(coefficients[0]),
        planet_varpi_coefficients=tuple(int(c) for c in coefficients[1 : n + 1]),
        body_node_coefficient=int(coefficients[n + 1]),
        planet_node_coefficients=planet_node if any(planet_node) else (),
    )


def is_nodal(spec: ResonanceSpec) -> bool:
    return bool(spec.body_node_coefficient) or any(spec.planet_node_coefficient(i) for i in range(len(spec.planets)))


def member_j(spec: ResonanceSpec) -> int:
    """The power of the planets' eccentricity: u_P for a two-body member (the j of sigma_j = sigma_0 + j beta),
    the sum of the planets' varpi coefficients for a three-body one."""
    return int(sum(spec.planet_varpi_coefficients))


def member_degree(spec: ResonanceSpec) -> int:
    """The degree of the term: sum of |slow coefficients|, the power of e, e_P, sin i/2, sin i_P/2 in its leading
    d'Alembert factor. Not the order |q| = |sum k| of the commensurability, which every member shares."""
    return int(sum(abs(c) for c in slow_coefficients(spec)))


def slow_sum(spec: ResonanceSpec) -> int:
    """q: the signed sum of the slow coefficients of every member, -(sum k_j + k) (d'Alembert)."""
    return -(sum(spec.planet_coefficients) + spec.body_coefficient)


def enumerate_subresonances(
    spec: ResonanceSpec, degree_max: int, nodal: bool = False, omega_harmonics: int = 0
) -> List[Tuple[ResonanceSpec, int, bool]]:
    """Members of the multiplet of degree |q| to max(|q|, degree_max): (spec, j, nodal), the catalogue angle excluded.
    Every d'Alembert-valid split of the slow coefficients: u + sum u_P + v + sum v_P = q with v + sum v_P even
    (M&D 1999, §6.7, eqs. 6.139–6.140), so mixed signs too (3J-1S-1+2-2-1, degree 5, of 3J-1S-1+0+0-1).

    The members of degree |q| exist whatever `degree_max` (82831 13M-24 is of order 11 against MATRIX_2BODY_ORDER_MAX
    10). The degree has the parity of q, so a degree_max of the other parity adds nothing over the one below. With
    `nodal` also the members with node coefficients (v or v_P != 0, SPEC §6.3), and sigma +- 2 m omega for m = 1..
    `omega_harmonics` whatever their degree (`omega_vectors`). Retrograde angles (varpi* convention) and 1:1 angles
    (`preprocess.law_planet`: no expansion in e, the averaged potential of `coorbital.py`) have no members here; an
    angle with |k_P| = |k| but k_P = +k (624 1J+1+0-2) is not 1:1 and has them. Sorted by degree, then
    planar first, then by the coefficient vector."""
    if spec.kind == ResonanceKind.RETROGRADE or law_planet(spec) is not None:
        return []
    budget = max(abs(slow_sum(spec)), int(degree_max))
    n = len(spec.planets)
    members = []
    for vector in member_vectors(spec, budget, nodal, omega_harmonics):
        member = member_spec(spec, vector)
        members.append((member, member_j(member), any(vector[n + 1 :])))
    members.sort(key=lambda m: (member_degree(m[0]), m[2], [-c for c in slow_coefficients(m[0])]))
    return members


def member_vectors(spec: ResonanceSpec, degree_max: int, nodal: bool, omega_harmonics: int = 0) -> Iterator[Tuple[int, ...]]:
    """The slow-coefficient vectors (order of `_slow_names`) of the members up to `degree_max`: signed sum q, node
    coefficients summing to an even number, zero nodes unless `nodal`; the catalogue angle excluded. With `nodal` also
    the members sigma +- 2 m omega beyond `degree_max` (`omega_vectors`)."""
    q = slow_sum(spec)
    n = len(spec.planets)
    own = tuple(slow_coefficients(spec))
    if nodal:
        vectors = _vectors(2 * (n + 1), q, degree_max)
    else:
        vectors = (planar + (0,) * (n + 1) for planar in _vectors(n + 1, q, degree_max))
    for vector in vectors:
        if sum(vector[n + 1 :]) % 2 == 0 and tuple(vector) != own:
            yield tuple(vector)
    if nodal:
        for vector in omega_vectors(spec, omega_harmonics):
            if sum(abs(c) for c in vector) > degree_max:  # those within the limit came above
                yield vector


def omega_vectors(spec: ResonanceSpec, harmonics: int) -> List[Tuple[int, ...]]:
    """The slow-coefficient vectors of sigma +- 2 m omega, m = 1..harmonics (omega = varpi - Omega of the body): the
    offset 2 m (varpi - Omega) keeps the signed sum q and an even node sum (d'Alembert). The libration centre of an
    inclined orbit moves with omega (Gallardo 2006, Icarus 184, sect. 3, p. 30), so these members can stand while sigma0's
    centre drifts at -2 omega_dot (2003 UG22 10E-17+0+7: sigma + 2 omega, degree 11)."""
    own = list(slow_coefficients(spec))
    node = len(spec.planets) + 1
    vectors = []
    for m in range(1, int(harmonics) + 1):
        for sign in (1, -1):
            vector = list(own)
            vector[0] += sign * 2 * m
            vector[node] -= sign * 2 * m
            vectors.append(tuple(vector))
    return vectors


# --------------------------------------------------------------------------------------------- strength (test E')


def term_factor(coefficients: Sequence[int], factors: Sequence[float]) -> float:
    """The leading d'Alembert factor prod x^|c| of a term: x = e, e_P.., sin i/2, sin i_P/2.. in the order of
    `_slow_names`; a zero coefficient contributes 1 whatever its x."""
    value = 1.0
    for c, x in zip(coefficients, factors):
        if c:
            value *= float(x) ** abs(int(c))
    return value


def strongest_factor(q: int, eccentricity_factors: Sequence[float], node_factors: Sequence[float]) -> float:
    """The largest d'Alembert factor of the commensurability: a term of degree |q| (one sign, every factor < 1) with
    an even number of node powers, max over that number of x_e^(|q| - m) x_n^m (x_e, x_n the largest eccentricity and
    inclination factors). Without a finite eccentricity factor: nan."""
    eccentricities = [float(x) for x in eccentricity_factors if np.isfinite(x)]
    if not eccentricities:
        return NAN
    x_e = max(eccentricities)
    x_n = max((float(x) for x in node_factors if np.isfinite(x)), default=0.0)
    return float(max(x_e ** (abs(q) - m) * x_n**m for m in range(0, abs(q) + 1, 2)))


def _median_sin_half_inc(inc, Omega, omega) -> float:
    """Median sin(i/2) of an orbit with i from the invariable plane (on <= TRACK_POINTS samples)."""
    step = max(1, -(-len(inc) // TRACK_POINTS))
    inc_inv, _, _ = to_invariable_plane(np.asarray(inc)[::step], np.asarray(Omega)[::step], np.asarray(omega)[::step])
    return float(np.median(np.sin(inc_inv / 2)))


def strength_factors(series: MMRSeries, context: Context) -> List[float]:
    """x of every slow angle in the order of `_slow_names`: the body's median e (context.e_med), the planets' mean
    e (saved planets only, else nan), the median sin(i/2) of the body and the planets from the invariable plane."""
    spec = series.resonance
    planets = [series.planets.get(name) for name in spec.planets]
    node_body = _median_sin_half_inc(series.inc, series.Omega, series.omega)
    node_planets = [NAN if p is None else _median_sin_half_inc(p.inc, p.Omega, np.asarray(p.varpi) - np.asarray(p.Omega)) for p in planets]
    eccentricities = [context.planet_e_mean.get(name, NAN) for name in spec.planets]
    return [context.e_med] + eccentricities + [node_body] + node_planets


def nodal_applies(sin_half_inc: float, settings: AnalysisSettings) -> bool:
    """Members with node coefficients are enumerated: the body's median sin(i/2) (i from the invariable plane,
    `strength_factors`) reaches that of mmr_subresonance_nodal_inc_min_deg. A nodal term of degree |q| trades two
    powers of e for two of sin(i/2), a factor (sin(i/2) / e)^2: at 10 deg sin(i/2) = 0.087, comparable with the
    e = 0.1-0.2 of the scanned bodies; below it the planar members dominate, and a weak nodal member can still win
    the centre-range test (12218 4M-7 at i = 5.5 deg: 4M-7+2+1[N-1,NM+1], factor 1.2e-3, against the accepted +1+2)."""
    return bool(np.isfinite(sin_half_inc) and sin_half_inc >= np.sin(np.radians(settings.mmr_subresonance_nodal_inc_min_deg) / 2))


def rate_residual(rate_rad_yr: float, tolerance_rad_yr: float) -> float:
    """|centre rate| in units of the step-1 rate tolerance (nan when either is not finite or the tolerance is 0)."""
    if not (np.isfinite(rate_rad_yr) and np.isfinite(tolerance_rad_yr) and tolerance_rad_yr > 0):
        return NAN
    return float(abs(rate_rad_yr) / tolerance_rad_yr)


# --------------------------------------------------------------------------------------------- slow angles


def _unwrap(x) -> np.ndarray:
    return np.unwrap(np.asarray(x, dtype=float))


def model_years_since_j2000(t_yr, epoch_jd: float):
    """Time since J2000 in the model's (Gaussian) years: t_yr (already Gaussian years from the epoch) plus the epoch's
    offset (JD - J2000) / (2 pi / k) days; an unknown epoch (nan) counts as J2000. The one conversion of the module."""
    shift = (epoch_jd - J2000_JD) / GAUSSIAN_YEAR_DAYS if np.isfinite(epoch_jd) else 0.0
    return np.asarray(t_yr, dtype=float) + shift


def _mode_longitude(t_yr: np.ndarray, planet: str, kind: str, epoch_jd: float) -> np.ndarray:
    """g_j t + phase_j (kind 'g') or s_j t + phase_j ('s') of the planet's own mode [rad] (const tables, J2000).

    The rates of const.PLANETARY_FREQUENCIES are read per model (Gaussian) year, the unit they were measured in for
    g1-g4, s1-s4 and the unit of t_yr; for the literature g5-g8, s6-s8 a Julian reading differs by 1.9e-5 of the rate
    (0.15 deg per Myr for g6), below the precision of the phases. t runs from J2000 (`model_years_since_j2000`)."""
    index = PLANET_NAMES_ORDER.index(planet) + 1
    rate = planet_frequency(planet, kind)
    return rate * model_years_since_j2000(t_yr, epoch_jd) + np.radians(const.PLANETARY_PHASES[f'{kind}{index}'])


def slow_angles(series: MMRSeries, invariable: bool = False) -> Dict[str, np.ndarray]:
    """The unwrapped slow angles of the angle's members [rad]: 'varpi', 'Omega' of the body and 'varpi:<P>',
    'Omega:<P>' of each planet of the resonance; ecliptic, or all from the invariable plane. A planet that is not
    saved gets its mode longitudes g_j t + phase_j, s_j t + phase_j (rates usable, phases not: SPEC §6.3)."""
    spec = series.resonance
    t = np.asarray(series.t_yr, dtype=float)
    angles = {}
    if invariable:
        _, node, varpi = to_invariable_plane(series.inc, series.Omega, series.omega)
        angles['varpi'], angles['Omega'] = _unwrap(varpi), _unwrap(node)
    else:
        angles['varpi'], angles['Omega'] = _unwrap(series.varpi), _unwrap(series.Omega)
    for name in spec.planets:
        planet = series.planets.get(name)
        if planet is None:
            angles[f'varpi:{name}'] = _mode_longitude(t, name, 'g', series.regime.epoch_jd)
            angles[f'Omega:{name}'] = _mode_longitude(t, name, 's', series.regime.epoch_jd)
        elif invariable:
            _, node, varpi = to_invariable_plane(planet.inc, planet.Omega, np.asarray(planet.varpi) - np.asarray(planet.Omega))
            angles[f'varpi:{name}'], angles[f'Omega:{name}'] = _unwrap(varpi), _unwrap(node)
        else:
            angles[f'varpi:{name}'], angles[f'Omega:{name}'] = _unwrap(planet.varpi), _unwrap(planet.Omega)
    return angles


def offset_coefficients(spec: ResonanceSpec, member: ResonanceSpec) -> Dict[str, int]:
    """Delta_s = sum_name d_name x angle_name: the member's slow coefficients minus the catalogue's (non-zero only)."""
    names = _slow_names(spec)
    return {name: b - a for name, a, b in zip(names, slow_coefficients(spec), slow_coefficients(member)) if b != a}


def _combine(coefficients: Dict[str, int], angles: Dict[str, np.ndarray], n: int) -> np.ndarray:
    total = np.zeros(n)
    for name, coefficient in coefficients.items():
        total = total + coefficient * angles[name]
    return total


def slow_offset(spec: ResonanceSpec, member: ResonanceSpec, series: MMRSeries) -> np.ndarray:
    """Delta_s(t) = sigma_member - sigma0 [rad], from the body's and the planets' varpi (and Omega for nodal members).

    Planar members: ecliptic elements, so sigma0 + Delta_s is exactly the member's angle in the frame of the
    saved sigma0 (a nodal catalogue angle included: its planar members differ from it by ecliptic nodes). Nodal
    members: the member's angle in the invariable plane, Delta_s from the invariable plane plus sigma0's change of frame
    (`plane_shift`; the mixed frame, Delta_s alone, when a planet of the angle is not saved).
    ValueError for a retrograde angle or a member of another commensurability."""
    if spec.retrograde or member.retrograde:
        raise ValueError('retrograde angles have no subresonances here (varpi* convention)')
    if (
        spec.planets != member.planets
        or spec.planet_coefficients != member.planet_coefficients
        or spec.body_coefficient != member.body_coefficient
    ):
        raise ValueError(f'{member.key} is not a member of the multiplet of {spec.key}')
    if not is_nodal(member):
        return _combine(offset_coefficients(spec, member), slow_angles(series, False), len(series.t_yr))
    angles = {frame: slow_angles(series, frame) for frame in (False, True)}
    offset = _combine(offset_coefficients(spec, member), angles[True], len(series.t_yr))
    shift = plane_shift(series, angles)
    return offset if shift is None else offset + shift


def member_series(series: MMRSeries, member: ResonanceSpec) -> MMRSeries:
    """The member's own record: angle_wrapped = (sigma0 + Delta_s) mod 2 pi, from the saved elements (no integration)."""
    offset = slow_offset(series.resonance, member, series)
    wrapped = np.mod(np.asarray(series.angle_wrapped, dtype=float) + offset, TWO_PI)
    numpy_angle = None if series.angle_unwrapped_numpy is None else np.asarray(series.angle_unwrapped_numpy, dtype=float) + offset
    return replace(series, resonance=member, angle_wrapped=wrapped, angle_unwrapped_numpy=numpy_angle)


def plane_shift(series: MMRSeries, angles: Optional[Dict[bool, Dict[str, np.ndarray]]] = None) -> Optional[np.ndarray]:
    """sigma0(invariable) - sigma0(ecliptic) [rad] along the record: every longitude changes by its varpi' - varpi (the
    mean anomalies do not depend on the plane: lambda' = varpi' + M), so the difference is sum_j (k_j + p_j)(varpi_j' -
    varpi_j) + (k + p)(varpi' - varpi) + the node terms; the d'Alembert sum makes it independent of the longitude
    origin. Continuous (each difference of two unwrapped angles), defined up to a constant multiple of 2 pi. None for
    a retrograde angle or without a planet of the angle (its mode longitudes are no osculating elements).
    `angles`: the slow angles of both frames ({False: ecliptic, True: invariable}) when the caller has them already."""
    spec = series.resonance
    if spec.retrograde or any(p not in series.planets for p in spec.planets):
        return None
    angles = angles if angles is not None and False in angles and True in angles else {f: slow_angles(series, f) for f in (False, True)}
    ecliptic, invariable = angles[False], angles[True]
    total = (spec.body_coefficient + spec.body_varpi_coefficient) * (invariable['varpi'] - ecliptic['varpi'])
    total = total + spec.body_node_coefficient * (invariable['Omega'] - ecliptic['Omega'])
    for i, name in enumerate(spec.planets):
        coefficient = spec.planet_coefficients[i] + spec.planet_varpi_coefficients[i]
        total = total + coefficient * (invariable[f'varpi:{name}'] - ecliptic[f'varpi:{name}'])
        total = total + spec.planet_node_coefficient(i) * (invariable[f'Omega:{name}'] - ecliptic[f'Omega:{name}'])
    return np.asarray(total, dtype=float)


def plane_offset(series: MMRSeries, angles: Optional[Dict[bool, Dict[str, np.ndarray]]] = None) -> float:
    """Largest |sigma0(invariable) - sigma0(ecliptic)| [rad] (`plane_shift`): how far the mixed frame (sigma0 from the
    ecliptic, Delta_s of a nodal member from the invariable plane) would put a nodal member. nan without the planets."""
    shift = plane_shift(series, angles)
    return NAN if shift is None else float(np.max(np.abs(wrap_pi(shift))))


# --------------------------------------------------------------------------------------------- triggers


def triggers(
    geometry: Geometry,
    response: Response,
    pending_reason: str,
    binomial_mode_j: int,
    stationary_full: bool,
    settings: AnalysisSettings,
) -> Tuple[Tuple[SubresonanceTrigger, ...], Tuple[str, ...]]:
    """(scan triggers T1, T2, T5, T6; support T3, T4) of SPEC §6.2. `pending_reason` is the reason the tree would give
    without the scan (T5 for kinematic_* and drift_mechanism_unknown).

    T1: the drift fraction reaches libration_drift_min_window_fraction, a run of >= libration_min_turning_points TPs
        is rejected (centre_moves, or no_closed_cycle: a saw that never returns) or an accepted episode is not
        stationary (modulated, moving_centre), or the centre makes a turn
        (largest pass >= 2 pi) while a TP-rich run exists (a saw on a slope).
    T2: median force-law R^2 over the windows >= mmr_subresonance_t2_r2_min and phi0 coherence < _t2_coh_max.
    T5: `pending_reason` in kinematic_drift, kinematic_libration, drift_mechanism_unknown.
    T6: `binomial_mode_j` != 0 and the angle is not a clean stationary libration with full margins.
    T3 and T4 need the slow angles; they are measured by `run_subresonances` (`support` is empty here)."""
    scratch = Ledger.from_settings(settings)
    found = []
    tp_min = settings.libration_min_turning_points
    rich = [run for run in geometry.runs if run.n_tp >= tp_min]
    drift = scratch.compare_setting(
        'T1 drift fraction', geometry.drift_fraction, Op.GE, settings, 'libration_drift_min_window_fraction', '', 'geo_drift_fraction'
    )
    moving = any((not run.accepted and str(run.reject_reason) in TRIGGER_REJECTIONS) for run in rich)
    modulated = any(run.accepted and str(run.shape) in ('modulated', 'moving_centre') for run in rich)
    turning = bool(rich) and bool(scratch.compare('T1 centre pass', geometry.centre.max_pass_rad, Op.GE, TWO_PI, 'rad', ''))
    if drift or moving or modulated or turning:
        found.append(SubresonanceTrigger.T1)
    if response is not None:
        r2 = [w.r2 for w in response.windows if np.isfinite(w.r2)]
        responds = scratch.compare_setting(
            'T2 force R2', float(np.median(r2)) if r2 else NAN, Op.GE, settings, 'mmr_subresonance_t2_r2_min', '', ''
        )
        rotates = scratch.compare_setting(
            'T2 phi0 coherence', response.phi0_coherence, Op.LT, settings, 'mmr_subresonance_t2_coh_max', '', 'rsp_phi0_coherence'
        )
        if responds and rotates:
            found.append(SubresonanceTrigger.T2)
    if str(pending_reason) in PENDING_KINEMATIC:
        found.append(SubresonanceTrigger.T5)
    if binomial_mode_j and not stationary_full:
        found.append(SubresonanceTrigger.T6)
    return tuple(found), ()


def leading_member(series: MMRSeries, context: Context, settings: AnalysisSettings) -> Tuple[Optional[int], Optional[int]]:
    """(leading j, binomial j*) of a two-body angle at the record's median e and the planet's mean e, a: the member with
    the largest |C_j| by the averaging, and the binomial mode with the weight c measured from C_1 / C_0 (T6 input;
    SPEC §6.1). (None, None) for three-body or retrograde angles, without the planet's saved elements, for crossing
    orbits and when the coefficients are below round-off."""
    spec = series.resonance
    if spec.kind != ResonanceKind.TWO_BODY:
        return None, None
    planet = spec.planets[0]
    a_p, e_p, e = context.planet_axis_mean_au.get(planet), context.planet_e_mean.get(planet), context.e_med
    if not all(v is not None and np.isfinite(v) for v in (a_p, e_p, e)) or not (0 <= e < 1):
        return None, None
    q = -(spec.planet_coefficients[0] + spec.body_coefficient)
    js = multiplet.planar_members(spec.planet_coefficients[0], spec.body_coefficient)
    if planet in context.crosses or multiplet.crossing(context.a_mean_au, e, a_p, e_p):
        return None, None
    mass = context.planet_mass.get(planet, const.PLANETS_MASS.get(planet, NAN))
    coefficients, _ = multiplet.cached_coefficients(
        spec.key, spec.planet_coefficients[0], spec.body_coefficient, context.a_mean_au, a_p, e, e_p, mass, js
    )
    if not multiplet.resolved(coefficients, mass, context.a_mean_au, a_p):
        return None, None
    leading = int(js[int(np.argmax(np.abs(coefficients)))])
    ratio = multiplet.binomial_ratio(coefficients, js, q, e, e_p)
    return leading, multiplet.binomial_mode(q, e, e_p, ratio)


# --------------------------------------------------------------------------------------------- step 1


def centre_track(centre0: np.ndarray, offset: np.ndarray, t_yr: np.ndarray, window_yr: float, filter_order: int) -> np.ndarray:
    """c_s = c0 + smooth_{W_c}(Delta_s) [rad] (the same low-pass as the centre; the raw offset when W_c is not longer
    than two steps)."""
    centre0 = np.asarray(centre0, dtype=float)
    try:
        smooth = filtering.lowpass(t_yr, offset, window_yr, filter_order)
    except ValueError:
        smooth = np.asarray(offset, dtype=float)
    return centre0 + smooth


def _line(t_yr: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    """(slope, its error from the residual wander 2 std(resid) sqrt(3) / T) of a least-squares line."""
    good = np.isfinite(values)
    if np.count_nonzero(good) < 3:
        return NAN, NAN
    t, v = t_yr[good], values[good]
    slope, intercept = np.polyfit(t - t[0], v, 1)
    resid = v - (intercept + slope * (t - t[0]))
    span = float(t[-1] - t[0])
    return float(slope), float(SQRT_12 * np.std(resid) / span) if span > 0 else NAN


def screen(
    candidate_track: np.ndarray, centre0: np.ndarray, t_yr: np.ndarray, settings: AnalysisSettings, ledger: Optional[Ledger] = None
) -> Tuple[bool, float, float, float]:
    """Step 1: range <= mmr_subresonance_range_ratio x range(c0), |rate| <= max(2 sigma_rate, floor), largest pass
    < mmr_subresonance_pass_max_rad. Returns (kept, range [rad], rate [rad/yr], largest pass [rad]).

    The arrays are those of the window already. sigma_rate is the slope error from the wander of the track's residual
    about its line (2 std sqrt(3) / T: not white noise, SISTERS §3 T3); the floor is mmr_subresonance_rate_floor_rad_kyr.
    Passes are measured by max/min between the track's turns of prominence libration_centre_tp_prominence_deg."""
    ledger = ledger if ledger is not None else Ledger.from_settings(settings)
    track = np.asarray(candidate_track, dtype=float)
    t = np.asarray(t_yr, dtype=float)
    good = np.isfinite(track)
    if np.count_nonzero(good) < 3:
        return False, NAN, NAN, NAN
    span = float(np.ptp(track[good]))
    span0 = float(np.ptp(np.asarray(centre0, dtype=float)[np.isfinite(centre0)])) if np.any(np.isfinite(centre0)) else NAN
    rate, error = _line(t, track)
    tolerance = max(2 * error if np.isfinite(error) else 0.0, settings.mmr_subresonance_rate_floor_rad_kyr / KYR)
    window = (float(t[good][0]), float(t[good][-1]))
    turns = find_turning_points(t, track, np.radians(settings.libration_centre_tp_prominence_deg), window)
    best = largest_monotone_pass(monotone_passes(t, track, turns, window))
    largest = abs(best.delta_rad) if best is not None else span
    in_range = ledger.compare_setting(
        'S1 centre range / range(c0)',
        span / span0 if span0 > 0 else NAN,
        Op.LE,
        settings,
        'mmr_subresonance_range_ratio',
        '',
        'sub_best_centre_range_deg',
    )
    slow = ledger.compare('S1 |centre rate|', abs(rate), Op.LE, tolerance, 'rad/yr', 'sub_best_rate_rad_kyr')
    short = ledger.compare_setting(
        'S1 largest centre pass', largest, Op.LT, settings, 'mmr_subresonance_pass_max_rad', 'rad', 'sub_best_max_pass_deg'
    )
    return bool(in_range and slow and short), span, rate, largest


def _screen_reason(track, centre0, t, settings) -> Tuple[str, float, float, float, float]:
    """(reason or '', range, rate, pass, rate tolerance): which part of step 1 failed first."""
    scratch = Ledger.from_settings(settings)
    kept, span, rate, largest = screen(track, centre0, t, settings, scratch)
    entries = list(scratch)
    tolerance = entries[1].threshold if len(entries) > 1 else NAN
    if kept:
        return '', span, rate, largest, tolerance
    reasons = [REASON_RANGE, REASON_RATE, REASON_PASS]
    failed = [reason for reason, entry in zip(reasons, entries) if not entry]
    return (failed[0] if failed else REASON_RANGE), span, rate, largest, tolerance


# --------------------------------------------------------------------------------------------- step 2


def _distance_to_symmetric(centre_rad: float) -> float:
    """Distance of an angle from the nearer of 0 and pi [rad]."""
    if not np.isfinite(centre_rad):
        return NAN
    return float(min(abs(wrap_pi(centre_rad)), abs(wrap_pi(centre_rad - np.pi))))


def _passes_a(
    c: SubresonanceCandidate, sigma0_coherence: float, settings: AnalysisSettings, ledger: Ledger, small_swing: bool = False
) -> str:
    """Test A: '' when the force-law phase relative to the member is coherent and gains over sigma0, else the reason
    (A_small_swing when the scan did not measure it because sigma0 swings below the floor, `_force_law`)."""
    if small_swing:
        return REASON_A_SMALL_SWING
    if not np.isfinite(c.phi0_coherence):
        return REASON_A_NOT_MEASURED
    name = f'A {c.spec.key} phi0 coherence'
    column = 'sub_best_phi0_coherence'
    if not ledger.compare_setting(name, c.phi0_coherence, Op.GE, settings, 'mmr_subresonance_phi0_coh_min', '', column):
        return REASON_A_COHERENCE
    gain = c.phi0_coherence - sigma0_coherence if np.isfinite(sigma0_coherence) else NAN
    name = f'A {c.spec.key} coherence gain over sigma0'
    column = 'sub_sigma0_phi0_coherence'
    if not ledger.compare_setting(name, gain, Op.GE, settings, 'mmr_subresonance_phi0_coh_gain', '', column):
        return REASON_A_GAIN
    return ''


def _librates_itself(c: SubresonanceCandidate, settings: AnalysisSettings, ledger: Ledger) -> str:
    """Test L: '' when the member's own angle librates in the scan window, else the reason. It librates when it has
    an episode there that the row of the member itself would count as a libration (`member_libration`: accepted by
    the geometry, >= libration_min_turning_points TPs at sigma0's prominence, a stationary or modulated centre, a
    closed cycle; and `decide.librating_episodes`: not faster than the period floor, no centre revolution inside).
    A centre track that stands still and a coherent force-law phase do not show that the member turns back: on a
    member that only creeps (63047 3J+7S-2-8+0+0: no TP in the window, its own row circulates) sin(sigma_s - phi0) is
    nearly constant and phi0 is set by the noise (coherence 0.86, R^2 0.16). A candidate without `member_tp` (a bare
    `SubresonanceCandidate`, or a member with member_tp None: callers of `decide`/`evaluate` with hand-made rows)
    passes: it holds no angle to measure. The scan sets it for every member step 1 kept."""
    n_tp = getattr(c, 'member_tp', None)
    if n_tp is None:
        return ''
    if n_tp < 0:
        ledger.compare(
            f'L {c.spec.key} own TPs in the window', None, Op.GE, settings.libration_min_turning_points, '', 'sub_best_member_tp'
        )
        return REASON_L_NOT_MEASURED
    name = f'L {c.spec.key} own TPs in the window'
    if not ledger.compare_setting(name, n_tp, Op.GE, settings, 'libration_min_turning_points', '', 'sub_best_member_tp'):
        return REASON_L
    name = f'L {c.spec.key} own librating episodes in the window'
    if not ledger.compare(name, c.member_episodes, Op.GE, 1, '', 'sub_best_member_episodes'):
        return REASON_L
    return ''


def _degree(c: SubresonanceCandidate) -> int:
    return int(getattr(c, 'degree', 0) or member_degree(c.spec))


def _rank_tie(group: List[SubresonanceCandidate]) -> Tuple[SubresonanceCandidate, bool, bool]:
    """(winner, decided by degree, decided at all) among candidates with tied centre ranges: lower degree, then the
    larger coefficient; else the smallest range, undecided."""
    lowest = min(_degree(c) for c in group)
    by_order = [c for c in group if _degree(c) == lowest]
    if len(by_order) == 1:
        return by_order[0], True, True
    coefficients = [c.coefficient_rel for c in by_order]
    if all(np.isfinite(coefficients)) and len(set(coefficients)) == len(coefficients):
        return by_order[int(np.argmax(coefficients))], False, True
    return min(by_order, key=lambda c: c.centre_range_rad), False, False


def _unique(
    passing: List[SubresonanceCandidate], settings: AnalysisSettings, reasons: Dict[str, str]
) -> Tuple[SubresonanceCandidate, bool]:
    """Test C: the smallest centre range, or the tie-break among ranges within mmr_subresonance_tie_fraction of it.
    Returns (winner, the tie was decided by order only or not at all)."""
    best = min(passing, key=lambda c: c.centre_range_rad)
    fraction = settings.mmr_subresonance_tie_fraction
    tie = [c for c in passing if abs(c.centre_range_rad - best.centre_range_rad) <= fraction * best.centre_range_rad]
    if len(tie) == 1:
        return best, False
    best, by_order, decided = _rank_tie(tie)
    for c in tie:
        if c is not best:
            reasons[c.spec.key] = REASON_C
    return best, by_order or not decided


def _symmetric(best: SubresonanceCandidate, e_mean: float, inc_max_rad: float, applies: bool, settings, ledger: Ledger) -> bool:
    """Test B: the mean centre within mmr_subresonance_centre_tol_deg of 0 or pi (True when B does not apply)."""
    if not applies or (np.isfinite(e_mean) and e_mean > settings.mmr_subresonance_centre_e_max):
        return True
    if np.isfinite(inc_max_rad) and inc_max_rad > np.radians(settings.mmr_subresonance_centre_inc_max_deg):
        return True
    distance = np.degrees(_distance_to_symmetric(best.centre_mean_rad))
    tolerance = settings.mmr_subresonance_centre_tol_deg
    return bool(
        ledger.compare(
            'B centre from 0 or pi',
            distance,
            Op.LE,
            tolerance,
            'deg',
            'sub_best_centre_dist_deg',
            setting='mmr_subresonance_centre_tol_deg',
        )
    )


def _strong_relative(c: SubresonanceCandidate, settings: AnalysisSettings, ledger: Ledger) -> bool:
    """Test E': the member's d'Alembert factor >= mmr_subresonance_strength_rel_min x that of the strongest term of
    the commensurability (True when not measured: no strength on the candidate). The pendulum frequency goes as the
    square root of the term, so a member below 1e-3 would librate >= 30 times slower than the strongest term: it
    cannot be what sets the oscillation every member shares. A rough order of magnitude (the expansion in e does not
    converge for crossing orbits), enough for the gap it has to resolve (1e-10 for 10E-17+7+0 of 2003 UG22 against
    >= 0.17 for every accepted member of the benchmark)."""
    value = getattr(c, 'strength_rel', NAN)
    if not np.isfinite(value):
        return True
    limit = settings.mmr_subresonance_strength_rel_min
    name = f"E' {c.spec.key} factor / strongest term"
    return bool(ledger.compare(name, value, Op.GE, limit, '', 'sub_best_strength_rel', setting='mmr_subresonance_strength_rel_min'))


def _strong_enough(theory: Optional[TheoryCheck], settings: AnalysisSettings, ledger: Ledger) -> bool:
    """Test E: the strongest predicted T0 <= mmr_subresonance_t0_ratio_max x the observed period (True when not measured)."""
    if theory is None or not np.isfinite(theory.t0_ratio):
        return True
    limit = settings.mmr_subresonance_t0_ratio_max
    return bool(
        ledger.compare(
            'E predicted T0 / observed period', theory.t0_ratio, Op.LE, limit, '', 'sub_t0_ratio', setting='mmr_subresonance_t0_ratio_max'
        )
    )


def _rival_single_rms(best: SubresonanceCandidate, candidates: Sequence[SubresonanceCandidate], theory: TheoryCheck) -> float:
    """The best single-member hypothesis other than `best` (sigma0 included): what the composite theory must beat."""
    values = [getattr(c, 'single_rms_rad', NAN) for c in candidates if c is not best and not c.nodal]
    values = [v for v in values + [theory.sigma0_single_rms_rad] if np.isfinite(v)]
    return float(min(values)) if values else theory.single_rms_rad


def theory_confirms(
    best: SubresonanceCandidate,
    candidates: Sequence[SubresonanceCandidate],
    theory: TheoryCheck,
    settings: AnalysisSettings,
    ledger: Ledger,
) -> Optional[bool]:
    """Test D (None when not measured): the composite centre pi - arg P(beta(t)) follows c0 (RMS <= _theory_rms_deg and
    <= _theory_single_ratio x the best rival single-member hypothesis), `best` has the largest coefficient or within
    _theory_coef_fraction of it, and the cycle periods follow T0(t) F(A) (correlation >= _period_corr_min, median ratio
    within x _period_ratio_max). D only raises the confidence: a failure does not reject the member."""
    if not (theory.measured and settings.mmr_subresonance_theory):
        return None
    rms_deg = np.degrees(theory.composite_rms_rad)
    rms = ledger.compare_setting(
        'D composite centre RMS', rms_deg, Op.LE, settings, 'mmr_subresonance_theory_rms_deg', 'deg', 'sub_composite_rms_deg'
    )
    rival = _rival_single_rms(best, candidates, theory)
    ratio = theory.composite_rms_rad / rival if rival > 0 else NAN
    better = ledger.compare_setting(
        'D composite / best rival single RMS', ratio, Op.LE, settings, 'mmr_subresonance_theory_single_ratio', '', 'sub_single_rms_deg'
    )
    strongest = True
    if np.isfinite(best.coefficient_rel):
        floor = 1 - settings.mmr_subresonance_theory_coef_fraction
        strongest = ledger.compare(
            'D coefficient / largest',
            best.coefficient_rel,
            Op.GE,
            floor,
            '',
            'sub_best_coef_rel',
            setting='mmr_subresonance_theory_coef_fraction',
        )
    periods = True
    if np.isfinite(theory.period_corr):
        corr = ledger.compare_setting(
            'D period correlation', theory.period_corr, Op.GE, settings, 'mmr_subresonance_period_corr_min', '', 'sub_period_corr'
        )
        spread = abs(math.log(theory.period_ratio)) if np.isfinite(theory.period_ratio) and theory.period_ratio > 0 else NAN
        within = ledger.compare(
            'D |ln period ratio|',
            spread,
            Op.LE,
            math.log(settings.mmr_subresonance_period_ratio_max),
            '',
            'sub_period_ratio',
            setting='mmr_subresonance_period_ratio_max',
        )
        periods = bool(corr and within)
    return bool(rms and better and strongest and periods)


@dataclass
class Outcome:
    """What step 2 found: the winner (None when rejected; may be nodal), how the tie went, the theory verdict, the reason
    of every candidate and the flags."""

    winner: Optional[SubresonanceCandidate]
    tie_weak: bool
    theory_confirms: Optional[bool]
    reasons: Dict[str, str]
    flags: List[str]
    undetermined: bool = False  # more than one member passed E' and L with A not measured: nothing tells them apart


def _indistinguishable(candidates, reasons, e_mean, inc_max_rad, symmetric_centre, theory, settings, ledger) -> bool:
    """More than one planar member was left to test A by E' and L, A was not measured (A_small_swing), and they pass
    the record's test E and each its test B: nothing else could have told them apart."""
    left = [c for c in candidates if reasons.get(c.spec.key) == REASON_A_SMALL_SWING and not c.nodal]
    if len(left) < 2 or not _strong_enough(theory, settings, ledger):
        return False
    return sum(1 for c in left if _symmetric(c, e_mean, inc_max_rad, symmetric_centre, settings, ledger)) > 1


def evaluate(
    candidates: Sequence[SubresonanceCandidate],
    sigma0_phi0_coherence: float,
    e_mean: float,
    inc_max_rad: float,
    settings: AnalysisSettings,
    symmetric_centre: bool = True,
    theory: Optional[TheoryCheck] = None,
    ledger: Optional[Ledger] = None,
    small_swing: bool = False,
) -> Outcome:
    """Step 2 with every reason (`decide` is its short form): E', L and A on every member kept by step 1, C among
    those passing all three, B and E on the winner, D (`theory_confirms`) on the accepted one. With `small_swing` (sigma0
    swings below mmr_a_response_min_amp_deg in the window) A is not measured; when more than one member that passed
    E' and L then also passes B (and E holds), only A could have told them apart: the outcome is undetermined (no
    winner, flag subresonance_undetermined)."""
    ledger = ledger if ledger is not None else Ledger.from_settings(settings)
    reasons: Dict[str, str] = {}
    passing = []
    for c in candidates:
        if c.reason in SCREEN_REASONS or not np.isfinite(c.centre_range_rad):
            reasons[c.spec.key] = c.reason or REASON_RANGE
            continue
        if not _strong_relative(c, settings, ledger):
            reasons[c.spec.key] = REASON_E_STRENGTH
            continue
        reasons[c.spec.key] = _librates_itself(c, settings, ledger) or _passes_a(c, sigma0_phi0_coherence, settings, ledger, small_swing)
        if not reasons[c.spec.key]:
            passing.append(c)
    if not passing:
        if _indistinguishable(candidates, reasons, e_mean, inc_max_rad, symmetric_centre, theory, settings, ledger):
            return Outcome(None, False, None, reasons, [str(Flag.SUBRESONANCE_UNDETERMINED)], undetermined=True)
        return Outcome(None, False, None, reasons, [])
    best, tie_weak = _unique(passing, settings, reasons)
    flags = [FLAG_TIE] if any(reason == REASON_C for reason in reasons.values()) else []
    if not _symmetric(best, e_mean, inc_max_rad, symmetric_centre, settings, ledger):
        reasons[best.spec.key] = REASON_B
        return Outcome(None, tie_weak, None, reasons, flags)
    if not _strong_enough(theory, settings, ledger):
        reasons[best.spec.key] = REASON_E
        return Outcome(None, tie_weak, None, reasons, flags)
    confirms = theory_confirms(best, candidates, theory, settings, ledger) if theory is not None else None
    if confirms is False:
        flags.append(FLAG_THEORY_DISAGREES)
    reasons[best.spec.key] = REASON_NODAL if best.nodal else ''
    return Outcome(best, tie_weak, confirms, reasons, flags)


def decide(
    candidates: Sequence[SubresonanceCandidate],
    sigma0_phi0_coherence: float,
    e_mean: float,
    inc_max_rad: float,
    settings: AnalysisSettings,
    *,
    symmetric_centre: bool = True,
    theory: Optional[TheoryCheck] = None,
    beta_librates: bool = False,
    ledger: Optional[Ledger] = None,
    small_swing: bool = False,
) -> Tuple[Optional[SubresonanceCandidate], bool, bool]:
    """Step 2: tests E' (the member's d'Alembert factor against the strongest term, always), L (the member's own
    angle librates in the window), A (force law coherence and gain), B (centre near 0 or pi, skipped for 1:n,
    co-orbitals, high e or i), C (uniqueness; a tie is broken by lower degree, then larger coefficient), D (composite
    theory, optional), E (T0 test).
    Returns (accepted or None, tie decided by degree only, undetermined).

    A candidate whose `reason` is a step-1 failure ('range', 'rate', 'pass') is not considered; one without a
    `strength_rel` (a bare SubresonanceCandidate) passes E', one without `member_tp` passes L. B is skipped when
    `symmetric_centre` is False (1:n, co-orbital: the caller knows the resonance) or e_mean > _centre_e_max or
    inc_max > _centre_inc_max_deg. E runs with a measured `theory` (t0_ratio); D never rejects (it only raises the
    confidence: `evaluate().theory_confirms`). A tie that neither order nor coefficient resolves accepts the smallest
    range and also returns True (cap medium). A nodal winner is not accepted (None). With `beta_librates` every
    member's centre is bounded: nothing is decided (None, False, True); so it is when, with `small_swing`, several
    members are left that only the unmeasured test A could tell apart (`evaluate`)."""
    if beta_librates:
        return None, False, True
    outcome = evaluate(candidates, sigma0_phi0_coherence, e_mean, inc_max_rad, settings, symmetric_centre, theory, ledger, small_swing)
    if outcome.winner is None or outcome.winner.nodal:
        return None, outcome.tie_weak, outcome.undetermined
    return outcome.winner, outcome.tie_weak, False


# --------------------------------------------------------------------------------------------- window and numbers


def regular_stretch(tp_times_yr: np.ndarray, ratio: float) -> Optional[Tuple[float, float]]:
    """The longest stretch of turning points whose full periods t(TP_i+2) - t(TP_i) stay <= ratio x their median:
    (time of its first TP, time of its last TP) [yr]; None for fewer than three TPs. The slow cycles at the edges of a
    run are those near the separatrix of the modulated pendulum (SISTERS §1.3: the episodes of both prototypes end at a
    minimum of |P|), where no centre is stationary."""
    times = np.sort(np.asarray(tp_times_yr, dtype=float))
    if len(times) < 3:
        return None
    periods = times[2:] - times[:-2]
    regular = np.append(periods <= ratio * float(np.median(periods)), False)
    best, start = None, None
    for i, ok in enumerate(regular):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            stretch = (float(times[start]), float(times[i + 1]))  # cycles start..i-1 span TPs start..i+1
            if best is None or stretch[1] - stretch[0] > best[1] - best[0]:
                best = stretch
            start = None
    return best


def decisive_window(geometry: Geometry, settings: AnalysisSettings) -> Tuple[float, float]:
    """The scan window [yr]: the longest run of >= libration_min_turning_points TPs (accepted or not) inside the analysis
    window, trimmed to its regular cycles (`regular_stretch` with mmr_subresonance_cycle_period_ratio); else the analysis window."""
    start, end = geometry.window
    pieces = []
    for run in geometry.runs:
        if run.n_tp < settings.libration_min_turning_points or run.before_last_jump:
            continue
        times = np.asarray(run.tp_times_yr, dtype=float)
        stretch = regular_stretch(times[(times >= start) & (times <= end)], float(settings.mmr_subresonance_cycle_period_ratio))
        if stretch is not None and stretch[1] > stretch[0]:
            pieces.append(stretch)
    if not pieces:
        return float(start), float(end)
    return max(pieces, key=lambda p: p[1] - p[0])


def phase_coherence(
    t: np.ndarray, angle: np.ndarray, adot: np.ndarray, window: Tuple[float, float], length_yr: float, settings: AnalysisSettings
) -> Tuple[float, float, int]:
    """(phi0 coherence |<e^{i phi0}>|, median R^2, number of windows) of a_dot = R sin(angle - phi0) + C fitted in sliding
    windows of `length_yr` (step mmr_force_window_step) inside `window`; nan with fewer than mmr_force_min_windows."""
    inside = samples_between(t, *window)
    if not (np.isfinite(length_yr) and length_yr > 0) or inside.stop - inside.start < 2:
        return NAN, NAN, 0
    windows = force_windows(t[inside], angle[inside], adot[inside], length_yr, settings.mmr_force_window_step)
    if len(windows) < settings.mmr_force_min_windows:
        return NAN, NAN, len(windows)
    r2 = [w.r2 for w in windows if np.isfinite(w.r2)]
    return filtering.circular_resultant([w.phi0_rad for w in windows]), float(np.median(r2)) if r2 else NAN, len(windows)


def beta_librates(t: np.ndarray, beta_smooth: np.ndarray, window: Tuple[float, float]) -> bool:
    """beta = varpi_P - varpi librates in the window: it turns back at least twice (prominence pi/2) and never covers
    a full turn. Then every member's centre is bounded and the owner cannot be read from the centres (SISTERS §1.4.3)."""
    inside = filtering.window_mask(t, window)
    if np.count_nonzero(inside) < 3:
        return False
    turns = find_turning_points(t, beta_smooth, np.pi / 2, window)
    return len(turns) >= 2 and float(np.ptp(beta_smooth[inside])) < TWO_PI


def _cycles(geometry: Geometry, window: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mid times, full periods, half-amplitudes) of the cycles TP_i -> TP_i+2 inside the window."""
    tp = geometry.turning_points
    times, values = np.asarray(tp.times_yr, dtype=float), np.asarray(tp.values_rad, dtype=float)
    inside = (times >= window[0]) & (times <= window[1])
    times, values = times[inside], values[inside]
    if len(times) < 3:
        return np.empty(0), np.empty(0), np.empty(0)
    half = half_amplitude_values(values)
    return (times[2:] + times[:-2]) / 2, times[2:] - times[:-2], (half[1:] + half[:-1]) / 2


def _nodes(values: np.ndarray, count: int) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    return np.array([0.5 * (lo + hi)]) if hi <= lo or count < 2 else np.linspace(lo, hi, count)


def _interpolate(table: np.ndarray, x_nodes: np.ndarray, y_nodes: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of table[ix, iy, :] at the points (x, y) (constant along a single-node axis)."""

    def weights(nodes, values):
        if len(nodes) == 1:
            return np.zeros(len(values), dtype=int), np.zeros(len(values))
        i = np.clip(np.searchsorted(nodes, values) - 1, 0, len(nodes) - 2)
        return i, np.clip((values - nodes[i]) / (nodes[i + 1] - nodes[i]), 0.0, 1.0)

    ix, wx = weights(x_nodes, x)
    iy, wy = weights(y_nodes, y)
    jx, jy = np.minimum(ix + 1, len(x_nodes) - 1), np.minimum(iy + 1, len(y_nodes) - 1)
    wx, wy = wx[:, None], wy[:, None]
    return table[ix, iy] * (1 - wx) * (1 - wy) + table[jx, iy] * wx * (1 - wy) + table[ix, jy] * (1 - wx) * wy + table[jx, jy] * wx * wy


@dataclass
class _Theory:
    """The composite theory over the window (two-body): its coefficients and the check of the predicted centre and
    periods."""

    j_values: np.ndarray
    mean_coefficients: np.ndarray  # C_j averaged over the window
    check: TheoryCheck
    cache: str


def _coefficient_track(
    scan: '_Scan', e_g: np.ndarray, ep_g: np.ndarray, a_p: float, mass: float
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """(j values, C_j(t) on the grid or None below round-off, cache key): C_j / (e^|q-j| e_P^|j|) interpolated in
    (e, e_P) between table nodes. A member below round-off at any node is set to 0 (dividing its noise by e_P^|j| would
    blow it up); None when no member stands above round-off (`multiplet.resolved`)."""
    spec = scan.series.resonance
    k_p, k = spec.planet_coefficients[0], spec.body_coefficient
    q = -(k_p + k)
    js = np.arange(min(0, q) - THEORY_J_MARGIN, max(0, q) + THEORY_J_MARGIN + 1)
    powers_e, powers_p = np.abs(q - js), np.abs(js)
    e_nodes, p_nodes = _nodes(e_g, THEORY_E_NODES), _nodes(ep_g, THEORY_EP_NODES)
    raw = np.empty((len(e_nodes), len(p_nodes), len(js)), dtype=complex)
    key = ''
    for i, e_node in enumerate(e_nodes):
        for j, p_node in enumerate(p_nodes):
            raw[i, j], key = multiplet.cached_coefficients(spec.key, k_p, k, scan.context.a_mean_au, a_p, e_node, p_node, mass, js)
    if not multiplet.resolved(raw, mass, scan.context.a_mean_au, a_p):
        return js, None, key
    usable = np.all(np.abs(raw) > multiplet.round_off_floor(mass, scan.context.a_mean_au, a_p), axis=(0, 1))
    scale = e_nodes[:, None, None] ** powers_e * p_nodes[None, :, None] ** powers_p
    table = np.where(usable, raw / scale, 0.0)
    coefficients = _interpolate(table, e_nodes, p_nodes, e_g, ep_g) * (e_g[:, None] ** powers_e * ep_g[:, None] ** powers_p)
    return js, coefficients, key


def _periods(scan: '_Scan', t_grid: np.ndarray, t0_track: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """(mid times, observed, predicted T0(t) F(A) periods, correlation, median ratio, cycles used)."""
    mids, periods, amplitudes = _cycles(scan.geometry, scan.window)
    pred = np.interp(mids, t_grid, t0_track) * amplitude_factor(amplitudes) if len(mids) else np.empty(0)
    good = np.isfinite(pred) & np.isfinite(periods) & (pred > 0)
    corr = ratio = NAN
    if np.count_nonzero(good) >= MIN_PERIOD_CYCLES:
        corr = correlation(periods[good], pred[good])
    if np.any(good):
        ratio = float(np.median(periods[good] / pred[good]))
    return mids, periods, pred, corr, ratio, int(np.count_nonzero(good))


def composite_theory(scan: '_Scan') -> Tuple[Optional[_Theory], str]:
    """Composite multiplet theory on a coarse grid of the window (two-body, the planet saved): c0 = pi - arg P(beta(t)),
    P with C_j(e(t), e_P(t)) by the averaging, against the observed centre (one constant offset), and the cycle
    periods against T0(t) F(A). Returns (theory or None, why not: '' | FLAG_THEORY_CROSSING (the orbits can
    intersect, the averaging is singular) | FLAG_THEORY_ROUND_OFF (the coefficients are below round-off) |
    FLAG_THEORY_RANGE (e outside (0, 1) or an empty window))."""
    spec, t = scan.series.resonance, scan.t
    planet = spec.planets[0]
    if np.count_nonzero(scan.inside) < 3:
        return None, FLAG_THEORY_RANGE
    inside = np.flatnonzero(scan.inside)
    step = max(1, int(round(scan.centre_window_yr / THEORY_GRID_PER_WINDOW / np.median(np.diff(t)))), -(-len(inside) // THEORY_MAX_POINTS))
    index = inside[::step]
    e_g = scan.smooth_series(np.asarray(scan.series.e, float))[index]
    ep_g = scan.smooth_series(np.asarray(scan.series.planets[planet].e, float))[index]
    if not (np.all((e_g > 0) & (e_g < 1)) and np.all((ep_g > 0) & (ep_g < 1))):
        return None, FLAG_THEORY_RANGE
    a_p = float(np.mean(scan.series.planets[planet].a))
    if planet in scan.context.crosses or multiplet.crossing(scan.context.a_mean_au, float(np.max(e_g)), a_p, float(np.max(ep_g))):
        return None, FLAG_THEORY_CROSSING
    mass = scan.context.planet_mass.get(planet, const.PLANETS_MASS.get(planet, NAN))
    js, coefficients, key = _coefficient_track(scan, e_g, ep_g, a_p, mass)
    if coefficients is None:
        return None, FLAG_THEORY_ROUND_OFF
    beta = scan.smooth(False, f'varpi:{planet}')[index] - scan.smooth(False, 'varpi')[index]
    strength = multiplet.composite(coefficients, js, beta)
    predicted = np.pi - np.angle(strength) + spec.planet_varpi_coefficients[0] * beta
    diff = wrap_pi(scan.centre0[index] - predicted)
    offset = float(np.angle(np.mean(np.exp(1j * diff))))
    rms = float(np.sqrt(np.mean(wrap_pi(diff - offset) ** 2)))
    t0_track = multiplet.small_period_yr(np.abs(strength), scan.context.a_mean_au, spec.body_coefficient)
    _, periods, _, corr, ratio, n_cycles = _periods(scan, t[index], t0_track)
    observed = float(np.median(periods)) if len(periods) >= MIN_PERIOD_CYCLES else float(scan.geometry.period_cycle_yr)
    check = TheoryCheck(
        composite_rms_rad=rms,
        composite_offset_rad=offset,
        sigma0_single_rms_rad=_single_rms(scan.centre0[scan.inside]),
        period_corr=corr,
        period_ratio=ratio,
        n_period_cycles=n_cycles,
        t0_ratio=float(np.min(t0_track)) / observed if np.isfinite(observed) and observed > 0 else NAN,
    )
    return _Theory(js, np.mean(coefficients, axis=0), check, key), ''


def _single_rms(track: np.ndarray) -> float:
    """RMS of a centre track about its circular mean [rad]: the hypothesis 'this angle owns the oscillation'."""
    if not len(track):
        return NAN
    mean = np.angle(np.mean(np.exp(1j * track)))
    return float(np.sqrt(np.mean(wrap_pi(track - mean) ** 2)))


def _symmetric_centre_applies(spec: ResonanceSpec) -> bool:
    """Test B is skipped for 1:n (|k_P| = 1 two-body: asymmetric centres) and co-orbitals (|k_P| = |k|)."""
    if spec.kind != ResonanceKind.TWO_BODY:
        return True
    k_p, k = abs(spec.planet_coefficients[0]), abs(spec.body_coefficient)
    return not (k_p == 1 or k_p == k)


def degree_max(spec: ResonanceSpec, settings: AnalysisSettings) -> int:
    """The largest degree of the members scanned and listed: the catalogue's largest order (MATRIX_2BODY_ORDER_MAX /
    MATRIX_3BODY_ORDER_MAX), or |q| when the angle itself is beyond it."""
    order_max = settings.matrix_2body_order_max if spec.kind == ResonanceKind.TWO_BODY else settings.matrix_3body_order_max
    return max(abs(slow_sum(spec)), int(order_max))


def _drift_support(scan: '_Scan', members: Sequence[SubresonanceMember], settings: AnalysisSettings, ledger: Ledger) -> bool:
    """T3: |c0_dot + Delta_dot_s| < max(t3_sigma x sigma_rate, t3_rate_frac x |c0_dot|) for some member."""
    rate, error = scan.centre0_rate, scan.centre0_err
    if not np.isfinite(rate):
        return False
    sigma = settings.mmr_subresonance_t3_sigma * error if np.isfinite(error) else 0.0
    tolerance = max(sigma, settings.mmr_subresonance_t3_rate_frac * abs(rate))
    best = min((abs(rate + m.offset_rate_rad_yr) for m in members if np.isfinite(m.offset_rate_rad_yr)), default=NAN)
    return bool(ledger.compare('T3 drift matches a member', best, Op.LT, tolerance, 'rad/yr', 'sub_drift_meas_rad_kyr'))


def _break_support(geometry: Geometry, members: Sequence[SubresonanceMember], settings: AnalysisSettings) -> bool:
    """T4: two consecutive breaks (slips, edges of TP-rich runs inside the record) spaced by 2 pi / |Delta_dot_s| within the
    relative tolerance mmr_subresonance_t3_rate_frac (a rate tolerance read as a period tolerance)."""
    start, end = geometry.window
    events = [s.time_yr for s in geometry.slips if not s.suspect]
    for run in geometry.runs:
        if run.n_tp >= settings.libration_min_turning_points:
            events += [t for t in (run.t_start_yr, run.t_end_yr) if start < t < end]
    if len(events) < 2:
        return False
    spacings = np.diff(np.sort(np.asarray(events, dtype=float)))
    for m in members:
        if np.isfinite(m.offset_rate_rad_yr) and m.offset_rate_rad_yr != 0:
            period = TWO_PI / abs(m.offset_rate_rad_yr)
            if np.any(np.abs(spacings / period - 1) < settings.mmr_subresonance_t3_rate_frac):
                return True
    return False


# --------------------------------------------------------------------------------------------- the scan


@dataclass
class _Scan:
    """Arrays shared by the steps of one scan (the record's grid)."""

    series: MMRSeries
    context: Context
    geometry: Geometry
    settings: AnalysisSettings
    t: np.ndarray
    window: Tuple[float, float]
    inside: np.ndarray
    centre0: np.ndarray
    centre_window_yr: float
    # frame (True = invariable plane) -> slow angles; the invariable frame also holds sigma0's change of frame
    # FRAME_SHIFT (`plane_shift`) when every planet of the angle is saved
    raw: Dict[bool, Dict[str, np.ndarray]]
    smoothed: Dict[Tuple[bool, str], np.ndarray] = field(default_factory=dict)  # (frame, name) -> its W_c low-pass, on demand
    centre0_rate: float = NAN
    centre0_err: float = NAN
    screen_index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    frame_mixed: bool = False  # nodal members scanned without sigma0's change of frame (a planet not saved)

    @classmethod
    def build(cls, series, prep, geometry, context, settings, any_nodal: bool) -> '_Scan':
        t = np.asarray(prep.t_yr, dtype=float)
        window = decisive_window(geometry, settings)
        centre0 = np.interp(t, np.asarray(geometry.centre.t_yr, float), np.asarray(geometry.centre.centre_rad, float))
        raw = {False: slow_angles(series, False)}
        frame_mixed = False
        if any_nodal:
            raw[True] = slow_angles(series, True)
            shift = plane_shift(series, raw)
            if shift is None:
                frame_mixed = True
            else:
                raw[True][FRAME_SHIFT] = shift
        scan = cls(
            series, context, geometry, settings, t, window, filtering.window_mask(t, window), centre0, geometry.centre.window_yr, raw
        )
        scan.frame_mixed = frame_mixed
        scan.centre0_rate, scan.centre0_err = _line(t[scan.inside], centre0[scan.inside])
        inside = np.flatnonzero(scan.inside)
        scan.screen_index = inside[:: max(1, -(-len(inside) // SCREEN_POINTS))]
        return scan

    def smooth_series(self, values: np.ndarray) -> np.ndarray:
        """The W_c low-pass of the centre (linear, so the smoothed offset of a member is the combination of these)."""
        return centre_track(np.zeros_like(self.t), values, self.t, self.centre_window_yr, self.settings.filter_order)

    def smooth(self, nodal: bool, name: str) -> np.ndarray:
        """The W_c low-pass of one slow angle, computed when first asked for (a planar scan never smooths the nodes)."""
        if (nodal, name) not in self.smoothed:
            self.smoothed[nodal, name] = self.smooth_series(self.raw[nodal][name])
        return self.smoothed[nodal, name]

    def offset(self, spec: ResonanceSpec, member: ResonanceSpec, nodal: bool, smoothed: bool = True) -> np.ndarray:
        """Delta_s of a member on the record's grid (its W_c low-pass with `smoothed`): ecliptic for a planar member,
        the invariable plane with sigma0's change of frame for a nodal one (`frame_coefficients`)."""
        coefficients = self.frame_coefficients(offset_coefficients(spec, member), nodal)
        if not smoothed:
            return _combine(coefficients, self.raw[nodal], len(self.t))
        return _combine(coefficients, {name: self.smooth(nodal, name) for name in coefficients}, len(self.t))

    def frame_coefficients(self, coefficients: Dict[str, int], nodal: bool) -> Dict[str, int]:
        """The coefficients of a member's offset in its frame: a nodal member adds sigma0's change of frame once."""
        if nodal and FRAME_SHIFT in self.raw.get(True, {}):
            return {**coefficients, FRAME_SHIFT: 1}
        return dict(coefficients)


def _measure_member(scan: _Scan, member: ResonanceSpec, j: int, nodal: bool) -> dict:
    """Step 1 numbers of one member over the scan window (on the window samples thinned to <= SCREEN_POINTS: the
    tracks are smooth on W_c, and the pass search is the cost of a scan)."""
    index = scan.screen_index
    offset = scan.offset(scan.series.resonance, member, nodal)
    track = scan.centre0[index] + offset[index]
    reason, span, rate, largest, tolerance = _screen_reason(track, scan.centre0[index], scan.t[index], scan.settings)
    mean = float(np.angle(np.mean(np.exp(1j * track)))) if len(track) else NAN
    offset_rate, _ = _line(scan.t[index], offset[index])
    return dict(
        member=member,
        j=j,
        nodal=nodal,
        reason=reason,
        range=span,
        rate=rate,
        max_pass=largest,
        tol=tolerance,
        mean=mean,
        offset_rate=offset_rate,
        single=_single_rms(track),
    )


def _split_by_strength(
    members: Sequence[Tuple[ResonanceSpec, int, bool]],
    factors: Sequence[float],
    strongest: float,
    settings: AnalysisSettings,
    ledger: Ledger,
) -> Tuple[List[Tuple[ResonanceSpec, int, bool]], List[Tuple[ResonanceSpec, int, bool]]]:
    """(members to measure, members test E' rejects before step 1): a member below mmr_subresonance_strength_rel_min
    can never be accepted, so its track is not built (the cost of a scan at high q with the nodes: 6V-41+0+35 has
    thousands of members). The ledger keeps one comparison, that of the strongest rejected member; the candidates
    keep the rejected planar members (rows of candidates.csv, reason E_strength), the nodal ones are only counted."""
    floor = settings.mmr_subresonance_strength_rel_min
    if not (np.isfinite(strongest) and strongest > 0) or floor <= 0:
        return list(members), []
    strength = [term_factor(slow_coefficients(m[0]), factors) / strongest for m in members]
    strong = [m for m, value in zip(members, strength) if not (np.isfinite(value) and value < floor)]
    weak = [(m, value) for m, value in zip(members, strength) if np.isfinite(value) and value < floor]
    if weak:
        closest = max(weak, key=lambda item: item[1])
        name = f"E' {closest[0][0].key} factor / strongest term (the strongest of {len(weak)} rejected)"
        ledger.compare(name, closest[1], Op.GE, floor, '', '', setting='mmr_subresonance_strength_rel_min')
    return strong, [m for m, _ in weak]


def _unmeasured(member: ResonanceSpec, j: int, nodal: bool) -> dict:
    """The step-1 numbers of a member test E' rejected before step 1 (all nan, reason E_strength)."""
    return dict(
        member=member,
        j=j,
        nodal=nodal,
        reason=REASON_E_STRENGTH,
        range=NAN,
        rate=NAN,
        max_pass=NAN,
        tol=NAN,
        mean=NAN,
        offset_rate=NAN,
        single=NAN,
    )


def window_swing(geometry: Geometry, window: Tuple[float, float]) -> float:
    """The largest half-amplitude of sigma0 between consecutive turning points inside `window` [rad]; nan with fewer
    than two there. The members share sigma0's fast oscillation, so this is the swing every member's force law sees."""
    times = np.asarray(geometry.turning_points.times_yr, dtype=float)
    values = np.asarray(geometry.turning_points.values_rad, dtype=float)
    inside = (times >= window[0]) & (times <= window[1])
    half = half_amplitude_values(values[inside])
    return float(np.max(half)) if len(half) else NAN


def _force_law(
    scan: _Scan, prep: Preprocessed, measured: List[dict], ledger: Optional[Ledger] = None
) -> Tuple[float, Dict[str, Tuple[float, float, int]], float, bool]:
    """(phi0 coherence of sigma0, {key: (coherence, median R^2, windows)}, sigma0's swing [rad], below the floor) for
    the members kept by step 1 and the best mmr_subresonance_top_n by centre range; the same response series and
    windows as sigma0 (`response_series`). Every coherence is nan (test A not measured, `swing_gate`) when the swing
    (`window_swing`) is below mmr_a_response_min_amp_deg, as for the force law of the row itself
    (`response.a_response_check`): a sine and a line differ by < 10 % there, and any member's phase is coherent with
    the identity sigma_dot = c (a - a*) alone."""
    settings = scan.settings
    ledger = ledger if ledger is not None else Ledger.from_settings(settings)
    period = scan.geometry.period_cycle_yr
    angle, _, _, adot, cutoff = response_series(prep, period, settings)
    length = settings.mmr_force_window_cycles * period if np.isfinite(period) else NAN
    sigma0, _, _ = phase_coherence(scan.t, angle, adot, scan.window, length, settings)
    laws = {}
    for rank, m in enumerate(measured, start=1):
        if m['reason'] == REASON_E_STRENGTH or (m['reason'] and rank > settings.mmr_subresonance_top_n):
            continue
        offset = scan.offset(scan.series.resonance, m['member'], m['nodal'], smoothed=False)
        filtered = filtering.lowpass_or_none(scan.t, offset, cutoff, settings)
        offset = offset if filtered is None else filtered
        laws[m['member'].key] = phase_coherence(scan.t, angle + offset, adot, scan.window, length, settings)
    swing = window_swing(scan.geometry, scan.window)
    sigma0, laws, small = swing_gate(sigma0, laws, swing, settings, ledger)
    return sigma0, laws, swing, small


def swing_gate(
    sigma0: float, laws: Dict[str, Tuple[float, float, int]], swing_rad: float, settings: AnalysisSettings, ledger: Ledger
) -> Tuple[float, Dict[str, Tuple[float, float, int]], bool]:
    """(sigma0's coherence, the members' laws, small swing): every coherence nan when the swing is below
    mmr_a_response_min_amp_deg (R^2 and the window counts stay); unchanged without a swing (no turning points)."""
    if not np.isfinite(swing_rad):
        return sigma0, laws, False
    floor = settings.mmr_a_response_min_amp_deg
    name = 'A sigma0 swing in the scan window'
    if ledger.compare(name, np.degrees(swing_rad), Op.GE, floor, 'deg', 'sub_swing_deg', setting='mmr_a_response_min_amp_deg'):
        return sigma0, laws, False
    return NAN, {key: (NAN, r2, n_windows) for key, (_, r2, n_windows) in laws.items()}, True


@dataclass(frozen=True)
class MemberLibration:
    """Test L numbers of one member (`member_libration`); n_tp < 0: the member's angle could not be built."""

    n_tp: int = -1  # turning points of the member's own angle in the scan window
    n_episodes: int = 0  # its librating episodes there
    reject: str = ''  # why its longest run is not a librating episode (empty when one is, or without a run)
    window: Tuple[float, float] = (NAN, NAN)  # the longest librating episode (first TP, last TP) [yr]

    @property
    def librates(self) -> bool:
        return self.n_episodes > 0


def member_libration(scan: _Scan, prep: Preprocessed, member: ResonanceSpec, nodal: bool) -> MemberLibration:
    """Test L: does the member's own angle librate in the scan window (n_tp < 0: the angle could not be built).

    The angle is the ripple-cleaned sigma0 plus Delta_s, low-passed at sigma0's cutoff P_c (sigma0_f + Delta_s,f: the
    filter is linear, and the members of one multiplet share the lambda part the ripple removal took). Its turning
    points are taken at the prominences sigma0's geometry used (the record's, and each stretch of turning points its
    own: `TurningPoints.local`), its runs classified by `geometry.find_episodes`
    (>= libration_min_turning_points TPs, a stationary or modulated centre, a closed cycle), and an accepted run
    counts when the member's own row would count it (`decide.librating_episodes`: its period not below the period
    floor, its centre c0 + smooth(Delta_s) not completing a revolution inside it). Nothing new is tuned: these are the
    rules every row's episodes pass.

    sigma0's prominence and not the member's own fast residual: L asks whether the oscillation found in sigma0 is the
    member's, and a member whose swing stays below the prominence that oscillation was found with does not carry it
    (a conservative reading: such a member is not shown to circulate either, only not shown to own sigma0's cycles).
    A nodal member is sigma0 (ecliptic) plus Delta_s from the invariable plane, a hybrid angle; nodal winners are
    never accepted (not_measured), their L numbers only rank them against planar members."""
    t, settings = scan.t, scan.settings
    offset = scan.offset(scan.series.resonance, member, nodal, smoothed=False)
    offset = filtering.lowpass_or_none(t, offset, prep.cutoff.cutoff_yr, settings)
    if offset is None:
        return MemberLibration()
    angle = np.asarray(prep.angle_f, dtype=float) + offset
    if not np.all(np.isfinite(angle)):
        return MemberLibration()
    sigma0 = scan.geometry.turning_points
    turns = find_turning_points(t, angle, sigma0.prominence_rad, scan.window, local=sigma0.local)
    runs = find_episodes(t, turns, scan.window, settings)
    accepted = [run for run in runs if run.accepted]
    floor, _ = decide_module.period_floor(scan.geometry, settings)
    index = scan.screen_index
    track = scan.centre0[index] + scan.offset(scan.series.resonance, member, nodal)[index]
    centre_turns = find_turning_points(t[index], track, np.radians(settings.libration_centre_tp_prominence_deg), scan.window)
    centre = SimpleNamespace(t_yr=t[index], centre_rad=track, passes=monotone_passes(t[index], track, centre_turns, scan.window))
    own = SimpleNamespace(episodes=accepted, centre=centre)
    # `decide.librating_episodes` on the member's runs: the floor of sigma0's geometry, its centre plus Delta_s
    librating = librating_episodes_with(own, lambda episode: floor)
    if librating:
        longest = max(librating, key=lambda run: run.t_end_yr - run.t_start_yr)
        return MemberLibration(len(turns), len(librating), '', (longest.t_start_yr, longest.t_end_yr))
    longest = max(accepted or runs, key=lambda run: run.n_tp, default=None)
    if longest is None:
        reject = ''
    elif not longest.accepted:
        reject = str(longest.reject_reason)
    else:
        reject = REJECT_PERIOD if longest.period_med_yr < floor else REJECT_CENTRE_TURN
    return MemberLibration(len(turns), 0, reject)


def _candidates(
    measured: List[dict],
    laws: Dict[str, Tuple[float, float, int]],
    coefficient_abs: Dict[int, float],
    factors: Sequence[float],
    strongest: float,
    own: Optional[Dict[str, MemberLibration]] = None,
) -> List[SubresonanceMember]:
    """The members as `SubresonanceMember` rows, ranked as `measured`; `own` holds the test-L numbers
    (`member_libration`) of the members step 1 kept."""
    largest = max(coefficient_abs.values(), default=0.0)
    own = own or {}
    result = []
    for rank, m in enumerate(measured, start=1):
        coherence, r2, n_windows = laws.get(m['member'].key, (NAN, NAN, 0))
        libration = own.get(m['member'].key, MemberLibration())
        c_abs = coefficient_abs.get(m['j'], NAN) if not m['nodal'] else NAN
        factor = term_factor(slow_coefficients(m['member']), factors)
        strength = factor / strongest if np.isfinite(strongest) and strongest > 0 else NAN
        result.append(
            SubresonanceMember(
                spec=m['member'],
                j=m['j'],
                nodal=m['nodal'],
                centre_range_rad=m['range'],
                centre_rate_rad_yr=m['rate'],
                centre_max_pass_rad=m['max_pass'],
                centre_mean_rad=m['mean'],
                phi0_coherence=coherence,
                coefficient_rel=c_abs / largest if largest > 0 and np.isfinite(c_abs) else NAN,
                composite_rms_rad=NAN,
                period_corr=NAN,
                period_ratio=NAN,
                rank=rank,
                accepted=False,
                reason=m['reason'],
                degree=member_degree(m['member']),
                strength_rel=float(strength) if np.isfinite(strength) else NAN,
                kept=not m['reason'],
                rate_tolerance_rad_yr=m['tol'],
                centre_distance_rad=_distance_to_symmetric(m['mean']),
                force_r2_median=r2,
                n_force_windows=n_windows,
                single_rms_rad=m['single'],
                coefficient_abs=c_abs,
                offset_rate_rad_yr=m['offset_rate'],
                member_tp=libration.n_tp,
                member_episodes=libration.n_episodes,
                member_reject=libration.reject,
                member_t0_yr=libration.window[0],
                member_t1_yr=libration.window[1],
            )
        )
    return result


# --------------------------------------------------------------------------------------------- the ranked list


def centre_statistics(
    t_yr: np.ndarray, centre0: np.ndarray, angles: np.ndarray, offsets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre rate [rad/yr], E, slow rate Delta_dot_s [rad/yr]) of the tracks c_s = c0 + sum_k d_sk angle_k for every row
    d_s of `offsets` (M x K) on the samples t_yr: `angles` (K x N) are the smoothed slow angles, `centre0` (N) the
    catalogue centre. E = 1 - var(c_s) / var(c0) is the share of the catalogue centre's motion the member's slow angle
    explains (1: the member's centre stands, 0: as sigma0's own, < 0: it moves more). Linear in d, so every member
    follows from the covariances and the least-squares slopes of c0 and the K angles, without a track each."""
    t = np.asarray(t_yr, dtype=float)
    rows = np.vstack([np.asarray(centre0, dtype=float)[None, :], np.asarray(angles, dtype=float).reshape(-1, len(t))])
    offsets = np.asarray(offsets, dtype=float).reshape(-1, rows.shape[0] - 1)
    if len(t) < 3:
        nan = np.full(len(offsets), NAN)
        return nan, nan, nan
    design = np.column_stack([t - t[0], np.ones(len(t))])
    slopes = np.linalg.lstsq(design, rows.T, rcond=None)[0][0]
    centred = rows - rows.mean(axis=1, keepdims=True)
    covariance = centred @ centred.T / len(t)
    d = np.column_stack([np.ones(len(offsets)), offsets])
    variance = np.maximum(np.einsum('mk,kl,ml->m', d, covariance, d), 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        explained = 1.0 - variance / covariance[0, 0] if covariance[0, 0] > 0 else np.full(len(d), NAN)
    return d @ slopes, explained, offsets @ slopes[1:]


def fast_full_swing(geometry: Geometry, window: Tuple[float, float]) -> float:
    """A: the median full swing |sigma(TP_i+1) - sigma(TP_i)| of sigma0 between consecutive turning points inside
    `window` [rad] (the fast oscillation every member shares); nan with fewer than two turning points there."""
    times = np.asarray(geometry.turning_points.times_yr, dtype=float)
    values = np.asarray(geometry.turning_points.values_rad, dtype=float)
    inside = (times >= window[0]) & (times <= window[1])
    swings = np.abs(np.diff(values[inside]))
    return float(np.median(swings)) if len(swings) else NAN


def table_order(explained: Sequence[float], width: Sequence[float], weak: Optional[Sequence[bool]] = None) -> List[int]:
    """The order of the ranked list: the members test E' lets through first (`weak` False), each group by E descending
    (nan last), ties by the smaller W."""
    explained, width = np.asarray(explained, dtype=float), np.asarray(width, dtype=float)
    weak = np.zeros(len(explained), dtype=bool) if weak is None else np.asarray(weak, dtype=bool)
    key_e = np.where(np.isfinite(explained), -explained, np.inf)
    key_w = np.where(np.isfinite(width), width, np.inf)
    return [int(i) for i in np.lexsort((key_w, key_e, weak))]


def too_weak(strength_rel: Sequence[float], settings: AnalysisSettings) -> np.ndarray:
    """Members test E' rejects (factor < mmr_subresonance_strength_rel_min x the strongest; a nan factor is not weak):
    the list puts them after the others, since such a member cannot set the oscillation they all share."""
    strength = np.asarray(strength_rel, dtype=float)
    with np.errstate(invalid='ignore'):
        return np.isfinite(strength) & (strength < settings.mmr_subresonance_strength_rel_min)


def _frame_angles(scan: _Scan, nodal: bool, names: Sequence[str]) -> np.ndarray:
    """The smoothed slow angles `names` on the step-1 samples, in the frame of the members (`_Scan.frame_coefficients`)."""
    return np.vstack([scan.smooth(nodal, name)[scan.screen_index] for name in names]) if names else np.empty((0, len(scan.screen_index)))


def rank_members(
    scan: _Scan, factors: Sequence[float], strongest: float, nodal: bool, caveat: str = ''
) -> Tuple[List[ListedMember], int, Optional[ListedMember]]:
    """(the best mmr_subresonance_top_n members by `table_order`, members ranked, the catalogue angle's own row):
    every d'Alembert-valid member up to `degree_max` (and sigma +- 2 m omega with the nodes), the decision's members
    included (nodal ones with `nodal`, in the invariable frame). Display only (S2 of W20): E and the slow rate from
    `centre_statistics` for every member, W and A/W from the track of the listed ones, the members test E' rejects after
    the others (`table_order`) unless the d'Alembert factors are unreliable (`caveat`, `strength_caveat`: then by E
    and W only, the strength stays a column; W20 round C, review B #11: 101703's -1+1+3, E 0.92, behind +0-1+4, E
    0.81); nothing here decides."""
    spec, settings = scan.series.resonance, scan.settings
    index = scan.screen_index
    t, centre0 = scan.t[index], scan.centre0[index]
    if len(t) < 3:
        return [], 0, None
    span = float(t[-1] - t[0])
    swing = fast_full_swing(scan.geometry, scan.window)
    width0 = float(np.ptp(centre0))
    own_strength = term_factor(slow_coefficients(spec), factors) / strongest if np.isfinite(strongest) and strongest > 0 else NAN
    catalogue = _listed(spec.key, member_degree(spec), is_nodal(spec), 0.0, own_strength, 0.0, swing, width0, width0, _line(t, centre0)[0])
    own = np.array(slow_coefficients(spec))
    n = len(spec.planets)
    harmonics = settings.mmr_subresonance_omega_harmonics
    vectors = np.array(list(member_vectors(spec, degree_max(spec, settings), nodal, harmonics)), dtype=int).reshape(-1, len(own))
    if not len(vectors) or settings.mmr_subresonance_top_n <= 0:
        return [], len(vectors), catalogue
    member_nodal = np.any(vectors[:, n + 1 :] != 0, axis=1)
    names = _slow_names(spec)
    rate, explained, slow_rate = (np.full(len(vectors), NAN) for _ in range(3))
    for frame in (False, True):
        rows = np.flatnonzero(member_nodal == frame)
        if not len(rows):
            continue
        offsets = [scan.frame_coefficients({names[k]: int(v) for k, v in enumerate(vector) if v}, frame) for vector in vectors[rows] - own]
        used = sorted({name for offset in offsets for name in offset})
        matrix = np.array([[offset.get(name, 0) for name in used] for offset in offsets], dtype=float).reshape(len(rows), len(used))
        rate[rows], explained[rows], slow_rate[rows] = centre_statistics(t, centre0, _frame_angles(scan, frame, used), matrix)
    with np.errstate(invalid='ignore', divide='ignore'):
        strength = np.prod(np.where(vectors != 0, np.asarray(factors, dtype=float) ** np.abs(vectors), 1.0), axis=1) / strongest
    top_n = settings.mmr_subresonance_top_n
    weak = too_weak(strength, settings) if not caveat else np.zeros(len(vectors), dtype=bool)
    first = table_order(explained, np.zeros(len(vectors)), weak)
    edge = first[min(top_n, len(first)) - 1]
    head = set(first[:top_n])
    # W is measured on the listed members only; members tied with the last one on E (and the group) are measured too
    chosen = [i for i in first if i in head or (weak[i] == weak[edge] and explained[i] == explained[edge])]
    listed = []
    for i in chosen:
        track = centre0 + scan.offset(spec, member_spec(spec, vectors[i]), bool(member_nodal[i]))[index]
        width = float(np.ptp(track))
        cycles = span * abs(slow_rate[i]) / TWO_PI if np.isfinite(slow_rate[i]) else NAN
        listed.append(
            _listed(
                member_key(spec, vectors[i]),
                int(np.abs(vectors[i]).sum()),
                bool(member_nodal[i]),
                float(explained[i]),
                float(strength[i]),
                cycles,
                swing,
                width,
                width0,
                float(rate[i]),
            )
        )
    order = table_order([m.explained for m in listed], [m.width_rad for m in listed], weak[chosen])
    return [listed[i] for i in order][:top_n], len(vectors), catalogue


def _listed(key, degree, nodal, explained, strength, cycles, swing, width, width0, rate) -> ListedMember:
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = width / width0 if width0 > 0 else NAN
        swing_ratio = swing / width if width > 0 else NAN
    return ListedMember(
        key=key,
        degree=degree,
        nodal=nodal,
        rate_rad_yr=float(rate),
        strength_rel=float(strength),
        width_rad=float(width),
        width_ratio=float(ratio),
        explained=float(explained),
        slow_cycles=float(cycles),
        swing_ratio=float(swing_ratio),
    )


def potential_row(scan: _Scan) -> Tuple[float, float]:
    """(W [rad], E) of the two-body model 'sigma0 librates about the minimum c(t) of the averaged potential': the
    residual centre c0 - c(t) over the scan window, c(t) the well track of the run that covers the most of the window
    (`geometry.Episode.well`, evaluated only for a run whose centre moves); (nan, nan) without one. Display only."""
    index = scan.screen_index
    t, centre0 = scan.t[index], scan.centre0[index]
    best, best_count = None, 0
    for run in scan.geometry.runs:
        well = getattr(run, 'well', None)
        if well is None or not well.available:
            continue
        covered = (t >= well.t_yr[0]) & (t <= well.t_yr[-1])
        if np.count_nonzero(covered) > best_count:
            best, best_count = (well, covered), int(np.count_nonzero(covered))
    if best is None or best_count < 3:
        return NAN, NAN
    well, covered = best
    residual = centre0[covered] - well.at(t[covered])
    base = float(np.var(centre0[covered]))
    return float(np.ptp(residual)), (1.0 - float(np.var(residual)) / base if base > 0 else NAN)


def strength_caveat(series: MMRSeries, context: Context, factors: Sequence[float], settings: AnalysisSettings) -> str:
    """Why the d'Alembert factors of the list are not reliable ('' when they are): 'e' above
    mmr_subresonance_dalembert_e_max, 'i' (to the invariable plane) above mmr_subresonance_dalembert_inc_max_deg,
    'crossing' when the orbits of a planet of the angle and the body cross (the expansion in e does not converge)."""
    spec = series.resonance
    words = []
    if np.isfinite(context.e_med) and context.e_med > settings.mmr_subresonance_dalembert_e_max:
        words.append('e')
    sin_half = factors[len(spec.planets) + 1]
    if np.isfinite(sin_half) and sin_half > np.sin(np.radians(settings.mmr_subresonance_dalembert_inc_max_deg) / 2):
        words.append('i')
    if any(planet in context.crosses for planet in spec.planets):
        words.append('crossing')
    return join_list(words)


def _planar_coefficients(spec: ResonanceSpec, theory: Optional[_Theory]) -> Tuple[List[int], Dict[int, float]]:
    """(planar j values 0..q, {j: |mean C_j|}) of a two-body angle with the theory; empty otherwise."""
    if theory is None:
        return [], {}
    planar = [int(j) for j in multiplet.planar_members(spec.planet_coefficients[0], spec.body_coefficient)]
    lookup = dict(zip(theory.j_values.tolist(), theory.mean_coefficients))
    return planar, {j: float(abs(lookup[j])) for j in planar}


def _outcome(
    scan: _Scan, candidates, sigma0: float, planets_saved: bool, theory: TheoryCheck, ledger: Ledger, small_swing: bool = False
) -> Tuple[Outcome, CheckValue, bool]:
    """(outcome, check, beta librates) of step 2 with the guards: planets missing, beta librating, members only an
    unmeasured test A could tell apart, a nodal winner."""
    spec = scan.series.resonance
    reasons = {c.spec.key: c.reason for c in candidates}
    if not planets_saved:
        reasons = {key: reason or REASON_PLANETS for key, reason in reasons.items()}
        return Outcome(None, False, None, reasons, [str(Flag.PLANETS_MISSING)]), CheckValue.NOT_MEASURED, False
    librating = False
    if spec.kind == ResonanceKind.TWO_BODY:
        planet = spec.planets[0]
        librating = beta_librates(scan.t, scan.smooth(False, f'varpi:{planet}') - scan.smooth(False, 'varpi'), scan.window)
    if librating:
        reasons = {key: reason or REASON_UNDETERMINED for key, reason in reasons.items()}
        return Outcome(None, False, None, reasons, [str(Flag.SUBRESONANCE_UNDETERMINED)]), CheckValue.NOT_MEASURED, True
    context = scan.context
    symmetric = _symmetric_centre_applies(spec)
    outcome = evaluate(candidates, sigma0, context.e_med, context.inc_max_rad, scan.settings, symmetric, theory, ledger, small_swing)
    if outcome.undetermined:
        return outcome, CheckValue.NOT_MEASURED, False
    if outcome.winner is not None and outcome.winner.nodal:
        outcome.flags.append(str(Flag.NODAL_SUBRESONANCE_CANDIDATE))
        outcome.winner = None
        return outcome, CheckValue.NOT_MEASURED, False
    return outcome, (CheckValue.FAIL if outcome.winner is not None else CheckValue.PASS), False


@dataclass
class _MultipletNumbers:
    leading_j: Optional[int] = None
    binomial_mode: int = 0
    binomial_ratio: float = NAN
    rouche: Optional[bool] = None
    winding: Optional[int] = None


def _multiplet_numbers(scan: _Scan, theory: Optional[_Theory], planar: List[int], owner_j: Optional[int]) -> _MultipletNumbers:
    """Leading member, binomial mode and weight, Rouche for the owner (else the leading member), winding number."""
    if theory is None or not planar:
        return _MultipletNumbers()
    spec, context = scan.series.resonance, scan.context
    q = -(sum(spec.planet_coefficients) + spec.body_coefficient)
    lookup = dict(zip(theory.j_values.tolist(), theory.mean_coefficients))
    planar_mean = np.array([lookup[j] for j in planar])
    leading = int(planar[int(np.argmax(np.abs(planar_mean)))])
    e_planet = context.planet_e_mean.get(spec.planets[0], NAN)
    ratio = multiplet.binomial_ratio(planar_mean, np.array(planar), q, context.e_med, e_planet)
    return _MultipletNumbers(
        leading_j=leading,
        binomial_mode=multiplet.binomial_mode(q, context.e_med, e_planet, ratio),
        binomial_ratio=ratio,
        rouche=multiplet.rouche_holds(theory.mean_coefficients, theory.j_values, leading if owner_j is None else owner_j),
        winding=multiplet.winding_number(theory.mean_coefficients, theory.j_values),
    )


def _finalise(candidates: List[SubresonanceMember], outcome: Outcome, theory: TheoryCheck) -> List[SubresonanceMember]:
    """Reasons and the accepted mark; the record-level theory numbers on the accepted (else the best) member."""
    final = []
    winner = outcome.winner
    for c in candidates:
        accepted = winner is not None and c.spec.key == winner.spec.key
        updates = dict(reason=outcome.reasons.get(c.spec.key, c.reason), accepted=accepted)
        if accepted or (winner is None and c.rank == 1):
            updates.update(composite_rms_rad=theory.composite_rms_rad, period_corr=theory.period_corr, period_ratio=theory.period_ratio)
        final.append(replace(c, **updates))
    return final


def _rates(scan: _Scan) -> Tuple[float, float]:
    """(member spacing [au], adiabaticity |varpi_dot - varpi_dot_P| / omega_lib) of a two-body angle."""
    spec, context = scan.series.resonance, scan.context
    if spec.kind != ResonanceKind.TWO_BODY:
        return NAN, NAN
    rate_body, rate_planet = context.body_varpi_rate_rad_yr, context.planet_varpi_rate_rad_yr.get(spec.planets[0], NAN)
    if not (np.isfinite(rate_body) and np.isfinite(rate_planet)):
        return NAN, NAN
    spacing = multiplet.member_spacing_au(context.a_mean_au, rate_body, rate_planet, spec.body_coefficient)
    period = scan.geometry.period_cycle_yr
    adiabatic = abs(rate_body - rate_planet) / (TWO_PI / period) if np.isfinite(period) and period > 0 else NAN
    return spacing, adiabatic


def run_subresonances(
    series: MMRSeries,
    prep: Preprocessed,
    geometry: Geometry,
    response: Response,
    context: Context,
    triggered: Tuple[SubresonanceTrigger, ...],
    settings: AnalysisSettings,
) -> SubresonanceResult:
    """The whole check for a triggered record; check pass when scanned and none accepted, fail when one is accepted,
    not_measured without the planets (flag planets_missing).

    Returns a `SubresonanceScan` (a SubresonanceResult). Not applicable to retrograde and 1:1 angles (`law_planet`). Never
    integrates: the members' angles are sigma0 + Delta_s from the saved elements. `response` is not read (the force
    law is refitted in the scan window for sigma0 and the members alike). The decision and the ranked list `listed`
    (display only) take the members up to `degree_max`, and with the nodal members sigma +- 2 m omega
    (`omega_vectors`)."""
    spec = series.resonance
    if spec.retrograde or law_planet(spec) is not None:
        return _empty(triggered, CheckValue.NOT_APPLICABLE, [])
    factors = strength_factors(series, context)
    nodal = nodal_applies(factors[len(spec.planets) + 1], settings)
    members = enumerate_subresonances(spec, degree_max(spec, settings), nodal, settings.mmr_subresonance_omega_harmonics)
    if not members:
        return _empty(triggered, CheckValue.NOT_APPLICABLE, [])
    n = len(spec.planets)
    strongest = strongest_factor(slow_sum(spec), factors[: n + 1], factors[n + 1 :])
    ledger = Ledger.from_settings(settings)
    planets_saved = all(p in series.planets for p in spec.planets)
    scan = _Scan.build(series, prep, geometry, context, settings, nodal)
    strong, weak = _split_by_strength(members, factors, strongest, settings, ledger)
    measured = sorted((_measure_member(scan, *m) for m in strong), key=lambda m: m['range'] if np.isfinite(m['range']) else np.inf)
    measured += [_unmeasured(*m) for m in weak if not m[2]]  # the weak nodal ones are only counted (`n_weak`)
    own = {m['member'].key: member_libration(scan, prep, m['member'], m['nodal']) for m in measured if not m['reason']}
    sigma0_coherence, laws, swing, small_swing = _force_law(scan, prep, measured, ledger)
    theory, theory_flag = None, ''
    needed = any(not m['reason'] for m in measured)  # D and E judge a kept member only (the cost of a scan)
    if spec.kind == ResonanceKind.TWO_BODY and planets_saved and settings.mmr_subresonance_theory and needed:
        theory, theory_flag = composite_theory(scan)
    planar, coefficient_abs = _planar_coefficients(spec, theory)
    candidates = _candidates(measured, laws, coefficient_abs, factors, strongest, own)
    check_theory = theory.check if theory is not None else TheoryCheck()
    outcome, check, librating = _outcome(scan, candidates, sigma0_coherence, planets_saved, check_theory, ledger, small_swing)
    if outcome.winner is not None:
        check_theory = replace(check_theory, single_rms_rad=_rival_single_rms(outcome.winner, candidates, check_theory))
    check_theory = replace(check_theory, confirms=outcome.theory_confirms)
    final = _finalise(candidates, outcome, check_theory)
    accepted = next((c for c in final if c.accepted), None)
    shown = accepted or (final[0] if final else None)
    numbers = _multiplet_numbers(scan, theory, planar, accepted.j if accepted is not None else None)
    flags = list(outcome.flags) + ([theory_flag] if theory_flag else []) + ([FLAG_FRAME_MIXED] if scan.frame_mixed else [])
    composite = bool(accepted is not None and numbers.rouche is False)
    if composite:
        flags.append(str(Flag.SUBRESONANCE_COMPOSITE))
    support = (('T3',) if _drift_support(scan, final, settings, ledger) else ()) + (
        ('T4',) if _break_support(geometry, final, settings) else ()
    )
    spacing, adiabatic = _rates(scan)
    caveat = strength_caveat(series, context, factors, settings)
    listed, n_listed, catalogue_row = rank_members(scan, factors, strongest, nodal, caveat)
    potential_width, potential_explained = potential_row(scan)
    return SubresonanceScan(
        triggers=tuple(triggered),
        support=support,
        candidates=final,
        accepted=accepted,
        check=check,
        composite=composite,
        undetermined=librating or outcome.undetermined,
        tie_by_order=bool(accepted is not None and outcome.tie_weak),
        theory_measured=bool(outcome.theory_confirms),
        sigma0_phi0_coherence=sigma0_coherence,
        composite_rms_rad=check_theory.composite_rms_rad,
        single_rms_rad=check_theory.single_rms_rad,
        binomial_mode=numbers.binomial_mode,
        drift_pred_rad_yr=-shown.offset_rate_rad_yr if shown is not None else NAN,
        drift_meas_rad_yr=scan.centre0_rate,
        plane_offset_rad=plane_offset(series, scan.raw) if True in scan.raw else NAN,
        theory_cache=theory.cache if theory is not None else '',
        flags=list(dict.fromkeys(flags)),
        window=scan.window,
        n_kept=sum(1 for c in final if c.kept),
        swing_rad=swing,
        centre0_range_rad=float(np.ptp(scan.centre0[scan.inside])) if np.any(scan.inside) else NAN,
        centre0_rate_rad_yr=scan.centre0_rate,
        centre0_rate_err_rad_yr=scan.centre0_err,
        leading_j=numbers.leading_j,
        binomial_ratio=numbers.binomial_ratio,
        rouche=numbers.rouche,
        winding=numbers.winding,
        theory=check_theory,
        spacing_au=spacing,
        adiabaticity=adiabatic,
        beta_librates=librating,
        strongest_factor=strongest,
        nodal_scanned=nodal,
        listed=listed,
        n_listed=n_listed,
        catalogue_row=catalogue_row,
        fast_swing_rad=fast_full_swing(geometry, scan.window),
        potential_width_rad=potential_width,
        potential_explained=potential_explained,
        strength_caveat=caveat,
        n_weak=len(weak),
        entries=list(ledger),
    )


def _empty(triggered, check: CheckValue, flags: List[str]) -> SubresonanceScan:
    return SubresonanceScan(
        triggers=tuple(triggered),
        support=(),
        candidates=[],
        accepted=None,
        check=check,
        composite=False,
        undetermined=False,
        tie_by_order=False,
        theory_measured=False,
        sigma0_phi0_coherence=NAN,
        composite_rms_rad=NAN,
        single_rms_rad=NAN,
        binomial_mode=0,
        drift_pred_rad_yr=NAN,
        drift_meas_rad_yr=NAN,
        plane_offset_rad=NAN,
        flags=list(flags),
    )


def summary(result: SubresonanceResult) -> str:
    """One line for logs and the figure: triggers, accepted member or the best, check."""
    best = result.accepted or min(result.candidates, key=lambda c: c.rank, default=None)
    name = best.spec.key if best is not None else '-'
    return f'{join_list(str(t) for t in result.triggers)} -> {name} ({result.check})'
