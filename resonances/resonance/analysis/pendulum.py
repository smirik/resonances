"""The pendulum and the energy of an angle relative to its separatrix (shared; SEPARATRIX §§1–4).

Model: sigma_dot = c (x - x*), x_dot = -V'(sigma) / c, hence sigma_ddot = -V'(sigma) and
E = sigma_dot^2 / 2 + V(sigma), with V(sigma) = sum_{n<=N} (a_n cos n sigma + b_n sin n sigma).
For an MMR x is a [au]; for a secular or ZLK angle a slow element. The normalised energy is
h = (E - V_min) / (V_max - V_min): h < 1 libration, h > 1 circulation. Angles in rad, times in yr.

Closed forms (M&D 1999, eqs. 8.50–8.58; scipy `ellipk(m)`/`ellipe(m)` take m = k^2):
  F(A) = (2/pi) K(sin^2(A/2)),  T0 = P / F(A),  omega0 = 2 pi / T0,
  libration period 4 K(h) / Omega, revolution period 2 K(1/h) / (Omega sqrt(h)),
  J / J_s = E(h) - (1 - h) K(h)  (-> pi h / 4 as h -> 0, -> 1 as h -> 1),
  r_pend(h) = min/mean rate over a revolution = (2/pi) sqrt((h - 1)/h) K(1/h).
These hold for one harmonic only. A fitted V with more harmonics gets quadratures over its deepest well
(`action_ratio_in_potential`, `period_in_potential`, `half_width_in_potential`): with
J(E) = (1/pi) int sqrt(2 (E - V)) d sigma over the libration loop, J_s = J at the well's own separatrix
energy (the lower of the two maxima that bound it). V = -cos s - 0.2 cos 2s at h = 0.5: J/J_s = 0.38417 where the
cosine formula gives 0.42361.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from scipy import integrate, optimize
from scipy.special import ellipe, ellipk

from resonances.resonance.analysis.filtering import circular_resultant
from resonances.resonance.analysis.filtering import r2 as filtering_r2
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue, join_list

NAN = float('nan')
TWO_PI = 2 * np.pi
MAD_TO_SIGMA = 1.4826  # robust std of a normal distribution from its median absolute deviation
PERMUTATION_METHOD = 'cyclic_shift'
# Flag of a record whose crossings all fall among cycles inside the model's separatrix zone (crossings, every cycle in
# the zone): a model-zone test, not a measurement of chaos; a column, never a status (DECISIONS 2026-10-15 evening).
# 'separatrix_chaos' before 2026-10-17 (`verdict.OLD_FLAGS` reads old files). Same spelling as
# verdict.Flag.EPISODES_IN_SEPARATRIX_ZONE.
EPISODES_IN_SEPARATRIX_ZONE = 'episodes_in_separatrix_zone'
# directions of a change of regime (`regime_transitions`)
EXIT = 'exit'  # a libration half-cycle followed by a slip or a turn
CAPTURE = 'capture'  # a slip or a turn followed by a libration half-cycle
GRID_POINTS = 720  # sampling of V(sigma) over one turn before the extremum is refined (0.5 deg)
# Per-window fits solve their normal equations when cond(Gram) stays below this (error ~ cond x 1e-16 <= 1e-10),
# else least squares on the window's samples
GRAM_CONDITION_MAX = 1e6


class CycleClass(StrEnum):
    DEEP = 'deep'  # below the zone the separatrix reaches (with margin)
    ZONE = 'zone'  # inside the zone
    ABOVE = 'above'  # revolution above the zone


class ZoneMethod(StrEnum):
    NUMERIC = 'numeric'
    EQ_ONLY = 'eq_only'
    ADIABATIC = 'adiabatic'


class SeparatrixMechanism(StrEnum):
    PULSATION = 'pulsation'  # crossings in the weak phase of Omega(t)
    LAYER = 'layer'  # adiabaticity lambda > 1/2
    KICK = 'kick'  # energy jump at a step of the slow variable
    TREND = 'trend'  # significant trend of J/J_s over cycles
    UNEXPLAINED = 'unexplained'
    NONE = ''


@dataclass(frozen=True)
class Potential:
    """V(sigma) fitted from sigma_dot^2 (V_E) or from the force (V_F); coefficients in rad^2/yr^2."""

    n_harmonics: int
    cos_coeffs: np.ndarray  # a_1..a_N
    sin_coeffs: np.ndarray  # b_1..b_N
    r2: float
    well_rad: float  # sigma_s, minimum of V
    top_rad: float  # maximum of V
    omega_rad_yr: float  # Omega = sqrt(V''(sigma_s))
    v_min: float
    v_max: float

    def value(self, sigma: np.ndarray) -> np.ndarray:
        """V(sigma) [rad^2/yr^2]."""
        return _harmonics(np.asarray(sigma, dtype=float), self.n_harmonics) @ _coefficients(self.cos_coeffs, self.sin_coeffs)

    def energy_h(self, sigma: np.ndarray, rate: np.ndarray) -> np.ndarray:
        """h = (rate^2/2 + V(sigma) - V_min) / (V_max - V_min) (dimensionless)."""
        rate = np.asarray(rate, dtype=float)
        return (rate**2 / 2 + self.value(sigma) - self.v_min) / (self.v_max - self.v_min)

    @property
    def depth(self) -> float:
        """V_max - V_min [rad^2/yr^2] (= 2 Omega^2 for a pendulum)."""
        return self.v_max - self.v_min


@dataclass(frozen=True)
class ModulatedFrame:
    """A pendulum whose strength and stable point move: sigma_ddot = -m(t) Omega_bar^2 sin(sigma - sigma_s(t)).

    `strength_rel_sq` is m(t) = Omega(t)^2 / Omega_bar^2 (Omega_bar the mean of Omega over the record, so that
    <J_s> = 8 Omega_bar / pi), `well_shift_rad` the motion of the stable point sigma_s(t) - <sigma_s> (unwrapped) and
    `well_shift_rate_rad_yr` its time derivative. Energies are measured in phi = sigma - well_shift. The fixed pendulum
    is m = 1, shift = 0 (`fixed_frame`); every function below takes frame=None for it.
    """

    t_yr: np.ndarray
    strength_rel_sq: np.ndarray
    well_shift_rad: np.ndarray
    well_shift_rate_rad_yr: np.ndarray

    def on(self, t_yr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(m, shift, shift rate) interpolated to `t_yr`."""
        t_yr = np.asarray(t_yr, dtype=float)
        return (
            np.interp(t_yr, self.t_yr, self.strength_rel_sq),
            np.interp(t_yr, self.t_yr, self.well_shift_rad),
            np.interp(t_yr, self.t_yr, self.well_shift_rate_rad_yr),
        )


def fixed_frame(t_yr: np.ndarray) -> ModulatedFrame:
    """The unmodulated pendulum on `t_yr` (m = 1, no motion of the stable point)."""
    t_yr = np.asarray(t_yr, dtype=float)
    return ModulatedFrame(t_yr, np.ones_like(t_yr), np.zeros_like(t_yr), np.zeros_like(t_yr))


def in_frame(
    t_yr: np.ndarray, angle: np.ndarray, rate: np.ndarray, frame: Optional[ModulatedFrame]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(phi, phi_dot, m) of the samples: the angle and its rate relative to the moving stable point, and m(t)."""
    angle, rate = np.asarray(angle, dtype=float), np.asarray(rate, dtype=float)
    if frame is None:
        return angle, rate, np.ones_like(angle)
    strength, shift, shift_rate = frame.on(t_yr)
    return angle - shift, rate - shift_rate, strength


@dataclass(frozen=True)
class CycleEnergy:
    """One cycle (libration) or one turn (revolution) and its energy.

    `h` is the energy over the separatrix energy at the time of the cycle (the instantaneous pendulum of the frame);
    `h_mean` over the mean separatrix energy 2 Omega_bar^2 (h m), and `action_ratio` is J/<J_s>, the action over the
    mean separatrix action (J/J_s(h) sqrt(m)): the zone of SPEC §4.6.1 is measured in these two. Without modulation
    m = 1 and the three agree with the fixed pendulum.
    """

    t_start_yr: float
    t_end_yr: float
    rotation: bool
    h: float
    action_ratio: float  # J/<J_s> (libration side), nan for a revolution
    period_yr: float
    period_law_ratio: float  # T_obs / T_pend(h)
    h_mean: float = float('nan')  # E / (2 Omega_bar^2): energy over the mean separatrix energy
    strength_rel_sq: float = 1.0  # m = Omega^2 / Omega_bar^2 averaged over the cycle

    @property
    def energy_over_mean(self) -> float:
        """h_mean, or h when the cycle carries no modulation."""
        return self.h_mean if np.isfinite(self.h_mean) else self.h


@dataclass(frozen=True)
class Zone:
    """The energies the pulsating separatrix reaches on the length of the record (SPEC §4.6.1)."""

    method: ZoneMethod
    j_lo: float  # smallest J0/<J_s> that crosses (libration side)
    h_lo: float  # h(J_lo)
    h_hi: float  # largest h0 on the rotation side that falls below the separatrix
    adiabatic_lo: float  # Omega_min / Omega_bar
    adiabatic_hi: float  # Omega_max / Omega_bar
    mismatch: bool  # numeric vs adiabatic J_lo differ by > tolerance at small lambda

    @property
    def width_h(self) -> float:
        return self.h_hi - self.h_lo


def has_zone_cycle(cycle_classes: Sequence) -> bool:
    """Some cycle lies in the separatrix zone (`EnergyAnalysis.cycle_classes`)."""
    return CycleClass.ZONE in [CycleClass(c) for c in cycle_classes]


@dataclass
class EnergyAnalysis:
    """Energy of the angle relative to its separatrix (columns `sep_*`)."""

    potential_e: Optional[Potential]  # V_E from sigma only
    potential_f: Optional[Potential]  # V_F from the force
    ve_vf_agree_fraction: float
    well_coherence: float  # |<e^{i sigma_s}>| over windows
    cycles: List[CycleEnergy] = field(default_factory=list)
    cycle_kinds: List[str] = field(default_factory=list)  # per cycle: 'episode' | 'diagnostic' | 'turn' (W3-G4)
    h_med: float = float('nan')
    h_p10: float = float('nan')
    h_p90: float = float('nan')
    sigma_h: float = float('nan')
    period_law_med: float = float('nan')
    period_law_p10: float = float('nan')
    period_law_p90: float = float('nan')
    # where the predicted period of the libration cycles comes from: 'full_well' (the averaged potential of a two-body
    # angle, `mmr.zone.full_well_laws`, for the swings that librate in it) or 'fitted' (V_E only); the median law of the
    # fitted V_E stays a column
    period_law_source: str = ''
    period_law_n_full_well: int = 0  # the cycles whose T_pred came from the full well
    period_law_fitted_med: float = float('nan')
    eps: Tuple[float, ...] = ()  # relative strengths of the modulating terms
    omega_min_rel: float = float('nan')  # Omega_min / Omega_bar
    omega_max_rel: float = float('nan')
    adiabaticity: float = float('nan')  # lambda = nu / Omega_bar
    overlap_s: float = float('nan')  # 2 (1 + sqrt(eps)) / lambda, descriptive
    eq_dk_over_k: float = float('nan')  # e^q modulation of the force (bounded oscillatory e only)
    e_trend_per_kyr: float = float('nan')
    zone: Optional[Zone] = None
    cycle_classes: List[CycleClass] = field(default_factory=list)
    crossing_times_yr: List[float] = field(default_factory=list)
    # the ordered directed changes of regime, before the times are merged (`regime_transitions`): (time, exit|capture)
    transitions: List[Tuple[float, str]] = field(default_factory=list)
    mechanism: SeparatrixMechanism = SeparatrixMechanism.NONE
    permutation_p: float = float('nan')
    sigma_only: CheckValue = CheckValue.NOT_MEASURED  # R0 sigma-only (geometry)
    with_slow: CheckValue = CheckValue.NOT_MEASURED  # R0 with a (physics: check_mmr_potential)
    # the 1:1 law (mmr coorbital.CoorbitalLaw) R0 with a was read from (`response.with_coorbital_potential`); None for
    # the harmonic V_E ~ V_F test. The figure prints R0 from it, so the number shown is the one that decided
    potential_law: Optional[Any] = None
    flags: List[str] = field(default_factory=list)
    # how the pulsation was measured and what it means (columns only: none of these changes a status)
    modulation_source: str = ''  # 'members' (force fit with the subresonances) | 'eccentricity' (e^q) | 'none'
    frame: str = ''  # the frame V_E was measured in: 'fixed' | 'members' | 'eccentricity' | 'centre'
    modulation_planet: str = ''  # the planet whose varpi term modulates the force (members)
    modulation_gain: float = float('nan')  # R^2 of the members' fit over the single harmonic
    modulation_period_yr: float = float('nan')  # 2 pi / nu
    omega_bar_rad_yr: float = float('nan')  # Omega_bar: the time scale of the zone integration
    well_wobble_rad: float = float('nan')  # half-range of the moving stable point sigma_s(t)
    zone_seconds: float = float('nan')  # cost of the numeric zone
    # crossings and every cycle inside the model's zone ('separatrix_chaos' before 2026-10-17)
    episodes_in_separatrix_zone: bool = False
    model_frame: Optional[ModulatedFrame] = None  # the frame of V_E (not a column; the filter guard re-reads energies in it)
    series_cutoff_yr: float = float('nan')  # sigma and a were smoothed at this cutoff for the energies
    # where the libration cycles' classes come from: 'pendulum' (the fitted V_E) or 'averaged_potential' (the energy h of
    # each turning point in a two-body angle's averaged well, `mmr.zone.averaged_classes`); h of those turning points
    zone_source: str = 'pendulum'
    h_avgpot: Tuple[float, ...] = ()

    @property
    def n_crossings(self) -> int:
        """Distinct boundary times between regimes: a slip between two librations is one time (an exit and a capture)."""
        return len(self.crossing_times_yr)

    @property
    def n_exits(self) -> int:
        """Libration half-cycles followed by a slip or a turn (`regime_transitions`)."""
        return sum(1 for _, kind in self.transitions if kind == EXIT)

    @property
    def n_captures(self) -> int:
        """Slips or turns followed by a libration half-cycle (`regime_transitions`)."""
        return sum(1 for _, kind in self.transitions if kind == CAPTURE)

    @property
    def n_outside(self) -> int:
        """Cycles outside the zone (deep or above)."""
        return sum(1 for c in self.cycle_classes if CycleClass(c) != CycleClass.ZONE)

    @property
    def turn_zone_widths(self) -> float:
        """(median h_mean of the turns - h_hi) / (h_hi - h_lo): how far above the zone a circulation runs, in zone
        widths (SPEC §5.4 R6); nan without turns or zone."""
        turns = [c.energy_over_mean for c in self.cycles if c.rotation and np.isfinite(c.energy_over_mean)]
        zone = self.zone
        if not turns or zone is None or not (np.isfinite(zone.width_h) and zone.width_h > 0):
            return NAN
        return float((np.median(turns) - zone.h_hi) / zone.width_h)

    def to_columns(self) -> dict:
        """CSV columns `sep_*` in CSV units (angles in deg, rates in rad/yr as named)."""
        pe, pf, zone = self.potential_e, self.potential_f, self.zone
        classes = [CycleClass(c) for c in self.cycle_classes]
        finite_p = np.isfinite(self.permutation_p)
        return {
            'sep_N': pe.n_harmonics if pe is not None else None,
            'sep_VE_R2': pe.r2 if pe is not None else NAN,
            'sep_VF_R2': pf.r2 if pf is not None else NAN,
            'sep_VE_VF_agree_fraction': self.ve_vf_agree_fraction,
            'sep_Omega_rad_yr': pe.omega_rad_yr if pe is not None else NAN,
            'sep_well_deg': _deg_0_360(pe.well_rad) if pe is not None else NAN,
            'sep_top_deg': _deg_0_360(pe.top_rad) if pe is not None else NAN,
            'sep_well_coherence': self.well_coherence,
            'sep_n_cycles': len(self.cycles),
            'sep_n_cycles_diagnostic': sum(1 for kind in self.cycle_kinds if kind == 'diagnostic'),
            'sep_h_med': self.h_med,
            'sep_h_p10': self.h_p10,
            'sep_h_p90': self.h_p90,
            'sep_sigma_h': self.sigma_h,
            'sep_period_law_med': self.period_law_med,
            'sep_period_law_p10': self.period_law_p10,
            'sep_period_law_p90': self.period_law_p90,
            'sep_period_law_source': self.period_law_source,
            'sep_period_law_n_full_well': self.period_law_n_full_well,
            'sep_period_law_fitted_med': self.period_law_fitted_med,
            'sep_eps': join_list(f'{value:.4g}' for value in self.eps),
            'sep_Omega_min_rel': self.omega_min_rel,
            'sep_Omega_max_rel': self.omega_max_rel,
            'sep_lambda': self.adiabaticity,
            'sep_overlap_s': self.overlap_s,
            'sep_eq_dK_over_K': self.eq_dk_over_k,
            'sep_e_trend_per_kyr': self.e_trend_per_kyr,
            'sep_zone_method': str(zone.method) if zone is not None else '',
            'sep_zone_J_lo': zone.j_lo if zone is not None else NAN,
            'sep_zone_h_lo': zone.h_lo if zone is not None else NAN,
            'sep_zone_h_hi': zone.h_hi if zone is not None else NAN,
            'sep_zone_adiabatic_lo': zone.adiabatic_lo if zone is not None else NAN,
            'sep_zone_adiabatic_hi': zone.adiabatic_hi if zone is not None else NAN,
            'sep_zone_mismatch': zone.mismatch if zone is not None else None,
            'sep_zone_source': self.zone_source,
            'sep_h_avgpot_med': float(np.median(self.h_avgpot)) if self.h_avgpot else NAN,
            'sep_h_avgpot_max': float(np.max(self.h_avgpot)) if self.h_avgpot else NAN,
            'sep_h_avgpot_n': len(self.h_avgpot),
            'sep_n_deep': classes.count(CycleClass.DEEP),
            'sep_n_zone': classes.count(CycleClass.ZONE),
            'sep_n_above': classes.count(CycleClass.ABOVE),
            'sep_n_crossings': self.n_crossings,
            'sep_n_exits': self.n_exits,
            'sep_n_captures': self.n_captures,
            'sep_crossing_times_yr': join_list(f'{t:.6g}' for t in self.crossing_times_yr),
            'sep_mechanism': str(self.mechanism),
            'sep_permutation_p': self.permutation_p,
            'sep_permutation_method': PERMUTATION_METHOD if finite_p else '',
            'sep_R0_sigma_only': str(self.sigma_only),
            'sep_R0_with_a': str(self.with_slow),
            'sep_flags': join_list(self.flags),
            'sep_modulation_source': self.modulation_source,
            'sep_frame': self.frame,
            'sep_modulation_planet': self.modulation_planet,
            'sep_modulation_gain': self.modulation_gain,
            'sep_modulation_period_yr': self.modulation_period_yr,
            'sep_Omega_bar_rad_yr': self.omega_bar_rad_yr,
            'sep_well_wobble_deg': float(np.degrees(self.well_wobble_rad)) if np.isfinite(self.well_wobble_rad) else NAN,
            'sep_cycle_h': join_list(f'{c.h:.4g}' for c in self.cycles),
            'sep_cycle_J': join_list(f'{c.action_ratio:.4g}' for c in self.cycles),
            'sep_cycle_classes': join_list(str(c) for c in self.cycle_classes),
            'sep_episodes_in_separatrix_zone': self.episodes_in_separatrix_zone,
            'sep_n_outside': self.n_outside,
            'sep_turn_zone_widths': self.turn_zone_widths,
            'sep_zone_seconds': self.zone_seconds,
        }


def _deg_0_360(angle_rad: float) -> float:
    return float(np.degrees(angle_rad) % 360.0) if np.isfinite(angle_rad) else NAN


def _harmonics(sigma: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Columns [cos sigma, sin sigma, cos 2 sigma, sin 2 sigma, ...] (V = this @ coefficients)."""
    columns = []
    for n in range(1, n_harmonics + 1):
        columns += [np.cos(n * sigma), np.sin(n * sigma)]
    return np.column_stack(columns) if columns else np.zeros((np.size(sigma), 0))


def _minus_derivative(sigma: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Columns of -V'(sigma) for the same coefficients: a_n n sin(n sigma) - b_n n cos(n sigma)."""
    columns = []
    for n in range(1, n_harmonics + 1):
        columns += [n * np.sin(n * sigma), -n * np.cos(n * sigma)]
    return np.column_stack(columns)


def _coefficients(cos_coeffs: np.ndarray, sin_coeffs: np.ndarray) -> np.ndarray:
    """Interleave (a_1, b_1, a_2, b_2, ...) in the column order of `_harmonics`."""
    return np.column_stack([np.asarray(cos_coeffs, float), np.asarray(sin_coeffs, float)]).ravel()


def potential_from_coefficients(cos_coeffs: Sequence[float], sin_coeffs: Sequence[float], r2: float = NAN) -> Optional[Potential]:
    """The Potential of V(sigma) = sum a_n cos n sigma + b_n sin n sigma: well, top, Omega, V_min, V_max.

    None for a flat potential (all coefficients zero) or non-finite coefficients.
    """
    cos_coeffs = np.asarray(cos_coeffs, dtype=float)
    sin_coeffs = np.asarray(sin_coeffs, dtype=float)
    if not (np.all(np.isfinite(cos_coeffs)) and np.all(np.isfinite(sin_coeffs))) or not np.any(np.hypot(cos_coeffs, sin_coeffs) > 0):
        return None
    n_harmonics = len(cos_coeffs)
    coef = _coefficients(cos_coeffs, sin_coeffs)
    if n_harmonics == 1:  # V = R cos(sigma - theta): top at theta, well at theta + pi, V'' = R at the well
        amplitude, phase = float(np.hypot(cos_coeffs[0], sin_coeffs[0])), float(np.arctan2(sin_coeffs[0], cos_coeffs[0]))
        top, well = float(np.mod(phase, TWO_PI)), float(np.mod(phase + np.pi, TWO_PI))
        return Potential(1, cos_coeffs, sin_coeffs, float(r2), well, top, float(np.sqrt(amplitude)), -amplitude, amplitude)

    def value(x):
        return float((_harmonics(np.atleast_1d(x), n_harmonics) @ coef)[0])

    grid = np.linspace(0.0, TWO_PI, GRID_POINTS, endpoint=False)
    values = _harmonics(grid, n_harmonics) @ coef
    step = grid[1] - grid[0]

    def refine(index: int, sign: float) -> Tuple[float, float]:
        centre = grid[index]
        res = optimize.minimize_scalar(lambda x: sign * value(x), bounds=(centre - step, centre + step), method='bounded')
        angle = float(np.mod(res.x, TWO_PI))
        return (0.0 if np.isclose(angle, TWO_PI) else angle), value(res.x)

    well, v_min = refine(int(np.argmin(values)), 1.0)
    top, v_max = refine(int(np.argmax(values)), -1.0)
    orders = np.arange(1, n_harmonics + 1)
    curvature = -np.sum(orders**2 * (cos_coeffs * np.cos(orders * well) + sin_coeffs * np.sin(orders * well)))
    omega = float(np.sqrt(curvature)) if curvature > 0 else NAN
    return Potential(n_harmonics, cos_coeffs, sin_coeffs, float(r2), well, top, omega, v_min, v_max)


def amplitude_factor(amplitude_rad):
    """F(A) = (2/pi) K(m = sin^2(A/2)): period of a pendulum of half-amplitude A over its small period
    (F(90 deg) = 1.180, F(120 deg) = 1.373, F(150 deg) = 1.762). inf at A >= pi, nan for a missing A."""
    amplitude = np.abs(np.asarray(amplitude_rad, dtype=float))
    with np.errstate(invalid='ignore'):
        m = np.where(amplitude >= np.pi, 1.0, np.sin(amplitude / 2) ** 2)
    result = 2 / np.pi * ellipk(m)
    return float(result) if np.ndim(result) == 0 else result


def small_period(period_yr, amplitude_rad):
    """T0 = P / F(A) [yr]."""
    result = np.asarray(period_yr, dtype=float) / amplitude_factor(amplitude_rad)
    return float(result) if np.ndim(result) == 0 else result


def half_width(omega0_rad_yr, identity_coefficient):
    """Resonance half-width in the slow variable of the COSINE pendulum: 2 omega0 / |c| (au for an MMR).

    For a fitted V with more harmonics the separatrix rate is sqrt(2 (E_sep - V_min)), not 2 sqrt(V''): use
    `half_width_in_potential` (V = -cos 2s: 2 against 4)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 2 * np.asarray(omega0_rad_yr, float) / abs(identity_coefficient)
    return float(result) if np.ndim(result) == 0 else result


def excursion(omega0_rad_yr, amplitude_rad, identity_coefficient):
    """Predicted half-excursion of the slow variable: 2 omega0 sin(A/2) / |c|."""
    result = half_width(omega0_rad_yr, identity_coefficient) * np.sin(np.abs(np.asarray(amplitude_rad, float)) / 2)
    return float(result) if np.ndim(result) == 0 else result


def _first_top(potential: Potential, direction: float) -> float:
    """sigma of the first maximum of V from the well in `direction` (+1 or -1), refined; unwrapped near the well."""
    offsets = np.linspace(0.0, TWO_PI, GRID_POINTS + 1)
    x = potential.well_rad + direction * offsets
    values = potential.value(x)
    falling = np.flatnonzero(values[1:-1] >= values[2:])  # V stops growing at index i + 1
    i = int(falling[0]) + 1 if len(falling) else GRID_POINTS // 2
    lo, hi = sorted((x[i - 1], x[i + 1]))
    res = optimize.minimize_scalar(lambda y: -float(potential.value(np.atleast_1d(y))[0]), bounds=(lo, hi), method='bounded')
    return float(res.x)


def well_separatrix(potential: Potential) -> Tuple[float, float, float]:
    """(sigma_left_top, sigma_right_top, E_sep) of the deepest well: the first maxima of V on either side (unwrapped
    around `well_rad`) and the lower of their energies, the separatrix of this well [rad^2/yr^2]. One harmonic:
    well -+ pi and V_max."""
    if potential.n_harmonics == 1:
        return potential.well_rad - np.pi, potential.well_rad + np.pi, potential.v_max
    left, right = _first_top(potential, -1.0), _first_top(potential, 1.0)
    energy = min(float(potential.value(np.atleast_1d(left))[0]), float(potential.value(np.atleast_1d(right))[0]))
    return left, right, energy


def _loop_action(potential: Potential, energy: float, left_top: float, right_top: float) -> float:
    """J(E) = (1/pi) int sqrt(2 (E - V)) d sigma between the turning points of the loop around the well [rad^2/yr]."""

    def excess(x):
        return float(potential.value(np.atleast_1d(x))[0]) - energy

    well = potential.well_rad
    if not excess(well) < 0:
        return 0.0
    upper = right_top if excess(right_top) <= 0 else optimize.brentq(excess, well, right_top)
    lower = left_top if excess(left_top) <= 0 else optimize.brentq(excess, left_top, well)
    middle, half = (upper + lower) / 2, (upper - lower) / 2

    def integrand(theta):  # sigma = middle + half sin(theta) takes the square-root ends smoothly
        return half * np.cos(theta) * np.sqrt(max(-2 * excess(middle + half * np.sin(theta)), 0.0))

    value, _ = integrate.quad(integrand, -np.pi / 2, np.pi / 2, limit=200)
    return float(value / np.pi)


def separatrix_action(potential: Potential, separatrix: Optional[Tuple[float, float, float]] = None) -> float:
    """J_s of the deepest well: J at its separatrix energy (`well_separatrix`) [rad^2/yr]; 8 Omega / pi for one harmonic."""
    if potential.n_harmonics == 1:
        return 8 * potential.omega_rad_yr / np.pi
    left, right, energy = separatrix if separatrix is not None else well_separatrix(potential)
    return _loop_action(potential, energy, left, right)


def action_ratio_in_potential(
    potential: Potential,
    h: float,
    separatrix: Optional[Tuple[float, float, float]] = None,
    full_action: Optional[float] = None,
) -> float:
    """J/J_s of the libration of energy h (V_min + h (V_max - V_min)) in the deepest well of V.

    One harmonic: the closed form `action_ratio`. Otherwise quadratures over the well, J_s at the well's separatrix
    energy; `separatrix` (`well_separatrix`) and `full_action` (`separatrix_action`) are computed when not given (a
    caller with many cycles passes them once). 1 at and above the separatrix, 0 at h <= 0, nan for nan."""
    if potential is None or h is None or not np.isfinite(h):
        return NAN
    if potential.n_harmonics == 1:
        return action_ratio(h)
    if h <= 0:
        return 0.0
    separatrix = separatrix if separatrix is not None else well_separatrix(potential)
    left, right, separatrix_energy = separatrix
    energy = potential.v_min + h * potential.depth
    if energy >= separatrix_energy:
        return 1.0
    full = full_action if full_action is not None else separatrix_action(potential, separatrix)
    return _loop_action(potential, energy, left, right) / full if full > 0 else NAN


def action_slope_in_potential(
    potential: Optional[Potential],
    h: float,
    separatrix: Optional[Tuple[float, float, float]] = None,
    full_action: Optional[float] = None,
) -> float:
    """d(J/J_s)/dh of the libration of energy h in the deepest well of V.

    For J = (1/2 pi) oint sigma_dot d sigma, dJ/dE = T(E) / (2 pi) (Landau & Lifshitz, Mechanics, §49: dI/dE = T/2 pi),
    and E = V_min + h (V_max - V_min), so d(J/J_s)/dh = depth T(h) / (2 pi J_s), T from `period_in_potential`.
    One harmonic (or no potential): K(m = h)/2 (`_action_ratio_slope`; the formula gives 2 Omega^2 (4 K/Omega) /
    (2 pi 8 Omega/pi) = K/2). inf at and above the well's separatrix (the period diverges), nan for nan."""
    if h is None or not np.isfinite(h):
        return NAN
    if potential is None or potential.n_harmonics == 1:
        return float(_action_ratio_slope(h))
    separatrix = separatrix if separatrix is not None else well_separatrix(potential)
    if potential.v_min + max(h, 0.0) * potential.depth >= separatrix[2]:
        return math.inf
    full = full_action if full_action is not None else separatrix_action(potential, separatrix)
    period = period_in_potential(potential, max(h, 0.0))
    return float(potential.depth * period / (TWO_PI * full)) if full > 0 and np.isfinite(period) else NAN


def h_from_action_ratio_in_potential(
    potential: Optional[Potential],
    j: float,
    separatrix: Optional[Tuple[float, float, float]] = None,
    full_action: Optional[float] = None,
) -> float:
    """Inverse of `action_ratio_in_potential` on the libration side: the h whose loop in the deepest well has
    J/J_s = j (root finding between the bottom and the well's separatrix energy). One harmonic (or no potential):
    `h_from_action_ratio`. 0 for j <= 0, the separatrix h for j >= 1, nan for nan."""
    if j is None or not np.isfinite(j):
        return NAN
    if potential is None or potential.n_harmonics == 1:
        return float(h_from_action_ratio(j))
    separatrix = separatrix if separatrix is not None else well_separatrix(potential)
    h_top = (separatrix[2] - potential.v_min) / potential.depth
    if j <= 0:
        return 0.0
    if j >= 1:
        return float(h_top)
    full = full_action if full_action is not None else separatrix_action(potential, separatrix)

    def excess(h):
        return action_ratio_in_potential(potential, h, separatrix, full) - j

    return float(optimize.brentq(excess, 0.0, float(np.nextafter(h_top, 0.0)), xtol=1e-12))


@dataclass(frozen=True)
class Extrema:
    """The extrema of V over one turn, found exactly (`extrema`): angles sorted in [0, 2 pi) and which are minima."""

    sigma_rad: np.ndarray
    is_min: np.ndarray

    @property
    def n_wells(self) -> int:
        return int(np.count_nonzero(self.is_min))


@dataclass(frozen=True)
class Loop:
    """The libration loop of energy E through one sigma (`loop_through`): its turning points [rad] (unwrapped about
    that sigma) and its period [yr]; nan turning points and period when there is none (`reason`)."""

    lower_rad: float
    upper_rad: float
    period_yr: float
    reason: str = ''  # '' a loop; LOOP_NO_MOTION, LOOP_CIRCULATES, LOOP_SEPARATRIX (period inf)


LOOP_NO_MOTION = 'no_motion'  # V >= E where the loop should pass
LOOP_CIRCULATES = 'circulates'  # V < E all round on a side: the orbit turns
LOOP_SEPARATRIX = 'separatrix'  # E equals a maximum that bounds the loop: the period diverges
ROOT_ON_CIRCLE_TOL = 1e-3  # |z| - 1 of a root of z^N V'(z) taken as a candidate extremum (spurious ones show no sign change)


def _derivative(potential: Potential, x: np.ndarray) -> np.ndarray:
    return -(
        _minus_derivative(np.atleast_1d(np.asarray(x, dtype=float)), potential.n_harmonics)
        @ _coefficients(potential.cos_coeffs, potential.sin_coeffs)
    )


def extrema(potential: Potential) -> Extrema:
    """The extrema of V(sigma) = sum a_n cos n sigma + b_n sin n sigma over one turn, exactly (no grid).

    With z = e^{i sigma}, V'(sigma) z^N is a polynomial of degree 2N in z, coefficients n (b_n +- i a_n) / 2 at z^{N +- n}:
    its roots on the unit circle are the zeros of V'. Every root near the circle is a candidate; a candidate where V'
    changes sign is an extremum (a minimum where it goes from - to +), refined by bracketing V' between the neighbouring
    midpoints. A narrow barrier or well a grid would step over is found (V = -cos s - 0.01 cos 30 s)."""
    n = potential.n_harmonics
    coefficients = np.zeros(2 * n + 1, dtype=complex)  # ascending powers of z
    for k in range(1, n + 1):
        a, b = potential.cos_coeffs[k - 1], potential.sin_coeffs[k - 1]
        coefficients[n + k] = k * (b + 1j * a) / 2
        coefficients[n - k] = k * (b - 1j * a) / 2
    roots = np.roots(coefficients[::-1])
    near = roots[np.abs(np.abs(roots) - 1) < ROOT_ON_CIRCLE_TOL]
    candidates = np.unique(np.mod(np.angle(near), TWO_PI))
    if not len(candidates):
        return Extrema(np.zeros(0), np.zeros(0, dtype=bool))
    gaps = np.diff(np.concatenate([candidates, [candidates[0] + TWO_PI]]))
    after = candidates + gaps / 2  # the midpoint to the next candidate (cyclic)
    before = np.roll(after, 1) - np.where(np.arange(len(candidates)) == 0, TWO_PI, 0.0)
    left, right = _derivative(potential, before), _derivative(potential, after)
    angles, minima = [], []
    for lo, hi, d_lo, d_hi in zip(before, after, left, right):
        if not d_lo * d_hi < 0:
            continue
        root = optimize.brentq(lambda x: float(_derivative(potential, x)[0]), lo, hi, xtol=1e-14)
        angles.append(float(np.mod(root, TWO_PI)))
        minima.append(bool(d_lo < 0))
    order = np.argsort(angles)
    return Extrema(np.asarray(angles, dtype=float)[order], np.asarray(minima, dtype=bool)[order])


def well_holding(potential: Potential, sigma_rad: float, found: Optional[Extrema] = None) -> float:
    """The minimum of V whose basin holds `sigma_rad` (descending V from it), unwrapped near it [rad]; `sigma_rad` for
    a V without extrema."""
    found = found if found is not None else extrema(potential)
    if not len(found.sigma_rad):
        return float(sigma_rad)
    slope = float(_derivative(potential, sigma_rad)[0])
    offsets = np.mod(found.sigma_rad - sigma_rad, TWO_PI)  # ahead of sigma, in (0, 2 pi]
    if slope > 0:  # V grows ahead: the minimum is the first extremum behind
        i = int(np.argmax(offsets))
        return float(sigma_rad + offsets[i] - TWO_PI)
    i = int(np.argmin(offsets))
    return float(sigma_rad + offsets[i])


def _turning_point(potential: Potential, energy: float, start: float, direction: float, found: Extrema) -> Tuple[float, str]:
    """The first sigma from `start` in `direction` where V = E: V is monotone between neighbouring extrema, so the
    root is bracketed exactly on the first stretch whose far end reaches E (brentq); (nan, LOOP_CIRCULATES) when none
    does within one turn, (the maximum, LOOP_SEPARATRIX) when E equals it."""

    def excess(x):
        return float(potential.value(np.atleast_1d(x))[0]) - energy

    offsets = np.sort(np.mod(direction * (found.sigma_rad - start), TWO_PI))
    offsets = offsets[offsets > 0]
    previous = start
    for offset in list(offsets) + [TWO_PI]:
        x = start + direction * offset
        value = excess(x)
        if value == 0:
            return float(x), LOOP_SEPARATRIX
        if value > 0:
            return float(optimize.brentq(excess, *sorted((previous, x)), xtol=1e-14)), ''
        previous = x
    return NAN, LOOP_CIRCULATES


def loop_through(potential: Optional[Potential], energy: float, sigma_rad: float, found: Optional[Extrema] = None) -> Loop:
    """The libration loop of energy `energy` [rad^2/yr^2] (E = sigma_dot^2 / 2 + V) through `sigma_rad`, in whichever
    well of V holds it: its turning points (the first V = E on either side, bracketed between V's extrema,
    `_turning_point`) and its period T = 2 int d sigma / sqrt(2 (E - V)) between them. `found`: V's `extrema`, when
    the caller has them. No loop (nan) where V(sigma_rad) >= E, when a side circulates, without a potential; inf period
    on the separatrix."""
    none = Loop(NAN, NAN, NAN, LOOP_NO_MOTION)
    if potential is None or not (np.isfinite(energy) and np.isfinite(sigma_rad)):
        return none

    def kinetic(x):
        return 2 * (energy - float(potential.value(np.atleast_1d(x))[0]))

    if not kinetic(sigma_rad) > 0:
        return none
    found = found if found is not None else extrema(potential)
    (upper, up_reason), (lower, low_reason) = (_turning_point(potential, energy, sigma_rad, d, found) for d in (1.0, -1.0))
    reason = up_reason or low_reason
    if LOOP_CIRCULATES in (up_reason, low_reason):
        return Loop(NAN, NAN, NAN, LOOP_CIRCULATES)
    if reason == LOOP_SEPARATRIX:
        return Loop(lower, upper, math.inf, LOOP_SEPARATRIX)
    middle, half = (upper + lower) / 2, (upper - lower) / 2

    def integrand(theta):  # sigma = middle + half sin(theta) takes the square-root ends smoothly
        energy_left = kinetic(middle + half * np.sin(theta))
        # at a turning point the rounding of E - V may leave a negative zero: the limit there is finite, 0 is exact enough
        return half * np.cos(theta) / np.sqrt(energy_left) if energy_left > 0 else 0.0

    with warnings.catch_warnings():  # a loop a hair above its bottom or below its separatrix: quad's estimate stands
        warnings.simplefilter('ignore', integrate.IntegrationWarning)
        value, _ = integrate.quad(integrand, -np.pi / 2, np.pi / 2, limit=200)
    return Loop(float(lower), float(upper), float(2 * value))


def period_through(potential: Optional[Potential], energy: float, sigma_rad: float) -> float:
    """Period [yr] of the libration loop of energy `energy` through `sigma_rad` (`loop_through`); nan without a loop,
    inf on the separatrix. V = -cos s at E = -cos A: 4 K(sin^2(A/2)) (A = 90 deg: 7.416; E = 1 - 1e-8: 43.773)."""
    return loop_through(potential, energy, sigma_rad).period_yr


def well_count(potential: Optional[Potential]) -> int:
    """Number of minima of V over one turn (`extrema`); 0 without a potential. One harmonic: 1; V = -cos s + cos 2s: 2
    (wells at +-75.5 deg, cos s = 1/4; a barrier at 0)."""
    return extrema(potential).n_wells if potential is not None else 0


def half_width_in_potential(potential: Potential, identity_coefficient: float) -> float:
    """Half-width of the deepest well in the slow variable: sqrt(2 (E_sep - V_min)) / |c| (the largest |sigma_dot| on
    the well's separatrix over |c|); = 2 Omega / |c| for one harmonic."""
    if potential is None or identity_coefficient == 0:
        return NAN
    _, _, separatrix_energy = well_separatrix(potential)
    return float(np.sqrt(2 * max(separatrix_energy - potential.v_min, 0.0)) / abs(identity_coefficient))


def libration_period(h, omega_rad_yr):
    """4 K(m = h) / Omega [yr] for 0 <= h < 1 (nan elsewhere)."""
    h = np.asarray(h, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where((h >= 0) & (h < 1), 4 * ellipk(np.clip(h, 0, 1)) / omega_rad_yr, NAN)
    return float(result) if np.ndim(result) == 0 else result


def revolution_period(h, omega_rad_yr):
    """2 K(m = 1/h) / (Omega sqrt(h)) [yr] for h > 1 (nan elsewhere)."""
    h = np.asarray(h, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        safe = np.where(h > 1, h, 2.0)
        result = np.where(h > 1, 2 * ellipk(1 / safe) / (omega_rad_yr * np.sqrt(safe)), NAN)
    return float(result) if np.ndim(result) == 0 else result


def action_ratio(h):
    """J / J_s = E(m = h) - (1 - h) K(m = h) for 0 <= h <= 1 (h clipped to [0, 1]; nan stays nan)."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    with np.errstate(invalid='ignore'):
        result = np.where(h >= 1, 1.0, ellipe(h) - (1 - h) * ellipk(np.minimum(h, np.nextafter(1.0, 0.0))))
    result = np.where(np.isnan(h), NAN, result)
    return float(result) if np.ndim(result) == 0 else result


def _action_ratio_slope(h):
    """d(J/J_s)/dh = K(m = h) / 2 (from dE/dm = (E - K)/2m and dK/dm = (E - (1 - m)K) / (2m(1 - m)))."""
    return ellipk(np.clip(np.asarray(h, dtype=float), 0.0, np.nextafter(1.0, 0.0))) / 2


def h_from_action_ratio(j):
    """Inverse of `action_ratio` on [0, 1] (root finding); 0 below 0, 1 above 1, nan for nan."""
    values = np.atleast_1d(np.asarray(j, dtype=float))
    out = np.full(values.shape, NAN)
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        if value <= 0:
            out[i] = 0.0
        elif value >= 1:
            out[i] = 1.0
        else:
            out[i] = optimize.brentq(lambda h, target=value: action_ratio(h) - target, 0.0, 1.0, xtol=1e-14)
    return float(out[0]) if np.ndim(j) == 0 else out


def rate_min_over_mean(h):
    """r_pend(h) = (2/pi) sqrt((h - 1)/h) K(m = 1/h) for h > 1 (0.25 at h = 1.012, 0.554 at h = 1.14); nan for h <= 1."""
    h = np.asarray(h, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        safe = np.where(h > 1, h, 2.0)
        result = np.where(h > 1, 2 / np.pi * np.sqrt((safe - 1) / safe) * ellipk(1 / safe), NAN)
    return float(result) if np.ndim(result) == 0 else result


def period_in_potential(potential: Potential, h: float) -> float:
    """Period of the orbit of energy h in V(sigma): 4 K(h)/Omega or 2 K(1/h)/(Omega sqrt h) for one harmonic,
    otherwise the quadrature of d sigma / sqrt(2 (E - V)) over the libration loop around the deepest well
    (`loop_through`) or one turn [yr]. A cycle in another well takes `cycle_period` instead."""
    if potential is None or not np.isfinite(h) or h < 0:
        return NAN
    if potential.n_harmonics == 1:
        return libration_period(h, potential.omega_rad_yr) if h < 1 else revolution_period(h, potential.omega_rad_yr)
    energy = potential.v_min + h * potential.depth

    def kinetic(x):
        return 2 * (energy - float(potential.value(np.atleast_1d(x))[0]))

    if h > 1:
        value, _ = integrate.quad(lambda x: 1 / np.sqrt(max(kinetic(x), np.finfo(float).tiny)), 0.0, TWO_PI, limit=200)
        return float(value)
    if h == 1:
        return NAN
    return loop_through(potential, energy, potential.well_rad).period_yr


def cycle_period(potential: Potential, h: float, sigma_fast_rad: float, found: Optional[Extrema] = None) -> float:
    """Predicted period [yr] of a cycle of energy h whose fastest sample is at `sigma_fast_rad`: a revolution (h > 1)
    as `period_in_potential`; a libration in the loop of its energy around the minimum whose basin holds that sample
    (`well_holding`, `loop_through`): in V of one well the deepest well's loop, in a double well the loop of the well
    the cycle is in (nan when its energy lies below that well's bottom, inf on its separatrix)."""
    if potential is None or not np.isfinite(h) or h < 0:
        return NAN
    if h > 1 or potential.n_harmonics == 1 or not np.isfinite(sigma_fast_rad):
        return period_in_potential(potential, h)
    found = found if found is not None else extrema(potential)
    energy = potential.v_min + h * potential.depth
    return loop_through(potential, energy, well_holding(potential, sigma_fast_rad, found), found).period_yr


def _segments(t_yr: np.ndarray, windows: Sequence[Tuple[float, ...]]) -> Tuple[np.ndarray, np.ndarray]:
    """(first, stop) sample indices of each window [start, end] (closed) on an increasing time grid."""
    starts = np.array([w[0] for w in windows], dtype=float)
    ends = np.array([w[1] for w in windows], dtype=float)
    return np.searchsorted(t_yr, starts, side='left'), np.searchsorted(t_yr, ends, side='right')


def _segment_sums(values: np.ndarray, first: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """Sum of values[first_k:stop_k] along axis 0 for every k (the segments may overlap; an empty one sums to 0)."""
    if not len(first):
        return np.zeros((0,) + values.shape[1:], dtype=values.dtype)
    padded = np.concatenate([values, np.zeros((1,) + values.shape[1:], dtype=values.dtype)])
    sums = np.add.reduceat(padded, np.column_stack([first, stop]).ravel(), axis=0)[::2]
    sums[stop <= first] = 0
    return sums


def _gather(first: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """The indices first_0..stop_0 - 1, first_1..stop_1 - 1, ... concatenated (a sample in two windows twice)."""
    lengths = stop - first
    offsets = np.repeat(first - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
    return np.arange(int(lengths.sum())) + offsets


@dataclass
class _WindowSamples:
    """The samples of the fit windows, window after window, and the window bounds in that concatenation."""

    angle: np.ndarray
    rate: np.ndarray
    strength: Optional[np.ndarray]  # m(t) of a modulated frame; None = 1
    first: np.ndarray  # start of each window in the arrays above
    stop: np.ndarray


def _window_samples(t_yr, angle, rate, windows, settings, frame) -> Optional[_WindowSamples]:
    """The finite samples of every window with more than 2 separatrix_harmonics_max + 1 of them, in the frame."""
    t_yr = np.asarray(t_yr, dtype=float)
    first, stop = _segments(t_yr, windows)
    stop = np.maximum(stop, first)
    if not len(first) or not np.any(stop > first):
        return None
    span = slice(int(first.min()), int(stop.max()))  # the frame and the finiteness only where a window looks
    phi, phi_rate, strength = in_frame(t_yr[span], np.asarray(angle)[span], np.asarray(rate)[span], frame)
    finite = np.isfinite(phi) & np.isfinite(phi_rate) & np.isfinite(strength)
    index = _gather(first - span.start, stop - span.start)
    index = index[finite[index]]
    counts = _segment_sums(finite.astype(np.int64), first - span.start, stop - span.start)
    keep = counts > 2 * settings.separatrix_harmonics_max + 1
    if not np.any(keep):
        return None
    owner = np.repeat(np.arange(len(counts)), counts)
    index = index[keep[owner]]
    bounds = np.concatenate([[0], np.cumsum(counts[keep])])
    return _WindowSamples(phi[index], phi_rate[index], strength[index] if frame is not None else None, bounds[:-1], bounds[1:])


def _demeaned(values: np.ndarray, samples: _WindowSamples) -> np.ndarray:
    """values minus the mean of its window (the window constants of the fit eliminated, Frisch-Waugh)."""
    lengths = samples.stop - samples.first
    means = _segment_sums(values, samples.first, samples.stop) / lengths.reshape((-1,) + (1,) * (values.ndim - 1))
    return values - np.repeat(means, lengths, axis=0)


def _r2(target: np.ndarray, residual: np.ndarray) -> float:
    total = float(np.sum(target**2))
    return 1 - float(np.sum(residual**2)) / total if total > 0 else NAN


def _rate_design(samples: _WindowSamples, n_harmonics: int) -> Tuple[np.ndarray, np.ndarray]:
    """(design, target) of rate^2 = 2E_w - 2 m V(angle), both with each window's mean removed."""
    design = -2 * _harmonics(samples.angle, n_harmonics)
    if samples.strength is not None:
        design *= samples.strength[:, None]
    return _demeaned(design, samples), _demeaned(samples.rate**2, samples)


def _fit_rate(samples: _WindowSamples, n_harmonics: int) -> Tuple[Optional[np.ndarray], float]:
    """Shared V with one constant 2E per window: rate^2 = 2E_w - 2 m V(angle). Returns (V coefficients, within-window R^2).

    With m(t) the strength of the modulated pendulum (angle and rate are then phi and phi_dot of `in_frame`). The
    window constants are eliminated by removing each window's mean from the target and the harmonics (Frisch-Waugh):
    the same coefficients and residuals as with one dummy column per window, without a (samples x windows) design."""
    design, target = _rate_design(samples, n_harmonics)
    if design.shape[0] <= design.shape[1] + len(samples.first):
        return None, NAN
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef, _r2(target, target - design @ coef)


def _window_fits(samples: _WindowSamples, n_harmonics: int) -> List[Potential]:
    """V of every window alone (`_fit_rate` on one window), all windows at once: each window's normal equations,
    solved together; a window whose equations are ill-conditioned (a libration of a few degrees, two harmonics) is
    fitted by least squares on its own samples, as `_fit_rate` would."""
    design, target = _rate_design(samples, n_harmonics)
    columns = design.shape[1]
    gram = np.empty((len(samples.first), columns, columns))
    for i in range(columns):
        for j in range(i, columns):
            gram[:, i, j] = gram[:, j, i] = _segment_sums(design[:, i] * design[:, j], samples.first, samples.stop)
    moment = _segment_sums(design * target[:, None], samples.first, samples.stop)
    with np.errstate(all='ignore'):
        conditioned = np.linalg.cond(gram) < GRAM_CONDITION_MAX
    coef = np.full((len(gram), columns), NAN)
    if np.any(conditioned):
        coef[conditioned] = np.linalg.solve(gram[conditioned], moment[conditioned][..., None])[..., 0]
    for k in np.flatnonzero(~conditioned):
        part = slice(samples.first[k], samples.stop[k])
        coef[k] = np.linalg.lstsq(design[part], target[part], rcond=None)[0]
    lengths = samples.stop - samples.first
    residual = target - np.einsum('ij,ij->i', design, np.repeat(coef, lengths, axis=0))
    total = _segment_sums(target**2, samples.first, samples.stop)
    squares = _segment_sums(residual**2, samples.first, samples.stop)
    potentials = []
    for k in range(len(coef)):
        r2 = 1 - float(squares[k]) / float(total[k]) if total[k] > 0 else NAN
        potential = _potential(coef[k], r2)
        if potential is not None:
            potentials.append(potential)
    return potentials


def _potential(coef: Optional[np.ndarray], r2: float) -> Optional[Potential]:
    if coef is None:
        return None
    return potential_from_coefficients(coef[0::2], coef[1::2], r2)


def fit_potential_from_rate(
    t_yr: np.ndarray,
    angle: np.ndarray,
    rate: np.ndarray,
    windows: Sequence[Tuple[float, float]],
    settings: AnalysisSettings,
    frame: Optional[ModulatedFrame] = None,
    with_windows: bool = True,
) -> Tuple[Optional[Potential], List[Potential]]:
    """V_E: least squares sigma_dot^2 = 2E - 2V(sigma) in each window (one cycle or one turn); N = 2 when
    R^2 grows by >= separatrix_harmonic_gain. Returns (fit over all windows, per-window fits).

    The fit over all windows shares V and gives each window its own E; its R^2 is taken within the windows
    (the spread of E between windows is not credited). The number of harmonics grows from 1 up to
    separatrix_harmonics_max while each one adds >= separatrix_harmonic_gain to that R^2. A window enters with more
    than 2 separatrix_harmonics_max + 1 finite samples. `t_yr` increases.

    With a `frame` the fit is phi_dot^2 = 2E - 2 m(t) V(phi), phi = sigma - sigma_s(t) (`in_frame`): the shape and
    the phase of V and its scale Omega_bar still come from sigma alone, the modulation m(t), sigma_s(t) from the frame.
    `with_windows=False` skips the per-window fits (the second element is then empty).
    """
    samples = _window_samples(t_yr, angle, rate, windows, settings, frame)
    if samples is None:
        return None, []
    n_harmonics = 1
    coef, r2 = _fit_rate(samples, n_harmonics)
    while n_harmonics < settings.separatrix_harmonics_max:
        more_coef, more_r2 = _fit_rate(samples, n_harmonics + 1)
        if not (np.isfinite(more_r2) and np.isfinite(r2) and more_r2 - r2 >= settings.separatrix_harmonic_gain):
            break
        n_harmonics, coef, r2 = n_harmonics + 1, more_coef, more_r2
    return _potential(coef, r2), _window_fits(samples, n_harmonics) if with_windows else []


def fit_potential_from_force(
    angle: np.ndarray,
    slow_rate: np.ndarray,
    identity_coefficient: float,
    n_harmonics: int,
    strength_rel_sq: Optional[np.ndarray] = None,
) -> Optional[Potential]:
    """V_F: least squares c * x_dot = -m V'(sigma) on the whole window (x_dot the slow-variable rate).

    `strength_rel_sq` is m(t) of a modulated frame (then `angle` is phi of `in_frame`); None = 1."""
    angle = np.asarray(angle, dtype=float)
    target = identity_coefficient * np.asarray(slow_rate, dtype=float)
    weight = np.ones_like(angle) if strength_rel_sq is None else np.asarray(strength_rel_sq, dtype=float)
    mask = np.isfinite(angle) & np.isfinite(target) & np.isfinite(weight)
    angle, target, weight = angle[mask], target[mask], weight[mask]
    if n_harmonics < 1 or len(angle) <= 2 * n_harmonics + 1:
        return None
    design = np.column_stack([_minus_derivative(angle, n_harmonics) * weight[:, None], np.ones_like(angle)])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    return _potential(coef[:-1], filtering_r2(target, residual))


def well_coherence(windows: Sequence[Potential]) -> float:
    """|<e^{i sigma_s}>| over the per-window potentials (0..1); nan for no window."""
    return circular_resultant([p.well_rad for p in windows if p is not None])


def cycle_energies(
    t_yr: np.ndarray,
    angle: np.ndarray,
    rate: np.ndarray,
    bounds: Sequence[Tuple[float, float, bool]],
    potential: Potential,
    frame: Optional[ModulatedFrame] = None,
) -> List[CycleEnergy]:
    """Mean h of each cycle or turn (`bounds`: t_start, t_end, rotation), its J/<J_s> and the period law; `t_yr`
    increases, a cycle without samples is skipped.

    In a modulated `frame` h(t) = (phi_dot^2 / 2 + m (V(phi) - V_min)) / (m (V_max - V_min)) is the energy over the
    separatrix energy of that moment, the predicted period is that of the cycle's mean strength,
    T_pend(h) / sqrt(m_cycle), J/<J_s> = J/J_s(h) sqrt(m_cycle) and h_mean = h m_cycle (SEPARATRIX §2.3).
    J/J_s(h) is that of `potential` itself (`action_ratio_in_potential`): the closed form for one harmonic,
    quadratures over the deepest well otherwise. The predicted period of a libration cycle is that of the loop of its
    energy in the well that holds it (`cycle_period`: the basin of its fastest sample), so the period law holds in a
    potential of several wells too; with one well it is the deepest well's (`period_in_potential`)."""
    if potential is None or not len(bounds):
        return []
    t_yr = np.asarray(t_yr, dtype=float)
    first, stop = _segments(t_yr, bounds)
    present = stop > first
    if not np.any(present):
        return []
    first, stop = first[present], stop[present]
    bounds = [bound for bound, keep in zip(bounds, present) if keep]
    span = slice(int(first.min()), int(stop.max()))
    phi, phi_rate, strength = in_frame(t_yr[span], np.asarray(angle)[span], np.asarray(rate)[span], frame)
    first, stop = first - span.start, stop - span.start
    energy = phi_rate**2 / 2 + strength * (potential.value(phi) - potential.v_min)
    h = _nan_means(energy / (strength * potential.depth), first, stop)
    m_cycle = _nan_means(strength, first, stop)
    with np.errstate(invalid='ignore'):
        scale = np.where(m_cycle > 0, np.sqrt(np.abs(m_cycle)), NAN)
    start = np.array([bound[0] for bound in bounds], dtype=float)
    end = np.array([bound[1] for bound in bounds], dtype=float)
    rotation = np.array([bool(bound[2]) for bound in bounds])
    if potential.n_harmonics == 1:
        ratio = np.where(np.isfinite(h), action_ratio(h), NAN)
        with np.errstate(invalid='ignore'):
            libration = np.isfinite(h) & (h >= 0) & (h < 1)
        period = np.where(libration, libration_period(h, potential.omega_rad_yr), revolution_period(h, potential.omega_rad_yr))
    else:
        separatrix = well_separatrix(potential)
        full_action = separatrix_action(potential, separatrix)
        ratio = np.array([NAN if r else action_ratio_in_potential(potential, x, separatrix, full_action) for x, r in zip(h, rotation)])
        found = extrema(potential)
        fastest = [
            phi[a + int(np.nanargmax(phi_rate[a:b] ** 2))] if np.any(np.isfinite(phi_rate[a:b])) else NAN for a, b in zip(first, stop)
        ]
        period = np.array([cycle_period(potential, x, f, found) for x, f in zip(h, fastest)])
    ratio = np.where(rotation, NAN, ratio * scale)
    with np.errstate(invalid='ignore', divide='ignore'):
        predicted = period / scale
        law = np.where(np.isfinite(predicted) & (predicted > 0), (end - start) / predicted, NAN)
    return [
        CycleEnergy(float(a), float(b), bool(r), float(x), float(j), float(b - a), float(w), float(x * m), float(m))
        for a, b, r, x, j, w, m in zip(start, end, rotation, h, ratio, law, m_cycle)
    ]


def _nan_means(values: np.ndarray, first: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """np.nanmean of values[first_k:stop_k] for every k: NaN left out, +-inf kept (nan where every value is NaN)."""
    present = ~np.isnan(values)
    with np.errstate(invalid='ignore', divide='ignore'):
        return _segment_sums(np.where(present, values, 0.0), first, stop) / _segment_sums(present.astype(float), first, stop)


def energy_scatter(h: Sequence[float], floor: float) -> float:
    """sigma_h: robust scatter of h between neighbouring cycles (1.4826 MAD of differences / sqrt 2), >= floor;
    nan with fewer than two finite values."""
    values = np.asarray([x for x in h if np.isfinite(x)], dtype=float)
    if len(values) < 2:
        return NAN
    differences = np.diff(values)
    mad = float(np.median(np.abs(differences - np.median(differences))))
    return max(MAD_TO_SIGMA * mad / np.sqrt(2), floor)


def sigma_only_check(period_law_med: float, coherence: float, settings: AnalysisSettings) -> CheckValue:
    """R0 sigma-only: pass when |period_law_med - 1| <= separatrix_period_tol and coherence >=
    separatrix_phase_coherence; not_measured with fewer than two cycles/turns (nan input)."""
    if not (np.isfinite(period_law_med) and np.isfinite(coherence)):
        return CheckValue.NOT_MEASURED
    law_holds = abs(period_law_med - 1) <= settings.separatrix_period_tol
    well_fixed = coherence >= settings.separatrix_phase_coherence
    return CheckValue.PASS if law_holds and well_fixed else CheckValue.FAIL


def with_slow_check(sigma_only: CheckValue, window_correlations: Sequence[float], settings: AnalysisSettings) -> CheckValue:
    """R0 with the slow variable: sigma-only passes and V_E ~ V_F (r >= separatrix_vevf_r_min) in a
    fraction >= separatrix_vevf_fraction of the windows. A sigma-only fail, not_measured or not_applicable
    is passed through; no finite window correlation is not_measured."""
    if sigma_only != CheckValue.PASS:
        return sigma_only
    values = np.asarray([r for r in window_correlations if r is not None and np.isfinite(r)], dtype=float)
    if not len(values):
        return CheckValue.NOT_MEASURED
    fraction = float(np.mean(values >= settings.separatrix_vevf_r_min))
    return CheckValue.PASS if fraction >= settings.separatrix_vevf_fraction else CheckValue.FAIL


def _libration_action(cycle: CycleEnergy, potential: Optional[Potential], scatter: float, well) -> Tuple[float, float]:
    """(J/<J_s>, sigma_J) of a libration cycle in `potential` (the cosine without one): the cycle's own ratio (else
    J/J_s(h) of the potential) and the scatter of h carried by the slope, both times sqrt(m_cycle) (J/<J_s> =
    J/J_s(h) sqrt(m)). `well` = (separatrix, J_s) of the potential, computed once by the caller."""
    separatrix, full = well
    scale = np.sqrt(cycle.strength_rel_sq) if cycle.strength_rel_sq > 0 else 1.0
    ratio = cycle.action_ratio
    if not np.isfinite(ratio):
        ratio = (
            action_ratio_in_potential(potential, cycle.h, separatrix, full) if potential is not None else action_ratio(cycle.h)
        ) * scale
    return ratio, action_slope_in_potential(potential, cycle.h, separatrix, full) * scale * scatter


def _well(potential: Optional[Potential]):
    """(separatrix, J_s) of a multi-harmonic potential, (None, None) otherwise (the closed forms need neither)."""
    if potential is None or potential.n_harmonics == 1:
        return None, None
    separatrix = well_separatrix(potential)
    return separatrix, separatrix_action(potential, separatrix)


def classify_cycles(
    cycles: Sequence[CycleEnergy], zone: Zone, sigma_h: float, settings: AnalysisSettings, potential: Optional[Potential] = None
) -> List[CycleClass]:
    """Deep: J/<J_s> + margin sigma_J < J_lo (libration side); above: h_mean - margin sigma_h > h_hi; else zone
    (margin = separatrix_sigma_margin). sigma_J = d(J/J_s)/dh sigma_h sqrt(m), the slope in `potential`
    (`action_slope_in_potential`: depth T / (2 pi J_s), K(h)/2 for one harmonic, W3-G7); a missing sigma_h counts as
    the floor separatrix_sigma_floor. The zone edges are in the mean normalisation (J0/<J_s>, h0 over 2 Omega_bar^2),
    hence J/<J_s> and h_mean of the cycle."""
    scatter = sigma_h if np.isfinite(sigma_h) else settings.separatrix_sigma_floor
    margin = settings.separatrix_sigma_margin
    well = _well(potential)
    classes = []
    for cycle in cycles:
        if cycle.rotation:
            above = cycle.energy_over_mean - margin * scatter > zone.h_hi
            classes.append(CycleClass.ABOVE if above else CycleClass.ZONE)
            continue
        ratio, sigma_j = _libration_action(cycle, potential, scatter, well)
        deep = ratio + margin * sigma_j < zone.j_lo
        classes.append(CycleClass.DEEP if deep else CycleClass.ZONE)
    return classes


def edge_cycles(
    cycles: Sequence[CycleEnergy], zone: Zone, sigma_h: float, settings: AnalysisSettings, potential: Optional[Potential] = None
) -> List[bool]:
    """Per cycle: lies within one sigma (sigma_J or sigma_h) of a zone edge, so its class is not stable (zone_edge_cycle).
    sigma_J as in `classify_cycles` (the slope of J/J_s in `potential`)."""
    scatter = sigma_h if np.isfinite(sigma_h) else settings.separatrix_sigma_floor
    well = _well(potential)
    edges = []
    for cycle in cycles:
        if cycle.rotation:
            edges.append(bool(abs(cycle.energy_over_mean - zone.h_hi) < scatter))
            continue
        ratio, sigma_j = _libration_action(cycle, potential, scatter, well)
        edges.append(bool(abs(ratio - zone.j_lo) < sigma_j))
    return edges


def crossings(cycles: Sequence[CycleEnergy], slip_times_yr: Sequence[float]) -> List[float]:
    """Times [yr] where h - 1 changes sign between neighbouring cycles (the boundary between them), and slips."""
    times = [float(t) for t in slip_times_yr]
    for previous, current in zip(cycles[:-1], cycles[1:]):
        if np.isfinite(previous.h) and np.isfinite(current.h) and (previous.h - 1) * (current.h - 1) < 0:
            times.append((previous.t_end_yr + current.t_start_yr) / 2)
    return sorted(set(times))


def regime_transitions(
    t_yr: np.ndarray,
    angle: np.ndarray,
    turning_times_yr: Sequence[float],
    turning_values_rad: Sequence[float],
    revolution_mid_times_yr: Sequence[float],
) -> List[Tuple[float, str]]:
    """The ordered directed changes of regime read from the events of the angle itself, frame-free: (time [yr],
    'exit' | 'capture').

    A half-cycle between consecutive turning points belongs to a libration when sigma swings by at most 2pi (a
    half-amplitude never exceeds pi) and is a slip when it swings by more: the angle went over the top once, at the
    moment it passes the middle of the swing (the top for a slip from c - A to c + 2pi + A'). Every full turn without
    turning points is a circulation event at its middle. A transition is the boundary between consecutive events of
    different kinds, at the slip's moment when a slip is involved, else at the midpoint between the two events: an exit
    from a libration half-cycle to a slip or a turn, a capture from a slip or a turn to a libration half-cycle.

    A slip between two librations gives an exit and a capture at the same moment, a libration that ends in turns one
    exit, a run of turns nothing; a libration whose swing reaches the top of a moving well but turns back gives
    nothing. Nothing is inferred beyond the first and the last event of the record."""
    t_yr, angle = np.asarray(t_yr, dtype=float), np.asarray(angle, dtype=float)
    times = np.asarray(turning_times_yr, dtype=float)
    values = np.asarray(turning_values_rad, dtype=float)
    events = [(float(t), 1, False) for t in revolution_mid_times_yr]  # (time, circulation, slip)
    for i in range(len(times) - 1):
        swing = abs(values[i + 1] - values[i])
        if swing <= TWO_PI:
            events.append(((times[i] + times[i + 1]) / 2, 0, False))
            continue
        inside = (t_yr >= times[i]) & (t_yr <= times[i + 1])
        middle = (values[i] + values[i + 1]) / 2
        passed = np.flatnonzero(np.diff(np.sign(angle[inside] - middle)) != 0)
        moment = float(t_yr[inside][passed[0] + 1]) if len(passed) else (times[i] + times[i + 1]) / 2
        events.append((moment, 1, True))
    events.sort()
    found = []
    for (a, kind_a, slip_a), (b, kind_b, slip_b) in zip(events[:-1], events[1:]):
        if kind_a == kind_b:
            continue
        moment = a if slip_a else b if slip_b else (a + b) / 2
        found.append((moment, EXIT if kind_a == 0 else CAPTURE))
    return found


def cyclic_shift_p_value(crossing_times_yr: Sequence[float], weak_phase: np.ndarray, t_yr: np.ndarray, n_shifts: int = 0) -> float:
    """p of 'crossings sit in the weak phase of Omega(t)' under cyclic shifts of the whole crossing pattern
    (keeps the gaps between crossings; independent placement overstates significance). `n_shifts` 0 = all.

    The statistic is the number of crossings in the weak phase (a boolean mask on `t_yr`); p is the fraction of
    shifts (the zero shift included) that put at least as many crossings there. nan without crossings.
    """
    t_yr = np.asarray(t_yr, dtype=float)
    weak = np.asarray(weak_phase, dtype=bool)
    times = np.asarray(crossing_times_yr, dtype=float)
    times = times[(times >= t_yr[0]) & (times <= t_yr[-1])]
    if not len(times) or len(t_yr) < 2:
        return NAN
    span = t_yr[-1] - t_yr[0] + float(np.median(np.diff(t_yr)))  # the record as a circle of its samples
    shifts = t_yr - t_yr[0] if n_shifts <= 0 else np.linspace(0.0, span, n_shifts, endpoint=False)
    shifted = t_yr[0] + np.mod(times[None, :] - t_yr[0] + shifts[:, None], span)
    index = np.clip(np.searchsorted(t_yr, shifted), 0, len(t_yr) - 1)
    counts = weak[index].sum(axis=1)
    return float(np.mean(counts >= counts[0]))  # shifts[0] = 0 is the observed pattern


def action_trend(cycles: Sequence[CycleEnergy]) -> Tuple[float, float]:
    """(slope of J/J_s over cycle mid-times [1/yr], its standard error); nan with fewer than three libration cycles."""
    points = [((c.t_start_yr + c.t_end_yr) / 2, c.action_ratio) for c in cycles if not c.rotation and np.isfinite(c.action_ratio)]
    if len(points) < 3:
        return NAN, NAN
    x, y = np.asarray(points, dtype=float).T
    x_centred = x - x.mean()
    spread = float(np.sum(x_centred**2))
    if spread <= 0:
        return NAN, NAN
    slope = float(np.sum(x_centred * (y - y.mean())) / spread)
    residual = y - y.mean() - slope * x_centred
    error = float(np.sqrt(np.sum(residual**2) / (len(x) - 2) / spread))
    return slope, error
