"""Energy relative to the separatrix for an MMR, and the zone the pulsating separatrix reaches (SPEC §4.6, §4.6.1).

Model of the record (SEPARATRIX §2.3): the force of an MMR is that of its multiplet,
a_dot = sum_j [A_j sin sigma_j + B_j cos sigma_j], sigma_j = sigma + j (varpi - varpi_P), which is one harmonic of
sigma with a slowly turning complex amplitude W(t) = sum_j (A_j + i B_j) e^{i j (varpi - varpi_P)}:
a_dot = |W| sin(sigma + arg W), hence sigma_ddot = c a_dot = -Omega(t)^2 sin(sigma - sigma_s(t)) with
Omega(t)^2 = |c| |W(t)| and sigma_s = -arg W (+ pi when c > 0). Without resolved members (the R^2 of the members' fit
gains less than mmr_force_multi_gain) the strength follows the bounded oscillatory part of e, |Z| ~ (e_osc / e_bar)^q
(q = |p|, the body's power of e in the term), with a fixed sigma_s (flag zone_eq_only); the trend of e is left out (it
is the `trend` mechanism, not the zone: PHYSICS B1c).

Energies are measured in a frame (`pendulum.ModulatedFrame`): fixed, the modulation's (members or e^q), or the centre
of the oscillation (sigma alone; a Hilda's centre follows the forced varpi); the one whose V_E explains most of
sigma_dot^2 is taken (`best_frame`). The shape of the modulation comes from the force (or e), the scale Omega_bar and
the phase of the well from sigma (V_E). R0 is measured only where it means something (`r0_checks`). The zone (SPEC
§4.6.1 with the corrected factor of PHYSICS B1a: the e^q half-range is q de/e_bar) is computed numerically:
sigma_ddot = -Omega_bar^2 m(t) sin(sigma - sigma_s(t)) from a grid of starts, J0/<J_s> from mmr_zone_j_min to 1
and h0 from 1 to mmr_zone_h_max (step mmr_zone_j_step), at mmr_zone_phases start times over one modulation period,
each to the end of the record (DOP853, rtol mmr_zone_rtol, on a smooth spline of the model). J_lo is the smallest J0
that ever goes over the top, h_hi the largest h0 that is ever trapped (its relative rate changes sign).

The cycles are full cycles inside each run of turning points (the pairing restarts at every run) plus the turns that
hold no turning point, labelled episode / diagnostic / turn (`labelled_cycle_bounds`, W3-G4). The zone integration
evaluates the model once per start time and is skipped when the small period 2 pi / Omega_bar is shorter than the
smoothing of sigma (`_zone_of`: a degenerate V_E).

The zone leaf 6 does not read a status from the zone (DELTA 13b): libration, episodic and near_separatrix decide.
The classes are not columns only, though (W25 review): the cycle classes give L2 (thin with a cycle in the zone,
not_measured with a cycle not measured and none in the zone) and the flag libration_in_separatrix_zone;
`episodes_in_separatrix_zone` (crossings with every cycle inside the model's zone) caps a transient at medium, makes O1
not applicable and withholds the onset objection of the unmeasured-force branch (leaf 5o: uncertain/origin_unknown);
zone_edge_cycle thins Z. R0 (sigma-only and with a) is a check of the tree (§5.3). The mechanism, the exits and
captures (`pendulum.regime_transitions`) and the cycle energies themselves are columns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import DOP853
from scipy.interpolate import CubicSpline, PPoly

from resonances.resonance.analysis import filtering, pendulum
from resonances.resonance.analysis.geometry import (
    Geometry,
    confirm_jumps,
    half_amplitudes,
    jump_baseline_yr,
    jump_candidates,
    well_shift,
)
from resonances.resonance.analysis.mmr import centre_track as centre_track_module
from resonances.resonance.analysis.mmr.models import Context, MMRSeries, Preprocessed, Response, SlowComponents
from resonances.resonance.analysis.mmr.response import members_fit, multiplet_harmonics, smoothed_series
from resonances.resonance.analysis.pendulum import (
    CycleClass,
    EnergyAnalysis,
    ModulatedFrame,
    SeparatrixMechanism,
    Zone,
    ZoneMethod,
)
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue, Flag

NAN = float('nan')
TWO_PI = 2 * np.pi
YR_PER_KYR = 1.0e3
SOURCE_MEMBERS, SOURCE_ECCENTRICITY, SOURCE_NONE = 'members', 'eccentricity', 'none'
# zone trajectories are integrated in pieces of this many small periods (memory; decided ones leave the state)
ZONE_CHUNK_PERIODS = 64
# nodes of the smooth model per small period (or modulation period): a resampling resolution, not a threshold
ZONE_NODES_PER_PERIOD = 32
ZONE_BATCH_SAMPLES = 256  # samples per batch of the trajectory checks (memory: trajectories x this, not x a chunk)

# separatrix_ve_r2_min: R0 needs a measured potential: V_E must explain this share of sigma_dot^2 within the cycles
# (Hilda 153 0.96, 145480 0.93, 52356 0.79 in their best frames; 24820 0.63 with 4 cycles at lambda 0.8, 469333 0.37,
# 78085 0.002). separatrix_cutoff_cycles: V_E, V_F, the members' modulation and the zone read sigma and a smoothed at
# max(P_c, this x P_cyc): the 121-yr forced wave of 52356 passes P_c = 20 yr and, differentiated, swamps a_dot
# (members' R^2 0.04 at 0.05 P_cyc, 0.93 at 0.2 P_cyc; SEPARATRIX §3 used 500 yr). G(P_cyc) = 0.998, third harmonic 0.89.


# --------------------------------------------------------------------------------------------- series


@dataclass
class Modulation:
    """The pulsation of the force over the record: Omega(t)^2 and sigma_s(t) on `t_yr` (SEPARATRIX §2.3)."""

    source: str  # SOURCE_MEMBERS | SOURCE_ECCENTRICITY | SOURCE_NONE
    t_yr: np.ndarray
    omega_squared: np.ndarray  # rad^2/yr^2 (members: |c||W|; e^q and none: relative, scaled later by Omega_bar)
    well_rad: np.ndarray  # sigma_s(t), unwrapped (fixed for e^q and none)
    planet: str = ''
    eps: Tuple[float, ...] = ()
    gain: float = NAN  # R^2 of the members' fit minus that of the single harmonic
    nu_rad_yr: float = NAN  # frequency of the modulation
    eq_dk_over_k: float = NAN
    e_trend_per_kyr: float = NAN

    @property
    def omega_rel(self) -> np.ndarray:
        """Omega(t) / Omega_bar, Omega_bar the mean of Omega(t) (so that <J_s> ~ Omega_bar)."""
        omega = np.sqrt(np.clip(self.omega_squared, 0, None))
        mean = float(np.nanmean(omega)) if np.any(np.isfinite(omega)) else NAN
        return omega / mean if np.isfinite(mean) and mean > 0 else np.ones_like(omega)

    @property
    def modulation_period_yr(self) -> float:
        return TWO_PI / self.nu_rad_yr if np.isfinite(self.nu_rad_yr) and self.nu_rad_yr > 0 else NAN

    def frame(self) -> ModulatedFrame:
        """m(t) = (Omega/Omega_bar)^2 and the zero-mean motion of the stable point."""
        shift = np.asarray(self.well_rad, dtype=float) - float(np.mean(self.well_rad))
        rate = filtering.time_derivative(self.t_yr, shift) if len(shift) > 1 else np.zeros_like(shift)
        return ModulatedFrame(self.t_yr, self.omega_rel**2, shift, rate)


def separatrix_cutoff_yr(cutoff_yr: float, period_yr: float, settings: AnalysisSettings) -> float:
    """max(P_c, separatrix_cutoff_cycles x P) [yr]; P_c without a period."""
    if not (np.isfinite(period_yr) and period_yr > 0):
        return float(cutoff_yr)
    return float(max(cutoff_yr, settings.separatrix_cutoff_cycles * period_yr))


def separatrix_series(
    prep: Preprocessed, period_yr: float, settings: AnalysisSettings
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """(sigma, sigma_dot, a, a_dot, cutoff) for the energy and the zone: prep's filtered series low-passed again at
    `separatrix_cutoff_yr` (`smoothed_series`: the derivatives are those of the smoothed series)."""
    return smoothed_series(prep, separatrix_cutoff_yr(prep.cutoff.cutoff_yr, period_yr, settings), settings)


CYCLE_EPISODE, CYCLE_DIAGNOSTIC, CYCLE_TURN = 'episode', 'diagnostic', 'turn'
# where the libration cycles' classes come from: the fitted pendulum, or the averaged well at each turning point
SOURCE_PENDULUM, SOURCE_AVERAGED_POTENTIAL = 'pendulum', 'averaged_potential'


def _run_of_turning_points(geometry: Geometry) -> Tuple[np.ndarray, dict]:
    """(run index of each turning point of the window, -1 where no run holds it; {run index: accepted episode})."""
    times = np.asarray(geometry.turning_points.times_yr, float)
    owner = np.full(len(times), -1, dtype=int)
    accepted = {}
    for number, run in enumerate(geometry.runs):
        owner[np.isin(times, np.asarray(run.tp_times_yr, float))] = number
        accepted[number] = bool(run.accepted and not run.before_last_jump)
    return owner, accepted


def labelled_cycle_bounds(geometry: Geometry) -> List[Tuple[float, float, bool, str]]:
    """(start, end, rotation, kind) of the cycles the energy reads (W3-G4).

    Full cycles (TP_i, TP_i+2) are formed inside each run of turning points, the pairing restarting at every run, so a
    cycle never joins the turning points of two runs (52356: 10049-19539 yr crossed a gap; 182223: 92 of 105 records
    did). kind 'episode' for a run accepted as an episode of the window, 'diagnostic' for a rejected run (or turning
    points no run holds): those are measured but are not the energies of a confirmed episode. Every full turn of a
    circulation that holds no turning point is a 'turn' (the geometry also counts a net 2 pi advance over librations as
    a turn: 52356, 0-74.6 kyr)."""
    times = np.asarray(geometry.turning_points.times_yr, float)
    owner, accepted = _run_of_turning_points(geometry)
    bounds = []
    start = 0
    while start < len(times):
        stop = start
        while stop + 1 < len(times) and owner[stop + 1] == owner[start]:
            stop += 1
        kind = CYCLE_EPISODE if accepted.get(int(owner[start]), False) else CYCLE_DIAGNOSTIC
        bounds += [(float(times[i]), float(times[i + 2]), False, kind) for i in range(start, stop - 1, 2)]
        start = stop + 1
    for turn in geometry.revolutions:
        if not np.any((times > turn.t_start_yr) & (times < turn.t_end_yr)):
            bounds.append((turn.t_start_yr, turn.t_end_yr, True, CYCLE_TURN))
    return sorted(bounds)


def cycle_bounds(geometry: Geometry) -> List[Tuple[float, float, bool]]:
    """(start, end, rotation) of `labelled_cycle_bounds`: cycles inside each run of turning points and the turns."""
    return [(start, end, rotation) for start, end, rotation, _ in labelled_cycle_bounds(geometry)]


def cycle_kinds(geometry: Geometry, cycles: Sequence[pendulum.CycleEnergy]) -> List[str]:
    """The kind ('episode', 'diagnostic', 'turn') of each measured cycle, matched by its bounds."""
    kinds = {(start, end): kind for start, end, _, kind in labelled_cycle_bounds(geometry)}
    return [kinds.get((c.t_start_yr, c.t_end_yr), CYCLE_DIAGNOSTIC) for c in cycles]


def record_period(geometry: Geometry) -> float:
    """The period the smoothing follows: P_cyc, else the median turn of a circulation [yr]."""
    if np.isfinite(geometry.period_cycle_yr) and geometry.period_cycle_yr > 0:
        return float(geometry.period_cycle_yr)
    turns = [r.t_end_yr - r.t_start_yr for r in geometry.revolutions]
    return float(np.median(turns)) if turns else NAN


# --------------------------------------------------------------------------------------------- modulation


def modulation_from_members(
    t_yr: np.ndarray,
    angle_f: np.ndarray,
    adot_f: np.ndarray,
    slow_offset: np.ndarray,
    identity_coefficient: float,
    harmonics: Sequence[int] = (0, 1),
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...], float]:
    """Fit a_dot = sum_j [A_j sin sigma_j + B_j cos sigma_j] + C over `mask`; returns (Omega(t)^2 = |c||Z(t)|
    [rad^2/yr^2], sigma_s(t) [rad], eps_j = |C_j|/|C_0| (j != 0, in the order of `harmonics`), gain in R^2 over the
    single harmonic j = 0). sigma_j = sigma + j slow_offset (slow_offset = varpi - varpi_P).

    With C_j = A_j + i B_j, a_dot = Im[W e^{i sigma}] = |W| sin(sigma + arg W), W(t) = sum_j C_j e^{i j slow_offset}:
    sigma_ddot = c a_dot = -|c||W| sin(sigma - sigma_s) with sigma_s = -arg W (c < 0) or -arg W - pi (c > 0).
    All nan when the fit is impossible (no finite samples, c = 0)."""
    t_yr, angle_f, adot_f, offset = (np.asarray(x, dtype=float) for x in (t_yr, angle_f, adot_f, slow_offset))
    harmonics = tuple(int(j) for j in harmonics)
    if 0 not in harmonics:
        harmonics = (0,) + harmonics
    use = np.ones_like(t_yr, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    use = use & np.isfinite(angle_f) & np.isfinite(adot_f) & np.isfinite(offset)
    blank = np.full_like(t_yr, NAN)
    if not (np.isfinite(identity_coefficient) and identity_coefficient != 0) or np.count_nonzero(use) <= 2 * len(harmonics) + 1:
        return blank, blank.copy(), tuple(NAN for j in harmonics if j != 0), NAN

    angle_use, offset_use, target = angle_f[use], offset[use], adot_f[use]
    _, single_r2 = members_fit(angle_use, offset_use, target, (0,))
    coefficients, r2 = members_fit(angle_use, offset_use, target, harmonics)
    amplitude = sum(c * np.exp(1j * j * offset) for c, j in zip(coefficients, harmonics))
    omega_squared = abs(identity_coefficient) * np.abs(amplitude)
    well = -np.unwrap(np.angle(amplitude)) - (np.pi if identity_coefficient > 0 else 0.0)
    main = abs(coefficients[harmonics.index(0)])
    eps = tuple(float(abs(c) / main) if main > 0 else NAN for c, j in zip(coefficients, harmonics) if j != 0)
    return omega_squared, well, eps, float(r2 - single_r2)


def modulation_from_eccentricity(
    t_yr: np.ndarray, e_slow: np.ndarray, order: int, omega_bar_rad_yr: float, nu_rad_yr: float = NAN
) -> Tuple[np.ndarray, float, float]:
    """Omega(t)^2 = Omega_bar^2 (e_osc / e_bar)^q from the bounded oscillatory part of e (linear trend removed, mean
    kept); returns (Omega^2(t), dK/K = q de/e_bar, e trend [1/kyr]) with de, e_bar the half-range and mid-range of
    e_osc (PHYSICS B1a: a half-range, not the full one). `order` q = 0 or e_bar <= 0: no modulation.

    The trend is fitted together with a sinusoid at `nu_rad_yr` (the e-cycle, when known): a plain straight line
    takes a tilt from an unfinished cycle (4 % of dK/K at 11.8 cycles). The sinusoid enters only when the segment
    covers at least one full period 2 pi / nu: on a shorter one it cannot be told from the line and the constant, and
    the fit trades them off freely (33128 6N-7: a 6 Myr cycle on 40 kyr gave a trend of -0.107 / kyr against e = 0.38-0.40
    and a false dK/K of 5.55)."""
    t, e = np.asarray(t_yr, dtype=float), np.asarray(e_slow, dtype=float)
    finite = np.isfinite(t) & np.isfinite(e)
    constant = np.full_like(t, omega_bar_rad_yr**2)
    if np.count_nonzero(finite) < 2:
        return constant, NAN, NAN
    centred = t - t[finite].mean()
    columns = [np.ones_like(centred), centred]
    span = float(np.ptp(t[finite]))
    if np.isfinite(nu_rad_yr) and nu_rad_yr > 0 and np.count_nonzero(finite) > 4 and nu_rad_yr * span >= TWO_PI:
        columns += [np.cos(nu_rad_yr * t), np.sin(nu_rad_yr * t)]
    design = np.column_stack(columns)
    slope = float(np.linalg.lstsq(design[finite], e[finite], rcond=None)[0][1])
    oscillation = e - slope * centred
    high, low = float(np.nanmax(oscillation)), float(np.nanmin(oscillation))
    mid, half = (high + low) / 2, (high - low) / 2
    trend = slope * YR_PER_KYR
    if order == 0 or mid <= 0:
        return constant, 0.0, trend
    ratio = np.clip(oscillation / mid, 0, None)
    return omega_bar_rad_yr**2 * ratio**order, float(order * half / mid), trend


def _varpi_offset(series: MMRSeries, planet: str, context: Context, t_yr: np.ndarray) -> Optional[np.ndarray]:
    """varpi - varpi_P on `t_yr`: the saved planet's varpi when there is one, else its rate g_P t (a constant phase
    only rotates the C_j)."""
    varpi = np.unwrap(np.asarray(series.varpi, dtype=float))
    saved = series.planets.get(planet)
    if saved is not None and saved.varpi is not None and np.all(np.isfinite(saved.varpi)):
        return varpi - np.unwrap(np.asarray(saved.varpi, dtype=float))
    rate = context.planet_varpi_rate_rad_yr.get(planet, NAN)
    return varpi - rate * t_yr if np.isfinite(rate) else None


def _offset_rate(offset: np.ndarray, t_yr: np.ndarray, mask: np.ndarray) -> float:
    """|d offset / dt| by least squares over `mask` [rad/yr] (nu of the modulation)."""
    if np.count_nonzero(mask) < 2:
        return NAN
    return float(abs(np.polyfit(t_yr[mask], offset[mask], 1)[0]))


def best_members(
    series: MMRSeries,
    t_yr: np.ndarray,
    angle: np.ndarray,
    adot: np.ndarray,
    mask: np.ndarray,
    context: Context,
    cutoff_yr: float,
    settings: AnalysisSettings,
) -> Optional[Modulation]:
    """The members' modulation over each planet of the term (multiplet_harmonics of the varpi coefficients); the
    planet with the largest gain wins. None for a retrograde angle or when no planet gives a finite fit."""
    spec = series.resonance
    if spec.retrograde:
        return None
    best = None
    for planet, planet_p in zip(spec.planets, spec.planet_varpi_terms):
        harmonics = multiplet_harmonics(spec.body_varpi_coefficient, planet_p)
        offset = _varpi_offset(series, planet, context, t_yr)
        if offset is None or len(harmonics) < 2:
            continue
        offset = filtering.lowpass_with(t_yr, offset, cutoff_yr, settings)
        omega_squared, well, eps, gain = modulation_from_members(t_yr, angle, adot, offset, context.identity_coefficient, harmonics, mask)
        if not np.isfinite(gain) or (best is not None and gain <= best.gain):
            continue
        nu = _offset_rate(offset, t_yr, mask)
        best = Modulation(SOURCE_MEMBERS, t_yr, omega_squared, well, planet, eps, gain, nu)
    return best


def eccentricity_modulation(series: MMRSeries, slow: SlowComponents, context: Context, t_yr: np.ndarray) -> Modulation:
    """The e^q route: q = |p| of the body, nu = |varpi_dot - g_P| of the most massive planet of the term; relative
    Omega^2 (Omega_bar = 1), fixed stable point; dK/K and the trend of e over the trimmed slow window."""
    spec = series.resonance
    e = np.interp(t_yr, np.asarray(slow.t_yr, float), np.asarray(slow.e_slow, float))
    nu = NAN
    masses = {name: context.planet_mass.get(name, NAN) for name in spec.planets}
    if masses and not spec.retrograde:
        planet = max(masses, key=lambda name: masses[name] if np.isfinite(masses[name]) else -np.inf)
        rate = context.planet_varpi_rate_rad_yr.get(planet, NAN)
        nu = abs(context.body_varpi_rate_rad_yr - rate) if np.isfinite(rate) else NAN
    inside = filtering.window_mask(t_yr, slow.window)
    use = inside if np.count_nonzero(inside) >= 2 else np.ones_like(t_yr, dtype=bool)
    order = abs(spec.body_varpi_coefficient)
    _, dk, trend = modulation_from_eccentricity(t_yr[use], e[use], order, 1.0, nu)
    omega_squared, _, _ = modulation_from_eccentricity(t_yr, e, order, 1.0, nu)
    if not np.all(np.isfinite(omega_squared)):
        omega_squared = np.ones_like(t_yr)
    source = SOURCE_ECCENTRICITY if order != 0 else SOURCE_NONE
    return Modulation(source, t_yr, omega_squared, np.zeros_like(t_yr), '', (), NAN, nu, dk, trend)


# --------------------------------------------------------------------------------------------- the zone


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    """start, start + step, ... up to stop (inclusive within half a step)."""
    if step <= 0 or stop < start:
        return np.zeros(0)
    count = int(np.floor((stop - start) / step + 0.5)) + 1
    return start + step * np.arange(count)


def zone_starts(t0: float, t1: float, modulation_period_yr: float, n_phases: int) -> np.ndarray:
    """n_phases start times spread evenly over one modulation period from t0 (over the record when the period is
    longer or unknown)."""
    span = t1 - t0
    length = min(modulation_period_yr, span) if np.isfinite(modulation_period_yr) and modulation_period_yr > 0 else span
    return t0 + length * np.arange(max(int(n_phases), 1)) / max(int(n_phases), 1)


def _initial_states(
    starts: np.ndarray, strength: np.ndarray, well: np.ndarray, j_grid: np.ndarray, h_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per trajectory (phase-major): start time, sigma0, sigma0' (in units of Omega_bar), side (0 = libration J0,
    1 = rotation h0) and whether it is decided at the start (J0 beyond the instantaneous separatrix action, or E0 below
    the instantaneous separatrix energy). At the bottom of the well: J/J_s(t0) = J0 / sqrt(m0) gives h, and
    phi' = 2 sqrt(m0 h); a rotation of h0 (over 2 Omega_bar^2) has phi' = 2 sqrt(h0)."""
    times, sigma, speed, side, decided = [], [], [], [], []
    for start, m0, s0 in zip(starts, strength, well):
        root = np.sqrt(m0) if m0 > 0 else 0.0
        inst = j_grid / root if root > 0 else np.full_like(j_grid, np.inf)
        # the model integrated here is one harmonic, so its J <-> h is the cosine one (not V_E's)
        h = np.asarray(pendulum.h_from_action_ratio(np.minimum(inst, 1.0)), dtype=float).reshape(-1)
        times += [start] * (len(j_grid) + len(h_grid))
        sigma += [s0] * (len(j_grid) + len(h_grid))
        speed += list(2 * root * np.sqrt(h)) + list(2 * np.sqrt(h_grid))
        side += [0] * len(j_grid) + [1] * len(h_grid)
        decided += list(inst >= 1) + list(h_grid < m0)
    return tuple(np.asarray(x) for x in (times, sigma, speed, side, decided))


def model_splines(
    t_yr: np.ndarray, strength: np.ndarray, well: np.ndarray, omega_bar: float, modulation_period_yr: float
) -> Tuple[CubicSpline, CubicSpline]:
    """m(t) and sigma_s(t) as cubic splines on nodes ZONE_NODES_PER_PERIOD per small period 2 pi / (Omega_bar sqrt(m_max))
    or per modulation period, whichever is shorter (never denser than the record). A piecewise-linear model on the
    record's own samples has a kink at every sample, and DOP853 then crawls through the record sample by sample."""
    t = np.asarray(t_yr, dtype=float)
    strength, well = np.asarray(strength, dtype=float), np.asarray(well, dtype=float)
    small = TWO_PI / (omega_bar * np.sqrt(max(float(np.max(strength)), np.finfo(float).tiny)))
    scales = [small] + ([modulation_period_yr] if np.isfinite(modulation_period_yr) and modulation_period_yr > 0 else [])
    spacing = max(min(scales) / ZONE_NODES_PER_PERIOD, float(np.median(np.diff(t))))
    count = max(int(np.ceil((t[-1] - t[0]) / spacing)) + 1, 4)
    nodes = np.linspace(t[0], t[-1], count)
    return CubicSpline(nodes, np.interp(nodes, t, strength)), CubicSpline(nodes, np.interp(nodes, t, well))


class _ZoneChecks:
    """Running checks of the active trajectories over the samples of one chunk, fed in short batches (memory: n x
    ZONE_BATCH_SAMPLES, never n x the samples of a chunk): over the top on the libration side, a sign change of the
    relative rate on the rotation side."""

    def __init__(self, n: int, previous_rate: np.ndarray, duration: np.ndarray, starts: np.ndarray, of_start: np.ndarray):
        self.n, self.duration, self.starts, self.of_start = n, duration, starts, of_start
        self.over_top = np.zeros(n, dtype=bool)
        self.flips = np.zeros(n, dtype=bool)
        self.rate = previous_rate
        self.last = None  # the state at the last sample

    def add(self, u: np.ndarray, y: np.ndarray, end_yr: float, well, well_rate, omega_bar: float) -> None:
        n = self.n
        times = np.minimum(self.starts[:, None] + u[None, :] / omega_bar, end_yr)
        valid = u[None, :] <= self.duration[:, None]
        phi = y[:n] - well(times)[self.of_start]
        rel = y[n:] - well_rate(times)[self.of_start] / omega_bar
        self.over_top |= np.any((np.abs(phi) > np.pi) & valid, axis=1)  # phi starts at 0, the bottom of the well
        signs = np.sign(np.concatenate([self.rate[:, None], rel], axis=1))
        self.flips |= np.any((signs[:, 1:] * signs[:, :-1] < 0) & valid, axis=1)
        self.rate = rel[:, -1]
        self.last = y[:, -1]


def _integrate_chunk(rhs, tau, stop, y0, samples, rtol, checks: _ZoneChecks, end_yr, well, well_rate, omega_bar) -> None:
    """DOP853 from tau to stop, step by step as `solve_ivp(..., t_eval=samples)` does (the same steps and the same
    dense-output values), feeding the samples to `checks` in batches instead of keeping all of them."""
    solver = DOP853(rhs, float(tau), y0, float(stop), rtol=rtol, atol=rtol)
    first, batch_u, batch_y, held = 0, [], [], 0
    while solver.status == 'running':
        solver.step()
        if solver.status == 'failed':
            break
        last = int(np.searchsorted(samples, solver.t, side='right'))
        if last > first:
            batch_u.append(samples[first:last])
            batch_y.append(solver.dense_output()(samples[first:last]))
            held += last - first
            first = last
        if batch_u and (held >= ZONE_BATCH_SAMPLES or solver.status != 'running'):
            checks.add(np.concatenate(batch_u), np.hstack(batch_y), end_yr, well, well_rate, omega_bar)
            batch_u, batch_y, held = [], [], 0
    if batch_u:
        checks.add(np.concatenate(batch_u), np.hstack(batch_y), end_yr, well, well_rate, omega_bar)


def numeric_zone(
    t_yr: np.ndarray,
    omega_squared: Callable[[np.ndarray], np.ndarray],
    well: Callable[[np.ndarray], np.ndarray],
    settings: AnalysisSettings,
    modulation_period_yr: float = NAN,
    omega_bar_rad_yr: float = NAN,
) -> Tuple[float, float]:
    """(J_lo, h_hi): the smallest J0/<J_s> whose trajectory goes over the top (|sigma - sigma_s| > pi), the largest
    h0 > 1 (over 2 Omega_bar^2) whose relative rate changes sign, integrating every grid start to the end of the record.

    `omega_squared` and `well` take arrays of times; Omega_bar defaults to the mean of sqrt(Omega^2) on `t_yr`.
    Nothing reaches the separatrix: J_lo = 1 and h_hi = 1."""
    t = np.asarray(t_yr, dtype=float)
    omega2 = np.clip(np.asarray(omega_squared(t), dtype=float), 0, None)
    omega_bar = omega_bar_rad_yr if np.isfinite(omega_bar_rad_yr) and omega_bar_rad_yr > 0 else float(np.mean(np.sqrt(omega2)))
    if not (np.isfinite(omega_bar) and omega_bar > 0) or len(t) < 2:
        return NAN, NAN
    strength_on_t, well_on_t = model_splines(t, omega2 / omega_bar**2, np.asarray(well(t), dtype=float), omega_bar, modulation_period_yr)
    well_rate_on_t = well_on_t.derivative()
    # both splines in one piecewise polynomial (same nodes, columns m and sigma_s): one call per right-hand side,
    # the same numbers (each column is evaluated by the same code)
    model_on_t = PPoly(np.stack([strength_on_t.c, well_on_t.c], axis=-1), strength_on_t.x)
    j_grid = _grid(settings.mmr_zone_j_min, 1.0, settings.mmr_zone_j_step)
    h_grid = _grid(1.0 + settings.mmr_zone_j_step, settings.mmr_zone_h_max, settings.mmr_zone_j_step)
    starts = zone_starts(t[0], t[-1], modulation_period_yr, settings.mmr_zone_phases)
    start_strength = strength_on_t(starts)
    start_well = well_on_t(starts)
    begin, sigma0, speed0, side, decided = _initial_states(starts, start_strength, start_well, j_grid, h_grid)
    values = np.concatenate([np.concatenate([j_grid, h_grid]) for _ in starts])
    reached = decided.copy()
    active = ~decided
    state = np.concatenate([sigma0, speed0])
    duration = (t[-1] - begin) * omega_bar  # elapsed tau of each trajectory
    previous_rate = speed0 - well_rate_on_t(begin) / omega_bar
    tau, tau_end = 0.0, float(np.max(duration))
    chunk = ZONE_CHUNK_PERIODS * TWO_PI / max(float(np.sqrt(np.max(omega2))) / omega_bar, 1.0)
    end_yr = t[-1]
    while tau < tau_end and np.any(active):
        index = np.flatnonzero(active)
        n = len(index)
        y0 = np.concatenate([state[index], state[len(begin) + index]])
        # the trajectories share mmr_zone_phases start times: the model is evaluated once per start time and spread
        # (the same numbers; a spline call on every trajectory cost 15x more at ~1000 trajectories)
        # (begin is phase-major and index increasing, so the trajectories of one start time are consecutive)
        starts_u, of_start, counts = np.unique(begin[index], return_inverse=True, return_counts=True)

        # past its own end a trajectory sees the model frozen at the record's end: the cubic extrapolation of the
        # spline explodes there and DOP853 crawls (385266: 4.4e6 calls, 6 GB in one piece); those samples are masked.
        # u >= 0, so the times never fall before t[0]. phi'' = -m sin(phi - sigma_s), written in place (cost)
        def rhs(u, y, starts_u=starts_u, counts=counts, n=n):
            model = model_on_t(np.minimum(starts_u + u / omega_bar, end_yr))
            out = np.empty(2 * n)
            out[:n] = y[n:]
            force = out[n:]
            np.subtract(y[:n], np.repeat(model[:, 1], counts), out=force)
            np.sin(force, out=force)
            np.multiply(np.repeat(model[:, 0], counts), force, out=force)
            np.negative(force, out=force)
            return out

        stop = min(tau + chunk, tau_end)
        # the checks read ZONE_NODES_PER_PERIOD samples per small period, from the dense output of each step
        samples = np.linspace(tau, stop, max(int(np.ceil((stop - tau) / chunk * ZONE_CHUNK_PERIODS * ZONE_NODES_PER_PERIOD)), 2) + 1)
        checks = _ZoneChecks(n, previous_rate[index], duration[index], starts_u, of_start)
        _integrate_chunk(rhs, tau, stop, y0, samples, settings.mmr_zone_rtol, checks, end_yr, well_on_t, well_rate_on_t, omega_bar)
        hit = np.where(side[index] == 0, checks.over_top, checks.flips)
        reached[index[hit]] = True
        active[index[hit]] = False
        if checks.last is not None:
            state[index] = checks.last[:n]
            state[len(begin) + index] = checks.last[n:]
        previous_rate[index] = checks.rate
        tau = stop
        active &= duration > tau
    j_hit = values[(side == 0) & reached]
    h_hit = values[(side == 1) & reached]
    j_lo = float(np.min(j_hit)) if len(j_hit) else 1.0
    h_hi = float(np.max(h_hit)) if len(h_hit) else 1.0
    return j_lo, h_hi


def build_zone(
    t_yr: np.ndarray,
    omega_squared: np.ndarray,
    well: np.ndarray,
    adiabaticity: float,
    settings: AnalysisSettings,
    modulation_period_yr: float = NAN,
    omega_bar_rad_yr: float = NAN,
    method: Optional[ZoneMethod] = None,
    potential: Optional[pendulum.Potential] = None,
) -> Zone:
    """The zone by `mmr_zone_method`, with the adiabatic cross-check below mmr_zone_adiabatic_lambda_max.

    'adiabatic': J_lo = Omega_min/Omega_bar, h_hi = (Omega_max/Omega_bar)^2 (the energy of the widest separatrix);
    otherwise the numeric zone, reported as `method` (NUMERIC for the members' modulation, EQ_ONLY for e^q).
    The zone is computed on the one-harmonic model of the header (its J0 <-> h is the cosine one); J_lo is a ratio
    J/<J_s>, so h_lo, the energy of the cycles' normalisation at that ratio, is read in the measured `potential` V_E
    (`pendulum.h_from_action_ratio_in_potential`, W3-G7; the cosine without one; its deepest well). h_lo is a column and
    the figure's band: the classes compare actions with J_lo (`pendulum.classify_cycles`, `averaged_classes`)."""
    t = np.asarray(t_yr, dtype=float)
    omega2 = np.clip(np.asarray(omega_squared, dtype=float), 0, None)
    omega = np.sqrt(omega2)
    omega_bar = omega_bar_rad_yr if np.isfinite(omega_bar_rad_yr) and omega_bar_rad_yr > 0 else float(np.mean(omega))
    relative = omega / float(np.mean(omega)) if np.mean(omega) > 0 else np.ones_like(omega)
    # the adiabatic J-zone [Omega_min/Omega_bar, Omega_max/Omega_bar] (J_s ~ Omega; sqrt(1 - m) for strength 1 + m cos nu t)
    lo, hi = float(np.min(relative)), float(np.max(relative))
    if settings.mmr_zone_method == str(ZoneMethod.ADIABATIC):
        return Zone(ZoneMethod.ADIABATIC, lo, pendulum.h_from_action_ratio_in_potential(potential, lo), hi**2, lo, hi, False)
    scaled = relative**2 * omega_bar**2
    j_lo, h_hi = numeric_zone(
        t,
        lambda times: np.interp(times, t, scaled),
        lambda times: np.interp(times, t, np.asarray(well, dtype=float)),
        settings,
        modulation_period_yr,
        omega_bar,
    )
    mismatch = bool(
        np.isfinite(adiabaticity)
        and adiabaticity < settings.mmr_zone_adiabatic_lambda_max
        and np.isfinite(j_lo)
        and abs(j_lo - lo) > settings.mmr_zone_adiabatic_tol
    )
    h_lo = pendulum.h_from_action_ratio_in_potential(potential, j_lo)
    return Zone(method or ZoneMethod.NUMERIC, j_lo, h_lo, h_hi, lo, hi, mismatch)


# --------------------------------------------------------------------------------------------- analysis


def _window_correlation(
    t: np.ndarray,
    angle: np.ndarray,
    rate: np.ndarray,
    adot: np.ndarray,
    window: Tuple[float, float],
    frame: ModulatedFrame,
    identity_coefficient: float,
    settings: AnalysisSettings,
) -> float:
    """corr(V_E, V_F) on a grid of phi for the potentials fitted in one force window (both in the frame)."""
    start, end = window
    inside = (t >= start) & (t <= end)
    if np.count_nonzero(inside) < 2 * settings.separatrix_harmonics_max + 2:
        return NAN
    v_e, _ = pendulum.fit_potential_from_rate(t, angle, rate, [window], settings, frame=frame)
    if v_e is None:
        return NAN
    phi, _, strength = pendulum.in_frame(t[inside], angle[inside], rate[inside], frame)
    v_f = pendulum.fit_potential_from_force(phi, adot[inside], identity_coefficient, v_e.n_harmonics, strength)
    if v_f is None:
        return NAN
    grid = np.linspace(0, TWO_PI, 2 * v_e.n_harmonics * settings.mmr_zone_phases, endpoint=False)
    first, second = v_e.value(grid), v_f.value(grid)
    if np.std(first) <= 0 or np.std(second) <= 0:
        return NAN
    return float(np.corrcoef(first, second)[0, 1])


def _percentile(values: Sequence[float], q: float) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.percentile(finite, q)) if finite else NAN


def _mechanism(
    crossing_times: Sequence[float],
    cycles: Sequence[pendulum.CycleEnergy],
    zone: Zone,
    adiabaticity: float,
    permutation_p: float,
    kick_times: Sequence[float],
    period_yr: float,
    settings: AnalysisSettings,
) -> SeparatrixMechanism:
    """What moved the orbit across the separatrix (SPEC §4.6): a kick (an energy-changing step of a within one cycle
    of a crossing), a trend (J/<J_s> drifts at separatrix_trend_sigma and by more than the libration side of the zone),
    pulsation (crossings in the weak phase, p <= separatrix_permutation_p_max), layer (lambda >
    separatrix_layer_adiabaticity_min), else unexplained."""
    if not crossing_times:
        return SeparatrixMechanism.NONE
    near = period_yr if np.isfinite(period_yr) else 0.0
    if any(abs(k - c) <= near for k in kick_times for c in crossing_times):
        return SeparatrixMechanism.KICK
    slope, error = pendulum.action_trend(cycles)
    spans = [(c.t_start_yr + c.t_end_yr) / 2 for c in cycles if not c.rotation]
    if np.isfinite(slope) and np.isfinite(error) and len(spans) >= 2:
        change = abs(slope) * (max(spans) - min(spans))
        width = 1.0 - zone.j_lo if np.isfinite(zone.j_lo) else NAN
        if abs(slope) > settings.separatrix_trend_sigma * error and np.isfinite(width) and change > width:
            return SeparatrixMechanism.TREND
    if np.isfinite(permutation_p) and permutation_p <= settings.separatrix_permutation_p_max:
        return SeparatrixMechanism.PULSATION
    if np.isfinite(adiabaticity) and adiabaticity > settings.separatrix_layer_adiabaticity_min:
        return SeparatrixMechanism.LAYER
    return SeparatrixMechanism.UNEXPLAINED


def _kick_times(t: np.ndarray, a_f: np.ndarray, prep: Preprocessed, cycles, sigma_h, step_yr, period_yr, settings) -> List[float]:
    """Times of the energy-changing steps of a (geometry's jump locator confirmed by the cycle energies)."""
    tau = jump_baseline_yr(prep.cutoff.cutoff_yr, step_yr, settings)
    candidates = jump_candidates(t, a_f, tau, period_yr, settings)
    mid = np.array([(c.t_start_yr + c.t_end_yr) / 2 for c in cycles])
    h = np.array([c.h for c in cycles])
    jumps = confirm_jumps(candidates, mid, h, sigma_h, NAN, settings)
    return [j.time_yr for j in jumps if not j.kick and not j.by_slow_only]


FRAME_FIXED, FRAME_CENTRE, FRAME_WELL = 'fixed', 'centre', 'well'


def candidate_frames(t_yr: np.ndarray, modulation: Modulation, geometry: Geometry) -> dict:
    """The frames V_E is tried in: fixed; the modulation's (members: m(t) and sigma_s(t); e^q: m(t)); the centre of the
    oscillation (sigma alone: the stable point follows the centre track, m = 1) when the centre does not circulate
    (its largest monotone pass < pi); the moving well (the stable point follows c(t) of the averaged potential, m = 1)
    when an episode was accepted in its frame (`geometry.well_shift`, continuous between such episodes). A Hilda's
    centre follows the forced varpi by +-31 deg while it librates by 19 deg: only the centre frame fits it (R^2 0.96
    against 0.08 fixed and 0.38 members; 153)."""
    t_yr = np.asarray(t_yr, dtype=float)
    frames = {FRAME_FIXED: None}
    well = well_shift(t_yr, geometry.runs, continuous=True)
    if well is not None:
        frames[FRAME_WELL] = ModulatedFrame(t_yr, np.ones_like(t_yr), well, filtering.time_derivative(t_yr, well))
    if modulation.source != SOURCE_NONE:
        frames[modulation.source] = modulation.frame()
    centre = geometry.centre
    if centre is not None and len(centre.t_yr) > 1 and np.isfinite(centre.max_pass_rad) and centre.max_pass_rad < np.pi:
        track = np.interp(t_yr, np.asarray(centre.t_yr, float), np.asarray(centre.centre_rad, float))
        shift = track - float(np.mean(track))
        frames[FRAME_CENTRE] = ModulatedFrame(t_yr, np.ones_like(t_yr), shift, filtering.time_derivative(t_yr, shift))
    return frames


def best_frame(t_yr, angle, rate, windows, frames: dict, settings: AnalysisSettings):
    """(name, frame, V_E, per-window fits) of the frame whose V_E explains most of sigma_dot^2 within the windows
    (the same number of fitted coefficients in every frame); (None, None, None, []) when no fit exists."""
    best = (None, None, None)
    for name, frame in frames.items():
        potential, _ = pendulum.fit_potential_from_rate(t_yr, angle, rate, windows, settings, frame=frame, with_windows=False)
        if potential is None or not np.isfinite(potential.r2):
            continue
        if best[2] is None or potential.r2 > best[2].r2:
            best = (name, frame, potential)
    if best[2] is None:
        return None, None, None, []
    potential, per_window = pendulum.fit_potential_from_rate(t_yr, angle, rate, windows, settings, frame=best[1])
    return best[0], best[1], potential, per_window


def _amplitude_max(geometry: Geometry) -> float:
    """Largest half-amplitude of the window's half-cycles [rad] (nan without turning points)."""
    values = half_amplitudes(geometry.turning_points)
    return float(np.max(values)) if len(values) else NAN


def r0_checks(
    potential: pendulum.Potential,
    period_law: float,
    coherence: float,
    n_windows: int,
    correlations: Optional[Sequence[float]],
    geometry: Geometry,
    prep: Preprocessed,
    settings: AnalysisSettings,
    moving_well: bool = False,
) -> Tuple[CheckValue, Optional[CheckValue]]:
    """(R0 sigma-only, R0 with a) of SPEC §4.6, measured only where the measurement means something (R0 with a is
    None when it depends on the V_E ~ V_F `correlations` and they were not given):

    - not_measured both under filter_too_slow (shape-dependent, DELTA item 1) and when V_E explains less than
      separatrix_ve_r2_min of sigma_dot^2 within the cycles (no potential was measured, so neither a period law nor a
      well can be tested against it: 469333 0.37, 78085 0.002);
    - R0 with a not_measured below mmr_a_response_min_amp_deg without a turn: there the identity alone makes
      c a_dot ~ -Omega^2 (sigma - sigma_s), so V_E ~ V_F holds for any oscillation carried by a (PHYSICS N5, the same
      reason L4 is not measured there);
    - in a frame with a moving well (`moving_well`: members or centre) the well's phase is the frame's own model, so the
      coherence of the per-cycle wells (the fixed-potential test: 0.95-0.99 against 0.71 for a kinematic swing,
      SEPARATRIX §4.1) is reported but not required; the period law and the R^2 of V_E carry the test (115978: R^2 0.95,
      law 1.01 over 38 cycles, residual coherence 0.83 under a 142-deg wobble of a composite multiplet)."""
    nm = CheckValue.NOT_MEASURED
    if prep.filter_too_slow or not potential.r2 >= settings.separatrix_ve_r2_min:
        return nm, nm
    if moving_well:  # the frame carries the motion of the well: the coherence of the residual wells is not the test
        coherence = 1.0 if n_windows >= 2 else NAN
    sigma_only = pendulum.sigma_only_check(period_law, coherence if n_windows >= 2 else NAN, settings)
    small = _amplitude_max(geometry) < np.radians(settings.mmr_a_response_min_amp_deg) and not geometry.revolutions
    if sigma_only != CheckValue.PASS:
        return sigma_only, sigma_only
    if small:
        return sigma_only, nm
    if correlations is None:  # the caller measures V_E ~ V_F only now that they decide
        return sigma_only, None
    return sigma_only, pendulum.with_slow_check(sigma_only, correlations, settings)


def _zone_of(
    modulation: Modulation,
    window: np.ndarray,
    omega_bar: float,
    adiabaticity: float,
    settings,
    cutoff_yr: float = NAN,
    potential: Optional[pendulum.Potential] = None,
) -> Tuple[Optional[Zone], float]:
    """(zone, seconds it took): the time scale Omega_bar from sigma (V_E), the pulsation from the modulation.

    No zone without a finite Omega_bar, nor when the small period 2 pi / Omega_bar is shorter than `cutoff_yr`, the
    smoothing of the sigma that V_E was fitted to: such a pendulum is faster than anything that series can show, the
    fit is degenerate, and integrating the record in its periods never ends (46162 6V-41: Omega_bar 1.1e6 rad/yr,
    a 6e-6-yr period over 50 kyr)."""
    if not (np.isfinite(omega_bar) and omega_bar > 0):
        return None, NAN
    if np.isfinite(cutoff_yr) and TWO_PI / omega_bar < cutoff_yr:
        return None, NAN
    began = time.perf_counter()
    method = ZoneMethod.NUMERIC if modulation.source == SOURCE_MEMBERS else ZoneMethod.EQ_ONLY
    relative = modulation.omega_rel[window]
    zone = build_zone(
        modulation.t_yr[window],
        (relative * omega_bar) ** 2,
        modulation.well_rad[window],
        adiabaticity,
        settings,
        modulation.modulation_period_yr,
        omega_bar,
        method,
        potential,
    )
    return zone, time.perf_counter() - began


def _zone_flags(modulation: Modulation, zone: Optional[Zone], edges: Sequence[bool], in_zone: bool) -> List[str]:
    """zone_eq_only (no members), zone_numeric_mismatch, zone_edge_cycle, episodes_in_separatrix_zone."""
    flags = []
    if modulation.source != SOURCE_MEMBERS:
        flags.append(str(Flag.ZONE_EQ_ONLY))
    if zone is not None and zone.mismatch:
        flags.append(str(Flag.ZONE_NUMERIC_MISMATCH))
    if any(edges):
        flags.append(str(Flag.ZONE_EDGE_CYCLE))
    if in_zone:
        flags.append(pendulum.EPISODES_IN_SEPARATRIX_ZONE)
    return flags


def analyse_separatrix(
    series: MMRSeries,
    prep: Preprocessed,
    slow: SlowComponents,
    geometry: Geometry,
    response: Response,
    context: Context,
    settings: AnalysisSettings,
    well: Optional[centre_track_module.WellAlongRecord] = None,
) -> Optional[EnergyAnalysis]:
    """V_E, V_F, R0 (sigma-only and with a), cycle energies, modulation, adiabaticity, zone, cycle classes,
    crossings and the mechanism (pulsation, layer, kick, trend, unexplained); None without a cycle or a turn.

    R0 sigma-only: the period law (median T_obs/T_pred over the cycles) and the coherence of the per-cycle wells, both
    in the frame that explains sigma_dot^2 best (`best_frame`). T_pred of a libration cycle is its period in the full
    averaged well when `well` exists (a two-body angle with its planet saved: `full_well_laws`) and the swing librates
    in it, else in the fitted V_E; a turn keeps V_E's. R0 with a: sigma-only passes and V_E ~ V_F cycle by
    cycle (`r0_checks` says when they are not measured). Both not_measured under
    filter_too_slow (shape-dependent, DELTA item 1). The flags: zone_eq_only (no members: e^q or no modulation),
    zone_numeric_mismatch, zone_edge_cycle, episodes_in_separatrix_zone."""
    t = np.asarray(prep.t_yr, float)
    bounds = cycle_bounds(geometry)
    if not bounds:
        return None
    period = record_period(geometry)
    angle, rate, a_s, adot, cutoff = separatrix_series(prep, period, settings)
    window = filtering.window_mask(t, prep.window)
    if np.count_nonzero(window) < 2:
        return None

    modulation = None if settings.mmr_zone_method == 'eq_only' else best_members(series, t, angle, adot, window, context, cutoff, settings)
    if modulation is None or not modulation.gain >= settings.mmr_force_multi_gain:
        modulation = eccentricity_modulation(series, slow, context, t)
    windows = [(start, end) for start, end, _ in bounds]
    frame_name, frame, v_e, per_window = best_frame(t, angle, rate, windows, candidate_frames(t, modulation, geometry), settings)
    if v_e is None:
        return None
    coherence = pendulum.well_coherence(per_window)
    cycles = pendulum.cycle_energies(t, angle, rate, bounds, v_e, frame=frame)
    fitted = [c.period_law_ratio for c in cycles]
    laws, law_source, n_full_well = full_well_laws(well, cycles, geometry, fitted)
    period_law = float(np.nanmedian(laws)) if np.any(np.isfinite(laws)) else NAN
    phi, _, strength = pendulum.in_frame(t, angle, rate, frame)
    v_f = pendulum.fit_potential_from_force(phi[window], adot[window], context.identity_coefficient, v_e.n_harmonics, strength[window])
    # V_E ~ V_F cycle by cycle (SEPARATRIX §5: 29 of 37 windows for 52356); the force windows of the response are
    # 3.5 cycles long and do not exist inside the short episodes of a chaotic record. Computed only when R0 with a
    # is decided by them (a fast circulation has tens of thousands of turns and no measured potential).
    moving = frame is not None and bool(np.any(frame.well_shift_rad != 0))
    sigma_only, with_slow = r0_checks(v_e, period_law, coherence, len(per_window), None, geometry, prep, settings, moving)
    correlations = []
    if with_slow is None:
        correlations = [_window_correlation(t, angle, rate, adot, w, frame, context.identity_coefficient, settings) for w in windows]
        with_slow = pendulum.with_slow_check(sigma_only, correlations, settings)
    finite = [r for r in correlations if np.isfinite(r)]

    # the zone: time scale Omega_bar from sigma (V_E), the pulsation from the modulation
    omega_bar = v_e.omega_rad_yr
    relative = modulation.omega_rel
    t_w = t[window]
    nu = modulation.nu_rad_yr
    adiabaticity = nu / omega_bar if np.isfinite(nu) and np.isfinite(omega_bar) and omega_bar > 0 else NAN
    eps_sum = float(np.nansum(modulation.eps)) if modulation.eps else NAN
    overlap = 2 * (1 + np.sqrt(eps_sum)) / adiabaticity if np.isfinite(eps_sum) and np.isfinite(adiabaticity) and adiabaticity > 0 else NAN
    zone, seconds = _zone_of(modulation, window, omega_bar, adiabaticity, settings, cutoff, v_e)

    h_values = [c.h for c in cycles]
    sigma_h = pendulum.energy_scatter(h_values, settings.separatrix_sigma_floor)
    classes = pendulum.classify_cycles(cycles, zone, sigma_h, settings, v_e) if zone is not None and np.isfinite(zone.j_lo) else []
    edges = pendulum.edge_cycles(cycles, zone, sigma_h, settings, v_e) if classes else []
    turning = geometry.turning_points
    # J/<J_s> of the turning points needs the record's mean J_s only when there is a zone edge to compare it with
    record = (float(t[0]), float(t[-1])) if zone is not None and np.isfinite(zone.j_lo) else None
    avgpot = tp_energies(well, geometry, cycles, settings, record)
    classes, edges, zone_source = averaged_classes(cycles, cycle_kinds(geometry, cycles), classes, edges, avgpot, zone)

    # crossings: changes between libration half-cycles and circulation events (slips, turns), from the angle itself;
    # the directed transitions (exits, captures) before their times are merged
    turns = [(start + end) / 2 for start, end, rotation in bounds if rotation]
    transitions = pendulum.regime_transitions(t, angle, turning.times_yr, turning.values_rad, turns)
    crossing_times = sorted(set(moment for moment, _ in transitions))
    weak = np.asarray(relative[window] < 1.0)
    permutation_p = NAN
    if crossing_times and modulation.source == SOURCE_MEMBERS and np.any(weak) and not np.all(weak):
        permutation_p = pendulum.cyclic_shift_p_value(crossing_times, weak, t_w)
    kicks = _kick_times(t, a_s, prep, cycles, sigma_h, float(np.median(np.diff(t))), period, settings) if crossing_times else []
    mechanism = (
        _mechanism(crossing_times, cycles, zone, adiabaticity, permutation_p, kicks, period, settings)
        if zone
        else (SeparatrixMechanism.UNEXPLAINED if crossing_times else SeparatrixMechanism.NONE)
    )
    in_zone = episodes_in_zone(crossing_times, classes)

    flags = _zone_flags(modulation, zone, edges, in_zone)
    h_finite = [h for h in h_values if np.isfinite(h)]
    shift = modulation.frame().well_shift_rad[window]
    return EnergyAnalysis(
        potential_e=v_e,
        potential_f=v_f,
        ve_vf_agree_fraction=float(np.mean(np.asarray(finite) >= settings.separatrix_vevf_r_min)) if finite else NAN,
        well_coherence=coherence,
        cycles=cycles,
        h_med=float(np.median(h_finite)) if h_finite else NAN,
        h_p10=_percentile(h_values, 10),
        h_p90=_percentile(h_values, 90),
        sigma_h=sigma_h,
        period_law_med=period_law,
        period_law_p10=_percentile(laws, 10),
        period_law_p90=_percentile(laws, 90),
        period_law_source=law_source,
        period_law_n_full_well=n_full_well,
        period_law_fitted_med=float(np.nanmedian(fitted)) if np.any(np.isfinite(fitted)) else NAN,
        eps=tuple(modulation.eps),
        omega_min_rel=float(np.min(relative[window])),
        omega_max_rel=float(np.max(relative[window])),
        adiabaticity=adiabaticity,
        overlap_s=overlap,
        eq_dk_over_k=modulation.eq_dk_over_k,
        e_trend_per_kyr=modulation.e_trend_per_kyr,
        zone=zone,
        cycle_classes=classes,
        crossing_times_yr=list(crossing_times),
        transitions=list(transitions),
        mechanism=mechanism,
        permutation_p=permutation_p,
        sigma_only=sigma_only,
        with_slow=with_slow,
        flags=flags,
        modulation_source=modulation.source,
        frame=frame_name or '',
        modulation_planet=modulation.planet,
        modulation_gain=modulation.gain,
        modulation_period_yr=modulation.modulation_period_yr,
        omega_bar_rad_yr=omega_bar,
        well_wobble_rad=float(np.ptp(shift) / 2) if len(shift) else NAN,
        zone_seconds=seconds,
        episodes_in_separatrix_zone=in_zone,
        model_frame=frame,
        series_cutoff_yr=cutoff,
        cycle_kinds=cycle_kinds(geometry, cycles),
        zone_source=zone_source,
        h_avgpot=tuple(float(h) for h in avgpot.h[np.isfinite(avgpot.h)]),
        j_avgpot=tuple(float(j) for j in avgpot.j[np.isfinite(avgpot.j)]),
    )


def episodes_in_zone(crossing_times: Sequence[float], classes: Sequence[CycleClass]) -> bool:
    """`episodes_in_separatrix_zone`: the angle crosses and every cycle lies in the model's zone. A cycle not measured
    (a loop around several wells, W25) is not known to be in the zone, so it cannot satisfy 'every cycle'."""
    return bool(crossing_times) and bool(classes) and all(CycleClass(c) == CycleClass.ZONE for c in classes)


@dataclass(frozen=True)
class TurningPointEnergies:
    """h and J/<J_s> of the turning points of the librating episodes in the averaged well
    (`centre_track.turning_point_energies`): their times and values (nan where not measured; J/<J_s> also nan for a
    loop around several wells of the profile, and for all of them when the record's mean <J_s> is not measured). h is
    in the units of its own moment (0 at the bottom, 1 at the barrier); J/<J_s> is the action over the record's mean
    separatrix action, the normalisation of the zone's J_lo."""

    t_yr: np.ndarray
    h: np.ndarray
    j: np.ndarray


def tp_energies(
    well: Optional[centre_track_module.WellAlongRecord],
    geometry: Geometry,
    cycles: Sequence[pendulum.CycleEnergy],
    settings: AnalysisSettings,
    record_yr: Optional[Tuple[float, float]] = None,
) -> TurningPointEnergies:
    """The turning points of the episode cycles (`cycle_kinds`) with their h and J/<J_s> in the averaged well at their
    own time; empty without a well (three-body, 1:1, the planet not saved) or without such cycles.

    J/<J_s> = (J/J_s(t)) x J_s(t) / <J_s>: the ratio in the profile at the turning point's time times that moment's
    separatrix action (the profile's scale included) over its time mean on `record_yr` (`centre_track.
    mean_separatrix_action`, the reference of J_lo). The action of a libration scales as the square root of the well's
    strength, so in a well at a quarter of its mean strength J/J_s(t) = 0.6 is J/<J_s> = 0.3 (Codex W25 #3). J/<J_s>
    is all nan without `record_yr` (no zone to compare with: the mean is not computed) or when the mean is not
    measured."""
    turning = geometry.turning_points
    times, values = np.asarray(turning.times_yr, dtype=float), np.asarray(turning.values_rad, dtype=float)
    empty = TurningPointEnergies(np.zeros(0), np.zeros(0), np.zeros(0))
    if well is None or not well.available or not len(times):
        return empty
    inside = np.zeros(len(times), dtype=bool)
    for cycle, kind in zip(cycles, cycle_kinds(geometry, cycles)):
        if kind == CYCLE_EPISODE:
            inside |= (times >= cycle.t_start_yr) & (times <= cycle.t_end_yr)
    if not np.any(inside):
        return empty
    index = np.flatnonzero(inside)
    # the swing of each turning point to its neighbour in the record (the next, the previous for the last)
    neighbour = np.where(index + 1 < len(times), index + 1, index - 1)
    middles = (values[index] + values[neighbour]) / 2
    h, j_local, j_s = centre_track_module.turning_point_energies(well, times[index], values[index], settings, middles)
    reference = NAN
    if record_yr is not None and np.any(np.isfinite(j_local) & np.isfinite(j_s)):
        reference = centre_track_module.mean_separatrix_action(well, record_yr[0], record_yr[1], times[index], middles, settings)
    j = j_local * j_s / reference if np.isfinite(reference) and reference > 0 else np.full(len(index), NAN)
    return TurningPointEnergies(times[index], h, j)


def averaged_classes(
    cycles: Sequence[pendulum.CycleEnergy],
    kinds: Sequence[str],
    classes: Sequence[CycleClass],
    edges: Sequence[bool],
    energies: TurningPointEnergies,
    zone: Optional[Zone],
) -> Tuple[List[CycleClass], List[bool], str]:
    """(classes, edges, source) of the cycles with the action of a two-body angle's turning points in its averaged well
    (W20 F5, the zone's source 'averaged_potential'): an episode cycle is in the zone when the largest J/<J_s> of its
    turning points reaches the zone's J_lo, deep when every one of them is measured and below it; its edge flag is
    off (J/J_s is measured at each turning point, not fitted, so the fit's scatter sigma_h does not apply). A cycle with
    a turning point not measured (nan: a failed profile, or a loop around several wells of the profile, which has no
    single-well ratio) and none at J_lo keeps the pendulum's class and edge: the missing one may be the high one (Codex
    W20 A #5). The fitted pendulum's h assumes fixed parameters, and a well whose depth changes 100 times breaks it:
    2004 TK14 31M-34 has h ~ 1.3 there and h 0.07 (max 0.59) in the averaged well, at least 80 deg from the barrier at
    every turning point (B, 2026-10-19). Turns and the cycles of rejected runs keep the pendulum's class. The
    pendulum's classes and source 'pendulum' without a measured J/J_s or a zone.

    Actions, not energies (W25): J/J_s of a turning point is computed in the averaged profile at its own time
    (`averaged_potential.action_in_well`), J_s at that well's own barrier. Mapping J_lo to an energy through V_E's
    action-energy law misreads another well's shape (V = -cos s and V = -cos s - 0.2 cos 2s have J/J_s = 0.5 at
    h = 0.5808 and 0.6278). J_lo is J0/<J_s>, the action over the record's mean separatrix action, so the ratio is
    carried to that normalisation by J_s(t)/<J_s> of the averaged well itself (`tp_energies`, `energies.j`): a well whose
    depth changes is not read in its own units at each time (Codex W25 #3). Without the mean (a profile on the record's
    grid not measured) every J/<J_s> is nan and the pendulum's classes stay."""
    classes, edges = list(classes), list(edges)
    if not classes or zone is None or not np.isfinite(zone.j_lo) or not np.any(np.isfinite(energies.j)):
        return classes, edges, SOURCE_PENDULUM
    changed = False
    for k, (cycle, kind) in enumerate(zip(cycles, kinds)):
        j = energies.j[(energies.t_yr >= cycle.t_start_yr) & (energies.t_yr <= cycle.t_end_yr)]
        finite = j[np.isfinite(j)]
        if kind != CYCLE_EPISODE or cycle.rotation or not len(finite):
            continue
        if float(np.max(finite)) >= zone.j_lo:
            classes[k] = CycleClass.ZONE
        elif len(finite) == len(j):
            classes[k] = CycleClass.DEEP
        else:
            continue
        edges[k] = False
        changed = True
    return classes, edges, SOURCE_AVERAGED_POTENTIAL if changed else SOURCE_PENDULUM


def full_well_laws(
    well: Optional[centre_track_module.WellAlongRecord],
    cycles: Sequence[pendulum.CycleEnergy],
    geometry: Geometry,
    fitted: Sequence[float],
) -> Tuple[List[float], str, int]:
    """(T_obs / T_pred per cycle, source, cycles read in the full well): T_pred of each cycle of a librating episode in
    the full averaged well (`centre_track.full_well_periods`) where it exists there, else in the fitted V_E (`fitted`):
    a turn (no turning point gives its energy), a cycle of a rejected run (no libration to model: the record's
    oscillation noise, thousands of cycles in a circulation), a swing beside the theory's well (the strength check
    reports that contradiction, F), a period not measured. Source 'full_well' when some cycle was read there.

    The full well knows the shape (a flat-bottomed well is no cosine: 2003 UG22 10E-17 0.77 in V_E, 0.95 here) and the
    scale (absolute: R0 then reads the theory's strength too and is not independent of F)."""
    if well is None or not well.available:
        return list(fitted), 'fitted', 0
    turning = geometry.turning_points
    kinds = cycle_kinds(geometry, cycles)
    bounds = [(c.t_start_yr, c.t_end_yr, c.rotation or kind != CYCLE_EPISODE) for c, kind in zip(cycles, kinds)]
    predicted = centre_track_module.full_well_periods(well, bounds, turning.times_yr, turning.values_rad)
    laws, used = [], 0
    for cycle, law, period in zip(cycles, fitted, predicted):
        if np.isfinite(period) and period > 0:
            laws.append((cycle.t_end_yr - cycle.t_start_yr) / period)
            used += 1
        else:
            laws.append(law)
    return laws, 'full_well' if used else 'fitted', used


def energies_at_cutoff(
    prep: Preprocessed, geometry: Geometry, energy: EnergyAnalysis, cutoff_yr: float, settings: AnalysisSettings
) -> List[pendulum.CycleEnergy]:
    """The cycle energies of `geometry`'s cycles with sigma taken at the filter cutoff `cutoff_yr` (then smoothed as
    `separatrix_series` does), in the potential and frame of `energy`: the filter guard's energy per cycle (DELTA 13a)."""
    if energy is None or energy.potential_e is None:
        return []
    t = np.asarray(prep.t_yr, float)
    smoothing = separatrix_cutoff_yr(cutoff_yr, record_period(geometry), settings)
    angle = filtering.lowpass_or_none(t, prep.angle_clean, smoothing, settings)
    if angle is None:
        return []
    rate = filtering.time_derivative(t, angle)
    return pendulum.cycle_energies(t, angle, rate, cycle_bounds(geometry), energy.potential_e, frame=energy.model_frame)
