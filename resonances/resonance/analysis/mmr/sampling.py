"""Output step, alias branches and the predictive unwrap (SPEC §3.1, §4.2; SAMPLING.md).

The angle mod 2pi is exact at every output point; only np.unwrap errs, adding the increment of
smallest modulus. The predictive unwrap adds the increment closest to the rate model instead:

  sigma_i = sigma_{i-1} + nu_i dt_i + wrap(sigma_i^mod - sigma_{i-1} - nu_i dt_i),  wrap in [-pi, pi),
  nu_i = sum_j k_j n_j + k n((a_{i-1} + a_i)/2) + slow.

The residual wrap(...) is in [-pi, pi) by construction: a model error of 230 deg looks like -130
deg and silently adds 2 pi, so the alias confidence rests on the branch margin and on
far_from_resonance (computed from a, not from the unwrap).

Far from the resonance (`far_ratio`) is read from the data, not from a strength: a libration about a*
needs sigma_dot = c (a - a*) to change sign, so the slow a must reach a*. far_ratio = |<a> - a*| / s with
s = max(half the a range of the record, |a(lambda_dot) - <a>|): the body's own a excursion, or the error
of the Keplerian n(a) of the rate model when the saved mean longitude says it is larger (a body at the
resonance has lambda_dot ~ n*, so a(lambda_dot) ~ a* and the ratio stays ~1). mu e^q is not a bound on the
half-width (no coefficient of the disturbing function in it) and does not enter.

Branches: m = round(nu dt / 2pi) (round half up), the apparent rate np.unwrap draws is nu - 2 pi m / dt
(Knezevic & Milani 2001, p. 22), and a branch m looks like an exact resonance ("ghost") at a_m with
nu(a_m) = 2 pi m / dt.

The alias question is one of the lambda part only (W15). sigma = psi_lambda + s, s = p varpi + v Omega + sum_j (p_j
varpi_j + v_j Omega_j): the slow elements are sampled exactly, but where e passes near 0 the body's varpi turns by up
to pi between two samples at any step, which no finer step cures and which is no alias. So psi_lambda = sigma - s
(mod 2pi) is unwrapped by the rate model without the slow term and judged alone (unwrap_ok, p99/max, suspects); a step
where only s turned fast is the flag varpi_fast_low_e (an excess step of the slow elements; e near 0 is its usual
cause, not a measured fact). The body's varpi in s takes its winding from the low-passed vector
z_f = <e exp(i varpi)> (band mmr_jitter_band_yr, the fast-jitter band of jitter_low_e), not from np.unwrap: a fast
loop of z around the origin winds the osculating varpi once per loop, z_f not at all. Terms longer than the band
(a 600-yr synodic e wave of a TNO) stay in z_f: this is a fast-jitter diagnostic, not a proper-element filter.
Where |p| x the local RMS of z - z_f exceeds mmr_jitter_max_rad |z_f| (the record-wide jitter_low_e test, taken
locally: the precision of the resonant argument, not of varpi alone) the phase is undefined there
(phase_undefined_local; columns only, one band from either edge not measured).

undefined_phase (W20, 258640 46J-19+0-27) means varpi itself is undefined over most of the record: jitter_low_e, the
full angle failing the residual limits, and varpi's own (p = 1) scatter s/|z_f| above mmr_jitter_max_rad on more than
mmr_varpi_undefined_max_share of the measured samples (varpi_undefined_fraction; when it is not measured, as before
W20). jitter_low_e alone is the osculating short-period wobble of p varpi, resolved and measured: a flag and a cap.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.geometry import wrap_pi
from resonances.resonance.analysis.mmr.context import (
    axis_from_mean_motion,
    body_angles,
    expected_rate,
    linear_slope,
    mean_motion,
    predictive_unwrap,
)
from resonances.resonance.analysis.mmr.models import Context, MMRSeries, ResonanceSpec, Sampling, StepVerdict
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Flag

TWO_PI = 2 * math.pi
NAN = float('nan')
DEFAULT_LINE_BAND_FACTOR = AnalysisSettings.__dataclass_fields__['mmr_aliased_line_band_factor'].default


def _turns(rate_rad_yr, step_yr: float):
    return np.asarray(rate_rad_yr, dtype=float) * step_yr / TWO_PI


def branch_index(rate_rad_yr, step_yr: float):
    """m = round(nu dt / 2pi): the alias branch of a rate at a step. Scalar or array (half-turns round up)."""
    m = np.floor(_turns(rate_rad_yr, step_yr) + 0.5).astype(int)
    return int(m) if np.ndim(m) == 0 else m


def apparent_rate(rate_rad_yr, step_yr: float):
    """nu - 2 pi m / dt [rad/yr]: the rate np.unwrap shows (Knezevic & Milani 2001, p. 22); nan for an unknown rate."""
    branch = np.floor(_turns(rate_rad_yr, step_yr) + 0.5)  # branch_index in floats: nan stays nan
    rate = np.asarray(rate_rad_yr, dtype=float) - TWO_PI * branch / step_yr
    return float(rate) if np.ndim(rate) == 0 else rate


def _ordered(rate_lo_rad_yr: float, rate_hi_rad_yr: float) -> Tuple[float, float]:
    return min(rate_lo_rad_yr, rate_hi_rad_yr), max(rate_lo_rad_yr, rate_hi_rad_yr)


def step_rotation_range(rate_lo_rad_yr: float, rate_hi_rad_yr: float, step_yr: float) -> Tuple[float, float]:
    """(min, max) of |nu| dt [rad] over the rate range [lo, hi] (min = 0 when the range contains 0)."""
    lo, hi = _ordered(rate_lo_rad_yr, rate_hi_rad_yr)
    smallest = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
    return smallest * step_yr, max(abs(lo), abs(hi)) * step_yr


def step_verdict(rate_lo_rad_yr: float, rate_hi_rad_yr: float, step_yr: float, resolved_rad: float) -> Tuple[StepVerdict, Tuple[int, ...]]:
    """Verdict over the a range: resolved (|nu| dt <= resolved_rad), marginal (<= pi, one branch),
    aliased_unambiguous (one branch m != 0), ambiguous (several). Returns (verdict, branches)."""
    lo, hi = _ordered(rate_lo_rad_yr, rate_hi_rad_yr)
    branches = tuple(range(branch_index(lo, step_yr), branch_index(hi, step_yr) + 1))
    _, rotation_max = step_rotation_range(lo, hi, step_yr)
    if len(branches) > 1:
        return StepVerdict.AMBIGUOUS, branches
    if branches[0] != 0:
        return StepVerdict.ALIASED_UNAMBIGUOUS, branches
    if rotation_max <= resolved_rad:
        return StepVerdict.RESOLVED, branches
    return StepVerdict.MARGINAL, branches


def branch_margin_turns(rate_lo_rad_yr: float, rate_hi_rad_yr: float, step_yr: float) -> float:
    """Distance (turns per step) from nu dt / 2pi over [lo, hi] to the nearest half-integer; 0 when one is crossed."""
    lo, hi = _ordered(rate_lo_rad_yr, rate_hi_rad_yr)
    if branch_index(lo, step_yr) != branch_index(hi, step_yr):
        return 0.0
    m = branch_index(lo, step_yr)
    return float(min(_turns(lo, step_yr) - (m - 0.5), (m + 0.5) - _turns(hi, step_yr)))


def nyquist_halfwidth(a_au: float, body_coefficient: int, step_yr: float) -> float:
    """delta a_N = a^(5/2) / (3 |k| dt) [au]: the a offset that turns the angle by pi per step."""
    if body_coefficient == 0:
        return math.inf
    return float(a_au**2.5 / (3 * abs(body_coefficient) * step_yr))


def branch_distances(rate_rad_yr: float, identity_coefficient: float, step_yr: float) -> Tuple[float, float]:
    """(Delta a to the nearer branch boundary nu = 2 pi (m +- 1/2) / dt, Delta a between neighbouring branches
    2 pi / (dt |c|)) [au] at the rate nu of <a> (NUMBERS §2.5 b, c)."""
    if identity_coefficient == 0 or not math.isfinite(rate_rad_yr):
        return float('nan'), float('nan')
    m = branch_index(rate_rad_yr, step_yr)
    boundary = min(abs(rate_rad_yr - TWO_PI * (m - 0.5) / step_yr), abs(rate_rad_yr - TWO_PI * (m + 0.5) / step_yr))
    return boundary / abs(identity_coefficient), TWO_PI / (step_yr * abs(identity_coefficient))


def ghost_axis(spec: ResonanceSpec, planet_n: Mapping[str, float], slow_rate_rad_yr: float, step_yr: float, branch: int) -> float:
    """a_m where branch m looks like an exact resonance: nu(a_m) = 2 pi m / dt, i.e.
    n(a_m) = n_res + 2 pi m / (k dt), n_res = -(sum_j k_j n_j + slow) / k [au]; nan if k = 0 or n(a_m) <= 0.
    At a_m, branch_index(nu) = m, so `nearest_ghost` finds the ghost of the body's own branch.
    """
    k = spec.body_coefficient
    if k == 0:
        return float('nan')
    rate_without_body = expected_rate(spec, 1.0, planet_n, slow_rate_rad_yr) - k * float(mean_motion(1.0))
    n_m = (TWO_PI * branch / step_yr - rate_without_body) / k
    return float(axis_from_mean_motion(n_m)) if n_m > 0 else float('nan')


def far_ratio(context: Context) -> float:
    """|<a> - a*| / max(half the a range [a_lo, a_hi], |a(lambda_dot) - <a>|) (module docstring).

    inf when the combination has no zero at any a > 0 (`no_commensurability`) or when the scale is 0 (a constant
    a away from a*); nan when a* is unknown otherwise, or when <a> = a* with a zero scale.
    """
    if not math.isfinite(context.exact_axis_au):
        return math.inf if no_commensurability(context) else float('nan')
    distance = abs(context.a_mean_au - context.exact_axis_au)
    scales = [0.5 * (context.a_p99_au - context.a_p01_au), abs(context.axis_from_mean_longitude_rate_au - context.a_mean_au)]
    scale = max((value for value in scales if math.isfinite(value)), default=float('nan'))
    if not math.isfinite(distance) or not math.isfinite(scale):
        return float('nan')
    if scale > 0:
        return distance / scale
    return math.inf if distance > 0 else float('nan')


def rate_model(spec: ResonanceSpec, a_au: np.ndarray, planet_n: Mapping[str, float], slow_rate_rad_yr: float) -> np.ndarray:
    """nu_i at the midpoints (a_{i-1} + a_i)/2, length N-1 [rad/yr]."""
    a_au = np.asarray(a_au, dtype=float)
    return np.asarray(expected_rate(spec, 0.5 * (a_au[1:] + a_au[:-1]), planet_n, slow_rate_rad_yr), dtype=float)


def unwrap_planet_longitude(t_yr: np.ndarray, longitude_wrapped: np.ndarray, mean_motion_rad_yr: float) -> np.ndarray:
    """A planet's lambda unwrapped with the constant model rate n_j [rad]."""
    return predictive_unwrap(t_yr, longitude_wrapped, mean_motion_rad_yr)[0]


def short_period_scatter(t_yr: np.ndarray, values: np.ndarray, band_yr: float, filter_order: int) -> float:
    """Std of `values` at periods shorter than `band_yr`: values minus their Butterworth low-pass at band_yr.

    When the band is at or above the Nyquist frequency of the step (2 dt >= band_yr), every sampled
    variation is short-period: the std of the first differences / sqrt(2) (white-noise estimate).
    """
    values = np.asarray(values, dtype=float)
    step = float(np.median(np.diff(np.asarray(t_yr, dtype=float))))
    normalised_cutoff = 2 * step / band_yr
    if normalised_cutoff < 1:
        try:
            sos = signal.butter(filter_order, normalised_cutoff, output='sos')
            return float(np.std(values - signal.sosfiltfilt(sos, values)))
        except ValueError:  # record too short for filtfilt padding
            pass
    return float(np.std(np.diff(values)) / math.sqrt(2))


def perihelion_jitter(t_yr: np.ndarray, e: np.ndarray, varpi_coefficient: int, band_yr: float, filter_order: int) -> float:
    """|p| de_sp / mean e, de_sp = std of e above the band (periods < band_yr): varpi noise at small e [rad]."""
    if varpi_coefficient == 0:
        return 0.0
    e_mean = float(np.mean(e))
    if e_mean <= 0:
        return math.inf
    return abs(varpi_coefficient) * short_period_scatter(t_yr, e, band_yr, filter_order) / e_mean


def revolutions(angle_unwrapped: np.ndarray) -> float:
    """(last - first) / 2pi."""
    if len(angle_unwrapped) < 2:
        return float('nan')
    return float((angle_unwrapped[-1] - angle_unwrapped[0]) / TWO_PI)


def safe_steps(
    rate_lo_rad_yr: float, rate_hi_rad_yr: float, a_au: float, body_coefficient: int, da_short_au: float, eta: float
) -> Tuple[float, float]:
    """(plain: eta pi / max|nu|, predictive: eta a^(5/2) / (3 |k| da_sp)) [yr]; eta = mmr_step_resolved_deg / 180 deg,
    the share of pi a step may turn the angle and still be resolved (the dry-run uses the same)."""
    largest = max(abs(rate_lo_rad_yr), abs(rate_hi_rad_yr))
    plain = eta * math.pi / largest if largest > 0 else math.inf
    if body_coefficient == 0 or da_short_au == 0:
        return plain, math.inf
    return plain, eta * a_au**2.5 / (3 * abs(body_coefficient) * da_short_au)


def resample_step(
    safe_step_plain_yr: float,
    safe_step_pred_yr: float,
    period_cycle_yr: float,
    step_yr: float,
    settings: AnalysisSettings,
    cutoff_yr: Optional[float] = None,
) -> Optional[float]:
    """The step to integrate again with [yr]: the largest of `mmr_resample_steps_yr` <= min(safe_plain, safe_pred,
    P_cyc / mmr_resample_steps_per_cycle) and < cutoff_yr / 2 (the filter must exist at that step,
    `preprocess.filter_realizable`; an explicit MMR_FILTER_CUTOFF_YR can be shorter than the period bound), finer than
    the record's `step_yr` by construction; None when the record's step already meets that bound (a finer step cures
    nothing: a person looks) or no round value fits.

    safe_plain is the alias condition (|nu| dt <= mmr_step_resolved_deg over the a range: one branch, m = 0), which an
    aliased record breaks; without it a smooth a (large safe_pred) recommended 10 yr for a record at 0.2 yr (99942).
    An unknown (nan) bound is ignored."""
    filterable = math.nextafter(cutoff_yr / 2, 0.0) if cutoff_yr is not None and cutoff_yr > 0 else math.nan
    bounds = (safe_step_plain_yr, safe_step_pred_yr, period_cycle_yr / settings.mmr_resample_steps_per_cycle, filterable)
    largest = min((value for value in bounds if not math.isnan(value)), default=math.inf)
    if step_yr <= largest:
        return None
    fitting = [step for step in settings.mmr_resample_steps_yr if step <= largest]
    return float(max(fitting)) if fitting else None


def short_period_step(orbital_period_yr: float, synodic_periods_yr: Mapping[str, float]) -> float:
    """min(P_orb, P_syn of the angle's planets) / 2 [yr], the Nyquist step of the fastest terms the elements carry
    (inf when none is known). At or above it those terms fold into slow lines (11.9-yr terms of a Trojan at 15.9 yr
    look like 47 yr) that no filter of the record removes and the record cannot measure: flag short_period_aliased,
    confidence <= medium. Their size is set by the amplitude of the folded terms (0.1-0.3 deg in sigma of the Trojans,
    12.5 deg in 62243 1M-4J-1 at 16 yr, inv-09), which only a finely sampled pilot measures (dry-run)."""
    periods = [float(p) for p in (orbital_period_yr, *synodic_periods_yr.values()) if p is not None and math.isfinite(p) and p > 0]
    return 0.5 * min(periods) if periods else math.inf


def folded_slowest_yr(orbital_period_yr: float, synodic_periods_yr: Mapping[str, float], step_yr: float, harmonics: int) -> float:
    """The longest period [yr] into which the step folds m / P, m = 1..`harmonics` (MMR_SHORT_PERIOD_HARMONICS), P =
    P_orb and the P_syn of the angle's planets: |m/P - round(m dt / P)/dt|^-1 for every line above the Nyquist
    frequency (m dt / P >= 1/2; a slower one is sampled, not folded: the synodic period of a co-orbital); inf for a line
    folded onto zero frequency (the search stops there), 0 when none is folded, nan when no period is known or
    `harmonics` is 0. A line folded slower than the angle's period floor sits in sigma_f's and a_f's own band, where no
    filter of the record removes it (`analyse.short_period_caps`). The count is the setting's: the amplitude of a
    harmonic is not bounded from the record (W20 round D: the folded lines in a are the planets' synodic harmonics,
    whatever e; the e^m order of round C is gone)."""
    periods = [float(p) for p in (orbital_period_yr, *synodic_periods_yr.values()) if p is not None and math.isfinite(p) and p > 0]
    if not periods or harmonics < 1 or not (math.isfinite(step_yr) and step_yr > 0):
        return NAN
    slowest = 0.0
    for period in periods:
        for m in range(1, harmonics + 1):
            frequency = m / period
            if frequency * step_yr < 0.5:
                continue
            folded = abs(frequency - math.floor(frequency * step_yr + 0.5) / step_yr)
            if folded == 0:
                return math.inf
            slowest = max(slowest, 1 / folded)
    return slowest


SHORT_PERIOD_NYQUIST = 'nyquist'  # dt >= min(P_orb, P_syn) / 2
SHORT_PERIOD_HARMONIC = 'harmonic'  # a harmonic m <= MMR_SHORT_PERIOD_HARMONICS folded slower than the period floor


def harmonic_folds(folded_slowest: float, period_floor_yr: float) -> bool:
    """A harmonic folded slower than the angle's period floor (`folded_slowest_yr` > P_floor): it sits in sigma_f's and
    a_f's own band. False when no line is folded (0), the count is 0 (nan) or there is no floor (0 = off)."""
    return bool(folded_slowest > 0 and period_floor_yr > 0 and folded_slowest > period_floor_yr)


def short_period_trigger(step_yr: float, short_step_yr: float, folded_slowest: float, period_floor_yr: float) -> str:
    """The one predicate of "the step folds short periods" (W20 round E, review D #3), read by the analysis
    (`analyse.short_period_folds`) and the dry-run (`dry_run._row_at_step`, hence its expected flag and the pilot's
    acceptance): SHORT_PERIOD_NYQUIST when dt >= min(P_orb, P_syn) / 2 (`short_period_step`), else
    SHORT_PERIOD_HARMONIC when a harmonic m = 1..MMR_SHORT_PERIOD_HARMONICS folds slower than the period floor
    (`folded_slowest_yr`, `harmonic_folds`), else ''. Only m <= MMR_SHORT_PERIOD_HARMONICS (2) is searched: the
    amplitude of a higher harmonic is not bounded from the record, so the count is a convention (P_orb 12 yr at 4.002 yr
    folds its third harmonic to 8 kyr and is not flagged at the default). The two sides agree for equal inputs only:
    the static dry-run takes P_orb(a0), the analysis P_orb(<a>) (`context`), and the pilot replans with the measured
    <a>; a fold near the floor can fall on either side of it (a0 = 13.1^(2/3), <a> = 13^(2/3) au, dt 5.999 yr, 1:1
    with Jupiter: 71.3 yr against 77.8 yr, floor 73.9 yr)."""
    if step_yr >= short_step_yr:
        return SHORT_PERIOD_NYQUIST
    return SHORT_PERIOD_HARMONIC if harmonic_folds(folded_slowest, period_floor_yr) else ''


def fold_free_step(
    orbital_period_yr: float,
    synodic_periods_yr: Mapping[str, float],
    step_yr: float,
    period_floor_yr: float,
    steps_yr: Sequence[float],
    harmonics: int,
) -> float:
    """smp_fold_free_step_yr (W21 A, the author's Q1 of 23.10): the largest of `steps_yr` (MMR_RESAMPLE_STEPS_YR, the
    grid every resample step of the analysis comes from; at and below 10 yr the dry-run's DRYRUN_AUTO_STEPS_YR) below
    the record's step at which `short_period_trigger` is '' with the record's P_orb, P_syn and period floor: a rerun at
    that output step folds no short period, so the cap short_period_aliased does not arise there. The walk goes down
    the grid and takes the first free step (as `dry_run.fold_free_step` does for a whole run): a harmonic folds slower
    than the floor only near m dt / P = j, so a smaller step is not always freer. nan when the record's step folds
    nothing or no grid step below it is free."""
    if not (math.isfinite(step_yr) and step_yr > 0):
        return NAN
    short_step = short_period_step(orbital_period_yr, synodic_periods_yr)

    def trigger(step: float) -> str:
        folded = folded_slowest_yr(orbital_period_yr, synodic_periods_yr, step, harmonics)
        return short_period_trigger(step, short_step, folded, period_floor_yr)

    if not trigger(step_yr):
        return NAN
    for step in sorted((float(s) for s in steps_yr if s < step_yr), reverse=True):
        if not trigger(step):
            return step
    return NAN


def aliased_lines(
    synodic_periods_yr: Mapping[str, float],
    step_yr: float,
    period_cycle_yr: float,
    harmonics: int,
    band_factor: Optional[float] = None,
) -> Tuple[float, ...]:
    """Periods |m/P_syn - round(m dt / P_syn)/dt|^-1 [yr], m = 1..harmonics, that fall into [P_cyc/f, f P_cyc],
    f = `band_factor` (default MMR_ALIASED_LINE_BAND_FACTOR)."""
    if not math.isfinite(period_cycle_yr):
        return ()
    factor = DEFAULT_LINE_BAND_FACTOR if band_factor is None else band_factor
    lines = []
    for period in synodic_periods_yr.values():
        for m in range(1, harmonics + 1):
            frequency = m / period
            if frequency * step_yr < 0.5:
                continue
            folded = abs(frequency - math.floor(frequency * step_yr + 0.5) / step_yr)
            aliased_period = 1 / folded if folded > 0 else math.inf
            if period_cycle_yr / factor <= aliased_period <= factor * period_cycle_yr:
                lines.append(aliased_period)
    return tuple(sorted(lines))


def _reconstructed_angle(series: MMRSeries) -> Optional[np.ndarray]:
    """sigma from the saved longitudes (the body's in the convention of the angle); None if any is missing."""
    spec = series.resonance
    planets = [series.planets.get(name) for name in spec.planets]
    if any(p is None or p.mean_longitude is None for p in planets):
        return None
    body_varpi, body_lambda = body_angles(series)
    if body_lambda is None:
        return None
    sigma = spec.body_coefficient * body_lambda + spec.body_varpi_coefficient * body_varpi + spec.body_node_coefficient * series.Omega
    for index, planet in enumerate(planets):
        sigma = sigma + spec.planet_coefficients[index] * planet.mean_longitude
        sigma = sigma + spec.planet_varpi_coefficients[index] * planet.varpi
        sigma = sigma + spec.planet_node_coefficient(index) * planet.Omega
    return sigma


def angle_reconstruction_error(series: MMRSeries) -> float:
    """max |sigma_saved - sum coef x saved longitudes| mod 2pi after removing the constant offset [rad]; nan without planets/lambda."""
    sigma = _reconstructed_angle(series)
    if sigma is None:
        return float('nan')
    difference = wrap_pi(np.asarray(series.angle_wrapped, dtype=float) - sigma)
    offset = float(np.angle(np.mean(np.exp(1j * difference))))
    return float(np.max(np.abs(wrap_pi(difference - offset))))


def no_commensurability(context: Context) -> bool:
    """The angle's rate nu(a) = nu(<a>) + k (n(a) - n(<a>)) has no zero at any a > 0: n* = n(<a>) - nu(<a>) / k <= 0
    (16397 29M+50+0-79: both coefficients positive). False when k = 0 or the rates are unknown."""
    k, rate, n = context.body_coefficient, context.rate_expected_rad_yr, context.mean_motion_rad_yr
    if k == 0 or not (math.isfinite(rate) and math.isfinite(n)):
        return False
    return n - rate / k <= 0


def _alias_columns(context: Context, step_yr: float, settings: AnalysisSettings) -> dict:
    """Branch verdict, margins, far-from-resonance and the nearest ghost: from a, not from the unwrap."""
    lo, hi, rate = context.rate_expected_lo_rad_yr, context.rate_expected_hi_rad_yr, context.rate_expected_rad_yr
    verdict, branches = step_verdict(lo, hi, step_yr, math.radians(settings.mmr_step_resolved_deg))
    rotation_min, rotation_max = step_rotation_range(lo, hi, step_yr)
    boundary_da, next_branch_da = branch_distances(rate, context.identity_coefficient, step_yr)
    far = far_ratio(context)
    return {
        'step_rotation_min_rad': rotation_min,
        'step_rotation_max_rad': rotation_max,
        'branches': branches,
        'verdict': verdict,
        'branch_margin_turns': branch_margin_turns(lo, hi, step_yr),
        'nyquist_boundary_da_au': boundary_da,
        'next_branch_da_au': next_branch_da,
        'far_ratio': far,
        'far_from_resonance': bool(far >= settings.mmr_far_from_res_factor),
        'rate_apparent_pred_rad_yr': apparent_rate(rate, step_yr),
        'nyquist_halfwidth_au': nyquist_halfwidth(context.a_mean_au, context.body_coefficient, step_yr),
    }


def nearest_ghost(spec: ResonanceSpec, context: Context, step_yr: float) -> float:
    """The ghost a_m nearest to <a> [au]: m = the branch of nu(<a>), or the nearer existing of +-1 when that is 0."""
    m = branch_index(context.rate_expected_rad_yr, step_yr)
    candidates = (m,) if m != 0 else (1, -1)
    ghosts = [ghost_axis(spec, context.planet_mean_motion_rad_yr, context.slow_rate_rad_yr, step_yr, branch) for branch in candidates]
    ghosts = [ghost for ghost in ghosts if math.isfinite(ghost)]
    return min(ghosts, key=lambda ghost: abs(ghost - context.a_mean_au)) if ghosts else float('nan')


def _same_winding_as(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """`reference` with the winding of `candidate` (both exact mod 2pi at every sample): `reference` itself, bit for bit,
    when the two differ by one constant number of turns everywhere, else `reference` + 2pi x the change of that number."""
    turns = np.rint((candidate - reference) / TWO_PI)
    if not len(turns) or np.all(turns == turns[0]):
        return reference
    return reference + TWO_PI * (turns - turns[0])


def slow_vector_unwrap(
    t_yr: np.ndarray, amplitude: np.ndarray, angle: np.ndarray, band_yr: float, filter_order: int, grid_rtol: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """The angle of a vector z = amplitude exp(i angle) (e and varpi) unwrapped with the winding of z_f, its low-pass at
    `band_yr`, and the local phase scatter [rad] (module docstring).

    z_f = Butterworth low-pass of Re z and Im z; the angle becomes unwrap(arg z_f) + wrap(angle - arg z_f), still exact
    mod 2pi at every sample (np.unwrap's own result wherever the two windings agree). This fixes a winding convention:
    the angle itself still holds its deviation from arg z_f, and across a stretch where the phase is undefined the
    winding stays a convention. The scatter is s / |z_f|, s = the RMS of |z - z_f| over a running window of `band_yr`;
    inf where z_f = 0, nan (not measured) within one band of either edge (the filter's edge transient). Without a usable
    band (not longer than two steps, a grid lowpass refuses, or z_f turning by pi or more per band on average - a
    mean-precession guard, not a bound on the local rate) the angle is np.unwrap-ed and the scatter is nan."""
    t = np.asarray(t_yr, dtype=float)
    angle = np.asarray(angle, dtype=float)
    plain = np.unwrap(angle)
    unmeasured = np.full(len(t), np.nan)
    if len(t) < 2:
        return plain, unmeasured
    step = float(np.median(np.diff(t)))
    if not band_yr > 2 * step:
        return plain, unmeasured
    z = np.asarray(amplitude, dtype=float) * np.exp(1j * angle)
    try:
        z_f = filtering.lowpass(t, z.real, band_yr, filter_order, grid_rtol=grid_rtol) + 1j * filtering.lowpass(
            t, z.imag, band_yr, filter_order, grid_rtol=grid_rtol
        )
    except ValueError:
        return plain, unmeasured
    slow = np.unwrap(np.angle(z_f))
    # the band must leave the vector's own turning alone: less than pi per band on average (one carrier of period > 2
    # bands passes at G >= 0.99); read on z_f, whose phase keeps the vector's turning where np.unwrap counts the loops
    if abs(linear_slope(t, slow)) * band_yr >= math.pi:
        return plain, unmeasured
    wound = _same_winding_as(plain, slow + wrap_pi(angle - slow))
    window = max(1, int(round(band_yr / step)))
    rms = np.sqrt(np.maximum(uniform_filter1d(np.abs(z - z_f) ** 2, window, mode='nearest'), 0.0))
    radius = np.abs(z_f)
    scatter = np.full(len(t), np.inf)
    np.divide(rms, radius, out=scatter, where=radius > 0)
    scatter[(t < t[0] + band_yr) | (t > t[-1] - band_yr)] = np.nan
    return wound, scatter


def body_varpi_slow_unwrap(series: MMRSeries, settings: AnalysisSettings) -> Tuple[np.ndarray, np.ndarray]:
    """The body's varpi in the convention of the angle (varpi* = omega - Omega when retrograde) unwrapped with the
    winding of its slow vector z_f = <e exp(i varpi)> at mmr_jitter_band_yr, and its local phase scatter [rad]
    (`slow_vector_unwrap`). Shared by the unwrap of sigma, the slow components and the lambda part of the kinematics,
    so that sigma - p varpi is the same lambda part everywhere."""
    varpi = body_angles(series)[0]
    return slow_vector_unwrap(series.t_yr, series.e, varpi, settings.mmr_jitter_band_yr, settings.filter_order, settings.filter_grid_rtol)


def slow_element_part(series: MMRSeries, settings: AnalysisSettings) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(s, |p| x the body's varpi phase scatter) [rad]: s = p varpi + v Omega + sum_j (p_j varpi_j + v_j Omega_j) of
    sigma, the body's varpi by `body_varpi_slow_unwrap`, every other element np.unwrap-ed (planets' elements, and the
    body's Omega, are slow and their e or i stay away from 0). None when a planet of a varpi or node term is not saved:
    the whole angle is then judged, as before W15. The scatter is 0 without a varpi term, nan where not measured."""
    spec = series.resonance
    n = len(series.t_yr)
    part, scatter = np.zeros(n), np.zeros(n)
    if spec.body_varpi_coefficient:
        varpi, varpi_scatter = body_varpi_slow_unwrap(series, settings)
        part = part + spec.body_varpi_coefficient * varpi
        scatter = abs(spec.body_varpi_coefficient) * varpi_scatter
    if spec.body_node_coefficient:
        part = part + spec.body_node_coefficient * np.unwrap(np.asarray(series.Omega, dtype=float))
    for index, name in enumerate(spec.planets):
        p_j = spec.planet_varpi_coefficients[index] if spec.planet_varpi_coefficients else 0
        v_j = spec.planet_node_coefficient(index)
        if not (p_j or v_j):
            continue
        planet = series.planets.get(name)
        if planet is None or len(planet.t_yr) != n:
            return None
        if p_j:
            part = part + p_j * np.unwrap(np.asarray(planet.varpi, dtype=float))
        if v_j:
            part = part + v_j * np.unwrap(np.asarray(planet.Omega, dtype=float))
    return part, scatter


def undefined_stretches(t_yr: np.ndarray, undefined: np.ndarray) -> Tuple[float, float]:
    """(share of the samples, longest run [yr]) of a boolean mask; a run of m samples lasts m steps."""
    t = np.asarray(t_yr, dtype=float)
    mask = np.asarray(undefined, dtype=bool)
    if not len(mask) or not mask.any():
        return 0.0, 0.0
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return float(mask.mean()), float(np.max(ends - starts) * step)


def _residual_stats(residual: np.ndarray) -> Tuple[float, float]:
    size = np.abs(residual)
    if not len(size):
        return float('nan'), float('nan')
    return float(np.percentile(size, 99)), float(np.max(size))


def _within_limits(p99: float, largest: float, settings: AnalysisSettings) -> bool:
    return bool(p99 <= math.radians(settings.mmr_unwrap_max_resid_p99_deg) and largest <= math.radians(settings.mmr_unwrap_max_resid_deg))


def _unwrap_columns(series: MMRSeries, context: Context, settings: AnalysisSettings) -> dict:
    """The predictive (or numpy) unwrap, its residuals (lambda part: the gate; full angle: a column) and the revolutions
    by both unwraps (module docstring, W15)."""
    t = np.asarray(series.t_yr, dtype=float)
    rates = rate_model(series.resonance, series.a, context.planet_mean_motion_rad_yr, context.slow_rate_rad_yr)
    predictive, full_residual = predictive_unwrap(t, series.angle_wrapped, rates)
    lambda_residual, scatter = full_residual, np.full(len(t), np.nan)
    numpy_version = series.angle_unwrapped_numpy
    if numpy_version is None:
        numpy_version = np.unwrap(np.asarray(series.angle_wrapped, dtype=float))
    slow = slow_element_part(series, settings)
    if slow is not None:
        part, scatter = slow
        psi_wrapped = np.mod(np.asarray(series.angle_wrapped, dtype=float) - part, TWO_PI)
        psi, lambda_residual = predictive_unwrap(t, psi_wrapped, rates - context.slow_rate_rad_yr)
        predictive = _same_winding_as(predictive, psi + part)
        # np.unwrap against the model on the lambda part too: the winding of the slow elements is no alias
        revs_lambda_predictive, revs_lambda_numpy = revolutions(psi), revolutions(np.unwrap(psi_wrapped))
    else:
        revs_lambda_predictive, revs_lambda_numpy = revolutions(predictive), revolutions(numpy_version)
    unwrapped = predictive if settings.mmr_unwrap == 'predictive' else np.asarray(numpy_version, dtype=float)
    p99, largest = _residual_stats(lambda_residual)
    full_p99, full_largest = _residual_stats(full_residual)
    threshold = math.radians(settings.mmr_unwrap_suspect_resid_deg)
    measured = ~np.isnan(scatter)
    scattered = np.where(measured, scatter, 0.0)
    undefined = measured & (scattered > settings.mmr_jitter_max_rad)  # the |p|-scaled phase (column, flag only)
    p = abs(series.resonance.body_varpi_coefficient)
    # varpi itself undefined: its own scatter, scatter / |p| (without a varpi term, the slow part's)
    lost = measured & (scattered / (p or 1) > settings.mmr_jitter_max_rad)
    # steps of the stitched angle beyond the model: a lambda suspect, or a fast turn of the slow elements; the latter is
    # suspect (a 2pi step there is no slip) only where varpi itself is undefined (FIX1: the |p|-scaled mask marked
    # 258640 46J-19+0-27, varpi defined everywhere, with unwrap_step_suspect)
    step_error = np.abs(np.diff(predictive) - rates * np.diff(t))
    fast_slow = (step_error > threshold) & (np.abs(lambda_residual) <= threshold)
    suspect = (np.abs(lambda_residual) > threshold) | (fast_slow & (lost[1:] | lost[:-1]))
    undefined_share, undefined_longest = undefined_stretches(t[measured], undefined[measured]) if measured.any() else (NAN, NAN)
    varpi_undefined = float(np.mean(lost[measured])) if p and measured.any() else NAN
    revs_predictive, revs_numpy = revolutions(predictive), revolutions(numpy_version)
    return {
        'angle_unwrapped': unwrapped,
        'residual_rad': full_residual,
        'unwrap_resid_p99_rad': p99,
        'unwrap_resid_max_rad': largest,
        'unwrap_resid_full_p99_rad': full_p99,
        'unwrap_resid_full_max_rad': full_largest,
        'unwrap_ok': _within_limits(p99, largest, settings),
        'full_angle_ok': _within_limits(full_p99, full_largest, settings),
        'suspect_times_yr': tuple(float(x) for x in (0.5 * (t[1:] + t[:-1]))[suspect]),
        'n_lambda_suspect_steps': int(np.count_nonzero(np.abs(lambda_residual) > threshold)),
        'n_slow_element_steps': int(np.count_nonzero(fast_slow)),
        'phase_undefined_fraction': undefined_share,
        'phase_undefined_longest_yr': undefined_longest,
        'varpi_undefined_fraction': varpi_undefined,
        'revs_predictive': revs_predictive,
        'revs_numpy': revs_numpy,
        'revs_lambda_predictive': revs_lambda_predictive,
        'revs_lambda_numpy': revs_lambda_numpy,
        'alias_numpy_differs': bool(abs(revs_lambda_predictive - revs_lambda_numpy) > settings.mmr_alias_numpy_rev_tol),
        'rate_apparent_meas_rad_yr': linear_slope(t, numpy_version),
        'rate_measured_rad_yr': linear_slope(t, unwrapped),
    }


def _flags(columns: dict) -> list:
    flags = [str(Flag.SHORT_PERIOD_ALIASED)] if columns.get('short_period_aliased') else []
    if columns['jitter_low_e']:
        flags.append(str(Flag.JITTER_LOW_E))
    if columns['alias_numpy_differs']:
        flags.append(str(Flag.ALIAS_NP_UNWRAP_DIFFERS))
    if columns['far_from_resonance']:
        flags.append(str(Flag.FAR_FROM_RESONANCE))
    if columns['n_lambda_suspect_steps']:
        flags.append(str(Flag.UNWRAP_STEP_SUSPECT))
    if columns['n_slow_element_steps']:
        flags.append(str(Flag.VARPI_FAST_LOW_E))
    if columns['phase_undefined_fraction'] > 0:
        flags.append(str(Flag.PHASE_UNDEFINED_LOCAL))
    return flags


def analyse_sampling(series: MMRSeries, context: Context, settings: AnalysisSettings) -> Sampling:
    """SPEC §4.2 for the record, the predictive unwrap and its checks; with `mmr_unwrap='numpy'` the unwrap is np.unwrap.

    Sets `context.rate_measured_rad_yr`. The residual limits (p99 <= mmr_unwrap_max_resid_p99_deg,
    max <= mmr_unwrap_max_resid_deg) are read on the lambda part (module docstring); undefined_phase = jitter_low_e, the
    full angle fails them and varpi itself is undefined (varpi_undefined_fraction above mmr_varpi_undefined_max_share,
    or not measured); unwrap_ok = the lambda part passes and not undefined_phase, so G0 says alias only for the lambda
    part.

    The residual is always that of the model (whether the record can be stitched); 'numpy' changes only
    the unwrapped angle. `aliased_lines_yr` is empty and `resample_step_yr` bounded by the safe steps only:
    both need the cycle period P_cyc, which the caller supplies later (`aliased_lines`, `resample_step`).
    """
    spec, t = series.resonance, np.asarray(series.t_yr, dtype=float)
    step = float(np.median(np.diff(t)))
    alias = _alias_columns(context, step, settings)
    ghost = nearest_ghost(spec, context, step)
    unwrap = _unwrap_columns(series, context, settings)
    context.rate_measured_rad_yr = unwrap.pop('rate_measured_rad_yr')
    band, order = settings.mmr_jitter_band_yr, settings.filter_order
    jitter = perihelion_jitter(t, series.e, spec.body_varpi_coefficient, band, order)
    da_short = short_period_scatter(t, series.a, band, order)
    plain, predictive = safe_steps(
        context.rate_expected_lo_rad_yr,
        context.rate_expected_hi_rad_yr,
        context.a_mean_au,
        spec.body_coefficient,
        da_short,
        settings.mmr_step_resolved_deg / 180.0,
    )
    short_step = short_period_step(context.orbital_period_yr, context.synodic_periods_yr)
    folded = short_period_trigger(step, short_step, NAN, NAN) == SHORT_PERIOD_NYQUIST  # the harmonic part needs the floor
    columns = {**alias, **unwrap, 'jitter_low_e': bool(jitter > settings.mmr_jitter_max_rad), 'short_period_aliased': folded}
    full_angle_ok = columns.pop('full_angle_ok')
    varpi_lost = not columns['varpi_undefined_fraction'] <= settings.mmr_varpi_undefined_max_share  # nan: lost
    undefined_phase = bool(columns['jitter_low_e'] and not full_angle_ok and varpi_lost)
    columns['unwrap_ok'] = bool(columns['unwrap_ok'] and not undefined_phase)
    flags = _flags(columns)
    columns.pop('short_period_aliased')
    return Sampling(
        step_yr=step,
        span_yr=float(t[-1] - t[0]),
        n_rows=len(t),
        nyquist_rad_yr=math.pi / step,
        ghost_axis_au=ghost,
        ghost_distance_au=abs(context.a_mean_au - ghost),
        jitter=jitter,
        undefined_phase=undefined_phase,
        safe_step_plain_yr=plain,
        safe_step_pred_yr=predictive,
        angle_reconstruction_rad=angle_reconstruction_error(series),
        aliased_lines_yr=(),
        resample_step_yr=resample_step(plain, predictive, float('nan'), step, settings),
        flags=flags,
        short_period_step_yr=short_step,
        folded_slowest_yr=folded_slowest_yr(
            context.orbital_period_yr, context.synodic_periods_yr, step, settings.mmr_short_period_harmonics
        ),
        short_period_da_au=da_short,
        rate_apparent_from_mean_longitude_rad_yr=apparent_rate(context.rate_from_mean_longitude_rad_yr, step),
        unwrap_method=settings.mmr_unwrap,
        **columns,
    )
