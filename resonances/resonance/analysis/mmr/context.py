"""Constants of the run and of the record (SPEC §4.1).

Mean motions are the package model's own: the slope of the predictively unwrapped lambda_j of a
saved planet, else `const.PLANET_N_DEG_PER_GAUSS_YR` (measured on the package model in Gaussian
years; Kepler errs by 0.003–0.014 deg/yr, enough to flip the alias branch of 1999 UQ).
`const.DAYS_IN_YEAR` (365.2422) is never used here.

The rate model of the angle (M&D 1999, p. 331, eq. 8.22) is

    nu(a) = sum_j k_j n_j + k n(a) + slow,   slow = p varpi_dot + sum_j p_j varpi_dot_j (+ v Omega_dot + sum_j v_j Omega_dot_j),

and a* (exact resonance with precession) solves nu(a*) = 0. For a retrograde angle the body's terms
are lambda* = M + omega - Omega (rate ~ +n(a)) and varpi* = omega - Omega, and the synodic frequency
with a prograde planet is n + n_j.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from resonances.data import const
from resonances.resonance.analysis.geometry import wrap_pi
from resonances.resonance.analysis.mmr.coorbital import GM_SUN
from resonances.resonance.analysis.mmr.models import PLANET_NAMES_ORDER, Context, GreatInequality, MMRSeries, PlanetSeries, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings

TWO_PI = 2 * math.pi
ARCSEC_TO_RAD = math.pi / (180 * 3600)  # the one arcsec -> rad of the MMR analysis

# Defaults of the two context settings, for callers without settings (tests, the dry-run probes).
DEFAULT_PLANET_UNWRAP_MAX_STEP_DEG = AnalysisSettings.__dataclass_fields__['mmr_planet_unwrap_max_step_deg'].default
DEFAULT_AXIS_RANGE_PERCENTILE = AnalysisSettings.__dataclass_fields__['mmr_axis_range_percentile'].default


def mean_motion(a_au):
    """n(a) = 2 pi a^(-3/2) [rad/yr] (au, Gaussian year, G M_sun = 4 pi^2)."""
    return 2 * np.pi * np.asarray(a_au, dtype=float) ** -1.5


def axis_from_mean_motion(n_rad_yr):
    """Inverse of `mean_motion`: a = (2 pi / n)^(2/3) [au]."""
    return (TWO_PI / np.asarray(n_rad_yr, dtype=float)) ** (2.0 / 3.0)


def predictive_unwrap(t_yr: np.ndarray, angle_wrapped: np.ndarray, rate_rad_yr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unwrap by the model: returns (unwrapped [rad] starting at angle_wrapped[0], residual [rad], length N-1).

    `rate_rad_yr` is a scalar (planet longitudes: n_j) or length N-1 (`rate_model`) [rad/yr]."""
    angle = np.asarray(angle_wrapped, dtype=float)
    predicted = np.asarray(rate_rad_yr, dtype=float) * np.diff(np.asarray(t_yr, dtype=float))
    residual = wrap_pi(np.diff(angle) - predicted)
    predicted += residual  # the increment of each step, in place (no second N-1 array)
    unwrapped = np.empty_like(angle)
    unwrapped[0] = angle[0]
    np.cumsum(predicted, out=unwrapped[1:])
    unwrapped[1:] += angle[0]
    return unwrapped, residual


def identity_coefficient(body_coefficient: int, a_au: float) -> float:
    """c = d nu / d a = -1.5 k n(a) / a [rad yr^-1 au^-1] (sigma_dot = c (a - a*))."""
    return float(-1.5 * body_coefficient * mean_motion(a_au) / a_au)


def table_mean_motion(name: str) -> float:
    """n_j of `const.PLANET_N_DEG_PER_GAUSS_YR` [rad/yr]."""
    return math.radians(const.PLANET_N_DEG_PER_GAUSS_YR[name])


# Planets whose lambda the owner search may take from the catalogue line lambda_j0 + n_j t: those with a measured error
# of that line (const.PLANET_N_SPREAD_DEG_PER_GAUSS_YR; Mercury has none).
CATALOGUE_PLANETS = tuple(name for name in PLANET_NAMES_ORDER if name in const.PLANET_N_SPREAD_DEG_PER_GAUSS_YR)


def catalogue_longitudes(longitudes: Mapping[str, float], t_ref_yr: float, t_start_yr: float) -> Dict[str, float]:
    """lambda_j(t_start) = lambda_j(t_ref) + n_j (t_start - t_ref) mod 2 pi [rad] (table n_j) of the `CATALOGUE_PLANETS`
    among `longitudes` (lambda_j at t_ref, e.g. the epoch of the run). A constant error of lambda_j0 only turns the phase
    of every owner candidate by the same angle (R^2 and the phase coherence do not change), so its precision is not
    important; the error of the rate is (owner.catalogue_phase_error)."""
    return {
        name: float(np.mod(longitudes[name] + table_mean_motion(name) * (t_start_yr - t_ref_yr), TWO_PI))
        for name in CATALOGUE_PLANETS
        if name in longitudes and np.isfinite(longitudes[name])
    }


def planet_frequency(name: str, kind: str) -> float:
    """g_j (kind 'g') or s_j ('s') of planet `name` from `const.PLANETARY_FREQUENCIES` [rad/yr]."""
    index = PLANET_NAMES_ORDER.index(name) + 1
    return const.PLANETARY_FREQUENCIES[f'{kind}{index}'] * ARCSEC_TO_RAD


def linear_slope(t_yr: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope [unit/yr]; nan with fewer than 2 points.

    np.polyfit, not `filtering.linear_fit`: the closed form differs by round-off, which the rate model carries into
    a* and the zone (W5: a flag of 1627 11E-28 flipped); the numbers stay those of the benchmark."""
    t_yr = np.asarray(t_yr, dtype=float)
    if len(t_yr) < 2:
        return float('nan')
    return float(np.polyfit(t_yr - t_yr.mean(), np.asarray(values, dtype=float), 1)[0])


def trend_line(t_yr: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The least-squares line of `values` on the samples `t_yr` (np.polyfit on t - t[0]: see `linear_slope` for why
    polyfit)."""
    since = t_yr - t_yr[0]
    return np.polyval(np.polyfit(since, values, 1), since)


def windowed_slope(t_yr: np.ndarray, angle: np.ndarray, t_lo: float, t_hi: float) -> float:
    """Slope of the np.unwrap-ed angle on [t_lo, t_hi] (the whole record when the window holds < 2 points); t increasing.

    Only the record up to the window's end is unwrapped: np.unwrap is a running sum, its prefix is the same."""
    t_yr = np.asarray(t_yr, dtype=float)
    start, end = int(np.searchsorted(t_yr, t_lo, 'left')), int(np.searchsorted(t_yr, t_hi, 'right'))
    if end - start < 2:
        start, end = 0, len(t_yr)
    unwrapped = np.unwrap(np.asarray(angle, dtype=float)[:end])
    return linear_slope(t_yr[start:end], unwrapped[start:])


def _measured_mean_motion(planet: Optional[PlanetSeries], table_rad_yr: float, max_step_error_rad: float) -> float:
    """Slope of lambda_j unwrapped with the table rate; nan when not saved or when a step errs by > `max_step_error_rad`."""
    if planet is None or planet.mean_longitude is None or len(planet.t_yr) < 2:
        return float('nan')
    unwrapped, residual = predictive_unwrap(planet.t_yr, planet.mean_longitude, table_rad_yr)
    if np.max(np.abs(residual)) > max_step_error_rad:
        return float('nan')
    return linear_slope(planet.t_yr, unwrapped)


def planet_mean_motions(
    planets: Mapping[str, PlanetSeries], names: Tuple[str, ...], max_step_error_deg: Optional[float] = None
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """n_j [rad/yr] for `names`: slope of the predictively unwrapped saved lambda_j ('measured'), else
    `const.PLANET_N_DEG_PER_GAUSS_YR` ('table'). A saved lambda_j whose unwrap errs from the table rate by more than
    `max_step_error_deg` (default MMR_PLANET_UNWRAP_MAX_STEP_DEG) in a step is not used. Returns (values, sources)."""
    limit = math.radians(DEFAULT_PLANET_UNWRAP_MAX_STEP_DEG if max_step_error_deg is None else max_step_error_deg)
    values, sources = {}, {}
    for name in names:
        table = table_mean_motion(name)
        measured = _measured_mean_motion(planets.get(name), table, limit)
        if math.isfinite(measured):
            values[name], sources[name] = measured, 'measured'
        else:
            values[name], sources[name] = table, 'table'
    return values, sources


def _measured_planet_rates(
    planets: Mapping[str, PlanetSeries], names: Tuple[str, ...], attribute: str, window_yr: float
) -> Dict[str, float]:
    """Slope of each saved planet angle `attribute` over the last `window_yr` [rad/yr]; nan when not saved."""
    rates = {}
    for name in names:
        planet = planets.get(name)
        if planet is None or getattr(planet, attribute) is None or len(planet.t_yr) < 2:
            rates[name] = float('nan')
            continue
        t_end = float(planet.t_yr[-1])
        rates[name] = windowed_slope(planet.t_yr, getattr(planet, attribute), t_end - window_yr, t_end)
    return rates


def _or_frequency(measured: Mapping[str, float], kind: str) -> Dict[str, float]:
    """The measured rates, the planet's g_j (kind 'g') or s_j ('s') where not measured."""
    return {name: rate if math.isfinite(rate) else planet_frequency(name, kind) for name, rate in measured.items()}


def planet_varpi_rates(planets: Mapping[str, PlanetSeries], names: Tuple[str, ...], window_yr: float) -> Dict[str, float]:
    """varpi_dot_j [rad/yr]: slope of the saved varpi_j over the last `window_yr`, else g_j of const.PLANETARY_FREQUENCIES."""
    return _or_frequency(_measured_planet_rates(planets, names, 'varpi', window_yr), 'g')


def body_varpi_rate(t_yr: np.ndarray, varpi: np.ndarray, window_yr: float) -> float:
    """Slope of the unwrapped varpi over `window_yr` (centred on the record) [rad/yr]: the slow term of the rate model."""
    t_yr = np.asarray(t_yr, dtype=float)
    centre = 0.5 * (t_yr[0] + t_yr[-1])
    return windowed_slope(t_yr, varpi, centre - 0.5 * window_yr, centre + 0.5 * window_yr)


def slow_rate(
    spec: ResonanceSpec,
    body_varpi_rate_rad_yr: float,
    planet_varpi_rate_rad_yr: Mapping[str, float],
    body_node_rate_rad_yr: float = 0.0,
    planet_node_rate_rad_yr: Optional[Mapping[str, float]] = None,
) -> float:
    """p varpi_dot + sum_j p_j varpi_dot_j [rad/yr] (nodal terms: v Omega_dot + sum v_j Omega_dot_j added the same way).

    A term with a zero coefficient is skipped, so an unmeasured rate (nan) there does not spoil the sum.
    """
    terms = [(spec.body_varpi_coefficient, body_varpi_rate_rad_yr), (spec.body_node_coefficient, body_node_rate_rad_yr)]
    node_rates = planet_node_rate_rad_yr or {}
    for index, name in enumerate(spec.planets):
        terms.append((spec.planet_varpi_coefficients[index], planet_varpi_rate_rad_yr.get(name, float('nan'))))
        terms.append((spec.planet_node_coefficient(index), node_rates.get(name, float('nan'))))
    return float(sum(coefficient * rate for coefficient, rate in terms if coefficient != 0))


def _lambda_part(spec: ResonanceSpec, planet_n: Mapping[str, float]) -> float:
    """sum_j k_j n_j [rad/yr]."""
    return float(sum(coefficient * planet_n[name] for name, coefficient in zip(spec.planets, spec.planet_coefficients)))


def expected_rate(spec: ResonanceSpec, a_au, planet_n: Mapping[str, float], slow_rate_rad_yr: float):
    """nu(a) = sum_j k_j n_j + k n(a) + slow [rad/yr] (M&D 1999, p. 331, eq. 8.22). Scalar or array a."""
    rate = _lambda_part(spec, planet_n) + spec.body_coefficient * mean_motion(a_au) + slow_rate_rad_yr
    return float(rate) if np.ndim(rate) == 0 else rate


def exact_axis(spec: ResonanceSpec, planet_n: Mapping[str, float], slow_rate_rad_yr: float) -> float:
    """a*: nu(a*) = 0, i.e. n(a*) = -(sum_j k_j n_j + slow) / k [au]; nan if k = 0 or no solution."""
    if spec.body_coefficient == 0:
        return float('nan')
    n_star = -(_lambda_part(spec, planet_n) + slow_rate_rad_yr) / spec.body_coefficient
    if not math.isfinite(n_star) or n_star <= 0:
        return float('nan')
    return float(axis_from_mean_motion(n_star))


def synodic_periods(n_rad_yr: float, planet_n: Mapping[str, float], names: Tuple[str, ...], retrograde: bool = False) -> Dict[str, float]:
    """P_syn,j = 2 pi / |n - n_j| [yr] for the planets of the angle (|n + n_j| for a retrograde body)."""
    periods = {}
    for name in names:
        relative = n_rad_yr + planet_n[name] if retrograde else n_rad_yr - planet_n[name]
        periods[name] = TWO_PI / abs(relative) if relative != 0 else math.inf
    return periods


def mean_longitude_rate(t_yr: np.ndarray, mean_longitude: Optional[np.ndarray], a_au: Optional[np.ndarray] = None) -> float:
    """Mean lambda_dot [rad/yr]: slope of the turn-counted mean longitude; nan without lambda.

    With `a_au` the longitude is unwrapped predictively with n(a) (needed at steps of a revolution or
    more); without it np.unwrap is used, which is right only when n dt < pi.
    """
    if mean_longitude is None or len(t_yr) < 2:
        return float('nan')
    if a_au is None:
        unwrapped = np.unwrap(np.asarray(mean_longitude, dtype=float))
    else:
        a_au = np.asarray(a_au, dtype=float)
        unwrapped, _ = predictive_unwrap(t_yr, mean_longitude, mean_motion(0.5 * (a_au[1:] + a_au[:-1])))
    return linear_slope(t_yr, unwrapped)


def axis_from_mean_longitude_rate(t_yr: np.ndarray, mean_longitude: Optional[np.ndarray], a_au: Optional[np.ndarray] = None) -> float:
    """a from the mean lambda_dot: (G M_sun / lambda_dot^2)^(1/3) [au]; nan without lambda (`a_au`: see `mean_longitude_rate`)."""
    return axis_from_rate(mean_longitude_rate(t_yr, mean_longitude, a_au))


def axis_from_rate(rate_rad_yr: float) -> float:
    """(G M_sun / rate^2)^(1/3) [au]; nan for a zero or unmeasured rate."""
    if not math.isfinite(rate_rad_yr) or rate_rad_yr == 0:
        return float('nan')
    return float((GM_SUN / rate_rad_yr**2) ** (1.0 / 3.0))


def crossed_planets(q_min_au: float, big_q_max_au: float, planets: Mapping[str, Tuple[float, float]]) -> Tuple[str, ...]:
    """Planets (name -> (q_j, Q_j) [au]) whose orbits the body crosses: q < Q_j and Q > q_j."""
    return tuple(name for name, (q_j, big_q_j) in planets.items() if q_min_au < big_q_j and big_q_max_au > q_j)


def near_commensurability_periods(planet_n: Mapping[str, float]) -> Dict[str, float]:
    """Periods [yr] of `const.PLANET_NEAR_COMMENSURABILITIES`: 2 pi / |sum k_j n_j + sum p_j g_j|, with the
    given n_j where present, else the table."""
    periods = {}
    for name, term in const.PLANET_NEAR_COMMENSURABILITIES.items():
        rate = sum(k * planet_n.get(planet, table_mean_motion(planet)) for planet, k in term['mean_longitudes'].items())
        rate += sum(p * planet_frequency(planet, 'g') for planet, p in term['perihelia'].items())
        periods[name] = TWO_PI / abs(rate) if rate != 0 else math.inf
    return periods


def _wave_half_amplitude(planet: PlanetSeries, n_rad_yr: float, period_yr: float) -> float:
    """Half-amplitude [rad] of the term of `period_yr` in the unwrapped lambda: least squares on [1, t, cos, sin]."""
    t = np.asarray(planet.t_yr, dtype=float) - planet.t_yr[0]
    unwrapped, _ = predictive_unwrap(planet.t_yr, planet.mean_longitude, n_rad_yr)
    omega = TWO_PI / period_yr
    design = np.column_stack([np.ones_like(t), t, np.cos(omega * t), np.sin(omega * t)])
    coefficients = np.linalg.lstsq(design, unwrapped, rcond=None)[0]
    return float(math.hypot(coefficients[2], coefficients[3]))


def great_inequality(
    planets: Mapping[str, PlanetSeries], planet_n: Mapping[str, float], angle_planets: Optional[Tuple[str, ...]] = None
) -> Optional[GreatInequality]:
    """P_GI = 2 pi / |2 n_J - 5 n_S + 3 varpi_dot_S| [yr] and the half-amplitudes of the lambda waves, measured from
    saved Jupiter/Saturn longitudes, else `const.GREAT_INEQUALITY` (0.34 deg, 0.83 deg). None if neither planet is involved.

    "Involved" = in `angle_planets` (default: the keys of `planet_n`). varpi_dot_S is g6, the rate of Saturn's
    own mode: the osculating varpi_S over a few kyr mixes in g5 (20 kyr: 47.6"/yr against g6 = 28.2"/yr).
    The amplitudes are measured when both longitudes are saved over at least two periods.
    """
    involved = tuple(planet_n) if angle_planets is None else tuple(angle_planets)
    if 'Jupiter' not in involved and 'Saturn' not in involved:
        return None
    period = near_commensurability_periods(planet_n)['great_inequality']
    jupiter, saturn = planets.get('Jupiter'), planets.get('Saturn')
    saved = all(p is not None and p.mean_longitude is not None for p in (jupiter, saturn))
    if saved and jupiter.t_yr[-1] - jupiter.t_yr[0] >= 2 * period:
        n_j = planet_n.get('Jupiter', table_mean_motion('Jupiter'))
        n_s = planet_n.get('Saturn', table_mean_motion('Saturn'))
        return GreatInequality(period, _wave_half_amplitude(jupiter, n_j, period), _wave_half_amplitude(saturn, n_s, period), True)
    table = const.GREAT_INEQUALITY
    return GreatInequality(period, math.radians(table['dlambda_jupiter_deg']), math.radians(table['dlambda_saturn_deg']), False)


def mu_scale_no_eccentricity(spec: ResonanceSpec, masses: Mapping[str, float]) -> float:
    """mu_P (two-body, retrograde) or mu_i mu_j (three-body): the strength scale without e^q."""
    return float(np.prod([masses[name] for name in spec.planets]))


def _planet_orbit_band(planet: Optional[PlanetSeries], name: str) -> Tuple[float, float]:
    """(q_j, Q_j) [au]: from the saved elements (min q, max Q), else const.PLANETS_AXIS and the J2000 e."""
    if planet is not None:
        return float(np.min(planet.a * (1 - planet.e))), float(np.max(planet.a * (1 + planet.e)))
    a, e = const.PLANETS_AXIS[name], const.PLANETS_ECCENTRICITY[name]
    return a * (1 - e), a * (1 + e)


def body_angles(series: MMRSeries) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(varpi, mean longitude) of the body in the convention of the angle: varpi* and lambda* when retrograde."""
    if series.resonance.retrograde:
        varpi = series.omega - series.Omega
        mean_longitude = None if series.mean_anomaly is None else series.mean_anomaly + varpi
        return varpi, mean_longitude
    return series.varpi, series.mean_longitude


def _context_planets(series: MMRSeries) -> Tuple[str, ...]:
    """The angle's planets and every saved planet, in solar-system order."""
    wanted = set(series.resonance.planets) | set(series.planets)
    return tuple(name for name in PLANET_NAMES_ORDER if name in wanted)


def _planet_columns(series: MMRSeries, names: Tuple[str, ...], window_yr: float, max_step_error_deg: float) -> dict:
    """The per-planet dicts of Context (run-level, constants.csv)."""
    planet_n, sources = planet_mean_motions(series.planets, names, max_step_error_deg)
    saved = [name for name in names if name in series.planets]
    varpi_rates = _measured_planet_rates(series.planets, names, 'varpi', window_yr)
    node_rates = _measured_planet_rates(series.planets, names, 'Omega', window_yr)
    return {
        'planet_mean_motion_rad_yr': planet_n,
        'planet_mean_motion_source': sources,
        'planet_mass': {name: const.PLANETS_MASS[name] for name in names},
        'planet_varpi_rate_rad_yr': _or_frequency(varpi_rates, 'g'),
        'planet_node_rate_rad_yr': _or_frequency(node_rates, 's'),
        'planet_varpi_rate_measured_rad_yr': varpi_rates,
        'planet_node_rate_measured_rad_yr': node_rates,
        'planet_g_rad_yr': {name: planet_frequency(name, 'g') for name in names},
        'planet_s_rad_yr': {name: planet_frequency(name, 's') for name in names},
        'planet_axis_mean_au': {name: float(np.mean(series.planets[name].a)) for name in saved},
        'planet_e_mean': {name: float(np.mean(series.planets[name].e)) for name in saved},
        'near_commensurability_periods_yr': near_commensurability_periods(planet_n),
    }


def _orbit_columns(series: MMRSeries, range_percentile: float = DEFAULT_AXIS_RANGE_PERCENTILE) -> dict:
    """Mean elements, ranges and the crossing test (SPEC §4.1, rec). The a range is the [x, 100 - x] percentiles of a,
    x = `range_percentile` (MMR_AXIS_RANGE_PERCENTILE): nu_lo, nu_hi, the alias branches and the far ratio use it."""
    a, e = series.a, series.e
    a_lo, a_hi = np.percentile(a, (range_percentile, 100.0 - range_percentile))
    q_min, big_q_max = float(np.min(a * (1 - e))), float(np.max(a * (1 + e)))
    bands = {name: _planet_orbit_band(series.planets.get(name), name) for name in series.resonance.planets}
    return {
        'a_mean_au': float(np.mean(a)),
        'a_p01_au': float(a_lo),
        'a_p99_au': float(a_hi),
        'e_min': float(np.min(e)),
        'e_med': float(np.median(e)),
        'e_max': float(np.max(e)),
        'inc_min_rad': float(np.min(series.inc)),
        'inc_med_rad': float(np.median(series.inc)),
        'inc_max_rad': float(np.max(series.inc)),
        'q_min_au': q_min,
        'big_q_max_au': big_q_max,
        'crosses': crossed_planets(q_min, big_q_max, bands),
    }


def _rate_columns(spec: ResonanceSpec, orbit: dict, planet_n: Mapping[str, float], slow: float, body_rate: float) -> dict:
    """nu at <a> and over the a 1..99 % range, nu with the body's measured lambda_dot, a* and the nominal a_res."""
    rates = [expected_rate(spec, orbit[key], planet_n, slow) for key in ('a_p01_au', 'a_p99_au')]
    rate_at_mean = expected_rate(spec, orbit['a_mean_au'], planet_n, slow)
    nominal = spec.nominal_axis_au
    if nominal is None:  # no catalogue value: the commensurability with the table n_j, no slow terms
        nominal = exact_axis(spec, {name: table_mean_motion(name) for name in spec.planets}, 0.0)
    return {
        'rate_expected_rad_yr': rate_at_mean,
        'rate_from_mean_longitude_rad_yr': rate_at_mean + spec.body_coefficient * (body_rate - float(mean_motion(orbit['a_mean_au']))),
        'rate_expected_lo_rad_yr': float(min(rates)),
        'rate_expected_hi_rad_yr': float(max(rates)),
        'exact_axis_au': exact_axis(spec, planet_n, slow),
        'nominal_axis_au': float(nominal),
    }


def build_context(series: MMRSeries, settings: AnalysisSettings) -> Context:
    """Every number of SPEC §4.1 for the record (rate_measured is filled after the sampling step)."""
    spec = series.resonance
    window = settings.mmr_varpi_ref_window_yr
    planets = _planet_columns(series, _context_planets(series), window, settings.mmr_planet_unwrap_max_step_deg)
    planet_n = planets['planet_mean_motion_rad_yr']
    varpi, mean_longitude = body_angles(series)
    varpi_rate = body_varpi_rate(series.t_yr, varpi, window)
    node_rate = body_varpi_rate(series.t_yr, series.Omega, window)
    slow = slow_rate(spec, varpi_rate, planets['planet_varpi_rate_rad_yr'], node_rate, planets['planet_node_rate_rad_yr'])
    orbit = _orbit_columns(series, settings.mmr_axis_range_percentile)
    n = float(mean_motion(orbit['a_mean_au']))
    step = float(np.median(np.diff(series.t_yr)))
    body_rate = mean_longitude_rate(series.t_yr, mean_longitude, series.a)
    return Context(
        **planets,
        **orbit,
        **_rate_columns(spec, orbit, planet_n, slow, body_rate),
        great_inequality=great_inequality(series.planets, planet_n, spec.planets),
        order=spec.order,
        orbital_period_yr=TWO_PI / n,
        mean_motion_rad_yr=n,
        identity_coefficient=identity_coefficient(spec.body_coefficient, orbit['a_mean_au']),
        axis_from_mean_longitude_rate_au=axis_from_rate(body_rate),
        synodic_periods_yr=synodic_periods(n, planet_n, spec.planets, spec.retrograde),
        slow_rate_rad_yr=slow,
        rate_measured_rad_yr=float('nan'),
        mu_scale_no_eq=mu_scale_no_eccentricity(spec, planets['planet_mass']),
        fair_step_fraction=step * n / TWO_PI,
        angle_planets=spec.planets,
        body_coefficient=spec.body_coefficient,
        body_varpi_coefficient=spec.body_varpi_coefficient,
        retrograde=spec.retrograde,
        body_varpi_rate_rad_yr=varpi_rate,
    )
