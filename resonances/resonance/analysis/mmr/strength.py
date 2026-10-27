"""Strength of the resonance: theory, required strength and the support check (SPEC §4.9).

Required. The small-amplitude period T0 of the decisive episode, the median over its cycles of P_i / F(A_i)
(`response.episode_physics`), gives omega0 = 2 pi / T0 = |k| n sqrt(3 mu_eff) (M&D 1999, §8.7, eqs. 8.53–8.58):
mu_eff_required (`mu_eff_required`).

Theory, two-body (method 'direct_averaging'). The disturbing function averaged over the fast angle at fixed sigma,
three-dimensional, with the planet on its actual eccentric inclined orbit (`averaged_potential.py`, which holds the
derivation): sigma is a pendulum in (3 k^2 / a^2) <R>(sigma), so about the occupied minimum sigma_c

    omega0^2 = 3 k^2 <R>''(sigma_c) / a^2   =>   mu_eff_theory = a <R>''(sigma_c) / (G M_sun) = mu_P a <D>''(sigma_c),

the same convention as the required one. The occupied minimum is the minimum of <R> whose basin holds the measured
centre of the libration (an asymmetric well, 119979 1N-2 at 75 deg, is a minimum of its own). The descent always
finds some minimum, so an evaluation counts only when the observed half-cycle nearest to its time (`observed_swing`)
librates in that well: it contains sigma_c and does not pass the global maximum of <R> (`averaging.swing_in_well`; a
1:n horseshoe crosses only the inner maximum and counts). 482824 19N-32 is the case: its measured centre is 178 deg
from the minimum, every half-cycle crosses the top of <R>, and the strengths agree (ratio 7.8) by accident. The
distance of sigma_c from the measured centre stays a column (`str_centre_offset_max_deg`). The theory is taken at
`mmr_strength_snapshots` evenly spaced times inside the decisive episode, each with the body's and the planet's
simultaneous elements (medians over one cutoff P_c about the time, circular means of the angles: the short-period
terms out), because the planet's eccentricity enters through varpi - varpi_P and changes the theory by a factor of up
to 10^3 for Mars (2005 GQ107 40M-57). Without the planet's saved elements its phase is unknown: its mean J2000 orbit
(`coorbital.PLANET_MEAN_ORBITS_J2000`) is turned about the ecliptic pole by `mmr_strength_unsaved_planet_phases`
evenly spaced angles at every snapshot; the consistent ones (`str_n_consistent` of `str_n_evaluations`) go into the
range. The check reads the range [ratio_min, ratio_max] of required/theory:

    pass  the range overlaps [1/x, x], x = mmr_strength_check_ratio (some snapshot agrees within an order of magnitude:
          the measured T0 is an average over the episode, so a theory that varies across it agrees with it somewhere
          in its range);
    fail  the whole range lies outside, on one side or the other; or no evaluation is consistent (reason
          'centre_outside_well': the range over every evaluation is then a column only);
    not_measured  no required strength (no episode with T0), no planet mass, or a snapshot whose curvature is not
          measured: no stable well near the measured centre, the quadrature not converged at
          mmr_strength_max_nodes_per_turn (nearly crossing orbits), or the body within mmr_strength_min_distance_hill
          Hill radii of the planet at the centre (an encounter: the averaged problem does not hold there).

`ratio` is the end of the range nearest to 1 (1 when the range contains it), so pass <=> 1/x <= ratio <= x and
implausible_strength <=> ratio >= or <= the implausible factor either way.

Theory, two-body, high order (method 'direct_harmonic'). A term of order q is ~ e^q of <D>: beyond ~1e-12 of D0 the
double-precision profile is rounding (2897 35S-4: 2|C| / D0 ~ 1e-39), so its 'well' is no stable centre or a
curvature of noise (`needs_harmonic`: below mmr_strength_double_floor x |D0|). There the theory is mu_P a 2|C| of
the resonant harmonic C itself, in extended precision (`averaging.harmonic_amplitude_mp`, mmr_strength_mp = auto):
<D>(sigma) = D0 + 2|C| cos(...) + harmonics of order e^(2q), so 2|C| is the curvature at the minimum. Every
evaluation counts (no well to test the swing against), and it is read exactly as 'direct_averaging' (check, ratio,
implausible, F a physics margin). mu e^q is never used as a bound: it misses the coefficient by 1e-11 ... 1e15 at
high order (W20 B).

Theory, three-body and 1:1 (method 'eq_scale'): the direct averaging is not applicable here (a three-body term is a
second-order average over two planets; a 1:1 angle has its own law, `coorbital.py`), so the order of magnitude
mu_scale = mu_P e^q (two-body) or mu_i mu_j e^q (three-body) stays (Gallardo 2014, pp. 5, 7; Nesvorny & Morbidelli
1998, pp. 3031–3032): the coefficient of the term is missing and reaches ~1e2 (Gallardo 2006, p. 30). e^q is used only
without crossing the planets of the term and for mean e < mmr_strength_e_max; without e^q the ratio is taken against
mu alone and only a required strength far above it is implausible (one-sided). The margin F of this method is a
support margin but not a physics measurement (`physics_margins`). mu_scale is a column on every path.

Small-divisor envelope (three-body, legal e^q, ratio >= mmr_strength_implausible_ratio; `small_divisor_envelope`).
A second-order term can be enhanced by a near two-body commensurability with a planet of the term: Codex's generous
stress test (W17 Q3) takes the enhancement up to B^2, B = n / |Delta|, n the body's mean motion at <a> and Delta the
rate of a two-body angle k1 lambda_P - k2 lambda + (k2 - k1) varpi of that planet (the owner enumeration,
`owner._two_body`: k1 <= mmr_owner_coef_max, |k2 - k1| <= mmr_owner_order_max, rate within mmr_owner_rate_factor x
2 pi / P somewhere in the a range [p01, p99]). |Delta| is the smallest over the candidates of the smaller rate at the
two ends of that range (the most favourable endpoint), 0 for a range that crosses zero. The envelope is derived for a
fast divisor only: a two-body angle that turns slower than sigma swings is not one, so |Delta| counts at least
2 pi / min(P, P_cyc), i.e. B <= n min(P, P_cyc) / 2 pi (W22 R-DIV, author 2026-10-24: a crossing of zero is no longer
an unbounded B; 8306 2M-5S-3: 14M-25 crosses, B = 53 instead of inf); without a candidate every such angle turns
faster than the bound, so B < n / bound. The residual ratio / B^2 is the factor the
envelope still misses (a lower bound without a candidate). It is not a theorem: the coefficient C of C B^2 is not
measured, which is why the decision asks the residual to stay >= mmr_strength_implausible_ratio (the same factor the
ratio itself is judged by) before it calls the strength implausible (`decide`, rule 5s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.geometry import Episode, Geometry, run_of
from resonances.resonance.analysis.mmr import averaged_potential as averaging
from resonances.resonance.analysis.mmr.coorbital import GM_SUN, PLANET_MEAN_ORBITS_J2000, Orbit
from resonances.resonance.analysis.mmr.models import Context, EpisodePhysics, MMRSeries, ResonanceKind, ResonanceSpec, Strength
from resonances.resonance.analysis.mmr.preprocess import law_planet
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

NAN = float('nan')
TWO_PI = 2 * np.pi
METHOD_DIRECT = 'direct_averaging'
METHOD_HARMONIC = 'direct_harmonic'  # the resonant harmonic in extended precision (double precision cannot resolve it)
METHOD_SCALE = 'eq_scale'
DIRECT_METHODS = (METHOD_DIRECT, METHOD_HARMONIC)  # the theory computed from the disturbing function: a measurement
MP_AUTO = 'auto'
PLANET_SAVED = 'saved'
PLANET_PHASE_SCAN = 'mean_orbit_phase_scan'
# why the direct theory is missing (column str_not_measured_reason)
REASON_NO_EPISODE = 'no_required_strength'
REASON_NO_PLANET = 'no_planet_orbit'
REASON_CLOSE_APPROACH = 'close_approach_at_centre'
REASON_CENTRE_OUTSIDE_WELL = 'centre_outside_well'  # measured, but no observed swing is a libration in the well (F fails)


# how the small-divisor envelope was bounded (column str_divisor_kind; '' = not evaluated)
DIVISOR_NEAREST = 'nearest'  # the smallest endpoint rate of the enumerated two-body angles
DIVISOR_CROSSES_ZERO = 'crosses_zero'  # some enumerated angle passes its exact commensurability: B at the swing cap
DIVISOR_NO_CANDIDATE = 'no_candidate'  # every such angle turns faster than the bound: B < n / bound
DIVISOR_NO_PERIOD = 'no_period'  # no period to bound the rates: the envelope cannot be formed


@dataclass(frozen=True)
class DivisorEnvelope:
    """The generous small-divisor envelope of a three-body strength (module docstring): B = n / max(|Delta|,
    2 pi / min(P, P_cyc)) (the upper bound n / bound without a candidate), the residual ratio / B^2, the candidate that
    set |Delta| and whether the swing cap set B (`capped`: the candidate turns slower than sigma swings)."""

    kind: str
    b: float = NAN
    residual: float = NAN
    candidate: str = ''
    capped: bool = False


def small_divisor_envelope(
    spec: ResonanceSpec, context: Context, ratio: float, period_yr: float, period_cycle_yr: float, settings: AnalysisSettings
) -> DivisorEnvelope:
    """`DivisorEnvelope` of a three-body angle whose required strength is `ratio` x mu_i mu_j e^q, its episode period
    `period_yr` and the record's cycle period `period_cycle_yr` setting the rate bound (`owner.rate_bound`) and the
    swing cap |Delta| >= 2 pi / min(P, P_cyc) (module docstring)."""
    from resonances.resonance.analysis.mmr import owner as owner_module  # owner imports this module through response
    from resonances.resonance.analysis.mmr.context import mean_motion

    bound = owner_module.rate_bound(period_yr, period_cycle_yr, settings)
    if not (np.isfinite(bound) and bound > 0 and np.isfinite(context.a_p01_au) and np.isfinite(context.a_p99_au)):
        return DivisorEnvelope(DIVISOR_NO_PERIOD)
    swing_rate = TWO_PI / min(p for p in (period_yr, period_cycle_yr) if np.isfinite(p) and p > 0)  # bound is finite
    ends = (float(mean_motion(context.a_p01_au)), float(mean_motion(context.a_p99_au)))
    varpi = context.body_varpi_rate_rad_yr if np.isfinite(context.body_varpi_rate_rad_yr) else 0.0
    n = context.mean_motion_rad_yr
    nearest, name, crosses = np.inf, '', False
    for planet in spec.planets:
        n_planet = owner_module.planet_mean_motion(context, planet)
        for _, (k1,), k in owner_module._two_body(planet, n_planet, ends, varpi, bound, settings):
            k2 = -k
            rates = [k1 * n_planet - k2 * n_end + (k2 - k1) * varpi for n_end in ends]
            through = bool(np.sign(rates[0]) != np.sign(rates[1]))
            rate = 0.0 if through else min(abs(rates[0]), abs(rates[1]))
            if rate < nearest:  # the first candidate that crosses zero keeps the name
                nearest, name, crosses = rate, f'{k1}{planet[0]}-{k2}', through
    if not name:
        b = n / bound
        return DivisorEnvelope(DIVISOR_NO_CANDIDATE, float(b), float(ratio / b**2) if np.isfinite(ratio) else NAN, '')
    b = n / max(nearest, swing_rate)
    kind = DIVISOR_CROSSES_ZERO if crosses else DIVISOR_NEAREST
    residual = float(ratio / b**2) if np.isfinite(ratio) else NAN
    return DivisorEnvelope(kind, float(b), residual, name, capped=bool(nearest < swing_rate))


def mu_scale(mu_scale_no_eq: float, e_mean: float, order: int) -> float:
    """mu_scale = mu_scale_no_eq e^q."""
    return float(mu_scale_no_eq * e_mean**order)


def eq_legal(e_mean: float, crosses: Sequence[str], settings: AnalysisSettings) -> bool:
    """e^q legal: no planet of the term crossed and mean e < mmr_strength_e_max."""
    return bool(np.isfinite(e_mean) and not crosses and e_mean < settings.mmr_strength_e_max)


def mu_eff_required(small_period_yr: float, body_coefficient: int, mean_motion_rad_yr: float) -> float:
    """(omega0 / (|k| n))^2 / 3 with omega0 = 2 pi / T0 (M&D 1999, §8.7, eqs. 8.53–8.58); nan without T0 or k."""
    if not (np.isfinite(small_period_yr) and small_period_yr > 0) or body_coefficient == 0:
        return NAN
    omega0 = TWO_PI / small_period_yr
    return float((omega0 / (abs(body_coefficient) * mean_motion_rad_yr)) ** 2 / 3)


def omega0_theory(body_coefficient: int, mean_motion_rad_yr: float, scale: float) -> float:
    """omega0_th = |k| n sqrt(3 mu) [rad/yr] of a theoretical mu (mu_scale or the direct mu_eff)."""
    if not (np.isfinite(scale) and scale >= 0):
        return NAN
    return float(abs(body_coefficient) * mean_motion_rad_yr * np.sqrt(3 * scale))


def strength_check(eq_is_legal: bool, ratio: float, settings: AnalysisSettings) -> CheckValue:
    """pass: `ratio` within [1/mmr_strength_check_ratio, mmr_strength_check_ratio]; fail outside; not_measured when the
    theory is not measured (`eq_is_legal` False: e^q not legal, or no direct theory) or nothing was required.
    With the direct averaging `ratio` is the end of the range nearest to 1, so this is the overlap of the range."""
    if not eq_is_legal or not (np.isfinite(ratio) and ratio > 0):
        return CheckValue.NOT_MEASURED
    limit = settings.mmr_strength_check_ratio
    return CheckValue.PASS if 1 / limit <= ratio <= limit else CheckValue.FAIL


def implausible_strength(eq_is_legal: bool, ratio: float, settings: AnalysisSettings) -> bool:
    """ratio >= mmr_strength_implausible_ratio, or <= its inverse when the theory is a value (direct averaging, or a
    legal e^q); without e^q the scale is mu alone, so only a required strength far above it is implausible. False
    when nothing was required."""
    if not (np.isfinite(ratio) and ratio > 0):
        return False
    limit = settings.mmr_strength_implausible_ratio
    return bool(ratio >= limit or (eq_is_legal and ratio <= 1 / limit))


def nearest_to_one(low: float, high: float) -> float:
    """The point of [low, high] nearest to 1 (1 when it lies inside): what the overlap with [1/x, x] reads."""
    return float(min(max(1.0, low), high))


def physics_margins(margins: Sequence[str], strength: Optional[Strength]) -> Tuple[str, ...]:
    """The physics margins of a record (confidence.raw_level): F only when the strength was measured by direct
    averaging; the e^q scale is an order of magnitude, not a measurement of the term (W6 Q11)."""
    if strength is not None and strength.method in DIRECT_METHODS:
        return tuple(margins)
    return tuple(margin for margin in margins if margin != 'F')


def _decisive(episodes: Sequence[EpisodePhysics]) -> Optional[EpisodePhysics]:
    """The first episode with a measured mu_eff (the orchestration passes the decisive episode first)."""
    for episode in episodes:
        if np.isfinite(episode.mu_eff):
            return episode
    return None


def one_to_one(spec: Optional[ResonanceSpec]) -> bool:
    """A 1:1 angle the co-orbital law judges (`preprocess.law_planet`: prograde co-orbital or retrograde 1/-1): that law
    judges its strength (K / K_theory). The same predicate as the force law's (W20 round D, review C #8: 624 1J+1+0-2
    has |k| = |k_J| = 1 but no co-orbital law, so its e^q strength is judged)."""
    return law_planet(spec) is not None


def direct_applicable(spec: Optional[ResonanceSpec]) -> bool:
    """A two-body angle (prograde or retrograde) that is not 1:1 (the co-orbital law judges those)."""
    if spec is None or spec.kind == ResonanceKind.THREE_BODY or len(spec.planet_coefficients) != 1:
        return False
    return not one_to_one(spec)


# --- the direct averaging at the snapshots -------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectTheory:
    """mu_eff_theory at every evaluation (snapshot x planet phase) and how it was obtained. `values` holds the
    evaluations whose observed swing lies in the theoretical well (`averaging.swing_in_well`), `values_all` every
    measured one (the columns when none is consistent)."""

    values: Tuple[float, ...]
    centre_deg: float
    offset_max_deg: float
    min_distance_hill: float
    nodes_per_turn: int
    planet_elements: str
    reason: str  # '' when every evaluation is measured and some is consistent
    values_all: Tuple[float, ...] = ()
    n_consistent: int = 0
    method: str = METHOD_DIRECT
    # per evaluation of values_all: the body's averaged a [au] and e (the R1 columns: periods and widths per snapshot)
    a_all: Tuple[float, ...] = ()
    e_all: Tuple[float, ...] = ()

    @property
    def measured(self) -> bool:
        return self.reason == '' and len(self.values) > 0


def snapshot_times(t_start_yr: float, t_end_yr: float, count: int) -> np.ndarray:
    """`count` evenly spaced times inside [t_start, t_end], at the middles of equal parts."""
    return t_start_yr + (np.arange(count) + 0.5) / count * (t_end_yr - t_start_yr)


def observed_centre(episode: Episode, t_yr: float) -> float:
    """The midpoint (sigma(TP_i) + sigma(TP_i+1)) / 2 of the half-cycle nearest to `t_yr` [rad, 0..2pi)."""
    times, values = np.asarray(episode.tp_times_yr, float), np.asarray(episode.tp_values_rad, float)
    if len(times) < 2:
        return float(np.mod(episode.centre_mod_2pi_rad, TWO_PI))
    middles = (times[1:] + times[:-1]) / 2
    i = int(np.argmin(np.abs(middles - t_yr)))
    return float(np.mod((values[i] + values[i + 1]) / 2, TWO_PI))


def observed_swing(episode: Episode, t_yr: float) -> Tuple[float, float]:
    """(lo, hi): the unwrapped sigma of the two turning points of the half-cycle nearest to `t_yr` (the one
    `observed_centre` takes), sorted [rad]; (nan, nan) with fewer than two turning points."""
    times, values = np.asarray(episode.tp_times_yr, float), np.asarray(episode.tp_values_rad, float)
    if len(times) < 2:
        return NAN, NAN
    middles = (times[1:] + times[:-1]) / 2
    i = int(np.argmin(np.abs(middles - t_yr)))
    return float(min(values[i], values[i + 1])), float(max(values[i], values[i + 1]))


def _window(t: np.ndarray, centre_yr: float, width_yr: float) -> slice:
    """The samples within +-width/2 of `centre_yr`; the nearest sample when none."""
    start = int(np.searchsorted(t, centre_yr - width_yr / 2, 'left'))
    end = int(np.searchsorted(t, centre_yr + width_yr / 2, 'right'))
    if end <= start:
        i = int(np.clip(np.searchsorted(t, centre_yr), 0, len(t) - 1))
        return slice(i, i + 1)
    return slice(start, end)


def body_ellipse(series: MMRSeries, part: slice, spec: Optional[ResonanceSpec] = None) -> Tuple[averaging.Ellipse, float]:
    """(the body's mean orbit over `part`, the slow phase phi_B of sigma = k M + phi_B + ...); sigma is `spec`, default
    the record's resonance (the owner search passes its candidate).

    Prograde: lambda = M + varpi, phi_B = (k + p) varpi + v Omega (varpi averaged as itself: defined at i -> 0).
    Retrograde: lambda* = M + omega - Omega, varpi* = omega - Omega, phi_B = (k + p)(omega - Omega) + v Omega
    (never REBOUND's l or pomega: `RetrogradeTwoBody`)."""
    spec = series.resonance if spec is None else spec
    a, e, inc = (float(np.median(np.asarray(x, float)[part])) for x in (series.a, series.e, series.inc))
    node = filtering.circular_mean(np.asarray(series.Omega, float)[part])
    k, p, v = spec.body_coefficient, spec.body_varpi_coefficient, spec.body_node_coefficient
    if spec.retrograde:
        omega = filtering.circular_mean(np.asarray(series.omega, float)[part])
        phase = (k + p) * (omega - node) + v * node
    else:
        varpi = filtering.circular_mean(np.asarray(series.varpi, float)[part])
        omega = varpi - node
        phase = (k + p) * varpi + v * node
    return averaging.Ellipse(a, Orbit(e, inc, omega, node)), float(phase)


def planet_phase(spec: ResonanceSpec, varpi_rad: float, node_rad: float) -> float:
    """phi_P = p_P varpi_P + v_P Omega_P (the planet's slow terms of sigma)."""
    p_planet = spec.planet_varpi_coefficients[0] if spec.planet_varpi_coefficients else 0
    return float(p_planet * varpi_rad + spec.planet_node_coefficient(0) * node_rad)


def saved_planet(series: MMRSeries, name: str, part: slice) -> Tuple[averaging.Ellipse, float, float]:
    """(the planet's mean orbit over `part`, varpi_P, Omega_P) from its saved elements."""
    planet = series.planets[name]
    a, e, inc = (float(np.median(np.asarray(x, float)[part])) for x in (planet.a, planet.e, planet.inc))
    node = filtering.circular_mean(np.asarray(planet.Omega, float)[part])
    varpi = filtering.circular_mean(np.asarray(planet.varpi, float)[part])
    return averaging.Ellipse(a, Orbit(e, inc, varpi - node, node)), varpi, node


def mean_planet(context: Context, name: str, turn_rad: float) -> Optional[Tuple[averaging.Ellipse, float, float]]:
    """The planet's mean J2000 orbit turned by `turn_rad` about the ecliptic pole (node and perihelion together); a from
    its mean axis or mean motion. None without either."""
    a = context.planet_axis_mean_au.get(name, NAN)
    if not (np.isfinite(a) and a > 0):
        n = context.planet_mean_motion_rad_yr.get(name, NAN)
        a = (GM_SUN / n**2) ** (1.0 / 3.0) if np.isfinite(n) and n > 0 else NAN
    if not np.isfinite(a) or name not in PLANET_MEAN_ORBITS_J2000:
        return None
    e, inc_deg, node_deg, varpi_deg = PLANET_MEAN_ORBITS_J2000[name]
    node, varpi = np.radians(node_deg) + turn_rad, np.radians(varpi_deg) + turn_rad
    return averaging.Ellipse(a, Orbit(e, np.radians(inc_deg), varpi - node, node)), float(varpi), float(node)


def quadrature(settings: AnalysisSettings) -> averaging.Quadrature:
    return averaging.Quadrature(
        nodes_per_turn=settings.mmr_strength_nodes_per_turn,
        max_nodes_per_turn=settings.mmr_strength_max_nodes_per_turn,
        sigma_grid=settings.mmr_strength_sigma_grid,
        half_width_rad=np.radians(settings.mmr_strength_fit_half_width_deg),
        tolerance=settings.mmr_strength_quadrature_tol,
    )


def hill_radius(planet_a_au: float, mu: float) -> float:
    """r_H = a_P (mu / 3)^(1/3) [au]."""
    return float(planet_a_au * (mu / 3.0) ** (1.0 / 3.0))


def harmonic_rule(settings: AnalysisSettings) -> averaging.HarmonicRule:
    return averaging.HarmonicRule(
        dps=settings.mmr_strength_mp_dps,
        dps_max=settings.mmr_strength_mp_dps_max,
        guard_digits=settings.mmr_strength_mp_guard_digits,
        nodes_body=settings.mmr_strength_mp_nodes_body,
        nodes_planet=settings.mmr_strength_mp_nodes_planet,
        tolerance=settings.mmr_strength_quadrature_tol,
    )


def needs_harmonic(result: averaging.CentreCurvature, settings: AnalysisSettings) -> bool:
    """Double precision cannot resolve the well (mmr_strength_mp 'auto' only): no stable centre or no converged
    curvature, or a curvature below mmr_strength_double_floor x |D0| (the rounding of <D> over the fit's span)."""
    if settings.mmr_strength_mp != MP_AUTO:
        return False
    if result.reason in (averaging.REASON_NO_CENTRE, averaging.REASON_NOT_CONVERGED):
        return True
    level = abs(result.level)
    return bool(result.measured and np.isfinite(level) and result.curvature < settings.mmr_strength_double_floor * level)


def _evaluations(series: MMRSeries, context: Context, episode: Episode, element_window_yr: float, settings: AnalysisSettings):
    """(planet name, mu, saved, list of (time, body ellipse, phase of sigma, planet ellipse)) of every evaluation
    (snapshot x planet phase); the list is None without the planet's mass or orbit."""
    spec = series.resonance
    name = spec.planets[0]
    mu = context.planet_mass.get(name, NAN)
    saved = name in series.planets
    if not (np.isfinite(mu) and mu > 0):
        return name, mu, saved, None
    t = np.asarray(series.t_yr, float)
    step = float(np.median(np.diff(t)))
    width = element_window_yr if np.isfinite(element_window_yr) and element_window_yr > step else step
    phases = settings.mmr_strength_unsaved_planet_phases
    turns = [0.0] if saved else list(TWO_PI * np.arange(phases) / phases)
    out = []
    for time in snapshot_times(episode.t_start_yr, episode.t_end_yr, settings.mmr_strength_snapshots):
        part = _window(t, time, width)
        body, body_phase = body_ellipse(series, part)
        for turn in turns:
            planet = saved_planet(series, name, part) if saved else mean_planet(context, name, turn)
            if planet is None:
                return name, mu, saved, None
            ellipse, varpi_p, node_p = planet
            out.append((time, body, body_phase + planet_phase(spec, varpi_p, node_p), ellipse))
    return name, mu, saved, out


def direct_theory(
    series: MMRSeries, context: Context, episode: Episode, element_window_yr: float, settings: AnalysisSettings
) -> DirectTheory:
    """mu_eff_theory = mu_P a <D>''(sigma_c) at the snapshots of `episode` (module docstring); reason when missing.
    Where double precision cannot resolve the well (`needs_harmonic`), every evaluation is the resonant harmonic in
    extended precision instead (`harmonic_theory`)."""
    spec = series.resonance
    _, mu, saved, evaluations = _evaluations(series, context, episode, element_window_yr, settings)
    if evaluations is None:
        return DirectTheory((), NAN, NAN, NAN, 0, '', REASON_NO_PLANET)
    rule = quadrature(settings)
    values: List[float] = []
    values_all: List[float] = []
    a_all: List[float] = []
    e_all: List[float] = []
    centres, offsets, distances, nodes = [], [], [], []
    for time, body, phase, ellipse in evaluations:
        sigma_observed = observed_centre(episode, time)
        lo, hi = observed_swing(episode, time)
        result = averaging.centre_curvature(body, ellipse, spec.body_coefficient, spec.planet_coefficients[0], phase, sigma_observed, rule)
        if needs_harmonic(result, settings):
            return harmonic_theory(series, context, episode, evaluations, mu, saved, settings)
        nodes.append(result.nodes_per_turn)
        distance = result.min_distance_au / hill_radius(ellipse.a, mu)
        distances.append(distance)
        if not result.measured:
            return _missing(result.reason, centres, offsets, distances, nodes, saved)
        if distance < settings.mmr_strength_min_distance_hill:
            return _missing(REASON_CLOSE_APPROACH, centres, offsets, distances, nodes, saved)
        value = float(mu * body.a * result.curvature)
        values_all.append(value)
        a_all.append(body.a)
        e_all.append(body.orbit.e)
        # no half-cycle (fewer than two turning points): nothing to test against, the evaluation counts
        no_swing = not (np.isfinite(lo) and np.isfinite(hi))
        if no_swing or averaging.swing_in_well(result.centre_rad, result.global_max_rad, lo, hi):
            values.append(value)
        centres.append(result.centre_rad)
        offsets.append(result.offset_rad)
    return DirectTheory(
        tuple(values),
        float(np.degrees(centres[0])),
        float(np.degrees(np.max(offsets))),
        float(np.min(distances)),
        int(np.max(nodes)),
        PLANET_SAVED if saved else PLANET_PHASE_SCAN,
        '' if values else REASON_CENTRE_OUTSIDE_WELL,
        tuple(values_all),
        len(values),
        METHOD_DIRECT,
        tuple(a_all),
        tuple(e_all),
    )


def harmonic_theory(
    series: MMRSeries, context: Context, episode: Episode, evaluations, mu: float, saved: bool, settings: AnalysisSettings
) -> DirectTheory:
    """mu_eff_theory = mu_P a 2 |C| of the resonant harmonic in extended precision at every evaluation
    (`averaging.harmonic_amplitude_mp`; method direct_harmonic). There is no well to test the observed swing against
    (the double-precision profile shows none), so every evaluation counts. The body must keep
    mmr_strength_min_distance_hill Hill radii from the planet along the average at the harmonic's minimum, checked
    first (an encounter is outside the averaged problem whatever the harmonic)."""
    spec = series.resonance
    k, k_p = spec.body_coefficient, spec.planet_coefficients[0]
    rule = harmonic_rule(settings)
    values: List[float] = []
    a_all: List[float] = []
    e_all: List[float] = []
    centres, offsets, distances = [], [], []
    nodes = settings.mmr_strength_nodes_per_turn
    elements = PLANET_SAVED if saved else PLANET_PHASE_SCAN
    for time, body, phase, ellipse in evaluations:
        harmonic = averaging.harmonic_amplitude_mp(body, ellipse, k, k_p, phase, rule)
        if np.isfinite(harmonic.centre_rad):
            line = averaging.LineAverage(body, ellipse, k, k_p, phase, nodes)
            closest = float(line.evaluate([int(round(harmonic.centre_rad * nodes / TWO_PI))])[2][0])
            distances.append(closest / hill_radius(ellipse.a, mu))
            if distances[-1] < settings.mmr_strength_min_distance_hill:
                return _missing(REASON_CLOSE_APPROACH, centres, offsets, distances, [nodes], saved, METHOD_HARMONIC)
        if not harmonic.measured:
            return _missing(harmonic.reason, centres, offsets, distances, [nodes], saved, METHOD_HARMONIC)
        values.append(float(mu * body.a * harmonic.curvature))
        a_all.append(body.a)
        e_all.append(body.orbit.e)
        centres.append(harmonic.centre_rad)
        offsets.append(float(averaging.circular_distance(harmonic.centre_rad, observed_centre(episode, time))))
    return DirectTheory(
        tuple(values),
        float(np.degrees(centres[0])),
        float(np.degrees(np.max(offsets))),
        float(np.min(distances)),
        nodes,
        elements,
        '',
        tuple(values),
        len(values),
        METHOD_HARMONIC,
        tuple(a_all),
        tuple(e_all),
    )


def _missing(reason, centres, offsets, distances, nodes, saved, method: str = METHOD_DIRECT) -> DirectTheory:
    return DirectTheory(
        (),
        float(np.degrees(centres[0])) if centres else NAN,
        float(np.degrees(np.max(offsets))) if offsets else NAN,
        float(np.min(distances)) if distances else NAN,
        int(np.max(nodes)) if nodes else 0,
        PLANET_SAVED if saved else PLANET_PHASE_SCAN,
        reason,
        method=method,
    )


# --- the strength of a record --------------------------------------------------------------------------------------


def analyse_strength(
    context: Context,
    episodes: Sequence[EpisodePhysics],
    settings: AnalysisSettings,
    series: Optional[MMRSeries] = None,
    geometry: Optional[Geometry] = None,
    element_window_yr: float = NAN,
) -> Strength:
    """Theory, required strength from the decisive episode's T0, ratio, the support check and the implausible flag.

    The decisive episode is the first of `episodes` with a measured mu_eff (the orchestration passes the decisive one
    first). A two-body angle that is not 1:1 (`direct_applicable`) with `series` and `geometry` is judged by the direct
    averaging at snapshots of that episode (elements averaged over `element_window_yr`, the cutoff P_c); anything else
    by the e^q scale (module docstring). mu_scale is a column on both paths.
    """
    legal = eq_legal(context.e_med, context.crosses, settings)
    scale = mu_scale(context.mu_scale_no_eq, context.e_med, context.order)
    k = body_coefficient(context)
    decisive = _decisive(episodes)
    required = decisive.mu_eff if decisive is not None else NAN
    spec = series.resonance if series is not None else None
    if series is not None and geometry is not None and direct_applicable(spec):
        return _direct_strength(context, decisive, required, scale, legal, k, series, geometry, element_window_yr, settings)
    omega0 = omega0_theory(k, context.mean_motion_rad_yr, scale)
    observed = _display_numbers(DirectTheory((), NAN, NAN, NAN, 0, '', ''), _decisive_run(geometry, decisive), required, k, context)
    reference = scale if legal else context.mu_scale_no_eq
    ratio = required / reference if np.isfinite(required) and reference > 0 else NAN
    three_body = spec is not None and spec.kind == ResonanceKind.THREE_BODY
    envelope = DivisorEnvelope('')
    if three_body and legal and np.isfinite(ratio) and ratio >= settings.mmr_strength_implausible_ratio:
        envelope = small_divisor_envelope(spec, context, ratio, _episode_period(geometry, decisive), _cycle_period(geometry), settings)
    return Strength(
        mu_scale=scale,
        mu_scale_no_eq=context.mu_scale_no_eq,
        eq_legal=legal,
        mu_eff_required=required,
        ratio=ratio,
        omega0_theory_rad_yr=omega0,
        period_theory_yr=_period(omega0),
        # a 1:1 angle: the harmonic pendulum's e^q scale does not describe a horseshoe or a quasi-satellite (2010 CG236
        # at 1 kyr: ratio 1800 on a resonant quasi-satellite), its own law judges the scale (W20 round C)
        implausible=implausible_strength(legal, ratio, settings) and not one_to_one(spec),
        check=CheckValue.NOT_APPLICABLE if one_to_one(spec) else strength_check(legal, ratio, settings),
        method=METHOD_SCALE,
        three_body=three_body,
        divisor_kind=envelope.kind,
        divisor_b=envelope.b,
        divisor_residual=envelope.residual,
        divisor_candidate=envelope.candidate,
        divisor_capped=envelope.capped,
        t_obs_yr=observed['t_obs_yr'],
        t_obs_source=observed.get('t_obs_source', ''),
    )


def _decisive_run(geometry: Optional[Geometry], decisive: Optional[EpisodePhysics]) -> Optional[Episode]:
    """The run (accepted or not) of the decisive episode; None without it."""
    if geometry is None or decisive is None:
        return None
    return run_of(geometry, decisive.episode_id)


def _episode_period(geometry: Optional[Geometry], decisive: Optional[EpisodePhysics]) -> float:
    """The period of the decisive episode [yr] (nan without it)."""
    episode = _decisive_run(geometry, decisive)
    return float(episode.period_med_yr) if episode is not None else NAN


def _cycle_period(geometry: Optional[Geometry]) -> float:
    return float(geometry.period_cycle_yr) if geometry is not None else NAN


def _period(omega0: float) -> float:
    return TWO_PI / omega0 if np.isfinite(omega0) and omega0 > 0 else NAN


def _direct_strength(context, decisive, required, scale, legal, k, series, geometry, element_window_yr, settings) -> Strength:
    episode = _decisive_run(geometry, decisive)
    if episode is None or not np.isfinite(required):
        theory = DirectTheory((), NAN, NAN, NAN, 0, '', REASON_NO_EPISODE)
    else:
        theory = direct_theory(series, context, episode, element_window_yr, settings)
    # outside the well: the theory is a value but no observed swing librates in its well, so F fails; its numbers
    # over every evaluation are kept for the columns only
    outside = theory.reason == REASON_CENTRE_OUTSIDE_WELL
    numbers = theory.values if theory.measured else theory.values_all if outside else ()
    low = high = ratio = ratio_low = ratio_high = omega0 = NAN
    if numbers:
        low, high = float(np.min(numbers)), float(np.max(numbers))
        ratio_low, ratio_high = required / high, required / low
        ratio = nearest_to_one(ratio_low, ratio_high)
        omega0 = omega0_theory(k, context.mean_motion_rad_yr, float(np.sqrt(low * high)))
    return Strength(
        mu_scale=scale,
        mu_scale_no_eq=context.mu_scale_no_eq,
        eq_legal=legal,
        mu_eff_required=required,
        ratio=ratio,
        omega0_theory_rad_yr=omega0,
        period_theory_yr=_period(omega0),
        implausible=implausible_strength(theory.measured or outside, ratio, settings),
        check=CheckValue.FAIL if outside else strength_check(theory.measured, ratio, settings),
        method=theory.method,
        mu_eff_theory_min=low,
        mu_eff_theory_max=high,
        ratio_min=ratio_low,
        ratio_max=ratio_high,
        centre_theory_deg=theory.centre_deg,
        centre_offset_max_deg=theory.offset_max_deg,
        min_distance_hill=theory.min_distance_hill,
        nodes_per_turn=theory.nodes_per_turn,
        n_evaluations=len(theory.values_all),
        n_consistent=theory.n_consistent,
        planet_elements=theory.planet_elements,
        not_measured_reason=theory.reason,
        **_display_numbers(theory, episode, required, k, context),
    )


def _display_numbers(theory: DirectTheory, episode: Optional[Episode], required: float, k: int, context: Context) -> dict:
    """The R1 columns of the figure's physics block (display only: no decision reads them). Over EVERY evaluation
    (`values_all`, in the well or not), each with n at its own averaged a: the theory's mu range, its small-amplitude
    period T0 = 2 pi / (|k| n sqrt(3 mu)), required/theory, and the separatrix half-width in a 2 omega0 / |c|, with
    c = 1.5 |k| n / a of the same snapshot: (4/3) a sqrt(3 mu) (Codex W20 A #15: the record's c mixed with a snapshot's n
    is off by (a_i / a_mean)^(-5/2)); the observed
    T0 of the decisive episode from mu_required (the same formula), where it came from, the window the snapshots were
    taken in and the averaged e there (never the record's instantaneous e range)."""
    out = {'t_obs_yr': _period(omega0_theory(k, context.mean_motion_rad_yr, required)) if np.isfinite(required) else NAN}
    if episode is not None:
        source = f'episode:{episode.episode_id}' if episode.accepted else f'rejected_run:{episode.t_start_yr:.6g}-{episode.t_end_yr:.6g}'
        out.update(t_obs_source=source, snap_t0_yr=float(episode.t_start_yr), snap_t1_yr=float(episode.t_end_yr))
    mu = np.asarray(theory.values_all, float)
    if not len(mu) or len(theory.a_all) != len(mu):
        return out
    a = np.asarray(theory.a_all, float)
    n = np.sqrt(GM_SUN / a**3)
    omega0 = abs(k) * n * np.sqrt(3 * mu)
    periods = TWO_PI / omega0
    with np.errstate(divide='ignore', invalid='ignore'):
        width = 2 * omega0 / (1.5 * abs(k) * n / a)
    out.update(
        mu_eff_theory_all_min=float(mu.min()),
        mu_eff_theory_all_max=float(mu.max()),
        period_theory_min_yr=float(periods.min()),
        period_theory_max_yr=float(periods.max()),
        ratio_all_min=float(required / mu.max()),
        ratio_all_max=float(required / mu.min()),
        e_snap_min=float(np.min(theory.e_all)),
        e_snap_max=float(np.max(theory.e_all)),
        sep_halfwidth_min_au=float(np.min(width)),
        sep_halfwidth_max_au=float(np.max(width)),
    )
    return out


def body_coefficient(context: Context, spec: Optional[ResonanceSpec] = None) -> int:
    """k of the body's lambda: from the spec, else the context's k, else back from c = -1.5 k n / a (0 when that is
    not finite: a <= 0 or no mean motion)."""
    if spec is not None:
        return spec.body_coefficient
    if context.body_coefficient:
        return int(context.body_coefficient)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = -context.identity_coefficient * context.a_mean_au / (1.5 * np.float64(context.mean_motion_rad_yr))
    return int(round(k)) if np.isfinite(k) else 0


def strength_columns(strength: Strength) -> dict:
    """CSV columns `str_*` (the body of `Strength.to_columns`)."""
    return {
        'str_method': strength.method,
        'str_mu_scale': strength.mu_scale,
        'str_mu_scale_no_eq': strength.mu_scale_no_eq,
        'str_eq_legal': strength.eq_legal,
        'str_mu_eff_req': strength.mu_eff_required,
        'str_mu_eff_theory_min': strength.mu_eff_theory_min,
        'str_mu_eff_theory_max': strength.mu_eff_theory_max,
        'str_ratio': strength.ratio,
        'str_ratio_min': strength.ratio_min,
        'str_ratio_max': strength.ratio_max,
        'str_omega0_theory_rad_yr': strength.omega0_theory_rad_yr,
        'str_P_lib_theory_kyr': strength.period_theory_yr / 1e3,
        'str_implausible': strength.implausible,
        'str_centre_theory_deg': strength.centre_theory_deg,
        'str_centre_offset_max_deg': strength.centre_offset_max_deg,
        'str_min_distance_hill': strength.min_distance_hill,
        'str_nodes_per_turn': strength.nodes_per_turn,
        'str_n_evaluations': strength.n_evaluations,
        'str_n_consistent': strength.n_consistent,
        'str_planet_elements': strength.planet_elements,
        'str_not_measured_reason': strength.not_measured_reason,
        'str_divisor_kind': strength.divisor_kind,
        'str_divisor_B': strength.divisor_b,
        'str_divisor_residual': strength.divisor_residual,
        'str_divisor_candidate': strength.divisor_candidate,
        'str_divisor_capped': strength.divisor_capped,
        'str_mu_eff_theory_all_min': strength.mu_eff_theory_all_min,
        'str_mu_eff_theory_all_max': strength.mu_eff_theory_all_max,
        'str_P_lib_theory_min_kyr': strength.period_theory_min_yr / 1e3,
        'str_P_lib_theory_max_kyr': strength.period_theory_max_yr / 1e3,
        'str_ratio_all_min': strength.ratio_all_min,
        'str_ratio_all_max': strength.ratio_all_max,
        'str_T_obs_yr': strength.t_obs_yr,
        'str_T_obs_source': strength.t_obs_source,
        'str_snap_t0_yr': strength.snap_t0_yr,
        'str_snap_t1_yr': strength.snap_t1_yr,
        'str_e_snap_min': strength.e_snap_min,
        'str_e_snap_max': strength.e_snap_max,
        'str_sep_halfwidth_min_au': strength.sep_halfwidth_min_au,
        'str_sep_halfwidth_max_au': strength.sep_halfwidth_max_au,
    }
