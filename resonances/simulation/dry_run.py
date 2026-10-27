"""Dry-run: what a run will see, checked before it is started (SPEC §9; author item 10; docs/dry-run.md).

Two levels:

- `static`: the bodies' initial elements only, no integration (milliseconds per body). Per MMR row: the
  expected rate of the angle over a plausible range of a, the alias branch and verdict at the output step, the
  ghost "resonances" of that step, the steps the record can afford, and the cutoff the analysis will choose; per
  body: the integrator and its step; per run: the 'auto' output step and the cost.
- `pilot`: one short, finely sampled integration per body (the run's integrator and dt, `DRYRUN_PILOT_*`) that
  measures what the static level guesses: the range and short-period scatter of a, the precession, the leak of
  short-period lines into the filtered angle and into the filtered a_dot at the chosen step (and whether that step
  is acceptable although it is above the short-period Nyquist step), and the slow lines the filter passes.

Nothing here changes a verdict. The analysis' own rules are reused (`analysis.mmr.sampling`, `preprocess`,
`context`), so the dry-run predicts what the analysis will compute.

Formulas (a in au, t in Gaussian years of 2 pi REBOUND units, n = 2 pi a^(-3/2) rad/yr):

- Rate of sigma = sum_j k_j lambda_j + k lambda + p varpi + sum_j p_j varpi_j (+ nodes):
  nu(a) = sum_j k_j n_j + k n(a) + slow (Murray & Dermott 1999, p. 331, eq. 8.22), n_j the package model's mean
  motions (`const.PLANET_N_DEG_PER_GAUSS_YR`), slow = p varpi_dot + sum_j p_j g_j + v Omega_dot + sum_j v_j s_j.
  Static varpi_dot: the Laplace-Lagrange free precession of a test particle,
  A = (n/4) sum_j mu_j alpha_j alphabar_j b_{3/2}^{(1)}(alpha_j), Omega_dot = -A (Murray & Dermott 1999, sec. 7.9,
  from knowledge, equation number not checked); not used for planet-crossers (alpha -> 1) nor retrograde orbits,
  where slow = sum_j p_j g_j + v_j s_j only. It is wrong inside a resonance (1362 Griqua: -115 measured
  against +104 arcsec/yr), but its term moves the rate by at most |p| x 0.3 arcsec/yr ... 0.3 deg/yr: never the
  alias branch. The pilot measures it.
- Alias: branch m = round(nu dt / 2 pi), apparent rate nu - 2 pi m / dt (Knezevic & Milani 2001, p. 22), verdict
  over [a_lo, a_hi] by `sampling.step_verdict` with MMR_STEP_RESOLVED_DEG.
- Nyquist half-width: |nu| dt = pi at |a - a*| = delta a_N = a^(5/2) / (3 |k| dt), from
  d nu / d a = -(3/2) k n / a (linearised; `sampling.nyquist_halfwidth`).
- Ghosts: the sampled angle stands still at a_m where nu(a_m) = 2 pi m / dt, i.e. n(a_m) = n* + 2 pi m / (k dt)
  (`sampling.ghost_axis`); near a_m it obeys the a <-> sigma identity of a real resonance, hence the U-curves and
  "librations" of an aliased record (1999 UQ 7E+2M-9 at 5.0012 yr: a_1 is 1.2e-4 au from a0).
- Largest safe steps, with eta = MMR_STEP_RESOLVED_DEG / 180 deg: plain np.unwrap eta pi / max|nu|; predictive
  unwrap (model error from the short-period scatter delta a_sp of a) eta a^(5/2) / (3 |k| delta a_sp).
- Perihelion jitter: |p| delta e_sp / e > MMR_JITTER_MAX_RAD (varpi undefined at small e). Static delta e_sp is
  taken as delta a_sp / a: short-period terms of a/a and of e have the same order (mu) in Lagrange's equations;
  an order of magnitude only, the pilot measures it.
- Cutoff: the analysis' rule `preprocess.choose_cutoff`, P_c = max(MMR_FILTER_SYNODIC_FACTOR x longest P_syn,
  P_floor / FILTER_MIN_LIBRATION_CUTOFFS), P_floor the shortest libration the angle can have
  (`preprocess.libration_period_floor`: MMR_LIBRATION_PERIOD_MIN_YR, or for a co-orbital
  MMR_COORBITAL_PERIOD_FLOOR_FACTOR x the tadpole period). The step does not enter.
- Validity of the step (the analysis' G0): P_c <= 2 dt -> invalid / undersampled (`preprocess.filter_realizable`):
  where the floor term sets P_c, dt >= 37.5 yr for a non-co-orbital MMR, 18.5 yr for a Jupiter Trojan; the resample
  step is the largest of MMR_RESAMPLE_STEPS_YR <= P_floor / MMR_RESAMPLE_STEPS_PER_CYCLE within the alias bounds and
  below P_c / 2 (`sampling.resample_step`: 2.5 yr for a Jupiter Trojan). That bound is also the 'libration' limit of
  the auto step (150 / 20 = 7.5 yr), P_c / 2 (exclusive) its 'filter' limit.
- Short periods: dt >= min(P_orb, P_syn of the angle's planets) / 2 (`sampling.short_period_step`) folds the
  short-period terms of the elements into slow lines: flag short_period_aliased, confidence <= medium in the
  analysis; the 'short_period' limit of the auto step. Below it, a harmonic m <= MMR_SHORT_PERIOD_HARMONICS of P_orb
  or P_syn folded slower than the period floor sets the same flag (column folded_slowest_yr, short_period_trigger
  'harmonic'): the row reads the analysis' own predicate (`sampling.short_period_trigger`) on its own inputs (the
  static level's P_orb(a0), the analysis' P_orb(<a>): the flags agree for equal inputs; the pilot replans with the
  measured ones). The auto step avoids such steps (`fold_free_step`). The pilot accepts such a step when both of its leaks are
  small: 2 x the leak of sigma_f <= the TP prominence max(MMR_TP_PROMINENCE_FLOOR_DEG, MMR_TP_RIPPLE_FACTOR x r)
  (a wave of half-amplitude r turns by up to 2 r; r from the fine pilot as `analyse.fast_residual`, on the raw angle:
  the pilot does not subtract the planets' ripple), and the relative error eps of the filtered a_dot
  <= sqrt(b / (1 - b)), b = CONFIDENCE_THIN_BAND (0.33: an error uncorrelated with a_dot leaves a force law
  R^2 = 1 / (1 + eps^2) >= 1 - b; a correlated one can cost up to eps^2, 1 - eps^2 = 0.89). A sampling gate, not a
  certificate of the force law: the analysis still measures it. The pilot's fine step divides the planned step, so
  the leak is measured at that cadence. The record itself cannot measure either (inv-09: 0.1-0.3 deg and 2-8 % of a
  in the Trojans at 16 yr, 12.5 deg and 190 % in 62243 1M-4J-1).
- Baseline: (LIBRATION_MIN_TURNING_POINTS / 2) cycles of the shortest libration plus FILTER_EDGE_TRIM_CUTOFFS x P_c
  trimmed at each end.
- Integrator: ias15 when the radial band [q, Q] overlaps a planet's [q_j - h R_H, Q_j + h R_H]
  (R_H = a_j (mu_j / 3)^(1/3), h = DRYRUN_HILL_MARGIN; ias15 has adaptive steps for close encounters, Rein &
  Spiegel 2015, abstract) or when q < DRYRUN_NEA_Q (nodes precess: a non-crossing NEA today is not one in 20 kyr);
  else SABA(10,6,4) (splitting integrators assume no close encounters, Rein, Tamayo & Brown 2019, p. 2) with
  dt = min(the run's dt, P / DRYRUN_STEPS_PER_ORBIT, tau_peri / DRYRUN_STEPS_PER_PERI), where
  tau_peri = P sqrt((1 - e)^3 / (1 + e)) is the effective period at pericentre (Wisdom 2015, via Lu, Hernandez &
  Rein 2024, eq. 33, p. 9; they needed tau_peri / 50 for WHFast at e -> 0.99; the /10 rule is provisional).
- FAIR: x = M mod 360 deg must advance < ~90 deg per step: dt <= DRYRUN_FAIR_FRACTION x P.
- OUTPUT_STEP='auto': dt = min(libration, short_period, filter, trend, FAIR if DRYRUN_FAIR=required,
  DRYRUN_MAX_STEP_YR), rounded down to DRYRUN_AUTO_STEPS_YR (short_period and filter exclusive: a step on the bound
  itself is flagged or undersampled), then walked down DRYRUN_AUTO_STEPS_YR to the first step at which no row folds a
  harmonic slower than its period floor (binding rule 'short_period_harmonic' when it moved; P_orb 10.001 yr: 5 yr
  folds 2 / P_orb to 50 kyr, 2.5 yr folds nothing); Nout = span / dt + 1. A configured step whose harmonic row the
  pilot rejects is reported above the 'short_period_harmonic' limit, the fold-free step below it. Defined for runs
  with MMRs only (a purely secular run keeps Nout).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from resonances.data import const
from resonances.logger import logger
from resonances.resonance.analysis.filtering import butterworth_gain, edge_window, lowpass, window_mask
from resonances.resonance.analysis.mmr import context as context_module
from resonances.resonance.analysis.mmr import preprocess, sampling
from resonances.resonance.analysis.mmr.models import PLANET_NAMES_ORDER, ResonanceSpec, StepVerdict

LEVELS = ('static', 'pilot')
SHORT_PERIOD_HARMONIC_LIMIT = 'short_period_harmonic'  # the auto step moved below a step that folds a harmonic
TWO_PI = 2 * math.pi
ARCSEC = math.pi / (180 * 3600)
BODY_CLASSES = ('main', 'nea', 'crosser')  # the order of DRYRUN_DA_STATIC and DRYRUN_DA_SP_STATIC

# Cost coefficients measured on the author's Mac (SAMPLING.md §8, 2026-10-15): benchmark runs A/B, the asteroid
# pairs runs, rebound timing. They estimate, they decide nothing.
GZ_BYTES_PER_VALUE = 7.9  # data-<body>.csv.gz: 3.0 MB / (20001 rows x 19 columns), 1999 UQ R run
DATA_COLUMNS_BASE = 10  # time and elements of the body
DATA_COLUMNS_PER_ANGLE = 4  # angle, unwrapped, filtered, filtered unwrapped
RAM_BYTES_PER_ROW_PER_ANGLE = {'mmr': 17.0, 'secular': 106.0}  # pairs runs: 95 MB at 83k rows x 68 MMRs; 0.7 GB at 100k x 66
PACKAGE_S_PER_ANGLE_ROW = 1.2e-5  # bookkeeping, analysis, saving; pairs run (3 bodies, 68 angles, 83k rows, 7 min)
IAS15_S_PER_KYR = 2.1  # one NEA per Simulation (B runs, 100 kyr in 207-220 s)
IAS15_S_PER_ROW = 1.2e-4
SABA_S_PER_KYR_PER_TASK = 0.14  # 8 bodies + planets at dt = 1 (A runs)
SABA_S_PER_ROW_PER_TASK = 1.33e-4
SABA_BODIES_PER_TASK = 8
BYTES_PER_GB = 1e9
SECONDS_PER_HOUR = 3600.0
LAPLACE_GRID = 512  # trapezoid nodes for b_{3/2}^{(1)}: periodic integrand, exponentially convergent
MOID_GRID = 360  # true anomalies per orbit for the MOID screen (resolution 2 pi r / 360; information only)
SABA = 'SABA(10,6,4)'
IAS15 = 'ias15'


# ----------------------------------------------------------------------------------------------- orbit physics
def orbital_period_yr(a_au: float) -> float:
    """P = a^(3/2) [yr] (test particle)."""
    return float(a_au) ** 1.5


def tau_peri_yr(a_au: float, e: float) -> float:
    """Effective period at pericentre P sqrt((1 - e)^3 / (1 + e)) [yr] (Wisdom 2015; Lu, Hernandez & Rein 2024, eq. 33)."""
    return orbital_period_yr(a_au) * math.sqrt((1 - e) ** 3 / (1 + e))


def hill_radius_au(planet: str) -> float:
    """R_H = a_j (mu_j / 3)^(1/3) [au]."""
    return const.PLANETS_AXIS[planet] * (const.PLANETS_MASS[planet] / 3.0) ** (1.0 / 3.0)


def crossing_planets(q_au: float, big_q_au: float, hill_margin: float) -> Tuple[str, ...]:
    """Planets whose band [q_j - h R_H, Q_j + h R_H] overlaps [q, Q] (J2000 a_j, e_j; inclinations ignored: nodes
    precess). One Hill radius, not three: 3 R_H of Jupiter (1.07 au) would send every Griqua to ias15."""
    out = []
    for planet in const.SOLAR_SYSTEM:
        a, e = const.PLANETS_AXIS[planet], const.PLANETS_ECCENTRICITY[planet]
        band = hill_margin * hill_radius_au(planet)
        if q_au < a * (1 + e) + band and big_q_au > a * (1 - e) - band:
            out.append(planet)
    return tuple(out)


def body_class(q_au: float, crossing: Tuple[str, ...], nea_q_au: float) -> str:
    """'crosser' (a planet band is crossed), 'nea' (q < DRYRUN_NEA_Q) or 'main'."""
    if crossing:
        return 'crosser'
    return 'nea' if q_au < nea_q_au else 'main'


def recommend_integrator(a_au: float, e: float, crossing: Tuple[str, ...], dt_run_yr: float, dry) -> dict:
    """ias15 for crossers and NEAs, else SABA(10,6,4) with dt = min(run dt, P / steps_per_orbit, tau_peri / steps_per_peri)."""
    q = a_au * (1 - e)
    if crossing:
        return {
            'integrator': IAS15,
            'dt_yr': float('nan'),
            'why': f"crosses the band of {', '.join(crossing)}: a fixed step does not resolve encounters",
        }
    if q < dry.dryrun_nea_q:
        return {
            'integrator': IAS15,
            'dt_yr': float('nan'),
            'why': f'NEA (q = {q:.3f} < {dry.dryrun_nea_q} au): encounters come as the node and perihelion precess',
        }
    period, tau = orbital_period_yr(a_au), tau_peri_yr(a_au, e)
    bounds = {
        'run dt': dt_run_yr,
        'P / steps per orbit': period / dry.dryrun_steps_per_orbit,
        'tau_peri / steps per peri': tau / dry.dryrun_steps_per_peri,
    }
    binding = min(bounds, key=bounds.get)
    return {'integrator': SABA, 'dt_yr': float(bounds[binding]), 'why': f'no crossing; dt bound by {binding}'}


def laplace_coefficient(s: float, j: int, alpha: float) -> float:
    """b_s^{(j)}(alpha) = (1/pi) int_0^{2 pi} cos(j psi) / (1 - 2 alpha cos psi + alpha^2)^s d psi (trapezoid, periodic)."""
    psi = np.linspace(0.0, TWO_PI, LAPLACE_GRID, endpoint=False)
    integrand = np.cos(j * psi) / (1 - 2 * alpha * np.cos(psi) + alpha**2) ** s
    return float(2 * np.mean(integrand))


def laplace_lagrange_precession(a_au: float) -> float:
    """Laplace-Lagrange free precession of a test particle A [rad/yr] (Murray & Dermott 1999, sec. 7.9):
    A = (n/4) sum_j mu_j alpha_j alphabar_j b_{3/2}^{(1)}(alpha_j); alpha_j = a_j/a, alphabar_j = 1 for an inner planet,
    alpha_j = alphabar_j = a/a_j for an outer one."""
    n = TWO_PI * a_au**-1.5
    total = 0.0
    for planet in const.SOLAR_SYSTEM:
        a_j = const.PLANETS_AXIS[planet]
        alpha, alpha_bar = (a_j / a_au, 1.0) if a_j < a_au else (a_au / a_j, a_au / a_j)
        total += const.PLANETS_MASS[planet] * alpha * alpha_bar * laplace_coefficient(1.5, 1, alpha)
    return 0.25 * n * total


def _orbit_positions(a, e, inc, Omega, omega) -> np.ndarray:
    nu = np.linspace(0.0, TWO_PI, MOID_GRID, endpoint=False)
    r = a * (1 - e * e) / (1 + e * np.cos(nu))
    u = omega + nu
    x = r * (math.cos(Omega) * np.cos(u) - math.sin(Omega) * np.sin(u) * math.cos(inc))
    y = r * (math.sin(Omega) * np.cos(u) + math.cos(Omega) * np.sin(u) * math.cos(inc))
    z = r * np.sin(u) * math.sin(inc)
    return np.stack([x, y, z], axis=-1)


def moid_au(body: dict, planet: dict) -> float:
    """Minimum distance between two Keplerian ellipses on a grid of true anomalies [au] (a screen, not a MOID solver)."""
    if not (0 <= body['e'] < 1 and 0 <= planet['e'] < 1):
        return float('nan')
    one = _orbit_positions(body['a'], body['e'], body['inc'], body['Omega'], body['omega'])
    other = _orbit_positions(planet['a'], planet['e'], planet['inc'], planet['Omega'], planet['omega'])
    return float(np.sqrt(np.min(np.sum((one[:, None, :] - other[None, :, :]) ** 2, axis=-1))))


# ----------------------------------------------------------------------------------------------- inputs
@dataclass
class BodyInputs:
    """What the row rules need about one body: from the elements (static) or measured (pilot)."""

    name: str
    a_au: float
    e: float
    a_lo_au: float
    a_hi_au: float
    da_sp_au: float
    de_sp: float
    varpi_rate_rad_yr: Optional[float]  # None: Laplace-Lagrange does not apply (crosser, retrograde)
    node_rate_rad_yr: Optional[float]
    source: str = 'static'


def _planet_tables() -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """n_j, g_j, s_j [rad/yr] and mu_j of the package model (`const`)."""
    names = PLANET_NAMES_ORDER
    planet_n = {name: context_module.table_mean_motion(name) for name in names}
    g = {name: context_module.planet_frequency(name, 'g') for name in names}
    s = {name: context_module.planet_frequency(name, 's') for name in names}
    masses = {name: const.PLANETS_MASS[name] for name in names}
    return planet_n, g, s, masses


def slow_rate(spec: ResonanceSpec, inputs: BodyInputs, g: Dict[str, float], s: Dict[str, float]) -> float:
    """p varpi_dot + v Omega_dot + sum_j (p_j g_j + v_j s_j) [rad/yr]; the body's terms are 0 when unknown."""
    varpi = 0.0 if inputs.varpi_rate_rad_yr is None else inputs.varpi_rate_rad_yr
    node = 0.0 if inputs.node_rate_rad_yr is None else inputs.node_rate_rad_yr
    return context_module.slow_rate(spec, varpi, g, node, s)


def static_inputs(name: str, elements: dict, dry) -> Tuple[BodyInputs, dict]:
    """BodyInputs from the initial elements and the body's row of `bodies` (class, integrator...)."""
    a, e = float(elements['a']), float(elements['e'])
    q, big_q = a * (1 - e), a * (1 + e)
    crossing = crossing_planets(q, big_q, dry.dryrun_hill_margin)
    kind = body_class(q, crossing, dry.dryrun_nea_q)
    index = BODY_CLASSES.index(kind)
    da, da_sp = dry.dryrun_da_static[index], dry.dryrun_da_sp_static[index]
    retrograde = float(elements.get('inc', 0.0)) > math.pi / 2
    precession = None if (crossing or retrograde) else laplace_lagrange_precession(a)
    inputs = BodyInputs(
        name=name,
        a_au=a,
        e=e,
        a_lo_au=a - da,
        a_hi_au=a + da,
        da_sp_au=da_sp,
        de_sp=da_sp / a,
        varpi_rate_rad_yr=precession,
        node_rate_rad_yr=None if precession is None else -precession,
    )
    body = {
        'body': name,
        'a': a,
        'e': e,
        'inc_deg': math.degrees(float(elements.get('inc', float('nan')))),
        'q': q,
        'Q': big_q,
        'P_yr': orbital_period_yr(a),
        'class': kind,
        'crossing': ','.join(crossing),
        'tau_peri_yr': tau_peri_yr(a, e),
        'fair_step_yr': dry.dryrun_fair_fraction * orbital_period_yr(a),
        'a_spread_au': da,
        'da_sp_au': da_sp,
        'de_sp': da_sp / a,
        'varpi_rate_arcsec_yr': float('nan') if precession is None else precession / ARCSEC,
        'source': 'static',
    }
    return inputs, body


# ----------------------------------------------------------------------------------------------- one MMR row
def _row_limits(spec: ResonanceSpec, inputs: BodyInputs, settings, tables) -> dict:
    """The step-independent part of a row: rates over [a_lo, a_hi], a*, safe steps, libration floor, jitter."""
    planet_n, g, s, masses = tables
    slow = slow_rate(spec, inputs, g, s)
    k = spec.body_coefficient
    rate = context_module.expected_rate(spec, inputs.a_au, planet_n, slow)
    rates = [context_module.expected_rate(spec, value, planet_n, slow) for value in (inputs.a_lo_au, inputs.a_hi_au)]
    lo, hi = min(rates), max(rates)
    a_res = context_module.exact_axis(spec, planet_n, slow)
    safe_plain, safe_pred = sampling.safe_steps(lo, hi, inputs.a_au, k, inputs.da_sp_au, settings.mmr_step_resolved_deg / 180.0)
    floor, floor_setting = preprocess.libration_period_floor(
        spec, SimpleNamespace(planet_mean_motion_rad_yr=planet_n, planet_mass=masses), settings
    )
    n_body = float(context_module.mean_motion(inputs.a_au))
    planet = preprocess.coorbital_planet(spec)  # its synodic angle is the resonant angle: no bound (`choose_cutoff`)
    synodic_all = context_module.synodic_periods(n_body, planet_n, spec.planets, spec.retrograde)
    synodic = {name: period for name, period in synodic_all.items() if name != planet}
    cutoff = preprocess.choose_cutoff(synodic, 1.0, settings, floor).cutoff_yr  # independent of the step
    step_libration = floor / settings.mmr_resample_steps_per_cycle if floor > 0 else math.inf
    synodic_cutoff = settings.mmr_filter_synodic_factor * max(synodic.values()) if synodic else 0.0
    jitter = abs(spec.body_varpi_coefficient) * inputs.de_sp / inputs.e if spec.body_varpi_coefficient and inputs.e > 0 else 0.0
    return {
        'slow': slow,
        'rate': rate,
        'rate_lo': lo,
        'rate_hi': hi,
        'a_res': a_res,
        'safe_plain': safe_plain,
        'safe_pred': safe_pred,
        'safe_trend': safe_pred if settings.mmr_unwrap == 'predictive' else safe_plain,
        'floor': floor,
        'floor_setting': floor_setting,
        'coorbital': planet is not None,
        'synodic': synodic,
        'synodic_cutoff': synodic_cutoff,
        'cutoff': cutoff,
        'step_valid': cutoff / 2,  # a step at or above it: P_c <= 2 dt, invalid / undersampled
        'step_filter': math.nextafter(cutoff / 2, 0.0) if cutoff > 0 else 0.0,  # the largest step that filters P_c
        'step_libration': step_libration,
        'step_short': sampling.short_period_step(orbital_period_yr(inputs.a_au), synodic_all),
        'orbital_period': orbital_period_yr(inputs.a_au),
        'synodic_all': synodic_all,
        'jitter': jitter,
    }


def _nearest_ghost(spec: ResonanceSpec, a_au: float, rate: float, slow: float, step_yr: float, planet_n) -> float:
    """The ghost a_m of the branch of nu(a) (or the nearer of m = +-1 when that branch is 0) [au]."""
    branch = sampling.branch_index(rate, step_yr)
    candidates = (branch,) if branch != 0 else (1, -1)
    ghosts = [sampling.ghost_axis(spec, planet_n, slow, step_yr, m) for m in candidates]
    ghosts = [ghost for ghost in ghosts if math.isfinite(ghost)]
    return min(ghosts, key=lambda ghost: abs(ghost - a_au)) if ghosts else float('nan')


def _row_at_step(spec, inputs: BodyInputs, limits: dict, step_yr: float, span_yr: float, settings, dry, tables) -> dict:
    """The row of `rows` at the output step: verdict, branches, ghost, cutoff, baseline, action, warnings."""
    planet_n = tables[0]
    k = spec.body_coefficient
    verdict, branches = sampling.step_verdict(limits['rate_lo'], limits['rate_hi'], step_yr, math.radians(settings.mmr_step_resolved_deg))
    halfwidth = sampling.nyquist_halfwidth(inputs.a_au, k, step_yr)
    ghost = _nearest_ghost(spec, inputs.a_au, limits['rate'], limits['slow'], step_yr, planet_n)
    cutoff = preprocess.choose_cutoff(limits['synodic'], step_yr, settings, limits['floor'])
    baseline = 0.5 * settings.libration_min_turning_points * limits['floor'] + 2 * settings.filter_edge_trim_cutoffs * cutoff.cutoff_yr
    distance = abs(inputs.a_au - limits['a_res'])
    undersampled = not preprocess.filter_realizable(cutoff.cutoff_yr, step_yr)
    action = 'resample' if undersampled else _action(verdict, step_yr, limits, distance, halfwidth, settings, dry)
    folded = sampling.folded_slowest_yr(limits['orbital_period'], limits['synodic_all'], step_yr, settings.mmr_short_period_harmonics)
    trigger = sampling.short_period_trigger(step_yr, limits['step_short'], folded, limits['floor'])  # the analysis' predicate
    row = {
        'body': inputs.name,
        'resonance': spec.key,
        'type': str(spec.kind),
        'k': k,
        'p': spec.body_varpi_coefficient,
        'a0': inputs.a_au,
        'a_lo': inputs.a_lo_au,
        'a_hi': inputs.a_hi_au,
        'a_res': limits['a_res'],
        'da_res': inputs.a_au - limits['a_res'],
        'rate_deg_yr': math.degrees(limits['rate']),
        'rate_min': math.degrees(limits['rate_lo']),
        'rate_max': math.degrees(limits['rate_hi']),
        'step_yr': step_yr,
        'turns_per_step': limits['rate'] * step_yr / TWO_PI,
        'branches': ' '.join(str(b) for b in branches),
        'verdict': str(verdict),
        'apparent_deg_yr': math.degrees(sampling.apparent_rate(limits['rate'], step_yr)),
        'nyquist_halfwidth_au': halfwidth,
        'ghost_axis_au': ghost,
        'nearest_ghost_au': abs(ghost - inputs.a_au) if math.isfinite(ghost) else float('nan'),
        'max_safe_step_plain': limits['safe_plain'],
        'max_safe_step_pred': limits['safe_pred'],
        'jitter': limits['jitter'],
        'jitter_flag': bool(limits['jitter'] > settings.mmr_jitter_max_rad),
        'cutoff_pred_yr': cutoff.cutoff_yr,
        'cutoff_binding': cutoff.binding,
        'samples_per_cutoff': cutoff.cutoff_yr / step_yr,
        'libration_floor_yr': limits['floor'],
        'step_valid_yr': limits['step_valid'],
        'step_libration_yr': limits['step_libration'],
        'step_short_period_yr': limits['step_short'],
        'folded_slowest_yr': folded,
        'short_period_trigger': trigger,
        'baseline_min_yr': baseline,
        'baseline_ok': bool(span_yr >= baseline),
        'expected_status': 'invalid' if undersampled else '',
        'expected_reason': 'undersampled' if undersampled else '',
        'expected_angle_status': 'circulation' if verdict == StepVerdict.ALIASED_UNAMBIGUOUS and not undersampled else '',
        'expected_flags': 'short_period_aliased' if trigger else '',
        'action': action,
        'resample_step_yr': _undersampled_step(limits, step_yr, settings) if undersampled else float('nan'),
        'leak_deg': float('nan'),
        'leak_adot_rel': float('nan'),
        'leak_tp_prominence_deg': float('nan'),
        'pilot_step_ok': None,
        'leak_step_yr': float('nan'),
        'forced_terms': '',
        'source': inputs.source,
    }
    row['warnings'] = '; '.join(_row_warnings(row, limits, span_yr, settings))
    return row


def _undersampled_step(limits: dict, step_yr: float, settings) -> float:
    """The resample step of an undersampled row, as the analysis names it (`analyse._undersampled`): the largest of
    MMR_RESAMPLE_STEPS_YR <= P_floor / MMR_RESAMPLE_STEPS_PER_CYCLE within the alias bounds and below P_c / 2; nan when
    none fits."""
    step = sampling.resample_step(limits['safe_plain'], limits['safe_pred'], limits['floor'], step_yr, settings, limits['cutoff'])
    return float('nan') if step is None else float(step)


def _action(verdict, step_yr: float, limits: dict, distance_au: float, halfwidth_au: float, settings, dry) -> str:
    """none | predictive_unwrap | resample | drop (SPEC §9.3)."""
    if verdict == StepVerdict.RESOLVED:
        return 'none'
    if dry.dryrun_drop_aliased and verdict == StepVerdict.ALIASED_UNAMBIGUOUS and distance_au > dry.dryrun_drop_ghost_factor * halfwidth_au:
        return 'drop'
    if settings.mmr_unwrap == 'predictive' and step_yr <= limits['safe_pred']:
        return 'predictive_unwrap'
    return 'resample'


def _short_period_warning(row: dict, limits: dict, settings) -> str:
    """The warning of a row whose step folds short periods (`sampling.short_period_trigger`): which condition fired."""
    step = row['step_yr']
    if row['short_period_trigger'] == sampling.SHORT_PERIOD_HARMONIC:
        why = (
            f"step {step:.4g} yr < {limits['step_short']:.4g} yr = min(P_orb, P_syn) / 2, but a harmonic (m <= "
            f"{settings.mmr_short_period_harmonics}) folds to {row['folded_slowest_yr']:.4g} yr > the period floor "
            f"{limits['floor']:.0f} yr"
        )
    else:
        why = f"step {step:.4g} yr >= {limits['step_short']:.4g} yr = min(P_orb, P_syn) / 2"
    return f"{why}: short-period terms fold into slow ones (flag short_period_aliased, confidence <= medium); the pilot measures how much"


def _row_warnings(row: dict, limits: dict, span_yr: float, settings, libration: bool = True) -> List[str]:
    """The row's warnings; `libration=False` leaves out the step-against-libration one (the run states it once)."""
    out = []
    step = row['step_yr']
    if row['verdict'] != str(StepVerdict.RESOLVED):
        if row['action'] == 'predictive_unwrap':
            unwrap = 'the predictive unwrap recovers the true angle'
        else:  # the step the analysis names (`sampling.resample_step`)
            better = sampling.resample_step(limits['safe_plain'], limits['safe_pred'], math.nan, step, settings)
            unwrap = f'resample the record at {better:g} yr' if better is not None else 'resample the record'
        out.append(
            f"{row['verdict']} at {step:.4g} yr: {row['turns_per_step']:.4f} turns per step (branches {row['branches']}); "
            f"np.unwrap would draw {row['apparent_deg_yr']:.4g} deg/yr instead of {row['rate_deg_yr']:.4g}, "
            f"ghost at a = {row['ghost_axis_au']:.6f} au ({row['nearest_ghost_au']:.2e} au from a0); {unwrap} "
            f"(safe step {row['max_safe_step_pred']:.3g} yr)"
        )
    if row['expected_reason'] == 'undersampled':
        better = row['resample_step_yr']
        out.append(
            f"step {step:.4g} yr >= {limits['step_valid']:.4g} yr: the cutoff P_c = {row['cutoff_pred_yr']:.4g} yr is not longer than "
            f"two steps, no filter keeps a libration of {limits['floor']:.0f} yr ({limits['floor_setting']}); the row will be "
            f"invalid / undersampled" + (f' (resample at {better:g} yr)' if math.isfinite(better) else '')
        )
    elif libration and step > limits['step_libration']:
        out.append(
            f"step {step:.4g} yr > {limits['step_libration']:.3g} yr: a libration of {limits['floor']:.0f} yr "
            f"({limits['floor_setting']}) gets fewer than {settings.mmr_resample_steps_per_cycle:g} points per cycle"
        )
    if row['expected_flags'] == 'short_period_aliased' and row['expected_reason'] != 'undersampled':
        out.append(_short_period_warning(row, limits, settings))
    if settings.mmr_filter_cutoff_yr is None and limits['floor'] > 0:
        if limits['synodic_cutoff'] * settings.filter_min_libration_cutoffs > limits['floor']:
            out.append(
                f"the synodic cutoff {limits['synodic_cutoff']:.4g} yr alone makes a {limits['floor']:.0f}-yr libration filter_too_slow"
            )
    if row['jitter_flag']:
        out.append(
            f"perihelion jitter |p| de_sp/e = {row['jitter']:.2f} rad > {settings.mmr_jitter_max_rad} rad: varpi is undefined at this e"
        )
    if not row['baseline_ok']:
        out.append(f"span {span_yr:.0f} yr < {row['baseline_min_yr']:.0f} yr needed for the shortest libration and the filter edges")
    return out


# ----------------------------------------------------------------------------------------------- run level
def snap_step(limit_yr: float, allowed: Tuple[float, ...]) -> float:
    """The largest of `allowed` <= limit; the limit itself when it is below all of them (or inf -> the largest)."""
    if not math.isfinite(limit_yr):
        return float(max(allowed))
    fitting = [step for step in allowed if step <= limit_yr]
    return float(max(fitting)) if fitting else float(limit_yr)


def _resonance_counts(body) -> Tuple[int, int]:
    """(MMR angles, secular + ZLK angles) of a body."""
    mmr = len(getattr(body, 'mmrs', []) or [])
    other = len(getattr(body, 'secular_resonances', []) or []) + len(getattr(body, 'lidov_kozai_resonances', []) or [])
    return mmr, other


def cost_estimate(
    n_rows: int, span_yr: float, bodies: List, integrator: str, dt_yr: float, bodies_per_worker: int, pilot_s_per_kyr=None
) -> dict:
    """Rows, disk (if every body is saved), RAM per worker and CPU hours from the measured coefficients (SAMPLING §8).

    `pilot_s_per_kyr` ({body: s/kyr}) replaces the integration coefficient by the pilot's measured wall time."""
    disk = ram_body_max = cpu = 0.0
    kyr = abs(span_yr) / 1000.0
    for body in bodies:
        mmr, other = _resonance_counts(body)
        angles = mmr + other
        disk += n_rows * (DATA_COLUMNS_BASE + DATA_COLUMNS_PER_ANGLE * angles) * GZ_BYTES_PER_VALUE
        ram = n_rows * (mmr * RAM_BYTES_PER_ROW_PER_ANGLE['mmr'] + other * RAM_BYTES_PER_ROW_PER_ANGLE['secular'])
        ram_body_max = max(ram_body_max, ram)
        measured = (pilot_s_per_kyr or {}).get(str(body.name))
        if measured is not None and math.isfinite(measured):
            cpu += measured * kyr
        elif str(integrator).lower() == IAS15:
            cpu += IAS15_S_PER_KYR * kyr + IAS15_S_PER_ROW * n_rows
        else:
            dt_units = dt_yr * TWO_PI if math.isfinite(dt_yr) and dt_yr > 0 else 1.0
            cpu += (SABA_S_PER_KYR_PER_TASK / dt_units * kyr + SABA_S_PER_ROW_PER_TASK * n_rows) / SABA_BODIES_PER_TASK
        cpu += PACKAGE_S_PER_ANGLE_ROW * n_rows * angles
    return {
        'rows': int(n_rows),
        'disk_GB': disk / BYTES_PER_GB,
        'ram_per_worker_GB': ram_body_max * max(1, bodies_per_worker) / BYTES_PER_GB,
        'cpu_hours': cpu / SECONDS_PER_HOUR,
    }


def _bodies_per_worker(simulation) -> int:
    n = len(simulation.bodies)
    manager = getattr(simulation, 'batch_manager', None)
    try:
        if manager is not None and manager.should_batch(n):
            return int(manager._determine_batch_size(n))
    except Exception:  # a manager without these (tests): one worker holds every body
        pass
    return max(1, n)


# ----------------------------------------------------------------------------------------------- pilot
def _record_pilot(simulation, body, step_yr: float, span_yr: float, direction: float) -> dict:
    """Integrate one body alone with the run's integrator and dt; elements of the body and of every planet."""
    from resonances.simulation.integration import IntegrationEngine

    engine = simulation.integration_engine
    if engine.sim is None:
        simulation.create_solar_system()
    probe = IntegrationEngine(simulation.config)
    probe.sim = engine.sim.copy()
    probe.massive_bodies = list(engine.massive_bodies)
    elements = body.initial_data
    probe.sim.add(
        m=float(elements.get('mass', 0.0)),
        a=elements['a'],
        e=elements['e'],
        inc=elements['inc'],
        Omega=elements['Omega'],
        omega=elements['omega'],
        M=elements['M'],
        primary=probe.sim.particles[0],
    )
    probe.setup_integrator()
    times = np.arange(0.0, span_yr + 0.5 * step_yr, step_yr)
    n, planets = len(times), [name for name in PLANET_NAMES_ORDER if name in probe.planets_without_sun]
    body_columns = ('a', 'e', 'inc', 'Omega', 'omega', 'M', 'l', 'pomega')
    record = {column: np.empty(n) for column in body_columns}
    planet_record = {name: {column: np.empty(n) for column in ('a', 'e', 'inc', 'Omega', 'pomega', 'l')} for name in planets}
    index = {name: probe.planets_without_sun.index(name) for name in planets}
    particles = probe.sim.particles
    start = time.perf_counter()
    for i, t in enumerate(times):
        probe.sim.integrate(direction * t * TWO_PI)
        orbits = probe.sim.orbits(primary=particles[0])
        orbit = orbits[-1]
        for column in body_columns:
            record[column][i] = getattr(orbit, column)
        for name in planets:
            planet_orbit = orbits[index[name]]
            for column, values in planet_record[name].items():
                values[i] = getattr(planet_orbit, column)
    wall = time.perf_counter() - start
    if direction < 0:  # the analysis wants increasing times: reverse a backward pilot
        record = {column: values[::-1] for column, values in record.items()}
        planet_record = {name: {column: values[::-1] for column, values in data.items()} for name, data in planet_record.items()}
        times = -times[::-1]
    return {'t': times, 'body': record, 'planets': planet_record, 'wall_s': wall}


def _angle(spec: ResonanceSpec, pilot: dict) -> np.ndarray:
    """sigma from the pilot's elements (retrograde: lambda* = M + omega - Omega, varpi* = omega - Omega)."""
    body = pilot['body']
    if spec.retrograde:
        varpi = body['omega'] - body['Omega']
        longitude = body['M'] + varpi
    else:
        varpi, longitude = body['pomega'], body['l']
    sigma = spec.body_coefficient * longitude + spec.body_varpi_coefficient * varpi + spec.body_node_coefficient * body['Omega']
    for index, name in enumerate(spec.planets):
        planet = pilot['planets'][name]
        sigma = sigma + spec.planet_coefficients[index] * planet['l'] + spec.planet_varpi_coefficients[index] * planet['pomega']
        sigma = sigma + spec.planet_node_coefficient(index) * planet['Omega']
    return sigma


def _slope(t: np.ndarray, angle: np.ndarray) -> float:
    return float(np.polyfit(t, np.unwrap(angle), 1)[0])


def pilot_inputs(name: str, pilot: dict, static: BodyInputs, settings, retrograde: bool) -> Tuple[BodyInputs, dict]:
    """Measured BodyInputs: <a>, the a range widened to at least the static class spread (the pilot is short), delta a_sp
    and delta e_sp as the analysis measures them (`sampling.short_period_scatter`, band MMR_JITTER_BAND_YR), the
    precession of varpi (varpi* when retrograde) and Omega, and n from lambda."""
    t, body = pilot['t'], pilot['body']
    a = body['a']
    band, order = settings.mmr_jitter_band_yr, settings.filter_order
    lo, hi = np.percentile(a, (settings.mmr_axis_range_percentile, 100 - settings.mmr_axis_range_percentile))
    mean = float(np.mean(a))
    spread = 0.5 * (static.a_hi_au - static.a_lo_au)
    varpi = body['omega'] - body['Omega'] if retrograde else body['pomega']
    inputs = BodyInputs(
        name=name,
        a_au=mean,
        e=float(np.mean(body['e'])),
        a_lo_au=float(min(lo, mean - spread)),
        a_hi_au=float(max(hi, mean + spread)),
        da_sp_au=sampling.short_period_scatter(t, a, band, order),
        de_sp=sampling.short_period_scatter(t, body['e'], band, order),
        varpi_rate_rad_yr=_slope(t, varpi),
        node_rate_rad_yr=_slope(t, body['Omega']),
        source='pilot',
    )
    measured = {
        'a_mean_au': mean,
        'da_sp_au': inputs.da_sp_au,
        'de_sp': inputs.de_sp,
        'varpi_rate_arcsec_yr': inputs.varpi_rate_rad_yr / ARCSEC,
        'n_from_lambda_deg_yr': math.degrees(_slope(t, body['l'])),
        'source': 'pilot',
    }
    return inputs, measured


@dataclass(frozen=True)
class Leak:
    """What the short-period lines of the pilot do to the filtered record at a coarse step (`leak`).

    `sigma_deg`: largest |filtered decimated - filtered fine| of sigma inside the edge window, offset removed;
    `adot_rel`: eps = rms(a_dot decimated - a_dot fine) / std(a_dot fine) (centred: the force law fits an intercept),
    both the central differences on the coarse grid of a filtered at P_c (the difference is the leak only, not the
    derivative's own sinc); `step_yr`: the cadence evaluated (the pilot's stride x its fine step); `prominence_deg`: the
    TP prominence the fine sigma_f would have, max(MMR_TP_PROMINENCE_FLOOR_DEG, MMR_TP_RIPPLE_FACTOR x r), r the 95th
    percentile of |sigma_f - lowpass(sigma, P_floor)| (`analyse.fast_residual`); `slips`: turns the predictive unwrap of
    the decimated record loses. nan where the pilot cannot hold the step or the cutoff."""

    sigma_deg: float = float('nan')
    adot_rel: float = float('nan')
    prominence_deg: float = float('nan')
    slips: int = 0
    step_yr: float = float('nan')

    def accepts(self, settings) -> Optional[bool]:
        """The step is acceptable: 2 x sigma_deg <= prominence_deg (a wave of half-amplitude L turns by up to 2 L) and
        adot_rel <= sqrt(b / (1 - b)), b = CONFIDENCE_THIN_BAND (R^2 = 1 / (1 + eps^2) >= 1 - b); None unmeasured."""
        if not (math.isfinite(self.sigma_deg) and math.isfinite(self.adot_rel) and math.isfinite(self.prominence_deg)):
            return None
        return bool(2 * self.sigma_deg <= self.prominence_deg and self.adot_rel <= adot_rel_max(settings) and not self.slips)


def adot_rel_max(settings) -> float:
    """sqrt(b / (1 - b)), b = CONFIDENCE_THIN_BAND: the a_dot error that keeps a perfect force law within the thin band."""
    band = settings.confidence_thin_band
    return math.sqrt(band / (1 - band))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float('nan')


def leak(spec: ResonanceSpec, pilot: dict, limits: dict, step_yr: float, settings, planet_n) -> Leak:
    """The short-period lines that fold into the filtered angle and the filtered a_dot at this step (`Leak`)."""
    t = pilot['t']
    fine_step = float(np.median(np.diff(t)))
    stride = int(round(step_yr / fine_step))
    if stride < 2:
        return Leak()
    sigma = np.unwrap(_angle(spec, pilot))
    a = np.asarray(pilot['body']['a'], dtype=float)
    index = np.arange(0, len(t), stride)
    t_dec, a_dec = t[index], a[index]
    decimated_step = float(np.median(np.diff(t_dec)))
    cutoff = preprocess.choose_cutoff(limits['synodic'], decimated_step, settings, limits['floor']).cutoff_yr
    window = edge_window(t_dec, settings.filter_edge_trim_cutoffs * cutoff)
    inside = window_mask(t_dec, window)
    if inside.sum() < 3 or not preprocess.filter_realizable(cutoff, decimated_step):
        return Leak(step_yr=decimated_step)
    order = settings.filter_order
    rates = sampling.rate_model(spec, a_dec, planet_n, limits['slow'])
    unwrapped, _ = sampling.predictive_unwrap(t_dec, np.mod(sigma[index], TWO_PI), rates)
    unwrapped = unwrapped + (sigma[0] - unwrapped[0])
    slips = int(round((unwrapped[-1] - sigma[index[-1]]) / TWO_PI))
    sigma_fine = lowpass(t, sigma, cutoff, order)
    difference = (lowpass(t_dec, unwrapped, cutoff, order) - sigma_fine[index])[inside]
    adot_fine = np.gradient(lowpass(t, a, cutoff, order)[index], t_dec)[inside]
    adot_coarse = np.gradient(lowpass(t_dec, a_dec, cutoff, order), t_dec)[inside]
    spread = _rms(adot_fine - np.mean(adot_fine))
    fast = 0.0
    if limits['floor'] > cutoff:
        fine_inside = window_mask(t, window)
        fast = float(np.percentile(np.abs(sigma_fine - lowpass(t, sigma, limits['floor'], order))[fine_inside], 95))
    prominence = max(settings.mmr_tp_prominence_floor_deg, settings.mmr_tp_ripple_factor * math.degrees(fast))
    return Leak(
        sigma_deg=float(np.degrees(np.max(np.abs(difference - np.median(difference))))),
        adot_rel=_rms(adot_coarse - adot_fine) / spread if spread > 0 else float('nan'),
        prominence_deg=float(prominence),
        slips=slips,
        step_yr=decimated_step,
    )


def _line_name(frequency: float, n_body: float, planet_n, max_coefficient: int, tolerance: float) -> str:
    """The simplest |p n_j - q n| within `tolerance` [cycles/yr] of `frequency`, '' when none."""
    best = None
    for name in PLANET_NAMES_ORDER:
        n_j = planet_n[name] / TWO_PI
        for p in range(1, max_coefficient + 1):
            for q in range(1, max_coefficient + 1):
                if math.gcd(p, q) == 1 and abs(abs(p * n_j - q * n_body) - frequency) < tolerance:
                    if best is None or p + q < best[0]:
                        best = (p + q, f'{p}n_{name}-{q}n')
    return '' if best is None else best[1]


def forced_terms(spec: ResonanceSpec, pilot: dict, cutoff_yr: float, settings, dry, planet_n) -> List[Tuple[float, float, str]]:
    """Lines of the pilot's sigma the filter passes: (period [yr], filtered half-amplitude [deg], name), from the
    amplitude spectrum (Hann window, quadratic trend removed) up to half the pilot, times the filter gain G(P) at P_c,
    with a filtered amplitude >= DRYRUN_FORCED_TERM_MIN_DEG. A named line is a forced near-commensurability wave (1999 UQ: 97 yr, 7n_E - 8n);
    an unnamed one may be the libration itself."""
    t = pilot['t']
    sigma = np.unwrap(_angle(spec, pilot))
    residual = sigma - np.polyval(np.polyfit(t - t[0], sigma, 2), t - t[0])
    window = np.hanning(len(t))
    amplitude = 2 * np.abs(np.fft.rfft(residual * window)) / window.sum()
    frequency = np.fft.rfftfreq(len(t), float(np.median(np.diff(t))))
    span = float(t[-1] - t[0])
    with np.errstate(divide='ignore'):
        period = np.where(frequency > 0, 1.0 / np.where(frequency > 0, frequency, 1.0), np.inf)
    band = period <= 0.5 * span
    passed = np.degrees(amplitude) * np.where(
        band, butterworth_gain(np.where(band, period, cutoff_yr), cutoff_yr, settings.filter_order), 0.0
    )
    peaks, _ = signal.find_peaks(np.concatenate([[0.0], passed, [0.0]]))
    peaks = peaks - 1
    n_body = float(np.polyfit(t, np.unwrap(pilot['body']['l']), 1)[0]) / TWO_PI
    lines = []
    for i in peaks[np.argsort(passed[peaks])[::-1]]:
        if passed[i] < dry.dryrun_forced_term_min_deg:
            break
        lines.append(
            (
                float(period[i]),
                float(passed[i]),
                _line_name(frequency[i], n_body, planet_n, dry.dryrun_forced_term_max_coefficient, 1.0 / span),
            )
        )
    return lines


# ----------------------------------------------------------------------------------------------- report
@dataclass
class DryRunReport:
    """The dry-run's result: `rows` (body x MMR), `bodies`, `run` (step, cost) and the warnings."""

    level: str
    rows: pd.DataFrame
    bodies: pd.DataFrame
    run: dict
    warnings: List[str] = field(default_factory=list)
    simulation: object = field(default=None, repr=False, compare=False)

    def summary(self, echo: bool = True) -> str:
        """A table per row and per body, the run's step and cost, then the warnings (printed when `echo`)."""
        run = self.run
        lines = [f"Dry-run ({self.level}): {run['n_bodies']} bodies, {run['n_mmr_rows']} MMR rows, span {run['span_yr']:.0f} yr"]
        lines.append(
            f"  output step: configured {run['step_configured_yr']:.4g} yr (Nout {run['Nout_configured']}), "
            f"auto {_fmt(run['step_auto_yr'])} yr ({run['binding_rule'] or 'n/a'}), used {run['step_used_yr']:.4g} yr"
        )
        limits = ', '.join(f'{name} {_fmt(value)}' for name, value in run['step_limits_yr'].items())
        lines.append(f'  step limits [yr]: {limits}')
        lines.append(
            f"  integrator: configured {run['integrator_configured']} (dt {run['dt_configured_yr']:.4g} yr), "
            f"recommended {run['integrator_recommended']} (dt {_fmt(run['dt_recommended_yr'])} yr)"
            + ('; mixed: run the ias15 bodies apart' if run['mixed_integrators'] else '')
        )
        lines.append(
            f"  cost at the used step: {run['rows']} rows, {run['disk_GB']:.3g} GB if all saved, "
            f"{run['ram_per_worker_GB']:.3g} GB RAM per worker, {run['cpu_hours']:.3g} CPU h"
        )
        if not self.rows.empty:
            columns = [
                'body',
                'resonance',
                'rate_deg_yr',
                'turns_per_step',
                'verdict',
                'nearest_ghost_au',
                'max_safe_step_pred',
                'cutoff_pred_yr',
                'action',
            ]
            columns += [c for c in ('expected_status', 'expected_flags') if c in self.rows]
            columns += [c for c in ('leak_deg', 'leak_adot_rel', 'pilot_step_ok') if self.level == 'pilot']
            lines.append(self.rows[columns].to_string(index=False, float_format=lambda x: f'{x:.4g}'))
        if not self.bodies.empty:
            columns = ['body', 'class', 'q', 'P_yr', 'crossing', 'integrator_recommended', 'dt_recommended_yr', 'fair_step_yr']
            lines.append(self.bodies[columns].to_string(index=False, float_format=lambda x: f'{x:.4g}'))
        lines += [f'  ! {warning}' for warning in self.warnings]
        text = '\n'.join(lines)
        if echo:
            print(text)
        return text

    def log_warnings(self) -> None:
        """Every warning to the package log (what find/check write before integrating)."""
        for warning in self.warnings:
            logger.warning(f'dry-run: {warning}')

    def apply(self, step: bool = True, integrator: bool = True) -> dict:
        """Set the config of the simulation: the auto output step (output_step, Nout) and the recommended integrator
        and dt. Returns what changed. Without MMR rows 'auto' has no step: Nout is kept, with a warning."""
        config = self.simulation.config
        changes = {}
        if step:
            value = self.run['step_auto_yr']
            if value is None:
                logger.warning("dry-run: OUTPUT_STEP='auto' is defined for runs with MMRs only; the default Nout is kept")
            else:
                from resonances.simulation.config import nout_for_step

                config.output_step = float(value)
                config.Nout = nout_for_step(config.tmax_yrs, float(value))
                changes.update(output_step=config.output_step, Nout=config.Nout)
        if integrator and self.run['integrator_recommended']:
            changes.update(self._apply_integrator(config))
        for name, value in changes.items():
            logger.info(f'dry-run: {name} = {value}')
        return changes

    def _apply_integrator(self, config) -> dict:
        recommended = self.run['integrator_recommended']
        if getattr(config, 'solar_system_moon', False) and recommended != IAS15:
            logger.warning('dry-run: the separate Moon needs ias15; the integrator is kept')
            return {}
        changes = {'integrator': recommended}
        config.integrator = recommended
        if recommended != IAS15 and math.isfinite(self.run['dt_recommended_yr']):
            config.dt = float(self.run['dt_recommended_yr'] * TWO_PI)
            changes['dt'] = config.dt
        return changes

    def to_csv(self, path: str) -> dict:
        """dryrun_rows.csv, dryrun_bodies.csv and dryrun_run.json in the folder `path`; returns their paths."""
        os.makedirs(path, exist_ok=True)
        files = {name: os.path.join(path, name) for name in ('dryrun_rows.csv', 'dryrun_bodies.csv', 'dryrun_run.json')}
        self.rows.to_csv(files['dryrun_rows.csv'], index=False)
        self.bodies.to_csv(files['dryrun_bodies.csv'], index=False)
        with open(files['dryrun_run.json'], 'w') as handle:
            json.dump({**self.run, 'level': self.level, 'warnings': self.warnings}, handle, indent=1, default=_json_default)
        return files


def _fmt(value) -> str:
    return 'n/a' if value is None or (isinstance(value, float) and not math.isfinite(value)) else f'{value:.4g}'


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


# ----------------------------------------------------------------------------------------------- driver
def _valid_elements(elements) -> bool:
    try:
        return float(elements['a']) > 0 and 0 <= float(elements['e']) < 1
    except (KeyError, TypeError, ValueError):
        return False


def _body_rows(body, inputs: BodyInputs, step_yr: float, span_yr: float, settings, dry, tables) -> List[Tuple[ResonanceSpec, dict, dict]]:
    """(spec, limits, row) for every MMR of a body."""
    out = []
    for resonance in getattr(body, 'mmrs', []) or []:
        spec = ResonanceSpec.from_resonance(resonance)
        limits = _row_limits(spec, inputs, settings, tables)
        out.append((spec, limits, _row_at_step(spec, inputs, limits, step_yr, span_yr, settings, dry, tables)))
    return out


def _step_plan(simulation, all_rows, bodies_rows: List[dict], settings, dry) -> dict:
    """The 'auto' step and the step the run will use."""
    config = simulation.config
    limits = {
        'libration': min((limits['step_libration'] for _, limits, _ in all_rows), default=math.inf),
        # exclusive bounds: a step equal to min(P_orb, P_syn) / 2 is already flagged, one equal to P_c / 2 undersampled
        'short_period': min((math.nextafter(limits['step_short'], 0.0) for _, limits, _ in all_rows), default=math.inf),
        'filter': min((limits['step_filter'] for _, limits, _ in all_rows), default=math.inf),
        'trend': min((limits['safe_trend'] for _, limits, _ in all_rows), default=math.inf),
        'fair': min((b['fair_step_yr'] for b in bodies_rows), default=math.inf) if dry.dryrun_fair == 'required' else math.inf,
        'user': dry.dryrun_max_step_yr if dry.dryrun_max_step_yr is not None else math.inf,
    }
    configured = float(config.output_step_yr)
    if all_rows:
        binding = min(limits, key=limits.get)
        snapped = snap_step(limits[binding], dry.dryrun_auto_steps_yr)
        auto = fold_free_step(all_rows, snapped, dry.dryrun_auto_steps_yr, settings)
        if auto is None:  # every step down to the smallest folds a harmonic: the snapped one, flagged
            auto = snapped
        elif auto != snapped:
            binding = SHORT_PERIOD_HARMONIC_LIMIT
        fold_free = fold_free_step(all_rows, min(configured, auto), dry.dryrun_auto_steps_yr, settings)
    else:
        binding, auto, fold_free = '', None, None
    used = auto if (getattr(config, 'output_step', None) == 'auto' and auto is not None) else configured
    return {'limits': limits, 'binding': binding, 'auto': auto, 'configured': configured, 'used': used, 'fold_free': fold_free}


def folds_short_periods(limits: dict, step_yr: float, settings) -> str:
    """The analysis' short-period predicate (`sampling.short_period_trigger`) of a row with these limits at a step."""
    folded = sampling.folded_slowest_yr(limits['orbital_period'], limits['synodic_all'], step_yr, settings.mmr_short_period_harmonics)
    return sampling.short_period_trigger(step_yr, limits['step_short'], folded, limits['floor'])


def fold_free_step(all_rows, upper_yr: float, allowed: Tuple[float, ...], settings) -> Optional[float]:
    """The first step, walking down from `upper_yr` through the DRYRUN_AUTO_STEPS_YR below it, at which no row's step
    folds short periods (`folds_short_periods` is '' for every row): the step limits are upper bounds, but a harmonic
    of P_orb or P_syn folds slower than the period floor only near m dt / P = j, so a smaller step can be free of it
    (P_orb 10.001 yr: 5 yr folds the second harmonic to 50 kyr, 2.5 yr folds nothing). None when none is free."""
    candidates = [upper_yr] + sorted((step for step in allowed if step < upper_yr), reverse=True)
    for step in candidates:
        if all(folds_short_periods(limits, step, settings) == '' for _, limits, _ in all_rows):
            return float(step)
    return None


def _integrator_plan(bodies_rows: List[dict], config) -> dict:
    recommended = [b['integrator_recommended'] for b in bodies_rows]
    if not recommended:
        return {'integrator': '', 'dt_yr': float('nan'), 'mixed': False}
    if IAS15 in recommended:
        return {'integrator': IAS15, 'dt_yr': float('nan'), 'mixed': len(set(recommended)) > 1}
    return {'integrator': SABA, 'dt_yr': float(min(b['dt_recommended_yr'] for b in bodies_rows)), 'mixed': False}


def _body_warnings(body_row: dict, config) -> List[str]:
    """The run's integrator against the recommendation."""
    from resonances.simulation.integration import is_fixed_step

    out = []
    name = body_row['body']
    crossing = [planet for planet in body_row['crossing'].split(',') if planet]
    if is_fixed_step(config.integrator) and body_row['integrator_recommended'] == IAS15:
        moids = [
            f"{planet} {body_row[f'moid_{planet}']:.3g} au = {body_row[f'moid_{planet}'] / hill_radius_au(planet):.3g} R_H"
            for planet in crossing
            if f'moid_{planet}' in body_row
        ]
        detail = f" (MOID now: {', '.join(moids)})" if moids else ''
        out.append(f"{name}: {config.integrator} with a fixed step, but {body_row['why']}{detail} -> ias15")
    if crossing and body_row['source'] == 'static':
        out.append(
            f"{name}: a crosser's a jumps at encounters; the alias verdicts hold for a0 +- {body_row['a_spread_au']:.3g} au (DRYRUN_DA_STATIC) only"
        )
    if is_fixed_step(config.integrator) and body_row['integrator_recommended'] != IAS15:
        dt_yr = float(config.dt) / TWO_PI
        if dt_yr > body_row['dt_recommended_yr'] * (1 + 1e-9):
            out.append(
                f"{name}: dt = {dt_yr:.4g} yr gives {body_row['steps_per_orbit']:.1f} steps per orbit and "
                f"{body_row['steps_per_peri']:.1f} per pericentre passage; recommended dt <= {body_row['dt_recommended_yr']:.4g} yr"
            )
    return out


def _moids(simulation, elements: dict) -> dict:
    """MOID to each planet from the planets' osculating elements at the epoch (pilot level: the solar system is loaded)."""
    sim = getattr(simulation.integration_engine, 'sim', None)
    if sim is None:
        return {}
    names = simulation.integration_engine.planets_without_sun
    orbits = sim.orbits(primary=sim.particles[0])
    body = {key: float(elements[key]) for key in ('a', 'e', 'inc', 'Omega', 'omega')}
    out = {}
    for index, name in enumerate(names):
        if name not in const.SOLAR_SYSTEM:
            continue
        orbit = orbits[index]
        out[f'moid_{name}'] = moid_au(body, {'a': orbit.a, 'e': orbit.e, 'inc': orbit.inc, 'Omega': orbit.Omega, 'omega': orbit.omega})
    return out


def dry_run(simulation, level: str = 'static') -> DryRunReport:
    """The dry-run of `simulation` (its bodies and config as they are now); see the module docstring."""
    if level not in LEVELS:
        raise ValueError(f"dry_run level is one of {', '.join(LEVELS)}, got {level!r}")
    config = simulation.config
    settings, dry = config.analysis_settings, config.dry_run_settings
    span = abs(config.tmax_yrs)
    tables = _planet_tables()
    warnings: List[str] = []
    prepared = []  # (body, inputs, body_row)
    for body in simulation.bodies:
        if not _valid_elements(body.initial_data):
            warnings.append(f'{body.name}: elements a, e not usable for an elliptic orbit; skipped')
            continue
        inputs, body_row = static_inputs(str(body.name), body.initial_data, dry)
        prepared.append((body, inputs, body_row))

    plan = _plan(simulation, prepared, span, settings, dry, tables)
    pilot_speed = {}
    if level == 'pilot':
        prepared, pilot_rows = _run_pilots(simulation, prepared, plan, span, settings, dry, tables, pilot_speed)
        plan = _plan(simulation, prepared, span, settings, dry, tables)
        _merge_pilot_rows(plan['rows'], pilot_rows, settings, plan['steps']['fold_free'])

    from resonances.simulation.integration import is_fixed_step

    run_dt_yr = float(config.dt) / TWO_PI
    fixed_step = is_fixed_step(config.integrator)
    for _, _, body_row in prepared:
        recommendation = recommend_integrator(
            body_row['a'], body_row['e'], tuple(filter(None, body_row['crossing'].split(','))), run_dt_yr, dry
        )
        body_row.update(
            integrator_recommended=recommendation['integrator'],
            dt_recommended_yr=recommendation['dt_yr'],
            why=recommendation['why'],
            steps_per_orbit=body_row['P_yr'] / run_dt_yr if fixed_step else float('nan'),
            steps_per_peri=body_row['tau_peri_yr'] / run_dt_yr if fixed_step else float('nan'),
        )
        warnings += _body_warnings(body_row, config)
    rows = [{key: value for key, value in row.items() if key != 'pilot_notes'} for _, _, row in plan['rows']]
    for _, limits, row in plan["rows"]:  # the step-against-libration warning is stated once, for the run
        notes = _row_warnings(row, limits, span, settings, libration=False) + row.get('pilot_notes', [])
        warnings += [f"{row['body']} {row['resonance']}: {note}" for note in notes]
    bodies_rows = [body_row for _, _, body_row in prepared]
    integrator = _integrator_plan(bodies_rows, config)
    steps = plan['steps']
    if steps['auto'] is None and getattr(config, 'output_step', None) == 'auto':
        warnings.append("OUTPUT_STEP='auto' is defined for runs with MMRs only: the default Nout is kept")
    from resonances.simulation.config import nout_for_step

    n_rows = nout_for_step(span, steps['used']) if math.isfinite(steps['used']) else int(config.Nout)
    cost = cost_estimate(
        n_rows,
        span,
        simulation.bodies,
        integrator['integrator'] or config.integrator,
        integrator['dt_yr'],
        _bodies_per_worker(simulation),
        pilot_speed,
    )
    per_body = [min([r['step_libration_yr'] for r in rows if r['body'] == b['body']] or [math.inf]) for b in bodies_rows]
    finite = [value for value in per_body if math.isfinite(value)]
    run = {
        'level': level,
        'n_bodies': len(bodies_rows),
        'n_mmr_rows': len(rows),
        'n_rows_not_resolved': sum(1 for row in rows if row['verdict'] != str(StepVerdict.RESOLVED)),
        'span_yr': span,
        'output_step': getattr(config, 'output_step', None),
        'step_configured_yr': steps['configured'],
        'Nout_configured': int(config.Nout),
        'step_limits_yr': steps['limits'],
        'step_auto_yr': steps['auto'],
        'binding_rule': steps['binding'],
        'step_fold_free_yr': steps['fold_free'],
        'Nout_auto': nout_for_step(span, steps['auto']) if steps['auto'] else None,
        'step_used_yr': steps['used'],
        'Nout': n_rows,
        'integrator_configured': str(config.integrator),
        'dt_configured_yr': run_dt_yr,
        'integrator_recommended': integrator['integrator'],
        'dt_recommended_yr': integrator['dt_yr'],
        'mixed_integrators': integrator['mixed'],
        'step_spread': (max(finite) / min(finite)) if finite else float('nan'),
        'unwrap': settings.mmr_unwrap,
        **cost,
    }
    exceeded = _exceeded_limits(steps, rows) if getattr(config, 'output_step', None) != 'auto' and steps['auto'] is not None else {}
    if exceeded:
        above = ', '.join(f'the {name} limit {value:.4g} yr' for name, value in exceeded.items())
        warnings.append(f"output step {steps['configured']:.4g} yr is above {above} (OUTPUT_STEP=auto would take {steps['auto']:.4g} yr)")
    return DryRunReport(level, pd.DataFrame(rows), pd.DataFrame(bodies_rows), run, warnings, simulation)


def _exceeded_limits(steps: dict, rows: List[dict]) -> dict:
    """{limit: value} of the step limits the configured step is above. The short-period limit is left out when the
    pilot measured every row above it and accepted the step (`Leak.accepts`: its leaks are small). A row below that
    limit whose step folds a harmonic (trigger 'harmonic') and whose leak the pilot rejected adds the limit
    'short_period_harmonic': its value is the step to use instead (`fold_free_step` below the configured one)."""
    exceeded = {name: value for name, value in steps['limits'].items() if steps['configured'] > value}
    flagged = [row for row in rows if row.get('expected_flags') == 'short_period_aliased']
    if 'short_period' in exceeded and flagged and all(row.get('pilot_step_ok') is True for row in flagged):
        del exceeded['short_period']
    rejected = [
        row
        for row in flagged
        if row.get('short_period_trigger') == sampling.SHORT_PERIOD_HARMONIC
        and row.get('expected_reason') != 'undersampled'
        and row.get('pilot_step_ok') is False
    ]
    if rejected and steps.get('fold_free') is not None and steps['configured'] > steps['fold_free']:
        exceeded[SHORT_PERIOD_HARMONIC_LIMIT] = steps['fold_free']
    return exceeded


def _plan(simulation, prepared, span: float, settings, dry, tables) -> dict:
    """Rows at the configured step (to know the limits), the step plan, then the rows at the used step."""
    step = float(simulation.config.output_step_yr)
    first = [entry for body, inputs, _ in prepared for entry in _body_rows(body, inputs, step, span, settings, dry, tables)]
    steps = _step_plan(simulation, first, [b for _, _, b in prepared], settings, dry)
    if steps['used'] != step:
        first = [entry for body, inputs, _ in prepared for entry in _body_rows(body, inputs, steps['used'], span, settings, dry, tables)]
    return {'rows': first, 'steps': steps}


def _run_pilots(simulation, prepared, plan, span: float, settings, dry, tables, pilot_speed: dict):
    """One pilot per body; returns the measured inputs and {(body, resonance): pilot columns}."""
    step = plan['steps']['used']
    direction = -1.0 if simulation.config.tmax < 0 else 1.0
    fine_step = pilot_step(step, dry.dryrun_pilot_step_yr)
    measured_prepared, pilot_rows = [], {}
    for body, inputs, body_row in prepared:
        cutoffs = [row['cutoff_pred_yr'] for _, _, row in plan['rows'] if row['body'] == inputs.name]
        length = min(dry.dryrun_pilot_max_yr, max(dry.dryrun_pilot_min_yr, dry.dryrun_pilot_cutoffs * max(cutoffs, default=0.0)))
        pilot = _record_pilot(simulation, body, fine_step, length, direction)
        body_row.update(_moids(simulation, body.initial_data))  # the pilot loaded the planets; ~40 ms per body
        retrograde = float(body.initial_data.get('inc', 0.0)) > math.pi / 2
        measured, columns = pilot_inputs(inputs.name, pilot, inputs, settings, retrograde)
        body_row.update(columns, pilot_years=length, pilot_step_yr=fine_step, pilot_wall_s=pilot['wall_s'])
        body_row['pilot_s_per_kyr'] = pilot['wall_s'] / (length / 1000.0)
        pilot_speed[inputs.name] = body_row['pilot_s_per_kyr']
        for resonance in getattr(body, 'mmrs', []) or []:
            spec = ResonanceSpec.from_resonance(resonance)
            limits = _row_limits(spec, measured, settings, tables)
            cutoff = preprocess.choose_cutoff(limits['synodic'], step, settings, limits['floor']).cutoff_yr
            measured_leak = leak(spec, pilot, limits, step, settings, tables[0])
            lines = forced_terms(spec, pilot, cutoff, settings, dry, tables[0])
            pilot_rows[(inputs.name, spec.key)] = {
                'leak_deg': measured_leak.sigma_deg,
                'leak_adot_rel': measured_leak.adot_rel,
                'leak_tp_prominence_deg': measured_leak.prominence_deg,
                'leak_slips': measured_leak.slips,
                'pilot_step_ok': measured_leak.accepts(settings),
                'leak_step_yr': measured_leak.step_yr,
                'forced_terms': '; '.join(f'{p:.4g} yr {amp:.2f} deg' + (f' ({label})' if label else '') for p, amp, label in lines),
                'forced_lines': lines,
            }
        del pilot
        measured_prepared.append((body, measured, body_row))
    return measured_prepared, pilot_rows


def pilot_step(step_yr: float, pilot_step_yr: float) -> float:
    """The pilot's fine step: the largest step <= DRYRUN_PILOT_STEP_YR that divides the planned output step, so the
    decimated pilot has exactly that cadence (15.9185 yr: 319 x 0.0499 yr, not 318 x 0.05 = 15.9 yr)."""
    if not (math.isfinite(step_yr) and step_yr > pilot_step_yr):
        return float(pilot_step_yr)
    return float(step_yr / math.ceil(step_yr / pilot_step_yr))


def _resample_where(row: dict, limits: dict, fold_free_yr: Optional[float]) -> str:
    """The step a rejected row should be resampled at: below the Nyquist step, or the run's fold-free step."""
    if row['short_period_trigger'] != sampling.SHORT_PERIOD_HARMONIC:
        return f"<= {limits['step_short']:.4g} yr"
    if fold_free_yr is not None:
        return f"{fold_free_yr:g} yr (no harmonic of the run's rows folds slower than its period floor there)"
    return 'a step at which no harmonic folds slower than the period floor (none of DRYRUN_AUTO_STEPS_YR is)'


def _merge_pilot_rows(rows, pilot_rows: dict, settings, fold_free_yr: Optional[float] = None) -> None:
    """The pilot's columns into the rows; a lost turn, a named forced wave as slow as a libration, and the pilot's
    verdict on a step that folds short periods (`Leak.accepts`) become warnings; a rejected harmonic row names
    `fold_free_yr` (the plan's `fold_free_step`). A leak measured at another cadence than the row's (the plan changed
    after the pilot) decides nothing: pilot_step_ok is left unmeasured."""
    for _, limits, row in rows:
        extra = pilot_rows.get((row['body'], row['resonance']))
        if not extra:
            continue
        if not math.isclose(extra['leak_step_yr'], row['step_yr']):
            extra = {**extra, 'pilot_step_ok': None}
        lines = extra['forced_lines']
        row.update({key: value for key, value in extra.items() if key != 'forced_lines'})
        notes = []
        if extra['leak_slips']:
            notes.append(f"the predictive unwrap of the decimated pilot lost {extra['leak_slips']} turns")
        if row['expected_flags'] == 'short_period_aliased' and row['expected_reason'] != 'undersampled':
            measured = (
                f"sigma leak {extra['leak_deg']:.3g} deg (2 x <= {extra['leak_tp_prominence_deg']:.3g} deg), a_dot error "
                f"{extra['leak_adot_rel']:.3g} (<= {adot_rel_max(settings):.2f})"
            )
            if extra['pilot_step_ok'] is None:
                notes.append(f"the pilot measured the leak at {extra['leak_step_yr']:.4g} yr, not at the planned {row['step_yr']:.4g} yr")
            elif extra['pilot_step_ok']:
                notes.append(f"the pilot accepts the step {row['step_yr']:.4g} yr: {measured}")
            else:
                where = _resample_where(row, limits, fold_free_yr)
                notes.append(f"the pilot does not accept the step {row['step_yr']:.4g} yr ({measured}): resample at {where}")
        slow = [(period, amplitude, name) for period, amplitude, name in lines if name and period >= limits['floor'] > 0]
        for period, amplitude, name in slow:
            notes.append(
                f"a forced wave {name} of {period:.4g} yr ({amplitude:.2f} deg after the filter) is as slow as a libration "
                f"(>= {limits['floor']:.0f} yr): the analysis cannot tell it from one by its period"
            )
        row['pilot_notes'] = notes
        row['warnings'] = '; '.join(([row['warnings']] if row['warnings'] else []) + notes)


def prepare_run(simulation) -> Optional[DryRunReport]:
    """Before integrating (find/check, `Simulation.run`): the static dry-run's warnings to the log, and the auto step
    applied when OUTPUT_STEP='auto'. Once per simulation; batch workers (which get the parent's step) skip it."""
    config = simulation.config
    if getattr(simulation, '_dry_run_prepared', False) is True or getattr(config, '_skip_path_verification', False) is True:
        return None
    if not simulation.bodies:
        return None
    try:
        report = dry_run(simulation, 'static')
    except Exception as error:  # advisory: a failed check never stops the run, but says so
        logger.warning(f'dry-run: the static check failed ({type(error).__name__}: {error}); the run goes on unchecked')
        return None
    report.log_warnings()
    if getattr(config, 'output_step', None) == 'auto':
        report.apply(step=True, integrator=False)
    simulation._dry_run_prepared = True
    return report
