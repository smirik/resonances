"""The owner's force from the averaged disturbing function: the potential route of the owner search (W20, R1 §2, R5 §1.4).

Lagrange's equation for a (Murray & Dermott 1999, eq. 6.145) at first order in the planet's mass, with the disturbing
function averaged over the fast angle at fixed phi_o (`averaged_potential.LineAverage`, as Gallardo 2006a, Icarus 184,
p. 30, eq. 6, here with the planet's actual eccentric, inclined orbit):

    a_dot = (2 k_o / (n a)) G m_P d<D>/dphi_o (phi_o(t)),   G M_sun = 4 pi^2 au^3 / yr^2,

k_o the body's coefficient of the candidate phi_o = k_P lambda_P + k_o lambda + p varpi (the sign as in the docstring of
`averaged_potential`). In each chosen window of the candidate's regime the body's and the planet's elements are the
window medians, <D>(phi) is evaluated on the full periodic grid of L nodes per turn and differentiated by FFT; L is
doubled until the derivative with L / 2 nodes per turn agrees within mmr_strength_quadrature_tol (else not_converged).
The prediction along the recorded phi_o is band-passed exactly as a (the owner frame's [P/f, f P]) and fitted to the
band-passed a_dot on the window's fit-grid samples:

    s_w = (f . y) / (f . f),   R^2_w = 1 - |y - s_w f|^2 / |y - <y>|^2   (means removed).

One law for a libration and a forced wave of the owner; an owner whose equilibrium moves with e, i and varpi (Gallardo
2006a, pp. 30-31) is followed, because every window has its own elements. s = 1 means the owner's own term, with its
real coefficient, supplies the observed force; the model error of the frozen first-order average is about 0.1 (Codex
19.10: scales 0.90-1.12 on GK3, HW7, 78085). For the columns the band-passed prediction is also projected on sin/cos
phi_o (K_theory, matched to the fitted K: both through the same filter, whose gain is ~0.88 at the band centre for
order 2) and the unfiltered prediction on the phi grid gives K_direct and phi0_direct in the owner fit's convention
a_dot = K sin(phi_o - phi0).

Applicable only to a two-body, prograde candidate that is not 1:1, with the planet's saved elements and lambda and a
known mass (a catalogue line has no elements). Close approach: along the phi_o values the window visits the body must
stay mmr_strength_min_distance_hill Hill radii from the planet (an encounter is not a secular effect). Pure: arrays in,
no Body/Simulation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.mmr import averaged_potential as averaging
from resonances.resonance.analysis.mmr import strength
from resonances.resonance.analysis.mmr.models import Context, MMRSeries, PotentialFit, PotentialWindow, ResonanceSpec
from resonances.resonance.analysis.mmr.response import sine_coefficients
from resonances.resonance.analysis.settings import AnalysisSettings

NAN = float('nan')
TWO_PI = 2 * math.pi

REASON_OFF = 'off'
REASON_THREE_BODY = 'three_body_owner'
REASON_COORBITAL = 'coorbital_owner'
REASON_RETROGRADE = 'retrograde'
REASON_CATALOGUE = 'catalogue'  # the planet's elements or lambda not saved
REASON_NO_MASS = 'no_planet_mass'
REASON_COST = 'potential_cost'
REASON_NOT_CONVERGED = 'not_converged'
REASON_CLOSE_APPROACH = 'close_approach'
REASON_NO_WINDOWS = 'no_windows'

PLANET_ELEMENTS = ('a', 'e', 'inc', 'Omega', 'varpi', 'mean_longitude')
# reasons that say the route cannot measure the candidate at all (owner_unresolved reads them)
NOT_APPLICABLE = frozenset({REASON_OFF, REASON_THREE_BODY, REASON_RETROGRADE, REASON_COORBITAL, REASON_CATALOGUE, REASON_NO_MASS})


def not_applicable(spec: ResonanceSpec, series: MMRSeries, context: Context, settings: AnalysisSettings) -> str:
    """'' when the potential route applies to the candidate, else the reason (module docstring)."""
    if settings.mmr_owner_potential == 'off':
        return REASON_OFF
    if len(spec.planets) != 1:
        return REASON_THREE_BODY
    if spec.retrograde:
        return REASON_RETROGRADE
    if not strength.direct_applicable(spec):
        return REASON_COORBITAL
    planet = series.planets.get(spec.planets[0])
    if planet is None or any(getattr(planet, name, None) is None for name in PLANET_ELEMENTS):
        return REASON_CATALOGUE
    mass = context.planet_mass.get(spec.planets[0], NAN)
    if not (np.isfinite(mass) and mass > 0):
        return REASON_NO_MASS
    return ''


def periodic_derivative(values: np.ndarray) -> np.ndarray:
    """d/dphi of samples on the uniform periodic grid phi_m = 2 pi m / L (spectral; the Nyquist term dropped)."""
    values = np.asarray(values, float)
    size = len(values)
    coefficients = np.fft.rfft(values)
    coefficients *= 1j * np.arange(len(coefficients))
    if size % 2 == 0:
        coefficients[-1] = 0.0
    return np.fft.irfft(coefficients, size)


def first_harmonic(values: np.ndarray) -> float:
    """A1 = 2 |<D exp(-i phi)>| on the periodic grid: <D> = D0 + A1 cos(phi - c) + ..."""
    phi = TWO_PI * np.arange(len(values)) / len(values)
    return float(2 * abs(np.mean(np.asarray(values, float) * np.exp(-1j * phi))))


def profile_kernels(body_coefficient: int, nodes_per_turn: int) -> float:
    """Kernels 1/|r - r_P| of one profile <D>(phi) on L nodes per turn: L values of phi, each the mean over |k| L nodes
    (`averaged_potential.LineAverage`)."""
    return float(abs(body_coefficient)) * float(nodes_per_turn) ** 2


def derivative_profile(
    body: averaging.Ellipse,
    planet: averaging.Ellipse,
    body_coefficient: int,
    planet_coefficient: int,
    phase_rad: float,
    settings: AnalysisSettings,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int, str]:
    """(<D>, d<D>/dphi, closest approach [au]) on the grid of L nodes per turn, L, reason: L starts at
    mmr_strength_nodes_per_turn and doubles up to mmr_strength_max_nodes_per_turn until the derivative with L / 2 agrees
    within mmr_strength_quadrature_tol of its largest value; potential_cost when the profile on L nodes, |k_o| L^2
    kernels (L values of phi, |k_o| L nodes each), would exceed mmr_owner_potential_max_kernels."""
    nodes = int(settings.mmr_strength_nodes_per_turn)
    while True:
        if profile_kernels(body_coefficient, nodes) > settings.mmr_owner_potential_max_kernels:
            return None, None, None, nodes, REASON_COST
        line = averaging.LineAverage(body, planet, body_coefficient, planet_coefficient, phase_rad, nodes)
        full, half, closest = line.evaluate(np.arange(nodes))
        slope_full, slope_half = periodic_derivative(full), periodic_derivative(half)
        scale = float(np.max(np.abs(slope_full)))
        if np.all(np.isfinite(slope_full)) and np.max(np.abs(slope_full - slope_half)) <= settings.mmr_strength_quadrature_tol * scale:
            return full, slope_full, closest, nodes, ''
        if 2 * nodes > settings.mmr_strength_max_nodes_per_turn:
            return None, None, None, nodes, REASON_NOT_CONVERGED
        nodes *= 2


def chosen_starts(starts: Sequence[float], count: int) -> List[float]:
    """At most `count` of the window starts, evenly spread (first and last included)."""
    ordered = sorted(float(s) for s in starts)
    if len(ordered) <= count:
        return ordered
    index = np.unique(np.round(np.linspace(0, len(ordered) - 1, count)).astype(int))
    return [ordered[i] for i in index]


def sine_projection(angle: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    """(K, phi0) of values = K sin(angle - phi0) + C by least squares (nan below three samples)."""
    if len(angle) < 3:
        return NAN, NAN
    a_sin, b_cos, _, _ = sine_coefficients(angle, values)
    return float(np.hypot(a_sin, b_cos)), float(np.arctan2(-b_cos, a_sin))


def profile_projection(grid: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
    """(K, phi0) of a periodic profile values(phi) = K sin(phi - phi0) + ... (its first harmonic on the uniform grid)."""
    a_sin = 2 * float(np.mean(values * np.sin(grid)))
    b_cos = 2 * float(np.mean(values * np.cos(grid)))
    return float(np.hypot(a_sin, b_cos)), float(np.arctan2(-b_cos, a_sin))


def scale_fit(prediction: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """(s, R^2) of target ~ s prediction with the means removed (nan when the prediction is flat)."""
    f = np.asarray(prediction, float) - np.mean(prediction)
    y = np.asarray(target, float) - np.mean(target)
    norm, total = float(np.dot(f, f)), float(np.dot(y, y))
    if not (norm > 0 and total > 0):
        return NAN, NAN
    s = float(np.dot(f, y) / norm)
    return s, 1.0 - float(np.sum((y - s * f) ** 2)) / total


def owner_potential(
    spec: ResonanceSpec,
    starts: Sequence[float],
    frame,
    phi_wrapped: np.ndarray,
    phi_fit: np.ndarray,
    series: MMRSeries,
    context: Context,
    settings: AnalysisSettings,
) -> PotentialFit:
    """The potential route of a candidate on the windows `starts` of one regime (module docstring).

    `frame`: the owner frame the candidate was measured on (`owner.OwnerFrame`: t_yr, fit_mask, window_yr, period_yr,
    adot_band); `phi_wrapped`: phi_o mod 2 pi on the full grid (the recorded angle); `phi_fit`: phi_o low-passed at P_c on
    the frame's fit grid (the owner fit's angle, for K_fit and phi0_fit)."""
    reason = not_applicable(spec, series, context, settings)
    if reason:
        return PotentialFit(reason)
    if not len(starts):
        return PotentialFit(REASON_NO_WINDOWS)
    name = spec.planets[0]
    mu = float(context.planet_mass[name])
    t = np.asarray(frame.t_yr, float)
    fit_index = np.flatnonzero(frame.fit_mask)
    t_fit = t[fit_index]
    factor, order = settings.mmr_owner_band_factor, settings.filter_order
    pad = settings.filter_edge_trim_cutoffs * factor * frame.period_yr
    k, k_p = spec.body_coefficient, spec.planet_coefficients[0]
    windows: List[PotentialWindow] = []
    for start in chosen_starts(starts, settings.mmr_owner_potential_snapshots):
        end = start + frame.window_yr
        part = slice(int(np.searchsorted(t, start, 'left')), int(np.searchsorted(t, end, 'right')))
        body, body_phase = strength.body_ellipse(series, part, spec)
        planet, varpi_p, node_p = strength.saved_planet(series, name, part)
        phase = body_phase + strength.planet_phase(spec, varpi_p, node_p)
        values, slope, closest, nodes, why = derivative_profile(body, planet, k, k_p, phase, settings)
        if why:
            return PotentialFit(why, tuple(windows), name)
        grid = TWO_PI * np.arange(nodes) / nodes
        visited = np.unique(np.round(np.mod(phi_wrapped[part], TWO_PI) / (TWO_PI / nodes)).astype(int) % nodes)
        distance_hill = float(np.min(closest[visited])) / strength.hill_radius(planet.a, mu)
        if distance_hill < settings.mmr_strength_min_distance_hill:
            return PotentialFit(REASON_CLOSE_APPROACH, tuple(windows), name)
        n = math.sqrt(strength.GM_SUN / body.a**3)
        coefficient = 2 * k / (n * body.a) * strength.GM_SUN * mu
        lo, hi = int(np.searchsorted(t, start - pad, 'left')), int(np.searchsorted(t, end + pad, 'right'))
        prediction = coefficient * np.interp(np.mod(phi_wrapped[lo:hi], TWO_PI), grid, slope, period=TWO_PI)
        predicted_band = filtering.band_pass(t[lo:hi], prediction, frame.period_yr, factor, order)
        first, stop = np.searchsorted(t_fit, start, 'left'), np.searchsorted(t_fit, end, 'right')
        inside = fit_index[first:stop]
        if len(inside) < 3:
            continue
        f, y, angle = predicted_band[inside - lo], np.asarray(frame.adot_band, float)[inside], np.asarray(phi_fit)[first:stop]
        scale, r2 = scale_fit(f, y)
        k_theory, phi0_theory = sine_projection(angle, f)
        k_fit, phi0_fit = sine_projection(angle, y)
        mu_eff = mu * body.a * first_harmonic(values)
        k_profile, phi0_profile = profile_projection(grid, coefficient * slope)
        windows.append(
            PotentialWindow(
                t_start_yr=float(start),
                t_end_yr=float(end),
                scale=scale,
                r2=r2,
                k_theory=k_theory,
                phi0_theory=phi0_theory,
                k_fit=k_fit,
                phi0_fit=phi0_fit,
                k_profile=k_profile,
                phi0_profile=phi0_profile,
                a_au=float(body.a),
                e=float(body.orbit.e),
                min_distance_hill=distance_hill,
                nodes_per_turn=int(nodes),
                mu_eff=float(mu_eff),
                omega0=float(abs(k) * n * math.sqrt(3 * mu_eff)),
            )
        )
    return PotentialFit('' if windows else REASON_NO_WINDOWS, tuple(windows), name)
