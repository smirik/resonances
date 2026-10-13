"""The planets' mode longitudes measured in the integration itself: g_j*t + phi_j(t).

The linear law g_j*(t + epoch) + phase_j (`PLANETARY_FREQUENCIES`, `PLANETARY_PHASES`) holds for the
outer planets over any baseline; the inner modes wander (Laskar 1990). In the package's own model g4
differs by 0.014"/yr between 0-10 and 10-20 Myr and the phase of s4 by 51 deg between the two windows
(experiments/2026-10-planetary-frequencies): over tens of Myr the linear law is wrong by as much as a
libration amplitude. Here the phase phi_j(t) of mode j is fitted in sliding windows of planet j's own
vector — e*exp(i*varpi) for g_j, sin(i/2)*exp(i*Omega) for s_j — by least squares on all eight modes
of that kind (s5 = 0 is the invariable plane, the constant of every inclination vector). The window
must resolve the closest pair of lines, g3/g4 (0.54"/yr, a 2.4 Myr beat): `DEFAULT_WINDOW` is 5 Myr.
The longitude keeps g_j as its carrier, so any change of the mode's frequency lands in phi_j(t).
"""

from typing import Dict, Iterable

import numpy as np

from resonances.data.const import PLANETARY_FREQUENCIES, SOLAR_SYSTEM

DEFAULT_WINDOW = 5.0e6  # years
MAX_SAMPLES_PER_FIT = 4000  # a window is decimated to this many samples; the lines are slow
RAD_PER_ARCSEC = np.pi / 180.0 / 3600.0


def mode_planet(label: str) -> str:
    """The planet whose own vector defines mode `label` ('g4' -> 'Mars')."""
    return SOLAR_SYSTEM[int(label[1:]) - 1]


def planet_vector(planet: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    """e*exp(i*varpi) ('g') or sin(i/2)*exp(i*Omega) ('s') of one planet's saved elements."""
    if mode == 'g':
        return np.asarray(planet['e']) * np.exp(1j * np.asarray(planet['varpi']))
    return np.sin(np.asarray(planet['inc']) / 2.0) * np.exp(1j * np.asarray(planet['Omega']))


def mode_phase(times_years, z, label: str, window: float = DEFAULT_WINDOW, frequencies=None) -> np.ndarray:
    """phi(t), radians, unwrapped: the phase of mode `label` in the vector `z`, against its carrier
    g_j*t (t as given). Fitted at window centres a tenth of a window apart and interpolated; within
    half a window of either end it goes on along the slope of the nearest centres (a line through
    up to six). A baseline shorter than the window is refused: a shorter fit cannot separate g3 from
    g4 (a 200 kyr fit puts g3 51 deg off)."""
    freqs = PLANETARY_FREQUENCIES if frequencies is None else frequencies
    t = np.asarray(times_years, dtype=float)
    z = np.asarray(z)
    labels = [f'{label[0]}{j}' for j in range(1, 9)]
    omegas = np.array([freqs[k] for k in labels]) * RAD_PER_ARCSEC
    j = labels.index(label)

    span = t[-1] - t[0]
    if span < window:
        raise ValueError(f"a baseline of {span:.3g} yr is shorter than the phase window {window:.3g} yr")
    half = 0.5 * window
    centres = np.arange(t[0] + half, t[-1] - half + 1e-9 * span, window / 10.0)

    phases = np.empty(len(centres))
    for n, tc in enumerate(centres):
        inside = np.flatnonzero(np.abs(t - tc) <= half + 1e-9 * span)
        inside = inside[:: max(1, len(inside) // MAX_SAMPLES_PER_FIT)]
        basis = np.exp(1j * np.outer(t[inside] - tc, omegas))
        coefficients = np.linalg.lstsq(basis, z[inside], rcond=None)[0]
        # the mode at tc is c_j: arg(c_j) is its longitude there, omega_j*tc + phi(tc)
        phases[n] = np.angle(coefficients[j]) - omegas[j] * tc
    phases = np.unwrap(phases)
    result = np.interp(t, centres, phases)
    if len(centres) >= 3:  # half a window at either end: continue the phase along its slope there
        edge = min(len(centres), 6)
        for side, chosen in ((t < centres[0], slice(0, edge)), (t > centres[-1], slice(-edge, None))):
            slope, intercept = np.polyfit(centres[chosen], phases[chosen], 1)
            result[side] = slope * t[side] + intercept
    return result


def measured_longitudes(
    times_years, planets_data: Dict[str, Dict[str, np.ndarray]], labels: Iterable[str], window: float = DEFAULT_WINDOW, frequencies=None
) -> Dict[str, np.ndarray]:
    """{label: g_j*t + phi_j(t) mod 2 pi} for the modes in `labels` (s5 and unknown labels are skipped:
    s5 is the invariable plane, never in an angle). `planets_data` is `IntegrationEngine.planets_data`,
    sampled at `times_years` (years from the start of the integration). Empty for a baseline shorter
    than the window: the angles then keep the linear law."""
    freqs = PLANETARY_FREQUENCIES if frequencies is None else frequencies
    t = np.asarray(times_years, dtype=float)
    result = {}
    if t[-1] - t[0] < window:  # too short to separate the modes: the linear law stays
        return result
    for label in sorted(set(labels)):
        if label == 's5' or label not in freqs or label[0] not in 'gs':
            continue
        planet = planets_data[mode_planet(label)]
        if len(planet.get('e', ())) != len(t):
            raise ValueError(f"planet data of {mode_planet(label)} has {len(planet.get('e', ()))} samples, the angles {len(t)}")
        phi = mode_phase(t, planet_vector(planet, label[0]), label, window, freqs)
        result[label] = np.mod(freqs[label] * RAD_PER_ARCSEC * t + phi, 2.0 * np.pi)
    return result
