"""The planets' mode phases measured in the integration (resonances/secular/planet_modes.py).

The vectors are built by hand from lines whose phases are known, so the expected phase of every test
is written down before the fit: a constant phase must come back as it is, a drifting one (60 deg in
10 Myr, quadratic, like the wandering of s4 in the package's model) must be followed, and the
neighbouring mode 0.54"/yr away (g4 next to g3) must not leak into it.
"""

import numpy as np
import pytest

from resonances.data.const import PLANETARY_FREQUENCIES as F
from resonances.secular import planet_modes as pm

T = np.linspace(0.0, 1e7, 20001)
RAD = np.pi / 180.0 / 3600.0


def line(label, amplitude, phase):
    return amplitude * np.exp(1j * (F[label] * RAD * T + phase))


def wrapped(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def test_a_constant_phase_comes_back():
    # Earth's vector: g3 0.016 at 40 deg next to g4 0.012 and g5 0.019 (amplitudes of La90's size)
    z = line('g3', 0.016, np.radians(40.0)) + line('g4', 0.012, np.radians(200.0)) + line('g5', 0.019, np.radians(30.0))
    phi = pm.mode_phase(T, z, 'g3')
    assert np.max(np.abs(wrapped(phi - np.radians(40.0)))) < np.radians(0.5)


def test_a_drifting_phase_is_followed_to_the_ends():
    drift = np.radians(60.0) * (T / 1e7) ** 2
    z = 0.002 * np.exp(1j * (F['s4'] * RAD * T + np.radians(280.0) + drift)) + line('s3', 0.004, 1.0) + 0.0138 * np.exp(1j * 1.9)
    phi = pm.mode_phase(T, z, 's4')
    error = np.degrees(wrapped(phi - np.radians(280.0) - drift))
    assert np.max(np.abs(error[(T > 2.5e6) & (T < 7.5e6)])) < 2.0  # inside: windows of +-2.5 Myr
    # the ends, continued along the slope of the 2.5-5 Myr centres: a line under a parabola misses its
    # ends by 60 deg * (2.5 Myr / 10 Myr)^2 * ~2 = 7.5 deg at most
    assert np.max(np.abs(error)) < 8.0
    held = np.degrees(wrapped(np.interp(T, [2.5e6, 7.5e6], [phi[5000], phi[15000]]) - np.radians(280.0) - drift))
    assert np.abs(held[-1]) > 20.0  # what holding the phase would have cost at 10 Myr


def test_measured_longitudes_of_planets_on_the_linear_law_are_the_linear_law():
    phases = {'g6': 2.2, 's6': -0.7, 'g3': 0.3}
    planets = {}
    z6 = line('g6', 0.048, phases['g6']) + line('g5', 0.033, 0.5)
    zeta6 = line('s6', 0.008, phases['s6']) + 0.0138 * np.exp(1j * 1.9)
    planets['Saturn'] = {'e': np.abs(z6), 'varpi': np.angle(z6), 'inc': 2 * np.arcsin(np.abs(zeta6)), 'Omega': np.angle(zeta6)}
    z3 = line('g3', 0.016, phases['g3']) + line('g4', 0.012, 2.0)
    planets['Earth'] = {'e': np.abs(z3), 'varpi': np.angle(z3), 'inc': np.full(len(T), 1e-3), 'Omega': np.zeros(len(T))}
    longitudes = pm.measured_longitudes(T, planets, ['g6', 's6', 'g3', 's5'])
    assert set(longitudes) == {'g6', 's6', 'g3'}  # s5 is the invariable plane, never in an angle
    for label, phase in phases.items():
        expected = F[label] * RAD * T + phase
        assert np.max(np.abs(wrapped(longitudes[label] - expected))) < np.radians(0.5), label


def test_mode_planets_and_a_length_mismatch():
    assert pm.mode_planet('g4') == 'Mars' and pm.mode_planet('s8') == 'Neptune'
    with pytest.raises(ValueError):
        pm.measured_longitudes(T, {'Mars': {'e': np.ones(10), 'varpi': np.zeros(10)}}, ['g4'])


def test_a_baseline_shorter_than_the_window_keeps_the_linear_law():
    # 2 Myr cannot separate g3 from g4 (0.54"/yr apart, a 2.4 Myr beat): nothing is measured
    short = T[T <= 2e6]
    z3 = 0.016 * np.exp(1j * (F['g3'] * RAD * short + 0.3)) + 0.012 * np.exp(1j * (F['g4'] * RAD * short + 2.0))
    planets = {'Earth': {'e': np.abs(z3), 'varpi': np.angle(z3), 'inc': np.full(len(short), 1e-3), 'Omega': np.zeros(len(short))}}
    assert pm.measured_longitudes(short, planets, ['g3']) == {}
    with pytest.raises(ValueError):
        pm.mode_phase(short, z3, 'g3')
