import numpy as np

from resonances.resonance.secular import GeneralSecularResonance
import resonances.secular.proper_angle as proper_angle


class DummyBody:
    def __init__(self, ecc, inc, Omega, omega):
        self.ecc = ecc
        self.inc = inc
        self.Omega = Omega
        self.omega = omega


def test_non_singular_elements():
    e = np.array([0.1, 0.2])
    inc = np.array([np.deg2rad(10), np.deg2rad(20)])
    Omega = np.array([0.1, 0.2])
    varpi = np.array([0.3, 0.5])

    k, h, q, p = proper_angle.non_singular_elements(e, inc, Omega, varpi)

    np.testing.assert_allclose(k, e * np.cos(varpi))
    np.testing.assert_allclose(h, e * np.sin(varpi))
    np.testing.assert_allclose(q, np.sin(inc / 2.0) * np.cos(Omega))
    np.testing.assert_allclose(p, np.sin(inc / 2.0) * np.sin(Omega))


def test_proper_angle_reconstruction_matches_synthetic_series():
    # Synthetic secular evolution for z1 = g-g6+s-s6
    times = np.linspace(0, 2 * np.pi * 5e3, 4000)  # internal units
    times_years = times / (2.0 * np.pi)
    g_true = 0.04  # rad/yr
    s_true = -0.02  # rad/yr
    phi_g = 0.3
    phi_s = -0.5

    varpi_true = g_true * times_years + phi_g
    Omega_true = s_true * times_years + phi_s

    ecc = np.full_like(times, 0.1)
    inc = np.full_like(times, np.deg2rad(15.0))
    Omega = Omega_true
    omega = varpi_true - Omega_true

    body = DummyBody(ecc, inc, Omega, omega)
    resonance = GeneralSecularResonance(formula='g-g6+s-s6')

    # Use simplified planetary frequencies so reconstruction is predictable
    planetary_freqs = {'g6': np.rad2deg(0.03) * 3600.0, 's6': np.rad2deg(-0.015) * 3600.0}

    osculating_angle = (varpi_true + Omega_true) % (2.0 * np.pi)
    proper = proper_angle.build_proper_angle_series(
        times,
        body,
        resonance,
        planetary_freqs=planetary_freqs,
        existing_angle=osculating_angle,
        cutoff_period_years=0.0,
    )

    # Expected proper angle with same frequencies and phase alignment
    planet_varpi = 0.03 * times_years
    planet_Omega = -0.015 * times_years
    sigma_expected = (varpi_true + Omega_true - planet_varpi - planet_Omega) % (2.0 * np.pi)

    # Alignment step makes series start at same value
    assert np.isclose(proper[0], sigma_expected[0])
    diff = np.unwrap(proper) - np.unwrap(sigma_expected)
    # After filtering, residual drift should be small
    assert np.allclose(diff.mean(), 0.0, atol=0.5)
