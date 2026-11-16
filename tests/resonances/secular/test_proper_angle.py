import numpy as np
import pytest

from resonances.secular.secular_resonance import SecularResonance
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
    # Synthetic secular evolution for z1 = g-g6+s-s6 with planetary modes at 0.03 and -0.015 rad/yr
    # and phases 20 and 50 deg at t = 0 (epoch J2000): the angle is varpi + Omega minus those, exactly.
    times = np.linspace(0, 2 * np.pi * 5e3, 4000)  # internal units
    times_years = times / (2.0 * np.pi)
    varpi_true = 0.04 * times_years + 0.3
    Omega_true = -0.02 * times_years - 0.5

    body = DummyBody(np.full_like(times, 0.1), np.full_like(times, np.deg2rad(15.0)), Omega_true, varpi_true - Omega_true)
    proper = proper_angle.build_proper_angle_series(
        times,
        body,
        SecularResonance('g-g6+s-s6'),
        planetary_freqs={'g6': np.rad2deg(0.03) * 3600.0, 's6': np.rad2deg(-0.015) * 3600.0},
        planetary_phases={'g6': 20.0, 's6': 50.0},
    )

    expected = varpi_true + Omega_true - 0.03 * times_years - 0.015 * -times_years - np.deg2rad(20.0 + 50.0)
    np.testing.assert_allclose(np.unwrap(proper), np.unwrap(np.mod(expected, 2 * np.pi)), atol=1e-9)


def test_planetary_longitudes_with_phases_and_epoch():
    # g = 36 arcsec/yr = 0.01 deg/yr, phase 30 deg at J2000, t = 0 is 100 yr after J2000:
    # the longitude at t = 0 is 30 + 1 = 31 deg and at t = 1000 yr it is 31 + 10 = 41 deg.
    freqs = {'g4': 36.0, 's4': -72.0}
    phases = {'g4': 30.0, 's4': 200.0}
    varpi, Omega = proper_angle.build_planetary_longitudes_by_index(np.array([0.0, 1000.0]), [4], freqs, phases, epoch_years=100.0)
    np.testing.assert_allclose(np.rad2deg(varpi[4]), [31.0, 41.0])
    np.testing.assert_allclose(np.rad2deg(Omega[4]), [198.0, 178.0])
    # without phases the epoch is ignored and every longitude starts at zero
    varpi, _ = proper_angle.build_planetary_longitudes_by_index(np.array([0.0, 1000.0]), [4], freqs, None, epoch_years=100.0)
    np.testing.assert_allclose(np.rad2deg(varpi[4]), [0.0, 10.0])


def _low_inclination_orbit(times_years, s_rad, phi, i_free_deg, plane):
    """Orbit whose inclination vector is the invariable plane's constant plus a free circle at rate s."""
    half = np.deg2rad(plane['inc']) / 2.0
    node = np.deg2rad(plane['Omega'])
    zq = np.sin(half) * np.exp(1j * node) + np.sin(np.deg2rad(i_free_deg) / 2.0) * np.exp(1j * (s_rad * times_years + phi))
    inc = 2.0 * np.arcsin(np.abs(zq))
    Omega = np.angle(zq) % (2.0 * np.pi)
    return inc, Omega


def test_invariable_plane_restores_the_node_rate_of_a_low_inclination_orbit():
    # Free inclination 0.5 deg against an invariable plane tilted 1.58 deg from the reference plane:
    # measured from the reference plane, Omega only librates around the invariable node and the
    # angle s - s6 turns at -s6 instead of s - s6. Measured from the invariable plane it turns at s - s6.
    plane = {'inc': 1.5787, 'Omega': 107.5822}
    times = np.linspace(0, 2 * np.pi * 2e6, 20001)
    t = times / (2 * np.pi)
    s_true = np.deg2rad(-30.0 / 3600.0)
    s6 = -26.0
    inc, Omega = _low_inclination_orbit(t, s_true, 0.4, 0.5, plane)
    body = DummyBody(np.full_like(t, 0.1), inc, Omega, np.zeros_like(t))
    resonance = SecularResonance('s-s6')
    common = dict(planetary_freqs={'g6': 28.0, 's6': s6}, planetary_phases={'g6': 0.0, 's6': 10.0})

    with_plane = proper_angle.build_proper_angle_series(times, body, resonance, invariable_plane=plane, **common)
    without = proper_angle.build_proper_angle_series(times, body, resonance, **common)

    def rate(angle):
        return np.polyfit(t, np.unwrap(angle), 1)[0] * 206264.806

    assert rate(with_plane) == pytest.approx(-30.0 - s6, abs=1e-6)
    assert rate(without) == pytest.approx(-s6, abs=0.05)
    # and the phase is the one of the free circle minus the mode: 0.4 rad - 10 deg at t = 0
    assert np.isclose(with_plane[0], (0.4 - np.deg2rad(10.0)) % (2 * np.pi))


def _rad_per_year(arcsec_per_year):
    return np.deg2rad(arcsec_per_year / 3600.0)


def test_removed_modes_keep_the_angle_own_frequencies():
    # 10 Myr: every outer mode is resolved, so the basis is g5..g8 + const and const + s6..s8
    z1 = SecularResonance('g-g6+s-s6').formula.terms
    e, i = proper_angle.removed_modes(z1, 1e7)
    assert set(e) == {'g5', 'g7', 'g8', 'const'} and set(i) == {'const', 's7', 's8'}
    e, i = proper_angle.removed_modes(SecularResonance('g-g6').formula.terms, 1e7)
    assert 'g6' not in e and 'g5' in e and set(i) == {'const', 's6', 's7', 's8'}
    e, i = proper_angle.removed_modes(SecularResonance('s-s6').formula.terms, 1e7)
    assert set(e) == {'g5', 'g6', 'g7', 'g8', 'const'} and 's6' not in i


def test_reduced_vectors_remove_only_the_modes_not_in_the_angle():
    # e exp(i varpi) = free circle at g = 40 arcsec/yr + forced terms at g5 and g6; the inclination is a
    # free circle at s = -45 plus the invariable-plane constant. For g - g6 the g5 term must go and the
    # g6 term stay; for g - g5 the other way round.
    from resonances.data.const import PLANETARY_FREQUENCIES as F

    t = np.linspace(0.0, 1e7, 20001)
    free = 0.05 * np.exp(1j * (_rad_per_year(40.0) * t + 0.3))
    forced5 = 0.03 * np.exp(1j * (_rad_per_year(F['g5']) * t + 1.0))
    forced6 = 0.02 * np.exp(1j * (_rad_per_year(F['g6']) * t + 2.0))
    z_e = free + forced5 + forced6
    zeta = 0.08 * np.exp(1j * (_rad_per_year(-45.0) * t)) + 0.0138 * np.exp(1j * 1.88)
    ecc, varpi = np.abs(z_e), np.angle(z_e)
    inc, Omega = 2 * np.arcsin(np.abs(zeta)), np.angle(zeta)

    # on a finite window the modes are not exactly orthogonal, so the fit takes ~0.3% of the free term
    # into the removed ones: 5e-4 is 1% of the free amplitude
    e_nu6, i_nu6 = proper_angle.reduced_vectors(t, ecc, varpi, inc, Omega, SecularResonance('g-g6').formula.terms)
    np.testing.assert_allclose(e_nu6, free + forced6, atol=5e-4)
    np.testing.assert_allclose(i_nu6, 0.08 * np.exp(1j * (_rad_per_year(-45.0) * t)), atol=5e-4)
    # g5 makes only 3.3 turns in 10 Myr and leaks a little more into the constant term
    e_nu5, _ = proper_angle.reduced_vectors(t, ecc, varpi, inc, Omega, SecularResonance('g-g5').formula.terms)
    np.testing.assert_allclose(e_nu5, free + forced5, atol=1e-3)
    # and what was removed is gone: no g5 left for nu6
    assert abs(np.mean((e_nu6 - free - forced6) * np.exp(-1j * _rad_per_year(F['g5']) * t))) < 1e-4


def test_the_body_wrapper_passes_remove_forced():
    # a regression: build_proper_angle_series once dropped the flag, and the reduced angle silently
    # fell back to the unreduced one
    from resonances.data.const import PLANETARY_FREQUENCIES as F

    t = np.linspace(0.0, 1e7, 20001)
    z_e = 0.03 * np.exp(1j * _rad_per_year(40.0) * t) + 0.04 * np.exp(1j * _rad_per_year(F['g5']) * t)
    body = DummyBody(np.abs(z_e), np.full_like(t, 0.1), np.zeros_like(t), np.angle(z_e))
    args = dict(planetary_phases={'g6': 0.0, 's6': 0.0})
    reduced = proper_angle.build_proper_angle_series(t * 2 * np.pi, body, SecularResonance('g-g6'), remove_forced=True, **args)
    plain = proper_angle.build_proper_angle_series(t * 2 * np.pi, body, SecularResonance('g-g6'), **args)
    assert not np.allclose(reduced, plain)
