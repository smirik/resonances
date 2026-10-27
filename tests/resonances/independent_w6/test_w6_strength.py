"""Independent tests of the W6 two-body strength (direct averaging) and the free/forced split (C15 part 3).

Every expected value is derived here, not read from the implementation:

- the averaged disturbing function <D>(sigma) by a separate brute-force quadrature (own Kepler solver, own rotation
  matrices, explicit loop over |k| turns of lambda_P, no index-shift trick);
- the curvature at a low-e first-order (2:1) and second-order (3:1) interior resonance from the classical expansion
  <D> = const + f_d(alpha) e^q cos(sigma) / a_P (Murray & Dermott 1999, Table B.4 arguments 4D1.1 / 4D0.1), with the
  Laplace coefficients integrated here (f27(2:1) = -1.19049 as a cross-check against M&D Table 8.5);
- the co-orbital circular planar <D> = 1/(2 sin(sigma/2)) - cos(sigma), whose minima at 60/300 deg have
  <D>'' = 9/4 (derived by hand) and closest approach 2 sin 30 deg = 1;
- the varpi rate of z = A e^{igt} + B e^{ig5 t}: extrema (Ag + Bg5)/(A+B) and (Ag - Bg5)/(A-B) (hand derivation).
"""

import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from resonances.resonance.analysis.mmr import averaged_potential as ap
from resonances.resonance.analysis.mmr import kinematics as kin
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec, Strength
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

TWO_PI = 2 * math.pi
ARCSEC = math.pi / (180 * 3600)
QUAD = ap.Quadrature(nodes_per_turn=720, max_nodes_per_turn=5760, sigma_grid=72, half_width_rad=math.radians(2.0), tolerance=0.02)


# --------------------------------------------------------------------------------------------------------------
# Independent machinery
# --------------------------------------------------------------------------------------------------------------


def _laplace(j, alpha, n=4096):
    """b_{1/2}^{(j)}(alpha) = (1/pi) int_0^{2pi} cos(j psi) / sqrt(1 - 2 alpha cos psi + alpha^2) dpsi (trapezoid)."""
    psi = np.arange(n) * TWO_PI / n
    return 2.0 * np.mean(np.cos(j * psi) / np.sqrt(1 - 2 * alpha * np.cos(psi) + alpha**2))


def _laplace_derivs(j, alpha, h=1e-4):
    b0, bp, bm = _laplace(j, alpha), _laplace(j, alpha + h), _laplace(j, alpha - h)
    return b0, (bp - bm) / (2 * h), (bp - 2 * b0 + bm) / h**2


def _f27(alpha):
    """First-order direct coefficient of e cos(2 lambda' - lambda - varpi), j = 2: (1/2)(-2j - alpha D) b."""
    b, db, _ = _laplace_derivs(2, alpha)
    return 0.5 * (-4.0 * b - alpha * db)


def _f_second(j, alpha):
    """Second-order direct coefficient of e^2 cos(j lambda' + (2 - j) lambda - 2 varpi) (M&D 4D0.1):
    (1/8)[-5j + 4j^2 + (-2 + 4j) alpha D + alpha^2 D^2] b_{1/2}^{(j)}."""
    b, db, d2b = _laplace_derivs(j, alpha)
    return 0.125 * ((-5 * j + 4 * j * j) * b + (-2 + 4 * j) * alpha * db + alpha**2 * d2b)


def _kepler_E(M, e):
    E = np.array(M, dtype=float) + e * np.sin(M)
    for _ in range(50):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def _position(a, e, inc, omega, Omega, M):
    """Heliocentric position (n, 3): Rz(Omega) Rx(inc) Rz(omega) applied to the orbit-plane vector."""
    E = _kepler_E(np.asarray(M, dtype=float), e)
    x = a * (np.cos(E) - e)
    y = a * math.sqrt(1 - e * e) * np.sin(E)
    co, so, cO, sO, ci, si = math.cos(omega), math.sin(omega), math.cos(Omega), math.sin(Omega), math.cos(inc), math.sin(inc)
    Rz_o = np.array([[co, -so, 0], [so, co, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    Rz_O = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
    R = Rz_O @ Rx @ Rz_o
    return np.stack([x, y, np.zeros_like(x)], axis=1) @ R.T


def _avg_D(sigma, body, planet, k, kp, phase, per_turn=1500):
    """<D>(sigma) by brute force: lambda_P on |k| turns, M = (sigma - kp lambda_P - phase) / k on the line."""
    a, e, i, w, W = body
    ap_, ep, ip, wp, Wp = planet
    lam_p = np.arange(per_turn * abs(k)) * TWO_PI / per_turn
    M = (sigma - kp * lam_p - phase) / k
    Mp = lam_p - (wp + Wp)
    r = _position(a, e, i, w, W, M)
    rp = _position(ap_, ep, ip, wp, Wp, Mp)
    d = np.linalg.norm(r - rp, axis=1)
    rp3 = np.linalg.norm(rp, axis=1) ** 3
    return float(np.mean(1.0 / d - np.sum(r * rp, axis=1) / rp3))


def _ellipse(elements):
    a, e, i, w, W = elements
    return ap.Ellipse(a=a, orbit=Orbit(e=e, inc=i, omega=w, Omega=W))


# --------------------------------------------------------------------------------------------------------------
# averaged_potential: the profile against brute force (3D, eccentric inclined planet, direct + indirect)
# --------------------------------------------------------------------------------------------------------------

BODY_3D = (0.55, 0.12, math.radians(8.0), 0.9, 0.4)
PLANET_3D = (1.0, 0.048, math.radians(1.3), 4.8, 1.75)


@pytest.mark.parametrize('k,kp,phase', [(-1, 2, 0.37), (-3, 5, -1.1), (-2, 5, 2.2)])
def test_profile_matches_brute_force_3d(k, kp, phase):
    body = BODY_3D if k == -1 else (0.62 if k == -2 else 0.70, 0.12, math.radians(8.0), 0.9, 0.4)
    L = 120
    sigma, values = ap.averaged_profile(_ellipse(body), _ellipse(PLANET_3D), k, kp, phase, L)
    assert len(sigma) == L and len(values) == L
    assert np.allclose(np.mod(sigma, TWO_PI), np.arange(L) * TWO_PI / L, atol=1e-12)
    for m in (0, 17, 45, 77, 101):
        expected = _avg_D(sigma[m], body, PLANET_3D, k, kp, phase)
        assert values[m] == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize('k,kp,phase', [(-1, 2, 0.37), (-3, 5, -1.1)])
def test_centre_curvature_matches_brute_force_second_derivative(k, kp, phase):
    body = BODY_3D if k == -1 else (0.70, 0.12, math.radians(8.0), 0.9, 0.4)
    grid = np.arange(360) * TWO_PI / 360
    prof = np.array([_avg_D(s, body, PLANET_3D, k, kp, phase, per_turn=600) for s in grid])
    i0 = int(np.argmin(prof))
    res = minimize_scalar(
        lambda s: _avg_D(s, body, PLANET_3D, k, kp, phase, per_turn=600),
        bounds=(grid[i0] - 0.02, grid[i0] + 0.02),
        method='bounded',
        options={'xatol': 1e-7},
    )
    c, h = res.x, 0.01
    f = [_avg_D(c + d, body, PLANET_3D, k, kp, phase, per_turn=600) for d in (-h, 0.0, h)]
    d2 = (f[0] - 2 * f[1] + f[2]) / h**2
    assert d2 > 0
    out = ap.centre_curvature(_ellipse(body), _ellipse(PLANET_3D), k, kp, phase, c + math.radians(5.0), QUAD)
    assert out.measured
    assert out.curvature == pytest.approx(d2, rel=0.01)
    assert ap.circular_distance(out.centre_rad, np.mod(c, TWO_PI)) < math.radians(0.3)
    assert out.offset_rad == pytest.approx(ap.circular_distance(out.centre_rad, c + math.radians(5.0)), abs=1e-9)
    assert out.curvature_half == pytest.approx(out.curvature, rel=QUAD.tolerance)


# --------------------------------------------------------------------------------------------------------------
# Literature cross-checks: classical expansion at low e (planar, circular planet)
# --------------------------------------------------------------------------------------------------------------


def test_own_laplace_coefficient_reproduces_md_table():
    assert _f27(0.5 ** (2 / 3)) == pytest.approx(-1.19049, abs=2e-5)  # M&D Table 8.5, 2:1


@pytest.mark.parametrize('alpha', [0.5 ** (2 / 3), 0.6, 0.66])
@pytest.mark.parametrize('varpi', [0.0, 0.7])
def test_first_order_curvature_is_minus_f27_e(alpha, varpi):
    """sigma = 2 lambda' - lambda - varpi = -M - 2 varpi + 2 lambda': <D> = const + f27 e cos(sigma) + f e^2 cos(2 sigma)
    (f = 4D0.1 at j = 4; the indirect part has no such argument for a circular planet); f27 < 0, so the minimum is at
    sigma = 0 with <D>'' = -f27 e - 4 f e^2 (a_P = 1). At e = 0.002 the second harmonic is ~1 % of the curvature."""
    e = 0.002
    body = ap.Ellipse(alpha, Orbit(e=e, inc=0.0, omega=varpi, Omega=0.0))
    planet = ap.Ellipse(1.0, Orbit(e=0.0, inc=0.0, omega=0.0, Omega=0.0))
    out = ap.centre_curvature(body, planet, -1, 2, -2 * varpi, 0.3, QUAD)
    assert out.measured
    assert out.curvature == pytest.approx(-_f27(alpha) * e - 4 * _f_second(4, alpha) * e * e, rel=2e-3)
    assert ap.circular_distance(out.centre_rad, 0.0) < math.radians(0.5)
    assert out.min_distance_au == pytest.approx(1.0 - alpha, abs=3 * e)


def test_second_order_curvature_is_f_e_squared():
    """sigma = 3 lambda' - lambda - 2 varpi (= -M - 3 varpi + 3 lambda'): <D> = const + f e^2 cos(sigma); f > 0, so
    the minimum is at sigma = pi with <D>'' = f e^2."""
    alpha, e, varpi = (1 / 3) ** (2 / 3), 0.02, 1.2
    f = _f_second(3, alpha)
    assert f > 0
    body = ap.Ellipse(alpha, Orbit(e=e, inc=0.0, omega=varpi, Omega=0.0))
    planet = ap.Ellipse(1.0, Orbit(e=0.0, inc=0.0, omega=0.0, Omega=0.0))
    out = ap.centre_curvature(body, planet, -1, 3, -3 * varpi, math.pi + 0.4, QUAD)
    assert out.measured
    assert out.curvature == pytest.approx(f * e * e, rel=0.02)
    assert ap.circular_distance(out.centre_rad, math.pi) < math.radians(0.5)


# --------------------------------------------------------------------------------------------------------------
# Co-orbital closed form: the basin of the measured centre, scale with a
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('observed_deg,centre_deg', [(50.0, 60.0), (95.0, 60.0), (170.0, 60.0), (190.0, 300.0), (320.0, 300.0)])
def test_tadpole_basin_and_nine_quarters(observed_deg, centre_deg):
    """Circular planar a = a_P = 1: D(sigma) = 1/(2 sin(sigma/2)) - cos(sigma) exactly (no fast dependence);
    max at 180 deg, minima 60/300 deg with D'' = 1/(8 sin) + cos^2/(4 sin^3) + cos = 1/4 + 3/2 + 1/2 = 9/4."""
    circ = Orbit(e=0.0, inc=0.0, omega=0.0, Omega=0.0)
    out = ap.centre_curvature(ap.Ellipse(1.0, circ), ap.Ellipse(1.0, circ), 1, -1, 0.0, math.radians(observed_deg), QUAD)
    assert out.measured
    assert math.degrees(out.centre_rad) == pytest.approx(centre_deg, abs=0.05)
    assert out.curvature == pytest.approx(9 / 4, rel=0.005)
    assert out.min_distance_au == pytest.approx(1.0, abs=1e-3)
    assert math.degrees(out.offset_rad) == pytest.approx(abs(observed_deg - centre_deg), abs=0.05)


def test_curvature_scales_as_inverse_a():
    circ = Orbit(e=0.0, inc=0.0, omega=0.0, Omega=0.0)
    out = ap.centre_curvature(ap.Ellipse(2.0, circ), ap.Ellipse(2.0, circ), 1, -1, 0.0, math.radians(300.0), QUAD)
    assert out.curvature == pytest.approx(9 / 8, rel=0.005)
    assert out.min_distance_au == pytest.approx(2.0, abs=2e-3)


def test_basin_minimum_on_synthetic_sequence():
    x = np.arange(36) * TWO_PI / 36
    v = np.cos(2 * x) + 0.3 * np.cos(x)  # minima near 90 and 270 deg, the one at 90/270 split by the max at 0 and 180
    i90, i270 = int(np.argmin(np.where(x < math.pi, v, np.inf))), int(np.argmin(np.where(x > math.pi, v, np.inf)))
    assert ap.basin_minimum(v, 3) == i90
    assert ap.basin_minimum(v, 15) == i90
    assert ap.basin_minimum(v, 20) == i270
    assert ap.basin_minimum(v, 33) == i270  # descends away from the maximum at 0


def test_circular_helpers():
    assert ap.circular_distance(0.1, TWO_PI - 0.1) == pytest.approx(0.2)
    assert ap.circular_distance(0.0, math.pi) == pytest.approx(math.pi)
    assert ap.wrap_two_pi(-0.5) == pytest.approx(TWO_PI - 0.5)


# --------------------------------------------------------------------------------------------------------------
# strength: conventions, the check, the implausible flag
# --------------------------------------------------------------------------------------------------------------


def test_mu_eff_required_and_omega0_theory_hand_numbers():
    # k = 3, P = 10 yr (n = 0.6283185 rad/yr), mu = 1e-5: omega0 = 3 n sqrt(3e-5) = 0.01032433 rad/yr, T0 = 608.581 yr
    n = TWO_PI / 10.0
    assert st.omega0_theory(3, n, 1e-5) == pytest.approx(0.01032433, rel=1e-6)
    assert st.omega0_theory(-3, n, 1e-5) == pytest.approx(0.01032433, rel=1e-6)
    assert st.mu_eff_required(608.5806, 3, n) == pytest.approx(1e-5, rel=1e-5)
    assert math.isnan(st.mu_eff_required(float('nan'), 3, n))


def test_snapshot_times_and_hill_radius():
    assert np.allclose(st.snapshot_times(0.0, 30.0, 3), [5.0, 15.0, 25.0])
    assert np.allclose(st.snapshot_times(100.0, 200.0, 4), [112.5, 137.5, 162.5, 187.5])
    assert st.hill_radius(5.2026, 9.5479e-4) == pytest.approx(0.3552, rel=2e-3)  # Jupiter's Hill sphere 0.355 au


def test_nearest_to_one_is_the_overlap_reading():
    assert st.nearest_to_one(0.2, 0.5) == pytest.approx(0.5)
    assert st.nearest_to_one(2.0, 5.0) == pytest.approx(2.0)
    assert st.nearest_to_one(0.5, 3.0) == pytest.approx(1.0)


def test_strength_check_overlap_with_ten():
    s = AnalysisSettings()
    # range [12, 40] lies above 10: fail; [0.03, 0.09] below 1/10: fail; [5, 40] overlaps: pass
    assert st.strength_check(True, st.nearest_to_one(12, 40), s) == CheckValue.FAIL
    assert st.strength_check(True, st.nearest_to_one(0.03, 0.09), s) == CheckValue.FAIL
    assert st.strength_check(True, st.nearest_to_one(5, 40), s) == CheckValue.PASS
    assert st.strength_check(True, st.nearest_to_one(0.05, 0.11), s) == CheckValue.PASS
    assert st.strength_check(False, 1.0, s) == CheckValue.NOT_MEASURED
    assert st.strength_check(True, float('nan'), s) == CheckValue.NOT_MEASURED


def test_implausible_is_two_sided_with_a_theory_value():
    s = AnalysisSettings()
    assert st.implausible_strength(True, 2e3, s)
    assert st.implausible_strength(True, 5e-4, s)
    assert not st.implausible_strength(True, 1.0, s)
    assert not st.implausible_strength(True, 500.0, s)
    assert st.implausible_strength(False, 2e3, s)  # mu alone: only far above is implausible
    assert not st.implausible_strength(False, 5e-4, s)
    assert not st.implausible_strength(True, float('nan'), s)


def _strength(method):
    return Strength(
        mu_scale=1e-5,
        mu_scale_no_eq=1e-4,
        eq_legal=True,
        mu_eff_required=1e-5,
        ratio=1.0,
        omega0_theory_rad_yr=1e-3,
        period_theory_yr=6283.0,
        implausible=False,
        check=CheckValue.PASS,
        method=method,
    )


def test_physics_margins_only_count_direct_averaging():
    margins = ('L4', 'R0', 'Z', 'W', 'F')
    assert 'F' in st.physics_margins(margins, _strength('direct_averaging'))
    assert 'F' not in st.physics_margins(margins, _strength('eq_scale'))
    assert 'F' not in st.physics_margins(margins, None)
    assert set(st.physics_margins(margins, _strength('eq_scale'))) == {'L4', 'R0', 'Z', 'W'}


@pytest.mark.parametrize(
    'key,applicable',
    [('2N-3+0+1', True), ('1J-3r0', True), ('1E-1+0+0', False), ('3J-1S-1+0+0-1', False), (None, False)],
)
def test_direct_applicable(key, applicable):
    spec = ResonanceSpec.from_key(key) if key else None
    assert st.direct_applicable(spec) is applicable


def test_planet_phase_and_body_coefficient():
    spec = ResonanceSpec.from_key('3J-1-1-1')  # sigma = 3 lJ - l - varpi_J - varpi
    assert st.planet_phase(spec, 0.3, 0.9) == pytest.approx(-0.3)


def _series(key, n=201, varpi=0.4, Omega=1.0, inc=0.1, e=0.05, a=2.5):
    spec = ResonanceSpec.from_key(key)
    t = np.arange(n) * 10.0
    omega = np.mod(varpi - Omega, TWO_PI)
    return MMRSeries(
        t_yr=t,
        a=np.full(n, a),
        e=np.full(n, e),
        inc=np.full(n, inc),
        Omega=np.full(n, Omega),
        omega=np.full(n, omega),
        varpi=np.full(n, varpi),
        angle_wrapped=np.zeros(n),
        resonance=spec,
        regime=Regime(step_yr=10.0, span_yr=t[-1], n_rows=n),
        name='synthetic',
    )


def test_body_ellipse_prograde_phase():
    """2N-3+0+1: sigma = 2 lN - 3 l + varpi = -3 M + (k + p) varpi + ... with k = -3, p = 1: phi_B = -2 varpi."""
    series = _series('2N-3+0+1', varpi=0.4, Omega=1.0, inc=0.1)
    ell, phi = st.body_ellipse(series, slice(20, 180))
    assert ell.a == pytest.approx(2.5)
    assert ell.orbit.e == pytest.approx(0.05)
    assert ell.orbit.inc == pytest.approx(0.1)
    assert ap.circular_distance(ell.orbit.Omega, 1.0) < 1e-9
    assert ap.circular_distance(ell.orbit.omega + ell.orbit.Omega, 0.4) < 1e-9
    assert ap.circular_distance(phi, -0.8) < 1e-9


def test_body_ellipse_retrograde_phase():
    """1J-3r0: k = 3, p = -4, v = 0; varpi* = omega - Omega: phi_B = (3 - 4)(omega - Omega) = Omega - omega."""
    Omega, omega = 1.0, 0.7
    series = _series('1J-3r0', varpi=np.mod(omega + Omega, TWO_PI), Omega=Omega, inc=math.radians(150.0), e=0.2)
    ell, phi = st.body_ellipse(series, slice(0, 201))
    assert ell.a == pytest.approx(2.5)
    assert ell.orbit.e == pytest.approx(0.2)
    assert ap.circular_distance(phi, Omega - omega) < 1e-9


# --------------------------------------------------------------------------------------------------------------
# kinematics: free/forced split, rate range, the cutoff vector
# --------------------------------------------------------------------------------------------------------------

G5 = 4.25749319 * ARCSEC  # rad/yr
G_FREE = 231.29 * ARCSEC


def _two_vectors(t, a, g, b, g5, pa=0.4, pb=2.0):
    return a * np.exp(1j * (g * t + pa)) + b * np.exp(1j * (g5 * t + pb))


@pytest.mark.parametrize('A,B', [(0.05, 0.0286), (0.0286, 0.05)])
def test_free_forced_split_recovers_known_vectors(A, B):
    t = np.arange(0.0, 100_000.0 + 1, 10.0)
    z = _two_vectors(t, A, G_FREE, B, G5)
    e_free, e_forced, g = kin.free_forced_split(t, np.abs(z), np.mod(np.angle(z), TWO_PI), G5)
    assert e_free == pytest.approx(A, rel=1e-3)
    assert e_forced == pytest.approx(B, rel=1e-3)
    assert g == pytest.approx(G_FREE, rel=1e-3)


def test_free_forced_split_short_record():
    """0.6 of the modulation cycle 2 pi / (g - g5): the docstring promises the residual minimum handles it."""
    period = TWO_PI / (G_FREE - G5)
    t = np.arange(0.0, 0.6 * period, 5.0)
    z = _two_vectors(t, 0.05, G_FREE, 0.0286, G5)
    e_free, e_forced, g = kin.free_forced_split(t, np.abs(z), np.angle(z), G5)
    assert g == pytest.approx(G_FREE, rel=0.02)
    assert e_free == pytest.approx(0.05, rel=0.02)
    assert e_forced == pytest.approx(0.0286, rel=0.05)


def test_varpi_rate_range_hand_formula_and_sampling():
    A, B = 0.05, 0.0286
    lo_hi = sorted([(A * G_FREE + B * G5) / (A + B), (A * G_FREE - B * G5) / (A - B)])
    got = kin.varpi_rate_range(A, B, G_FREE, G5)
    assert got[0] == pytest.approx(lo_hi[0], rel=1e-9)
    assert got[1] == pytest.approx(lo_hi[1], rel=1e-9)
    # sampling the rate of the actual vector over one cycle stays inside and reaches both ends
    period = TWO_PI / (G_FREE - G5)
    t = np.linspace(0.0, period, 20001)
    rate = np.gradient(np.unwrap(np.angle(_two_vectors(t, A, G_FREE, B, G5))), t)
    assert rate.min() == pytest.approx(got[0], rel=1e-3)
    assert rate.max() == pytest.approx(got[1], rel=1e-3)
    assert kin.varpi_rate_range(0.03, 0.03, G_FREE, G5) == (-math.inf, math.inf)
    assert kin.varpi_rate_range(0.0, 0.03, G_FREE, G5) == pytest.approx((G5, G5))


def test_varpi_rate_modulation_from_modulus():
    A, B = 0.05, 0.0286
    t = np.linspace(0.0, 30_000.0, 30001)
    e = np.abs(_two_vectors(t, A, G_FREE, B, G5))
    e_free, e_forced, lo, hi, period = kin.varpi_rate_modulation(e, G_FREE, G5)
    assert e_free == pytest.approx(A, rel=1e-4)
    assert e_forced == pytest.approx(B, rel=1e-4)
    assert period == pytest.approx(TWO_PI / (G_FREE - G5), rel=1e-9)
    assert (lo, hi) == pytest.approx(kin.varpi_rate_range(A, B, G_FREE, G5), rel=1e-3)


def _vector_series(z, t):
    n = len(t)
    return MMRSeries(
        t_yr=t,
        a=np.full(n, 2.5),
        e=np.abs(z),
        inc=np.full(n, 0.1),
        Omega=np.full(n, 1.0),
        omega=np.mod(np.angle(z) - 1.0, TWO_PI),
        varpi=np.mod(np.angle(z), TWO_PI),
        angle_wrapped=np.zeros(n),
        resonance=ResonanceSpec.from_key('3J-1-1-1'),
        regime=Regime(step_yr=float(t[1] - t[0]), span_yr=float(t[-1] - t[0]), n_rows=n),
        name='synthetic',
    )


def test_short_period_vector_filters_components_not_modulus():
    """z = slow two-vector + 0.01 e^{i 2 pi t / 12 yr}: at P_c = 500 yr the vector components keep the slow part
    (gain > 0.9999 at 5.7 kyr) and lose the 12-yr term; e and varpi are the modulus and argument of the slow vector."""
    t = np.arange(0.0, 60_000.0, 2.0)
    slow = _two_vectors(t, 0.05, G_FREE, 0.0286, G5)
    z = slow + 0.01 * np.exp(1j * TWO_PI * t / 12.0)
    e, varpi = kin.short_period_vector(_vector_series(z, t), 500.0, AnalysisSettings())
    mid = slice(len(t) // 10, -len(t) // 10)
    assert np.max(np.abs(e[mid] - np.abs(slow[mid]))) < 2e-4
    dphi = np.angle(np.exp(1j * (varpi[mid] - np.angle(slow[mid]))))
    assert np.max(np.abs(dphi)) < 0.01
    # the modulus of the raw vector carries the 12-yr term at ~0.01: the filter really removed it
    assert np.max(np.abs(np.abs(z[mid]) - np.abs(slow[mid]))) > 5e-3


def test_short_period_vector_raw_on_irregular_grid():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 20_000.0, 4000))
    z = _two_vectors(t, 0.05, G_FREE, 0.0286, G5)
    e, varpi = kin.short_period_vector(_vector_series(z, t), 500.0, AnalysisSettings())
    assert np.allclose(e, np.abs(z))
    assert np.allclose(np.angle(np.exp(1j * (varpi - np.angle(z)))), 0.0, atol=1e-12)
