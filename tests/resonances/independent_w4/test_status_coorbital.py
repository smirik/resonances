"""Independent W4 tests: status rules of analysis/mmr/decide.py (via analyse_mmr) and the 1:1 law (coorbital.py).

Written from the public API (API.md, docstrings) and W4-TASKS.md only. Every expected value comes from a source
independent of the code under test:

* the planar circular co-orbital potential of Murray & Dermott (1999, eqs. 3.178-3.179):
  R~(theta) = 1 / (2 |sin(theta/2)|) - cos theta, F~ = dR~/dtheta, with hand values in the comments;
* a brute-force average of the direct + indirect disturbing function written here (own Kepler solver,
  own rotation, own gradient, quadrature nodes shifted from the package's);
* synthetic records with a known truth: the averaged co-orbital ODE theta'' = -3 mu n^2 F~(theta) (a from
  theta_dot = n - n_P, so the identity and the force law hold by construction), and a pendulum
  sigma'' = -w0^2 sin sigma for a 2J-1 angle with instantaneous kicks in sigma_dot (= jumps of a), whose turning
  points are counted here from the integrated (unfiltered) angle.
"""

from functools import lru_cache
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.signal import argrelextrema
from scipy.special import ellipk

from resonances.resonance.analysis.mmr import coorbital as co
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

SETTINGS = AnalysisSettings()

MU_EARTH = 1.0 / 332946.05  # m_Earth / M_sun (without the Moon)
N_EARTH = math.radians(360.000505)  # rad / Gaussian yr (the package's table value; a_P = (2 pi / n)^(2/3) ~ 1 au)
N_JUPITER = math.radians(30.349646)

PASS, FAIL, NOT_MEASURED = CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_MEASURED


def _s(value) -> str:
    return str(value.value) if hasattr(value, 'value') else str(value)


def _run(series, settings=SETTINGS):
    return analyse_mmr(series, settings, scan_subresonances=False, scan_owner=False)


def _n_turning_points(sigma: np.ndarray) -> int:
    """Strict local extrema of the unfiltered angle (the truth of the synthetic record)."""
    return len(argrelextrema(sigma, np.greater)[0]) + len(argrelextrema(sigma, np.less)[0])


# ======================================================================================================================
# 1. Closed form of the planar circular problem (M&D 1999 eqs. 3.178-3.179)
# ======================================================================================================================


def _r_circ(theta):
    return 1.0 / (2.0 * np.abs(np.sin(theta / 2))) - np.cos(theta)


def _f_circ(theta):
    # d/dtheta [ (1/2) |sin(theta/2)|^-1 ] = -(1/4) sign(s) cos(theta/2) / s^2 ; d/dtheta [-cos theta] = sin theta
    s = np.sin(theta / 2)
    return -0.25 * np.sign(s) * np.cos(theta / 2) / s**2 + np.sin(theta)


def test_averaged_table_circular_orbits_is_the_closed_form():
    # e = i = 0: R does not depend on lambda_P at fixed theta, the average is R itself.
    circ = co.Orbit(e=0.0, inc=0.0, omega=0.0, Omega=0.0)
    table = co.averaged_table(circ, circ, grid_points=360, softening=1e-7)
    th = np.asarray(table.theta_rad)
    assert len(th) == 360
    np.testing.assert_allclose(th, 2 * np.pi * np.arange(360) / 360, atol=1e-12)
    far = np.abs(np.sin(th / 2)) > math.sin(math.radians(2.5))  # away from the singularity at 0
    np.testing.assert_allclose(table.potential[far], _r_circ(th[far]), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(table.force[far], _f_circ(th[far]), rtol=1e-6, atol=1e-6)


# ======================================================================================================================
# 2. Brute-force average written here (direct + indirect part, M&D eq. 6.44-6.45; a = a_P = 1)
# ======================================================================================================================


def _kepler_e(mean_anomaly, e):
    big_e = mean_anomaly + e * np.sin(mean_anomaly)
    for _ in range(60):
        big_e = big_e - (big_e - e * np.sin(big_e) - mean_anomaly) / (1 - e * np.cos(big_e))
    return big_e


def _position(e, inc, omega, node, mean_anomaly):
    """r and dr/dM for a = 1 (perifocal -> R_z(Omega) R_x(i) R_z(omega))."""
    big_e = _kepler_e(np.asarray(mean_anomaly, float), e)
    b = math.sqrt(1 - e * e)
    xp, yp = np.cos(big_e) - e, b * np.sin(big_e)
    den = 1 - e * np.cos(big_e)
    dxp, dyp = -np.sin(big_e) / den, b * np.cos(big_e) / den
    cw, sw, cn, sn, ci, si = (
        math.cos(omega),
        math.sin(omega),
        math.cos(node),
        math.sin(node),
        math.cos(inc),
        math.sin(inc),
    )
    p = np.array([cn * cw - sn * sw * ci, sn * cw + cn * sw * ci, sw * si])
    q = np.array([-cn * sw - sn * cw * ci, -sn * sw + cn * cw * ci, cw * si])
    return np.outer(xp, p) + np.outer(yp, q), np.outer(dxp, p) + np.outer(dyp, q)


def _brute_average(body, planet, thetas, n_quad, eps):
    """(R~, F~) at theta = lambda - lambda_P; body/planet = (e, inc, omega, Omega); F~ = <grad_r R . dr/dM>."""
    lam_p = 2 * np.pi * np.arange(n_quad) / n_quad + 0.123  # nodes deliberately not the package's
    r_p, _ = _position(*planet, lam_p - planet[2] - planet[3])
    rp3 = np.linalg.norm(r_p, axis=1) ** 3
    pot, force = [], []
    for th in thetas:
        r, dr = _position(*body, th + lam_p - body[2] - body[3])
        d = r_p - r
        q = (d * d).sum(1) + eps**2
        pot.append(np.mean(q**-0.5 - (r * r_p).sum(1) / rp3))
        grad = d * q[:, None] ** -1.5 - r_p / rp3[:, None]
        force.append(np.mean((grad * dr).sum(1)))
    return np.array(pot), np.array(force)


BODY_INCLINED = (0.15, math.radians(12.0), math.radians(40.0), math.radians(110.0))
PLANET_EARTHLIKE = (0.0167, math.radians(1.5), math.radians(103.0), math.radians(-11.0))


def test_averaged_table_matches_brute_force_average():
    # A large softening keeps the integrand smooth, so both quadratures are converged and must agree.
    eps = 0.05
    table = co.averaged_table(co.Orbit(*BODY_INCLINED), co.Orbit(*PLANET_EARTHLIKE), grid_points=360, softening=eps)
    idx = np.arange(0, 360, 15)
    th = np.asarray(table.theta_rad)[idx]
    pot, force = _brute_average(BODY_INCLINED, PLANET_EARTHLIKE, th, 1024, eps)
    np.testing.assert_allclose(table.potential[idx], pot, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(table.force[idx], force, rtol=1e-6, atol=1e-7)


def test_averaged_force_is_the_theta_derivative_of_the_potential():
    # <dR/dlambda> at fixed theta = d<R>/dtheta (eq. 1 of the module): check on the table itself by differences.
    table = co.averaged_table(co.Orbit(*BODY_INCLINED), co.Orbit(*PLANET_EARTHLIKE), grid_points=720, softening=0.05)
    th = np.asarray(table.theta_rad)
    step = th[1] - th[0]
    deriv = (np.roll(table.potential, -1) - np.roll(table.potential, 1)) / (2 * step)
    scale = np.max(np.abs(table.force))
    assert np.max(np.abs(deriv - table.force)) < 2e-3 * scale


def test_planar_eccentric_body_circular_planet_symmetries():
    # Planet circular and planar: rotating the whole system changes nothing, so R~ does not depend on the body's
    # varpi; the reflection theta -> -theta then makes R~ even and F~ odd (F~(0) = F~(pi) = 0).
    planet = co.Orbit(e=0.0, inc=0.0, omega=0.0, Omega=0.0)
    t1 = co.averaged_table(co.Orbit(e=0.1, inc=0.0, omega=0.0, Omega=0.0), planet, grid_points=360, softening=0.01)
    t2 = co.averaged_table(co.Orbit(e=0.1, inc=0.0, omega=2.0, Omega=0.7), planet, grid_points=360, softening=0.01)
    np.testing.assert_allclose(t1.potential, t2.potential, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(t1.force, t2.force, rtol=1e-7, atol=1e-7)
    rev = (-np.arange(360)) % 360  # index of -theta
    np.testing.assert_allclose(t1.potential, t1.potential[rev], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(t1.force, -t1.force[rev], atol=1e-7 * np.max(np.abs(t1.force)))
    # an eccentric orbit removes the singularity at theta = 0: the average stays finite there
    assert np.isfinite(t1.potential[0]) and t1.potential[0] < 50


def test_interpolate_returns_the_table_on_the_grid_and_is_periodic():
    circ = co.Orbit(e=0.0, inc=0.0, omega=0.0, Omega=0.0)
    table = co.averaged_table(co.Orbit(e=0.05, inc=0.0, omega=0.0, Omega=0.0), circ, grid_points=360, softening=0.01)
    th = np.asarray(table.theta_rad)[[10, 100, 250]]
    r1, f1 = table.interpolate(th)
    r2, f2 = table.interpolate(th + 2 * np.pi)
    np.testing.assert_allclose(r1, table.potential[[10, 100, 250]], rtol=1e-12)
    np.testing.assert_allclose(f1, table.force[[10, 100, 250]], rtol=1e-12)
    np.testing.assert_allclose(r2, r1, rtol=1e-9)
    np.testing.assert_allclose(f2, f1, rtol=1e-9)


# ======================================================================================================================
# 3. Kepler helpers, softening
# ======================================================================================================================


@pytest.mark.parametrize("e", [0.0, 0.3, 0.9, 0.99])
def test_eccentric_anomaly_solves_kepler(e):
    m = np.linspace(-3.0, 9.0, 101)
    big_e = co.eccentric_anomaly(m, e)
    resid = np.angle(np.exp(1j * (big_e - e * np.sin(big_e) - m)))
    assert np.max(np.abs(resid)) < 1e-10


def test_kepler_position_hand_cases():
    # circular planar orbit at M = 90 deg: r = (0, 1, 0), dr/dM = (-1, 0, 0)
    r, dr = co.kepler_position(co.Orbit(e=0.0, inc=0.0, omega=0.0, Omega=0.0), np.array([math.pi / 2]))
    np.testing.assert_allclose(r[0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(dr[0], [-1, 0, 0], atol=1e-12)
    # e = 0.5 at pericentre with varpi = 90 deg: r = (1 - e)(cos varpi, sin varpi, 0) = (0, 0.5, 0)
    r, _ = co.kepler_position(co.Orbit(e=0.5, inc=0.0, omega=math.pi / 2, Omega=0.0), np.array([0.0]))
    np.testing.assert_allclose(r[0], [0, 0.5, 0], atol=1e-12)
    # polar circular orbit, node on x, argument of latitude 90 deg: the body is at the north pole
    r, _ = co.kepler_position(co.Orbit(e=0.0, inc=math.pi / 2, omega=math.pi / 2, Omega=0.0), np.array([0.0]))
    np.testing.assert_allclose(r[0], [0, 0, 1], atol=1e-12)


def test_kepler_position_radius_and_derivative():
    orbit = co.Orbit(e=0.4, inc=0.3, omega=1.1, Omega=-0.6)
    m = np.linspace(0, 2 * np.pi, 37)
    r, dr = co.kepler_position(orbit, m)
    big_e = _kepler_e(m, 0.4)
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), 1 - 0.4 * np.cos(big_e), rtol=1e-10)
    h = 1e-6
    rp, _ = co.kepler_position(orbit, m + h)
    rm, _ = co.kepler_position(orbit, m - h)
    np.testing.assert_allclose(dr, (rp - rm) / (2 * h), atol=1e-7)
    mine_r, mine_dr = _position(0.4, 0.3, 1.1, -0.6, m)
    np.testing.assert_allclose(r, mine_r, atol=1e-10)
    np.testing.assert_allclose(dr, mine_dr, atol=1e-9)


def test_hill_softening():
    # r_H / a_P = (mu/3)^(1/3): mu = 3e-6 -> (1e-6)^(1/3) = 0.01 ; x 0.1 = 0.001
    assert co.hill_softening(3e-6, 0.1) == pytest.approx(1e-3, rel=1e-12)
    # Jupiter: mu = 9.5479e-4 -> (3.1826e-4)^(1/3) = 0.068276
    assert co.hill_softening(9.5479e-4, 1.0) == pytest.approx(0.068276, rel=1e-4)


# ======================================================================================================================
# 4. Fits and regime labels
# ======================================================================================================================


def test_scale_fit_exact_line_and_group_constants():
    x = np.linspace(0, 20, 400)
    model = np.sin(x) + 0.3 * np.sin(2.7 * x)
    k, r2 = co.scale_fit(2.5 * model + 0.3, model)
    assert k == pytest.approx(2.5, rel=1e-10) and r2 == pytest.approx(1.0, abs=1e-10)
    groups = (x > 10).astype(int)
    target = -1.7 * model + np.where(groups == 1, 0.9, -0.4)
    k, r2 = co.scale_fit(target, model, groups)
    assert k == pytest.approx(-1.7, rel=1e-10) and r2 == pytest.approx(1.0, abs=1e-10)
    _, r2_one = co.scale_fit(target, model)  # one constant cannot absorb two offsets
    assert r2_one < 0.99


def test_scale_fit_not_measured_cases():
    k, r2 = co.scale_fit(np.arange(10.0), np.full(10, 3.0))
    assert math.isnan(k) and math.isnan(r2)
    k, r2 = co.scale_fit(np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.0, 1.5]))  # fewer than 4 samples
    assert math.isnan(k) and math.isnan(r2)


def test_harmonic_r2():
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    assert co.harmonic_r2(theta, 1 + 0.3 * np.sin(theta) - 0.2 * np.cos(2 * theta)) == pytest.approx(1.0, abs=1e-10)
    # sin 3 theta over the full circle is orthogonal to {1, sin, cos, sin 2, cos 2}
    assert co.harmonic_r2(theta, np.sin(3 * theta)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (30.0, 110.0, co.REGIME_TADPOLE_L4),
        (-110.0, -30.0, co.REGIME_TADPOLE_L5),
        (250.0, 330.0, co.REGIME_TADPOLE_L5),  # unwrapped above 180: still L5
        (20.0, 340.0, co.REGIME_HORSESHOE),
        (-25.0, 25.0, co.REGIME_QUASI_SATELLITE),
        (-30.0, 200.0, co.REGIME_COMPOUND),
    ],
)
def test_regime_of_one_piece(lo, hi, expected):
    t = np.linspace(0, 6 * np.pi, 3000)
    mid, amp = math.radians((lo + hi) / 2), math.radians((hi - lo) / 2)
    theta = mid + amp * np.sin(t)
    label, tmin, tmax = co.regime(theta, np.zeros(len(t), int))
    assert label == expected
    wrapped = np.degrees(np.angle(np.exp(1j * theta)))
    assert tmin == pytest.approx(wrapped.min(), abs=0.2)
    assert tmax == pytest.approx(wrapped.max(), abs=0.2)


def test_regime_pieces_of_different_kinds_are_compound():
    t = np.linspace(0, 6 * np.pi, 2000)
    l4 = math.radians(60) + math.radians(30) * np.sin(t)
    l5 = math.radians(-60) + math.radians(30) * np.sin(t)
    label, _, _ = co.regime(np.r_[l4, l5], np.r_[np.zeros(2000, int), np.ones(2000, int)])
    assert label == co.REGIME_COMPOUND


# ======================================================================================================================
# 5. Check values (docstring truth tables)
# ======================================================================================================================


def test_force_check_truth_table():
    full, fail = SETTINGS.mmr_force_r2_full, SETTINGS.mmr_force_r2_fail
    kmax = SETTINGS.mmr_coorbital_scale_ratio_max
    mid = 0.5 * (full + fail)
    assert co.force_check((0.99, 0.98), (1.0, 1.02, 0.97), SETTINGS) == (PASS, False)
    value, thin = co.force_check((0.99, mid), (1.0, 1.0, 1.0), SETTINGS)
    assert value == PASS and thin is True
    assert co.force_check((0.99, 0.5 * fail), (1.0, 1.0, 1.0), SETTINGS)[0] == FAIL
    assert co.force_check((0.99, 0.99), (1.2 * kmax, 1.0, 1.0), SETTINGS)[0] == FAIL
    assert co.force_check((0.99, 0.99), (1.0, 1.0, 0.8 / kmax), SETTINGS)[0] == FAIL
    assert co.force_check((0.99, 0.99), (-1.0, -1.0, -1.0), SETTINGS)[0] == FAIL  # the wrong sign
    assert co.force_check((0.99, float('nan')), (1.0, 1.0, 1.0), SETTINGS)[0] == NOT_MEASURED
    assert co.force_check((0.99, 0.99), (float('nan'), 1.0, 1.0), SETTINGS)[0] == NOT_MEASURED


def test_energy_check_truth_table():
    full, fail = SETTINGS.mmr_force_r2_full, SETTINGS.mmr_force_r2_fail
    kmax = SETTINGS.mmr_coorbital_scale_ratio_max
    assert co.energy_check(0.99, 1.0, SETTINGS) == PASS
    assert co.energy_check(0.5 * (full + fail), 1.0, SETTINGS) == PASS  # [fail, full) is not a failure
    assert co.energy_check(0.5 * fail, 1.0, SETTINGS) == FAIL
    assert co.energy_check(0.99, 1.5 * kmax, SETTINGS) == FAIL
    assert co.energy_check(0.99, -1.0, SETTINGS) == FAIL
    assert co.energy_check(float('nan'), float('nan'), SETTINGS) == NOT_MEASURED


@pytest.mark.parametrize(
    "force,energy,expected",
    [
        (PASS, PASS, PASS),
        (FAIL, PASS, PASS),
        (NOT_MEASURED, PASS, PASS),
        (PASS, FAIL, NOT_MEASURED),  # the local law holds, E is not conserved within the cycles
        (FAIL, FAIL, FAIL),
        (NOT_MEASURED, FAIL, FAIL),  # "fails and the force law does not pass"
    ],
)
def test_potential_check_truth_table(force, energy, expected):
    assert co.potential_check(force, energy) == expected


def test_needs_new_table_thresholds_on_vectors():
    tol_e, tol_i = 0.005, math.radians(0.5)
    planet = co.Orbit(e=0.0167, inc=0.0, omega=1.8, Omega=0.0)
    body = co.Orbit(e=0.1, inc=0.1, omega=0.0, Omega=0.0)
    near = co.Orbit(e=0.104, inc=0.1, omega=0.0, Omega=0.0)
    far = co.Orbit(e=0.106, inc=0.1, omega=0.0, Omega=0.0)
    assert co.needs_new_table((body, planet), (near, planet), tol_e, tol_i) is False
    assert co.needs_new_table((body, planet), (far, planet), tol_e, tol_i) is True
    # e -> 0: omega turned by pi moves the vector by only 2e = 0.002 < 0.005
    small_a = co.Orbit(e=0.001, inc=0.1, omega=0.0, Omega=0.0)
    small_b = co.Orbit(e=0.001, inc=0.1, omega=math.pi, Omega=0.0)
    assert co.needs_new_table((small_a, planet), (small_b, planet), tol_e, tol_i) is False
    # inclination vector: i = 1 deg, Omega turned by pi moves it by 2 sin 1 deg = 0.035 > sin 0.5 deg
    inc_a = co.Orbit(e=0.1, inc=math.radians(1.0), omega=0.0, Omega=0.0)
    inc_b = co.Orbit(e=0.1, inc=math.radians(1.0), omega=0.0, Omega=math.pi)
    assert co.needs_new_table((inc_a, planet), (inc_b, planet), tol_e, tol_i) is True
    # the planet's vector counts too
    planet_b = co.Orbit(e=0.0267, inc=0.0, omega=1.8, Omega=0.0)
    assert co.needs_new_table((body, planet), (body, planet_b), tol_e, tol_i) is True


def test_block_orbits_circular_mean_and_scalars():
    t = np.arange(100.0)
    omega = np.where(np.arange(100) % 2 == 0, math.radians(359.0), math.radians(1.0))
    starts, orbits = co.block_orbits(t, 0.1, 0.05, omega, 2.0, block_yr=10.0)
    assert len(starts) == len(orbits)
    assert 8 <= len(orbits) <= 11
    assert starts[0] == 0 and np.all(np.diff(starts) > 0)
    for orb in orbits:
        assert orb.e == pytest.approx(0.1) and orb.inc == pytest.approx(0.05) and orb.Omega == pytest.approx(2.0)
        assert abs(math.remainder(orb.omega, 2 * math.pi)) < math.radians(0.01)  # not 180 deg (a plain mean)


def test_coorbital_columns_empty_without_law():
    cols = co.coorbital_columns(None)
    assert set(cols) == set(co.COLUMN_FIELDS)
    for value in cols.values():
        assert value is None or value == '' or (isinstance(value, float) and math.isnan(value))


# ======================================================================================================================
# 6. The 1:1 law on synthetic averaged dynamics (direct call)
# ======================================================================================================================


def _coorbital_track(force_fn, theta0, span, step, mu=MU_EARTH, n_p=N_EARTH):
    """theta'' = -3 mu n_P^2 F~(theta) (V = 3 mu n^2 R~); a from n = n_P + theta_dot, n = 2 pi a^-3/2."""
    t = np.arange(0.0, span + step / 2, step)
    sol = solve_ivp(
        lambda _, y: [y[1], -3 * mu * n_p**2 * force_fn(y[0])],
        (0.0, t[-1]),
        [theta0, 0.0],
        t_eval=t,
        rtol=1e-11,
        atol=1e-13,
        max_step=step,
    )
    theta, theta_dot = sol.y
    theta_ddot = -3 * mu * n_p**2 * force_fn(theta)
    n = n_p + theta_dot
    a = (2 * np.pi / n) ** (2.0 / 3.0)
    a_dot = -(2.0 / 3.0) * a / n * theta_ddot  # da/dt = 2 mu n a F~ to first order (eq. 1 of the module)
    return t, theta, a, a_dot


def _law_input(t, theta, a, a_dot, e, varpi, cutoff_yr):
    nr = len(t)
    mask = np.ones(nr, bool)
    trim = t < t[0] + 2 * cutoff_yr
    mask[trim] = False
    mask[t > t[-1] - 2 * cutoff_yr] = False
    return co.CoorbitalInput(
        t_yr=t,
        angle=-theta,  # sigma of 1E-1 = lambda_E - lambda = -theta
        a=a,
        adot=a_dot,
        mask=mask,
        pieces=[(float(t[mask][0]), float(t[mask][-1]))],
        e=np.full(nr, e),
        inc=np.full(nr, 1e-4),
        omega=np.full(nr, varpi),
        Omega=np.zeros(nr),
        planet='Earth',
        planet_mass=MU_EARTH,
        planet_mean_motion_rad_yr=N_EARTH,
        body_coefficient=-1,
        cutoffs_yr=(cutoff_yr,),
    )


@lru_cache(maxsize=None)
def _qs_force_spline():
    """F~ of a planar e = 0.2 body against the J2000 Earth orbit (e 0.0167, varpi 102.94 deg), by brute force."""
    body = (0.2, 0.0, 0.3, 0.0)
    planet = (0.01671123, 0.0, math.radians(102.93768193), 0.0)
    eps = 0.1 * (MU_EARTH / 3) ** (1 / 3)
    th = np.linspace(-np.pi, np.pi, 721)
    _, force = _brute_average(body, planet, th[:-1], 720, eps)
    spline = CubicSpline(th, np.r_[force, force[0]], bc_type='periodic')
    return lambda x: spline(np.mod(x + np.pi, 2 * np.pi) - np.pi)


def test_coorbital_law_recovers_the_planet_force_on_a_tadpole():
    t, theta, a, a_dot = _coorbital_track(_f_circ, math.radians(30.0), 1500.0, 0.5)
    law = co.coorbital_law(_law_input(t, theta, a, a_dot, 1e-3, 0.5, 10.0), SETTINGS)
    assert law.regime == co.REGIME_TADPOLE_L4
    assert law.r2 > 0.99 and law.r2_h1 > 0.99 and law.r2_h2 > 0.99
    # truth K / K_theory = 1; the model's planet orbit is the J2000 Earth (e = 0.017), an O(e) effect at most
    assert law.scale == pytest.approx(1.0, abs=0.08)
    assert law.force_check == PASS
    # theta range of the truth: theta_min = 30 deg, theta_max from R~(theta_max) = R~(30 deg) (energy, theta_dot = 0)
    theta_max = math.degrees(brentq(lambda x: _r_circ(x) - _r_circ(math.radians(30.0)), math.radians(61), 3.0))
    assert law.theta_min_deg == pytest.approx(30.0, abs=0.5)
    assert law.theta_max_deg == pytest.approx(theta_max, abs=0.5)


def test_coorbital_law_wrong_sign_of_a_fails():
    t, theta, a, a_dot = _coorbital_track(_f_circ, math.radians(30.0), 1500.0, 0.5)
    law = co.coorbital_law(_law_input(t, theta, a, -a_dot, 1e-3, 0.5, 10.0), SETTINGS)
    assert law.scale == pytest.approx(-1.0, abs=0.1)
    assert law.force_check == FAIL


def test_coorbital_law_on_a_quasi_satellite():
    # QS of an e = 0.2 body: theta librates about 0 with |theta| < 17 deg, period ~ 66 yr (own ODE)
    force = _qs_force_spline()
    t, theta, a, a_dot = _coorbital_track(force, 0.3, 600.0, 0.1)
    assert np.max(np.abs(theta)) < math.radians(20.0)
    law = co.coorbital_law(_law_input(t, theta, a, a_dot, 0.2, 0.3, 2.0), SETTINGS)
    assert law.regime == co.REGIME_QUASI_SATELLITE
    assert law.r2 > 0.95
    assert law.scale == pytest.approx(1.0, abs=0.1)
    assert law.force_check == PASS


# ======================================================================================================================
# 7. analyse_mmr on synthetic co-orbitals (1E-1)
# ======================================================================================================================


@lru_cache(maxsize=None)
def _coorbital_series(theta0_deg: float, span: float, step: float = 1.0):
    t, theta, a, _ = _coorbital_track(_f_circ, math.radians(theta0_deg), span, step)
    nr = len(t)
    varpi = np.full(nr, 1.0)
    lam_p = N_EARTH * t
    lam = lam_p + theta
    series = MMRSeries(
        t_yr=t,
        a=a,
        e=np.full(nr, 1e-3),
        inc=np.full(nr, 1e-3),
        Omega=np.zeros(nr),
        omega=varpi.copy(),
        varpi=varpi,
        angle_wrapped=np.mod(lam_p - lam, 2 * np.pi),
        resonance=ResonanceSpec.from_key('1E-1'),
        regime=Regime(step_yr=step, span_yr=float(t[-1]), n_rows=nr),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name=f'coorbital-{theta0_deg}',
    )
    return series, theta


@lru_cache(maxsize=None)
def _coorbital_result(theta0_deg: float, span: float):
    series, theta = _coorbital_series(theta0_deg, span)
    return _run(series), theta


@pytest.mark.parametrize(
    "theta0_deg,regime",
    [(25.0, co.REGIME_TADPOLE_L4), (-25.0, co.REGIME_TADPOLE_L5), (20.0, co.REGIME_HORSESHOE)],
)
def test_coorbital_libration_is_resonant_and_dynamic(theta0_deg, regime):
    result, _ = _coorbital_result(theta0_deg, 3000.0)
    v = result.verdict
    assert _s(v.status) == 'resonant'
    assert _s(v.angle_status) == 'libration'
    assert _s(v.libration_origin) == 'dynamic'
    law = result.response.coorbital
    assert law is not None and law.planet == 'Earth'
    assert law.regime == regime
    assert law.scale == pytest.approx(1.0, abs=0.08)
    assert law.force_check == PASS
    assert v.checks['check_mmr_a_response'] == PASS


def test_horseshoe_is_not_harmonic_but_follows_the_coorbital_law():
    result, theta = _coorbital_result(20.0, 3000.0)
    assert np.degrees(theta.min()) == pytest.approx(20.0, abs=1e-6)
    # symmetric about 180 deg: R~ is even about pi in the circular problem
    assert np.degrees(theta.max()) == pytest.approx(340.0, abs=0.1)
    law = result.response.coorbital
    assert law.r2 > 0.99
    assert law.harmonic_r2 < 0.9  # the k <= 2 law cannot follow the near-encounter peaks


def test_coorbital_tadpole_theta_range_from_the_energy_integral():
    result, _ = _coorbital_result(25.0, 3000.0)
    law = result.response.coorbital
    # theta_dot = 0 at both ends: R~(theta_max) = R~(25 deg)
    theta_max = math.degrees(brentq(lambda x: _r_circ(x) - _r_circ(math.radians(25.0)), math.radians(61), 3.0))
    assert law.theta_min_deg == pytest.approx(25.0, abs=0.5)
    assert law.theta_max_deg == pytest.approx(theta_max, abs=0.5)


def test_small_tadpole_period_is_the_linear_one():
    # P = P_orb / sqrt(27 mu / 4) = (2 pi / n_E) / sqrt(27 * 3.0035e-6 / 4) = 222.1 yr (M&D eq. 3.191 region)
    result, _ = _coorbital_result(56.0, 2500.0)
    p_linear = (2 * np.pi / N_EARTH) / math.sqrt(27 * MU_EARTH / 4)
    assert p_linear == pytest.approx(222.1, abs=0.2)
    assert result.geometry.period_cycle_yr == pytest.approx(p_linear, rel=0.03)
    assert _s(result.verdict.angle_status) == 'libration'


@pytest.mark.parametrize("theta0_deg", [35.0, 45.0])
def test_moderate_tadpole_a_response_is_measured(theta0_deg):
    # The law itself passes (R^2 ~ 1 in both halves, K/K_theory ~ 1), so check_mmr_a_response must be pass, as for
    # the wider tadpole of 25 deg (force_check is "check_mmr_a_response of a co-orbital", coorbital.py).
    result, _ = _coorbital_result(theta0_deg, 3000.0)
    law = result.response.coorbital
    assert law.force_check == PASS
    assert result.verdict.checks['check_mmr_a_response'] == PASS


# ======================================================================================================================
# 8. Pendulum 2J-1 records: episodes before a jump of a, near-episodes
# ======================================================================================================================

W0_SLOW = 2 * np.pi / 600.0  # small-amplitude libration period 600 yr (> MMR_LIBRATION_PERIOD_MIN_YR = 150)


def _mmr_series(t, sigma, sigma_dot, name):
    """2J-1: sigma = 2 lambda_J - lambda - varpi, varpi constant, sigma_dot = 2 n_J - n(a)."""
    n = 2 * N_JUPITER - sigma_dot
    a = (2 * np.pi / n) ** (2.0 / 3.0)
    nr = len(t)
    varpi = np.full(nr, 0.7)
    lam = 2 * N_JUPITER * t - sigma - varpi
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full(nr, 0.15),
        inc=np.full(nr, 0.05),
        Omega=np.zeros(nr),
        omega=varpi.copy(),
        varpi=varpi,
        angle_wrapped=np.mod(sigma, 2 * np.pi),
        resonance=ResonanceSpec.from_key('2J-1'),
        regime=Regime(step_yr=float(t[1] - t[0]), span_yr=float(t[-1] - t[0]), n_rows=nr),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name=name,
    )


def _pendulum_with_kick(span, step, sigma0, kick_time, kick, w0=W0_SLOW):
    """sigma'' = -w0^2 sin sigma from (sigma0, 0); sigma_dot += kick at kick_time (a jumps)."""
    t = np.arange(0.0, span + step / 2, step)
    rhs = lambda _, y: [y[1], -(w0**2) * math.sin(y[0])]  # noqa: E731
    before = t < kick_time
    s1 = solve_ivp(rhs, (0.0, kick_time), [sigma0, 0.0], dense_output=True, rtol=1e-11, atol=1e-12).sol
    y_kick = s1(kick_time)
    y_kick[1] += kick
    s2 = solve_ivp(rhs, (kick_time, t[-1]), y_kick, dense_output=True, rtol=1e-11, atol=1e-12).sol
    y = np.where(before, s1(t), s2(t))
    return t, y[0], y[1]


@lru_cache(maxsize=None)
def _kick_result(kick_w0: float):
    # 0-2600 yr: libration of half-amplitude 90 deg (period 1.18 x 600 = 708 yr: 7 TPs, three full cycles);
    # then sigma_dot += kick: circulation (sigma_dot > 2 w0 at sigma = 0 needs kick > 0.59 w0)
    t, sigma, sigma_dot = _pendulum_with_kick(5000.0, 0.5, math.pi / 2, 2600.0, kick_w0 * W0_SLOW)
    return _run(_mmr_series(t, sigma, sigma_dot, f'kick-{kick_w0}')), t, sigma


@pytest.mark.parametrize("kick_w0", [3.0, 5.0, 8.0, 12.0])
def test_librating_episode_before_a_jump_makes_the_status_transient(kick_w0):
    """W4 P0 (2021 BA, 524522): accepted episodes anywhere in the record -> status >= transient. W6 C1 (author,
    2026-10-16): angle_status describes the whole record (episodic); the last regime, circulation after the kick,
    is `final_regime`: from the last turning point (G1c, at most half a libration period before the kick) or from
    the jump of a (G1e, the kick itself, found on the filtered a a few years late)."""
    result, t, sigma = _kick_result(kick_w0)
    assert _n_turning_points(sigma[t < 2600.0]) >= 7  # truth: three full cycles before the jump
    assert _n_turning_points(sigma[t > 2600.0]) == 0  # truth: circulation after it
    v = result.verdict
    assert _s(v.angle_status) == 'episodic'
    assert _s(v.final_regime) == 'circulation'
    half_period = np.pi / W0_SLOW  # 300 yr
    assert v.final_regime_start_yr is not None and abs(v.final_regime_start_yr - 2600.0) <= half_period
    assert _s(v.status) in ('transient', 'resonant'), (v.status, v.reason, v.flags)


def test_libration_through_a_kick_that_keeps_it_librating_stays_resonant():
    # half-amplitude 45 deg (sigma_dot max = 2 sin 22.5 w0 = 0.77 w0) kicked by 1.2 w0 at a moment of sigma_dot ~ 0
    # stays below the separatrix (needs 2 w0 at sigma = 0): libration over the whole record, 6+ TPs after the kick.
    t, sigma, sigma_dot = _pendulum_with_kick(6000.0, 0.5, math.pi / 4, 3000.0, 1.2 * W0_SLOW)
    assert _n_turning_points(sigma[t > 3000.0]) >= 6
    assert np.max(np.abs(sigma)) < np.pi
    v = _run(_mmr_series(t, sigma, sigma_dot, 'lib-kick-lib')).verdict
    assert _s(v.status) == 'resonant'
    assert _s(v.angle_status) == 'libration'


def _cycle_inside(w0=W0_SLOW, step=0.5, sd_circ=2.2, sd_lib=1.6, n_rev=3):
    """One libration cycle between two circulations, the kicks a quarter-turn away from the centre: the circulation
    (sd_circ w0 at sigma = 0) is kicked at sigma = -90 deg moving up to the libration of sd_lib w0 at sigma = 0
    (sqrt(sd_lib^2 - 2) w0 = 0.75 w0 there), which runs up through 0 (t_a), to its maximum, down through 0, to its
    minimum, up through 0 again (t_b) and is kicked back to the circulation's sqrt(sd_circ^2 - 2) w0 at +90 deg. t_a and
    t_b are one period of the 106-deg libration apart, the state at both the same (W20, closure by phase return)."""
    rhs = lambda _, y: [y[1], -(w0**2) * math.sin(y[0])]  # noqa: E731
    horizon = 50 * 2 * np.pi / w0
    base = 2 * np.pi * n_rev

    def until(y0, t0, target, count):
        ev = lambda _, y: y[0] - target  # noqa: E731
        ev.direction = 1
        sol = solve_ivp(rhs, (t0, t0 + horizon), y0, events=ev, dense_output=True, rtol=1e-11, atol=1e-12)
        times = [x for x in sol.t_events[0] if x > t0 + 1e-6]
        return sol.sol, times[count - 1]

    s1, t1 = until([0.0, sd_circ * w0], 0.0, base - np.pi / 2, 1)
    s2, t2 = until([base - np.pi / 2, math.sqrt(sd_lib**2 - 2) * w0], t1, base + np.pi / 2, 2)
    t3 = t2 + t1
    s3 = solve_ivp(rhs, (t2, t3), [base + np.pi / 2, math.sqrt(sd_circ**2 - 2) * w0], dense_output=True, rtol=1e-11, atol=1e-12).sol
    t = np.arange(0.0, t3, step)
    y = np.where(t < t1, s1(t), np.where(t < t2, s2(t), s3(t)))
    return t, y[0], y[1], (t1, t2)


def _capture(n_tp, w0=W0_SLOW, step=0.5, sd_circ=2.2, sd_lib=1.6, n_rev=3, open_exit=False):
    """Circulation (sigma_dot = sd_circ w0 at sigma = 0, just above the separatrix 2 w0) for n_rev turns; at
    sigma = 0 (mod 2 pi) a kick to sd_lib w0 (libration, half-amplitude arccos(1 - sd_lib^2/2) = 106 deg); after
    exactly n_tp turning points, at the next sigma = 0, a kick back to circulation. Small kicks (0.6 w0).

    With an even n_tp the capture then spans exactly one full cycle per two TPs (sigma = 0 upwards to sigma = 0
    upwards, the same rate): a cycle closed by its phase return (W20). `open_exit` (even n_tp): the kick back comes a
    quarter-turn earlier, at sigma = -90 deg moving up, to sigma_dot = sd_circ w0 there (energy conservation then gives
    sqrt(sd_circ^2 + 2) w0 = 2.62 w0 at sigma = 0, against 1.6 w0 at the entry): the state never returns, the
    two TPs are half a cycle of capture."""
    rhs = lambda _, y: [y[1], -(w0**2) * math.sin(y[0])]  # noqa: E731
    horizon = 50 * 2 * np.pi / w0

    def until(y0, t0, target, count, direction):
        ev = lambda _, y: y[0] - target  # noqa: E731
        ev.direction = direction
        sol = solve_ivp(rhs, (t0, t0 + horizon), y0, events=ev, dense_output=True, rtol=1e-11, atol=1e-12)
        times = [x for x in sol.t_events[0] if x > t0 + 1e-6]
        return sol.sol, times[count - 1]

    s1, t1 = until([0.0, sd_circ * w0], 0.0, 2 * np.pi * n_rev, 1, 1)
    base = 2 * np.pi * n_rev
    if n_tp % 2 == 0 and open_exit:  # max, min, ... kicked out at -90 deg moving up
        s2, t2 = until([base, sd_lib * w0], t1, base - np.pi / 2, n_tp // 2, 1)
        out = sd_circ * w0 - s2(t2)[1]
    elif n_tp % 2 == 0:  # max, min, ... ends moving up through the centre
        s2, t2 = until([base, sd_lib * w0], t1, base, n_tp // 2, 1)
        out = (sd_circ - sd_lib) * w0
    else:  # ends moving down
        s2, t2 = until([base, sd_lib * w0], t1, base, (n_tp + 1) // 2, -1)
        out = -(sd_circ - sd_lib) * w0
    y2 = s2(t2)
    y2[1] += out
    t3 = t2 + t1  # as much circulation after the capture as before it
    s3 = solve_ivp(rhs, (t2, t3), y2, dense_output=True, rtol=1e-11, atol=1e-12).sol
    t = np.arange(0.0, t3, step)
    y = np.where(t < t1, s1(t), np.where(t < t2, s2(t), s3(t)))
    return t, y[0], y[1], (t1, t2)


@lru_cache(maxsize=None)
def _capture_result(n_tp: int, w0: float = W0_SLOW, deficit: int = None, open_exit: bool = False):
    t, sigma, sigma_dot, window = _capture(n_tp, w0=w0, step=0.5 if w0 == W0_SLOW else 0.25, open_exit=open_exit)
    settings = SETTINGS if deficit is None else AnalysisSettings(libration_near_episode_tp_deficit=deficit)
    return _run(_mmr_series(t, sigma, sigma_dot, f'capture-{n_tp}-{open_exit}'), settings), t, sigma, window


def _confidence_at_most_medium(v):
    return _s(v.confidence) in ('medium', 'low')


def test_a_two_turning_point_capture_that_returns_is_one_cycle():
    """2016 CA138-like (W20, closure by phase return): one libration cycle between two circulations, seen through
    2 TPs (`_cycle_inside`): an accepted episode, transient, cap one_return_cycle (<= medium); no near-episode. The
    crossings of sigma = 0 that close it lie inside the libration, a full period of the 106-deg libration apart (by
    the pendulum, 1.35 x 600 yr; the time from the entry kick at -90 deg to the exit kick at +90 deg is that period
    plus the two quarter-turn legs, so t_b - t_a < t2 - t1)."""
    t, sigma, sigma_dot, (t1, t2) = _cycle_inside()
    result = _run(_mmr_series(t, sigma, sigma_dot, 'cycle-inside'))
    assert _n_turning_points(sigma) == 2
    v = result.verdict
    assert _s(v.status) == 'transient' and _s(v.angle_status) == 'episodic'
    assert 'near_episode' not in v.flags and 'one_return_cycle' in v.caps
    assert _confidence_at_most_medium(v)
    (episode,) = [run for run in result.geometry.runs if run.accepted]
    assert episode.closed_by_return
    amplitude = math.acos(1 - 1.6**2 / 2)  # 106.3 deg: sigma_dot = 1.6 w0 at the bottom
    period = 4 * ellipk(math.sin(amplitude / 2) ** 2) / W0_SLOW  # T = 4 K(k) / w0, k = sin(A / 2)
    assert episode.return_t_b_yr - episode.return_t_a_yr == pytest.approx(period, rel=0.02)
    assert t1 < episode.return_t_a_yr and episode.return_t_b_yr < t2


def test_near_episode_two_turning_points_flag_and_cap():
    """One half-cycle of capture (2 TPs, threshold 3, the state never returns): no libration, flag near_episode,
    <= medium."""
    result, t, sigma, (t1, t2) = _capture_result(2, open_exit=True)
    assert _n_turning_points(sigma) == 2
    assert _n_turning_points(sigma[(t > t1) & (t < t2)]) == 2
    v = result.verdict
    assert _s(v.status) == 'non_resonant'
    assert 'near_episode' in v.flags
    assert _confidence_at_most_medium(v)
    assert result.extra_columns.get('geo_near_episode_n_tp') == 2


def test_capture_with_three_turning_points_is_transient_without_near_episode():
    result, _, sigma, _ = _capture_result(3)
    assert _n_turning_points(sigma) == 3
    v = result.verdict
    assert _s(v.status) == 'transient'
    assert 'near_episode' not in v.flags


def test_near_episode_off_by_setting():
    result, _, sigma, _ = _capture_result(2, deficit=0, open_exit=True)
    assert _n_turning_points(sigma) == 2
    v = result.verdict
    assert _s(v.status) == 'non_resonant'
    assert 'near_episode' not in v.flags


def test_pure_circulation_has_no_near_episode_and_keeps_its_confidence():
    t = np.arange(0.0, 4000.0, 0.5)
    sol = solve_ivp(
        lambda _, y: [y[1], -(W0_SLOW**2) * math.sin(y[0])],
        (0.0, t[-1]),
        [0.0, 2.6 * W0_SLOW],
        t_eval=t,
        rtol=1e-11,
        atol=1e-12,
    )
    assert _n_turning_points(sol.y[0]) == 0
    v = _run(_mmr_series(t, sol.y[0], sol.y[1], 'circulation')).verdict
    assert _s(v.status) == 'non_resonant'
    assert _s(v.angle_status) == 'circulation'
    assert 'near_episode' not in v.flags
    assert _s(v.confidence) in ('high', 'certain')


def test_fast_back_and_forth_below_the_period_floor_is_not_a_near_episode():
    # w0 = 2 pi / 80 yr: the 106-deg capture has a full period ~ 1.35 x 80 = 108 yr < 150 yr (the shortest
    # libration of a non-co-orbital MMR): two TPs of a forced wave, not half a libration (2012 XE133 rule).
    w0 = 2 * np.pi / 80.0
    result, _, sigma, (t1, t2) = _capture_result(2, w0=w0)
    assert _n_turning_points(sigma) == 2
    assert t2 - t1 < SETTINGS.mmr_libration_period_min_yr  # t1..t2 is one full libration period here
    v = result.verdict
    assert 'near_episode' not in v.flags
