"""Independent checks of three W5 fixes (written from API.md, QUESTIONS-2026-10-16 item 6 and W4-SLOW-RESULTS KT19 only).

(a) The retrograde 1/-1 averaged potential of `coorbital.averaged_table`, against a quadrature written here from
    scratch (own Kepler solver, own rotation, own placement M = sigma + lambda' - ((2k - 1) omega + Omega)):
    closed-form limits (prograde planar circular: M&D eqs. 3.178-3.179; retrograde planar circular: a constant given by
    the complete elliptic integral K), the reflection symmetry of the planar retrograde problem, and the sign of
    da/dt = (2 / (n a)) d<R>/d sigma (Lagrange) cross-checked with the energy route da/dt = (2 a^2 / GM) v . f.
(b) A slow libration (period 40 kyr, amplitude 1 rad) with a fast modulation (3.5 kyr), the KT19 pattern, must not be
    read as circulation or drift by `analyse_mmr`.
(c) A resample recommendation is never coarser than the record's own step.

All fast: no integration.
"""

import math

import numpy as np
import pytest
from scipy.special import ellipk

from resonances.resonance.analysis.mmr import coorbital as co
from resonances.resonance.analysis.mmr import response
from resonances.resonance.analysis.mmr import sampling
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings

TWO_PI = 2 * np.pi
MU_JUPITER = 9.5479e-4  # m_J / M_sun
N_NEPTUNE = np.radians(2.184902)  # rad / yr
N_JUPITER = np.radians(30.349646)


# --------------------------------------------------------------------------------------------------------------
# (a) my own quadrature of the averaged direct + indirect disturbing function (a = a_P = 1, GM_sun = 1, n = 1)
# --------------------------------------------------------------------------------------------------------------


def _rotation(inc, omega, Omega):
    """R_z(Omega) R_x(inc) R_z(omega): orbital plane -> ecliptic frame."""

    def rz(x):
        return np.array([[np.cos(x), -np.sin(x), 0.0], [np.sin(x), np.cos(x), 0.0], [0.0, 0.0, 1.0]])

    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(inc), -np.sin(inc)], [0.0, np.sin(inc), np.cos(inc)]])
    return rz(Omega) @ rx @ rz(omega)


def _state(mean_anomaly, e, inc, omega, Omega):
    """Position and velocity (a = 1, GM = 1, so v = dr/dM) of a Keplerian orbit, shape (..., 3)."""
    m = np.mod(mean_anomaly, TWO_PI)
    big_e = m + e * np.sin(m)
    for _ in range(60):
        big_e = big_e - (big_e - e * np.sin(big_e) - m) / (1 - e * np.cos(big_e))
    f = np.arctan2(np.sqrt(1 - e * e) * np.sin(big_e), np.cos(big_e) - e)
    r = 1 - e * np.cos(big_e)
    p = 1 - e * e
    pos = np.stack([r * np.cos(f), r * np.sin(f), np.zeros_like(f)], axis=-1)
    vel = np.stack([-np.sin(f), e + np.cos(f), np.zeros_like(f)], axis=-1) / np.sqrt(p)
    rot = _rotation(inc, omega, Omega)
    return pos @ rot.T, vel @ rot.T


def _my_average(sigma, e, inc, omega, Omega, c, eps, n_lambda=4096):
    """(R~, F~, min distance) at each sigma: planet circular planar at lambda', body at M = sigma + lambda' - (c omega + Omega).

    R = 1/sqrt(|r_P - r|^2 + eps^2) - r . r_P; F = grad_r R . v (energy route: da/dt = 2 a^2 / GM v . f)."""
    lam = TWO_PI * np.arange(n_lambda) / n_lambda
    sig = np.asarray(sigma, dtype=float)[:, None]
    mean_anomaly = sig + lam[None, :] - (c * omega + Omega)
    r, v = _state(mean_anomaly, e, inc, omega, Omega)
    rp = np.stack([np.cos(lam), np.sin(lam), np.zeros_like(lam)], axis=-1)[None, :, :]
    d = rp - r
    d2 = np.sum(d * d, axis=-1)
    soft = d2 + eps * eps
    potential = 1 / np.sqrt(soft) - np.sum(r * rp, axis=-1)
    grad = d / soft[..., None] ** 1.5 - rp
    force = np.sum(grad * v, axis=-1)
    return potential.mean(axis=1), force.mean(axis=1), float(np.sqrt(d2.min()))


def test_hill_softening_closed_form():
    # eps / a_P = factor x (mu / 3)^(1/3): Jupiter 0.1 r_H = 0.1 x 0.06828 = 0.006828
    assert co.hill_softening(MU_JUPITER, 0.1) == pytest.approx(0.1 * (MU_JUPITER / 3) ** (1 / 3), rel=1e-12)
    assert co.hill_softening(MU_JUPITER, 0.1) == pytest.approx(0.006828, rel=1e-3)


def test_prograde_planar_circular_matches_murray_dermott():
    """e = i = 0: R~ = 1/sqrt(4 sin^2(theta/2) + eps^2) - cos theta, F~ = sin theta (1 - X^(-3/2))."""
    eps = 0.01
    table = co.averaged_table(co.Orbit(0.0, 0.0, 0.0, 0.0), co.Orbit(0.0, 0.0, 0.0, 0.0), 360, eps)
    theta = table.theta_rad
    x = 4 * np.sin(theta / 2) ** 2 + eps**2
    np.testing.assert_allclose(table.potential, x**-0.5 - np.cos(theta), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(table.force, np.sin(theta) * (1 - x**-1.5), rtol=1e-9, atol=1e-9)
    # Lagrange: da/dt has the sign of F~. A body just ahead of the planet (0 < theta << 60 deg) is pulled back: a falls.
    ahead = (theta > np.radians(2)) & (theta < np.radians(50))
    assert np.all(table.force[ahead] < 0)
    # and between L4 (60 deg) and L3 (180 deg) the force pushes the body out (a grows): the tadpole about 60 deg
    beyond = (theta > np.radians(70)) & (theta < np.radians(170))
    assert np.all(table.force[beyond] > 0)


@pytest.mark.parametrize('c', [1, -1, 3])
@pytest.mark.parametrize('eps, n_grid', [(0.3, 360), (0.05, 720)])
def test_retrograde_planar_circular_is_flat(c, eps, n_grid):
    """i = 180 deg, e = 0: the separation runs over the whole circle for any sigma, so
    <R> = (1/2pi) int dphi / sqrt(4 sin^2(phi/2) + eps^2) = 2 K(m) / (pi sqrt(4 + eps^2)), m = 4 / (4 + eps^2),
    the indirect term averages to zero, and F~ = 0 (no 1/-1 resonance at e = 0 in the plane, Morais & Namouni 2013)."""
    body = co.Orbit(0.0, np.pi, 0.7, 1.9, omega_coefficient=c)
    table = co.averaged_table(body, co.Orbit(0.0, 0.0, 0.0, 0.0), n_grid, eps)
    expected = 2 * ellipk(4 / (4 + eps**2)) / (np.pi * np.sqrt(4 + eps**2))
    np.testing.assert_allclose(table.potential, expected, rtol=1e-6)
    assert np.max(np.abs(table.force)) < 1e-5  # trapezoid error ~ exp(-2 pi eps / h) / eps^2 for eps = 0.05


@pytest.mark.parametrize(
    'e, inc_deg, omega, Omega, k_prime',
    [
        (0.38, 163.0, 1.0, 2.0, 1),  # 2015 BZ509-like, 1J-1r1
        (0.38, 163.0, 1.0, 2.0, 0),  # same orbit, 1J-1r0 (lambda* - lambda' - 2 varpi*)
        (0.25, 125.0, 2.5, 0.3, 1),
        (0.25, 125.0, 2.5, 0.3, 2),
    ],
)
def test_retrograde_table_matches_independent_quadrature(e, inc_deg, omega, Omega, k_prime):
    c = 2 * k_prime - 1
    eps = 0.1 * (MU_JUPITER / 3) ** (1 / 3)
    body = co.Orbit(e, np.radians(inc_deg), omega, Omega, omega_coefficient=c)
    table = co.averaged_table(body, co.Orbit(0.0, 0.0, 0.0, 0.0), 720, eps)
    sigma = table.theta_rad[::8]
    pot, force, d_min = _my_average(sigma, e, np.radians(inc_deg), omega, Omega, c, eps)
    assert d_min > 0.03  # precondition: no near-encounter (>> eps = 0.0068), both quadratures converge
    scale_r = np.max(np.abs(pot - pot.mean()))
    scale_f = np.max(np.abs(force))
    assert scale_f > 1e-3  # the angle matters: a non-trivial test
    np.testing.assert_allclose(table.potential[::8], pot, atol=2e-5 * max(scale_r, 1e-3))
    # 720-point trapezoid of the table: ~1e-7 of the scale away from the closest approach, a few 1e-6 at the sharp peak
    np.testing.assert_allclose(table.force[::8], force, atol=2e-5 * scale_f)

    # Lagrange: F~ = dR~/dsigma (central difference of my own R~) agrees with the energy route v . f
    h = 1e-4
    up, _, _ = _my_average(sigma + h, e, np.radians(inc_deg), omega, Omega, c, eps)
    down, _, _ = _my_average(sigma - h, e, np.radians(inc_deg), omega, Omega, c, eps)
    derivative = (up - down) / (2 * h)
    np.testing.assert_allclose(derivative, force, atol=1e-5 * scale_f)
    # so da/dt = 2 mu n a F~ has the sign of dR~/dsigma in the table too
    strong = np.abs(derivative) > 0.05 * scale_f
    assert np.all(np.sign(table.force[::8][strong]) == np.sign(derivative[strong]))


@pytest.mark.parametrize('c, omega_deg', [(1, 10.0), (-1, 10.0), (3, 5.0)])
def test_retrograde_planar_reflection_symmetry(c, omega_deg):
    """i = 180 deg: the separation is sigma - (1 + c) omega - (f + M), so R~ is even and F~ odd about
    sigma_0 = (1 + c) omega (M -> -M); F~ vanishes at sigma_0 and sigma_0 + pi."""
    n_grid = 360
    omega = np.radians(omega_deg)
    body = co.Orbit(0.3, np.pi, omega, np.radians(37.0), omega_coefficient=c)
    table = co.averaged_table(body, co.Orbit(0.0, 0.0, 0.0, 0.0), n_grid, 0.3)
    shift = int(round((1 + c) * omega_deg))  # grid of 1 deg
    j = np.arange(n_grid)
    plus, minus = (shift + j) % n_grid, (shift - j) % n_grid
    assert np.max(np.abs(table.force)) > 1e-3
    np.testing.assert_allclose(table.potential[plus], table.potential[minus], atol=1e-9)
    np.testing.assert_allclose(table.force[plus], -table.force[minus], atol=1e-9)
    assert abs(table.force[shift % n_grid]) < 1e-9
    assert abs(table.force[(shift + n_grid // 2) % n_grid]) < 1e-9


@pytest.mark.parametrize('k_prime', [0, 1, 2])
def test_retrograde_one_to_one_conventions(k_prime):
    """1J-1r<k>: sigma = lambda* - lambda' - (2 - 2k) varpi* + 2k Omega = M + (2k - 1) omega + Omega - lambda'
    (lambda* = M + omega - Omega, varpi* = omega - Omega); the law applies (planet Jupiter), Orbit.phase matches."""
    spec = ResonanceSpec.from_key(f'1J-1r{k_prime}')
    assert spec.retrograde
    rng = np.random.default_rng(k_prime)
    big_m, omega, Omega, lam_j = rng.uniform(0, TWO_PI, 4)
    lam_star, varpi_star = big_m + omega - Omega, omega - Omega
    from_spec = (
        spec.body_coefficient * lam_star
        + spec.body_varpi_coefficient * varpi_star
        + spec.body_node_coefficient * Omega
        + spec.planet_coefficients[0] * lam_j
    )
    expected = big_m + (2 * k_prime - 1) * omega + Omega - lam_j
    assert math.remainder(from_spec - expected, TWO_PI) == pytest.approx(0.0, abs=1e-9)
    orbit = co.Orbit(0.3, 2.8, omega, Omega, omega_coefficient=spec.body_node_coefficient - 1)
    assert math.remainder(orbit.phase - ((2 * k_prime - 1) * omega + Omega), TWO_PI) == pytest.approx(0.0, abs=1e-12)
    assert response.law_planet(spec) == 'Jupiter'
    t = np.arange(10.0)
    _, orbits = co.block_orbits(t, 0.3, 2.8, omega, Omega, 5.0, omega_coefficient=2 * k_prime - 1)
    assert all(o.omega_coefficient == 2 * k_prime - 1 for o in orbits)


def test_other_retrograde_angles_get_no_coorbital_law():
    assert response.law_planet(ResonanceSpec.from_key('1J-3r0')) is None
    assert response.law_planet(ResonanceSpec.from_key('7N-9r6')) is None


# --------------------------------------------------------------------------------------------------------------
# (b) slow libration + fast modulation (KT19 pattern)
# --------------------------------------------------------------------------------------------------------------

STEP_YR, SPAN_YR = 50.0, 200000.0
P_SLOW, P_FAST = 40000.0, 3500.0


def _synthetic(key, slow_amp, fast_amp, phase):
    """A self-consistent record: sigma = pi + A sin(2pi t/40 kyr) + a_f sin(2pi t/3.5 kyr + phase), a from the identity
    sigma_dot = sum k_j n_j + k n(a) (constant varpi, Omega, omega), wrapped angle, lambda or M consistent with sigma."""
    spec = ResonanceSpec.from_key(key)
    t = np.arange(0.0, SPAN_YR + STEP_YR / 2, STEP_YR)
    ws, wf = TWO_PI / P_SLOW, TWO_PI / P_FAST
    sig = np.pi + slow_amp * np.sin(ws * t) + fast_amp * np.sin(wf * t + phase)
    sdot = slow_amp * ws * np.cos(ws * t) + fast_amp * wf * np.cos(wf * t + phase)
    k, kp = spec.body_coefficient, spec.planet_coefficients[0]
    n = (sdot - kp * N_NEPTUNE) / k
    a = (TWO_PI / n) ** (2 / 3)
    omega, Omega = np.full_like(t, 0.8), np.full_like(t, 2.1)
    extra = {}
    if spec.retrograde:
        lam_star = (sig - kp * N_NEPTUNE * t - spec.body_varpi_coefficient * (omega - Omega) - spec.body_node_coefficient * Omega) / k
        extra['mean_anomaly'] = np.mod(lam_star - omega + Omega, TWO_PI)
        e, inc = 0.33, np.radians(110.0)
    else:
        lam = (sig - kp * N_NEPTUNE * t - spec.body_varpi_coefficient * (omega + Omega)) / k
        extra['mean_longitude'] = np.mod(lam, TWO_PI)
        e, inc = 0.05, 0.05
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full_like(t, e),
        inc=np.full_like(t, inc),
        Omega=Omega,
        omega=omega,
        varpi=np.mod(omega + Omega, TWO_PI),
        angle_wrapped=np.mod(sig, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=STEP_YR, span_yr=SPAN_YR, n_rows=len(t)),
        name='synthetic',
        **extra,
    )


@pytest.mark.parametrize('phase', [0.0, 1.3, 2.5])
@pytest.mark.parametrize('fast_amp', [0.05, 0.15, 0.3])
@pytest.mark.parametrize('key', ['3N-4+0+1', '7N-9r6'])
def test_slow_libration_with_fast_modulation_is_not_circulation_or_drift(key, fast_amp, phase):
    series = _synthetic(key, 1.0, fast_amp, phase)
    assert np.ptp(np.unwrap(series.angle_wrapped)) < TWO_PI - 1  # bounded by construction: never a full turn
    verdict = analyse_mmr(series, AnalysisSettings(), scan_subresonances=False, scan_owner=False).verdict
    assert verdict.angle_status not in ('circulation', 'drift'), (verdict.angle_status, verdict.angle_subtype, verdict.reason)


@pytest.mark.parametrize('phase', [0.0, 1.3, 2.5])
@pytest.mark.parametrize('fast_amp', [0.05, 0.15, 0.3])
@pytest.mark.parametrize('key', ['3N-4+0+1', '7N-9r6'])
def test_kt19_amplitude_slow_libration_is_not_circulation_or_drift(key, fast_amp, phase):
    """KT19's own amplitude: sigma in 61-305 deg, i.e. about 2.1 rad about 180 deg (W4-SLOW-RESULTS item 3)."""
    series = _synthetic(key, 2.1, fast_amp, phase)
    assert np.ptp(np.unwrap(series.angle_wrapped)) < TWO_PI - 1
    verdict = analyse_mmr(series, AnalysisSettings(), scan_subresonances=False, scan_owner=False).verdict
    assert verdict.angle_status not in ('circulation', 'drift'), (verdict.angle_status, verdict.angle_subtype, verdict.reason)


# --------------------------------------------------------------------------------------------------------------
# (c) a resample recommendation is never coarser than the record's step
# --------------------------------------------------------------------------------------------------------------

NAN = float('nan')


@pytest.mark.parametrize(
    'plain, pred, period, step, expected',
    [
        (0.1, 1e16, 1e4, 0.2, 0.1),  # 99942-like: aliased record at 0.2 yr; the plain bound rules (not 10 yr)
        (NAN, NAN, 100.0, 15.9, 5.0),  # P_cyc / 20 = 5
        (7.854, 1e16, NAN, 15.9, 5.0),  # 0.5 pi / 0.2 rad/yr
        (30.0, 1e5, 1e4, 15.9, None),  # the step already meets every bound
        (NAN, NAN, NAN, 1.0, None),  # nothing known
        (0.01, 1e3, 1e4, 1.0, None),  # no round value fits below 0.05 yr
        (3.0, 0.7, 1e4, 2.0, 0.5),
    ],
)
def test_resample_step_examples(plain, pred, period, step, expected):
    assert sampling.resample_step(plain, pred, period, step, AnalysisSettings()) == expected


def test_resample_step_never_coarser_than_the_step():
    settings = AnalysisSettings()
    values = [NAN, 0.01, 0.07, 0.3, 1.0, 3.0, 7.9, 12.0, 50.0, 1e3, 1e16]
    steps = [0.04, 0.05, 0.1, 0.2, 0.6, 1.0, 2.0, 5.0, 10.0, 15.9, 50.0, 300.0]
    for plain in values:
        for pred in values:
            for period in [NAN, 1.0, 40.0, 200.0, 4e4]:
                bound = min(
                    (x for x in (plain, pred, period / settings.mmr_resample_steps_per_cycle) if not math.isnan(x)), default=math.inf
                )
                for step in steps:
                    got = sampling.resample_step(plain, pred, period, step, settings)
                    if got is None:
                        continue
                    assert got < step, (plain, pred, period, step, got)
                    assert got <= bound and got in settings.mmr_resample_steps_yr
                    assert got == max(s for s in settings.mmr_resample_steps_yr if s <= bound)


def _record(key, n_planet, step, nu=0.0, lib_amp=0.0, lib_period=None, n_rows=2000):
    """Circulation at rate nu [rad/yr] or a libration of lib_amp about pi; a consistent with sigma_dot."""
    spec = ResonanceSpec.from_key(key)
    t = np.arange(n_rows) * step
    if lib_amp:
        w = TWO_PI / lib_period
        sig, sdot = np.pi + lib_amp * np.sin(w * t), lib_amp * w * np.cos(w * t)
    else:
        sig, sdot = 0.3 + nu * t, np.full_like(t, nu)
    n = (sdot - spec.planet_coefficients[0] * n_planet) / spec.body_coefficient
    a = (TWO_PI / n) ** (2 / 3)
    varpi, Omega = np.full_like(t, 1.2), np.full_like(t, 0.5)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full_like(t, 0.1),
        inc=np.full_like(t, 0.05),
        Omega=Omega,
        omega=varpi - Omega,
        varpi=varpi,
        angle_wrapped=np.mod(sig, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=step, span_yr=float(t[-1]), n_rows=len(t)),
        name='synthetic',
    )


RECORDS = [
    ('2J-1+0-1', N_JUPITER, 15.9, dict(nu=0.2)),  # aliased circulation (17346-like step)
    ('2J-1+0-1', N_JUPITER, 15.9, dict(nu=0.5)),
    ('2J-1+0-1', N_JUPITER, 2.0, dict(nu=0.2)),
    ('3N-4+0+1', N_NEPTUNE, 50.0, dict(nu=0.05)),  # marginal
    ('2J-1+0-1', N_JUPITER, 15.9, dict(lib_amp=1.0, lib_period=200.0)),  # libration with < 20 samples per cycle
    ('2J-1+0-1', N_JUPITER, 15.9, dict(lib_amp=2.5, lib_period=120.0)),
    ('2J-1+0-1', N_JUPITER, 15.9, dict(lib_amp=2.5, lib_period=300.0)),
    ('3N-4+0+1', N_NEPTUNE, 50.0, dict(lib_amp=1.0, lib_period=600.0)),
    ('2J-1+0-1', N_JUPITER, 1.0, dict(lib_amp=1.0, lib_period=400.0)),
]


@pytest.mark.parametrize('key, n_planet, step, kwargs', RECORDS)
def test_analysis_never_recommends_a_coarser_step(key, n_planet, step, kwargs):
    analysis = analyse_mmr(_record(key, n_planet, step, **kwargs), AnalysisSettings(), scan_subresonances=False, scan_owner=False)
    for got in (analysis.sampling.resample_step_yr if analysis.sampling else None, analysis.verdict.resample_step_yr):
        assert got is None or got < step, (got, step, analysis.verdict.action)


def test_aliased_record_gets_a_finer_step():
    """2:1 with Jupiter, sigma circulating at 0.2 rad/yr, saved every 15.9 yr: |nu| dt = 3.2 rad > pi (aliased).
    The plain bound is 0.5 pi / 0.2 = 7.85 yr, so the recommendation exists and is at most 5 yr."""
    analysis = analyse_mmr(_record('2J-1+0-1', N_JUPITER, 15.9, nu=0.2), AnalysisSettings(), scan_subresonances=False, scan_owner=False)
    got = analysis.sampling.resample_step_yr
    assert got is not None and got <= 0.5 * np.pi / 0.2 and got < 15.9
