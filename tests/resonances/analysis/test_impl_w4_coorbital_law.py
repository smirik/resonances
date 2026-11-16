"""The co-orbital (1:1) force law from the averaged disturbing function (`analysis/mmr/coorbital.py`).

Independent sources of the expected values:
- the planar circular closed form of Murray & Dermott (1999), eqs. 3.178-3.179, p. 105: (3/4) dr^2 + mu H(theta) = const,
  H = 1/sin(theta/2) - 2 cos(theta) - 2; the stable points at 60/300 deg, the maximum at 180 deg and the critical tadpole's
  smallest angle 'theta_min = 23.5 deg for all mu' (p. 106; the exact root of H = 1 is 23.9 deg);
- Kepler's equation solved by bisection here (not Newton);
- full three-body integrations with REBOUND (Sun, a planet of mu = 1e-4 on a circular orbit, a massless body): the
  osculating heliocentric a of a tadpole, a horseshoe and a quasi-satellite against the averaged theory, which knows
  nothing of these runs; the theory's scale must come out K / K_theory = 1 (the planet's mass) without any fit of mu;
- two benchmark records (skipped without benchmark/data): the Earth horseshoe/quasi-satellite 2002 AA29 and 3753 Cruithne.
"""

import numpy as np
import pytest
import rebound

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.mmr import coorbital as co
from resonances.resonance.analysis.mmr import io, response as response_module
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

pytestmark = pytest.mark.slow  # a shared setup of more than 1 s (tests/CLAUDE.md)

SETTINGS = AnalysisSettings()
MU = 1.0e-4
CUTOFF_YR = 3.0  # far above the 1-yr fast terms, far below the 40-200 yr librations at mu = 1e-4
CIRCULAR = co.Orbit(0.0, 0.0, 0.0, 0.0)


def m_and_d_h(theta):
    """M&D eq. 3.179."""
    return 1.0 / np.sin(theta / 2) - 2 * np.cos(theta) - 2


def bisect_kepler(m, e):
    low, high = 0.0, 2 * np.pi
    for _ in range(200):
        mid = (low + high) / 2
        if mid - e * np.sin(mid) - m > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


# --- the averaged potential --------------------------------------------------------------------------


def test_kepler_against_bisection():
    for m, e in ((1.0, 0.5), (0.1, 0.9), (5.5, 0.3), (3.0, 0.0)):
        assert co.eccentric_anomaly(np.array([m]), e)[0] == pytest.approx(bisect_kepler(m, e), abs=1e-12)


def test_circular_planar_average_is_murray_dermott():
    """e = i = 0: nothing to average; the quadrature must reproduce H(theta) = 2 R~ - 2 (eq. 3.179) and its slope."""
    table = co.averaged_table(CIRCULAR, CIRCULAR, 720, 0.0)
    inside = (table.theta_rad > np.radians(5)) & (table.theta_rad < np.radians(355))
    theta = table.theta_rad[inside]
    assert 2 * table.potential[inside] - 2 == pytest.approx(m_and_d_h(theta), rel=1e-10)
    slope = np.gradient(m_and_d_h(np.linspace(np.radians(5), np.radians(355), 200001)), np.linspace(np.radians(5), np.radians(355), 200001))
    expected = np.interp(theta, np.linspace(np.radians(5), np.radians(355), 200001), slope) / 2
    assert table.force[inside] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_lagrange_points_and_critical_tadpole():
    """Minimum of the potential (stable tadpole centre) at 60 and 300 deg, maximum between them at 180 deg (L3);
    the critical tadpole (energy of L3) turns at M&D's 23.5 deg (exact root of H = 1: 23.9 deg)."""
    table = co.averaged_table(CIRCULAR, CIRCULAR, 1440, 0.0)
    deg = np.degrees(table.theta_rad)
    leading = (deg > 10) & (deg < 170)
    trailing = (deg > 190) & (deg < 350)
    assert deg[leading][np.argmin(table.potential[leading])] == pytest.approx(60.0, abs=0.25)
    assert deg[trailing][np.argmin(table.potential[trailing])] == pytest.approx(300.0, abs=0.25)
    middle = (deg > 70) & (deg < 290)
    assert deg[middle][np.argmax(table.potential[middle])] == pytest.approx(180.0, abs=0.25)
    l3 = table.potential[np.argmin(np.abs(deg - 180))]
    near = (deg > 5) & (deg < 60)
    turning = deg[near][np.argmin(np.abs(table.potential[near] - l3))]
    assert turning == pytest.approx(23.5, abs=0.5)


def test_force_is_the_derivative_of_the_potential_for_an_inclined_eccentric_orbit():
    """F~ = <dR/dlambda> (gradient . velocity / n) equals d<R>/dtheta: two different computations of eq. (1)."""
    body = co.Orbit(0.3, np.radians(10), 1.0, 2.0)
    planet = co.Orbit(0.0167, 0.0, 1.8, 0.0)
    table = co.averaged_table(body, planet, 5760, co.hill_softening(3e-6, 1.0))
    slope = np.gradient(table.potential, table.theta_rad)
    assert np.max(np.abs(slope - table.force)) < 1e-3 * np.max(np.abs(table.force))


@pytest.mark.parametrize('e', [0.1, 0.2, 0.3])
def test_eccentricity_removes_the_singularity_at_zero(e):
    """At e = i = 0 the potential is infinite at theta = 0 (the planet itself). With an epicycle the body passes the
    planet only where its true longitude, lambda + 2 e sin M to first order in e (M&D eq. 2.88), meets the planet's:
    the barrier sits at |theta| = 2 e, and theta = 0 inside it is a local minimum (the quasi-satellite centre,
    Namouni 1999)."""
    table = co.averaged_table(co.Orbit(e, 0.0, 0.0, 0.0), CIRCULAR, 1440, 0.0)
    assert np.all(np.isfinite(table.potential))
    assert table.potential[0] < table.potential[1] and table.potential[0] < table.potential[-1]
    deg = np.degrees(table.theta_rad)
    near = (deg > 0) & (deg < 60)
    assert deg[near][np.argmax(table.potential[near])] == pytest.approx(np.degrees(2 * e), abs=0.5)


def test_tables_are_retabulated_only_beyond_the_tolerances():
    base = (co.Orbit(0.1, np.radians(5), 1.0, 2.0), CIRCULAR)
    nudged_e = (co.Orbit(0.105, np.radians(5), 1.0, 2.0), CIRCULAR)
    moved_e = (co.Orbit(0.13, np.radians(5), 1.0, 2.0), CIRCULAR)
    turned_node = (co.Orbit(0.1, np.radians(5), 1.0, 2.0 + np.radians(30)), CIRCULAR)
    assert not co.needs_new_table(base, nudged_e, 0.01, np.radians(1.0))
    assert co.needs_new_table(base, moved_e, 0.01, np.radians(1.0))
    assert co.needs_new_table(base, turned_node, 0.01, np.radians(1.0))  # |d (sin i e^{i Omega})| = 0.045 > sin 1 deg
    # e -> 0: omega is undefined, the eccentricity vector is not
    assert not co.needs_new_table((co.Orbit(0.001, 0, 0.0, 0), CIRCULAR), (co.Orbit(0.001, 0, 3.0, 0), CIRCULAR), 0.01, np.radians(1.0))


# --- full three-body integrations ----------------------------------------------------------------------


def three_body(dlambda_deg, da, e, varpi_deg=0.0, inc_deg=0.0, years=600.0, step=0.05):
    """Sun + planet (mu, circular, a = 1) + massless body; heliocentric osculating elements every `step` years."""
    sim = rebound.Simulation()
    sim.G = 4 * np.pi**2
    sim.add(m=1.0)
    sim.add(m=MU, a=1.0, l=0.0)
    sim.add(a=1.0 + da, e=e, inc=np.radians(inc_deg), pomega=np.radians(varpi_deg), l=np.radians(dlambda_deg), primary=sim.particles[0])
    sim.move_to_com()
    sim.integrator = 'ias15'
    t = np.arange(0.0, years, step)
    names = ('a', 'e', 'inc', 'omega', 'Omega', 'l', 'l_planet')
    out = {name: np.empty(len(t)) for name in names}
    for i, time in enumerate(t):
        sim.integrate(time)
        body = sim.particles[2].orbit(primary=sim.particles[0])
        planet = sim.particles[1].orbit(primary=sim.particles[0])
        for name, value in zip(names, (body.a, body.e, body.inc, body.omega, body.Omega, body.l, planet.l)):
            out[name][i] = value
    return t, out


def law_of(t, out, flip_adot=False):
    sigma = np.unwrap(out['l_planet'] - out['l'])  # key 1X-1+0+0: sigma = lambda_P - lambda (k = -1)
    n_planet = np.polyfit(t, np.unwrap(out['l_planet']), 1)[0]
    adot = filtering.time_derivative(t, filtering.lowpass(t, out['a'], CUTOFF_YR, SETTINGS.filter_order))
    inside = (t > 3 * CUTOFF_YR) & (t < t[-1] - 3 * CUTOFF_YR)
    zero = np.zeros(len(t))
    data = co.CoorbitalInput(
        t_yr=t,
        angle=sigma,
        a=out['a'],
        adot=-adot if flip_adot else adot,
        mask=inside,
        pieces=[(t[inside][0], t[inside][-1])],
        e=out['e'],
        inc=out['inc'],
        omega=out['omega'],
        Omega=out['Omega'],
        planet='Synthetic',
        planet_mass=MU,
        planet_mean_motion_rad_yr=n_planet,
        body_coefficient=-1,
        cutoffs_yr=(CUTOFF_YR,),
        planet_elements={'e': zero, 'inc': zero, 'Omega': zero, 'varpi': zero},
    )
    return co.coorbital_law(data, SETTINGS)


@pytest.fixture(scope='module')
def tadpole():
    return three_body(80.0, 0.0, 0.0)


@pytest.fixture(scope='module')
def horseshoe():
    return three_body(180.0, 0.02, 0.0)


def test_tadpole_follows_the_averaged_force(tadpole):
    law = law_of(*tadpole)
    assert law.regime == co.REGIME_TADPOLE_L4
    assert law.r2_h1 > 0.99 and law.r2_h2 > 0.99
    assert law.scale == pytest.approx(1.0, abs=0.05)  # the planet's mass, not fitted
    assert law.energy_r2 > 0.99 and law.energy_scale == pytest.approx(1.0, abs=0.05)
    assert (law.force_check, law.potential_check) == (CheckValue.PASS, CheckValue.PASS)


def test_horseshoe_follows_the_averaged_force_where_harmonics_fail(horseshoe):
    law = law_of(*horseshoe)
    assert law.regime == co.REGIME_HORSESHOE
    assert law.r2 > 0.99 and law.scale == pytest.approx(1.0, abs=0.1)
    assert law.harmonic_r2 < 0.5  # sin/cos theta, 2 theta cannot hold the spike at the planet's approach
    assert law.force_check == CheckValue.PASS


def test_inclined_eccentric_horseshoe():
    law = law_of(*three_body(180.0, 0.02, 0.05, inc_deg=10.0, years=800.0))
    assert law.regime == co.REGIME_HORSESHOE
    assert law.r2 > 0.99 and law.scale == pytest.approx(1.0, abs=0.1)
    assert law.n_tables > 1  # the elements moved beyond the tolerances: re-tabulated, not per sample


def test_quasi_satellite():
    """A quasi-satellite (theta about 0, e = 0.15): its e-vector turns with the libration, so the averaging at fixed
    elements is rougher than for tadpoles and horseshoes; still the scale of the planet's force."""
    law = law_of(*three_body(0.0, 0.0, 0.15, varpi_deg=90.0))
    assert law.regime == co.REGIME_QUASI_SATELLITE
    assert law.r2_h1 > 0.7 and law.r2_h2 > 0.7
    assert law.scale == pytest.approx(1.0, abs=0.3)
    assert law.force_check == CheckValue.PASS


def test_a_response_of_the_wrong_sign_fails(horseshoe):
    """The same horseshoe with a_dot reversed: a perfect correlation but a force pushing a the wrong way."""
    law = law_of(*horseshoe, flip_adot=True)
    assert law.scale < 0
    assert law.force_check == CheckValue.FAIL


def test_a_oscillation_unrelated_to_the_planet_fails(horseshoe):
    """a carries a 150-yr sine of the right size but no relation to the planet's force (a kinematic oscillation)."""
    t, out = horseshoe
    fake = dict(out)
    fake['a'] = 1.0 + 0.02 * np.sin(2 * np.pi * t / 150.0 + 1.0)
    law = law_of(t, fake)
    assert law.force_check == CheckValue.FAIL


def test_potential_check_logic():
    P, F, N = CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_MEASURED
    assert co.potential_check(P, P) == P
    assert co.potential_check(P, F) == N  # the local force holds, only E is not conserved: no measurement
    assert co.potential_check(F, F) == F
    assert co.potential_check(N, F) == F
    assert co.potential_check(P, N) == N


def test_columns_present_for_every_angle():
    empty = co.coorbital_columns(None)
    assert all(key.startswith('rsp_coorb_') for key in empty)
    assert empty['rsp_coorb_regime'] == '' and np.isnan(empty['rsp_coorb_R2'])


# --- retrograde 1/-1 (2015 BZ509-like orbit) ----------------------------------------------------------------
#
# sigma = M + (2k - 1) omega + Omega - lambda' of `RetrogradeTwoBody` 1J-1r<k>; the quadrature takes
# M = sigma + lambda' - ((2k - 1) omega + Omega). Independent pieces here: Kepler by vectorised bisection, positions by
# explicit rotation matrices Rz(Omega) Rx(i) Rz(omega), the average on a direct (sigma, lambda') mesh, and a REBOUND
# three-body run that knows nothing of the averaging.

RETRO = dict(e=0.38, inc=np.radians(163.0), omega=np.radians(257.0), Omega=np.radians(307.0))


def rotation(inc, omega, node):
    def rz(x):
        return np.array([[np.cos(x), -np.sin(x), 0.0], [np.sin(x), np.cos(x), 0.0], [0.0, 0.0, 1.0]])

    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(inc), -np.sin(inc)], [0.0, np.sin(inc), np.cos(inc)]])
    return rz(node) @ rx @ rz(omega)


def bisect_kepler_array(m, e):
    m = np.mod(m, 2 * np.pi)
    low, high = np.zeros_like(m), np.full_like(m, 2 * np.pi)
    for _ in range(80):
        mid = (low + high) / 2
        above = mid - e * np.sin(mid) - m > 0
        high, low = np.where(above, mid, high), np.where(above, low, mid)
    return (low + high) / 2


def independent_average(sigma, c, softening, n_lambda=4096, rows=32):
    """(<R>, <dR/dM>) at each sigma for the RETRO body (a = 1) and a circular planar planet (a = 1), on a direct mesh
    (`rows` values of sigma at a time: the mesh of 720 x 8192 points would hold about 1 GB)."""
    sigma = np.asarray(sigma, float)
    parts = [_direct_mesh_average(sigma[i : i + rows], c, softening, n_lambda) for i in range(0, len(sigma), rows)]
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


def _direct_mesh_average(sigma, c, softening, n_lambda):
    lam = 2 * np.pi * np.arange(n_lambda) / n_lambda
    sig, lp = np.meshgrid(sigma, lam, indexing='ij')
    big_e = bisect_kepler_array(sig + lp - (c * RETRO['omega'] + RETRO['Omega']), RETRO['e'])
    root = np.sqrt(1 - RETRO['e'] ** 2)
    local = np.stack([np.cos(big_e) - RETRO['e'], root * np.sin(big_e), np.zeros_like(big_e)])
    velocity = np.stack([-np.sin(big_e), root * np.cos(big_e), np.zeros_like(big_e)]) / (1 - RETRO['e'] * np.cos(big_e))
    turn = rotation(RETRO['inc'], RETRO['omega'], RETRO['Omega'])
    r, v = np.einsum('ij,j...->i...', turn, local), np.einsum('ij,j...->i...', turn, velocity)
    rp = np.stack([np.cos(lp), np.sin(lp), np.zeros_like(lp)])
    d = rp - r
    dist2 = np.sum(d * d, axis=0) + softening**2
    potential = 1 / np.sqrt(dist2) - np.sum(r * rp, axis=0)
    force = np.sum(d * v, axis=0) / dist2**1.5 - np.sum(rp * v, axis=0)
    return potential.mean(axis=1), force.mean(axis=1)


@pytest.mark.parametrize('k', [0, 1])
def test_retrograde_table_is_the_average_at_fixed_sigma(k):
    """The same trapezoid nodes give the same average to round-off (the construction: phase, 3D geometry, index shift);
    at the default grid the average is converged to 1e-4 of the peak force (Jupiter's softening, e = 0.38)."""
    c = 2 * k - 1
    eps = co.hill_softening(1e-3, SETTINGS.mmr_coorbital_softening_hill)
    body = co.Orbit(**RETRO, omega_coefficient=c)
    table = co.averaged_table(body, CIRCULAR, 360, eps)
    potential, force = independent_average(table.theta_rad, c, eps, n_lambda=360)
    assert np.max(np.abs(table.potential - potential)) < 1e-11 * np.max(np.abs(potential))
    assert np.max(np.abs(table.force - force)) < 1e-11 * np.max(np.abs(force))
    table = co.averaged_table(body, CIRCULAR, SETTINGS.mmr_coorbital_grid_points, eps)
    _, converged = independent_average(table.theta_rad, c, eps, n_lambda=8192)
    assert np.max(np.abs(table.force - converged)) < 1e-4 * np.max(np.abs(converged))
    assert np.ptp(table.force) > 0.1  # a real sigma-dependent force


def synthetic_retrograde(c, years=6000.0, step=1.0, mu=1e-3):
    """sigma(t), a(t) of the averaged equations da/dt = 2 mu n a F~(sigma), sigma_dot = -(3/2) (n/a) (a - a_P) with
    F~ from `independent_average` (periodic spline), a_P = 5.2 au: a 2015 BZ509-like retrograde co-orbital of Jupiter."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import CubicSpline

    n = 2 * np.pi / 11.86
    a_planet = (co.GM_SUN / n**2) ** (1 / 3)
    eps = co.hill_softening(mu, SETTINGS.mmr_coorbital_softening_hill)
    grid = np.linspace(0, 2 * np.pi, 721)
    _, force = independent_average(grid[:-1], c, eps)
    spline = CubicSpline(grid, np.append(force, force[0]), bc_type='periodic')

    def rhs(_, y):
        return [-1.5 * n / a_planet * (y[1] - a_planet), 2 * mu * n * a_planet * spline(np.mod(y[0], 2 * np.pi))]

    t = np.arange(0.0, years, step)
    sol = solve_ivp(rhs, (0, t[-1]), [np.pi + 0.5, a_planet], t_eval=t, method='DOP853', rtol=1e-11, atol=1e-13)
    adot = np.array([rhs(0, y)[1] for y in sol.y.T])
    return t, sol.y[0], sol.y[1], adot, n, mu


def law_of_synthetic(t, sigma, a, adot, n, mu, c, retrograde=True):
    zero, ones = np.zeros(len(t)), np.ones(len(t))
    data = co.CoorbitalInput(
        t_yr=t,
        angle=sigma,
        a=a,
        adot=adot,
        mask=np.ones(len(t), dtype=bool),
        pieces=[(t[0], t[-1])],
        e=RETRO['e'] * ones,
        inc=RETRO['inc'] * ones,
        omega=RETRO['omega'] * ones,
        Omega=RETRO['Omega'] * ones,
        planet='Synthetic',
        planet_mass=mu,
        planet_mean_motion_rad_yr=n,
        body_coefficient=1,
        cutoffs_yr=(),
        planet_elements={'e': zero, 'inc': zero, 'Omega': zero, 'varpi': zero},
        retrograde=retrograde,
        omega_coefficient=c,
    )
    return co.coorbital_law(data, SETTINGS)


@pytest.fixture(scope='module')
def synthetic_k1():
    return synthetic_retrograde(1)


def test_retrograde_synthetic_trajectory_gives_the_planet_mass(synthetic_k1):
    t, sigma, a, adot, n, mu = synthetic_k1
    assert np.ptp(sigma) < 2 * np.pi  # it librates (about 180 deg)
    law = law_of_synthetic(t, sigma, a, adot, n, mu, c=1)
    assert law.regime == co.REGIME_RETROGRADE
    assert law.r2_h1 > 0.999 and law.r2_h2 > 0.999
    assert law.scale == pytest.approx(1.0, abs=0.01)
    assert law.energy_r2 > 0.999 and law.energy_scale == pytest.approx(1.0, abs=0.01)
    assert (law.force_check, law.potential_check) == (CheckValue.PASS, CheckValue.PASS)


def test_retrograde_synthetic_trajectory_with_the_other_phase_fails(synthetic_k1):
    """The same trajectory read as the k = 0 angle (M - omega + Omega - lambda'): not the force that moved a."""
    law = law_of_synthetic(*synthetic_k1, c=-1)
    assert law.force_check == CheckValue.FAIL


def three_body_retrograde(years=600.0, step=0.05):
    """Sun + planet (MU, circular, a = 1) + a massless retrograde body with the RETRO shape, sigma (k = 1) = 180 deg
    at the start; heliocentric osculating elements."""
    sim = rebound.Simulation()
    sim.G = 4 * np.pi**2
    sim.add(m=1.0)
    sim.add(m=MU, a=1.0, l=0.0)
    mean_anomaly = np.pi - RETRO['omega'] - RETRO['Omega']
    sim.add(a=1.0, M=mean_anomaly, primary=sim.particles[0], **RETRO)
    sim.move_to_com()
    sim.integrator = 'ias15'
    t = np.arange(0.0, years, step)
    names = ('a', 'e', 'inc', 'omega', 'Omega', 'M', 'l_planet')
    out = {name: np.empty(len(t)) for name in names}
    for i, time in enumerate(t):
        sim.integrate(time)
        body = sim.particles[2].orbit(primary=sim.particles[0])
        planet = sim.particles[1].orbit(primary=sim.particles[0])
        for name, value in zip(names, (body.a, body.e, body.inc, body.omega, body.Omega, body.M, planet.l)):
            out[name][i] = value
    return t, out


def test_retrograde_three_body_follows_the_averaged_force():
    """The REBOUND body's a moves under the averaged 1/-1 force with the planet's mass (K / K_theory = 1, not fitted);
    the angle from M, omega, Omega (`RetrogradeTwoBody`), never from REBOUND's l or pomega."""
    t, out = three_body_retrograde()
    sigma = np.unwrap(out['M'] + out['omega'] + out['Omega'] - out['l_planet'])
    assert np.ptp(sigma) < 2 * np.pi
    n_planet = np.polyfit(t, np.unwrap(out['l_planet']), 1)[0]
    adot = filtering.time_derivative(t, filtering.lowpass(t, out['a'], CUTOFF_YR, SETTINGS.filter_order))
    inside = (t > 3 * CUTOFF_YR) & (t < t[-1] - 3 * CUTOFF_YR)
    zero = np.zeros(len(t))
    data = co.CoorbitalInput(
        t_yr=t,
        angle=sigma,
        a=out['a'],
        adot=adot,
        mask=inside,
        pieces=[(t[inside][0], t[inside][-1])],
        e=out['e'],
        inc=out['inc'],
        omega=out['omega'],
        Omega=out['Omega'],
        planet='Synthetic',
        planet_mass=MU,
        planet_mean_motion_rad_yr=n_planet,
        body_coefficient=1,
        cutoffs_yr=(CUTOFF_YR,),
        planet_elements={'e': zero, 'inc': zero, 'Omega': zero, 'varpi': zero},
        retrograde=True,
        omega_coefficient=1,
    )
    law = co.coorbital_law(data, SETTINGS)
    assert law.regime == co.REGIME_RETROGRADE
    assert law.r2_h1 > 0.98 and law.r2_h2 > 0.98
    assert law.scale == pytest.approx(1.0, abs=0.05)
    assert law.force_check == CheckValue.PASS


def test_law_planet_takes_the_retrograde_one_to_one_only():
    from resonances.resonance.analysis.mmr.models import ResonanceSpec

    planet = {key: response_module.law_planet(ResonanceSpec.from_key(key)) for key in ('1J-1r0', '1J-1r1', '1J-3r0', '1E-1+0+0', '2J-1')}
    assert planet == {'1J-1r0': 'Jupiter', '1J-1r1': 'Jupiter', '1J-3r0': None, '1E-1+0+0': 'Earth', '2J-1': None}


# --- benchmark records ---------------------------------------------------------------------------------


def _analysis(example_id):
    if not (io.example_path(example_id) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = io.load_series(example_id)
    analysis = analyse_mmr(series, SETTINGS)
    return series, analysis


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example_id, regime',
    [('coorbital__2002_AA29__1E-1+0+0', co.REGIME_COMPOUND), ('coorbital__3753__1E-1+0+0', co.REGIME_COMPOUND)],
)
def test_real_earth_coorbitals(example_id, regime):
    """2002 AA29 (horseshoe that touches the quasi-satellite side, Connors et al. 2002) and 3753 Cruithne: the force
    law holds with the Earth's mass (K / K_theory 1.02 and 1.04; the Moon adds 1.2 %)."""
    series, analysis = _analysis(example_id)
    response = response_module.analyse_response(
        analysis.prep, analysis.slow, analysis.geometry, analysis.context, analysis.sampling, SETTINGS, spec=series.resonance, series=series
    )
    law = response.coorbital
    assert law.regime == regime
    assert law.r2_h1 > 0.95 and law.r2_h2 > 0.95
    assert law.scale == pytest.approx(1.0, abs=0.1)
    assert response.force_law == co.LAW_COORBITAL
    assert response.a_response == CheckValue.PASS
    assert response.to_columns()['rsp_coorb_force_check'] == 'pass'
