"""Conversions of the fit coordinates to Keplerian elements, against values computed independently:

- MPC publishes the same orbit three ways at one epoch (CAR, COM, KEP): CAR -> Keplerian and COM -> Keplerian must give
  MPC's own KEP block;
- equinoctial -> Keplerian -> Cartesian (REBOUND) against equinoctial -> Cartesian by the equinoctial formulas of
  Broucke & Cefola (1972), written out here;
- AstDyS's equinoctial 463 against MPC's Keplerian 463 at the same epoch (two fits of one orbit: their difference is
  that of the fits, 1e-7, while a unit error in lambda or the angles would be degrees);
- Cartesian -> Keplerian against REBOUND on random prograde and retrograde orbits.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import rebound

from resonances.clones import parse_oef
from resonances.clones.elements import (
    MU_AU3_DAY2,
    cartesian_to_keplerian,
    cometary_to_keplerian,
    drift_mean_anomaly,
    equinoctial_to_keplerian,
)
from resonances.data.const import K

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures' / 'clones'
MJD_TO_JD = 2400000.5


def angle_difference_deg(a, b):
    return abs((np.degrees(a) - b + 180.0) % 360.0 - 180.0)


@pytest.fixture(scope='module')
def mpc_463():
    orbit = json.loads((FIXTURES / 'mpc/463.json').read_text())[0]['mpc_orb'][0]
    return {block: dict(zip(orbit[block]['coefficient_names'], orbit[block]['coefficient_values'])) for block in ('CAR', 'COM', 'KEP')}


def assert_matches_mpc_kep(kep, reference, rtol_a, angle_tol_deg):
    assert kep['valid'][0]
    assert kep['a'][0] == pytest.approx(reference['a'], rel=rtol_a)
    assert kep['e'][0] == pytest.approx(reference['e'], abs=1e-9)
    for key, name in (('inc', 'i'), ('Omega', 'node'), ('omega', 'argperi'), ('M', 'mean_anomaly')):
        assert angle_difference_deg(kep[key][0], reference[name]) < angle_tol_deg, key


def test_cartesian_to_keplerian_gives_mpc_kep(mpc_463):
    car = mpc_463['CAR']
    kep = cartesian_to_keplerian([[car[k] for k in ('x', 'y', 'z', 'vx', 'vy', 'vz')]])
    # 2e-10 in a: MPC's GM of the Sun against k^2
    assert_matches_mpc_kep(kep, mpc_463['KEP'], rtol_a=1e-9, angle_tol_deg=1e-6)


def test_cometary_to_keplerian_gives_mpc_kep(mpc_463):
    com = mpc_463['COM']
    row = [[com['e'], com['q'], com['peri_time'] + MJD_TO_JD, com['node'], com['argperi'], com['i']]]
    kep = cometary_to_keplerian(row, epoch_jd=61200.0 + MJD_TO_JD)
    assert_matches_mpc_kep(kep, mpc_463['KEP'], rtol_a=1e-12, angle_tol_deg=1e-7)


def test_astdys_equinoctial_agrees_with_mpc_keplerian(mpc_463):
    """Two fits of 463 at MJD 61200: AstDyS (equinoctial, lambda in degrees) and MPC (Keplerian, degrees)."""
    solution = parse_oef((FIXTURES / 'astdys/463.eq1').read_text())
    kep = equinoctial_to_keplerian(solution.nominal)
    reference = mpc_463['KEP']
    assert kep['a'][0] == pytest.approx(reference['a'], abs=1e-7)
    assert kep['e'][0] == pytest.approx(reference['e'], abs=1e-6)
    assert angle_difference_deg(kep['inc'][0], reference['i']) < 1e-4
    longitude = kep['Omega'][0] + kep['omega'][0] + kep['M'][0]
    assert angle_difference_deg(longitude, (reference['node'] + reference['argperi'] + reference['mean_anomaly']) % 360) < 1e-4


def equinoctial_position(a, h, k, p, q, lam_deg):
    """Broucke & Cefola (1972): position from equinoctial elements via the eccentric longitude F,
    lambda = F + h cos F - k sin F."""
    lam = np.radians(lam_deg)
    F = lam
    for _ in range(50):
        F = F - (F + h * np.cos(F) - k * np.sin(F) - lam) / (1 - h * np.sin(F) - k * np.cos(F))
    beta = 1.0 / (1.0 + np.sqrt(1.0 - h * h - k * k))
    x1 = a * ((1 - beta * h * h) * np.cos(F) + h * k * beta * np.sin(F) - k)
    y1 = a * ((1 - beta * k * k) * np.sin(F) + h * k * beta * np.cos(F) - h)
    s = 1 + p * p + q * q
    f_hat = np.array([1 - p * p + q * q, 2 * p * q, -2 * p]) / s
    g_hat = np.array([2 * p * q, 1 + p * p - q * q, 2 * q]) / s
    return x1 * f_hat + y1 * g_hat


def rebound_position(kep):
    sim = rebound.Simulation()
    sim.add(m=1.0)
    sim.add(m=0.0, a=kep['a'][0], e=kep['e'][0], inc=kep['inc'][0], Omega=kep['Omega'][0], omega=kep['omega'][0], M=kep['M'][0])
    particle = sim.particles[1]
    return np.array([particle.x, particle.y, particle.z])


@pytest.mark.parametrize(
    'row',
    [
        [2.3986620775145058, 0.022740843562434, 0.218291436497379, 0.070582641340646, 0.095430514455147, 316.1672474068565],
        [1.0018750437289299, 0.066922997951313, -0.003115254397443, 0.001762834949788, -0.010174807445825, 359.1355059835506],
        [5.2, -0.3, 0.4, 2.5, -1.2, 47.0],  # tan(i/2) = 2.77: a retrograde orbit, i = 140.3 deg
    ],
)
def test_equinoctial_to_keplerian_against_broucke_cefola(row):
    kep = equinoctial_to_keplerian(row)
    assert kep['valid'][0]
    np.testing.assert_allclose(rebound_position(kep), equinoctial_position(*row), atol=1e-12)


def test_retrograde_equinoctial_inclination():
    kep = equinoctial_to_keplerian([5.2, -0.3, 0.4, 2.5, -1.2, 47.0])
    assert np.degrees(kep['inc'][0]) == pytest.approx(2 * np.degrees(np.arctan(np.hypot(2.5, -1.2))))
    assert np.degrees(kep['inc'][0]) > 90


def test_cartesian_to_keplerian_against_rebound():
    rng = np.random.default_rng(7)
    sim = rebound.Simulation()
    sim.add(m=1.0)
    for _ in range(40):
        inc = rng.uniform(0.01, np.pi - 0.01)  # prograde and retrograde
        sim.add(
            m=0.0,
            a=rng.uniform(0.6, 40),
            e=rng.uniform(0.0, 0.95),
            inc=inc,
            Omega=rng.uniform(0, 2 * np.pi),
            omega=rng.uniform(0, 2 * np.pi),
            M=rng.uniform(0, 2 * np.pi),
        )
    rows = np.array([[p.x, p.y, p.z, p.vx * K, p.vy * K, p.vz * K] for p in sim.particles[1:]])  # code time 1/k day -> day
    kep = cartesian_to_keplerian(rows, mu=MU_AU3_DAY2)
    for i, particle in enumerate(sim.particles[1:]):
        orbit = particle.orbit(primary=sim.particles[0])
        assert kep['a'][i] == pytest.approx(orbit.a, rel=1e-10)
        assert kep['e'][i] == pytest.approx(orbit.e, abs=1e-10)
        for key, value in (('inc', orbit.inc), ('Omega', orbit.Omega), ('omega', orbit.omega), ('M', orbit.M)):
            assert abs((kep[key][i] - value + np.pi) % (2 * np.pi) - np.pi) < 1e-8, key


def test_unbound_rows_are_invalid_not_dropped():
    kep = cometary_to_keplerian([[0.5, 1.0, 2451545.0, 10, 20, 5], [1.2, 1.0, 2451545.0, 10, 20, 5]], epoch_jd=2451545.0)
    assert kep['valid'].tolist() == [True, False]
    assert kep['a'][0] == pytest.approx(2.0) and np.isnan(kep['a'][1])
    kep = equinoctial_to_keplerian([[2.0, 0.8, 0.8, 0.1, 0.1, 10.0]])  # e = 1.13
    assert not kep['valid'][0]


def test_cometary_mean_anomaly_by_hand():
    """q = 1.5 AU, e = 0.25: a = 2 AU, n = k / 2^1.5 rad/day; 100 days after perihelion M = 100 n."""
    kep = cometary_to_keplerian([[0.25, 1.5, 2451445.0, 0.0, 0.0, 0.0]], epoch_jd=2451545.0)
    assert kep['a'][0] == pytest.approx(2.0)
    assert kep['M'][0] == pytest.approx(100 * 0.0172020989484 / 2**1.5)


def test_drift_mean_anomaly():
    elements = {'a': np.array([1.0]), 'M': np.array([0.0])}
    assert drift_mean_anomaly(elements, 10.0)['M'][0] == pytest.approx(10 * K)


@pytest.mark.parametrize('inc', [0.0, 0.4, np.pi / 2, 2.5, np.pi])
def test_circular_orbits_keep_their_phase(inc):
    """e = 0: the pericentre has no direction and f cannot come from e (it gave f = 0, a 90-degree error for
    r = (0, 1, 0), v = (-k, 0, 0) with `valid` True). The position must come back: REBOUND places circular orbits at
    their true longitude (prograde, inclined, polar, retrograde, retrograde equatorial), the conversion's elements put
    back into REBOUND must give the same position and velocity."""
    sim = rebound.Simulation()
    sim.add(m=1.0)
    for theta in (0.3, 1.9, 4.4):
        sim.add(m=0.0, a=1.7, e=0.0, inc=inc, Omega=0.8, omega=0.0, f=theta)
    rows = [[p.x, p.y, p.z, p.vx * K, p.vy * K, p.vz * K] for p in sim.particles[1:]]
    rows.append([0.0, 1.0, 0.0, -K, 0.0, 0.0])  # the review's case: at (0, 1, 0), moving towards -x
    kep = cartesian_to_keplerian(rows, mu=MU_AU3_DAY2)
    assert kep['valid'].all()
    back = rebound.Simulation()
    back.add(m=1.0)
    for i in range(len(rows)):
        assert kep['e'][i] < 1e-12
        back.add(m=0.0, **{key: float(kep[key][i]) for key in ('a', 'e', 'inc', 'Omega', 'omega', 'M')})
    for row, particle in zip(rows, back.particles[1:]):
        np.testing.assert_allclose([particle.x, particle.y, particle.z], row[:3], atol=1e-12)
        np.testing.assert_allclose([particle.vx * K, particle.vy * K, particle.vz * K], row[3:], atol=1e-12)
