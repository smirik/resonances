"""RetrogradeTwoBody: the p/-q angle of Morais & Namouni (2013a) and Li et al. (2019, eq. 2).

The expectations come from outside the class: elements computed here from the state vectors
in a frame where the orbit is prograde, the angles as the papers write them, Kepler motion
with massless bodies, and resonant axes computed by hand from PLANETS_AXIS.
"""

import numpy as np
import pytest
import rebound

import resonances
import tests.tools as tools
from resonances.data import const
from resonances.mmr.mmr_finder import candidate_mmrs
from resonances.mmr.retrograde_two_body import RetrogradeTwoBody
from resonances.mmr.three_body import ThreeBody
from resonances.mmr.two_body import TwoBody

TWO_PI = 2 * np.pi


def wrap(x):
    return (np.asarray(x) + np.pi) % TWO_PI - np.pi


def elements_from_state(r, v, mu=1.0):
    """(i, Omega, omega, M) of an elliptic orbit from position and velocity, written here
    independently of REBOUND: angular momentum, node vector, eccentricity vector."""
    r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
    h = np.cross(r, v)
    h_hat = h / np.linalg.norm(h)
    node = np.cross([0.0, 0.0, 1.0], h)
    e_vec = np.cross(v, h) / mu - r / np.linalg.norm(r)
    e = np.linalg.norm(e_vec)
    inc = np.arccos(h_hat[2])
    Omega = np.arctan2(node[1], node[0])
    omega = np.arctan2(np.cross(node, e_vec) @ h_hat, node @ e_vec)
    f = np.arctan2(np.cross(e_vec, r) @ h_hat, e_vec @ r)
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f / 2))
    return inc, Omega, omega, E - e * np.sin(E)


def two_massless(planet_axis, body, planet_inc=0.02):
    """Unit central mass (G = 1), a massless planet and a massless body (dict of elements)."""
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=1.0)
    sim.add(m=0.0, a=planet_axis, e=0.05, inc=planet_inc, Omega=0.3, omega=1.0, M=0.5)
    sim.add(m=0.0, **body)
    return sim


def random_retrograde(rng):
    return dict(
        a=rng.uniform(3.0, 15.0),
        e=rng.uniform(0.05, 0.8),
        inc=rng.uniform(np.pi / 2 + 0.05, np.pi - 0.05),
        Omega=rng.uniform(0, TWO_PI),
        omega=rng.uniform(0, TWO_PI),
        M=rng.uniform(0, TWO_PI),
    )


def random_resonance(rng):
    while True:
        p, q = (int(x) for x in rng.integers(1, 10, 2))
        if np.gcd(p, q) == 1:
            return RetrogradeTwoBody(p=p, q=q, k=int(rng.integers(0, (p + q) // 2 + 1)), planet='Jupiter')


class TestAngle:
    def test_matches_the_angle_in_the_frame_where_the_orbit_is_prograde(self):
        """Turned by pi about x the orbit is prograde, with lam_f = lam*, varpi_f = varpi* and
        Omega_f = pi - Omega; there phi = q lam_f - p lam' - (p+q-2k) varpi_f - 2k Omega_f."""
        rng = np.random.default_rng(2026)
        for _ in range(8):
            mmr = random_resonance(rng)
            sim = two_massless(5.2, random_retrograde(rng))
            planet, body = sim.orbits()
            s = sim.particles[2]
            inc_f, Omega_f, omega_f, M_f = elements_from_state([s.x, -s.y, -s.z], [s.vx, -s.vy, -s.vz])
            assert inc_f < np.pi / 2
            p, q, k = mmr.p, mmr.q, mmr.k
            expected = q * (M_f + omega_f + Omega_f) - p * planet.l - (p + q - 2 * k) * (omega_f + Omega_f) - 2 * k * Omega_f
            assert abs(wrap(mmr.calc_angle(body, [planet]) - expected)) < 1e-9

    def test_does_not_change_when_the_frame_is_rotated_about_z(self):
        rng = np.random.default_rng(7)
        for _ in range(6):
            mmr = random_resonance(rng)
            sim = two_massless(5.2, random_retrograde(rng))
            before = mmr.calc_angle(sim.orbits()[1], [sim.orbits()[0]])
            delta = rng.uniform(0, TWO_PI)
            c, s = np.cos(delta), np.sin(delta)
            for particle in sim.particles:
                x, y, vx, vy = particle.x, particle.y, particle.vx, particle.vy
                particle.x, particle.y = c * x - s * y, s * x + c * y
                particle.vx, particle.vy = c * vx - s * vy, s * vx + c * vy
            planet, body = sim.orbits()
            assert abs(wrap(mmr.calc_angle(body, [planet]) - before)) < 1e-9

    @pytest.mark.parametrize(
        'notation, inc',
        [('1J-3r0', np.radians(150)), ('1J-1r1', np.radians(163)), ('7N-9r6', np.radians(110)), ('2S-3r0', np.radians(137))],
    )
    def test_is_constant_at_the_exact_commensurability(self, notation, inc):
        """Kepler motion: at a = (q/p)^(2/3) a' the angle stays put for 50 periods. Built from
        REBOUND's l and pomega of the body instead, it would run off by tens of thousands of degrees."""
        mmr = resonances.create_mmr(notation)
        planet_axis = const.PLANETS_AXIS[mmr.planets_names[0]]
        sim = two_massless(planet_axis, dict(a=mmr.resonant_axis, e=0.4, inc=inc, Omega=1.2, omega=2.1, M=0.7))
        sim.integrator = 'whfast'
        sim.dt = 0.001 * TWO_PI * min(planet_axis, mmr.resonant_axis) ** 1.5
        period = TWO_PI * max(planet_axis, mmr.resonant_axis) ** 1.5
        phi, phi_rebound = [], []
        for t in np.linspace(0, 50 * period, 200):
            sim.integrate(t)
            planet, body = sim.orbits()
            phi.append(mmr.calc_angle(body, [planet]))
            p, q, k = mmr.p, mmr.q, mmr.k
            phi_rebound.append(q * body.l - p * planet.l - (p + q - 2 * k) * body.pomega + 2 * k * body.Omega)
        assert np.ptp(np.unwrap(phi)) < 1e-6
        assert np.ptp(np.unwrap(phi_rebound)) > 100

    def test_rate_off_the_commensurability_is_q_n_minus_p_n_planet(self):
        mmr = resonances.create_mmr('1J-3r0')
        sim = two_massless(const.PLANETS_AXIS['Jupiter'], dict(a=11.2, e=0.3, inc=np.radians(150), Omega=0.4, omega=2.0, M=1.0))
        sim.integrator = 'whfast'
        sim.dt = 0.01
        planet, body = sim.orbits()
        expected = mmr.q * body.n - mmr.p * planet.n
        times = np.linspace(0, 2000, 400)
        phi = []
        for t in times:
            sim.integrate(t)
            planet, body = sim.orbits()
            phi.append(mmr.calc_angle(body, [planet]))
        rate = np.polyfit(times, np.unwrap(phi), 1)[0]
        assert rate == pytest.approx(expected, rel=1e-6)
        assert expected < 0  # a = 11.2 is outside 10.82: 3 n < n_J

    @pytest.mark.parametrize(
        'notation, written',
        [
            # Morais & Namouni 2017, 2011 KT19 in 7/-9 with Neptune: 9 lam - 7 lam' - 2 Omega - 4 omega, lam = M + omega + Omega
            ('7N-9r6', lambda M, w, W, lp: 9 * (M + w + W) - 7 * lp - 2 * W - 4 * w),
            # Namouni & Morais 2018, 2015 BZ509: phi = lam - lam_J (k = 1) and phi* = lam - lam_J - 2 omega (k = 0)
            ('1J-1r1', lambda M, w, W, lp: (M + w + W) - lp),
            ('1J-1r0', lambda M, w, W, lp: (M + w + W) - lp - 2 * w),
        ],
    )
    def test_matches_the_angles_written_in_the_papers(self, notation, written):
        mmr = resonances.create_mmr(notation)
        rng = np.random.default_rng(11)
        for _ in range(5):
            sim = two_massless(5.2, random_retrograde(rng))
            planet, body = sim.orbits()
            assert abs(wrap(mmr.calc_angle(body, [planet]) - written(body.M, body.omega, body.Omega, planet.l))) < 1e-9

    def test_rebound_reports_l_and_pomega_of_a_retrograde_orbit_with_the_opposite_sign(self):
        """Why calc_angle reads Omega, omega, M only: for i > 90 deg REBOUND 4.x gives l = -lam*
        and pomega = -varpi*. If this fails, REBOUND has changed its convention."""
        rng = np.random.default_rng(3)
        for _ in range(5):
            body = two_massless(5.2, random_retrograde(rng)).orbits()[1]
            assert abs(wrap(body.l - (body.Omega - body.omega - body.M))) < 1e-9
            assert abs(wrap(body.pomega - (body.Omega - body.omega))) < 1e-9


class TestResonance:
    @pytest.mark.parametrize(
        'notation, prograde, axis',
        [
            ('1J-3r0', '1J-3', 10.8216),
            ('3J-1r1', '3J-1', 2.5011),
            ('1J-1r0', '1J-1', 5.2025),
            ('1S-1r0', '1S-1', 9.5415),
            ('2S-3r0', '2S-3', 12.5029),
            ('7N-9r6', '7N-9', 35.5541),
            ('13S-5r0', '13S-5', 5.0462),
        ],
    )
    def test_resonant_axis(self, notation, prograde, axis):
        mmr = resonances.create_mmr(notation)
        assert mmr.resonant_axis == pytest.approx(axis, abs=1e-4)
        # Li et al. 2019, eq. 1: p/q and p/-q share the axis
        assert mmr.resonant_axis == pytest.approx(TwoBody(prograde).resonant_axis)

    def test_order_and_coefficients(self):
        mmr = resonances.create_mmr('7N-9r6')
        assert mmr.order() == 16
        assert (mmr.p, mmr.q, mmr.k, mmr.planets_names) == (7, 9, 6, ['Neptune'])
        assert mmr.retrograde is True and TwoBody('1J-1').retrograde is False
        assert mmr.mmr_type == 'two-body-retrograde' and mmr.type == 'mmr'

    def test_built_from_numbers(self):
        mmr = RetrogradeTwoBody(p=1, q=3, k=0, planet='Jupiter')
        assert mmr.to_s() == mmr.to_short() == '1J-3r0'

    @pytest.mark.parametrize(
        'notation, p, q, k, planet',
        [
            ('1J-3r0', 1, 3, 0, 'Jupiter'),
            ('1J-1r1', 1, 1, 1, 'Jupiter'),
            ('7N-9r6', 7, 9, 6, 'Neptune'),
            ('1S-1r0', 1, 1, 0, 'Saturn'),
            ('2S-3r0', 2, 3, 0, 'Saturn'),
            ('3J-1r1', 3, 1, 1, 'Jupiter'),
        ],
    )
    def test_notation_round_trip(self, notation, p, q, k, planet):
        mmr = resonances.create_resonance(notation)
        assert isinstance(mmr, RetrogradeTwoBody)
        assert (mmr.p, mmr.q, mmr.k, mmr.planets_names) == (p, q, k, [planet])
        assert resonances.create_resonance(mmr.to_s()).to_s() == mmr.to_s() == notation
        assert resonances.detect_resonance_type(notation) == 'mmr'

    @pytest.mark.parametrize(
        'notation, message',
        [
            ('1J-3r3', '2k <= p'),
            ('2J-4r0', 'gcd'),
            ('0J-3r0', 'positive'),
            ('1X-3r0', 'Bad notation'),
            ('1J-3r', 'k is required'),
            ('1J+3r0', '1J-3r0'),
            ('1J-3R0', 'Bad retrograde resonance notation'),
        ],
    )
    def test_bad_notation(self, notation, message):
        with pytest.raises(Exception, match=message):
            resonances.create_resonance(notation)

    @pytest.mark.parametrize('notation, cls', [('1J-3', TwoBody), ('3J-2', TwoBody), ('4J-2S-1', ThreeBody), ('1J+1+0-2', TwoBody)])
    def test_prograde_notation_is_unchanged(self, notation, cls):
        assert type(resonances.create_mmr(notation)) is cls

    def test_a_list_of_numbers_is_still_a_prograde_resonance(self):
        assert type(resonances.create_mmr([3, -1, 0, -2], ['Jupiter'])) is TwoBody


class TestCandidates:
    ELEMENTS = dict(a=10.76, e=0.79, Omega=4.88, omega=3.88, M=4.58)

    def test_a_retrograde_body_gets_only_retrograde_resonances(self):
        mmrs = candidate_mmrs({**self.ELEMENTS, 'inc': 2.26}, planets=['Jupiter'])
        assert mmrs and all(isinstance(m, RetrogradeTwoBody) for m in mmrs)
        assert {'1J-3r0', '1J-3r1', '1J-3r2'} <= {m.to_s() for m in mmrs}

    def test_a_prograde_body_gets_no_retrograde_resonance(self):
        mmrs = candidate_mmrs({**self.ELEMENTS, 'inc': 0.3}, planets=['Jupiter', 'Saturn'])
        assert mmrs and not any(m.retrograde for m in mmrs)

    def test_the_threshold_is_90_degrees(self):
        below = candidate_mmrs({**self.ELEMENTS, 'inc': np.pi / 2 - 1e-6}, planets=['Jupiter'], sigma2=1.0)
        above = candidate_mmrs({**self.ELEMENTS, 'inc': np.pi / 2 + 1e-6}, planets=['Jupiter'])
        assert not any(m.retrograde for m in below)
        assert all(m.retrograde for m in above)


class TestSimulation:
    @pytest.mark.slow
    def test_the_integrated_angle_is_built_from_omega_omega_and_m(self):
        """Through the whole pipeline: the stored angle equals q M + (2k-p) omega + p Omega - p lam_J
        from the saved series of the body and of Jupiter."""
        sim = tools.create_test_simulation_for_solar_system()
        sim.config.save_planets = True
        elem = dict(a=10.76, e=0.79, inc=2.26, Omega=4.88, omega=3.88, M=4.58, epoch=60000.0)
        sim.add_body(elem, ['1J-3r1', '1J-3'], name='retro')
        sim.run()
        body = sim.bodies[0]
        lam_j = sim.integration_engine.planets_data['Jupiter']['l']
        expected = 3 * body.M + (2 - 1) * body.omega + body.Omega - lam_j
        assert np.abs(wrap(body.angles_unwrapped['1J-3r1'] - expected)).max() < 1e-9
        summary = sim.data_manager.get_simulation_summary(sim.bodies)
        assert '1J-3r1' in set(summary['resonance'])

    def test_a_mismatched_orbit_direction_is_warned_not_refused(self, caplog):
        sim = tools.create_test_simulation_for_solar_system()
        with caplog.at_level('WARNING'):
            sim.add_body(dict(a=10.76, e=0.79, inc=0.3, Omega=4.88, omega=3.88, M=4.58, epoch=60000.0), '1J-3r0', name='pro')
        assert 'prograde orbit' in caplog.text and '1J-3r0' in caplog.text
        assert sim.bodies[0].mmrs[0].to_s() == '1J-3r0'
