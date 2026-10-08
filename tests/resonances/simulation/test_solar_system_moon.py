"""The Solar System with a separate Moon (option `solar_system_moon`).

Like the other solar-system tests, the planets come from Horizons for MJD 60000 (2023-02-25) and are cached in
cache/; the first run needs the network. The expected values are independent of the package:

- masses: the Horizons header constants BODY399_GM, BODY301_GM, BODY3_GM and BODY10_GM (km^3/s^2);
- the Earth-Moon barycentre: the 'Earth' (Horizons body 3) of the default system at the same date;
- 3753 Cruithne: heliocentric ecliptic J2000 osculating elements from a Horizons ELEMENTS query
  (CENTER 500@10, TLIST 2460000.5).
"""

import ssl

import astdys.util
import numpy as np
import pytest
import rebound

from resonances.data.const import SOLAR_SYSTEM_WITH_SUN
from resonances.simulation import Simulation, SimulationSerializer
from resonances.simulation.integration import horizons_tls

GM_SUN = 1.32712440041279419e11  # BODY10_GM
GM_EARTH = 3.9860043550702266e5  # BODY399_GM
GM_MOON = 4.9028001184575496e3  # BODY301_GM
GM_EMB = 4.0350323562548019e5  # BODY3_GM, the Earth-Moon barycentre

# Written out, not taken from resonances.data.const, so the order is checked independently.
MOON_ORDER = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto', 'Moon']

DATE = astdys.util.convert_mjd_to_datetime(60000)

# 3753 Cruithne at JD 2460000.5 TDB (Horizons, heliocentric ecliptic J2000), an Earth horseshoe object.
CRUITHNE = {
    'a': 9.977006072801362e-01,
    'e': 5.149688719057739e-01,
    'inc': np.radians(1.980377757357156e01),
    'Omega': np.radians(1.262077468480693e02),
    'omega': np.radians(4.387457816508109e01),
    'M': np.radians(7.542367787886232e01),
}


def _sim(tmp_path, **kwargs):
    return Simulation(date=DATE, save_path=str(tmp_path), plot_path=str(tmp_path), **kwargs)


@pytest.fixture(scope='module')
def systems(tmp_path_factory):
    """The default (barycentre) system and the Moon system at the same date."""
    path = tmp_path_factory.mktemp('moon')
    plain = _sim(path)
    plain.create_solar_system()
    moon = _sim(path, solar_system_moon=True, integrator='ias15')
    moon.create_solar_system()
    return plain, moon


def _state(p):
    return np.array(p.xyz), np.array(p.vxyz)


def test_default_config_is_the_barycentre_system(systems):
    plain, _ = systems
    assert plain.config.solar_system_moon is False
    sim = plain.integration_engine.sim
    assert sim.N == 10
    assert [p.hash.value for p in sim.particles] == [rebound.hash(name).value for name in SOLAR_SYSTEM_WITH_SUN]
    assert sim.particles['Earth'].m / sim.particles['Sun'].m == pytest.approx(GM_EMB / GM_SUN, rel=1e-12)


def test_moon_system_particles_in_order(systems):
    _, moon = systems
    sim = moon.integration_engine.sim
    assert sim.N == 11
    assert [p.hash.value for p in sim.particles] == [rebound.hash(name).value for name in MOON_ORDER]
    assert moon.integration_engine.massive_bodies == MOON_ORDER


def test_moon_system_masses_are_horizons_gm(systems):
    _, moon = systems
    ps = moon.integration_engine.sim.particles
    # Ratios to the Sun are the dynamically relevant numbers; the absolute masses are in solar masses.
    assert ps['Earth'].m / ps['Sun'].m == pytest.approx(GM_EARTH / GM_SUN, rel=1e-12)
    assert ps['Moon'].m / ps['Sun'].m == pytest.approx(GM_MOON / GM_SUN, rel=1e-12)
    assert ps['Earth'].m == pytest.approx(GM_EARTH / GM_SUN, abs=1e-12)
    assert ps['Moon'].m == pytest.approx(GM_MOON / GM_SUN, abs=1e-12)
    assert (ps['Earth'].m + ps['Moon'].m) / ps['Sun'].m == pytest.approx(GM_EMB / GM_SUN, rel=1e-12)


def test_earth_and_moon_centre_of_mass_is_the_barycentre(systems):
    plain, moon = systems
    ps = moon.integration_engine.sim.particles
    m_e, m_m = ps['Earth'].m, ps['Moon'].m
    (re, ve), (rm, vm) = _state(ps['Earth']), _state(ps['Moon'])
    r_emb, v_emb = _state(plain.integration_engine.sim.particles['Earth'])
    np.testing.assert_allclose((m_e * re + m_m * rm) / (m_e + m_m), r_emb, rtol=0, atol=1e-12)
    np.testing.assert_allclose((m_e * ve + m_m * vm) / (m_e + m_m), v_emb, rtol=0, atol=1e-12)
    # and the geocentre is not the barycentre: 4,671 km = 3.1e-5 AU apart
    assert 2.5e-5 < np.linalg.norm(re - r_emb) < 3.5e-5


def test_every_other_planet_is_identical(systems):
    plain, moon = systems
    a, b = plain.integration_engine.sim.particles, moon.integration_engine.sim.particles
    for name in SOLAR_SYSTEM_WITH_SUN:
        if name == 'Earth':
            continue
        assert a[name].m == b[name].m, name
        assert a[name].xyz == b[name].xyz, name
        assert a[name].vxyz == b[name].vxyz, name


def test_planet_indices_are_unchanged(systems):
    plain, moon = systems
    for name in SOLAR_SYSTEM_WITH_SUN:
        index = MOON_ORDER.index(name)
        assert moon.body_manager.get_index_of_planets([name]) == [index]
        assert plain.body_manager.get_index_of_planets([name]) == [index]
    moon.add_body(dict(CRUITHNE), ['1E-1', 'g-g5'], name='cruithne')
    body = moon.bodies[-1]
    assert body.mmrs[0].index_of_planets == [3]
    assert body.secular_resonances[0].index_of_planets == [5]
    # the Moon is a massive body, never a planet of an angle or of the saved planet series
    assert 'Moon' not in moon.integration_engine.planets
    assert 'Moon' not in moon.integration_engine.planets_data
    moon.body_manager.bodies.pop()


def test_n_active_is_the_number_of_massive_bodies(tmp_path):
    for moon_on, expected in ((False, 10), (True, 11)):
        sim = _sim(tmp_path, solar_system_moon=moon_on, integrator='ias15')
        sim.create_solar_system()
        sim.add_body(dict(CRUITHNE), '1E-1', name='cruithne')
        sim.body_manager.add_bodies_to_simulation(sim.integration_engine.sim)
        sim.integration_engine.setup_integrator()
        assert sim.integration_engine.sim.N_active == expected
        assert sim.integration_engine.sim.N == expected + 1
        assert sim.bodies[0].index_in_simulation == expected


@pytest.mark.parametrize('integrator', ['SABA(10,6,4)', 'SABA(8,6,4)', 'whfast', 'WHFast', 'mercurius', 'trace', 'leapfrog'])
def test_symplectic_integrators_are_refused_with_the_moon(tmp_path, integrator):
    with pytest.raises(ValueError, match='solar_system_moon'):
        _sim(tmp_path, solar_system_moon=True, integrator=integrator)
    # without the Moon the same integrator is accepted
    assert _sim(tmp_path, integrator=integrator).config.integrator == integrator


def test_integrator_changed_after_construction_is_refused(systems):
    _, moon = systems
    engine = moon.integration_engine
    moon.config.integrator = 'whfast'
    try:
        with pytest.raises(ValueError, match='solar_system_moon'):
            engine.setup_integrator()
        with pytest.raises(ValueError, match='solar_system_moon'):
            engine.create_solar_system()
    finally:
        moon.config.integrator = 'ias15'
    assert moon.config.solar_system_moon is True


def test_direct_call_turns_the_option_on(tmp_path):
    """Calling create_solar_system_with_moon() directly must leave the config describing the Moon system,
    so that batch workers rebuild the same model and a later symplectic integrator is refused."""
    sim = _sim(tmp_path, integrator='ias15')
    assert sim.config.solar_system_moon is False
    sim.integration_engine.create_solar_system_with_moon()
    assert sim.integration_engine.sim.N == 11
    assert sim.config.solar_system_moon is True
    assert sim.batch_manager._prepare_simulation_kwargs(sim)['solar_system_moon'] is True
    assert sim.batch_manager._get_config_dict()['solar_system_moon'] is True
    sim.config.integrator = 'whfast'
    with pytest.raises(ValueError, match='solar_system_moon'):
        sim.integration_engine.setup_integrator()

    # a direct call with a symplectic integrator is refused and leaves the option off
    plain = _sim(tmp_path, integrator='SABA(10,6,4)')
    with pytest.raises(ValueError, match='solar_system_moon'):
        plain.integration_engine.create_solar_system_with_moon()
    assert plain.config.solar_system_moon is False


def test_moon_system_has_its_own_cache_file(systems):
    plain, moon = systems
    stamp = int(DATE.timestamp())
    assert plain.integration_engine._solar_system_filename().endswith(f'/cache/solar-{stamp}.bin')
    assert moon.integration_engine._solar_system_filename(moon=True).endswith(f'/cache/solar-moon-{stamp}.bin')
    # loading the cached Moon file gives the same 11 bodies
    again = _sim('cache/tests/moon-reload', solar_system_moon=True, integrator='ias15')
    again.create_solar_system()
    first, second = moon.integration_engine.sim.particles, again.integration_engine.sim.particles
    assert again.integration_engine.sim.N == 11
    for i in range(11):
        assert first[i].xyz == second[i].xyz and first[i].m == second[i].m


def test_option_reaches_batch_workers_and_restore(tmp_path):
    sim = _sim(tmp_path, solar_system_moon=True, integrator='ias15', save='all', plot=None)
    kwargs = sim.batch_manager._prepare_simulation_kwargs(sim)
    assert kwargs['solar_system_moon'] is True
    worker = Simulation(**{**kwargs, '_skip_path_verification': True})
    assert worker.config.solar_system_moon is True
    # the resume hash changes with the Moon, and keeps its old content without it
    assert sim.batch_manager._get_config_dict()['solar_system_moon'] is True
    assert 'solar_system_moon' not in _sim(tmp_path).batch_manager._get_config_dict()

    sim.data_manager.save_configuration_details([], sim)
    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=False)
    assert restored.config.solar_system_moon is True
    assert restored.config.integrator == 'ias15'


def test_env_default_is_off(tmp_path):
    sim = _sim(tmp_path)
    assert sim.config.solar_system_moon is False
    assert sim.integration_engine.massive_bodies == SOLAR_SYSTEM_WITH_SUN


def test_horizons_tls_uses_certifi_and_restores(monkeypatch):
    import certifi
    import rebound.horizons as rh

    calls = []

    def recorder(url, *args, context=None, **kwargs):
        calls.append(context)

    monkeypatch.delenv('SSL_CERT_FILE', raising=False)
    monkeypatch.setattr(rh, 'urlopen', recorder)
    with horizons_tls():
        assert rh.urlopen is not recorder
        rh.urlopen('https://example.invalid', context=None)
        explicit = ssl.create_default_context()
        rh.urlopen('https://example.invalid', context=explicit)  # a context rebound chose itself is kept
    assert rh.urlopen is recorder

    context = calls[0]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    with open(certifi.where()) as f:
        n_certifi = f.read().count('BEGIN CERTIFICATE')
    assert context.cert_store_stats()['x509_ca'] == n_certifi
    assert calls[1] is explicit

    # a user-set SSL_CERT_FILE wins: nothing is replaced
    monkeypatch.setenv('SSL_CERT_FILE', certifi.where())
    with horizons_tls():
        assert rh.urlopen is recorder


@pytest.mark.slow
def test_earth_coorbital_run_uses_the_geocentre(tmp_path):
    """A 20-yr ias15 run of 3753 Cruithne with the Moon: the 1:1 angle is taken against the geocentre."""
    sim = _sim(tmp_path, solar_system_moon=True, integrator='ias15', dt=1.0, save=None, plot=None, save_planets=True)
    sim.config.tmax = 20 * 2 * np.pi
    sim.config.Nout = 201
    sim.config.libration_period_min = 1
    sim.create_solar_system()
    sim.add_body(dict(CRUITHNE), '1E-1', name='cruithne')
    sim.run()

    body = sim.bodies[0]
    key = '1E-1+0+0'
    assert np.all(np.isfinite(body.angles[key])) and len(body.angles[key]) == 201
    assert body.statuses[key] is not None

    # the whole series: sigma = lambda_Earth - lambda, with the Earth series the run saved
    earth = sim.integration_engine.planets_data['Earth']
    expected = np.mod(earth['l'] - body.longitude, 2 * np.pi)
    diff = np.angle(np.exp(1j * (body.angles[key] - expected)))
    assert np.max(np.abs(diff)) < 1e-12

    # the saved Earth at the end is the geocentre particle, not the Earth-Moon barycentre
    ps = sim.integration_engine.sim.particles
    sun, geo, moon = ps['Sun'], ps['Earth'], ps['Moon']
    lam_geo = geo.orbit(primary=sun).l
    assert abs(np.angle(np.exp(1j * (lam_geo - earth['l'][-1])))) < 1e-12
    m = geo.m + moon.m
    emb = rebound.Particle(
        simulation=sim.integration_engine.sim,
        m=m,
        x=(geo.m * geo.x + moon.m * moon.x) / m,
        y=(geo.m * geo.y + moon.m * moon.y) / m,
        z=(geo.m * geo.z + moon.m * moon.z) / m,
        vx=(geo.m * geo.vx + moon.m * moon.vx) / m,
        vy=(geo.m * geo.vy + moon.m * moon.vy) / m,
        vz=(geo.m * geo.vz + moon.m * moon.vz) / m,
    )
    lam_emb = emb.orbit(primary=sun, G=sim.integration_engine.sim.G).l
    assert abs(np.angle(np.exp(1j * (lam_emb - earth['l'][-1])))) > 1e-6

    # the Moon stayed bound to the Earth: distance 356,000-407,000 km
    assert 0.00235 < np.linalg.norm(np.array(moon.xyz) - np.array(geo.xyz)) < 0.00275
