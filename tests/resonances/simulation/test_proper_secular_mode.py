import numpy as np

from resonances import Body, Simulation
from resonances.secular.secular_resonance import SecularResonance


def test_simulation_rebuilds_proper_secular_angles():
    sim = Simulation(secular_angle_mode='proper', tmax=100)
    sim.times = np.linspace(0, 100, 200)

    body = Body()
    resonance = SecularResonance('g-g6+s-s6')
    body.secular_resonances.append(resonance)
    body.setup_vars_for_simulation(sim.times)

    times_years = sim.times / (2.0 * np.pi)
    g_true = 0.04
    s_true = -0.02
    varpi = g_true * times_years
    Omega = s_true * times_years
    omega = varpi - Omega

    body.ecc[:] = 0.1
    body.inc[:] = np.deg2rad(10.0)
    body.Omega[:] = Omega
    body.omega[:] = omega
    body.varpi[:] = varpi

    body.angles[resonance.to_s()] = (varpi + Omega) % (2.0 * np.pi)

    body.build_proper_angle(resonance)

    assert resonance.to_s() in body.secular_angles_proper
    assert body.angles[resonance.to_s()].shape == sim.times.shape


def _body_with_osculating_angle(sim, resonance):
    """A body whose integration angle is unrelated to its elements, as for a planet whose osculating
    longitude does not follow its mode."""
    body = Body()
    body.secular_resonances.append(resonance)
    body.setup_vars_for_simulation(sim.times)
    t = sim.times / (2.0 * np.pi)
    body.ecc[:] = 0.1
    body.inc[:] = np.deg2rad(10.0)
    body.omega[:] = 0.0
    body.Omega[:] = np.deg2rad(-30.0 / 3600.0) * t + 1.0
    body.varpi[:] = body.Omega
    body.axis[:] = 2.5
    body.angles_unwrapped[resonance.to_s()] = 0.3 * np.sin(t / 1e4)  # librating 'osculating' angle
    return body


def test_classifier_reads_the_proper_secular_angle():
    sim = Simulation(secular_angle_mode='proper', tmax=1e6, date='2000-01-01 12:00')
    sim.times = np.linspace(0, 2 * np.pi * 1e6, 1001)
    resonance = SecularResonance('s-s6')
    body = _body_with_osculating_angle(sim, resonance)
    sim.body_manager.bodies = [body]

    sim.prepare_angles()

    key = resonance.to_s()
    t = sim.times / (2.0 * np.pi)
    # s = -30, s6 from the package: the classified angle turns at s - s6 (~ -3.65 arcsec/yr)
    rate = np.polyfit(t, body.angles_unwrapped[key], 1)[0] * 206264.806
    from resonances.data.const import PLANETARY_FREQUENCIES

    assert np.isclose(rate, -30.0 - PLANETARY_FREQUENCIES['s6'], atol=0.05)
    assert np.allclose(body.angles[key], body.secular_angles_proper[key])
    # the reference series is the same angle from the osculating elements, not the integration's one:
    # it turns at the same rate as the reduced one
    osculating_rate = np.polyfit(t, np.unwrap(body.secular_angles_osculating[key]), 1)[0] * 206264.806
    assert np.isclose(osculating_rate, rate, atol=0.05)
    assert not np.allclose(body.secular_angles_osculating[key], np.mod(0.3 * np.sin(t / 1e4), 2 * np.pi))


def test_osculating_mode_keeps_the_integration_angle():
    sim = Simulation(secular_angle_mode='osculating', tmax=1e6, date='2000-01-01 12:00')
    sim.times = np.linspace(0, 2 * np.pi * 1e6, 1001)
    resonance = SecularResonance('s-s6')
    body = _body_with_osculating_angle(sim, resonance)
    sim.body_manager.bodies = [body]

    sim.prepare_angles()

    t = sim.times / (2.0 * np.pi)
    assert np.allclose(body.angles_unwrapped[resonance.to_s()], 0.3 * np.sin(t / 1e4))
