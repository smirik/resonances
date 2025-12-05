import numpy as np

from resonances import Body, Simulation
from resonances.resonance.secular import GeneralSecularResonance


def test_simulation_rebuilds_proper_secular_angles():
    sim = Simulation(secular_angle_mode='proper', tmax=100)
    sim.times = np.linspace(0, 100, 200)

    body = Body()
    resonance = GeneralSecularResonance(formula='g-g6+s-s6')
    body.secular_resonances.append(resonance)
    body.setup_vars_for_simulation(len(sim.times))

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

    body.secular_angles[resonance.to_s()] = (varpi + Omega) % (2.0 * np.pi)

    sim._rebuild_proper_secular_angles(body)

    assert resonance.to_s() in body.secular_angles_proper
    assert body.secular_angles[resonance.to_s()].shape == sim.times.shape
