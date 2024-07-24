import resonances
from resonances.data.astdys import astdys


def find(asteroids, planets=None):
    sim = resonances.Simulation()
    sim.create_solar_system()
    if isinstance(asteroids, str):
        asteroids = [asteroids]

    for asteroid in asteroids:
        elem = astdys.search(asteroid)
        mmrs = resonances.ThreeBodyMatrix.find_resonances(elem['a'], planets=planets)
        mmrs2 = resonances.TwoBodyMatrix.find_resonances(elem['a'], planets=planets)
        mmrs = mmrs + mmrs2
        for mmr in mmrs:
            resonances.logger.info('Adding a possible resonance: {}'.format(mmr.to_short()))
            sim.add_body(elem, mmr, '{};{}'.format(asteroid, mmr.to_short()))

    # default settings
    sim.dt = 1
    sim.plot = False
    sim.plot_only_identified = True
    return sim
