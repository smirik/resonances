import resonances
import astdys


def find(asteroids, planets=None):
    sim = resonances.Simulation()
    sim.create_solar_system()
    if isinstance(asteroids, str) or isinstance(asteroids, int):
        asteroids = [asteroids]

    elems = astdys.search(asteroids)
    for asteroid in asteroids:
        elem = elems[str(asteroid)]
        mmrs = resonances.ThreeBodyMatrix.find_resonances(elem['a'], planets=planets)
        mmrs2 = resonances.TwoBodyMatrix.find_resonances(elem['a'], planets=planets)
        mmrs = mmrs + mmrs2
        sim.add_body(elem, mmrs, f"{asteroid}")
        resonances.logger.info('Adding a possible resonance for an asteroid {} - {}'.format(asteroid, ', '.join(map(str, elems.values()))))

    # default settings
    sim.dt = 1
    sim.plot = False
    sim.plot_only_identified = True
    return sim
