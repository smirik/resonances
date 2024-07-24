import resonances

if __name__ == '__main__':
    print('You have installed the resonances package. Check please the documentation!')

asteroids = [463]
planets = [
    'Jupiter',
    'Saturn',
]
sim = resonances.Simulation()
sim.create_solar_system()
if isinstance(asteroids, str):
    asteroids = [asteroids]

for asteroid in asteroids:
    elem = resonances.astdys.search(asteroid)
    mmrs = resonances.ThreeBodyMatrix.find_resonances(elem['a'], planets=planets)
    mmrs2 = resonances.TwoBodyMatrix.find_resonances(elem['a'], planets=planets)
    mmrs = mmrs + mmrs2
    # resonances.logger.info('Adding a possible resonance: {}'.format(mmr.to_short()))
    sim.add_body(elem, mmrs, '{}'.format(asteroid))

# default settings
sim.dt = 1
sim.save = 'all'
sim.plot = 'nonzero'
sim.tmax = 1000
sim.run()
