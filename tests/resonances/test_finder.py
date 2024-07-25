import resonances


def test_find():
    asteroids = [1, 2]
    planets = ['Jupiter', 'Saturn']

    sim = resonances.find(asteroids, planets)

    assert isinstance(sim, resonances.Simulation)
    assert 2 == len(sim.bodies)

    sim = resonances.find(asteroids)
    assert 2 == len(sim.bodies)

    sim = resonances.find(asteroids[0])
    assert 1 == len(sim.bodies)
