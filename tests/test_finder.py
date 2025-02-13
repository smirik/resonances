import resonances
from resonances.finder import convert_input_to_list


def test_convert_input_to_list():
    # Test case 1: asteroids as an integer
    asteroids = 1
    expected_output = [1]
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = '2'
    expected_output = ['2']
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = [1, 2, 3]
    expected_output = [1, 2, 3]
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = ['4', '5', '6']
    expected_output = ['4', '5', '6']
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = [7, '8', 9, '10']
    expected_output = [7, '8', 9, '10']
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = []
    expected_output = []
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = None
    expected_output = []
    assert convert_input_to_list(asteroids) == expected_output


# def test_find():
#     asteroids = [1, 2]
#     planets = ['Jupiter', 'Saturn']

#     sim = resonances.find(asteroids, planets)

#     assert isinstance(sim, resonances.Simulation)
#     assert 2 == len(sim.bodies)

#     sim = resonances.find(asteroids)
#     assert 2 == len(sim.bodies)

#     sim = resonances.find(asteroids[0])
#     assert 1 == len(sim.bodies)


def test_check():
    asteroids = [1, 2, '3', 4]
    mmr = resonances.create_mmr('2J-1')

    sim = resonances.check(asteroids, mmr)
    assert isinstance(sim, resonances.Simulation)
    assert 4 == len(sim.bodies)

    asteroids = [1, 2]
    sim = resonances.check(asteroids, '2J-1')
    assert isinstance(sim, resonances.Simulation)
    assert 2 == len(sim.bodies)


def test_find_resonances():
    mmrs = resonances.find_resonances(a=2.39)

    mmrs_s = []
    for mmr in mmrs:
        mmrs_s.append(mmr.to_short())

    assert '4J-2S-1' in mmrs_s
