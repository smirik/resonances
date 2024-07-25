import resonances
import astdys
from resonances.matrix.three_body_matrix import ThreeBodyMatrix
from resonances.matrix.two_body_matrix import TwoBodyMatrix


def test_trojans():
    asteroids = [624, 588, 617]

    sim = resonances.Simulation()
    sim.create_solar_system()

    num = 624
    astdys_elem = astdys.search(str(num))

    mmrs = TwoBodyMatrix.find_resonances(astdys_elem['a'], sigma=0.1, planets=['Jupiter'])
    for mmr in mmrs:
        for asteroid in asteroids:
            sim.add_body(num, mmr, name='{}, resonance={}'.format(str(asteroid), mmr.to_short()))
            print(f"asteroid={asteroid}, MMR={mmr.to_short()}")

    sim.dt = 1
    sim.plot = False
    sim.run()
    summary = sim.get_simulation_summary()

    print(summary)

    assert 2 == summary.loc[summary['name'] == '624, resonance=1J-1', 'status'].values[0]
    assert 0 == summary.loc[summary['name'] == '624, resonance=1J+1', 'status'].values[0]
    assert 2 == summary.loc[summary['name'] == '588, resonance=1J-1', 'status'].values[0]
    assert 0 == summary.loc[summary['name'] == '588, resonance=1J+1', 'status'].values[0]
    assert 2 == summary.loc[summary['name'] == '617, resonance=1J-1', 'status'].values[0]
    assert 0 == summary.loc[summary['name'] == '617, resonance=1J+1', 'status'].values[0]


def test_3body():
    asteroids = [463]

    sim = resonances.Simulation()
    sim.create_solar_system()

    num = asteroids[0]
    astdys_elem = astdys.search(str(num))

    mmrs = ThreeBodyMatrix.find_resonances(astdys_elem['a'], sigma=0.1, planets=['Jupiter', 'Saturn'])
    for mmr in mmrs:
        for asteroid in asteroids:
            sim.add_body(num, mmr, name='{}, resonance={}'.format(str(asteroid), mmr.to_short()))
            print(f"asteroid={asteroid}, MMR={mmr.to_short()}")

    sim.dt = 1
    sim.plot = False
    sim.run()
    summary = sim.get_simulation_summary()

    print(summary)
    assert 2 == summary.loc[summary['name'] == '463, resonance=4J-2S-1', 'status'].values[0]
    assert 0 == summary.loc[summary['name'] == '463, resonance=5J-4S-1', 'status'].values[0]
