import resonances
import astdys
from resonances.matrix.three_body_matrix import ThreeBodyMatrix
from .tools import set_fast_integrator, reset_fast_integrator


def test_find():
    set_fast_integrator()
    sim = resonances.find(463, ['Jupiter', 'Saturn'], sigma3=0.1)
    sim.run()

    summary = sim.get_simulation_summary()

    assert 2 == summary.loc[summary['mmr'] == '4J-2S-1+0+0-1', 'status'].values[0]
    assert 0 == summary.loc[summary['mmr'] == '5J-4S-1+0+0+0', 'status'].values[0]

    reset_fast_integrator()


def test_trojans():
    set_fast_integrator()
    asteroids = [624, 588, 617]

    sim = resonances.find(asteroids, ['Jupiter'])
    sim.run()
    summary = sim.get_simulation_summary()

    assert 2 == summary.loc[(summary['mmr'] == '1J-1+0+0') & (summary['name'] == '624'), 'status'].iloc[0]
    assert 0 == summary.loc[(summary['mmr'] == '1J+1+0-2') & (summary['name'] == '624'), 'status'].iloc[0]

    assert 2 == summary.loc[(summary['mmr'] == '1J-1+0+0') & (summary['name'] == '588'), 'status'].iloc[0]
    assert 0 == summary.loc[(summary['mmr'] == '1J+1+0-2') & (summary['name'] == '588'), 'status'].iloc[0]

    assert 2 == summary.loc[(summary['mmr'] == '1J-1+0+0') & (summary['name'] == '617'), 'status'].iloc[0]
    assert 0 == summary.loc[(summary['mmr'] == '1J+1+0-2') & (summary['name'] == '617'), 'status'].iloc[0]

    reset_fast_integrator()


def test_3body():
    set_fast_integrator()

    asteroids = [463]

    sim = resonances.Simulation()
    sim.create_solar_system()

    num = asteroids[0]
    astdys_elem = astdys.search(str(num))

    mmrs = ThreeBodyMatrix.find_resonances(astdys_elem['a'], sigma=0.1, planets=['Jupiter', 'Saturn'])
    for mmr in mmrs:
        for asteroid in asteroids:
            sim.add_body(num, mmr, name='{}, resonance={}'.format(str(asteroid), mmr.to_short()))

    sim.run()
    summary = sim.get_simulation_summary()

    assert 2 == summary.loc[summary['name'] == '463, resonance=4J-2S-1', 'status'].values[0]
    assert 0 == summary.loc[summary['name'] == '463, resonance=5J-4S-1', 'status'].values[0]

    reset_fast_integrator()
