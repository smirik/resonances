import pytest
import resonances
import astdys
from resonances.mmr.three_body_matrix import ThreeBodyMatrix


@pytest.mark.slow
def test_find():
    sim = resonances.find([463, 490], ['Jupiter', 'Saturn'], save="all", plot="all", integration_years=40000, type=["mmr"])
    sim.run()

    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    status = summary.loc[(summary['name'] == '463') & (summary['resonance'] == '4J-2S-1+0+0-1'), 'status'].iloc[0]
    assert 2 == status
    assert 1 == summary.loc[(summary['name'] == '490') & (summary['resonance'] == '5J-2S-2+0+0-1'), 'status'].iloc[0]


@pytest.mark.slow
def test_trojans():
    asteroids = [624, 588, 617]

    sim = resonances.find(asteroids, ['Jupiter'], type="mmr")
    sim.run()
    summary = sim.data_manager.get_simulation_summary(sim.bodies)

    assert 2 == summary.loc[(summary['resonance'] == '1J-1+0+0') & (summary['name'] == '624'), 'status'].iloc[0]
    assert 0 == summary.loc[(summary['resonance'] == '1J+1+0-2') & (summary['name'] == '624'), 'status'].iloc[0]

    assert 2 == summary.loc[(summary['resonance'] == '1J-1+0+0') & (summary['name'] == '588'), 'status'].iloc[0]
    assert 0 == summary.loc[(summary['resonance'] == '1J+1+0-2') & (summary['name'] == '588'), 'status'].iloc[0]

    assert 2 == summary.loc[(summary['resonance'] == '1J-1+0+0') & (summary['name'] == '617'), 'status'].iloc[0]
    assert 0 == summary.loc[(summary['resonance'] == '1J+1+0-2') & (summary['name'] == '617'), 'status'].iloc[0]


@pytest.mark.slow
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

    sim.run()
    summary = sim.data_manager.get_simulation_summary(sim.bodies)

    assert 2 == summary.loc[summary['name'] == '463, resonance=4J-2S-1', 'status'].values[0]
    assert 0 == summary.loc[summary['name'] == '463, resonance=5J-4S-1', 'status'].values[0]
