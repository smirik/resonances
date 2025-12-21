import pytest
import resonances


@pytest.mark.slow
def test_find():
    asteroids = [463, 490]
    planets = ['Jupiter', 'Saturn']

    sim = resonances.find(asteroids, planets, integration_years=40000)

    assert isinstance(sim, resonances.Simulation)
    assert 2 == len(sim.bodies)

    sim.run(progress=True)
    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    status463 = summary.loc[(summary['name'] == '463') & (summary['resonance'] == '4J-2S-1+0+0-1'), 'status'].iloc[0]
    status490 = summary.loc[(summary['name'] == '490') & (summary['resonance'] == '5J-2S-2+0+0-1'), 'status'].iloc[0]

    assert 2 == status463
    assert 1 == status490


@pytest.mark.slow
def test_check():
    sim = resonances.check(463, resonance='4J-2S-1')
    sim.config.tmax = 200000  # enough for 463
    assert isinstance(sim, resonances.Simulation)
    assert 1 == len(sim.bodies)

    sim.run()

    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    status = summary.loc[(summary['name'] == '463') & (summary['resonance'] == '4J-2S-1+0+0-1'), 'status'].iloc[0]
    assert 2 == status
