import time
import resonances
from .tools import set_fast_integrator, reset_fast_integrator


def test_backward_integration():
    asteroids = [463, 490]
    planets = ['Jupiter', 'Saturn']

    set_fast_integrator()

    sim = resonances.find(asteroids, planets)
    sim.dt = -1.0
    sim.tmax = -100000
    sim.save = 'none'
    sim.plot = 'none'
    # timestamp = int(time.time())
    # sim.save_path = f'cache/test_{timestamp}'
    # sim.plot_path = f'cache/test_{timestamp}'

    assert isinstance(sim, resonances.Simulation)
    assert 2 == len(sim.bodies)

    sim.run(progress=True)
    summary = sim.get_simulation_summary()
    status463 = summary.loc[(summary['name'] == '463') & (summary['mmr'] == '4J-2S-1+0+0-1'), 'status'].iloc[0]

    assert 2 == status463

    reset_fast_integrator()
