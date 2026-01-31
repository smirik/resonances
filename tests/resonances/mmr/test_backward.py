import pytest
import resonances
from tests import tools


@pytest.mark.slow
def test_backward_integration():
    """
    Test backward integration functionality.

    This test verifies that the simulation can run with negative time values
    to perform backward integration, which is useful for studying the past
    evolution of celestial bodies.
    """
    # Create simulation with backward integration parameters
    sim = resonances.Simulation(
        tmax=-50000,
        name='backward',
        save='all',
        integrator='SABA(10,6,4)',
        dt=-5.0,
    )

    sim.create_solar_system()
    tools.add_test_asteroid_to_simulation(sim)
    sim.run(progress=True)
    summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)
    assert len(summary) > 0
    assert 'name' in summary.columns

    # Verify that we have data for the asteroid
    assert len(sim.bodies) == 1
    assert sim.bodies[0].name == 'asteroid'

    # Verify that times are negative (backward integration)
    assert sim.times[0] == 0.0  # Should start at 0
    assert sim.times[-1] < 0  # Should end at negative time
