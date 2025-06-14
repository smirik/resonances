import resonances
from tests import tools


def test_backward_integration():
    """
    Test backward integration functionality.

    This test verifies that the simulation can run with negative time values
    to perform backward integration, which is useful for studying the past
    evolution of celestial bodies.
    """
    # Create simulation with backward integration parameters
    sim = resonances.Simulation(
        tmax=-600000, dt=-1.0, save='all', save_summary=True  # Negative time for backward integration  # Negative timestep
    )

    # Create solar system first
    sim.create_solar_system()

    # Add a test asteroid
    tools.add_test_asteroid_to_simulation(sim)

    # Run the backward integration
    sim.run(progress=True)

    # Verify the simulation ran successfully by checking the summary
    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    assert len(summary) > 0
    assert 'name' in summary.columns

    # Verify that we have data for the asteroid
    assert len(sim.bodies) == 1
    assert sim.bodies[0].name == 'asteroid'

    # Verify that times are negative (backward integration)
    assert sim.times[0] == 0.0  # Should start at 0
    assert sim.times[-1] < 0  # Should end at negative time
