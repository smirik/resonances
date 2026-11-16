import pytest
import resonances
from resonances.resonance.analysis.verdict import Status
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
        date='2023-02-25',  # MJD 60000, the epoch of the sample elements
    )

    sim.create_solar_system()
    tools.add_test_asteroid_to_simulation(sim)
    sim.run(progress=True)
    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    assert len(summary) > 0
    assert 'name' in summary.columns

    # Verify that we have data for the asteroid
    assert len(sim.bodies) == 1
    assert sim.bodies[0].name == 'asteroid'

    # Verify that times are negative (backward integration)
    assert sim.times[0] == 0.0  # Should start at 0
    assert sim.times[-1] < 0  # Should end at negative time


@pytest.mark.slow
def test_backward_lola_librates():
    """463 Lola librates in 4J-2S-1 (resonant forward, tests/resonances/mmr/test_real_mmrs.py); 40 kyr into the
    past must give the same status and a libration period in the same range."""
    sim = resonances.check(
        463,
        '4J-2S-1',
        name='backward_lola',
        source='nasa',
        date='2026-09-29',  # fixed: the slow tests drift with the Horizons epoch otherwise
        integration_years=-40000,
        dt=-1.0,
        save='none',
        plot='none',
    )
    sim.run()

    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    row = summary.loc[(summary['name'] == '463') & (summary['resonance'] == '4J-2S-1+0+0-1')].iloc[0]
    assert sim.times[-1] < 0
    assert row['status'] == Status.RESONANT
    assert 1000 < row['geo_period_cycle_yr'] < 30000
