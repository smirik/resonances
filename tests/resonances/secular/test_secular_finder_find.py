import resonances.secular_finder
import resonances
import numpy as np
import pytest
from resonances.matrix.secular_matrix import SecularMatrix

from tests.resonances.secular import BASIC_CONFIG


@pytest.fixture(autouse=True)
def setup_test_config():
    """Setup test configuration before each test and restore after."""
    original_save_path = resonances.config.get('SAVE_PATH')
    original_plot_path = resonances.config.get('PLOT_PATH')

    resonances.config.set('SAVE_PATH', 'cache/tests')
    resonances.config.set('PLOT_PATH', 'cache/tests')

    yield

    resonances.config.set('SAVE_PATH', original_save_path)
    resonances.config.set('PLOT_PATH', original_plot_path)


class TestSecularFinderFind:
    """Test the secular_finder.find function."""

    def test_find_all_secular_resonances(self):
        """Test finding all secular resonances for asteroid 759 (equivalent to old test_759_all_secular_resonances)."""

        # Use find method to get all secular resonances
        sim = resonances.secular_finder.find(
            asteroids=759,
            name="test_759_all_secular",
            **BASIC_CONFIG,
        )

        # Verify simulation setup
        assert sim.config.name == "test_759_all_secular"
        assert len(sim.bodies) == 1
        assert sim.bodies[0].name == "759"

        # Check that multiple secular resonances were added
        assert len(sim.bodies[0].secular_resonances) > 1
        print(f"Testing asteroid 759 with {len(sim.bodies[0].secular_resonances)} secular resonances")

        # Run simulation and check results
        sim.run(progress=True)
        summary = sim.data_manager.get_simulation_summary(sim.bodies)

        non_zero_resonances = []
        for _, row in summary.iterrows():
            if row['name'] == '759':
                status = row['status']
                resonance = row['resonance']
                print(f"Resonance {resonance}: status = {status}")
                if abs(status) != 0:
                    non_zero_resonances.append((resonance, status))

        print(f"\nResonances with non-zero status: {non_zero_resonances}")

        # Verify nu6_Saturn shows expected behavior
        nu6_status = None
        for resonance_name, status in non_zero_resonances:
            if resonance_name == 'nu6_Saturn':
                nu6_status = status
                break

        assert nu6_status is not None, "Expected nu6_Saturn to have non-zero status, but it was not found"
        assert abs(nu6_status) == 2, f"Expected |status| = 2 for nu6_Saturn, got {abs(nu6_status)}"

    def test_find_specific_formulas(self):
        """Test finding specific secular resonance formulas."""

        sim = resonances.secular_finder.find(
            asteroids=[759, 760],
            formulas=['g-g5', 'g-g6'],
            name="test_specific_formulas",
            integration_years=100000,
            integrator='WHFAST',
            dt=2.0,
        )

        # Verify simulation setup
        assert sim.config.name == "test_specific_formulas"
        assert len(sim.bodies) == 2
        assert sim.config.integrator == 'WHFAST'
        assert sim.config.dt == 2.0

        # Check that only specified resonances were added
        for body in sim.bodies:
            assert len(body.secular_resonances) == 2
            # The g-g5 and g-g6 formulas create Nu5Resonance and Nu6Resonance objects
            # which have resonance_type 'nu5' and 'nu6' respectively
            resonance_types = [res.resonance_type for res in body.secular_resonances]
            assert 'nu5' in resonance_types  # g-g5 creates Nu5Resonance
            assert 'nu6' in resonance_types  # g-g6 creates Nu6Resonance

    def test_find_by_order(self):
        """Test finding secular resonances by order."""

        sim = resonances.secular_finder.find(
            asteroids=463,
            order=2,
            name="test_order_2",
            integration_years=50000,
        )

        # Verify simulation setup
        assert sim.config.name == "test_order_2"
        assert len(sim.bodies) == 1

        # Check that order 2 resonances were added
        body = sim.bodies[0]
        assert len(body.secular_resonances) > 0

        # Verify all resonances are order 2 (linear resonances)
        # Get expected order 2 formulas for comparison
        expected_order2 = SecularMatrix.build(order=2)
        assert len(body.secular_resonances) == len(expected_order2)

    def test_find_kwargs_propagation(self):
        """Test that kwargs parameters are correctly propagated to simulation."""

        params = {
            'integrator': 'SABA(10,6,4)',
            'dt': 1.5,
            'tmax': 50000,
            'oscillations_cutoff': 0.001,
            'plot': 'resonant',
            'save': 'nonzero',
        }

        sim = resonances.secular_finder.find(asteroids=759, formulas=['g-g6'], name="test_kwargs", **params)

        # Verify parameters were applied (except tmax which is overridden by integration_years)
        assert sim.config.integrator == params['integrator']
        assert sim.config.dt == params['dt']
        assert sim.config.oscillations_cutoff == params['oscillations_cutoff']
        assert sim.config.plot == params['plot']
        assert sim.config.save == params['save']

        # tmax should be set by integration_years (default 1000000), not by kwargs
        expected_tmax = int(1000000 * 2 * np.pi)
        assert sim.config.tmax == expected_tmax

    def test_find_no_resonances_found(self):
        """Test behavior when no resonances are found."""

        # This should not happen in practice, but test the edge case
        sim = resonances.secular_finder.find(asteroids=759, formulas=['nonexistent_formula'], name="test_no_resonances")

        # Should return a simulation with no bodies
        assert sim.config.name == "test_no_resonances"
        assert len(sim.bodies) == 0

    def test_find_multiple_asteroids(self):
        """Test finding resonances for multiple asteroids."""

        asteroids = [759, 1222, 760]

        sim = resonances.secular_finder.find(
            asteroids=asteroids,
            order=2,
            name="test_multiple_asteroids",
            integration_years=50000,
        )

        # Verify all asteroids were added
        assert len(sim.bodies) == len(asteroids)

        # Verify each body has the same set of resonances
        expected_resonances = len(SecularMatrix.build(order=2))
        for body in sim.bodies:
            assert len(body.secular_resonances) == expected_resonances

    def test_find_integration_engine_config_reference(self):
        """Test that integration engine maintains reference to the same config object."""

        test_params = {
            'integrator': 'WHFAST',
            'dt': 2.5,
        }

        sim = resonances.secular_finder.find(asteroids=759, formulas=['g-g5'], **test_params)

        # Verify integration engine has reference to same config object
        assert sim.integration_engine.config is sim.config
        assert sim.integration_engine.config.integrator == test_params['integrator']
        assert sim.integration_engine.config.dt == test_params['dt']
