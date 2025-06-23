import resonances.finder.secular_finder
import numpy as np


class TestSecularFinderCheck:
    """Test the secular_finder.check function parameter propagation."""

    def test_check_function_kwargs_propagation(self):
        """Test that old parameters passed via kwargs are correctly set up in simulation object."""

        # Define the old parameters that were previously in the function signature
        old_params = {
            'integrator': 'SABA(10,6,4)',
            'dt': 5.0,
            'Nout': 10000,
            'oscillations_cutoff': 0.00005,
            'plot': 'all',
            'save': 'all',
        }

        # Call check function with these parameters as kwargs
        sim = resonances.finder.secular_finder.check(asteroids=759, resonance='nu6', integration_years=200000, **old_params)

        # Verify that the simulation object has the parameters correctly set
        assert sim.config.integrator == old_params['integrator']
        assert sim.config.dt == old_params['dt']
        assert sim.config.Nout == old_params['Nout']
        assert sim.config.oscillations_cutoff == old_params['oscillations_cutoff']
        assert sim.config.plot == old_params['plot']
        assert sim.config.save == old_params['save']

        # Verify integration years parameter
        expected_tmax = int(200000 * 2 * np.pi)
        assert sim.config.tmax == expected_tmax

        # Verify the simulation has been set up correctly
        assert sim.config.name == "secular_check"
        assert len(sim.bodies) == 1  # One body (asteroid 759) should be added

    def test_check_function_default_params_override(self):
        """Test that kwargs parameters override default values."""

        # Use different values from defaults
        custom_params = {
            'integrator': 'WHFAST',
            'dt': 1.0,
            'Nout': 5000,
            'oscillations_cutoff': 0.001,
            'plot': 'resonant',
            'save': 'nonzero',
        }

        sim = resonances.finder.secular_finder.check(
            asteroids=760, resonance='g-g6', name="custom_secular_test", integration_years=100000, **custom_params
        )

        # Verify custom parameters were applied
        assert sim.config.integrator == custom_params['integrator']
        assert sim.config.dt == custom_params['dt']
        assert sim.config.Nout == custom_params['Nout']
        assert sim.config.oscillations_cutoff == custom_params['oscillations_cutoff']
        assert sim.config.plot == custom_params['plot']
        assert sim.config.save == custom_params['save']

        # Verify custom name was used
        assert sim.config.name == "custom_secular_test"

    def test_check_function_libration_params_override(self):
        """Test that libration-specific kwargs parameters are correctly set."""

        # Test the libration parameters that are explicitly handled in the check function
        sim = resonances.finder.secular_finder.check(
            asteroids=[759, 760],
            resonance='nu6',
            integration_years=200000,
            libration_period_min=5000,
            libration_period_critical=40000,
        )

        # Verify libration parameters were set correctly
        assert sim.config.libration_period_min == 5000
        assert sim.config.libration_period_critical == 40000

        # Verify multiple asteroids were processed
        assert len(sim.bodies) == 2  # Two bodies (asteroids 759, 760) should be added

    def test_check_function_partial_kwargs(self):
        """Test that only some kwargs parameters can be passed while others use defaults."""

        partial_params = {
            'integrator': 'SABA(8,6,4)',
            'save': 'resonant',
            # Omitting dt, Nout, oscillations_cutoff, plot intentionally
        }

        sim = resonances.finder.secular_finder.check(asteroids=1222, resonance='nu16', integration_years=150000, **partial_params)

        # Verify specified parameters were applied
        assert sim.config.integrator == partial_params['integrator']
        assert sim.config.save == partial_params['save']

        # Verify that other parameters have reasonable default values
        # (These would come from config defaults or Simulation defaults)
        assert sim.config.dt is not None
        assert sim.config.Nout is not None
        assert sim.config.oscillations_cutoff is not None
        assert sim.config.plot is not None

    def test_check_function_integration_engine_config_reference(self):
        """Test that integration engine maintains reference to the same config object."""

        test_params = {
            'integrator': 'SABA(10,6,4)',
            'dt': 2.5,
        }

        sim = resonances.finder.secular_finder.check(asteroids=759, resonance='nu6', integration_years=100000, **test_params)

        # Verify integration engine has reference to same config object
        assert sim.integration_engine.config is sim.config
        assert sim.integration_engine.config.integrator == test_params['integrator']
        assert sim.integration_engine.config.dt == test_params['dt']
