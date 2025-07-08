import resonances.finder.mmr_finder


class TestFinderCheck:
    """Test the finder.check function parameter propagation."""

    def test_check_function_kwargs_propagation(self):
        """Test that parameters passed via kwargs are correctly set up in simulation object."""

        # Define parameters that should be propagated to the simulation
        params = {
            'integrator': 'WHFAST',
            'dt': 5.0,
            'tmax': 100000,
            'Nout': 5000,
            'oscillations_cutoff': 0.00005,
            'plot': 'all',
            'save': 'all',
        }

        # Call check function with these parameters as kwargs
        sim = resonances.finder.mmr_finder.check(asteroids=463, resonance='4J-2S-1', **params)

        # Verify that the simulation object has the parameters correctly set
        assert sim.config.integrator == params['integrator']
        assert sim.config.dt == params['dt']
        assert sim.config.tmax == params['tmax']
        assert sim.config.Nout == params['Nout']
        assert sim.config.oscillations_cutoff == params['oscillations_cutoff']
        assert sim.config.plot == params['plot']
        assert sim.config.save == params['save']

        # Verify the simulation has been set up correctly
        assert sim.config.name == "mmr_check"
        assert len(sim.bodies) == 1  # One body (asteroid 463) should be added

    def test_check_function_custom_name(self):
        """Test that custom name parameter works correctly."""

        custom_name = "test_mmr_simulation"

        sim = resonances.finder.mmr_finder.check(
            asteroids=463,
            resonance='4J-2S-1',
            name=custom_name,
            integrator='SABA(10,6,4)',
            dt=1.0,
        )

        # Verify custom name was used
        assert sim.config.name == custom_name
        assert sim.config.integrator == 'SABA(10,6,4)'
        assert sim.config.dt == 1.0

    def test_check_function_multiple_asteroids(self):
        """Test that multiple asteroids can be processed."""

        asteroids = [463, 1222, 759]

        sim = resonances.finder.mmr_finder.check(
            asteroids=asteroids,
            resonance='3J-2S-1',
            name="multi_asteroid_test",
            integrator='WHFAST',
            dt=2.0,
        )

        # Verify multiple asteroids were processed
        assert len(sim.bodies) == len(asteroids)
        assert sim.config.name == "multi_asteroid_test"
        assert sim.config.integrator == 'WHFAST'
        assert sim.config.dt == 2.0

    def test_check_function_string_resonance(self):
        """Test that string resonance parameter is correctly converted to MMR object."""

        sim = resonances.finder.mmr_finder.check(asteroids=463, resonance='4J-2S-1', name="string_resonance_test")

        # Verify that the resonance was added
        assert len(sim.bodies) == 1
        body = sim.bodies[0]
        assert len(body.mmrs) == 1
        assert '4J-2S-1' in body.mmrs[0].to_s()  # Check that core resonance is present in the string

    def test_check_function_mmr_object_resonance(self):
        """Test that MMR object resonance parameter works correctly."""

        mmr = resonances.create_mmr('3J-2S-1')

        sim = resonances.finder.mmr_finder.check(asteroids=463, resonance=mmr, name="mmr_object_test")

        # Verify that the resonance was added
        assert len(sim.bodies) == 1
        body = sim.bodies[0]
        assert len(body.mmrs) == 1
        assert '3J-2S-1' in body.mmrs[0].to_s()  # Check that core resonance is present in the string

    def test_check_function_partial_kwargs(self):
        """Test that only some kwargs parameters can be passed while others use defaults."""

        partial_params = {
            'integrator': 'SABA(10,6,4)',
            'save': 'resonant',
            # Omitting dt, tmax, Nout, oscillations_cutoff, plot intentionally
        }

        sim = resonances.finder.mmr_finder.check(asteroids=463, resonance='2J-1S-1', **partial_params)

        # Verify specified parameters were applied
        assert sim.config.integrator == partial_params['integrator']
        assert sim.config.save == partial_params['save']

        # Verify that other parameters have reasonable default values
        assert sim.config.dt is not None
        assert sim.config.tmax is not None
        assert sim.config.Nout is not None
        assert sim.config.oscillations_cutoff is not None
        assert sim.config.plot is not None

    def test_check_function_integration_engine_config_reference(self):
        """Test that integration engine maintains reference to the same config object."""

        test_params = {
            'integrator': 'WHFAST',
            'dt': 2.5,
        }

        sim = resonances.finder.mmr_finder.check(asteroids=463, resonance='4J-2S-1', **test_params)

        # Verify integration engine has reference to same config object
        assert sim.integration_engine.config is sim.config
        assert sim.integration_engine.config.integrator == test_params['integrator']
        assert sim.integration_engine.config.dt == test_params['dt']
