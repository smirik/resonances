"""Tests for phase portrait plots."""

import numpy as np
from pathlib import Path
import tempfile

import resonances
import tests.tools as tools


class TestPhasePlotter:
    """Test suite for PhasePlotter class."""

    def test_phase_plotter_from_body(self):
        """Test creating PhasePlotter from Body object."""
        sim = tools.create_test_simulation_for_solar_system()
        sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
        sim.run()

        body = sim.bodies[0]
        mmr = body.mmrs[0]

        plotter = resonances.PhasePlotter.from_body(body, mmr, sim)
        assert plotter._times is not None
        assert plotter._sigma_filtered is not None
        assert plotter._sigma_dot_filtered is not None
        assert plotter._sigma_unfiltered is not None
        assert plotter._sigma_dot_unfiltered is not None
        assert plotter._metadata is not None
        assert plotter._metadata['body_name'] == body.name

    def test_phase_plotter_from_data(self):
        """Test creating PhasePlotter from raw data arrays."""
        times = np.linspace(0, 100000, 1000)
        sigma = np.sin(times * 0.001) * 2  # Oscillating signal

        plotter = resonances.PhasePlotter.from_data(times, sigma, body_name='Test', resonance_key='4J-2S-1')

        assert plotter._times is not None
        assert plotter._sigma_filtered is not None
        assert plotter._sigma_dot_filtered is not None
        assert plotter._sigma_unfiltered is not None
        assert plotter._sigma_dot_unfiltered is not None
        assert len(plotter._sigma_dot_filtered) == len(times)

    def test_plot_phase_portrait_filtered(self):
        """Test filtered phase portrait plotting."""
        sim = tools.create_test_simulation_for_solar_system()
        sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
        sim.run()

        body = sim.bodies[0]
        mmr = body.mmrs[0]

        plotter = resonances.PhasePlotter.from_body(body, mmr, sim)
        result = plotter.plot_phase_portrait_filtered()

        assert result is plotter  # Method chaining
        assert plotter._figure is not None
        assert plotter._ax is not None

        plotter.close()
        assert plotter._figure is None

    def test_plot_phase_portrait_unfiltered(self):
        """Test unfiltered phase portrait plotting."""
        sim = tools.create_test_simulation_for_solar_system()
        sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
        sim.run()

        body = sim.bodies[0]
        mmr = body.mmrs[0]

        plotter = resonances.PhasePlotter.from_body(body, mmr, sim)
        result = plotter.plot_phase_portrait_unfiltered()

        assert result is plotter
        assert plotter._figure is not None

        plotter.close()

    def test_plot_phase_portrait_slow(self):
        """Test slow points phase portrait plotting."""
        sim = tools.create_test_simulation_for_solar_system()
        sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
        sim.run()

        body = sim.bodies[0]
        mmr = body.mmrs[0]

        plotter = resonances.PhasePlotter.from_body(body, mmr, sim)
        result = plotter.plot_phase_portrait_slow(percentile=95)

        assert result is plotter
        assert plotter._figure is not None

        plotter.close()

    def test_save_phase_portrait(self):
        """Test saving phase portrait to file."""
        sim = tools.create_test_simulation_for_solar_system()
        sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
        sim.run()

        body = sim.bodies[0]
        mmr = body.mmrs[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'phase_portrait.png'

            plotter = resonances.PhasePlotter.from_body(body, mmr, sim)
            plotter.plot_phase_portrait().save(str(save_path))

            assert save_path.exists()
            plotter.close()


class TestPlotsConfigIntegration:
    """Test integration of plots config with simulation."""

    def test_plots_config_default(self):
        """Test default plots config is evolution only."""
        config = resonances.simulation.SimulationConfig()
        assert 'evolution' in config.plots
        assert len(config.plots) == 1

    def test_plots_config_custom_list(self):
        """Test custom plots config as list."""
        config = resonances.simulation.SimulationConfig(plots=['evolution', 'phase_portrait'])
        assert 'evolution' in config.plots
        assert 'phase_portrait' in config.plots
        assert len(config.plots) == 2

    def test_plots_config_single_string(self):
        """Test custom plots config as single string."""
        config = resonances.simulation.SimulationConfig(plots='phase_portrait')
        assert 'phase_portrait' in config.plots
        assert len(config.plots) == 1

    def test_phase_portrait_slow_percentile_config(self):
        """Test phase portrait slow percentile config."""
        config = resonances.simulation.SimulationConfig(phase_portrait_slow_percentile=90)
        assert config.phase_portrait_slow_percentile == 90

        config_default = resonances.simulation.SimulationConfig()
        assert config_default.phase_portrait_slow_percentile == 95


class TestPlotsGeneration:
    """Test actual plot file generation during simulation."""

    def test_generate_all_plot_types(self):
        """Test generating all plot types during simulation."""
        sim = resonances.Simulation(
            name='test_all_plots',
            integrator='SABA(10,6,4)',
            tmax=10000,
            save='all',
            plot='all',
            plot_type='save',
            plots=['evolution', 'phase_portrait'],
        )
        sim.create_solar_system()
        sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'), name='test_asteroid')
        sim.run()

        plot_path = Path(sim.config.plot_path)
        resonance_key = '4J-2S-1+0+0-1'

        # Check evolution plot exists
        evolution_file = plot_path / f'test_asteroid-{resonance_key}.png'
        assert evolution_file.exists(), f"Evolution plot not found: {evolution_file}"

        # Check all three phase portrait variants exist
        phase_filtered = plot_path / f'phase_filtered_test_asteroid-{resonance_key}.png'
        assert phase_filtered.exists(), f"Filtered phase portrait not found: {phase_filtered}"

        phase_unfiltered = plot_path / f'phase_unfiltered_test_asteroid-{resonance_key}.png'
        assert phase_unfiltered.exists(), f"Unfiltered phase portrait not found: {phase_unfiltered}"

        phase_slow = plot_path / f'phase_slow_test_asteroid-{resonance_key}.png'
        assert phase_slow.exists(), f"Slow phase portrait not found: {phase_slow}"

    def test_generate_only_phase_portrait(self):
        """Test generating only phase portrait."""
        sim = resonances.Simulation(
            name='test_phase_only',
            integrator='SABA(10,6,4)',
            tmax=10000,
            save='none',
            plot='all',
            plot_type='save',
            plots=['phase_portrait'],
        )
        sim.create_solar_system()
        sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'), name='test_asteroid')
        sim.run()

        plot_path = Path(sim.config.plot_path)
        resonance_key = '4J-2S-1+0+0-1'

        # Phase portraits should exist
        phase_filtered = plot_path / f'phase_filtered_test_asteroid-{resonance_key}.png'
        assert phase_filtered.exists()

        # Evolution plot should NOT exist
        evolution_file = plot_path / f'test_asteroid-{resonance_key}.png'
        assert not evolution_file.exists()
