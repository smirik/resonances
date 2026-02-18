"""Tests for plotting functionality using the new Plotter class."""

from pathlib import Path

import pytest
import resonances
import os
import tests.tools as tools


@pytest.mark.slow
def test_simple_run():
    """Test basic simulation with plotting enabled."""
    sim = tools.create_test_simulation_for_solar_system()
    sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
    sim.add_body(tools.get_2body_elements_sample(), resonances.TwoBody('1J-1'))
    sim.run()

    assert 1 == 1


def test_plotter_with_missing_data():
    """Test Plotter handles missing filtered data gracefully."""
    sim = tools.create_test_simulation_for_solar_system()
    sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
    sim.add_body(tools.get_2body_elements_sample(), resonances.TwoBody('1J-1'))
    sim.run()

    body = sim.bodies[0]
    mmr = body.mmrs[0]

    # Test with missing filtered angle
    body.angles_filtered[mmr.to_s()] = None
    plotter = resonances.Plotter.from_body(body, mmr, sim)
    plotter.configure('full').plot()  # Should not crash
    plotter.close()

    # Test with missing filtered axis
    body.axis_filtered = None
    plotter = resonances.Plotter.from_body(body, mmr, sim)
    plotter.configure('full').plot()  # Should not crash
    plotter.close()


def test_plotter_save_functionality():
    """Test Plotter save functionality with different plot_type settings."""
    sim = tools.create_test_simulation_for_solar_system()
    sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
    sim.add_body(tools.get_2body_elements_sample(), resonances.TwoBody('1J-1'))
    sim.run()

    body = sim.bodies[0]
    mmr = body.mmrs[0]
    resonance_key = mmr.to_s()

    # Test plot_type = 'save'
    sim.config.plot_type = 'save'
    file_path = f"{sim.config.save_path}/asteroid_{resonance_key}.png"

    plotter = resonances.Plotter.from_body(body, mmr, sim).configure('full').plot()
    plotter.save(file_path)
    plotter.close()

    assert Path(file_path).is_file() is True

    # Cleanup
    os.remove(file_path)
    assert Path(file_path).is_file() is False


def test_plotter_with_custom_config():
    """Test Plotter with custom configuration."""
    sim = tools.create_test_simulation_for_solar_system()
    sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
    sim.run()

    body = sim.bodies[0]
    mmr = body.mmrs[0]

    # Create custom config
    config = resonances.PlotConfig(figsize=(8, 6))
    config.add_panel(
        resonances.Panel(
            key='angle', data_column=f'{mmr.to_s()}_angle', x_column='times', style=resonances.StyleConfig(color='red', ylabel='Angle')
        )
    )

    # Test with custom config
    plotter = resonances.Plotter.from_body(body, mmr, sim)
    plotter.configure(config).plot()

    assert plotter._figure is not None
    assert len(plotter._axes) == 1
    plotter.close()


def test_plotter_presets():
    """Test Plotter with different presets."""
    sim = tools.create_test_simulation_for_solar_system()
    sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
    sim.run()

    body = sim.bodies[0]
    mmr = body.mmrs[0]

    # Test 'simple' preset
    plotter_simple = resonances.Plotter.from_body(body, mmr, sim)
    plotter_simple.configure('simple').plot()
    assert plotter_simple._figure is not None
    plotter_simple.close()

    # Test 'full' preset
    plotter_full = resonances.Plotter.from_body(body, mmr, sim)
    plotter_full.configure('full').plot()
    assert plotter_full._figure is not None
    plotter_full.close()
