from pathlib import Path
import resonances
import os
import tests.tools as tools


def test_simple_run():
    sim = tools.create_test_simulation_for_solar_system()
    sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
    sim.add_body(tools.get_2body_elements_sample(), resonances.TwoBody('1J-1'))
    sim.run()

    assert 1 == 1


def test_body():
    sim = tools.create_test_simulation_for_solar_system()
    sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
    sim.add_body(tools.get_2body_elements_sample(), resonances.TwoBody('1J-1'))
    sim.run()

    body = sim.bodies[0]
    mmr = body.mmrs[0]
    body.angles_filtered[mmr.to_s()] = None
    resonances.resonance.plot.body(sim, body, mmr)

    body.axis_filtered = None
    resonances.resonance.plot.body(sim, body, mmr)

    sim.config.plot_type = 'save'
    resonances.resonance.plot.body(sim, body, mmr)
    file_path = f"{sim.config.save_path}/asteroid_4J-2S-1+0+0-1.png"
    assert Path(file_path).is_file() is True

    os.remove(file_path)
    assert Path(file_path).is_file() is False

    sim.config.plot_type = None
    resonances.resonance.plot.body(sim, body, mmr)
    file_path = f"{sim.config.save_path}/asteroid_4J-2S-1+0+0-1.png"
    assert Path(file_path).is_file() is False
