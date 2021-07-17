import resonances
import tests.tools as tools


def test_simple_run():
    resonances.config.set('plot', 'skip')
    sim = tools.create_test_simulation_for_solar_system()
    sim.add_body(tools.get_body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
    sim.run()

    assert 1 == 1
