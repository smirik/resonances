import resonances
import tests.tools as tools


def test_simple_run():
    resonances.config.set('plot', None)
    sim = tools.create_test_simulation_for_solar_system()
    sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'))
    sim.add_body(tools.get_2body_elements_sample(), resonances.TwoBody('1J-1'))
    sim.run()

    assert 1 == 1
    resonances.config.set('plot', 'resonant')
