import resonances


def get_body_elements_sample():
    return {
        "a": 2.398825840331548,
        "e": 0.2194125828625336,
        "inc": 0.23627318991620527,
        "Omega": 0.6370508455573044,
        "omega": 5.752902062786396,
        "M": 2.4309844848211464,
    }


def create_test_simulation_for_solar_system(
    save=False, plot=False, save_summary=False, save_additional_data=False, save_only_undetermined=False
):
    sim = resonances.Simulation()
    sim.create_solar_system()

    # create to speedup
    sim.tmax = 20
    sim.dt = 1
    sim.Nout = 10
    sim.integrator = 'whfast'
    sim.integrator_corrector = None
    sim.save_path = 'cache/tests'
    sim.save_summary = save_summary
    sim.save = save
    sim.save_additional_data = save_additional_data
    sim.save_only_undetermined = save_only_undetermined
    sim.plot = plot

    return sim


def add_test_asteroid_to_simulation(sim):
    elem = get_body_elements_sample()
    mmr = resonances.ThreeBody('4J-2S-1')
    sim.add_body(elem, mmr, name='asteroid')
    return sim
