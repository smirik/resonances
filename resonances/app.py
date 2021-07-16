import time
import resonances

from resonances.data import loader
from resonances.experiment import shape
import json

start_time = time.time()


def main():
    with open('cache/examples/simulation-shape.json', "r") as read_file:
        c_config = json.load(read_file)

    mmr_template = resonances.ThreeBody(c_config['resonance']['integers'], c_config['resonance']['bodies'])
    shape.run(
        c_config['elem'],
        c_config['variations'],
        mmr_template,
        c_config['save'],
        c_config['save_path'],
        c_config['plot'],
        c_config['dump'],
    )


def main33333():
    sim = loader.create_simulation_from_json('cache/examples/libration-examples.json')
    # sim.Nout = sim.Nout * 10
    # print(sim.Nout)
    sim.run()


def heartbeat(sim_pointer):
    sim = sim_pointer.contents
    if int(sim.t) % 10000 == 0:
        print('t={:6.0f}, dt={:6.2f}, steps_done={:08d}'.format(sim.t, sim.dt, sim.steps_done))


def main23():
    sim = resonances.Simulation(save=True, plot=True, save_path='cache/res')
    sim.create_solar_system()
    # sim.Nout = sim.Nout * 10
    sim.add_body(490, resonances.ThreeBody('5J-2S-2'), 'A490')

    # sim.integrator = "SABA(10,6,4)"
    # sim.dt = 0.1
    # sim.sim.ri_saba.safe_mode = 0
    # sim.integrator = "whfast"
    # sim.sim.ri_whfast.corrector = 17
    # sim.sim.ri_whfast.safe_mode = 0

    # sim.integrator = "ias15"
    # sim.sim.ri_ias15.min_dt = 1e-3
    sim.sim.heartbeat = heartbeat
    sim.run()


def main_ias():
    sim = resonances.Simulation(save=True, plot=True, save_path='cache/res')
    sim.create_solar_system()
    # sim.Nout = sim.Nout * 10
    sim.add_body(490, resonances.ThreeBody('5J-2S-2'), 'A490')
    sim.integrator = "ias15"
    sim.dt = 0.1
    sim.sim.heartbeat = heartbeat
    sim.run()


def run():
    print("Starting program")
    main()
    print("Ending program")
    print("--- %s seconds ---" % (time.time() - start_time))


def version():
    return "0.1.0"
