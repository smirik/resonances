import time
import resonances

from resonances.data import loader

start_time = time.time()


def main():
    sim = loader.create_simulation_from_json('cache/examples/libration-examples.json')
    sim.Nout = 10000
    sim.run()


def main22():
    sim = resonances.Simulation(save=True, plot=True, save_path='cache/res')
    sim.create_solar_system()
    sim.Nout = 10000
    sim.add_body(463, resonances.ThreeBody('4J-2S-1'), 'A463')
    print(sim.Nout)
    sim.run()


def run():
    print("Starting program")
    main()
    print("Ending program")
    print("--- %s seconds ---" % (time.time() - start_time))


def version():
    return "0.1.0"
