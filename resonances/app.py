import time
import resonances

start_time = time.time()


def main():
    sim = resonances.Simulation(save=True, plot=True, save_path='cache/res')
    sim.create_solar_system()
    sim.add_body(463, resonances.ThreeBody('4J-2S-1'), 'A463')
    sim.run()
    print('Hello from end')


def run():
    print("Starting program")
    main()
    print("Ending program")
    print("--- %s seconds ---" % (time.time() - start_time))


def version():
    return "0.1.0"
