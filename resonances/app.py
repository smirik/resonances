import time

from resonances.simulation import structure
from resonances.resonance.three_body import ThreeBody

start_time = time.time()


def main():
    asteroid = 463
    mmr_template = ThreeBody([4.0, -2.0, -1.0, 0.0, 0.0, -1.0], [5, 6], 10, '{}'.format(asteroid))
    structure.run(asteroid, {'a': 0.01, 'e': 0.21}, {'a': 2, 'e': 2}, mmr_template, saveOutput=False, dump=100, plot=True, saveData=True)


def run():
    a = 5.0
    print("Starting program")
    main()
    print("Ending program")
    print("--- %s seconds ---" % (time.time() - start_time))
    # test.my.hello()


def version():
    return "0.1.0"
