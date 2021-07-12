import json, time, argparse

import resonances
from resonances.experiment import shape

parser = argparse.ArgumentParser(description='')


def get_parser():
    return argparse.ArgumentParser(description='')


def quick():
    parser.add_argument('asteroid', help='The number of the asteroid you want to research.', type=int)
    parser.add_argument('resonance', help='The resonance in a short notation like 4J-2S-1.', type=str)
    args = parser.parse_args()

    sim = resonances.Simulation()
    sim.create_solar_system()
    sim.add_body(args.asteroid, resonances.ThreeBody(args.resonance), 'ast-{}'.format(args.asteroid))
    sim.run()


def asteroid():
    parser.add_argument('--config', nargs='?', type=str)
    args = parser.parse_args()
    with open(args.config, "r") as read_file:
        c_config = json.load(read_file)

    sim = resonances.Simulation(
        save=c_config['save'], plot=c_config['plot'], save_path=c_config['save_path'], tmax=c_config['tmax'], Nout=c_config['Nout']
    )
    sim.create_solar_system()

    mmrs = []
    # @todo need to verify that data are full
    # Add checks for asteroids, resonances, Nout and stop, plot (or default values)
    for asteroid in c_config['asteroids']:
        if 'num' in asteroid['elem']:
            elem_or_num = asteroid['elem']['num']
        else:
            elem_or_num = asteroid['elem']
        # @todo validation
        for resonance in asteroid['resonances']:
            sim.add_body(
                elem_or_num, resonances.ThreeBody(resonance['integers'], resonance['bodies']), '{}'.format(asteroid['elem']['label'])
            )

    sim.run()


def calc_shape():
    start_time = time.time()
    parser.add_argument('--config', nargs='?', type=str)
    args = parser.parse_args()

    with open(args.config, "r") as read_file:
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

    print("--- %s seconds ---" % (time.time() - start_time))
