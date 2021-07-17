import json
import time
import argparse

import resonances
from resonances.experiment import shape
from resonances.experiment import loader

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


def asteroids():
    parser.add_argument('--config', nargs='?', type=str)
    args = parser.parse_args()

    sim = loader.create_simulation_from_json(args.config)
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
