import json
import time
import argparse

import resonances
from resonances.matrix.three_body_matrix import ThreeBodyMatrix
from resonances.data.astdys import astdys

from resonances.experiment import shape
from resonances.experiment import loader
from resonances.experiment import finder

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


def identifier():
    start_time = time.time()
    parser.add_argument('asteroid', help='The number of the asteroid you want to research.', type=int)
    args = parser.parse_args()

    resonances.config.set('save.path', 'cache/identifier')

    sim = resonances.Simulation()
    sim.create_solar_system()
    elem = astdys.search(args.asteroid)
    mmrs = ThreeBodyMatrix.find_resonances(elem['a'])
    print('The asteroid {} is found.'.format(args.asteroid))
    print('The value of semi-major axis is {:6.4f}'.format(elem['a']))
    for mmr in mmrs:
        print('Adding a possible resonance: {}'.format(mmr.to_short()))
        sim.add_body(args.asteroid, mmr, '{}-{}'.format(args.asteroid, mmr.to_short()))
    print('Running integration.')
    sim.run()
    print("--- %s seconds ---" % (time.time() - start_time))


def asteroids_in_resonance():
    parser.add_argument('resonance', help='The resonance in a short notation like 4J-2S-1.', type=str)
    args = parser.parse_args()

    finder.run(resonances.ThreeBody(args.resonance))
    return
