import json
import time
import argparse

import resonances
from resonances.matrix.three_body_matrix import ThreeBodyMatrix
from resonances.matrix.two_body_matrix import TwoBodyMatrix
from resonances.data.astdys import astdys

from resonances.experiment import shape
from resonances.experiment import loader
from resonances.experiment import finder

parser = argparse.ArgumentParser(description='')


def get_parser():
    return argparse.ArgumentParser(description='')


def quick():
    parser.add_argument('asteroid', help='The number of the asteroid you want to research.', type=int)
    parser.add_argument('resonance', help='The resonance in a short notation like 4J-2S-1 or 2J-1.', type=str)
    args = parser.parse_args()

    resonances.logger.info('Starting simulation "quick" for the asteroid {} and the resonance {}.'.format(args.asteroid, args.resonance))
    sim = resonances.Simulation()
    sim.create_solar_system()
    sim.add_body(args.asteroid, resonances.create_mmr(args.resonance), 'ast-{}'.format(args.asteroid))
    sim.run()
    resonances.logger.info('Successfully ended.')


def asteroids():
    parser.add_argument('--config', nargs='?', type=str)
    args = parser.parse_args()

    sim = loader.create_simulation_from_json(args.config)
    resonances.logger.info('Starting simulation "asteroids" based on the config {}'.format(args.config))
    sim.run()
    resonances.logger.info('Successfully ended.')


def calc_shape():
    start_time = time.time()
    parser.add_argument('--config', nargs='?', type=str)
    args = parser.parse_args()

    with open(args.config, "r") as read_file:
        c_config = json.load(read_file)

    mmr_template = resonances.create_mmr(c_config['resonance']['integers'], c_config['resonance']['bodies'])
    resonances.logger.info('Starting shape simulation based on the file {}'.format(args.config))
    shape.run(
        c_config['elem'],
        c_config['variations'],
        mmr_template,
        c_config['save'],
        c_config['save_path'],
        c_config['plot'],
        c_config['dump'],
    )

    resonances.logger.info("Successfully ended. --- %s seconds ---" % (time.time() - start_time))


def identifier():
    start_time = time.time()
    parser.add_argument('asteroid', help='The number of the asteroid you want to research.', type=int)
    parser.add_argument(
        '--planets', help='The planets, whose three body or two body resonances are searched. Use comma to separate planets.', type=str
    )
    args = parser.parse_args()

    resonances.config.set('save.path', 'cache/identifier')
    resonances.logger.info('Starting simulation "identifier" for the asteroid {}'.format(args.asteroid))

    planets = None
    if args.planets is not None:
        planets = [item.strip() for item in args.planets.split(',')]

    sim = resonances.Simulation()
    sim.create_solar_system()
    elem = astdys.search(args.asteroid)
    mmrs = ThreeBodyMatrix.find_resonances(elem['a'], planets=planets)
    mmrs2 = TwoBodyMatrix.find_resonances(elem['a'], planets=planets)
    mmrs = mmrs + mmrs2
    resonances.logger.info('The asteroid {} is found.'.format(args.asteroid))
    resonances.logger.info('The value of semi-major axis is {:6.4f}'.format(elem['a']))
    for mmr in mmrs:
        resonances.logger.info('Adding a possible resonance: {}'.format(mmr.to_short()))
        sim.add_body(args.asteroid, mmr, '{}-{}'.format(args.asteroid, mmr.to_short()))
    resonances.logger.info('Running integration.')
    sim.run()
    resonances.logger.info("Successfully ended. --- %s seconds ---" % (time.time() - start_time))


def asteroids_in_resonance():
    parser.add_argument('resonance', help='The resonance in a short notation like 4J-2S-1.', type=str)
    parser.add_argument(
        '--iterations',
        help='The maximum number of iterations. One iteration consists of 100 asteroid candidates. Should be integer.',
        type=int,
    )
    args = parser.parse_args()

    iterations = 1000
    if args.iterations is not None:
        iterations = args.iterations

    resonances.logger.info('Starting simulation "asteroids in resonance" for the resonance {}'.format(args.resonance))
    resonances.logger.info('The number of iterations: {}'.format(iterations))
    finder.run(resonances.create_mmr(args.resonance), dump=100, max_iterations=iterations)
    resonances.logger.info('Successfully ended.')
