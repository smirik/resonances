import numpy as np
import math
from itertools import product

import resonances


def run(elem, variations, mmr: resonances.MMR, save=False, save_path="cache", need_plot=False, dump=100):

    particles = []
    for variation in variations:
        a_arr = np.linspace(variation['a']['min'], variation['a']['max'], variation['a']['num'])
        e_arr = np.linspace(variation['e']['min'], variation['e']['max'], variation['e']['num'])
        particles += list(product(a_arr, e_arr))

    num_particles = len(particles)

    num_iterations = int(math.ceil(num_particles / dump))

    for j in range(num_iterations):
        sim = resonances.Simulation(save=save, save_path=save_path, plot=need_plot)
        sim.create_solar_system()
        # Find the number of calculations for a given step. It varies for the last one.
        if j < num_iterations - 1:
            num = dump
        else:
            num = num_particles % dump
            if num == 0:
                num = dump

        ae_data = np.zeros((num, 4))

        for i in range(num):
            key = j * dump + i
            elem = {
                'a': particles[key][0],
                'e': particles[key][1],
                'inc': elem['inc'],
                'Omega': elem['Omega'],
                'omega': elem['omega'],
                'M': elem['M'],
            }
            sim.add_body(elem, mmr, '{}'.format(key))

            ae_data[i][0] = key
            ae_data[i][1] = particles[key][0]
            ae_data[i][2] = particles[key][1]

        sim.run()

        if save:
            for i, body in enumerate(sim.bodies):
                ae_data[i][3] = body.status
            with open('{}/ae-plane.csv'.format(save_path), 'a') as f:
                np.savetxt(f, ae_data, delimiter=',')

        resonances.logger.info('Iteration {} has finished. Processed from {}. Starting the new one.'.format(j, j * dump))
