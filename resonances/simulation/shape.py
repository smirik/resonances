import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from itertools import product

from resonances.config import config
from resonances.resonance.three_body import ThreeBody
from resonances.resonance.libration import libration
from resonances.resonance import integration, plot

from resonances.data.astdys import astdys


def run(elem, variations, mmr_template: ThreeBody, save=False, save_path="cache", need_plot=False, dump=100):
    a_arr = np.linspace(variations['a']['min'], variations['a']['max'], variations['a']['num'])
    e_arr = np.linspace(variations['e']['min'], variations['e']['max'], variations['e']['num'])
    particles = list(product(a_arr, e_arr))

    num_particles = len(particles)

    num_iterations = int(math.ceil(num_particles / dump))
    Nout = config.get('Nout')

    for j in range(num_iterations):
        sim = integration.create_solar_system()
        mmrs = []

        if j < num_iterations - 1:
            num = dump
        else:
            num = num_particles % dump
            if num == 0:
                num = dump
        ae_data = np.zeros((num, 3))
        for i in range(num):
            key = j * dump + i
            sim.add(
                m=0.0,
                a=particles[key][0],
                e=particles[key][1],
                inc=elem['inc'],
                Omega=elem['Omega'],
                omega=elem['omega'],
                M=elem['M'],
                date=astdys.date,
                primary=sim.particles[0],
            )
            mmrs.append(
                ThreeBody(
                    mmr_template.coeff,
                    mmr_template.index_of_planets,
                    10 + i,
                    '{}-{:.6f}-{:.6f}'.format(key, particles[key][0], particles[key][1]),
                )
            )
            ae_data[i][0] = particles[key][0]
            ae_data[i][1] = particles[key][1]
            ae_data[i][2] = 0

        data = integration.integrate(sim, mmrs, config.get('interval'), config.get('Nout'))
        librations = integration.librations(data, mmrs, Nout)

        if need_plot:
            plot.asteroids(data, mmrs, librations, save, save_path)

        if save:
            for i, status in enumerate(librations['status']):
                ae_data[i][2] = status
            with open('{}/ae-plane.csv'.format(save_path), 'a') as f:
                np.savetxt(f, ae_data, delimiter=',')
        print('Iteration {} has finished. Processed from {}. Starting the new one.'.format(j, j * dump))
