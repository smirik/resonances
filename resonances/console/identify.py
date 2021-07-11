import json
import numpy as np
import pandas as pd
from resonances.console.console import console as cs
from resonances.config import config

from resonances.data.astdys import astdys
from resonances.resonance.three_body import ThreeBody
from resonances.resonance import integration
from resonances.resonance import plot


def quick():
    cs.quick()

    sim = integration.create_solar_system()
    sim = integration.add_asteroid_by_num(sim, cs.args.asteroid)
    mmrs = [
        ThreeBody(
            cs.args.resonance,
            [5, 6],
            10,
            '{}'.format(cs.args.asteroid),
        )
    ]

    os = sim.calculate_orbits(primary=sim.particles[0])
    sim.status()

    data = integration.integrate(sim, mmrs, config.get('interval'), config.get('Nout'))
    librations = integration.librations(data, mmrs, config.get('Nout'))
    plot.asteroids(data, mmrs, librations)


def asteroid():
    cs.asteroid()
    print('Getting data from file {}'.format(cs.args.config))
    with open(cs.args.config, "r") as read_file:
        a_config = json.load(read_file)

    sim = integration.create_solar_system()

    mmrs = []
    # @todo need to verify that data are full
    # Add checks for asteroids, resonances, Nout and stop, plot (or default values)
    i = 0
    for asteroid in a_config['asteroids']:
        if 'num' in asteroid['elements']:
            elem = astdys.search(asteroid['elements']['num'])
        else:
            elem = asteroid['elements']
        sim = integration.add_asteroid_by_elem(sim, elem)

        # @todo validation
        for resonance in asteroid['resonances']:
            mmrs.append(
                ThreeBody(
                    resonance['integers'],
                    integration.index_of_planets(resonance['bodies']),
                    10 + i,
                    '{}'.format(asteroid['elements']['label']),
                )
            )
        i += 1

    os = sim.calculate_orbits(primary=sim.particles[0])
    sim.status()

    data = integration.integrate(sim, mmrs, a_config['stop'], a_config['Nout'])
    librations = integration.librations(data, mmrs, a_config['Nout'])

    if a_config['plot']:
        plot.asteroids(data, mmrs, librations, a_config['save'], a_config['save_path'])
