import json
import numpy as np
import pandas as pd
from resonances.console.console import console as cs

from resonances.data.astdys import astdys
from resonances.resonance.three_body import ThreeBody
from resonances.resonance import integration
from resonances.resonance import plot


def asteroid():
    print('Getting data from file {}'.format(cs.args.config))
    with open(cs.args.config, "r") as read_file:
        config = json.load(read_file)

    sim = integration.create_solar_system()

    mmrs = []
    # @todo need to verify that data are full
    # Add checks for asteroids, resonances, Nout and stop, plot (or default values)
    i = 0
    for asteroid in config['asteroids']:
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

    data = integration.integrate(sim, mmrs, config['stop'], config['Nout'])
    librations = integration.librations(data, mmrs, config['Nout'])

    for i, mmr in enumerate(mmrs):
        if config['plot']:
            plot.asteroid(
                data['times'],
                data['angle'][i],
                data['axis'][i],
                data['ecc'][i],
                mmr,
                librations['status'][i],
                librations['libration_data'][i],
                config['Nout'],
                config['save_path'],
            )
        if config['save']:
            df_data = {
                'times': data['times'] / (2 * np.pi),
                'angle': data['angle'][i],
                'a': data['axis'][i],
                'ecc': data['ecc'][i],
            }
            df = pd.DataFrame(data=df_data)
            df.to_csv('{}/{}-{}.csv'.format(config['save_path'], mmr.body_name, mmr.to_s()))
