import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal.spectral import periodogram

from resonances.resonance.three_body import ThreeBody
from resonances.resonance import libration
from resonances.resonance import integration

from resonances.data.astdys import astdys


def run(asteroid, variations, num_variations, mmr_template: ThreeBody, saveOutput=False, dump=100, plot=False, saveData=False):
    sim = integration.create_solar_system()
    asteroids = [asteroid]

    elem = astdys.search(asteroid)
    sig_a = variations['a']
    sig_e = variations['e']

    var_a = num_variations['a']
    var_e = num_variations['e']

    a_arr = np.linspace(elem['a'] - sig_a, elem['a'] + sig_a, var_a)
    # e_arr = np.linspace(elem['e'] - sig_e, elem['e'] + sig_e, var_e)
    e_arr = np.linspace(0.0, 0.3, var_e)

    mmrs = []
    ae_data = np.zeros((var_a * var_e, 3))
    i = 0
    for axis in a_arr:
        for ecc in e_arr:
            sim.add(
                m=0.0,
                a=axis,
                e=ecc,
                inc=elem['i'],
                Omega=elem['Omega'],
                omega=elem['omega'],
                M=elem['M'],
                date=astdys.date,
                primary=sim.particles[0],
            )
            mmrs.append(
                ThreeBody(mmr_template.coeff, mmr_template.index_of_planets, 10 + i, '{}-{:.6f}-{:.6f}'.format(asteroid, axis, ecc))
            )
            ae_data[i][0] = axis
            ae_data[i][1] = ecc
            ae_data[i][2] = 0
            i += 1

    os = sim.calculate_orbits(primary=sim.particles[0])

    Nout = 10000
    data = integration.integrate(sim, mmrs, 6.28e5, Nout)

    for i, mmr in enumerate(mmrs):
        libration_status = libration.libration(data['times'] / (2 * np.pi), data['angle'][i], Nout)
        if libration_status['flag']:
            if libration_status['pure']:
                ae_data[i][2] = 2
                print('TRUE (PURE) LIBRATOR: Asteroid {} IS in the pure resonance {}'.format(mmr.body_name, mmr.to_s()))
            else:
                ae_data[i][2] = 1
                print('TRUE (NOT PURE) LIBRATOR: Asteroid {} IS in the resonance {}'.format(mmr.body_name, mmr.to_s()))
        else:
            print('FALSE: Asteroid {} IS NOT in the resonance {}'.format(mmr.body_name, mmr.to_s()))

        if plot:
            plt.style.use('default')
            fig, axs = plt.subplots(5, 1, sharex=False, figsize=(20, 10))
            fig.suptitle(
                "{} - {} - Asteroid {} - {} - {}".format(ae_data[i][2], i, mmrs[i].body_name, mmrs[i].to_s(), libration_status['pmax']),
                fontsize=14,
            )
            for axi in axs:
                # axi.grid()
                axi.xaxis.set_major_locator(plt.LinearLocator(numticks=21))
                axi.set_xlim([0, 100000])
            axs[0].plot(data['times'] / (2 * np.pi), data['angle'][i], linestyle='', marker=',')
            axs[1].plot(data['times'] / (2 * np.pi), libration.shift_apocentric(data['angle'][i]), linestyle='', marker=',')
            axs[2].plot(data['times'] / (2 * np.pi), data['axis'][i], linestyle='', marker=',')
            axs[3].plot(data['times'] / (2 * np.pi), data['ecc'][i], linestyle='', marker=',')
            axs[4].plot(libration_status['ps'], np.sqrt(4 * libration_status['periodogram'] / Nout), linestyle='', marker=',')

            plt.tight_layout()
            plt.savefig('cache/{}-{}-{}-{}.png'.format(ae_data[i][2], i, mmr.body_name, mmr.to_s()))
            # plt.show()

        if saveData:
            df_data = {
                'times': data['times'] / (2 * np.pi),
                'angle': data['angle'][i],
                'a': data['axis'][i],
                'ecc': data['ecc'][i],
            }
            df = pd.DataFrame(data=df_data)
            df.to_csv('cache/examples/{}.csv'.format(i))

    if saveOutput:
        np.savetxt('cache/ae-plane.csv', ae_data, delimiter=',')
