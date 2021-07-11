import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resonances.resonance.libration import libration


def asteroids(data, mmrs, librations, save=True, save_path='cache'):
    for i, mmr in enumerate(mmrs):
        asteroid(
            data['times'],
            data['angle'][i],
            data['axis'][i],
            data['ecc'][i],
            mmr,
            librations['status'][i],
            librations['libration_data'][i],
            save_path,
        )
        if save:
            df_data = {
                'times': data['times'] / (2 * np.pi),
                'angle': data['angle'][i],
                'a': data['axis'][i],
                'ecc': data['ecc'][i],
            }
            df = pd.DataFrame(data=df_data)
            df.to_csv('{}/{}-{}.csv'.format(save_path, mmr.body_name, mmr.to_s()))


def asteroid(times, angle, axis, ecc, mmr, status, libration_data, save_path='cache'):
    plt.style.use('default')
    fig, axs = plt.subplots(5, 1, sharex=False, figsize=(20, 10))
    fig.suptitle(
        "Asteroid {}, resonance = {}, type = {}".format(mmr.body_name, mmr.to_s(), status),
        fontsize=14,
    )
    for axi in axs:
        axi.xaxis.set_major_locator(plt.LinearLocator(numticks=21))
        axi.set_xlim([0, 100000])

    axs[0].plot(times / (2 * np.pi), angle, linestyle='', marker=',')
    axs[1].plot(times / (2 * np.pi), libration.shift(angle), linestyle='', marker=',')
    axs[2].plot(times / (2 * np.pi), axis, linestyle='', marker=',')
    axs[3].plot(times / (2 * np.pi), ecc, linestyle='', marker=',')
    axs[4].plot(libration_data['ps'], libration_data['periodogram'], linestyle='', marker=',')

    plt.tight_layout()
    plt.savefig('{}/{}-{}-{}.png'.format(save_path, status, mmr.body_name, mmr.to_s()))
