import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import resonances.config


def asteroids(data, mmrs, librations, save=True, save_path='cache'):
    if save:
        Path(save_path).mkdir(parents=True, exist_ok=True)

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
            df.to_csv('{}/data-{}-{}.csv'.format(save_path, mmr.body_name, mmr.to_s()))


def body(sim, body: resonances.Body):
    asteroid_plot(sim.times, body.angle, body.axis, body.ecc, body.name, body.mmr, body.status, body.libration_data)
    plt.savefig('{}/fig-{}-{}-{}.png'.format(sim.save_path, body.status, body.name, body.mmr.to_s()))


def asteroid(times, angle, axis, ecc, body_name, mmr, status, libration_data, save_path='cache'):
    fig, axs = asteroid_plot(times, angle, axis, ecc, body_name, mmr, status, libration_data)
    plt.savefig('{}/fig-{}-{}-{}.png'.format(save_path, status, body_name, mmr.to_s()))


def asteroid_plot(times, angle, axis, ecc, body_name, mmr, status, libration_data):
    plt.style.use('default')
    fig, axs = plt.subplots(6, 1, sharex=False, figsize=(20, 10))
    fig.suptitle(
        "Asteroid {}, resonance = {}, status = {}".format(body_name, mmr.to_s(), status),
        fontsize=14,
    )
    for axi in axs:
        axi.xaxis.set_major_locator(plt.LinearLocator(numticks=21))
        axi.set_xlim([0, 100000])

    axs[0].plot(times / (2 * np.pi), angle, linestyle='', marker=',')
    axs[1].plot(times / (2 * np.pi), resonances.libration.shift(angle), linestyle='', marker=',')
    axs[2].plot(times / (2 * np.pi), axis, linestyle='', marker=',')
    axs[3].plot(times / (2 * np.pi), ecc, linestyle='', marker=',')
    axs[4].set_xlim([resonances.config.get('libration.start'), resonances.config.get('libration.stop')])
    axs[4].plot(libration_data['ps'], libration_data['periodogram'], linestyle='', marker=',')
    axs[5].set_xlim([0, 2 * np.pi])
    axs[5].xaxis.set_major_locator(plt.LinearLocator(numticks=10))
    axs[5].plot(libration_data['density']['ps'], libration_data['density']['kdes'])

    plt.tight_layout()
    return (fig, axs)
