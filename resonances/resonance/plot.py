import matplotlib.pyplot as plt
import numpy as np

from resonances.resonance.libration import libration


def asteroid(times, angle, axis, ecc, mmr, status, libration_data, Nout, save_path='cache'):
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
    axs[1].plot(times / (2 * np.pi), libration.shift_apocentric(angle), linestyle='', marker=',')
    axs[2].plot(times / (2 * np.pi), axis, linestyle='', marker=',')
    axs[3].plot(times / (2 * np.pi), ecc, linestyle='', marker=',')
    axs[4].plot(libration_data['ps'], np.sqrt(4 * libration_data['periodogram'] / Nout), linestyle='', marker=',')

    plt.tight_layout()
    plt.savefig('{}/{}-{}-{}.png'.format(save_path, status, mmr.body_name, mmr.to_s()))
