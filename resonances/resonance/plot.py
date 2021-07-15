import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import resonances.config


def body(sim, body: resonances.Body):
    plt.plot(body.axis)

    plt.style.use('default')

    fig, axs = plt.subplots(5, 1, figsize=(15, 15))

    # fig.suptitle(
    #     "Asteroid {}, resonance = {}, status = {}".format(body.name, body.mmr.to_short(), body.status),
    #     fontsize=14,
    # )

    axs[0].set_xlim([0, sim.tmax_yrs])
    axs[0].xaxis.set_major_locator(plt.MultipleLocator(10000))
    axs[0].xaxis.set_minor_locator(plt.MultipleLocator(2000))
    axs[0].set_title('{} - {} ({})'.format(body.name, body.mmr.to_short(), body.status))
    axs[0].plot(sim.times / (2 * np.pi), body.angle, linestyle='', marker=',')
    axs[1].plot(sim.times / (2 * np.pi), body.axis_filtered, linestyle='', marker=',')
    axs[1].sharex(axs[0])

    axs[2].set_title('Periodograms')

    axs[2].set_xlim(0, 40000)
    # axs[2].set_ylim(0, 0.2)
    axs[2].axhline(y=0.05, color='r', linestyle='--')
    axs[2].axhline(y=0.1, color='g', linestyle='--')
    peaks = body.periodogram_peaks['peaks']
    for peak_width in body.periodogram_peaks['position']:
        axs[2].axvline(x=peak_width[0], color='gray', linestyle="dashed")
        axs[2].axvline(x=peak_width[1], color='gray', linestyle='--')
    axs[2].plot(1.0 / body.periodogram_frequency[peaks], body.periodogram_power[peaks], 'x', color='orange')
    axs[2].plot(1.0 / body.periodogram_frequency, body.periodogram_power)

    axs[3].set_xlim(0, 40000)
    # axs[3].set_ylim(0, 0.2)
    axs[3].axhline(y=0.05, color='r', linestyle='--')
    axs[3].axhline(y=0.1, color='g', linestyle='--')
    peaks = body.axis_periodogram_peaks['peaks']
    for peak_width in body.axis_periodogram_peaks['position']:
        axs[3].axvline(x=peak_width[0], color='gray', linestyle="dashed")
        axs[3].axvline(x=peak_width[1], color='gray', linestyle='--')
    axs[3].plot(1.0 / body.axis_periodogram_frequency[peaks], body.axis_periodogram_power[peaks], 'x', color='orange')
    axs[3].plot(1.0 / body.axis_periodogram_frequency, body.axis_periodogram_power)

    axs[3].sharex(axs[2])

    axs[4].plot(sim.times / (2 * np.pi), body.axis, linestyle='', marker=',')

    plt.tight_layout()

    # asteroid_plot(sim.times, body.angle, body.axis, body.ecc, body.name, body.mmr, body.status, body.libration_data)
    if sim.plot == True:
        plt.savefig('{}/fig-{}-{}-{}-{}.png'.format(sim.save_path, body.index_in_simulation, body.status, body.name, body.mmr.to_s()))
    elif sim.plot == 'show':
        plt.show()

