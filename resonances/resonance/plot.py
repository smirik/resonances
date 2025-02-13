import matplotlib.pyplot as plt
import numpy as np

import resonances.config
import resonances


def body(sim, body: resonances.Body, mmr: resonances.MMR, image_type='png'):
    plt.style.use('default')

    fig, axs = plt.subplots(6, 1, figsize=(10, 10))

    fig.suptitle(
        "{}, resonance = {}, status = {}".format(body.name, mmr.to_short(), body.statuses[mmr.to_s()]),
        fontsize=14,
    )

    axs[0].set_title('Resonant angle')
    axs[0].set_xlim([0, sim.tmax_yrs])
    axs[0].xaxis.set_major_locator(plt.MultipleLocator(10000))
    axs[0].xaxis.set_minor_locator(plt.MultipleLocator(2000))
    axs[0].plot(sim.times / (2 * np.pi), body.angle(mmr), linestyle='', marker=',', color='black')

    if body.angles_filtered[mmr.to_s()] is not None:  # pragma: no cover
        axs[1].set_title('Filtered resonant angle')
        axs[1].plot(sim.times / (2 * np.pi), body.angles_filtered[mmr.to_s()], linestyle='', marker=',', color='black')
    else:
        axs[1].set_title('Again resonant angle (no filtered available)')
        axs[1].plot(sim.times / (2 * np.pi), body.angle(mmr), linestyle='', marker=',', color='black')
    axs[1].sharex(axs[0])

    if body.axis_filtered is not None:  # pragma: no cover
        axs[2].set_title('Filtered semi-major axis')
        axs[2].plot(sim.times / (2 * np.pi), body.axis_filtered, linestyle='', marker=',', color='black')
    else:
        axs[2].set_title('Semi-major axis')
        axs[2].plot(sim.times / (2 * np.pi), body.axis, linestyle='', marker=',', color='black')
    axs[2].sharex(axs[0])

    axs[3].set_xlim(0, 40000)
    # axs[2].set_ylim(0, 0.2)
    axs[3].axhline(y=0.05, color='r', linestyle='--')
    axs[3].axhline(y=0.1, color='g', linestyle='--')
    if (
        (body.periodogram_peaks[mmr.to_s()] is not None)
        and ('peaks' in body.periodogram_peaks[mmr.to_s()])
        and (body.periodogram_peaks[mmr.to_s()]['peaks'].size)
    ):  # pragma: no cover
        peaks = body.periodogram_peaks[mmr.to_s()]['peaks']
        for peak_width in body.periodogram_peaks[mmr.to_s()]['position']:
            axs[3].axvline(x=peak_width[0], color='gray', linestyle="dashed")
            axs[3].axvline(x=peak_width[1], color='gray', linestyle='--')
        axs[3].plot(
            1.0 / body.periodogram_frequency[mmr.to_s()][peaks], body.periodogram_power[mmr.to_s()][peaks], 'x', color='blue', markersize=10
        )
        axs[3].plot(1.0 / body.periodogram_frequency[mmr.to_s()], body.periodogram_power[mmr.to_s()], color='black')

    axs[3].set_title('Periodogram (the resonant angle)')
    axs[4].set_title('Periodogram (semi-major axis)')
    axs[4].set_xlim(0, 40000)
    # axs[3].set_ylim(0, 0.2)
    axs[4].axhline(y=0.05, color='r', linestyle='--')
    axs[4].axhline(y=0.1, color='g', linestyle='--')
    if (
        (body.axis_periodogram_peaks is not None)
        and ('peaks' in body.axis_periodogram_peaks)
        and (body.axis_periodogram_peaks['peaks'].size)
    ):  # pragma: no cover
        peaks = body.axis_periodogram_peaks['peaks']
        for peak_width in body.axis_periodogram_peaks['position']:
            axs[4].axvline(x=peak_width[0], color='gray', linestyle="dashed")
            axs[4].axvline(x=peak_width[1], color='gray', linestyle='--')
        axs[4].plot(1.0 / body.axis_periodogram_frequency[peaks], body.axis_periodogram_power[peaks], 'x', color='blue', markersize=10)
        axs[4].plot(1.0 / body.axis_periodogram_frequency, body.axis_periodogram_power, color='black')

        axs[4].sharex(axs[3])

    axs[5].set_title('Eccentricity')
    axs[5].plot(sim.times / (2 * np.pi), body.ecc, linestyle='', marker=',', color='black')
    axs[5].sharex(axs[0])

    axs[0].set_ylabel(r"$\sigma$ (deg)", fontsize=12)
    axs[1].set_ylabel(r"$\sigma_f$ (deg)", fontsize=12)
    axs[2].set_ylabel(r"$a_f$ (AU)", fontsize=12)
    axs[3].set_ylabel(r"$p_{\sigma}$", fontsize=12)
    axs[4].set_ylabel(r"$p_{a}$", fontsize=12)
    axs[5].set_ylabel("e", fontsize=12)

    plt.tight_layout()

    if sim.plot_type in ['both', 'save']:
        plt.savefig('{}/{}_{}.{}'.format(sim.plot_path, body.name, mmr.to_s(), image_type))

    if sim.plot_type in ['both', 'show']:  # pragma: no cover
        plt.show()
