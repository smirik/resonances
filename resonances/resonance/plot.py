import matplotlib.pyplot as plt
import numpy as np

import resonances.config
import resonances


def body(sim, body: resonances.Body, resonance, image_type='png'):  # noqa: C901
    plt.style.use('default')

    fig, axs = plt.subplots(6, 1, figsize=(10, 10))

    resonance_key = resonance.to_s()
    status = body.statuses.get(resonance_key, 0)
    resonance_name = resonance.to_short()
    if isinstance(resonance, resonances.MMR):
        angle_data = body.angle(resonance)
        angles_filtered = body.angles_filtered.get(resonance_key, None)
    elif isinstance(resonance, resonances.SecularResonance):
        angle_data = body.secular_angles.get(resonance_key, None)
        angles_filtered = body.secular_angles_filtered.get(resonance_key, None) if hasattr(body, 'secular_angles_filtered') else None

    fig.suptitle(
        "{}, resonance = {}, status = {}".format(body.name, resonance_name, status),
        fontsize=14,
    )

    axs[0].set_title('Resonant angle')
    axs[0].set_xlim([0, sim.config.tmax_yrs])

    # Adaptive tick spacing based on simulation length
    tmax_years = sim.config.tmax_yrs
    if tmax_years <= 50000:  # Short simulations (MMRs)
        major_tick = 10000
        minor_tick = 2000
    elif tmax_years <= 200000:  # Medium simulations
        major_tick = 50000
        minor_tick = 10000
    elif tmax_years <= 500000:  # Long simulations
        major_tick = 100000
        minor_tick = 20000
    else:  # Very long simulations (secular resonances)
        major_tick = 200000
        minor_tick = 50000

    axs[0].xaxis.set_major_locator(plt.MultipleLocator(major_tick))
    axs[0].xaxis.set_minor_locator(plt.MultipleLocator(minor_tick))
    if angle_data is not None:
        axs[0].plot(sim.times / (2 * np.pi), angle_data, linestyle='', marker=',', color='black')

    if angles_filtered is not None:  # pragma: no cover
        axs[1].set_title('Filtered resonant angle')
        axs[1].plot(sim.times / (2 * np.pi), angles_filtered, linestyle='', marker=',', color='black')
    else:
        axs[1].set_title('Again resonant angle (no filtered available)')
        if angle_data is not None:
            axs[1].plot(sim.times / (2 * np.pi), angle_data, linestyle='', marker=',', color='black')
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
        (body.periodogram_peaks.get(resonance_key) is not None)
        and ('peaks' in body.periodogram_peaks[resonance_key])
        and (body.periodogram_peaks[resonance_key]['peaks'].size)
    ):  # pragma: no cover
        peaks = body.periodogram_peaks[resonance_key]['peaks']
        for peak_width in body.periodogram_peaks[resonance_key]['position']:
            axs[3].axvline(x=peak_width[0], color='gray', linestyle="dashed")
            axs[3].axvline(x=peak_width[1], color='gray', linestyle='--')
        axs[3].plot(
            1.0 / body.periodogram_frequency[resonance_key][peaks],
            body.periodogram_power[resonance_key][peaks],
            'x',
            color='blue',
            markersize=10,
        )
        axs[3].plot(1.0 / body.periodogram_frequency[resonance_key], body.periodogram_power[resonance_key], color='black')

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
        axs[4].plot(
            1.0 / body.axis_periodogram_frequency[peaks],
            body.axis_periodogram_power[peaks],
            'x',
            color='blue',
            markersize=10,
        )
        axs[4].plot(1.0 / body.axis_periodogram_frequency, body.axis_periodogram_power, color='black')

        axs[4].sharex(axs[3])

    axs[5].set_title('Eccentricity')
    axs[5].plot(sim.times / (2 * np.pi), body.ecc, linestyle='', marker=',', color='black')
    axs[5].sharex(axs[0])

    axs[0].set_ylabel(r"$\sigma$ (rad)", fontsize=12)
    axs[1].set_ylabel(r"$\sigma_f$ (rad)", fontsize=12)
    axs[2].set_ylabel(r"$a_f$ (AU)", fontsize=12)
    axs[3].set_ylabel(r"$p_{\sigma}$", fontsize=12)
    axs[4].set_ylabel(r"$p_{a}$", fontsize=12)
    axs[5].set_ylabel("e", fontsize=12)

    plt.tight_layout()

    if sim.config.plot_type in ['both', 'save']:
        plt.savefig('{}/{}_{}.{}'.format(sim.config.plot_path, body.name, resonance_key, image_type))

    if sim.config.plot_type in ['both', 'show']:  # pragma: no cover
        plt.show()

    plt.close(fig)  # Prevents display in Jupyter Notebook
