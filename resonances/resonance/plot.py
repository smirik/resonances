import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

import resonances.config
import resonances


def body(sim, body: resonances.Body, resonance, image_type='png'):  # noqa: C901
    plt.style.use('default')

    fig, axs = plt.subplots(7, 1, figsize=(10, 12))

    resonance_key = resonance.to_s()
    status = body.statuses.get(resonance_key, 0)
    resonance_name = resonance.to_short()
    if isinstance(resonance, resonances.MMR):
        angle_data = body.angle(resonance)
        angles_filtered = body.angles_filtered.get(resonance_key, None)
    elif isinstance(resonance, resonances.SecularResonance):
        angle_data = body.secular_angles.get(resonance_key, None)
        angles_filtered = body.angles_filtered.get(resonance_key, None) if hasattr(body, 'angles_filtered') else None

    fig.suptitle(
        "{}, resonance = {}, status = {}".format(body.name, resonance_name, status),
        fontsize=14,
    )

    axs[0].set_title('Resonant angle')
    axs[0].set_xlim([0, abs(sim.config.tmax_yrs)])

    # Adaptive tick spacing based on simulation length
    tmax_years = sim.config.tmax_yrs

    # Calculate major_tick as tmax_yrs / 5, rounded to nice values
    major_tick = round_to_nice_value(abs(tmax_years) / 5)
    minor_tick = major_tick // 5

    # Ensure ticks are positive for matplotlib
    if major_tick <= 0:
        major_tick = 1
    if minor_tick <= 0:
        minor_tick = 1

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

    axs[3].set_title('Eccentricity')
    axs[3].plot(sim.times / (2 * np.pi), body.ecc, linestyle='', marker=',', color='black')
    axs[3].sharex(axs[0])

    axs[4].set_xlim(0, abs(tmax_years))
    # axs[4].set_ylim(0, 0.2)
    axs[4].axhline(y=0.05, color='r', linestyle='--')
    axs[4].axhline(y=0.1, color='g', linestyle='--')
    if (
        (body.periodogram_peaks.get(resonance_key) is not None)
        and ('peaks' in body.periodogram_peaks[resonance_key])
        and (body.periodogram_peaks[resonance_key]['peaks'].size)
    ):  # pragma: no cover
        peaks = body.periodogram_peaks[resonance_key]['peaks']
        for peak_width in body.periodogram_peaks[resonance_key]['position']:
            axs[4].axvline(x=peak_width[0], color='gray', linestyle="dashed")
            axs[4].axvline(x=peak_width[1], color='gray', linestyle='--')
        axs[4].plot(
            1.0 / body.periodogram_frequency[resonance_key][peaks],
            body.periodogram_power[resonance_key][peaks],
            'x',
            color='blue',
            markersize=10,
        )
        axs[4].plot(1.0 / body.periodogram_frequency[resonance_key], body.periodogram_power[resonance_key], color='black')

    axs[4].set_title('Periodogram (the resonant angle)')
    axs[5].set_title('Periodogram (semi-major axis)')
    axs[5].set_xlim(0, abs(tmax_years))
    # axs[5].set_ylim(0, 0.2)
    axs[5].axhline(y=0.05, color='r', linestyle='--')
    axs[5].axhline(y=0.1, color='g', linestyle='--')
    if (
        (body.axis_periodogram_peaks is not None)
        and ('peaks' in body.axis_periodogram_peaks)
        and (body.axis_periodogram_peaks['peaks'].size)
    ):  # pragma: no cover
        peaks = body.axis_periodogram_peaks['peaks']
        for peak_width in body.axis_periodogram_peaks['position']:
            axs[5].axvline(x=peak_width[0], color='gray', linestyle="dashed")
            axs[5].axvline(x=peak_width[1], color='gray', linestyle='--')
        axs[5].plot(
            1.0 / body.axis_periodogram_frequency[peaks],
            body.axis_periodogram_power[peaks],
            'x',
            color='blue',
            markersize=10,
        )
        axs[5].plot(1.0 / body.axis_periodogram_frequency, body.axis_periodogram_power, color='black')

        axs[5].sharex(axs[4])

    axs[6].set_title('Periodogram (eccentricity)')
    axs[6].set_xlim(0, abs(tmax_years))
    # axs[6].set_ylim(0, 0.2)
    axs[6].axhline(y=0.05, color='r', linestyle='--')
    axs[6].axhline(y=0.1, color='g', linestyle='--')
    if (
        (body.eccentricity_periodogram_peaks is not None)
        and ('peaks' in body.eccentricity_periodogram_peaks)
        and (body.eccentricity_periodogram_peaks['peaks'].size)
    ):  # pragma: no cover
        peaks = body.eccentricity_periodogram_peaks['peaks']
        for peak_width in body.eccentricity_periodogram_peaks['position']:
            axs[6].axvline(x=peak_width[0], color='gray', linestyle="dashed")
            axs[6].axvline(x=peak_width[1], color='gray', linestyle='--')
        axs[6].plot(
            1.0 / body.eccentricity_periodogram_frequency[peaks],
            body.eccentricity_periodogram_power[peaks],
            'x',
            color='blue',
            markersize=10,
        )
        axs[6].plot(1.0 / body.eccentricity_periodogram_frequency, body.eccentricity_periodogram_power, color='black')

        axs[6].sharex(axs[4])

    axs[0].set_ylabel(r"$\sigma$ (rad)", fontsize=12)
    axs[1].set_ylabel(r"$\sigma_f$ (rad)", fontsize=12)
    axs[2].set_ylabel(r"$a_f$ (AU)", fontsize=12)
    axs[3].set_ylabel("e", fontsize=12)
    axs[4].set_ylabel(r"$p_{\sigma}$", fontsize=12)
    axs[5].set_ylabel(r"$p_{a}$", fontsize=12)
    axs[6].set_ylabel(r"$p_{e}$", fontsize=12)

    plt.tight_layout()

    if sim.config.plot_type in ['both', 'save']:
        # Ensure the plot directory exists before saving
        Path(sim.config.plot_path).mkdir(parents=True, exist_ok=True)
        plt.savefig('{}/{}_{}.{}'.format(sim.config.plot_path, body.name, resonance_key, image_type))

    if sim.config.plot_type in ['both', 'show']:  # pragma: no cover
        plt.show()

    plt.close(fig)  # Prevents display in Jupyter Notebook


def round_to_nice_value(value):
    """Round value to a nice number ending with zeros."""
    if value <= 0:
        return 0
    power = 10 ** math.floor(math.log10(value))
    rounded = round(value / power) * power
    return int(rounded)
