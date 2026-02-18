"""
Preset configurations for common plotting scenarios.

Provides 'simple' and 'full' presets as specified in requirements.
"""

import numpy as np
from .config import PlotConfig, Panel, StyleConfig


def create_simple_preset(resonance_key: str) -> PlotConfig:
    """
    Create a simple preset showing only resonant angle vs time.

    This preset is useful for quick checks of resonance behavior.

    Parameters
    ----------
    resonance_key : str
        Resonance key (e.g., '4J-2S-1+0+0-1')

    Returns
    -------
    PlotConfig
        Simple plot configuration
    """
    config = PlotConfig(figsize=(10, 4), plot_title='{body_name}, resonance = {resonance}, status = {status}')  # Smaller for single panel

    # Single panel: resonant angle
    config.add_panel(
        Panel(
            key='angle',
            data_column=f'{resonance_key}_angle',
            x_column='times',
            style=StyleConfig(
                ylabel=r"$\sigma$ (rad)",
                xlabel="Time (years)",
                title="Resonant angle",
                color='black',
                marker=',',
                linestyle='',
            ),
            required=True,
        )
    )

    return config


def create_full_preset(resonance_key: str) -> PlotConfig:
    """
    Create a full preset.

    This is the default preset and replicates existing functionality:
    - Resonant angle
    - Filtered resonant angle
    - Semi-major axis (filtered if available)
    - Eccentricity
    - Inclination
    - Periodogram of resonant angle
    - Periodogram of semi-major axis
    - Periodogram of eccentricity

    Parameters
    ----------
    resonance_key : str
        Resonance key (e.g., '4J-2S-1+0+0-1')

    Returns
    -------
    PlotConfig
        Full N-panel plot configuration
    """
    config = create_angle_preset(resonance_key)
    config.figsize = (10, 12)

    # Panel 4: Semi-major axis
    config.add_panel(
        Panel(
            key='axis',
            data_column='a_filtered',
            fallback_column='a',
            x_column='times',
            style=StyleConfig(ylabel=r"$a_f$ (AU)", title="Semi-major axis", color='black', marker=',', linestyle=''),
            required=True,
        )
    )

    # Panel 5: Eccentricity
    config.add_panel(
        Panel(
            key='eccentricity',
            data_column='e',
            x_column='times',
            style=StyleConfig(ylabel="e", title="Eccentricity", color='black', marker=',', linestyle=''),
            required=True,
        )
    )

    # Panel 6: Inclination
    config.add_panel(
        Panel(
            key='inclination',
            data_column='inc',
            x_column='times',
            style=StyleConfig(ylabel="i", title="Inclination", color='black', marker=',', linestyle=''),
            required=True,
        )
    )

    # Panel 7: Periodogram of resonant angle
    config.add_panel(
        Panel(
            key='periodogram_angle',
            data_column=f'{resonance_key}_power',
            x_column=f'{resonance_key}_frequency',
            style=StyleConfig(
                ylabel=r"$p_{\sigma}$",
                title="Periodogram (the resonant angle)",
                color='black',
                linestyle='-',
                marker='',
                reference_lines=[
                    {'type': 'axhline', 'y': 0.05, 'color': 'r', 'linestyle': '--'},
                    {'type': 'axhline', 'y': 0.1, 'color': 'g', 'linestyle': '--'},
                ],
            ),
            required=False,  # Periodograms may not exist
        )
    )

    # Panel 8: Periodogram of semi-major axis
    config.add_panel(
        Panel(
            key='periodogram_axis',
            data_column='a_power',
            x_column='a_frequency',
            style=StyleConfig(
                ylabel=r"$p_{a}$",
                title="Periodogram (semi-major axis)",
                color='black',
                linestyle='-',
                marker='',
                reference_lines=[
                    {'type': 'axhline', 'y': 0.05, 'color': 'r', 'linestyle': '--'},
                    {'type': 'axhline', 'y': 0.1, 'color': 'g', 'linestyle': '--'},
                ],
            ),
            required=False,
        )
    )

    # Panel 9: Periodogram of eccentricity
    config.add_panel(
        Panel(
            key='periodogram_ecc',
            data_column='e_power',
            x_column='e_frequency',
            style=StyleConfig(
                ylabel=r"$p_{e}$",
                title="Periodogram (eccentricity)",
                color='black',
                linestyle='-',
                marker='',
                reference_lines=[
                    {'type': 'axhline', 'y': 0.05, 'color': 'r', 'linestyle': '--'},
                    {'type': 'axhline', 'y': 0.1, 'color': 'g', 'linestyle': '--'},
                ],
            ),
            required=False,
        )
    )

    return config


def create_lk_preset(resonance_key: str) -> PlotConfig:
    """
    Create a Lidov-Kozai tuned preset.

    This is the default preset and replicates existing functionality:
    - Resonant angle
    - Semi-major axis (filtered if available)
    - Eccentricity
    - Inclination
    - Periodogram of resonant angle

    Parameters
    ----------
    resonance_key : str
        Resonance key (e.g., '4J-2S-1+0+0-1')

    Returns
    -------
    PlotConfig
        Full LK plot configuration
    """
    config = create_angle_preset(resonance_key)
    config.figsize = (10, 12)

    # Panel 3: Semi-major axis
    config.add_panel(
        Panel(
            key='axis',
            data_column='a_filtered',
            fallback_column='a',
            x_column='times',
            style=StyleConfig(ylabel=r"$a_f$ (AU)", title="Semi-major axis", color='black', marker=',', linestyle=''),
            required=True,
        )
    )

    # Panel 4: Eccentricity
    config.add_panel(
        Panel(
            key='eccentricity',
            data_column='e',
            x_column='times',
            style=StyleConfig(ylabel="e", title="Eccentricity", color='black', marker=',', linestyle=''),
            required=True,
        )
    )

    # Panel 5: Inclination
    config.add_panel(
        Panel(
            key='inclination',
            data_column='inc',
            x_column='times',
            style=StyleConfig(ylabel="i", title="Inclination", color='black', marker=',', linestyle=''),
            required=True,
        )
    )

    # Panel 6: Periodogram of resonant angle
    config.add_panel(
        Panel(
            key='periodogram_angle',
            data_column=f'{resonance_key}_power',
            x_column=f'{resonance_key}_frequency',
            style=StyleConfig(
                ylabel=r"$p_{\sigma}$",
                title="Periodogram (the resonant angle)",
                color='black',
                linestyle='-',
                marker='',
                reference_lines=[
                    {'type': 'axhline', 'y': 0.05, 'color': 'r', 'linestyle': '--'},
                    {'type': 'axhline', 'y': 0.1, 'color': 'g', 'linestyle': '--'},
                ],
            ),
            required=False,  # Periodograms may not exist
        )
    )

    return config


def create_angle_preset(resonance_key: str) -> PlotConfig:
    # Panel 1: Resonant angle
    config = PlotConfig(figsize=(10, 4), plot_title='{body_name}, resonance = {resonance}, status = {status}')
    config.add_panel(
        Panel(
            key='angle',
            data_column=f'{resonance_key}_angle',
            x_column='times',
            style=StyleConfig(ylabel=r"$\sigma$ (rad)", title="Resonant angle", color='black', marker=',', linestyle=''),
            required=True,
        )
    )

    # Panel 2: Wrapped filtered resonant angle
    config.add_panel(
        Panel(
            key='filtered_angle',
            data_column=f'{resonance_key}_angle_filtered',
            x_column='times',
            style=StyleConfig(ylabel=r"$\sigma$ (rad)", title="Filtered resonant angle", color='black', marker=',', linestyle=''),
            required=True,
        )
    )

    # Panel 3: Unwrapped filtered resonant angle
    config.add_panel(
        Panel(
            key='angle_unwrapped',
            data_column=f'{resonance_key}_angle_filtered_unwrapped',
            x_column='times',
            style=StyleConfig(
                ylabel=r"$\sigma_f$ (rad)", title="Unwrapped resonant angle (filtered)", color='black', marker=',', linestyle=''
            ),
            required=False,  # Optional - uses raw angle if missing
        )
    )
    return config


def create_full2pi_preset(resonance_key: str) -> PlotConfig:
    """
    Create a full preset with resonant angle panel limited to [0, 2π].

    Same as 'full' but the first panel (resonant angle) has fixed y-axis limits.
    """
    config = create_full_preset(resonance_key)
    config.panels[0].style.ylim = (0, 2 * np.pi)
    return config


def create_secular_preset(resonance_key: str) -> PlotConfig:
    """
    Create a secular tuned preset.

    This is the default preset and replicates existing functionality:
    - Proper angle
    - Resonant angle (osculating)
    - Semi-major axis (filtered if available)
    - Eccentricity

    Parameters
    ----------
    resonance_key : str
        Resonance key (e.g., 'g-g6+s-s6')

    Returns
    -------
    PlotConfig
        Full secular plot configuration
    """
    config = PlotConfig(figsize=(10, 12), plot_title='{body_name}, resonance = {resonance}, status = {status}')

    # Panel 1: Proper angle
    config.add_panel(
        Panel(
            key='angle_proper',
            data_column='{resonance_key}_angle_proper',
            x_column='times',
            style=StyleConfig(ylabel=r"$\sigma$ (rad)", title="Proper angle", color='black', marker=',', linestyle='', ylim=(0, 2 * np.pi)),
            required=True,
        )
    )

    angle_config = create_angle_preset(resonance_key)
    config.panels.extend(angle_config.panels)

    # Panel 3: Semi-major axis
    config.add_panel(
        Panel(
            key='axis',
            data_column='a_filtered',
            fallback_column='a',
            x_column='times',
            style=StyleConfig(ylabel=r"$a_f$ (AU)", title="Semi-major axis", color='black', marker=',', linestyle=''),
            required=True,
        )
    )
    # Panel 4: Eccentricity
    config.add_panel(
        Panel(
            key='eccentricity',
            data_column='e',
            x_column='times',
            style=StyleConfig(ylabel="e", title="Eccentricity", color='black', marker=',', linestyle=''),
            required=True,
        )
    )

    return config


def get_preset(preset_name: str, resonance_key: str) -> PlotConfig:
    """
    Get a preset configuration by name.

    Parameters
    ----------
    preset_name : str
        Name of preset: 'simple' or 'full'
    resonance_key : str
        Resonance key for column naming

    Returns
    -------
    PlotConfig
        Requested preset configuration

    Raises
    ------
    ValueError
        If preset name is unknown
    """
    presets = {
        'simple': create_simple_preset,
        'full': create_full_preset,
        'full2pi': create_full2pi_preset,
        'lk': create_lk_preset,
        'lkr': create_lk_preset,
        'secular': create_secular_preset,
        'angle': create_angle_preset,
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}")

    return presets[preset_name](resonance_key)
