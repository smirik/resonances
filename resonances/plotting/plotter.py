"""
Main plotter class for configurable plotting.

Provides Plotter class with method chaining API for creating customizable plots
from both live Body objects and CSV files.
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Union, Optional, Dict, Any

from resonances.body import Body
from resonances.logger import logger
from resonances.secular.secular_resonance import SecularResonance
from .config import PlotConfig, Panel, StyleConfig
from .presets import get_preset


def round_to_nice_value(value):
    """Round value to a nice number ending with zeros."""
    if value <= 0:
        return 0
    power = 10 ** math.floor(math.log10(value))
    rounded = round(value / power) * power
    return int(rounded)


class Plotter:
    """
    Main plotting interface with fluent API.

    Supports creating plots from both live Body objects and CSV files.
    Provides method chaining for convenient configuration and rendering.

    Examples
    --------
    From Body object:
        >>> plotter = Plotter.from_body(body, resonance, sim)
        >>> plotter.configure('full').plot().save('output.png')

    From CSV file:
        >>> plotter = Plotter.from_csv('data-463.csv', '4J-2S-1+0+0-1')
        >>> plotter.configure('simple').plot().show()

    Custom configuration:
        >>> config = PlotConfig()
        >>> config.add_panel(Panel(...))
        >>> plotter.configure(config).plot().save('custom.png')
    """

    def __init__(self):
        """Initialize plotter with empty state."""
        self._data: Optional[Dict[str, np.ndarray]] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._config: Optional[PlotConfig] = None
        self._periodogram_data: Optional[pd.DataFrame] = None
        self._figure = None
        self._axes = None

    @classmethod
    def from_body(cls, body: Body, resonance, sim) -> 'Plotter':
        """
        Create plotter from live Body object.

        Parameters
        ----------
        body : Body
            Body object with simulation data
        resonance : Resonance
            Resonance object (MMR, Secular, or LidovKozai)
        sim : Simulation
            Simulation object with times and config

        Returns
        -------
        Plotter
            Configured plotter instance
        """
        plotter = cls()
        plotter._load_from_body(body, resonance, sim)
        return plotter

    @classmethod
    def from_csv(cls, data_path: Union[str, Path], resonance_key: str, periodogram_path: Optional[Union[str, Path]] = None) -> 'Plotter':
        """
        Create plotter from CSV files.

        Parameters
        ----------
        data_path : str or Path
            Path to main data CSV file (data-{name}.csv)
        resonance_key : str
            Resonance key for column selection (e.g., '4J-2S-1+0+0-1')
        periodogram_path : str or Path, optional
            Path to periodogram CSV file. If None, auto-detects from data_path

        Returns
        -------
        Plotter
            Configured plotter instance
        """
        plotter = cls()
        plotter._load_from_csv(data_path, resonance_key, periodogram_path)
        return plotter

    def configure(self, config: Union[str, PlotConfig]) -> 'Plotter':
        """
        Configure the plotter.

        Parameters
        ----------
        config : str or PlotConfig
            Either preset name ('simple', 'full') or PlotConfig object

        Returns
        -------
        Plotter
            Self for method chaining
        """
        if isinstance(config, str):
            # Load preset
            resonance_key = self._metadata.get('resonance_key', '')
            self._config = get_preset(config, resonance_key)
        elif isinstance(config, PlotConfig):
            self._config = config
        else:
            raise TypeError(f"config must be str or PlotConfig, got {type(config)}")

        return self

    def plot(self) -> 'Plotter':
        """
        Render the plot.

        Creates matplotlib Figure and Axes according to configuration.

        Returns
        -------
        Plotter
            Self for method chaining
        """
        if self._config is None:
            # Default to 'full' preset
            self.configure('full')

        enabled_panels = self._config.get_enabled_panels()

        if not enabled_panels:
            logger.warning("No enabled panels to plot")
            return self

        import matplotlib.pyplot as plt

        # Create figure and axes
        n_panels = len(enabled_panels)
        self._figure, self._axes = plt.subplots(n_panels, 1, figsize=self._config.figsize, dpi=self._config.dpi)

        # Ensure axes is always a list
        if n_panels == 1:
            self._axes = [self._axes]

        # Set overall title
        if self._metadata:
            title = self._config.format_title(**self._metadata)
            self._figure.suptitle(title, fontsize=self._config.title_fontsize)

        # Render each panel
        tmax_years = self._metadata.get('tmax_years', None)
        for idx, panel in enumerate(enabled_panels):
            self._render_panel(self._axes[idx], panel, tmax_years)

        # Share x-axis for time-series plots
        self._setup_shared_axes(enabled_panels)

        plt.tight_layout()

        return self

    def save(self, path: Union[str, Path], **kwargs) -> 'Plotter':
        """
        Save the figure to file.

        Parameters
        ----------
        path : str or Path
            Output file path
        **kwargs : dict
            Additional arguments passed to plt.savefig()

        Returns
        -------
        Plotter
            Self for method chaining
        """
        if self._figure is None:
            raise RuntimeError("Must call plot() before save()")

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._figure.savefig(path, **kwargs)
        logger.info(f"Plot saved to {path}")

        return self

    def show(self) -> 'Plotter':
        """
        Display the figure.

        Returns
        -------
        Plotter
            Self for method chaining
        """
        if self._figure is None:
            raise RuntimeError("Must call plot() before show()")

        import matplotlib.pyplot as plt

        plt.show()
        return self

    def close(self):
        """Close the figure to free memory."""
        import matplotlib.pyplot as plt

        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
            self._axes = None

    def _load_from_body(self, body: Body, resonance, sim):  # noqa: C901
        """Load data from Body object."""
        resonance_key = resonance.to_s()

        # Extract data into dictionary
        self._data = {
            'times': sim.times / (2 * np.pi),
            'a': body.axis,
            'e': body.ecc,
            'inc': body.inc,
            'Omega': body.Omega,
            'omega': body.omega,
            'M': body.M,
            'longitude': body.longitude,
            'varpi': body.varpi,
        }

        # Add filtered axis if available
        if body.axis_filtered is not None:
            self._data['a_filtered'] = body.axis_filtered

        # Add resonance angles
        angle_data = body.angle(resonance)
        if angle_data is not None:
            self._data[f'{resonance_key}_angle'] = angle_data
            self._data[f'{resonance_key}_angle_unwrapped'] = body.angles_filtered_unwrapped[resonance_key]

        # Add filtered angles if available
        if resonance_key in body.angles_filtered:
            self._data[f'{resonance_key}_angle_filtered'] = body.angles_filtered[resonance_key]

        if resonance_key in body.angles_filtered_unwrapped:
            self._data[f'{resonance_key}_angle_filtered_unwrapped'] = body.angles_filtered_unwrapped[resonance_key]

        # Add proper angles if available
        if isinstance(resonance, SecularResonance):
            self._data[f'{resonance_key}_angle_proper'] = body.secular_angles_proper[resonance_key]
            self._data[f'{resonance_key}_angle_osculating'] = body.secular_angles_osculating[resonance_key]

        # Create periodogram dataframe if data exists
        periodogram_dict = {}

        # Semi-major axis periodogram
        if body.axis_periodogram_frequency is not None:
            periodogram_dict['a_frequency'] = body.axis_periodogram_frequency
            periodogram_dict['a_power'] = body.axis_periodogram_power

        # Eccentricity periodogram
        if body.eccentricity_periodogram_frequency is not None:
            periodogram_dict['e_frequency'] = body.eccentricity_periodogram_frequency
            periodogram_dict['e_power'] = body.eccentricity_periodogram_power

        # Resonance angle periodogram
        if resonance_key in body.periodogram_frequency:
            periodogram_dict[f'{resonance_key}_frequency'] = body.periodogram_frequency[resonance_key]
            periodogram_dict[f'{resonance_key}_power'] = body.periodogram_power[resonance_key]

        if periodogram_dict:
            try:
                # Check if all values are arrays (not scalars)
                all_arrays = all(isinstance(v, np.ndarray) and v.size > 0 for v in periodogram_dict.values())
                if all_arrays:
                    self._periodogram_data = pd.DataFrame(periodogram_dict)
                else:
                    logger.warning(f"Periodogram data contains non-array values, skipping DataFrame creation")
                    self._periodogram_data = None
            except Exception as e:
                logger.warning(f"Error creating periodogram DataFrame: {e}")
                self._periodogram_data = None

        # Store periodogram peaks for plotting
        self._periodogram_peaks = {
            'angle': body.periodogram_peaks.get(resonance_key),
            'axis': body.axis_periodogram_peaks,
            'ecc': body.eccentricity_periodogram_peaks,
        }

        # Metadata
        self._metadata = {
            'body_name': body.name,
            'resonance': resonance.to_short(),
            'resonance_key': resonance_key,
            'status': body.statuses.get(resonance_key, 0),
            'tmax_years': abs(sim.config.tmax_yrs),
        }

    def _load_from_csv(self, data_path: Union[str, Path], resonance_key: str, periodogram_path: Optional[Union[str, Path]] = None):
        """Load data from CSV files."""
        data_path = Path(data_path)

        # Load main data
        df = pd.read_csv(data_path, index_col=0)
        self._data = {col: df[col].values for col in df.columns}

        # Auto-detect periodogram path if not provided
        if periodogram_path is None:
            # Assuming naming convention: data-{name}.csv -> data-{name}-periodograms.csv
            periodogram_path = data_path.parent / f"{data_path.stem}-periodograms.csv"

        # Load periodogram data if exists
        if Path(periodogram_path).exists():
            self._periodogram_data = pd.read_csv(periodogram_path)
        else:
            logger.warning(f"Periodogram file not found: {periodogram_path}")
            self._periodogram_data = None

        # No peak information from CSV
        self._periodogram_peaks = {}

        # Extract metadata from data
        body_name = data_path.stem.replace('data-', '')
        tmax_years = self._data['times'].max() if 'times' in self._data else 100000

        self._metadata = {
            'body_name': body_name,
            'resonance': resonance_key,
            'resonance_key': resonance_key,
            'status': 'N/A',
            'tmax_years': tmax_years,
        }

    def _render_panel(self, ax, panel: Panel, tmax_years: Optional[float]):
        """Render a single panel."""
        # Get data for this panel
        x_data = self._get_column_data(panel.x_column)
        y_data = self._get_column_data(panel.data_column, panel.fallback_column)

        if x_data is None or y_data is None:
            self._handle_missing_data(ax, panel)
            return

        # Apply transformation if specified
        if panel.transform is not None:
            try:
                y_data = panel.transform(y_data)
            except Exception as e:
                logger.error(f"Error applying transformation to panel {panel.key}: {e}")
                self._handle_missing_data(ax, panel)
                return

        # Get merged style
        style = self._config.merge_panel_styles(panel)

        # Determine if this is a periodogram (x-axis is frequency)
        is_periodogram = 'frequency' in panel.x_column

        if is_periodogram:
            self._render_periodogram_panel(ax, panel, x_data, y_data, style, tmax_years)
        else:
            self._render_timeseries_panel(ax, x_data, y_data, style, tmax_years)

        # Apply common styling
        self._apply_panel_style(ax, style, panel)

    def _render_timeseries_panel(self, ax, x_data, y_data, style: StyleConfig, tmax_years):
        """Render a time-series panel."""
        import matplotlib.pyplot as plt

        # Plot data
        ax.plot(
            x_data,
            y_data,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            linewidth=style.linewidth,
            markersize=style.markersize,
        )

        # Set x-axis limits and ticks
        if tmax_years is not None:
            ax.set_xlim([0, abs(tmax_years)])

            # Adaptive tick spacing
            major_tick = round_to_nice_value(abs(tmax_years) / 5)
            minor_tick = major_tick // 5

            if major_tick <= 0:
                major_tick = 1
            if minor_tick <= 0:
                minor_tick = 1

            ax.xaxis.set_major_locator(plt.MultipleLocator(major_tick))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_tick))

    def _render_periodogram_panel(self, ax, panel: Panel, x_data, y_data, style: StyleConfig, tmax_years):
        """Render a periodogram panel."""
        # Convert frequency to period
        period_data = 1.0 / x_data

        # Plot main periodogram line
        ax.plot(period_data, y_data, color=style.color, linestyle=style.linestyle if style.linestyle else '-', linewidth=style.linewidth)

        # Plot peaks if available
        peak_info = None
        if 'angle' in panel.key and 'angle' in self._periodogram_peaks:
            peak_info = self._periodogram_peaks['angle']
        elif 'axis' in panel.key and 'axis' in self._periodogram_peaks:
            peak_info = self._periodogram_peaks['axis']
        elif 'ecc' in panel.key and 'ecc' in self._periodogram_peaks:
            peak_info = self._periodogram_peaks['ecc']

        if peak_info is not None and peak_info:
            if 'peaks' in peak_info and peak_info['peaks'].size > 0:
                peaks = peak_info['peaks']

                # Plot peak positions with gray lines
                if 'position' in peak_info:
                    for peak_width in peak_info['position']:
                        ax.axvline(x=peak_width[0], color='gray', linestyle='dashed')
                        ax.axvline(x=peak_width[1], color='gray', linestyle='--')

                # Plot peak markers
                peak_periods = 1.0 / x_data[peaks]
                peak_powers = y_data[peaks]
                ax.plot(peak_periods, peak_powers, 'x', color='blue', markersize=10)

        # Set x-axis limits for periodograms
        if tmax_years is not None:
            ax.set_xlim(0, abs(tmax_years))

    def _apply_panel_style(self, ax, style: StyleConfig, panel: Panel):  # noqa: C901
        """Apply styling to a panel."""
        # Title
        if style.title:
            ax.set_title(style.title)

        # Axis labels
        if style.ylabel:
            ax.set_ylabel(style.ylabel, fontsize=self._config.label_fontsize)
        if style.xlabel:
            ax.set_xlabel(style.xlabel, fontsize=self._config.label_fontsize)

        # Tick label fontsize
        ax.tick_params(axis='both', which='major', labelsize=self._config.tick_labelsize)

        # Axis limits
        if style.xlim is not None:
            ax.set_xlim(style.xlim)
        if style.ylim is not None:
            ax.set_ylim(style.ylim)

        # Grid
        if style.grid:
            ax.grid(True)

        # Reference lines
        for ref_line in style.reference_lines:
            line_type = ref_line.pop('type', 'axhline')
            if line_type == 'axhline':
                ax.axhline(**ref_line)
            elif line_type == 'axvline':
                ax.axvline(**ref_line)

        # Custom parameters
        if style.custom_params:
            for key, value in style.custom_params.items():
                if hasattr(ax, f'set_{key}'):
                    getattr(ax, f'set_{key}')(value)

    def _handle_missing_data(self, ax, panel: Panel):
        """Handle missing data for a panel."""
        if panel.required:
            logger.warning(f"Required data missing for panel '{panel.key}': {panel.data_column}")
        else:
            logger.info(f"Optional data missing for panel '{panel.key}': {panel.data_column}, panel will be empty")

        # Create empty panel with message
        ax.text(
            0.5,
            0.5,
            f"Data not available:\n{panel.data_column}",
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=10,
            color='gray',
        )
        ax.set_title(panel.style.title if panel.style.title else panel.key)
        ax.set_xticks([])
        ax.set_yticks([])

    def _replace_placeholders(self, column_name: str) -> str:
        """
        Replace placeholders in column name with actual values.

        Supports:
        - {resonance_key}: Full resonance key (e.g., '4J-2S-1+0+0-1')
        - {body_name}: Body name

        Parameters
        ----------
        column_name : str
            Column name potentially containing placeholders

        Returns
        -------
        str
            Column name with placeholders replaced
        """
        if not self._metadata:
            return column_name

        replacements = {'{resonance_key}': self._metadata.get('resonance_key', ''), '{body_name}': self._metadata.get('body_name', '')}

        result = column_name
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, str(value))

        return result

    def _get_column_data(self, column_name: str, fallback: Optional[str] = None) -> Optional[np.ndarray]:
        """Get data column, with optional fallback and placeholder replacement."""
        # Replace placeholders in column names
        column_name = self._replace_placeholders(column_name)
        if fallback:
            fallback = self._replace_placeholders(fallback)

        # Try main data first
        if column_name in self._data:
            return self._data[column_name]

        # Try periodogram data
        if self._periodogram_data is not None and column_name in self._periodogram_data.columns:
            return self._periodogram_data[column_name].values

        # Try fallback
        if fallback:
            if fallback in self._data:
                return self._data[fallback]
            if self._periodogram_data is not None and fallback in self._periodogram_data.columns:
                return self._periodogram_data[fallback].values

        return None

    def _setup_shared_axes(self, panels):
        """Setup shared x-axes for related panels."""
        if len(self._axes) < 2:
            return

        # Find time-series panels (x_column='times')
        timeseries_indices = []
        periodogram_indices = []

        for idx, panel in enumerate(panels):
            if panel.x_column == 'times':
                timeseries_indices.append(idx)
            elif 'frequency' in panel.x_column:
                periodogram_indices.append(idx)

        # Share x-axis for time-series panels
        if len(timeseries_indices) > 1:
            first_idx = timeseries_indices[0]
            for idx in timeseries_indices[1:]:
                self._axes[idx].sharex(self._axes[first_idx])

        # Share x-axis for periodogram panels
        if len(periodogram_indices) > 1:
            first_idx = periodogram_indices[0]
            for idx in periodogram_indices[1:]:
                self._axes[idx].sharex(self._axes[first_idx])
