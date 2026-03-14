"""
Phase portrait plots.

Provides specialized plotting functions for:
- Phase portrait (filtered): sigma_dot vs sigma (mod 2pi) with time colormap using filtered angle
- Phase portrait (unfiltered): same but using raw unwrapped angle
- Phase portrait (slow points): only points where |sigma_dot| is below a percentile threshold
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union

from resonances.body import Body
from resonances.resonance.classify.classify import calc_sigma_derivative
from resonances.logger import logger


class PhasePlotter:
    """
    Plotter for phase portraits.

    Examples
    --------
    From Body object:
        >>> plotter = PhasePlotter.from_body(body, resonance, sim)
        >>> plotter.plot_phase_portrait_filtered().save('phase_filtered.png')
        >>> plotter.plot_phase_portrait_unfiltered().save('phase_unfiltered.png')
        >>> plotter.plot_phase_portrait_slow(percentile=95).save('phase_slow.png')
    """

    def __init__(self):
        """Initialize plotter with empty state."""
        self._times: Optional[np.ndarray] = None
        # Filtered angle data
        self._sigma_filtered: Optional[np.ndarray] = None
        self._sigma_dot_filtered: Optional[np.ndarray] = None
        # Unfiltered angle data
        self._sigma_unfiltered: Optional[np.ndarray] = None
        self._sigma_dot_unfiltered: Optional[np.ndarray] = None
        self._metadata: Optional[dict] = None
        self._figure = None
        self._ax = None

    @classmethod
    def from_body(cls, body: Body, resonance, sim) -> 'PhasePlotter':
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
        PhasePlotter
            Configured plotter instance
        """
        plotter = cls()
        plotter._load_from_body(body, resonance, sim)
        return plotter

    @classmethod
    def from_data(
        cls,
        times: np.ndarray,
        sigma_unwrapped: np.ndarray,
        sigma_filtered: Optional[np.ndarray] = None,
        body_name: str = '',
        resonance_key: str = '',
    ) -> 'PhasePlotter':
        """
        Create plotter from raw data arrays.

        Parameters
        ----------
        times : np.ndarray
            Time array (in years, i.e., already divided by 2*pi)
        sigma_unwrapped : np.ndarray
            Unwrapped resonant angle (unfiltered)
        sigma_filtered : np.ndarray, optional
            Filtered unwrapped resonant angle. If None, uses sigma_unwrapped.
        body_name : str
            Name of the body for title
        resonance_key : str
            Resonance key for title

        Returns
        -------
        PhasePlotter
            Configured plotter instance
        """
        plotter = cls()
        plotter._times = times
        plotter._sigma_unfiltered = sigma_unwrapped
        plotter._sigma_dot_unfiltered = calc_sigma_derivative(times, sigma_unwrapped)
        plotter._sigma_filtered = sigma_filtered if sigma_filtered is not None else sigma_unwrapped
        plotter._sigma_dot_filtered = calc_sigma_derivative(times, plotter._sigma_filtered)
        plotter._metadata = {
            'body_name': body_name,
            'resonance_key': resonance_key,
            'tmax_years': times[-1] if len(times) > 0 else 0,
        }
        return plotter

    def _load_from_body(self, body: Body, resonance, sim):
        """Load data from Body object."""
        resonance_key = resonance.to_s()

        # Get times in years
        self._times = sim.times / (2 * np.pi)

        # Get unfiltered unwrapped angle
        self._sigma_unfiltered = body.angles_unwrapped.get(resonance_key)
        if self._sigma_unfiltered is not None:
            self._sigma_dot_unfiltered = calc_sigma_derivative(self._times, self._sigma_unfiltered)

        # Get filtered unwrapped angle
        self._sigma_filtered = body.angles_filtered_unwrapped.get(resonance_key)
        if self._sigma_filtered is None:
            logger.warning(f"No filtered unwrapped angle for {resonance_key}, using raw unwrapped")
            self._sigma_filtered = self._sigma_unfiltered
            self._sigma_dot_filtered = self._sigma_dot_unfiltered
        else:
            self._sigma_dot_filtered = calc_sigma_derivative(self._times, self._sigma_filtered)

        self._metadata = {
            'body_name': body.name,
            'resonance': resonance.to_short(),
            'resonance_key': resonance_key,
            'status': body.statuses.get(resonance_key, 0),
            'tmax_years': abs(sim.config.tmax_yrs),
        }

    def _plot_phase_portrait_base(
        self,
        sigma: np.ndarray,
        sigma_dot: np.ndarray,
        times: np.ndarray,
        title_suffix: str,
        figsize: tuple = (8, 6),
        dpi: int = 100,
        marker_size: float = 7.5,
    ) -> 'PhasePlotter':
        """
        Base method for plotting phase portrait.

        Parameters
        ----------
        sigma : np.ndarray
            Unwrapped angle data
        sigma_dot : np.ndarray
            Derivative of angle
        times : np.ndarray
            Time array for colormap
        title_suffix : str
            Suffix for the plot title (e.g., "filtered", "unfiltered", "slow")
        figsize : tuple
            Figure size (width, height)
        dpi : int
            Figure resolution
        marker_size : float
            Size of scatter points

        Returns
        -------
        PhasePlotter
            Self for method chaining
        """
        import matplotlib.pyplot as plt

        # Close any existing figure to prevent memory leaks
        if self._figure is not None:
            plt.close(self._figure)

        self._figure, self._ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Wrap sigma to [0, 2pi) and convert to degrees
        sigma_wrapped = sigma % (2 * np.pi)
        sigma_degrees = np.rad2deg(sigma_wrapped)

        # Create scatter plot with time colormap (dots only, no lines)
        scatter = self._ax.scatter(
            sigma_degrees,
            sigma_dot,
            c=times,
            cmap='viridis',
            s=marker_size,
            marker='.',
            linewidths=0,
            alpha=0.7,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=self._ax)
        cbar.set_label('Time', fontsize=10)

        # Labels and title
        self._ax.set_xlabel(r'$\sigma$ mod 360°', fontsize=12)
        self._ax.set_ylabel(r'$\dot{\sigma}$', fontsize=12)

        title = f'Phase portrait ({title_suffix})'
        if self._metadata:
            body_name = self._metadata.get('body_name', '')
            resonance = self._metadata.get('resonance', self._metadata.get('resonance_key', ''))
            if body_name or resonance:
                title = f'{body_name}: {resonance} - {title}'
        self._ax.set_title(title, fontsize=12)

        # Set x-axis limits to [0, 360] degrees with ticks at 90, 180, 270, 360
        self._ax.set_xlim(0, 360)
        self._ax.set_xticks([0, 90, 180, 270, 360])

        # Add vertical dashed lines at 90, 180, 270 degrees
        for deg in [90, 180, 270]:
            self._ax.axvline(x=deg, color='lightgray', linestyle='--', linewidth=0.8, alpha=0.8)

        # Add horizontal line at y=0
        self._ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()

        return self

    def plot_phase_portrait(self, figsize: tuple = (8, 6), dpi: int = 100) -> 'PhasePlotter':
        """
        Plot phase portrait using filtered angle (default behavior).

        Alias for plot_phase_portrait_filtered() for backwards compatibility.
        """
        return self.plot_phase_portrait_filtered(figsize=figsize, dpi=dpi)

    def plot_phase_portrait_filtered(self, figsize: tuple = (8, 6), dpi: int = 100) -> 'PhasePlotter':
        """
        Plot phase portrait using filtered unwrapped angle.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        dpi : int
            Figure resolution

        Returns
        -------
        PhasePlotter
            Self for method chaining
        """
        if self._sigma_filtered is None or self._sigma_dot_filtered is None:
            logger.warning("No filtered data available for phase portrait")
            return self

        return self._plot_phase_portrait_base(
            self._sigma_filtered,
            self._sigma_dot_filtered,
            self._times,
            'filtered',
            figsize=figsize,
            dpi=dpi,
        )

    def plot_phase_portrait_unfiltered(self, figsize: tuple = (8, 6), dpi: int = 100) -> 'PhasePlotter':
        """
        Plot phase portrait using raw (unfiltered) unwrapped angle.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        dpi : int
            Figure resolution

        Returns
        -------
        PhasePlotter
            Self for method chaining
        """
        if self._sigma_unfiltered is None or self._sigma_dot_unfiltered is None:
            logger.warning("No unfiltered data available for phase portrait")
            return self

        return self._plot_phase_portrait_base(
            self._sigma_unfiltered,
            self._sigma_dot_unfiltered,
            self._times,
            'unfiltered',
            figsize=figsize,
            dpi=dpi,
        )

    def plot_phase_portrait_slow(self, percentile: float = 95, figsize: tuple = (8, 6), dpi: int = 100) -> 'PhasePlotter':
        """
        Plot phase portrait showing only slow points (low |sigma_dot|).

        Uses unfiltered angle data. Points are filtered to only include those
        where |sigma_dot| is below the specified percentile threshold.

        Parameters
        ----------
        percentile : float
            Percentile threshold (0-100). Points with |sigma_dot| below this
            percentile of all |sigma_dot| values are shown. Default 95.
        figsize : tuple
            Figure size (width, height)
        dpi : int
            Figure resolution

        Returns
        -------
        PhasePlotter
            Self for method chaining
        """
        if self._sigma_unfiltered is None or self._sigma_dot_unfiltered is None:
            logger.warning("No unfiltered data available for slow phase portrait")
            return self

        # Calculate threshold and create mask
        sigma_dot_threshold = np.percentile(np.abs(self._sigma_dot_unfiltered), percentile)
        mask_slow = np.abs(self._sigma_dot_unfiltered) < sigma_dot_threshold

        # Filter data
        sigma_slow = self._sigma_unfiltered[mask_slow]
        sigma_dot_slow = self._sigma_dot_unfiltered[mask_slow]
        times_slow = self._times[mask_slow]

        if len(sigma_slow) == 0:
            logger.warning("No slow points found for phase portrait")
            return self

        return self._plot_phase_portrait_base(
            sigma_slow,
            sigma_dot_slow,
            times_slow,
            f'slow, {percentile}%',
            figsize=figsize,
            dpi=dpi,
            marker_size=15.0,  # Slightly larger markers for slow points
        )

    def save(self, path: Union[str, Path], **kwargs) -> 'PhasePlotter':
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
        PhasePlotter
            Self for method chaining
        """
        if self._figure is None:
            raise RuntimeError("Must call a plot method before save()")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(path, **kwargs)
        logger.info(f"Plot saved to {path}")

        return self

    def show(self) -> 'PhasePlotter':
        """
        Display the figure.

        Returns
        -------
        PhasePlotter
            Self for method chaining
        """
        if self._figure is None:
            raise RuntimeError("Must call a plot method before show()")

        import matplotlib.pyplot as plt

        plt.show()
        return self

    def close(self):
        """Close the figure to free memory."""
        import matplotlib.pyplot as plt

        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
            self._ax = None
