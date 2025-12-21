"""
Configuration dataclasses for the plotting system.

This module provides PlotConfig and Panel classes for configuring
customizable plots in the resonances package.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Tuple, List, Any


@dataclass
class StyleConfig:
    """
    Matplotlib styling parameters for a plot panel.

    All parameters use matplotlib-compatible syntax for ease of use.
    """

    # Line/marker styling
    color: str = 'black'
    marker: str = ','
    linestyle: str = ''
    linewidth: float = 1.0
    markersize: Optional[float] = None

    # Axis labels
    ylabel: str = ''
    xlabel: str = ''
    title: str = ''

    # Axis limits
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None

    # Grid and other settings
    grid: bool = False

    # Additional reference lines (e.g., significance thresholds)
    # Format: [{'type': 'axhline', 'y': 0.05, 'color': 'r', 'linestyle': '--'}, ...]
    reference_lines: List[Dict[str, Any]] = field(default_factory=list)

    # Custom matplotlib parameters (direct pass-through)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def merge_with(self, other: 'StyleConfig') -> 'StyleConfig':
        """
        Merge with another StyleConfig, with other taking precedence.

        Parameters
        ----------
        other : StyleConfig
            Configuration to merge with (takes precedence for non-default values)

        Returns
        -------
        StyleConfig
            New merged configuration
        """
        # Create a new StyleConfig with defaults from self
        merged = StyleConfig(
            color=other.color if other.color != 'black' else self.color,
            marker=other.marker if other.marker != ',' else self.marker,
            linestyle=other.linestyle if other.linestyle != '' else self.linestyle,
            linewidth=other.linewidth if other.linewidth != 1.0 else self.linewidth,
            markersize=other.markersize if other.markersize is not None else self.markersize,
            ylabel=other.ylabel if other.ylabel else self.ylabel,
            xlabel=other.xlabel if other.xlabel else self.xlabel,
            title=other.title if other.title else self.title,
            xlim=other.xlim if other.xlim is not None else self.xlim,
            ylim=other.ylim if other.ylim is not None else self.ylim,
            grid=other.grid if other.grid else self.grid,
            reference_lines=other.reference_lines if other.reference_lines else self.reference_lines,
            custom_params={**self.custom_params, **other.custom_params},
        )
        return merged


@dataclass
class Panel:
    """
    Definition of a single plot panel.

    Specifies what data to plot and how to style it.
    """

    # Panel identification
    key: str  # Unique identifier (e.g., 'angle', 'axis', 'periodogram_angle')

    # Data specification
    data_column: str  # Primary column name to plot
    x_column: str = 'times'  # X-axis column (default: times)
    fallback_column: Optional[str] = None  # Alternative if primary missing

    # Optional data transformation
    transform: Optional[Callable[[Any], Any]] = None

    # Styling
    style: StyleConfig = field(default_factory=StyleConfig)

    # Behavior
    required: bool = True  # If False, missing data causes warning, not error
    enabled: bool = True  # If False, panel is skipped

    def __post_init__(self):
        """Ensure style is a StyleConfig instance."""
        if not isinstance(self.style, StyleConfig):
            # Allow dict-style initialization
            if isinstance(self.style, dict):
                self.style = StyleConfig(**self.style)


@dataclass
class PlotConfig:
    """
    Complete configuration for a customizable plot.

    Defines panels to display, global styling, and figure settings.
    """

    # Figure settings
    figsize: Tuple[float, float] = (10, 12)
    dpi: int = 100

    # Panels to display
    panels: List[Panel] = field(default_factory=list)

    # Global styling (defaults for all panels)
    global_style: StyleConfig = field(default_factory=StyleConfig)

    # Font sizes
    title_fontsize: int = 14
    label_fontsize: int = 14
    tick_labelsize: int = 14  # Fontsize for tick labels (axis numbers)

    # Overall plot title template
    plot_title: str = '{body_name}, resonance = {resonance}, status = {status}'

    def add_panel(self, panel: Panel) -> 'PlotConfig':
        """
        Add a panel to the configuration.

        Supports method chaining.

        Parameters
        ----------
        panel : Panel
            Panel to add

        Returns
        -------
        PlotConfig
            Self for chaining
        """
        self.panels.append(panel)
        return self

    def get_enabled_panels(self) -> List[Panel]:
        """
        Get list of enabled panels.

        Returns
        -------
        List[Panel]
            Panels where enabled=True
        """
        return [p for p in self.panels if p.enabled]

    def merge_panel_styles(self, panel: Panel) -> StyleConfig:
        """
        Merge global style with panel-specific style.

        Panel-specific settings override global settings.

        Parameters
        ----------
        panel : Panel
            Panel with potential style overrides

        Returns
        -------
        StyleConfig
            Merged style configuration
        """
        return self.global_style.merge_with(panel.style)

    def format_title(self, **kwargs) -> str:
        """
        Format the plot title with provided values.

        Parameters
        ----------
        **kwargs : dict
            Values for title template (body_name, resonance, status, etc.)

        Returns
        -------
        str
            Formatted title
        """
        try:
            return self.plot_title.format(**kwargs)
        except KeyError:
            return self.plot_title
