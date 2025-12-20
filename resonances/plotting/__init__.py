"""
Configurable plotting system for resonances package.

Provides flexible, type-safe plotting configuration with support for
both live Body objects and CSV files.

Main exports:
- Plotter: Main plotting interface with method chaining
- PlotConfig: Configuration container
- Panel: Panel definition
- StyleConfig: Styling configuration
- get_preset: Load preset configurations
"""

from .plotter import Plotter
from .config import PlotConfig, Panel, StyleConfig
from .presets import get_preset, create_simple_preset, create_full_preset

__all__ = [
    'Plotter',
    'PlotConfig',
    'Panel',
    'StyleConfig',
    'get_preset',
    'create_simple_preset',
    'create_full_preset',
]
