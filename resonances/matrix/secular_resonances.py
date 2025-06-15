"""
Secular resonance definitions and helper functions.

This module contains the comprehensive list of secular resonance formulas
from the literature and helper functions for working with them.
"""

from typing import Dict
from resonances.config import config


# Complete list of secular resonance formulas from the literature
# Order is calculated as the sum of absolute values of integer coefficients
SECULAR_RESONANCES = {
    # Linear resonances (order 2)
    'g-g5': {'order': 2, 'type': 'linear'},
    'g-g6': {'order': 2, 'type': 'linear'},
    's-s6': {'order': 2, 'type': 'linear'},
    's-s7': {'order': 2, 'type': 'linear'},
    # Degree 4 resonances
    'g5-g6': {'order': 2, 'type': 'nonlinear'},  # This cannot give rise to a resonance (constant)
    's7-s6': {'order': 2, 'type': 'nonlinear'},
    'g+s-s7-g5': {'order': 4, 'type': 'nonlinear'},
    'g+s-s7-g6': {'order': 4, 'type': 'nonlinear'},
    'g+s-s6-g5': {'order': 4, 'type': 'nonlinear'},
    'g+s-s6-g6': {'order': 4, 'type': 'nonlinear'},
    '2g-2s': {'order': 4, 'type': 'kozai'},
    'g-2g5+g6': {'order': 4, 'type': 'nonlinear'},
    'g+g5-2g6': {'order': 4, 'type': 'nonlinear'},
    '2g-g5-g6': {'order': 4, 'type': 'nonlinear'},
    '-g+s+g5-s7': {'order': 4, 'type': 'nonlinear'},
    '-g+s+g6-s7': {'order': 4, 'type': 'nonlinear'},
    '-g+s+g5-s6': {'order': 4, 'type': 'nonlinear'},
    '-g+s+g6-s6': {'order': 4, 'type': 'nonlinear'},
    'g-g5+s7-s6': {'order': 4, 'type': 'nonlinear'},
    'g-g5-s7+s6': {'order': 4, 'type': 'nonlinear'},
    'g-g6+s7-s6': {'order': 4, 'type': 'nonlinear'},
    'g-g6-s7+s6': {'order': 4, 'type': 'nonlinear'},
    '2g-s-s7': {'order': 4, 'type': 'nonlinear'},
    '2g-s-s6': {'order': 4, 'type': 'nonlinear'},
    '-g+2s-g5': {'order': 4, 'type': 'nonlinear'},
    '-g+2s-g6': {'order': 4, 'type': 'nonlinear'},
    '2g-2s7': {'order': 4, 'type': 'nonlinear'},
    '2g-2s6': {'order': 4, 'type': 'nonlinear'},
    '2g-s7-s6': {'order': 4, 'type': 'nonlinear'},
    'g-s+g5-s7': {'order': 4, 'type': 'nonlinear'},
    'g-s+g5-s6': {'order': 4, 'type': 'nonlinear'},
    'g-s+g6-s7': {'order': 4, 'type': 'nonlinear'},
    'g-s+g6-s6': {'order': 4, 'type': 'nonlinear'},
    'g+g5-2s7': {'order': 4, 'type': 'nonlinear'},
    'g+g6-2s7': {'order': 4, 'type': 'nonlinear'},
    'g+g5-2s6': {'order': 4, 'type': 'nonlinear'},
    'g+g6-2s6': {'order': 4, 'type': 'nonlinear'},
    'g+g5-s7-s6': {'order': 4, 'type': 'nonlinear'},
    'g+g6-s7-s6': {'order': 4, 'type': 'nonlinear'},
    's-2s7+s6': {'order': 4, 'type': 'nonlinear'},
    's+s7-2s6': {'order': 4, 'type': 'nonlinear'},
    '2s-s7-s6': {'order': 4, 'type': 'nonlinear'},
    's+g5-g6-s7': {'order': 4, 'type': 'nonlinear'},
    's-g5+g6-s7': {'order': 4, 'type': 'nonlinear'},
    's+g5-g6-s6': {'order': 4, 'type': 'nonlinear'},
    's-g5+g6-s6': {'order': 4, 'type': 'nonlinear'},
    '2s-2g5': {'order': 4, 'type': 'nonlinear'},
    '2s-2g6': {'order': 4, 'type': 'nonlinear'},
    '2s-g5-g6': {'order': 4, 'type': 'nonlinear'},
    's-2g5+s7': {'order': 4, 'type': 'nonlinear'},
    's-2g5+s6': {'order': 4, 'type': 'nonlinear'},
    's-2g6+s7': {'order': 4, 'type': 'nonlinear'},
    's-2g6+s6': {'order': 4, 'type': 'nonlinear'},
    's-g5-g6+s7': {'order': 4, 'type': 'nonlinear'},
    's-g5-g6+s6': {'order': 4, 'type': 'nonlinear'},
    '2g-2g5': {'order': 4, 'type': 'nonlinear'},
    '2g-2g6': {'order': 4, 'type': 'nonlinear'},
    '2s-2s7': {'order': 4, 'type': 'nonlinear'},
    '2s-2s6': {'order': 4, 'type': 'nonlinear'},
    # Degree 6 resonances - divisors appearing only in forced terms
    'g-2g6+g7': {'order': 4, 'type': 'nonlinear'},
    'g-3g6+2g5': {'order': 6, 'type': 'nonlinear'},
    # Degree 6 divisor z2
    '2(g-g6)+(s-s6)': {'order': 6, 'type': 'nonlinear'},
    # Other nonlinear forced terms
    'g+g5-g6-g7': {'order': 4, 'type': 'nonlinear'},
    'g-g5-g6+g7': {'order': 4, 'type': 'nonlinear'},
    'g+g5-2g6-s6+s7': {'order': 6, 'type': 'nonlinear'},
}


def load_planetary_frequencies() -> Dict[str, float]:
    """
    Load planetary frequencies from configuration.

    Returns
    -------
    dict
        Dictionary mapping frequency names to their values in arcsec/yr
    """
    return {
        'g5': float(config.get('g5', 4.25749319)),
        'g6': float(config.get('g6', 28.24552984)),
        'g7': float(config.get('g7', 3.08675577)),
        'g8': float(config.get('g8', 0.67255084)),
        's5': float(config.get('s5', 0.0)),
        's6': float(config.get('s6', -26.34496354)),
        's7': float(config.get('s7', -2.99266093)),
        's8': float(config.get('s8', -0.69251386)),
    }


def get_available_secular_resonance_formulas(order: int = None) -> list:
    """
    Get list of available secular resonance formulas.

    Parameters
    ----------
    order : int, optional
        Filter by resonance order

    Returns
    -------
    list of str
        Available formulas
    """
    if order is not None:
        return [formula for formula, info in SECULAR_RESONANCES.items() if info['order'] == order]
    else:
        return list(SECULAR_RESONANCES.keys())


def get_available_orders() -> list:
    """
    Get list of available resonance orders.

    Returns
    -------
    list of int
        Available orders
    """
    return sorted(set(info['order'] for info in SECULAR_RESONANCES.values()))
