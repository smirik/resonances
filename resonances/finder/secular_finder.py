"""Backward-compatible secular finder helpers."""

from .finder import check as _check
from .finder import find as _find


def check(asteroids, resonance, name=None, integration_years=None, **kwargs):
    """Secular-specific check wrapper with legacy defaults."""
    if integration_years is not None:
        kwargs["integration_years"] = integration_years
    return _check(asteroids=asteroids, resonance=resonance, name=name or "secular_check", **kwargs)


def find(asteroids, resonance=None, name=None, integration_years=None, **kwargs):
    """Secular-specific find wrapper with legacy defaults."""
    if integration_years is not None:
        kwargs["integration_years"] = integration_years
    return _find(
        asteroids=asteroids,
        formulas=resonance,
        type="secular",
        name=name or "secular_find",
        **kwargs,
    )


__all__ = ["check", "find"]
