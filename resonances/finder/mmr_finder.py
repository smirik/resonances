"""Backward-compatible MMR finder helpers."""

from .finder import check as _check


def check(asteroids, resonance, name=None, **kwargs):
    """MMR-specific check wrapper with legacy defaults."""
    return _check(asteroids=asteroids, resonance=resonance, name=name or "mmr_check", **kwargs)


__all__ = ["check"]
