"""Oracles of the plotting tests that the package itself does not need (moved out of `resonances/plotting` by the
author's rule of 2026-11-14: a function only the tests use lives in the tests)."""

from typing import Dict, Iterable, Optional, Sequence

from resonances.plotting.combined_numbers import NumberLine, TextBlock


def printed_lines(blocks: Dict[str, TextBlock], order: Optional[Sequence[str]] = None) -> Iterable[NumberLine]:
    """Every line of every block of the 'all' right column (`combined_numbers.build_blocks`), in print order."""
    for name in list(blocks) if order is None else order:
        if name in blocks:
            yield from blocks[name].lines


def printed_columns(blocks: Dict[str, TextBlock], order: Optional[Sequence[str]] = None) -> Dict[str, NumberLine]:
    """Column -> the line that prints its value (a dropped duplicate maps to the line printed for it)."""
    printed = {}
    for line in printed_lines(blocks, order):
        if line.column and line.label:
            for column in (line.column,) + line.also:
                printed.setdefault(column, line)
    return printed
