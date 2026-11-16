"""The shared test helpers (tests/tools.py)."""

from dataclasses import dataclass

import numpy as np

from tests.tools import same_values


@dataclass
class Inner:
    x: float


@dataclass
class Outer:
    inner: Inner
    table: dict
    values: np.ndarray
    label: tuple


def _outer(x=float('nan'), table_value=float('nan'), last=np.nan):
    return Outer(Inner(x), {'k': [1.0, table_value]}, np.array([1.0, last]), (1, 'a'))


def test_same_values_takes_nan_as_equal_at_every_depth():
    # two separate nan objects: unequal under == on every Python, equal here
    assert Inner(float('nan')) != Inner(float('nan'))
    assert same_values(_outer(), _outer())


def test_same_values_still_tells_different_values_apart():
    assert not same_values(_outer(), _outer(x=1.0))
    assert not same_values(_outer(), _outer(table_value=2.0))
    assert not same_values(_outer(), _outer(last=2.0))
    assert not same_values(Inner(float('nan')), _outer())
    assert not same_values((1.0,), [1.0])
    assert same_values(np.array(['a', 'b']), np.array(['a', 'b']))
