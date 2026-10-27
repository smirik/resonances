"""Real result objects for tests of the decision code (W23 S2): a test states the fields it is about, the rest take
neutral defaults, and the decision reads every field directly (no `getattr` with a default for a stub that lacks it).

`make(cls, **fields)` fills every field of a dataclass the test does not give from its annotation: nan for a float,
0 for an int, False for a bool, '' for a str, not_measured for a check, None for an Optional, an empty sequence for a
list, a tuple or an array. The named builders add the nested parts a decision reads (a geometry's centre and turning
points) and the episode's turning points from `n_tp`."""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any

import numpy as np

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import models
from resonances.resonance.analysis.pendulum import EnergyAnalysis
from resonances.resonance.analysis.verdict import CheckValue

NAN = float('nan')


# a neutral value per annotation: the plain types, then the generic containers by their origin
PLAIN = {float: NAN, int: 0, bool: False, str: '', list: [], tuple: (), dict: {}}
CONTAINERS = {list: list, typing.List: list, tuple: tuple, typing.Tuple: tuple, dict: dict, typing.Dict: dict}


def _neutral(hint: Any) -> Any:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        return None
    if origin in CONTAINERS:
        return CONTAINERS[origin]()
    if hint in PLAIN:
        return type(PLAIN[hint])(PLAIN[hint])
    if hint is np.ndarray:
        return np.empty(0)
    if isinstance(hint, type) and issubclass(hint, CheckValue):
        return CheckValue.NOT_MEASURED
    if isinstance(hint, type) and issubclass(hint, str):  # another StrEnum: its empty member, else its first
        return hint('') if '' in {str(member) for member in hint} else next(iter(hint))
    return None


def make(cls, **fields):
    """An instance of the dataclass `cls` with `fields`; every other field without a default gets a neutral value."""
    hints = typing.get_type_hints(cls)
    for field in dataclasses.fields(cls):
        missing = field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
        if field.init and missing and field.name not in fields:
            fields[field.name] = _neutral(hints[field.name])
    return cls(**fields)


def make_turning_points(times=(), values=None, prominence_rad: float = 0.0, local=()) -> g.TurningPoints:
    """Turning points at `times` [yr] with `values` [rad] (0 without), alternately a maximum and a minimum."""
    times = np.asarray(times, dtype=float)
    values = np.zeros(len(times)) if values is None else np.asarray(values, dtype=float)
    kinds = np.where(np.arange(len(times)) % 2 == 0, 1, -1)
    return g.TurningPoints(np.arange(len(times)), times, values, kinds, prominence_rad, tuple(local))


def make_centre(**fields) -> g.CentreTrack:
    fields.setdefault('turns', make_turning_points())
    fields.setdefault('passes', [])
    return make(g.CentreTrack, **fields)


def make_episode(n_tp: int = 3, **fields) -> g.Episode:
    """An accepted episode (`accepted` True unless given) with `n_tp` turning points one year apart from t = 0 unless
    `tp_times_yr` is given."""
    fields.setdefault('tp_times_yr', np.arange(float(n_tp)))
    fields.setdefault('tp_values_rad', np.zeros(len(fields['tp_times_yr'])))
    fields.setdefault('accepted', True)
    return make(g.Episode, **fields)


def make_geometry(**fields) -> g.Geometry:
    """A geometry with no turning points, a centre without samples and the window (0, 1) yr unless given."""
    fields.setdefault('window', (0.0, 1.0))
    fields.setdefault('turning_points', make_turning_points())
    fields.setdefault('centre', make_centre())
    return make(g.Geometry, **fields)


def make_response(**fields) -> models.Response:
    return make(models.Response, **fields)


def make_sampling(**fields) -> models.Sampling:
    return make(models.Sampling, **fields)


def make_kinematics(**fields) -> models.Kinematics:
    return make(models.Kinematics, **fields)


def make_energy(**fields) -> EnergyAnalysis:
    return make(EnergyAnalysis, **fields)
