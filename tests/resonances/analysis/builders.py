"""Real result objects for tests of the decision code (W23 S2): a test states the fields it is about, the rest take
neutral defaults, and the decision reads every field directly (no `getattr` with a default for a stub that lacks it).

`make(cls, **fields)` fills every field of a dataclass the test does not give from its annotation: nan for a float,
0 for an int, False for a bool, '' for a str, not_measured for a check, None for an Optional, an empty sequence for a
list, a tuple, a frozenset (fired rules, `analysis.rules`) or an array. The named builders add the nested parts a
decision reads (a geometry's centre and turning points) and the episode's turning points from `n_tp`."""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any

import numpy as np

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import models
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.pendulum import EnergyAnalysis
from resonances.resonance.analysis.secular import models as secular_models
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue, ResonanceType, Status, Verdict

NAN = float('nan')


# a neutral value per annotation: the plain types, then the generic containers by their origin
PLAIN = {float: NAN, int: 0, bool: False, str: '', list: [], tuple: (), dict: {}, frozenset: frozenset()}
CONTAINERS = {
    list: list,
    typing.List: list,
    tuple: tuple,
    typing.Tuple: tuple,
    dict: dict,
    typing.Dict: dict,
    frozenset: frozenset,
    typing.FrozenSet: frozenset,
}


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


def make_verdict(status=Status.NON_RESONANT, angle_status=AngleStatus.CIRCULATION, **fields) -> Verdict:
    """A verdict of the tree (MMR unless `resonance_type` is given) with `status` and `angle_status`."""
    fields.setdefault('resonance_type', ResonanceType.MMR)
    return Verdict(status=Status(status), angle_status=AngleStatus(angle_status), **fields)


def make_mmr_analysis(
    verdict: Verdict, geometry=None, rec_rule: str = '', extra=None, span_yr: float = 1000.0, **fields
) -> models.MMRAnalysis:
    """A finished MMR analysis as the policy layer reads it (W29): the tree's `verdict`, its leaf `rec_rule`, the
    geometry, a sampling of `span_yr` and any other result object given; the rest neutral."""
    fields.setdefault('sampling', make_sampling(span_yr=span_yr, resample_step_yr=None, fold_free_step_yr=NAN))
    extra_columns = {'rec_rule': rec_rule, **(extra or {})}
    return make(models.MMRAnalysis, verdict=verdict, geometry=geometry, extra_columns=extra_columns, **fields)


def make_filtered_angle(t_yr, angle_f, window=None, **fields) -> g.FilteredAngle:
    """sigma_f on `t_yr` (the window the whole grid unless given); the other arrays empty."""
    t_yr = np.asarray(t_yr, dtype=float)
    fields.setdefault('run_break_half_amplitude_rad', np.pi)
    window = window if window is not None else (float(t_yr[0]), float(t_yr[-1]))
    return make(g.FilteredAngle, t_yr=t_yr, angle_f=np.asarray(angle_f, dtype=float), window=window, **fields)


def make_long_period_analysis(
    verdict: Verdict, geometry=None, gate=None, angle=None, span_yr: float = 1.0e6, **fields
) -> secular_models.LongPeriodAnalysis:
    """A finished secular or ZLK analysis (the type of `verdict`) over `span_yr` as the policy layer reads it."""
    t = np.array([0.0, span_yr])
    series = secular_models.LongPeriodSeries(verdict.resonance_type, 'K', 'body', t, np.zeros(2), np.ones(2), np.zeros(2), np.zeros(2))
    return secular_models.LongPeriodAnalysis(series, verdict, Ledger(0.1), geometry=geometry, gate=gate, angle=angle, **fields)


# --------------------------------------------------------------------------------------------- the W28 tree (W34)


class TreeVerdict(typing.NamedTuple):
    """The W28 tree's verdict that the minimal policy keeps on the final one (`Verdict.v2_verdict`)."""

    status: str
    confidence: str
    reason: str
    action: str
    angle_status: str


def tree_verdict(verdict: Verdict) -> TreeVerdict:
    """`verdict.v2_verdict` ('status/confidence/reason/action/angle_status') as fields; '' where the tree gave none."""
    parts = str(verdict.v2_verdict).split('/')
    assert len(parts) == len(TreeVerdict._fields), f'no tree verdict recorded: {verdict.v2_verdict!r}'
    return TreeVerdict(*parts)


def tree_path(verdict: Verdict) -> list:
    """The tree's decision path: the ledger without the policy's own `policy:*` comparisons."""
    return [entry for entry in verdict.decision_path if not entry.name.startswith('policy:')]


def tree_and_final(series, settings, **options) -> tuple:
    """(the W28 tree's MMR analysis, the final one) of one record from a single analysis: `analyse_mmr` is the policy
    layer on `_analyse_mmr_record`; the tree's copy keeps its caps, `confidence_raw`, action and resample step."""
    import copy

    from resonances.resonance.analysis import policy
    from resonances.resonance.analysis.mmr.analyse import _analyse_mmr_record

    defaults = dict(scan_subresonances=True, scan_owner=True, parent='', centre_scale=True, quasi_satellite=None, subfloor=None)
    defaults.update(options)
    tree = _analyse_mmr_record(series, settings, None, *defaults.values())
    return tree, policy.apply_mmr(copy.deepcopy(tree), settings)
