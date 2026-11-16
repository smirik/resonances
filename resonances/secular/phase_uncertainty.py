"""How far the planets' mode phases can stray from the linear law over a record of length T, and what that does to a
secular angle built from them (W28 D5, card g09-3).

Under `SECULAR_PLANET_PHASES=linear` the angle uses g_j*t + phase_j for every mode (`const.py`). The true longitude of
mode j is g_j*t + phi_j(t) (`planet_modes.mode_phase`), and phi_j is not constant: the inner modes wander by tens of
degrees over 10 Myr. Over a record of length T the angle built from the linear law then carries an error
sum_j k_j (phi_j(t) - const), whose swing is at most

    dphi(T) = sum_j |k_j| range_j(T),   range_j(T) = max over windows of length T of (max - min of phi_j there).

A libration whose whole swing (2 x the largest half-amplitude) is within dphi(T) is not resolved from that error: its
turning points may be those of the mode phases (5026 g+g2-g4-g6: half-amplitude 8-11 deg, a frequency below 1/T). The
decision is `analysis.secular.decide.phase_check`; a mode whose longitude is measured in the integration
(`SECULAR_PLANET_PHASES=measured`) contributes nothing.

range_j(T) is measured on the package's own model (experiments/2026-10-planetary-frequencies/phase_uncertainty.py):
phi_j from 5 Myr sliding fits of planet j's own vector, the maximum over sliding windows of every run, written to
`resonances/data/mode_phase_uncertainty.json` with its provenance. The fit window (5 Myr) smooths phi_j: below it,
range_j(T) is the drift of the phase (the error of g_j), not its faster wobble.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from resonances.secular.planet_modes import DEFAULT_WINDOW, mode_phase

TABLE_PATH = Path(__file__).resolve().parents[1] / 'data' / 'mode_phase_uncertainty.json'
MODE_LABELS = tuple(f'g{j}' for j in range(1, 9)) + tuple(f's{j}' for j in range(1, 9))
# s5 = 0 is the invariable plane: never in an angle, no phase (its range is 0 by definition)
NO_PHASE_MODES = ('s5',)


def window_ranges(
    times_years, phase, spans_years: Sequence[float], step_years: Optional[float] = None, latest_start_years: Optional[float] = None
) -> np.ndarray:
    """range(T) for each T of `spans_years` [same unit as `phase`]: the largest max - min of `phase` over the windows
    [t, t + T] inside the record whose start t is at most `latest_start_years` after the record's start (0: the window
    starting with the record only; None: anywhere), the start moving by `step_years` (default: the sample step). A
    span longer than the record gives nan. `phase` must be unwrapped."""
    t = np.asarray(times_years, dtype=float)
    x = np.asarray(phase, dtype=float)
    dt = float(np.median(np.diff(t)))
    stride = max(1, int(round((step_years or dt) / dt)))
    last = len(t) if latest_start_years is None else int(round(latest_start_years / dt)) + 1
    result = np.full(len(spans_years), np.nan)
    for n, span in enumerate(spans_years):
        width = int(round(span / dt))
        if width < 1 or width > len(t) - 1:
            continue
        starts = np.arange(0, min(len(t) - width, last), stride)
        result[n] = max(float(np.ptp(x[i : i + width + 1])) for i in starts)
    return result


def mode_phase_ranges(
    times_years,
    vectors: Mapping[str, np.ndarray],
    spans_years: Sequence[float],
    window: float = DEFAULT_WINDOW,
    step_years: Optional[float] = None,
    latest_start_years: Optional[float] = None,
    frequencies=None,
) -> Dict[str, np.ndarray]:
    """{mode: range_j(T) [rad] for each T}: phi_j(t) of mode j fitted in planet j's own vector (`vectors[label]`:
    e*exp(i*varpi) for g_j, sin(i/2)*exp(i*Omega) for s_j; `planet_modes.mode_phase`), then `window_ranges` (within
    half a fit window of either end phi_j goes on along its fitted slope)."""
    t = np.asarray(times_years, dtype=float)
    result = {}
    for label, z in vectors.items():
        if label in NO_PHASE_MODES:
            continue
        phi = mode_phase(t, z, label, window, frequencies)
        result[label] = window_ranges(t, phi, spans_years, step_years, latest_start_years)
    return result


@dataclass(frozen=True)
class PhaseUncertaintyTable:
    """range_j(T) of every mode on a grid of record lengths (`resonances/data/mode_phase_uncertainty.json`)."""

    spans_yr: Tuple[float, ...]
    ranges_deg: Mapping[str, Tuple[float, ...]]
    provenance: Mapping[str, object] = field(default_factory=dict)

    def range_rad(self, label: str, span_yr: float) -> float:
        """range_j(T) [rad]: linear between the grid's lengths; below the shortest, the shortest's value (an upper
        bound: the range only grows with T); above the longest T_max, ceil(T / T_max) x range_j(T_max) (the range over
        k consecutive windows is at most the sum of theirs). 0 for s5; nan for a mode the table lacks."""
        if label in NO_PHASE_MODES:
            return 0.0
        values = self.ranges_deg.get(label)
        if values is None or not np.isfinite(span_yr) or span_yr <= 0:
            return float('nan')
        spans = np.asarray(self.spans_yr, dtype=float)
        ranges = np.asarray(values, dtype=float)
        if span_yr > spans[-1]:
            degrees = math.ceil(span_yr / spans[-1]) * ranges[-1]
        else:
            degrees = float(np.interp(span_yr, spans, ranges))
        return math.radians(degrees)

    def delta_phi_rad(self, coefficients: Mapping[str, int], span_yr: float, measured: Iterable[str] = ()) -> float:
        """dphi(T) = sum_j |k_j| range_j(T) [rad] over the modes of the angle (`coefficients`: {'g4': -1, ...}) whose
        longitude follows the linear law (not in `measured`); nan when a mode is missing from the table."""
        skip = set(measured)
        total = 0.0
        for label, k in coefficients.items():
            if label in skip or k == 0:
                continue
            total += abs(k) * self.range_rad(label, span_yr)
        return float(total)

    def to_json(self) -> Dict[str, object]:
        return {
            'spans_yr': list(self.spans_yr),
            'ranges_deg': {k: list(v) for k, v in self.ranges_deg.items()},
            'provenance': dict(self.provenance),
        }


def table_from_json(payload: Mapping[str, object]) -> PhaseUncertaintyTable:
    spans = tuple(float(x) for x in payload['spans_yr'])
    ranges = {str(k): tuple(float(x) for x in v) for k, v in dict(payload['ranges_deg']).items()}
    for label, values in ranges.items():
        if len(values) != len(spans):
            raise ValueError(f'mode {label}: {len(values)} ranges for {len(spans)} spans')
    return PhaseUncertaintyTable(spans, ranges, dict(payload.get('provenance', {})))


@lru_cache(maxsize=4)
def load_table(path: Optional[str] = None) -> PhaseUncertaintyTable:
    """The packaged table (or the one at `path`)."""
    with open(path or TABLE_PATH) as handle:
        return table_from_json(json.load(handle))
