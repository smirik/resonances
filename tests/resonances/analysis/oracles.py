"""Helpers moved out of the package (2026-11-14, the author: code the package does not use, kept only where a test of
live code needs it as an independent reference or as a fixture builder).

- `fixed_frame`: the unmodulated pendulum as an explicit `ModulatedFrame` (m = 1, no motion of the stable point); the
  package passes frame=None for it, the tests check that both read the same.
- `owner_angle_reference`: the W2 per-candidate unwrap of an owner angle at the frame's body mean motion and varpi
  rate, the reference `AngleBasis` (the live path) is checked against.
- `lowest_ceiling`: the level the inputs of `policy.level_inputs` allow, the oracle the tests hold `policy.level` and
  the policy's final level to.
- `record_tables_kept`: how many co-orbital tables the open `coorbital.record_tables` scope holds (the memory probe).
- `elements_from_invariable_plane` (the inverse rotation: orbits built in the invariable plane) and
  `argument_of_pericentre` (omega' of the invariable plane): the fixture builder and the reference of the ZLK frame
  tests (the package rotates the other way only, `invariable_plane.elements_in_invariable_plane`).
- `averaged_profile`: <D>(sigma) on the full grid of `averaged_potential.LineAverage` (the averaged potential the
  strength and owner routes read), the profile the tests compare with brute force and closed forms.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np

from resonances.data.invariable_plane import elements_in_invariable_plane, rotate_elements, rotation_to_invariable
from resonances.resonance.analysis import pendulum
from resonances.resonance.analysis.mmr import coorbital, owner
from resonances.resonance.analysis.mmr.averaged_potential import Ellipse, LineAverage
from resonances.resonance.analysis.mmr.context import mean_motion
from resonances.resonance.analysis.mmr.models import MMRSeries, ResonanceSpec
from resonances.resonance.analysis.mmr.sampling import predictive_unwrap
from resonances.resonance.analysis.verdict import CONFIDENCE_ORDER, Confidence


def fixed_frame(t_yr: np.ndarray) -> pendulum.ModulatedFrame:
    """The unmodulated pendulum on `t_yr` (m = 1, no motion of the stable point)."""
    t_yr = np.asarray(t_yr, dtype=float)
    return pendulum.ModulatedFrame(t_yr, np.ones_like(t_yr), np.zeros_like(t_yr), np.zeros_like(t_yr))


def owner_angle_reference(
    candidate: ResonanceSpec, series: MMRSeries, planet_n: Mapping[str, float], body_n_offset: float, varpi_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """(phi_o unwrapped [rad], per-step model error [rad]) of the candidate unwrapped by its own model rate
    sum k_j n_j + k (n(a) + body_n_offset) + p varpi_rate, as `owner.owner_angle_with_residual` does with the
    record's own varpi rate and no offset."""
    t = np.asarray(series.t_yr, float)
    a = np.asarray(series.a, float)
    body = mean_motion(0.5 * (a[1:] + a[:-1])) + body_n_offset
    lam = sum(k * planet_n[name] for name, k in zip(candidate.planets, candidate.planet_coefficients))
    rate = lam + candidate.body_coefficient * body + candidate.body_varpi_coefficient * varpi_rate
    return predictive_unwrap(t, owner.wrapped_angle(candidate, series), rate)


def lowest_ceiling(inputs: Sequence) -> Confidence:
    """The level the inputs allow: their lowest ceiling (CONFIDENCE_ORDER); none without an input."""
    return max((entry.ceiling for entry in inputs), key=CONFIDENCE_ORDER.index, default=Confidence.NONE)


def record_tables_kept() -> int:
    """How many tables the open `coorbital.record_tables` scope holds (0 outside one)."""
    return len(coorbital._RECORD_TABLES) if coorbital._RECORD_TABLES is not None else 0


def averaged_profile(
    body: Ellipse, planet: Ellipse, body_coefficient: int, planet_coefficient: int, phase_rad: float, nodes_per_turn: int
) -> Tuple[np.ndarray, np.ndarray]:
    """(sigma [rad], <D>(sigma) [au^-1]) on the full grid sigma_m = 2 pi m / L."""
    line = LineAverage(body, planet, body_coefficient, planet_coefficient, phase_rad, nodes_per_turn)
    m = np.arange(line.nodes_per_turn)
    return line.sigma(m), line.evaluate(m)[0]


def elements_from_invariable_plane(inc, Omega, omega) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The inverse of `elements_in_invariable_plane`: ecliptic (inc, Omega, omega) [rad] of orbits given in the frame
    of the invariable plane (the transposed rotation)."""
    return rotate_elements(inc, Omega, omega, rotation_to_invariable().T)


def argument_of_pericentre(inc, Omega, omega) -> np.ndarray:
    """omega' [rad, (-pi, pi]] from the invariable plane of orbits given by ecliptic (inc, Omega, omega)."""
    return elements_in_invariable_plane(inc, Omega, omega)[2]
