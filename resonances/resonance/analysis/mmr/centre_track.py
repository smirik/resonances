"""The potential well of a two-body angle along the record: its moving centre c(t) and the libration period in it.

A two-body angle librates about the minimum of the averaged disturbing function <D>(sigma) of its planet
(`averaged_potential`, direct + indirect part, no expansion), and that minimum moves when the slow elements of the
body and the planet change (e, i, omega, varpi_P). When the well changes much more slowly than one libration (the
adiabatic condition: Wisdom 1985, pp. 274, 279; Saillenfest et al. 2016, eq. 39) the angle librates about a slowly
moving centre (Gallardo 2006, sect. 3, p. 30: "the libration centers evolve as omega evolves"; Pons & Gallardo 2022,
sect. 2.2.3). 2013 AS27 31E-36+0+5 follows the minimum of Earth's well 56 deg over 20 kyr with an RMS of 1.4 deg; the
Mars resonances 27011 4M-7, 65366 6M-11, 82831 13M-24 follow it 160-330 deg.

Two measurements, both from the same well:

- the track (`WellAlongRecord.__call__`, the `geometry.WellModel` of a run): nodes every max(P_c, span /
  mmr_centre_track_nodes) years, the body's and the planet's elements averaged over P_c about each node (as the
  strength check, `strength.direct_theory`), the minimum reached by steepest descent from the observed full-cycle
  centre of the run's middle cycle at the nearest node, then node by node forward and backward from the previous
  node's centre (`averaged_potential.local_minimum`), on sigma's unwrapped branch. A node without a minimum of positive
  curvature ends the track (no_well), and so do a quadrature that does not converge (quadrature_not_converged, as in
  the strength check) and the body within mmr_strength_min_distance_hill Hill radii of the planet along the average at
  the centre (close_approach: an encounter is outside the averaged problem). Whether the observed centres follow the
  track is the geometry's test (`geometry.well_track_valid`); a descent that slid into another well when the first
  one vanished leaves the swings of the run, and the band test in psi requires the bottom inside every cycle's band
  (`geometry.bands_hold_bottom`);
- the libration period of each full cycle in the full well (`full_well_periods`): the orbit through each of the
  cycle's first two turning points in V = 3 k^2 mu_P n^2 a <D>(sigma) at the cycle's middle time (absolute scale,
  nothing fitted; `averaged_potential.periods_from_turning_points`), moved into that well's frame with the cycle's
  net advance as the well's rate (kinetic energy c_dot^2 / 2 at a TP of sigma), the mean of the two. R0's period law reads it in place
  of the fitted pendulum where it exists (`zone.analyse_separatrix`): a flat-bottomed well is no cosine (2003 UG22
  10E-17+0+7, 20 kyr: T_obs / T_pred 0.77 in the fitted pendulum, 1.03 in the full well). It is first order in the
  planet's mass: 2013 AS27 31:36 runs 2-8 % faster than it, a deficit that grows in proportion to m_E (W17 Q8).

Neither exists for a three-body angle (this implementation has no directly averaged three-body potential: the term is
second order in the masses, Nesvorny & Morbidelli 1998, §3; only the scale mu_i mu_j e^q is used), without
the planet's saved elements (its phase is unknown) or its mass. The period is not used for a 1:1 angle, which has its
own energy integral (`coorbital.potential_check`); the track is (the descent works on any two-body line).

Pure arrays: no Body/Simulation imports.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from resonances.resonance.analysis import geometry as geometry_module
from resonances.resonance.analysis.geometry import MovingWell
from resonances.resonance.analysis.mmr import averaged_potential as averaging
from resonances.resonance.analysis.mmr import strength as strength_module
from resonances.resonance.analysis.mmr.coorbital import GM_SUN
from resonances.resonance.analysis.mmr.models import Context, MMRSeries, ResonanceKind
from resonances.resonance.analysis.settings import AnalysisSettings

NAN = float('nan')
TWO_PI = 2 * np.pi

# why there is no track (geometry.MovingWell.status; WELL_OK when there is one)
WELL_OK = geometry_module.WELL_OK
THREE_BODY = 'three_body'
PLANET_NOT_SAVED = 'planet_not_saved'
NO_PLANET_MASS = 'no_planet_mass'
NO_WELL = 'no_well'
NOT_CONVERGED = 'quadrature_not_converged'
CLOSE_APPROACH = 'close_approach'


def availability(series: MMRSeries, context: Context) -> str:
    """WELL_OK when the well can be computed: a two-body angle (prograde, retrograde or 1:1) whose planet's elements are
    saved and whose mass is known; else the reason."""
    spec = series.resonance
    if spec is None or spec.kind == ResonanceKind.THREE_BODY or len(spec.planet_coefficients) != 1:
        return THREE_BODY
    planet = (series.planets or {}).get(spec.planets[0])
    if planet is None or any(getattr(planet, name, None) is None for name in ('a', 'e', 'inc', 'Omega', 'varpi')):
        return PLANET_NOT_SAVED
    mass = context.planet_mass.get(spec.planets[0], NAN)
    if not (np.isfinite(mass) and mass > 0):
        return NO_PLANET_MASS
    return WELL_OK


class WellAlongRecord:
    """The well of a two-body angle along its record (module docstring): a `geometry.WellModel` (called with a run's
    turning points it returns the track of the run) and the full-well periods of R0. Node results are cached: the
    pieces of one run (`geometry._find_episodes`) and the regimes between jumps reuse them."""

    def __init__(self, series: MMRSeries, context: Context, settings: AnalysisSettings, element_window_yr: float):
        self.series, self.settings = series, settings
        self.reason = availability(series, context)
        if self.reason != WELL_OK:
            return
        spec = series.resonance
        self.spec, self.name = spec, spec.planets[0]
        self.mu = float(context.planet_mass[self.name])
        self.t = np.asarray(series.t_yr, dtype=float)
        step = float(np.median(np.diff(self.t)))
        self.width = float(element_window_yr) if np.isfinite(element_window_yr) and element_window_yr > step else step
        span = float(self.t[-1] - self.t[0])
        self.node_step = max(self.width, span / settings.mmr_centre_track_nodes)
        self.nodes = self.t[0] + self.node_step * np.arange(int(np.floor(span / self.node_step)) + 1)
        self.rule = strength_module.quadrature(settings)
        self._elements: Dict[float, Tuple[averaging.Ellipse, averaging.Ellipse, float]] = {}
        self._minima: Dict[Tuple[int, int], Tuple[float, float]] = {}

    @property
    def available(self) -> bool:
        return self.reason == WELL_OK

    # --- elements and lines -------------------------------------------------------------------------------------

    def elements(self, time_yr: float) -> Tuple[averaging.Ellipse, averaging.Ellipse, float]:
        """(body, planet, phase) averaged over P_c about `time_yr` (`strength.body_ellipse`, `strength.saved_planet`)."""
        key = float(time_yr)
        if key not in self._elements:
            part = strength_module._window(self.t, key, self.width)
            body, body_phase = strength_module.body_ellipse(self.series, part)
            planet, varpi_p, node_p = strength_module.saved_planet(self.series, self.name, part)
            self._elements[key] = (body, planet, body_phase + strength_module.planet_phase(self.spec, varpi_p, node_p))
        return self._elements[key]

    def line(self, time_yr: float, nodes_per_turn: int) -> averaging.LineAverage:
        body, planet, phase = self.elements(time_yr)
        return averaging.LineAverage(body, planet, self.spec.body_coefficient, self.spec.planet_coefficients[0], phase, nodes_per_turn)

    def hill_au(self, planet: averaging.Ellipse) -> float:
        return strength_module.hill_radius(planet.a, self.mu)

    # --- the track ---------------------------------------------------------------------------------------------

    def _minimum(self, node: int, start_rad: float) -> Tuple[float, float, str]:
        """(centre on the branch of `start_rad` [rad], closest approach [Hill radii], status) at a node: the descent of
        `averaged_potential.local_minimum`, L doubled up to mmr_strength_max_nodes_per_turn until the curvature with
        L / 2 agrees (the strength check's convergence: where the orbits nearly meet the quadrature distorts the well);
        centre nan with NO_WELL or NOT_CONVERGED."""
        key = (node, int(round(start_rad * self.rule.nodes_per_turn / TWO_PI)))
        if key not in self._minima:
            time = float(self.nodes[node])
            nodes = int(self.rule.nodes_per_turn)
            while True:
                result = averaging.local_minimum(self.line(time, nodes), start_rad, self.rule)
                if not result.measured or averaging.converged(result, self.rule.tolerance) or 2 * nodes > self.rule.max_nodes_per_turn:
                    break
                nodes *= 2
            planet = self.elements(time)[1]
            status = WELL_OK
            if not result.measured:
                status = NO_WELL
            elif not averaging.converged(result, self.rule.tolerance):
                status = NOT_CONVERGED
            centre = start_rad + float(geometry_module.wrap_pi(result.centre_rad - start_rad)) if status == WELL_OK else NAN
            self._minima[key] = (centre, result.min_distance_au / self.hill_au(planet), status)
        return self._minima[key]

    def __call__(self, times: np.ndarray, values: np.ndarray) -> MovingWell:
        """The track of the run with turning points (`times`, `values`): nodes from the one before its first TP to the
        one after its last, started at the node nearest the cycle closest to the run's middle (module docstring)."""
        if not self.available:
            return MovingWell(self.reason)
        times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
        cycle_times, cycle_mids, _ = geometry_module.cycle_centres(times, values)
        if not len(cycle_mids):
            return MovingWell(NO_WELL)
        first = max(0, int(np.searchsorted(self.nodes, times[0], 'right')) - 1)
        last = min(len(self.nodes) - 1, int(np.searchsorted(self.nodes, times[-1], 'left')))
        middle = int(np.argmin(np.abs(cycle_times - (times[0] + times[-1]) / 2)))
        start = int(np.clip(np.argmin(np.abs(self.nodes - cycle_times[middle])), first, last))
        centres: Dict[int, float] = {}
        closest = np.inf
        centre, distance, status = self._minimum(start, float(cycle_mids[middle]))
        if status != WELL_OK:
            return MovingWell(status)
        centres[start], closest = centre, distance
        for direction in (1, -1):
            node = start + direction
            while first <= node <= last:
                centre, distance, status = self._minimum(node, centres[node - direction])
                if status != WELL_OK:
                    return MovingWell(status)
                centres[node], closest = centre, min(closest, distance)
                node += direction
        if closest < self.settings.mmr_strength_min_distance_hill:
            return MovingWell(CLOSE_APPROACH)
        order = sorted(centres)
        return MovingWell(WELL_OK, self.nodes[order], np.array([centres[k] for k in order]))

    # --- the period in the full well ---------------------------------------------------------------------------

    def scale(self, body: averaging.Ellipse) -> float:
        """3 k^2 mu_P n^2 a [rad^2/yr^2 per au^-1]: V = scale x <D> is the potential of sigma_ddot = -dV/dsigma."""
        mean_motion_sq = GM_SUN / body.a**3
        return float(3 * self.spec.body_coefficient**2 * self.mu * mean_motion_sq * body.a)

    def periods(self, time_yr: float, starts_rad: Sequence[float], kinetic: Optional[Sequence[float]] = None) -> np.ndarray:
        """The libration periods [yr] of the orbits through `starts_rad` (the TPs of one swing, moved to the frame of
        the well at `time_yr`) in the full well at `time_yr`, at rest there or with the kinetic energies `kinetic`
        [rad^2/yr^2] (`full_well_periods`); nan for all where the swing does not librate in the well that holds its middle
        (`averaged_potential.swing_in_well`, as the strength check: an observed libration beside the theory's well is no
        orbit of it), where one is not measured (a node on the planet, no convergence), and when the swings come
        within mmr_strength_min_distance_hill Hill radii of the planet."""
        body, planet, _ = self.elements(time_yr)
        result = averaging.periods_from_turning_points(
            lambda nodes: self.line(time_yr, nodes), starts_rad, self.scale(body), self.rule, kinetic
        )
        blank = np.full(len(starts_rad), NAN)
        if not result.in_well or result.reason not in (averaging.REASON_NONE, averaging.REASON_NO_CENTRE):
            return blank
        if result.closest_au / self.hill_au(planet) < self.settings.mmr_strength_min_distance_hill:
            return blank
        return result.periods


def turning_point_energy(well: WellAlongRecord, time_yr: float, value_rad: float, middle_rad: float) -> float:
    """h of one turning point in the averaged well at its own time (`averaged_potential.energy_in_well`, the well whose
    basin holds `middle_rad`, the middle of its swing): <D> on the coarse sigma grid with L nodes per turn, L doubled up
    to mmr_strength_max_nodes_per_turn until h with L and with L / 2 agree within mmr_strength_quadrature_tol (in units
    of the well's depth). nan when it does not converge or there is no well."""
    rule = well.rule
    nodes = int(rule.nodes_per_turn)
    while True:
        line = well.line(time_yr, nodes)
        coarse = np.unique(np.round(np.arange(rule.sigma_grid) * nodes / rule.sigma_grid).astype(np.int64))
        full, half, _ = line.evaluate(coarse)
        sigma = line.sigma(coarse)
        h_full = averaging.energy_in_well(sigma, full, value_rad, middle_rad)
        h_half = averaging.energy_in_well(sigma, half, value_rad, middle_rad)
        if not (np.isfinite(h_full) and np.isfinite(h_half)):
            return NAN
        if abs(h_full - h_half) <= rule.tolerance:
            return h_full
        if 2 * nodes > rule.max_nodes_per_turn:
            return NAN
        nodes *= 2


def turning_point_energies(
    well: Optional[WellAlongRecord], tp_times: np.ndarray, tp_values: np.ndarray, settings: AnalysisSettings, middles_rad: np.ndarray
) -> np.ndarray:
    """h of every turning point in the averaged well at its own time (`turning_point_energy`, in the well that holds
    `middles_rad`, the middle of each one's swing), measured at no more than mmr_separatrix_avgpot_tp_max of them chosen
    evenly in order and interpolated linearly in time between two neighbouring evaluated points that are both finite
    (the well changes over many cycles: 2004 TK14 31M-34 deepens 100 times over 20 kyr). A point whose evaluation
    failed stays nan, and so do the points between it and its evaluated neighbours: nothing is bridged across a failed
    profile (a close approach, no well). All nan without a well."""
    times, values = np.asarray(tp_times, dtype=float), np.asarray(tp_values, dtype=float)
    middles = np.asarray(middles_rad, dtype=float)
    out = np.full(len(times), NAN)
    if well is None or not well.available or not len(times):
        return out
    chosen = np.unique(np.round(np.linspace(0, len(times) - 1, min(len(times), settings.mmr_separatrix_avgpot_tp_max))).astype(int))
    measured = []
    for i in chosen:
        measured.append(turning_point_energy(well, float(times[i]), float(values[i]), float(middles[i])))
    measured = np.asarray(measured, dtype=float)
    out[chosen] = measured
    for (i, j), (h_i, h_j) in zip(zip(chosen[:-1], chosen[1:]), zip(measured[:-1], measured[1:])):
        if j > i + 1 and np.isfinite(h_i) and np.isfinite(h_j):
            out[i + 1 : j] = np.interp(times[i + 1 : j], [times[i], times[j]], [h_i, h_j])
    return out


def full_well_periods(
    well: Optional[WellAlongRecord],
    bounds: Sequence[Tuple[float, float, bool]],
    tp_times: np.ndarray,
    tp_values: np.ndarray,
) -> List[float]:
    """The predicted period [yr] of each cycle of `bounds` (start, end, rotation; the cycles of `zone.cycle_bounds`, a
    libration cycle from TP_i to TP_i+2) in the full well at its middle time t_m: the mean of the periods of the orbits
    through TP_i and TP_i+1 (the same orbit: 2003 UG22 10E-17, laws 0.959 and 0.960 from the upper and the lower one).
    The well moves: sigma = c(t) + psi, and at a TP of sigma psi_dot = -c_dot, not 0. Over a closed cycle psi returns,
    so the cycle's net advance (sigma(TP_i+2) - sigma(TP_i)) / (t_i+2 - t_i) is c_dot, measured on sigma itself (no
    track is needed, and it is 0 in a fixed well). Each TP is moved to the well at t_m, sigma - c_dot (t - t_m), and
    starts with the kinetic energy c_dot^2 / 2: in a well translating at a steady rate this is the exact orbit of psi
    (Codex W18: a pendulum of 3.1 rad translating by 0.08 rad per period gave a law of 0.77 started at rest). A swing
    whose far side passes a lower top of <D> is predicted as the orbit of its own energy (over that top, or in another
    well), whose period then disagrees; an energy above the top behind the start gives no period. nan for a turn (no
    turning point gives its energy), for a cycle whose TPs are not found, for a swing that does not librate in the
    theory's well (`WellAlongRecord.periods`), where a period is not measured, and for every cycle
    without a well (`well` None or not available)."""
    out = [NAN] * len(bounds)
    if well is None or not well.available:
        return out
    times, values = np.asarray(tp_times, dtype=float), np.asarray(tp_values, dtype=float)
    for k, (start, end, rotation) in enumerate(bounds):
        i = int(np.searchsorted(times, start))
        if rotation or i + 1 >= len(times) or times[i] != start:
            continue
        if i + 2 >= len(times):
            continue
        middle = (start + end) / 2
        rate = float((values[i + 2] - values[i]) / (times[i + 2] - times[i]))
        starts = [float(values[j] - rate * (times[j] - middle)) for j in (i, i + 1)]
        periods = well.periods(middle, starts, [rate**2 / 2] * 2)
        if np.all(np.isfinite(periods)):
            out[k] = float(np.mean(periods))
    return out


def build(series: MMRSeries, context: Context, settings: AnalysisSettings, element_window_yr: float) -> Optional[WellAlongRecord]:
    """The well of `series` (`WellAlongRecord`), None for a three-body angle or without the planet's elements or mass."""
    well = WellAlongRecord(series, context, settings, element_window_yr)
    return well if well.available else None
