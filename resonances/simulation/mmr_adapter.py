"""The bridge between a `Body` of the simulation and the pure MMR analysis (`resonance/analysis/mmr`).

`series_from_body` builds the same `MMRSeries` the benchmark reader (`analysis.mmr.io.load_series`) builds from a
saved example: time-ordered rows, the angle wrapped mod 2 pi with its `np.unwrap` copy for the flag and the figure,
every saved planet, and the angle's own planets only when all of them are saved; the planets without a saved lambda get
catalogue longitudes from the run's solar-system cache at the epoch (`_catalogue`). `store_analysis` writes what the
analysis measured back onto the body (in the body's own time order), `derived_rows` lists the accepted subresonance
rows it found. Kept here, not in the pure package, so that the analysis never imports Body or Simulation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from resonances.data.util import mjd_from_datetime
from resonances.logger import logger
from resonances.resonance.analysis.mmr.context import CATALOGUE_PLANETS, catalogue_longitudes
from resonances.resonance.analysis.mmr.io import solar_system_longitudes
from resonances.resonance.analysis.mmr.models import (
    PLANET_NAMES_ORDER,
    SAME_GRID_TOLERANCE,
    MMRAnalysis,
    MMRSeries,
    PlanetSeries,
    Regime,
    ResonanceSpec,
)

TWO_PI = 2.0 * np.pi
MJD_TO_JD = 2400000.5


def time_order(body):
    """What puts the body's rows in increasing time: a slice for a forward run and for a backward one (strictly
    decreasing times), so the series are views and not copies (memory); else the stable sort's indices."""
    steps = np.diff(np.asarray(body.times, dtype=float))
    if np.all(steps >= 0):
        return slice(None)
    if np.all(steps < 0):
        return slice(None, None, -1)
    return np.argsort(np.asarray(body.times, dtype=float), kind='stable')


def in_time_order(values, order) -> np.ndarray:
    """`values` (float) in time order. A view of the run's array is read-only: the analysis never writes into its
    input, and a write would change the body's or the planet's own series."""
    values = np.asarray(values, dtype=float)[order]
    if isinstance(order, slice):
        values = values.view()
        values.flags.writeable = False
    return values


def _on_body_grid(name: str, data: Mapping[str, Any], order, t_yr: np.ndarray) -> bool:
    """The planet's own times (years, written by the integration and read back by restore) are the body's, row by
    row, within SAME_GRID_TOLERANCE of a step. A planet file of another run, shifted or in another row order holds
    the same number of rows but not the same instants: it is left out (with a warning), never realigned."""
    times = data.get('times')
    if times is None or len(t_yr) < 2:
        return True
    times = in_time_order(times, order)
    tolerance = SAME_GRID_TOLERANCE * float(np.median(np.diff(t_yr)))
    if np.all(np.abs(times - t_yr) <= tolerance):
        return True
    logger.warning(
        f'planet {name}: its saved times are not the body\'s (largest difference {np.max(np.abs(times - t_yr)):.6g} yr); left out'
    )
    return False


def _planets(planets_data: Mapping[str, Mapping[str, Any]], order, t_yr: np.ndarray) -> Dict[str, PlanetSeries]:
    """The saved planets on the body's grid (`IntegrationEngine.planets_data`: times, a, e, inc, Omega, omega, M, l, varpi);
    a planet whose saved times differ from the body's is left out (`_on_body_grid`)."""
    n = len(t_yr)
    planets = {}
    for name in PLANET_NAMES_ORDER:
        data = planets_data.get(name) if planets_data else None
        if not data or any(len(np.asarray(data.get(column, ()))) != n for column in ('a', 'e', 'inc', 'Omega', 'varpi')):
            continue
        if not _on_body_grid(name, data, order, t_yr):
            continue
        longitude = data.get('l', data.get('mean_longitude'))
        planets[name] = PlanetSeries(
            name=name,
            t_yr=t_yr,
            a=in_time_order(data['a'], order),
            e=in_time_order(data['e'], order),
            inc=in_time_order(data['inc'], order),
            Omega=in_time_order(data['Omega'], order),
            varpi=in_time_order(data['varpi'], order),
            mean_longitude=None if longitude is None or len(longitude) != n else in_time_order(longitude, order),
        )
    return planets


def series_from_body(body, resonance, simulation) -> MMRSeries:
    """The `MMRSeries` of one MMR of a body: its elements, the raw angle and the planets kept by the run.

    The angle is read from `body.angles[key]` (wrapped) or, before `prepare_angles`, from the raw integration angle
    in `body.angles_unwrapped[key]`; either way `angle_wrapped = angle mod 2 pi` and `angle_unwrapped_numpy =
    np.unwrap(angle_wrapped)`, so a run and its restore build identical series.
    """
    key = resonance.to_s()
    order = time_order(body)
    t_yr = in_time_order(_years(body), order)
    raw = body.angles.get(key)
    if raw is None or len(raw) != len(t_yr):
        raw = body.angles_unwrapped[key]
    angle_wrapped = np.mod(in_time_order(raw, order), TWO_PI)
    spec = ResonanceSpec.from_resonance(resonance)
    planets = _planets(simulation.integration_engine.planets_data, order, t_yr)
    if any(name not in planets for name in spec.planets):  # the angle's planets: all or none (MMRSeries.validate)
        planets = {name: planet for name, planet in planets.items() if name not in spec.planets}
    config = simulation.config

    def element(values) -> Optional[np.ndarray]:
        return None if values is None else in_time_order(values, order)

    series = MMRSeries(
        t_yr=t_yr,
        a=element(body.axis),
        e=element(body.ecc),
        inc=element(body.inc),
        Omega=element(body.Omega),
        omega=element(body.omega),
        varpi=element(body.varpi),
        angle_wrapped=angle_wrapped,
        resonance=spec,
        regime=_regime(config, t_yr, tuple(planets)),
        planets=planets,
        mean_longitude=element(body.longitude),
        mean_anomaly=element(body.M),
        angle_unwrapped_numpy=np.unwrap(angle_wrapped),
        name=str(body.name),
        catalogue_longitudes=_catalogue(simulation, planets, t_yr),
    )
    series.validate()
    return series


def _catalogue(simulation, planets: Mapping[str, PlanetSeries], t_yr: np.ndarray) -> Dict[str, float]:
    """`MMRSeries.catalogue_longitudes` of the catalogue planets without a saved lambda: their heliocentric lambda in the
    solar-system cache of the run's epoch (the initial state of the planets, integration time 0; the same file in a
    restore), carried to t_yr[0] with the table rates; {} when every planet is saved or the cache is not there."""
    missing = [name for name in CATALOGUE_PLANETS if name not in planets or planets[name].mean_longitude is None]
    if not missing or not len(t_yr):
        return {}
    try:
        moon = bool(getattr(simulation.config, 'solar_system_moon', False))
        path = simulation.integration_engine._solar_system_filename(moon=moon)
    except Exception:  # noqa: BLE001 - a configuration without a date: no epoch, no catalogue
        return {}
    source = solar_system_longitudes(path)
    if source is None:
        return {}
    t_ref, longitudes = source
    return catalogue_longitudes({name: longitudes[name] for name in missing}, t_ref, float(t_yr[0]))


def _years(body) -> np.ndarray:
    """The body's times in years: those read from its data file when restored (exact), else times / 2 pi."""
    years = getattr(body, 'times_years', None)
    if years is not None and len(years) == len(body.times):
        return np.asarray(years, dtype=float)
    return np.asarray(body.times, dtype=float) / TWO_PI


def _regime(config, t_yr: np.ndarray, planets_saved: Tuple[str, ...]) -> Regime:
    step = float(np.median(np.diff(t_yr))) if len(t_yr) > 1 else float('nan')
    date = getattr(config, 'date', None)
    epoch = mjd_from_datetime(date) + MJD_TO_JD if date is not None else float('nan')
    return Regime(
        step_yr=step,
        span_yr=float(t_yr[-1] - t_yr[0]) if len(t_yr) else 0.0,
        n_rows=len(t_yr),
        integrator=str(getattr(config, 'integrator', '') or ''),
        dt=float(getattr(config, 'dt', float('nan')) or float('nan')),
        direction='backward' if getattr(config, 'tmax', 0) < 0 else 'forward',
        epoch_jd=float(epoch),
        planets_saved=planets_saved,
    )


def in_body_order(values: Optional[np.ndarray], order, n_rows: int) -> Optional[np.ndarray]:
    """Put a time-ordered series back into the body's own row order (inverse of `time_order`): the analysis' array
    itself (forward run) or a reversed view of it (backward run), not a copy (memory). None unless it has n_rows."""
    if values is None or len(values) != n_rows:
        return None
    values = np.asarray(values, dtype=float)
    if isinstance(order, slice):
        return values[order]
    result = np.empty(n_rows, dtype=float)
    result[order] = values
    return result


def store_analysis(body, key: str, analysis: MMRAnalysis, order, angle_wrapped: Optional[np.ndarray] = None) -> None:
    """Write the analysis' series of one angle onto the body (body time order).

    `angles_unwrapped[key]` becomes the predictive unwrap (np.unwrap of the wrapped angle when the analysis stopped
    before it), `angles[key]` the wrapped angle when given, `angles_filtered*[key]` sigma_f at the record's cutoff
    (absent when the analysis stopped at G0: unphysical orbit or failed unwrap).
    """
    n_rows = len(body.times)
    if angle_wrapped is not None:
        body.angles[key] = in_body_order(angle_wrapped, order, n_rows)
    sampling = analysis.sampling  # None when the record was refused before sampling (invalid input)
    unwrapped = in_body_order(sampling.angle_unwrapped, order, n_rows) if sampling is not None else None
    if unwrapped is None and key in body.angles:
        unwrapped = np.unwrap(body.angles[key])
    if unwrapped is not None:
        body.angles_unwrapped[key] = unwrapped
    filtered = in_body_order(analysis.prep.angle_f, order, n_rows) if analysis.prep is not None else None
    if filtered is None:
        body.angles_filtered.pop(key, None)
        body.angles_filtered_unwrapped.pop(key, None)
        return
    body.angles_filtered_unwrapped[key] = filtered
    body.angles_filtered[key] = np.mod(filtered, TWO_PI)


def derived_rows(analysis: MMRAnalysis) -> List[Tuple[str, MMRAnalysis]]:
    """(key, analysis) of the derived rows of a catalogue angle (accepted planar subresonances, depth 1)."""
    return [(derived.resonance.key, derived) for derived in analysis.derived]
