"""Clones (virtual asteroids) in a run: `Simulation.add_ensemble`, `add_body(..., clones=N)` and their aggregation
(docs/clones.md).

- Epoch. The clones are drawn at the covariance's epoch. When it is not the run's date, the nominal and every clone are
  moved to the date: within EPOCH_TOLERANCE_DAYS (8.6 s) on their two-body ellipses, beyond it integrated in the
  package's model (`propagate.propagate_elements`: IAS15, the run's planets), the nominal alone and the clones together,
  so the nominal's start never depends on the number of clones. The covariance itself is never propagated linearly.
  clones.csv records the span (`propagation_days`) and the way (`propagation`).
- Integration. The nominal is an ordinary body of the run (its own simulation, as without clones: its verdict does not
  change). The clones are test particles in batches of CLONES_BATCH per REBOUND simulation (SABA's safe mode makes a
  particle's record depend at 1e-10 AU on how many share its simulation), run by the batch workers after the ordinary
  bodies; each clone is a body named `<parent>~c001`, ..., with its rows in summary.csv like any body.
- Output. clones.csv (one row per parent and resonance: the six status shares with Wilson 95 % intervals, the modal
  status, the nominal's verdict and agreement), the draw manifest `ensemble-<parent>.csv/.json` in the data folder, and
  one figure per parent and resonance (`plotting.clone_plots`).
"""

import copy
import datetime
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tqdm

from resonances.clones import Ensemble, orbit_solution
from resonances.clones.elements import KEPLERIAN_KEYS, drift_mean_anomaly, mean_longitude
from resonances.clones.ensemble import clone_name
from resonances.clones.stats import ensemble_row
from resonances.config import config as c
from resonances.data.util import mjd_from_datetime
from resonances.logger import logger
from resonances.mpcorb import datetime_from_mjd
from resonances.secular.secular_resonance import SecularResonance
from resonances.data.const import SOLAR_SYSTEM_WITH_SUN

from .body_manager import EPOCH_TOLERANCE_DAYS
from .config import SavePlanets, status_selected
from .parallel_executor import ParallelExecutor
from .propagate import propagate_elements

CLONES_TABLE = 'clones.csv'
# Points of a clone's angle kept for the ensemble figure (a figure, not an analysis: every k-th sample)
FIGURE_POINTS = 2000
# The data-source of the run -> the clone source when CLONES_SOURCE=auto: the same provider as the nominal
AUTO_SOURCE = {'astdys': 'astdys', 'mpc': 'mpc', 'nasa': 'sbdb'}


@dataclass
class EnsembleRun:
    """One ensemble in a run: its clones' names and elements at the run's date (None: a draw that is no bound orbit),
    the resonances they are checked in, the nominal's name (`parent`; a body of the run, or '' without one), and after
    the run the workers' `results` ({clone name: {'rows', 'a0', 'series', 'half_width'}})."""

    ensemble: Ensemble
    parent: str
    resonances: list
    names: List[str]
    elements: List[Optional[dict]]
    nominal_offset_sigma: float = float('nan')
    results: dict = field(default_factory=dict)

    @property
    def keys(self) -> List[str]:
        return [r.to_s() for r in self.resonances]


# ----------------------------------------------------------------------------------------------- elements at the date
def _aligned_epoch(epoch_mjd: float) -> float:
    """The epoch the planets are built at: whole seconds (rebound asks Horizons for '%Y-%m-%d %H:%M:%S')."""
    second = datetime_from_mjd(epoch_mjd).replace(microsecond=0)
    aligned = mjd_from_datetime(second)
    if datetime_from_mjd(aligned).replace(microsecond=0) != second:  # the float lands a microsecond early
        aligned = mjd_from_datetime(second + datetime.timedelta(microseconds=2))
    return aligned


def _propagate(elements: List[dict], epoch_mjd: float, config) -> List[dict]:
    """Elements at `epoch_mjd` carried to config.date: first drifted on their ellipses to the whole second the planets
    are built at (< 1 s), then integrated (`propagate_elements`)."""
    aligned = _aligned_epoch(epoch_mjd)
    shift = aligned - epoch_mjd
    moved = []
    for elem in elements:
        drifted = drift_mean_anomaly(elem, shift) if shift else dict(elem)
        moved.append({**{key: float(drifted[key]) for key in KEPLERIAN_KEYS}, 'epoch': aligned})
    return propagate_elements(moved, config)


def propagation_way(gap_days: float) -> str:
    """How elements `gap_days` from the run's date are moved to it: 'none' (no gap), 'kepler' (within
    EPOCH_TOLERANCE_DAYS: M drifts on the two-body ellipse), 'ias15' (integrated with the planets)."""
    if gap_days == 0:
        return 'none'
    return 'kepler' if abs(gap_days) <= EPOCH_TOLERANCE_DAYS else 'ias15'


def _propagate_max_days(config) -> Optional[float]:
    """MPC_PROPAGATE_MAX_DAYS (the run's for source 'mpc', else the .env's), None when empty: a warning for the clones,
    not a refusal."""
    if hasattr(config, 'mpc_propagate_max_days'):
        return config.mpc_propagate_max_days
    value = c.get('MPC_PROPAGATE_MAX_DAYS') if c.has('MPC_PROPAGATE_MAX_DAYS') else ''
    return None if value in (None, '') else float(value)


def _states_copy(states: tuple) -> tuple:
    """A copy of (nominal, [clones]): the memo and its callers never share element dicts."""
    nominal, clones = states
    return (None if nominal is None else dict(nominal)), [None if elem is None else dict(elem) for elem in clones]


def elements_at_date(ensemble: Ensemble, config) -> tuple:
    """(nominal elements or None, [clone elements or None]) at config.date, memoised on the ensemble per date (find
    computes them for the candidates, add_ensemble uses them); each call returns its own copies.

    Any gap from the covariance's epoch is crossed (`propagation_way`): a gap of a few seconds moves M by several
    sigma_lambda of a good orbit (463: 5 s = 3.8 sigma_lambda), so even a sub-tolerance gap is drifted, never relabelled."""
    date_mjd = mjd_from_datetime(config.date)
    memo_key = (round(date_mjd, 9), bool(getattr(config, 'solar_system_moon', False)))
    if memo_key in ensemble._at_date:
        return _states_copy(ensemble._at_date[memo_key])
    nominal = None if ensemble.nominal is None else {key: float(ensemble.nominal[key]) for key in KEPLERIAN_KEYS}
    clones = ensemble.clone_elements()
    gap = date_mjd - float(ensemble.epoch_mjd)
    way = propagation_way(gap)
    if way != 'ias15':

        def moved(elem):
            drifted = drift_mean_anomaly(elem, gap) if gap else elem
            return {**{key: float(drifted[key]) for key in KEPLERIAN_KEYS}, 'epoch': date_mjd}

        nominal = None if nominal is None else moved(nominal)
        clones = [None if elem is None else moved(elem) for elem in clones]
    else:
        logger.warning(
            f"{ensemble.name}: the clones are drawn at the covariance's epoch MJD {ensemble.epoch_mjd:.6f}, {gap:+.1f} days "
            f"from the run date {config.date}; the nominal and {ensemble.n_valid} clones are integrated to it (IAS15, the "
            "run's planets)"
        )
        max_days = _propagate_max_days(config)
        if max_days is not None and abs(gap) > max_days:
            logger.warning(
                f'{ensemble.name}: the clones are integrated over {abs(gap):.1f} days, beyond MPC_PROPAGATE_MAX_DAYS = '
                f"{max_days:g} (the package's propagation was checked against a provider's over 138 days only)"
            )
        if nominal is not None:
            nominal = _propagate([nominal], ensemble.epoch_mjd, config)[0]
        valid = [i for i, elem in enumerate(clones) if elem is not None]
        carried = _propagate([clones[i] for i in valid], ensemble.epoch_mjd, config) if valid else []
        clones = list(clones)
        for i, elem in zip(valid, carried):
            clones[i] = elem
    ensemble._at_date[memo_key] = _states_copy((nominal, clones))
    return nominal, clones


def offset_sigma(reference: dict, clones: List[Optional[dict]]) -> float:
    """How far a nominal (`reference`, elements at the date) lies from the cloud: max(|da| / sd(a), |dlambda| / sd(lambda))
    over the valid clones (nan with fewer than 3). Lesson of the co-orbital clones: a cloud carried in the package's
    model can miss the provider's own state by several of its widths."""
    valid = [elem for elem in clones if elem is not None]
    if len(valid) < 3:
        return float('nan')
    a = np.array([elem['a'] for elem in valid])
    lam = np.unwrap(np.array([float(mean_longitude(elem)) for elem in valid]))
    centre = np.mean(lam)
    dlam = (float(mean_longitude(reference)) - centre + np.pi) % (2 * np.pi) - np.pi
    lam = (lam - centre + np.pi) % (2 * np.pi) - np.pi
    ratios = []
    for delta, values in ((reference['a'] - a.mean(), a), (dlam, lam)):
        spread = np.std(values, ddof=1)
        ratios.append(abs(delta) / spread if spread > 0 else float('inf') if delta else 0.0)
    return float(max(ratios))


def cloud_candidates(elements: List[Optional[dict]], **find_kwargs) -> list:
    """The union of the MMR candidates (`candidate_mmrs`) of every member, in first-seen order (author 2026-10-27: in a
    find, a clone may sit in a resonance the nominal never visits)."""
    from resonances.mmr.mmr_finder import candidate_mmrs

    found, seen = [], set()
    for elem in elements:
        if elem is None:
            continue
        for mmr in candidate_mmrs(elem, **find_kwargs):
            if mmr.to_s() not in seen:
                seen.add(mmr.to_s())
                found.append(mmr)
    return found


# ----------------------------------------------------------------------------------------------- adding to a run
def _resonance_list(resonances) -> list:
    import resonances as package

    items = resonances if isinstance(resonances, (list, tuple)) else [resonances]
    out, seen = [], set()
    for item in items:
        resonance = package.create_resonance(item)
        if resonance.to_s() not in seen:
            seen.add(resonance.to_s())
            out.append(resonance)
    return out


def clone_names_in_run(simulation, ensemble: Ensemble, parent: str) -> list:
    """The clones' names under `parent`, refused (ValueError) when `parent` has an ensemble in the run already or a
    name is a body's. Called before anything is added to the run."""
    if any(run.parent == parent for run in simulation.ensembles):
        raise ValueError(f'{parent} has an ensemble in this run already')
    width = max(3, len(str(len(ensemble.clones))))
    names = [clone_name(parent, int(index), width) for index in ensemble.clones['clone']]
    taken = {str(body.name) for body in simulation.bodies}
    clash = [name for name in names if name in taken]
    if clash:
        raise ValueError(f'clone names {clash[:3]} are names of bodies of the run')
    return names


def attach(simulation, ensemble: Ensemble, parent: str, resonances, clone_elements: List[Optional[dict]], offset: float = 0.0):
    """Register an ensemble's clones (elements at the date) in `simulation` under `parent`."""
    names = clone_names_in_run(simulation, ensemble, parent)
    run = EnsembleRun(ensemble, parent, _resonance_list(resonances), names, list(clone_elements), offset)
    simulation.ensembles.append(run)
    if np.isfinite(offset) and offset > 3:
        source = ensemble.solution.source if ensemble.solution is not None else ensemble.method
        logger.warning(
            f'{parent}: the body ({simulation.config.source}) lies {offset:.1f} cloud widths from its clones ({source}) at '
            'the run date: the two orbit solutions (or their carriage in the package model) differ'
        )
    nongrav = ensemble.nongrav
    if nongrav:
        logger.warning(f'{parent}: non-gravitational parameters {nongrav} of the orbit fit are not modelled (flag in clones.csv)')
    logger.info(f'{parent}: {len(names)} clones ({ensemble.n_valid} bound orbits) in {", ".join(run.keys)}')
    return run


def add_ensemble(simulation, ensemble: Ensemble, resonances=None, name: Optional[str] = None, nominal: bool = True, **find_kwargs):
    """`Simulation.add_ensemble`: see there."""
    name = str(ensemble.name if name is None else name)
    nominal_elements, clone_elements = elements_at_date(ensemble, simulation.config)
    if resonances is None:
        # As in find: the nominal keeps its own candidates (its rows and its auto step are those of a run without
        # clones), the clones get them and the cloud's extra ones
        own = cloud_candidates([nominal_elements], **find_kwargs) if nominal_elements is not None else []
        known = {r.to_s() for r in own}
        extra = [r for r in cloud_candidates(clone_elements, **find_kwargs) if r.to_s() not in known]
        nominal_resonances, resonances = own, own + extra
        if not resonances:
            logger.warning(f'{name}: no MMR candidate for the nominal or any clone; ensemble not added')
            return None
        if extra:
            logger.info(f'{name}: {len(extra)} MMR candidates of its clones only: {[r.to_s() for r in extra]}')
    elif find_kwargs:
        raise TypeError(f'{", ".join(find_kwargs)}: only with resonances=None (the candidates of the cloud)')
    else:
        nominal_resonances = resonances
    clone_names_in_run(simulation, ensemble, name)  # a refusal leaves the run as it was: checked before the nominal is added
    if nominal and nominal_elements is not None:
        if nominal_resonances:
            simulation.body_manager.add_body(nominal_elements, nominal_resonances, name)
        else:
            logger.warning(f'{name}: no MMR candidate for the nominal; only its clones run')
    return attach(simulation, ensemble, name, resonances, clone_elements, 0.0 if nominal_elements is not None else float('nan'))


def clone_source(config) -> str:
    """CLONES_SOURCE, or for 'auto' the provider of the run's elements (astdys -> astdys, mpc -> mpc, nasa -> sbdb)."""
    source = str(getattr(config, 'clones_source', 'auto') or 'auto')
    return AUTO_SOURCE[config.source] if source == 'auto' else source


def ensemble_of(simulation, identifier, n: int, name: str) -> Ensemble:
    """The ensemble of an asteroid by the run's clone settings (source, epoch, seed)."""
    config = simulation.config
    solution = orbit_solution(identifier, source=clone_source(config), epoch=getattr(config, 'clones_epoch', None) or None)
    return solution.sample(n=n, seed=config.clones_seed, name=name)


def add_clones_of_body(simulation, identifier, name: str, n: int, resonances, reference: dict):
    """`add_body(..., clones=n)`: the body was added from the run's source (`reference`, its elements at the date); its
    clones come from the orbit solution of CLONES_SOURCE, carried to the date. The distance of the body from the cloud is
    kept (`nominal_offset_sigma`), a warning above 3."""
    ensemble = ensemble_of(simulation, identifier, n, name)
    _, clone_elements = elements_at_date(ensemble, simulation.config)
    return attach(simulation, ensemble, name, resonances, clone_elements, offset_sigma(reference, clone_elements))


# ----------------------------------------------------------------------------------------------- the run
def _planets_for_clones(simulation) -> list:
    """The planets a clone batch keeps: the run's (`planets_needed`) and those of the ensembles' resonances."""
    config = simulation.config
    every = list(simulation.integration_engine.planets_without_sun)
    if config.save_planets == SavePlanets.ALL:
        return every
    if config.save_planets == SavePlanets.NONE:
        return []
    names = set(simulation.planets_needed())
    has_mmr = False
    for run in simulation.ensembles:
        for resonance in run.resonances:
            has_mmr = has_mmr or resonance.type == 'mmr'
            names.update(getattr(resonance, 'planets_names', None) or [])
            if isinstance(resonance, SecularResonance):
                names.update(SOLAR_SYSTEM_WITH_SUN[index] for index in resonance.index_of_planets)
    if has_mmr and config.mmr_owner_search != 'off':
        names.update(config.mmr_owner_planets)
    return [planet for planet in every if planet in names]


def clone_batches(simulation) -> list:
    """(index, body dicts) of the clone batches: CLONES_BATCH clones per simulation, ensembles one after another."""
    size = max(1, int(simulation.config.clones_batch))
    bodies = [
        {'elem_or_num': elem, 'resonances': [copy.deepcopy(r) for r in run.resonances], 'name': name}
        for run in simulation.ensembles
        for name, elem in zip(run.names, run.elements)
        if elem is not None
    ]
    return [(i, bodies[start : start + size]) for i, start in enumerate(range(0, len(bodies), size))]


def run_ensembles(simulation, progress: bool = False):
    """Integrate and classify every clone in batch workers, then write clones.csv, the manifests and the figures."""
    config = simulation.config
    batches = clone_batches(simulation)
    kwargs = simulation.batch_manager._prepare_simulation_kwargs(simulation)
    kwargs.update(clones=0, _planets_needed=_planets_for_clones(simulation), _clone_batch=True)
    if not config.clones_plot_each:
        kwargs['plot'] = None
    results = {}
    bar = tqdm.tqdm(total=len(batches), desc='Clones', unit='batch') if progress and batches else None

    def drop_old_rows(pending):
        if config.save_summary:
            simulation.data_manager.drop_rows([body['name'] for _, batch in pending for body in batch])

    def collect(_index, result):
        results.update(result.get('clones') or {})
        if bar is not None:
            bar.update(1)

    try:
        if batches:
            logger.info(f'Clones: {sum(len(b) for _, b in batches)} in {len(batches)} batches on {config.n_cores} cores')
            ParallelExecutor(config.n_cores).execute_batches(batches, kwargs, collect, before_run=drop_old_rows)
    finally:
        if bar is not None:
            bar.close()
    for run in simulation.ensembles:
        run.results = {name: results[name] for name in run.names if name in results}
    write_outputs(simulation)


def worker_results(simulation) -> dict:
    """What a clone batch returns to the parent, per clone: the verdict of each resonance (status, confidence,
    angle_status, reason, libration_origin), the initial a, the half-width of the accepted MMR episodes (AU, median)
    and the angle thinned to FIGURE_POINTS samples (years, degrees in [0, 360))."""
    out = {}
    for body in simulation.bodies:
        rows, series, half_width = {}, {}, {}
        for resonance in body.resonances():
            key = resonance.to_s()
            verdict = body.librations.get(key)
            if verdict is None:
                continue
            rows[key] = {
                'status': str(verdict.status),
                'confidence': str(verdict.confidence),
                'angle_status': str(verdict.angle_status),
                'reason': str(verdict.reason),
                'libration_origin': str(verdict.libration_origin),
            }
            if key in body.derived_resonances:
                continue
            angle = body.angles.get(key)
            if angle is not None and body.times is not None and len(angle) == len(body.times):
                step = max(1, math.ceil(len(angle) / FIGURE_POINTS))
                t = np.asarray(body.times[::step], dtype=float) / (2 * np.pi)
                series[key] = (t.tolist(), np.mod(np.degrees(np.asarray(angle[::step], dtype=float)), 360.0).tolist())
            analysis = body.mmr_analyses.get(key)
            if analysis is not None:
                widths = [r.get('half_width_au') for r in analysis.episode_rows() if r.get('accepted')]
                widths = [float(w) for w in widths if w not in (None, '') and np.isfinite(float(w))]
                if widths:
                    half_width[key] = float(np.median(widths))
        out[str(body.name)] = {'a0': float(body.initial_data['a']), 'rows': rows, 'series': series, 'half_width': half_width}
    return out


# ----------------------------------------------------------------------------------------------- outputs
def _nominal_body(simulation, run: EnsembleRun):
    return next((body for body in simulation.bodies if str(body.name) == run.parent), None)


def member_statuses(run: EnsembleRun, key: str) -> list:
    """The status of every clone in a row: 'invalid' for a draw that is no bound orbit, None for a clone without a
    verdict there."""
    statuses = []
    for name, elem in zip(run.names, run.elements):
        if elem is None:
            statuses.append('invalid')
            continue
        row = (run.results.get(name) or {}).get('rows', {}).get(key)
        statuses.append(row['status'] if row else None)
    return statuses


def clone_rows(simulation) -> list:
    """The clones.csv rows of the run's ensembles (one per parent and resonance)."""
    rows = []
    date_mjd = mjd_from_datetime(simulation.config.date)
    for run in simulation.ensembles:
        gap = date_mjd - float(run.ensemble.epoch_mjd)
        body = _nominal_body(simulation, run)
        solution = run.ensemble.solution
        for resonance in run.resonances:
            key = resonance.to_s()
            nominal_status = body.statuses.get(key) if body is not None else None
            verdict = body.librations.get(key) if body is not None else None
            row = {
                'name': run.parent,
                'resonance': key,
                'resonance_type': resonance.type,
                'n_clones': len(run.names),
                'n_draw_invalid': sum(1 for elem in run.elements if elem is None),
                'nominal_status': nominal_status or '',
                'nominal_confidence': str(verdict.confidence) if verdict is not None else '',
            }
            row.update(ensemble_row(member_statuses(run, key), nominal_status))
            row.update(
                {
                    'nominal_offset_sigma': run.nominal_offset_sigma,
                    'source': solution.source if solution is not None else run.ensemble.method,
                    'product': (solution.provenance.get('product', '') if solution is not None else ''),
                    'epoch_mjd': float(run.ensemble.epoch_mjd),
                    'run_date_mjd': date_mjd,
                    'propagated': propagation_way(gap) != 'none',
                    'propagation': propagation_way(gap),
                    'propagation_days': gap,
                    'seed': run.ensemble.seed if run.ensemble.seed is not None else '',
                    'nongrav': json.dumps(run.ensemble.nongrav, sort_keys=True) if run.ensemble.nongrav else '',
                }
            )
            rows.append(row)
    return rows


def write_clones_table(simulation, rows: list) -> Optional[str]:
    """clones.csv at the top of save_path: these parents' rows replace any they had."""
    if not rows:
        return None
    from .data_manager import DataManager

    path = Path(simulation.config.save_path) / CLONES_TABLE
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    parents = set(frame['name'].astype(str))
    with DataManager._locked(str(path)):
        if path.exists() and path.stat().st_size > 0:
            existing = pd.read_csv(path, dtype=str, keep_default_na=False)
            existing = existing[~existing['name'].isin(parents)]
            frame = pd.concat([existing, frame.astype(object)], ignore_index=True)
        frame.to_csv(f'{path}.tmp', index=False)
        Path(f'{path}.tmp').replace(path)
    return str(path)


def write_manifest(simulation, run: EnsembleRun) -> tuple:
    """`ensemble-<parent>.csv/.json` in the data folder: every draw (clone index, name, elements at the covariance's
    epoch, valid, d^2, the draw in the fit's coordinates) and its elements at the run's date (`<element>_start`)."""
    frame = run.ensemble.clones.copy()
    if 'name' in frame.columns:  # an ensemble loaded from a run's manifest
        frame['name'] = run.names
    else:
        frame.insert(1, 'name', run.names)
    for key in KEPLERIAN_KEYS:
        frame[f'{key}_start'] = [elem[key] if elem is not None else np.nan for elem in run.elements]
    manifest = replace(run.ensemble, clones=frame, _at_date={})
    path = Path(simulation.data_manager.data_file(f'ensemble-{run.parent}'))
    return manifest.save(path)


def write_outputs(simulation):
    """clones.csv, the manifests and the figures of the run's ensembles."""
    rows = clone_rows(simulation)
    if simulation.config.save_summary:
        write_clones_table(simulation, rows)
    if simulation.config.save is not None:
        for run in simulation.ensembles:
            write_manifest(simulation, run)
    simulation.clone_summary = pd.DataFrame(rows)
    plot_ensembles(simulation, rows)


def _figure_wanted(simulation, run: EnsembleRun, key: str) -> bool:
    """The ensemble figure of a row is drawn with the plot kinds 'clones' or 'combined', when the PLOT mode selects the
    nominal's row or any clone's."""
    config = simulation.config
    if config.plot is None or not ({'clones', 'combined'} & set(config.plots)):
        return False
    body = _nominal_body(simulation, run)
    if body is not None:
        verdict = body.librations.get(key)
        if status_selected(config.plot, body.statuses.get(key), str(verdict.libration_origin) if verdict is not None else ''):
            return True
    for result in run.results.values():
        row = result.get('rows', {}).get(key)
        if row and status_selected(config.plot, row['status'], row['libration_origin']):
            return True
    return False


def _nominal_series(simulation, body, key: str):
    """(years, degrees) of the nominal's angle: in memory after a single run, else from its data file, else None."""
    if body is None:
        return None
    angle = body.angles.get(key)
    if angle is not None and body.times is not None and len(angle) == len(body.times):
        return np.asarray(body.times, dtype=float) / (2 * np.pi), np.mod(np.degrees(np.asarray(angle, dtype=float)), 360.0)
    path = Path(simulation.config.data_path) / f'data-{body.name}.csv'
    column = f'{key}_angle'
    if path.exists():
        frame = pd.read_csv(path, usecols=lambda c: c in ('times', column))
        if column in frame.columns:
            return frame['times'].to_numpy(), np.mod(np.degrees(frame[column].to_numpy()), 360.0)
    return None


def plot_ensembles(simulation, rows: list):
    """One figure per parent and resonance (`EnsemblePlotter`), in the figures folder of the nominal's row."""
    from resonances.plotting.clone_plots import EnsemblePlotter

    by_key = {(row['name'], row['resonance']): row for row in rows}
    for run in simulation.ensembles:
        body = _nominal_body(simulation, run)
        for resonance in run.resonances:
            key = resonance.to_s()
            if not _figure_wanted(simulation, run, key):
                continue
            members = []
            for name, elem in zip(run.names, run.elements):
                result = run.results.get(name) or {}
                row = result.get('rows', {}).get(key)
                status = 'invalid' if elem is None else (row['status'] if row else None)
                a0 = elem['a'] if elem is not None else np.nan
                members.append({'name': name, 'status': status, 'a0': a0, 'series': result.get('series', {}).get(key)})
            widths = [r['half_width'][key] for r in run.results.values() if key in r.get('half_width', {})]
            nominal_a0 = body.initial_data['a'] if body is not None else None
            nominal_status = body.statuses.get(key) if body is not None else None
            a_star = getattr(resonance, 'resonant_axis', None) if resonance.type == 'mmr' else None
            plotter = EnsemblePlotter().plot(
                parent=run.parent,
                key=key,
                members=members,
                row=by_key[(run.parent, key)],
                nominal_series=_nominal_series(simulation, body, key),
                nominal_a0=nominal_a0,
                nominal_status=nominal_status,
                a_star=a_star,
                half_width=float(np.median(widths)) if widths else None,
            )
            folder = simulation.data_manager.figure_folder(body, resonance) if body is not None else _plain_folder(simulation)
            simulation.data_manager._show_or_save(plotter, f'{folder}/{run.parent}-{key}-clones.{simulation.config.image_type}')
            plotter.close()


def _plain_folder(simulation) -> str:
    from . import layout

    return str(layout.layout_path(simulation.config.plot_path, simulation.config.plot_figures_folder, (), {}))
