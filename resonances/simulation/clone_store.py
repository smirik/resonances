"""What a run with clones keeps on disk, and how it is read back (docs/clones.md "Resume and restore",
docs/serialization.md).

- The checkpoint of the running run, `<save_path>/clones_state/`, removed after the whole run (ordinary bodies and
  clones) has succeeded:
  - `clones_state.json`: the run identity (`run_identity`: a SHA-256 over what was actually run, with one hash per
    component so a refusal can say which one changed), whether the ordinary part is complete, a failure;
  - `starts.npz`: every clone's elements at the run's date (float64), written before any clone batch runs; a resume
    compares its own starts with them bit for bit;
  - `ordinary.npz`: the nominals' angles thinned for the ensemble figure, written when the ordinary part completes;
  - `batch-<index>.npz`: one per finished clone batch, published atomically (a temporary file renamed into place).
    The file's existence, after validation, is the completion record of the batch.
- The results of a finished run, kept: `<data folder>/ensemble-<parent>-results.npz` (with SAVE on, next to the
  manifest `ensemble-<parent>.csv/.json`), read back by `SimulationSerializer.restore`.

A results archive (batch or per parent) holds a JSON header as uint8 bytes and plain numeric arrays (read with
allow_pickle=False): per clone its row verdicts (status, confidence, angle_status, reason, libration_origin), the
initial a and the half-widths as exact floats (JSON keeps a float's repr), its derived rows, and its angles thinned to
FIGURE_POINTS samples: years in float64, degrees in float32 (a figure thumbnail, never read by the classifier).
"""

import datetime
import hashlib
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from resonances.clones.elements import KEPLERIAN_KEYS
from resonances.logger import logger
from resonances.resonance.analysis.verdict import Status

SCHEMA = 1
STATE_DIR = 'clones_state'
STATE_FILE = 'clones_state.json'
STARTS_FILE = 'starts.npz'
ORDINARY_FILE = 'ordinary.npz'
RESULTS_SUFFIX = '-results'
SERIES_PRECISION = 'years float64, degrees float32 (thinned figure series)'
ROW_FIELDS = ('status', 'confidence', 'angle_status', 'reason', 'libration_origin')
# Clone settings that change what the clones are or how they are grouped (CLONES itself is in the ensembles' draws;
# CLONES_PLOT_EACH changes figures only)
CLONE_SETTINGS = ('clones_source', 'clones_epoch', 'clones_seed', 'clones_batch')


# ----------------------------------------------------------------------------------------------- canonical digests
def digest(value) -> str:
    """SHA-256 of a JSON-able value (keys sorted; a float is written as its repr, which reads back to the same bits)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def float_digest(array) -> str:
    """SHA-256 of an array's float64 bytes (C order; NaN marks a missing value)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array, dtype='<f8')).tobytes()).hexdigest()


def starts_array(elements: List[Optional[dict]]) -> np.ndarray:
    """(n, 6) float64 of the elements a, e, inc, Omega, omega, M; a row of NaN for an invalid draw (None)."""
    out = np.full((len(elements), len(KEPLERIAN_KEYS)), np.nan)
    for i, elem in enumerate(elements):
        if elem is not None:
            out[i] = [float(elem[key]) for key in KEPLERIAN_KEYS]
    return out


def _code_versions() -> dict:
    """The versions of the scientific code: a resume never mixes the results of two of them."""
    import rebound

    from resonances import __version__ as package_version

    return {'resonances': package_version, 'rebound': rebound.__version__, 'numpy': np.__version__}


def _planets_state(simulation) -> Optional[str]:
    """Digest of the massive bodies' initial state (m, x, y, z, vx, vy, vz) in the run's REBOUND simulation, or None
    without one."""
    sim = getattr(simulation.integration_engine, 'sim', None)
    if sim is None:
        return None
    rows = [[p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in (sim.particles[i] for i in range(sim.N)) if p.m > 0]
    return float_digest(rows)


def canonical_elements(initial_data) -> dict:
    """A body's initial elements as a JSON-able dict that `digest` writes exactly: every key a string, every float
    (numpy's too) a Python float, written as its repr; the other values (a get-orb sha256, a designation) as they
    are. Shared by the clone-run identity and the ordinary batch resume hash (batch_manager)."""
    return {str(k): (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in dict(initial_data).items()}


def _ordinary_bodies(simulation) -> list:
    """The ordinary bodies as run: name, mass, the requested rows (not the derived ones) and the initial elements."""
    out = []
    for body in simulation.bodies:
        keys = [r.to_s() for r in body.resonances() if r.to_s() not in body.derived_resonances]
        out.append([str(body.name), body.mass, keys, canonical_elements(body.initial_data)])
    return out


def _ensemble_identity(run) -> dict:
    """What one ensemble's clones were: the provider file, the solution, the draws, the starts at the date."""
    ensemble = run.ensemble
    solution = ensemble.solution
    draws = ensemble.clones[list(KEPLERIAN_KEYS)].to_numpy(dtype=float)
    return {
        'parent': run.parent,
        'resonances': run.keys,
        'names': digest(list(run.names)),
        'method': ensemble.method,
        'seed': ensemble.seed,
        'rank': ensemble.rank,
        'epoch_mjd': float(ensemble.epoch_mjd),
        'provider_sha256': (solution.provenance.get('sha256') if solution is not None else None),
        'solution': (float_digest(np.concatenate([solution.nominal, solution.covariance.ravel()])) if solution is not None else None),
        'draws': float_digest(draws),
        'valid': digest([bool(v) for v in ensemble.clones['valid']]),
        'starts': float_digest(starts_array(run.elements)),
        'nominal_offset_sigma': None if not math.isfinite(run.nominal_offset_sigma) else float(run.nominal_offset_sigma),
    }


def run_identity(simulation, layout: list, batched: bool) -> dict:
    """{'hash': ..., 'components': {...}} of a run with clones: the ordinary config (the resume hash's, without the
    clone settings), the ordinary bodies with their elements and grouping (batched or one simulation), the planets'
    initial state, the clone settings, the clone batch layout (names and rows per batch), each ensemble
    (`_ensemble_identity`) and the versions of the code."""
    config_dict = simulation.batch_manager._get_config_dict()
    ordinary = {
        'config': config_dict,
        'bodies': _ordinary_bodies(simulation),
        'grouping': (
            [[body['name'] for body in batch] for batch in simulation.batch_manager.create_batches(simulation.bodies)]
            if batched
            else 'single'
        ),
        'planets': _planets_state(simulation),
    }
    components = {
        'ordinary': digest(ordinary),
        'settings': digest({key: getattr(simulation.config, key, None) for key in CLONE_SETTINGS}),
        'layout': digest([[[body['name'], [r.to_s() for r in body['resonances']]] for body in batch] for _, batch in layout]),
        'code': digest(_code_versions()),
        'ensembles': {run.parent: digest(_ensemble_identity(run)) for run in simulation.ensembles},
    }
    details = {run.parent: _ensemble_identity(run) for run in simulation.ensembles}
    return {'hash': digest(components), 'components': components, 'ensembles': details}


# ----------------------------------------------------------------------------------------------- atomic files
def write_json(path: Path, data: dict) -> None:
    """`data` to `path` through a temporary file in the same folder, flushed and renamed into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w') as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """A compressed .npz written through an open temporary file (np.savez would append '.npz' to a name), flushed and
    renamed into place: the final name exists only for a complete archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _header_array(header: dict) -> np.ndarray:
    return np.frombuffer(json.dumps(header, sort_keys=True).encode(), dtype=np.uint8)


def _header_of(arrays: Dict[str, np.ndarray]) -> dict:
    return json.loads(bytes(arrays['header'].astype(np.uint8)).decode())


# ----------------------------------------------------------------------------------------------- results archives
def encode_results(results: dict, header: dict, nominal: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """The arrays of a results archive: `results` ({clone name: {'a0', 'rows', 'half_width', 'series', 'derived'}}),
    `nominal` ({parent: {key: (years, degrees)}}, the nominals' thumbnails) and `header` (identity fields)."""
    arrays, grids, clones, nominals = {}, [], {}, {}

    def put(series) -> list:
        t = np.asarray(series[0], dtype=np.float64)
        deg = np.asarray(series[1], dtype=np.float32)
        grid = next((g for g, known in enumerate(grids) if len(known) == len(t) and np.array_equal(known, t)), None)
        if grid is None:
            grid = len(grids)
            grids.append(t)
            arrays[f'grid_{grid}'] = t
        index = sum(1 for name in arrays if name.startswith('deg_'))
        arrays[f'deg_{index}'] = deg
        return [grid, index]

    for name, result in results.items():
        clones[str(name)] = {
            'a0': float(result['a0']),
            'rows': {key: {field: str(row[field]) for field in ROW_FIELDS} for key, row in result.get('rows', {}).items()},
            'half_width': {key: float(value) for key, value in result.get('half_width', {}).items()},
            'derived': [list(pair) for pair in result.get('derived', [])],
            'series': {key: put(series) for key, series in result.get('series', {}).items() if series is not None},
        }
    for parent, by_key in (nominal or {}).items():
        nominals[str(parent)] = {key: put(series) for key, series in by_key.items() if series is not None}
    arrays['header'] = _header_array(
        {**header, 'schema': SCHEMA, 'series_precision': SERIES_PRECISION, 'clones': clones, 'nominal': nominals}
    )
    return arrays


def decode_results(arrays: Dict[str, np.ndarray]) -> tuple:
    """(header, results, nominal) of a results archive; ValueError when it is not a valid one (schema, statuses,
    finite numbers, series that exist and match their time grid)."""
    if 'header' not in arrays:
        raise ValueError('no header')
    header = _header_of(arrays)
    if header.get('schema') != SCHEMA:
        raise ValueError(f"schema {header.get('schema')!r}, expected {SCHEMA}")
    statuses = {str(s) for s in Status}

    def get(ref) -> tuple:
        grid, index = (int(x) for x in ref)
        t, deg = arrays[f'grid_{grid}'], arrays[f'deg_{index}']
        if t.shape != deg.shape or t.dtype != np.float64 or deg.dtype != np.float32:
            raise ValueError(f'series {index}: shape or type does not match its time grid')
        return t, deg.astype(np.float64)

    results = {}
    for name, entry in header['clones'].items():
        if not math.isfinite(float(entry['a0'])):
            raise ValueError(f'{name}: initial a is not finite')
        for key, row in entry['rows'].items():
            if set(row) != set(ROW_FIELDS) or row['status'] not in statuses:
                raise ValueError(f'{name} {key}: not a verdict row')
        if not all(math.isfinite(float(w)) for w in entry['half_width'].values()):
            raise ValueError(f'{name}: a half-width is not finite')
        results[name] = {
            'a0': float(entry['a0']),
            'rows': {key: dict(row) for key, row in entry['rows'].items()},
            'half_width': {key: float(w) for key, w in entry['half_width'].items()},
            'derived': [list(pair) for pair in entry.get('derived', [])],
            'series': {key: get(ref) for key, ref in entry['series'].items()},
        }
    nominal = {parent: {key: get(ref) for key, ref in by_key.items()} for parent, by_key in header.get('nominal', {}).items()}
    return header, results, nominal


def load_results(path) -> tuple:
    """(header, results, nominal) of a results archive on disk (`decode_results`)."""
    return decode_results(read_npz(Path(path)))


def results_path(simulation, parent: str) -> Path:
    """`<data folder>/ensemble-<parent>-results.npz`."""
    return Path(simulation.config.data_path) / f'ensemble-{parent}{RESULTS_SUFFIX}.npz'


# ----------------------------------------------------------------------------------------------- the checkpoint
class CloneCheckpoint:
    """`<save_path>/clones_state/` of a run with clones (see the module's docstring). Only the coordinating simulation
    (the one with ensembles) opens one; batch workers never do."""

    def __init__(self, save_path):
        self.folder = Path(save_path) / STATE_DIR
        self.state_file = self.folder / STATE_FILE
        self.state: dict = {}

    # -- lifecycle
    def open(self, identity: dict, starts: Dict[str, tuple], resume: bool, name: str) -> bool:
        """Start the run's checkpoint, or continue an interrupted one with the same identity (returns True then).

        With `resume` off any old checkpoint is removed. A checkpoint of another run identity raises ValueError naming
        what changed; nothing is run or removed then."""
        if not resume:
            self.clear()
        old = self._read_state() if resume else None
        if old is not None:
            if old.get('hash') != identity['hash']:
                raise ValueError(self._refusal(old, identity, name))
            self._check_starts(starts, name)
            self.state = old
            self.state['failed'], self.state['error'] = False, None
            self._save()
            logger.info(
                f"Resuming the clones of '{name}' ({self.folder}): ordinary part {'complete' if old['ordinary']['complete'] else 'pending'}"
            )
            return True
        if self.folder.exists():
            shutil.rmtree(self.folder)
        now = _now()
        self.state = {
            'schema': SCHEMA,
            'kind': 'clones-run',
            'name': name,
            'hash': identity['hash'],
            'components': identity['components'],
            'ensembles': identity['ensembles'],
            'ordinary': {'complete': False},
            'failed': False,
            'error': None,
            'started_at': now,
            'updated_at': now,
        }
        arrays = {f'starts_{i}': np.asarray(matrix, dtype='<f8') for i, (_, matrix) in enumerate(starts.values())}
        arrays['header'] = _header_array({'schema': SCHEMA, 'parents': {p: [i, list(n)] for i, (p, (n, _)) in enumerate(starts.items())}})
        write_npz(self.folder / STARTS_FILE, arrays)
        self._save()
        return False

    def ordinary_done(self, thumbnails: dict) -> None:
        """The ordinary part is complete: its nominals' thumbnails, then the flag."""
        write_npz(self.folder / ORDINARY_FILE, encode_results({}, {'kind': 'ordinary', 'hash': self.state['hash']}, thumbnails))
        self.state['ordinary'] = {'complete': True}
        self._save()

    def ordinary_thumbnails(self) -> dict:
        path = self.folder / ORDINARY_FILE
        try:
            header, _, nominal = load_results(path)
            return nominal if header.get('hash') == self.state.get('hash') else {}
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            return {}

    def mark_failed(self, error: Exception) -> None:
        if not self.state:
            return
        self.state['failed'], self.state['error'] = True, str(error)
        self._save()

    def clear(self) -> None:
        """Remove the checkpoint (after the whole run succeeded, or before a run without resume)."""
        if self.folder.exists():
            shutil.rmtree(self.folder)
            logger.info(f'Clone checkpoint removed: {self.folder}')

    # -- batches
    def batch_file(self, index: int) -> Path:
        return self.folder / f'batch-{index:05d}.npz'

    def write_batch(self, index: int, names: list, keys: Dict[str, list], results: dict) -> None:
        header = {'kind': 'clone-batch', 'hash': self.state['hash'], 'batch': int(index), 'names': list(names), 'keys': keys}
        write_npz(self.batch_file(index), encode_results(results, header))

    def load_batch(self, index: int, names: list, keys: Dict[str, list]) -> Optional[dict]:
        """The results of a finished batch, or None when it has no valid file (absent, unreadable, another run, another
        membership or rows): such a batch runs again (its old table rows are dropped first)."""
        path = self.batch_file(index)
        if not path.exists():
            return None
        try:
            header, results, _ = load_results(path)
            problems = []
            if header.get('kind') != 'clone-batch' or header.get('hash') != self.state['hash'] or header.get('batch') != index:
                problems.append('another run or batch')
            if header.get('names') != list(names) or header.get('keys') != keys or sorted(results) != sorted(names):
                problems.append('other members or rows')
            for name, result in results.items():
                extra = set(result['rows']) - set(keys.get(name, [])) - {key for key, _ in result['derived']}
                if extra:
                    problems.append(f'{name}: rows {sorted(extra)} not asked for')
            if problems:
                raise ValueError('; '.join(problems))
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
            logger.warning(f'Clone batch {index}: {path.name} is not a valid result ({error}); the batch runs again')
            return None
        return results

    # -- internals
    def _read_state(self) -> Optional[dict]:
        if not self.state_file.exists():
            return None
        try:
            state = json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f'{self.state_file} cannot be read ({error}); remove {self.folder} to start over') from None
        if state.get('schema') != SCHEMA or state.get('kind') != 'clones-run':
            raise ValueError(f'{self.state_file} is of another version; remove {self.folder} to start over')
        return state

    def _save(self) -> None:
        self.state['updated_at'] = _now()
        write_json(self.state_file, self.state)

    def _check_starts(self, starts: Dict[str, tuple], name: str) -> None:
        """The clones' starting elements equal the saved ones bit for bit (else ValueError)."""
        arrays = read_npz(self.folder / STARTS_FILE)
        saved = _header_of(arrays)['parents']
        for parent, (names, matrix) in starts.items():
            index, saved_names = saved.get(parent, (None, None))
            if index is None or saved_names != list(names) or arrays[f'starts_{index}'].tobytes() != np.asarray(matrix, '<f8').tobytes():
                raise ValueError(
                    f"Cannot resume the clones of '{name}': the starting elements of {parent}'s clones differ from those "
                    f'saved in {self.folder / STARTS_FILE} (bit for bit). Remove {self.folder} to start over.'
                )

    def _refusal(self, old: dict, identity: dict, name: str) -> str:
        """Why a checkpoint of another run is refused: the components that changed."""
        before, now = old.get('components', {}), identity['components']
        changed = [part for part in ('ordinary', 'settings', 'layout', 'code') if before.get(part) != now.get(part)]
        what = {
            'ordinary': 'the ordinary bodies, their elements, grouping or settings, or the planets',
            'settings': 'the clone settings (CLONES_SOURCE, CLONES_EPOCH, CLONES_SEED, CLONES_BATCH)',
            'layout': 'the clone batches (CLONES_BATCH, the clones or their rows)',
            'code': 'the version of resonances, rebound or numpy',
        }
        reasons = [what[part] for part in changed]
        hint = ''
        old_ensembles, new_ensembles = old.get('ensembles', {}), identity['ensembles']
        for parent in sorted(set(old_ensembles) | set(new_ensembles)):
            a, b = old_ensembles.get(parent), new_ensembles.get(parent)
            if a is None or b is None:
                reasons.append(f'the ensemble of {parent} was {"added" if a is None else "removed"}')
                continue
            fields = [field for field in sorted(set(a) | set(b)) if a.get(field) != b.get(field)]
            if not fields:
                continue
            reasons.append(f'the ensemble of {parent} ({", ".join(fields)})')
            if 'provider_sha256' in fields:
                hint = (
                    f' The orbit-solution file of {parent} changed (orbit_solution(..., refresh=True) fetched it again): '
                    'the cloud is another one.'
                )
        return (
            f"Cannot resume '{name}' in {self.folder.parent}: {'; '.join(reasons) or 'the run identity'} differ from the "
            f'interrupted run.{hint} Pass the original bodies and settings, or remove {self.folder} (and '
            'simulation_state.json) to start over.'
        )


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec='seconds')


# ----------------------------------------------------------------------------------------------- simulation.json
def ensembles_entry(simulation) -> Optional[dict]:
    """The `ensembles` entry of simulation.json (None without ensembles): per ensemble its parent, rows, size, whether
    every clone batch finished, and the run-relative paths of its manifest and results (null while the clones are not
    finished, or with SAVE off: no manifest then)."""
    runs = getattr(simulation, 'ensembles', None)
    if not runs:
        return None
    save_path = Path(simulation.config.save_path)

    def relative(path: Path) -> Optional[str]:
        return Path(os.path.relpath(path, save_path)).as_posix() if path.exists() else None

    items = []
    for run in runs:
        solution = run.ensemble.solution
        complete = bool(run.complete)
        offset = float(run.nominal_offset_sigma)
        items.append(
            {
                'parent': run.parent,
                'nominal_in_run': any(str(body.name) == run.parent for body in simulation.bodies),
                'resonances': run.keys,
                'n_clones': len(run.names),
                'n_valid': sum(1 for elem in run.elements if elem is not None),
                'complete': complete,
                'manifest': relative(Path(simulation.config.data_path) / f'ensemble-{run.parent}.csv') if complete else None,
                'results': relative(results_path(simulation, run.parent)) if complete else None,
                'nominal_offset_sigma': offset if math.isfinite(offset) else None,
                'source': solution.source if solution is not None else run.ensemble.method,
                'seed': run.ensemble.seed,
                'clone_verdicts': run.clone_verdicts,
            }
        )
    return {'version': SCHEMA, 'items': items}


def _clones_table(path: Path):
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, dtype={'name': str, 'resonance': str, 'nominal_status': str, 'nominal_confidence': str})


def restore_ensembles(sim, base_dir: Path, simulation_data: dict) -> list:
    """Rebuild `sim.ensembles` and `sim.clone_summary` from simulation.json's `ensembles` entry (nothing for a file
    without one: a run of an earlier version, or one without clones). Returns the parents restored from clones.csv
    alone.

    An ensemble with its manifest and results archive comes back whole: its clones' names, their elements at the date
    (the manifest's `<element>_start`, read to the bit), the stored verdict rows, half-widths and thinned angles, the
    nominal's thumbnail. Without them (SAVE off, or files removed) only its clones.csv rows are restored, with a
    warning. The clones keep the verdicts of the run (`clone_verdicts = 'stored'`)."""
    from dataclasses import replace

    from resonances.clones import Ensemble
    from resonances.data.util import mjd_from_datetime
    from resonances.resonance.factory import create_resonance

    from .ensemble_run import EnsembleRun

    entry = simulation_data.get('ensembles')
    if not entry:
        return []
    if entry.get('version') != SCHEMA:
        logger.warning(f"simulation.json: ensembles of version {entry.get('version')!r}, not {SCHEMA}: not restored")
        return []
    table = _clones_table(base_dir / (simulation_data.get('data_files', {}).get('clones') or 'clones.csv'))
    date_mjd = mjd_from_datetime(sim.config.date)
    summary_only, parents = [], []
    for item in entry.get('items', []):
        parent = str(item['parent'])
        parents.append(parent)
        if not item.get('complete'):
            logger.warning(f'{parent}: its clones had not finished when simulation.json was written (an interrupted run)')
        manifest, results_file = item.get('manifest'), item.get('results')
        if not manifest or not (base_dir / manifest).exists() or not results_file or not (base_dir / results_file).exists():
            logger.warning(
                f'{parent}: no ensemble manifest or results archive (SAVE off, or removed): only its clones.csv rows are '
                'restored (sim.clone_summary), not its clones'
            )
            summary_only.append(parent)
            continue
        ensemble = Ensemble.load(base_dir / manifest)
        frame = ensemble.clones
        elements = [
            {**{key: float(getattr(row, f'{key}_start')) for key in KEPLERIAN_KEYS}, 'epoch': date_mjd} if row.valid else None
            for row in frame.itertuples(index=False)
        ]
        header, results, nominal = load_results(base_dir / results_file)
        if header.get('parent') != parent or header.get('names') != frame['name'].astype(str).tolist():
            raise ValueError(f'{base_dir / results_file}: not the results of the ensemble in {manifest}')
        offset = item.get('nominal_offset_sigma')
        sim.ensembles.append(
            EnsembleRun(
                ensemble=replace(ensemble, _at_date={}),
                parent=parent,
                resonances=[create_resonance(key) for key in item['resonances']],
                names=frame['name'].astype(str).tolist(),
                elements=elements,
                nominal_offset_sigma=float('nan') if offset is None else float(offset),
                results=results,
                complete=bool(item.get('complete')),
                nominal_series=nominal.get(parent, {}),
                clone_verdicts='stored',
            )
        )
    if table is not None:
        sim.clone_summary = table[table['name'].isin(parents)].reset_index(drop=True)
    elif parents:
        logger.warning(f'{base_dir / "clones.csv"} is missing: no clone summary (save_summary was off?)')
    return summary_only


def recompute_clone_rows(sim, base_dir: Path, summary_only: list) -> None:
    """After `restore(recompute_librations=True)` reclassified the nominals: the clones.csv rows of the restored
    ensembles with the nominals' new verdicts (nominal_status, nominal_confidence, nominal_agreement); the clones keep
    their stored verdicts. `sim.clone_summary` gets them and clones.csv in the run's folder (`base_dir`) is rewritten
    for these parents (author 2026-10-28)."""
    import pandas as pd

    from resonances.clones.stats import STATUSES, ensemble_row

    from .ensemble_run import clone_rows, write_clones_table

    rows = clone_rows(sim)
    table = sim.clone_summary
    if summary_only and table is not None:
        bodies = {str(body.name): body for body in sim.bodies}
        for record in table[table['name'].isin(summary_only)].to_dict('records'):
            body = bodies.get(str(record['name']))
            key = str(record['resonance'])
            status = body.statuses.get(key) if body is not None else None
            verdict = body.librations.get(key) if body is not None else None
            # the clones' statuses as counted in the row (their order does not enter the shares)
            statuses = [s for s in STATUSES for _ in range(int(record[f'n_{s}']))] + [None] * int(record['n_missing'])
            record.update({'nominal_status': status or '', 'nominal_confidence': str(verdict.confidence) if verdict is not None else ''})
            record.update(ensemble_row(statuses, status))
            rows.append(record)
    if not rows:
        return
    if sim.config.save_summary:
        write_clones_table(sim, rows, folder=base_dir)
    sim.clone_summary = pd.DataFrame(rows)
    logger.info('Clones: the nominals reclassified, the clones keep the verdicts of the run (clone_verdicts=stored)')
