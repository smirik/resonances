"""Read a benchmark example (benchmark/LAYOUT.md) into an `MMRSeries` — no Body, no integration.

Layout: <data>/examples/<example_id>/{series.csv.gz, meta.json, planets.csv.gz}; <data> is
`benchmark/data` or $RESONANCES_BENCHMARK_DATA. series.csv.gz columns: t_yr, a, e, inc, Omega, omega,
varpi, M, mean_longitude (formerly `lam`), angle_raw (np.unwrap-ed, unfiltered). planets.csv.gz:
t_yr and `<Planet>.<field>` for a, e, inc, Omega, omega, M, mean_longitude (formerly `l`), varpi.
The step is taken from the series itself (some examples are decimated, meta.regime.decimation).

Some examples keep only the angle's planets; the planets the run saved (<data>/runs/<run>/planets.csv.gz) complete them
when that file is on the same time grid (the planets do not depend on the body): the owner search needs every planet's
longitude (1997 GK3 3M+5S-3 is driven by 10E-17, and its example holds Mars and Saturn only). <run> is
meta.provenance.planets_run_id when set (a run of the same epoch, integrator, planets and output grid whose planets
reproduce the example's saved angle: the NEA production run saved none), else provenance.run_id.

A planet still without a saved lambda gets a catalogue longitude (`MMRSeries.catalogue_longitudes`, the owner search
alone): its lambda at the first row of the run's planets file (any grid), else from the solar-system cache of the
run's epoch (provenance.solar_system_cache, else the run's run.json).
"""

from __future__ import annotations

import functools
import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from resonances.resonance.analysis.mmr.context import CATALOGUE_PLANETS, catalogue_longitudes
from resonances.resonance.analysis.mmr.models import PLANET_NAMES_ORDER, SAME_GRID_TOLERANCE, MMRSeries, PlanetSeries, Regime, ResonanceSpec

DATA_ENV = 'RESONANCES_BENCHMARK_DATA'
LEGACY_SERIES_COLUMNS = {'lam': 'mean_longitude'}
LEGACY_PLANET_SUFFIXES = {'.l': '.mean_longitude'}
PLANET_FIELDS = ('a', 'e', 'inc', 'Omega', 'varpi', 'mean_longitude')
SERIES_COLUMNS = ('t_yr', 'a', 'e', 'inc', 'Omega', 'omega', 'varpi', 'M', 'mean_longitude', 'angle_raw')
TWO_PI = 2 * np.pi
CHUNK_ROWS = 300_000  # rows pandas parses at a time when numpy cannot (empty fields)


def benchmark_root() -> Path:
    """The tracked benchmark/ folder of the repository (labels.csv, DECISIONS.md, tools/)."""
    return Path(__file__).resolve().parents[4] / 'benchmark'


def data_root() -> Path:
    """$RESONANCES_BENCHMARK_DATA or benchmark/data (may not exist: tests then skip)."""
    configured = os.environ.get(DATA_ENV)
    return Path(configured).expanduser().resolve() if configured else benchmark_root() / 'data'


def example_path(example_id: str) -> Path:
    """<data>/examples/<example_id>."""
    return data_root() / 'examples' / example_id


def _folder(example: str | Path) -> Path:
    path = Path(example)
    return path if path.is_dir() else example_path(str(example))


def read_meta(example: str | Path) -> Dict[str, Any]:
    """meta.json of an example (id or folder)."""
    with open(_folder(example) / 'meta.json') as f:
        return json.load(f)


def list_examples(resonance_type: Optional[str] = 'mmr') -> List[str]:
    """Example ids with a meta.json, filtered by meta.resonance_type (None = all), sorted."""
    root = data_root() / 'examples'
    if not root.exists():
        return []
    ids = []
    for folder in sorted(root.iterdir()):
        if not (folder / 'meta.json').exists():
            continue
        if resonance_type is not None and read_meta(folder).get('resonance_type') != resonance_type:
            continue
        ids.append(folder.name)
    return ids


def _current_names(header: List[str]) -> Dict[str, str]:
    """Each column under its current name: a legacy name (`lam`, `<Planet>.l`) is renamed when the new one is absent."""
    names = {}
    for column in header:
        new = LEGACY_SERIES_COLUMNS.get(column, column)
        for old, suffix in LEGACY_PLANET_SUFFIXES.items():
            if column.endswith(old):
                new = column[: -len(old)] + suffix
        names[column] = column if new in header else new
    return names


def _read_columns(path: Path, wanted: Iterable[str]) -> Dict[str, np.ndarray]:
    """The `wanted` numeric columns of a CSV as float arrays, under their current names (`_current_names`). numpy parses
    them straight into one array (the columns are views of it); a file with empty fields goes through pandas in chunks.
    A whole DataFrame of the 73 planet columns at 1e6 rows peaked at 2.6 GB for 0.5 GB of arrays (285637 2 Myr); this
    reads them at 0.65 GB."""
    import pandas as pd

    header = list(pd.read_csv(path, nrows=0).columns)
    names = _current_names(header)
    wanted = set(wanted)
    usecols = [column for column in header if names[column] in wanted]
    if not usecols:
        return {}
    try:
        with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as handle:
            block = np.loadtxt(handle, delimiter=',', skiprows=1, usecols=[header.index(c) for c in usecols], dtype=np.float64, ndmin=2)
        return {names[column]: block[:, j] for j, column in enumerate(usecols)}
    except ValueError:  # an empty field (a column the run did not fill): pandas reads it as NaN
        pass
    chunks: Dict[str, list] = {names[column]: [] for column in usecols}
    for chunk in pd.read_csv(path, usecols=usecols, dtype={column: np.float64 for column in usecols}, chunksize=CHUNK_ROWS):
        for column in usecols:
            chunks[names[column]].append(chunk[column].to_numpy(np.float64, copy=True))
        del chunk
    columns = {}
    for name in list(chunks):  # join one column at a time, freeing its chunks
        parts = chunks.pop(name)
        columns[name] = np.concatenate(parts) if parts else np.empty(0)
    return columns


def _ordered(values: np.ndarray, order: Optional[np.ndarray]) -> np.ndarray:
    """values in time order (`order` None: already ordered, no copy)."""
    return values if order is None else values[order]


def _time_order(t_raw: np.ndarray) -> Optional[np.ndarray]:
    """The stable sort of the times, or None when they already increase (a forward run: no copies)."""
    if t_raw.size < 2 or bool(np.all(np.diff(t_raw) >= 0)):
        return None
    return np.argsort(t_raw, kind='stable')  # a backward run is reversed here


def _planet_columns(path: Path, names) -> Dict[str, np.ndarray]:
    """t_yr and the `PLANET_FIELDS` of `names` from a planets file, in one read."""
    return _read_columns(path, ['t_yr'] + [f'{name}.{field}' for name in names for field in PLANET_FIELDS])


def _planets(folder: Path, names, order: Optional[np.ndarray], n_rows: int, frame=None) -> Dict[str, PlanetSeries]:
    """The planets `names` of folder/planets.csv.gz (`frame`: its columns already read by `_planet_columns`)."""
    path = folder / 'planets.csv.gz'
    if frame is None:
        if not path.exists():
            return {}
        frame = _planet_columns(path, names)
    rows = len(frame['t_yr'])
    if rows != n_rows:
        raise ValueError(f'{folder.name}: planets have {rows} rows, the series {n_rows}')
    t = _ordered(frame.pop('t_yr'), order)
    planets = {}
    for name in names:
        columns = [f'{name}.{field}' for field in PLANET_FIELDS]
        if not all(column in frame for column in columns[:-1]):
            continue
        values = {field: _ordered(frame.pop(f'{name}.{field}'), order) for field in PLANET_FIELDS if f'{name}.{field}' in frame}
        planets[name] = PlanetSeries(
            name, t, values['a'], values['e'], values['inc'], values['Omega'], values['varpi'], values.get('mean_longitude')
        )
    return planets


def planets_run(meta: Dict[str, Any]) -> Optional[str]:
    """The run whose planets the example reads: provenance.planets_run_id, else provenance.run_id (module docstring)."""
    provenance = meta.get('provenance') or {}
    return provenance.get('planets_run_id') or provenance.get('run_id')


def _run_planets(meta: Dict[str, Any], t_yr: np.ndarray, missing) -> Dict[str, PlanetSeries]:
    """The `missing` planets from the run's planets file (`planets_run`), only when its times are those of the series."""
    run = planets_run(meta)
    folder = data_root() / 'runs' / str(run)
    path = folder / 'planets.csv.gz'
    if not run or not missing or not path.exists():
        return {}
    frame = _planet_columns(path, list(missing))  # one read: the times are checked on it (the file was read twice)
    t_raw = frame['t_yr']
    order = _time_order(t_raw)
    step = float(np.median(np.diff(t_yr))) if len(t_yr) > 1 else 0.0
    if len(t_raw) != len(t_yr) or not np.allclose(_ordered(t_raw, order), t_yr, rtol=0.0, atol=SAME_GRID_TOLERANCE * step):
        return {}
    return _planets(folder, list(missing), order, len(t_raw), frame)


def solar_system_longitudes(path: str | Path) -> Optional[Tuple[float, Dict[str, float]]]:
    """(t [yr], heliocentric lambda_j [rad] of every planet) of a solar-system cache (a REBOUND binary written by
    `IntegrationEngine.create_solar_system` at the run's epoch, particles hashed by name: the convention of the saved
    planets); None when the file is missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    return _read_solar_system(str(path), path.stat().st_mtime)


@functools.lru_cache(maxsize=4)
def _read_solar_system(path: str, mtime: float) -> Optional[Tuple[float, Dict[str, float]]]:
    """`solar_system_longitudes` of one file version (read once per process: every MMR of a run asks for it)."""
    import rebound

    try:
        sim = rebound.Simulation(path)
        sun = sim.particles[0]
        longitudes = {name: float(sim.particles[name].orbit(primary=sun).l) for name in PLANET_NAMES_ORDER}
    except Exception:  # noqa: BLE001 - another REBOUND version or a cache without these hashes: no catalogue
        return None
    return float(sim.t / TWO_PI), longitudes


def _run_file_longitudes(meta: Dict[str, Any], names: List[str]) -> Optional[Tuple[float, Dict[str, float]]]:
    """(t [yr], lambda_j [rad]) of `names` at the first row of the run's planets file (`planets_run`); None without it."""
    path = data_root() / 'runs' / str(planets_run(meta)) / 'planets.csv.gz'
    if not path.exists():
        return None
    import pandas as pd

    header = list(pd.read_csv(path, nrows=0).columns)
    names_now = _current_names(header)
    columns = {names_now[column]: column for column in header}
    wanted = {name: columns[f'{name}.mean_longitude'] for name in names if f'{name}.mean_longitude' in columns}
    if 't_yr' not in columns or not wanted:
        return None
    row = pd.read_csv(path, usecols=['t_yr', *wanted.values()], nrows=1).iloc[0]
    return float(row['t_yr']), {name: float(row[column]) for name, column in wanted.items()}


def _cache_path(meta: Dict[str, Any]) -> Optional[str]:
    """The solar-system cache of the example's run: provenance.solar_system_cache.path, else that of its run.json."""
    cache = (meta.get('provenance') or {}).get('solar_system_cache') or {}
    if cache.get('path'):
        return str(cache['path'])
    run_json = data_root() / 'runs' / str((meta.get('provenance') or {}).get('run_id')) / 'run.json'
    if not run_json.exists():
        return None
    with open(run_json) as f:
        return ((json.load(f).get('solar_system_cache')) or {}).get('path')


def _catalogue(meta: Dict[str, Any], t_start_yr: float, planets: Dict[str, PlanetSeries]) -> Dict[str, float]:
    """`MMRSeries.catalogue_longitudes`: lambda_j(t_start) of the catalogue planets without a saved lambda, from the run's
    planets file, else from its solar-system cache (module docstring); {} when neither is there."""
    missing = [name for name in CATALOGUE_PLANETS if name not in planets or planets[name].mean_longitude is None]
    if not missing:
        return {}
    source = _run_file_longitudes(meta, missing)
    if source is None and _cache_path(meta):
        source = solar_system_longitudes(_cache_path(meta))
    if source is None:
        return {}
    t_ref, longitudes = source
    return catalogue_longitudes({name: longitudes[name] for name in missing if name in longitudes}, t_ref, t_start_yr)


def _column(frame: Dict[str, np.ndarray], name: str, order: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if name not in frame or np.isnan(frame[name]).all():
        return None
    return _ordered(frame[name], order)


def load_series(example: str | Path, with_planets: bool = True) -> MMRSeries:
    """The example as an MMRSeries: time-ordered, angle_wrapped = angle_raw mod 2pi, angle_unwrapped_numpy = angle_raw,
    resonance from meta.resonance, regime from meta.regime with the step measured from t_yr, planets from planets.csv.gz.
    Accepts the legacy column names.

    Every saved planet is loaded (the context measures mean motions and the great inequality from them), the run's
    planets file completing the example's (`_run_planets`); the angle's own planets are kept only when all of them are
    saved (`MMRSeries.validate`). The planets left without a lambda get catalogue longitudes (`_catalogue`)."""
    folder = _folder(example)
    meta = read_meta(folder)
    frame = _read_columns(folder / 'series.csv.gz', SERIES_COLUMNS)
    t_raw = frame['t_yr']
    order = _time_order(t_raw)
    t = _ordered(t_raw, order)
    spec = ResonanceSpec.from_key(meta['resonance'])
    planets = _planets(folder, PLANET_NAMES_ORDER, order, len(t_raw)) if with_planets else {}
    if with_planets:
        planets.update(_run_planets(meta, t, [name for name in PLANET_NAMES_ORDER if name not in planets]))
        planets = {name: planets[name] for name in PLANET_NAMES_ORDER if name in planets}
    if any(name not in planets for name in spec.planets):  # the angle's planets: all or none (MMRSeries.validate)
        planets = {name: planet for name, planet in planets.items() if name not in spec.planets}
    regime_meta = meta.get('regime', {})
    regime = Regime(
        step_yr=float(np.median(np.diff(t))),
        span_yr=float(t[-1] - t[0]),
        n_rows=len(t),
        integrator=str(regime_meta.get('integrator') or ''),
        dt=float(regime_meta['dt']) if regime_meta.get('dt') is not None else float('nan'),
        direction=str(regime_meta.get('direction') or 'forward'),
        epoch_jd=float(regime_meta['epoch_jd']) if regime_meta.get('epoch_jd') is not None else float('nan'),
        decimation=int((regime_meta.get('decimation') or {}).get('factor', 1) or 1),
        planets_saved=tuple(planets),
    )
    angle_raw = _ordered(frame['angle_raw'], order)
    series = MMRSeries(
        t_yr=t,
        a=_ordered(frame['a'], order),
        e=_ordered(frame['e'], order),
        inc=_ordered(frame['inc'], order),
        Omega=_ordered(frame['Omega'], order),
        omega=_ordered(frame['omega'], order),
        varpi=_ordered(frame['varpi'], order),
        angle_wrapped=np.mod(angle_raw, TWO_PI),
        resonance=spec,
        regime=regime,
        planets=planets,
        mean_longitude=_column(frame, 'mean_longitude', order),
        mean_anomaly=_column(frame, 'M', order),
        angle_unwrapped_numpy=angle_raw,
        name=str(meta.get('name', folder.name)),
        catalogue_longitudes=_catalogue(meta, float(t[0]), planets) if with_planets else {},
    )
    series.validate()
    return series
