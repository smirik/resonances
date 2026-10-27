"""Read and write the benchmark layout (see ../LAYOUT.md), and rebuild package objects from it.

An example is one body in one resonance: `data/examples/<example_id>/` with `series.csv.gz` (raw series),
`meta.json` and optionally a `planets.csv.gz` symlink to the run's planets. Nothing here integrates:
`to_body` rebuilds a `resonances.Body`, its resonance and a minimal simulation stand-in (times, config,
planets) from the saved files, which is all the classifier and the plotters read.
"""

from __future__ import annotations

import contextlib
import datetime
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parent
# The series and plots live outside git (5+ GB); RESONANCES_BENCHMARK_DATA points elsewhere if they are not in benchmark/data.
DATA = Path(os.environ.get('RESONANCES_BENCHMARK_DATA') or BENCH / 'data').expanduser().resolve()
EXAMPLES = DATA / 'examples'
RUNS = DATA / 'runs'
REVIEW = DATA / 'review'
LABELS = BENCH / 'labels.csv'

SCHEMA_VERSION = 1
FLOAT_FORMAT = '%.12g'
NEA_Q = 1.3  # au: perihelion below this = NEA = ias15 (SABA cannot follow Earth crossers, see memory zlk-coherence-gate)

# Every example has these, in this order. Secular examples add angle_reduced and src_angle.
BASE_COLUMNS = ['t_yr', 'a', 'e', 'inc', 'Omega', 'omega', 'varpi', 'M', 'mean_longitude', 'angle_raw']
SECULAR_COLUMNS = ['angle_reduced', 'src_angle']
# Keys of IntegrationEngine.planets_data; in planets.csv.gz a column is '<Planet>.<name>' (PLANET_COLUMN_NAMES renames 'l').
PLANET_FIELDS = ['a', 'e', 'inc', 'Omega', 'omega', 'M', 'l', 'varpi']
PLANET_COLUMN_NAMES = {'l': 'mean_longitude'}
# Files written before 2026-10-15 used short names; readers accept both (tools/rename_columns.py rewrites them).
LEGACY_SERIES_COLUMNS = {'lam': 'mean_longitude'}
LEGACY_PLANET_SUFFIXES = {'.l': '.mean_longitude'}
LABEL_COLUMNS = ['example_id', 'candidate_id', 'rtype', 'proposed_bucket', 'confidence', 'author_bucket', 'claude_bucket', 'comment']
# Never overwritten by scripts once set. author_* beyond author_bucket come from tools/import_author_labels.py.
HUMAN_LABEL_COLUMNS = [
    'author_bucket',
    'claude_bucket',
    'comment',
    'author_confidence',
    'author_angle_status',
    'author_comment',
    'author_discuss',
    'author_agreed_with_claude',
    'author_labelled_at',
]
CONFIDENCES = ['high', 'medium', 'low']
# The verdicts' statuses, confidences and angle words (`analysis.verdict` Status, Confidence without '', AngleStatus,
# in their order; tested against the enums), for the tools that read or write labels without loading the package
STATUSES = ['resonant', 'transient', 'near_separatrix', 'non_resonant', 'uncertain', 'invalid']
LABEL_CONFIDENCES = ['certain', 'high', 'medium', 'low']
ANGLE_STATUSES = ['libration', 'episodic', 'near_separatrix', 'circulation', 'drift', 'insufficient_time', 'undefined']
# Old bucket (labels.csv) -> the statuses of the new model it is compatible with, the first the one it maps to
# (DECISIONS.md: slow circulation is circulation (>= 1 turn) or insufficient_time (< 1 turn); chaotic became invalid;
# kinematic libration is non_resonant). The one table of buckets: `bucket_status`, `status_spelling`
BUCKET_STATUSES = {
    'resonant': ('resonant',),
    'transient': ('transient',),
    'non-resonant': ('non_resonant',),
    'near-separatrix': ('near_separatrix',),
    'slow-circulation': ('non_resonant', 'uncertain'),
    'kinematic-libration': ('non_resonant',),
    'insufficient-time': ('uncertain',),
    'uncertain': ('uncertain',),
    'chaotic': ('invalid',),
}

# Settings of the source run that the classifier, the gates and the plotters read. Passed back into
# SimulationConfig by `make_config`, so a reloaded example is classified as its run classified it.
CLASSIFIER_PREFIXES = (
    'oscillations_',
    'periodogram_',
    'libration_',
    'classify_',
    'secular_',
    'free_',
    'zlk_',
    'coherence_',
    'phase_portrait_',
)
CLASSIFIER_KEYS = ('filter',)

_SAFE = re.compile(r'[^A-Za-z0-9+\-.]')


# ------------------------------------------------------------------ ids


def bucket_status(bucket: str) -> str:
    """The status an old bucket maps to (the first of `BUCKET_STATUSES`); '' for no bucket or an unknown one."""
    return BUCKET_STATUSES.get(bucket, ('',))[0]


def status_spelling(bucket: str) -> str:
    """A bucket that names a status in the status' spelling ('non-resonant' -> 'non_resonant'); any other unchanged."""
    status = bucket.replace('-', '_')
    return status if status in STATUSES else bucket


def status_bucket(status: str) -> str:
    """A status in the buckets' spelling ('non_resonant' -> 'non-resonant')."""
    return status.replace('_', '-')


def safe_token(text: str) -> str:
    """Filesystem-safe token: spaces, brackets and anything outside [A-Za-z0-9+-.] become '_'."""
    return _SAFE.sub('_', str(text).strip())


def make_example_id(rtype: str, name: str, resonance: str, variant: Optional[str] = None) -> str:
    parts = [safe_token(rtype), safe_token(name), safe_token(resonance)]
    if variant:
        parts.append(safe_token(variant))
    return '__'.join(parts)


def planet_column(planet: str, field: str) -> str:
    """'<Planet>.<name>' column of planets.csv.gz for a planets_data key."""
    return f'{planet}.{PLANET_COLUMN_NAMES.get(field, field)}'


def rename_legacy_series(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={old: new for old, new in LEGACY_SERIES_COLUMNS.items() if old in df.columns and new not in df.columns})


def rename_legacy_planets(df: pd.DataFrame) -> pd.DataFrame:
    renames = {}
    for column in df.columns:
        for old, new in LEGACY_PLANET_SUFFIXES.items():
            if column.endswith(old) and column[: -len(old)] + new not in df.columns:
                renames[column] = column[: -len(old)] + new
    return df.rename(columns=renames)


def example_dir(example_id: str) -> Path:
    return EXAMPLES / example_id


def list_examples() -> List[str]:
    if not EXAMPLES.exists():
        return []
    return sorted(p.name for p in EXAMPLES.iterdir() if (p / 'meta.json').exists())


# ------------------------------------------------------------------ workers

WORKER_RAM_BYTES = 3 * 1024**3  # a reclassify worker peaks at ~2.6 GB (v2-w21a: 37087 3J+8S-3 at 1 Myr)
# with the low-memory allocator (resonances.config.low_memory_allocator) a worker's footprint peaks at 0.91 GB (37087 at
# 1 Myr; p90 0.29 GB) on all 719 examples with the W23 levers (memo, streamed owner columns, memory-mapped planets)
LOW_MEMORY_WORKER_RAM_BYTES = int(1.6 * 1024**3)
MAX_AUTO_WORKERS = 8
WORKERS_ENV = 'BENCHMARK_WORKERS'


def available_ram_bytes() -> Optional[int]:
    """Memory free for new processes: psutil's `available` when psutil is importable, else the free pages (os.sysconf,
    Linux), else the physical memory (sysctl hw.memsize, macOS); None when none of them answers."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - psutil is not a dependency of the package
        pass
    try:
        return int(os.sysconf('SC_AVPHYS_PAGES')) * int(os.sysconf('SC_PAGE_SIZE'))
    except (ValueError, OSError, AttributeError):
        pass
    try:
        return int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def auto_workers(cpus: Optional[int] = None, ram_bytes: Optional[int] = None, allocator: Optional[str] = None) -> int:
    """min(CPUs - 2, available RAM / per-worker RAM, MAX_AUTO_WORKERS), at least 1. The per-worker RAM is
    WORKER_RAM_BYTES, or LOW_MEMORY_WORKER_RAM_BYTES when the workers get the low-memory allocator: LOW_MEMORY_ALLOCATOR
    (`allocator`) on, or auto when the memory at WORKER_RAM_BYTES would limit the count below the CPUs."""
    cpus = cpus if cpus is not None else (os.cpu_count() or 2)
    ram_bytes = ram_bytes if ram_bytes is not None else available_ram_bytes()
    limits = [cpus - 2, MAX_AUTO_WORKERS]
    if ram_bytes is not None:
        per_worker = WORKER_RAM_BYTES
        if low_memory_allocator(ram_bytes // WORKER_RAM_BYTES < min(limits), allocator):
            per_worker = LOW_MEMORY_WORKER_RAM_BYTES
        limits.append(ram_bytes // per_worker)
    return max(1, int(min(limits)))


def low_memory_allocator(ram_limited: bool, allocator: Optional[str] = None) -> bool:
    """`resonances.config.low_memory_allocator` (imported here: bench_io does not load the package at import)."""
    from resonances.config import low_memory_allocator as policy

    return policy(ram_limited, allocator)


def workers_allocator(workers: int, ram_bytes: Optional[int] = None, allocator: Optional[str] = None) -> bool:
    """Whether a pool of `workers` gets the low-memory allocator: auto turns it on when they would need more than the
    available RAM at WORKER_RAM_BYTES each (`auto_workers` sized them for it, or the count was given)."""
    ram_bytes = ram_bytes if ram_bytes is not None else available_ram_bytes()
    return low_memory_allocator(ram_bytes is not None and workers * WORKER_RAM_BYTES > ram_bytes, allocator)


def resolve_workers(value: Optional[str | int] = None) -> int:
    """A worker count: an integer, or 'auto'/empty/None for `auto_workers`; None reads $BENCHMARK_WORKERS first."""
    if value is None:
        value = os.environ.get(WORKERS_ENV, '')
    text = str(value).strip().lower()
    if text in ('', 'auto'):
        return auto_workers()
    workers = int(text)
    if workers < 1:
        raise ValueError(f'workers must be >= 1 or auto, not {value!r}')
    return workers


# ------------------------------------------------------------------ small helpers


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def package_commit() -> Optional[str]:
    try:
        out = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
        commit = out.stdout.strip()
        dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--', 'resonances'], capture_output=True, text=True)
        return commit + ('+dirty(resonances/)' if dirty.stdout.strip() else '')
    except Exception:
        return None


def date_to_jd(date: datetime.datetime) -> float:
    epoch = datetime.datetime(1970, 1, 1)
    return 2440587.5 + (date.replace(tzinfo=None) - epoch).total_seconds() / 86400.0


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'not JSON serialisable: {type(value)}')


def _clean_nan(obj):
    """NaN is not JSON: replace float NaN/inf by None recursively."""
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(_clean_nan(data), f, indent=2, ensure_ascii=False, default=_json_default)
        f.write('\n')
    os.replace(tmp, path)


RUN_RECORD_LISTS = ('bodies', 'elements_sources')


def merge_run_record(path: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """Write the run record `info` to `path` (run.json); when a record is already there, its `bodies` and
    `elements_sources` are kept and joined with the new ones (sorted, no duplicates). A rerun adds a body to a run
    of other bodies; replacing the lists lost them (R6, 2026-10-19). Every other field is the new record's."""
    merged = dict(info)
    if path.exists():
        with open(path) as f:
            old = json.load(f)
        for key in RUN_RECORD_LISTS:
            merged[key] = sorted(set(old.get(key, [])) | set(info.get(key, [])))
    write_json(path, merged)
    return merged


def write_csv_gz(path: Path, df: pd.DataFrame) -> None:
    """Atomic gzip CSV with 12 significant digits (mtime 0, so equal data gives equal bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=6, mtime=0) as gz:
        gz.write(df.to_csv(index=False, float_format=FLOAT_FORMAT, na_rep='').encode())
    os.replace(tmp, path)


# ------------------------------------------------------------------ writing


def write_run(run_id: str, info: Dict[str, Any], planets: Optional[pd.DataFrame] = None) -> Path:
    """data/runs/<run_id>/run.json (+ planets.csv.gz: t_yr and '<Planet>.<field>' columns)."""
    run_dir = RUNS / run_id
    if planets is not None:
        if 't_yr' not in planets.columns:
            raise ValueError('planets table needs a t_yr column')
        planets = rename_legacy_planets(planets)
        write_csv_gz(run_dir / 'planets.csv.gz', planets)
        info = dict(info, planets_file='planets.csv.gz', planets_columns=list(planets.columns), planets_rows=len(planets))
    write_json(run_dir / 'run.json', info)
    return run_dir


def write_example(
    example_id: str,
    series: pd.DataFrame,
    meta: Dict[str, Any],
    planets_run_id: Optional[str] = None,
    planets: Optional[pd.DataFrame] = None,
    extra_files: Optional[Dict[str, Path]] = None,
) -> Path:
    """Write one example. `planets_run_id` links the run's planets.csv.gz; `planets` writes the example's own."""
    series = rename_legacy_series(series)
    missing = [c for c in BASE_COLUMNS if c not in series.columns]
    if missing:
        raise ValueError(f'{example_id}: series lacks {missing}')
    if meta.get('resonance_type') == 'secular':
        missing = [c for c in SECULAR_COLUMNS if c not in series.columns]
        if missing:
            raise ValueError(f'{example_id}: secular series lacks {missing}')
    t = series['t_yr'].to_numpy()
    if len(t) < 2 or not np.all(np.diff(t) != 0):
        raise ValueError(f'{example_id}: time column is degenerate')

    target = example_dir(example_id)
    target.mkdir(parents=True, exist_ok=True)
    order = BASE_COLUMNS + [c for c in series.columns if c not in BASE_COLUMNS]
    write_csv_gz(target / 'series.csv.gz', series[order])

    link = target / 'planets.csv.gz'
    if link.is_symlink() or link.exists():
        link.unlink()
    if planets is not None:
        write_csv_gz(link, rename_legacy_planets(planets))
        meta.setdefault('provenance', {})['planets'] = 'own file'
    elif planets_run_id is not None:
        source = RUNS / planets_run_id / 'planets.csv.gz'
        if not source.exists():
            raise FileNotFoundError(source)
        os.symlink(os.path.relpath(source, target), link)
        meta.setdefault('provenance', {})['planets'] = f'runs/{planets_run_id}/planets.csv.gz'

    for name, src in (extra_files or {}).items():
        dst = target / name
        if str(src).endswith('.gz') or not name.endswith('.gz'):
            dst.write_bytes(Path(src).read_bytes())
        else:
            write_csv_gz(dst, pd.read_csv(src))

    meta = dict(meta)
    meta['schema'] = SCHEMA_VERSION
    meta['example_id'] = example_id
    meta.setdefault('regime', {})['n_rows'] = int(len(series))
    meta['columns'] = order
    write_json(target / 'meta.json', meta)
    return target


# ------------------------------------------------------------------ reading


@dataclass
class Example:
    example_id: str
    path: Path
    meta: Dict[str, Any]
    series: pd.DataFrame
    planets: Optional[Dict[str, Dict[str, np.ndarray]]] = field(default=None)

    @property
    def resonance(self) -> str:
        return self.meta['resonance']

    @property
    def t_yr(self) -> np.ndarray:
        return self.series['t_yr'].to_numpy()


def read_meta(example_id: str) -> Dict[str, Any]:
    with open(example_dir(example_id) / 'meta.json') as f:
        return json.load(f)


def read_header(path: Path) -> List[str]:
    """The column names of a CSV file, gzipped ('.gz') or plain."""
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as f:
        return f.readline().strip().split(',')


def log(msg: str) -> None:
    print(f'[{datetime.datetime.now():%H:%M:%S}] {msg}', flush=True)


def none_or_str(value, empty: bool = False) -> Optional[str]:
    """A cell as text, None for a missing one (None, NaN; with `empty` also '')."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or (empty and value == ''):
        return None
    return str(value)


@contextlib.contextmanager
def labels_lock():
    """labels.csv is read-modified-written: one writer at a time (batches and imports that run together)."""
    import fcntl

    with open(LABELS.with_suffix('.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_planets(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """{planet: {'times', 'a', 'e', ..., 'l', 'varpi'}} — the shape of IntegrationEngine.planets_data."""
    df = rename_legacy_planets(pd.read_csv(path))
    field_of = {name: key for key, name in PLANET_COLUMN_NAMES.items()}
    out: Dict[str, Dict[str, np.ndarray]] = {}
    times = df['t_yr'].to_numpy(dtype=float)
    for column in df.columns:
        if column == 't_yr' or '.' not in column:
            continue
        planet, element = column.split('.', 1)
        out.setdefault(planet, {'times': times})[field_of.get(element, element)] = df[column].to_numpy(dtype=float)
    return out


def load_example(example_id: str, with_planets: bool = True) -> Example:
    path = example_dir(example_id)
    meta = read_meta(example_id)
    series = rename_legacy_series(pd.read_csv(path / 'series.csv.gz'))
    planets = None
    planets_path = path / 'planets.csv.gz'
    if with_planets and planets_path.exists():
        planets = read_planets(planets_path)
        n = len(next(iter(planets.values()))['times']) if planets else 0
        if n != len(series):
            raise ValueError(f'{example_id}: planets have {n} rows, the series {len(series)}')
    return Example(example_id, path, meta, series, planets)


# ------------------------------------------------------------------ package objects


def make_config(meta: Dict[str, Any], series: pd.DataFrame, has_planets: bool = False, **overrides):
    """SimulationConfig for this series: span and Nout from the saved times, classifier settings from
    the source run (`meta.classifier_config`), then `overrides` (e.g. a redesigned classifier's)."""
    from resonances.simulation.config import SimulationConfig

    t = series['t_yr'].to_numpy(dtype=float)
    span = float(t[-1] - t[0])
    kwargs: Dict[str, Any] = dict(save=None, plot=None, save_summary=False, _skip_path_verification=True)
    kwargs.update(meta.get('classifier_config') or {})
    if kwargs.get('secular_planet_phases') == 'measured' and not has_planets:
        kwargs['secular_planet_phases'] = 'linear'
    kwargs['save_planets'] = bool(has_planets)
    # The filter's sampling rate is Nout / |tmax|: both come from the saved grid.
    kwargs['tmax'] = span * 2 * np.pi
    kwargs['Nout'] = len(t)
    kwargs.update(overrides)
    return SimulationConfig(**kwargs)


def to_body(example: Example, config=None, secular_angle: str = 'reduced', **config_overrides):
    """(Body, resonance, sim) rebuilt from an example.

    The body carries the raw elements and the angle the package would classify; filtered series are
    recomputed here with `config` (by default the source run's settings). For a secular angle
    `secular_angle` picks what is classified: 'reduced' (the package default), 'raw' (unreduced) or 'src'
    (the source run's own angle). `sim` has `.times` (REBOUND units), `.config` and
    `.integration_engine.planets_data`, which is what `MMRDiagnostics.from_body` and friends read.
    """
    from resonances.body import Body
    from resonances.resonance.factory import create_resonance
    from resonances.resonance.filtering import filter_angle, wrap

    s, meta = example.series, example.meta
    if config is None:
        config = make_config(meta, s, has_planets=example.planets is not None, **config_overrides)

    times = s['t_yr'].to_numpy(dtype=float) * 2 * np.pi
    body = Body()
    body.name = str(meta['name'])
    body.times = times
    col = lambda c: s[c].to_numpy(dtype=float) if c in s.columns else None  # noqa: E731
    body.axis, body.ecc, body.inc = col('a'), col('e'), col('inc')
    body.Omega, body.omega, body.varpi = col('Omega'), col('omega'), col('varpi')
    body.M, body.longitude = col('M'), col('mean_longitude')
    body.axis_filtered = filter_angle(body.axis, config)

    resonance = create_resonance(meta['resonance'])
    key = resonance.to_s()
    if resonance.type == 'mmr':
        body.mmrs = [resonance]
    elif resonance.type == 'secular':
        body.secular_resonances = [resonance]
    else:
        body.lidov_kozai_resonances = [resonance]

    raw = col('angle_raw')
    if resonance.type == 'secular':
        reduced = col('angle_reduced')
        body.secular_angles_osculating[key] = wrap(raw)
        body.secular_angles_proper[key] = wrap(reduced)
        chosen = {'reduced': reduced, 'raw': raw, 'src': col('src_angle')}[secular_angle]
        unwrapped = np.unwrap(chosen)
    else:
        unwrapped = raw
    body.angles_unwrapped[key] = unwrapped
    body.angles[key] = wrap(unwrapped)
    body.angles_filtered_unwrapped[key] = filter_angle(unwrapped, config)
    body.angles_filtered[key] = wrap(body.angles_filtered_unwrapped[key])

    planets_data = example.planets or {}
    sim = SimpleNamespace(
        times=times,
        config=config,
        integration_engine=SimpleNamespace(planets_data=planets_data),
        bodies=[body],
    )
    return body, resonance, sim


# ------------------------------------------------------------------ reclassify runs (reclassify.py, reclassify_secular_zlk.py)


def select_ids(ids: List[str], only: Optional[str], rtypes: Optional[str], default: Iterable[str], what: str) -> List[str]:
    """The examples of a run: `only` (comma-separated ids, each one of `ids`), else the `ids` of the comma-separated
    `rtypes` (default `default`)."""
    if only:
        wanted = [s.strip() for s in only.split(',') if s.strip()]
        missing = [w for w in wanted if w not in ids]
        if missing:
            raise SystemExit(f'unknown {what} example ids: {missing}')
        return wanted
    allowed = set((rtypes or ','.join(default)).split(','))
    return [i for i in ids if i.split('__', 1)[0] in allowed]


def previous_timings(ids: Iterable[str], runs: Iterable[Path] = ()) -> Dict[str, float]:
    """example_id -> seconds from the newest run.json that timed at least half of `ids`: the files `runs` first, then the
    run folders under DATA, newest first; {} when none did."""
    ids = set(ids)
    found = sorted(DATA.glob('*/run.json'), key=lambda path: path.stat().st_mtime, reverse=True) if DATA.exists() else []
    for path in [Path(run) for run in runs if Path(run).exists()] + found:
        try:
            timings = json.loads(path.read_text()).get('timings') or []
        except (OSError, ValueError):
            continue
        seconds = {t['example_id']: float(t['seconds']) for t in timings if 'example_id' in t and 'seconds' in t}
        if len(ids & set(seconds)) * 2 >= len(ids):
            return seconds
    return {}


def longest_first(ids: List[str], runs: Iterable[Path] = ()) -> List[str]:
    """`ids` longest first by the previous run's timings (`previous_timings`; untimed ids first, ties in their order):
    the pool then ends with the short examples instead of waiting for a long one started last (v2-w22 timings, 6
    workers: makespan 275 -> 254 s). The results do not depend on the order (`run_examples` sorts them back)."""
    seconds = previous_timings(ids, runs)
    return sorted(ids, key=lambda example: -seconds.get(example, float('inf')))


def run_examples(
    ids: List[str], job: Callable, figures: bool, workers: int, tasks_per_worker: int, in_process: bool, runs: Iterable[Path] = ()
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """`job((example_id, figures))` over `ids` (longest first: `longest_first`, from the timings of `runs` or of the
    newest run), in this process or in a spawn pool of `workers` processes, each replaced after `tasks_per_worker`
    examples (0: never), with one progress line per example; the pool's workers get the low-memory allocator when
    `workers_allocator` says so. Returns (the ok results in the order of `ids`, the errors, the wall time [s])."""
    from resonances.config import worker_allocator

    start = time.perf_counter()
    results = []
    jobs = [(i, figures) for i in longest_first(ids, runs)]
    allocator = contextlib.ExitStack()
    if in_process:
        iterator = (job(args) for args in jobs)
        pool = None
    else:
        # the environment every worker inherits, also those that replace a recycled one: kept for the pool's life
        allocator.enter_context(worker_allocator(workers_allocator(workers)))
        pool = mp.get_context('spawn').Pool(max(workers, 1), maxtasksperchild=tasks_per_worker if tasks_per_worker > 0 else None)
        iterator = pool.imap_unordered(job, jobs)
    try:
        for n, res in enumerate(iterator, 1):
            results.append(res)
            if res['status'] == 'ok':
                row = res['summary'][0]
                line = f"{row['status']}/{row['angle_status']}/{row['confidence'] or '-'} {row['reason']}  {res['seconds']:.1f}s"
            else:
                line = res['error']
            print(f'[{n}/{len(jobs)}] {res["status"]:5s} {res["example_id"]}  {line}', flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        allocator.close()
    wall = time.perf_counter() - start
    order = {example: i for i, example in enumerate(ids)}
    ok = sorted((r for r in results if r['status'] == 'ok'), key=lambda r: order[r['example_id']])
    return ok, [r for r in results if r['status'] != 'ok'], wall


def write_run_record(out: Path, script: str, n_examples: int, ok, errors, wall: float, workers: int, figures: bool, timings, **extra):
    """run.json of a reclassify run; `extra` (e.g. tasks_per_worker) follows `workers`."""
    from resonances.resonance.analysis.settings import settings_from_config, settings_hash

    write_json(
        out / 'run.json',
        {
            'script': script,
            'package_commit': package_commit(),
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
            'settings_hash': settings_hash(settings_from_config()),
            'n_examples': n_examples,
            'n_ok': len(ok),
            'n_errors': len(errors),
            'wall_seconds': round(wall, 1),
            'workers': workers,
            **extra,
            'figures': figures,
            'timings': timings,
            'errors': [{'example_id': r['example_id'], 'error': r['error']} for r in errors],
        },
    )


def report_errors(ok, errors, wall: float) -> int:
    """The closing lines of a reclassify run, every error with its trace; the exit code (1 with an error)."""
    print(f'\nanalysed {len(ok)}, errors {len(errors)}; wall {wall:.0f} s')
    for r in errors:
        print('ERROR', r['example_id'], r['error'])
        print(r['trace'])
    return 1 if errors else 0


def save_figure(fig, path: Path, dpi: float) -> None:
    """Save through a hidden temporary file, so a reader never sees half a PNG."""
    tmp = path.with_name('.' + path.name + '.tmp.png')
    fig.savefig(tmp, dpi=dpi)
    os.replace(tmp, path)


# ------------------------------------------------------------------ labels and review tree


def read_summary(path: Path) -> pd.DataFrame:
    """A run's summary.csv as strings ('' for empty cells), angle_status, final_regime and reason in the current
    spelling: runs before 2026-10-17 wrote 'transient' for an episodic angle (`verdict.OLD_ANGLE_STATUS`), runs before
    2026-10-26 dynamic_* for the geometry alone (origin unknown, `verdict.reason_from_text`)."""
    from resonances.resonance.analysis.verdict import OLD_ANGLE_STATUS, reason_from_text

    frame = pd.read_csv(path, dtype=str).fillna('')
    old = {spelling: str(value) for spelling, value in OLD_ANGLE_STATUS.items()}
    for column in ('angle_status', 'final_regime'):
        if column in frame:
            frame[column] = frame[column].replace(old)
    if 'reason' in frame and 'libration_origin' in frame:
        frame['reason'] = [str(reason_from_text(r, o)) for r, o in zip(frame['reason'], frame['libration_origin'])]
    return frame


def read_labels() -> pd.DataFrame:
    if not LABELS.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    df = pd.read_csv(LABELS, dtype=str, keep_default_na=False)
    for c in LABEL_COLUMNS:
        if c not in df.columns:
            df[c] = ''
    return df[LABEL_COLUMNS + [c for c in df.columns if c not in LABEL_COLUMNS]]


def upsert_labels(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Add or update rows by example_id. Machine columns are refreshed; author_bucket, claude_bucket and
    comment are never overwritten once non-empty."""
    current = read_labels().set_index('example_id', drop=False)
    for row in rows:
        row = {k: ('' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) for k, v in row.items()}
        eid = row['example_id']
        if eid in current.index:
            for k, v in row.items():
                if k in HUMAN_LABEL_COLUMNS and current.at[eid, k]:
                    continue
                current.at[eid, k] = v
        else:
            new = {c: row.get(c, '') for c in current.columns}
            current.loc[eid] = new
    out = current.reset_index(drop=True).sort_values('example_id')
    tmp = LABELS.with_suffix('.csv.tmp')
    out.to_csv(tmp, index=False)
    os.replace(tmp, LABELS)
    return out


def build_review_tree(png_name: str = 'combined.png') -> int:
    """Regenerate data/review/<rtype>/<bucket>/<confidence>/<example_id>.png symlinks from labels.csv.
    The bucket is author_bucket when set, else proposed_bucket. Returns the number of links."""
    import shutil

    if REVIEW.exists():
        shutil.rmtree(REVIEW)
    n = 0
    for row in read_labels().itertuples(index=False):
        png = example_dir(row.example_id) / png_name
        if not png.exists():
            continue
        bucket = row.author_bucket or row.proposed_bucket or 'unlabelled'
        folder = REVIEW / safe_token(row.rtype) / safe_token(bucket) / safe_token(row.confidence or 'none')
        folder.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(png, folder), folder / f'{row.example_id}.png')
        n += 1
    return n
