"""Cut benchmark examples out of runs already on disk (no integration).

For every candidates.csv row with needs_integration != 'yes' and a data_path to a saved series, write
data/examples/<example_id>/ (series.csv.gz, meta.json, planets link) per ../LAYOUT.md, add the row to
labels.csv, and log every candidate that was not extracted, with the reason, to extract_report.csv.

    uv run python benchmark/tools/extract_existing.py [--only c0001,c0002] [--runs substr]

Secular angles are rebuilt with the current package from the body's elements at full resolution
(`build_proper_angle_series`, unreduced and reduced; measured planet phases when the run saved the
planets and used them), then the series is thinned (see `decimation_factor`). MMR and ZLK angles are the
source's unwrapped angles as saved.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as io  # noqa: E402

from resonances.data.const import (  # noqa: E402
    DAYS_IN_YEAR,
    INVARIABLE_PLANE,
    J2000,
    PLANETARY_PHASES,
    SOLAR_SYSTEM_WITH_MOON,
    SOLAR_SYSTEM_WITH_SUN,
)
from resonances.resonance.factory import create_resonance  # noqa: E402
from resonances.secular import planet_modes  # noqa: E402
from resonances.secular.proper_angle import calc_proper_angle_series  # noqa: E402

CANDIDATES = io.BENCH / 'candidates.csv'
REPORT = io.BENCH / 'extract_report.csv'
FIXTURE_DIR = io.REPO / 'tests' / 'fixtures' / 'free-omega'
V4_DIR = io.REPO / 'experiments/2026-09-binary-asteroids/output/secular-10myr-v4'

# Candidate rows whose resonance names several angles: each becomes its own example.
EXPANSIONS = {'z1/z2 angles': ['g-g6+s-s6', '2(g-g6)+(s-s6)']}
UNSPECIFIC = {'all secular angles'}

SECULAR_MAX_STEP_YR = 500.0  # thin 10-20 Myr secular series to at most this step...
SECULAR_SAMPLES_PER_PERIOD = 20  # ...and at least this many samples per shortest period

BODY_COLUMNS = {
    'a': 'a',
    'e': 'e',
    'inc': 'inc',
    'Omega': 'Omega',
    'omega': 'omega',
    'varpi': 'varpi',
    'M': 'M',
    'longitude': 'mean_longitude',
}

CREATED = datetime.datetime.now().isoformat(timespec='seconds')
EXTRACTOR_COMMIT = io.package_commit()


# ------------------------------------------------------------------ run descriptions


def read_json_head(path: Path, key: str = 'config') -> Optional[dict]:
    """The `key` object of a JSON file without parsing the rest (NEA simulation.json is 400 MB)."""
    with open(path) as f:
        head = f.read(4 << 20)
    marker = f'"{key}":'
    pos = head.find(marker)
    if pos < 0:
        return None
    start = head.index('{', pos)
    obj, _ = json.JSONDecoder().raw_decode(head[start:])
    return obj


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    if path.stat().st_size > 20 << 20:
        return {'config': read_json_head(path, 'config')}
    with open(path) as f:
        return json.load(f)


def parse_cfg_text(path: Path) -> dict:
    """The old text simulation.cfg (output/batch_mmr_test)."""
    out = {}
    for line in path.read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            out[k.strip()] = v.strip()
    cfg = {
        'name': out.get('Name'),
        'date': out.get('Date', '').replace(' ', 'T'),
        'source': out.get('Source'),
        'tmax': float(out['Tmax']) if 'Tmax' in out else None,
        'integrator': out.get('Integrator'),
        'dt': float(out['dt']) if 'dt' in out else None,
        'oscillations_cutoff': float(out['Cutoff']) if 'Cutoff' in out else None,
        'oscillations_filter_order': int(out['Filter order']) if 'Filter order' in out else None,
        'periodogram_frequency_min': float(out['Frequency min']) if 'Frequency min' in out else None,
        'periodogram_frequency_max': float(out['Frequency max']) if 'Frequency max' in out else None,
        'periodogram_soft': float(out['Soft']) if 'Soft' in out else None,
        'libration_period_critical': float(out['Period critical']) if 'Period critical' in out else None,
    }
    return {k: v for k, v in cfg.items() if v is not None}


@dataclass
class Run:
    run_id: str
    kind: str  # package | fixture | batch_mmr_test
    run_dir: Path
    config: dict
    run_json: Optional[dict]
    sim_json_path: Optional[Path]
    run_json_path: Optional[Path]
    summary_path: Optional[Path]
    commit: Optional[str]
    commit_note: str = ''
    planet_files: Dict[str, Path] = field(default_factory=dict)
    statuses: Dict[tuple, int] = field(default_factory=dict)
    regimes: Optional[pd.DataFrame] = None
    notes: str = ''

    # ---- regime
    def epoch(self) -> Optional[datetime.datetime]:
        if self.run_json and self.run_json.get('epoch'):
            return datetime.datetime.fromisoformat(str(self.run_json['epoch']))
        d = self.config.get('date')
        return datetime.datetime.fromisoformat(d) if d else None

    def epoch_jd(self) -> Optional[float]:
        if self.run_json and self.run_json.get('epoch_jd'):
            return float(self.run_json['epoch_jd'])
        e = self.epoch()
        return io.date_to_jd(e) if e else None

    def direction(self) -> str:
        args = (self.run_json or {}).get('args') or {}
        if args.get('direction'):
            return args['direction']
        tmax = self.config.get('tmax')
        return 'backward' if tmax is not None and float(tmax) < 0 else 'forward'

    def moon(self) -> bool:
        return bool(self.config.get('solar_system_moon'))

    def planets(self) -> List[str]:
        system = SOLAR_SYSTEM_WITH_MOON if self.moon() else SOLAR_SYSTEM_WITH_SUN
        return [p for p in system if p != 'Sun']

    def classifier_config(self) -> dict:
        out = {}
        for k, v in self.config.items():
            if k in ('tmax', 'tmax_yrs', 'Nout'):
                continue
            if k.startswith(io.CLASSIFIER_PREFIXES) or k in io.CLASSIFIER_KEYS:
                if isinstance(v, (int, float, str, bool, list)) or v is None:
                    out[k] = v
        return out

    def info(self) -> dict:
        """What goes into data/runs/<run_id>/run.json."""
        scalars = {k: v for k, v in self.config.items() if not isinstance(v, (dict,)) and not (isinstance(v, list) and len(v) > 50)}
        run_json = None
        if self.run_json:
            run_json = {k: v for k, v in self.run_json.items() if k not in ('preset',)}
        return {
            'run_id': self.run_id,
            'kind': self.kind,
            'source_dir': str(self.run_dir),
            'simulation_json': str(self.sim_json_path) if self.sim_json_path else None,
            'run_json_path': str(self.run_json_path) if self.run_json_path else None,
            'package_commit': self.commit,
            'commit_note': self.commit_note,
            'epoch_date': self.epoch().isoformat() if self.epoch() else None,
            'epoch_jd': self.epoch_jd(),
            'direction': self.direction(),
            'planets': self.planets(),
            'moon': self.moon(),
            'config': scalars,
            'run_json': run_json,
            'notes': self.notes,
            'created': CREATED,
        }


_RUNS: Dict[str, Run] = {}


def _commit_from(run_json: Optional[dict]) -> Optional[str]:
    if not run_json:
        return None
    return run_json.get('git_commit') or run_json.get('commit')


def _timing_start(sim: Optional[dict]) -> Optional[str]:
    try:
        return sim['timing']['running_time']['start']
    except Exception:
        return None


def resolve_data_path(text: str) -> Path:
    """A candidates.csv data_path: relative to the repository, or under an external disk written `<external>/...`
    (set RESONANCES_EXTERNAL_DATA to its mount point; unset, the path does not exist and the row is reported)."""
    if text.startswith('<external>/'):
        return Path(os.environ.get('RESONANCES_EXTERNAL_DATA', '/nonexistent')) / text[len('<external>/') :]
    p = Path(text)
    return p if p.is_absolute() else io.REPO / p


def resolve_run(data_path: Path, name: str) -> Run:
    p = data_path
    if p.parent == FIXTURE_DIR:
        key = 'fixture-free-omega'
        if key not in _RUNS:
            _RUNS[key] = Run(
                run_id=key,
                kind='fixture',
                run_dir=FIXTURE_DIR,
                config={'integrator': 'mercurius', 'tmax': 5e5 * 2 * np.pi},
                run_json=None,
                sim_json_path=None,
                run_json_path=None,
                summary_path=None,
                commit='0df809c1a78e',
                commit_note='commit that added the fixtures; the integrations themselves live outside the repository',
                notes='tests/fixtures/free-omega: first 500 kyr of mercurius integrations thinned to 100 yr '
                '(tests/resonances/resonance/test_omega_free_gate_reference.py). Epoch and element source not recorded.',
            )
        return _RUNS[key]

    if p.parent.name == 'batch_mmr_test':
        key = 'batch_mmr_test'
        if key not in _RUNS:
            cfg = parse_cfg_text(p.parent / 'simulation.cfg')
            _RUNS[key] = Run(
                run_id=key,
                kind='batch_mmr_test',
                run_dir=p.parent,
                config=cfg,
                run_json=None,
                sim_json_path=p.parent / 'simulation.cfg',
                run_json_path=None,
                summary_path=p.parent / 'summary.csv',
                commit=None,
                commit_note='old package (run dated 2025-12-20); angle/angle_osculating/angle_proper had the old meaning',
            )
        return _RUNS[key]

    run_dir = p.parent
    if run_dir == V4_DIR / 'data':
        # v4 merged dir: pair 12 (9068, 455327) came from the ias15 rerun, everything else from SABA.
        ias_sim = read_json(V4_DIR / 'simulation-ias15-pair12.json') or {}
        ias_bodies = {str(b['name']) for b in (ias_sim.get('simulation') or {}).get('bodies', [])} or {'9068', '455327'}
        variant = 'ias15-pair12' if name in ias_bodies else 'saba'
        key = f'secular-10myr-v4-{variant}'
        if key not in _RUNS:
            sim_path = V4_DIR / f'simulation-{variant}.json'
            run_path = V4_DIR / f'run-{variant}.json'
            sim = read_json(sim_path)
            rj = read_json(run_path)
            _RUNS[key] = Run(
                run_id=key,
                kind='package',
                run_dir=run_dir,
                config=sim['config'],
                run_json=rj,
                sim_json_path=sim_path,
                run_json_path=run_path,
                summary_path=V4_DIR / 'summary.csv',
                commit=_commit_from(rj),
            )
        return _RUNS[key]

    if 'lkr-ias15' in run_dir.parts:
        key = f'lkr-ias15-{run_dir.name}'
    elif run_dir.name == 'pair74':
        key = f'{run_dir.parent.name}-pair74'
    elif run_dir.name.startswith('production-'):
        key = f'nea-{run_dir.name}'
    else:
        key = run_dir.name
    if key in _RUNS:
        return _RUNS[key]

    sim_path = run_dir / 'simulation.json'
    run_path = run_dir / 'run.json'
    sim = read_json(sim_path)
    if sim is None or not sim.get('config'):
        raise RuntimeError(f'no simulation.json in {run_dir}')
    rj = read_json(run_path)
    commit = _commit_from(rj)
    note = ''
    if commit is None:
        note = f'no run.json with a commit; simulation started {_timing_start(sim)}'
    # data-planet-*.csv at the top of the run (runs before the package's data folder) or in data/ (DATA_FOLDER)
    planet_paths = sorted(run_dir.glob('data-planet-*.csv')) + sorted(run_dir.glob('data/data-planet-*.csv'))
    planet_files = {f.name[len('data-planet-') : -4]: f for f in planet_paths}
    run = Run(
        run_id=key,
        kind='package',
        run_dir=run_dir,
        config=sim['config'],
        run_json=rj,
        sim_json_path=sim_path,
        run_json_path=run_path if rj else None,
        summary_path=run_dir / 'summary.csv' if (run_dir / 'summary.csv').exists() else None,
        commit=commit,
        commit_note=note,
        planet_files=planet_files,
    )
    if (run_dir / 'regimes_summary.csv').exists():
        run.regimes = pd.read_csv(run_dir / 'regimes_summary.csv', dtype={'key': str})
    _RUNS[key] = run
    return run


def load_statuses(run: Run, names: set) -> None:
    """Status of each (name, resonance) in the source run's summary.csv."""
    if run.summary_path is None or not run.summary_path.exists():
        return
    # a v2 summary.csv has the status as a string and the old integer as `status_code`
    header = pd.read_csv(run.summary_path, nrows=0).columns
    status_column = 'status_code' if 'status_code' in header else 'status'
    usecols = ['name', 'resonance', status_column]
    for chunk in pd.read_csv(run.summary_path, usecols=usecols, dtype={'name': str, 'resonance': str}, chunksize=2_000_000):
        chunk = chunk[chunk['name'].isin(names)]
        for r in chunk.itertuples(index=False):
            try:
                run.statuses[(r.name, r.resonance)] = int(getattr(r, status_column))
            except (TypeError, ValueError):
                pass


# ------------------------------------------------------------------ reading source series


def read_source(path: Path, keys: List[str], run: Run) -> pd.DataFrame:
    header = io.read_header(path)
    if run.kind == 'batch_mmr_test':
        wanted = ['times', 'angle', 'angle_osculating', 'angle_proper']
    else:
        wanted = ['times']
        for k in keys:
            wanted += [f'{k}_angle_unwrapped', f'{k}_angle', f'{k}_angle_osculating', f'{k}_angle_proper']
    wanted += list(BODY_COLUMNS)
    usecols = [c for c in header if c in set(wanted)]
    return pd.read_csv(path, usecols=usecols)


def times_in_years(df: pd.DataFrame, run: Run) -> tuple[np.ndarray, str]:
    """Saved `times` are years in every package CSV (data_manager divides by 2*pi once); check it."""
    t = df['times'].to_numpy(dtype=float)
    tmax = run.config.get('tmax')
    if tmax is None or run.kind == 'fixture':
        return t, 'years (no tmax to check against)'
    expected = abs(float(tmax)) / (2 * np.pi)
    span = abs(t[-1] - t[0])
    ratio = span / expected if expected else np.nan
    if abs(ratio - 1) < 0.01:
        return t, f'years (span {span:.6g} vs tmax/2pi {expected:.6g})'
    if abs(ratio - 2 * np.pi) < 0.05:
        return t / (2 * np.pi), f'converted from 2pi-yr units (span {span:.6g} vs tmax/2pi {expected:.6g})'
    raise ValueError(f'times span {span} does not match tmax/2pi {expected} (ratio {ratio:.4f})')


def circ_diff_deg(x, y) -> Dict[str, float]:
    d = np.degrees(np.abs(np.angle(np.exp(1j * (np.asarray(x, float) - np.asarray(y, float))))))
    d = d[np.isfinite(d)]
    if not len(d):
        return {'max_deg': None, 'p99_deg': None, 'median_deg': None}
    return {'max_deg': float(d.max()), 'p99_deg': float(np.percentile(d, 99)), 'median_deg': float(np.median(d))}


def mean_period(x_unwrapped: np.ndarray, t: np.ndarray) -> float:
    rate = abs(x_unwrapped[-1] - x_unwrapped[0]) / abs(t[-1] - t[0])
    return float(2 * np.pi / rate) if rate > 0 else float('inf')


def decimation_factor(t: np.ndarray, series: Dict[str, np.ndarray]) -> tuple[int, dict]:
    """k such that the step k*dt is <= 500 yr and <= 1/20 of the shortest mean period among varpi,
    Omega and the angles. 1 for anything shorter than 1 Myr."""
    step = float(np.median(np.diff(t)))
    span = abs(t[-1] - t[0])
    periods = {name: mean_period(np.unwrap(v), t) for name, v in series.items() if v is not None and np.all(np.isfinite(v))}
    shortest = min(periods.values()) if periods else float('inf')
    if span < 1e6:
        k = 1
    else:
        k = max(1, int(min(SECULAR_MAX_STEP_YR, shortest / SECULAR_SAMPLES_PER_PERIOD) // abs(step)))
    return k, {
        'factor': k,
        'source_step_yr': step,
        'step_yr': step * k,
        'shortest_period_yr': shortest,
        'periods_yr': periods,
        'method': 'every k-th row, no smoothing',
        'rule': f'step <= {SECULAR_MAX_STEP_YR:g} yr and <= shortest/{SECULAR_SAMPLES_PER_PERIOD}; none below 1 Myr',
    }


# ------------------------------------------------------------------ planets


def read_planets_full(run: Run, planets: List[str]) -> Optional[pd.DataFrame]:
    if not run.planet_files:
        return None
    frames = []
    t = None
    for planet in planets:
        f = run.planet_files.get(planet)
        if f is None:
            continue
        df = pd.read_csv(f)
        if t is None:
            t = df['times'].to_numpy(dtype=float)
        frames.append(pd.DataFrame({io.planet_column(planet, c): df[c].to_numpy() for c in io.PLANET_FIELDS if c in df.columns}))
    if not frames:
        return None
    out = pd.concat([pd.DataFrame({'t_yr': t})] + frames, axis=1)
    return out


def planets_data_from_frame(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    t = df['t_yr'].to_numpy(dtype=float)
    for c in df.columns:
        if '.' in c:
            planet, element = c.split('.', 1)
            element = {name: key for key, name in io.PLANET_COLUMN_NAMES.items()}.get(element, element)
            out.setdefault(planet, {'times': t})[element] = df[c].to_numpy(dtype=float)
    return out


# ------------------------------------------------------------------ building one example


def base_series(df: pd.DataFrame, t: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(t)
    out = {'t_yr': t}
    for src, dst in BODY_COLUMNS.items():
        out[dst] = df[src].to_numpy(dtype=float) if src in df.columns else np.full(n, np.nan)
    return out


def secular_angles(cols: Dict[str, np.ndarray], key: str, run: Run, planets_frame: Optional[pd.DataFrame]):
    """Unreduced and reduced angles with the current package, at full resolution."""
    resonance = create_resonance(key)
    epoch = run.epoch()
    epoch_years = (epoch - J2000).total_seconds() / 86400.0 / DAYS_IN_YEAR if epoch else 0.0
    longitudes = None
    phases = 'linear'
    if run.config.get('secular_planet_phases') == 'measured' and planets_frame is not None:
        labels = {f'{t.mode}{t.index}' for t in resonance.formula.terms if t.index is not None}
        window = float(run.config.get('secular_phase_window') or planet_modes.DEFAULT_WINDOW)
        longitudes = planet_modes.measured_longitudes(cols['t_yr'], planets_data_from_frame(planets_frame), labels, window=window) or None
        phases = 'measured' if longitudes else 'linear (baseline shorter than the phase window)'
    common = dict(
        times=cols['t_yr'] * 2 * np.pi,
        omega=cols['omega'],
        Omega=cols['Omega'],
        ecc=cols['e'],
        inc=cols['inc'],
        resonance=resonance,
        planetary_phases=PLANETARY_PHASES,
        epoch_years=epoch_years,
        invariable_plane=INVARIABLE_PLANE,
        planetary_longitudes=longitudes,
    )
    raw = np.unwrap(calc_proper_angle_series(**common))
    reduced = np.unwrap(calc_proper_angle_series(**common, remove_forced=True))
    return raw, reduced, phases, epoch_years


def source_angle(df: pd.DataFrame, key: str, run: Run) -> tuple[Optional[np.ndarray], Optional[str]]:
    if run.kind == 'batch_mmr_test':
        return (np.unwrap(df['angle'].to_numpy(dtype=float)), 'angle') if 'angle' in df.columns else (None, None)
    for col in (f'{key}_angle_unwrapped',):
        if col in df.columns:
            return df[col].to_numpy(dtype=float), col
    col = f'{key}_angle'
    if col in df.columns:
        return np.unwrap(df[col].to_numpy(dtype=float)), col + ' (np.unwrap applied)'
    return None, None


def old_semantics(run: Run) -> str:
    c = run.commit or ''
    if run.kind == 'batch_mmr_test':
        return 'old package (2025-12): `angle` = classified angle, built from the planets\' osculating elements'
    if c.startswith(('d0c31b4', 'fa23598')):
        return f'package {c[:7]}: reduced angle against linear mode longitudes (classified in proper mode)'
    return (
        'pre-d0c31b4 package: `_angle_unwrapped` = old "proper" angle (planets\' osculating elements, '
        '700 kyr low-pass); see angles.check for how far it is from the current one'
    )


def build_example(row, key: str, run: Run, df: pd.DataFrame, data_path: Path, planets_frame: Optional[pd.DataFrame]):
    t, tcheck = times_in_years(df, run)
    cols = base_series(df, t)
    derived = []
    if np.all(np.isnan(cols['varpi'])) and not np.all(np.isnan(cols['omega'])):
        cols['varpi'] = cols['Omega'] + cols['omega']
        derived.append('varpi = Omega + omega (not in source)')
    missing = [c for c in ('M', 'mean_longitude') if np.all(np.isnan(cols[c]))]
    if missing:
        derived.append(f'{", ".join(missing)} not in source (left empty)')

    resonance = create_resonance(key)
    rtype_pkg = resonance.type
    src, src_col = source_angle(df, key, run)
    if src is None:
        raise LookupError(f'no angle column for {key}')
    source_columns = {'t_yr': 'times', **{v: k for k, v in BODY_COLUMNS.items() if k in df.columns}}

    angles_meta: Dict[str, Any] = {}
    decimation = None
    k = 1
    phases = None
    if rtype_pkg == 'secular':
        raw, reduced, phases, epoch_years = secular_angles(cols, key, run, planets_frame)
        cols['angle_raw'] = raw
        cols['angle_reduced'] = reduced
        cols['src_angle'] = src
        source_columns['src_angle'] = src_col
        osc_col = 'angle_osculating' if run.kind == 'batch_mmr_test' else f'{key}_angle_osculating'
        check = {'reduced_vs_src_angle': circ_diff_deg(reduced, src)}
        if osc_col in df.columns:
            check['raw_vs_src_osculating'] = circ_diff_deg(raw, df[osc_col].to_numpy(dtype=float))
        angles_meta = {
            'angle_raw': 'body osculating varpi/Omega (Omega from the invariable plane) against planet mode longitudes; '
            'not reduced; current package (calc_proper_angle_series, remove_forced=False)',
            'angle_reduced': 'same with the forced terms not in the angle removed (remove_forced=True): what the package classifies',
            'src_angle': f'source column {src_col}: {old_semantics(run)}',
            'phases': phases,
            'epoch_years_from_J2000': epoch_years,
            'computed_with_commit': EXTRACTOR_COMMIT,
            'check': check,
        }
        k, decimation = decimation_factor(t, {'varpi': cols['varpi'], 'Omega': cols['Omega'], 'angle_raw': raw, 'angle_reduced': reduced})
    else:
        cols['angle_raw'] = src
        source_columns['angle_raw'] = src_col
        angles_meta = {'angle_raw': f'source column {src_col}: unwrapped resonant angle as integrated, no filter'}

    series = pd.DataFrame(cols)
    if k > 1:
        series = series.iloc[::k].reset_index(drop=True)

    for c in ('a', 'e', 'angle_raw'):
        if series[c].isna().any():
            raise ValueError(f'NaN in {c}')
    tt = series['t_yr'].to_numpy()
    dt = np.diff(tt)
    if not (np.all(dt > 0) or np.all(dt < 0)):
        raise ValueError('time column is not monotonic')

    cfg = run.config
    name = str(row.name)
    meta = {
        'candidate_id': row.id,
        'name': name,
        'resonance': key,
        'candidate_resonance': row.resonance,
        'rtype': row.rtype,
        'resonance_type': rtype_pkg,
        'regime': {
            'integrator': cfg.get('integrator'),
            'dt': cfg.get('dt'),
            'span_yr': float(tt[-1] - tt[0]),
            'step_yr': float(np.median(dt)),
            'direction': run.direction(),
            'epoch_jd': run.epoch_jd(),
            'epoch_date': run.epoch().isoformat() if run.epoch() else None,
            'elements_source': cfg.get('source'),
            'planets': run.planets(),
            'moon': run.moon(),
            'source_Nout': cfg.get('Nout'),
            'secular_phases': phases,
            'decimation': decimation,
            'times_check': tcheck,
        },
        'provenance': {
            'run_id': run.run_id,
            'source_files': [{'path': str(data_path), 'size': data_path.stat().st_size, 'sha256': None}],
            'source_columns': source_columns,
            'derived': derived,
            'source_package_commit': run.commit,
            'source_commit_note': run.commit_note,
            'extracted': {'script': 'benchmark/tools/extract_existing.py', 'package_commit': EXTRACTOR_COMMIT, 'created': CREATED},
        },
        'labels': {
            'old_statuses': io.none_or_str(row.old_statuses),
            'author_verdict': io.none_or_str(row.author_verdict),
            'reliability': io.none_or_str(row.best_reliability),
            'sources': io.none_or_str(row.sources),
            'tags': [x for x in (io.none_or_str(row.tags) or '').split('|') if x],
            'proposed_bucket': io.none_or_str(row.proposed_bucket),
            'confidence': io.none_or_str(row.confidence),
            'needs_integration': io.none_or_str(row.needs_integration),
            'source_status': run.statuses.get((name, key)),
        },
        'angles': angles_meta,
        'classifier_config': run.classifier_config(),
        'source_extra': {},
        'notes': io.none_or_str(row.notes),
        'data_desc': io.none_or_str(row.data_desc),
    }
    if run.regimes is not None:
        m = run.regimes[(run.regimes['key'] == name) & (run.regimes['resonance'] == key)]
        if len(m):
            meta['source_extra']['coorbital_regime'] = {kk: _py(v) for kk, v in m.iloc[0].to_dict().items()}
    return series, meta, k


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


# ------------------------------------------------------------------ main loop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', help='comma-separated candidate ids')
    ap.add_argument('--runs', help='only data paths containing this substring')
    ap.add_argument('--no-sha', action='store_true', help='skip sha256 of source files')
    args = ap.parse_args()

    cand = pd.read_csv(CANDIDATES, dtype={'name': str})
    if args.only:
        cand = cand[cand['id'].isin(args.only.split(','))]
    report: List[dict] = []
    todo = []
    for row in cand.itertuples(index=False):
        base = dict(candidate_id=row.id, name=row.name, resonance=row.resonance, rtype=row.rtype, data_path=io.none_or_str(row.data_path))
        if row.needs_integration == 'yes':
            report.append(dict(base, status='skipped', reason='needs_integration=yes (batches A-E)'))
            continue
        if str(row.rtype).startswith(('group:', 'model')):
            report.append(dict(base, status='skipped', reason=f'{row.rtype} row: not a single body x resonance'))
            continue
        if not isinstance(row.data_path, str) or not row.data_path:
            report.append(dict(base, status='failed', reason='no data_path'))
            continue
        if args.runs and args.runs not in row.data_path:
            continue
        p = resolve_data_path(row.data_path)
        if not p.is_file():
            report.append(dict(base, status='failed', reason=f'data_path is not a file ({"missing" if not p.exists() else "directory"})'))
            continue
        if row.resonance in UNSPECIFIC:
            report.append(dict(base, status='failed', reason=f'resonance "{row.resonance}" names no single angle'))
            continue
        keys = EXPANSIONS.get(row.resonance, [row.resonance])
        todo.append((row, p, keys, base))

    # group by run, then by file
    by_run: Dict[str, list] = {}
    for item in todo:
        row, p, keys, base = item
        try:
            run = resolve_run(p, str(row.name))
        except Exception as exc:
            report.append(dict(base, status='failed', reason=f'cannot read run settings: {exc}'))
            continue
        by_run.setdefault(run.run_id, []).append(item)

    labels = []
    seen_ids: Dict[str, tuple] = {}
    sha_cache: Dict[Path, str] = {}
    t_start = time.time()
    for run_id, items in by_run.items():
        run = _RUNS[run_id]
        io.log(f'run {run_id}: {len(items)} candidate rows')
        load_statuses(run, {str(r.name) for r, *_ in items})

        # planets: MMR runs keep the planets the examples' resonances use; secular runs keep all
        planets_needed = set()
        for _row, _path, keys, _base in items:
            for k in keys:
                try:
                    r = create_resonance(k)
                    planets_needed |= set(getattr(r, 'planets_names', []) or []) if r.type == 'mmr' else set(run.planet_files)
                except Exception:
                    pass
        existing = io.RUNS / run_id / 'planets.csv.gz'
        if existing.exists():  # never drop planets other examples of this run already link to
            planets_needed |= {c.split('.', 1)[0] for c in io.read_header(existing) if '.' in c}
        planets_frame_full = read_planets_full(run, [pl for pl in run.planets() if pl in planets_needed]) if planets_needed else None
        run_planet_factor = None
        planets_written = False

        files: Dict[Path, list] = {}
        for item in items:
            files.setdefault(item[1], []).append(item)
        for path, file_items in files.items():
            keys_all = sorted({k for _, _, keys, _ in file_items for k in keys})
            t0 = time.time()
            try:
                df = read_source(path, keys_all, run)
            except Exception as exc:
                for _row, _path, _keys, base in file_items:
                    report.append(dict(base, status='failed', reason=f'cannot read source: {exc}'))
                continue
            if planets_frame_full is not None and len(planets_frame_full) != len(df):
                io.log(f'  planets of {run_id} have {len(planets_frame_full)} rows, {path.name} {len(df)}: planets dropped')
                planets_frame_full = None
            if not args.no_sha:
                sha_cache.setdefault(path, io.sha256_of(path))
            for row, _path, keys, base in file_items:
                for key in keys:
                    variant = None
                    eid = io.make_example_id(row.rtype, row.name, key, variant)
                    if eid in seen_ids and seen_ids[eid][1] == path:
                        report.append(dict(base, status='skipped', reason=f'same series as {seen_ids[eid][0]} ({key})', example_id=eid))
                        continue
                    if eid in seen_ids:
                        variant = run_id
                        eid = io.make_example_id(row.rtype, row.name, key, variant)
                    try:
                        series, meta, k = build_example(row, key, run, df, path, planets_frame_full)
                        meta['provenance']['source_files'][0]['sha256'] = sha_cache.get(path)
                        planets_kw: Dict[str, Any] = {}
                        if planets_frame_full is not None:
                            if run_planet_factor is None:
                                run_planet_factor = k
                            if not planets_written:
                                pf = planets_frame_full.iloc[::run_planet_factor].reset_index(drop=True)
                                io.write_run(run_id, dict(run.info(), planets_decimation=run_planet_factor), pf)
                                planets_written = True
                            if k == run_planet_factor:
                                planets_kw['planets_run_id'] = run_id
                            else:
                                planets_kw['planets'] = planets_frame_full.iloc[::k].reset_index(drop=True)
                        io.write_example(eid, series, meta, **planets_kw)
                        seen_ids[eid] = (row.id, path)
                        labels.append(
                            dict(
                                example_id=eid,
                                candidate_id=row.id,
                                rtype=row.rtype,
                                proposed_bucket=io.none_or_str(row.proposed_bucket),
                                confidence=io.none_or_str(row.confidence),
                            )
                        )
                        report.append(
                            dict(base, status='extracted', reason='' if key == row.resonance else f'expanded to {key}', example_id=eid)
                        )
                    except Exception as exc:
                        reason = f'{type(exc).__name__}: {exc}'
                        report.append(dict(base, status='failed', reason=reason, example_id=eid))
                        io.log(f'  FAILED {row.id} {row.name} {key}: {reason}')
                        if not isinstance(exc, (LookupError, ValueError)):
                            traceback.print_exc()
            io.log(f'  {path.name}: {len(file_items)} rows, {len(df)} lines, {time.time() - t0:.1f}s')
        if not planets_written:
            io.write_run(run_id, run.info())

    io.upsert_labels(labels)
    rep = pd.DataFrame(report, columns=['candidate_id', 'name', 'resonance', 'rtype', 'status', 'reason', 'example_id', 'data_path'])
    rep = rep.sort_values(['status', 'candidate_id'])
    if args.only or args.runs:
        if REPORT.exists():
            old = pd.read_csv(REPORT, dtype=str)
            old = old[~old['candidate_id'].isin(rep['candidate_id'])]
            rep = pd.concat([old, rep]).sort_values(['status', 'candidate_id'])
    rep.to_csv(REPORT, index=False)
    io.log(f'done in {time.time() - t_start:.0f}s: ' + ', '.join(f'{k} {v}' for k, v in rep['status'].value_counts().items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
