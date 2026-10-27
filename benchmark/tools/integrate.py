"""Integrate the benchmark candidates of batches A-D and write examples in the layout of ../LAYOUT.md.

    uv run python benchmark/tools/integrate.py --batch A --plan
    uv run python benchmark/tools/integrate.py --batch A --n-cores 6
    uv run python benchmark/tools/integrate.py --batch D --verify
    uv run python benchmark/tools/integrate.py --batch C --dry-run --years 2000 --only "2025 ON9,524522"

Batches (rows of candidates.csv with needs_integration yes/partial; see INVENTORY.md and RUN.md):
  A  main belt, TNOs, Jupiter Trojans: SABA(10,6,4) dt 1, 100 kyr, AstDyS osculating elements at the catalog epoch,
     output step 0.5 yr (P < 4.2 yr) or 1 yr, planets saved (FAIR). NEAs found among them (q < 1.3 au) go to B.
  B  NEAs (MPC neam00 at the NEA survey epoch, as experiments/nea-mmrs/sim.py), the 88710 clones of
     benchmark/clones.json (epoch MJD 58800) and three asteroid-pair bodies (AstDyS): ias15, 100 kyr
     (501710: 500 kyr), step from the orbital period, planets saved.
  C  terrestrial co-orbitals: ias15 + Moon, 20 kyr, step 0.1 yr, MPC/JPL elements of
     experiments/nea-mmrs/coorbitals/data/sample_coorbitals.parquet (JD 2461319.5), as coorbitals/run_20kyr.sh.
  D  Lidov-Kozai: 500 kyr, step 10 yr; SABA dt 0.5 for q >= 1.3 au, ias15 for planet-crossers; plus the free-omega
     gate re-checks of the NEA survey (2023 RS2, 2010 HY22, 2020 PP1; MPC elements; their 1:1 Earth angle too).

Initial conditions: every body is added as an element dict whose epoch is asserted equal to the simulation date
(`Simulation(date=epoch)`); the planets are taken from Horizons at that date (package cache/solar-<ts>.bin). Nothing
calls `finder.find` or lets the package pick a date. Bodies of one regime and epoch form one run
(`data/runs/<run_id>/`). SABA runs put several bodies in one Simulation (test particles do not change a fixed-step
integration); ias15 runs integrate every body alone, so a body's orbit never depends on its batch companions.

Execution: tasks (one Simulation each) run in spawned worker processes, one task per process. A body is done when
`runs/<batch>/done/<body>.json` exists; a restart skips done bodies (`--force` redoes them). Examples are written
as soon as a task finishes. The package's own output (summary.csv, segments.csv, simulation.json, data-*.csv
gzipped) stays in `runs/<batch>/<run_id>/<task>/` for provenance. Only the main process writes labels.csv.
"""

from __future__ import annotations

import os

# One BLAS thread per process: a spinning OpenBLAS pool stalled an earlier run for 17 h. Set before numpy loads,
# here and (since spawn re-imports this module) in every worker.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import datetime  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as io  # noqa: E402

REPO = io.REPO
BENCH = io.BENCH
CANDIDATES = BENCH / 'candidates.csv'
RUNS_ROOT = BENCH / 'runs'
DRY_ROOT = BENCH / 'dryrun'
CLONES = REPO / 'benchmark/clones.json'
NEA_SAMPLE = REPO / 'experiments/nea-mmrs/data/sample.parquet'  # MPC neam00 2026-09-15, epoch JD 2461297.5
COORB_SAMPLE = REPO / 'experiments/nea-mmrs/coorbitals/data/sample_coorbitals.parquet'  # epoch JD 2461319.5
JPL_COMPARE = REPO / 'experiments/nea-mmrs/coorbitals/research/input_check/jpl_compare.csv'

SCRIPT = 'benchmark/tools/integrate.py'

# experiments/nea-mmrs/config.py CLASSIFIER_NEA (cadence is set per regime here): the terrestrial-planet tuning of
# the NEA survey and the co-orbital runs. The other batches keep the package defaults, as their source runs did.
CLASSIFIER_NEA = {'oscillations_cutoff': 0.01, 'periodogram_frequency_max': 0.02, 'libration_period_min': 50.0}

NEA_Q = io.NEA_Q
FAIR_FRACTION = 4.2  # output step <= P / 4.2 (FAIR warns above P/4)
STEPS = (1.0, 0.5, 0.25, 0.2, 0.1, 0.05)

# Bodies of batch B by name (INVENTORY.md "Что надо интегрировать"); NEAs found among batch A rows join them.
B_NEA = {'88710', '1980', '1995 HM', '2000 SF8', '1992 SZ', '2002 BG', '2003 TG2', '2004 YG1'}
B_PAIRS = {'21436', '26420', '501710'}
PAIR_REGIME = {'501710': {'span_yr': 500_000, 'step_yr': 2.0}}  # author: "extend time"; 200 kyr already extracted

# D extras: free-omega gate confirmations of the NEA survey (group row c0767), re-run at 500 kyr. Their 1:1 angle
# with Earth is integrated too: the gate's method does not apply to co-orbitals (docs/free-elements.md).
ZLK_RECHECK = ['2023 RS2', '2010 HY22', '2020 PP1']
ZLK_RECHECK_GROUP = 'c0767'
COORBITAL_CHECK_RESONANCE = '1E-1+0+0'
COORBITAL_CHECK_DA = 0.1  # au: the 1:1 Earth angle is added only this close to a = 1 au

# MPC neam00 elements known to be wrong (bodies discovered at ~20,000-km Earth passes; research/input_check):
# replaced by the JPL elements at the same epoch when such a body is in a batch.
ELEMENT_OVERRIDES = {'2026 SZ2', '2026 PC6'}

CREATED = datetime.datetime.now().isoformat(timespec='seconds')


# ------------------------------------------------------------------ helpers


def jd_to_datetime(jd: float) -> datetime.datetime:
    """Naive datetime of a JD, the calendar value of the epoch (experiments/nea-mmrs/config.mpc_epoch_to_datetime)."""
    from astropy.time import Time

    return Time(float(jd), format='jd').to_datetime()


def step_for_period(period_yr: float) -> float:
    for s in STEPS:
        if s <= period_yr / FAIR_FRACTION:
            return s
    return STEPS[-1]


def span_tag(span_yr: float) -> str:
    return f'{span_yr / 1000:g}kyr'


def step_tag(step_yr: float) -> str:
    return f'{step_yr:g}yr'


def mpc_row_to_elements(row) -> dict:
    """experiments/nea-mmrs/config.mpc_row_to_elements: degrees of the sample -> radians."""
    return {
        'a': float(row['a']),
        'e': float(row['e']),
        'inc': float(np.radians(row['i_deg'])),
        'Omega': float(np.radians(row['node_deg'])),
        'omega': float(np.radians(row['peri_deg'])),
        'M': float(np.radians(row['M_deg'])),
    }


def solar_cache_file(date: datetime.datetime, moon: bool) -> Path:
    """The package's planets cache for this date (IntegrationEngine._solar_system_filename, cwd = repo)."""
    stamp = int(date.timestamp())
    return REPO / 'cache' / (f'solar-moon-{stamp}.bin' if moon else f'solar-{stamp}.bin')


def sha1_of(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha1(path.read_bytes()).hexdigest()


def done_name(batch: str, key: str) -> str:
    return f'{batch}__{io.safe_token(key)}.json'


def upsert_labels_locked(rows: List[dict]) -> None:
    """labels.csv is read-modified-written: serialise batches that run at the same time."""
    with io.labels_lock():
        io.upsert_labels(rows)


def set_data_root(root: Optional[str]) -> None:
    """Point bench_io at another data root (dry runs never touch the real data/)."""
    if not root:
        return
    io.DATA = Path(root)
    io.EXAMPLES = io.DATA / 'examples'
    io.RUNS = io.DATA / 'runs'
    io.REVIEW = io.DATA / 'review'


# ------------------------------------------------------------------ element sources


class Elements:
    """Element dicts with their epoch (JD) and a description of where they come from."""

    def __init__(self):
        self._astdys = None
        self._nea = None
        self._coorb = None
        self._clones = None
        self._jpl = None

    def astdys(self, name: str):
        import astdys

        if self._astdys is None:
            astdys.set_type('osculating')
            self._astdys = astdys
        el = self._astdys.search(int(name) if str(name).isdigit() else name)
        if el is None:
            return None
        elements = {k: float(el[k]) for k in ('a', 'e', 'inc', 'Omega', 'omega', 'M')}
        return elements, float(el['epoch']) + 2400000.5, 'AstDyS allnum.cat osculating'

    def nea(self, name: str):
        if self._nea is None:
            self._nea = pd.read_parquet(NEA_SAMPLE)
            self._nea['key'] = self._nea['key'].astype(str).str.strip()
        m = self._nea[self._nea['key'] == name]
        if not len(m):
            return None
        row = m.iloc[0]
        return mpc_row_to_elements(row), float(row['epoch_jd']), f'MPC neam00 2026-09-15 ({NEA_SAMPLE.relative_to(REPO)})'

    def coorbital(self, name: str):
        if self._coorb is None:
            self._coorb = pd.read_parquet(COORB_SAMPLE)
            self._coorb['key'] = self._coorb['key'].astype(str).str.strip()
        m = self._coorb[self._coorb['key'] == name]
        if not len(m):
            return None
        row = m.iloc[0]
        epoch = float(row['epoch_jd'])
        if name in ELEMENT_OVERRIDES:
            return self.jpl_override(name, epoch)
        return mpc_row_to_elements(row), epoch, f'{row["element_source"]} ({COORB_SAMPLE.relative_to(REPO)})'

    def jpl_override(self, name: str, epoch: float):
        if self._jpl is None:
            self._jpl = pd.read_csv(JPL_COMPARE, dtype={'key': str})
        m = self._jpl[self._jpl['key'] == name]
        if not len(m):
            raise LookupError(f'{name}: MPC elements are wrong and no JPL elements in {JPL_COMPARE}')
        r = m.iloc[0]
        el = {
            'a': float(r['A']),
            'e': float(r['EC']),
            'inc': float(np.radians(r['IN'])),
            'Omega': float(np.radians(r['OM'])),
            'omega': float(np.radians(r['W'])),
            'M': float(np.radians(r['MA'])),
        }
        return el, epoch, f'JPL Horizons {r["jpl_soln"]} at JD {epoch} (MPC neam00 wrong; {JPL_COMPARE.relative_to(REPO)})'

    def clone(self, name: str):
        if self._clones is None:
            self._clones = json.load(open(CLONES))
        if name not in self._clones:
            return None
        d = self._clones[name]['initial_data']
        el = {k: float(d[k]) for k in ('a', 'e', 'inc', 'Omega', 'omega', 'M')}
        return el, float(d['epoch']) + 2400000.5, f'clone of 88710 ({CLONES.relative_to(REPO)})'


# ------------------------------------------------------------------ the plan


@dataclass
class ExampleSpec:
    example_id: str
    candidate_id: str
    resonance: str
    rtype: str
    needs_integration: str
    labels: Dict[str, Any]
    notes: Optional[str] = None
    data_desc: Optional[str] = None
    candidate_resonance: Optional[str] = None


@dataclass
class BodySpec:
    key: str
    batch: str
    regime: str
    elements: Dict[str, float]
    epoch_jd: float
    elements_source: str
    integrator: str
    dt: float
    span_yr: float
    step_yr: float
    moon: bool
    save_planets: bool
    classifier: Dict[str, Any]
    examples: List[ExampleSpec] = field(default_factory=list)
    period_yr: float = float('nan')
    q_au: float = float('nan')
    run_id: str = ''
    note: str = ''

    def run_key(self) -> str:
        integ = 'saba' if self.integrator.upper().startswith('SABA') else self.integrator.lower()
        moon = '-moon' if self.moon else ''
        return f'bench2-{self.batch}-{integ}{moon}-{span_tag(self.span_yr)}-{step_tag(self.step_yr)}-jd{self.epoch_jd:.1f}'


def regime_settings(regime: str) -> Dict[str, Any]:
    """Integrator and classifier of each regime. `classifier` holds Simulation kwargs beyond the package defaults."""
    if regime == 'A':
        return dict(integrator='SABA(10,6,4)', dt=1.0, span_yr=100_000, moon=False, save_planets=True, classifier={})
    if regime == 'B-nea':
        return dict(integrator='ias15', dt=1.0, span_yr=100_000, moon=False, save_planets=True, classifier=dict(CLASSIFIER_NEA))
    if regime == 'B-pair':
        return dict(integrator='ias15', dt=1.0, span_yr=100_000, moon=False, save_planets=True, classifier={})
    if regime == 'C':
        return dict(
            integrator='ias15',
            dt=1.0,
            span_yr=20_000,
            step_yr=0.1,
            moon=True,
            save_planets=True,
            classifier=dict(CLASSIFIER_NEA, coherence_enabled=False),
        )
    if regime == 'D-saba':
        return dict(integrator='SABA(10,6,4)', dt=0.5, span_yr=500_000, step_yr=10.0, moon=False, save_planets=False, classifier={})
    if regime == 'D-ias15':
        return dict(integrator='ias15', dt=1.0, span_yr=500_000, step_yr=10.0, moon=False, save_planets=False, classifier={})
    raise ValueError(regime)


def label_fields(row) -> Dict[str, Any]:
    return {
        'old_statuses': io.none_or_str(row.old_statuses, empty=True),
        'author_verdict': io.none_or_str(row.author_verdict, empty=True),
        'reliability': io.none_or_str(row.best_reliability, empty=True),
        'sources': io.none_or_str(row.sources, empty=True),
        'tags': [x for x in (io.none_or_str(row.tags, empty=True) or '').split('|') if x],
        'proposed_bucket': io.none_or_str(row.proposed_bucket, empty=True),
        'confidence': io.none_or_str(row.confidence, empty=True),
        'needs_integration': io.none_or_str(row.needs_integration, empty=True),
    }


def existing_run_id(example_id: str) -> Optional[str]:
    meta = io.EXAMPLES / example_id / 'meta.json'
    if not meta.exists():
        return None
    try:
        return json.load(open(meta)).get('provenance', {}).get('run_id') or '?'
    except Exception:
        return '?'


def assign_example_id(rtype: str, name: str, resonance: str, needs: str, spec: BodySpec) -> str:
    """No variant for a first series; the span as variant for 'partial' rows (data of another regime already
    extracted) and whenever another run already owns the plain id."""
    variant = span_tag(spec.span_yr) if needs == 'partial' else None
    eid = io.make_example_id(rtype, name, resonance, variant)
    owner = existing_run_id(eid)
    if owner is not None and owner != spec.run_id:
        eid = io.make_example_id(rtype, name, resonance, span_tag(spec.span_yr))
        owner = existing_run_id(eid)
        if owner is not None and owner != spec.run_id:
            raise RuntimeError(f'{eid} already exists from run {owner}')
    return eid


def build_plan(batches: List[str], only: Optional[List[str]] = None):  # noqa: C901 - one rule table
    """(bodies, excluded): one BodySpec per body with its examples, and every candidate row not integrated."""
    cand = pd.read_csv(CANDIDATES, dtype=str).fillna('')
    src = Elements()
    bodies: Dict[str, BodySpec] = {}
    excluded: List[dict] = []
    rows = cand[cand['needs_integration'].isin(['yes', 'partial'])]

    def exclude(row, reason):
        excluded.append(dict(candidate_id=row.id, name=row.name, resonance=row.resonance, rtype=row.rtype, reason=reason))

    for row in rows.itertuples(index=False):
        name = row.name.strip()
        if row.rtype.startswith('group:') or row.rtype.startswith('model'):
            exclude(row, f'{row.rtype} row: names no single body x resonance (decide on expansion separately)')
            continue
        if row.rtype == 'secular':
            exclude(row, 'secular: batch E, not run now')
            continue
        if row.rtype not in ('mmr2', 'mmr3', 'coorbital', 'zlk'):
            exclude(row, f'rtype {row.rtype} not handled')
            continue

        # batch and element source
        if row.rtype == 'zlk':
            batch, how = 'D', 'astdys'
        elif row.rtype == 'coorbital' and row.resonance.startswith('1J-1'):
            batch, how = 'A', 'astdys'  # Jupiter Trojans: main-belt regime
        elif row.rtype == 'coorbital':
            batch, how = 'C', 'coorbital'
        elif name in B_NEA:
            batch, how = 'B', 'nea'
        elif name.startswith('88710_c'):
            batch, how = 'B', 'clone'
        elif name in B_PAIRS:
            batch, how = 'B', 'astdys'
        else:
            batch, how = 'A', 'astdys'

        got = getattr(src, {'astdys': 'astdys', 'nea': 'nea', 'coorbital': 'coorbital', 'clone': 'clone'}[how])(name)
        if got is None:
            exclude(row, f'no elements for {name!r} ({how})')
            continue
        el, epoch_jd, el_source = got
        q = el['a'] * (1 - el['e'])
        note = ''
        if batch == 'A' and q < NEA_Q:
            # A planet-crosser in the main-belt list: integrate it as an NEA (MPC elements when the survey has it).
            batch = 'B'
            nea = src.nea(name)
            if nea is not None:
                el, epoch_jd, el_source = nea
                how = 'nea'
            note = f'q = {q:.3f} au < {NEA_Q}: moved from batch A (SABA) to B (ias15)'

        if batch == 'A':
            regime = 'A'
        elif batch == 'B':
            regime = 'B-pair' if name in B_PAIRS else 'B-nea'
        elif batch == 'C':
            regime = 'C'
        else:
            regime = 'D-ias15' if q < NEA_Q or el['a'] * (1 + el['e']) > 4.95 and not 5.0 < el['a'] < 5.5 else 'D-saba'
        if batch not in batches:
            continue
        if only and name not in only:
            continue

        key = f'{batch}:{name}'
        if key not in bodies:
            st = regime_settings(regime)
            period = el['a'] ** 1.5
            # FAIR (two-body angles only) needs step <= P/4; a main-belt body with three-body angles only keeps
            # 1 yr. NEAs always get P/4.2: at 1 yr an orbit of P ~ 0.9-1.5 yr is sampled less than once or twice
            # per revolution, so the raw M, lambda and the short-period terms in a would be aliased.
            two_body = rows[(rows['name'].str.strip() == name) & rows['rtype'].isin(['mmr2', 'coorbital'])]
            dense = len(two_body) > 0 or regime == 'B-nea'
            step = st.pop('step_yr', None) or (step_for_period(period) if dense else 1.0)
            span = st.pop('span_yr')
            if name in PAIR_REGIME:
                span = PAIR_REGIME[name]['span_yr']
                step = PAIR_REGIME[name]['step_yr']
            bodies[key] = BodySpec(
                key=name,
                batch=batch,
                regime=regime,
                elements=el,
                epoch_jd=epoch_jd,
                elements_source=el_source,
                span_yr=span,
                step_yr=step,
                period_yr=period,
                q_au=q,
                note=note,
                **st,
            )
            bodies[key].run_id = bodies[key].run_key()
        spec = bodies[key]
        spec.examples.append(
            ExampleSpec(
                example_id='',
                candidate_id=row.id,
                resonance=row.resonance,
                rtype=row.rtype,
                needs_integration=row.needs_integration,
                labels=label_fields(row),
                notes=io.none_or_str(row.notes, empty=True),
                data_desc=io.none_or_str(row.data_desc, empty=True),
                candidate_resonance=row.resonance,
            )
        )

    # D extras: the NEA survey's gate confirmations, with the 1:1 angle with Earth
    if 'D' in batches:
        group = cand[cand['id'] == ZLK_RECHECK_GROUP].iloc[0]
        for name in ZLK_RECHECK:
            if only and name not in only:
                continue
            el, epoch_jd, el_source = src.nea(name)
            st = regime_settings('D-ias15')
            spec = BodySpec(
                key=name,
                batch='D',
                regime='D-ias15',
                elements=el,
                epoch_jd=epoch_jd,
                elements_source=el_source,
                span_yr=st.pop('span_yr'),
                step_yr=st.pop('step_yr'),
                period_yr=el['a'] ** 1.5,
                q_au=el['a'] * (1 - el['e']),
                note='free-omega gate re-check of the NEA survey (group row c0767); 1:1 Earth angle added to test co-orbital motion',
                **st,
            )
            spec.run_id = spec.run_key()
            labels = label_fields(group)
            spec.examples.append(
                ExampleSpec(
                    '',
                    ZLK_RECHECK_GROUP,
                    'LK',
                    'zlk',
                    'yes',
                    labels,
                    notes=io.none_or_str(group.notes, empty=True),
                    candidate_resonance=group.resonance,
                )
            )
            if abs(el['a'] - 1.0) > COORBITAL_CHECK_DA:
                spec.note += f'; a = {el["a"]:.3f} au, far from Earth: no 1:1 angle'
                bodies[f'D:{name}'] = spec
                continue
            extra = dict(labels, proposed_bucket=None, confidence=None, tags=['coorbital-check'])
            spec.examples.append(
                ExampleSpec(
                    '',
                    ZLK_RECHECK_GROUP,
                    COORBITAL_CHECK_RESONANCE,
                    'coorbital',
                    'yes',
                    extra,
                    notes='added by integrate.py: is the ZLK gate confirmation a co-orbital? (not a candidate row; unlabelled)',
                    candidate_resonance=None,
                )
            )
            bodies[f'D:{name}'] = spec

    for spec in bodies.values():
        for ex in spec.examples:
            ex.example_id = assign_example_id(ex.rtype, spec.key, ex.resonance, ex.needs_integration, spec)
    return list(bodies.values()), excluded


def make_tasks(bodies: List[BodySpec], chunk_saba: int) -> List[dict]:
    """Group bodies into Simulations: SABA bodies of one run share a Simulation (chunks of `chunk_saba`),
    ias15 bodies get one each. Longest first (span / step, ias15 weighted)."""
    by_run: Dict[str, List[BodySpec]] = {}
    for b in bodies:
        by_run.setdefault(b.run_id, []).append(b)
    tasks = []
    for run_id, members in by_run.items():
        members = sorted(members, key=lambda b: b.key)
        size = chunk_saba if members[0].integrator.upper().startswith('SABA') else 1
        for i in range(0, len(members), size):
            part = members[i : i + size]
            tid = f'{run_id}__{io.safe_token(part[0].key)}' + (f'+{len(part) - 1}' if len(part) > 1 else '')
            weight = part[0].span_yr * (20 if part[0].integrator == 'ias15' else 1) * (3 if part[0].moon else 1)
            tasks.append({'task_id': tid, 'run_id': run_id, 'bodies': [asdict(b) for b in part], 'weight': weight})
    tasks.sort(key=lambda t: -t['weight'])
    return tasks


# ------------------------------------------------------------------ worker


def classifier_config_of(config) -> dict:
    out = {}
    for k, v in vars(config).items():
        if k.startswith(io.CLASSIFIER_PREFIXES) or k in io.CLASSIFIER_KEYS:
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
                out[k] = list(v)
    return out


def planets_frame(sim) -> Optional[pd.DataFrame]:
    data = sim.integration_engine.planets_data
    if not data or not any(len(d) for d in data.values()):
        return None
    t = sim.times / (2 * np.pi)
    cols = {'t_yr': t}
    for planet, d in data.items():
        if not d:
            continue
        for f in io.PLANET_FIELDS:
            if f in d:
                cols[io.planet_column(planet, f)] = np.asarray(d[f], dtype=float)
    return pd.DataFrame(cols)


def ensure_run_planets(run_id: str, frame: pd.DataFrame, wait_s: float = 1800) -> Path:
    """Write data/runs/<run_id>/planets.csv.gz once (the first task to get the lock); others wait for it."""
    run_dir = io.RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / 'planets.csv.gz'
    lock = run_dir / 'planets.lock'
    if target.exists():
        return target
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        io.write_csv_gz(target, frame)
        lock.unlink(missing_ok=True)
        return target
    except FileExistsError:
        t0 = time.time()
        while not target.exists():
            if time.time() - t0 > wait_s:
                raise TimeoutError(f'{target} not written within {wait_s} s (stale {lock}?)')
            time.sleep(5)
        time.sleep(1)
        return target


def compare_planets(own: pd.DataFrame, run_file: Path) -> Dict[str, Any]:
    """Max |difference| between this Simulation's planets and the run's file (mean longitude in rad, a in au)."""
    ref = io.rename_legacy_planets(pd.read_csv(run_file))
    own = io.rename_legacy_planets(own)
    if len(ref) != len(own) or not np.allclose(ref['t_yr'].to_numpy(), own['t_yr'].to_numpy(), rtol=0, atol=1e-9):
        return {'same_grid': False}
    dl, da = 0.0, 0.0
    for c in own.columns:
        if c.endswith('.mean_longitude') and c in ref:
            d = np.angle(np.exp(1j * (own[c].to_numpy() - ref[c].to_numpy())))
            dl = max(dl, float(np.nanmax(np.abs(d))))
        if c.endswith('.a') and c in ref:
            da = max(da, float(np.nanmax(np.abs(own[c].to_numpy() - ref[c].to_numpy()))))
    return {'same_grid': True, 'max_dl_rad': dl, 'max_da_au': da}


PLANETS_TOL_RAD = 1e-6  # a body's own planets that differ from the run's by more than this get their own file


def gzip_package_output(task_dir: Path) -> List[str]:
    """data-*.csv and *-omega-free.csv -> .csv.gz (summary/segments/coherence stay plain)."""
    done = []
    for lock in task_dir.glob('*.lock'):  # DataManager._append_csv's lock files
        lock.unlink()
    for f in sorted(task_dir.glob('*.csv')):
        if f.name in ('summary.csv', 'segments.csv', 'coherence.csv'):
            continue
        gz = f.with_name(f.name + '.gz')
        with open(f, 'rb') as src, gzip.GzipFile(gz, 'wb', compresslevel=6, mtime=0) as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
        f.unlink()
        done.append(gz.name)
    return done


def flat_libration(libration) -> Optional[dict]:
    try:
        d = libration.to_flat_dict()
        return {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in d.items()}
    except Exception:
        return None


def body_series(body, key: str, times: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            't_yr': times / (2 * np.pi),
            'a': body.axis,
            'e': body.ecc,
            'inc': body.inc,
            'Omega': body.Omega,
            'omega': body.omega,
            'varpi': body.varpi,
            'M': body.M,
            'mean_longitude': body.longitude,
            'angle_raw': body.angles_unwrapped[key],
        }
    )


def run_task(task: dict, opts: dict) -> dict:  # noqa: C901 - one linear driver
    """Integrate one Simulation and write its bodies' examples. Runs in a spawned worker."""
    os.chdir(REPO)
    set_data_root(opts.get('data_root'))
    import resonances
    from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance

    t0 = time.time()
    specs = [BodySpec(**{**b, 'examples': [ExampleSpec(**e) for e in b['examples']]}) for b in task['bodies']]
    first = specs[0]
    span = opts.get('years_override') or first.span_yr
    step = first.step_yr
    nout = int(round(span / step)) + 1
    date = jd_to_datetime(first.epoch_jd)
    task_dir = Path(opts['runs_dir']) / task['run_id'] / task['task_id']
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    sim = resonances.Simulation(
        name=task['task_id'],
        date=date,
        source='astdys' if 'AstDyS' in first.elements_source else 'nasa',
        integrator=first.integrator,
        dt=first.dt,
        tmax=span * 2 * np.pi,
        Nout=nout,
        solar_system_moon=first.moon,
        save='all',
        save_summary=True,
        save_planets=first.save_planets,
        plot=None,
        save_path=str(task_dir),
        plot_path=str(task_dir),
        data_folder='',  # the series next to summary.csv: the task folder's own layout (gzip_package_output, below)
        batch_enabled=False,
        **first.classifier,
    )
    sim_jd = io.date_to_jd(sim.config.date)
    for spec in specs:
        if abs(spec.epoch_jd - sim_jd) > 1e-6:
            raise AssertionError(f'{spec.key}: element epoch JD {spec.epoch_jd} != simulation date JD {sim_jd} ({sim.config.date})')
        sim.add_body(dict(spec.elements, epoch_jd=spec.epoch_jd), [ex.resonance for ex in spec.examples], name=spec.key)
    sim.create_solar_system()
    cache = solar_cache_file(sim.config.date, first.moon)
    t_setup = time.time()
    sim.run()
    t_run = time.time()
    if abs(sim.times[-1] / (2 * np.pi) - span) > 1e-6 * span or len(sim.times) != nout:
        raise AssertionError(f'time grid: {len(sim.times)} points to {sim.times[-1] / (2 * np.pi)} yr, expected {nout} to {span}')

    package_files = gzip_package_output(task_dir)
    pframe = planets_frame(sim) if first.save_planets else None
    run_planets = ensure_run_planets(task['run_id'], pframe) if pframe is not None else None
    planets_check = compare_planets(pframe, run_planets) if pframe is not None else None
    if planets_check and planets_check.get('same_grid') and planets_check['max_dl_rad'] <= PLANETS_TOL_RAD:
        # the package's per-task planet files equal the run's planets.csv.gz: keep one copy
        for f in task_dir.glob('data-planet-*.csv.gz'):
            f.unlink()
        (task_dir / 'PLANETS.txt').write_text(f'data-planet-*.csv removed: equal to {run_planets} ({planets_check})\n')

    commit = io.package_commit()
    results = []
    per_body_s = (t_run - t_setup) / len(specs)
    for body, spec in zip(sim.bodies, specs):
        assert body.name == spec.key
        data_file = task_dir / f'data-{body.name}.csv.gz'
        free_file = task_dir / f'{body.name}-omega-free.csv.gz'
        written = []
        for ex in spec.examples:
            res = resonances.create_resonance(ex.resonance)
            key = res.to_s()
            series = body_series(body, key, sim.times)
            for c in ('a', 'e', 'angle_raw'):
                if not np.all(np.isfinite(series[c].to_numpy())):
                    raise ValueError(f'{spec.key} {key}: non-finite {c}')
            libration = body.librations.get(key)
            gate = body.free_gates.get(key) if hasattr(body, 'free_gates') else None
            status = body.statuses.get(key)
            meta = {
                'candidate_id': ex.candidate_id,
                'name': spec.key,
                'resonance': key,
                'candidate_resonance': ex.candidate_resonance,
                'rtype': ex.rtype,
                'resonance_type': res.type,
                'regime': {
                    'integrator': first.integrator,
                    'dt': first.dt,
                    'span_yr': float(series['t_yr'].iloc[-1] - series['t_yr'].iloc[0]),
                    'step_yr': float(np.median(np.diff(series['t_yr'].to_numpy()))),
                    'direction': 'forward',
                    'epoch_jd': spec.epoch_jd,
                    'epoch_date': sim.config.date.isoformat(),
                    'elements_source': spec.elements_source,
                    'planets': [p for p in sim.integration_engine.massive_bodies if p != 'Sun'],
                    'moon': bool(first.moon),
                    'source_Nout': nout,
                    'secular_phases': None,
                    'decimation': None,
                    'times_check': 'years (written from Simulation.times / 2pi by integrate.py)',
                    'batch': spec.batch,
                    'regime_name': spec.regime,
                    'period_yr': spec.period_yr,
                    'q_au': spec.q_au,
                    'fair_step_fraction': step / spec.period_yr,
                },
                'initial_elements': dict(spec.elements, epoch_jd=spec.epoch_jd, units='au, rad; heliocentric ecliptic J2000'),
                'provenance': {
                    'run_id': task['run_id'],
                    'task_id': task['task_id'],
                    'task_bodies': [s.key for s in specs],
                    'source_files': [
                        {'path': str(data_file), 'size': data_file.stat().st_size if data_file.exists() else None, 'sha256': None}
                    ],
                    'source_columns': {
                        't_yr': 'Simulation.times / 2pi',
                        'a': 'body.axis',
                        'e': 'body.ecc',
                        'inc': 'body.inc',
                        'Omega': 'body.Omega',
                        'omega': 'body.omega',
                        'varpi': 'body.varpi',
                        'M': 'body.M',
                        'mean_longitude': 'body.longitude',
                        'angle_raw': f'body.angles_unwrapped[{key!r}] after prepare_angles (np.unwrap of the raw angle)',
                    },
                    'derived': [],
                    'source_package_commit': commit,
                    'solar_system_cache': {'path': str(cache), 'sha1': sha1_of(cache)},
                    'planets_check': planets_check,
                    'integrated': {
                        'script': SCRIPT,
                        'created': CREATED,
                        'finished': datetime.datetime.now().isoformat(timespec='seconds'),
                        'wall_s_task': round(t_run - t0, 1),
                        'wall_s_per_body': round(per_body_s, 1),
                        'dry_run': bool(opts.get('dry_run')),
                    },
                },
                'labels': dict(ex.labels, source_status=None if status is None else str(status)),
                'angles': {'angle_raw': 'unwrapped resonant angle as integrated by the package, no filter'},
                'classifier_config': classifier_config_of(sim.config),
                'source_extra': {
                    'libration': flat_libration(libration),
                    'free_gate': (
                        {'outcome': str(gate.outcome), 'comment': gate.comment}
                        if gate is not None and isinstance(res, LidovKozaiResonance)
                        else None
                    ),
                },
                'notes': ex.notes,
                'data_desc': ex.data_desc,
                'integrate_note': spec.note or None,
            }
            kw: Dict[str, Any] = {}
            if pframe is not None:
                if planets_check and planets_check.get('same_grid') and planets_check['max_dl_rad'] <= PLANETS_TOL_RAD:
                    kw['planets_run_id'] = task['run_id']
                else:
                    keep = ['t_yr'] + [c for c in pframe.columns if c.split('.', 1)[0] in (getattr(res, 'planets_names', None) or [])]
                    kw['planets'] = pframe[keep] if len(keep) > 1 else pframe
            extra = {}
            if isinstance(res, LidovKozaiResonance) and free_file.exists():
                extra['free_omega.csv.gz'] = free_file
            io.write_example(ex.example_id, series, meta, extra_files=extra, **kw)
            written.append(
                {
                    'example_id': ex.example_id,
                    'candidate_id': ex.candidate_id,
                    'rtype': ex.rtype,
                    'proposed_bucket': ex.labels.get('proposed_bucket'),
                    'confidence': ex.labels.get('confidence'),
                    'status': None if status is None else str(status),
                    'bytes': (io.EXAMPLES / ex.example_id / 'series.csv.gz').stat().st_size,
                    'rows': len(series),
                }
            )
        marker = {
            'body': spec.key,
            'batch': spec.batch,
            'run_id': task['run_id'],
            'task_id': task['task_id'],
            'span_yr': span,
            'step_yr': step,
            'rows': nout,
            'wall_s_task': round(t_run - t0, 1),
            'wall_s_integrate_per_body': round(per_body_s, 1),
            'n_bodies_in_task': len(specs),
            'examples': written,
            'finished': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        io.write_json(Path(opts['runs_dir']) / 'done' / done_name(spec.batch, spec.key), marker)
        results.append(marker)
    return {
        'task_id': task['task_id'],
        'wall_s': round(time.time() - t0, 1),
        'integrate_s': round(t_run - t_setup, 1),
        'bodies': results,
        'package_files': package_files,
    }


# ------------------------------------------------------------------ main process


def write_run_records(bodies: List[BodySpec], runs_dir: Path, opts: dict, merge: bool = False) -> None:
    """Write data/runs/<run_id>/run.json per run. With `merge` (rerun.py) the bodies and element sources of an
    existing run.json are kept (`bench_io.merge_run_record`); a batch run replaces the record."""
    by_run: Dict[str, List[BodySpec]] = {}
    for b in bodies:
        by_run.setdefault(b.run_id, []).append(b)
    for run_id, members in by_run.items():
        b = members[0]
        date = jd_to_datetime(b.epoch_jd)
        cache = solar_cache_file(date, b.moon)
        span = opts.get('years_override') or b.span_yr
        info = {
            'run_id': run_id,
            'kind': 'integrate',
            'script': SCRIPT,
            'batch': b.batch,
            'regime': b.regime,
            'source_dir': str(runs_dir / run_id),
            'package_commit': io.package_commit(),
            'epoch_date': date.isoformat(),
            'epoch_jd': b.epoch_jd,
            'direction': 'forward',
            'integrator': b.integrator,
            'dt': b.dt,
            'span_yr': span,
            'step_yr': b.step_yr,
            'Nout': int(round(span / b.step_yr)) + 1,
            'moon': b.moon,
            'save_planets': b.save_planets,
            'classifier_overrides': b.classifier,
            'elements_sources': sorted({m.elements_source for m in members}),
            'bodies': sorted(m.key for m in members),
            'solar_system_cache': {'path': str(cache), 'sha1_at_plan': sha1_of(cache)},
            'tasks': 'SABA: shared Simulation per chunk; ias15: one Simulation per body',
            'dry_run': bool(opts.get('dry_run')),
            'created': CREATED,
        }
        if b.save_planets:
            info['planets_file'] = 'planets.csv.gz'
        old_json = io.RUNS / run_id / 'run.json'
        old_planets = io.RUNS / run_id / 'planets.csv.gz'
        if old_planets.exists():
            with gzip.open(old_planets, 'rt') as f:
                n_rows = sum(1 for _ in f) - 1
            if n_rows != info['Nout']:
                io.log(f'{run_id}: planets.csv.gz has {n_rows} rows, this grid {info["Nout"]}: removed')
                old_planets.unlink()
        if merge:
            io.merge_run_record(old_json, info)
        else:
            io.write_json(old_json, info)


def prefetch_solar_systems(bodies: List[BodySpec]) -> None:
    """Fetch every run's planets from Horizons into the package cache before any worker starts, so that
    workers only ever load cache/solar-<ts>.bin (two workers fetching the same date would race on the file)."""
    import resonances

    seen = set()
    for b in bodies:
        date = jd_to_datetime(b.epoch_jd)
        if (date, b.moon) in seen:
            continue
        seen.add((date, b.moon))
        if solar_cache_file(date, b.moon).exists():
            continue
        io.log(f'fetching the planets at {date.isoformat()} (moon={b.moon}) from Horizons')
        sim = resonances.Simulation(
            name='prefetch',
            date=date,
            integrator=b.integrator,
            solar_system_moon=b.moon,
            save=None,
            plot=None,
            save_summary=False,
            _skip_path_verification=True,
        )
        sim.create_solar_system()
        if not solar_cache_file(date, b.moon).exists():
            raise RuntimeError(f'Horizons fetch did not write {solar_cache_file(date, b.moon)}')


def plan_table(bodies: List[BodySpec]) -> pd.DataFrame:
    rows = []
    for b in bodies:
        for ex in b.examples:
            rows.append(
                dict(
                    batch=b.batch,
                    run_id=b.run_id,
                    body=b.key,
                    regime=b.regime,
                    integrator=b.integrator,
                    dt=b.dt,
                    span_yr=b.span_yr,
                    step_yr=b.step_yr,
                    nout=int(round(b.span_yr / b.step_yr)) + 1,
                    period_yr=round(b.period_yr, 3),
                    q_au=round(b.q_au, 3),
                    epoch_jd=b.epoch_jd,
                    elements_source=b.elements_source,
                    example_id=ex.example_id,
                    candidate_id=ex.candidate_id,
                    resonance=ex.resonance,
                    note=b.note,
                )
            )
    return pd.DataFrame(rows)


def verify(batch_dirs: List[Path], data_root: Optional[str]) -> pd.DataFrame:
    """Reload every example the batch wrote and reclassify it from the saved files alone; compare with the status of
    the integration itself. An MMR goes through `analyse_mmr`, a secular or ZLK angle through the package's own path
    (`reclassify_secular_zlk.build_simulation` + `Simulation.identify_librations`, the free-omega gate included). The
    verdict's status string is compared with the integration's `status` column when that is a string, its
    `status_code` with an integer source status."""
    set_data_root(data_root)
    import reclassify_secular_zlk as long_period
    from resonances.resonance.analysis.mmr import io as mmr_io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.resonance.analysis.settings import settings_from_config

    rows = []
    for d in batch_dirs:
        for marker in sorted((d / 'done').glob('*.json')):
            m = json.load(open(marker))
            for ex in m['examples']:
                eid = ex['example_id']
                try:
                    e = io.load_example(eid)
                    src = e.meta['labels'].get('source_status')
                    if e.meta['rtype'] in ('mmr2', 'mmr3', 'coorbital'):
                        verdict = analyse_mmr(mmr_io.load_series(eid), settings_from_config()).verdict
                        mine = verdict.status_code if isinstance(src, (int, float)) else str(verdict.status)
                        rows.append(
                            dict(
                                example_id=eid,
                                rows=len(e.series),
                                planets=e.planets is not None,
                                source_status=src,
                                reclassified=mine,
                                match=src == mine,
                            )
                        )
                        continue
                    sim, body, res = long_period.build_simulation(e)
                    sim.identify_librations()
                    verdict = body.librations[res.to_s()]
                    mine = verdict.status_code if isinstance(src, (int, float)) else str(verdict.status)
                    rows.append(
                        dict(
                            example_id=eid,
                            rows=len(e.series),
                            planets=e.planets is not None,
                            source_status=src,
                            reclassified=mine,
                            match=src == mine,
                        )
                    )
                except Exception as exc:
                    rows.append(dict(example_id=eid, error=f'{type(exc).__name__}: {exc}', match=False))
    return pd.DataFrame(rows)


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch', required=True, help='A, B, C, D or a comma list')
    ap.add_argument('--only', help='comma-separated body names')
    ap.add_argument('--n-cores', type=int, default=4)
    ap.add_argument('--chunk-saba', type=int, default=8, help='bodies per SABA Simulation (default 8)')
    ap.add_argument('--plan', action='store_true', help='write runs/<batch>/plan.csv and the exclusions, do not integrate')
    ap.add_argument('--dry-run', action='store_true', help='write into benchmark/dryrun/, never labels.csv')
    ap.add_argument('--years', type=float, default=None, help='override every span (dry runs)')
    ap.add_argument('--limit', type=int, default=None, help='first N tasks only')
    ap.add_argument('--force', action='store_true', help='redo bodies already done')
    ap.add_argument('--verify', action='store_true', help='reclassify the written examples from the files and compare')
    args = ap.parse_args()

    os.chdir(REPO)
    batches = [b.strip().upper() for b in args.batch.split(',') if b.strip()]
    only = [x.strip() for x in args.only.split(',')] if args.only else None
    if args.years and not args.dry_run:
        ap.error('--years is for dry runs only')
    root = DRY_ROOT if args.dry_run else BENCH
    data_root = str(DRY_ROOT / 'data') if args.dry_run else None
    tag = ''.join(batches)
    runs_dir = root / 'runs' / tag

    if args.verify:
        df = verify([runs_dir], data_root)
        out = runs_dir / 'verify.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        io.log(f'verify: {int(df["match"].sum())} of {len(df)} match the integration status -> {out}')
        if (~df['match']).any():
            print(df[~df['match']].to_string())
        return 0 if df['match'].all() else 1

    bodies, excluded = build_plan(batches, only)
    plan = plan_table(bodies)
    runs_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(runs_dir / 'plan.csv', index=False)
    pd.DataFrame(excluded).to_csv(runs_dir / 'excluded.csv', index=False)
    io.log(
        f'batches {batches}: {len(bodies)} bodies, {len(plan)} examples, {plan["run_id"].nunique() if len(plan) else 0} runs; '
        f'{len(excluded)} candidate rows not integrated (all batches) -> {runs_dir}/excluded.csv'
    )
    if len(plan):
        print(plan.groupby('run_id').agg(bodies=('body', 'nunique'), examples=('example_id', 'size'), nout=('nout', 'first')).to_string())
    if args.plan:
        return 0

    opts = {'runs_dir': str(runs_dir), 'data_root': data_root, 'dry_run': args.dry_run, 'years_override': args.years}
    set_data_root(data_root)
    done_dir = runs_dir / 'done'
    prefetch_solar_systems(bodies)
    write_run_records(bodies, runs_dir, opts)
    for lock in io.RUNS.glob('*/planets.lock'):  # left by a killed worker; no worker runs now
        if not (lock.parent / 'planets.csv.gz').exists():
            lock.unlink()
    if not args.dry_run and done_dir.exists():
        # A main process killed after a task finished leaves done markers whose labels were never written; a
        # restart skips those bodies, so re-sync their labels here (upsert is idempotent).
        resync = [
            {k: ex.get(k) for k in ('example_id', 'candidate_id', 'rtype', 'proposed_bucket', 'confidence')}
            for marker in sorted(done_dir.glob('*.json'))
            for ex in json.load(open(marker)).get('examples', [])
        ]
        if resync:
            upsert_labels_locked(resync)
    if not args.force:
        todo = [b for b in bodies if not (done_dir / done_name(b.batch, b.key)).exists()]
        if len(todo) < len(bodies):
            io.log(f'{len(bodies) - len(todo)} bodies already done, skipped (--force to redo)')
        bodies = todo
    if not bodies:
        io.log('nothing to do')
        return 0
    tasks = make_tasks(bodies, args.chunk_saba)
    if args.limit:
        tasks = tasks[: args.limit]
    io.log(f'{len(tasks)} tasks on {args.n_cores} workers')

    failed = []
    t0 = time.time()
    ctx = multiprocessing.get_context('spawn')
    with cf.ProcessPoolExecutor(max_workers=args.n_cores, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_task, t, opts): t for t in tasks}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            task = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                failed.append(task['task_id'])
                fdir = runs_dir / 'failed'
                fdir.mkdir(parents=True, exist_ok=True)
                (fdir / f'{task["task_id"]}.txt').write_text(''.join(traceback.format_exception(exc)))
                io.log(f'[{i}/{len(tasks)}] FAILED {task["task_id"]}: {type(exc).__name__}: {exc}')
                continue
            labels = [
                {k: ex[k] for k in ('example_id', 'candidate_id', 'rtype', 'proposed_bucket', 'confidence')}
                for b in res['bodies']
                for ex in b['examples']
            ]
            if not args.dry_run:
                upsert_labels_locked(labels)
            statuses = ', '.join(f'{ex["example_id"]}={ex["status"]}' for b in res['bodies'] for ex in b['examples'])
            io.log(f'[{i}/{len(tasks)}] {task["task_id"]}: {res["wall_s"]} s (integration {res["integrate_s"]} s); {statuses}')
    io.log(f'done in {time.time() - t0:.0f} s; failed tasks: {len(failed)}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
