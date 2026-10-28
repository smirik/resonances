"""Regenerate every figure of the documentation, idempotently, without integrating anything.

    make docs-figures                                   # everything
    make docs-figures ONLY=alias,pendulum_portrait      # a few (gallery anchors, schematic or kind-set names)
    uv run python docs/tools/build_figures.py --list    # the names

Three kinds of figures:

- the gallery (docs/img/gallery/<example_id>.png, docs/physics/gallery.md): one-page verdict figures of benchmark
  examples. Each example's saved series (benchmark/data/examples/<id>, or $RESONANCES_BENCHMARK_DATA) is analysed
  again by the package (`analyse_mmr` for MMRs and co-orbitals; a `Simulation` rebuilt from the series and
  `identify_librations` for secular and Lidov-Kozai angles, as benchmark/tools/reclassify*.py do) and drawn by the
  package's own figure code. The verdicts go to docs/img/gallery/verdicts.json, the page is rewritten from it, and
  each verdict is compared with the latest saved benchmark run (`SAVED_RUNS`): a difference is printed, never hidden.
  Without the benchmark data the gallery figures are kept as they are and only the page is rewritten;
- schematics (docs/img/physics/*.png): pure numpy/matplotlib drawings of the physics pages, using the package's own
  formulas (pendulum period factor, filter gain, closure of a cycle) where it has them;
- standalone kinds (docs/img/figures/*.png, `KIND_SETS`): the figure kinds other than `combined` (recurrence, FAIR,
  portrait, cycles; secular angle, rate, portrait; ecc_vector and free_omega) of a few benchmark examples, drawn by
  the package's own plotters for docs/reading-figures.md.

Nothing is written outside docs/img/ and docs/physics/gallery.md; a file whose bytes did not change is not rewritten.
"""

from __future__ import annotations

import os

# One BLAS thread per process: the worker processes are the parallelism (as in benchmark/tools/reclassify.py).
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse  # noqa: E402
import io as stdio  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / 'docs'
GALLERY_DIR = DOCS / 'img' / 'gallery'
PHYSICS_DIR = DOCS / 'img' / 'physics'
FIGURES_DIR = DOCS / 'img' / 'figures'
GALLERY_PAGE = DOCS / 'physics' / 'gallery.md'
VERDICTS = GALLERY_DIR / 'verdicts.json'
TOOLS = REPO / 'benchmark' / 'tools'
SAVED_RUNS = {'mmr': 'v2-w25-final', 'long_period': 'v2-w25-final-secular-zlk'}  # under the benchmark data folder

GALLERY_WIDTH_PX = 2000  # the verdict figures are 24 in wide: 83 dpi
GALLERY_COLOURS = 256  # palette PNG: the figures are flat colours and text
KIND_WIDTH_PX = 1100  # the standalone kinds of docs/reading-figures.md
SCHEMATIC_DPI = 150
COMPARED = ('status', 'angle_status', 'final_regime', 'reason', 'confidence', 'libration_origin', 'action')


# ------------------------------------------------------------------ the gallery's cases


@dataclass(frozen=True)
class Case:
    anchor: str
    example_id: str
    group: str
    title: str
    caption: str
    member: str = ''  # also draw this derived row (a subresonance member) as a second figure


GROUPS = (
    ('resonant', 'Resonant', 'The angle librates over the whole record and the libration belongs to this resonance.'),
    ('transient', 'Transient', 'Librating episodes in part of the record, circulation or a drift in the rest.'),
    ('near_separatrix', 'Near the separatrix', 'No closed libration cycle, but the angle stalls at the top of its own well.'),
    ('non_resonant', 'Non-resonant', 'Circulation, or a bounded angle that this resonance does not make.'),
    ('uncertain', 'Uncertain', 'The record cannot decide.'),
    ('invalid', 'Invalid', 'The record cannot be measured at these settings.'),
    ('secular', 'Secular resonances', 'Angles of the proper perihelion and node against the planets\' secular modes.'),
    ('zlk', 'Lidov-Kozai', 'The argument of perihelion $\\omega$, judged together with its free part (the free-omega gate).'),
)

CASES = (
    # resonant
    Case(
        'resonant-two-body',
        'mmr2__1921__2J-1+0-1',
        'resonant',
        'A two-body resonance: the textbook pendulum',
        '$\\sigma$ librates about a fixed centre over the whole record, $a$ oscillates in step with it, and every test of the force '
        'law and of the pendulum\'s energy passes.',
    ),
    Case(
        'resonant-three-body',
        'mmr3__2487__4J-2S-1+0+0-1',
        'resonant',
        'A three-body resonance with Jupiter and Saturn',
        'The same reading for a three-body angle: $\\sigma$ librates over the whole record and $a$ follows it as the pendulum of this '
        'resonance requires.',
    ),
    Case(
        'quick-start-463',
        'mmr3__463__4J-2S-1+0+0-1',
        'resonant',
        'The body of the quick start',
        'The resonance checked in the [Quick start](../quick-start.md). The libration is as clear as above; one margin is '
        'thin, so the confidence is high rather than certain.',
    ),
    Case(
        'resonant-coorbital',
        'coorbital__2002_AA29__1E-1+0+0',
        'resonant',
        'An Earth co-orbital',
        'The angle $\\lambda - \\lambda_E$ alternates between horseshoe and quasi-satellite motion and never circulates. A 1:1 angle is judged '
        'by its own force law, from the averaged direct and indirect potential of the planet.',
    ),
    Case(
        'coorbital-regime-change',
        'coorbital__164207__1E-1+0+0',
        'resonant',
        'Two co-orbital regimes in turn',
        'Quasi-satellite motion, then horseshoe motion: two regimes of the same 1:1 resonance follow each other without a '
        'circulation in between, so together they cover the record (subtype regime_change).',
    ),
    Case(
        'moving-centre',
        'mmr2__2013_AS27__31E-36+0+5',
        'resonant',
        'Libration about a moving centre',
        'The centre of libration follows the slowly moving minimum of the averaged potential; the libration is tested in '
        '$\\psi = \\sigma - \\sigma_c(t)$. The output step folds short periods, which holds the confidence at medium and asks for a resample.',
    ),
    Case(
        'centre-scale',
        'mmr2__385458__9N-17+0+8__300kyr',
        'resonant',
        'A slow libration on a long record',
        'A high-order resonance with Neptune whose libration period is long: over this record the angle closes enough cycles '
        'to be called resonant. The same body on a shorter record is [uncertain](#insufficient-time).',
    ),
    Case(
        'insufficient-time',
        'mmr2__385458__9N-17+0+8',
        'uncertain',
        'The same body, too short a record',
        'The body of [the slow libration](#centre-scale) on a shorter record. The angle completes too few cycles to tell '
        'a libration from a slow circulation: uncertain, insufficient_time, and the action asks for a longer integration.',
    ),
    Case(
        'geometric-libration',
        'mmr3__2348__4J-2S-1+0+0-1',
        'resonant',
        'Libration decided by the geometry alone',
        '$\\sigma$ librates over the whole record, but neither the force law nor the energy of the pendulum could be measured: '
        'only the geometry decided (reason geometric_libration, origin unknown), and the confidence is at most medium.',
    ),
    # transient
    Case(
        'transient-recurrent',
        'mmr3__52356__5J-2S-2+0+0-1',
        'transient',
        'Repeated capture and release',
        'Librating episodes alternate with circulation (subtype recurrent). The episodes stay close to the separatrix, which '
        'caps the confidence.',
    ),
    Case(
        'transient-exit',
        'coorbital__2017_UM52__1E-1+0+0',
        'transient',
        'A co-orbital that leaves the 1:1 resonance',
        'The exit is seen in the orbit-averaged angle: after it $\\lambda - \\lambda_E$ circulates to the end of the record (final_regime circulation).',
    ),
    Case(
        'transient-capture-exit',
        'mmr2__115978__11M-19+0+8',
        'transient',
        'An episode, then circulation',
        'A libration episode followed by circulation to the end of the record. The episode closes only about one cycle, '
        'which caps the confidence.',
    ),
    Case(
        'transient-before-jump',
        'mmr2__2003_UG22__10E-17+0+7__40kyr',
        'transient',
        'An episode ended by a jump of the semi-major axis',
        'The episode ends where $a$ jumps (a kick, as at a close approach to a planet); after the jump the angle no longer librates. An '
        'episode that ends at a jump caps the confidence.',
    ),
    Case(
        'transient-subresonance',
        'mmr2__82831__13M-24+0+11',
        'transient',
        'Episodes made by a member of the multiplet',
        'The catalogue angle drifts with $\\varpi$, but its member $\\sigma + 2(\\varpi_M - \\varpi)$ (13M-24+2+9) librates about a fixed centre: the '
        'episodes belong to that member (reason subresonance_librates). The second figure is the member\'s own verdict.',
        member='13M-24+2+9',
    ),
    Case(
        'geometric-episode',
        'mmr3__24820__3J-1S-1+0+0-1',
        'transient',
        'An episode decided by the geometry alone',
        'A librating episode in part of the record; no force could be measured, so the geometry alone decided '
        '(reason geometric_episode, origin unknown).',
    ),
    # near the separatrix
    Case(
        'near-separatrix',
        'mmr2__523899__3N-5+0+2',
        'near_separatrix',
        'Stalls at the top of the well',
        'No closed libration cycle, but $\\sigma$ repeatedly stalls near the top of its own potential well while $a$ responds '
        '(pattern B): motion close to the separatrix. This status is rare.',
    ),
    # non-resonant
    Case(
        'circulation',
        'mmr2__42355__3N-4+0+1',
        'non_resonant',
        'Circulation',
        '$\\sigma$ circulates over the whole record: no libration anywhere.',
    ),
    Case(
        'alias',
        'mmr3__1999_UQ__7E+2M-9+0+0+0',
        'non_resonant',
        'A stroboscope: the output step aliases a fast angle',
        'The output step is close to a multiple of the fast period of $\\sigma$, so the sampled angle seems to move slowly. The '
        'analysis finds the folded branch and reads the true motion as a fast circulation (reason alias).',
    ),
    Case(
        'alias-fine-step',
        'mmr3__1999_UQ__7E+2M-9+0+0+0__step1yr',
        'non_resonant',
        'The same body at a fine step',
        'Sampled finely, the angle circulates fast, as the alias analysis of the coarse record predicted.',
    ),
    Case(
        'kinematic-libration',
        'mmr3__285637__3M+7J-6+0+0-4',
        'non_resonant',
        'A kinematic libration',
        '$\\sigma$ librates over the whole record, but $a$ does not follow it: the bound comes from the secular cycle of the '
        'elements, not from the resonant force (origin kinematic).',
    ),
    Case(
        'kinematic-drift',
        'mmr3__62243__1M-4J-1+0+0+4',
        'non_resonant',
        'A kinematic drift',
        '$\\sigma$ drifts slowly with the secular terms of the angle while $a$ does not respond: no resonant force is at work.',
    ),
    Case(
        'induced-forced-wave',
        'mmr3__117306__1V+5J-7+0+0+1',
        'non_resonant',
        'Episodes induced by another resonance',
        '$a$ oscillates because of another resonance (the owner, 11M-20+0+9), which circulates with the period of that '
        'oscillation; $\\sigma$ follows $a$ only through its rate (origin induced, a forced wave).',
    ),
    Case(
        'induced-owner-librates',
        'mmr3__1997_GK3__3M+5S-3+0+0-5',
        'non_resonant',
        'The owner librates',
        'Here the owner, the two-body resonance 10E-17+0+7 with Earth, librates itself and drives $a$; the episodes of the '
        'three-body angle are its consequence (origin induced).',
    ),
    Case(
        'implausible-strength',
        'mmr2__2897__35S-4+0-31',
        'non_resonant',
        'An implausible strength',
        'A resonance of order 31 with Saturn: its episodes would need a resonant force many orders of magnitude stronger '
        'than the averaged potential provides, so they are not this resonance.',
    ),
    # uncertain
    Case(
        'uncertain-drift',
        'mmr3__142694__1V+3S-7+0+0+3',
        'uncertain',
        'An unexplained drift',
        '$\\sigma$ drifts, and neither a kinematic nor an induced explanation passes: uncertain, extend the integration.',
    ),
    Case(
        'uncertain-origin',
        'mmr3__2003_TX9__3E+2J-10+0+0+5',
        'uncertain',
        'A libration the physics does not confirm',
        '$\\sigma$ librates about a moving centre, but the energy test of the pendulum fails and no other origin explains the '
        'libration: uncertain, origin_unknown, review.',
    ),
    # invalid
    Case(
        'invalid-alias',
        'mmr2__1864__51M-50+0-1',
        'invalid',
        'An ambiguous alias',
        'The output step folds the angle so that its alias branches cannot be told apart: the angle cannot be measured '
        'at this step, and the action asks for a resample at the step given in the right column.',
    ),
    # secular
    Case(
        'secular-nu6',
        'secular__1222__g-g6',
        'secular',
        'The ν6 resonance',
        'The angle of the proper perihelion against the phase of Saturn\'s g6 mode librates over the whole record.',
    ),
    Case(
        'secular-z1-resonant',
        'secular__1020__g-g6+s-s6',
        'secular',
        'The z1 resonance',
        'A slow libration of g − g6 + s − s6: few cycles fit in the record, so the confidence is high rather than certain.',
    ),
    Case(
        'secular-z1-transient',
        'secular__2306__g-g6+s-s6',
        'secular',
        'z1 in episodes',
        'The z1 angle librates in part of the record and circulates in the rest.',
    ),
    Case(
        'secular-z1-circulation',
        'secular__847__g-g6+s-s6',
        'secular',
        'Just outside z1',
        'The z1 angle circulates slowly over the whole record: the body is close to the resonance but not in it.',
    ),
    Case(
        'secular-insufficient-time',
        'secular__363__g-g6+s-s6',
        'secular',
        'Too few cycles',
        'The period of the angle is comparable with the record: too few cycles to decide, extend the integration.',
    ),
    # Lidov-Kozai
    Case(
        'zlk-resonant',
        'zlk__3040__LK',
        'zlk',
        'A Lidov-Kozai libration',
        '$\\omega$ librates and its free part does too, so the free-omega gate confirms a Lidov-Kozai libration.',
    ),
    Case(
        'zlk-kinematic',
        'zlk__591986__LK',
        'zlk',
        'A kinematic libration of the argument of perihelion',
        'The osculating $\\omega$ librates, but the free $\\omega$ (the forced terms removed) winds: the forced eccentricity holds the '
        'osculating $\\omega$, and the libration is kinematic.',
    ),
    Case(
        'zlk-transient',
        'zlk__162474__LK__1500kyr',
        'zlk',
        'Lidov-Kozai episodes',
        'Two long episodes of libration of $\\omega$, separated by an interval without libration.',
    ),
    Case(
        'zlk-gate-unavailable',
        'zlk__1981__LK',
        'zlk',
        'No verdict from the gate',
        '$\\omega$ librates, but the free-omega gate cannot be evaluated on this record: no verdict (uncertain, gate_unavailable).',
    ),
    Case(
        'zlk-circulation',
        'zlk__15527__LK',
        'zlk',
        'Circulation of the argument of perihelion',
        '$\\omega$ circulates over the whole record.',
    ),
    Case(
        'zlk-invalid',
        'zlk__2010_HY22__LK',
        'zlk',
        'An unphysical orbit',
        'The body is ejected: $e$ grows above 1 and $a$ becomes negative, so the record cannot be classified (reason unphysical).',
    ),
)
CASE_BY_ANCHOR = {case.anchor: case for case in CASES}


# ------------------------------------------------------------------ small helpers


def data_root() -> Path:
    configured = os.environ.get('RESONANCES_BENCHMARK_DATA')
    return Path(configured).expanduser().resolve() if configured else REPO / 'benchmark' / 'data'


def write_if_changed(path: Path, data: bytes) -> bool:
    """Write `data` unless the file already holds exactly these bytes; True when written."""
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def gallery_png(fig, width_px: int = GALLERY_WIDTH_PX) -> bytes:
    """The figure at `width_px` wide, as a palette PNG without metadata (the same figure gives the same bytes)."""
    from PIL import Image

    buffer = stdio.BytesIO()
    fig.savefig(buffer, format='png', dpi=width_px / fig.get_size_inches()[0], metadata={'Software': None})
    buffer.seek(0)
    image = Image.open(buffer).convert('RGB').quantize(colors=GALLERY_COLOURS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = stdio.BytesIO()
    image.save(out, format='PNG', optimize=True)
    return out.getvalue()


def schematic_png(fig) -> bytes:
    buffer = stdio.BytesIO()
    fig.savefig(buffer, format='png', dpi=SCHEMATIC_DPI, metadata={'Software': None})
    return buffer.getvalue()


def text(value: Any) -> str:
    """A verdict field as text: '' for None and NaN, the value of an enum."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    return str(value)


def regime_fields(meta: Dict[str, Any], t_yr: np.ndarray) -> Dict[str, Any]:
    regime = meta.get('regime') or {}
    steps = np.diff(np.asarray(t_yr, dtype=float))
    fields = {
        'name': str(meta.get('name', '')),
        'integrator': text(regime.get('integrator')),
        'span_yr': float(t_yr[-1] - t_yr[0]),
        'step_yr': float(np.median(steps)) if len(steps) else math.nan,
        'moon': bool(regime.get('moon')),
    }
    thinned = regime.get('decimation') or {}
    if thinned.get('factor'):  # the benchmark keeps every k-th row of a finer output (no smoothing)
        fields['output_step_yr'] = float(thinned['source_step_yr'])
        fields['thinning'] = int(thinned['factor'])
    return fields


def verdict_fields(row: Dict[str, Any]) -> Dict[str, str]:
    return {field: text(row.get(field)) for field in COMPARED + ('resonance',)}


# ------------------------------------------------------------------ rendering one case (in a worker process)


def render_case(anchor: str) -> Dict[str, Any]:
    """Analyse and draw one case; returns {'anchor', 'verdicts': {file stem: fields}, 'seconds'} or an 'error'."""
    warnings.simplefilter('ignore')
    start = time.perf_counter()
    case = CASE_BY_ANCHOR[anchor]
    try:
        if case.example_id.startswith(('secular__', 'zlk__')):
            verdicts = _render_long_period(case)
        else:
            verdicts = _render_mmr(case)
        return {'anchor': anchor, 'verdicts': verdicts, 'seconds': time.perf_counter() - start}
    except Exception as exc:  # noqa: BLE001 - one broken case must not stop the others
        return {'anchor': anchor, 'error': repr(exc), 'trace': traceback.format_exc()}


def _render_mmr(case: Case) -> Dict[str, Dict[str, Any]]:
    import matplotlib.pyplot as plt

    from resonances.plotting.mmr_combined import draw_combined
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.resonance.analysis.settings import settings_from_config

    series = io.load_series(case.example_id)
    analysis = analyse_mmr(series, settings_from_config())
    regime = regime_fields(io.read_meta(io.example_path(case.example_id)), series.t_yr)
    drawn = [(case.example_id, analysis)]
    if case.member:
        member = [a for a in analysis.derived if str(a.resonance.key) == case.member]
        if not member:
            raise ValueError(f'{case.example_id}: no derived row {case.member}')
        drawn.append((f'{case.example_id}__{case.member}', member[0]))
    verdicts = {}
    for stem, item in drawn:
        fig = draw_combined(item, item.derived_series or series)
        write_if_changed(GALLERY_DIR / f'{stem}.png', gallery_png(fig))
        plt.close(fig)
        verdicts[stem] = {'example_id': case.example_id, **verdict_fields(item.summary_row()), **regime}
    return verdicts


def _render_long_period(case: Case) -> Dict[str, Dict[str, Any]]:
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(TOOLS))
    import bench_io as bio
    import reclassify_secular_zlk as rsz

    from resonances.plotting.secular_plots import draw_long_period_combined

    example = bio.load_example(case.example_id, with_planets=False)
    sim, body, resonance = rsz.build_simulation(example)
    sim.identify_librations()
    row = rsz.summary_row(case.example_id, body, resonance)
    elements = body.free_elements if resonance.type == 'lidov_kozai' else None
    analysis = sim.long_period_analysis(body, resonance)
    fig = draw_long_period_combined(analysis, row=rsz._extra_columns(row), free_elements=elements)
    write_if_changed(GALLERY_DIR / f'{case.example_id}.png', gallery_png(fig))
    plt.close(fig)
    regime = regime_fields(example.meta, example.series['t_yr'].to_numpy(dtype=float))
    return {case.example_id: {'example_id': case.example_id, **verdict_fields(row), **regime}}


# ------------------------------------------------------------------ the gallery: run, compare, page


def build_gallery(anchors: List[str], workers: int) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Render the cases (≤ `workers` processes); returns the verdicts by file stem and the errors."""
    verdicts: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    if workers <= 1:
        results = map(render_case, anchors)
    else:
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'))
        results = pool.map(render_case, anchors)
    for result in results:
        if 'error' in result:
            errors.append(f"{result['anchor']}: {result['error']}")
            print(result['trace'], file=sys.stderr)
            continue
        verdicts.update(result['verdicts'])
        print(f"  {result['anchor']:<26} {result['seconds']:6.1f} s", flush=True)
    if workers > 1:
        pool.shutdown()
    return verdicts, errors


def compare_with_saved(verdicts: Dict[str, Dict[str, Any]]) -> List[str]:
    """Differences between the fresh verdicts and the saved benchmark runs, one line each."""
    import pandas as pd

    frames = {}
    for kind, run in SAVED_RUNS.items():
        path = data_root() / run / 'summary.csv'
        frames[kind] = pd.read_csv(path, dtype=str).fillna('') if path.exists() else None
    lines = []
    for stem, fresh in sorted(verdicts.items()):
        kind = 'long_period' if fresh['example_id'].startswith(('secular__', 'zlk__')) else 'mmr'
        frame = frames[kind]
        if frame is None:
            lines.append(f'{stem}: no saved run {SAVED_RUNS[kind]}')
            continue
        rows = frame[(frame.example_id == fresh['example_id']) & (frame.resonance == fresh['resonance'])]
        if rows.empty:
            lines.append(f'{stem}: not in {SAVED_RUNS[kind]}')
            continue
        saved = rows.iloc[0]
        diffs = [f'{f} {saved[f]!r} -> {fresh[f]!r}' for f in COMPARED if f in saved and saved[f] != fresh[f]]
        if diffs:
            lines.append(f'{stem}: ' + '; '.join(diffs))
    return lines


def fmt_years(years: float) -> str:
    if not math.isfinite(years):
        return '?'
    years = float(f'{years:.3g}')  # round first: 999.97 yr is '1 kyr'
    for unit, scale in (('Myr', 1e6), ('kyr', 1e3)):
        if years >= scale:
            return f'{years / scale:g} {unit}'
    return f'{years:g} yr'


def verdict_line(entry: Dict[str, Any]) -> str:
    words = [entry['status'], entry['angle_status'], entry['reason'], entry['confidence']]
    if entry['final_regime'] and entry['final_regime'] != entry['angle_status']:
        words.insert(2, f"ends: {entry['final_regime']}")
    verdict = ' · '.join(f'`{word}`' for word in words if word)
    integrator = entry['integrator'] + (' + Moon' if entry.get('moon') else '')
    step = f"step {fmt_years(entry['step_yr'])}"
    if entry.get('thinning'):
        step += f" (one point in {entry['thinning']} of a {fmt_years(entry['output_step_yr'])} output)"
    record = f"{fmt_years(entry['span_yr'])}, {step}, {integrator}"
    name = entry['name'].replace('_', ' ')
    return f"**{name}**, `{entry['resonance']}`: {verdict}. Record: {record}."


PAGE_INTRO = """\
# Gallery

One benchmark example for every status, every way of reaching it and every resonance type, each with its one-page
verdict figure. The figures are drawn by the package itself from the saved integrations; the verdict line under
each caption is the package's verdict on that record (`status` · `angle_status` · `reason` · `confidence`), so a
caption and its figure always agree. Click a figure to open it at full size.

A resonance key lists integer coefficients: first those of the mean longitudes of the planets (J Jupiter, S Saturn,
E Earth, M Mars, V Venus, N Neptune) and of the body, then those of the longitudes of perihelion in the same order
(and of the nodes, when present): `4J-2S-1+0+0-1` is $4\\lambda_J - 2\\lambda_S - \\lambda - \\varpi$. See
[Resonance matrices](../matrix.md). Secular keys name the frequencies ($g - g_6 + s - s_6$ is z1), `LK` is the
Lidov-Kozai angle $\\omega$.

## How to read the figure

The left half of each figure holds the time series and the phase portraits, the right half the verdict: the plaque
with the status, the verdict in words, the steps of the [status tree](status-tree.md), the margins and caps of the
[confidence](confidence.md) and the numbers behind them, each a column of the [output files](../output.md). What
every panel, line and icon means: [How to read the figures](../reading-figures.md).
"""

PAGE_OUTRO = """\
## Regenerating the figures

Every figure of this documentation (this gallery, the schematic drawings of the physics pages and the standalone
figures of [How to read the figures](../reading-figures.md)) is made by one command from the repository root:

```bash
make docs-figures                                  # everything (a few minutes, no integration)
make docs-figures ONLY=alias,pendulum_portrait     # some figures, by anchor, schematic or set name
uv run python docs/tools/build_figures.py --list   # the names
```

The script (`docs/tools/build_figures.py`) reads the saved series of the benchmark examples (`benchmark/data`, or
the folder in `RESONANCES_BENCHMARK_DATA`), analyses each again with the installed package and draws it with the
package's own figure code; nothing is integrated. It rewrites this page from the fresh verdicts and prints any
difference from the last saved benchmark run. Without the benchmark data the gallery and standalone figures are kept
and only the schematics are redrawn. Running it twice changes nothing: a file whose content is the same is not rewritten.
"""


def gallery_page(verdicts: Dict[str, Dict[str, Any]]) -> str:
    parts = [PAGE_INTRO]
    for group, title, intro in GROUPS:
        parts.append(f'## {title}\n\n{intro}\n')
        for case in (c for c in CASES if c.group == group):
            parts.append(_case_block(case, verdicts))
    parts.append(PAGE_OUTRO)
    return '\n'.join(parts)


def _case_block(case: Case, verdicts: Dict[str, Dict[str, Any]]) -> str:
    lines = [f'### {case.title} {{#{case.anchor}}}\n', case.caption + '\n']
    stems = [case.example_id] + ([f'{case.example_id}__{case.member}'] if case.member else [])
    for stem in stems:
        entry = verdicts.get(stem)
        if entry is None:
            lines.append(f'*The figure of `{stem}` has not been generated.*\n')
            continue
        src = f'../img/gallery/{stem}.png'
        lines.append(verdict_line(entry) + '\n')
        lines.append(f'[![{stem}]({src}){{ loading=lazy }}]({src})\n')
    return '\n'.join(lines)


# ------------------------------------------------------------------ schematics


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'legend.fontsize': 8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.constrained_layout.use': True,
            'savefig.facecolor': 'white',
        }
    )


INK, ACCENT, SECOND, GREY = '#222222', '#c0392b', '#2471a3', '#9a9a9a'


def schematic_pendulum_portrait():
    """Phase portraits of V = -omega0^2 cos psi: (psi, psi_dot / omega0) and (psi, (a - a*) / Delta a)."""
    import matplotlib.pyplot as plt

    psi = np.linspace(-np.pi, np.pi, 1441)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for h in (0.1, 0.35, 0.65, 0.9, 1.0, 1.25, 1.7):
        rate = 2 * np.sqrt(np.clip(h - np.sin(psi / 2) ** 2, 0, None))  # psi_dot / omega0 from h
        inside = h - np.sin(psi / 2) ** 2 >= 0
        colour, width = (ACCENT, 1.6) if h == 1.0 else ((INK, 0.9) if h < 1 else (SECOND, 0.9))
        for ax, scale in zip(axes, (1.0, 0.5)):  # (a - a*) / Delta a = (psi_dot / omega0) / 2
            for sign in (1, -1):
                ax.plot(np.degrees(psi[inside]), sign * scale * rate[inside], color=colour, lw=width)
    axes[0].set_ylabel(r'$\dot\psi\,/\,\omega_0$')
    axes[1].set_ylabel(r'$\mathrm{sign}(c)\,(a - a_*)\,/\,\Delta a$')
    axes[1].axhline(1, color=GREY, ls='--', lw=0.8)
    axes[1].axhline(-1, color=GREY, ls='--', lw=0.8)
    axes[1].text(-175, 1.04, r'$\Delta a = 2\omega_0/|c|$', va='bottom', fontsize=8, color=GREY)
    for ax in axes:
        ax.set_xlabel(r'$\psi = \sigma - \sigma_c$, deg')
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
    axes[0].set_title('libration (black), separatrix (red), circulation (blue)', fontsize=9)
    axes[1].set_title('the same curves in the semi-major axis', fontsize=9)
    return fig


def schematic_period_factor():
    """F(A) = P / T0 and the half-excursion of a over the half-width, from the package's pendulum."""
    import matplotlib.pyplot as plt

    from resonances.resonance.analysis.pendulum import amplitude_factor, excursion

    amplitude = np.radians(np.linspace(0.5, 178, 600))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    axes[0].plot(np.degrees(amplitude), amplitude_factor(amplitude), color=INK)
    for degrees in (90, 150):
        value = amplitude_factor(np.radians(degrees))
        axes[0].plot([degrees], [value], 'o', color=ACCENT, ms=4)
        axes[0].annotate(f'F({degrees}°) = {value:.2f}', (degrees, value), xytext=(-70, 8), textcoords='offset points', fontsize=8)
    axes[0].set_ylabel(r'$P\,/\,T_0 = F(A) = \frac{2}{\pi}\mathsf{K}(\sin^2\frac{A}{2})$')
    axes[0].set_ylim(0.9, 3.5)
    axes[1].plot(np.degrees(amplitude), excursion(1.0, amplitude, 2.0), color=INK)  # omega0 = 1, |c| = 2: sin(A/2)
    axes[1].set_ylabel(r'$\delta a\,/\,\Delta a = \sin(A/2)$')
    for ax in axes:
        ax.set_xlabel('amplitude A, deg')
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    axes[0].set_title('period of a finite libration', fontsize=9)
    axes[1].set_title('swing of a over the half-width', fontsize=9)
    return fig


def schematic_filter_gain():
    """The analysis' zero-phase Butterworth gain at an example step and cutoff (`filtering.butterworth_gain`)."""
    import matplotlib.pyplot as plt

    from resonances.resonance.analysis.filtering import butterworth_gain
    from resonances.resonance.analysis.settings import settings_from_config

    settings = settings_from_config()
    step = 1.0
    floor = settings.mmr_libration_period_min_yr
    cutoff = floor / settings.filter_min_libration_cutoffs  # where the floor term sets P_c
    period = np.logspace(np.log10(2 * step * 1.0001), 4, 800)
    gain = butterworth_gain(period, cutoff, settings.filter_order, step)
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.semilogx(period, gain, color=INK)
    marks = (
        (2 * step, r'$2\Delta t$', GREY),
        (cutoff, r'$P_c$: $G = 1/2$', ACCENT),
        (floor, rf'$P_{{\rm floor}} = {settings.filter_min_libration_cutoffs:g}\,P_c$', SECOND),
    )
    for (x, label, colour), (y, side) in zip(marks, ((0.5, 'left'), (0.12, 'right'), (0.3, 'left'))):
        ax.axvline(x, color=colour, ls='--', lw=0.8)
        ax.text(x * (0.94 if side == 'right' else 1.06), y, label, color=colour, fontsize=8, ha=side)
    ax.plot([floor], [butterworth_gain(floor, cutoff, settings.filter_order, step)], 'o', color=SECOND, ms=4)
    ax.set_xlabel('period P, yr')
    ax.set_ylabel('amplitude gain G(P)')
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        rf'forward-backward Butterworth, order N = {settings.filter_order}, $\Delta t$ = {step:g} yr, $P_c$ = {cutoff:g} yr',
        fontsize=9,
    )
    return fig


def schematic_aliasing():
    """A fast angle sampled every dt (a stroboscope) and the folding of the rate into (-pi/dt, pi/dt]."""
    import matplotlib.pyplot as plt

    step, period = 5.0, 5.05  # yr: one turn a little longer than the step
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    t = np.linspace(0, 120, 6000)
    samples = np.arange(0, 121, step)
    wrapped = np.degrees(2 * np.pi * t / period) % 360
    wrapped[1:][np.diff(wrapped) < -180] = np.nan  # no vertical line at the wrap
    axes[0].plot(t, wrapped, color=GREY, lw=0.6)
    axes[0].plot(samples, np.degrees(2 * np.pi * samples / period) % 360, 'o', color=ACCENT, ms=3.5)
    axes[0].set_xlabel('t, yr')
    axes[0].set_ylabel(r'$\sigma$ mod 360, deg')
    axes[0].set_title(rf'true $\sigma$ (grey), samples every {step:g} yr (red)', fontsize=9)
    nu = np.linspace(-0.5, 3.6, 2000) * 2 * np.pi / step  # true rate, rad/yr
    branch = np.round(nu * step / (2 * np.pi))
    apparent = nu - 2 * np.pi * branch / step
    jumps = np.flatnonzero(np.diff(branch)) + 1
    for part in np.split(np.arange(len(nu)), jumps):
        axes[1].plot(nu[part] * step / (2 * np.pi), apparent[part] * step / (2 * np.pi), color=INK)
    for m in range(0, 4):
        axes[1].plot([m], [0], 'o', color=ACCENT, ms=4)
        axes[1].text(m, 0.06, f'm = {m}', ha='center', fontsize=8, color=ACCENT)
    axes[1].axhline(0.5, color=GREY, ls='--', lw=0.8)
    axes[1].axhline(-0.5, color=GREY, ls='--', lw=0.8)
    axes[1].set_xlabel(r'true rate $\nu\,\Delta t / 2\pi$')
    axes[1].set_ylabel(r'apparent rate $\nu_{\rm app}\,\Delta t / 2\pi$')
    axes[1].set_title(r'$\nu_{\rm app} = \nu - 2\pi m/\Delta t$; red: ghosts $\nu = 2\pi m/\Delta t$', fontsize=9)
    return fig


def schematic_turning_points():
    """Turning points found by the package on synthetic filtered angles: a closed cycle, an open arc, a phase return."""
    import matplotlib.pyplot as plt

    from resonances.resonance.analysis.geometry import closure_ratios, find_turning_points

    t = np.linspace(0, 1000, 4001)
    cases = (
        ('libration: three alternating turning points close one cycle', np.radians(60) * np.sin(2 * np.pi * t / 400)),
        ('circulation with a small turn back: the cycle stays open', 2 * np.pi * t / 500 + np.radians(45) * np.sin(2 * np.pi * t / 250)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for ax, (title, sigma) in zip(axes[:2], cases):
        tp = find_turning_points(t, sigma, np.radians(5), (t[0], t[-1]))
        ratios = closure_ratios(tp.values_rad)
        ax.plot(t, np.degrees(sigma), color=INK, lw=1)
        ax.plot(tp.times_yr, np.degrees(tp.values_rad), 'v', color=ACCENT, ms=5)
        if len(ratios):
            ax.annotate(f'Q = {ratios[0]:.2f}', (tp.times_yr[1], np.degrees(tp.values_rad[1])), xytext=(8, -4), textcoords='offset points')
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel('t, yr')
    axes[0].set_ylabel(r'$\sigma_f$, deg')
    rate = np.radians(40) * 2 * np.pi / 500  # one libration loop on [200, 700] yr inside a slow circulation
    loop = (t >= 200) & (t <= 700)
    sigma = np.where(loop, np.radians(40) * np.sin(2 * np.pi * (t - 200) / 500), rate * np.where(t < 200, t - 200, t - 700))
    tp = find_turning_points(t, sigma, np.radians(5), (t[0], t[-1]))
    level = np.degrees(tp.values_rad[:2].mean())
    axes[2].plot(t, np.degrees(sigma), color=INK, lw=1)
    axes[2].plot(tp.times_yr[:2], np.degrees(tp.values_rad[:2]), 'v', color=ACCENT, ms=5)
    axes[2].axhline(level, color=SECOND, ls='--', lw=0.8)
    axes[2].text(5, level + 4, r'$\sigma_m$', color=SECOND)
    for x, label in ((200, '$t_a$'), (700, '$t_b$')):
        axes[2].plot([x], [level], 'o', color=SECOND, ms=4)
        axes[2].annotate(label, (x, level), xytext=(4, -14), textcoords='offset points', color=SECOND)
    axes[2].set_title(r'two turning points and a return to $\sigma_m$', fontsize=8.5)
    axes[2].set_xlabel('t, yr')
    return fig


def schematic_centre_types():
    """Stationary, modulated and moving-centre librations (synthetic)."""
    import matplotlib.pyplot as plt

    t = np.linspace(0, 20000, 8001)
    phase = 2 * np.pi * t / 1500
    centres = (
        ('stationary', np.full_like(t, 180.0), 60 * np.sin(phase)),
        ('modulated', 180 + 12 * np.sin(2 * np.pi * t / 9000), (45 + 25 * np.sin(2 * np.pi * t / 9000)) * np.sin(phase)),
        ('moving centre', 120 + 120 * t / t[-1], 50 * np.sin(phase)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7), sharey=True)
    for ax, (title, centre, swing) in zip(axes, centres):
        ax.plot(t / 1000, centre + swing, color=INK, lw=0.7)
        ax.plot(t / 1000, centre, color=ACCENT, lw=1.4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('t, kyr')
    axes[0].set_ylabel(r'$\sigma_f$, deg (red: centre $\sigma_c$)')
    return fig


def schematic_episodes():
    """An exit (episode, then circulation) against two regimes whose union covers the record."""
    import matplotlib.pyplot as plt

    t = np.linspace(0, 30000, 12001)
    phase = 2 * np.pi * t / 1800
    exit_at = 17000
    sigma_exit = np.where(t < exit_at, 180 + 70 * np.sin(phase), 180 + 70 * np.sin(2 * np.pi * exit_at / 1800) + 360 * (t - exit_at) / 2500)
    sigma_exit = sigma_exit % 360
    sigma_exit[1:][np.abs(np.diff(sigma_exit)) > 180] = np.nan  # no vertical line at the wrap
    sigma_union = 180 + (40 + 55 / (1 + np.exp(-(t - 14000) / 600))) * np.sin(phase)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.8))
    axes[0].plot(t / 1000, sigma_exit, color=INK, lw=0.7)
    axes[0].axvspan(0, exit_at / 1000, color=SECOND, alpha=0.15)
    axes[0].set_title('an episode, then circulation: transient', fontsize=9)
    axes[1].plot(t / 1000, sigma_union, color=INK, lw=0.7)
    axes[1].axvspan(0, 14, color=SECOND, alpha=0.15)
    axes[1].axvspan(14, 30, color=ACCENT, alpha=0.12)
    axes[1].set_title('two regimes, no circulation between: they cover the record', fontsize=9)
    for ax in axes:
        ax.set_xlabel('t, kyr')
    axes[0].set_ylabel(r'$\sigma_f$ mod 360, deg')
    axes[1].set_ylabel(r'$\sigma_f$, deg')
    return fig


SCHEMATICS: Dict[str, Callable] = {
    'pendulum_portrait': schematic_pendulum_portrait,
    'period_factor': schematic_period_factor,
    'filter_gain': schematic_filter_gain,
    'aliasing': schematic_aliasing,
    'turning_points': schematic_turning_points,
    'centre_types': schematic_centre_types,
    'episodes': schematic_episodes,
}


# ------------------------------------------------------------------ the standalone kinds (docs/reading-figures.md)

# Each set draws the standalone figure kinds of one benchmark example with the package's own plotters, as
# `DataManager.plot_body` does in a run; the files are docs/img/figures/<stem>.png.
KIND_EXAMPLES = {
    'kinds_mmr': 'mmr3__463__4J-2S-1+0+0-1',  # a libration of ~10 kyr: slower than the body-level filter of a
    'kinds_fair': 'mmr2__1921__2J-1+0-1',
    'kinds_secular': 'secular__1222__g-g6',
    'kinds_zlk': 'zlk__3040__LK',
    'kinds_zlk_kinematic': 'zlk__591986__LK',
}


def _mmr_kinds(example_id: str, kinds: Tuple[str, ...]) -> Dict[str, Any]:
    """MMR standalone kinds: the body rebuilt from the saved series, sigma_f the analysis' own."""
    sys.path.insert(0, str(TOOLS))
    import bench_io as bio

    from resonances.plotting.mmr_plots import MMRPlotter
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.resonance.analysis.settings import settings_from_config

    analysis = analyse_mmr(io.load_series(example_id), settings_from_config(), scan_subresonances=False, scan_owner=False)
    body, resonance, sim = bio.to_body(bio.load_example(example_id, with_planets=True))
    body.angles_filtered_unwrapped[resonance.to_s()] = analysis.prep.angle_f  # as after a run: the analysis' sigma_f
    body.mmr_analyses[resonance.to_s()] = analysis  # and its a_f (the a of `cycles`), as for a plotted row
    plotter = MMRPlotter.from_body(body, resonance, sim)
    figures = {}
    for kind in kinds:
        if plotter.plot(kind):
            figures[f'mmr_{kind}'] = plotter.figure
            plotter._figure = None  # keep it open: written below
    return figures


def kinds_mmr(example_id: str) -> Dict[str, Any]:
    return _mmr_kinds(example_id, ('recurrence', 'portrait', 'cycles'))


def kinds_fair(example_id: str) -> Dict[str, Any]:
    return _mmr_kinds(example_id, ('fair',))


def _long_period_simulation(example_id: str):
    sys.path.insert(0, str(TOOLS))
    import bench_io as bio
    import reclassify_secular_zlk as rsz

    sim, body, resonance = rsz.build_simulation(bio.load_example(example_id, with_planets=False))
    sim.identify_librations()
    return sim, body, resonance


def kinds_secular(example_id: str) -> Dict[str, Any]:
    """angle, rate, portrait of a secular row (`SecularPlotter`)."""
    from resonances.plotting.secular_plots import SecularPlotter

    from resonances.plotting.plotter import Plotter

    sim, body, resonance = _long_period_simulation(example_id)
    plotter = SecularPlotter.from_body(body, resonance, sim)
    figures = {}
    for kind in ('angle', 'rate', 'portrait'):
        if plotter.plot(kind):
            figures[f'secular_{kind}'] = plotter.figure
            plotter._figure = None
    sim.build_periodograms(bodies=[body])  # the evolution figure's bottom panel, as after a run
    figures['evolution'] = Plotter.from_body(body, resonance, sim).configure(sim.config.plot_config or 'full').plot().figure
    return figures


def _free_omega_figures(example_id: str, prefix: str, kinds: Tuple[str, ...]) -> Dict[str, Any]:
    """ecc_vector and the free_omega figures of a Lidov-Kozai row, as `DataManager._plot_ecc_vector` and
    `_plot_free_omega` draw them (the free vector on the window of the osculating one)."""
    from resonances.plotting.free_omega_plots import FreeOmegaPlotter
    from resonances.plotting.phase_plots import EccentricityVectorPlotter

    sim, body, resonance = _long_period_simulation(example_id)
    figures = {}
    if 'ecc_vector' in kinds:
        figures[f'{prefix}_ecc_vector'] = EccentricityVectorPlotter.from_body(body, sim=sim).plot().figure
    for kind in ('drift', 'portrait', 'vector'):
        if f'free_omega_{kind}' not in kinds:
            continue
        plotter = FreeOmegaPlotter.from_body(body, resonance)
        if kind == 'vector':
            plotter.plot_vector(limit=EccentricityVectorPlotter.from_body(body, resonance).extent())
        else:
            getattr(plotter, f'plot_{kind}')()
        figures[f'{prefix}_free_omega_{kind}'] = plotter.figure
    return figures


def kinds_zlk(example_id: str) -> Dict[str, Any]:
    return _free_omega_figures(example_id, 'zlk', ('free_omega_drift', 'free_omega_portrait'))


def kinds_zlk_kinematic(example_id: str) -> Dict[str, Any]:
    return _free_omega_figures(example_id, 'zlk_kinematic', ('ecc_vector', 'free_omega_drift', 'free_omega_vector'))


KIND_SETS: Dict[str, Callable] = {
    'kinds_mmr': kinds_mmr,
    'kinds_fair': kinds_fair,
    'kinds_secular': kinds_secular,
    'kinds_zlk': kinds_zlk,
    'kinds_zlk_kinematic': kinds_zlk_kinematic,
}


def figure_png(fig, width_px: int = KIND_WIDTH_PX) -> bytes:
    """A standalone figure at `width_px` wide, as a palette PNG without metadata (as `gallery_png`)."""
    return gallery_png(fig, width_px)


def build_kinds(names: List[str]) -> int:
    """Draw the standalone kinds of the named sets (skipped without the benchmark data); returns the errors."""
    import matplotlib.pyplot as plt

    errors = 0
    for name in names:
        example_id = KIND_EXAMPLES[name]
        if not (data_root() / 'examples' / example_id).is_dir():
            print(f'{name}: benchmark data not found under {data_root()}: its figures are kept as they are')
            continue
        start = time.perf_counter()
        warnings.simplefilter('ignore')
        try:
            figures = KIND_SETS[name](example_id)
        except Exception:  # noqa: BLE001 - one broken set must not stop the others
            print(f'ERROR {name}:\n{traceback.format_exc()}', file=sys.stderr)
            errors += 1
            continue
        for stem, fig in figures.items():
            write_if_changed(FIGURES_DIR / f'{stem}.png', figure_png(fig))
            plt.close(fig)
        print(f'  {name:<26} {len(figures)} figure(s), {time.perf_counter() - start:6.1f} s', flush=True)
    return errors


def build_schematics(names: List[str]) -> int:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    written = 0
    with mpl.rc_context():  # the schematics' style must not reach the package's own plotters
        _style()
        for name in names:
            fig = SCHEMATICS[name]()
            written += write_if_changed(PHYSICS_DIR / f'{name}.png', schematic_png(fig))
            plt.close(fig)
    return written


# ------------------------------------------------------------------ main


def select(only: Optional[str]) -> Tuple[List[str], List[str], List[str]]:
    """(gallery anchors, schematic names, kind sets) to build; `only` is a comma-separated list of any of them."""
    if not only:
        return [case.anchor for case in CASES], list(SCHEMATICS), list(KIND_SETS)
    names = [name.strip() for name in only.split(',') if name.strip()]
    unknown = [name for name in names if name not in CASE_BY_ANCHOR and name not in SCHEMATICS and name not in KIND_SETS]
    if unknown:
        raise SystemExit(f'unknown figure(s): {", ".join(unknown)} (see --list)')
    return [n for n in names if n in CASE_BY_ANCHOR], [n for n in names if n in SCHEMATICS], [n for n in names if n in KIND_SETS]


def load_verdicts() -> Dict[str, Dict[str, Any]]:
    return json.loads(VERDICTS.read_text()) if VERDICTS.exists() else {}


def run_gallery(anchors: List[str], workers: int) -> int:
    """Render the gallery cases when the data is there, then rewrite verdicts.json and the page; returns the errors."""
    verdicts = load_verdicts()
    errors: List[str] = []
    missing = [a for a in anchors if not (data_root() / 'examples' / CASE_BY_ANCHOR[a].example_id).is_dir()]
    if missing:
        print(f'benchmark data not found under {data_root()} for {len(missing)} case(s): their figures are kept as they are')
    present = [a for a in anchors if a not in missing]
    if present:
        print(f'gallery: {len(present)} case(s), {workers} process(es)')
        fresh, errors = build_gallery(present, workers)
        verdicts.update(fresh)
        differences = compare_with_saved(fresh)
        print(
            f'compared with the saved runs {SAVED_RUNS}: ' + ('no difference' if not differences else f'{len(differences)} difference(s)')
        )
        for line in differences:
            print('  ' + line)
    write_if_changed(VERDICTS, (json.dumps(verdicts, indent=1, sort_keys=True) + '\n').encode())
    write_if_changed(GALLERY_PAGE, gallery_page(verdicts).encode())
    for line in errors:
        print('ERROR ' + line, file=sys.stderr)
    return len(errors)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', help='comma-separated gallery anchors and/or schematic names')
    ap.add_argument('--list', action='store_true', help='print the names and exit')
    ap.add_argument('--workers', type=int, default=2, help='processes for the gallery (at most 2: ~2.6 GB each)')
    args = ap.parse_args(argv)
    if args.list:
        for case in CASES:
            print(f'{case.anchor:<26} {case.example_id}')
        for name in SCHEMATICS:
            print(f'{name:<26} schematic')
        for name, example_id in KIND_EXAMPLES.items():
            print(f'{name:<26} standalone kinds of {example_id}')
        return 0
    anchors, schematics, kinds = select(args.only)
    start = time.perf_counter()
    written = build_schematics(schematics)
    print(f'schematics: {len(schematics)} drawn, {written} changed')
    errors = build_kinds(kinds)
    # the gallery page is always rewritten from verdicts.json (cheap); only the named cases are analysed again
    errors += run_gallery(anchors, max(1, min(args.workers, 2)))
    print(f'done in {time.perf_counter() - start:.0f} s')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
