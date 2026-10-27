"""Classify the benchmark's secular and Lidov-Kozai examples from the saved files, through the package's own path.

    uv run python benchmark/tools/reclassify.py --rtype secular,zlk --workers 3 --out benchmark/data/v2-w4-secular_zlk
    uv run python benchmark/tools/reclassify_secular_zlk.py --only zlk__591986__LK --figures

Each example becomes a `Simulation` holding one body rebuilt from series.csv.gz (`bench_io.to_body`: the angle the
package classifies, the elements, the source run's settings) and `Simulation.identify_librations` gives the verdict
exactly as a run would (the secular/ZLK analysis, the free-omega gate, the e-i flag). Nothing is integrated. Writes
summary.csv (one row per example, keyed by example_id), episodes.csv, agreement.csv and run.json into OUT; with
`--figures` the one-page figure is drawn as combined_v2.png into the example folder.

Agreement is printed against the author's unambiguous verdicts on these records (`AUTHOR_VERDICTS`, Claude's reading
of `candidates.csv` author_verdict and DECISIONS.md, each with its source) and, separately, against the old
`proposed_bucket` of labels.csv (literature, adjudicated and algorithmic labels: a hint, not the author's word).
"""

from __future__ import annotations

import os

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as bio  # noqa: E402

RTYPES = ('secular', 'zlk')
FIGURE = 'combined_v2.png'

# The author's verdicts on these very records, as Claude reads them (candidates.csv author_verdict, DECISIONS.md).
# Rows whose statement is not a verdict about the record (2008 UZ220: a question; 1981, 622577: expectations never
# checked on a run) are left out; 1373 at 100 kyr is the same body on another record (the label came from 500 kyr).
AUTHOR_VERDICTS = {
    'secular__1020__g-g6+s-s6': ('resonant', "c0670: '3 разворота достаточно' -> status 2"),
    'secular__1222__g-g6': ('resonant', "c0673: 'либраций много' - librator"),
    'secular__4305__g-3g6+2g5': ('resonant', "c0688: 'явный резонанс' -> status 2"),
    'secular__8089__2_g-g6_+_s-s6_': ('resonant', 'c0699: z2 libration expected; weak'),
    'secular__847__g-g6+s-s6': ('non_resonant', 'c0701: right outside the resonance (slow circulation)'),
    'secular__2005_GQ107__g-2g2+g8': ('uncertain', "c0650: 'надо на 30 млн ... должно быть -6'"),
    'secular__363__g-g6+s-s6': ('uncertain', 'DECISIONS 2026-10-13: 363, P ~ 10 Myr on 10 Myr -> uncertain + extend'),
    'zlk__591986__LK': ('non_resonant', 'c0718: -2 kinematic libration, not a ZLK libration'),
    'zlk__15527__LK': ('non_resonant', 'c0721: non-resonant (second statement)'),
    'zlk__88710__LK': ('non_resonant', 'c0729: not resonant'),
    'zlk__1373__LK__500kyr': ('resonant', "c0730: 'should be resonant, status=2' (Nov 2025, not re-validated)"),
    'zlk__162474__LK': ('resonant', 'c0731: a genuine Lidov-Kozai resonance; the gate must confirm it'),
    'zlk__162474__LK__1500kyr': ('transient', 'c0792: author 2026-10-17, an ordinary transient (270 -> 90 deg island)'),
    'zlk__3040__LK': ('resonant', 'c0732: resonant (status 2)'),
}
# Same body, another record: shown, not counted.
AUTHOR_HINTS = {'zlk__1373__LK': ('resonant', 'c0730 on 500 kyr; this record is 100 kyr')}


# ------------------------------------------------------------------ one example


def build_simulation(example: 'bio.Example'):
    """A `Simulation` with the example's body, its times and the source run's settings (`bio.make_config`)."""
    from resonances.simulation.simulation import Simulation

    config = bio.make_config(example.meta, example.series, has_planets=False)
    body, resonance, _ = bio.to_body(example, config=config)
    sim = Simulation(**_config_kwargs(example))
    sim.times = body.times
    sim.body_manager.bodies.append(body)
    return sim, body, resonance


def _config_kwargs(example: 'bio.Example') -> Dict[str, Any]:
    """The keyword arguments `bio.make_config` passes to SimulationConfig (the same settings, for a Simulation)."""
    t = example.series['t_yr'].to_numpy(dtype=float)
    kwargs: Dict[str, Any] = dict(save=None, plot=None, save_summary=False, _skip_path_verification=True)
    kwargs.update(example.meta.get('classifier_config') or {})
    if kwargs.get('secular_planet_phases') == 'measured':
        kwargs['secular_planet_phases'] = 'linear'  # the saved angle is used as it is; no planets are loaded
    kwargs['save_planets'] = False
    kwargs['tmax'] = float(t[-1] - t[0]) * 2 * np.pi
    kwargs['Nout'] = len(t)
    return kwargs


def summary_row(example_id: str, body, resonance) -> Dict[str, Any]:
    """The summary.csv row of the example: identity, the verdict, the measured numbers, gate and e-i columns."""
    from resonances.resonance import omega_free_gate
    from resonances.secular import free_elements

    key = resonance.to_s()
    verdict = body.librations[key]
    row: Dict[str, Any] = {'example_id': example_id, 'name': body.name, 'resonance': key, 'parent': ''}
    row.update(verdict.to_flat_dict())
    row['decision_text'] = ' | '.join(entry.describe() for entry in verdict.decision_path if entry.result is not None)
    numbers = dict(body.classifier_numbers.get(key) or {})
    row.update(numbers)
    if resonance.type == 'lidov_kozai':
        row.update(omega_free_gate.summary_fields(body.free_gates.get(key)))
        row.update(free_elements.summary_fields(body.free_elements))
        gate = body.zlk_gates.get(key)
        row['confirmed_ei'] = None if gate is None else gate.confirmed_ei
    return row


def analyse_one(example_id: str, figure: bool = False) -> Dict[str, Any]:
    """Analyse one example; never raises (errors come back in the result)."""
    warnings.simplefilter('ignore')
    start = time.perf_counter()
    try:
        example = bio.load_example(example_id, with_planets=False)
        sim, body, resonance = build_simulation(example)
        loaded = time.perf_counter()
        sim.identify_librations()
        analysed = time.perf_counter()
        key = resonance.to_s()
        result = {
            'example_id': example_id,
            'status': 'ok',
            'summary': [summary_row(example_id, body, resonance)],
            'episodes': [{'example_id': example_id, **row} for row in _episode_rows(body, key)],
            'seconds_load': loaded - start,
            'seconds_analyse': analysed - loaded,
        }
        if figure:
            result['figures'] = [str(_draw(example_id, sim, body, resonance))]
        result['seconds'] = time.perf_counter() - start
        return result
    except Exception as exc:  # noqa: BLE001 - one broken example must not stop the run
        return {'example_id': example_id, 'status': 'error', 'error': repr(exc), 'trace': traceback.format_exc()}


def _extra_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    """The columns of the body (free elements, e-i) that the analysis' own row does not have."""
    keep = ('free_', 'confirmed_ei', 'ei_')
    return {column: value for column, value in row.items() if column.startswith(keep)}


def _episode_rows(body, key: str) -> List[Dict[str, Any]]:
    analysis = getattr(body, 'secular_analyses', {}).get(key)
    return analysis.episode_rows() if analysis is not None else []


def _draw(example_id: str, sim, body, resonance) -> Path:
    import matplotlib.pyplot as plt

    from resonances.plotting.secular_plots import draw_long_period_combined

    path = bio.example_dir(example_id) / FIGURE
    row = summary_row(example_id, body, resonance)
    elements = body.free_elements if resonance.type == 'lidov_kozai' else None
    fig = draw_long_period_combined(sim.long_period_analysis(body, resonance), row=_extra_columns(row), free_elements=elements)
    bio.save_figure(fig, path, fig.dpi)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ agreement


def agreement(summary: pd.DataFrame) -> pd.DataFrame:
    """The verdict next to the author's verdict (where there is one) and the old bucket."""
    columns = ['example_id', 'status', 'angle_status', 'angle_subtype', 'confidence', 'reason', 'libration_origin', 'flags']
    frame = summary[[c for c in columns if c in summary]].copy()
    labels = bio.read_labels()[['example_id', 'proposed_bucket']]
    frame = frame.merge(labels, on='example_id', how='left')
    frame['bucket_status'] = frame['proposed_bucket'].fillna('').map(bio.bucket_status)
    frame['author_status'] = frame['example_id'].map(lambda e: AUTHOR_VERDICTS.get(e, ('', ''))[0])
    frame['author_source'] = frame['example_id'].map(lambda e: AUTHOR_VERDICTS.get(e, AUTHOR_HINTS.get(e, ('', '')))[1])
    frame['author_hint'] = frame['example_id'].map(lambda e: AUTHOR_HINTS.get(e, ('', ''))[0])
    labelled = frame['author_status'] != ''
    frame['author_agrees'] = (frame['status'] == frame['author_status']).where(labelled)
    has_bucket = frame['bucket_status'] != ''
    frame['bucket_agrees'] = (frame['status'] == frame['bucket_status']).where(has_bucket)
    return frame


def print_report(frame: pd.DataFrame, timings: List[Dict[str, Any]]) -> None:
    frame = frame.assign(family=frame['example_id'].str.split('__').str[0])
    print('\nstatus x family:')
    print(pd.crosstab(frame['status'], frame['family']).to_string())
    print('\nstatus x angle_status:')
    print(pd.crosstab(frame['status'], frame['angle_status']).to_string())
    labelled = frame[frame['author_status'] != '']
    for family, part in labelled.groupby('family'):
        print(f'author verdicts, {family}: {int(part.author_agrees.sum())}/{len(part)}')
    print(f'author verdicts, all: {int(labelled.author_agrees.sum())}/{len(labelled)}')
    shown = ['example_id', 'author_status', 'status', 'angle_status', 'confidence', 'reason', 'flags']
    print(labelled[shown].to_string(index=False))
    hints = frame[frame['author_hint'] != '']
    if len(hints):
        print('\nauthor hints (another record of the same body):')
        print(hints[['example_id', 'author_hint', 'status', 'angle_status', 'reason']].to_string(index=False))
    bucketed = frame[frame['bucket_status'] != '']
    for family, part in bucketed.groupby('family'):
        print(f'old bucket (hint), {family}: {int(part.bucket_agrees.sum())}/{len(part)}')
    print(pd.crosstab(bucketed['bucket_status'], bucketed['status'], margins=True).to_string())
    if timings:
        seconds = pd.Series([t['seconds'] for t in timings])
        print(f'\ntiming per example: median {seconds.median():.2f} s, max {seconds.max():.1f} s, sum {seconds.sum():.0f} s')


# ------------------------------------------------------------------ main


def select(rtypes: Optional[str] = None, only: Optional[str] = None) -> List[str]:
    ids = [i for i in bio.list_examples() if i.split('__', 1)[0] in RTYPES]
    return bio.select_ids(ids, only, rtypes, RTYPES, 'secular/ZLK')


def _job(args):
    return analyse_one(*args)


TASKS_PER_WORKER = 20  # worker recycling of the secular/ZLK runs


def run(ids: List[str], out: Path, workers: int, figures: bool) -> int:
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(ids)} secular/ZLK examples, {workers} workers -> {out}')
    ok, errors, wall = bio.run_examples(ids, _job, figures, workers, TASKS_PER_WORKER, workers <= 1, runs=[out / 'run.json'])
    summary = pd.DataFrame([row for r in ok for row in r['summary']])
    summary.to_csv(out / 'summary.csv', index=False)
    pd.DataFrame([row for r in ok for row in r['episodes']]).to_csv(out / 'episodes.csv', index=False)
    frame = agreement(summary) if len(summary) else pd.DataFrame()
    if len(frame):
        frame.to_csv(out / 'agreement.csv', index=False)
    timings = [{'example_id': r['example_id'], 'seconds': r['seconds']} for r in ok]
    bio.write_run_record(out, 'benchmark/tools/reclassify_secular_zlk.py', len(ids), ok, errors, wall, workers, figures, timings)
    if len(frame):
        print_report(frame, timings)
    return bio.report_errors(ok, errors, wall)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', help='comma-separated example ids')
    ap.add_argument('--rtype', help='comma-separated rtypes: secular, zlk (default both)')
    ap.add_argument('--out', type=Path, default=bio.DATA / 'v2-secular-zlk', help='output folder')
    ap.add_argument('--workers', type=int, default=3, help='worker processes (3 at most next to other runs)')
    ap.add_argument('--figures', action='store_true', help='also draw combined_v2.png into each example folder')
    args = ap.parse_args(argv)
    return run(select(args.rtype, args.only), args.out, args.workers, args.figures)


if __name__ == '__main__':
    sys.exit(main())
