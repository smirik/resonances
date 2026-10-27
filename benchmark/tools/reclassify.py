"""Classify the benchmark's MMR examples with the v2 analysis, from the saved files only.

    uv run python benchmark/tools/reclassify.py                       # every mmr2, mmr3, coorbital example
    uv run python benchmark/tools/reclassify.py --only ID,ID --figures
    uv run python benchmark/tools/reclassify.py --rtype mmr3 --workers 8 --out benchmark/data/v2
    uv run python benchmark/tools/reclassify.py --rtype secular,zlk --workers 3 --out benchmark/data/v2-secular-zlk

--workers auto (the default) is min(CPUs - 2, available RAM / 3 GB, 8): ~2.6 GB per worker at the peak (v2-w21a); where
the memory limits it, the workers get the low-memory allocator and 1.6 GB each (LOW_MEMORY_ALLOCATOR, docs/config.md;
bench_io.auto_workers). The examples go longest first by the previous run's timings (bench_io.longest_first).
The CSVs are read through a binary copy (column_cache.py, <data>/cache/columns, the planets memory-mapped;
--no-column-cache parses them).

For each example `analyse_mmr` (resonances.resonance.analysis.mmr) reads series.csv.gz, meta.json and
planets.csv.gz and decides; nothing is integrated. Writes into OUT (default benchmark/data/v2):
summary.csv, mmr_numbers.csv, episodes.csv, candidates.csv, constants.csv (rows keyed by example_id),
agreement.csv (the verdict next to the labels) and run.json (commit, settings hash, timing). With
`--figures` the combined v2 figure is drawn too, as combined_v2.png next to combined.png in the example.

Agreement is printed against the author's labels: the newest row per example of author_labels_*.csv
(rows without a status are skipped), updated by author_label_decisions_*.csv (the author's later decisions,
DECISIONS.md). labels.csv proposed_bucket is shown as the old bucket.
"""

from __future__ import annotations

import os

# One BLAS thread per worker: the workers are the parallelism (see the ACF/OpenBLAS stall of Sept 2026).
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as bio  # noqa: E402

MMR_RTYPES = ('mmr2', 'mmr3', 'coorbital')
FIGURE = 'combined_v2.png'
DEFAULT_OUT = bio.DATA / 'v2'


# ------------------------------------------------------------------ one example


def analyse_one(example_id: str, figure: bool = False, settings_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyse one example (and draw its figure); never raises: errors come back in the result."""
    warnings.simplefilter('ignore')
    import column_cache
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.resonance.analysis.settings import settings_from_config

    column_cache.install()  # a binary copy of each CSV (BENCHMARK_COLUMN_CACHE=0: off)
    start = time.perf_counter()
    try:
        series = io.load_series(example_id)
        loaded = time.perf_counter()
        analysis = analyse_mmr(series, settings_from_config(**(settings_overrides or {})))
        analysed = time.perf_counter()
        analyses = [analysis] + list(analysis.derived)
        result = {
            'example_id': example_id,
            'status': 'ok',
            'summary': [_keyed(example_id, a.summary_row()) for a in analyses],
            'numbers': [_keyed(example_id, a.numbers_row()) for a in analyses],
            'episodes': [_keyed(example_id, row) for a in analyses for row in a.episode_rows()],
            'candidates': [_keyed(example_id, row) for row in analysis.candidate_rows()],
            'constants': [_keyed(example_id, row) for row in analysis.context.constants_rows()] if analysis.context is not None else [],
            'seconds_load': loaded - start,
            'seconds_analyse': analysed - loaded,
        }
        if figure:
            result['figures'] = [str(_draw(example_id, a, series, i)) for i, a in enumerate(analyses)]
        result['seconds'] = time.perf_counter() - start
        return result
    except Exception as exc:  # noqa: BLE001 - one broken example must not stop the run
        return {'example_id': example_id, 'status': 'error', 'error': repr(exc), 'trace': traceback.format_exc()}


def _keyed(example_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    return {'example_id': example_id, **row}


def _draw(example_id: str, analysis, series, index: int) -> Path:
    """combined_v2.png (a derived subresonance row: combined_v2__<key>.png) in the example folder."""
    import matplotlib.pyplot as plt

    from resonances.plotting.mmr_combined import draw_combined

    name = FIGURE if index == 0 else FIGURE.replace('.png', f'__{bio.safe_token(analysis.resonance.key)}.png')
    path = bio.example_dir(example_id) / name
    fig = draw_combined(analysis, analysis.derived_series or series)  # a derived row draws its own angle
    bio.save_figure(fig, path, fig.dpi)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ labels


def author_labels(bench: Optional[Path] = None) -> pd.DataFrame:
    """example_id -> status, confidence, angle_status of the author (newest export row, then later decisions).
    `bench`: the folder with the label files (default: benchmark/)."""
    bench = Path(bench) if bench is not None else bio.BENCH
    rows: Dict[str, Dict[str, str]] = {}
    for path in sorted(bench.glob('author_labels_*.csv')):
        for row in pd.read_csv(path, dtype=str).fillna('').to_dict('records'):
            if not row.get('status'):
                continue
            previous = rows.get(row['example_id'])
            if previous is None or row.get('timestamp', '') >= previous.get('timestamp', ''):
                rows[row['example_id']] = {**row, 'label_source': path.name}
    for path in sorted(bench.glob('author_label_decisions_*.csv')):
        for row in pd.read_csv(path, dtype=str).fillna('').to_dict('records'):
            rows[row['example_id']] = {**rows.get(row['example_id'], {}), **row, 'label_source': f'{path.name}: {row["source"]}'}
    frame = pd.DataFrame(rows.values())
    if frame.empty:
        return pd.DataFrame(columns=['example_id', 'author_status', 'author_confidence', 'author_angle_status', 'label_source'])
    frame = frame.rename(columns={'status': 'author_status', 'confidence': 'author_confidence', 'angle_status': 'author_angle_status'})
    return frame[['example_id', 'author_status', 'author_confidence', 'author_angle_status', 'label_source']]


def agreement(summary: pd.DataFrame, author: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The catalogue rows (not derived) next to the old bucket and the author's label (`author` default
    `author_labels()`)."""
    rows = summary[summary.get('parent', '').fillna('') == ''] if 'parent' in summary else summary
    columns = ['example_id', 'status', 'angle_status', 'angle_subtype', 'confidence', 'reason', 'libration_origin', 'flags']
    frame = rows[[c for c in columns if c in rows]].copy()
    labels = bio.read_labels()[['example_id', 'proposed_bucket']]
    author = author if author is not None else author_labels()
    frame = frame.merge(labels, on='example_id', how='left').merge(author, on='example_id', how='left')
    frame['old_status'] = frame['proposed_bucket'].fillna('').map(bio.status_spelling)
    labelled = frame['author_status'].fillna('') != ''
    frame['status_agrees'] = (frame['status'] == frame['author_status']).where(labelled)
    frame['angle_agrees'] = (frame['angle_status'] == frame['author_angle_status']).where(
        labelled & (frame['author_angle_status'].fillna('') != '')
    )
    frame['confidence_agrees'] = (frame['confidence'].fillna('') == frame['author_confidence'].fillna('')).where(labelled)
    return frame


def print_report(frame: pd.DataFrame, summary: pd.DataFrame, timings: List[Dict[str, Any]]) -> None:
    print('\nstatus distribution (catalogue rows):')
    print(frame['status'].value_counts().to_string())
    print('\nangle_status distribution:')
    print(frame['angle_status'].value_counts().to_string())
    print('\nstatus x confidence:')
    print(pd.crosstab(frame['status'], frame['confidence'].fillna('')).to_string())
    labelled = frame[frame['author_status'].fillna('') != '']
    if len(labelled):
        n = len(labelled)
        print(
            f'\nagreement with the author ({n} labels): status {int(labelled.status_agrees.sum())}/{n}, '
            f'angle_status {int(labelled.angle_agrees.eq(True).sum())}/{int(labelled.angle_agrees.notna().sum())}, '
            f'confidence {int(labelled.confidence_agrees.sum())}/{n}'
        )
        print(pd.crosstab(labelled['author_status'], labelled['status'], margins=True).to_string())
        wrong = labelled[labelled.status_agrees != True]  # noqa: E712 - NaN-aware
        if len(wrong):
            print('\nstatus differs:')
            cols = [
                'example_id',
                'author_status',
                'author_angle_status',
                'status',
                'angle_status',
                'confidence',
                'reason',
                'libration_origin',
            ]
            print(wrong[cols].to_string(index=False))
    if timings:
        seconds = pd.Series([t['seconds'] for t in timings])
        print(
            f'\ntiming per example: median {seconds.median():.2f} s, p90 {seconds.quantile(0.9):.2f} s, max {seconds.max():.1f} s, '
            f'sum {seconds.sum():.0f} s'
        )


# ------------------------------------------------------------------ main


def select(args) -> List[str]:
    from resonances.resonance.analysis.mmr import io

    return bio.select_ids(io.list_examples('mmr'), args.only, args.rtype, MMR_RTYPES, 'MMR')


def _write(frame_rows: List[Dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(frame_rows)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return frame


TASKS_PER_WORKER = 5  # worker recycling: W20 round E, peak RSS per worker on 40 rows with figures (DECISIONS.md)


def run(ids: List[str], out: Path, workers: int, figures: bool, tasks_per_worker: int = TASKS_PER_WORKER) -> int:
    """Analyse `ids` into `out`. Each worker process is replaced after `tasks_per_worker` examples (0: never): a process
    keeps the memory its largest record took, so recycling bounds only the growth between rows, not the peak of one
    row. The heavy rows set the peak whatever the recycling (W20 round E, 40 heaviest rows, one worker): 37087
    3J+8S-3 at 1 Myr alone 2.23 GB without a figure (~2.5 GB with it), 285637 at 2 Myr 1.72 GB, 50166 85J-27 1.45 GB;
    with recycling every 5 rows three of eight processes still went above 2 GB. With one worker and recycling the
    examples run in a pool of one process; `--workers 1 --tasks-per-worker 0` runs them in this process (debugging)."""
    out.mkdir(parents=True, exist_ok=True)
    print(f'{len(ids)} examples, {workers} workers, {tasks_per_worker or "no"} recycling -> {out}')
    in_process = workers <= 1 and tasks_per_worker <= 0
    ok, errors, wall = bio.run_examples(ids, _job, figures, workers, tasks_per_worker, in_process, runs=[out / 'run.json'])
    summary = _write([row for r in ok for row in r['summary']], out / 'summary.csv')
    for name in ('numbers', 'episodes', 'candidates', 'constants'):
        _write([row for r in ok for row in r[name]], out / ('mmr_numbers.csv' if name == 'numbers' else f'{name}.csv'))
    frame = agreement(summary) if len(summary) else pd.DataFrame()
    if len(frame):
        frame.to_csv(out / 'agreement.csv', index=False)
    timings = [
        {'example_id': r['example_id'], 'seconds': r['seconds'], 'load': r['seconds_load'], 'analyse': r['seconds_analyse']} for r in ok
    ]
    script = 'benchmark/tools/reclassify.py'
    bio.write_run_record(out, script, len(ids), ok, errors, wall, workers, figures, timings, tasks_per_worker=tasks_per_worker)
    if len(frame):
        print_report(frame, summary, timings)
    return bio.report_errors(ok, errors, wall)


def _job(args):
    return analyse_one(*args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', help='comma-separated example ids')
    ap.add_argument('--rtype', help=f'comma-separated rtypes (default {",".join(MMR_RTYPES)}); secular, zlk: reclassify_secular_zlk')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT, help='output folder (default benchmark/data/v2)')
    ap.add_argument(
        '--workers',
        default='auto',
        help='worker processes, or auto (default): min(CPUs - 2, available RAM / 3 GB (1.6 GB with the low-memory allocator), 8), '
        'bench_io.auto_workers',
    )
    ap.add_argument('--no-column-cache', action='store_true', help='parse the CSVs every time (column_cache.py: the binary copy)')
    ap.add_argument('--figures', action='store_true', help='also draw combined_v2.png into each example folder')
    ap.add_argument(
        '--tasks-per-worker',
        type=int,
        default=TASKS_PER_WORKER,
        help=f'replace a worker process after this many examples (default {TASKS_PER_WORKER}; 0 = never, and with --workers 1 '
        'run in this process)',
    )
    args = ap.parse_args()
    args.workers = bio.resolve_workers(args.workers)
    if args.no_column_cache:
        os.environ['BENCHMARK_COLUMN_CACHE'] = '0'  # the spawned workers inherit it
    rtypes = set((args.rtype or ','.join(MMR_RTYPES)).split(','))
    if rtypes & {'secular', 'zlk'}:  # secular and ZLK examples go through the package's own path
        import reclassify_secular_zlk

        if rtypes - {'secular', 'zlk'}:
            raise SystemExit('--rtype secular,zlk cannot be mixed with MMR rtypes in one run (their outputs differ)')
        ids = reclassify_secular_zlk.select(args.rtype, args.only)
        return reclassify_secular_zlk.run(ids, args.out, args.workers, args.figures)
    return run(select(args), args.out, args.workers, args.figures, args.tasks_per_worker)


if __name__ == '__main__':
    sys.exit(main())
