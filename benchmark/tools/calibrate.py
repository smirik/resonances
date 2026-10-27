"""Confidence calibration: the cells file of `analysis/confidence.py` from the author's verdicts on a benchmark run.

    uv run python benchmark/tools/calibrate.py                          # run benchmark/data/v2
    uv run python benchmark/tools/calibrate.py --run v2 --report tmp/W4-CALIBRATION.md
    uv run python benchmark/tools/calibrate.py --dry-run                # print, write no cells file

A cell is the key `confidence.cell_of` uses: (status, angle_status, reason) of the run's verdict. For every catalogue
row (no `parent`) that the author labelled, the verdict is an error when the run's status differs from the author's.

The author's status of an example, the newest source winning:
  1. candidates.csv `author_verdict` (read by `agree_author_verdicts.author_status`; controversial verdicts give none).
     A candidate's verdict was given on its own record (data_path), so it labels the candidate's example without a
     variant suffix only; `__20kyr`-like variants are other regimes (DECISIONS: a label belongs to the same record).
  2. author_labels_*.csv, the newest row per example, updated by author_label_decisions_*.csv (`reclassify.author_labels`).

Not counted, and listed as abstentions: a run verdict `uncertain` or `invalid` (it claims nothing) and an author
status `uncertain` (no truth to compare with).

What is counted for a cell (`n_examples`, `n_errors` and the Wilson 95 % interval `accuracy_95` in the file) are the
labelled verdicts at confidence `high` or `certain`, the strong claims. Medium and low verdicts of the cell carry a
weaker claim; they are reported (`n_labelled_all`, `n_errors_all`) but do not count. The package reports the cell of
every verdict in summary.csv (`cell_n_examples`, `cell_n_errors`, `cell_accuracy_low_95`, `cell_accuracy_high_95`,
`confidence.CellRecord`); the file changes no level (author, 2026-10-16: no minimum number of labels), so a
recalibration never changes the population it is measured on.

Writes the cells file (default resonances/data/confidence_cells.json, whose sha256 is `confidence_cells_sha` in a
run's CSV), <run>/calibration_rows.csv (every labelled row: truth, its source, error or abstention) and, with
--report, the human table.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as bio  # noqa: E402
from agree_author_verdicts import author_status  # noqa: E402
from reclassify import author_labels  # noqa: E402

from resonances.resonance.analysis import confidence as conf  # noqa: E402
from resonances.resonance.analysis.verdict import Confidence, Status  # noqa: E402

Z_95 = 1.959963984540054  # two-sided 95 % normal quantile of the Wilson interval
CALIBRATED_LEVELS = (Confidence.CERTAIN, Confidence.HIGH)  # the levels a cell counts (n_examples, n_errors)
ABSTAINING = (str(Status.UNCERTAIN), str(Status.INVALID))
CELL = ['status', 'angle_status', 'reason']


# ------------------------------------------------------------------ statistics


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """Wilson score interval of a binomial proportion (Wilson 1927, JASA 22, 209, eq. for the score interval;
    Brown, Cai & DasGupta 2001, Stat. Sci. 16, 101, eq. 4). (nan, nan) for n = 0."""
    if n <= 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


# ------------------------------------------------------------------ truth


def candidate_truth(candidates: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """example_id -> author status from candidates.csv `author_verdict`, on the candidate's own record only
    (an example id without a variant suffix: rtype__name__resonance)."""
    frame = labels[['example_id', 'candidate_id']].merge(
        candidates[['id', 'author_verdict']], left_on='candidate_id', right_on='id', how='inner'
    )
    frame['author_status'] = frame['author_verdict'].fillna('').map(author_status)
    frame = frame[frame['author_status'].notna() & (frame['example_id'].str.count('__') == 2)]
    frame = frame.assign(truth_source='candidates.csv: ' + frame['author_verdict'].str.slice(0, 60))
    return frame[['example_id', 'author_status', 'truth_source']].reset_index(drop=True)


def author_truth(candidates: pd.DataFrame, labels: pd.DataFrame, label_rows: pd.DataFrame) -> pd.DataFrame:
    """The author's status per example: candidates.csv verdicts, replaced by the newer label files
    (`label_rows` = `reclassify.author_labels()`: example_id, author_status, label_source)."""
    from_candidates = candidate_truth(candidates, labels)
    newer = label_rows[label_rows['author_status'].fillna('') != ''][['example_id', 'author_status', 'label_source']]
    newer = newer.rename(columns={'label_source': 'truth_source'})
    older = from_candidates[~from_candidates['example_id'].isin(newer['example_id'])]
    return pd.concat([older, newer], ignore_index=True)


# ------------------------------------------------------------------ cells


def labelled_rows(summary: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Catalogue rows of the run with the author's status: `counted` (a claim that can be wrong), `error`,
    `calibrated_level` (confidence high or certain) and `certain` (confidence certain)."""
    rows = summary.fillna('')
    if 'parent' in rows:
        rows = rows[rows['parent'] == '']
    frame = rows.merge(truth, on='example_id', how='inner')
    frame['abstention'] = ''
    frame.loc[frame['author_status'] == str(Status.UNCERTAIN), 'abstention'] = 'author_uncertain'
    frame.loc[frame['status'].isin(ABSTAINING), 'abstention'] = 'run_' + frame['status']
    frame['counted'] = frame['abstention'] == ''
    frame['error'] = frame['counted'] & (frame['status'] != frame['author_status'])
    frame['calibrated_level'] = frame['confidence'].isin([str(level) for level in CALIBRATED_LEVELS])
    frame['certain'] = frame['confidence'] == str(Confidence.CERTAIN)
    frame['family'] = frame['example_id'].str.split('__').str[0]
    return frame.reset_index(drop=True)


def cell_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One line per cell of the counted rows: n_examples/n_errors at the calibrated levels, the same for the
    certain rows and for all levels, the Wilson 95 % interval of the accuracy at the calibrated levels."""
    counted = rows[rows['counted']]
    records = []
    for key, group in counted.groupby(CELL, sort=True):
        calibrated = group[group['calibrated_level']]
        certain = group[group['certain']]
        n, errors = len(calibrated), int(calibrated['error'].sum())
        low, high = wilson_interval(n - errors, n)
        records.append(
            dict(
                zip(CELL, key),
                n_examples=n,
                n_errors=errors,
                accuracy=(n - errors) / n if n else math.nan,
                accuracy_low_95=low,
                accuracy_high_95=high,
                n_certain=len(certain),
                n_errors_certain=int(certain['error'].sum()),
                n_labelled_all=len(group),
                n_errors_all=int(group['error'].sum()),
                error_examples=sorted(calibrated.loc[calibrated['error'], 'example_id']),
                families=_families(calibrated),
            )
        )
    columns = CELL + [
        'n_examples',
        'n_errors',
        'accuracy',
        'accuracy_low_95',
        'accuracy_high_95',
        'n_certain',
        'n_errors_certain',
        'n_labelled_all',
        'n_errors_all',
        'error_examples',
        'families',
    ]
    table = pd.DataFrame(records, columns=columns)
    return table.sort_values(['n_examples', 'n_labelled_all'], ascending=False, kind='stable').reset_index(drop=True)


def _families(rows: pd.DataFrame) -> str:
    """'coorbital 70, mmr2 31': the resonance families behind a cell (the cell key does not hold the type)."""
    counts = rows['family'].value_counts()
    return ', '.join(f'{name} {count}' for name, count in counts.items())


def cells_document(table: pd.DataFrame, run: str, n_rows: int) -> dict:
    """The cells file: every cell with its counts (`confidence.load_cells` reads them for the cell_* columns)."""
    cells = []
    for row in table.to_dict('records'):
        entry = {key: row[key] for key in CELL + ['n_examples', 'n_errors']}
        if row['n_examples']:
            entry['accuracy_95'] = [round(row['accuracy_low_95'], 4), round(row['accuracy_high_95'], 4)]
        entry.update(n_labelled_all=int(row['n_labelled_all']), n_errors_all=int(row['n_errors_all']), error_examples=row['error_examples'])
        cells.append(entry)
    return {
        'version': 1,
        'note': (
            'Written by benchmark/tools/calibrate.py from the author verdicts. n_examples/n_errors/accuracy_95: labelled '
            'verdicts of the cell at confidence high or certain; uncertain/invalid verdicts and author-uncertain labels '
            'are abstentions. Reported in summary.csv (cell_* columns); the file changes no confidence level.'
        ),
        'run': run,
        'labelled_rows': n_rows,
        'cells': cells,
    }


def write_cells(document: dict, path: Path) -> str:
    """Write the cells file; returns its sha256 (the `confidence_cells_sha` of later runs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + '\n')
    return conf.file_sha(path)


def level_table(rows: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
    """Accuracy per confidence level, with Wilson 95 %."""
    counted = rows[rows['counted']].copy()
    counted['level'] = counted['confidence']
    keys = ['level'] if by is None else [by, 'level']
    records = []
    for key, group in counted.groupby(keys, sort=True):
        n, errors = len(group), int(group['error'].sum())
        low, high = wilson_interval(n - errors, n)
        key = key if isinstance(key, tuple) else (key,)
        records.append(dict(zip(keys, key), n=n, errors=errors, accuracy=(n - errors) / n, low_95=low, high_95=high))
    return pd.DataFrame(records)


# ------------------------------------------------------------------ report


def _fmt(value: float) -> str:
    return '' if value != value else f'{100 * value:.1f} %'


def report(rows: pd.DataFrame, table: pd.DataFrame, run: str, cells_path: str, sha: str) -> str:
    """The human table (markdown)."""
    counted = rows[rows['counted']]
    lines = [
        '# Confidence calibration',
        '',
        f'Run `benchmark/data/{run}`, {datetime.date.today().isoformat()}; cells file `{cells_path}`, sha256 `{sha}`.',
        '',
        f'- Labelled catalogue rows: {len(rows)}; counted {len(counted)}, errors {int(counted["error"].sum())}; '
        f'abstentions {int((~rows["counted"]).sum())} ({_abstentions(rows)}).',
        f'- Truth sources: {_sources(rows)}.',
        '- Counted per cell: verdicts at confidence high or certain. The package reports each verdict\'s cell '
        '(n, errors, Wilson 95 %) in summary.csv next to its level; the file changes no level.',
        '- The cell key (status, angle_status, reason) does not hold the resonance type: a cell pools the families '
        'listed in its last column.',
        '- `all levels n (errors)` is the same cell counted with its medium and low verdicts too.',
        '',
        '## Method',
        '',
        'An error is a counted verdict whose status differs from the author\'s. The author\'s status: candidates.csv '
        '`author_verdict` read by `agree_author_verdicts.author_status` (controversial verdicts give none), on the '
        'candidate\'s own record only (no variant suffix such as `__20kyr`: a label belongs to the record it was given '
        'on), replaced by the newer author_labels_*.csv rows and author_label_decisions_*.csv. Abstentions: a run '
        'verdict uncertain or invalid, an author label uncertain. A cell counts its verdicts at confidence high or '
        'certain; their level does not depend on the file.',
        '',
        '## Cells',
        '',
        '| status / angle_status / reason | n (high+) | errors | accuracy | Wilson 95 % | n certain (errors) '
        '| all levels n (errors) | families (high+) |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for row in table.to_dict('records'):
        cell = tuple(row[k] for k in CELL)
        lines.append(
            f'| {"/".join(cell)} | {row["n_examples"]} | {row["n_errors"]} | {_fmt(row["accuracy"])} | '
            f'{_fmt(row["accuracy_low_95"])} – {_fmt(row["accuracy_high_95"])} | {row["n_certain"]} '
            f'({row["n_errors_certain"]}) | {row["n_labelled_all"]} ({row["n_errors_all"]}) | {row["families"]} |'
        )
    lines += ['', '## Accuracy per level', '', '| level | n | errors | accuracy | Wilson 95 % |', '|---|---|---|---|---|']
    for row in level_table(rows).to_dict('records'):
        lines.append(
            f'| {row["level"]} | {row["n"]} | {row["errors"]} | {_fmt(row["accuracy"])} | '
            f'{_fmt(row["low_95"])} – {_fmt(row["high_95"])} |'
        )
    lines += ['', '## Accuracy per family and level', '', '| family | level | n | errors | Wilson 95 % |', '|---|---|---|---|---|']
    for row in level_table(rows, by='family').to_dict('records'):
        lines.append(
            f'| {row["family"]} | {row["level"]} | {row["n"]} | {row["errors"]} | ' f'{_fmt(row["low_95"])} – {_fmt(row["high_95"])} |'
        )
    errors = counted[counted['error']].sort_values(CELL + ['example_id'])
    lines += [
        '',
        '## Labelled errors',
        '',
        '| example | run: status / angle_status / reason, confidence | author | source |',
        '|---|---|---|---|',
    ]
    for row in errors.to_dict('records'):
        lines.append(
            f'| {row["example_id"]} | {row["status"]} / {row["angle_status"]} / {row["reason"]}, {row["confidence"] or "-"} '
            f'| {row["author_status"]} | {_cell_text(row["truth_source"])} |'
        )
    return '\n'.join(lines) + '\n'


def _cell_text(text: str) -> str:
    """Free text in a markdown table cell."""
    return str(text).replace('|', '/').replace('\n', ' ')


def _abstentions(rows: pd.DataFrame) -> str:
    counts = rows.loc[~rows['counted'], 'abstention'].value_counts()
    return ', '.join(f'{name} {count}' for name, count in counts.items()) or 'none'


def _sources(rows: pd.DataFrame) -> str:
    source = rows['truth_source'].str.split(':').str[0]
    return ', '.join(f'{name} {count}' for name, count in source.value_counts().items())


# ------------------------------------------------------------------ main


def read_run(run_dir: Path) -> pd.DataFrame:
    """summary.csv of the run plus secular_zlk/summary.csv when the run has it."""
    frames = [bio.read_summary(run_dir / 'summary.csv')]
    secular = run_dir / 'secular_zlk' / 'summary.csv'
    if secular.exists():
        frames.append(bio.read_summary(secular))
    return pd.concat(frames, ignore_index=True).fillna('')


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run', default='v2', help='run folder under benchmark/data (default v2)')
    parser.add_argument('--cells', type=Path, default=conf.PACKAGE_CELLS, help='cells file to write (default: the package file)')
    parser.add_argument('--report', type=Path, help='write the human table (markdown) here')
    parser.add_argument('--dry-run', action='store_true', help='print only; write no cells file and no rows CSV')
    args = parser.parse_args(argv)

    run_dir = bio.DATA / args.run
    summary = read_run(run_dir)
    candidates = pd.read_csv(bio.BENCH / 'candidates.csv', dtype=str).fillna('')
    truth = author_truth(candidates, bio.read_labels(), author_labels())
    rows = labelled_rows(summary, truth)
    table = cell_table(rows)
    document = cells_document(table, args.run, len(rows))

    if args.dry_run:
        with tempfile.TemporaryDirectory() as folder:
            sha = write_cells(document, Path(folder) / 'cells.json')
    else:
        sha = write_cells(document, args.cells)
        columns = ['example_id', 'family', *CELL, 'confidence', 'flags', 'author_status', 'truth_source', 'abstention', 'error']
        rows[columns].to_csv(run_dir / 'calibration_rows.csv', index=False)

    text = report(rows, table, args.run, str(args.cells), sha)
    if args.report:
        args.report.write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
