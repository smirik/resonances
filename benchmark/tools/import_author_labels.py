"""Merge the author's export from review_page/ (author_labels_<date>.csv) into labels.csv.

Writes only author_* columns: author_bucket (the status in bucket spelling: non_resonant -> non-resonant),
author_confidence, author_angle_status, author_comment, author_discuss, author_agreed_with_claude,
author_labelled_at. claude_* and every other column are never touched (checked before writing).

Per row of the export:
  - unknown example_id -> reported, skipped;
  - author_labelled_at in labels.csv newer than the row's timestamp -> skipped (an older export never wins);
  - status set -> author_bucket, author_confidence, author_angle_status replaced (confidence may be empty:
    uncertain/invalid carry none);
  - status empty (only a comment or the discuss flag) -> author_bucket etc. kept;
  - author_comment, author_discuss, author_agreed_with_claude, author_labelled_at always taken from the row.
labels.csv is rewritten atomically under labels.lock (the lock tools/integrate.py takes).

Usage (from the repository root):
  uv run python benchmark/tools/import_author_labels.py                 # newest author_labels_*.csv
  uv run python benchmark/tools/import_author_labels.py FILE.csv [--dry-run] [--tree]
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as bio  # noqa: E402

BENCH, LABELS = bio.BENCH, bio.LABELS

STATUSES, CONFIDENCES, ANGLES = bio.STATUSES, bio.LABEL_CONFIDENCES, bio.ANGLE_STATUSES
AUTHOR_COLUMNS = [
    'author_bucket',
    'author_confidence',
    'author_angle_status',
    'author_comment',
    'author_discuss',
    'author_agreed_with_claude',
    'author_labelled_at',
]


def truthy(value: str) -> str:
    return '1' if str(value).strip().lower() in ('1', 'true', 'yes') else '0'


def read_export(path: Path) -> list[dict]:
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        lacking = [c for c in ('example_id', 'status') if c not in (reader.fieldnames or [])]
        if lacking:
            raise SystemExit(f'{path}: no column(s) {lacking}; is this an export of review_page?')
        return [{k: (v or '') for k, v in row.items() if k} for row in reader]


def validate(row: dict) -> list[str]:
    problems = []
    if row.get('status') and row['status'] not in STATUSES:
        problems.append(f'status={row["status"]!r}')
    if row.get('confidence') and row['confidence'] not in CONFIDENCES:
        problems.append(f'confidence={row["confidence"]!r}')
    if row.get('angle_status') and row['angle_status'] not in ANGLES:
        problems.append(f'angle_status={row["angle_status"]!r}')
    return problems


def merge(labels: list[dict], export: list[dict]) -> tuple[list[dict], collections.Counter, list[str]]:
    by_id = {row['example_id']: row for row in labels}
    stats: collections.Counter = collections.Counter()
    log: list[str] = []
    for row in export:
        eid = row.get('example_id', '').strip()
        if eid not in by_id:
            stats['unknown id'] += 1
            log.append(f'unknown example_id: {eid}')
            continue
        problems = validate(row)
        if problems:
            stats['invalid'] += 1
            log.append(f'{eid}: skipped, {", ".join(problems)}')
            continue
        target = by_id[eid]
        ts = row.get('timestamp', '').strip()
        current_ts = target.get('author_labelled_at', '')
        if current_ts and ts and current_ts > ts:
            stats['older than labels.csv'] += 1
            continue
        before = {c: target.get(c, '') for c in AUTHOR_COLUMNS}
        status = row.get('status', '').strip()
        if status:
            target['author_bucket'] = bio.status_bucket(status)
            target['author_confidence'] = row.get('confidence', '').strip()
            target['author_angle_status'] = row.get('angle_status', '').strip()
        target['author_comment'] = row.get('comment', '')
        target['author_discuss'] = truthy(row.get('discuss', ''))
        target['author_agreed_with_claude'] = truthy(row.get('agreed_with_claude', ''))
        target['author_labelled_at'] = ts or current_ts
        changed = {c: target.get(c, '') for c in AUTHOR_COLUMNS} != before
        stats['updated' if changed else 'unchanged'] += 1
        if status:
            stats[f'status {status}'] += 1
    return labels, stats, log


def newest_export() -> Path:
    files = sorted(BENCH.glob('author_labels_*.csv'), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f'no author_labels_*.csv in {BENCH}; pass the file explicitly')
    return files[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('csv', nargs='?', type=Path, help='export from review_page (default: newest author_labels_*.csv here)')
    ap.add_argument('--dry-run', action='store_true', help='report only, do not write labels.csv')
    ap.add_argument('--tree', action='store_true', help='rebuild data/review/ (bench_io.build_review_tree) afterwards')
    args = ap.parse_args(argv)
    path = args.csv or newest_export()
    export = read_export(path)

    with bio.labels_lock():
        with open(LABELS, newline='') as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            labels = list(reader)
        columns += [c for c in AUTHOR_COLUMNS if c not in columns]
        protected = [c for c in columns if c not in AUTHOR_COLUMNS]
        snapshot = [{c: row.get(c, '') for c in protected} for row in labels]

        labels, stats, log = merge(labels, export)

        if [{c: row.get(c, '') for c in protected} for row in labels] != snapshot:
            raise SystemExit('internal error: a non-author column changed; labels.csv not written')
        print(f'{path.name}: {len(export)} rows')
        for key in sorted(stats):
            print(f'  {key}: {stats[key]}')
        for line in log:
            print(f'  {line}')
        if args.dry_run:
            print('dry run: labels.csv not written')
            return 0
        tmp = LABELS.with_suffix('.csv.tmp')
        with open(tmp, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            for row in labels:
                writer.writerow({c: row.get(c, '') for c in columns})
        os.replace(tmp, LABELS)
    print(f'written {LABELS}')
    if args.tree:
        sys.path.insert(0, str(Path(__file__).parent))
        from bench_io import build_review_tree

        print(f'review tree: {build_review_tree()} links')
    return 0


if __name__ == '__main__':
    sys.exit(main())
