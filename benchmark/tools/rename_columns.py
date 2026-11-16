"""Rename the short column names of the saved benchmark files to readable ones (2026-10-15).

    uv run python benchmark/tools/rename_columns.py                    # all of data/ (or $RESONANCES_BENCHMARK_DATA)
    uv run python benchmark/tools/rename_columns.py --root benchmark/dryrun/data --dry-run

series.csv.gz: 'lam' -> 'mean_longitude'. planets.csv.gz: '<Planet>.l' -> '<Planet>.mean_longitude'. Only the header
line is rewritten: the data lines are copied byte for byte, so the %.12g formatting stays. Every rewritten file is
checked before it replaces the old one: same data bytes (sha256) and same row count, and for a sample of files the
values are compared once more through pandas. meta.json ('columns', 'provenance.source_columns') and run.json
('planets_columns') follow. Symlinked planets.csv.gz are skipped (their target in data/runs is rewritten).
Idempotent: a file already in the new names is left untouched.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as io  # noqa: E402

CHUNK = 1 << 22


def rename_series_columns(columns: List[str]) -> List[str]:
    return [io.LEGACY_SERIES_COLUMNS.get(c, c) for c in columns]


def rename_planet_columns(columns: List[str]) -> List[str]:
    out = []
    for c in columns:
        for old, new in io.LEGACY_PLANET_SUFFIXES.items():
            if c.endswith(old):
                c = c[: -len(old)] + new
        out.append(c)
    return out


def _body_digest(handle) -> tuple[str, int]:
    """sha256 and line count of everything after the header line."""
    digest, rows = hashlib.sha256(), 0
    for chunk in iter(lambda: handle.read(CHUNK), b''):
        digest.update(chunk)
        rows += chunk.count(b'\n')
    return digest.hexdigest(), rows


def _same_values(old: Path, new: Path, renamed: List[str]) -> bool:
    a, b = pd.read_csv(old, compression='gzip'), pd.read_csv(new, compression='gzip')
    if len(a) != len(b) or list(b.columns) != renamed:
        return False
    return all(np.array_equal(a[c].to_numpy(), b[n].to_numpy(), equal_nan=a[c].dtype.kind == 'f') for c, n in zip(a.columns, b.columns))


def rewrite_header(path: Path, rename: Callable[[List[str]], List[str]], dry_run: bool = False, check_values: bool = False) -> str:
    """Rewrite the header of one gzip CSV; returns 'renamed', 'unchanged' or raises on a failed check."""
    with gzip.open(path, 'rb') as f:
        header = f.readline()
    ending = b'\r\n' if header.endswith(b'\r\n') else b'\n'
    columns = header[: -len(ending)].decode().split(',')
    renamed = rename(columns)
    if renamed == columns:
        return 'unchanged'
    if len(set(renamed)) != len(renamed):
        raise ValueError(f'{path}: renaming {columns} gives duplicate columns')
    if dry_run:
        return 'renamed'
    new_header = ','.join(renamed).encode() + ending
    tmp = path.with_name(path.name + '.tmp')
    try:
        with gzip.open(path, 'rb') as src, open(tmp, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=6, mtime=0) as dst:
            src.readline()
            dst.write(new_header)
            for chunk in iter(lambda: src.read(CHUNK), b''):
                dst.write(chunk)
        with gzip.open(path, 'rb') as f:
            f.readline()
            old_digest, old_rows = _body_digest(f)
        with gzip.open(tmp, 'rb') as f:
            if f.readline() != new_header:
                raise ValueError(f'{path}: header not written')
            new_digest, new_rows = _body_digest(f)
        if (old_digest, old_rows) != (new_digest, new_rows):
            raise ValueError(f'{path}: data changed ({old_rows} -> {new_rows} rows)')
        if check_values and not _same_values(path, tmp, renamed):
            raise ValueError(f'{path}: values differ after renaming')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return 'renamed'


def _rename_keys(mapping: dict, rename: Callable[[List[str]], List[str]]) -> dict:
    return dict(zip(rename(list(mapping)), mapping.values()))


def update_json(path: Path, dry_run: bool = False) -> str:
    with open(path) as f:
        data = json.load(f)
    before = json.dumps(data, sort_keys=False)
    if isinstance(data.get('columns'), list):
        data['columns'] = rename_series_columns(data['columns'])
    source_columns = (data.get('provenance') or {}).get('source_columns')
    if isinstance(source_columns, dict):
        data['provenance']['source_columns'] = _rename_keys(source_columns, rename_series_columns)
    if isinstance(data.get('planets_columns'), list):
        data['planets_columns'] = rename_planet_columns(data['planets_columns'])
    if json.dumps(data, sort_keys=False) == before:
        return 'unchanged'
    if not dry_run:
        io.write_json(path, data)
    return 'renamed'


def _job(args):
    kind, path, dry_run, check_values = args
    if kind == 'json':
        return kind, str(path), update_json(path, dry_run)
    rename = rename_series_columns if kind == 'series' else rename_planet_columns
    return kind, str(path), rewrite_header(path, rename, dry_run, check_values)


def collect(root: Path, sample_every: int) -> List[tuple]:
    jobs = []
    for i, path in enumerate(sorted(root.glob('examples/*/series.csv.gz'))):
        jobs.append(('series', path, i % sample_every == 0 if sample_every else False))
    for path in sorted(root.glob('**/planets.csv.gz')):
        if not path.is_symlink():
            jobs.append(('planets', path, True))
    for path in sorted(root.glob('examples/*/meta.json')) + sorted(root.glob('runs/*/run.json')):
        jobs.append(('json', path, False))
    return jobs


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=io.DATA, help='data root (default: %(default)s)')
    ap.add_argument('--dry-run', action='store_true', help='only report what would change')
    ap.add_argument('--n-cores', type=int, default=8)
    ap.add_argument('--sample-every', type=int, default=20, help='compare values through pandas for every N-th series (0: none)')
    args = ap.parse_args(argv)

    jobs = [(kind, path, args.dry_run, check) for kind, path, check in collect(args.root.resolve(), args.sample_every)]
    counts: dict = {}
    failures = []
    with ProcessPoolExecutor(max_workers=args.n_cores) as pool:
        # CSV files first; the JSON column lists follow only if every CSV passed its check.
        for group in ([j for j in jobs if j[0] != 'json'], [j for j in jobs if j[0] == 'json']):
            if failures:
                break
            futures = [pool.submit(_job, job) for job in group]
            for job, future in zip(group, futures):
                try:
                    kind, _, result = future.result()
                except Exception as exc:  # report every failure, keep going: each file is replaced only after its check
                    failures.append(f'{job[1]}: {exc}')
                    continue
                counts[(kind, result)] = counts.get((kind, result), 0) + 1
    for (kind, result), n in sorted(counts.items()):
        print(f'{kind:8s} {result:10s} {n}')
    for line in failures:
        print('FAILED', line)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
