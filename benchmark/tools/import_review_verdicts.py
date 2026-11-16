"""The author's verdicts on a review folder (`sort_figures.py --review NAME`) as label rows and proposed episode cases.

    uv run python benchmark/tools/import_review_verdicts.py verdicts.json --review w28            # print the diff
    uv run python benchmark/tools/import_review_verdicts.py verdicts.json --review w28 --write    # apply it

The export is a JSON object, one entry per reviewed row, keyed by `example_id` (the record's own row) or
`example_id::resonance` (any row, e.g. a derived subresonance row):

    {"mmr2__12218__4M-7+0+3": {"verdict": "agree", "comment": ""},
     "zlk__10563__LK": {"verdict": "status", "status": "transient", "confidence": "medium", "angle_status": "episodic"},
     "coorbital__2002_PN__1E-1+0+0": {"episodes_note": "one episode 300-900 yr, not two", "comment": "..."}}

- `verdict: agree`: a label row with the run's verdict of that row (new_status, new_confidence, new_angle_status of
  the review CSV).
- `verdict: status`: a label row with the author's fields; a field he did not give keeps the run's value (an empty
  value is a value: no confidence for an uncertain row).
- no verdict: no label row (a comment or an episode note only).
- `episodes_note` (any verdict): a proposed episode case, tier A (his words), with empty checks: its ops are written
  by hand before it goes into EPISODE_EXPECTATIONS.csv, so the importer never touches the episode gate.

Label rows go to `author_label_decisions_2026-10-15.csv` (columns example_id, status, confidence, angle_status,
source; source `author:<date> (review <NAME>: ...)` with his comment), proposed cases to
`benchmark/review/<NAME>_PROPOSED_CASES.csv` (the columns of EPISODE_EXPECTATIONS.csv, ids R1, R2, ...). Without
--write the tool prints both as a unified diff and changes nothing. A key that is not a row of the review CSV, an
unknown verdict, status, confidence or angle_status rejects the whole export (nothing is written). A derived row takes
no label (labels are per record): it is listed and skipped. Idempotent: a label row identical to one in the file, or a
case with the same example_id and note, is not added again.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import difflib
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from resonances.resonance.analysis.verdict import AngleStatus, Confidence, Status

BENCH = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('RESONANCES_BENCHMARK_DATA') or BENCH / 'data').expanduser().resolve()
LABELS_CSV = BENCH / 'author_label_decisions_2026-10-15.csv'
CASES_CSV = BENCH / 'EPISODE_EXPECTATIONS.csv'
LABEL_COLUMNS = ['example_id', 'status', 'confidence', 'angle_status', 'source']
CASE_COLUMNS = ['case', 'example_id', 'tier', 'scope', 'mechanism', 'expected', 'checks', 'sources', 'known_fail', 'known_reason']
CASE_PREFIX = 'R'  # proposed case ids R1, R2, ... (EPISODE_EXPECTATIONS.csv uses E and C)
VERDICTS = ('agree', 'status')
LABEL_FIELDS = ('status', 'confidence', 'angle_status')
# The package's own spellings; a typo is rejected
ALLOWED = {'status': tuple(Status), 'confidence': tuple(Confidence), 'angle_status': tuple(AngleStatus)}
ENTRY_KEYS = {'verdict', 'status', 'confidence', 'angle_status', 'episodes_note', 'comment'}


class ExportError(ValueError):
    """The export cannot be imported as it is (unknown id, verdict or value); nothing was written."""


@dataclass
class Plan:
    labels: List[Dict[str, str]] = field(default_factory=list)  # rows to append to the label file
    cases: List[Dict[str, str]] = field(default_factory=list)  # rows to append to the proposed cases file
    skipped: List[str] = field(default_factory=list)  # entries that give nothing to write, with why


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """The rows of a CSV whose comment lines start with '#' (none when the file does not exist)."""
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding='utf-8').splitlines(keepends=True) if not line.startswith('#')]
    return [{k: (v or '') for k, v in row.items()} for row in csv.DictReader(io.StringIO(''.join(lines)))]


def review_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """(example_id, resonance) -> the review CSV line."""
    rows = read_csv_rows(path)
    if not rows:
        raise ExportError(f'{path}: no review rows (run sort_figures.py --review first)')
    return {(r['example_id'], r['resonance']): r for r in rows}


def resolve(key: str, rows: Mapping[Tuple[str, str], Dict[str, str]]) -> Optional[Dict[str, str]]:
    """The review row of an export key: `example_id::resonance`, or `example_id` for the record's own (not derived) row."""
    if '::' in key:
        example_id, resonance = key.split('::', 1)
        return rows.get((example_id, resonance))
    own = [r for (example_id, _), r in rows.items() if example_id == key and not r.get('parent')]
    return own[0] if len(own) == 1 else None


def validate(export: Mapping[str, Mapping], rows: Mapping[Tuple[str, str], Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """key -> review row; raises ExportError listing every problem of the export."""
    problems, resolved = [], {}
    for key, entry in export.items():
        row = resolve(key, rows)
        if row is None:
            problems.append(f'{key}: not a row of the review CSV')
            continue
        if not isinstance(entry, Mapping):
            problems.append(f'{key}: the entry is not an object')
            continue
        extra = set(entry) - ENTRY_KEYS
        if extra:
            problems.append(f'{key}: unknown fields {sorted(extra)}')
        verdict = str(entry.get('verdict') or '')
        if verdict and verdict not in VERDICTS:
            problems.append(f'{key}: verdict {verdict!r} is not one of {VERDICTS}')
        for name in LABEL_FIELDS:
            if name in entry and str(entry[name] or '') not in ALLOWED[name]:
                problems.append(f'{key}: {name} {entry[name]!r} is not one of {[str(v) for v in ALLOWED[name]]}')
        if verdict == 'status' and not any(name in entry for name in LABEL_FIELDS):
            problems.append(f'{key}: verdict status without status, confidence or angle_status')
        resolved[key] = row
    if problems:
        raise ExportError('the export is rejected, nothing written:\n  ' + '\n  '.join(problems))
    return resolved


def _clean(text) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def label_row(entry: Mapping, row: Mapping[str, str], name: str, date: str) -> Dict[str, str]:
    """The label file row of one entry with a verdict."""
    new = {f: row.get(f'new_{f}', '') for f in LABEL_FIELDS}
    if entry['verdict'] == 'agree':
        values = new
        what = f'agrees with the run verdict {"/".join(new.values())}'
    else:
        values = {f: str(entry[f] or '') if f in entry else new[f] for f in LABEL_FIELDS}
        what = f'corrects the run verdict {"/".join(new.values())} to {"/".join(values.values())}'
    comment = _clean(entry.get('comment'))
    source = f'author:{date} (review {name}: {what}' + (f'; comment: {comment}' if comment else '') + ')'
    return {'example_id': row['example_id'], **values, 'source': source}


def scope_of(example_id: str) -> str:
    """The EPISODE_EXPECTATIONS scope of an example (mmr also for co-orbitals)."""
    for prefix, scope in (('secular__', 'secular'), ('zlk__', 'zlk')):
        if example_id.startswith(prefix):
            return scope
    return 'mmr'


def case_row(entry: Mapping, row: Mapping[str, str], key: str, name: str, date: str, number: int, export_name: str) -> Dict[str, str]:
    """A proposed episode case: tier A (his note), empty checks to be written as ops by hand."""
    return {
        'case': f'{CASE_PREFIX}{number}',
        'example_id': row['example_id'],
        'tier': 'A',
        'scope': scope_of(row['example_id']),
        'mechanism': f'author review {name} {date} (proposed: write the checks)',
        'expected': _clean(entry['episodes_note']),
        'checks': '[]',
        'sources': f'{export_name}: {key}',
        'known_fail': '0',
        'known_reason': '',
    }


def plan(
    export: Mapping[str, Mapping],
    rows: Mapping[Tuple[str, str], Dict[str, str]],
    labels: Sequence[Mapping[str, str]],
    cases: Sequence[Mapping[str, str]],
    gate_cases: Sequence[Mapping[str, str]],
    name: str,
    date: str,
    export_name: str = 'verdicts.json',
) -> Plan:
    """What the export adds to the label file (`labels`: its rows) and to the proposed cases (`cases`; `gate_cases`:
    EPISODE_EXPECTATIONS.csv, for the idempotency check). Raises ExportError on an invalid export."""
    resolved = validate(export, rows)
    out = Plan()
    have = {tuple(r.get(c, '') for c in LABEL_COLUMNS) for r in labels}
    noted = {(r.get('example_id', ''), _clean(r.get('expected'))) for r in [*cases, *gate_cases]}
    numbers = [int(m.group(1)) for r in cases if (m := re.fullmatch(rf'{CASE_PREFIX}(\d+)', r.get('case', '')))]
    number = max(numbers, default=0)
    for key in sorted(export):
        entry, row = export[key], resolved[key]
        derived = bool(row.get('parent'))
        if entry.get('verdict'):
            if derived:
                out.skipped.append(f'{key}: a derived row takes no label (labels are per record)')
            else:
                new = label_row(entry, row, name, date)
                if tuple(new[c] for c in LABEL_COLUMNS) in have:
                    out.skipped.append(f'{key}: label row already in the file')
                else:
                    have.add(tuple(new[c] for c in LABEL_COLUMNS))
                    out.labels.append(new)
        note = _clean(entry.get('episodes_note'))
        if note:
            if (row['example_id'], note) in noted:
                out.skipped.append(f'{key}: episode case already proposed')
            else:
                number += 1
                noted.add((row['example_id'], note))
                out.cases.append(case_row(entry, row, key, name, date, number, export_name))
        if not entry.get('verdict') and not note:
            out.skipped.append(f'{key}: no verdict and no episode note (comment only: {_clean(entry.get("comment"))!r})')
    return out


def appended(path: Path, rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> Tuple[str, str]:
    """(the file's text, the text with the rows appended; a header first when the file is new)."""
    old = path.read_text(encoding='utf-8') if path.exists() else ''
    if not rows:
        return old, old
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator='\n')
    if not old:
        writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, '') for c in columns})
    separator = '' if not old or old.endswith('\n') else '\n'
    return old, old + separator + buffer.getvalue()


def diff(path: Path, old: str, new: str) -> str:
    return ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f'a/{path.name}', f'b/{path.name}', n=0))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('export', help="the author's verdict export (JSON)")
    ap.add_argument('--review', required=True, help='the review name (data/figures/review-NAME/review-NAME.csv)')
    ap.add_argument('--review-csv', help='the review CSV (default from --review and --data)')
    ap.add_argument('--data', default=str(DATA), help='the data root (default benchmark/data or RESONANCES_BENCHMARK_DATA)')
    ap.add_argument('--labels', default=str(LABELS_CSV), help='the label file to append to')
    ap.add_argument('--cases', help='the proposed cases file (default benchmark/review/<NAME>_PROPOSED_CASES.csv)')
    ap.add_argument(
        '--gate-cases', default=str(CASES_CSV), help='EPISODE_EXPECTATIONS.csv (read only: a note already there is not proposed)'
    )
    ap.add_argument('--date', default=datetime.date.today().isoformat(), help='the date of the answers (label_source author:<date>)')
    ap.add_argument('--write', action='store_true', help='apply the diff (default: print it only)')
    args = ap.parse_args(argv)
    name = args.review.strip()
    review_csv = Path(args.review_csv) if args.review_csv else Path(args.data) / 'figures' / f'review-{name}' / f'review-{name}.csv'
    labels_path = Path(args.labels)
    cases_path = Path(args.cases) if args.cases else BENCH / 'review' / f'{name.upper()}_PROPOSED_CASES.csv'
    export_path = Path(args.export)
    try:
        export = json.loads(export_path.read_text(encoding='utf-8'))
        if not isinstance(export, dict):
            raise ExportError(f'{export_path}: the export is not a JSON object')
        result = plan(
            export,
            review_rows(review_csv),
            read_csv_rows(labels_path),
            read_csv_rows(cases_path),
            read_csv_rows(Path(args.gate_cases)),
            name,
            args.date,
            export_path.name,
        )
    except (ExportError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 2
    changes = [
        (labels_path, *appended(labels_path, result.labels, LABEL_COLUMNS)),
        (cases_path, *appended(cases_path, result.cases, CASE_COLUMNS)),
    ]
    for path, old, new in changes:
        if old != new:
            print(diff(path, old, new), end='')
    for line in result.skipped:
        print(f'skipped: {line}')
    print(
        f'{len(result.labels)} label rows, {len(result.cases)} proposed episode cases' + ('' if args.write else ' (dry: --write applies)')
    )
    if args.write:
        for path, old, new in changes:
            if old != new:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new, encoding='utf-8')
                print(f'written: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
