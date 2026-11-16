"""The exact gate of `make test-benchmark`: a run's disagreements with the author against benchmark/KNOWN_DISAGREEMENTS.csv.

    uv run python benchmark/tools/known_disagreements.py v2-w21a            # check a run folder (name under benchmark/data or a path)
    uv run python benchmark/tools/known_disagreements.py v2-w21a --write    # rewrite the list from the run, keeping the reasons
    uv run python benchmark/tools/known_disagreements.py v2-w21a --secular-zlk v2-w23-secular-zlk   # with the secular/ZLK run

A disagreement is one (example_id, field) where the package and the author differ, only where the author gave a value:
- status, angle_status, confidence: the author's labels (author_labels_*.csv, then author_label_decisions_*.csv;
  `reclassify.author_labels`) on the run's catalogue rows; the confidence only where the author gave a level;
- author_verdict: the unambiguous `author_verdict` of candidates.csv against the status (`agree_author_verdicts`).
Labels made by Claude never count.
The secular and Lidov-Kozai examples are in a run of their own (`reclassify_secular_zlk.py`, --secular-zlk): there the
same author labels, and as author_verdict the author's verdicts `reclassify_secular_zlk.AUTHOR_VERDICTS` (skipped where
the record has its own label, as above; a record missing from the run counts as 'missing'). Without that run the list's
secular/ZLK rows are neither checked nor dropped.

The check fails on a disagreement that is not in the list (new), on a listed one that now agrees (resolved: update the
list) and on a listed one whose package or author value differs from the recorded one (changed). The list's category and
reason are Claude's reading of each case (benchmark/LAYOUT.md), not an author label.
"""

from __future__ import annotations

import argparse
import datetime
import io as _io
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as bio  # noqa: E402

KNOWN = bio.BENCH / 'KNOWN_DISAGREEMENTS.csv'
FIELDS = ('status', 'angle_status', 'confidence', 'author_verdict')
COLUMNS = ['example_id', 'field', 'author', 'package', 'category', 'reason', 'since']
KEY = ['example_id', 'field']
UNREVIEWED = 'unreviewed'
# the label column of reclassify.agreement for each field read from the author's labels
LABEL_FIELDS = {'status': 'author_status', 'angle_status': 'author_angle_status', 'confidence': 'author_confidence'}


def _text(series: pd.Series) -> pd.Series:
    return series.fillna('').astype(str).str.strip()


def _compared(labels: pd.DataFrame, verdicts: pd.DataFrame):
    """(field, example_id, author, package, given) per field: the author's value against the package's, `given` where
    the author gave one (`disagreements` and `agreement_numbers` read the same rows)."""
    for field, label in LABEL_FIELDS.items():
        if label not in labels or field not in labels:
            continue
        author, package = _text(labels[label]), _text(labels[field])
        given = (author != '') & (_text(labels.get('author_status', author)) != '')
        yield field, labels['example_id'], author, package, given
    if len(verdicts):
        author = _text(verdicts['author'])
        yield 'author_verdict', verdicts['example_id'], author, _text(verdicts['status']), author != ''


def disagreements(labels: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Every (example_id, field) where the package differs from the author.

    labels: `reclassify.agreement` rows (example_id, status, angle_status, confidence and their author_* columns);
    verdicts: `agree_author_verdicts.agreement_frame` rows (example_id, status, author)."""
    parts = []
    for field, example_id, author, package, given in _compared(labels, verdicts):
        differs = given & (author != package)
        parts.append(
            pd.DataFrame({'example_id': example_id[differs], 'field': field, 'author': author[differs], 'package': package[differs]})
        )
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY + ['author', 'package'])
    return _sorted(frame.drop_duplicates(KEY))


def agreement_numbers(labels: pd.DataFrame, verdicts: pd.DataFrame) -> Dict[str, tuple]:
    """(agreeing, compared) per field, on the rows where the author gave a value (as `disagreements` counts)."""
    return {
        field: (int((given & (author == package)).sum()), int(given.sum()))
        for field, _, author, package, given in _compared(labels, verdicts)
    }


def compare(current: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    """The differences between a run's disagreements and the list: one row per problem, `problem` = new | resolved |
    changed, with the recorded (author_known, package_known) and the run's values (author_now, package_now)."""
    merged = current[KEY + ['author', 'package']].merge(
        known[KEY + ['author', 'package', 'category', 'reason']], on=KEY, how='outer', suffixes=('_now', '_known'), indicator=True
    )
    for column in ('author_now', 'package_now', 'author_known', 'package_known', 'category', 'reason'):
        merged[column] = _text(merged[column])
    merged['problem'] = ''
    merged.loc[merged['_merge'] == 'left_only', 'problem'] = 'new'
    merged.loc[merged['_merge'] == 'right_only', 'problem'] = 'resolved'
    both = merged['_merge'] == 'both'
    changed = both & ((merged['author_now'] != merged['author_known']) | (merged['package_now'] != merged['package_known']))
    merged.loc[changed, 'problem'] = 'changed'
    problems = merged[merged['problem'] != '']
    columns = ['problem'] + KEY + ['author_known', 'package_known', 'author_now', 'package_now', 'category', 'reason']
    return _sorted(problems[columns])


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    order = {field: i for i, field in enumerate(FIELDS)}
    keys = frame.assign(_f=frame['field'].map(order))
    return frame.loc[keys.sort_values(['example_id', '_f']).index].reset_index(drop=True)


# ------------------------------------------------------------------ files


def read_known(path: Path = KNOWN) -> pd.DataFrame:
    """The list (leading '#' lines are its header comment)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    text = Path(path).read_text()
    body = '\n'.join(line for line in text.splitlines() if not line.startswith('#'))
    if not body.strip():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(_io.StringIO(body), dtype=str, keep_default_na=False)
    return frame[COLUMNS]


def header_comment(path: Path = KNOWN) -> str:
    if not Path(path).exists():
        return ''
    lines = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith('#'):
            break
        lines.append(line)
    return '\n'.join(lines) + '\n' if lines else ''


def write_known(current: pd.DataFrame, since: str, path: Path = KNOWN) -> pd.DataFrame:
    """Write `current` as the list, keeping category, reason and since of the rows already listed; a new row is
    `unreviewed` with `since` (the run that recorded it)."""
    known = read_known(path)
    frame = current[KEY + ['author', 'package']].merge(known[KEY + ['category', 'reason', 'since']], on=KEY, how='left')
    frame['category'] = _text(frame['category']).replace('', UNREVIEWED)
    frame['reason'] = _text(frame['reason'])
    frame['since'] = _text(frame['since']).replace('', since)
    frame = _sorted(frame[COLUMNS])
    comment = header_comment(path)
    with open(path, 'w') as handle:
        handle.write(comment)
        frame.to_csv(handle, index=False)
    return frame


def run_folder(run: str | Path) -> Path:
    path = Path(run)
    return path if path.is_dir() else bio.DATA / str(run)


LONG_PERIOD = ('secular', 'zlk')  # the rtypes of reclassify_secular_zlk.py, not in a reclassify.py run


def is_long_period(example_id: str) -> bool:
    return str(example_id).split('__', 1)[0] in LONG_PERIOD


def long_period_verdicts(summary: pd.DataFrame, author: pd.DataFrame, verdicts: Optional[Dict[str, tuple]] = None) -> pd.DataFrame:
    """The author's secular/ZLK verdicts (`reclassify_secular_zlk.AUTHOR_VERDICTS`: example_id -> (status, source)) next
    to the run's status, in the shape of `agree_author_verdicts.agreement_frame` (example_id, status, author); a record
    missing from the run has the status 'missing'; records with their own author label are left to the label."""
    import agree_author_verdicts as agree

    if verdicts is None:
        import reclassify_secular_zlk as szlk

        verdicts = szlk.AUTHOR_VERDICTS
    status = dict(zip(summary['example_id'], _text(summary['status']))) if len(summary) else {}
    frame = pd.DataFrame(
        {
            'example_id': list(verdicts),
            'status': [status.get(example_id, 'missing') for example_id in verdicts],
            'author': [value[0] for value in verdicts.values()],
        }
    )
    return agree.without_labelled(frame, author)


def run_disagreements(run: str | Path, secular_zlk: str | Path | None = None):
    """(disagreements, agreement numbers) of a reclassify output folder, against the labels as they are now; with
    `secular_zlk` (a reclassify_secular_zlk output folder) its rows too, their numbers under 'secular/zlk <field>'."""
    import agree_author_verdicts as agree
    import reclassify

    folder = run_folder(run)
    summary, author = bio.read_summary(folder / 'summary.csv'), reclassify.author_labels()  # read once for both views
    labels = reclassify.agreement(summary, author)
    verdicts = agree.agreement_frame(str(folder), author, summary)
    current, numbers = disagreements(labels, verdicts), agreement_numbers(labels, verdicts)
    if secular_zlk is not None:
        long_summary = bio.read_summary(run_folder(secular_zlk) / 'summary.csv')
        long_labels, long_verdicts = reclassify.agreement(long_summary, author), long_period_verdicts(long_summary, author)
        current = _sorted(pd.concat([current, disagreements(long_labels, long_verdicts)], ignore_index=True))
        for field, value in agreement_numbers(long_labels, long_verdicts).items():
            if value[1]:
                numbers[f'secular/zlk {field}'] = value
    return current, numbers


def checked_known(known: pd.DataFrame, secular_zlk) -> pd.DataFrame:
    """The listed rows a run can check: without its secular/ZLK run, the MMR and co-orbital rows only."""
    # astype(bool): on an empty list the mask is an empty object Series, which pandas reads as a column selection
    return known if secular_zlk is not None else known[~known['example_id'].map(is_long_period).astype(bool)]


def format_numbers(numbers: Dict[str, tuple]) -> str:
    return ', '.join(f'{field} {ok}/{n}' for field, (ok, n) in numbers.items())


def format_problems(problems: pd.DataFrame) -> str:
    if problems.empty:
        return 'no differences from the list'
    hint = {
        'new': 'not in the list: fix the code, or add it (with a reason) by --write',
        'resolved': 'listed but agrees now: remove it from the list (--write)',
        'changed': 'listed with other values: check, then --write',
    }
    lines = [f'{len(problems)} differences from {KNOWN.name}:']
    for problem, rows in problems.groupby('problem', sort=False):
        lines.append(f'\n{problem} ({hint[problem]}):')
        lines.append(rows.drop(columns=['problem']).to_string(index=False, max_colwidth=60))
    return '\n'.join(lines)


def check(run: str | Path, known_path: Path = KNOWN, verbose: bool = True, secular_zlk: str | Path | None = None) -> pd.DataFrame:
    """Compare a run (and its secular/ZLK run, when given) with the list; returns the problems (empty: the run matches
    the list exactly)."""
    current, numbers = run_disagreements(run, secular_zlk)
    problems = compare(current, checked_known(read_known(known_path), secular_zlk))
    if verbose:
        print(f'{run_folder(run).name}: agreement with the author: {format_numbers(numbers)}; {len(current)} disagreements')
        print(format_problems(problems))
    return problems


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='reclassify output: folder name under benchmark/data, or a path')
    ap.add_argument('--write', action='store_true', help=f'rewrite {KNOWN.name} from this run, keeping category, reason and since')
    ap.add_argument('--list', type=Path, default=KNOWN, help='the list to check against (default benchmark/KNOWN_DISAGREEMENTS.csv)')
    ap.add_argument('--secular-zlk', help='the reclassify_secular_zlk.py output of the same code: its rows are checked too')
    args = ap.parse_args(argv)
    if args.write:
        current, numbers = run_disagreements(args.run, args.secular_zlk)
        if args.secular_zlk is None:  # the secular/ZLK rows were not checked: keep them as they are
            known = read_known(args.list)
            current = pd.concat([current, known.loc[known['example_id'].map(is_long_period), KEY + ['author', 'package']]])
        frame = write_known(current, since=run_folder(args.run).name or datetime.date.today().isoformat(), path=args.list)
        print(
            f'wrote {len(frame)} rows to {args.list} ({int((frame["category"] == UNREVIEWED).sum())} unreviewed); {format_numbers(numbers)}'
        )
        return 0
    return 1 if len(check(args.run, args.list, secular_zlk=args.secular_zlk)) else 0


if __name__ == '__main__':
    raise SystemExit(main())
