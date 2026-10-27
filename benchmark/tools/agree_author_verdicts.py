"""Agreement of a reclassify run with every unambiguous author verdict in candidates.csv (`author_verdict`).

    uv run python benchmark/tools/agree_author_verdicts.py [run]      # run = folder under benchmark/data (default v2)

Controversial verdicts are skipped, and so is every record with its own author label (a non-empty status in
author_labels_*.csv / author_label_decisions_*.csv, `reclassify.author_labels`): the label is the newer verdict on the
same record and is compared on its own (R-AV, author 2026-10-24). Writes <run>/agreement_author_verdicts.csv and prints the cross tables.
"""

import re
import sys
from typing import Optional

import pandas as pd

import bench_io as bio

STATUS = {
    'non-resonant': 'non_resonant',
    'resonant': 'resonant',
    'pure libration, resonant': 'resonant',
    'transient': 'transient',
    'uncertain': 'uncertain',
    'near-separatrix': 'near_separatrix',
}


def author_status(verdict: str):
    verdict = verdict.strip().lower()
    if not verdict or verdict.startswith('controversial'):
        return None
    return STATUS.get(re.split(r'[:;(|]', verdict)[0].strip())


def without_labelled(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The rows of `frame` whose example_id has no author label with a non-empty status (`labels`: example_id,
    author_status as `reclassify.author_labels` gives them)."""
    labelled = labels.loc[labels['author_status'].fillna('').astype(str).str.strip() != '', 'example_id']
    return frame[~frame['example_id'].isin(set(labelled))].copy()


def agreement_frame(run: str, author: Optional[pd.DataFrame] = None, summary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Catalogue rows of the run (no derived rows) with an unambiguous author verdict and no author label of their own
    (`without_labelled`; `author` default `reclassify.author_labels()`): status, author, ok, family. `summary`: the
    run's summary.csv when the caller has read it."""
    summary = summary if summary is not None else bio.read_summary(bio.DATA / run / 'summary.csv')
    summary = summary[summary['parent'] == ''] if 'parent' in summary else summary
    labels = bio.read_labels()[['example_id', 'candidate_id']]
    candidates = pd.read_csv(bio.BENCH / 'candidates.csv', dtype=str).fillna('')[['id', 'author_verdict']]
    frame = summary.merge(labels, on='example_id', how='left').merge(candidates, left_on='candidate_id', right_on='id', how='left')
    frame['author'] = frame['author_verdict'].fillna('').map(author_status)
    frame = frame[frame['author'].notna()].copy()
    if author is None:
        import reclassify

        author = reclassify.author_labels()
    frame = without_labelled(frame, author)
    frame['ok'] = frame['status'] == frame['author']
    frame['family'] = frame['example_id'].str.split('__').str[0]
    return frame


def main() -> int:
    run = sys.argv[1] if len(sys.argv) > 1 else 'v2'
    frame = agreement_frame(run)
    print(f'{run}: {int(frame["ok"].sum())}/{len(frame)} statuses agree with the author')
    print(frame.groupby('family')['ok'].agg(['sum', 'count']))
    print(pd.crosstab(frame['author'], frame['status']))
    print(pd.crosstab([frame['status'], frame['confidence']], frame['ok']))
    columns = ['example_id', 'family', 'status', 'angle_status', 'confidence', 'reason', 'flags', 'author', 'ok']
    frame[columns].to_csv(bio.DATA / run / 'agreement_author_verdicts.csv', index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
