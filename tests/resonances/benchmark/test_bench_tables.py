"""benchmark/tools/bench_io.py: the one table of old buckets and the status lists the label tools read without loading
the package. Checked against the package's enums and against the tables the tools kept before W23 (copied here as
they were: reclassify_secular_zlk.BUCKET_STATUS, build_review_page.BUCKET_TO_STATUS, reclassify.STATUS_SPELLING)."""

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import bench_io as bio  # noqa: E402

from resonances.resonance.analysis.verdict import AngleStatus, Confidence, Status  # noqa: E402

OLD_BUCKET_STATUS = {
    'resonant': 'resonant',
    'transient': 'transient',
    'non-resonant': 'non_resonant',
    'near-separatrix': 'near_separatrix',
    'slow-circulation': 'non_resonant',
    'kinematic-libration': 'non_resonant',
    'insufficient-time': 'uncertain',
    'uncertain': 'uncertain',
    'chaotic': 'invalid',
}
OLD_BUCKET_TO_STATUS = {
    'resonant': {'resonant'},
    'transient': {'transient'},
    'non-resonant': {'non_resonant'},
    'near-separatrix': {'near_separatrix'},
    'slow-circulation': {'non_resonant', 'uncertain'},
    'insufficient-time': {'uncertain'},
    'uncertain': {'uncertain'},
    'chaotic': {'invalid'},
    'kinematic-libration': {'non_resonant'},
}
OLD_STATUS_SPELLING = {'non-resonant': 'non_resonant', 'near-separatrix': 'near_separatrix'}


def test_the_lists_are_the_verdict_enums():
    assert bio.STATUSES == [str(s) for s in Status]
    assert bio.LABEL_CONFIDENCES == [str(c) for c in Confidence if str(c)]
    assert bio.ANGLE_STATUSES == [str(a) for a in AngleStatus]


def test_the_bucket_table_gives_the_old_tables():
    assert {bucket: bio.bucket_status(bucket) for bucket in OLD_BUCKET_STATUS} == OLD_BUCKET_STATUS
    assert {bucket: set(statuses) for bucket, statuses in bio.BUCKET_STATUSES.items()} == OLD_BUCKET_TO_STATUS
    assert all(set(statuses) <= set(bio.STATUSES) for statuses in bio.BUCKET_STATUSES.values())


@pytest.mark.parametrize('bucket', list(OLD_BUCKET_STATUS) + ['', 'invalid', 'something-else'])
def test_spelling_matches_the_old_rule(bucket):
    assert bio.status_spelling(bucket) == OLD_STATUS_SPELLING.get(bucket, bucket)


def test_unknown_buckets_and_the_reverse_spelling():
    assert bio.bucket_status('') == '' and bio.bucket_status('libration') == ''
    assert [bio.status_bucket(s) for s in ('non_resonant', 'near_separatrix', 'resonant')] == [
        'non-resonant',
        'near-separatrix',
        'resonant',
    ]


# --- upsert_labels (rewritten on plain dicts 2026-11-14: repeated DataFrame.at writes crashed pandas 2.3.3) ----------


def _labels(tmp_path, monkeypatch, text):
    path = tmp_path / 'labels.csv'
    path.write_text(text)
    monkeypatch.setattr(bio, 'LABELS', path)
    return path


def test_upsert_keeps_human_labels_refreshes_machine_ones_and_adds_rows(tmp_path, monkeypatch):
    head = ','.join(bio.LABEL_COLUMNS)
    path = _labels(tmp_path, monkeypatch, f'{head}\nb,c2,mmr2,transient,low,resonant,,kept\na,c1,mmr3,resonant,high,,,\n')
    bio.upsert_labels(
        [
            {'example_id': 'b', 'proposed_bucket': 'resonant', 'author_bucket': 'transient', 'comment': 'new'},
            {'example_id': 'a', 'author_bucket': 'transient', 'confidence': None},
            {'example_id': 'c', 'candidate_id': 'c3', 'rtype': 'coorbital', 'not_a_column': 'x'},
        ]
    )
    rows = {r['example_id']: r for r in bio.read_labels().to_dict('records')}
    assert list(rows) == ['a', 'b', 'c']  # sorted by id
    # b: a machine column refreshed, the human ones kept once non-empty
    assert (rows['b']['proposed_bucket'], rows['b']['author_bucket'], rows['b']['comment']) == ('resonant', 'resonant', 'kept')
    # a: an empty human column is filled, None becomes empty
    assert (rows['a']['author_bucket'], rows['a']['confidence']) == ('transient', '')
    # c: a new row gets the existing columns only
    assert rows['c']['rtype'] == 'coorbital' and 'not_a_column' not in rows['c']
    assert path.read_text().splitlines()[0] == head


def test_upsert_refuses_duplicate_ids(tmp_path, monkeypatch):
    head = ','.join(bio.LABEL_COLUMNS)
    path = _labels(tmp_path, monkeypatch, f'{head}\nx,c1,mmr2,resonant,high,resonant,,human\nx,c1,mmr2,,,,,\n')
    before = path.read_text()
    with pytest.raises(ValueError, match='duplicate example_id'):
        bio.upsert_labels([])
    assert path.read_text() == before
