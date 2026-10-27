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
