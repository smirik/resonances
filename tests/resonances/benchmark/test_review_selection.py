"""The review selection on the benchmark (W28 E1, cards g10-1a and g10-1b, the author 2026-10-31).

`review`, the default of SAVE and PLOT, must select exactly the rows of `nonzero` plus the non_resonant rows at medium
or low confidence, read through the same path a restore takes (summary.csv -> `Verdict.from_flat_dict` -> the
selection inputs of `DataManager`). It moves no status. At the card (both reclassifications of 2026-10-31) that was
540 rows instead of 490: 421 + 50 MMR and co-orbital rows, 69 secular and ZLK rows (none added there). The count
moves with the verdicts of later work, so it is printed, not gated; the identity is gated. The `review_required`
column of every row must equal the queue rebuilt from its other columns.

Local-only data (benchmark/data): marker benchmark.
"""

import pandas as pd
import pytest

from resonances.resonance.analysis.verdict import Verdict
from resonances.simulation.config import status_selected

pytestmark = pytest.mark.benchmark


def _selection(run):
    """(rows, nonzero, review, by hand) of a reclassification's summary.csv."""
    frame = pd.read_csv(run / 'summary.csv', dtype=str, keep_default_na=False)
    rows = frame.to_dict('records')
    nonzero, review, by_hand = set(), set(), set()
    for index, row in enumerate(rows):
        verdict = Verdict.from_flat_dict(row)
        fields = (str(verdict.status), str(verdict.libration_origin), str(verdict.action), str(verdict.confidence))
        if status_selected('nonzero', *fields):
            nonzero.add(index)
        if status_selected('review', *fields):
            review.add(index)
        doubtful = row['status'] == 'non_resonant' and row['confidence'] in ('medium', 'low')
        if row['status'] != 'non_resonant' or row['libration_origin'] != '' or doubtful:
            by_hand.add(index)
        if 'review_required' in row:
            assert row['review_required'] == str(verdict.review_required), (row['name'], row['resonance'])
    return rows, nonzero, review, by_hand


@pytest.mark.parametrize('fixture', ['reclassified_run', 'secular_zlk_run'])
def test_review_is_nonzero_plus_the_doubtful_negatives(fixture, request):
    rows, nonzero, review, by_hand = _selection(request.getfixturevalue(fixture))
    print(f'\n{fixture}: {len(rows)} rows, nonzero {len(nonzero)}, review {len(review)}')
    assert review == by_hand
    assert nonzero <= review
    for index in review - nonzero:
        assert rows[index]['status'] == 'non_resonant' and rows[index]['confidence'] in ('medium', 'low')
