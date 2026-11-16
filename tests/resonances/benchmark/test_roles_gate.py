"""The roles of the other resonance on the whole benchmark run (W28 A3a; cards g06-3, g06-1, g05-3; `mmr.roles`).

On this session's reclassification (`reclassified_run`, conftest.py):

- every row an accepted owner does not make (flag owner_outside_episode or owner_psi_too_fast) has an empty `owner`
  and that owner in `neighbours` (card g06-3 A: the 19 rows of 2026-10-31, 1999 GL4 among them);
- every rule C row (origin subresonance) has its member in `resonant_argument` and an empty `owner` (g06-1 A);
- every kinematic row names its term in `cause_term` (g05-3);
- `owner` is filled only on induced rows, and every MMR row has its `commensurability`.

Statuses, angle words and levels are gated by the author gates (test_classify_benchmark.py); the roles change none.
"""

import pandas as pd
import pytest

REFUSED = ('owner_outside_episode', 'owner_psi_too_fast')


@pytest.mark.benchmark
def test_the_role_columns_on_the_run(reclassified_run):
    summary = pd.read_csv(reclassified_run / 'summary.csv', low_memory=False, keep_default_na=False)
    mmr = summary[summary['resonance_type'] == 'mmr']
    flags = mmr['flags'].astype(str)
    refused = mmr[flags.apply(lambda cell: any(flag in cell.split(';') for flag in REFUSED))]
    assert len(refused) > 0
    assert (refused['owner'] == '').all(), refused[['example_id', 'owner']]
    assert (refused['neighbours'] != '').all(), refused[['example_id', 'neighbours']]
    assert (refused['cause_role'] == 'neighbour').all()
    members = mmr[mmr['libration_origin'] == 'subresonance']
    assert ((members['resonant_argument'] != '') & (members['owner'] == '')).all(), members[['example_id', 'owner']]
    kinematic = mmr[mmr['libration_origin'] == 'kinematic']
    assert ((kinematic['cause_term'] != '') & (kinematic['cause_role'] == 'kinematic')).all()
    with_owner = mmr[mmr['owner'] != '']
    assert (with_owner['libration_origin'] == 'induced').all(), with_owner[['example_id', 'libration_origin']]
    assert (mmr['commensurability'] != '').all()
