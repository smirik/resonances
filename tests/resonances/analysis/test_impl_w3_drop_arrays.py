"""Releasing the arrays of an MMR analysis keeps its CSV rows (requests-w3-simcore.md).

`drop_arrays` frees the preprocessed series, the slow components and the centre track. Before the fix the
mmr_numbers row lost its 15 `flt_*` columns and the episodes their cutoff, so the package could not release an
analysis before writing its tables. The rows are now frozen first: every value must be the same after the release.
"""

import math

import numpy as np

from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.test_impl_w3_guards_ledger_columns import libration_then_circulation

S = settings_from_config()


def _same(x, y) -> bool:
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y


def _assert_rows_equal(before: dict, after: dict):
    assert before.keys() == after.keys()
    differ = [key for key in before if not _same(before[key], after[key])]
    assert differ == []


def test_rows_survive_the_release_of_the_arrays():
    analysis = analyse_mmr(libration_then_circulation(), S, scan_owner=False, scan_subresonances=False)
    numbers, summary, episodes = analysis.numbers_row(), analysis.summary_row(), analysis.episode_rows()
    flt = [key for key in numbers if key.startswith('flt_')]
    assert len(flt) > 15 and numbers['flt_cutoff_yr'] > 0  # the columns the release used to lose
    assert episodes and all(row['cutoff_yr'] == numbers['flt_cutoff_yr'] for row in episodes)

    analysis.drop_arrays()

    assert analysis.prep is None and analysis.slow is None and analysis.geometry.centre.t_yr.size == 0
    # W5: the centres of the regimes before a jump and the energy's model frame go too (full-length arrays)
    assert analysis.geometry.record_centres == []
    assert analysis.energy is None or analysis.energy.model_frame is None
    # W20 round C (review B #8): the per-TP h of the averaged potential goes too, its sep_h_avgpot_* stay frozen
    assert analysis.energy is None or analysis.energy.h_avgpot == ()
    _assert_rows_equal(numbers, analysis.numbers_row())
    _assert_rows_equal(summary, analysis.summary_row())
    assert len(episodes) == len(analysis.episode_rows())
    for before, after in zip(episodes, analysis.episode_rows()):
        _assert_rows_equal(before, after)
    # a second release changes nothing and the frozen rows are copies (a caller may edit what it gets)
    analysis.drop_arrays()
    analysis.numbers_row()['flt_cutoff_yr'] = -1.0
    _assert_rows_equal(numbers, analysis.numbers_row())


def test_a_refused_record_can_be_released():
    series = libration_then_circulation()
    refused = analyse_mmr(MMRSeries(**{**series.__dict__, 'a': np.full(series.t_yr.size, -1.0)}), S)
    assert refused.sampling is None and refused.verdict.status == 'invalid'
    numbers = refused.numbers_row()
    refused.drop_arrays()
    _assert_rows_equal(numbers, refused.numbers_row())
