"""An MMR row that will not be plotted releases its analysis' arrays right after its verdict (memory: a batch keeps
only the arrays of the rows it draws). The rows written to summary.csv and mmr_numbers.csv must not change."""

import math

from resonances.simulation.data_manager import DataManager
from tests.resonances.simulation.test_restore_and_resume import CIRC, LIB, _saved_run


def _rows(sim) -> dict:
    return {
        (body.name, key): (analysis.summary_row(), analysis.numbers_row(), analysis.episode_rows())
        for body in sim.bodies
        for key, analysis in body.mmr_analyses.items()
    }


def _same(x, y) -> bool:
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y


def test_unplotted_rows_are_released_and_their_rows_do_not_change(tmp_path, monkeypatch):
    monkeypatch.setattr(DataManager, 'save_data', lambda manager, bodies, times, simulation=None: None)
    kept = _saved_run(tmp_path / 'all', plot='all')
    released = _saved_run(tmp_path / 'resonant', plot='resonant')

    librator = {b.name: b for b in released.bodies}['librator']
    assert librator.statuses == {LIB: 'resonant', CIRC: 'non_resonant'}
    assert librator.mmr_analyses[LIB].prep is not None  # plotted under 'resonant': its figure needs the arrays
    assert librator.mmr_analyses[CIRC].prep is None  # not plotted: released at once
    assert all(a.prep is not None for b in kept.bodies for a in b.mmr_analyses.values())

    before, after = _rows(kept), _rows(released)
    assert before.keys() == after.keys() == {('librator', LIB), ('librator', CIRC), ('circulator', CIRC)}
    for key in before:
        for row_before, row_after in zip(before[key][:2], after[key][:2]):
            assert row_before.keys() == row_after.keys()
            timing = [c for c in row_before if c.endswith('_seconds')]  # wall-clock cost, differs between any two runs
            assert [c for c in row_before if c not in timing and not _same(row_before[c], row_after[c])] == []
        assert len(before[key][2]) == len(after[key][2])
