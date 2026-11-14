"""angle_status 'episodic' (author, 2026-10-17 10:56: the angle's word is no longer the status's 'transient').

Files written before the rename (angle_status 'transient') are not read any more (v1.0: no back-compat readers)."""

import json

import pandas as pd

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    Reason,
    ResonanceType,
    Status,
    Verdict,
)


def episodic_verdict():
    return Verdict(
        ResonanceType.MMR,
        Status.TRANSIENT,
        AngleStatus.EPISODIC,
        final_regime=AngleStatus.CIRCULATION,
        reason=Reason.DYNAMIC_EPISODE,
    )


def test_the_value_is_episodic_and_the_status_keeps_transient():
    assert str(AngleStatus.EPISODIC) == 'episodic'
    assert str(Status.TRANSIENT) == 'transient'
    assert 'transient' not in {str(a) for a in AngleStatus}
    row = episodic_verdict().to_flat_dict()
    assert (row['status'], row['angle_status'], row['final_regime']) == ('transient', 'episodic', 'circulation')


def test_package_cells_file_has_no_old_spelling():
    document = json.loads(conf.PACKAGE_CELLS.read_text())
    assert all(cell['angle_status'] != 'transient' for cell in document['cells'])
    assert any(cell['angle_status'] == 'episodic' for cell in document['cells'])


def test_figure_check_compares_the_saved_row(tmp_path):
    """Redrawing a figure compares today's verdict with the saved row: the same verdict is no difference, a real change
    is."""
    from types import SimpleNamespace

    from resonances.simulation.data_manager import DataManager

    row = dict(episodic_verdict().to_flat_dict(), name='463', resonance='4J-2S-1+0+0-1')
    pd.DataFrame([row]).to_csv(tmp_path / 'summary.csv', index=False)
    manager = DataManager(SimpleNamespace(save_path=str(tmp_path)))
    saved = manager._saved_summary_row(SimpleNamespace(name='463'), '4J-2S-1+0+0-1')
    assert saved['angle_status'] == 'episodic'
    recomputed = episodic_verdict().to_flat_dict()
    assert not [d for d in DataManager.figure_differences(saved, recomputed) if d.startswith('angle_status')]
    moved = dict(recomputed, angle_status='libration')
    assert 'angle_status episodic -> libration' in DataManager.figure_differences(saved, moved)
