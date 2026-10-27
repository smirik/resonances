"""angle_status 'episodic' (author, 2026-10-17 10:56: the angle's word is no longer the status's 'transient').

Files written before the rename hold 'transient' in angle_status (summary.csv of a run, a calibration cells file);
they are read as 'episodic', the status 'transient' stays as it is."""

import json

import pandas as pd

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    Reason,
    ResonanceType,
    Status,
    Verdict,
    angle_status_from_text,
    verdict_from_status_code,
)
from resonances.simulation.serializer import SimulationSerializer


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


def test_old_spelling_is_read_as_episodic():
    assert angle_status_from_text('transient') == AngleStatus.EPISODIC
    assert angle_status_from_text(' transient ') == AngleStatus.EPISODIC
    assert angle_status_from_text('episodic') == AngleStatus.EPISODIC
    assert angle_status_from_text('circulation') == AngleStatus.CIRCULATION


def test_old_summary_row_restores_the_same_verdict(tmp_path):
    """A summary.csv row of a run before the rename (angle_status 'transient') rebuilds the verdict of today's run,
    through the serializer's restore path as well."""
    new = episodic_verdict().to_flat_dict()
    old = dict(new, angle_status='transient')
    restored_old = Verdict.from_flat_dict(old)
    restored_new = Verdict.from_flat_dict(new)
    assert restored_old.angle_status == AngleStatus.EPISODIC
    assert restored_old.status == Status.TRANSIENT
    assert restored_old.to_flat_dict() == restored_new.to_flat_dict() == Verdict.from_flat_dict(restored_new.to_flat_dict()).to_flat_dict()
    path = tmp_path / 'summary.csv'
    pd.DataFrame([dict(old, name='463', resonance='4J-2S-1+0+0-1')]).to_csv(path, index=False)
    frame_row = SimulationSerializer._summary_frame(path).iloc[0].to_dict()
    via_restore = SimulationSerializer._verdict_of_row(frame_row, '4J-2S-1+0+0-1', True)
    assert via_restore.angle_status == AngleStatus.EPISODIC and via_restore.status == Status.TRANSIENT


def test_old_final_regime_spelling_is_converted():
    row = dict(episodic_verdict().to_flat_dict(), angle_status='libration', final_regime='transient')
    assert Verdict.from_flat_dict(row).final_regime == AngleStatus.EPISODIC


def test_old_integer_code_one_is_an_episodic_angle():
    verdict = verdict_from_status_code(1, ResonanceType.SECULAR)
    assert (verdict.status, verdict.angle_status) == (Status.TRANSIENT, AngleStatus.EPISODIC)


def test_old_cells_file_keys_are_read_as_episodic(tmp_path):
    cells = [
        {'status': 'transient', 'angle_status': 'transient', 'reason': 'dynamic_episode', 'n_examples': 7, 'n_errors': 1},
        {'status': 'resonant', 'angle_status': 'libration', 'reason': 'dynamic_libration', 'n_examples': 3, 'n_errors': 0},
    ]
    path = tmp_path / 'cells.json'
    path.write_text(json.dumps({'version': 1, 'cells': cells}))
    loaded = conf.load_cells(path)
    assert set(loaded.records) == {('transient', 'episodic', 'dynamic_episode'), ('resonant', 'libration', 'dynamic_libration')}
    assert loaded.records[conf.cell_of(Status.TRANSIENT, AngleStatus.EPISODIC, Reason.DYNAMIC_EPISODE)].n_examples == 7


def test_package_cells_file_has_no_old_spelling():
    document = json.loads(conf.PACKAGE_CELLS.read_text())
    assert all(cell['angle_status'] != 'transient' for cell in document['cells'])
    assert any(cell['angle_status'] == 'episodic' for cell in document['cells'])


def test_figure_check_reads_an_old_summary_without_a_false_difference(tmp_path):
    """Redrawing a figure of a run before the rename compares today's verdict with the saved row: the old spelling is
    not a difference, a real change still is."""
    from types import SimpleNamespace

    from resonances.simulation.data_manager import DataManager

    row = dict(episodic_verdict().to_flat_dict(), angle_status='transient', name='463', resonance='4J-2S-1+0+0-1')
    pd.DataFrame([row]).to_csv(tmp_path / 'summary.csv', index=False)
    manager = DataManager(SimpleNamespace(save_path=str(tmp_path)))
    saved = manager._saved_summary_row(SimpleNamespace(name='463'), '4J-2S-1+0+0-1')
    assert saved['angle_status'] == 'episodic'
    recomputed = episodic_verdict().to_flat_dict()
    assert not [d for d in DataManager.figure_differences(saved, recomputed) if d.startswith('angle_status')]
    moved = dict(recomputed, angle_status='libration')
    assert 'angle_status episodic -> libration' in DataManager.figure_differences(saved, moved)
