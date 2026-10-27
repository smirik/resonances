"""Reasons geometric_libration / geometric_episode (author 2026-10-26, Q3): when no force was measured and the origin is
unknown (MMR leaf 5u/6), only the geometry decided; before the rename those rows read dynamic_libration /
dynamic_episode. A row with origin dynamic (a measured force) keeps dynamic_*.

Files written before the rename are read in today's spelling: the reason alone cannot tell the two cases apart, the
origin column can (origin unknown -> geometric_*, origin dynamic -> dynamic_*)."""

import pandas as pd
import pytest

from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
    reason_from_text,
    status_code,
)
from resonances.simulation.serializer import SimulationSerializer


def verdict(status, angle_status, reason, origin):
    return Verdict(ResonanceType.MMR, status, angle_status, reason=reason, libration_origin=origin)


GEOMETRIC = verdict(Status.TRANSIENT, AngleStatus.EPISODIC, Reason.GEOMETRIC_EPISODE, LibrationOrigin.UNKNOWN)
DYNAMIC = verdict(Status.RESONANT, AngleStatus.LIBRATION, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC)


def test_the_new_spellings():
    assert (str(Reason.GEOMETRIC_LIBRATION), str(Reason.GEOMETRIC_EPISODE)) == ('geometric_libration', 'geometric_episode')
    assert (str(Reason.DYNAMIC_LIBRATION), str(Reason.DYNAMIC_EPISODE)) == ('dynamic_libration', 'dynamic_episode')
    assert decide.GEOMETRIC_REASONS == {
        Reason.DYNAMIC_LIBRATION: Reason.GEOMETRIC_LIBRATION,
        Reason.DYNAMIC_EPISODE: Reason.GEOMETRIC_EPISODE,
    }


@pytest.mark.parametrize(
    'text, origin, expected',
    [
        # the eight benchmark rows of v2-w23 (2348 4J-2S-1, 24820, 1862 2M-6J-1, ...): origin unknown
        ('dynamic_libration', 'unknown', Reason.GEOMETRIC_LIBRATION),
        ('dynamic_episode', 'unknown', Reason.GEOMETRIC_EPISODE),
        (' dynamic_episode ', ' unknown ', Reason.GEOMETRIC_EPISODE),
        # a measured force (2001 ND13 1V-3: the harmonic law passes, origin dynamic) keeps its reason
        ('dynamic_libration', 'dynamic', Reason.DYNAMIC_LIBRATION),
        ('dynamic_episode', 'dynamic', Reason.DYNAMIC_EPISODE),
        ('dynamic_episode', '', Reason.DYNAMIC_EPISODE),  # an old file without the origin column
        # today's spellings and the other reasons are read as they are
        ('geometric_episode', 'unknown', Reason.GEOMETRIC_EPISODE),
        ('origin_unknown', 'unknown', Reason.ORIGIN_UNKNOWN),
        ('separatrix_zone', 'unknown', Reason.SEPARATRIX_ZONE),
        ('no_such_reason', 'unknown', 'no_such_reason'),
    ],
)
def test_old_reasons_are_read_by_their_origin(text, origin, expected):
    assert reason_from_text(text, origin) == expected


def test_old_summary_row_restores_today_s_verdict(tmp_path):
    """A summary.csv row of a run before the rename rebuilds the verdict of today's run, through the serializer's
    restore path as well; the old integer code is the same (statuses do not change)."""
    new = GEOMETRIC.to_flat_dict()
    old = dict(new, reason='dynamic_episode')
    assert Verdict.from_flat_dict(old).to_flat_dict() == Verdict.from_flat_dict(new).to_flat_dict()
    assert Verdict.from_flat_dict(old).reason == Reason.GEOMETRIC_EPISODE
    path = tmp_path / 'summary.csv'
    pd.DataFrame([dict(old, name='24820', resonance='3J-1S-1+0+0-1')]).to_csv(path, index=False)
    row = SimulationSerializer._summary_frame(path).iloc[0].to_dict()
    restored = SimulationSerializer._verdict_of_row(row, '3J-1S-1+0+0-1', True)
    assert (restored.status, restored.reason, restored.libration_origin) == (
        Status.TRANSIENT,
        Reason.GEOMETRIC_EPISODE,
        LibrationOrigin.UNKNOWN,
    )
    assert status_code(ResonanceType.MMR, Status.TRANSIENT, AngleStatus.EPISODIC, Reason.GEOMETRIC_EPISODE) == 1
    assert status_code(ResonanceType.MMR, Status.RESONANT, AngleStatus.LIBRATION, Reason.GEOMETRIC_LIBRATION) == 2


def test_a_measured_force_row_keeps_dynamic():
    row = DYNAMIC.to_flat_dict()
    assert Verdict.from_flat_dict(row).reason == Reason.DYNAMIC_LIBRATION


def test_figure_check_reads_an_old_summary_in_today_s_spelling(tmp_path):
    from types import SimpleNamespace

    from resonances.simulation.data_manager import DataManager

    rows = [
        dict(GEOMETRIC.to_flat_dict(), reason='dynamic_episode', name='24820', resonance='3J-1S-1+0+0-1'),
        dict(DYNAMIC.to_flat_dict(), name='463', resonance='4J-2S-1+0+0-1'),
    ]
    pd.DataFrame(rows).to_csv(tmp_path / 'summary.csv', index=False)
    manager = DataManager(SimpleNamespace(save_path=str(tmp_path)))
    assert manager._saved_summary_row(SimpleNamespace(name='24820'), '3J-1S-1+0+0-1')['reason'] == 'geometric_episode'
    assert manager._saved_summary_row(SimpleNamespace(name='463'), '4J-2S-1+0+0-1')['reason'] == 'dynamic_libration'


def test_benchmark_summary_reader_uses_today_s_spelling(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'))
    import bench_io

    rows = [
        dict(GEOMETRIC.to_flat_dict(), reason='dynamic_episode', example_id='a'),
        dict(DYNAMIC.to_flat_dict(), example_id='b'),
    ]
    pd.DataFrame(rows).to_csv(tmp_path / 'summary.csv', index=False)
    frame = bench_io.read_summary(tmp_path / 'summary.csv')
    assert list(frame['reason']) == ['geometric_episode', 'dynamic_libration']
