"""Reasons geometric_libration / geometric_episode (author 2026-10-26, Q3): when no force was measured and the origin is
unknown (MMR leaf 5u/6), only the geometry decided; before the rename those rows read dynamic_libration /
dynamic_episode. A row with origin dynamic (a measured force) keeps dynamic_*.

The package reads only today's spelling (v1.0: no back-compat readers); the benchmark tools still read the old runs
in benchmark/data in today's spelling (`bench_io.read_summary`: the origin column tells the two cases apart, origin
unknown -> geometric_*, origin dynamic -> dynamic_*)."""

import pandas as pd

from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    LibrationOrigin,
    Reason,
    ResonanceType,
    Status,
    Verdict,
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


def test_summary_row_restores_the_geometric_verdict(tmp_path):
    """A summary.csv row rebuilds the verdict through the serializer's restore path; the old integer code is the
    same as before the rename (statuses do not change)."""
    new = GEOMETRIC.to_flat_dict()
    assert Verdict.from_flat_dict(new).to_flat_dict() == new
    path = tmp_path / 'summary.csv'
    pd.DataFrame([dict(new, name='24820', resonance='3J-1S-1+0+0-1')]).to_csv(path, index=False)
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
