"""W5: a restore reads the saved series of a body named by a number, and of a derived row known only from summary.csv
(codex astra W5 findings 5 and 6). The files are written here by hand, as a run writes them."""

import json

import pandas as pd

from resonances.resonance.analysis.verdict import AngleStatus, ResonanceType, Status, Verdict
from resonances.simulation.serializer import SimulationSerializer

PARENT, CHILD = '4J-2S-1+0+0-1', '4J-2S-1-1+0+0'


def _row(name, resonance, parent=''):
    """A summary.csv row with the verdict columns a run writes (a resonant libration)."""
    return dict(
        name=name, resonance=resonance, parent=parent, **Verdict(ResonanceType.MMR, Status.RESONANT, AngleStatus.LIBRATION).to_flat_dict()
    )


def _write_run(folder, name, summary_rows):
    config = dict(name='restore-w5', date='2026-09-14', save=None, plot=None, save_summary=False, integration_years=1000)
    body = {'name': name, 'initial_data': {'a': 2.6, 'e': 0.1}, 'resonances': [{'key': PARENT}]}
    doc = {'config': config, 'simulation': {'bodies': [body], 'data_files': {'bodies': [f'data-{name}.csv']}}}
    (folder / 'simulation.json').write_text(json.dumps(doc))
    data = {
        'times': [0.0, 1.0, 2.0],
        'a': [2.6, 2.61, 2.62],
        'e': [0.1] * 3,
        f'{PARENT}_angle': [0.0, 0.1, 0.2],
        f'{CHILD}_angle': [1.0, 1.1, 1.2],
    }
    pd.DataFrame(data).to_csv(folder / f'data-{name}.csv')
    pd.DataFrame(summary_rows).to_csv(folder / 'summary.csv', index=False)
    return SimulationSerializer.restore(str(folder / 'simulation.json'))


def test_a_body_named_by_a_number_gets_its_series(tmp_path):
    sim = _write_run(tmp_path, 123, [_row(123, PARENT)])
    body = sim.bodies[0]
    assert list(body.axis) == [2.6, 2.61, 2.62]
    assert list(body.angles[PARENT]) == [0.0, 0.1, 0.2]
    assert SimulationSerializer._saved_resonances(body) == {PARENT}
    assert body.statuses[PARENT] == 'resonant'


def test_a_derived_row_from_the_summary_gets_its_series(tmp_path):
    rows = [
        _row('child', PARENT),
        _row('child', CHILD, PARENT),
    ]
    sim = _write_run(tmp_path, 'child', rows)
    body = sim.bodies[0]
    assert body.derived_resonances[CHILD] == PARENT
    assert list(body.angles[CHILD]) == [1.0, 1.1, 1.2]
    assert SimulationSerializer._saved_resonances(body) == {PARENT, CHILD}
    assert body.statuses[CHILD] == 'resonant'
