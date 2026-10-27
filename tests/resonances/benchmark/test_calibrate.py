"""benchmark/tools/calibrate.py on tiny synthetic tables: truth precedence, abstentions, the counted population,
the Wilson interval and the cells file as the package reads it."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import binomtest

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import bench_io as bio  # noqa: E402
import calibrate as cal  # noqa: E402

from resonances.resonance.analysis import confidence as conf  # noqa: E402

Z = 1.959963984540054
CIRCULATION = ('non_resonant', 'circulation', 'circulation')
LIBRATION = ('resonant', 'libration', 'dynamic_libration')


def summary_row(example_id, cell, confidence, flags='', parent=''):
    status, angle_status, reason = cell
    return dict(
        example_id=example_id,
        parent=parent,
        status=status,
        angle_status=angle_status,
        reason=reason,
        confidence=confidence,
        flags=flags,
    )


def truth(rows):
    return pd.DataFrame(rows, columns=['example_id', 'author_status', 'truth_source'])


# ------------------------------------------------------------------ Wilson


@pytest.mark.parametrize('successes, n', [(10, 10), (0, 10), (99, 100), (100, 101), (37, 50), (1, 3)])
def test_wilson_matches_scipy(successes, n):
    low, high = cal.wilson_interval(successes, n)
    expected = binomtest(successes, n).proportion_ci(confidence_level=0.95, method='wilson')
    assert low == pytest.approx(expected.low, abs=1e-12)
    assert high == pytest.approx(expected.high, abs=1e-12)


def test_wilson_closed_forms_and_score_equation():
    # all successes: lower bound n / (n + z^2); no success: upper bound z^2 / (n + z^2) (Wilson 1927)
    assert cal.wilson_interval(10, 10) == pytest.approx((10 / (10 + Z * Z), 1.0))
    assert cal.wilson_interval(0, 10) == pytest.approx((0.0, Z * Z / (10 + Z * Z)))
    # each bound p solves the score equation (p_hat - p)^2 = z^2 p (1 - p) / n
    p_hat, n = 0.99, 100
    for p in cal.wilson_interval(99, 100):
        assert (p_hat - p) ** 2 == pytest.approx(Z * Z * p * (1 - p) / n, rel=1e-9)
    assert all(math.isnan(x) for x in cal.wilson_interval(0, 0))


# ------------------------------------------------------------------ truth


def test_candidate_verdict_labels_its_own_record_only():
    candidates = pd.DataFrame(
        [
            dict(id='c1', author_verdict='transient'),
            dict(id='c2', author_verdict="controversial: one capture?"),
            dict(id='c3', author_verdict='Non-resonant (as 1620-7E+1S-10) | likely nonresonant'),
            dict(id='c4', author_verdict=''),
        ]
    )
    labels = pd.DataFrame(
        [
            dict(example_id='coorbital__X__1V-1+0+0', candidate_id='c1'),
            dict(example_id='coorbital__X__1V-1+0+0__20kyr', candidate_id='c1'),  # another regime: no label
            dict(example_id='mmr2__Y__2J-1+0-1', candidate_id='c2'),
            dict(example_id='mmr3__Z__1V-2E+1+0+0+0', candidate_id='c3'),
            dict(example_id='mmr2__W__3J-1+0-2', candidate_id='c4'),
        ]
    )
    got = cal.candidate_truth(candidates, labels).set_index('example_id')['author_status'].to_dict()
    assert got == {'coorbital__X__1V-1+0+0': 'transient', 'mmr3__Z__1V-2E+1+0+0+0': 'non_resonant'}


def test_label_files_override_candidate_verdicts():
    candidates = pd.DataFrame([dict(id='c1', author_verdict='transient'), dict(id='c2', author_verdict='resonant')])
    labels = pd.DataFrame(
        [dict(example_id='mmr2__A__5N-23+0+18', candidate_id='c1'), dict(example_id='mmr2__B__7S-1+0-6', candidate_id='c2')]
    )
    label_rows = pd.DataFrame(
        [
            dict(example_id='mmr2__A__5N-23+0+18', author_status='non_resonant', label_source='author_labels_x.csv'),
            dict(example_id='mmr2__C__9M-16+0+7', author_status='uncertain', label_source='author_labels_x.csv'),
            dict(example_id='mmr2__D__1J-1+0+0', author_status='', label_source='author_labels_x.csv'),  # no status
        ]
    )
    got = cal.author_truth(candidates, labels, label_rows).set_index('example_id')
    assert got['author_status'].to_dict() == {
        'mmr2__B__7S-1+0-6': 'resonant',
        'mmr2__A__5N-23+0+18': 'non_resonant',
        'mmr2__C__9M-16+0+7': 'uncertain',
    }
    assert got.loc['mmr2__A__5N-23+0+18', 'truth_source'] == 'author_labels_x.csv'


# ------------------------------------------------------------------ rows and cells


def small_case():
    summary = pd.DataFrame(
        [
            summary_row('e1', CIRCULATION, 'high'),
            summary_row('e2', CIRCULATION, 'certain', flags='zone_eq_only'),
            summary_row('e3', CIRCULATION, 'medium'),  # wrong, but medium: not in n_examples
            summary_row('e4', CIRCULATION, 'high'),  # wrong at high
            summary_row('e5', ('uncertain', 'insufficient_time', 'baseline_needed'), ''),  # abstention
            summary_row('e6', ('invalid', 'undefined', 'alias'), ''),  # abstention
            summary_row('e7', LIBRATION, 'high'),  # author uncertain: abstention
            summary_row('e8', LIBRATION, 'high', parent='e7'),  # derived row: never counted
            summary_row('e9', LIBRATION, 'low'),  # no author label
        ]
    )
    author = truth(
        [
            ('e1', 'non_resonant', 's'),
            ('e2', 'non_resonant', 's'),
            ('e3', 'transient', 's'),
            ('e4', 'resonant', 's'),
            ('e5', 'resonant', 's'),
            ('e6', 'non_resonant', 's'),
            ('e7', 'uncertain', 's'),
            ('e8', 'non_resonant', 's'),
        ]
    )
    return summary, author


def test_labelled_rows_abstentions_and_errors():
    rows = cal.labelled_rows(*small_case()).set_index('example_id')
    assert sorted(rows.index) == ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7']
    assert rows['abstention'].to_dict() == {
        'e1': '',
        'e2': '',
        'e3': '',
        'e4': '',
        'e5': 'run_uncertain',
        'e6': 'run_invalid',
        'e7': 'author_uncertain',
    }
    assert rows.loc[rows['error'], :].index.tolist() == ['e3', 'e4']  # abstentions are never errors
    assert rows['certain'].to_dict() == {k: k == 'e2' for k in rows.index}
    assert rows['calibrated_level'].to_dict() == {k: k in ('e1', 'e2', 'e4', 'e7') for k in rows.index}


def test_cell_table_counts_the_calibrated_levels():
    table = cal.cell_table(cal.labelled_rows(*small_case()))
    assert len(table) == 1  # e7 abstains, so its cell is absent
    row = table.iloc[0]
    assert tuple(row[c] for c in cal.CELL) == CIRCULATION
    assert (row['n_examples'], row['n_errors']) == (3, 1)  # e1, e2, e4; e4 wrong
    assert (row['n_certain'], row['n_errors_certain']) == (1, 0)
    assert (row['n_labelled_all'], row['n_errors_all']) == (4, 2)
    assert row['error_examples'] == ['e4']
    assert row['accuracy'] == pytest.approx(2 / 3)
    assert (row['accuracy_low_95'], row['accuracy_high_95']) == pytest.approx(cal.wilson_interval(2, 3))


def many(prefix, cell, n, n_wrong, author_right, author_wrong, confidence='high'):
    summary = [summary_row(f'{prefix}{i}', cell, confidence) for i in range(n)]
    author = [(f'{prefix}{i}', author_wrong if i < n_wrong else author_right, 's') for i in range(n)]
    return summary, author


def test_cells_file_as_the_package_reads_it(tmp_path):
    parts = [
        many('a', LIBRATION, 100, 0, 'resonant', 'transient'),
        many('b', CIRCULATION, 101, 1, 'non_resonant', 'transient'),
        # 150 medium verdicts, all wrong, and 100 right high ones: the medium ones are not the cell's population
        many('d', ('non_resonant', 'circulation', 'alias'), 150, 150, 'non_resonant', 'resonant', 'medium'),
        many('e', ('non_resonant', 'circulation', 'alias'), 100, 0, 'non_resonant', 'resonant'),
    ]
    summary = pd.DataFrame([row for part in parts for row in part[0]])
    author = truth([row for part in parts for row in part[1]])
    rows = cal.labelled_rows(summary, author)
    document = cal.cells_document(cal.cell_table(rows), 'synthetic', len(rows))
    path = tmp_path / 'cells.json'
    sha = cal.write_cells(document, path)
    assert sha == conf.file_sha(path)
    alias = ('non_resonant', 'circulation', 'alias')
    records = conf.load_cells(path).records
    assert (records[alias].n_examples, records[alias].n_errors) == (100, 0)
    # 100 of 100 right: the Wilson lower bound is n / (n + z^2) (Wilson 1927)
    assert records[alias].accuracy_low_95 == pytest.approx(round(100 / (100 + Z * Z), 4))
    assert (records[CIRCULATION].n_examples, records[CIRCULATION].n_errors) == (101, 1)
    assert (records[CIRCULATION].accuracy_low_95, records[CIRCULATION].accuracy_high_95) == pytest.approx(
        tuple(round(x, 4) for x in cal.wilson_interval(100, 101))
    )
    written = {tuple(c[k] for k in cal.CELL): c for c in json.loads(path.read_text())['cells']}
    assert (written[alias]['n_labelled_all'], written[alias]['n_errors_all']) == (250, 150)
    assert written[CIRCULATION]['error_examples'] == ['b0']


def test_level_table_by_confidence():
    summary = pd.DataFrame(
        [
            summary_row('x1', CIRCULATION, 'certain'),
            summary_row('x2', CIRCULATION, 'high'),
            summary_row('x3', CIRCULATION, 'medium'),
        ]
    )
    author = truth([('x1', 'non_resonant', 's'), ('x2', 'transient', 's'), ('x3', 'non_resonant', 's')])
    table = cal.level_table(cal.labelled_rows(summary, author)).set_index('level')
    assert table[['n', 'errors']].to_dict('index') == {
        'certain': {'n': 1, 'errors': 0},
        'high': {'n': 1, 'errors': 1},
        'medium': {'n': 1, 'errors': 0},
    }


def test_main_end_to_end(tmp_path, monkeypatch):
    bench, data = tmp_path / 'bench', tmp_path / 'data'
    (data / 'run' / 'secular_zlk').mkdir(parents=True)
    bench.mkdir()
    pd.DataFrame(
        [dict(id='c1', author_verdict='non-resonant'), dict(id='c2', author_verdict='resonant'), dict(id='c3', author_verdict='resonant')]
    ).to_csv(bench / 'candidates.csv', index=False)
    pd.DataFrame(
        [
            dict(example_id='mmr2__A__2J-1+0-1', candidate_id='c1'),
            dict(example_id='mmr2__B__3J-1+0-2', candidate_id='c2'),
            dict(example_id='zlk__C__LK', candidate_id='c3'),
        ]
    ).to_csv(bench / 'labels.csv', index=False)
    pd.DataFrame(
        [dict(example_id='mmr2__B__3J-1+0-2', status='transient', confidence='high', angle_status='episodic', timestamp='t')]
    ).to_csv(bench / 'author_labels_2026-01-01.csv', index=False)
    pd.DataFrame([summary_row('mmr2__A__2J-1+0-1', CIRCULATION, 'high'), summary_row('mmr2__B__3J-1+0-2', LIBRATION, 'high')]).to_csv(
        data / 'run' / 'summary.csv', index=False
    )
    pd.DataFrame([summary_row('zlk__C__LK', LIBRATION, 'certain')]).to_csv(data / 'run' / 'secular_zlk' / 'summary.csv', index=False)
    monkeypatch.setattr(bio, 'BENCH', bench)
    monkeypatch.setattr(bio, 'DATA', data)
    monkeypatch.setattr(bio, 'LABELS', bench / 'labels.csv')
    cells, report = tmp_path / 'cells.json', tmp_path / 'report.md'
    assert cal.main(['--run', 'run', '--cells', str(cells), '--report', str(report)]) == 0
    written = {tuple(c[k] for k in cal.CELL): (c['n_examples'], c['n_errors']) for c in json.loads(cells.read_text())['cells']}
    # B: the author's label file (transient) overrides the candidate verdict (resonant); C comes from secular_zlk/
    assert written == {CIRCULATION: (1, 0), LIBRATION: (2, 1)}
    rows = pd.read_csv(data / 'run' / 'calibration_rows.csv', dtype=str).fillna('')
    assert rows.set_index('example_id')['truth_source'].to_dict()['mmr2__B__3J-1+0-2'] == 'author_labels_2026-01-01.csv'
    text = report.read_text()
    assert '| resonant/libration/dynamic_libration | 2 | 1 |' in text and 'mmr2__B__3J-1+0-2' in text


def test_run_before_the_rename_gives_episodic_cells(tmp_path, monkeypatch):
    """A run written before 2026-10-17 has angle_status 'transient' for an episodic angle: its cells are keyed
    'episodic', the cells the package reads for today's verdicts (`bench_io.read_summary`)."""
    bench, data = tmp_path / 'bench', tmp_path / 'data'
    (data / 'old').mkdir(parents=True)
    bench.mkdir()
    pd.DataFrame([dict(id='c1', author_verdict='transient'), dict(id='c2', author_verdict='non-resonant')]).to_csv(
        bench / 'candidates.csv', index=False
    )
    pd.DataFrame([dict(example_id='mmr2__A__2J-1+0-1', candidate_id='c1'), dict(example_id='mmr2__B__3J-1+0-2', candidate_id='c2')]).to_csv(
        bench / 'labels.csv', index=False
    )
    old_cell = ('transient', 'transient', 'dynamic_episode')
    rows = [summary_row('mmr2__A__2J-1+0-1', old_cell, 'high'), summary_row('mmr2__B__3J-1+0-2', old_cell, 'high')]
    rows[0]['final_regime'] = 'circulation'
    pd.DataFrame(rows).to_csv(data / 'old' / 'summary.csv', index=False)
    monkeypatch.setattr(bio, 'BENCH', bench)
    monkeypatch.setattr(bio, 'DATA', data)
    monkeypatch.setattr(bio, 'LABELS', bench / 'labels.csv')
    summary = bio.read_summary(data / 'old' / 'summary.csv')
    assert summary['angle_status'].tolist() == ['episodic', 'episodic']
    assert summary['final_regime'].tolist() == ['circulation', '']
    cells = tmp_path / 'cells.json'
    assert cal.main(['--run', 'old', '--cells', str(cells)]) == 0
    written = {tuple(c[k] for k in cal.CELL): (c['n_examples'], c['n_errors']) for c in json.loads(cells.read_text())['cells']}
    assert written == {('transient', 'episodic', 'dynamic_episode'): (2, 1)}
    loaded = conf.load_cells(cells)
    assert loaded.records[('transient', 'episodic', 'dynamic_episode')].n_errors == 1
