"""benchmark/tools/import_review_verdicts.py on a hand-made review CSV: the author's verdict export becomes label rows
and proposed episode cases. Fast: no benchmark data; the expected rows are written out here by hand."""

import csv
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import import_review_verdicts as irv  # noqa: E402

REVIEW_COLUMNS = ['folder', 'example_id', 'resonance', 'parent', 'new_status', 'new_confidence', 'new_angle_status']
REVIEW_ROWS = [
    ('changed', 'mmr2__12218__4M-7+0+3', '4M-7+0+3', '', 'resonant', 'high', 'libration'),
    ('changed', 'mmr2__12218__4M-7+0+3', '4M-7+1+2', '4M-7+0+3', 'transient', 'medium', 'episodic'),
    ('discussed', 'coorbital__2002_PN__1E-1+0+0', '1E-1+0+0', '', 'transient', 'medium', 'episodic'),
    ('same_rule', 'zlk__10563__LK', 'LK', '', 'uncertain', '', 'episodic'),
    ('discussed', 'mmr3__2003_UG22__10E-17+0+7', '10E-17+0+7', '', 'non_resonant', 'certain', 'circulation'),
]
LABEL_HEADER = 'example_id,status,confidence,angle_status,source\n'
OLD_LABEL = 'mmr2__16017__7S-1+0-6,non_resonant,high,circulation,AUTHOR-QUESTIONS Q4 accepted (DECISIONS 2026-10-15)\n'
GATE = '# a comment line\n' + ','.join(irv.CASE_COLUMNS) + '\n'

# Two agreements, one status correction, one episode correction (a note, no verdict)
EXPORT = {
    'mmr2__12218__4M-7+0+3': {'verdict': 'agree', 'comment': ''},
    'mmr3__2003_UG22__10E-17+0+7::10E-17+0+7': {'verdict': 'agree', 'comment': 'clean  circulation'},
    'zlk__10563__LK': {'verdict': 'status', 'status': 'transient', 'confidence': 'medium'},
    'coorbital__2002_PN__1E-1+0+0': {'episodes_note': 'one episode 300-900 yr, not two', 'comment': 'see the tadpole'},
}
# 2002 PN has a note and no verdict: no label row
EXPECTED_LABELS = [
    [
        'mmr2__12218__4M-7+0+3',
        'resonant',
        'high',
        'libration',
        'author:2026-11-02 (review w28: agrees with the run verdict resonant/high/libration)',
    ],
    [
        'mmr3__2003_UG22__10E-17+0+7',
        'non_resonant',
        'certain',
        'circulation',
        'author:2026-11-02 (review w28: agrees with the run verdict non_resonant/certain/circulation; comment: clean circulation)',
    ],
    [
        'zlk__10563__LK',
        'transient',
        'medium',
        'episodic',  # not given: the run's value is kept
        'author:2026-11-02 (review w28: corrects the run verdict uncertain//episodic to transient/medium/episodic)',
    ],
]
EXPECTED_CASE = [
    'R1',
    'coorbital__2002_PN__1E-1+0+0',
    'A',
    'mmr',
    'author review w28 2026-11-02 (proposed: write the checks)',
    'one episode 300-900 yr, not two',
    '[]',
    'verdicts.json: coorbital__2002_PN__1E-1+0+0',
    '0',
    '',
]


@pytest.fixture
def files(tmp_path):
    review = tmp_path / 'review-w28.csv'
    with review.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(REVIEW_COLUMNS)
        writer.writerows(REVIEW_ROWS)
    (tmp_path / 'labels.csv').write_text(LABEL_HEADER + OLD_LABEL)
    (tmp_path / 'gate.csv').write_text(GATE)
    return tmp_path


def run(files, export, *extra):
    path = files / 'verdicts.json'
    path.write_text(json.dumps(export))
    argv = [str(path), '--review', 'w28', '--review-csv', str(files / 'review-w28.csv'), '--labels', str(files / 'labels.csv')]
    argv += ['--cases', str(files / 'cases.csv'), '--gate-cases', str(files / 'gate.csv'), '--date', '2026-11-02', *extra]
    return irv.main(argv)


def rows(path):
    with path.open(newline='') as f:
        return list(csv.reader(f))


def test_export_gives_exactly_the_expected_rows(files, capsys):
    assert run(files, EXPORT, '--write') == 0
    labels = rows(files / 'labels.csv')
    assert labels[0] == irv.LABEL_COLUMNS
    assert labels[1] == OLD_LABEL.strip().split(',', 4)[:4] + ['AUTHOR-QUESTIONS Q4 accepted (DECISIONS 2026-10-15)']
    assert labels[2:] == EXPECTED_LABELS
    cases = rows(files / 'cases.csv')
    assert cases == [irv.CASE_COLUMNS, EXPECTED_CASE]
    out = capsys.readouterr().out
    assert '3 label rows, 1 proposed episode cases' in out
    assert '+mmr2__12218__4M-7+0+3,resonant,high,libration,' in out  # the diff


def test_dry_run_prints_the_diff_and_writes_nothing(files, capsys):
    assert run(files, EXPORT) == 0
    assert (files / 'labels.csv').read_text() == LABEL_HEADER + OLD_LABEL
    assert not (files / 'cases.csv').exists()
    out = capsys.readouterr().out
    assert '+R1,coorbital__2002_PN__1E-1+0+0,A,mmr,' in out
    assert '(dry: --write applies)' in out


def test_second_run_adds_nothing(files, capsys):
    assert run(files, EXPORT, '--write') == 0
    labels, cases = (files / 'labels.csv').read_text(), (files / 'cases.csv').read_text()
    capsys.readouterr()
    assert run(files, EXPORT, '--write') == 0
    assert (files / 'labels.csv').read_text() == labels
    assert (files / 'cases.csv').read_text() == cases
    out = capsys.readouterr().out
    assert '0 label rows, 0 proposed episode cases' in out
    assert 'label row already in the file' in out and 'episode case already proposed' in out


@pytest.mark.parametrize(
    'export, problem',
    [
        ({'mmr2__99999__1J-2+0+1': {'verdict': 'agree'}}, 'not a row of the review CSV'),
        ({'zlk__10563__LK::ZZ': {'verdict': 'agree'}}, 'not a row of the review CSV'),
        ({'zlk__10563__LK': {'verdict': 'maybe'}}, 'verdict'),
        ({'zlk__10563__LK': {'verdict': 'status', 'status': 'librating'}}, 'status'),
        ({'zlk__10563__LK': {'verdict': 'status'}}, 'without status'),
    ],
)
def test_invalid_export_is_rejected_whole(files, capsys, export, problem):
    export = {**EXPORT, **export}
    assert run(files, export, '--write') == 2
    assert problem in capsys.readouterr().err
    assert (files / 'labels.csv').read_text() == LABEL_HEADER + OLD_LABEL  # the valid entries are not written either
    assert not (files / 'cases.csv').exists()


def test_derived_row_takes_no_label(files, capsys):
    assert run(files, {'mmr2__12218__4M-7+0+3::4M-7+1+2': {'verdict': 'agree'}}, '--write') == 0
    assert (files / 'labels.csv').read_text() == LABEL_HEADER + OLD_LABEL
    assert 'a derived row takes no label' in capsys.readouterr().out


def test_note_already_in_the_gate_is_not_proposed(files):
    gate = GATE + 'E99,coorbital__2002_PN__1E-1+0+0,A,mmr,m,"one episode  300-900 yr, not two",[],s,0,\n'
    (files / 'gate.csv').write_text(gate)
    assert run(files, {'coorbital__2002_PN__1E-1+0+0': EXPORT['coorbital__2002_PN__1E-1+0+0']}, '--write') == 0
    assert not (files / 'cases.csv').exists()
