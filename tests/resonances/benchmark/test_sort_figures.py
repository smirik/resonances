"""benchmark/tools/sort_figures.py on a hand-made data root: the tree of the combined figures by a run's verdicts.
Fast: no benchmark data; the expected paths are written out here."""

import sys
from pathlib import Path

import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import sort_figures as sf  # noqa: E402

COLUMNS = ['example_id', 'name', 'resonance', 'parent', 'resonance_type', 'status', 'confidence']
MMR_ROWS = [
    ('mmr2__12218__4M-7+0+3', '12218', '4M-7+0+3', '', 'mmr', 'resonant', 'high'),
    ('mmr2__12218__4M-7+0+3', '12218', '4M-7+1+2', '4M-7+0+3', 'mmr', 'transient', 'medium'),
    ('mmr3__2003_UG22__10E-17+0+7', '2003 UG22', '10E-17+0+7', '', 'mmr', 'uncertain', ''),
    ('mmr2__lost__1J-2+0+1', 'lost', '1J-2+0+1', '', 'mmr', 'non_resonant', 'certain'),  # no figure drawn
]
SECULAR_ROWS = [('zlk__10563__LK', '10563', 'LK', '', 'lidov_kozai', 'non_resonant', 'low')]


@pytest.fixture
def data(tmp_path):
    """examples/<id>/combined_v2.png (and the derived row's), a run and its -secular-zlk sibling."""
    for example_id, figure in [
        ('mmr2__12218__4M-7+0+3', 'combined_v2.png'),
        ('mmr2__12218__4M-7+0+3', 'combined_v2__4M-7+1+2.png'),
        ('mmr3__2003_UG22__10E-17+0+7', 'combined_v2.png'),
        ('zlk__10563__LK', 'combined_v2.png'),
    ]:
        folder = tmp_path / 'examples' / example_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / figure).write_bytes(f'{example_id}/{figure}'.encode())
    for run, rows in (('w', MMR_ROWS), ('w-secular-zlk', SECULAR_ROWS)):
        (tmp_path / run).mkdir()
        pd.DataFrame(rows, columns=COLUMNS).to_csv(tmp_path / run / 'summary.csv', index=False)
    return tmp_path


def tree(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*.png'))


def test_confidence_status_tree_of_both_runs(data):
    assert sf.main(['--run', 'w', '--data', str(data)]) == 0
    root = data / 'figures' / 'confidence-status'
    assert tree(root) == [
        'high/resonant/mmr2__12218__4M-7+0+3.png',
        'low/non_resonant/zlk__10563__LK.png',
        'medium/transient/mmr2__12218__4M-7+0+3__4M-7+1+2.png',
        'none/uncertain/mmr3__2003_UG22__10E-17+0+7.png',
    ]
    # hard links to the examples' figures, nothing written into the examples
    linked = root / 'medium' / 'transient' / 'mmr2__12218__4M-7+0+3__4M-7+1+2.png'
    source = data / 'examples' / 'mmr2__12218__4M-7+0+3' / 'combined_v2__4M-7+1+2.png'
    assert linked.stat().st_ino == source.stat().st_ino
    assert sorted(p.name for p in (data / 'examples' / 'mmr2__12218__4M-7+0+3').iterdir()) == [
        'combined_v2.png',
        'combined_v2__4M-7+1+2.png',
    ]


def test_other_presets(data):
    sf.main(['--run', 'w', '--data', str(data), '--preset', 'status/confidence'])
    sf.main(['--run', 'w', '--data', str(data), '--preset', 'type/status'])
    assert 'resonant/high/mmr2__12218__4M-7+0+3.png' in tree(data / 'figures' / 'status-confidence')
    assert tree(data / 'figures' / 'type-status') == [
        'lidov_kozai/non_resonant/zlk__10563__LK.png',
        'mmr/resonant/mmr2__12218__4M-7+0+3.png',
        'mmr/transient/mmr2__12218__4M-7+0+3__4M-7+1+2.png',
        'mmr/uncertain/mmr3__2003_UG22__10E-17+0+7.png',
    ]


def test_rerun_rebuilds_only_its_own_tree(data):
    sf.main(['--run', 'w', '--data', str(data)])
    sf.main(['--run', 'w', '--data', str(data), '--preset', 'status'])
    other = tree(data / 'figures' / 'status')
    # a verdict changes in the run: the figure moves, no stale copy stays; the other preset's tree is untouched
    frame = pd.read_csv(data / 'w' / 'summary.csv', dtype=str, keep_default_na=False)
    frame.loc[frame['resonance'] == '4M-7+0+3', 'confidence'] = 'certain'
    frame.to_csv(data / 'w' / 'summary.csv', index=False)
    sf.main(['--run', 'w', '--data', str(data)])
    files = tree(data / 'figures' / 'confidence-status')
    assert 'certain/resonant/mmr2__12218__4M-7+0+3.png' in files and 'high/resonant/mmr2__12218__4M-7+0+3.png' not in files
    assert len(files) == 4
    assert tree(data / 'figures' / 'status') == other


def test_dry_run_and_limit_write_nothing(data, capsys):
    sf.main(['--run', 'w', '--data', str(data), '--dry-run', '--limit', '2'])
    out = capsys.readouterr().out
    assert '(2 figures, 0 missing)' in out
    assert not (data / 'figures').exists()


def test_missing_figure_is_listed(data, capsys):
    sf.main(['--run', 'w', '--data', str(data), '--dry-run'])
    out = capsys.readouterr().out
    assert '(4 figures, 1 missing)' in out and 'mmr2__lost__1J-2+0+1' in out


def test_never_clears_outside_the_tree(data, tmp_path_factory):
    with pytest.raises(ValueError):
        sf.build([], data / 'examples', data)
    with pytest.raises(ValueError):
        sf.build([], data / 'figures' / '..', data)
    # data/figures a symlink to another folder: nothing there is deleted
    elsewhere = tmp_path_factory.mktemp('elsewhere')
    (elsewhere / 'status').mkdir()
    (elsewhere / 'status' / 'keep.txt').write_text('keep')
    (data / 'figures').symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError):
        sf.main(['--run', 'w', '--data', str(data), '--preset', 'status'])
    assert (elsewhere / 'status' / 'keep.txt').exists()


def test_the_same_verdict_in_two_runs_is_linked_once(data):
    """A copy of the run: every target once, no error on the second link of the same figure."""
    (data / 'w2').mkdir()
    (data / 'w2' / 'summary.csv').write_text((data / 'w' / 'summary.csv').read_text())
    assert sf.main(['--run', 'w,w2', '--data', str(data)]) == 0
    assert len(tree(data / 'figures' / 'confidence-status')) == 4
