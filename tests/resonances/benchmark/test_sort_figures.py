"""benchmark/tools/sort_figures.py on a hand-made data root: the tree of the combined figures by a run's verdicts.
Fast: no benchmark data; the expected paths are written out here."""

import math
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


# ------------------------------------------------------------------ review mode (card s-4)

SUMMARY = ['example_id', 'name', 'resonance', 'parent', 'resonance_type', 'status', 'angle_status', 'confidence']
SUMMARY += ['reason', 'libration_origin', 'rules_fired', 'smp_span_yr']
EPISODES = ['example_id', 'resonance', 'episode_id', 'accepted', 't_start_yr', 't_end_yr', 'period_med_yr']
LIB = ('resonant', 'libration', 'high', 'dynamic_libration', 'dynamic')


def verdict_row(example_id, verdict=LIB, rules='', span='1000'):
    return (example_id, example_id.split('__')[1], example_id.split('__')[2], '', 'mmr', *verdict, rules, span)


# Record span 1000 yr and P = 100 yr: the shift limit is max(1 x 100, 0.02 x 1000) = 100 yr.
BASE = {
    'mmr2__A__1J-2+0+1': (LIB, [(100, 900, 100)]),
    'mmr2__B__1J-2+0+1': (LIB, [(100, 900, 100)]),
    'mmr2__C__1J-2+0+1': (LIB, [(100, 900, 100)]),
    'mmr2__D__1J-2+0+1': (LIB, [(100, 900, 100)]),
    'mmr2__E__1J-2+0+1': (LIB, [(100, 900, 100)]),
    'mmr2__F__1J-2+0+1': (LIB, [(100, 900, 100)]),
    'mmr2__H__1J-2+0+1': (LIB, [(100, 900, 100)]),
    'mmr2__I__1J-2+0+1': (LIB, [(100, 900, 100)]),
    'mmr2__J__1J-2+0+1': (LIB, [(100, 900, 100)]),
    'mmr2__gone__1J-2+0+1': (LIB, []),
}
NEW = {
    # status changed -> changed/
    'mmr2__A__1J-2+0+1': (('transient', 'episodic', 'medium', 'dynamic_episode', 'dynamic'), [(100, 900, 100)], ''),
    # same verdict, the end moved 150 yr = 1.5 P > 100 -> changed/
    'mmr2__B__1J-2+0+1': (LIB, [(100, 1050, 100)], ''),
    # same verdict, the end moved 50 yr = 0.5 P -> absent
    'mmr2__C__1J-2+0+1': (LIB, [(100, 950, 100)], ''),
    # same verdict and episodes, a W28 rule fired -> same_rule/
    'mmr2__D__1J-2+0+1': (LIB, [(100, 900, 100)], 'bounded_gap_merge'),
    # discussed, nothing changed -> discussed/
    'mmr2__E__1J-2+0+1': (LIB, [(100, 900, 100)], ''),
    # untouched -> absent
    'mmr2__F__1J-2+0+1': (LIB, [(100, 900, 100)], ''),
    # a new row -> changed/
    'mmr2__G__1J-2+0+1': (LIB, [(100, 900, 100)], ''),
    # discussed and changed: discussed wins
    'mmr2__H__1J-2+0+1': (('uncertain', 'episodic', '', 'insufficient_cycles', 'unknown'), [(100, 900, 100)], ''),
    # a rule name outside the registry (an older file) is no W28 rule -> absent
    'mmr2__I__1J-2+0+1': (LIB, [(100, 900, 100)], 'some_old_rule'),
    # one episode split into two -> changed/ (count)
    'mmr2__J__1J-2+0+1': (LIB, [(100, 400, 100), (500, 900, 100)], ''),
}


def write_run(folder, rows):
    folder.mkdir(parents=True)
    summary, episodes = [], []
    for example_id, (verdict, eps, *rules) in rows.items():
        summary.append(verdict_row(example_id, verdict, rules[0] if rules else ''))
        for k, (t0, t1, period) in enumerate(eps):
            episodes.append((example_id, example_id.split('__')[2], k, True, t0, t1, period))
        # a rejected run is never an episode of the comparison
        episodes.append((example_id, example_id.split('__')[2], len(eps), False, 0, 10, 5))
    pd.DataFrame(summary, columns=SUMMARY).to_csv(folder / 'summary.csv', index=False)
    pd.DataFrame(episodes, columns=EPISODES).to_csv(folder / 'episodes.csv', index=False)


@pytest.fixture
def review_data(tmp_path):
    for example_id in NEW:
        folder = tmp_path / 'examples' / example_id
        folder.mkdir(parents=True)
        (folder / 'combined_v2.png').write_bytes(example_id.encode())
    write_run(tmp_path / 'base', BASE)
    write_run(tmp_path / 'new', NEW)
    discussed = tmp_path / 'discussed.csv'
    pd.DataFrame([('mmr2__E__1J-2+0+1', 'g01-1'), ('mmr2__H__1J-2+0+1', 'g02-5;s-1')], columns=['example_id', 'cards']).to_csv(
        discussed, index=False
    )
    bench = tmp_path / 'bench'
    bench.mkdir()
    pd.DataFrame(
        [('mmr2__H__1J-2+0+1', 'uncertain', '', 'episodic', 'author:2026-10-31 (W28 g02-5)')],
        columns=['example_id', 'status', 'confidence', 'angle_status', 'source'],
    ).to_csv(bench / 'author_label_decisions_2026-10-15.csv', index=False)
    return tmp_path


def run_review(data, *extra):
    argv = ['--review', 'w', '--run', 'new', '--baseline', 'base', '--data', str(data)]
    argv += ['--discussed', str(data / 'discussed.csv'), '--bench', str(data / 'bench'), *extra]
    return sf.main(argv)


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob('*')) if p.is_file()}


def test_review_folders(review_data):
    examples = snapshot(review_data / 'examples')
    assert run_review(review_data) == 0
    root = review_data / 'figures' / 'review-w'
    assert tree(root) == [
        'changed/resonant/high/mmr2__B__1J-2+0+1.png',
        'changed/resonant/high/mmr2__G__1J-2+0+1.png',
        'changed/resonant/high/mmr2__J__1J-2+0+1.png',
        'changed/transient/medium/mmr2__A__1J-2+0+1.png',
        'discussed/resonant/high/mmr2__E__1J-2+0+1.png',
        'discussed/uncertain/none/mmr2__H__1J-2+0+1.png',
        'same_rule/resonant/high/mmr2__D__1J-2+0+1.png',
    ]
    # hard links; the examples are untouched
    linked = root / 'same_rule' / 'resonant' / 'high' / 'mmr2__D__1J-2+0+1.png'
    assert linked.stat().st_ino == (review_data / 'examples' / 'mmr2__D__1J-2+0+1' / 'combined_v2.png').stat().st_ino
    assert snapshot(review_data / 'examples') == examples


def test_review_csv(review_data):
    run_review(review_data)
    frame = pd.read_csv(review_data / 'figures' / 'review-w' / 'review-w.csv', dtype=str, keep_default_na=False)
    assert list(frame.columns) == sf.REVIEW_COLUMNS
    rows = frame.set_index('example_id')
    assert sorted(rows.index) == [f'mmr2__{x}__1J-2+0+1' for x in 'ABDEGHJ']
    assert (frame['author_verdict'] == '').all() and (frame['author_comment'] == '').all()
    assert rows.loc['mmr2__A__1J-2+0+1', 'change_why'] == 'status; angle_status; confidence; reason'
    assert rows.loc['mmr2__A__1J-2+0+1', 'old_status'] == 'resonant'
    assert rows.loc['mmr2__A__1J-2+0+1', 'new_status'] == 'transient'
    assert rows.loc['mmr2__B__1J-2+0+1', 'change_why'] == 'episodes@main: shift 150 > 100 yr (100-1050)'
    assert rows.loc['mmr2__B__1J-2+0+1', 'old_episodes'] == '100-900 (P 100)'
    assert rows.loc['mmr2__B__1J-2+0+1', 'new_episodes'] == '100-1050 (P 100)'
    assert rows.loc['mmr2__G__1J-2+0+1', 'change_why'] == 'new row'
    assert rows.loc['mmr2__J__1J-2+0+1', 'change_why'] == 'episodes@main: count 1 -> 2'
    assert rows.loc['mmr2__D__1J-2+0+1', 'changed'] == 'False'
    # the rule's doc id in brackets (rules.RULES: 'pending' until W28 E16 wrote G2u)
    assert rows.loc['mmr2__D__1J-2+0+1', 'rules_fired'] == 'bounded_gap_merge [G2u]'
    assert rows.loc['mmr2__D__1J-2+0+1', 'rules_since'] == 'bounded_gap_merge'
    assert rows.loc['mmr2__H__1J-2+0+1', 'folder'] == 'discussed'
    assert rows.loc['mmr2__H__1J-2+0+1', 'changed'] == 'True'
    assert rows.loc['mmr2__H__1J-2+0+1', 'cards'] == 'g02-5;s-1'
    assert rows.loc['mmr2__H__1J-2+0+1', 'prior_label'] == 'uncertain//episodic'
    assert rows.loc['mmr2__H__1J-2+0+1', 'figure'] == 'discussed/uncertain/none/mmr2__H__1J-2+0+1.png'


def test_review_shift_limits_are_flags(review_data):
    """A limit of 0.4 P (40 yr) makes the 50-yr move of C a change; 2 P (200 yr) hides the 150-yr move of B."""
    run_review(review_data, '--shift-periods', '0.4')
    assert 'changed/resonant/high/mmr2__C__1J-2+0+1.png' in tree(review_data / 'figures' / 'review-w')
    run_review(review_data, '--shift-periods', '2')
    assert 'changed/resonant/high/mmr2__B__1J-2+0+1.png' not in tree(review_data / 'figures' / 'review-w')
    # the record part: 0.2 x 1000 yr = 200 yr also hides it
    run_review(review_data, '--shift-fraction', '0.2')
    assert 'changed/resonant/high/mmr2__B__1J-2+0+1.png' not in tree(review_data / 'figures' / 'review-w')


def test_review_dry_run_writes_nothing(review_data, capsys):
    run_review(review_data, '--dry-run')
    out = capsys.readouterr().out
    assert 'discussed 2, changed 4, same_rule 1' in out
    assert 'not in the run (baseline only): mmr2__gone__1J-2+0+1' in out
    assert not (review_data / 'figures').exists()


def test_review_refuses_a_path_as_name(review_data):
    with pytest.raises(ValueError):
        sf.main(['--review', '../x', '--run', 'new', '--baseline', 'base', '--data', str(review_data)])
    with pytest.raises(ValueError):
        sf.main(['--review', 'x', '--run', 'new', '--data', str(review_data)])


def test_episode_pairing_is_deterministic_when_counts_differ():
    E = sf.Episode
    # one old episode split in two: the larger overlap (500 yr with 0-500, 350 with 650-1000) pairs, 650-1000 is left
    pairs, lone_old, lone_new = sf.pair_episodes([E(0, 1000, 100)], [E(0, 500, 100), E(650, 1000, 100)])
    assert (pairs, lone_old, lone_new) == ([(0, 0)], [], [1])
    # equal overlaps (100 yr each): the smaller boundary shift wins (|0-0|, |300-200| = 100 against 200)
    pairs, lone_old, lone_new = sf.pair_episodes([E(100, 200, 50), E(0, 300, 50)], [E(100, 200, 50)])
    assert (pairs, lone_old, lone_new) == ([(0, 0)], [1], [])
    # no overlap: nothing pairs
    assert sf.pair_episodes([E(0, 100, 10)], [E(200, 300, 10)]) == ([], [0], [0])
    # a full tie (overlap 400 and shift 500 yr both ways): the earlier episode pairs, on every call
    old, new = [E(0, 400, 100), E(500, 900, 100)], [E(0, 900, 100)]
    assert sf.pair_episodes(old, new) == sf.pair_episodes(old, new) == ([(0, 0)], [1], [])


def test_episode_changes_by_scale():
    E = sf.Episode
    old = {'main': [E(100, 900, 100)]}
    # a slow-scale row only in the new run is a count change of that scale
    assert sf.episode_changes(old, {'main': [E(100, 900, 100)], 'slow': [E(0, 1000, 500)]}, 1000, 1, 0.02) == [
        'episodes@slow: count 0 -> 1'
    ]
    # a period missing on both sides: only the record part (0.02 x 1000 = 20 yr) limits
    assert sf.episode_changes({'main': [E(100, 900, math.nan)]}, {'main': [E(100, 930, math.nan)]}, 1000, 1, 0.02) == [
        'episodes@main: shift 30 > 20 yr (100-930)'
    ]


def test_review_with_an_empty_discussed_file(review_data):
    """An empty --discussed file names no record: E falls out, H (changed) goes to changed/."""
    (review_data / 'empty.csv').write_text('')
    sf.main(
        ['--review', 'w', '--run', 'new', '--baseline', 'base', '--data', str(review_data)]
        + ['--discussed', str(review_data / 'empty.csv'), '--bench', str(review_data / 'bench')]
    )
    files = tree(review_data / 'figures' / 'review-w')
    assert 'changed/uncertain/none/mmr2__H__1J-2+0+1.png' in files
    assert not any(f.startswith('discussed/') for f in files) and len(files) == 6
