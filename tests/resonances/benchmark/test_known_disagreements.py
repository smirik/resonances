"""benchmark/tools/known_disagreements.py on small hand-made frames: what counts as a disagreement, the agreement
numbers, and the new / resolved / changed rows of the comparison with the list. Fast: no benchmark data."""

import sys
from pathlib import Path

import pandas as pd

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import known_disagreements as kd  # noqa: E402

NAN = float('nan')


def labels():
    """reclassify.agreement-shaped rows: package values and the author's (NaN: not given)."""
    rows = [
        # example_id, status, angle_status, confidence, author_status, author_angle_status, author_confidence
        ('a', 'resonant', 'libration', 'high', 'resonant', 'libration', 'high'),  # agrees on everything
        ('b', 'uncertain', 'episodic', NAN, 'resonant', 'libration', 'high'),  # status, angle, confidence differ
        ('c', 'transient', 'episodic', 'high', 'transient', NAN, NAN),  # author gave no angle, no level: agrees
        ('d', 'non_resonant', 'circulation', 'medium', 'non_resonant', 'circulation', 'high'),  # confidence only
        ('e', 'resonant', 'libration', 'high', NAN, NAN, 'low'),  # no author status: the row is not labelled
    ]
    columns = ['example_id', 'status', 'angle_status', 'confidence', 'author_status', 'author_angle_status', 'author_confidence']
    return pd.DataFrame(rows, columns=columns)


def verdicts():
    """agree_author_verdicts.agreement_frame-shaped rows."""
    return pd.DataFrame(
        {
            'example_id': ['a', 'f', 'g'],
            'status': ['resonant', 'uncertain', 'transient'],
            'author': ['resonant', 'non_resonant', 'transient'],
        }
    )


def known(rows):
    return pd.DataFrame(rows, columns=kd.COLUMNS)


def test_disagreements_only_where_the_author_gave_a_value():
    frame = kd.disagreements(labels(), verdicts())
    got = [tuple(r) for r in frame[['example_id', 'field', 'author', 'package']].itertuples(index=False)]
    assert got == [
        ('b', 'status', 'resonant', 'uncertain'),
        ('b', 'angle_status', 'libration', 'episodic'),
        ('b', 'confidence', 'high', ''),  # no level (uncertain) against the author's high
        ('d', 'confidence', 'high', 'medium'),
        ('f', 'author_verdict', 'non_resonant', 'uncertain'),
    ]


def test_agreement_numbers():
    numbers = kd.agreement_numbers(labels(), verdicts())
    # labelled rows a-d; angle given on a, b, d; level given on a, b, d; verdicts on a, f, g
    assert numbers == {'status': (3, 4), 'angle_status': (2, 3), 'confidence': (1, 3), 'author_verdict': (2, 3)}


def test_compare_finds_new_resolved_and_changed():
    current = kd.disagreements(labels(), verdicts())
    listed = known(
        [
            ('b', 'status', 'resonant', 'uncertain', 'cs12', 'r1', 'run1'),  # unchanged
            ('b', 'angle_status', 'libration', 'drift', 'cs12', 'r2', 'run1'),  # package value changed
            ('b', 'confidence', 'medium', '', 'cs12', 'r3', 'run1'),  # author value changed
            ('d', 'confidence', 'high', 'medium', 'calibration', 'r4', 'run1'),  # unchanged
            ('z', 'status', 'resonant', 'uncertain', 'open_question', 'r5', 'run1'),  # agrees now
        ]
    )
    problems = kd.compare(current, listed)
    got = sorted(tuple(r) for r in problems[['problem', 'example_id', 'field']].itertuples(index=False))
    assert got == [
        ('changed', 'b', 'angle_status'),
        ('changed', 'b', 'confidence'),
        ('new', 'f', 'author_verdict'),
        ('resolved', 'z', 'status'),
    ]
    row = problems[(problems.example_id == 'b') & (problems.field == 'angle_status')].iloc[0]
    assert (row.package_known, row.package_now) == ('drift', 'episodic')
    assert kd.compare(current, current.assign(category='x', reason='y', since='s')).empty
    assert 'resolved' in kd.format_problems(problems)


def test_write_keeps_reasons_and_marks_new_rows(tmp_path):
    path = tmp_path / 'KNOWN.csv'
    path.write_text(
        '# a comment, kept\n' + known([('b', 'status', 'resonant', 'uncertain', 'cs12', 'kept reason', 'run1')]).to_csv(index=False)
    )
    current = kd.disagreements(labels(), verdicts())
    written = kd.write_known(current, since='run2', path=path)
    assert path.read_text().startswith('# a comment, kept\n')
    back = kd.read_known(path)
    pd.testing.assert_frame_equal(back, written.astype(str))
    status = back[(back.example_id == 'b') & (back.field == 'status')].iloc[0]
    assert (status.category, status.reason, status.since) == ('cs12', 'kept reason', 'run1')
    new = back[(back.example_id == 'f')].iloc[0]
    assert (new.category, new.reason, new.since) == (kd.UNREVIEWED, '', 'run2')
    assert kd.compare(current, back).empty


def test_the_tracked_list_is_well_formed():
    frame = kd.read_known()
    assert list(frame.columns) == kd.COLUMNS
    assert not frame.duplicated(kd.KEY).any()
    assert set(frame['field']) <= set(kd.FIELDS)
    assert (frame['category'] != '').all() and (frame['reason'] != '').all()


def test_author_verdict_skips_records_with_their_own_label():
    """R-AV (author 2026-10-24): candidates.csv's old author_verdict is not compared on a record whose author label
    has a status; a label without a status (only a level) keeps the comparison."""
    import agree_author_verdicts as agree

    frame = pd.DataFrame(
        {
            'example_id': ['a', 'f', 'g', 'h'],
            'status': ['resonant', 'uncertain', 'transient', 'resonant'],
            'author': ['resonant', 'non_resonant', 'transient', 'transient'],
        }
    )
    author = pd.DataFrame({'example_id': ['f', 'h', 'z'], 'author_status': ['uncertain', '  ', 'resonant']})
    kept = agree.without_labelled(frame, author)
    assert kept['example_id'].tolist() == ['a', 'g', 'h']  # f has its own label; h's label has no status
    assert kd.agreement_numbers(labels(), kept)['author_verdict'] == (2, 3)  # a, g agree; h does not


def test_long_period_verdicts_against_a_secular_zlk_run():
    """The author's secular/ZLK verdicts (AUTHOR_VERDICTS-shaped) against a reclassify_secular_zlk summary: an agreeing
    record is no row, a differing one is an author_verdict row, a record missing from the run is 'missing', and a
    record with its own author label is left to the label."""
    verdicts = {
        'secular__1020__g-g6+s-s6': ('resonant', 's1'),
        'zlk__591986__LK': ('non_resonant', 's2'),
        'secular__363__g-g6+s-s6': ('uncertain', 's3'),
        'zlk__162474__LK__1500kyr': ('transient', 's4'),
    }
    summary = pd.DataFrame(
        {
            'example_id': ['secular__1020__g-g6+s-s6', 'zlk__591986__LK', 'zlk__162474__LK__1500kyr'],
            'status': ['resonant', 'resonant', 'resonant'],
        }
    )
    author = pd.DataFrame({'example_id': ['zlk__162474__LK__1500kyr'], 'author_status': ['transient']})
    frame = kd.long_period_verdicts(summary, author, verdicts)
    assert frame[['example_id', 'status', 'author']].values.tolist() == [
        ['secular__1020__g-g6+s-s6', 'resonant', 'resonant'],
        ['zlk__591986__LK', 'resonant', 'non_resonant'],
        ['secular__363__g-g6+s-s6', 'missing', 'uncertain'],
    ]
    got = kd.disagreements(labels().iloc[:0], frame)
    assert [tuple(r) for r in got[['example_id', 'field', 'author', 'package']].itertuples(index=False)] == [
        ('secular__363__g-g6+s-s6', 'author_verdict', 'uncertain', 'missing'),
        ('zlk__591986__LK', 'author_verdict', 'non_resonant', 'resonant'),
    ]


def test_the_package_verdicts_list_is_the_author_reading():
    """The gated secular/ZLK verdicts: 14 records, each with its source, statuses in the package's spelling."""
    import reclassify_secular_zlk as szlk

    from resonances.resonance.analysis.verdict import Status

    assert len(szlk.AUTHOR_VERDICTS) == 14
    assert all(kd.is_long_period(example_id) for example_id in szlk.AUTHOR_VERDICTS)
    assert all(status in {str(s) for s in Status} and source for status, source in szlk.AUTHOR_VERDICTS.values())


def test_without_the_secular_zlk_run_its_listed_rows_are_not_checked():
    listed = known(
        [
            ('b', 'status', 'resonant', 'uncertain', 'cs12', 'r1', 'run1'),
            ('zlk__3040__LK', 'author_verdict', 'resonant', 'transient', 'open_question', 'r2', 'run1'),
        ]
    )
    assert kd.checked_known(listed, None)['example_id'].tolist() == ['b']
    assert kd.checked_known(listed, 'secular-run')['example_id'].tolist() == ['b', 'zlk__3040__LK']
