"""The episodes the author asked for (benchmark/EPISODE_EXPECTATIONS.csv, quotes in benchmark/EPISODE_QUOTES.json),
checked by benchmark/tools/episode_expectations.py.

- The gate (`make test-benchmark`) reads the session's reclassifications (`reclassified_run` and, for the secular and
  ZLK cases, `secular_zlk_run`; conftest.py, shared with the disagreement gate). As KNOWN_DISAGREEMENTS.csv: a
  tier A/B case that fails and is not `known_fail` is a failure, so is a `known_fail` case that passes (update the
  file) and a case whose record is missing; tiers C (old verdicts) and D (analogies) are printed, never gate.
- The fast tests check the file itself (every quote verbatim at its line, every example and op known, the known
  failures well formed) and the ops on hand-made episode tables, each with a pass and a fail built by hand.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import bench_io as bio  # noqa: E402
import episode_expectations as ee  # noqa: E402

CASES = ee.load_cases()
NAN = float('nan')


@pytest.mark.benchmark
def test_episodes_are_the_ones_the_author_asked_for(reclassified_run, secular_zlk_run):
    table = ee.evaluate(reclassified_run, long_period=ee.read_run(secular_zlk_run))
    print('\n' + ee.format_table(table))
    found = ee.problems(table)
    assert found.empty, ee.format_problems(found)


# --- the file -------------------------------------------------------------------------------------------------------


def test_the_cases_are_well_formed():
    names = [case.case for case in CASES]
    assert len(names) == len(set(names)) == 79
    assert {case.tier for case in CASES} <= set(ee.TIERS)
    assert {case.scope for case in CASES} <= set(ee.SCOPES)
    for case in CASES:
        assert case.checks and case.quotes, case.case
        assert not case.known_fail or (case.gates and case.known_reason), case.case  # only A/B, always with a reason
    # the known failures at W23 S3 (attrib.json of the history design); E16 removed by the author 2026-10-26 (Q4)
    assert sorted(case.case for case in CASES if case.known_fail) == ['E15', 'E17', 'E21', 'E30', 'E43']


def test_every_quote_is_verbatim_at_its_line():
    """Files under tmp/ are not tracked: a quote there is checked where the file exists; a tracked file must hold it."""
    checked = 0
    for case in CASES:
        for quote in case.quotes:
            if not (bio.REPO / quote.file).exists():
                assert quote.file.startswith('tmp/'), (case.case, quote.file)
                continue
            assert ee.quote_line(quote.file, quote.text) == quote.line, (case.case, quote.file, quote.line)
            checked += 1
    assert checked > 0


def test_the_sources_are_the_quotes():
    for case in CASES:
        assert case.sources == '; '.join(f'{q.file}:{q.line}' for q in case.quotes), case.case
    quoted = json.loads(ee.QUOTES_JSON.read_text())
    assert [entry['case'] for entry in quoted] == [case.case for case in CASES]


def test_every_example_exists():
    known = set(bio.list_examples())
    if not known:
        pytest.skip(f'no saved benchmark examples in {bio.DATA}')
    assert [case.example_id for case in CASES if case.example_id not in known] == []


def test_every_op_is_known_with_its_parameters():
    for case in CASES:
        for check in ee.flatten(case.checks):
            assert ee.missing_keys(check) == [], (case.case, check)
    assert ee.missing_keys({'op': 'episode_near', 't0': 0}) == ['t1', 'tol']
    assert ee.missing_keys({'op': 'no_such_op'}) == ['op']
    with pytest.raises(ValueError):
        ee.check(ee.table_from_tuples([]), {'op': 'boundary_in', 't0': 1.0})


# --- the ops on hand-made tables ------------------------------------------------------------------------------------
# an episode: (t_start, t_end, n_tp, P, start gap [P], end gap [P], forced wave or None[, half-amplitude, centre [deg]])

COVERING = [(100, 19900, 60, 300, 0.4, 0.3, None)]
TWO = [(100, 9000, 30, 300, 0.4, 36, None), (9400, 19900, 30, 300, 1.3, 0.3, None)]
WAVES = [(89, 483, 6, 168, 0, 600, 'lattice_line')]
XO3 = [(90, 608, 9, 116, 0.5, 3, None), (668, 945, 5, 141, 5, 0.1, None)]
ZLK = [(24000, 820000, 27, 63000, 0.4, 10.7, None, 25.0, 270.0), (930000, 1486000, 17, 67000, 13.7, 0.2, None, 30.0, 90.0)]


@pytest.mark.parametrize(
    'episodes, check, expected',
    [
        (COVERING, {'op': 'single_covers'}, True),
        (TWO, {'op': 'single_covers'}, False),  # two episodes
        ([(100, 19000, 60, 300, 0.4, 3.0, None)], {'op': 'single_covers'}, False),  # 3 P short of the end
        (TWO, {'op': 'n_accepted', 'min': 2, 'max': 2}, True),
        (TWO, {'op': 'n_accepted', 'max': 1}, False),
        (WAVES, {'op': 'n_accepted', 'max': 0}, True),  # a forced wave is no episode
        (WAVES, {'op': 'n_accepted', 'min': 1, 'count_forced': True}, True),
        ([(31700, 38850, 12, 1175, 9, 9, None)], {'op': 'episode_near', 't0': 31700, 't1': 38900, 'tol': 700}, True),
        ([(31700, 40190, 13, 1175, 9, 8, None)], {'op': 'episode_near', 't0': 31700, 't1': 38900, 'tol': 700}, False),  # 1290 yr late
        ([(31700, 38850, 12, 1175, 9, 9, None)], {'op': 'episode_near', 't0': 31700, 't1': 38900, 'tol': 700, 'min_tp': 13}, False),
        (ZLK, {'op': 'episode_near', 't0': 930000, 't1': 1486000, 'tol': 30000, 'centre_deg': 90}, True),
        (ZLK, {'op': 'episode_near', 't0': 930000, 't1': 1486000, 'tol': 30000, 'centre_deg': 270}, False),  # the other island
        (TWO, {'op': 'single_episode_near', 't0': 100, 't1': 9000, 'tol': 100}, False),  # near, but not single
        (TWO[:1], {'op': 'single_episode_near', 't0': 100, 't1': 9000, 'tol': 100}, True),
        ([(31700, 38850, 12, 1175, 9, 9, None)], {'op': 'no_episode_in', 't0': 39500, 't1': 40200}, True),
        ([(31700, 40190, 13, 1175, 9, 8, None)], {'op': 'no_episode_in', 't0': 39500, 't1': 40200}, False),
        (WAVES, {'op': 'no_plain_episode'}, True),
        (COVERING, {'op': 'no_plain_episode'}, False),
        (WAVES, {'op': 'all_forced'}, True),
        (WAVES + COVERING, {'op': 'all_forced'}, False),
        ([], {'op': 'all_forced'}, False),  # nothing accepted is not "all forced"
        (XO3, {'op': 'boundary_in', 't0': 600, 't1': 800}, True),  # boundary 608-668
        (XO3, {'op': 'boundary_in', 't0': 250, 't1': 400}, False),
        (TWO, {'op': 'last_reaches_end'}, True),
        (TWO[:1], {'op': 'last_reaches_end'}, False),
        (TWO[:1], {'op': 'last_ends_before_end'}, True),
        ([], {'op': 'last_ends_before_end'}, False),
        ([(13100, 20000, 5, 1500, 8, 0, None)], {'op': 'first_start_before', 't': 12000}, True),  # 13100 <= 12000 + 1500
        ([(13600, 20000, 5, 1500, 8, 0, None)], {'op': 'first_start_before', 't': 12000}, False),
        ([(100, 900, 9, 100, 0, 0, None, 0.6)], {'op': 'max_half_amp', 'max_deg': 1.0}, True),
        ([(100, 900, 9, 100, 0, 0, None, 7.2)], {'op': 'max_half_amp', 'max_deg': 1.0}, False),
        (TWO, {'op': 'episode_covers_t', 't': 9000}, True),
        (TWO, {'op': 'episode_covers_t', 't': 9200}, False),  # inside the gap 9000-9400
        (TWO, {'op': 'any_of', 'alternatives': [[{'op': 'single_covers'}], [{'op': 'boundary_in', 't0': 9100, 't1': 9200}]]}, True),
        (TWO, {'op': 'any_of', 'alternatives': [[{'op': 'single_covers'}], [{'op': 'boundary_in', 't0': 5000, 't1': 6000}]]}, False),
    ],
)
def test_ops_on_hand_made_tables(episodes, check, expected):
    assert ee.run_check(ee.table_from_tuples(episodes), check) is expected


@pytest.mark.parametrize(
    'summary, numbers, check, expected',
    [
        ({}, {'geo_union_n_episodes': 2.0}, {'op': 'union', 'min': 2}, True),
        ({}, {'geo_union_n_episodes': 0.0}, {'op': 'union', 'min': 2}, False),
        ({}, {'geo_union_n_episodes': NAN}, {'op': 'no_union'}, True),
        ({}, {'geo_union_n_episodes': 3.0}, {'op': 'no_union'}, False),
        ({}, {'exit_observed': True}, {'op': 'exit_observed'}, True),
        ({}, {'exit_observed': False}, {'op': 'exit_observed'}, False),
        ({}, {'entry_observed': 'True'}, {'op': 'entry_observed'}, True),
        ({'final_regime': 'circulation'}, None, {'op': 'final_regime', 'value': 'circulation'}, True),
        ({'final_regime': 'libration'}, None, {'op': 'final_regime', 'value': 'circulation'}, False),
        ({'angle_status': 'libration', 'angle_subtype': 'stationary'}, None, {'op': 'angle', 'value': 'libration'}, True),
        (
            {'angle_status': 'libration', 'angle_subtype': 'stationary'},
            None,
            {'op': 'angle', 'value': 'libration', 'subtype': 'moving_centre'},
            False,
        ),
    ],
)
def test_ops_on_the_summary_and_the_numbers(summary, numbers, check, expected):
    assert ee.run_check(ee.table_from_tuples(COVERING, summary=summary, numbers=numbers), check) is expected


@pytest.mark.parametrize(
    'end_gap_p, window_gap_yr, step_max_yr, hidden_p, expected, exit_observed',
    [
        (1.07, 1400.0, 2851.0, 0.47, True, False),  # TX9 0.5 yr after G2a: no TP missed in the 1400 yr to the window's end
        (0.5, 9000.0, 2851.0, 3.0, True, False),  # within one period of the record's end
        (1.07, 2500.0, 2851.0, 0.47, False, False),  # longer than one period (2350 yr): a turning point may be missed
        (1.07, 1400.0, 1000.0, 0.47, False, False),  # longer than the longest step inside the episode
        (1.07, 1400.0, 2851.0, 1.3, False, False),  # the trim hides more than a period
        (1.07, 1400.0, 2851.0, 0.47, False, True),  # as the first, but the record shows the exit (`covers_edges`)
        (0.5, 9000.0, 2851.0, 3.0, False, True),  # an observed exit within one period of the end
    ],
)
def test_last_covers_end_reads_the_end_as_the_package_does(end_gap_p, window_gap_yr, step_max_yr, hidden_p, expected, exit_observed):
    frame = ee.table_from_tuples([(2211, 5923, 3, 3712, 0.6, 3.8, None), (7994, 17492, 8, 2350, 0.0, end_gap_p, None)]).runs
    frame = frame.assign(window_end_gap_yr=window_gap_yr, tp_step_max_yr=step_max_yr, window_end_hidden_P=hidden_p)
    frame = frame.assign(exit_observed=[False, exit_observed])
    assert ee.run_check(ee.Record({}, frame), {'op': 'last_covers_end'}) is expected


def test_return_cycle_and_runs_read_their_own_columns():
    episodes = pd.DataFrame(
        {
            't_start_yr': [79.0, 900.0, 950.0],
            't_end_yr': [453.0, 940.0, 990.0],
            'accepted': [True, False, False],
            'closed_by_return': [True, False, False],
            'return_t_a_yr': [79.0, NAN, NAN],
            'return_t_b_yr': [453.0, NAN, NAN],
        }
    )
    rec = ee.Record({}, episodes)
    assert ee.run_check(rec, {'op': 'return_cycle', 't_a': 100, 't_b': 400, 'tol': 60}) is True  # 2016 CA138: 80.1 -> 453.9
    assert ee.run_check(rec, {'op': 'return_cycle', 't_a': 100, 't_b': 400, 'tol': 20}) is False
    # n_runs_in counts runs whether accepted or not: two start in 900-1000
    assert ee.run_check(rec, {'op': 'n_runs_in', 't0': 900, 't1': 1000, 'max': 2}) is True
    assert ee.run_check(rec, {'op': 'n_runs_in', 't0': 900, 't1': 1000, 'max': 1}) is False


# --- the gate on a hand-made run ------------------------------------------------------------------------------------


def _case(name, example_id, tier, checks, known_fail=False):
    return ee.Case(name, example_id, tier, 'mmr', '', '', tuple(checks), '', known_fail, 'reason' if known_fail else '')


def test_the_gate_reports_new_resolved_and_missing_only_for_tiers_a_and_b():
    summary = pd.DataFrame(
        {
            'example_id': ['one', 'two', 'two'],
            'resonance': ['1J-2+0+1', '3J-1+0-2', '3J-1+1-3'],
            'parent': ['', '', '3J-1+0-2'],  # a member row of 'two' is not its record
            'angle_status': ['libration', 'episodic', 'libration'],
        }
    )
    episodes = pd.DataFrame(
        {
            'example_id': ['one', 'two', 'two'],
            'resonance': ['1J-2+0+1', '3J-1+0-2', '3J-1+1-3'],
            'accepted': [True, True, True],
            't_start_yr': [100.0, 500.0, 0.0],
            't_end_yr': [9900.0, 3000.0, 10000.0],
            'coverage_start_gap_P': [0.2, 5.0, 0.0],
            'coverage_end_gap_P': [0.1, 9.0, 0.0],
            'forced_wave': [NAN, NAN, NAN],
        }
    )
    tables = ee.RunTables(summary, episodes)
    covers, libration = [{'op': 'single_covers'}], [{'op': 'angle', 'value': 'libration'}]
    cases = [
        _case('A1', 'one', 'A', covers),  # passes
        _case('A2', 'two', 'A', covers),  # fails, not known: new
        _case('A3', 'two', 'B', libration, known_fail=True),  # fails, known
        _case('A4', 'one', 'A', libration, known_fail=True),  # passes but known: resolved
        _case('A5', 'three', 'A', covers),  # not in the run: missing
        _case('C1', 'two', 'C', covers),  # fails, tier C: printed only
        _case('D1', 'three', 'D', covers),
    ]
    table = ee.evaluate(tables, cases)
    assert list(table['result']) == ['PASS', 'FAIL', 'FAIL', 'PASS', 'MISSING', 'FAIL', 'MISSING']
    found = ee.problems(table)
    assert list(zip(found['problem'], found['case'])) == [('new', 'A2'), ('resolved', 'A4'), ('missing', 'A5')]
    assert ee.counts(table) == 'A: 2 PASS, 1 FAIL, 1 MISSING | B: 1 FAIL (1 known) | C: 1 FAIL (not gating) | D: 1 MISSING (not gating)'


def test_the_cover_threshold_is_one_period():
    """The edge rule of the cases: a gap of exactly 1 P still covers (DECISIONS, LIBRATION_COVERAGE_PERIODS = 1)."""
    assert ee.COVER_P == 1.0
    assert ee.run_check(ee.table_from_tuples([(0, 1, 2, 1, 1.0, 1.0, None)]), {'op': 'single_covers'})
    assert not ee.run_check(ee.table_from_tuples([(0, 1, 2, 1, 1.0, 1.0 + 1e-9, None)]), {'op': 'single_covers'})
    assert math.isclose(ee.CENTRE_TOL_DEG, 30.0)
