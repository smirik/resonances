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
    # W28 F3: 30 cases added, E17 rewritten, C04 and C11 replaced; A1: E65 (2000 SF8); A2: E98 (78085); D2: E85, E85b (555527)
    # D6: E86 (2306); D3 moved D01 (88899) to tier B; B6b: E84 (2004 YG1 20 kyr); B6c: E76 (523683 2U-9); B7: C28 (385533)
    # B10a: C38 (37087 slow scale), E94, E95 (37087 x2 K3s), C39 (HW7 slow cause); C3: E68 (2003 YN107), C21 (2026 AC4);
    # C4: E69 (2006 BZ147), E67 (2007 VU6); C8: E70 (2009 BE81 20 kyr, tier C); C6: C25, C26, C40-C53 (tier C, leaf G4c)
    # C14: C54 (2022 BB2, flag extend_time, tier C); W29 (2026-11-02 review page): E100-E105 (no episodes / at most one)
    # W36 (the author's review of the pairs secular wave 1, 2026-11-07): E107-E117 (A/B), C55-C60 (Codex/Claude, tier C)
    # W36 F2 guard: C61 (455327 g-s-g5+s6, Codex, tier C: the episode the local prominence lost)
    assert len(names) == len(set(names)) == 171
    assert {case.tier for case in CASES} <= set(ee.TIERS)
    assert {case.scope for case in CASES} <= set(ee.SCOPES)
    for case in CASES:
        assert case.checks and case.quotes, case.case
        assert not case.known_fail or (case.gates and case.known_reason), case.case  # only A/B, always with a reason
    # the known failures at W23 S3 (attrib.json of the history design); E16 removed by the author 2026-10-26 (Q4); E17
    # rewritten by his W28 answer g11-4 A (two episodes allowed) and passing; E21 and E30 pass with rules q and s (B1);
    # E15 passes with the forced (strength) mark of rule 5s (B2, g11-1 A); E67 (2007 VU6): its edge turn swings back
    # 18.8 deg, below MMR_COORBITAL_EDGE_TURN_MIN_DEG (C4, for F9); E43 (2021 XO3) passes as one episode with its
    # amplitude changes as marks (C7, g03-1 A), and E07-E09 are one episode each (g03-9 A)
    # W29 (minimal policy, 2026-11-02): vetoed episodes are candidates (E02, E28, E31, E46, E47, E64), a subresonance
    # parent's runs are its member's (E21, E77, E79), ZLK gate flag (D01); W33: E60 (WK15) passes, single_covers reads
    # each edge as geometry.covers_edges does; E76 (523683) passes, a slow swing's baseline sets the extension
    # W34 (the author's epcount A): the nine cases read the geometric pool (counted + candidates) and pass; D01 (the
    # gate reads 88899's episodes) and E67 (uncertain + extend, his e67 A) rewritten: no known failure left
    # W36 test-first: the eleven secular cases of 2026-11-07 fail on the unchanged classifier; each fix (F1-F4) clears
    # its own: E107/E108 (exit at the end), E109/E110 (prominence), E111 (moving centre), E112-E117 (ripple, slow scale);
    # F1 (the edge rule on sigma_f) cleared E107/E108, F2 (each run at its own prominence) E109/E110, F3 (a bounded
    # run whose centre turns back) E111, F4 (slow waves, the centre-turns scale) E112, E113, E116; E114 (142694), E115
    # (165548: medium, not low) and E117 (455327 s-s3) stayed known failures; the author accepted the package's reading
    # of all three on 2026-11-08 (transient; medium; the episode 9.20-10.84 Myr), so they pass: no known failure left
    assert not [case.case for case in CASES if case.known_fail]


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


@pytest.mark.benchmark
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
    assert ee.missing_keys({'op': 'field', 'column': 'reason'}) == ['value|contains|min']  # what to compare is missing
    assert ee.missing_keys({'op': 'field', 'column': 'reason', 'value': 'x', 'min': 1}) == ['value|contains|min']  # two
    assert ee.missing_keys({'op': 'episode_mark', 't0': 330, 't1': 450, 'tol': 30}) == ['mark']
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
        # period_in, decisive: the decisive episode's period from mmr_numbers.csv (MMR) or the summary (secular/ZLK)
        ({}, {'ep_period_med_yr': 100.0}, {'op': 'period_in', 'lo': 95, 'hi': 120}, True),
        ({}, {'ep_period_med_yr': 130.0}, {'op': 'period_in', 'lo': 95, 'hi': 120, 'which': 'decisive'}, False),
        ({'ep_period_med_yr': '100.0'}, None, {'op': 'period_in', 'lo': 95, 'hi': 120}, True),
        ({}, {'ep_period_med_yr': NAN}, {'op': 'period_in', 'lo': 95, 'hi': 120}, False),  # no decisive episode
        # period_in, any: COVERING's one episode has P = 300 yr
        ({}, None, {'op': 'period_in', 'lo': 250, 'hi': 350, 'which': 'any'}, True),
        ({}, None, {'op': 'period_in', 'lo': 95, 'hi': 120, 'which': 'any'}, False),
        # action: extend to 30 000 yr
        ({'action': 'extend', 'extend_to_years': '30000.0'}, None, {'op': 'action', 'value': 'extend', 'min_extend_to': 25000}, True),
        ({'action': 'extend', 'extend_to_years': '30000.0'}, None, {'op': 'action', 'value': 'extend', 'min_extend_to': 40000}, False),
        ({'action': 'extend', 'extend_to_years': ''}, None, {'op': 'action', 'value': 'extend', 'min_extend_to': 25000}, False),
        ({'action': 'none', 'extend_to_years': ''}, None, {'op': 'action', 'value': 'extend'}, False),
        # flag: present (default) and absent
        ({'flags': 'jump_locator_aliased;zone_eq_only'}, None, {'op': 'flag', 'value': 'zone_eq_only'}, True),
        ({'flags': 'jump_locator_aliased;zone_eq_only'}, None, {'op': 'flag', 'value': 'zone_eq_only', 'present': False}, False),
        ({'flags': 'jump_locator_aliased'}, None, {'op': 'flag', 'value': 'second_capture_unresolved', 'present': False}, True),
        ({'flags': ''}, None, {'op': 'flag', 'value': 'second_capture_unresolved'}, False),
        # field: equal (numbers compared as numbers), an item of a ';' list (not a substring), at least min, no column
        ({'reason': 'kinematic_libration'}, None, {'op': 'field', 'column': 'reason', 'value': 'kinematic_libration'}, True),
        ({'reason': 'induced_libration'}, None, {'op': 'field', 'column': 'reason', 'value': 'kinematic_libration'}, False),
        ({'extend_to_years': '60000.0'}, None, {'op': 'field', 'column': 'extend_to_years', 'value': 60000}, True),
        ({'rules_fired': 'owner_per_episode;frame_join'}, None, {'op': 'field', 'column': 'rules_fired', 'contains': 'frame_join'}, True),
        ({'rules_fired': 'owner_per_episode;frame_join'}, None, {'op': 'field', 'column': 'rules_fired', 'contains': 'frame'}, False),
        ({'rules_fired': ''}, None, {'op': 'field', 'column': 'rules_fired', 'contains': 'frame_join'}, False),
        ({'extend_to_years': '30000'}, None, {'op': 'field', 'column': 'extend_to_years', 'min': 25000}, True),
        ({'extend_to_years': '20000'}, None, {'op': 'field', 'column': 'extend_to_years', 'min': 25000}, False),
        ({}, None, {'op': 'field', 'column': 'gate_outcome', 'value': 'unavailable'}, False),  # no such column
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


def _scaled_runs():
    """Three accepted episodes: one at the main scale by name, one without a scale (main), one slow (P 5700 yr)."""
    return pd.DataFrame(
        {
            't_start_yr': [100.0, 9400.0, 17200.0],
            't_end_yr': [9000.0, 19900.0, 62400.0],
            'period_med_yr': [300.0, 300.0, 5700.0],
            'accepted': [True, True, True],
            'scale': ['main', NAN, 'slow'],
        }
    )


@pytest.mark.parametrize(
    'check, expected',
    [
        ({'op': 'n_accepted', 'min': 2, 'max': 2}, True),  # main by default: the named row and the empty one
        ({'op': 'n_accepted', 'min': 1, 'max': 1, 'scale': 'slow'}, True),
        ({'op': 'n_accepted', 'max': 0, 'scale': 'fast'}, True),  # no rows at that scale
        ({'op': 'episode_near', 't0': 17200, 't1': 62400, 'tol': 2000, 'scale': 'slow'}, True),
        ({'op': 'episode_near', 't0': 17200, 't1': 62400, 'tol': 2000}, False),  # not a main-scale episode
        ({'op': 'period_in', 'lo': 5000, 'hi': 6500, 'which': 'any', 'scale': 'slow'}, True),
        ({'op': 'period_in', 'lo': 5000, 'hi': 6500, 'which': 'any'}, False),
        ({'op': 'any_of', 'alternatives': [[{'op': 'single_covers'}], [{'op': 'n_accepted', 'min': 1, 'scale': 'slow'}]]}, True),
    ],
)
def test_the_scale_key_selects_the_episode_rows(check, expected):
    assert ee.run_check(ee.Record({}, _scaled_runs()), check) is expected


def test_a_run_without_a_scale_column_is_the_main_scale():
    rec = ee.table_from_tuples(TWO)
    assert ee.run_check(rec, {'op': 'n_accepted', 'min': 2, 'max': 2, 'scale': 'main'})
    assert ee.run_check(rec, {'op': 'n_accepted', 'max': 0, 'scale': 'slow'})
    with pytest.raises(ValueError):  # the decisive period belongs to the main scale
        ee.check(rec, {'op': 'period_in', 'lo': 5000, 'hi': 6500, 'scale': 'slow'})


@pytest.mark.parametrize(
    'check, expected',
    [
        ({'op': 'episode_mark', 't0': 330, 't1': 450, 'tol': 30, 'mark': 'amplitude_change'}, True),
        ({'op': 'episode_mark', 't0': 710, 't1': 810, 'tol': 30, 'mark': 'amplitude_change'}, True),
        ({'op': 'episode_mark', 't0': 500, 't1': 600, 'tol': 30, 'mark': 'amplitude_change'}, False),  # none there
        ({'op': 'episode_mark', 't0': 348, 't1': 348, 'tol': 5, 'mark': 'kick'}, True),  # a mark at one time
        ({'op': 'episode_mark', 't0': 330, 't1': 450, 'tol': 30, 'mark': 'sub_regime'}, False),  # absent
        ({'op': 'episode_mark', 't0': 8600, 't1': 9240, 'tol': 10, 'mark': 'forced_strength'}, True),  # spans its run
    ],
)
def test_episode_marks(check, expected):
    runs = pd.DataFrame(
        {
            't_start_yr': [90.0, 8600.0],
            't_end_yr': [945.0, 9240.0],
            'accepted': [True, False],  # a mark on a rejected run counts too
            'marks': ['amplitude_change@330:450;kick@348;amplitude_change@710:810', 'forced_strength'],
        }
    )
    assert ee.run_check(ee.Record({}, runs), check) is expected


def test_episode_mark_fails_when_the_marks_are_not_written():
    good, note = ee.check(ee.table_from_tuples(XO3), {'op': 'episode_mark', 't0': 330, 't1': 450, 'tol': 30, 'mark': 'amplitude_change'})
    assert (good, note) == (False, 'marks not written')


def _cores(rows):
    """A hand-made W28 C8 episodes table: (accepted, core t0, core t1, residence t0, t1, bounded at the start, end)."""
    names = ['accepted', 'core_t0_yr', 'core_t1_yr', 'residence_t0_yr', 'residence_t1_yr', 'residence_bounded_by_start']
    frame = pd.DataFrame([dict(zip(names + ['residence_bounded_by_end'], row)) for row in rows])
    frame['t_start_yr'] = frame['core_t0_yr'].fillna(0.0)
    frame['t_end_yr'] = frame['core_t1_yr'].fillna(0.0)
    return ee.Record({}, frame)


# 2020 CB 1E-1 as the package reads it (W28 C8): a rejected over-the-top run, then the quasi-satellite core 443-601 yr
# whose inferred residence runs from 443 yr (a leg of 3.1 turns before it) to the record's end (1000 yr)
CB = [(False, NAN, NAN, NAN, NAN, '', ''), (True, 442.9, 600.5, 442.9, 1000.0, 'circulation', 'edge')]
# 2024 TY11 1M-1 20 kyr: the core 2.65-8.69 kyr, its residence from the record's start to the core's end (4 turns after)
TY11 = [(True, 2652.2, 8687.2, 0.0, 8687.2, 'edge', 'circulation')]


@pytest.mark.parametrize(
    'rows, check, expected',
    [
        (CB, {'op': 'core_near', 't0': 443, 't1': 601, 'tol': 30}, True),
        (CB, {'op': 'core_near', 't0': 443, 't1': 1000, 'tol': 30}, False),  # the residence is no core
        (CB, {'op': 'residence_reaches_end'}, True),
        (TY11, {'op': 'residence_reaches_end'}, False),  # a circulation bounds it
        (TY11, {'op': 'residence_starts_before', 't': 1000}, True),
        (CB, {'op': 'residence_starts_before', 't': 400}, False),  # 443 > 400
        (TY11, {'op': 'core_near', 't0': 2654, 't1': 8688, 'tol': 100}, True),
    ],
)
def test_core_and_residence_ops(rows, check, expected):
    assert ee.run_check(_cores(rows), check) is expected


def test_core_and_residence_ops_fail_when_the_columns_are_not_written():
    rec = ee.table_from_tuples(XO3)
    assert ee.check(rec, {'op': 'core_near', 't0': 90, 't1': 945, 'tol': 30}) == (False, 'core not written')
    assert ee.check(rec, {'op': 'residence_reaches_end'}) == (False, 'residence not written')
    assert ee.check(rec, {'op': 'residence_starts_before', 't': 100}) == (False, 'residence not written')


@pytest.mark.parametrize('value, check, expected', [('False', {}, False), ('True', {}, True), ('False', {'value': False}, True)])
def test_exit_observed_takes_the_value_it_expects(value, check, expected):
    rec = ee.Record({}, pd.DataFrame({'t_start_yr': [0.0], 't_end_yr': [1.0], 'accepted': [True]}), {'exit_observed': value})
    assert ee.run_check(rec, {'op': 'exit_observed', **check}) is expected


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


@pytest.mark.parametrize('hidden, covered', [(0.52, True), (1.2, False)])
def test_single_covers_reads_an_edge_as_the_package_does(hidden, covered):
    # 2001 WK15 (E60): the start gap is 1.0000000004 P, but no turning point is missed at the window's start (70 yr <
    # min(longest TP step 180 yr, P 145 yr)) and the filter trim hides 0.52 P: covered (geometry.covers_edges)
    frame = ee.table_from_tuples([(145.0, 19920.0, 274, 145.0, 1.0000000004, 0.55, None)]).all_episodes
    frame = frame.assign(window_start_gap_yr=70.0, window_start_hidden_P=hidden, tp_step_max_yr=180.0)
    assert ee.run_check(ee.Record({}, frame), {'op': 'single_covers'}) == covered


def test_the_geometric_pool_reads_the_candidates_too():
    # 433 Eros 1R-5V+7 (E46, the author's epcount A): two episodes the owner veto made candidates; the counted pool is
    # empty under the minimal policy, the geometric one holds both
    frame = ee.table_from_tuples([(3520.0, 6620.0, 7, 1000.0, 3.5, 4.0, None), (7620.0, 10620.0, 7, 1000.0, 7.6, 1.0, None)]).all_episodes
    frame = frame.assign(counted=False, candidate=True)
    record = ee.Record({'policy': 'minimal'}, frame)
    two = {'op': 'n_accepted', 'min': 2, 'max': 2}
    assert not ee.run_check(record, two)
    assert ee.run_check(record, {**two, 'pool': 'geometric'})
    # a run neither counted nor a candidate (an accepted run below the floor) is in neither pool
    record = ee.Record({'policy': 'minimal'}, frame.assign(candidate=[True, False]))
    assert ee.run_check(record, {'op': 'n_accepted', 'min': 1, 'max': 1, 'pool': 'geometric'})
    with pytest.raises(ValueError):
        ee.run_check(record, {**two, 'pool': 'all'})
