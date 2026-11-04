"""The W29 acceptance script (`benchmark/tools/w29_acceptance.py`) on tiny hand-made runs: every invariant catches a
violation planted in an otherwise consistent row, the level and extend tables give the values the spec's tables give
by hand, the clear sets and hard swaps classify hand-picked rows, and the pre-label whitelist picks the newest plain
final run before a label's date. Also the W29 parts of the other benchmark tools: `--policy` of the reclassify tools
and the counted episodes of `episode_expectations` under the minimal policy. No benchmark data is read."""

import json

import pandas as pd
import pytest

import episode_expectations as ee
import reclassify
import w29_acceptance as acc
from resonances.resonance.analysis.settings import AnalysisSettings

S = AnalysisSettings()  # LIBRATION_MIN_TURNING_POINTS 3: clause B holds when P_dec > 2 T / 3


def row(**fields):
    """A consistent minimal-policy row: an MMR libration of 12 cycles over the whole record, certain, no extend (its end
    travels 0.2 rev, N_last 12 >= 3, P_dec 1000 yr on a 50 kyr record), one counted episode."""
    base = {
        'example_id': 'mmr2__1__2J-1+0+1',
        'resonance': '2J-1+0+1',
        'parent': '',
        'policy': 'minimal',
        'status': 'resonant',
        'angle_status': 'libration',
        'confidence': 'certain',
        'reason': 'dynamic_libration',
        'action': 'none',
        'flags': '',
        'caps': '',
        'cell_n_examples': '',
        'veto': '',
        'n_cycles': '12',
        'n_counted_episodes': '1',
        'counted_episode_ids': '0',
        'candidate_episode_ids': '',
        'confidence_basis': 'N=12(episode);M=full;V=clean;S=full;veto=-',
        'decision_path': 'G0 orbit is physical',
        'pol_end_travel_rev': '0.2',
        'pol_last_reaches_end': 'True',
        'pol_n_last': '12',
        'pol_period_dec_yr': '1000',
        'pol_extend_clause': '',
        'smp_span_yr': '50000',
        'geo_centre_max_pass_rev': '0.1',
        'geo_n_tp': '25',
        'status_code': '2',
        'check_zlk_free_omega': '',
    }
    base.update(fields)
    return base


EPISODES = pd.DataFrame([{'example_id': 'mmr2__1__2J-1+0+1', 'resonance': '2J-1+0+1', 'episode_id': '0', 'counted': 'True'}])


def violations(*rows, episodes=EPISODES):
    found = acc.invariants(pd.DataFrame(list(rows)), episodes, S)
    return sorted(set(found['invariant']))


def test_a_consistent_row_has_no_violation():
    assert violations(row()) == []


@pytest.mark.parametrize(
    'fields, invariant',
    [
        (dict(status='non_resonant', reason='circulation', confidence='high', n_counted_episodes='0', counted_episode_ids=''), '1'),
        (dict(geo_centre_max_pass_rev='1.4', n_counted_episodes='0', counted_episode_ids=''), '2'),
        (dict(status='non_resonant', reason='kinematic_libration', confidence='high', n_cycles='2'), '3'),
        (dict(veto='identity'), '3'),
        (dict(reason='modulated_circulation', status='non_resonant', confidence='high', n_cycles='2', geo_centre_max_pass_rev='0.4'), '3'),
        (dict(action='extend'), '4'),
        (dict(pol_n_last='1.6', confidence_basis='N=1.6(episode);M=full;V=clean;S=full;veto=-', n_cycles='1.6', confidence='medium'), '4'),
        (dict(confidence='high'), '5'),
        (dict(caps='no_full_physics_support'), '5'),
        (dict(n_cycles='11'), '5'),
        (dict(status='uncertain', confidence='', action='none', confidence_basis=''), '6'),
        (dict(n_counted_episodes='2'), '8'),
        (dict(policy='v2'), '0'),
    ],
)
def test_each_invariant_catches_a_planted_violation(fields, invariant):
    assert invariant in violations(row(**fields))


def test_the_extend_clauses_recomputed_from_the_policy_columns():
    """§5 by hand: an unfinished libration (N_last 1.5 < 2, W31) at the end extends (A); with 5 cycles it does not; B: P_dec
    20 kyr > 2 x 50 kyr / 3 = 33.3 kyr? no -> no extend; 40 kyr > 33.3 kyr -> extend; an end that circulates (1.2 rev)
    never extends; a start-edge feature (A1 with no ':end' leg) does not either."""
    assert acc.extend_expected(row(pol_n_last='1.5'), S) == (True, 'A')
    assert acc.extend_expected(row(pol_n_last='5'), S) == (False, '')
    assert acc.extend_expected(row(pol_last_reaches_end='False', pol_period_dec_yr='20000'), S) == (False, '')
    assert acc.extend_expected(row(pol_last_reaches_end='False', pol_period_dec_yr='40000'), S) == (True, 'B')
    assert acc.extend_expected(row(pol_n_last='1.5', pol_end_travel_rev='1.2'), S) == (False, '')
    leg = row(pol_last_reaches_end='False', pol_extend_clause='A1', geo_leg_read='839-994:start:1/1:0.8:pass/pass')
    assert acc.extend_expected(leg, S) == (False, '')
    assert acc.extend_expected({**leg, 'geo_leg_read': '839-994:end:1/1:0.8:pass/pass'}, S) == (True, 'A1')
    arc = row(pol_last_reaches_end='False', pol_period_dec_yr='', geo_n_tp='0', angle_status='circulation')
    assert acc.extend_expected(arc, S) == (True, 'B')


def test_clause_c_extends_a_veto_that_needs_length_within_the_budget():
    """Q5 (2026-11-02): a veto not measured for lack of length extends when the record it needs is within
    MMR_NEXT_CHECK_SPAN_FACTOR x T = 4 x 50 kyr = 200 kyr: 150 kyr and exactly 200 kyr extend, 200.001 kyr does not,
    nor does an unnamed length (blank) or the flag missing, nor an end that circulates."""
    base = row(pol_last_reaches_end='False', pol_period_dec_yr='1000', flags='veto_needs_length')
    assert acc.extend_expected({**base, 'pol_veto_need_yr': '150000'}, S) == (True, 'C')
    assert acc.extend_expected({**base, 'pol_veto_need_yr': '200000'}, S) == (True, 'C')
    assert acc.extend_expected({**base, 'pol_veto_need_yr': '200001'}, S) == (False, '')
    assert acc.extend_expected({**base, 'pol_veto_need_yr': ''}, S) == (False, '')
    assert acc.extend_expected({**base, 'pol_veto_need_yr': '150000', 'flags': ''}, S) == (False, '')
    assert acc.extend_expected({**base, 'pol_veto_need_yr': '150000', 'pol_end_travel_rev': '1.2'}, S) == (False, '')


def test_an_alias_circulation_is_certain_on_c5_and_c2():
    """Q18 (2026-11-02): an alias has no C1; C5 full and C2 full give certain, either thin gives high; C1 does not
    stand in for C5 on an alias."""
    basis = acc.parse_basis('N=0(episode);M=not_measured;V=clean;S=full;veto=-')
    alias = {'angle_subtype': 'alias', 'status': 'non_resonant', 'reason': 'circulation'}
    assert acc.level('non_resonant', basis, row(**alias, mrg_C5='full', mrg_C2='full'), S) == 'certain'
    assert acc.level('non_resonant', basis, row(**alias, mrg_C5='thin', mrg_C2='full'), S) == 'high'
    assert acc.level('non_resonant', basis, row(**alias, mrg_C5='full', mrg_C2='thin'), S) == 'high'
    assert acc.level('non_resonant', basis, row(**alias, mrg_C1='full', mrg_C2='full'), S) == 'high'


NO_FORCE = {'check_mmr_a_response': 'not_measured', 'check_mmr_potential': 'fail', 'check_mmr_strength': 'not_measured'}


@pytest.mark.parametrize(
    'status, basis, extra, level',
    [
        ('resonant', 'N=40(episode);M=full;V=clean;S=full;veto=-', {}, 'certain'),
        ('resonant', 'N=40(episode);M=thin;V=clean;S=full;veto=-', {}, 'high'),
        ('resonant', 'N=9.5(episode);M=full;V=clean;S=full;veto=-', {}, 'certain'),  # W31: certain from 3 cycles
        ('resonant', 'N=9.5(cover);M=full;V=clean;S=full;veto=-', {'ep_n_tp': '5'}, 'high'),  # 2 cycles in the episode
        ('resonant', 'N=3(episode);M=full;V=partial;S=thin;veto=-', {}, 'high'),
        ('resonant', 'N=40(episode);M=full;V=not_measured;S=full;veto=-', {}, 'medium'),
        ('transient', 'N=2.5(episode);M=full;V=clean;S=full;veto=-', {}, 'high'),  # W31: high from 2 cycles
        ('transient', 'N=1.6(episode);M=full;V=clean;S=full;veto=-', {}, 'medium'),
        ('transient', 'N=0.5(leg);M=full;V=clean;S=full;veto=-', {}, 'low'),  # W30: below 1.5 cycles
        ('transient', 'N=40(episode);M=full;V=ambiguous;S=full;veto=-', {}, 'low'),
        ('near_separatrix', 'N=5(episode);M=thin;V=clean;S=full;veto=-', {}, 'medium'),
        ('non_resonant', 'N=4(episode);M=full;V=clean;S=full;veto=owner', {'veto': 'owner'}, 'high'),
        ('non_resonant', 'N=4(episode);M=full;V=clean;S=full;veto=owner:thin', {'veto': 'owner'}, 'medium'),
        ('non_resonant', 'N=0(episode);M=not_measured;V=clean;S=full;veto=-', {'candidate_episode_ids': 'leg'}, 'medium'),
        ('non_resonant', 'N=0(episode);M=not_measured;V=clean;S=full;veto=-', {'mrg_C1': 'full', 'mrg_C2': 'full'}, 'certain'),
        ('non_resonant', 'N=0(episode);M=not_measured;V=clean;S=full;veto=-', {'mrg_C1': 'full', 'mrg_C2': 'thin'}, 'high'),
        ('uncertain', '', {}, ''),
        # W33 F: no force check passes (the identity is no force) -> at most low; a row without the columns predates it
        ('resonant', 'N=40(episode);M=full;V=clean;S=full;veto=-', NO_FORCE, 'low'),
        ('resonant', 'N=40(episode);M=full;V=clean;S=full;veto=-', {**NO_FORCE, 'check_mmr_potential': 'pass'}, 'certain'),
        ('resonant', 'N=40(episode);M=full;V=clean;S=full;veto=-', {**NO_FORCE, 'resonance_type': 'secular'}, 'certain'),
    ],
)
def test_the_level_table(status, basis, extra, level):
    assert acc.level(status, acc.parse_basis(basis), row(**extra), S) == level


def test_the_level_never_falls_as_the_cycles_rise():
    for m in ('full', 'thin'):
        for v in ('clean', 'partial', 'not_measured', 'ambiguous'):
            levels = [
                acc.LEVELS.index(acc.level('resonant', acc.parse_basis(f'N={n}(episode);M={m};V={v};S=full;veto=-'), row(), S))
                for n in (0.5, 1, 1.5, 2.5, 3, 9.5, 10, 40)
            ]
            assert levels == sorted(levels)


def test_zlk_invariants():
    zlk = dict(example_id='zlk__9__LK', resonance='LK', check_zlk_free_omega='pass')
    episodes = pd.DataFrame([{'example_id': 'zlk__9__LK', 'resonance': 'LK', 'counted': 'True'}])
    assert violations(row(**zlk), episodes=episodes) == []
    unavailable = dict(zlk, flags='gate_unavailable', check_zlk_free_omega='not_measured')
    assert '7' in violations(row(**unavailable), episodes=episodes)  # certain with a gate flag
    ok = dict(unavailable, confidence='medium', confidence_basis='N=12(episode);M=full;V=not_measured;S=full;veto=-')
    assert violations(row(**ok), episodes=episodes) == []
    assert '7' in violations(row(**dict(ok, check_zlk_free_omega='pass')), episodes=episodes)
    assert '7' in violations(row(**dict(zlk, status_code='-2')), episodes=episodes)
    assert '7' in violations(row(**dict(zlk, status_code='-9')), episodes=episodes)


def test_invariant_9_excuses_only_the_rows_s1_moved():
    base = pd.DataFrame([row(), row(resonance='3J-1+0+2', geo_half_libration_read='1-2:end:0.7/0.6:1', action='extend')])
    v2 = pd.DataFrame([row(confidence='high'), row(resonance='3J-1+0+2', geo_half_libration_read='1-2:0.7/0.6:1', action='review')])
    found, excused = acc.invariant_9(v2, base)
    assert list(found['resonance']) == ['2J-1+0+1']
    assert list(excused['resonance']) == ['3J-1+0+2']


def test_clear_sets():
    librator = row(mrg_L1='full', mrg_R0='full', owner='', jump_times_yr='')
    jumped = dict(librator, jump_times_yr='12000')
    circ = row(
        example_id='mmr3__2__1J+1S-1+0+0+0',
        status='non_resonant',
        angle_status='circulation',
        confidence='certain',
        n_cycles='0',
        geo_centre_max_pass_rev='4',
        geo_n_tp='0',
        geo_n_tp_cut_half='0',
        geo_n_tp_cut_double='0',
        mrg_C1='full',
    )
    near = dict(circ, geo_near_episode_n_tp='2')
    wrong = dict(circ, example_id='mmr3__3__x', status='transient', confidence='medium')
    assert [acc.clear_kind(r) for r in (librator, jumped, circ, near)] == ['librator', '', 'circulator', '']
    table, per = acc.clear_set(pd.DataFrame([librator, circ, wrong]))
    per = per.set_index('family')
    assert per.loc['mmr2', 'pass'] and per.loc['mmr2', 'n'] == 1
    assert per.loc['mmr3', 'opposite'] == 1 and not per.loc['mmr3', 'pass']


def test_hard_swaps_and_held_out():
    labels = pd.DataFrame(
        [
            {'example_id': 'a', 'author_status': 'resonant', 'author_confidence': 'high'},
            {'example_id': 'b', 'author_status': 'non_resonant', 'author_confidence': 'certain'},
            {'example_id': 'c', 'author_status': 'resonant', 'author_confidence': 'medium'},
        ]
    )
    minimal = pd.DataFrame([row(example_id='a', status='non_resonant'), row(example_id='b'), row(example_id='c', status='non_resonant')])
    v2 = pd.DataFrame([row(example_id='a'), row(example_id='b'), row(example_id='c')])
    frame = acc.label_frame(minimal, v2, labels)
    swaps = acc.hard_swaps(frame, {'a': 'measured owner'})
    assert list(swaps['example_id']) == ['a', 'b']  # c is medium: not a hard-swap label
    assert list(swaps['v2_swapped']) == [False, True]
    assert list(swaps['reason']) == ['measured owner', '']
    held = acc.held_out(frame)
    assert (held['minimal_hard_swaps'], held['v2_hard_swaps'], held['expectation_holds']) == (2, 1, False)
    assert held['minimal_agreement'] == pytest.approx(0.0) and held['v2_agreement'] == pytest.approx(2 / 3)


def test_the_prelabel_whitelist(tmp_path):
    def make(name, created, status):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'run.json').write_text(json.dumps({'created': created}))
        pd.DataFrame([row(example_id='mmr2__1__x', status=status)]).to_csv(tmp_path / name / 'summary.csv', index=False)

    make('v2-w24-final', '2026-10-27T10:00:00', 'transient')
    make('v2-w26-final', '2026-10-29T10:00:00', 'resonant')
    make('v2-w26-final-figs', '2026-10-29T11:00:00', 'uncertain')  # not whitelisted
    make('v2-w4-status-p0', '2026-10-29T12:00:00', 'uncertain')  # not whitelisted
    make('v2-w28', '2026-11-01T10:00:00', 'non_resonant')  # after the label
    assert [name for _, name, _ in acc.whitelisted_runs(tmp_path)] == ['v2-w24-final', 'v2-w26-final', 'v2-w28']
    labels = pd.DataFrame(
        [
            {'example_id': 'mmr2__1__x', 'author_status': 'resonant', 'label_source': 'x.csv: author:2026-10-31 (W28 g01-1 A)'},
            {'example_id': 'mmr2__1__x', 'author_status': 'resonant', 'label_source': 'x.csv: author:2026-10-28 16:04 (W25)'},
            {'example_id': 'mmr2__1__x', 'author_status': 'resonant', 'label_source': 'author_labels_2026-10-14_1612.csv'},
        ]
    )
    pre, skipped = acc.prelabel(labels, tmp_path)
    assert list(pre['prelabel_run']) == ['v2-w26-final', 'v2-w24-final'] and skipped == 1
    assert list(pre['prelabel_status']) == ['resonant', 'transient']


def test_parse_basis():
    assert acc.parse_basis('N=2.5(union);M=thin;V=partial;S=full;veto=owner:thin') == {
        'N': '2.5',
        'N_kind': 'union',
        'M': 'thin',
        'V': 'partial',
        'S': 'full',
        'veto': 'owner',
        'veto_thin': 'thin',
    }
    assert acc.parse_basis('N=10(cover);M=full;V=clean;S=full;veto=-')['veto'] == ''


# --- the W29 parts of the other tools -----------------------------------------------------------------------------------


def test_the_policy_option_of_the_reclassify_tools():
    assert reclassify.policy_overrides(None) == {}
    assert reclassify.policy_overrides('v2') == {'classifier_policy': 'v2'}
    with pytest.raises(SystemExit):
        reclassify.policy_overrides('w28')


def test_the_episode_gate_counts_only_counted_episodes_under_minimal():
    """Two accepted runs, one counted (the other vetoed): under minimal the plain episodes are the counted one; a v2
    run (no `policy`) keeps both."""
    episodes = pd.DataFrame(
        [
            {'t_start_yr': 0.0, 't_end_yr': 10.0, 'n_tp': 5, 'accepted': True, 'counted': True},
            {'t_start_yr': 20.0, 't_end_yr': 30.0, 'n_tp': 5, 'accepted': True, 'counted': False},
        ]
    )
    minimal = ee.Record({'policy': 'minimal'}, episodes)
    assert minimal.counted_only and len(minimal.plain) == 1 and len(minimal.accepted) == 2
    v2 = ee.Record({'policy': ''}, episodes)
    assert not v2.counted_only and len(v2.plain) == 2
    assert ee.run_policy(ee.RunTables(pd.DataFrame({'policy': ['minimal', 'minimal']}), pd.DataFrame())) == 'minimal'
    assert ee.run_policy(ee.RunTables(pd.DataFrame({'status': ['x']}), pd.DataFrame())) is None


def test_the_integration_readings_of_the_invariants():
    """W29 integration (RESULTS.md): the spec's own rules that the first reading of the invariants contradicted."""
    zlk = dict(example_id='zlk__9__LK', resonance='LK')
    episodes = pd.DataFrame([{'example_id': 'zlk__9__LK', 'resonance': 'LK', 'counted': 'True'}])
    # §8: a kinematic outcome that pins no episode -> transient/low with the measured objection, the check failed
    objection = dict(
        zlk,
        status='transient',
        angle_status='episodic',
        confidence='low',
        flags='measured_objection',
        check_zlk_free_omega='fail',
        status_code='1',
        confidence_basis='N=12(episode);M=full;V=ambiguous;S=full;veto=-',
    )
    assert violations(row(**objection), episodes=episodes) == []
    assert '7' in violations(row(**dict(objection, flags='')), episodes=episodes)
    # the gate's <= medium is for a bounded angle: a circulation at high with a gate flag is consistent
    circulation = dict(
        zlk,
        status='non_resonant',
        angle_status='insufficient_time',
        reason='circulation',
        confidence='high',
        flags='gate_unavailable',
        check_zlk_free_omega='not_measured',
        status_code='0',
        n_counted_episodes='0',
        counted_episode_ids='',
        n_cycles='nan',
        pol_end_travel_rev='1.5',
        confidence_basis='N=nan(-);M=not_measured;V=not_measured;S=thin;veto=-',
    )
    assert '7' not in violations(row(**circulation), episodes=pd.DataFrame(columns=['example_id', 'resonance', 'counted']))
    # §5: secular phase_unresolved is reviewed, never extended
    unresolved = dict(
        example_id='secular__9__g-g6',
        resonance='g-g6',
        status='uncertain',
        angle_status='insufficient_time',
        angle_subtype='phase_unresolved',
        confidence='',
        action='review',
        pol_last_reaches_end='False',
        confidence_basis='',
        n_counted_episodes='0',
        counted_episode_ids='',
    )
    assert '4' not in violations(row(**unresolved))
    # the basis prints N to three significant digits
    assert violations(row(n_cycles='12.04', confidence_basis='N=12(episode);M=full;V=clean;S=full;veto=-')) == []
    # §2.1: a subresonance parent counts its member's episodes, not its own
    parent = row(reason='subresonance_librates', n_counted_episodes='0', counted_episode_ids='', geo_centre_max_pass_rev='1.4')
    member = row(resonance='2J-1+1+0', n_counted_episodes='1', counted_episode_ids='0')
    members_episodes = pd.DataFrame([{'example_id': 'mmr2__1__2J-1+0+1', 'resonance': '2J-1+1+0', 'episode_id': '0', 'counted': 'True'}])
    assert violations(parent, member, episodes=members_episodes) == []
    assert {'2', '8'} <= set(violations(parent, episodes=members_episodes))


def test_a_claude_reading_row_is_no_author_label(tmp_path):
    """Card d16 B (2026-11-02): a decision row marked claude-reading is skipped, so the label before it stands; a later
    author row wins as before."""
    (tmp_path / 'author_label_decisions_2026-10-15.csv').write_text(
        'example_id,status,confidence,angle_status,source\n'
        'secular__1__g-g6,resonant,high,libration,author:2026-10-20 (a)\n'
        'secular__1__g-g6,uncertain,,insufficient_time,claude-reading:2026-10-31 (not an author label)\n'
        'secular__2__g-g6,uncertain,,insufficient_time,claude-reading:2026-10-31 (not an author label)\n'
        'secular__3__g-g6,uncertain,,,author:2026-10-31 (b)\n'
        'secular__3__g-g6,resonant,,libration,author:2026-11-02 (c)\n'
    )
    labels = reclassify.author_labels(tmp_path).set_index('example_id')
    assert labels.loc['secular__1__g-g6', 'author_status'] == 'resonant'
    assert 'secular__2__g-g6' not in labels.index
    assert labels.loc['secular__3__g-g6', 'author_status'] == 'resonant'
