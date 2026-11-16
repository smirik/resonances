"""W28 B2 (card g11-1, the author's answer A of 2026-10-31): the episodes of a row that rule 5s decides are forced
waves with the route 'strength'.

Rule 5s (non_resonant/implausible_strength) says that the period of the angle's loops needs a force at least
MMR_STRENGTH_IMPLAUSIBLE_RATIO (10^3) times the one the disturbing function gives: the loops are no libration of this
angle. Their accepted runs are then stored as forced waves of route 'strength' (`decide.strength_forced_waves`,
`Geometry.forced_waves`, episodes.csv `forced_wave`), after the verdict: nothing is measured again, so the status, the
word, the reason and the level are the same with MMR_FORCED_WAVE_STRENGTH on or off. The uncertain 5s* (a three-body
ratio the small-divisor envelope may close) marks nothing, and neither does another reason.

The expected numbers are written by hand: the 5s threshold is the setting's 10^3 (a ratio of 2000 passes it, 500 does
not); the three-body route needs the envelope's residual >= 10^3 too (50 leaves 5s*). The benchmark case 2012 KL9
5E-2M-10 has two accepted runs, 8.60-9.24 and 27.86-28.08 kyr, read off its episodes.csv on 2026-10-17
(EPISODE_EXPECTATIONS E15, before this rule existed).
"""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.analyse import _mark_strength_waves, analyse_mmr
from resonances.resonance.analysis.mmr.decide import StatusDecision
from resonances.resonance.analysis.mmr.models import ForcedWave, Strength
from resonances.resonance.analysis.mmr.strength import METHOD_DIRECT
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, LibrationOrigin, Reason, Status
from tests.resonances.analysis.builders import make, make_episode, make_geometry, tree_verdict

S = settings_from_config()
NAN = float('nan')


def _strength(ratio, three_body=False, residual=NAN):
    method = 'eq_scale' if three_body else METHOD_DIRECT
    return make(Strength, ratio=ratio, method=method, three_body=three_body, eq_legal=True, divisor_residual=residual)


def _rule_5s(strength) -> StatusDecision:
    """The decision of the 5s leaf for `strength` (`decide._implausible_strength`), or a geometric transient below it."""
    decision = decide._implausible_strength(strength, S, Ledger.from_settings(S))
    return decision or StatusDecision(Status.TRANSIENT, Reason.GEOMETRIC_EPISODE, LibrationOrigin.UNKNOWN, '', '', rule='5u/6')


def _wave(episode_id, **routes):
    fields = dict(wrong_sign=False, on_line=False, by_strength=False) | routes
    return ForcedWave(episode_id, 6, 168.0, -0.35, -0.85, '3N-5', 0.1, **fields)


def _geometry():
    """Four runs: #0 accepted, 7 TPs 100 yr apart (period 200 yr); #1 rejected; #2 accepted before the last jump of a,
    5 TPs 50 yr apart (period 100 yr); #3 accepted but already an F6 forced wave (wrong sign)."""
    runs = [
        make_episode(episode_id=0, tp_times_yr=np.arange(0.0, 700.0, 100.0), period_med_yr=200.0),
        make_episode(episode_id=1, tp_times_yr=np.arange(800.0, 950.0, 50.0), accepted=False),
        make_episode(episode_id=2, tp_times_yr=np.arange(1000.0, 1250.0, 50.0), period_med_yr=100.0, before_last_jump=True),
        make_episode(episode_id=3, tp_times_yr=np.arange(1400.0, 1700.0, 50.0), period_med_yr=100.0),
    ]
    return make_geometry(window=(0.0, 2000.0), runs=runs, forced_waves=(_wave(3, wrong_sign=True),))


def test_the_5s_leaf_marks_every_accepted_episode_left():
    decision = _rule_5s(_strength(2000.0))  # 2000 >= 10^3: rule 5s
    assert (decision.status, decision.reason, decision.rule) == (Status.NON_RESONANT, Reason.IMPLAUSIBLE_STRENGTH, '5s')
    waves = decide.strength_forced_waves(_geometry(), decision, S)
    # #0 and #2 (before the jump too); #1 is rejected and #3 already an F6 wave
    assert [(w.episode_id, w.n_tp, w.period_yr, w.route) for w in waves] == [(0, 7, 200.0, 'strength'), (2, 5, 100.0, 'strength')]
    assert all(w.by_strength and not w.wrong_sign and not w.on_line and w.force_line == '' for w in waves)
    assert all(math.isnan(w.identity_slope_ratio) and math.isnan(w.identity_corr) for w in waves)


@pytest.mark.parametrize(
    'strength, expected_rule',
    [
        (_strength(500.0), None),  # below 10^3: no 5s, another leaf decides
        (_strength(2000.0, three_body=True, residual=50.0), '5s*'),  # the envelope may close the gap: uncertain
    ],
)
def test_no_mark_without_rule_5s(strength, expected_rule):
    decision = _rule_5s(strength)
    assert decision.reason != Reason.IMPLAUSIBLE_STRENGTH
    assert (decision.rule == expected_rule) if expected_rule else decision.rule == '5u/6'
    assert decide.strength_forced_waves(_geometry(), decision, S) == ()


def test_the_three_body_5s_marks_too():
    decision = _rule_5s(_strength(2000.0, three_body=True, residual=1500.0))  # residual >= 10^3: still 5s
    assert decision.rule == '5s'
    assert [w.episode_id for w in decide.strength_forced_waves(_geometry(), decision, S)] == [0, 2]


def test_the_setting_off_marks_nothing():
    decision = _rule_5s(_strength(2000.0))
    assert decide.strength_forced_waves(_geometry(), decision, replace(S, mmr_forced_wave_strength=False)) == ()
    assert decide.strength_forced_waves(None, decision, S) == ()


def test_the_routes():
    assert _wave(0, by_strength=True).route == 'strength'
    assert _wave(0, wrong_sign=True, on_line=True).route == 'wrong_sign;lattice_line'  # F6 as before
    assert _wave(0, on_line=True).route == 'lattice_line'


def test_marking_takes_the_runs_out_of_the_episodes_and_forgets_the_derived_views():
    geometry = _geometry()
    assert [e.episode_id for e in geometry.episodes] == [0]  # #2 is before the jump, #3 an F6 wave
    geometry.derived(('probe',), S, lambda: 'cached')
    waves = decide.strength_forced_waves(geometry, _rule_5s(_strength(2000.0)), S)
    _mark_strength_waves(geometry, waves)
    assert geometry.episodes == [] and geometry.derived_views == {}
    assert geometry.forced_wave_ids == frozenset({0, 2, 3})
    _mark_strength_waves(geometry, ())  # nothing to mark: unchanged
    assert geometry.forced_wave_ids == frozenset({0, 2, 3})


def test_g2_columns_and_the_word_read_the_f6_waves_only():
    """The G2 columns and the angle word describe F6 (identity, lattice line): a strength mark changes neither."""
    geometry = _geometry()
    before = decide.forced_wave_columns(geometry)
    assert (before['geo_n_forced_wave_episodes'], before['geo_forced_wave_episode_id']) == (1, 3)
    _mark_strength_waves(geometry, decide.strength_forced_waves(geometry, _rule_5s(_strength(2000.0)), S))
    after = decide.forced_wave_columns(geometry)
    assert after['geo_n_forced_wave_episodes'] == 1 and after['geo_forced_wave_episode_id'] == 3
    only_strength = make_geometry(runs=geometry.runs, forced_waves=(_wave(0, by_strength=True),))
    assert decide.forced_wave_columns(only_strength)['geo_n_forced_wave_episodes'] == 0
    reported = decide.AngleDecision(AngleStatus.CIRCULATION, AngleSubtype(''), 'G4', None)
    ledger = Ledger.from_settings(S)
    assert decide.forced_wave_word(reported, None, None, only_strength, None, S, ledger) is reported
    assert list(ledger) == []


# --- the benchmark: 2012 KL9 (E15) ------------------------------------------------------------------------------------

KL9 = 'mmr3__2012_KL9__5E-2M-10+0+0+7'


@pytest.mark.benchmark
def test_kl9_verdict_is_the_same_and_its_two_runs_are_marked():
    if not io.example_path(KL9).exists():
        pytest.skip('benchmark data not available')
    series = io.load_series(KL9)
    on = analyse_mmr(series, S)
    off = analyse_mmr(series, replace(S, mmr_forced_wave_strength=False))
    flat_on, flat_off = on.verdict.to_flat_dict(), off.verdict.to_flat_dict()
    assert str(Rule.FORCED_STRENGTH_MARK) in flat_on.pop('rules_fired')
    assert str(Rule.FORCED_STRENGTH_MARK) not in flat_off.pop('rules_fired')
    assert {k: str(v) for k, v in flat_on.items()} == {k: str(v) for k, v in flat_off.items()}
    # the W28 tree's leaf 5s; W34: the minimal policy's veto is the W31 background wave (status-tree.md#minimal-policy,
    # step 2, 2012 KL9): non_resonant / kinematic_libration, word circulation
    tree = tree_verdict(on.verdict)
    assert (tree.status, tree.reason) == (Status.NON_RESONANT, Reason.IMPLAUSIBLE_STRENGTH)
    assert (on.verdict.status, on.verdict.reason, on.verdict.veto, on.verdict.angle_status) == (
        Status.NON_RESONANT,
        Reason.KINEMATIC_LIBRATION,
        'forced_wave',
        'circulation',
    )
    rows_on, rows_off = pd.DataFrame(on.episode_rows()), pd.DataFrame(off.episode_rows())
    accepted = rows_on[rows_on['accepted']]
    spans = [t / 1e3 for span in sorted(zip(accepted['t_start_yr'], accepted['t_end_yr'])) for t in span]
    assert spans == pytest.approx([8.60, 9.24, 27.86, 28.08], abs=0.05)
    assert set(accepted['forced_wave']) == {'strength'}
    assert set(rows_off['forced_wave']) == {''}
    # the numbers the decision read stay; only the derived count of the angle's episodes drops
    numbers_on, numbers_off = on.numbers_row(), off.numbers_row()
    assert numbers_on['geo_n_forced_wave_episodes'] == numbers_off['geo_n_forced_wave_episodes'] == 0
    assert numbers_on['geo_n_librating_episodes'] == numbers_off['geo_n_librating_episodes']
    assert (numbers_on['geo_n_episodes'], numbers_off['geo_n_episodes']) == (0, 2)
