"""W28 B9 (cards g08-5 B, g08-6 A, g06-6 A, g06-7 A): an uncertain row whose deciding not-measured check is
length-limited asks for extend with a computed length (`decide.length_limited_extension`).

The expected lengths are computed by hand from the construction (numbers of the benchmark rows 2003 TX9 3E+2J-10,
2003 ND 6J-6S-1 and 79653 3M+3J-7 as the package measured them on 2026-11-01), not by the package:

- A-FW = round_up(t_start + MMR_EXTEND_FORCE_WINDOWS x MMR_FORCE_WINDOW_CYCLES x P) of the last episode reaching the
  end: TX9 2210 + 2 x 3.5 x 3136 = 24162 -> 30 000 yr; ND 4316 + 2 x 3.5 x 7029 = 53519 -> 60 000 yr; 79653
  61279 + 2 x 3.5 x 8622 = 121633 -> 200 000 yr (one significant digit, rounded up).
- 5n: N = last TP + (2 m + 1 - n_TP) x P / 2 + FILTER_EDGE_TRIM_CUTOFFS x P_c, then N8 round_up(max(2T, min(N, 4T))):
  ND 14549 + 1 x 3514.5 + 3528 = 21591.5 -> max(40 000, 21591.5) = 40 000 yr.
"""

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import analyse, decide
from resonances.resonance.analysis.mmr.decide import AngleDecision, StatusDecision
from resonances.resonance.analysis.mmr import models
from resonances.resonance.analysis.mmr.models import ForceWindow
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import Action, AngleStatus, AngleSubtype, CheckValue, LibrationOrigin, Reason, Status
from tests.resonances.analysis.builders import make, make_episode, make_geometry, make_response

S = settings_from_config()
T = 20000.0  # the record length of TX9, ND and EH16 [yr]
LIBRATION = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', 0)


def test_the_hand_values_use_the_default_settings():
    assert (S.mmr_extend_force_windows, S.mmr_force_window_cycles, S.mmr_force_min_windows) == (2.0, 3.5, 2)
    assert (S.libration_centre_min_estimates, S.filter_edge_trim_cutoffs) == (2, 1.0)
    assert (S.mmr_extend_min_span_factor, S.mmr_next_check_span_factor, S.libration_coverage_periods) == (2.0, 4.0, 1.0)


def _window(t0, t1):
    return make(ForceWindow, t_start_yr=t0, t_end_yr=t1)


def _response(n_windows, a_response=CheckValue.NOT_MEASURED, contaminated=False):
    windows = [_window(1000.0 * i, 1000.0 * (i + 1)) for i in range(n_windows)]
    return make_response(a_response=a_response, windows=windows, force_contaminated=contaminated)


def _episode(tp_times, period, end_gap_p, episode_id=0, **fields):
    tp = np.asarray(tp_times, float)
    fields.setdefault('window_end_gap_yr', T - tp[-1])
    fields.setdefault('tp_step_max_yr', float(np.max(np.diff(tp))))
    return make_episode(
        episode_id=episode_id,
        tp_times_yr=tp,
        period_med_yr=period,
        coverage_start_gap_p=0.5,
        coverage_end_gap_p=end_gap_p,
        centre_estimates=max(0, (len(tp) - 1) // 2),
        **fields,
    )


def _decision(rule, status=Status.UNCERTAIN):
    reason = Reason.ORIGIN_UNKNOWN if status == Status.UNCERTAIN else Reason.DYNAMIC_LIBRATION
    return StatusDecision(status, reason, LibrationOrigin.UNKNOWN, '', '', rule=rule)


# TX9 3E+2J-10: one episode 2210-17489 yr (11 TPs, P 3136 yr, end gap 0.80 P), L4 with one window of 10975 yr
TX9 = _episode(np.linspace(2210.0, 17489.0, 11), 3136.0, 0.80)
# ND 6J-6S-1: TPs 4316-14549 yr (4 TPs: one independent cycle), P 7029 yr, end gap 0.78 P, P_c 3528 yr
ND = _episode([4316.0, 7800.0, 11200.0, 14549.0], 7029.0, 0.78)
ND_CUTOFF = 3528.0


def _extension(rule, episodes, response, angle=LIBRATION, status=Status.UNCERTAIN, span=T, cutoff=1107.8):
    geometry = make_geometry(runs=list(episodes), window=(0.0, span))
    return decide.length_limited_extension(_decision(rule, status), angle, geometry, response, span, cutoff, S)


@pytest.mark.parametrize('rule', ['5e', '5g', '5x'])
def test_too_few_force_windows_with_the_last_episode_at_the_end_asks_for_two_windows(rule):
    # TX9 (5e): t_start + 2 x 3.5 x P = 2210 + 21952 = 24162 yr -> 30 000 yr (the author's ~30 kyr, card g08-5 B)
    assert _extension(rule, [TX9], _response(1)) == 30000.0
    # 79653 (5x): the late episode 61279-98970 yr of a 100-kyr record, P 8622 yr: 121633 -> 200 000 yr
    late = _episode(np.linspace(61279.0, 98970.0, 9), 8622.0, 0.12, episode_id=1, window_end_gap_yr=1030.0)
    early = _episode(np.linspace(4813.0, 32501.0, 5), 10414.0, 6.48, window_end_gap_yr=67499.0)
    angle = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 1)
    assert _extension(rule, [early, late], _response(1), angle=angle, span=100000.0) == 200000.0


def test_an_episode_that_ended_before_the_record_end_gains_nothing_from_a_longer_record():
    # the same TX9 episode ending at 11.4 kyr: end gap 2.7 P, the window gap 8.6 kyr longer than its TP steps
    ended = _episode(np.linspace(2210.0, 11400.0, 7), 3136.0, 2.75)
    assert not decide.covers_edges(ended, S)[1]
    assert _extension('5e', [ended], _response(1)) is None


def test_a_resolved_status_or_an_objection_leaf_gets_no_extension():
    assert _extension('5d/6', [TX9], _response(1), status=Status.RESONANT) is None
    for rule in ('5a', '5a*', '5b', '5f', '5o', '5s*', '5u/6'):
        assert _extension(rule, [TX9], _response(1)) is None, rule


def test_a_swing_below_45_deg_alone_keeps_review():
    # L4 not measured with two windows: the half-amplitude, not the length, is what is missing
    assert _extension('5e', [TX9], _response(2)) is None
    # a forced line at the period (5f's reason) is no length limit either; nor a measured L4
    assert _extension('5e', [TX9], _response(1, contaminated=True)) is None
    assert _extension('5e', [TX9], _response(1, a_response=CheckValue.FAIL)) is None
    assert not decide.force_windows_short(None, S)


def test_one_independent_cycle_asks_for_the_n8_next_check():
    # ND, L4 not measured for its swing only: N = 14549 + 3514.5 + 3528 = 21591.5 yr -> round_up(max(2T, N)) = 40 kyr
    assert decide.cycles_extension(ND, ND_CUTOFF, S) == pytest.approx(21591.5)
    assert _extension('5n', [ND], _response(2), cutoff=ND_CUTOFF) == 40000.0
    # with L4 short of windows too (ND has none: 3.5 P = 24.6 kyr > T) the force windows ask for more: 60 kyr
    assert _extension('5n', [ND], _response(0), cutoff=ND_CUTOFF) == 60000.0
    assert _extension('5n', [ND], _response(0), cutoff=ND_CUTOFF) >= 28000.0  # card g06-7: >= 28 kyr


def test_one_cycle_in_an_episode_that_is_not_the_last_one_keeps_review():
    # 2018 TX1 3M-5S-2: the decisive one-cycle episode 850-2621 yr is over; a later one reaches the end
    over = _episode([850.0, 1500.0, 2100.0, 2621.0], 1200.0, 14.0, window_end_gap_yr=17379.0)
    late = _episode([16839.0, 17500.0, 18200.0, 18965.0], 1400.0, 0.7, episode_id=1)
    assert _extension('5n', [over, late], _response(2)) is None


def test_a_length_that_the_record_already_has_is_no_limit():
    # a 60-kyr record already holds the two TX9 windows (24162 -> 30 000 yr <= 60 000 yr): review stays
    long = _episode(np.linspace(2210.0, 59000.0, 37), 3136.0, 0.3, window_end_gap_yr=1000.0)
    assert _extension('5e', [long], _response(1), span=60000.0) is None


def test_the_action_turns_to_extend_only_from_review_and_fires_its_rule():
    geometry = make_geometry(runs=[TX9], window=(0.0, T))
    view = decide.TreeView(LIBRATION, _response(1), None)
    sampling = make(models.Sampling, span_yr=T)
    prep = make(models.Preprocessed, cutoff=make(models.Cutoff, cutoff_yr=1107.8))
    decision = _decision('5e')
    action, extend = analyse._length_limited_action(Action.REVIEW, None, decision, view, geometry, view.response, sampling, prep, S)
    assert (action, extend) == (Action.EXTEND, 30000.0)
    assert 'extend_length_limited' in decision.rules_fired
    # another action (a resample the short-period cap made actionable) is kept, and nothing fires
    decision = _decision('5e')
    kept = analyse._length_limited_action(Action.RESAMPLE, None, decision, view, geometry, view.response, sampling, prep, S)
    assert kept == (Action.RESAMPLE, None) and not decision.rules_fired
