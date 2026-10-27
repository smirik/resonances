"""W6 verdict package (AUTHOR-FEEDBACK 2026-10-16, C1-C3, C5, C6).

- C1/C6: episodes followed by a circulation (G1c) or by a regime without episodes after a jump of a (G1e) are reported
  as angle_status episodic (recurrent with >= 2 episodes); how the record ends goes to final_regime and
  final_regime_start_yr. Only the output changes: status, checks and confidence stay.
- C2: margin W of a 1:1 angle is not_applicable (no owner exists), of a search that did not run not_measured; margin S
  of a rejected scan compares the closest member's centre range with that of sigma0 in the same scan window.
- C3: an edge is covered when no turning point is missing between the analysis window's edge and the episode.
- C5: a libration found at the centre scale W_c that the filter damps (filter_too_slow) is analysed again at
  P / FILTER_MIN_LIBRATION_CUTOFFS (15875 2N-3).
The expected numbers are worked out by hand from the synthetic turning points (see each test).
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr import owner as owner_module
from resonances.resonance.analysis.mmr import subresonances as sub
from resonances.resonance.analysis.mmr.models import ResonanceSpec, SubresonanceCandidate
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, ResonanceType, Status, Verdict
from tests.resonances.analysis.test_impl_preprocess_geometry_geometry import ONE_DEG, filtered, tp_from
from tests.resonances.analysis.test_impl_w2_zone_rules_decide import SAMPLING
from tests.resonances.analysis.test_impl_w4_status_episodes import jump_record

S = settings_from_config()


def _ledger():
    return Ledger.from_settings(S)


# --- C3: coverage at the window's edge ------------------------------------------------------------------------------


def _episode(times, window, record=(0.0, 1400.0)):
    values = [1.0 if i % 2 == 0 else -1.0 for i in range(len(times))]
    tp = tp_from(times, values)
    t = np.linspace(record[0], record[1], 1401)
    return g.classify_run(0, tp, np.arange(len(times)), t, window, S)


def test_an_edge_without_a_missing_turning_point_is_covered():
    # TPs every 200 yr (P 400 yr) from 100 to 900 yr; the record ends at 1400 yr: 500 yr = 1.25 P from its end
    covered = _episode([100, 300, 500, 700, 900], (0.0, 1000.0))
    assert covered.coverage_end_gap_p == pytest.approx(1.25) and covered.coverage_start_gap_p == pytest.approx(0.25)
    assert (covered.window_end_gap_yr, covered.tp_step_max_yr) == (pytest.approx(100.0), pytest.approx(200.0))
    assert g.covers_edges(covered, S) == (True, True)  # 100 yr from the window's end < the 200-yr step
    assert g.covering_episode([covered], S) is covered
    # the window ends 250 yr after the last TP: one TP (200 yr after it) is missing
    missing = _episode([100, 300, 500, 700, 900], (0.0, 1150.0))
    assert g.covers_edges(missing, S) == (True, False) and g.covering_episode([missing], S) is None


def test_a_long_stretch_hidden_by_the_trim_is_not_covered():
    # the same TPs, but the record runs to 2000 yr: the window's end hides 1000 yr = 2.5 P (a long cutoff's trim, G2c)
    episode = _episode([100, 300, 500, 700, 900], (0.0, 1000.0), record=(0.0, 2000.0))
    assert episode.window_end_gap_yr == pytest.approx(100.0) and episode.window_end_hidden_p == pytest.approx(2.5)
    assert g.covers_edges(episode, S) == (True, False)
    # a record to 1400 yr hides 400 yr = 1.0 P: within LIBRATION_COVERAGE_PERIODS
    assert _episode([100, 300, 500, 700, 900], (0.0, 1000.0)).window_end_hidden_p == pytest.approx(1.0)


def test_the_longest_step_of_uneven_turning_points_sets_the_limit():
    # steps 150, 250, 150, 250 yr: the longest is 250; the window's end 220 yr after the last TP misses none
    episode = _episode([100, 250, 500, 650, 900], (0.0, 1120.0))
    assert episode.tp_step_max_yr == pytest.approx(250.0) and episode.coverage_end_gap_p == pytest.approx(500 / 400)
    assert g.covers_edges(episode, S)[1]
    assert not g.covers_edges(_episode([100, 250, 500, 650, 900], (0.0, 1160.0)), S)[1]  # 260 > 250


def test_a_hole_in_the_episode_is_not_a_step_of_its_libration():
    # TPs every 200 yr (P 400 yr) at 1100-1900 yr, a 1500-yr hole, again at 3400-4200 yr: median full period 400 yr,
    # longest step 1500 yr = 3.75 P. The window starts 1000 yr = 2.5 P before the first TP: shorter than the longest
    # step, but TPs are missing there; the step limit is LIBRATION_COVERAGE_PERIODS x P = 400 yr
    times = [1100, 1300, 1500, 1700, 1900, 3400, 3600, 3800, 4000, 4200]
    episode = _episode(times, (100.0, 4300.0), record=(0.0, 4400.0))
    assert episode.period_med_yr == pytest.approx(400.0) and episode.tp_step_max_yr == pytest.approx(1500.0)
    assert episode.window_start_gap_yr == pytest.approx(1000.0)
    assert g.window_step_limit_yr(episode, S) == pytest.approx(S.libration_coverage_periods * episode.period_med_yr)
    assert not g.covers_edges(episode, S)[0]
    # a regular episode keeps its longest step as the limit (200 yr < 1 P)
    regular = _episode([100, 300, 500, 700, 900], (0.0, 1000.0))
    assert g.window_step_limit_yr(regular, S) == pytest.approx(200.0)


def test_the_record_edge_rule_still_covers_on_its_own():
    # 350 yr = 0.875 P from the record's end (the window untrimmed): covered, although 350 yr > the 200-yr step
    episode = _episode([100, 300, 500, 700, 900, 1100], (0.0, 1450.0), record=(0.0, 1450.0))
    assert episode.coverage_end_gap_p == pytest.approx(0.875) and episode.window_end_gap_yr == pytest.approx(350.0)
    assert g.covers_edges(episode, S) == (True, True)


# --- C1/C6: the reported angle and final_regime ---------------------------------------------------------------------


def test_a_circulation_after_the_episodes_is_reported_as_a_transient_that_ends_in_circulation():
    t = np.arange(0, 20000, 1.0)
    libration = np.pi + np.radians(60) * np.sin(2 * np.pi * t / 500)
    circulation = np.pi + 2 * np.pi * 5 * (t - 10000) / 10000
    geometry = g.measure_geometry(filtered(t, np.where(t < 10000, libration, circulation), 60.0), ONE_DEG, S)
    angle = decide.decide_angle(SAMPLING, False, geometry, None, S, _ledger())
    assert (angle.angle_status, angle.rule) == (AngleStatus.CIRCULATION, 'G1c')  # the rule itself is unchanged
    reported = decide.reported_angle(angle, geometry, S)
    assert (reported.angle_status, reported.subtype) == (AngleStatus.EPISODIC, AngleSubtype.NONE)  # one episode
    regime, start = decide.final_regime(angle, geometry, S)
    # from the last TP, inside the last libration period before the circulation starts at 10 kyr
    assert regime == 'circulation' and start == geometry.turning_points.times_yr[-1] and 9500 < start < 10200


def test_a_regime_without_episodes_after_a_jump_ends_the_record():
    geometry = jump_record('circulation')
    angle = decide.decide_angle(SAMPLING, False, geometry, None, S, _ledger())
    assert angle.earlier_episode_id is not None  # G1e
    assert decide.reported_angle(angle, geometry, S).angle_status == AngleStatus.EPISODIC
    regime, start = decide.final_regime(angle, geometry, S)
    assert regime == 'circulation' and start == geometry.regime_start_yr and 10000 < start < 11000


def test_two_episodes_before_the_circulation_are_recurrent():
    geometry = jump_record('circulation')
    angle = decide.decide_angle(SAMPLING, False, geometry, None, S, _ledger())
    (episode,) = geometry.episodes_before_jump
    second = replace(episode, episode_id=episode.episode_id + 100)
    geometry.runs = list(geometry.runs) + [second]
    assert len(decide.librating_episodes(geometry, S, before_jump=True)) == 2
    assert decide.reported_angle(angle, geometry, S).subtype == AngleSubtype.RECURRENT


def test_a_libration_reaching_the_end_and_a_plain_circulation():
    t = np.arange(0, 20000, 1.0)
    geometry = g.measure_geometry(filtered(t, np.pi + np.radians(60) * np.sin(2 * np.pi * t / 500), 60.0), ONE_DEG, S)
    angle = decide.decide_angle(SAMPLING, False, geometry, None, S, _ledger())
    assert angle.angle_status == AngleStatus.LIBRATION and decide.reported_angle(angle, geometry, S) is angle
    regime, start = decide.final_regime(angle, geometry, S)
    assert regime == 'libration' and start == geometry.episodes[-1].t_start_yr
    rotating = g.measure_geometry(filtered(t, 2 * np.pi * t / 1000, 60.0), ONE_DEG, S)
    circulation = decide.decide_angle(SAMPLING, False, rotating, None, S, _ledger())
    assert circulation.angle_status == AngleStatus.CIRCULATION and decide.final_regime(circulation, rotating, S)[0] == ''


def test_final_regime_round_trips_through_summary_csv():
    verdict = Verdict(ResonanceType.MMR, Status.TRANSIENT, AngleStatus.EPISODIC, AngleSubtype.RECURRENT, final_regime='circulation')
    verdict.final_regime_start_yr = 18999.4
    row = verdict.to_flat_dict()
    assert (row['final_regime'], row['final_regime_start_yr']) == ('circulation', 18999.4)
    back = Verdict.from_flat_dict(row)
    assert (back.final_regime, back.final_regime_start_yr) == (AngleStatus.CIRCULATION, 18999.4)
    plain = Verdict.from_flat_dict(Verdict(ResonanceType.SECULAR, Status.RESONANT, AngleStatus.LIBRATION).to_flat_dict())
    assert (plain.final_regime, plain.final_regime_start_yr) == ('', None)


# --- C2: margins W and S --------------------------------------------------------------------------------------------


def test_w_of_a_coorbital_is_not_applicable_and_of_a_retrograde_not_measured():
    coorbital = owner_module._empty(CheckValue.NOT_APPLICABLE, [owner_module.FLAG_COORBITAL])
    retrograde = owner_module._empty(CheckValue.NOT_APPLICABLE, [owner_module.FLAG_RETROGRADE])
    assert decide._w_margin(coorbital, _ledger(), S).margin == Margin.NOT_APPLICABLE
    assert decide._w_margin(retrograde, _ledger(), S).margin == Margin.NOT_MEASURED


def _member(range_deg):
    return SubresonanceCandidate(
        ResonanceSpec.from_key('3J-2+1-2'),
        1,
        False,
        np.radians(range_deg),
        0.0,
        0.0,
        0.0,
        np.nan,
        1.0,
        np.nan,
        np.nan,
        np.nan,
        1,
        False,
        '',
    )


def test_s_of_a_rejected_scan_compares_with_sigma0_in_the_scan_window():
    scan = replace(sub._empty((), CheckValue.PASS, []), candidates=[_member(500.0), _member(120.0)], centre0_range_rad=np.radians(100.0))
    margin = decide._s_margin(scan, _ledger(), S)
    assert (margin.margin, margin.value) == (Margin.FULL, pytest.approx(1.2))  # 120 / 100 > 0.8
    close = replace(scan, candidates=[_member(70.0)])
    assert decide._s_margin(close, _ledger(), S).margin == Margin.THIN  # 70 / 100 below MMR_S_CLOSE_RATIO
    unscanned = replace(scan, check=CheckValue.NOT_MEASURED)
    assert decide._s_margin(unscanned, _ledger(), S).margin == Margin.NOT_MEASURED


def test_s_of_an_accepted_member_without_the_theory_is_thin():
    accepted = _member(10.0)
    scan = replace(sub._empty((), CheckValue.FAIL, []), candidates=[accepted], accepted=accepted, theory_measured=False)
    assert decide._s_margin(scan, _ledger(), S).margin == Margin.THIN
    assert decide._s_margin(replace(scan, theory_measured=True), _ledger(), S).margin == Margin.FULL


# --- benchmark records (skipped without the data) -------------------------------------------------------------------


def _analysed(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), S)


def test_524522_ends_in_circulation_with_its_status_unchanged():
    verdict = _analysed('coorbital__524522__1V-1+0+0__20kyr').verdict
    assert (verdict.status, verdict.angle_status, verdict.angle_subtype) == (
        Status.TRANSIENT,
        AngleStatus.EPISODIC,
        AngleSubtype.RECURRENT,
    )
    assert verdict.final_regime == 'circulation' and verdict.final_regime_start_yr == pytest.approx(18999.4, abs=1)
    assert verdict.margins['W'].margin == Margin.NOT_APPLICABLE  # a co-orbital has no owner


def test_2003_sl215_covers_the_record_without_a_missing_turning_point():
    analysis = _analysed('mmr2__2003_SL215__8J-3+0-5')
    assert (analysis.verdict.status, analysis.verdict.angle_status) == (Status.RESONANT, AngleStatus.LIBRATION)
    # at P_c = 75 yr (the floor bound; 100 yr = 20 x 5 yr before) the first TP is 15 yr after the window's start, less
    # than one 280-yr period: the episode covers the start without the missing-turning-point rule
    episode = analysis.geometry.episodes[0]
    assert episode.t_start_yr - analysis.geometry.window[0] < analysis.geometry.period_cycle_yr


def test_154634_fine_turning_points_seen_only_at_centre_turns_do_not_cover_the_record():
    # at the fine cutoff (12 yr) the 308-yr TPs appear only near the turns of the 29-kyr centre libration (a 25.6-kyr
    # step, the first TP 71 P after the window's start): not a covering libration; G2c decides at the centre scale
    analysis = _analysed('mmr3__154634__1M+1J-2+0+0+0')
    verdict = analysis.verdict
    assert (verdict.status, verdict.angle_status) == (Status.RESONANT, AngleStatus.LIBRATION)
    assert analysis.numbers_row()['geo_centre_scale_cutoff_yr'] > 10 * analysis.numbers_row()['geo_fine_cutoff_yr']


def test_15875_is_analysed_again_at_the_period_of_its_libration():
    analysis = _analysed('mmr2__15875__2N-3+0+1')
    numbers = analysis.numbers_row()
    assert (analysis.verdict.status, analysis.verdict.angle_status) == (Status.RESONANT, AngleStatus.LIBRATION)
    assert numbers['geo_centre_scale_basis'] == 'period_cycle' and not analysis.prep.filter_too_slow
    # W_c = 3 x 4.09 kyr; P = 21.6 kyr; the retry at P / 2
    assert numbers['geo_centre_scale_first_cutoff_yr'] == pytest.approx(12276, rel=0.01)
    assert numbers['geo_centre_scale_cutoff_yr'] == pytest.approx(
        analysis.geometry.period_cycle_yr / S.filter_min_libration_cutoffs, rel=0.01
    )
