"""W29 minimal policy: the extend predicate on synthetic sigma_f arrays (spec §5, PRINCIPLES §5). Expected by hand:
an unfinished libration at the record's end (fewer than CONFIDENCE_HIGH_CYCLES = 3 cycles in the last episode) extends;
five cycles do not; a record whose sigma_f turns a whole revolution after its last accepted or candidate turning point is
decided now, whatever came before; a period longer than 2 T / 3 extends; a needed veto not measured for lack of length
(L4 short of force windows, a ZLK gate short of cycles) extends when the record it needs is within the next-check budget
MMR_NEXT_CHECK_SPAN_FACTOR x T = 4 T (clause C, the author W29 Q5), goes to review beyond it."""

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis import policy
from resonances.resonance.analysis.mmr import models
from resonances.resonance.analysis.mmr.models import ForceWindow
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Action, AngleStatus, CheckValue, Flag, Reason, Status
from tests.resonances.analysis.builders import (
    make,
    make_episode,
    make_geometry,
    make_mmr_analysis,
    make_response,
    make_turning_points,
    make_verdict,
)

MINIMAL = AnalysisSettings()
SPAN, PERIOD = 4000.0, 400.0
T = np.linspace(0.0, SPAN, 4001)


def prep(angle_f):
    return make(models.Preprocessed, t_yr=T, angle_f=np.asarray(angle_f, dtype=float), window=(0.0, SPAN))


def libration_at_end(n_tp, **fields):
    """An accepted episode of `n_tp` turning points ending 0.1 period before the window's end."""
    times = SPAN - 0.1 * PERIOD - np.arange(n_tp)[::-1] * PERIOD / 2
    run = make_episode(
        n_tp, episode_id=0, tp_times_yr=times, period_med_yr=PERIOD, half_amp_max_rad=0.5, coverage_start_gap_p=5.0, coverage_end_gap_p=0.1
    )
    geometry = make_geometry(window=(0.0, SPAN), runs=[run], turning_points=make_turning_points(times), period_cycle_yr=PERIOD)
    sigma = np.where(T >= times[0], 0.5 * np.sin(2 * np.pi * (T - times[0]) / PERIOD), 2 * np.pi * (T - times[0]) / 300.0)
    verdict = make_verdict(Status.TRANSIENT, AngleStatus.EPISODIC, reason=Reason.DYNAMIC_EPISODE)
    return make_mmr_analysis(verdict, geometry, '5d/6', span_yr=SPAN, prep=prep(sigma), decisive_episode_id=0, **fields)


def test_a_libration_reaching_the_end_with_few_cycles_extends():
    # 4 TPs: 1.5 cycles + the covered end gap 0.1 = 1.6 < CONFIDENCE_HIGH_CYCLES 2
    v = policy.apply_mmr(libration_at_end(4), MINIMAL).verdict
    assert v.pol_n_last == pytest.approx(1.6) and v.pol_last_reaches_end is True
    assert (v.action, v.pol_extend_clause) == (Action.EXTEND, 'A')
    # next check: span + (2 - 1.6) -> max(.., 1) x P, at least MMR_EXTEND_MIN_SPAN_FACTOR x T = 8000, rounded
    assert v.extend_to_years is not None and np.isfinite(v.extend_to_years) and v.extend_to_years >= 2 * SPAN


def test_five_cycles_at_the_end_do_not_extend():
    v = policy.apply_mmr(libration_at_end(11), MINIMAL).verdict
    assert v.pol_n_last == pytest.approx(5.1)
    assert v.action != Action.EXTEND and v.pol_extend_clause == ''


def test_a_half_swing_followed_by_turns_is_decided_now():
    # a half-swing at 400-600 yr, then sigma_f turns 8 revolutions to the end: no end-edge half-swing, the end circulates
    half = g.HalfLibration(400.0, 600.0, '', 1.0, 2.0, r2_h1=0.7, r2_h2=0.7, restoring=True)
    sigma = np.where(T < 600.0, 0.0, 2 * np.pi * 8 * (T - 600.0) / (SPAN - 600.0))
    geometry = make_geometry(window=(0.0, SPAN), half_librations=(half,), turning_points=make_turning_points([400.0, 600.0]))
    verdict = make_verdict(Status.UNCERTAIN, AngleStatus.EPISODIC, reason=Reason.ONE_HALF_LIBRATION)
    v = policy.apply_mmr(make_mmr_analysis(verdict, geometry, 'G4h', span_yr=SPAN, prep=prep(sigma)), MINIMAL).verdict
    assert v.pol_end_travel_rev == pytest.approx(8.0, rel=1e-3)
    assert (v.status, v.action, v.pol_extend_clause) == (Status.NON_RESONANT, Action.SPOT_CHECK, '')


def test_an_arc_without_turning_points_extends():
    # sigma_f sweeps 0.6 rev in the record, no turning point: insufficient_time, A3
    sigma = 2 * np.pi * 0.6 * T / SPAN
    geometry = make_geometry(window=(0.0, SPAN))
    verdict = make_verdict(Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, reason=Reason.BASELINE_NEEDED)
    v = policy.apply_mmr(make_mmr_analysis(verdict, geometry, 'G6', span_yr=SPAN, prep=prep(sigma)), MINIMAL).verdict
    assert (v.status, v.action, v.pol_extend_clause) == (Status.UNCERTAIN, Action.EXTEND, 'A')
    assert v.extend_to_years is not None and np.isfinite(v.extend_to_years)


def test_the_period_clause():
    end = policy.EndReading(0.0, 0.1, 2700.0, 0.0, False)
    # 2 T / 3 = 2666.7 yr for T = 4000 yr
    assert policy.extend_clause(Status.RESONANT, end, SPAN, MINIMAL) == 'B'
    assert policy.extend_clause(Status.RESONANT, policy.EndReading(0.0, 0.1, 2600.0, 0.0, False), SPAN, MINIMAL) == ''
    # the record ends in circulation: nothing extends
    assert policy.extend_clause(Status.RESONANT, policy.EndReading(0.0, 1.2, 2700.0, 0.0, False), SPAN, MINIMAL, leg_end=True) == ''
    assert policy.extend_clause(Status.INVALID, end, SPAN, MINIMAL) == ''
    stricter = AnalysisSettings(libration_min_turning_points=4, mmr_l1_tp_full=5)
    assert policy.extend_clause(Status.RESONANT, policy.EndReading(0.0, 0.1, 2100.0, 0.0, False), SPAN, stricter) == 'B'  # 2T/4 = 2000


def test_a_start_edge_leg_with_circulation_at_the_end_does_not_extend():
    leg = g.SingleLeg(
        0.0,
        300.0,
        g.LEG_EDGE_START,
        g.LEG_HORSESHOE,
        2.0,
        None,
        g.LegTurn(300.0, 'planet', 1.0, 0.5),
        force='pass',
        force_scale=1.0,
        energy='pass',
    )
    sigma = np.where(T < 300.0, 0.0, 2 * np.pi * 5 * (T - 300.0) / (SPAN - 300.0))
    geometry = make_geometry(window=(0.0, SPAN), coorbital=True, single_legs=(leg,), turning_points=make_turning_points([300.0]))
    verdict = make_verdict(Status.UNCERTAIN, AngleStatus.INSUFFICIENT_TIME, reason=Reason.BASELINE_NEEDED)
    v = policy.apply_mmr(make_mmr_analysis(verdict, geometry, 'G4c', span_yr=SPAN, prep=prep(sigma)), MINIMAL).verdict
    assert (v.status, v.action, v.pol_extend_clause) == (Status.NON_RESONANT, Action.SPOT_CHECK, '')


def test_rejected_noise_near_the_end_does_not_bound_it():
    # an accepted episode at 400-1400 yr, rejected 2-TP noise runs near the end; sigma_f circulates after 1400 yr
    times = 400.0 + np.arange(6) * 200.0
    accepted = make_episode(
        6, episode_id=0, tp_times_yr=times, period_med_yr=PERIOD, half_amp_max_rad=0.5, coverage_start_gap_p=1.5, coverage_end_gap_p=9.0
    )
    noise = make_episode(2, episode_id=1, tp_times_yr=np.array([3800.0, 3850.0]), accepted=False)
    sigma = np.where(T < times[-1], 0.5 * np.sin(2 * np.pi * T / PERIOD), 2 * np.pi * 6 * (T - times[-1]) / (SPAN - times[-1]))
    geometry = make_geometry(window=(0.0, SPAN), runs=[accepted, noise], turning_points=make_turning_points(list(times) + [3800.0, 3850.0]))
    verdict = make_verdict(Status.TRANSIENT, AngleStatus.EPISODIC, reason=Reason.DYNAMIC_EPISODE)
    v = policy.apply_mmr(
        make_mmr_analysis(verdict, geometry, '5d/6', span_yr=SPAN, prep=prep(sigma), decisive_episode_id=0), MINIMAL
    ).verdict
    assert v.pol_t_last_tp_yr == pytest.approx(times[-1])
    assert v.pol_end_travel_rev >= 1 and v.action != Action.EXTEND


def short_of_force_windows():
    """L4 not measured with one force window (< MMR_FORCE_MIN_WINDOWS = 2): the veto needs length (`force_windows_short`)."""
    return make_response(a_response=CheckValue.NOT_MEASURED, windows=[make(ForceWindow, t_start_yr=2000.0, t_end_yr=3400.0)])


def test_a_veto_short_of_length_within_the_budget_extends():
    # 11 TPs from 1960 yr: N_last = 5 + 0.1 >= 3, P = 400 yr < 2T/3: neither A nor B. The force windows need
    # t_start + MMR_EXTEND_FORCE_WINDOWS x MMR_FORCE_WINDOW_CYCLES x P = 1960 + 2 x 3.5 x 400 = 4760 -> 5000 yr <= 4 T
    # = 16000 yr: clause C; next check round_up(max(2 T, min(5000, 4 T))) = 8000 yr
    v = policy.apply_mmr(libration_at_end(11, response=short_of_force_windows()), MINIMAL).verdict
    assert str(Flag.VETO_NEEDS_LENGTH) in v.flags and v.pol_veto_need_yr == pytest.approx(5000.0)
    assert (v.status, v.action, v.pol_extend_clause, v.extend_to_years) == (Status.TRANSIENT, Action.EXTEND, 'C', 8000.0)
    assert any(entry.name.startswith('policy:C veto need') and entry.result for entry in v.decision_path)


def test_a_veto_short_of_length_beyond_the_budget_goes_to_review():
    # six force windows: 1960 + 6 x 3.5 x 400 = 10360 -> 20000 yr > 4 T = 16000 yr: no extend, review (1981 Midas: a
    # gate that needs ~70 Myr on a 1-Myr record); with the budget 6 T = 24000 yr the same need extends to 20000 yr
    wide = AnalysisSettings(mmr_extend_force_windows=6.0)
    v = policy.apply_mmr(libration_at_end(11, response=short_of_force_windows()), wide).verdict
    assert v.pol_veto_need_yr == pytest.approx(20000.0)
    assert (v.action, v.pol_extend_clause, v.extend_to_years) == (Action.REVIEW, '', None)
    wider = AnalysisSettings(mmr_extend_force_windows=6.0, mmr_next_check_span_factor=6.0)
    v = policy.apply_mmr(libration_at_end(11, response=short_of_force_windows()), wider).verdict
    assert (v.action, v.pol_extend_clause, v.extend_to_years) == (Action.EXTEND, 'C', 20000.0)


def test_the_veto_clause_needs_a_bounded_end_and_a_named_length():
    end = policy.EndReading(0.0, 0.1, 400.0, 5.0, True)
    assert policy.extend_clause(Status.RESONANT, end, SPAN, MINIMAL, veto_need_yr=16000.0) == 'C'  # = 4 T: inside
    assert policy.extend_clause(Status.RESONANT, end, SPAN, MINIMAL, veto_need_yr=16001.0) == ''
    assert policy.extend_clause(Status.RESONANT, end, SPAN, MINIMAL, veto_need_yr=float('nan')) == ''
    assert policy.extend_clause(Status.RESONANT, end, SPAN, MINIMAL, veto_need_yr=float('inf')) == ''
    circulating = policy.EndReading(0.0, 1.5, 400.0, 5.0, False)
    assert policy.extend_clause(Status.RESONANT, circulating, SPAN, MINIMAL, veto_need_yr=5000.0) == ''
    # review only for a named length beyond the budget at a bounded end: inf (d_min = 0) yes, nan (none named) no
    assert policy.veto_beyond_budget(float('inf'), SPAN, MINIMAL, end)
    assert policy.veto_beyond_budget(16001.0, SPAN, MINIMAL, end)
    assert not policy.veto_beyond_budget(16000.0, SPAN, MINIMAL, end)
    assert not policy.veto_beyond_budget(float('nan'), SPAN, MINIMAL, end)
    assert not policy.veto_beyond_budget(1.0e9, SPAN, MINIMAL, circulating)


@pytest.mark.parametrize(
    'source, estimate, expected',
    [
        # by hand: span 4000 + max(2 - 1.6, 1) x 400 = 4400 < MMR_EXTEND_MIN_SPAN_FACTOR x T = 8000 -> 8000
        (None, None, 8000.0),
        # a slow swing seen in part asks for its own slow cycles (the author, W30 epext B): 12 000 -> round_up 20 000
        ('arc_slow', 12000.0, 20000.0),
        ('two_omega', 12000.0, 20000.0),
        # capped at MMR_NEXT_CHECK_SPAN_FACTOR x T = 16 000, rounded up to 20 000
        ('arc_slow', 30000.0, 20000.0),
        # a baseline from the libration's own turning points is no slow swing: the deciding period decides
        ('tp_spacing', 12000.0, 8000.0),
    ],
)
def test_a_slow_swing_baseline_sets_the_next_check(source, estimate, expected):
    baseline = None
    if source is not None:
        baseline = make(
            models.Baseline,
            required_yr=estimate,
            extend_to_years=estimate,
            check=CheckValue.FAIL,
            period_yr=estimate / 3,
            period_source=source,
        )
    v = policy.apply_mmr(libration_at_end(4, baseline=baseline), MINIMAL).verdict
    assert (v.action, v.extend_to_years) == (Action.EXTEND, expected)


@pytest.mark.parametrize(
    'confidence, clause, expected',
    [
        ('certain', 'C', Action.NONE),  # the author 2026-11-05: certain asks for nothing, even with clause C
        ('certain', '', Action.NONE),
        ('high', 'C', Action.EXTEND),
        ('high', '', Action.NONE),
        ('medium', '', Action.SPOT_CHECK),
    ],
)
def test_certain_takes_no_action(confidence, clause, expected):
    from resonances.resonance.analysis.verdict import Confidence

    level = Confidence(confidence)
    kept = policy._certain_clause(level, clause)
    assert kept == ('' if confidence == 'certain' else clause)
    assert policy._action(Status.RESONANT, level, kept, None, Action.EXTEND) == expected
