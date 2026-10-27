"""W16 checks: the force law kept off the confirmed jumps of a (L4), the onset test inside the episode's regime (O1),
the e-modulation fit of the zone, test A of the subresonance scan at small swings, and the edge caps of a transient.

The expected numbers are worked out by hand in each test (window bounds, zone limits, half-amplitudes), or come from a
physical property of the synthetic record (an exact pendulum fits the force law with R^2 ~ 1 away from the step)."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.mmr import decide, response as rsp, subresonances as sub, zone
from resonances.resonance.analysis.mmr.models import ResonanceKind, ResonanceSpec, SubresonanceCandidate
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Flag
from tests.resonances.analysis.builders import make_centre, make_episode, make_geometry, make_turning_points
from tests.resonances.analysis.impl_response_pendulum_synthetic import pendulum_record

S = AnalysisSettings()
NAN = float('nan')


def _jump(time_yr, kick=False):
    return geo.Jump(time_yr, 0.01, 10.0, NAN if kick else 1.0, False, kick)


# ---------------------------------------------------------------- L4 next to a confirmed jump of a


def test_the_guard_takes_confirmed_jumps_only():
    geometry = SimpleNamespace(jumps=[_jump(5000.0), _jump(2000.0, kick=True), _jump(1000.0)], jump_tau_yr=40.0)
    guard = rsp.JumpGuard.from_geometry(geometry)
    assert guard.times_yr == (1000.0, 5000.0)
    assert guard.half_tau_yr == 20.0
    # zone = jump -+ (tau/2 + 1 x P_cut): 5000 -+ (20 + 30) with P_cut = 30
    assert rsp.jump_zones(guard, 30.0, S) == [(950.0, 1050.0), (4950.0, 5050.0)]
    assert rsp.jump_zones(rsp.JumpGuard(), 30.0, S) == []
    assert rsp.JumpGuard.from_geometry(None).times_yr == ()


def test_windows_and_samples_overlapping_a_zone_are_left_out():
    windows = [(0.0, 1000.0), (500.0, 1500.0), (1000.0, 2000.0), (1500.0, 2500.0)]
    # zone [1900, 2100]: only the windows ending before 1900 or starting after 2100 stay
    assert rsp.outside_zones(windows, [(1900.0, 2100.0)], bounds=tuple) == [(0.0, 1000.0), (500.0, 1500.0)]
    t = np.arange(0.0, 10.0, 1.0)
    assert rsp.zones_mask(t, [(2.5, 4.0), (8.0, 8.0)]).tolist() == [1, 1, 1, 0, 0, 1, 1, 1, 0, 1]


def _with_step(rec, t_jump, size):
    a = rec.prep.a_f + size * (rec.t >= t_jump)
    rec.prep.a_f, rec.prep.adot_f = a, np.gradient(a, rec.t)
    return rec


def _l4(rec, pieces, guard):
    period = rec.geometry.period_cycle_yr
    amp = math.radians(90.0)
    judge = rsp.ResponseJudge(False, False, rec.sampling, None)
    return rsp.measure_pieces(rec.prep, rec.context, S, pieces, period, (amp, amp), judge=judge, guard=guard)


def test_force_law_windows_at_a_jump_are_dropped_and_the_rest_fit_the_pendulum():
    """A step of 10 a-wave amplitudes at 3000 yr: the smoothed step puts a peak into a_dot that the windows over it fit
    instead of the pendulum (inv-06-07 Q6: R^2 0.02). Every degraded window is in the zone; every window kept fits."""
    rec = pendulum_record()
    amplitude_a = float(np.ptp(rec.prep.a_f)) / 2
    rec = _with_step(rec, 3000.0, 10 * amplitude_a)
    pieces = [rec.geometry.window]
    plain = _l4(rec, pieces, None)
    guarded = _l4(rec, pieces, rsp.JumpGuard((3000.0,), 40.0))
    poor = [(w.t_start_yr, w.t_end_yr) for w in plain.windows if w.r2 < 0.9]
    assert poor, 'the step must spoil the windows over it'
    kept = [(w.t_start_yr, w.t_end_yr) for w in guarded.windows]
    assert not set(poor) & set(kept)
    assert min(w.r2 for w in guarded.windows) > 0.99
    assert guarded.force_jump_dropped == len(plain.windows) - len(guarded.windows) >= len(poor)
    assert guarded.a_response == CheckValue.PASS and plain.a_response in (CheckValue.FAIL, CheckValue.PASS)
    assert min(plain.force_r2_h1, plain.force_r2_h2) < min(guarded.force_r2_h1, guarded.force_r2_h2)
    assert guarded.to_columns()['force_n_windows_jump'] == guarded.force_jump_dropped


def test_fewer_than_two_clean_windows_leave_the_force_law_not_measured():
    rec = pendulum_record()
    period = rec.geometry.period_cycle_yr
    length = S.mmr_force_window_cycles * period
    # a piece of exactly three windows (L, step L/2); a jump in the middle of the third window's first half
    start = 200.0
    piece = (start, start + 2 * length + 1.0)
    jump = start + 1.1 * length
    guarded = _l4(rec, [piece], rsp.JumpGuard((jump,), 20.0))
    plain = _l4(rec, [piece], None)
    assert len(plain.windows) == 3 and plain.a_response == CheckValue.PASS
    assert len(guarded.windows) < S.mmr_force_min_windows
    assert guarded.a_response == CheckValue.NOT_MEASURED


# ---------------------------------------------------------------- O1 inside the episode's regime


def test_onset_stretch_bounds_by_hand():
    """P = 250 yr, FILTER_EDGE_TRIM_CUTOFFS = 1: reach 2P = 500 yr from the record's ends and from each jump + tau/2."""
    t = np.arange(0.0, 12001.0, 1.0)
    P = 250.0
    assert S.filter_edge_trim_cutoffs == 1.0
    # record start: the stretch before 600 starts at 0 + 500 -> 100 yr left < P
    clear, why = rsp.onset_clear(t, P, 600.0, 2000.0, None, S)
    assert clear == (500.0, 11500.0) and why == rsp.ONSET_EDGE
    # 1400: [900, 1400], the whole 2P stretch
    assert rsp.onset_clear(t, P, 1400.0, 2000.0, None, S)[1] == ''
    guard = rsp.JumpGuard((3000.0, 9000.0), 40.0)
    # after the jump at 3000: 3000 + 20 + 500 = 3520; before 3600 only 80 yr are left
    clear, why = rsp.onset_clear(t, P, 3600.0, 5000.0, guard, S)
    assert clear == (3520.0, 9000.0 - 20.0 - 500.0) and why == rsp.ONSET_JUMP
    assert rsp.onset_clear(t, P, 4200.0, 5000.0, guard, S)[1] == ''
    # the episode's own end next to a jump: the stretch inside is cut too (jump at 8650: 8650 - 20 - 500 = 8130)
    clear, why = rsp.onset_clear(t, P, 8000.0, 8600.0, rsp.JumpGuard((8650.0,), 40.0), S)
    assert clear == (500.0, 8130.0) and why == rsp.ONSET_JUMP
    # a jump inside the episode's last tau/2 still bounds its regime (the regime is the one around its middle)
    clear, _ = rsp.onset_clear(t, P, 7000.0, 9010.0, rsp.JumpGuard((9000.0,), 40.0), S)
    assert clear[1] == 9000.0 - 20.0 - 500.0
    # the record's end cuts the episode: 11500 - 11300 = 200 yr < P
    assert rsp.onset_clear(t, P, 10000.0, 11800.0, None, S)[1] == ''  # 10000-11500 is left: measured
    assert rsp.onset_clear(t, P, 11300.0, 11900.0, None, S)[1] == rsp.ONSET_EDGE
    # no period: nothing to clear
    assert rsp.onset_clear(t, NAN, 4200.0, 5000.0, guard, S) == ((-np.inf, np.inf), '')


def test_a_step_before_the_episode_no_longer_fails_the_onset():
    """a: a step of 0.2 au (20 a-wave amplitudes) at 3000 yr and an a-wave of 0.01 au (P = 250 yr) inside the episode 3600-8000 only. The
    band-passed step rings in the stretch before the episode (O1 = before / inside large); cleared, O1 is not measured."""
    t = np.arange(0.0, 12001.0, 1.0)
    P = 250.0
    inside = (t >= 3600.0) & (t <= 8000.0)
    a = 3.97 + 0.2 * (t >= 3000.0) + np.where(inside, 0.01 * np.sin(2 * np.pi * t / P), 0.0)
    before, during, _ = rsp.onset_bands(t, a, P, 3600.0, 8000.0)
    assert before / during > S.mmr_o1_ratio_fail  # the step alone fails O1
    clear, why = rsp.onset_clear(t, P, 3600.0, 8000.0, rsp.JumpGuard((3000.0,), 40.0), S)
    before, during_clear, _ = rsp.onset_bands(t, a, P, 3600.0, 8000.0, clear=clear)
    assert math.isnan(before) and why == rsp.ONSET_JUMP
    assert during_clear == pytest.approx(during)  # the episode itself is not cut
    assert during == pytest.approx(0.01 / np.sqrt(2), rel=0.15)


# ---------------------------------------------------------------- zone: the e-cycle fitted only when covered


def test_an_unfinished_e_cycle_does_not_take_the_trend():
    """e = 0.39 + 1e-6 t + 0.005 sin(2 pi t / 2000) over 40 kyr with nu = 2 pi / 6 Myr (33128 6N-7). The 6 Myr cycle
    covers 1/150 of itself: a sine of it is a line and a constant there and cannot be fitted. The trend is the line's,
    1e-3 per kyr (the line through 20 whole sine cycles tilts it by 1.2 %); the half-range of e without it is 0.005 about
    0.39 + 0.02 = 0.41: dK/K = q 0.005 / 0.41. With the sine in the design the slope came out -0.95 per kyr."""
    t = np.arange(0.0, 40000.0, 5.0)
    e = 0.39 + 1e-6 * t + 0.005 * np.sin(2 * np.pi * t / 2000.0)
    nu = 2 * np.pi / 6.0e6
    _, dk, trend = zone.modulation_from_eccentricity(t, e, 1, 1.0, nu)
    assert trend == pytest.approx(1e-3, rel=0.02)
    assert dk == pytest.approx(0.005 / 0.41, rel=0.05)
    # a covered cycle is still fitted: e = 0.3 + 0.01 sin(nu t) + 1e-6 t over two cycles of 20 kyr
    t = np.arange(0.0, 40000.0, 5.0)
    nu = 2 * np.pi / 20000.0
    e = 0.3 + 0.01 * np.sin(nu * t) + 1e-6 * t
    _, dk, trend = zone.modulation_from_eccentricity(t, e, 1, 1.0, nu)
    assert trend == pytest.approx(1e-3, rel=0.01)
    assert dk == pytest.approx(0.01 / 0.32, rel=0.02)


# ---------------------------------------------------------------- subresonance scan: test A at a small swing


def test_window_swing_is_the_largest_half_amplitude_inside_the_window():
    tp = SimpleNamespace(times_yr=np.array([0.0, 10.0, 20.0, 30.0, 40.0]), values_rad=np.array([0.0, 0.1, -0.1, 0.3, 0.2]))
    geometry = SimpleNamespace(turning_points=tp)
    # inside [5, 35]: 0.1, -0.1, 0.3 -> half-amplitudes 0.1, 0.2
    assert sub.window_swing(geometry, (5.0, 35.0)) == pytest.approx(0.2)
    assert math.isnan(sub.window_swing(geometry, (11.0, 19.0)))


def _member(key, coherence, centre=0.0):
    spec = ResonanceSpec(key, ResonanceKind.THREE_BODY, ('Earth', 'Saturn'), (4, 9), -5, -8, (0, 0))
    return SubresonanceCandidate(spec, 0, False, 0.1, 0.0, 0.1, centre, coherence, NAN, NAN, NAN, NAN, 1, False, '')


KEYS = ('4E+9S-5-1-1-6', '4E+9S-5-2+0-6', '4E+9S-5+0-1-7')


def test_the_swing_gate_hides_test_a_below_45_degrees():
    """2013 AS27: 4 deg in the scan window. At 44.9 deg the coherences go (R^2 and window counts stay); at 45 deg and
    without turning points (no swing) nothing changes."""
    from resonances.resonance.analysis.ledger import Ledger

    laws = {KEYS[0]: (0.99, 0.8, 5), KEYS[1]: (0.4, 0.3, 5)}
    ledger = Ledger.from_settings(S)
    sigma0, gated, small = sub.swing_gate(0.5, laws, math.radians(44.9), S, ledger)
    assert small and math.isnan(sigma0) and all(math.isnan(c) for c, _, _ in gated.values())
    assert [(r2, n) for _, r2, n in gated.values()] == [(0.8, 5), (0.3, 5)]
    assert sub.swing_gate(0.5, laws, math.radians(45.0), S, ledger) == (0.5, laws, False)
    assert sub.swing_gate(0.5, laws, NAN, S, ledger) == (0.5, laws, False)
    assert S.mmr_a_response_min_amp_deg == 45.0


def test_several_members_that_only_test_a_could_tell_apart_are_undetermined():
    """2013 AS27 4E+9S-5: sigma0 swings 3-4 deg in the scan window, below the 45 deg floor; several members keep a still
    centre and librate themselves. Test A is not measured and nothing else separates them: no member is accepted."""
    members = [_member(key, NAN) for key in KEYS]
    outcome = sub.evaluate(members, NAN, 0.1, 0.1, S, symmetric_centre=False, small_swing=True)
    assert outcome.winner is None and outcome.undetermined
    assert str(Flag.SUBRESONANCE_UNDETERMINED) in outcome.flags
    assert set(outcome.reasons.values()) == {sub.REASON_A_SMALL_SWING}
    assert sub.decide(members, NAN, 0.1, 0.1, S, symmetric_centre=False, small_swing=True) == (None, False, True)
    # the swing gate also hides a coherence the windows did measure
    coherent = [_member(key, 0.99) for key in KEYS[:2]]
    assert sub.evaluate(coherent, 0.1, 0.1, 0.1, S, symmetric_centre=False, small_swing=True).undetermined
    # a nodal member is never accepted: one planar and one nodal left is not undetermined
    nodal = replace(members[1], nodal=True)
    assert not sub.evaluate([members[0], nodal], NAN, 0.1, 0.1, S, symmetric_centre=False, small_swing=True).undetermined
    # one member alone: rejected by the unmeasured test A, nothing undetermined
    outcome = sub.evaluate(members[:1], NAN, 0.1, 0.1, S, symmetric_centre=False, small_swing=True)
    assert outcome.winner is None and not outcome.undetermined and outcome.flags == []


def test_members_excluded_by_b_or_e_do_not_make_the_scan_undetermined():
    # B: centres 57 deg from 0 and pi (tolerance MMR_SUBRESONANCE_CENTRE_TOL_DEG) at low e and i
    assert S.mmr_subresonance_centre_tol_deg < 57.0
    off_centre = [_member(key, NAN, centre=1.0) for key in KEYS]
    outcome = sub.evaluate(off_centre, NAN, 0.05, 0.01, S, symmetric_centre=True, small_swing=True)
    assert outcome.winner is None and not outcome.undetermined
    # one of them at the symmetric centre: a single survivor, not undetermined either
    mixed = [_member(KEYS[0], NAN, centre=np.pi)] + off_centre[1:]
    assert not sub.evaluate(mixed, NAN, 0.05, 0.01, S, symmetric_centre=True, small_swing=True).undetermined
    # E: the strongest predicted T0 10^6 times the observed period
    slow = sub.TheoryCheck(t0_ratio=1.0e6)
    members = [_member(key, NAN) for key in KEYS]
    outcome = sub.evaluate(members, NAN, 0.1, 0.1, S, symmetric_centre=False, theory=slow, small_swing=True)
    assert outcome.winner is None and not outcome.undetermined


def test_a_coherence_missing_for_other_reasons_stays_a_plain_rejection():
    """Fewer than two force windows in the scan window: A is not measured, but not because of the swing; the members
    are rejected as before and the scan is not undetermined (the swing gate is the only new route)."""
    members = [_member(key, NAN) for key in KEYS]
    outcome = sub.evaluate(members, NAN, 0.1, 0.1, S, symmetric_centre=False)
    assert outcome.winner is None and not outcome.undetermined
    assert set(outcome.reasons.values()) == {sub.REASON_A_NOT_MEASURED}
    # a measured, coherent test A still decides
    outcome = sub.evaluate([_member(KEYS[0], 0.99), _member(KEYS[1], 0.1)], 0.1, 0.1, 0.1, S, symmetric_centre=False)
    assert outcome.winner is not None and not outcome.undetermined


# ---------------------------------------------------------------- exit_suspected only on a verdict that rests on it


def _caps_geometry(first_tp, between_tp, second_tp, exit_suspected=True):
    """Two librating episodes on the turning points `first_tp` and `second_tp` (1 kyr apart) with `between_tp` between
    them; an exit 0.5 period before the record's end."""
    values = np.asarray(list(first_tp) + list(between_tp) + list(second_tp), float)
    times = np.arange(len(values), dtype=float) * 1000.0
    tp = make_turning_points(times, values)
    n1, n2 = len(first_tp), len(second_tp)
    first = make_episode(episode_id=0, tp_times_yr=times[:n1], tp_values_rad=values[:n1], period_med_yr=2000.0)
    second = make_episode(episode_id=1, tp_times_yr=times[-n2:], tp_values_rad=values[-n2:], period_med_yr=2000.0)
    centre = make_centre(t_yr=times, centre_rad=np.zeros_like(times))
    edge = geo.EdgeEvent('exit', times[-1] + 500.0, 0.5, 0.5, exit_suspected, True)
    return make_geometry(turning_points=tp, runs=[first, second], centre=centre, exit=edge)


def _caps(angle_status, geometry):
    angle = decide.AngleDecision(angle_status, AngleSubtype.NONE, 'G2', 1)
    return decide._episode_caps(angle, geometry, None, S)


LIBRATING = [0.0, 1.0, -1.0, 1.0, -1.0]


def test_exit_suspected_is_dropped_for_a_transient_that_went_over_the_top_between_episodes():
    """Between the episodes sigma_f covers a range: TPs from -1 (the first episode's last) to the second's first."""
    # one swing from -1 up to 6: 7 rad > 2 pi, the angle went over the top
    over = _caps_geometry(LIBRATING, [6.0, 5.0], [6.0, 7.0, 5.0, 7.0, 5.0])
    assert decide.break_between_episodes(over, S) == pytest.approx(7.0)  # -1, 6, 5, 6 (the second's first TP)
    assert str(Cap.EXIT_SUSPECTED) not in _caps(AngleStatus.EPISODIC, over)
    # a libration rests on its edge: the cap stays
    assert str(Cap.EXIT_SUSPECTED) in _caps(AngleStatus.LIBRATION, over)
    # without a revolution (-1 .. 5: 6 rad < 2 pi) the transient rests on the edge too
    small = _caps_geometry(LIBRATING, [5.0, 4.0], [5.0, 4.5, 5.0, 4.5, 5.0])
    assert decide.break_between_episodes(small, S) == pytest.approx(6.0)
    assert str(Cap.EXIT_SUSPECTED) in _caps(AngleStatus.EPISODIC, small)
    # a drift with small reversals (Codex's counterexample): every swing <= 3 rad, the range -1 .. 10 = 11 rad
    drift = _caps_geometry(LIBRATING, [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0], [10.0, 9.0, 10.0, 9.0, 10.0])
    assert decide.break_between_episodes(drift, S) == pytest.approx(11.0)
    assert str(Cap.EXIT_SUSPECTED) not in _caps(AngleStatus.EPISODIC, drift)
    # nothing suspected, nothing capped
    assert str(Cap.EXIT_SUSPECTED) not in _caps(AngleStatus.LIBRATION, _caps_geometry(LIBRATING, [], LIBRATING, False))


def test_the_multiplet_fallback_halves_are_dropped_at_a_jump():
    """The multiplet law of a 6-kyr pendulum: its modulation period 2 pi / |1e-4 - g5| ~ 63 kyr is longer than the
    record, so it is fitted on the two halves of the piece. A jump zone inside one half leaves no fit; one outside
    the piece changes nothing."""
    rec = pendulum_record()
    rec.context.planet_varpi_rate_rad_yr['Jupiter'] = float(sub.const.PLANETARY_FREQUENCIES['g5'] * np.pi / 180 / 3600)
    angle, _, _, adot, _ = rsp.response_series(rec.prep, rec.geometry.period_cycle_yr, S)
    piece = [rec.geometry.window]
    length = S.mmr_force_window_cycles * rec.geometry.period_cycle_yr
    args = (rec.t, angle, adot, piece, length, rec.spec, rec.slow, rec.context, S)
    plain = rsp._best_multiplet(*args)
    assert plain is not None and len(plain.coefficients) == 2
    assert rsp._best_multiplet(*args, [(1000.0, 1100.0)]) is None
    outside = rsp._best_multiplet(*args, [(-500.0, 0.0)])
    assert outside is not None and outside.r2 == pytest.approx(plain.r2)
