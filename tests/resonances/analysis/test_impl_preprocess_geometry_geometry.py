import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.filtering import edge_window, lowpass
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
ONE_DEG = np.radians(1.0)


def filtered(t, angle, cutoff, slow=None, c=-30.0, rate=None):
    angle_f = lowpass(t, angle, cutoff, 2)
    slow = np.zeros_like(t) if slow is None else slow
    return g.FilteredAngle(
        t_yr=t,
        angle_raw=angle,
        angle_f=angle_f,
        rate_f=np.gradient(angle_f, t) if rate is None else rate,
        angle_f_half=lowpass(t, angle, cutoff / 2, 2),
        angle_f_double=lowpass(t, angle, 2 * cutoff, 2),
        slow_f=lowpass(t, slow, cutoff, 2),
        identity_coefficient=c,
        cutoff_yr=cutoff,
        step_yr=float(t[1] - t[0]),
        window=edge_window(t, cutoff),
    )


def pendulum(t, omega0, h, eps=0.0, modulation_yr=1.0):
    """sigma'' = -omega0^2 (1 + eps sin(2 pi t / T_mod)) sin sigma from sigma = 0 with energy h (h = 1: separatrix)."""

    def rhs(tt, y):
        return [y[1], -(omega0**2) * (1 + eps * np.sin(2 * np.pi * tt / modulation_yr)) * np.sin(y[0])]

    sol = solve_ivp(rhs, (t[0], t[-1]), [0.0, 2 * omega0 * np.sqrt(h)], t_eval=t, rtol=1e-10, atol=1e-12)
    return sol.y[0], sol.y[1]


def tp_from(times, values):
    values = np.asarray(values, dtype=float)
    kinds = np.where(np.r_[values[1:] < values[:-1], values[-1] > values[-2]], 1, -1)
    return g.TurningPoints(np.arange(len(times)), np.asarray(times, dtype=float), values, kinds, ONE_DEG)


def test_prominence_rule():
    # 4 x 0.5 deg = 2 deg beats the 1-deg floor; without ripple the floor stays
    assert g.turning_prominence(ONE_DEG, 4.0, np.radians(0.5)) == pytest.approx(np.radians(2.0))
    assert g.turning_prominence(ONE_DEG, 4.0, 0.0) == pytest.approx(ONE_DEG)


def test_turning_points_ignore_ripple_below_prominence_and_alternate():
    t = np.arange(0, 1000, 0.1)
    angle = np.sin(2 * np.pi * t / 100) + 0.004 * np.sin(2 * np.pi * t / 3.0)  # ripple extrema <= 0.008 rad
    tp = g.find_turning_points(t, angle, 0.02, (50.0, 950.0))
    assert len(tp) == 18  # extrema at 25 + 50 k inside [50, 950]
    assert np.all(np.diff(tp.kinds) != 0)
    assert np.all(np.abs(tp.values_rad) > 0.99)
    assert g.full_periods(tp) == pytest.approx(100.0, abs=2.0)  # the ripple moves the extrema slightly
    assert len(g.find_turning_points(t, angle, 0.02, (np.nan, np.nan))) == 0


def test_period_amplitude_midpoint_by_hand():
    tp = tp_from([0.0, 5.0, 11.0, 15.0], [1.0, -0.5, 1.5, 0.1])
    assert g.full_periods(tp) == pytest.approx([11.0, 10.0])
    assert g.half_amplitudes(tp) == pytest.approx([0.75, 1.0, 0.7])
    assert g.midpoints(tp) == pytest.approx([0.25, 0.5, 0.8])


def test_split_runs_on_gap_top_and_midpoint_jump():
    # half-periods 10, 10, then a gap of 50 (> 3 x 10): break
    gap = tp_from([0, 10, 20, 70, 80, 90], [1, -1, 1, -1, 1, -1])
    assert [list(r) for r in g.split_runs(gap, S)] == [[0, 1, 2], [3, 4, 5]]
    # half-cycle 2 -> 3 goes from -1 to 7.5: half-amplitude 4.25 > pi (over the top)
    top = tp_from([0, 10, 20, 30, 40, 50], [1, -1, 1, 7.5, 5.5, 7.5])
    assert [list(r) for r in g.split_runs(top, S)] == [[0, 1, 2], [3, 4, 5]]
    # midpoints 0, 0, then 3.5 (|3.5 - 0| > pi) while the half-amplitude stays 2.5
    jump = tp_from([0, 10, 20, 30, 40], [1, -1, 1, 6, 1])
    assert [list(r) for r in g.split_runs(jump, S)] == [[0, 1, 2], [3, 4]]


def test_short_runs_rejected_with_reason():
    tp = tp_from([0, 10, 20, 30, 40, 50], [1, -1, 1, 7.5, 5.5, 7.5])
    t = np.arange(0, 60, 1.0)
    first = g.classify_run(0, tp, np.array([0, 1]), t, (0, 59), S)
    assert not first.accepted and first.reject_reason == g.RejectReason.TOO_FEW_TP
    second = g.classify_run(1, tp, np.array([2]), t, (0, 59), S)
    assert second.reject_reason == g.RejectReason.OVER_THE_TOP


def test_stationary_libration_covers_the_record():
    t = np.arange(0, 20000, 1.0)
    geo = g.measure_geometry(filtered(t, np.pi + np.radians(60) * np.sin(2 * np.pi * t / 500), 100.0), ONE_DEG, S)
    assert geo.period_cycle_yr == pytest.approx(500.0, abs=1.0)
    episode = g.covering_episode(geo.episodes, S)
    assert episode is not None and episode.shape == g.EpisodeShape.STATIONARY
    assert episode.coverage_start_gap_p < 1 and episode.coverage_end_gap_p < 1
    assert episode.half_amp_med_rad == pytest.approx(np.radians(60), rel=0.02)
    assert episode.centre_mod_2pi_rad == pytest.approx(np.pi, abs=0.01)
    assert geo.wiggle_half_amp_rad == pytest.approx(np.radians(60), rel=0.03)
    assert geo.wiggle_period_yr == pytest.approx(500.0, rel=0.03)
    assert geo.centre.max_pass_rev < 0.1 and geo.bands_overlap


def test_centre_out_and_back_is_a_circulation_by_the_largest_pass():
    # the centre goes +1.25 turns in 15 kyr and back: net ~0, largest monotone pass ~1.25 turns
    t = np.arange(0, 30000, 1.0)
    centre = 1.25 * 2 * np.pi * np.where(t < 15000, t / 15000, (30000 - t) / 15000)
    geo = g.measure_geometry(filtered(t, centre + np.radians(30) * np.sin(2 * np.pi * t / 300), 60.0), ONE_DEG, S)
    assert abs(geo.centre.net_rev) < 0.05
    assert geo.centre.max_pass_rev == pytest.approx(1.25, abs=0.05)
    assert geo.centre.max_pass_rev_lo <= geo.centre.max_pass_rev <= geo.centre.max_pass_rev_hi
    assert len(geo.centre.turns) == 1


def test_bounded_centre_turning_back_is_modulated():
    # centre +-60 deg over 2.25 of its periods (4 returns); first and last bands do not overlap
    t = np.arange(0, 30000, 1.0)
    centre = np.radians(60) * np.sin(2 * np.pi * t / (30000 / 2.25))
    geo = g.measure_geometry(filtered(t, centre + np.radians(20) * np.sin(2 * np.pi * t / 300), 60.0), ONE_DEG, S)
    (run,) = geo.runs
    assert run.accepted and run.shape == g.EpisodeShape.MODULATED
    assert not run.bands_overlap and run.centre_returns >= S.libration_centre_returns_modulated


def test_small_wave_on_a_slope_is_drift():
    t = np.arange(0, 30000, 1.0)
    angle = 0.4 * 2 * np.pi * t / 30000 + np.radians(20) * np.sin(2 * np.pi * t / 300)
    geo = g.measure_geometry(filtered(t, angle, 60.0), ONE_DEG, S)
    (run,) = geo.runs
    assert not run.accepted and run.reject_reason == g.RejectReason.CENTRE_MOVES and not run.bands_overlap
    assert geo.drift_fraction > S.libration_drift_min_window_fraction
    assert geo.centre.max_pass_rev == pytest.approx(0.4, abs=0.03)


def test_slips_of_a_modulated_pendulum_and_their_duration():
    t = np.arange(0, 20000, 1.0)
    angle, _ = pendulum(t, 2 * np.pi / 1000, 0.97, eps=0.15, modulation_yr=5000)
    filtered_angle = filtered(t, angle, 100.0)
    geo = g.measure_geometry(filtered_angle, ONE_DEG, S)
    slips, rejected = g.find_slips(t, filtered_angle.angle_f, geo.turning_points, 1000.0, [], S)
    assert rejected == 0 and len(slips) == 2
    assert [round(slip.size_rev) for slip in slips] == [-1, -1]
    assert sum(slip.size_rev for slip in slips) == pytest.approx(geo.net_rev_filtered, abs=0.2)
    assert all(slip.duration_yr >= 500.0 for slip in slips)
    # the same steps are unwrap errors if the pendulum were ten times slower (T0 = 10 kyr)
    assert g.find_slips(t, filtered_angle.angle_f, geo.turning_points, 1e4, [], S) == ([], 2)
    marked, _ = g.find_slips(t, filtered_angle.angle_f, geo.turning_points, 1000.0, [slips[0].time_yr], S)
    assert marked[0].suspect and not marked[1].suspect


def test_inserted_two_pi_step_is_not_a_slip():
    t = np.arange(0, 20000, 1.0)
    angle, _ = pendulum(t, 2 * np.pi / 1000, np.sin(np.radians(50)) ** 2)  # A = 100 deg
    for jump_yr in (10000.0, 10100.0, 10300.0, 10500.0):
        stepped = filtered(t, angle + 2 * np.pi * (t >= jump_yr), 100.0)
        geo = g.measure_geometry(stepped, ONE_DEG, S)
        assert g.find_slips(t, stepped.angle_f, geo.turning_points, 1000.0, [], S)[0] == []


def test_revolution_rate_minimum_matches_the_pendulum():
    # above the separatrix at h = 1.14 a pendulum's min/mean rate over a turn is 0.554 (at the top, pi)
    t = np.arange(0, 20000, 0.5)
    angle, rate = pendulum(t, 2 * np.pi / 1000, 1.14)
    turns = g.revolutions(t, angle, rate, (0.0, t[-1]))
    assert len(turns) == int(abs(angle[-1] - angle[0]) // (2 * np.pi))
    assert np.median([turn.rate_min_over_mean for turn in turns]) == pytest.approx(0.554, abs=0.01)
    assert all(abs(g.wrap_pi(turn.phase_min_rate_rad - np.pi)) < 0.05 for turn in turns)


def test_pattern_b_reads_energy_and_phase():
    revs = [g.Revolution(i * 100.0, (i + 1) * 100.0, np.pi + 0.1 * (-1) ** i, 0.3, 1.05) for i in range(8)]
    found = g.pattern_b(revs, np.pi, 1.1, 0.01, S)
    assert found.detected and found.in_zone and found.stall_fraction == 1.0
    assert found.stall_phase_mean_rad == pytest.approx(np.pi, abs=1e-6)
    assert not g.pattern_b(revs, np.pi, 1.0, 0.01, S).detected  # turns above the zone
    assert not g.pattern_b(revs, 0.0, 1.1, 0.01, S).detected  # minima at the stable point
    assert g.pattern_b([], np.pi, 1.1, 0.01, S).n_revolutions == 0


def test_stall_bounds_by_hand():
    t = np.arange(0, 10000, 1.0)
    angle = 0.6 + 2e-5 * t  # 34.4 .. 45.8 deg: a stall
    slow = 2.0 + 1e-8 * t  # da = 1e-4 au over tau
    arc = g.arc_or_stall(t, angle, slow, -30.0, (0.0, t[-1]))
    tau, d_slow, sin_min = t[-1], 1e-8 * t[-1], np.sin(0.6)
    assert arc.kind == 'stall' and arc.tau_yr == tau
    assert arc.rate_rad_yr == pytest.approx(2e-5)
    assert arc.force_bound_range == pytest.approx(d_slow / (tau * sin_min))
    assert arc.force_bound_trend == pytest.approx(1e-8 / sin_min)
    assert arc.period_min_yr == pytest.approx(2 * np.pi * np.sqrt(tau * sin_min / (30.0 * d_slow)))
    assert np.degrees(arc.wrapped_min_rad) == pytest.approx(np.degrees(0.6), abs=1e-6)
    assert g.arc_or_stall(t, 1e-3 * t, slow, -30.0, (0.0, t[-1])).kind == 'arc'


def test_visible_exit_after_the_last_turning_point():
    # libration around pi (A = 60 deg, P = 500 yr) for 15 kyr, then circulation at 2 rad/kyr
    t = np.arange(0, 18000, 1.0)
    libration = np.pi + np.radians(60) * np.sin(2 * np.pi * t / 500)
    angle = np.where(t < 15000, libration, libration[15000] + 2e-3 * (t - 15000))
    geo = g.measure_geometry(filtered(t, angle, 50.0), ONE_DEG, S)
    exit_event = geo.exit
    assert exit_event is not None and exit_event.kind == 'exit'
    # |sigma - pi| > pi once the circulation has added pi - 0 rad at 2e-3 rad/yr
    assert exit_event.time_yr == pytest.approx(15000 + np.pi / 2e-3, abs=60)
    assert exit_event.remaining_periods == pytest.approx((t[-1] - exit_event.time_yr) / 500, rel=0.01)
    assert exit_event.suspected is False and exit_event.raw_monotonic
    assert geo.entry is None


def test_jump_locator_confirmation_and_last_regime():
    t = np.arange(0, 20000, 1.0)
    period = 500.0
    angle = np.where(
        t < 10000, np.pi + np.radians(40) * np.sin(2 * np.pi * t / period), np.pi + 2 + np.radians(100) * np.sin(2 * np.pi * t / period)
    )
    slow = 1.0 + 1e-4 * np.cos(2 * np.pi * t / period) + 1e-3 * (t >= 10000)
    angle_in = filtered(t, angle, 50.0, slow=slow)
    tau = g.jump_baseline_yr(50.0, 1.0, S)
    assert tau == pytest.approx(100.0)  # max(2 x 50, 5 x 1)
    (candidate,) = g.jump_candidates(t, angle_in.slow_f, tau, period, S)
    assert candidate.time_yr == pytest.approx(10000, abs=tau)

    cycles = np.arange(250, 20000, 500.0)
    energies = np.where(cycles < 10000, 0.2, 0.8)
    (jump,) = g.confirm_jumps([candidate], cycles, energies, 0.02, 5e-4, S)
    assert not jump.kick and jump.dh == pytest.approx(0.6)
    (kick,) = g.confirm_jumps([candidate], cycles, np.full_like(cycles, 0.5), 0.02, 5e-4, S)
    assert kick.kick
    (by_slow,) = g.confirm_jumps([candidate], cycles[:1], energies[:1], 0.02, 5e-4, S)
    assert by_slow.by_slow_only and not by_slow.kick and np.isnan(by_slow.dh)
    assert g.last_regime_start(t, [jump], tau) == pytest.approx(candidate.time_yr + tau / 2)
    assert g.last_regime_start(t, [kick], tau) == t[0]

    geo = g.measure_geometry(angle_in, ONE_DEG, S)
    geo = g.finish_geometry(geo, angle_in, period, np.nan, np.nan, 0.02, cycles, energies, 5e-4, period, [], S)
    assert geo.regime_start_yr == pytest.approx(candidate.time_yr + tau / 2)
    assert geo.window[0] == pytest.approx(geo.regime_start_yr)
    assert 'last_regime_after_jump' in geo.flags
    assert all(run.before_last_jump for run in geo.runs if run.t_start_yr < geo.regime_start_yr)
    episode = g.covering_episode(geo.episodes, S)
    assert episode is not None and episode.half_amp_med_rad == pytest.approx(np.radians(100), rel=0.02)
    assert [run.episode_id for run in geo.runs] == list(range(len(geo.runs)))


def test_columns_in_csv_units():
    t = np.arange(0, 20000, 1.0)
    geo = g.measure_geometry(filtered(t, np.pi + np.radians(60) * np.sin(2 * np.pi * t / 500), 100.0), ONE_DEG, S)
    columns = geo.to_columns()
    assert columns['geo_tp_prominence_deg'] == pytest.approx(1.0)
    assert columns['geo_n_tp'] == len(geo.turning_points)
    assert columns['geo_wiggle_half_amp_deg'] == pytest.approx(60, rel=0.03)
    assert columns['geo_period_cycle_yr'] == pytest.approx(500, abs=1)
    for key in ('stall_P_lib_min_kyr', 'exit_time_yr', 'jump_times_yr', 'regime_start_yr', 'geo_centre_max_pass_rev_lo'):
        assert key in columns
