"""W20 CS4: the edge rule (variant a), closure by phase return, the 1:1 cap transient_unexplained (R2) and the shared
geometry staying unchanged for secular/ZLK.

Expected numbers are computed by hand from the construction of each record:
- E1: a libration of half-amplitude 60 deg and period 500 yr ends at a maximum 400 yr before the record's end, then sigma
  runs down at 2 pi per 150 yr: the raw arc from that turning point to the end is 400 / 150 x 2 pi = 16.755 rad
  (2.667 turns), and sigma leaves the centre by pi after (60 + 180) / 360 x 150 = 100 yr. With the exit 0.75 turn
  before the end the arc is 1.5 pi = 4.712 rad < 2 pi.
- E2: sigma = c + A sin(2 pi (t - t0) / P), A = 170 deg, P = 374 yr, over one cycle from c upwards to c upwards; the two
  turning points are P/2 apart and their mid-level is c, crossed upwards at t0 and t0 + P: t_b - t_a = P = 374 yr,
  (t_b - t_a) / (2 Delta) = 1. Outside the cycle sigma circulates upwards at the rate of the sine at the joins.
- The 1864 type: a circulation at -1 deg/yr whose rate w (1 - D exp(-((t - t0)/50)^2)) turns positive for a while; for
  D = 1.2 sigma runs back by 1 deg/yr x [1.2 x 50 sqrt(pi) erf(sqrt(ln 1.2)) - 2 x 50 sqrt(ln 1.2)] = 5.6 deg: two
  turning points, crossed again right next to them, (t_b - t_a) / (2 Delta) ~ 0.5.
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.filtering import edge_window, lowpass
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.decide import AngleDecision
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.pendulum import SeparatrixMechanism
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype
from tests.resonances.analysis.builders import make_episode, make_geometry

S = settings_from_config()
N_JUPITER = 2 * np.pi / 11.862615
ONE_DEG = math.radians(1.0)


def mmr_series(t, sigma, sigma_dot, name):
    """2J-1: sigma = 2 lambda_J - lambda - varpi, varpi constant, sigma_dot = 2 n_J - n(a)."""
    n = 2 * N_JUPITER - sigma_dot
    a = (2 * np.pi / n) ** (2.0 / 3.0)
    rows = len(t)
    varpi = np.full(rows, 0.7)
    lam = 2 * N_JUPITER * t - sigma - varpi
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full(rows, 0.15),
        inc=np.full(rows, 0.05),
        Omega=np.zeros(rows),
        omega=varpi.copy(),
        varpi=varpi,
        angle_wrapped=np.mod(sigma, 2 * np.pi),
        resonance=ResonanceSpec.from_key('2J-1'),
        regime=Regime(step_yr=float(t[1] - t[0]), span_yr=float(t[-1] - t[0]), n_rows=rows),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name=name,
    )


def analysed(series):
    return analyse_mmr(series, S, scan_owner=False, scan_subresonances=False)


# --- E1, the edge rule ---------------------------------------------------------------------------------------------

CENTRE, AMPLITUDE, PERIOD, CIRCULATION = np.pi, math.radians(60.0), 500.0, 150.0


def libration_then_exit(turns_after: float, entry: bool = False):
    """A libration (60 deg, 500 yr) whose last turning point, a maximum at t_last = 4625 yr, is followed by
    `turns_after` turns of a downward circulation (2 pi per 150 yr) to the record's end. `entry`: the record reversed
    in time (a circulation coming down into the libration, then the same libration)."""
    t_last = 4625.0
    t = np.arange(0.0, t_last + turns_after * CIRCULATION + 0.5, 1.0)
    rate = -2 * np.pi / CIRCULATION
    sigma = np.where(t <= t_last, CENTRE + AMPLITUDE * np.sin(2 * np.pi * t / PERIOD), CENTRE + AMPLITUDE + rate * (t - t_last))
    sigma_dot = np.where(t <= t_last, AMPLITUDE * 2 * np.pi / PERIOD * np.cos(2 * np.pi * t / PERIOD), rate)
    if entry:
        sigma, sigma_dot = sigma[::-1], -sigma_dot[::-1]
    return t, sigma, sigma_dot


@pytest.fixture(scope='module')
def observed_exit():
    t, sigma, sigma_dot = libration_then_exit(400.0 / CIRCULATION)
    return analysed(mmr_series(t, sigma, sigma_dot, 'exit-observed')), t


def test_an_exit_whose_raw_arc_exceeds_a_turn_is_observed(observed_exit):
    result, t = observed_exit
    exit_event = result.geometry.exit
    assert exit_event is not None and exit_event.observed and not exit_event.suspected
    # 400 / 150 x 2 pi from the last maximum to the end (the 1-yr step is below the 5.9-yr orbit: averaged over it)
    assert exit_event.raw_arc_rad == pytest.approx(400.0 / CIRCULATION * 2 * np.pi, rel=0.02)
    v = result.verdict
    assert (str(v.status), str(v.angle_status)) == ('transient', 'episodic')
    assert 'exit_suspected' not in v.caps and 'exit_suspected' not in v.flags
    # the last regime: circulation from the exit, sigma leaves the centre by pi 100 yr after the last maximum
    assert str(v.final_regime) == 'circulation'
    assert v.final_regime_start_yr == pytest.approx(4625.0 + 100.0, abs=20.0)
    assert result.numbers_row()['exit_raw_arc_rev'] == pytest.approx(400.0 / CIRCULATION, rel=0.02)


def test_an_exit_of_less_than_a_turn_stays_suspected():
    t, sigma, sigma_dot = libration_then_exit(0.75)
    result = analysed(mmr_series(t, sigma, sigma_dot, 'exit-suspected'))
    exit_event = result.geometry.exit
    assert exit_event is not None and not exit_event.observed and exit_event.suspected
    assert exit_event.raw_arc_rad == pytest.approx(1.5 * np.pi, rel=0.05)
    v = result.verdict
    assert (str(v.status), str(v.angle_status)) == ('resonant', 'libration')
    assert 'exit_suspected' in v.caps and str(v.confidence) in ('medium', 'low')


def test_an_entry_whose_raw_arc_exceeds_a_turn_is_observed():
    t, sigma, sigma_dot = libration_then_exit(400.0 / CIRCULATION, entry=True)
    result = analysed(mmr_series(t, sigma, sigma_dot, 'entry-observed'))
    entry_event = result.geometry.entry
    assert entry_event is not None and entry_event.observed and not entry_event.suspected
    assert entry_event.raw_arc_rad == pytest.approx(400.0 / CIRCULATION * 2 * np.pi, rel=0.02)
    v = result.verdict
    # the angle shows an episode, not a libration over the record (the status reads its physics: a sine and a
    # uniform circulation are no pendulum, R0 objects here)
    assert str(v.angle_status) == 'episodic' and str(v.status) != 'resonant'
    assert 'entry_suspected' not in v.caps
    assert str(v.final_regime) == 'libration'


def _edge_episode():
    # an episode about the centre pi, period 500 yr, its last turning point at 1000 yr (a maximum, pi + 60 deg)
    return SimpleNamespace(centre_mean_rad=np.pi, period_med_yr=500.0, t_start_yr=0.0, t_end_yr=1000.0)


def test_an_exit_the_filter_hides_is_read_on_the_averaged_angle():
    """FIX1 (Codex A #10): after the turning point the averaged angle runs down at 2 pi per 150 yr for 200 yr (1.33
    turns, monotonic), while sigma_f, bent at the record's edge, stays within pi of the centre: the exit is observed on
    the averaged angle, at its own crossing pi + 60 deg - 240 deg = centre - pi, (60 + 180) / 360 x 150 = 100 yr after
    the turning point."""
    t = np.arange(0.0, 1200.5, 1.0)
    top = np.pi + math.radians(60.0)
    averaged = np.where(t <= 1000.0, top, top - 2 * np.pi / 150.0 * (t - 1000.0))
    hidden = np.minimum(np.abs(averaged - np.pi), np.pi - 0.1) * np.sign(averaged - np.pi) + np.pi
    event = g._visible_edge(t, hidden, averaged, _edge_episode(), 'exit', (0.0, 1200.0), S, averaged)
    assert event is not None and event.observed and not event.suspected and event.raw_monotonic
    assert event.raw_arc_rad == pytest.approx(200.0 / 150.0 * 2 * np.pi, rel=1e-6)
    assert event.time_yr == pytest.approx(1100.0, abs=1.0)
    assert g._visible_edge(t, hidden, averaged, _edge_episode(), 'exit', (0.0, 1200.0), S) is None  # rule off: no crossing


def test_a_reversal_before_the_crossing_is_no_observed_exit():
    """FIX1 (Codex A #10): the averaged angle first swings down by 100 deg over 600 yr (more than two half-period blocks
    of 250 yr), then runs up at 2 pi per 150 yr for 400 yr (2.67 turns up, 2.39 net from the turning point). After its
    crossing it is monotonic (the old reading observed the exit); from the turning point it is not: no observed exit."""
    t = np.arange(0.0, 2000.5, 1.0)
    top = np.pi + math.radians(60.0)
    down = top - math.radians(100.0) * (t - 1000.0) / 600.0
    up = top - math.radians(100.0) + 2 * np.pi / 150.0 * (t - 1600.0)
    averaged = np.where(t <= 1000.0, top, np.where(t <= 1600.0, down, up))
    event = g._visible_edge(t, averaged, averaged, _edge_episode(), 'exit', (0.0, 2000.0), S, averaged)
    assert event is not None and not event.observed and not event.raw_monotonic
    assert event.raw_arc_rad == pytest.approx(math.radians(960.0 - 100.0), rel=1e-6)
    # sigma crosses pi + pi (above the centre) at 1600 + (180 + 40) / 360 x 150 = 1691.7 yr: 308 yr = 0.62 P left
    assert event.time_yr == pytest.approx(1691.7, abs=1.5) and event.suspected


def test_the_orbit_average_is_a_running_mean_over_one_orbit():
    # 0.1-yr steps, a 1-yr orbit: windows of 11 samples (half = 5); a line keeps its values inside, a sine of the
    # orbital period averages to sin(x) sinc-like: mean of sin over 11 samples of 0.1 x 2 pi spacing about x
    t = np.arange(0.0, 20.0, 0.1)
    line = 3.0 * t
    assert g.orbit_averaged(t, line, 1.0)[50:-50] == pytest.approx(line[50:-50])
    wave = np.sin(2 * np.pi * t)
    expected = np.sin(2 * np.pi * t[100]) * np.sum(np.cos(2 * np.pi * 0.1 * np.arange(-5, 6))) / 11
    assert g.orbit_averaged(t, wave, 1.0)[100] == pytest.approx(expected, abs=1e-12)
    assert g.orbit_averaged(t, wave, 0.05) is not None and np.array_equal(g.orbit_averaged(t, wave, 0.05), wave)


@pytest.mark.parametrize(
    'example, kind, arc_rev',
    [
        ('coorbital__2017_UM52__1E-1+0+0', 'exit', 712.0 / 360.0),
        ('coorbital__2022_VR1__1E-1+0+0', 'exit', 568.0 / 360.0),
        ('coorbital__2026_FW12__1E-1+0+0', 'entry', 720.0 / 360.0),
    ],
)
def test_benchmark_edges_observed(example, kind, arc_rev):
    """The arcs of the E report (19.10, measured independently on the raw angle averaged over one orbit; Codex 1.96,
    1.57, 2.00 turns) within 5 %; the three co-orbitals are transient at high (FW12: no transient_unexplained, R2)."""
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    event = getattr(result.geometry, kind)
    assert event.observed and event.raw_arc_rad / (2 * np.pi) == pytest.approx(arc_rev, rel=0.05)
    v = result.verdict
    assert (str(v.status), str(v.angle_status), str(v.confidence)) == ('transient', 'episodic', 'high')
    assert f'{kind}_suspected' not in v.caps and 'transient_unexplained' not in v.caps


# --- E2, closure by phase return -------------------------------------------------------------------------------------


def filtered(t, angle, cutoff, slow, c=-30.0, **rules):
    angle_f = lowpass(t, angle, cutoff, 2)
    return g.FilteredAngle(
        t_yr=t,
        angle_raw=angle,
        angle_f=angle_f,
        rate_f=np.gradient(angle_f, t),
        angle_f_half=lowpass(t, angle, cutoff / 2, 2),
        angle_f_double=lowpass(t, angle, 2 * cutoff, 2),
        slow_f=lowpass(t, slow, cutoff, 2),
        identity_coefficient=c,
        cutoff_yr=cutoff,
        step_yr=float(t[1] - t[0]),
        window=edge_window(t, cutoff),
        **rules,
    )


def one_cycle(period=374.0, amplitude_deg=170.0, t0=1000.0, c=-30.0):
    """sigma = A sin(2 pi (t - t0) / P) on [t0 - P/8, t0 + P + P/8], a circulation upwards at the sine's rate at the
    joins outside it; a = sigma_dot / c (the identity)."""
    t = np.arange(0.0, 2 * t0 + period, 0.5)
    amplitude, omega = math.radians(amplitude_deg), 2 * np.pi / period
    start, end = t0 - period / 8, t0 + period + period / 8
    join_rate = amplitude * omega * math.cos(math.pi / 4)
    sine = amplitude * np.sin(omega * (t - t0))
    sigma = np.where(
        t < start,
        -amplitude * math.sin(math.pi / 4) + join_rate * (t - start),
        np.where(t > end, amplitude * math.sin(math.pi / 4) + join_rate * (t - end), sine),
    )
    rate = np.where((t >= start) & (t <= end), amplitude * omega * np.cos(omega * (t - t0)), join_rate)
    return t, sigma, rate / c


def two_tp_run(geometry):
    (run,) = [run for run in geometry.runs if run.n_tp == 2]
    return run


def test_one_cycle_through_two_turning_points_is_closed_by_its_return():
    t, sigma, slow = one_cycle()
    geometry = g.measure_geometry(filtered(t, sigma, 20.0, slow, return_closure=True), ONE_DEG, S)
    run = two_tp_run(geometry)
    assert run.accepted and run.closed_by_return and str(run.shape) == 'stationary'
    assert run.return_t_b_yr - run.return_t_a_yr == pytest.approx(374.0, abs=2.0)
    assert run.period_med_yr == pytest.approx(374.0, abs=2.0)
    assert (run.return_t_a_yr, run.return_t_b_yr) == (pytest.approx(1000.0, abs=2.0), pytest.approx(1374.0, abs=2.0))
    # the rule is off for secular/ZLK (FilteredAngle default): the same run stays short of an episode
    off = two_tp_run(g.measure_geometry(filtered(t, sigma, 20.0, slow), ONE_DEG, S))
    assert not off.accepted and str(off.reject_reason) == 'too_few_tp'


def back_step(t, fast_rate=-ONE_DEG, start=1500.0, length=100.0, climb_deg=5.0, edge=2.0):
    """A circulation at `fast_rate` that turns back by `climb_deg` over [start, start + length] (smooth steps of width
    `edge` yr): the rate, and sigma integrated from it."""
    inside = 0.5 * (np.tanh((t - start) / edge) - np.tanh((t - start - length) / edge))
    rate = fast_rate + inside * (math.radians(climb_deg) / length - fast_rate)
    step = float(t[1] - t[0])
    return rate, np.concatenate([[0.0], np.cumsum((rate[1:] + rate[:-1]) / 2 * step)])


def test_a_circulation_with_a_short_back_step_is_no_cycle():
    """1864 3E-5S-5 type (311 deg down, 5 deg back, 233 deg down): sigma reaches its turning points fast and creeps
    between them, so it crosses their mid-level right next to them: t_b - t_a = Delta + 2 x 2.5 deg / (1 deg/yr) =
    105 yr against 2 Delta = 200 yr, ratio 0.525 (rule 4)."""
    t = np.arange(0.0, 3000.0, 0.5)
    rate, sigma = back_step(t)
    angle = filtered(t, sigma, 5.0, rate / -30.0, return_closure=True)
    geometry = g.measure_geometry(angle, ONE_DEG, S)
    run = two_tp_run(geometry)
    assert not run.accepted
    numbers = g.return_cycle(angle, geometry.turning_points, np.array([0, 1]))
    assert numbers.period_ratio == pytest.approx(0.525, abs=0.05)  # the 2-yr steps and the 5-yr filter round the turns
    assert numbers.identity_corr > S.libration_return_identity_corr_min  # a follows sigma: only rule 4 rejects it
    assert g.phase_return(angle, geometry.turning_points, np.array([0, 1]), S) is None


def test_a_smooth_reversal_of_a_circulation_is_no_cycle():
    """A rate w (1 - 1.2 exp(-((t - 1500)/50)^2)), w = -1 deg/yr: sigma turns back by 5.6 deg as smoothly as it came.
    Rules 1-5 hold (the mid-level is crossed a quarter of the way out and in, and a follows sigma), but sigma crosses
    the mid-level forth at the speed of its circulation and back at the small speed of the reversal: rule 6."""
    t = np.arange(0.0, 3000.0, 0.5)
    rate = -math.radians(1.0) * (1 - 1.2 * np.exp(-(((t - 1500.0) / 50.0) ** 2)))
    sigma = np.concatenate([[0.0], np.cumsum((rate[1:] + rate[:-1]) / 2 * 0.5)])
    angle = filtered(t, sigma, 5.0, rate / -30.0, return_closure=True)
    geometry = g.measure_geometry(angle, ONE_DEG, S)
    assert not two_tp_run(geometry).accepted
    numbers = g.return_cycle(angle, geometry.turning_points, np.array([0, 1]))
    assert numbers.slow_mismatch <= S.libration_return_tol and numbers.rate_mismatch <= S.libration_return_tol
    assert abs(numbers.period_ratio - 1) <= S.libration_return_period_tol
    assert numbers.identity_corr >= S.libration_return_identity_corr_min
    assert numbers.swing_asymmetry > S.libration_return_tol


def test_a_back_swing_with_a_flat_a_is_no_cycle():
    """The near-episode record of test_impl_w4_status_episodes (D = 1.6): rule 4 alone would pass it, rule 5 (a does
    not carry the swing) does not."""
    t = np.arange(0, 6000, 1.0)
    rate = 2 * np.pi / 400 * (1 - 1.6 * np.exp(-(((t - 3000) / 150.0) ** 2)))
    angle_raw = np.concatenate([[0.0], np.cumsum((rate[1:] + rate[:-1]) / 2)])
    slow = 1e-4 * np.sin(2 * np.pi * t / 37.0)  # a wanders, unrelated to sigma
    angle = filtered(t, angle_raw, 20.0, slow, return_closure=True)
    geometry = g.measure_geometry(angle, ONE_DEG, S)
    run = two_tp_run(geometry)
    assert not run.accepted and str(run.reject_reason) == 'too_few_tp'
    numbers = g.return_cycle(angle, geometry.turning_points, np.array([0, 1]))
    assert abs(numbers.period_ratio - 1) <= S.libration_return_period_tol  # rule 4 passes
    assert not numbers.identity_corr >= S.libration_return_identity_corr_min  # rule 5 fails
    assert decide.near_episode_tp(geometry, S) == 2


def test_one_return_cycle_caps_medium_by_setting():
    episode = make_episode(n_tp=2, episode_id=0, closed_by_return=True)
    geometry = make_geometry(runs=[episode])
    angle = AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', 0)
    assert decide._episode_caps(angle, geometry, None, S) == [str(Cap.ONE_RETURN_CYCLE)]
    off = replace(S, libration_return_cycle_caps_medium=False)
    assert decide._episode_caps(angle, geometry, None, off) == []


def test_a_return_beyond_the_regime_does_not_close_its_run():
    """FIX1 (Codex A #3): the cycle of `one_cycle` has its turning points at t0 + P/4 = 1093.5 and t0 + 3P/4 = 1280.5 yr
    and crosses its mid-level at t0 = 1000 and t0 + P = 1374 yr. In a regime [1050, 1330) between two jumps both
    crossings lie beyond the jumps: the run is not closed; in a regime [900, 1450) it is, with the same crossings."""
    t, sigma, slow = one_cycle()
    angle = filtered(t, sigma, 20.0, slow, return_closure=True)
    record, window = (float(t[0]), float(t[-1])), tuple(angle.window)

    def runs_in(start, end):
        regime = g.JumpRegime(start_yr=start, end_yr=end, next_start_yr=end, prominence_rad=ONE_DEG)
        return [run for run in g._runs_before(angle, [regime], record, window, S) if run.n_tp == 2]

    (inside,) = runs_in(1050.0, 1330.0)
    assert inside.tp_times_yr[0] == pytest.approx(1093.5, abs=2.0) and inside.tp_times_yr[1] == pytest.approx(1280.5, abs=2.0)
    assert not inside.closed_by_return and not inside.accepted
    (wide,) = runs_in(900.0, 1450.0)
    assert wide.closed_by_return and wide.accepted
    assert (wide.return_t_a_yr, wide.return_t_b_yr) == (pytest.approx(1000.0, abs=2.0), pytest.approx(1374.0, abs=2.0))


@pytest.mark.parametrize(
    'example, expected',
    [
        ('coorbital__2016_CA138__1E-1+0+0', ('transient', 'episodic', 'medium')),
        ('coorbital__3753__1E-1+0+0', ('resonant', 'libration', 'medium')),
    ],
)
def test_benchmark_cycles_closed_by_return(example, expected):
    """CA138 (E report: -180 deg crossed downwards at 80.1 and 453.9 yr, a mismatch 1e-5 au) and 3753 at 1 kyr (the
    return 12.6 -> 782.9 yr on the raw angle; the filtered angle crosses its mid-level at 23 and 793 yr): one cycle
    each, capped at medium (one_return_cycle; author 2026-10-19 q5)."""
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    v = result.verdict
    assert (str(v.status), str(v.angle_status), str(v.confidence)) == expected
    assert 'one_return_cycle' in v.caps
    (episode,) = [run for run in result.geometry.runs if run.closed_by_return]
    if 'CA138' in example:
        assert (episode.return_t_a_yr, episode.return_t_b_yr) == (pytest.approx(80.1, abs=3.0), pytest.approx(453.9, abs=3.0))
        a = np.interp([episode.return_t_a_yr, episode.return_t_b_yr], result.prep.t_yr, result.prep.a_f)
        assert abs(a[1] - a[0]) <= 2e-5


# --- R2, no transient_unexplained for a 1:1 angle --------------------------------------------------------------------


def test_transient_unexplained_is_not_a_cap_of_a_one_to_one_angle():
    geometry = make_geometry(runs=[])
    angle = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', None)
    energy = SimpleNamespace(mechanism=SeparatrixMechanism.UNEXPLAINED, episodes_in_separatrix_zone=False)
    assert decide._episode_caps(angle, geometry, energy, S) == [str(Cap.TRANSIENT_UNEXPLAINED)]
    assert decide._episode_caps(angle, geometry, energy, S, one_to_one=True) == []
    kept = replace(S, mmr_coorbital_lifts_unexplained=False)
    assert decide._episode_caps(angle, geometry, energy, kept, one_to_one=True) == [str(Cap.TRANSIENT_UNEXPLAINED)]


@pytest.mark.parametrize(
    'example, expected, unexplained',
    [
        ('coorbital__2013_BS45__1E-1+0+0', ('transient', 'high'), False),
        ('mmr2__33128__6N-7+0+1', ('transient', 'high'), False),
        ('mmr2__471155__6N-7+0+1', ('transient', 'medium'), True),
    ],
)
def test_benchmark_transient_unexplained_only_off_the_one_to_one(example, expected, unexplained):
    """R2 (author 2026-10-19): 2013 BS45 was medium by transient_unexplained alone; the non-1:1 rows keep theirs."""
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    v = analyse_mmr(io.load_series(example), S).verdict
    assert (str(v.status), str(v.confidence)) == expected
    assert ('transient_unexplained' in v.caps) == unexplained


# --- secular / ZLK -------------------------------------------------------------------------------------------------


def test_the_mmr_rules_are_off_for_secular_and_zlk_angles():
    """The shared geometry applies the W20 rules only when the MMR preprocessing sets them (`geometry.FilteredAngle`):
    the defaults are off, so a secular or ZLK angle (built without them) keeps its geometry."""
    t, sigma, slow = one_cycle()
    plain = filtered(t, sigma, 20.0, slow)
    assert rules_of(plain) == {
        'orbit_period_yr': pytest.approx(float('nan'), nan_ok=True),
        'edge_arc_rule': False,
        'return_closure': False,
        'split_regimes': False,
        'cycle_travel': False,
        'coorbital': False,
        'floor_well': None,  # W20 CS5: the quasi-satellite well of the G1q pass
    }
    geometry = g.measure_geometry(plain, ONE_DEG, S)
    assert not geometry.coorbital and not geometry.cycle_travel and geometry.averaged_raw is None
    for event in (geometry.exit, geometry.entry):
        assert event is None or (not event.observed and math.isnan(event.raw_arc_rad))
    # the angle at the centre scale keeps the caller's rules (off here)
    coarse = g.angle_at_centre_window(plain, 200.0, S)
    assert coarse is not None and not any(value for key, value in rules_of(coarse).items() if key != 'orbit_period_yr')


def rules_of(angle) -> dict:
    """The MMR rule fields of a `FilteredAngle` (set only by the MMR preprocessing)."""
    names = ('orbit_period_yr', 'edge_arc_rule', 'return_closure', 'split_regimes', 'coorbital', 'cycle_travel', 'floor_well')
    return {name: getattr(angle, name) for name in names}
