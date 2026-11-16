"""Rule coorbital_qs_run_floor (G1q pass): a quasi-satellite run has the floor of the loop through its own turns.

The QS floor is 0.5 x the small-amplitude P_QS, the period at the bottom of the well. A low-e QS well softens towards
its barrier (the period falls with the amplitude), so P(0) is no lower bound: 54509 YORP's 11-TP QS run of 50 yr was
below its 55.7-yr QS floor. In the G1q pass a run whose turns all lie in the well takes min(QS floor, factor x its own
loop), and the libration image of the TP prominence reads the same floor.

The truth comes from the independent averaged ODE theta'' = -3 mu n^2 F~(theta) with F~ from the brute-force average of
the W4 tests (an e = 0.2 planar body against the J2000 Earth), not from the package's table; the small-amplitude limit
from omega^2 = 3 mu n^2 F~'(0). The decision tests build real geometries (`builders`).
"""

import math
from functools import lru_cache

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import coorbital_geometry as cg
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.context import build_context
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import AnalysisSettings, settings_from_config
from tests.resonances.analysis.builders import make_episode, make_geometry
from tests.resonances.analysis.test_quasi_satellite import _series, _track
from tests.resonances.independent_w4.test_status_coorbital import MU_EARTH, N_EARTH, _qs_force_spline

S = settings_from_config()
OFF = settings_from_config(mmr_coorbital_qs_run_floor=False)
FACTOR = S.mmr_coorbital_period_floor_factor
SETTING = 'mmr_coorbital_period_floor_factor'
QS_FACTOR = S.mmr_coorbital_qs_loop_factor
QS_SETTING = 'mmr_coorbital_qs_loop_factor'
STEP_YR = 0.05


def _ode_period(amplitude_deg: float) -> float:
    """The period of the truth's QS loop started at rest at +amplitude: the mean spacing of its downward crossings of 0
    over the 400 yr of `_loop_record`."""
    t, theta, _, _, _ = _loop_record(amplitude_deg)
    down = t[1:][(theta[:-1] > 0) & (theta[1:] <= 0)]
    return float(np.mean(np.diff(down)))


def _small_amplitude_period() -> float:
    """2 pi / omega, omega^2 = 3 mu n^2 F~'(0) (the A -> 0 limit; F~ = dR~/dtheta)."""
    force, h = _qs_force_spline(), 1e-4
    return 2 * math.pi / math.sqrt(3 * MU_EARTH * N_EARTH**2 * (force(h) - force(-h)) / (2 * h))


@lru_cache(maxsize=4)
def _loop_record(amplitude_deg: float, span_yr: float = 400.0):
    """t, theta, theta_dot of the truth's QS loop at rest at +amplitude, and its turning points (times, theta)."""
    theta, rate = _track(math.radians(amplitude_deg), 0.0, 0.0, span_yr, STEP_YR, _qs_force_spline())
    t = np.arange(0.0, span_yr + STEP_YR / 2, STEP_YR)[: len(theta)]
    turns = np.flatnonzero(np.diff(np.sign(rate)) != 0) + 1
    return t, theta, rate, t[turns], theta[turns]


def _reading(t, theta, rate, turn_t, turn_theta, settings=S, qs_pass=True, a_sign=1.0):
    """The package's `CoorbitalRunFloor` reading of the run of these turns (sigma = -theta of the 1E-1 record)."""
    if a_sign < 0:  # a pushed the wrong way: theta_dot -> -theta_dot in a only
        rate = -rate
    series = _series(theta, rate, t, 'qs-run')
    context = build_context(series, settings)
    sigma = -theta
    model = cg.coorbital_run_floor(series, context, t, sigma, STEP_YR, 10.0, settings, qs_pass=qs_pass)
    return model.read(turn_t, -turn_theta)


# --- the loop period of a softening QS well, from two sources ---------------------------------------------------------


@pytest.mark.slow
def test_the_low_e_quasi_satellite_well_softens_towards_its_barrier():
    """The truth: P(2 deg) is the small-amplitude 2 pi / omega (0.5 %); P falls with the amplitude and P(22 deg), 1.2
    deg inside the top at 23.2 deg, is below half of P(0), so 0.5 x P(0) is no lower bound of the well's loops."""
    p0 = _small_amplitude_period()
    assert _ode_period(2.0) == pytest.approx(p0, rel=5e-3)
    periods = [_ode_period(a) for a in (10.0, 20.0, 22.0)]
    assert periods[0] > periods[1] > periods[2]
    assert periods[2] / p0 < 0.5


@pytest.mark.parametrize('amplitude_deg', [10.0, 20.0])
def test_the_package_loop_through_qs_turns_is_the_truth(amplitude_deg):
    """`loop_period` through the turns of the truth's loop (the package's own table of the record's elements) gives the
    truth's period within 3 % (the J2000 Earth and the table's softening against the brute force)."""
    t, theta, _, turn_t, turn_theta = _loop_record(amplitude_deg)
    series = _series(theta, np.zeros_like(t), t, 'qs-loop')
    profile = cg.series_profile(series, build_context(series, S), float(t[0]), float(t[-1]), S)
    assert cg.loop_period(profile, turn_theta[:2]) == pytest.approx(_ode_period(amplitude_deg), rel=0.03)
    assert {cg.turn_site(profile, x) for x in turn_theta} == {cg.SITE_QS}


# --- the reading of one run ---------------------------------------------------------------------------------------


def test_a_run_of_qs_turns_in_the_pass_has_the_floor_of_its_own_loop():
    """All turns in the QS well, the model of the G1q pass: source qs_loop, floor = MMR_COORBITAL_QS_LOOP_FACTOR x the
    median loop of its pairs (here every pair has the same loop: the truth's period within 3 %), the planet's law
    (K / K_theory ~ 1)."""
    t, theta, rate, turn_t, turn_theta = _loop_record(20.0)
    floor, setting, source, columns = _reading(t, theta, rate, turn_t[:7], turn_theta[:7])
    assert source == cg.FLOOR_SOURCE_QS_LOOP and setting == QS_SETTING
    assert columns['cpl_turn_sites'] == ';'.join([cg.SITE_QS] * 7)
    assert floor == pytest.approx(QS_FACTOR * columns['cpl_loop_period_yr'])
    assert columns['cpl_loop_period_yr'] == pytest.approx(_ode_period(20.0), rel=0.03)
    assert columns['cpl_law_K_over_K_theory'] == pytest.approx(1.0, abs=0.15) and columns['cpl_law_R2'] > 0.9


@pytest.mark.parametrize('qs_floor_yr', [20.0, 50.0])
def test_a_qs_loop_floor_that_cannot_act_is_moot(qs_floor_yr):
    """The 20-deg loop (53 yr, own floor about 26.6 yr) in a pass whose QS floor is 20 yr (the own floor is not below
    it) or 50 yr (the run is not faster than it): the QS floor decides either way, so no law is fitted and the source
    is qs_loop_moot; the loop is still reported."""
    t, theta, rate, turn_t, turn_theta = _loop_record(20.0)
    series = _series(theta, rate, t, 'qs-run')
    model = cg.coorbital_run_floor(series, build_context(series, S), t, -theta, STEP_YR, 10.0, S, qs_pass=True, qs_floor_yr=qs_floor_yr)
    floor, _, source, columns = model.read(turn_t[:7], -turn_theta[:7])
    assert source == cg.FLOOR_SOURCE_QS_MOOT and np.isnan(floor) and np.isnan(columns['cpl_law_K_over_K_theory'])
    assert columns['cpl_loop_period_yr'] == pytest.approx(_ode_period(20.0), rel=0.03)


def test_a_moot_reading_decides_as_the_full_one():
    """The moot shortcut skips the law fit only where the full reading cannot change the decision: for the 20-deg loop
    (53 yr) at a QS floor of 20 or 50 yr, the run passes min(QS floor, full floor) exactly when it passes the QS floor
    (the full floor read with no QS floor given, `qs_floor_yr` nan). The moot period is the decision's: the median of
    t_i+2 - t_i of the same TPs (`Episode.period_med_yr`, `geometry.full_periods`)."""
    t, theta, rate, turn_t, turn_theta = _loop_record(20.0)
    series = _series(theta, rate, t, 'qs-run')
    full = cg.coorbital_run_floor(series, build_context(series, S), t, -theta, STEP_YR, 10.0, S, qs_pass=True)
    floor, _, source, _ = full.read(turn_t[:7], -turn_theta[:7])
    assert source == cg.FLOOR_SOURCE_QS_LOOP
    kinds = np.where(np.arange(7) % 2 == 0, 1, -1)
    period = float(np.median(g.full_periods(g.TurningPoints(np.arange(7), turn_t[:7], -turn_theta[:7], kinds, 0.0))))
    assert period == pytest.approx(_ode_period(20.0), rel=0.01)
    for qs_floor in (20.0, 50.0):
        assert (period >= min(qs_floor, floor)) == (period >= qs_floor)


def test_the_qs_loop_floor_is_not_below_the_gain_fail_period():
    """The QS loop floor is bounded below by the period the pass filter keeps at gain MMR_L8_GAIN_FAIL: for the
    analog Butterworth of order N, G(P) = 1 / (1 + (P_c / P)^2N) = 0.5 at P = P_c (independently: G(P_c) = 1/2 for any
    N); at a 60-yr cutoff a 20-deg loop of 53 yr (0.7 x 53 = 37 yr) gets the floor 60 yr from that bound."""
    assert cg.qs_gain_floor_yr(27.83, S) == pytest.approx(27.83)
    assert cg.qs_gain_floor_yr(27.83, settings_from_config(mmr_l8_gain_fail=0.8)) == pytest.approx(27.83 * 4**0.25)
    t, theta, rate, turn_t, turn_theta = _loop_record(20.0)
    series = _series(theta, rate, t, 'qs-run')
    model = cg.coorbital_run_floor(series, build_context(series, S), t, -theta, STEP_YR, 60.0, S, qs_pass=True)
    assert model.qs_loop_floor(53.0) == (pytest.approx(60.0), 'mmr_l8_gain_fail')
    assert model.qs_loop_floor(100.0) == (pytest.approx(70.0), QS_SETTING)


def test_a_qs_run_outside_the_pass_well_is_no_qs_loop_and_the_image_reads_qs_loops_only():
    """With the pass's FloorWell given, a run of QS turns that the well does not hold (a well about 180 deg) reads
    not_reflection, and `floor_of` (the libration image's floor) is nan for it; inside a well about 0 it is the
    qs_loop floor (review 1: the image and the decision read the same floor of every run)."""
    t, theta, rate, turn_t, turn_theta = _loop_record(20.0)
    series = _series(theta, rate, t, 'qs-run')
    context = build_context(series, S)
    grid = np.linspace(-1.0, 401.0, 101)
    for centre, expected in ((0.0, cg.FLOOR_SOURCE_QS_LOOP), (math.pi, cg.FLOOR_SOURCE_NOT_REFLECTION)):
        well = g.FloorWell(grid, np.full_like(grid, centre), np.full_like(grid, 0.6), 111.0, SETTING)
        model = cg.coorbital_run_floor(series, context, t, -theta, STEP_YR, 10.0, S, qs_pass=True, floor_well=well)
        floor, _, source, _ = model.read(turn_t[:7], -turn_theta[:7])
        assert source == expected
        image = model.floor_of(turn_t[:7], -turn_theta[:7])
        assert (image == pytest.approx(floor)) if expected == cg.FLOOR_SOURCE_QS_LOOP else np.isnan(image)


def test_the_image_reads_no_profile_loop_floor():
    """A reflection run (profile_loop) keeps P_floor in the image of the G1q pass: `floor_of` gives nan, as
    `decide.period_floor` reads no profile_loop floor inside the well."""
    t, theta, rate, turn_t, turn_theta = _loop_record(20.0)
    series = _series(theta, rate, t, 'qs-run')
    model = cg.coorbital_run_floor(series, build_context(series, S), t, -theta, STEP_YR, 10.0, S, qs_pass=True)
    model.read = lambda times, values: (60.0, SETTING, cg.FLOOR_SOURCE_LOOP, {})
    assert np.isnan(model.floor_of(turn_t[:5], -turn_theta[:5]))


def test_outside_the_pass_or_switched_off_a_qs_run_keeps_the_angle_floor():
    """The first pass (qs_pass False) and MMR_COORBITAL_QS_RUN_FLOOR off read the same turns as before the rule:
    not_reflection, no floor of their own."""
    t, theta, rate, turn_t, turn_theta = _loop_record(20.0)
    for settings, qs_pass in ((S, False), (OFF, True)):
        floor, _, source, _ = _reading(t, theta, rate, turn_t[:5], turn_theta[:5], settings, qs_pass)
        assert source == cg.FLOOR_SOURCE_NOT_REFLECTION and np.isnan(floor)


def test_mixed_qs_and_planet_turns_keep_the_angle_floor():
    """A run whose last turn is a reflection on the planet's top beyond the well (theta 30 deg, the top at 23.2 deg)
    is neither all reflections nor all QS turns: not_reflection."""
    t, theta, rate, turn_t, turn_theta = _loop_record(20.0)
    # after the fourth turn (a minimum near -20 deg) the record swings out to +30 deg in 30 yr and back
    after = t > turn_t[3]
    phase = np.clip((t[after] - turn_t[3]) / 30.0, 0.0, 2.0)
    theta = theta.copy()
    theta[after] = turn_theta[3] + (math.radians(30.0) - turn_theta[3]) * (1 - np.cos(np.pi * phase)) / 2
    rate = np.gradient(theta, t)
    times, values = np.append(turn_t[:4], turn_t[3] + 30.0), np.append(turn_theta[:4], math.radians(30.0))
    floor, _, source, columns = _reading(t, theta, rate, times, values)
    assert source == cg.FLOOR_SOURCE_NOT_REFLECTION and np.isnan(floor)
    assert columns['cpl_turn_sites'].split(';')[-1] == cg.SITE_PLANET


def test_a_qs_run_pushed_the_wrong_way_has_no_floor_of_its_own():
    """The same QS turns with a's swing reversed: the force is not the planet's (K < 0), law_not_positive."""
    t, theta, rate, turn_t, turn_theta = _loop_record(20.0)
    floor, _, source, columns = _reading(t, theta, rate, turn_t[:5], turn_theta[:5], a_sign=-1.0)
    assert source == cg.FLOOR_SOURCE_LAW and np.isnan(floor) and columns['cpl_law_K_over_K_theory'] < 0


# --- the decision -------------------------------------------------------------------------------------------------


def _qs_geometry(episode, own_floor, source=cg.FLOOR_SOURCE_QS_LOOP, qs_floor=45.0, centre=0.0):
    """The G1q pass's geometry: QS floor `qs_floor`, a well about 0 of half-width 0.4 rad (tadpole floor 111 yr outside
    it), and the run's own reading."""
    t = np.linspace(-1.0, 1000.0, 101)
    well = g.FloorWell(t, np.full_like(t, centre), np.full_like(t, 0.4), 111.0, SETTING)
    floors = {g.run_key(episode): g.RunFloor(own_floor, SETTING, source)}
    return make_geometry(
        runs=[episode], period_floor_yr=qs_floor, period_floor_setting=SETTING, run_floors=floors, floor_well=well, coorbital=True
    )


def _qs_episode(period_yr, centre=0.0):
    times = 100.0 + period_yr / 2 * np.arange(9)
    return make_episode(tp_times_yr=times, period_med_yr=period_yr, centre_mod_2pi_rad=centre)


def test_an_in_well_qs_run_takes_the_lower_of_its_own_and_the_qs_floor():
    """YORP c046: P 50.1 yr, QS floor 55.7 yr, own loop floor 25.2 yr -> floor 25.2, a libration; an own floor above
    the QS floor (a hardening well) never raises it; the rule fires as coorbital_qs_run_floor."""
    episode = _qs_episode(50.1)
    lower = _qs_geometry(episode, 25.2, qs_floor=55.7)
    assert decide.period_floor(lower, S, episode) == (25.2, SETTING)
    assert decide.librating_episodes(lower, S) == [episode]
    assert decide.run_floor_acts(lower, S) == (Rule.COORBITAL_QS_RUN_FLOOR,)
    higher = _qs_geometry(episode, 70.0, qs_floor=55.7)
    assert decide.period_floor(higher, S, episode)[0] == 55.7
    assert decide.run_floor_acts(higher, S) == ()


def test_a_profile_loop_floor_in_the_well_keeps_the_qs_floor():
    """Only a qs_loop reading lowers the floor inside the well (`test_coorbital_floor`: the QS well keeps its own)."""
    episode = _qs_episode(50.1)
    geometry = _qs_geometry(episode, 25.2, source=cg.FLOOR_SOURCE_LOOP, qs_floor=55.7)
    assert decide.period_floor(geometry, S, episode)[0] == 55.7
    assert decide.librating_episodes(geometry, S) == []


def test_a_qs_loop_reading_outside_the_well_is_no_floor():
    """A run centred at L4 (60 deg, outside the 0.4-rad well) keeps the tadpole floor, its qs_loop reading is ignored
    by period_floor and by run_floor_acts."""
    episode = _qs_episode(50.1, centre=math.radians(60.0))
    geometry = _qs_geometry(episode, 25.2, qs_floor=55.7)
    assert decide.period_floor(geometry, S, episode)[0] == 111.0
    assert decide.run_floor_acts(geometry, S) == ()


def test_a_reflection_run_outside_the_well_still_fires_the_profile_rule():
    """A horseshoe run (profile_loop 60 yr) of P 96 yr outside the well: rejected at the 111-yr tadpole floor, kept at
    its own; the rule is coorbital_floor_profile."""
    episode = _qs_episode(96.0, centre=math.pi)
    geometry = _qs_geometry(episode, 60.0, source=cg.FLOOR_SOURCE_LOOP)
    assert decide.period_floor(geometry, S, episode)[0] == 60.0
    assert decide.run_floor_acts(geometry, S) == (Rule.COORBITAL_FLOOR_PROFILE,)


def test_why_a_qs_pass_was_rejected():
    """`qs_rejection` on the G1q pass's geometry (well about 0, QS floor 55.7 yr; c046 without the rule): an in-well
    accepted run of P 50.1 yr whose own reading is the QS floor (moot: nan floor) is below_period_floor with its
    numbers; an in-well found episode is keeps_outside; a run the geometry rejected is not_accepted:<reason>; a run
    centred at L4 only is no_run_in_well; no run of >= 2 TPs is no_turning_points."""
    from resonances.resonance.analysis.mmr.analyse import qs_rejection

    episode = _qs_episode(50.1)
    geometry = _qs_geometry(episode, float('nan'), source=cg.FLOOR_SOURCE_QS_MOOT, qs_floor=55.7)
    well = geometry.floor_well
    columns = qs_rejection(geometry, well, [], S)
    assert columns['geo_qs_reject'] == 'below_period_floor'
    assert columns['geo_qs_run_n_tp'] == 9 and columns['geo_qs_run_period_yr'] == 50.1
    assert columns['geo_qs_run_floor_yr'] == 55.7 and columns['geo_qs_run_floor_source'] == ''
    assert qs_rejection(geometry, well, [episode], S)['geo_qs_reject'] == 'keeps_outside'
    rejected = make_episode(
        tp_times_yr=episode.tp_times_yr, centre_mod_2pi_rad=0.0, accepted=False, reject_reason=g.RejectReason.TOO_FEW_TP
    )
    assert qs_rejection(_qs_geometry(rejected, 25.2, qs_floor=55.7), well, [], S)['geo_qs_reject'] == 'not_accepted:too_few_tp'
    outside = _qs_episode(50.1, centre=math.radians(60.0))
    assert qs_rejection(_qs_geometry(outside, 25.2, qs_floor=55.7), well, [], S)['geo_qs_reject'] == 'no_run_in_well'
    empty = make_geometry(runs=[], period_floor_yr=55.7, period_floor_setting=SETTING, floor_well=well, coorbital=True)
    assert qs_rejection(empty, well, [], S)['geo_qs_reject'] == 'no_turning_points'


def test_the_switch_needs_the_profile_floor():
    """MMR_COORBITAL_PROFILE_FLOOR off: no run-floor model at all, so no qs_loop either."""
    t, theta, rate, _, _ = _loop_record(20.0)
    series = _series(theta, rate, t, 'qs-run')
    off = AnalysisSettings(mmr_coorbital_profile_floor=False)
    assert cg.coorbital_run_floor(series, build_context(series, off), t, -theta, STEP_YR, 10.0, off, qs_pass=True) is None


# --- the whole analysis -----------------------------------------------------------------------------------------------


CIRCULATION_RATE = math.radians(-0.75)


def _qs_loop_in_circulation(amplitude_deg=22.0, cycles=8, step=0.25):
    """Circulation to t1 = 300 yr, a kick to rest at +amplitude (a jump of a), `cycles` periods of the truth's loop,
    a kick back to the circulation rate to 1000 yr: a QS loop whose period (40.8 yr at 22 deg) is below the QS floor
    0.5 x P_QS (45.3 yr)."""
    t1 = 300.0
    period = _ode_period(amplitude_deg)
    loop, rate_loop = _track(math.radians(amplitude_deg), 0.0, t1, t1 + cycles * period, step, _qs_force_spline())
    before = np.arange(0.0, t1, step)
    t_loop = t1 + step * np.arange(len(loop))
    after = np.arange(t_loop[-1] + step, 1000.0 + step / 2, step)
    t = np.concatenate([before, t_loop, after])
    theta = np.concatenate(
        [math.radians(amplitude_deg) + CIRCULATION_RATE * (before - t1), loop, loop[-1] + CIRCULATION_RATE * (after - t_loop[-1])]
    )
    rate = np.concatenate([np.full(len(before), CIRCULATION_RATE), rate_loop, np.full(len(after), CIRCULATION_RATE)])
    return t, theta, rate, period


@pytest.mark.slow
def test_a_qs_loop_below_the_qs_floor_is_counted_with_the_rule_and_flagged_without():
    """The YORP case on the truth's record: 8 loops of 40.8 yr (below the QS floor 0.5 x 90.6 yr). Without the rule
    the G1q pass is rejected and the audit flags floor_above_own_loop (the stay's loop is below the QS floor) and the
    uncovered bounded run; with it the in-well episode counts at its own loop floor (about 20 yr), the row is
    transient, the rule fires and neither flag remains."""
    t, theta, rate, period = _qs_loop_in_circulation()
    series = _series(theta, rate, t, 'qs-22')
    off = analyse_mmr(series, OFF, scan_subresonances=False, scan_owner=False)
    assert off.extra_columns['geo_qs_pass'] == 'rejected'
    assert off.extra_columns['flt_qs_floor_yr'] > period
    assert 'floor_above_own_loop' in off.verdict.flags and 'uncovered_bounded_run' in off.verdict.flags
    assert 'floor_above_own_loop' in off.verdict.review_reasons and off.verdict.review_required
    on = analyse_mmr(series, S, scan_subresonances=False, scan_owner=False)
    assert on.extra_columns['geo_qs_pass'] == 'kept' and str(on.verdict.status) == 'transient'
    assert str(Rule.COORBITAL_QS_RUN_FLOOR) in on.verdict.rules_fired
    counted = [r for r in on.episode_rows() if r['counted'] and r['cpl_floor_source'] == cg.FLOOR_SOURCE_QS_LOOP]
    assert len(counted) == 1 and counted[0]['period_med_yr'] == pytest.approx(period, rel=0.05)
    assert counted[0]['cpl_floor_yr'] == pytest.approx(QS_FACTOR * period, rel=0.1)
    assert 300.0 <= counted[0]['t_start_yr'] and counted[0]['t_end_yr'] <= 300.0 + 8 * period + 5.0  # the exit kick's turn
    assert not {'floor_above_own_loop', 'uncovered_bounded_run'} & set(on.verdict.flags)


@pytest.mark.slow
def test_the_audit_changes_no_status_level_or_action():
    """The same record without the rule, the audit on and off: only the flags, review reasons and aud_* columns differ."""
    t, theta, rate, _ = _qs_loop_in_circulation()
    series = _series(theta, rate, t, 'qs-22')
    flagged = analyse_mmr(series, OFF, scan_subresonances=False, scan_owner=False).verdict
    quiet = settings_from_config(mmr_coorbital_qs_run_floor=False, mmr_coorbital_coverage_audit=False)
    silent = analyse_mmr(series, quiet, scan_subresonances=False, scan_owner=False)
    plain = silent.verdict
    fields = ('status', 'angle_status', 'reason', 'confidence', 'action', 'caps', 'libration_origin', 'counted_episode_ids')
    assert {f: getattr(flagged, f) for f in fields} == {f: getattr(plain, f) for f in fields}
    assert set(flagged.flags) - set(plain.flags) == {'floor_above_own_loop', 'uncovered_bounded_run'}
    assert np.isnan(silent.extra_columns['aud_n_bounded_runs'])
