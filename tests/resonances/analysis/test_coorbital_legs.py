"""One leg of a 1:1 angle between two passages, leaf G4c, and the structural near-episode (W28 C6; cards g02-4 A,
g02-5 B, g02-6 B, g02-7 A, g01-1 A, g01-4 A, g01-5 A, g01-8 A; q-4 A).

The measurement is checked on the averaged circular coplanar problem of the Earth, integrated here from
theta_ddot = -3 mu n^2 R~'(theta) (M&D eqs. 3.178-3.179, softened as the package's table) with
a = a_P (1 - (2/3) theta_dot / n): a horseshoe is a reflection at each turn by construction (out/in = 1, the planet's
own force, K / K_theory = 1, energy conserved), and a kick is put in by hand (at a turn the energy is set to the loop
of another turning angle: out/in far from 1). The decision tests build real geometries (`builders`) holding the
legs, their numbers set by hand."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import coorbital as co
from resonances.resonance.analysis.mmr import coorbital_geometry as cg
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr.context import build_context
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, Flag, Reason, Status
from tests.resonances.analysis.builders import make_centre, make_episode, make_geometry
from tests.resonances.analysis.test_coorbital_floor import closed_r, earth_series
from tests.resonances.analysis.test_impl_w2_zone_rules_decide import SAMPLING

SETTINGS = AnalysisSettings()
FLOOR = 111.0  # the Earth's tadpole floor, 0.5 P / sqrt(27 mu / 4) [yr]
STEP = 0.05


def earth_problem():
    """(mu, n, eps) of the Earth as the package's context has them, the softening of its table."""
    t0 = np.linspace(0.0, 10.0, 101)
    context = build_context(earth_series(t0, np.full(t0.size, np.pi)), SETTINGS)
    mu, n = context.planet_mass['Earth'], context.planet_mean_motion_rad_yr['Earth']
    return mu, n, co.hill_softening(mu, SETTINGS.mmr_coorbital_softening_hill)


def track(levels_deg, span_yr: float):
    """theta(t), a(t) of the averaged circular problem from L3 towards the planet, and the times of its turns.
    `levels_deg` = (the level of the start, then the level after each turn): the energy of a loop turning at that
    angle. With one level the motion is a horseshoe (each turn a reflection). With more, a turn is a kick: at the k-th
    turn the energy is set to the next level, the body leaving the turn with |a - a_P| of that loop (a kick that
    removes energy is made by stopping the body where the lower level turns): out/in is then far from 1, larger for a
    level nearer the planet."""
    mu, n, eps = earth_problem()

    def rhs(_, y):
        d2 = 4 * np.sin(y[0] / 2) ** 2 + eps**2
        return [y[1], -3 * mu * n**2 * np.sin(y[0]) * (1.0 - d2**-1.5)]

    def kinetic(level_deg, theta):  # theta_dot^2 / 2 on the loop of that level
        return 3 * mu * n**2 * (closed_r(np.radians(level_deg), eps) - closed_r(theta, eps))

    def turn(_, y):
        return y[1]

    def reaches(level_deg):  # the body reaches the place where the loop of level_deg turns (on its side of L3)
        def event(_, y):
            return np.mod(y[0], 2 * np.pi) - np.pi - np.sign(np.mod(y[0], 2 * np.pi) - np.pi) * (np.pi - np.radians(level_deg))

        return event

    levels = list(levels_deg)
    state, start = [np.pi, -np.sqrt(2 * kinetic(levels[0], np.pi))], 0.0
    times, thetas, rates, turns = [], [], [], []
    for k in range(len(levels)):
        stopping = k + 1 < len(levels) and levels[k + 1] > levels[k]  # the next loop turns farther out: stop the body there
        events = [reaches(levels[k + 1]) if stopping else turn]
        run = solve_ivp(rhs, (start, span_yr), state, events=events, rtol=1e-11, atol=1e-13, method='DOP853', dense_output=True)
        found = [x for x in run.t_events[0] if x > start + 1e-6]
        stop = found[0] if k + 1 < len(levels) and found else span_yr
        grid = np.arange(np.ceil(start / STEP) * STEP, stop, STEP)
        y = run.sol(grid)
        times.append(grid)
        thetas.append(y[0])
        rates.append(y[1])
        turns += [x for x in found if x <= stop] if k + 1 < len(levels) else [x for x in found]
        if stop >= span_yr:
            break
        theta_k = float(run.sol(stop)[0])
        away = 1.0 if np.mod(theta_k, 2 * np.pi) < np.pi else -1.0  # back towards L3
        speed = np.sqrt(max(2 * kinetic(levels[k + 1], theta_k), 0.0))
        state, start = [theta_k, away * speed], stop
    t, theta, rate = (np.concatenate(x) for x in (times, thetas, rates))
    a_planet = (co.GM_SUN / n**2) ** (1.0 / 3.0)
    return t, theta, a_planet * (1.0 - (2.0 / 3.0) * rate / n), np.unique(np.round(turns, 6))


def legs_of(t, theta, a, outside=()):
    """The package's single legs of the synthetic record (sigma = -theta, a body of a 1-yr orbit)."""
    series = earth_series(t, theta, a)
    context = build_context(series, SETTINGS)
    wells = cg.coorbital_wells(series, context, SETTINGS)
    return wells.single_legs(t, -theta, 1.0, FLOOR, 1.0 - SETTINGS.confidence_thin_band, outside)


@pytest.fixture(scope='module')
def horseshoe():
    """A horseshoe turning 20 deg from the Earth (T_hs = 471 yr) over 500 yr: start leg, one whole leg, end leg."""
    return track([20.0], 500.0)


def interior(legs):
    found = [leg for leg in legs if not leg.edge]
    assert len(found) == 1
    return found[0]


# --- the measurement ------------------------------------------------------------------------------------------------


def test_a_horseshoe_leg_reflects_at_both_turns_and_follows_the_planets_law(horseshoe):
    """The leg between the two turns of the integrated horseshoe: crosses L3 only, both turns at the planet mirror
    |a - a_P| (out/in 1) at a level the potential reverses (zone <= 1), the local law is the Earth's (K / K_theory 1)
    and conserves the energy: leaf G4c's transient."""
    t, theta, a, turns = horseshoe
    leg = interior(legs_of(t, theta, a))
    assert (leg.t0_yr, leg.t1_yr) == pytest.approx(tuple(turns[:2]), abs=2.0)
    assert leg.topology == g.LEG_HORSESHOE and leg.clean
    for turn in leg.turns:
        assert turn.site == cg.SITE_PLANET
        assert turn.out_in == pytest.approx(1.0, abs=0.03)
        assert turn.zone <= 1.0
    assert leg.force == 'pass' and leg.force_scale == pytest.approx(1.0, abs=0.1) and leg.energy == 'pass'
    assert leg.period_over_floor == pytest.approx(2 * (turns[1] - turns[0]) / FLOOR, rel=0.02)
    assert decide.leg_outcome(leg, SETTINGS) == (decide.LEG_TRANSIENT, '')


def test_the_record_edges_give_edge_legs_and_an_accepted_run_hides_its_legs(horseshoe):
    """The stretches before the first and after the last turn are edge legs (one turn each); a leg inside an
    accepted run is never read (its run is a closed cycle: rule 7 of the W28 C1 study)."""
    t, theta, a, turns = horseshoe
    legs = legs_of(t, theta, a)
    assert [leg.edge for leg in legs] == [g.LEG_EDGE_START, '', g.LEG_EDGE_END]
    assert legs[0].entry is None and legs[0].exit is not None and legs[-1].exit is None
    hidden = legs_of(t, theta, a, outside=[(turns[0] + 1.0, turns[1] - 1.0)])
    assert [leg.edge for leg in hidden] == [g.LEG_EDGE_START, g.LEG_EDGE_END]


def test_two_kicks_scatter_at_both_turns():
    """The body arrives on the loop that turns 5 deg from the Earth, is stopped at 20 deg (energy taken away) and
    leaves its next turn on the 5-deg loop again (energy given): |a - a_P| after / before is < 1 at the first turn and
    > 1 at the second, both beyond the reflection band of MMR_COORBITAL_TURN_ENERGY_TOL, so neither turn is a
    reflection: a near miss of a capture, no capture (1991 VG: 0.61 and 1.59)."""
    t, theta, a, turns = track([5.0, 20.0, 5.0], 500.0)
    leg = next(leg for leg in legs_of(t, theta, a) if not leg.edge and leg.t0_yr == pytest.approx(turns[0], abs=3.0))
    band = np.exp(SETTINGS.mmr_coorbital_turn_energy_tol)
    assert leg.entry.out_in < 1 / band and leg.exit.out_in > band
    assert decide.leg_outcome(leg, SETTINGS) == (decide.LEG_SCATTER, decide.LEG_TURN)


def test_crosses_planet_and_l3():
    """A swing that passes both 0 and pi (mod 2 pi) is a lap past the planet; one that passes one of them is not."""
    assert decide.short_runs is g.short_runs
    assert g.crosses_planet_and_l3(np.radians(-10.0), np.radians(200.0))
    assert not g.crosses_planet_and_l3(np.radians(10.0), np.radians(350.0))  # a horseshoe: L3 only
    assert not g.crosses_planet_and_l3(np.radians(-30.0), np.radians(30.0))  # a quasi-satellite: the planet only
    assert g.crosses_planet_and_l3(np.radians(370.0), np.radians(170.0))


# --- the decision ---------------------------------------------------------------------------------------------------


def turn(t_yr, out_in=1.0, zone=0.8, site=cg.SITE_PLANET):
    return g.LegTurn(t_yr, site, out_in, zone)


def leg(t0=500.0, t1=580.0, topology=g.LEG_HORSESHOE, edge='', entry=1.0, exit_=1.0, force='pass', scale=1.0, energy='pass'):
    turns = (None if edge == g.LEG_EDGE_START else turn(t0, entry), None if edge == g.LEG_EDGE_END else turn(t1, exit_))
    return g.SingleLeg(t0, t1, edge, topology, 2 * (t1 - t0) / FLOOR, *turns, force=force, force_scale=scale, energy=energy)


def decide_legs(*legs, runs=(), settings=SETTINGS):
    """The status of a 1:1 circulation (G4: the centre passes 2 pi) whose geometry holds `legs`."""
    geometry = make_geometry(
        coorbital=True,
        period_floor_yr=FLOOR,
        period_floor_setting='mmr_coorbital_period_floor_factor',
        single_legs=tuple(legs),
        runs=list(runs),
        window=(0.0, 1000.0),
        centre=make_centre(max_pass_rad=10.0),
    )
    ledger = Ledger.from_settings(settings)
    angle = decide.decide_angle(SAMPLING, False, geometry, None, settings, ledger)
    assert angle.angle_status == AngleStatus.CIRCULATION
    decision = decide.decide_status(angle, SAMPLING, geometry, None, None, None, None, None, None, settings, ledger)
    return decision, decide.reported_angle(angle, geometry, settings, decision)


def test_one_clean_conservative_leg_is_a_transient_of_one_leg():
    """A clean horseshoe leg, its law and energy pass, both turns reflect: transient, origin dynamic, the word
    episodic, capped at medium by one_coorbital_leg (2023 OE5)."""
    decision, reported = decide_legs(leg())
    assert (decision.status, decision.reason, decision.rule) == (Status.TRANSIENT, Reason.DYNAMIC_EPISODE, 'G4c')
    assert reported.angle_status == AngleStatus.EPISODIC
    assert 'one_coorbital_leg' in decision.caps and Rule.COORBITAL_LEG in decision.rules_fired


@pytest.mark.parametrize(
    'changed, discrepancy',
    [
        ({'topology': g.LEG_PASSING}, decide.LEG_NOT_CLEAN),  # 2018 KS: a lap past the Earth
        ({'force': 'fail'}, decide.LEG_FORCE),
        ({'scale': -1.0}, decide.LEG_FORCE),  # a pushed the wrong way
        ({'energy': 'fail'}, decide.LEG_ENERGY),  # 2024 AV2: energy R^2 0.09
        ({'exit_': 1.6}, decide.LEG_TURN),  # one kick: |ln 1.6| = 0.47 > 0.34
    ],
)
def test_one_discrepancy_makes_the_leg_uncertain(changed, discrepancy):
    """q-4 A: any one test of the clean leg failing -> uncertain/one_coorbital_leg, the word episodic, action review."""
    one = leg(**changed)
    assert decide.leg_outcome(one, SETTINGS) == (decide.LEG_UNCERTAIN, discrepancy)
    decision, reported = decide_legs(one)
    assert (decision.status, decision.reason, decision.origin) == (Status.UNCERTAIN, Reason.ONE_COORBITAL_LEG, 'unknown')
    assert reported.angle_status == AngleStatus.EPISODIC


def test_a_leg_cut_by_the_record_end_asks_for_a_longer_record():
    """A clean leg from its turn to the record's end whose turn reflects: uncertain/insufficient_time (extend)."""
    decision, reported = decide_legs(leg(t0=930.0, t1=1000.0, edge=g.LEG_EDGE_END))
    assert (decision.status, decision.reason) == (Status.UNCERTAIN, Reason.BASELINE_NEEDED)
    assert reported.angle_status == AngleStatus.INSUFFICIENT_TIME


def test_two_kicks_give_a_near_episode_and_no_leg_keeps_the_circulation():
    """Both turns scatter (out/in 0.5 and 2): non_resonant/circulation with the near-episode cap (1991 VG). No leg at
    all (one reflection between passages, 2008 EZ84): the circulation without a cap."""
    decision, reported = decide_legs(leg(entry=0.5, exit_=2.0))
    assert (decision.status, decision.reason) == (Status.NON_RESONANT, Reason.CIRCULATION)
    assert str(Flag.NEAR_EPISODE) in decision.flags and 'near_episode' in decision.caps
    assert reported.angle_status == AngleStatus.CIRCULATION
    decision, _ = decide_legs()
    assert (decision.status, decision.caps, decision.flags) == (Status.NON_RESONANT, [], [])


def test_the_strongest_leg_decides_and_the_force_gate_reads_a_failed_law_as_scattering():
    """A scattering leg and a transient one: transient. The C1 force gate (off by default, q-4 A literally) reads a
    clean leg whose law fails as a scattering (near-episode) instead of uncertain."""
    assert decide_legs(leg(t0=100.0, t1=180.0, entry=0.5, exit_=2.0), leg())[0].status == Status.TRANSIENT
    gate = AnalysisSettings(mmr_coorbital_leg_force_gate=True)
    decision, _ = decide_legs(leg(force='fail'), settings=gate)
    assert decision.status == Status.NON_RESONANT and 'near_episode' in decision.caps
    assert decide_legs(leg(force='fail'))[0].status == Status.UNCERTAIN


def test_an_episodic_angle_and_the_switch_off_leave_the_leaf_out():
    """Three reflections make an episode (G2): the leaf is never reached; with MMR_COORBITAL_SINGLE_LEG off a 1:1
    circulation keeps the circulation leaf."""
    geometry = make_geometry(coorbital=True, single_legs=(leg(),))
    episodic = decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)
    assert decide._g4c(episodic, geometry, SETTINGS, Ledger.from_settings(SETTINGS)) is None
    off = AnalysisSettings(mmr_coorbital_single_leg=False)
    assert decide_legs(leg(), settings=off)[0].status == Status.NON_RESONANT


# --- the structural near-episode (g01-8 A) ----------------------------------------------------------------------------


@pytest.mark.parametrize('reason', [g.RejectReason.TOO_FEW_TP, g.RejectReason.OVER_THE_TOP])
def test_a_two_tp_run_at_0999_of_the_floor_is_a_near_episode_with_or_without_an_over_the_top_neighbour(reason):
    """A 2-TP horseshoe half-swing (10 -> 350 deg: L3 only) of period 0.999 x floor: a near-episode whether its
    neighbour went over the top (reject reason over_the_top) or not (too_few_tp); the rule coorbital_near_episode
    fires only for the first. A lap past the planet (-10 -> 200 deg) rejected over_the_top is none."""
    half = 0.999 * FLOOR / 2
    values = np.radians([10.0, 350.0])
    run = make_episode(tp_times_yr=np.array([300.0, 300.0 + half]), tp_values_rad=values, accepted=False, reject_reason=reason)
    run.half_amp_max_rad = float(abs(values[1] - values[0]) / 2)
    off = AnalysisSettings(mmr_coorbital_single_leg=False)
    decision, _ = decide_legs(runs=[run], settings=off)
    assert 'near_episode' in decision.caps
    assert (Rule.COORBITAL_NEAR_EPISODE in decision.rules_fired) == (reason == g.RejectReason.OVER_THE_TOP)
    lap = make_episode(
        tp_times_yr=run.tp_times_yr, tp_values_rad=np.radians([-10.0, 200.0]), accepted=False, reject_reason=g.RejectReason.OVER_THE_TOP
    )
    lap.half_amp_max_rad = float(np.radians(105.0))
    assert decide_legs(runs=[lap], settings=off)[0].caps == []


# --- one half-swing of an angle that is not 1:1 (leaf G4h, q-5 i) -----------------------------------------------------


def pendulum_swing(amplitude_deg=120.0, c=-0.05, k=4e-4, span_yr=2000.0, step=1.0):
    """A pendulum sigma_ddot = -k sin sigma (k [rad yr^-2]) of the given half-amplitude, and a from the identity
    sigma_dot = c (a - a*): a_dot = sigma_ddot / c = -(k / c) sin sigma, the one-harmonic law exactly."""
    run = solve_ivp(
        lambda _, y: [y[1], -k * np.sin(y[0])],
        (0.0, span_yr),
        [np.radians(amplitude_deg), 0.0],
        max_step=step / 4,
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )
    t = np.arange(0.0, span_yr, step)
    sigma, rate = run.sol(t)
    return t, sigma, 1.0 + rate / c, c


def test_a_pendulum_half_swing_follows_the_one_harmonic_law_and_a_slow_turn_does_not():
    """Between two turning points of a pendulum (half-amplitude 120 deg) the one-harmonic law fits each half of the
    window (R^2 ~ 1) and its force turns sigma back at both turns. The same swing of sigma with a that does not answer
    it (a flat a with a small independent scatter: a kinematic swing) follows no law (R^2 << MMR_FORCE_R2_FAIL)."""
    t, sigma, a, c = pendulum_swing()
    turns = t[1:-1][(np.diff(np.sign(np.diff(sigma))) != 0)]
    t0, t1 = float(turns[0]), float(turns[1])
    lo, hi = sorted((float(np.interp(t0, t, sigma)), float(np.interp(t1, t, sigma))))
    factor = SETTINGS.mmr_half_libration_window_factor
    r2_h1, r2_h2, restoring = g.harmonic_half_test(t, sigma, a, c, t0, t1, lo, hi, factor)
    assert r2_h1 > 0.99 and r2_h2 > 0.99 and restoring
    flat = 1.0 + 1e-6 * np.random.default_rng(1).standard_normal(len(t))
    r2_h1, r2_h2, _ = g.harmonic_half_test(t, sigma, flat, c, t0, t1, lo, hi, factor)
    assert max(r2_h1, r2_h2) < SETTINGS.mmr_force_r2_fail


def decide_half(*halves, settings=SETTINGS):
    """The status of a circulation (G4) of an angle that is not 1:1 whose geometry holds `halves`."""
    geometry = make_geometry(half_librations=tuple(halves), window=(0.0, 1e5), centre=make_centre(max_pass_rad=10.0))
    ledger = Ledger.from_settings(settings)
    angle = decide.decide_angle(SAMPLING, False, geometry, None, settings, ledger)
    decision = decide.decide_status(angle, SAMPLING, geometry, None, None, None, None, None, None, settings, ledger)
    return decision, decide.reported_angle(angle, geometry, settings, decision)


def test_a_half_swing_with_the_law_is_uncertain_and_without_it_the_circulation_stays():
    """q-5 i: between two passages -> uncertain/one_half_libration (review); at the record's edge -> extend (42946
    43S-6: R^2 0.72/0.62); without the law (R^2 0.08/0.06, 15874 5N-23) or without the restoring force: circulation."""
    between = g.HalfLibration(40000.0, 47000.0, '', 2.0, 3.0, 0.72, 0.62, True)
    decision, reported = decide_half(between)
    assert (decision.status, decision.reason, decision.rule) == (Status.UNCERTAIN, Reason.ONE_HALF_LIBRATION, 'G4h')
    assert reported.angle_status == AngleStatus.EPISODIC and Rule.SINGLE_HALF_LIBRATION in decision.rules_fired
    at_end = g.HalfLibration(28476.0, 32285.0, g.LEG_EDGE_END, 2.25, 56.0, 0.72, 0.62, True)
    decision, reported = decide_half(at_end)
    assert (decision.status, decision.reason, reported.angle_status) == (
        Status.UNCERTAIN,
        Reason.BASELINE_NEEDED,
        AngleStatus.INSUFFICIENT_TIME,
    )
    for weak in (
        g.HalfLibration(68540.0, 78541.0, '', 2.6, 148.0, 0.08, 0.06, True),
        g.HalfLibration(1.0, 2.0, '', 1.0, 3.0, 0.9, 0.9, False),
    ):
        assert decide_half(weak)[0].status == Status.NON_RESONANT
    assert decide_half(between, settings=AnalysisSettings(mmr_single_half_libration=False))[0].status == Status.NON_RESONANT
