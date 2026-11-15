"""The 1:1 period floor from the loop through a run's own turning points, and the thin band of the near-episode
(W28 C3; cards g02-1 A, g01-6 A, g01-8 A, q-3 A).

The reference periods come from a quad of the period integral of the circular coplanar problem (M&D eqs. 3.178-3.179,
R~ = 1 / (2 |sin(theta/2)|) - cos theta, softened as the package's table), not from `coorbital_geometry.loop_period`;
the horseshoe track is integrated from the averaged equation of motion theta_ddot = -3 mu n^2 R~'(theta) with
a = a_P (1 - (2/3) theta_dot / n). The decision tests build real geometries (`builders`)."""

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import coorbital as co
from resonances.resonance.analysis.mmr import coorbital_geometry as cg
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr.context import build_context
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings
from tests.resonances.analysis.builders import make_episode, make_geometry

SETTINGS = AnalysisSettings()
CIRCULAR = co.Orbit(0.0, 0.0, 0.0, 0.0)
FACTOR = SETTINGS.mmr_coorbital_period_floor_factor


def closed_r(theta, eps):
    return 1.0 / np.sqrt(4 * np.sin(theta / 2) ** 2 + eps**2) - np.cos(theta)


def quad_horseshoe_period(mu, n, turn, eps):
    """2 int dtheta / sqrt(2 (E - V)) from theta_t to 2 pi - theta_t, V = 3 mu n^2 R~, at the turn's level."""
    level = closed_r(turn, eps)
    mid, half = np.pi, np.pi - turn

    def integrand(u):
        return half * np.cos(u) / np.sqrt(2 * 3 * mu * n**2 * (level - closed_r(mid + half * np.sin(u), eps)))

    return 2 * quad(integrand, -np.pi / 2, np.pi / 2, limit=400)[0]


@pytest.fixture(scope='module')
def earth():
    """The Earth's mass and mean motion as the package's context has them, and a circular body of a = 1 au."""
    t = np.linspace(0.0, 10.0, 101)
    context = build_context(earth_series(t, np.full(t.size, np.pi)), SETTINGS)
    mu, n = context.planet_mass['Earth'], context.planet_mean_motion_rad_yr['Earth']
    eps = co.hill_softening(mu, SETTINGS.mmr_coorbital_softening_hill)
    return mu, n, eps, cg.profile_from_orbits(CIRCULAR, CIRCULAR, mu, n, SETTINGS)


def earth_series(t, theta, a=None):
    """A 1:1 Earth record (key 1E-1: sigma = lambda_E - lambda = -theta) of a circular coplanar body, with the Earth's
    circular coplanar elements saved (the profile is then the circular problem's)."""
    zero, one = np.zeros_like(t), np.ones_like(t)
    lam_earth = 2 * np.pi * t
    planet = PlanetSeries('Earth', t, one, zero, zero, zero, zero, mean_longitude=lam_earth)
    return MMRSeries(
        t,
        one if a is None else a,
        np.full(t.size, 1e-4),
        zero,
        zero,
        zero,
        zero,
        np.mod(-theta, 2 * np.pi),
        ResonanceSpec.from_key('1E-1'),
        Regime(float(t[1] - t[0]), float(t[-1]), t.size),
        planets={'Earth': planet},
    )


def horseshoe_track(mu, n, eps, turn_deg, span_yr, step_yr=0.05):
    """theta(t), a(t) of the averaged circular problem from L3 with the energy of a turn at turn_deg; times of the
    turns (theta_dot = 0)."""

    def slope(theta):
        d2 = 4 * np.sin(theta / 2) ** 2 + eps**2
        return np.sin(theta) * (1.0 - d2**-1.5)

    def rhs(_, y):
        return [y[1], -3 * mu * n**2 * slope(y[0])]

    def turn(_, y):
        return y[1]

    level = closed_r(np.radians(turn_deg), eps) - closed_r(np.pi, eps)
    start = [np.pi, -np.sqrt(2 * 3 * mu * n**2 * level)]
    t = np.arange(0.0, span_yr, step_yr)
    run = solve_ivp(rhs, (0.0, span_yr), start, t_eval=t, events=turn, rtol=1e-11, atol=1e-13, method='DOP853')
    theta, rate = run.y
    a_planet = (co.GM_SUN / n**2) ** (1.0 / 3.0)
    return t, theta, a_planet * (1.0 - (2.0 / 3.0) * rate / n), np.asarray(run.t_events[0])


def run_floor_of(t, theta, a, turns_yr, settings=SETTINGS, cutoff_yr=10.0):
    """The package's `CoorbitalRunFloor` reading of the run of turning points at `turns_yr` (sigma = -theta). The
    synthetic track has no orbital wiggle, so it is averaged over one step (none): a 1-yr mean moves a turn 3.7 deg
    from the planet out by ~0.2 deg (theta_ddot / 24 yr^2), +3 % on T_hs."""
    series = earth_series(t, theta, a)
    context = build_context(series, settings)
    sigma = -theta
    model = cg.coorbital_run_floor(series, context, t, sigma, float(t[1] - t[0]), cutoff_yr, settings)
    values = np.interp(turns_yr, t, sigma)
    return model.read(turns_yr, values)


# --- the fastest physical horseshoe ---------------------------------------------------------------------------------


@pytest.mark.parametrize('hill', [3.0, 10.0])
def test_fastest_horseshoe_turns_closest_in_the_circular_problem(earth, hill):
    """Circular problem: T_hs falls as the turn nears the planet, so the fastest loop turning >= f r_H away turns at
    f r_H (3 r_H = 1.72 deg: about 113 yr for the Earth); against quad."""
    mu, n, eps, profile = earth
    theta_min = hill * (mu / 3) ** (1 / 3)
    expected = quad_horseshoe_period(mu, n, theta_min, eps)
    assert cg.fastest_horseshoe_period(profile, theta_min) == pytest.approx(expected, rel=2e-3)
    assert cg.fastest_horseshoe_period(profile, np.pi) == np.inf  # no turn left to scan


def test_fast_reference_is_at_most_the_tadpole_floor(earth):
    """The fast-residual reference of a 1:1 angle is the shortest floor any run can get: min(tadpole floor, factor x
    the fastest horseshoe at MMR_COORBITAL_MIN_TURN_HILL); the switch off keeps the tadpole floor."""
    mu, n, eps, _ = earth
    t = np.linspace(0.0, 300.0, 601)
    series = earth_series(t, np.full(t.size, np.pi))  # at rest on L3: only the elements make the profile
    context = build_context(series, SETTINGS)
    tadpole_floor = FACTOR * (2 * np.pi / n) / np.sqrt(27 * mu / 4)  # 111 yr
    fastest = quad_horseshoe_period(mu, n, SETTINGS.mmr_coorbital_min_turn_hill * (mu / 3) ** (1 / 3), eps)
    assert cg.fast_reference_yr(series, context, SETTINGS, tadpole_floor) == pytest.approx(FACTOR * fastest, rel=5e-3)
    assert cg.fast_reference_yr(series, context, SETTINGS, 30.0) == 30.0  # never above the floor
    off = AnalysisSettings(mmr_coorbital_profile_floor=False)
    assert cg.fast_reference_yr(series, context, off, tadpole_floor) == tadpole_floor


# --- the reading of one run -----------------------------------------------------------------------------------------


def test_horseshoe_turning_near_the_planet_has_the_floor_of_its_own_loop(earth):
    """Turns 3.7 deg from the Earth: both are reflections on the planet's top, the floor is the factor x T_hs(3.7 deg)
    (quad), below the tadpole floor, and the local law is the planet's (K / K_theory ~ 1)."""
    mu, n, eps, _ = earth
    t, theta, a, turns = horseshoe_track(mu, n, eps, 3.7, 300.0)
    floor, setting, source, columns = run_floor_of(t, theta, a, turns[:2])
    reference = quad_horseshoe_period(mu, n, np.radians(3.7), eps)
    assert source == cg.FLOOR_SOURCE_LOOP and setting == 'mmr_coorbital_period_floor_factor'
    assert floor == pytest.approx(FACTOR * reference, rel=5e-3)
    assert columns['cpl_turn_sites'] == 'planet;planet' and columns['cpl_loop_period_yr'] == pytest.approx(reference, rel=5e-3)
    assert columns['cpl_law_K_over_K_theory'] == pytest.approx(1.0, abs=0.1)
    assert columns['cpl_turn_out_in_entry'] == pytest.approx(1.0, abs=0.05)  # a reflection mirrors |da|
    tadpole_floor = FACTOR * (2 * np.pi / n) / np.sqrt(27 * mu / 4)
    assert 0.8 * reference < tadpole_floor  # the case of the card: 0.8 T_hs is under the tadpole floor


def test_a_pushed_the_wrong_way_keeps_the_tadpole_floor(earth):
    """The same turns with a's swing reversed: the force is not the planet's (K < 0), so the run has no floor of its
    own (the angle's tadpole floor stays)."""
    mu, n, eps, _ = earth
    t, theta, a, turns = horseshoe_track(mu, n, eps, 3.7, 300.0)
    a_planet = (co.GM_SUN / n**2) ** (1.0 / 3.0)
    floor, _, source, columns = run_floor_of(t, theta, 2 * a_planet - a, turns[:2])
    assert source == cg.FLOOR_SOURCE_LAW and np.isnan(floor) and columns['cpl_law_K_over_K_theory'] < 0


def test_turns_about_l3_are_no_reflections():
    """Turns at 174 / 212 deg (2022 AP1's): no reflection on a top, no floor of their own."""
    t = np.arange(0.0, 200.0, 0.05)
    theta = np.radians(193.0 + 19.0 * np.sin(2 * np.pi * t / 80.0))
    turns = np.array([20.0, 60.0])
    floor, _, source, columns = run_floor_of(t, theta, np.ones_like(t), turns)
    assert source == cg.FLOOR_SOURCE_NOT_REFLECTION and np.isnan(floor)
    assert cg.SITE_PLANET not in columns['cpl_turn_sites']


def test_averaged_turns_read_the_turn_the_filter_rounds_off():
    """Turns at 4 / 356 deg read 30 deg inwards by a filter: the averaged angle's extremum of each TP's kind near it,
    moved to the TP's branch (here 3 turns up)."""
    t = np.arange(0.0, 100.05, 0.1)
    averaged = np.radians(180.0 - 176.0 * np.cos(np.pi * t / 50.0))
    tp_times = np.array([0.3, 49.0, 99.0])
    tp_values = 6 * np.pi + np.radians([34.0, 326.0, 34.0])
    times, values = cg.averaged_turns(t, averaged, tp_times, tp_values)
    assert np.degrees(values - 6 * np.pi) == pytest.approx([4.0, 356.0, 4.0], abs=0.01)
    assert times == pytest.approx([0.0, 50.0, 100.0], abs=0.15)
    assert np.all(np.isnan(cg.averaged_turns(t, averaged, tp_times[:1], tp_values[:1])[1]))  # one TP: no half-cycle


# --- the decision -------------------------------------------------------------------------------------------------


def coorbital_geometry_with(run, own_floor=None, angle_floor=111.0):
    floors = {g.run_key(run): g.RunFloor(own_floor, 'mmr_coorbital_period_floor_factor', 'profile_loop')} if own_floor else {}
    return make_geometry(
        runs=[run], period_floor_yr=angle_floor, period_floor_setting='mmr_coorbital_period_floor_factor', run_floors=floors, coorbital=True
    )


def test_an_episode_at_its_own_floor_is_a_libration():
    """0.8 T_hs (T_hs = 120 yr: 96 yr) under the 111-yr tadpole floor: rejected by the angle's floor, kept by its own
    (0.5 x 120 = 60 yr); a run without its own floor (turns about L3) keeps the tadpole floor."""
    episode = make_episode(tp_times_yr=np.array([0.0, 48.0, 96.0]), period_med_yr=96.0)
    assert decide.librating_episodes(coorbital_geometry_with(episode, own_floor=60.0), SETTINGS) == [episode]
    assert decide.librating_episodes(coorbital_geometry_with(episode), SETTINGS) == []
    assert decide.period_floor(coorbital_geometry_with(episode, own_floor=60.0), SETTINGS, episode) == (
        60.0,
        'mmr_coorbital_period_floor_factor',
    )


def test_quasi_satellite_well_keeps_its_own_floor():
    """In the G1q pass an episode inside the QS well has the QS floor, whatever its turns read."""
    episode = make_episode(tp_times_yr=np.array([0.0, 5.0, 10.0]), period_med_yr=10.0, centre_mod_2pi_rad=0.0)
    t = np.linspace(-1.0, 11.0, 13)
    well = g.FloorWell(t, np.zeros_like(t), np.full_like(t, 0.5), 111.0, 'mmr_coorbital_period_floor_factor')
    geometry = coorbital_geometry_with(episode, own_floor=60.0, angle_floor=4.0)
    geometry.floor_well = well
    assert decide.period_floor(geometry, SETTINGS, episode)[0] == 4.0


@pytest.mark.parametrize('ratio, counted', [(0.95, 2), (0.85, 0)])
@pytest.mark.parametrize('coorbital', [True, False])
def test_near_episode_within_the_thin_band_of_the_floor(ratio, counted, coorbital):
    """A two-TP short run of period ratio x floor (twice its half-cycle) is a near-episode within the thin band
    (1 - CONFIDENCE_THIN_BAND = 0.9) of its floor, for every MMR (g01-8 A): 0.95 counts, 0.85 does not."""
    floor = 150.0
    half = ratio * floor / 2
    run = make_episode(
        tp_times_yr=np.array([100.0, 100.0 + half]),
        accepted=False,
        reject_reason=g.RejectReason.TOO_FEW_TP,
        half_amp_max_rad=1.0,
    )
    if coorbital:
        geometry = coorbital_geometry_with(run, own_floor=floor, angle_floor=400.0)
    else:
        geometry = make_geometry(runs=[run], period_floor_yr=floor, period_floor_setting='mmr_libration_period_min_yr')
    assert decide.near_episode_tp(geometry, SETTINGS) == counted


# --- rule coorbital_pair_loop_floor: a run with a turn that is no reflection -------------------------------------------


def mixed_run(t, theta, turns):
    """Three horseshoe turns at the planet's top and a fourth turning point read half-way between the third turn and
    the next (theta ~ 130 deg there, below L3's top: a tadpole site, a turn that is no reflection; 2003 YN107 c032's
    planet -> qs pair is of this kind). [times]"""
    return np.append(turns[:3], 0.5 * (turns[2] + turns[3]))


def test_a_turn_that_is_no_reflection_takes_the_floor_of_the_reflecting_pairs(earth):
    """2003 YN107 (the author 2026-11-14): the run kept the tadpole floor (111 yr) for one turn that is no reflection.
    Now its reflecting pairs (both turns 2.5 deg from the Earth, >= 3 r_H = 1.72 deg) give 0.7 x T_hs(2.5 deg) (quad),
    since its own force law is the planet's (K / K_theory ~ 1, R^2 above MMR_FORCE_R2_FULL); the switch off keeps the
    fallback."""
    mu, n, eps, _ = earth
    t, theta, a, turns = horseshoe_track(mu, n, eps, 2.5, 400.0)
    run = mixed_run(t, theta, turns)
    floor, setting, source, columns = run_floor_of(t, theta, a, run)
    reference = quad_horseshoe_period(mu, n, np.radians(2.5), eps)
    assert columns['cpl_turn_sites'].split(';')[:3] == ['planet'] * 3 and 'tadpole' in columns['cpl_turn_sites']
    assert source == cg.FLOOR_SOURCE_PAIR_LOOP and setting == 'mmr_coorbital_qs_loop_factor'
    assert floor == pytest.approx(SETTINGS.mmr_coorbital_qs_loop_factor * reference, rel=5e-3)
    assert columns['cpl_law_K_over_K_theory'] == pytest.approx(1.0, abs=0.15)
    assert columns['cpl_law_R2'] >= SETTINGS.mmr_force_r2_full
    off = AnalysisSettings(mmr_coorbital_pair_loop_floor=False)
    series = earth_series(t, theta, a)
    model = cg.coorbital_run_floor(series, build_context(series, off), t, -theta, float(t[1] - t[0]), 10.0, off)
    reading = model.read(run, np.interp(run, t, -theta))
    assert reading[2] == cg.FLOOR_SOURCE_NOT_REFLECTION and np.isnan(reading[0])


def test_a_pair_loop_floor_is_at_most_the_tadpole_floor(earth):
    """Turns 15 deg from the Earth: 0.7 x T_hs(15 deg) (quad) is above the tadpole floor P / sqrt(27 mu / 4) / 2, which
    caps it (the rule only relaxes the fallback)."""
    mu, n, eps, _ = earth
    t, theta, a, turns = horseshoe_track(mu, n, eps, 15.0, 1400.0)
    floor, setting, source, _ = run_floor_of(t, theta, a, mixed_run(t, theta, turns))
    tadpole_floor = FACTOR * (2 * np.pi / n) / np.sqrt(27 * mu / 4)
    assert SETTINGS.mmr_coorbital_qs_loop_factor * quad_horseshoe_period(mu, n, np.radians(15.0), eps) > tadpole_floor
    assert source == cg.FLOOR_SOURCE_PAIR_LOOP and setting == 'mmr_coorbital_period_floor_factor'
    assert floor == pytest.approx(tadpole_floor, rel=1e-6)


def test_a_pair_loop_floor_needs_the_planets_force(earth):
    """The same run with a's swing reversed (K < 0) or a held fixed (no force at all): no floor of its own."""
    mu, n, eps, _ = earth
    t, theta, a, turns = horseshoe_track(mu, n, eps, 2.5, 400.0)
    a_planet = (co.GM_SUN / n**2) ** (1.0 / 3.0)
    for wrong in (2 * a_planet - a, np.full(t.size, a_planet)):
        floor, _, source, _ = run_floor_of(t, theta, wrong, mixed_run(t, theta, turns))
        assert source == cg.FLOOR_SOURCE_NOT_REFLECTION and np.isnan(floor)


def test_a_planet_turn_inside_the_hill_sphere_is_no_reflecting_pair(earth):
    """MMR_COORBITAL_PAIR_LOOP_MIN_HILL = 1 (the author 2026-11-15, 2022 RD2 c075): turns 1.2 deg from the Earth
    (2.1 r_H, r_H = 0.573 deg; inside the averaging margin 3 r_H of MMR_COORBITAL_MIN_TURN_HILL) still give the
    reflecting pairs' floor 0.7 x T_hs(1.2 deg) (quad), the run's own law passing in full; turns 0.45 deg away
    (0.8 r_H, inside the Hill sphere: the planet holds the body) give none, and the mixed run keeps the tadpole floor,
    as the 1.2-deg turns do with the limit at 3 r_H."""
    mu, n, eps, _ = earth
    t, theta, a, turns = horseshoe_track(mu, n, eps, 1.2, 300.0)
    floor, setting, source, columns = run_floor_of(t, theta, a, mixed_run(t, theta, turns))
    assert source == cg.FLOOR_SOURCE_PAIR_LOOP and setting == 'mmr_coorbital_qs_loop_factor'
    assert floor == pytest.approx(SETTINGS.mmr_coorbital_qs_loop_factor * quad_horseshoe_period(mu, n, np.radians(1.2), eps), rel=5e-3)
    floor, _, source, _ = run_floor_of(t, theta, a, mixed_run(t, theta, turns), AnalysisSettings(mmr_coorbital_pair_loop_min_hill=3.0))
    assert source == cg.FLOOR_SOURCE_NOT_REFLECTION and np.isnan(floor)
    t, theta, a, turns = horseshoe_track(mu, n, eps, 0.45, 300.0)
    floor, _, source, _ = run_floor_of(t, theta, a, mixed_run(t, theta, turns))
    assert source == cg.FLOOR_SOURCE_NOT_REFLECTION and np.isnan(floor)
