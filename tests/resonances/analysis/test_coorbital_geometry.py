"""Co-orbital geometry from the averaged potential (`analysis/mmr/coorbital_geometry.py`, W28 C2).

Independent sources of the expected values:
- the circular coplanar closed form of Murray & Dermott (1999), eqs. 3.178-3.179, p. 105:
  R~ = 1 / (2 sin(theta/2)) - cos theta = H / 2 + 1, (da / a_P)^2 + (8/3) mu R~ = const; periods by scipy quad of
  T = 2 int dtheta / sqrt(2 (E - 3 mu n^2 R~)) and the linear tadpole period P / sqrt(27 mu / 4) (M&D eq. 3.151 at
  small mu: Earth + Moon 221 yr, Venus 151 yr);
- the quasi-satellite well of an e = 0.1 orbit (Namouni 1999, Icarus 137, 293): its small-amplitude period from the
  curvature at theta = 0 of the averaged potential computed here by scipy quad over the planet's longitude, with
  Kepler's equation solved by brentq (not the package's Newton, not its trapezoidal tables);
- trajectories of the averaged circular problem integrated here by scipy solve_ivp (the turning point of a horseshoe
  mirrors |da| by time reversal; a's response is the planet's own force, K/K_theory = 1).
"""

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from resonances.resonance.analysis.mmr import coorbital as co
from resonances.resonance.analysis.mmr import coorbital_geometry as cg
from resonances.resonance.analysis.mmr.context import build_context
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

SETTINGS = AnalysisSettings()
CIRCULAR = co.Orbit(0.0, 0.0, 0.0, 0.0)
MU_EARTH, P_EARTH = 3.0404e-6, 1.0  # Earth + Moon / Sun; the period in years (n = 2 pi rad/yr, a_P = 1 au)
MU_VENUS, P_VENUS = 2.4478e-6, 0.61520
N_EARTH = 2 * np.pi / P_EARTH
EPS = co.hill_softening(MU_EARTH, SETTINGS.mmr_coorbital_softening_hill)


def closed_r(theta, eps=0.0):
    """R~ of the circular coplanar problem (M&D eq. 3.179 with H = 2 R~ - 2), softened as the package's table."""
    return 1.0 / np.sqrt(4 * np.sin(theta / 2) ** 2 + eps**2) - np.cos(theta)


def closed_h(theta):
    """M&D eq. 3.179 itself."""
    return 1.0 / np.sin(theta / 2) - 2 * np.cos(theta) - 2


def quad_period(mu, n, theta_lo, theta_hi, potential):
    """2 int dtheta / sqrt(2 (E - V)) between two turning points of equal level, sigma = mid + half sin u."""
    level = potential(theta_lo)
    mid, half = (theta_hi + theta_lo) / 2, (theta_hi - theta_lo) / 2

    def integrand(u):
        return half * np.cos(u) / np.sqrt(2 * 3 * mu * n**2 * (level - potential(mid + half * np.sin(u))))

    return 2 * quad(integrand, -np.pi / 2, np.pi / 2, limit=400)[0]


@pytest.fixture(scope='module')
def earth():
    return cg.profile_from_orbits(CIRCULAR, CIRCULAR, MU_EARTH, N_EARTH, SETTINGS)


@pytest.fixture(scope='module')
def eccentric():
    """A body of e = 0.1 on the circular Earth's orbit: a quasi-satellite well about theta = 0."""
    return cg.profile_from_orbits(co.Orbit(0.1, 0.0, 0.0, 0.0), CIRCULAR, MU_EARTH, N_EARTH, SETTINGS)


# --- the profile ---------------------------------------------------------------------------------------------------


def test_profile_reproduces_the_closed_form_between_the_nodes(earth):
    """The interpolation (spline of 1 / R~) agrees with M&D between the 0.5-deg nodes, also 2.9 deg from the planet,
    where a spline of R~ itself rings from the softened one-node peak (1.2 %)."""
    theta = np.radians(np.array([2.9, 7.3, 33.3, 60.2, 121.7, 179.9, 301.1]))
    assert earth.potential(theta) == pytest.approx(closed_r(theta, EPS), rel=2e-5)
    assert earth.a_planet_au == pytest.approx(1.0, rel=1e-12)


def test_extremes_of_the_circular_problem(earth):
    names = [(round(np.degrees(x.theta_rad), 3), x.name) for x in earth.extremes]
    assert names == [(60.0, cg.SITE_L4), (180.0, cg.SITE_L3), (300.0, cg.SITE_L5), (360.0, cg.SITE_PLANET)]


def test_quasi_satellite_well_of_an_eccentric_orbit(eccentric):
    """e = 0.1: a minimum at theta = 0 between the planet's two tops (Namouni 1999); L3 stays a top."""
    named = {x.name: x for x in eccentric.extremes if not x.is_top}
    assert abs(np.angle(np.exp(1j * named[cg.SITE_QS].theta_rad))) < np.radians(0.1)
    tops = [x for x in eccentric.extremes if x.is_top]
    planet_tops = sorted(np.degrees(np.angle(np.exp(1j * x.theta_rad))) for x in tops if x.name == cg.SITE_PLANET)
    assert len(planet_tops) == 2 and planet_tops[0] < 0 < planet_tops[1]
    assert [x.name for x in tops if abs(x.theta_rad - np.pi) < 0.01] == [cg.SITE_L3]
    assert cg.well_of(eccentric, 0.0).name == cg.SITE_QS
    assert cg.well_of(eccentric, np.radians(1.0), np.radians(4.0)).name == cg.SITE_QS


# --- sites and wells -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'theta_deg, site',
    [
        (60.0, cg.SITE_L4),  # the bottom of L4
        (300.0, cg.SITE_L5),
        (180.0, cg.SITE_L3),  # at the top itself
        (3.0, cg.SITE_PLANET),  # R~ = 18.1 above L3's 1.5: a horseshoe reflection at the planet
        (357.0, cg.SITE_PLANET),
        (30.0, cg.SITE_L4),  # R~ = 1.066 < 1.5: a tadpole's turn on the planet side
        (20.0, cg.SITE_PLANET),  # R~ = 1.940 > 1.5: above the separatrix (critical tadpole 23.9 deg)
        (150.0, cg.SITE_L4),  # a tadpole's turn on the L3 side
    ],
)
def test_turn_site_in_the_circular_problem(earth, theta_deg, site):
    assert cg.turn_site(earth, np.radians(theta_deg)) == site
    assert cg.turn_site(earth, np.radians(theta_deg) + 2 * np.pi) == site  # any branch of the unwrapped angle


def test_turn_site_inside_and_above_the_quasi_satellite_well(eccentric):
    planet_top = max(x.value for x in eccentric.extremes if x.name == cg.SITE_PLANET)
    assert cg.turn_site(eccentric, np.radians(3.0)) == cg.SITE_QS
    assert float(eccentric.potential(np.radians(3.0))) < planet_top
    assert cg.turn_site(eccentric, np.radians(20.0)) == cg.SITE_PLANET  # outside, R~ 2.47 above L3's 1.50: horseshoe
    assert cg.turn_site(eccentric, np.radians(30.0)) == cg.SITE_L4  # R~ 1.20 below L3: a tadpole's turn


def test_well_of_names_the_minima_the_swing_holds(earth, eccentric):
    assert cg.well_of(earth, np.radians(60), np.radians(20)).name == cg.SITE_L4
    assert cg.well_of(earth, np.radians(300), np.radians(20)).name == cg.SITE_L5
    assert cg.well_of(earth, np.pi, np.radians(170)).name == cg.WELL_HORSESHOE
    assert cg.well_of(earth, np.radians(61)).name == cg.SITE_L4  # no amplitude: the slope's basin
    assert cg.well_of(earth, np.pi, np.radians(185)).name == cg.SITE_NONE  # a whole turn: no libration
    assert cg.well_of(earth, np.radians(170), np.radians(5)).name == cg.SITE_NONE  # about L3's top: holds no minimum
    assert cg.well_of(eccentric, np.radians(40), np.radians(50)).name == cg.WELL_COMPOUND  # holds 0 and L4


# --- da: the energy integral ---------------------------------------------------------------------------------------


def test_loop_da_at_l3_is_the_horseshoe_closed_form(earth):
    """|da| at L3 of the loop that turns at 2.9 deg: sqrt((4/3) mu (H(theta_t) - H(pi))) (M&D eq. 3.178, a_P = 1)."""
    turn = np.radians(2.9)
    expected = np.sqrt(4.0 / 3.0 * MU_EARTH * (closed_h(turn) - closed_h(np.pi)))
    assert float(cg.loop_da(earth, turn, np.pi)) == pytest.approx(expected, rel=1e-3)
    assert np.isnan(cg.loop_da(earth, np.radians(60), np.pi))  # L3 is above a small tadpole's level


def test_da_max_below_the_l3_top_is_the_closed_form(earth):
    """A tadpole moving from 90 deg towards a turn at 150 deg: the top ahead is L3 (R~ = 1.5), so
    da_max(90 deg) = a_P sqrt((8/3) mu (R~(pi) - R~(90 deg))) = sqrt((4/3) mu (H(pi) - H(90 deg)))."""
    expected = np.sqrt(4.0 / 3.0 * MU_EARTH * (closed_h(np.pi) - closed_h(np.pi / 2)))
    assert float(cg.da_max(earth, np.radians(150), np.radians(90))) == pytest.approx(expected, rel=1e-6)
    # an array of positions on the same side: each reads its own arc, the top is still L3
    theta = np.radians(np.array([80.0, 100.0, 120.0]))
    expected = np.sqrt(4.0 / 3.0 * MU_EARTH * (closed_h(np.pi) - closed_h(theta)))
    assert cg.da_max(earth, np.radians(150), theta) == pytest.approx(expected, rel=1e-6)
    assert np.isnan(cg.da_max(earth, np.radians(150), np.radians(150)))  # no direction at the turn itself


def test_da_max_of_a_horseshoe_turn_reaches_the_planet_top(eccentric):
    """Smoke (e = 0.1, no closed form): moving towards the planet the top ahead is the planet's; a level below it is
    reversed (ratio < 1), a level above it is not."""
    top = max(x.value for x in eccentric.extremes if x.name == cg.SITE_PLANET)
    theta = np.radians(90.0)
    limit = float(cg.da_max(eccentric, np.radians(30.0), theta))
    assert limit == pytest.approx(np.sqrt(8.0 / 3.0 * MU_EARTH * (top - float(eccentric.potential(theta)))), rel=1e-4)
    below = float(cg.loop_da(eccentric, np.radians(30.0), theta))
    assert below / limit < 1


# --- periods -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('mu, period, expected', [(MU_EARTH, P_EARTH, 221.0), (MU_VENUS, P_VENUS, 151.0)])
def test_small_tadpole_period_is_the_linear_one(mu, period, expected):
    profile = cg.profile_from_orbits(CIRCULAR, CIRCULAR, mu, 2 * np.pi / period, SETTINGS)
    linear = period / np.sqrt(27 * mu / 4)
    assert linear == pytest.approx(expected, abs=1.0)
    turn = np.radians(61.0)
    far = brentq(lambda x: closed_r(x) - closed_r(turn), np.radians(40), np.radians(59.99))
    reference = quad_period(mu, 2 * np.pi / period, far, turn, closed_r)
    assert cg.loop_period(profile, [turn]) == pytest.approx(reference, rel=2e-3)
    assert cg.loop_period(profile, [turn]) == pytest.approx(linear, rel=5e-3)


@pytest.mark.parametrize('turn_deg', [2.9, 10.0])
def test_horseshoe_period_at_its_turning_angle(earth, turn_deg):
    """T_hs(theta_t) of M&D's horseshoe against quad (2.9 deg: the WP's case, within 2 %; it is within 1e-3)."""
    turn = np.radians(turn_deg)
    reference = quad_period(MU_EARTH, N_EARTH, turn, 2 * np.pi - turn, lambda x: closed_r(x, EPS))
    assert cg.loop_period(earth, [turn]) == pytest.approx(reference, rel=1e-3)
    assert cg.loop_period(earth, [turn, 2 * np.pi - turn]) == pytest.approx(reference, rel=1e-3)


def test_loop_period_at_a_separatrix_and_without_a_loop(earth):
    assert cg.loop_period(earth, [np.pi]) == np.inf  # at rest on L3
    assert np.isnan(cg.loop_period(earth, [np.radians(174.0), np.radians(212.0)]))  # a swing over L3's top (2022 AP1)
    assert np.isnan(cg.loop_period(earth, []))


def test_quasi_satellite_period_from_an_independent_average(eccentric):
    """The small-amplitude period of the e = 0.1 quasi-satellite well, 2 pi / sqrt(3 mu n^2 R~''(0)), with R~'' from
    a quad average of the softened direct + indirect potential over the planet's longitude (Richardson-extrapolated
    second difference: R~ is even in theta for a circular planet and varpi = 0)."""

    def body_xy(mean_anomaly, e=0.1):
        anomaly = brentq(lambda x: x - e * np.sin(x) - mean_anomaly, mean_anomaly - 1.0, mean_anomaly + 1.0)
        return np.cos(anomaly) - e, np.sqrt(1 - e * e) * np.sin(anomaly)

    def averaged(theta):
        def integrand(longitude):
            x, y = body_xy(theta + longitude)
            dx, dy = x - np.cos(longitude), y - np.sin(longitude)
            return 1.0 / np.sqrt(dx * dx + dy * dy + EPS**2) - (x * np.cos(longitude) + y * np.sin(longitude))

        return quad(integrand, 0.0, 2 * np.pi, epsabs=1e-13, epsrel=1e-13, limit=400)[0] / (2 * np.pi)

    centre = averaged(0.0)
    h1, h2 = np.radians(0.5), np.radians(0.25)
    second = [2 * (averaged(h) - centre) / h**2 for h in (h1, h2)]
    curvature = (4 * second[1] - second[0]) / 3
    period = 2 * np.pi / np.sqrt(3 * MU_EARTH * N_EARTH**2 * curvature)
    assert cg.loop_period(eccentric, [np.radians(1.0)]) == pytest.approx(period, rel=0.01)


# --- turns along a trajectory --------------------------------------------------------------------------------------


def horseshoe_track(turn_deg=10.0, span_yr=400.0, step_yr=0.05):
    """theta(t) and a(t) of the averaged circular problem from L3 towards the planet: theta_ddot = -3 mu n^2 R~',
    a = a_P (1 - (2/3) theta_dot / n). Returns (t, theta, a, the first turn's time)."""

    def slope(theta):
        d2 = 4 * np.sin(theta / 2) ** 2 + EPS**2
        return np.sin(theta) * (1.0 - d2**-1.5)

    def rhs(_, y):
        return [y[1], -3 * MU_EARTH * N_EARTH**2 * slope(y[0])]

    def turn(_, y):
        return y[1]

    turn.direction = 1.0  # theta_dot from negative to positive: the turn near the planet on the L4 side
    level = closed_r(np.radians(turn_deg), EPS) - closed_r(np.pi, EPS)
    start = [np.pi, -np.sqrt(2 * 3 * MU_EARTH * N_EARTH**2 * level)]
    t = np.arange(0.0, span_yr, step_yr)
    run = solve_ivp(rhs, (0.0, span_yr), start, t_eval=t, events=turn, rtol=1e-11, atol=1e-13, method='DOP853')
    theta, rate = run.y
    return t, theta, 1.0 - (2.0 / 3.0) * rate / N_EARTH, float(run.t_events[0][0])


def test_turn_conservation_of_a_reflection_and_of_a_kick():
    t, theta, a, t_turn = horseshoe_track()
    da_in, da_out, out_in = cg.turn_conservation(t, a, t_turn, (5.0, 40.0), 1.0)
    assert out_in == pytest.approx(1.0, abs=0.01)  # time reversal of the energy integral
    assert da_in == pytest.approx(da_out, rel=0.01) and da_in > 0
    kicked = np.where(t > t_turn, 1.0 + 0.5 * (a - 1.0), a)  # |da| halved right after the turn
    assert cg.turn_conservation(t, kicked, t_turn, (5.0, 40.0), 1.0)[2] == pytest.approx(0.5, rel=0.01)
    assert np.isnan(cg.turn_conservation(t, a, t_turn, (5.0, 5.01), 1.0)[2])  # too few samples on a side


def test_turn_physics_and_leg_columns(earth):
    t, theta, a, t_turn = horseshoe_track()
    turn = cg.turn_physics(earth, t, theta, a, t_turn, (5.0, 40.0))
    assert turn.site == cg.SITE_PLANET
    assert turn.theta_deg == pytest.approx(10.0, abs=0.05)
    assert turn.out_in == pytest.approx(1.0, abs=0.01)
    # circular: the planet's softened top (R~ ~ 1/eps) is far above the turn's level, so the ratio is small but finite
    expected = np.sqrt((closed_r(np.radians(10.0), EPS) - 1.0) / (closed_r(0.0, EPS) - 1.0))
    assert 0 < turn.da_ratio_in < 1 and turn.da_ratio_in == pytest.approx(expected, rel=0.5)
    columns = cg.leg_columns(turn, None, 240.6)
    assert set(columns) == set(cg.LEG_COLUMNS)
    assert columns['cpl_turn_site_entry'] == cg.SITE_PLANET and columns['cpl_turn_site_exit'] == ''
    assert columns['cpl_turn_da_ratio_entry'] == turn.da_ratio_out and np.isnan(columns['cpl_turn_out_in_exit'])
    assert columns['cpl_loop_period_yr'] == 240.6


# --- the local law -------------------------------------------------------------------------------------------------


def record_input(t, theta, a):
    zero = np.zeros_like(t)
    return co.CoorbitalInput(
        t_yr=t,
        angle=theta,
        a=a,
        adot=zero,
        mask=np.ones(len(t), dtype=bool),
        pieces=((t[0], t[-1]),),
        e=zero,
        inc=zero,
        omega=zero,
        Omega=zero,
        planet='Earth',
        planet_mass=MU_EARTH,
        planet_mean_motion_rad_yr=N_EARTH,
        body_coefficient=1,
        cutoffs_yr=(),
        planet_elements={'e': zero, 'inc': zero, 'Omega': zero, 'varpi': zero},
    )


def test_local_law_on_one_leg_is_the_planets_force():
    """One leg of the horseshoe (from the first turn to the second): K/K_theory = 1 and the energy integral holds; the
    same leg with a's wiggle reversed is pushed the wrong way (K < 0: fail)."""
    t, theta, a, t_turn = horseshoe_track()
    period = cg.loop_period(cg.profile_from_orbits(CIRCULAR, CIRCULAR, MU_EARTH, N_EARTH, SETTINGS), [np.radians(10.0)])
    window = (t_turn, t_turn + period / 2)
    law = cg.local_law(record_input(t, theta, a), window, 3.0, SETTINGS)
    assert law.force_check == CheckValue.PASS and law.scale == pytest.approx(1.0, abs=0.03)
    assert law.energy_check == CheckValue.PASS
    assert law.regime == co.REGIME_HORSESHOE
    wrong = cg.local_law(record_input(t, theta, 2.0 - a), window, 3.0, SETTINGS)
    assert wrong.force_check == CheckValue.FAIL and wrong.scale < 0
    assert cg.local_law(record_input(t, theta, a), (100.0, 102.0), 3.0, SETTINGS) is None  # shorter than the cutoff
    assert cg.local_law(record_input(t, theta, a), window, 0.05, SETTINGS) is None  # cutoff not above two steps


# --- the profile of a record's window ------------------------------------------------------------------------------


def earth_series(t, theta):
    """A 1:1 Earth record (key 1E-1: sigma = lambda_E - lambda, so theta = -sigma) without saved planets."""
    zero = np.zeros_like(t)
    return MMRSeries(
        t,
        np.ones_like(t),
        np.full(t.size, 0.1),
        np.full(t.size, np.radians(2.0)),
        np.full(t.size, 0.4),
        np.full(t.size, 0.3),
        zero,
        np.mod(-theta, 2 * np.pi),
        ResonanceSpec.from_key('1E-1'),
        Regime(float(t[1] - t[0]), float(t[-1]), t.size),
    )


def test_profile_at_uses_the_window_and_the_planets_orbit():
    t = np.linspace(0.0, 100.0, 1001)
    e = np.where(t < 50, 0.05, 0.15)
    profile = cg.profile_at(t, e, 0.01, 0.2, 0.3, 60.0, 90.0, 'Earth', MU_EARTH, N_EARTH, SETTINGS)
    assert profile.table.body.e == pytest.approx(0.15) and profile.table.body.omega == pytest.approx(0.2)
    j2000 = co.PLANET_MEAN_ORBITS_J2000['Earth']
    assert profile.table.planet.e == pytest.approx(j2000[0])  # J2000 without saved elements
    saved = {'e': np.full(t.size, 0.02), 'inc': np.zeros(t.size), 'Omega': np.zeros(t.size), 'varpi': np.full(t.size, 1.0)}
    with co.record_tables(1e6):
        kept = co.record_tables_kept()
        profile = cg.profile_at(t, e, 0.01, 0.2, 0.3, 60.0, 90.0, 'Earth', MU_EARTH, N_EARTH, SETTINGS, saved)
        assert co.record_tables_kept() == kept + 1  # the record's table budget holds it
    assert profile.table.planet.e == pytest.approx(0.02) and profile.table.planet.omega == pytest.approx(1.0)
    assert cg.profile_at(t, e, 0.01, 0.2, 0.3, 200.0, 300.0, 'Earth', MU_EARTH, N_EARTH, SETTINGS) is None
    assert cg.profile_at(t, e, 0.01, 0.2, 0.3, 60.0, 90.0, 'Earth', 0.0, N_EARTH, SETTINGS) is None


def test_series_wrappers_of_a_record():
    t, theta, a, _ = horseshoe_track(span_yr=100.0, step_yr=0.1)
    series = earth_series(t, theta)
    context = build_context(series, SETTINGS)
    profile = cg.series_profile(series, context, 10.0, 90.0, SETTINGS)
    assert profile.table.body.e == pytest.approx(0.1) and profile.planet_mass == context.planet_mass['Earth']
    theta_back = cg.theta_of(np.unwrap(series.angle_wrapped), series.resonance.body_coefficient)
    assert np.mod(theta_back[0], 2 * np.pi) == pytest.approx(np.pi)  # the track starts at L3
    assert np.diff(theta_back)[0] == pytest.approx(np.diff(theta)[0], rel=1e-6)  # and moves as theta does
    record = cg.series_input(series, context, np.unwrap(series.angle_wrapped))
    assert record.planet == 'Earth' and record.body_coefficient == -1 and record.planet_elements is None
    other = MMRSeries(
        *[getattr(series, name) for name in ('t_yr', 'a', 'e', 'inc', 'Omega', 'omega', 'varpi', 'angle_wrapped')],
        ResonanceSpec.from_key('3J-1'),
        series.regime,
    )
    assert cg.series_profile(other, context, 10.0, 90.0, SETTINGS) is None
    assert cg.series_input(other, context, series.angle_wrapped) is None
