"""Context of an MMR record (SPEC §4.1): notation, rates, a*, synodic periods, planets, columns.

Reference values are computed by hand from Kepler's third law in Gaussian years (P = a^1.5) and the
package's planet table, independently of the code under test.
"""

import math

import numpy as np
import pytest

from resonances.data import const
from resonances.resonance.analysis.mmr import context as ctx
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceKind, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config

DEG = math.pi / 180
TABLE_N = {name: value * DEG for name, value in const.PLANET_N_DEG_PER_GAUSS_YR.items()}


# --- notation ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'key, planets, planet_k, k, planet_p, p, order, kind',
    [
        ('9M-16+0+7', ('Mars',), (9,), -16, (0,), 7, 7, ResonanceKind.TWO_BODY),
        ('2J-1', ('Jupiter',), (2,), -1, (0,), -1, 1, ResonanceKind.TWO_BODY),
        ('2J-1+0-1', ('Jupiter',), (2,), -1, (0,), -1, 1, ResonanceKind.TWO_BODY),
        ('36J-13+0-23', ('Jupiter',), (36,), -13, (0,), -23, 23, ResonanceKind.TWO_BODY),
        ('3J-1S-1+0+0-1', ('Jupiter', 'Saturn'), (3, -1), -1, (0, 0), -1, 1, ResonanceKind.THREE_BODY),
        ('4J-2S-1', ('Jupiter', 'Saturn'), (4, -2), -1, (0, 0), -1, 1, ResonanceKind.THREE_BODY),
        ('7E+2M-9+0+0+0', ('Earth', 'Mars'), (7, 2), -9, (0, 0), 0, 0, ResonanceKind.THREE_BODY),
        ('1R-4+0+3', ('Mercury',), (1,), -4, (0,), 3, 3, ResonanceKind.TWO_BODY),
    ],
)
def test_from_key_long_and_short_forms(key, planets, planet_k, k, planet_p, p, order, kind):
    spec = ResonanceSpec.from_key(key)
    assert spec.planets == planets
    assert spec.planet_coefficients == planet_k
    assert spec.body_coefficient == k
    assert spec.planet_varpi_coefficients == planet_p
    assert spec.body_varpi_coefficient == p
    assert spec.order == order
    assert spec.kind == kind


def test_retrograde_key_follows_li_et_al_angle():
    # phi = q lam* - p lam' - (p + q - 2k) varpi* + 2k Omega: 1J-3r0 -> (3, -1, -4, 0), order p + q = 4
    spec = ResonanceSpec.from_key('1J-3r0')
    assert (spec.body_coefficient, spec.planet_coefficients, spec.body_varpi_coefficient, spec.body_node_coefficient) == (3, (-1,), -4, 0)
    assert spec.order == 4 and spec.retrograde
    spec = ResonanceSpec.from_key('2S-5r2')
    assert (spec.body_coefficient, spec.planet_coefficients, spec.body_varpi_coefficient, spec.body_node_coefficient) == (5, (-2,), -3, 4)
    assert spec.order == 7 and spec.key == '2S-5r2'


@pytest.mark.parametrize('key', ['2J-2+0+1', 'J-1', '2X-1', '2J', '2J-1+0', '-1+2J', '1J-3r'])
def test_from_key_rejects_bad_notation(key):
    with pytest.raises(ValueError):
        ResonanceSpec.from_key(key)


def test_from_resonance_matches_package_objects():
    from resonances.mmr.retrograde_two_body import RetrogradeTwoBody
    from resonances.mmr.three_body import ThreeBody
    from resonances.mmr.two_body import TwoBody

    resonances = (
        TwoBody([9, -16, 0, 7], ['Mars']),
        TwoBody('2J-1'),
        ThreeBody([3, -1, -1, 0, 0, -1], ['Jupiter', 'Saturn']),
        RetrogradeTwoBody('1J-3r1'),
    )
    for resonance in resonances:
        spec = ResonanceSpec.from_resonance(resonance)
        assert spec == ResonanceSpec.from_key(resonance.to_s(), nominal_axis_au=spec.nominal_axis_au)
        assert spec.nominal_axis_au == pytest.approx(resonance.resonant_axis)


# --- rate model --------------------------------------------------------------------------------------


def test_mean_motion_round_trip_and_identity_coefficient():
    # P = a^1.5 yr: a = 4 au -> P = 8 yr -> n = 2 pi / 8
    assert float(ctx.mean_motion(4.0)) == pytest.approx(2 * math.pi / 8)
    assert float(ctx.axis_from_mean_motion(2 * math.pi / 8)) == pytest.approx(4.0)
    # c = -1.5 k n / a; k = -1 at 4 au: 1.5 * (pi/4) / 4
    assert ctx.identity_coefficient(-1, 4.0) == pytest.approx(1.5 * (math.pi / 4) / 4)


def test_exact_axis_of_the_2_1_with_jupiter_is_keplers_law():
    # n* = 2 n_J -> P* = P_J / 2 = 11.86175 / 2 yr -> a* = 5.930877^(2/3) = 3.27652 au
    spec = ResonanceSpec.from_key('2J-1+0-1')
    assert ctx.exact_axis(spec, TABLE_N, 0.0) == pytest.approx(3.276518, abs=2e-6)
    # rate zero there, and nu = -n + 2 n_J elsewhere
    assert ctx.expected_rate(spec, 3.276518, TABLE_N, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert ctx.expected_rate(spec, 4.0, TABLE_N, 0.0) == pytest.approx(2 * TABLE_N['Jupiter'] - math.pi / 4)


def test_slow_term_shifts_exact_axis_by_p_varpi_dot():
    # a* moves by da = -slow / c (linear): p varpi_dot = -23 x 55"/yr for 21514's 36J-13
    spec = ResonanceSpec.from_key('36J-13+0-23')
    slow = ctx.slow_rate(spec, 55.0 / 3600 * DEG, {'Jupiter': 0.0})
    assert slow == pytest.approx(-23 * 55.0 / 3600 * DEG)
    a0 = ctx.exact_axis(spec, TABLE_N, 0.0)
    a1 = ctx.exact_axis(spec, TABLE_N, slow)
    assert a1 - a0 == pytest.approx(-slow / ctx.identity_coefficient(-13, a0), rel=1e-3)


def test_slow_rate_skips_zero_coefficients_and_adds_nodal_terms():
    spec = ResonanceSpec.from_key('1J-3r1')  # p = -2 (varpi*), v = 2 (Omega)
    assert ctx.slow_rate(spec, 1e-5, {'Jupiter': float('nan')}, body_node_rate_rad_yr=-3e-5) == pytest.approx(-2e-5 - 6e-5)
    spec = ResonanceSpec.from_key('7E+2M-9+0+0+0')  # p = 0: a nan body rate does not matter
    assert ctx.slow_rate(spec, float('nan'), {'Earth': 1.0, 'Mars': 1.0}) == 0.0


def test_synodic_periods_prograde_and_retrograde():
    # Hilda: P = 7.9 yr, P_J = 11.86175 yr -> 1/(1/7.9 - 1/11.86175) = 23.653 yr; retrograde 1/(1/7.9 + 1/P_J) = 4.742 yr
    n = 2 * math.pi / 7.9
    assert ctx.synodic_periods(n, TABLE_N, ('Jupiter',))['Jupiter'] == pytest.approx(23.653, abs=1e-3)
    assert ctx.synodic_periods(n, TABLE_N, ('Jupiter',), retrograde=True)['Jupiter'] == pytest.approx(4.7419, abs=1e-3)


def test_crossed_planets_and_strength_scale():
    planets = {'Earth': (0.983, 1.017), 'Mars': (1.381, 1.666), 'Jupiter': (4.95, 5.46)}
    # 2003 UG22-like orbit q 0.91, Q 2.4: crosses Earth and Mars, not Jupiter
    assert ctx.crossed_planets(0.91, 2.4, planets) == ('Earth', 'Mars')
    # 285637-like Hungaria: q = 1.77 > Q_Mars
    assert ctx.crossed_planets(1.77, 2.1, planets) == ()
    masses = {'Jupiter': 9.5e-4, 'Saturn': 2.9e-4}
    assert ctx.mu_scale_no_eccentricity(ResonanceSpec.from_key('2J-1'), masses) == pytest.approx(9.5e-4)
    assert ctx.mu_scale_no_eccentricity(ResonanceSpec.from_key('3J-1S-1'), masses) == pytest.approx(9.5e-4 * 2.9e-4)


def test_near_commensurabilities_of_the_planets():
    periods = ctx.near_commensurability_periods(TABLE_N)
    # 2 pi / |2 n_J - 5 n_S + 3 g6| with the table: 937.98 yr (handbook: 935-939 yr measured)
    assert periods['great_inequality'] == pytest.approx(937.98, abs=0.05)
    # 13 n_E - 8 n_V = 4680.0066 - 4681.5139 deg/yr -> 238.8 yr (the 243-yr Venus transit cycle family)
    assert periods['venus_earth_8_13'] == pytest.approx(360 / (8 * 585.189239 - 13 * 360.000505), rel=1e-9)
    assert periods['uranus_neptune_1_2'] == pytest.approx(360 / (2 * 2.184902 - 4.284750), rel=1e-9)


# --- planets from saved series -------------------------------------------------------------------------


def _planet(name, t, n_rad_yr, wave=None):
    longitude = n_rad_yr * t
    if wave is not None:
        amplitude, period = wave
        longitude = longitude + amplitude * np.sin(2 * math.pi * t / period)
    ones = np.ones_like(t)
    return PlanetSeries(name, t, 5.2 * ones, 0.05 * ones, 0.02 * ones, ones, 0.3 + 1e-5 * t, np.mod(longitude, 2 * math.pi))


def test_planet_mean_motion_measured_at_a_coarse_step():
    # Earth sampled every 5.0012 yr turns ~5 times per step: np.unwrap is blind, the table-rate unwrap is not
    t = np.arange(4000) * 5.0012222
    true_n = TABLE_N['Earth'] + 1e-4 * DEG
    values, sources = ctx.planet_mean_motions({'Earth': _planet('Earth', t, true_n)}, ('Earth', 'Mars'))
    assert sources == {'Earth': 'measured', 'Mars': 'table'}
    assert values['Earth'] == pytest.approx(true_n, rel=1e-12)
    assert values['Mars'] == TABLE_N['Mars']


def test_planet_varpi_rates_measured_else_g():
    t = np.arange(0, 20000.0, 10.0)
    rates = ctx.planet_varpi_rates({'Saturn': _planet('Saturn', t, TABLE_N['Saturn'])}, ('Saturn', 'Jupiter'), 5000.0)
    assert rates['Saturn'] == pytest.approx(1e-5)
    assert rates['Jupiter'] == pytest.approx(const.PLANETARY_FREQUENCIES['g5'] / 3600 * DEG)


def test_great_inequality_measured_from_saved_longitudes_and_from_table():
    t = np.arange(0, 10000.0, 2.0)
    period = ctx.near_commensurability_periods(TABLE_N)['great_inequality']
    planets = {
        'Jupiter': _planet('Jupiter', t, TABLE_N['Jupiter'], (0.34 * DEG, period)),
        'Saturn': _planet('Saturn', t, TABLE_N['Saturn'], (-0.83 * DEG, period)),
    }
    gi = ctx.great_inequality(planets, TABLE_N, ('Jupiter',))
    assert gi.measured and gi.period_yr == pytest.approx(period)
    assert gi.dlambda_jupiter_rad == pytest.approx(0.34 * DEG, rel=1e-3)
    assert gi.dlambda_saturn_rad == pytest.approx(0.83 * DEG, rel=1e-3)
    table = ctx.great_inequality({}, TABLE_N, ('Saturn',))
    assert not table.measured and table.dlambda_saturn_rad == pytest.approx(0.83 * DEG)
    assert ctx.great_inequality({}, TABLE_N, ('Earth', 'Mars')) is None


def test_axis_from_mean_longitude_rate_at_a_step_longer_than_the_orbit():
    # a = 1.0942 au (P = 1.1446 yr) sampled every 5.0012 yr: only the n(a)-guided unwrap recovers lambda_dot
    t = np.arange(4000) * 5.0012222
    a = np.full_like(t, 1.0942)
    longitude = np.mod(float(ctx.mean_motion(1.0942)) * t + 0.3, 2 * math.pi)
    assert ctx.axis_from_mean_longitude_rate(t, longitude, a) == pytest.approx(1.0942, rel=1e-10)
    assert ctx.axis_from_mean_longitude_rate(t, None, a) != ctx.axis_from_mean_longitude_rate(t, None, a)  # nan


# --- build_context and columns ---------------------------------------------------------------------------


def _two_body_series(step, span=2000.0, a0=3.0, key='2J-1+0-1'):
    t = np.arange(0.0, span + step / 2, step)
    a = a0 + 1e-4 * np.sin(2 * math.pi * t / 37.0)
    n_body = float(ctx.mean_motion(a0))
    longitude = n_body * t
    jupiter = _planet('Jupiter', t, TABLE_N['Jupiter'])
    spec = ResonanceSpec.from_key(key)
    varpi = 0.5 + 1e-4 * t
    sigma = 2 * TABLE_N['Jupiter'] * t - longitude - varpi
    ones = np.ones_like(t)
    return MMRSeries(
        t, a, 0.1 * ones, 0.1 * ones, ones, varpi - 1, varpi, np.mod(sigma, 2 * math.pi), spec,
        Regime(step, span, len(t)), {'Jupiter': jupiter}, np.mod(longitude, 2 * math.pi), name='synthetic',
    )  # fmt: skip


def test_build_context_on_a_synthetic_circulation():
    series = _two_body_series(0.5)
    series.validate()
    context = ctx.build_context(series, settings_from_config())
    n_body = float(ctx.mean_motion(3.0))
    assert context.planet_mean_motion_source['Jupiter'] == 'measured'
    assert context.body_varpi_rate_rad_yr == pytest.approx(1e-4)
    assert context.slow_rate_rad_yr == pytest.approx(-1e-4)
    assert context.rate_expected_rad_yr == pytest.approx(2 * TABLE_N['Jupiter'] - n_body - 1e-4, abs=2e-6)
    assert context.rate_from_mean_longitude_rad_yr == pytest.approx(2 * TABLE_N['Jupiter'] - n_body - 1e-4, abs=1e-9)
    assert context.axis_from_mean_longitude_rate_au == pytest.approx(3.0, rel=1e-9)
    assert context.exact_axis_au == pytest.approx(3.2765, abs=1e-3)
    assert context.orbital_period_yr == pytest.approx(3.0**1.5, rel=1e-6)
    assert context.fair_step_fraction == pytest.approx(0.5 / 3.0**1.5, rel=1e-6)
    assert context.great_inequality is not None and not context.great_inequality.measured  # 2 kyr < 2 P_GI
    columns = context.to_columns()
    assert columns['ctx_order'] == 1 and columns['ctx_k'] == -1 and columns['ctx_p'] == -1
    assert columns['ctx_n_deg_yr'] == pytest.approx(n_body / DEG)
    assert columns['ctx_rate_measured_deg_yr'] is None  # filled by sampling
    assert columns['ctx_P_syn_yr'].startswith('J:')
    rows = context.constants_rows()
    assert rows[0]['planet'] == 'Jupiter' and rows[0]['n_source'] == 'measured'
    assert any(row['planet'] == 'great_inequality' for row in rows)


def test_validate_rejects_inconsistent_series():
    series = _two_body_series(1.0)
    with pytest.raises(ValueError, match='increase'):
        MMRSeries(**{**series.__dict__, 't_yr': series.t_yr[::-1].copy()}).validate()
    with pytest.raises(ValueError, match='length'):
        MMRSeries(**{**series.__dict__, 'e': series.e[:-1]}).validate()
    three_body = ResonanceSpec.from_key('3J-1S-1')
    with pytest.raises(ValueError, match='partly'):
        MMRSeries(**{**series.__dict__, 'resonance': three_body}).validate()
