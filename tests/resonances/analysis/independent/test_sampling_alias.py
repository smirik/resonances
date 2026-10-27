"""Independent tests of the MMR context and sampling (SPEC §3.1, §4.1, §4.2, §9.2).

Written from the public API (impl/API.md) and the SPEC only. Every expected value comes from a
closed form, a hand calculation shown in the comment, or a synthetic series whose truth is built here.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from resonances.data import const
from resonances.resonance.analysis.mmr import context as ctx
from resonances.resonance.analysis.mmr import sampling as smp
from resonances.resonance.analysis.mmr.models import (
    MMRSeries,
    PlanetSeries,
    Regime,
    ResonanceKind,
    ResonanceSpec,
    StepVerdict,
)
from resonances.resonance.analysis.settings import settings_from_config

DEG = np.pi / 180.0
ARCSEC = DEG / 3600.0
TWO_PI = 2.0 * np.pi

N_JUPITER = const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'] * DEG
N_SATURN = const.PLANET_N_DEG_PER_GAUSS_YR['Saturn'] * DEG
N_VENUS = const.PLANET_N_DEG_PER_GAUSS_YR['Venus'] * DEG


def wrap(x):
    """Wrap to [-pi, pi)."""
    return (np.asarray(x) + np.pi) % TWO_PI - np.pi


def kepler_n(a):
    return TWO_PI * np.asarray(a, dtype=float) ** -1.5


def spec_2_1_jupiter():
    """Interior 2:1 with Jupiter: sigma = 2 lambda_J - lambda - varpi (k_J = 2, k = -1, p = -1)."""
    return ResonanceSpec(
        key='2J-1-1',
        kind=ResonanceKind.TWO_BODY,
        planets=('Jupiter',),
        planet_coefficients=(2,),
        body_coefficient=-1,
        body_varpi_coefficient=-1,
        planet_varpi_coefficients=(0,),
    )


def spec_three_body():
    """4J-2S-1: sigma = 4 lambda_J - 2 lambda_S - lambda - varpi_S (p_S = -1 as a planet varpi term)."""
    return ResonanceSpec(
        key='4J-2S-1+0+0-1',
        kind=ResonanceKind.THREE_BODY,
        planets=('Jupiter', 'Saturn'),
        planet_coefficients=(4, -2),
        body_coefficient=-1,
        body_varpi_coefficient=0,
        planet_varpi_coefficients=(0, -1),
    )


# ----------------------------------------------------------------------------------------------
# context.py
# ----------------------------------------------------------------------------------------------


def test_mean_motion_and_inverse_kepler_third_law():
    # Kepler III in au / Gaussian yr: P = a^1.5. a = 4 au -> P = 8 yr -> n = 2 pi / 8.
    assert ctx.mean_motion(4.0) == pytest.approx(np.pi / 4.0, rel=1e-12)
    assert ctx.mean_motion(1.0) == pytest.approx(TWO_PI, rel=1e-12)
    assert ctx.axis_from_mean_motion(np.pi / 4.0) == pytest.approx(4.0, rel=1e-12)
    # P = 27 yr -> a = 27^(2/3) = 9
    assert ctx.axis_from_mean_motion(TWO_PI / 27.0) == pytest.approx(9.0, rel=1e-12)
    a = np.array([0.7, 1.0, 2.5, 3.2768, 5.2])
    np.testing.assert_allclose(ctx.axis_from_mean_motion(ctx.mean_motion(a)), a, rtol=1e-12)


def test_identity_coefficient_hand_and_numerical_derivative():
    # k = -2, a = 4: n = pi/4, c = -1.5 * (-2) * (pi/4) / 4 = 3 pi / 16
    assert ctx.identity_coefficient(-2, 4.0) == pytest.approx(3.0 * np.pi / 16.0, rel=1e-12)
    # c is d nu / d a; check against a central difference of the expected rate
    spec = spec_2_1_jupiter()
    planet_n = {'Jupiter': N_JUPITER}
    a0, h = 3.1, 1e-6
    numeric = (ctx.expected_rate(spec, a0 + h, planet_n, 0.0) - ctx.expected_rate(spec, a0 - h, planet_n, 0.0)) / (2 * h)
    assert ctx.identity_coefficient(spec.body_coefficient, a0) == pytest.approx(numeric, rel=1e-6)


def test_expected_rate_two_body_scalar_and_array():
    spec = spec_2_1_jupiter()
    planet_n = {'Jupiter': N_JUPITER}
    slow = -3e-5
    # nu = 2 n_J - n(a) + slow, by hand from M&D eq. 8.22
    a = np.array([2.5, 3.0, 3.3])
    expected = 2 * N_JUPITER - kepler_n(a) + slow
    np.testing.assert_allclose(ctx.expected_rate(spec, a, planet_n, slow), expected, rtol=1e-12)
    assert ctx.expected_rate(spec, 3.0, planet_n, slow) == pytest.approx(expected[1], rel=1e-12)
    # a = 2.5: n = 360 / 2.5^1.5 = 360 / 3.952847 = 91.0736 deg/yr; 2 n_J = 60.6993 deg/yr -> nu = -30.3743
    assert ctx.expected_rate(spec, 2.5, planet_n, 0.0) / DEG == pytest.approx(-30.3743, abs=2e-4)


def test_expected_rate_three_body():
    spec = spec_three_body()
    planet_n = {'Jupiter': N_JUPITER, 'Saturn': N_SATURN}
    slow = 2e-5
    a = 3.08
    expected = 4 * N_JUPITER - 2 * N_SATURN - kepler_n(a) + slow
    assert ctx.expected_rate(spec, a, planet_n, slow) == pytest.approx(expected, rel=1e-12)


def test_exact_axis_2_1_is_hecuba_gap():
    spec = spec_2_1_jupiter()
    planet_n = {'Jupiter': N_JUPITER}
    # n(a*) = 2 n_J: P = 360 / (2 * 30.349646) = 5.930876 yr -> a* = 5.930876^(2/3) = 3.2765 au
    a_star = ctx.exact_axis(spec, planet_n, 0.0)
    assert a_star == pytest.approx((TWO_PI / (2 * N_JUPITER)) ** (2.0 / 3.0), rel=1e-10)
    assert a_star == pytest.approx(3.2765, abs=2e-4)
    assert ctx.expected_rate(spec, a_star, planet_n, 0.0) == pytest.approx(0.0, abs=1e-12)
    # a slow term shifts a*: nu(a*) = 0 must still hold
    slow = 1e-3
    a_star_slow = ctx.exact_axis(spec, planet_n, slow)
    assert ctx.expected_rate(spec, a_star_slow, planet_n, slow) == pytest.approx(0.0, abs=1e-12)
    # 2 n_J - n + slow = 0 -> n larger -> a smaller
    assert a_star_slow < a_star


def test_exact_axis_three_body_and_no_solution():
    spec = spec_three_body()
    planet_n = {'Jupiter': N_JUPITER, 'Saturn': N_SATURN}
    # n(a*) = 4 n_J - 2 n_S = 121.3986 - 24.4427 = 96.9559 deg/yr -> P = 3.713028 yr -> a* = 2.3978 au
    a_star = ctx.exact_axis(spec, planet_n, 0.0)
    assert a_star == pytest.approx((TWO_PI / (4 * N_JUPITER - 2 * N_SATURN)) ** (2 / 3), rel=1e-10)
    assert a_star == pytest.approx(2.3978, abs=3e-4)
    # -2 lambda_J - lambda: n(a*) = -2 n_J < 0 has no solution
    no_solution = ResonanceSpec(
        key='-2J-1+0',
        kind=ResonanceKind.TWO_BODY,
        planets=('Jupiter',),
        planet_coefficients=(-2,),
        body_coefficient=-1,
        body_varpi_coefficient=0,
        planet_varpi_coefficients=(0,),
    )
    assert np.isnan(ctx.exact_axis(no_solution, {'Jupiter': N_JUPITER}, 0.0))
    # k = 0: no body term, no a*
    k_zero = ResonanceSpec(
        key='2J-5S',
        kind=ResonanceKind.THREE_BODY,
        planets=('Jupiter', 'Saturn'),
        planet_coefficients=(2, -5),
        body_coefficient=0,
        body_varpi_coefficient=0,
        planet_varpi_coefficients=(0, 0),
    )
    assert np.isnan(ctx.exact_axis(k_zero, planet_n, 0.0))


def test_synodic_periods_hand():
    # P = 4 yr body and Jupiter P_J = 360/30.349646 = 11.86175 yr:
    # P_syn = 1 / |1/4 - 1/11.86175| = 1 / (0.25 - 0.0843046) = 6.03518 yr
    n = TWO_PI / 4.0
    p_j = TWO_PI / N_JUPITER
    result = ctx.synodic_periods(n, {'Jupiter': N_JUPITER, 'Saturn': N_SATURN}, ('Jupiter',))
    assert set(result) == {'Jupiter'}
    assert result['Jupiter'] == pytest.approx(1.0 / (0.25 - 1.0 / p_j), rel=1e-12)
    assert result['Jupiter'] == pytest.approx(6.0352, abs=2e-4)
    both = ctx.synodic_periods(n, {'Jupiter': N_JUPITER, 'Saturn': N_SATURN}, ('Jupiter', 'Saturn'))
    assert both['Saturn'] == pytest.approx(TWO_PI / abs(n - N_SATURN), rel=1e-12)


def test_axis_from_mean_longitude_rate_synthetic():
    # lambda = 0.3 + (2 pi / 8) t, saved mod 2 pi -> lambda_dot = pi/4 -> a = 4 au
    t = np.arange(0.0, 400.0, 0.1)
    lam = wrap(0.3 + np.pi / 4.0 * t)
    assert ctx.axis_from_mean_longitude_rate(t, lam) == pytest.approx(4.0, rel=1e-6)
    assert np.isnan(ctx.axis_from_mean_longitude_rate(t, None))


def test_crossed_planets_hand():
    # body q = 0.9, Q = 1.5. Venus (0.718, 0.728): q < Q_V false. Earth (0.983, 1.017): crosses.
    # Mars (1.381, 1.666): 0.9 < 1.666 and 1.5 > 1.381: crosses. Jupiter (4.95, 5.46): 1.5 > 4.95 false.
    planets = {
        'Venus': (0.718, 0.728),
        'Earth': (0.983, 1.017),
        'Mars': (1.381, 1.666),
        'Jupiter': (4.95, 5.46),
    }
    assert set(ctx.crossed_planets(0.9, 1.5, planets)) == {'Earth', 'Mars'}
    assert tuple(ctx.crossed_planets(2.1, 3.3, planets)) == ()


def test_mu_scale_no_eccentricity():
    masses = {'Jupiter': 9.5e-4, 'Saturn': 2.9e-4}
    assert ctx.mu_scale_no_eccentricity(spec_2_1_jupiter(), masses) == pytest.approx(9.5e-4, rel=1e-12)
    assert ctx.mu_scale_no_eccentricity(spec_three_body(), masses) == pytest.approx(9.5e-4 * 2.9e-4, rel=1e-12)


def test_slow_rate_hand():
    # p varpi_dot + sum p_j varpi_dot_j
    spec = spec_2_1_jupiter()
    assert ctx.slow_rate(spec, 2e-4, {'Jupiter': 1e-5}) == pytest.approx(-2e-4, rel=1e-12)
    three = spec_three_body()
    # p = 0, p_S = -1: -varpi_dot_S
    rates = {'Jupiter': 4.26 * ARCSEC, 'Saturn': 28.25 * ARCSEC}
    assert ctx.slow_rate(three, 5e-4, rates) == pytest.approx(-28.25 * ARCSEC, rel=1e-12)


def test_body_varpi_rate_negative_and_wrapped():
    t = np.arange(0.0, 20000.0, 1.0)
    g = -2e-4  # rad/yr, ~ -41 arcsec/yr; 20 kyr wrap the angle ~0.6 times through +-pi
    varpi = wrap(3.0 + g * t)
    assert ctx.body_varpi_rate(t, varpi, 5000.0) == pytest.approx(g, rel=1e-6)
    g_fast = 3e-3  # wraps ~9.5 times over the record
    assert ctx.body_varpi_rate(t, wrap(-3.1 + g_fast * t), 5000.0) == pytest.approx(g_fast, rel=1e-6)
    # constant varpi: zero rate
    assert ctx.body_varpi_rate(t, np.full_like(t, 1.2), 5000.0) == pytest.approx(0.0, abs=1e-12)


def _planet(name, t, mean_longitude=None, varpi=None):
    zeros = np.zeros_like(t)
    return PlanetSeries(
        name=name,
        t_yr=t,
        a=zeros + 5.2,
        e=zeros + 0.05,
        inc=zeros + 0.02,
        Omega=zeros + 1.7,
        varpi=zeros + 0.25 if varpi is None else varpi,
        mean_longitude=mean_longitude,
    )


def test_planet_mean_motions_table_and_measured():
    values, sources = ctx.planet_mean_motions({}, ('Jupiter', 'Saturn'))
    assert values['Jupiter'] == pytest.approx(30.349646 * DEG, rel=1e-9)
    assert values['Saturn'] == pytest.approx(12.221327 * DEG, rel=1e-9)
    assert sources == {'Jupiter': 'table', 'Saturn': 'table'}
    # Venus saved at a 1-yr step: 585 deg per step, np.unwrap would lose whole turns.
    # True rate differs from the table by 0.02 deg/yr: the slope must be the true one.
    t = np.arange(0.0, 2000.0, 1.0)
    n_true = N_VENUS + 0.02 * DEG
    venus = _planet('Venus', t, mean_longitude=wrap(0.4 + n_true * t))
    values, sources = ctx.planet_mean_motions({'Venus': venus}, ('Venus', 'Jupiter'))
    assert values['Venus'] == pytest.approx(n_true, rel=1e-8)
    assert sources['Venus'] == 'measured'
    assert sources['Jupiter'] == 'table'


def test_planet_varpi_rates_measured_and_table():
    t = np.arange(0.0, 20000.0, 2.0)
    g_saturn = 30.0 * ARCSEC
    saturn = _planet('Saturn', t, varpi=wrap(1.0 + g_saturn * t))
    rates = ctx.planet_varpi_rates({'Saturn': saturn}, ('Jupiter', 'Saturn'), 5000.0)
    assert rates['Saturn'] == pytest.approx(g_saturn, rel=1e-6)
    # Jupiter not saved: g5 of PLANETARY_FREQUENCIES (arcsec/yr)
    assert rates['Jupiter'] == pytest.approx(const.PLANETARY_FREQUENCIES['g5'] * ARCSEC, rel=1e-9)


def test_great_inequality_table_and_measured():
    planet_n = {'Jupiter': N_JUPITER, 'Saturn': N_SATURN}
    gi = ctx.great_inequality({}, planet_n)
    assert gi is not None
    assert not gi.measured
    assert gi.dlambda_jupiter_rad == pytest.approx(0.34 * DEG, rel=1e-9)
    assert gi.dlambda_saturn_rad == pytest.approx(0.83 * DEG, rel=1e-9)
    # 2 n_J - 5 n_S + 3 g6 = 60.699292 - 61.106635 + 0.023539 = -0.383804 deg/yr -> 938 yr
    assert 900.0 < gi.period_yr < 980.0
    assert ctx.great_inequality({}, {'Mars': 191.4 * DEG}) is None

    # Synthetic Jupiter/Saturn with GI waves of known half-amplitudes 0.5 and 1.2 deg
    g6 = const.PLANETARY_FREQUENCIES['g6'] * ARCSEC
    period = TWO_PI / abs(2 * N_JUPITER - 5 * N_SATURN + 3 * g6)
    t = np.arange(0.0, 20000.0, 2.0)
    phase = TWO_PI * t / period
    jupiter = _planet('Jupiter', t, mean_longitude=wrap(N_JUPITER * t + 0.5 * DEG * np.sin(phase + 0.3)))
    saturn = _planet(
        'Saturn',
        t,
        mean_longitude=wrap(1.0 + N_SATURN * t - 1.2 * DEG * np.sin(phase + 0.3)),
        varpi=wrap(1.6 + g6 * t),
    )
    gi = ctx.great_inequality({'Jupiter': jupiter, 'Saturn': saturn}, planet_n)
    assert gi.measured
    assert gi.period_yr == pytest.approx(period, rel=0.02)
    assert gi.dlambda_jupiter_rad == pytest.approx(0.5 * DEG, rel=0.15)
    assert gi.dlambda_saturn_rad == pytest.approx(1.2 * DEG, rel=0.15)


# ----------------------------------------------------------------------------------------------
# sampling.py: branches and apparent rates
# ----------------------------------------------------------------------------------------------


def test_branch_index_and_apparent_rate_knezevic_milani():
    # nu = 71.98 deg/yr, dt = 5.0012 yr: nu dt = 359.9864 deg -> m = 1.
    # apparent = 71.98 - 360/5.0012 = 71.98 - 71.982724 = -0.002724 deg/yr (hand).
    # (K&M 2001 quote -0.0039 deg/yr for the unrounded nu = 71.97882 deg/yr; 71.98 is rounded.)
    nu, dt = 71.98 * DEG, 5.0012
    assert smp.branch_index(nu, dt) == 1
    assert smp.apparent_rate(nu, dt) / DEG == pytest.approx(-0.002724, abs=2e-6)
    assert smp.apparent_rate(71.97882 * DEG, dt) / DEG == pytest.approx(-0.0039, abs=5e-5)


@pytest.mark.parametrize(
    'nu_deg, dt, m_expected, apparent_deg',
    [
        (10.0, 5.0, 0, 10.0),  # 50 deg/step: unaliased
        (-100.0, 3.0, -1, 20.0),  # -300 deg/step -> m = round(-0.833) = -1; -100 + 120 = 20
        (250.0, 2.0, 1, 70.0),  # 500 deg/step -> m = round(1.389) = 1; 250 - 180 = 70
        (-30.3743, 10.0, -1, 5.6257),  # -303.743 deg/step -> m = -1; -30.3743 + 36
    ],
)
def test_branch_and_apparent_rate_hand_and_sampled(nu_deg, dt, m_expected, apparent_deg):
    nu = nu_deg * DEG
    assert smp.branch_index(nu, dt) == m_expected
    assert smp.apparent_rate(nu, dt) / DEG == pytest.approx(apparent_deg, abs=1e-9)
    # synthetic truth: what np.unwrap of the sampled circulation shows
    t = np.arange(0.0, 400 * dt, dt)
    slope = np.polyfit(t, np.unwrap(wrap(0.7 + nu * t)), 1)[0]
    assert smp.apparent_rate(nu, dt) == pytest.approx(slope, abs=1e-9)


def test_branch_index_array():
    rates = np.array([10.0, -100.0, 250.0, 359.0]) * DEG
    np.testing.assert_array_equal(np.asarray(smp.branch_index(rates, 1.0)), [0, 0, 1, 1])
    np.testing.assert_allclose(np.asarray(smp.apparent_rate(rates, 1.0)) / DEG, [10.0, -100.0, -110.0, -1.0], atol=1e-9)


@pytest.mark.parametrize(
    'lo_deg, hi_deg, dt, verdict, branches',
    [
        (10.0, 15.0, 5.0, StepVerdict.RESOLVED, (0,)),  # 50..75 deg per step <= 90
        (-10.0, 10.0, 5.0, StepVerdict.RESOLVED, (0,)),  # |nu| dt <= 50 across zero
        (20.0, 30.0, 5.0, StepVerdict.MARGINAL, (0,)),  # 100..150 deg: one branch, not resolved
        (70.0, 71.0, 5.0, StepVerdict.ALIASED_UNAMBIGUOUS, (1,)),  # 350..355 deg: m = 1 only
        (-71.0, -70.0, 5.0, StepVerdict.ALIASED_UNAMBIGUOUS, (-1,)),
        (30.0, 40.0, 5.0, StepVerdict.AMBIGUOUS, (0, 1)),  # 150..200 deg: crosses 180
        (-40.0, 40.0, 5.0, StepVerdict.AMBIGUOUS, (-1, 0, 1)),
    ],
)
def test_step_verdict_hand(lo_deg, hi_deg, dt, verdict, branches):
    got_verdict, got_branches = smp.step_verdict(lo_deg * DEG, hi_deg * DEG, dt, 90.0 * DEG)
    assert got_verdict == verdict
    assert sorted(got_branches) == sorted(branches)


@pytest.mark.parametrize(
    'lo_deg, hi_deg, dt, margin',
    [
        (10.0, 15.0, 5.0, 0.5 - 75.0 / 360.0),  # x in [0.1389, 0.2083]; nearest half-integer 0.5
        (-15.0, -10.0, 5.0, 0.5 - 75.0 / 360.0),  # mirror image
        (70.0, 71.0, 5.0, 0.5 - (1.0 - 350.0 / 360.0)),  # x in [0.9722, 0.9861]; 0.5 is 0.4722 away
        (30.0, 40.0, 5.0, 0.0),  # x in [0.4167, 0.5556] crosses 0.5
        (36.0, 36.0, 5.0, 0.0),  # exactly a half-turn per step
    ],
)
def test_branch_margin_turns_hand(lo_deg, hi_deg, dt, margin):
    assert smp.branch_margin_turns(lo_deg * DEG, hi_deg * DEG, dt) == pytest.approx(margin, abs=1e-12)


def test_nyquist_halfwidth_closed_form_and_linearisation():
    # delta a_N = a^2.5 / (3 |k| dt); a = 2.5, k = -3, dt = 5: 9.88212 / 45 = 0.219603 au
    assert smp.nyquist_halfwidth(2.5, -3, 5.0) == pytest.approx(2.5**2.5 / 45.0, rel=1e-12)
    assert smp.nyquist_halfwidth(2.5, -3, 5.0) == pytest.approx(0.219603, abs=1e-6)
    # independent: linearised k (n(a + da) - n(a)) dt = pi at da = delta a_N (small da regime)
    a, k, dt = 2.0, -7, 200.0
    da = smp.nyquist_halfwidth(a, k, dt)
    turn = abs(k * (kepler_n(a + da) - kepler_n(a - da)) / 2.0) * dt
    assert turn == pytest.approx(np.pi, rel=1e-4)


def test_ghost_axis_looks_like_exact_resonance():
    spec = spec_2_1_jupiter()
    planet_n = {'Jupiter': N_JUPITER}
    dt = 10.0
    a_star = ctx.exact_axis(spec, planet_n, 0.0)
    assert smp.ghost_axis(spec, planet_n, 0.0, dt, 0) == pytest.approx(a_star, rel=1e-10)
    for m in (1, -1, -2):
        a_m = smp.ghost_axis(spec, planet_n, 0.0, dt, m)
        # Branch m is the one with m = round(nu dt / 2 pi) (branch_index); it looks exact where nu(a_m) = 2 pi m / dt,
        # i.e. n(a_m) = n(a*) + 2 pi m / (k dt) (SPEC §4.2 prints "-": a sign slip, see TEST-DISPUTES.md).
        # With k = -1, dt = 10, m = 1: n(a_m) = 1.0594 - 0.6283 = 0.4311 rad/yr -> a_m = 5.9669 au.
        n_m = kepler_n(a_star) + TWO_PI * m / (spec.body_coefficient * dt)
        assert a_m == pytest.approx((TWO_PI / n_m) ** (2 / 3), rel=1e-10)
        nu = ctx.expected_rate(spec, a_m, planet_n, 0.0)
        # a whole number of turns per step, nonzero, so np.unwrap sees a standing angle, on branch m itself
        assert smp.branch_index(nu, dt) == m
        assert smp.apparent_rate(nu, dt) == pytest.approx(0.0, abs=1e-10)
    assert smp.ghost_axis(spec, planet_n, 0.0, dt, 1) == pytest.approx(5.9669, abs=1e-4)
    # m = 2: n(a_2) = 1.0594 - 1.2566 < 0, no such orbit
    assert np.isnan(smp.ghost_axis(spec, planet_n, 0.0, dt, 2))


def test_far_ratio_hand():
    # a* = 3.2776 au, <a> = 3.0, a range 2.98..3.02 (half 0.02), a(lambda_dot) = 3.001: 0.2776 / 0.02 = 13.88
    context = SimpleNamespace(exact_axis_au=3.2776, a_mean_au=3.0, a_p01_au=2.98, a_p99_au=3.02, axis_from_mean_longitude_rate_au=3.001)
    assert smp.far_ratio(context) == pytest.approx(0.2776 / 0.02, rel=1e-12)
    # a lambda_dot that says the body moves at 3.2 au (model error 0.2 au) takes over the scale: 0.2776 / 0.2
    context.axis_from_mean_longitude_rate_au = 3.2
    assert smp.far_ratio(context) == pytest.approx(1.388, rel=1e-12)


def test_rate_model_midpoints():
    spec = spec_2_1_jupiter()
    planet_n = {'Jupiter': N_JUPITER}
    a = np.array([3.0, 3.1, 3.05, 3.2])
    got = smp.rate_model(spec, a, planet_n, 1e-4)
    assert len(got) == 3
    mid = 0.5 * (a[:-1] + a[1:])
    np.testing.assert_allclose(got, 2 * N_JUPITER - kepler_n(mid) + 1e-4, rtol=1e-12)


# ----------------------------------------------------------------------------------------------
# sampling.py: the predictive unwrap
# ----------------------------------------------------------------------------------------------


def test_predictive_unwrap_counts_turns_numpy_misses():
    # sigma = nu t with nu = 71.98 deg/yr sampled at 5.0012 yr over ~20 kyr (~3998 revolutions)
    nu, dt = 71.98 * DEG, 5.0012
    t = np.arange(4000) * dt
    wrapped = wrap(0.2 + nu * t)
    unwrapped, residual = smp.predictive_unwrap(t, wrapped, nu)
    assert len(unwrapped) == len(t) and len(residual) == len(t) - 1
    assert unwrapped[0] == pytest.approx(wrapped[0], abs=1e-12)
    true_revs = nu * (t[-1] - t[0]) / TWO_PI  # 3997.6
    assert smp.revolutions(unwrapped) == pytest.approx(true_revs, abs=1e-6)
    assert np.max(np.abs(residual)) < 1e-8
    # np.unwrap sees only the apparent rate: -0.0027 deg/yr * 20 kyr / 360 = -0.15 revolutions
    assert abs(smp.revolutions(np.unwrap(wrapped))) < 0.5


def test_predictive_unwrap_negative_rate_and_wrapped_convention():
    # nu = -50 deg/yr at dt = 4 yr: -200 deg per step; np.unwrap would add +160 deg per step
    nu, dt = -50.0 * DEG, 4.0
    t = np.arange(0.0, 400.0, dt)
    true = 3.1 + nu * t
    for wrapped in (wrap(true), np.mod(true, TWO_PI)):  # [-pi, pi) and [0, 2 pi) inputs
        unwrapped, _ = smp.predictive_unwrap(t, wrapped, nu)
        np.testing.assert_allclose(unwrapped - unwrapped[0], true - true[0], atol=1e-9)
    assert np.polyfit(t, np.unwrap(wrap(true)), 1)[0] > 0  # numpy goes the wrong way


def test_predictive_unwrap_array_model_and_residual_is_model_error():
    # truth: sigma = nu0 t + A sin(W t). Model: nu0 per step. Residual must be the exact per-step
    # model error A (sin W t_i - sin W t_{i-1}), |.| <= 2A = 2 rad < pi, so the unwrap is exact.
    nu0, amp, w, dt = 200.0 * DEG, 1.0, TWO_PI / 37.0, 2.0
    t = np.arange(0.0, 2000.0, dt)
    true = -2.9 + nu0 * t + amp * np.sin(w * t)
    rate = np.full(len(t) - 1, nu0)
    unwrapped, residual = smp.predictive_unwrap(t, wrap(true), rate)
    np.testing.assert_allclose(unwrapped - unwrapped[0], true - true[0], atol=1e-9)
    np.testing.assert_allclose(residual, amp * np.diff(np.sin(w * t)), atol=1e-9)


def test_predictive_unwrap_230_deg_error_appears_as_minus_130():
    # one step where the angle moves by the model + 230 deg: the residual reads -130 deg and
    # the unwrap silently adds -130 deg instead of +230 deg (2 pi short, SPEC §3.1)
    nu, dt = 10.0 * DEG, 1.0
    t = np.arange(0.0, 50.0, dt)
    increments = np.full(len(t) - 1, nu * dt)
    increments[20] += 230.0 * DEG
    true = np.concatenate([[0.5], 0.5 + np.cumsum(increments)])
    unwrapped, residual = smp.predictive_unwrap(t, wrap(true), nu)
    assert residual[20] / DEG == pytest.approx(-130.0, abs=1e-9)
    assert np.all(np.abs(np.delete(residual, 20)) < 1e-9)
    assert (unwrapped[-1] - unwrapped[0]) - (true[-1] - true[0]) == pytest.approx(-TWO_PI, abs=1e-9)
    assert np.all((residual >= -np.pi) & (residual < np.pi))


def test_predictive_unwrap_short_series():
    t = np.array([0.0, 3.0])
    wrapped = wrap(np.array([3.0, 3.0 + 3.0 * 100.0 * DEG]))  # 300 deg in one step
    unwrapped, residual = smp.predictive_unwrap(t, wrapped, 100.0 * DEG)
    assert len(residual) == 1
    assert unwrapped[1] - unwrapped[0] == pytest.approx(300.0 * DEG, abs=1e-12)


def test_unwrap_planet_longitude_venus_coarse_step():
    # 585.19 deg per 1-yr step: the constant-rate model recovers every turn
    t = np.arange(0.0, 500.0, 1.0)
    true = 1.3 + N_VENUS * t
    unwrapped = smp.unwrap_planet_longitude(t, wrap(true), N_VENUS)
    np.testing.assert_allclose(unwrapped - unwrapped[0], true - true[0], atol=1e-8)


def test_revolutions_hand_and_constant():
    assert smp.revolutions(np.array([0.0, 1.0, 5 * TWO_PI])) == pytest.approx(5.0)
    assert smp.revolutions(np.array([1.0, -3 * np.pi])) == pytest.approx((-3 * np.pi - 1.0) / TWO_PI)
    assert smp.revolutions(np.full(10, 2.2)) == 0.0


def test_perihelion_jitter_synthetic():
    # e = 0.05 + 0.002 sin(2 pi t / 5) + 0.01 sin(2 pi t / 1000): only the 5-yr wave is above the
    # 50-yr band; its std is 0.002/sqrt(2) = 0.0014142, mean e = 0.05 (two whole slow cycles).
    # |p| = 3: jitter = 3 * 0.0014142 / 0.05 = 0.084853
    t = np.arange(0.0, 2000.0, 0.25)
    e = 0.05 + 0.002 * np.sin(TWO_PI * t / 5.0) + 0.01 * np.sin(TWO_PI * t / 1000.0)
    jitter = smp.perihelion_jitter(t, e, -3, 50.0, 2)
    assert jitter == pytest.approx(3 * 0.002 / np.sqrt(2) / 0.05, rel=0.05)
    assert smp.perihelion_jitter(t, e, -6, 50.0, 2) == pytest.approx(2 * jitter, rel=1e-9)
    assert smp.perihelion_jitter(t, e, 0, 50.0, 2) == pytest.approx(0.0, abs=1e-15)
    assert smp.perihelion_jitter(t, np.full_like(t, 0.05), -3, 50.0, 2) == pytest.approx(0.0, abs=1e-12)


def test_safe_steps_hand():
    # plain: 0.5 pi / max|nu| = 90 deg / 20 deg/yr = 4.5 yr
    # predictive: 0.5 a^2.5 / (3 |k| da) = 0.5 * 9.882118 / (9 * 0.001) = 549.006 yr
    plain, pred = smp.safe_steps(-20.0 * DEG, 10.0 * DEG, 2.5, -3, 0.001, 0.5)
    assert plain == pytest.approx(4.5, rel=1e-12)
    assert pred == pytest.approx(0.5 * 2.5**2.5 / 0.009, rel=1e-12)
    # consistency: at the predictive safe step the Nyquist half-width is twice da
    assert smp.nyquist_halfwidth(2.5, -3, pred) == pytest.approx(0.002, rel=1e-12)


@pytest.mark.parametrize(
    'safe_plain, safe_pred, p_cyc, step, expected',
    [
        (np.inf, 3.0, 100.0, 1000.0, 2.5),  # min(3, 100/20 = 5) = 3 -> 2.5
        (np.inf, 100.0, 30.0, 1000.0, 1.0),  # min(100, 1.5) = 1.5 -> 1.0
        (np.inf, 2.0, 1000.0, 1000.0, 2.0),  # exactly a round value is allowed (<=)
        (np.inf, 500.0, 1e6, 1000.0, 10.0),  # capped by the largest round value
        (np.inf, 0.01, 1000.0, 1000.0, None),  # nothing fits
        (0.3, 500.0, 1e6, 1000.0, 0.25),  # the plain (alias) bound: 0.3 -> 0.25
        (np.inf, 500.0, 1e6, 0.6, None),  # the record's 0.6 yr meets the bounds: no step, never the coarser 10 yr
        (0.07, 500.0, 1e6, 0.2, 0.05),  # the alias bound below the record's step: the round value under it
        (0.04, 500.0, 1e6, 0.2, None),  # nothing in the list under 0.04 yr
    ],
)
def test_resample_step_hand(safe_plain, safe_pred, p_cyc, step, expected):
    settings = settings_from_config()
    got = smp.resample_step(safe_plain, safe_pred, p_cyc, step, settings)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_aliased_lines_hand_and_periodogram():
    # P_syn = 1.1 yr, dt = 1 yr: m = 1: |0.90909 - 1| -> 11 yr; m = 2: |1.81818 - 2| -> 5.5 yr;
    # m = 3: |2.72727 - 3| -> 3.6667 yr. Band [P_cyc/2, 2 P_cyc] = [5, 20] keeps 11 and 5.5.
    got = smp.aliased_lines({'Jupiter': 1.1}, 1.0, 10.0, 3)
    np.testing.assert_allclose(sorted(got), [5.5, 11.0], rtol=1e-9)
    # with one harmonic only 11 yr
    np.testing.assert_allclose(smp.aliased_lines({'Jupiter': 1.1}, 1.0, 10.0, 1), [11.0], rtol=1e-9)
    # the m = 1 line is what a sampled wave shows: periodogram peak of cos(2 pi t / 1.1) at dt = 1
    t = np.arange(0.0, 1100.0, 1.0)
    power = np.abs(np.fft.rfft(np.cos(TWO_PI * t / 1.1))) ** 2
    freq = np.fft.rfftfreq(len(t), 1.0)
    assert 1.0 / freq[np.argmax(power[1:]) + 1] == pytest.approx(11.0, rel=1e-6)
    # P_syn = dt exactly folds to zero frequency (infinite period): never in a finite band
    assert tuple(smp.aliased_lines({'Jupiter': 2.0}, 2.0, 10.0, 1)) == ()


# ----------------------------------------------------------------------------------------------
# angle reconstruction and the whole sampling step on a synthetic record
# ----------------------------------------------------------------------------------------------


def _synthetic_series(a_body, dt, n_rows, angle_offset=0.0, with_planet=False, kink=None):
    """A body on a fixed circular-ish orbit (a, e, varpi constant) and the 2:1 angle built exactly."""
    spec = spec_2_1_jupiter()
    t = np.arange(n_rows) * dt
    n_body = kepler_n(a_body)
    mean_longitude = 0.9 + n_body * t
    jupiter_longitude = 0.1 + N_JUPITER * t
    varpi = np.full_like(t, 0.6)
    sigma = 2 * jupiter_longitude - mean_longitude - varpi + angle_offset
    if kink is not None:
        sigma[kink] += 0.01
    planets = {}
    if with_planet:
        planets['Jupiter'] = _planet('Jupiter', t, mean_longitude=wrap(jupiter_longitude))
    series = MMRSeries(
        t_yr=t,
        a=np.full_like(t, a_body),
        e=np.full_like(t, 0.1),
        inc=np.full_like(t, 0.05),
        Omega=np.full_like(t, 1.0),
        omega=wrap(varpi - 1.0),
        varpi=varpi,
        angle_wrapped=wrap(sigma),
        resonance=spec,
        regime=Regime(step_yr=dt, span_yr=t[-1] - t[0], n_rows=n_rows),
        planets=planets,
        mean_longitude=wrap(mean_longitude),
        angle_unwrapped_numpy=np.unwrap(wrap(sigma)),
        name='synthetic',
    )
    true_rate = 2 * N_JUPITER - n_body
    return series, true_rate


def test_angle_reconstruction_error():
    # offset 3.0 rad pushes the angle across +-pi; only the constant offset must be removed
    series, _ = _synthetic_series(2.5, 0.5, 400, angle_offset=3.0, with_planet=True)
    assert smp.angle_reconstruction_error(series) == pytest.approx(0.0, abs=1e-8)
    kinked, _ = _synthetic_series(2.5, 0.5, 400, angle_offset=3.0, with_planet=True, kink=123)
    assert smp.angle_reconstruction_error(kinked) == pytest.approx(0.01, abs=1e-3)
    no_planets, _ = _synthetic_series(2.5, 0.5, 400)
    assert np.isnan(smp.angle_reconstruction_error(no_planets))


def test_build_context_synthetic_record():
    settings = settings_from_config()
    series, true_rate = _synthetic_series(2.5, 0.5, 4001)
    context = ctx.build_context(series, settings)
    assert context.a_mean_au == pytest.approx(2.5)
    assert context.mean_motion_rad_yr == pytest.approx(kepler_n(2.5), rel=1e-12)
    # c = -1.5 * (-1) * n / a
    assert context.identity_coefficient == pytest.approx(1.5 * kepler_n(2.5) / 2.5, rel=1e-12)
    assert context.planet_mean_motion_source['Jupiter'] == 'table'
    assert context.planet_mean_motion_rad_yr['Jupiter'] == pytest.approx(N_JUPITER, rel=1e-9)
    # constant varpi: no slow term; nu = 2 n_J - n(2.5) = -30.3743 deg/yr
    assert context.slow_rate_rad_yr == pytest.approx(0.0, abs=1e-12)
    assert context.rate_expected_rad_yr == pytest.approx(true_rate, rel=1e-9)
    assert context.rate_expected_lo_rad_yr == pytest.approx(true_rate, rel=1e-9)
    assert context.rate_expected_hi_rad_yr == pytest.approx(true_rate, rel=1e-9)
    assert context.exact_axis_au == pytest.approx(3.2765, abs=2e-4)
    assert context.axis_from_mean_longitude_rate_au == pytest.approx(2.5, rel=1e-6)
    assert context.synodic_periods_yr['Jupiter'] == pytest.approx(TWO_PI / (kepler_n(2.5) - N_JUPITER), rel=1e-9)
    assert context.order == 1
    # P_orb = a^1.5 = 3.9528 yr; dt / P_orb = 0.12649
    assert context.orbital_period_yr == pytest.approx(2.5**1.5, rel=1e-9)
    assert context.fair_step_fraction == pytest.approx(0.5 / 2.5**1.5, rel=1e-9)
    assert context.mu_scale_no_eq == pytest.approx(const.PLANETS_MASS['Jupiter'], rel=1e-9)


@pytest.mark.parametrize(
    'dt, verdict, m',
    [
        (0.5, StepVerdict.RESOLVED, 0),  # -15.19 deg per step
        (10.0, StepVerdict.ALIASED_UNAMBIGUOUS, -1),  # -303.74 deg per step
    ],
)
def test_analyse_sampling_synthetic_circulation(dt, verdict, m):
    settings = settings_from_config()
    n_rows = int(round(20000.0 / dt)) + 1
    series, true_rate = _synthetic_series(2.5, dt, n_rows)
    context = ctx.build_context(series, settings)
    sampling = smp.analyse_sampling(series, context, settings)
    span = (n_rows - 1) * dt

    assert sampling.step_yr == pytest.approx(dt, rel=1e-12)
    assert sampling.span_yr == pytest.approx(span, rel=1e-12)
    assert sampling.n_rows == n_rows
    assert sampling.nyquist_rad_yr == pytest.approx(np.pi / dt, rel=1e-12)
    assert sampling.verdict == verdict
    assert tuple(sampling.branches) == (m,)
    assert sampling.step_rotation_max_rad == pytest.approx(abs(true_rate) * dt, rel=1e-9)
    assert sampling.rate_apparent_pred_rad_yr == pytest.approx(true_rate - TWO_PI * m / dt, rel=1e-9)
    # the predictive unwrap is exact here: revolutions = nu * span / 2 pi (-1687.5 revolutions)
    assert sampling.unwrap_ok
    assert sampling.unwrap_resid_max_rad < 1e-6
    assert sampling.revs_predictive == pytest.approx(true_rate * span / TWO_PI, abs=1e-6)
    assert context.rate_measured_rad_yr == pytest.approx(true_rate, rel=1e-9)
    assert sampling.alias_numpy_differs == (m != 0)
    if m != 0:
        # np.unwrap only sees the apparent rate: 5.6257 deg/yr * 20 kyr / 360 = 312.5 revolutions
        assert sampling.revs_numpy == pytest.approx(sampling.rate_apparent_pred_rad_yr * span / TWO_PI, abs=0.01)
    # Nyquist half-width a^2.5 / (3 |k| dt); a constant a = 2.5 au never reaches a* = 3.28 au: the far ratio (distance
    # over the a scale of the record, here only the round-off of a(lambda_dot)) is enormous
    assert sampling.nyquist_halfwidth_au == pytest.approx(2.5**2.5 / (3 * dt), rel=1e-12)
    assert sampling.far_ratio > 1e6
    assert sampling.far_from_resonance
    # constant e: no perihelion jitter
    assert sampling.jitter == pytest.approx(0.0, abs=1e-9)
    assert not sampling.jitter_low_e
    assert not sampling.undefined_phase
    columns = sampling.to_columns()
    assert columns['smp_nyquist_deg_yr'] == pytest.approx(180.0 / dt, rel=1e-9)
    assert columns['smp_step_yr'] == pytest.approx(dt, rel=1e-12)
    assert columns['smp_verdict'] == str(verdict)
