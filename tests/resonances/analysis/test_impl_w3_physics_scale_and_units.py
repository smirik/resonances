"""W3 F3, G9, G10: the strength scale is not a bound; numbers moved to settings; one year for g/s.

F3 (Codex astra §5-6; Gallardo 2006 p. 30, N&M 1998 p. 3031): mu e^q has no coefficient of the disturbing function,
so far_from_resonance is read from the data (a libration about a* needs the slow a to reach a*) and the strength
flag fires only three orders of magnitude off. The far ratio of 1865 Cerberus 10E-11 (benchmark v2: <a> = 1.079994,
a* = 1.065601, a range 1.078557..1.080324, a(lambda_dot) = 1.080037) is 16.29 from these numbers, where the old
mu-based ratio was 1.05 (invalid/alias against the author's non_resonant).

G10: the analysis year is the Gaussian 2 pi / k = 365.2568983 d; JD 2461200.5 is 9655.5 d = 26.43482 such years
after J2000 (26.43532 Julian).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.data import const
from resonances.resonance.analysis.mmr import context as ctx
from resonances.resonance.analysis.mmr import sampling as smp
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr import subresonances as sub
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config

SETTINGS = settings_from_config()
N_JUPITER = math.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'])
A_STAR_2_1 = (2 * math.pi / (2 * N_JUPITER)) ** (2 / 3)  # n(a*) = 2 n_J: 3.2776 au
ARCSEC = math.pi / 180 / 3600


def _series(a, step=1.0, spec='2J-1'):
    t = np.arange(len(a)) * step
    zeros = np.zeros_like(t)
    return MMRSeries(
        t_yr=t,
        a=np.asarray(a, dtype=float),
        e=zeros + 0.1,
        inc=zeros + 0.05,
        Omega=zeros + 1.0,
        omega=zeros + 0.5,
        varpi=zeros + 1.5,
        angle_wrapped=np.mod(0.01 * t, 2 * math.pi),
        resonance=ResonanceSpec.from_key(spec),
        regime=Regime(step_yr=step, span_yr=t[-1], n_rows=len(t)),
    )


# --- F3: far from the resonance by the data ---------------------------------------------------------------


def test_far_ratio_of_1865_from_its_numbers():
    context = SimpleNamespace(
        exact_axis_au=1.065601, a_mean_au=1.079994, a_p01_au=1.078557, a_p99_au=1.080324, axis_from_mean_longitude_rate_au=1.080037
    )
    assert smp.far_ratio(context) == pytest.approx(0.014393 / 0.0008835, rel=1e-9)
    assert smp.far_ratio(context) == pytest.approx(16.29, abs=0.01)


@pytest.mark.parametrize('offset, expected', [(0.0, 0.0), (0.05, 0.05 / (0.01 * math.cos(0.01 * math.pi)))])
def test_far_ratio_of_a_librating_and_of_a_shifted_body(offset, expected):
    # a = a* + offset + 0.01 sin(2 pi t / 600): the 1..99 % range of a sine is +-0.01 cos(0.01 pi)
    t = np.arange(0.0, 60000.0, 1.0)
    series = _series(A_STAR_2_1 + offset + 0.01 * np.sin(2 * math.pi * t / 600.0))
    context = ctx.build_context(series, SETTINGS)
    assert context.exact_axis_au == pytest.approx(A_STAR_2_1, rel=1e-12)
    sampling = smp.analyse_sampling(series, context, SETTINGS)
    assert sampling.far_ratio == pytest.approx(expected, rel=2e-3, abs=2e-3)
    assert not sampling.far_from_resonance  # a librator never reads as far; a* five half-ranges away is not "far" either
    # no strength enters: a planet a thousand times heavier changes nothing
    context.mu_scale_no_eq *= 1e3
    assert smp.far_ratio(context) == sampling.far_ratio


def test_far_ratio_edge_cases():
    base = dict(exact_axis_au=2.0, a_mean_au=2.1, a_p01_au=2.1, a_p99_au=2.1, axis_from_mean_longitude_rate_au=float('nan'))
    assert smp.far_ratio(SimpleNamespace(**base)) == math.inf  # a constant a away from a*
    assert math.isnan(smp.far_ratio(SimpleNamespace(**{**base, 'a_mean_au': 2.0, 'a_p01_au': 2.0, 'a_p99_au': 2.0})))
    # a lambda_dot that puts the body at a* keeps the ratio ~1 even with a narrow a range (model error scale)
    assert smp.far_ratio(SimpleNamespace(**{**base, 'axis_from_mean_longitude_rate_au': 2.0})) == pytest.approx(1.0)


def test_strength_flag_only_three_orders_of_magnitude_off():
    assert SETTINGS.mmr_strength_implausible_ratio == 1000.0 and SETTINGS.mmr_strength_check_ratio == 10.0
    # 2003 EH16 1M-3J-1 (resonant): mu_eff 7.811099e-08 over mu_M mu_J 3.080625e-10 = 253.6, e^q not legal: plausible
    ratio = 7.811099e-08 / 3.080625e-10
    assert ratio == pytest.approx(253.56, abs=0.01)
    assert not st.implausible_strength(False, ratio, SETTINGS)
    assert st.implausible_strength(False, 1000.0, SETTINGS) and st.implausible_strength(True, 1000.0, SETTINGS)
    assert not st.implausible_strength(True, 999.0, SETTINGS)
    # far below: implausible only against a legal e^q (without it the scale is mu alone, an over-estimate)
    assert st.implausible_strength(True, 1e-3, SETTINGS) and not st.implausible_strength(False, 1e-3, SETTINGS)
    assert not st.implausible_strength(True, float('nan'), SETTINGS)
    # the support check keeps its order-of-magnitude band
    assert st.strength_check(True, 11.4, SETTINGS).value == 'fail' and st.strength_check(True, 9.0, SETTINGS).value == 'pass'


def test_body_coefficient_without_a_finite_mean_motion():
    context = SimpleNamespace(body_coefficient=0, identity_coefficient=float('nan'), a_mean_au=-1.0, mean_motion_rad_yr=float('nan'))
    assert st.body_coefficient(context) == 0
    context.body_coefficient = -3
    assert st.body_coefficient(context) == -3


# --- G9: the numbers of context and sampling are settings ---------------------------------------------------


def test_axis_range_percentile_setting():
    a = np.linspace(2.0, 3.0, 1001)
    series = _series(a)
    default = ctx.build_context(series, SETTINGS)
    assert (default.a_p01_au, default.a_p99_au) == pytest.approx((2.01, 2.99))
    wide = ctx.build_context(series, settings_from_config(mmr_axis_range_percentile=10.0))
    assert (wide.a_p01_au, wide.a_p99_au) == pytest.approx((2.1, 2.9))


def test_planet_unwrap_step_setting():
    t = np.arange(0.0, 2000.0, 1.0)
    longitude = N_JUPITER * t + 0.2
    longitude[1000:] += math.radians(100.0)  # one step errs by 100 deg against the table rate
    planet = PlanetSeries('Jupiter', t, t * 0 + 5.2, t * 0 + 0.05, t * 0, t * 0, t * 0, np.mod(longitude, 2 * math.pi))
    values, sources = ctx.planet_mean_motions({'Jupiter': planet}, ('Jupiter',))
    assert sources == {'Jupiter': 'table'} and values['Jupiter'] == N_JUPITER
    values, sources = ctx.planet_mean_motions({'Jupiter': planet}, ('Jupiter',), max_step_error_deg=120.0)
    assert sources == {'Jupiter': 'measured'} and values['Jupiter'] != N_JUPITER


def test_aliased_line_band_setting():
    # P_syn = 1 / (1 + 1/500) at dt = 1 folds to 500 yr: outside [600, 2400] for P_cyc = 1200, inside [400, 3600]
    synodic = {'Jupiter': 1 / (1 + 1 / 500)}
    assert smp.aliased_lines(synodic, 1.0, 1200.0, 1) == ()
    assert smp.aliased_lines(synodic, 1.0, 1200.0, 1, band_factor=3.0) == pytest.approx((500.0,), rel=1e-9)
    assert SETTINGS.mmr_aliased_line_band_factor == 2.0


# --- G10: one year for the planets' mode longitudes ---------------------------------------------------------


def test_model_years_since_j2000_are_gaussian():
    assert sub.GAUSSIAN_YEAR_DAYS == pytest.approx(365.2568983, abs=1e-6)
    assert sub.model_years_since_j2000(0.0, 2461200.5) == pytest.approx(9655.5 / 365.2568983, rel=1e-9)
    assert sub.model_years_since_j2000(0.0, 2461200.5) == pytest.approx(26.43482, abs=1e-5)
    assert sub.model_years_since_j2000(5.0, float('nan')) == 5.0  # unknown epoch: J2000


def test_mode_longitude_uses_t_as_given():
    epoch = 2461200.5
    g6 = const.PLANETARY_FREQUENCIES['g6'] * ARCSEC
    phase = math.radians(const.PLANETARY_PHASES['g6'])
    values = sub._mode_longitude(np.array([0.0, 1e6]), 'Saturn', 'g', epoch)
    assert values[0] == pytest.approx(g6 * 9655.5 / 365.2568983 + phase, rel=1e-12)
    assert values[1] - values[0] == pytest.approx(g6 * 1e6, rel=1e-12)  # no rescaling of the model time
