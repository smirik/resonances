"""Independent tests of the MMR subresonance check (analysis/mmr/subresonances.py) and of the multiplet
coefficients (analysis/mmr/multiplet.py).

Written from tmp/classifier-v2-2026-10-15/impl/API.md, the public signatures/docstrings, SPEC §6,
SPEC-DELTA items 2, 3, 13 and research/SISTERS.md §§1-3 only (no implementation bodies read).

Expected values come from independent sources:
- Murray & Dermott (1999) disturbing-function coefficients computed here from Laplace coefficients by quadrature
  (appendix B formulas, including the indirect terms of the 2:1 e' term and of the 3:1 e'^2 term);
- closed forms (P(beta) = sum C_j e^{i j beta}, the Rouche bound arcsin(sum/|C_N|), the argument principle,
  the binomial mode, the pendulum T0, the member spacing of SISTERS §1.1 with its table values);
- synthetic records with a known truth: sigma_N an integrated pendulum about pi, beta = varpi_P - varpi
  rotating at the secular rates of 31935 (g = 31.75"/yr, g_Mars = 18.43"/yr), sigma_0 = sigma_N - N beta, and a
  built from sigma_0 by the identity sigma_0_dot = 9 n_Mars - 16 n + 7 g.
"""

import dataclasses
import math
import typing

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from resonances.resonance.analysis.geometry import (
    CentreTrack,
    Episode,
    EpisodeShape,
    Geometry,
    RejectReason,
    TurningPoints,
)
from resonances.resonance.analysis.mmr import multiplet as mp
from resonances.resonance.analysis.mmr import subresonances as sr
from resonances.resonance.analysis.mmr.models import (
    ForceWindow,
    MMRSeries,
    PlanetSeries,
    Regime,
    ResonanceSpec,
    Response,
    SubresonanceCandidate,
    SubresonanceTrigger,
)
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, Status

pytestmark = pytest.mark.slow  # a shared setup of more than 1 s (tests/CLAUDE.md)

TWO_PI = 2.0 * np.pi
DEG = np.pi / 180.0
ARCSEC = DEG / 3600.0
NAN = float('nan')

# secular rates of 31935 and Mars (SISTERS §1.1 table) and the mean motion of Mars (API.md §2.1)
G_BODY = 31.75 * ARCSEC  # rad/yr
G_MARS = 18.43 * ARCSEC
N_MARS = 191.4066 * DEG  # rad/yr
MU_MARS = 3.227e-7


@pytest.fixture(scope='module')
def settings():
    return settings_from_config()


# ----------------------------------------------------------------------------------------------
# independent references: Laplace coefficients and Murray & Dermott f_d
# ----------------------------------------------------------------------------------------------


def laplace_b(j, alpha, s=0.5):
    """b_s^(j)(alpha) = (1/pi) int_0^2pi cos(j psi) / (1 - 2 alpha cos psi + alpha^2)^s dpsi."""
    value, _ = quad(lambda p: np.cos(j * p) / (1.0 - 2.0 * alpha * np.cos(p) + alpha * alpha) ** s, 0.0, TWO_PI, limit=200)
    return value / np.pi


def laplace_d(j, alpha, n, h=1e-4):
    if n == 1:
        return (laplace_b(j, alpha + h) - laplace_b(j, alpha - h)) / (2 * h)
    return (laplace_b(j, alpha + h) - 2 * laplace_b(j, alpha) + laplace_b(j, alpha - h)) / h**2


def md_f_21(alpha):
    """2:1 (M&D j = 2): f27 (e) and f31 + indirect -2 alpha (e'); M&D Table 8.5 gives -1.190 and 0.428."""
    f27 = 0.5 * (-4.0 * laplace_b(2, alpha) - alpha * laplace_d(2, alpha, 1))
    f31 = 0.5 * (3.0 * laplace_b(1, alpha) + alpha * laplace_d(1, alpha, 1)) - 2.0 * alpha
    return f27, f31


def md_f_31(alpha):
    """3:1 (M&D j = 3): f45 (e^2), f49 (e e'), f53 - 27/8 alpha (e'^2, indirect for an exterior perturber)."""
    a, d1, d2 = alpha, laplace_d, laplace_d
    f45 = (21 * laplace_b(3, a) + 10 * a * d1(3, a, 1) + a * a * d2(3, a, 2)) / 8
    f49 = (-20 * laplace_b(2, a) - 10 * a * d1(2, a, 1) - a * a * d2(2, a, 2)) / 4
    f53 = (17 * laplace_b(1, a) + 10 * a * d1(1, a, 1) + a * a * d2(1, a, 2)) / 8 - 27.0 / 8.0 * a
    return f45, f49, f53


def test_reference_f_values_match_murray_dermott():
    # sanity of the references themselves against the printed M&D Table 8.5 values
    f27, f31 = md_f_21(0.5 ** (2 / 3))
    assert f27 == pytest.approx(-1.19049, abs=2e-4)
    assert f31 == pytest.approx(0.42839, abs=2e-4)
    f45, f49, _ = md_f_31((1 / 3) ** (2 / 3))
    assert f45 == pytest.approx(0.5988, abs=2e-4)
    assert f49 == pytest.approx(-2.2130, abs=2e-3)


# ----------------------------------------------------------------------------------------------
# multiplet.py: closed forms
# ----------------------------------------------------------------------------------------------


def test_composite_closed_form():
    # P(beta) = 1 + 0.5 e^{i beta}: 1.5, 1 + 0.5i, 0.5, 1 - 0.5i
    beta = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    p = mp.composite(np.array([1.0, 0.5]), np.array([0, 1]), beta)
    np.testing.assert_allclose(p, [1.5, 1 + 0.5j, 0.5, 1 - 0.5j], atol=1e-12)


def test_composite_time_dependent_coefficients():
    # (n_t, n_j) coefficients: row t uses its own C_j
    coef = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    beta = np.array([0.3, np.pi / 2, np.pi])
    p = mp.composite(coef, np.array([0, 1]), beta)
    np.testing.assert_allclose(p, [1.0, 2j, 2.0], atol=1e-12)


def test_composite_centre_single_member_rotates_with_beta():
    # only member j = 1 (C_1 = 1): P = e^{i beta}, c0 = pi - beta exactly, unwrapped over two turns
    beta = np.linspace(0.0, 4 * np.pi, 801)
    c0 = mp.composite_centre(np.array([0.0, 1.0]), np.array([0, 1]), beta)
    np.testing.assert_allclose(np.diff(c0), -np.diff(beta), atol=1e-9)
    assert math.remainder(c0[0] - np.pi, TWO_PI) == pytest.approx(0.0, abs=1e-9)


def test_composite_centre_rouche_bound_is_reached():
    # C_0 = 1, C_1 = 0.3: arg(1 + 0.3 e^{i beta}) peaks at arcsin(0.3) = 17.4576 deg (tangent from the origin),
    # so c0 swings about pi by exactly +-arcsin(0.3) and does not wind
    beta = np.linspace(0.0, TWO_PI, 20001)
    c0 = mp.composite_centre(np.array([1.0, 0.3]), np.array([0, 1]), beta)
    dev = np.array([math.remainder(x - np.pi, TWO_PI) for x in c0])
    assert np.max(np.abs(dev)) == pytest.approx(math.asin(0.3), abs=1e-5)
    assert abs(c0[-1] - c0[0]) < 1e-9
    # a negative coefficient puts the centre at 0: P = -1, c0 = pi - pi = 0
    c_neg = mp.composite_centre(np.array([-1.0]), np.array([0]), np.array([0.0, 1.0]))
    assert math.remainder(c_neg[0], TWO_PI) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    'coef, j, expected',
    [
        ([1.0, 0.3], [0, 1], 0),  # the j = 0 term dominates: no winding
        ([0.3, 1.0], [0, 1], 1),  # 0.3 + w: one zero (w = -0.3) inside the unit circle
        ([0.2, 0.3, 1.0], [0, 1, 2], 2),  # j = 2 dominates (Rouche): two zeros inside
        ([0.5, 1.2, 1.0], [0, 1, 2], 2),  # w^2 + 1.2 w + 0.5: complex pair, |w|^2 = 0.5: both inside, no member dominates
        ([1.0, -2.5, 1.0], [0, 1, 2], 1),  # (w - 2)(w - 0.5): one zero inside
        ([1.0, 0.2], [-1, 0], -1),  # e^{-i beta} + 0.2: exterior-planet member j = -1 dominates
    ],
)
def test_winding_number_argument_principle(coef, j, expected):
    assert mp.winding_number(np.array(coef), np.array(j)) == expected


def test_rouche_holds_strict_inequality():
    c = np.array([1.0, 0.3, 0.6])
    assert mp.rouche_holds(c, np.array([0, 1, 2]), 0)  # 1 > 0.9
    assert not mp.rouche_holds(c, np.array([0, 1, 2]), 2)  # 0.6 < 1.3
    assert not mp.rouche_holds(np.array([1.0, 0.5, 0.5]), np.array([0, 1, 2]), 0)  # 1 = 1: not strictly
    assert mp.rouche_holds(np.array([-1.0, 0.4, 0.5]), np.array([0, 1, 2]), 0)  # magnitudes: |-1| > 0.9
    # the prototypes: Rouche fails (sum of the others 1.8 |C_N|)
    assert not mp.rouche_holds(np.array([0.5, 1.0, 0.91, 0.39]), np.array([0, 1, 2, 3]), 1)


@pytest.mark.parametrize(
    'order, e, ep, ratio, expected',
    [
        # 31935 9M-16, q = 7: x = 0.72 * 0.093 / 0.159 = 0.42113; 8 x / (1 + x) = 2.3707 -> 2 (SISTERS: j = 2)
        (7, 0.159, 0.093, 0.72, 2),
        # 65366 6M-11, q = 5: x = 0.71 * 0.099 / 0.143 = 0.49154; 6 x / (1 + x) = 1.9773 -> 1 (SISTERS: j = 1)
        (5, 0.143, 0.099, 0.71, 1),
        # 3J-1, q = -2, e = 0.15, e_J = 0.048, c = 1.85: x = 0.592; 3 x / 1.592 = 1.1156 -> -1 (sign of q kept)
        (-2, 0.15, 0.048, 1.85, -1),
        # x = 0.2499 just below 1/|q| = 0.25 for q = 4: 5 x / (1 + x) = 0.9996 -> 0
        (4, 1.0, 0.2499, 1.0, 0),
        # x = 0.2501 just above: 1.0004 -> 1
        (4, 1.0, 0.2501, 1.0, 1),
        # e = 0: only the planet term is left -> q
        (3, 0.0, 0.05, 1.0, 3),
        # not finite -> 0
        (3, NAN, 0.05, 1.0, 0),
    ],
)
def test_binomial_mode_hand_values(order, e, ep, ratio, expected):
    assert mp.binomial_mode(order, e, ep, ratio) == expected


@pytest.mark.parametrize('q', [1, 2, 3, 5, 7, 11])
def test_binomial_mode_is_argmax_of_binomial_terms(q):
    # brute force: the largest of C(q, j) x^j (avoid exact ties at x = (j+1)/(q-j))
    ties = {(j + 1) / (q - j) for j in range(q)}
    for x in np.geomspace(0.01, 30.0, 97):
        if min(abs(x - t) for t in ties) < 1e-6:
            continue
        terms = [math.comb(q, j) * x**j for j in range(q + 1)]
        assert mp.binomial_mode(q, 1.0, x, 1.0) == int(np.argmax(terms)), x
        assert mp.binomial_mode(-q, 1.0, x, 1.0) == -int(np.argmax(terms)), x


def test_mu_eff_and_small_period_closed_form():
    # 2 C cos sigma with |C| = 5e-7, a = 1: mu_eff = 1e-6; omega0 = |k| n sqrt(3e-6), n = 2 pi rad/yr
    # T0 = 2 pi / (2 pi * 1.7320508e-3) = 577.350 yr for |k| = 1, 288.675 yr for |k| = 2
    assert mp.mu_eff(5e-7, 1.0) == pytest.approx(1e-6, rel=1e-12)
    assert mp.small_period_yr(5e-7, 1.0, 1) == pytest.approx(577.3503, rel=1e-6)
    assert mp.small_period_yr(5e-7, 1.0, -2) == pytest.approx(288.6751, rel=1e-6)
    # a = 4: n = 2 pi / 8; mu_eff = 2 * 5e-7 * 4 = 4e-6; T0 = 8 / sqrt(1.2e-5) = 2309.40 yr
    assert mp.small_period_yr(5e-7, 4.0, 1) == pytest.approx(2309.401, rel=1e-6)


@pytest.mark.parametrize(
    'a, k, g, gp, expected',
    [
        # SISTERS §1.1 table: 65366 6M-11 -> 6.3e-6 au, 31935 9M-16 -> 3.2e-6 au
        # hand: n = 2 pi / 2.2822^1.5 = 1.82245; (2/3)(2.2822/1.82245)(17.11"/yr)/11 = 6.2956e-6
        (2.2822, -11, 35.54, 18.43, 6.2956e-6),
        # n = 2 pi / 2.2359^1.5 = 1.87929; (2/3)(2.2359/1.87929)(13.32"/yr)/16 = 3.2014e-6
        (2.2359, -16, 31.75, 18.43, 3.2014e-6),
    ],
)
def test_member_spacing_sisters_table(a, k, g, gp, expected):
    assert mp.member_spacing_au(a, g * ARCSEC, gp * ARCSEC, k) == pytest.approx(expected, rel=2e-3)


def test_crossing_orbits():
    assert mp.crossing(1.0, 0.5, 1.4, 0.0)  # aphelion 1.5 > 1.4
    assert mp.crossing(1.4, 0.0, 1.0, 0.5)  # the same, roles swapped
    assert not mp.crossing(2.0, 0.1, 5.2, 0.05)  # 2.2 < 4.94
    assert not mp.crossing(2.2359, 0.159, 1.5237, 0.093)  # 31935: 1.880 > 1.665


def test_cache_key_is_deterministic_and_identifies_its_inputs():
    k1 = mp.cache_key('9M-16+0+7', 0.6815, 0.159, 0.093)
    assert k1 == mp.cache_key('9M-16+0+7', 0.6815, 0.159, 0.093)
    assert '9M-16+0+7' in k1
    assert str(mp.MULTIPLET_VERSION) in k1
    assert k1 != mp.cache_key('9M-16+0+7', 0.6815, 0.259, 0.093)
    assert k1 != mp.cache_key('9M-16+0+7', 0.6815, 0.159, 0.193)
    assert k1 != mp.cache_key('9M-16+0+7', 0.5815, 0.159, 0.093)
    assert k1 != mp.cache_key('6M-11+0+5', 0.6815, 0.159, 0.093)


def test_signed_order_and_planar_members():
    assert mp.signed_order(9, -16) == 7
    assert mp.signed_order(3, -1) == -2
    np.testing.assert_array_equal(mp.planar_members(9, -16), np.arange(0, 8))
    np.testing.assert_array_equal(np.sort(mp.planar_members(3, -1)), [-2, -1, 0])


# ----------------------------------------------------------------------------------------------
# multiplet.py: the averaging against Murray & Dermott
# ----------------------------------------------------------------------------------------------


def _coefficient(js, cs, j):
    return cs[list(np.asarray(js)).index(j)]


def test_resonant_coefficients_21_jupiter_matches_md():
    # 2J-1: sigma0 = 2 lambda_J - lambda - varpi (j = 0), sigma_-1 = 2 lambda_J - lambda - varpi_J.
    # R = (mu / a_P) [f27 e cos sigma0 + (f31 - 2 alpha) e_P cos sigma_-1] = sum 2 C_j cos(...)
    # => C_0 = mu f27 e / (2 a_P), C_-1 = mu (f31 - 2 alpha) e_P / (2 a_P)
    a_p = 5.2026
    alpha = 0.5 ** (2 / 3)
    e, ep = 0.01, 0.008
    f27, f31 = md_f_21(alpha)
    js, cs = mp.resonant_coefficients(2, -1, alpha * a_p, a_p, e, ep, 1.0, -1, 0)
    c0 = _coefficient(js, cs, 0)
    cm1 = _coefficient(js, cs, -1)
    assert c0.real == pytest.approx(f27 * e / (2 * a_p), rel=0.01)
    assert cm1.real == pytest.approx(f31 * ep / (2 * a_p), rel=0.02)
    # real up to round-off (a cosine-only expansion)
    assert abs(c0.imag) < 1e-3 * abs(c0.real)
    assert abs(cm1.imag) < 1e-3 * abs(c0.real)
    # SISTERS §1.3 weight c = f31/f27 = 0.36
    assert abs((cm1.real / c0.real) * e / ep) == pytest.approx(0.36, abs=0.01)


def test_resonant_coefficients_31_jupiter_matches_md():
    # 3J-1 (q = -2): C_0 = mu f45 e^2 / (2 a_P) with f45 = 0.5988 (the module's own check gives 0.5987),
    # C_-1 / C_0 = (f49 / f45) e_P / e = -3.696 e_P / e, C_-2 / C_-1 = (f53' / f49) e_P / e = -0.164 e_P / e
    # (SISTERS: steps c = 1.85 and 0.33 of the binomial form; C_-1/C_0 = -1.2 at e = 0.15, e_J = 0.048)
    a_p = 5.2026
    alpha = (1 / 3) ** (2 / 3)
    e, ep = 0.02, 0.01
    f45, f49, f53 = md_f_31(alpha)
    js, cs = mp.resonant_coefficients(3, -1, alpha * a_p, a_p, e, ep, 1.0, -2, 0)
    c0, cm1, cm2 = (_coefficient(js, cs, j).real for j in (0, -1, -2))
    assert 2 * a_p * c0 / e**2 == pytest.approx(f45, rel=0.01)
    assert (cm1 / c0) * e / ep == pytest.approx(f49 / f45, rel=0.02)
    assert (cm2 / cm1) * e / ep == pytest.approx(f53 / f49, rel=0.05)


def test_resonant_coefficients_scale_with_mu_and_eccentricity():
    a_p = 5.2026
    a = (1 / 3) ** (2 / 3) * a_p
    js, c1 = mp.resonant_coefficients(3, -1, a, a_p, 0.01, 0.0, 1e-3, 0, 0)
    _, c2 = mp.resonant_coefficients(3, -1, a, a_p, 0.01, 0.0, 2e-3, 0, 0)
    _, c3 = mp.resonant_coefficients(3, -1, a, a_p, 0.02, 0.0, 1e-3, 0, 0)
    # linear in mu; C_0 ~ e^|q| = e^2 at small e (next term O(e^2) relative: < 1e-3)
    assert c2[0].real / c1[0].real == pytest.approx(2.0, rel=1e-6)
    assert c3[0].real / c1[0].real == pytest.approx(4.0, rel=5e-3)


def test_resonant_coefficients_exterior_body_mars_binomial_weight():
    # 9M-16 (31935, alpha = 1.5237/2.2359 = 0.6815): SISTERS measures c = 0.723-0.728 per step at small e,
    # binomial to 1 %: |C_1 / C_0| = q c e_P / e and |C_2 / C_1| = (q - 1)/2 c e_P / e, q = 7
    a, a_p, e, ep = 2.2359, 1.5237, 0.02, 0.01
    js, cs = mp.resonant_coefficients(9, -16, a, a_p, e, ep, 1.0, 0, 2)
    c0, c1, c2 = (abs(_coefficient(js, cs, j)) for j in (0, 1, 2))
    step1 = c1 / c0 * e / ep / 7
    step2 = c2 / c1 * e / ep / 3
    assert step1 == pytest.approx(0.725, abs=0.01)
    assert step2 == pytest.approx(0.725, abs=0.015)
    assert abs(mp.binomial_ratio(cs, js, 7, e, ep)) == pytest.approx(step1, rel=1e-3)


@pytest.mark.parametrize('e, ep, a', [(1.0, 0.0, 2.0), (-0.1, 0.0, 2.0), (0.1, 1.2, 2.0), (0.1, 0.0, 0.0)])
def test_resonant_coefficients_rejects_bad_elements(e, ep, a):
    with pytest.raises(ValueError):
        mp.resonant_coefficients(3, -1, a, 5.2, e, ep, 1.0, -2, 0)


# ----------------------------------------------------------------------------------------------
# subresonances.py: the members of a multiplet
# ----------------------------------------------------------------------------------------------


def _check_dalembert(spec, member):
    total = sum(member.planet_coefficients) + member.body_coefficient + member.body_varpi_coefficient
    total += sum(member.planet_varpi_coefficients) + member.body_node_coefficient + sum(member.planet_node_coefficients or ())
    assert total == 0
    assert member.planet_coefficients == spec.planet_coefficients
    assert member.body_coefficient == spec.body_coefficient
    assert (member.body_node_coefficient + sum(member.planet_node_coefficients or ())) % 2 == 0


def test_enumerate_two_body_planar_members():
    # 9M-16+0+7: sigma_j = 9 lM - 16 l + (7 - j) varpi + j varpi_M, j = 1..7 (the catalogue j = 0 excluded)
    spec = ResonanceSpec.from_key('9M-16+0+7')
    members = sr.enumerate_subresonances(spec, 7)
    assert len(members) == 7
    assert sorted(j for _, j, _ in members) == list(range(1, 8))
    for member, j, nodal in members:
        assert not nodal
        assert member.body_varpi_coefficient == 7 - j
        assert tuple(member.planet_varpi_coefficients) == (j,)
        _check_dalembert(spec, member)
    keys = {m.key for m, _, _ in members}
    assert '9M-16+2+5' in keys and '9M-16+0+7' not in keys


def test_enumerate_degree_q_plus_one_adds_only_nodal_members_of_degree_q():
    # Parity: the degree of a term has the parity of q, so a largest degree q + 1 adds no planar member; with the nodal
    # members (SPEC §6.3) it adds exactly the nodal members of degree q (see impl/TEST-DISPUTES.md, W2 item 1).
    # Hand count for q = 7: degree 7 = q forces u, u_P, v, v_P >= 0 with u + u_P + v + v_P = 7 and v + v_P = 2m even,
    # m = 1..3: (2m + 1)(8 - 2m) = 18 + 20 + 14 = 52 nodal members. (Same rule at q = 5 gives 12 + 10 = 22 nodal
    # + 6 planar = 28, the SISTERS §5 number.)
    spec = ResonanceSpec.from_key('9M-16+0+7')
    base = {m.key for m, _, _ in sr.enumerate_subresonances(spec, 7)}
    assert {m.key for m, _, _ in sr.enumerate_subresonances(spec, 8)} == base
    one = sr.enumerate_subresonances(spec, 8, nodal=True)
    added = [(m, nodal) for m, _, nodal in one if m.key not in base]
    assert base <= {m.key for m, _, _ in one}
    assert len(added) == 52
    assert all(nodal and sr.member_degree(m) == 7 for m, nodal in added)
    assert {m.key for m, _, n in one if not n} == base
    spec5 = ResonanceSpec.from_key('6M-11+0+5')
    assert len(sr.enumerate_subresonances(spec5, 6, nodal=True)) == 28 - 1  # the catalogue angle excluded


def test_enumerate_degree_q_members_exist_whatever_the_largest_degree():
    # 82831 13M-24 is of order 11 > MATRIX_2BODY_ORDER_MAX 10: its order-11 members exist anyway
    spec11 = ResonanceSpec.from_key('13M-24+0+11')
    members = sr.enumerate_subresonances(spec11, 10)
    assert sorted(j for _, j, _ in members) == list(range(1, 12))
    # the largest degree = q: no member above degree 5, and the planar ones are those of degree 5
    # (nodal members of degree q itself join with nodal=True, SPEC §6.3)
    spec5 = ResonanceSpec.from_key('6M-11+0+5')
    capped = sr.enumerate_subresonances(spec5, 5, nodal=True)
    assert all(sr.member_degree(m) == 5 for m, _, _ in capped)
    assert {m.key for m, _, n in capped if not n} == {m.key for m, _, _ in sr.enumerate_subresonances(spec5, 5)}


def test_enumerate_nodal_members_up_to_degree_q_plus_two():
    # 6M-11+0+5 up to degree q + 2 = 7 with the nodes: nodal members appear, among them SISTERS' 6M-11+0+5[Omega+1, Omega_P-1]
    spec = ResonanceSpec.from_key('6M-11+0+5')
    members = sr.enumerate_subresonances(spec, 7, nodal=True)
    nodal = [(m, j) for m, j, n in members if n]
    assert nodal
    found = False
    for member, _j, is_nodal in members:
        _check_dalembert(spec, member)
        order = (
            abs(member.body_varpi_coefficient)
            + sum(abs(x) for x in member.planet_varpi_coefficients)
            + abs(member.body_node_coefficient)
            + sum(abs(x) for x in (member.planet_node_coefficients or ()))
        )
        assert order in (5, 7)
        nodes = (member.body_node_coefficient, tuple(member.planet_node_coefficients or (0,)))
        assert is_nodal == (nodes != (0, (0,)))
        if nodes == (1, (-1,)) and member.body_varpi_coefficient == 5 and tuple(member.planet_varpi_coefficients) == (0,):
            found = True
    assert found
    # no duplicates, the catalogue angle excluded
    sigs = [
        (m.body_varpi_coefficient, tuple(m.planet_varpi_coefficients), m.body_node_coefficient, tuple(m.planet_node_coefficients or (0,)))
        for m, _, _ in members
    ]
    assert len(sigs) == len(set(sigs))
    assert (5, (0,), 0, (0,)) not in sigs
    # sorted by degree, planar first within the degree
    orders = [sr.member_degree(m) for m, _, _ in members]
    assert orders == sorted(orders)
    # all 5 planar members of degree 5 still there
    assert {j for m, j, n in members if not n and sr.member_degree(m) == 5} == {1, 2, 3, 4, 5}


def test_enumerate_three_body_members():
    # 5J-2S-1, q = 2 (varpi sum -2): (p_J, p_S, p) <= 0 with sum -2: C(4, 2) = 6 terms, 5 without the catalogue
    spec = ResonanceSpec.from_key('5J-2S-1+0+0-2')
    members = sr.enumerate_subresonances(spec, 2)
    assert len(members) == 5
    for member, _, nodal in members:
        assert not nodal
        coeffs = (*member.planet_varpi_coefficients, member.body_varpi_coefficient)
        assert all(c <= 0 for c in coeffs) and sum(coeffs) == -2
        _check_dalembert(spec, member)


def test_enumerate_retrograde_is_empty():
    assert sr.enumerate_subresonances(ResonanceSpec.from_key('1J-3r0'), 10) == []


# ----------------------------------------------------------------------------------------------
# synthetic multiplet record: sigma_N a pendulum about pi, beta rotating
# ----------------------------------------------------------------------------------------------


def pendulum(t, period_yr, amp_rad, centre=np.pi):
    """sigma_ddot = -omega0^2 sin(sigma - centre) from sigma = centre + amp, sigma_dot = 0 (DOP853)."""
    w2 = (TWO_PI / period_yr) ** 2
    sol = solve_ivp(
        lambda _, y: [y[1], -w2 * np.sin(y[0] - centre)],
        (t[0], t[-1]),
        [centre + amp_rad, 0.0],
        t_eval=t,
        method='DOP853',
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.y[0], sol.y[1]


def multiplet_record(n_member, span_yr=100_000.0, step_yr=5.0, period_yr=2000.0, amp_rad=1.0, e=0.159):
    """9M-16 record in which member N (sigma_N = sigma0 + N beta) librates about pi.

    Truth: sigma0 = sigma_N - N beta, beta = varpi_M - varpi rotating at g_M - g = -6.458e-5 rad/yr
    (period 97 kyr), so c0 = pi - N beta drifts at +N 0.0646 rad/kyr (0.129 rad/kyr for N = 2, SISTERS §1.4).
    a follows from sigma0 by the identity: sigma0_dot = 9 n_M - 16 n + 7 g.
    """
    t = np.arange(0.0, span_yr + step_yr / 2, step_yr)
    sig_n, sig_n_dot = pendulum(t, period_yr, amp_rad)
    varpi = 1.0 + G_BODY * t
    varpi_m = 4.0 + G_MARS * t
    beta = varpi_m - varpi
    sigma0 = sig_n - n_member * beta
    sigma0_dot = sig_n_dot - n_member * (G_MARS - G_BODY)
    lam_m = 0.5 + N_MARS * t
    lam = (9 * lam_m + 7 * varpi - sigma0) / 16
    n = (9 * N_MARS + 7 * G_BODY - sigma0_dot) / 16
    a = (TWO_PI / n) ** (2.0 / 3.0)
    node = 2.0 - 30.0 * ARCSEC * t
    node_m = 0.86 - 18.0 * ARCSEC * t
    ones = np.ones_like(t)
    mars = PlanetSeries(
        name='Mars',
        t_yr=t,
        a=1.5237 * ones,
        e=0.093 * ones,
        inc=1.85 * DEG * ones,
        Omega=np.mod(node_m, TWO_PI),
        varpi=np.mod(varpi_m, TWO_PI),
        mean_longitude=np.mod(lam_m, TWO_PI),
    )
    series = MMRSeries(
        t_yr=t,
        a=a,
        e=e * ones,
        inc=5.0 * DEG * ones,
        Omega=np.mod(node, TWO_PI),
        omega=np.mod(varpi - node, TWO_PI),
        varpi=np.mod(varpi, TWO_PI),
        angle_wrapped=np.mod(sigma0, TWO_PI),
        resonance=ResonanceSpec.from_key('9M-16+0+7'),
        regime=Regime(step_yr=step_yr, span_yr=span_yr, n_rows=len(t), planets_saved=('Mars',)),
        planets={'Mars': mars},
        mean_longitude=np.mod(lam, TWO_PI),
        mean_anomaly=np.mod(lam - varpi, TWO_PI),
        name=f'synthetic-9M-16-N{n_member}',
    )
    truth = dict(t=t, sigma_n=sig_n, sigma0=sigma0, beta=beta, centre0=np.pi - n_member * beta)
    return series, truth


@pytest.fixture(scope='module')
def record_n2():
    return multiplet_record(2)


def _circ(x):
    return np.angle(np.exp(1j * np.asarray(x)))


def test_slow_offset_is_j_beta(record_n2):
    series, truth = record_n2
    spec = series.resonance
    for j in (1, 2, 5):
        member = ResonanceSpec.from_key(f'9M-16+{j}+{7 - j}')
        delta = sr.slow_offset(spec, member, series)
        assert delta.shape == truth['t'].shape
        # exactly j (varpi_M - varpi) up to a constant multiple of 2 pi, and continuous (inputs were wrapped)
        assert np.max(np.abs(_circ(delta - j * truth['beta']))) < 1e-9
        assert np.max(np.abs(np.diff(delta))) < 0.01


def test_member_series_recovers_the_librating_member(record_n2):
    # sigma0 + Delta_2 = sigma_2: the member's own wrapped angle is the pendulum
    series, truth = record_n2
    member = ResonanceSpec.from_key('9M-16+2+5')
    ms = sr.member_series(series, member)
    assert np.max(np.abs(_circ(ms.angle_wrapped - truth['sigma_n']))) < 1e-8
    assert ms.resonance.key == '9M-16+2+5'


def test_slow_offset_rejects_retrograde_and_foreign_members(record_n2):
    series, _ = record_n2
    with pytest.raises(ValueError):
        sr.slow_offset(series.resonance, ResonanceSpec.from_key('6M-11+1+4'), series)


def test_centre_track_freezes_the_librating_member_and_screen_keeps_only_it(record_n2, settings):
    series, truth = record_n2
    t = truth['t']
    c0 = truth['centre0']  # pi - 2 beta: drifts by 2 x 6.458e-5 x 1e5 = 12.9 rad over the record
    window = 6000.0  # three libration periods
    inner = (t > 3 * window) & (t < t[-1] - 3 * window)
    tracks = {}
    for j in range(1, 8):
        member = ResonanceSpec.from_key(f'9M-16+{j}+{7 - j}')
        delta = sr.slow_offset(series.resonance, member, series)
        tracks[j] = sr.centre_track(c0, delta, t, window, settings.filter_order)
    # member 2: c_2 = pi - 2 beta + 2 beta = pi (mod 2 pi) inside the window
    assert np.max(np.abs(_circ(tracks[2][inner] - np.pi))) < 0.02
    # member 3: c_3 = pi + beta: moves at -6.458e-5 rad/yr
    rate3 = np.polyfit(t[inner], tracks[3][inner], 1)[0]
    assert rate3 == pytest.approx(G_MARS - G_BODY, rel=0.01)

    kept, rng, rate, max_pass = sr.screen(tracks[2][inner], c0[inner], t[inner], settings)
    assert kept
    assert rng < 0.05
    assert abs(rate) < 1e-6
    for j in (1, 3, 4, 7):
        kept_j, rng_j, rate_j, pass_j = sr.screen(tracks[j][inner], c0[inner], t[inner], settings)
        assert not kept_j, j
        # c_j = pi + (j - 2) beta: rate (j - 2)(g_M - g)
        assert rate_j == pytest.approx((j - 2) * (G_MARS - G_BODY), rel=0.02), j
        assert pass_j > np.pi


def test_centre_track_removes_a_fast_wiggle_of_the_offset(settings):
    # Delta = 1e-4 t + 0.5 sin(2 pi t / 500): smoothing over W_c = 5000 yr keeps the line, drops the wiggle
    # (filtfilt gain at P = 500 with P_c = 5000 is ~1e-4 for order >= 2)
    t = np.arange(0.0, 50_000.0, 5.0)
    delta = 1e-4 * t + 0.5 * np.sin(TWO_PI * t / 500.0)
    c0 = np.zeros_like(t)
    track = sr.centre_track(c0, delta, t, 5000.0, settings.filter_order)
    inner = (t > 10_000) & (t < 40_000)
    np.testing.assert_allclose(track[inner], 1e-4 * t[inner], atol=0.01)


def test_screen_isolated_conditions(settings):
    t = np.arange(0.0, 20_000.0, 10.0)
    span = t[-1] - t[0]
    c0 = 1e-4 * t  # range 2.0 rad
    # range: cosine over 4 whole periods (regression slope 0); range 0.8 <= 0.5 x 2.0 -> kept
    ok = 0.4 * np.cos(TWO_PI * t / 5000.0)
    kept, rng, _, max_pass = sr.screen(ok, c0, t, settings)
    assert kept and rng == pytest.approx(0.8, abs=0.01) and max_pass == pytest.approx(0.8, abs=0.02)
    # amplitude 0.6: range 1.2 > 1.0 -> not kept (pass 1.2 < pi, slope 0)
    kept, rng, rate, _ = sr.screen(0.6 * np.cos(TWO_PI * t / 5000.0), c0, t, settings)
    assert not kept and rng == pytest.approx(1.2, abs=0.01) and abs(rate) < 1e-6
    # rate: a pure line has no residual, so the bound is the floor 0.02 rad/kyr = 2e-5 rad/yr
    kept, rng, rate, _ = sr.screen(1e-5 * t, c0, t, settings)  # 0.01 rad/kyr, range 0.2
    assert kept and rate == pytest.approx(1e-5, rel=1e-3)
    kept, rng, rate, _ = sr.screen(3e-5 * t, c0, t, settings)  # 0.03 rad/kyr, range 0.6
    assert not kept and rate == pytest.approx(3e-5, rel=1e-3)
    # pass: c0 range 20 rad; one cosine period of amplitude 2 (slope 0, range 4 <= 10) but a pass of 4 > pi
    c0_big = 1e-3 * t
    one = 2.0 * np.cos(TWO_PI * (t - t[0]) / span)
    kept, rng, _, max_pass = sr.screen(one, c0_big, t, settings)
    assert not kept and max_pass == pytest.approx(4.0, abs=0.05)
    # the same with amplitude 1.2: pass 2.4 < pi -> kept
    kept, _, _, max_pass = sr.screen(0.6 * one, c0_big, t, settings)
    assert kept and max_pass == pytest.approx(2.4, abs=0.05)


def test_beta_librates():
    t = np.arange(0.0, 400_000.0, 50.0)
    window = (t[0], t[-1])
    # libration of beta about 0 with amplitude 1 rad, period 100 kyr: 8 turns, never a full turn
    assert sr.beta_librates(t, 1.0 * np.sin(TWO_PI * t / 100_000.0), window)
    # rotation at g_M - g: a full turn every 97 kyr
    assert not sr.beta_librates(t, (G_MARS - G_BODY) * t, window)
    # a single turn back (half a period over the window): not enough
    assert not sr.beta_librates(t, 1.0 * np.sin(np.pi * t / t[-1]), window)
    # large swings (amplitude 4 rad: range 8 > 2 pi) cover a full turn
    assert not sr.beta_librates(t, 4.0 * np.sin(TWO_PI * t / 100_000.0), window)


def test_regular_stretch_drops_slow_edge_cycles():
    # TPs every 1000 yr from 3000 to 12000, slow cycles at the edges (0 and 15000)
    # full periods: 4000, 2000 x 8, 4000; median 2000; ratio 1.5 keeps the 2000-yr ones -> TPs 3000..12000
    tp = np.array([0.0] + list(np.arange(3000.0, 12001.0, 1000.0)) + [15000.0])
    assert sr.regular_stretch(tp, 1.5) == pytest.approx((3000.0, 12000.0))
    assert sr.regular_stretch(np.array([0.0, 1000.0]), 1.5) is None


# ----------------------------------------------------------------------------------------------
# subresonances.py: step 2 (decide)
# ----------------------------------------------------------------------------------------------


def cand(key, j, *, rng=1.68, coh=0.94, mean_deg=0.1, coef=1.0, nodal=False, spec=None, reason=''):
    return SubresonanceCandidate(
        spec=spec if spec is not None else ResonanceSpec.from_key(key),
        j=j,
        nodal=nodal,
        centre_range_rad=rng,
        centre_rate_rad_yr=1.7e-5,
        centre_max_pass_rad=rng,
        centre_mean_rad=mean_deg * DEG,
        phi0_coherence=coh,
        coefficient_rel=coef,
        composite_rms_rad=NAN,
        period_corr=NAN,
        period_ratio=NAN,
        rank=0,
        accepted=False,
        reason=reason,
    )


def nodal_6m():
    base = ResonanceSpec.from_key('6M-11+0+5')
    return dataclasses.replace(base, key='6M-11+0+5[N+1,NM-1]', body_node_coefficient=1, planet_node_coefficients=(-1,))


E65366 = 0.143
INC_LOW = 5.0 * DEG


def test_decide_prototype_65366_accepts_the_varpi_p_member(settings):
    # SISTERS §2: 6M-11+1+4 range 1.68, coherence 0.94 against 0.32 for sigma0, mean centre 0.1 deg
    best = cand('6M-11+1+4', 1)
    other = cand('6M-11+2+3', 2, rng=3.0, coh=0.5, coef=0.91)
    accepted, tie_by_order, undetermined = sr.decide([other, best], 0.32, E65366, INC_LOW, settings)
    assert accepted is not None and accepted.spec.key == '6M-11+1+4'
    assert not tie_by_order and not undetermined


def test_decide_empty(settings):
    assert sr.decide([], 0.3, E65366, INC_LOW, settings) == (None, False, False)


@pytest.mark.parametrize('coh, sigma0_coh', [(0.79, 0.3), (0.85, 0.7)])
def test_decide_test_a_coherence_and_gain(settings, coh, sigma0_coh):
    # A: >= 0.8 and >= sigma0 + 0.2
    accepted, _, _ = sr.decide([cand('6M-11+1+4', 1, coh=coh)], sigma0_coh, E65366, INC_LOW, settings)
    assert accepted is None


def test_decide_test_b_symmetric_centre_and_its_exemptions(settings):
    off = [cand('6M-11+1+4', 1, mean_deg=90.0)]
    assert sr.decide(off, 0.32, E65366, INC_LOW, settings)[0] is None
    # skipped: e > 0.3, i > 20 deg, or the caller says 1:n / co-orbital
    assert sr.decide(off, 0.32, 0.35, INC_LOW, settings)[0] is not None
    assert sr.decide(off, 0.32, E65366, 25 * DEG, settings)[0] is not None
    assert sr.decide(off, 0.32, E65366, INC_LOW, settings, symmetric_centre=False)[0] is not None
    # within 30 deg of pi or of 0 (from either side of the wrap)
    for mean in (160.0, 200.0, 340.0, -20.0, 25.0):
        assert sr.decide([cand('6M-11+1+4', 1, mean_deg=mean)], 0.32, E65366, INC_LOW, settings)[0] is not None, mean
    assert sr.decide([cand('6M-11+1+4', 1, mean_deg=45.0)], 0.32, E65366, INC_LOW, settings)[0] is None


def test_decide_test_c_uniqueness_and_ties(settings):
    # distinct ranges (1.0 vs 2.0, beyond 20 %): the smaller range is the unique one, whatever the coefficient
    a = cand('6M-11+1+4', 1, rng=1.0, coef=0.5)
    b = cand('6M-11+2+3', 2, rng=2.0, coef=1.0)
    acc, tie, _ = sr.decide([b, a], 0.32, E65366, INC_LOW, settings)
    assert acc.spec.key == '6M-11+1+4' and not tie
    # tie (1.68 vs 1.75, same order 5): the larger coefficient wins, not by order
    a = cand('6M-11+1+4', 1, rng=1.68, coef=0.84)
    b = cand('6M-11+2+3', 2, rng=1.75, coef=1.0)
    acc, tie, _ = sr.decide([a, b], 0.32, E65366, INC_LOW, settings)
    assert acc.spec.key == '6M-11+2+3' and not tie
    # tie that neither order nor coefficient resolves: the smallest range, flagged (cap medium)
    a = cand('6M-11+1+4', 1, rng=1.68, coef=NAN)
    b = cand('6M-11+2+3', 2, rng=1.75, coef=NAN)
    acc, tie, _ = sr.decide([b, a], 0.32, E65366, INC_LOW, settings)
    assert acc.spec.key == '6M-11+1+4' and tie


def test_decide_nodal_twin_tie_goes_to_lower_order(settings):
    # SISTERS §1.4: varpi_P member (order 5) range 1.68, coherence 0.94; node member (order 7) 2.60/0.87 there,
    # here made a near-tie (1.60 < 1.68 within 20 %) so that only the order decides
    planar = cand('6M-11+1+4', 1, rng=1.68, coh=0.94, coef=1.0)
    node = cand('', 0, rng=1.60, coh=0.92, coef=NAN, nodal=True, spec=nodal_6m())
    acc, tie_by_order, undetermined = sr.decide([node, planar], 0.32, E65366, INC_LOW, settings)
    assert acc is not None and acc.spec.key == '6M-11+1+4'
    assert tie_by_order and not undetermined
    # a nodal member that wins outright is not accepted
    node_only = cand('', 0, rng=0.5, coh=0.95, nodal=True, spec=nodal_6m())
    acc, _, _ = sr.decide([node_only, cand('6M-11+1+4', 1, rng=1.68)], 0.32, E65366, INC_LOW, settings)
    assert acc is None


def test_decide_step1_failures_are_not_considered(settings):
    for reason in ('range', 'rate', 'pass'):
        acc, _, _ = sr.decide([cand('6M-11+1+4', 1, reason=reason)], 0.32, E65366, INC_LOW, settings)
        assert acc is None, reason


def test_decide_beta_librating_is_undetermined(settings):
    assert sr.decide([cand('6M-11+1+4', 1)], 0.32, E65366, INC_LOW, settings, beta_librates=True) == (None, False, True)


def test_decide_test_e_and_d(settings):
    c = [cand('6M-11+1+4', 1)]
    # E (49030: strongest predicted T0 45 kyr against 0.94 kyr observed: ratio 48 > 2) -> rejected
    assert sr.decide(c, 0.32, E65366, INC_LOW, settings, theory=sr.TheoryCheck(t0_ratio=48.0))[0] is None
    # E passes at 1.1 (65366: 2.6 vs 2.4 kyr)
    ok = sr.TheoryCheck(t0_ratio=1.1, composite_rms_rad=10 * DEG, single_rms_rad=27 * DEG, period_corr=0.99, period_ratio=1.11)
    assert sr.decide(c, 0.32, E65366, INC_LOW, settings, theory=ok)[0] is not None
    # D failing (49030's 88 deg / 76 deg, correlation -0.03) never rejects while E passes
    bad_d = sr.TheoryCheck(t0_ratio=1.1, composite_rms_rad=88 * DEG, single_rms_rad=76 * DEG, period_corr=-0.03, period_ratio=1.0)
    assert sr.decide(c, 0.32, E65366, INC_LOW, settings, theory=bad_d)[0] is not None


# ----------------------------------------------------------------------------------------------
# subresonances.py: triggers
# ----------------------------------------------------------------------------------------------


def _filler(cls, **over):
    """An instance of a dataclass with every required field 'not measured' (nan, '', [], None, ...)."""
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in over or f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue
        tp = hints[f.name]
        origin = typing.get_origin(tp)
        if tp is float:
            kwargs[f.name] = NAN
        elif tp is int:
            kwargs[f.name] = 0
        elif tp is bool:
            kwargs[f.name] = False
        elif tp is str:
            kwargs[f.name] = ''
        elif tp is CheckValue:
            kwargs[f.name] = CheckValue.NOT_MEASURED
        elif origin is list:
            kwargs[f.name] = []
        elif origin is tuple:
            kwargs[f.name] = ()
        else:
            kwargs[f.name] = None
    kwargs.update(over)
    return cls(**kwargs)


def _tps(times=()):
    times = np.asarray(times, dtype=float)
    kinds = np.where(np.arange(len(times)) % 2 == 0, 1, -1)
    return TurningPoints(
        indices=np.arange(len(times), dtype=int),
        times_yr=times,
        values_rad=np.where(kinds > 0, 1.0, -1.0) if len(times) else np.zeros(0),
        kinds=kinds,
        prominence_rad=0.1,
    )


def neutral_geometry(runs=(), span=100_000.0):
    t = np.linspace(0.0, span, 101)
    centre = CentreTrack(
        t_yr=t,
        centre_rad=np.zeros_like(t),
        window_yr=5000.0,
        turns=_tps(),
        passes=[],
        max_pass_rad=0.0,
        max_pass_t0_yr=NAN,
        max_pass_t1_yr=NAN,
        net_rev=0.0,
        range_rad=0.0,
    )
    return _filler(
        Geometry,
        window=(0.0, span),
        regime_start_yr=0.0,
        turning_points=_tps(),
        centre=centre,
        runs=list(runs),
        n_tp_cut_half=0,
        n_tp_cut_double=0,
    )


def rejected_run(t0, t1, n_tp):
    tp = np.linspace(t0, t1, n_tp)
    return _filler(
        Episode,
        episode_id=0,
        tp_times_yr=tp,
        tp_values_rad=np.where(np.arange(n_tp) % 2 == 0, 1.0, -1.0),
        accepted=False,
        shape=EpisodeShape.NONE,
        reject_reason=RejectReason.CENTRE_MOVES,
        bands_overlap=False,
        centre_returns=0,
    )


def response_with(r2=NAN, coherence=NAN):
    windows = [] if math.isnan(r2) else [ForceWindow(i * 7000.0, i * 7000.0 + 7000.0, r2, 1e-6, 0.0, 0.0) for i in range(10)]
    return _filler(Response, windows=windows, phi0_coherence=coherence, force_r2_h1=r2, force_r2_h2=r2)


def test_triggers_nothing_on_a_neutral_record(settings):
    trig, support = sr.triggers(neutral_geometry(), response_with(), '', 0, True, settings)
    assert trig == () and support == ()


@pytest.mark.parametrize('reason', ['kinematic_drift', 'kinematic_libration', 'drift_mechanism_unknown'])
def test_trigger_t5_before_mechanism_verdicts(settings, reason):
    trig, _ = sr.triggers(neutral_geometry(), response_with(), reason, 0, True, settings)
    assert SubresonanceTrigger.T5 in trig


def test_trigger_t6_binomial_mode(settings):
    assert SubresonanceTrigger.T6 in sr.triggers(neutral_geometry(), response_with(), '', 2, False, settings)[0]
    assert SubresonanceTrigger.T6 in sr.triggers(neutral_geometry(), response_with(), '', -1, False, settings)[0]
    # a clean stationary libration with full margins, or j* = 0: no T6
    assert SubresonanceTrigger.T6 not in sr.triggers(neutral_geometry(), response_with(), '', 2, True, settings)[0]
    assert SubresonanceTrigger.T6 not in sr.triggers(neutral_geometry(), response_with(), '', 0, False, settings)[0]


def test_trigger_t2_response_with_rotating_phase(settings):
    # a responds (R^2 0.8 >= 0.4) but phi0 coherence 0.32 < 0.7 (65366's sigma0)
    assert SubresonanceTrigger.T2 in sr.triggers(neutral_geometry(), response_with(0.8, 0.32), '', 0, True, settings)[0]
    # coherent phase: no T2; no response: no T2
    assert SubresonanceTrigger.T2 not in sr.triggers(neutral_geometry(), response_with(0.8, 0.94), '', 0, True, settings)[0]
    assert SubresonanceTrigger.T2 not in sr.triggers(neutral_geometry(), response_with(0.2, 0.32), '', 0, True, settings)[0]


def test_trigger_t1_centre_moves_run(settings):
    # a centre_moves run of 60 TPs over 5-95 kyr of a 100 kyr window (fraction 0.9 >= 0.5)
    geo = neutral_geometry(runs=[rejected_run(5_000.0, 95_000.0, 60)])
    assert SubresonanceTrigger.T1 in sr.triggers(geo, response_with(), '', 0, True, settings)[0]


# ----------------------------------------------------------------------------------------------
# the whole check through analyse_mmr on the synthetic multiplet (SPEC §6.4 step 3, DELTA items 2-3)
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def analysed_n2(record_n2, settings):
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series, _ = record_n2
    return analyse_mmr(series, settings, scan_owner=False)


def test_analyse_finds_the_librating_member(analysed_n2):
    res = analysed_n2
    sub = res.subresonances
    assert sub is not None
    assert sub.triggers  # the centre of sigma0 drifts at 0.129 rad/kyr: at least T1
    assert sub.accepted is not None
    assert sub.accepted.spec.key == '9M-16+2+5'
    assert sub.accepted.j == 2 and not sub.accepted.nodal
    assert sub.check == CheckValue.FAIL
    assert not sub.undetermined
    # its centre: fixed near pi (mean 180 deg within the B tolerance), coherence well above sigma0's
    assert abs(math.remainder(sub.accepted.centre_mean_rad - np.pi, TWO_PI)) < 30 * DEG
    assert sub.accepted.phi0_coherence >= 0.8
    assert sub.accepted.phi0_coherence >= sub.sigma0_phi0_coherence + 0.2


def test_analyse_verdict_names_the_subresonance(analysed_n2):
    v = analysed_n2.verdict
    # rule C (W7 'ug22', author 2026-10-16): the members of one commensurability are one resonance, so the catalogue
    # row takes the status of the member that librates over the whole record
    assert v.status == Status.RESONANT == analysed_n2.derived[0].verdict.status
    assert str(v.reason) == 'subresonance_librates'
    assert str(v.libration_origin) == 'subresonance'
    assert v.owner == '9M-16+2+5'


def test_analyse_derived_row_of_the_member(analysed_n2):
    # DELTA item 3: the accepted member is a row of its own with the catalogue angle as its parent;
    # it librates about pi over the whole record, and the multiplet status is the best of the members
    res = analysed_n2
    assert len(res.derived) == 1
    d = res.derived[0]
    assert d.resonance.key == '9M-16+2+5'
    assert d.parent == '9M-16+0+7'
    assert d.verdict.status == Status.RESONANT
    assert d.derived == []  # depth 1
    assert res.multiplet_status == Status.RESONANT
    assert d.multiplet_status == Status.RESONANT


def test_analyse_control_catalogue_angle_librates_itself(settings):
    # N = 0: sigma0 is the pendulum; no member may take the oscillation away from it
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series, _ = multiplet_record(0)
    res = analyse_mmr(series, settings, scan_owner=False)
    assert res.verdict.status == Status.RESONANT
    assert res.subresonances is None or res.subresonances.accepted is None
    assert res.derived == []
