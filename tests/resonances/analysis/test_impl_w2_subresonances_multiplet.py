"""Multiplet coefficients by direct averaging (analysis/mmr/multiplet.py), checked against values from other sources:
Murray & Dermott (1999) for the 3:1 and 2:1 with Jupiter, the e^|q-j| e_P^|j| homogeneity of the expansion, a
brute-force average written here (the prototype's algorithm, not the 2-D Fourier sum of the module), and the numbers
of SISTERS.md for 65366 and 31935."""

import math

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import multiplet

MU_JUPITER = 9.5479e-4
MU_MARS = 3.2272e-7
A_JUPITER = 5.2026


def _position(a, e, varpi, mean_longitude):
    """Planar Keplerian position by fixed-point iteration (independent of the module's Newton solver)."""
    mean_anomaly = np.mod(mean_longitude - varpi, 2 * np.pi)
    anomaly = mean_anomaly.copy()
    for _ in range(40):  # converges as e^n
        anomaly = mean_anomaly + e * np.sin(anomaly)
    x, y = a * (np.cos(anomaly) - e), a * np.sqrt(1 - e * e) * np.sin(anomaly)
    return x * np.cos(varpi) - y * np.sin(varpi), x * np.sin(varpi) + y * np.cos(varpi)


def _brute_force(k_p, k, a, a_p, e, e_p, mu, js, n_sigma=8, n_beta=32, per_lap=1024):
    """C_j by averaging R along the curve k_p lambda_P + k lambda = sigma0 for each (sigma0, beta), then projecting."""
    sigmas = np.arange(n_sigma) * 2 * np.pi / n_sigma
    betas = np.arange(n_beta) * 2 * np.pi / n_beta
    s = np.arange(per_lap * k_p) * 2 * np.pi * k_p / (per_lap * k_p)
    averaged = np.zeros((n_sigma, n_beta))
    for i, sigma in enumerate(sigmas):
        x, y = _position(a, e, 0.0, s)
        for j, beta in enumerate(betas):
            xp, yp = _position(a_p, e_p, beta, (sigma - k * s) / k_p)
            r = np.hypot(x - xp, y - yp)
            averaged[i, j] = np.mean(mu * (1 / r - (x * xp + y * yp) / np.hypot(xp, yp) ** 3))
    first = (averaged * np.exp(-1j * sigmas)[:, None]).mean(axis=0)
    return np.array([np.mean(first * np.exp(-1j * jv * betas)) for jv in js])


def test_three_to_one_with_jupiter_gives_the_laplace_coefficient_of_murray_dermott():
    # M&D 1999, Table 8.5 (term 4D2.1 for the 3:1): f = 0.5988; R = (mu / a') f e^2 cos(sigma) = 2 C_0 cos(sigma)
    alpha = (1 / 3) ** (2 / 3)
    for e in (0.005, 0.01):
        js, c = multiplet.resonant_coefficients(3, -1, alpha * A_JUPITER, A_JUPITER, e, 0.0, MU_JUPITER, 0, 0)
        f = 2 * c[0].real * A_JUPITER / (MU_JUPITER * e * e)
        assert f == pytest.approx(0.5988, rel=2e-3)
        assert abs(c[0].imag) < 1e-9 * abs(c[0].real)  # real: a cosine-only expansion


def test_two_to_one_relative_eccentricity_weight_is_f31_over_f27():
    # M&D 1999, Table 8.5: f27 = -1.190, f31 = 0.428 for the 2:1; C_-1 / C_0 = -q c e_P / e gives c = |f31 / f27| = 0.360
    q = multiplet.signed_order(2, -1)
    js, c = multiplet.resonant_coefficients(2, -1, 3.278, A_JUPITER, 0.005, 0.005, MU_JUPITER, -1, 0)
    assert q == -1 and list(js) == [-1, 0]
    assert multiplet.binomial_ratio(c, js, q, 0.005, 0.005) == pytest.approx(0.428 / 1.190, abs=3e-3)


def test_small_eccentricity_coefficients_are_homogeneous_of_degree_q():
    # every member of a q-th order multiplet is e^(q-j) e_P^j: doubling e and e_P multiplies each C_j by 2^q
    js, c1 = multiplet.resonant_coefficients(9, -16, 2.2359, 1.5237, 0.004, 0.003, MU_MARS, 0, 7)
    _, c2 = multiplet.resonant_coefficients(9, -16, 2.2359, 1.5237, 0.008, 0.006, MU_MARS, 0, 7)
    np.testing.assert_allclose((c2 / c1).real, 2.0**7, rtol=0.02)


@pytest.mark.parametrize(
    'k_p, k, a, a_p, e, e_p, mu',
    [(3, -1, 2.5, A_JUPITER, 0.15, 0.048, MU_JUPITER), (6, -11, 2.2822, 1.5237, 0.143, 0.099, MU_MARS)],
)
def test_fourier_average_agrees_with_a_brute_force_average(k_p, k, a, a_p, e, e_p, mu):
    q = multiplet.signed_order(k_p, k)
    js = np.arange(min(0, q) - 1, max(0, q) + 2)
    _, c = multiplet.resonant_coefficients(k_p, k, a, a_p, e, e_p, mu, int(js[0]), int(js[-1]))
    reference = _brute_force(k_p, k, a, a_p, e, e_p, mu, js)
    np.testing.assert_allclose(c.real, reference.real, atol=1e-6 * np.max(np.abs(reference)))


def test_the_strongest_member_of_the_prototypes_is_that_of_sisters():
    # SISTERS §2: j = 1 for 65366 6M-11 (e 0.143, e_M 0.099), j = 2 for 31935 9M-16 (e 0.159, e_M 0.093)
    for k_p, k, a, e, e_p, expected in ((6, -11, 2.2822, 0.143, 0.099, 1), (9, -16, 2.2359, 0.159, 0.093, 2)):
        planar = multiplet.planar_members(k_p, k)
        js, c = multiplet.resonant_coefficients(k_p, k, a, 1.5237, e, e_p, MU_MARS, -3, int(planar[-1]) + 3)
        in_planar = np.isin(js, planar)
        assert int(js[in_planar][np.argmax(np.abs(c[in_planar]))]) == expected
        # Rouche fails (composite regime, SISTERS §1.3) and P(beta) over j = -3..q+3 does not wind (N = 0)
        assert not multiplet.rouche_holds(c, js, expected)
        assert multiplet.winding_number(c, js) == 0


def test_composite_of_a_single_term_and_its_winding():
    beta = np.linspace(0, 2 * np.pi, 9)
    coefficients, js = np.array([0.0, 0.0, 2.0]), np.array([0, 1, 2])
    np.testing.assert_allclose(multiplet.composite(coefficients, js, beta), 2.0 * np.exp(2j * beta), atol=1e-12)
    # pi - arg(C e^{2 i beta}) = pi - 2 beta, unwrapped
    np.testing.assert_allclose(multiplet.composite_centre(coefficients, js, beta), np.pi - 2 * beta, atol=1e-9)
    assert multiplet.winding_number(coefficients, js) == 2
    assert multiplet.rouche_holds(np.array([0.2, 0.3, 1.0]), js, 2)
    assert not multiplet.rouche_holds(np.array([0.5, 0.6, 1.0]), js, 2)
    # a negative coefficient puts the centre at 0 instead of pi
    np.testing.assert_allclose(np.mod(multiplet.composite_centre(np.array([-1.0]), np.array([0]), beta), 2 * np.pi), 0.0, atol=1e-9)


@pytest.mark.parametrize(
    'q, e, e_p, ratio, expected',
    [
        (7, 0.159, 0.093, 0.72, 2),  # 31935: x = 0.42, (q+1) x / (1+x) = 2.4 (SISTERS §1.3)
        (5, 0.143, 0.099, 0.71, 1),  # 65366: x = 0.49, 2.0 -> 1
        (5, 0.1, 0.9 / 5 * 0.1, 1.0, 0),  # x = 0.9 / q: the catalogue angle leads (x <= 1/q)
        (5, 0.1, 1.1 / 5 * 0.1, 1.0, 1),
        (-3, 0.1, 0.1, 1.0, -2),  # q < 0: j* <= 0
    ],
)
def test_binomial_mode(q, e, e_p, ratio, expected):
    assert multiplet.binomial_mode(q, e, e_p, ratio) == expected


def test_small_period_and_spacing_of_the_prototypes():
    # SISTERS §1.2: 65366 j=1 mu_eff 4.8e-9 -> T0 2.6 kyr; §1.1: delta a 6.3e-6 au (65366) and 3.2e-6 au (31935)
    a, k = 2.2822, -11
    magnitude = 4.8e-9 / (2 * a)
    n = 2 * math.pi / a**1.5
    expected = 2 * math.pi / (abs(k) * n * math.sqrt(3 * 4.8e-9))
    assert multiplet.small_period_yr(magnitude, a, k) == pytest.approx(expected, rel=1e-12)
    assert multiplet.small_period_yr(magnitude, a, k) == pytest.approx(2600, rel=0.02)
    arcsec = math.pi / 180 / 3600
    assert multiplet.member_spacing_au(2.2822, 35.54 * arcsec, 18.43 * arcsec, -11) == pytest.approx(6.3e-6, rel=0.03)
    assert multiplet.member_spacing_au(2.2359, 31.75 * arcsec, 18.43 * arcsec, -16) == pytest.approx(3.2e-6, rel=0.03)


def test_cached_coefficients_are_those_of_the_rounded_point():
    js = np.array([0, 1, 2])
    first, key = multiplet.cached_coefficients('3J-1', 3, -1, 2.5, A_JUPITER, 0.15, 0.048, MU_JUPITER, js)
    again, key2 = multiplet.cached_coefficients('3J-1', 3, -1, 2.5, A_JUPITER, 0.15, 0.048, MU_JUPITER, js[::-1])
    assert key == key2 == multiplet.cache_key('3J-1', 2.5 / A_JUPITER, 0.15, 0.048)
    np.testing.assert_allclose(again, first[::-1])
    _, direct = multiplet.resonant_coefficients(3, -1, 2.5, A_JUPITER, 0.15, 0.048, MU_JUPITER, 0, 2)
    np.testing.assert_allclose(first, direct, rtol=2e-3)


def test_bad_orbits_are_refused():
    with pytest.raises(ValueError):
        multiplet.resonant_coefficients(3, -1, 2.5, A_JUPITER, 1.2, 0.05, MU_JUPITER, 0, 2)
    with pytest.raises(ValueError):
        multiplet.resonant_coefficients(3, -1, 2.5, A_JUPITER, 0.1, 0.05, MU_JUPITER, 2, 0)
