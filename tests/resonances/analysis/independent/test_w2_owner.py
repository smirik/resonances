"""Independent functional tests of the owner search (SPEC §4.8, `analysis/mmr/owner.py`).

Written from the API, the docstrings and the SPEC only. The truth comes from closed forms and from synthetic
records built here:

* the owner phi_o = 2 lambda_J - lambda - varpi (`2J-1+0-1`) is an integrated pendulum
  phi_o'' = -omega0^2 sin phi_o; the body's mean motion follows by the identity phi_o' = 2 n_J - n - varpi',
  lambda = 2 lambda_J - varpi - phi_o, a = (2 pi / n)^(2/3) (au, yr, GM_sun = 4 pi^2). Hence, exactly,
  a_dot = (2 a / 3 n) phi_o'' = -(2 a / 3 n) omega0^2 sin phi_o: a is driven by the owner with the fixed phase.
* the analysed angle is another resonance with a different lambda part, so that sigma = (k / k_o) phi_o + psi:
  - sigma = 5J-2+0-3:  sigma = 2 phi_o + (lambda_J - varpi), psi turns uniformly at n_J - varpi';
  - sigma = 4J-5S-1+0+0+2: sigma = phi_o + (2 lambda_J - 5 lambda_S + 3 varpi), psi turns once per 25 kyr
    (the synthetic Saturn is placed so that 2 n_J - 5 n_S + 3 varpi' = 2 pi / 25 kyr);
  - sigma = 2J-1+0-1 is the owner itself: its libration is its own, no owner may be found.
* the owner librates (40 deg), circulates (a forced wave, energy above the separatrix) or crosses the separatrix
  in the middle of the record (omega0^2 decreasing linearly).
"""

import dataclasses
import functools
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis.mmr import owner
from resonances.resonance.analysis.mmr.context import build_context
from resonances.resonance.analysis.mmr.models import (
    MMRSeries,
    OwnerCandidate,
    PlanetSeries,
    Regime,
    ResonanceKind,
    ResonanceSpec,
)
from resonances.resonance.analysis.mmr.preprocess import preprocess
from resonances.resonance.analysis.mmr.sampling import analyse_sampling
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue

S = settings_from_config()
TWO_PI = 2 * np.pi
NAN = float('nan')

# Jupiter's period 11.862615 yr; the body's perihelion turns once per 40 kyr.
N_J = TWO_PI / 11.862615  # 0.529662 rad/yr
VARPI_RATE = TWO_PI / 40000.0  # 1.5708e-4 rad/yr
PSI_PERIOD_3B = 25000.0  # psi of the three-body sigma: 2 n_J - 5 n_S + 3 varpi' = 2 pi / 25 kyr
N_S = (2 * N_J + 3 * VARPI_RATE - TWO_PI / PSI_PERIOD_3B) / 5
# Masses m/M_sun from the IAU ratios M_sun/M_J = 1047.3486, M_sun/M_S = 3497.898 (not from the package).
MU_J = 1 / 1047.3486
MU_S = 1 / 3497.898

T_REC = np.arange(0.0, 12000.0 + 0.5, 1.0)  # 12 kyr at 1 yr: 30 cycles of a 400-yr libration
OMEGA_LIB = TWO_PI / 400.0
P_LIB_40 = 412.5  # 400 yr x (1 + A^2/16 + 11 A^4/3072) for A = 40 deg = 0.698 rad: 1 + 0.0305 + 0.0009


def pendulum(omega0, phi0, rate0, t):
    sol = solve_ivp(
        lambda tt, y: [y[1], -(omega0**2) * np.sin(y[0])],
        (t[0], t[-1]),
        [phi0, rate0],
        t_eval=t,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
    )
    return sol.y[0], sol.y[1]


def crossing_pendulum(t):
    """omega0^2 falls linearly to 15 % of (2 pi / 400)^2: the 120-deg libration grows past the separatrix."""

    def rhs(tt, y):
        return [y[1], -(OMEGA_LIB**2) * (1 - 0.85 * tt / t[-1]) * np.sin(y[0])]

    sol = solve_ivp(rhs, (t[0], t[-1]), [math.radians(120), 0.0], t_eval=t, method='DOP853', rtol=1e-11, atol=1e-13)
    return sol.y[0], sol.y[1]


def planet(name, a, n, lam0, t):
    return PlanetSeries(
        name=name,
        t_yr=t,
        a=np.full_like(t, a),
        e=np.full_like(t, 0.05),
        inc=np.full_like(t, 0.01),
        Omega=np.full_like(t, 1.0),
        varpi=np.full_like(t, 0.3),
        mean_longitude=np.mod(n * t + lam0, TWO_PI),
    )


def build_series(key, phi, phidot, t, with_saturn):
    """A body whose owner 2J-1+0-1 follows phi (see the module docstring); the analysed angle is `key`."""
    lam_j = N_J * t + 0.4
    lam_s = N_S * t + 2.1
    varpi = 1.3 + VARPI_RATE * t
    lam = 2 * lam_j - varpi - phi
    n = 2 * N_J - VARPI_RATE - phidot
    a = (TWO_PI / n) ** (2.0 / 3.0)
    spec = ResonanceSpec.from_key(key)
    longitudes = {'Jupiter': lam_j, 'Saturn': lam_s}
    sigma = spec.body_coefficient * lam + spec.body_varpi_coefficient * varpi
    for name, k in zip(spec.planets, spec.planet_coefficients):
        sigma = sigma + k * longitudes[name]
    planets = {'Jupiter': planet('Jupiter', 5.2026, N_J, 0.4, t)}
    if with_saturn:
        planets['Saturn'] = planet('Saturn', 9.5549, N_S, 2.1, t)
    Omega = np.full_like(t, 0.7)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full_like(t, 0.1),
        inc=np.full_like(t, 0.05),
        Omega=Omega,
        omega=np.mod(varpi - Omega, TWO_PI),
        varpi=np.mod(varpi, TWO_PI),
        angle_wrapped=np.mod(sigma, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=t[1] - t[0], span_yr=t[-1] - t[0], n_rows=len(t), planets_saved=tuple(planets)),
        planets=planets,
        mean_longitude=np.mod(lam, TWO_PI),
        name='synthetic',
    )


def chain(series, settings=S):
    context = build_context(series, settings)
    sampling = analyse_sampling(series, context, settings)
    return context, preprocess(series, context, sampling, settings)


@functools.lru_cache(maxsize=None)
def owner_motion(kind):
    if kind == 'libration':
        return pendulum(OMEGA_LIB, math.radians(40), 0.0, T_REC)
    if kind == 'forced_wave':
        # Released at the bottom with rate 2 pi / 350: rate^2 - 4 omega0^2 = (2 pi)^2 (8.16e-6 - 6.25e-6) > 0,
        # so the pendulum circulates; omega0 = 2 pi / 800 yr.
        return pendulum(TWO_PI / 800, 0.0, TWO_PI / 350, T_REC)
    return crossing_pendulum(T_REC)


@functools.lru_cache(maxsize=None)
def scenario(kind, key, period_cycle_yr):
    phi, phidot = owner_motion(kind)
    series = build_series(key, phi, phidot, T_REC, with_saturn='S' in key or key == '2J-1+0-1')
    context, prep = chain(series)
    return series, context, prep, owner.run_owner(series, prep, context, period_cycle_yr, S)


def candidate(**overrides):
    """An OwnerCandidate that passes every test of `accept` by a wide margin."""
    values = dict(
        spec=ResonanceSpec.from_key('2J-1+0-1'),
        rate_rad_yr=0.0,
        r2_median=0.95,
        r2_windows=(0.95, 0.95, 0.95),
        phi0_coherence=0.98,
        psi_share=0.02,
        force_au_yr=1e-4,
        force_theory_au_yr=1e-2,
        strength_ratio=0.01,
        rank=1,
        accepted=False,
        reason='',
    )
    values.update(overrides)
    return OwnerCandidate(**values)


# --------------------------------------------------------------------------------------------- notation, closed forms


def test_owner_key_examples():
    # Docstring examples, and two keys checked by hand: 2 - 1 - 1 = 0 and 4 - 5 - 1 + 2 = 0 (d'Alembert).
    assert owner.owner_key(('Mars',), (11,), -20, 9) == '11M-20+0+9'
    assert owner.owner_key(('Venus', 'Jupiter'), (1, 5), -7, 1) == '1V+5J-7+0+0+1'
    assert owner.owner_key(('Jupiter',), (2,), -1, -1) == '2J-1+0-1'
    assert owner.owner_key(('Jupiter', 'Saturn'), (4, -5), -1, 2) == '4J-5S-1+0+0+2'


@pytest.mark.parametrize(
    'planets, coefficients, k, p, kind',
    [
        (('Jupiter',), (2,), -1, -1, ResonanceKind.TWO_BODY),  # p = -(2 - 1)
        (('Mars',), (11,), -20, 9, ResonanceKind.TWO_BODY),  # p = -(11 - 20)
        (('Jupiter', 'Saturn'), (4, -5), -1, 2, ResonanceKind.THREE_BODY),  # p = -(4 - 5 - 1)
    ],
)
def test_owner_spec_dalembert(planets, coefficients, k, p, kind):
    spec = owner.owner_spec(planets, coefficients, k)
    assert spec.kind == kind
    assert spec.planets == planets
    assert tuple(spec.planet_coefficients) == coefficients
    assert spec.body_coefficient == k
    assert spec.body_varpi_coefficient == p
    assert all(c == 0 for c in spec.planet_varpi_coefficients)
    assert sum(coefficients) + k + p == 0
    assert spec.order == abs(sum(coefficients) + k)


@pytest.mark.parametrize(
    'first, second, expected',
    [
        ('2J-1+0-1', '4J-2+0-2', True),  # harmonic: lambda parts 2:-1 and 4:-2
        ('2J-1+0-1', '2J-1+1-2', True),  # subresonance: same lambda part, another eccentricity term
        ('2J-1+0-1', '5J-2+0-3', False),  # another ratio
        ('2J-1+0-1', '2S-1+0-1', False),  # another planet
        ('2J-1+0-1', '4J-5S-1+0+0+2', False),  # three-body
    ],
)
def test_same_lambda_part(first, second, expected):
    a, b = ResonanceSpec.from_key(first), ResonanceSpec.from_key(second)
    assert owner.same_lambda_part(a, b) is expected
    assert owner.same_lambda_part(b, a) is expected


def test_candidate_rate_hand_calculation():
    # 2J-1+0-1: 2 x 0.5 - 0.9 - 0.01 = 0.09 rad/yr.
    rate = owner.candidate_rate(ResonanceSpec.from_key('2J-1+0-1'), {'Jupiter': 0.5}, 0.9, 0.01)
    assert rate == pytest.approx(0.09, abs=1e-12)
    # 4J-5S-1+0+0+2: 4 x 0.5 - 5 x 0.2 - 0.9 + 2 x 0.01 = 0.12 rad/yr.
    rate = owner.candidate_rate(ResonanceSpec.from_key('4J-5S-1+0+0+2'), {'Jupiter': 0.5, 'Saturn': 0.2}, 0.9, 0.01)
    assert rate == pytest.approx(0.12, abs=1e-12)


def test_rate_bound_uses_the_shorter_period():
    # 1.5 x 2 pi / min(1000, 500) = 0.0188496 rad/yr, whichever argument holds the shorter period.
    expected = 1.5 * TWO_PI / 500.0
    assert owner.rate_bound(1000.0, 500.0, S) == pytest.approx(expected, rel=1e-12)
    assert owner.rate_bound(500.0, 1000.0, S) == pytest.approx(expected, rel=1e-12)
    assert owner.rate_bound(500.0, 500.0, settings_from_config(mmr_owner_rate_factor=3.0)) == pytest.approx(2 * expected)


def test_theory_force_closed_form():
    # 2J-1 at a = 3.2776 au, n = 1.0593 rad/yr:
    # omega0_th^2 = k^2 n^2 3 mu_J = 1 x 1.1222 x 3 x 9.5479e-4 = 3.2143e-3; |c| = 1.5 n / a = 0.48479;
    # K_th = 3.2143e-3 / 0.48479 = 6.6304e-3 au/yr = 2 |k| n a mu_J.
    n, a = 1.0593, 3.2776
    two_body = owner.theory_force(ResonanceSpec.from_key('2J-1+0-1'), n, a)
    omega0_sq = n**2 * 3 * MU_J
    assert two_body == pytest.approx(omega0_sq / (1.5 * n / a), rel=1e-3)
    assert two_body == pytest.approx(6.6304e-3, rel=1e-3)
    # |k| = 2 doubles it; three-body: mu = mu_J mu_S.
    assert owner.theory_force(ResonanceSpec.from_key('5J-2+0-3'), n, a) == pytest.approx(2 * two_body, rel=1e-9)
    three_body = owner.theory_force(ResonanceSpec.from_key('4J-5S-1+0+0+2'), n, a)
    assert three_body == pytest.approx(2 * n * a * MU_J * MU_S, rel=1e-3)


def test_slope():
    t = np.linspace(0.0, 100.0, 51)
    assert owner.slope(t, 3.5 - 0.25 * t) == pytest.approx(-0.25, abs=1e-12)
    # An even function of t - mean(t) is orthogonal to t - mean(t) term by term: the slope stays 0.1.
    t = np.arange(0.0, 100.0, 0.5)
    even = np.cos(TWO_PI * (t - t.mean()) / 20)
    assert owner.slope(t, 0.1 * t + even) == pytest.approx(0.1, abs=1e-12)
    assert math.isnan(owner.slope(np.array([1.0]), np.array([2.0])))


def test_unwrap_ok_limits():
    # Limits: p99 <= 45 deg, max <= 135 deg.
    assert owner.unwrap_ok(np.zeros(1000), S)
    assert owner.unwrap_ok(np.full(1000, math.radians(40)), S)
    assert not owner.unwrap_ok(np.full(1000, math.radians(60)), S)  # p99 = 60 deg
    one_jump = np.zeros(1000)
    one_jump[500] = math.radians(170)  # p99 = 0, max = 170 deg
    assert not owner.unwrap_ok(one_jump, S)


def test_windows_needed_span():
    # Windows of W = 3.5 P stepped by 0.5 W: two windows need W + 0.5 W = 5.25 P.
    assert owner.windows_needed_span(100.0, S) == pytest.approx(525.0, rel=1e-9)
    assert owner.windows_needed_span(400.0, S) == pytest.approx(2100.0, rel=1e-9)


def test_band_pass_gain_closed_form():
    # filtfilt Butterworth N = 2: G(P) = 1 / (1 + (P_c / P)^4). Band [P/2, 2P] at period P:
    # G(P/2 cutoff) - G(2P cutoff) = 1/(1 + 1/16) - 1/(1 + 16) = 0.94118 - 0.05882 = 0.88235.
    # At P/20: 1/(1 + 10^4) - 1/(1 + 40^4) ~ 1e-4. At 20 P: 1/(1 + 40^-4) - 1/(1 + 10^-4) ~ 1e-4.
    t = np.arange(0.0, 40000.0, 2.0)
    period = 400.0
    mid = (t > 8 * period) & (t < t[-1] - 8 * period)
    out = owner.band_pass(t, np.sin(TWO_PI * t / period), period, 2.0, 2)
    amplitude = np.sqrt(2 * np.mean(out[mid] ** 2))
    assert amplitude == pytest.approx(0.88235, abs=0.01)
    fast = owner.band_pass(t, np.sin(TWO_PI * t / (period / 20)), period, 2.0, 2)
    slow = owner.band_pass(t, np.sin(TWO_PI * t / (period * 20)), period, 2.0, 2)
    line = owner.band_pass(t, 1.0 + 1e-3 * t, period, 2.0, 2)
    assert np.max(np.abs(fast[mid])) < 3e-3
    assert np.max(np.abs(slow[mid])) < 3e-3
    assert np.max(np.abs(line[mid])) < 1e-6


def test_dominant_period():
    t = np.arange(0.0, 30000.0, 1.0)
    values = np.sin(TWO_PI * t / 700.0) + 3 * np.sin(TWO_PI * t / 3000.0) + 1e-3 * t
    assert owner.dominant_period(t, values, 400.0, 1500.0) == pytest.approx(700.0, rel=0.02)
    assert owner.dominant_period(t, values, 2000.0, 5000.0) == pytest.approx(3000.0, rel=0.03)
    # The Nyquist period of a 1-yr step is 2 yr: nothing of the grid lies in [0.5, 1.5] yr.
    assert math.isnan(owner.dominant_period(t, values, 0.5, 1.5))


def test_body_mean_motion_at_steps_of_several_revolutions():
    # a = 3.2776 au: n(a) = 2 pi a^-1.5 = 1.05928 rad/yr. The true lambda_dot is n(a)(1 + 2e-4); at a 17-yr step
    # (~2.9 revolutions) the per-step model error is 2e-4 x 1.059 x 17 = 3.6e-3 rad, so the stitch is unique.
    t = np.arange(0.0, 17.0 * 2000, 17.0)
    a = np.full_like(t, 3.2776)
    n_true = TWO_PI * 3.2776**-1.5 * (1 + 2e-4)
    lam = np.mod(0.3 + n_true * t, TWO_PI)
    mask = np.ones_like(t, dtype=bool)
    assert owner.body_mean_motion(t, lam, a, mask) == pytest.approx(n_true, rel=1e-8)
    mask[: len(t) // 2] = False
    assert owner.body_mean_motion(t, lam, a, mask) == pytest.approx(n_true, rel=1e-8)


def test_planet_slow_deviations_keep_the_slow_term():
    # lambda_J = n_J t + 0.4 + 0.01 cos(2 pi t / 900) + 0.02 sin(2 pi t / 7), 10 whole 900-yr cycles: the cosine is
    # orthogonal to 1 and t, so the line is n_J t + 0.4. Cutoff 60 yr: gain 1/(1 + (60/900)^4) = 0.99998 at 900 yr,
    # 1/(1 + (60/7)^4) = 1.85e-4 at 7 yr. Expected deviation: 0.01 cos(2 pi t / 900).
    t = np.arange(0.0, 9000.0, 1.0)
    lam = N_J * t + 0.4 + 0.01 * np.cos(TWO_PI * t / 900) + 0.02 * np.sin(TWO_PI * t / 7)
    out = owner.planet_slow_deviations(t, {'Jupiter': np.mod(lam, TWO_PI)}, {'Jupiter': N_J}, 60.0, 2)
    mid = (t > 600) & (t < t[-1] - 600)
    expected = 0.01 * np.cos(TWO_PI * t / 900)
    assert np.max(np.abs(out['Jupiter'][mid] - expected[mid])) < 5e-4  # 5 % of the 0.01-rad term


# --------------------------------------------------------------------------------------------------- owner_required


def test_owner_required():
    # SPEC §4.8: needed when the angle oscillates; mandatory when the force law is not_measured; never when 'off'.
    assert owner.owner_required('pass', 1.0, True, S)
    assert owner.owner_required('fail', 1.0, True, S)
    assert not owner.owner_required('pass', 1.0, False, S)
    assert owner.owner_required('not_measured', math.radians(20), True, S)
    off = settings_from_config(mmr_owner_search='off')
    assert not owner.owner_required('not_measured', math.radians(20), True, off)
    assert not owner.owner_required('pass', 1.0, True, off)


# ------------------------------------------------------------------------------------- accept, better_than_sigma


def test_accept_every_rule_in_order():
    # Thresholds: R^2 >= 0.7, coherence >= 0.8, psi share <= 0.1, no slow turn, K_o <= 10 K_th, beats sigma.
    ok, reason = owner.accept(candidate(), 0.3, S, sigma_coherence=0.9, sigma_strength_ratio=0.5)
    assert ok and reason == 'accepted:r2'  # 0.95 >= 0.3 + 0.2
    cases = [
        (candidate(r2_median=0.6), 'r2_low'),
        (candidate(r2_median=NAN), 'r2_low'),  # an unmeasured number fails its test
        (candidate(phi0_coherence=0.7), 'phase_incoherent'),
        (candidate(psi_share=0.2), 'psi_share_high'),
    ]
    for cand, expected in cases:
        assert owner.accept(cand, 0.3, S, sigma_coherence=0.9, sigma_strength_ratio=0.5) == (False, expected)
    assert owner.accept(candidate(), 0.3, S, sigma_coherence=0.9, sigma_slow_turns=1) == (False, 'ripple_only')
    # W3-F3: the force over its mu scale (15 x) neither rejects nor accepts: mu e^q is a scale, not a bound.
    assert owner.accept(candidate(strength_ratio=15.0, force_au_yr=0.15), 0.3, S, sigma_coherence=0.9) == (True, 'accepted:r2')
    # The first failing test wins: R^2 before the coherence and the psi share.
    assert owner.accept(candidate(r2_median=0.5, phi0_coherence=0.1, psi_share=0.9), 0.3, S)[1] == 'r2_low'
    # sigma explains a_dot as well, with the same fixed phase and a reachable force: not better.
    assert owner.accept(candidate(), 0.9, S, sigma_coherence=0.97, sigma_strength_ratio=0.5) == (
        False,
        'not_better_than_sigma',
    )


def test_better_than_sigma_routes():
    cand = candidate()  # R^2 0.95, coherence 0.98
    assert owner.better_than_sigma(cand, 0.5, 0.97, 0.5, S) == 'r2'  # 0.95 >= 0.5 + 0.2
    assert owner.better_than_sigma(cand, 0.9, 0.6, 0.5, S) == 'phase'  # 0.98 >= 0.6 + 0.2
    assert owner.better_than_sigma(cand, 0.9, 0.97, 300.0, S) == ''  # W3-F3: no route by the mu scale
    assert owner.better_than_sigma(cand, 0.9, 0.97, 0.5, S) == ''
    # An unmeasured R^2 or coherence of sigma counts as 0: the owner wins by R^2.
    assert owner.better_than_sigma(cand, NAN, NAN, NAN, S) == 'r2'
    assert owner.accept(cand, 0.9, S, sigma_coherence=0.6, sigma_strength_ratio=0.5) == (True, 'accepted:phase')
    assert owner.accept(cand, 0.9, S, sigma_coherence=0.97, sigma_strength_ratio=300.0) == (False, 'not_better_than_sigma')


# ------------------------------------------------------------------------------------------- candidates and angles


def test_enumerate_owners_matches_a_brute_force_two_body_scan():
    # Bound 1.5 x 2 pi / 412.5 = 0.022848 rad/yr; n in 2 n_J +- 0.03. A candidate k1 lambda_J - k2 lambda +
    # (k2 - k1) varpi qualifies when its rate, linear in n, comes within the bound of 0 on the range; k1 <= 60,
    # |k1 - k2| <= 20, gcd 1; 5:2 (sigma's own lambda part) is excluded. Near 2:1, k1 = 2 k2 +- 1 turn at
    # +-n_J -+ 0.03 k2 at the ends, so k2 ~ 17..20 qualify besides 2:1 itself.
    series, context, _, _ = scenario('libration', '5J-2+0-3', P_LIB_40)
    n_lo, n_hi = 2 * N_J - 0.03, 2 * N_J + 0.03
    bound = 1.5 * TWO_PI / P_LIB_40
    found = owner.enumerate_owners(
        series.resonance,
        context,
        P_LIB_40,
        ('Jupiter',),
        S,
        period_yr=P_LIB_40,
        body_n_range=(n_lo, n_hi),
        body_varpi_rate=VARPI_RATE,
    )
    keys = {spec.key for spec in found}
    n_jupiter = owner.planet_mean_motion(context, 'Jupiter')
    assert n_jupiter == pytest.approx(N_J, rel=1e-6)
    expected, marginal = set(), set()
    for k1 in range(1, 61):
        for k2 in range(max(1, k1 - 20), k1 + 21):
            if math.gcd(k1, k2) != 1 or (k1, k2) == (5, 2):
                continue
            ends = [k1 * N_J - k2 * n + (k2 - k1) * VARPI_RATE for n in (n_lo, n_hi)]
            distance = 0.0 if ends[0] * ends[1] <= 0 else min(abs(ends[0]), abs(ends[1]))
            key = owner.owner_key(('Jupiter',), (k1,), -k2, k2 - k1)
            if abs(distance - bound) < 0.02 * bound:
                marginal.add(key)
            elif distance <= bound:
                expected.add(key)
    assert '2J-1+0-1' in expected and len(expected) > 3
    assert keys - marginal == expected
    assert all(spec.kind == ResonanceKind.TWO_BODY and spec.planets == ('Jupiter',) for spec in found)


def test_enumerate_owners_limits_and_exclusions():
    series, context, _, _ = scenario('libration', '4J-5S-1+0+0+2', P_LIB_40)
    spec = series.resonance
    found = owner.enumerate_owners(spec, context, P_LIB_40, ('Jupiter', 'Saturn'), S)
    keys = [c.key for c in found]
    # The true owner and the commensurate 5S-1+0-4 (5 n_S - n - 4 varpi' = phi_o' - 2 pi / 25 kyr) are candidates.
    assert '2J-1+0-1' in keys and '5S-1+0-4' in keys
    # Far from zero: 3J-1+0-2 turns at 3 n_J - n = 0.53 rad/yr >> 0.023.
    assert '3J-1+0-2' not in keys
    for cand in found:
        assert not owner.same_lambda_part(cand, spec)
        assert set(cand.planets) <= {'Jupiter', 'Saturn'}
        coefficients = list(cand.planet_coefficients) + [cand.body_coefficient]
        assert math.gcd(*[abs(c) for c in coefficients]) == 1
        assert sum(coefficients) + cand.body_varpi_coefficient == 0
        if cand.kind == ResonanceKind.TWO_BODY:
            assert 1 <= cand.planet_coefficients[0] <= 60
            assert abs(cand.planet_coefficients[0] + cand.body_coefficient) <= 20
        else:
            assert max(abs(c) for c in coefficients + [cand.body_varpi_coefficient]) <= 12
    # Only planets with a saved mean longitude qualify.
    only_jupiter = owner.enumerate_owners(spec, context, P_LIB_40, ('Jupiter',), S)
    assert only_jupiter and all(c.planets == ('Jupiter',) for c in only_jupiter)
    assert owner.enumerate_owners(spec, context, P_LIB_40, (), S) == []


def test_owner_angle_reproduces_the_true_angles():
    series, context, _, _ = scenario('libration', '4J-5S-1+0+0+2', P_LIB_40)
    phi, _ = owner_motion('libration')
    planet_n = {name: owner.planet_mean_motion(context, name) for name in ('Jupiter', 'Saturn')}
    psi = 2 * (N_J * T_REC + 0.4) - 5 * (N_S * T_REC + 2.1) + 3 * (1.3 + VARPI_RATE * T_REC)
    for key, truth in [('2J-1+0-1', phi), ('5S-1+0-4', phi - psi)]:
        # 5S-1+0-4 = 5 lambda_S - lambda - 4 varpi = phi_o - psi (the identity of the synthetic Saturn).
        angle = owner.owner_angle(ResonanceSpec.from_key(key), series, planet_n)
        offset = angle - truth
        turns = np.round(offset / TWO_PI)
        assert np.all(turns == turns[0])  # unwrapped as the truth, no 2 pi slips
        assert np.max(np.abs(offset - TWO_PI * turns[0])) < 1e-6
    angle, residual = owner.owner_angle_with_residual(ResonanceSpec.from_key('2J-1+0-1'), series, planet_n)
    assert len(residual) == len(T_REC) - 1
    assert owner.unwrap_ok(residual, S)


# ------------------------------------------------------------------------------------------------------- run_owner


def test_induced_libration_is_found_by_r2():
    # sigma = 5J-2+0-3 = 2 phi_o + psi, psi turning uniformly (a line in the band): a is fixed to phi_o.
    _, _, _, result = scenario('libration', '5J-2+0-3', P_LIB_40)
    assert result.check == CheckValue.FAIL
    assert result.accepted is not None and result.accepted.spec.key == '2J-1+0-1'
    assert result.accepted.accepted and result.accepted.rank == 1
    assert result.accepted.reason.startswith('accepted')
    assert result.detail == 'owner_libration'
    assert result.accepted.r2_median > 0.95 and result.accepted.phi0_coherence > 0.95
    assert result.accepted.psi_share < 0.01
    assert result.sigma_r2 < 0.1  # sin/cos of a fast-turning sigma do not fit a_dot
    # psi = lambda_J - varpi turns at n_J - varpi' = 0.529662 - 0.000157 = 0.529505 rad/yr.
    assert result.psi_rate_rad_yr == pytest.approx(N_J - VARPI_RATE, rel=1e-3)
    # The owner librates: its mean rate is ~0 (bound 0.023 rad/yr).
    assert abs(result.accepted.rate_rad_yr) < 1e-3


def test_induced_force_matches_the_identity():
    # a_dot = -(2 a / 3 n) omega0^2 sin phi_o: R = 2 x 3.2776 / (3 x 1.05925) x (2 pi / 400)^2 = 5.090e-4 au/yr;
    # the band [P/2, 2P] passes 0.882 of it (test_band_pass_gain_closed_form) -> K_o ~ 4.49e-4.
    # K_th = 2 |k| n a mu_J = 2 x 1.05925 x 3.2776 x 9.5479e-4 = 6.630e-3 au/yr.
    _, context, _, result = scenario('libration', '5J-2+0-3', P_LIB_40)
    n = 2 * N_J - VARPI_RATE
    a = (TWO_PI / n) ** (2.0 / 3.0)
    amplitude = 2 * a / (3 * n) * OMEGA_LIB**2
    best = result.accepted
    assert best.force_au_yr == pytest.approx(0.882 * amplitude, rel=0.05)
    assert best.force_theory_au_yr == pytest.approx(2 * n * a * MU_J, rel=2e-3)
    assert best.strength_ratio == pytest.approx(best.force_au_yr / best.force_theory_au_yr, rel=1e-6)
    columns = result.to_columns()
    assert columns['own_name'] == '2J-1+0-1'
    assert columns['own_K_au_yr'] == pytest.approx(best.force_au_yr)
    assert columns['own_K_theory_au_yr'] == pytest.approx(best.force_theory_au_yr)
    assert columns['own_psi_rate_rad_kyr'] == pytest.approx(1000 * (N_J - VARPI_RATE), rel=1e-3)
    assert columns['own_n_candidates'] == result.n_candidates >= 1
    assert len(str(columns['own_R2_windows']).split(';')) == len(best.r2_windows) >= 2


def test_induced_libration_of_a_three_body_angle():
    # sigma = 4J-5S-1+0+0+2 = phi_o + psi, psi slow (25 kyr): sigma fits a_dot per window too, but its phase drifts
    # with psi (~170 deg over the record) and a three-body sigma lacks the force (mu_J mu_S = 2.7e-7).
    _, _, _, result = scenario('libration', '4J-5S-1+0+0+2', P_LIB_40)
    assert result.check == CheckValue.FAIL
    assert result.accepted.spec.key == '2J-1+0-1'
    assert result.detail == 'owner_libration'
    # 5S-1+0-4 = phi_o - psi follows a too, but its phase drifts with psi: never the owner.
    others = [c for c in result.candidates if c.spec.key == '5S-1+0-4']
    assert others and not others[0].accepted


def test_own_libration_has_no_owner():
    # sigma = 2J-1+0-1 is the pendulum itself: nothing explains a_dot better than sigma.
    _, _, _, result = scenario('libration', '2J-1+0-1', P_LIB_40)
    assert result.check == CheckValue.PASS
    assert result.accepted is None
    assert result.sigma_r2 > 0.95
    assert all(not c.accepted for c in result.candidates)


def test_forced_wave_owner_without_a_cycle_period():
    # The owner circulates (energy above the separatrix); sigma = 5J-2+0-3 circulates as well, so there is no
    # P_cyc. Mean period of the owner: 2 pi / mean phi_o' (measured on the synthetic track, ~488 yr).
    phi, _ = owner_motion('forced_wave')
    period = TWO_PI * T_REC[-1] / (phi[-1] - phi[0])
    _, _, _, result = scenario('forced_wave', '5J-2+0-3', NAN)
    assert result.check == CheckValue.FAIL
    assert result.accepted.spec.key == '2J-1+0-1'
    assert result.detail == 'owner_forced_wave'
    assert abs(result.accepted.rate_rad_yr) == pytest.approx(TWO_PI / period, rel=0.05)
    # a waves at the owner's rate, not at sigma's.
    assert result.freq_r2_owner > 0.9 > 0.1 > result.freq_r2_sigma
    assert result.freq_period_yr == pytest.approx(period, rel=0.03)


def test_forced_wave_of_a_three_body_angle():
    # W3-F3: without the mu route the owner 2J-1 is not accepted (phi0 coherence gain 0.18 < 0.2, R^2 0.994 vs 0.976),
    # but it leads the candidates and is flagged probable; the frequency test separates it from sigma.
    _, _, _, result = scenario('forced_wave', '4J-5S-1+0+0+2', NAN)
    assert result.check == CheckValue.PASS and result.accepted is None
    assert result.candidates[0].spec.key == '2J-1+0-1' and owner.FLAG_PROBABLE in result.flags
    assert result.freq_r2_owner > 0.9 > result.freq_r2_sigma


def test_circulating_angle_driving_a_itself_has_no_owner():
    _, _, _, result = scenario('forced_wave', '2J-1+0-1', NAN)
    assert result.check == CheckValue.PASS
    assert result.accepted is None


def test_owner_crossing_its_separatrix():
    phi, _ = owner_motion('crossing')
    half = len(T_REC) // 2
    assert np.max(np.abs(phi[:half])) < np.pi  # librates in the first half
    assert np.max(np.abs(phi[half:])) > 4 * np.pi  # circulates in the second
    _, _, _, induced = scenario('crossing', '5J-2+0-3', 450.0)
    assert induced.check == CheckValue.FAIL
    assert induced.accepted.spec.key == '2J-1+0-1'
    assert induced.detail in ('owner_libration', 'owner_forced_wave')
    _, _, _, own = scenario('crossing', '2J-1+0-1', 450.0)
    assert own.check == CheckValue.PASS
    assert own.accepted is None


def test_fit_owner_on_the_true_owner():
    series, context, prep, _ = scenario('libration', '5J-2+0-3', P_LIB_40)
    spec = ResonanceSpec.from_key('2J-1+0-1')
    planet_n = {'Jupiter': owner.planet_mean_motion(context, 'Jupiter')}
    fitted = owner.fit_owner(spec, owner.owner_angle(spec, series, planet_n), prep, context, P_LIB_40, S, series=series)
    assert fitted.spec.key == '2J-1+0-1'
    assert fitted.r2_median > 0.95 and fitted.phi0_coherence > 0.95
    assert fitted.psi_share < 0.01
    assert fitted.strength_ratio < 1.0


# ------------------------------------------------------------------------------------------------------ edge cases


def test_not_measured_without_planet_longitudes():
    phi, phidot = owner_motion('libration')
    series = build_series('5J-2+0-3', phi, phidot, T_REC, with_saturn=False)
    bare = dataclasses.replace(series, planets={}, regime=dataclasses.replace(series.regime, planets_saved=()))
    context, prep = chain(bare)
    result = owner.run_owner(bare, prep, context, P_LIB_40, S)
    assert result.check == CheckValue.NOT_MEASURED
    assert result.accepted is None
    assert 'owner_no_longitudes' in result.flags


def test_not_measured_when_search_is_off():
    series, context, prep, _ = scenario('libration', '5J-2+0-3', P_LIB_40)
    result = owner.run_owner(series, prep, context, P_LIB_40, settings_from_config(mmr_owner_search='off'))
    assert result.check == CheckValue.NOT_MEASURED
    assert result.accepted is None
    assert 'owner_search_off' in result.flags


def test_not_measured_on_a_record_too_short_for_two_windows():
    # Two windows need 5.25 P = 2166 yr at P = 412.5 yr; the record is 1200 yr.
    t = T_REC[:1200]
    phi, phidot = pendulum(OMEGA_LIB, math.radians(40), 0.0, t)
    series = build_series('5J-2+0-3', phi, phidot, t, with_saturn=False)
    context, prep = chain(series)
    result = owner.run_owner(series, prep, context, P_LIB_40, S)
    assert result.check == CheckValue.NOT_MEASURED
    assert result.accepted is None
    assert {'owner_too_few_windows', 'owner_no_period'} & set(result.flags)


def test_retrograde_angle_is_not_applicable():
    series, context, prep, _ = scenario('libration', '5J-2+0-3', P_LIB_40)
    retro = dataclasses.replace(series, resonance=ResonanceSpec.from_key('1J-3r0'))
    result = owner.run_owner(retro, prep, context, P_LIB_40, S)
    assert result.check == CheckValue.NOT_APPLICABLE
    assert result.accepted is None
