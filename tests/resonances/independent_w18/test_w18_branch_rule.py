"""Independent tests of W18 item 3: the branch "L4 and R0 both not measured" (docs/physics/status-tree.md, the table
under "The unmeasured branch"; author's table in ANSWERS 2026-10-17c).

Every record here is a three-body pendulum of half-amplitude 20 deg: below the 45-deg floor of the force law (L4) and
of R0 without a turn, so both are not measured by design, and the leaf is chosen by the strength alone.

References computed here, not read from the package:
- required strength mu_req = (omega0 / (|k| n))^2 / 3, omega0 = 2 pi / T0 (Murray & Dermott 1999, eqs. 8.53-8.58),
  T0 the pendulum's small period (a 20-deg swing is longer by F(20 deg) - 1 = 0.8 %, not more);
- the three-body scale mu_J mu_S e^q (IAU mass ratios 1/1047.3486 and 1/3497.9018; Nesvorny & Morbidelli 1998);
- the small-divisor envelope by our own enumeration of the documented rule: two-body angles k1 lambda_P - k2 lambda
  of Jupiter and Saturn, k1 <= 60, |k2 - k1| <= 20, |rate| <= 1.5 x 2 pi / P somewhere in the a range [p01, p99] of the
  record; a rate that changes sign across that range gives no bound; B = n / min |rate| (n / bound without one).
"""

import functools

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.context import table_mean_motion
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings

SETTINGS = AnalysisSettings()
MU = {'Jupiter': 1.0 / 1047.3486, 'Saturn': 1.0 / 3497.9018}
# a, e, inc, Omega, varpi of the planets (held constant)
PLANETS = {'Jupiter': (5.2026, 0.048, 0.0228, 1.753, 0.257), 'Saturn': (9.5549, 0.054, 0.0434, 1.984, 1.613)}
HALF_AMPLITUDE_DEG = 20.0


def _pendulum(t, period0_yr, amp_deg):
    w0 = 2 * np.pi / period0_yr
    sol = solve_ivp(
        lambda _, y: [y[1], -(w0**2) * np.sin(y[0])],
        (t[0], t[-1]),
        [np.radians(amp_deg), 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        method='DOP853',
    )
    return sol.y[0], sol.y[1]


def _series(t, sigma, sigma_dot, key, e):
    """The body's lambda and a follow sigma exactly (sigma_dot = k n + sum k_j n_j, varpi fixed)."""
    spec = ResonanceSpec.from_key(key)
    ones = np.ones_like(t)
    lam_rest, n_rest, planets = np.zeros_like(t), 0.0, {}
    for planet, k_p in zip(spec.planets, spec.planet_coefficients):
        n_p = table_mean_motion(planet)
        lam_p = n_p * t + 0.3
        lam_rest, n_rest = lam_rest + k_p * lam_p, n_rest + k_p * n_p
        a_p, e_p, i_p, node_p, varpi_p = PLANETS[planet]
        planets[planet] = PlanetSeries(
            planet, t, a_p * ones, e_p * ones, i_p * ones, node_p * ones, varpi_p * ones, np.mod(lam_p, 2 * np.pi)
        )
    k, p, varpi = spec.body_coefficient, spec.body_varpi_coefficient, 1.0
    lam = (sigma - lam_rest - p * varpi) / k
    n = (sigma_dot - n_rest) / k
    return MMRSeries(
        t_yr=t,
        a=(2 * np.pi / n) ** (2.0 / 3.0),
        e=e * ones,
        inc=0.05 * ones,
        Omega=0.4 * ones,
        omega=(varpi - 0.4) * ones,
        varpi=varpi * ones,
        angle_wrapped=np.mod(lam_rest + k * lam + p * varpi, 2 * np.pi),
        resonance=spec,
        regime=Regime(float(t[1] - t[0]), float(t[-1] - t[0]), len(t)),
        planets=planets,
        mean_longitude=np.mod(lam, 2 * np.pi),
        name='synthetic',
    )


@functools.lru_cache(maxsize=None)
def _run(key, period0_yr, e, span_yr, step_yr):
    t = np.arange(0.0, span_yr + step_yr / 2, step_yr)
    sigma, sigma_dot = _pendulum(t, period0_yr, HALF_AMPLITUDE_DEG)
    series = _series(t, sigma, sigma_dot, key, e)
    return series, analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False)


# --------------------------------------------------------------------------------------------------------------------
# independent references


def _reference_ratio(series, period0_yr, e):
    spec = series.resonance
    n = 2 * np.pi / np.mean(series.a) ** 1.5
    mu_required = (2 * np.pi / period0_yr / (abs(spec.body_coefficient) * n)) ** 2 / 3
    order = abs(sum(spec.planet_coefficients) + spec.body_coefficient)
    return mu_required / (MU['Jupiter'] * MU['Saturn'] * e**order)


def _reference_envelope(series, period0_yr):
    """(B, crosses_zero) by the documented enumeration (module docstring)."""
    a01, a99 = np.percentile(series.a, [1, 99])
    ends = [2 * np.pi / a01**1.5, 2 * np.pi / a99**1.5]
    bound = 1.5 * 2 * np.pi / period0_yr
    nearest = np.inf
    for planet in series.resonance.planets:
        n_p = table_mean_motion(planet)
        for k1 in range(1, 61):
            for k2 in range(max(1, k1 - 20), k1 + 21):
                rates = [k1 * n_p - k2 * n for n in ends]
                if np.sign(rates[0]) != np.sign(rates[1]):
                    return np.inf, True
                low = min(abs(r) for r in rates)
                if low <= bound:
                    nearest = min(nearest, low)
    n = 2 * np.pi / np.mean(series.a) ** 1.5
    return n / (nearest if np.isfinite(nearest) else bound), False


def _row(analysis):
    return analysis.numbers_row()


# --------------------------------------------------------------------------------------------------------------------
# both physics checks are out by design


@pytest.mark.parametrize('case', [('7J-1S-2', 300.0, 0.02, 20000.0, 1.0), ('3J-1S-1', 8000.0, 0.2, 80000.0, 2.0)])
def test_small_swing_leaves_force_law_and_potential_unmeasured(case):
    _, analysis = _run(*case)
    checks = analysis.verdict.checks
    assert str(checks['check_mmr_a_response']) == 'not_measured'
    assert str(checks['check_mmr_potential']) == 'not_measured'
    assert _row(analysis)['str_method'] == 'eq_scale'  # three-body: no direct averaging, leaf (i) cannot fire


# --------------------------------------------------------------------------------------------------------------------
# 5s: KL9-like, the gap cannot be bridged


def test_kl9_like_implausible_three_body_is_non_resonant():
    """7J-1S-2 (order 4) at a = 2.347 au, e = 0.02, T0 = 300 yr. Required 1.20e-5, scale 2.73e-7 x 0.02^4 = 4.37e-14:
    ratio 2.7e8. No two-body angle crosses its commensurability in the a range; the nearest (23J-7) leaves B = 77.5,
    B^2 = 6.0e3: the envelope still misses 4.6e4 >= 1e3."""
    series, analysis = _run('7J-1S-2', 300.0, 0.02, 20000.0, 1.0)
    ratio = _reference_ratio(series, 300.0, 0.02)
    b, crosses = _reference_envelope(series, 300.0)
    assert not crosses
    assert ratio == pytest.approx(2.75e8, rel=0.02)
    assert ratio / b**2 > 10 * SETTINGS.mmr_strength_implausible_ratio
    row = _row(analysis)
    assert row['str_ratio'] == pytest.approx(ratio, rel=0.03)
    assert row['str_divisor_B'] == pytest.approx(b, rel=0.01)
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.reason), str(verdict.libration_origin)) == ('non_resonant', 'implausible_strength', 'unknown')
    assert 'implausible_strength' in verdict.caps
    assert str(verdict.confidence) == 'low'
    assert str(verdict.angle_status) == 'libration'  # the geometry is not denied, only its origin


# --------------------------------------------------------------------------------------------------------------------
# 5s*: small divisors can close the gap


def test_same_resonance_rescuable_by_the_nearest_divisor_is_uncertain():
    """The same 7J-1S-2 at e = 0.09: ratio 2.75e8 x (0.02 / 0.09)^4 = 6.7e5, still >= 1e3, but B^2 = 6.0e3 leaves
    only ~1e2 < 1e3: uncertain, not non_resonant."""
    series, analysis = _run('7J-1S-2', 300.0, 0.09, 20000.0, 1.0)
    ratio = _reference_ratio(series, 300.0, 0.09)
    b, crosses = _reference_envelope(series, 300.0)
    assert not crosses
    assert ratio >= 100 * SETTINGS.mmr_strength_implausible_ratio
    assert ratio / b**2 < SETTINGS.mmr_strength_implausible_ratio / 3
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.reason)) == ('uncertain', 'origin_unknown')
    assert 'implausible_strength' not in verdict.caps


def test_a_range_across_a_two_body_commensurability_gives_no_bound():
    """3J-1S-1 at a = 2.753 au sits next to 13J-5: 5 (3 n_J - n_S) = 13 n_J + (2 n_J - 5 n_S), the great inequality.
    With T0 = 300 yr the swing of a passes 13 n_J = 5 n. Since W22 R-DIV (author 2026-10-24) a crossing is no unbounded
    B: 13J-5 turns slower than sigma swings, so B = n T0 / 2 pi = 1.3746 x 300 / 2 pi = 65.6, B^2 = 4.3e3; ratio
    ~2.8e3 leaves 0.66 < 1e3 -> still uncertain."""
    series, analysis = _run('3J-1S-1', 300.0, 0.1, 20000.0, 1.0)
    ratio = _reference_ratio(series, 300.0, 0.1)
    _, crosses = _reference_envelope(series, 300.0)
    assert crosses
    a01, a99 = np.percentile(series.a, [1, 99])
    n_j = table_mean_motion('Jupiter')
    assert np.sign(13 * n_j - 5 * 2 * np.pi / a01**1.5) != np.sign(13 * n_j - 5 * 2 * np.pi / a99**1.5)
    assert ratio == pytest.approx(2.84e3, rel=0.03)
    assert ratio >= SETTINGS.mmr_strength_implausible_ratio
    row = _row(analysis)
    assert row['str_divisor_kind'] == 'crosses_zero'
    assert row['str_divisor_candidate'] == '13J-5'
    n = 2 * np.pi / np.mean(series.a) ** 1.5
    assert n * 300.0 / (2 * np.pi) == pytest.approx(65.6, rel=0.01)
    assert row['str_divisor_B'] == pytest.approx(n * 300.0 / (2 * np.pi), rel=0.02)
    assert row['str_divisor_residual'] < SETTINGS.mmr_strength_implausible_ratio
    verdict = analysis.verdict
    assert (str(verdict.status), str(verdict.reason)) == ('uncertain', 'origin_unknown')


# --------------------------------------------------------------------------------------------------------------------
# 5n against 5u: one independent cycle is no measurement of the centre


def test_one_cycle_with_a_plausible_strength_is_uncertain():
    """3J-1S-1, e = 0.2, T0 = 8 kyr on 13 kyr: the theory's period is 2 pi / (n sqrt(3 mu_J mu_S e)) = 11.3 kyr, so
    the ratio (11.3 / 8)^2 = 2.0 passes F (no objection). The TPs inside [75 yr, 12925 yr] are 4, 8 and 12 kyr: one
    full cycle, one independent centre (< libration_centre_min_estimates = 2)."""
    series, analysis = _run('3J-1S-1', 8000.0, 0.2, 13000.0, 2.0)
    ratio = _reference_ratio(series, 8000.0, 0.2)
    assert ratio == pytest.approx(1.99, rel=0.03)
    accepted = [e for e in analysis.geometry.episodes if e.accepted]
    assert len(accepted) == 1
    assert accepted[0].n_tp == 3
    assert (accepted[0].n_tp - 1) // 2 < SETTINGS.libration_centre_min_estimates
    verdict = analysis.verdict
    assert str(verdict.checks['check_mmr_strength']) == 'pass'
    assert str(verdict.checks['check_mmr_onset']) != 'fail'
    assert (str(verdict.status), str(verdict.reason), str(verdict.libration_origin)) == ('uncertain', 'origin_unknown', 'unknown')


def test_many_cycles_with_nothing_measured_but_the_scale_keep_the_geometry_at_medium():
    """The same pendulum on 80 kyr: ten cycles, nothing measured but the order-of-magnitude scale (not a physics
    measurement): resonant by the geometry, origin unknown. The scale agrees within [1/10, 10], so the cap is
    strength_scale_only <= medium, not geometry_only <= low (W22 R-F, author 2026-10-24)."""
    series, analysis = _run('3J-1S-1', 8000.0, 0.2, 80000.0, 2.0)
    accepted = [e for e in analysis.geometry.episodes if e.accepted]
    assert len(accepted) == 1 and (accepted[0].n_tp - 1) // 2 >= SETTINGS.libration_centre_min_estimates
    ratio = _reference_ratio(series, 8000.0, 0.2)
    assert 1 / SETTINGS.mmr_strength_check_ratio <= ratio <= SETTINGS.mmr_strength_check_ratio
    verdict = analysis.verdict
    assert str(verdict.status) == 'resonant'
    assert str(verdict.libration_origin) == 'unknown'
    assert 'strength_scale_only' in verdict.caps and 'geometry_only' not in verdict.caps
    assert str(verdict.confidence) == 'medium'


def test_ratio_between_ten_and_a_thousand_is_a_measured_objection():
    """3J-1S-1 at e = 0.1 with T0 = 2 kyr: ratio 2.84e3 x (300 / 2000)^2 = 64, outside [1/10, 10] and below 1e3:
    F fails -> rule 5o, uncertain (neither implausible nor the geometry's status)."""
    series, analysis = _run('3J-1S-1', 2000.0, 0.1, 40000.0, 2.0)
    ratio = _reference_ratio(series, 2000.0, 0.1)
    assert SETTINGS.mmr_strength_check_ratio < ratio < SETTINGS.mmr_strength_implausible_ratio
    verdict = analysis.verdict
    assert str(verdict.checks['check_mmr_strength']) == 'fail'
    assert (str(verdict.status), str(verdict.reason)) == ('uncertain', 'origin_unknown')
    assert 'implausible_strength' not in verdict.caps and 'geometry_only' not in verdict.caps
