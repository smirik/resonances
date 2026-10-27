"""W2 zone-rules: the modulation of the force and the numeric zone the separatrix reaches (SPEC §4.6, §4.6.1).

Expected values come from outside the code under test: the multiplet algebra done by hand (W(t) = C0 + C1 e^{i beta}),
the e^q half-range of PHYSICS B1a (52356: e 0.078-0.146 gives dK/K = 0.304, not the 0.609 of the first draft), the
adiabatic edge sqrt(1 - m) (Neishtadt 1986 via Shevchenko 2020, §5.4) and the reviewer's independent integration
`reviews/scripts/zone_reach_J.py` (m = 0.55: lowest crossing J0/<J_s> 0.62 at lambda = 0.1, 0.32 at lambda = 0.5).
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import pendulum
from resonances.resonance.analysis.mmr import zone
from resonances.resonance.analysis.pendulum import ZoneMethod
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
# a coarser grid keeps the integrations of the tests to seconds
COARSE = settings_from_config(mmr_zone_j_step=0.05, mmr_zone_phases=8, mmr_zone_rtol=1e-6, mmr_zone_j_min=0.2)


def multiplet_record(c, c0, c1, nu, span=60000.0, step=5.0):
    """sigma_ddot = c a_dot with a_dot = Im[W e^{i sigma}], W = c0 + c1 e^{i nu t}: the multiplet of SEPARATRIX §2.3."""

    def amplitude(t):
        return c0 + c1 * np.exp(1j * nu * t)

    def rhs(t, y):
        return [y[1], c * np.imag(amplitude(t) * np.exp(1j * y[0]))]

    t = np.arange(0.0, span, step)
    sol = solve_ivp(rhs, (t[0], t[-1]), [2.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-13)
    sigma = sol.y[0]
    return t, sigma, np.imag(amplitude(t) * np.exp(1j * sigma)), nu * t, amplitude(t)


@pytest.mark.parametrize('c', [-0.75, 0.6])
def test_members_modulation_recovers_the_multiplet(c):
    c0 = 2e-7 * np.exp(0.4j)
    t, sigma, adot, beta, w = multiplet_record(c, c0, 0.5 * c0 * np.exp(1.1j), 2.5e-4)
    omega2, well, eps, gain = zone.modulation_from_members(t, sigma, adot, beta, c, (0, 1))
    # by hand: Omega^2 = |c||W|; the stable point -arg W (c < 0) or -arg W - pi (c > 0); eps = |C1|/|C0| = 0.5
    assert omega2 == pytest.approx(abs(c) * np.abs(w), rel=1e-9)
    expected = -np.angle(w) - (np.pi if c > 0 else 0.0)
    assert np.max(np.abs(np.angle(np.exp(1j * (well - expected))))) < 1e-9
    assert eps == pytest.approx((0.5,), rel=1e-9)
    assert gain > 0.1
    # the model is the equation of motion: sigma_ddot = -Omega^2 sin(sigma - sigma_s)
    acceleration = c * adot
    assert acceleration == pytest.approx(-omega2 * np.sin(sigma - well), abs=1e-12)


def test_members_modulation_without_a_usable_fit_is_nan():
    t = np.linspace(0, 100, 50)
    omega2, well, eps, gain = zone.modulation_from_members(t, t, t, t, 0.0, (0, 1))
    assert np.all(np.isnan(omega2)) and np.all(np.isnan(well)) and np.isnan(gain) and np.isnan(eps[0])


def test_eccentricity_modulation_takes_the_half_range_and_leaves_the_trend_out():
    t = np.linspace(0, 100000, 4001)
    oscillation = 0.112 + 0.034 * np.cos(2 * np.pi * t / 8500)  # e from 0.078 to 0.146 (52356, PHYSICS B1a)
    trend = 1.0e-5  # per kyr
    e = oscillation + trend * (t - t.mean()) / 1000
    nu = 2 * np.pi / 8500  # 11.8 cycles: an unfinished one tilts a plain straight line
    omega2, dk, e_trend = zone.modulation_from_eccentricity(t, e, 1, 2.0, nu)
    assert dk == pytest.approx(0.034 / 0.112, rel=1e-3)  # 0.304, half of the first draft's 0.609
    assert e_trend == pytest.approx(trend, rel=1e-3)
    assert omega2 == pytest.approx(4.0 * oscillation / 0.112, rel=1e-3)  # Omega_bar^2 (e_osc / e_bar)^q, no trend
    _, plain, _ = zone.modulation_from_eccentricity(t, e, 1, 2.0)  # without the frequency: within 5 %
    assert plain == pytest.approx(dk, rel=0.05)
    _, dk2, _ = zone.modulation_from_eccentricity(t, e, 2, 2.0, nu)
    assert dk2 == pytest.approx(2 * 0.034 / 0.112, rel=1e-3)
    flat, dk0, _ = zone.modulation_from_eccentricity(t, e, 0, 2.0, nu)
    assert dk0 == 0.0 and np.all(flat == 4.0)


def test_zone_starts_spread_over_one_modulation_period():
    assert zone.zone_starts(0.0, 100.0, 40.0, 4) == pytest.approx([0, 10, 20, 30])
    assert zone.zone_starts(0.0, 100.0, np.nan, 4) == pytest.approx([0, 25, 50, 75])
    assert zone.zone_starts(0.0, 100.0, 400.0, 2) == pytest.approx([0, 50])


def _modulated(m, lam, periods):
    span = periods * 2 * np.pi / lam
    t = np.linspace(0.0, span, 4001)
    omega2 = 1 + m * np.cos(lam * t)
    return t, omega2


def test_no_modulation_reaches_nothing():
    t = np.linspace(0, 300, 1001)
    j_lo, h_hi = zone.numeric_zone(t, lambda x: np.ones_like(x), lambda x: np.zeros_like(x), COARSE, np.nan, 1.0)
    assert (j_lo, h_hi) == (1.0, 1.0)


def test_slow_pulsation_reaches_the_adiabatic_edge():
    # lambda = 0.05: the action is conserved, the zone is J0/<J_s> >= Omega_min/Omega_bar = sqrt(1 - m) (0.67)
    t, omega2 = _modulated(0.55, 0.05, 4)
    omega_bar = float(np.mean(np.sqrt(omega2)))
    j_lo, h_hi = zone.numeric_zone(t, lambda x: np.interp(x, t, omega2), lambda x: np.zeros_like(x), COARSE, 2 * np.pi / 0.05, omega_bar)
    edge = np.sqrt(1 - 0.55) / omega_bar  # Omega_min over the mean Omega
    assert j_lo == pytest.approx(edge, abs=COARSE.mmr_zone_j_step + 0.01)
    assert h_hi > 1.0


def test_faster_pulsation_reaches_deeper_through_secondary_resonances():
    # lambda = 0.5, m = 0.55: the reviewer's integration finds orbits down to J0/<J_s> = 0.32 crossing (zone_reach_J.py)
    t, omega2 = _modulated(0.55, 0.5, 40)
    j_lo, _ = zone.numeric_zone(t, lambda x: np.interp(x, t, omega2), lambda x: np.zeros_like(x), COARSE, 2 * np.pi / 0.5, 1.0)
    assert j_lo == pytest.approx(0.32, abs=0.06)
    assert j_lo < np.sqrt(1 - 0.55) - 0.2  # far below the adiabatic edge


def test_build_zone_methods_and_the_adiabatic_cross_check():
    t, omega2 = _modulated(0.3, 0.05, 4)
    relative = np.sqrt(omega2) / np.mean(np.sqrt(omega2))
    adiabatic = settings_from_config(mmr_zone_method='adiabatic')
    analytic = zone.build_zone(t, omega2, np.zeros_like(t), 0.05, adiabatic)
    assert analytic.method == ZoneMethod.ADIABATIC
    assert analytic.j_lo == pytest.approx(relative.min()) and analytic.h_hi == pytest.approx(relative.max() ** 2)
    numeric = zone.build_zone(t, omega2, np.zeros_like(t), 0.05, COARSE, 2 * np.pi / 0.05, float(np.mean(np.sqrt(omega2))))
    assert numeric.method == ZoneMethod.NUMERIC
    assert numeric.adiabatic_lo == pytest.approx(relative.min())
    assert not numeric.mismatch  # lambda < 0.1 and the numeric J_lo sits at the adiabatic edge
    assert numeric.h_lo == pytest.approx(pendulum.h_from_action_ratio(numeric.j_lo))
    eq = zone.build_zone(t, omega2, np.zeros_like(t), 0.05, COARSE, method=ZoneMethod.EQ_ONLY)
    assert eq.method == ZoneMethod.EQ_ONLY


def test_separatrix_cutoff_follows_the_cycle():
    assert zone.separatrix_cutoff_yr(20.0, 6000.0, S) == pytest.approx(max(20.0, S.separatrix_cutoff_cycles * 6000))
    assert zone.separatrix_cutoff_yr(20.0, np.nan, S) == 20.0
    assert zone.separatrix_cutoff_yr(500.0, 1000.0, S) == 500.0  # never below P_c
