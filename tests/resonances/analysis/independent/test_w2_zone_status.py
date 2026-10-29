"""Independent tests of the separatrix zone, the status tree, the confidence rule and the filter guard.

Written from tmp/classifier-v2-2026-10-15/impl/API.md, the public docstrings and SPEC §4.6, §5.3, §5.4, §7,
SPEC-DELTA item 13 and benchmark/DECISIONS.md (2026-10-15). Expected values come from closed forms, from
pendulums integrated here with solve_ivp, and from hand calculations written next to each test.

Units as in the package: years, radians, au; c = identity coefficient [rad yr^-1 au^-1].
"""

import dataclasses
import itertools
import json
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis import filtering
from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis import pendulum
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import decide, models, preprocess, zone
from resonances.resonance.analysis.pendulum import CycleClass, CycleEnergy, EnergyAnalysis, Zone, ZoneMethod
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    LibrationOrigin,
    MarginClass,
    MarginValue,
    Reason,
    Status,
)

SETTINGS = settings_from_config()
NAN = float('nan')
C_ID = -3.0  # identity coefficient of a typical inner-belt MMR: -1.5 k n / a with k ~ 3, n ~ 1.6 rad/yr, a ~ 2.5 au


# ----------------------------------------------------------------------------------------------------------------
# helpers (independent of the package)
# ----------------------------------------------------------------------------------------------------------------


NOT_MEASURED_BY_TYPE = (
    (lambda k: k.startswith('Optional') or 'None' in k, lambda: None),
    (lambda k: k == 'float', lambda: NAN),
    (lambda k: k == 'int', lambda: 0),
    (lambda k: k == 'bool', lambda: False),
    (lambda k: k == 'str', lambda: ''),
    (lambda k: 'ndarray' in k, lambda: np.array([])),
    (lambda k: k.lower().startswith('tuple'), tuple),
    (lambda k: k.lower().startswith('list'), list),
    (lambda k: k.lower().startswith('dict'), dict),
    (lambda k: k == 'CheckValue', lambda: CheckValue.NOT_MEASURED),
)


def make(cls, **values):
    """A result dataclass with every required field 'not measured' (nan, None, empty) and the given overrides."""
    filled = {}
    for f in dataclasses.fields(cls):
        if f.name in values or f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue
        kind = str(f.type)
        factory = next((build for match, build in NOT_MEASURED_BY_TYPE if match(kind)), None)
        if factory is None:
            raise TypeError(f'{cls.__name__}.{f.name}: {kind}')
        filled[f.name] = factory()
    filled.update(values)
    return cls(**filled)


def action_over_separatrix(h):
    """J/J_s = E(h) - (1 - h) K(h) (scipy's m = k^2 = h)."""
    return float(ellipe(h) - (1.0 - h) * ellipk(h))


def h_of_action(j):
    return brentq(lambda h: action_over_separatrix(h) - j, 1e-12, 1.0 - 1e-15)


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def pendulum_series(omega, amplitude_rad, t, centre=0.0):
    """sigma_ddot = -omega^2 sin(sigma - centre), released at rest at centre + amplitude."""
    sol = solve_ivp(
        lambda _, y: [y[1], -(omega**2) * np.sin(y[0] - centre)],
        (t[0], t[-1]),
        [centre + amplitude_rad, 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        method='DOP853',
    )
    return sol.y[0], sol.y[1]


def resolved_sampling(n_rows, step_yr, span_yr):
    return make(
        models.Sampling,
        step_yr=step_yr,
        span_yr=span_yr,
        n_rows=n_rows,
        verdict=models.StepVerdict.RESOLVED,
        branches=(0,),
        unwrap_ok=True,
        undefined_phase=False,
        jitter_low_e=False,
        far_from_resonance=False,
        alias_numpy_differs=False,
        unwrap_resid_p99_rad=0.0,
        unwrap_resid_max_rad=0.0,
        step_rotation_min_rad=0.0,
        step_rotation_max_rad=0.03,
        branch_margin_turns=0.5,
        revs_predictive=0.0,
        revs_numpy=0.0,
    )


# ----------------------------------------------------------------------------------------------------------------
# filter guard (DELTA item 1): P_cyc < FILTER_MIN_LIBRATION_CUTOFFS x P_c, gain G(P) = 1 / (1 + (P_c/P)^(2N))
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'period, cutoff, too_slow, gain',
    [
        # 1500 < 2 x 1000; G = 1 / (1 + (1000/1500)^4) = 1 / 1.197531 = 0.835052
        (1500.0, 1000.0, True, 0.835052),
        # exactly 2 P_c is not below the guard; G(2 P_c) = 1 / (1 + 1/16) = 0.941176 (the 0.94 of the settings table)
        (2000.0, 1000.0, False, 0.941176),
        # 5 P_c: G = 1 / (1 + 0.2^4) = 0.998403
        (5000.0, 1000.0, False, 0.998403),
        # P_cyc = P_c: G = 0.5
        (300.0, 300.0, True, 0.5),
    ],
)
def test_filter_guard_by_hand(period, cutoff, too_slow, gain):
    flag, g = preprocess.filter_guard(period, cutoff, SETTINGS)
    assert bool(flag) is too_slow
    assert g == pytest.approx(gain, abs=1e-5)


def test_filter_guard_without_a_period():
    flag, g = preprocess.filter_guard(NAN, 1000.0, SETTINGS)
    assert flag is False or flag == False  # noqa: E712 (numpy bool allowed)
    assert math.isnan(g)


def test_filter_guard_follows_the_setting():
    # the guard is G(P_cyc) < MMR_L8_GAIN_FULL: with 0.99, G(3000; 1000) = 1 / (1 + 3^-4) = 0.9878 is too slow,
    # G(4000; 1000) = 1 / (1 + 4^-4) = 0.9961 is not; at the default 0.94 neither is
    s = settings_from_config(mmr_l8_gain_full=0.99)
    assert preprocess.filter_guard(3000.0, 1000.0, s)[0]
    assert not preprocess.filter_guard(4000.0, 1000.0, s)[0]
    assert not preprocess.filter_guard(3000.0, 1000.0, SETTINGS)[0]


@pytest.mark.parametrize('period_over_cutoff', [1.5, 2.0, 4.0])
def test_filter_guard_gain_is_what_the_filter_does_to_a_sinusoid(period_over_cutoff):
    """Independent measurement: filter a unit sinusoid with scipy's sos Butterworth forward-backward
    (the package's lowpass) and compare the surviving amplitude with the guard's gain."""
    cutoff, step = 200.0, 1.0
    period = period_over_cutoff * cutoff
    t = np.arange(0.0, 200 * period, step)
    x = np.sin(2 * np.pi * t / period)
    y = filtering.lowpass(t, x, cutoff, SETTINGS.filter_order, detrend=False)
    core = slice(len(t) // 4, 3 * len(t) // 4)
    measured = np.sqrt(2.0) * np.std(y[core])
    _, gain = preprocess.filter_guard(period, cutoff, SETTINGS)
    assert gain == pytest.approx(measured, rel=0.01)


# ----------------------------------------------------------------------------------------------------------------
# zone.modulation_from_members: a_dot = sum_j [A_j sin sigma_j + B_j cos sigma_j], sigma_j = sigma + j beta
# ----------------------------------------------------------------------------------------------------------------


AMPLITUDE_60 = math.radians(60)


def _member_record(c, k_main, k_member, nu, years=40000.0, step=5.0, amplitude=AMPLITUDE_60):
    """A pendulum driven by the multiplet: sigma_ddot = c a_dot, a_dot = K0 sin sigma + K1 sin(sigma + nu t).

    Integrated here; returns t, sigma, a_dot (exact, from the trajectory), beta = nu t."""
    t = np.arange(0.0, years, step)

    def adot(tt, s):
        return k_main * np.sin(s) + k_member * np.sin(s + nu * tt)

    centre = 0.0 if c < 0 else np.pi  # c K sin(sigma): stable at 0 for c < 0, at pi for c > 0
    sol = solve_ivp(
        lambda tt, y: [y[1], c * adot(tt, y[0])],
        (0.0, t[-1]),
        [centre + amplitude, 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-13,
        method='DOP853',
    )
    sigma = sol.y[0]
    return t, sigma, adot(t, sigma), nu * t


@pytest.mark.parametrize('c', [C_ID, 2.0])
def test_members_subresonance_librates_main_angle_turns(c):
    """sigma1 = sigma0 + beta is an integrated pendulum (the subresonance), beta = varpi - varpi_P turns at a known
    rate, so sigma0 circulates; a_dot = K sin sigma1 only.

    Truth: C_0 = 0, C_1 = K, W(t) = K e^{i beta}: Omega^2 = |c| K constant, the well of sigma0 moves as -beta
    (sigma1 at its own stable point 0 for c < 0, pi for c > 0), the single harmonic of sigma0 explains nothing
    (gain ~ 1), and c a_dot = -Omega^2 sin(sigma0 - sigma_s) at every sample (the model of SPEC §4.6.1 item 1).
    """
    omega = 2 * np.pi / 1000.0
    k = omega**2 / abs(c)
    nu = 2 * np.pi / 7000.0
    t = np.arange(0.0, 40000.0, 5.0)
    sol = solve_ivp(
        lambda tt, y: [y[1], c * k * np.sin(y[0])],
        (0.0, t[-1]),
        [(0.0 if c < 0 else np.pi) + np.radians(60), 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-13,
        method='DOP853',
    )
    sigma1 = sol.y[0]
    beta = nu * t
    sigma0 = sigma1 - beta
    adot = k * np.sin(sigma1)

    om2, well, eps, gain = zone.modulation_from_members(t, sigma0, adot, beta, c, harmonics=(0, 1))
    assert np.allclose(om2, abs(c) * k, rtol=1e-3)
    assert gain == pytest.approx(1.0, abs=0.02)
    assert len(eps) == 1 and (not np.isfinite(eps[0]) or eps[0] > 50)
    # the model identity, independent of any sign convention for sigma_s
    assert np.allclose(c * adot, -om2 * np.sin(sigma0 - well), atol=1e-3 * abs(c) * k)
    # the stable point of sigma0 follows -beta (+ pi for c > 0)
    expected_well = -beta + (0.0 if c < 0 else np.pi)
    assert np.max(np.abs(wrap(well - expected_well))) < 1e-3


@pytest.mark.parametrize('c', [C_ID, 2.0])
def test_members_modulated_main_term(c):
    """K0 = 1, K1 = 0.3 (relative): |W| = K0 sqrt(1 + 0.09 + 0.6 cos beta), so Omega^2 / (|c| K0) runs from 0.7
    to 1.3, eps = (0.3,), and the single harmonic leaves the beat unexplained (gain > 0)."""
    omega = 2 * np.pi / 1000.0
    k0 = omega**2 / abs(c)
    k1 = 0.3 * k0
    nu = 2 * np.pi / 6000.0
    t, sigma, adot, beta = _member_record(c, k0, k1, nu)

    om2, well, eps, gain = zone.modulation_from_members(t, sigma, adot, beta, c, harmonics=(0, 1))
    truth = abs(c) * k0 * np.sqrt(1.0 + 0.3**2 + 2 * 0.3 * np.cos(beta))
    assert np.allclose(om2, truth, rtol=1e-3)
    assert om2.max() / om2.min() == pytest.approx(1.3 / 0.7, rel=2e-3)
    assert eps == pytest.approx((0.3,), rel=1e-3)
    assert gain > 0.01
    assert np.allclose(c * adot, -om2 * np.sin(sigma - well), atol=1e-3 * abs(c) * k0)


def test_members_with_zero_identity_coefficient_is_not_measured():
    t = np.arange(0.0, 1000.0, 1.0)
    sigma = 0.3 * np.sin(t / 50)
    om2, well, eps, gain = zone.modulation_from_members(t, sigma, np.sin(sigma), 0.001 * t, 0.0)
    assert np.all(np.isnan(om2))
    assert math.isnan(gain)


# ----------------------------------------------------------------------------------------------------------------
# zone.modulation_from_eccentricity: Omega^2 = Omega_bar^2 (e_osc / e_bar)^q, bounded oscillatory e only
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('give_nu', [True, False])
def test_eccentricity_modulation_excludes_the_trend(give_nu):
    """e = 0.10 + 0.02 sin(2 pi t / 1000) + 1e-3 per kyr x t over 40 kyr (40 whole cycles).

    Trend removed with the mean kept: e_osc = 0.12 + 0.02 sin, e_bar = 0.12, de = 0.02 (half-range, PHYSICS B1a),
    dK/K = q de / e_bar = 2 x 0.02 / 0.12 = 1/3 for q = 2; trend 1e-3 per kyr;
    Omega^2(t) = Omega_bar^2 (1 + sin / 6)^2."""
    period, years = 1000.0, 40000.0
    t = np.arange(0.0, years, 2.0)
    wave = np.sin(2 * np.pi * t / period)
    e = 0.10 + 0.02 * wave + 1e-6 * t
    omega_bar = 2 * np.pi / 500.0
    nu = 2 * np.pi / period if give_nu else NAN
    om2, dk_over_k, trend = zone.modulation_from_eccentricity(t, e, 2, omega_bar, nu_rad_yr=nu)
    assert dk_over_k == pytest.approx(1.0 / 3.0, rel=0.03)
    assert trend == pytest.approx(1e-3, rel=0.03)
    truth = omega_bar**2 * (1.0 + wave / 6.0) ** 2
    assert np.allclose(om2, truth, rtol=0.02)


def test_eccentricity_modulation_order_zero_is_constant():
    t = np.arange(0.0, 20000.0, 5.0)
    e = 0.1 + 0.03 * np.sin(2 * np.pi * t / 2000.0)
    omega_bar = 2 * np.pi / 800.0
    om2, dk_over_k, _ = zone.modulation_from_eccentricity(t, e, 0, omega_bar)
    finite = om2[np.isfinite(om2)]
    assert finite.size == 0 or np.allclose(finite, omega_bar**2)
    assert dk_over_k == 0 or math.isnan(dk_over_k)


# ----------------------------------------------------------------------------------------------------------------
# zone.numeric_zone / build_zone: phi'' = -Omega0^2 (1 + m cos nu t) sin phi (SPEC §4.6.1)
# ----------------------------------------------------------------------------------------------------------------

OMEGA0 = 2 * np.pi / 100.0  # small period 100 yr


def _model(m, lam):
    nu = lam * OMEGA0

    def om2(tt):
        return OMEGA0**2 * (1.0 + m * np.cos(nu * np.asarray(tt, dtype=float)))

    def well(tt):
        return np.zeros_like(np.asarray(tt, dtype=float))

    return nu, om2, well


def _mean_sqrt_strength(m):
    """<sqrt(1 + m cos x)> over a period: <J_s> ~ <Omega> (J_s = 8 Omega / pi)."""
    x = np.linspace(0.0, 2 * np.pi, 20001)[:-1]
    return float(np.mean(np.sqrt(1.0 + m * np.cos(x))))


@pytest.fixture(scope='module')
def adiabatic_zone_m03():
    m, lam = 0.3, 0.05
    nu, om2, well = _model(m, lam)
    period = 2 * np.pi / nu
    t = np.arange(0.0, 5 * period, 2.0)
    return m, nu, t, om2, well, zone.numeric_zone(t, om2, well, SETTINGS, modulation_period_yr=period)


def test_numeric_zone_adiabatic_edge(adiabatic_zone_m03):
    """lambda = 0.05: J_lo = Omega_min / Omega_bar = sqrt(1 - m) / <sqrt(1 + m cos)> = 0.83666 / 0.99435 = 0.8414
    (SPEC: 0.84 at m = 0.3; Neishtadt via Shevchenko 2020 §5.4). Within the mismatch tolerance 0.05."""
    m, _, _, _, _, (j_lo, h_hi) = adiabatic_zone_m03
    edge = math.sqrt(1 - m) / _mean_sqrt_strength(m)
    assert edge == pytest.approx(0.8414, abs=5e-4)
    assert abs(j_lo - edge) <= SETTINGS.mmr_zone_adiabatic_tol
    # J_lo is a grid value: 0.1 + k x 0.025
    k = (j_lo - SETTINGS.mmr_zone_j_min) / SETTINGS.mmr_zone_j_step
    assert k == pytest.approx(round(k), abs=1e-6)


def test_numeric_zone_rotation_side_sudden_bound(adiabatic_zone_m03):
    """A rotation started at the strongest moment with h0 < Omega_max^2 / Omega0^2 = 1 + m (energy over 2 Omega0^2)
    is inside the separatrix at once, so h_hi >= 1 + m - one grid step; and nothing above h_max is trapped."""
    m, _, _, _, _, (j_lo, h_hi) = adiabatic_zone_m03
    assert h_hi >= 1 + m - SETTINGS.mmr_zone_j_step - 1e-9
    assert h_hi <= SETTINGS.mmr_zone_h_max + 1e-9


def test_numeric_zone_without_modulation_reaches_nothing():
    """Constant strength: a libration below J_s never goes over the top, a rotation above never turns back;
    the docstring's convention is J_lo = 1, h_hi = 1."""
    _, om2, well = _model(0.0, 0.05)
    t = np.arange(0.0, 6000.0, 2.0)
    j_lo, h_hi = zone.numeric_zone(t, lambda tt: OMEGA0**2 + 0 * np.asarray(tt, dtype=float), well, SETTINGS)
    assert j_lo == pytest.approx(1.0, abs=SETTINGS.mmr_zone_j_step)
    assert h_hi == pytest.approx(1.0, abs=SETTINGS.mmr_zone_j_step)


def test_numeric_zone_non_adiabatic_reaches_far_below_the_adiabatic_edge():
    """lambda = 0.5, m = 0.55 over 40 modulation periods (the 52356 / 24820 regime): secondary resonances
    k omega(J) = nu carry orbits from J0 ~ 0.3 to the separatrix (SPEC §4.6.1: 0.28-0.30), far below the adiabatic
    edge sqrt(1 - 0.55) = 0.67 (sudden bound sqrt(0.45)/<sqrt(1 + 0.55 cos)> = 0.6708/0.9795 = 0.6849)."""
    m, lam = 0.55, 0.5
    nu, om2, well = _model(m, lam)
    period = 2 * np.pi / nu
    t = np.arange(0.0, 40 * period, 1.0)
    j_lo, h_hi = zone.numeric_zone(t, om2, well, SETTINGS, modulation_period_yr=period)
    sudden = math.sqrt(1 - m) / _mean_sqrt_strength(m)
    assert sudden == pytest.approx(0.6849, abs=2e-3)
    assert j_lo < 0.45
    assert j_lo >= SETTINGS.mmr_zone_j_min
    assert h_hi >= 1 + m - SETTINGS.mmr_zone_j_step - 1e-9


def test_build_zone_numeric_with_adiabatic_cross_check(adiabatic_zone_m03):
    m, nu, t, om2, well, (j_lo, h_hi) = adiabatic_zone_m03
    z = zone.build_zone(t, om2(t), well(t), 0.05, SETTINGS, modulation_period_yr=2 * np.pi / nu)
    assert isinstance(z, Zone)
    assert z.method == ZoneMethod.NUMERIC
    assert z.j_lo == pytest.approx(j_lo, abs=1e-9)
    assert z.h_hi == pytest.approx(h_hi, abs=1e-9)
    # adiabatic edges: Omega_min / Omega_bar and Omega_max / Omega_bar, Omega_bar = <Omega>
    mean = _mean_sqrt_strength(m)
    assert z.adiabatic_lo == pytest.approx(math.sqrt(1 - m) / mean, abs=0.01)
    assert z.adiabatic_hi == pytest.approx(math.sqrt(1 + m) / mean, abs=0.01)
    assert not z.mismatch  # lambda < 0.1 and the numbers agree within 0.05
    # h_lo = h(J_lo) by the pendulum law J/J_s = E(h) - (1 - h) K(h)
    assert z.h_lo == pytest.approx(h_of_action(min(z.j_lo, 1 - 1e-9)), abs=1e-4)
    assert z.width_h == pytest.approx(z.h_hi - z.h_lo)


def test_build_zone_adiabatic_method():
    """'adiabatic': J_lo = Omega_min/Omega_bar, h_hi = (Omega_max/Omega_bar)^2 (build_zone docstring)."""
    m = 0.3
    nu, om2, well = _model(m, 0.05)
    t = np.arange(0.0, 3 * 2 * np.pi / nu, 2.0)
    z = zone.build_zone(t, om2(t), well(t), 0.05, SETTINGS, modulation_period_yr=2 * np.pi / nu, method=ZoneMethod.ADIABATIC)
    mean = _mean_sqrt_strength(m)
    assert z.method == ZoneMethod.ADIABATIC
    assert z.j_lo == pytest.approx(math.sqrt(1 - m) / mean, abs=0.01)
    assert z.h_hi == pytest.approx((math.sqrt(1 + m) / mean) ** 2, abs=0.02)


# ----------------------------------------------------------------------------------------------------------------
# a modulated pendulum that stays inside vs one that crosses: cycle classes, crossings, the zone rule
# ----------------------------------------------------------------------------------------------------------------

LIB_OMEGA0 = 2 * np.pi / 1000.0  # small period 1000 yr (above MMR_LIBRATION_PERIOD_MIN_YR)
LIB_M, LIB_LAMBDA = 0.3, 0.05


def _modulated_cycles(j0):
    """Integrate phi'' = -Omega0^2 (1 + m cos nu t) sin phi from the bottom of the well at t = 0 (strongest moment)
    with J0/<J_s> = j0, cut it into windows of one small period, and measure each window's energy by hand:
    h = E / (2 Omega(t)^2) (instantaneous), h_mean = E / (2 Omega_bar^2), J/<J_s> = J/J_s(h) Omega(t)/Omega_bar."""
    nu = LIB_LAMBDA * LIB_OMEGA0
    t = np.arange(0.0, 3 * 2 * np.pi / nu, 10.0)
    mean = _mean_sqrt_strength(LIB_M)
    omega_bar = LIB_OMEGA0 * mean
    m0 = 1.0 + LIB_M
    h0 = h_of_action(j0 * mean / math.sqrt(m0))  # J/J_s(t0) = J0 <J_s> / J_s(t0)
    rate0 = 2 * LIB_OMEGA0 * math.sqrt(m0 * h0)  # at the bottom E = rate^2 / 2 = 2 Omega(t0)^2 h0
    sol = solve_ivp(
        lambda tt, y: [y[1], -(LIB_OMEGA0**2) * (1 + LIB_M * np.cos(nu * tt)) * np.sin(y[0])],
        (0.0, t[-1]),
        [0.0, rate0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        method='DOP853',
    )
    phi, rate = sol.y
    strength = LIB_OMEGA0**2 * (1 + LIB_M * np.cos(nu * t))
    energy = 0.5 * rate**2 + strength * (1 - np.cos(phi))
    h_inst = energy / (2 * strength)
    cycles = []
    width = 2 * np.pi / LIB_OMEGA0
    for start in np.arange(0.0, t[-1] - width, width):
        sel = (t >= start) & (t < start + width)
        h = float(np.mean(h_inst[sel]))
        rel = float(np.mean(strength[sel])) / omega_bar**2
        rotation = h > 1
        j = NAN if rotation else action_over_separatrix(h) * math.sqrt(rel)
        cycles.append(
            CycleEnergy(
                t_start_yr=float(start),
                t_end_yr=float(start + width),
                rotation=bool(rotation),
                h=h,
                action_ratio=j,
                period_yr=width,
                period_law_ratio=1.0,
                h_mean=float(np.mean(energy[sel])) / (2 * omega_bar**2),
                strength_rel_sq=rel,
            )
        )
    return t, phi, rate, cycles


@pytest.fixture(scope='module')
def modulated_zone():
    nu = LIB_LAMBDA * LIB_OMEGA0
    period = 2 * np.pi / nu
    t = np.arange(0.0, 3 * period, 10.0)
    strength = LIB_OMEGA0**2 * (1 + LIB_M * np.cos(nu * t))
    return zone.build_zone(t, strength, np.zeros_like(t), LIB_LAMBDA, SETTINGS, modulation_period_yr=period)


def _energy(cycles, z, sigma_h=0.02, with_slow=CheckValue.PASS):
    classes = pendulum.classify_cycles(cycles, z, sigma_h, SETTINGS)
    times = pendulum.crossings(cycles, [])
    chaos = bool(times) and all(CycleClass(c) == CycleClass.ZONE for c in classes)
    return make(
        EnergyAnalysis,
        potential_e=None,
        potential_f=None,
        ve_vf_agree_fraction=1.0,
        well_coherence=0.98,
        cycles=list(cycles),
        sigma_h=sigma_h,
        zone=z,
        cycle_classes=classes,
        crossing_times_yr=times,
        sigma_only=CheckValue.PASS,
        with_slow=with_slow,
        episodes_in_separatrix_zone=chaos,
    )


def test_modulated_zone_edge_is_adiabatic(modulated_zone):
    assert modulated_zone.j_lo == pytest.approx(math.sqrt(1 - LIB_M) / _mean_sqrt_strength(LIB_M), abs=0.05)


def test_pendulum_deep_inside_never_crosses(modulated_zone):
    """J0 = 0.5 < J_lo ~ 0.84: the action is an adiabatic invariant, J/<J_s> stays 0.5, every cycle is deep."""
    _, _, _, cycles = _modulated_cycles(0.5)
    assert all(not c.rotation for c in cycles)
    assert np.allclose([c.action_ratio for c in cycles], 0.5, atol=0.03)  # the invariant, measured by hand
    energy = _energy(cycles, modulated_zone)
    assert all(CycleClass(c) == CycleClass.DEEP for c in energy.cycle_classes)
    assert energy.crossing_times_yr == []


def test_pendulum_inside_the_zone_without_crossing(modulated_zone):
    """J0 = 0.82: below the adiabatic edge 0.841 (never crosses) but within 2 sigma_J of J_lo, so 'zone' (R2)."""
    _, _, _, cycles = _modulated_cycles(0.82)
    assert all(not c.rotation for c in cycles)
    energy = _energy(cycles, modulated_zone)
    assert energy.crossing_times_yr == []
    assert any(CycleClass(c) == CycleClass.ZONE for c in energy.cycle_classes)


def test_pendulum_crossing_its_pulsating_separatrix(modulated_zone):
    """J0 = 0.97 > J_lo: when J_s(t) / <J_s> = sqrt(1 + 0.3 cos nu t) / 0.994 falls below 0.97 the orbit rotates,
    and it is recaptured when the separatrix grows again: crossings in each modulation period, every cycle in the
    zone (the layer's own dynamics: episodes_in_separatrix_zone)."""
    _, _, _, cycles = _modulated_cycles(0.97)
    assert any(c.rotation for c in cycles) and any(not c.rotation for c in cycles)
    energy = _energy(cycles, modulated_zone)
    assert len(energy.crossing_times_yr) >= 3  # at least one exit and one return per modulation period
    assert all(CycleClass(c) == CycleClass.ZONE for c in energy.cycle_classes)
    assert energy.episodes_in_separatrix_zone


@pytest.fixture(scope='module')
def libration_geometry():
    """A clean pendulum of 1000-yr small period and 90-deg amplitude (P = 1000 F(90) = 1180 yr), 20 kyr."""
    t = np.arange(0.0, 20000.0, 5.0)
    sigma, rate = pendulum_series(LIB_OMEGA0, np.radians(90), t)
    a = 2.5 + rate / C_ID
    angle = geo.FilteredAngle(t, sigma, sigma, rate, sigma, sigma, a, C_ID, 100.0, 5.0, (100.0, t[-1] - 100.0))
    g = geo.measure_geometry(angle, np.radians(1.0), SETTINGS)
    return t, g


def test_geometry_of_the_reference_pendulum(libration_geometry):
    t, g = libration_geometry
    # P = T0 F(A) = 1000 x (2/pi) K(sin^2 45 deg) = 1000 x 1.18034
    assert g.period_cycle_yr == pytest.approx(1000.0 * 2 / np.pi * ellipk(0.5), rel=0.01)
    angle = decide.decide_angle(resolved_sampling(len(t), 5.0, t[-1]), False, g, None, SETTINGS, Ledger.from_settings(SETTINGS))
    assert angle.angle_status == AngleStatus.LIBRATION


def test_zone_rule_deep_libration_is_resonant(libration_geometry, modulated_zone):
    _, g = libration_geometry
    energy = _energy(_modulated_cycles(0.5)[3], modulated_zone)
    angle = decide.AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', episode_id=0)
    status, reason, _, flags = decide.zone_rule(angle, g, energy, SETTINGS, Ledger.from_settings(SETTINGS))
    assert status == Status.RESONANT
    assert reason == Reason.DYNAMIC_LIBRATION
    assert 'libration_in_separatrix_zone' not in flags
    assert 'episodes_in_separatrix_zone' not in flags


def test_zone_rule_libration_in_the_zone_is_resonant_with_flag(libration_geometry, modulated_zone):
    """SPEC §5.4 R2: no crossings, some cycles in the zone -> resonant + libration_in_separatrix_zone."""
    _, g = libration_geometry
    energy = _energy(_modulated_cycles(0.82)[3], modulated_zone)
    angle = decide.AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', episode_id=0)
    status, _, _, flags = decide.zone_rule(angle, g, energy, SETTINGS, Ledger.from_settings(SETTINGS))
    assert status == Status.RESONANT
    assert 'libration_in_separatrix_zone' in flags


def test_zone_rule_transient_inside_the_chaotic_layer_stays_transient(libration_geometry, modulated_zone):
    """DELTA 13(b) / DECISIONS 2026-10-15 evening: the observed episodes decide (52356 -> transient); the zone, the
    energies and episodes_in_separatrix_zone are columns and flags, they do not turn a transient into near_separatrix."""
    _, g = libration_geometry
    energy = _energy(_modulated_cycles(0.97)[3], modulated_zone)
    assert energy.episodes_in_separatrix_zone
    angle = decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', episode_id=0)
    status, reason, _, flags = decide.zone_rule(angle, g, energy, SETTINGS, Ledger.from_settings(SETTINGS))
    assert status == Status.TRANSIENT
    assert reason == Reason.DYNAMIC_EPISODE
    assert 'episodes_in_separatrix_zone' in flags


def test_zone_rule_near_separatrix_stays_near_separatrix(libration_geometry, modulated_zone):
    _, g = libration_geometry
    energy = _energy(_modulated_cycles(0.97)[3], modulated_zone)
    angle = decide.AngleDecision(AngleStatus.NEAR_SEPARATRIX, AngleSubtype.PATTERN_A, 'G3')
    status, reason, _, _ = decide.zone_rule(angle, g, energy, SETTINGS, Ledger.from_settings(SETTINGS))
    assert status == Status.NEAR_SEPARATRIX
    assert reason == Reason.SEPARATRIX_ZONE


def test_zone_rule_without_energy_keeps_the_observed_status(libration_geometry):
    """Zone not measurable (SPEC §5.4 last paragraph): transient by geometry."""
    _, g = libration_geometry
    for angle_status, expected in [(AngleStatus.EPISODIC, Status.TRANSIENT), (AngleStatus.LIBRATION, Status.RESONANT)]:
        angle = decide.AngleDecision(angle_status, AngleSubtype.NONE, 'G2', episode_id=0)
        status, _, _, _ = decide.zone_rule(angle, g, None, SETTINGS, Ledger.from_settings(SETTINGS))
        assert status == expected


# ----------------------------------------------------------------------------------------------------------------
# decide_status: the tree of SPEC §5.3 (with DELTA 13b)
# ----------------------------------------------------------------------------------------------------------------

ANGLES = {
    AngleStatus.LIBRATION: decide.AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', episode_id=0),
    AngleStatus.EPISODIC: decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', episode_id=0),
    AngleStatus.NEAR_SEPARATRIX: decide.AngleDecision(AngleStatus.NEAR_SEPARATRIX, AngleSubtype.PATTERN_A, 'G3'),
    AngleStatus.DRIFT: decide.AngleDecision(AngleStatus.DRIFT, AngleSubtype.NONE, 'G5'),
    AngleStatus.CIRCULATION: decide.AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.NONE, 'G4'),
    AngleStatus.INSUFFICIENT_TIME: decide.AngleDecision(AngleStatus.INSUFFICIENT_TIME, AngleSubtype.FEW_CYCLES, 'G6'),
    AngleStatus.UNDEFINED: decide.AngleDecision(AngleStatus.UNDEFINED, AngleSubtype.UNPHYSICAL, 'G0'),
}


def _spec(key):
    return models.ResonanceSpec(
        key=key,
        kind=models.ResonanceKind.TWO_BODY,
        planets=('Mars',),
        planet_coefficients=(9,),
        body_coefficient=-16,
        body_varpi_coefficient=5,
        planet_varpi_coefficients=(2,),
    )


def _subresonances(check):
    accepted = None
    if check == CheckValue.FAIL:
        accepted = make(models.SubresonanceCandidate, spec=_spec('9M-16+2+5'), j=2, nodal=False, rank=1, accepted=True, reason='')
    return make(
        models.SubresonanceResult,
        triggers=(models.SubresonanceTrigger.T1,),
        candidates=[accepted] if accepted else [],
        accepted=accepted,
        check=check,
        theory_measured=True,
    )


def _owner(check):
    accepted = None
    if check == CheckValue.FAIL:
        accepted = make(
            models.OwnerCandidate,
            spec=_spec('4J-2S-1'),
            r2_windows=(0.9, 0.9),
            rank=1,
            accepted=True,
            reason='',
            r2_median=0.9,
        )
    return make(models.OwnerResult, n_candidates=3, candidates=[accepted] if accepted else [], accepted=accepted, check=check)


def _response(l4=CheckValue.PASS, b1=CheckValue.PASS, b5=CheckValue.PASS, exact=False):
    return make(
        models.Response,
        identity=CheckValue.PASS,
        identity_slope_ratio=1.0,
        identity_corr=0.99,
        a_response=l4,
        a_response_thin=False,
        a_amplitude=b1,
        b1_ratio=1.0 if b1 != CheckValue.FAIL else 0.02,
        force_r2_h1=0.95 if l4 == CheckValue.PASS else 0.1,
        force_r2_h2=0.95 if l4 == CheckValue.PASS else 0.1,
        phi0_coherence=0.99,
        slow_libration=b5,
        b5_ratio=3.0 if b5 == CheckValue.PASS else 0.2,
        b5_route='data',
        exact_commensurability=exact,
        centre_rate_rad_yr=1e-6 if exact else 1e-3,
        centre_rate_err_rad_yr=1e-5,
    )


def _kinematics(check=CheckValue.PASS, detail=''):
    return make(models.Kinematics, check=check, detail=detail, corr_rate=0.99 if check == CheckValue.FAIL else 0.1)


def _strength(check=CheckValue.PASS, method='eq_scale'):
    return make(models.Strength, check=check, eq_legal=True, implausible=False, ratio=1.0, method=method)


@pytest.fixture(scope='module')
def clean_energy(modulated_zone):
    return _energy(_modulated_cycles(0.5)[3], modulated_zone)


def _decide(
    angle_status,
    geometry,
    energy,
    l4=CheckValue.PASS,
    r0=CheckValue.PASS,
    b1=CheckValue.PASS,
    b5=CheckValue.PASS,
    exact=False,
    kin=CheckValue.PASS,
    sub=CheckValue.PASS,
    own=CheckValue.PASS,
    strength=CheckValue.PASS,
    sampling=None,
):
    if energy is not None:
        energy = dataclasses.replace(energy, with_slow=r0)
    sampling = sampling or resolved_sampling(4000, 5.0, 20000.0)
    return decide.decide_status(
        ANGLES[angle_status],
        sampling,
        geometry,
        energy,
        _response(l4, b1, b5, exact),
        _kinematics(kin, 'varpi_term' if kin == CheckValue.FAIL else ''),
        _subresonances(sub) if sub is not None else None,
        _owner(own) if own is not None else None,
        _strength(strength),
        SETTINGS,
        Ledger.from_settings(SETTINGS),
    )


def test_status_dynamic_libration(libration_geometry, clean_energy):
    d = _decide(AngleStatus.LIBRATION, libration_geometry[1], clean_energy)
    assert d.status == Status.RESONANT
    assert d.reason == Reason.DYNAMIC_LIBRATION
    assert d.origin == LibrationOrigin.DYNAMIC


def test_status_dynamic_transient(libration_geometry, clean_energy):
    d = _decide(AngleStatus.EPISODIC, libration_geometry[1], clean_energy)
    assert d.status == Status.TRANSIENT
    assert d.reason == Reason.DYNAMIC_EPISODE
    assert d.origin == LibrationOrigin.DYNAMIC


@pytest.mark.parametrize('angle_status', [AngleStatus.LIBRATION, AngleStatus.EPISODIC, AngleStatus.DRIFT])
def test_status_accepted_subresonance_comes_first(libration_geometry, clean_energy, angle_status):
    """Step 1 before step 2: with both a subresonance and an owner accepted, the subresonance carries it."""
    d = _decide(angle_status, libration_geometry[1], clean_energy, sub=CheckValue.FAIL, own=CheckValue.FAIL, b1=CheckValue.FAIL)
    assert d.status == Status.NON_RESONANT
    assert d.reason == Reason.SUBRESONANCE_LIBRATES
    assert d.origin == LibrationOrigin.SUBRESONANCE
    assert d.owner == '9M-16+2+5'


@pytest.mark.parametrize('angle_status', [AngleStatus.LIBRATION, AngleStatus.EPISODIC, AngleStatus.NEAR_SEPARATRIX])
def test_status_accepted_owner_is_induced(libration_geometry, clean_energy, angle_status):
    d = _decide(angle_status, libration_geometry[1], clean_energy, own=CheckValue.FAIL, b1=CheckValue.FAIL)
    assert d.status == Status.NON_RESONANT
    assert d.reason == Reason.INDUCED_LIBRATION
    assert d.origin == LibrationOrigin.INDUCED
    assert d.owner == '4J-2S-1'


@pytest.mark.parametrize('angle_status', [AngleStatus.LIBRATION, AngleStatus.EPISODIC, AngleStatus.NEAR_SEPARATRIX])
def test_status_b1_fail_is_kinematic_libration(libration_geometry, clean_energy, angle_status):
    """Step 3: a carries < B1 of the needed amplitude -> non_resonant, kinematic_libration, origin kinematic."""
    d = _decide(angle_status, libration_geometry[1], clean_energy, b1=CheckValue.FAIL)
    assert d.status == Status.NON_RESONANT
    assert d.reason == Reason.KINEMATIC_LIBRATION
    assert d.origin == LibrationOrigin.KINEMATIC


def test_status_kinematic_drift(libration_geometry, clean_energy):
    d = _decide(AngleStatus.DRIFT, libration_geometry[1], clean_energy, kin=CheckValue.FAIL)
    assert d.status == Status.NON_RESONANT
    assert d.reason == Reason.KINEMATIC_DRIFT
    assert d.origin == LibrationOrigin.KINEMATIC
    assert d.origin_detail == 'varpi_term'


def test_status_kinematic_with_exact_commensurability_needs_a_longer_baseline(libration_geometry, clean_energy):
    """Step 3 exception: B5 fails and |nu_c| < 2 dnu_c (theory does not forbid a slow libration) -> uncertain,
    baseline_needed."""
    d = _decide(AngleStatus.DRIFT, libration_geometry[1], clean_energy, kin=CheckValue.FAIL, b5=CheckValue.FAIL, exact=True)
    assert d.status == Status.UNCERTAIN
    assert d.reason == Reason.BASELINE_NEEDED


def test_status_drift_without_mechanism(libration_geometry, clean_energy):
    d = _decide(AngleStatus.DRIFT, libration_geometry[1], clean_energy)
    assert d.status == Status.UNCERTAIN
    assert d.reason == Reason.DRIFT_MECHANISM_UNKNOWN
    assert d.origin == LibrationOrigin.UNKNOWN


@pytest.mark.parametrize('angle_status', [AngleStatus.LIBRATION, AngleStatus.EPISODIC])
def test_status_5a_neither_force_law_nor_potential(libration_geometry, clean_energy, angle_status):
    d = _decide(angle_status, libration_geometry[1], clean_energy, l4=CheckValue.FAIL, r0=CheckValue.FAIL)
    assert d.status == Status.UNCERTAIN
    assert d.reason == Reason.ORIGIN_UNKNOWN
    assert d.origin == LibrationOrigin.UNKNOWN


def test_status_5a_near_separatrix_is_a_circulation(libration_geometry, clean_energy):
    d = _decide(AngleStatus.NEAR_SEPARATRIX, libration_geometry[1], clean_energy, l4=CheckValue.FAIL, r0=CheckValue.FAIL)
    assert d.status == Status.NON_RESONANT
    assert d.reason in (Reason.MODULATED_CIRCULATION, Reason.NO_RESONANT_POTENTIAL)


def test_status_5b_force_law_without_potential(libration_geometry, clean_energy):
    d = _decide(AngleStatus.LIBRATION, libration_geometry[1], clean_energy, l4=CheckValue.PASS, r0=CheckValue.FAIL)
    assert d.status == Status.UNCERTAIN
    assert d.reason == Reason.ORIGIN_UNKNOWN
    assert 'force_law_disagrees' in d.flags


def test_status_5c_potential_without_force_law_is_dynamic_capped(libration_geometry, clean_energy):
    """5c (52356: phi0 wanders 41 deg, V_E ~ V_F in 29 of 37 windows): dynamics, ceiling medium."""
    d = _decide(AngleStatus.LIBRATION, libration_geometry[1], clean_energy, l4=CheckValue.FAIL, r0=CheckValue.PASS)
    assert d.status == Status.RESONANT
    assert d.origin == LibrationOrigin.DYNAMIC
    assert 'force_law_disagrees' in d.flags + d.caps


def test_status_insufficient_time_undefined_circulation(libration_geometry, clean_energy):
    g = libration_geometry[1]
    d = _decide(AngleStatus.INSUFFICIENT_TIME, g, clean_energy)
    assert (d.status, d.reason) == (Status.UNCERTAIN, Reason.BASELINE_NEEDED)
    d = _decide(AngleStatus.UNDEFINED, g, clean_energy)
    assert d.status == Status.INVALID
    d = _decide(AngleStatus.CIRCULATION, g, clean_energy)
    assert d.status == Status.NON_RESONANT
    assert d.origin in (LibrationOrigin.NONE, '')


def _circulation_energy(h_turns, z):
    cycles = [
        CycleEnergy(
            t_start_yr=1000.0 * i,
            t_end_yr=1000.0 * (i + 1),
            rotation=True,
            h=h_turns,
            action_ratio=NAN,
            period_yr=1000.0,
            period_law_ratio=1.0,
            h_mean=h_turns,
        )
        for i in range(12)
    ]
    return make(
        EnergyAnalysis,
        potential_e=None,
        potential_f=None,
        ve_vf_agree_fraction=1.0,
        well_coherence=0.98,
        cycles=cycles,
        h_med=h_turns,
        sigma_h=0.02,
        zone=z,
        cycle_classes=[CycleClass.ABOVE] * len(cycles),
        sigma_only=CheckValue.PASS,
        with_slow=CheckValue.PASS,
    )


@pytest.mark.parametrize('h_turns, flagged', [(1.2, True), (1.6, False)])
def test_circulation_near_separatrix_energy_flag(libration_geometry, h_turns, flagged):
    """SPEC §5.4 R6: zone h_lo = 0.9, h_hi = 1.1 (width 0.2). Turns at h = 1.2 lie 0.5 widths above the zone
    (<= 1.5: flag), at h = 1.6 2.5 widths (no flag). Either way non_resonant: the flag is a column."""
    z = Zone(ZoneMethod.NUMERIC, 0.8, 0.9, 1.1, 0.85, 1.1, False)
    energy = _circulation_energy(h_turns, z)
    d = decide.decide_status(
        ANGLES[AngleStatus.CIRCULATION],
        resolved_sampling(4000, 5.0, 20000.0),
        libration_geometry[1],
        energy,
        _response(),
        _kinematics(),
        None,
        None,
        _strength(),
        SETTINGS,
        Ledger.from_settings(SETTINGS),
    )
    assert d.status == Status.NON_RESONANT
    assert ('near_separatrix_energy' in d.flags) is flagged


def test_small_amplitude_libration_without_owner_search_is_capped():
    """A < 45 deg: L4 not_measured -> step 6 (dynamics); the owner search is then required, and when it did not run
    the ceiling is medium with the flag owner_search_skipped (SPEC §5.3 last line; collect_caps, owner_searched)."""
    t = np.arange(0.0, 20000.0, 5.0)
    sigma, rate = pendulum_series(LIB_OMEGA0, np.radians(30), t)
    a = 2.5 + rate / C_ID
    angle = geo.FilteredAngle(t, sigma, sigma, rate, sigma, sigma, a, C_ID, 100.0, 5.0, (100.0, t[-1] - 100.0))
    g = geo.measure_geometry(angle, np.radians(1.0), SETTINGS)
    d = decide.decide_status(
        ANGLES[AngleStatus.LIBRATION],
        resolved_sampling(len(t), 5.0, t[-1]),
        g,
        None,
        _response(l4=CheckValue.NOT_MEASURED),
        _kinematics(),
        None,
        None,
        _strength(method='direct_averaging'),  # the strength by direct averaging passes: the dynamic leaf (W18)
        SETTINGS,
        Ledger.from_settings(SETTINGS),
    )
    assert d.status == Status.RESONANT
    assert d.origin == LibrationOrigin.DYNAMIC
    angle_decision = ANGLES[AngleStatus.LIBRATION]
    margins = {'L4': MarginValue(margin=Margin.NOT_MEASURED, value=None, margin_class=MarginClass.DECISIVE)}
    skipped = decide.collect_caps(decide.tree_view(angle_decision, None, None), d, g, margins, SETTINGS, owner_searched=False)
    searched = decide.collect_caps(decide.tree_view(angle_decision, None, None), d, g, margins, SETTINGS, owner_searched=True)
    assert 'owner_search_skipped' in skipped
    assert 'owner_search_skipped' not in searched
    level = conf.apply_caps(Confidence.CERTAIN, skipped)
    assert level in (Confidence.MEDIUM, Confidence.LOW)


CHECK_VALUES = (CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_MEASURED, CheckValue.NOT_APPLICABLE)


SUB_OWNER_VALUES = (CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_MEASURED, None)
KIN_R0_VALUES = (CheckValue.PASS, CheckValue.FAIL, CheckValue.NOT_MEASURED)
# the angle a dynamic leaf reads: the status transient is an episodic angle (renamed 2026-10-17)
DYNAMIC_ANGLE = {Status.TRANSIENT: AngleStatus.EPISODIC, Status.NEAR_SEPARATRIX: AngleStatus.NEAR_SEPARATRIX}
LEAF_BY_ANGLE = {
    AngleStatus.UNDEFINED: Status.INVALID,
    AngleStatus.INSUFFICIENT_TIME: Status.UNCERTAIN,
    AngleStatus.CIRCULATION: Status.NON_RESONANT,
}


def _check_leaf(angle_status, sub, d):
    assert isinstance(d.status, Status)
    assert d.reason != ''
    # an own-dynamics status has origin dynamic, or unknown when only the geometry decides (rule 5u/6, cap
    # geometry_only: L4 and R0 not measured, the strength not measured by direct averaging; W18; strength_scale_only
    # when the order-of-magnitude scale agrees, W22 R-F)
    if d.status in (Status.RESONANT, Status.TRANSIENT, Status.NEAR_SEPARATRIX):
        assert d.origin == LibrationOrigin.DYNAMIC or (d.origin == LibrationOrigin.UNKNOWN and d.rule == '5u/6')
        assert d.origin == LibrationOrigin.DYNAMIC or {'geometry_only', 'strength_scale_only'} & set(d.caps)
    if d.status == Status.RESONANT:
        assert angle_status == AngleStatus.LIBRATION
    if d.status in (Status.TRANSIENT, Status.NEAR_SEPARATRIX):
        assert angle_status == DYNAMIC_ANGLE[d.status]
    if d.origin in (LibrationOrigin.SUBRESONANCE, LibrationOrigin.INDUCED):
        assert d.status == Status.NON_RESONANT and d.owner
    if angle_status in LEAF_BY_ANGLE:
        assert d.status == LEAF_BY_ANGLE[angle_status]
    if angle_status in decide.OSCILLATING and sub == CheckValue.FAIL:
        assert d.reason == Reason.SUBRESONANCE_LIBRATES


@pytest.mark.slow
def test_every_combination_reaches_a_leaf(libration_geometry, clean_energy):
    """SPEC §5.3 completeness: every combination of check values x angle_status reaches a leaf, and the leaves obey
    the tree's invariants (resonant only from a libration with dynamic origin; a foreign origin names its owner;
    undefined is invalid, insufficient_time uncertain, circulation non_resonant)."""
    g = libration_geometry[1]
    combos = itertools.product(
        ANGLES, SUB_OWNER_VALUES, SUB_OWNER_VALUES, CHECK_VALUES, KIN_R0_VALUES, CHECK_VALUES, KIN_R0_VALUES, (True, False)
    )
    n = 0
    for angle_status, sub, own, b1, kin, l4, r0, b5_fails in combos:
        b5 = CheckValue.FAIL if b5_fails else CheckValue.PASS
        d = _decide(angle_status, g, clean_energy, l4, r0, b1, b5, b5_fails, kin, sub, own)
        _check_leaf(angle_status, sub, d)
        n += 1
    assert n == 7 * 4 * 4 * 4 * 3 * 4 * 3 * 2


# ----------------------------------------------------------------------------------------------------------------
# baseline (SPEC §4.10): required = (TP_min / 2) P + 2 trim P_c
# ----------------------------------------------------------------------------------------------------------------


def test_baseline_by_hand(libration_geometry):
    """P = 1180 yr, P_c = 100 yr: required = 1.5 x 1180 + 2 x 1 x 100 = 1970 yr."""
    _, g = libration_geometry
    p = g.period_cycle_yr
    required = SETTINGS.libration_min_turning_points / 2 * p + 2 * SETTINGS.filter_edge_trim_cutoffs * 100.0
    assert required == pytest.approx(1970.0, rel=0.01)
    ok = decide.baseline(g, 100.0, 20000.0, NAN, SETTINGS)
    assert ok.required_yr == pytest.approx(required, rel=1e-9)
    assert ok.check == CheckValue.PASS
    short = decide.baseline(g, 100.0, 1500.0, NAN, SETTINGS)
    assert short.check == CheckValue.FAIL
    # a round value >= max(required, 2 T) = max(1970, 3000)
    assert short.extend_to_years >= 3000.0
    digits = f'{short.extend_to_years:.0f}'.rstrip('0')
    assert len(digits) == 1


def test_baseline_without_any_period():
    b = decide.baseline(None, 100.0, 5000.0, NAN, SETTINGS)
    assert b.check == CheckValue.NOT_MEASURED


# ----------------------------------------------------------------------------------------------------------------
# confidence (SPEC §7.2, DECISIONS 2026-10-15): the W2 cases
# ----------------------------------------------------------------------------------------------------------------


def _m(margin, cls=MarginClass.DECISIVE, value=1.0):
    return MarginValue(margin=margin, value=value, margin_class=cls)


FULL_LIBRATION = {
    'L1': _m(Margin.FULL),
    'L2': _m(Margin.FULL),
    'L4': _m(Margin.FULL),
    'R0': _m(Margin.FULL),
    'Z': _m(Margin.FULL),
    'L3': _m(Margin.FULL, MarginClass.SUPPORT),
    'L5': _m(Margin.FULL, MarginClass.SUPPORT),
}
NO_CELLS = conf.ConfidenceCells()


def _assess(margins, caps=(), status=Status.RESONANT, angle='libration', reason='dynamic_libration', cells=NO_CELLS):
    return conf.assess(status, angle, reason, margins, list(caps), [], decide.PHYSICS_MARGINS, cells)


def test_r2_libration_thin_l2_stays_high():
    """§5.4 R2 / §7.2: libration_in_separatrix_zone is a thin L2, not a ceiling -> high."""
    margins = dict(FULL_LIBRATION, L2=_m(Margin.THIN))
    a = _assess(margins)
    assert a.confidence == Confidence.HIGH
    assert 'L2' in a.thin_margins


def test_all_full_without_a_cell_is_certain():
    # author 2026-10-16 (C2): no minimum of labels; the level from the margins is shown
    a = _assess(FULL_LIBRATION)
    assert a.confidence_raw == Confidence.CERTAIN
    assert a.confidence == Confidence.CERTAIN


def test_episodes_in_separatrix_zone_transient_is_medium():
    """DECISIONS 2026-10-15 evening: 52356 -> transient/medium + flag."""
    a = _assess(FULL_LIBRATION, caps=['episodes_in_separatrix_zone'], status=Status.TRANSIENT, angle='episodic', reason='dynamic_episode')
    assert a.confidence == Confidence.MEDIUM


def test_force_law_disagrees_caps_medium_and_onset_failure_low():
    assert _assess(FULL_LIBRATION, caps=['force_law_disagrees']).confidence == Confidence.MEDIUM
    assert _assess(FULL_LIBRATION, caps=['onset_failed', 'exit_suspected']).confidence == Confidence.LOW


def test_all_physics_not_measured_is_at_most_medium():
    """Every physics margin not measured: at most medium (DECISIONS 2026-10-15: 'если не измерено ничего из физики')."""
    margins = dict(FULL_LIBRATION)
    for k in decide.PHYSICS_MARGINS:
        margins[k] = _m(Margin.NOT_MEASURED, value=None)
    a = _assess(margins)
    assert a.confidence in (Confidence.MEDIUM, Confidence.LOW)


def test_one_support_fail_medium_two_low():
    one = dict(FULL_LIBRATION, L5=_m(Margin.FAIL, MarginClass.SUPPORT))
    two = dict(one, L3=_m(Margin.FAIL, MarginClass.SUPPORT))
    assert _assess(one).confidence == Confidence.MEDIUM
    assert _assess(two).confidence == Confidence.LOW


def test_cells_from_settings_are_reported_and_never_gate(tmp_path):
    """Author 2026-10-16 (C2, decision 7) replaces DELTA 13(d): the cells file is read from CONFIDENCE_CELLS and
    reported; any number of examples and errors leaves the level the margins give."""
    path = tmp_path / 'cells.json'
    rows = [
        dict(status='resonant', angle_status='libration', reason='dynamic_libration', n_examples=100, n_errors=0),
        dict(status='transient', angle_status='episodic', reason='dynamic_episode', n_examples=120, n_errors=1),
        dict(status='non_resonant', angle_status='circulation', reason='circulation', n_examples=99, n_errors=0),
    ]
    path.write_text(json.dumps({'version': 1, 'cells': rows}))
    cells = conf.cells_from_settings(settings_from_config(confidence_cells=str(path)))
    assert {key: (r.n_examples, r.n_errors) for key, r in cells.records.items()} == {
        ('resonant', 'libration', 'dynamic_libration'): (100, 0),
        ('transient', 'episodic', 'dynamic_episode'): (120, 1),
        ('non_resonant', 'circulation', 'circulation'): (99, 0),
    }
    a = _assess(FULL_LIBRATION, cells=cells)
    assert a.confidence == Confidence.CERTAIN and a.cell.n_examples == 100
    b = _assess(FULL_LIBRATION, status=Status.TRANSIENT, angle='episodic', reason='dynamic_episode', cells=cells)
    assert b.confidence == Confidence.CERTAIN and (b.cell.n_examples, b.cell.n_errors) == (120, 1)


@pytest.mark.parametrize('status', [Status.UNCERTAIN, Status.INVALID])
def test_uncertain_and_invalid_have_no_level(status):
    a = _assess(FULL_LIBRATION, status=status, angle='drift', reason='drift_mechanism_unknown')
    assert a.confidence == Confidence.NONE and a.confidence_raw == Confidence.NONE
