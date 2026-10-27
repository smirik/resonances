"""Independent tests of the W3 physics fixes (F2, F4, G1, G2, G5, G6, G7) of the v2 libration analysis.

Written from the public API, the docstrings and the formulas of W3-FIXES.md / PHYSICS-MATH.md only. Every expected
value is a closed form worked out by hand in a comment, an independent quadrature / ODE integration in the test, or
the known truth of a synthetic series.
"""

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from resonances.resonance.analysis import filtering, geometry, pendulum
from resonances.resonance.analysis.geometry import EpisodeShape, FilteredAngle, TurningPoints
from resonances.resonance.analysis.mmr import kinematics
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import (
    MMRSeries,
    PlanetSeries,
    Regime,
    ResonanceKind,
    ResonanceSpec,
)
from resonances.resonance.analysis.mmr.response import phase_tolerance
from resonances.resonance.analysis.settings import AnalysisSettings

SETTINGS = AnalysisSettings()


# ----------------------------------------------------------------------------------------------------------------------
# G5. Digital Butterworth gain of filtfilt: G = 1 / (1 + (tan(pi dt/P) / tan(pi dt/Pc))^(2N))
# ----------------------------------------------------------------------------------------------------------------------


def test_butterworth_gain_digital_hand_values():
    # dt = 1, Pc = 5, N = 2.
    # P = 10: tan(18 deg) / tan(36 deg) = 0.3249197 / 0.7265425 = 0.4472136 (= 1/sqrt 5); ^4 = 0.04; G = 1/1.04.
    assert filtering.butterworth_gain(10.0, 5.0, 2, step_yr=1.0) == pytest.approx(1 / 1.04, rel=1e-9)
    # P = 10/3: tan(54 deg) / tan(36 deg) = 1 / tan^2(36 deg) = 1 / (5 - 2 sqrt 5) = 1.8944272;
    # ^4 = 1 / (4025 - 1800 sqrt 5) = 12.87987; G = 1/13.87987 = 0.0720468.
    ratio4 = 1 / (4025 - 1800 * np.sqrt(5))
    assert filtering.butterworth_gain(10.0 / 3.0, 5.0, 2, step_yr=1.0) == pytest.approx(1 / (1 + ratio4), rel=1e-9)
    # At the cutoff the ratio is 1 for any step: G = 1/2.
    for dt in (0.1, 1.0, 2.0):
        assert filtering.butterworth_gain(5.0, 5.0, 2, step_yr=dt) == pytest.approx(0.5, rel=1e-12)


def test_butterworth_gain_analog_limit():
    # step None: G = 1/(1 + (Pc/P)^(2N)). N = 2: G(2Pc) = 1/(1 + 1/16) = 0.9411765, G(Pc/1.5) = 1/(1 + 5.0625).
    assert filtering.butterworth_gain(10.0, 5.0, 2) == pytest.approx(16 / 17, rel=1e-12)
    assert filtering.butterworth_gain(5.0, 5.0, 2) == pytest.approx(0.5, rel=1e-12)
    assert filtering.butterworth_gain(5.0 / 1.5, 5.0, 2) == pytest.approx(1 / 6.0625, rel=1e-12)
    # N = 3, P = Pc/2: 1/(1 + 2^6) = 1/65.
    assert filtering.butterworth_gain(2.5, 5.0, 3) == pytest.approx(1 / 65, rel=1e-12)


def test_butterworth_gain_array_and_nyquist():
    periods = np.array([1.5, 2.0, 10.0, 100.0])
    gains = np.asarray(filtering.butterworth_gain(periods, 5.0, 2, step_yr=1.0), dtype=float)
    assert gains.shape == periods.shape
    # P <= 2 dt is at or above Nyquist: no steady-state gain (nan by the docstring).
    assert np.isnan(gains[0]) and np.isnan(gains[1])
    assert gains[2] == pytest.approx(1 / 1.04, rel=1e-9)
    # P = 100: tan(pi/100)/tan(pi/5) = 0.0314263/0.7265425 = 0.0432544; ^4 = 3.50e-6.
    assert gains[3] == pytest.approx(1 / (1 + (np.tan(np.pi / 100) / np.tan(np.pi / 5)) ** 4), rel=1e-12)


def _measured_gain(period, cutoff, dt=1.0, n=6000, order=2):
    """Amplitude of a filtered unit sinusoid in the interior, by least squares on sin and cos."""
    t = np.arange(n) * dt
    x = np.sin(2 * np.pi * t / period)
    y = filtering.lowpass(t, x, cutoff, order, detrend=False)
    inner = slice(n // 6, n - n // 6)
    design = np.column_stack([np.sin(2 * np.pi * t[inner] / period), np.cos(2 * np.pi * t[inner] / period)])
    coef, *_ = np.linalg.lstsq(design, y[inner], rcond=None)
    return float(np.hypot(*coef))


@pytest.mark.parametrize("period", [10.0, 10.0 / 3.0, 7.0])
def test_lowpass_matches_digital_gain_not_analog(period):
    # Pc/dt = 5: the analog form is off by > 2 % at P = 10 and by a factor 2.3 at P = 10/3 (PHYSICS-MATH §3).
    measured = _measured_gain(period, 5.0)
    digital = float(filtering.butterworth_gain(period, 5.0, 2, step_yr=1.0))
    analog = float(filtering.butterworth_gain(period, 5.0, 2))
    assert measured == pytest.approx(digital, abs=2e-4)
    assert abs(measured - analog) > 0.01


def test_lowpass_preserves_line_with_detrend_and_removes_fast_wave():
    t = np.arange(3000) * 2.0
    line = 0.3 + 1e-3 * t
    out = filtering.lowpass(t, line + 0.5 * np.sin(2 * np.pi * t / 8.0), 100.0, 2, detrend=True)
    inner = slice(300, -300)
    # G(8 yr) with dt=2, Pc=100: tan(pi/4)/tan(pi/50) = 1/0.0629 = 15.9; ^4 = 6.4e4; residual wave ~ 0.5/6.4e4.
    assert np.max(np.abs(out[inner] - line[inner])) < 1e-4
    assert out.shape == t.shape


def test_lowpass_input_checks_and_grid_tolerance():
    t = np.arange(1000, dtype=float)
    x = np.sin(t / 30.0)
    with pytest.raises(ValueError):
        filtering.lowpass(t, x, 1.5, 2)  # cutoff shorter than two steps
    t_bad = t.copy()
    t_bad[500:] += 0.5  # one step of 1.5: off the median by 50 %
    with pytest.raises(ValueError):
        filtering.lowpass(t_bad, x, 50.0, 2)
    rng = np.random.default_rng(1)
    jitter = rng.uniform(-1, 1, t.size)
    t_small = t + 2e-4 * jitter  # steps off by <= 4e-4 < default 1e-3
    assert np.all(np.isfinite(filtering.lowpass(t_small, x, 50.0, 2)))
    t_large = t + 4e-3 * jitter  # steps off by up to 8e-3
    with pytest.raises(ValueError):
        filtering.lowpass(t_large, x, 50.0, 2)
    assert np.all(np.isfinite(filtering.lowpass(t_large, x, 50.0, 2, grid_rtol=2e-2)))


def test_edge_window_and_mask():
    t = np.arange(0.0, 101.0)
    assert filtering.edge_window(t, 10.0) == (10.0, 90.0)
    w = filtering.edge_window(t, 60.0)
    assert np.isnan(w[0]) and np.isnan(w[1])
    mask = filtering.window_mask(t, (10.0, 90.0))
    assert mask.sum() == 81 and mask[10] and mask[90] and not mask[9] and not mask[91]


# ----------------------------------------------------------------------------------------------------------------------
# G7. Pendulum: closed forms against quadratures; action and half-width of a multi-harmonic V
# ----------------------------------------------------------------------------------------------------------------------


def _factor_by_quadrature(amplitude):
    # T/T0 = (2/pi) int_0^{pi/2} d theta / sqrt(1 - m sin^2 theta), m = sin^2(A/2) (no ellipk).
    m = np.sin(amplitude / 2) ** 2
    return 2 / np.pi * quad(lambda th: 1 / np.sqrt(1 - m * np.sin(th) ** 2), 0, np.pi / 2)[0]


@pytest.mark.parametrize("deg", [10.0, 90.0, 120.0, 150.0, 170.0])
def test_amplitude_factor_against_quadrature(deg):
    a = np.radians(deg)
    assert pendulum.amplitude_factor(a) == pytest.approx(_factor_by_quadrature(a), rel=1e-9)


def test_amplitude_factor_published_values():
    # M&D 1999 eq. 8.50-8.58: F(90) = 1.18034, F(120) = 1.37288, F(150) = 1.76220.
    for deg, value in ((90, 1.18034), (120, 1.37288), (150, 1.76220)):
        assert pendulum.amplitude_factor(np.radians(deg)) == pytest.approx(value, abs=1e-5)


def _libration_period_by_ode(omega, amplitude):
    """Period of phi'' = -omega^2 sin(phi) from phi = A, phi' = 0: four times the time to reach phi = 0."""

    def rhs(_, y):
        return [y[1], -(omega**2) * np.sin(y[0])]

    def zero(_, y):
        return y[0]

    zero.terminal = True
    zero.direction = -1
    sol = solve_ivp(rhs, (0, 100 / omega), [amplitude, 0.0], events=zero, rtol=1e-11, atol=1e-12)
    return 4 * sol.t_events[0][0]


def test_libration_period_against_ode():
    omega, amp = 2.0, np.radians(130.0)
    h = np.sin(amp / 2) ** 2  # h_turn = sin^2(A/2)
    assert pendulum.libration_period(h, omega) == pytest.approx(_libration_period_by_ode(omega, amp), rel=1e-7)


def test_revolution_rate_ratio_against_quadrature():
    # V = -omega^2 cos phi, E = -omega^2 + 2 omega^2 h; phi_dot = sqrt(2 (E - V)).
    omega, h = 1.5, 1.14
    energy = -(omega**2) + 2 * omega**2 * h

    def rate(phi):
        return np.sqrt(2 * (energy + omega**2 * np.cos(phi)))

    period = quad(lambda p: 1 / rate(p), -np.pi, np.pi)[0]
    mean_rate = 2 * np.pi / period
    min_rate = rate(np.pi)  # = 2 omega sqrt(h - 1)
    assert pendulum.revolution_period(h, omega) == pytest.approx(period, rel=1e-8)
    assert pendulum.rate_min_over_mean(h) == pytest.approx(min_rate / mean_rate, rel=1e-8)
    assert pendulum.rate_min_over_mean(h) == pytest.approx(0.554, abs=1e-3)  # docstring value


def _action(potential_fn, energy, left, right):
    """J(E) = (1/pi) int sqrt(2 (E - V)) d phi between the turning points of the loop around the well at 0."""
    if energy >= potential_fn(left) and energy >= potential_fn(right):
        lo, hi = left, right
    else:
        lo = brentq(lambda p: potential_fn(p) - energy, left, 0.0)
        hi = brentq(lambda p: potential_fn(p) - energy, 0.0, right)
    return quad(lambda p: np.sqrt(max(0.0, 2 * (energy - potential_fn(p)))), lo, hi, limit=200)[0] / np.pi


def test_cosine_action_ratio_against_quadrature():
    omega = 1.3

    def v(p):
        return -(omega**2) * np.cos(p)

    js = _action(v, omega**2, -np.pi, np.pi)
    assert js == pytest.approx(8 * omega / np.pi, rel=1e-6)
    for h in (0.05, 0.5, 0.9):
        expected = _action(v, -(omega**2) + 2 * omega**2 * h, -np.pi, np.pi) / js
        assert pendulum.action_ratio(h) == pytest.approx(expected, rel=1e-6)
    assert pendulum.action_ratio(1e-4) == pytest.approx(np.pi * 1e-4 / 4, rel=1e-3)
    assert pendulum.action_ratio(1.0) == pytest.approx(1.0, abs=1e-9)
    assert pendulum.h_from_action_ratio(pendulum.action_ratio(0.37)) == pytest.approx(0.37, abs=1e-8)


# V = -cos(phi) - 0.2 cos(2 phi): V' = sin(phi) (1 + 0.8 cos(phi)) vanishes only at 0 and pi.
# V_min = V(0) = -1.2, V_max = V(pi) = -(-1) - 0.2 = 0.8, depth 2, V''(0) = 1 + 0.8 = 1.8, Omega = sqrt(1.8).
TWO_HARMONIC = (np.array([-1.0, -0.2]), np.array([0.0, 0.0]))


def _v2(p):
    return -np.cos(p) - 0.2 * np.cos(2 * p)


def test_two_harmonic_potential_shape():
    pot = pendulum.potential_from_coefficients(*TWO_HARMONIC)
    assert pot.v_min == pytest.approx(-1.2, abs=1e-9)
    assert pot.v_max == pytest.approx(0.8, abs=1e-9)
    assert np.cos(pot.well_rad) == pytest.approx(1.0, abs=1e-9)
    assert np.cos(pot.top_rad) == pytest.approx(-1.0, abs=1e-9)
    assert pot.omega_rad_yr == pytest.approx(np.sqrt(1.8), rel=1e-6)
    left, right, e_sep = pendulum.well_separatrix(pot)
    assert e_sep == pytest.approx(0.8, abs=1e-9)
    assert right - left == pytest.approx(2 * np.pi, abs=1e-6)


def test_two_harmonic_action_ratio_by_quadrature():
    # PHYSICS-MATH §8: at h = 0.5 the true J/J_s = 0.38417, the cosine closed form gives 0.42361 (10 % off).
    pot = pendulum.potential_from_coefficients(*TWO_HARMONIC)
    js = _action(_v2, 0.8, -np.pi, np.pi)
    assert pendulum.separatrix_action(pot) == pytest.approx(js, rel=1e-5)
    for h in (0.1, 0.5, 0.8):
        energy = -1.2 + 2.0 * h
        expected = _action(_v2, energy, -np.pi, np.pi) / js
        assert pendulum.action_ratio_in_potential(pot, h) == pytest.approx(expected, rel=1e-4)
    assert _action(_v2, -0.2, -np.pi, np.pi) / js == pytest.approx(0.3841667556, rel=1e-6)
    assert pendulum.action_ratio_in_potential(pot, 1.0) == pytest.approx(1.0, abs=1e-9)
    assert pendulum.action_ratio_in_potential(pot, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_one_harmonic_quadrature_route_matches_closed_form():
    omega = 0.7
    pot = pendulum.potential_from_coefficients([-(omega**2)], [0.0])
    assert pendulum.separatrix_action(pot) == pytest.approx(8 * omega / np.pi, rel=1e-6)
    assert pendulum.action_ratio_in_potential(pot, 0.5) == pytest.approx(pendulum.action_ratio(0.5), rel=1e-6)
    c = -0.04
    assert pendulum.half_width_in_potential(pot, c) == pytest.approx(2 * omega / abs(c), rel=1e-6)
    assert pendulum.half_width(omega, c) == pytest.approx(2 * omega / abs(c), rel=1e-12)


def test_half_width_cos2_is_not_two_sqrt_curvature():
    # V = -cos(2 s): V'' at the well = 4 -> the cosine formula 2 sqrt(V'')/|c| = 4/|c|,
    # but the separatrix rate is sqrt(2 (E_sep - V_min)) = sqrt(2 * 2) = 2: half-width 2/|c|.
    pot = pendulum.potential_from_coefficients([0.0, -1.0], [0.0, 0.0])
    c = 0.5
    assert pendulum.half_width_in_potential(pot, c) == pytest.approx(2 / c, rel=1e-6)
    left, right, e_sep = pendulum.well_separatrix(pot)
    assert e_sep == pytest.approx(1.0, abs=1e-9)
    assert right - left == pytest.approx(np.pi, abs=1e-6)


def _two_harmonic_cycle(h):
    """One libration of phi'' = -V'(phi) for V = -cos - 0.2 cos 2 at energy h, on a uniform grid; (t, phi, rate, T)."""
    energy = -1.2 + 2.0 * h

    def rhs(_, y):
        return [y[1], -(np.sin(y[0]) + 0.4 * np.sin(2 * y[0]))]

    def up_zero(_, y):
        return y[0]

    up_zero.direction = 1
    rate0 = np.sqrt(2 * (energy + 1.2))
    probe = solve_ivp(rhs, (0, 40), [0.0, rate0], events=up_zero, rtol=1e-11, atol=1e-12)
    period = probe.t_events[0][1]  # the first event is t = 0
    t = np.linspace(0, 3 * period, 6001)
    sol = solve_ivp(rhs, (0, t[-1]), [0.0, rate0], t_eval=t, rtol=1e-11, atol=1e-12)
    return t, sol.y[0], sol.y[1], period


def test_cycle_energies_two_harmonic_action_and_period():
    h = 0.5
    t, phi, rate, period = _two_harmonic_cycle(h)
    pot = pendulum.potential_from_coefficients(*TWO_HARMONIC)
    assert pendulum.period_in_potential(pot, h) == pytest.approx(period, rel=1e-5)
    bounds = [(k * period, (k + 1) * period, False) for k in range(3)]
    cycles = pendulum.cycle_energies(t, phi, rate, bounds, pot)
    assert len(cycles) == 3
    js = _action(_v2, 0.8, -np.pi, np.pi)
    expected = _action(_v2, -1.2 + 2.0 * h, -np.pi, np.pi) / js  # 0.38417
    for cyc in cycles:
        assert not cyc.rotation
        assert cyc.h == pytest.approx(h, abs=1e-4)
        assert cyc.action_ratio == pytest.approx(expected, rel=2e-3)
        assert abs(cyc.action_ratio - pendulum.action_ratio(h)) > 0.03  # not the cosine 0.4236
        assert cyc.period_law_ratio == pytest.approx(1.0, abs=2e-3)


# ----------------------------------------------------------------------------------------------------------------------
# G6. Phase tolerance of a multiplet: arcsin(r) only while r < 1
# ----------------------------------------------------------------------------------------------------------------------


def test_phase_tolerance_against_disk_geometry():
    theta = np.linspace(-np.pi, np.pi, 200001)
    for r in (0.1, 0.5, 0.9):
        # largest |arg(1 + r e^{i theta})| on the circle |z| = r (the disk's extreme phase is on its rim)
        widest = np.max(np.abs(np.angle(1 + r * np.exp(1j * theta))))
        assert phase_tolerance([r], 0.0, 0.0) == pytest.approx(widest, abs=1e-6)
        assert phase_tolerance([r / 2, r / 2], 0.0, 0.0) == pytest.approx(widest, abs=1e-6)
    # r >= 1: 1 + 1.5 e^{i pi} = -0.5, the phase turns by pi; no bound.
    assert np.angle(1 + 1.5 * np.exp(1j * np.pi)) == pytest.approx(np.pi)
    assert np.isnan(phase_tolerance([1.5], 0.0, 0.0))
    assert np.isnan(phase_tolerance([0.6, 0.4], 0.0, 0.0))
    # floor and 2 sigma
    assert phase_tolerance([0.5], 0.1, 0.0) == pytest.approx(np.pi / 6 + 0.2, abs=1e-12)
    assert phase_tolerance([0.01], 0.0, 1.0) == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------------------------------------
# G1. Free / forced branches of z = e_free exp(i g t) + e_forced exp(i g5 t)
# ----------------------------------------------------------------------------------------------------------------------


def _varpi_rate_numeric(e_free, e_forced, g, g5, n=200001):
    """varpi_dot by finite differences of the unwrapped argument of z over one relative cycle."""
    period = 2 * np.pi / abs(g - g5)
    t = np.linspace(0, period, n)
    z = e_free * np.exp(1j * g * t) + e_forced * np.exp(1j * g5 * t)
    rate = np.gradient(np.unwrap(np.angle(z)), t)
    return rate[1:-1].min(), rate[1:-1].max()


def test_varpi_rate_range_forced_dominant_counterexample():
    # W3-FIXES G1: e_free = .05, e_forced = .2, g = 1, g5 = 0.
    # varpi_dot = (ef^2 + ef eF cos th)/(ef^2 + eF^2 + 2 ef eF cos th):
    # th = 0: (.0025 + .01)/.0625 = .2;  th = pi: (.0025 - .01)/.0225 = -1/3.
    lo, hi = kinematics.varpi_rate_range(0.05, 0.2, 1.0, 0.0)
    assert lo == pytest.approx(-1 / 3, rel=1e-9)
    assert hi == pytest.approx(0.2, rel=1e-9)
    # free-dominant mirror: th = 0: (.04 + .01)/.0625 = .8; th = pi: (.04 - .01)/.0225 = 4/3.
    lo, hi = kinematics.varpi_rate_range(0.2, 0.05, 1.0, 0.0)
    assert lo == pytest.approx(0.8, rel=1e-9)
    assert hi == pytest.approx(4 / 3, rel=1e-9)


@pytest.mark.parametrize("e_free,e_forced,g,g5", [(0.07, 0.11, 0.5, -0.3), (0.11, 0.07, 0.5, -0.3), (0.02, 0.3, -2.0, 0.4)])
def test_varpi_rate_range_against_finite_differences(e_free, e_forced, g, g5):
    lo_num, hi_num = _varpi_rate_numeric(e_free, e_forced, g, g5)
    lo, hi = kinematics.varpi_rate_range(e_free, e_forced, g, g5)
    assert lo == pytest.approx(lo_num, rel=1e-4, abs=1e-6)
    assert hi == pytest.approx(hi_num, rel=1e-4, abs=1e-6)


def test_varpi_rate_range_degenerate_cases():
    assert kinematics.varpi_rate_range(0.0, 0.2, 1.0, 0.3) == pytest.approx((0.3, 0.3))
    lo, hi = kinematics.varpi_rate_range(0.1, 0.1, 1.0, 0.0)
    assert lo == -np.inf and hi == np.inf
    lo, hi = kinematics.varpi_rate_range(0.0, 0.0, 1.0, 0.0)
    assert np.isnan(lo) and np.isnan(hi)


def test_varpi_rate_modulation_zero_e_min_no_division_by_zero():
    # e = |0.1 e^{i th} + 0.1| reaches 0: e_max = .2, e_min = 0 -> e_free = e_forced = .1, rate unbounded.
    e = np.array([0.2, 0.15, 0.1, 0.0, 0.1, 0.2])
    g, g5 = 2e-4, 2e-5
    e_free, e_forced, lo, hi, period = kinematics.varpi_rate_modulation(e, g, g5)
    assert e_free == pytest.approx(0.1) and e_forced == pytest.approx(0.1)
    assert lo == -np.inf and hi == np.inf
    assert period == pytest.approx(2 * np.pi / 1.8e-4, rel=1e-12)


ARCSEC = np.pi / 180 / 3600
G5 = 4.2575 * ARCSEC  # rad/yr
G_FREE = 40.0 * ARCSEC  # rad/yr; relative period 2 pi / (35.7425") = 36 260 yr


def _synthetic_vector(e_free, e_forced, span_yr, n=4001):
    t = np.linspace(0.0, span_yr, n)
    z = e_free * np.exp(1j * (G_FREE * t + 0.3)) + e_forced * np.exp(1j * (G5 * t + 1.1))
    return t, np.abs(z), np.mod(np.angle(z), 2 * np.pi)


@pytest.mark.parametrize("e_free,e_forced", [(0.05, 0.2), (0.2, 0.05), (0.12, 0.1)])
def test_free_forced_split_recovers_the_branch(e_free, e_forced):
    t, e, varpi = _synthetic_vector(e_free, e_forced, 200_000.0)
    # independent sanity of the branch: varpi turns on average at the frequency of the larger radius
    slope = np.polyfit(t, np.unwrap(varpi), 1)[0]
    assert slope == pytest.approx(G5 if e_forced > e_free else G_FREE, rel=0.1)
    ef, efo, g = kinematics.free_forced_split(t, e, varpi, G5)
    assert ef == pytest.approx(e_free, rel=1e-3)
    assert efo == pytest.approx(e_forced, rel=1e-3)
    assert g == pytest.approx(G_FREE, rel=1e-3)
    lo, hi = kinematics.varpi_rate_range(ef, efo, g, G5)
    rate = np.gradient(np.unwrap(varpi), t)[1:-1]
    assert lo == pytest.approx(rate.min(), rel=2e-2)
    assert hi == pytest.approx(rate.max(), rel=2e-2)


def test_free_forced_split_short_record():
    # 0.6 of a relative cycle; the docstring promises the two-vector fit, not the biased projection.
    t, e, varpi = _synthetic_vector(0.05, 0.2, 0.6 * 2 * np.pi / (G_FREE - G5), n=2001)
    ef, efo, g = kinematics.free_forced_split(t, e, varpi, G5)
    assert ef == pytest.approx(0.05, rel=1e-2)
    assert efo == pytest.approx(0.2, rel=1e-2)
    assert g == pytest.approx(G_FREE, rel=1e-2)


def test_free_forced_split_without_free_vector_and_without_data():
    t = np.linspace(0.0, 100_000.0, 2001)
    z = 0.2 * np.exp(1j * (G5 * t + 0.4))
    ef, efo, g = kinematics.free_forced_split(t, np.abs(z), np.mod(np.angle(z), 2 * np.pi), G5)
    assert efo == pytest.approx(0.2, rel=1e-6)
    assert ef < 1e-6
    assert np.isnan(g)  # no free vector, no phase: undefined, not a division by zero
    out = kinematics.free_forced_split(np.array([0.0, 1.0]), np.array([0.1, 0.1]), np.array([0.0, 0.1]), G5)
    assert all(np.isnan(x) for x in out)


# ----------------------------------------------------------------------------------------------------------------------
# G2. Three turning points give one centre: stationarity is not measured
# ----------------------------------------------------------------------------------------------------------------------


def _turning_points(times, values):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    kinds = np.where(np.arange(len(values)) % 2 == 0, 1, -1) * (1 if values[0] > values[1] else -1)
    # the time grids of these tests start at 0 with a 1-yr step: the index of a TP is its time
    return TurningPoints(np.round(times).astype(int), times, values, kinds, 0.05)


def test_centre_bands_overlap_all_pairs():
    # bands [m - A, m + A]: max(m - A) < min(m + A)
    assert geometry.centre_bands_overlap(np.array([0.0, 0.2, 0.4]), np.array([0.3, 0.3, 0.3]))  # 0.1 < 0.3
    # consecutive bands overlap, first and last do not: 0.7 > 0.3
    assert not geometry.centre_bands_overlap(np.array([0.0, 0.5, 1.0]), np.array([0.3, 0.3, 0.3]))
    assert not geometry.centre_bands_overlap(np.array([]), np.array([]))


def test_three_turning_points_centre_not_measured():
    # PHYSICS-MATH §7: TPs [2, 0, 5.9] at [0, 500, 1000]: half-cycle midpoints 1 and 2.95, one full cycle.
    t = np.arange(0.0, 1001.0)
    tp = _turning_points([0, 500, 1000], [2.0, 0.0, 5.9])
    ep = geometry.classify_run(0, tp, np.arange(3), t, (0.0, 1000.0), SETTINGS)
    assert ep.accepted  # the >= 3 TP rule stays (author)
    assert ep.centre_estimates == 1
    assert ep.centre_stationarity == 'not_measured'


def test_five_turning_points_stationary_centre_is_measured():
    t = np.arange(0.0, 2001.0)
    tp = _turning_points([0, 500, 1000, 1500, 2000], [1.0, -1.0, 1.05, -0.95, 1.0])
    ep = geometry.classify_run(0, tp, np.arange(5), t, (0.0, 2000.0), SETTINGS)
    assert ep.accepted and ep.shape == EpisodeShape.STATIONARY
    assert ep.centre_estimates >= 2
    assert ep.centre_stationarity not in ('not_measured', '')


def test_drifting_centre_is_not_stationary():
    # 15874-like: 10 TPs, the centre moves monotonically by 262 deg, half-amplitude 0.5 rad.
    n = 10
    mids = np.linspace(0.0, np.radians(262.0), n)
    values = mids + 0.5 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    t = np.arange(0.0, 500.0 * (n - 1) + 1)
    tp = _turning_points(500.0 * np.arange(n), values)
    ep = geometry.classify_run(0, tp, np.arange(n), t, (0.0, t[-1]), SETTINGS)
    assert not (ep.accepted and ep.shape == EpisodeShape.STATIONARY)


# ----------------------------------------------------------------------------------------------------------------------
# F2. A jump of a at the very edge cannot be confirmed and must not invert the window
# ----------------------------------------------------------------------------------------------------------------------


def test_jump_edge_margin():
    # max(tau, libration_jump_edge_periods x P); default edge periods = 1
    k = SETTINGS.libration_jump_edge_periods
    assert geometry.jump_edge_margin_yr(400.0, 2000.0, SETTINGS) == pytest.approx(max(400.0, k * 2000.0))
    assert geometry.jump_edge_margin_yr(3000.0, 2000.0, SETTINGS) == pytest.approx(max(3000.0, k * 2000.0))


PERIOD = 2000.0
STEP = 10.0
CUTOFF = 200.0


def _libration_with_steps(step_times, step_size=0.01):
    t = np.arange(0.0, 100_000.0 + STEP / 2, STEP)
    phase = 2 * np.pi * t / PERIOD
    angle = np.sin(phase)
    rate = 2 * np.pi / PERIOD * np.cos(phase)
    slow = 2.5 + 1e-4 * np.cos(phase + 0.4)
    for ts in step_times:
        slow = slow + step_size * (t >= ts)
    fa = FilteredAngle(
        t_yr=t,
        angle_raw=angle,
        angle_f=angle,
        rate_f=rate,
        angle_f_half=angle,
        angle_f_double=angle,
        slow_f=slow,
        identity_coefficient=-0.5,
        cutoff_yr=CUTOFF,
        step_yr=STEP,
        window=(t[0] + CUTOFF, t[-1] - CUTOFF),
    )
    return fa


def _finish(fa, cycle_h):
    geo = geometry.measure_geometry(fa, 0.1, SETTINGS)
    cycle_times = np.arange(PERIOD / 2, 100_000.0, PERIOD)
    return geometry.finish_geometry(
        geo,
        fa,
        small_period_yr=PERIOD / float(pendulum.amplitude_factor(1.0)),
        top_rad=np.pi,
        h_hi=1.0,
        sigma_h=0.02,
        cycle_times_yr=cycle_times,
        cycle_h=cycle_h(cycle_times),
        half_width=1e-3,  # the steps (0.01) exceed it: a slow-only rule alone would confirm them
        reference_period_yr=PERIOD,
        suspect_times_yr=[],
        settings=SETTINGS,
    )


def test_steps_at_the_edges_do_not_cut_the_record():
    fa = _libration_with_steps([300.0, 99_700.0])
    geo = _finish(fa, lambda tc: np.full(tc.shape, 0.3))
    w0, w1 = geo.window
    assert w0 < w1
    assert w1 - w0 > 0.9 * (fa.window[1] - fa.window[0])
    assert geo.regime_start_yr <= fa.window[0] + 1e-9
    assert not any((not j.kick) and (not j.at_edge) for j in geo.jumps)
    assert any(j.at_edge for j in geo.jumps)  # the steps were seen, and kept as edge kicks
    assert 'jump_at_edge' in geo.flags
    assert not any(ep.before_last_jump for ep in geo.runs)


def test_mid_record_jump_with_energy_change_cuts_the_record():
    fa = _libration_with_steps([50_000.0])
    geo = _finish(fa, lambda tc: np.where(tc < 50_000.0, 0.3, 0.6))
    real = [j for j in geo.jumps if not j.kick and not j.at_edge]
    assert len(real) == 1
    assert real[0].time_yr == pytest.approx(50_000.0, abs=500.0)
    assert real[0].dh == pytest.approx(0.3, abs=0.05)
    assert 49_500.0 < geo.regime_start_yr < 51_000.0
    w0, w1 = geo.window
    assert geo.regime_start_yr - 1e-9 <= w0 < w1


# ----------------------------------------------------------------------------------------------------------------------
# F4. Domain validation before the analysis: invalid with a reason, never a crash
# ----------------------------------------------------------------------------------------------------------------------

N_ROWS = 3000


def _series(spec=None, a=None, e=None, inc=None, planets=None, angle=None):
    t = np.arange(N_ROWS, dtype=float)
    spec = spec or ResonanceSpec.from_key('2J-1')
    a = np.full(N_ROWS, 3.30) if a is None else a
    e = np.full(N_ROWS, 0.1) if e is None else e
    inc = np.full(N_ROWS, 0.05) if inc is None else inc
    angle = np.mod(0.1 * t, 2 * np.pi) if angle is None else angle
    return MMRSeries(
        t_yr=t,
        a=a,
        e=e,
        inc=inc,
        Omega=np.zeros(N_ROWS),
        omega=np.zeros(N_ROWS),
        varpi=np.zeros(N_ROWS),
        angle_wrapped=angle,
        resonance=spec,
        regime=Regime(step_yr=1.0, span_yr=N_ROWS - 1.0, n_rows=N_ROWS),
        planets=planets or {},
        name='synthetic',
    )


def _with(arr_name, index, value):
    arr = np.full(N_ROWS, {'a': 3.30, 'e': 0.1, 'inc': 0.05}[arr_name])
    arr[index] = value
    return {arr_name: arr}


@pytest.mark.parametrize(
    "kwargs,problem",
    [
        (_with('a', 1500, -3.3), 'a<=0'),
        (_with('a', 10, 0.0), 'a<=0'),
        (_with('e', 700, -0.01), 'e<0'),
        (_with('e', 700, 1.0), 'e>=1'),
        (_with('inc', 5, np.nan), 'non_finite'),
    ],
)
def test_unphysical_records_are_invalid_not_crashes(kwargs, problem):
    series = _series(**kwargs)
    problems = series.domain_problems()
    assert any(p.startswith(problem) for p in problems), problems
    result = analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False)
    assert str(result.verdict.status) == 'invalid'
    assert str(result.verdict.reason) == 'unphysical'


def test_angle_without_body_longitude_is_invalid_input():
    spec = ResonanceSpec(
        key='1J+0-1',
        kind=ResonanceKind.TWO_BODY,
        planets=('Jupiter',),
        planet_coefficients=(1,),
        body_coefficient=0,
        body_varpi_coefficient=-1,
        planet_varpi_coefficients=(0,),
    )
    series = _series(spec=spec)
    assert 'k=0' in series.domain_problems()
    result = analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False)
    assert str(result.verdict.status) == 'invalid'
    assert str(result.verdict.reason) == 'invalid_input'


def test_circular_orbit_is_not_a_crash():
    # e = 0 is a valid (circular) elliptic orbit: no domain problem, the analysis must run through.
    series = _series(e=np.zeros(N_ROWS))
    assert series.domain_problems() == []
    result = analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False)
    assert str(result.verdict.status) in ('resonant', 'transient', 'near_separatrix', 'non_resonant', 'uncertain')


def _jupiter(t):
    n = t.size
    return PlanetSeries(
        name='Jupiter',
        t_yr=t,
        a=np.full(n, 5.2),
        e=np.full(n, 0.048),
        inc=np.full(n, 0.02),
        Omega=np.zeros(n),
        varpi=np.mod(1e-4 * t, 2 * np.pi),
        mean_longitude=np.mod(0.5297 * t, 2 * np.pi),
    )


def test_planet_grid_shifted_by_half_a_step_is_refused():
    t = np.arange(N_ROWS, dtype=float)
    _series(planets={'Jupiter': _jupiter(t)}).validate()  # same grid: fine
    shifted = _series(planets={'Jupiter': _jupiter(t + 0.5)})
    with pytest.raises(ValueError):
        shifted.validate()
    with pytest.raises(ValueError):
        analyse_mmr(shifted, SETTINGS, scan_subresonances=False, scan_owner=False)
