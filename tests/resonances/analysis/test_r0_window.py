"""R0 with a, window by window (W28 B3; cards g07-1, g08-1): V_F without a free constant, the comparison on the visited
arc, and only the windows where sigma swings >= mmr_a_response_min_amp_deg (or turns) counted.

The expected potentials are analytic: the series are integrated from sigma_ddot = -V'(sigma) + C with V known, so the
true V on the arc is V(sigma) - C sigma, independent of any fit."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.signal import argrelextrema

from resonances.resonance.analysis import pendulum
from resonances.resonance.analysis.mmr import zone
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue
from tests.resonances.analysis.builders import make_geometry, make_turning_points
from tests.resonances.analysis.oracles import fixed_frame

S = AnalysisSettings()
OMEGA2 = (2 * np.pi / 1000.0) ** 2  # a 1000-yr small period [rad^2/yr^2]
C_IDENTITY = -0.7  # sigma_dot = c (a - a*)
A2, PHASE, TILT = 0.3 * OMEGA2, 0.5, 0.3 * OMEGA2  # V = -W^2 cos x + A2 cos(2x + PHASE), x = sigma - theta; force + TILT


def _v_prime(sigma, theta):
    x = sigma - theta
    return OMEGA2 * np.sin(x) - 2 * A2 * np.sin(2 * x + PHASE)


def _v_tilted(sigma, theta):
    x = sigma - theta
    return -OMEGA2 * np.cos(x) + A2 * np.cos(2 * x + PHASE) - TILT * sigma


def _one_cycle(theta, start, tilt, periods=2.2, n=4000):
    """(t, sigma, sigma_dot, a_dot, window) of sigma_ddot = -V'(sigma) + tilt from rest at `start`; the window is the
    first full cycle (rest at a minimum of sigma to the next minimum)."""
    t = np.linspace(0.0, periods * 2 * np.pi / np.sqrt(OMEGA2), n)
    sol = solve_ivp(lambda _, y: [y[1], -_v_prime(y[0], theta) + tilt], (0.0, t[-1]), [start, 0.0], t_eval=t, rtol=1e-11, atol=1e-14)
    sigma, rate = sol.y
    a_dot = (-_v_prime(sigma, theta) + tilt) / C_IDENTITY  # c a_dot = sigma_ddot
    end = t[argrelextrema(sigma, np.less)[0][0]]
    return t, sigma, rate, a_dot, (0.0, float(end))


def test_asymmetric_tilted_well_agrees_on_its_arc_only_without_the_free_constant():
    # bottom of V - TILT sigma at 276 deg (137295 1N-2's well): V'(b) = TILT, solved for x = b - theta
    x_bottom = brentq(lambda x: OMEGA2 * np.sin(x) - 2 * A2 * np.sin(2 * x + PHASE) - TILT, -1.0, 1.5)
    bottom = np.radians(276.0)
    theta = bottom - x_bottom
    t, sigma, rate, a_dot, window = _one_cycle(theta, bottom - np.radians(50.0), TILT)
    comparison = zone._window_correlation(t, sigma, rate, a_dot, window, None, C_IDENTITY, S)
    assert np.degrees(comparison.arc_rad) < 120.0  # 90 deg visited
    assert comparison.r_reference < 0.9  # constant fitted and dropped, full circle: 0.74
    assert comparison.r >= 0.9  # no constant, the visited arc: 0.95
    # V_F (no constant) on the arc is the analytic tilted potential
    inside = t <= window[1]
    v_f = pendulum.fit_potential_from_force(sigma[inside], a_dot[inside], C_IDENTITY, 2)
    arc = np.linspace(sigma[inside].min(), sigma[inside].max(), 200)
    assert np.corrcoef(v_f.value(arc), _v_tilted(arc, theta))[0, 1] > 0.999


def test_held_out_prediction_admits_the_second_harmonic_of_an_asymmetric_well():
    # untilted asymmetric well, one cycle of 81 deg: one harmonic explains 94 % in sample (< the 0.2 gain), but the
    # two-harmonic V predicts the unseen half of the cycle better
    grid = np.linspace(-np.pi, np.pi, 20001)
    bottom = grid[np.argmin(-OMEGA2 * np.cos(grid) + A2 * np.cos(2 * grid + PHASE))]
    t, sigma, rate, a_dot, window = _one_cycle(0.0, bottom - np.radians(50.0), 0.0, periods=2.5, n=5000)
    in_sample, _ = pendulum.fit_potential_from_rate(t, sigma, rate, [window], S)
    held_out = pendulum.fit_window_potential(t, sigma, rate, window, S)
    assert in_sample.n_harmonics == 1 and held_out.n_harmonics == 2
    assert np.asarray(held_out.cos_coeffs) == pytest.approx([-OMEGA2, A2 * np.cos(PHASE)], rel=1e-3)
    assert np.asarray(held_out.sin_coeffs) == pytest.approx([0.0, -A2 * np.sin(PHASE)], abs=1e-3 * OMEGA2)
    comparison = zone._window_correlation(t, sigma, rate, a_dot, window, None, C_IDENTITY, S)
    assert comparison.r > 0.999 and comparison.r_reference < 0.9


def test_moving_well_needs_its_own_acceleration_in_the_force():
    # phi = sigma - s(t), s = alpha t^2 / 2: phi'' = -W^2 sin phi - alpha while c a_dot = sigma'' = -W^2 sin phi
    alpha = 0.2 * OMEGA2
    t = np.linspace(0.0, 1.2 * 2 * np.pi / np.sqrt(OMEGA2), 3000)
    sol = solve_ivp(
        lambda _, y: [y[1], -OMEGA2 * np.sin(y[0]) - alpha], (0.0, t[-1]), [np.radians(-60.0), 0.0], t_eval=t, rtol=1e-11, atol=1e-14
    )
    phi, phi_rate = sol.y
    shift = 0.5 * alpha * t**2
    frame = pendulum.ModulatedFrame(t, np.ones_like(t), shift, alpha * t)
    acceleration = pendulum.frame_acceleration(frame, t)
    assert acceleration == pytest.approx(np.full_like(t, alpha), rel=1e-9)
    assert pendulum.frame_acceleration(None, t) is None
    assert pendulum.frame_acceleration(fixed_frame(t), t) is None
    a_dot = -OMEGA2 * np.sin(phi) / C_IDENTITY
    window = (0.0, float(t[argrelextrema(phi, np.less)[0][0]]))
    args = (t, phi + shift, phi_rate + alpha * t, a_dot, window, frame, C_IDENTITY, S, True)
    assert zone._window_correlation(*args, acceleration).r > 0.99  # 0.997
    assert zone._window_correlation(*args, None).r < 0.9  # 0.72: the frame's inertial term is missing from V_F


def _geometry(amplitudes_deg, period=100.0):
    """Turning points of a libration about 0: cycle k runs from +A_k through -A_k to +A_k+1 (its second half-swing is
    (A_k + A_k+1) / 2, so a cycle next to a larger one is wider)."""
    times, values = [], []
    for k, amplitude in enumerate(amplitudes_deg):
        times += [2 * k * period / 2, (2 * k + 1) * period / 2]
        values += [np.radians(amplitude), -np.radians(amplitude)]
    times.append(len(amplitudes_deg) * period)
    values.append(np.radians(amplitudes_deg[-1]))
    return make_geometry(turning_points=make_turning_points(times, values)), period


def test_window_eligibility_reads_the_swing_inside_each_window():
    geometry, period = _geometry([25.0, 25.0, 80.0, 80.0])
    windows = [(k * period, (k + 1) * period) for k in range(4)]
    # half-swings per window: (25, 25), (25, 52.5), (80, 80), (80, 80) deg against 45 deg
    assert [zone.window_eligible(geometry, w, False, S) for w in windows] == [False, True, True, True]
    assert zone.window_eligible(geometry, windows[0], True, S)  # a turn always counts
    assert not zone.window_eligible(make_geometry(), (0.0, 1.0), False, S)  # no turning points: no swing measured


def test_mixed_amplitudes_pass_on_the_eligible_windows():
    # BG41-like: r 0.64-0.90 where sigma swings +-20-30 deg, 0.98-1.00 at +-60-108 deg
    geometry, period = _geometry([25.0] * 4 + [80.0] * 4)
    comparisons = []
    for k in range(8):
        window = (k * period, (k + 1) * period)
        r = 0.75 if k < 3 else 0.98  # the three +-25-deg cycles disagree
        comparisons.append(zone.WindowComparison(r, zone.window_eligible(geometry, window, False, S), r_reference=r, vf_law=1.0))
    assert [c.eligible for c in comparisons] == [False] * 3 + [True] * 5
    check, fired = zone.r0_with_a(CheckValue.PASS, comparisons, S)
    assert check == CheckValue.PASS  # 5 of 5 eligible windows agree
    assert pendulum.with_slow_check(CheckValue.PASS, [c.r for c in comparisons], S) == CheckValue.FAIL  # 5 of 8 < 0.666
    assert fired == frozenset({Rule.R0_WINDOWS})


def test_kinematic_swing_still_fails_sigma_only():
    # period law 0.61 (a kinematic swing): R0 sigma-only fails whatever V_E ~ V_F says, as before W28
    assert pendulum.sigma_only_check(0.61, 0.95, S) == CheckValue.FAIL
    comparisons = [zone.WindowComparison(1.0, True, r_reference=1.0)] * 4
    assert zone.r0_with_a(CheckValue.FAIL, comparisons, S) == (CheckValue.FAIL, frozenset())


def test_no_eligible_window_is_not_measured():
    comparisons = [zone.WindowComparison(0.99, False, r_reference=0.99)] * 3
    check, fired = zone.r0_with_a(CheckValue.PASS, comparisons, S)
    assert check == CheckValue.NOT_MEASURED  # never pass
    assert fired == frozenset({Rule.R0_WINDOWS})
    assert np.isnan(pendulum.agree_fraction([0.99] * 3, S, [False] * 3))


def test_both_parts_fire_when_together_they_change_the_check():
    # 2004 BG41: pre-W28 on every window fails (0.63), each part alone passes
    comparisons = [zone.WindowComparison(0.95, k % 3 == 0, r_reference=0.95 if k % 3 == 0 else 0.5, vf_law=1.0) for k in range(9)]
    check, fired = zone.r0_with_a(CheckValue.PASS, comparisons, S)
    assert check == CheckValue.PASS
    assert fired == frozenset({Rule.R0_WINDOWS, Rule.POTENTIAL_ON_ARC})


def test_pre_w28_fit_is_kept_behind_its_setting():
    # with a constant tilt the free constant takes it (and is dropped); without it the tilt stays in V_F
    t, sigma, rate, a_dot, window = _one_cycle(0.0, -np.radians(50.0), TILT)
    inside = t <= window[1]
    free = pendulum.fit_potential_from_force(sigma[inside], a_dot[inside], C_IDENTITY, 1, free_constant=True)
    none = pendulum.fit_potential_from_force(sigma[inside], a_dot[inside], C_IDENTITY, 1)
    assert free.r2 > none.r2  # one more parameter
    legacy = AnalysisSettings(separatrix_vf_free_constant=True, separatrix_vevf_arc=False)
    comparison = zone._window_correlation(t, sigma, rate, a_dot, window, None, C_IDENTITY, legacy)
    assert np.isfinite(comparison.r) and np.isfinite(comparison.r_reference)
