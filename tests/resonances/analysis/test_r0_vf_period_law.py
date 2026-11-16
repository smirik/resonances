"""R0 with a must carry the observed period in V_F (SEPARATRIX_VF_PERIOD_LAW; the author's D10 A, 2026-11-04): the
correlation of V_E and V_F is blind to a positive factor between them, so V_F = k V_E passes the correlation for every
k > 0. The median T_obs/T_pred(V_F) over the eligible compared cycles must lie within SEPARATRIX_PERIOD_TOL of 1.

The expectations are computed here from the complete elliptic integrals of the pendulum V = -omega0^2 cos(phi),
independently of the package's period functions. A libration of amplitude A (m = sin^2(A/2)) has T = 4 K(m) / omega0
and the time mean <cos phi> = 2 E(m)/K(m) - 1. In V_F = k V the energy of the observed swing is
E(t) = phi_dot^2/2 + k omega0^2 (1 - cos phi) = omega0^2 (cos phi - cos A) + k omega0^2 (1 - cos phi); its time mean
over the separatrix energy 2 k omega0^2 is h_F, and T_pred = 4 K(h_F) / (sqrt(k) omega0). A turn of energy
h = E / (2 omega0^2) > 1 has T = 2 K(1/h) / (omega0 sqrt(h))."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
from scipy.special import ellipe, ellipk

from resonances.resonance.analysis.mmr import zone
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue
from tests.resonances.analysis.builders import make_energy
from tests.resonances.analysis.test_r0_episodes import _first_cycles, _r0
from tests.resonances.analysis.impl_response_pendulum_synthetic import pendulum_record

S = AnalysisSettings()
OFF = AnalysisSettings(separatrix_vf_period_law=False)
OMEGA2 = (2 * np.pi / 1000.0) ** 2  # a 1000-yr small period [rad^2/yr^2]
C_IDENTITY = -0.7  # c a_dot = sigma_ddot


def _expected_libration_law(amplitude_rad, k):
    """T_obs / T_pred(k V) of a libration of `amplitude_rad` in V = -omega0^2 cos(phi) (module docstring); nan when
    h_F >= 1: the observed libration lies above the top of k V, whose loop there turns (no libration period)."""
    m = np.sin(amplitude_rad / 2) ** 2
    mean_cos = 2 * ellipe(m) / ellipk(m) - 1
    h_f = ((mean_cos - np.cos(amplitude_rad)) + k * (1 - mean_cos)) / (2 * k)
    if h_f >= 1:
        return float('nan')
    return (4 * ellipk(m) / np.sqrt(OMEGA2)) / (4 * ellipk(h_f) / np.sqrt(k * OMEGA2))


def _libration(amplitude_deg=90.0, periods=3.3, n=20000):
    """(t, sigma, sigma_dot, a_dot, windows) of the pendulum from rest at +A; windows: maximum to maximum."""
    amplitude = np.radians(amplitude_deg)
    period = 4 * ellipk(np.sin(amplitude / 2) ** 2) / np.sqrt(OMEGA2)
    t = np.linspace(0.0, periods * period, n)
    sol = solve_ivp(lambda _, y: [y[1], -OMEGA2 * np.sin(y[0])], (0.0, t[-1]), [amplitude, 0.0], t_eval=t, rtol=1e-12, atol=1e-14)
    sigma, rate = sol.y
    maxima = argrelextrema(sigma, np.greater)[0]
    windows = [(float(t[i]), float(t[j])) for i, j in zip(maxima[:-1], maxima[1:])]
    return t, sigma, rate, -OMEGA2 * np.sin(sigma) / C_IDENTITY, windows


def _comparisons(t, sigma, rate, a_dot, windows, k, settings=S, rotation=False):
    return [zone._window_correlation(t, sigma, rate, k * a_dot, w, None, C_IDENTITY, settings, True, None, rotation) for w in windows]


def test_independent_expectations():
    # hand values at A = 90 deg: K(1/2) = 1.85407, E(1/2) = 1.35064, <cos phi> = 0.45694
    assert _expected_libration_law(np.radians(90.0), 1.0) == pytest.approx(1.0, abs=1e-12)
    assert _expected_libration_law(np.radians(90.0), 2.0) == pytest.approx(1.4833, abs=1e-3)
    assert _expected_libration_law(np.radians(90.0), 0.5) == pytest.approx(0.6185, abs=1e-3)


@pytest.mark.parametrize('amplitude_deg', [60.0, 90.0, 120.0])
@pytest.mark.parametrize('k', [1.0, 2.0, 0.5])
def test_window_law_is_the_period_of_the_observed_swing_in_k_v(amplitude_deg, k):
    t, sigma, rate, a_dot, windows = _libration(amplitude_deg)
    # 120 deg in V/2: h_F = 1.06, the swing lies above V_F's top (a topology mismatch: no law)
    expected = _expected_libration_law(np.radians(amplitude_deg), k)
    assert np.isnan(expected) == (amplitude_deg == 120.0 and k == 0.5)
    for comparison in _comparisons(t, sigma, rate, a_dot, windows, k):
        assert comparison.r > 0.999  # the shape is right whatever the scale
        assert comparison.vf_law == pytest.approx(expected, rel=2e-3, nan_ok=True)


def test_right_scale_passes():
    t, sigma, rate, a_dot, windows = _libration()
    comparisons = _comparisons(t, sigma, rate, a_dot, windows, 1.0)
    assert zone.r0_with_a(CheckValue.PASS, comparisons, S) == (CheckValue.PASS, frozenset())
    law, n = zone.vf_period_law(comparisons)
    assert law == pytest.approx(1.0, abs=1e-3) and n == len(windows) == 2


@pytest.mark.parametrize('k', [2.0, 0.5])
def test_wrong_scale_fails_only_with_the_period_law(k):
    t, sigma, rate, a_dot, windows = _libration()
    comparisons = _comparisons(t, sigma, rate, a_dot, windows, k)
    assert zone.r0_with_a(CheckValue.PASS, comparisons, S)[0] == CheckValue.FAIL  # 1.48 / 0.62 against 1 +- 0.2
    assert zone.r0_with_a(CheckValue.PASS, comparisons, OFF)[0] == CheckValue.PASS  # the correlation alone
    assert zone.r0_with_a(CheckValue.PASS, comparisons, S, vf_check=False)[0] == CheckValue.PASS  # the 1:1 route


def test_without_the_law_the_correlation_is_unchanged_and_no_law_is_computed():
    # where the 1:1 energy integral replaces the comparison (vf_check False) the callers skip the period law
    t, sigma, rate, a_dot, windows = _libration()
    with_law = _comparisons(t, sigma, rate, 2.0 * a_dot, windows, 1.0)
    without = [zone._window_correlation(t, sigma, rate, 2.0 * a_dot, w, None, C_IDENTITY, S, True, None, vf_law=False) for w in windows]
    assert [c.r for c in without] == [c.r for c in with_law] and [c.r_reference for c in without] == [c.r_reference for c in with_law]
    assert [c.vf_law for c in with_law] == pytest.approx([1.4833] * 2, rel=2e-3)  # k = 2 (module docstring)
    assert all(np.isnan(c.vf_law) for c in without)
    assert zone.r0_with_a(CheckValue.PASS, without, S, vf_check=False)[0] == CheckValue.PASS


def test_law_inside_the_tolerance_passes():
    # k = 1.25 at 90 deg: law 1.14 (inside 0.2); k = 1.6: 1.31 (outside)
    t, sigma, rate, a_dot, windows = _libration()
    assert _expected_libration_law(np.radians(90.0), 1.25) == pytest.approx(1.141, abs=2e-3)
    assert zone.r0_with_a(CheckValue.PASS, _comparisons(t, sigma, rate, a_dot, windows, 1.25), S)[0] == CheckValue.PASS
    assert _expected_libration_law(np.radians(90.0), 1.6) == pytest.approx(1.312, abs=2e-3)
    assert zone.r0_with_a(CheckValue.PASS, _comparisons(t, sigma, rate, a_dot, windows, 1.6), S)[0] == CheckValue.FAIL


def _rotation(h=1.5, n=20000):
    """(t, sigma, sigma_dot, a_dot, windows) of a turning pendulum of energy h (> 1) in separatrix units; windows: from
    one passage of the bottom to the next (one turn each)."""
    rate0 = 2 * np.sqrt(OMEGA2 * h)  # at phi = 0: phi_dot^2 / 2 = 2 omega0^2 h
    period = 2 * ellipk(1 / h) / (np.sqrt(OMEGA2) * np.sqrt(h))
    t = np.linspace(0.0, 3.2 * period, n)
    sol = solve_ivp(lambda _, y: [y[1], -OMEGA2 * np.sin(y[0])], (0.0, t[-1]), [0.0, rate0], t_eval=t, rtol=1e-12, atol=1e-14)
    sigma, rate = sol.y
    passages = [float(np.interp(2 * np.pi * j, sigma, t)) for j in range(int(sigma[-1] // (2 * np.pi)) + 1)]
    return t, sigma, rate, -OMEGA2 * np.sin(sigma) / C_IDENTITY, list(zip(passages[:-1], passages[1:])), period


def test_a_turn_reads_the_rotation_period_and_a_loop_that_does_not_turn_is_no_law():
    t, sigma, rate, a_dot, windows, period = _rotation()
    assert len(windows) == 3 and np.diff(windows[0])[0] == pytest.approx(period, rel=1e-4)
    comparisons = _comparisons(t, sigma, rate, a_dot, windows, 1.0, rotation=True)
    assert [c.vf_law for c in comparisons] == pytest.approx([1.0] * 3, abs=2e-3)
    assert zone.r0_with_a(CheckValue.PASS, comparisons, S)[0] == CheckValue.PASS
    # V_F ten times deeper: the observed turn lies below V_F's top (h_F <= (3 + 9 x 2) / 20 < 1.05, ~0.7 here), so
    # V_F's loop is a libration: no period of a turn to compare -> not_measured, never a pass
    deep = _comparisons(t, sigma, rate, a_dot, windows, 10.0, rotation=True)
    assert all(c.r > 0.99 for c in deep)
    assert all(np.isnan(c.vf_law) for c in deep)
    assert zone.r0_with_a(CheckValue.PASS, deep, S)[0] == CheckValue.NOT_MEASURED


def test_only_counted_windows_enter_the_median():
    counted = [zone.WindowComparison(0.99, True, vf_law=1.05), zone.WindowComparison(0.98, True, vf_law=0.95)]
    ignored = [
        zone.WindowComparison(0.99, False, vf_law=3.0),  # not eligible
        zone.WindowComparison(float('nan'), True, vf_law=3.0),  # no correlation
        zone.WindowComparison(0.99, True),  # no matched loop
    ]
    assert zone.vf_period_law(counted + ignored) == (pytest.approx(1.0), 2)
    assert zone.vf_period_law(ignored[2:]) == (pytest.approx(float('nan'), nan_ok=True), 0)
    # a failure or a missing comparison is never read again
    assert zone.r0_with_a(CheckValue.FAIL, counted, S)[0] == CheckValue.FAIL
    assert zone.r0_with_a(CheckValue.PASS, [zone.WindowComparison(0.5, True, vf_law=1.0)] * 3, S)[0] == CheckValue.FAIL


@pytest.mark.parametrize('k, passes', [(1.0, True), (2.0, False)])
def test_an_episode_r0_reads_the_law_too(k, passes):
    # pendulum_record: T0 = 300 yr at h = 0.5 (A = 90 deg); a's answer scaled by k around a*
    record = _first_cycles(pendulum_record(), 5)
    a_star = 4.0
    record.prep.a_f = a_star + k * (record.prep.a_f - a_star)
    record.prep.adot_f = np.gradient(record.prep.a_f, record.t)
    found = _r0(record)
    if passes:
        assert found[:3] == (CheckValue.PASS, CheckValue.PASS, 0)
        law, n = found[3]
        assert law == pytest.approx(1.0, abs=0.02) and n == 2
    else:
        assert found is None  # the episode fails: lift-only gives nothing


def test_columns():
    columns = make_energy(vf_period_law_med=1.02, vf_period_law_n=7).to_columns()
    assert columns['sep_VF_period_law_med'] == 1.02 and columns['sep_VF_period_law_n'] == 7
    empty = make_energy().to_columns()
    assert np.isnan(empty['sep_VF_period_law_med']) and empty['sep_VF_period_law_n'] == 0
