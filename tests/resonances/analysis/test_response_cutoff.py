"""The response cutoff max(P_c, MMR_RESPONSE_CUTOFF_CYCLES x P_cyc) on exact pendulum records (P_cyc = 400 yr, P_c = 20 yr).

Expected numbers come from outside the package: the Butterworth gain of a filtfilt pass, 1/(1 + (P_c/P)^4) for order 2;
the force-law R^2 of a pendulum with a forced wave, var(resonant a_dot) / (var(resonant) + var(forced wave left by the
filters)); the energy of the derivative of a smoothed unit step, which scales as 1/cutoff.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import ellipk

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.mmr import response as r
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue

SETTINGS = settings_from_config()
NARROW = settings_from_config(mmr_response_cutoff_cycles=0.05)  # the old default
P_CYC = 400.0
P_C = 20.0
STEP = 1.0
C = -1.0  # sigma_dot = c (a - a*); the sign does not enter R^2
FORCED_PERIOD = 30.0  # a forced line between the two response cutoffs (20 and 40 yr)


def butterworth_gain(period, cutoff):
    return 1 / (1 + (cutoff / period) ** (2 * SETTINGS.filter_order))


def omega0(amplitude):
    """omega0 of the pendulum whose period at this amplitude is P_CYC: P = 4 K(sin^2(A/2)) / omega0."""
    return 4 * ellipk(np.sin(amplitude / 2) ** 2) / P_CYC


def pendulum(amplitude, t, start=0.0):
    """sigma, sigma_dot of the pendulum released from the turning point sigma = -A at t = start (both directions)."""
    w0 = omega0(amplitude)
    rhs = lambda _, y: [y[1], -(w0**2) * np.sin(y[0])]  # noqa: E731
    after, before = t >= start, t < start
    sigma, rate = np.empty_like(t), np.empty_like(t)
    fwd = solve_ivp(rhs, (start, t[-1]), [-amplitude, 0.0], t_eval=t[after], rtol=1e-11, atol=1e-12)
    sigma[after], rate[after] = fwd.y
    if before.any():  # u(tau) = sigma(start - tau) obeys the same equation with the rate reversed
        back = solve_ivp(rhs, (0.0, start - t[0]), [-amplitude, 0.0], t_eval=(start - t[before])[::-1], rtol=1e-11, atol=1e-12)
        sigma[before], rate[before] = back.y[0][::-1], -back.y[1][::-1]
    return sigma, rate


def force_law(t, sigma, a, amplitude, settings, piece):
    """The package's L4 path: low-pass at P_c (preprocessing), again at the response cutoff, force windows on `piece`."""
    angle_f = filtering.lowpass(t, sigma, P_C, settings.filter_order)
    a_f = filtering.lowpass(t, a, P_C, settings.filter_order)
    cutoff = r.response_cutoff_yr(P_C, P_CYC, settings)
    if cutoff > P_C:
        angle_f = filtering.lowpass(t, angle_f, cutoff, settings.filter_order)
        a_f = filtering.lowpass(t, a_f, cutoff, settings.filter_order)
    inside = r.samples_between(t, *piece)
    window = settings.mmr_force_window_cycles * P_CYC
    windows = r.force_windows(t[inside], angle_f[inside], filtering.time_derivative(t, a_f)[inside], window, settings.mmr_force_window_step)
    check, _, _ = r.a_response_check(windows, None, amplitude, settings)
    return check, float(np.median([w.r2 for w in windows]))


def test_default_cutoff_keeps_the_fifth_harmonic():
    assert r.response_cutoff_yr(P_C, P_CYC, SETTINGS) == pytest.approx(40.0)  # 0.1 x 400 yr
    assert r.response_cutoff_yr(P_C, 150.0, SETTINGS) == P_C  # 0.1 P_cyc below P_c: P_c
    t = np.arange(0.0, 8000.0, STEP)
    fifth = np.sin(2 * np.pi * t / (P_CYC / 5))
    kept = filtering.lowpass(t, fifth, r.response_cutoff_yr(P_C, P_CYC, SETTINGS), SETTINGS.filter_order, detrend=False)
    middle = slice(1000, -1000)
    assert np.std(kept[middle]) / np.std(fifth[middle]) == pytest.approx(butterworth_gain(80.0, 40.0), abs=2e-3)  # 0.941


@pytest.mark.parametrize('share', [0.2, 0.3])
def test_a_forced_line_between_the_cutoffs_fails_the_law_at_005_not_at_01(share):
    # a pendulum (A = 90 deg) with a 30-yr wave in a (and the matching wave in sigma, sigma_dot = c delta a) of `share`
    # times the libration's a amplitude: the wave's a_dot is left by G(30; 20) at 0.05 and by G(30; 20) G(30; 40) at 0.1
    amplitude = np.radians(90.0)
    t = np.arange(0.0, 12 * P_CYC + STEP, STEP)
    free, rate = pendulum(amplitude, t)
    w0 = omega0(amplitude)
    eps = share * 2 * w0 * np.sin(amplitude / 2) / abs(C)
    a = rate / C + eps * np.sin(2 * np.pi * t / FORCED_PERIOD)
    sigma = free - C * eps * FORCED_PERIOD / (2 * np.pi) * np.cos(2 * np.pi * t / FORCED_PERIOD)
    resonant_var = np.var(w0**2 * np.sin(free[t < P_CYC]) / C)  # a_dot = sigma_ddot / c over one cycle
    piece = (P_CYC, t[-1] - P_CYC)
    expected = {}
    for name, gain in (('0.05', butterworth_gain(30, 20)), ('0.1', butterworth_gain(30, 20) * butterworth_gain(30, 40))):
        forced_var = (eps * 2 * np.pi / FORCED_PERIOD * gain) ** 2 / 2
        expected[name] = resonant_var / (resonant_var + forced_var)
    narrow, narrow_r2 = force_law(t, sigma, a, amplitude, NARROW, piece)
    wide, wide_r2 = force_law(t, sigma, a, amplitude, SETTINGS, piece)
    assert narrow_r2 == pytest.approx(expected['0.05'], abs=0.02) and narrow == CheckValue.FAIL  # 0.16, 0.08
    assert wide_r2 == pytest.approx(expected['0.1'], abs=0.02) and wide == CheckValue.PASS  # 0.76, 0.59


@pytest.mark.parametrize('amplitude_deg, loss_max', [(90.0, 1e-4), (175.0, 0.005), (178.0, 0.01)])
def test_the_wider_cutoff_costs_an_exact_pendulum_little(amplitude_deg, loss_max):
    amplitude = np.radians(amplitude_deg)
    t = np.arange(0.0, 12 * P_CYC + STEP, STEP)
    sigma, rate = pendulum(amplitude, t)
    piece = (P_CYC, t[-1] - P_CYC)
    _, narrow_r2 = force_law(t, sigma, rate / C, amplitude, NARROW, piece)
    check, wide_r2 = force_law(t, sigma, rate / C, amplitude, SETTINGS, piece)
    assert check == CheckValue.PASS and narrow_r2 > 0.9995 and 0 <= narrow_r2 - wide_r2 < loss_max


def test_a_smoothed_jump_leaks_less_energy_into_a_dot_at_the_wider_cutoff():
    # the derivative of a unit step smoothed at P has width ~P and height ~1/P: its energy halves when P doubles
    t = np.arange(0.0, 4000.0, STEP)
    step = (t >= 2000.0).astype(float)
    energy = [
        np.sum(filtering.time_derivative(t, filtering.lowpass(t, step, p, SETTINGS.filter_order, detrend=False)) ** 2) for p in (20.0, 40.0)
    ]
    assert energy[0] / energy[1] == pytest.approx(2.0, rel=0.02)


@pytest.mark.parametrize('offset_cycles, both_clean', [(0.0, False), (0.05, True)])
def test_a_capture_after_a_jump_of_a(offset_cycles, both_clean):
    # 6 libration cycles (A = 120 deg) between circulations at 1.5 x the peak libration rate; a jumps at
    # `offset_cycles` P_cyc before the first and after the last turning point. At the turning point the jump's a_dot
    # leaks into the first and last force windows at either cutoff, less at 0.1; 20 yr away (one old cutoff, half the
    # new one) it does not.
    amplitude = np.radians(120.0)
    n_cycles, offset = 6, offset_cycles * P_CYC
    t = np.arange(-3000.0 - offset, n_cycles * P_CYC + offset + 3000.0 + STEP / 2, STEP)
    sigma, rate = pendulum(amplitude, t)
    circulation = 1.5 * 2 * omega0(amplitude) * np.sin(amplitude / 2)
    i0, i1 = np.searchsorted(t, -offset), np.searchsorted(t, n_cycles * P_CYC + offset)
    before, after = np.arange(len(t)) < i0, np.arange(len(t)) > i1
    sigma[before], rate[before] = sigma[i0] + circulation * (t[before] - t[i0]), circulation
    sigma[after], rate[after] = sigma[i1] - circulation * (t[after] - t[i1]), -circulation
    piece = (0.0, n_cycles * P_CYC)
    narrow, narrow_r2 = force_law(t, sigma, rate / C, amplitude, NARROW, piece)
    wide, wide_r2 = force_law(t, sigma, rate / C, amplitude, SETTINGS, piece)
    assert narrow == wide == CheckValue.PASS
    if both_clean:
        assert min(narrow_r2, wide_r2) > 0.99
    else:
        assert narrow_r2 < wide_r2 < 0.9
