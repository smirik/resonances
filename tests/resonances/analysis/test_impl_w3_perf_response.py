"""W3 'perf' package, response.py: closed-form windowed sine fits (W2 time tail), the phase bound for sum eps >= 1
(W3-G6), B1 in the libration band (W3-G11: the first harmonics and all others in [P/f, fP]) and B5 without the mu e^q
route (W3-F3).

Expected numbers are computed independently: per-window np.linalg.lstsq, the geometry of the disk 1 + z, and series
built from known harmonics.
"""

import math
import time

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import response
from resonances.resonance.analysis.mmr.models import ForceWindow
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

NAN = float('nan')
SETTINGS = AnalysisSettings()


def _lstsq_window(angle, target):
    """(A, B, C, R^2) of target = A sin + B cos + C by np.linalg.lstsq (the W2 per-window fit)."""
    keep = np.isfinite(angle) & np.isfinite(target)
    angle, target = angle[keep], target[keep]
    design = np.column_stack([np.sin(angle), np.cos(angle), np.ones_like(angle)])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    total = np.sum((target - target.mean()) ** 2)
    return coef[0], coef[1], coef[2], 1 - np.sum(residual**2) / total


# --------------------------------------------------------------------------------------------- windowed fits


def test_windowed_fits_equal_per_window_lstsq():
    rng = np.random.default_rng(7)
    t = np.arange(0.0, 5000.0, 1.0)
    angle = 0.004 * t + 1.3 * np.sin(t / 90.0) + 0.05 * rng.standard_normal(len(t))
    target = 2e-6 * np.sin(angle - 0.4) + 3e-7 + 4e-7 * rng.standard_normal(len(t)) + 1e-3  # a large offset too
    target[[100, 2500, 2501]] = NAN  # non-finite samples are left out of their windows
    windows = response.sliding_windows(t, 700.0, 0.5)
    a_sin, b_cos, offset, r2 = response.sine_fits_in_windows(t, angle, target, windows)
    for i, (start, end) in enumerate(windows):
        inside = (t >= start) & (t <= end)
        expected = _lstsq_window(angle[inside], target[inside])
        assert (a_sin[i], b_cos[i], offset[i], r2[i]) == pytest.approx(expected, rel=1e-7, abs=1e-12)


def test_force_windows_by_hand():
    # a_dot = 2 sin(sigma) - 1 cos(sigma) + 0.5 = sqrt(5) sin(sigma - phi0) + 0.5 with phi0 = atan2(1, 2)
    t = np.arange(0.0, 2000.0, 0.5)
    sigma = 0.01 * t + 0.3 * np.sin(t / 40.0)
    windows = response.force_windows(t, sigma, 2 * np.sin(sigma) - np.cos(sigma) + 0.5, 400.0, 0.5)
    assert len(windows) == 8  # starts 0, 200, ..., 1400: a window ends by the last sample 1999.5 + half a step
    for w in windows:
        assert w.amplitude_au_yr == pytest.approx(math.sqrt(5), rel=1e-10)
        assert w.phi0_rad == pytest.approx(math.atan2(1, 2), abs=1e-10)
        assert w.offset_au_yr == pytest.approx(0.5, abs=1e-10)
        assert w.r2 == pytest.approx(1.0, abs=1e-10)


def test_degenerate_window_falls_back_to_lstsq():
    # sigma does not move in the first windows: sin, cos and 1 are collinear there, lstsq gives the minimum norm
    t = np.arange(0.0, 1000.0, 1.0)
    sigma = np.where(t < 400, 0.7, 0.7 + 0.02 * (t - 400))
    target = np.sin(sigma) + 0.1 * np.cos(3 * t)
    windows = response.sliding_windows(t, 200.0, 0.5)
    a_sin, b_cos, offset, _ = response.sine_fits_in_windows(t, sigma, target, windows)
    expected = _lstsq_window(sigma[:201], target[:201])
    assert (a_sin[0], b_cos[0], offset[0]) == pytest.approx(expected[:3], rel=1e-6, abs=1e-9)
    assert np.all(np.isfinite(a_sin))


def test_too_few_samples_give_no_window():
    t = np.arange(0.0, 10.0, 1.0)
    a_sin, _, _, r2 = response.sine_fits_in_windows(t, t, t, [(0.0, 2.0), (3.0, 9.0)])
    assert np.isnan(a_sin[0]) and np.isnan(r2[0])  # 3 samples < MIN_FIT_SAMPLES
    assert np.isfinite(a_sin[1])


def test_windowed_fits_are_fast():
    t = np.arange(0.0, 200000.0, 1.0)
    angle = 0.003 * t
    target = np.sin(angle)
    windows = response.sliding_windows(t, 1000.0, 0.5)  # starts 0, 500, ..., 198500 over 2e5 samples
    began = time.perf_counter()
    response.force_windows(t, angle, target, 1000.0, 0.5)
    assert len(windows) == 398
    assert time.perf_counter() - began < 1.0  # the per-window lstsq took ~1 s here and grew with the window count


# --------------------------------------------------------------------------------------------- G6


def test_phase_bound_is_undefined_from_sum_eps_one():
    floor = math.radians(30)
    # arcsin(0.4 + 0.3) + 2 x 0.05 = 0.775397 + 0.1
    assert response.phase_tolerance((0.4, 0.3), 0.05, floor) == pytest.approx(0.875397, abs=1e-6)
    assert np.isnan(response.phase_tolerance((0.5, 0.5), 0.1, floor))
    assert np.isnan(response.phase_tolerance((0.8, 0.7), 0.05, floor))
    assert response.phase_tolerance((0.99,), 0.0, floor) == pytest.approx(math.asin(0.99))


def test_disk_geometry_behind_the_bound():
    """Largest |arg(1 + z)| over |z| <= r: arcsin(r) for r < 1 (tangent from the origin), pi once r > 1 (Codex astra:
    1 + 1.5 e^{i pi} = -0.5). The clipped arcsin(min(1, r)) = pi/2 is wrong there."""
    phases = np.linspace(-np.pi, np.pi, 20001)
    radii = np.linspace(0, 1, 201)[:, None]
    for r in (0.3, 0.7, 1.5):
        z = r * radii * np.exp(1j * phases)
        largest = float(np.max(np.abs(np.angle(1 + z))))
        if r < 1:
            assert largest == pytest.approx(math.asin(r), abs=1e-3)
        else:
            assert largest == pytest.approx(math.pi, abs=1e-3)
    assert np.angle(1 + 1.5 * np.exp(1j * np.pi)) == pytest.approx(np.pi)


def _windows(r2, phases):
    return [ForceWindow(1000.0 * i, 1000.0 * (i + 1), r2, 1e-6, p, 0.0) for i, p in enumerate(phases)]


def test_the_single_law_is_judged_by_r2_alone():
    """W7 'phi0' (variant D): the phase of the single law decides nothing, bounded (sum eps < 1) or not; R^2 does."""
    big = math.radians(90)
    spun = _windows(0.95, [0.0, 2.0, -2.0, 1.0])  # the phase wanders, R^2 is high
    assert response.a_response_check(spun, None, big, SETTINGS) == (CheckValue.PASS, False, 'single')
    # a law that fails R^2 fails whatever the phase
    weak = _windows(0.1, [0.0, 0.0, 0.0, 0.0])
    assert response.a_response_check(weak, None, big, SETTINGS)[0] == CheckValue.FAIL


# --------------------------------------------------------------------------------------------- G11


def _harmonic_record(period, c, a1=1.0, a3=0.25, cycles=20):
    """sigma = a1 sin(w t) + a3 sin(3 w t) and a from the identity sigma_dot = c (a - a*), a* = 2.5."""
    omega = 2 * math.pi / period
    t = np.arange(0.0, cycles * period, 1.0)
    sigma = a1 * np.sin(omega * t) + a3 * np.sin(3 * omega * t)
    a = 2.5 + (a1 * omega * np.cos(omega * t) + 3 * a3 * omega * np.cos(3 * omega * t)) / c
    return t, sigma, a


def test_b1_in_the_band_follows_the_identity():
    """In [P/2, 2P] only the first harmonic is left: a_band = sigma_dot_band / c, ratio 1, and the required wave is the
    band gain at P times w A1 / |c| (gain from the digital Butterworth formula, independent of the filter code path)."""
    period, c = 500.0, -0.8
    t, sigma, a = _harmonic_record(period, c)
    window = (t[0], t[-1])
    required, measured, ratio = response.b1_amplitude(t, a, period, 0.89, c, window, 3.5, sigma, SETTINGS)
    assert ratio == pytest.approx(1.0, abs=2e-3)
    f, order = SETTINGS.mmr_b1_band_factor, SETTINGS.filter_order
    gain = 1 / (1 + (math.tan(math.pi / period) / math.tan(math.pi * f / period)) ** (2 * order)) - 1 / (
        1 + (math.tan(math.pi / period) / math.tan(math.pi / (f * period))) ** (2 * order)
    )
    assert required == pytest.approx(gain * (2 * math.pi / period) / abs(c), rel=0.01)


def test_b1_half_amplitude_approximation_is_off_for_a_non_harmonic_libration():
    """The W2 formula (no sigma given): w x half-amplitude / |c|. The turning points of sin x + 0.25 sin 3x are at
    cos x (3 cos^2 x - 1.25) = 0: sin x = sqrt(7/12) = 0.763763, sin 3x = 0.509175, half = 0.891057, so the ratio
    is 1 / 0.891 = 1.122 although a follows the identity exactly."""
    period, c = 500.0, -0.8
    t, sigma, a = _harmonic_record(period, c)
    half = float(np.max(sigma))
    assert half == pytest.approx(0.891057, abs=1e-5)
    _, _, ratio = response.b1_amplitude(t, a, period, half, c, (t[0], t[-1]), 3.5)
    assert ratio == pytest.approx(1 / half, rel=5e-3)


def test_b1_in_the_band_with_a_wandering_period():
    """sigma = sin(phi(t)) with the period drifting 400 -> 700 yr, a from the identity. A fixed frequency 2 pi / P_med
    dephases (the W2 fit); the band [P/2, 2P] holds the whole wandering line: ratio 1."""
    c = 0.6
    t = np.arange(0.0, 20000.0, 1.0)
    period = 400.0 + 300.0 * t / t[-1]
    phase = np.cumsum(2 * math.pi / period)
    sigma = 0.8 * np.sin(phase)
    a = 1.7 + 0.8 * np.cos(phase) * (2 * math.pi / period) / c
    median_period = float(np.median(period))
    _, _, band_ratio = response.b1_amplitude(t, a, median_period, 0.8, c, (t[0], t[-1]), 3.5, sigma, SETTINGS)
    assert band_ratio == pytest.approx(1.0, abs=0.03)


def test_b1_of_a_kinematic_oscillation_stays_small():
    # sigma oscillates (in its varpi part), a does not: no wave of a in the band
    period, c = 400.0, 0.6
    t = np.arange(0.0, 10 * period, 1.0)
    sigma = 0.8 * np.sin(2 * math.pi * t / period)
    a = 1.7 + 1e-9 * np.sin(t / 3.0)
    _, _, ratio = response.b1_amplitude(t, a, period, 0.8, c, (t[0], t[-1]), 3.5, sigma, SETTINGS)
    assert ratio < SETTINGS.mmr_b1_ratio_min


def test_b1_without_a_band_compares_first_harmonics():
    """2000 yr of a 500-yr cycle: the band's edge trims (f P each side) leave nothing, so the first harmonics at
    w = 2 pi / P are compared (a1 = w sigma1 / |c|; over 4 periods sigma1 = A1 = 1 up to the line term)."""
    period, c = 500.0, -0.8
    t, sigma, a = _harmonic_record(period, c, cycles=4)
    window = (t[0], t[-1])
    assert all(math.isnan(x) for x in response.b1_in_band(t, a, sigma, period, c, window, SETTINGS))
    required, _, ratio = response.b1_amplitude(t, a, period, 0.89, c, window, 3.5, sigma, SETTINGS)
    assert required == pytest.approx((2 * math.pi / period) / abs(c), rel=5e-3)  # the line term takes a little
    assert ratio == pytest.approx(1.0, abs=5e-3)


# --------------------------------------------------------------------------------------------- F3


def test_b5_has_no_theory_route():
    """|nu_c| = 1e-3 rad/yr is 10 x 2 omega0_th, yet omega0_th from mu e^q is no upper bound: not excluded."""
    bound = {'centre_rate': 1e-3, 'centre_rate_err': 1e-5, 'ratio': 0.3}
    assert response.slow_libration_check(bound, 5e-5, NAN, NAN, SETTINGS) == (CheckValue.FAIL, 'none', False)
    assert response.slow_libration_check(bound, 5e-5, 0.001, 0.01, SETTINGS) == (CheckValue.PASS, 'owner', False)
    bound['ratio'] = 1.2
    assert response.slow_libration_check(bound, NAN, NAN, NAN, SETTINGS) == (CheckValue.PASS, 'data', False)
