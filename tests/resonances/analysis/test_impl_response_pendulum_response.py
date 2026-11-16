"""Response of a, force law, B1, C4, B5 (analysis.mmr.response) on synthetic records with known answers.

Independent truths: a pendulum with the a <-> sigma identity has K = omega0^2/|c|, slope/c = 1, B1 ratio 1; the B5 bound
K_max = |nu_c| da / (1 - cos(S/2)) recovers the force of a slow libration centred on its stable point, and
K_max / (da / T_w) = S / (1 - cos(S/2)) = 3.81 at S = 2.36 rad (SPEC §4.5.8); toy coherences (0, 10, 20 deg) -> 0.99,
(0, 120, 240 deg) -> 0 (NUMBERS §5.4).
"""

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import models as m
from resonances.resonance.analysis.mmr import response as r
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue
from tests.resonances.analysis.impl_response_pendulum_synthetic import kinematic_record, make, pendulum_record

SETTINGS = settings_from_config()
NAN = float('nan')


def test_identity_exact_for_both_signs_of_c():
    a = np.linspace(-1e-3, 1e-3, 200) + 2.5
    for c in (0.8, -0.8):
        ratio, corr, check = r.identity(c * (a - 2.5), a, c, SETTINGS)
        assert ratio == pytest.approx(1.0) and corr == pytest.approx(1.0) and check == CheckValue.PASS
    ratio, _, check = r.identity(0.5 * 0.8 * (a - 2.5), a, 0.8, SETTINGS)  # slope half of c: a varpi share
    assert ratio == pytest.approx(0.5) and check == CheckValue.FAIL
    assert r.identity(np.ones(3), np.ones(3), 0.8, SETTINGS)[2] == CheckValue.NOT_MEASURED


def test_identity_wave_of_a_pure_a_oscillation():
    # sigma - c = k int (n(a) - <n>) dt exactly when a alone drives the wave
    t = np.arange(0.0, 3000.0, 0.5)
    a = 4.0 + 1e-3 * np.sin(2 * np.pi * t / 300.0)
    n = 2 * np.pi * a**-1.5
    wave = -2 * np.concatenate([[0.0], np.cumsum((n[1:] + n[:-1]) / 2 * np.diff(t))])
    wave -= np.polyval(np.polyfit(t, wave, 1), t)
    corr, ratio = r.identity_wave(t, a, -2, wave + 1.0, np.full_like(t, 1.0))
    assert corr == pytest.approx(1.0, abs=1e-9) and ratio == pytest.approx(1.0, abs=1e-6)


def test_force_windows_recover_the_law():
    t = np.arange(0.0, 2000.0, 1.0)
    sigma = 0.3 * t / 50.0
    adot = 2e-6 * np.sin(sigma - 1.2) + 5e-8
    windows = r.force_windows(t, sigma, adot, 400.0, 0.5)
    assert [w.t_start_yr for w in windows] == [200.0 * i for i in range(8)]  # the last must end inside the record
    for w in windows:
        assert w.amplitude_au_yr == pytest.approx(2e-6) and w.phi0_rad == pytest.approx(1.2)
        assert w.offset_au_yr == pytest.approx(5e-8, abs=1e-12) and w.r2 == pytest.approx(1.0)


def test_multiplet_fit_recovers_members():
    t = np.arange(0.0, 20000.0, 1.0)
    sigma = 2 * np.pi * t / 200.0
    offset = 2 * np.pi * t / 5000.0  # varpi - varpi_P
    adot = 1e-6 * np.sin(sigma) + 0.4e-6 * np.cos(sigma - offset)  # members j = 0 and j = -1
    windows = [(5000.0 * i, 5000.0 * (i + 1)) for i in range(4)]
    fit = r.multiplet_fit(t, sigma, adot, offset, (-1, 0), windows, 0.8)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.eps == pytest.approx((0.4,))
    assert fit.coefficient_coherence == pytest.approx(1.0)
    assert fit.coefficients.shape == (4, 2)
    assert fit.coefficients[0] == pytest.approx([0.4e-6j, 1e-6 + 0j], abs=1e-12)


def test_multiplet_harmonics_keep_the_order():
    assert r.multiplet_harmonics(3, 0) == (-3, -2, -1, 0)  # 1N-4+0+3: varpi^3 ... varpi_N^3
    assert r.multiplet_harmonics(-1, 0) == (0, 1)
    assert r.multiplet_harmonics(2, 1) == (-2, -1, 0, 1)  # varpi^2 varpi_P -> ... -> varpi_P^3 / varpi^3
    assert r.multiplet_harmonics(0, 0) == (0,)


def test_phase_tolerance_and_coherence():
    floor = np.radians(30)
    assert r.phase_tolerance((0.5,), 0.0, floor) == pytest.approx(np.radians(30))
    assert np.isnan(r.phase_tolerance((0.5, 0.5), 0.1, floor))  # W3-G6: sum eps >= 1, no bound
    assert r.phase_tolerance((), NAN, floor) == pytest.approx(floor)
    assert r.circular_coherence(np.radians([0, 10, 20])) == pytest.approx(0.99, abs=5e-3)
    assert r.circular_coherence(np.radians([0, 120, 240])) == pytest.approx(0.0, abs=1e-12)


def window(r2, phi0, t0=0.0):
    return m.ForceWindow(t0, t0 + 1.0, r2, 1e-6, phi0, 0.0)


def test_a_response_check_branches():
    big = np.radians(90)
    good = [window(0.9, 0.1, i) for i in range(4)]
    assert r.a_response_check(good, None, big, SETTINGS) == (CheckValue.PASS, False, 'single')
    thin = [window(0.5, 0.0, i) for i in range(4)]
    assert r.a_response_check(thin, None, big, SETTINGS) == (CheckValue.PASS, True, 'single')
    weak = [window(0.3, 0.0, i) for i in range(4)]
    assert r.a_response_check(weak, None, big, SETTINGS)[0] == CheckValue.FAIL
    # a jumping phase with a good R^2 passes: the spread of phi0 is a column only (W7 'phi0', variant D)
    jumping = [window(0.85, phase, i) for i, phase in enumerate(np.radians([0, 80, 0, 80]))]
    assert r.a_response_check(jumping, None, big, SETTINGS) == (CheckValue.PASS, False, 'single')
    # the multiplet law is taken when it gains >= 0.1 in R^2, and then its coefficients must be stable
    multiplet = m.MultipletFit((-1, 0), np.zeros((4, 2), complex), 0.98, 0.95, (0.5,))
    assert r.a_response_check(jumping, multiplet, big, SETTINGS) == (CheckValue.PASS, False, 'multiplet')
    unstable = m.MultipletFit((-1, 0), np.zeros((4, 2), complex), 0.98, 0.5, (0.5,))
    assert r.a_response_check(jumping, unstable, big, SETTINGS)[0] == CheckValue.FAIL
    assert r.a_response_check(good, None, np.radians(30), SETTINGS)[0] == CheckValue.NOT_MEASURED
    assert r.a_response_check(good[:1], None, big, SETTINGS)[0] == CheckValue.NOT_MEASURED
    assert r.a_response_check(good, None, NAN, SETTINGS)[0] == CheckValue.NOT_APPLICABLE


def test_b1_amplitude_of_a_pendulum_is_one():
    t = np.arange(0.0, 3000.0, 1.0)
    c, amplitude, period = 0.6, 1.0, 300.0
    omega = 2 * np.pi / period
    a = 4.0 + amplitude * omega / c * np.cos(omega * t) + 1e-7 * t  # a slow drift does not matter
    required, measured, ratio = r.b1_amplitude(t, a, period, amplitude, -c, (0.0, 3000.0))
    assert required == pytest.approx(amplitude * omega / c)
    assert ratio == pytest.approx(1.0, abs=1e-6)


def test_circulation_strength_and_omega0():
    assert r.omega0_from_force(-0.5, 2e-4) == pytest.approx(1e-2)
    ratio, wave = r.circulation_strength(np.full(10, -0.04), 1e-2, 2e-4)
    assert ratio == pytest.approx(2.0) and wave == pytest.approx(5e-3)


def test_b5_bound_recovers_a_centred_slow_libration():
    # sigma = sigma_s + nu t over an arc S centred on the stable point; a = a0 + (K/nu)(cos(sigma - s) - cos(S/2))
    c, force, nu, span = 0.6, 3e-9, 2.36 / 30000.0, 30000.0
    t = np.linspace(0.0, span, 3001)
    sigma = -1.18 + nu * t
    a = 2.0 + force / nu * np.cos(sigma)
    bound = r.slow_libration_bound(t, sigma, a, c, (0.0, span))
    assert bound['arc'] == pytest.approx(2.36)
    assert bound['force_max'] == pytest.approx(force, rel=1e-6)
    da = np.ptp(a)
    assert bound['force_max'] / (da / span) == pytest.approx(3.81, abs=0.01)
    assert bound['omega0_max'] == pytest.approx(np.sqrt(c * force), rel=1e-6)
    assert bound['ratio'] == pytest.approx(nu / (2 * np.sqrt(c * force)), rel=1e-6)
    off_centre = r.slow_libration_bound(t, sigma + 0.8, 2.0 + force / nu * np.cos(sigma + 0.8), c, (0.0, span))
    assert off_centre['force_max'] >= force  # the bound is from above


def test_b5_wide_arc_uses_half_the_range():
    t = np.linspace(0.0, 1000.0, 1001)
    bound = r.slow_libration_bound(t, 0.01 * t, 1.0 + 1e-6 * np.sin(t / 50.0), 1.0, (0.0, 1000.0))
    assert bound['arc'] == pytest.approx(10.0)
    assert bound['force_max'] == pytest.approx(0.01 * 2e-6 / 2, rel=1e-3)


def test_slow_libration_check_routes():
    bound = {'centre_rate': 1e-4, 'centre_rate_err': 1e-6, 'ratio': 1.5}
    assert r.slow_libration_check(bound, NAN, NAN, NAN, SETTINGS) == (CheckValue.PASS, 'data', False)
    bound['ratio'] = 0.5
    assert r.slow_libration_check(bound, 4e-5, NAN, NAN, SETTINGS) == (CheckValue.FAIL, 'none', False)  # W3-F3: no theory route
    assert r.slow_libration_check(bound, 6e-5, 1e-6, 4e-6, SETTINGS) == (CheckValue.PASS, 'owner', False)
    assert r.slow_libration_check(bound, 6e-5, 3e-6, 4e-6, SETTINGS) == (CheckValue.FAIL, 'none', False)
    exact = {'centre_rate': 1e-7, 'centre_rate_err': 1e-6, 'ratio': 0.01}
    assert r.slow_libration_check(exact, NAN, NAN, NAN, SETTINGS) == (CheckValue.FAIL, 'none', True)
    assert r.slow_libration_check({'centre_rate': NAN}, NAN, NAN, NAN, SETTINGS)[0] == CheckValue.NOT_MEASURED


def test_analyse_response_on_a_pendulum():
    rec = pendulum_record()
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, SETTINGS, spec=rec.spec)
    omega0 = 2 * np.pi / 300.0
    assert resp.identity == CheckValue.PASS and resp.identity_slope_ratio == pytest.approx(1.0, abs=1e-4)  # smoothed at 0.05 P_cyc
    assert resp.identity_wave_corr == pytest.approx(1.0, abs=1e-3)
    assert resp.a_response == CheckValue.PASS and resp.force_law == 'single' and not resp.a_response_thin
    assert resp.force_amplitude_au_yr == pytest.approx(omega0**2 / rec.c, rel=1e-3)
    assert resp.phi0_rad == pytest.approx(0.0, abs=1e-3)  # c > 0: phi0 = sigma_s + pi = 2 pi
    assert resp.omega0_force_rad_yr == pytest.approx(omega0, rel=1e-3)
    assert resp.omega0_ratio == pytest.approx(1.0, abs=2e-3)
    assert resp.b1_ratio == pytest.approx(1.0, abs=0.03) and resp.a_amplitude == CheckValue.PASS
    cols = r.response_columns(resp)
    assert cols['rsp_force_K_au_yr'] == resp.force_amplitude_au_yr and cols['rsp_force_law'] == 'single'
    assert cols['rsp_b5_P_min_kyr'] == pytest.approx(resp.b5_period_min_yr / 1e3)


def test_analyse_response_aliased_and_too_slow_filter():
    rec = pendulum_record(span=3000.0)
    aliased = make(m.Sampling, verdict=m.StepVerdict.AMBIGUOUS)
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, aliased, SETTINGS)
    assert resp.identity == CheckValue.NOT_APPLICABLE and resp.a_response == CheckValue.NOT_APPLICABLE
    rec.prep.filter_too_slow = True
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, SETTINGS)
    assert resp.identity == CheckValue.NOT_MEASURED and resp.a_response == CheckValue.NOT_MEASURED
    assert resp.a_amplitude == CheckValue.NOT_MEASURED
    assert np.isfinite(resp.force_r2_h1)  # the numbers are still written


def test_analyse_response_on_a_kinematic_swing():
    rec = kinematic_record()
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, SETTINGS, spec=rec.spec)
    assert resp.a_amplitude == CheckValue.FAIL and resp.b1_ratio < SETTINGS.mmr_b1_ratio_min
    assert resp.identity == CheckValue.FAIL
    assert resp.slow_adot_r2 < SETTINGS.mmr_kin_slow_r2_max


def test_analyse_response_fits_the_multiplet_when_planet_varpi_is_known():
    rec = pendulum_record(span=20000.0)
    rec.context.planet_varpi_rate_rad_yr = {'Jupiter': 2e-5}
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, SETTINGS, spec=rec.spec)
    assert resp.multiplet is not None and resp.multiplet.harmonics == (0, 1)
    # the response cutoff 0.1 P_cyc (30 yr here) costs the exact pendulum 1 - R^2 ~ 1e-6 at A = 90 deg
    assert resp.multiplet.r2 == pytest.approx(1.0, abs=1e-5)
    assert resp.multiplet.eps[0] == pytest.approx(0.0, abs=1e-3)  # no member but the angle itself
    assert resp.force_law == 'single'  # no gain over the single law


def test_onset_bands_and_episode_physics():
    # a sinusoid of period P through the band [P/2, 2P]: gain G(P; P/2) - G(P; 2P) = 1/(1 + 1/16) - 1/(1 + 16) for N = 2
    t = np.arange(0.0, 6000.0, 0.5)
    gain = 1 / (1 + 0.5**4) - 1 / (1 + 2.0**4)
    before, inside, after = r.onset_bands(t, 1e-3 * np.sin(2 * np.pi * t / 300.0), 300.0, 2000.0, 4000.0)
    assert (before, inside, after) == pytest.approx([gain * 1e-3 / np.sqrt(2)] * 3, rel=0.02)
    assert np.isnan(r.onset_bands(t, t * 0, 300.0, 100.0, 4000.0)[0])  # no room before the episode
    rec = pendulum_record()
    (ep,) = r.analyse_episodes(rec.prep, rec.geometry, rec.context, None, SETTINGS, spec=rec.spec)
    # T0 = 300 yr, A = 90 deg; omega0_vmax = max|sigma_dot| / (2 sin 45 deg) = omega0
    assert ep.small_period_min_yr == pytest.approx(300.0, rel=2e-3) and ep.small_period_ratio == pytest.approx(1.0, abs=3e-3)
    assert ep.omega0_vmax_rad_yr == pytest.approx(2 * np.pi / 300.0, rel=2e-3)
    assert ep.omega0_vmax_ratio == pytest.approx(np.sin(np.pi / 4), rel=2e-3)
    assert ep.half_width_au == pytest.approx(2 * (2 * np.pi / 300.0) / rec.c, rel=2e-3)
    assert ep.a_excursion_meas_au == pytest.approx(ep.a_excursion_pred_au, rel=2e-3)
    n = 2 * np.pi / 4.0**1.5
    assert ep.mu_eff == pytest.approx(((2 * np.pi / 300.0) / (2 * n)) ** 2 / 3, rel=5e-3)
    assert ep.response.force_r2_h1 == pytest.approx(1.0, abs=1e-5) and ep.response.b1_ratio == pytest.approx(1.0, abs=0.03)


def test_response_cutoff_removes_the_short_period_leak_of_a_dot():
    # a 7-yr wave of 2e-3 au in the osculating a, left at 1.5 % by a 20-yr cutoff, still gives an a_dot wave
    # 3.6 times the resonant force K = omega0^2 / c = 7.4e-6 au/yr (T0 = 3 kyr): R^2 ~ 1 / (1 + 3.6^2) = 0.07
    rec = pendulum_record(omega0=2 * np.pi / 3000.0, span=60000.0, step=1.0)
    leak = 2e-3 * (1 / (1 + (20 / 7.0) ** 4)) * np.sin(2 * np.pi * rec.t / 7.0)
    rec.prep.a_f = rec.a + leak
    rec.prep.adot_f = np.gradient(rec.prep.a_f, rec.t)
    rec.prep.cutoff = m.Cutoff(20.0, 'synodic', 20.0, 20.0, 20.0)
    plain = r.analyse_response(
        rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, settings_from_config(mmr_response_cutoff_cycles=0.0)
    )
    assert plain.force_r2_h1 < SETTINGS.mmr_force_r2_fail
    settings = SETTINGS
    period = rec.geometry.period_cycle_yr
    assert r.response_cutoff_yr(20.0, period, settings) == pytest.approx(settings.mmr_response_cutoff_cycles * period)
    assert r.response_cutoff_yr(20.0, NAN, settings) == 20.0
    smoothed = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, settings)
    assert smoothed.force_r2_h1 > 0.99 and smoothed.a_response == CheckValue.PASS
    assert smoothed.identity == CheckValue.PASS
