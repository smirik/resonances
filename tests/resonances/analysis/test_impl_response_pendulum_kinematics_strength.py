"""Kinematic origin (analysis.mmr.kinematics) and strength (analysis.mmr.strength).

Independent truths: 21514 (handbook part 2): e 0.151-0.256 gives e_free 0.2035, e_forced 0.0525 and with g - g5 = 49.24"/yr
a predicted varpi_dot of 43.4-70.6 "/yr (NUMBERS §6.2); mu_eff = (P_orb/T0)^2 / (3 k^2) with T0 = P/F(A) by hand
(NUMBERS §7.1); omega0_th = |k| n sqrt(3 mu) inverts it.
"""

import numpy as np
import pytest

from resonances.resonance.analysis import pendulum as pend
from resonances.resonance.analysis.mmr import kinematics as kin
from resonances.resonance.analysis.mmr import models as m
from resonances.resonance.analysis.mmr import response as r
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue
from tests.resonances.analysis.impl_response_pendulum_synthetic import kinematic_record, make, pendulum_record

SETTINGS = settings_from_config()
ARCSEC = np.pi / 180 / 3600
NAN = float('nan')


def test_varpi_rate_modulation_21514():
    g5 = 4.257 * ARCSEC
    g = g5 + 49.24 * ARCSEC
    e = np.array([0.151, 0.2, 0.256])
    e_free, e_forced, low, high, period = kin.varpi_rate_modulation(e, g, g5)
    assert (e_free, e_forced) == pytest.approx((0.2035, 0.0525))
    assert (low / ARCSEC, high / ARCSEC) == pytest.approx((43.4, 70.6), abs=0.1)
    assert period == pytest.approx(2 * np.pi / (49.24 * ARCSEC))
    # the same extremes from the formula evaluated over theta (independent of the closed-form ends)
    theta = np.linspace(0, 2 * np.pi, 3601)
    rate = g5 + (g - g5) * (e_free**2 + e_free * e_forced * np.cos(theta)) / (
        e_free**2 + e_forced**2 + 2 * e_free * e_forced * np.cos(theta)
    )
    assert (low, high) == pytest.approx((rate.min(), rate.max()), rel=1e-6)


def test_two_omega_cycle():
    t = np.linspace(0.0, 100000.0, 10001)
    omega_dot = 0.175e-3  # 285637: pi / omega_dot = 17954 yr
    omega = np.mod(omega_dot * t, 2 * np.pi)
    period, corr = kin.two_omega_cycle(t, omega, 0.1 + 0.02 * np.cos(2 * omega_dot * t))
    assert period == pytest.approx(17952.0, rel=1e-3) and corr == pytest.approx(1.0)


def test_cancellation():
    t = np.linspace(0.0, 10000.0, 1001)
    psi = 1e-3 * t + 0.01 * np.sin(t / 100.0)
    varpi = -5e-4 * t  # p = 2: p varpi_dot = -1e-3
    rate, rms, p_rate, low, high = kin.cancellation(t, psi, 2, varpi)
    assert rate == pytest.approx(1e-3, rel=1e-3) and rms == pytest.approx(0.01 / np.sqrt(2), rel=0.02)
    assert (p_rate, low, high) == pytest.approx((-1e-3, -1e-3, -1e-3))


def test_kinematic_check_branches():
    assert kin.kinematic_check(0.97, 0.1, 5, 3, SETTINGS) == CheckValue.FAIL
    assert kin.kinematic_check(0.97, 0.6, 5, 3, SETTINGS) == CheckValue.PASS  # a answers: dynamics
    assert kin.kinematic_check(0.5, 0.1, 5, 3, SETTINGS) == CheckValue.PASS
    assert kin.kinematic_check(0.97, 0.1, 2, 3, SETTINGS) == CheckValue.NOT_MEASURED
    assert kin.kinematic_check(0.97, 0.1, 5, 0, SETTINGS) == CheckValue.NOT_APPLICABLE
    assert kin.kinematic_check(NAN, 0.1, 5, 3, SETTINGS) == CheckValue.NOT_MEASURED


def test_lambda_part_removes_the_varpi_terms():
    rec = kinematic_record()
    psi = kin.lambda_part(rec.series, rec.angle)
    expected = -3 * 2e-4 * rec.t + np.pi  # sigma - 3 varpi of the synthetic record
    assert psi - psi[0] == pytest.approx(expected - expected[0], abs=1e-9)


def test_analyse_kinematics_on_a_kinematic_swing():
    rec = kinematic_record()
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, SETTINGS, spec=rec.spec)
    k = kin.analyse_kinematics(rec.series, rec.prep, rec.slow, rec.geometry, rec.context, resp, SETTINGS)
    assert k.check == CheckValue.FAIL
    assert k.corr_rate == pytest.approx(1.0, abs=1e-6)
    assert k.lambda_part_rate_rad_yr == pytest.approx(-3 * 2e-4, rel=1e-6)
    assert k.cancel_ratio == pytest.approx(-1.0, abs=0.05)
    # the angle's period is the varpi_dot modulation period 2 pi / (g - g5): a secular cycle
    assert k.modulation_period_yr == pytest.approx(2 * np.pi / rec.nu, rel=0.02)
    assert k.detail == 'secular_cycle'
    cols = kin.kinematics_columns(k)
    assert cols['kin_lambda_part_rate_rad_kyr'] == pytest.approx(-0.6, rel=1e-6)
    assert cols['kin_origin_detail'] == 'secular_cycle'


def test_analyse_kinematics_of_a_pendulum_and_retrograde():
    rec = pendulum_record(span=20000.0)
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, SETTINGS, spec=rec.spec)
    k = kin.analyse_kinematics(rec.series, rec.prep, rec.slow, rec.geometry, rec.context, resp, SETTINGS)
    assert k.check != CheckValue.FAIL and k.detail == ''
    retro = m.ResonanceSpec('1J-3r0', m.ResonanceKind.RETROGRADE, ('Jupiter',), (-1,), 3, -2, (0,))
    rec.series = m.MMRSeries(**{**rec.series.__dict__, 'resonance': retro})
    k = kin.analyse_kinematics(rec.series, rec.prep, rec.slow, rec.geometry, rec.context, resp, SETTINGS)
    assert k.check == CheckValue.NOT_APPLICABLE and np.isnan(k.corr_rate)


def test_origin_detail_order():
    base = make(m.Kinematics, check=CheckValue.FAIL, detail='', modulation_period_yr=5714.0, two_omega_period_yr=17954.0)
    response = make(m.Response, a_amplitude=CheckValue.PASS)
    assert kin.origin_detail(base, response, SETTINGS, 5750.0) == 'secular_cycle'  # 37087
    assert kin.origin_detail(base, response, SETTINGS, 18153.0) == 'secular_cycle'  # 285637
    assert kin.origin_detail(base, response, SETTINGS, 940.0, 920.0) == 'planet_ripple'
    assert kin.origin_detail(base, response, SETTINGS, 30000.0) == 'varpi_term'
    not_found = make(m.Kinematics, check=CheckValue.PASS, detail='')
    assert kin.origin_detail(not_found, response, SETTINGS, 5750.0) == ''
    b1_fail = make(m.Response, a_amplitude=CheckValue.FAIL)
    assert kin.origin_detail(not_found, b1_fail, SETTINGS) == 'varpi_term'


def test_mu_eff_by_hand():
    # a = 2.5 au: P_orb = 2.5^1.5 = 3.9528 yr; T0 = 10 kyr; k = -2 -> mu = (P_orb / T0)^2 / (3 * 4)
    p_orb = 2.5**1.5
    assert st.mu_eff_required(10000.0, -2, 2 * np.pi / p_orb) == pytest.approx((p_orb / 1e4) ** 2 / 12, rel=1e-12)
    # a = 3.0 au, k = 3, P = 15 kyr at A = 120 deg: T0 = 15000 / 1.3729 = 10925.8 yr
    p_orb = 3.0**1.5
    t0 = 15000.0 / pend.amplitude_factor(np.radians(120.0))
    assert t0 == pytest.approx(10925.8, abs=0.5)
    assert st.mu_eff_required(t0, 3, 2 * np.pi / p_orb) == pytest.approx((p_orb / t0) ** 2 / 27, rel=1e-12)
    assert np.isnan(st.mu_eff_required(NAN, 3, 1.0))


def test_omega0_theory_inverts_mu_eff():
    n = 2 * np.pi / 2.5**1.5
    mu = st.mu_eff_required(8000.0, -3, n)
    assert st.omega0_theory(-3, n, mu) == pytest.approx(2 * np.pi / 8000.0)
    assert st.mu_scale(1e-3, 0.1, 3) == pytest.approx(1e-6)


def test_eq_legal_and_strength_check():
    assert st.eq_legal(0.1, (), SETTINGS)
    assert not st.eq_legal(0.35, (), SETTINGS)
    assert not st.eq_legal(0.1, ('Mars',), SETTINGS)
    assert st.strength_check(True, 5.0, SETTINGS) == CheckValue.PASS
    assert st.strength_check(True, 0.1, SETTINGS) == CheckValue.PASS
    assert st.strength_check(True, 11.0, SETTINGS) == CheckValue.FAIL
    assert st.strength_check(True, 0.05, SETTINGS) == CheckValue.FAIL
    assert st.strength_check(False, 5.0, SETTINGS) == CheckValue.NOT_MEASURED
    assert st.strength_check(True, NAN, SETTINGS) == CheckValue.NOT_MEASURED


def test_analyse_strength():
    rec = pendulum_record()
    episodes = r.analyse_episodes(rec.prep, rec.geometry, rec.context, None, SETTINGS, spec=rec.spec)
    strength = st.analyse_strength(rec.context, episodes, SETTINGS)
    # mu_scale = 1e-3 e^q with e = 0.1, q = 1
    assert strength.eq_legal and strength.mu_scale == pytest.approx(1e-4)
    assert strength.mu_eff_required == pytest.approx(episodes[0].mu_eff)
    assert strength.ratio == pytest.approx(episodes[0].mu_eff / 1e-4)
    n = 2 * np.pi / 4.0**1.5
    assert strength.omega0_theory_rad_yr == pytest.approx(2 * n * np.sqrt(3e-4))
    assert strength.check == CheckValue.PASS and not strength.implausible
    crossing = make(m.Context, **{**rec.context.__dict__, 'crosses': ('Jupiter',), 'mu_scale_no_eq': 1e-7})
    strength = st.analyse_strength(crossing, episodes, SETTINGS)
    assert not strength.eq_legal and strength.check == CheckValue.NOT_MEASURED
    # 593 times mu alone: within the reach of a coefficient f(alpha), not implausible (MMR_STRENGTH_IMPLAUSIBLE_RATIO 1e3)
    assert strength.ratio == pytest.approx(episodes[0].mu_eff / 1e-7) and not strength.implausible
    crossing = make(m.Context, **{**rec.context.__dict__, 'crosses': ('Jupiter',), 'mu_scale_no_eq': 1e-8})
    strength = st.analyse_strength(crossing, episodes, SETTINGS)
    assert strength.ratio == pytest.approx(episodes[0].mu_eff / 1e-8) and strength.implausible
    cols = st.strength_columns(strength)
    assert cols['str_P_lib_theory_kyr'] == pytest.approx(strength.period_theory_yr / 1e3)


def test_episode_energy_columns_from_the_energy_analysis():
    rec = pendulum_record()
    run = rec.geometry.runs[0]
    times = run.tp_times_yr
    cycles = [pend.CycleEnergy(a, b, False, 0.5, pend.action_ratio(0.5), b - a, 1.0) for a, b in zip(times[:-2:2], times[2::2])]
    energy = pend.EnergyAnalysis(
        potential_e=None,
        potential_f=None,
        ve_vf_agree_fraction=NAN,
        well_coherence=NAN,
        cycles=cycles,
        cycle_classes=[pend.CycleClass.DEEP] * len(cycles),
    )
    (ep,) = r.analyse_episodes(rec.prep, rec.geometry, rec.context, energy, SETTINGS)
    assert (ep.h_min, ep.h_med, ep.h_max) == pytest.approx((0.5, 0.5, 0.5))
    assert ep.action_ratio_max == pytest.approx(pend.action_ratio(0.5))
    assert ep.zone_class == 'deep'


def test_short_record_keeps_the_line_fits():
    # W_c longer than half the record: no slow window, but the cancellation is still measured on the analysis window
    rec = kinematic_record()
    rec.slow = m.SlowComponents(**{**rec.slow.__dict__, 'window_yr': 150000.0, 'window': (150000.0, 50000.0)})
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, SETTINGS, spec=rec.spec)
    k = kin.analyse_kinematics(rec.series, rec.prep, rec.slow, rec.geometry, rec.context, resp, SETTINGS)
    assert k.lambda_part_rate_rad_yr == pytest.approx(-3 * 2e-4, rel=1e-6)
    assert k.check == CheckValue.NOT_MEASURED
