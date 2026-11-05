"""A5 (W28, cards g05-3 B, g05-4 A, g10-2): K1 on the decisive episode, the swing shares, kin_check_reason and the
margin K1e that lets a kinematic libration with measured physics reach high.

Truths built here, independently of the code:
- sigma = (uniform lambda part) + p varpi, varpi = g t + B sin(nu t): the episode's detrended sigma is p B sin(nu t), so
  K1e = 1, swing share = 1, lambda share = 0.
- with a lambda part A_l cos(nu t) in quadrature with p B sin(nu t): sigma's half-range is sqrt((pB)^2 + A_l^2), so the
  swing share is |p| B / sqrt(...) and the lambda share A_l / sqrt(...) (1.0 and 0.2 rad: 0.98058 and 0.19612; the
  correlation of sigma with p varpi is the same 0.98058).
- a three-body angle 3M+7J-6+0+0-4 (the 285637 pattern) whose sigma swings only through -4 varpi at the double omega
  cycle (nu = 2 omega_dot) with a flat a: the tree reads non_resonant/kinematic_libration, secular_cycle; F does not
  count as physics there (e^q scale), so without K1e every physics margin is not_measured.
"""

import math

import numpy as np
import pytest

from resonances.resonance.analysis import confidence
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr import kinematics as kin
from resonances.resonance.analysis.mmr import models
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.context import table_mean_motion
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    LibrationOrigin,
    MarginClass,
    MarginValue,
    OriginDetail,
    Reason,
    Status,
)
from tests.resonances.analysis.builders import make, make_geometry, make_kinematics, make_response, tree_and_final

S = settings_from_config()
TWO_PI = 2 * math.pi
P_SWING = 3000.0  # yr, the double omega cycle of the synthetic records
NU = TWO_PI / P_SWING
G_RATE = TWO_PI / 60000.0  # mean varpi rate [rad/yr]


def _series(key, varpi_amplitude, lambda_amplitude=0.0, step=1.0, span=12000.0):
    """MMRSeries of `key` (its planets on uniform longitudes) whose sigma = pi + p B sin(nu t) + A_l cos(nu t): the
    body's lambda carries the uniform lambda part (and A_l), varpi = 0.3 + g t + B sin(nu t), omega turns at nu / 2,
    a is flat (a 1e-9 au ripple at 37 yr keeps its derivative finite)."""
    spec = ResonanceSpec.from_key(key)
    t = np.arange(0.0, span + step / 2, step)
    zeros = np.zeros_like(t)
    elements = {'Mars': (1.5237, 0.0934, 2.0), 'Jupiter': (5.2026, 0.048, 0.4)}
    planets, total_rate, total_lambda = {}, 0.0, zeros.copy()
    for name, k in zip(spec.planets, spec.planet_coefficients):
        a_p, e_p, phase = elements[name]
        n_p = table_mean_motion(name)
        lam_p = n_p * t + phase
        planets[name] = PlanetSeries(name, t, a_p + zeros, e_p + zeros, 0.03 + zeros, 1.0 + zeros, 0.3 + zeros, np.mod(lam_p, TWO_PI))
        total_rate += k * n_p
        total_lambda = total_lambda + k * lam_p
    k = spec.body_coefficient  # negative: sigma = sum k_j lambda_j + k lambda + p varpi
    p = spec.body_varpi_coefficient
    varpi = 0.3 + G_RATE * t + varpi_amplitude * np.sin(NU * t)
    lam = (math.pi - total_lambda - p * (0.3 + G_RATE * t) - lambda_amplitude * np.cos(NU * t)) / k
    n = -(total_rate + p * G_RATE) / k
    a = (TWO_PI / n) ** (2 / 3) + 1e-9 * np.sin(TWO_PI * t / 37.0)
    sigma = total_lambda + k * lam + p * varpi
    Omega = 1.0 + (G_RATE - NU / 2) * t  # omega = varpi - Omega turns at nu / 2: the swing is at pi / |omega_dot|
    series = MMRSeries(
        t_yr=t,
        a=a,
        e=0.12 + zeros,
        inc=0.05 + zeros,
        Omega=np.mod(Omega, TWO_PI),
        omega=np.mod(varpi - Omega, TWO_PI),
        varpi=np.mod(varpi, TWO_PI),
        angle_wrapped=np.mod(sigma, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=step, span_yr=span, n_rows=len(t), planets_saved=tuple(planets)),
        planets=planets,
        mean_longitude=np.mod(lam, TWO_PI),
        name='synthetic-kinematic',
    )
    series.validate()
    return series, np.unwrap(sigma)


def _episode(key, varpi_amplitude, lambda_amplitude, window, cutoff=300.0):
    series, sigma = _series(key, varpi_amplitude, lambda_amplitude, step=5.0)
    prep = make(models.Preprocessed, t_yr=series.t_yr, angle_unwrapped=sigma)
    return kin.episode_kinematics(series, prep, window, cutoff, S)


# --------------------------------------------------------------------------------------------- K1 on the episode
THREE_CYCLES = (1500.0, 1500.0 + 3 * P_SWING)


def test_a_pure_varpi_swing_follows_p_varpi_with_share_one():
    corr, share, lambda_share, reason = _episode('3M+7J-6+0+0-4', 0.25, 0.0, THREE_CYCLES)
    assert reason == ''
    assert corr == pytest.approx(1.0, abs=1e-6)
    assert share == pytest.approx(1.0, abs=1e-3) and lambda_share == pytest.approx(0.0, abs=1e-3)


def _detrended_half_range(t, x):
    residual = x - np.polyval(np.polyfit(t, x, 1), t)
    return (residual.max() - residual.min()) / 2, residual


def test_swing_shares_against_the_generating_amplitudes():
    # p = -4, B = 0.25 rad: p varpi swings 1.0 rad; the lambda part 0.2 rad in quadrature. Closed form over whole cycles
    # without the trend removal: 1 / sqrt(1.04) = 0.98058 and 0.2 / sqrt(1.04) = 0.19612. Over three cycles the line
    # fit also takes a few % of the odd part of each sinusoid, so the exact truth detrends the generating terms alone
    corr, share, lambda_share, reason = _episode('3M+7J-6+0+0-4', 0.25, 0.2, THREE_CYCLES)
    t = np.linspace(*THREE_CYCLES, 60001)
    varpi_half, varpi_d = _detrended_half_range(t, -4 * 0.25 * np.sin(NU * t))
    lambda_half, lambda_d = _detrended_half_range(t, -0.2 * np.cos(NU * t))
    sigma_half, sigma_d = _detrended_half_range(t, -4 * 0.25 * np.sin(NU * t) - 0.2 * np.cos(NU * t))
    assert reason == ''
    assert share == pytest.approx(varpi_half / sigma_half, abs=1e-3)
    assert lambda_share == pytest.approx(lambda_half / sigma_half, abs=1e-3)
    assert corr == pytest.approx(np.corrcoef(sigma_d, varpi_d)[0, 1], abs=1e-3)
    assert (share, lambda_share) == pytest.approx((1 / math.sqrt(1.04), 0.2 / math.sqrt(1.04)), abs=0.03)  # 0.983, 0.170


def test_no_body_varpi_term_gives_nan_with_its_reason():
    # 2J-1-1+0: the planet's varpi only, p = 0
    corr, share, lambda_share, reason = _episode('2J-1-1+0', 0.25, 0.0, THREE_CYCLES)
    assert np.isnan([corr, share, lambda_share]).all() and reason == kin.K1E_NO_VARPI


def test_an_episode_shorter_than_the_independent_points_is_not_measured():
    # 2.5 cutoffs < MMR_KIN_MIN_POINTS = 3
    window = (1500.0, 1500.0 + 2.5 * 300.0)
    corr, share, lambda_share, reason = _episode('3M+7J-6+0+0-4', 0.25, 0.0, window)
    assert np.isnan([corr, share, lambda_share]).all() and reason == kin.K1E_FEW_POINTS


def test_check_reason_names_why_the_slow_correlation_has_no_number():
    assert kin.check_reason(float('nan'), 10.0, 0, S) == kin.K1E_NO_VARPI
    # 285637: 50 kyr against W_c = 54 kyr, 0.9 independent points
    assert kin.check_reason(float('nan'), 0.93, -4, S) == kin.K1E_FEW_POINTS
    assert kin.check_reason(0.95, 5.0, -4, S) == ''
    assert kin.check_reason(float('nan'), 5.0, -4, S) == kin.K1E_NO_SWING


def test_the_columns_carry_the_episode_numbers_and_both_reasons():
    k = make_kinematics(k1_episode=0.97, swing_share=1.13, lambda_swing_share=0.24, check_reason='too_few_points')
    columns = kin.kinematics_columns(k)
    assert columns['kin_k1_episode'] == 0.97 and columns['kin_swing_share'] == 1.13
    assert columns['kin_lambda_swing_share'] == 0.24 and columns['kin_check_reason'] == 'too_few_points'
    assert 'kin_k1_episode_reason' in columns


# --------------------------------------------------------------------------------------------- margin K1e
LIBRATION = decide.AngleDecision(AngleStatus.LIBRATION, AngleSubtype.STATIONARY, 'G2', None)
KINEMATIC = decide.StatusDecision(
    Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, LibrationOrigin.KINEMATIC, str(OriginDetail.SECULAR_CYCLE), '', rule='3'
)


def _k1e(k1=0.971, b1=6.0e-5, cycle_yr=18029.0, two_omega_yr=18106.0, reason=''):
    kinematics = make_kinematics(
        k1_episode=k1, two_omega_period_yr=two_omega_yr, modulation_period_yr=float('nan'), k1_episode_reason=reason
    )
    geometry = make_geometry(period_cycle_yr=cycle_yr)
    return decide.kinematic_episode_margin(LIBRATION, make_response(b1_ratio=b1), kinematics, geometry, Ledger.from_settings(S), S)


def test_k1e_is_full_when_the_three_measurements_hold():
    # 285637: K1e 0.971, B1 6e-5, |18029 / 18106 - 1| = 0.0043
    margin = _k1e()
    assert margin.margin == Margin.FULL and margin.margin_class == MarginClass.SUPPORT
    assert margin.value == pytest.approx(0.971)


@pytest.mark.parametrize(
    'changes',
    [
        {'b1': 0.05},  # a carries 5 % of a pendulum's need: no flat a (41588 has 0.095)
        {'k1': 0.8},  # sigma does not follow p varpi on the episode
        {'cycle_yr': 21800.0},  # 20 % off the double omega cycle
        {'k1': float('nan')},
        {'b1': float('nan')},
    ],
)
def test_k1e_is_not_measured_when_one_part_does_not_hold(changes):
    assert _k1e(**changes).margin == Margin.NOT_MEASURED


def test_k1e_does_not_apply_without_the_body_varpi_or_retrograde():
    assert _k1e(k1=float('nan'), reason=kin.K1E_NO_VARPI).margin == Margin.NOT_APPLICABLE
    assert _k1e(k1=float('nan'), reason=kin.K1E_RETROGRADE).margin == Margin.NOT_APPLICABLE


def test_the_physics_list_gets_k1e_only_for_a_kinematic_origin():
    base = ('L4', 'R0', 'Z', 'W')
    assert decide.physics_margins_of(KINEMATIC, base) == base + ('K1e',)
    dynamic = decide.StatusDecision(Status.RESONANT, Reason.DYNAMIC_LIBRATION, LibrationOrigin.DYNAMIC, '', '', rule='5a')
    assert decide.physics_margins_of(dynamic, base) == base


def _margins(k1e: Margin):
    def mv(margin, margin_id):
        return MarginValue(margin, None, decide.MARGIN_CLASSES[margin_id])

    return {
        'L2': mv(Margin.THIN, 'L2'),
        'W': mv(Margin.NOT_MEASURED, 'W'),
        'K1': mv(Margin.NOT_MEASURED, 'K1'),
        'B5': mv(Margin.NOT_MEASURED, 'B5'),
        'K1e': mv(k1e, 'K1e'),
    }


def test_a_full_k1e_lifts_all_physics_not_measured_and_fires_kinematic_high():
    physics = decide.physics_margins_of(KINEMATIC, ('L4', 'R0', 'Z', 'W'))
    margins = _margins(Margin.FULL)
    level = confidence.assess(
        Status.NON_RESONANT, 'libration', 'kinematic_libration', margins, [], [], physics, confidence.ConfidenceCells()
    )
    assert level.confidence == Confidence.HIGH and str(Cap.ALL_PHYSICS_NOT_MEASURED) not in level.caps
    assert decide.kinematic_high_rules(KINEMATIC, margins, physics) == frozenset({str(Rule.KINEMATIC_HIGH)})


def test_without_k1e_the_cap_stays_at_medium_and_no_rule_fires():
    physics = decide.physics_margins_of(KINEMATIC, ('L4', 'R0', 'Z', 'W'))
    margins = _margins(Margin.NOT_MEASURED)
    level = confidence.assess(
        Status.NON_RESONANT, 'libration', 'kinematic_libration', margins, [], [], physics, confidence.ConfidenceCells()
    )
    assert level.confidence == Confidence.MEDIUM and str(Cap.ALL_PHYSICS_NOT_MEASURED) in level.caps
    assert decide.kinematic_high_rules(KINEMATIC, margins, physics) == frozenset()


def test_kinematic_high_does_not_fire_when_other_physics_was_measured():
    physics = decide.physics_margins_of(KINEMATIC, ('L4', 'R0', 'Z', 'W'))
    margins = {**_margins(Margin.FULL), 'W': MarginValue(Margin.FULL, None, MarginClass.DECISIVE)}
    assert decide.kinematic_high_rules(KINEMATIC, margins, physics) == frozenset()


# --------------------------------------------------------------------------------------------- the whole analysis
def test_a_kinematic_libration_with_a_flat_a_reaches_high():
    series, _ = _series('3M+7J-6+0+0-4', 0.25)
    verdict = analyse_mmr(series, S).verdict
    assert (verdict.status, verdict.reason, verdict.origin_detail) == (
        Status.NON_RESONANT,
        Reason.KINEMATIC_LIBRATION,
        str(OriginDetail.SECULAR_CYCLE),
    )
    assert verdict.margins['K1e'].margin == Margin.FULL and verdict.margins['K1e'].value >= S.mmr_kin_corr_min
    assert str(Cap.ALL_PHYSICS_NOT_MEASURED) not in verdict.caps
    assert verdict.confidence == Confidence.HIGH
    assert str(Rule.KINEMATIC_HIGH) in verdict.rules_fired


def test_the_same_record_off_the_secular_period_stays_medium():
    # a period tolerance below the record's 1 % offset from pi / |omega_dot|: K1e is not measured, the cap holds (the
    # W28 tree's level, W34)
    series, _ = _series('3M+7J-6+0+0-4', 0.25)
    analysis, final = tree_and_final(series, settings_from_config(mmr_kin_period_tol=0.001))
    verdict = analysis.verdict
    assert verdict.reason == Reason.KINEMATIC_LIBRATION and verdict.margins['K1e'].margin == Margin.NOT_MEASURED
    assert str(Cap.ALL_PHYSICS_NOT_MEASURED) in verdict.caps and verdict.confidence == Confidence.MEDIUM
    assert str(Rule.KINEMATIC_HIGH) not in verdict.rules_fired
    assert analysis.kinematics.k1_episode == pytest.approx(1.0, abs=1e-6)
    # W34, the minimal policy: non_resonant by the measured kinematic veto, not thin and no second capture -> high; the
    # cap is a flag
    verdict = final.verdict
    assert (verdict.status, verdict.reason, verdict.veto, verdict.confidence) == (
        Status.NON_RESONANT,
        Reason.KINEMATIC_LIBRATION,
        'kinematic',
        Confidence.HIGH,
    )
    assert str(Cap.ALL_PHYSICS_NOT_MEASURED) in verdict.flags and verdict.caps == []


# --------------------------------------------------------------------------------------------- benchmark rows
def analyse_example(example):
    from resonances.resonance.analysis.mmr import io

    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    return analyse_mmr(io.load_series(example), S)


@pytest.mark.benchmark
def test_285637_is_a_kinematic_libration_at_high():
    # g05-3 B: K1e 0.971, B1 6e-5, offset 0.004; the swing share is 1.13, not ~1: the lambda part carries a slow
    # +-7 deg wave longer than the 18-kyr cycle (lambda share 0.24; with a cubic trend removed the share is 0.99)
    analysis = analyse_example('mmr3__285637__3M+7J-6+0+0-4')
    verdict, k = analysis.verdict, analysis.kinematics
    assert (verdict.status, verdict.reason, verdict.confidence) == (Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, Confidence.HIGH)
    assert verdict.origin_detail == str(OriginDetail.SECULAR_CYCLE)
    # K1e is full; kinematic_high (K1e lifting a row whose other physics is all unmeasured) no longer acts since the
    # owner's stage 3 (2026-11-05) measures the owner: the level is high without the lift
    assert verdict.margins['K1e'].margin == Margin.FULL and str(Rule.KINEMATIC_HIGH) not in verdict.rules_fired
    assert 1.05 <= k.swing_share <= 1.2 and k.k1_episode >= S.mmr_kin_corr_min
    assert k.check_reason == kin.K1E_FEW_POINTS  # K1's slow correlation: 50 kyr against W_c 54 kyr


@pytest.mark.benchmark
def test_481085_is_a_kinematic_libration_at_high():
    verdict = analyse_example('mmr3__481085__7E-10M-6+0+0+9').verdict
    assert (verdict.status, verdict.reason, verdict.confidence) == (Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, Confidence.HIGH)
    assert verdict.origin_detail == str(OriginDetail.SECULAR_CYCLE)


@pytest.mark.benchmark
def test_41588_with_its_owner_measured_on_short_windows_is_high():
    # W33: 3.5-P windows do not fit the record; on 2-P windows the owner search runs and finds none: check pass, the cap
    # all_physics_not_measured goes (high, as under the minimal policy and the author's label of 2026-11-03)
    analysis = analyse_example('mmr3__41588__3J+9S-2+0+0-10')
    assert analysis.verdict.confidence == Confidence.HIGH and analysis.verdict.checks['check_mmr_owner'] == CheckValue.PASS
    assert 'owner_short_windows' in analysis.owner.flags


@pytest.mark.benchmark
def test_41588_stays_medium_its_a_answers_too_much():
    # B1 0.095 > MMR_KIN_B1_CLEAR: K1e is not measured; read without the W33 owner retry on short windows, which measures
    # the owner (none) and so lifts all_physics_not_measured (high, as the minimal policy and the author's label)
    from dataclasses import replace

    from resonances.resonance.analysis.mmr import io

    # the W28 tree's level (W34: the minimal policy's level reads the evidence, not the cap)
    analysis, _ = tree_and_final(io.load_series('mmr3__41588__3J+9S-2+0+0-10'), replace(S, mmr_owner_short_window_cycles=0.0))
    verdict = analysis.verdict
    assert verdict.reason == Reason.KINEMATIC_LIBRATION and verdict.confidence == Confidence.MEDIUM
    assert verdict.margins['K1e'].margin == Margin.NOT_MEASURED
    assert 0.9 <= analysis.kinematics.swing_share <= 1.1
